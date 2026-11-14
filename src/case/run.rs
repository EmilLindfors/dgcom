//! Running a case: the 2D production path of `examples/froya_real_data.rs`
//! (`Simulation` + SSP-RK(4,3) or SSP-RK3, optionally multirate, +
//! `SWEPhysics2D`) with its wet/dry defaults: the positivity limiter, the
//! `WetDry` scheme and point-implicit bed friction. A nested boundary's
//! relaxation band is a source term after the Coriolis force, as there.

use std::fs::{self, File};

use crate::analysis::{GaugeComparison, TimeSeries};
use crate::waves::{CoupledWaves2D, SeaStateFields, SeaStateFrames};

mod mesh;
mod restart;
mod three_d;
mod waves;

pub use mesh::MeshReport;
use std::io::{BufWriter, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use super::domain::Domain;
use super::schema::{BoundaryVelocity, Friction, SideTag};
use super::{CaseError, CaseFile};
use crate::boundary::{
    CharacteristicOBC, ExternalStateProvider, HarmonicTide, InverseBarometer,
    MultiBoundaryCondition2D, Reflective2D, SWEBoundaryCondition2D, StillWater, TidalConstituent,
};
use crate::equations::ShallowWater2D;
use crate::io::{Restart2D, SnapshotWriter, write_vtk_swe};
use crate::physics::{PhysicsBuilder, SWEPhysics2D, SWEPhysics2DBuilder};
use crate::simulation::{Simulation, SimulationResult};
use crate::solver::{
    SWEPointSample, SWESolution2D, SWEState2D, StandardLimiter2D, WetDryConfig, total_mass_2d,
};
use crate::source::{
    CageDrag2D, ChezyFriction2D, GriddedAtmosphere2D, HorizontalViscosity2D, ManningFriction2D,
};
use crate::tides::{canonical_name, constituent_period};
use crate::time::{Multirate, TimeIntegrator};
use crate::types::ElementIndex;

/// Depth below which the velocity is reported as 0 (m).
const H_VELOCITY: f64 = WetDryConfig::DEFAULT_H_DRY;

/// Where a run is, at each output.
#[derive(Clone, Copy, Debug)]
pub struct Progress {
    /// Model time (s)
    pub time_s: f64,
    /// End of the run (s)
    pub end_s: f64,
    /// Lowest and highest surface elevation (m) over the nodes at least
    /// [`Self::DEPTH`] deep (over the wet nodes where none is)
    pub eta_range: (f64, f64),
    /// Largest depth-averaged speed (m/s) over the same nodes
    pub max_speed: f64,
    /// Largest depth-averaged speed (m/s) over the wet nodes shallower than
    /// [`Self::DEPTH`]
    pub max_film_speed: f64,
    /// Largest significant wave height (m), with `[waves]`
    pub max_hs: Option<f64>,
}

impl Progress {
    /// Depth (m) from which a node counts in [`Self::eta_range`] and
    /// [`Self::max_speed`]. Shallower water is a film on a beach or a drying
    /// flat, whose level is its bed's and says nothing of the tide (1–50 mm
    /// of water on land at +3 to +4 m through an ebb, on Chiloé), and whose
    /// speed is a film's (`notes/case-files.md`, P8.13).
    pub const DEPTH: f64 = 1.0;

    /// The progress at `time_s` of a run to `end_s`, of the nodes'
    /// `(depth, surface elevation, speed)` (m, m, m/s).
    fn of(nodes: impl Iterator<Item = (f64, f64, f64)>, time_s: f64, end_s: f64) -> Self {
        // Lowest and highest surface and largest speed: [deeper, films]
        let mut seen = [(f64::INFINITY, f64::NEG_INFINITY, 0.0_f64); 2];
        for (h, eta, speed) in nodes {
            if h > H_VELOCITY {
                let s = &mut seen[usize::from(h < Self::DEPTH)];
                *s = (s.0.min(eta), s.1.max(eta), s.2.max(speed));
            }
        }
        let [deeper, films] = seen;
        // No node that deep (a flume, a flat): the wet nodes
        let of = if deeper.0 <= deeper.1 { deeper } else { films };
        Self {
            time_s,
            end_s,
            eta_range: (of.0, of.1),
            max_speed: of.2,
            max_film_speed: films.2,
            max_hs: None,
        }
    }
}

/// What a station records at one time: the surface at its `level` point, the
/// depth-mean current and the water depth at its `current` point
/// ([`super::StationProbe`]).
#[derive(Clone, Copy, Debug, PartialEq)]
struct StationSample {
    eta: f64,
    u: f64,
    v: f64,
    depth: f64,
}

impl StationSample {
    /// The surface of `level`, the rest of `current`.
    fn of(level: &SWEPointSample, current: &SWEPointSample) -> Self {
        Self {
            eta: level.eta,
            u: current.u,
            v: current.v,
            depth: current.h,
        }
    }

    /// `self` and `other` mixed linearly, `w` of the way to `other`.
    fn mix(&self, other: &Self, w: f64) -> Self {
        let mix = |x: f64, y: f64| x + w * (y - x);
        Self {
            eta: mix(self.eta, other.eta),
            u: mix(self.u, other.u),
            v: mix(self.v, other.v),
            depth: mix(self.depth, other.depth),
        }
    }
}

/// What a station recorded.
#[derive(Clone, Debug, Default)]
pub struct StationSeries {
    /// The station's name
    pub name: String,
    /// Output times (s)
    pub time_s: Vec<f64>,
    /// Surface elevation (m), at the station's `level` point
    /// ([`super::StationProbe`])
    pub eta_m: Vec<f64>,
    /// Depth-averaged velocity along x (m/s), at its `current` point
    pub u_m_s: Vec<f64>,
    /// Depth-averaged velocity along y (m/s), at its `current` point
    pub v_m_s: Vec<f64>,
    /// Water depth (m), at its `current` point
    pub depth_m: Vec<f64>,
    /// With `[waves]`, the sea state at the station at the same times
    /// (empty without waves): the significant wave height (m), the peak
    /// period (s) and the mean direction the waves come from (degrees
    /// clockwise from north; from the mesh's y axis on an unplaced mesh)
    pub hs_m: Vec<f64>,
    /// The peak period (s), 0 without waves there
    pub tp_s: Vec<f64>,
    /// The mean direction the waves come from (degrees clockwise from north),
    /// NaN without waves there
    pub wave_from_deg: Vec<f64>,
    /// Its surface against its tide gauge after the spin-up, if it has one
    pub gauge: Option<GaugeComparison>,
}

/// The outcome of a run.
#[derive(Clone, Debug)]
pub struct RunReport {
    /// Model time reached (s)
    pub final_time_s: f64,
    /// Time steps taken
    pub n_steps: usize,
    /// Smallest and largest time step (s)
    pub dt_range_s: (f64, f64),
    /// Wall-clock time (s)
    pub wall_time_s: f64,
    /// Water volume at the start and the end (m³)
    pub volume_m3: (f64, f64),
    /// The stations' records
    pub stations: Vec<StationSeries>,
    /// What the run set up and did beyond the domain's notes, line by line
    /// (a 3D run's start and open faces)
    pub notes: Vec<String>,
}

impl CaseFile {
    /// Run the case, writing its output and reporting `progress` at every
    /// output time.
    pub fn run(&self, progress: &mut dyn FnMut(&Progress)) -> Result<RunReport, CaseError> {
        self.run_on(&self.build_domain()?, progress)
    }

    /// Run the case on `domain`, built by [`Self::build_domain`].
    pub fn run_on(
        &self,
        domain: &Domain,
        progress: &mut dyn FnMut(&Progress),
    ) -> Result<RunReport, CaseError> {
        if self.case.three_d.is_some() {
            return self.run_3d(domain, progress);
        }
        let case = &self.case;
        let clock = case.clock()?;
        let t_end = 3600.0 * case.run.duration_hours;
        // The run lands on the finer of the field and station intervals
        let interval = 60.0
            * case
                .output
                .interval_minutes
                .min(case.output.station_interval());

        let opens = self.open_boundaries(domain, clock.as_ref())?;
        let wall = Reflective2D::new();
        let physics = self.physics(domain, boundary_conditions(&wall, &opens));
        let mut notes = Vec::new();
        // A resumed run: the restart's state, time and records
        let resumed = match self.resume_path() {
            Some(path) => {
                let restart = Restart2D::read(&path)
                    .map_err(|e| CaseError::Io(format!("{}: {e}", path.display())))?;
                physics.check_restart(&restart).map_err(|e| {
                    CaseError::Invalid(format!("cannot resume from {}: {e}", path.display()))
                })?;
                self.check_resumed(&path, &restart.driver)?;
                let resumed = restart::Resumed {
                    path,
                    time: restart.time,
                    driver: restart.driver,
                };
                notes.push(resumed.note());
                Some((restart.state, resumed))
            }
            None => None,
        };
        let (mut q, resumed) = match resumed {
            Some((state, resumed)) => (state, Some(resumed)),
            None => (still_water(case, domain), None),
        };
        let t_start = resumed.as_ref().map_or(0.0, |r| r.time);
        if t_start >= t_end - 1e-6 {
            return Err(CaseError::Invalid(format!(
                "the restart is at {:.2} h, at or past the run's end ({} h)",
                t_start / 3600.0,
                case.run.duration_hours
            )));
        }
        // The callbacks fall on the restart's time plus multiples of the
        // interval, and the outputs on multiples of theirs
        let n = t_start / interval;
        if (n - n.round()).abs() > 1e-6 {
            return Err(CaseError::Invalid(format!(
                "the restart is at {t_start} s, not a multiple of this run's {} min output \
                 interval",
                interval / 60.0
            )));
        }
        let fingerprint = physics.domain_fingerprint();
        let mut waves = match (&case.waves, &domain.waves) {
            (Some(spec), Some(w)) => Some(self.coupled_waves(spec, w, &physics, &mut notes)?),
            _ => None,
        };
        // The waves' sea state at the outputs, between the exchanges
        let sea = waves
            .as_ref()
            .map(|_| std::cell::RefCell::new(SeaStateFrames::default()));

        let surface: &[&str] = if waves.is_some() {
            &SeaStateFields::NAMES
        } else {
            &[]
        };
        let mut output = Output::create(
            self,
            domain,
            clock.as_ref(),
            None,
            surface,
            resumed.as_ref(),
        )?;
        let volume = |q: &SWESolution2D| volume(domain, q);
        let volume_start = match &resumed {
            Some(r) => restart::counters(&r.driver)?.1,
            None => volume(&q),
        };

        let stop = Arc::new(AtomicBool::new(false));
        let mut failure = None;
        let mut restarts = restart::RestartSchedule::new(self, t_start);
        // A resumed run's first callback is at the restart's time, which the
        // stopped run wrote out: only the files that start with this run
        // take it
        let mut at_restart = resumed.is_some();
        let callback = |q: &SWESolution2D, t: f64| {
            if failure.is_some() {
                return;
            }
            let mut frames = sea.as_ref().map(|s| s.borrow_mut());
            let fields = frames.as_mut().map_or(Vec::new(), |f| f.at(t));
            if std::mem::take(&mut at_restart) {
                if let Err(e) = output.write_at_restart(t, q, &fields) {
                    failure = Some(e);
                    stop.store(true, Ordering::Relaxed);
                }
                return;
            }
            let mut written = output.write(t, q, false, &fields);
            if let (Ok(_), Some(restarts)) = (&written, restarts.as_mut().filter(|r| r.due(t))) {
                let restart = Restart2D {
                    time: t,
                    domain: fingerprint,
                    state: q.clone(),
                    driver: output.driver_data(self, volume_start),
                };
                if let Err(e) = restarts.write(t, |path| restart.write(path)) {
                    written = Err(e);
                }
            }
            match written {
                Ok(false) => {}
                Ok(true) => progress(&Progress {
                    max_hs: max_hs(&fields),
                    ..summarize(domain, q, t, t_end)
                }),
                Err(e) => {
                    failure = Some(e);
                    stop.store(true, Ordering::Relaxed);
                }
            }
        };
        let numerics = &case.numerics;
        let cfl = numerics.cfl_at(case.mesh.order);
        let stepper = numerics.integrator.standard();
        // The waves against a parent wave model's own H_s at its points
        let points = match (&domain.waves, waves.is_some()) {
            (Some(w), true) => w.parent_points.as_ref().map(|parent| {
                let (points, line) = waves::WavePoints::new(parent, w, wave_east(domain));
                notes.push(line);
                std::cell::RefCell::new(points)
            }),
            _ => None,
        };
        let exchange = match (&case.waves, waves.as_mut(), sea.as_ref()) {
            (Some(spec), Some(waves), Some(sea)) => Some(WaveExchange {
                waves,
                every: 60.0 * spec.exchange_minutes,
                sea,
                points: points.as_ref(),
            }),
            _ => None,
        };
        let (result, clips) = match numerics.local_time_stepping_levels {
            0 => drive(
                Simulation::new(physics, stepper),
                cfl,
                interval,
                numerics.dt_max_s,
                &stop,
                &mut q,
                (t_start, t_end),
                exchange,
                callback,
            ),
            levels => drive(
                Simulation::new(physics, Multirate::with_base(stepper, levels)),
                cfl,
                interval,
                numerics.dt_max_s,
                &stop,
                &mut q,
                (t_start, t_end),
                exchange,
                callback,
            ),
        };
        if let Some(e) = failure {
            return Err(e);
        }
        if !result.success {
            return Err(CaseError::Run(format!(
                "the run stopped at {:.1} h: {}",
                result.final_time / 3600.0,
                result.error.as_deref().unwrap_or("unknown error")
            )));
        }
        // The last output, unless the run ended on an output time
        if output.last_time < result.final_time - 1e-6 * interval {
            let mut frames = sea.as_ref().map(|s| s.borrow_mut());
            let fields = frames
                .as_mut()
                .map_or(Vec::new(), |f| f.at(result.final_time));
            output.write(result.final_time, &q, true, &fields)?;
        }
        if let Some(stats) = &result.local_time_stepping {
            notes.push(stats.describe(domain.mesh.n_elements));
        }
        notes.push(format!("{clips} negative-depth clips"));
        if let Some(waves) = &waves {
            let stats = waves.stats();
            notes.push(format!(
                "waves: {} exchanges, {} wave steps (mean {:.2} s); stepping {:.1} s, exchanges \
                 {:.1} s of the {:.1} s wall time",
                stats.exchanges,
                stats.wave_steps,
                result.final_time / stats.wave_steps.max(1) as f64,
                stats.stepping_time,
                stats.exchange_time,
                result.wall_time
            ));
        }
        if let Some(points) = &points {
            let points = points.borrow();
            points.write(&self.output_dir().join("wave_points.csv"))?;
            notes.extend(points.summary(3600.0 * case.spinup_hours()));
        }
        notes.extend(restarts.map(|r| r.note()));
        let mut stations = output.finish();
        if let Some(clock) = &clock {
            score_gauges(domain, &mut stations, clock, 3600.0 * case.spinup_hours());
            if stations.iter().any(|s| s.gauge.is_some()) {
                write_gauge_scores(&self.output_dir(), &stations)?;
            }
        }
        Ok(RunReport {
            final_time_s: result.final_time,
            n_steps: result.n_steps,
            dt_range_s: (result.dt_min, result.dt_max),
            wall_time_s: result.wall_time,
            volume_m3: (volume_start, volume(&q)),
            stations,
            notes,
        })
    }

    /// The 2D physics of the case on `domain` with boundary conditions `bc`.
    fn physics<BC: SWEBoundaryCondition2D>(&self, domain: &Domain, bc: BC) -> SWEPhysics2D<BC> {
        let p = &self.case.physics;
        let mut builder = self.swe_builder(domain, bc);
        // The farm's cages on the depth mean (a 3D run's take the layers)
        if let Some(drag) = cage_drag(domain) {
            builder = builder.with_cage_drag(drag);
        }
        match p.friction {
            Some(Friction::Manning(n)) => {
                builder = builder.with_implicit_friction(ManningFriction2D::new(p.gravity, n));
            }
            Some(Friction::DragCoefficient(c_d)) => {
                builder = builder.with_implicit_friction(ChezyFriction2D::new(c_d));
            }
            None => {}
        }
        builder.build()
    }

    /// The case's open boundaries, one characteristic condition per tag.
    pub(super) fn open_boundaries(
        &self,
        domain: &Domain,
        clock: Option<&crate::time::ModelClock>,
    ) -> Result<OpenBoundaries, CaseError> {
        let case = &self.case;
        // Open boundaries: one characteristic condition per tag
        let weather_level = domain
            .atmosphere
            .as_ref()
            .filter(|_| case.atmosphere.as_ref().is_some_and(|a| a.pressure));
        case.boundaries
            .iter()
            .map(|b| -> Result<_, CaseError> {
                // The weather's inverse-barometer level, the fraction this
                // boundary adds
                let level = weather_level.map(|a| (a, b.inverse_barometer_fraction()));
                let bc: Box<dyn SWEBoundaryCondition2D> = match (&b.still, &b.tide, &b.atlas) {
                    _ if b.nesting.is_some() => {
                        let parent = domain
                            .parents
                            .iter()
                            .find_map(|(tag, parent)| (*tag == b.tag).then_some(parent))
                            .expect("read with the domain");
                        open_boundary(parent.clone(), b.velocity, level)
                    }
                    (Some(still), _, _) => {
                        open_boundary(StillWater::at(still.eta_m), b.velocity, level)
                    }
                    (None, Some(tide), _) => {
                        let constituents = tide
                            .constituents
                            .iter()
                            .map(|c| {
                                let name = canonical_name(&c.name).expect("validated");
                                let period = constituent_period(name).expect("validated");
                                // η = A cos(ωt − G)
                                TidalConstituent::new(
                                    name,
                                    c.amplitude_m,
                                    period,
                                    -c.phase_deg.to_radians(),
                                )
                            })
                            .collect();
                        let mut harmonic = HarmonicTide::new(constituents)
                            .with_mean_elevation(tide.mean_m)
                            .with_ramp_up(3600.0 * tide.ramp_hours);
                        if let Some(clock) = clock {
                            harmonic = harmonic.with_nodal_corrections(clock);
                        }
                        open_boundary(harmonic, b.velocity, level)
                    }
                    (None, None, Some(spec)) => {
                        let atlas = domain
                            .atlases
                            .iter()
                            .find_map(|(tag, atlas)| (*tag == b.tag).then_some(atlas))
                            .expect("read with the domain");
                        let clock = clock.expect("validated: an atlas needs a start");
                        let projection = domain.projection.as_ref().expect("validated: placed");
                        let mut tides = atlas
                            .boundary_tides(
                                &domain.mesh,
                                &domain.ops,
                                projection,
                                b.tag.boundary_tag(),
                                clock,
                                spec.coverage_m,
                            )
                            .map_err(|e| {
                                CaseError::Invalid(format!("{}: {e}", spec.file.display()))
                            })?
                            .with_ramp_up(3600.0 * spec.ramp_hours);
                        if let Some(ratio) = spec.transport_scaling {
                            tides = tides.with_transport_scaling(ratio);
                        }
                        open_boundary(tides, b.velocity, level)
                    }
                    (None, None, None) => unreachable!("validated"),
                };
                Ok((b.tag, bc))
            })
            .collect()
    }

    /// The 2D module of the case on `domain` with boundary conditions `bc`,
    /// without its bed friction: the 2D run adds it, and a 3D run's bottom
    /// drag replaces it.
    pub(super) fn swe_builder<BC: SWEBoundaryCondition2D>(
        &self,
        domain: &Domain,
        bc: BC,
    ) -> SWEPhysics2DBuilder<BC> {
        let p = &self.case.physics;
        let mut builder = PhysicsBuilder::swe_2d(
            domain.mesh.clone(),
            domain.ops.clone(),
            domain.geom.clone(),
            ShallowWater2D::new(p.gravity),
            bc,
        )
        .with_bathymetry(domain.bathymetry.clone());
        if p.wetting_drying {
            builder = builder
                .with_limiter(StandardLimiter2D::Positivity(WetDryConfig::DEFAULT_H_DRY))
                .with_wet_dry(WetDryConfig::default());
        }
        if let Some(coriolis) = self.case.coriolis_source() {
            builder = builder.with_source(coriolis);
        }
        // The nesting's relaxation bands
        for (tag, parent) in &domain.parents {
            let spec = self
                .case
                .boundaries
                .iter()
                .find(|b| b.tag == *tag)
                .and_then(|b| b.nesting.as_ref())
                .expect("a parent per nested boundary");
            if spec.band_km > 0.0 {
                builder = builder.with_source(parent.relaxation(60.0 * spec.band_minutes));
            }
        }
        // The weather: wind stress and pressure gradient
        if let Some(atmosphere) = &domain.atmosphere {
            builder = builder.with_source(atmosphere.clone());
        }
        if p.viscosity_m2_s > 0.0 || p.smagorinsky > 0.0 {
            builder = builder.with_viscosity(
                HorizontalViscosity2D::smagorinsky(p.smagorinsky).with_background(p.viscosity_m2_s),
            );
        }
        builder
    }
}

/// Run `sim` over `(t_start, t_end)`, calling `callback` every `interval`
/// (and at the start and the end).
#[allow(clippy::too_many_arguments)]
fn drive<BC, I>(
    sim: Simulation<SWESolution2D, SWEPhysics2D<BC>, I>,
    cfl: f64,
    interval: f64,
    dt_max: Option<f64>,
    stop: &Arc<AtomicBool>,
    q: &mut SWESolution2D,
    (t_start, t_end): (f64, f64),
    exchange: Option<WaveExchange<'_, '_>>,
    callback: impl FnMut(&SWESolution2D, f64),
) -> (SimulationResult, usize)
where
    BC: SWEBoundaryCondition2D,
    I: TimeIntegrator<SWESolution2D>,
{
    let mut sim = sim
        .with_cfl(cfl)
        .with_callback_interval(interval)
        .with_stop(stop.clone());
    if let Some(dt) = dt_max {
        sim = sim.with_dt_max(dt);
    }
    let result = match exchange {
        None => sim.run_with_callback(q, t_start, t_end, callback),
        // The waves exchange with the run every interval, their sea state
        // kept at the interval's ends
        Some(WaveExchange {
            waves,
            every,
            sea,
            points,
        }) => sim.run_with_exchange(
            q,
            t_start,
            t_end,
            every,
            |physics, q, t, t_next| {
                let record = |waves: &CoupledWaves2D, t: f64| {
                    sea.borrow_mut().push(waves, t);
                    if let Some(points) = points {
                        points.borrow_mut().record(waves, t);
                    }
                };
                if sea.borrow().is_empty() {
                    record(waves, t);
                }
                waves.exchange(physics, q, t, t_next);
                record(waves, t_next);
            },
            callback,
        ),
    };
    (result, sim.physics().negative_depth_clips())
}

/// The coupled waves, their exchange interval (s), their sea states and the
/// parent's points they are compared at.
struct WaveExchange<'a, 'p> {
    waves: &'a mut CoupledWaves2D,
    every: f64,
    sea: &'a std::cell::RefCell<SeaStateFrames>,
    points: Option<&'a std::cell::RefCell<waves::WavePoints<'p>>>,
}

/// Geographic east in `domain`'s mesh axes (rad from the x axis): at the
/// projection's centre, the x axis on an unplaced mesh.
fn wave_east(domain: &Domain) -> f64 {
    domain.projection.as_ref().map_or(0.0, |p| {
        let (ex, ey) = crate::io::east_axis(p, p.ref_lat(), p.ref_lon());
        ey.atan2(ex)
    })
}

/// The largest H_s of a frame's sea-state fields (the first,
/// `SeaStateFields::NAMES`), if it has them.
fn max_hs(fields: &[&[f32]]) -> Option<f64> {
    fields
        .first()
        .map(|hs| hs.iter().fold(0.0_f64, |m, &h| m.max(f64::from(h))))
}

/// The open boundaries of a case, one characteristic condition per tag.
pub(super) type OpenBoundaries = Vec<(SideTag, Box<dyn SWEBoundaryCondition2D>)>;

/// The farm's cages' drag on `domain`'s mesh, if any drag: none without a
/// farm or with `drag_share = 0`.
pub(super) fn cage_drag(domain: &Domain) -> Option<CageDrag2D> {
    if !domain.cages.iter().any(|c| c.drag_per_length > 0.0) {
        return None;
    }
    let drag = CageDrag2D::new(&domain.mesh, &domain.ops, &domain.cages);
    (!drag.is_empty()).then_some(drag)
}

/// The walls and the open boundaries `opens`, by tag.
pub(super) fn boundary_conditions<'a>(
    wall: &'a Reflective2D,
    opens: &'a [(SideTag, Box<dyn SWEBoundaryCondition2D>)],
) -> MultiBoundaryCondition2D<'a> {
    let mut bc = MultiBoundaryCondition2D::new(wall);
    for (tag, open) in opens {
        bc = match tag {
            SideTag::Open => bc.with_open(open.as_ref()),
            SideTag::Tidal => bc.with_tidal(open.as_ref()),
            SideTag::River => bc.with_river(open.as_ref()),
            SideTag::Wall => unreachable!("validated"),
        };
    }
    bc
}

/// Each station with a gauge against it, from `spinup` (s) on, on the
/// gauge's Unix time axis.
fn score_gauges(
    domain: &Domain,
    stations: &mut [StationSeries],
    clock: &crate::time::ModelClock,
    spinup: f64,
) {
    for (series, station) in stations.iter_mut().zip(&domain.stations) {
        let Some(gauge) = &station.gauge else {
            continue;
        };
        let first = series.time_s.partition_point(|&t| t < spinup - 1e-6);
        let times: Vec<f64> = series.time_s[first..]
            .iter()
            .map(|&t| clock.unix(t))
            .collect();
        let model = TimeSeries::new(&times, &series.eta_m[first..]);
        series.gauge = Some(GaugeComparison::compute(&model, gauge));
    }
}

/// `gauge_skill.csv` (each gauged station against the observations and
/// the gauge's tidal prediction) and `gauge_constituents.csv` (its reference
/// constants against the gauge's) in `dir`.
fn write_gauge_scores(dir: &std::path::Path, stations: &[StationSeries]) -> Result<(), CaseError> {
    use std::fmt::Write as _;
    let mut skill = String::from(
        "station,reference,samples,rmse_m,centred_rmse_m,bias_m,correlation,std_model_m,std_reference_m\n",
    );
    let mut constants = String::from(
        "station,constituent,model_amplitude_m,model_lag_deg,inferred,gauge_amplitude_m,gauge_lag_deg,amplitude_ratio,lag_difference_deg,abs_dz_m\n",
    );
    for s in stations {
        let Some(g) = &s.gauge else { continue };
        for (reference, score) in [
            ("observations", &g.vs_observations),
            ("tidal_prediction", &g.vs_prediction),
        ] {
            if let Ok(v) = score {
                let _ = writeln!(
                    skill,
                    "{},{reference},{},{},{},{},{},{},{}",
                    s.name, v.n, v.rmse, v.centred_rmse, v.bias, v.correlation, v.std.0, v.std.1
                );
            }
        }
        let Ok(model) = &g.model_fit else { continue };
        for c in &g.constituents {
            let inferred = model.get(c.name).is_some_and(|m| m.inferred);
            let _ = writeln!(
                constants,
                "{},{},{},{},{inferred},{},{},{},{},{}",
                s.name,
                c.name,
                c.model_amplitude,
                c.model_phase.to_degrees(),
                c.obs_amplitude,
                c.obs_phase.to_degrees(),
                c.amplitude_ratio,
                c.phase_error_degrees(),
                c.complex_difference()
            );
        }
    }
    for (name, text) in [
        ("gauge_skill.csv", skill),
        ("gauge_constituents.csv", constants),
    ] {
        let path = dir.join(name);
        fs::write(&path, text).map_err(|e| CaseError::Io(format!("{}: {e}", path.display())))?;
    }
    Ok(())
}

/// A characteristic open boundary on `provider`'s external state, with
/// `velocity` for level-only data, raised by the given fraction of the
/// weather's inverse-barometer level (`InverseBarometer`).
fn open_boundary<P: ExternalStateProvider + 'static>(
    provider: P,
    velocity: BoundaryVelocity,
    level: Option<(&GriddedAtmosphere2D, f64)>,
) -> Box<dyn SWEBoundaryCondition2D> {
    match level {
        Some((atmosphere, fraction)) if fraction > 0.0 => {
            let raised = InverseBarometer::new(provider, atmosphere.clone()).with_factor(fraction);
            Box::new(with_velocity(CharacteristicOBC::new(raised), velocity))
        }
        _ => Box::new(with_velocity(CharacteristicOBC::new(provider), velocity)),
    }
}

/// `obc` with the external velocity `velocity` for level-only data.
fn with_velocity<P: ExternalStateProvider>(
    obc: CharacteristicOBC<P>,
    velocity: BoundaryVelocity,
) -> CharacteristicOBC<P> {
    match velocity {
        BoundaryVelocity::AtRest => obc,
        BoundaryVelocity::IncomingWave => obc.with_incoming_wave(),
    }
}

/// Water volume ∫ h dA (m³).
fn volume(domain: &Domain, q: &SWESolution2D) -> f64 {
    total_mass_2d(q, &domain.ops, &domain.geom)
}

fn summarize(domain: &Domain, q: &SWESolution2D, t: f64, t_end: f64) -> Progress {
    let nn = domain.ops.n_nodes;
    let nodes = ElementIndex::iter(domain.mesh.n_elements)
        .flat_map(|k| (0..nn).map(move |i| (k, i)))
        .map(|(k, i)| {
            let s = q.get_state(k, i);
            let eta = s.h + domain.bathymetry.get(k, i);
            // A dry node's speed is not read
            (s.h, eta, s.hu.hypot(s.hv) / s.h.max(H_VELOCITY))
        });
    Progress::of(nodes, t, t_end)
}

/// A case snapshot's header metadata: the run and the build that made it,
/// and what the viewer draws over the fields: the stations where they are
/// sampled, the rivers where their water enters, the mesh's projection (for
/// the land around it) and the farm's cages, in `froya_real_data`'s and
/// `farm_3d`'s keys.
fn snapshot_metadata(name: &str, domain: &Domain) -> Vec<(&'static str, String)> {
    let mut metadata = vec![
        ("title", name.to_string()),
        ("source", "dgcom".to_string()),
        ("case", name.to_string()),
        ("build", crate::provenance::describe()),
    ];
    for s in &domain.stations {
        let [x, y] = s.level.position();
        metadata.push(("station", format!("{},{x:.1},{y:.1}", s.name)));
    }
    // Where each river's water enters, the centre of its element, the largest
    // river first (the viewer's number keys reach the first nine places)
    if let Some(rivers) = &domain.rivers {
        let mut order: Vec<usize> = (0..rivers.len()).collect();
        order.sort_by(|&a, &b| {
            rivers
                .discharge(b, 0.0)
                .total_cmp(&rivers.discharge(a, 0.0))
        });
        for i in order {
            let corners = domain.mesh.element_vertices(rivers.element(i));
            let x = corners.iter().map(|c| c[0]).sum::<f64>() / 4.0;
            let y = corners.iter().map(|c| c[1]).sum::<f64>() / 4.0;
            metadata.push((
                "river",
                format!("river {},{x:.1},{y:.1}", rivers.river(i).name),
            ));
        }
    }
    if let Some(projection) = &domain.projection {
        metadata.push((
            "projection",
            format!("local,{},{}", projection.ref_lat(), projection.ref_lon()),
        ));
    }
    for cage in &domain.cages {
        if let crate::source::CageFootprint::Circle {
            center: [x, y],
            radius,
        } = cage.footprint
        {
            metadata.push((
                "cage",
                format!(
                    "{x},{y},{radius},{},{}",
                    cage.net_depth, cage.drag_per_length
                ),
            ));
        }
    }
    metadata
}

/// `station_points.csv`: where each station samples the model, its surface
/// (`level`) and its current, depth and sea state (`current`): the point,
/// its distance from the station's position and the depth of the bed there
/// below mean sea level (`domain`'s `StationProbe`s). A station's current is
/// comparable between runs whose `current` rows agree.
fn write_station_points(path: &std::path::Path, domain: &Domain) -> Result<(), CaseError> {
    use crate::io::CoordinateProjection;
    let mut text = String::from("station,sample,x_m,y_m,lon,lat,offset_m,bed_depth_m\n");
    for s in &domain.stations {
        for (sample, probe, offset) in [
            ("level", &s.level, s.level_offset_m),
            ("current", &s.current, s.current_offset_m),
        ] {
            let [x, y] = probe.position();
            let (lat, lon) = domain
                .projection
                .as_ref()
                .map_or((f64::NAN, f64::NAN), |p| p.xy_to_geo(x, y));
            let depth = -probe.evaluate(domain.bathymetry.element(probe.element()));
            text += &format!(
                "{},{sample},{x:.2},{y:.2},{lon:.6},{lat:.6},{offset:.2},{depth:.3}\n",
                s.name
            );
        }
    }
    fs::write(path, text).map_err(|e| CaseError::Io(format!("{}: {e}", path.display())))
}

/// The sea state at `probe` from the `surface` fields
/// (`SeaStateFields::NAMES`: H_s, T_p, the direction's x and y, …): H_s, T_p
/// and the direction the waves come from (degrees clockwise from north, with
/// geographic east at `east` rad from the mesh's x axis; NaN without waves).
fn sea_state_at(probe: &crate::solver::Probe2D, surface: &[&[f32]], east: f64) -> [f64; 3] {
    let nn = probe.weights().len();
    let start = probe.element().as_usize() * nn;
    let at = |field: &[f32]| -> f64 {
        probe
            .weights()
            .iter()
            .zip(&field[start..start + nn])
            .map(|(w, &v)| w * f64::from(v))
            .sum()
    };
    let (hs, tp) = (at(surface[0]), at(surface[1]));
    let (dx, dy) = (at(surface[2]), at(surface[3]));
    // Travelling to `to` (rad from the mesh's x axis) is coming from the
    // bearing 270° − (to − east)
    let from = if dx.hypot(dy) > 1e-12 {
        (270.0 - (dy.atan2(dx) - east).to_degrees()).rem_euclid(360.0)
    } else {
        f64::NAN
    };
    [hs, tp, from]
}

/// Still water at the case's initial level on `domain`.
fn still_water(case: &super::Case, domain: &Domain) -> SWESolution2D {
    let eta0 = case.initial.eta_m;
    let mut q = SWESolution2D::new(domain.mesh.n_elements, domain.ops.n_nodes);
    for k in ElementIndex::iter(domain.mesh.n_elements) {
        for i in 0..domain.ops.n_nodes {
            let h = (eta0 - domain.bathymetry.get(k, i)).max(0.0);
            q.set_state(k, i, SWEState2D::new(h, 0.0, 0.0));
        }
    }
    q
}

/// The run's writers.
struct Output<'a> {
    domain: &'a Domain,
    name: String,
    dir: Option<std::path::PathBuf>,
    snapshot: Option<SnapshotWriter>,
    /// A resumed run's snapshot file is the stopped run's, continued
    snapshot_continued: bool,
    vtk_frames: Option<usize>,
    #[cfg(feature = "netcdf")]
    netcdf: Option<crate::io::NetCDFWriter>,
    stations: Vec<StationSeries>,
    csv: Option<BufWriter<File>>,
    /// With waves: geographic east in the mesh axes (rad from the x axis),
    /// for the stations' wave directions
    wave_east: Option<f64>,
    last_time: f64,
    /// Time between field outputs and between station records (s)
    field_every: f64,
    station_every: f64,
}

impl<'a> Output<'a> {
    /// The writers of `file`'s run on `domain`; with `sigma`, of a 3D run
    /// (its snapshots 3D); `resumed`, continuing the outputs of the run that
    /// wrote that restart.
    fn create(
        file: &CaseFile,
        domain: &'a Domain,
        clock: Option<&crate::time::ModelClock>,
        sigma: Option<&crate::vertical::SigmaGrid>,
        surface: &[&str],
        resumed: Option<&restart::Resumed>,
    ) -> Result<Self, CaseError> {
        let case = &file.case;
        let io = |what: &str, e: &dyn std::fmt::Display| CaseError::Io(format!("{what}: {e}"));
        let mut stations: Vec<StationSeries> = domain
            .stations
            .iter()
            .map(|s| StationSeries {
                name: s.name.clone(),
                ..Default::default()
            })
            .collect();
        let vtk_done = match resumed {
            Some(r) => {
                restart::restore_stations(&mut stations, &r.driver)?;
                restart::counters(&r.driver)?.0
            }
            None => 0,
        };
        let dir = file.output_dir();
        fs::create_dir_all(&dir).map_err(|e| io(&dir.display().to_string(), &e))?;
        let resolved = dir.join("case.resolved.toml");
        let header = format!("# {}\n", crate::provenance::describe());
        fs::write(&resolved, header + &case.resolved().to_toml())
            .map_err(|e| io(&resolved.display().to_string(), &e))?;
        let name = case.run.name.clone();
        let mut snapshot_continued = false;
        let snapshot = if case.output.snapshot {
            let path = dir.join(format!("{name}.dgsnap"));
            let metadata = snapshot_metadata(&name, domain);
            let metadata: Vec<(&str, &str)> =
                metadata.iter().map(|(k, v)| (*k, v.as_str())).collect();
            // A resumed run continues the file, without the frames the
            // stopped run wrote after its restart
            let continued = resumed.filter(|_| path.exists());
            snapshot_continued = continued.is_some();
            let writer = match (continued, sigma) {
                (Some(r), _) => SnapshotWriter::append(&path, r.time, H_VELOCITY),
                (None, Some(sigma)) => SnapshotWriter::create_3d(
                    &path,
                    &domain.mesh,
                    &domain.ops,
                    &domain.bathymetry.data,
                    sigma,
                    clock,
                    &metadata,
                ),
                (None, None) => SnapshotWriter::create_with(
                    &path,
                    &domain.mesh,
                    &domain.ops,
                    &domain.bathymetry.data,
                    clock,
                    &metadata,
                    H_VELOCITY,
                    crate::io::SnapshotLayout {
                        surface,
                        levels: None,
                    },
                ),
            };
            Some(writer.map_err(|e| io(&path.display().to_string(), &e))?)
        } else {
            None
        };
        let vtk_frames = if case.output.vtk {
            let vtk = dir.join("vtk");
            fs::create_dir_all(&vtk).map_err(|e| io(&vtk.display().to_string(), &e))?;
            Some(vtk_done)
        } else {
            None
        };
        #[cfg(feature = "netcdf")]
        let netcdf = if case.output.netcdf {
            use crate::io::{
                CoordinateProjection, NetCDFMeshInfo, NetCDFWriter, NetCDFWriterConfig,
            };
            // A resumed run's in a file of its own
            let path = dir.join(match resumed {
                Some(r) => format!("{name}_from_{:.2}h.nc", r.time / 3600.0),
                None => format!("{name}.nc"),
            });
            let positions = domain.node_positions();
            let (x, y): (Vec<f64>, Vec<f64>) = positions.iter().map(|&[x, y]| (x, y)).unzip();
            let mut mesh_info = NetCDFMeshInfo::from_xy(x, y);
            if let Some(projection) = &domain.projection {
                let (lat, lon) = positions
                    .iter()
                    .map(|&[x, y]| projection.xy_to_geo(x, y))
                    .unzip();
                mesh_info = mesh_info.with_latlon(lat, lon);
            }
            let mut config = NetCDFWriterConfig::new(path.display().to_string())
                .with_title(name.clone())
                .with_source(crate::provenance::describe())
                .with_velocity(true);
            if let Some(clock) = clock {
                config = config.with_clock(*clock);
            }
            Some(
                NetCDFWriter::create(config, &mesh_info)
                    .map_err(|e| io(&path.display().to_string(), &e))?,
            )
        } else {
            None
        };
        let csv = if stations.is_empty() {
            None
        } else {
            write_station_points(&dir.join("station_points.csv"), domain)?;
            let path = dir.join("stations.csv");
            let mut csv = BufWriter::new(
                File::create(&path).map_err(|e| io(&path.display().to_string(), &e))?,
            );
            let waves = if surface.is_empty() {
                ""
            } else {
                ",hs_m,tp_s,wave_from_deg"
            };
            writeln!(csv, "station,time_s,eta_m,u_m_s,v_m_s,depth_m{waves}")
                .map_err(|e| io("stations.csv", &e))?;
            // A resumed run's records so far, as they were written
            let n = stations.first().map_or(0, |s| s.time_s.len());
            for i in 0..n {
                for s in &stations {
                    let sea = [&s.hs_m, &s.tp_s, &s.wave_from_deg].map(|x| x.get(i));
                    let mut row = format!(
                        "{},{},{},{},{},{}",
                        s.name, s.time_s[i], s.eta_m[i], s.u_m_s[i], s.v_m_s[i], s.depth_m[i]
                    );
                    if let [Some(hs), Some(tp), Some(from)] = sea {
                        row += &format!(",{hs},{tp},{from}");
                    }
                    writeln!(csv, "{row}").map_err(|e| io("stations.csv", &e))?;
                }
            }
            Some(csv)
        };
        let wave_east = (!surface.is_empty()).then(|| wave_east(domain));
        Ok(Self {
            domain,
            name,
            dir: Some(dir),
            snapshot,
            snapshot_continued,
            vtk_frames,
            #[cfg(feature = "netcdf")]
            netcdf,
            stations,
            csv,
            wave_east,
            last_time: resumed.map_or(f64::NEG_INFINITY, |r| r.time),
            field_every: 60.0 * case.output.interval_minutes,
            station_every: 60.0 * case.output.station_interval(),
        })
    }

    /// Record the stations at `t`, with the sea state of the `surface`
    /// fields (`SeaStateFields::NAMES`) if the run has waves.
    fn record_stations(
        &mut self,
        t: f64,
        q: &SWESolution2D,
        surface: &[&[f32]],
    ) -> Result<(), CaseError> {
        let domain = self.domain;
        let bed = Some(&*domain.bathymetry);
        let samples: Vec<StationSample> = domain
            .stations
            .iter()
            .map(|s| {
                StationSample::of(
                    &s.level.sample_swe(q, bed, H_VELOCITY),
                    &s.current.sample_swe(q, bed, H_VELOCITY),
                )
            })
            .collect();
        let sea: Option<Vec<[f64; 3]>> =
            self.wave_east.filter(|_| surface.len() >= 4).map(|east| {
                domain
                    .stations
                    .iter()
                    .map(|s| sea_state_at(&s.current, surface, east))
                    .collect()
            });
        self.record_samples(t, &samples, sea.as_deref())
    }

    /// Record the stations' `samples` (one per station) at `t`, with their
    /// sea states (H_s, T_p, from) if the run has waves.
    fn record_samples(
        &mut self,
        t: f64,
        samples: &[StationSample],
        sea: Option<&[[f64; 3]]>,
    ) -> Result<(), CaseError> {
        let io = |what: &str, e: &dyn std::fmt::Display| CaseError::Io(format!("{what}: {e}"));
        for (j, (s, series)) in samples.iter().zip(&mut self.stations).enumerate() {
            series.time_s.push(t);
            series.eta_m.push(s.eta);
            series.u_m_s.push(s.u);
            series.v_m_s.push(s.v);
            series.depth_m.push(s.depth);
            let waves = match sea.map(|sea| sea[j]) {
                Some([hs, tp, from]) => {
                    series.hs_m.push(hs);
                    series.tp_s.push(tp);
                    series.wave_from_deg.push(from);
                    format!(",{hs},{tp},{from}")
                }
                None => String::new(),
            };
            if let Some(csv) = &mut self.csv {
                writeln!(
                    csv,
                    "{},{t},{},{},{},{}{waves}",
                    series.name, s.eta, s.u, s.v, s.depth
                )
                .map_err(|e| io("stations.csv", &e))?;
            }
        }
        if let Some(csv) = &mut self.csv {
            csv.flush().map_err(|e| io("stations.csv", &e))?;
        }
        Ok(())
    }

    /// Record the stations at `t` if it is a station time, and the fields if
    /// it is a field time (both at the `last` output); whether the fields
    /// were written.
    fn write(
        &mut self,
        t: f64,
        q: &SWESolution2D,
        last: bool,
        surface: &[&[f32]],
    ) -> Result<bool, CaseError> {
        let on = |every: f64| {
            let n = t / every;
            last || (n - n.round()).abs() < 1e-6
        };
        self.last_time = t;
        if on(self.station_every) {
            self.record_stations(t, q, surface)?;
        }
        if !on(self.field_every) {
            return Ok(false);
        }
        self.write_fields(t, q, surface, true)?;
        Ok(true)
    }

    /// A resumed run's fields at the restart's time `t`, if it is a field
    /// time, into the files that start there: the NetCDF file and a new
    /// snapshot file (the stopped run's files have them).
    fn write_at_restart(
        &mut self,
        t: f64,
        q: &SWESolution2D,
        surface: &[&[f32]],
    ) -> Result<(), CaseError> {
        let n = t / self.field_every;
        if (n - n.round()).abs() < 1e-6 {
            self.write_fields(t, q, surface, false)?;
        }
        Ok(())
    }

    /// The fields at `t`: to every writer, or (`all` false, at a resumed
    /// run's start) only to the files that start with the resumed run.
    fn write_fields(
        &mut self,
        t: f64,
        q: &SWESolution2D,
        surface: &[&[f32]],
        all: bool,
    ) -> Result<(), CaseError> {
        let io = |what: &str, e: &dyn std::fmt::Display| CaseError::Io(format!("{what}: {e}"));
        let domain = self.domain;
        if let Some(snapshot) = self
            .snapshot
            .as_mut()
            .filter(|_| all || !self.snapshot_continued)
        {
            snapshot
                .write_state(t, q, surface)
                .map_err(|e| io("snapshot", &e))?;
        }
        if let (Some(frame), Some(dir), true) = (&mut self.vtk_frames, &self.dir, all) {
            let path = dir
                .join("vtk")
                .join(format!("{}_{:04}.vtu", self.name, frame));
            write_vtk_swe(
                &path,
                &domain.mesh,
                &domain.ops,
                q,
                Some(&domain.bathymetry),
                t,
                H_VELOCITY,
            )
            .map_err(|e| io(&path.display().to_string(), &e))?;
            *frame += 1;
        }
        #[cfg(feature = "netcdf")]
        if let Some(netcdf) = &mut self.netcdf {
            let n = q.data[0].len();
            let (mut h, mut eta, mut u, mut v) = (
                Vec::with_capacity(n),
                Vec::with_capacity(n),
                Vec::with_capacity(n),
                Vec::with_capacity(n),
            );
            for k in ElementIndex::iter(domain.mesh.n_elements) {
                for i in 0..domain.ops.n_nodes {
                    let s = q.get_state(k, i);
                    let wet = s.h > H_VELOCITY;
                    h.push(s.h);
                    eta.push(s.h + domain.bathymetry.get(k, i));
                    u.push(if wet { s.hu / s.h } else { 0.0 });
                    v.push(if wet { s.hv / s.h } else { 0.0 });
                }
            }
            netcdf
                .write_timestep(t, &h, &eta, Some(&u), Some(&v))
                .map_err(|e| io("NetCDF", &e))?;
        }
        Ok(())
    }

    /// A 3D run's snapshot at `t`, if snapshots are written.
    fn write_snapshot_3d(
        &mut self,
        t: f64,
        state: &crate::solver::state::Solution3D,
    ) -> Result<(), CaseError> {
        self.last_time = t;
        if let Some(snapshot) = &mut self.snapshot {
            snapshot
                .write_solution_3d(t, state)
                .map_err(|e| CaseError::Io(format!("snapshot: {e}")))?;
        }
        Ok(())
    }

    /// A restart's driver data: the case, the stations' records so far and
    /// the counters, with `volume_start` the run's volume at time 0.
    fn driver_data(&self, file: &CaseFile, volume_start: f64) -> crate::io::DriverData {
        restart::driver_data(
            file,
            &self.stations,
            self.vtk_frames.unwrap_or(0),
            volume_start,
        )
    }

    /// The stations' records; the writers flush every frame as it is written.
    fn finish(self) -> Vec<StationSeries> {
        self.stations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The progress line reads the water at least `Progress::DEPTH` deep: a
    /// film on a beach sets neither its range nor its largest speed, and is
    /// reported beside it. Where no node is that deep, the wet nodes.
    #[test]
    fn progress_is_of_the_deeper_water_and_names_the_films() {
        // (depth, surface, speed): the tide, a film high on a beach, a dry node
        let nodes = [
            (40.0, -0.2, 0.6),
            (1.0, 0.3, 0.1),
            (0.02, 3.5, 0.97),
            (0.999, -0.5, 0.2),
            (0.0, 9.0, 50.0),
        ];
        let p = Progress::of(nodes.into_iter(), 3600.0, 7200.0);
        assert_eq!((p.time_s, p.end_s), (3600.0, 7200.0));
        assert_eq!(p.eta_range, (-0.2, 0.3));
        assert_eq!((p.max_speed, p.max_film_speed), (0.6, 0.97));
        assert!(p.max_hs.is_none());

        // A flat: the films are the water
        let flat = Progress::of(nodes[2..].iter().copied(), 0.0, 1.0);
        assert_eq!(flat.eta_range, (-0.5, 3.5));
        assert_eq!((flat.max_speed, flat.max_film_speed), (0.97, 0.97));

        // Dry everywhere: an empty range, no speed
        let dry = Progress::of(nodes[4..].iter().copied(), 0.0, 1.0);
        assert!(dry.eta_range.0 > dry.eta_range.1);
        assert_eq!((dry.max_speed, dry.max_film_speed), (0.0, 0.0));
    }
}

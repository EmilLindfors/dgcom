//! Running a case: the 2D production path of `examples/froya_real_data.rs`
//! (`Simulation` + SSP-RK(4,3) or SSP-RK3, optionally multirate, +
//! `SWEPhysics2D`) with its wet/dry defaults: the positivity limiter, the
//! `WetDry` scheme and point-implicit bed friction. A nested boundary's
//! relaxation band is a source term after the Coriolis force, as there.

use std::fs::{self, File};

use crate::analysis::{GaugeComparison, TimeSeries};

mod three_d;
use std::io::{BufWriter, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use super::domain::Domain;
use super::schema::{BoundaryVelocity, Coriolis, Friction, SideTag};
use super::{CaseError, CaseFile};
use crate::boundary::{
    CharacteristicOBC, ExternalStateProvider, HarmonicTide, InverseBarometer,
    MultiBoundaryCondition2D, Reflective2D, SWEBoundaryCondition2D, StillWater, TidalConstituent,
};
use crate::equations::ShallowWater2D;
use crate::io::{SnapshotWriter, write_vtk_swe};
use crate::physics::{PhysicsBuilder, SWEPhysics2D, SWEPhysics2DBuilder};
use crate::simulation::{Simulation, SimulationResult};
use crate::solver::{SWESolution2D, SWEState2D, StandardLimiter2D, WetDryConfig, total_mass_2d};
use crate::source::{
    ChezyFriction2D, CoriolisSource2D, GriddedAtmosphere2D, HorizontalViscosity2D,
    ManningFriction2D, coriolis_parameter,
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
    /// Lowest and highest surface elevation over the wet nodes (m)
    pub eta_range: (f64, f64),
    /// Largest depth-averaged speed (m/s)
    pub max_speed: f64,
}

/// What a station recorded.
#[derive(Clone, Debug, Default)]
pub struct StationSeries {
    /// The station's name
    pub name: String,
    /// Output times (s)
    pub time_s: Vec<f64>,
    /// Surface elevation (m)
    pub eta_m: Vec<f64>,
    /// Depth-averaged velocity along x (m/s)
    pub u_m_s: Vec<f64>,
    /// Depth-averaged velocity along y (m/s)
    pub v_m_s: Vec<f64>,
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

        // Still water at the initial level
        let eta0 = case.initial.eta_m;
        let mut q = SWESolution2D::new(domain.mesh.n_elements, domain.ops.n_nodes);
        for k in ElementIndex::iter(domain.mesh.n_elements) {
            for i in 0..domain.ops.n_nodes {
                let h = (eta0 - domain.bathymetry.get(k, i)).max(0.0);
                q.set_state(k, i, SWEState2D::new(h, 0.0, 0.0));
            }
        }

        let mut output = Output::create(self, domain, clock.as_ref(), None)?;
        let volume = |q: &SWESolution2D| volume(domain, q);
        let volume_start = volume(&q);

        let stop = Arc::new(AtomicBool::new(false));
        let mut failure = None;
        let callback = |q: &SWESolution2D, t: f64| {
            if failure.is_some() {
                return;
            }
            match output.write(t, q, false) {
                Ok(false) => {}
                Ok(true) => progress(&summarize(domain, q, t, t_end)),
                Err(e) => {
                    failure = Some(e);
                    stop.store(true, Ordering::Relaxed);
                }
            }
        };
        let numerics = &case.numerics;
        let cfl = numerics.cfl_at(case.mesh.order);
        let stepper = numerics.integrator.standard();
        let result = match numerics.local_time_stepping_levels {
            0 => drive(
                Simulation::new(physics, stepper),
                cfl,
                interval,
                numerics.dt_max_s,
                &stop,
                &mut q,
                t_end,
                callback,
            ),
            levels => drive(
                Simulation::new(physics, Multirate::with_base(stepper, levels)),
                cfl,
                interval,
                numerics.dt_max_s,
                &stop,
                &mut q,
                t_end,
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
            output.write(result.final_time, &q, true)?;
        }
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
            notes: Vec::new(),
        })
    }

    /// The 2D physics of the case on `domain` with boundary conditions `bc`.
    fn physics<BC: SWEBoundaryCondition2D>(&self, domain: &Domain, bc: BC) -> SWEPhysics2D<BC> {
        let p = &self.case.physics;
        let mut builder = self.swe_builder(domain, bc);
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
        match p.coriolis {
            Some(Coriolis::FPlane(f)) => {
                builder = builder.with_source(CoriolisSource2D::f_plane(f))
            }
            Some(Coriolis::Latitude(lat)) => {
                builder = builder.with_source(CoriolisSource2D::f_plane(coriolis_parameter(lat)));
            }
            Some(Coriolis::BetaPlane) => {
                // The projection's centre is at y = 0
                let center = self.case.projection_center().expect("validated: placed");
                builder =
                    builder.with_source(CoriolisSource2D::beta_plane_at_latitude(center.lat, 0.0));
            }
            None => {}
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

/// Run `sim` from 0 to `t_end`, calling `callback` every `interval` (and at
/// the start and the end).
#[allow(clippy::too_many_arguments)]
fn drive<BC, I>(
    sim: Simulation<SWESolution2D, SWEPhysics2D<BC>, I>,
    cfl: f64,
    interval: f64,
    dt_max: Option<f64>,
    stop: &Arc<AtomicBool>,
    q: &mut SWESolution2D,
    t_end: f64,
    callback: impl FnMut(&SWESolution2D, f64),
) -> SimulationResult
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
    sim.run_with_callback(q, 0.0, t_end, callback)
}

/// The open boundaries of a case, one characteristic condition per tag.
pub(super) type OpenBoundaries = Vec<(SideTag, Box<dyn SWEBoundaryCondition2D>)>;

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
    let (mut lo, mut hi, mut speed) = (f64::INFINITY, f64::NEG_INFINITY, 0.0_f64);
    for k in ElementIndex::iter(domain.mesh.n_elements) {
        for i in 0..domain.ops.n_nodes {
            let s = q.get_state(k, i);
            if s.h > H_VELOCITY {
                let eta = s.h + domain.bathymetry.get(k, i);
                lo = lo.min(eta);
                hi = hi.max(eta);
                speed = speed.max(s.hu.hypot(s.hv) / s.h);
            }
        }
    }
    Progress {
        time_s: t,
        end_s: t_end,
        eta_range: (lo, hi),
        max_speed: speed,
    }
}

/// The run's writers.
struct Output<'a> {
    domain: &'a Domain,
    name: String,
    dir: Option<std::path::PathBuf>,
    snapshot: Option<SnapshotWriter>,
    vtk_frames: Option<usize>,
    #[cfg(feature = "netcdf")]
    netcdf: Option<crate::io::NetCDFWriter>,
    stations: Vec<StationSeries>,
    csv: Option<BufWriter<File>>,
    last_time: f64,
    /// Time between field outputs and between station records (s)
    field_every: f64,
    station_every: f64,
}

impl<'a> Output<'a> {
    /// The writers of `file`'s run on `domain`; with `sigma`, of a 3D run
    /// (its snapshots 3D).
    fn create(
        file: &CaseFile,
        domain: &'a Domain,
        clock: Option<&crate::time::ModelClock>,
        sigma: Option<&crate::vertical::SigmaGrid>,
    ) -> Result<Self, CaseError> {
        let case = &file.case;
        let io = |what: &str, e: &dyn std::fmt::Display| CaseError::Io(format!("{what}: {e}"));
        let stations: Vec<StationSeries> = domain
            .stations
            .iter()
            .map(|s| StationSeries {
                name: s.name.clone(),
                ..Default::default()
            })
            .collect();
        let dir = file.output_dir();
        fs::create_dir_all(&dir).map_err(|e| io(&dir.display().to_string(), &e))?;
        let resolved = dir.join("case.resolved.toml");
        fs::write(&resolved, case.resolved().to_toml())
            .map_err(|e| io(&resolved.display().to_string(), &e))?;
        let name = case.run.name.clone();
        let snapshot = if case.output.snapshot {
            let path = dir.join(format!("{name}.dgsnap"));
            let metadata = [("source", "dgcom"), ("case", name.as_str())];
            let writer = match sigma {
                Some(sigma) => SnapshotWriter::create_3d(
                    &path,
                    &domain.mesh,
                    &domain.ops,
                    &domain.bathymetry.data,
                    sigma,
                    clock,
                    &metadata,
                ),
                None => SnapshotWriter::create(
                    &path,
                    &domain.mesh,
                    &domain.ops,
                    &domain.bathymetry.data,
                    clock,
                    &metadata,
                    H_VELOCITY,
                ),
            };
            Some(writer.map_err(|e| io(&path.display().to_string(), &e))?)
        } else {
            None
        };
        let vtk_frames = if case.output.vtk {
            let vtk = dir.join("vtk");
            fs::create_dir_all(&vtk).map_err(|e| io(&vtk.display().to_string(), &e))?;
            Some(0)
        } else {
            None
        };
        #[cfg(feature = "netcdf")]
        let netcdf = if case.output.netcdf {
            use crate::io::{
                CoordinateProjection, NetCDFMeshInfo, NetCDFWriter, NetCDFWriterConfig,
            };
            let path = dir.join(format!("{name}.nc"));
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
                .with_source("dgcom")
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
            let path = dir.join("stations.csv");
            let mut csv = BufWriter::new(
                File::create(&path).map_err(|e| io(&path.display().to_string(), &e))?,
            );
            writeln!(csv, "station,time_s,eta_m,u_m_s,v_m_s,depth_m")
                .map_err(|e| io("stations.csv", &e))?;
            Some(csv)
        };
        Ok(Self {
            domain,
            name,
            dir: Some(dir),
            snapshot,
            vtk_frames,
            #[cfg(feature = "netcdf")]
            netcdf,
            stations,
            csv,
            last_time: f64::NEG_INFINITY,
            field_every: 60.0 * case.output.interval_minutes,
            station_every: 60.0 * case.output.station_interval(),
        })
    }

    /// Record the stations at `t`.
    fn record_stations(&mut self, t: f64, q: &SWESolution2D) -> Result<(), CaseError> {
        let io = |what: &str, e: &dyn std::fmt::Display| CaseError::Io(format!("{what}: {e}"));
        let domain = self.domain;
        for (station, series) in domain.stations.iter().zip(&mut self.stations) {
            let s = station
                .probe
                .sample_swe(q, Some(&domain.bathymetry), H_VELOCITY);
            series.time_s.push(t);
            series.eta_m.push(s.eta);
            series.u_m_s.push(s.u);
            series.v_m_s.push(s.v);
            if let Some(csv) = &mut self.csv {
                writeln!(csv, "{},{t},{},{},{},{}", series.name, s.eta, s.u, s.v, s.h)
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
    fn write(&mut self, t: f64, q: &SWESolution2D, last: bool) -> Result<bool, CaseError> {
        let io = |what: &str, e: &dyn std::fmt::Display| CaseError::Io(format!("{what}: {e}"));
        let domain = self.domain;
        let on = |every: f64| {
            let n = t / every;
            last || (n - n.round()).abs() < 1e-6
        };
        self.last_time = t;
        if on(self.station_every) {
            self.record_stations(t, q)?;
        }
        if !on(self.field_every) {
            return Ok(false);
        }
        if let Some(snapshot) = &mut self.snapshot {
            snapshot
                .write_state(t, q, &[])
                .map_err(|e| io("snapshot", &e))?;
        }
        if let (Some(frame), Some(dir)) = (&mut self.vtk_frames, &self.dir) {
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
        Ok(true)
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

    /// The stations' records; the writers flush every frame as it is written.
    fn finish(self) -> Vec<StationSeries> {
        self.stations
    }
}

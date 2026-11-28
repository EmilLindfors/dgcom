//! A case's 3D run (`[three_d]`): `Hydrostatic3D` under the case's 2D
//! module, in a `Simulation3D` with mode splitting.
//!
//! The order follows `examples/froya_real_data.rs`'s `tidal_run_3d`, so that
//! a case reproduces its 3D runs bit for bit (`notes/case-files.md`):
//!
//! 1. The open boundaries and the 2D module as in 2D, without the bed
//!    friction (the 3D bottom drag's depth mean replaces it).
//! 2. The start at rest at mean sea level: the parent's columns where it
//!    covers, sampled twice (the second time with the water below its bed
//!    relaxing to the first sample's `MeanProfile`), the rest of the T, S
//!    from that mean (or `three_d.profile` without a parent), and the
//!    parent's shear about a depth mean at rest in columns at least the thin
//!    depth deep.
//! 3. The open faces' band relaxing to the parent's columns (or the start).
//! 4. The physics, its density, and the reference stratification (the mean
//!    profile, or `three_d.profile`); the run, landing on the field outputs.
//!
//! A resumed run (`[restart] from`) builds all of this from the start as
//! the run that wrote the restart did (the band's reference and the
//! reference stratification are of the start), then continues from the
//! restart's state, time and mode-splitting history. Its restarts are
//! written at field outputs, where the steps land, so that the resumed
//! run's steps land where the uninterrupted run's do.

use std::sync::Arc;

use super::{
    CaseDevice, DeviceRun3D, H_VELOCITY, Output, Progress, RunReport, StateView3D, StationSample,
    boundary_conditions, restart, score_gauges, summarize, volume,
};
use crate::boundary::{
    DeepReference, Nesting3D, NestingBand3D, OceanColumnsOptions, OceanModelColumns,
    ParentColumns3D, ReferenceColumns, Reflective2D, sample_columns,
};
use crate::case::schema::{Integrator, Mixing3D, ThreeDSection};
use crate::case::{CaseError, CaseFile, Domain};
use crate::io::Restart3D;
use crate::physics::{
    BottomDrag3D, EquationOfState, Forcing, GlsMixing, Hydrostatic3D, ImplicitVerticalAdvection,
    JerlovWater, LinearEOS,
};
use crate::simulation::{Simulation3D, restart_of, resume_into};
use crate::solver::state::Solution3D;
use crate::solver::{
    Probe2D, SWEPointSample, SWESolution2D, SWEState2D, TracerLimiter3DConfig, TracerLimiterType3D,
    TracerReferenceProfile,
};
use crate::source::{CoriolisSource2D, SpongeProfile};
use crate::time::{ModeSplitIntegrator, SspScheme};
use crate::types::ElementIndex;
use crate::vertical::{MeanProfile, SigmaGrid, SongHaidvogelStretching};

/// Reference density of the momentum equations (kg/m³)
const RHO0: f64 = 1025.0;
/// e-folding depth of the parent's anomaly below its bed (m)
const DEEP_DECAY: f64 = 10.0;

fn ssp(integrator: Integrator) -> SspScheme {
    match integrator {
        Integrator::SspRk43 => SspScheme::Rk43,
        Integrator::SspRk3 => SspScheme::Rk3,
    }
}

impl CaseFile {
    /// Run the 3D case on `domain` (see the module docs), on `device` (its
    /// ordinal and backend) if given.
    pub(super) fn run_3d(
        &self,
        domain: &Domain,
        device: Option<(usize, &dyn CaseDevice)>,
        progress: &mut dyn FnMut(&Progress),
    ) -> Result<RunReport, CaseError> {
        let case = &self.case;
        let d: &ThreeDSection = case.three_d.as_ref().expect("a 3D case");
        let p = &case.physics;
        let clock = case.clock()?;
        let t_end = 3600.0 * case.run.duration_hours;
        let mut notes = Vec::new();

        // 1. The open boundaries and the 2D module, without bed friction
        let opens = self.open_boundaries(domain, clock.as_ref())?;
        let wall = Reflective2D::new();
        let swe = self
            .swe_builder(domain, boundary_conditions(&wall, &opens))
            .build();

        // 2. The start
        let nl = d.levels;
        let sigma = Arc::new(SigmaGrid::new(
            nl,
            SongHaidvogelStretching::new(d.theta_s, d.theta_b, d.hc_m),
        ));
        let eos = LinearEOS::default();
        let (mesh, ops) = (&domain.mesh, &domain.ops);
        let (ne, nn) = (mesh.n_elements, ops.n_nodes);
        let bed = &domain.bathymetry.data;
        let mut state = Solution3D::new(ne, nn, nl);
        let parent_columns = |deep: Option<DeepReference>| -> Option<Arc<dyn ParentColumns3D>> {
            match (&domain.parent_reader, domain.projection, clock) {
                (Some(reader), Some(projection), Some(clock)) if reader.has_tracer_profiles() => {
                    let columns = OceanModelColumns::new(
                        reader.clone(),
                        mesh,
                        ops,
                        projection,
                        clock,
                        OceanColumnsOptions::default(),
                    );
                    Some(Arc::new(match deep {
                        Some(deep) => columns.with_deep_reference(deep),
                        None => columns,
                    }))
                }
                _ => None,
            }
        };
        let mut covered = vec![false; ne * nn];
        let mut columns = parent_columns(None);
        sample_columns(
            columns.as_deref(),
            mesh,
            ops,
            &sigma,
            bed,
            &mut state,
            &mut covered,
        );
        let n_covered = covered.iter().filter(|&&c| c).count();
        let mean = if n_covered > 0 {
            MeanProfile::of(
                &state,
                bed,
                &sigma,
                |idx| covered[idx],
                |t, s| eos.compute_density(t, s, 0.0),
            )
        } else {
            None
        };
        // Sampled again with the water below the parent's bed relaxing to
        // the mean: held, its bottom water ran down the deep holes' walls
        if let Some(mean) = mean.clone() {
            columns = parent_columns(Some(DeepReference {
                profile: mean.clone(),
                decay: DEEP_DECAY,
            }));
            sample_columns(
                columns.as_deref(),
                mesh,
                ops,
                &sigma,
                bed,
                &mut state,
                &mut covered,
            );
        }
        if mean.is_none() && d.profile.is_empty() {
            return Err(CaseError::Invalid(
                "[three_d]: the parent has no temperature and salinity profiles (fetch it with \
                 `norkyst_nesting_subset profiles=1`), and three_d.profile gives none"
                    .into(),
            ));
        }
        // The reference stratification: the parent's mean profile, or the
        // case's. The nodes no parent covers start exactly in it
        let reference = Arc::new(match &mean {
            Some(mean) => TracerReferenceProfile::from_mean(mean),
            None => d.reference_profile().expect("checked: a profile"),
        });
        state.fill_tracers(&sigma, bed, |idx| !covered[idx], |z| reference.at(z));
        let thin = d.thin_depth_m;
        let eta0 = state.eta.data.clone();
        let n_sheared =
            state.keep_shear(&sigma, |idx| covered[idx] && eta0[idx] - bed[idx] >= thin);
        let wet = bed.iter().filter(|&&b| b < 0.0).count();
        notes.push(format!(
            "3D: {ne} elements × {nn} nodes × {nl} levels = {:.2} M points",
            (ne * nn * nl) as f64 / 1e6
        ));
        notes.push(match &columns {
            Some(_) => format!(
                "initial T, S from the parent at {n_covered} of {wet} wet nodes (the others its \
                 mean profile); its shear about a depth mean at rest at {n_sheared} nodes"
            ),
            None => "initial T, S from three_d.profile".into(),
        });

        // 3. The open faces' band: to the parent's columns, or the start
        let (band_km, band_minutes) = case
            .boundaries
            .iter()
            .find_map(|b| b.nesting.as_ref())
            .map_or((d.band_km, d.band_minutes), |n| (n.band_km, n.band_minutes));
        let band = NestingBand3D {
            width: 1000.0 * band_km,
            profile: SpongeProfile::default(),
            velocity_timescale: Some(60.0 * band_minutes),
            tracer_timescale: Some(60.0 * band_minutes),
        };
        let relaxed_to: Arc<dyn ParentColumns3D> = match &columns {
            Some(columns) => columns.clone(),
            None => Arc::new(ReferenceColumns::from_state(&state)),
        };
        let tags: Vec<_> = domain.open_tags.iter().map(|t| t.boundary_tag()).collect();
        // None without open faces (a closed basin)
        let nesting = Nesting3D::new(relaxed_to, mesh, ops, nl, &tags, &band).ok();
        let band_text = format!("over a {band_km} km band, {band_minutes} min");
        notes.push(match (&nesting, &columns) {
            (None, _) => "open faces: none".into(),
            (Some(_), Some(_)) => {
                format!("open faces: nested in the parent's u, v, T, S {band_text}")
            }
            (Some(_), None) => {
                format!("open faces: relaxed to the stratification at rest {band_text}")
            }
        });

        // 4. The physics
        let limiter = TracerLimiter3DConfig {
            limiter_type: TracerLimiterType3D::HorizontalKuzmin { relaxation: 1.0 },
            ..TracerLimiter3DConfig::default()
        }
        .with_reference_profile(reference.clone());
        let coriolis = Arc::new(
            case.coriolis_source()
                .unwrap_or_else(|| CoriolisSource2D::f_plane(0.0)),
        );
        let mixing = match d.mixing {
            Mixing3D::KEpsilon => GlsMixing::k_epsilon(),
            Mixing3D::KOmega => GlsMixing::k_omega(),
            Mixing3D::Generic => GlsMixing::generic(),
        };
        let viscous = p.viscosity_m2_s > 0.0 || p.smagorinsky > 0.0;
        let mut physics = Hydrostatic3D::new(
            mesh.clone(),
            ops.clone(),
            domain.geom.clone(),
            sigma.clone(),
            domain.bathymetry.clone(),
            coriolis,
            eos,
            mixing,
            swe,
            Forcing {
                surface_stress: [0.0, 0.0],
                bottom_stress: [0.0, 0.0],
                surface_heat_flux: 0.0,
            },
            p.gravity,
            RHO0,
        )
        .with_bottom_drag(BottomDrag3D::log_layer(d.bottom_roughness_m))
        .with_min_column_depth(thin)
        .with_horizontal_viscosity(p.viscosity_m2_s)
        .with_smagorinsky_viscosity(if viscous {
            p.smagorinsky
        } else {
            d.shear_smagorinsky
        })
        .with_tracer_limiter(limiter);
        // The farm's cages on the layers their nets reach
        if let Some(drag) = super::cage_drag(domain) {
            physics = physics.with_cage_drag(drag);
        }
        if let Some(nesting) = nesting {
            physics = physics.with_nesting(nesting);
        }
        if let Some(wind) = &domain.wind_3d {
            physics = physics.with_surface_stress(wind.clone());
        }
        if let Some(fluxes) = &domain.fluxes_3d {
            let water: JerlovWater = d.jerlov.parse().expect("validated");
            physics = physics
                .with_surface_fluxes(fluxes.clone())
                .with_shortwave_absorption(water);
        }
        if let Some(rivers) = &domain.rivers {
            physics = physics.with_rivers(rivers.clone());
        }
        // The vertical advection beyond an explicit Courant number
        // implicitly: the columns just deeper than the thin depth set the
        // step otherwise
        physics = physics.with_implicit_vertical_advection(ImplicitVerticalAdvection::default());
        physics.update_density(&mut state);
        // The anomalies about the mean profile: its pressure gradient
        // (Mellor et al. 1998) and the advection that is its adjoint
        let mut physics = physics.with_reference_profile(reference);

        // A resumed run: the restart's state, time, history and records. A
        // patch started from the whole run's restart takes its elements'
        // columns and history at its time, with outputs of its own
        let mut seeded = false;
        let restart = match self.resume_path() {
            Some(path) => {
                let restart = Restart3D::read(&path)
                    .map_err(|e| CaseError::Io(format!("{}: {e}", path.display())))?;
                // The domain before any output is touched (`Simulation3D::resume`
                // checks it again)
                let fingerprint = crate::io::domain_fingerprint_3d(mesh, bed, &sigma);
                let restart = match domain.patch.as_ref() {
                    Some(patch) if patch.is_whole(restart.state.n_elements) => {
                        self.check_seed(&path, &restart.driver, restart.state.n_nodes, domain)?;
                        if restart.state.n_levels != nl {
                            return Err(CaseError::Invalid(format!(
                                "cannot start the patch from {}: {} levels, the case {nl}",
                                path.display(),
                                restart.state.n_levels
                            )));
                        }
                        notes.push(restart::seed_note(&path, restart.time, &restart.driver));
                        seeded = true;
                        patch.seed_3d(&restart, fingerprint)
                    }
                    _ => {
                        if restart.domain != fingerprint {
                            return Err(CaseError::Invalid(format!(
                                "cannot resume from {}: its mesh, bed or σ-grid differ",
                                path.display()
                            )));
                        }
                        self.check_resumed(&path, &restart.driver)?;
                        restart
                    }
                };
                Some((path, restart))
            }
            None => None,
        };
        let t_start = restart.as_ref().map_or(0.0, |(_, r)| r.time);
        if t_start >= t_end - 1e-6 {
            return Err(CaseError::Invalid(format!(
                "the restart is at {:.2} h, at or past the run's end ({} h)",
                t_start / 3600.0,
                case.run.duration_hours
            )));
        }
        // A seeded patch's outputs are its own, from its start
        let resumed = restart
            .as_ref()
            .filter(|_| !seeded)
            .map(|(path, r)| restart::Resumed {
                path: path.clone(),
                time: r.time,
                driver: r.driver.clone(),
            });
        notes.extend(resumed.as_ref().map(restart::Resumed::note));

        // The run: the steps land on the field outputs. The stations are
        // sampled at every step and recorded at their times, between two
        // steps by linear interpolation, so their series are regular without
        // their times moving the run's steps (the 2D run lands on both)
        let mut output = Output::create(
            self,
            domain,
            clock.as_ref(),
            Some(&sigma),
            &[],
            resumed.as_ref(),
        )?;
        let field_every = 60.0 * case.output.interval_minutes;
        let station_every = 60.0 * case.output.station_interval();
        let mut q = SWESolution2D::new(ne, nn);
        let barotropic = |s: &Solution3D, q: &mut SWESolution2D| {
            for k in ElementIndex::iter(ne) {
                for i in 0..nn {
                    let idx = k.as_usize() * nn + i;
                    let h = (s.eta.data[idx] - bed[idx]).max(0.0);
                    q.set_state(
                        k,
                        i,
                        SWEState2D::new(h, h * s.ubar.data[idx], h * s.vbar.data[idx]),
                    );
                }
            }
        };
        // The volume at the start: a seeded patch's is its restart's
        barotropic(restart.as_ref().map_or(&state, |(_, r)| &r.state), &mut q);
        let volume_start = match &resumed {
            Some(r) => restart::counters(&r.driver)?.1,
            None => volume(domain, &q),
        };
        // The first station time after the restart's (which the stopped run
        // recorded), or from the start
        let mut next_station = match &resumed {
            Some(r) => ((r.time / station_every + 1e-9).floor() + 1.0) * station_every,
            None => (t_start / station_every - 1e-9).ceil().max(0.0) * station_every,
        };
        let mut failure = None;
        let mut restarts = restart::RestartSchedule::new(self, t_start);
        let mut nodes = vec![0.0; 3 * nn];
        let mut previous: Option<(f64, Vec<StationSample>)> = None;
        // A resumed run's first call is at the restart's time, which the
        // stopped run wrote out: its samples are the next interpolation's
        // start only
        let mut at_restart = resumed.is_some();
        // Every step reads the depth means (the stations), a field output
        // the whole state: a device run copies only what is read
        let mut on_step = |view: &mut dyn StateView3D,
                           t: f64,
                           ranges: &dyn Fn(&Solution3D) -> crate::physics::TracerRanges,
                           capture: &dyn Fn(&Solution3D) -> Restart3D| {
            if failure.is_some() {
                return;
            }
            let first = std::mem::take(&mut at_restart);
            let field_time = {
                let n = t / field_every;
                (n - n.round()).abs() < 1e-6 || t >= t_end - 1e-6
            };
            let on_field = field_time && !first;
            let now: Vec<StationSample> = if domain.stations.is_empty() {
                Vec::new()
            } else {
                let s = view.depth_means();
                domain
                    .stations
                    .iter()
                    .map(|st| {
                        StationSample::of(
                            &sample_depth_mean(&st.level, s, bed, &mut nodes),
                            &sample_depth_mean(&st.current, s, bed, &mut nodes),
                        )
                    })
                    .collect()
            };
            let written = (|| -> Result<(), CaseError> {
                while next_station <= t + 1e-6 {
                    if (t - next_station).abs() <= 1e-6 {
                        output.record_samples(t, &now, None)?;
                    } else {
                        let (t0, before) = previous.as_ref().expect("time 0 is a station time");
                        let w = (next_station - t0) / (t - t0);
                        let between: Vec<StationSample> =
                            before.iter().zip(&now).map(|(a, b)| a.mix(b, w)).collect();
                        output.record_samples(next_station, &between, None)?;
                    }
                    next_station += station_every;
                }
                // A new snapshot file starts with the restart's frame
                if first && field_time && !output.snapshot_continued {
                    output.write_snapshot_3d(t, view.full())?;
                }
                if on_field {
                    let s = view.full();
                    barotropic(s, &mut q);
                    output.write_snapshot_3d(t, s)?;
                    if let Some(restarts) = restarts.as_mut().filter(|r| r.due(t)) {
                        let mut restart = capture(s);
                        restart.driver = output.driver_data(self, volume_start);
                        restarts.write(t, |path| restart.write(path))?;
                    }
                }
                Ok(())
            })();
            previous = Some((t, now));
            match written {
                Ok(()) if on_field => progress(&Progress {
                    tracers: Some(ranges(view.full())),
                    ..summarize(domain, &q, t, t_end)
                }),
                Ok(()) => {}
                Err(e) => failure = Some(e),
            }
        };
        let mut integrator = ModeSplitIntegrator::new()
            .with_barotropic_cfl(d.barotropic_cfl)
            .with_barotropic_scheme(ssp(d.barotropic_scheme))
            .with_baroclinic_scheme(ssp(d.scheme))
            .with_barotropic_levels(d.barotropic_levels)
            .with_baroclinic_levels(d.levels_lts);
        if let Some((path, restart)) = restart {
            resume_into(&mut physics, &mut integrator, &restart).map_err(|e| {
                CaseError::Invalid(format!("cannot resume from {}: {e}", path.display()))
            })?;
            state = restart.state;
        }
        // On the CPU, or on the device (whose callbacks see the state on the
        // host)
        let ran = match device {
            None => {
                let mut sim = Simulation3D::new(physics, integrator)
                    .with_cfl(d.cfl())
                    .with_dt_max(d.dt_max_s)
                    .with_landing_interval(field_every);
                let result =
                    sim.run_with_context_callback(&mut state, t_start, t_end, |s, t, context| {
                        on_step(&mut &*s, t, &|s| context.physics.tracer_ranges(s), &|s| {
                            context.restart(s, t)
                        })
                    });
                let integrator = sim.integrator();
                let mut steps = vec![0.0; domain.mesh.n_elements];
                sim.physics()
                    .element_steps_into(&state, d.cfl(), ssp(d.scheme), &mut steps);
                Ran3D {
                    last_substeps: integrator.last_substeps(),
                    clips: sim.physics().swe_physics.negative_depth_clips(),
                    barotropic: integrator.barotropic_multirate_stats(),
                    baroclinic: integrator.baroclinic_multirate_stats(),
                    steps,
                    result,
                }
            }
            Some((ordinal, device)) => {
                let run = DeviceRun3D {
                    cfl: d.cfl(),
                    dt_max: d.dt_max_s,
                    landing_interval: field_every,
                    span: (t_start, t_end),
                };
                let physics = &physics;
                let report = device
                    .run_3d(
                        ordinal,
                        physics,
                        integrator,
                        &mut state,
                        &run,
                        &mut |view, t, record| {
                            on_step(view, t, &|s| physics.tracer_ranges(s), &|s| {
                                restart_of(physics, record(), s, t)
                            })
                        },
                    )
                    .map_err(|e| CaseError::Invalid(format!("the 3D run on the device: {e}")))?;
                notes.push(format!("computed on cuda:{ordinal}, {}", report.device));
                let mut steps = vec![0.0; domain.mesh.n_elements];
                physics.element_steps_into(&state, d.cfl(), ssp(d.scheme), &mut steps);
                Ran3D {
                    last_substeps: report.last_substeps,
                    clips: report.negative_depth_clips,
                    barotropic: report.barotropic,
                    baroclinic: report.baroclinic,
                    steps,
                    result: report.result,
                }
            }
        };
        let result = &ran.result;
        if let Some(e) = failure {
            return Err(e);
        }
        if !result.success {
            return Err(CaseError::Run(format!(
                "the 3D run stopped at {:.2} h: {}",
                result.final_time / 3600.0,
                result.error.as_deref().unwrap_or("unknown error")
            )));
        }
        barotropic(&state, &mut q);
        notes.extend(restarts.map(|r| r.note()));
        let mut stations = output.finish();
        if let Some(clock) = &clock {
            score_gauges(
                &domain.tide_constituents,
                domain.stations.iter().map(|s| s.gauge.as_ref()),
                &mut stations,
                clock,
                3600.0 * case.spinup_hours(),
            );
            if stations.iter().any(|s| s.gauge.is_some()) {
                super::write_gauge_scores(&self.output_dir(), &stations)?;
            }
        }
        notes.push(format!(
            "{} barotropic substeps in the last step, {} negative-depth clips",
            ran.last_substeps, ran.clips
        ));
        // The work of the barotropic pass and of the 3D stages on local time
        // steps, and the steps the elements could take at the end: what
        // global steps, more levels or another cap would cost
        if let Some(stats) = ran.barotropic {
            notes.push(format!(
                "barotropic pass: {}",
                stats.describe(domain.mesh.n_elements)
            ));
        }
        if let Some(stats) = ran.baroclinic {
            notes.push(format!(
                "3D stages: {}",
                stats.describe(domain.mesh.n_elements)
            ));
        }
        let steps = &ran.steps;
        notes.extend(describe_element_steps(steps, d.dt_max_s).map(|steps| {
            format!(
                "3D element steps at {:.2} h: {steps}",
                result.final_time / 3600.0
            )
        }));
        Ok(RunReport {
            final_time_s: result.final_time,
            n_steps: result.n_steps,
            dt_range_s: (result.dt_min, result.dt_max),
            wall_time_s: result.wall_time,
            volume_m3: (volume_start, volume(domain, &q)),
            stations,
            notes,
        })
    }
}

/// What a 3D run leaves for its report, on the CPU or a device.
struct Ran3D {
    result: crate::simulation::SimulationResult,
    last_substeps: usize,
    clips: usize,
    barotropic: Option<crate::time::MultirateStats>,
    baroclinic: Option<crate::time::MultirateStats>,
    /// Every element's own 3D step at the end
    steps: Vec<f64>,
}

/// The elements' own 3D steps `steps` (s; infinite where nothing bounds
/// one) in a line: the smallest, which global steps take, the 1 %, 5 % and
/// median steps, and the share of the elements that could step past `cap`
/// (`three_d.dt_max_s`), which the levels of local time stepping cannot
/// use. Nothing where no element has a bound.
fn describe_element_steps(steps: &[f64], cap: f64) -> Option<String> {
    let mut bounded: Vec<f64> = steps.iter().copied().filter(|dt| dt.is_finite()).collect();
    bounded.sort_by(f64::total_cmp);
    let smallest = *bounded.first()?;
    // Of all elements: an unbounded one is past every step
    let quantile = |q: f64| {
        let i = ((steps.len() - 1) as f64 * q) as usize;
        bounded.get(i).copied().unwrap_or(f64::INFINITY)
    };
    let over = steps.iter().filter(|&&dt| dt > cap).count();
    Some(format!(
        "smallest {smallest:.1} s, 1 % under {:.1} s, 5 % under {:.1} s, median {:.1} s; \
         {:.0} % over the {cap} s cap (three_d.dt_max_s)",
        quantile(0.01),
        quantile(0.05),
        quantile(0.5),
        100.0 * over as f64 / steps.len() as f64
    ))
}

/// The depth-mean state of `s` at `probe`, as [`Probe2D::sample_swe`] reads
/// it from the 2D state (`h = max(η − B, 0)`, `hū`, `hv̄` at the nodes) on
/// bed `bed`: the same values at the same time. `nodes` holds the element's
/// `3 × n_nodes` node values.
fn sample_depth_mean(
    probe: &Probe2D,
    s: &Solution3D,
    bed: &[f64],
    nodes: &mut [f64],
) -> SWEPointSample {
    let nn = probe.weights().len();
    let start = probe.element().as_usize() * nn;
    let (h, rest) = nodes.split_at_mut(nn);
    let (hu, hv) = rest.split_at_mut(nn);
    for i in 0..nn {
        let j = start + i;
        h[i] = (s.eta.data[j] - bed[j]).max(0.0);
        hu[i] = h[i] * s.ubar.data[j];
        hv[i] = h[i] * s.vbar.data[j];
    }
    let (h, hu, hv) = (probe.evaluate(h), probe.evaluate(hu), probe.evaluate(hv));
    let bed = probe.evaluate(&bed[start..start + nn]);
    let (u, v) = if h > H_VELOCITY {
        (hu / h, hv / h)
    } else {
        (0.0, 0.0)
    };
    SWEPointSample {
        h,
        hu,
        hv,
        bed,
        eta: h + bed,
        u,
        v,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The run's note on the elements' own steps: quantiles over all
    /// elements, the unbounded ones past every step.
    #[test]
    fn element_steps_are_described_by_their_quantiles_and_the_cap() {
        let mut steps: Vec<f64> = (1..=100).map(|k| k as f64).collect();
        steps.extend([f64::INFINITY; 100]);
        assert_eq!(
            describe_element_steps(&steps, 60.0).unwrap(),
            "smallest 1.0 s, 1 % under 2.0 s, 5 % under 10.0 s, median 100.0 s; \
             70 % over the 60 s cap (three_d.dt_max_s)"
        );
        assert_eq!(describe_element_steps(&[f64::INFINITY; 4], 60.0), None);
    }
}

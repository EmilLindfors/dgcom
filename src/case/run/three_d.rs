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
//! 4. The physics, its density, and the pressure gradient's reference
//!    profile; the run, landing on the field outputs.
//!
//! A resumed run (`[restart] from`) builds all of this from the start as
//! the run that wrote the restart did (the band's and the pressure
//! gradient's references are of the start), then continues from the
//! restart's state, time and mode-splitting history. Its restarts are
//! written at field outputs, where the steps land, so that the resumed
//! run's steps land where the uninterrupted run's do.

use std::sync::Arc;

use super::{
    H_VELOCITY, Output, Progress, RunReport, boundary_conditions, restart, score_gauges, summarize,
    volume,
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
use crate::simulation::Simulation3D;
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
    /// Run the 3D case on `domain` (see the module docs).
    pub(super) fn run_3d(
        &self,
        domain: &Domain,
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
                profile: Arc::new(move |depth| mean.at(-depth)),
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
        let profile = |z: f64| match &mean {
            Some(m) => m.at(z),
            None => d.profile_at(z).expect("checked: a profile"),
        };
        state.fill_tracers(&sigma, bed, |idx| !covered[idx], profile);
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
        let nesting = (!tags.is_empty())
            .then(|| Nesting3D::new(relaxed_to, mesh, ops, nl, &tags, &band).ok())
            .flatten();
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
        let deepest = bed.iter().copied().fold(0.0, f64::min);
        let limiter = TracerLimiter3DConfig {
            limiter_type: TracerLimiterType3D::HorizontalKuzmin { relaxation: 1.0 },
            ..TracerLimiter3DConfig::default()
        }
        .with_reference_profile(TracerReferenceProfile::from_fn(
            deepest - 1.0,
            1.0,
            4001,
            profile,
        ));
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
        // The pressure gradient about the mean profile (Mellor et al. 1998)
        let physics = physics.with_reference_profile(&state, |z| {
            let (t, s) = profile(z);
            eos.compute_density(t, s, z)
        });

        // A resumed run: the restart's state, time, history and records
        let restart = match self.resume_path() {
            Some(path) => {
                let restart = Restart3D::read(&path)
                    .map_err(|e| CaseError::Io(format!("{}: {e}", path.display())))?;
                // The domain before any output is touched (`Simulation3D::resume`
                // checks it again)
                let domain = crate::io::domain_fingerprint_3d(mesh, bed, &sigma);
                if restart.domain != domain {
                    return Err(CaseError::Invalid(format!(
                        "cannot resume from {}: its mesh, bed or σ-grid differ",
                        path.display()
                    )));
                }
                self.check_resumed(&path, &restart.driver)?;
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
        let resumed = restart.as_ref().map(|(path, r)| restart::Resumed {
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
        barotropic(&state, &mut q);
        let volume_start = match &resumed {
            Some(r) => restart::counters(&r.driver)?.1,
            None => volume(domain, &q),
        };
        // The first station time after the restart's (which the stopped run
        // recorded)
        let mut next_station = match &resumed {
            Some(r) => ((r.time / station_every + 1e-9).floor() + 1.0) * station_every,
            None => 0.0,
        };
        let mut failure = None;
        let mut restarts = restart::RestartSchedule::new(self, t_start);
        let mut nodes = vec![0.0; 3 * nn];
        let mut previous: Option<(f64, Vec<SWEPointSample>)> = None;
        // A resumed run's first call is at the restart's time, which the
        // stopped run wrote out: its samples are the next interpolation's
        // start only
        let mut at_restart = resumed.is_some();
        let mut on_step = |s: &Solution3D, t: f64, capture: &dyn Fn() -> Restart3D| {
            if failure.is_some() {
                return;
            }
            let first = std::mem::take(&mut at_restart);
            let field_time = {
                let n = t / field_every;
                (n - n.round()).abs() < 1e-6 || t >= t_end - 1e-6
            };
            let on_field = field_time && !first;
            let now: Vec<SWEPointSample> = domain
                .stations
                .iter()
                .map(|st| sample_depth_mean(&st.probe, s, bed, &mut nodes))
                .collect();
            let written = (|| -> Result<(), CaseError> {
                while next_station <= t + 1e-6 {
                    if (t - next_station).abs() <= 1e-6 {
                        output.record_samples(t, &now, None)?;
                    } else {
                        let (t0, before) = previous.as_ref().expect("time 0 is a station time");
                        let w = (next_station - t0) / (t - t0);
                        let between: Vec<SWEPointSample> = before
                            .iter()
                            .zip(&now)
                            .map(|(a, b)| interpolate(a, b, w))
                            .collect();
                        output.record_samples(next_station, &between, None)?;
                    }
                    next_station += station_every;
                }
                // A new snapshot file starts with the restart's frame
                if first && field_time && !output.snapshot_continued {
                    output.write_snapshot_3d(t, s)?;
                }
                if on_field {
                    barotropic(s, &mut q);
                    output.write_snapshot_3d(t, s)?;
                    if let Some(restarts) = restarts.as_mut().filter(|r| r.due(t)) {
                        let mut restart = capture();
                        restart.driver = output.driver_data(self, volume_start);
                        restarts.write(t, |path| restart.write(path))?;
                    }
                }
                Ok(())
            })();
            previous = Some((t, now));
            match written {
                Ok(()) if on_field => progress(&summarize(domain, &q, t, t_end)),
                Ok(()) => {}
                Err(e) => failure = Some(e),
            }
        };
        let integrator = ModeSplitIntegrator::new()
            .with_barotropic_cfl(d.barotropic_cfl)
            .with_barotropic_scheme(ssp(d.barotropic_scheme))
            .with_baroclinic_scheme(ssp(d.scheme))
            .with_barotropic_levels(d.barotropic_levels)
            .with_baroclinic_levels(d.levels_lts);
        let mut sim = Simulation3D::new(physics, integrator)
            .with_cfl(d.cfl())
            .with_dt_max(d.dt_max_s)
            .with_landing_interval(field_every);
        if let Some((path, restart)) = restart {
            sim.resume(&restart).map_err(|e| {
                CaseError::Invalid(format!("cannot resume from {}: {e}", path.display()))
            })?;
            state = restart.state;
        }
        let result = sim.run_with_context_callback(&mut state, t_start, t_end, |s, t, context| {
            on_step(s, t, &|| context.restart(s, t))
        });
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
            score_gauges(domain, &mut stations, clock, 3600.0 * case.spinup_hours());
            if stations.iter().any(|s| s.gauge.is_some()) {
                super::write_gauge_scores(&self.output_dir(), &stations)?;
            }
        }
        notes.push(format!(
            "{} barotropic substeps in the last step, {} negative-depth clips",
            sim.integrator().last_substeps(),
            sim.physics().swe_physics.negative_depth_clips()
        ));
        // The 3D stages' work on local time steps
        if let Some(stats) = sim.integrator().baroclinic_multirate_stats() {
            notes.push(format!(
                "3D stages: {}",
                stats.describe(domain.mesh.n_elements)
            ));
        }
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

/// `a` and `b` mixed linearly, `w` of the way from `a` to `b`.
fn interpolate(a: &SWEPointSample, b: &SWEPointSample, w: f64) -> SWEPointSample {
    let mix = |x: f64, y: f64| x + w * (y - x);
    SWEPointSample {
        h: mix(a.h, b.h),
        hu: mix(a.hu, b.hu),
        hv: mix(a.hv, b.hv),
        bed: mix(a.bed, b.bed),
        eta: mix(a.eta, b.eta),
        u: mix(a.u, b.u),
        v: mix(a.v, b.v),
    }
}

//! A case's spectral waves (`[waves]`), coupled to its 2D run: the model on
//! the waves' mesh, as `examples/froya_real_data.rs`'s `wave_model` and
//! `coupled_waves` build it.

use crate::boundary::SWEBoundaryCondition2D;
use crate::case::schema::{WaveForce, WaveIntegration, WaveLimiter, WavesSection};
use crate::case::{CaseError, CaseFile, WaveDomain};
use crate::physics::SWEPhysics2D;
use crate::waves::{
    BoundarySpectra, CoupledWaves2D, DEFAULT_RATE_LIMITER, GrowthLimiter, SourceIntegration,
    SourceTerms, SpectralGrid, WaveForceForm, WaveModel2D,
};

impl CaseFile {
    /// The waves of `spec` on `w`, coupled to `physics` from time 0, and
    /// what was set up, line by line.
    pub(super) fn coupled_waves<BC: SWEBoundaryCondition2D>(
        &self,
        spec: &WavesSection,
        w: &WaveDomain,
        physics: &SWEPhysics2D<BC>,
        notes: &mut Vec<String>,
    ) -> Result<CoupledWaves2D, CaseError> {
        let g = self.case.physics.gravity;
        let grid = SpectralGrid::new(
            spec.f_min_hz,
            spec.f_max_hz,
            spec.frequencies,
            spec.directions,
        );
        // Coming from `from` (clockwise from north) is travelling to
        // 270° − from (counter-clockwise from east, the mesh's x)
        let sea = spec
            .sea
            .as_ref()
            .map(|s| grid.jonswap(s.hs_m, s.tp_s, 3.3, (270.0 - s.from_deg).to_radians(), 4.0));
        let (integration, limiter, substeps, cfl) = spec.integration();
        let mut sources = SourceTerms::swan_defaults(g).with_depth_limit(spec.depth_limit);
        if integration == WaveIntegration::Implicit {
            sources = sources.with_integration(SourceIntegration::Implicit);
        }
        match limiter {
            Some(WaveLimiter::Rate) => {
                sources = sources.with_limiter(Some(GrowthLimiter::Rate(DEFAULT_RATE_LIMITER)));
            }
            Some(WaveLimiter::RateBothSigns) => {
                sources =
                    sources.with_limiter(Some(GrowthLimiter::RateBothSigns(DEFAULT_RATE_LIMITER)));
            }
            Some(WaveLimiter::Ris) | None => {}
        }
        let mut model = WaveModel2D::new(
            w.mesh.clone(),
            w.ops.clone(),
            w.geom.clone(),
            &w.bathymetry,
            grid,
            g,
        )
        .with_sources(sources);
        if let Some(sea) = &sea {
            model = model.with_boundary_spectrum(sea);
        }
        let model = model
            .with_turning_limit(spec.turning_rad_s)
            .with_implicit_refraction(spec.implicit)
            .with_implicit_frequency_shift(spec.implicit)
            .with_substeps(substeps)
            .with_absorbing_land(spec.absorbing_land);
        notes.push(format!(
            "waves: exchanging every {} min, {} elements (P{}), {} nodes, {} × {} components, \
             {:?} integration, {} propagation steps per source step at CFL {cfl}",
            spec.exchange_minutes,
            w.mesh.n_elements,
            w.ops.order,
            model.n_points(),
            spec.frequencies,
            spec.directions,
            integration,
            substeps
        ));
        let force = match spec.force {
            WaveForce::Dissipation => WaveForceForm::Dissipation,
            WaveForce::Stress => WaveForceForm::RadiationStress,
        };
        let ramp = 3600.0 * spec.ramp_hours.max(1e-3);
        let mut waves = match (&w.parent, sea) {
            (Some((spectra, wind)), _) => {
                let boundary = BoundarySpectra::for_model(
                    &model,
                    spectra,
                    spec.spectra.as_ref().expect("validated").neighbours,
                );
                // The boundary's mean spectrum everywhere at the start
                let nc = model.grid.n_components();
                let at_start = boundary.at(0.0);
                let mut mean = vec![0.0; nc];
                for spectrum in at_start.chunks_exact(nc) {
                    mean.iter_mut().zip(spectrum).for_each(|(m, x)| *m += x);
                }
                let n = boundary.n_targets().max(1) as f64;
                mean.iter_mut().for_each(|m| *m /= n);
                notes.push(format!(
                    "waves: the boundary from {} parent points at {} open-boundary nodes, H_s \
                     {:.2} m at the start on average",
                    spectra.positions.len(),
                    boundary.n_targets(),
                    model.grid.parameters(&mean).hs
                ));
                let state = model.uniform_state(&mean);
                let waves = CoupledWaves2D::new(model, state, physics, 0.0)
                    .with_ramp(ramp)
                    .with_cfl(cfl)
                    .with_force_form(force)
                    .with_boundary(boundary);
                match (&w.weather, wind) {
                    (Some(_), _) => waves,
                    (None, Some(series)) => {
                        notes.push("waves: the parent's wind".into());
                        waves.with_wind(series.clone())
                    }
                    (None, None) => waves,
                }
            }
            (None, Some(sea)) => {
                let s = spec.sea.as_ref().expect("validated");
                notes.push(format!(
                    "waves: JONSWAP H_s {} m, T_p {} s from {}° at the boundary and everywhere \
                     at the start",
                    s.hs_m, s.tp_s, s.from_deg
                ));
                let state = model.uniform_state(&sea);
                CoupledWaves2D::new(model, state, physics, 0.0)
                    .with_ramp(ramp)
                    .with_cfl(cfl)
                    .with_force_form(force)
            }
            (None, None) => unreachable!("validated: a sea or spectra"),
        };
        if let Some(weather) = &w.weather {
            notes.push("waves: the weather's wind per wave node".into());
            waves = waves.with_gridded_wind(weather.clone());
        }
        let (to_waves, to_circulation) = (
            waves.coupling().to_waves(),
            waves.coupling().to_circulation(),
        );
        notes.push(format!(
            "waves: wave nodes outside the run's mesh {} of {} (up to {:.0} m from it); the \
             run's nodes outside the waves' {} of {} (up to {:.0} m)",
            to_waves.n_outside(),
            to_waves.n_target_points(),
            to_waves.largest_gap(),
            to_circulation.n_outside(),
            to_circulation.n_target_points(),
            to_circulation.largest_gap()
        ));
        Ok(waves)
    }
}

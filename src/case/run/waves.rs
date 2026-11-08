//! A case's spectral waves (`[waves]`), coupled to its 2D run: the model on
//! the waves' mesh, as `examples/froya_real_data.rs`'s `wave_model` and
//! `coupled_waves` build it.

use crate::boundary::SWEBoundaryCondition2D;
use crate::case::schema::{WaveForce, WaveIntegration, WaveLimiter, WavesSection};
use std::path::Path;

use crate::case::{CaseError, CaseFile, ParentWavePoints, WaveDomain};
use crate::mesh::PointLocator2D;
use crate::physics::SWEPhysics2D;
use crate::solver::Probe2D;
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

/// Our waves at a parent wave model's points against the parent's own H_s
/// there, at every exchange (the waves' own time), as
/// `froya_real_data`'s `wave_points.txt`: H_s from the basis at the point, the
/// peak period and the mean direction at its nearest wave node.
pub(super) struct WavePoints<'a> {
    parent: &'a ParentWavePoints,
    /// Each point's probe on the waves' mesh (none outside it)
    probes: Vec<Option<Probe2D>>,
    /// Geographic east in the mesh axes (rad from the x axis)
    east: f64,
    /// The time (s) and, per point, our H_s, the parent's, our T_p and our
    /// direction (coming from, degrees clockwise from north)
    rows: Vec<(f64, Vec<[f64; 4]>)>,
    spectrum: Vec<f64>,
}

impl<'a> WavePoints<'a> {
    /// The parent's points located on `w`'s mesh, with geographic east at
    /// `east`, and a line for the log.
    pub(super) fn new(parent: &'a ParentWavePoints, w: &WaveDomain, east: f64) -> (Self, String) {
        let locator = PointLocator2D::new(&w.mesh);
        let probes: Vec<Option<Probe2D>> = parent
            .positions
            .iter()
            .map(|&p| Probe2D::at(&locator, &w.ops, p))
            .collect();
        let inside = probes.iter().filter(|p| p.is_some()).count();
        let outside = match probes.len() - inside {
            0 => String::new(),
            n => format!(" ({n} outside the waves' mesh)"),
        };
        let line = format!(
            "waves: compared with the parent's H_s at {inside} of its {} points{outside}, in \
             wave_points.csv",
            probes.len()
        );
        let points = Self {
            parent,
            probes,
            east,
            rows: Vec::new(),
            spectrum: Vec::new(),
        };
        (points, line)
    }

    /// Record the waves at their time `t`.
    pub(super) fn record(&mut self, waves: &CoupledWaves2D, t: f64) {
        let (model, state) = (waves.model(), waves.state());
        let nn = model.ops.n_nodes;
        self.spectrum.resize(model.grid.n_components(), 0.0);
        let mut row = Vec::with_capacity(self.probes.len());
        for (p, probe) in self.probes.iter().enumerate() {
            let parent = self.parent.hs_at(p, t);
            let Some(probe) = probe else {
                row.push([f64::NAN, parent, f64::NAN, f64::NAN]);
                continue;
            };
            let start = probe.element().as_usize() * nn;
            let weights = probe.weights();
            let mut hs = 0.0;
            let mut nearest = (f64::NEG_INFINITY, None);
            for (i, &w) in weights.iter().enumerate() {
                model.energy_spectrum_into(state, start + i, &mut self.spectrum);
                let params = model.grid.parameters(&self.spectrum);
                hs += w * params.hs;
                if w > nearest.0 {
                    nearest = (w, Some(params));
                }
            }
            let node = nearest.1.expect("an element has nodes");
            // Travelling to `direction` (rad from the mesh's x axis) is
            // coming from the bearing 270° − (direction − east)
            let from = if node.hs > 0.0 {
                (270.0 - (node.direction - self.east).to_degrees()).rem_euclid(360.0)
            } else {
                f64::NAN
            };
            row.push([hs, parent, node.tp, from]);
        }
        self.rows.push((t, row));
    }

    /// Bias and RMSE of our H_s against the parent's over every point and
    /// record from `after` (s) on, and the parent's mean H_s there.
    pub(super) fn summary(&self, after: f64) -> Option<String> {
        let pairs: Vec<(f64, f64)> = self
            .rows
            .iter()
            .filter(|(t, _)| *t >= after - 1e-6)
            .flat_map(|(_, row)| row.iter().map(|r| (r[0], r[1])))
            .filter(|(ours, parent)| ours.is_finite() && parent.is_finite())
            .collect();
        if pairs.is_empty() {
            return None;
        }
        let n = pairs.len() as f64;
        let bias = pairs.iter().map(|(a, b)| a - b).sum::<f64>() / n;
        let rmse = (pairs.iter().map(|(a, b)| (a - b).powi(2)).sum::<f64>() / n).sqrt();
        let mean = pairs.iter().map(|(_, b)| b).sum::<f64>() / n;
        Some(format!(
            "waves against the parent's H_s from hour {:.1}: parent mean {mean:.2} m, bias \
             {bias:+.3} m, RMSE {rmse:.3} m over {} samples",
            after / 3600.0,
            pairs.len()
        ))
    }

    /// The records as CSV: one row per point and time.
    pub(super) fn write(&self, path: &Path) -> Result<(), CaseError> {
        let mut text = String::from("point,lon,lat,time_s,hs_m,parent_hs_m,tp_s,wave_from_deg\n");
        for (t, row) in &self.rows {
            for (p, [hs, parent, tp, from]) in row.iter().enumerate() {
                let [lon, lat] = self.parent.lon_lat[p];
                text += &format!("{p},{lon},{lat},{t},{hs},{parent},{tp},{from}\n");
            }
        }
        std::fs::write(path, text).map_err(|e| CaseError::Io(format!("{}: {e}", path.display())))
    }
}

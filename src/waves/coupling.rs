//! The waves and the circulation on meshes of their own (TODO F.4 cost).
//!
//! The spectral waves are smooth on the scale of a kilometre and costly per
//! node (hundreds of components), while the circulation needs the coastline
//! at tens of metres. At Frøya a 1 km P1 wave grid costs ≈ 3× the tidal
//! circulation per model hour, the coastline mesh at P2 ≈ 1000×. So the waves
//! run on a coarse mesh of their own and [`WaveCoupling2D`] carries the
//! exchanges across, by interpolation of the nodal fields
//! ([`MeshTransfer2D`]: each target node evaluates the source element's
//! polynomial there):
//!
//! - circulation → waves ([`WaveCoupling2D::update_waves`]): the surface
//!   elevation η and the depth-averaged current. The current is the ratio of
//!   the interpolated transport and depth, `(hu, hv)/h`, as a station samples
//!   it ([`crate::solver::Probe2D`]): a nearly dry node's `hu/h` does not
//!   count as much as a deep one's. On dry nodes η is the bed, which would
//!   pull a wave node at the shore up to the land; there η is the mean of
//!   the element's wet nodes, or 0 (still water) in a dry element. Wave
//!   nodes outside the circulation's mesh (water the circulation does not
//!   model) take the level at the nearest point of it, and no current.
//! - waves → circulation: the radiation stress `S/ρg`, interpolated and then
//!   differentiated on the circulation's mesh ([`WaveCoupling2D::force`]):
//!   the force `−g ∇·(S/ρg)` keeps its flux form there (its total is the
//!   boundary integral of `S` on the circulation's mesh), which a force
//!   interpolated from the coarse mesh would not, and it is exact for the
//!   interpolated stress. Or the force of the dissipation
//!   ([`WaveCoupling2D::dissipation_force`], [`WaveForceForm::Dissipation`]),
//!   interpolated as it is. The waves' bed stress and surface roughness, and the
//!   Stokes drift ([`StokesDriftField::transferred`]), are interpolated as
//!   they are, the first two kept non-negative.
//!
//! On the same mesh and order every transfer is the identity (to round-off),
//! so the coupling reproduces the same-mesh one.
//!
//! [`CoupledWaves2D`] runs the waves alongside a running circulation
//! ([`crate::simulation::Simulation::run_with_exchange`]), exchanging every
//! coupling interval `[t, t + Δ]`, as COAWST's couplers do but in turn:
//!
//! 1. the waves take the circulation's level and current at `t`;
//! 2. the waves step to `t + Δ` (their own step, landing on `t + Δ`);
//! 3. the circulation steps to `t + Δ` under the force linear in time from
//!    the waves' at `t` to theirs at `t + Δ` ([`WaveForce2D::between`]), so
//!    the force is continuous in time and an exchange does not ring the
//!    basin, and on its bed friction enhanced by the mean of the waves' bed
//!    stress at the two ends ([`crate::source::WaveCurrentFriction2D`]).
//!
//! The waves see the circulation lagged by up to Δ (the level and current
//! change over a tide, Δ is minutes); the circulation sees the waves without
//! a lag.

use std::sync::Arc;
use std::time::Instant;

use crate::boundary::{SWEBoundaryCondition2D, tidal_ramp};
use crate::mesh::{Bathymetry2D, Mesh2D};
use crate::operators::{DGOperators2D, GeometricFactors2D, MeshTransfer2D};
use crate::physics::SWEPhysics2D;
use crate::solver::SWESolution2D;
use crate::source::{
    BottomFriction2D, GriddedAtmosphere2D, SourceTerm2D, SourceTerms2D, WaveCurrentFriction2D,
    WaveForce2D,
};

use super::boundary::{BoundarySpectra, WindSeries};
use super::model::{WaveModel2D, WaveWorkspace};
use super::state::WaveSolution;
use super::stokes::StokesDriftField;

/// Default depth (m) below which a circulation node is dry for the coupling.
pub const DEFAULT_COUPLING_H_DRY: f64 = 1e-3;

/// Battjes–Janssen's breaker index γ (the largest wave `H_m = γ d`, SWAN's
/// default) with which [`WaveCoupling2D::depth_limited_force`] caps the
/// radiation stress at the circulation's own depth.
pub const DEFAULT_BREAKER_INDEX: f64 = 0.73;

/// Default bed roughness length z₀ (m) of the waves' bed stress in
/// [`CoupledWaves2D`] ([`WaveModel2D::bed_wave_stress`]): 1 mm, a rippled
/// sand or gravel bed (a Manning n of 0.025 is z₀ ≈ 2 mm in 10 m of water).
pub const DEFAULT_BED_ROUGHNESS: f64 = 1e-3;

/// Which force of the waves a [`CoupledWaves2D`] gives the circulation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WaveForceForm {
    /// The radiation stress's divergence `−∇·S`, capped at the
    /// circulation's depth ([`WaveCoupling2D::depth_limited_force`]): setup,
    /// set-down and the wave-driven currents
    #[default]
    RadiationStress,
    /// The momentum the sinks take, `g Σ (k/σ) e_θ D` (Dingemans et al.
    /// 1987; [`WaveCoupling2D::dissipation_force`]): the wave-driven
    /// currents and the setup where the waves break, without the set-down
    /// where they shoal. Where a coarse wave field shoals and refracts
    /// against a steep coast, `−∇·S` is large and only nearly a gradient,
    /// and what is left over drives currents the dissipation does not.
    Dissipation,
}

/// The exchanges between a wave model and a circulation on different meshes
/// (see the module docs).
#[derive(Clone)]
pub struct WaveCoupling2D {
    /// Circulation nodes → wave nodes
    to_waves: MeshTransfer2D,
    /// Wave nodes → circulation nodes
    to_circulation: MeshTransfer2D,
    mesh: Arc<Mesh2D>,
    ops: Arc<DGOperators2D>,
    geom: Arc<GeometricFactors2D>,
    h_dry: f64,
}

impl WaveCoupling2D {
    /// The coupling of `waves` to a circulation on `mesh` at the order of
    /// `ops`.
    pub fn new(
        waves: &WaveModel2D,
        mesh: Arc<Mesh2D>,
        ops: Arc<DGOperators2D>,
        geom: Arc<GeometricFactors2D>,
    ) -> Self {
        Self {
            to_waves: MeshTransfer2D::new(&mesh, &ops, &waves.mesh, &waves.ops),
            to_circulation: MeshTransfer2D::new(&waves.mesh, &waves.ops, &mesh, &ops),
            mesh,
            ops,
            geom,
            h_dry: DEFAULT_COUPLING_H_DRY,
        }
    }

    /// Circulation nodes shallower than `h_dry` (m) are dry: they pass no
    /// level and no current to the waves.
    pub fn with_h_dry(mut self, h_dry: f64) -> Self {
        assert!(h_dry > 0.0);
        self.h_dry = h_dry;
        self
    }

    /// The transfer from the circulation's nodes onto the waves'.
    pub fn to_waves(&self) -> &MeshTransfer2D {
        &self.to_waves
    }

    /// The transfer from the waves' nodes onto the circulation's.
    pub fn to_circulation(&self) -> &MeshTransfer2D {
        &self.to_circulation
    }

    /// The surface elevation η (m) of the circulation state `q` over its bed
    /// `bathymetry` at the waves' nodes; dry nodes take their element's mean
    /// wet level, or 0 in a dry element.
    pub fn water_level(&self, q: &SWESolution2D, bathymetry: &Bathymetry2D) -> Vec<f64> {
        let nn = self.ops.n_nodes;
        assert_eq!(
            q.n_nodes, nn,
            "the state on the coupling's circulation order"
        );
        let mut eta = vec![0.0; q.h_data().len()];
        for ((eta, h), bed) in eta
            .chunks_exact_mut(nn)
            .zip(q.h_data().chunks_exact(nn))
            .zip(bathymetry.data.chunks_exact(nn))
        {
            let (mut wet, mut sum) = (0usize, 0.0);
            for (e, (&h, &b)) in eta.iter_mut().zip(h.iter().zip(bed)) {
                *e = h + b;
                if h > self.h_dry {
                    wet += 1;
                    sum += *e;
                }
            }
            let fill = if wet > 0 { sum / wet as f64 } else { 0.0 };
            for (e, &h) in eta.iter_mut().zip(h) {
                if h <= self.h_dry {
                    *e = fill;
                }
            }
        }
        self.to_waves.apply(&eta)
    }

    /// The depth-averaged current `(u, v)` (m/s, mesh axes) of `q` at the
    /// waves' nodes: the interpolated transport over the interpolated depth,
    /// 0 where that depth is below `h_dry` and at wave nodes outside the
    /// circulation's mesh (the circulation knows nothing of the flow there).
    ///
    /// The speed is at most the largest of the source element's wet nodes
    /// (`h > h_dry`). Near a shore the depth's polynomial can dip towards
    /// `h_dry` between nodes where the transport's does not, and the ratio
    /// blew up: 75 m/s at a wave node at Frøya, which set the waves' step.
    /// Where both are affine the ratio is within its nodal values anyway.
    pub fn currents(&self, q: &SWESolution2D) -> (Vec<f64>, Vec<f64>) {
        let t = &self.to_waves;
        let nn = self.ops.n_nodes;
        let n = t.n_target_points();
        let (mut u, mut v) = (vec![0.0; n], vec![0.0; n]);
        let (hs, hus, hvs) = (q.h_data(), q.hu_data(), q.hv_data());
        for p in 0..n {
            let h = t.value_at(p, hs);
            if h <= self.h_dry || t.is_outside(p) {
                continue;
            }
            let (up, vp) = (t.value_at(p, hus) / h, t.value_at(p, hvs) / h);
            let base = t.source_element(p) * nn;
            let fastest = (base..base + nn)
                .filter(|&i| hs[i] > self.h_dry)
                .map(|i| hus[i].hypot(hvs[i]) / hs[i])
                .fold(0.0, f64::max);
            let speed = up.hypot(vp);
            let scale = if speed > fastest {
                fastest / speed
            } else {
                1.0
            };
            (u[p], v[p]) = (scale * up, scale * vp);
        }
        (u, v)
    }

    /// Give `waves` the level and the current of the circulation state `q`
    /// over its bed `bathymetry` ([`WaveModel2D::set_water_level`],
    /// [`WaveModel2D::set_currents`]).
    pub fn update_waves(
        &self,
        waves: &mut WaveModel2D,
        q: &SWESolution2D,
        bathymetry: &Bathymetry2D,
    ) {
        waves.set_water_level(&self.water_level(q, bathymetry));
        let (u, v) = self.currents(q);
        waves.set_currents(&u, &v);
    }

    /// The radiation stress per ρg `[S_xx, S_xy, S_yy]` (m²) of the wave state
    /// `n` at the circulation's nodes.
    pub fn radiation_stress(&self, waves: &WaveModel2D, n: &WaveSolution) -> Vec<[f64; 3]> {
        self.to_circulation
            .apply_components(&waves.radiation_stress(n))
    }

    /// The force of the wave state `n` on the circulation: its radiation
    /// stress interpolated onto the circulation's nodes and differentiated
    /// there ([`WaveForce2D::from_radiation_stress`]).
    pub fn force(&self, waves: &WaveModel2D, n: &WaveSolution) -> WaveForce2D {
        WaveForce2D::from_radiation_stress(
            &self.mesh,
            &self.ops,
            &self.geom,
            &self.radiation_stress(waves, n),
            waves.g(),
        )
    }

    /// [`Self::force`] with the stress at each circulation node capped at
    /// what a depth-limited wave on that node's own depth `h` (of the
    /// circulation state `q`) carries: `tr S/ρg ≤ γ² h²/4`, the trace of a
    /// shallow-water wave's `S = E (3/2, 1/2)` with `E = H²/8` and
    /// `H = γ h` (`breaker_index` γ, Battjes–Janssen's `H_m`).
    ///
    /// Interpolated from a coarser wave mesh, the stress belongs to the
    /// waves' depth there, and on a reef or a shore the circulation resolves
    /// (0.3 m where the wave node has metres) it pushed a film of water no
    /// such wave could stand in: in a storm at Frøya, 4.3 m/s and 136
    /// negative-depth clips in a day. Where the waves themselves are depth
    /// limited on the circulation's depth the cap does nothing.
    pub fn depth_limited_force(
        &self,
        waves: &WaveModel2D,
        n: &WaveSolution,
        q: &SWESolution2D,
        breaker_index: f64,
    ) -> WaveForce2D {
        let mut stress = self.radiation_stress(waves, n);
        let cap = 0.25 * breaker_index * breaker_index;
        for (s, &h) in stress.iter_mut().zip(q.h_data()) {
            let trace = s[0] + s[2];
            let largest = cap * h.max(0.0) * h.max(0.0);
            if trace > largest {
                let scale = largest / trace;
                s.iter_mut().for_each(|x| *x *= scale);
            }
        }
        WaveForce2D::from_radiation_stress(&self.mesh, &self.ops, &self.geom, &stress, waves.g())
    }

    /// The force of the wave state `n`'s dissipation on the circulation
    /// ([`WaveForce2D::from_dissipation`]): `g Σ (k/σ) e_θ D` per wave node,
    /// interpolated onto the circulation's nodes. No set-down, and none of
    /// the unresolved shoaling's `−∇·S` (see [`WaveForceForm::Dissipation`]).
    /// `D` is what the sinks remove over a wave step `dt` (s)
    /// ([`WaveModel2D::dissipation`]).
    pub fn dissipation_force(&self, waves: &WaveModel2D, n: &WaveSolution, dt: f64) -> WaveForce2D {
        let force: Vec<[f64; 2]> = waves.dissipation(n, dt).iter().map(|d| d.force).collect();
        WaveForce2D::from_force(
            &self.mesh,
            &self.ops,
            self.to_circulation.apply_components(&force),
        )
    }

    /// The waves' bed stress per ρ (m²/s²) on a bed of roughness length `z0`
    /// (m) at the circulation's nodes ([`WaveModel2D::bed_wave_stress`]), for
    /// [`crate::source::WaveCurrentFriction2D`].
    pub fn bed_wave_stress(&self, waves: &WaveModel2D, n: &WaveSolution, z0: f64) -> Vec<f64> {
        self.non_negative(&waves.bed_wave_stress(n, z0))
    }

    /// The surface roughness `alpha · H_s` (m) at the circulation's nodes
    /// ([`WaveModel2D::surface_roughness`]), for
    /// [`crate::physics::Hydrostatic3D::set_surface_roughness`].
    pub fn surface_roughness(&self, waves: &WaveModel2D, n: &WaveSolution, alpha: f64) -> Vec<f64> {
        self.non_negative(&waves.surface_roughness(n, alpha))
    }

    /// The Stokes drift of the wave state `n` as a field on the circulation's
    /// mesh, for the particle trackers.
    pub fn stokes_drift(&self, waves: &WaveModel2D, n: &WaveSolution) -> StokesDriftField {
        StokesDriftField::new(waves, n).transferred(&self.to_circulation, self.ops.n_nodes)
    }

    fn non_negative(&self, field: &[f64]) -> Vec<f64> {
        let mut out = self.to_circulation.apply(field);
        out.iter_mut().for_each(|x| *x = x.max(0.0));
        out
    }
}

/// A wave model running alongside a 2D circulation and exchanging with it
/// every coupling interval (see the module docs): the level and the current
/// to the waves, the radiation-stress force and the wave-enhanced bed
/// friction back.
///
/// ```ignore
/// let mut waves = CoupledWaves2D::new(model, state, &physics, 0.0).with_ramp(3600.0);
/// let mut sim = Simulation::new(physics, SSPRK3);
/// sim.run_with_exchange(
///     &mut q, 0.0, t_end, 600.0,
///     |physics, q, t, t_next| waves.exchange(physics, q, t, t_next),
///     |q, t| { /* output */ },
/// );
/// ```
pub struct CoupledWaves2D {
    model: WaveModel2D,
    state: WaveSolution,
    coupling: WaveCoupling2D,
    workspace: WaveWorkspace,
    /// The circulation's bed, for its level
    bathymetry: Arc<Bathymetry2D>,
    /// The circulation's own source terms and bed friction, which the waves'
    /// force is added to and their bed stress enhances
    sources: Option<Arc<dyn SourceTerm2D>>,
    friction: Option<Arc<dyn BottomFriction2D>>,
    /// The waves' time
    time: f64,
    /// The force and bed stress of the wave state at `time` (none before the
    /// first exchange)
    last: Option<(WaveForce2D, Vec<f64>)>,
    cfl: f64,
    z0: f64,
    ramp: Option<f64>,
    force_form: WaveForceForm,
    stats: CoupledWavesStats,
    /// A parent model's spectra on the open boundary and its wind, set at
    /// every wave step's midpoint
    boundary: Option<BoundarySpectra>,
    boundary_buffer: Vec<f64>,
    wind: Option<WaveWind>,
}

/// Where the waves of a [`CoupledWaves2D`] take their wind from.
enum WaveWind {
    /// A parent model's, the same everywhere
    Series(WindSeries),
    /// A weather model's per wave node, and the nodes' winds
    Gridded(GriddedAtmosphere2D, Vec<[f64; 2]>),
}

/// Work counts of a [`CoupledWaves2D`].
#[derive(Clone, Copy, Debug, Default)]
pub struct CoupledWavesStats {
    /// Exchanges (coupling intervals) so far.
    pub exchanges: usize,
    /// Wave steps so far.
    pub wave_steps: usize,
    /// Wall time (s) of the wave steps.
    pub stepping_time: f64,
    /// Wall time (s) of the exchanges: the transfers, the force and the bed
    /// stress.
    pub exchange_time: f64,
}

impl CoupledWaves2D {
    /// The waves `model` in the state `state` at time `t` (s), coupled to the
    /// circulation `physics`: its mesh, order and bed, and its source terms
    /// and bed friction as they are now (each exchange adds the waves' to
    /// these).
    ///
    /// # Panics
    /// If `physics` has no bathymetry.
    pub fn new<BC: SWEBoundaryCondition2D>(
        model: WaveModel2D,
        state: WaveSolution,
        physics: &SWEPhysics2D<BC>,
        t: f64,
    ) -> Self {
        let bathymetry = physics
            .bathymetry
            .clone()
            .expect("a coupled circulation needs its bathymetry");
        let coupling = WaveCoupling2D::new(
            &model,
            physics.mesh.clone(),
            physics.ops.clone(),
            physics.geom.clone(),
        );
        Self {
            model,
            state,
            coupling,
            workspace: WaveWorkspace::default(),
            bathymetry,
            sources: physics.source.clone(),
            friction: physics.friction.clone(),
            time: t,
            last: None,
            cfl: 0.5,
            z0: DEFAULT_BED_ROUGHNESS,
            ramp: None,
            force_form: WaveForceForm::default(),
            stats: CoupledWavesStats::default(),
            boundary: None,
            boundary_buffer: Vec::new(),
            wind: None,
        }
    }

    /// The waves' CFL number ([`WaveModel2D::compute_dt`]; default 0.5,
    /// which SWAN's frozen-rate sources need; 1.0 with WAM's integration,
    /// [`super::SourceIntegration::Implicit`] and [`super::GrowthLimiter::Rate`]).
    pub fn with_cfl(mut self, cfl: f64) -> Self {
        assert!(cfl > 0.0);
        self.cfl = cfl;
        self
    }

    /// The bed roughness length z₀ (m) of the waves' bed stress (default
    /// [`DEFAULT_BED_ROUGHNESS`]).
    pub fn with_bed_roughness(mut self, z0: f64) -> Self {
        assert!(z0 > 0.0);
        self.z0 = z0;
        self
    }

    /// Ramp the force and the bed stress up from 0 at t = 0 to full at
    /// `seconds` (the tidal ramp's `3τ² − 2τ³`), for waves started at once
    /// over a circulation at rest.
    pub fn with_ramp(mut self, seconds: f64) -> Self {
        self.ramp = Some(seconds);
        self
    }

    /// Which force the waves give the circulation (default
    /// [`WaveForceForm::RadiationStress`]).
    pub fn with_force_form(mut self, form: WaveForceForm) -> Self {
        self.force_form = form;
        self
    }

    /// The wave model (its level and current are the last exchange's).
    pub fn model(&self) -> &WaveModel2D {
        &self.model
    }

    /// The wave state at [`Self::time`].
    pub fn state(&self) -> &WaveSolution {
        &self.state
    }

    /// The transfers between the meshes.
    pub fn coupling(&self) -> &WaveCoupling2D {
        &self.coupling
    }

    /// The waves' CFL number.
    pub fn cfl(&self) -> f64 {
        self.cfl
    }

    /// The waves' time (s).
    pub fn time(&self) -> f64 {
        self.time
    }

    /// The force of the wave state at [`Self::time`] on the circulation,
    /// unramped (per circulation node, in the [`WaveForceForm`] chosen; none
    /// before the first exchange).
    pub fn force(&self) -> Option<&[[f64; 2]]> {
        self.last.as_ref().map(|(force, _)| force.force())
    }

    /// Take the open boundary's spectra from a parent wave model
    /// ([`BoundarySpectra::for_model`] on this model), at every wave step.
    pub fn with_boundary(mut self, boundary: BoundarySpectra) -> Self {
        assert_eq!(
            boundary.n_targets(),
            self.model.open_boundary_points().len(),
            "boundary spectra for another model"
        );
        self.boundary_buffer = vec![0.0; boundary.n_targets() * self.model.grid.n_components()];
        self.boundary = Some(boundary);
        self
    }

    /// Take the wind from a parent model's series, at every wave step.
    pub fn with_wind(mut self, wind: WindSeries) -> Self {
        self.wind = Some(WaveWind::Series(wind));
        self
    }

    /// Take the wind per node from a weather model sampled on the wave
    /// model's mesh (e.g. the circulation's
    /// [`GriddedAtmosphere2D::on_mesh`] of the wave mesh), at every wave
    /// step: its 10 m wind, not ramped ([`GriddedAtmosphere2D::wind_into`]).
    ///
    /// # Panics
    /// If `atmosphere` is sampled on another mesh than the wave model's.
    pub fn with_gridded_wind(mut self, atmosphere: GriddedAtmosphere2D) -> Self {
        let n = self.model.n_points();
        assert_eq!(atmosphere.n_points(), n, "a weather model on another mesh");
        self.wind = Some(WaveWind::Gridded(atmosphere, vec![[0.0; 2]; n]));
        self
    }

    /// The parent's spectra on the open boundary, if any.
    pub fn boundary(&self) -> Option<&BoundarySpectra> {
        self.boundary.as_ref()
    }

    /// Work counts so far.
    pub fn stats(&self) -> CoupledWavesStats {
        self.stats
    }

    /// One coupling interval from `t` (the waves' time) to `t_next`: the
    /// waves take the level and current of the circulation state `q` and step
    /// to `t_next`; then `physics` gets its own source terms plus the force
    /// linear in time between the waves' at `t` and at `t_next`, and its own
    /// bed friction enhanced by the mean of their bed stresses (see the
    /// module docs). The exchange of
    /// [`crate::simulation::Simulation::run_with_exchange`].
    ///
    /// # Panics
    /// If `t` is not the waves' time, or `t_next ≤ t`.
    pub fn exchange<BC: SWEBoundaryCondition2D>(
        &mut self,
        physics: &mut SWEPhysics2D<BC>,
        q: &SWESolution2D,
        t: f64,
        t_next: f64,
    ) {
        assert!(
            (t - self.time).abs() <= 1e-9 * t.abs().max(1.0),
            "the waves are at {} s, the circulation at {t} s",
            self.time
        );
        assert!(t_next > t, "coupling interval {t} → {t_next}");
        let start = Instant::now();
        let (f0, b0) = match self.last.take() {
            Some(last) => last,
            // No step taken yet: over the step the waves would take now
            None => self.forcing(q, self.model.compute_dt(self.cfl)),
        };
        self.coupling
            .update_waves(&mut self.model, q, &self.bathymetry);
        let exchanged = start.elapsed().as_secs_f64();

        // The waves' own steps, landing on t_next
        let start = Instant::now();
        let span = t_next - t;
        let steps = (span / self.model.compute_dt(self.cfl) * (1.0 - 1e-9))
            .ceil()
            .max(1.0) as usize;
        let dt = span / steps as f64;
        for s in 0..steps {
            let start = t + s as f64 * dt;
            // The parent's boundary and wind at the step's midpoint
            let middle = start + 0.5 * dt;
            if let Some(boundary) = &self.boundary {
                boundary.at_into(middle, &mut self.boundary_buffer);
                self.model.set_boundary_spectra(&self.boundary_buffer);
            }
            match &mut self.wind {
                Some(WaveWind::Series(series)) => self.model.set_wind(series.at(middle)),
                Some(WaveWind::Gridded(atmosphere, winds)) => {
                    atmosphere.wind_into(middle, winds);
                    self.model.set_winds(winds);
                }
                None => {}
            }
            self.model.step(&mut self.state, dt, &mut self.workspace);
        }
        self.time = t_next;
        let stepping = start.elapsed().as_secs_f64();

        let start = Instant::now();
        let (f1, b1) = self.forcing(q, dt);
        let force: Arc<dyn SourceTerm2D> = Arc::new(WaveForce2D::between(t, f0, t_next, &f1));
        physics.source = Some(match &self.sources {
            None => force,
            Some(own) => Arc::new(SourceTerms2D::new(vec![own.clone(), force])),
        });
        if let Some(inner) = &self.friction {
            let ramp = tidal_ramp(0.5 * (t + t_next), self.ramp);
            let stress = b0
                .iter()
                .zip(&b1)
                .map(|(a, b)| 0.5 * ramp * (a + b))
                .collect();
            physics.friction = Some(Arc::new(WaveCurrentFriction2D::new(inner.clone(), stress)));
        }
        self.last = Some((f1, b1));

        self.stats.exchanges += 1;
        self.stats.wave_steps += steps;
        self.stats.stepping_time += stepping;
        self.stats.exchange_time += exchanged + start.elapsed().as_secs_f64();
    }

    /// The force (ramped) and the bed stress (unramped; empty without a bed
    /// friction to enhance) of the wave state on the circulation; the
    /// dissipation's force is over the waves' step `dt`.
    fn forcing(&self, q: &SWESolution2D, dt: f64) -> (WaveForce2D, Vec<f64>) {
        let mut force = match self.force_form {
            WaveForceForm::RadiationStress => self.coupling.depth_limited_force(
                &self.model,
                &self.state,
                q,
                DEFAULT_BREAKER_INDEX,
            ),
            // Over the waves' own step: what their sinks can remove in it
            WaveForceForm::Dissipation => {
                self.coupling
                    .dissipation_force(&self.model, &self.state, dt)
            }
        };
        if let Some(seconds) = self.ramp {
            force = force.with_ramp(seconds);
        }
        let stress = match self.friction {
            Some(_) => self
                .coupling
                .bed_wave_stress(&self.model, &self.state, self.z0),
            None => Vec::new(),
        };
        (force, stress)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::SWEState2D;
    use crate::types::ElementIndex;
    use crate::waves::SpectralGrid;

    const G: f64 = 9.81;

    struct Circulation {
        mesh: Arc<Mesh2D>,
        ops: Arc<DGOperators2D>,
        geom: Arc<GeometricFactors2D>,
    }

    fn circulation(nx: usize, ny: usize, order: usize) -> Circulation {
        let mut mesh = Mesh2D::uniform_rectangle(0.0, 600.0, 0.0, 400.0, nx, ny);
        for v in &mut mesh.vertices {
            v[0] += 0.1 * v[1];
        }
        let ops = DGOperators2D::new(order);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        Circulation {
            mesh: Arc::new(mesh),
            ops: Arc::new(ops),
            geom: Arc::new(geom),
        }
    }

    fn waves_on(c: &Circulation, bed: impl Fn(f64, f64) -> f64) -> WaveModel2D {
        let bathymetry = Bathymetry2D::from_function(&c.mesh, &c.ops, &c.geom, bed);
        WaveModel2D::new(
            c.mesh.clone(),
            c.ops.clone(),
            c.geom.clone(),
            &bathymetry,
            SpectralGrid::new(0.08, 0.3, 8, 12),
            G,
        )
    }

    fn positions(mesh: &Mesh2D, ops: &DGOperators2D) -> Vec<[f64; 2]> {
        ElementIndex::iter(mesh.n_elements)
            .flat_map(|k| {
                (0..ops.n_nodes)
                    .map(move |i| mesh.reference_to_physical(k, ops.nodes_r[i], ops.nodes_s[i]))
            })
            .collect()
    }

    /// A sea that varies over the domain, so every exchange has something
    /// to carry.
    fn varied_sea(waves: &WaveModel2D) -> WaveSolution {
        let mut n = waves.uniform_state(&waves.grid.jonswap(1.2, 6.0, 3.3, 0.4, 6.0));
        let xy = positions(&waves.mesh, &waves.ops);
        for (p, [px, py]) in xy.iter().enumerate() {
            let factor = 1.0 + 0.5 * (px / 300.0).sin() * (py / 250.0).cos();
            n.spectrum_mut(p).iter_mut().for_each(|x| *x *= factor);
        }
        n
    }

    /// On the same mesh every exchange is the same-mesh one: the forces of
    /// `WaveForce2D::new` and `from_dissipation`, the stresses, the roughness and the Stokes drift of
    /// the wave model itself, to round-off.
    #[test]
    fn on_the_same_mesh_the_coupling_is_the_direct_one() {
        let c = circulation(5, 4, 2);
        let waves = waves_on(&c, |x, _| -(4.0 + 0.02 * x));
        let n = varied_sea(&waves);
        let coupling = WaveCoupling2D::new(&waves, c.mesh.clone(), c.ops.clone(), c.geom.clone());
        let close = |a: f64, b: f64, scale: f64| (a - b).abs() <= 1e-12 * scale;

        let direct = WaveForce2D::new(&waves, &n);
        let scale = direct
            .force()
            .iter()
            .map(|f| f[0].hypot(f[1]))
            .fold(0.0, f64::max);
        assert!(scale > 0.0);
        for (a, b) in coupling
            .force(&waves, &n)
            .force()
            .iter()
            .zip(direct.force())
        {
            assert!(
                close(a[0], b[0], scale) && close(a[1], b[1], scale),
                "{a:?} {b:?}"
            );
        }
        // The dissipation's force, with sinks to give one
        let sinks = waves_on(&c, |x, _| -(4.0 + 0.02 * x))
            .with_sources(crate::waves::SourceTerms::swan_defaults(G));
        let direct = WaveForce2D::from_dissipation(&sinks, &n, 30.0);
        let scale = direct
            .force()
            .iter()
            .map(|f| f[0].hypot(f[1]))
            .fold(0.0, f64::max);
        assert!(scale > 0.0);
        for (a, b) in coupling
            .dissipation_force(&sinks, &n, 30.0)
            .force()
            .iter()
            .zip(direct.force())
        {
            assert!(
                close(a[0], b[0], scale) && close(a[1], b[1], scale),
                "{a:?} {b:?}"
            );
        }
        for (a, b) in [
            (
                coupling.bed_wave_stress(&waves, &n, 1e-3),
                waves.bed_wave_stress(&n, 1e-3),
            ),
            (
                coupling.surface_roughness(&waves, &n, 0.6),
                waves.surface_roughness(&n, 0.6),
            ),
        ] {
            let scale = b.iter().cloned().fold(0.0, f64::max);
            assert!(scale > 0.0);
            assert!(a.iter().zip(&b).all(|(a, b)| close(*a, *b, scale)));
        }
        let field = coupling.stokes_drift(&waves, &n);
        let nn = c.ops.n_nodes;
        let mut w = vec![0.0; nn];
        let expected = waves.stokes_drift(&n, -0.5);
        for (p, e) in expected.iter().enumerate() {
            w.fill(0.0);
            w[p % nn] = 1.0;
            let got = field.at(ElementIndex::new(p / nn), &w, 0.5);
            let scale = e[0].hypot(e[1]);
            assert!(close(got[0], e[0], scale) && close(got[1], e[1], scale));
        }
    }

    /// From a P2 circulation onto a coarser P1 wave mesh: a linear level and a
    /// linear transport over a uniform depth arrive exactly; dry nodes pass
    /// their element's wet level, or 0 in a dry element, and no current.
    #[test]
    fn the_waves_get_the_level_and_the_current() {
        let c = circulation(6, 4, 2);
        let fine_bed = |x: f64, _: f64| if x > 450.0 { 2.0 } else { -10.0 };
        let bathymetry = Bathymetry2D::from_function(&c.mesh, &c.ops, &c.geom, fine_bed);
        let coarse = circulation(3, 2, 1);
        let waves = waves_on(&coarse, |_, _| -10.0);
        let coupling = WaveCoupling2D::new(&waves, c.mesh.clone(), c.ops.clone(), c.geom.clone());

        // Wet everywhere: η linear, a uniform depth of 10 m, transport linear
        let level = |x: f64, y: f64| 0.1 + 1e-4 * x - 2e-4 * y;
        let transport = |x: f64, y: f64| [0.5 + 1e-3 * x, -0.2 + 5e-4 * y];
        let flat = Bathymetry2D::from_function(&c.mesh, &c.ops, &c.geom, |_, _| -10.0);
        let mut q = SWESolution2D::new(c.mesh.n_elements, c.ops.n_nodes);
        let xy = positions(&c.mesh, &c.ops);
        for (p, &[x, y]) in xy.iter().enumerate() {
            let [hu, hv] = transport(x, y);
            let k = ElementIndex::new(p / c.ops.n_nodes);
            q.set_state(
                k,
                p % c.ops.n_nodes,
                SWEState2D::new(10.0 + level(x, y), hu, hv),
            );
        }
        // The depth is 10 + η, not uniform: compare the transport and depth
        let eta = coupling.water_level(&q, &flat);
        let (u, v) = coupling.currents(&q);
        for (p, &[x, y]) in positions(&waves.mesh, &waves.ops).iter().enumerate() {
            let h = 10.0 + level(x, y);
            let [hu, hv] = transport(x, y);
            assert!((eta[p] - level(x, y)).abs() < 1e-12);
            assert!((u[p] * h - hu).abs() < 1e-12 && (v[p] * h - hv).abs() < 1e-12);
        }

        // Over the land x > 450 m: still water 0.3 m up, dry above the bed
        for (p, &[x, _]) in xy.iter().enumerate() {
            let k = ElementIndex::new(p / c.ops.n_nodes);
            let state = if fine_bed(x, 0.0) < 0.0 {
                SWEState2D::new(10.3, 1.0, 0.0)
            } else {
                SWEState2D::new(0.0, 0.0, 0.0)
            };
            q.set_state(k, p % c.ops.n_nodes, state);
        }
        let eta = coupling.water_level(&q, &bathymetry);
        let (u, _) = coupling.currents(&q);
        for (p, &[x, y]) in positions(&waves.mesh, &waves.ops).iter().enumerate() {
            // The wave nodes at x = 600 m (sheared) are in fine elements whose
            // nodes are all on land; the rest in wet or partly wet ones
            let expected = if x - 0.1 * y > 550.0 { 0.0 } else { 0.3 };
            assert!((eta[p] - expected).abs() < 1e-12, "x = {x}: {}", eta[p]);
            assert!(u[p].is_finite() && u[p].abs() <= 0.1 + 1e-12);
        }

        // Waves beyond the circulation's mesh (the coarse grid moved 150 m
        // west): the level at its nearest point, no current
        let mut wide = coarse;
        let mut moved = (*wide.mesh).clone();
        moved.vertices.iter_mut().for_each(|v| v[0] -= 150.0);
        wide.geom = Arc::new(GeometricFactors2D::compute(&moved, &wide.ops));
        wide.mesh = Arc::new(moved);
        let waves = waves_on(&wide, |_, _| -10.0);
        let coupling = WaveCoupling2D::new(&waves, c.mesh.clone(), c.ops.clone(), c.geom.clone());
        let (eta, (u, _)) = (coupling.water_level(&q, &bathymetry), coupling.currents(&q));
        let mut outside = 0;
        for (p, &[x, y]) in positions(&waves.mesh, &waves.ops).iter().enumerate() {
            if x - 0.1 * y < 0.0 {
                outside += 1;
                assert!(coupling.to_waves().is_outside(p));
                assert_eq!(u[p], 0.0);
                assert!((eta[p] - 0.3).abs() < 1e-12);
            } else if x - 0.1 * y < 400.0 {
                assert!((u[p] - 1.0 / 10.3).abs() < 1e-12, "x = {x}: {}", u[p]);
            }
        }
        assert!(outside > 0);
    }

    /// The depth-limited force is the plain one where the circulation is deep
    /// enough for the waves, and on a film it holds at most the stress of a
    /// wave `γ h` high: `tr S/ρg ≤ γ² h²/4` at every node.
    #[test]
    fn the_force_is_capped_by_the_circulations_own_depth() {
        let c = circulation(5, 4, 2);
        let waves = waves_on(&c, |x, _| -(4.0 + 0.02 * x));
        let n = varied_sea(&waves);
        let coupling = WaveCoupling2D::new(&waves, c.mesh.clone(), c.ops.clone(), c.geom.clone());
        let nn = c.ops.n_nodes;
        let mut q = SWESolution2D::new(c.mesh.n_elements, nn);
        let set = |q: &mut SWESolution2D, h: &dyn Fn(usize) -> f64| {
            for p in 0..c.mesh.n_elements * nn {
                q.set_state(
                    ElementIndex::new(p / nn),
                    p % nn,
                    SWEState2D::new(h(p), 0.0, 0.0),
                );
            }
        };
        // Deep: unchanged
        set(&mut q, &|_| 10.0);
        let plain = coupling.force(&waves, &n);
        let limited = coupling.depth_limited_force(&waves, &n, &q, DEFAULT_BREAKER_INDEX);
        assert_eq!(plain.force(), limited.force());
        // A film of 5 cm on every node: the stress is capped, so is its force
        set(&mut q, &|_| 0.05);
        let film = coupling.depth_limited_force(&waves, &n, &q, DEFAULT_BREAKER_INDEX);
        let largest = |f: &WaveForce2D| {
            f.force()
                .iter()
                .map(|f| f[0].hypot(f[1]))
                .fold(0.0, f64::max)
        };
        let stress = coupling.radiation_stress(&waves, &n);
        let cap = 0.25 * DEFAULT_BREAKER_INDEX.powi(2) * 0.05f64.powi(2);
        assert!(
            stress.iter().all(|s| s[0] + s[2] > cap),
            "the sea must exceed the cap"
        );
        // A uniform cap scales every node's stress to the same trace
        assert!(
            largest(&film) < 0.02 * largest(&plain),
            "{} {}",
            largest(&film),
            largest(&plain)
        );
    }

    /// Regression (Frøya, 2026-10-07): a P2 element wet at its sides and dry
    /// in its middle (h 1, 5e-4, 1 m across it; the dry nodes carry no
    /// transport, as the circulation's desingularization leaves them). Between
    /// the nodes the depth's parabola dips to a few millimetres where the
    /// transport's does not, and its ratio there reached 2.4 m/s at the wave
    /// nodes, against nodal currents of at most 0.5 m/s. The current passed
    /// to the waves is at most the element's fastest wet node.
    #[test]
    fn a_dip_in_the_depth_between_nodes_gives_no_spurious_current() {
        let mesh = Mesh2D::uniform_rectangle(0.0, 600.0, 0.0, 200.0, 3, 1);
        let ops = DGOperators2D::new(2);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let c = Circulation {
            mesh: Arc::new(mesh),
            ops: Arc::new(ops),
            geom: Arc::new(geom),
        };
        let nn = c.ops.n_nodes;
        let mut q = SWESolution2D::new(c.mesh.n_elements, nn);
        for k in ElementIndex::iter(c.mesh.n_elements) {
            for i in 0..nn {
                let r = c.ops.nodes_r[i];
                let (h, hu) = if r < -0.5 {
                    (1.0, 0.5)
                } else if r > 0.5 {
                    (1.0, 0.3)
                } else {
                    (5e-4, 0.0)
                };
                q.set_state(k, i, SWEState2D::new(h, hu, 0.0));
            }
        }
        let fine = circulation(97, 3, 1);
        let mut moved = (*fine.mesh).clone();
        moved.vertices.iter_mut().for_each(|v| v[0] -= 0.1 * v[1]);
        let fine = Circulation {
            geom: Arc::new(GeometricFactors2D::compute(&moved, &fine.ops)),
            mesh: Arc::new(moved),
            ops: fine.ops,
        };
        let waves = waves_on(&fine, |_, _| -1.0);
        let coupling = WaveCoupling2D::new(&waves, c.mesh.clone(), c.ops.clone(), c.geom.clone());
        let (u, v) = coupling.currents(&q);
        let t = coupling.to_waves();
        let mut ratio: f64 = 0.0;
        for p in 0..u.len() {
            let h = t.value_at(p, q.h_data());
            if h > DEFAULT_COUPLING_H_DRY {
                ratio = ratio.max((t.value_at(p, q.hu_data()) / h).abs());
            }
        }
        let fastest = u
            .iter()
            .zip(&v)
            .map(|(u, v)| u.hypot(*v))
            .fold(0.0, f64::max);
        println!("largest current to the waves {fastest:.3} m/s; the ratio alone {ratio:.2} m/s");
        assert!(ratio > 1.0, "the dip does not bite: {ratio}");
        assert!(fastest <= 0.5 + 1e-12, "{fastest} m/s");
        assert!(u.iter().all(|u| u.is_finite()));
    }
}

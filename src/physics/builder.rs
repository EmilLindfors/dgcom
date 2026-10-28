//! Physics module builders.
//!
//! This module provides builder patterns for constructing physics modules.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use crate::boundary::SWEBoundaryCondition2D;
use crate::equations::ShallowWater2D;
use crate::flux::StandardFlux2D;
use crate::io::{DriverData, Restart2D, RestartError, domain_fingerprint_2d};
use crate::mesh::{Bathymetry2D, Mesh2D};
use crate::operators::{DGOperators2D, GeometricFactors2D};
use crate::solver::core::disjoint::DisjointChunks;
use crate::solver::{
    ImplicitDamping2D, Limiter2D, LimiterContext2D, PositivityBound, SWEFormulation2D,
    SWESolution2D, StandardLimiter2D, WetDryConfig, positivity_cfl_swe_2d,
};
#[cfg(not(feature = "parallel"))]
use crate::solver::{
    apply_implicit_damping_2d as implicit_damping,
    apply_wet_dry_correction_all as wet_dry_correction,
};
#[cfg(feature = "parallel")]
use crate::solver::{
    apply_implicit_damping_2d_parallel as implicit_damping,
    apply_wet_dry_correction_all_parallel as wet_dry_correction,
};
use crate::solver::{
    apply_wet_dry_correction_element, compute_rhs_swe_2d_subset_then, element_dt_swe_2d,
    element_dt_viscous_swe_2d, largest_viscosity_swe_2d, min_element_dt_swe_2d,
};
use crate::source::{
    BottomFriction2D, CageDrag2D, HorizontalViscosity2D, SourceTerm2D, SourceTerms2D,
};
use crate::time::multirate::{ElementStage, RhsStencil};
use crate::time::{LocalTimeStepping, SspScheme};
use crate::types::ElementIndex;

use super::traits::{PhysicsModule, PhysicsModuleInfo};

// =============================================================================
// SWE Physics 2D
// =============================================================================

/// 2D Shallow Water Equations physics module.
///
/// This module encapsulates all components needed to simulate the 2D SWE:
/// - Mesh and geometric data
/// - DG operators
/// - Numerical flux
/// - Boundary conditions
/// - Source terms
/// - Limiters
/// - Wetting/drying, point-implicit bottom friction and net-cage drag
///
/// # Wetting and drying
///
/// A run has wetting/drying when it has a positivity limiter
/// (`StandardLimiter2D::Positivity`/`KuzminWithPositivity`) or a
/// [`WetDryConfig`] (`with_wet_dry`). Then:
/// - the formulation defaults to `SWEFormulation2D::WetDry`, and the interface
///   flux of `Standard` to HLL (Roe is not positivity preserving);
/// - `max_cfl` reports `positivity_cfl_swe_2d(N)`, and the time step
///   (`compute_dt_ssp`, local time stepping) enforces it, relaxed per element
///   where the water is not at risk of running out ([`PositivityBound`]);
/// - after every RK stage, depths are limited to h ≥ 0 and near-dry velocities
///   desingularized ([`crate::solver::apply_wet_dry_correction_all`]);
/// - in every RK stage, bottom friction (`with_implicit_friction`), net-cage
///   drag (`with_cage_drag`) and the thin-layer relaxation are applied
///   point-implicitly
///   ([`crate::solver::apply_implicit_damping_2d`]).
///
/// Elements whose mean depth went negative anyway are emptied (creating mass)
/// and counted in [`SWEPhysics2D::negative_depth_clips`].
///
/// `SWEFormulation2D::WetDry` is exactly well-balanced at shorelines and
/// more accurate on moving shorelines than `Standard` with hydrostatic
/// reconstruction; its dry-node threshold is `WetDryConfig::h_dry`. It
/// includes the bed slope, so the source terms must not contain
/// `BathymetrySource2D`. `EntropyStable`/`EntropyConservative` have no wet/dry
/// interface treatment.
pub struct SWEPhysics2D<BC: SWEBoundaryCondition2D> {
    /// The mesh
    pub mesh: Arc<Mesh2D>,
    /// DG operators
    pub ops: Arc<DGOperators2D>,
    /// Geometric factors
    pub geom: Arc<GeometricFactors2D>,
    /// Shallow water equation parameters
    pub equation: ShallowWater2D,
    /// Numerical flux
    pub flux: StandardFlux2D,
    /// Boundary condition
    pub bc: BC,
    /// Optional source terms
    pub source: Option<Arc<dyn SourceTerm2D>>,
    /// Optional bathymetry
    pub bathymetry: Option<Arc<Bathymetry2D>>,
    /// Limiter
    pub limiter: StandardLimiter2D,
    /// Whether to use well-balanced scheme
    pub well_balanced: bool,
    /// Spatial formulation of the SWE operator
    pub formulation: SWEFormulation2D,
    /// Wetting/drying treatment, if enabled
    pub wet_dry: Option<WetDryConfig>,
    /// `WetDry` only: elements with a node shallower than this (m) take the
    /// subcell finite volumes (default `h_dry`). The 3D model sets it to its
    /// thin-column depth, so that its layers move through the same subcells
    /// ([`crate::physics::Hydrostatic3D::with_min_column_depth`]).
    pub subcell_depth: Option<f64>,
    /// Bottom friction applied point-implicitly, if any
    pub friction: Option<Arc<dyn BottomFriction2D>>,
    /// Net-cage drag applied point-implicitly, if any
    pub cages: Option<Arc<CageDrag2D>>,
    /// Horizontal eddy viscosity (BR1), if any
    pub viscosity: Option<Arc<HorizontalViscosity2D>>,
    /// Polynomial order
    pub order: usize,
    /// Elements emptied because their mean depth was negative
    negative_depth_clips: AtomicUsize,
    /// Viscous time step of every element at CFL 1 for ν = 1 m²/s, which
    /// scales as 1/ν (geometry only; computed at first use)
    unit_viscous_dt: OnceLock<Vec<f64>>,
    /// Every element's viscous time step at CFL 1 for the current viscosity,
    /// reused between steps
    viscous_dt: Mutex<ViscousDt>,
}

/// Buffers of [`SWEPhysics2D`]'s viscous time step.
#[derive(Default)]
struct ViscousDt {
    /// The largest Smagorinsky ν of every element
    nu: Vec<f64>,
    /// The step of every element at CFL 1
    dt: Vec<f64>,
}

impl<BC: SWEBoundaryCondition2D> PhysicsModuleInfo for SWEPhysics2D<BC> {
    fn name(&self) -> &'static str {
        "swe-2d"
    }

    fn description(&self) -> &str {
        "2D Shallow Water Equations"
    }

    fn n_variables(&self) -> usize {
        3
    }

    fn variable_names(&self) -> &[&'static str] {
        &["h", "hu", "hv"]
    }
}

impl<BC: SWEBoundaryCondition2D> SWEPhysics2D<BC> {
    /// Whether this run has wetting/drying (a positivity limiter or a
    /// [`WetDryConfig`]).
    pub fn has_wetting_drying(&self) -> bool {
        self.wet_dry.is_some() || self.limiter.preserves_positivity()
    }

    /// The positivity bound of a wet/dry run under the SSP scheme `scheme`,
    /// capped at the subcells' stability limit in elements with a node at or
    /// below the RHS's subcell threshold (`WetDryConfig::h_dry`, else its
    /// default).
    fn positivity_bound(&self, scheme: Option<SspScheme>) -> Option<PositivityBound> {
        let h_dry = self
            .wet_dry
            .as_ref()
            .map_or(WetDryConfig::DEFAULT_H_DRY, |config| config.h_dry.meters());
        // (Its subcell cap takes elements with a node within 100 h_dry, 0.1 m
        // by default: a 3D model's subcell depth, see `subcell_depth`)
        self.has_wetting_drying()
            .then(|| PositivityBound::new(self.order, scheme, h_dry))
    }

    /// Number of elements (summed over all stages so far) whose mean depth
    /// was negative and that were emptied, creating mass.
    ///
    /// Zero under the positivity CFL with HLL or Rusanov; anything else means
    /// the time step or the flux broke the positivity guarantee.
    pub fn negative_depth_clips(&self) -> usize {
        self.negative_depth_clips.load(Ordering::Relaxed)
    }

    /// [`domain_fingerprint_2d`] of this module's mesh and bed (a flat bed
    /// without bathymetry).
    pub fn domain_fingerprint(&self) -> u64 {
        let bed = self.bathymetry.as_ref().map_or(&[][..], |b| &b.data[..]);
        domain_fingerprint_2d(&self.mesh, bed)
    }

    /// The run's state `state` at time `t` between two steps, to save with
    /// [`Restart2D::write`]. A 2D run carries nothing else from one step to
    /// the next: local time stepping assigns its levels from the state every
    /// step, and this module's caches are recomputed from it.
    pub fn restart(&self, state: &SWESolution2D, t: f64) -> Restart2D {
        Restart2D {
            time: t,
            domain: self.domain_fingerprint(),
            state: state.clone(),
            driver: DriverData::default(),
        }
    }

    /// Check that `restart` is of this module's domain (its mesh and bed), so
    /// that a run can continue from [`Restart2D::time`] with
    /// [`Restart2D::state`]. The rest of the physics (forcing, boundaries,
    /// friction, viscosity, cages) is not in the file: the run continues
    /// under this module's, which is how a branch run adds cages to a spun-up
    /// open run.
    pub fn check_restart(&self, restart: &Restart2D) -> Result<(), RestartError> {
        let s = &restart.state;
        let expected = (self.mesh.n_elements, self.ops.n_nodes);
        if (s.n_elements, s.n_nodes) != expected {
            return Err(RestartError::Domain(format!(
                "{} elements × {} nodes, this run has {} × {}",
                s.n_elements, s.n_nodes, expected.0, expected.1
            )));
        }
        let domain = self.domain_fingerprint();
        if restart.domain != domain {
            return Err(RestartError::Domain(format!(
                "its mesh or bed differ (fingerprint {:016x}, this run {domain:016x})",
                restart.domain
            )));
        }
        Ok(())
    }

    /// The wet/dry correction applies the same positivity limiter to every
    /// element with a node below h_dry, which includes every element a
    /// positivity limiter with a threshold ≤ h_dry changes: that pass can be
    /// skipped.
    fn positivity_pass_is_redundant(&self) -> bool {
        matches!(
            (&self.limiter, &self.wet_dry),
            (StandardLimiter2D::Positivity(h), Some(config)) if *h <= config.h_dry.meters()
        )
    }

    /// [`PhysicsModule::post_process`] of element `k` alone (local time
    /// stepping): the same element kernels. Returns the clips.
    fn post_process_element(&self, k: usize, [h, hu, hv]: [&mut [f64]; 3]) -> usize {
        let mut clips = 0;
        if !self.positivity_pass_is_redundant() {
            clips += self
                .limiter
                .apply_element(k, [&mut *h, &mut *hu, &mut *hv], &self.geom)
                as usize;
        }
        if let Some(ref config) = self.wet_dry {
            clips += apply_wet_dry_correction_element(k, [h, hu, hv], &self.geom, config) as usize;
        }
        clips
    }

    fn count_clips(&self, clips: usize) {
        if clips > 0 {
            self.negative_depth_clips
                .fetch_add(clips, Ordering::Relaxed);
        }
    }

    fn damping(&self) -> ImplicitDamping2D<'_> {
        ImplicitDamping2D {
            friction: self.friction.as_deref(),
            cages: self.cages.as_deref(),
            wet_dry: self.wet_dry.as_ref(),
            h_min: self.equation.h_min,
        }
    }

    /// Stable time step of the viscous term alone on every element at CFL 1
    /// ([`element_dt_viscous_swe_2d`]); `None` without viscosity.
    ///
    /// BR1 couples an element to its face neighbours' gradients, so each
    /// element is bounded by the largest ν of its own nodes and theirs. A
    /// Smagorinsky ν follows the strain: it is taken from `state`'s
    /// ([`largest_viscosity_swe_2d`]), so the step follows the shear as it
    /// develops (one BR1 gradient pass per call).
    fn viscous_dt(&self, state: &SWESolution2D) -> Option<MutexGuard<'_, ViscousDt>> {
        let viscosity = self.viscosity.as_deref()?;
        if viscosity.background <= 0.0 && !viscosity.is_strain_dependent() {
            return None;
        }
        let unit = self.unit_viscous_dt.get_or_init(|| {
            (0..self.mesh.n_elements)
                .map(|k| {
                    element_dt_viscous_swe_2d(
                        &self.mesh, &self.ops, &self.geom, 1.0, self.order, 1.0, k,
                    )
                })
                .collect()
        });
        let mut guard = self.viscous_dt.lock().expect("Failed to lock viscous_dt");
        let ViscousDt { nu, dt } = &mut *guard;
        dt.resize(self.mesh.n_elements, 0.0);
        if viscosity.is_strain_dependent() {
            nu.resize(self.mesh.n_elements, 0.0);
            largest_viscosity_swe_2d(nu, state, &self.mesh, &self.ops, &self.geom, viscosity);
            for (k, (dt, &unit)) in dt.iter_mut().zip(unit).enumerate() {
                let largest = (0..4)
                    .filter_map(|face| self.mesh.neighbor(ElementIndex::new(k), face))
                    .map(|nb| nu[nb.element])
                    .fold(nu[k], f64::max);
                *dt = unit / largest;
            }
        } else {
            for (dt, &unit) in dt.iter_mut().zip(unit) {
                *dt = unit / viscosity.background;
            }
        }
        Some(guard)
    }

    /// RHS configuration for this module's components.
    fn rhs_config(&self) -> crate::solver::SWE2DRhsConfig<'_, BC> {
        use crate::solver::SWE2DRhsConfig;

        let mut config = SWE2DRhsConfig::new(&self.equation, &self.bc)
            .with_formulation(self.formulation)
            .with_flux_type(self.flux.into())
            .with_coriolis(false); // Use source terms instead

        if let Some(ref source) = self.source {
            config.source_terms = Some(source.as_ref());
        }

        if let Some(ref bathy) = self.bathymetry {
            config = config.with_bathymetry(bathy.as_ref());
            if self.well_balanced {
                config = config.with_well_balanced(true);
            }
        }
        if let Some(ref wet_dry) = self.wet_dry {
            config = config.with_dry_threshold(wet_dry.h_dry.meters());
        }
        if let Some(depth) = self.subcell_depth {
            config = config.with_subcell_depth(depth);
        }
        if let Some(ref viscosity) = self.viscosity {
            config = config.with_viscosity(viscosity.as_ref());
        }
        config
    }
}

/// Time step for two rates at once: `1/(1/a + 1/b)`, with `a` and `b` the
/// steps each allows alone (advection and viscosity).
fn combine_dt(a: f64, b: f64) -> f64 {
    if a.is_infinite() {
        b
    } else if b.is_infinite() {
        a
    } else {
        1.0 / (1.0 / a + 1.0 / b)
    }
}

impl<BC: SWEBoundaryCondition2D> PhysicsModule<SWESolution2D> for SWEPhysics2D<BC> {
    fn compute_rhs(&self, state: &SWESolution2D, time: f64) -> SWESolution2D {
        let mut out = SWESolution2D::new(self.mesh.n_elements, self.ops.n_nodes);
        self.compute_rhs_into(state, time, &mut out);
        out
    }

    /// Parallel element kernel when `parallel` is enabled (identical result to
    /// the serial one); allocation-free after the first call.
    fn compute_rhs_into(&self, state: &SWESolution2D, time: f64, out: &mut SWESolution2D) {
        #[cfg(not(feature = "parallel"))]
        use crate::solver::compute_rhs_swe_2d_into as rhs_into;
        #[cfg(feature = "parallel")]
        use crate::solver::compute_rhs_swe_2d_parallel_into as rhs_into;

        let config = self.rhs_config();
        rhs_into(state, &self.mesh, &self.ops, &self.geom, &config, time, out);
    }

    fn compute_dt(&self, state: &SWESolution2D, cfl: f64) -> f64 {
        #[cfg(not(feature = "parallel"))]
        use crate::solver::compute_dt_swe_2d as dt;
        #[cfg(feature = "parallel")]
        use crate::solver::compute_dt_swe_2d_parallel as dt;

        let dt = dt(
            state,
            &self.mesh,
            &self.geom,
            &self.equation,
            self.order,
            cfl,
        );
        match self.viscous_dt(state) {
            Some(viscous) => {
                let dt_viscous = viscous.dt.iter().copied().fold(f64::INFINITY, f64::min);
                combine_dt(dt, cfl * dt_viscous)
            }
            None => dt,
        }
    }

    /// With wetting/drying, the positivity bound relaxed per element
    /// ([`PositivityBound`]): the smallest element step.
    fn compute_dt_ssp(&self, state: &SWESolution2D, cfl: f64, scheme: Option<SspScheme>) -> f64 {
        let Some(positivity) = self.positivity_bound(scheme) else {
            return self.compute_dt(state, cfl);
        };
        let dt = min_element_dt_swe_2d(
            state,
            &self.mesh,
            &self.ops,
            &self.geom,
            &self.equation,
            self.order,
            cfl,
            Some(positivity),
        );
        match self.viscous_dt(state) {
            Some(viscous) => {
                let dt_viscous = viscous.dt.iter().copied().fold(f64::INFINITY, f64::min);
                combine_dt(dt, cfl * dt_viscous)
            }
            None => dt,
        }
    }

    /// Limiter, then positivity and velocity desingularization (wet/dry).
    fn post_process(&self, state: &mut SWESolution2D) {
        let ctx = LimiterContext2D::new(&self.mesh, &self.ops, &self.geom);
        let mut clips = if self.positivity_pass_is_redundant() {
            0
        } else {
            self.limiter.apply_counting(state, &ctx)
        };
        if let Some(ref config) = self.wet_dry {
            clips += wet_dry_correction(state, &self.geom, config);
        }
        self.count_clips(clips);
    }

    /// Point-implicit bottom friction, cage drag and thin-layer relaxation.
    fn implicit_damping(&self, stage: &mut SWESolution2D, from: &SWESolution2D, dt: f64) {
        implicit_damping(stage, from, dt, &self.damping());
    }

    /// The DGSEM positivity bound (for forward Euler) when the run has
    /// wetting/drying.
    fn max_cfl(&self) -> Option<f64> {
        self.has_wetting_drying()
            .then(|| positivity_cfl_swe_2d(self.order))
    }

    fn mesh(&self) -> &Mesh2D {
        &self.mesh
    }

    fn operators(&self) -> &DGOperators2D {
        &self.ops
    }

    fn geometry(&self) -> &GeometricFactors2D {
        &self.geom
    }

    fn order(&self) -> usize {
        self.order
    }

    fn local_time_stepping(&self) -> Option<&dyn LocalTimeStepping<SWESolution2D>> {
        Some(self)
    }
}

/// Local time stepping ([`crate::time::Multirate`]). Supports every
/// formulation, source term, boundary condition, wetting/drying, the
/// point-implicit damping and horizontal viscosity (a two-hop stencil); not
/// the Kuzmin limiters (not element-local; they panic).
impl<BC: SWEBoundaryCondition2D> LocalTimeStepping<SWESolution2D> for SWEPhysics2D<BC> {
    /// Not with the Kuzmin limiters (bounds from the neighbours' means).
    fn is_element_local(&self) -> bool {
        self.limiter.is_element_local()
    }

    /// Wider with horizontal viscosity: BR1 reads the neighbours' gradients.
    fn stencil(&self) -> RhsStencil {
        if self.viscosity.is_some() {
            RhsStencil::FacesAndCorners
        } else {
            RhsStencil::Faces
        }
    }

    fn element_dt(&self, state: &SWESolution2D, cfl: f64, scheme: SspScheme, out: &mut [f64]) {
        element_dt_swe_2d(
            state,
            &self.mesh,
            &self.ops,
            &self.geom,
            &self.equation,
            self.order,
            cfl,
            self.positivity_bound(Some(scheme)),
            out,
        );
        if let Some(viscous) = self.viscous_dt(state) {
            for (dt, &dt_viscous) in out.iter_mut().zip(viscous.dt.iter()) {
                *dt = combine_dt(*dt, cfl * dt_viscous);
            }
        }
    }

    /// The RHS, then per element and in the same pass: the stage
    /// combination, the implicit damping, and either the sum into `acc` or
    /// the limiter and wet/dry correction.
    fn stage_where(
        &self,
        base: &SWESolution2D,
        input: &SWESolution2D,
        elements: &[u32],
        plan: &(dyn Fn(usize) -> ElementStage + Sync),
        out: &mut SWESolution2D,
        acc: Option<&mut SWESolution2D>,
    ) {
        let config = self.rhs_config();
        let damping = self.damping();
        let n = self.ops.n_nodes;
        let clips = AtomicUsize::new(0);
        let then = |k: usize, [h, hu, hv]: [&mut [f64]; 3], sum: Option<[&mut [f64]; 3]>| {
            let stage = plan(k);
            let nodes = k * n..(k + 1) * n;
            // out = a·base + b·input + c·F, F in the rows (as scale + axpy
            // round it)
            for (var, row) in [&mut *h, &mut *hu, &mut *hv].into_iter().enumerate() {
                let (base, input) = (
                    &base.data[var][nodes.clone()],
                    &input.data[var][nodes.clone()],
                );
                for ((f, &s), &x) in row.iter_mut().zip(base).zip(input) {
                    let mut v = if stage.a != 0.0 { stage.a * s } else { 0.0 };
                    if stage.b != 0.0 {
                        v += stage.b * x;
                    }
                    if stage.c != 0.0 {
                        v += stage.c * *f;
                    }
                    *f = v;
                }
            }
            let from = [0, 1, 2].map(|var| &input.data[var][nodes.clone()]);
            damping.damp_element(k, [&mut *h, &mut *hu, &mut *hv], from, stage.c);
            match (stage.accumulate, sum) {
                (Some((w, keep)), Some(sum)) => {
                    for (row, sum) in [&*h, &*hu, &*hv].into_iter().zip(sum) {
                        for (s, &v) in sum.iter_mut().zip(row.iter()) {
                            *s = if keep != 0.0 {
                                w * v + keep * *s
                            } else {
                                w * v
                            };
                        }
                    }
                }
                (Some(_), None) => panic!("an accumulating stage needs the sum buffer"),
                (None, _) => {
                    let clipped = self.post_process_element(k, [h, hu, hv]);
                    if clipped > 0 {
                        clips.fetch_add(clipped, Ordering::Relaxed);
                    }
                }
            }
        };
        compute_rhs_swe_2d_subset_then(
            input,
            &self.mesh,
            &self.ops,
            &self.geom,
            &config,
            elements,
            &|k| plan(k).time,
            out,
            acc,
            &then,
        );
        self.count_clips(clips.into_inner());
    }

    fn finish_where(&self, state: &mut SWESolution2D, acc: &SWESolution2D, elements: &[u32]) {
        let n = self.ops.n_nodes;
        let [h, hu, hv] = &mut state.data;
        let rows = [
            DisjointChunks::new(h, n),
            DisjointChunks::new(hu, n),
            DisjointChunks::new(hv, n),
        ];
        let element = |&k: &u32| {
            let k = k as usize;
            // SAFETY: the listed elements are distinct
            let mut rows = rows.each_ref().map(|r| unsafe { r.chunk(k) });
            for (var, row) in rows.iter_mut().enumerate() {
                row.copy_from_slice(&acc.data[var][k * n..(k + 1) * n]);
            }
            self.post_process_element(k, rows)
        };
        #[cfg(feature = "parallel")]
        let clips = if elements.len() >= 64 {
            use rayon::prelude::*;
            elements.par_iter().map(element).sum()
        } else {
            elements.iter().map(element).sum()
        };
        #[cfg(not(feature = "parallel"))]
        let clips = elements.iter().map(element).sum();
        self.count_clips(clips);
    }
}

impl<BC: SWEBoundaryCondition2D> crate::time::BarotropicPhysics for SWEPhysics2D<BC>
where
    SWEPhysics2D<BC>: PhysicsModule<SWESolution2D>,
{
    /// [`PhysicsModule::compute_rhs_into`] that also writes the numerical mass
    /// flux of every element face and subcell interface (layouts of
    /// [`crate::solver::compute_rhs_swe_2d_mass_fluxes_into`]).
    fn compute_rhs_mass_fluxes_into(
        &self,
        state: &SWESolution2D,
        time: f64,
        out: &mut SWESolution2D,
        face_mass: &mut [f64],
        subcell_mass: &mut [f64],
    ) {
        #[cfg(not(feature = "parallel"))]
        use crate::solver::compute_rhs_swe_2d_mass_fluxes_into as rhs_into;
        #[cfg(feature = "parallel")]
        use crate::solver::compute_rhs_swe_2d_parallel_mass_fluxes_into as rhs_into;

        let config = self.rhs_config();
        rhs_into(
            state,
            &self.mesh,
            &self.ops,
            &self.geom,
            &config,
            time,
            out,
            face_mass,
            subcell_mass,
        );
    }

    fn metric_form(&self) -> crate::solver::rhs::MetricForm {
        crate::solver::rhs::MetricForm::of(self.formulation)
    }

    /// `S_h` of the source terms that change the water's volume
    /// ([`SourceTerm2D::changes_mass`]), element by element as the RHS
    /// kernels add them.
    fn finishes_by_element(&self) -> bool {
        self.limiter.is_element_local()
    }

    fn damp_element(&self, k: usize, stage: [&mut [f64]; 3], from: [&[f64]; 3], dt: f64) {
        self.damping().damp_element(k, stage, from, dt);
    }

    fn post_process_element(&self, k: usize, rows: [&mut [f64]; 3]) -> usize {
        SWEPhysics2D::post_process_element(self, k, rows)
    }

    fn stage_subset_then(
        &self,
        input: &SWESolution2D,
        elements: &[u32],
        time_of: &(dyn Fn(usize) -> f64 + Sync),
        out: &mut SWESolution2D,
        then: crate::solver::rhs::ElementRhsMassThen,
    ) {
        crate::solver::rhs::compute_rhs_swe_2d_subset_mass_fluxes_then(
            input,
            &self.mesh,
            &self.ops,
            &self.geom,
            &self.rhs_config(),
            elements,
            time_of,
            out,
            then,
        );
    }

    fn mass_source_element(
        &self,
        state: &SWESolution2D,
        k: usize,
        time: f64,
        out: &mut [f64],
    ) -> bool {
        let Some(source) = self.source.as_deref().filter(|s| s.changes_mass()) else {
            return false;
        };
        out.fill(0.0);
        let element = crate::source::ElementSources {
            element: ElementIndex::new(k),
            time,
            solution: state,
            mesh: &self.mesh,
            ops: &self.ops,
            bathymetry: self.bathymetry.as_deref(),
            g: self.equation.g,
            h_min: self.equation.h_min.meters(),
        };
        source.add_mass_element(&element, out);
        out.iter().any(|&s| s != 0.0)
    }

    fn record_clips(&self, clips: usize) {
        self.count_clips(clips);
    }

    fn mass_sources_into(&self, state: &SWESolution2D, time: f64, out: &mut [f64]) -> bool {
        let Some(source) = self.source.as_deref().filter(|s| s.changes_mass()) else {
            return false;
        };
        let n_elements = self.mesh.n_elements;
        crate::solver::core::blocks::for_each_block(
            n_elements,
            [&mut out[..n_elements * self.ops.n_nodes]],
            || (),
            |_, k, [h]| {
                h.fill(0.0);
                let element = crate::source::ElementSources {
                    element: ElementIndex::new(k),
                    time,
                    solution: state,
                    mesh: &self.mesh,
                    ops: &self.ops,
                    bathymetry: self.bathymetry.as_deref(),
                    g: self.equation.g,
                    h_min: self.equation.h_min.meters(),
                };
                source.add_mass_element(&element, h);
            },
        );
        true
    }
}

// =============================================================================
// SWE Physics 2D Builder
// =============================================================================

/// Builder for 2D Shallow Water Equations physics module.
///
/// # Example
/// ```ignore
/// use dg_rs::physics::SWEPhysics2DBuilder;
/// use dg_rs::flux::StandardFlux2D;
/// use dg_rs::solver::StandardLimiter2D;
///
/// let physics = SWEPhysics2DBuilder::new(mesh, ops, geom, equation, bc)
///     .with_flux(StandardFlux2D::Roe)
///     .with_limiter(StandardLimiter2D::TvbWithPositivity { tvb, h_min })
///     .with_bathymetry(bathymetry)
///     .with_well_balanced(true)
///     .build();
/// ```
pub struct SWEPhysics2DBuilder<BC: SWEBoundaryCondition2D> {
    mesh: Arc<Mesh2D>,
    ops: Arc<DGOperators2D>,
    geom: Arc<GeometricFactors2D>,
    equation: ShallowWater2D,
    bc: BC,
    flux: Option<StandardFlux2D>,
    sources: Vec<Arc<dyn SourceTerm2D>>,
    bathymetry: Option<Arc<Bathymetry2D>>,
    limiter: StandardLimiter2D,
    well_balanced: bool,
    formulation: Option<SWEFormulation2D>,
    wet_dry: Option<WetDryConfig>,
    friction: Option<Arc<dyn BottomFriction2D>>,
    cages: Option<Arc<CageDrag2D>>,
    viscosity: Option<Arc<HorizontalViscosity2D>>,
    order: usize,
}

impl<BC: SWEBoundaryCondition2D> SWEPhysics2DBuilder<BC> {
    /// Create a new builder with required components.
    pub fn new(
        mesh: Arc<Mesh2D>,
        ops: Arc<DGOperators2D>,
        geom: Arc<GeometricFactors2D>,
        equation: ShallowWater2D,
        bc: BC,
    ) -> Self {
        let order = ops.order;
        Self {
            mesh,
            ops,
            geom,
            equation,
            bc,
            flux: None,
            sources: Vec::new(),
            bathymetry: None,
            limiter: StandardLimiter2D::default(),
            well_balanced: false,
            formulation: None,
            wet_dry: None,
            friction: None,
            cages: None,
            viscosity: None,
            order,
        }
    }

    /// Set the numerical flux.
    ///
    /// Default: HLL for runs with wetting/drying (a positivity limiter or
    /// [`Self::with_wet_dry`]), Roe otherwise. Roe is not positivity preserving
    /// (Einfeldt et al. 1991), so wet/dry runs should keep HLL or use Rusanov.
    pub fn with_flux(mut self, flux: StandardFlux2D) -> Self {
        self.flux = Some(flux);
        self
    }

    /// Add a source term; repeated calls add up (Coriolis, wind, …).
    pub fn with_source<S: SourceTerm2D + 'static>(self, source: S) -> Self {
        self.with_source_arc(Arc::new(source))
    }

    /// Add a source term held by an `Arc`.
    pub fn with_source_arc(mut self, source: Arc<dyn SourceTerm2D>) -> Self {
        self.sources.push(source);
        self
    }

    /// Set the bathymetry (bed elevation `B`).
    ///
    /// A bed that is not flat needs a formulation that applies its slope (see
    /// [`Self::with_formulation`]): without wetting/drying the default is then
    /// `EntropyStable`, unless the source terms carry the slope
    /// (`BathymetrySource2D`, for `Standard`).
    pub fn with_bathymetry(mut self, bathymetry: Arc<Bathymetry2D>) -> Self {
        self.bathymetry = Some(bathymetry);
        self
    }

    /// Set the limiter.
    pub fn with_limiter(mut self, limiter: StandardLimiter2D) -> Self {
        self.limiter = limiter;
        self
    }

    /// Enable well-balanced scheme for bathymetry.
    pub fn with_well_balanced(mut self, enabled: bool) -> Self {
        self.well_balanced = enabled;
        self
    }

    /// Set the spatial formulation of the SWE operator.
    ///
    /// Default: `WetDry` for runs with wetting/drying (a positivity limiter or
    /// [`Self::with_wet_dry`]); otherwise `EntropyStable` over a bed that is
    /// not flat, unless the source terms contain `BathymetrySource2D`, and
    /// `Standard` for a flat bed or with that source. The split-form
    /// formulations include the bed slope in the operator, so the source terms
    /// must not contain `BathymetrySource2D`; choose `Standard` (with
    /// [`Self::with_well_balanced`]) to keep it.
    ///
    /// `Standard` gets the bed slope only from `BathymetrySource2D`: over a bed
    /// that is not flat without it, [`Self::build`] panics (the bed would push
    /// nothing, and a lake at rest over a slope would pile up at its shallow
    /// end).
    pub fn with_formulation(mut self, formulation: SWEFormulation2D) -> Self {
        self.formulation = Some(formulation);
        self
    }

    /// Enable wetting/drying with the given configuration (see
    /// [`SWEPhysics2D`]): positivity, velocity desingularization and thin-layer
    /// relaxation below `config.h_dry`.
    pub fn with_wet_dry(mut self, config: WetDryConfig) -> Self {
        self.wet_dry = Some(config);
        self
    }

    /// Enable (with [`WetDryConfig::default`], h_dry = 1 mm) or disable
    /// wetting/drying.
    pub fn with_wet_dry_correction(mut self, enabled: bool) -> Self {
        self.wet_dry = enabled.then(WetDryConfig::default);
        self
    }

    /// Apply a bottom friction law point-implicitly in every RK stage
    /// (`TimeIntegrator::step_with_relaxation`), instead of as an explicit
    /// source term.
    ///
    /// Unconditionally stable and sign-preserving: explicit friction flips the
    /// momentum once Δt·C_f|u|/h > 2.5, which happens at every wet/dry front.
    /// Do not also put the same law in the source terms.
    pub fn with_implicit_friction<F: BottomFriction2D + 'static>(mut self, friction: F) -> Self {
        self.friction = Some(Arc::new(friction));
        self
    }

    /// Apply the drag of fish-farm net cages point-implicitly in every RK
    /// stage, together with the bottom friction (see
    /// [`crate::source::CageDrag2D`]).
    ///
    /// # Panics
    /// At the first step, if `cages` was built for another mesh or order.
    pub fn with_cage_drag(mut self, cages: CageDrag2D) -> Self {
        self.cages = (!cages.is_empty()).then(|| Arc::new(cages));
        self
    }

    /// Add horizontal eddy viscosity (a constant background plus,
    /// optionally, Smagorinsky's; BR1 face coupling). It couples each element
    /// to its neighbours' gradients, so the RHS of an element also reads the
    /// elements at its neighbours' far corners; local time stepping follows
    /// that wider stencil. The time step is bounded by the viscous limit,
    /// with a Smagorinsky ν from the current strain.
    pub fn with_viscosity(mut self, viscosity: HorizontalViscosity2D) -> Self {
        self.viscosity = Some(Arc::new(viscosity));
        self
    }

    /// Build the physics module.
    ///
    /// # Panics
    /// If a wet/dry run leaves the formulation at its default (`WetDry`) but
    /// its source terms contain `BathymetrySource2D`, which would count the
    /// bed slope twice.
    pub fn build(self) -> SWEPhysics2D<BC> {
        let wetting_drying = self.wet_dry.is_some() || self.limiter.preserves_positivity();
        let source: Option<Arc<dyn SourceTerm2D>> = match self.sources.len() {
            0 => None,
            1 => self.sources.first().cloned(),
            _ => Some(Arc::new(SourceTerms2D::new(self.sources))),
        };
        let slope_source = source
            .as_ref()
            .is_some_and(|s| s.includes_bathymetry_slope());
        let sloped_bed = self.bathymetry.as_ref().is_some_and(|b| !is_flat(b));
        let formulation = self.formulation.unwrap_or_else(|| {
            if !wetting_drying {
                return if sloped_bed && !slope_source {
                    SWEFormulation2D::EntropyStable
                } else {
                    SWEFormulation2D::Standard
                };
            }
            assert!(
                !source
                    .as_ref()
                    .is_some_and(|s| s.includes_bathymetry_slope()),
                "wet/dry runs default to SWEFormulation2D::WetDry, which includes the \
                 bed slope: remove BathymetrySource2D from the sources, or keep it with \
                 .with_formulation(SWEFormulation2D::Standard)"
            );
            SWEFormulation2D::WetDry
        });
        assert!(
            !(formulation == SWEFormulation2D::Standard && sloped_bed && !slope_source),
            "SWEFormulation2D::Standard gets the bed slope only from BathymetrySource2D: \
             add it with .with_source(BathymetrySource2D::new(g)), or use a split form \
             (EntropyStable, or WetDry for wetting and drying), which applies the slope itself"
        );
        let flux = self.flux.unwrap_or(if wetting_drying {
            StandardFlux2D::HLL
        } else {
            StandardFlux2D::Roe
        });
        if wetting_drying && flux == StandardFlux2D::Roe {
            eprintln!(
                "warning: SWEPhysics2D with wetting/drying and the Roe flux: \
                 Roe is not positivity preserving; use HLL or Rusanov"
            );
        }

        // Per-node fields must be the mesh's (the per-element damping of
        // the fused and local stages does not check them)
        let n_total = self.mesh.n_elements * self.ops.n_nodes;
        if let Some(n) = self.friction.as_ref().and_then(|f| f.n_total_nodes()) {
            assert_eq!(
                n, n_total,
                "the friction field was built for a different mesh or order"
            );
        }
        if let Some(cages) = &self.cages {
            assert_eq!(
                cages.n_total_nodes(),
                n_total,
                "CageDrag2D was built for a different mesh or order"
            );
        }

        SWEPhysics2D {
            mesh: self.mesh,
            ops: self.ops,
            geom: self.geom,
            equation: self.equation,
            flux,
            bc: self.bc,
            source,
            bathymetry: self.bathymetry,
            limiter: self.limiter,
            well_balanced: self.well_balanced,
            formulation,
            wet_dry: self.wet_dry,
            subcell_depth: None,
            friction: self.friction,
            cages: self.cages,
            viscosity: self.viscosity,
            order: self.order,
            negative_depth_clips: AtomicUsize::new(0),
            unit_viscous_dt: OnceLock::new(),
            viscous_dt: Mutex::default(),
        }
    }
}

// =============================================================================
// Generic Physics Builder
// =============================================================================

/// Generic entry point for physics builders.
///
/// Provides factory methods for creating specialized physics builders.
pub struct PhysicsBuilder;

impl PhysicsBuilder {
    /// Create a builder for 2D Shallow Water Equations.
    pub fn swe_2d<BC: SWEBoundaryCondition2D>(
        mesh: Arc<Mesh2D>,
        ops: Arc<DGOperators2D>,
        geom: Arc<GeometricFactors2D>,
        equation: ShallowWater2D,
        bc: BC,
    ) -> SWEPhysics2DBuilder<BC> {
        SWEPhysics2DBuilder::new(mesh, ops, geom, equation, bc)
    }
}

/// Whether every nodal bed elevation is the same (to round-off).
fn is_flat(bathymetry: &Bathymetry2D) -> bool {
    let (lo, hi) = bathymetry
        .data
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &b| {
            (lo.min(b), hi.max(b))
        });
    hi - lo <= 1e-12 * hi.abs().max(lo.abs()).max(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::Reflective2D;
    use crate::types::{Depth, ElementIndex};

    fn k(idx: usize) -> ElementIndex {
        ElementIndex::new(idx)
    }

    fn create_test_components() -> (Arc<Mesh2D>, Arc<DGOperators2D>, Arc<GeometricFactors2D>) {
        let mesh = Arc::new(Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 2, 2));
        let ops = Arc::new(DGOperators2D::new(2));
        let geom = Arc::new(GeometricFactors2D::compute(&mesh, &ops));
        (mesh, ops, geom)
    }

    /// A positivity limiter at or below h_dry is subsumed by the wet/dry
    /// correction, which `post_process` then runs alone: the result, clip
    /// count included, is bitwise that of running both.
    #[test]
    fn redundant_positivity_pass_is_skipped_exactly() {
        let mesh = Arc::new(Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 6, 6));
        let ops = Arc::new(DGOperators2D::new(3));
        let geom = Arc::new(GeometricFactors2D::compute(&mesh, &ops));
        let config = WetDryConfig::default();
        let h_dry = config.h_dry.meters();
        let physics = PhysicsBuilder::swe_2d(
            mesh.clone(),
            ops.clone(),
            geom.clone(),
            ShallowWater2D::new(9.81),
            Reflective2D::default(),
        )
        .with_limiter(StandardLimiter2D::Positivity(h_dry))
        .with_wet_dry(config.clone())
        .build();

        // Wet, shoreline, film, negative-node and negative-mean elements
        let mut state = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        let mut seed = 12345_u64;
        let mut rand = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        for e in 0..mesh.n_elements {
            let scale = [1.0, 1e-2, 1e-3, 1e-4][e % 4];
            let shift = [0.0, -0.3, 0.2, -0.9][e / 9 % 4];
            for i in 0..ops.n_nodes {
                let h = scale * (rand() + shift);
                state.set_state(
                    k(e),
                    i,
                    crate::solver::SWEState2D::new(h, rand() - 0.5, rand() - 0.5),
                );
            }
        }

        let mut expected = state.clone();
        let ctx = LimiterContext2D::new(&mesh, &ops, &geom);
        let clips = StandardLimiter2D::Positivity(h_dry).apply_counting(&mut expected, &ctx)
            + wet_dry_correction(&mut expected, &geom, &config);
        assert!(clips > 0, "the state should have negative-mean elements");

        physics.post_process(&mut state);
        assert_eq!(physics.negative_depth_clips(), clips);
        for (a, b) in state.data.iter().zip(&expected.data) {
            assert!(a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()));
        }
    }

    #[test]
    fn test_builder_basic() {
        let (mesh, ops, geom) = create_test_components();
        let equation = ShallowWater2D::with_h_min(9.81, Depth::new(1e-6));
        let bc = Reflective2D::default();

        let physics = PhysicsBuilder::swe_2d(mesh, ops, geom, equation, bc).build();

        assert_eq!(physics.name(), "swe-2d");
        assert_eq!(physics.n_variables(), 3);
        assert_eq!(physics.order(), 2);
    }

    #[test]
    fn test_builder_with_options() {
        let (mesh, ops, geom) = create_test_components();
        let equation = ShallowWater2D::with_h_min(9.81, Depth::new(1e-6));
        let bc = Reflective2D::default();

        let physics = PhysicsBuilder::swe_2d(mesh, ops, geom, equation, bc)
            .with_flux(StandardFlux2D::HLL)
            .with_limiter(StandardLimiter2D::None)
            .with_well_balanced(true)
            .with_formulation(SWEFormulation2D::EntropyStable)
            .with_wet_dry_correction(true)
            .build();

        assert!(physics.well_balanced);
        assert_eq!(physics.formulation, SWEFormulation2D::EntropyStable);
        assert!(physics.wet_dry.is_some());
    }

    #[test]
    fn test_wet_dry_defaults() {
        let (mesh, ops, geom) = create_test_components();
        let builder = || {
            PhysicsBuilder::swe_2d(
                mesh.clone(),
                ops.clone(),
                geom.clone(),
                ShallowWater2D::new(9.81),
                Reflective2D::default(),
            )
        };

        // Fully wet run: Standard with Roe, no CFL cap
        let wet = builder().build();
        assert_eq!(wet.formulation, SWEFormulation2D::Standard);
        assert_eq!(wet.flux, StandardFlux2D::Roe);
        assert!(!wet.has_wetting_drying());
        assert_eq!(wet.max_cfl(), None);

        // Wetting/drying via the wet/dry treatment or a positivity limiter:
        // WetDry, HLL (for Standard) and the positivity CFL (order 2)
        for physics in [
            builder().with_wet_dry_correction(true).build(),
            builder()
                .with_limiter(StandardLimiter2D::Positivity(1e-3))
                .build(),
        ] {
            assert_eq!(physics.formulation, SWEFormulation2D::WetDry);
            assert_eq!(physics.flux, StandardFlux2D::HLL);
            assert_eq!(physics.max_cfl(), Some(positivity_cfl_swe_2d(2)));
        }

        // An explicit choice wins
        let rusanov = builder()
            .with_wet_dry_correction(true)
            .with_formulation(SWEFormulation2D::Standard)
            .with_source(crate::source::BathymetrySource2D::new(9.81))
            .with_flux(StandardFlux2D::Rusanov)
            .build();
        assert_eq!(rusanov.formulation, SWEFormulation2D::Standard);
        assert_eq!(rusanov.flux, StandardFlux2D::Rusanov);
        assert_eq!(
            rusanov.wet_dry.unwrap().h_dry,
            Depth::new(WetDryConfig::DEFAULT_H_DRY)
        );
    }

    #[test]
    #[should_panic(expected = "remove BathymetrySource2D")]
    fn test_wet_dry_default_rejects_bathymetry_source() {
        let (mesh, ops, geom) = create_test_components();
        PhysicsBuilder::swe_2d(
            mesh,
            ops,
            geom,
            ShallowWater2D::new(9.81),
            Reflective2D::default(),
        )
        .with_wet_dry_correction(true)
        .with_source(crate::source::BathymetrySource2D::new(9.81))
        .build();
    }

    /// Regression: `with_bathymetry` without wet/dry used `Standard` with no
    /// bed-slope term unless the caller added `BathymetrySource2D`, so a bed
    /// that is not flat pushed nothing. A lake at rest in a 30 km channel over
    /// a 20 → 10 m slope then piled up to η = +10 m at the shallow end within
    /// an hour. The default is now the balanced `EntropyStable`: at rest to
    /// round-off.
    #[test]
    fn a_sloping_bed_is_balanced_by_default() {
        use crate::simulation::Simulation;
        use crate::time::SSPRK3;

        let length = 30e3;
        let mesh = Arc::new(Mesh2D::uniform_rectangle(0.0, length, 0.0, 3e3, 10, 1));
        let ops = Arc::new(DGOperators2D::new(2));
        let geom = Arc::new(GeometricFactors2D::compute(&mesh, &ops));
        let bathymetry = Arc::new(Bathymetry2D::from_function(&mesh, &ops, &geom, |x, _| {
            -20.0 + 10.0 * x / length
        }));
        let physics = PhysicsBuilder::swe_2d(
            mesh.clone(),
            ops.clone(),
            geom,
            ShallowWater2D::new(9.81),
            Reflective2D::default(),
        )
        .with_bathymetry(bathymetry.clone())
        .build();
        assert_eq!(physics.formulation, SWEFormulation2D::EntropyStable);

        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for (h, b) in q.data[0].iter_mut().zip(&bathymetry.data) {
            *h = -b;
        }
        let result = Simulation::new(physics, SSPRK3)
            .with_cfl(0.5)
            .run(&mut q, 0.0, 3600.0);
        assert!(result.success, "{:?}", result.error);
        let eta = q.data[0]
            .iter()
            .zip(&bathymetry.data)
            .fold(0.0_f64, |m, (h, b)| m.max((h + b).abs()));
        assert!(eta < 1e-10, "the lake moved: |η| up to {eta:.3e} m");
    }

    /// `Standard` stays the default where it is right: a flat bed, or a
    /// sloping one with `BathymetrySource2D`.
    #[test]
    fn standard_stays_the_default_for_flat_beds_and_with_the_slope_source() {
        let (mesh, ops, geom) = create_test_components();
        let n = (mesh.n_elements, ops.n_nodes);
        let build = |bed: Bathymetry2D, source: bool| {
            let builder = PhysicsBuilder::swe_2d(
                mesh.clone(),
                ops.clone(),
                geom.clone(),
                ShallowWater2D::new(9.81),
                Reflective2D::default(),
            )
            .with_bathymetry(Arc::new(bed));
            let builder = if source {
                builder.with_source(crate::source::BathymetrySource2D::new(9.81))
            } else {
                builder
            };
            builder.build().formulation
        };
        let sloped = || Bathymetry2D::from_function(&mesh, &ops, &geom, |x, y| -10.0 + x + y);
        assert_eq!(
            build(Bathymetry2D::constant(n.0, n.1, -10.0), false),
            SWEFormulation2D::Standard
        );
        assert_eq!(build(sloped(), true), SWEFormulation2D::Standard);
        assert_eq!(build(sloped(), false), SWEFormulation2D::EntropyStable);
    }

    #[test]
    #[should_panic(expected = "gets the bed slope only from BathymetrySource2D")]
    fn standard_over_a_sloping_bed_needs_the_slope_source() {
        let (mesh, ops, geom) = create_test_components();
        let bed = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, _| -10.0 + x);
        PhysicsBuilder::swe_2d(
            mesh,
            ops,
            geom,
            ShallowWater2D::new(9.81),
            Reflective2D::default(),
        )
        .with_bathymetry(Arc::new(bed))
        .with_formulation(SWEFormulation2D::Standard)
        .build();
    }

    #[test]
    fn test_physics_module_info() {
        let (mesh, ops, geom) = create_test_components();
        let equation = ShallowWater2D::with_h_min(9.81, Depth::new(1e-6));
        let bc = Reflective2D::default();

        let physics = PhysicsBuilder::swe_2d(mesh, ops, geom, equation, bc).build();

        let info: &dyn PhysicsModuleInfo = &physics;
        assert_eq!(info.name(), "swe-2d");
        assert_eq!(info.variable_names(), &["h", "hu", "hv"]);
    }

    #[test]
    fn test_compute_dt() {
        let (mesh, ops, geom) = create_test_components();
        let equation = ShallowWater2D::with_h_min(9.81, Depth::new(1e-6));
        let bc = Reflective2D::default();

        let physics = PhysicsBuilder::swe_2d(mesh.clone(), ops.clone(), geom, equation, bc).build();

        // Create a state with some water
        let mut state = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for ki in 0..mesh.n_elements {
            for i in 0..ops.n_nodes {
                state.set_state(k(ki), i, crate::solver::SWEState2D::new(1.0, 0.0, 0.0));
            }
        }

        let dt = physics.compute_dt(&state, 0.5);
        assert!(dt > 0.0);
        assert!(dt < 1.0); // Should be a reasonable time step
    }
}

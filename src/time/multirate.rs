//! Multirate (local) time stepping: each element advances with its own
//! stable time step, a power-of-two fraction of the global one.
//!
//! On a locally refined mesh (a fish farm at 20 m inside a kilometre-scale
//! domain, fjord arms, deep channels next to shallow banks), global explicit
//! stepping lets the few smallest or fastest elements set the time step of
//! all. Here element k gets a level ℓₖ ∈ 0..=L and takes 2^ℓₖ substeps of
//! Δt_ℓ = Δt/2^ℓ per (coarse) step Δt, with Δt_L the smallest element time
//! step.
//!
//! # Scheme
//!
//! A multirate partitioned Runge–Kutta method in the construction of
//! Constantinescu & Sandu (2007, J. Sci. Comput. 33:239–278,
//! doi:10.1007/s10915-007-9151-y), which SLIM uses for DG ocean models (Seny
//! et al. 2013, IJNMF 71:41–64, doi:10.1002/fld.3646; 2014, JCP 256:135–160,
//! doi:10.1016/j.jcp.2013.07.041), on a base SSP scheme in Shu–Osher form
//! ([`SspScheme`]: SSP-RK3, or SSP-RK(4,3) with twice its SSP coefficient).
//!
//! The coarse step is split into P = 2^L finest substeps. In finest substep p
//! every element evaluates one step of the base scheme of its own. For
//! SSP-RK3:
//!
//! ```text
//! Y₂ = S + Δt_ℓ F₁
//! Y₃ = ¾ S + ¼ (Y₂ + Δt_ℓ F₂)
//! H  = ⅓ S + ⅔ (Y₃ + Δt_ℓ F₃)
//! ```
//!
//! S is the element's value at the start of its current substep, and Fₛ its
//! RHS with every face neighbour at *its* stage-s value of the same finest
//! substep p. An element's substep result is the mean of H over the finest
//! substeps it spans. Stage s of an element on level ℓ is evaluated at time
//! tₖ + cₛΔt_ℓ, with tₖ its substep start (c = (0, 1, ½) for SSP-RK3,
//! (0, ½, 1, ½) for SSP-RK(4,3)).
//!
//! - **SSP / positivity:** every element update is a convex combination of
//!   forward-Euler steps of size Δt_ℓ/C from limited stage values (limiters
//!   and wet/dry correction run on every stage value), C the base scheme's
//!   SSP coefficient. The Zhang–Shu positivity bound therefore holds for each
//!   element at its own Δt_ℓ, up to C times the forward-Euler bound.
//! - **Conservation:** every element gives stage s of finest substep p the
//!   same weight, bₛΔt/P with b the base scheme's weights ((⅙, ⅙, ⅔) for
//!   SSP-RK3). The flux through a face at (p, s) comes from both sides'
//!   (p, s) values, so it is the same for both sides. Mass is conserved to
//!   rounding.
//! - **Order:** the base scheme's (third) inside each level. Across level
//!   interfaces the coupling satisfies the second-order partitioned
//!   conditions (Σ bᵢcᵢ = ½ for every level) but not the third-order ones
//!   (for SSP-RK3, Σ bᵢ(A_coarse c_fine)ᵢ = 5/24 ≠ ⅙ for two levels). The gate
//!   (`tests/local_time_stepping_test.rs`) measures orders 2.6–2.8 against the
//!   global solution. The interface error dominates the time error, though:
//!   ≈ 50× SSP-RK3's at the same CFL for a gravity wave crossing four levels,
//!   1.5e-3 relative at CFL 0.8. It scales as (λΔt)², so it is negligible for
//!   tides (ωΔt ~ 1e-3).
//! - **One level** is the base scheme, bit for bit.
//!
//! # Levels
//!
//! An element's time step comes from its own wave speeds and those of its
//! face neighbours (`LocalTimeStepping::element_dt`). The coarse step is
//! 2^L times the smallest element time step, with L the most levels (up to the
//! maximum) that keep it at most the largest one. Neighbouring levels are then
//! made to differ by at most one, refining the coarser side. Together these
//! keep water that floods in from a neighbour within one substep from
//! outrunning an element's time step (a nearly dry element otherwise gets an
//! enormous one). Levels are reassigned every coarse step.
//!
//! **Cost.** An element's RHS reads the state of its stencil
//! ([`LocalTimeStepping::stencil`]: its face neighbours for the fluxes, and
//! with BR1 horizontal viscosity also the elements at their far corners), so
//! its stage-s values change only when those within s stencil steps change.
//! Its stage-s RHS is therefore evaluated only at the rate rₛ(k), the finest
//! level within s stencil steps, and reused in between. Near a level
//! interface the coarse side pays up to one stencil width per stage (three
//! for SSP-RK3, four for SSP-RK(4,3)) at the finer rate; elsewhere every
//! element runs at its own level.
//!
//! A reused RHS is the one the element would compute again: its inputs,
//! including its substep's stage time, are unchanged. So the flux through a
//! face at (p, s) is the same seen from both sides, also for a viscous flux
//! that reads the neighbours' gradients (each evaluated at its own
//! element's stage time). A stencil that is too narrow breaks exactly this,
//! and with it conservation (`tests/local_time_stepping_test.rs`).

use crate::mesh::Mesh2D;
use crate::types::ElementIndex;

use super::integrator::{
    Integrable, IntegratorInfo, SSPRK3, SSPRK43, SspScheme, StageWorkspace, TimeIntegrator,
};

/// What one element does in one stage of [`LocalTimeStepping::stage_where`].
///
/// With `F = L(input)` the element's RHS at `time`, the new stage value is
/// `out = a·base + b·input + c·F` (a Shu–Osher stage: `c = β·Δt`), summed in
/// that order, with zero terms left out. The stiff damping is then applied
/// point-implicitly over `c` with rates from `input`
/// ([`PhysicsModule::implicit_damping`](crate::physics::PhysicsModule::implicit_damping)).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ElementStage {
    /// Time of the RHS evaluation.
    pub time: f64,
    /// Weight of the base value (the substep start).
    pub a: f64,
    /// Weight of the input value (the RHS argument).
    pub b: f64,
    /// Weight of the RHS.
    pub c: f64,
    /// Then, if `Some((w, keep))`: `acc ← w·out + keep·acc` (the `acc` term
    /// only if `keep ≠ 0`). Otherwise `out` is
    /// [post-processed](crate::physics::PhysicsModule::post_process).
    pub accumulate: Option<(f64, f64)>,
}

/// The elements whose state the RHS of an element k reads
/// ([`LocalTimeStepping::stencil`]), besides k itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RhsStencil {
    /// Its face neighbours (numerical fluxes).
    Faces,
    /// Its face neighbours j and, for each, the elements across the two
    /// faces of j that meet the shared face at its corners: BR1 viscosity,
    /// whose face flux reads j's gradient at the shared face's nodes. With
    /// the diagonal GLL LIFT, that gradient reads beyond j only at the corner
    /// nodes. On a structured mesh: the 3 × 3 block around k.
    FacesAndCorners,
    /// The elements sharing a vertex with k (a vertex-patch limiter, e.g.
    /// the 3D tracers' Kuzmin limiter): `FacesAndCorners`, and at an
    /// irregular vertex every element around it.
    VertexPatches,
}

/// The element-subset operations that local time stepping needs from a
/// physics module (see
/// [`PhysicsModule::local_time_stepping`](crate::physics::PhysicsModule::local_time_stepping)).
///
/// Each works on a list of distinct elements in one pass, with work
/// proportional to the list (a multirate step makes many passes over small
/// subsets), gives them bit for bit what the whole-state operations give
/// them, and leaves the other elements untouched.
pub trait LocalTimeStepping<S>: Sync {
    /// Largest stable time step of every element, into `out` (one value per
    /// mesh element; `f64::INFINITY` where nothing limits it): the linear
    /// bound at `cfl` and the module's forward-Euler bounds for the SSP
    /// scheme `scheme`, as in
    /// [`PhysicsModule::compute_dt_ssp`](crate::physics::PhysicsModule::compute_dt_ssp).
    fn element_dt(&self, state: &S, cfl: f64, scheme: SspScheme, out: &mut [f64]);

    /// The elements whose state an element's RHS reads.
    fn stencil(&self) -> RhsStencil {
        RhsStencil::Faces
    }

    /// Whether the element operations below work in this configuration
    /// (e.g. not with a limiter that reads the neighbours' means). `false`
    /// makes `Simulation` run global steps through the whole-state stages
    /// instead of the fused ones; local time stepping panics.
    fn is_element_local(&self) -> bool {
        true
    }

    /// One stage for every listed element k, with stage `plan(k)` (see
    /// [`ElementStage`]): its RHS from `input`, which may read the values
    /// of its [`Self::stencil`], nothing further away; the new stage value
    /// into its rows of `out`, and its sum into `acc` when the stage
    /// accumulates (`acc` must then be given). `plan` may also be called for
    /// the face neighbours of the listed elements (their stage time, e.g. for
    /// the boundary values of their gradients).
    fn stage_where(
        &self,
        base: &S,
        input: &S,
        elements: &[u32],
        plan: &(dyn Fn(usize) -> ElementStage + Sync),
        out: &mut S,
        acc: Option<&mut S>,
    );

    /// `state ← acc` and
    /// [`PhysicsModule::post_process`](crate::physics::PhysicsModule::post_process)
    /// on the listed elements: the end of their substep.
    fn finish_where(&self, state: &mut S, acc: &S, elements: &[u32]);
}

// =============================================================================
// Integrator
// =============================================================================

/// Multirate local time stepping on the SSP scheme of the base integrator
/// `B` (see the [module documentation](self)): `Simulation` gives every
/// element its own time step, `Δt/2^ℓ` with `ℓ ≤ max_levels`.
/// [`MultirateSSPRK3`] and [`MultirateSSPRK43`] name the two bases.
///
/// It needs a physics module with
/// [`PhysicsModule::local_time_stepping`](crate::physics::PhysicsModule::local_time_stepping)
/// (`SWEPhysics2D`). As a plain [`TimeIntegrator`], without a physics module,
/// it takes global steps of `B`.
#[derive(Clone, Copy, Debug)]
pub struct Multirate<B> {
    base: B,
    max_levels: usize,
}

/// Multirate SSP-RK3: local time stepping on [`SSPRK3`].
pub type MultirateSSPRK3 = Multirate<SSPRK3>;

/// Multirate SSP-RK(4,3): local time stepping on [`SSPRK43`], whose SSP
/// coefficient of 2 doubles every element's step under a positivity bound.
pub type MultirateSSPRK43 = Multirate<SSPRK43>;

/// Largest supported number of levels below the coarsest (a time-step ratio
/// of 2¹⁶).
const MAX_LEVELS: usize = 16;

impl<B: IntegratorInfo + Default> Multirate<B> {
    /// At most `max_levels` levels below the coarsest, i.e. time steps from
    /// Δt down to Δt/2^max_levels. Zero gives global steps of `B`.
    ///
    /// # Panics
    /// As [`Self::with_base`].
    pub fn new(max_levels: usize) -> Self {
        Self::with_base(B::default(), max_levels)
    }
}

impl<B: IntegratorInfo> Multirate<B> {
    /// Largest supported number of levels below the coarsest (a time-step
    /// ratio of 2¹⁶).
    pub const MAX_LEVELS: usize = MAX_LEVELS;

    /// [`Self::new`] with a base chosen at run time, e.g. a
    /// [`StandardIntegrator`](crate::time::StandardIntegrator).
    ///
    /// # Panics
    /// If `max_levels` exceeds [`Self::MAX_LEVELS`], or `base` has no
    /// [`SspScheme`] to step elements with.
    pub fn with_base(base: B, max_levels: usize) -> Self {
        assert!(max_levels <= MAX_LEVELS, "at most {MAX_LEVELS} levels");
        assert!(
            base.ssp_scheme().is_some(),
            "{} has no SSP scheme for local time stepping",
            base.name()
        );
        Self { base, max_levels }
    }

    /// The largest number of levels below the coarsest.
    pub fn max_levels(&self) -> usize {
        self.max_levels
    }
}

impl<B: IntegratorInfo> IntegratorInfo for Multirate<B> {
    fn name(&self) -> &'static str {
        match self.base.ssp_scheme() {
            Some(SspScheme::Rk3) => "multirate-ssp-rk3",
            Some(SspScheme::Rk43) => "multirate-ssp-rk43",
            None => "multirate",
        }
    }

    /// Second order across level interfaces (the base's inside a level).
    fn order(&self) -> usize {
        2
    }

    fn n_stages(&self) -> usize {
        self.base.n_stages()
    }

    fn is_ssp(&self) -> bool {
        true
    }

    fn stage_times(&self, dt: f64) -> Vec<f64> {
        self.base.stage_times(dt)
    }

    fn max_local_levels(&self) -> Option<usize> {
        Some(self.max_levels)
    }

    fn ssp_scheme(&self) -> Option<SspScheme> {
        self.base.ssp_scheme()
    }
}

/// Without a physics module there are no elements: global steps of `B`.
impl<S: Integrable, B: TimeIntegrator<S>> TimeIntegrator<S> for Multirate<B> {
    fn step_with_relaxation<F, R, H>(
        &self,
        state: &mut S,
        dt: f64,
        t: f64,
        rhs: F,
        relax: R,
        stage_hook: H,
        workspace: &mut StageWorkspace<S>,
    ) where
        F: FnMut(&S, f64, &mut S),
        R: FnMut(&mut S, &S, f64),
        H: FnMut(&mut S),
    {
        self.base
            .step_with_relaxation(state, dt, t, rhs, relax, stage_hook, workspace);
    }
}

// =============================================================================
// Levels
// =============================================================================

/// Levels of the elements for element time steps `element_dt`: the coarse
/// step `Δt = 2^L·min(element_dt)`, with L the largest level count up to
/// `max_levels` that keeps `Δt ≤ max(element_dt)` and `Δt ≤ dt_cap`, and
/// `levels[k]` the smallest ℓ ≤ L with `Δt/2^ℓ ≤ element_dt[k]`.
///
/// Returns `(Δt, L)`; `(∞, 0)` when no element limits the step.
pub fn assign_levels(
    element_dt: &[f64],
    max_levels: usize,
    dt_cap: Option<f64>,
    levels: &mut Vec<u8>,
) -> (f64, usize) {
    assert!(max_levels <= MAX_LEVELS);
    let finite = element_dt.iter().copied().filter(|dt| dt.is_finite());
    let (dt_min, dt_max) = finite.fold((f64::INFINITY, 0.0_f64), |(lo, hi), dt| {
        (lo.min(dt), hi.max(dt))
    });
    levels.clear();
    levels.resize(element_dt.len(), 0);
    if !dt_min.is_finite() {
        return (f64::INFINITY, 0);
    }

    let mut finest = 0;
    while finest < max_levels {
        // Powers of two scale exactly: level L steps exactly dt_min
        let coarser = dt_min * (2u64 << finest) as f64;
        if coarser > dt_max || dt_cap.is_some_and(|cap| coarser > cap) {
            break;
        }
        finest += 1;
    }
    let dt = dt_min * (1u64 << finest) as f64;

    for (level, &dt_k) in levels.iter_mut().zip(element_dt) {
        let (mut l, mut dt_l) = (0, dt);
        while l < finest && dt_l > dt_k {
            dt_l *= 0.5;
            l += 1;
        }
        *level = l as u8;
    }
    (dt, finest)
}

/// `to[k] = max(from[k], from[j] − drop)` over the face neighbours j of k;
/// whether any `to[k]` differs from `from[k]`.
fn spread_to_neighbours(mesh: &Mesh2D, from: &[u8], to: &mut [u8], drop: u8) -> bool {
    let element = |(k, out): (usize, &mut u8)| {
        let k_idx = ElementIndex::new(k);
        *out = (0..4)
            .filter_map(|face| mesh.neighbor(k_idx, face))
            .map(|nb| from[nb.element].saturating_sub(drop))
            .fold(from[k], u8::max);
        *out != from[k]
    };
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        to.par_iter_mut()
            .enumerate()
            .map(element)
            .reduce(|| false, |a, b| a | b)
    }
    #[cfg(not(feature = "parallel"))]
    to.iter_mut()
        .enumerate()
        .map(element)
        .fold(false, |a, b| a | b)
}

/// `to[k]` = the largest `from` over k and its `stencil`.
fn spread_over_stencil(mesh: &Mesh2D, from: &[u8], to: &mut [u8], stencil: RhsStencil) {
    match stencil {
        RhsStencil::Faces => {
            spread_to_neighbours(mesh, from, to, 0);
        }
        RhsStencil::FacesAndCorners => {
            let element = |(k, out): (usize, &mut u8)| {
                let mut max = from[k];
                for face in 0..4 {
                    let Some(nb) = mesh.neighbor(ElementIndex::new(k), face) else {
                        continue;
                    };
                    max = max.max(from[nb.element]);
                    // Across the faces of the neighbour adjacent to the
                    // shared one (faces are numbered around the element)
                    let nb_idx = ElementIndex::new(nb.element);
                    for side in [1, 3] {
                        if let Some(corner) = mesh.neighbor(nb_idx, (nb.face + side) % 4) {
                            max = max.max(from[corner.element]);
                        }
                    }
                }
                *out = max;
            };
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                to.par_iter_mut().enumerate().for_each(element);
            }
            #[cfg(not(feature = "parallel"))]
            to.iter_mut().enumerate().for_each(element);
        }
        RhsStencil::VertexPatches => {
            let element = |(k, out): (usize, &mut u8)| {
                *out = mesh
                    .element_vertex_indices(ElementIndex::new(k))
                    .iter()
                    .flat_map(|&vertex| mesh.elements_at_vertex(vertex))
                    .map(|&e| from[e])
                    .fold(from[k], u8::max);
            };
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                to.par_iter_mut().enumerate().for_each(element);
            }
            #[cfg(not(feature = "parallel"))]
            to.iter_mut().enumerate().for_each(element);
        }
    }
}

/// Raise levels until face neighbours differ by at most one (the coarser
/// side is refined; smaller steps stay stable). An element then sees at most
/// two neighbour substeps per substep of its own, so water from a neighbour
/// cannot outrun its time step. `scratch` is resized as needed.
fn limit_level_jumps(mesh: &Mesh2D, level: &mut Vec<u8>, scratch: &mut Vec<u8>) {
    scratch.resize(level.len(), 0);
    while spread_to_neighbours(mesh, level, scratch, 1) {
        std::mem::swap(level, scratch);
    }
}

/// `lists[d]` = the elements with `key ≥ d`, in increasing order, for `d`
/// in `0..=max_key`.
fn at_or_above(key: &[u8], max_key: usize, lists: &mut Vec<Vec<u32>>) {
    lists.resize_with(max_key + 1, Vec::new);
    lists.truncate(max_key + 1);
    for (d, list) in lists.iter_mut().enumerate() {
        list.clear();
        list.extend(
            key.iter()
                .enumerate()
                .filter(|&(_, &r)| r as usize >= d)
                .map(|(k, _)| k as u32),
        );
    }
}

/// Level of finest substep p's alignment: the coarsest level whose substeps
/// start at p (0 at the start and end of the coarse step).
#[inline]
fn depth(p: usize, finest: usize) -> usize {
    if p == 0 || p == 1 << finest {
        0
    } else {
        finest - p.trailing_zeros() as usize
    }
}

// =============================================================================
// Stepper
// =============================================================================

/// Work counts of a multirate run.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MultirateStats {
    /// Coarse steps taken.
    pub steps: u64,
    /// Element RHS evaluations.
    pub element_evaluations: u64,
    /// Element RHS evaluations that global steps of the base scheme at the
    /// finest time step would have needed for the same steps.
    pub global_evaluations: u64,
    /// Finest level used (the largest time-step ratio is 2^this).
    pub finest_level: usize,
}

impl MultirateStats {
    /// Saving in RHS work against global stepping at the finest time step.
    pub fn speedup(&self) -> f64 {
        self.global_evaluations as f64 / self.element_evaluations.max(1) as f64
    }
}

/// Levels, stage storage and work counts of a multirate run; one per
/// simulation (`Simulation` keeps it).
pub struct MultirateStepper<S> {
    scheme: SspScheme,
    max_levels: usize,
    /// Finest level L of the current coarse step
    finest: usize,
    element_dt: Vec<f64>,
    level: Vec<u8>,
    /// A rate's spreading buffer
    scratch: Vec<u8>,
    /// `rate[s][k]`: the level whose substeps element k's stage-(s + 1) RHS
    /// follows, the finest level within s + 1 steps of the RHS stencil (one
    /// per stage)
    rate: Vec<Vec<u8>>,
    /// `active[s][d]`: the elements with `rate[s] ≥ d`, the active elements of
    /// stage s at a finest substep of depth d; `ending[d]`: those with a
    /// level ≥ d, whose substeps end at depth d. In increasing order, so the
    /// passes over most of the mesh read it in memory order
    active: Vec<Vec<Vec<u32>>>,
    ending: Vec<Vec<u32>>,
    /// Element RHS evaluations of one coarse step with the current levels
    evaluations_per_step: u64,
    /// Stage values 1 to s − 1 (the RHS arguments of stages 2 to s)
    values: Vec<S>,
    /// The last stage's value in the current finest substep
    last: Option<S>,
    acc: Option<S>,
    stats: MultirateStats,
}

impl<S: Integrable> MultirateStepper<S> {
    /// Empty stepper for `scheme` with up to `max_levels` levels below the
    /// coarsest.
    pub fn new(max_levels: usize, scheme: SspScheme) -> Self {
        assert!(max_levels <= MAX_LEVELS);
        let n_stages = scheme.n_stages();
        Self {
            scheme,
            max_levels,
            finest: 0,
            element_dt: Vec::new(),
            level: Vec::new(),
            scratch: Vec::new(),
            rate: vec![Vec::new(); n_stages],
            active: vec![Vec::new(); n_stages],
            ending: Vec::new(),
            evaluations_per_step: 0,
            values: Vec::new(),
            last: None,
            acc: None,
            stats: MultirateStats::default(),
        }
    }

    /// Assign levels from the current element time steps at the linear CFL
    /// `cfl` (with the forward-Euler bounds of `local` for the scheme), and
    /// return the coarse step (see [`assign_levels`]; at
    /// most `dt_cap`). Call it before every [`Self::step`].
    pub fn assign_levels(
        &mut self,
        local: &dyn LocalTimeStepping<S>,
        mesh: &Mesh2D,
        state: &S,
        cfl: f64,
        dt_cap: Option<f64>,
    ) -> f64 {
        let mut element_dt = std::mem::take(&mut self.element_dt);
        element_dt.resize(mesh.n_elements, 0.0);
        local.element_dt(state, cfl, self.scheme, &mut element_dt);
        let dt = self.assign_levels_from(&element_dt, mesh, &[local.stencil()], dt_cap);
        self.element_dt = element_dt;
        dt
    }

    /// [`Self::assign_levels`] from given element time steps, for a stage
    /// value of an element that reads the elements of `stencils` applied in
    /// turn (e.g. an RHS's, then a limiter's on the combination).
    pub fn assign_levels_from(
        &mut self,
        element_dt: &[f64],
        mesh: &Mesh2D,
        stencils: &[RhsStencil],
        dt_cap: Option<f64>,
    ) -> f64 {
        let n = mesh.n_elements;
        assert_eq!(element_dt.len(), n, "one time step per element");
        assert!(
            !stencils.is_empty(),
            "a stage reads at least its RHS stencil"
        );
        let (dt, finest) = assign_levels(element_dt, self.max_levels, dt_cap, &mut self.level);
        self.finest = finest;

        for rate in &mut self.rate {
            rate.resize(n, 0);
        }
        let mut scratch = std::mem::take(&mut self.scratch);
        scratch.resize(n, 0);
        limit_level_jumps(mesh, &mut self.level, &mut self.rate[0]);
        for s in 0..self.rate.len() {
            let (done, rest) = self.rate.split_at_mut(s);
            let from = if s == 0 { &self.level } else { &done[s - 1] };
            let to = &mut rest[0];
            to.copy_from_slice(from);
            for &stencil in stencils {
                spread_over_stencil(mesh, to, &mut scratch, stencil);
                std::mem::swap(to, &mut scratch);
            }
        }
        self.scratch = scratch;
        self.sort_by_rates();
        dt
    }

    /// Element RHS evaluations of a global step of the scheme at the finest
    /// level over those of a coarse step with the current levels and rates:
    /// the speedup the last [`Self::assign_levels`] plans.
    pub fn planned_speedup(&self) -> f64 {
        let global = (self.rate.len() * self.level.len()) as u64 * (1u64 << self.finest);
        global as f64 / self.evaluations_per_step.max(1) as f64
    }

    /// Put all `n_elements` elements on one level, for global steps of the
    /// scheme through the fused per-element stages (no element time steps
    /// needed: the caller takes the global step). Call it once; the levels
    /// stay until the next [`Self::assign_levels`].
    pub fn assign_one_level(&mut self, n_elements: usize) {
        self.finest = 0;
        self.level.clear();
        self.level.resize(n_elements, 0);
        for rate in &mut self.rate {
            rate.clear();
            rate.resize(n_elements, 0);
        }
        self.sort_by_rates();
    }

    /// The element orders and the work count of the current levels and rates.
    fn sort_by_rates(&mut self) {
        let finest = self.finest;
        self.evaluations_per_step = self.rate.iter().flatten().map(|&r| 1u64 << r).sum();
        for (lists, key) in self
            .active
            .iter_mut()
            .chain(std::iter::once(&mut self.ending))
            .zip(self.rate.iter().chain(std::iter::once(&self.level)))
        {
            at_or_above(key, finest, lists);
        }
    }

    /// Level of every element from the last [`Self::assign_levels`].
    pub fn levels(&self) -> &[u8] {
        &self.level
    }

    /// Finest level of the last [`Self::assign_levels`].
    pub fn finest_level(&self) -> usize {
        self.finest
    }

    /// Work counts so far.
    pub fn stats(&self) -> MultirateStats {
        self.stats
    }

    /// The finest level whose substeps the stage-`stage` RHS of every element
    /// follows (0-based stages): its own level, refined by its RHS stencil's
    /// within `stage + 1` steps. An evaluation of element k's stage `stage`
    /// stands for `2^(L − rates(stage)[k])` of the `2^L` finest substeps.
    pub fn rates(&self, stage: usize) -> &[u8] {
        &self.rate[stage]
    }

    /// The passes of one coarse step of `dt` from time `t`, with the levels
    /// of the last [`Self::assign_levels`], in order, to `pass`: for every
    /// finest substep, each stage of the elements whose RHS changes there,
    /// then the end of the substeps that end there.
    ///
    /// A stage writes its value into the buffer of its stage (the RHS
    /// argument of the next stage; stage 0 reads the substep start), the
    /// last into a buffer of its own, summed into the substep's result as
    /// its [`ElementStage::accumulate`] says. [`MultiratePass::Finish`] then
    /// makes the result the elements' state. [`Self::step`] is this schedule
    /// on a [`LocalTimeStepping`] module; the mode splitter's barotropic pass
    /// runs it with its own element stages.
    pub fn schedule(&self, t: f64, dt: f64, mut pass: impl FnMut(MultiratePass<'_>)) {
        let finest = self.finest;
        let stages = self.scheme.stages();
        let n_stages = stages.len();
        let (level, rate) = (&self.level, &self.rate);
        let step_of = |l: u8| dt / (1u64 << l) as f64;
        for p in 0..1usize << finest {
            let d = depth(p, finest);
            // Start time and step of element k's current substep
            let substep = |k: usize| {
                let l = level[k];
                let dt_l = step_of(l);
                let q = p >> (finest - l as usize);
                (t + q as f64 * dt_l, dt_l)
            };
            for (s, stage) in stages.iter().enumerate() {
                // Stage s + 1 = start·S + input·Yₛ + βΔt_ℓ F(Yₛ), with Y₀ = S
                let plan = |k: usize| {
                    let (t_k, dt_l) = substep(k);
                    ElementStage {
                        time: t_k + stage.c * dt_l,
                        a: stage.start,
                        b: stage.input,
                        c: stage.beta * dt_l,
                        accumulate: None,
                    }
                };
                // The elements whose stage-(s + 1) RHS follows a level ≥ d
                let elements = &self.active[s][d];
                if s + 1 < n_stages {
                    pass(MultiratePass::Stage {
                        stage: s,
                        elements,
                        plan: &plan,
                    });
                    continue;
                }
                // The last stage, H: the substep result is the mean of H over
                // its 2^(r − ℓ) blocks; a block starting with the substep
                // restarts the sum
                pass(MultiratePass::Stage {
                    stage: s,
                    elements,
                    plan: &|k| {
                        let blocks = 1u64 << (rate[s][k] - level[k]);
                        let first = level[k] as usize >= d;
                        ElementStage {
                            accumulate: Some((1.0 / blocks as f64, if first { 0.0 } else { 1.0 })),
                            ..plan(k)
                        }
                    },
                });
            }
            // Substeps ending here
            let next = depth(p + 1, finest);
            pass(MultiratePass::Finish {
                elements: &self.ending[next],
            });
        }
    }

    /// One coarse step of `dt` from time `t`, with the levels of the last
    /// [`Self::assign_levels`]. `dt` may be shorter than the step it returned
    /// (all levels shrink with it).
    pub fn step(&mut self, local: &dyn LocalTimeStepping<S>, state: &mut S, t: f64, dt: f64) {
        let n_stages = self.scheme.n_stages();
        if self.values.len() != n_stages - 1 {
            self.values = vec![state.clone(); n_stages - 1];
        }
        let mut values = std::mem::take(&mut self.values);
        let mut last = self.last.take().unwrap_or_else(|| state.clone());
        let mut acc = self.acc.take().unwrap_or_else(|| state.clone());
        self.schedule(t, dt, |pass| match pass {
            MultiratePass::Stage {
                stage: s,
                elements,
                plan,
            } => {
                let (done, rest) = values.split_at_mut(s);
                let input: &S = if s == 0 { &*state } else { &done[s - 1] };
                if s + 1 < n_stages {
                    local.stage_where(state, input, elements, plan, &mut rest[0], None);
                } else {
                    local.stage_where(state, input, elements, plan, &mut last, Some(&mut acc));
                }
            }
            MultiratePass::Finish { elements } => local.finish_where(state, &acc, elements),
        });
        (self.values, self.last, self.acc) = (values, Some(last), Some(acc));
        self.count_step();
    }

    /// Count one coarse step of a [`Self::schedule`] run by the caller.
    pub fn count_step(&mut self) {
        let n_stages = self.scheme.n_stages() as u64;
        self.stats.steps += 1;
        self.stats.element_evaluations += self.evaluations_per_step;
        self.stats.global_evaluations += n_stages * self.level.len() as u64 * (1u64 << self.finest);
        self.stats.finest_level = self.stats.finest_level.max(self.finest);
    }
}

/// One pass of [`MultirateStepper::schedule`].
pub enum MultiratePass<'a> {
    /// Stage `stage` (0-based) of the listed (distinct) elements, element k
    /// as `plan(k)` says (its time, its Shu–Osher weights, and for the last
    /// stage how it is summed into the substep's result).
    Stage {
        stage: usize,
        elements: &'a [u32],
        plan: &'a (dyn Fn(usize) -> ElementStage + Sync),
    },
    /// The substeps of the listed elements end: their result is their state.
    Finish { elements: &'a [u32] },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels_are_powers_of_two_below_the_coarse_step() {
        let element_dt = [1.0, 0.3, 0.26, 0.1, f64::INFINITY, 0.55];
        let mut levels = Vec::new();
        let (dt, finest) = assign_levels(&element_dt, 8, None, &mut levels);
        // 0.1·2³ = 0.8 ≤ 1.0 < 1.6
        assert_eq!(finest, 3);
        assert_eq!(dt, 0.8);
        assert_eq!(levels, [0, 2, 2, 3, 0, 1]);
        for (&l, &dt_k) in levels.iter().zip(&element_dt) {
            let dt_l = dt / (1 << l) as f64;
            assert!(dt_l <= dt_k, "level {l} steps {dt_l} > {dt_k}");
            // and it is the coarsest such level
            assert!(l == 0 || 2.0 * dt_l > dt_k);
        }
    }

    #[test]
    fn levels_respect_the_caps() {
        let element_dt = [1.0, 0.1];
        let mut levels = Vec::new();
        assert_eq!(assign_levels(&element_dt, 2, None, &mut levels), (0.4, 2));
        assert_eq!(levels, [0, 2]);
        assert_eq!(
            assign_levels(&element_dt, 8, Some(0.25), &mut levels),
            (0.2, 1)
        );
        assert_eq!(levels, [0, 1]);
        assert_eq!(assign_levels(&element_dt, 0, None, &mut levels), (0.1, 0));
        assert_eq!(levels, [0, 0]);
        let dry = [f64::INFINITY; 3];
        assert_eq!(
            assign_levels(&dry, 4, None, &mut levels),
            (f64::INFINITY, 0)
        );
        assert_eq!(levels, [0, 0, 0]);
    }

    #[test]
    fn lists_hold_the_elements_at_or_above_each_key_in_order() {
        let key = [0u8, 2, 1, 2, 0, 3];
        let mut lists = Vec::new();
        at_or_above(&key, 3, &mut lists);
        assert_eq!(
            lists,
            [
                vec![0, 1, 2, 3, 4, 5],
                vec![1, 2, 3, 5],
                vec![1, 3, 5],
                vec![5]
            ]
        );
        at_or_above(&key[..2], 1, &mut lists);
        assert_eq!(lists, [vec![0, 1], vec![1]]);
    }

    #[test]
    fn depth_is_the_coarsest_aligned_level() {
        let depths: Vec<_> = (0..=8).map(|p| depth(p, 3)).collect();
        assert_eq!(depths, [0, 3, 2, 3, 1, 3, 2, 3, 0]);
        assert_eq!(depth(0, 0), 0);
        assert_eq!(depth(1, 0), 0);
    }

    #[test]
    fn rates_spread_one_face_per_stage() {
        // A row of 6 elements; only the last one is on level 2
        let mesh = Mesh2D::uniform_rectangle(0.0, 6.0, 0.0, 1.0, 6, 1);
        let level = [0, 0, 0, 0, 0, 2];
        let mut r1 = vec![0; 6];
        let mut r2 = vec![0; 6];
        let mut r3 = vec![0; 6];
        spread_to_neighbours(&mesh, &level, &mut r1, 0);
        spread_to_neighbours(&mesh, &r1, &mut r2, 0);
        spread_to_neighbours(&mesh, &r2, &mut r3, 0);
        assert_eq!(r1, [0, 0, 0, 0, 2, 2]);
        assert_eq!(r2, [0, 0, 0, 2, 2, 2]);
        assert_eq!(r3, [0, 0, 2, 2, 2, 2]);
    }

    /// The viscous stencil of an element on a structured mesh is the 3 × 3
    /// block around it: face neighbours and diagonals, not two faces away.
    #[test]
    fn corner_stencil_is_the_block_around_an_element() {
        // 5 × 5 elements, row-major from the bottom left; the centre on level 1
        let mesh = Mesh2D::uniform_rectangle(0.0, 5.0, 0.0, 5.0, 5, 5);
        let at = |i: usize, j: usize| j * 5 + i;
        let mut level = vec![0u8; 25];
        level[at(2, 2)] = 1;
        let mut rate = vec![0u8; 25];
        spread_over_stencil(&mesh, &level, &mut rate, RhsStencil::FacesAndCorners);
        for j in 0..5_usize {
            for i in 0..5_usize {
                let in_block = i.abs_diff(2) <= 1 && j.abs_diff(2) <= 1;
                assert_eq!(rate[at(i, j)], in_block as u8, "element ({i}, {j})");
            }
        }
        spread_over_stencil(&mesh, &level, &mut rate, RhsStencil::Faces);
        for j in 0..5_usize {
            for i in 0..5_usize {
                let in_cross = i.abs_diff(2) + j.abs_diff(2) <= 1;
                assert_eq!(rate[at(i, j)], in_cross as u8, "element ({i}, {j})");
            }
        }
        // On a structured mesh the vertex patches are the same 3 × 3 block
        spread_over_stencil(&mesh, &level, &mut rate, RhsStencil::VertexPatches);
        for j in 0..5_usize {
            for i in 0..5_usize {
                let in_block = i.abs_diff(2) <= 1 && j.abs_diff(2) <= 1;
                assert_eq!(rate[at(i, j)], in_block as u8, "element ({i}, {j})");
            }
        }
    }

    /// A stage that reads two stencils in turn (an RHS's, then a limiter's)
    /// spreads its rate over both: one stencil step each, every stage.
    #[test]
    fn rates_spread_over_every_stencil_of_a_stage() {
        let mesh = Mesh2D::uniform_rectangle(0.0, 8.0, 0.0, 1.0, 8, 1);
        let element_dt = [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 2.0];
        let mut one = MultirateStepper::<crate::solver::DGSolution2D>::new(2, SspScheme::Rk3);
        one.assign_levels_from(&element_dt, &mesh, &[RhsStencil::Faces], None);
        assert_eq!(one.levels(), [0, 0, 0, 0, 0, 0, 0, 1]);
        assert_eq!(one.rates(0), [0, 0, 0, 0, 0, 0, 1, 1]);
        assert_eq!(one.rates(2), [0, 0, 0, 0, 1, 1, 1, 1]);
        let mut two = MultirateStepper::<crate::solver::DGSolution2D>::new(2, SspScheme::Rk3);
        let stencils = [RhsStencil::Faces, RhsStencil::Faces];
        two.assign_levels_from(&element_dt, &mesh, &stencils, None);
        assert_eq!(two.rates(0), [0, 0, 0, 0, 0, 1, 1, 1]);
        assert_eq!(two.rates(2), [0, 1, 1, 1, 1, 1, 1, 1]);
        // 3 stages × 8 elements × 2 finest substeps against the evaluations
        assert_eq!(one.planned_speedup(), 48.0 / (3.0 * 8.0 + 2.0 + 3.0 + 4.0));
        assert_eq!(two.planned_speedup(), 48.0 / (3.0 * 8.0 + 3.0 + 5.0 + 7.0));
    }

    #[test]
    fn level_jumps_are_limited_to_one() {
        let mesh = Mesh2D::uniform_rectangle(0.0, 6.0, 0.0, 1.0, 6, 1);
        let mut level = vec![0, 0, 0, 0, 0, 3];
        let mut scratch = Vec::new();
        limit_level_jumps(&mesh, &mut level, &mut scratch);
        assert_eq!(level, [0, 0, 0, 1, 2, 3]);
        let mut level = vec![2, 0, 0, 0, 1, 0];
        limit_level_jumps(&mesh, &mut level, &mut scratch);
        assert_eq!(level, [2, 1, 0, 0, 1, 0]);
    }
}

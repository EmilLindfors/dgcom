//! Multirate (local) time stepping: each element advances with its own
//! stable time step, a power-of-two fraction of the global one.
//!
//! On a locally refined mesh (a fish farm at 20 m inside a kilometre-scale
//! domain, fjord arms, deep channels next to shallow banks), global explicit
//! stepping lets the few smallest or fastest elements set the time step of
//! all. Here element k gets a level ℓₖ ∈ 0..=L and takes 2^ℓₖ substeps of
//! Δt_ℓ = Δt/2^ℓ per (coarse) step Δt, with Δt_L at most the smallest element
//! time step.
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
//! maximum) that keep it at most the largest one (and the module's bound,
//! below; a run that must land on a time takes a shorter one, further below).
//! Neighbouring levels are then made to differ by at most one, refining the
//! coarser side. Together these
//! keep water that floods in from a neighbour within one substep from
//! outrunning an element's time step (a nearly dry element otherwise gets an
//! enormous one). Levels are reassigned every coarse step.
//!
//! The levels are those of the coarse step's start, and they hold only while
//! the element time steps do. A film of water on an element that has just
//! flooded has an enormous time step of its own (its waves are slow), and
//! as the largest one it would stretch the coarse step to the most levels
//! allowed, minutes on a beach whose water steps in seconds, with the levels
//! frozen while the front runs up it. So the module bounds the coarse step
//! ([`LocalTimeStepping::coarse_step_bound`]: the largest step of an element
//! in water, not in a film).
//!
//! A run that must reach a time (an output, the end) takes equal coarse steps
//! to it, with the levels of that step ([`MultirateStepper::assign_levels`]'s
//! `landing`): an element's level is the coarsest whose step it can take,
//! whatever the coarse step is. Shortening the last coarse step alone would
//! shrink every level with it, and a 9 s step to an output would cost what an
//! 84 s one does: on the Chiloé tide, with eight steps to each ten-minute
//! output where 7.1 fit, an eighth of the run.
//!
//! The same holds for a cap on the step that lies below most elements' own
//! steps (a 3D run's `dt_max` of a minute where the internal waves allow
//! minutes): a power of two times the smallest element step is anywhere
//! between half the cap and the cap, and every element that could step at
//! the cap steps at that. [`MultirateStepper::with_longest_coarse_step`]
//! takes the longest coarse step the cap, the largest element step and the
//! level count allow, with that step's levels.
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
//! **Line-local elements.** One stencil step per stage is the most a stage
//! can spread, not what a DGSEM stage does. On GLL nodes with the diagonal
//! LIFT, a face's flux enters only the face's own nodes, and the volume term
//! of a node reads the two GLL lines through it. A finer neighbour therefore
//! changes a coarse element's first stage value on the shared face only, the
//! whole element one stage later, and the element behind it (through the
//! opposite face, whose nodes the first stage left alone) only in the stage
//! after that: two stages per element across a level interface, not one.
//! Where the physics module marks such elements
//! ([`LocalTimeStepping::line_local`]) the rates are spread node by node
//! (`spread_over_lines`), and an element's stage follows the finest rate
//! any of its nodes reads. Normal to an interface SSP-RK(4,3)'s four stages
//! then cost 1 + 1 + 2 + 2 element evaluations at the finer rate, against
//! 1 + 2 + 3 + 4. The evaluations left out are the ones whose inputs are the
//! same to the bit.
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

/// The tensor-product node layout of the elements, for the stage rates of
/// [line-local](LocalTimeStepping::line_local) elements.
#[derive(Clone, Copy, Debug)]
pub struct NodeLines<'a> {
    /// Nodes per GLL line (at least 2): node `j·n_1d + i` lies on r-line `j`
    /// and s-line `i`.
    pub n_1d: usize,
    /// The nodes of the four faces (`DGOperators2D::face_nodes`): node `fi`
    /// of a face coincides with node `n_1d − 1 − fi` of the neighbour's.
    pub face_nodes: &'a [Vec<usize>; 4],
}

impl NodeLines<'_> {
    /// The node one in from `node` of `face`, normal to the face.
    #[inline]
    fn inner(&self, face: usize, node: usize) -> usize {
        match face {
            0 => node + self.n_1d,
            1 => node - 1,
            2 => node - self.n_1d,
            _ => node + 1,
        }
    }
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

    /// Marks the *line-local* elements of the coming coarse step, which
    /// starts from `state`, in `out` (one flag per mesh element, all `false`
    /// on entry) and returns the elements' node layout; `None` (the default)
    /// if no element is. Asked only with the [`RhsStencil::Faces`] stencil.
    ///
    /// The stage value of a line-local element at one of its nodes reads, of
    /// the stage's input, the element's own nodes on the two GLL lines
    /// through the node and, at a face node, the neighbour's coincident node;
    /// nothing else (the base value and the time are the element's own).
    /// That is flux differencing on GLL nodes with a diagonal LIFT and a
    /// pointwise interface flux, finished node by node (stage combination,
    /// damping, post-processing). Every other element may read all of its
    /// own nodes and, of each face neighbour, the nodes of the shared face
    /// and the nodes one in from it. The flags must hold for every stage of
    /// the coarse step: the stage rates are spread by them
    /// (see the [module documentation](self)).
    fn line_local(&self, _state: &S, _out: &mut [bool]) -> Option<NodeLines<'_>> {
        None
    }

    /// The longest coarse step for which the levels assigned from `state`
    /// (its element time steps `element_dt`) hold; `∞` (the default) if
    /// nothing bounds it beyond the element time steps themselves.
    ///
    /// The levels are frozen for a coarse step, so it must stay short
    /// against the time in which the element time steps change: with
    /// wetting and drying, the time a front takes to flood an element.
    fn coarse_step_bound(&self, _state: &S, _element_dt: &[f64]) -> f64 {
        f64::INFINITY
    }

    /// Stage values so far that broke [`Self::line_local`]'s promise: of an
    /// element marked line-local that was not by the time of the stage (a
    /// node fell through the depth where its kernel changes within the
    /// coarse step). Each may have left a neighbour's stage one substep
    /// behind; the count belongs in the run's report and should be zero.
    fn line_local_breaks(&self) -> u64 {
        0
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
    /// of its [`Self::stencil`], nothing further away (and, where
    /// [`Self::line_local`] marks elements, no more than it says: the nodes
    /// of the shared faces and one in from them, less for a line-local
    /// element); the new stage value
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

/// Levels of the elements for a coarse step `dt` that the finest of
/// `max_levels` levels can take: `levels[k]` the smallest ℓ with
/// `dt/2^ℓ ≤ element_dt[k]` (at most `max_levels`). Returns the finest level
/// used.
fn levels_for_step(element_dt: &[f64], dt: f64, max_levels: usize, levels: &mut [u8]) -> usize {
    let mut finest = 0;
    for (level, &dt_k) in levels.iter_mut().zip(element_dt) {
        let (mut l, mut dt_l) = (0, dt);
        while l < max_levels && dt_l > dt_k {
            dt_l *= 0.5;
            l += 1;
        }
        *level = l as u8;
        finest = finest.max(l);
    }
    finest
}

/// Relative slack in landing on a target time: a step that ends within this
/// fraction of itself before the target is stretched to land on it (ten
/// steps of 0.1 to 1.0 accumulate round-off, and the sliver left would cost
/// a whole step).
pub(crate) const LANDING_SLACK: f64 = 1e-9;

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

/// One stage's spreading of the node rates: `to` = the finest level each
/// node's new stage value follows, from the node rates `from` of the stage's
/// input (`n_1d²` per element), and `rate[k]` = the finest of element k's
/// nodes, the rate of its RHS in this stage.
///
/// A node of a [line-local](LocalTimeStepping::line_local) element reads its
/// two GLL lines and, on a face, the neighbour's coincident node; any other
/// element reads all of its own nodes and each neighbour's face nodes and the
/// nodes one in from them, and all its nodes follow.
fn spread_over_lines(
    mesh: &Mesh2D,
    lines: NodeLines<'_>,
    line_local: &[bool],
    from: &[u8],
    to: &mut [u8],
    rate: &mut [u8],
) {
    let n1 = lines.n_1d;
    let nn = n1 * n1;
    let element = |(k, (to, rate)): (usize, (&mut [u8], &mut u8))| {
        let own = &from[k * nn..(k + 1) * nn];
        let links = &mesh.face_links[k];
        if !line_local[k] {
            let mut finest = own.iter().copied().fold(0, u8::max);
            for link in links.iter().filter(|link| !link.is_boundary()) {
                let other = &from[link.element() * nn..][..nn];
                for &node in &lines.face_nodes[link.face()] {
                    finest = finest
                        .max(other[node])
                        .max(other[lines.inner(link.face(), node)]);
                }
            }
            to.fill(finest);
            *rate = finest;
            return;
        }
        for (to, own) in to.chunks_exact_mut(n1).zip(own.chunks_exact(n1)) {
            to.fill(own.iter().copied().fold(0, u8::max));
        }
        for i in 0..n1 {
            let finest = (0..n1).map(|j| own[j * n1 + i]).fold(0, u8::max);
            for j in 0..n1 {
                to[j * n1 + i] = to[j * n1 + i].max(finest);
            }
        }
        for (face, link) in links.iter().enumerate() {
            if link.is_boundary() {
                continue;
            }
            let other = &from[link.element() * nn..][..nn];
            let other_nodes = &lines.face_nodes[link.face()];
            for (fi, &node) in lines.face_nodes[face].iter().enumerate() {
                to[node] = to[node].max(other[other_nodes[n1 - 1 - fi]]);
            }
        }
        *rate = to.iter().copied().fold(0, u8::max);
    };
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        to.par_chunks_exact_mut(nn)
            .zip(rate.par_iter_mut())
            .enumerate()
            .for_each(element);
    }
    #[cfg(not(feature = "parallel"))]
    to.chunks_exact_mut(nn)
        .zip(rate.iter_mut())
        .enumerate()
        .for_each(element);
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
    /// Model time stepped (s).
    pub model_seconds: f64,
    /// Element RHS evaluations had every element stepped at its own stable
    /// step (`element_dt`): the least an explicit scheme of these stages
    /// needs.
    pub ideal_evaluations: f64,
    /// Element RHS evaluations had every element stepped at its level's step
    /// (power-of-two fractions of the coarse step, neighbours at most one
    /// level apart) and nothing else: what the stage rates' spreading at
    /// level interfaces adds is `element_evaluations` over this.
    pub level_evaluations: f64,
    /// [Line-local](LocalTimeStepping::line_local) elements, summed over
    /// the coarse steps.
    pub line_local_elements: u64,
    /// [`LocalTimeStepping::line_local_breaks`] at the last step.
    pub line_local_breaks: u64,
}

impl MultirateStats {
    /// Saving in RHS work against global stepping at the finest time step.
    pub fn speedup(&self) -> f64 {
        self.global_evaluations as f64 / self.element_evaluations.max(1) as f64
    }

    /// The run's work in a line: the element evaluations per second of
    /// model time, what of them the power-of-two levels and the stage rates'
    /// spreading at level interfaces add to every element stepping at its
    /// own step, and the line-local elements.
    pub fn describe(&self, n_elements: usize) -> String {
        let seconds = self.model_seconds.max(f64::MIN_POSITIVE);
        let scheduled = self.element_evaluations as f64;
        let mut line = format!(
            "local time stepping: {:.0} element evaluations per model second, {:.2}× less than \
             global steps (time-step ratio up to 2^{}); {:.2}× the {:.0} of every element at its \
             own step (levels ×{:.2}, the stage rates' spreading ×{:.2})",
            scheduled / seconds,
            self.speedup(),
            self.finest_level,
            scheduled / self.ideal_evaluations.max(f64::MIN_POSITIVE),
            self.ideal_evaluations / seconds,
            self.level_evaluations / self.ideal_evaluations.max(f64::MIN_POSITIVE),
            scheduled / self.level_evaluations.max(f64::MIN_POSITIVE),
        );
        if self.line_local_elements > 0 {
            let share = self.line_local_elements as f64
                / (self.steps.max(1) as f64 * n_elements.max(1) as f64);
            line.push_str(&format!(
                "; {:.0} % of the elements line-local, {} of their stage values broke it",
                100.0 * share,
                self.line_local_breaks
            ));
        }
        line
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
    /// follows, the finest level any of its inputs follows: within s + 1
    /// steps of the RHS stencil, or less far where the rates are spread node
    /// by node (one per stage)
    rate: Vec<Vec<u8>>,
    /// `active[s][d]`: the elements with `rate[s] ≥ d`, the active elements of
    /// stage s at a finest substep of depth d; `ending[d]`: those with a
    /// level ≥ d, whose substeps end at depth d. In increasing order, so the
    /// passes over most of the mesh read it in memory order
    active: Vec<Vec<Vec<u32>>>,
    ending: Vec<Vec<u32>>,
    /// Element RHS evaluations of one coarse step with the current levels
    evaluations_per_step: u64,
    /// The same without the stage rates' spreading (every stage at the
    /// element's level)
    level_evaluations_per_step: u64,
    /// Σ 1/element_dt over the elements with a finite step
    inverse_dt_sum: f64,
    /// The coarse step the elements allow (before fitting it to a landing)
    stable_step: f64,
    /// Whether the rates of line-local elements are spread node by node
    line_rates: bool,
    /// Whether the coarse step is the longest the bounds allow, not a power
    /// of two times the smallest element step
    longest_step: bool,
    /// The line-local elements of the current coarse step, and two buffers of
    /// node rates for their spreading
    line_local: Vec<bool>,
    node_rate: [Vec<u8>; 2],
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
            level_evaluations_per_step: 0,
            inverse_dt_sum: 0.0,
            stable_step: f64::INFINITY,
            line_rates: true,
            longest_step: false,
            line_local: Vec::new(),
            node_rate: [Vec::new(), Vec::new()],
            values: Vec::new(),
            last: None,
            acc: None,
            stats: MultirateStats::default(),
        }
    }

    /// Whether the stage rates of the module's
    /// [line-local](LocalTimeStepping::line_local) elements are spread node
    /// by node (the default). Without it every element's rate spreads one
    /// stencil per stage: more evaluations, each of whose results the
    /// node-by-node rates would have reused to the bit.
    pub fn with_line_local_rates(mut self, on: bool) -> Self {
        self.line_rates = on;
        self
    }

    /// Take the longest coarse step the bounds allow: the cap, the largest
    /// element step and `2^max_levels` times the smallest one, whichever is
    /// least, with every element on the coarsest level of that step it can
    /// take. Off (the default), the coarse step is a power of two times the
    /// smallest element step ([`assign_levels`]), which a cap leaves anywhere
    /// between half of itself and itself: where most elements could step at
    /// the cap or longer, they then step at up to half of it.
    pub fn with_longest_coarse_step(mut self, on: bool) -> Self {
        self.longest_step = on;
        self
    }

    /// Assign levels from the current element time steps at the linear CFL
    /// `cfl` (with the forward-Euler bounds of `local` for the scheme), and
    /// return the coarse step (see [`assign_levels`]; at
    /// most `dt_cap`). Call it before every [`Self::step`].
    ///
    /// `landing`: the time left to a time the run must reach. The coarse
    /// step is then that time over the fewest equal steps the elements allow,
    /// and the levels are that step's (see the
    /// [module documentation](self)); [`Self::stable_step`] stays the step
    /// the elements allow.
    pub fn assign_levels(
        &mut self,
        local: &dyn LocalTimeStepping<S>,
        mesh: &Mesh2D,
        state: &S,
        cfl: f64,
        dt_cap: Option<f64>,
        landing: Option<f64>,
    ) -> f64 {
        let mut element_dt = std::mem::take(&mut self.element_dt);
        element_dt.resize(mesh.n_elements, 0.0);
        local.element_dt(state, cfl, self.scheme, &mut element_dt);
        let mut line_local = std::mem::take(&mut self.line_local);
        line_local.clear();
        line_local.resize(mesh.n_elements, false);
        let stencil = local.stencil();
        let lines = (self.line_rates && stencil == RhsStencil::Faces)
            .then(|| local.line_local(state, &mut line_local))
            .flatten()
            .filter(|lines| lines.n_1d >= 2);
        let lines = lines.map(|lines| (lines, &line_local[..]));
        let bound = local.coarse_step_bound(state, &element_dt);
        let dt_cap = Some(dt_cap.map_or(bound, |cap| cap.min(bound))).filter(|cap| cap.is_finite());
        let dt = self.assign(&element_dt, mesh, &[stencil], dt_cap, landing, lines);
        (self.element_dt, self.line_local) = (element_dt, line_local);
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
        landing: Option<f64>,
    ) -> f64 {
        self.assign(element_dt, mesh, stencils, dt_cap, landing, None)
    }

    /// The coarse step the elements (and the cap) allowed at the last
    /// [`Self::assign_levels`], before it was fitted to a `landing`.
    pub fn stable_step(&self) -> f64 {
        self.stable_step
    }

    /// [`Self::assign_levels_from`]; with `lines` (the node layout and the
    /// [line-local](LocalTimeStepping::line_local) elements) the rates are
    /// spread node by node over the face stencil instead of over `stencils`.
    fn assign(
        &mut self,
        element_dt: &[f64],
        mesh: &Mesh2D,
        stencils: &[RhsStencil],
        dt_cap: Option<f64>,
        landing: Option<f64>,
        lines: Option<(NodeLines<'_>, &[bool])>,
    ) -> f64 {
        let n = mesh.n_elements;
        assert_eq!(element_dt.len(), n, "one time step per element");
        assert!(
            !stencils.is_empty(),
            "a stage reads at least its RHS stencil"
        );
        let (mut dt, mut finest) =
            assign_levels(element_dt, self.max_levels, dt_cap, &mut self.level);
        // The longest step the bounds allow, with its own levels: the finest
        // of them still steps at most the smallest element step
        if self.longest_step && dt.is_finite() {
            let largest = element_dt
                .iter()
                .copied()
                .filter(|dt| dt.is_finite())
                .fold(0.0, f64::max);
            let smallest = dt / (1u64 << finest) as f64;
            let longest = (smallest * (1u64 << self.max_levels) as f64)
                .min(largest)
                .min(dt_cap.unwrap_or(f64::INFINITY));
            if longest > dt {
                dt = longest;
                finest = levels_for_step(element_dt, dt, self.max_levels, &mut self.level);
            }
        }
        self.stable_step = dt;
        // Equal steps to the time to land on, each with its own levels (one
        // level has nothing to gain: its steps cost what they span)
        if let Some(left) = landing.filter(|&left| finest > 0 && left > 0.0 && dt.is_finite()) {
            let steps = (left / (dt * (1.0 + LANDING_SLACK))).ceil().max(1.0);
            let fitted = left / steps;
            if fitted < dt {
                dt = fitted;
                finest = levels_for_step(element_dt, dt, self.max_levels, &mut self.level);
            }
        }
        self.finest = finest;
        self.inverse_dt_sum = element_dt
            .iter()
            .filter(|dt| dt.is_finite())
            .map(|dt| 1.0 / dt)
            .sum();

        for rate in &mut self.rate {
            rate.resize(n, 0);
        }
        let mut scratch = std::mem::take(&mut self.scratch);
        scratch.resize(n, 0);
        limit_level_jumps(mesh, &mut self.level, &mut self.rate[0]);
        self.level_evaluations_per_step =
            self.rate.len() as u64 * self.level.iter().map(|&l| 1u64 << l).sum::<u64>();
        if let Some((lines, line_local)) = lines {
            assert_eq!(line_local.len(), n, "one flag per element");
            let nn = lines.n_1d * lines.n_1d;
            let [mut from, mut to] = std::mem::take(&mut self.node_rate);
            from.resize(n * nn, 0);
            to.resize(n * nn, 0);
            // The substep start follows the element's own level at every node
            for (nodes, &level) in from.chunks_exact_mut(nn).zip(&self.level) {
                nodes.fill(level);
            }
            for rate in &mut self.rate {
                spread_over_lines(mesh, lines, line_local, &from, &mut to, rate);
                std::mem::swap(&mut from, &mut to);
            }
            self.node_rate = [from, to];
        } else {
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
        }
        self.scratch = scratch;
        self.sort_by_rates();
        dt
    }

    /// Element RHS evaluations per second of model time for a coarse step
    /// of `dt` with the levels of the last [`Self::assign_levels`], as
    /// `(ideal, levels, scheduled)`: every element at its own stable step;
    /// at its level's step; and what the stage rates schedule
    /// (see [`MultirateStats`]).
    pub fn planned_work(&self, dt: f64) -> (f64, f64, f64) {
        (
            self.rate.len() as f64 * self.inverse_dt_sum,
            self.level_evaluations_per_step as f64 / dt,
            self.evaluations_per_step as f64 / dt,
        )
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
        self.level_evaluations_per_step = (self.rate.len() * n_elements) as u64;
        self.inverse_dt_sum = 0.0;
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
    /// within `stage + 1` steps (by the levels its nodes read, where the
    /// rates are spread node by node). An evaluation of element k's stage `stage`
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
        self.count_step(dt);
        self.stats.line_local_breaks = local.line_local_breaks();
    }

    /// Count one coarse step of `dt` of a [`Self::schedule`] run by the
    /// caller.
    pub fn count_step(&mut self, dt: f64) {
        let n_stages = self.scheme.n_stages() as u64;
        self.stats.steps += 1;
        self.stats.element_evaluations += self.evaluations_per_step;
        self.stats.global_evaluations += n_stages * self.level.len() as u64 * (1u64 << self.finest);
        self.stats.finest_level = self.stats.finest_level.max(self.finest);
        self.stats.model_seconds += dt;
        self.stats.ideal_evaluations += n_stages as f64 * self.inverse_dt_sum * dt;
        self.stats.level_evaluations += self.level_evaluations_per_step as f64;
        self.stats.line_local_elements += self.line_local.iter().filter(|&&l| l).count() as u64;
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
        one.assign_levels_from(&element_dt, &mesh, &[RhsStencil::Faces], None, None);
        assert_eq!(one.levels(), [0, 0, 0, 0, 0, 0, 0, 1]);
        assert_eq!(one.rates(0), [0, 0, 0, 0, 0, 0, 1, 1]);
        assert_eq!(one.rates(2), [0, 0, 0, 0, 1, 1, 1, 1]);
        let mut two = MultirateStepper::<crate::solver::DGSolution2D>::new(2, SspScheme::Rk3);
        let stencils = [RhsStencil::Faces, RhsStencil::Faces];
        two.assign_levels_from(&element_dt, &mesh, &stencils, None, None);
        assert_eq!(two.rates(0), [0, 0, 0, 0, 0, 1, 1, 1]);
        assert_eq!(two.rates(2), [0, 1, 1, 1, 1, 1, 1, 1]);
        // 3 stages × 8 elements × 2 finest substeps against the evaluations
        assert_eq!(one.planned_speedup(), 48.0 / (3.0 * 8.0 + 2.0 + 3.0 + 4.0));
        assert_eq!(two.planned_speedup(), 48.0 / (3.0 * 8.0 + 3.0 + 5.0 + 7.0));
    }

    /// The node rates of a row of 6 P2 elements whose last is on level 2,
    /// stage by stage: the elements each stage evaluates at the finer rate.
    fn line_rates(line_local: [bool; 6]) -> [[u8; 6]; 4] {
        let mesh = Mesh2D::uniform_rectangle(0.0, 6.0, 0.0, 1.0, 6, 1);
        let ops = crate::operators::DGOperators2D::new(2);
        let lines = NodeLines {
            n_1d: ops.n_1d,
            face_nodes: &ops.face_nodes,
        };
        let level = [0u8, 0, 0, 0, 0, 2];
        let mut from: Vec<u8> = level.iter().flat_map(|&l| [l; 9]).collect();
        let mut to = vec![0; from.len()];
        let mut rates = [[0; 6]; 4];
        for rate in &mut rates {
            spread_over_lines(&mesh, lines, &line_local, &from, &mut to, rate);
            std::mem::swap(&mut from, &mut to);
        }
        rates
    }

    /// A finer neighbour changes a line-local element's shared face in one
    /// stage and the rest of it in the next, so the finer rate crosses an
    /// element in two stages: 1 + 1 + 2 + 2 evaluations normal to a level
    /// interface over four stages, against 1 + 2 + 3 + 4.
    #[test]
    fn line_local_rates_cross_an_element_in_two_stages() {
        assert_eq!(
            line_rates([true; 6]),
            [
                [0, 0, 0, 0, 2, 2],
                [0, 0, 0, 0, 2, 2],
                [0, 0, 0, 2, 2, 2],
                [0, 0, 0, 2, 2, 2]
            ]
        );
    }

    /// With no line-local element the node rates are the face stencil's.
    #[test]
    fn rates_of_whole_elements_spread_one_face_per_stage() {
        assert_eq!(
            line_rates([false; 6]),
            [
                [0, 0, 0, 0, 2, 2],
                [0, 0, 0, 2, 2, 2],
                [0, 0, 2, 2, 2, 2],
                [0, 2, 2, 2, 2, 2]
            ]
        );
    }

    /// An element that is not line-local (a shoreline on its subcells)
    /// follows whole and at once, and reads its neighbours one node in.
    #[test]
    fn whole_elements_among_line_local_ones() {
        // Element 4, beside the fine one, is whole: all of it follows in
        // the first stage, so element 3's face does in the second
        assert_eq!(
            line_rates([true, true, true, true, false, true]),
            [
                [0, 0, 0, 0, 2, 2],
                [0, 0, 0, 2, 2, 2],
                [0, 0, 0, 2, 2, 2],
                [0, 0, 2, 2, 2, 2]
            ]
        );
        // Element 3 is whole: of element 4 it reads the shared face and the
        // nodes one in from it, which follow from the second stage on
        assert_eq!(
            line_rates([true, true, true, false, true, true]),
            [
                [0, 0, 0, 0, 2, 2],
                [0, 0, 0, 0, 2, 2],
                [0, 0, 0, 2, 2, 2],
                [0, 0, 2, 2, 2, 2]
            ]
        );
    }

    /// Element time steps without a physics module, for the stepper's level
    /// assignment.
    struct Steps {
        element_dt: Vec<f64>,
        bound: f64,
    }

    type Field = crate::solver::DGSolution2D;

    impl LocalTimeStepping<Field> for Steps {
        fn element_dt(&self, _: &Field, _: f64, _: SspScheme, out: &mut [f64]) {
            out.copy_from_slice(&self.element_dt);
        }

        fn coarse_step_bound(&self, _: &Field, _: &[f64]) -> f64 {
            self.bound
        }

        fn stage_where(
            &self,
            _: &Field,
            _: &Field,
            _: &[u32],
            _: &(dyn Fn(usize) -> ElementStage + Sync),
            _: &mut Field,
            _: Option<&mut Field>,
        ) {
            unreachable!()
        }

        fn finish_where(&self, _: &mut Field, _: &Field, _: &[u32]) {
            unreachable!()
        }
    }

    /// A run that must land on a time takes equal coarse steps to it, each
    /// with the levels of its own length: an element is on the coarsest
    /// level whose step it can take, whatever the coarse step is. (Cutting
    /// the last coarse step instead shrinks every level with it: a short
    /// step at the cost of a whole one.)
    #[test]
    fn a_landing_takes_equal_steps_with_their_own_levels() {
        let mesh = Mesh2D::uniform_rectangle(0.0, 8.0, 0.0, 1.0, 8, 1);
        let state = Field::new(8, 4);
        let steps = Steps {
            element_dt: vec![8.0, 8.0, 8.0, 8.0, 4.0, 2.0, 1.0, 1.0],
            bound: f64::INFINITY,
        };
        let mut stepper = MultirateStepper::<Field>::new(8, SspScheme::Rk3);

        // Free: 2³ times the smallest step
        let dt = stepper.assign_levels(&steps, &mesh, &state, 1.0, None, None);
        assert_eq!(dt, 8.0);
        assert_eq!(stepper.levels(), [0, 0, 0, 0, 1, 2, 3, 3]);
        let (ideal, by_level, _) = stepper.planned_work(8.0);
        assert_eq!(ideal, 3.0 * 3.25);
        assert_eq!(by_level, ideal);

        // 20 s to go: three steps of 20/3 s, and the levels of 20/3 s
        let dt = stepper.assign_levels(&steps, &mesh, &state, 1.0, None, Some(20.0));
        assert_eq!(dt, 20.0 / 3.0);
        assert_eq!(stepper.stable_step(), 8.0);
        assert_eq!(stepper.levels(), [0, 0, 0, 0, 1, 2, 3, 3]);
        for (&l, &dt_k) in stepper.levels().iter().zip(&steps.element_dt) {
            assert!(dt / (1 << l) as f64 <= dt_k);
        }

        // 3 s to go: one step of 3 s, which the 4 s element takes whole and
        // the 1 s elements in four: 5 + 2 + 2·4 = 15 element steps, where the
        // 8 s step's levels, shrunk, take 4 + 2 + 4 + 16 = 26
        let dt = stepper.assign_levels(&steps, &mesh, &state, 1.0, None, Some(3.0));
        assert_eq!(dt, 3.0);
        assert_eq!(stepper.levels(), [0, 0, 0, 0, 0, 1, 2, 2]);
        assert_eq!(stepper.finest_level(), 2);
        let (_, by_level, _) = stepper.planned_work(3.0);
        assert_eq!(by_level * 3.0, 3.0 * 15.0);

        // Exactly a step to go: that step
        let dt = stepper.assign_levels(&steps, &mesh, &state, 1.0, None, Some(8.0));
        assert_eq!(dt, 8.0);
        assert_eq!(stepper.levels(), [0, 0, 0, 0, 1, 2, 3, 3]);
    }

    /// A cap below most elements' steps is the coarse step itself with
    /// [`MultirateStepper::with_longest_coarse_step`], and a landing is
    /// fitted to it: every element steps at the longest power-of-two
    /// fraction of the coarse step it can take, none longer than its own.
    #[test]
    fn the_longest_coarse_step_is_the_cap_where_it_binds() {
        let mesh = Mesh2D::uniform_rectangle(0.0, 16.0, 0.0, 1.0, 16, 1);
        // Twelve elements that could step at minutes and one at 19 s, under
        // a cap of 60 s
        let mut element_dt = vec![300.0; 12];
        element_dt.extend([120.0, 70.0, 70.0, 19.0]);
        let faces = [RhsStencil::Faces];
        let work = |stepper: &MultirateStepper<Field>, dt: f64| stepper.planned_work(dt).1;
        let levels = |coarse: usize, fine: &[u8]| [vec![0; coarse], fine.to_vec()].concat();

        // A power of two times the smallest: 38 s, everything else at 38 s
        let mut stepper = MultirateStepper::<Field>::new(3, SspScheme::Rk3);
        let dt = stepper.assign_levels_from(&element_dt, &mesh, &faces, Some(60.0), None);
        assert_eq!(dt, 38.0);
        assert_eq!(stepper.levels(), levels(15, &[1]));
        let power_of_two = work(&stepper, dt);
        assert_eq!(power_of_two, 3.0 * 17.0 / 38.0);

        // The cap: 60 s, the 19 s element at 15 s and its neighbour, a level
        // from it, at 30 s
        let mut stepper = stepper.with_longest_coarse_step(true);
        let dt = stepper.assign_levels_from(&element_dt, &mesh, &faces, Some(60.0), None);
        assert_eq!(dt, 60.0);
        assert_eq!(stepper.stable_step(), 60.0);
        assert_eq!(stepper.levels(), levels(14, &[1, 2]));
        let longest = work(&stepper, dt);
        assert_eq!(longest, 3.0 * 20.0 / 60.0);
        assert!(longest < 0.8 * power_of_two);

        // 100 s to an output: two steps of 50 s, on the levels of 50 s
        let dt = stepper.assign_levels_from(&element_dt, &mesh, &faces, Some(60.0), Some(100.0));
        assert_eq!(dt, 50.0);
        assert_eq!(stepper.stable_step(), 60.0);
        assert_eq!(stepper.levels(), levels(14, &[1, 2]));

        // No cap, or one out of reach: as far as the levels go, 2³ × 19 s
        for cap in [None, Some(1e3)] {
            let dt = stepper.assign_levels_from(&element_dt, &mesh, &faces, cap, None);
            assert_eq!(dt, 152.0);
            assert_eq!(stepper.levels(), levels(12, &[1, 2, 2, 3]));
        }
        // Fewer levels than the cap needs: 2² × 19 s
        let mut two =
            MultirateStepper::<Field>::new(2, SspScheme::Rk3).with_longest_coarse_step(true);
        let dt = two.assign_levels_from(&element_dt, &mesh, &faces, Some(100.0), None);
        assert_eq!(dt, 76.0);
        assert_eq!(two.levels(), levels(13, &[1, 1, 2]));

        // The largest element step where it is what binds: 70 s for 2 × 19 s
        let dt = stepper.assign_levels_from(
            &[70.0, 70.0, 30.0, 19.0],
            &Mesh2D::uniform_rectangle(0.0, 4.0, 0.0, 1.0, 4, 1),
            &faces,
            None,
            None,
        );
        assert_eq!(dt, 70.0);
        assert_eq!(stepper.levels(), [0, 1, 2, 2]);

        // Every element at one step: that step, on one level
        let dt = stepper.assign_levels_from(&[7.0; 16], &mesh, &faces, Some(60.0), None);
        assert_eq!(dt, 7.0);
        assert_eq!(stepper.finest_level(), 0);
    }

    /// The module's bound on the coarse step takes levels off the coarse end:
    /// the smallest step and every element's own are as they were.
    #[test]
    fn the_modules_bound_caps_the_coarse_step() {
        let mesh = Mesh2D::uniform_rectangle(0.0, 4.0, 0.0, 1.0, 4, 1);
        let state = Field::new(4, 4);
        // A film's 500 s beside water that steps at 1 to 4 s
        let steps = Steps {
            element_dt: vec![500.0, 4.0, 2.0, 1.0],
            bound: 4.0,
        };
        let mut stepper = MultirateStepper::<Field>::new(8, SspScheme::Rk3);
        let dt = stepper.assign_levels(&steps, &mesh, &state, 1.0, None, None);
        assert_eq!(dt, 4.0);
        assert_eq!(stepper.levels(), [0, 0, 1, 2]);
        // Without it the film's step makes the coarse step 2⁸ s
        let free = Steps {
            bound: f64::INFINITY,
            ..steps
        };
        let dt = stepper.assign_levels(&free, &mesh, &state, 1.0, None, None);
        assert_eq!(dt, 256.0);
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

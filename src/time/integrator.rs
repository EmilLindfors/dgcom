//! Trait-based time integrator abstraction.
//!
//! This module provides traits for time integration that enable:
//! - Generic time integrators that work with any solution type
//! - Extensible integrator implementations
//! - Both compile-time and runtime dispatch options
//!
//! # Example
//! ```
//! use dgcom::time::{Integrable, TimeIntegrator, SSPRK3};
//! use dgcom::solver::DGSolution1D;
//!
//! // DGSolution1D implements Integrable
//! let mut u = DGSolution1D::new(4, 3);
//! for v in &mut u.data { *v = 1.0; }
//!
//! let integrator = SSPRK3;
//! let dt = 0.01;
//! let t = 0.0;
//!
//! // Simple linear RHS: du/dt = -u (exponential decay)
//! integrator.step(&mut u, dt, t, |state, _time| {
//!     let mut rhs = state.clone();
//!     rhs.scale(-1.0);
//!     rhs
//! });
//! ```

// =============================================================================
// Integrable Trait
// =============================================================================

/// Trait for solution types that can be time-integrated.
///
/// This provides the vector space operations needed by explicit time integrators:
/// - `scale`: Multiply by scalar (x <- c * x)
/// - `axpy`: Add scaled vector (x <- x + c * y)
///
/// These operations should be implemented efficiently without allocations
/// beyond what's needed for intermediate stages.
///
/// # Example
/// ```
/// use dgcom::time::Integrable;
/// use dgcom::solver::DGSolution1D;
///
/// let mut u = DGSolution1D::new(4, 3);
/// for v in &mut u.data { *v = 1.0; }
///
/// let v = u.clone();
/// u.scale(2.0);      // u = 2.0 * u
/// u.axpy(0.5, &v);   // u = u + 0.5 * v
/// ```
pub trait Integrable: Clone + Send + Sized {
    /// Scale the solution by a constant: self <- c * self
    fn scale(&mut self, c: f64);

    /// Add a scaled vector: self <- self + c * other
    fn axpy(&mut self, c: f64, other: &Self);

    /// Create a zero-initialized solution with the same shape.
    ///
    /// Default implementation clones and scales by zero.
    fn zeros_like(&self) -> Self {
        let mut result = self.clone();
        result.scale(0.0);
        result
    }

    /// Overwrite `self` with `other` (same shape expected).
    ///
    /// Defaults to [`Clone::clone_from`], which reuses the allocation for the
    /// SoA solution types, so the stage buffers of
    /// [`TimeIntegrator::step_with_workspace`] are not reallocated.
    fn copy_from(&mut self, other: &Self) {
        self.clone_from(other);
    }

    /// A stage combination `self ← a·base + Σᵢ cᵢ·xᵢ`, with `base` the
    /// current `self` when `None`.
    ///
    /// Defined as [`Self::copy_from`]`(base)`, [`Self::scale`]`(a)` (left out
    /// for `a = 1`), then one [`Self::axpy`] per term in order. An override
    /// that fuses them into one pass must evaluate every value in that same
    /// order (`((base·a) + c₁x₁) + c₂x₂`), so the integrators give the same
    /// bits either way.
    fn combine(&mut self, base: Option<&Self>, a: f64, terms: &[(f64, &Self)]) {
        if let Some(base) = base {
            self.copy_from(base);
        }
        if a != 1.0 {
            self.scale(a);
        }
        for &(c, x) in terms {
            self.axpy(c, x);
        }
    }

    /// Whether every value is finite. [`crate::simulation::Simulation`]
    /// stops at the first step that leaves a NaN or an infinity, which the
    /// steps after it would otherwise carry on to the end (the step bounds
    /// skip NaN). The default, for types without an override, is `true`.
    fn is_finite(&self) -> bool {
        true
    }
}

/// Reusable stage storage for [`TimeIntegrator::step_with_workspace`].
///
/// The buffers are cloned from the state on first use and reused afterwards,
/// so a step allocates nothing once the workspace is warm (given an RHS that
/// writes into its output). Keep one per simulation.
pub struct StageWorkspace<S> {
    u1: Option<S>,
    u2: Option<S>,
    k: Option<S>,
}

impl<S> Default for StageWorkspace<S> {
    fn default() -> Self {
        Self {
            u1: None,
            u2: None,
            k: None,
        }
    }
}

impl<S: Integrable> StageWorkspace<S> {
    /// Empty workspace; buffers are created on the first step.
    pub fn new() -> Self {
        Self::default()
    }

    /// Two stage buffers and one RHS buffer shaped like `like`.
    fn buffers(&mut self, like: &S) -> (&mut S, &mut S, &mut S) {
        let u1 = self.u1.get_or_insert_with(|| like.clone());
        let u2 = self.u2.get_or_insert_with(|| like.clone());
        let k = self.k.get_or_insert_with(|| like.clone());
        (u1, u2, k)
    }
}

// =============================================================================
// IntegratorInfo Trait (non-generic, dyn-compatible)
// =============================================================================

/// Non-generic information about a time integrator.
///
/// This trait is separate from [`TimeIntegrator`] to allow calling info methods
/// without specifying a solution type. It is also dyn-compatible.
pub trait IntegratorInfo: Send + Sync {
    /// Human-readable name for debugging and logging.
    fn name(&self) -> &'static str;

    /// Order of accuracy of the integrator.
    fn order(&self) -> usize;

    /// Number of stages in the integrator.
    fn n_stages(&self) -> usize;

    /// Whether the integrator is strong stability preserving (SSP).
    ///
    /// SSP integrators maintain TVD and other nonlinear stability properties.
    fn is_ssp(&self) -> bool;

    /// Times at which RHS is evaluated relative to current time.
    ///
    /// For SSP-RK3: [0, dt, dt/2] (stages evaluate at t, t+dt, t+dt/2)
    fn stage_times(&self, dt: f64) -> Vec<f64>;

    /// Largest number of levels below the coarsest if this integrator steps
    /// elements with their own time steps (local time stepping, see
    /// [`crate::time::MultirateSSPRK3`]); `None` for global time steps.
    fn max_local_levels(&self) -> Option<usize> {
        None
    }

    /// The [`SspScheme`] a step of this integrator is, stage for stage and
    /// with the same arithmetic, if any. `Simulation` then runs global steps
    /// through the physics module's fused per-element stages
    /// ([`crate::time::LocalTimeStepping::stage_where`]: RHS, stage
    /// combination, implicit damping and post-processing in one pass per
    /// stage) when the module supports local time stepping, bit for bit the
    /// same; a multirate integrator steps every element with it.
    fn ssp_scheme(&self) -> Option<SspScheme> {
        None
    }

    /// SSP coefficient C: every step is a convex combination of forward
    /// Euler steps of at most `dt/C`, so a nonlinear bound that holds for
    /// forward Euler up to `Δt_FE` (positivity, a maximum principle) holds
    /// up to `C·Δt_FE`. `Simulation` scales
    /// [`PhysicsModule::max_cfl`](crate::physics::PhysicsModule::max_cfl) by
    /// it. That of the [`Self::ssp_scheme`], else 1 (forward Euler).
    fn ssp_coefficient(&self) -> f64 {
        self.ssp_scheme().map_or(1.0, SspScheme::ssp_coefficient)
    }
}

// =============================================================================
// Two-register SSP schemes (Shu–Osher form)
// =============================================================================

/// One stage of an [`SspScheme`]: from the step start `u⁽⁰⁾` and the previous
/// stage value `u⁽ⁱ⁻¹⁾`,
/// `u⁽ⁱ⁾ = start·u⁽⁰⁾ + input·u⁽ⁱ⁻¹⁾ + beta·dt·L(u⁽ⁱ⁻¹⁾, t + c·dt)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShuOsherStage {
    /// Weight of the step start.
    pub start: f64,
    /// Weight of the previous stage value (the RHS argument).
    pub input: f64,
    /// Weight of `dt` times the RHS.
    pub beta: f64,
    /// Time of the RHS evaluation, as a fraction of `dt`.
    pub c: f64,
}

/// SSP Runge–Kutta methods whose every stage combines only the step start
/// and the previous stage value ([`ShuOsherStage`]), with non-negative
/// weights (`start + input = 1`). The fused and the multirate steppers
/// ([`crate::time::MultirateStepper`]) run any of them; the last stage is the
/// new state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SspScheme {
    /// SSP-RK3 (Shu & Osher 1988): three stages, SSP coefficient 1 (an
    /// efficiency of 1/3 per RHS evaluation).
    Rk3,
    /// SSP-RK(4,3) (Kraaijevanger 1991; Spiteri & Ruuth 2002, SIAM J. Numer.
    /// Anal. 40:469–491): four stages, SSP coefficient 2 (an efficiency of
    /// 1/2): twice SSP-RK3's step under a forward-Euler bound for 4/3 of its
    /// work.
    Rk43,
}

impl SspScheme {
    /// The stages, in order.
    pub const fn stages(self) -> &'static [ShuOsherStage] {
        const fn stage(start: f64, input: f64, beta: f64, c: f64) -> ShuOsherStage {
            ShuOsherStage {
                start,
                input,
                beta,
                c,
            }
        }
        const RK3: [ShuOsherStage; 3] = [
            stage(1.0, 0.0, 1.0, 0.0),
            stage(0.75, 0.25, 0.25, 1.0),
            stage(1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 0.5),
        ];
        const RK43: [ShuOsherStage; 4] = [
            stage(1.0, 0.0, 0.5, 0.0),
            stage(0.0, 1.0, 0.5, 0.5),
            stage(2.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0, 1.0),
            stage(0.0, 1.0, 0.5, 0.5),
        ];
        match self {
            SspScheme::Rk3 => &RK3,
            SspScheme::Rk43 => &RK43,
        }
    }

    /// Number of stages (RHS evaluations per step).
    pub const fn n_stages(self) -> usize {
        self.stages().len()
    }

    /// SSP coefficient: the smallest ratio of a stage's weights on the
    /// previous stage and on its RHS, `input/beta` (the first stage's
    /// input is the step start).
    pub const fn ssp_coefficient(self) -> f64 {
        match self {
            SspScheme::Rk3 => 1.0,
            SspScheme::Rk43 => 2.0,
        }
    }

    /// Stage times as offsets from the step start.
    pub fn stage_times(self, dt: f64) -> Vec<f64> {
        self.stages().iter().map(|stage| stage.c * dt).collect()
    }
}

// =============================================================================
// TimeIntegrator Trait
// =============================================================================

/// Trait for explicit time integrators.
///
/// Time integrators advance the solution from time `t` to `t + dt`
/// using one or more RHS evaluations. The RHS function receives
/// the current state and time, returning the time derivative.
///
/// # Implementation Notes
///
/// - Integrators should use the `Integrable` trait operations
/// - RHS functions should not allocate (hot path)
/// - Stage intermediate values may need to be stored in the integrator
///
/// # Example
/// ```
/// use dgcom::time::{Integrable, TimeIntegrator, SSPRK3};
/// use dgcom::solver::DGSolution1D;
///
/// let mut u = DGSolution1D::new(4, 3);
/// for v in &mut u.data { *v = 1.0; }
///
/// let integrator = SSPRK3;
/// integrator.step(&mut u, 0.01, 0.0, |state, _time| {
///     let mut rhs = state.clone();
///     rhs.scale(-1.0);
///     rhs
/// });
/// ```
pub trait TimeIntegrator<S: Integrable>: IntegratorInfo {
    /// Advance the solution by one time step.
    ///
    /// # Arguments
    /// * `state` - Solution to advance (modified in place)
    /// * `dt` - Time step size
    /// * `t` - Current time
    /// * `rhs` - Function computing the RHS: f(state, time) -> time_derivative
    fn step<F>(&self, state: &mut S, dt: f64, t: f64, rhs: F)
    where
        F: FnMut(&S, f64) -> S,
    {
        self.step_with_stage_hook(state, dt, t, rhs, |_| {});
    }

    /// Advance one step, applying `stage_hook` to every stage value, including
    /// the final one, before it is used.
    ///
    /// This is where nonlinear projections (positivity and slope limiters,
    /// wet/dry correction) belong: the Zhang–Shu positivity guarantee for SSP
    /// methods holds only if every RHS evaluation sees a limited state.
    ///
    /// Allocates stage storage and an RHS result per stage; long runs should
    /// use [`Self::step_with_workspace`].
    fn step_with_stage_hook<F, H>(&self, state: &mut S, dt: f64, t: f64, mut rhs: F, stage_hook: H)
    where
        F: FnMut(&S, f64) -> S,
        H: FnMut(&mut S),
    {
        let mut workspace = StageWorkspace::new();
        self.step_with_workspace(
            state,
            dt,
            t,
            |s, time, out: &mut S| *out = rhs(s, time),
            stage_hook,
            &mut workspace,
        );
    }

    /// Allocation-free version of [`Self::step_with_stage_hook`].
    ///
    /// `rhs(state, time, out)` must overwrite `out` with the time derivative;
    /// stage values live in `workspace`, reused across steps.
    fn step_with_workspace<F, H>(
        &self,
        state: &mut S,
        dt: f64,
        t: f64,
        rhs: F,
        stage_hook: H,
        workspace: &mut StageWorkspace<S>,
    ) where
        F: FnMut(&S, f64, &mut S),
        H: FnMut(&mut S),
    {
        self.step_with_relaxation(state, dt, t, rhs, |_, _, _| {}, stage_hook, workspace);
    }

    /// [`Self::step_with_workspace`] with a point-implicit relaxation of every
    /// stage. This is the one method an integrator implements; the others are
    /// built on it.
    ///
    /// In Shu–Osher form every stage is
    /// `u⁽ⁱ⁾ = Σₖ αᵢₖ u⁽ᵏ⁾ + βᵢ dt·L(u⁽ⁱ⁻¹⁾)` with `Σₖ αᵢₖ = 1`. Each is
    /// followed by `relax(u⁽ⁱ⁾, u⁽ⁱ⁻¹⁾, βᵢ dt)`, which applies a stiff damping
    /// source `−Λ(u⁽ⁱ⁻¹⁾)·q` implicitly, `u⁽ⁱ⁾ ← u⁽ⁱ⁾ / (1 + βᵢ dt·Λ)`, with the
    /// rate frozen at the state `L` was evaluated at:
    /// - the damping shrinks the stage value but never flips its sign, for any
    ///   `dt` (explicit SSP-RK3 goes unstable once `dt·Λ > 2.5`);
    /// - a steady state of `L(u) − Λ(u)u = 0` is a fixed point of every stage
    ///   (the numerator is `u*(1 + βᵢ dt Λ)`), so balances such as forcing
    ///   against friction are kept exactly, independently of `dt`;
    /// - as `dt·Λ → ∞` every stage is driven to zero (L-stable), unlike
    ///   relaxing only the Euler substeps, which leaves `u/3` per SSP-RK3 step.
    ///
    /// The damping term is integrated to first order; the rest keeps the
    /// integrator's order. `stage_hook` then runs on each stage value as in
    /// [`Self::step_with_workspace`].
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
        H: FnMut(&mut S);
}

// =============================================================================
// SSP-RK3 Implementation
// =============================================================================

/// Strong Stability Preserving Runge-Kutta 3rd order integrator.
///
/// The SSP-RK3 method (Shu-Osher form) is optimal for hyperbolic conservation laws.
/// It maintains the TVD property of the spatial discretization.
///
/// Stages:
/// ```text
/// u1 = u + dt * L(u, t)
/// u2 = 3/4 * u + 1/4 * u1 + 1/4 * dt * L(u1, t + dt)
/// u_new = 1/3 * u + 2/3 * u2 + 2/3 * dt * L(u2, t + dt/2)
/// ```
///
/// Stage times: t, t + dt, t + dt/2
#[derive(Clone, Copy, Debug, Default)]
pub struct SSPRK3;

impl IntegratorInfo for SSPRK3 {
    fn name(&self) -> &'static str {
        "ssp-rk3"
    }

    fn ssp_scheme(&self) -> Option<SspScheme> {
        Some(SspScheme::Rk3)
    }

    fn order(&self) -> usize {
        3
    }

    fn n_stages(&self) -> usize {
        3
    }

    fn is_ssp(&self) -> bool {
        true
    }

    fn stage_times(&self, dt: f64) -> Vec<f64> {
        vec![0.0, dt, 0.5 * dt]
    }
}

impl<S: Integrable> TimeIntegrator<S> for SSPRK3 {
    fn step_with_relaxation<F, R, H>(
        &self,
        state: &mut S,
        dt: f64,
        t: f64,
        mut rhs: F,
        mut relax: R,
        mut stage_hook: H,
        workspace: &mut StageWorkspace<S>,
    ) where
        F: FnMut(&S, f64, &mut S),
        R: FnMut(&mut S, &S, f64),
        H: FnMut(&mut S),
    {
        let (u1, u2, k) = workspace.buffers(state);

        // Stage 1: u1 = u + dt * L(u, t)
        rhs(state, t, k);
        u1.combine(Some(state), 1.0, &[(dt, k)]);
        relax(u1, state, dt);
        stage_hook(u1);

        // Stage 2: u2 = 3/4 * u + 1/4 * u1 + 1/4 * dt * L(u1, t + dt)
        rhs(u1, t + dt, k);
        u2.combine(Some(state), 0.75, &[(0.25, u1), (0.25 * dt, k)]);
        relax(u2, u1, 0.25 * dt);
        stage_hook(u2);

        // Stage 3: u_new = 1/3 * u + 2/3 * u2 + 2/3 * dt * L(u2, t + dt/2)
        rhs(u2, t + 0.5 * dt, k);
        state.combine(None, 1.0 / 3.0, &[(2.0 / 3.0, u2), (2.0 / 3.0 * dt, k)]);
        relax(state, u2, 2.0 / 3.0 * dt);
        stage_hook(state);
    }
}

// =============================================================================
// SSP-RK(4,3)
// =============================================================================

/// Four-stage, third-order SSP Runge–Kutta method, SSP-RK(4,3)
/// (Kraaijevanger 1991; Spiteri & Ruuth 2002), [`SspScheme::Rk43`]:
///
/// ```text
/// u1    = u + dt/2 · L(u, t)
/// u2    = u1 + dt/2 · L(u1, t + dt/2)
/// u3    = 2/3 · u + 1/3 · u2 + dt/6 · L(u2, t + dt)
/// u_new = u3 + dt/2 · L(u3, t + dt/2)
/// ```
///
/// Every stage is a forward-Euler step of `dt/2` (stage 3 inside a convex
/// combination), so its SSP coefficient is 2: a nonlinear bound that holds
/// for forward Euler up to `Δt_FE`, such as the DGSEM positivity bound of
/// wet/dry runs, holds up to `2·Δt_FE`, against `Δt_FE` for [`SSPRK3`]. That
/// is 1.5× less RHS work where such a bound sets the step. Its linear
/// stability region is larger than SSP-RK3's too (the negative real axis to
/// 5.15 against 2.51, the imaginary axis to 2.16 against 1.73), and at CFL
/// numbers up to that doubled bound DGSEM stays well inside it for N = 1–4.
/// Where only linear stability limits the step (no wetting/drying), it saves
/// nothing over SSP-RK3 unless the CFL number is raised.
///
/// Stage times: t, t + dt/2, t + dt, t + dt/2 (weights 1/6, 1/6, 1/6, 1/2).
#[derive(Clone, Copy, Debug, Default)]
pub struct SSPRK43;

impl IntegratorInfo for SSPRK43 {
    fn name(&self) -> &'static str {
        "ssp-rk43"
    }

    fn ssp_scheme(&self) -> Option<SspScheme> {
        Some(SspScheme::Rk43)
    }

    fn order(&self) -> usize {
        3
    }

    fn n_stages(&self) -> usize {
        4
    }

    fn is_ssp(&self) -> bool {
        true
    }

    fn stage_times(&self, dt: f64) -> Vec<f64> {
        SspScheme::Rk43.stage_times(dt)
    }
}

impl<S: Integrable> TimeIntegrator<S> for SSPRK43 {
    /// Two stage buffers: u3 reuses u1's. The weights are the
    /// [`SspScheme::Rk43`] table's, multiplied in the order of the fused
    /// stages, so both give the same result bit for bit.
    fn step_with_relaxation<F, R, H>(
        &self,
        state: &mut S,
        dt: f64,
        t: f64,
        mut rhs: F,
        mut relax: R,
        mut stage_hook: H,
        workspace: &mut StageWorkspace<S>,
    ) where
        F: FnMut(&S, f64, &mut S),
        R: FnMut(&mut S, &S, f64),
        H: FnMut(&mut S),
    {
        let [s1, s2, s3, s4] = SspScheme::Rk43.stages() else {
            unreachable!("SSP-RK(4,3) has four stages")
        };
        let (u1, u2, k) = workspace.buffers(state);

        // Stage 1: u1 = u + dt/2 · L(u, t)
        rhs(state, t + s1.c * dt, k);
        u1.combine(Some(state), 1.0, &[(s1.beta * dt, k)]);
        relax(u1, state, s1.beta * dt);
        stage_hook(u1);

        // Stage 2: u2 = u1 + dt/2 · L(u1, t + dt/2)
        rhs(u1, t + s2.c * dt, k);
        u2.combine(Some(u1), 1.0, &[(s2.beta * dt, k)]);
        relax(u2, u1, s2.beta * dt);
        stage_hook(u2);

        // Stage 3: u3 = 2/3 · u + 1/3 · u2 + dt/6 · L(u2, t + dt), into u1
        rhs(u2, t + s3.c * dt, k);
        u1.combine(Some(state), s3.start, &[(s3.input, u2), (s3.beta * dt, k)]);
        relax(u1, u2, s3.beta * dt);
        stage_hook(u1);

        // Stage 4: u_new = u3 + dt/2 · L(u3, t + dt/2)
        rhs(u1, t + s4.c * dt, k);
        state.combine(Some(u1), 1.0, &[(s4.beta * dt, k)]);
        relax(state, u1, s4.beta * dt);
        stage_hook(state);
    }
}

// =============================================================================
// Forward Euler (for comparison/testing)
// =============================================================================

/// Forward Euler integrator (1st order).
///
/// Simple but only 1st order accurate. Useful for testing and debugging.
///
/// ```text
/// u_new = u + dt * L(u, t)
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct ForwardEuler;

impl IntegratorInfo for ForwardEuler {
    fn name(&self) -> &'static str {
        "forward-euler"
    }

    fn order(&self) -> usize {
        1
    }

    fn n_stages(&self) -> usize {
        1
    }

    fn is_ssp(&self) -> bool {
        true // Forward Euler is SSP with C_eff = 1
    }

    fn stage_times(&self, _dt: f64) -> Vec<f64> {
        vec![0.0]
    }
}

impl<S: Integrable> TimeIntegrator<S> for ForwardEuler {
    fn step_with_relaxation<F, R, H>(
        &self,
        state: &mut S,
        dt: f64,
        t: f64,
        mut rhs: F,
        mut relax: R,
        mut stage_hook: H,
        workspace: &mut StageWorkspace<S>,
    ) where
        F: FnMut(&S, f64, &mut S),
        R: FnMut(&mut S, &S, f64),
        H: FnMut(&mut S),
    {
        let (u0, _, k) = workspace.buffers(state);
        rhs(state, t, k);
        u0.copy_from(state);
        state.axpy(dt, k);
        relax(state, u0, dt);
        stage_hook(state);
    }
}

// =============================================================================
// Standard Integrator Enum (Zero-Cost Dispatch)
// =============================================================================

/// Enum wrapper for built-in integrators.
///
/// Provides zero-cost dispatch when integrator type is known at compile time,
/// while still allowing runtime selection via configuration.
#[derive(Clone, Copy, Debug, Default)]
pub enum StandardIntegrator {
    /// SSP-RK3 (default, recommended for hyperbolic problems)
    #[default]
    SSPRK3,
    /// SSP-RK(4,3): twice SSP-RK3's step under positivity bounds
    SSPRK43,
    /// Forward Euler (1st order, for testing)
    ForwardEuler,
}

impl IntegratorInfo for StandardIntegrator {
    fn name(&self) -> &'static str {
        match self {
            StandardIntegrator::SSPRK3 => SSPRK3.name(),
            StandardIntegrator::SSPRK43 => SSPRK43.name(),
            StandardIntegrator::ForwardEuler => ForwardEuler.name(),
        }
    }

    fn order(&self) -> usize {
        match self {
            StandardIntegrator::SSPRK3 | StandardIntegrator::SSPRK43 => 3,
            StandardIntegrator::ForwardEuler => 1,
        }
    }

    fn n_stages(&self) -> usize {
        match self {
            StandardIntegrator::SSPRK3 => 3,
            StandardIntegrator::SSPRK43 => 4,
            StandardIntegrator::ForwardEuler => 1,
        }
    }

    fn is_ssp(&self) -> bool {
        true // All are SSP
    }

    fn stage_times(&self, dt: f64) -> Vec<f64> {
        match self {
            StandardIntegrator::SSPRK3 => SSPRK3.stage_times(dt),
            StandardIntegrator::SSPRK43 => SSPRK43.stage_times(dt),
            StandardIntegrator::ForwardEuler => vec![0.0],
        }
    }

    fn ssp_scheme(&self) -> Option<SspScheme> {
        match self {
            StandardIntegrator::SSPRK3 => SSPRK3.ssp_scheme(),
            StandardIntegrator::SSPRK43 => SSPRK43.ssp_scheme(),
            StandardIntegrator::ForwardEuler => None,
        }
    }
}

impl<S: Integrable> TimeIntegrator<S> for StandardIntegrator {
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
        match self {
            StandardIntegrator::SSPRK3 => {
                SSPRK3.step_with_relaxation(state, dt, t, rhs, relax, stage_hook, workspace)
            }
            StandardIntegrator::SSPRK43 => {
                SSPRK43.step_with_relaxation(state, dt, t, rhs, relax, stage_hook, workspace)
            }
            StandardIntegrator::ForwardEuler => {
                ForwardEuler.step_with_relaxation(state, dt, t, rhs, relax, stage_hook, workspace)
            }
        }
    }
}

// =============================================================================
// Boxed Integrator Info (Runtime Polymorphism for Info Only)
// =============================================================================

/// Type alias for boxed integrator info (runtime polymorphism).
///
/// Note: The full `TimeIntegrator` trait is not dyn-compatible due to the
/// generic closure parameter in `step`. Use `StandardIntegrator` enum for
/// runtime selection of integrators.
pub type BoxedIntegratorInfo = Box<dyn IntegratorInfo>;

/// Create a boxed integrator info from a standard integrator type.
pub fn create_integrator_info(integrator: StandardIntegrator) -> BoxedIntegratorInfo {
    match integrator {
        StandardIntegrator::SSPRK3 => Box::new(SSPRK3),
        StandardIntegrator::SSPRK43 => Box::new(SSPRK43),
        StandardIntegrator::ForwardEuler => Box::new(ForwardEuler),
    }
}

// =============================================================================
// Integrable Implementations for Existing Types
// =============================================================================

use crate::solver::core::blocks::{combine_values, update_values, update_with};

/// Whether every value is finite, checked in branch-free chunks that
/// vectorise (an early-exit loop does not): a pass per step must stay cheap.
fn all_finite(values: &[f64]) -> bool {
    values
        .chunks(64)
        .all(|chunk| chunk.iter().fold(true, |ok, x| ok & x.is_finite()))
}
use crate::solver::{DGSolution1D, DGSolution2D, SWESolution, SWESolution2D, state::Solution3D};

impl Integrable for DGSolution1D {
    fn scale(&mut self, c: f64) {
        self.scale(c);
    }

    fn axpy(&mut self, c: f64, other: &Self) {
        self.axpy(c, other);
    }

    fn is_finite(&self) -> bool {
        all_finite(&self.data)
    }
}

impl Integrable for DGSolution2D {
    fn scale(&mut self, c: f64) {
        self.scale(c);
    }

    fn axpy(&mut self, c: f64, other: &Self) {
        self.axpy(c, other);
    }

    fn is_finite(&self) -> bool {
        all_finite(&self.data)
    }
}

impl Integrable for SWESolution {
    fn scale(&mut self, c: f64) {
        self.scale(c);
    }

    fn axpy(&mut self, c: f64, other: &Self) {
        self.axpy(c, other);
    }

    fn is_finite(&self) -> bool {
        all_finite(&self.data)
    }
}

impl Integrable for SWESolution2D {
    fn scale(&mut self, c: f64) {
        self.scale(c);
    }

    fn axpy(&mut self, c: f64, other: &Self) {
        self.axpy(c, other);
    }

    fn is_finite(&self) -> bool {
        self.data.iter().all(|values| all_finite(values))
    }
}

impl Integrable for Solution3D {
    fn scale(&mut self, c: f64) {
        // Scale 2D barotropic state
        self.eta.scale(c);
        self.ubar.scale(c);
        self.vbar.scale(c);

        // Scale 3D baroclinic state
        for field in [
            &mut self.u,
            &mut self.v,
            &mut self.w,
            &mut self.temp,
            &mut self.salt,
            &mut self.rho,
            &mut self.tke,
            &mut self.gls,
        ] {
            update_values(field, |x| *x *= c);
        }
    }

    fn axpy(&mut self, c: f64, other: &Self) {
        // AXPY 2D barotropic state
        self.eta.axpy(c, &other.eta);
        self.ubar.axpy(c, &other.ubar);
        self.vbar.axpy(c, &other.vbar);

        // AXPY 3D baroclinic state
        for (x, y) in [
            (&mut self.u, &other.u),
            (&mut self.v, &other.v),
            (&mut self.w, &other.w),
            (&mut self.temp, &other.temp),
            (&mut self.salt, &other.salt),
            (&mut self.rho, &other.rho),
        ] {
            update_with(x, y, |x, y| *x += c * y);
        }
        // The turbulence of a prognostic closure (empty without one): an
        // empty `other` adds nothing
        for (x, y) in [(&mut self.tke, &other.tke), (&mut self.gls, &other.gls)] {
            if !y.is_empty() {
                update_with(x, y, |x, y| *x += c * y);
            }
        }
    }

    /// In place (the derived `Clone` reallocates every field).
    fn copy_from(&mut self, other: &Self) {
        if (self.n_elements, self.n_nodes, self.n_levels)
            != (other.n_elements, other.n_nodes, other.n_levels)
        {
            *self = other.clone();
            return;
        }
        self.eta.copy_from(&other.eta);
        self.ubar.copy_from(&other.ubar);
        self.vbar.copy_from(&other.vbar);
        for (x, y) in [
            (&mut self.u, &other.u),
            (&mut self.v, &other.v),
            (&mut self.w, &other.w),
            (&mut self.temp, &other.temp),
            (&mut self.salt, &other.salt),
            (&mut self.rho, &other.rho),
            (&mut self.eddy_viscosity, &other.eddy_viscosity),
            (&mut self.eddy_diffusivity, &other.eddy_diffusivity),
        ] {
            update_with(x, y, |x, y| *x = y);
        }
        // Empty without a prognostic closure
        for (x, y) in [(&mut self.tke, &other.tke), (&mut self.gls, &other.gls)] {
            if x.len() == y.len() {
                update_with(x, y, |x, y| *x = y);
            } else {
                x.clone_from(y);
            }
        }
    }

    /// One pass per field instead of one per operation (bit for bit the
    /// `copy_from`, `scale`, `axpy` sequence): the fields `scale` and `axpy`
    /// change are combined, the eddy coefficients are copied from `base`.
    fn combine(&mut self, base: Option<&Self>, a: f64, terms: &[(f64, &Self)]) {
        crate::profiling::scope("3d: rk combination", || self.combine_fields(base, a, terms));
    }
}

impl Solution3D {
    fn combine_fields(&mut self, base: Option<&Self>, a: f64, terms: &[(f64, &Self)]) {
        const MAX_TERMS: usize = 4;
        let shape = |s: &Self| {
            (
                s.n_elements,
                s.n_nodes,
                s.n_levels,
                s.tke.len(),
                s.gls.len(),
            )
        };
        if terms.len() > MAX_TERMS || base.is_some_and(|b| shape(b) != shape(self)) {
            // A reshaping copy (or many terms): the plain sequence
            if let Some(base) = base {
                Integrable::copy_from(self, base);
            }
            if a != 1.0 {
                Integrable::scale(self, a);
            }
            for &(c, x) in terms {
                Integrable::axpy(self, c, x);
            }
            return;
        }
        // `field` of `base` and of every term whose `field` is not empty
        // (an empty turbulence field adds nothing, as in `axpy`)
        let combine = |out: &mut [f64], field: fn(&Self) -> &[f64]| {
            let mut buffer: [(f64, &[f64]); MAX_TERMS] = [(0.0, &[]); MAX_TERMS];
            let mut n = 0;
            for &(c, x) in terms {
                if !field(x).is_empty() {
                    buffer[n] = (c, field(x));
                    n += 1;
                }
            }
            combine_values(out, base.map(field), a, &buffer[..n]);
        };
        combine(&mut self.eta.data, |s| &s.eta.data);
        combine(&mut self.ubar.data, |s| &s.ubar.data);
        combine(&mut self.vbar.data, |s| &s.vbar.data);
        combine(&mut self.u, |s| &s.u);
        combine(&mut self.v, |s| &s.v);
        combine(&mut self.w, |s| &s.w);
        combine(&mut self.temp, |s| &s.temp);
        combine(&mut self.salt, |s| &s.salt);
        combine(&mut self.rho, |s| &s.rho);
        combine(&mut self.tke, |s| &s.tke);
        combine(&mut self.gls, |s| &s.gls);
        if let Some(base) = base {
            update_with(&mut self.eddy_viscosity, &base.eddy_viscosity, |x, y| {
                *x = y
            });
            update_with(
                &mut self.eddy_diffusivity,
                &base.eddy_diffusivity,
                |x, y| *x = y,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `Solution3D`'s fused `combine` gives the bits of the `copy_from`,
    /// `scale`, `axpy` sequence it replaces: with and without a base, with a
    /// prognostic turbulence and with an empty one in a term, and over more
    /// values than one streaming chunk.
    #[test]
    fn solution_3d_combine_matches_the_operation_sequence_bit_for_bit() {
        let (ne, nn, nl) = (40, 9, 50);
        let fill = |seed: u64, turbulence: bool| {
            let mut s = Solution3D::new(ne, nn, nl);
            let mut x = seed;
            let mut next = || {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (x >> 11) as f64 / (1u64 << 53) as f64 - 0.3
            };
            for field in [&mut s.eta.data, &mut s.ubar.data, &mut s.vbar.data] {
                field.iter_mut().for_each(|v| *v = next());
            }
            for field in [
                &mut s.u,
                &mut s.v,
                &mut s.w,
                &mut s.temp,
                &mut s.salt,
                &mut s.rho,
                &mut s.eddy_viscosity,
                &mut s.eddy_diffusivity,
            ] {
                field.iter_mut().for_each(|v| *v = next());
            }
            if turbulence {
                s.tke = (0..ne * nn * (nl + 1)).map(|_| next()).collect();
                s.gls = (0..ne * nn * (nl + 1)).map(|_| next()).collect();
            }
            s
        };
        assert!(ne * nn * nl > 16384, "more than one streaming chunk");
        let (state, x1, x2, x2_laminar) =
            (fill(1, true), fill(2, true), fill(3, true), fill(4, false));
        let bits = |s: &Solution3D| {
            [
                &s.eta.data,
                &s.ubar.data,
                &s.vbar.data,
                &s.u,
                &s.v,
                &s.w,
                &s.temp,
                &s.salt,
                &s.rho,
                &s.eddy_viscosity,
                &s.eddy_diffusivity,
                &s.tke,
                &s.gls,
            ]
            .map(|f| f.iter().map(|v| v.to_bits()).collect::<Vec<_>>())
        };
        type Case<'a> = (Option<&'a Solution3D>, f64, Vec<(f64, &'a Solution3D)>);
        let cases: [Case; 4] = [
            (Some(&state), 1.0, vec![(0.7, &x1)]),
            (Some(&state), 0.75, vec![(0.25, &x1), (0.25 * 0.7, &x2)]),
            (
                None,
                1.0 / 3.0,
                vec![(2.0 / 3.0, &x1), (2.0 / 3.0 * 0.7, &x2)],
            ),
            (None, 1.0, vec![(0.3, &x2_laminar), (0.1, &x1)]),
        ];
        for (case, (base, a, terms)) in cases.iter().enumerate() {
            let mut fused = fill(5, true);
            let mut sequence = fused.clone();
            fused.combine(*base, *a, terms);
            if let Some(base) = base {
                Integrable::copy_from(&mut sequence, base);
            }
            if *a != 1.0 {
                Integrable::scale(&mut sequence, *a);
            }
            for &(c, x) in terms {
                Integrable::axpy(&mut sequence, c, x);
            }
            assert!(bits(&fused) == bits(&sequence), "case {case}");
        }
    }

    #[test]
    fn test_ssprk3_order() {
        // Test RK3 order with exponential growth: du/dt = u, u(0) = 1
        // Exact: u(t) = exp(t)
        let mut u = DGSolution1D::new(1, 3);
        u.data.fill(1.0);

        let integrator = SSPRK3;
        let dt = 0.01;
        let n_steps = 10;

        for i in 0..n_steps {
            let t = dt * i as f64;
            integrator.step(&mut u, dt, t, |state, _time| state.clone());
        }

        let t = dt * n_steps as f64;
        let expected = t.exp();

        for &v in &u.data {
            let error = (v - expected).abs();
            assert!(
                error < 1e-4,
                "Expected {}, got {} (error {})",
                expected,
                v,
                error
            );
        }
    }

    /// Error of `integrator` at t = 1 on du/dt = u(1 + cos t), whose exact
    /// solution is exp(t + sin t): a time-dependent RHS, so wrong stage times
    /// would lower the order.
    fn exponential_error<I: TimeIntegrator<DGSolution1D>>(integrator: &I, n_steps: usize) -> f64 {
        let dt = 1.0 / n_steps as f64;
        let mut u = DGSolution1D::new(1, 1);
        u.data[0] = 1.0;
        for i in 0..n_steps {
            integrator.step(&mut u, dt, i as f64 * dt, |state, time| {
                let mut rhs = state.clone();
                rhs.scale(1.0 + time.cos());
                rhs
            });
        }
        (u.data[0] - (1.0 + 1.0_f64.sin()).exp()).abs()
    }

    #[test]
    fn test_ssp_schemes_converge_at_third_order() {
        for integrator in [StandardIntegrator::SSPRK3, StandardIntegrator::SSPRK43] {
            let (coarse, fine) = (
                exponential_error(&integrator, 20),
                exponential_error(&integrator, 40),
            );
            let rate = (coarse / fine).log2();
            assert!(
                (rate - 3.0).abs() < 0.1,
                "{}: order {rate:.3} (errors {coarse:e}, {fine:e})",
                integrator.name()
            );
        }
    }

    /// The Shu–Osher tables against their Butcher form: the order conditions
    /// up to third order, convex non-negative weights, the stated SSP
    /// coefficient and the stage times.
    #[test]
    fn test_ssp_scheme_tables() {
        for scheme in [SspScheme::Rk3, SspScheme::Rk43] {
            let stages = scheme.stages();
            let s = stages.len();
            // Butcher rows: stage value i (0 = the step start) as the start
            // plus dt·Σ a[i][j] F_j, with F_j the RHS at stage value j. The
            // start carries no RHS, so only the input's row propagates
            let mut a = vec![vec![0.0; s]; s + 1];
            for (i, stage) in stages.iter().enumerate() {
                a[i + 1] = a[i].iter().map(|&x| stage.input * x).collect();
                a[i + 1][i] += stage.beta;
            }
            let b = &a[s];
            let c: Vec<f64> = (0..s).map(|i| a[i].iter().sum()).collect();
            let close = |x: f64, y: f64| (x - y).abs() < 1e-15;
            let dot = |x: &[f64], y: &[f64]| x.iter().zip(y).map(|(x, y)| x * y).sum::<f64>();
            let ac: Vec<f64> = (0..s).map(|i| dot(&a[i], &c)).collect();
            let c2: Vec<f64> = c.iter().map(|c| c * c).collect();
            assert!(close(b.iter().sum(), 1.0), "{scheme:?}: Σb");
            assert!(close(dot(b, &c), 0.5), "{scheme:?}: Σbc");
            assert!(close(dot(b, &c2), 1.0 / 3.0), "{scheme:?}: Σbc²");
            assert!(close(dot(b, &ac), 1.0 / 6.0), "{scheme:?}: Σb(Ac)");
            for (i, stage) in stages.iter().enumerate() {
                assert!(close(stage.c, c[i]), "{scheme:?}: c of stage {i}");
                assert!(close(stage.start + stage.input, 1.0));
                assert!(stage.start >= 0.0 && stage.input >= 0.0 && stage.beta > 0.0);
            }
            // SSP coefficient: min over the stages of (weight of the RHS
            // argument)/beta; the first stage's argument is the start
            let coefficient = stages
                .iter()
                .enumerate()
                .map(|(i, st)| if i == 0 { st.start } else { st.input } / st.beta)
                .fold(f64::INFINITY, f64::min);
            assert!(close(coefficient, scheme.ssp_coefficient()), "{scheme:?}");
        }
        assert_eq!(SSPRK3.stage_times(0.1), SspScheme::Rk3.stage_times(0.1));
        assert_eq!(SSPRK43.ssp_coefficient(), 2.0);
        assert_eq!(SSPRK3.ssp_coefficient(), 1.0);
        assert_eq!(ForwardEuler.ssp_coefficient(), 1.0);
    }

    #[test]
    fn test_ssprk43_stage_hook_runs_after_each_stage() {
        let mut u = DGSolution1D::new(1, 1);
        u.data[0] = 1.0;
        let mut times = Vec::new();
        let mut n_hooks = 0;
        SSPRK43.step_with_stage_hook(
            &mut u,
            0.1,
            2.0,
            |state, time| {
                times.push(time);
                state.clone()
            },
            |_| n_hooks += 1,
        );
        assert_eq!(n_hooks, 4);
        let expected = [2.0, 2.05, 2.1, 2.05];
        assert!(
            times
                .iter()
                .zip(expected)
                .all(|(t, e)| (t - e).abs() < 1e-15),
            "stage times {times:?}"
        );
    }

    #[test]
    fn test_ssprk3_stage_hook_runs_after_each_stage() {
        let mut u = DGSolution1D::new(1, 1);
        u.data[0] = 1.0;

        let integrator = SSPRK3;
        let mut n_hooks = 0;
        integrator.step_with_stage_hook(
            &mut u,
            0.1,
            0.0,
            |state, _time| {
                let mut rhs = state.clone();
                rhs.scale(1.0);
                rhs
            },
            |stage_state| {
                n_hooks += 1;
                stage_state.data[0] = stage_state.data[0].min(1.05);
            },
        );

        assert_eq!(n_hooks, 3);
        assert!(u.data[0] <= 1.05);
    }

    /// du/dt = F − Λ|u|u (constant forcing, quadratic drag) with the drag
    /// relaxed point-implicitly: Λ|u_in| frozen at the RHS input.
    fn step_forced_drag<I: TimeIntegrator<DGSolution1D>>(
        integrator: &I,
        u: &mut DGSolution1D,
        dt: f64,
        forcing: f64,
        drag: f64,
    ) {
        integrator.step_with_relaxation(
            u,
            dt,
            0.0,
            |_, _, out: &mut DGSolution1D| out.data.fill(forcing),
            |w: &mut DGSolution1D, from: &DGSolution1D, dt| {
                for (w, &u) in w.data.iter_mut().zip(&from.data) {
                    *w /= 1.0 + dt * drag * u.abs();
                }
            },
            |_| {},
            &mut StageWorkspace::new(),
        );
    }

    #[test]
    fn test_relaxation_keeps_forced_steady_state_for_any_dt() {
        // Steady state F = Λ u*²: a fixed point of every relaxed stage,
        // however stiff Λ·dt is.
        let (forcing, drag) = (2.0_f64, 50.0);
        let u_star = (forcing / drag).sqrt();
        for integrator in [
            StandardIntegrator::SSPRK3,
            StandardIntegrator::SSPRK43,
            StandardIntegrator::ForwardEuler,
        ] {
            for dt in [1e-3, 1.0, 1e3] {
                let mut u = DGSolution1D::new(1, 2);
                u.data.fill(u_star);
                for _ in 0..10 {
                    step_forced_drag(&integrator, &mut u, dt, forcing, drag);
                }
                for &v in &u.data {
                    assert!(
                        (v - u_star).abs() < 1e-14,
                        "{}: dt = {dt}: {v} drifted from {u_star}",
                        integrator.name()
                    );
                }
            }
        }
    }

    #[test]
    fn test_relaxation_is_l_stable() {
        // Λ·dt → ∞ must remove the damped quantity within one step (relaxing
        // only the Euler substeps of SSP-RK3 would leave u/3)
        for integrator in [StandardIntegrator::SSPRK3, StandardIntegrator::SSPRK43] {
            let mut u = DGSolution1D::new(1, 1);
            u.data[0] = 1.0;
            step_forced_drag(&integrator, &mut u, 1e8, 0.0, 1.0);
            assert!(u.data[0] > 0.0 && u.data[0] < 1e-6, "{}", u.data[0]);
        }
    }

    #[test]
    fn test_relaxation_never_flips_sign() {
        // Pure drag, Λ|u|·dt up to 1e4: explicit SSP-RK3 would oscillate and
        // blow up; the relaxed steps decay monotonically towards zero.
        for dt in [0.1, 10.0, 1e4] {
            let mut u = DGSolution1D::new(1, 1);
            u.data[0] = 1.0;
            let mut previous = 1.0;
            for _ in 0..20 {
                step_forced_drag(&SSPRK3, &mut u, dt, 0.0, 1.0);
                let v = u.data[0];
                assert!(v > 0.0 && v < previous, "dt = {dt}: {previous} -> {v}");
                previous = v;
            }
        }
    }

    #[test]
    fn test_relaxation_converges_to_exact_decay() {
        // du/dt = −Λ|u|u has u(t) = u0 / (1 + Λ u0 t); the relaxed drag is
        // first-order accurate inside SSP-RK3.
        let error_at = |dt: f64| {
            let t_end = 1.0;
            let mut u = DGSolution1D::new(1, 1);
            u.data[0] = 1.0;
            for _ in 0..(t_end / dt).round() as usize {
                step_forced_drag(&SSPRK3, &mut u, dt, 0.0, 3.0);
            }
            (u.data[0] - 1.0 / (1.0 + 3.0 * t_end)).abs()
        };
        let (coarse, fine) = (error_at(0.02), error_at(0.01));
        let rate = (coarse / fine).log2();
        assert!(fine < 1e-2, "error {fine}");
        assert!(
            rate > 0.9,
            "convergence rate {rate} (errors {coarse}, {fine})"
        );
    }

    #[test]
    fn test_forward_euler_order() {
        // Test Euler with exponential decay: du/dt = -u, u(0) = 1
        // Exact: u(t) = exp(-t)
        let mut u = DGSolution1D::new(1, 3);
        u.data.fill(1.0);

        let integrator = ForwardEuler;
        let dt = 0.001;
        let n_steps = 100;

        for i in 0..n_steps {
            let t = dt * i as f64;
            integrator.step(&mut u, dt, t, |state, _time| {
                let mut rhs = state.clone();
                rhs.scale(-1.0);
                rhs
            });
        }

        let t = dt * n_steps as f64;
        let expected = (-t).exp();

        for &v in &u.data {
            let error = (v - expected).abs();
            // Forward Euler is 1st order, so error is O(dt)
            assert!(
                error < 0.02,
                "Expected {}, got {} (error {})",
                expected,
                v,
                error
            );
        }
    }

    #[test]
    fn test_standard_integrator_dispatch() {
        let mut u = DGSolution1D::new(1, 3);
        u.data.fill(1.0);

        // Test that enum dispatch works
        let integrator = StandardIntegrator::SSPRK3;
        integrator.step(&mut u, 0.01, 0.0, |state, _time| state.clone());

        // Values should have changed
        assert!(u.data[0] > 1.0);
    }

    #[test]
    fn test_integrator_names() {
        assert_eq!(SSPRK3.name(), "ssp-rk3");
        assert_eq!(ForwardEuler.name(), "forward-euler");
        assert_eq!(StandardIntegrator::SSPRK3.name(), "ssp-rk3");
    }

    #[test]
    fn test_stage_times() {
        let dt = 0.1;
        let times = SSPRK3.stage_times(dt);
        assert_eq!(times.len(), 3);
        assert!((times[0] - 0.0).abs() < 1e-14);
        assert!((times[1] - dt).abs() < 1e-14);
        assert!((times[2] - 0.5 * dt).abs() < 1e-14);
    }

    #[test]
    fn test_ssp_flag() {
        assert!(SSPRK3.is_ssp());
        assert!(ForwardEuler.is_ssp());
    }

    #[test]
    fn test_zeros_like() {
        let u = DGSolution1D::new(2, 3);
        let zeros = u.zeros_like();
        for &v in &zeros.data {
            assert!((v - 0.0).abs() < 1e-14);
        }
    }

    #[test]
    fn test_boxed_integrator_info() {
        let info = create_integrator_info(StandardIntegrator::SSPRK3);
        assert_eq!(info.name(), "ssp-rk3");
        assert_eq!(info.order(), 3);
        assert!(info.is_ssp());
    }
}

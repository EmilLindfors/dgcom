//! 2D Boundary conditions for shallow water equations.
//!
//! Extends the 1D boundary conditions to 2D, accounting for:
//! - 2D position (x, y)
//! - 2D normal vectors (nx, ny)
//! - Tangential velocity preservation
//!
//! Available boundary conditions:
//! - Reflective (wall): no-flux through boundary, tangential velocity preserved
//! - Tidal / harmonic tidal: prescribed (clamped) water surface elevation
//! - Discharge: prescribed normal flow rate
//! - Extrapolation and fixed state
//!
//! Open boundaries (radiation, tides, nesting) are
//! [`CharacteristicOBC`](crate::boundary::CharacteristicOBC).

use crate::mesh::BoundaryTag;
use crate::solver::SWEState2D;
use crate::types::Depth;

use super::HarmonicTide;
use crate::time::ModelClock;

/// Context for 2D boundary condition evaluation.
///
/// Provides all information needed to compute the ghost state at a boundary.
#[derive(Clone, Copy, Debug)]
pub struct BCContext2D {
    /// Current simulation time
    pub time: f64,
    /// Physical position of the boundary face (x, y)
    pub position: (f64, f64),
    /// Interior state at the boundary
    pub interior_state: SWEState2D,
    /// Bathymetry at the boundary
    pub bathymetry: f64,
    /// Outward unit normal direction (nx, ny)
    pub normal: (f64, f64),
    /// Gravitational acceleration
    pub g: f64,
    /// Minimum depth threshold
    pub h_min: f64,
    /// Optional boundary tag for multi-BC dispatch
    pub boundary_tag: Option<BoundaryTag>,
    /// Flat nodal index `k · n_nodes + i` of the boundary node, when known
    /// (set by the RHS kernels). Lets boundary data precomputed per node be
    /// looked up without searching by position.
    pub node_index: Option<usize>,
}

impl BCContext2D {
    /// Create a new 2D boundary condition context.
    ///
    /// For multi-BC dispatch, use `with_tag` to add the boundary tag.
    pub fn new(
        time: f64,
        position: (f64, f64),
        interior_state: SWEState2D,
        bathymetry: f64,
        normal: (f64, f64),
        g: f64,
        h_min: f64,
    ) -> Self {
        Self {
            time,
            position,
            interior_state,
            bathymetry,
            normal,
            g,
            h_min,
            boundary_tag: None,
            node_index: None,
        }
    }

    /// Create a context with a boundary tag for multi-BC dispatch.
    pub fn with_tag(
        time: f64,
        position: (f64, f64),
        interior_state: SWEState2D,
        bathymetry: f64,
        normal: (f64, f64),
        g: f64,
        h_min: f64,
        boundary_tag: BoundaryTag,
    ) -> Self {
        Self {
            time,
            position,
            interior_state,
            bathymetry,
            normal,
            g,
            h_min,
            boundary_tag: Some(boundary_tag),
            node_index: None,
        }
    }

    /// Set the flat nodal index of the boundary node.
    pub fn with_node_index(mut self, index: usize) -> Self {
        self.node_index = Some(index);
        self
    }

    /// Water surface elevation at the interior: η = h + B
    pub fn interior_surface_elevation(&self) -> f64 {
        self.interior_state.h + self.bathymetry
    }

    /// Interior velocity components (u, v).
    pub fn interior_velocity(&self) -> (f64, f64) {
        self.interior_state.velocity_simple(Depth::new(self.h_min))
    }

    /// Normal velocity component: u·n = u*nx + v*ny
    pub fn interior_normal_velocity(&self) -> f64 {
        let (u, v) = self.interior_velocity();
        let (nx, ny) = self.normal;
        u * nx + v * ny
    }

    /// Tangential velocity component: u×n = -u*ny + v*nx
    pub fn interior_tangential_velocity(&self) -> f64 {
        let (u, v) = self.interior_velocity();
        let (nx, ny) = self.normal;
        -u * ny + v * nx
    }

    /// Wave celerity at interior: c = sqrt(g*h)
    pub fn interior_celerity(&self) -> f64 {
        (self.g * self.interior_state.h.max(0.0)).sqrt()
    }
}

/// What a boundary condition supplies at a boundary face node.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BoundaryState {
    /// An exterior state: the face flux is the Riemann solver's `F*(q_int, q_ghost)`.
    Ghost(SWEState2D),
    /// The state on the boundary itself: the face flux is its physical flux
    /// `F(q_b)·n`, whatever the Riemann solver (see
    /// [`CharacteristicOBC`](crate::boundary::CharacteristicOBC)).
    Exact(SWEState2D),
}

impl BoundaryState {
    /// The state, either kind.
    pub fn state(&self) -> SWEState2D {
        match *self {
            Self::Ghost(q) | Self::Exact(q) => q,
        }
    }
}

/// Trait for 2D shallow water boundary conditions.
///
/// Implementations compute a "ghost" state that represents the exterior
/// state at a boundary face. This ghost state is then used in the
/// numerical flux computation.
pub trait SWEBoundaryCondition2D: Send + Sync {
    /// Compute the ghost state for flux evaluation.
    ///
    /// # Arguments
    /// * `ctx` - Boundary condition context with interior state and metadata
    ///
    /// # Returns
    /// The ghost (exterior) state to use in flux computation
    fn ghost_state(&self, ctx: &BCContext2D) -> SWEState2D;

    /// The boundary state used by the face flux: the ghost state for the
    /// Riemann solver by default; boundary conditions that construct the state
    /// on the boundary (characteristic OBCs) return [`BoundaryState::Exact`].
    /// Wrappers that dispatch to other boundary conditions must forward this.
    fn boundary_state(&self, ctx: &BCContext2D) -> BoundaryState {
        BoundaryState::Ghost(self.ghost_state(ctx))
    }

    /// Name of this boundary condition for debugging/logging.
    fn name(&self) -> &'static str;

    /// Check if this BC allows inflow (used for validation).
    fn allows_inflow(&self) -> bool {
        true
    }

    /// Check if this BC allows outflow (used for validation).
    fn allows_outflow(&self) -> bool {
        true
    }

    /// Whether faces tagged `tag` (`None`: untagged faces) are walls, with
    /// no mass flux, or open, for the 3D kernels' face classification
    /// ([`crate::solver::rhs::Boundaries3D::matching`]). `None` if the
    /// condition cannot tell (the default); the 3D kernels then treat
    /// [`BoundaryTag::Wall`] and untagged faces as walls. Wrappers that
    /// dispatch to other boundary conditions must forward this.
    fn is_wall(&self, _tag: Option<BoundaryTag>) -> Option<bool> {
        None
    }
}

/// Reflective (wall) boundary condition for 2D.
///
/// Creates a mirror state with reversed normal velocity and preserved
/// tangential velocity, ensuring zero mass flux through the boundary.
///
/// # Mathematical Formulation
///
/// For a boundary with outward normal n = (nx, ny):
/// - h_ghost = h_interior
/// - (u·n)_ghost = -(u·n)_interior  (normal velocity reversed)
/// - (u×n)_ghost = (u×n)_interior   (tangential velocity preserved)
///
/// This gives:
/// - u_ghost = u - 2*(u·n)*nx
/// - v_ghost = v - 2*(u·n)*ny
#[derive(Clone, Debug, Default)]
pub struct Reflective2D {
    /// Minimum depth for velocity computation
    pub h_min: f64,
}

impl Reflective2D {
    /// Create a new reflective BC.
    pub fn new() -> Self {
        Self { h_min: 1e-6 }
    }

    /// Create with custom minimum depth.
    pub fn with_h_min(h_min: f64) -> Self {
        Self { h_min }
    }
}

impl SWEBoundaryCondition2D for Reflective2D {
    fn is_wall(&self, _tag: Option<BoundaryTag>) -> Option<bool> {
        Some(true)
    }

    fn ghost_state(&self, ctx: &BCContext2D) -> SWEState2D {
        let h = ctx.interior_state.h;
        let (nx, ny) = ctx.normal;

        if h <= self.h_min {
            return SWEState2D::new(h, 0.0, 0.0);
        }

        let (u, v) = ctx.interior_velocity();

        // Reflect normal component, preserve tangential (free-slip wall).
        let (u_ghost, v_ghost) = super::reflect_velocity(u, v, nx, ny);

        SWEState2D::from_primitives(h, u_ghost, v_ghost)
    }

    fn name(&self) -> &'static str {
        "reflective_2d"
    }

    fn allows_inflow(&self) -> bool {
        false
    }

    fn allows_outflow(&self) -> bool {
        false
    }
}

/// Tidal boundary condition for 2D (Dirichlet for elevation).
///
/// Prescribes water surface elevation, extrapolates velocity.
/// Use Flather BC for better wave absorption.
#[derive(Clone, Debug)]
pub struct Tidal2D<F>
where
    F: Fn(f64, f64, f64) -> f64 + Send + Sync,
{
    /// Function returning tidal elevation η(x, y, t)
    pub tidal_elevation: F,
    /// Minimum depth
    pub h_min: f64,
}

impl<F> Tidal2D<F>
where
    F: Fn(f64, f64, f64) -> f64 + Send + Sync,
{
    /// Create a new tidal BC.
    pub fn new(tidal_elevation: F) -> Self {
        Self {
            tidal_elevation,
            h_min: 1e-6,
        }
    }
}

impl<F> SWEBoundaryCondition2D for Tidal2D<F>
where
    F: Fn(f64, f64, f64) -> f64 + Send + Sync,
{
    fn is_wall(&self, _tag: Option<BoundaryTag>) -> Option<bool> {
        Some(false)
    }

    fn ghost_state(&self, ctx: &BCContext2D) -> SWEState2D {
        let (x, y) = ctx.position;
        let t = ctx.time;

        // Prescribed surface elevation
        let eta_tidal = (self.tidal_elevation)(x, y, t);
        let h_ghost = (eta_tidal - ctx.bathymetry).max(self.h_min);

        // Extrapolate velocity from interior
        let (u, v) = ctx.interior_velocity();

        SWEState2D::from_primitives(h_ghost, u, v)
    }

    fn name(&self) -> &'static str {
        "tidal_2d"
    }
}

/// Discharge (river inflow) boundary condition for 2D.
///
/// Prescribes normal flow rate per unit width Q = h * u_n.
/// The depth is either prescribed or taken from interior.
#[derive(Clone, Debug)]
pub struct Discharge2D<F>
where
    F: Fn(f64, f64, f64) -> f64 + Send + Sync,
{
    /// Function returning discharge per unit width Q(x, y, t)
    pub discharge: F,
    /// Optional prescribed depth (if None, use interior depth)
    pub prescribed_depth: Option<f64>,
    /// Minimum depth
    pub h_min: f64,
}

impl<F> Discharge2D<F>
where
    F: Fn(f64, f64, f64) -> f64 + Send + Sync,
{
    /// Create a new discharge BC with interior depth.
    pub fn new(discharge: F) -> Self {
        Self {
            discharge,
            prescribed_depth: None,
            h_min: 1e-6,
        }
    }

    /// Create with prescribed depth.
    pub fn with_depth(discharge: F, depth: f64) -> Self {
        Self {
            discharge,
            prescribed_depth: Some(depth),
            h_min: 1e-6,
        }
    }
}

impl<F> SWEBoundaryCondition2D for Discharge2D<F>
where
    F: Fn(f64, f64, f64) -> f64 + Send + Sync,
{
    fn is_wall(&self, _tag: Option<BoundaryTag>) -> Option<bool> {
        Some(false)
    }

    fn ghost_state(&self, ctx: &BCContext2D) -> SWEState2D {
        let (x, y) = ctx.position;
        let t = ctx.time;
        let (nx, ny) = ctx.normal;

        // Get depth
        let h = self
            .prescribed_depth
            .unwrap_or(ctx.interior_state.h)
            .max(self.h_min);

        // Prescribed discharge gives normal velocity
        let q = (self.discharge)(x, y, t);
        let un = -q / h; // Negative because discharge flows into domain (against outward normal)

        // Zero tangential velocity for river inflow
        let ut = 0.0;

        // Convert to (u, v)
        let u = un * nx - ut * ny;
        let v = un * ny + ut * nx;

        SWEState2D::from_primitives(h, u, v)
    }

    fn name(&self) -> &'static str {
        "discharge_2d"
    }

    fn allows_outflow(&self) -> bool {
        false
    }
}

/// Extrapolation (zero-gradient) boundary condition.
///
/// Simply copies the interior state to the exterior.
#[derive(Clone, Debug, Default)]
pub struct Extrapolation2D;

impl SWEBoundaryCondition2D for Extrapolation2D {
    fn is_wall(&self, _tag: Option<BoundaryTag>) -> Option<bool> {
        Some(false)
    }

    fn ghost_state(&self, ctx: &BCContext2D) -> SWEState2D {
        ctx.interior_state
    }

    fn name(&self) -> &'static str {
        "extrapolation_2d"
    }
}

/// Fixed state (Dirichlet) boundary condition.
#[derive(Clone, Debug)]
pub struct FixedState2D {
    /// Fixed state to impose
    pub state: SWEState2D,
}

impl FixedState2D {
    /// Create a new fixed state BC.
    pub fn new(h: f64, hu: f64, hv: f64) -> Self {
        Self {
            state: SWEState2D::new(h, hu, hv),
        }
    }

    /// Create from primitive variables.
    pub fn from_primitives(h: f64, u: f64, v: f64) -> Self {
        Self {
            state: SWEState2D::from_primitives(h, u, v),
        }
    }
}

impl SWEBoundaryCondition2D for FixedState2D {
    fn is_wall(&self, _tag: Option<BoundaryTag>) -> Option<bool> {
        Some(false)
    }

    fn ghost_state(&self, _ctx: &BCContext2D) -> SWEState2D {
        self.state
    }

    fn name(&self) -> &'static str {
        "fixed_state_2d"
    }
}

// ============================================================================
// Convenience Boundary Conditions (Non-Generic)
// ============================================================================

/// Harmonic tidal constituent for 2D boundaries.
///
/// This is a re-export for convenience; see [`super::TidalConstituent`] for details.
pub use super::TidalConstituent;

/// Harmonic tidal elevation clamped at the boundary (Dirichlet for elevation).
///
/// Prescribes the elevation of a [`HarmonicTide`] and extrapolates the
/// interior velocity.
///
/// # Stability
///
/// This BC clamps the boundary elevation and extrapolates velocity, so in the
/// weak ghost-state setting it reflects outgoing waves (coefficient ≈ −1),
/// which can lead to phase errors and trapped energy. Prefer a
/// [`CharacteristicOBC`](crate::boundary::CharacteristicOBC) with the same
/// tide where outgoing waves must leave the domain.
///
/// # Bathymetry convention
///
/// The ghost depth is `η_tide − B`; with B = −h₀ for a basin of depth h₀ it
/// is h₀ at mean sea level.
#[derive(Clone, Debug)]
pub struct HarmonicTidal2D {
    /// The prescribed tide.
    pub tide: HarmonicTide,
    /// Minimum depth
    pub h_min: f64,
}

impl HarmonicTidal2D {
    /// Create a new harmonic tidal BC.
    pub fn new(constituents: Vec<TidalConstituent>) -> Self {
        Self::from_tide(HarmonicTide::new(constituents))
    }

    /// Clamp the elevation to `tide`.
    pub fn from_tide(tide: HarmonicTide) -> Self {
        Self { tide, h_min: 1e-6 }
    }

    /// Create with mean elevation.
    pub fn with_mean_elevation(mut self, mean: f64) -> Self {
        self.tide.mean_elevation = mean;
        self
    }

    /// Create M2-only.
    pub fn m2_only(amplitude: f64, phase: f64) -> Self {
        Self::new(vec![TidalConstituent::m2(amplitude, phase)])
    }

    /// Set minimum depth threshold.
    pub fn with_h_min(mut self, h_min: f64) -> Self {
        self.h_min = h_min;
        self
    }

    /// Ramp the tide up over `duration` seconds.
    pub fn with_ramp_up(mut self, duration: f64) -> Self {
        self.tide.ramp_duration = Some(duration);
        self
    }

    /// Nodal corrections for a run on `clock`; see
    /// [`HarmonicTide::with_nodal_corrections`].
    pub fn with_nodal_corrections(mut self, clock: &ModelClock) -> Self {
        self.tide = self.tide.with_nodal_corrections(clock);
        self
    }

    /// Surface elevation η(t).
    pub fn elevation(&self, t: f64) -> f64 {
        self.tide.elevation(t)
    }
}

impl SWEBoundaryCondition2D for HarmonicTidal2D {
    fn is_wall(&self, _tag: Option<BoundaryTag>) -> Option<bool> {
        Some(false)
    }

    fn ghost_state(&self, ctx: &BCContext2D) -> SWEState2D {
        let eta = self.elevation(ctx.time);
        let h_ghost = (eta - ctx.bathymetry).max(self.h_min);
        let (u, v) = ctx.interior_velocity();
        SWEState2D::from_primitives(h_ghost, u, v)
    }

    fn name(&self) -> &'static str {
        "harmonic_tidal_2d"
    }
}

/// Constant discharge BC (non-generic convenience).
///
/// Prescribes a constant flow rate per unit width.
#[derive(Clone, Debug)]
pub struct ConstantDischarge2D {
    /// Discharge per unit width (m²/s)
    pub discharge: f64,
    /// Optional prescribed depth
    pub prescribed_depth: Option<f64>,
    /// Minimum depth
    pub h_min: f64,
}

impl ConstantDischarge2D {
    /// Create with given discharge.
    ///
    /// # Arguments
    /// * `discharge` - Flow rate per unit width (m²/s), positive = into domain
    pub fn new(discharge: f64) -> Self {
        Self {
            discharge,
            prescribed_depth: None,
            h_min: 1e-6,
        }
    }

    /// Create with prescribed depth.
    pub fn with_depth(discharge: f64, depth: f64) -> Self {
        Self {
            discharge,
            prescribed_depth: Some(depth),
            h_min: 1e-6,
        }
    }
}

impl SWEBoundaryCondition2D for ConstantDischarge2D {
    fn is_wall(&self, _tag: Option<BoundaryTag>) -> Option<bool> {
        Some(false)
    }

    fn ghost_state(&self, ctx: &BCContext2D) -> SWEState2D {
        let (nx, ny) = ctx.normal;
        let h = self
            .prescribed_depth
            .unwrap_or(ctx.interior_state.h)
            .max(self.h_min);

        // Normal velocity from discharge (negative = inflow)
        let un = -self.discharge / h;
        let ut = 0.0;

        let u = un * nx - ut * ny;
        let v = un * ny + ut * nx;

        SWEState2D::from_primitives(h, u, v)
    }

    fn name(&self) -> &'static str {
        "constant_discharge_2d"
    }

    fn allows_outflow(&self) -> bool {
        self.discharge < 0.0 // Negative discharge means outflow
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-12;
    const G: f64 = 10.0;
    const H_MIN: f64 = 1e-6;

    fn make_context(h: f64, hu: f64, hv: f64, normal: (f64, f64)) -> BCContext2D {
        BCContext2D::new(
            0.0,
            (0.0, 0.0),
            SWEState2D::new(h, hu, hv),
            0.0,
            normal,
            G,
            H_MIN,
        )
    }

    #[test]
    fn test_reflective_still_water() {
        let bc = Reflective2D::new();
        let ctx = make_context(2.0, 0.0, 0.0, (1.0, 0.0));

        let ghost = bc.ghost_state(&ctx);

        assert!((ghost.h - 2.0).abs() < TOL);
        assert!(ghost.hu.abs() < TOL);
        assert!(ghost.hv.abs() < TOL);
    }

    #[test]
    fn test_reflective_normal_flow() {
        let bc = Reflective2D::new();

        // Flow towards x-boundary (normal = (1, 0))
        // h=2, u=3, v=0
        let ctx = make_context(2.0, 6.0, 0.0, (1.0, 0.0));
        let ghost = bc.ghost_state(&ctx);

        // Normal velocity should be reversed
        assert!((ghost.h - 2.0).abs() < TOL);
        assert!((ghost.hu - (-6.0)).abs() < TOL); // u reversed
        assert!(ghost.hv.abs() < TOL); // v unchanged
    }

    #[test]
    fn test_reflective_tangential_preserved() {
        let bc = Reflective2D::new();

        // Pure tangential flow (v=3, normal = (1, 0))
        let ctx = make_context(2.0, 0.0, 6.0, (1.0, 0.0));
        let ghost = bc.ghost_state(&ctx);

        // Tangential velocity should be preserved
        assert!((ghost.h - 2.0).abs() < TOL);
        assert!(ghost.hu.abs() < TOL); // No normal velocity
        assert!((ghost.hv - 6.0).abs() < TOL); // Tangential preserved
    }

    #[test]
    fn test_reflective_diagonal_normal() {
        let bc = Reflective2D::new();

        // 45-degree normal
        let sqrt2_inv = 1.0 / 2.0_f64.sqrt();
        let ctx = make_context(1.0, 2.0, 0.0, (sqrt2_inv, sqrt2_inv));
        let ghost = bc.ghost_state(&ctx);

        // Normal component: u·n = 2 * 1/√2 = √2
        // Reflected: u_ghost = u - 2*(u·n)*n
        let un = 2.0 * sqrt2_inv;
        let u_expected = 2.0 - 2.0 * un * sqrt2_inv;
        let v_expected = 0.0 - 2.0 * un * sqrt2_inv;

        assert!((ghost.h - 1.0).abs() < TOL);
        assert!((ghost.hu / ghost.h - u_expected).abs() < TOL);
        assert!((ghost.hv / ghost.h - v_expected).abs() < TOL);
    }

    #[test]
    fn test_reflective_zero_normal_flux() {
        let bc = Reflective2D::new();

        // Various flow directions
        for (u, v) in [(3.0, 0.0), (0.0, 3.0), (2.0, 1.0)] {
            let h = 2.0;
            let ctx = make_context(h, h * u, h * v, (1.0, 0.0));
            let ghost = bc.ghost_state(&ctx);

            // Average normal velocity should be zero
            let un_int = u;
            let un_ghost = ghost.hu / ghost.h;
            let un_avg = 0.5 * (un_int + un_ghost);

            assert!(
                un_avg.abs() < TOL,
                "Average normal velocity should be zero, got {}",
                un_avg
            );
        }
    }

    #[test]
    fn test_extrapolation() {
        let bc = Extrapolation2D;
        let ctx = make_context(2.0, 3.0, 4.0, (1.0, 0.0));

        let ghost = bc.ghost_state(&ctx);

        assert!((ghost.h - 2.0).abs() < TOL);
        assert!((ghost.hu - 3.0).abs() < TOL);
        assert!((ghost.hv - 4.0).abs() < TOL);
    }

    #[test]
    fn test_fixed_state() {
        let bc = FixedState2D::from_primitives(1.5, 0.5, 0.25);
        let ctx = make_context(2.0, 6.0, 4.0, (1.0, 0.0));

        let ghost = bc.ghost_state(&ctx);

        assert!((ghost.h - 1.5).abs() < TOL);
        assert!((ghost.hu - 0.75).abs() < TOL); // 1.5 * 0.5
        assert!((ghost.hv - 0.375).abs() < TOL); // 1.5 * 0.25
    }

    #[test]
    fn test_tidal_elevation() {
        // Constant tidal elevation
        let bc = Tidal2D::new(|_x, _y, _t| 0.5);
        let ctx = BCContext2D::new(
            0.0,
            (0.0, 0.0),
            SWEState2D::from_primitives(2.0, 1.0, 0.5),
            -0.5, // Bathymetry
            (1.0, 0.0),
            G,
            H_MIN,
        );

        let ghost = bc.ghost_state(&ctx);

        // h = η - B = 0.5 - (-0.5) = 1.0
        assert!((ghost.h - 1.0).abs() < TOL);
        // Velocity extrapolated from interior
        assert!((ghost.hu / ghost.h - 1.0).abs() < TOL);
        assert!((ghost.hv / ghost.h - 0.5).abs() < TOL);
    }

    #[test]
    fn test_discharge() {
        // Constant discharge of 5 m²/s
        let bc = Discharge2D::with_depth(|_x, _y, _t| 5.0, 2.0);
        let ctx = make_context(2.0, 0.0, 0.0, (1.0, 0.0));

        let ghost = bc.ghost_state(&ctx);

        // h = 2.0 (prescribed)
        assert!((ghost.h - 2.0).abs() < TOL);
        // u_n = -Q/h = -5/2 = -2.5 (flowing into domain)
        assert!((ghost.hu / ghost.h - (-2.5)).abs() < TOL);
        // v = 0
        assert!(ghost.hv.abs() < TOL);
    }

    #[test]
    fn test_context_methods() {
        let ctx = BCContext2D::new(
            1.0,
            (5.0, 10.0),
            SWEState2D::from_primitives(2.0, 3.0, 1.0),
            0.5,
            (0.6, 0.8),
            G,
            H_MIN,
        );

        // Surface elevation
        assert!((ctx.interior_surface_elevation() - 2.5).abs() < TOL);

        // Normal velocity: u·n = 3*0.6 + 1*0.8 = 2.6
        assert!((ctx.interior_normal_velocity() - 2.6).abs() < TOL);

        // Tangential velocity: -u*ny + v*nx = -3*0.8 + 1*0.6 = -1.8
        assert!((ctx.interior_tangential_velocity() - (-1.8)).abs() < TOL);

        // Celerity
        assert!((ctx.interior_celerity() - (G * 2.0_f64).sqrt()).abs() < TOL);
    }

    #[test]
    fn test_harmonic_tidal_clamps_elevation() {
        let clock = ModelClock::at_datetime(2024, 6, 1, 0, 0);
        let bc = HarmonicTidal2D::m2_only(0.5, 0.0)
            .with_mean_elevation(0.1)
            .with_nodal_corrections(&clock);
        let n = clock.nodal_correction("M2", 0.0).unwrap();
        let eta = 0.1 + n.f * 0.5 * n.phase_offset_rad().cos();
        let ctx = BCContext2D::new(
            0.0,
            (0.0, 0.0),
            SWEState2D::from_primitives(20.0, 0.3, -0.2),
            -20.0,
            (1.0, 0.0),
            G,
            H_MIN,
        );
        let ghost = bc.ghost_state(&ctx);
        assert!((ghost.h - (20.0 + eta)).abs() < TOL);
        assert!((ghost.hu / ghost.h - 0.3).abs() < TOL);
    }
}

//! Trait-based numerical flux abstraction.
//!
//! This module provides a trait-based interface for numerical fluxes,
//! enabling extensible flux implementations while maintaining performance
//! through both generic and dynamic dispatch options.
//!
//! # Example
//! ```
//! use dgcom::flux::{NumericalFlux2D, FluxContext2D, RoeFlux2D, HLLFlux2D};
//! use dgcom::solver::SWEState2D;
//!
//! let q_l = SWEState2D::new(2.0, 1.0, 0.0);
//! let q_r = SWEState2D::new(1.0, 0.5, 0.0);
//! let ctx = FluxContext2D {
//!     g: 9.81,
//!     h_min: 1e-6,
//!     normal: (1.0, 0.0),
//! };
//!
//! // Using concrete type
//! let roe = RoeFlux2D;
//! let flux_roe = roe.compute(&q_l, &q_r, &ctx);
//!
//! // Using trait object
//! let flux_fn: &dyn NumericalFlux2D = &HLLFlux2D;
//! let flux_hll = flux_fn.compute(&q_l, &q_r, &ctx);
//! ```

use crate::solver::SWEState2D;

// =============================================================================
// Flux Context
// =============================================================================

/// Context provided to numerical flux computations.
///
/// Contains physical parameters and geometric information needed
/// to evaluate the flux at an element interface.
#[derive(Clone, Copy, Debug)]
pub struct FluxContext2D {
    /// Gravitational acceleration (m/s²).
    pub g: f64,
    /// Minimum depth for velocity desingularization.
    pub h_min: f64,
    /// Outward unit normal vector (nx, ny).
    pub normal: (f64, f64),
}

impl FluxContext2D {
    /// Create a new flux context.
    #[inline]
    pub fn new(g: f64, h_min: f64, normal: (f64, f64)) -> Self {
        Self { g, h_min, normal }
    }
}

// =============================================================================
// Numerical Flux Trait
// =============================================================================

/// Trait for numerical flux functions in 2D shallow water equations.
///
/// A numerical flux computes the flux F* · n at an element interface
/// given the left (interior) and right (exterior) states.
///
/// # Implementation Notes
///
/// - Fluxes should be consistent: F*(q, q) = F(q) · n
/// - Fluxes should be conservative: F*(q_l, q_r; n) = -F*(q_r, q_l; -n)
/// - The `compute` method should not allocate memory
///
/// # Extending
///
/// To add a new flux type:
/// 1. Create a struct (can be zero-sized if no parameters)
/// 2. Implement `NumericalFlux2D` for it
/// 3. Optionally add it to `StandardFlux2D` enum for zero-cost dispatch
pub trait NumericalFlux2D: Send + Sync {
    /// Compute the numerical flux at an interface.
    ///
    /// # Arguments
    /// * `q_l` - Left (interior) state
    /// * `q_r` - Right (exterior/ghost) state
    /// * `ctx` - Flux context with parameters and normal
    ///
    /// # Returns
    /// The numerical flux F* · n as a state vector.
    fn compute(&self, q_l: &SWEState2D, q_r: &SWEState2D, ctx: &FluxContext2D) -> SWEState2D;

    /// Human-readable name for debugging and logging.
    fn name(&self) -> &'static str;

    /// Whether this flux satisfies an entropy inequality.
    ///
    /// Entropy-stable fluxes are preferred for robustness.
    fn is_entropy_stable(&self) -> bool {
        false
    }

    /// Recommended CFL factor for this flux.
    ///
    /// Some fluxes (like Rusanov) may require lower CFL numbers.
    fn recommended_cfl(&self) -> f64 {
        0.5
    }
}

// =============================================================================
// Concrete Flux Implementations
// =============================================================================

/// Roe approximate Riemann solver.
///
/// Computes the flux using Roe-averaged eigenvalues and eigenvectors.
/// Accurate for smooth flows and contact discontinuities.
#[derive(Clone, Copy, Debug, Default)]
pub struct RoeFlux2D;

impl NumericalFlux2D for RoeFlux2D {
    #[inline]
    fn compute(&self, q_l: &SWEState2D, q_r: &SWEState2D, ctx: &FluxContext2D) -> SWEState2D {
        super::roe_flux_swe_2d(q_l, q_r, ctx.normal, ctx.g, ctx.h_min)
    }

    fn name(&self) -> &'static str {
        "roe"
    }

    fn is_entropy_stable(&self) -> bool {
        false // Standard Roe can violate entropy at sonic points
    }
}

/// HLL (Harten-Lax-van Leer) Riemann solver.
///
/// More robust than Roe for strong shocks but more diffusive.
/// Uses bounds on the fastest wave speeds.
#[derive(Clone, Copy, Debug, Default)]
pub struct HLLFlux2D;

impl NumericalFlux2D for HLLFlux2D {
    #[inline]
    fn compute(&self, q_l: &SWEState2D, q_r: &SWEState2D, ctx: &FluxContext2D) -> SWEState2D {
        super::hll_flux_swe_2d(q_l, q_r, ctx.normal, ctx.g, ctx.h_min)
    }

    fn name(&self) -> &'static str {
        "hll"
    }

    fn is_entropy_stable(&self) -> bool {
        true // HLL satisfies entropy conditions
    }
}

/// Rusanov (Local Lax-Friedrichs) flux.
///
/// Simple and robust but diffusive. Uses the maximum wave speed.
/// Good choice when robustness is more important than accuracy.
#[derive(Clone, Copy, Debug, Default)]
pub struct RusanovFlux2D;

impl NumericalFlux2D for RusanovFlux2D {
    #[inline]
    fn compute(&self, q_l: &SWEState2D, q_r: &SWEState2D, ctx: &FluxContext2D) -> SWEState2D {
        super::rusanov_flux_swe_2d(q_l, q_r, ctx.normal, ctx.g, ctx.h_min)
    }

    fn name(&self) -> &'static str {
        "rusanov"
    }

    fn is_entropy_stable(&self) -> bool {
        true // Rusanov is entropy-stable
    }

    fn recommended_cfl(&self) -> f64 {
        0.4 // Slightly more conservative due to maximum wave speed estimate
    }
}

// =============================================================================
// Standard Flux Enum (Zero-Cost Dispatch)
// =============================================================================

/// Enum wrapper for built-in flux types.
///
/// This provides zero-cost dispatch when the flux type is known at compile time,
/// while still allowing runtime selection via enum matching.
///
/// # Performance
///
/// Using this enum directly is as fast as calling the flux functions directly,
/// whereas using `&dyn NumericalFlux2D` incurs virtual dispatch overhead.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum StandardFlux2D {
    /// Roe approximate Riemann solver
    #[default]
    Roe,
    /// HLL solver (more robust for strong shocks)
    HLL,
    /// Rusanov/Local Lax-Friedrichs (simple, robust, diffusive)
    Rusanov,
}

impl NumericalFlux2D for StandardFlux2D {
    #[inline]
    fn compute(&self, q_l: &SWEState2D, q_r: &SWEState2D, ctx: &FluxContext2D) -> SWEState2D {
        match self {
            StandardFlux2D::Roe => super::roe_flux_swe_2d(q_l, q_r, ctx.normal, ctx.g, ctx.h_min),
            StandardFlux2D::HLL => super::hll_flux_swe_2d(q_l, q_r, ctx.normal, ctx.g, ctx.h_min),
            StandardFlux2D::Rusanov => {
                super::rusanov_flux_swe_2d(q_l, q_r, ctx.normal, ctx.g, ctx.h_min)
            }
        }
    }

    fn name(&self) -> &'static str {
        match self {
            StandardFlux2D::Roe => "roe",
            StandardFlux2D::HLL => "hll",
            StandardFlux2D::Rusanov => "rusanov",
        }
    }

    fn is_entropy_stable(&self) -> bool {
        match self {
            StandardFlux2D::Roe => false,
            StandardFlux2D::HLL | StandardFlux2D::Rusanov => true,
        }
    }

    fn recommended_cfl(&self) -> f64 {
        match self {
            StandardFlux2D::Roe | StandardFlux2D::HLL => 0.5,
            StandardFlux2D::Rusanov => 0.4,
        }
    }
}

impl From<super::SWEFluxType2D> for StandardFlux2D {
    fn from(flux_type: super::SWEFluxType2D) -> Self {
        match flux_type {
            super::SWEFluxType2D::Roe => StandardFlux2D::Roe,
            super::SWEFluxType2D::HLL => StandardFlux2D::HLL,
            super::SWEFluxType2D::Rusanov => StandardFlux2D::Rusanov,
        }
    }
}

impl From<StandardFlux2D> for super::SWEFluxType2D {
    fn from(flux: StandardFlux2D) -> Self {
        match flux {
            StandardFlux2D::Roe => super::SWEFluxType2D::Roe,
            StandardFlux2D::HLL => super::SWEFluxType2D::HLL,
            StandardFlux2D::Rusanov => super::SWEFluxType2D::Rusanov,
        }
    }
}

// =============================================================================
// Boxed Flux (Runtime Polymorphism)
// =============================================================================

/// Type alias for boxed flux (runtime polymorphism).
///
/// Use this when you need to store different flux types in collections
/// or select flux type at runtime based on configuration.
pub type BoxedFlux2D = Box<dyn NumericalFlux2D>;

/// Create a boxed flux from a flux type enum.
///
/// Useful for configuration-driven flux selection.
pub fn create_flux(flux_type: StandardFlux2D) -> BoxedFlux2D {
    match flux_type {
        StandardFlux2D::Roe => Box::new(RoeFlux2D),
        StandardFlux2D::HLL => Box::new(HLLFlux2D),
        StandardFlux2D::Rusanov => Box::new(RusanovFlux2D),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const G: f64 = 10.0;
    const H_MIN: f64 = 1e-6;
    const TOL: f64 = 1e-10;

    fn test_state() -> SWEState2D {
        SWEState2D::new(2.0, 6.0, 2.0) // h=2, u=3, v=1
    }

    fn x_normal_ctx() -> FluxContext2D {
        FluxContext2D::new(G, H_MIN, (1.0, 0.0))
    }

    #[test]
    fn test_flux_trait_consistency() {
        // For continuous state, flux should equal physical flux
        let state = test_state();
        let ctx = x_normal_ctx();

        let roe = RoeFlux2D;
        let hll = HLLFlux2D;
        let rusanov = RusanovFlux2D;

        let flux_roe = roe.compute(&state, &state, &ctx);
        let flux_hll = hll.compute(&state, &state, &ctx);
        let flux_rus = rusanov.compute(&state, &state, &ctx);

        // All should give same result for continuous state
        assert!((flux_roe.h - flux_hll.h).abs() < TOL);
        assert!((flux_roe.hu - flux_hll.hu).abs() < TOL);
        assert!((flux_roe.hv - flux_hll.hv).abs() < TOL);

        assert!((flux_roe.h - flux_rus.h).abs() < TOL);
        assert!((flux_roe.hu - flux_rus.hu).abs() < TOL);
        assert!((flux_roe.hv - flux_rus.hv).abs() < TOL);
    }

    #[test]
    fn test_standard_flux_enum() {
        let state = test_state();
        let ctx = x_normal_ctx();

        // Enum dispatch should match concrete type dispatch
        let roe_concrete = RoeFlux2D.compute(&state, &state, &ctx);
        let roe_enum = StandardFlux2D::Roe.compute(&state, &state, &ctx);

        assert!((roe_concrete.h - roe_enum.h).abs() < TOL);
        assert!((roe_concrete.hu - roe_enum.hu).abs() < TOL);
        assert!((roe_concrete.hv - roe_enum.hv).abs() < TOL);
    }

    #[test]
    fn test_flux_trait_object() {
        let state = test_state();
        let ctx = x_normal_ctx();

        // Test with trait object
        let flux: &dyn NumericalFlux2D = &HLLFlux2D;
        let result = flux.compute(&state, &state, &ctx);

        // Should match direct call
        let direct = HLLFlux2D.compute(&state, &state, &ctx);
        assert!((result.h - direct.h).abs() < TOL);
    }

    #[test]
    fn test_boxed_flux() {
        let state = test_state();
        let ctx = x_normal_ctx();

        let flux = create_flux(StandardFlux2D::Rusanov);
        let result = flux.compute(&state, &state, &ctx);

        let direct = RusanovFlux2D.compute(&state, &state, &ctx);
        assert!((result.h - direct.h).abs() < TOL);
    }

    #[test]
    fn test_flux_names() {
        assert_eq!(RoeFlux2D.name(), "roe");
        assert_eq!(HLLFlux2D.name(), "hll");
        assert_eq!(RusanovFlux2D.name(), "rusanov");
        assert_eq!(StandardFlux2D::Roe.name(), "roe");
    }

    #[test]
    fn test_entropy_stability_flags() {
        assert!(!RoeFlux2D.is_entropy_stable());
        assert!(HLLFlux2D.is_entropy_stable());
        assert!(RusanovFlux2D.is_entropy_stable());
    }

    #[test]
    fn test_flux_type_conversion() {
        assert_eq!(
            StandardFlux2D::from(super::super::SWEFluxType2D::Roe),
            StandardFlux2D::Roe
        );
        assert_eq!(
            super::super::SWEFluxType2D::from(StandardFlux2D::HLL),
            super::super::SWEFluxType2D::HLL
        );
    }
}

//! Horizontal viscosity models for 2D shallow water equations.
//!
//! Adds turbulent viscosity diffusion to the momentum equations:
//!   ∂(hu)/∂t + ... = ... + ∇·(ν h ∇u)
//!   ∂(hv)/∂t + ... = ... + ∇·(ν h ∇v)
//!
//! The eddy viscosity is a constant background plus, optionally,
//! Smagorinsky's (1963) strain-dependent part:
//!
//! ```text
//!     ν = ν₀ + (C_s Δ)² |S|,    Δ = √(area)/N,
//! ```
//!
//! `|S|` the magnitude of the strain rate tensor and `Δ` the node spacing of
//! an element of order `N` (the filter width of the 3D shear's viscosity,
//! `solver::rhs::viscosity_3d`, too).
//!
//! These terms are injected directly in the RHS (not via `SourceTerm2D`)
//! because they require access to differentiation operators and geometric
//! factors for gradient computation.

/// Horizontal viscosity configuration for 2D SWE momentum diffusion:
/// `ν = ν₀ + (C_s Δ)²|S|` (m²/s; see the [module documentation](self)).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HorizontalViscosity2D {
    /// The constant background `ν₀` (m²/s).
    pub background: f64,
    /// Smagorinsky's coefficient `C_s` (typically 0.1–0.2; 0 for none).
    pub smagorinsky: f64,
    /// Depth (m) at or below which a node carries no viscosity, and takes
    /// its element out of the viscous domain (`solver::rhs::swe_2d_viscosity`).
    pub h_min: f64,
}

impl HorizontalViscosity2D {
    /// Create a constant eddy viscosity model.
    ///
    /// # Arguments
    /// * `nu` - Constant viscosity coefficient [m²/s]
    pub fn constant(nu: f64) -> Self {
        Self {
            background: nu,
            smagorinsky: 0.0,
            h_min: 1e-3,
        }
    }

    /// Create a Smagorinsky viscosity model, without a background.
    ///
    /// # Arguments
    /// * `cs` - Smagorinsky coefficient (typically 0.1–0.2)
    pub fn smagorinsky(cs: f64) -> Self {
        Self {
            background: 0.0,
            smagorinsky: cs,
            h_min: 1e-3,
        }
    }

    /// The same model with the constant background `nu` (m²/s) added.
    pub fn with_background(mut self, nu: f64) -> Self {
        self.background = nu;
        self
    }

    /// Whether ν follows the strain (a nonzero Smagorinsky coefficient).
    pub fn is_strain_dependent(&self) -> bool {
        self.smagorinsky != 0.0
    }

    /// Smagorinsky's filter width `Δ = √(area)/N`, the node spacing of an
    /// element of area `area` and order `order`.
    #[inline]
    pub fn filter_width(area: f64, order: usize) -> f64 {
        area.sqrt() / order.max(1) as f64
    }

    /// Compute the viscosity at a point given velocity gradients and the
    /// filter width.
    ///
    /// # Arguments
    /// * `du_dx`, `du_dy` - Velocity gradients of u
    /// * `dv_dx`, `dv_dy` - Velocity gradients of v
    /// * `delta` - Filter width ([`Self::filter_width`])
    pub fn compute_viscosity(
        &self,
        du_dx: f64,
        du_dy: f64,
        dv_dx: f64,
        dv_dy: f64,
        delta: f64,
    ) -> f64 {
        if self.smagorinsky == 0.0 {
            return self.background;
        }
        let cs_delta = self.smagorinsky * delta;
        self.background + cs_delta * cs_delta * strain_rate_magnitude(du_dx, du_dy, dv_dx, dv_dy)
    }
}

/// The magnitude `|S| = √(2·(S₁₁² + S₂₂² + 2·S₁₂²))` of the horizontal
/// strain rate tensor, `S₁₁ = ∂u/∂x`, `S₂₂ = ∂v/∂y`,
/// `S₁₂ = ½·(∂u/∂y + ∂v/∂x)`.
#[inline]
pub fn strain_rate_magnitude(du_dx: f64, du_dy: f64, dv_dx: f64, dv_dy: f64) -> f64 {
    let s12 = 0.5 * (du_dy + dv_dx);
    (2.0 * (du_dx * du_dx + dv_dy * dv_dy + 2.0 * s12 * s12)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_viscosity() {
        let visc = HorizontalViscosity2D::constant(1.5);
        // Should return constant regardless of strain
        assert_eq!(visc.compute_viscosity(1.0, 2.0, 3.0, 4.0, 100.0), 1.5);
        assert_eq!(visc.compute_viscosity(0.0, 0.0, 0.0, 0.0, 100.0), 1.5);
    }

    #[test]
    fn test_smagorinsky_zero_strain() {
        let visc = HorizontalViscosity2D::smagorinsky(0.1);
        // Zero velocity gradients → zero viscosity
        let nu = visc.compute_viscosity(0.0, 0.0, 0.0, 0.0, 100.0);
        assert!(
            nu.abs() < 1e-15,
            "Zero strain should give zero viscosity, got {nu}"
        );
    }

    #[test]
    fn test_smagorinsky_pure_shear() {
        let cs = 0.1;
        let delta = 100.0;
        let visc = HorizontalViscosity2D::smagorinsky(cs);

        // Pure shear: du/dy = 1.0, dv/dx = 1.0, all others zero
        // S₁₁ = 0, S₂₂ = 0, S₁₂ = 0.5·(1+1) = 1.0
        // |S| = √(2·(0 + 0 + 2·1²)) = √4 = 2
        // ν = (0.1·100)² · 2 = 100 · 2 = 200
        let nu = visc.compute_viscosity(0.0, 1.0, 1.0, 0.0, delta);
        assert!(
            (nu - 200.0).abs() < 1e-10,
            "Pure shear: expected 200.0, got {nu}"
        );
    }

    #[test]
    fn test_smagorinsky_scaling() {
        let delta = 50.0;
        let visc1 = HorizontalViscosity2D::smagorinsky(0.1);
        let visc2 = HorizontalViscosity2D::smagorinsky(0.2);

        // Same strain field
        let nu1 = visc1.compute_viscosity(1.0, 0.5, -0.5, -1.0, delta);
        let nu2 = visc2.compute_viscosity(1.0, 0.5, -0.5, -1.0, delta);

        // ν ∝ cs², so doubling cs should give 4× viscosity
        let ratio = nu2 / nu1;
        assert!(
            (ratio - 4.0).abs() < 1e-10,
            "Doubling cs should give 4x viscosity, got ratio {ratio}"
        );
    }

    #[test]
    fn background_adds_to_smagorinsky() {
        let smagorinsky = HorizontalViscosity2D::smagorinsky(0.1);
        let combined = smagorinsky.with_background(0.5);
        let strained = |v: HorizontalViscosity2D| v.compute_viscosity(0.0, 1.0, 1.0, 0.0, 100.0);
        assert_eq!(strained(combined), 0.5 + strained(smagorinsky));
        assert_eq!(combined.compute_viscosity(0.0, 0.0, 0.0, 0.0, 100.0), 0.5);
        assert!(combined.is_strain_dependent());
        assert!(!HorizontalViscosity2D::constant(0.5).is_strain_dependent());
    }

    #[test]
    fn filter_width_is_the_node_spacing() {
        // A 90 m square: 90, 45, 30 m between nodes at P1, P2, P3
        for (order, spacing) in [(1, 90.0), (2, 45.0), (3, 30.0)] {
            let width = HorizontalViscosity2D::filter_width(8100.0, order);
            assert!((width - spacing).abs() < 1e-12, "P{order}: {width}");
        }
    }
}

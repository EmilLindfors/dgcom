//! 2D Shallow Water Equations.
//!
//! The 2D shallow water equations (depth-averaged free-surface flow):
//!
//! ∂h/∂t + ∂(hu)/∂x + ∂(hv)/∂y = 0                                    (mass)
//! ∂(hu)/∂t + ∂(hu² + gh²/2)/∂x + ∂(huv)/∂y = fhv - gh ∂B/∂x          (x-momentum)
//! ∂(hv)/∂t + ∂(huv)/∂x + ∂(hv² + gh²/2)/∂y = -fhu - gh ∂B/∂y         (y-momentum)
//!
//! where:
//! - h = water depth
//! - u, v = depth-averaged velocities in x and y directions
//! - hu, hv = momenta
//! - g = gravitational acceleration
//! - B = bottom topography (bathymetry)
//! - f = Coriolis parameter (f-plane: constant, β-plane: f = f₀ + βy)
//!
//! The Coriolis source terms and bathymetry terms are handled separately.
//!
//! # Flux formulation
//!
//! F(q) = [hu, hu² + gh²/2, huv]ᵀ  (x-direction flux)
//! G(q) = [hv, huv, hv² + gh²/2]ᵀ  (y-direction flux)
//!
//! # Norwegian coast context
//!
//! At 60°N latitude (Norwegian coast):
//! - f ≈ 1.2×10⁻⁴ s⁻¹
//! - Coriolis effects are significant for mesoscale circulation

use crate::solver::SWEState2D;
use crate::types::Depth;

/// 2D Shallow Water Equations with optional Coriolis.
///
/// Supports both f-plane (constant Coriolis) and β-plane (linear variation)
/// approximations for the Coriolis parameter.
///
/// # Example
///
/// ```
/// use dgcom::equations::ShallowWater2D;
///
/// // Norwegian coast at 60°N
/// let swe = ShallowWater2D::with_coriolis(9.81, 1.2e-4);
///
/// // Compute fluxes
/// let state = dgcom::SWEState2D::from_primitives(2.0, 1.0, 0.5);
/// let fx = swe.flux_x(&state);
/// let fy = swe.flux_y(&state);
/// ```
#[derive(Clone, Debug)]
pub struct ShallowWater2D {
    /// Gravitational acceleration (default 9.81 m/s²)
    pub g: f64,
    /// Minimum depth for wet/dry treatment (default 1e-6 m)
    pub h_min: Depth,
    /// Coriolis parameter f₀ for f-plane (s⁻¹)
    pub f0: f64,
    /// Beta-plane parameter β = ∂f/∂y (m⁻¹ s⁻¹)
    pub beta: f64,
    /// Reference latitude y₀ for beta-plane (m)
    pub y0: f64,
}

impl ShallowWater2D {
    /// Default minimum depth (1e-6 m)
    const DEFAULT_H_MIN: f64 = 1e-6;

    /// Create 2D shallow water equations with gravity only (no Coriolis).
    pub fn new(g: f64) -> Self {
        Self {
            g,
            h_min: Depth::new(Self::DEFAULT_H_MIN),
            f0: 0.0,
            beta: 0.0,
            y0: 0.0,
        }
    }

    /// Create with custom minimum depth threshold.
    pub fn with_h_min(g: f64, h_min: Depth) -> Self {
        Self {
            g,
            h_min,
            f0: 0.0,
            beta: 0.0,
            y0: 0.0,
        }
    }

    /// Create with constant Coriolis (f-plane approximation).
    ///
    /// # Arguments
    /// * `g` - Gravitational acceleration
    /// * `f` - Coriolis parameter (typical: 1.2e-4 s⁻¹ at 60°N)
    pub fn with_coriolis(g: f64, f: f64) -> Self {
        Self {
            g,
            h_min: Depth::new(Self::DEFAULT_H_MIN),
            f0: f,
            beta: 0.0,
            y0: 0.0,
        }
    }

    /// Create with beta-plane Coriolis: f(y) = f₀ + β(y - y₀).
    ///
    /// # Arguments
    /// * `g` - Gravitational acceleration
    /// * `f0` - Coriolis parameter at reference latitude
    /// * `beta` - Rate of change of f with latitude (∂f/∂y)
    /// * `y0` - Reference y-coordinate (typically 0)
    pub fn with_beta_plane(g: f64, f0: f64, beta: f64, y0: f64) -> Self {
        Self {
            g,
            h_min: Depth::new(Self::DEFAULT_H_MIN),
            f0,
            beta,
            y0,
        }
    }

    /// Create with standard gravity (9.81 m/s²) and no Coriolis.
    pub fn standard() -> Self {
        Self::new(9.81)
    }

    /// Create configured for Norwegian coast modeling.
    ///
    /// Uses f ≈ 1.2×10⁻⁴ s⁻¹ appropriate for 60°N latitude.
    pub fn norwegian_coast() -> Self {
        Self::with_coriolis(9.81, 1.2e-4)
    }

    /// Compute the Coriolis parameter at a given y-coordinate.
    ///
    /// f(y) = f₀ + β(y - y₀)
    #[inline]
    pub fn coriolis_at(&self, y: f64) -> f64 {
        self.f0 + self.beta * (y - self.y0)
    }

    /// Compute velocity from state with desingularization.
    ///
    /// (u, v) = 2h(hu, hv) / (h² + max(h, h_min)²)
    pub fn velocity(&self, state: &SWEState2D) -> (f64, f64) {
        state.velocity(self.h_min)
    }

    /// Compute velocity without desingularization (for wet cells).
    pub fn velocity_simple(&self, state: &SWEState2D) -> (f64, f64) {
        state.velocity_simple(self.h_min)
    }

    /// Compute wave celerity c = sqrt(gh).
    #[inline]
    pub fn celerity(&self, h: f64) -> f64 {
        (self.g * h.max(0.0)).sqrt()
    }

    /// Froude number Fr = |u| / c where |u| = sqrt(u² + v²).
    pub fn froude(&self, state: &SWEState2D) -> f64 {
        let (u, v) = self.velocity_simple(state);
        let speed = (u * u + v * v).sqrt();
        let c = self.celerity(state.h);
        if c > 1e-10 { speed / c } else { 0.0 }
    }

    /// Check if flow is subcritical (Fr < 1).
    pub fn is_subcritical(&self, state: &SWEState2D) -> bool {
        self.froude(state) < 1.0
    }

    /// Check if flow is supercritical (Fr > 1).
    pub fn is_supercritical(&self, state: &SWEState2D) -> bool {
        self.froude(state) > 1.0
    }

    /// Compute the x-direction flux F(q).
    ///
    /// F(q) = [hu, hu² + gh²/2, huv]ᵀ
    pub fn flux_x(&self, state: &SWEState2D) -> SWEState2D {
        let h = state.h;
        let hu = state.hu;
        let hv = state.hv;

        if h <= self.h_min.meters() {
            return SWEState2D::zero();
        }

        let u = hu / h;

        SWEState2D {
            h: hu,
            hu: h * u * u + 0.5 * self.g * h * h,
            hv: hu * hv / h,
        }
    }

    /// Compute the y-direction flux G(q).
    ///
    /// G(q) = [hv, huv, hv² + gh²/2]ᵀ
    pub fn flux_y(&self, state: &SWEState2D) -> SWEState2D {
        let h = state.h;
        let hu = state.hu;
        let hv = state.hv;

        if h <= self.h_min.meters() {
            return SWEState2D::zero();
        }

        let v = hv / h;

        SWEState2D {
            h: hv,
            hu: hu * hv / h,
            hv: h * v * v + 0.5 * self.g * h * h,
        }
    }

    /// Compute the normal flux F·n where n = (nx, ny).
    ///
    /// F·n = nx * F(q) + ny * G(q)
    ///
    /// This is used in surface integrals for DG discretization.
    pub fn normal_flux(&self, state: &SWEState2D, normal: (f64, f64)) -> SWEState2D {
        let (nx, ny) = normal;
        let fx = self.flux_x(state);
        let fy = self.flux_y(state);

        SWEState2D {
            h: nx * fx.h + ny * fy.h,
            hu: nx * fx.hu + ny * fy.hu,
            hv: nx * fx.hv + ny * fy.hv,
        }
    }

    /// Compute the Coriolis source term.
    ///
    /// S_coriolis = [0, fhv, -fhu]ᵀ
    ///
    /// # Arguments
    /// * `state` - Current SWE state
    /// * `y` - y-coordinate for beta-plane (ignored for f-plane)
    pub fn coriolis_source(&self, state: &SWEState2D, y: f64) -> SWEState2D {
        let f = self.coriolis_at(y);

        SWEState2D {
            h: 0.0,
            hu: f * state.hv,
            hv: -f * state.hu,
        }
    }

    /// Maximum wave speed for CFL computation.
    ///
    /// λ_max = |u| + c where |u| = sqrt(u² + v²) and c = sqrt(gh)
    pub fn max_wave_speed(&self, state: &SWEState2D) -> f64 {
        if state.h <= self.h_min.meters() {
            return 0.0;
        }

        let (u, v) = self.velocity_simple(state);
        let speed = (u * u + v * v).sqrt();
        let c = self.celerity(state.h);

        speed + c
    }

    /// Maximum wave speed in the normal direction.
    ///
    /// λ_n = |u·n| + c where u·n = u*nx + v*ny
    pub fn max_wave_speed_normal(&self, state: &SWEState2D, normal: (f64, f64)) -> f64 {
        if state.h <= self.h_min.meters() {
            return 0.0;
        }

        let (u, v) = self.velocity_simple(state);
        let (nx, ny) = normal;
        let un = u * nx + v * ny;
        let c = self.celerity(state.h);

        un.abs() + c
    }

    /// Eigenvalues of the flux Jacobian in the normal direction.
    ///
    /// For normal n = (nx, ny):
    /// λ₁ = u·n - c
    /// λ₂ = u·n
    /// λ₃ = u·n + c
    ///
    /// where u·n = u*nx + v*ny and c = sqrt(gh).
    pub fn eigenvalues_normal(&self, state: &SWEState2D, normal: (f64, f64)) -> [f64; 3] {
        if state.h <= self.h_min.meters() {
            return [0.0, 0.0, 0.0];
        }

        let (u, v) = self.velocity_simple(state);
        let (nx, ny) = normal;
        let un = u * nx + v * ny;
        let c = self.celerity(state.h);

        [un - c, un, un + c]
    }

    /// Roe-averaged state for linearized Riemann solver.
    ///
    /// Uses standard Roe averaging for shallow water:
    /// - h_roe = (h_l + h_r) / 2
    /// - u_roe = (√h_l u_l + √h_r u_r) / (√h_l + √h_r)
    /// - v_roe = (√h_l v_l + √h_r v_r) / (√h_l + √h_r)
    pub fn roe_average(&self, q_l: &SWEState2D, q_r: &SWEState2D) -> SWEState2D {
        let sqrt_h_l = q_l.h.max(0.0).sqrt();
        let sqrt_h_r = q_r.h.max(0.0).sqrt();

        let h_roe = 0.5 * (q_l.h + q_r.h);

        let denom = sqrt_h_l + sqrt_h_r;
        if denom > 1e-10 {
            let (u_l, v_l) = q_l.velocity_simple(self.h_min);
            let (u_r, v_r) = q_r.velocity_simple(self.h_min);

            let u_roe = (sqrt_h_l * u_l + sqrt_h_r * u_r) / denom;
            let v_roe = (sqrt_h_l * v_l + sqrt_h_r * v_r) / denom;

            SWEState2D::from_primitives(h_roe, u_roe, v_roe)
        } else {
            SWEState2D::new(h_roe, 0.0, 0.0)
        }
    }

    /// Compute the pressure term gh²/2.
    #[inline]
    pub fn pressure(&self, h: f64) -> f64 {
        0.5 * self.g * h * h
    }

    /// Check if state is valid (positive depth, finite values).
    pub fn is_valid(&self, state: &SWEState2D) -> bool {
        state.h >= 0.0 && state.h.is_finite() && state.hu.is_finite() && state.hv.is_finite()
    }

    /// Enforce positivity of depth (for limiters).
    ///
    /// If h < 0, sets state to zero (dry cell).
    pub fn enforce_positivity(&self, state: &mut SWEState2D) {
        if state.h < 0.0 {
            state.h = 0.0;
            state.hu = 0.0;
            state.hv = 0.0;
        } else if state.h < self.h_min.meters() {
            // Nearly dry cell - zero out velocities to prevent spurious flow
            state.hu = 0.0;
            state.hv = 0.0;
        }
    }
}

impl Default for ShallowWater2D {
    fn default() -> Self {
        Self::standard()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-12;

    #[test]
    fn test_creation() {
        let swe = ShallowWater2D::standard();
        assert!((swe.g - 9.81).abs() < TOL);
        assert!((swe.h_min.meters() - 1e-6).abs() < TOL);
        assert!((swe.f0 - 0.0).abs() < TOL);
    }

    #[test]
    fn test_norwegian_coast() {
        let swe = ShallowWater2D::norwegian_coast();
        assert!((swe.g - 9.81).abs() < TOL);
        assert!((swe.f0 - 1.2e-4).abs() < TOL);
    }

    #[test]
    fn test_flux_x_still_water() {
        let swe = ShallowWater2D::new(10.0);

        // Still water: h = 2, u = v = 0
        let state = SWEState2D::new(2.0, 0.0, 0.0);
        let flux = swe.flux_x(&state);

        // F = [hu, hu²/h + gh²/2, huv/h]
        //   = [0, 0 + 10 * 4 / 2, 0]
        //   = [0, 20, 0]
        assert!(flux.h.abs() < TOL);
        assert!((flux.hu - 20.0).abs() < TOL);
        assert!(flux.hv.abs() < TOL);
    }

    #[test]
    fn test_flux_y_still_water() {
        let swe = ShallowWater2D::new(10.0);

        let state = SWEState2D::new(2.0, 0.0, 0.0);
        let flux = swe.flux_y(&state);

        // G = [hv, huv/h, hv²/h + gh²/2]
        //   = [0, 0, 20]
        assert!(flux.h.abs() < TOL);
        assert!(flux.hu.abs() < TOL);
        assert!((flux.hv - 20.0).abs() < TOL);
    }

    #[test]
    fn test_flux_x_moving_water() {
        let swe = ShallowWater2D::new(10.0);

        // h = 2, u = 3, v = 1, hu = 6, hv = 2
        let state = SWEState2D::from_primitives(2.0, 3.0, 1.0);
        let flux = swe.flux_x(&state);

        // F = [hu, hu²/h + gh²/2, huv/h]
        //   = [6, 36/2 + 10*4/2, 6*2/2]
        //   = [6, 18 + 20, 6]
        //   = [6, 38, 6]
        assert!((flux.h - 6.0).abs() < TOL);
        assert!((flux.hu - 38.0).abs() < TOL);
        assert!((flux.hv - 6.0).abs() < TOL);
    }

    #[test]
    fn test_flux_y_moving_water() {
        let swe = ShallowWater2D::new(10.0);

        // h = 2, u = 3, v = 1, hu = 6, hv = 2
        let state = SWEState2D::from_primitives(2.0, 3.0, 1.0);
        let flux = swe.flux_y(&state);

        // G = [hv, huv/h, hv²/h + gh²/2]
        //   = [2, 6*2/2, 4/2 + 20]
        //   = [2, 6, 2 + 20]
        //   = [2, 6, 22]
        assert!((flux.h - 2.0).abs() < TOL);
        assert!((flux.hu - 6.0).abs() < TOL);
        assert!((flux.hv - 22.0).abs() < TOL);
    }

    #[test]
    fn test_flux_dry_cell() {
        let swe = ShallowWater2D::new(10.0);

        let state = SWEState2D::new(1e-10, 1e-10, 1e-10);
        let fx = swe.flux_x(&state);
        let fy = swe.flux_y(&state);

        assert!(fx.h.abs() < TOL);
        assert!(fx.hu.abs() < TOL);
        assert!(fx.hv.abs() < TOL);
        assert!(fy.h.abs() < TOL);
        assert!(fy.hu.abs() < TOL);
        assert!(fy.hv.abs() < TOL);
    }

    #[test]
    fn test_normal_flux() {
        let swe = ShallowWater2D::new(10.0);
        let state = SWEState2D::new(2.0, 6.0, 2.0); // h=2, u=3, v=1

        // Unit normal in x-direction
        let flux_x = swe.normal_flux(&state, (1.0, 0.0));
        let fx = swe.flux_x(&state);
        assert!((flux_x.h - fx.h).abs() < TOL);
        assert!((flux_x.hu - fx.hu).abs() < TOL);
        assert!((flux_x.hv - fx.hv).abs() < TOL);

        // Unit normal in y-direction
        let flux_y = swe.normal_flux(&state, (0.0, 1.0));
        let fy = swe.flux_y(&state);
        assert!((flux_y.h - fy.h).abs() < TOL);
        assert!((flux_y.hu - fy.hu).abs() < TOL);
        assert!((flux_y.hv - fy.hv).abs() < TOL);
    }

    #[test]
    fn test_eigenvalues() {
        let swe = ShallowWater2D::new(10.0);

        // h = 1, u = 2, v = 0
        let state = SWEState2D::from_primitives(1.0, 2.0, 0.0);
        let c = (10.0_f64).sqrt();

        // Normal in x-direction: u·n = 2
        let eigs_x = swe.eigenvalues_normal(&state, (1.0, 0.0));
        assert!((eigs_x[0] - (2.0 - c)).abs() < TOL);
        assert!((eigs_x[1] - 2.0).abs() < TOL);
        assert!((eigs_x[2] - (2.0 + c)).abs() < TOL);

        // Normal in y-direction: u·n = 0
        let eigs_y = swe.eigenvalues_normal(&state, (0.0, 1.0));
        assert!((eigs_y[0] - (-c)).abs() < TOL);
        assert!((eigs_y[1] - 0.0).abs() < TOL);
        assert!((eigs_y[2] - c).abs() < TOL);
    }

    #[test]
    fn test_max_wave_speed() {
        let swe = ShallowWater2D::new(10.0);

        // h = 1, u = 3, v = 4 -> |u| = 5
        let state = SWEState2D::from_primitives(1.0, 3.0, 4.0);
        let speed = swe.max_wave_speed(&state);

        let expected = 5.0 + (10.0_f64).sqrt();
        assert!((speed - expected).abs() < TOL);
    }

    #[test]
    fn test_coriolis_source() {
        let swe = ShallowWater2D::with_coriolis(9.81, 1.0e-4);

        // h = 10, hu = 100, hv = 50
        let state = SWEState2D::new(10.0, 100.0, 50.0);
        let source = swe.coriolis_source(&state, 0.0);

        // S = [0, fhv, -fhu] = [0, 1e-4 * 50, -1e-4 * 100]
        assert!(source.h.abs() < TOL);
        assert!((source.hu - 0.005).abs() < TOL);
        assert!((source.hv - (-0.01)).abs() < TOL);
    }

    #[test]
    fn test_beta_plane_coriolis() {
        let swe = ShallowWater2D::with_beta_plane(9.81, 1.0e-4, 1.0e-11, 0.0);

        // At y = 0: f = f0
        assert!((swe.coriolis_at(0.0) - 1.0e-4).abs() < TOL);

        // At y = 1e6 m (1000 km): f = f0 + beta * y = 1e-4 + 1e-11 * 1e6 = 1.1e-4
        assert!((swe.coriolis_at(1.0e6) - 1.1e-4).abs() < 1e-14);
    }

    #[test]
    fn test_roe_average_symmetric() {
        let swe = ShallowWater2D::new(10.0);

        let state = SWEState2D::new(2.0, 6.0, 4.0);
        let q_roe = swe.roe_average(&state, &state);

        assert!((q_roe.h - state.h).abs() < TOL);
        assert!((q_roe.hu - state.hu).abs() < TOL);
        assert!((q_roe.hv - state.hv).abs() < TOL);
    }

    #[test]
    fn test_roe_average_different() {
        let swe = ShallowWater2D::new(10.0);

        // h_l=1, u_l=1, v_l=0
        // h_r=4, u_r=2, v_r=1
        let q_l = SWEState2D::from_primitives(1.0, 1.0, 0.0);
        let q_r = SWEState2D::from_primitives(4.0, 2.0, 1.0);

        let q_roe = swe.roe_average(&q_l, &q_r);

        // h_roe = (1 + 4) / 2 = 2.5
        assert!((q_roe.h - 2.5).abs() < TOL);

        // u_roe = (1*1 + 2*2) / (1 + 2) = 5/3
        let u_roe = 5.0 / 3.0;
        // v_roe = (1*0 + 2*1) / (1 + 2) = 2/3
        let v_roe = 2.0 / 3.0;

        let hu_roe = 2.5 * u_roe;
        let hv_roe = 2.5 * v_roe;

        assert!((q_roe.hu - hu_roe).abs() < TOL);
        assert!((q_roe.hv - hv_roe).abs() < TOL);
    }

    #[test]
    fn test_froude_number() {
        let swe = ShallowWater2D::new(10.0);

        // Subcritical: Fr < 1
        // h = 10, |u| = 1, c = sqrt(100) = 10, Fr = 0.1
        let subcritical = SWEState2D::from_primitives(10.0, 1.0, 0.0);
        assert!(swe.is_subcritical(&subcritical));
        assert!(!swe.is_supercritical(&subcritical));

        // Supercritical: Fr > 1
        // h = 0.1, |u| = 10, c = sqrt(1) = 1, Fr = 10
        let supercritical = SWEState2D::from_primitives(0.1, 10.0, 0.0);
        assert!(swe.is_supercritical(&supercritical));
        assert!(!swe.is_subcritical(&supercritical));
    }

    #[test]
    fn test_enforce_positivity() {
        let swe = ShallowWater2D::new(10.0);

        // Negative depth
        let mut state = SWEState2D::new(-1.0, 5.0, 3.0);
        swe.enforce_positivity(&mut state);
        assert!((state.h - 0.0).abs() < TOL);
        assert!((state.hu - 0.0).abs() < TOL);
        assert!((state.hv - 0.0).abs() < TOL);

        // Very small depth
        let mut nearly_dry = SWEState2D::new(1e-10, 1e-5, 1e-5);
        swe.enforce_positivity(&mut nearly_dry);
        assert!((nearly_dry.h - 1e-10).abs() < TOL);
        assert!((nearly_dry.hu - 0.0).abs() < TOL);
        assert!((nearly_dry.hv - 0.0).abs() < TOL);
    }

    #[test]
    fn test_flux_conservation() {
        // Test that flux is consistent: if we rotate velocity by 90°,
        // the flux should rotate accordingly
        let swe = ShallowWater2D::new(10.0);

        // State with u=3, v=0
        let state_x = SWEState2D::from_primitives(2.0, 3.0, 0.0);
        // State with u=0, v=3 (rotated 90°)
        let state_y = SWEState2D::from_primitives(2.0, 0.0, 3.0);

        let fx = swe.flux_x(&state_x);
        let gy = swe.flux_y(&state_y);

        // The mass flux should be the same
        assert!((fx.h - gy.h).abs() < TOL);
        // The momentum flux in the direction of motion should be the same
        assert!((fx.hu - gy.hv).abs() < TOL);
        // The transverse momentum flux should be zero
        assert!(fx.hv.abs() < TOL);
        assert!(gy.hu.abs() < TOL);
    }
}

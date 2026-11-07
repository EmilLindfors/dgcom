//! 2D Hydrostatic reconstruction for well-balanced shallow water schemes.
//!
//! Implements the method of Audusse et al. (2004) extended to 2D:
//! At each face, reconstruct water depths relative to the maximum
//! bed level to ensure lake-at-rest preservation.
//!
//! Reference: Audusse, Bouchut, Bristeau, Klein, Perthame (2004)
//! "A fast and stable well-balanced scheme with hydrostatic reconstruction
//! for shallow water flows", SIAM J. Sci. Comput.

use crate::solver::SWEState2D;

/// Hydrostatic reconstruction for 2D well-balanced SWE schemes.
///
/// Modifies interface states at element faces to ensure:
/// 1. Lake-at-rest (η = const, u = v = 0) gives zero RHS to machine precision
/// 2. Wetting/drying is handled correctly (h* >= 0)
/// 3. Velocity is preserved during reconstruction
///
/// # Mathematical Formulation
///
/// At each interface, we compute:
/// - Water surface elevations: η_L = h_L + B_L, η_R = h_R + B_R
/// - Interface bathymetry: B* = max(B_L, B_R)
/// - Reconstructed depths: h_L* = max(0, η_L - B*), h_R* = max(0, η_R - B*)
///
/// For lake-at-rest (η_L = η_R = const):
/// - Both sides have same reconstructed depth: h_L* = h_R* = η - B*
/// - With zero velocity, the numerical flux is exactly zero
///
/// # Example
/// ```ignore
/// use dgcom::{HydrostaticReconstruction2D, SWEState2D};
///
/// let hr = HydrostaticReconstruction2D::standard();
///
/// // Lake at rest: η = 10.0
/// let q_l = SWEState2D::new(8.0, 0.0, 0.0);  // h_L = 8, B_L = 2
/// let q_r = SWEState2D::new(5.0, 0.0, 0.0);  // h_R = 5, B_R = 5
///
/// let (q_l_star, q_r_star) = hr.reconstruct(&q_l, &q_r, 2.0, 5.0);
///
/// // Both reconstructed states have h* = 10 - 5 = 5
/// assert!((q_l_star.h - q_r_star.h).abs() < 1e-14);
/// ```
#[derive(Clone, Debug)]
pub struct HydrostaticReconstruction2D {
    /// Gravitational acceleration
    pub g: f64,
    /// Minimum depth threshold for velocity desingularization
    pub h_min: f64,
}

impl HydrostaticReconstruction2D {
    /// Create a new hydrostatic reconstruction.
    pub fn new(g: f64, h_min: f64) -> Self {
        Self { g, h_min }
    }

    /// Standard gravity (9.81 m/s²) with default h_min (1e-6 m).
    pub fn standard() -> Self {
        Self::new(9.81, 1e-6)
    }

    /// Create with custom h_min.
    pub fn with_h_min(h_min: f64) -> Self {
        Self::new(9.81, h_min)
    }

    /// Reconstruct interface states for well-balanced flux computation.
    ///
    /// # Arguments
    /// * `q_l` - Left (interior) state
    /// * `q_r` - Right (exterior) state
    /// * `b_l` - Left bathymetry
    /// * `b_r` - Right bathymetry
    ///
    /// # Returns
    /// Tuple of (reconstructed left state, reconstructed right state)
    #[inline]
    pub fn reconstruct(
        &self,
        q_l: &SWEState2D,
        q_r: &SWEState2D,
        b_l: f64,
        b_r: f64,
    ) -> (SWEState2D, SWEState2D) {
        // Water surface elevations
        let eta_l = q_l.h + b_l;
        let eta_r = q_r.h + b_r;

        // Interface bathymetry (maximum of both sides)
        let b_star = b_l.max(b_r);

        // Reconstructed water depths
        let h_l_star = (eta_l - b_star).max(0.0);
        let h_r_star = (eta_r - b_star).max(0.0);

        // Preserve velocity using branchless-friendly code
        // For wet cells: u* = hu/h, so hu* = h* * hu / h
        // For dry cells: hu* = 0
        let (hu_l_star, hv_l_star) = if q_l.h > self.h_min {
            // Use ratio h*/h to preserve velocity
            let ratio = h_l_star / q_l.h;
            (q_l.hu * ratio, q_l.hv * ratio)
        } else {
            (0.0, 0.0)
        };

        let (hu_r_star, hv_r_star) = if q_r.h > self.h_min {
            let ratio = h_r_star / q_r.h;
            (q_r.hu * ratio, q_r.hv * ratio)
        } else {
            (0.0, 0.0)
        };

        (
            SWEState2D::new(h_l_star, hu_l_star, hv_l_star),
            SWEState2D::new(h_r_star, hu_r_star, hv_r_star),
        )
    }

    /// Hydrostatic interface correction for one side of a face.
    ///
    /// Audusse et al. (2004) give the side with nodal depth `h` and reconstructed
    /// depth `h*` the flux `F*(q*⁻, q*⁺)·n + ½g(h² − h*²)·n` (momentum only).
    /// In the DG strong form the surface term is `F(q⁻)·n − (that flux)`, where
    /// `F(q⁻)` must be the physical flux of the actual nodal state so that the
    /// surface term telescopes with the volume term (SBP property, discrete mass
    /// conservation). This returns the resulting addition to `F(q⁻)·n − F*·n`:
    ///
    /// ```text
    /// ½g(h*² − h²)·n   in (hu, hv),   0 in h
    /// ```
    ///
    /// The pressure `½gh²` is taken as zero for `h ≤ h_min`, matching the dry-node
    /// cutoff of `ShallowWater2D::flux_x`/`flux_y`, so that at lake-at-rest the
    /// correction cancels `F(q⁻)·n − F*·n` exactly (also at dry nodes).
    #[inline]
    pub fn pressure_correction(&self, h: f64, h_star: f64, normal: (f64, f64)) -> SWEState2D {
        let pressure = |depth: f64| {
            if depth > self.h_min {
                0.5 * self.g * depth * depth
            } else {
                0.0
            }
        };
        let dp = pressure(h_star) - pressure(h);
        SWEState2D::new(0.0, dp * normal.0, dp * normal.1)
    }

    /// Fast reconstruction when both sides are known to be wet.
    ///
    /// Skips dry-cell checks for better performance in deep water.
    #[inline(always)]
    pub fn reconstruct_wet(
        &self,
        q_l: &SWEState2D,
        q_r: &SWEState2D,
        b_l: f64,
        b_r: f64,
    ) -> (SWEState2D, SWEState2D) {
        let eta_l = q_l.h + b_l;
        let eta_r = q_r.h + b_r;
        let b_star = b_l.max(b_r);

        let h_l_star = (eta_l - b_star).max(0.0);
        let h_r_star = (eta_r - b_star).max(0.0);

        // Both wet: use ratio directly
        let ratio_l = h_l_star / q_l.h;
        let ratio_r = h_r_star / q_r.h;

        (
            SWEState2D::new(h_l_star, q_l.hu * ratio_l, q_l.hv * ratio_l),
            SWEState2D::new(h_r_star, q_r.hu * ratio_r, q_r.hv * ratio_r),
        )
    }

    /// Check if state satisfies lake-at-rest condition.
    ///
    /// Returns true if the water surface elevation is within `elevation_tol`
    /// of the reference and velocity magnitude is below `velocity_tol`.
    pub fn is_lake_at_rest(
        &self,
        q: &SWEState2D,
        b: f64,
        reference_eta: f64,
        velocity_tol: f64,
        elevation_tol: f64,
    ) -> bool {
        let (u, v) = if q.h > self.h_min {
            (q.hu / q.h, q.hv / q.h)
        } else {
            (0.0, 0.0)
        };
        let speed = (u * u + v * v).sqrt();
        let eta = q.h + b;

        speed < velocity_tol && (eta - reference_eta).abs() < elevation_tol
    }
}

impl Default for HydrostaticReconstruction2D {
    fn default() -> Self {
        Self::standard()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-14;

    #[test]
    fn test_lake_at_rest_equal_depths() {
        let hr = HydrostaticReconstruction2D::standard();

        // Lake at rest: η = 10.0, same bathymetry on both sides
        let eta = 10.0;
        let b = 2.0;
        let h = eta - b;

        let q_l = SWEState2D::new(h, 0.0, 0.0);
        let q_r = SWEState2D::new(h, 0.0, 0.0);

        let (q_l_star, q_r_star) = hr.reconstruct(&q_l, &q_r, b, b);

        // Reconstructed depths should equal original
        assert!((q_l_star.h - h).abs() < TOL);
        assert!((q_r_star.h - h).abs() < TOL);
        assert!(q_l_star.hu.abs() < TOL);
        assert!(q_r_star.hu.abs() < TOL);
    }

    #[test]
    fn test_lake_at_rest_different_bathymetry() {
        let hr = HydrostaticReconstruction2D::standard();

        // Lake at rest: η = 10.0, different bathymetries
        let eta = 10.0;
        let b_l = 2.0;
        let b_r = 5.0; // Higher bed on right
        let h_l = eta - b_l; // 8.0
        let h_r = eta - b_r; // 5.0

        let q_l = SWEState2D::new(h_l, 0.0, 0.0);
        let q_r = SWEState2D::new(h_r, 0.0, 0.0);

        let (q_l_star, q_r_star) = hr.reconstruct(&q_l, &q_r, b_l, b_r);

        // B* = max(2, 5) = 5
        // h_L* = max(0, 10 - 5) = 5
        // h_R* = max(0, 10 - 5) = 5
        // Both reconstructed states should have same h!
        let h_expected = eta - b_l.max(b_r);

        assert!(
            (q_l_star.h - h_expected).abs() < TOL,
            "h_L* = {}, expected {}",
            q_l_star.h,
            h_expected
        );
        assert!(
            (q_r_star.h - h_expected).abs() < TOL,
            "h_R* = {}, expected {}",
            q_r_star.h,
            h_expected
        );
        assert!(q_l_star.hu.abs() < TOL);
        assert!(q_l_star.hv.abs() < TOL);
        assert!(q_r_star.hu.abs() < TOL);
        assert!(q_r_star.hv.abs() < TOL);
    }

    #[test]
    fn test_preserves_velocity() {
        let hr = HydrostaticReconstruction2D::standard();

        let h_l = 5.0;
        let u_l = 2.0;
        let v_l = 1.5;
        let b_l = 1.0;

        let h_r = 3.0;
        let u_r = 1.0;
        let v_r = 0.5;
        let b_r = 3.0;

        let q_l = SWEState2D::from_primitives(h_l, u_l, v_l);
        let q_r = SWEState2D::from_primitives(h_r, u_r, v_r);

        let (q_l_star, q_r_star) = hr.reconstruct(&q_l, &q_r, b_l, b_r);

        // Velocities should be preserved
        if q_l_star.h > hr.h_min {
            let u_l_star = q_l_star.hu / q_l_star.h;
            let v_l_star = q_l_star.hv / q_l_star.h;
            assert!((u_l_star - u_l).abs() < TOL, "Left u preserved");
            assert!((v_l_star - v_l).abs() < TOL, "Left v preserved");
        }
        if q_r_star.h > hr.h_min {
            let u_r_star = q_r_star.hu / q_r_star.h;
            let v_r_star = q_r_star.hv / q_r_star.h;
            assert!((u_r_star - u_r).abs() < TOL, "Right u preserved");
            assert!((v_r_star - v_r).abs() < TOL, "Right v preserved");
        }
    }

    #[test]
    fn test_dry_bed_handling() {
        let hr = HydrostaticReconstruction2D::standard();

        // Water on left, bed higher than surface on right
        let eta_l = 2.0;
        let b_l = 0.0;
        let h_l = eta_l - b_l; // 2.0
        let b_r = 5.0; // Bed above water surface

        let q_l = SWEState2D::from_primitives(h_l, 1.0, 0.5);
        let q_r = SWEState2D::new(0.0, 0.0, 0.0);

        let (q_l_star, q_r_star) = hr.reconstruct(&q_l, &q_r, b_l, b_r);

        // B* = 5, η_L = 2, so h_L* = max(0, 2-5) = 0
        // Both should be dry at interface
        assert!(q_l_star.h < 1e-10, "Left should be dry at interface");
        assert!(q_r_star.h < 1e-10, "Right should be dry");
    }

    #[test]
    fn test_steep_gradient() {
        let hr = HydrostaticReconstruction2D::standard();

        // Simulate steep Norwegian bathymetry: gradient ~0.66
        // Left: shallow, Right: deep (going offshore)
        let eta = 50.0; // 50m surface elevation
        let b_l = 40.0; // 40m depth at left (10m water)
        let b_r = 10.0; // 10m depth at right (40m water)

        let h_l = eta - b_l; // 10m water
        let h_r = eta - b_r; // 40m water

        let q_l = SWEState2D::new(h_l, 0.0, 0.0);
        let q_r = SWEState2D::new(h_r, 0.0, 0.0);

        let (q_l_star, q_r_star) = hr.reconstruct(&q_l, &q_r, b_l, b_r);

        // B* = max(40, 10) = 40
        // h_L* = max(0, 50 - 40) = 10
        // h_R* = max(0, 50 - 40) = 10
        let h_expected = eta - b_l.max(b_r);

        assert!(
            (q_l_star.h - h_expected).abs() < TOL,
            "Steep gradient: h_L* = {}, expected {}",
            q_l_star.h,
            h_expected
        );
        assert!(
            (q_r_star.h - h_expected).abs() < TOL,
            "Steep gradient: h_R* = {}, expected {}",
            q_r_star.h,
            h_expected
        );
    }

    #[test]
    fn test_pressure_correction_balances_lake_at_rest() {
        use crate::equations::ShallowWater2D;
        use crate::flux::{SWEFluxType2D, compute_flux_swe_2d};

        let equation = ShallowWater2D::new(9.81);
        let hr = HydrostaticReconstruction2D::new(equation.g, equation.h_min.meters());

        // Lake at rest across a bathymetry jump, arbitrary face normal
        let eta = 10.0;
        let (b_int, b_ext) = (2.0, 5.0);
        let q_int = SWEState2D::new(eta - b_int, 0.0, 0.0);
        let q_ext = SWEState2D::new(eta - b_ext, 0.0, 0.0);
        let normal = (0.6, 0.8);

        let (q_int_star, q_ext_star) = hr.reconstruct(&q_int, &q_ext, b_int, b_ext);
        for flux_type in [
            SWEFluxType2D::Roe,
            SWEFluxType2D::HLL,
            SWEFluxType2D::Rusanov,
        ] {
            let f_star =
                compute_flux_swe_2d(&q_int_star, &q_ext_star, normal, hr.g, hr.h_min, flux_type);
            let flux_diff = equation.normal_flux(&q_int, normal) - f_star
                + hr.pressure_correction(q_int.h, q_int_star.h, normal);
            assert!(flux_diff.h.abs() < 1e-12, "{flux_type:?}: {flux_diff:?}");
            assert!(flux_diff.hu.abs() < 1e-12, "{flux_type:?}: {flux_diff:?}");
            assert!(flux_diff.hv.abs() < 1e-12, "{flux_type:?}: {flux_diff:?}");
        }

        // Continuous bathymetry: reconstruction is the identity, no correction
        let c = hr.pressure_correction(8.0, 8.0, normal);
        assert_eq!((c.h, c.hu, c.hv), (0.0, 0.0, 0.0));

        // Dry on both sides of the reconstruction: no pressure either way
        let c = hr.pressure_correction(0.5 * hr.h_min, 0.0, normal);
        assert_eq!((c.h, c.hu, c.hv), (0.0, 0.0, 0.0));
    }

    #[test]
    fn test_is_lake_at_rest() {
        let hr = HydrostaticReconstruction2D::standard();

        let eta_ref = 10.0;
        let b = 3.0;
        let h = eta_ref - b;

        // Lake at rest
        let q_still = SWEState2D::new(h, 0.0, 0.0);
        assert!(hr.is_lake_at_rest(&q_still, b, eta_ref, 1e-10, 1e-10));

        // Moving water
        let q_moving = SWEState2D::from_primitives(h, 1.0, 0.5);
        assert!(!hr.is_lake_at_rest(&q_moving, b, eta_ref, 1e-10, 1e-10));

        // Wrong elevation
        let q_wrong_eta = SWEState2D::new(h + 0.1, 0.0, 0.0);
        assert!(!hr.is_lake_at_rest(&q_wrong_eta, b, eta_ref, 1e-10, 1e-10));
    }
}

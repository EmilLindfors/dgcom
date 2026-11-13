//! 2D bathymetry source term for shallow water equations.
//!
//! The bathymetry source term accounts for the pressure force due to
//! variable bottom topography:
//!
//!   S = (0, -gh ∂B/∂x, -gh ∂B/∂y)
//!
//! where B(x, y) is the bottom elevation and h is the water depth.
//!
//! This source term is critical for correct modeling of flow over
//! varying bathymetry, including fjord sills and narrow straits.
//!
//! # Well-Balanced Property
//!
//! For a lake-at-rest (h + B = const, u = v = 0), the bathymetry source term
//! must balance exactly with the pressure gradient in the flux term to give
//! zero RHS. This requires consistent treatment of bathymetry in both
//! the source term and flux computation.
//!
//! # Note on Implementation
//!
//! This simple implementation evaluates the source term directly from the
//! nodal values. For steep bathymetry, more sophisticated approaches like
//! hydrostatic reconstruction (Audusse et al. 2004) may be needed to
//! maintain well-balanced property to machine precision.

use crate::solver::SWEState2D;
use crate::source::{SourceContext2D, SourceTerm2D};

/// Bathymetry source term for 2D shallow water equations.
///
/// Implements S = (0, -gh ∂B/∂x, -gh ∂B/∂y).
///
/// This term represents the pressure force on the water column due to
/// a sloping bottom. It must be included whenever there is non-zero
/// bathymetry gradient.
///
/// # Example
///
/// ```ignore
/// use dg::source::{BathymetrySource2D, CombinedSource2D, CoriolisSource2D};
/// use dg::mesh::Bathymetry2D;
///
/// // Create bathymetry with a sill
/// let bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, _y| {
///     0.5 * (-((x - 5.0).powi(2)) / 2.0).exp()
/// });
///
/// // Create source term
/// let bathy_source = BathymetrySource2D::new(9.81);
/// let coriolis = CoriolisSource2D::f_plane_at_latitude(60.0);
///
/// // Combine with other source terms
/// let combined = CombinedSource2D::new(vec![&bathy_source, &coriolis]);
///
/// // Use in RHS computation
/// let config = SWE2DRhsConfig::new(&equation, &bc)
///     .with_bathymetry(&bathymetry)
///     .with_source_terms(&combined);
/// ```
pub struct BathymetrySource2D {
    /// Gravitational acceleration (m/s²)
    pub g: f64,
}

impl BathymetrySource2D {
    /// Create a new bathymetry source term.
    ///
    /// # Arguments
    /// * `g` - Gravitational acceleration (default: 9.81 m/s²)
    pub fn new(g: f64) -> Self {
        Self { g }
    }

    /// Create with standard gravity (9.81 m/s²).
    pub fn standard() -> Self {
        Self::new(9.81)
    }
}

impl SourceTerm2D for BathymetrySource2D {
    fn evaluate(&self, ctx: &SourceContext2D) -> SWEState2D {
        let (db_dx, db_dy) = ctx.bathymetry_gradient;
        let h = ctx.state.h;

        // S = (0, -gh ∂B/∂x, -gh ∂B/∂y)
        SWEState2D {
            h: 0.0,
            hu: -self.g * h * db_dx,
            hv: -self.g * h * db_dy,
        }
    }

    fn is_node_local(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "bathymetry_source_2d"
    }

    fn includes_bathymetry_slope(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_context(h: f64, db_dx: f64, db_dy: f64) -> SourceContext2D {
        SourceContext2D::new(
            0.0,
            (0.0, 0.0),
            SWEState2D::new(h, 0.0, 0.0),
            0.0,
            (db_dx, db_dy),
            9.81,
            1e-6,
        )
    }

    #[test]
    fn test_flat_bathymetry_zero_source() {
        let source = BathymetrySource2D::new(9.81);
        let ctx = make_context(10.0, 0.0, 0.0);
        let result = source.evaluate(&ctx);

        assert!(result.h.abs() < 1e-14);
        assert!(result.hu.abs() < 1e-14);
        assert!(result.hv.abs() < 1e-14);
    }

    #[test]
    fn test_x_slope_source() {
        let g = 9.81;
        let source = BathymetrySource2D::new(g);
        let h = 10.0;
        let db_dx = 0.1; // 10% slope in x
        let ctx = make_context(h, db_dx, 0.0);
        let result = source.evaluate(&ctx);

        assert!(result.h.abs() < 1e-14);
        let expected_hu = -g * h * db_dx;
        assert!(
            (result.hu - expected_hu).abs() < 1e-12,
            "hu: {} != {}",
            result.hu,
            expected_hu
        );
        assert!(result.hv.abs() < 1e-14);
    }

    #[test]
    fn test_y_slope_source() {
        let g = 9.81;
        let source = BathymetrySource2D::new(g);
        let h = 10.0;
        let db_dy = -0.2; // -20% slope in y (downhill)
        let ctx = make_context(h, 0.0, db_dy);
        let result = source.evaluate(&ctx);

        assert!(result.h.abs() < 1e-14);
        assert!(result.hu.abs() < 1e-14);
        let expected_hv = -g * h * db_dy;
        assert!(
            (result.hv - expected_hv).abs() < 1e-12,
            "hv: {} != {}",
            result.hv,
            expected_hv
        );
    }

    #[test]
    fn test_diagonal_slope_source() {
        let g = 9.81;
        let source = BathymetrySource2D::new(g);
        let h = 5.0;
        let db_dx = 0.1;
        let db_dy = 0.15;
        let ctx = make_context(h, db_dx, db_dy);
        let result = source.evaluate(&ctx);

        let expected_hu = -g * h * db_dx;
        let expected_hv = -g * h * db_dy;

        assert!(result.h.abs() < 1e-14);
        assert!(
            (result.hu - expected_hu).abs() < 1e-12,
            "hu: {} != {}",
            result.hu,
            expected_hu
        );
        assert!(
            (result.hv - expected_hv).abs() < 1e-12,
            "hv: {} != {}",
            result.hv,
            expected_hv
        );
    }

    #[test]
    fn test_dry_cell_small_source() {
        let g = 9.81;
        let source = BathymetrySource2D::new(g);
        let h = 1e-10; // Very shallow
        let db_dx = 0.5; // Steep slope
        let ctx = make_context(h, db_dx, 0.0);
        let result = source.evaluate(&ctx);

        // Source should be proportional to h, so very small
        assert!(result.hu.abs() < 1e-8);
    }

    #[test]
    fn test_source_opposes_upslope() {
        // Water on an upslope (positive db_dx) should feel force in -x direction
        let source = BathymetrySource2D::new(9.81);
        let ctx = make_context(10.0, 0.1, 0.0); // Upslope in +x
        let result = source.evaluate(&ctx);

        // hu source should be negative (force in -x direction)
        assert!(result.hu < 0.0, "Upslope should give negative hu source");
    }

    #[test]
    fn test_source_pushes_downslope() {
        // Water on a downslope (negative db_dx) should feel force in +x direction
        let source = BathymetrySource2D::new(9.81);
        let ctx = make_context(10.0, -0.1, 0.0); // Downslope in +x direction
        let result = source.evaluate(&ctx);

        // hu source should be positive (force in +x direction)
        assert!(result.hu > 0.0, "Downslope should give positive hu source");
    }

    #[test]
    fn test_standard_gravity() {
        let source = BathymetrySource2D::standard();
        assert!((source.g - 9.81).abs() < 1e-10);
    }

    #[test]
    fn test_name() {
        let source = BathymetrySource2D::new(9.81);
        assert_eq!(source.name(), "bathymetry_source_2d");
    }
}

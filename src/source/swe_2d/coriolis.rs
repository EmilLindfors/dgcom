//! Coriolis source term for 2D shallow water equations.
//!
//! The Coriolis effect arises from Earth's rotation and deflects moving
//! water to the right in the Northern Hemisphere. The source term is:
//!
//! S_coriolis = [0, f·hv, -f·hu]ᵀ
//!
//! where f is the Coriolis parameter:
//! - f-plane: f = f₀ (constant)
//! - β-plane: f(y) = f₀ + β(y - y₀)
//!
//! # Norwegian Coast Context
//!
//! At latitude φ, f = 2Ω sin φ and β = 2Ω cos φ / R (Earth's rotation rate Ω,
//! mean radius R). At 60°N (Norwegian coast):
//! - f ≈ 1.26×10⁻⁴ s⁻¹
//! - Coriolis effects are significant for mesoscale circulation
//! - β ≈ 1.14×10⁻¹¹ m⁻¹ s⁻¹

use crate::mesh::Mesh2D;
use crate::solver::SWEState2D;
use crate::source::{ElementSources, SourceContext2D, SourceTerm2D};

/// Earth's rotation rate, one turn per sidereal day (rad/s)
pub const EARTH_ROTATION_RATE: f64 = 7.292_115e-5;
/// Earth's mean radius (m)
pub const EARTH_MEAN_RADIUS: f64 = 6_371_000.0;

/// Coriolis parameter f = 2Ω sin φ at a latitude in degrees (s⁻¹).
pub fn coriolis_parameter(latitude_deg: f64) -> f64 {
    2.0 * EARTH_ROTATION_RATE * latitude_deg.to_radians().sin()
}

/// Its northward gradient β = 2Ω cos φ / R at a latitude in degrees
/// (m⁻¹ s⁻¹).
pub fn coriolis_beta(latitude_deg: f64) -> f64 {
    2.0 * EARTH_ROTATION_RATE * latitude_deg.to_radians().cos() / EARTH_MEAN_RADIUS
}

/// Coriolis source term implementing SourceTerm2D trait.
///
/// Supports both f-plane (constant f) and β-plane (f varies with y).
///
/// # Example
///
/// ```
/// use dgcom::source::{CoriolisSource2D, SourceTerm2D, SourceContext2D};
/// use dgcom::SWEState2D;
///
/// // An f-plane at 60°N
/// let coriolis = CoriolisSource2D::f_plane_at_latitude(60.0);
///
/// let ctx = SourceContext2D::new(
///     0.0,
///     (0.0, 0.0),
///     SWEState2D::new(10.0, 100.0, 50.0), // h=10, hu=100, hv=50
///     0.0,
///     (0.0, 0.0),
///     9.81,
///     1e-6,
/// );
///
/// let source = coriolis.evaluate(&ctx);
/// // source.hu = f * hv = 1.2e-4 * 50 = 0.006
/// // source.hv = -f * hu = -1.2e-4 * 100 = -0.012
/// ```
#[derive(Clone, Copy, Debug)]
pub struct CoriolisSource2D {
    /// Coriolis parameter f₀ (s⁻¹)
    pub f0: f64,
    /// Beta-plane parameter β = ∂f/∂y (m⁻¹ s⁻¹)
    pub beta: f64,
    /// Reference y-coordinate for beta-plane (m)
    pub y0: f64,
}

impl CoriolisSource2D {
    /// Create a new Coriolis source with full parameters.
    pub fn new(f0: f64, beta: f64, y0: f64) -> Self {
        Self { f0, beta, y0 }
    }

    /// Create f-plane Coriolis (constant f).
    ///
    /// # Arguments
    /// * `f` - Coriolis parameter (s⁻¹)
    pub fn f_plane(f: f64) -> Self {
        Self {
            f0: f,
            beta: 0.0,
            y0: 0.0,
        }
    }

    /// Create beta-plane Coriolis: f(y) = f₀ + β(y - y₀).
    ///
    /// # Arguments
    /// * `f0` - Coriolis parameter at reference latitude (s⁻¹)
    /// * `beta` - Rate of change of f with latitude (m⁻¹ s⁻¹)
    /// * `y0` - Reference y-coordinate (m)
    pub fn beta_plane(f0: f64, beta: f64, y0: f64) -> Self {
        Self { f0, beta, y0 }
    }

    /// f-plane at a latitude (degrees): f = 2Ω sin φ.
    pub fn f_plane_at_latitude(latitude_deg: f64) -> Self {
        Self::f_plane(coriolis_parameter(latitude_deg))
    }

    /// β-plane tangent at a latitude (degrees) at `y = y0`: f₀ = 2Ω sin φ,
    /// β = 2Ω cos φ / R.
    pub fn beta_plane_at_latitude(latitude_deg: f64, y0: f64) -> Self {
        Self::beta_plane(
            coriolis_parameter(latitude_deg),
            coriolis_beta(latitude_deg),
            y0,
        )
    }

    /// Compute Coriolis parameter at given y-coordinate.
    ///
    /// f(y) = f₀ + β(y - y₀)
    #[inline]
    pub fn f_at(&self, y: f64) -> f64 {
        self.f0 + self.beta * (y - self.y0)
    }

    /// Check if this is a pure f-plane (no beta variation).
    pub fn is_f_plane(&self) -> bool {
        self.beta.abs() < 1e-20
    }
}

impl SourceTerm2D for CoriolisSource2D {
    fn evaluate(&self, ctx: &SourceContext2D) -> SWEState2D {
        let (_x, y) = ctx.position;
        let f = self.f_at(y);

        // S_coriolis = [0, f·hv, -f·hu]
        SWEState2D {
            h: 0.0,
            hu: f * ctx.state.hv,
            hv: -f * ctx.state.hu,
        }
    }

    /// Plain loop over the element's momentum; the node position (a bilinear
    /// map) only on a β-plane. Same arithmetic as [`Self::evaluate`].
    fn add_element(
        &self,
        element: &ElementSources<'_>,
        _h: &mut [f64],
        hu: &mut [f64],
        hv: &mut [f64],
    ) {
        // The element's corners once, not once per node
        let corners = (self.beta != 0.0).then(|| element.mesh.element_vertices(element.element));
        for i in 0..element.n_nodes() {
            let f = match &corners {
                None => self.f0,
                Some(corners) => {
                    let (r, s) = (element.ops.nodes_r[i], element.ops.nodes_s[i]);
                    self.f_at(Mesh2D::bilinear(corners, r, s)[1])
                }
            };
            let state = element.state(i);
            hu[i] += f * state.hv;
            hv[i] += -f * state.hu;
        }
    }

    fn name(&self) -> &'static str {
        "coriolis_2d"
    }
}

impl Default for CoriolisSource2D {
    /// Default is no Coriolis (f = 0).
    fn default() -> Self {
        Self::f_plane(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-14;

    /// f and β from the latitude (a 60°N preset once had β = 1.6e-11, the
    /// 45°N value).
    #[test]
    fn the_coriolis_parameter_and_beta_follow_the_latitude() {
        assert!((coriolis_parameter(60.0) - 1.2630e-4).abs() < 1e-8);
        assert!((coriolis_beta(60.0) - 1.1445e-11).abs() < 1e-14);
        assert!((coriolis_beta(45.0) - 1.6187e-11).abs() < 1e-14);
        assert!(coriolis_parameter(0.0).abs() < TOL);
        assert!(coriolis_parameter(-30.0) < 0.0);

        let plane = CoriolisSource2D::f_plane_at_latitude(60.0);
        assert_eq!(plane.f0, coriolis_parameter(60.0));
        assert!(plane.is_f_plane());

        let c = CoriolisSource2D::beta_plane_at_latitude(60.0, 0.0);
        assert_eq!(c.f0, coriolis_parameter(60.0));
        assert_eq!(c.beta, coriolis_beta(60.0));
        // One degree of latitude north (111 km) changes f by ≈ β · 111 km
        let dy = EARTH_MEAN_RADIUS * 1f64.to_radians();
        let exact = coriolis_parameter(61.0) - coriolis_parameter(60.0);
        assert!((c.f_at(dy) - c.f0 - exact).abs() < 0.02 * exact);

        // Southern hemisphere (Chiloé, `cases/chiloe`): f < 0, β > 0, so f
        // grows towards 0 northwards
        let s = CoriolisSource2D::beta_plane_at_latitude(-42.5, 0.0);
        assert!((s.f0 + 9.853e-5).abs() < 1e-8, "{}", s.f0);
        assert!(s.beta > 0.0 && s.f_at(dy) > s.f0);
        let exact = coriolis_parameter(-41.5) - coriolis_parameter(-42.5);
        assert!((s.f_at(dy) - s.f0 - exact).abs() < 0.02 * exact);
    }

    fn make_context(h: f64, hu: f64, hv: f64, y: f64) -> SourceContext2D {
        SourceContext2D::new(
            0.0,
            (0.0, y),
            SWEState2D::new(h, hu, hv),
            0.0,
            (0.0, 0.0),
            9.81,
            1e-6,
        )
    }

    #[test]
    fn test_f_plane_source() {
        let coriolis = CoriolisSource2D::f_plane(1.0e-4);
        let ctx = make_context(10.0, 100.0, 50.0, 0.0);

        let source = coriolis.evaluate(&ctx);

        // S = [0, f·hv, -f·hu] = [0, 1e-4 * 50, -1e-4 * 100]
        assert!(source.h.abs() < TOL);
        assert!((source.hu - 0.005).abs() < TOL);
        assert!((source.hv - (-0.01)).abs() < TOL);
    }

    #[test]
    fn test_beta_plane_f_variation() {
        let coriolis = CoriolisSource2D::beta_plane(1.0e-4, 1.0e-11, 0.0);

        // At y = 0: f = f0
        assert!((coriolis.f_at(0.0) - 1.0e-4).abs() < TOL);

        // At y = 1e6 m (1000 km): f = f0 + beta * y = 1e-4 + 1e-11 * 1e6 = 1.1e-4
        assert!((coriolis.f_at(1.0e6) - 1.1e-4).abs() < TOL);
    }

    #[test]
    fn test_beta_plane_source() {
        let coriolis = CoriolisSource2D::beta_plane(1.0e-4, 1.0e-11, 0.0);

        // At y = 1e6 m, f = 1.1e-4
        let ctx = make_context(10.0, 100.0, 50.0, 1.0e6);
        let source = coriolis.evaluate(&ctx);

        // S = [0, f·hv, -f·hu] = [0, 1.1e-4 * 50, -1.1e-4 * 100]
        assert!(source.h.abs() < TOL);
        assert!((source.hu - 0.0055).abs() < TOL);
        assert!((source.hv - (-0.011)).abs() < TOL);
    }

    #[test]
    fn test_zero_coriolis() {
        let coriolis = CoriolisSource2D::default();
        let ctx = make_context(10.0, 100.0, 50.0, 0.0);

        let source = coriolis.evaluate(&ctx);

        assert!(source.h.abs() < TOL);
        assert!(source.hu.abs() < TOL);
        assert!(source.hv.abs() < TOL);
    }

    #[test]
    fn test_coriolis_sign_convention() {
        // Northern hemisphere: Coriolis deflects to the right
        // Eastward flow (positive hu) should get deflected southward (negative hv tendency)
        let coriolis = CoriolisSource2D::f_plane(1.0e-4);
        let ctx = make_context(10.0, 100.0, 0.0, 0.0); // Pure eastward flow

        let source = coriolis.evaluate(&ctx);

        // hu tendency from hv is zero (hv = 0)
        // hv tendency from -f*hu should be negative
        assert!(
            source.hv < 0.0,
            "Eastward flow should be deflected southward"
        );
    }

    #[test]
    fn test_geostrophic_balance_tendency() {
        // In geostrophic balance: Coriolis = pressure gradient
        // Test that northward flow (positive hv) creates positive hu tendency
        let coriolis = CoriolisSource2D::f_plane(1.0e-4);
        let ctx = make_context(10.0, 0.0, 100.0, 0.0); // Pure northward flow

        let source = coriolis.evaluate(&ctx);

        // hu tendency from f*hv should be positive
        assert!(
            source.hu > 0.0,
            "Northward flow should gain eastward momentum"
        );
    }

    /// The per-element path gives exactly what per-node `evaluate` gives, on
    /// an f-plane (no positions) and a β-plane, alone and inside a
    /// `SourceTerms2D` next to a source on the default path.
    #[test]
    fn add_element_matches_per_node_evaluate() {
        use std::sync::Arc;

        use crate::mesh::{Bathymetry2D, Mesh2D};
        use crate::operators::DGOperators2D;
        use crate::solver::SWESolution2D;
        use crate::source::{ManningFriction2D, SourceTerms2D};
        use crate::types::ElementIndex;

        let mesh = Mesh2D::uniform_rectangle(0.0, 2e4, -1e4, 1e4, 3, 2);
        let ops = DGOperators2D::new(3);
        let bathymetry = Bathymetry2D::flat(mesh.n_elements, ops.n_nodes);
        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let x = (k.as_usize() * ops.n_nodes + i) as f64;
                q.set_state(k, i, SWEState2D::new(10.0 + x.sin(), x.cos(), 0.3 * x));
            }
        }

        let friction: Arc<dyn SourceTerm2D> = Arc::new(ManningFriction2D::new(9.81, 0.03));
        for coriolis in [
            CoriolisSource2D::f_plane(1.3e-4),
            CoriolisSource2D::beta_plane(1.3e-4, 1.6e-11, 5e3),
        ] {
            let combined = SourceTerms2D::new(vec![Arc::new(coriolis), friction.clone()]);
            let sources: [(&dyn SourceTerm2D, bool); 2] = [(&coriolis, true), (&combined, false)];
            for (source, exact) in sources {
                for k in ElementIndex::iter(mesh.n_elements) {
                    let element = ElementSources {
                        element: k,
                        time: 0.0,
                        solution: &q,
                        mesh: &mesh,
                        ops: &ops,
                        bathymetry: Some(&bathymetry),
                        g: 9.81,
                        h_min: 1e-3,
                    };
                    let n = ops.n_nodes;
                    let (mut h, mut hu, mut hv) = (vec![1.0; n], vec![2.0; n], vec![3.0; n]);
                    source.add_element(&element, &mut h, &mut hu, &mut hv);
                    for i in 0..n {
                        let s = source.evaluate(&element.context(i));
                        let expected = [1.0 + s.h, 2.0 + s.hu, 3.0 + s.hv];
                        for (got, want) in [h[i], hu[i], hv[i]].into_iter().zip(expected) {
                            if exact {
                                assert_eq!(got, want);
                            } else {
                                // Summation order differs: sources are added one at a time
                                assert!((got - want).abs() <= 4.0 * f64::EPSILON * want.abs());
                            }
                        }
                    }
                }
            }
        }
    }
}

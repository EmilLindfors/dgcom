//! 2D Sponge layer (relaxation zone) for absorbing boundaries.
//!
//! Sponge layers provide gradual damping toward a reference state,
//! reducing reflections at open boundaries. The damping coefficient
//! increases smoothly from zero at the interior edge to maximum
//! at the boundary.
//!
//! The source term is:
//! S = γ(x, y) · (q_ref - q)
//!
//! where γ is the spatially-varying damping coefficient.
//!
//! # Sponge Profiles
//!
//! Several damping profiles are supported:
//! - Linear: γ = γ_max · ξ
//! - Quadratic: γ = γ_max · ξ²
//! - Cosine (smooth): γ = γ_max · (1 - cos(πξ))/2
//! - Exponential: γ = γ_max · (e^(αξ) - 1)/(e^α - 1)
//!
//! where ξ ∈ [0, 1] is the normalized distance (0 at interior, 1 at boundary).
//!
//! # Example
//!
//! ```
//! use dgcom::source::{SpongeLayer2D, SpongeProfile, SourceTerm2D};
//! use dgcom::SWEState2D;
//!
//! // Create sponge zones on left and right boundaries
//! let domain = (0.0, 1000.0, 0.0, 500.0);
//! let sponge_width = 100.0;
//!
//! let sponge = SpongeLayer2D::rectangular(
//!     |_x, _y, _t| SWEState2D::new(10.0, 0.0, 0.0), // Reference: still water
//!     0.1,  // gamma_max
//!     sponge_width,
//!     domain,
//!     [true, true, false, false], // Left and right only
//! );
//! ```

use crate::solver::SWEState2D;
use crate::source::{SourceContext2D, SourceTerm2D};
use std::f64::consts::PI;

/// Shape function for sponge layer damping profile.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum SpongeProfile {
    /// Linear ramp: γ(ξ) = γ_max · ξ
    Linear,
    /// Quadratic: γ(ξ) = γ_max · ξ²
    Quadratic,
    /// Cosine (smooth, default): γ(ξ) = γ_max · (1 - cos(πξ))/2
    #[default]
    Cosine,
    /// Exponential: γ(ξ) = γ_max · (e^(αξ) - 1)/(e^α - 1)
    Exponential {
        /// Steepness parameter (typically 2-5)
        alpha: f64,
    },
}

impl SpongeProfile {
    /// Evaluate profile at normalized distance ξ ∈ [0, 1].
    ///
    /// Returns γ/γ_max ∈ [0, 1].
    pub fn evaluate(&self, xi: f64) -> f64 {
        let xi = xi.clamp(0.0, 1.0);
        match self {
            SpongeProfile::Linear => xi,
            SpongeProfile::Quadratic => xi * xi,
            SpongeProfile::Cosine => 0.5 * (1.0 - (PI * xi).cos()),
            SpongeProfile::Exponential { alpha } => {
                if alpha.abs() < 1e-10 {
                    xi // Degenerate to linear
                } else {
                    ((alpha * xi).exp() - 1.0) / (alpha.exp() - 1.0)
                }
            }
        }
    }
}

/// Function type for computing normalized distance in sponge zone.
///
/// Returns Some(ξ) with ξ ∈ [0, 1] if point is in sponge zone,
/// where 0 = interior edge, 1 = boundary edge.
/// Returns None if point is outside the zone.
pub type SpongeDistanceFn = Box<dyn Fn(f64, f64) -> Option<f64> + Send + Sync>;

/// Which side of a rectangular domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RectangularBoundary {
    /// x = x_min boundary
    Left,
    /// x = x_max boundary
    Right,
    /// y = y_min boundary
    Bottom,
    /// y = y_max boundary
    Top,
}

/// Create a sponge distance function for a rectangular boundary.
///
/// # Arguments
/// * `boundary` - Which boundary (Left, Right, Bottom, Top)
/// * `domain` - Domain extents (x_min, x_max, y_min, y_max)
/// * `width` - Width of sponge zone
pub fn rectangular_sponge_fn(
    boundary: RectangularBoundary,
    domain: (f64, f64, f64, f64),
    width: f64,
) -> SpongeDistanceFn {
    let (x_min, x_max, y_min, y_max) = domain;

    Box::new(move |x, y| match boundary {
        RectangularBoundary::Left => {
            if x < x_min + width && y >= y_min && y <= y_max {
                Some(1.0 - (x - x_min) / width)
            } else {
                None
            }
        }
        RectangularBoundary::Right => {
            if x > x_max - width && y >= y_min && y <= y_max {
                Some((x - (x_max - width)) / width)
            } else {
                None
            }
        }
        RectangularBoundary::Bottom => {
            if y < y_min + width && x >= x_min && x <= x_max {
                Some(1.0 - (y - y_min) / width)
            } else {
                None
            }
        }
        RectangularBoundary::Top => {
            if y > y_max - width && x >= x_min && x <= x_max {
                Some((y - (y_max - width)) / width)
            } else {
                None
            }
        }
    })
}

/// 2D Sponge layer source term.
///
/// Implements relaxation toward a reference state with spatially-varying
/// damping coefficient. The source term is:
///
/// S = γ(x, y) · (q_ref - q)
///
/// Multiple sponge zones can be defined, with overlapping zones taking
/// the maximum damping coefficient.
pub struct SpongeLayer2D<F>
where
    F: Fn(f64, f64, f64) -> SWEState2D + Send + Sync,
{
    /// Reference state function (x, y, t) -> SWEState2D
    pub reference_state: F,
    /// Maximum damping coefficient (1/s)
    pub gamma_max: f64,
    /// Damping profile shape
    pub profile: SpongeProfile,
    /// Sponge zone distance functions
    pub zones: Vec<SpongeDistanceFn>,
    /// Whether to only damp momentum (preserve mass conservation)
    pub momentum_only: bool,
}

impl<F> SpongeLayer2D<F>
where
    F: Fn(f64, f64, f64) -> SWEState2D + Send + Sync,
{
    /// Create a new sponge layer with given zones.
    pub fn new(reference_state: F, gamma_max: f64, zones: Vec<SpongeDistanceFn>) -> Self {
        Self {
            reference_state,
            gamma_max,
            profile: SpongeProfile::Cosine,
            zones,
            momentum_only: false,
        }
    }

    /// Create a sponge layer for rectangular domain boundaries.
    ///
    /// # Arguments
    /// * `reference_state` - Reference state function
    /// * `gamma_max` - Maximum damping coefficient
    /// * `width` - Width of sponge zones
    /// * `domain` - Domain extents (x_min, x_max, y_min, y_max)
    /// * `boundaries` - Which boundaries to add sponge [left, right, bottom, top]
    pub fn rectangular(
        reference_state: F,
        gamma_max: f64,
        width: f64,
        domain: (f64, f64, f64, f64),
        boundaries: [bool; 4],
    ) -> Self {
        let mut zones = Vec::new();

        if boundaries[0] {
            zones.push(rectangular_sponge_fn(
                RectangularBoundary::Left,
                domain,
                width,
            ));
        }
        if boundaries[1] {
            zones.push(rectangular_sponge_fn(
                RectangularBoundary::Right,
                domain,
                width,
            ));
        }
        if boundaries[2] {
            zones.push(rectangular_sponge_fn(
                RectangularBoundary::Bottom,
                domain,
                width,
            ));
        }
        if boundaries[3] {
            zones.push(rectangular_sponge_fn(
                RectangularBoundary::Top,
                domain,
                width,
            ));
        }

        Self {
            reference_state,
            gamma_max,
            profile: SpongeProfile::Cosine,
            zones,
            momentum_only: false,
        }
    }

    /// Set the damping profile.
    pub fn with_profile(mut self, profile: SpongeProfile) -> Self {
        self.profile = profile;
        self
    }

    /// Only damp momentum (useful for strict mass conservation).
    ///
    /// When enabled, the depth (h) is not damped, only the momentum (hu, hv).
    pub fn with_momentum_only(mut self, momentum_only: bool) -> Self {
        self.momentum_only = momentum_only;
        self
    }

    /// Compute damping coefficient at a point.
    ///
    /// Takes the maximum over all overlapping zones.
    pub fn damping_at(&self, x: f64, y: f64) -> f64 {
        let mut gamma: f64 = 0.0;
        for zone_fn in &self.zones {
            if let Some(xi) = zone_fn(x, y) {
                let zone_gamma = self.gamma_max * self.profile.evaluate(xi);
                gamma = gamma.max(zone_gamma);
            }
        }
        gamma
    }

    /// Check if a point is in any sponge zone.
    pub fn is_in_sponge(&self, x: f64, y: f64) -> bool {
        for zone_fn in &self.zones {
            if zone_fn(x, y).is_some() {
                return true;
            }
        }
        false
    }

    /// Add a custom sponge zone.
    pub fn add_zone(&mut self, zone: SpongeDistanceFn) {
        self.zones.push(zone);
    }
}

impl<F> SourceTerm2D for SpongeLayer2D<F>
where
    F: Fn(f64, f64, f64) -> SWEState2D + Send + Sync,
{
    fn evaluate(&self, ctx: &SourceContext2D) -> SWEState2D {
        let (x, y) = ctx.position;
        let gamma = self.damping_at(x, y);

        if gamma < 1e-14 {
            return SWEState2D::zero();
        }

        let q_ref = (self.reference_state)(x, y, ctx.time);
        let q = ctx.state;

        // S = γ · (q_ref - q)
        if self.momentum_only {
            SWEState2D {
                h: 0.0,
                hu: gamma * (q_ref.hu - q.hu),
                hv: gamma * (q_ref.hv - q.hv),
            }
        } else {
            SWEState2D {
                h: gamma * (q_ref.h - q.h),
                hu: gamma * (q_ref.hu - q.hu),
                hv: gamma * (q_ref.hv - q.hv),
            }
        }
    }

    fn name(&self) -> &'static str {
        "sponge_2d"
    }

    fn changes_mass(&self) -> bool {
        !self.momentum_only
    }

    fn is_stiff(&self) -> bool {
        // Sponge can be stiff for large gamma_max
        self.gamma_max > 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-12;

    fn make_context(x: f64, y: f64, state: SWEState2D) -> SourceContext2D {
        SourceContext2D::new(0.0, (x, y), state, 0.0, (0.0, 0.0), 9.81, 1e-6)
    }

    #[test]
    fn test_sponge_profile_linear() {
        let profile = SpongeProfile::Linear;
        assert!((profile.evaluate(0.0) - 0.0).abs() < TOL);
        assert!((profile.evaluate(0.5) - 0.5).abs() < TOL);
        assert!((profile.evaluate(1.0) - 1.0).abs() < TOL);
    }

    #[test]
    fn test_sponge_profile_cosine() {
        let profile = SpongeProfile::Cosine;
        assert!((profile.evaluate(0.0) - 0.0).abs() < TOL);
        assert!((profile.evaluate(0.5) - 0.5).abs() < TOL);
        assert!((profile.evaluate(1.0) - 1.0).abs() < TOL);
    }

    #[test]
    fn test_sponge_profile_quadratic() {
        let profile = SpongeProfile::Quadratic;
        assert!((profile.evaluate(0.0) - 0.0).abs() < TOL);
        assert!((profile.evaluate(0.5) - 0.25).abs() < TOL);
        assert!((profile.evaluate(1.0) - 1.0).abs() < TOL);
    }

    #[test]
    fn test_rectangular_sponge_left() {
        let domain = (0.0, 100.0, 0.0, 50.0);
        let width = 10.0;
        let zone_fn = rectangular_sponge_fn(RectangularBoundary::Left, domain, width);

        // At boundary (x=0): xi = 1
        assert!((zone_fn(0.0, 25.0).unwrap() - 1.0).abs() < TOL);
        // Just inside interior edge (x=9.9): xi ≈ 0.01
        assert!((zone_fn(9.9, 25.0).unwrap() - 0.01).abs() < TOL);
        // Middle (x=5): xi = 0.5
        assert!((zone_fn(5.0, 25.0).unwrap() - 0.5).abs() < TOL);
        // At interior edge (x=10): outside zone
        assert!(zone_fn(10.0, 25.0).is_none());
        // Outside: None
        assert!(zone_fn(50.0, 25.0).is_none());
    }

    #[test]
    fn test_rectangular_sponge_right() {
        let domain = (0.0, 100.0, 0.0, 50.0);
        let width = 10.0;
        let zone_fn = rectangular_sponge_fn(RectangularBoundary::Right, domain, width);

        // At boundary (x=100): xi = 1
        assert!((zone_fn(100.0, 25.0).unwrap() - 1.0).abs() < TOL);
        // Just inside interior edge (x=90.1): xi ≈ 0.01
        assert!((zone_fn(90.1, 25.0).unwrap() - 0.01).abs() < TOL);
        // At interior edge (x=90): outside zone
        assert!(zone_fn(90.0, 25.0).is_none());
        // Outside: None
        assert!(zone_fn(50.0, 25.0).is_none());
    }

    #[test]
    fn test_sponge_source_outside_zone() {
        let sponge = SpongeLayer2D::rectangular(
            |_, _, _| SWEState2D::new(10.0, 0.0, 0.0),
            1.0,
            10.0,
            (0.0, 100.0, 0.0, 50.0),
            [true, false, false, false], // Left only
        );

        // Outside sponge zone (x=50)
        let ctx = make_context(50.0, 25.0, SWEState2D::new(5.0, 10.0, 5.0));
        let source = sponge.evaluate(&ctx);

        assert!(source.h.abs() < TOL);
        assert!(source.hu.abs() < TOL);
        assert!(source.hv.abs() < TOL);
    }

    #[test]
    fn test_sponge_source_at_boundary() {
        let gamma_max = 1.0;
        let sponge = SpongeLayer2D::rectangular(
            |_, _, _| SWEState2D::new(10.0, 0.0, 0.0),
            gamma_max,
            10.0,
            (0.0, 100.0, 0.0, 50.0),
            [true, false, false, false],
        );

        // At boundary (x=0), state differs from reference
        let state = SWEState2D::new(5.0, 10.0, 5.0);
        let ctx = make_context(0.0, 25.0, state);
        let source = sponge.evaluate(&ctx);

        // S = gamma_max * (q_ref - q)
        let expected_h = gamma_max * (10.0 - 5.0);
        let expected_hu = gamma_max * (0.0 - 10.0);
        let expected_hv = gamma_max * (0.0 - 5.0);

        assert!(
            (source.h - expected_h).abs() < TOL,
            "Expected h source = {}, got {}",
            expected_h,
            source.h
        );
        assert!(
            (source.hu - expected_hu).abs() < TOL,
            "Expected hu source = {}, got {}",
            expected_hu,
            source.hu
        );
        assert!(
            (source.hv - expected_hv).abs() < TOL,
            "Expected hv source = {}, got {}",
            expected_hv,
            source.hv
        );
    }

    #[test]
    fn test_sponge_momentum_only() {
        let sponge = SpongeLayer2D::rectangular(
            |_, _, _| SWEState2D::new(10.0, 0.0, 0.0),
            1.0,
            10.0,
            (0.0, 100.0, 0.0, 50.0),
            [true, false, false, false],
        )
        .with_momentum_only(true);

        let state = SWEState2D::new(5.0, 10.0, 5.0);
        let ctx = make_context(0.0, 25.0, state);
        let source = sponge.evaluate(&ctx);

        // h should not be damped
        assert!(source.h.abs() < TOL);
        // Momentum should be damped
        assert!(source.hu.abs() > 0.0);
        assert!(source.hv.abs() > 0.0);
    }

    #[test]
    fn test_sponge_equilibrium() {
        // When state matches reference, source should be zero
        let sponge = SpongeLayer2D::rectangular(
            |_, _, _| SWEState2D::new(10.0, 5.0, 3.0),
            1.0,
            10.0,
            (0.0, 100.0, 0.0, 50.0),
            [true, true, true, true],
        );

        let state = SWEState2D::new(10.0, 5.0, 3.0);
        let ctx = make_context(0.0, 0.0, state);
        let source = sponge.evaluate(&ctx);

        assert!(source.h.abs() < TOL);
        assert!(source.hu.abs() < TOL);
        assert!(source.hv.abs() < TOL);
    }

    #[test]
    fn test_overlapping_zones_max() {
        // Corner should take max of both zones
        let sponge = SpongeLayer2D::rectangular(
            |_, _, _| SWEState2D::new(10.0, 0.0, 0.0),
            1.0,
            10.0,
            (0.0, 100.0, 0.0, 50.0),
            [true, false, true, false], // Left and bottom
        );

        // At corner (0, 0), should be in both zones with xi=1
        let gamma = sponge.damping_at(0.0, 0.0);
        assert!((gamma - 1.0).abs() < TOL);
    }

    #[test]
    fn test_sponge_is_stiff() {
        let sponge_normal =
            SpongeLayer2D::new(|_, _, _| SWEState2D::new(10.0, 0.0, 0.0), 0.5, vec![]);
        assert!(!sponge_normal.is_stiff());

        let sponge_stiff =
            SpongeLayer2D::new(|_, _, _| SWEState2D::new(10.0, 0.0, 0.0), 2.0, vec![]);
        assert!(sponge_stiff.is_stiff());
    }
}

//! Enhanced friction source term for narrow straits.
//!
//! Narrow straits in fjord systems require enhanced friction to account for
//! sub-grid scale effects that cannot be resolved:
//! - Lateral friction from nearby walls
//! - Flow separation and recirculation
//! - Increased turbulence and mixing
//!
//! This module provides a friction enhancement model based on strait width:
//!
//! n_eff = n_base * (1 + α * (1 - W/W_crit))  for W < W_crit
//! n_eff = n_base                              for W >= W_crit
//!
//! where:
//! - n_base is the base Manning coefficient
//! - α is the enhancement factor
//! - W is the local strait width
//! - W_crit is the critical width below which enhancement applies
//!
//! # Norwegian Fjord Context
//!
//! Many Norwegian fjords connect through narrow straits (sunds) where
//! tidal currents are amplified. Examples:
//! - Drøbaksundet (Oslo fjord entrance): ~500m width
//! - Tjeldsundet: ~1km width
//! - Various smaller sounds: 100-500m
//!
//! These straits often have strong currents (> 1 m/s) that require
//! enhanced friction treatment for accurate flow simulation.

use crate::solver::SWEState2D;
use crate::source::swe_2d::ManningFriction2D;
use crate::source::{SourceContext2D, SourceTerm2D};

/// Enhanced friction for narrow straits.
///
/// This source term enhances Manning friction in regions where the strait
/// width falls below a critical threshold. The enhancement accounts for
/// sub-grid scale lateral friction and turbulence effects.
///
/// # Type Parameters
///
/// * `F` - Function `(x, y) -> Option<f64>` returning the local strait width
///   in meters, or `None` if the point is not in a strait region.
///
/// # Example
///
/// ```ignore
/// use dg::source::{StraitFriction2D, ManningFriction2D};
///
/// // Define strait region: a narrow channel between x=490 and x=510
/// // with width varying from 100m at the center to wider at edges
/// let strait_width = |x: f64, _y: f64| {
///     if x >= 490.0 && x <= 510.0 {
///         let w = 100.0 + 50.0 * (x - 500.0).abs() / 10.0;
///         Some(w)
///     } else {
///         None
///     }
/// };
///
/// let strait_friction = StraitFriction2D::new(
///     ManningFriction2D::fjord(),
///     strait_width,
///     500.0,  // critical width (m)
///     2.0,    // enhancement factor
/// );
/// ```
pub struct StraitFriction2D<F>
where
    F: Fn(f64, f64) -> Option<f64> + Send + Sync,
{
    /// Base Manning friction (used everywhere)
    pub base_friction: ManningFriction2D,
    /// Function returning local strait width at (x,y), or None if not in strait
    pub width_fn: F,
    /// Critical width below which enhancement applies (m)
    pub critical_width: f64,
    /// Enhancement factor α (dimensionless)
    pub enhancement_factor: f64,
}

impl<F> StraitFriction2D<F>
where
    F: Fn(f64, f64) -> Option<f64> + Send + Sync,
{
    /// Create a new strait friction source term.
    ///
    /// # Arguments
    /// * `base_friction` - Base Manning friction parameters
    /// * `width_fn` - Function returning strait width at (x,y), or None
    /// * `critical_width` - Width threshold for enhancement (m)
    /// * `enhancement_factor` - Maximum friction enhancement factor
    pub fn new(
        base_friction: ManningFriction2D,
        width_fn: F,
        critical_width: f64,
        enhancement_factor: f64,
    ) -> Self {
        Self {
            base_friction,
            width_fn,
            critical_width,
            enhancement_factor,
        }
    }

    /// Create for very narrow straits (sounds).
    ///
    /// Uses parameters suitable for very narrow passages:
    /// - critical_width: 200m
    /// - enhancement_factor: 3.0
    pub fn narrow_sound(base_friction: ManningFriction2D, width_fn: F) -> Self {
        Self::new(base_friction, width_fn, 200.0, 3.0)
    }

    /// Compute the effective Manning coefficient at a location.
    ///
    /// Returns n_eff = n_base * (1 + α * (1 - W/W_crit)) for narrow straits,
    /// or n_base elsewhere.
    pub fn effective_manning(&self, x: f64, y: f64) -> f64 {
        let n_base = self.base_friction.manning_n;

        match (self.width_fn)(x, y) {
            Some(width) if width < self.critical_width && width > 0.0 => {
                // Enhancement factor increases as width decreases
                let width_ratio = width / self.critical_width;
                n_base * (1.0 + self.enhancement_factor * (1.0 - width_ratio))
            }
            _ => n_base,
        }
    }

    /// Check if a location is in a strait region.
    pub fn is_in_strait(&self, x: f64, y: f64) -> bool {
        (self.width_fn)(x, y).is_some()
    }

    /// Get the local enhancement factor (1.0 = no enhancement).
    pub fn local_enhancement(&self, x: f64, y: f64) -> f64 {
        match (self.width_fn)(x, y) {
            Some(width) if width < self.critical_width && width > 0.0 => {
                1.0 + self.enhancement_factor * (1.0 - width / self.critical_width)
            }
            _ => 1.0,
        }
    }

    /// Compute friction source with enhanced coefficient.
    fn evaluate_with_effective_manning(&self, ctx: &SourceContext2D) -> SWEState2D {
        if ctx.state.h < self.base_friction.h_min {
            return SWEState2D::zero();
        }

        let (x, y) = ctx.position;
        let n_eff = self.effective_manning(x, y);

        // Compute velocity
        let u = ctx.state.hu / ctx.state.h;
        let v = ctx.state.hv / ctx.state.h;
        let speed = (u * u + v * v).sqrt();

        if speed < 1e-14 {
            return SWEState2D::zero();
        }

        // Compute friction coefficient with effective Manning number
        let h_eff = ctx.state.h.max(self.base_friction.h_min);
        let c_f = self.base_friction.g * n_eff * n_eff / h_eff.powf(1.0 / 3.0);

        // S = (0, -C_f |u| u, -C_f |u| v)
        SWEState2D {
            h: 0.0,
            hu: -c_f * speed * u,
            hv: -c_f * speed * v,
        }
    }

    /// Semi-implicit update with enhanced friction.
    ///
    /// Uses the effective Manning coefficient for the semi-implicit friction update.
    pub fn semi_implicit_update(&self, state: &SWEState2D, x: f64, y: f64, dt: f64) -> SWEState2D {
        if state.h < self.base_friction.h_min {
            return *state;
        }

        let n_eff = self.effective_manning(x, y);

        let u = state.hu / state.h;
        let v = state.hv / state.h;
        let speed = (u * u + v * v).sqrt();

        if speed < 1e-14 {
            return *state;
        }

        let h_eff = state.h.max(self.base_friction.h_min);
        let c_f = self.base_friction.g * n_eff * n_eff / h_eff.powf(1.0 / 3.0);

        // Damping factor
        let denom = 1.0 + dt * c_f * speed / state.h;
        let u_new = u / denom;
        let v_new = v / denom;

        SWEState2D {
            h: state.h,
            hu: state.h * u_new,
            hv: state.h * v_new,
        }
    }
}

impl<F> SourceTerm2D for StraitFriction2D<F>
where
    F: Fn(f64, f64) -> Option<f64> + Send + Sync,
{
    fn evaluate(&self, ctx: &SourceContext2D) -> SWEState2D {
        self.evaluate_with_effective_manning(ctx)
    }

    fn name(&self) -> &'static str {
        "strait_friction_2d"
    }

    fn is_stiff(&self) -> bool {
        true
    }
}

/// Helper to create a strait width function from a simple geometric definition.
///
/// Creates a strait region as a rectangular area with a specified center width.
///
/// # Arguments
/// * `x_start` - Start of strait in x direction (m)
/// * `x_end` - End of strait in x direction (m)
/// * `y_min` - Minimum y of strait region (m)
/// * `y_max` - Maximum y of strait region (m)
/// * `center_width` - Width at strait center (m)
///
/// # Returns
/// A closure that returns the strait width or None
pub fn rectangular_strait(
    x_start: f64,
    x_end: f64,
    y_min: f64,
    y_max: f64,
    center_width: f64,
) -> impl Fn(f64, f64) -> Option<f64> + Send + Sync {
    move |x, y| {
        if x >= x_start && x <= x_end && y >= y_min && y <= y_max {
            // Simple: constant width throughout
            Some(center_width)
        } else {
            None
        }
    }
}

/// Helper to create a strait width function with tapering ends.
///
/// Creates a strait where width varies smoothly from wider at the ends
/// to narrower at the center.
///
/// # Arguments
/// * `x_start` - Start of strait in x direction (m)
/// * `x_center` - Center of strait (m)
/// * `x_end` - End of strait in x direction (m)
/// * `y_min` - Minimum y of strait region (m)
/// * `y_max` - Maximum y of strait region (m)
/// * `center_width` - Minimum width at center (m)
/// * `end_width` - Width at strait ends (m)
///
/// # Returns
/// A closure that returns the strait width or None
pub fn tapered_strait(
    x_start: f64,
    x_center: f64,
    x_end: f64,
    y_min: f64,
    y_max: f64,
    center_width: f64,
    end_width: f64,
) -> impl Fn(f64, f64) -> Option<f64> + Send + Sync {
    move |x, y| {
        if x >= x_start && x <= x_end && y >= y_min && y <= y_max {
            // Linear taper from ends to center
            let dx = (x - x_center).abs();
            let half_length = ((x_end - x_start) / 2.0).max(1e-10);
            let t = (dx / half_length).min(1.0); // 0 at center, 1 at ends
            Some(center_width + t * (end_width - center_width))
        } else {
            None
        }
    }
}

/// Helper to create a strait width function from bathymetry contours.
///
/// In many applications, strait width is defined by where depth exceeds
/// a threshold. This helper computes width from a bathymetry function
/// by integration perpendicular to the strait axis.
///
/// # Arguments
/// * `strait_axis` - Direction of strait (0 = x-aligned, 1 = y-aligned)
/// * `depth_fn` - Bathymetry function B(x, y)
/// * `depth_threshold` - Depth threshold for "navigable" water
/// * `x_range` - Range in x where strait exists
/// * `y_range` - Range in y where strait exists
/// * `integration_step` - Step size for width integration
///
/// # Returns
/// A closure that returns the strait width or None
pub fn bathymetry_based_width<B>(
    strait_axis: usize,
    depth_fn: B,
    depth_threshold: f64,
    x_range: (f64, f64),
    y_range: (f64, f64),
    integration_step: f64,
) -> impl Fn(f64, f64) -> Option<f64> + Send + Sync
where
    B: Fn(f64, f64) -> f64 + Send + Sync,
{
    move |x, y| {
        // Check if in strait region
        if x < x_range.0 || x > x_range.1 || y < y_range.0 || y > y_range.1 {
            return None;
        }

        // Integrate perpendicular to strait axis to find width
        let width = if strait_axis == 0 {
            // x-aligned strait: integrate in y
            let mut w = 0.0;
            let mut yi = y_range.0;
            while yi <= y_range.1 {
                if depth_fn(x, yi) >= depth_threshold {
                    w += integration_step;
                }
                yi += integration_step;
            }
            w
        } else {
            // y-aligned strait: integrate in x
            let mut w = 0.0;
            let mut xi = x_range.0;
            while xi <= x_range.1 {
                if depth_fn(xi, y) >= depth_threshold {
                    w += integration_step;
                }
                xi += integration_step;
            }
            w
        };

        if width > 0.0 { Some(width) } else { None }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const G: f64 = 10.0;
    const TOL: f64 = 1e-10;

    fn make_context_at(x: f64, y: f64, h: f64, hu: f64, hv: f64) -> SourceContext2D {
        SourceContext2D::new(
            0.0,
            (x, y),
            SWEState2D::new(h, hu, hv),
            0.0,
            (0.0, 0.0),
            G,
            1e-6,
        )
    }

    #[test]
    fn test_no_enhancement_outside_strait() {
        let width_fn = |x: f64, _y: f64| {
            if (400.0..=600.0).contains(&x) {
                Some(300.0)
            } else {
                None
            }
        };

        let friction = StraitFriction2D::new(ManningFriction2D::new(G, 0.03), width_fn, 500.0, 2.0);

        // Outside strait region
        let n_eff = friction.effective_manning(100.0, 0.0);
        assert!(
            (n_eff - 0.03).abs() < TOL,
            "Should use base n outside strait"
        );
    }

    #[test]
    fn test_enhancement_in_narrow_strait() {
        let width_fn = |_x: f64, _y: f64| Some(100.0); // Very narrow

        let friction = StraitFriction2D::new(ManningFriction2D::new(G, 0.03), width_fn, 500.0, 2.0);

        // Enhancement should apply
        let n_eff = friction.effective_manning(0.0, 0.0);
        // n_eff = 0.03 * (1 + 2 * (1 - 100/500)) = 0.03 * (1 + 2 * 0.8) = 0.03 * 2.6
        let expected = 0.03 * 2.6;
        assert!(
            (n_eff - expected).abs() < 1e-10,
            "n_eff={}, expected={}",
            n_eff,
            expected
        );
    }

    #[test]
    fn test_no_enhancement_wide_strait() {
        let width_fn = |_x: f64, _y: f64| Some(600.0); // Wider than critical

        let friction = StraitFriction2D::new(ManningFriction2D::new(G, 0.03), width_fn, 500.0, 2.0);

        let n_eff = friction.effective_manning(0.0, 0.0);
        assert!(
            (n_eff - 0.03).abs() < TOL,
            "No enhancement above critical width"
        );
    }

    #[test]
    fn test_enhancement_at_critical_width() {
        let width_fn = |_x: f64, _y: f64| Some(500.0); // Exactly at critical

        let friction = StraitFriction2D::new(ManningFriction2D::new(G, 0.03), width_fn, 500.0, 2.0);

        // At critical width: (1 - 500/500) = 0, so no enhancement
        let n_eff = friction.effective_manning(0.0, 0.0);
        assert!(
            (n_eff - 0.03).abs() < TOL,
            "No enhancement at critical width"
        );
    }

    #[test]
    fn test_enhanced_friction_stronger() {
        let width_fn = |x: f64, _y: f64| {
            if (0.0..=20.0).contains(&x) {
                Some(100.0) // Narrow
            } else {
                None
            }
        };

        let friction = StraitFriction2D::new(ManningFriction2D::new(G, 0.03), width_fn, 500.0, 2.0);

        // Same state, different locations
        let h = 2.0;
        let u = 2.0;

        let ctx_strait = make_context_at(10.0, 0.0, h, u * h, 0.0);
        let ctx_outside = make_context_at(50.0, 0.0, h, u * h, 0.0);

        let s_strait = friction.evaluate(&ctx_strait);
        let s_outside = friction.evaluate(&ctx_outside);

        assert!(
            s_strait.hu.abs() > s_outside.hu.abs(),
            "Friction should be stronger in narrow strait: strait={}, outside={}",
            s_strait.hu.abs(),
            s_outside.hu.abs()
        );
    }

    #[test]
    fn test_zero_velocity_no_friction() {
        let width_fn = |_x: f64, _y: f64| Some(100.0);

        let friction = StraitFriction2D::new(ManningFriction2D::new(G, 0.03), width_fn, 500.0, 2.0);

        let ctx = make_context_at(0.0, 0.0, 2.0, 0.0, 0.0);
        let s = friction.evaluate(&ctx);

        assert!(s.h.abs() < TOL);
        assert!(s.hu.abs() < TOL);
        assert!(s.hv.abs() < TOL);
    }

    #[test]
    fn test_dry_cell_no_friction() {
        let width_fn = |_x: f64, _y: f64| Some(100.0);

        let friction = StraitFriction2D::new(ManningFriction2D::new(G, 0.03), width_fn, 500.0, 2.0);

        let ctx = make_context_at(0.0, 0.0, 1e-10, 1e-10, 0.0);
        let s = friction.evaluate(&ctx);

        assert!(s.h.abs() < TOL);
        assert!(s.hu.abs() < TOL);
        assert!(s.hv.abs() < TOL);
    }

    #[test]
    fn test_local_enhancement_factor() {
        let width_fn = |_x: f64, _y: f64| Some(250.0); // Half of critical

        let friction = StraitFriction2D::new(ManningFriction2D::new(G, 0.03), width_fn, 500.0, 2.0);

        // enhancement = 1 + 2 * (1 - 250/500) = 1 + 2 * 0.5 = 2.0
        let enh = friction.local_enhancement(0.0, 0.0);
        assert!((enh - 2.0).abs() < TOL, "Enhancement should be 2.0");
    }

    #[test]
    fn test_is_in_strait() {
        let width_fn = |x: f64, _y: f64| {
            if (0.0..=100.0).contains(&x) {
                Some(200.0)
            } else {
                None
            }
        };

        let friction = StraitFriction2D::new(ManningFriction2D::new(G, 0.03), width_fn, 500.0, 2.0);

        assert!(friction.is_in_strait(50.0, 0.0));
        assert!(!friction.is_in_strait(200.0, 0.0));
    }

    #[test]
    fn test_semi_implicit_update() {
        let width_fn = |_x: f64, _y: f64| Some(100.0);

        let friction = StraitFriction2D::new(ManningFriction2D::new(G, 0.03), width_fn, 500.0, 2.0);

        let state = SWEState2D::new(2.0, 4.0, 2.0);
        let state_new = friction.semi_implicit_update(&state, 0.0, 0.0, 0.1);

        // Depth unchanged
        assert!((state_new.h - state.h).abs() < TOL);

        // Velocities should decrease
        assert!(state_new.hu.abs() < state.hu.abs());
        assert!(state_new.hv.abs() < state.hv.abs());
    }

    #[test]
    fn test_rectangular_strait_helper() {
        let width_fn = rectangular_strait(0.0, 100.0, -50.0, 50.0, 200.0);

        assert!(width_fn(50.0, 0.0).is_some());
        assert_eq!(width_fn(50.0, 0.0), Some(200.0));
        assert!(width_fn(200.0, 0.0).is_none());
    }

    #[test]
    fn test_tapered_strait_helper() {
        let width_fn = tapered_strait(0.0, 50.0, 100.0, -50.0, 50.0, 100.0, 300.0);

        // At center: width = center_width = 100
        let w_center = width_fn(50.0, 0.0).unwrap();
        assert!((w_center - 100.0).abs() < 1.0);

        // At ends: width = end_width = 300
        let w_end = width_fn(100.0, 0.0).unwrap();
        assert!((w_end - 300.0).abs() < 1.0);
    }

    #[test]
    fn test_narrow_sound_preset() {
        let width_fn = |_x: f64, _y: f64| Some(100.0);
        let friction = StraitFriction2D::narrow_sound(ManningFriction2D::fjord(), width_fn);

        assert_eq!(friction.critical_width, 200.0);
        assert_eq!(friction.enhancement_factor, 3.0);
    }

    #[test]
    fn test_name() {
        let width_fn = |_x: f64, _y: f64| Some(100.0);
        let friction = StraitFriction2D::new(ManningFriction2D::new(G, 0.03), width_fn, 500.0, 2.0);

        assert_eq!(friction.name(), "strait_friction_2d");
    }

    #[test]
    fn test_is_stiff() {
        let width_fn = |_x: f64, _y: f64| Some(100.0);
        let friction = StraitFriction2D::new(ManningFriction2D::new(G, 0.03), width_fn, 500.0, 2.0);

        assert!(friction.is_stiff());
    }
}

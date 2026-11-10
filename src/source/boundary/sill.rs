//! Sill overflow dynamics source term for shallow water equations.
//!
//! Sills are shallow areas in fjords that separate deeper basins. Flow over
//! sills exhibits hydraulic control, enhanced mixing, and form drag effects
//! that cannot be fully resolved in typical coastal models.
//!
//! This module provides a parameterization based on bathymetry gradient:
//! - Where bathymetry gradient exceeds a threshold, enhanced form drag is applied
//! - This accounts for pressure drag, flow separation, and sub-grid turbulence
//!
//! # Physical Background
//!
//! At sills, several processes occur that affect flow dynamics:
//!
//! 1. **Form drag**: Pressure differences across the sill create drag
//!    proportional to the bathymetry slope
//!
//! 2. **Hydraulic control**: When flow becomes critical (Fr ≈ 1) at the sill,
//!    the upstream conditions are "locked" independent of downstream
//!
//! 3. **Internal hydraulics**: In stratified flows (not modeled here), internal
//!    waves and lee waves form behind sills
//!
//! 4. **Mixing**: Strong velocity shear over sills enhances vertical mixing
//!
//! # Norwegian Fjord Context
//!
//! Many Norwegian fjords have entrance sills that are much shallower than
//! the basin behind:
//! - Sognefjord: 165m sill, 1308m maximum depth
//! - Hardangerfjord: ~150m sill, ~800m maximum depth
//! - Oslofjord (Drøbak): ~20m sill depth
//!
//! These sills control water exchange between fjords and the open ocean.
//!
//! # Implementation
//!
//! The form drag is parameterized as:
//!
//! S_hu = -C_d * |grad B|² * |u| * u / h   (when |grad B| > threshold)
//! S_hv = -C_d * |grad B|² * |u| * v / h
//!
//! where C_d is a form drag coefficient.

use crate::solver::SWEState2D;
use crate::source::{SourceContext2D, SourceTerm2D};

/// Sill overflow dynamics source term.
///
/// Applies enhanced form drag in regions where bathymetry gradient exceeds
/// a threshold, parameterizing sub-grid scale effects of flow over sills.
///
/// # Example
///
/// ```ignore
/// use dg::source::SillOverflow2D;
///
/// // Create sill overflow source term
/// let sill = SillOverflow2D::new(0.1, 0.5);  // gradient threshold = 0.1, Cd = 0.5
/// ```
#[derive(Clone, Debug)]
pub struct SillOverflow2D {
    /// Gravitational acceleration (m/s²)
    pub g: f64,
    /// Bathymetry gradient threshold for sill detection (dimensionless = rise/run)
    /// Typical values: 0.05-0.2 (5-20% slope)
    pub gradient_threshold: f64,
    /// Form drag coefficient (dimensionless)
    /// Scales the drag enhancement over sills
    pub form_drag_coeff: f64,
    /// Minimum depth for source evaluation (m)
    pub h_min: f64,
    /// Maximum enhancement factor to prevent extreme values at steep slopes
    pub max_enhancement: f64,
}

impl SillOverflow2D {
    /// Create a new sill overflow source term.
    ///
    /// # Arguments
    /// * `gradient_threshold` - Bathymetry gradient magnitude threshold (dimensionless)
    /// * `form_drag_coeff` - Form drag coefficient
    pub fn new(gradient_threshold: f64, form_drag_coeff: f64) -> Self {
        Self {
            g: 9.81,
            gradient_threshold,
            form_drag_coeff,
            h_min: 1e-6,
            max_enhancement: 10.0,
        }
    }

    /// Create with custom parameters.
    pub fn with_parameters(
        g: f64,
        gradient_threshold: f64,
        form_drag_coeff: f64,
        h_min: f64,
        max_enhancement: f64,
    ) -> Self {
        Self {
            g,
            gradient_threshold,
            form_drag_coeff,
            h_min,
            max_enhancement,
        }
    }

    /// Preset for gradual sills with moderate enhancement.
    pub fn gentle_sill() -> Self {
        Self::new(0.05, 0.2)
    }

    /// Preset for very steep sills (>20% grade).
    pub fn steep_sill() -> Self {
        Self::new(0.2, 0.5)
    }

    /// Check if a location is on a sill (gradient exceeds threshold).
    pub fn is_on_sill(&self, gradient: (f64, f64)) -> bool {
        let grad_mag = self.gradient_magnitude(gradient);
        grad_mag > self.gradient_threshold
    }

    /// Compute the gradient magnitude.
    fn gradient_magnitude(&self, gradient: (f64, f64)) -> f64 {
        let (db_dx, db_dy) = gradient;
        (db_dx * db_dx + db_dy * db_dy).sqrt()
    }

    /// Compute the form drag enhancement factor.
    ///
    /// Returns a factor between 0 and max_enhancement based on
    /// how much the gradient exceeds the threshold.
    pub fn enhancement_factor(&self, gradient: (f64, f64)) -> f64 {
        let grad_mag = self.gradient_magnitude(gradient);

        if grad_mag <= self.gradient_threshold {
            return 0.0;
        }

        // Enhancement scales with gradient magnitude squared
        // (excess gradient beyond threshold)
        let excess = grad_mag - self.gradient_threshold;
        let enhancement = self.form_drag_coeff * excess * excess;

        enhancement.min(self.max_enhancement)
    }

    /// Compute form drag source term.
    fn evaluate_form_drag(&self, ctx: &SourceContext2D) -> SWEState2D {
        if ctx.state.h < self.h_min {
            return SWEState2D::zero();
        }

        let (db_dx, db_dy) = ctx.bathymetry_gradient;
        let grad_mag = self.gradient_magnitude((db_dx, db_dy));

        // Only apply if gradient exceeds threshold
        if grad_mag <= self.gradient_threshold {
            return SWEState2D::zero();
        }

        // Compute velocity
        let u = ctx.state.hu / ctx.state.h;
        let v = ctx.state.hv / ctx.state.h;
        let speed = (u * u + v * v).sqrt();

        if speed < 1e-14 {
            return SWEState2D::zero();
        }

        // Form drag coefficient: C_d * (|grad B| - threshold)²
        let enhancement = self.enhancement_factor((db_dx, db_dy));

        // Form drag: S = -enhancement * |u| * u / h
        // (opposes the flow direction, stronger for faster flow)
        let h_eff = ctx.state.h.max(self.h_min);

        SWEState2D {
            h: 0.0,
            hu: -enhancement * speed * u / h_eff,
            hv: -enhancement * speed * v / h_eff,
        }
    }

    /// Semi-implicit update for stiff sill drag.
    ///
    /// Uses same semi-implicit treatment as Manning friction.
    pub fn semi_implicit_update(
        &self,
        state: &SWEState2D,
        gradient: (f64, f64),
        dt: f64,
    ) -> SWEState2D {
        if state.h < self.h_min {
            return *state;
        }

        let enhancement = self.enhancement_factor(gradient);
        if enhancement < 1e-14 {
            return *state;
        }

        let u = state.hu / state.h;
        let v = state.hv / state.h;
        let speed = (u * u + v * v).sqrt();

        if speed < 1e-14 {
            return *state;
        }

        // Damping factor
        let h_eff = state.h.max(self.h_min);
        let denom = 1.0 + dt * enhancement * speed / h_eff;
        let u_new = u / denom;
        let v_new = v / denom;

        SWEState2D {
            h: state.h,
            hu: state.h * u_new,
            hv: state.h * v_new,
        }
    }

    /// Compute local Froude number.
    ///
    /// Useful for diagnosing hydraulic control at sills.
    /// Fr = |u| / sqrt(g*h)
    pub fn froude_number(&self, state: &SWEState2D) -> f64 {
        if state.h < self.h_min {
            return 0.0;
        }

        let u = state.hu / state.h;
        let v = state.hv / state.h;
        let speed = (u * u + v * v).sqrt();
        let wave_speed = (self.g * state.h).sqrt();

        speed / wave_speed
    }

    /// Check if flow is critical (Fr ≈ 1) indicating hydraulic control.
    pub fn is_critical_flow(&self, state: &SWEState2D) -> bool {
        let fr = self.froude_number(state);
        (fr - 1.0).abs() < 0.2 // Within 20% of critical
    }

    /// Check if flow is supercritical (Fr > 1).
    pub fn is_supercritical(&self, state: &SWEState2D) -> bool {
        self.froude_number(state) > 1.0
    }
}

impl SourceTerm2D for SillOverflow2D {
    fn evaluate(&self, ctx: &SourceContext2D) -> SWEState2D {
        self.evaluate_form_drag(ctx)
    }

    fn name(&self) -> &'static str {
        "sill_overflow_2d"
    }

    fn is_stiff(&self) -> bool {
        true
    }
}

/// Combined sill overflow and friction source.
///
/// For efficiency, this combines sill drag and bottom friction in a single
/// source term, sharing the velocity computation.
#[derive(Clone, Debug)]
pub struct SillWithFriction {
    /// Sill overflow parameters
    pub sill: SillOverflow2D,
    /// Base Manning coefficient
    pub manning_n: f64,
}

impl SillWithFriction {
    /// Create combined sill and friction source.
    pub fn new(sill: SillOverflow2D, manning_n: f64) -> Self {
        Self { sill, manning_n }
    }
}

impl SourceTerm2D for SillWithFriction {
    fn evaluate(&self, ctx: &SourceContext2D) -> SWEState2D {
        if ctx.state.h < self.sill.h_min {
            return SWEState2D::zero();
        }

        let u = ctx.state.hu / ctx.state.h;
        let v = ctx.state.hv / ctx.state.h;
        let speed = (u * u + v * v).sqrt();

        if speed < 1e-14 {
            return SWEState2D::zero();
        }

        let h_eff = ctx.state.h.max(self.sill.h_min);

        // Manning friction coefficient
        let c_f = self.sill.g * self.manning_n * self.manning_n / h_eff.powf(1.0 / 3.0);

        // Sill enhancement
        let sill_enhancement = self.sill.enhancement_factor(ctx.bathymetry_gradient);

        // Total drag = Manning + Sill form drag
        let total_drag = c_f + sill_enhancement / h_eff;

        SWEState2D {
            h: 0.0,
            hu: -total_drag * speed * u,
            hv: -total_drag * speed * v,
        }
    }

    fn name(&self) -> &'static str {
        "sill_with_friction_2d"
    }

    fn is_stiff(&self) -> bool {
        true
    }
}

/// Helper to detect sill locations from bathymetry.
///
/// Identifies sill crests (local maxima in bathymetry along flow paths)
/// for diagnostic purposes.
pub struct SillDetector {
    /// Gradient threshold for sill detection
    pub gradient_threshold: f64,
    /// Minimum depth to consider a sill
    pub min_depth: f64,
}

impl SillDetector {
    /// Create a new sill detector.
    pub fn new(gradient_threshold: f64, min_depth: f64) -> Self {
        Self {
            gradient_threshold,
            min_depth,
        }
    }

    /// Check if a point is on a sill based on bathymetry gradient.
    pub fn is_sill_region(&self, depth: f64, gradient: (f64, f64)) -> bool {
        if depth < self.min_depth {
            return false;
        }

        let (db_dx, db_dy) = gradient;
        let grad_mag = (db_dx * db_dx + db_dy * db_dy).sqrt();
        grad_mag > self.gradient_threshold
    }

    /// Estimate sill depth from surrounding gradients.
    ///
    /// The sill crest is where the gradient changes sign (upslope to downslope).
    pub fn estimate_sill_crest(
        &self,
        depths: &[f64],
        gradients: &[(f64, f64)],
    ) -> Option<(usize, f64)> {
        if depths.len() < 3 || depths.len() != gradients.len() {
            return None;
        }

        // Look for gradient sign change (upslope to downslope)
        // This indicates a local maximum = sill crest
        for i in 1..gradients.len() - 1 {
            let (grad_x_prev, _) = gradients[i - 1];
            let (grad_x_curr, _) = gradients[i];
            let (grad_x_next, _) = gradients[i + 1];

            // Sign change in x-gradient (simplified 1D along x)
            if grad_x_prev > 0.0 && grad_x_next < 0.0 {
                // Local maximum found
                return Some((i, depths[i]));
            }
            // Also check if gradient magnitude is small (crest = flat spot)
            if grad_x_curr.abs() < self.gradient_threshold * 0.1 && grad_x_prev * grad_x_next < 0.0
            {
                return Some((i, depths[i]));
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const G: f64 = 10.0;
    const TOL: f64 = 1e-10;

    fn make_context(h: f64, hu: f64, hv: f64, db_dx: f64, db_dy: f64) -> SourceContext2D {
        SourceContext2D::new(
            0.0,
            (0.0, 0.0),
            SWEState2D::new(h, hu, hv),
            0.0,
            (db_dx, db_dy),
            G,
            1e-6,
        )
    }

    #[test]
    fn test_no_drag_below_threshold() {
        let sill = SillOverflow2D::with_parameters(G, 0.1, 0.3, 1e-6, 10.0);

        // Gradient below threshold
        let ctx = make_context(2.0, 4.0, 0.0, 0.05, 0.0);
        let s = sill.evaluate(&ctx);

        assert!(s.h.abs() < TOL);
        assert!(s.hu.abs() < TOL);
        assert!(s.hv.abs() < TOL);
    }

    #[test]
    fn test_drag_above_threshold() {
        let sill = SillOverflow2D::with_parameters(G, 0.1, 0.3, 1e-6, 10.0);

        // Gradient above threshold (0.2 > 0.1)
        let ctx = make_context(2.0, 4.0, 0.0, 0.2, 0.0);
        let s = sill.evaluate(&ctx);

        assert!(s.h.abs() < TOL, "h should not change");
        assert!(s.hu < 0.0, "Should oppose positive x-velocity");
        assert!(s.hv.abs() < TOL, "No y-component when v=0");
    }

    #[test]
    fn test_drag_opposes_flow() {
        let sill = SillOverflow2D::with_parameters(G, 0.1, 0.5, 1e-6, 10.0);

        // Test negative velocity
        let ctx = make_context(2.0, -4.0, -2.0, 0.2, 0.0);
        let s = sill.evaluate(&ctx);

        assert!(s.hu > 0.0, "Should oppose negative x-velocity");
        assert!(s.hv > 0.0, "Should oppose negative y-velocity");
    }

    #[test]
    fn test_zero_velocity_no_drag() {
        let sill = SillOverflow2D::with_parameters(G, 0.1, 0.3, 1e-6, 10.0);

        let ctx = make_context(2.0, 0.0, 0.0, 0.3, 0.0);
        let s = sill.evaluate(&ctx);

        assert!(s.h.abs() < TOL);
        assert!(s.hu.abs() < TOL);
        assert!(s.hv.abs() < TOL);
    }

    #[test]
    fn test_dry_cell_no_drag() {
        let sill = SillOverflow2D::with_parameters(G, 0.1, 0.3, 1e-6, 10.0);

        let ctx = make_context(1e-10, 1e-10, 0.0, 0.3, 0.0);
        let s = sill.evaluate(&ctx);

        assert!(s.h.abs() < TOL);
        assert!(s.hu.abs() < TOL);
        assert!(s.hv.abs() < TOL);
    }

    #[test]
    fn test_steeper_gradient_more_drag() {
        let sill = SillOverflow2D::with_parameters(G, 0.1, 0.3, 1e-6, 10.0);

        let ctx_mild = make_context(2.0, 4.0, 0.0, 0.15, 0.0);
        let ctx_steep = make_context(2.0, 4.0, 0.0, 0.3, 0.0);

        let s_mild = sill.evaluate(&ctx_mild);
        let s_steep = sill.evaluate(&ctx_steep);

        assert!(
            s_steep.hu.abs() > s_mild.hu.abs(),
            "Steeper gradient should give more drag: mild={}, steep={}",
            s_mild.hu.abs(),
            s_steep.hu.abs()
        );
    }

    #[test]
    fn test_diagonal_gradient() {
        let sill = SillOverflow2D::with_parameters(G, 0.1, 0.3, 1e-6, 10.0);

        // Diagonal gradient (0.15, 0.15) -> magnitude ≈ 0.21 > threshold 0.1
        let ctx = make_context(2.0, 4.0, 2.0, 0.15, 0.15);
        let s = sill.evaluate(&ctx);

        // Should have drag in both directions
        assert!(s.hu < 0.0, "Should have x-drag");
        assert!(s.hv < 0.0, "Should have y-drag");
    }

    #[test]
    fn test_enhancement_factor() {
        let sill = SillOverflow2D::with_parameters(G, 0.1, 1.0, 1e-6, 10.0);

        // At threshold: no enhancement
        let enh_at = sill.enhancement_factor((0.1, 0.0));
        assert!(enh_at.abs() < TOL);

        // Below threshold: no enhancement
        let enh_below = sill.enhancement_factor((0.05, 0.0));
        assert!(enh_below.abs() < TOL);

        // Above threshold: positive enhancement
        let enh_above = sill.enhancement_factor((0.2, 0.0));
        assert!(enh_above > 0.0);
        // enhancement = Cd * (0.2 - 0.1)² = 1.0 * 0.01 = 0.01
        assert!((enh_above - 0.01).abs() < TOL);
    }

    #[test]
    fn test_max_enhancement_cap() {
        let sill = SillOverflow2D::with_parameters(G, 0.1, 1000.0, 1e-6, 5.0);

        // Very steep gradient, but should be capped at max_enhancement=5.0
        let enh = sill.enhancement_factor((1.0, 0.0));
        assert!(enh <= 5.0 + TOL, "Enhancement should be capped: {}", enh);
    }

    #[test]
    fn test_is_on_sill() {
        let sill = SillOverflow2D::new(0.1, 0.3);

        assert!(!sill.is_on_sill((0.05, 0.0)), "Below threshold");
        assert!(!sill.is_on_sill((0.1, 0.0)), "At threshold");
        assert!(sill.is_on_sill((0.15, 0.0)), "Above threshold");
        assert!(sill.is_on_sill((0.08, 0.08)), "Diagonal above threshold");
    }

    #[test]
    fn test_froude_number() {
        let sill = SillOverflow2D::with_parameters(G, 0.1, 0.3, 1e-6, 10.0);

        // h=10, u=sqrt(g*h)=10 -> Fr=1
        let state = SWEState2D::new(10.0, 100.0, 0.0); // h=10, u=10
        let fr = sill.froude_number(&state);
        assert!((fr - 1.0).abs() < 0.01, "Fr should be ~1.0, got {}", fr);

        // Subcritical
        let state_sub = SWEState2D::new(10.0, 50.0, 0.0); // u=5
        assert!(sill.froude_number(&state_sub) < 1.0);

        // Supercritical
        let state_sup = SWEState2D::new(10.0, 200.0, 0.0); // u=20
        assert!(sill.froude_number(&state_sup) > 1.0);
    }

    #[test]
    fn test_is_critical_flow() {
        let sill = SillOverflow2D::with_parameters(G, 0.1, 0.3, 1e-6, 10.0);

        // Critical: Fr = 1
        let state_crit = SWEState2D::new(10.0, 100.0, 0.0);
        assert!(sill.is_critical_flow(&state_crit));

        // Subcritical: Fr << 1
        let state_sub = SWEState2D::new(10.0, 30.0, 0.0);
        assert!(!sill.is_critical_flow(&state_sub));

        // Supercritical: Fr >> 1
        let state_sup = SWEState2D::new(10.0, 300.0, 0.0);
        assert!(!sill.is_critical_flow(&state_sup));
    }

    #[test]
    fn test_semi_implicit_update() {
        let sill = SillOverflow2D::with_parameters(G, 0.1, 0.5, 1e-6, 10.0);

        let state = SWEState2D::new(2.0, 4.0, 2.0);
        let gradient = (0.3, 0.0); // Above threshold
        let dt = 0.1;

        let state_new = sill.semi_implicit_update(&state, gradient, dt);

        // Depth should be unchanged
        assert!((state_new.h - state.h).abs() < TOL);

        // Velocities should decrease
        assert!(state_new.hu.abs() < state.hu.abs());
        assert!(state_new.hv.abs() < state.hv.abs());

        // Signs should be preserved
        assert!(state_new.hu > 0.0);
        assert!(state_new.hv > 0.0);
    }

    #[test]
    fn test_semi_implicit_no_change_below_threshold() {
        let sill = SillOverflow2D::with_parameters(G, 0.1, 0.5, 1e-6, 10.0);

        let state = SWEState2D::new(2.0, 4.0, 2.0);
        let gradient = (0.05, 0.0); // Below threshold
        let dt = 0.1;

        let state_new = sill.semi_implicit_update(&state, gradient, dt);

        // Should be unchanged
        assert!((state_new.h - state.h).abs() < TOL);
        assert!((state_new.hu - state.hu).abs() < TOL);
        assert!((state_new.hv - state.hv).abs() < TOL);
    }

    #[test]
    fn test_presets() {
        let gentle = SillOverflow2D::gentle_sill();
        assert!((gentle.gradient_threshold - 0.05).abs() < TOL);

        let steep = SillOverflow2D::steep_sill();
        assert!((steep.gradient_threshold - 0.2).abs() < TOL);
    }

    #[test]
    fn test_sill_with_friction() {
        let combined = SillWithFriction::new(SillOverflow2D::new(0.1, 0.3), 0.025);

        // Test on flat bottom (no sill enhancement)
        let ctx_flat = make_context(2.0, 4.0, 0.0, 0.0, 0.0);
        let s_flat = combined.evaluate(&ctx_flat);

        // Should still have Manning friction
        assert!(s_flat.hu < 0.0, "Should have friction even on flat bottom");

        // Test on sill (additional enhancement)
        let ctx_sill = make_context(2.0, 4.0, 0.0, 0.2, 0.0);
        let s_sill = combined.evaluate(&ctx_sill);

        // Should have more drag on sill
        assert!(
            s_sill.hu.abs() > s_flat.hu.abs(),
            "Sill should increase drag"
        );
    }

    #[test]
    fn test_name() {
        let sill = SillOverflow2D::new(0.1, 0.3);
        assert_eq!(sill.name(), "sill_overflow_2d");

        let combined = SillWithFriction::new(SillOverflow2D::new(0.1, 0.3), 0.025);
        assert_eq!(combined.name(), "sill_with_friction_2d");
    }

    #[test]
    fn test_is_stiff() {
        let sill = SillOverflow2D::new(0.1, 0.3);
        assert!(sill.is_stiff());

        let combined = SillWithFriction::new(SillOverflow2D::new(0.1, 0.3), 0.025);
        assert!(combined.is_stiff());
    }

    #[test]
    fn test_sill_detector() {
        let detector = SillDetector::new(0.1, 5.0);

        // Below threshold
        assert!(!detector.is_sill_region(10.0, (0.05, 0.0)));

        // Above threshold
        assert!(detector.is_sill_region(10.0, (0.2, 0.0)));

        // Too shallow
        assert!(!detector.is_sill_region(2.0, (0.2, 0.0)));
    }

    #[test]
    fn test_sill_crest_detection() {
        let detector = SillDetector::new(0.1, 5.0);

        // Depths showing upslope then downslope (sill at index 2)
        let depths = vec![20.0, 15.0, 10.0, 15.0, 20.0];
        let gradients = vec![
            (0.3, 0.0),  // upslope
            (0.2, 0.0),  // upslope
            (0.0, 0.0),  // crest (flat)
            (-0.2, 0.0), // downslope
            (-0.3, 0.0), // downslope
        ];

        let crest = detector.estimate_sill_crest(&depths, &gradients);
        assert!(crest.is_some());
        let (idx, depth) = crest.unwrap();
        assert_eq!(idx, 2);
        assert!((depth - 10.0).abs() < TOL);
    }
}

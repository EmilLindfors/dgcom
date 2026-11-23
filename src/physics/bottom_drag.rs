//! Quadratic bottom drag of the 3D model.
//!
//! The bottom stress is the quadratic drag of the bottom-layer velocity `u_b`,
//!
//! ```text
//!     τ_b/ρ₀ = C_d |u_b| u_b,
//! ```
//!
//! with a constant `C_d`, or the drag coefficient of a logarithmic layer
//! between the bed and the bottom-layer centre, `z_b` above the bed:
//!
//! ```text
//!     C_d = (κ / ln(z_b/z₀))²,    κ = 0.41,
//! ```
//!
//! bounded to `[C_d,min, C_d,max]` (ROMS `UV_LOGDRAG`; FVCOM bounds it below
//! by 0.0025). `z₀` is the bed roughness length.
//!
//! # Time discretisation (with [`crate::time::ModeSplitIntegrator`])
//!
//! The drag is linearised over each baroclinic step: `τ_b/ρ₀ = r·u_b` with
//! the rate `r = C_d|u_bⁿ|` (m/s) frozen at `tⁿ`.
//! - The depth mean feels `−r·ū` in the barotropic pass, point-implicitly
//!   (`(hu, hv) ← (hu, hv)/(1 + Δt_s·r/h)` per stage): stable for any
//!   `r·Δt/D`, and it never reverses the flow. The slow forcing `G` carries
//!   the rest, `−r·(u_bⁿ − ūⁿ)`, which vanishes without vertical shear. So
//!   for unsheared flow the 3D model has the 2D quadratic friction
//!   `C_d|ū|ū/h` (Chézy with `C = √(g/C_d)`).
//! - The vertical diffusion takes `r·u_bⁿ⁺¹` as its bottom flux, implicitly.
//!   It shapes the profile only: the splitter resets the depth mean to the
//!   pass's `ū` afterwards.
//!
//! Freezing `r` makes the drag first order in time, like the point-implicit
//! 2D friction.

/// Von Kármán's constant.
pub const VON_KARMAN: f64 = 0.41;

/// Bottom drag law of the 3D model (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BottomDrag3D {
    /// Constant drag coefficient `C_d`.
    Quadratic {
        /// Drag coefficient (dimensionless)
        cd: f64,
    },
    /// Log-layer drag coefficient of the bottom-layer height, bounded.
    LogLayer {
        /// Bed roughness length `z₀` (m)
        z0: f64,
        /// Lower bound of `C_d` (deep bottom layers)
        cd_min: f64,
        /// Upper bound of `C_d` (bottom layers not much thicker than `z₀`)
        cd_max: f64,
    },
}

impl BottomDrag3D {
    /// Default lower bound of the log-layer coefficient (FVCOM's `CBCMIN`).
    pub const DEFAULT_CD_MIN: f64 = 2.5e-3;
    /// Default upper bound of the log-layer coefficient.
    pub const DEFAULT_CD_MAX: f64 = 0.1;

    /// Constant drag coefficient `cd`.
    pub fn quadratic(cd: f64) -> Self {
        assert!(
            cd >= 0.0 && cd.is_finite(),
            "drag coefficient must be finite and non-negative, got {cd}"
        );
        Self::Quadratic { cd }
    }

    /// Log-layer drag over roughness length `z0` (m), bounded to
    /// [`Self::DEFAULT_CD_MIN`], [`Self::DEFAULT_CD_MAX`] (see
    /// [`Self::with_bounds`]).
    pub fn log_layer(z0: f64) -> Self {
        assert!(
            z0 > 0.0 && z0.is_finite(),
            "roughness length must be finite and positive, got {z0}"
        );
        Self::LogLayer {
            z0,
            cd_min: Self::DEFAULT_CD_MIN,
            cd_max: Self::DEFAULT_CD_MAX,
        }
    }

    /// The log-layer coefficient bounded to `[cd_min, cd_max]` instead.
    ///
    /// # Panics
    /// On a [`Self::Quadratic`] drag, or if `0 ≤ cd_min ≤ cd_max` fails.
    pub fn with_bounds(self, cd_min: f64, cd_max: f64) -> Self {
        assert!(
            0.0 <= cd_min && cd_min <= cd_max && cd_max.is_finite(),
            "need 0 ≤ cd_min ≤ cd_max < ∞, got [{cd_min}, {cd_max}]"
        );
        match self {
            Self::LogLayer { z0, .. } => Self::LogLayer { z0, cd_min, cd_max },
            Self::Quadratic { .. } => panic!("bounds apply to the log-layer drag only"),
        }
    }

    /// Drag coefficient for a bottom-layer centre `z_b` (m) above the bed.
    #[inline]
    pub fn drag_coefficient(&self, z_b: f64) -> f64 {
        match *self {
            Self::Quadratic { cd } => cd,
            Self::LogLayer { z0, cd_min, cd_max } => {
                // `crate::math`'s logarithm: the same bits on the GPU
                let log = crate::math::ln(z_b / z0);
                // At or below z₀ the log layer has no positive coefficient
                if log > 0.0 {
                    let ratio = VON_KARMAN / log;
                    (ratio * ratio).clamp(cd_min, cd_max)
                } else {
                    cd_max
                }
            }
        }
    }

    /// Linear drag rate `r = C_d(z_b)·speed` (m/s), `τ_b/ρ₀ = r·u_b`.
    #[inline]
    pub fn rate(&self, z_b: f64, speed: f64) -> f64 {
        self.drag_coefficient(z_b) * speed
    }

    /// The rate of a column of depth `depth` (m) whose bottom layer's
    /// centre is at `σ = sigma_bottom`: [`Self::rate`] at `z_b = (1 + σ) D`
    /// with the speed of the bottom layer's velocity `bottom`, or of the
    /// depth mean `mean` in a column shallower than `thin_depth`; zero where
    /// it is dry.
    #[inline]
    pub fn column_rate(
        &self,
        depth: f64,
        thin_depth: f64,
        sigma_bottom: f64,
        bottom: [f64; 2],
        mean: [f64; 2],
    ) -> f64 {
        if depth <= 0.0 {
            return 0.0;
        }
        let [u, v] = if depth < thin_depth { mean } else { bottom };
        let z_b = (1.0 + sigma_bottom) * depth;
        self.rate(z_b, (u * u + v * v).sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quadratic_coefficient_is_constant() {
        let drag = BottomDrag3D::quadratic(2.5e-3);
        for z in [1e-3, 0.5, 100.0] {
            assert_eq!(drag.drag_coefficient(z), 2.5e-3);
        }
        assert_eq!(drag.rate(3.0, 2.0), 5e-3);
    }

    #[test]
    fn log_layer_coefficient_follows_the_law_of_the_wall() {
        let drag = BottomDrag3D::log_layer(0.01).with_bounds(0.0, 1.0);
        // z_b = z₀·e⁴: C_d = (κ/4)²
        let z = 0.01 * 4.0_f64.exp();
        let expect = (VON_KARMAN / 4.0).powi(2);
        assert!((drag.drag_coefficient(z) - expect).abs() < 1e-15 * expect);
        // Falls with the height of the bottom layer
        assert!(drag.drag_coefficient(10.0) < drag.drag_coefficient(1.0));
    }

    #[test]
    fn log_layer_coefficient_is_bounded() {
        let drag = BottomDrag3D::log_layer(0.01);
        // Deep bottom layer: (0.41/ln(5000))² = 2.3e-3 → the lower bound
        assert_eq!(drag.drag_coefficient(50.0), BottomDrag3D::DEFAULT_CD_MIN);
        // Just above z₀, at z₀ and below it (a film): the upper bound
        for z in [0.011, 0.01, 1e-4, 0.0] {
            assert_eq!(
                drag.drag_coefficient(z),
                BottomDrag3D::DEFAULT_CD_MAX,
                "z_b = {z}"
            );
        }
        // In between, the law itself
        let z = 0.5;
        let expect = (VON_KARMAN / (z / 0.01_f64).ln()).powi(2);
        assert_eq!(drag.drag_coefficient(z), expect);
    }

    #[test]
    #[should_panic(expected = "log-layer drag only")]
    fn bounds_on_a_constant_coefficient_are_rejected() {
        let _ = BottomDrag3D::quadratic(1e-3).with_bounds(0.0, 1.0);
    }
}

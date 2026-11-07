//! Atmospheric pressure forcing for shallow water equations.
//!
//! Atmospheric pressure gradients drive currents and affect sea level through the
//! inverse barometer effect. This module implements the pressure gradient force
//! as a source term in the momentum equations.
//!
//! # Physics
//!
//! The atmospheric pressure gradient force per unit mass is:
//! ```text
//! a = -∇P / ρ_water
//! ```
//!
//! For the shallow water momentum equations (in terms of hu, hv), the source terms are:
//! ```text
//! S_h  = 0
//! S_hu = -h/ρ * ∂P/∂x
//! S_hv = -h/ρ * ∂P/∂y
//! ```
//!
//! # Inverse Barometer Effect
//!
//! Sea level responds to atmospheric pressure changes:
//! ```text
//! Δη ≈ -ΔP / (ρg) ≈ -1 cm per hPa
//! ```
//!
//! This quasi-static adjustment is typically handled through boundary conditions
//! or initial conditions, while this source term handles the dynamic forcing.
//!
//! # Typical Values
//!
//! - Standard sea level pressure: 101325 Pa (1013.25 hPa)
//! - Storm central pressure: 960-990 hPa (severe: 940-960 hPa)
//! - Pressure gradient in storm: 1-3 hPa per 100 km
//!
//! # Example
//!
//! ```rust
//! use dgcom::source::AtmosphericPressure2D;
//!
//! // Uniform pressure gradient (1 hPa per 100 km from west)
//! let pressure = AtmosphericPressure2D::uniform_gradient(1.0e-3, 0.0);
//!
//! // Moving storm (Holland model)
//! let storm = AtmosphericPressure2D::moving_storm(
//!     |t| (10.0 * t, 0.0),  // Storm center moves east at 10 m/s
//!     970.0,                 // Central pressure 970 hPa
//!     100.0,                 // Radius of max winds 100 km
//!     1.5,                   // Holland B parameter
//! );
//! ```

use crate::solver::SWEState2D;
use crate::source::{SourceContext2D, SourceTerm2D};

/// Standard atmospheric pressure at sea level (Pa).
pub const P_STANDARD: f64 = 101325.0;

/// Water density (kg/m³) for pressure forcing.
pub const RHO_WATER_PRESSURE: f64 = 1025.0;

/// Gravity (m/s²).
const G: f64 = 9.81;

/// Pressure field specification.
enum PressureField {
    /// Constant pressure everywhere (no gradient, no forcing).
    Constant(f64),

    /// Uniform pressure gradient (Pa/m in x and y directions).
    UniformGradient { p_ref: f64, dp_dx: f64, dp_dy: f64 },

    /// Time-varying uniform gradient.
    TimeVaryingGradient(Box<dyn Fn(f64) -> (f64, f64) + Send + Sync>),

    /// Spatio-temporal pressure field: (x, y, t) -> P(Pa).
    SpatioTemporal(Box<dyn Fn(f64, f64, f64) -> f64 + Send + Sync>),

    /// Analytical moving storm (Holland model).
    MovingStorm {
        /// Storm track: t -> (x_center, y_center)
        track: Box<dyn Fn(f64) -> (f64, f64) + Send + Sync>,
        /// Central pressure (Pa)
        p_center: f64,
        /// Ambient pressure (Pa)
        p_ambient: f64,
        /// Radius of maximum winds (m)
        r_max: f64,
        /// Holland B parameter (shape, typically 1.0-2.5)
        b_param: f64,
    },
}

/// Atmospheric pressure source term for 2D shallow water equations.
///
/// Computes the pressure gradient force on the water column:
/// ```text
/// S_hu = -h/ρ * ∂P/∂x
/// S_hv = -h/ρ * ∂P/∂y
/// ```
pub struct AtmosphericPressure2D {
    field: PressureField,
    rho_water: f64,
    h_min: f64,
    /// Finite difference step for numerical gradient (m).
    grad_delta: f64,
}

impl AtmosphericPressure2D {
    /// Create with constant pressure (no forcing).
    ///
    /// Useful as a placeholder or for testing.
    pub fn constant(pressure_pa: f64) -> Self {
        Self {
            field: PressureField::Constant(pressure_pa),
            rho_water: RHO_WATER_PRESSURE,
            h_min: 0.01,
            grad_delta: 100.0,
        }
    }

    /// Create with uniform pressure gradient.
    ///
    /// # Arguments
    /// * `dp_dx` - Pressure gradient in x direction (Pa/m)
    /// * `dp_dy` - Pressure gradient in y direction (Pa/m)
    ///
    /// # Example
    /// ```rust
    /// use dgcom::source::AtmosphericPressure2D;
    ///
    /// // 1 hPa per 100 km from west (pressure increasing eastward)
    /// let pressure = AtmosphericPressure2D::uniform_gradient(1.0e-3, 0.0);
    /// ```
    pub fn uniform_gradient(dp_dx: f64, dp_dy: f64) -> Self {
        Self {
            field: PressureField::UniformGradient {
                p_ref: P_STANDARD,
                dp_dx,
                dp_dy,
            },
            rho_water: RHO_WATER_PRESSURE,
            h_min: 0.01,
            grad_delta: 100.0,
        }
    }

    /// Create with uniform gradient from meteorological direction.
    ///
    /// # Arguments
    /// * `gradient_magnitude` - Pressure gradient magnitude (Pa/m)
    /// * `from_direction` - Direction gradient is FROM (meteorological convention, degrees)
    ///   - 0° = North (pressure decreasing northward)
    ///   - 90° = East
    ///   - 180° = South
    ///   - 270° = West
    ///
    /// # Example
    /// ```rust
    /// use dgcom::source::AtmosphericPressure2D;
    ///
    /// // Gradient from southwest (typical Norwegian storm direction)
    /// let pressure = AtmosphericPressure2D::from_direction(1.5e-3, 225.0);
    /// ```
    pub fn from_direction(gradient_magnitude: f64, from_direction: f64) -> Self {
        let dir_rad = from_direction.to_radians();
        // Gradient points FROM the direction (opposite to wind convention)
        let dp_dx = -gradient_magnitude * dir_rad.sin();
        let dp_dy = -gradient_magnitude * dir_rad.cos();

        Self::uniform_gradient(dp_dx, dp_dy)
    }

    /// Create with time-varying uniform gradient.
    ///
    /// # Arguments
    /// * `gradient_fn` - Function returning (dp_dx, dp_dy) at time t
    ///
    /// # Example
    /// ```rust
    /// use dgcom::source::AtmosphericPressure2D;
    ///
    /// // Rotating gradient (e.g., passing weather system)
    /// let pressure = AtmosphericPressure2D::time_varying_gradient(|t| {
    ///     let omega = 2.0 * std::f64::consts::PI / 86400.0; // One rotation per day
    ///     let mag = 1.0e-3; // 1 hPa per 100 km
    ///     (mag * (omega * t).cos(), mag * (omega * t).sin())
    /// });
    /// ```
    pub fn time_varying_gradient<F>(gradient_fn: F) -> Self
    where
        F: Fn(f64) -> (f64, f64) + Send + Sync + 'static,
    {
        Self {
            field: PressureField::TimeVaryingGradient(Box::new(gradient_fn)),
            rho_water: RHO_WATER_PRESSURE,
            h_min: 0.01,
            grad_delta: 100.0,
        }
    }

    /// Create with full spatio-temporal pressure field.
    ///
    /// # Arguments
    /// * `pressure_fn` - Function (x, y, t) -> pressure in Pa
    ///
    /// The gradient will be computed numerically using finite differences.
    ///
    /// # Example
    /// ```rust
    /// use dgcom::source::AtmosphericPressure2D;
    ///
    /// // Gaussian low pressure system moving eastward
    /// let pressure = AtmosphericPressure2D::spatio_temporal(|x, y, t| {
    ///     let x_center = 50000.0 + 10.0 * t; // 10 m/s eastward
    ///     let y_center = 100000.0_f64;
    ///     let r2 = (x - x_center).powi(2) + (y - y_center).powi(2);
    ///     let r_scale = 100000.0_f64; // 100 km scale
    ///     let p_drop = 3000.0_f64; // 30 hPa central drop
    ///     101325.0 - p_drop * (-r2 / r_scale.powi(2)).exp()
    /// });
    /// ```
    pub fn spatio_temporal<F>(pressure_fn: F) -> Self
    where
        F: Fn(f64, f64, f64) -> f64 + Send + Sync + 'static,
    {
        Self {
            field: PressureField::SpatioTemporal(Box::new(pressure_fn)),
            rho_water: RHO_WATER_PRESSURE,
            h_min: 0.01,
            grad_delta: 100.0,
        }
    }

    /// Create an idealized moving storm using the Holland pressure profile.
    ///
    /// The Holland (1980) model gives pressure as:
    /// ```text
    /// P(r) = P_center + (P_ambient - P_center) * exp(-(R_max/r)^B)
    /// ```
    ///
    /// # Arguments
    /// * `track` - Storm track function: t -> (x_center, y_center)
    /// * `p_center_hpa` - Central pressure (hPa)
    /// * `r_max_km` - Radius of maximum winds (km)
    /// * `b_param` - Holland B parameter (typically 1.0-2.5, higher = steeper gradient)
    ///
    /// # Example
    /// ```rust
    /// use dgcom::source::AtmosphericPressure2D;
    ///
    /// // Storm moving northeast at 15 m/s
    /// let storm = AtmosphericPressure2D::moving_storm(
    ///     |t| (10.0 * t, 10.0 * t),  // Track
    ///     970.0,  // Central pressure 970 hPa
    ///     50.0,   // R_max = 50 km
    ///     1.5,    // Holland B
    /// );
    /// ```
    pub fn moving_storm<F>(track: F, p_center_hpa: f64, r_max_km: f64, b_param: f64) -> Self
    where
        F: Fn(f64) -> (f64, f64) + Send + Sync + 'static,
    {
        Self {
            field: PressureField::MovingStorm {
                track: Box::new(track),
                p_center: p_center_hpa * 100.0,
                p_ambient: P_STANDARD,
                r_max: r_max_km * 1000.0,
                b_param,
            },
            rho_water: RHO_WATER_PRESSURE,
            h_min: 0.01,
            grad_delta: 100.0,
        }
    }

    /// Norwegian winter storm preset.
    ///
    /// Creates a typical North Atlantic low pressure system:
    /// - Central pressure: 970 hPa
    /// - Radius of max winds: 200 km
    /// - Moving northeast at 15 m/s
    /// - Holland B = 1.3 (broad storm)
    ///
    /// # Arguments
    /// * `x0`, `y0` - Initial storm center position (m)
    pub fn norwegian_winter_storm(x0: f64, y0: f64) -> Self {
        Self::moving_storm(
            move |t| {
                // Northeast movement at 15 m/s (typical for Norwegian coast)
                let speed = 15.0;
                let angle = 45.0_f64.to_radians();
                (x0 + speed * angle.cos() * t, y0 + speed * angle.sin() * t)
            },
            970.0,
            200.0,
            1.3,
        )
    }

    /// Severe Norwegian storm preset.
    ///
    /// Creates an intense North Atlantic low:
    /// - Central pressure: 950 hPa
    /// - Radius of max winds: 150 km
    /// - Holland B = 1.5
    ///
    /// # Arguments
    /// * `x0`, `y0` - Initial storm center position (m)
    pub fn severe_storm(x0: f64, y0: f64) -> Self {
        Self::moving_storm(
            move |t| {
                let speed = 20.0;
                let angle = 50.0_f64.to_radians();
                (x0 + speed * angle.cos() * t, y0 + speed * angle.sin() * t)
            },
            950.0,
            150.0,
            1.5,
        )
    }

    /// Set water density (default: 1025 kg/m³).
    pub fn with_rho_water(mut self, rho: f64) -> Self {
        self.rho_water = rho;
        self
    }

    /// Set minimum water depth for forcing (default: 0.01 m).
    pub fn with_h_min(mut self, h_min: f64) -> Self {
        self.h_min = h_min;
        self
    }

    /// Set finite difference step for gradient computation (default: 100 m).
    pub fn with_grad_delta(mut self, delta: f64) -> Self {
        self.grad_delta = delta;
        self
    }

    /// Compute pressure at a point.
    fn pressure_at(&self, x: f64, y: f64, t: f64) -> f64 {
        match &self.field {
            PressureField::Constant(p) => *p,
            PressureField::UniformGradient {
                p_ref,
                dp_dx,
                dp_dy,
            } => p_ref + dp_dx * x + dp_dy * y,
            PressureField::TimeVaryingGradient(f) => {
                let (dp_dx, dp_dy) = f(t);
                P_STANDARD + dp_dx * x + dp_dy * y
            }
            PressureField::SpatioTemporal(f) => f(x, y, t),
            PressureField::MovingStorm {
                track,
                p_center,
                p_ambient,
                r_max,
                b_param,
            } => {
                let (xc, yc) = track(t);
                let r = ((x - xc).powi(2) + (y - yc).powi(2)).sqrt();
                if r < 1.0 {
                    *p_center
                } else {
                    let ratio = r_max / r;
                    p_center + (p_ambient - p_center) * (-ratio.powf(*b_param)).exp()
                }
            }
        }
    }

    /// Compute pressure gradient at a point.
    fn gradient_at(&self, x: f64, y: f64, t: f64) -> (f64, f64) {
        match &self.field {
            PressureField::Constant(_) => (0.0, 0.0),
            PressureField::UniformGradient { dp_dx, dp_dy, .. } => (*dp_dx, *dp_dy),
            PressureField::TimeVaryingGradient(f) => f(t),
            PressureField::SpatioTemporal(_) | PressureField::MovingStorm { .. } => {
                // Numerical gradient using central differences
                let dx = self.grad_delta;
                let dy = self.grad_delta;
                let dp_dx =
                    (self.pressure_at(x + dx, y, t) - self.pressure_at(x - dx, y, t)) / (2.0 * dx);
                let dp_dy =
                    (self.pressure_at(x, y + dy, t) - self.pressure_at(x, y - dy, t)) / (2.0 * dy);
                (dp_dx, dp_dy)
            }
        }
    }

    /// Compute inverse barometer sea level adjustment.
    ///
    /// Returns the sea level change (m) due to atmospheric pressure difference
    /// from standard pressure.
    ///
    /// ```text
    /// Δη = -(P - P_standard) / (ρg)
    /// ```
    ///
    /// This is useful for setting initial conditions or boundary conditions,
    /// not as a dynamic source term.
    pub fn inverse_barometer(&self, x: f64, y: f64, t: f64) -> f64 {
        let p = self.pressure_at(x, y, t);
        -(p - P_STANDARD) / (self.rho_water * G)
    }
}

impl SourceTerm2D for AtmosphericPressure2D {
    fn evaluate(&self, ctx: &SourceContext2D) -> SWEState2D {
        // No forcing in very shallow water
        if ctx.state.h < self.h_min {
            return SWEState2D::zero();
        }

        let (x, y) = ctx.position;
        let (dp_dx, dp_dy) = self.gradient_at(x, y, ctx.time);

        // Source terms: S_hu = -h/ρ * ∂P/∂x, S_hv = -h/ρ * ∂P/∂y
        let factor = -ctx.state.h / self.rho_water;
        let s_hu = factor * dp_dx;
        let s_hv = factor * dp_dy;

        SWEState2D::new(0.0, s_hu, s_hv)
    }

    fn name(&self) -> &'static str {
        "atmospheric_pressure_2d"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create a test context with default values.
    fn test_ctx(h: f64, position: (f64, f64), time: f64) -> SourceContext2D {
        SourceContext2D {
            time,
            position,
            state: SWEState2D::new(h, 0.0, 0.0),
            bathymetry: 50.0,
            bathymetry_gradient: (0.0, 0.0),
            g: G,
            h_min: 0.01,
        }
    }

    #[test]
    fn test_constant_pressure_no_forcing() {
        let pressure = AtmosphericPressure2D::constant(P_STANDARD);
        let ctx = test_ctx(10.0, (1000.0, 2000.0), 0.0);

        let source = pressure.evaluate(&ctx);
        assert!(source.h.abs() < 1e-14);
        assert!(source.hu.abs() < 1e-14);
        assert!(source.hv.abs() < 1e-14);
    }

    #[test]
    fn test_uniform_gradient_forcing() {
        // 1 hPa per 100 km = 0.001 Pa/m in x direction
        let dp_dx = 0.001;
        let pressure = AtmosphericPressure2D::uniform_gradient(dp_dx, 0.0);

        let h = 10.0;
        let ctx = test_ctx(h, (0.0, 0.0), 0.0);

        let source = pressure.evaluate(&ctx);

        // Expected: S_hu = -h/ρ * dp_dx = -10/1025 * 0.001 ≈ -9.76e-6 m²/s²
        let expected_hu = -h / RHO_WATER_PRESSURE * dp_dx;
        assert!(source.h.abs() < 1e-14);
        assert!((source.hu - expected_hu).abs() < 1e-12);
        assert!(source.hv.abs() < 1e-14);
    }

    #[test]
    fn test_from_direction() {
        // Gradient from south (180°) means pressure increases northward
        let mag = 0.001;
        let pressure = AtmosphericPressure2D::from_direction(mag, 180.0);

        let ctx = test_ctx(10.0, (0.0, 0.0), 0.0);

        let source = pressure.evaluate(&ctx);

        // From south means dp_dy > 0 (pressure increases northward)
        // S_hv = -h/ρ * dp_dy < 0 (force pushes southward against gradient)
        assert!(source.hu.abs() < 1e-10); // No x-gradient
        assert!(source.hv < 0.0); // Force is southward
    }

    #[test]
    fn test_time_varying_gradient() {
        let pressure = AtmosphericPressure2D::time_varying_gradient(|t| {
            // Gradient rotates
            let omega = 0.1;
            (0.001 * (omega * t).cos(), 0.001 * (omega * t).sin())
        });

        let ctx_t0 = test_ctx(10.0, (0.0, 0.0), 0.0);

        let source_t0 = pressure.evaluate(&ctx_t0);
        // At t=0, gradient is (0.001, 0), so S_hu < 0, S_hv = 0
        assert!(source_t0.hu < 0.0);
        assert!(source_t0.hv.abs() < 1e-12);

        let ctx_t_quarter = test_ctx(10.0, (0.0, 0.0), std::f64::consts::PI / 2.0 / 0.1);

        let source_t_quarter = pressure.evaluate(&ctx_t_quarter);
        // At t=π/2/ω, gradient is (0, 0.001), so S_hu = 0, S_hv < 0
        assert!(source_t_quarter.hu.abs() < 1e-12);
        assert!(source_t_quarter.hv < 0.0);
    }

    #[test]
    fn test_spatio_temporal_field() {
        // Gaussian pressure low centered at (1000, 1000)
        let pressure = AtmosphericPressure2D::spatio_temporal(|x, y, _t| {
            let r2 = (x - 1000.0).powi(2) + (y - 1000.0).powi(2);
            let r_scale: f64 = 500.0;
            let p_drop: f64 = 1000.0; // 10 hPa drop
            P_STANDARD - p_drop * (-r2 / r_scale.powi(2)).exp()
        });

        // At the center, gradient should be near zero
        let ctx_center = test_ctx(10.0, (1000.0, 1000.0), 0.0);

        let source_center = pressure.evaluate(&ctx_center);
        assert!(source_center.hu.abs() < 1e-10);
        assert!(source_center.hv.abs() < 1e-10);

        // East of center, pressure increases eastward (away from low)
        // So dp_dx > 0, and S_hu = -h/ρ * dp_dx < 0 (force toward low = westward)
        let ctx_east = test_ctx(10.0, (1500.0, 1000.0), 0.0);

        let source_east = pressure.evaluate(&ctx_east);
        assert!(source_east.hu < 0.0); // Force points west toward low
        assert!(source_east.hv.abs() < 1e-10);
    }

    #[test]
    fn test_holland_storm() {
        // Stationary storm at origin
        let storm = AtmosphericPressure2D::moving_storm(
            |_t| (0.0, 0.0),
            970.0, // Central pressure 970 hPa
            100.0, // R_max = 100 km
            1.5,   // Holland B
        );

        // At the center, pressure should be central pressure
        let p_center = storm.pressure_at(0.0, 0.0, 0.0);
        assert!((p_center - 97000.0).abs() < 1.0);

        // Far away (1000km), pressure should approach ambient
        // With Holland model, pressure recovers slowly; at r=10*r_max, expect ~97% recovery
        let p_far = storm.pressure_at(1e6, 0.0, 0.0);
        assert!((p_far - P_STANDARD).abs() < 200.0); // Within 2 hPa of ambient

        // At r_max, pressure gradient should be strongest
        // East of center: dp_dx > 0, S_hu = -h/ρ * dp_dx < 0 (force toward center = west)
        let ctx = test_ctx(10.0, (100000.0, 0.0), 0.0);

        let source = storm.evaluate(&ctx);
        assert!(source.hu < 0.0); // Force toward center (westward)
        assert!(source.hv.abs() < 1e-10);
    }

    #[test]
    fn test_moving_storm() {
        // Storm moving east at 100 m/s (fast-moving for test)
        // Use small storm (r_max=10km) so it moves out of range quickly
        let storm = AtmosphericPressure2D::moving_storm(|t| (100.0 * t, 0.0), 970.0, 10.0, 1.5);

        // At t=0, storm is at origin
        let p_origin_t0 = storm.pressure_at(0.0, 0.0, 0.0);
        assert!((p_origin_t0 - 97000.0).abs() < 1.0);

        // At t=1000s, storm has moved to x=100000 (100km east)
        let p_100km_t1000 = storm.pressure_at(100000.0, 0.0, 1000.0);
        assert!((p_100km_t1000 - 97000.0).abs() < 1.0);

        // Origin is now 100km from storm center (r_max=10km)
        // Pressure should have significantly recovered
        let p_origin_t1000 = storm.pressure_at(0.0, 0.0, 1000.0);
        assert!(p_origin_t1000 > 100000.0); // Pressure recovered toward ambient
    }

    #[test]
    fn test_inverse_barometer() {
        let pressure = AtmosphericPressure2D::constant(100325.0); // 10 hPa below standard

        let ib = pressure.inverse_barometer(0.0, 0.0, 0.0);

        // Δη = -(P - P_standard) / (ρg) = -(-1000) / (1025 * 9.81) ≈ 0.0995 m ≈ 10 cm
        let expected = 1000.0 / (RHO_WATER_PRESSURE * G);
        assert!((ib - expected).abs() < 0.001);
    }

    #[test]
    fn test_inverse_barometer_low_pressure() {
        // Deep low pressure: 960 hPa
        let pressure = AtmosphericPressure2D::constant(96000.0);

        let ib = pressure.inverse_barometer(0.0, 0.0, 0.0);

        // Δη = -(96000 - 101325) / (1025 * 9.81) ≈ 0.53 m
        // Low pressure causes sea level RISE (positive)
        assert!(ib > 0.0);
        assert!((ib - 0.53).abs() < 0.01);
    }

    #[test]
    fn test_shallow_water_no_forcing() {
        let pressure = AtmosphericPressure2D::uniform_gradient(0.001, 0.0);

        let ctx = test_ctx(0.005, (0.0, 0.0), 0.0); // Below h_min

        let source = pressure.evaluate(&ctx);
        assert!(source.hu.abs() < 1e-14);
        assert!(source.hv.abs() < 1e-14);
    }

    #[test]
    fn test_norwegian_presets() {
        let storm = AtmosphericPressure2D::norwegian_winter_storm(0.0, 0.0);

        // Storm center at t=0 should be at origin
        let p_center = storm.pressure_at(0.0, 0.0, 0.0);
        assert!((p_center - 97000.0).abs() < 1.0);

        let severe = AtmosphericPressure2D::severe_storm(0.0, 0.0);
        let p_severe = severe.pressure_at(0.0, 0.0, 0.0);
        assert!((p_severe - 95000.0).abs() < 1.0);
    }

    #[test]
    fn test_force_direction_toward_low() {
        // Physical test: Force should push water toward low pressure
        // This is the geostrophic setup (ignoring Coriolis)

        // Low pressure at x=0, high pressure at x=10000
        let pressure = AtmosphericPressure2D::uniform_gradient(0.001, 0.0);

        // Water at x=5000 should experience westward force (toward low)
        let ctx = test_ctx(10.0, (5000.0, 0.0), 0.0);

        let source = pressure.evaluate(&ctx);

        // dp_dx > 0 means pressure increases to east
        // Force = -∇P/ρ points west (toward low)
        // S_hu = -h/ρ * dp_dx < 0
        assert!(source.hu < 0.0);
    }

    #[test]
    fn test_typical_storm_surge_forcing() {
        // Sanity check: typical Norwegian storm surge forcing magnitude

        // Strong gradient: 3 hPa per 100 km (severe storm)
        let dp_dx = 3.0e-3;
        let pressure = AtmosphericPressure2D::uniform_gradient(dp_dx, 0.0);

        let h = 100.0; // 100m deep water
        let ctx = test_ctx(h, (0.0, 0.0), 0.0);

        let source = pressure.evaluate(&ctx);

        // S_hu = -h/ρ * dp_dx = -100/1025 * 0.003 ≈ -2.9e-4 m²/s²
        // This gives acceleration a = S_hu/h = -2.9e-6 m/s²
        // Over 1 hour: Δu ≈ 0.01 m/s
        // Over 12 hours: Δu ≈ 0.12 m/s (reasonable storm surge velocity)
        let accel = source.hu / h;
        assert!(accel.abs() > 1e-7);
        assert!(accel.abs() < 1e-3);
    }
}

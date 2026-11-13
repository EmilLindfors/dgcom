//! Wind stress forcing for 2D shallow water equations.
//!
//! Wind stress is a major driver of surface currents, especially important for:
//! - Coastal upwelling/downwelling
//! - Storm surge
//! - Wind-driven circulation in fjords
//!
//! # Physics
//!
//! Wind stress τ = ρ_air * C_d * |U_10| * U_10
//!
//! where:
//! - ρ_air ≈ 1.225 kg/m³ (air density at sea level)
//! - C_d = drag coefficient (depends on wind speed)
//! - U_10 = (u_10, v_10) = 10m wind velocity
//!
//! The source term in the momentum equations:
//! - S_hu = τ_x / ρ_water
//! - S_hv = τ_y / ρ_water
//!
//! # Drag Coefficient Formulations
//!
//! Several formulations are available:
//! - **Constant**: Simple fixed value (typically 1.0-1.5 × 10⁻³)
//! - **Large & Pond (1981)**: C_d = 1.2×10⁻³ for |U| < 11 m/s, increases linearly above
//! - **Wu (1982)**: C_d = (0.8 + 0.065×|U|) × 10⁻³
//! - **COARE 3.0**: More accurate for operational use
//!
//! # Example
//!
//! ```
//! use dgcom::source::{WindStress2D, DragCoefficient};
//!
//! // Constant 10 m/s wind from the west
//! let wind = WindStress2D::constant(10.0, 0.0);
//!
//! // Time-varying wind with Large-Pond drag
//! let wind = WindStress2D::time_varying(
//!     |t| (10.0 * (t / 3600.0).sin(), 5.0),  // u_10, v_10 as function of time
//!     DragCoefficient::LargePond,
//! );
//! ```
//!
//! # Norwegian Coast
//!
//! Typical conditions:
//! - Winter storms: 15-25 m/s from SW-W
//! - Summer: 5-10 m/s variable
//! - Coastal jets in fjords can be stronger

use crate::solver::SWEState2D;
use crate::source::{SourceContext2D, SourceTerm2D};
use std::f64::consts::PI;

/// Air density at sea level (kg/m³).
pub const RHO_AIR: f64 = 1.225;

/// Sea water density (kg/m³).
pub const RHO_WATER: f64 = 1025.0;

/// Wind speed (m/s) above which [`DragCoefficient::LargePond`] is held constant.
pub const LARGE_POND_MAX_WIND: f64 = 25.0;

/// Drag coefficient formulation for wind stress.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum DragCoefficient {
    /// Constant drag coefficient.
    ///
    /// Simple but less accurate. Typical values: 1.0-2.0 × 10⁻³
    Constant(f64),

    /// Large & Pond (1981) formulation.
    ///
    /// C_d = 1.2×10⁻³ for |U| ≤ 11 m/s
    /// C_d = (0.49 + 0.065×|U|) × 10⁻³ for 11 < |U| ≤ 25 m/s
    ///
    /// Fitted for |U| < 25 m/s; above that the formula is held at its 25 m/s
    /// value (2.1 × 10⁻³) instead of growing without bound: observed drag
    /// saturates in strong winds (Powell et al. 2003).
    #[default]
    LargePond,

    /// Wu (1982) formulation.
    ///
    /// C_d = (0.8 + 0.065×|U|) × 10⁻³
    ///
    /// Simple linear relationship, widely used.
    Wu,

    /// Smith (1988) formulation.
    ///
    /// C_d = (0.61 + 0.063×|U|) × 10⁻³
    ///
    /// Similar to Wu but with different coefficients.
    Smith,

    /// Yelland & Taylor (1996) for open ocean.
    ///
    /// C_d = (0.50 + 0.071×|U|) × 10⁻³ for |U| > 6 m/s
    /// C_d = 1.1×10⁻³ for |U| ≤ 6 m/s
    YellandTaylor,
}

impl DragCoefficient {
    /// Compute drag coefficient for given wind speed.
    ///
    /// # Arguments
    /// * `wind_speed` - 10m wind speed magnitude (m/s)
    pub fn compute(&self, wind_speed: f64) -> f64 {
        match self {
            DragCoefficient::Constant(cd) => *cd,

            DragCoefficient::LargePond => {
                if wind_speed <= 11.0 {
                    1.2e-3
                } else {
                    (0.49 + 0.065 * wind_speed.min(LARGE_POND_MAX_WIND)) * 1e-3
                }
            }

            DragCoefficient::Wu => (0.8 + 0.065 * wind_speed) * 1e-3,

            DragCoefficient::Smith => (0.61 + 0.063 * wind_speed) * 1e-3,

            DragCoefficient::YellandTaylor => {
                if wind_speed <= 6.0 {
                    1.1e-3
                } else {
                    (0.50 + 0.071 * wind_speed) * 1e-3
                }
            }
        }
    }
}

/// Wind field specification.
enum WindField {
    /// Constant wind (u_10, v_10) in m/s
    Constant(f64, f64),

    /// Time-varying wind: (u_10, v_10) = f(t)
    TimeVarying(Box<dyn Fn(f64) -> (f64, f64) + Send + Sync>),

    /// Spatially and temporally varying: (u_10, v_10) = f(x, y, t)
    SpatioTemporal(Box<dyn Fn(f64, f64, f64) -> (f64, f64) + Send + Sync>),
}

impl std::fmt::Debug for WindField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WindField::Constant(u, v) => write!(f, "Constant({}, {})", u, v),
            WindField::TimeVarying(_) => write!(f, "TimeVarying(...)"),
            WindField::SpatioTemporal(_) => write!(f, "SpatioTemporal(...)"),
        }
    }
}

/// Wind stress source term for 2D shallow water equations.
///
/// Adds momentum source from wind drag on the water surface:
/// - S_hu = τ_x / ρ_water
/// - S_hv = τ_y / ρ_water
///
/// where τ = ρ_air × C_d × |U_10| × U_10
#[derive(Debug)]
pub struct WindStress2D {
    /// Wind field specification
    wind: WindField,
    /// Drag coefficient formulation
    drag: DragCoefficient,
    /// Air density (kg/m³)
    rho_air: f64,
    /// Water density (kg/m³)
    rho_water: f64,
    /// Minimum depth for applying wind stress
    h_min: f64,
}

impl WindStress2D {
    /// Create wind stress with constant wind.
    ///
    /// # Arguments
    /// * `u_10` - 10m eastward wind component (m/s)
    /// * `v_10` - 10m northward wind component (m/s)
    ///
    /// Uses Large & Pond drag coefficient by default.
    pub fn constant(u_10: f64, v_10: f64) -> Self {
        Self {
            wind: WindField::Constant(u_10, v_10),
            drag: DragCoefficient::LargePond,
            rho_air: RHO_AIR,
            rho_water: RHO_WATER,
            h_min: 0.01,
        }
    }

    /// Create wind stress with constant wind from direction.
    ///
    /// # Arguments
    /// * `speed` - Wind speed (m/s)
    /// * `direction` - Wind direction in degrees (meteorological convention:
    ///   0° = from north, 90° = from east)
    pub fn from_direction(speed: f64, direction: f64) -> Self {
        // Meteorological convention: direction wind is coming FROM
        // Convert to components: direction 0 = from N means wind blows southward
        let dir_rad = direction * PI / 180.0;
        let u_10 = -speed * dir_rad.sin();
        let v_10 = -speed * dir_rad.cos();
        Self::constant(u_10, v_10)
    }

    /// Create wind stress with time-varying wind.
    ///
    /// # Arguments
    /// * `wind_fn` - Function returning (u_10, v_10) given time t
    /// * `drag` - Drag coefficient formulation
    pub fn time_varying<F>(wind_fn: F, drag: DragCoefficient) -> Self
    where
        F: Fn(f64) -> (f64, f64) + Send + Sync + 'static,
    {
        Self {
            wind: WindField::TimeVarying(Box::new(wind_fn)),
            drag,
            rho_air: RHO_AIR,
            rho_water: RHO_WATER,
            h_min: 0.01,
        }
    }

    /// Create wind stress with spatially and temporally varying wind.
    ///
    /// # Arguments
    /// * `wind_fn` - Function returning (u_10, v_10) given (x, y, t)
    /// * `drag` - Drag coefficient formulation
    pub fn spatio_temporal<F>(wind_fn: F, drag: DragCoefficient) -> Self
    where
        F: Fn(f64, f64, f64) -> (f64, f64) + Send + Sync + 'static,
    {
        Self {
            wind: WindField::SpatioTemporal(Box::new(wind_fn)),
            drag,
            rho_air: RHO_AIR,
            rho_water: RHO_WATER,
            h_min: 0.01,
        }
    }

    /// Set drag coefficient formulation.
    pub fn with_drag(mut self, drag: DragCoefficient) -> Self {
        self.drag = drag;
        self
    }

    /// Set air density (default: 1.225 kg/m³).
    pub fn with_rho_air(mut self, rho_air: f64) -> Self {
        self.rho_air = rho_air;
        self
    }

    /// Set water density (default: 1025 kg/m³).
    pub fn with_rho_water(mut self, rho_water: f64) -> Self {
        self.rho_water = rho_water;
        self
    }

    /// Set minimum depth for wind stress application.
    pub fn with_h_min(mut self, h_min: f64) -> Self {
        self.h_min = h_min;
        self
    }

    /// Create a diurnal wind pattern (sea breeze cycle).
    ///
    /// Wind varies sinusoidally over 24 hours:
    /// - Maximum onshore (from sea) at solar noon
    /// - Calm or light offshore at night
    ///
    /// # Arguments
    /// * `max_speed` - Maximum wind speed (m/s)
    /// * `onshore_direction` - Direction of onshore wind (degrees, from which direction)
    /// * `period` - Period in seconds (default 86400 for 24h)
    pub fn sea_breeze(max_speed: f64, onshore_direction: f64, period: f64) -> Self {
        let dir_rad = onshore_direction * PI / 180.0;
        Self::time_varying(
            move |t| {
                // Peak at t=period/4 (solar noon), minimum at 3*period/4 (night)
                let phase = 2.0 * PI * t / period - PI / 2.0;
                let speed = max_speed * 0.5 * (1.0 + phase.cos());
                let u = -speed * dir_rad.sin();
                let v = -speed * dir_rad.cos();
                (u, v)
            },
            DragCoefficient::LargePond,
        )
    }

    /// Create typical Norwegian coast winter storm.
    ///
    /// Strong southwesterly wind (15 m/s from 225°).
    pub fn norwegian_winter_storm() -> Self {
        Self::from_direction(15.0, 225.0)
    }

    /// Create typical Norwegian coast summer breeze.
    ///
    /// Light variable wind (5 m/s from west).
    pub fn norwegian_summer() -> Self {
        Self::from_direction(5.0, 270.0)
    }

    /// Get wind velocity at given position and time.
    fn wind_at(&self, x: f64, y: f64, t: f64) -> (f64, f64) {
        match &self.wind {
            WindField::Constant(u, v) => (*u, *v),
            WindField::TimeVarying(f) => f(t),
            WindField::SpatioTemporal(f) => f(x, y, t),
        }
    }

    /// Compute wind stress components.
    ///
    /// Returns (τ_x, τ_y) in N/m².
    pub fn compute_stress(&self, u_10: f64, v_10: f64) -> (f64, f64) {
        let wind_speed = (u_10 * u_10 + v_10 * v_10).sqrt();
        let cd = self.drag.compute(wind_speed);

        // τ = ρ_air × C_d × |U| × U
        let tau_x = self.rho_air * cd * wind_speed * u_10;
        let tau_y = self.rho_air * cd * wind_speed * v_10;

        (tau_x, tau_y)
    }
}

impl SourceTerm2D for WindStress2D {
    fn evaluate(&self, ctx: &SourceContext2D) -> SWEState2D {
        // Don't apply wind to dry cells
        if ctx.state.h < self.h_min {
            return SWEState2D::zero();
        }

        let (x, y) = ctx.position;
        let (u_10, v_10) = self.wind_at(x, y, ctx.time);
        let (tau_x, tau_y) = self.compute_stress(u_10, v_10);

        // Source terms: S_hu = τ_x / ρ_water, S_hv = τ_y / ρ_water
        SWEState2D::new(0.0, tau_x / self.rho_water, tau_y / self.rho_water)
    }

    fn is_node_local(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "wind_stress_2d"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    fn make_context(h: f64) -> SourceContext2D {
        SourceContext2D::new(
            0.0,
            (0.0, 0.0),
            SWEState2D::new(h, 0.0, 0.0),
            0.0,
            (0.0, 0.0),
            9.81,
            1e-6,
        )
    }

    #[test]
    fn test_drag_coefficient_constant() {
        let cd = DragCoefficient::Constant(1.5e-3);
        assert!((cd.compute(5.0) - 1.5e-3).abs() < TOL);
        assert!((cd.compute(20.0) - 1.5e-3).abs() < TOL);
    }

    #[test]
    fn test_drag_coefficient_large_pond() {
        let cd = DragCoefficient::LargePond;

        // Below 11 m/s: constant 1.2e-3
        assert!((cd.compute(5.0) - 1.2e-3).abs() < TOL);
        assert!((cd.compute(11.0) - 1.2e-3).abs() < TOL);

        // Above 11 m/s: linear increase
        // C_d = (0.49 + 0.065 * 15) * 1e-3 = 1.465e-3
        assert!((cd.compute(15.0) - 1.465e-3).abs() < TOL);

        // Held at the 25 m/s value beyond the fitted range
        let cd_max = (0.49 + 0.065 * 25.0) * 1e-3;
        assert!((cd.compute(25.0) - cd_max).abs() < TOL);
        assert!((cd.compute(40.0) - cd_max).abs() < TOL);
    }

    #[test]
    fn test_drag_coefficient_wu() {
        let cd = DragCoefficient::Wu;

        // C_d = (0.8 + 0.065 * U) * 1e-3
        assert!((cd.compute(10.0) - 1.45e-3).abs() < TOL);
        assert!((cd.compute(20.0) - 2.1e-3).abs() < TOL);
    }

    #[test]
    fn test_wind_stress_constant() {
        let wind = WindStress2D::constant(10.0, 0.0);
        let ctx = make_context(10.0);
        let s = wind.evaluate(&ctx);

        // τ_x = ρ_air × C_d × |U| × u = 1.225 × 1.2e-3 × 10 × 10 = 0.147
        // S_hu = τ_x / ρ_water = 0.147 / 1025 ≈ 1.43e-4
        assert!(s.h.abs() < TOL);
        assert!((s.hu - 1.435e-4).abs() < 1e-6);
        assert!(s.hv.abs() < TOL);
    }

    #[test]
    fn test_wind_stress_diagonal() {
        // 45-degree wind: u=v=10/sqrt(2) ≈ 7.07 m/s
        let speed = 10.0;
        let u = speed / 2.0_f64.sqrt();
        let v = speed / 2.0_f64.sqrt();
        let wind = WindStress2D::constant(u, v);
        let ctx = make_context(10.0);
        let s = wind.evaluate(&ctx);

        // Should have equal x and y components
        assert!((s.hu - s.hv).abs() < 1e-10);
    }

    #[test]
    fn test_wind_stress_dry_cell() {
        let wind = WindStress2D::constant(20.0, 10.0);
        let ctx = make_context(0.001); // Very shallow
        let s = wind.evaluate(&ctx);

        // No wind stress on dry cells
        assert!(s.hu.abs() < TOL);
        assert!(s.hv.abs() < TOL);
    }

    #[test]
    fn test_wind_from_direction() {
        // Wind from the north (direction=0) means flow southward (v<0)
        let wind = WindStress2D::from_direction(10.0, 0.0);
        let ctx = make_context(10.0);
        let s = wind.evaluate(&ctx);

        assert!(s.hu.abs() < TOL); // No eastward component
        assert!(s.hv < 0.0); // Southward flow
    }

    #[test]
    fn test_wind_from_west() {
        // Wind from the west (direction=270) means flow eastward (u>0)
        let wind = WindStress2D::from_direction(10.0, 270.0);
        let ctx = make_context(10.0);
        let s = wind.evaluate(&ctx);

        assert!(s.hu > 0.0); // Eastward flow
        assert!(s.hv.abs() < 1e-10); // No northward component
    }

    #[test]
    fn test_time_varying_wind() {
        // Sinusoidal wind
        let wind = WindStress2D::time_varying(
            |t| (10.0 * (t * 2.0 * PI).sin(), 0.0),
            DragCoefficient::LargePond,
        );

        // At t=0: u=0
        let ctx0 = SourceContext2D::new(
            0.0,
            (0.0, 0.0),
            SWEState2D::new(10.0, 0.0, 0.0),
            0.0,
            (0.0, 0.0),
            9.81,
            1e-6,
        );
        let s0 = wind.evaluate(&ctx0);
        assert!(s0.hu.abs() < TOL);

        // At t=0.25: u=10
        let ctx1 = SourceContext2D::new(
            0.25,
            (0.0, 0.0),
            SWEState2D::new(10.0, 0.0, 0.0),
            0.0,
            (0.0, 0.0),
            9.81,
            1e-6,
        );
        let s1 = wind.evaluate(&ctx1);
        assert!(s1.hu > 0.0);
    }

    #[test]
    fn test_compute_stress() {
        let wind = WindStress2D::constant(10.0, 0.0);

        let (tau_x, tau_y) = wind.compute_stress(10.0, 0.0);

        // τ = ρ_air × C_d × |U| × U
        // C_d for 10 m/s (Large-Pond) = 1.2e-3
        // τ_x = 1.225 × 1.2e-3 × 10 × 10 = 0.147 N/m²
        assert!((tau_x - 0.147).abs() < 0.01);
        assert!(tau_y.abs() < TOL);
    }

    #[test]
    fn test_norwegian_presets() {
        // Winter storm: 15 m/s from SW (225°)
        let winter = WindStress2D::norwegian_winter_storm();
        let ctx = make_context(50.0);
        let s = winter.evaluate(&ctx);

        // From SW means flow to NE: u>0, v>0
        assert!(s.hu > 0.0);
        assert!(s.hv > 0.0);

        // Summer breeze: 5 m/s from W (270°)
        let summer = WindStress2D::norwegian_summer();
        let s_summer = summer.evaluate(&ctx);

        // From W means flow to E: u>0, v≈0
        assert!(s_summer.hu > 0.0);
        assert!(s_summer.hv.abs() < 1e-10);
    }

    #[test]
    fn test_stress_increases_with_speed() {
        let ctx = make_context(10.0);

        let wind_5 = WindStress2D::constant(5.0, 0.0);
        let wind_10 = WindStress2D::constant(10.0, 0.0);
        let wind_20 = WindStress2D::constant(20.0, 0.0);

        let s5 = wind_5.evaluate(&ctx);
        let s10 = wind_10.evaluate(&ctx);
        let s20 = wind_20.evaluate(&ctx);

        // Stress increases nonlinearly with speed (τ ∝ U²)
        assert!(s10.hu > s5.hu);
        assert!(s20.hu > s10.hu);
        assert!(s20.hu > 2.0 * s10.hu); // More than linear due to U² dependence
    }

    #[test]
    fn test_custom_densities() {
        let wind = WindStress2D::constant(10.0, 0.0)
            .with_rho_air(1.0)
            .with_rho_water(1000.0);

        let ctx = make_context(10.0);
        let s = wind.evaluate(&ctx);

        // Lower air density = less stress
        // Different water density changes the source magnitude
        assert!(s.hu > 0.0);
    }

    #[test]
    fn test_sea_breeze() {
        let breeze = WindStress2D::sea_breeze(10.0, 270.0, 86400.0); // From west

        // At t=0 (midnight-ish, phase=-π/2): speed = 0.5 * (1 + 0) = 0.5 * max
        let ctx0 = SourceContext2D::new(
            0.0,
            (0.0, 0.0),
            SWEState2D::new(10.0, 0.0, 0.0),
            0.0,
            (0.0, 0.0),
            9.81,
            1e-6,
        );

        // At t=period/4 (solar noon): speed = max
        let ctx_noon = SourceContext2D::new(
            21600.0,
            (0.0, 0.0),
            SWEState2D::new(10.0, 0.0, 0.0),
            0.0,
            (0.0, 0.0),
            9.81,
            1e-6,
        );

        let s0 = breeze.evaluate(&ctx0);
        let s_noon = breeze.evaluate(&ctx_noon);

        // Noon should have stronger wind than midnight
        assert!(s_noon.hu.abs() > s0.hu.abs());
    }
}

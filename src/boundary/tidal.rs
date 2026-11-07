//! Tidal boundary conditions for shallow water equations.
//!
//! Tidal boundaries prescribe time-varying water surface elevation based on
//! harmonic constituents (M2, S2, K1, O1, etc.). These are essential for
//! coastal ocean modeling.
//!
//! The tidal elevation is computed as:
//! η(t) = η₀ + Σᵢ Aᵢ cos(ωᵢ t + φᵢ)
//!
//! where Aᵢ is amplitude, ωᵢ is angular frequency, and φᵢ is phase.

use super::{BCContext, SWEBoundaryCondition};
use crate::solver::SWEState;
use std::f64::consts::PI;

/// Tidal constituent data.
///
/// Represents a single harmonic component of the tide.
#[derive(Clone, Debug)]
pub struct TidalConstituent {
    /// Name of the constituent (e.g., "M2", "S2", "K1")
    pub name: &'static str,
    /// Amplitude (meters)
    pub amplitude: f64,
    /// Period (seconds)
    pub period: f64,
    /// Phase (radians)
    pub phase: f64,
}

impl TidalConstituent {
    /// Create a new tidal constituent.
    pub fn new(name: &'static str, amplitude: f64, period: f64, phase: f64) -> Self {
        Self {
            name,
            amplitude,
            period,
            phase,
        }
    }

    /// Angular frequency ω = 2π/T.
    pub fn angular_frequency(&self) -> f64 {
        2.0 * PI / self.period
    }

    /// Evaluate the constituent at time t.
    pub fn evaluate(&self, t: f64) -> f64 {
        let omega = self.angular_frequency();
        self.amplitude * (omega * t + self.phase).cos()
    }

    /// Return a copy with a nodal/equilibrium correction applied.
    ///
    /// The amplitude is scaled by the nodal factor `f` and the internal phase is
    /// shifted by the equilibrium argument plus nodal phase `(V₀ + u)`, so that
    /// `evaluate(t)` becomes `f·A·cos(ω t + φ + V₀ + u)` — the Schureman-corrected
    /// prediction for elapsed time `t` measured from the correction's epoch.
    ///
    /// Apply this **once**, to a raw catalogue constituent; applying it twice
    /// double-counts the correction. See [`crate::tides`].
    pub fn with_nodal_correction(&self, correction: &crate::tides::NodalCorrection) -> Self {
        let (amplitude, phase) =
            crate::tides::correct_amplitude_phase(self.amplitude, self.phase, correction);
        Self {
            name: self.name,
            amplitude,
            period: self.period,
            phase,
        }
    }

    /// Principal lunar semidiurnal (M2) constituent.
    ///
    /// Period 12.4206 h (28.9841042°/h), the dominant tidal constituent in most locations.
    pub fn m2(amplitude: f64, phase: f64) -> Self {
        Self::new("M2", amplitude, doodson_period("M2"), phase)
    }

    /// Principal solar semidiurnal (S2) constituent.
    ///
    /// Period = 12 hours exactly (30°/h).
    pub fn s2(amplitude: f64, phase: f64) -> Self {
        Self::new("S2", amplitude, doodson_period("S2"), phase)
    }

    /// Lunar diurnal (K1) constituent.
    ///
    /// Period 23.9345 h (15.0410686°/h).
    pub fn k1(amplitude: f64, phase: f64) -> Self {
        Self::new("K1", amplitude, doodson_period("K1"), phase)
    }

    /// Lunar diurnal (O1) constituent.
    ///
    /// Period 25.8193 h (13.9430356°/h).
    pub fn o1(amplitude: f64, phase: f64) -> Self {
        Self::new("O1", amplitude, doodson_period("O1"), phase)
    }

    /// Larger lunar elliptic semidiurnal (N2) constituent.
    ///
    /// Period 12.6583 h (28.4397295°/h).
    pub fn n2(amplitude: f64, phase: f64) -> Self {
        Self::new("N2", amplitude, doodson_period("N2"), phase)
    }

    /// Lunisolar diurnal (P1) constituent.
    ///
    /// Period 24.0659 h (14.9589314°/h).
    pub fn p1(amplitude: f64, phase: f64) -> Self {
        Self::new("P1", amplitude, doodson_period("P1"), phase)
    }
}

/// Exact period (s) of a named constituent from its Doodson angular speed.
///
/// Rounded periods are not good enough for harmonic work: 12.42 h for M2
/// accumulates ≈ 1° of phase error over 60 days and ≈ 6° over a year.
fn doodson_period(name: &str) -> f64 {
    crate::tides::constituent_period(name).expect("constituent is in the tides catalogue")
}

/// Tidal boundary condition.
///
/// Prescribes water surface elevation as a sum of tidal constituents.
/// Uses a Flather-type radiation condition to allow outgoing waves to exit.
///
/// η(t) = η₀ + R(t) × Σᵢ Aᵢ cos(ωᵢ t + φᵢ)
///
/// where R(t) is a smooth ramp function (0 at t=0, 1 after ramp_duration).
#[derive(Clone, Debug)]
pub struct TidalBC {
    /// Mean water surface elevation
    pub mean_elevation: f64,
    /// Tidal constituents
    pub constituents: Vec<TidalConstituent>,
    /// Gravitational acceleration
    pub g: f64,
    /// Minimum depth
    pub h_min: f64,
    /// Use radiation condition for velocity (Flather-type)
    pub use_radiation: bool,
    /// Ramp-up duration in seconds (None = no ramp-up)
    pub ramp_duration: Option<f64>,
}

impl TidalBC {
    /// Create a new tidal BC with given constituents.
    pub fn new(mean_elevation: f64, constituents: Vec<TidalConstituent>, g: f64) -> Self {
        Self {
            mean_elevation,
            constituents,
            g,
            h_min: 1e-6,
            use_radiation: true,
            ramp_duration: None,
        }
    }

    /// Create with standard gravity.
    pub fn standard(mean_elevation: f64, constituents: Vec<TidalConstituent>) -> Self {
        Self::new(mean_elevation, constituents, 9.81)
    }

    /// Create a simple sinusoidal tide (single constituent).
    pub fn simple(mean_elevation: f64, amplitude: f64, period: f64, phase: f64, g: f64) -> Self {
        Self::new(
            mean_elevation,
            vec![TidalConstituent::new("simple", amplitude, period, phase)],
            g,
        )
    }

    /// Create M2-only tidal forcing.
    pub fn m2_only(mean_elevation: f64, amplitude: f64, phase: f64, g: f64) -> Self {
        Self::new(
            mean_elevation,
            vec![TidalConstituent::m2(amplitude, phase)],
            g,
        )
    }

    /// Evaluate tidal elevation at time t.
    ///
    /// If ramp-up is enabled, the tidal constituents are scaled by the
    /// ramp factor. The mean elevation is NOT ramped (starts at full value).
    pub fn elevation(&self, t: f64) -> f64 {
        let ramp = self.ramp_factor(t);
        let mut eta = self.mean_elevation;
        for constituent in &self.constituents {
            eta += ramp * constituent.evaluate(t);
        }
        eta
    }

    /// Evaluate rate of change of elevation (dη/dt).
    ///
    /// If ramp-up is enabled, this includes both the effect of the
    /// time-varying constituents and the ramp derivative.
    pub fn elevation_rate(&self, t: f64) -> f64 {
        let ramp = self.ramp_factor(t);
        let ramp_deriv = self.ramp_derivative(t);

        let mut deta_dt = 0.0;
        for c in &self.constituents {
            let omega = c.angular_frequency();
            let cos_term = (omega * t + c.phase).cos();
            let sin_term = (omega * t + c.phase).sin();
            // d/dt [R(t) * A * cos(ωt + φ)] = R'(t) * A * cos(...) - R(t) * A * ω * sin(...)
            deta_dt += ramp_deriv * c.amplitude * cos_term;
            deta_dt -= ramp * c.amplitude * omega * sin_term;
        }
        deta_dt
    }

    /// Compute derivative of ramp factor.
    fn ramp_derivative(&self, t: f64) -> f64 {
        match self.ramp_duration {
            None => 0.0,
            Some(duration) if duration <= 0.0 => 0.0,
            Some(duration) => {
                if t <= 0.0 || t >= duration {
                    0.0
                } else {
                    // d/dt [3τ² - 2τ³] = (6τ - 6τ²) / duration
                    let tau = t / duration;
                    6.0 * tau * (1.0 - tau) / duration
                }
            }
        }
    }

    /// Disable radiation condition (use pure Dirichlet for elevation).
    pub fn without_radiation(mut self) -> Self {
        self.use_radiation = false;
        self
    }

    /// Enable smooth ramp-up of tidal forcing.
    ///
    /// The ramp gradually increases tidal amplitudes from 0 to full amplitude
    /// over the specified duration. This prevents initial impulse from
    /// causing spurious oscillations.
    ///
    /// # Arguments
    /// * `duration` - Ramp-up period in seconds (typically 1-3 tidal periods)
    ///
    /// # Example
    ///
    /// ```
    /// use dgcom::boundary::TidalBC;
    ///
    /// // Create M2 tide with 6-hour ramp-up (half an M2 period)
    /// let bc = TidalBC::m2_only(0.0, 0.5, 0.0, 9.81)
    ///     .with_ramp_up(6.0 * 3600.0);
    /// ```
    pub fn with_ramp_up(mut self, duration: f64) -> Self {
        self.ramp_duration = Some(duration);
        self
    }

    /// Compute ramp factor at time t.
    ///
    /// Returns:
    /// - 0 at t=0
    /// - 1 for t >= ramp_duration
    /// - Smooth Hermite interpolation in between: 3t² - 2t³
    pub fn ramp_factor(&self, t: f64) -> f64 {
        match self.ramp_duration {
            None => 1.0,
            Some(duration) if duration <= 0.0 => 1.0,
            Some(duration) => {
                if t <= 0.0 {
                    0.0
                } else if t >= duration {
                    1.0
                } else {
                    // Smooth Hermite interpolation: 3t² - 2t³
                    let tau = t / duration;
                    tau * tau * (3.0 - 2.0 * tau)
                }
            }
        }
    }
}

impl SWEBoundaryCondition for TidalBC {
    fn ghost_state(&self, ctx: &BCContext) -> SWEState {
        // Tidal elevation at current time
        let eta_tidal = self.elevation(ctx.time);

        // Depth from tidal elevation
        let h_ghost = (eta_tidal - ctx.bathymetry).max(0.0);

        // Velocity: use radiation condition or extrapolate
        let u_ghost = if self.use_radiation && h_ghost > self.h_min {
            // Flather-type radiation: allow outgoing waves
            let h_int = ctx.interior_state.h;
            let u_int = ctx.interior_state.velocity_simple(self.h_min);
            let eta_int = h_int + ctx.bathymetry;

            let c = (self.g * h_ghost).sqrt();

            if ctx.normal > 0.0 {
                // Right boundary
                u_int + (eta_int - eta_tidal) / c * self.g.sqrt()
            } else {
                // Left boundary
                u_int - (eta_int - eta_tidal) / c * self.g.sqrt()
            }
        } else {
            // Simple extrapolation
            ctx.interior_state.velocity_simple(self.h_min)
        };

        SWEState::from_primitives(h_ghost, u_ghost)
    }

    fn name(&self) -> &'static str {
        "tidal"
    }
}

/// Time-varying tidal boundary that can be updated during simulation.
///
/// This is useful when tidal data comes from external sources (e.g., TPXO)
/// and needs to be interpolated during the simulation.
#[derive(Clone, Debug)]
pub struct InterpolatedTidalBC {
    /// Time values for interpolation
    pub times: Vec<f64>,
    /// Elevation values at each time
    pub elevations: Vec<f64>,
    /// Gravitational acceleration
    pub g: f64,
    /// Minimum depth
    pub h_min: f64,
}

impl InterpolatedTidalBC {
    /// Create a new interpolated tidal BC.
    pub fn new(times: Vec<f64>, elevations: Vec<f64>, g: f64) -> Self {
        assert_eq!(times.len(), elevations.len());
        assert!(!times.is_empty());
        Self {
            times,
            elevations,
            g,
            h_min: 1e-6,
        }
    }

    /// Interpolate elevation at time t.
    pub fn elevation(&self, t: f64) -> f64 {
        // Find bracketing indices
        if t <= self.times[0] {
            return self.elevations[0];
        }
        if t >= *self.times.last().unwrap() {
            return *self.elevations.last().unwrap();
        }

        let i = self
            .times
            .binary_search_by(|t_i| t_i.partial_cmp(&t).unwrap())
            .unwrap_or_else(|ins| ins.saturating_sub(1));

        // Linear interpolation
        let t0 = self.times[i];
        let t1 = self.times[i + 1];
        let e0 = self.elevations[i];
        let e1 = self.elevations[i + 1];

        let alpha = (t - t0) / (t1 - t0);
        e0 + alpha * (e1 - e0)
    }
}

impl SWEBoundaryCondition for InterpolatedTidalBC {
    fn ghost_state(&self, ctx: &BCContext) -> SWEState {
        let eta = self.elevation(ctx.time);
        let h_ghost = (eta - ctx.bathymetry).max(0.0);

        // Use interior velocity (simple approach)
        let u_ghost = ctx.interior_state.velocity_simple(self.h_min);

        SWEState::from_primitives(h_ghost, u_ghost)
    }

    fn name(&self) -> &'static str {
        "interpolated_tidal"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const G: f64 = 10.0;
    const TOL: f64 = 1e-10;

    fn make_context(h: f64, hu: f64, bath: f64, normal: f64, time: f64) -> BCContext {
        BCContext {
            time,
            position: 0.0,
            interior_state: SWEState::new(h, hu),
            bathymetry: bath,
            normal,
        }
    }

    #[test]
    fn test_constituent_evaluate() {
        let c = TidalConstituent::new("test", 1.0, 2.0 * PI, 0.0);

        // At t=0, cos(0) = 1
        assert!((c.evaluate(0.0) - 1.0).abs() < TOL);

        // At t=π (half period), cos(π) = -1
        assert!((c.evaluate(PI) - (-1.0)).abs() < TOL);

        // At t=2π (full period), cos(2π) = 1
        assert!((c.evaluate(2.0 * PI) - 1.0).abs() < TOL);
    }

    #[test]
    fn test_constituent_with_phase() {
        // Phase of π/2 shifts cosine to sine
        let c = TidalConstituent::new("test", 1.0, 2.0 * PI, -PI / 2.0);

        // At t=0, cos(-π/2) = 0
        assert!(c.evaluate(0.0).abs() < TOL);

        // At t=π/2, cos(0) = 1
        assert!((c.evaluate(PI / 2.0) - 1.0).abs() < TOL);
    }

    #[test]
    fn test_tidal_bc_mean_elevation() {
        let bc = TidalBC::new(2.0, vec![], G);

        // No constituents, should always return mean elevation
        assert!((bc.elevation(0.0) - 2.0).abs() < TOL);
        assert!((bc.elevation(100.0) - 2.0).abs() < TOL);
    }

    #[test]
    fn test_tidal_bc_with_constituent() {
        let bc = TidalBC::simple(2.0, 0.5, 2.0 * PI, 0.0, G);

        // At t=0: η = 2.0 + 0.5 * cos(0) = 2.5
        assert!((bc.elevation(0.0) - 2.5).abs() < TOL);

        // At t=π: η = 2.0 + 0.5 * cos(π) = 1.5
        assert!((bc.elevation(PI) - 1.5).abs() < TOL);
    }

    #[test]
    fn test_tidal_bc_ghost_state() {
        let bc = TidalBC::simple(2.0, 0.5, 2.0 * PI, 0.0, G).without_radiation();

        // At t=0, η = 2.5, h = η - bath = 2.5
        let ctx = make_context(2.0, 0.0, 0.0, 1.0, 0.0);
        let ghost = bc.ghost_state(&ctx);

        assert!((ghost.h - 2.5).abs() < TOL);
    }

    #[test]
    fn test_tidal_bc_elevation_rate() {
        let bc = TidalBC::simple(2.0, 0.5, 2.0 * PI, 0.0, G);

        // dη/dt = -A * ω * sin(ωt)
        // At t=0: dη/dt = 0
        assert!(bc.elevation_rate(0.0).abs() < TOL);

        // At t=π/2: dη/dt = -0.5 * 1 * sin(π/2) = -0.5
        assert!((bc.elevation_rate(PI / 2.0) - (-0.5)).abs() < TOL);
    }

    #[test]
    fn test_with_nodal_correction_scales_and_shifts() {
        use crate::tides::NodalCorrection;

        let raw = TidalConstituent::m2(0.5, 0.2);
        let corr = NodalCorrection {
            f: 1.04,
            u_deg: 3.0,
            v0_deg: 47.0,
        };
        let corrected = raw.with_nodal_correction(&corr);

        assert!((corrected.amplitude - 1.04 * 0.5).abs() < TOL);
        assert!((corrected.phase - (0.2 + 50.0_f64.to_radians())).abs() < TOL);
        // Name and period are preserved.
        assert_eq!(corrected.name, "M2");
        assert!((corrected.period - raw.period).abs() < TOL);
    }

    #[test]
    fn test_with_identity_nodal_correction_is_noop() {
        use crate::tides::NodalCorrection;

        let raw = TidalConstituent::s2(0.3, 0.9);
        let corrected = raw.with_nodal_correction(&NodalCorrection::IDENTITY);
        assert!((corrected.amplitude - raw.amplitude).abs() < TOL);
        assert!((corrected.phase - raw.phase).abs() < TOL);
    }

    /// Regression: each named constructor must carry the exact period
    /// 360°/speed of its tabulated Doodson angular speed (Schureman 1958),
    /// not a two-decimal rounding of it (M2 was 12.42 h instead of
    /// 12.4206 h, which drifts the phase by ~6° over a year).
    #[test]
    fn test_constructor_periods_match_doodson_speeds() {
        // (constituent, tabulated speed in °/h)
        let cases = [
            (TidalConstituent::m2(1.0, 0.0), 28.984_104_2),
            (TidalConstituent::s2(1.0, 0.0), 30.0),
            (TidalConstituent::n2(1.0, 0.0), 28.439_729_5),
            (TidalConstituent::k1(1.0, 0.0), 15.041_068_6),
            (TidalConstituent::o1(1.0, 0.0), 13.943_035_6),
            (TidalConstituent::p1(1.0, 0.0), 14.958_931_4),
        ];
        for (c, speed_deg_per_hour) in cases {
            let speed = c.angular_frequency().to_degrees() * 3600.0;
            assert!(
                (speed - speed_deg_per_hour).abs() < 1e-6,
                "{}: speed {speed} °/h vs tabulated {speed_deg_per_hour} °/h",
                c.name
            );
        }
    }

    #[test]
    fn test_interpolated_tidal() {
        let times = vec![0.0, 1.0, 2.0];
        let elevations = vec![1.0, 2.0, 1.5];
        let bc = InterpolatedTidalBC::new(times, elevations, G);

        // At exact times
        assert!((bc.elevation(0.0) - 1.0).abs() < TOL);
        assert!((bc.elevation(1.0) - 2.0).abs() < TOL);
        assert!((bc.elevation(2.0) - 1.5).abs() < TOL);

        // Interpolated
        assert!((bc.elevation(0.5) - 1.5).abs() < TOL);

        // Extrapolated (clamped)
        assert!((bc.elevation(-1.0) - 1.0).abs() < TOL);
        assert!((bc.elevation(10.0) - 1.5).abs() < TOL);
    }

    #[test]
    fn test_multiple_constituents() {
        let m2 = TidalConstituent::m2(1.0, 0.0);
        let s2 = TidalConstituent::s2(0.5, 0.0);
        let bc = TidalBC::new(0.0, vec![m2, s2], G);

        // At t=0, both cosines are 1
        assert!((bc.elevation(0.0) - 1.5).abs() < TOL);
    }

    #[test]
    fn test_ramp_factor() {
        let bc = TidalBC::simple(0.0, 1.0, 2.0 * PI, 0.0, G).with_ramp_up(10.0);

        // At t=0: ramp = 0
        assert!(bc.ramp_factor(0.0).abs() < TOL);

        // At t=duration: ramp = 1
        assert!((bc.ramp_factor(10.0) - 1.0).abs() < TOL);

        // At t > duration: ramp = 1
        assert!((bc.ramp_factor(20.0) - 1.0).abs() < TOL);

        // At t=duration/2: ramp = 0.5 (Hermite interpolation)
        // f(0.5) = 3*0.25 - 2*0.125 = 0.75 - 0.25 = 0.5
        assert!((bc.ramp_factor(5.0) - 0.5).abs() < TOL);

        // At t=duration/4: tau=0.25, f(0.25) = 3*0.0625 - 2*0.015625 = 0.15625
        assert!((bc.ramp_factor(2.5) - 0.15625).abs() < TOL);
    }

    #[test]
    fn test_ramp_factor_no_ramp() {
        let bc = TidalBC::simple(0.0, 1.0, 2.0 * PI, 0.0, G);
        // No ramp-up: factor is always 1
        assert!((bc.ramp_factor(0.0) - 1.0).abs() < TOL);
        assert!((bc.ramp_factor(5.0) - 1.0).abs() < TOL);
    }

    #[test]
    fn test_elevation_with_ramp() {
        let bc = TidalBC::simple(2.0, 1.0, 2.0 * PI, 0.0, G).with_ramp_up(10.0);

        // At t=0: ramp=0, so η = mean_elevation only = 2.0
        // (constituent contribution is ramped to zero)
        assert!((bc.elevation(0.0) - 2.0).abs() < TOL);

        // At t=duration: ramp=1, so η = 2.0 + 1.0*cos(10) ≈ 2.0 - 0.839
        let expected = 2.0 + (10.0_f64).cos();
        assert!((bc.elevation(10.0) - expected).abs() < TOL);

        // At t=duration/2: ramp=0.5, so η = 2.0 + 0.5*cos(5)
        let expected_mid = 2.0 + 0.5 * (5.0_f64).cos();
        assert!((bc.elevation(5.0) - expected_mid).abs() < TOL);
    }

    #[test]
    fn test_ramp_smooth_derivative() {
        let bc = TidalBC::simple(0.0, 1.0, 2.0 * PI, 0.0, G).with_ramp_up(10.0);

        // Derivative should be zero at endpoints (smooth entry/exit)
        assert!(bc.ramp_derivative(0.0).abs() < TOL);
        assert!(bc.ramp_derivative(10.0).abs() < TOL);

        // Maximum derivative at midpoint: 6*0.5*(1-0.5)/10 = 0.15
        assert!((bc.ramp_derivative(5.0) - 0.15).abs() < TOL);
    }

    #[test]
    fn test_interpolated_tidal_binary_search() {
        // Dense time series to exercise binary search properly
        let n = 1000;
        let times: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
        let elevations: Vec<f64> = times.iter().map(|t| (t * 0.5).sin()).collect();
        let bc = InterpolatedTidalBC::new(times, elevations, G);

        // Test at exact grid points
        assert!((bc.elevation(0.0) - 0.0).abs() < TOL);
        assert!((bc.elevation(5.0) - (2.5_f64).sin()).abs() < TOL);

        // Test interpolation at midpoints between grid nodes
        for i in 0..n - 1 {
            let t = i as f64 * 0.1 + 0.05;
            let e0 = (i as f64 * 0.1 * 0.5).sin();
            let e1 = ((i + 1) as f64 * 0.1 * 0.5).sin();
            let expected = 0.5 * (e0 + e1); // midpoint linear interpolation
            assert!(
                (bc.elevation(t) - expected).abs() < 1e-8,
                "Mismatch at t={}: got {}, expected {}",
                t,
                bc.elevation(t),
                expected,
            );
        }

        // Test clamping at boundaries
        assert!((bc.elevation(-1.0) - 0.0).abs() < TOL);
        assert!((bc.elevation(200.0) - ((n - 1) as f64 * 0.1 * 0.5).sin()).abs() < TOL);
    }
}

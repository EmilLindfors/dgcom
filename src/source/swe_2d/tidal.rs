//! Tidal potential body force for shallow water equations.
//!
//! The tidal potential represents the gravitational forcing from the moon and sun
//! that drives astronomical tides. The equilibrium tide formula:
//!
//! Φ(λ, φ, t) = Σᵢ Aᵢ Gᵢ(φ) cos(ωᵢt + mλ + φᵢ)
//!
//! where:
//! - Aᵢ = amplitude for constituent i (meters)
//! - Gᵢ(φ) = latitude-dependent factor (cos²φ for semidiurnal, sin(2φ) for diurnal)
//! - ωᵢ = angular frequency
//! - λ = longitude, φ = latitude
//! - m = wavenumber (2 for semidiurnal, 1 for diurnal)
//!
//! The source term for shallow water equations is:
//! S = (0, -gh ∂Φ/∂x, -gh ∂Φ/∂y)
//!
//! # Norwegian Coast Context
//!
//! At 60°N latitude:
//! - M2 is the dominant constituent (amplitude ~1m on outer coast)
//! - Strong tidal forcing in narrow fjords and straits
//!
//! # References
//!
//! - Doodson (1921): Harmonic development of tide-generating potential
//! - Pugh & Woodworth (2014): Sea-Level Science

use crate::boundary::TidalConstituent;
use crate::solver::SWEState2D;
use crate::source::{SourceContext2D, SourceTerm2D};
use std::f64::consts::PI;

/// Tidal potential configuration for a single constituent.
///
/// Combines the basic tidal constituent (period, amplitude, phase) with
/// the additional parameters needed for potential computation.
#[derive(Clone, Debug)]
pub struct TidalPotentialConstituent {
    /// Base constituent (defines period, amplitude, phase)
    pub constituent: TidalConstituent,
    /// Love number reduction factor: (1 + k - h) ≈ 0.69 for most constituents
    ///
    /// This accounts for the solid Earth's elastic response to tidal forcing.
    pub love_factor: f64,
    /// Wavenumber m in the harmonic expansion
    /// - m = 2 for semidiurnal (M2, S2, N2)
    /// - m = 1 for diurnal (K1, O1, P1)
    /// - m = 0 for long-period (Mf, Mm)
    pub wavenumber: i32,
}

impl TidalPotentialConstituent {
    /// Create a new tidal potential constituent.
    pub fn new(constituent: TidalConstituent, love_factor: f64, wavenumber: i32) -> Self {
        Self {
            constituent,
            love_factor,
            wavenumber,
        }
    }

    /// Create M2 (principal lunar semidiurnal) potential constituent.
    ///
    /// Period ≈ 12.42 hours (12h 25min 14s)
    pub fn m2(amplitude: f64, phase: f64) -> Self {
        Self {
            constituent: TidalConstituent::m2(amplitude, phase),
            love_factor: 0.69,
            wavenumber: 2,
        }
    }

    /// Create S2 (principal solar semidiurnal) potential constituent.
    ///
    /// Period = 12.00 hours exactly
    pub fn s2(amplitude: f64, phase: f64) -> Self {
        Self {
            constituent: TidalConstituent::s2(amplitude, phase),
            love_factor: 0.69,
            wavenumber: 2,
        }
    }

    /// Create N2 (larger lunar elliptic semidiurnal) potential constituent.
    ///
    /// Period ≈ 12.66 hours
    pub fn n2(amplitude: f64, phase: f64) -> Self {
        Self {
            constituent: TidalConstituent::n2(amplitude, phase),
            love_factor: 0.69,
            wavenumber: 2,
        }
    }

    /// Create K1 (lunar diurnal) potential constituent.
    ///
    /// Period ≈ 23.93 hours
    pub fn k1(amplitude: f64, phase: f64) -> Self {
        Self {
            constituent: TidalConstituent::k1(amplitude, phase),
            love_factor: 0.69,
            wavenumber: 1,
        }
    }

    /// Create O1 (principal lunar diurnal) potential constituent.
    ///
    /// Period ≈ 25.82 hours
    pub fn o1(amplitude: f64, phase: f64) -> Self {
        Self {
            constituent: TidalConstituent::o1(amplitude, phase),
            love_factor: 0.69,
            wavenumber: 1,
        }
    }

    /// Create P1 (principal solar diurnal) potential constituent.
    ///
    /// Period ≈ 24.07 hours
    pub fn p1(amplitude: f64, phase: f64) -> Self {
        Self {
            constituent: TidalConstituent::p1(amplitude, phase),
            love_factor: 0.69,
            wavenumber: 1,
        }
    }
}

/// Tidal potential body force source term.
///
/// Computes the gravitational forcing from astronomical tides and adds
/// it to the momentum equations:
///
/// S_hu = -g h ∂Φ/∂x
/// S_hv = -g h ∂Φ/∂y
///
/// # Coordinate Systems
///
/// Two coordinate modes are supported:
/// - **Cartesian**: (x, y) in meters from reference point
/// - **Geographic**: (x, y) interpreted as (longitude, latitude) in degrees
///
/// # Example
///
/// ```
/// use dgcom::source::{TidalPotential, TidalPotentialConstituent, SourceTerm2D, SourceContext2D};
/// use dgcom::SWEState2D;
///
/// // The M2 potential on a local plane about 5°E, 60°N
/// let m2 = TidalPotentialConstituent::m2(0.5, 0.0);
/// let potential = TidalPotential::cartesian(vec![m2], 5.0, 60.0);
///
/// // Evaluate at a point
/// let ctx = SourceContext2D::new(
///     0.0, (0.0, 0.0), SWEState2D::new(10.0, 0.0, 0.0),
///     0.0, (0.0, 0.0), 9.81, 1e-6,
/// );
/// let source = potential.evaluate(&ctx);
/// ```
#[derive(Clone, Debug)]
pub struct TidalPotential {
    /// Tidal potential constituents
    pub constituents: Vec<TidalPotentialConstituent>,
    /// Reference longitude (radians) for phase calculation
    pub ref_longitude: f64,
    /// Reference latitude (radians) for latitude factor calculation
    pub ref_latitude: f64,
    /// Coordinate system flag:
    /// - true: (x, y) are in meters from reference point (Cartesian/UTM)
    /// - false: (x, y) are (longitude, latitude) in degrees
    pub cartesian_coords: bool,
    /// Earth radius (meters) for coordinate conversion
    pub earth_radius: f64,
}

impl TidalPotential {
    /// Create a new tidal potential with full configuration.
    pub fn new(
        constituents: Vec<TidalPotentialConstituent>,
        ref_longitude: f64,
        ref_latitude: f64,
        cartesian_coords: bool,
    ) -> Self {
        Self {
            constituents,
            ref_longitude,
            ref_latitude,
            cartesian_coords,
            earth_radius: 6.371e6,
        }
    }

    /// Create on a local plane: (x, y) in metres from the reference point
    /// at `ref_longitude_deg`, `ref_latitude_deg` (degrees east and north).
    pub fn cartesian(
        constituents: Vec<TidalPotentialConstituent>,
        ref_longitude_deg: f64,
        ref_latitude_deg: f64,
    ) -> Self {
        // `x * PI / 180` rather than `to_radians` (`x * (PI / 180)`), which
        // can differ in the last bit: the same as the preset this replaced
        Self::new(
            constituents,
            ref_longitude_deg * PI / 180.0,
            ref_latitude_deg * PI / 180.0,
            true,
        )
    }

    /// Create with geographic coordinates (lon, lat in degrees).
    pub fn geographic(constituents: Vec<TidalPotentialConstituent>) -> Self {
        Self {
            constituents,
            ref_longitude: 0.0,
            ref_latitude: 0.0,
            cartesian_coords: false,
            earth_radius: 6.371e6,
        }
    }

    /// Evaluate tidal potential and its gradients at a point.
    ///
    /// Returns (Φ, ∂Φ/∂x, ∂Φ/∂y) where x, y are in physical coordinates.
    pub fn evaluate_potential(&self, x: f64, y: f64, t: f64) -> (f64, f64, f64) {
        let mut phi = 0.0;
        let mut dphi_dx = 0.0;
        let mut dphi_dy = 0.0;

        for c in &self.constituents {
            let omega = c.constituent.angular_frequency();
            let a = c.constituent.amplitude * c.love_factor;
            let m = c.wavenumber as f64;

            // Convert position to longitude/latitude
            let (lon, lat) = if self.cartesian_coords {
                // x = R * cos(lat) * dlon, y = R * dlat
                let dlat = y / self.earth_radius;
                let dlon = x / (self.earth_radius * self.ref_latitude.cos());
                (self.ref_longitude + dlon, self.ref_latitude + dlat)
            } else {
                // (x, y) are (longitude, latitude) in degrees
                (x * PI / 180.0, y * PI / 180.0)
            };

            // Latitude-dependent factor G(lat)
            // For semidiurnal (m=2): G = cos²(lat)
            // For diurnal (m=1): G = sin(2*lat) = 2*sin(lat)*cos(lat)
            // For long-period (m=0): G = (3*cos²(lat) - 1)/2
            let (lat_factor, dlat_factor_dlat) = match c.wavenumber {
                2 => (lat.cos().powi(2), -2.0 * lat.cos() * lat.sin()),
                1 => ((2.0 * lat).sin(), 2.0 * (2.0 * lat).cos()),
                0 => {
                    let cos2 = lat.cos().powi(2);
                    ((3.0 * cos2 - 1.0) / 2.0, -3.0 * lat.cos() * lat.sin())
                }
                _ => (1.0, 0.0),
            };

            // Phase: ω*t + m*λ + φ_0
            let phase = omega * t + m * lon + c.constituent.phase;

            // Potential: Φ = A * G(lat) * cos(phase)
            phi += a * lat_factor * phase.cos();

            // Gradients in physical coordinates
            if self.cartesian_coords {
                // ∂Φ/∂x = ∂Φ/∂λ * ∂λ/∂x
                // ∂phase/∂λ = m, ∂λ/∂x = 1/(R*cos(lat))
                let dlon_dx = 1.0 / (self.earth_radius * lat.cos());
                dphi_dx += -a * lat_factor * phase.sin() * m * dlon_dx;

                // ∂Φ/∂y = ∂Φ/∂φ * ∂φ/∂y
                // ∂φ/∂y = 1/R
                let dlat_dy = 1.0 / self.earth_radius;
                dphi_dy += a * dlat_factor_dlat * dlat_dy * phase.cos();
            } else {
                // Geographic: gradients in degrees (less common use case)
                // User should convert if needed
                dphi_dx += -a * lat_factor * phase.sin() * m;
                dphi_dy += a * dlat_factor_dlat * phase.cos();
            }
        }

        (phi, dphi_dx, dphi_dy)
    }

    /// Get the number of constituents.
    pub fn n_constituents(&self) -> usize {
        self.constituents.len()
    }
}

impl SourceTerm2D for TidalPotential {
    fn evaluate(&self, ctx: &SourceContext2D) -> SWEState2D {
        if self.constituents.is_empty() {
            return SWEState2D::zero();
        }

        let (x, y) = ctx.position;
        let (_phi, dphi_dx, dphi_dy) = self.evaluate_potential(x, y, ctx.time);

        // S = (0, -gh ∂Φ/∂x, -gh ∂Φ/∂y)
        let h = ctx.state.h;
        let g = ctx.g;

        SWEState2D {
            h: 0.0,
            hu: -g * h * dphi_dx,
            hv: -g * h * dphi_dy,
        }
    }

    fn is_node_local(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "tidal_potential"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    fn make_context(h: f64, x: f64, y: f64, t: f64) -> SourceContext2D {
        SourceContext2D::new(
            t,
            (x, y),
            SWEState2D::new(h, 0.0, 0.0),
            0.0,
            (0.0, 0.0),
            9.81,
            1e-6,
        )
    }

    #[test]
    fn test_m2_periodicity() {
        let m2 = TidalPotentialConstituent::m2(1.0, 0.0);
        let potential = TidalPotential::cartesian(vec![m2.clone()], 5.0, 60.0);

        let period = m2.constituent.period; // ~12.42 hours in seconds
        let (phi_0, _, _) = potential.evaluate_potential(0.0, 0.0, 0.0);
        let (phi_t, _, _) = potential.evaluate_potential(0.0, 0.0, period);

        assert!(
            (phi_0 - phi_t).abs() < TOL,
            "M2 should be periodic: phi(0) = {}, phi(T) = {}",
            phi_0,
            phi_t
        );
    }

    #[test]
    fn test_empty_constituents() {
        let potential = TidalPotential::cartesian(vec![], 5.0, 60.0);
        let ctx = make_context(10.0, 0.0, 0.0, 0.0);
        let source = potential.evaluate(&ctx);

        assert!(source.h.abs() < TOL);
        assert!(source.hu.abs() < TOL);
        assert!(source.hv.abs() < TOL);
    }

    #[test]
    fn test_potential_gradient_finite_diff() {
        let m2 = TidalPotentialConstituent::m2(1.0, 0.0);
        let potential = TidalPotential::cartesian(vec![m2], 5.0, 60.0);

        let x0 = 1000.0; // 1 km from reference
        let y0 = 1000.0;
        let t = 0.0;
        let eps = 1.0; // 1 meter for finite difference

        let (_, dphi_dx, dphi_dy) = potential.evaluate_potential(x0, y0, t);

        // Finite difference approximation
        let (phi_xp, _, _) = potential.evaluate_potential(x0 + eps, y0, t);
        let (phi_xm, _, _) = potential.evaluate_potential(x0 - eps, y0, t);
        let dphi_dx_fd = (phi_xp - phi_xm) / (2.0 * eps);

        let (phi_yp, _, _) = potential.evaluate_potential(x0, y0 + eps, t);
        let (phi_ym, _, _) = potential.evaluate_potential(x0, y0 - eps, t);
        let dphi_dy_fd = (phi_yp - phi_ym) / (2.0 * eps);

        // Allow some tolerance due to finite difference approximation
        let rel_tol = 0.01; // 1% relative error
        let abs_tol = 1e-14; // For very small gradients

        let err_x = (dphi_dx - dphi_dx_fd).abs();
        let err_y = (dphi_dy - dphi_dy_fd).abs();

        assert!(
            err_x < rel_tol * dphi_dx.abs() + abs_tol,
            "dPhi/dx mismatch: analytical = {}, FD = {}, error = {}",
            dphi_dx,
            dphi_dx_fd,
            err_x
        );

        assert!(
            err_y < rel_tol * dphi_dy.abs() + abs_tol,
            "dPhi/dy mismatch: analytical = {}, FD = {}, error = {}",
            dphi_dy,
            dphi_dy_fd,
            err_y
        );
    }

    #[test]
    fn test_source_scaling_with_depth() {
        let m2 = TidalPotentialConstituent::m2(1.0, 0.0);
        let potential = TidalPotential::cartesian(vec![m2], 5.0, 60.0);

        let ctx1 = make_context(10.0, 1000.0, 1000.0, 0.0);
        let ctx2 = make_context(20.0, 1000.0, 1000.0, 0.0);

        let source1 = potential.evaluate(&ctx1);
        let source2 = potential.evaluate(&ctx2);

        // Source should scale linearly with depth
        assert!(
            (source2.hu / source1.hu - 2.0).abs() < TOL,
            "Source should scale with depth: ratio = {}",
            source2.hu / source1.hu
        );
    }

    #[test]
    fn test_latitude_factor_semidiurnal() {
        // For semidiurnal tides, amplitude maximizes at equator (lat=0)
        // and vanishes at poles (lat=±90°)
        let m2 = TidalPotentialConstituent::m2(1.0, 0.0);
        let potential = TidalPotential::geographic(vec![m2]);

        // At equator (lat=0)
        let (phi_eq, _, _) = potential.evaluate_potential(0.0, 0.0, 0.0);

        // At 60°N
        let (phi_60, _, _) = potential.evaluate_potential(0.0, 60.0, 0.0);

        // At pole (lat=90°)
        let (phi_pole, _, _) = potential.evaluate_potential(0.0, 90.0, 0.0);

        // cos²(0) = 1, cos²(60°) = 0.25, cos²(90°) = 0
        assert!(phi_eq.abs() > phi_60.abs());
        assert!(phi_pole.abs() < TOL);
    }

    #[test]
    fn test_multiple_constituents() {
        let m2 = TidalPotentialConstituent::m2(1.0, 0.0);
        let s2 = TidalPotentialConstituent::s2(0.5, 0.0);
        let combined = TidalPotential::cartesian(vec![m2.clone(), s2.clone()], 5.0, 60.0);
        let m2_only = TidalPotential::cartesian(vec![m2], 5.0, 60.0);
        let s2_only = TidalPotential::cartesian(vec![s2], 5.0, 60.0);

        let x = 1000.0;
        let y = 1000.0;
        let t = 1000.0;

        let (phi_combined, _, _) = combined.evaluate_potential(x, y, t);
        let (phi_m2, _, _) = m2_only.evaluate_potential(x, y, t);
        let (phi_s2, _, _) = s2_only.evaluate_potential(x, y, t);

        // Constituents should superpose linearly
        assert!(
            (phi_combined - (phi_m2 + phi_s2)).abs() < TOL,
            "Constituents should superpose: {} vs {}",
            phi_combined,
            phi_m2 + phi_s2
        );
    }

    #[test]
    fn test_diurnal_constituent() {
        let k1 = TidalPotentialConstituent::k1(1.0, 0.0);
        let potential = TidalPotential::cartesian(vec![k1.clone()], 5.0, 60.0);

        let period = k1.constituent.period; // ~23.93 hours in seconds
        let (phi_0, _, _) = potential.evaluate_potential(0.0, 0.0, 0.0);
        let (phi_t, _, _) = potential.evaluate_potential(0.0, 0.0, period);

        assert!(
            (phi_0 - phi_t).abs() < TOL,
            "K1 should be periodic: phi(0) = {}, phi(T) = {}",
            phi_0,
            phi_t
        );
    }
}

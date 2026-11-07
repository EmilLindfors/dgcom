//! Physical quantity newtypes for vertical coordinates.
//!
//! These types prevent mixing up different physical quantities
//! that all have the same underlying type (f64).

use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};

// =============================================================================
// Depth (water column depth, always positive)
// =============================================================================

/// Water depth (H), always positive.
///
/// Represents the distance from the undisturbed water surface
/// to the bathymetry (bottom).
///
/// # Convention
///
/// Depth is **always positive**. A location with 100m of water
/// has `Depth(100.0)`, not `Depth(-100.0)`.
///
/// # Example
///
/// ```
/// use dgcom::types::Depth;
///
/// let h = Depth::new(200.0);  // 200m deep water
/// assert_eq!(h.meters(), 200.0);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct Depth(f64);

impl Depth {
    /// Create a new depth value.
    ///
    /// # Panics
    ///
    /// Panics if depth is negative.
    #[inline]
    pub fn new(meters: f64) -> Self {
        debug_assert!(meters >= 0.0, "Depth must be non-negative, got {}", meters);
        Self(meters)
    }

    /// Create depth without validation (for hot paths).
    ///
    /// # Safety
    ///
    /// Caller must ensure the value is non-negative.
    #[inline]
    pub const fn new_unchecked(meters: f64) -> Self {
        Self(meters)
    }

    /// Zero depth (dry cell).
    pub const ZERO: Self = Self(0.0);

    /// Get the depth in meters.
    #[inline]
    pub fn meters(self) -> f64 {
        self.0
    }

    /// Convert to raw f64.
    #[inline]
    pub fn into_inner(self) -> f64 {
        self.0
    }
}

impl fmt::Display for Depth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.2}m", self.0)
    }
}

impl From<Depth> for f64 {
    #[inline]
    fn from(d: Depth) -> f64 {
        d.0
    }
}

// =============================================================================
// Elevation (free surface elevation, can be positive or negative)
// =============================================================================

/// Free surface elevation (η), relative to mean sea level.
///
/// Can be positive (high tide, storm surge) or negative (low tide).
///
/// # Example
///
/// ```
/// use dgcom::types::Elevation;
///
/// let high_tide = Elevation::new(1.5);   // 1.5m above MSL
/// let low_tide = Elevation::new(-0.8);   // 0.8m below MSL
/// ```
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct Elevation(f64);

impl Elevation {
    /// Create a new elevation value.
    #[inline]
    pub const fn new(meters: f64) -> Self {
        Self(meters)
    }

    /// Zero elevation (mean sea level).
    pub const ZERO: Self = Self(0.0);

    /// Get the elevation in meters.
    #[inline]
    pub fn meters(self) -> f64 {
        self.0
    }

    /// Convert to raw f64.
    #[inline]
    pub fn into_inner(self) -> f64 {
        self.0
    }
}

impl fmt::Display for Elevation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:+.2}m", self.0)
    }
}

impl From<Elevation> for f64 {
    #[inline]
    fn from(e: Elevation) -> f64 {
        e.0
    }
}

impl Add for Elevation {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl Sub for Elevation {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl Neg for Elevation {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

// =============================================================================
// Sigma (terrain-following vertical coordinate)
// =============================================================================

/// Sigma coordinate (σ), terrain-following vertical coordinate.
///
/// σ ∈ [-1, 0] where:
/// - σ = -1 at the bottom (z = -H)
/// - σ = 0 at the surface (z = η)
///
/// # Example
///
/// ```
/// use dgcom::types::{Sigma, Depth, Elevation};
///
/// let sigma = Sigma::new(-0.5);  // Mid-depth
/// let h = Depth::new(100.0);
/// let eta = Elevation::new(0.0);
///
/// // Physical depth at sigma = -0.5 is 50m below surface
/// let z = sigma.to_z(eta, h);
/// assert!((z.meters() - (-50.0)).abs() < 1e-10);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct Sigma(f64);

impl Sigma {
    /// Create a new sigma coordinate.
    ///
    /// # Panics
    ///
    /// Debug-panics if sigma is outside [-1, 0].
    #[inline]
    pub fn new(value: f64) -> Self {
        debug_assert!(
            (-1.0..=0.0).contains(&value),
            "Sigma must be in [-1, 0], got {}",
            value
        );
        Self(value)
    }

    /// Create sigma without validation (for hot paths).
    #[inline]
    pub const fn new_unchecked(value: f64) -> Self {
        Self(value)
    }

    /// Sigma at the surface.
    pub const SURFACE: Self = Self(0.0);

    /// Sigma at the bottom.
    pub const BOTTOM: Self = Self(-1.0);

    /// Get the raw sigma value.
    #[inline]
    pub fn value(self) -> f64 {
        self.0
    }

    /// Convert to raw f64.
    #[inline]
    pub fn into_inner(self) -> f64 {
        self.0
    }

    /// Convert sigma to physical z-coordinate.
    ///
    /// z = η + (η + H) × σ
    #[inline]
    pub fn to_z(self, eta: Elevation, h: Depth) -> PhysicalZ {
        let total_depth = eta.0 + h.0;
        PhysicalZ::new(eta.0 + total_depth * self.0)
    }

    /// Compute sigma from physical z-coordinate.
    ///
    /// σ = (z - η) / (η + H)
    #[inline]
    pub fn from_z(z: PhysicalZ, eta: Elevation, h: Depth) -> Self {
        let total_depth = eta.0 + h.0;
        if total_depth.abs() < 1e-12 {
            Self::SURFACE
        } else {
            Self((z.0 - eta.0) / total_depth)
        }
    }
}

impl fmt::Display for Sigma {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "σ={:.3}", self.0)
    }
}

impl From<Sigma> for f64 {
    #[inline]
    fn from(s: Sigma) -> f64 {
        s.0
    }
}

// =============================================================================
// PhysicalZ (absolute vertical position)
// =============================================================================

/// Physical z-coordinate (vertical position in meters).
///
/// Follows oceanographic convention:
/// - z = 0 at mean sea level
/// - z > 0 above MSL (rare in water)
/// - z < 0 below MSL (typical)
///
/// # Example
///
/// ```
/// use dgcom::types::PhysicalZ;
///
/// let z_surface = PhysicalZ::new(0.5);    // 0.5m above MSL (high tide)
/// let z_midwater = PhysicalZ::new(-50.0); // 50m below MSL
/// let z_bottom = PhysicalZ::new(-200.0);  // 200m below MSL
/// ```
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct PhysicalZ(f64);

impl PhysicalZ {
    /// Create a new physical z-coordinate.
    #[inline]
    pub const fn new(meters: f64) -> Self {
        Self(meters)
    }

    /// Z at mean sea level.
    pub const MSL: Self = Self(0.0);

    /// Get the z-coordinate in meters.
    #[inline]
    pub fn meters(self) -> f64 {
        self.0
    }

    /// Convert to raw f64.
    #[inline]
    pub fn into_inner(self) -> f64 {
        self.0
    }

    /// Check if this z is below mean sea level.
    #[inline]
    pub fn is_below_msl(self) -> bool {
        self.0 < 0.0
    }
}

impl fmt::Display for PhysicalZ {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "z={:+.2}m", self.0)
    }
}

impl From<PhysicalZ> for f64 {
    #[inline]
    fn from(z: PhysicalZ) -> f64 {
        z.0
    }
}

impl Add for PhysicalZ {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl Sub for PhysicalZ {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl Mul<f64> for PhysicalZ {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f64) -> Self {
        Self(self.0 * rhs)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_depth() {
        let h = Depth::new(100.0);
        assert_eq!(h.meters(), 100.0);
        assert_eq!(f64::from(h), 100.0);
    }

    #[test]
    fn test_elevation() {
        let eta = Elevation::new(1.5);
        assert_eq!(eta.meters(), 1.5);

        let low = Elevation::new(-0.8);
        assert_eq!(low.meters(), -0.8);
    }

    #[test]
    fn test_sigma_to_z() {
        let h = Depth::new(100.0);
        let eta = Elevation::ZERO;

        // Surface
        let z_surf = Sigma::SURFACE.to_z(eta, h);
        assert!((z_surf.meters() - 0.0).abs() < 1e-10);

        // Bottom
        let z_bot = Sigma::BOTTOM.to_z(eta, h);
        assert!((z_bot.meters() - (-100.0)).abs() < 1e-10);

        // Mid-depth
        let z_mid = Sigma::new(-0.5).to_z(eta, h);
        assert!((z_mid.meters() - (-50.0)).abs() < 1e-10);
    }

    #[test]
    fn test_sigma_to_z_with_elevation() {
        let h = Depth::new(100.0);
        let eta = Elevation::new(2.0); // 2m above MSL

        // Surface should be at eta
        let z_surf = Sigma::SURFACE.to_z(eta, h);
        assert!((z_surf.meters() - 2.0).abs() < 1e-10);

        // Bottom should be at eta - total_depth = 2 - 102 = -100
        // Wait: z = η + (η + H) × σ = 2 + 102 × (-1) = 2 - 102 = -100
        let z_bot = Sigma::BOTTOM.to_z(eta, h);
        assert!((z_bot.meters() - (-100.0)).abs() < 1e-10);
    }

    #[test]
    fn test_sigma_from_z() {
        let h = Depth::new(100.0);
        let eta = Elevation::ZERO;

        let sigma = Sigma::from_z(PhysicalZ::new(-50.0), eta, h);
        assert!((sigma.value() - (-0.5)).abs() < 1e-10);
    }

    #[test]
    fn test_physical_z() {
        let z = PhysicalZ::new(-50.0);
        assert!(z.is_below_msl());
        assert_eq!(z.meters(), -50.0);
    }
}

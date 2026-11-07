//! Coordinate type abstractions for dimension-independent mesh operations.
//!
//! The [`Point`] trait provides a unified interface for coordinates in 1D, 2D, and 3D.
//! This enables generic mesh algorithms that work across dimensions.

use std::fmt::Debug;

/// A point in physical or reference space.
///
/// This trait abstracts over different coordinate representations:
/// - 1D: `f64`
/// - 2D: `[f64; 2]`
/// - 3D: `[f64; 3]`
///
/// # Example
/// ```
/// use dgcom::mesh::Point;
///
/// fn distance_from_origin<P: Point>(p: &P) -> f64 {
///     p.norm()
/// }
///
/// assert!((distance_from_origin(&3.0_f64) - 3.0).abs() < 1e-10);
/// assert!((distance_from_origin(&[3.0, 4.0]) - 5.0).abs() < 1e-10);
/// ```
pub trait Point: Copy + Clone + Debug + Default + Send + Sync + 'static {
    /// Spatial dimension (1, 2, or 3).
    const DIM: usize;

    /// Access coordinate by index.
    ///
    /// # Panics
    /// Panics if `idx >= Self::DIM`.
    fn coord(&self, idx: usize) -> f64;

    /// Create a point from a slice of coordinates.
    ///
    /// # Panics
    /// Panics if `coords.len() < Self::DIM`.
    fn from_slice(coords: &[f64]) -> Self;

    /// Create a point with all coordinates set to zero.
    fn zero() -> Self {
        Self::default()
    }

    // =========================================================================
    // Arithmetic Operations
    // =========================================================================

    /// Add two points component-wise: self + other
    fn add(&self, other: &Self) -> Self {
        let mut coords = [0.0; 3];
        for i in 0..Self::DIM {
            coords[i] = self.coord(i) + other.coord(i);
        }
        Self::from_slice(&coords[..Self::DIM])
    }

    /// Subtract two points component-wise: self - other
    fn sub(&self, other: &Self) -> Self {
        let mut coords = [0.0; 3];
        for i in 0..Self::DIM {
            coords[i] = self.coord(i) - other.coord(i);
        }
        Self::from_slice(&coords[..Self::DIM])
    }

    /// Scale a point by a scalar: c * self
    fn scale(&self, c: f64) -> Self {
        let mut coords = [0.0; 3];
        for i in 0..Self::DIM {
            coords[i] = c * self.coord(i);
        }
        Self::from_slice(&coords[..Self::DIM])
    }

    /// Dot product of two points.
    fn dot(&self, other: &Self) -> f64 {
        let mut sum = 0.0;
        for i in 0..Self::DIM {
            sum += self.coord(i) * other.coord(i);
        }
        sum
    }

    /// Euclidean norm (magnitude) of the point.
    fn norm(&self) -> f64 {
        self.dot(self).sqrt()
    }

    /// Squared Euclidean norm (avoids sqrt for comparisons).
    fn norm_squared(&self) -> f64 {
        self.dot(self)
    }

    /// Distance between two points.
    fn distance(&self, other: &Self) -> f64 {
        self.sub(other).norm()
    }

    /// Squared distance between two points (avoids sqrt for comparisons).
    fn distance_squared(&self, other: &Self) -> f64 {
        self.sub(other).norm_squared()
    }

    /// Normalize to unit length. Returns zero vector if norm is zero.
    fn normalize(&self) -> Self {
        let n = self.norm();
        if n < 1e-14 {
            Self::zero()
        } else {
            self.scale(1.0 / n)
        }
    }

    /// Linear interpolation: self + t * (other - self)
    fn lerp(&self, other: &Self, t: f64) -> Self {
        self.add(&other.sub(self).scale(t))
    }
}

// =============================================================================
// 1D Implementation: f64
// =============================================================================

impl Point for f64 {
    const DIM: usize = 1;

    #[inline]
    fn coord(&self, idx: usize) -> f64 {
        assert!(idx == 0, "1D point has only index 0, got {idx}");
        *self
    }

    #[inline]
    fn from_slice(coords: &[f64]) -> Self {
        coords[0]
    }

    #[inline]
    fn zero() -> Self {
        0.0
    }
}

// =============================================================================
// 2D Implementation: [f64; 2]
// =============================================================================

impl Point for [f64; 2] {
    const DIM: usize = 2;

    #[inline]
    fn coord(&self, idx: usize) -> f64 {
        self[idx]
    }

    #[inline]
    fn from_slice(coords: &[f64]) -> Self {
        [coords[0], coords[1]]
    }

    #[inline]
    fn zero() -> Self {
        [0.0, 0.0]
    }
}

// =============================================================================
// 3D Implementation: [f64; 3]
// =============================================================================

impl Point for [f64; 3] {
    const DIM: usize = 3;

    #[inline]
    fn coord(&self, idx: usize) -> f64 {
        self[idx]
    }

    #[inline]
    fn from_slice(coords: &[f64]) -> Self {
        [coords[0], coords[1], coords[2]]
    }

    #[inline]
    fn zero() -> Self {
        [0.0, 0.0, 0.0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-14;

    #[test]
    fn test_point_1d() {
        let p: f64 = 3.5;
        assert_eq!(f64::DIM, 1);
        assert_eq!(p.coord(0), 3.5);
        assert_eq!(f64::from_slice(&[2.0]), 2.0);
        assert_eq!(f64::zero(), 0.0);
    }

    #[test]
    fn test_point_2d_array() {
        let p: [f64; 2] = [1.0, 2.0];
        assert_eq!(<[f64; 2]>::DIM, 2);
        assert_eq!(p.coord(0), 1.0);
        assert_eq!(p.coord(1), 2.0);
        assert_eq!(<[f64; 2]>::from_slice(&[3.0, 4.0]), [3.0, 4.0]);
        assert_eq!(<[f64; 2]>::zero(), [0.0, 0.0]);
    }

    #[test]
    fn test_point_3d() {
        let p: [f64; 3] = [1.0, 2.0, 3.0];
        assert_eq!(<[f64; 3]>::DIM, 3);
        assert_eq!(p.coord(0), 1.0);
        assert_eq!(p.coord(1), 2.0);
        assert_eq!(p.coord(2), 3.0);
        assert_eq!(<[f64; 3]>::from_slice(&[4.0, 5.0, 6.0]), [4.0, 5.0, 6.0]);
    }

    #[test]
    #[should_panic]
    fn test_point_1d_out_of_bounds() {
        let p: f64 = 1.0;
        let _ = p.coord(1);
    }

    #[test]
    #[should_panic]
    fn test_point_2d_out_of_bounds() {
        let p: [f64; 2] = [1.0, 2.0];
        let _ = p.coord(2);
    }

    // =========================================================================
    // Arithmetic Operation Tests
    // =========================================================================

    #[test]
    fn test_add_1d() {
        let a: f64 = 3.0;
        let b: f64 = 4.0;
        assert!((a.add(&b) - 7.0).abs() < TOL);
    }

    #[test]
    fn test_add_2d() {
        let a: [f64; 2] = [1.0, 2.0];
        let b: [f64; 2] = [3.0, 4.0];
        let c = a.add(&b);
        assert!((c[0] - 4.0).abs() < TOL);
        assert!((c[1] - 6.0).abs() < TOL);
    }

    #[test]
    fn test_add_3d() {
        let a: [f64; 3] = [1.0, 2.0, 3.0];
        let b: [f64; 3] = [4.0, 5.0, 6.0];
        let c = a.add(&b);
        assert!((c[0] - 5.0).abs() < TOL);
        assert!((c[1] - 7.0).abs() < TOL);
        assert!((c[2] - 9.0).abs() < TOL);
    }

    #[test]
    fn test_sub_2d() {
        let a: [f64; 2] = [5.0, 7.0];
        let b: [f64; 2] = [2.0, 3.0];
        let c = a.sub(&b);
        assert!((c[0] - 3.0).abs() < TOL);
        assert!((c[1] - 4.0).abs() < TOL);
    }

    #[test]
    fn test_scale_2d() {
        let a: [f64; 2] = [2.0, 3.0];
        let c = a.scale(2.5);
        assert!((c[0] - 5.0).abs() < TOL);
        assert!((c[1] - 7.5).abs() < TOL);
    }

    #[test]
    fn test_dot_2d() {
        let a: [f64; 2] = [1.0, 2.0];
        let b: [f64; 2] = [3.0, 4.0];
        assert!((a.dot(&b) - 11.0).abs() < TOL); // 1*3 + 2*4 = 11
    }

    #[test]
    fn test_dot_3d() {
        let a: [f64; 3] = [1.0, 2.0, 3.0];
        let b: [f64; 3] = [4.0, 5.0, 6.0];
        assert!((a.dot(&b) - 32.0).abs() < TOL); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_norm_2d() {
        let p: [f64; 2] = [3.0, 4.0];
        assert!((p.norm() - 5.0).abs() < TOL);
        assert!((p.norm_squared() - 25.0).abs() < TOL);
    }

    #[test]
    fn test_norm_3d() {
        let p: [f64; 3] = [2.0, 3.0, 6.0];
        assert!((p.norm() - 7.0).abs() < TOL); // sqrt(4+9+36) = 7
    }

    #[test]
    fn test_distance_2d() {
        let a: [f64; 2] = [0.0, 0.0];
        let b: [f64; 2] = [3.0, 4.0];
        assert!((a.distance(&b) - 5.0).abs() < TOL);
        assert!((a.distance_squared(&b) - 25.0).abs() < TOL);
    }

    #[test]
    fn test_normalize_2d() {
        let p: [f64; 2] = [3.0, 4.0];
        let n = p.normalize();
        assert!((n[0] - 0.6).abs() < TOL);
        assert!((n[1] - 0.8).abs() < TOL);
        assert!((n.norm() - 1.0).abs() < TOL);
    }

    #[test]
    fn test_normalize_zero() {
        let p: [f64; 2] = [0.0, 0.0];
        let n = p.normalize();
        assert!((n[0]).abs() < TOL);
        assert!((n[1]).abs() < TOL);
    }

    #[test]
    fn test_lerp_2d() {
        let a: [f64; 2] = [0.0, 0.0];
        let b: [f64; 2] = [10.0, 20.0];

        let mid = a.lerp(&b, 0.5);
        assert!((mid[0] - 5.0).abs() < TOL);
        assert!((mid[1] - 10.0).abs() < TOL);

        let quarter = a.lerp(&b, 0.25);
        assert!((quarter[0] - 2.5).abs() < TOL);
        assert!((quarter[1] - 5.0).abs() < TOL);

        // t=0 should give a, t=1 should give b
        let at_zero = a.lerp(&b, 0.0);
        let at_one = a.lerp(&b, 1.0);
        assert!((at_zero[0] - a[0]).abs() < TOL);
        assert!((at_one[0] - b[0]).abs() < TOL);
    }
}

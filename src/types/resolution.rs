//! 2D mesh resolution types.

use std::fmt;

/// 2D mesh resolution (number of elements in each direction).
///
/// Provides a strongly-typed way to specify mesh resolution,
/// preventing mix-ups between nx/ny and other integer parameters.
///
/// # Example
///
/// ```
/// use dgcom::types::Resolution2D;
///
/// let res = Resolution2D::new(100, 50);
/// assert_eq!(res.nx(), 100);
/// assert_eq!(res.ny(), 50);
/// assert_eq!(res.total_elements(), 5000);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Resolution2D {
    /// Number of elements in x-direction
    nx: usize,
    /// Number of elements in y-direction
    ny: usize,
}

impl Resolution2D {
    /// Create a new resolution specification.
    ///
    /// # Arguments
    ///
    /// * `nx` - Number of elements in x-direction
    /// * `ny` - Number of elements in y-direction
    ///
    /// # Panics
    ///
    /// Panics if either `nx` or `ny` is zero.
    pub fn new(nx: usize, ny: usize) -> Self {
        assert!(nx > 0, "nx must be positive, got {}", nx);
        assert!(ny > 0, "ny must be positive, got {}", ny);
        Self { nx, ny }
    }

    /// Create a square resolution (same in both directions).
    pub fn square(n: usize) -> Self {
        Self::new(n, n)
    }

    /// Number of elements in x-direction.
    #[inline]
    pub fn nx(&self) -> usize {
        self.nx
    }

    /// Number of elements in y-direction.
    #[inline]
    pub fn ny(&self) -> usize {
        self.ny
    }

    /// Total number of elements.
    #[inline]
    pub fn total_elements(&self) -> usize {
        self.nx * self.ny
    }

    /// Number of vertices in a structured grid.
    #[inline]
    pub fn total_vertices(&self) -> usize {
        (self.nx + 1) * (self.ny + 1)
    }

    /// Aspect ratio of resolution (nx / ny).
    #[inline]
    pub fn aspect_ratio(&self) -> f64 {
        self.nx as f64 / self.ny as f64
    }

    /// Return as tuple (nx, ny).
    #[inline]
    pub fn as_tuple(&self) -> (usize, usize) {
        (self.nx, self.ny)
    }
}

impl fmt::Display for Resolution2D {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}×{}", self.nx, self.ny)
    }
}

impl From<(usize, usize)> for Resolution2D {
    fn from((nx, ny): (usize, usize)) -> Self {
        Self::new(nx, ny)
    }
}

impl From<Resolution2D> for (usize, usize) {
    fn from(res: Resolution2D) -> Self {
        (res.nx, res.ny)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolution_creation() {
        let r = Resolution2D::new(100, 50);
        assert_eq!(r.nx(), 100);
        assert_eq!(r.ny(), 50);
    }

    #[test]
    fn test_square_resolution() {
        let r = Resolution2D::square(32);
        assert_eq!(r.nx(), 32);
        assert_eq!(r.ny(), 32);
    }

    #[test]
    fn test_total_elements() {
        let r = Resolution2D::new(10, 5);
        assert_eq!(r.total_elements(), 50);
        assert_eq!(r.total_vertices(), 66); // 11 × 6
    }

    #[test]
    fn test_from_tuple() {
        let r: Resolution2D = (20, 10).into();
        assert_eq!(r.nx(), 20);
        assert_eq!(r.ny(), 10);
    }

    #[test]
    #[should_panic(expected = "nx must be positive")]
    fn test_zero_nx() {
        Resolution2D::new(0, 10);
    }

    #[test]
    #[should_panic(expected = "ny must be positive")]
    fn test_zero_ny() {
        Resolution2D::new(10, 0);
    }
}

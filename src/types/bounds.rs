//! 2D domain bounds.

use std::fmt;

/// 2D rectangular domain bounds.
///
/// Stores the spatial extent of a rectangular domain with
/// clear semantics for each boundary.
///
/// # Example
///
/// ```
/// use dgcom::types::Bounds2D;
///
/// // Norwegian coastal domain (approximate)
/// let bounds = Bounds2D::new(
///     0.0,    // x_min (west)
///     100e3,  // x_max (east)
///     0.0,    // y_min (south)
///     50e3,   // y_max (north)
/// );
///
/// assert_eq!(bounds.width(), 100e3);
/// assert_eq!(bounds.height(), 50e3);
/// assert_eq!(bounds.center(), (50e3, 25e3));
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bounds2D {
    /// Minimum x-coordinate (western boundary)
    pub x_min: f64,
    /// Maximum x-coordinate (eastern boundary)
    pub x_max: f64,
    /// Minimum y-coordinate (southern boundary)
    pub y_min: f64,
    /// Maximum y-coordinate (northern boundary)
    pub y_max: f64,
}

impl Bounds2D {
    /// Create new domain bounds.
    ///
    /// # Arguments
    ///
    /// * `x_min` - Western boundary (minimum x)
    /// * `x_max` - Eastern boundary (maximum x)
    /// * `y_min` - Southern boundary (minimum y)
    /// * `y_max` - Northern boundary (maximum y)
    ///
    /// # Panics
    ///
    /// Panics if `x_max <= x_min` or `y_max <= y_min`.
    pub fn new(x_min: f64, x_max: f64, y_min: f64, y_max: f64) -> Self {
        assert!(
            x_max > x_min,
            "x_max ({}) must be greater than x_min ({})",
            x_max,
            x_min
        );
        assert!(
            y_max > y_min,
            "y_max ({}) must be greater than y_min ({})",
            y_max,
            y_min
        );

        Self {
            x_min,
            x_max,
            y_min,
            y_max,
        }
    }

    /// Create a unit square [0, 1] × [0, 1].
    pub fn unit_square() -> Self {
        Self::new(0.0, 1.0, 0.0, 1.0)
    }

    /// Create a square domain centered at origin.
    pub fn square(half_width: f64) -> Self {
        Self::new(-half_width, half_width, -half_width, half_width)
    }

    /// Domain width (x_max - x_min).
    #[inline]
    pub fn width(&self) -> f64 {
        self.x_max - self.x_min
    }

    /// Domain height (y_max - y_min).
    #[inline]
    pub fn height(&self) -> f64 {
        self.y_max - self.y_min
    }

    /// Domain area.
    #[inline]
    pub fn area(&self) -> f64 {
        self.width() * self.height()
    }

    /// Domain center point.
    #[inline]
    pub fn center(&self) -> (f64, f64) {
        (
            (self.x_min + self.x_max) / 2.0,
            (self.y_min + self.y_max) / 2.0,
        )
    }

    /// Aspect ratio (width / height).
    #[inline]
    pub fn aspect_ratio(&self) -> f64 {
        self.width() / self.height()
    }

    /// Check if a point is inside the domain (inclusive).
    #[inline]
    pub fn contains(&self, x: f64, y: f64) -> bool {
        x >= self.x_min && x <= self.x_max && y >= self.y_min && y <= self.y_max
    }

    /// Expand bounds by a factor (centered).
    pub fn expand(&self, factor: f64) -> Self {
        let (cx, cy) = self.center();
        let half_w = self.width() / 2.0 * factor;
        let half_h = self.height() / 2.0 * factor;
        Self::new(cx - half_w, cx + half_w, cy - half_h, cy + half_h)
    }

    /// Return bounds as tuple (x_min, x_max, y_min, y_max).
    #[inline]
    pub fn as_tuple(&self) -> (f64, f64, f64, f64) {
        (self.x_min, self.x_max, self.y_min, self.y_max)
    }
}

impl fmt::Display for Bounds2D {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{:.2}, {:.2}] × [{:.2}, {:.2}]",
            self.x_min, self.x_max, self.y_min, self.y_max
        )
    }
}

impl Default for Bounds2D {
    fn default() -> Self {
        Self::unit_square()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bounds_creation() {
        let b = Bounds2D::new(0.0, 100.0, 0.0, 50.0);
        assert_eq!(b.x_min, 0.0);
        assert_eq!(b.x_max, 100.0);
        assert_eq!(b.y_min, 0.0);
        assert_eq!(b.y_max, 50.0);
    }

    #[test]
    fn test_dimensions() {
        let b = Bounds2D::new(0.0, 100.0, 0.0, 50.0);
        assert_eq!(b.width(), 100.0);
        assert_eq!(b.height(), 50.0);
        assert_eq!(b.area(), 5000.0);
        assert_eq!(b.aspect_ratio(), 2.0);
    }

    #[test]
    fn test_center() {
        let b = Bounds2D::new(0.0, 100.0, 0.0, 50.0);
        assert_eq!(b.center(), (50.0, 25.0));
    }

    #[test]
    fn test_contains() {
        let b = Bounds2D::new(0.0, 100.0, 0.0, 50.0);
        assert!(b.contains(50.0, 25.0));
        assert!(b.contains(0.0, 0.0));
        assert!(b.contains(100.0, 50.0));
        assert!(!b.contains(-1.0, 25.0));
        assert!(!b.contains(50.0, 51.0));
    }

    #[test]
    fn test_unit_square() {
        let b = Bounds2D::unit_square();
        assert_eq!(b.width(), 1.0);
        assert_eq!(b.height(), 1.0);
    }

    #[test]
    #[should_panic(expected = "x_max")]
    fn test_invalid_x() {
        Bounds2D::new(100.0, 0.0, 0.0, 50.0);
    }

    #[test]
    #[should_panic(expected = "y_max")]
    fn test_invalid_y() {
        Bounds2D::new(0.0, 100.0, 50.0, 0.0);
    }
}

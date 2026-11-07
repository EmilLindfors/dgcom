//! Builder pattern for 2D mesh construction.
//!
//! Provides a fluent API for creating structured quadrilateral meshes
//! with various boundary conditions.
//!
//! # Example
//!
//! ```
//! use dgcom::mesh::{Mesh2DBuilder, BoundaryTag};
//!
//! // Simple uniform mesh with default (Wall) boundaries
//! let mesh = Mesh2DBuilder::new(0.0, 100.0, 0.0, 50.0)
//!     .with_resolution(10, 5)
//!     .build();
//!
//! // Mesh with per-side boundary conditions
//! let mesh = Mesh2DBuilder::new(0.0, 100.0, 0.0, 50.0)
//!     .with_resolution(20, 10)
//!     .with_side_bcs(
//!         BoundaryTag::Wall,  // south
//!         BoundaryTag::Open,  // east
//!         BoundaryTag::Wall,  // north
//!         BoundaryTag::River, // west
//!     )
//!     .build();
//!
//! // Periodic channel (periodic in x, wall in y)
//! let mesh = Mesh2DBuilder::new(0.0, 1.0, 0.0, 0.1)
//!     .with_resolution(100, 10)
//!     .periodic_x()
//!     .build();
//!
//! // Fully periodic domain
//! let mesh = Mesh2DBuilder::new(0.0, 1.0, 0.0, 1.0)
//!     .with_resolution(32, 32)
//!     .fully_periodic()
//!     .build();
//! ```

use super::mesh2d::Mesh2D;
use crate::mesh::data::BoundaryTag;
use crate::types::{Bounds2D, Resolution2D, SideBoundaries};

/// Boundary configuration for the mesh.
#[derive(Clone, Debug)]
pub enum BoundaryConfig {
    /// Same boundary tag on all sides.
    Uniform(BoundaryTag),
    /// Different boundary tags for each side: [south, east, north, west].
    PerSide {
        south: BoundaryTag,
        east: BoundaryTag,
        north: BoundaryTag,
        west: BoundaryTag,
    },
    /// Periodic in x-direction, walls in y-direction.
    PeriodicX,
    /// Fully periodic in both directions.
    FullyPeriodic,
}

impl Default for BoundaryConfig {
    fn default() -> Self {
        BoundaryConfig::Uniform(BoundaryTag::Wall)
    }
}

/// Builder for creating 2D structured quadrilateral meshes.
///
/// # Example
///
/// ```
/// use dgcom::mesh::{Mesh2DBuilder, BoundaryTag};
///
/// let mesh = Mesh2DBuilder::new(0.0, 100.0, 0.0, 50.0)
///     .with_resolution(20, 10)
///     .with_uniform_bc(BoundaryTag::Open)
///     .build();
///
/// assert_eq!(mesh.n_elements, 200);
/// ```
#[derive(Clone, Debug)]
pub struct Mesh2DBuilder {
    /// Domain x-bounds
    x0: f64,
    x1: f64,
    /// Domain y-bounds
    y0: f64,
    y1: f64,
    /// Number of elements in x-direction
    nx: usize,
    /// Number of elements in y-direction
    ny: usize,
    /// Boundary configuration
    bc_config: BoundaryConfig,
}

impl Mesh2DBuilder {
    /// Create a new mesh builder for a rectangular domain.
    ///
    /// # Arguments
    ///
    /// * `x0`, `x1` - x-coordinate bounds (x0 < x1)
    /// * `y0`, `y1` - y-coordinate bounds (y0 < y1)
    ///
    /// Default resolution is 1x1 elements with Wall boundaries.
    pub fn new(x0: f64, x1: f64, y0: f64, y1: f64) -> Self {
        assert!(x1 > x0, "x1 must be greater than x0");
        assert!(y1 > y0, "y1 must be greater than y0");

        Self {
            x0,
            x1,
            y0,
            y1,
            nx: 1,
            ny: 1,
            bc_config: BoundaryConfig::default(),
        }
    }

    /// Create a builder for a unit square [0,1] × [0,1].
    pub fn unit_square() -> Self {
        Self::new(0.0, 1.0, 0.0, 1.0)
    }

    // =========================================================================
    // Type-safe constructors and methods using newtypes
    // =========================================================================

    /// Create a new mesh builder from typed bounds.
    ///
    /// # Example
    ///
    /// ```
    /// use dgcom::mesh::Mesh2DBuilder;
    /// use dgcom::types::Bounds2D;
    ///
    /// let bounds = Bounds2D::new(0.0, 100e3, 0.0, 50e3);
    /// let mesh = Mesh2DBuilder::from_bounds(bounds)
    ///     .with_resolution(100, 50)
    ///     .build();
    /// ```
    pub fn from_bounds(bounds: Bounds2D) -> Self {
        Self::new(bounds.x_min, bounds.x_max, bounds.y_min, bounds.y_max)
    }

    /// Set mesh resolution using typed Resolution2D.
    ///
    /// # Example
    ///
    /// ```
    /// use dgcom::mesh::Mesh2DBuilder;
    /// use dgcom::types::Resolution2D;
    ///
    /// let mesh = Mesh2DBuilder::unit_square()
    ///     .with_grid(Resolution2D::new(32, 32))
    ///     .build();
    /// ```
    pub fn with_grid(mut self, res: Resolution2D) -> Self {
        self.nx = res.nx();
        self.ny = res.ny();
        self
    }

    /// Set boundary conditions using typed SideBoundaries.
    ///
    /// # Example
    ///
    /// ```
    /// use dgcom::mesh::{Mesh2DBuilder, BoundaryTag};
    /// use dgcom::types::SideBoundaries;
    ///
    /// let bcs = SideBoundaries::new(
    ///     BoundaryTag::Wall,   // south
    ///     BoundaryTag::Open,   // east
    ///     BoundaryTag::Wall,   // north
    ///     BoundaryTag::River,  // west
    /// );
    ///
    /// let mesh = Mesh2DBuilder::unit_square()
    ///     .with_resolution(10, 10)
    ///     .with_boundaries(bcs)
    ///     .build();
    /// ```
    pub fn with_boundaries(mut self, bcs: SideBoundaries<BoundaryTag>) -> Self {
        self.bc_config = BoundaryConfig::PerSide {
            south: bcs.south,
            east: bcs.east,
            north: bcs.north,
            west: bcs.west,
        };
        self
    }

    // =========================================================================
    // Original methods (for backwards compatibility)
    // =========================================================================

    /// Set the mesh resolution.
    ///
    /// # Arguments
    ///
    /// * `nx` - Number of elements in x-direction
    /// * `ny` - Number of elements in y-direction
    pub fn with_resolution(mut self, nx: usize, ny: usize) -> Self {
        assert!(
            nx > 0 && ny > 0,
            "Need at least one element in each direction"
        );
        self.nx = nx;
        self.ny = ny;
        self
    }

    /// Set uniform boundary condition on all sides.
    pub fn with_uniform_bc(mut self, tag: BoundaryTag) -> Self {
        self.bc_config = BoundaryConfig::Uniform(tag);
        self
    }

    /// Set different boundary conditions for each side.
    ///
    /// # Arguments
    ///
    /// * `south` - Boundary tag for y = y0 (bottom)
    /// * `east` - Boundary tag for x = x1 (right)
    /// * `north` - Boundary tag for y = y1 (top)
    /// * `west` - Boundary tag for x = x0 (left)
    pub fn with_side_bcs(
        mut self,
        south: BoundaryTag,
        east: BoundaryTag,
        north: BoundaryTag,
        west: BoundaryTag,
    ) -> Self {
        self.bc_config = BoundaryConfig::PerSide {
            south,
            east,
            north,
            west,
        };
        self
    }

    /// Make the mesh periodic in x-direction (channel flow).
    ///
    /// East and west boundaries become connected, while
    /// north and south remain walls.
    pub fn periodic_x(mut self) -> Self {
        self.bc_config = BoundaryConfig::PeriodicX;
        self
    }

    /// Make the mesh fully periodic in both directions.
    pub fn fully_periodic(mut self) -> Self {
        self.bc_config = BoundaryConfig::FullyPeriodic;
        self
    }

    /// Build the mesh with the specified configuration.
    pub fn build(self) -> Mesh2D {
        match self.bc_config {
            BoundaryConfig::Uniform(tag) => Mesh2D::uniform_rectangle_with_bc(
                self.x0, self.x1, self.y0, self.y1, self.nx, self.ny, tag,
            ),
            BoundaryConfig::PerSide {
                south,
                east,
                north,
                west,
            } => Mesh2D::uniform_rectangle_with_sides(
                self.x0,
                self.x1,
                self.y0,
                self.y1,
                self.nx,
                self.ny,
                [south, east, north, west],
            ),
            BoundaryConfig::PeriodicX => {
                Mesh2D::channel_periodic_x(self.x0, self.x1, self.y0, self.y1, self.nx, self.ny)
            }
            BoundaryConfig::FullyPeriodic => {
                Mesh2D::uniform_periodic(self.x0, self.x1, self.y0, self.y1, self.nx, self.ny)
            }
        }
    }

    // =========================================================================
    // Accessor methods for inspection
    // =========================================================================

    /// Get the domain bounds.
    pub fn bounds(&self) -> (f64, f64, f64, f64) {
        (self.x0, self.x1, self.y0, self.y1)
    }

    /// Get the resolution.
    pub fn resolution(&self) -> (usize, usize) {
        (self.nx, self.ny)
    }

    /// Get the element size.
    pub fn element_size(&self) -> (f64, f64) {
        let dx = (self.x1 - self.x0) / self.nx as f64;
        let dy = (self.y1 - self.y0) / self.ny as f64;
        (dx, dy)
    }

    /// Get the boundary configuration.
    pub fn bc_config(&self) -> &BoundaryConfig {
        &self.bc_config
    }

    /// Get the domain bounds as typed Bounds2D.
    pub fn get_bounds(&self) -> Bounds2D {
        Bounds2D::new(self.x0, self.x1, self.y0, self.y1)
    }

    /// Get the resolution as typed Resolution2D.
    pub fn get_resolution(&self) -> Resolution2D {
        Resolution2D::new(self.nx, self.ny)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_default() {
        let mesh = Mesh2DBuilder::new(0.0, 1.0, 0.0, 1.0)
            .with_resolution(2, 2)
            .build();

        assert_eq!(mesh.n_elements, 4);
        assert_eq!(mesh.n_vertices, 9);
    }

    #[test]
    fn test_builder_uniform_bc() {
        let mesh = Mesh2DBuilder::new(0.0, 10.0, 0.0, 5.0)
            .with_resolution(10, 5)
            .with_uniform_bc(BoundaryTag::Open)
            .build();

        assert_eq!(mesh.n_elements, 50);

        // All boundary edges should be Open
        for edge in &mesh.edges {
            if edge.is_boundary() {
                assert_eq!(edge.boundary_tag, Some(BoundaryTag::Open));
            }
        }
    }

    #[test]
    fn test_builder_side_bcs() {
        let mesh = Mesh2DBuilder::new(0.0, 1.0, 0.0, 1.0)
            .with_resolution(2, 2)
            .with_side_bcs(
                BoundaryTag::Wall,  // south
                BoundaryTag::Open,  // east
                BoundaryTag::River, // north
                BoundaryTag::Wall,  // west
            )
            .build();

        assert_eq!(mesh.n_elements, 4);
    }

    #[test]
    fn test_builder_periodic_x() {
        let mesh = Mesh2DBuilder::new(0.0, 1.0, 0.0, 0.5)
            .with_resolution(4, 2)
            .periodic_x()
            .build();

        assert_eq!(mesh.n_elements, 8);
        // Should have fewer boundary edges due to periodic connection
        assert!(mesh.n_boundary_edges < 12); // Would be 12 for non-periodic
    }

    #[test]
    fn test_builder_fully_periodic() {
        let mesh = Mesh2DBuilder::new(0.0, 1.0, 0.0, 1.0)
            .with_resolution(4, 4)
            .fully_periodic()
            .build();

        assert_eq!(mesh.n_elements, 16);
        // No boundary edges for fully periodic
        assert_eq!(mesh.n_boundary_edges, 0);
    }

    #[test]
    fn test_builder_unit_square() {
        let mesh = Mesh2DBuilder::unit_square().with_resolution(10, 10).build();

        assert_eq!(mesh.n_elements, 100);
    }

    #[test]
    fn test_builder_accessors() {
        let builder = Mesh2DBuilder::new(0.0, 100.0, 0.0, 50.0).with_resolution(20, 10);

        assert_eq!(builder.bounds(), (0.0, 100.0, 0.0, 50.0));
        assert_eq!(builder.resolution(), (20, 10));
        assert_eq!(builder.element_size(), (5.0, 5.0));
    }

    #[test]
    #[should_panic(expected = "x1 must be greater than x0")]
    fn test_builder_invalid_x_bounds() {
        Mesh2DBuilder::new(1.0, 0.0, 0.0, 1.0);
    }

    #[test]
    #[should_panic(expected = "y1 must be greater than y0")]
    fn test_builder_invalid_y_bounds() {
        Mesh2DBuilder::new(0.0, 1.0, 1.0, 0.0);
    }

    #[test]
    #[should_panic(expected = "Need at least one element")]
    fn test_builder_zero_resolution() {
        Mesh2DBuilder::new(0.0, 1.0, 0.0, 1.0)
            .with_resolution(0, 1)
            .build();
    }

    // =========================================================================
    // Tests for typed API
    // =========================================================================

    #[test]
    fn test_from_bounds() {
        use crate::types::Bounds2D;

        let bounds = Bounds2D::new(0.0, 100.0, 0.0, 50.0);
        let mesh = Mesh2DBuilder::from_bounds(bounds)
            .with_resolution(10, 5)
            .build();

        assert_eq!(mesh.n_elements, 50);
    }

    #[test]
    fn test_with_grid() {
        use crate::types::Resolution2D;

        let mesh = Mesh2DBuilder::unit_square()
            .with_grid(Resolution2D::new(8, 8))
            .build();

        assert_eq!(mesh.n_elements, 64);
    }

    #[test]
    fn test_with_boundaries() {
        use crate::types::SideBoundaries;

        let bcs = SideBoundaries::new(
            BoundaryTag::Wall,
            BoundaryTag::Open,
            BoundaryTag::Wall,
            BoundaryTag::River,
        );

        let mesh = Mesh2DBuilder::unit_square()
            .with_resolution(4, 4)
            .with_boundaries(bcs)
            .build();

        assert_eq!(mesh.n_elements, 16);
    }

    #[test]
    fn test_typed_accessors() {
        let builder = Mesh2DBuilder::new(0.0, 100.0, 0.0, 50.0).with_resolution(20, 10);

        let bounds = builder.get_bounds();
        assert_eq!(bounds.x_min, 0.0);
        assert_eq!(bounds.x_max, 100.0);
        assert_eq!(bounds.y_min, 0.0);
        assert_eq!(bounds.y_max, 50.0);

        let res = builder.get_resolution();
        assert_eq!(res.nx(), 20);
        assert_eq!(res.ny(), 10);
    }

    #[test]
    fn test_full_typed_api() {
        use crate::types::{Bounds2D, Resolution2D, SideBoundaries};

        let bounds = Bounds2D::new(0.0, 1000.0, 0.0, 500.0);
        let res = Resolution2D::new(100, 50);
        let bcs = SideBoundaries::new(
            BoundaryTag::Wall,
            BoundaryTag::Open,
            BoundaryTag::Wall,
            BoundaryTag::Open,
        );

        let mesh = Mesh2DBuilder::from_bounds(bounds)
            .with_grid(res)
            .with_boundaries(bcs)
            .build();

        assert_eq!(mesh.n_elements, 5000);
    }
}

//! Abstract mesh traits for dimension-independent DG solvers.
//!
//! This module provides a layered trait hierarchy for meshes:
//!
//! - [`MeshTopology`]: Element and face connectivity
//! - [`MeshGeometry`]: Coordinate mappings and Jacobians
//! - [`MeshGeometryExt`]: Extended geometry with face normals (2D/3D)
//! - [`MeshCFL`]: CFL time step computation (blanket impl)
//!
//! The design follows patterns from std (Iterator, Read) and well-known crates
//! (nalgebra, petgraph) with associated types for dimension-specific data.
//!
//! # Example
//! ```ignore
//! use dgcom::mesh::{MeshTopology, MeshGeometry, FaceConnection};
//!
//! fn print_mesh_info<M: MeshGeometry>(mesh: &M) {
//!     println!("Elements: {}", mesh.n_elements());
//!     println!("h_min: {}", mesh.h_min());
//!
//!     for k in 0..mesh.n_elements() {
//!         for f in 0..M::FACES_PER_ELEMENT {
//!             match mesh.face_connection(k, f) {
//!                 FaceConnection::Interior(n) => {
//!                     println!("  Element {} face {} -> element {}", k, f, n.element);
//!                 }
//!                 FaceConnection::Boundary(tag) => {
//!                     println!("  Element {} face {} -> boundary {:?}", k, f, tag);
//!                 }
//!             }
//!         }
//!     }
//! }
//! ```

use std::fmt::Debug;

use super::point::Point;

// =============================================================================
// Supporting Types
// =============================================================================

/// Information about a neighbor element across a face.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Neighbor {
    /// Index of the neighboring element.
    pub element: usize,
    /// Local face index on the neighboring element that shares this interface.
    pub face: usize,
}

/// Result of querying face connectivity.
///
/// A face is either interior (shared with a neighbor) or on the boundary.
#[derive(Clone, Copy, Debug)]
pub enum FaceConnection<Tag> {
    /// Interior face connecting to a neighbor element.
    Interior(Neighbor),
    /// Boundary face with an associated tag.
    Boundary(Tag),
}

impl<Tag> FaceConnection<Tag> {
    /// Returns `true` if this is an interior face.
    #[inline]
    pub fn is_interior(&self) -> bool {
        matches!(self, FaceConnection::Interior(_))
    }

    /// Returns `true` if this is a boundary face.
    #[inline]
    pub fn is_boundary(&self) -> bool {
        matches!(self, FaceConnection::Boundary(_))
    }

    /// Returns the neighbor if this is an interior face.
    #[inline]
    pub fn neighbor(&self) -> Option<&Neighbor> {
        match self {
            FaceConnection::Interior(n) => Some(n),
            FaceConnection::Boundary(_) => None,
        }
    }

    /// Returns the boundary tag if this is a boundary face.
    #[inline]
    pub fn boundary_tag(&self) -> Option<&Tag> {
        match self {
            FaceConnection::Interior(_) => None,
            FaceConnection::Boundary(t) => Some(t),
        }
    }
}

// =============================================================================
// MeshTopology Trait
// =============================================================================

/// Base trait providing mesh topology (elements, faces, connectivity).
///
/// This is the fundamental trait that all meshes implement. It provides
/// access to element counts and face connectivity without geometric details.
///
/// # Associated Types
/// - `Coord`: Physical coordinate type (`f64` for 1D, `[f64; 2]` for 2D)
/// - `RefCoord`: Reference coordinate type (usually same as `Coord`)
/// - `BoundaryTag`: Type for boundary condition tags
///
/// # Associated Constants
/// - `FACES_PER_ELEMENT`: Number of faces per element (2 for 1D, 4 for 2D quads)
pub trait MeshTopology: Send + Sync {
    /// Physical coordinate type.
    type Coord: Point;

    /// Reference coordinate type.
    type RefCoord: Point;

    /// Boundary tag type for labeling boundary faces.
    type BoundaryTag: Copy + Clone + Debug + Default + Send + Sync;

    /// Number of faces per element.
    ///
    /// - 1D intervals: 2 faces (left, right)
    /// - 2D quadrilaterals: 4 faces
    /// - 2D triangles: 3 faces
    /// - 3D hexahedra: 6 faces
    const FACES_PER_ELEMENT: usize;

    /// Total number of elements in the mesh.
    fn n_elements(&self) -> usize;

    /// Total number of unique faces (edges in 2D, faces in 3D).
    fn n_faces(&self) -> usize;

    /// Number of boundary faces.
    fn n_boundary_faces(&self) -> usize;

    /// Query connectivity across a face.
    ///
    /// Returns [`FaceConnection::Interior`] with neighbor information if the face
    /// is shared with another element, or [`FaceConnection::Boundary`] with a tag
    /// if the face is on the domain boundary.
    ///
    /// # Arguments
    /// * `element` - Element index
    /// * `local_face` - Local face index (0..FACES_PER_ELEMENT)
    fn face_connection(
        &self,
        element: usize,
        local_face: usize,
    ) -> FaceConnection<Self::BoundaryTag>;

    /// Get neighbor element across a face, if interior.
    ///
    /// This is a convenience method; the default implementation uses `face_connection`.
    #[inline]
    fn neighbor(&self, element: usize, local_face: usize) -> Option<Neighbor> {
        match self.face_connection(element, local_face) {
            FaceConnection::Interior(n) => Some(n),
            FaceConnection::Boundary(_) => None,
        }
    }

    /// Check if a face is on the boundary.
    #[inline]
    fn is_boundary(&self, element: usize, local_face: usize) -> bool {
        matches!(
            self.face_connection(element, local_face),
            FaceConnection::Boundary(_)
        )
    }

    /// Get boundary tag for a face, if on boundary.
    #[inline]
    fn boundary_tag(&self, element: usize, local_face: usize) -> Option<Self::BoundaryTag> {
        match self.face_connection(element, local_face) {
            FaceConnection::Boundary(tag) => Some(tag),
            FaceConnection::Interior(_) => None,
        }
    }

    /// Check if the mesh has periodic boundaries.
    fn is_periodic(&self) -> bool {
        false
    }
}

// =============================================================================
// MeshGeometry Trait
// =============================================================================

/// Geometric operations: coordinate mappings and Jacobians.
///
/// This trait extends [`MeshTopology`] with methods for mapping between
/// reference and physical coordinates, computing Jacobians, and querying
/// element sizes.
pub trait MeshGeometry: MeshTopology {
    /// Map reference coordinates to physical coordinates for an element.
    ///
    /// # Arguments
    /// * `element` - Element index
    /// * `ref_coord` - Coordinates in reference space (e.g., r ∈ [-1, 1] for 1D)
    fn reference_to_physical(&self, element: usize, ref_coord: Self::RefCoord) -> Self::Coord;

    /// Map physical coordinates to reference coordinates for an element.
    ///
    /// For curved elements, this may require Newton iteration.
    fn physical_to_reference(&self, element: usize, coord: Self::Coord) -> Self::RefCoord;

    /// Get the Jacobian determinant for an element.
    ///
    /// For affine (straight-sided) elements, this is constant per element.
    /// For 1D: J = h/2 where h is the element size.
    fn jacobian_det(&self, element: usize) -> f64;

    /// Get the inverse Jacobian determinant for an element.
    #[inline]
    fn jacobian_det_inv(&self, element: usize) -> f64 {
        1.0 / self.jacobian_det(element)
    }

    /// Get the minimum element size (diameter) in the mesh.
    ///
    /// Used for CFL time step computation.
    fn h_min(&self) -> f64;

    /// Get the maximum element size in the mesh.
    fn h_max(&self) -> f64;

    /// Get the diameter (characteristic length) of a specific element.
    fn element_diameter(&self, element: usize) -> f64;
}

// =============================================================================
// MeshGeometryExt Trait
// =============================================================================

/// Extended geometry for 2D and 3D meshes with face normals and surface Jacobians.
///
/// This trait provides additional geometric information needed for
/// computing surface integrals in the DG formulation.
pub trait MeshGeometryExt: MeshGeometry {
    /// Normal vector type (same dimension as coordinates).
    type Normal: Point;

    /// Get the outward unit normal vector for a face.
    ///
    /// # Arguments
    /// * `element` - Element index
    /// * `local_face` - Local face index
    fn face_normal(&self, element: usize, local_face: usize) -> Self::Normal;

    /// Get the surface Jacobian for a face.
    ///
    /// This is the ratio of physical face area to reference face area,
    /// used for scaling surface integrals.
    fn surface_jacobian(&self, element: usize, local_face: usize) -> f64;
}

// =============================================================================
// MeshCFL Trait (with blanket implementation)
// =============================================================================

/// Trait for computing CFL-stable time steps.
///
/// This trait has a blanket implementation for all types implementing [`MeshGeometry`].
pub trait MeshCFL: MeshGeometry {
    /// Compute a CFL-stable time step.
    ///
    /// Uses the formula: dt = CFL * h_min / (wave_speed * (2N + 1))
    /// where N is the polynomial order.
    ///
    /// # Arguments
    /// * `wave_speed` - Maximum wave speed in the domain
    /// * `order` - Polynomial order of the DG approximation
    /// * `cfl` - CFL number (typically 0.1-0.5 for explicit DG)
    fn compute_dt(&self, wave_speed: f64, order: usize, cfl: f64) -> f64 {
        if wave_speed < 1e-14 {
            return f64::INFINITY;
        }
        let dg_factor = 2.0 * order as f64 + 1.0;
        cfl * self.h_min() / (wave_speed * dg_factor)
    }
}

// Blanket implementation: any mesh with geometry can compute CFL
impl<M: MeshGeometry> MeshCFL for M {}

// =============================================================================
// Dimension-Specific Extension Traits
// =============================================================================

/// Extension trait for 1D meshes.
///
/// Provides 1D-specific operations like element bounds and simplified normals.
pub trait Mesh1DGeometry: MeshGeometry<Coord = f64, RefCoord = f64> {
    /// Get the left and right physical coordinates of an element.
    fn element_bounds(&self, element: usize) -> (f64, f64);

    /// Get the size of an element directly.
    fn element_size(&self, element: usize) -> f64;

    /// Get the face normal for a 1D element.
    ///
    /// In 1D, normals are simply -1 (left face) or +1 (right face).
    #[inline]
    fn face_normal_1d(&self, _element: usize, local_face: usize) -> f64 {
        if local_face == 0 { -1.0 } else { 1.0 }
    }
}

/// Extension trait for 2D meshes with quadrilateral elements.
///
/// Provides 2D-specific operations for quadrilateral meshes.
pub trait Mesh2DGeometry:
    MeshGeometry<Coord = [f64; 2], RefCoord = [f64; 2]> + MeshGeometryExt<Normal = [f64; 2]>
{
    /// Get all four face normals for an element.
    ///
    /// Returns normals in order [bottom, right, top, left] following
    /// the counter-clockwise face numbering convention.
    fn face_normals_array(&self, element: usize) -> [[f64; 2]; 4];

    /// Get all four surface Jacobians for an element.
    fn surface_jacobians_array(&self, element: usize) -> [f64; 4];

    /// Get the four vertex coordinates of an element.
    ///
    /// Returns vertices in counter-clockwise order starting from bottom-left.
    fn element_vertices(&self, element: usize) -> [[f64; 2]; 4];
}

// =============================================================================
// GPU Data Access Trait
// =============================================================================

/// Trait for meshes providing contiguous data arrays for GPU kernels.
///
/// This is an optional trait for meshes optimized for GPU computation.
/// Data is pre-computed and laid out for coalesced memory access.
pub trait MeshGPUData: MeshTopology {
    /// Jacobian determinants as a contiguous slice `[n_elements]`.
    fn jacobians(&self) -> &[f64];

    /// Inverse Jacobian determinants `[n_elements]`.
    fn jacobians_inv(&self) -> &[f64];

    /// Neighbor element indices, packed `[n_elements * FACES_PER_ELEMENT]`.
    ///
    /// Uses `usize::MAX` to indicate boundary faces.
    fn neighbor_indices(&self) -> &[usize];

    /// Neighbor face indices, packed `[n_elements * FACES_PER_ELEMENT]`.
    fn neighbor_faces(&self) -> &[usize];

    /// Boundary tags as integers, packed `[n_elements * FACES_PER_ELEMENT]`.
    ///
    /// Uses 0 for interior faces, >0 for boundary tag values.
    fn boundary_tags_packed(&self) -> &[u32];
}

// =============================================================================
// Iteration Helpers
// =============================================================================

/// Iterator over all element indices.
pub struct ElementIter {
    current: usize,
    end: usize,
}

impl Iterator for ElementIter {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.end {
            let idx = self.current;
            self.current += 1;
            Some(idx)
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.end - self.current;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for ElementIter {}

/// Extension trait for iterating over mesh elements.
pub trait MeshIter: MeshTopology {
    /// Returns an iterator over all element indices.
    fn elements(&self) -> ElementIter {
        ElementIter {
            current: 0,
            end: self.n_elements(),
        }
    }
}

// Blanket implementation
impl<M: MeshTopology> MeshIter for M {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_face_connection() {
        let interior: FaceConnection<u32> = FaceConnection::Interior(Neighbor {
            element: 5,
            face: 1,
        });
        let boundary: FaceConnection<u32> = FaceConnection::Boundary(42);

        assert!(interior.is_interior());
        assert!(!interior.is_boundary());
        assert_eq!(interior.neighbor().unwrap().element, 5);
        assert!(interior.boundary_tag().is_none());

        assert!(!boundary.is_interior());
        assert!(boundary.is_boundary());
        assert!(boundary.neighbor().is_none());
        assert_eq!(*boundary.boundary_tag().unwrap(), 42);
    }

    #[test]
    fn test_neighbor() {
        let n = Neighbor {
            element: 10,
            face: 2,
        };
        assert_eq!(n.element, 10);
        assert_eq!(n.face, 2);
    }
}

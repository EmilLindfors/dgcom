//! Abstract mesh traits for dimension-independent operations.
//!
//! This module provides:
//! - [`Point`]: Coordinate type abstraction for 1D, 2D, 3D
//! - [`MeshTopology`]: Element and face connectivity
//! - [`MeshGeometry`]: Coordinate mappings and Jacobians
//! - [`MeshGeometryExt`]: Face normals and surface Jacobians (2D/3D)
//! - [`MeshCFL`]: CFL time step computation

pub mod mesh_traits;
pub mod point;

pub use mesh_traits::{
    FaceConnection, Mesh1DGeometry, Mesh2DGeometry, MeshCFL, MeshGeometry, MeshGeometryExt,
    MeshIter, MeshTopology, Neighbor,
};
pub use point::Point;

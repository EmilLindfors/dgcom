//! Mesh representation for DG discretizations.
//!
//! This module provides mesh data structures organized into submodules:
//!
//! # Submodules
//!
//! - [`core`]: Core mesh implementations (Mesh1D, Mesh2D, builders)
//! - [`traits`]: Abstract traits for dimension-independent operations
//! - [`data`]: Mesh-associated data (bathymetry, land masks, boundary tags)
//! - [`io`]: Mesh file I/O (Gmsh format)
//!
//! # Mesh Traits
//!
//! The [`traits`] module provides abstract interfaces for mesh operations:
//! - [`Point`]: Coordinate type abstraction for 1D, 2D, 3D
//! - [`MeshTopology`]: Element and face connectivity
//! - [`MeshGeometry`]: Coordinate mappings and Jacobians
//! - [`MeshGeometryExt`]: Face normals and surface Jacobians (2D/3D)
//! - [`MeshCFL`]: CFL time step computation
//!
//! These traits enable generic code that works across mesh dimensions.

pub mod core;
pub mod data;
pub mod io;
pub mod traits;

// Re-export core mesh types
pub use core::{
    BoundaryConfig, BoundaryFace, Edge, ElementFace, FaceLink, Mesh1D, Mesh2D, Mesh2DBuilder,
    MeshPoint, PointLocator2D, QuadMeshError, inverse_bilinear,
};

// Re-export data types
pub use data::{
    Bathymetry1D, Bathymetry2D, BoundaryTag, CliffShores, ElementSlopeBound, ElementSlopeSmoothing,
    LandMask2D, LandMaskStatistics, Rx0, Rx0Smoothing, TidalPonds, WallLandLowering,
    bathymetry_profiles, bathymetry_profiles_2d,
};

// Re-export I/O types
pub use io::{GmshError, parse_gmsh_mesh, read_gmsh_mesh, write_gmsh_mesh};

// Re-export trait types
pub use traits::{
    FaceConnection, Mesh1DGeometry, Mesh2DGeometry, MeshCFL, MeshGPUData, MeshGeometry,
    MeshGeometryExt, MeshIter, MeshTopology, Neighbor, Point,
};

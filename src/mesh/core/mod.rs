//! Core mesh implementations.
//!
//! This module contains the actual mesh data structures:
//! - [`Mesh1D`]: 1D mesh for interval elements
//! - [`Mesh2D`]: 2D mesh for quadrilateral elements
//! - [`Mesh2DBuilder`]: Fluent builder for constructing 2D meshes
//! - [`PointLocator2D`]: the element and reference coordinates of a point

mod mesh1d;
mod mesh2d;
mod mesh2d_builder;
mod point_locator;

pub use mesh1d::{BoundaryFace, Mesh1D};
pub use mesh2d::{Edge, ElementFace, FaceLink, Mesh2D, QuadMeshError};
pub use mesh2d_builder::{BoundaryConfig, Mesh2DBuilder};
pub use point_locator::{MeshPoint, PointLocator2D, inverse_bilinear};

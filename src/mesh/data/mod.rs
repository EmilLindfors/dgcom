//! Mesh-associated data structures.
//!
//! This module contains data that lives on meshes:
//! - [`Bathymetry1D`], [`Bathymetry2D`]: Bottom topography
//! - [`BoundaryTag`]: Boundary condition labels

mod bathymetry;
mod bathymetry_2d;
mod boundary_tags;

pub use bathymetry::{Bathymetry1D, profiles as bathymetry_profiles};
pub use bathymetry_2d::{
    Bathymetry2D, CliffShores, ElementSlopeBound, ElementSlopeSmoothing, Rx0, Rx0Smoothing,
    TidalPonds, WallLandLowering, profiles as bathymetry_profiles_2d,
};
pub use boundary_tags::BoundaryTag;

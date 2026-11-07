//! Strongly-typed domain types for safer APIs.
//!
//! This module provides newtypes and structured types to make APIs
//! self-documenting and prevent parameter mix-ups.
//!
//! # Design Philosophy
//!
//! - **Newtypes prevent mix-ups**: `Depth(100.0)` vs `Elevation(0.5)` are distinct types
//! - **Named fields over positional**: `SideBoundaries { south, east, north, west }`
//! - **Compile-time guarantees**: Typestate patterns ensure valid construction
//! - **Zero-cost abstractions**: All newtypes are `#[repr(transparent)]`
//!
//! # Example
//!
//! ```
//! use dgcom::types::{Bounds2D, Resolution2D, Depth, Elevation, Sigma};
//!
//! // Domain bounds with clear semantics
//! let bounds = Bounds2D::new(0.0, 100e3, 0.0, 50e3);
//! assert_eq!(bounds.width(), 100e3);
//! assert_eq!(bounds.height(), 50e3);
//!
//! // Physical quantities that can't be mixed up
//! let h = Depth::new(200.0);      // Water depth (positive)
//! let eta = Elevation::new(0.5);   // Surface elevation
//! let sigma = Sigma::new(-0.5);    // Sigma coordinate
//!
//! // Convert sigma to physical z
//! let z = sigma.to_z(eta, h);
//! ```

mod bounds;
mod indices;
mod physical;
mod resolution;
mod sides;

pub use bounds::Bounds2D;
pub use indices::{ElementIndex, FaceIndex, LevelIndex, NodeIndex};
pub use physical::{Depth, Elevation, PhysicalZ, Sigma};
pub use resolution::Resolution2D;
pub use sides::SideBoundaries;

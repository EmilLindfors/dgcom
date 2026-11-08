//! Vertical coordinate systems for 3D ocean modeling.
//!
//! This module provides terrain-following (sigma) coordinates for ROMS-style
//! hybrid discretization: DG horizontal + finite-difference vertical.
//!
//! # Performance
//!
//! This module is designed for high performance:
//! - Uses `faer::Col` for SIMD-friendly contiguous memory layout
//! - Provides batch operations to avoid per-element allocations
//! - All coordinate transforms are `#[inline]` for zero-cost abstraction
//!
//! # Sigma Coordinates
//!
//! Sigma coordinates follow the terrain, mapping the water column to a
//! fixed computational domain [-1, 0] regardless of local depth:
//!
//! - σ = -1 at bottom (z = -H)
//! - σ = 0 at surface (z = η)
//!
//! The physical depth is: `z = η + (η + H) × σ`
//!
//! # Stretching Functions
//!
//! Different stretching functions control vertical resolution distribution:
//!
//! - [`UniformStretching`]: Equal spacing (for testing)
//! - [`SongHaidvogelStretching`]: ROMS default with surface/bottom clustering
//! - [`DoubleTanhStretching`]: Independent control of boundary layers
//!
//! # Example
//!
//! ```
//! use dgcom::vertical::{SigmaGrid, UniformStretching, SongHaidvogelStretching};
//!
//! // Create a 35-level grid with ROMS-style stretching
//! let stretching = SongHaidvogelStretching {
//!     theta_s: 7.0,   // Strong surface refinement
//!     theta_b: 0.1,   // Weak bottom refinement
//!     hc: 250.0,      // Critical depth
//! };
//! let grid = SigmaGrid::new(35, stretching);
//!
//! // Convert sigma to physical depth
//! let eta = 0.5;    // Surface elevation (m)
//! let h = 200.0;    // Water depth (m)
//!
//! // Get depth at first (bottom) level center
//! let z_bottom = grid.sigma_to_z(grid.sigma_at_level(0), eta, h);
//!
//! // Get all level depths (returns faer::Col<f64>)
//! let z_levels = grid.z_at_levels(eta, h);
//!
//! // Get layer thicknesses
//! let dz = grid.layer_thicknesses(eta, h);
//! ```
//!
//! # Batch Operations for 3D Simulations
//!
//! For 3D simulations, use batch operations to avoid allocations in hot paths:
//!
//! ```
//! use dgcom::vertical::{SigmaGrid, UniformStretching};
//!
//! let grid = SigmaGrid::new(35, UniformStretching);
//!
//! // Pre-allocate buffers (do this once, outside the hot loop)
//! let mut z_buffer = vec![0.0; grid.n_levels()];
//!
//! // In the hot loop, reuse buffers - no allocation!
//! let eta = 0.5;
//! let h = 200.0;
//! grid.z_at_levels_into(eta, h, &mut z_buffer);
//!
//! // For multiple water columns at once (SIMD-friendly via LLVM auto-vectorization)
//! let n_columns = 1000;
//! let eta_all: Vec<f64> = vec![0.5; n_columns];
//! let h_all: Vec<f64> = vec![200.0; n_columns];
//! let mut z_out = vec![0.0; n_columns];
//!
//! grid.z_at_level_batch(10, &eta_all, &h_all, &mut z_out);
//! ```
//!
//! # Norwegian Coastal Applications
//!
//! For fjord modeling, typical parameters are:
//!
//! - 35-40 vertical levels
//! - Strong surface stretching (theta_s = 5-7) for mixed layer
//! - Weak bottom stretching (theta_b = 0-1) unless BBL is important
//! - Critical depth hc = 200-300m for deep fjords
//!
//! ```
//! use dgcom::vertical::{SigmaGrid, SongHaidvogelStretching};
//!
//! // Norwegian fjord configuration
//! let fjord_grid = SigmaGrid::new(
//!     40,
//!     SongHaidvogelStretching::new(7.0, 0.5, 250.0),
//! );
//!
//! // For 500m deep fjord with 0.3m tidal elevation
//! let z = fjord_grid.z_at_levels(0.3, 500.0);
//! assert!(z[0] < -450.0);   // Bottom level near seafloor
//! assert!(z[39] > -5.0);    // Top level near surface
//! ```

mod mean_profile;
mod sigma;
mod stretching;

pub use mean_profile::MeanProfile;
pub use sigma::SigmaGrid;
pub use stretching::{
    DoubleTanhStretching, ROMSVstretching4, SongHaidvogelStretching, Stretching, UniformStretching,
};

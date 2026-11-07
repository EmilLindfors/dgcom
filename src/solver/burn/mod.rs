//! GPU-accelerated DG solver using Burn tensor framework.
//!
//! This module provides GPU-accelerated implementations of the DG SWE solver
//! using the [Burn](https://github.com/tracel-ai/burn) deep learning framework.
//!
//! # Backend Selection
//!
//! Burn supports multiple backends:
//! - **CUDA**: NVIDIA GPUs (highest performance, requires `burn-cuda` feature)
//! - **WGPU**: Cross-platform GPU (Vulkan/DX12/Metal, requires `burn-wgpu` feature)
//! - **NdArray**: CPU reference implementation (requires `burn-ndarray` feature)
//!
//! # Usage
//!
//! ```ignore
//! use dgcom::solver::burn::{BurnOperators2D, BurnSWESolution2D, compute_rhs_swe_2d_burn};
//! use burn::backend::Wgpu;
//!
//! // Initialize operators on GPU
//! let device = Default::default();
//! let burn_ops = BurnOperators2D::<Wgpu>::from_cpu(&ops, &device);
//!
//! // Upload solution to GPU
//! let mut burn_sol = BurnSWESolution2D::<Wgpu>::from_cpu(&solution, &device);
//!
//! // Compute RHS on GPU
//! let burn_rhs = compute_rhs_swe_2d_burn(&burn_sol, &burn_ops, &burn_geom, ...);
//!
//! // Download result to CPU
//! let rhs = burn_rhs.to_cpu();
//! ```
//!
//! # Performance Considerations
//!
//! GPU acceleration is most beneficial for:
//! - Large meshes (>1000 elements)
//! - High polynomial orders (P≥3)
//! - Long simulations with many timesteps
//!
//! For small problems, CPU implementations may be faster due to data transfer overhead.

mod backend;
mod connectivity;
mod error;
mod flux;
mod kernels;
mod operators;
pub mod rhs;
mod solution;
mod surface;

pub use backend::DGBackend;
pub use connectivity::BurnConnectivity;
pub use error::BurnError;
pub use flux::{hll_flux_batched, roe_flux_batched};
pub use kernels::{
    apply_diff_matrix_batched, combine_derivatives_batched, compute_swe_fluxes,
    coriolis_source_batched, friction_source_batched,
};
pub use operators::BurnOperators2D;
pub use rhs::{BurnGeometricFactors2D, BurnRhsConfig, compute_rhs_swe_2d_burn};
pub use solution::BurnSWESolution2D;
pub use surface::{BurnFaceStates, apply_lift_all_faces, gather_face_states};

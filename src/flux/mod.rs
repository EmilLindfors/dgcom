//! Numerical flux functions.
//!
//! Provides numerical fluxes for DG discretizations:
//! - Scalar fluxes: upwind, Lax-Friedrichs
//! - Shallow water fluxes: Roe, HLL/HLLC
//! - 2D shallow water fluxes with rotation to face-aligned coordinates
//! - Entropy-conservative/-stable two-point fluxes for the split-form 2D SWE
//!   DGSEM with bathymetry (Wintermeyer et al. 2017)
//! - Tracer fluxes: upwind-based for temperature and salinity
//!
//! # Flux Trait
//!
//! The [`NumericalFlux2D`] trait provides a generic interface for numerical fluxes.
//! This enables extending the solver with custom fluxes while maintaining performance
//! through both compile-time and runtime dispatch options.
//!
//! ## Built-in Flux Types
//! - [`RoeFlux2D`]: Roe approximate Riemann solver
//! - [`HLLFlux2D`]: HLL solver (robust for strong shocks)
//! - [`RusanovFlux2D`]: Local Lax-Friedrichs (simple and robust)
//! - [`StandardFlux2D`]: Enum for zero-cost dispatch when type is known at compile time

mod hll;
mod roe;
mod swe_2d;
mod swe_2d_entropy;
pub mod traits;
mod upwind;

pub use hll::{hll_flux_jump_swe, hll_flux_swe, hll_flux_swe_normal, hllc_flux_swe};
pub use roe::{roe_flux_jump_swe, roe_flux_swe, roe_flux_swe_normal};
pub(crate) use swe_2d::{HllSide, hll_flux_face_aligned, rotate_from_normal};
pub use swe_2d::{
    SWEFluxType2D, compute_flux_swe_2d, hll_flux_swe_2d, roe_flux_swe_2d, rusanov_flux_swe_2d,
};
pub use swe_2d_entropy::{
    SWENodeState2D, entropy_stable_dissipation_2d, node_velocity, swe_entropy_2d,
    swe_entropy_variables_2d, wintermeyer_bed_interface_term_2d, wintermeyer_flux_2d,
};
pub use upwind::{lax_friedrichs_flux, upwind_flux};

// Re-export trait-based flux types
pub use traits::{
    BoxedFlux2D, FluxContext2D, HLLFlux2D, NumericalFlux2D, RoeFlux2D, RusanovFlux2D,
    StandardFlux2D, create_flux,
};

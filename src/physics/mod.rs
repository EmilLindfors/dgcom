//! Physics module abstraction for DG simulations.
//!
//! This module provides a high-level interface for configuring and running
//! physics simulations. It abstracts over the details of:
//! - Numerical flux selection
//! - Boundary condition handling
//! - Source term composition
//! - Limiter application
//! - Time stepping
//!
//! # Key Traits
//!
//! - [`PhysicsModule`]: Core trait for physics computations (RHS, dt, post-processing)
//! - [`PhysicsConfig`]: Configuration for building physics modules
//!
//! # Example
//! ```ignore
//! use dg_rs::physics::{SWEPhysics2D, PhysicsBuilder};
//!
//! let physics = PhysicsBuilder::swe_2d()
//!     .with_flux(StandardFlux2D::Roe)
//!     .with_limiter(StandardLimiter2D::TvbWithPositivity { ... })
//!     .with_bathymetry(&bathymetry)
//!     .with_source(&combined_sources)
//!     .build();
//! ```

pub mod air_sea;
pub mod bottom_drag;
pub mod builder;
pub mod cage_drag;
pub mod eos;
pub mod gls;
pub mod hydrostatic_3d;
pub mod implicit_advection;
pub mod surface_fluxes;
pub mod surface_stress;
pub mod traits;
pub mod vertical_diffusion;
pub mod vertical_mixing;

pub use air_sea::{AirState, Coare35, JerlovWater, SurfaceBudget};
pub use bottom_drag::BottomDrag3D;
pub use builder::{PhysicsBuilder, SWEPhysics2D, SWEPhysics2DBuilder};
pub use eos::{EquationOfState, LinearEOS, UnescoEOS};
pub use gls::{GlsMixing, GlsParameters, StabilityFunctions, WaveBreaking};
pub use hydrostatic_3d::{
    ElementStep, Hydrostatic3D, ImplicitAdvectionStats, TimeStepLimit, TimeStepLimits,
};
pub use implicit_advection::ImplicitVerticalAdvection;
pub use surface_fluxes::{SeaSurface, SurfaceFluxFields, SurfaceFluxes3D};
pub use surface_stress::{AnalyticSurfaceStress, SurfaceStress3D};
pub use traits::{PhysicsConfig, PhysicsModule, PhysicsModuleInfo};
pub use vertical_diffusion::{SurfaceFields, SurfaceTracerFluxes, apply_vertical_diffusion};
pub use vertical_mixing::{
    Column, ConstantMixing, Forcing, PacanowskiPhilanderMixing, Turbulence, VerticalMixing,
};

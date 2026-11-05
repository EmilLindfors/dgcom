//! DG solver components.
//!
//! # Submodules
//!
//! - [`core`]: Core solution containers (DGSolution1D, DGSolution2D, SystemSolution)
//! - [`state`]: Physical state types (SWEState, etc.)
//! - [`rhs`]: Right-hand side computation for various equations
//! - [`limiters`]: Slope limiters for oscillation control
//! - [`algorithms`]: Specialized algorithms (wetting/drying)
//! - [`diagnostics`]: Runtime diagnostics and progress tracking
//! - [`probe`]: Sampling the solution at arbitrary points (stations)
//! - [`simd`]: SIMD-optimized data structures and kernels
//!
//! # Limiter Traits
//!
//! The [`Limiter2D`] trait provides a generic interface for slope limiters.
//! This enables composable limiters and runtime selection:
//!
//! - [`TVBLimiter2D`]: Total Variation Bounded slope limiter
//! - [`KuzminLimiter2D`]: Vertex-based Kuzmin limiter
//! - [`PositivityLimiter2D`]: Positivity-preserving limiter
//! - [`LimiterChain2D`]: Chain multiple limiters together
//! - [`StandardLimiter2D`]: Enum for zero-cost dispatch

pub mod algorithms;
#[cfg(feature = "burn")]
pub mod burn;
pub mod core;
pub mod diagnostics;
pub mod limiters;
pub mod probe;
pub mod rhs;
pub mod simd;
pub mod state;

// Re-export core solution types
pub use core::{Blocks, DGSolution1D, DGSolution2D, SystemSolution, SystemSolution2D};

pub use probe::{Probe2D, SWEPointSample};

// Re-export state types
pub use state::{SWESolution, SWESolution2D, SWEState, SWEState2D};

// Re-export RHS types and functions
pub use rhs::{
    // 2D Advection
    AdvectionBoundaryCondition2D,
    AdvectionFluxType,
    // 1D
    BoundaryCondition,
    ConstantBC2D,
    DirichletBC2D,
    // 2D Tracer
    // 3D RHS
    ExtrapolationTracerBC3D,
    FixedTracerBC3D,
    LINEAR_CFL_SAFETY,
    POSITIVITY_RELAXATION_SAFETY,
    PeriodicBC2D,
    PositivityBound,
    Rhs3DConfig,
    SUBCELL_CAP_DEPTH_FACTOR,
    // 2D SWE
    SWE2DRhsConfig,
    SWEFluxType,
    SWEFormulation2D,
    SWERhsConfig,
    TracerBCContext3D,
    TracerBoundaryCondition3D,
    UpwindTracerBC3D,
    compute_dt_advection_2d,
    compute_dt_swe,
    compute_dt_swe_2d,
    compute_dt_viscosity,
    compute_momentum_rhs_3d,
    compute_rhs,
    compute_rhs_advection_2d,
    compute_rhs_swe,
    compute_rhs_swe_2d,
    compute_rhs_swe_2d_face_mass_into,
    compute_rhs_swe_2d_into,
    compute_rhs_swe_2d_mass_fluxes_into,
    compute_rhs_swe_2d_subset_then,
    compute_rhs_swe_2d_where_into,
    compute_transport_rhs_3d,
    element_dt_swe_2d,
    element_dt_viscous_swe_2d,
    face_mass_len,
    largest_viscosity_swe_2d,
    linear_cfl_subcells_swe_2d,
    linear_cfl_swe_2d,
    min_element_dt_swe_2d,
    positivity_cfl_swe_2d,
    subcell_mass_len,
};

// Re-export limiter types
pub use limiters::{
    // Traits and context
    BoxedLimiter2D,
    // 2D SWE limiters
    KuzminLimiter2D,
    KuzminParameter2D,
    Limiter2D,
    LimiterChain2D,
    LimiterContext2D,
    NoLimiter2D,
    PositivityLimiter2D,
    StandardLimiter2D,
    // 1D limiters
    TVBParameter,
    TracerAveragePolicy3D,
    TracerBounds,
    TracerLimiter3DConfig,
    TracerLimiter3DStats,
    TracerLimiterType3D,
    TracerReferenceProfile,
    apply_swe_limiters,
    apply_swe_limiters_kuzmin_2d,
    apply_tracer_limiters_3d,
    create_limiter,
    positivity_limiter,
    swe_cell_averages_2d,
    swe_kuzmin_limiter_2d,
    swe_positivity_limiter_2d,
    tvb_limiter,
};

// Re-export algorithms
pub use algorithms::{
    ImplicitDamping2D, WetDryConfig, apply_implicit_damping_2d, apply_wet_dry_correction,
    apply_wet_dry_correction_all, apply_wet_dry_correction_element,
};

#[cfg(feature = "parallel")]
pub use algorithms::{apply_implicit_damping_2d_parallel, apply_wet_dry_correction_all_parallel};

// Re-export diagnostics
pub use diagnostics::{
    DiagnosticsTracker, PotentialEnergy, PotentialEnergy3D, ProgressReporter, SWEDiagnostics2D,
    current_cfl_2d, total_energy_2d, total_mass_2d, total_momentum_2d,
};

// Re-export SIMD types
pub use simd::{FaceWorkspace, SIMDWorkspace, SWESoABuffer};
pub use simd::{
    apply_diff_matrix_scalar, apply_lift_scalar, combine_derivatives_scalar,
    coriolis_source_scalar, manning_friction_scalar,
};

#[cfg(feature = "simd")]
pub use simd::{apply_diff_matrix, apply_lift, combine_derivatives, coriolis_source};

#[cfg(feature = "parallel")]
pub use rhs::compute_rhs_parallel;

#[cfg(feature = "parallel")]
pub use limiters::{
    apply_swe_limiters_kuzmin_2d_parallel, swe_cell_averages_2d_parallel,
    swe_kuzmin_limiter_2d_parallel, swe_positivity_limiter_2d_parallel,
};

#[cfg(feature = "parallel")]
pub use rhs::{
    compute_dt_swe_2d_parallel, compute_rhs_swe_2d_parallel,
    compute_rhs_swe_2d_parallel_face_mass_into, compute_rhs_swe_2d_parallel_into,
    compute_rhs_swe_2d_parallel_mass_fluxes_into,
};

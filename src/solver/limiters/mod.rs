//! Slope limiters for DG methods.
//!
//! Limiters control oscillations near discontinuities while preserving
//! high-order accuracy in smooth regions.
//!
//! - [`Limiter2D`]: Trait for 2D slope limiters
//! - [`KuzminLimiter2D`], [`PositivityLimiter2D`]: Concrete implementations
//! - [`LimiterChain2D`]: Compose multiple limiters
//! - [`StandardLimiter2D`]: Zero-cost dispatch enum

mod limiters_1d;
mod standard;
mod swe_2d;
pub mod tracer_3d;
mod traits;

// Traits
pub use traits::{BoxedLimiter2D, Limiter2D, LimiterContext2D};

// Standard limiter types (chain, enum, factory)
pub use standard::{
    KuzminLimiter2D, LimiterChain2D, NoLimiter2D, PositivityLimiter2D, StandardLimiter2D,
    create_limiter,
};

// 1D limiters
pub use limiters_1d::{TVBParameter, apply_swe_limiters, positivity_limiter, tvb_limiter};

// 2D SWE limiters
pub use swe_2d::{
    KuzminParameter2D, apply_swe_limiters_kuzmin_2d, swe_cell_averages_2d, swe_kuzmin_limiter_2d,
    swe_positivity_limiter_2d,
};
// Element kernels shared with the wet/dry correction
pub(crate) use swe_2d::{element_mass, element_mean, positivity_limit_element};

// Parallel 2D SWE limiters
#[cfg(feature = "parallel")]
pub use swe_2d::{
    apply_swe_limiters_kuzmin_2d_parallel, swe_cell_averages_2d_parallel,
    swe_kuzmin_limiter_2d_parallel, swe_positivity_limiter_2d_parallel,
};

// 3D Tracer limiters
pub use tracer_3d::{
    ReferenceProfileView, TracerAveragePolicy3D, TracerBounds, TracerLimiter3DConfig,
    TracerLimiter3DStats, TracerLimiterType3D, TracerReferenceProfile, apply_tracer_limiters_3d,
};

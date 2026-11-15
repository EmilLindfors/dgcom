//! Specialized algorithms for DG methods.
//!
//! - [`WetDryConfig`]: Configuration for wetting/drying treatment
//! - [`apply_wet_dry_correction_all`]: positivity and velocity desingularization
//!   for shallow water, after every RK stage
//! - [`apply_implicit_damping_2d`]: point-implicit bottom friction and thin-layer
//!   relaxation, in every RK stage

pub mod tridiagonal;
mod wetting_drying;

pub use wetting_drying::{
    ImplicitDamping2D, WetDryConfig, apply_implicit_damping_2d, apply_wet_dry_correction,
    apply_wet_dry_correction_all, apply_wet_dry_correction_element, damp_node, wet_dry_element,
};

#[cfg(feature = "parallel")]
pub use wetting_drying::{
    apply_implicit_damping_2d_parallel, apply_wet_dry_correction_all_parallel,
};

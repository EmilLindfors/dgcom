//! Right-hand side computation for DG discretizations.
//!
//! Computes the semi-discrete RHS for various equation systems:
//! - Scalar advection (1D and 2D)
//! - Shallow water equations (1D and 2D)
//! - Tracer transport (2D)

mod advection_2d;
pub mod baroclinic;
mod diffusion_2d;
mod scalar_1d;
pub mod subcells;
mod swe_1d;
mod swe_2d;
mod swe_2d_split_form;
mod swe_2d_viscosity;

// 1D scalar
#[cfg(feature = "parallel")]
pub use scalar_1d::compute_rhs_parallel;
pub use scalar_1d::{BoundaryCondition, compute_rhs};

// 1D SWE
pub use swe_1d::{SWEFluxType, SWERhsConfig, compute_dt_swe, compute_rhs_swe};

// 2D advection
pub use advection_2d::{
    AdvectionBoundaryCondition2D, AdvectionFluxType, ConstantBC2D, DirichletBC2D, PeriodicBC2D,
    compute_dt_advection_2d, compute_rhs_advection_2d,
};

// 2D SWE
pub use swe_2d::{
    ElementMassFluxes, ElementRhsMassThen, ElementRhsThen, LINEAR_CFL_SAFETY,
    POSITIVITY_RELAXATION_SAFETY, PositivityBound, SUBCELL_CAP_DEPTH_FACTOR, SWE2DRhsConfig,
    SWEFormulation2D, compute_dt_swe_2d, compute_dt_viscosity, compute_rhs_swe_2d,
    compute_rhs_swe_2d_face_mass_into, compute_rhs_swe_2d_into,
    compute_rhs_swe_2d_mass_fluxes_into, compute_rhs_swe_2d_subset_mass_fluxes_then,
    compute_rhs_swe_2d_subset_then, compute_rhs_swe_2d_where_into, element_dt_swe_2d,
    element_dt_viscous_swe_2d, face_mass_len, linear_cfl_subcells_swe_2d, linear_cfl_swe_2d,
    min_element_dt_swe_2d, positivity_cfl_swe_2d, subcell_mass_len,
};
#[cfg(feature = "parallel")]
pub use swe_2d::{
    compute_dt_swe_2d_parallel, compute_rhs_swe_2d_parallel,
    compute_rhs_swe_2d_parallel_face_mass_into, compute_rhs_swe_2d_parallel_into,
    compute_rhs_swe_2d_parallel_mass_fluxes_into,
};
pub use swe_2d_viscosity::largest_viscosity_swe_2d;

// 3D Baroclinic
pub use baroclinic::{BalancedReference, PressureGradientForm, compute_pressure_gradient};

pub mod coriolis_3d;
pub use coriolis_3d::apply_coriolis_3d;

pub mod boundary_3d;
pub use boundary_3d::{Boundaries3D, Exterior3D, ExteriorField, FaceExterior};

pub mod advection_3d;
pub use advection_3d::{
    ExtrapolationTracerBC3D, FixedTracerBC3D, TracerBCContext3D, TracerBoundaryCondition3D,
    UpwindTracerBC3D,
};

pub mod transport_3d;
pub use transport_3d::{
    AdvectedTracer, BarotropicFlux, LayerTransport, MetricForm, MomentumAdvectionForm, UNBOUNDED,
    VerticalAdvection, advective_divergence_element, apply_momentum_transport_3d,
    apply_tracer_transport_3d, from_inventory, subcell_divergence_element, to_inventory,
    transport_divergence_element, w_cell_thicknesses,
};

pub mod viscosity_3d;
pub use viscosity_3d::{
    HorizontalViscosity3D, ViscosityScratch3D, apply_horizontal_viscosity_3d,
    largest_horizontal_viscosity_3d,
};

pub mod rhs_3d;
pub use rhs_3d::{Rhs3DConfig, compute_momentum_rhs_3d, compute_transport_rhs_3d};

//! Time integration methods.
//!
//! # Integrator Traits
//!
//! The [`TimeIntegrator`] trait provides a generic interface for time integrators.
//! This enables composable integrators and runtime selection:
//!
//! - [`SSPRK3`]: Strong Stability Preserving RK3 (optimal for hyperbolic systems)
//! - [`SSPRK43`]: four-stage SSP-RK3 with SSP coefficient 2: 1.5× less work
//!   where a positivity bound sets the step (wet/dry runs)
//! - [`ForwardEuler`]: Simple 1st order (for testing)
//! - [`StandardIntegrator`]: Enum for zero-cost dispatch
//! - [`MultirateSSPRK3`], [`MultirateSSPRK43`]: local time stepping, every
//!   element at its own power-of-two fraction of the time step (conservative,
//!   SSP)
//!
//! The [`Integrable`] trait marks solution types that can be time-stepped.

mod clock;
pub mod integrator;
pub mod mode_split;
pub mod multirate;
mod ssp_rk3;
mod ssp_rk3_2d;
mod ssp_rk3_swe;
mod ssp_rk3_swe_2d;

pub use clock::ModelClock;
pub use mode_split::{
    BarotropicFilter, BarotropicPhysics, BarotropicTransport, Combination, GlobalPass,
    MIN_BAROTROPIC_SUBSTEPS, ModeSplitIntegrator, ModeSplitPhysics, PassDrag, PassWork, RkStage,
    Slot, SlowForcingRecord, SplitMethod, StageElements, StageStash, StageTime, StepDrag,
    barotropic_node_rates, barotropic_pass, barotropic_rates, barotropic_stages, column_drag_rates,
    filtered_state, pooled_element, pooled_elements, step_average_weights, to_transport,
};
pub use multirate::{
    ElementStage, LocalTimeStepping, Multirate, MultiratePass, MultirateSSPRK3, MultirateSSPRK43,
    MultirateStats, MultirateStepper, NodeLines, RhsStencil, StagePlan, assign_levels,
    element_stage,
};
pub use ssp_rk3::{compute_dt, ssp_rk3_step, ssp_rk3_step_timed};
pub use ssp_rk3_2d::{run_advection_2d, ssp_rk3_step_2d, ssp_rk3_step_2d_timed};
pub use ssp_rk3_swe::{
    SWETimeConfig, run_swe_simulation, ssp_rk3_swe_step, ssp_rk3_swe_step_timed, total_energy,
    total_mass, total_momentum,
};
pub use ssp_rk3_swe_2d::{
    SWE2DTimeConfig, SWELimiterType, run_swe_2d_simulation, ssp_rk3_swe_2d_step_limited,
};

// Integrator trait exports
pub use integrator::{
    BoxedIntegratorInfo, ForwardEuler, Integrable, IntegratorInfo, SSPRK3, SSPRK43, ShuOsherStage,
    SspScheme, StageWorkspace, StandardIntegrator, TimeIntegrator, create_integrator_info,
};

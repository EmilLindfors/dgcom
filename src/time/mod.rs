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

#[cfg(feature = "burn")]
mod burn_ssp_rk3;

pub use clock::ModelClock;
pub use mode_split::{
    BarotropicFilter, BarotropicPhysics, BarotropicTransport, MIN_BAROTROPIC_SUBSTEPS,
    ModeSplitIntegrator, ModeSplitPhysics, SlowForcingRecord, SplitMethod, StageElements,
    StageStash, StageTime, StepDrag, step_average_weights,
};
pub use multirate::{
    ElementStage, LocalTimeStepping, Multirate, MultiratePass, MultirateSSPRK3, MultirateSSPRK43,
    MultirateStats, MultirateStepper, NodeLines, RhsStencil, assign_levels,
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

// Burn GPU time integration exports
#[cfg(feature = "burn")]
pub use burn_ssp_rk3::{BurnTimeConfig, compute_dt_burn, run_swe_2d_burn, ssp_rk3_step_burn};

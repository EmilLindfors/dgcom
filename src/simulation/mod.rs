//! Simulation runner abstraction.
//!
//! This module provides a high-level interface for running DG simulations
//! that ties together:
//! - Physics modules (RHS computation, dt calculation)
//! - Time integrators (SSP-RK3, etc.)
//! - Diagnostics and callbacks
//!
//! # Example
//! ```ignore
//! use dgcom::simulation::Simulation;
//! use dgcom::physics::PhysicsBuilder;
//! use dgcom::time::SSPRK3;
//!
//! let physics = PhysicsBuilder::swe_2d(mesh, ops, geom, equation, bc).build();
//! let integrator = SSPRK3;
//!
//! let result = Simulation::new(physics, integrator)
//!     .with_cfl(0.5)
//!     .with_callback(|state, time| println!("t = {:.2}", time))
//!     .run(&mut state, 0.0, 100.0);
//! ```

pub(crate) mod runner;
mod simulation_3d;

pub use runner::{Simulation, SimulationConfig, SimulationResult};
pub use simulation_3d::{
    NON_FINITE_FIELDS, NonFinite, RunContext3D, RunModel3D, Simulation3D, non_finite_field,
    non_finite_of_masks, restart_of, resume_into,
};

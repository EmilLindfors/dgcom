//! Core solution containers for DG discretizations.
//!
//! This module contains the fundamental solution storage types:
//! - [`DGSolution1D`]: Scalar 1D nodal solution storage
//! - [`DGSolution2D`]: Scalar 2D nodal solution storage
//! - [`SystemSolution`]: Generic N-variable 1D solution storage
//! - [`SystemSolution2D`]: Generic N-variable 2D solution storage

pub(crate) mod blocks;
pub(crate) mod disjoint;
mod solution_1d;
mod solution_2d;
mod system_solution;

pub use blocks::Blocks;
pub use solution_1d::DGSolution1D;
pub use solution_2d::DGSolution2D;
pub use system_solution::{SystemSolution, SystemSolution2D};

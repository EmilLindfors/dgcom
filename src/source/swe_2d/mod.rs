//! 2D shallow water source terms.
//!
//! Physical forces acting on 2D shallow water flow:
//! - Coriolis effect from Earth's rotation
//! - Bottom friction (Manning, Chezy)
//! - Drag of fish-farm net cages
//! - Wind stress at the surface
//! - Atmospheric pressure gradients
//! - Wind and pressure from a weather model's grid
//! - Tidal potential forcing
//! - The waves' radiation stress
//! - Bathymetry gradients

mod atmospheric;
mod bathymetry;
mod cage;
mod coriolis;
mod friction;
mod gridded_atmosphere;
mod tidal;
pub mod viscosity;
mod waves;
mod wind;

pub use atmospheric::{AtmosphericPressure2D, P_STANDARD, RHO_WATER_PRESSURE};
pub use bathymetry::BathymetrySource2D;
pub use cage::{CageDrag2D, CageFootprint, CageNode, NetCage, loland_drag_coefficient};
pub use coriolis::{CoriolisSource2D, coriolis_beta, coriolis_parameter};
pub use friction::{
    BottomFriction2D, ChezyFriction2D, ManningFriction2D, SpatiallyVaryingManning2D,
};
pub use gridded_atmosphere::{GriddedAtmosphere2D, GriddedSurfaceFluxes, GriddedWindStress};
pub use tidal::{TidalPotential, TidalPotentialConstituent};
pub use viscosity::HorizontalViscosity2D;
pub use waves::{WaveCurrentFriction2D, WaveForce2D};
pub use wind::{DragCoefficient, RHO_AIR, RHO_WATER, WindStress2D};

//! Source terms for shallow water equations.
//!
//! Source terms represent forces that are not part of the hyperbolic flux:
//! - Bathymetry (bottom topography)
//! - Bottom friction (Manning/Chezy)
//! - Coriolis effect
//! - Wind stress
//! - Atmospheric pressure
//! - Sponge layers (boundary damping)
//! - Rivers (volume sources)
//!
//! # Submodules
//!
//! - [`traits`]: Abstract source term traits (SourceTerm, SourceTerm2D)
//! - [`swe_1d`]: 1D shallow water source terms
//! - [`swe_2d`]: 2D shallow water source terms
//! - [`boundary`]: Boundary treatment sources (sponge layers)
//! - [`river`]: Rivers as volume sources (2D and 3D)
//! - [`wellbalanced`]: Well-balanced scheme components

pub mod boundary;
pub mod river;
pub mod swe_1d;
pub mod swe_2d;
pub mod traits;
pub mod wellbalanced;

// Re-export traits
pub use traits::{
    CombinedSource, CombinedSource2D, DeviceSource2D, ElementSources, SourceContext2D, SourceTerm,
    SourceTerm2D, SourceTerms2D,
};

// Re-export 1D source terms
pub use swe_1d::{BathymetrySource, ChezyFriction, HydrostaticReconstruction, ManningFriction};

// Re-export 2D SWE source terms
pub use swe_2d::{
    AtmosphericPressure2D, BathymetrySource2D, BottomFriction2D, CageDrag2D, CageFootprint,
    CageNode, ChezyFriction2D, CoriolisSource2D, DeviceFriction2D, DragCoefficient,
    GriddedAtmosphere2D, GriddedSurfaceFluxes, GriddedWindStress, HorizontalViscosity2D,
    ManningFriction2D, NetCage, P_STANDARD, RHO_AIR, RHO_WATER, RHO_WATER_PRESSURE,
    SpatiallyVaryingManning2D, WaveCurrentFriction2D, WaveForce2D, WindStress2D, coriolis_beta,
    coriolis_parameter, loland_drag_coefficient,
};

// Re-export boundary sources
pub use boundary::{
    RectangularBoundary, SpongeDistanceFn, SpongeLayer2D, SpongeProfile, rectangular_sponge_fn,
};

// Rivers as volume sources
pub use river::{River, RiverError, RiverInflow, RiverProfile, RiverSeries, RiverSources};

// Re-export well-balanced components
pub use wellbalanced::HydrostaticReconstruction2D;

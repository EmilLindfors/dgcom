//! Spectral wind waves: a phase-averaged wave model on the DG mesh.
//!
//! The sea state is the wave action density `N(x, y, σ, θ, t)`, one nodal DG field
//! per spectral component (frequency × direction), evolved by the action balance
//! (Komen et al. 1994, "Dynamics and Modelling of Ocean Waves"; Holthuijsen 2007,
//! "Waves in Oceanic and Coastal Waters"; the SWAN Scientific Documentation):
//!
//! - [`dispersion`]: the linear dispersion relation, group velocity, `∂σ/∂d`;
//! - [`spectrum`]: the spectral grid, JONSWAP spectra and the integrated parameters
//!   (H_s, periods, direction, spread), Stokes drift, radiation stress and the bed
//!   orbital velocity that the circulation, mixing and particles need;
//! - [`sources`]: wind input, whitecapping, bottom friction, depth-induced breaking,
//!   and the diagnostic high-frequency tail;
//! - [`nonlinear`]: four-wave (quadruplet) interactions by the DIA;
//! - [`stokes`]: [`StokesDriftField`], the Stokes drift of a wave state at any
//!   point and depth, for the particle trackers;
//! - [`model`]: [`WaveModel2D`], the geographic DG propagation with refraction and
//!   current-induced frequency shifting, and the step that combines them;
//! - [`coupling`]: [`WaveCoupling2D`], the exchanges with a circulation on a mesh
//!   and order of its own (level and currents to the waves; radiation stress,
//!   bed stress, surface roughness and Stokes drift back), and
//!   [`CoupledWaves2D`], the waves run alongside a circulation, exchanging
//!   every coupling interval.
//! - [`boundary`]: [`BoundarySpectra`], a parent wave model's point spectra
//!   (e.g. MET Norway's WAM 800 m, read by `io::WaveSpectraFile`) regridded
//!   onto the model's grid and interpolated onto its open boundary in space
//!   and time, and [`WindSeries`], the parent's wind.

pub mod boundary;
pub mod coupling;
pub mod dispersion;
pub mod model;
pub mod nonlinear;
pub mod sources;
pub mod spectrum;
pub mod state;
pub mod stokes;

pub use boundary::{BoundarySpectra, PointSpectra, WindSeries, regrid_spectrum};
pub use coupling::{
    CoupledWaves2D, CoupledWavesStats, DEFAULT_BED_ROUGHNESS, DEFAULT_COUPLING_H_DRY,
    SeaStateFields, WaveCoupling2D, WaveForceForm,
};
pub use dispersion::{dsigma_ddepth, group_velocity, group_velocity_ratio, wavenumber};
pub use model::{
    DEFAULT_DEPTH_MIN, SpectralAdvection, WaveModel2D, WaveTimeStepLimit, WaveTimeStepLimits,
    WaveWorkspace,
};
pub use nonlinear::{Quadruplets, shallow_water_factor};
pub use sources::{
    DEFAULT_LIMITER, DEFAULT_RATE_LIMITER, DEFAULT_TAIL_POWER, Dissipation, GrowthLimiter,
    SourceIntegration, SourceTerms, Wind, breaking_fraction,
};
pub use spectrum::{SpectralGrid, WaveParameters, wrap_angle};
pub use state::WaveSolution;
pub use stokes::StokesDriftField;

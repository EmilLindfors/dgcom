//! # dg-rs
//!
//! A Discontinuous Galerkin library for solving hyperbolic PDEs.
//!
//! This crate provides the core building blocks for DG methods:
//! - Polynomial basis functions (Legendre)
//! - Quadrature rules (Gauss-Lobatto)
//! - DG operators (mass, differentiation, LIFT)
//! - Mesh representation
//! - Numerical fluxes
//! - Time integration (SSP-RK3)
//! - Conservation law abstractions (advection, shallow water)
//! - Harmonic analysis for tidal time series
//!
//! ## Performance Features
//!
//! Enable high-performance features with:
//! ```bash
//! cargo build --release --features "parallel,simd,mimalloc"
//! ```
//!
//! - `parallel`: Multi-threaded RHS computation via Rayon
//! - `simd`: SIMD-optimized kernels via Pulp (auto-detects AVX2/AVX-512)
//! - `mimalloc`: High-performance allocator (5-15% speedup)

// Use mimalloc as the global allocator when the feature is enabled
#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

pub mod analysis;
pub mod basis;
pub mod boundary;
pub mod equations;
pub mod flux;
pub mod io;
pub mod mesh;
pub mod operators;
pub mod particles;
pub mod physics;
pub mod polynomial;
pub mod profiling;
pub mod simulation;
pub mod solver;
pub mod source;
pub mod tides;
pub mod time;
pub mod types;
pub mod vertical;
pub mod waves;

// Re-export main types for convenience
// 1D types
pub use basis::Vandermonde;
pub use equations::{Advection1D, ConservationLaw, ShallowWater1D, ShallowWater2D};
pub use flux::{
    SWEFluxType2D, compute_flux_swe_2d, hll_flux_swe_2d, lax_friedrichs_flux, roe_flux_swe_2d,
    rusanov_flux_swe_2d, upwind_flux,
};
pub use mesh::Mesh1D;
pub use operators::DGOperators1D;
pub use solver::{
    BoundaryCondition, DGSolution1D, SWEFluxType, SWERhsConfig, SWESolution, SWEState,
    TVBParameter, apply_swe_limiters, compute_dt_swe, compute_rhs, compute_rhs_swe,
    positivity_limiter, tvb_limiter,
};

#[cfg(feature = "parallel")]
pub use solver::compute_rhs_parallel;
pub use time::{
    ModelClock, SWETimeConfig, compute_dt, run_swe_simulation, ssp_rk3_step, ssp_rk3_step_timed,
    ssp_rk3_swe_step, ssp_rk3_swe_step_timed, total_energy, total_mass, total_momentum,
};

// 2D types
pub use basis::Vandermonde2D;
pub use boundary::{
    BCContext2D, BathymetryValidationConfig, BathymetryValidationResult, BoundaryState,
    BoundaryTides, CharacteristicOBC, ConstantDischarge2D, Discharge2D, ElevationOnly,
    ExternalState, ExternalStateProvider, Extrapolation2D, FixedState2D, HarmonicTidal2D,
    HarmonicTide, ParentTimeSeries, Reflective2D, SWEBoundaryCondition2D, SpongeConfig, StillWater,
    Tidal2D, TidalAtlas, TidalBCType, TidalConstituent, TidalSimulationBuilder,
    format_bathymetry_warning, validate_bathymetry_convention,
};
pub use boundary::{NestingOptions, NestingRelaxation2D, OceanModelState};
pub use equations::Advection2D;
pub use mesh::{BoundaryConfig, BoundaryTag, Mesh2D, Mesh2DBuilder};
pub use operators::{DGOperators2D, GeometricFactors2D, MeshTransfer2D};
pub use solver::{
    AdvectionBoundaryCondition2D,
    AdvectionFluxType,
    ConstantBC2D,
    DGSolution2D,
    // Diagnostics
    DiagnosticsTracker,
    DirichletBC2D,
    ExtrapolationTracerBC3D,
    FixedTracerBC3D,
    PeriodicBC2D,
    ProgressReporter,
    Rhs3DConfig,
    SWE2DRhsConfig,
    SWEDiagnostics2D,
    SWEFormulation2D,
    SWESolution2D,
    SWEState2D,
    SystemSolution2D,
    TracerBCContext3D,
    TracerBoundaryCondition3D,
    UpwindTracerBC3D,
    // Wetting/drying
    WetDryConfig,
    // SWE 2D limiters
    apply_swe_limiters_kuzmin_2d,
    apply_wet_dry_correction,
    apply_wet_dry_correction_all,
    compute_dt_advection_2d,
    compute_dt_swe_2d,
    // Tracer transport
    compute_dt_viscosity,
    // 3D RHS
    compute_momentum_rhs_3d,
    compute_rhs_advection_2d,
    compute_rhs_swe_2d,
    compute_rhs_swe_2d_into,
    compute_transport_rhs_3d,
    current_cfl_2d,
    positivity_cfl_swe_2d,
    swe_kuzmin_limiter_2d,
    swe_positivity_limiter_2d,
    total_energy_2d,
    total_mass_2d,
    total_momentum_2d,
};
#[cfg(feature = "parallel")]
pub use solver::{
    compute_dt_swe_2d_parallel, compute_rhs_swe_2d_parallel, compute_rhs_swe_2d_parallel_into,
};

// Burn GPU acceleration exports
#[cfg(feature = "burn")]
pub use solver::burn::{
    BurnConnectivity, BurnError, BurnOperators2D, BurnSWESolution2D, compute_rhs_swe_2d_burn,
    hll_flux_batched,
    rhs::{BurnGeometricFactors2D, BurnRhsConfig},
    roe_flux_batched,
};
pub use source::{
    AtmosphericPressure2D, CombinedSource2D, CoriolisSource2D, DragCoefficient,
    GriddedAtmosphere2D, GriddedSurfaceFluxes, GriddedWindStress, HorizontalViscosity2D,
    HydrostaticReconstruction2D, P_STANDARD, RectangularBoundary, SourceContext2D, SourceTerm2D,
    SpongeLayer2D, SpongeProfile, TidalPotential, TidalPotentialConstituent, WindStress2D,
};
#[cfg(feature = "burn")]
pub use time::{BurnTimeConfig, compute_dt_burn, run_swe_2d_burn, ssp_rk3_step_burn};
pub use time::{
    // Coupled SWE + tracer integration
    run_advection_2d,
    ssp_rk3_step_2d,
    ssp_rk3_step_2d_timed,
};

// Analysis types
pub use analysis::{
    AnalysisError,
    ComparisonMetrics,
    ConstituentComparison,
    ConstituentComparisonSummary,
    HarmonicAnalysis,
    Inference,
    // Tide gauge validation
    ModelExtractor,
    PrecomputedExtractor,
    ReferenceConstant,
    ReferenceFit,
    // Stability monitoring
    StabilityMonitor,
    StabilityStatus,
    StabilityThresholds,
    StabilityWarning,
    StationValidationResult,
    TideGaugeStation,
    TimeSeries,
    ValidationSummary,
    compare_harmonics,
    norwegian_stations,
    validate_stations,
};

// I/O types
pub use io::{
    AtmosphereField, AtmosphereReader, AtmosphereState, OceanModelReader, OceanState, RiverForcing,
};
pub use io::{
    BathymetryStatistics,
    BedRaster,
    BoundaryTimeSeries,
    CoastlineData,
    CoastlineError,
    CoastlineStatistics,
    ConstituentData,
    ConstituentEntry,
    ConstituentFileError,
    CoordinateProjection,
    FROYA_BBOX,
    GeoBoundingBox,
    GeoTiffBathymetry,
    GeoTiffError,
    LocalProjection,
    NORWAY_BBOX,
    // NorKyst text ingest
    NorKystLevel,
    NorKystRecord,
    NorKystTextData,
    NorKystTextError,
    // Tide gauge I/O
    TideGaugeFile,
    TideGaugeFileError,
    TimeSeriesFileError,
    TimeSeriesRecord,
    UtmProjection,
    VtkError,
    constituent_period,
    files_to_observation_map,
    parse_constituents,
    parse_norkyst_text_str,
    parse_timeseries,
    read_constituent_file,
    read_norkyst_text_file,
    read_tide_gauge_directory,
    read_tide_gauge_file,
    read_timeseries_file,
    write_tide_gauge_file,
    write_vtk_series,
    write_vtk_swe,
};
#[cfg(feature = "netcdf")]
pub use io::{
    FILL_VALUE_F32, FILL_VALUE_F64, NetCDFError, NetCDFMeshInfo, NetCDFWriter, NetCDFWriterConfig,
    is_valid_f32, is_valid_f64,
};
#[cfg(feature = "parquet")]
pub use io::{
    NorKystGridData, NorKystGridError, NorKystGridRow, read_norkyst_parquet,
    read_norkyst_parquet_glob,
};

// Mesh types (additional exports)
pub use mesh::{LandMask2D, LandMaskStatistics};

// Physics module types
pub use physics::{
    PhysicsBuilder, PhysicsConfig, PhysicsModule, PhysicsModuleInfo, SWEPhysics2DBuilder,
};

// Simulation runner types
pub use simulation::{Simulation, SimulationConfig, SimulationResult};

// Vertical coordinate types (for 3D)
pub use vertical::{
    DoubleTanhStretching, ROMSVstretching4, SigmaGrid, SongHaidvogelStretching, Stretching,
    UniformStretching,
};

// Strongly-typed domain types
pub use types::{
    Bounds2D, Depth, ElementIndex, Elevation, FaceIndex, LevelIndex, NodeIndex, PhysicalZ,
    Resolution2D, SideBoundaries, Sigma,
};

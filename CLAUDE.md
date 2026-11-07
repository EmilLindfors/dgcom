# Claude Guidelines for DGCOM

This is a high-performance Discontinuous Galerkin (DG) solver for coastal ocean modeling, targeting simulation of currents along the Norwegian coast. Numerical accuracy and computational efficiency are critical.

## Project Context

### Primary Application
- **Norwegian coastal current simulation**: Complex fjord geometry, strong tidal forcing, steep bathymetry
- **Shallow water equations**: Mass and momentum conservation with Coriolis, bottom friction, wetting/drying
- **Operational oceanography**: Results must be accurate enough for real-world decision support

### Design Philosophy
- **Long-term architecture over short-term patches**: the crate is an early-stage experiment with no external users. Breaking the public API is fine (rename, merge, delete, re-shape types and functions) whenever it gives a better design or better performance; do not keep compatibility shims, deprecated paths or duplicate APIs, and port the callers (examples, tests) instead. Fix the cause, not the symptom.
- **Correctness first**: Numerical methods must converge at theoretical rates
- **Performance matters**: Will scale to millions of elements; every unnecessary allocation counts
- **GPU-ready architecture**: Design data structures for future GPU port (cudarc)
- **Composable**: Clean separation between operators, mesh, flux, and time integration

## Code Standards

### Mathematical Rigor
- All numerical methods must have theoretical backing (cite Hesthaven-Warburton for DG)
- Convergence tests are mandatory for any spatial/temporal discretization
- Conservation properties must be verified for hyperbolic systems
- Document the mathematical formulation in module-level doc comments

### Performance Requirements
- Prefer stack allocation and slices over heap allocation in hot paths
- Use `faer` for linear algebra (already a dependency)
- Element-local operations should be vectorization-friendly
- Profile before optimizing; use `cargo bench` with criterion when benchmarks exist

### Testing Requirements
- **Unit tests**: Every operator must be tested for polynomial exactness
- **Convergence tests**: Verify (N+1) order accuracy for smooth solutions
- **Conservation tests**: Total mass/momentum must be preserved (periodic BCs)
- **Regression tests**: Any bug fix must include a test that would have caught it

### Note Down Opportunities
The crate is at an early stage, so experimenting is welcome. While working on any task, watch for opportunities: suboptimal code, better approaches, simplifications, in the numerics, performance (e.g. how parallel code is run), API shape or duplicated paths.
- **Note them in `TODO.md`**: one checkbox line each, starting with the item's ID, under the matching area (not a separate list). Put the measurement or the concrete pointer (file, function) and the reasoning in that `notes/<area>.md` file when it needs more than the line.
- **Report them**: list what you noted in the end-of-task summary.

### Roadmap and notes
- `TODO.md` holds **open items only**, one line each, with an ID (P1.3, F.4, …) and a pointer into `notes/`.
- `notes/` holds the context, one file per area (index in `notes/README.md`): findings, measurements, decisions, dead ends, finished work. Before starting on an item, read its notes file.
- When an item is done, delete its `TODO.md` line and record the outcome (date, PR, numbers) under its heading in the notes file; notable changes also go in `CHANGELOG.md`.

## Architecture Overview

See `docs/architecture.md` for the full annotated module tree and layer-by-layer
status, `docs/accuracy.md` for verified convergence and conservation, and
`TODO.md` + `notes/` for the open work and its context (the 2026-09-25 review
that set the direction is `notes/review-2026-09-25.md`).

```
src/
├── types/          # Newtypes (indices, Depth, Sigma, bounds)
├── polynomial/     # Legendre polynomials, GLL nodes/weights (1D + 2D)
├── basis/          # Vandermonde matrices for nodal-modal transforms
├── operators/      # Dr/Ds, Mass, LIFT, geometric factors (reference element)
├── mesh/           # Mesh1D/Mesh2D, connectivity, bathymetry, land mask, Gmsh
├── flux/           # Numerical fluxes (upwind, Lax-Friedrichs, Roe, HLL)
├── equations/      # ConservationLaw trait, advection, SWE 1D/2D, EOS
├── source/         # Source terms (Coriolis, friction, wind, tidal, sponge,
│                   # well-balanced bathymetry)
├── boundary/       # CharacteristicOBC + providers (tides, atlas, nesting), walls
├── solver/         # Solution state (SoA), RHS kernels, limiters,
│                   # wetting/drying, SIMD, burn GPU prototype
├── time/           # Integrable/TimeIntegrator traits, SSP-RK3, mode splitting
├── vertical/       # Sigma grid, stretching functions (3D)
├── physics/        # PhysicsModule trait, SWEPhysics2D, Hydrostatic3D,
│                   # vertical mixing/diffusion/velocity
├── tides/          # Tidal astronomy (V₀, nodal f/u)
├── simulation/     # Simulation / Simulation3D runners
├── waves/          # Spectral wave model (action balance on the DG mesh)
├── particles/      # Lagrangian particle tracking (2D/3D, lice behaviour)
├── io/             # NetCDF, VTK, GeoTIFF, coastline, projections, obs readers
└── analysis/       # Harmonic analysis, skill metrics, tide gauge, ADCP
```

### Key Types
- `DGOperators1D` / `DGOperators2D`: reference element operators bundled together
- `Mesh1D` / `Mesh2D`: physical mesh with neighbor/face connectivity
- `DGSolution2D` / `SWESolution2D`: SoA nodal storage `[n_elements × n_nodes]` per variable
- `Solution3D`: 3D state, `[element × node × level]` columns over a `SigmaGrid`
- `Integrable` + `SSPRK3`: the generic time-integration path (prefer over the
  legacy per-type `ssp_rk3_*` free functions, which are slated for deletion)
- `MultirateSSPRK3`: local time stepping for `Simulation` (every element at a power-of-two fraction of the step; conservative and SSP, second order across levels); physics modules opt in with `PhysicsModule::local_time_stepping`
- `MultiBoundaryCondition2D`: per-tag dispatch of open/closed boundary conditions
- `CharacteristicOBC<P>`: every open boundary (radiation, tides, nesting); the external data comes from an `ExternalStateProvider` (`StillWater`, `HarmonicTide`, `BoundaryTides` from a `TidalAtlas`, `OceanModelState` nesting in an `OceanModelReader` parent, with its `NestingRelaxation2D` band)
- `GeoGrid` / `FieldSeries`: external models' structured lon/lat grids and strided `[time][point]` fields; `OceanModelReader` (NorKyst/ROMS) and `AtmosphereReader` (MET Nordic/MEPS/ERA5, → `GriddedAtmosphere2D` wind stress and pressure) are built on them
- `ModelClock`: the UTC instant of simulation time 0, for tides (V₀, nodal f/u), parent-model time and output units

## Common Tasks

### Adding a New Flux Function
1. Add function to `src/flux/upwind.rs` (or new file)
2. Follow signature: `fn flux(u_minus, u_plus, a, normal) -> f64`
3. Add unit tests verifying continuous solution gives physical flux
4. Export from `src/flux/mod.rs` and `src/lib.rs`

### Adding a New Time Integrator
1. Add to `src/time/` module
2. For time-dependent BCs, pass time to RHS function (see `ssp_rk3_step_timed`)
3. Verify order of accuracy with ODE test (exponential growth)
4. Document stability region if relevant

### Extending the 3D layer
The 3D solver is ROMS-style: 2D DG horizontal × finite-difference vertical on
sigma coordinates with mode splitting (open work: `TODO.md` P4; context:
`notes/3d-*.md`). When working there:
- Depth convention is `h = eta - B` with `B` = bed elevation (negative under water) — everywhere
- New vertical/3D discretizations need convergence + conservation tests before merge (the existing gates are listed in `notes/3d-validation.md`)
- Layer-thickness (Hz) weighted fluxes for anything advected; divide back to concentration/velocity after
- Known open numerics: see `TODO.md` P4 (the history is in `notes/3d-*.md`)

## Norwegian Coast Specifics

### Physical Considerations
- **Fjords**: Long, narrow, deep - need anisotropic mesh refinement
- **Tides**: M2 dominant, strong currents in narrow straits
- **Fresh water**: River runoff creates stratification (future 3D)
- **Coriolis**: f ≈ 1.2×10⁻⁴ s⁻¹ at 60°N, important for circulation

### Numerical Considerations
- **Wetting/drying**: Tidal flats require robust treatment
- **Steep bathymetry**: Well-balanced schemes for lake-at-rest
- **Open boundaries**: Radiation conditions, tidal forcing
- **Bottom friction**: Manning/Chézy formulation

## What NOT to Do

- **Don't break convergence**: Any change to spatial discretization must pass convergence tests
- **Don't ignore conservation**: DG should conserve mass exactly (up to machine precision)
- **Don't allocate in RHS**: The RHS function is called thousands of times per simulation
- **Don't use f32**: Coastal models need f64 precision for stability
- **Don't skip BC timing**: Time-dependent BCs must be evaluated at correct RK stage times

## Verification Checklist

Before any PR:
- [ ] `cargo test` passes (all tests)
- [ ] `cargo test --features parallel` passes
- [ ] `cargo clippy` has no warnings
- [ ] Convergence rates match theory (check `docs/accuracy.md`)
- [ ] No new allocations in hot paths (check with profiler if unsure)
- [ ] `CHANGELOG.md` is updated for notable numerical, API, dependency, or documentation changes

## Windows Native Dependencies

The default feature set includes NetCDF I/O, so Windows builds need native HDF5 and netCDF-C. Prefer the Miniforge/conda-forge workflow documented in `docs/windows-native-deps.md`.

Important details:
- Use `hdf5=1.14.4` and `libnetcdf<4.10`; current HDF5 `2.x` packages are rejected by `hdf5-metno-sys 0.10.1`.
- Persist `HDF5_DIR` and `NETCDF_DIR` to the conda environment root, not directly to `Library`.
- Activate `roms-rs` before running default-feature Cargo commands.

```powershell
conda activate roms-rs
cargo check
```

For checks that do not need NetCDF I/O:

```powershell
cargo check --no-default-features --features parallel,simd
```

## References

- Hesthaven & Warburton, "Nodal Discontinuous Galerkin Methods" (2008) - DG bible
- Karniadakis & Sherwin, "Spectral/hp Element Methods" (2005) - Polynomial bases
- LeVeque, "Finite Volume Methods for Hyperbolic Problems" (2002) - Conservation laws
- Toro, "Riemann Solvers and Numerical Methods for Fluid Dynamics" (2009) - Numerical fluxes

## Quick Commands

```bash
# Run all tests
cargo test

# Run with parallel feature
cargo test --features parallel

# Run the tests the way CI does (opt-level 1, ~15x faster than the unoptimized suite)
cargo test --profile ci

# The fast tier that pull requests run (all but the slow gates in .config/nextest.toml)
cargo nextest run --profile pr --cargo-profile ci --no-default-features --features parallel,simd

# Windows default-feature build with native HDF5/netCDF-C
conda activate roms-rs
cargo check

# Run convergence tests with output
cargo test --test convergence_test -- --nocapture

# Run example
cargo run --release --example advection_1d

# Check for issues
cargo clippy

# Profiling (see scripts/ for details)
./scripts/flamegraph.sh froya_real_data       # Generate flamegraph SVG
./scripts/flamegraph.sh froya_real_data netcdf  # With features
./scripts/perf-stat.sh froya_real_data        # Quick CPU stats
./scripts/samply.sh froya_real_data           # Interactive profiler UI
```

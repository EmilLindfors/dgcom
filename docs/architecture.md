# Architecture

A DG (Discontinuous Galerkin) solver for Norwegian coastal ocean modeling: a
2D barotropic SWE model (well-balanced split-form DGSEM with wetting and
drying, characteristic open boundaries, NorKyst nesting, local time stepping),
a ROMS-style 3D layer (2D DG horizontal × finite-difference vertical, sigma
coordinates, mode splitting), a spectral wave model and Lagrangian particle
tracking, aimed at currents, waves and sea-lice dispersion around salmon farms.

This document describes what exists and how it fits together. Open work is in
`TODO.md`, with its context in `notes/`; verified accuracy is in
`docs/accuracy.md`. The 2026-09-25 review that set the current direction is
`notes/review-2026-09-25.md`.

## Design Philosophy

**Evolve, don't over-engineer.** Start concrete, abstract when patterns emerge.

| Principle | Approach |
|-----------|----------|
| Single crate (for now) | 19 modules, ~127k lines; a `dg-core` / `roms-rs` workspace split is the likely end state (TODO P6) |
| Concrete first | `Mesh1D`, `Mesh2D`; generic traits (`Integrable`, `SourceTerm2D`) added once 2+ implementations existed |
| faer for LA | Dense element-local operators; QR/SVD available for least-squares (harmonic analysis) |
| f64 everywhere | Coastal stability requires double precision |
| Norwegian focus | Fjords, tides, wetting/drying — not generality |

## Module Structure (current)

```
src/
├── lib.rs            # Public API, re-exports (oversized — TODO P6)
├── types/            # Newtypes: indices, bounds, physical (Depth, Sigma), sides
│
│  # ---- DG core (dimension/equation-agnostic) ----
├── polynomial/       # Legendre P_n, GLL nodes/weights (1D + tensor-product 2D)
├── basis/            # Vandermonde matrices, nodal<->modal (1D + 2D)
├── operators/        # Dr/Ds, mass, LIFT, GeometricFactors2D (per-node
│                     # isoparametric), MeshTransfer2D between meshes
├── mesh/
│   ├── core/         # Mesh1D, Mesh2D (quads), Mesh2DBuilder, PointLocator2D
│   ├── data/         # Bathymetry, land mask, boundary tags
│   ├── io/           # Gmsh MSH 4.1 (ASCII/binary) + 2.2 reader, 2.2 writer
│   └── traits/       # Mesh traits, Point; MeshGPUData (unimplemented)
├── flux/             # Upwind, Lax-Friedrichs, Roe, HLL; 2D SWE fluxes;
│                     # entropy-conservative/-stable SWE fluxes (Wintermeyer)
│
│  # ---- Equations & physics ----
├── equations/        # ConservationLaw trait; advection 1D/2D, SWE 1D/2D, UNESCO EOS
├── source/           # SourceTerm1D/2D traits; Coriolis, friction, wind, tidal
│                     # potential, viscosity, bathymetry, sponge; net-cage drag
│                     # (point-implicit); gridded weather-model wind + pressure
│                     # (split for 3D: wind to the columns); rivers as volume
│                     # sources (2D source term; 3D via the mode splitter, with
│                     # a vertical profile and the river water's T/S);
│                     # well-balanced hydrostatic reconstruction
├── boundary/         # CharacteristicOBC + external-state providers (still
│                     # water, harmonic tide, tidal atlas, NorKyst nesting with
│                     # relaxation band, inverse barometer), reflective,
│                     # clamped tidal, discharge, multi-BC dispatch
│
│  # ---- Solver ----
├── solver/
│   ├── state/        # DGSolution1D/2D, SWESolution2D (SoA), Solution3D
│   ├── core/         # Solution containers; element-block loops of the 3D
│   │                 # kernels (blocks.rs: parallel with `parallel`, bit-
│   │                 # identical to serial; thread-local scratch pool)
│   ├── rhs/          # RHS kernels: scalar/SWE 1D/2D (collocated or entropy-stable
│   │                 # split form), diffusion (BR1), 3D advection,
│   │                 # 3D layer transports (Ω and inventory-form tracers,
│   │                 # corrected to the barotropic DU_avg2; vertical
│   │                 # advection upwind/centred/Akima/TVD/limited Akima,
│   │                 # end layers bounded by their extrapolated gradient),
│   │                 # (split-form horizontal tracer and momentum
│   │                 # advection, MomentumAdvectionForm), baroclinic
│   │                 # PGF (σ-pairs or constant depth, balanced reference),
│   │                 # 3D horizontal viscosity of the shear
│   │                 # (BR1 per layer, constant + Smagorinsky),
│   │                 # 3D RHS assembly (momentum / transport parts)
│   ├── limiters/     # Zhang-Shu positivity, Kuzmin/TVB slope limiters (SWE, 3D tracers)
│   ├── algorithms/   # Wetting/drying, tridiagonal (Thomas) solve
│   ├── simd/         # fearless_simd kernels of the Standard kernel + batched faer paths
│   ├── diagnostics/  # Runtime diagnostics (2D conservation); 3D potential
│   │                 # and reference potential energy (PotentialEnergy3D:
│   │                 # spurious mixing, Ilıcak et al. 2012)
│   ├── probe.rs      # Probe2D: the DG solution at a point (stations)
│   └── burn/         # Burn GPU prototype — incomplete, physically wrong
│                     # (TODO P0.11/P2.7)
├── time/             # Integrable + TimeIntegrator traits, generic SSPRK3, SSPRK43,
│                     # multirate (local time stepping, Multirate<B>),
│                     # mode_split (barotropic subcycling); legacy per-type
│                     # ssp_rk3_* variants slated for deletion (TODO P6)
│
│  # ---- 3D vertical layer ----
├── vertical/         # SigmaGrid, Song-Haidvogel / ROMS Vstretching=4
├── physics/          # PhysicsModule trait, SWEPhysics2D(+builder),
│                     # Hydrostatic3D, vertical mixing (GLS k-ε/k-ω/generic
│                     # with wave breaking and Charnock roughness, its k, ψ
│                     # advected over the w-cells,
│                     # Pacanowski–Philander)/diffusion/velocity,
│                     # quadratic bottom drag (constant or log-layer C_d),
│                     # net-cage drag per layer (cage_drag),
│                     # per-column surface stress (SurfaceStress3D) and
│                     # heat/freshwater fluxes (SurfaceFluxes3D; COARE 3.5
│                     # bulk formulae and Jerlov absorption in air_sea), EOS
│
│  # ---- Application layer ----
├── tides/            # Tidal astronomy: Doodson/Schureman V₀, nodal f and u
├── simulation/       # Simulation runner (2D), Simulation3D
├── waves/            # Spectral wind waves (F.4): action balance per (σ, θ)
│                     # component on the DG mesh (WaveModel2D: DG propagation,
│                     # refraction, current frequency shift), dispersion,
│                     # SpectralGrid (JONSWAP, H_s/T_p/direction, Stokes drift,
│                     # radiation stress), sources (Komen wind/whitecapping,
│                     # DIA quadruplets, bed friction, breaking, tail, limiter),
│                     # StokesDriftField for the particles
├── particles/        # Lagrangian tracking: RK4 on the DG velocity, random
│                     # walk, face-by-face walk with wall reflection, exits;
│                     # 3D: σ-levels, Visser's vertical walk, sinking/settling;
│                     # behaviour (swimming salmon-lice larvae, clear-sky light);
│                     # the waves' Stokes drift added to any flow (stokes.rs);
│                     # site-to-site connectivity: exposure in contact zones,
│                     # matrix per release group, spread across seeds
│                     # (connectivity.rs)
├── io/               # NetCDF output; parent-ocean and weather-model readers
│                     # on lon/lat grids (GeoGrid, FieldSeries); VTK, GeoTIFF
│                     # bathymetry, GSHHS coastline, projections, observations;
│                     # snapshot files for replay; 2D and 3D restart files (restart.rs)
└── analysis/         # Harmonic (tidal) analysis, tidal current ellipses,
                      # skill metrics, tide gauge, ADCP, stability monitoring
```

Layering intent (dependencies point up this list): `types` → DG core →
equations/physics → solver → simulation/io/analysis. The io/boundary modules
are the adapter edge; the DG core must stay free of I/O and feature flags
(`netcdf`, `parallel`, `simd` gate adapters and kernels, not math).

## Status by Layer

Summary as of 2026-10-10; the details and open items are in `notes/` and `TODO.md`.

| Layer | State |
|-------|-------|
| 1D/2D DG core | Solid operators/fluxes/SSP-RK3: verified convergence (P1–P5 1D, P1–P3 2D), machine-precision conservation, correct Roe/HLL. Open: nonlinear convergence with bathymetry on curved meshes (P1.7) |
| Well-balancing, wet/dry | Split forms (`SWEFormulation2D::EntropyStable`, and `WetDry` with second-order shoreline subcells, the wet/dry default) are well-balanced for any nodal bathymetry with exact mass conservation; lake at rest over the real Frøya bed holds to 1.5e-10 m/s. Open: η-based limiting, dry-front lag (P1.2) |
| Geometry | General straight-sided quadrilaterals in 2D and 3D (per-node isoparametric factors, free-stream preserving and well-balanced); all-quad Gmsh MSH 4.1/2.2 meshes, a coastline-fitted Frøya mesh (`docs/gmsh-meshes.md`). Open: triangles, curved faces, CSR, the stratified rest state over unsmoothed cliffs (P1.3) |
| Time integration | Generic `SSPRK3`/`SSPRK43` over `Integrable`, fused per-element stages, local time stepping (`MultirateSSPRK3`); legacy copies pending deletion (P6) |
| Mode splitting | One filtered barotropic pass per step, second-order slow coupling, and the DU_avg2 transport for 3D continuity (P4.1). Open: a `rufrc`-style G, the forward-backward fast mode |
| 3D physics | Sigma grid, tridiagonal and implicit diffusion solid; mode splitting gated (P4.1); tracers constancy-preserving and conservative in inventory form, with layer fluxes and Ω consistent with the barotropic transport (P4.2); PGF from pairwise pressure differences, σ-pairs (consistent in energy with the split-form tracer advection, so a stratified fluid at rest over a seamount stays at rest) or at constant depth, both exact for constant N² at rest on any slope, the σ form's error at rest taken back by a balanced reference state, face jumps lifted (P4.3, P4.6); a horizontal Kuzmin tracer limiter that leaves a stratified fluid at rest over slopes (bounds at each node's height from a 3D vertex patch, the departure from the element's own column scaled, an optional reference profile for curved stratification; P4.6); 3D wetting and drying (thin columns, element-mean tracers where the 2D pass balances only elements, P4.5); quadratic bottom drag, implicit in the pass and the vertical solve (P4.4); net-cage drag on the layers a net reaches, treated like the bottom drag (F.1); open boundaries with the interior's layer profile and extrapolated 3D velocity (P4.2); Akima/TVD vertical advection, split-form (kinetic-energy preserving) momentum advection that holds a sharp interface without viscosity, and horizontal viscosity of the shear, whose spurious mixing (the reference potential energy's growth) is set by the grid Reynolds number (P4.5); GLS turbulence per column, gated by Kato–Phillips entrainment and the open-channel log layer, with Craig & Banner wave breaking gated by the shear-free layer, and advected by the 3D flow over the w-cells, constancy-preserving (P4.4); a surface stress per column (`SurfaceStress3D`: gridded weather-model wind, analytic fields); surface heat and freshwater fluxes per column (`SurfaceFluxes3D`: the COARE 3.5 bulk formulae on a weather model's fields, the shortwave absorbed with depth by a Jerlov water type, conservative to round-off; P4.4); rivers as volume sources with their own T/S, constancy-preserving (P1.6/P5.1), gated by an estuarine circulation in an idealised fjord (P4.6). Open: the 3D items of P4.2–P4.6 |
| Boundaries/nesting | One characteristic OBC (`CharacteristicOBC` + external-state providers), NorKyst boundary tides from a harmonic atlas corrected to the gauges, `ModelClock`; 2D and 3D NorKyst nesting with a relaxation band; gridded MET Nordic/MEPS wind and pressure. Open: a real NorKyst 3D nested run (P1.5) |
| Waves | Spectral action balance on the DG mesh (F.4): DIA, implicit refraction and frequency shifting, a depth limit, MET Norway boundary spectra, coupled to the currents on a mesh of its own (radiation stress or dissipation force, Stokes drift, bed stress, GLS roughness) |
| Particles | 2D and 3D tracking on the DG velocity, random walks, salmon-lice behaviour, site-to-site connectivity (F.2) |
| Validation | Mausund gauge, 15 days on the coastline mesh: RSS 3.37 cm over the main constituents (NorKyst 14 cm). Open: M2 gain, weak overtides, a month-long run, observed currents, the ROMS cost benchmark (P3) |
| Performance | SoA + rayon, local time stepping, SSP-RK(4,3), the split form batched across elements with fearless_simd. The speed claim against ROMS is unproven (P3.2); the cost levers are in P2 |
| GPU | Burn prototype physically wrong; fix or replace with fused f64 kernels over CSR (P0.11/P2.7) |

## What NOT to Do

1. **Don't break convergence or conservation** — any spatial/temporal change
   must keep the convergence and conservation tests passing.
2. **Don't allocate in the RHS** — and don't add new APIs shaped
   `fn rhs(&state) -> State`; use write-into signatures.
3. **Don't extend the Burn module** — fix or replace it is the open P2.7
   decision; a future GPU port goes through flat SoA/CSR data + fused
   f64 kernels.
4. **Don't add another hand-rolled integrator or run loop** — implement
   `Integrable` and use `SSPRK3`/`Simulation`.
5. **Don't preserve backwards compatibility** — pre-1.0, no external
   consumers: rename/delete cleanly and fix call sites in the same change.

## Testing Strategy

Four tiers, mandatory for new discretizations in 2D and 3D (the 3D gates
are listed in `notes/3d-validation.md`):

1. **Unit**: operator exactness on polynomials, flux consistency.
2. **Convergence**: observed order ≥ N+1 on smooth solutions, multiple
   resolutions (`tests/convergence_test.rs`).
3. **Conservation**: mass/momentum to machine precision on periodic domains,
   including long-time (50+ period) runs.
4. **Physics regression**: lake-at-rest (must use non-trivial bathymetry),
   dam-break vs exact Riemann, dynamic wet/dry (Ritter dry dam break and
   Thacker bowl in `tests/wet_dry_2d_test.rs`),
   stratified lake-at-rest and seamount for the 3D PGF
   (`src/solver/rhs/baroclinic.rs`, `src/simulation/simulation_3d.rs`).

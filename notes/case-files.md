# Generic cases and case files (P8)

## Goal (2026-11-07)

A modeller from the ROMS community, or anyone with a coastline, a bed and some forcing, should be able to run their own coastal domain without writing Rust: one case file, one binary. Until now each domain has been an example with its constants hard-wired. The real application, `examples/froya_real_data.rs`, is 6,655 lines: 77 `key=value` options plus Frøya constants (`FROYA_BBOX`, `DEM`, `F_CORIOLIS = 1.31e-4`, `MANNING_N`, the gauge constituent lists). The library pieces are already generic: `Mesh2D` and the Gmsh reader, `BedRaster`, `coriolis_parameter(lat)`, `CharacteristicOBC` with its providers, `OceanModelReader`, `AtmosphereReader` and `NetCDFWriter`. What is missing is the layer that assembles them from a description.

The aquaculture work continues on top of this. Frøya and Bodø become case files, and the farm, waves and particles become optional sections.

## Decisions proposed

- **Format: TOML.**
  - It allows comments, it is what Rust tooling uses (Cargo), and it is the closest of the three to ROMS's `ocean.in` (`KEY = value`, with sections).
  - The schema is plain `serde` structs, so JSON comes almost for free for scripts that generate cases, from the file extension.
  - No YAML: `serde_yaml` is unmaintained, and indentation errors are a poor first experience.
  - New dependencies: `serde`, `toml` (and `serde_json` for JSON), behind a default `case` feature.
- **Strict schema.**
  - `#[serde(deny_unknown_fields)]` everywhere, so a misspelt key is an error. This replaces `Arguments::refuse_unread` (P6) for cases.
  - Units in key names (`duration_hours`, `band_km`).
  - Each default lives in one place, the schema, and the docs list it.
- **Effective configuration.** The binary resolves defaults, writes the complete case next to the output (`case.resolved.toml`) and into each restart, and compares that on resume. This closes P5.3's gap: a changed default or an omitted option is no longer missed.
- **Binary:** `src/bin/` subcommands:
  - `run case.toml`
  - `check case.toml`: validate and print the resolved case
  - `mesh case.toml`: build the mesh and write it to Gmsh/VTK to inspect
  - Overrides as `--set physics.manning = 0.03` for sweeps.

  The binary is `dgcom` (`dgcom run case.toml`), after the crate (renamed 2026-11-07, PR #192).
- **One path.** Each part of `froya_real_data` that moves into the case layer is deleted from the example (CLAUDE.md: no duplicate paths). The example ends as a thin wrapper or disappears.

## Schema sketch

```toml
[run]
start = "2025-06-15T00:00:00Z"    # ModelClock epoch
duration_hours = 24
ramp_hours = 1

[domain]
# One of: a Gmsh file, or a rectangle over a bbox clipped to water by the bed
mesh = { gmsh = "meshes/froya.msh" }
# mesh = { grid = { bbox = [8.0, 63.6, 9.2, 64.0], nx = 120, ny = 60 } }
order = 2
projection = "auto"               # local transverse Mercator about the centre

[bathymetry]
source = "data/froya_topobathy.tif"  # GeoTIFF, NetCDF grid, or a ROMS grid file
land_mask = "gshhs:f"                # optional, when the raster has no land
min_depth_m = 0.05

[physics]
coriolis = "latitude"             # per node, or { f_plane = 1.31e-4 }
friction = { manning = 0.025 }
wetting_drying = true

[[boundary]]
tag = "open"                      # or "auto": the sides that cross water
condition = "characteristic"
external = { tides = { atlas = "data/froya_atlas.nc" } }
# external = { nesting = { parent = "data/norkyst.nc", band_km = 2, band_minutes = 30 } }

[forcing.atmosphere]
source = ["data/meps.nc"]
pressure = true
inverse_barometer = 0.6

[output]
directory = "output/froya"
fields_minutes = 60
stations = "stations.toml"
```

`[vertical]` (3D), `[waves]`, `[particles]` and `[farm]` are optional sections that switch the solver up.

## For the ROMS community

- **Reading ROMS inputs:**
  - the grid file (`h`, `mask_rho`, `lon_rho`/`lat_rho`) as bed and land mask;
  - the tidal forcing file (`tide_period`, `tide_Eamp`/`Ephase`, `tide_Cmax`/`Cmin`/`Cangle`/`Cphase`, as made from TPXO/FES by the usual tools) as a `TidalAtlas`;
  - boundary and climatology files (`zeta_west`, `ubar_west`, …) as a parent. `OceanModelReader` reads NorKyst's history files today.
- **Writing ROMS names** in the output (P5.2), so existing post-processing works.
- **`docs/coming-from-roms.md`:**
  - map `ocean.in` keywords to case keys (`DT`/`NTIMES` → `run`, `LBC` → `[[boundary]]`, `RDRG`/`Zob` → `friction`, `TNU2` → viscosity);
  - say plainly what has no equivalent yet.
- **A test case ROMS users know,** run from a case file and compared against ROMS: an idealized estuary or a tidal channel with an analytic bed. It doubles as the README quickstart, with no download, and as a CI smoke run.

## Order

1. P8.1: the schema and the binary for the 2D path (mesh, bed, Coriolis, friction, still/harmonic/atlas/nesting boundaries, atmosphere, NetCDF output, stations), with an idealized case and a CI smoke run.
2. P8.2: move `froya_real_data` onto it in stages (2D tides, nesting, 3D, waves, farm), deleting the options from the example as they go; Frøya and Bodø as case files. The 2026-06-12 nested scores (3.0 cm Bodø, 4.2 cm Mausund) must reproduce bit for bit at each stage.
3. P8.3 to P8.5: ROMS inputs, ROMS output names, the "coming from ROMS" doc and the README quickstart.

## P8.1, first part: the case layer and `dgcom` (2026-11-07)

- **In the crate:**
  - `src/case/` holds the schema (`schema.rs`), the loader with `--set` overrides (`mod.rs`) and the builder and runner (`run.rs`).
  - `src/bin/dgcom.rs` provides `run` and `check`.
  - `serde`, `toml` and `serde_json` are plain dependencies, not behind a feature, so CI's `parallel,simd` builds test the case layer too. All three were already in the lock file through other crates.
- **What a case holds:**
  - mesh: a rectangle with a tag per side, or a Gmsh file;
  - bed: a uniform depth or a linear ramp;
  - physics: gravity, Coriolis (`f_plane`, or `latitude` as an f-plane), Manning or quadratic-drag friction (point-implicit), constant or Smagorinsky viscosity, wetting and drying (`froya_real_data`'s positivity limiter and `WetDry`);
  - initial: still water at a level;
  - boundaries: one `[[boundary]]` per open tag (`open`, `tidal`, `river`). Each is a `CharacteristicOBC` with still water or a harmonic tide, nodal-corrected when `run.start` is given;
  - output: `.dgsnap`, VTK, NetCDF (with the feature) and `stations.csv`.
- **Strictness:**
  - A mesh with open faces and no `[[boundary]]` for their tag is refused, as is a `[[boundary]]` whose tag the mesh doesn't have, and a station outside the mesh. All three are refused before the run.
  - Unknown keys fail at parse time. Without overrides, the error carries its line.
  - A Gmsh physical group the case layer can't name (a `Custom` tag) is an error, not a wall.
- **Boundary velocity, found while testing:**
  - A level-only external state takes the velocity outside as zero (`ElevationOnly::AtRest`, the library's default). A tide given as A at the boundary then enters as an incoming wave of A/2, so the boundary's level equals A only at an antinode of a standing tide.
  - That is documented in `src/boundary/characteristic.rs`, but a case author would not expect it, so `[[boundary]] velocity = "at_rest" | "incoming_wave"` makes the choice explicit. The example states it in a comment.
- **Gates (`tests/case_test.rs`, 2.8 s in the CI profile):**
  - The shipped example runs, 2 h of it, and its resolved case reads back equal.
  - A frictionless channel 60 km long and 10 m deep, closed at its head, with M2 of 0.1 m at its mouth. Linear theory gives an amplitude of 2 a_in cos k(L − x) along the channel. The model gives:

    | Boundary velocity | Mouth, model (m) | Mouth, theory (m) | Head, model (m) | Head, theory (m) | Phase difference |
    |---|---|---|---|---|---|
    | `at_rest` | 0.0690 | 0.0690 | 0.1000 | 0.1000 | 0.00° |
    | `incoming_wave` | 0.1381 | 0.1381 | 0.2000 | 0.2000 | 0.01° |

  - Plus the refusal cases above and ten unit tests of the schema: defaults, unknown keys and sections, exactly-one choices, overrides, JSON, the resolved round trip, the ramp.
- **The example:** `cases/tidal_channel/case.toml` is a channel 40 km long and 4 km wide, its bed going from 30 m down to 2 m above mean sea level, with M2 1.0 m, S2 0.35 m, Coriolis at 63.5°N and Manning 0.025. 25 h runs in 11,503 steps and 7.4 s (release build, `parallel,simd`). The time step falls to 0.08 s on the drying flats.
- **Not yet in a case:** bed rasters and projections, atlas tides, nesting, the atmosphere, 3D, waves, particles, farms, restarts and `dgcom mesh`. These are the remaining P8.1 lines in `TODO.md`, then P8.2.

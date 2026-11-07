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

  The binary's name waits on the crate-vs-repo name (P6).
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

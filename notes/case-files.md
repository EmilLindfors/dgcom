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

## P8.1, second part: placed meshes, bed rasters, the production numerics (2026-11-07)

- **Placing a mesh:**
  - `mesh.geographic` (a bbox with nx × ny) meshes the box as a rectangle in the `LocalProjection` about its centre, with open sides, as `froya_real_data` does.
  - `mesh.origin` places a rectangle or a Gmsh mesh: its x = y = 0 is that point.
  - Only a placed mesh takes a raster, `coriolis = "beta_plane"` (f₀ and β of the centre's latitude, at y = 0) and stations by lon/lat. Validation refuses these without a placement.
- **Bed rasters** (`[bathymetry.raster]`):
  - `kind = "bathymetry"` is `BedRaster::from_geotiff`: dry and no-data pixels are land, with a GSHHS `coastline` mask at 4 × 4 cells per pixel if one is given.
  - `kind = "topobathy"` is `BedRaster::elevation_model`, then `fill_holes` with `fill_zero_holes`, then `clamp_land(land_elevation_m)`.
  - The crop window is the geographic box, or the mesh's lon/lat extent plus 2 %. A raster that misses the mesh is an error, not a panic.
  - The raster reaches the nodes `projected` (the default, at the raster's resolution) or by `point`. Isolated wet nodes are then raised, and the bed is smoothed if `smooth_rx0` is set.
  - `mesh.drop_land` (default true) then drops the elements with no node below mean sea level. `cases/tidal_channel` sets it false, since its flats above mean sea level flood.
- **Numerics:** `froya_real_data` ran SSP-RK(4,3) at 0.9 × the linear CFL limit, while the first part had SSP-RK3 at CFL 1. The case defaults are now the same as the example's:
  - `integrator = "ssp_rk43"`;
  - `cfl` from `linear_cfl_swe_2d × LINEAR_CFL_SAFETY`, e.g. 1.359 at order 2. `case.resolved.toml` writes it out;
  - `local_time_stepping_levels` (`Multirate`).

  The standing-wave gate gives the same four digits under the new defaults.
- **Stage 0 of P8.2: bit for bit against `froya_real_data`.**
  - `cases/froya/case.toml` states the example's defaults: `FROYA_BBOX`, 120 × 90, P2, the 50 m topobathy model with its zero holes filled and land capped at 5 m, f = 1.31e-4, Manning 0.025, M2 0.8 m at rest-velocity with a 1 h ramp, and 10-minute outputs, which equal the example's callbacks. Callbacks shorten the step to land on them, so the outputs must match for the steps to.
  - Against `froya_real_data tides=none hours=1 snapshot_minutes=10 station_minutes=10 rest_hours=0`, `snapshot_compare_3d` finds **0 values differing** (η, u, v at every node) in all 7 frames: 1,740 steps each, dt 0.798–2.073 s.
  - On the coastline-fitted `froya_coast.msh` (`mesh=`, with the case's `gmsh` + `origin` at the box centre), 0 values differ over 0.5 h: 12,101 wet elements, 5,889 steps.
  - The domain logs agree too: 5,183 zero pixels filled, a 1,180 × 891 raster at 50 m with 85.5 % water, 33 isolated wet nodes raised, 9,912 of 10,800 elements kept.
  - The atlas tides and nesting, the example's real modes, are the next stage. Until they are in a case, no option can leave the example.
- **Gates (`tests/case_test.rs`):** a synthetic topobathy GeoTIFF of a bay is written in the test, with a sloping bed, land beyond 8.133°E and an island. It is meshed geographically, with the beta plane, Manning friction, open sides and a lon/lat station.
  - A lake at rest over it stays at rest to |u| ≤ 6.8e-14 m/s and |η| ≤ 1.4e-14 m over 1 h, and to 6.6e-14 and 2.1e-14 with local time stepping and SSP-RK3.
  - A raster that misses the mesh, and a station on dropped land, are refused. Three unit tests cover what needs a placed mesh.

## P8.1, third part: tidal atlases (2026-11-07)

- **`[boundary.atlas]`:** a `TidalAtlas` file (the text format of `norkyst_boundary_tides`) projected onto the tag's boundary nodes (`TidalAtlas::boundary_tides`), nodal-corrected from `run.start`, ramped over `ramp_hours`.
  - It needs a placed mesh and `run.start`, since atlas lags are Greenwich lags. Validation refuses it without them.
  - An atlas with currents gives the external velocity. One without them takes `boundary.velocity`.
- **Coverage:** before the boundary is built, `BoundaryCoverage::of_atlas` walls the open faces with a node farther than `coverage_m` (default 5 km, `froya_real_data`'s `ATLAS_COVERAGE`) from an atlas point. More than `max_closed` (default 0.05, the example's `max_closed`) of the tag's faces is an error, as in the example. The domain's notes record how many faces were closed.
- **Options:** `correction = "theory"` (`TidalAtlas::infer_from_theory`: K2 and P1 at their equilibrium ratios, the minor semidiurnals by admittance) and `transport_scaling` (`BoundaryTides::with_transport_scaling`, the example's `tide_transport`). The gauge correction, the example's default `gauge_gains`/`gauge_ratios` from a gauge record and a station atlas, is not in yet.
- **Stage 1 of P8.2, bit for bit:** `cases/froya` with the atlas against `froya_real_data gauge_gains= gauge_ratios= hours=1 snapshot_minutes=10 station_minutes=10 rest_hours=0` gives **0 values differing** in all 7 frames: 12 constituents at 1,139 boundary nodes, 0 of 381 faces closed, 1,740 steps each.
- **`cases/bodo`:**
  - `bodo_coast.msh` with `origin` at the centre of 13.7–14.55°E, 67.05–67.5°N, the 50 m topobathy and the NorKyst atlas.
  - The domain: 11,118 of 11,344 elements kept, 905 zero pixels filled, 26 isolated wet nodes raised, 0 of 186 open faces closed. 1 h runs in 15,059 steps, 97 s.
  - It takes the beta plane at Bodø's latitude. `froya_real_data` applies Frøya's f = 1.31e-4 there, 2.6 % below Bodø's 1.345e-4, so this case is not compared bit for bit with the example.
- **Gates (`tests/case_test.rs`):**
  - An atlas holding one tide (M2, S2, K1) at every point forces the same tide as `tide` with those constants: the largest difference at the stations over 6 h is 7e-15 m on a tide of up to 0.39 m. This checks that the two routes share the nodal corrections and the Greenwich-lag convention.
  - An atlas covering the lower quarter of a mouth walls the rest, leaving 2 open faces, when `max_closed` allows it, and is refused at the default.
  - A unit test checks that an atlas needs a placed mesh and a start.

## P8.1, fourth part: the gauge correction (2026-11-07)

- **Into the library:** `froya_real_data`'s default atlas correction moved to the library, so the example and case files share one path.
  - `HarmonicAnalysis::fit_gauge_record` is the whole-record fit of `HarmonicAnalysis::GAUGE_RECORD` (M2 … M6 plus Mf, Mm and Ssa), Rayleigh 1, with P1, K2, N2 and Q1 at equilibrium where the record cannot separate them.
  - `TidalAtlas::correct_at_gauge(fit, source point, gains, ratios)` scales `gains` by the gauge-over-source complex ratio, then re-infers `ratios` with the gauge's ratios. It returns `GaugeAdjustment`s, whose `Display` is the example's log line.
  - The example's `corrected_atlas` now calls it. Its release binary from before the change and after it give 0 differing values over 7 frames (1 h, default options), with the same seven adjustments logged.
- **`[boundary.atlas.gauge]`:** `record` (a gauge file with its position), `source` (a station atlas from the atlas's model), `max_offset_m` (3 km, the example's `STATION_MAX_OFFSET`), `gains` (K1, O1, S2) and `ratios` (N2, Q1, P1, K2).
  - A source without a point near the gauge is an error. The example instead skips the gains with a message.
  - It is exclusive with `correction = "theory"`.
- **Stage 1b of P8.2, bit for bit:** `cases/froya` with the Mausund correction against `froya_real_data hours=1 snapshot_minutes=10 station_minutes=10 rest_hours=0` (its default tide) gives 0 values differing in all 7 frames: K1 × 0.875, −8.6°; O1 × 1.052, +17.2°; S2 × 0.939, +0.1°; N2 = 0.208 × M2; Q1 = 0.434 × O1; P1 = 0.271 × K1; K2 = 0.283 × S2.
- **A finding at Bodø:** `cases/bodo`'s correction scales NorKyst's O1 by 0.589 (+18.4°) and lags K1 by +16.5°, which `froya_real_data`'s Bodø runs share.
  - NorKyst's median O1 ratio over the 17 gauges is 1.00 (`notes/norkyst-issues.md`), so this is suspect.
  - The station-atlas point is 25.5 m deep with zero current. TODO P1.5.
- **Gates:**
  - A unit test of `correct_at_gauge`: the gauge's M2 at the source point, the same gain elsewhere, K2 at the gauge's ratio, and the errors.
  - A case test with a synthetic year-long gauge record (M2 1.2×, +10°; S2 0.9×, −5° over the model) and a synthetic source atlas. The case logs M2 × 1.199, +10.0° and S2 × 0.900, −5.0°; the 0.1 % on M2 is the fit's nodal treatment over the year. A source with no point near the gauge is refused.
  - A 30-day synthetic record without K2 failed this at first. The fit infers K2 at equilibrium from S2 where 30 days cannot separate them, and a record lacking K2 then biases S2 (0.944, +15°). Real records have K2; the test uses a year.

## P8.1, fifth part: nesting (2026-11-07)

- **`[boundary.nesting]`:** `froya_real_data`'s 2D nesting (`norkyst=`) in a case file, on the library's `OceanModelState`. The domain build does what the example's `Domain::nesting` and `correct_parent_tides` do, in the same order:
  1. Read the parent (`OceanModelReader::from_file`; the `netcdf` feature. Without it the case is refused with that reason).
  2. The datum: `level_m`, else minus `TidalAtlas::mean_level` of `atlas` (the raw atlas, as the example), else `ParentDatum::RecordMean`, which refuses records shorter than 29 days.
  3. Wall the open faces the parent has no water near (`BoundaryCoverage::of_parent`, `max_closed`).
  4. Sample the parent at the boundary and the band, then blend the bed (`blend_bathymetry` with `blend_shore_depth_m`, the example's `-WALL_LOW_WATER` = 2 m).
  5. Correct the parent's tide by (corrected − raw) atlas, from theory or at a gauge (the `[boundary.atlas.gauge]` code), within `atlas_coverage_m` + the band.
  6. Check the parent covers the run.
  - The run adds the band (`OceanModelState::relaxation`) as a source after the Coriolis force, as the example does, and the boundary is `CharacteristicOBC` on the parent (`velocity` applies only to a parent without currents).
- **Stage 2 of P8.2, bit for bit:** `cases/froya/nested.toml` against `froya_real_data norkyst=data/froya_norkyst.nc hours=1 snapshot_minutes=10 station_minutes=10 rest_hours=0`, 0 values differing in all 7 frames, for `nest_tides=corrected` (the gauge section), `raw` (without it) and `theory` (`correction = "theory"`). Both logs agree on the datum (+0.282 m), 1139 boundary nodes (16 snapped), 19084 band nodes, 16804 blended nodes, the seven gauge adjustments and the correction's largest term (N2 0.147 m; L2 0.019 m for theory, whose atlas has K2 and P1 at the equilibrium ratios: the case notes it as the example does).
- **Not yet the scored runs:** the 2026-06-12 runs (`output/p34_2026-06-12/run.sh`) also used `slopes3d=on`, whose bed treatment (3D rx0 smoothing, cliff shores raised to land, tidal ponds filled) case files do not have. That, and `ib=` with the atmosphere, are what is left of the 2D path (TODO P8.1). (The bed treatment followed the same day, in the sixth part.)
- **`cases/bodo/nested.toml`:** builds on `data/bodo_norkyst_3d.nc` (datum +0.288 m, no walls, 579 boundary nodes, 17433 band nodes). It takes the beta plane, so it is not the example's run.
- **Gates** (`tests/case_test.rs`, with the `netcdf` feature; a synthetic parent written with the `netcdf` crate):
  - `a_nested_channel_fills_to_its_parents_level_less_its_mean`: a parent at ζ = 0.3 m with `level_m = -0.1`, and again with an atlas of `Z0` = 0.1: the channel's head settles at 0.2000 m, its volume within 5 % of 0.2 m over the channel. Without either, the 8 h record is refused for the datum.
  - `a_parent_that_misses_the_boundary_or_the_run_is_refused`: a parent cut 5 km east of the mouth (`max_closed`), and a 2 h parent for a 3 h run.
  - `the_bed_is_blended_to_the_parents_across_the_band`: the parent's 20 m on the boundary, the case's 10 m beyond the 3 km band, in between at 1.5 km, and 10 m everywhere with `blend_bed = false`.
  - Unit (no NetCDF): `nesting_needs_a_placed_mesh_a_start_and_an_atlas_to_correct`, the defaults and the value checks.

## P8.1, sixth part: the 3D bed and wall land (2026-11-07)

- **`[bathymetry.slopes_3d]`** is `froya_real_data slopes3d=on`, on the library's `Bathymetry2D` methods, in the example's order after `smooth_rx0`:
  1. `smooth_element_slopes` with `ElementSlopeBound { r_max, free_depth }` (0.15 and 19 m, `ElementSlopeBound::for_pycnocline(19.0)`) between columns at least `thin_depth_m` deep (0.1 m, `Hydrostatic3D::DEFAULT_MIN_COLUMN_DEPTH`; the scored runs' `debug_3d=thin=1` is `thin_depth_m = 1.0`).
  2. With `raise_cliffs` (default on, `cliffs3d=`): `raise_cliff_shores` at `cliff_depth_m` (20 m) to `high_water_m` (+2 m), then the isolated wet nodes raised.
  3. With `fill_ponds` (default on, `ponds3d=`): `fill_tidal_ponds` from `pond_low_water_m` (−1.43 m) to `high_water_m`, at most one element's nodes, then the isolated wet nodes raised.
  - The defaults are the example's constants, which are Mausund's lowest and highest astronomical tides and the summer pycnocline of the 3D runs: a case for another coast sets its own.
- **`lower_wall_land`** is `wall_land=lower` (`Bathymetry2D::lower_wall_land` at `wall_low_water_m`, −2 m). It comes before a raster's isolated wet nodes are raised, as in the example.
- **The scored 2026-06-12 runs, bit for bit (stage 2 of P8.2 finished).** `cases/bodo/scored_2026-06-12.toml` and `cases/froya/scored_2026-06-12.toml` are `output/p34_2026-06-12/run.sh bodo_2d_theory` and `froya_2d_theory`: the coastline meshes, the 2026-06-12 NorKyst days, the `_nkratios` atlases with `correction = "theory"`, `ramp_hours = 3`, eight LTS levels, Frøya's f-plane at both, and the 3D bed with `thin_depth_m = 1`.
  - **24 h against the scored runs' own output** (`output/p34_2026-06-12/{bodo,froya}_2d_theory/froya.dgsnap`, written by the build the scores were made with), `snapshot_compare_3d`: 0 values differ in all 25 hourly frames at both sites, in the same 2499 (Bodø) and 2421 (Mausund) steps. The cases' `stations.csv` equal the scored station files at all 145 ten-minute times (to their printed 1e-6 m) and score, centred RMSE against the gauges' observations over hours 3–24, **2.96 cm at Bodø and 4.25 cm at Mausund** (the logs' 3.0 and 4.2; computed by hand, since case files have no gauge scoring yet).
  - Against the current example, 1 h with 10-minute output: 0 values differ in all 7 frames at both sites, and the setup lines match (Bodø: 2595 smoothing sweeps, 35405 nodes changed; 1890 cliff shore nodes raised; 136 ponds of 212 nodes filled; datum +0.287 m; Mausund: 31063 nodes, 1917 cliff nodes, 300 ponds, +0.311 m).
- **Stations in open water (a fix).** A case station sampled the element at its position. A tide gauge stands at the shore, in an element that is partly land in the mesh, and there `η = h + B` mixes the land nodes' bed into the surface through the basis (a P2 weight can be negative): Bodø's gauge read −0.313 m at t = 0 with the sea at rest, where the example's read 0. The example moved its stations to the nearest element whose nodes are all at least 3 m deep (`STATION_MIN_DEPTH`, within `STATION_MAX_OFFSET` = 3 km). That search is now the library's `Probe2D::in_open_water`, which the example calls, and case stations use it by default (`open_water_depth_m = 3`, `max_offset_m = 3000`; `open_water_depth_m = 0` samples where the station is). Stations are placed when the domain is built, so `dgcom check` reports a moved one (Bodø's: sampled 9 m from its position, 6.3 m deep there, as the example's log says). Bodø's 1 h station series then equals the example's to its printed 1e-6 m at every output. The earlier case files' station series (`cases/froya/case.toml`'s Mausund, `cases/bodo/case.toml`'s Bodø) were wrong in the same way before this; their snapshots were not affected.
- **Station and field intervals.** `Simulation` lands its steps on the callback times, so the output schedule is part of the run. The example records its stations every `station_minutes` (10 by default) and its snapshots every `snapshot_minutes`, and steps to both. A first 24 h case run with hourly output only matched the scored run's hourly frames to round-off (η RMS ≤ 0.011 mm, max 1.8 mm; one ū node 66 mm/s), not bit for bit, while the example from the current build matched the scored file exactly at 1 h. `[output] station_interval_minutes` (default `interval_minutes`; one must be a whole multiple of the other) gives case files the same two schedules.
- **Gates** (`tests/case_test.rs`, no NetCDF needed):
  - `the_3d_bed_is_the_librarys_in_the_examples_order`: a raster fjord (a 25 m shelf, a 60 m basin, a shore strip at +0.5 m and a pond in +3 m land) whose bed with `[bathymetry.slopes_3d]` equals, value for value, the library's three steps and the isolated-node raises applied in the example's order to the bed without it; each step acts on it (125 nodes smoothed, 25 cliff shore nodes raised, one pond of 6 nodes filled), and with `raise_cliffs` and `fill_ponds` off the bed is the smoothing's alone.
  - `wall_land_is_lowered_to_the_water_beside_it`: a channel whose bed rises to +1 m at its north wall has that land lowered, equal to `lower_wall_land` on the untreated bed, and no node left above mean sea level.
  - `stations_can_be_recorded_more_often_than_the_fields`: the shipped channel with stations every 10 minutes and fields every 30 writes 13 station records and 5 snapshot frames, and its station series and step count equal those of the run that writes everything every 10 minutes; intervals that are not multiples are refused.
  - `a_gauge_on_land_beside_the_water_is_moved_into_open_water`: a station on the bay's dropped land 2 km from open water is moved into it and reads 0 at rest to 1e-9 m; with `open_water_depth_m = 0` it is refused as on dropped land.
  - Unit: `the_3d_bed_takes_the_examples_defaults_and_checks_its_values`, `stations_sample_in_open_water_by_default`, and `Probe2D`'s `a_station_on_the_shore_is_moved_into_open_water` (a shore element reads its land's bed at rest; the moved probe reads 0, 250 m away; none beyond `max_offset`).
- **Gauge scores are not in case files yet.** The example scores its stations against `gauges=` (centred RMSE over hours 3–24, harmonic constants). The case writes `stations.csv`; the scores above were computed from it by hand. A `gauge =` key on `[[output.station]]` is TODO P8.1.

## P8.1, seventh part: the weather (2026-11-07)

- **`[atmosphere]`** is `froya_real_data met=`: `AtmosphereReader::from_files` on `files` (joined in time), `GriddedAtmosphere2D` on the final mesh (after the land is dropped and the bed blended), ramped up over `ramp_hours` (1 h, the example's `TIDAL_RAMP_HOURS`), checked to cover the run. `wind = false` and `pressure = false` switch either part off (`without_wind`, `without_pressure`). It needs a placed mesh, `run.start` and the `netcdf` feature (checked last, so a build without NetCDF still checks the rest of the case). The source comes after the Coriolis force and the relaxation bands, as in the example.
- **The inverse barometer at the open boundaries.** With the pressure on, each `[[boundary]]` adds `inverse_barometer` times the weather's inverse-barometer level (`InverseBarometer` on its provider): 1 by default for `still`, `tide` and `atlas`, which lack the response, as the example's tide does; 0 for `nesting`, whose parent carries its own (the example's `ib=`, default 0). A fraction is 0–2 and needs `[atmosphere]` with its pressure.
- **Bit for bit, 1 h with 10-minute output (stage 2 of P8.2):** 0 values differ in all 7 frames for
  - `cases/froya/weather.toml` against `froya_real_data norkyst=data/froya_norkyst.nc met=data/froya_met.nc`;
  - the same with `--set boundary.0.inverse_barometer=0.5` against `ib=0.5` (which moves η by 12 mm RMS at 1 h from the run without it, so the check is not empty);
  - `cases/froya/case.toml` with `--set 'atmosphere.files=["../../data/froya_met.nc"]'` against `met=data/froya_met.nc`: the atlas tide with the full inverse barometer.
- **`--set` into arrays of tables:** `boundary.0.ramp_hours` picks the first `[[boundary]]`; a missing index or one past the end is an error that says so.
- **Gates** (`tests/case_test.rs`, with the `netcdf` feature; a synthetic weather file of uniform `u10` and `msl`):
  - `a_low_lowers_the_channel_to_its_inverse_barometer_level`: under +10 hPa the closed channel behind still water settles at −0.09945 m in 12 h, −Δp/(ρg) to 0.5 %; half of it with `inverse_barometer = 0.5`; with 0 the uniform pressure moves nothing (2e-14 m).
  - `a_wind_sets_the_channel_up_against_its_head`: 10 m/s along the channel sets the head 13.15 mm above the mouth 9 km away after 24 h, against τL/(ρgh) = 13.16 mm (to 1 %).
  - Unit: `the_atmosphere_needs_a_placed_mesh_and_a_start` (placement, start, ramp, wind or pressure, `inverse_barometer` needing the pressure and lying in 0–2, the defaults by boundary kind), and the indexed overrides in `overrides_set_values_and_create_tables`.
- **The 2D real-data path is in case files.** What `froya_real_data` still has that case files do not: the 3D model (P8.2's next stage), the waves and the farm, the gauge scoring (P8.1), restarts (P8.1), and the analytic `wind` option (a uniform wind and pressure gradient, a test of the example's own).

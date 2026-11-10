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

## P8.1, eighth part: gauge scores (2026-11-08)

- **`[[output.station]] gauge`** names a tide gauge's record (`read_tide_gauge_file`, the format of `scripts/kartverket_gauge.sh`); it needs `run.start`. The record is read and fitted when the domain is built (`dgcom check` reports its length and fit). After the run the station's surface from `[output] spinup_hours` on is put on the gauge's Unix axis and compared (`GaugeComparison::compute`): the skill against the observations and against the gauge's whole-record tidal prediction, and the model's reference constants against the gauge's. They go into `RunReport.stations[i].gauge`, `gauge_skill.csv` (one row per station and reference), `gauge_constituents.csv` (one row per constituent: amplitudes, lags, ratio, lag difference, |ΔZ|, inferred or not) and `dgcom run`'s log.
- **`spinup_hours`** defaults to the longest ramp of the open boundaries and the atmosphere: the scores start when the forcing is fully on (3 h in the scored cases, as their `spinup_hours=3`; 2 h in the shipped channel).
- **Into the library, out of the example** (one path): `froya_real_data`'s `Gauge`, `fit_record` and `report_station`'s arithmetic are now `analysis::GaugeRecord`, `HarmonicAnalysis::fit_with_reference` (with `HarmonicAnalysis::STATION_RECORD`, the example's `STATION_CONSTITUENTS`) and `analysis::GaugeComparison` / `SkillScore`. The example prints from them; a 6 h Mausund run with `spinup_hours=1` prints the same station report before and after (the constants table, the |ΔZ| sums, both skill lines). One change of behaviour: a window too short for any constituent now still gets its skill lines.
- **The scored cases score themselves (24 h, `gauge_skill.csv`):** `cases/bodo/scored_2026-06-12.toml` 2.96 cm centred against Bodø's observations (bias +5.9 cm, correlation 0.9990, 22 samples), 6.6 cm against the gauge's prediction (127 samples), |ΔZ| summed 10.3 cm; `cases/froya/scored_2026-06-12.toml` 4.25 cm at Mausund (bias +7.3 cm, correlation 0.9984), 6.7 cm against the prediction, |ΔZ| 7.9 cm. Every figure is the scored runs' log's (`output/p34_2026-06-12/*_2d_theory.log`), down to the N2 row (0.1962 m at 312.6° inferred against the gauge's 0.2017 m at 308.7°). Bodø took 8943 s of wall time this time against 1332 s on 2026-11-07, in the same 2499 steps: the machine was shared.
- **Gates:**
  - `a_station_is_scored_against_its_gauge` (`tests/case_test.rs`): a gauge that read the shipped channel's mouth 10 cm high through a 14 h run (and a pure M2 for a month on either side) scores a bias of −0.1000 m, a centred RMSE of 0 and a correlation of 1 over the 13 hourly samples from the 2 h ramp's end, the run itself unchanged; both CSVs are written, M2 among the constituents.
  - `a_comparison_finds_the_models_constants_against_the_gauges` (`analysis::tide_gauge`): a 60-day gauge of M2 and K1 (P1 at equilibrium) with a gap, and a 3-day model with 1.2 × its M2 and 2.5 × its K1: the ratios to 1e-6 and no phase error; the prediction scored at every model sample, the observations at the 72 hourly samples in the window.
  - Unit: `gauge_scores_start_when_the_forcing_is_fully_on` (a gauge needs `run.start`; the spin-up's default and its check).

## P8.2, stage 3: the 3D tide (2026-11-08)

- **First, the example's 3D start into the library** (one path): `vertical::MeanProfile` (the stable mean T, S profile; pool-adjacent-violators on density), `boundary::sample_columns` (a parent's columns at every wet node at the start, marking what it covers), `Solution3D::fill_tracers` (the uncovered columns from a profile) and `Solution3D::keep_shear` (the parent's shear about a depth mean at rest). `tidal_run_3d` calls them and keeps its `debug_3d` switches around them. A 1 h 3D Bodø run (`levels=20 tide3d=1`, 2026-06-12) is unchanged: 0 values differ in 7 frames, and the logs differ in the output path only.
- **`[three_d]`** (`src/case/run/three_d.rs`, a child of `run.rs` that reuses its open boundaries, `swe_builder` and writers) follows `tidal_run_3d`'s order:
  1. The open boundaries and the 2D module without bed friction (`swe_builder`; the 2D run adds the friction in `physics`).
  2. The start at rest: the parent's columns (`OceanModelColumns` on the domain's `parent_reader`, read `from_file_with_profiles` when the case is 3D), sampled twice (the second time with the water below the parent's bed relaxing to the first sample's `MeanProfile`, `DEEP_DECAY` 10 m), the uncovered columns from that mean or from `three_d.profile`, the shear kept in columns at least `thin_depth_m` deep.
  3. The band (`Nesting3D`, the nesting's `band_km`/`band_minutes`, else `[three_d]`'s) relaxing to the parent's columns, or to the start (`ReferenceColumns`); none without open faces (`Nesting3D::new` panics on an empty tag list, so the case skips it).
  4. `Hydrostatic3D`: Song–Haidvogel σ, `LinearEOS`, GLS, log-layer drag, the thin depth, the shear's Smagorinsky viscosity (0.1 unless `[physics]` has one), the Kuzmin tracer limiter about the mean profile, the band, the weather's wind and fluxes, implicit vertical advection, the density and the pressure gradient's reference profile. `ModeSplitIntegrator` with `[three_d]`'s schemes, CFLs and levels; `Simulation3D` landing on the field outputs; the stations at the first step on or after their times, from the depth mean (at their times since 2026-11-08, below).
  - The domain splits `[atmosphere]` for 3D (`split_for_3d`, before anything clones it): the pressure stays the 2D module's and the boundaries' inverse barometer, the wind goes to the columns, and with `heat` and a heat budget COARE 3.5 fluxes (`Coare35::at_latitude` of the weather grid's mid-latitude). The parent and the weather must cover `Case::forcing_end_s`: a third of `dt_max_s` past the end.
  - Refused with `[three_d]`: `[physics] friction`, a non-default `[numerics]`, `output.vtk`/`output.netcdf` (TODO P8.1), nesting only some of the open boundaries, more than one parent, and no parent without `profile`.
- **Bit for bit:**
  - `cases/bodo/scored_3d_2026-06-12.toml` against `froya_real_data levels=20 tide3d=1` with the scored `bodo_3d` command (`nest_tides=raw`, `slopes3d=on`, `ramp_hours=3`), 1 h with 10-minute snapshots: 0 values differ in all 7 frames in η, ū, the surface layer's u and T.
  - The same case against the scored run's own output (`output/p34_2026-06-12/bodo_3d_raw/froya.dgsnap`), 2 h with hourly snapshots: 0 values differ at 0, 1 and 2 h. That build (`6f9e558`) is not in main's history, and the 3D numerics agree with main's.
- The weather in 3D: `cases/froya/weather.toml` with the parent `data/froya_norkyst_3d.nc` (profiles), the weather `data/froya_met_heat.nc` (MET Nordic with its heat budget), no bed friction and `[three_d] levels = 20`, against `froya_real_data norkyst=data/froya_norkyst_3d.nc met=data/froya_met_heat.nc levels=20 tide3d=1` (2025-06-15, the gauge-corrected parent tide, the 120 × 90 grid), 1 h with 10-minute snapshots: 0 values differ in all 7 frames. That checks the split weather (the wind on the columns, the pressure on the depth mean and the boundary's level) and the COARE 3.5 fluxes into Jerlov III water.
- **Gates** (`tests/case_test.rs`, no NetCDF needed):
  - `a_stratified_basin_at_rest_stays_at_rest_in_3d`: a closed 20 m basin with a 4 °C pycnocline (`three_d.profile`) stays at rest to round-off for 2 h (|ū| 1.8e-14 m/s, |η| 3.6e-14 m), writing 3D snapshots at 0, 30, …, 120 min and the station. Over a slope (30 → 10 m) with its mouth open to still water, the depth mean drifts at 2e-7 m/s, as much closed: the drift comes with the mixing (the flat basin mixes and stays at rest), the stratification diffusing differently in columns of different depth (a real density gradient, Phillips 1970 and Wunsch 1970) and departing from the pressure gradient's reference. Bounded at 1 μm/s.
  - Unit: `a_3d_run_takes_the_examples_defaults_and_refuses_what_it_replaces` (the defaults, the CFL by scheme, the profile's interpolation, the forcing reach, the slopes' thin depth, and the refusals); `vertical::mean_profile` (a stable linear profile is its own mean where the columns' layer centres reach; an inversion is pooled).

## P8.2, stage 4: the coupled waves (2026-11-08)

- **`[waves]`** is `froya_real_data wave_coupling=MINUTES` (`src/case/run/waves.rs`, the example's `wave_model` and `coupled_waves`): `WaveModel2D` with SWAN's source physics, its integration as WAM's (default: implicit, Hersbach & Janssen's rate limiter, 4 propagation steps per source step, CFL 1) or SWAN's (frozen, the per-step limiter, 1, 0.5), the depth limit, absorbing land, implicit refraction and frequency shifting (the case's default is on, as the example's docs advise; the example's own default is off), coupled by `CoupledWaves2D` (`run_with_exchange`, the exchange a whole multiple of the outputs' interval) with its force (`dissipation` or `stress`) and bed stress ramped over `ramp_hours`.
  - **The waves' mesh:** the run's, or `[waves.mesh]` (the `[mesh]` schema, checked by the shared `MeshSection::validate_shape`), built by the same pipeline as the run's (`mesh_and_bed`, factored out of `build_domain`: the bed, its treatments, the land dropped) in the run's projection. No nesting blend: the example's wave domain has none either.
  - **The sea:** `[waves.sea]` JONSWAP (γ 3.3, cos⁴ spreading) on the boundary and everywhere at the start, or `[waves.spectra]` (`WaveSpectraFile`, `BoundarySpectra` from the `neighbours` nearest points, their mean everywhere at the start; the model's own boundary spectrum is then the parent's at every node, so no JONSWAP is set).
  - **Wind:** `[atmosphere]` resampled on the wave nodes (`GriddedAtmosphere2D::on_mesh`), else the parent spectra's (`WindSeries`), else none. The example's uniform `wind` option has no case key.
  - **Output:** the snapshots carry `SeaStateFields::NAMES` (H_s, T_p, direction, …) at every circulation node, linear in time between the exchanges (`waves::SeaStateFrames`, moved from the example, which now uses it); the progress line the largest H_s; the run's notes the exchanges, wave steps and their cost. Not at stations yet (TODO P8.1).
  - Refused: `[waves]` with `[three_d]` (the example couples the 2D tide only), both or neither of `sea` and `spectra`, `[waves.mesh] origin`, spectra without placement or a start.
- **Bit for bit:** `cases/froya/storm_2025-10-09.toml` against `froya_real_data mesh=data/froya_coast.msh lts=8 start=2025-10-09T00:00:00Z met=data/froya_met_20251008.nc wave_coupling=10 wave_mesh=60,45 implicit=1 wave_spectra=data/froya_wave_spectra.nc` (the storm of `notes/waves.md`), 1 h with 10-minute snapshots: 0 values differ in all 7 frames, the 7 sea-state fields included (`snapshot_compare_3d` now counts the surface fields; the two files are the same size to 123 bytes of metadata, so both carry them). The waves' mesh is the notes' 10,040 nodes; 6 exchanges, 66 wave steps of 54.5 s.
- **Gates:**
  - `a_swell_through_a_deep_channel_keeps_its_height` (`tests/case_test.rs`, no NetCDF): a JONSWAP swell (1 m, 8 s, from the west) in a 30 m channel coupled every 10 minutes keeps the largest H_s at 1.000–1.015 m over an hour, the run near rest (|ū| 2e-6 m/s), 6 exchanges, and the last snapshot's H_s field is the progress line's.
  - Unit: `waves_take_the_examples_defaults_and_their_checks` (WAM's and SWAN's integration defaults; the refusals).

## P8.2, stage 5: the farm's cages (2026-11-08)

- **`[farm]`** is the physics of `froya_real_data site=`: the site file (`read_farm_site_file`), its cages (`FarmSite::cage_layout` with a `CageGrid` of `cage_spacing_m`, `cage_radius_m`, `net_depth_m`, `solidity`; `RING_160M`'s by default), each checked inside the mesh, their drag per length scaled by `drag_share` (the example's `FarmRun::Open`, `Control` and `Cages` are 0, its `farm_control`, and 1). In 2D `CageDrag2D` on the depth mean (`physics()`; not in `swe_builder`, which the 3D run shares); in 3D on the layers the nets reach (`Hydrostatic3D::with_cage_drag`, after the tracer limiter, as the example), not on the 3D run's 2D module (the example's `builder_without_friction` has none).
- **Not moved:** the example's farm analysis (`examples/common/farm_site.rs`: the wake against the open run, the residual, the drag on the farm, the runs' divergence, the particles' connectivity). A case runs one of the runs; the analysis over runs and the particles are TODO F.1 and F.2.
- **The raster's cut (found on the way).** The first comparison differed at t = 0 in 25 values of η (33.5 mm at most): the example cuts the raster to its `bbox=` (444 × 380 pixels, 1404 zero pixels filled), the case cut it to the mesh's extent with a 2 % margin (461 × 395, 1540 filled), and `fill_holes` fills from the surroundings within the cut, so the beds differed at a few nodes (82 tidal ponds against 81). The Frøya cases matched because `froya_topobathy.tif` is the Frøya box, so any wider window is clipped to it; the Mausund box lies inside it. `[bathymetry.raster] crop` now states the cut; the dependence itself is TODO P1.6. The window was also the run's `[mesh.geographic]` box for every mesh: a geographic `[waves.mesh]` now takes its own box.
- **Bit for bit:** `cases/froya/kattholmen_farm.toml` (with `crop`) against `froya_real_data mesh=data/mausund_kattholmen.msh bbox=8.45,63.78,8.90,63.95 tides=data/mausund_boundary_tides.txt norkyst=data/froya_norkyst.nc slopes3d=on debug_3d=thin=1 lts=8 ramp_hours=3 cs=0.1 nu=0.1 site=data/sites/14042.txt farm_runs=cages`, 1 h with 10-minute snapshots: 0 values differ in all 7 frames (8 cages, C_d·a 0.0207 1/m). The same case with `--set farm.drag_share=0` departs from it by up to 5.5 mm/s in ū at 1 h (RMS 0.25 mm/s): the drag acts.
- **Gates:**
  - `a_farms_cages_slow_the_current_through_it` (`tests/case_test.rs`, no NetCDF): three cages of 25 m radius across a placed 10 km channel, 20 m deep, with an M2 tide through it: the mean speed through the farm 0.1515 m/s against 0.1612 without, and `drag_share = 0` the channel without a farm value for value.
  - Unit: `a_farm_takes_a_ring_frames_defaults_and_needs_a_placed_mesh`.

## Opportunities noticed during the migration (2026-11-08)

Each is a `TODO.md` line under its area; the context:

- **Provenance (P8.1).** The scored 2026-06-12 3D run was built from `6f9e558`, which is not in main's history. Its 3D numerics happened to equal main's (stage 3); nothing in its output said which build made it beyond the `BUILD_COMMIT` file kept by hand. A case's `case.resolved.toml` and snapshot metadata should carry the version and commit.
- **Viewer metadata (P8.1).** `froya_real_data` writes the stations and the projection into its snapshots (`snapshot_metadata`), which `dg-viz` draws the land and the stations from; case snapshots carry `source` and `case` only.
- **The output schedule is part of the run (P8.1).** `Simulation` lands its steps on the callback times, so a run's outputs change it by round-off (stage 2: a case writing hourly differed from the scored run, which recorded stations every 10 minutes, by 1.8 mm in η over the day). `station_interval_minutes` made the schedules equal; the general question is whether outputs should perturb a run at all.
- **3D stations (P8.1)** are recorded at the first step on or after their times, so their series are irregular by up to a step; the 2D run's land exactly.
- **The example's duplicated drivers (P8.2).** With every mode in case files, `froya_real_data`'s `tidal_run`, `tidal_run_3d`, `coupled_waves` and the farm's runs duplicate the case runner. Its diagnostics (`debug_3d=`, `profile=`, `waves=N`, the farm's report) are worth keeping as tools over case files; the rest can go with its options.
- **The cut-dependent bed (P1.6)**, above.
- **`Nesting3D::new` panics on no tags (P4.2)** although it returns a `Result`; the case guards it for closed basins.
- **The slope drift at rest (P4.2):** 2e-7 m/s in 2 h over a 30 → 10 m slope, open or closed, against 1e-14 flat (stage 3's gate). Running it with mixing off would split the physical diffusion-driven flow from the pressure gradient's error as the profile leaves its reference.
- **`GriddedAtmosphere2D` configuration panics once cloned (P6):** `split_for_3d` needs sole ownership, so a case's domain splits the weather before the waves, the boundary's level or anything else can hold it. Switches per handle would remove the build order.
- **The farm's analysis (F.1) and particles (F.2)** as case-level tools, and **waves with 3D (F.4)**, which case files refuse as the example does not couple them.

## P8.1: provenance and the viewer's metadata (2026-11-08)

- **Provenance.** `build.rs` sets `DGCOM_GIT_COMMIT` (12 hex digits) and `DGCOM_GIT_MODIFIED` (whether `git status` lists changes in `src`, `Cargo.toml`, `Cargo.lock` or `build.rs`); without git neither is set and the provenance is the version alone. `dgcom::provenance::describe()` writes the line, `# dgcom 0.1.0 (commit …[, modified])`, as the first line of `case.resolved.toml` (a comment, so the file still reads back as the case), as the snapshot's `build` key, as the NetCDF file's `source` attribute, and from `dgcom --version` and at the start of `dgcom run`. The script reruns when those sources, `HEAD`, the index, `packed-refs` or the branch's ref change, so a commit rebuilds the crate once (its commit changed) and an edit marks it modified. Only those sources count: a new untracked file under `src` marks the build modified too, while edits to examples, tests or docs do not.
- **The viewer's metadata.** Case snapshots now carry `title` (the run's name), `build`, a `station` line per station (`name,x,y` where the probe samples, so after a move into open water), the `projection` (`local,lat,lon`) of a placed mesh, and a `cage` line per circular cage (`x,y,radius,net depth,C_d·a`), the keys `froya_real_data` and `farm_3d` write and `dg-viz` and `scripts/dgsnap_check.py` read.
- **Gates** (`tests/case_test.rs`): the example case's resolved file starts with the build's line and its snapshot carries the build, the title and the three stations (no projection, unplaced); the farm channel's snapshot carries its three cages where the site file put them, the station and the projection. Unit: `provenance::tests`.

## P8.1: `dgcom mesh` (2026-11-08)

- **What it writes.** `CaseFile::write_mesh(&domain)` (`src/case/run/mesh.rs`) into `<output>/mesh/`: `<name>.msh` (`write_gmsh_mesh`, Gmsh 2.2 with the boundary tags as physical groups, so Gmsh shows which faces are open, tidal or wall) and `<name>.vtu` (`io::write_vtk_fields`, new): at the nodes `bed_m` and `depth_at_rest_m` (the case's initial level); per element `element_id`, `diameter_m`, `dt_s` (the 2D physics' `LocalTimeStepping::element_dt` at rest at the case's CFL and scheme; −1 where nothing limits it, as VTK readers take no infinity) and, with `local_time_stepping_levels`, `lts_level` (`MultirateStepper::assign_levels`, the run's own, with its level-jump limiting). A separate `[waves.mesh]` goes to `<name>_waves.*` with its bed and diameters. A 3D case gets the mesh, the bed and the diameters only: its columns step by the 3D CFL, not the 2D module's.
- **Its step is the run's.** `the_mesh_is_written_with_the_step_the_run_takes` (`tests/case_test.rs`): the tidal channel's reported step equals the run's first (and longest) step exactly, without local time stepping (15.739 s) and with 2 levels; the Gmsh file reads back with the same elements and tags. Frøya (`cases/froya/case.toml`): 2.073 s, the run's 2.073 s; with 8 levels, a 530.781 s coarse step, the run's.
- **A first look with it** (`cases/froya/storm_2025-10-09.toml`, 8 levels, 12,101 elements): a 39.1 s coarse step and levels 1–7. The finest level (0.31–0.61 s) is 80 partly dry coastal elements, 68–345 m across, with a node about 3 m above the still water and the deepest at 67 m (median); they take 3.6 % of the element steps (Σ n_ℓ·2^ℓ), so local time stepping already absorbs them.

## P8.1: 3D stations at their times (2026-11-08)

- **Before:** a 3D case recorded its stations at the first step on or after their times, as `froya_real_data` does, so the series were irregular by up to a step (the 2D run lands its steps on the station times).
- **Interpolate, not land.** Landing the 3D steps on the station times too would move the run whenever the station interval changes (the 2D run's round-off moves, P8.1 above), and the scored 3D Bodø case, whose stations are every 10 minutes and fields hourly, would no longer equal the scored run. So the steps still land on the field outputs only; every step samples the stations from the depth mean (`sample_depth_mean`: the same `h = max(η − B, 0)`, `hū`, `hv̄` at the element's nodes as the 2D state the station read before, so a sample at a step is the old value bit for bit), and a station time between two steps gets the linear interpolation of the samples around it.
- **Its error** is the interpolation's, Δt²/8·|η''|: `a_3d_runs_stations_are_recorded_at_their_times` (`tests/case_test.rs`) runs an M2 tide (0.5 m, 30 min ramp) into a 3D channel at `dt_max_s = 47` with the stations every 10 minutes, the fields hourly (interpolated) and every 10 minutes (landed). After the ramp the two head series differ by at most 2.2e-6 m in η (theory: 47²/8 · 0.48 m · ω_M2² ≈ 2.6e-6 m) and 9e-8 m/s in ū; at the hours, where both land, by 4e-8 m (their other steps differ); in the ramp's transients by up to 2e-4 m.
- **The scored Bodø 3D case** (`cases/bodo/scored_3d_2026-06-12.toml`, 2 h, the stations every 10 minutes): its snapshots equal the earlier case run's (stage 3's, which equals the scored run) value for value in 694 steps, so the run did not move. Bodø's station times are now 600, 1200, 1800 s, where they were 600.8, 1207.6, 1801.2 s; at 20 minutes the surface reads 13.84 mm, where the first step after it read 13.60 mm.

## P8.1: the waves at stations and at the parent's points (2026-11-08)

- **Stations.** With `[waves]` a station records H_s, T_p and the direction the waves come from (degrees clockwise from north; from the mesh's y axis on an unplaced mesh) at its times, from the sea-state fields the snapshots carry (`SeaStateFrames`: the waves' state at the exchanges, linear in time between them, on the run's mesh, the period and the direction weighted by m₀) at the station's probe (`sea_state_at`). `StationSeries` has `hs_m`, `tp_s`, `wave_from_deg` (empty without waves); `stations.csv` adds `hs_m,tp_s,wave_from_deg`. A 3D case has no waves (F.4).
- **The parent's points.** With `[waves.spectra]` whose file has H_s (`WaveSpectraFile::hs`), the domain keeps the parent's points and its H_s (`ParentWavePoints`, linear in time), and at every exchange (the waves' own time, `WavePoints::record`) our H_s at each point from the basis on the waves' mesh, with T_p and the direction at its nearest wave node, as `froya_real_data`'s `WavePoints::report`. `wave_points.csv` has a row per point and exchange (`point,lon,lat,time_s,hs_m,parent_hs_m,tp_s,wave_from_deg`), and the log the bias and RMSE over every point from the spin-up (`[output] spinup_hours`, by default the longest ramp).
- **Checks.** `cases/froya/storm_2025-10-09.toml` for 1 h: its snapshots equal the earlier case run's value for value (recording moves nothing), and at WAM's five points its H_s, the parent's, T_p and the direction equal the example's `wave_points.txt` (the same 1 h command) to its printed precision (4.9e-5 m, 4.3e-4 s, 0.04°). Mausund's station: H_s 2.77 m at the start (the boundary's mean spectrum everywhere) and 1.13 m after 10 minutes in the skerries' lee.
- **The swell gate was weaker than it read.** `a_swell_through_a_deep_channel_keeps_its_height` checked the largest H_s, which the open boundary holds at 1 m. A station mid-way along its 2 km wide channel read 0.68 m, and 0.66 m at the head: the closed side walls take the part of the spread sea that runs into them (as a coast does; `absorbing_land`, which is about dry land, changes nothing). 20 km wide, the same sea holds 1.000 → 0.998 m mid-way (0.979 m beside the head wall). The test now runs 10 km wide with a station mid-channel: H_s 0.980 m after the hour, T_p 7.91 s, from 270.00°.

## P8.1: restarts for case runs (2026-11-08)

- **Writing.** `[restart] interval_hours = H` writes a restart at the first output at or after every multiple of H model hours: in 2D at the callbacks (the field and station times), in 3D at the field outputs, where its steps land (a restart between landings would resume on other steps). The run's end is one too when due. The file is `<output>/<name>.restart`, each replacing the last in one rename, or with `keep = true` `<output>/<name>_<hours>h.restart` per time (`RestartSchedule` in `src/case/run/restart.rs`, as `froya_real_data`'s). Beside the state (`Restart2D`, `Restart3D`), the driver's data holds the resolved case with its build line (`text.case`, a new kind of section: `DriverData::text`; restart files keep their format version and older ones read), the build (`build`), each station's records (`station.<j>.<field>`, with `StationSeries::depth_m`, new, so that `stations.csv` can be written again from them) and the counters (`output.counters`: VTK frames, the volume at the start).
- **Resuming.** `restart.from`, or `dgcom run case.toml --resume <file>` (relative to the working directory). The domain's fingerprint is checked (`SWEPhysics2D::check_restart`, `Simulation3D::resume`), then the case: both resolved cases (the restart's read again by this build, so a key added since takes its default) are flattened to `key.path → value` (arrays of tables by index, `boundary.0.tag`) and compared. A resume may change `run.duration_hours`, `[output]` but its stations, `[numerics]`, the 3D time stepping (`three_d.scheme`, `cfl`, `dt_max_s`, `levels_lts`, `barotropic_*`) and `[restart]`; anything else must equal the restart's unless `restart.changes` names it or a table above it (`["farm"]` for a branch run). A refusal names each key with both values. Comparing resolved cases closes P5.3's two gaps for case runs: a default changed between two builds shows as a changed value, and an option left out is its default. A 3D run builds its start as the stopped run did (the band's and the pressure gradient's references are of it) and then takes the restart's state and mode-splitting history.
- **The outputs go on.** The stations' records continue and `stations.csv` is written again from them; the snapshot file keeps its frames up to the restart and is continued (`SnapshotWriter::append`); the VTK frames go on numbering; the NetCDF file is a new `<name>_from_<hours>h.nc`. The resumed run's first callback is the restart's time, which the stopped run wrote out: only the files that start with the resumed run take it (the NetCDF file, and a snapshot file in another output directory, which a branch run writes to). The volume at the start comes from the restart, so the report's is the run's.
- **Limits.** A 2D resume must be at a multiple of its callback interval (the callbacks fall at the restart's time plus multiples of it, the outputs at multiples of theirs); changing the intervals otherwise is refused. A case with `[waves]` cannot write or resume restarts (the spectra are not kept; TODO P8.1). VTK into another directory continues the numbering without the earlier frames.
- **Gates** (`tests/case_test.rs`). `a_resumed_case_run_ends_bit_for_bit_where_the_uninterrupted_one_does`: the example with local time stepping (2 levels), fields every 15 min, stations every 5, VTK, restarts every 0.5 h; stopped at 1.25 h and resumed from its 1 h restart in the same directory, the station records bit for bit, and `stations.csv`, the snapshot file and the 9 VTK frames byte for byte. Another viscosity is refused naming `physics.viscosity_m2_s = 1.0 (the restart's run: 0.0)`; the viscosity given at its default is no change; a branch with `restart.changes = ["physics.viscosity_m2_s"]` into another directory runs, its snapshot from 1 h. `a_resumed_3d_case_run_ends_bit_for_bit_where_the_uninterrupted_one_does`: an M2 channel in 3D (6 levels, `dt_max_s = 47`, so the 5-minute stations are interpolated between steps across the restart), restarts every 0.25 h land at 20, 30, 50 and 60 min; stopped at 36 min and resumed from 30 min, the stations and the snapshot file byte for byte.
- **The binary.** `cases/tidal_channel` with NetCDF and hourly restarts, stopped at 2.5 h and resumed to 4 h with `--resume`: the log names the restart, its time and the build that wrote it; `tidal_channel_from_2.00h.nc` holds 2, 2.5, …, 4 h, and the stopped run's `tidal_channel.nc` keeps its frames to 2.5 h.

## P8.1: rivers in 3D cases (2026-11-08)

- **`[rivers]`** is `froya_real_data rivers=` (`file`, `snap_km` = `river_snap_km`, default 2; `profile` = `river_profile`, default `"tapered:3"`). The domain reads the river file (`RiverForcing::from_file`, NorKyst-800's NVE rivers as `norkyst_river_subset` cuts them), refuses one that does not cover the run, places the rivers (`RiverSources::on_mesh`: the element of the mouth, or the nearest boundary element within `snap_km`; the others are left out and listed) and logs their discharge over the run, the five largest and the first five left out, as the example prints them. The 3D run adds them after the weather's fluxes (`Hydrostatic3D::with_rivers`), where the example does.
  - `RiverProfile` reads from its text (`FromStr`: `uniform`, `fraction:F`, `top:D`, `tapered:D`), which the example now uses too.
  - Needs `[three_d]` (the 2D module has no rivers: their volume would go in, their T and S nowhere), a placed mesh, `run.start` and NetCDF. A resume checks `[rivers]` as any other key.
- **Bit for bit:** `cases/froya/rivers_3d.toml` (new: `nested.toml` in 3D with 20 levels, the 3D parent, the rivers) against `froya_real_data norkyst=data/froya_norkyst_3d.nc levels=20 tide3d=1 rivers=data/froya_rivers.nc rest_hours=0`, 1 h with 10-minute snapshots: 0 values differ in all 7 frames (η, ū, the surface layer's u and T). Both place 12 of the file's 13 rivers (1.7 m³/s over the hour), river 845 left out at 4.1 km.
- **Found: river water on land nodes (TODO P1.6).** The 1 h run's highest wet η is 5.001 m with the rivers and 1.81 m without. `RiverSources::volume_rate` gives every node of a river's element the same ∂h/∂t, and two river elements hold a node at the land elevation (5 m): element 59 (river 837, snapped 1.5 km onto the mesh; bed 4.47, 2.83, −5.18, 3.23, 5.0, … m) and element 1912. The water stands there (≈ 2 mm in the hour) instead of entering the sea. The example does the same, so the two runs still agree. Fixed 2026-11-09 (`notes/inputs-forcing.md`, rivers): η max 1.807 m after 1 h, the same as without rivers.
- **Gate** (`tests/case_test.rs`, NetCDF): `a_rivers_water_fills_a_closed_3d_basin`: a river of 10 m³/s into a closed 2 × 2 km basin in 3D adds 35,999.999999 m³ in the hour (36,000 to 3e-11), and a second 5 km off is left out and listed. Unit: `rivers_take_the_examples_defaults_and_need_a_placed_3d_run` (the defaults and the refusals), `a_profile_reads_from_its_text`.

## A second coast: Chiloé (P8.6–P8.12, 2026-11-09)

Tracking: #220 (Chiloé) and #221 (a coast-agnostic core). Chiloé (Los Lagos, Chile) is the first domain outside Norway and in the southern hemisphere: what it needs shows where Norway is built in.

- **Inputs mapped:** the parent is IFOP's MOSA (ROMS/CROCO, 1 km, 42 σ, on CHONOS; P8.7), or Copernicus Marine's global 1/12° for the residual and the tracers; the boundary tide is TPXO10-atlas or FES2022 through pyTMD; the bed is GEBCO 15″ (+ SHOA charts); the gauges are SHOA's (Puerto Montt, Ancud, Castro, Melinka); the weather is MOSA-WRF 3 km (ERA5 as fallback); the rivers are FLOW-VIC (climatology) and DGA gauges.
- **Physics that set the domain:** M2 resonates in the inner sea (≈ 2 m range at Boca del Guafo, ≈ 6 m in the Gulf of Ancud, > 7 m at Puerto Montt), so the open boundary lies in the Pacific; Chacao's narrows (2 km, 3–4 m/s) want 100–200 m elements; f < 0.
- **Done here (PR for #212):**
  - `OceanModelReader` reads Copernicus Marine's names (`zos`, `uo`/`vo`, `thetao`/`so`, `deptho`). Gate: `copernicus_marine_names_are_read` (1D lat/lon in the south-west quadrant, z-levels with fill below the bed, no depth variable).
  - `scripts/cmems_parent_subset.py` merges the product's split datasets into one parent file. The dataset IDs and variables were checked against the Copernicus Marine catalogue on 2026-11-09; the download itself is untested here (it needs an account).
  - `scripts/tide_model_boundary_atlas.py` writes a `TidalAtlas` from any pyTMD model along a bbox's sides (and at gauges), leaving out constituents outside `tides::CONSTITUENT_NAMES` (TPXO's S1). It was tested on a synthetic global model (land beyond `extrapolate_km` dropped, currents in m/s). pyTMD 3's extrapolation `cutoff` is in km, not m: the first draft passed metres and filled points 60 km inland.
  - `cases/chiloe/case.toml`: a lon/lat box (−74.6…−72.3, −44.2…−41.3), GEBCO bed, the atlas, `coriolis = "beta_plane"` (f₀ = −9.85e-5 at the box's centre, now in the Coriolis gate), the four SHOA stations.
- **End-to-end on synthetic inputs:** a synthetic GeoTIFF (shelf, a Chiloé-like island, mainland) and an M2/K1 atlas from a synthetic model. `dgcom check` is fine: 41,152 P2 elements at 1 km, 2 of 548 open faces walled for coverage. 6 h at 2 km (10,300 elements) ran stably in 93 s on this machine (|u| ≤ 2.1 m/s, η amplified 1.3 → 2.6 m in the closed basin), so the southern-hemisphere path (placement, raster, atlas, β-plane) works. Nothing is scored until the real GEBCO, TPXO and gauges are in (P8.6).
- **Noticed:**
  - Stations are checked against the mesh: a gauge inside an element dropped as land is an error at `check`. Expect to move some SHOA gauges once the real bed is in.
  - The log's "placed about -42.7500°N": fixed 2026-11-10 (below).
  - The tide-range keys default to Mausund's even under a 7 m range (P8.12).
- **On the real TPXO10-atlas-v2 and GEBCO (2026-11-10):**
  - `scripts/tide_model_boundary_atlas.py` as first written did not run on the real model: pyTMD read every constituent's global 1/30° grid (0.9 GB each as complex128) before the crop, and the currents ran out of memory on a 31 GB machine. It now opens the model lazily (`chunks="auto"`, dask) and computes after the crop: 46 s for the box. TPXO's zip unpacks to `TPXO10_atlas_v2_nc/`; pyTMD wants `TPXO10_atlas_v2/`.
  - The box's west side moved from 74.6°W to 75.3°W. At 74.6°W it ran past Isla Guafo, through TPXO M2 currents of 0.3–0.65 m/s south of 43.2°S; at 75.3°W it lies in 700–3400 m with M2 at 0.42 m and currents under 0.05 m/s. The cost is the deep water's time step (c ≈ 180 m/s).
  - The southern side still cuts the Chonos passages and Moraleda: TPXO's M2 there is 0.7–1.0 m with currents up to 1.3 m/s, the same value at neighbouring points (the nearest model cell's, at filled land points). It prescribes part of the resonance; wall it on a coastline mesh or nest it (P8.6, P8.7).
  - TPXO10 has the amplification already: M2 0.46 m at Ancud, 0.85 m at Melinka, 1.85 m (S2 0.78 m) at Puerto Montt. Castro has no TPXO water within 10 km. So TPXO at the gauges is the baseline the case must beat, not only its boundary.
  - `scripts/cmems_parent_subset.py` read the top 500 m by default, and `depth_average_z` extends the deepest level read down to `deptho`: on a 3000 m boundary the depth-mean current would be the upper 500 m's. It reads the whole column by default now.
  - GEBCO: `download.gebco.net` is a web form; NOAA's ERDDAP serves `GEBCO_2020` as NetCDF by URL (its `.geotif` is an 8-bit picture), converted to a float GeoTIFF with rasterio. `dgcom check` passes on it with all four gauges on the mesh.
- **First runs on real inputs, scored at SHOA's gauges (2026-11-10):**
  - `cases/chiloe/case.toml` as committed: 51,172 P2 elements at 1 km (bed to −3816 m), 21 of 606 open faces walled for coverage, 24.84 h in 64,088 steps (dt 0.26–1.40 s) and 1348 s on this machine, |u| ≤ 3.4 m/s, stable with wetting and drying. At 2 km (13,221 elements): 227 s, but Castro has no wet element within 8 km and Ancud's is 3 km off.
  - The gauges are on the IOC Sea Level Station Monitoring Facility (`pmon`, `ancu`, `cstr`, `pmel`; 1-minute records). `scripts/ioc_gauge_skill.py` (new) fetches them and scores a run's `stations.csv` per window, with a tide model's own prediction as the baseline.
  - Second M2 period (12.42–24.84 h) of 2025-11-12, a neap day, centred RMSE and range:

    | gauge | observed range | run, 1 km | run, 2 km | TPXO10 |
    |---|---|---|---|---|
    | Puerto Montt | 2.97 m | 3.79 m, 33 cm, 10 min late | 3.90 m, 51 cm, 40 min late | 2.90 m, 6 cm |
    | Ancud | 1.11 m | 1.10 m, 7 cm, 20 min early | 0.92 m, 13 cm | 0.72 m, 18 cm |
    | Castro | 2.62 m | 0.00 m (cut off) | no element | no water within 10 km |
    | Melinka | 1.47 m | 1.50 m, 7 cm, 10 min late | 1.53 m, 9 cm | 1.47 m, 1 cm |

  - Read: the tide enters right (Melinka, Ancud, where the run beats TPXO), and the resonance is there but 28 % too strong at Puerto Montt. One day after a 3 h ramp is not a spun-up resonance, so this is a first look, not a score: candidates are the spin-up, the friction (Manning 0.025 everywhere), GEBCO's interpolated bed in the channels, and the southern side's prescribed 1 m M2. TPXO10 is very good at Puerto Montt and Melinka (it likely assimilates them) and poor at Ancud.
  - **Four days, neap towards spring (99.36 h, 256,349 steps in 5240 s; `run.name=chiloe_4d`):** the over-amplification at Puerto Montt is not the spin-up. Run range over observed per M2 period, from the second: 1.28, 1.28, 1.25, 1.22, 1.21, 1.19, 1.20 (3.79/2.97 m to 4.95/4.13 m), centred RMSE 26–33 cm against TPXO10's 6–15 cm, timing within 10 min. It falls as the tide grows, which is what too little friction would look like (quadratic drag), but the bed and the southern side are not ruled out. Ancud holds at 7–10 cm (TPXO10: 18–26 cm, 40–60 min early) and Melinka at 7–13 cm in the periods its gauge covers.
  - Melinka's record has holes of hours (56–80 h here). The script interpolated across them, which showed as 27–35 cm for the run and TPXO10 alike; it now leaves out run samples further than `max_gap_min` from a gauge sample and does not score a window under 80 % coverage.
  - Not looked into: the domain's highest water stays at one value for an hour at a time while the tide falls (+3.80 m at 88.5–89.3 h, +4.73 m at the end), which read as water left in a drying cell. It is not (2026-11-10, the Taitao run's snapshot): no element holds more than 5 cm above +2 m through a tidal cycle. The progress line takes η over every node deeper than `h_dry` (1 mm), and 500–1100 nodes carry a film of 1–50 mm on beds of +1 to +4 m (one had 22 cm on a +2.8 m bed, 1.2 m above the tide around it). Among nodes at least 1 m deep the highest water was the tide's +1.86 m. The line's largest speed is real: 3.1 m/s in 47 m of water in Chacao (TODO P8.13).
  - Castro's station sits in a 4.1 m deep element that GEBCO at 1 km leaves cut off from the sea (range 0.00 m): its fjord needs the coastline mesh.
- **Where the over-amplification comes from (2026-11-10; scratch scripts, 2 km screening runs of 37.26 h scored on 12.42–37.26 h, then 1 km):**
  - Method: one frequency per band (D1, D2, D4, D6) fitted by least squares to the run and the gauge at the same samples, so amplitude ratio and lag come apart; a lattice of 296 stations every 0.1° against TPXO10's own tide at the same points; one factor changed per run.
  - By band, 1 km, four days: Puerto Montt D2 ×1.23 (+7 min) and D1 ×1.23 (−2 min); Ancud D2 ×0.98, Melinka D2 ×1.06. The quarter-diurnal band is 2–3 times the gauges' at Ancud and Melinka (1–4 cm amplitudes).
  - The lattice (2 km), run over TPXO10 in D2: 1.00 off Ancud, 0.92 in Boca del Guafo, 1.13 in Corcovado, 1.28 at Desertores, 1.36 in the Gulf of Ancud, 1.39 in Reloncaví: it grows along the whole axis, at no single passage, the mark of a basin nearer its quarter-wave resonance than the real one. On the shelf by the south-west corner the run is 0.76–0.86 of the atlas it is forced with. D1 is 1.04–1.08 everywhere, the open ocean too.
  - One factor at a time, 2 km (Puerto Montt D2, D1, D2 lag; Melinka D2; Ancud D2 — run over gauge):

    | run | PM D2 | PM D1 | PM lag | Melinka D2 | Ancud D2 |
    |---|---|---|---|---|---|
    | Manning 0.025 (the case) | 1.28 | 1.17 | +39 min | 1.01 | 0.91 |
    | C_D 0.0025 | 1.22 | 1.18 | +42 | 0.98 | 0.88 |
    | C_D 0.004 | 1.16 | 1.21 | +47 | 0.94 | 0.83 |
    | Manning 0.035 | 1.22 | 1.20 | +46 | 0.96 | 0.85 |
    | `transport_scaling = 3` | 1.27 | 1.17 | +38 | 1.00 | 0.91 |
    | atlas without land-filled points (2 km fill) | 1.28 | 1.17 | +40 | 1.00 | 0.90 |
    | southern side's currents capped at √(g/h)·η | 1.26 | 1.17 | +35 | 0.99 | 0.91 |
    | southern side's currents zero | 1.21 | 1.13 | +20 | 0.94 | 0.93 |
    | southern side walled east of 74.05°W | 1.15 | 1.07 | +25 | 0.90 | 0.91 |
    | box to 46.6°S (below) | 1.05 | 1.15 | +55 | 0.79 | 0.83 |
    | inner sea's bed ×1.2 (east of 74°W, under 600 m) | 1.24 | 1.20 | +8 | 1.15 | 0.97 |
    | inner sea's bed ×0.8 | 1.21 | 1.42 | +86 | 0.79 | 0.76 |

  - Read. The friction is a weak lever: three times the drag takes 12 points off D2, adds to D1 and costs Ancud 8 (Manning 0.025 is C_D 1.3e-3 at 100 m and 1.05e-3 at 200 m, half the 2.5–3e-3 tide models use; there is no floor). The currents copied into land-filled atlas points do nothing. The southern cut is the strong lever on the level of the whole inner sea, and Melinka's score there was the boundary's, not the model's. The bed is the strong lever on the mouth-to-head ratio and the timing (Puerto Montt over Melinka: 1.27, 1.08 with the bed ×1.2, 1.53 with ×0.8); the 2 km mesh holds the same volume as the 1 km one to 0.04 %, so its late, over-amplified tide is the passages' representation, not lost depth. GEBCO's basin means are within 4 % of TPXO10's own bed, which is no independent check.
  - **The box now reaches 46.6°S** (`bbox = [-75.9, -46.6, -72.3, -41.3]`, 288 × 589, 103,016 elements): Taitao closes the Moraleda and Aysén channels on land, the west side is in more than 1500 m everywhere, and the only southern water is 25 km of deep ocean. At 1 km, 12.42–37.26 h, against the box cut at 44.2°S:

    | gauge | D2 ratio, was → is | centred RMSE per M2 period, was → is | D2 lag |
    |---|---|---|---|
    | Puerto Montt | 1.30 → 1.10 | 33, 30 → 17, 13 cm | +11 min |
    | Ancud | 1.00 → 0.92 | 7, 8 → 8, 9 cm | −24 min |
    | Melinka | 1.08 → 0.94 | 7, 7 → 5, 6 cm | +11 min |

    The run is 97,473 steps in 3897 s for 37.26 h (the 44.2°S box: about 2000 s), at one step throughout (37.26 h over 97,473 steps is 1.376 s, the largest being 1.377 s): the cost per step doubled with the elements. The reported smallest step of 0.003 s (0.26 s before) was a step shortened to land on an output time, not a stability bound; `SimulationResult::dt_min` no longer counts those (2026-11-10). Not yet run over the four days. Castro has no element within 3 km on this grid and is out of the case. The old inputs are kept as `data/chiloe_gebco_box44.tif` and `data/chiloe_boundary_tides_box44.txt`.
  - Left: Puerto Montt's 10 % and D1's 11–15 % at all three gauges (the lattice has D1 5–8 % over TPXO10 in the open ocean, so that part comes in at the boundary). Next in order: the coastline mesh (the passages), an independent bed for the inner sea (SHOA soundings, or MOSA's grid, P8.7), and only then the friction, with a floor on C_D rather than a larger Manning.
- **`Case::coriolis_source` (2026-11-10):** the 2D and 3D runners each built the Coriolis term from `physics.coriolis`; they take it from one place now, and `dgcom check`/`run` log it and the placement with hemispheres ("placed about 42.7500°S 73.8000°W", "Coriolis f = -9.900e-5 s⁻¹ + 1.681e-11 m⁻¹ s⁻¹ × y"). Gate: `the_coriolis_term_follows_the_placement_in_both_hemispheres`.


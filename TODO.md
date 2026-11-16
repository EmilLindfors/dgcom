# TODO

What is worth doing next, one line each, and nothing else. The outcome these lines serve: **farm-site currents with a measured error, on any coast** (and the particles and waves around the cages built on them). A line belongs here if it is on that path now, or is a known defect of the runs made at farm depths.

The triage of 2026-11-15 (`notes/overview.md`) cut this list from 399 lines to these. The rest is in each area's notes file under "Out of TODO.md (2026-11-15)", verbatim: folded into a line here, parked (it comes back when a run, a measurement or a user asks for it) or dropped (with the reason). A new opportunity goes to its notes file's Parked list; a new line here needs a reason it beats the ones already here.

The context of each line (measurements, decisions, what was tried) is in [`notes/`](notes/README.md): each section heading links its notes file, and searching it for the item's ID finds the details. When an item is done, delete its line and record the outcome in the notes file (and `CHANGELOG.md` if notable).

Direction (from the 2026-09-25 review, `notes/review-2026-09-25.md`): the salmon-farm application (currents, waves and particles around cages inside a NorKyst-driven coastal domain) on a 2D model that is correct, cheap and validated, and a 3D model rebuilt on the ROMS recipe. "Faster than ROMS" is to be proven by P3.2 (parked), not assumed.

**Up next** (2026-11-15): at a nested gauge the η score measures the parent (the model follows NorKyst within 0.5–0.7 cm), and where dgcom differs from NorKyst, in currents under 50 m, nothing measured checks it. (1) **P3.4** the nested 2D model over 15 days, then (2) **P1.5** the default `nest_tides` from it; (3) **P3.1** measured currents beside them. The sections below are in order of priority.

## Validation (P3) — [notes/validation.md](notes/validation.md)

- [ ] **P3.4** Score the nested 2D model at Bodø and Mausund over 15 days or more (a spring–neap cycle) as case runs, `nest_tides` raw and theory, so that `dgcom score` keeps them current: NorKyst's own daily centred RMSE cycles 2.3–12.1 cm at Bodø and 2.1–9.6 cm at Mausund over June 2026, and the one pre-registered day (2026-06-12) fell on its worst (`notes/validation.md` P3.4). 2D parent files of 15 days load whole (≈ 22 MB per 3 days); the 3D runs wait for parents read lazily (P1.5, parked).
- [ ] **P3.1** Measured currents at farm sites inside the Frøya and Bodø boxes, a run of their window nested in NorKyst v3's hindcast (2012-01 to 2025-08), and `dgcom score` scoring a station's current against the record (`ADCPValidationResult`, the ellipses of `fit_ellipses`). The only check of the model where it differs from NorKyst: the island lee's M2 current 0.12–0.19 against 0.255 m/s, 0.56–0.61 of NorKyst where NorKyst is under 50 m deep, 3D 0.78 of 2D in 3–20 m; about ten parked lines wait on it (the 3D bed roughness, Kattholmen's rerun and wake, the particles' K, Frohavet). Leads in `notes/validation.md` (Olausskjæret, Sulfjorden, 2019; SINTEF's Frøya cage-current study; the site reports in hearings); the raw records need requests to operators and consultancies.

## Boundaries and nesting (P1.4, P1.5) — [notes/boundaries-nesting.md](notes/boundaries-nesting.md)

- [ ] **P1.5** Choose the default `nest_tides` (gauge-`corrected` or `theory`) on the 15-day score (P3.4): theory needs no gauge, so it is the one an out-of-sample domain can use.

## Generic cases and case files (P8) — [notes/case-files.md](notes/case-files.md)

- [ ] **P8.6** Chiloé over a spring–neap cycle with a harmonic score per constituent at SHOA's gauges (is Puerto Montt's excess, 22 % over the gauge at neaps and 7–9 % three days later, in M2 or in N2 and the minors?), before any drag tuning or finer mesh. Its inner sea is where η tests dgcom rather than a parent: it is forced by an atlas (TPXO10). The four-day scores and the rest of the list (100–200 m in the channels, a measured bed for Guafo and Moraleda, the shelf, the minor diurnals) are in `notes/case-files.md`.
- [ ] **P8.1** `dgcom score` against the parent as well as the gauge: the parent's own ζ at the gauge (as `parent_tide_check` samples it) scored against the observations over the run's window and over the parent file's month, and the model against the parent, beside the model's score. A run that leaves its parent shows at once (the walled parent file put Mausund ≥ 5.4 cm from NorKyst, against ≈ 0.5 cm for an honest nest), and a day's rank in the parent's spring–neap error is printed with it (`notes/overview.md`, the 2026-11-15 pass).
- [ ] **P8.2** Retire what has moved: the example's 2D, 3D, wave and farm drivers now duplicate the case runner (`tidal_run`, `tidal_run_3d`, `coupled_waves`, the farm's runs), and its 3D start keeps its own orchestration around the library's blocks for the `debug_3d` switches. Keep the diagnostics (`debug_3d=`, `profile=`, `waves=N`, the farm's wake report) as tools that read case files, and delete the rest with its options; the example's documented commands become case files.
- [ ] **P8.6** Puerto Montt's gauge lies on the coastline mesh's land, so both its sampling points are on the mesh's wall: 10.8–12.2 m deep and 45–184 m from the gauge over the #231 sweep, moving with the meshed coast. Keep the gauges in the water when meshing (a point to keep at each gauge in `scripts/gmsh_coastline_mesh.py`), so a station is sampled at its own position (`notes/case-files.md`, "a station's current", 2026-11-14).
- [ ] **P8.6** Chiloé's gauges in the gauge format (`read_tide_gauge_file`: the IOC 1-minute records made hourly, as `scripts/uhslc_gauge.py` does UHSLC's) and as `gauge =` on its stations, so its runs write `gauge_skill.csv` and `dgcom score` covers them; `scripts/ioc_gauge_skill.py` is then a second scoring path to retire.
- [ ] **P8.12** Mausund's tide as the 3D bed's defaults (#219, the part left; the presets and station lists went 2026-11-10): `[bathymetry.slopes_3d]`'s `pond_low_water_m` −1.43, `high_water_m` 2 and `cliff_depth_m` 20, `[bathymetry] wall_low_water_m` −2, `[bathymetry.raster] land_elevation_m` 5 and the 19 m pycnocline (`free_depth_m`). Derive the levels from the case's tide (± `TidalAtlas::largest_amplitude_sum`), require them without one; Frøya's atlas gives ±1.40 m, so the shipped 3D cases change bits and need their scores rerun (`notes/tech-debt.md`).

## Salmon farm and cage drag (F.1) — [notes/farm.md](notes/farm.md)

- [ ] **F.1** The farm's analysis for case files: the wake against the open run, the residual current, the drag on the farm and the runs' divergence (`examples/common/farm_site.rs`, 1,800 lines) as a library report over two (or three, with a control) case runs that differ only in `farm.drag_share`, sampling the snapshots on the farm's grid.
- [ ] **F.1** Check a site's guessed cage count and size (`FarmSite::cage_layout`, Kattholmen 2 × 4) against an aerial photo or the operator; Kattholmen's two southern cages stand on a 9–10 m bed under 20 m nets (the 50 m topobathy), so check the bed at the frame too.

## Particles (F.2) — [notes/particles.md](notes/particles.md)

- [ ] **F.2** Particles in case files (`[particles]`): releases in zones (a farm's cages, a polygon), the walks, seeds and the connectivity matrix, as `froya_real_data particles=` tracks them online in the farm's run.

## 3D: numerics (P4.2, P4.3, P4.5) — [notes/3d-numerics.md](notes/3d-numerics.md)

- [ ] **P4.5** 3D tracers leave the physical range where columns are dry or thin: a dry node at 28.8 °C beside wet neighbours at 10.4 (`output/froya_3day/3d_rk43`, 57 h); thin columns from −2.00 °C (the bound) to 21–36 °C in the 5–8 June 2026 runs while the others stay at 6.9–14.8; the lowest S from 9.4 to 7.8 after a day of the MET run. The lice's behaviour reads T and S. Find the step that does it (the shoreline subcells, `from_inventory`, the surface fluxes thin columns do not get), and hold a dry node's T and S at their last wet values (`notes/3d-numerics.md`, `notes/3d-mixing.md`, `notes/3d-validation.md`).
- [ ] **P4.5** Decide the 3D defaults on one 72 h nested Frøya run on today's bed, scored against `output/froya_3day/3d_rk43` with `norkyst_profiles_3d`: local time stepping (`levels_3d`; Kattholmen 640 → 370 s per model hour at 3 levels), SSP-RK(4,3) with a doubled 3D CFL, implicit vertical advection, the viscosity once per step, the horizontal Kuzmin tracer limiter, the step cap (`dt_max_s` 60 against 120 s: 450 against 376 s per model hour on Chiloé) and a 3D `cfl` of its own. Each is a switch the cases set by hand today (`notes/3d-numerics.md`, `notes/performance.md` P2.9).

## Geometry and meshes (P1.3) — [notes/geometry-mesh.md](notes/geometry-mesh.md)

- [ ] **P1.3** The 3D bed layers over steep beds do not stay at rest: the shipped rest state grows at terrace vertices beside shore elements (oscillatory, e-folding 2–2.4 h, period ≈ 3 h); the coastline day cools a 41–56 m basin's bed layer (element 5143) from 8.4 to 7.0 °C in an hour under 0.2–0.33 m/s and a depth mean of 0.01–0.04; Mausund's pinnacle element 291 cools its bed layer below NorKyst's minimum. Farm depths, so a Stage B defect: find the mode (the straddle or terrace mode at the r_x0 bound), gate `a_terrace_beside_a_shore_element_stays_at_rest_for_twelve_hours`, then rerun the coastline mesh at rest for 24–48 h, seeded (`notes/geometry-mesh.md`).

## 3D: validation (P4.6) — [notes/3d-validation.md](notes/3d-validation.md)

- [ ] **P4.6** The model starts 0.2–0.25 °C colder than NorKyst at the farms (0 and 3 m, from day 0, inside the mesh): find which start-up step cools the coastal columns (initial T/S at the 1442 nodes on NorKyst's mean profile, the bed blending, the cliffs and ponds).

## 2D numerics (P1.1, P1.2, P1.7) — [notes/2d-numerics.md](notes/2d-numerics.md)

- [ ] **P1.7** Nonlinear SWE convergence with bathymetry on curved and jittered meshes, `EntropyStable` and `WetDry` included, asserting N+1 in `tests/convergence_test.rs` (thresholds 1.5/2.5 today; the averaged metric loses an order pointwise on jittered meshes, `notes/geometry-mesh.md`).
- [ ] **P1.2** Films on land that do not drain: on Chiloé 500–1100 nodes carry 1–50 mm of water 1–2 m above the high water through the ebb (one 22 cm on a +2.8 m bed); on the Frøya coastline bed a 0.2 m film on +0.27 m land runs at 0.8 m/s for hours; under viscosity a film node reaches the 20 m/s cap; in 3D a 1 m shore element sets the step (0.24 s). One cause or several: log the face and subcell mass fluxes at one such node first (`notes/2d-numerics.md`, `notes/case-files.md` P8.13; the original lines are folded there and in `notes/geometry-mesh.md`).

## Inputs and forcing (P1.6) — [notes/inputs-forcing.md](notes/inputs-forcing.md)

- [ ] **P1.6** Settle the topobathy sea part's vertical datum (NN2000 or chart datum, 1.43 m too shallow at Mausund): check against WFS `wfs.dybdedata` points; add the 8 cm NN2000 → MSL offset if it matters.

## Tech debt (P6) — [notes/tech-debt.md](notes/tech-debt.md)

- [ ] **P6** Delete the legacy 2D paths in one go: move `profile_cpu`, `quick_profile` and `high_res_benchmark` onto `Simulation` + `SWEPhysics2D`, then delete the `ssp_rk3*` steppers and `SWE2DTimeConfig`, the `Standard` wet/dry path with the cell-average/`linearize` workarounds and its kernels, `solver/simd/`, and `Discharge2D`/`ConstantDischarge2D`. Merge the three legacy conservation tests into one on `Simulation`, and fold `Simulation3D::run_with_callback` into one time loop that lands callbacks exactly (`notes/tech-debt.md`).
- [ ] **P6** Delete the code without callers: `LandMask2D`; `SillOverflow2D`, `SillWithFriction`, `SillDetector`, `StraitFriction2D` and `TidalPotential` with their presets (≈ 1,800 lines); `TidalConstituent::with_nodal_correction` and `tides::correct_amplitude_phase` (`notes/tech-debt.md`).

## Viewer (F.3) — [notes/viz.md](notes/viz.md)

- [ ] **F.3** Build `viz/` in CI (a `cargo check` in its workspace, or a job on PRs touching `src/` public API): it is its own workspace, so dg-rs's tests never compile it, and it stayed broken for a week after `4d3a160` (`notes/viz.md` 2026-11-05).

## GPU (P2.7) — [notes/gpu.md](notes/gpu.md)

- [ ] **P2.7** Time a case on the RTX PRO 6000: `gpu/time.sh` runs the gates on the card, then the refined tidal channel on its CPU and GPU, and Nsight Systems (setup, and what to record: `gpu/README.md`); predicted ≈ 1.7 ms per step against the laptop CPU's 16. The RTX 3090 holds every gate bit for bit and takes 7.5 ms at 64,000 elements (predicted ≈ 6; 2.0× the laptop's GPU and its 24 threads), 23.3 ms at 256,000, FP64-bound but for the subcells and the wet/dry correction (`notes/gpu.md`, 2026-11-16); then Nsight Compute's FP64 counts (and what holds `wet_dry_correction` on the 3090); then fewer FP64 instructions: since each pass's elements are split by kind, short passes spread over the SMs, the host keeps unchanged schedules and the shared physics divides less (2026-11-16) the 3090 takes 34.7 s per Chiloé model hour (54 before; the laptop's 24 threads 55), its large passes FP64-bound near the card's rate (`notes/gpu.md`, "the passes split by kind, and fewer divisions": Manning's cube root, the GLL constants, then the short passes' chains); then the weather's device form (`GriddedAtmosphere2D`, `InverseBarometer`). The scored Bodø and Mausund nests score on the GPU as on the CPU, to the printed digits; local time stepping (`Multirate`, bit for bit, 1.31× on the refined channel), atlas tides per boundary node (`cases/tidal_channel/atlas.toml`) and nesting with its relaxation band (`dgcom-gpu nest`), all bit for bit, run on the device since 2026-11-16. A case runs on the GPU since 2026-11-15 (`[compute] device = "cuda"`, `gpu/`'s `dgcom-gpu`), bit for bit with the CPU (`cases/tidal_channel`). Phases 1–2 would put the 15-day runs (P3.4) and Chiloé's spring–neap cycle (P8.6) at minutes per model day.

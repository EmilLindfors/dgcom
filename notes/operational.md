# Operational features and future work (P5, P7)

## Priority 5: Operational Features

### P5.1 River forcing
- [x] A ROMS river-file reader onto `source::RiverSources` (2026-10-21, PR #164, `io::RiverForcing`; see CHANGELOG). NorKyst-800 v3's river forcing is public: `romshindcast/norkyst_v3/forcing/river.nc` on MET's THREDDS, 1801 rivers "created by makerivers.f90 based on data from NVE and other sources", daily from 2011-01-01 to 2026-07-01, on the grid of `fou-hi/norkystv3_his_files/static_nc_files/norkyst_grd_v31.nc` (2747 × 1148 ρ-points). NorKyst runs them as `LuvSrc` (`roms_norkyst.in` in Zenodo 10.5281/zenodo.16810677): 785 rivers on u-faces, 1016 on v-faces, none vertical.
  - Mouths: the midpoint of the face, half-way between NorKyst's land cell (its 10 m minimum depth) and the sea cell (`roms_source_position`). The discharge is |transport| (its sign is the direction across the face; a river whose sign changes is refused, as is a `river_flag` with a tracer off, which in ROMS brings the receiving cell's own value).
  - T and S: the levels' values weighted by `river_Vshape`.
  - `examples/norkyst_river_subset.rs` cuts a box's rivers out over OPeNDAP (the grid window of `common/dap.rs`, then only the rivers' and days' range of `river_transport`, `river_temp`, `river_salt`; the whole file has 3.3 GB of temperatures) into a small file with `river_lon`/`river_lat`; `froya_real_data rivers=<file>` runs them in the 3D tide (`RiverSources::on_mesh`: rivers more than `river_snap_km` from the mesh are listed and left out).
  - Frøya (`bbox=8.0,63.6,9.2,64.0`, +0.05°, 2025-06-15 + 3 days): 13 rivers, small island and coast catchments, 2.05 → 11.2 m³/s in total over the four days (rain on the 18th; the largest ≈ 1 m³/s), water at 9.6–10.2 °C and **S = 1** (NorKyst's river salinity, not 0). On the coastline mesh (`froya_coast.msh`) 12 are placed, 5 of them snapped 1.6–1.9 km (NorKyst's 800 m coast against the mesh's), two into the same element; one (845) is 4.1 km off and left out at `river_snap_km=2`. The first 0.5 h of the 72 h 3D configuration with them runs as before (79 s); the lowest S near the sources goes 10.6 → 7.7 → 10.0 over the half hour. The subset took 10 min, nearly all of it the DAP grid lookup (a slow THREDDS day), the rivers' 4 days 12 s.
  - **NorKyst's vertical shape is a fixed depth: the top ≈ 3 m.** `river_Vshape` spans 3 to 35 of the 40 levels, more in shallower water. On NorKyst's s-levels (Vtransform 2, Vstretching 5, θ_s 5, θ_b 4, h_c 100 m) the shaped levels end at 2.55 m (h = 286 m, 3 levels), 2.79 m (61 m, 6), 2.98 m (23.5 m, 11), 3.01 m (10 m, 19), with the weight per metre falling about linearly from the surface to zero at 3 m. `TopFraction(0.2)` (the default then) spread the water over the top 2–60 m instead; `RiverProfile::TaperedTopMetres(3.0)` is NorKyst's and the default since 2026-10-21 (P1.6 in `inputs-forcing.md`).
- [ ] `Discharge2D`/`ConstantDischarge2D` (weak boundary inflow, `boundary/boundary_2d.rs`, the `River` tag's dispatch in `multi_bc_2d.rs`) are superseded by `RiverSources`: delete them with the P6 cleanup (`RiverTracerSource` went with the 2D tracer path, 2026-10-15).
- [ ] The mouths are NorKyst's 800 m coastline faces, up to ≈ 1 km from the real mouth. On farm-scale meshes (tens of metres) take the outlets from NVE's river network (e.g. Elvenett, REGINE catchments) and match them to NorKyst's rivers, which carry only an id.

### P5.2 ROMS-compatible output
- [ ] `NetCDFWriter` produces CF-1.8 but not ROMS-compatible output; add a ROMS variable-name mapping.

### P5.3 Restart / checkpointing
- [ ] HDF5 or NetCDF restart files (operational daily forecasts must hot-start).
  - [x] 3D restarts, for the multi-day 3D runs (15 days of the Frøya coastline mesh in 3D take ≈ 3–18 days of wall time, P1.3). Done 2026-10-06 (`io::Restart3D`, see CHANGELOG). A restart holds the whole `Solution3D`, the splitter's AB3 history of `G` (`SlowForcingRecord`) and the vertical reference, plus a domain fingerprint checked on resume. The barotropic pass starts from the state, and the forcing and clock are functions of time, so nothing else carries over. `froya_real_data tide3d=1 restart_hours=H` writes restarts, and `resume=<file>` continues one, with the station records and the snapshot file. Gate: `a_restarted_run_ends_bit_for_bit_where_the_uninterrupted_run_does`. Mausund, resumed in a new process: outputs byte-identical.
    - [x] Only one restart file was kept (each replaces the last, atomically). Done 2026-10-28: `restart_keep=1` keeps every restart as `<domain>_<hours>h.restart`, in 2D and 3D (`RestartSchedule` in `froya_real_data`).
    - [ ] The physics' configuration is not in the file: a resume with other options (mixing, limiter, nesting, wind) runs on silently from the restart's state. The example records the command line (`command` metadata) and prints it on resume. Compare the options that change the physics and refuse a mismatch, or store a configuration hash. A 2D branch run changes the physics on purpose (`site=` adds cages to an open run's state, TODO F.1), so the check needs a list of the options a branch may change, not a hash of all of them.
    - [x] 2D runs. Done 2026-10-28 (`io::Restart2D`, see CHANGELOG). A 2D run carries nothing but its `SWESolution2D` from step to step: `MultirateStepper::assign_levels` recomputes the levels from the state every step (no hysteresis), its stage buffers are scratch, and `SWEPhysics2D`'s `Mutex`/`OnceLock` fields are caches of the state or the geometry (the viscous step bound) or diagnostics (`negative_depth_clips`, which restarts from 0). So the restart is the state, the time and a fingerprint of the mesh and bed (`domain_fingerprint_2d`; the 3D fingerprint adds the σ-grid and is unchanged, so existing 3D files read). The format is the 3D one's with magic `DGRST2D`; `Restart3D`'s driver metadata and arrays moved into `DriverData`, shared by both.
      - Gate: `a_restarted_2d_run_ends_bit_for_bit_where_the_uninterrupted_run_does` (a `WetDry` beach with Coriolis and Smagorinsky over a background, global SSP-RK3 and `MultirateSSPRK3` at up to 3 levels, restarted at 140 s of 300).
      - `froya_real_data` (2D tide): `restart_hours=H` writes at the first callback at or after every H hours, with the station records and the counters (callbacks, VTU frames, the initial volume); `resume=` continues them, the snapshot file (`SnapshotWriter::append`) and the VTU numbering, and writes NetCDF to a new `froya_from_<hours>h.nc` (the stopped run's file stays). A resumed run's first callback (the restart's time) only samples the farm, whose records are not in the restart. Refused with `wave_coupling=` (the spectra are not kept).
      - Kattholmen (`cs=0.1 nu=0.1 lts=8`, 24 threads, 136 s per model hour), 0 → 1 h with restarts at 0.5 and 1 h against a copy resumed from 0.5 h in a new process: the snapshot file, the three VTUs, `froya.nc` and the station and current files are byte-identical; the 1 h restarts differ only in the `command` metadata (first difference 400 bytes from the end).
      - Callbacks land at `t_restart + k·interval` in the resumed run against `k·interval` in the uninterrupted one: the same in floating point for whole-second times and intervals (all of `froya_real_data`'s), possibly not in the last bit otherwise.
    - [ ] A NetCDF export of the restart (ROMS-like `ocean_rst.nc`) for other tools, once P5.2's output conventions are settled. The binary format stays the restart itself (exact `f64`, no native dependency, runs in CI).

### P5.4 Operational pipeline
- [ ] Automated run scripts, monitoring and alerting, THREDDS/OPeNDAP serving.

## Priority 7: Advanced Features (Future)

### Research numerics
- [ ] IMEX time stepping via diffsol (implicit vertical diffusion / free surface; see P2.5).
- [ ] Subcell positivity preservation with convex limiting (Wu et al. 2024).
- [ ] hp-adaptive mesh refinement.
- Entropy-stable DG (split form done, see P1.2), sum factorisation and curved elements are now in P1.2, P2.2 and P1.3.

### Operational
- [ ] Data assimilation (start with EnKF, then 4D-Var).
- [ ] Two-way nesting (child feeds back to parent).
- [ ] MPI for distributed memory (see P2.7).
- [ ] Wave–current interaction: now F.4 (the spectral wave model, stage 1 done 2026-10-03; coupling is a stage there).
- [ ] Biological coupling (NPZD). Salmon-lice dispersion is particle tracking, now F.2 under "Application target".
- [ ] Sediment transport.

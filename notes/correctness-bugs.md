# Confirmed correctness bugs from the 2026-09-25 review (P0)

## Priority 0 (2026-09-25 review): Confirmed Correctness Bugs — OPEN

All confirmed in code or by measurement (`review-2026-09-25.md` "Top correctness bugs"). Each is small. Each needs a regression test that would have caught it.

### P0.12 Flather characteristic relation applied twice (BLOCKER, open boundaries) — FIXED: [PR #4](https://github.com/EmilLindfors/roms-rs/pull/4)
- [x] **The bug.** Ghost normal velocity is set to `un_ext + √(g/h)(η_int − η_ext)`, and the Riemann solver then applies the characteristic relation again.
  - Result: reflection coefficient ≈ −1/3; a forced tide arrives at ≈ 0.65 amplitude; −0.57 reflection for the `ChapmanFlather2D` defaults.
  - Affected: `Flather2D`, `HarmonicFlather2D` (`boundary_2d.rs:381,821`), `TSTOBC2D` (`tst_obc.rs:347`), `NestingBC2D` (`nesting_bc.rs:189`), `OceanNestingBC2D` (`ocean_nesting.rs:218`), `ChapmanFlather2D` (`chapman.rs:278`).
- [x] **Fix.** Ghost = external state (η_ext, u_ext) and let the upwind flux impose Flather. `OceanNestingBC2D::with_flather(false)` is already correct.
- [x] **Tests.** Outgoing-pulse reflection < 1 %; delivered progressive-tide amplitude ≥ 0.95. (`tests/open_boundary_flather_test.rs`: reflection < 1e-5, amplitude 0.997–1.001.)

### P0.13 Hydrostatic reconstruction leaks mass when B jumps across faces (BLOCKER, 2D) — FIXED (2D and 1D, [PR #6](https://github.com/EmilLindfors/roms-rs/pull/6))
- [x] **The bug.** `f_int = normal_flux(&q_int_flux)` uses the *reconstructed* state (`swe_2d.rs:494`, parallel `:1021`). Measured: relative mass change −5.5e-5 per second with cell-averaged B and slope-correlated flow; ~1e-20 with continuous B.
  - Active in `froya_real_data` (`to_cell_average` + `with_well_balanced(true)`).
- [x] **Fix.** Use `normal_flux(&q_int)`, keep F* from the reconstructed states, and add ½g(h*⁻² − h⁻²)n to the momentum components only (DG Audusse / Xing–Shu). Apply identically in serial and parallel.
- [x] **Tests.**
  - A periodic mass-rate test with cell-averaged B and u ∝ cos·cos (a uniform u hides the bug by symmetry).
  - Lake-at-rest with cell-averaged B, serial and parallel.

### P0.14 Documentation directs users to the unbalanced configuration (MAJOR, 2D) — FIXED: [PR #11](https://github.com/EmilLindfors/roms-rs/pull/11)
- [x] The `SWE2DRhsConfig::well_balanced` / `with_well_balanced` docs (`swe_2d.rs:50-53,142-153`) say to omit `BathymetrySource2D`. Doing so gives ≈ 0.6 m/s² lake-at-rest residuals with nodal B. The advice holds only for cell-constant B.
- [x] The `Bathymetry2D::linearize()` doc says "linear B is well-balanced", which is false at p = 1 (0.07 m/s² measured).
  - Doc corrected (p ≥ 2 only, face jumps need `with_well_balanced`, split form balanced for any B); `test_lake_at_rest_linearized_bathymetry`.

### P0.15 Truncated tidal constituent periods (MAJOR, tides/analysis) — FIXED: [PR #2](https://github.com/EmilLindfors/roms-rs/pull/2)
- [x] **The bug.** `TidalConstituent::{m2,k1,o1,n2,p1}` (`boundary/tidal.rs:76-113`) use 12.42 / 23.93 / 25.82 / 12.66 / 24.07 h. They feed `HarmonicAnalysis::standard()`/`norwegian_coast()` and the harmonic BCs.
  - Phase error on a 60-day fit: ~1° (M2) to ~7° (P1); >10° over a year.
- [x] **Fix.** Use `constituent_period()` (`constituent_reader.rs:156`) or Doodson speeds.
- [x] **Test.** Fit a *true-frequency* synthetic signal; the phase error must be < 0.1°. The current round-trips synthesise with the same truncated periods.

### P0.16 3D vertical momentum advection too large by a factor of D (BLOCKER, 3D) — FIXED: [PR #2](https://github.com/EmilLindfors/roms-rs/pull/2)
- [x] `apply_vertical_advection_field` divides the Ω·u flux difference by Δσ (`advection_3d.rs:641`). Ω is in m/s, so the divisor must be Hz = D·Δσ; the tracer version (`:688`) already does this.
- [x] Test: a column with known linear Ω and u(σ) at D = 200 m, compared analytically.
- [PR #2](https://github.com/EmilLindfors/roms-rs/pull/2) also moves Ω to w-points (`n_levels + 1` interface values), which covers the w-point item in P4.2.

### P0.17 UNESCO EOS pressure units (MEDIUM, latent) — FIXED: [PR #2](https://github.com/EmilLindfors/roms-rs/pull/2)
- [x] `EquationOfState::density`/`secant_bulk_modulus` (`equation_of_state.rs:144-160,281-307`) feed dbar into the bar-based formula. ρ(5 °C, 35, 1000 dbar) ≈ 1069.5 instead of ≈ 1032.3. Convert p_bar = p_dbar / 10.
- [x] Test against the UNESCO check values (ρ(35, 25 °C, 1000 bar) = 1062.53817; ρ(35, 5 °C, 0) = 1027.67547).

### P0.18 Chezy friction units (MEDIUM, 2D + 1D) — FIXED: [PR #2](https://github.com/EmilLindfors/roms-rs/pull/2)
- [x] `ChezyFriction2D` (`friction.rs:200-239`) and the 1D Chezy compute −C_D|u|u/h with dimensionless C_D. The momentum source is −C_D|u|u (currently 50× too weak in 50 m of water).
- [x] Test the magnitude, not just the sign.

### P0.19 Limiter plumbing (MAJOR, 2D) — FIXED: [PR #11](https://github.com/EmilLindfors/roms-rs/pull/11)
- [x] The parallel fused Kuzmin+positivity limiter (`limiters/swe_2d.rs:686-703`) lacks the serial dry-element branch (`:118-129`): dry cells keep momentum and negative means survive. Add the branch plus a serial-vs-parallel equivalence test with dry cells.
  - Done: serial and parallel limiters now run the same per-element kernels (`positivity_limit_element`, `kuzmin_limit_element`), in place (no `to_vec`/copy-back), and agree bitwise. `StandardLimiter2D` uses the fused kernel and the parallel versions under `parallel`.
- [x] The high-level `Simulation` API limits once per step, not per RK stage (`runner.rs:286-287`, `builder.rs:115-125`). That voids the Zhang–Shu guarantee; move limiting into the stage hook.
  - Done: `step_with_stage_hook` is now a `TimeIntegrator` method (`step` = no-op hook) and `Simulation` passes `post_process` as the hook.

### P0.20 NorKyst nesting time base and ingest (BLOCKER, nesting) — FIXED
- [x] `read_time` returns raw values and `find_bracket` clamps out-of-range times (`netcdf_io.rs:1226-1236`), so the forcing freezes at the first snapshot. Parse CF `units` and add an epoch/offset to `OceanNestingBC2D`. Error on out-of-range times.
  - Done: `OceanModelReader::time` is Unix seconds from CF `units`/`calendar` (missing units, non-Gregorian calendars and non-increasing times are errors). Time interpolation no longer clamps. `OceanNestingBC2D::with_epoch` (default: first snapshot), `check_time_coverage`, and a panic in `ghost_state` outside the file.
- [x] Remove `"h"` (bathymetry) from the SSH candidate names (`netcdf_io.rs:1068`).
- [x] `read_variable` tries i16 before f32 (`:1258`), so unpacked float fields are truncated to integers. Read by declared type.
  - Done: branches on `vartype()`, reads numerics as f64, CF unpacking with `_FillValue`/`missing_value` (netCDF default integer fills when absent).

### P0.21 Time step (CRITICAL for cost, 2D) — FIXED: [PR #11](https://github.com/EmilLindfors/roms-rs/pull/11)
- [x] `compute_dt_swe_2d` (`swe_2d.rs:582-602`, parallel `:620-650`) pairs the global minimum element size with the global maximum wave speed. Use min over elements of hₖ/λₖ with a direction-aware (anisotropic) length.
  - Done: per node, Δt = CFL/(2N+1)·4/(λ_r + λ_s) with λ_r = |u·∇r| + c|∇r|; unchanged on squares at rest. Serial and parallel share one kernel.
- [x] `froya_real_data.rs:163` uses CFL = 0.1 where ~0.5 is stable (5× cost). Also respect the DGSEM positivity bound for wet/dry runs (CFL ≤ 0.75/0.42/0.29/0.23 for N = 1–4 in current units).
  - Done: `positivity_cfl_swe_2d(N)`; with the directional dt the bound holds for any element shape. Froya now runs at it (5/12 at P2). Not yet enforced automatically by the steppers.

### P0.22 3D tracer flux leaks through coastal walls (MAJOR, 3D; P0.7 follow-up) — FIXED: [PR #12](https://github.com/EmilLindfors/roms-rs/pull/12)
- [x] The tracer kernel sets `u_ext = u_int` at physical boundaries (`advection_3d.rs:489-491`), so heat and salt cross the coastline while the mass flux is zero. Use `boundary::reflect_velocity` as for Ω and momentum.
  - Done: reflected velocity **and** the interior tracer/Hz at the ghost, so the Rusanov tracer flux is exactly zero like the volume flux (a ghost value from the BC would still leak through the dissipation term). Test: `horizontal_tracer_advection_closed_basin_conserves_inventory` (was −0.39 of the advective scale).
  - Consequence: `TracerBoundaryCondition3D` (`Hydrostatic3D::temp_bc/salt_bc`) is no longer consulted. It comes back with 3D open boundaries (P4.2).

### P0.11 Burn GPU RHS is physically wrong (BLOCKER, GPU) — CLOSED (deleted 2026-11-15)
Decision (2026-07-09): keep `src/solver/burn/` behind the `burn` feature for now (burn-cuda is the stated GPU target).
- [x] `src/solver/burn/rhs.rs` computes the HLL flux, then discards it. There is no surface term, so it is not solving the SWE. Closed 2026-11-15: the Burn prototype is deleted (`burn*` and `gpu` features, `src/solver/burn/`, `burn_ssp_rk3.rs`, `profile_gpu`, `MeshGPUData`); the GPU port starts again in Rust with cuda-oxide (`gpu.md`).
- [x] Implement the surface term plus a burn-vs-CPU equivalence test before trusting any GPU result. Superseded: the new kernels are gated bit for bit against the CPU (`gpu.md`).
- See P2.7: the review recommends custom fused f64 kernels over CSR faces rather than Burn tensor ops with fusion disabled (`review-2026-09-25.md` §5.2).

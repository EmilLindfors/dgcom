# 2D numerics: RHS kernel, well-balanced/positivity DGSEM, gating tests (P1.1, P1.2, P1.7)

## Priority 1: A Correct 2D Barotropic Tide Model (`review-2026-09-25.md` §1, §4, §7 Phase A)

### P1.1 Non-allocating RHS and one kernel — DONE for the `Simulation` path
- [x] `compute_rhs_into(&self, state, t, out: &mut S)` plus a workspace-owning integrator. Today the production stepper allocates ~33 full-field arrays per step (`review-2026-09-25.md` §5.3).
  - Done: `PhysicsModule::compute_rhs_into`, `TimeIntegrator::step_with_workspace` + `StageWorkspace` (the only method an integrator implements), buffer-reusing `clone_from` for the SoA solutions. A warm `Simulation` step allocates 0 B (`tests/allocation_test.rs`; was 1.3 MB per step at 24² P3).
- [x] Unify the serial (`swe_2d.rs:318-563`) and parallel (`:775-1110`) RHS into one per-element kernel. `iter` vs `par_iter` should be the only difference.
  - Done: `SWE2DRhsKernel`; bitwise serial = parallel; per-thread cached, padded scratch. Parallel RHS 1.2–1.7× faster with `_into` (false sharing between cached workspaces cost ~25 % until padded).
- [x] Route the `Simulation`/`SWEPhysics2D` path through the parallel kernel (`builder.rs:99`).
- [ ] Remaining allocations: the limiters allocate the cell-average and vertex-bound `Vec`s every stage (P2.3); (the 2D tracer's whole-mesh BR1 diffusion went with the 2D tracers, 2026-10-15); the legacy `ssp_rk3_swe_2d` stepper used by the examples still clones stages (P6: move the examples to `Simulation`); `DGSolution1D`/`DGSolution2D`/`Solution3D` use the allocating default `clone_from`, and `Simulation3D`/mode splitting do not use the stage workspace (P4.1).
- [ ] Use the new split to specialise hot paths without duplicating them again: e.g. a `const N` (order) generic kernel for the fixed-size matrix products, and skipping the `reference_to_physical` + `SourceContext2D` build for sources that need neither (P2.2).

### P1.2 Well-balanced, entropy-stable, positivity-preserving DGSEM
- [x] Flux-differencing volume term with the Wintermeyer et al. (2017) two-point flux and the g·hᵢ(DB)ᵢ term. It is exactly well-balanced for any nodal B (including face-discontinuous B), conservative, and fixes GLL aliasing.
  - Done ([PR #5](https://github.com/EmilLindfors/roms-rs/pull/5)): opt-in via `SWEFormulation2D::EntropyStable` (`EntropyConservative` for verification). Exact mass conservation, entropy conserved/dissipated to round-off, P2/P3 convergence 3.06/4.00 on a nonlinear manufactured solution with bathymetry.
  - Follow-up done: `examples/froya_real_data.rs` uses nodal B and `WetDry`.
- [x] Wet/dry and positivity: Zhang–Shu towards h ≥ 0 (not h_min), Kurganov–Petrova velocity desingularization below `h_dry` (1 mm). Done for the collocated (`Standard`) formulation (`solver/algorithms/wetting_drying.rs`, `limiters/swe_2d.rs`). Gates in `tests/wet_dry_2d_test.rs`: Ritter dry dam break (L1 1.5 %, first order), Thacker's paraboloid (L1 7.4 % after one period, was 9.2 %), shoreline lake at rest (|u| < 0.06 m/s where h > 1 cm, was up to the 20 m/s cap). All conserve mass to round-off with h ≥ 0.
- [x] **Shoreline well-balancing and wet/dry for the split form: `SWEFormulation2D::WetDry`.** HLL on Audusse-reconstructed states at faces, flux differencing in wet elements, and subcell finite volumes with the same flux in partially dry elements.
  - A shoreline lake at rest is kept to round-off (P1–P4, smooth/rough beds; the P1.7 gate `lake_at_rest_with_shoreline_is_exact` passes).
  - Fully wet it keeps N+1 (P2 3.06, P3 4.02) and dissipates energy.
  - `Standard` is still not balanced in partially dry elements: on a 2 % beach it settles into a stationary circulation of ≈ 5 cm/s where h > 1 cm, and 1–4 m/s in the film.
- [x] **`WetDry` accuracy on moving shorelines.** Every element the shoreline crosses uses subcell FV, which was first order: 4.8 % on Thacker (P2, 40²) against 1.3 % for `Standard`.
  - Done: a second-order subcell reconstruction of h, η, u, v. It uses a monotonized-central limiter on the non-uniform GLL subcells, end-subcell slopes from the adjacent element, and the Audusse second-order bed term.
    - It stays well-balanced at shorelines and keeps h ≥ 0 at faces.
    - With subcells forced everywhere it converges at second order (P1–P3 ≈ 1.9).
    - Thacker at 40²: P1 3.2 %, P2 1.2 %, P3 0.9 %, P4 0.7 %, against `Standard` 10 / 1.3 / 1.8 / 2.3 %.
  - Also fixed: HLL returns zero, pressure included, when both reconstructed depths are below `h_min`; the hydrostatic correction now supplies all of ½g h².
  - [ ] End subcells on physical boundaries (no neighbour element) stay first order. Take the outer slope neighbour from the boundary condition (mirror the interior node for walls). (Note 2026-09-28: a mirrored node makes h, η and the tangential velocity even across the wall, so the MC limiter gives them zero slope and only the normal velocity gains a slope; η stays first order, and with Coriolis ∂η/∂n = −f u_t/g ≠ 0 at a wall. One-sided slopes (node 0 to node 1, bounded because the subcell face lies between them) would be second order but break the balance argument at a dry end node, whose face depth must stay zero: use them only where the end node and its neighbour are wet.)
  - [ ] `outer_nodes` scales the neighbour's node distance by the affine element heights normal to the face. Revisit with per-node isoparametric factors (P1.3).
  - [ ] Any node below `h_dry` switches the whole element to subcells. Convex DG/FV blending (Hennemann–Gassner) could keep DG in the wet part, but it needs a DG part that is balanced with dry nodes present. Lower priority now that the subcells are second order.
  - [x] `WetDry` is now the default for wet/dry runs in `SWEPhysics2DBuilder`. `Standard` stays available via `with_formulation`, and `SWE2DRhsConfig` still defaults to it.
  - [ ] Retire the `Standard` wet/dry path (hydrostatic reconstruction + `BathymetrySource2D` in partially dry elements) together with the cell-average workarounds (below), once the legacy `ssp_rk3_swe_2d` users (Frøya, P6) are on `Simulation`.
- [ ] Guide the choice of `h_dry`. It should be about 1e-4 of the characteristic depth: the 1 mm default suits field scale, but it dominates the error of the 0.1 m Thacker case (7.4 % vs 4.5 %). Consider deriving it from the bathymetry range, or making it relative.
- [ ] The Chen–Noelle (2017) reconstruction instead of Audusse. It is more accurate for bed steps larger than the depth (thin films on steep subcell beds, e.g. fjord walls); it changes only B* and h* (`b* = min(max B, min η)`, `h* = min(η − b*, h)`).
- [ ] Dry fronts lag: in the Ritter test the h = 1 mm front is at 71.5 m instead of 79.8 m (P2, 1 m elements; the same on `main`). It moves only to 73 m at 0.5 m resolution, or 75 m with h_dry = 1e-5. The thin-layer relaxation plays no role. Suspects: strict Kuzmin at the front, and zeroing the momentum of elements with mean < h_dry. Revisit with η-based limiting.
- [ ] Positivity alone does not control a dry front: without a slope limiter, P1/P2 dam breaks shed a thin film that runs ahead at the 20 m/s velocity cap. Partially dry cells need limiting (next item).
- [ ] Limit η (and velocities or characteristic variables), not h, with a troubled-cell indicator (TVB-M/KXRCF). Limit h only in partially dry cells (Vater et al. 2019).
  - Today, strict Kuzmin on h with cell-average B collapses sloping elements to P0 on every flood/ebb half-cycle.
- [x] The positivity CFL is enforced: `PhysicsModule::max_cfl`, applied by `Simulation`; `SWEPhysics2D` reports `positivity_cfl_swe_2d(N)` when it has wetting/drying. Roe with wetting/drying prints a warning at build time.
- [x] Negative-mean elements are counted, not hidden: the positivity limiters and wet/dry correction return the count; `SWEPhysics2D::negative_depth_clips`.
- [x] Point-implicit friction in every RK stage (`TimeIntegrator::step_with_relaxation`, `BottomFriction2D`, `SWEPhysics2DBuilder::with_implicit_friction`). It keeps friction balances exact for any dt, is L-stable, and is first order in the friction term (13–32 % above the exact decay at dt·Λ ≈ 3; explicit flips the sign there).
  - [x] `SpatiallyVaryingManning2D` did not implement `BottomFriction2D`: the rate needs the node position. Done 2026-09-28: it holds g·n² per node (`new` from n(x, y) at the nodes, `from_nodal` from values), `damping_rate` takes the node index, and the damping checks the field against the mesh. Its explicit source goes through `add_element` only (`evaluate` has no node index and panics).
    - [ ] A roughness map from data (sediment type or a calibrated n per region, e.g. lower n over deep rock, higher over tidal flats), projected onto the nodes like the bed (`Bathymetry2D::project`). Frøya runs n = 0.025 everywhere, and friction is one of the phase suspects at Mausund (P3.1).
  - [ ] A second-order IMEX treatment (e.g. an SSP IMEX-RK) if the friction term's first-order error matters in shallow tidal flats.
- [x] The dt-dependent wet/dry momentum damping is gone. It is replaced by the point-implicit thin-layer relaxation r(h) = (h_dry/h − 1)²/τ, which is zero at h ≥ h_dry.
- [x] HLL is the default flux for wet/dry runs in `SWEPhysics2DBuilder`. `SWE2DRhsConfig::new` still defaults to Roe (low-level API).
- [ ] Retire the cell-average/`linearize` workarounds: `WetDry` now covers wet/dry with nodal B.
- [x] `SWEPhysics2DBuilder` footgun (found 2026-09-30): `with_bathymetry` without wet/dry defaults to `Standard`, which gets the bed-slope force only from an explicit `BathymetrySource2D`. Without it a sloping bed pushes nothing: a lake at rest over a linear 20→10 m slope reaches η = +10 m at the shallow end within an hour (P2, 30 km channel). Default to `EntropyStable` when a bed is set and no source carries the slope, or refuse to build.
  - Done 2026-09-30: both. Without wet/dry, a bed that is not flat now defaults to `EntropyStable` unless a source carries the slope; an explicit `Standard` over such a bed without `BathymetrySource2D` panics at `build`. The channel stays at rest to round-off (`a_sloping_bed_is_balanced_by_default`). No example changes (the only affected one, `minimal_3d`, has a flat bed). One test relied on the trap: `several_levels_conserve_mass` (LTS) got its third level from the spurious flow; with the slope applied (either formulation) it has two.
  - [ ] The low-level `SWE2DRhsConfig::new` still defaults to `Standard` with no check; a `with_bathymetry` there could warn the same way.
- [x] Move `examples/froya_real_data.rs` off the legacy `SWE2DTimeConfig` (`H_MIN` = 5 m, land as a 5 m film, cell-averaged B). It now uses `Simulation` + `WetDry` (nodal B) + `with_wet_dry` + `with_implicit_friction` on a water-only mesh, with all four sides open.
- [x] `SWEPhysics2D::post_process` built a `LimiterContext2D` every stage, whose `new` computed `mesh.h_min()` serially (a sqrt per edge). No limiter read it: the field is gone (2026-09-26).

### P1.7 Gating tests (write first; most fail today)
- [x] Lake-at-rest with steep nodal B (e.g. 30→400 m), p = 1–4, serial and parallel: residual < 1e-10. Done 2026-09-28 (`tests/swe_2d_test.rs::test_lake_at_rest`: a slope from 30 to 400 m plus bumps, walls; `EntropyStable` and `WetDry` ≤ 1e-11 m²/s², serial = parallel bitwise; `Standard` with reconstruction 2.5–115 as the negative control).
- [x] Lake-at-rest with face-discontinuous B, and with wet/dry shorelines.
  - Face-discontinuous B: split forms, `test_lake_at_rest_any_nodal_bathymetry` (cell-averaged and rough beds, P1–P4).
  - Shorelines: `WetDry`, `test_wet_dry_lake_at_rest_with_shorelines` (RHS) and `tests/wet_dry_2d_test.rs::lake_at_rest_with_shoreline_is_exact` (100 s runs). `Standard` is bounded by `…_stays_near_rest`.
- [x] Mass conservation with discontinuous B and slope-correlated flow: split forms `test_mass_conservation`, `WetDry` with dry regions `test_wet_dry_mass_conservation_with_dry_regions`; `Standard` covered by the P0.13 tests.
- [x] Thacker parabolic bowl (dynamic wet/dry), planar SWASHES case: `thacker_planar_oscillation_converges_{standard,wet_dry}`. Both converge at ≈ 2nd order. Also the Ritter dry dam break, for both formulations.
- [x] Outgoing-pulse reflection < 1 %; delivered tidal amplitude: `tests/open_boundary_flather_test.rs` for all three fluxes and both split forms, plus oblique incidence and elevation-only forcing (P1.4).
- [ ] Nonlinear SWE convergence with bathymetry on curved meshes. Assert N+1: current thresholds 1.5/2.5 are below it (`tests/convergence_test.rs`).
- [x] Fix vacuous tests (2026-09-28):
  - `test_geostrophic_balance` passed with Coriolis removed: it put a linear η on a periodic mesh, whose 0.1 m seam swamped the 1e-3 Coriolis term. Now a periodic geostrophic jet (an exact nonlinear steady state): residual 3e-6 against the Coriolis term 6e-3, converging at 2.9 (P3); 6 h drift 1.1e-4 of the jet speed. Both checks have a no-Coriolis negative control (residual = the Coriolis term, drift 5e-2).
  - `tests/swe_2d_test.rs::test_lake_at_rest` used a flat bottom: now the steep-bed gate above.
  - The well-balanced tests use only linear B at p ≥ 2: they test the collocated `Standard` form, which is balanced only for such beds. Arbitrary nodal beds are covered for the split forms (`test_lake_at_rest_any_nodal_bathymetry`, the steep-bed gate), with `Standard` as the negative control.
  - "Multi-day stability" ran 316 s: renamed `test_long_run_mass_conservation`. A real one is `test_multi_day_tidal_run`: 3 days of M2 through an open boundary into a shoaling basin with Coriolis and implicit friction on `Simulation`; the last two cycles repeat to 5e-7 m (volume) and 1.4e-6 m (head η), and the head amplitude is the forcing's to 0.4 %. 13 s in debug.
  - `test_parallel_matches_serial` excluded bathymetry, boundaries and limiters: bathymetry and walls were already in (cases 2–3), limiters and wet/dry have their own (`test_parallel_limiters_match_serial`, `test_parallel_correction_matches_serial`); an open tidal boundary with all three formulations is added.
  - Also found: `test_standing_wave_period` timed the peaks of η at one node, 0.9 % off, under a 5 % tolerance, and skipped the check when it found fewer than two peaks. It now times the zero crossings of the cos(kx) mode amplitude: 3e-5, asserted to 1e-4.
  - [ ] `tests/swe_2d_test.rs` has three overlapping long-run conservation tests on the legacy stepper (`test_long_term_stability`, `test_long_time_conservation`, `test_long_run_mass_conservation`); merge them into one on `Simulation` when the examples and tests move off the legacy steppers (P6).

### Horizontal viscosity at the shoreline (found and fixed 2026-10-25)

The F.1 ν sweep at Kattholmen (`froya_real_data site=data/sites/14042.txt nu=10`, `notes/farm.md`) put water on land: within 2 h, η = +4.7 m on films of 2–20 cm at (−4.8, −0.9) km, running at 0.2–0.5 m/s. Element 8373 there is a cliff shore whose nodes span −0.55 to +4.64 m; at ν = 0.1 its land nodes stay dry and still (the wet node 0.02 m/s against 0.61 at ν = 10). At 5 h a 1.47 m/s current stood in 3 m of water 120 m from the farm's centre.

Cause: the BR1 viscous term (`solver/rhs/swe_2d_viscosity.rs`) ran in the shoreline (subcell) elements. Dry nodes entered the gradient with u = 0 and carried no stress (νh = 0), but the divergence at a dry node differentiates its wet neighbours' stress, so dry nodes got momentum, and the subcell update moved water with it.

Fix: viscosity on the water only. An element with a node shallower than the subcell depth (`WetDry`) or at or below `HorizontalViscosity2D::h_min` gets no viscous term; a face from a wet element toward it copies the wet side's velocity in the gradient (zero gradient) and passes no viscous flux, the pair that keeps BR1 dissipative (`diffusion_2d::br1_diffusion_element`, `R = I`), and conserves momentum. The BR1 helpers take a `coupled` neighbour predicate; the 3D viscosity (thin columns at rest, `viscosity_3d.rs`) passes `|_| true`. Gate `viscosity_keeps_off_the_shore` (a beach at ν = 10: no change in shoreline or dry elements, dissipative, no mass); it fails on the old kernel. The earlier Kattholmen run (ν = 1, 2026-10-24) used the old kernel; the sweep reruns it.

The shoreline elements now have no lateral momentum diffusion at all: their first-order subcells dissipate more than any ν in use.

### Walled wet nodes keep their velocity (found 2026-10-11; the wall fixed 2026-10-15)

On the Frøya coastline day in 3D (`froya_real_data mesh=data/froya_coast.msh … tide3d=1 slopes3d=on cfl_3d=1`), element 7790 held a depth-mean ū of 3.3–3.8 m/s (4.2 m/s at the surface) from 5 to 10 h, through the whole ebb. It set the 3D step, at ≈ 1850 s wall per model hour against ≈ 820 without it.
- **Where.** Node P at (−1.395, −9.278) km, bed −25.5 m, is the middle node of the edge shared by elements 7790 and 7791.
  - The edge's corners are land (+2.72, +4.76 m).
  - The two centres are a 0.44 m sill (7790) and land (+2.79 m, 7791).
  - Below η ≈ −0.44 m, P is a one-node pond, connected to the sound only diagonally.
- **What happens.** P's two copies split by ≈ 0.6 m in η (7790's at +0.24, 7791's at −0.37, while the sound is at −0.88).
  - The velocity points almost exactly along the edge (5.6° off), into the dry corner.
  - η at P stays put for 5 h, so no water moves.
  - The jet dies at 10–11 h, when the flood reconnects P.
- **It is the 2D kernel's, not the 3D coupling.** The 2D model alone on the same bed (`slopes3d=on debug_3d=thin=1`, so the same 1 m smoothing threshold) has ū 1.94 m/s at P at 5 h. That run was stopped there (`output/froya_day/2d_thin1`).
  - The original 2D day (thin depth 0.1 m) has no jet. Its smoothing deepened the 7790 sill from 0.44 to 17.6 m, so P stayed connected.
- **Mechanism (suspected).** At a wet node beside a higher dry node, `hydrostatic_hll` makes h* = 0 on the wet side. The interface then carries only the pressure correction ½g h², whatever the node's velocity, so a walled node's velocity is never reflected or stopped; only bottom friction acts on it.
  - At 25 m depth, friction balances 3.7 m/s with a forcing of only ≈ 1.4e-3 m/s².
  - Two candidate sources of that forcing are still to be identified: an imbalance of the walled interfaces' metrics on a non-affine element, and the copies' η jump through the face.
- **Most ponds stay still.** Of the 23 ponds deeper than 5 m at η = −0.6 m, only P (and, more weakly, a 2.2 m pond at (−2.15, 6.61) km with 1.15 m/s, and one at (1.35, −0.89) km with 0.6 m/s) carries a current.
- **Fixed (2026-10-15): a wall where no water passes** (`wall_ram_pressure` in `solver/rhs/swe_2d_split_form.rs`, scalar and SIMD). The suspected mechanism was right: where both reconstructed depths are dry, Audusse's flux gives a wet side `½g h²` whatever its velocity.
  - Gate `a_walled_pond_stops_its_current` (`tests/wet_dry_2d_test.rs`): one 25 m node walled in by +2 m land, 1 m/s, no friction. Before, a pond at an element's centre or at a mid-edge node kept exactly 1.000 m/s for 600 s (a vertex pond already stopped); now all three are at round-off.
  - Brufau, Vázquez-Cendón & García-Navarro (2002): such an interface is a solid wall, and the HLL flux against the mirror state adds `h uₙ(2uₙ + c)` to the normal momentum flux (`uₙ` towards the wall). The sign matters per side: both fluxes are along m, so the wall lies along +m from side a and −m from side b; with one sign the two walls of a subcell cancelled and a vertex pond accelerated to 11 m/s.
  - The full mirror cost Thacker accuracy (WetDry P2 L1 depth error 5.67 % / 1.75 % at 16² / 32² before, 6.01 % / 1.98 % with it). Two restrictions take it back to 5.68 % / 1.71 %, with the ponds still stopped:
    - only against the flow (`uₙ > 0`): flow away from the wall would get the mirror's rarefaction `½g h² + c h uₙ`, holding a receding shoreline back (alone: 5.89 % / 1.98 %);
    - only a sill the flow cannot climb, weighted `w = max(0, 1 − (uₙ²/2g)/δ)` with `δ = B_face − η` the sill above the water: an advancing front lies within the velocity head of its dry neighbour (Thacker: 2.5 cm), a pond metres below its walls. It also keeps the flux continuous where `h*` crosses `h_min`.
  - Lake at rest, the dam break and the shoreline lake are unchanged; SIMD and scalar bit for bit (`test_batched_kernels_are_bit_for_bit`, whose rough shoreline has wet nodes moving against dry ones).
  - **Frøya, 2D on the 3D bed** (`slopes3d=on cliffs3d=off debug_3d=thin=1`, as `output/froya_day/2d_thin1`; `output/froya_day/2d_thin1_wall`): P carries 0.38 m/s at 5–7 h against 1.94 m/s at 5 h; the fastest current over 3 m deep is a real one (0.66 m/s at 37 m). The coastline at rest stays at 2.3e-11 m/s.
- **Still open: ponds that hold a current the wall does not stop.**
  - **P's copies stay apart.** In 7790 P stands at η = +0.16…+0.18 m (rising), in 7791 at −0.43, with the sound at −0.8. Its 0.38 m/s is almost normal to the shared face (−0.37 m/s into 7790): a steady exchange in which the copies' HLL jump term (−½λ[[h]], 0.6 m) balances the advective flux. 7790's copy lies 0.62 m above its interior sill node (bed −0.44 m, h 0.00–0.01), which should drain it within minutes and does not. Log the face and subcell mass fluxes at P from a snapshot state.
  - **A 2.2 m pond at (−2.153, 6.606) km** (elements 7586/9322, a mid-edge node, its copies within 1 cm) keeps 1.23 m/s at 5–7 h, unchanged by the wall: it flows south over a sill only 6 cm under its water (node 8 of 7586, bed −0.44 m), so it is not walled. A pond at (1.347, −0.887) km (8309–8312) went from 0.35 to 0.68 m/s. Whether these drain a perched flat for hours (physical) or circulate (not) is still to be seen.
  - The bed side (`notes/geometry-mesh.md` "Tidal ponds"): since 2026-10-15 `ponds3d` fills the ponds of up to one element's nodes over the tidal range, P and the 2.2 m pond among them. A 0.2 m film on +0.27 m land at (13.228, −4.318) km still runs at 0.8 m/s at the same η from 5 to 8 h: the kernel's part is open (TODO P1.2).
- **Tools.** The temporary snapshot tools `examples/tmp_snap_elem.rs` (one element over time, and the fastest deep nodes) and `examples/tmp_snap_ponds.rs` (ponds at a level, with their speeds) are uncommitted in the working tree.


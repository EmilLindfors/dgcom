# TODO

Open items only, one line each. The context (measurements, decisions, what was tried, finished work) is in [`notes/`](notes/README.md): each section heading links its notes file, and searching that file for the item's ID finds the details. When an item is done, delete its line here and record the outcome in the notes file (and `CHANGELOG.md` if notable).

Direction (from the 2026-09-25 review, `notes/review-2026-09-25.md`): the salmon-farm application (currents, waves and particles around cages inside a NorKyst-driven coastal domain) on a 2D model that is correct, cheap and validated, and a 3D model rebuilt on the ROMS recipe. "Faster than ROMS" is to be proven by P3.2, not assumed.

## Next up

1. **F.1** The wake before the runs diverge at Kattholmen: against the 10 % control's floor (2026-10-27) the cages' slowdown and the westward wake within 200 m stand, while half the wake area and most of the residual change are noise. See Salmon farm.

## Validation (P3) — [notes/validation.md](notes/validation.md)

- [ ] **P3.1** Remove M2's +3 % at Mausund (2.6 cm |ΔZ|, the largest error, NorKyst's own) with `froya_real_data gauge_gains=M2,…` so N2 follows; run it with the overtide work, since M4 scales with M2².
- [ ] **P3.1** Find why the overtides are too weak: on the 3D bed (`slopes3d=on debug_3d=thin=1`) the model's M4 is a uniform 0.36 cm within 10 km of Mausund, 0.12× the gauge (0.7 cm, 0.34×, on the unsmoothed 2D bed, whose shoals and pinnacles make more; NorKyst 0.17 cm). Compare M4 at other gauges (Kristiansund, Heimsjø, Trondheim) with NorKyst's to tell a local harbour effect from a regional one; `notes/validation.md` 2026-10-16.
- [ ] **P3.1** Measured currents at a Frøya farm (NS 9415 site surveys, IMR/NIVA moorings): the only way to settle the island lee and which current to trust at farm sites; NorKyst cannot (its bed is 50 m at Kattholmen, the real 25 m).
- [ ] **P3.1 / P3.3** The month-long real-domain run (`froya_real_data … hours=720`, or 30+ days on the coastline mesh): resolves N2/Q1, checks long-run stability, mass and no residual currents at rest.
- [ ] **P3.1** The "island lee" at the NorKyst point and Kattholmen: on the 3D bed with the whole boundary open the depth-mean M2 is 0.18–0.19 m/s in 2D and 0.12 in 3D against NorKyst's 0.255, and ≈ 40° early (G 26° against 65°; our bed 22 m, NorKyst's 49 m). Map-wide, both models are 0.56–0.61 of NorKyst where NorKyst is under 50 m deep and the beds agree, 0.93–1.03 deeper. Settle which is right with observed currents (the measured-currents item below).
- [ ] **P3.1** Mausund's centred RMSE over hours 3–24 of 2025-06-15 is 4.5 cm with the whole boundary open (2D and 3D alike), 1.9–2.1 cm in the runs whose parent file walled 120 of its 223 faces (NorKyst itself 7.5 cm): the closed box was right at the gauge for the wrong reason. The open runs' one-day M2 is 1.11×, 14° early: look at the boundary's M2 phase (NorKyst's 10–20 min lag, P1.5) before tuning anything at Mausund.
- [ ] **P3.1** Re-run P3 at 1 km against P2 at 500 m with the relaxed positivity CFL (dt ratio now ≈ 1.3–1.4) and stations interpolated at the gauge.
- [ ] **P3.1** The 500 m, 15-day harmonic run (≈ 2.5 h) for the constituents at resolution.
- [ ] **P3.1** Fix the approximate `norwegian_stations` positions from Kartverket's station list (`scripts/norkyst_tide_gauges.csv` has 17 permanent gauges; Heimsjø is at 63.425, 9.102).
- [ ] **P3.1** Send the NorKyst-800 report to MET (the draft message in `notes/norkyst-issues.md`) and record the answer there: N2 was 0.19× at all 17 gauges until 2026-02 and is right since 2026-03 (the new tide file); open are K2 0.8× (no nodal factor), P1 and Q1 1.7×, the missing minor semidiurnals, the datum (−0.28 m), the missing inverse barometer, ubar/vbar.
- [ ] **P3.1** Boundary atlases from NorKyst after 2026-03 (its N2 right) with K2 and P1 at the equilibrium ratios instead of `gauge_ratios`, so no gauge enters the forcing for N2, K2, P1 (Q1, 1.7×, still needs one); `notes/norkyst-issues.md` "What this means for us".
- [ ] **P3.1** 2N2, μ2, ν2, L2 in the constituent catalogue (`tides::astronomy`, M2's nodal factors): NorKyst lacks them (2–7 cm of |ΔZ| at the gauges), so atlases could add them from N2/M2 at gauge or equilibrium ratios; and a 30-day fit could infer ν2 from N2, which removes the ±20 % beat of monthly N2 (`notes/norkyst-issues.md` §1).
- [ ] **P3.1** Run the model against NorKyst-800 barotropic tides for a test period.
- [ ] **P3.1** Document skill scores (RMSE, bias, correlation) per station.
- [ ] **P3.4** Score Bodø's gauge again on a day after 2026-03, when NorKyst's N2 is right (same frozen config, `nest_tides=raw`): on 2025-06-15 the raw score failed (11.4 cm centred against ≤ 5) and the gauge-corrected boundary tide alone brought it to 3.4 cm, so the failure is the parent's N2; a post-fix day tests the model with no gauge in the forcing.
- [ ] **P3.4** A second out-of-sample domain in a fjord (Saltstraumen's tidal jet, or a deep sill fjord) where the shallow-water drag gap (P1.3, 3D/2D 0.79 in 3–50 m on both Frøya and Bodø) and stratification matter more than at Bodø's open skerries.
- [ ] **P3.2** The Pareto benchmark on the same hardware: M2/S2/K1/O1 gauge error against core-hours, ROMS 2D at 800/400/200 m against DG P1–P4.
- [ ] **P3.1** Low priority (≤ 4 mm left): whether the interior amplifies the diurnals (shelf waves).
- [ ] **P3.1** On hold: η-based limiting; revisit only if a real run shows oscillations.

## Salmon farm and cage drag (F.1) — [notes/farm.md](notes/farm.md)

- [ ] **F.1** The wake before the runs diverge: start the cages from the open run's state at several phases of the tide and compare over the first ≈ 6 h, where the departure is the farm's (0.11 within 500 m at 6 h, the 10 % control's 0.02). Needs 2D restarts (P5.3) in `froya_real_data`.
- [ ] **F.1** A spread for the wake's floor: two 10 % controls (`farm_control=0.1`, `solidity=0.06`) differ by 10–20 % in wake area and 2× in residual change, so test the cage effect against several controls (different F or start times), not one.
- [ ] **F.1** The Kattholmen case in 3D (`froya_real_data levels=…` with `site=`, refused today): the wake under the nets, the drag (≈ 20 % above 2D in `farm_3d`) and larvae in the surface layer.
- [ ] **F.1** Rerun Kattholmen when the island lee is settled (P3.1): at ν = 0.1 (with or without Smagorinsky) the farm's current is 0.08–0.09 m/s mean, 0.26–0.32 at most (0.04 at ν = 1), and the NorKyst point's M2 0.08–0.13 m/s against NorKyst's 0.25, 60–90° off in phase at every ν.
- [ ] **F.1** Net-made wake TKE (`P = Σ λ_l |u_l|²`) as a source in GLS's `k` and `ψ` per caged layer (Beudin et al. 2017, `c_ψ4`).
- [ ] **F.1** Compare the cage wake with a farm survey or flume data before trusting the default a = 4/(πR) (Løland's r = 1 − 0.46 C_d is left to the resolved flow).
- [ ] **F.1** Check a site's guessed cage count and size (`FarmSite::cage_layout`, Kattholmen 2 × 4) against an aerial photo or the operator; Kattholmen's two southern cages stand on a 9–10 m bed under 20 m nets (the 50 m topobathy), so check the bed at the frame too.
- [ ] **F.1** Angle dependence C_d(θ) and the lift term, for square cages in oblique flow.
- [ ] **F.1** Net deformation in strong currents (the net lifts, d_net shrinks).
- [ ] **F.1** Low priority: a sparse layer-drag buffer over `CageDrag2D::nodes` (`ModeSplitIntegrator` `layer_drag_rate` is dense and zeroed every step by `Hydrostatic3D::layer_drag_into`).
- [ ] **F.1** Low priority: drag on the feed raft and mooring lines (read by `io::farm_site`, not modelled).
- [ ] **F.1** Not worth it at farm steps: make the explicit `G` shear part of the 3D cage (and bottom) drag consistent with the implicit pass; at λΔt ≥ 10 it reverses stopped flow by < 1 % (keep λΔt ≲ 5).

## Particles (F.2) — [notes/particles.md](notes/particles.md)

- [ ] **F.2** A physical basis for the horizontal K from a site's drifter or dye spreading (connectivity shares move by 9–73 over 0.01–1 m²/s).
- [ ] **F.2** Check connectivity at the pens converges with mesh refinement under Smagorinsky K.
- [ ] **F.2** Log K along the larvae's paths, to explain why `kh=smag` keeps more larvae at home (82.5 %) than a constant 0.01 (73.3 %).
- [ ] **F.2** Draw release points per seed and per batch, so the seed spread measures release uncertainty, not only walk noise.
- [ ] **F.2** Log the larvae's largest displacement in `farm_3d` before longer runs (the 6 km periodic channel is shorter than the 7.1 km tidal excursion).
- [ ] **F.2** Copepodid infectivity(age, T) (Skern-Mauritzen et al. 2020) next to `SalmonLice::survival`, as the `ConnectivityRecorder::record` weight.
- [ ] **F.2** Light for lice from `AtmosphereReader` shortwave or cloud cover, salinity/CDOM-linked attenuation (Aksnes et al. 2009), a per-position sun for wide domains.
- [ ] **F.2** Keep Ω at the w-points in `Solution3D`, so the tracker's dσ/dt matches continuity exactly (a layer-centre average today).
- [ ] **F.2** Cubic time interpolation between snapshots for long output intervals.
- [ ] **F.2** Resuspension of settled particles above a critical bed stress (3D stranding also uses column depth only).
- [ ] **F.2** Locate lice cue crossings within the step, or keep √(2KΔt) ≪ K/w_s near haloclines (the light gate's χ² is 13/23/41 at Δt = 5/10/20 s).
- [ ] **F.2** Second-order contact exposure with the exact passage time (segment–circle intersection) if UΔt cannot be kept small against the cage.
- [ ] **F.2** Polygon contact zones (point-in-polygon behind `ContactZone::covers`) for site polygons and whole farms.
- [ ] **F.2** A bucket grid over contact zones in `ConnectivityRecorder::record` (O(particles × zones)) for region-wide registers.
- [ ] **F.2** Parallel connectivity recording per particle (`FarmParticles::steps`); matters at 10⁵+ particles.
- [ ] **F.2** Low priority: split `Solution3DVelocity` sampling by what the caller needs (RK stages sample K they discard; 0.6 s of 163 s).
- [ ] **F.2** Curved isoparametric elements in particle walking, `inverse_bilinear` and `PointLocator2D`'s Newton, once P1.3 has them.
- [ ] **F.2** Only if needed: pass `&Particle3D` to behaviours for per-particle traits; per-tidal-phase exposure bins (needs contact times beyond `Contact::first`).

## Waves (F.4) — [notes/waves.md](notes/waves.md)

#### Sources
- [ ] **F.4** Sweep `GrowthLimiter::Rate`'s C (3e-7 and 1e-6 against today's 5e-7) at the lee and Frohavet points and on the fetch/duration gates: is the lee's −9 % the cap or real?
- [ ] **F.4** Retune whitecapping (Komen δ=1: E* 0.60× KC92, 1.5× Pierson–Moskowitz at 1000 h): try van der Westhuysen 2007 with Yan input or ST6, then tighten the gates.
- [ ] **F.4** Triads (LTA, Eldeberky 1996) for the surf zone.
- [ ] **F.4** `c_σ` from a changing depth (∂σ/∂d ∂d/∂t): tides shift frequencies over flats.
- [ ] **F.4** Diffraction behind islands (phase-decoupled, Holthuijsen et al. 2003) for farms sheltered by skerries.
- [ ] **F.4** With substeps the boundary spectra and wind are held over the outer step; check against hourly forcing before ~10 min steps.
- [ ] **F.4** `Quadruplets::source_and_diagonal` leaves out a component's share in its own gather on grids with frequency ratio ≥ 1+λ: assert or handle it.
- [ ] **F.4** Drop `SourceIntegration::Midpoint` and `with_time_integrator` unless the midpoint helps with `Implicit`.
- [ ] **F.4** Low priority (not farm water): under 1 m depth the 56 s step has +8 % H_s bias; try moving the DIA's `quadruplet_scale` into the substeps.
- [ ] **F.4** Low priority: SWAN's tail beyond σ_max in the whitecapping means (`Means::of`, `src/waves/sources.rs`).

#### Depth limit
- [ ] **F.4** Re-run the storm with `wave_depth_limit=0` on current code, for like-for-like and the limit's uncontended cost.
- [ ] **F.4** Read the coupled storm's limiter counts by depth band (`depth_limit_bands`); if it binds in metres of water, count the removed variance as breaking in `dissipation()`.
- [ ] **F.4** `the_depth_limit_leaves_a_resolved_surf_zone_to_breaking` takes 10.8 s: trim it or move it to the slow tier (`.config/nextest.toml`).
- [ ] **F.4** Check the depth limit's γ against SWAN's very-shallow limiter before trusting the surf zone or a coarse grid's coastal nodes.

#### Validation
- [ ] **F.4** When a farm's own wind sensor (or a buoy in its fjord) is at hand: blend it into `met=` as Ålesund's attribution test did (F's SI 0.44 → 0.25), as a `froya_real_data` option rather than a scratch file. MEPS and MET Nordic carry along-fjord jets at half strength, and a mass-consistent downscaling does not recover them (2026-10-23).
- [ ] **F.4** Ålesund's C: T_p 14 s against 9.1 on the 500 m grid at P1 and P2 alike, so not the grid. Candidates: the boundary's swell (≈ 1 s long at D) and diffraction. Also its first-day local sea from 190° (0.5–0.96 m against 0.3–0.5): check the buoy's wind as at F (THREDDS failed on 2026-10-22).
- [ ] **F.4** A 500 m P2 wave grid (`wave_mesh=120,90,2`): are Frohavet (17 % low) and the lee (19 % low) the 1 km grid's or WAM's error?
- [ ] **F.4** A reflection coefficient for walls (cliffs, quays); waves dying on the coast drive spurious coastal currents.
- [ ] **F.4** Check which wave nodes along 63.75°N (8.22–8.28°E skerries) are dry at mean and low water (the reef point loses 7.5 cm more with absorbing land under SWAN's sources).
- [ ] **F.4** Nest NorKyst for the storm days (`norkyst=`) for the remote surge (+0.29 m observed against +0.08 m), then compare the wave setup at Mausund with the gauge's residual.

#### Coupling
- [ ] **F.4** H_s, T_p and direction at the stations (gauges, NorKyst's point, farms) in the station files.
- [ ] **F.4** Measure the coupling lag's error at Δ = 10 / 5 / 2.5 min; if it matters, extrapolate (`2q(t) − q(t−Δ)`) or a second pass.
- [ ] **F.4** Make `WaveForceForm::Dissipation` the `CoupledWaves2D` default once a surf-zone case with a measured set-down is checked.
- [ ] **F.4** A wave-dependent breaking flux into GLS: ρg × (whitecapping + breaking) per column from `WaveModel2D::dissipation` (COAWST `TKE_WAVEDISS`).
- [ ] **F.4** 3D vertical structure (a): the Stokes transport in continuity (Eulerian return flow `D ū = −M_s`); changes the 2D model too.
- [ ] **F.4** 3D vertical structure (b): the breaking force near the surface (Uchiyama 2010 `B(z)` or a roller), gated against the constant-ν undertow (Svendsen 1984, Stive & Wind 1986).
- [ ] **F.4** 3D vertical structure (c): the vortex force and Bernoulli head (McWilliams et al. 2004; ROMS `WEC_VF`).
- [ ] **F.4** Waves on the 3D bed: `BottomDrag3D` with Soulsby's enhancement or Grant–Madsen roughness (ROMS `SSW_BBL`).
- [ ] **F.4** The maximum bed stress τ_max with the wave–current angle, for sediment and mooring loads.
- [ ] **F.4** `farm_3d` with `wind=`: run the wave model on the farm mesh, couple the Stokes drift each output interval and use `surface_roughness` instead of the constant `waves=`.
- [ ] **F.4** Lateral mixing for the longshore current, and a runup case with `WetDry` at a real shoreline (the force on partly dry elements is untested).
- [ ] **F.4** Cheaper Stokes drift sampling for 10⁵ particles: tabulate per node, or Breivik 2016's two-parameter profile.
- [ ] **F.4** Low priority: the vertical Stokes velocity (left out, as in OpenDrift/LADiM). Later: cage drag from the orbital velocity.

#### Cost
- [ ] **F.4** Sources without the DIA (84 ms on 1 thread, 9 of the 60 ms bench step at 24; libm 40 % of it, 2026-10-19): pass the cached `cg` into `advance`, cache the friction's `sinh(kd)` in `set_water_level`, try a vector `exp` (≈ 3 ms at 24 threads, not bit for bit).
- [ ] **F.4** The implicit lanes kernels (`waves/implicit_lanes.rs`) zero or copy ≈ 11 arrays of 72 vectors per batch (memset/memcpy 18 % of the refraction's samples, 37 % of the shift's): build the Thomas coefficients in the sweep, reload the original from the spectrum, solve in place (≈ 2 ms of the 60 ms bench step).
- [ ] **F.4** Fuse adjacent implicit half-passes across steps and between substeps (a pair was 9.6 ms of 141, now ≈ 9.7 of the 60 ms bench step, all of it the kernels: not bit for bit, a splitting change); measure first.
- [ ] **F.4** Parallelise an exchange's per-node diagnostics (`parameters`, `bed_wave_stress`, `radiation_stress`, `StokesDriftField::new`, wavenumber tables) in one pass per spectrum (an exchange takes 148 ms, 2.7 wave steps, since 2026-10-19).
- [ ] **F.4** L2 projection from circulation to waves in `WaveCoupling2D` (`M_c⁻¹ Pᵀ M_f`), replacing the cap in `WaveCoupling2D::currents`.
- [ ] **F.4** Measure a 2 km P2 wave grid (`wave_mesh=30,23,2`) against 1 km P1 at Frøya, for cost and lee H_s.
- [ ] **F.4** The 0.04 Hz bin in 150 m sets the 7 s step: check `f_min` 0.05 Hz (≈ 25 % longer step) against a 0.04 Hz run, or LTS by frequency group.
- [ ] **F.4** Mask the wave grid to the coastline mesh's footprint (638 of 10,040 nodes lie outside it).
- [ ] **F.4** Lower priority: implicit geographic stepping with Gauss–Seidel sweeps (SWAN), fewer components (24 × 25).
- [ ] **F.4** Local time stepping in the waves (as `MultirateSSPRK3` for the tide): a coastline-fitted wave mesh (`wave_mesh=FILE.msh`) keeps ≈ 100 m edges in narrow sounds and cuts the global step 3–5× against a uniform 500 m grid (Ålesund C, `notes/waves.md`).

#### Wet/dry and forcing
- [ ] **F.4** Put the coast where the water ends in partly dry elements, with `WetDry`'s shoreline subcells (dry nodes are zeroed today).
- [ ] **F.4** Then decide whether `WaveModel2D`'s default becomes absorbing land (changes the bits of every gate with land).
- [ ] **F.4** Boundary spectra for past dates (the June 2025 runs) from MyWave 800 m's hourly partitions (wind sea + three swells, JONSWAP with cos^m), gated on days where both exist.
- [ ] **F.4** Low priority: mask sources at dry nodes in mixed DIA chunks (≤ 2–3 ms of 105); a dry-node list for `absorb_on_land`; a wind-only atmosphere snapshot on the wave mesh.

#### Viewer
- [ ] **F.4** The waves' force per node in the coupled run's snapshots (arrows of what drives the wave-driven currents); per-component amplitudes from each node's spectrum in the photo view, so directions can turn within a view (`notes/waves.md` Viewer).
- [ ] **F.4** A wave model in the viewer's live Frøya scenario (`viz/src/solver.rs` runs `run_with_callback`; `CoupledWaves2D` and `run_with_exchange` as `froya_real_data`), so the wave colourings work live, not only in replays.

## Viewer (F.3) — [notes/viz.md](notes/viz.md)

- [ ] **F.3** Separate physics from σ-truncation in the fjord3d rest-state flow under GLS (3.6e-5 m/s at 1 h, growing): halve level spacing and κ in turn, compare with Phillips's boundary-layer velocity.
- [ ] **F.3** Compare the fjord3d cage wake at P1 on 40 m quads with the 2D 20 m run, a 3D P2 run and the integrated `CageDrag2D` drag (3D LTS would allow P2, see P1.3).
- [ ] **F.3** Switch runs in one window: build scenario, terrain and resources `OnEnter` a run state (the plugins' `Startup` systems there, their entities despawned and the solver thread stopped on leaving; `Inspector::new` leaks its mesh for a `'static` locator); keep terrain between replays of one domain. Then the menu bar's Run menu can open another run or file and save a snapshot, and "Back to the list of runs" is a state change, not a new process.
- [ ] **F.3** More menu options: threads, order, LTS levels, particles, sites (`data/sites/*.txt`), places, terrain patches.
- [ ] **F.3** Report to Bevy: `pbr_fragment.wesl` reads `uv` with only `VERTEX_UVS_B` set (a mesh with `UV_1` and no `UV_0`; still so in 0.20.0); `terrain.rs` carries an unused `UV_0` until then.
- [ ] **F.3** The 15-day Frøya run with 10-minute frames (≈ 4 h, 2.3k frames) for on-demand replay.
- [ ] **F.3** A close-up of tidal flats wetting and drying: find a resolved flat, add a `--place`.
- [ ] **F.3** Offline connectivity from `.dgsnap`/`.dgpart` (needs each particle's source cage and interpolation between frames), or record online.
- [ ] **F.3** Save the solver state on live saves, not the viewer's f32 snapshot; add S to the viewer's `Layers`.
- [ ] **F.3** The photo view's wind (short waves, whitecaps) from the run's weather: the 10 m wind as two more surface fields of the snapshot (`met=` runs), in place of `--wind`; ride the waves on the top σ-layer's current in 3D.
- [ ] **F.3** Builders in `viz/` for other live scenarios (nested runs, the Mausund sub-domain).
- [ ] **F.3** Cut terrain patches out of coarser patches too, so overlapping patches work; a world-space label per farm site.
- [ ] **F.3** The photo view costs ≈ 15 ms a frame over the chart view (21.6 against 5.4 ms at 4000 × 2250 on an RTX 4060 Laptop, `dg-viz --frame-times`): bake the waves into a tiled height/slope texture once per frame if a slower GPU or more waves need it. Make the wave-group noise periodic over the hourly `misc.z` wrap.
- [ ] **F.3** Reflections of land, cages and frame (SSR or a planar reflection) and shadows for low sun; shallow-water optics (caustics, bed refraction, swash foam).
- [ ] **F.3** Particle trails, and a density field instead of points for 10⁵+ particles.
- [ ] **F.3** Low priority: a bucket grid for terrain boundary snapping (0.2 s at 7k segments); `bracket` returning nothing on a far seek until both frames arrive.
- [ ] **F.3** Blocked: salmon in the cages (pfish-bevy, once the add-on has a salmon).

## 2D numerics (P1.1, P1.2, P1.7) — [notes/2d-numerics.md](notes/2d-numerics.md)

- [ ] **P1.2** Limit η (and velocities or characteristic variables), not h, with a troubled-cell indicator (TVB-M/KXRCF); limit h only in partially dry cells (Vater et al. 2019). Strict Kuzmin on h collapses sloping elements to P0 every half-cycle.
- [ ] **P1.7** Nonlinear SWE convergence with bathymetry on curved meshes, asserting N+1 in `tests/convergence_test.rs` (thresholds 1.5/2.5 today).
- [ ] **P1.1** Remaining stage allocations: a buffer-reusing `clone_from` for `DGSolution1D`/`DGSolution2D`/`Solution3D`; limiter `Vec`s (P2.3); `Simulation3D` off the stage workspace (P4.1).
- [ ] **P1.2** End subcells on physical boundaries stay first order: the outer slope neighbour from the BC, or one-sided slopes where both nodes are wet.
- [ ] **P1.2** Dry fronts lag (Ritter 1 mm front at 71.5 m against 79.8 m): suspects strict Kuzmin at the front and zeroed momentum below h_dry; revisit with η-based limiting.
- [ ] **P1.2** Retire the `Standard` wet/dry path and the cell-average/`linearize` workarounds once the legacy `ssp_rk3_swe_2d` users are on `Simulation` (P6).
- [ ] **P1.2** A roughness map from data (sediment type or calibrated n per region), projected like `Bathymetry2D::project` (Frøya uses n = 0.025 everywhere; a Mausund phase suspect).
- [ ] **P1.2** Guidance for `h_dry` (≈ 1e-4 of the characteristic depth), derived from the bathymetry or relative (1 mm dominates the 0.1 m Thacker error, 7.4 % against 4.5 %).
- [ ] **P1.2** Chen–Noelle (2017) reconstruction instead of Audusse, for bed steps larger than the depth.
- [ ] **P1.2** Water that flows steadily without draining, in 2D on the Frøya coastline bed (`slopes3d=on debug_3d=thin=1`): before `ponds3d`, node P at (−1.395, −9.278) km kept its two copies 0.6 m apart in η (7790 at +0.18 m, perched 0.62 m above a dry interior sill it should drain over; 7791 at −0.43) with 0.38 m/s across their face, and a 2.2 m pond at (−2.153, 6.606) km held 1.23 m/s over a 6 cm sill for hours; both are filled now, but a 0.2 m film on +0.27 m land at (13.228, −4.318) km still runs at 0.8 m/s from 5 to 8 h at the same η. Log the face and subcell mass fluxes there (`notes/2d-numerics.md`).
- [ ] **P1.2** `SWE2DRhsConfig::new` defaults to `Standard` unchecked: warn on an unbalanced sloping bed in `with_bathymetry`, as the builder does.
- [ ] **P1.2** `outer_nodes` scales by affine element heights: per-node isoparametric factors (P1.3).
- [ ] **P1.1** Specialise hot paths through the kernel split: a `const N` kernel for fixed-size products; skip `reference_to_physical` + `SourceContext2D` for sources that need neither (P2.2).
- [ ] **P1.7** Merge the three legacy long-run conservation tests in `tests/swe_2d_test.rs` into one on `Simulation` when tests move off the legacy steppers.
- [ ] **P1.2** Only if friction's first-order error matters on flats: second-order IMEX for point-implicit friction (13–32 % above exact decay at dt·Λ ≈ 3).
- [ ] **P1.2** Viscosity on 1 cm films at large ν: at ν = 10 a Kattholmen shoreline element (1183) with a 1.2 cm node above the 1 mm subcell depth runs at 0.2–0.8 m/s and its film node at the 20 m/s cap (still at ν ≤ 1). Scale the viscous depth threshold with ν, or taper νh over a depth band.
- [ ] **P1.2** Lower priority: convex DG/FV blending (Hennemann–Gassner) so one node below `h_dry` doesn't switch a whole element to subcells.

## Geometry and meshes (P1.3) — [notes/geometry-mesh.md](notes/geometry-mesh.md)

- [ ] **P1.3** The 3D depth-mean M2 current is 0.78 of 2D's in 3–20 m, 0.85 in 20–50 m and 1.00 below 200 m on the same bed and parent (2026-10-16, `output/froya_p13`; Bodø the same: 0.80 in 3–20 m, 0.79 in 20–50 m, 1.00 below 200 m): the 3D log-layer drag (z₀ = 3 mm) against 2D's Manning 0.025. Choose the 3D bed roughness from observed currents in shallow water (with P3.1's measured currents), not from the 2D model.
- [ ] **P1.3** Shore films still set the 3D step: element 7507 at (4.2, −13.5) km, 1.0 m deep, ran its surface layer at 3.1 m/s at 13.5 h of the coastline day and the step fell to 0.24 s (209 steps for the half hour against ≈ 110).
- [ ] **P1.3** A jittered-mesh case for `EntropyStable`/`WetDry` in `convergence_test.rs` (the averaged metric loses an order pointwise on such meshes).
- [ ] **P1.3** Rerun the unsmoothed full-mesh 3-day baseline on current code (without `slopes3d`, ≈ 1–2.5 h) for a like-for-like comparison with the 19 m free depth.
- [ ] **P1.3** Decide the 2D `rx0` default, after the map-wide NorKyst current comparison (`output_minutes=60`, `scripts/norkyst_current_comparison.py`).
- [ ] **P1.3** Make implicit vertical advection (`Hydrostatic3D::with_implicit_vertical_advection`) the library default once the full Frøya mesh has run with it.
- [ ] **P1.3 / P2.5 / P4.5** Barotropic LTS runs (`ModeSplitIntegrator::with_barotropic_levels`, `froya_real_data bt_levels=2`, 2026-10-16): 2.3× less RHS work against the ideal 3.2× at two levels, from the rate spreading one ring per stage around the thin cliff-shore bands; each element evaluation ≈ 0.19 µs against 0.15 in the global pass (per-pass edge lists collected serially, scattered rows). Also on refined farm meshes (P2.5), where the 20 m elements set `n_bt` for the fjord.
- [ ] **P1.3** The M2 current at the NorKyst point grows with depth in 3D (0.082 m/s at the surface, 0.144 at 10 m, with the whole boundary open) where NorKyst's falls (0.27 → 0.25): the surface layers held back (GLS, surface roughness) or the island lee's 3D structure (`notes/3d-validation.md`).
- [ ] **P1.3** The MET-wind 3D run's step fell from 16–18 s to ≈ 12 s after ≈ 24 h, set by advection and internal waves at (18.8, −21.5) km in 14–22 m water (elements 229, 11912, 5107): what drives it (wind-driven shore currents, a front) and whether it lasts.
- [ ] **P1.3** Cap or ramp the first 3D steps from rest (`compute_dt` sees no Ω at rest; thin layers need |Ω|Δt/H_z ≤ 1).
- [ ] **P1.3** The 3D viscous step bound (`element_dt_viscous_swe_2d`) is harsh (ν = 50 m²/s: 0.54 s against 5 s): implicit shear viscosity, or the bound from the operator's spectrum.
- [ ] **P1.3** What dropping baroclinic pressure in thin-column elements (a one-element coastal band) does to estuarine and wind-driven shore currents, once tides and rivers run in 3D.
- [ ] **P1.3** Watch how often the splitter falls back to element-mean tracers (`CONSTANCY_TOLERANCE`) in a tidal run.
- [ ] **P1.3** Is the GLS creep (1 cm/s after a day at the 1 km grid's shallow slopes) physical slope flow (Phillips 1970; Wunsch 1970)? Compare with K/(slope·L) or run a slice.
- [ ] **P1.3** Lighten the 3D slope bound now that the vertical reference exists (does r_x0 0.25–0.3 hold Frøya at rest? At 0.2, element 11163 grows, e-folding ≈ 55 min).
- [ ] **P1.3** Connect the larger tidal ponds `fill_tidal_ponds` leaves (basins of more than one element's nodes behind an inlet the mesh does not resolve, 25 on the Frøya coastline bed, e.g. a 35 m lagoon of 52 nodes at (−0.27, −9.28) km): lower the sill along the lowest path to the open water (`notes/geometry-mesh.md` "Tidal ponds").
- [ ] **P1.3** Why only a handful of the coastline mesh's ≈ 1400 cliff elements (a resolved column under 1.5 m beside one over 19 m) ran fast before `cliffs3d` (5179, 4897, 3434): snapshots every few minutes around 1.5 h of the old bed (`cliffs3d=off`). Then whether D = 40 m can rise, giving back shore (`notes/geometry-mesh.md`).
- [ ] **P1.3** Mausund's pinnacle element 291 (a 1.2 m node among 42–55 m, unsmoothed) cools its bed layer from 8.6 to 6.0 °C within 30 min at 2 h of the 3D tide, below NorKyst's 7.50 minimum: the straddle below, seen in T.
- [ ] **P1.3** The straddle instability over unsmoothed cliffs (smoothing works around it): a per-element Haney rx1 bound, a capped or implicit pair density, or geopotential diffusion; run the spectrum harness (≥ 1800 s, check the residuals) on the slices. The coastline day's cliff shores (element 5179: 0.74 m among 61–82 m, 24–30 m/s at the surface) are land now (`cliffs3d`, `notes/geometry-mesh.md` "Cliff shores raised to land").
- [ ] **P1.3** The coastline day cools a 41–56 m basin's bed layer below NorKyst's minimum through the ebb: element 5143 at (−1.6, −8.1) km (r_x0 0.15, at the 3D bound; shore elements 4122/4123/6171 rise to land beside it) goes from 8.4 to 7.0 °C between 5 and 6 h, with 0.2–0.33 m/s in the bed layer under a depth mean of 0.01–0.04 m/s (`output/froya_day/3d_ponds`, and 7.05 °C at 6.5 h on the bed before `ponds3d`). The straddle or terrace mode below, at the bound?
- [ ] **P1.3** The shipped 3D rest state still grows at terrace vertices beside shore elements: an oscillatory mode, e-folding ≈ 2–2.4 h, period ≈ 3 h, in the bed layers across the pycnocline's foot (LimitedAkima about the reference; any Akima weight). Find its mechanism (a fully wet slice of the same terrace is stable) and fix it; gate `a_terrace_beside_a_shore_element_stays_at_rest_for_twelve_hours`.
- [ ] **P1.3** Rerun the full coastline mesh at rest with today's defaults for 24–48 h, seeded (`debug_3d=…,perturb=1e-6`): the 12 h round-off runs cannot see a 2 h mode.
- [ ] **P1.3** Coastline-mesh cost (≈ 4× the 500 m grid; 4.4e4 element-steps against 2.3e4 ideal): multirate level coherence (three-hop stage dependency ≈ 1.9×) and a non-subdividing quad mesher.
- [ ] **P1.3** `batched` SIMD and the Burn prototype assume parallelograms (`affine_metric`, `compute_volume_terms_batched`, `BurnGeometricFactors2D::from_cpu`).
- [ ] **P1.3** Triangles or quad-dominant meshes; remove the hardcoded 4 faces (`swe_2d.rs:386,908`).
- [ ] **P1.3** CSR connectivity: `MeshGPUData` as the single mesh representation.
- [ ] **P1.3** Curved high-order faces from an interpolated high-order map (Kopriva 2006), with a boundary projection of face nodes.
- [ ] **P1.3** Mass-weight `Bathymetry2D::to_cell_average`/`linearize` node means, unless retired first (P1.2).
- [ ] **P1.3** Profile the remaining curvilinear RHS overhead with AMD uProf (+2–3 % at 1 thread, +4–5 % at 24 on Frøya).
- [ ] **P1.3** If suite time matters: shorten `a_terrace_beside_a_shore_element_stays_at_rest_about_its_reference` (1–4 min, slow tier).
- [ ] **P1.3** Only if needed: implicit vertical advection's upwind flux about the vertical reference (`apply_tracer_transport_3d` with a `reference`); decaying the parent's bottom velocity below its bed in `OceanModelColumns`.
- [ ] **P1.3** Low priority: `raise_isolated_wet_nodes` on the rx0 functions' `NodeGraph`; `max_rx0` rebuilds the graph each call.
- [ ] **P1.3** When needed: Gmsh `$NodeData` (a bed in the mesh file) and `$Periodic` faces.

## Boundaries and nesting (P1.4, P1.5) — [notes/boundaries-nesting.md](notes/boundaries-nesting.md)

- [ ] **P1.5** Parent files longer than 3 days for the nested 3D runs (the per-depth comparison is `norkyst_profiles_3d`, P4.6).
- [ ] **P1.5** A datum for the parent ζ in the library default: NorKyst sits 0.20–0.40 m below the gauges' mean along the coast (median 0.28, deeper northward, 4–10 cm between seasons), neither MSL nor NN2000; its ζ carries about a quarter of the inverse barometer, so `ib=1` does not count it twice (`notes/norkyst-issues.md` §5–6).
- [ ] **P1.4** `BoundaryTides::with_transport_scaling` (`tide_transport=3`) as the default for atlas tides (Mausund sub-domain 8.1 → 4.2 cm); check the full mesh with it.
- [ ] **P1.4** Blend the bed towards the atlas depths near the open boundary for `BoundaryTides`, as `OceanModelState::blend_bathymetry` does.
- [ ] **P1.5 / P4.2** A point-implicit nesting relaxation band, 2D and 3D, so FRS-like τ ≈ Δt works (explicit today: τ 30 min against 0.25–10 s steps).
- [ ] **P1.5 / P4.2** Read parent snapshots lazily around the current time, 2D (`OceanModelReader`) and 3D (`from_file_with_profiles`, 46 MB per field for 3 days), for months of forcing or OPeNDAP.
- [ ] **P1.5** Correct NorKyst's own M2/S2 (0.3–1 cm, 10–20 min lag) in the corrected atlas from another reference (gauges along the boundary, TPXO/FES).
- [ ] **P1.5** `OceanModelState::with_tidal_correction` takes its nodal f/u at half the run's length, so the correction depends on `hours=` (a resume to 71.99 h of a 72 h run differs by 0.01 m/s in the progress lines): nodal factors at each snapshot's time, or a fixed reference time.
- [ ] **P1.4** Velocity from z-levels stops at 300 m: extend to full depth.
- [ ] **P1.4** Measure `io::depth_average_z`'s error (z-levels, 300 m cap) against the NorKyst v3 hindcast's `ubar_eastward`/`vbar_northward` (`romshindcast/norkyst_v3/norkyst_v3_sdepth_aggr.ncml`, hourly 2012-01 to 2025-08) at the Frøya boundary cells (`notes/norkyst-issues.md` §7).
- [ ] **P1.5** `NestingRelaxation2D` relaxes the level at every band node ≥ 5 cm deep, puddles cut off by land included (Mausund element 58: a 5 cm puddle filled at 3e-7 m per step towards NorKyst's level): relax only water connected to the open faces, or momentum only in shore elements.
- [ ] **P1.5** A depth range in `norkyst-client`'s grid mode (`depth = "0:1:0"` today) as a second source of 3D profiles.

## Inputs and forcing (P1.6) — [notes/inputs-forcing.md](notes/inputs-forcing.md)

- [ ] **P1.6** Settle the topobathy sea part's vertical datum (NN2000 or chart datum, 1.43 m too shallow at Mausund): check against WFS `wfs.dybdedata` points; add the 8 cm NN2000 → MSL offset if it matters.
- [ ] **P1.6** Measure `lower_wall_land` in 3D (stratified rest, and a tide with `levels=20 slopes3d=on`) before making it the default, then in `viz/src/scenario.rs`.
- [ ] **P1.6** Put the coastline mesh's walls in the water in `scripts/gmsh_coastline_mesh.py` (dilate the opened land back, or the projected bed's 0 contour), never closing a sound.
- [ ] **P1.6** A farm-scale bed: tile the 1 m topobathy level over the 50 m model using the "Dybdedata - dekning" coverage layer; Kattholmen needs another source (dybdedata WFS or a survey).
- [ ] **P1.6** Pass the last step's `RiverInflow` to `Hydrostatic3D::update_vertical_velocity` (output Ω spreads river volume over the column).
- [ ] **P1.6** UTM33/Lambert projections for coast-scale domains (the local projection's error grows with the domain).
- [ ] **P1.6** A spatial index for point-in-polygon and GSHHS inner rings in `coastline.rs:81-95` (the land mask costs ≈ 20 s of setup at Frøya).
- [ ] **P1.6** Fix or delete the unused `LandMask2D::from_coastline_and_bathymetry`.
- [ ] **P1.6** Count wet as h > 1 cm in the example's η statistics, so mm films don't set "η max".
- [ ] **P1.6** Cut the atmosphere forcing's memory (≈ 210 MB at 10⁶ nodes): share stencils per element, or drop them after regridding.
- [ ] **P1.6** River details: river momentum `Q/(width·depth)` for jets at narrow mouths; spreading over elements within a radius if one-element plumes show.
- [ ] **P1.6** If the wind-driven residual matters at farms: stress from the wind relative to the current, with a stability-dependent COARE drag.
- [ ] **P1.6** Low priority (< 5 mm): the tidal-potential sign against its docs (`source/swe_2d/tidal.rs:299-305`) and the diurnal Love factor.
- [ ] **P1.6** Not worth it now: a constrained bed projection at walls (median −0.2 m, 1st percentile −12 m).

## Performance (P2) — [notes/performance.md](notes/performance.md)

- [ ] **P2.5** A semi-implicit free surface (θ-method, as TRIM/UnTRIM, SLIM, Thetis), the main cost lever: farm meshes step at ≈ 0.024 s on gravity waves. Needs a global SPD solve and wet/dry care.
- [ ] **P2.1** Port `compute_dt_advection_2d`, `compute_dt_tracer_2d`, `compute_dt_viscosity`, `Mesh::compute_dt`, `time/ssp_rk3.rs:110` and 1D `compute_dt_swe` to `compute_dt_swe_2d`'s per-node metric form in one helper (`2/(λ_r+λ_s)` for advection).
- [ ] **P2.5** LTS oversubscription: under ≈ 40 % external load the farm took 358 s at 24 threads against 3.9 s at 4. Serial cut-off by work, a capped pool for small meshes, or documented thread counts.
- [ ] **P2.5** Viscous cost (≈ 2× inviscid): a per-stage BR1 gradient cache for subset-RHS neighbours; store `νh` with the gradients in pass 1.
- [ ] **P2.5** Smagorinsky `compute_dt` (≈ 13 % of a step at 24 threads): record each element's max ν in the last stage's gradient pass (lagged), or bound every few steps.
- [ ] **P2.2** Line-by-line 3D horizontal kernels (`transport_divergence_element`, `advective_divergence_element`, the PGF volume loop) with contravariant vectors loaded once: ≈ 3× fewer pairs at P2.
- [ ] **P2.2** Finish SIMD batching of `element_rhs`: one node-state pass, a batched surface loop/LIFT, no copy of the batched volume.
- [ ] **P2.2** Shorten the face-pass gather with per-edge node-index lists, or an AoSoA layout (3.8× against 2.5× in the microbenchmark).
- [ ] **P2.2** Precompute the subcell batch's outer-node distances per face node with the geometric factors.
- [ ] **P2.2** Fuse the face pass into the element loop (coloured or two-phase schedule), removing the second memory pass.
- [ ] **P2.3** Move the Kuzmin limiters' per-stage `Vec`s to a workspace; fuse limiter, wet/dry correction and implicit damping into one in-place pass.
- [ ] **P2.5** Make the Kuzmin limiters work under LTS (they panic today).
- [ ] **P2.5** Validate LTS on Frøya over the 15-day harmonic run (`lts=6` matched global M2 to 1e-4 over 6 h).
- [ ] **P2.5** Check that cage-shed eddies survive leaving the farm's LTS level (interface error ≈ 50× SSP-RK3's).
- [ ] **P2.2** Measure `-C target-cpu=native` for the SIMD paths and `mimalloc` for the allocating paths.
- [ ] **P2.2** Override `SourceTerm2D::add_element` for wind stress, pressure and tidal potential with precomputed per-node fields (Coriolis gave ≈ 11 %). `GriddedAtmosphere2D::add_element` already reads regridded per-node snapshots, but per element and stage it searches the time stencil, looks the snapshots up in a thread-local list and evaluates the drag law: 1.7 % of the 3D coastline run's CPU time (uProf, 2026-10-19); a per-stage field of the wind stress and pressure gradient would leave one read per node.
- [ ] **P2.3** One-level fused stepper: the third stage writes the state directly; measure first.
- [ ] **P2.5** Reduce LTS interface spreading (2.6× of an ideal 4.6× on the farm): coherent levels, grading ≤ 1.2, or decoupled coarse stages.
- [ ] **P2.5** Lag the viscous flux across LTS interfaces where ν is small (2.60× → 2.44× on the farm).
- [ ] **P2.1** Measure linear stability limits for curved elements and N ≥ 5 (`PositivityBound` does not relax without a table entry).
- [ ] **P2.6** A wave-step bench before the F.4 SIMD port.
- [ ] **P2.6** Full-RHS and full-step throughput at realistic size (≥ 100k elements, bathymetry, limiter, open BCs); DOFs/s, scaling over 1–16 cores, bandwidth; allocation counting in CI.
- [ ] **P2.6** Audit the slow test tier (`.config/nextest.toml`, 25 gates, 28 test-minutes): spread-sea ladders, 3D rest gates' model hours, particle counts.
- [ ] **P2.4** Limit only troubled cells.
- [ ] **P2.7 / P0.11** Decide GPU: fix Burn (`src/solver/burn/rhs.rs` computes the HLL flux and discards it, so no surface term; fusion disabled, ~13× the bytes per node-stage) with a burn-vs-CPU test, or replace it with fused f64 CubeCL/cudarc kernels over CSR faces. Pays only from ≈ 10⁵ elements; fewer steps come first.
- [ ] **P2.7** An MPI/domain-decomposition plan (METIS, halo exchange of face traces).
- [ ] **P2.1** Low priority: more SSP stages (SSP-RK(9,3)/(10,4)) need a non-two-register `SspScheme`; the Zhang–Shu r/s split in ρ (≈ 2 % on Mausund).
- [ ] **P2.2** Only if `Standard` stays: sum-factorised volume term (`swe_2d.rs:835-874`), diagonal LIFT (`kernels.rs:490-517`), one Riemann solve per face.
- [ ] **P2.5** Low priority: the LTS coarse-step anchor from a log₂ histogram (≤ 0.5 % on Frøya); keep wet/dry films from setting a tiny Smagorinsky step, only if it shows.

## 3D: mode splitting (P4.1) — [notes/3d-mode-splitting.md](notes/3d-mode-splitting.md)

- [ ] **P4.1** Revisit the ROMS `rufrc`-style G (full PGF in the 3D RHS) now that the σ-pairs PGF has the DG face coupling of η.
- [ ] **P4.1 / P4.2** Check whether the 3D kernels still read η rather than h (PR 1 left moving them to h to P4.2).
- [ ] **P4.1** Low priority: the internal seiche's small dt-dependent phase residual (+0.011 / −0.018 / −0.039 % at 240 / 120 / 60 s).

## 3D: numerics (P4.2, P4.3, P4.5) — [notes/3d-numerics.md](notes/3d-numerics.md)

- [ ] **P4.5** Dry nodes' tracers drift (28.8 °C at element 7601, 57 h, in `output/froya_3day/3d_rk43`, wet neighbours 10.4; 10.3 again once wet): not the surface fluxes (thin columns get none), so the shoreline subcell or `from_inventory` transport. Hold a dry node's T and S at their last wet values, so no consumer of the state needs a mask (the viewer's readout has one: `Probe::eval_level_wet`).
- [ ] **P4.3** Fix the cliff-shore class instead of masking it: `cliffs3d` (20 m since 2026-10-17, 40 blew up the 72 h run at 37 h) makes land of 15.3 km² of shore water at MSL. A shore column within an element whose other nodes are ≥ 20 m deep runs away near high water once it is resolved (≈ 1 m of 5 cm layers beside 1.4 m ones): a thin-column criterion relative to the element's deepest node (subcells while D < c·D_max), or the pressure gradient and advection between such columns (`notes/geometry-mesh.md` cliff shores).
- [ ] **P4.5** The 72 h nested run at 150 s per model hour: 199 since 2026-10-19 (SSP-RK(4,3) stages, `notes/3d-numerics.md` "Third depth pass"; day 1 160, day 2 230), ≈ 1010 ms per 16 s step on the 70 h bench: 3D stages 546 (4 × transport RHS 60, momentum RHS 33, limiter 33), barotropic pass 274, post_process 56, slow forcing 40, vertical implicit 39. The levers are the lines below, the largest 3D local time stepping.
- [ ] **P4.5** Local time stepping in the 3D stages (on `MultirateStepper`'s schedule, as the barotropic pass): the step is set by a few shoreline elements 1–2.4 m deep with ≈ 1 m/s; the ideal gain is 3.3–4.6× at 2–4 levels at 70 h and 3.8× / 5.4× / 5.8× at 2 / 3 / 4 levels on day 2's hard hours (own steps' p1 3×, median 9.5× the smallest; `froya_real_data debug_3d=dt_dist`). The stage hooks (the limiter's patches, the conversions) and the barotropic coupling's constant rates need a subset form.
- [ ] **P4.5** The splitter's per-stage data movement is ≈ 15 field passes since 2026-10-15 (from ≈ 60): what is left is one inventory → value conversion and two copies per stage, which kernels reading `H_z φ` would remove.
- [ ] **P4.5** The forward-backward (AB3-AM4) barotropic fast mode, fewer RHS per substep.
- [ ] **P4.5** Make SSP-RK(4,3) `ModeSplitIntegrator`'s own default for wet/dry runs, the barotropic substeps at the caps and the 3D stages (`with_baroclinic_scheme`) with a doubled 3D CFL (changes every 3D gate's bits; `froya_real_data` takes both since 2026-10-15 and 2026-10-19, the 72 h run's scores equal SSP-RK3's).
- [ ] **P4.5** Decide `Hydrostatic3D::with_viscosity_once_per_step` (`debug_3d=visc_step`, off): −11 % at 70 h, but the NorKyst point's current 0.26 mm/s off the reference against 0.05 (`notes/3d-numerics.md` "Third depth pass"). Lagging only ν (the strain) and keeping the per-stage BR1 tendency would keep most of the accuracy and little of the gain; the turbulence's advection once per step (4 × 21 ms) keeps constancy and is first order alike.
- [ ] **P4.5** The coastline day's Kuzmin limiter changes about 40 % of all element-layers at every stage, by 1e-5 to 1e-3 (°C or PSU; 22 steps from 6 h: 17.2 M of 42.4 M, 98 % of them by more than 1e-8; 21 by more than 1), so real if mostly small cuts, not round-off; and 95 % of the element-layers fail the own-column test, so the patch bounds (≈ 60 column interpolations each) are computed nearly everywhere (≈ 42 ms per call, four per step). Find whether P2's horizontal variation within an element is what it cuts. (The last stage's call is gone since 2026-10-16: the post-step call does its work after the vertical diffusion.)
- [ ] **P4.5** The Kuzmin kernel is still ≈ 12 % of the 3D run's CPU time after the T/S fusion (32 ms per call, four per step; ≈ 3 µs per element layer for ≈ 60 bracket searches, each with a division, and their T and S evaluations; `notes/3d-numerics.md` "Third depth pass"): the vertex patches' brackets are each computed again for the union's nodes; precomputing `1/spacing` per column layer would remove the divisions (not bit for bit).
- [ ] **P4.5** The barotropic filter integrates 1.3 n_bt substeps (30 % past tⁿ⁺¹); the fused stage costs half a 2D RHS (0.56 ms, 10 streamed arrays): a shorter window or fewer streams.
- [ ] **P4.5** Fewer parallel regions per barotropic stage: the damping and wet/dry correction run in the fused stage's pass since 2026-10-15; the fused stage as the RHS's per-element continuation is left (it streams ≈ 24 MB per stage at ≈ 44 GB/s, DRAM speed: the pass's 2D arrays, ≈ 31 MB, do not fit the 24 MB L3).
- [ ] **P4.5** The turbulence's advection still costs as much as T, S, u and v together (22 ms per call, three per step, since the all-levels kernels: the pair's kernel 12, re-staggering the w-cell transports 5.5, copies and sources 4). Average the layer transports on the fly; or advect the turbulence once per step, as ROMS's GLS does in its own predictor/corrector.
- [ ] **P4.5** The vertical reconstruction is ≈ 35 % of the tracer kernels since the horizontal part runs on whole levels (`subtract_vertical_flux`: limited Akima per column, T and S twice with their reference): run it on an element's columns as SIMD lanes, or share the reference's surface values between stages (Ω changes, the reference does not).
- [ ] **P4.5** The other 3D kernels still loop level by level at a stride of the level count (Smagorinsky viscosity `viscosity_3d.rs`, PGF, the layer transports' face fluxes and correction): put them on `[node][level]` blocks as `transport_divergence_levels`/`advective_divergence_levels` (2026-10-16: 1.6–2× on the advection; the layout is also the GPU port's).
- [ ] **P4.5** `LayerTransport::stagger_from` averages the subcell fluxes of every element (≈ a third of its 5.5 ms); only subcell elements read them (zeros elsewhere would keep `LayerTransport::subcell`'s contract).
- [ ] **P4.5** `Simulation3D` refreshes ρ at every step's start, which `Hydrostatic3D::post_process` now leaves current (it refreshes it after the limiter since 2026-10-16): only the first step and states changed between steps need it.
- [ ] **P4.5** A depth-aware coastline mesh (OceanMesh2D's CFL bound, element size ≥ c·Δt/CFL): 1 % of the coastline mesh's elements lie within 1.66× of the smallest barotropic step and 5 % within 2.35×, cliff-shore elements 65–100 m across in 50–80 m of water; the positivity cap alone is only 1.29× (2026-10-15). Coarsening them would raise the global substep without LTS, at the cost of the validated bed.
- [ ] **P4.2** A cheaper band parent-column evaluation (`OceanModelColumns` searches per node, stage and layer): precompute vertical weights, or time-interpolate once per baroclinic step. 3.4 % of the 3D coastline run's CPU time (uProf 2026-10-19: `ProfileSeries::sample` 2.4 %, `OceanModelState::evaluate`, `TimeStencil::new`); u, v, T and S repeat the same depth search at each of the four parent points and two snapshots (bit for bit if shared).
- [ ] **P4.2** The parent's shear dispersion in G's depth mean at nested faces.
- [ ] **P4.3 / P1.3** ROMS `Vtransform=2` and the true Vstretching 4 (`ROMSVstretching4` is not ROMS Vs4; `SongHaidvogelStretching::hc` is unused in `vertical/sigma.rs` `z_at_levels_into`); Ω must use ∇·(H_z u) once H_z varies horizontally.
- [ ] **P4.3** An r_x1 (Haney) diagnostic on the σ levels and an r_x1 bound for the 3D grid.
- [ ] **P4.3** The σ-pairs PGF on curvilinear elements (affine metric today; `Hydrostatic3D` asserts affine): the chain rule `rx_i Dr + sx_i Ds`.
- [ ] **P4.5** The horizontal Kuzmin tracer limiter as the default (`farm_3d`, `Hydrostatic3D`; the P1 farm channel undershoots 0.07 °C without it), after a realistic run over slopes.
- [ ] **P4.5** Positivity of the tracers beyond the vertical flux: the end layers' range bound follows the vertical outflow Courant number since 2026-10-21 (`in_range`, `LayerTransport::dt_over_depth`), but not the layer's horizontal outflow (DG needs λ ≤ w₁ = ⅙ of the mean), which Zhang–Shu's sum would add to it. Measure its share on a river front over a steep flank first: no fjord run needed the vertical part at `cfl_3d` ≤ 8 (`notes/3d-validation.md`, sill fjord).
- [ ] **P4.5** The interior vertical limiters are monotone only to an outflow Courant number of ½ (Koren's φ ≤ 2; beyond it φ ≤ 2(1 − c)/c), and the default `ImplicitVerticalAdvection::new(0.4, 0.8)` leaves the explicit part up to 0.8: scale `tvd_limited`'s `behind`/`ahead` bounds by the same (1 − c)/c with the layer's own Courant number (`dt_over_depth` and Ω are in `surface_values`' reach); measure `profile_3d` and the gates, as the range bound changes nothing below ½.
- [ ] **P4.5** Give the turbulence advection its range (`k`, `ψ` ≥ 0 as `AdvectedTracer::range` in `Hydrostatic3D::turbulence_transport_rhs`, now `UNBOUNDED`): the closure resets a non-positive point to no turbulence (`notes/3d-mixing.md` positivity); measure `profile_3d` and the gates first.
- [ ] **P4.5** Why P3 on 250 m without viscosity ends 3.3e-9 °C out of range (2–8e-12 at P1/P2): the vertical advection's element mean or the Kuzmin P3 bounds?
- [ ] **P4.5** 3D Smagorinsky on a fjord with a sill and tidal shear (may over-damp a tidal mixed layer; else Leith or Ri-dependent ν).
- [ ] **P4.5** Rotated/geopotential horizontal diffusion of 3D tracers via BR1 (none exists; `viscosity_3d` is momentum only).
- [ ] **P4.5** Subcell (GLL-cell) finite volumes for shoreline tracers instead of element means per level (needs 2D subcell fluxes in DU_avg2).
- [ ] **P4.5** Port the 3D advective dt bound to the 2D per-node metric form (`compute_dt_swe_2d`), with a linear stability measurement (changes what `cfl` means). Mausund holds `cfl_3d=2` and blows up at 3 in a 50 m element during the ramp; the coastline mesh blows up at 1.5 in a cliff element (1.2 m beside 82 m); `froya_real_data` takes 2 since 2026-10-15 (the coastline day holds it on hours 0–2 and 6–8). Remeasure with `cliffs3d` (on by default since 2026-10-14), which makes that shore land: the limit may rise to Mausund's 2. Found 2026-10-19: in the runs the bound binds in 1–2.4 m shoreline elements (`c₁` at its floor), whose subcell finite volumes have a positivity limit of Courant 0.118 diagonally at P2, which `cfl_3d=2` meets; the DG elements' linear limit is 0.318 (Fourier analysis, SSP-RK3), 2.9× the bound. A per-element bound that knows which elements are subcell ones could let the DG elements step at their own limit (with 3D LTS). The WKB `c₁` underestimates a sharp pycnocline (∫N dz → √(g′δ) for an interface δ thick), which matters only where internal waves bind.
- [ ] **P4.5** `Simulation3D`'s default `cfl` (0.5, `SimulationConfig`'s, shared with 2D) is 0.17–0.25 of the measured limit: give 3D its own default once the bound is ported.
- [ ] **P4.5** EOS: delegate to the fixed UNESCO EOS with a linear fast path; TEOS-10 later. In `UnescoEOS::update_density`, `s * s.sqrt()` for `s.powf(1.5)` and a flat parallel loop.
- [ ] **P4.5** RPE (`PotentialEnergy3D`) in 3D run output (a `Simulation3D` callback or NetCDF attributes).
- [ ] **P4.5** One in-place Thomas solver for `tridiagonal::solve_tridiagonal`, `implicit_advection::ColumnSolve::solve` and the waves' `thomas`; in `solve_diffusion_pair` (one elimination per pair since 2026-10-19), 8 columns as SIMD lanes.
- [ ] **P4.5** Keep the per-element viscosity in `ViscosityScratch3D` (allocated each step in `time_step_limits`/`compute_dt`).
- [ ] **P4.5** Move the 2D kernels' serial/parallel entry points and workspace guards onto `solver::core::blocks`; keep one of `DisjointChunks`/`blocks`.
- [ ] **P4.5** Bundle `apply_momentum_transport_3d`'s 12 arguments into `MomentumAdvection { horizontal, vertical }`, likewise for tracers.
- [ ] **P4.5** Remove `GriddedAtmosphere2D::split_for_3d`'s panic on a clone (split through the shared state, or `&self` cloning the `Arc`).
- [ ] **P4.5** Revisit or drop `TracerLimiter3DConfig::vertical_column_bounds`.
- [ ] **P4.5** Relabel or convert Ω and w in output (`Solution3D::w` is Ω at layer centres); report film speeds only where D ≥ the minimum column depth.
- [ ] **P4.2** Low priority: a baroclinic radiation OBC (per-mode Flather or Orlanski, ROMS RadNud), for narrow nesting bands (1 km: 18 % left).
- [ ] **P4.3** Low priority: the pycnocline PGF error on uniform levels (2–4e-4 m/s²): stretched levels or a tabulated ρ̄(z); higher-order vertical reconstruction in the PGF.
- [ ] **P4.5** Low priority, only if a profile shows it: G's advection from the previous step's transports; `Integrable::combine` overrides; PGF cubic integrals vectorised across levels; the remaining serial loops (≈ 3 ms/step at 8192 elements); a parallel-sorted or binned RPE; implicit vertical advection folded into the diffusion matrix.
- [ ] **P4.5** Low priority: the transient 1e-8 °C excess in the inviscid 125 m TVD run (Kuzmin bounds extrapolated past the end layers). Not worth it now: 2D passes on one thread below a size threshold.

## 3D: mixing (P4.4) — [notes/3d-mixing.md](notes/3d-mixing.md)

- [ ] **P4.4** The top 25 m runs 0.1–0.2 °C under NorKyst from day 1 (NorKyst's interior heats ≈ 20 % more than MET Nordic's weather can): run COARE on NorKyst's own MEPS forcing at its SST to tell the weather product from the model.
- [ ] **P4.4** Weaker restratification than NorKyst after a wind drop (Frohavet 63–69 h, T(0) − T(10 m) +0.30 against +0.66 °C): GLS diffusivity under weak wind against NorKyst's `AKs`.
- [ ] **P4.4** Wind stress from the same COARE iteration (stability-dependent, relative to the surface current) instead of `GriddedWindStress`'s Large & Pond.
- [ ] **P4.4** COARE cost: 10 iterations against ROMS's 3; exit on convergence (≈ 1 % of the 3D step now).
- [ ] **P4.4** De-accumulate forecast radiation (MEPS/AROME) in `AtmosphereReader::from_file`, so forecasts work without `met_forcing_subset`'s conversion.
- [ ] **P4.4** Low priority: the rain's heat, COARE's cool skin and warm layer, the latitude per column.
- [ ] **P4.4** Surface stress and heat/freshwater fluxes in thin columns (`with_min_column_depth`): τ/ρ₀ through `G`, the flux over the whole thin column (tidal flats get no wind or heat in 3D today).
- [ ] **P4.4** Thin columns' temperature leaves the physical range in the 5–8 June 2026 runs: down to −2.00 °C (the bound) at low water and up to 21 °C (k-ε) or 36 °C (generic) at 54–58 h, while the ordinary columns stay at 6.9–14.8 °C. Find which flux or advection step does it (`notes/3d-validation.md` "The 2026 window").
- [ ] **P4.4** Wave height for the surface roughness, z₀ₛ ≈ 0.6 H_s per column, from a MyWaveWAM reader or fetch-limited H_s(U₁₀, fetch).
- [ ] **P4.4** Positivity-safe advected `k`/`ψ` (Zhang–Shu scaling toward the element mean, or a length cap); check on a real run with fronts.
- [ ] **P4.4** k-ε with waves: σ_ε as a function of P/ε (Burchard 2001, GOTM `sig_peps`); until then k-ω or generic with waves.
- [ ] **P4.4** One bed roughness: `with_bottom_drag` hands its z₀ to `GlsMixing` (κ 0.41 in the drag, 0.416 in the closure); a spatially varying z₀ and a C_d consistent with the 2D Manning n (2D/3D undisturbed currents differ, 0.31 against 0.34 m/s at the farm).
- [ ] **P4.4** Thin columns take C_d,max = 0.1: check the 3D wet/dry front speed on a real beach against 2D `WetDry` with Manning.
- [ ] **P4.4** GLS cost (`vertical_implicit` 2.1 ms against 0.8 for Pacanowski–Philander, mostly `powf`): a k-ε fast path, one ε per w-point.
- [ ] **P4.4** Turbulence-advection cost: average layer transports on the fly in `turbulence_transport_rhs`, the end-point override as a reconstruction BC.
- [ ] **P4.4** Drop `PacanowskiPhilanderMixing`'s own `g`/`ρ₀` (API change).
- [ ] **P4.4** Low priority: reset thin columns' `k`/`ψ` on re-wetting; second-order bottom drag; reuse `GriddedWindStress`'s `tⁿ⁺¹` evaluation.

## 3D: validation (P4.6) — [notes/3d-validation.md](notes/3d-validation.md)

- [ ] **P4.6** The afternoon peak against observations. ACSPO and Copernicus Marine's ODYSSEA L3S are night and foundation SST only, and Copernicus Marine has no in situ SST in the domain. Daytime needs SLSTR L2P from CDSE (09–10 UTC, its own account) or VIIRS L2P from PO.DAAC (13:30 local, Earthdata) (`notes/3d-validation.md` "What the accounts give").
- [ ] **P4.6** The afternoon warm layer: at the farms on 5–8 June 2026 the model's 0–3 m difference peaks at +0.34 °C at 15 UTC against NorKyst's +0.47, forms earlier and lingers into the evening, the same with k-ε and generic. Try finer levels at the surface (`levels=30`, or stronger surface stretching) and the surface k/ψ boundary conditions before the closure (`notes/3d-validation.md` "The 2026 window").
- [ ] **P4.6** The model starts 0.2–0.25 °C colder than NorKyst at the farms (0 and 3 m, from day 0, inside the mesh): find which start-up step cools the coastal columns (initial T/S at the 1442 nodes on NorKyst's mean profile, the bed blending, the cliffs and ponds).
- [ ] **P4.6** Draugen (64.35° N, 7.78° E; Copernicus Marine `AR_TS_MO_Draugen`, 10 min) against the forcing and the surface current: wind at 10 m, air temperature, dew point and current against MET Nordic, MEPS and NorKyst. It needs MET and NorKyst subsets about 30 km wider to the north-west.
- [ ] **P4.6** A source for `TracerReferenceProfile` (a z-binned horizontal mean of the state or the parent), needed before the Kuzmin limiter over slopes in a realistic run.
- [ ] **P4.6** Move the smooth references to `from_smooth_fn` and measure (Frøya `tidal_run_3d`, `levels=N`, `seamount_3d.rs kuzmin=2`, `src/simulation/simulation_3d.rs:3359`).
- [ ] **P4.6** Tanh-pycnocline seamount at r_x0 0.08 spins up 0.38 → 1.5 cm/s over 5 days: 30 days, refine nx and levels, compare with Shchepetkin & McWilliams (2003, Fig. 10).
- [ ] **P4.6** Exact vertical PGF/advection energy exchange for curved profiles (the Hermite mean as the advection's face value); does it remove the residual seamount growth?
- [ ] **P4.6** The balanced reference sampled at the reference η gives an η/D error under a tide: refresh it or interpolate to current level depths; measure with a tide over the seamount.
- [ ] **P4.6** At r_x0 0.32 the Kuzmin limiter adds to the error flow: a detector for steep continuous fronts (Persson–Peraire), isopycnal bounds for spice.
- [ ] **P4.6** The sill fjord 1 km wide is less critical at its mouth (F₁² 0.84, 0.69 without rotation): the exit's expansion, a control seaward of the mouth, or the width mean (`notes/3d-validation.md`).
- [ ] **P4.6** The sill fjord's entrainment against Stigebrandt's: P under wind and tidal mixing (the model's P 1.4–1.7 is the outflow's shear alone).
- [ ] **P4.6** Exact section budgets from the model's own face fluxes, RK-weighted (salt budget off 1–3 % from nodal `H_z u S`); needed for P5.2 farm exchange output.
- [ ] **P4.6** Periodic meshes: `Mesh2D::elements_at_vertex` has no periodic images, so Kuzmin bounds there are one-sided (also `limiters/swe_2d.rs:260`, `tracer_2d.rs:593`).
- [ ] **P4.6** The 3D limiter's remaining cost (1.6 ms/call): one buffer of inventory weights per call, T and S in one pass.
- [ ] **P4.6** The lowest S in the MET run fell from 9.4 to 7.8 after ≈ 24 h: find the node (a river source, a pond or a thin column) and whether it is bounded.
- [ ] **P4.6** Frohavet's depth-mean M2 current is 1.41× NorKyst's in the 72 h open-boundary run (0.388 against 0.275 m/s, G 58° against 55°; 0.89× when walled): its 83 m column against NorKyst's 55 m, or the strait's transport? A section across Frohavet over the 3 days (`norkyst_profiles_3d section=`).

## Operational (P5) — [notes/operational.md](notes/operational.md)

- [ ] **P5.1** Delete the superseded `Discharge2D`/`ConstantDischarge2D` (and the `River` tag's boundary dispatch) with P6.
- [ ] **P5.1** River mouths from NVE's river network: NorKyst's are its 800 m coastline faces, snapped 1.6–1.9 km onto the Frøya coastline mesh.
- [ ] **P5.3** Refuse a restart resumed with different physics options (compare them, or a configuration hash).
- [ ] **P5.3** Numbered restart copies (`restart_keep=`) or per-time names, for spin-up branching.
- [ ] **P5.2** A ROMS variable-name mapping for `NetCDFWriter` (CF-1.8 today).
- [ ] **P5.3** 2D restarts (`Simulation`, `SWESolution2D`, the multirate stepper's state).
- [ ] **P5.3** A NetCDF export of restarts (`ocean_rst.nc`) once P5.2's conventions are settled.
- [ ] **P5.4** The operational pipeline: run scripts, monitoring/alerting, THREDDS/OPeNDAP serving.

## Tech debt (P6) — [notes/tech-debt.md](notes/tech-debt.md)

- [ ] **P6** Move `profile_cpu`, `quick_profile` and `high_res_benchmark` off `ssp_rk3_swe_2d`/`SWE2DTimeConfig` onto `Simulation` + `SWEPhysics2D` (precondition for the next item).
- [ ] **P6** Delete the legacy integrators (`ssp_rk3*.rs`, `burn_ssp_rk3.rs`; the `coupled_swe_tracer` ones went with the 2D tracers, 2026-10-15); one time loop with `Simulation3D::run_with_callback` folded in, landing callbacks exactly (they drift today: 300 s asked, 316 s apart; `Simulation3D::with_landing_interval` lands steps on an interval since 2026-10-19, `froya_real_data` on its snapshot times).
- [ ] **P6** Build the `netcdf` feature in CI with `--features netcdf/static` (~5.5 min cold); consider it the documented alternative to conda.
- [ ] **P6** Collapse duplicate config types: `SWEFluxType2D`/`StandardFlux2D`, the three limiter enums, `HorizontalViscosity2D`/`3D` (differ only by `h_min`).
- [ ] **P6** Error handling: `SimulationResult` → `Result<SimulationStats, SimulationError>`; reachable panics (`timeseries_reader.rs:198`, `geotiff.rs:239-245`); library `eprintln!` in `tide_gauge.rs`.
- [ ] **P6** Prune ~200 root re-exports to a `prelude`; export `Simulation3D`.
- [ ] **P6** Feature-gate `tiff`/`shapefile`/`geo` behind `geodata`.
- [ ] **P6** Retire the mostly dead `solver/simd/` with the `Standard` kernel (P1.2).
- [ ] **P6** `froya_real_data` ignores unknown `key=value` options: a binary built before `site=`/`nu=` existed ran a farm sweep as plain tides (2026-10-25). Record the keys `Options::parse` reads (`args.get`, `get`, `constituents`) and reject the rest after it.
- [ ] **P6** Decide crate vs repo name and a `dg-core`/`roms-rs` split.
- [ ] **P6** Characteristic-based limiting for 2D SWE (likely subsumed by P1.2).
- [ ] **P6** Low priority: faer 0.23 → 0.24 (setup-time solves only).
- [ ] **P6** `Hydrostatic3D`'s configuration is 20 public fields and 23 consuming `with_*` builders, so it can change between steps while caches assume it fixed (the kept advection of `G` since 2026-10-16, keyed on the state only): make it private once built, or invalidate the caches on change.
- [ ] **P6** One test-support constructor for a distorted quadrilateral mesh: eight test modules have their own `distorted_mesh` (`operators/geometric.rs`, `bathymetry_2d.rs`, `point_locator.rs`, `particles/walk.rs`, `solver/probe.rs`, `tests/cage_drag_test.rs`, `tests/curvilinear_3d_test.rs`, `tests/particle_tracking_test.rs`), and `transport_3d.rs`'s kernel gate a ninth inline.
- [ ] **P6** The coastline-day bench as a script (`scripts/`): alternating A/B pairs from the 6 h restart with `DG_PROFILE=1`, and the byte comparison of two runs' restart files for bit-for-bit claims; each perf pass rebuilds it in a scratchpad.

## Future (P7) — [notes/operational.md](notes/operational.md)

- [ ] **P7** IMEX time stepping via diffsol (implicit vertical diffusion).
- [ ] **P7** Subcell positivity preservation with convex limiting (Wu et al. 2024).
- [ ] **P7** hp-adaptive mesh refinement.
- [ ] **P7** Data assimilation (EnKF, then 4D-Var).
- [ ] **P7** Two-way nesting.
- [ ] **P7** Biological coupling (NPZD); sediment transport.

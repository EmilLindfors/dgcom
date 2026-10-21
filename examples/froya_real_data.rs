//! Frøya–Smøla–Hitra: tides on real bathymetry.
//!
//! The 2D production path (`Simulation` + `SSPRK3` + `SWEPhysics2D`) with the
//! wet/dry defaults: the `WetDry` split form, HLL, positivity limiting, velocity
//! desingularization, point-implicit Manning friction, and the positivity CFL.
//!
//! 1. **Domain.** A bed raster (`BedRaster`): Kartverket's topobathy model
//!    (`dem=`, land heights and depths in one 50 m grid) when present; else
//!    the GeoTIFF bathymetry (0 at its dry pixels) with a land mask from
//!    GSHHS, four times finer, where the bed is `land_elevation`. The raster
//!    is L2-projected onto the nodes of a rectangular grid
//!    (`Bathymetry2D::project`; `bed=point` samples it at the nodes), and only elements
//!    with a node below mean sea level are kept (`Mesh2D::retain_elements`);
//!    their faces to dropped elements are coastline walls. Land nodes inside
//!    kept elements are dry shore for `WetDry`. The sides of the rectangle
//!    are open where they cross water.
//! 2. **Lake at rest.** Walls everywhere, no forcing: the largest spurious
//!    current and surface error after `rest_hours` (exact balance keeps both
//!    at round-off; the collocated scheme reached m/s on steep beds).
//! 3. **Tides.** A characteristic open boundary (`CharacteristicOBC`) forced
//!    by NorKyst-800 boundary tides (`BoundaryTides` from the tidal atlas
//!    `tides=<file>`, made by `examples/norkyst_boundary_tides.rs`: η, ū, v̄ of
//!    10 constituents plus P1 and K2, varying along the boundary), on a
//!    `ModelClock` starting at `start`. `tide_transport=3` scales the atlas
//!    velocity to carry NorKyst's transport over the child's bed, by up to a
//!    factor 3 (`BoundaryTides::with_transport_scaling`; off by default). Coriolis, Manning friction, optionally
//!    wind and an atmospheric pressure gradient (with the inverse-barometer
//!    level at the open boundaries): uniform (`wind`) or from a weather model
//!    (`met=<file,…>`: MET Nordic, MEPS, AROME-Arctic or ERA5 NetCDF, e.g.
//!    from `met_forcing_subset`; `GriddedAtmosphere2D`, ramped like the tide).
//! 4. **Nesting** (`norkyst=<file>`, e.g. from `norkyst_nesting_subset`):
//!    instead of the atlas tides, NorKyst-800's hourly ζ, ū and v̄ at the open
//!    boundary (`OceanModelState`: tides plus the coastal current and the
//!    wind-driven flow), with the velocity scaled to conserve the parent's
//!    transport, a relaxation band `band_km` wide with timescale
//!    `band_minutes` at the boundary, and the bed blended to NorKyst's across
//!    the band (`blend=0` to keep it), ramped up from rest over `ramp_hours`.
//!    Open faces NorKyst does not cover become walls; more than `max_closed`
//!    (default 0.05) of them is an error, since a parent file cut for a
//!    smaller domain turns the open boundary into a closed box (the 3D
//!    coastline day in `mausund_norkyst_3d.nc`, cut for Mausund, closed 120
//!    faces and halved the M2 current, TODO P1.3). NorKyst's ζ is shifted
//!    by minus the mean level `Z0` of the boundary atlas (its datum sits
//!    0.28 m below mean sea level here), or by `nest_level=`. NorKyst's
//!    `gauge_gains` and `gauge_ratios` constituents are replaced by the
//!    gauge-corrected ones (below) by adding the difference between the
//!    corrected and the raw atlas, which is NorKyst's own harmonic fit, so
//!    the residual flow is kept (`OceanModelState::with_tidal_correction`;
//!    `nest_tides=raw` to keep NorKyst's tides). The parent ζ is taken as is; `ib=1` adds the
//!    inverse-barometer level of `met=` to it (for a parent run without
//!    pressure forcing).
//!    Without an atlas: M2 of `M2_AMPLITUDE` in one phase along the boundary.
//!    NorKyst-800 has too little N2 here (0.027 m at Mausund against 0.156 m
//!    observed) and about twice the Q1, its diurnals are off (K1 1.14×,
//!    +8.6°; O1 0.95×, −17.3° at Mausund) and its S2 is 1.065×. The atlas is
//!    corrected with the first gauge's whole-record fit (`TidalAtlas::infer`):
//!    `gauge_gains=K1,O1,S2` (the default) scales each by the gauge's
//!    constant over NorKyst's at the gauge (`station_atlas=`), keeping
//!    NorKyst's spatial structure; then `gauge_ratios=N2,Q1,P1,K2` (the
//!    default) re-infers N2, Q1, P1 and K2 from M2 and the corrected O1, K1
//!    and S2 with the gauge's ratios. Over 15 days at Mausund this takes S2
//!    from 1.10× to 1.00× of the gauge (K2 is not separable from S2 in a
//!    15-day fit, so the boundary's K2 matters as much as its S2).
//! 5. **Validation.** Every `station_minutes` the surface and the
//!    depth-averaged current (east/north) are sampled at the tide gauges
//!    `gauges=` (files as written by `scripts/kartverket_gauge.sh`) and at
//!    the current records `currents=` (ADCP file format, depth-averaged). The
//!    DG polynomial is evaluated at the station itself (`PointLocator2D`,
//!    `Probe2D`) if its element is submerged (every node at least
//!    `STATION_MIN_DEPTH` deep), else at the nearest point of a submerged
//!    element: a coarse mesh leaves perched pockets in shoreline elements
//!    that hold water above the tide. After the run
//!    each station's series are written to the output directory
//!    (`station_*.txt`, `currents_*.txt`). Currents are fitted for tidal
//!    ellipses and compared with NorKyst-800's (from the velocity constants
//!    of `station_atlas=`; each gauge gets a second station at its nearest
//!    atlas point, where NorKyst's currents apply) and with the record's, by
//!    constituent and by the complex difference |ΔW|, plus time-series
//!    skill against the record. For the surface, the part after
//!    `spinup_hours` is fitted for reference constants and compared with the
//!    gauge (its whole record fitted, which also supplies the ratios of the
//!    constituents the run is too short to resolve) and with NorKyst-800 at
//!    the gauge (`station_atlas=`, from `norkyst_boundary_tides points=`):
//!    amplitude ratio, phase difference and complex difference per
//!    constituent, and RMSE against the observations and against the gauge's
//!    tidal prediction.
//!
//! Without the data files it runs a synthetic basin with an island and a
//! beach.
//!
//! ## Run
//!
//! ```bash
//! cargo run --release --example froya_real_data -- [nx=120] [ny=90] [order=2] \
//!     [hours=12.42] [rest_hours=1] [ramp_hours=1] [output_minutes=60] \
//!     [snapshot_minutes=0] [wind] \
//!     [start=2025-06-15T00:00:00Z] [tides=data/froya_boundary_tides.txt] [norkyst=<file>] \
//!     [gauges=data/tide_gauges/mausund_obs.txt] [currents=<file,…>] \
//!     [station_atlas=data/froya_station_tides.txt] \
//!     [station_minutes=10] [spinup_hours=24] [gauge_gains=K1,O1,S2] \
//!     [gauge_ratios=N2,Q1,P1,K2] [land_elevation=5] [manning=0.025] \
//!     [bed=projected|point] [dem=data/froya_topobathy.tif|none] [rx0=] [rx0_min_depth=3] \
//!     [wall_land=keep|lower] \
//!     [bbox=8.0,63.6,9.2,64.0] [lts=0] [rk=43|3] [cfl=] [output=output/froya] \
//!     [met=<file,…>] [heat=1] [jerlov=III] [mixing=k-epsilon] [band_km=3] [band_minutes=30] [blend=1] [max_closed=0.05] [ib=0] [nest_level=] \
//!     [nest_tides=corrected|raw] [waves=0] [wave_grid=25,36] [turning=] [implicit=0] \
//!     [wave_sources=wam|swan] [wave_integration=] [wave_limiter=] [wave_substeps=4|1] [wave_cfl=1.0|0.5] [wave_hours=0] \
//!     [wave_land=absorbing|floored] [wave_force=dissipation|stress] [wave_depth_limit=1] \
//!     [wave_reference=<file>] \
//!     [wave_mesh=NX,NY[,ORDER]] [wave_coupling=0] [wave_sea=2.5,10,285] \
//!     [wave_spectra=data/froya_wave_spectra.nc] [wave_neighbours=2] \
//!     [wave_stations=NAME:LON,LAT;…] \
//!     [levels=0] [tide3d=0] [rk_3d=43|3] [cfl_3d=4|2] [rk_bt=43|3] [cfl_bt=2] [bt_levels=2] \
//!     [restart_hours=0] [resume=<file>] [rivers=<file>] [river_snap_km=2] [river_top=0.2]
//! ```
//!
//! `waves=N` (N > 0) times N steps of the spectral wave model on the domain
//! instead of the tidal run (`wave_grid=frequencies,directions`, refraction
//! capped at `turning=` rad/s, or refraction and frequency shifting stepped
//! implicitly with `implicit=1`): the step, what sets it, and the cost per
//! model hour. With
//! `wave_mesh=NX,NY[,ORDER]` the waves run on an NX × NY grid of their own (P1
//! by default) over the same bed, and the coupling to the run's mesh
//! (`WaveCoupling2D`) is built and each exchange timed. `wave_hours=H` runs on
//! after the timed steps to H model hours; the sea at every node is then
//! written to `<output>/wave_nodes.txt`, and `wave_reference=<file>` compares
//! it node by node with an earlier run's (H_s by depth, T_m01, the mean
//! direction): how a step or a scheme changes the sea.
//!
//! `wave_coupling=MINUTES` runs the waves with the 2D tide, two-way coupled
//! every MINUTES (`CoupledWaves2D`, `Simulation::run_with_exchange`; a
//! multiple of `station_minutes`): the waves take the tide's level and
//! current, and give it their radiation-stress force (linear in time across
//! each interval) and Soulsby's enhancement of its Manning friction, both
//! ramped up over `ramp_hours`. On the run's mesh, or on `wave_mesh=`. The
//! sea `wave_sea=HS,TP,FROM` (JONSWAP, H_s in m, T_p in s, coming from FROM
//! degrees) comes in through the open boundaries and fills the domain at the
//! start. The wind of `met=` blows on the waves too, per wave node and not
//! ramped (`GriddedAtmosphere2D::on_mesh`, `CoupledWaves2D::with_gridded_wind`);
//! without it, the uniform `wind`.
//! Every output interval a line of the waves (the largest H_s, the mean over
//! water ≥ 3 m deep, the largest force, the mean wave step and its cost, and
//! what sets the step where) comes before the tide's. Use `implicit=1`: the
//! tide's currents over the shallows otherwise set the waves' step by
//! frequency shifting (0.6 s against the geographic 7 s at 1 km).
//!
//! `wave_sources=wam` (the default) integrates the sources as WAM does:
//! implicit in the DIA's diagonal, with Hersbach & Janssen's growth limiter
//! proportional to the step (`SourceIntegration::Implicit`,
//! `GrowthLimiter::Rate`); `wave_sources=swan` takes SWAN's frozen rates and
//! per-step limiter instead. `wave_integration=frozen|implicit` and
//! `wave_limiter=ris|rate|rate2` set either part alone (`rate2`: the limit on
//! the losses too, as ecWAM; breaking stays outside it). WAM's sources
//! tolerate long steps, so `wave_substeps=N` runs them once per N
//! propagation steps (each with its own implicit refraction, shifting and
//! breaking; 4 with WAM's sources, 1 with SWAN's), and `wave_cfl=` sets the
//! propagation's CFL number (1.0 with WAM's, 0.5 with SWAN's). WAM's won the
//! Ålesund buoy test of `notes/waves.md` (2026-10-20: H_s scatter index
//! 0.271 against SWAN's 0.284 over the sheltered E39 buoys, a tie by the
//! pre-set rule, and 0.32 against 0.36 on a 1 km wave grid) at half the cost.
//!
//! `wave_land=absorbing` (the default) lets dry land take the waves, as the
//! coast does (`WaveModel2D::with_absorbing_land`). `wave_land=floored` treats
//! it as water of the minimum depth instead, where the waves pile up until the
//! sinks there remove them.
//!
//! `wave_depth_limit=1` (the default) scales each wave node's spectrum down
//! after every step to `H_rms ≤ γ d`, breaking's `H_max`
//! (`SourceTerms::with_depth_limit`): on the 1 km wave grid, nodes in a few
//! decimetres of water next to deep ones otherwise hold metres of H_s, which
//! Battjes–Janssen's saturated dissipation cannot remove. `wave_depth_limit=0`
//! leaves them. The waves' line reports the largest `H_s/d` and how many
//! nodes are at the limit, by depth. At the end of the run the sea at every
//! wave node goes to `<output>/wave_nodes.txt`, and `wave_reference=<file>`
//! compares it node by node with an earlier run's, as with `waves=N`.
//!
//! `wave_force=dissipation` (the default) drives the circulation by the
//! momentum the waves' sinks take over a wave step
//! (`WaveForceForm::Dissipation`, Dingemans et al. 1987): the wave-driven
//! currents and the setup where the waves break, without the set-down, and
//! without the unresolved shoaling's `−∇·S` that drove 6–10 m/s along the
//! cliff coasts in the 2025-10-09 storm. `wave_force=stress` takes the
//! radiation stress's divergence instead.
//!
//! `wave_spectra=<file>` takes the sea from a parent wave model instead:
//! MET Norway's MyWave WAM 800 m spectra at its points
//! (`examples/met_wave_subset.rs` fetches the latest forecast's points
//! around the domain; `io::WaveSpectraFile`). Each open-boundary node of the
//! waves takes the inverse-distance mean of its `wave_neighbours` nearest
//! points, regridded onto the waves' grid and linear in time, at every wave
//! step (`waves::BoundarySpectra`); the waves start from the boundary's mean
//! spectrum everywhere and take the parent's wind (the mean of its points',
//! `waves::WindSeries`), unless `met=` gives them the weather's. The run's clock (`start=`) must fall in the file's
//! times. Every output interval our H_s at the parent's points is printed
//! against the parent's, and at the end the bias and RMSE after twice
//! `ramp_hours`, with the series in `<output>/wave_points.txt` (H_s ours
//! and the parent's, and our T_p and mean direction, at each point's
//! nearest wave node). `wave_stations=NAME:LON,LAT;…` adds points of our
//! own there, e.g. wave buoys (`scripts/e39_buoy_series.py`), which have
//! no parent value. For a
//! storm with its weather, fetch MET Nordic for the same days
//! (`met_forcing_subset`) and pass it as `met=`.
//!
//! `levels=N tide3d=1` runs the tide in 3D instead (`tidal_run_3d`, TODO P1.3):
//! the 2D run's open boundary (NorKyst nesting with `norkyst=`, else the atlas
//! tides) drives the barotropic mode, and with `norkyst=` the same file, read
//! with its profiles (`norkyst_nesting_subset profiles=1`), gives the initial
//! T and S at every node, the initial velocity's shear (NorKyst's profile
//! less its depth mean: the tide's depth mean ramps up from rest), and nests
//! u, v, T, S in the band (`Nesting3D`, `OceanModelColumns`); without it the
//! open faces relax to the stratification at rest (the summer pycnocline).
//! The PGF, the tracer limiter and the vertical advection take the domain's
//! mean initial profile as their reference. GLS k-ε, log-layer bottom drag
//! (no 2D Manning friction), Smagorinsky viscosity; the vertical advection
//! beyond an explicit Courant number implicit
//! (`Hydrostatic3D::with_implicit_vertical_advection`). `wind` and `met=` as
//! in 2D: the pressure gradient on the 2D module, the wind stress on the
//! columns (`GriddedAtmosphere2D::split_for_3d`). When the `met=` files also
//! carry the surface heat budget (2 m air temperature and humidity,
//! downwelling shortwave and longwave, precipitation; `met_forcing_subset`
//! writes them), the columns take the COARE 3.5 heat and freshwater fluxes
//! of their own surface (`GriddedAtmosphere2D::surface_fluxes`,
//! `Hydrostatic3D::with_surface_fluxes`), the shortwave absorbed as Jerlov's
//! `jerlov=` water type (III, NorKyst v3's); `heat=0` leaves them out. The vertical mixing
//! is GLS with Canuto A stability functions, `mixing=k-epsilon` (default),
//! `k-omega` or `generic` (Umlauf & Burchard's, NorKyst v3's "gen";
//! Christensen et al. 2026). `rivers=<file>` (from `norkyst_river_subset`:
//! NorKyst-800's NVE rivers, daily) adds the rivers as volume sources with
//! their water's temperature and salinity (`RiverForcing`,
//! `Hydrostatic3D::with_rivers`), each in the element of its mouth or snapped
//! onto the mesh from up to `river_snap_km` away (rivers farther off are listed
//! and left out), entering over the top `river_top` of the column. The stations, the progress
//! lines and a 3D snapshot file (`snapshot_minutes=`,
//! `SnapshotWriter::create_3d`) as in 2D; `dt_3d=` caps the step (60 s by
//! default; the parent and the weather must cover a third of it past the
//! end, where the last step's barotropic window reads them,
//! `ModeSplitIntegrator::FORCING_REACH`) and `cfl_3d=` sets its Courant
//! number (`Hydrostatic3D::compute_dt`; 4 with the SSP-RK(4,3) stages, 2
//! with SSP-RK3's, see `rk_3d=` below).
//! Measured 2026-10-10: on the Mausund tide (3 h, nested) 2 holds and 3
//! blows up within 0.8 h (a 50 m element during the ramp, at 25 s steps);
//! against 0.5, 1 moves ū by 0.3 mm/s RMS and 2 by ≤ 3 mm/s, T by 0.013
//! and 0.019 °C RMS (565 → 246 → 148 s per model hour). The full coastline
//! mesh blew up at 1.5 within 1.3 h in a cliff element (1.2 m beside 82 m),
//! now land (`cliffs3d`). Measured 2026-10-15 on the coastline day (hours
//! 6–8, `notes/3d-numerics.md`): 2 holds, with 0.1 mm RMS of η, 0.2 mm/s of
//! ū and 2 mm/s of the surface velocity against 1 (0.3 mm and 0.4 mm/s at
//! Mausund), and T spreads by 0.01 °C RMS as it does under any perturbation
//! (the barotropic pass alone gives the same). The barotropic substeps are
//! SSP-RK(4,3) (`rk_bt=43`; `ModeSplitIntegrator::with_barotropic_scheme`)
//! at `cfl_bt=` (2), above P2's capped linear limit (1.36), so the 2D
//! module's per-element positivity and linear bounds set them: shoreline
//! elements with a dry node take twice SSP-RK3's step there, half the
//! substeps for 4/3 of the work each. Together 773 → 388 s per model hour
//! on hours 6–8 (24 threads), 0 negative-depth clips. The barotropic pass
//! takes local time steps of up to `bt_levels=` (2) levels
//! (`ModeSplitIntegrator::with_barotropic_levels`; 0 for global substeps):
//! the cliff-shore elements that set the substep take it alone, the rest 2
//! or 4 times longer ones, and the run reports the RHS work saved. On hours
//! 6–8 against global substeps (2026-10-16): η 0.07 mm, ū 0.16 mm/s, T
//! 0.011 °C RMS at the nodes ≥ 10 m deep (the run's sensitivity), the pass
//! 472 → 236 ms per step. Use `slopes3d=on` for the stratified rest state on
//! steep beds.
//!
//! The 3D stages are SSP-RK(4,3) (`rk_3d=43`, the default;
//! `ModeSplitIntegrator::with_baroclinic_scheme`) at `cfl_3d=4`, or SSP-RK3
//! (`rk_3d=3`) at `cfl_3d=2`, the default before 2026-10-19.
//! The 3D step is set by shoreline elements whose tracers move as subcell
//! finite volumes, a forward-Euler bound that SSP-RK(4,3) doubles, and the
//! vertical advection's bound and implicit split double with it. Measured
//! 2026-10-19 from the 70 h restart of the 72 h MET-wind run
//! (`notes/3d-numerics.md` "Third depth pass"): 294 → 216 s per model hour.
//! With snapshots (`snapshot_minutes=`) the steps land on the snapshot times
//! (`Simulation3D::with_landing_interval`), so `snapshot_compare_3d`
//! compares runs with different steps frame by frame. `debug_3d=dt_dist`
//! prints, at every progress line, every element's own 3D step (quantiles,
//! what local time stepping in 3D could save, the five shortest and what
//! sets them). The 72 h MET-wind run with heat fluxes took 199 s per model
//! hour against 267 (SSP-RK3 at 2), with the same tide, currents and T/S
//! profiles against NorKyst to 0.01 °C and 1 mm/s. `debug_3d=visc_step`
//! takes the horizontal viscosity once per step
//! (`Hydrostatic3D::with_viscosity_once_per_step`): 11 % faster there, but
//! the NorKyst point's current moves 5× further from the reference.
//!
//! A long 3D tide can stop and resume bit for bit. `restart_hours=H` writes
//! `<output>/<domain>.restart` (`Restart3D`, replaced in one rename, so a
//! crash while writing keeps the previous one) at the first progress line at
//! or after every H model hours. It holds the state, the mode splitter's slow
//! forcing, the vertical reference, the station records and the sampling
//! counters. `resume=<file>`, with the same other options (the domain's
//! fingerprint is checked), builds the run as before from its initial state,
//! then continues from the file's time to `hours=`: the snapshot file drops
//! the frames written after the restart and goes on, and the stations' report
//! covers the whole run. A resumed run's outputs equal the uninterrupted
//! run's, byte for byte.
//!
//! `levels=N` (N > 0) times `steps_3d=` (10) mode-split steps of the 3D model
//! on the domain instead of the tidal run: N surface-stretched σ-levels, GLS
//! k-ε, log-layer bottom drag, Smagorinsky viscosity, the Kuzmin limiter, a
//! summer pycnocline at rest. It reports the baroclinic step and what sets it
//! (and what local time stepping in 3D could save), the barotropic substeps,
//! and the wall time per step and per model hour. `dt_3d=` overrides the step;
//! `debug_3d=` takes comma-separated switches for diagnosing the 3D model on
//! the real bed: `trace` (the largest speed and where, every step, and the
//! momentum tendency at rest), `uniform` (no stratification), `linear`
//! (linear in z instead of the pycnocline), `balanced` (the initial state as
//! the PGF's balanced reference), `nolimiter`, `constant` (constant mixing
//! instead of GLS), `thin=D` (the 3D thin-column depth, 0.1 m by default),
//! `vcentred` (centred vertical tracer advection), `nu=V` (a constant
//! horizontal viscosity of the shear, m²/s, added to Smagorinsky's),
//! `around=K:R` (only the elements within R m of element K, walls around:
//! a local growth reproduced in minutes), `every=N` (trace every N steps),
//! `dump=PREFIX` (the state at rest, `gap=N` steps before the end and at the
//! end, for `scripts/mode3d_dump.py`), `seed=PREFIX:AMP` (start from rest
//! plus a dumped mode at a largest speed of AMP m/s), `perturb=A` (a
//! deterministic random T perturbation of up to ±A/2 °C at every wet point,
//! so that modes with e-foldings of hours show within a run rather than
//! after round-off has grown for a day), `deep=G` (°C/m below
//! the pycnocline, 0.002), `vadv=centred|akima|tvd|upwind|hermite` (the
//! vertical tracer scheme), `noref` (that scheme without the reference of
//! the state at rest, which 3D runs take by default; TODO P1.3),
//! `tadv=none|centred` (the turbulence's advection) and
//! `export=PATH` (with `around=`: the patch as a test fixture).
//!
//! `snapshot_minutes=N` (N > 0) also writes the state every N minutes to
//! `<output>/froya.dgsnap` (`io::SnapshotWriter`: f32 η, u, v per node, with
//! the mesh, bed, clock, stations and projection), a tenth of the VTU frames'
//! size. The viewer replays it without rebuilding the domain, with the land
//! around it from the elevation model:
//! `cd viz && cargo run --release -- --replay ../output/froya/froya.dgsnap`.
//! With stations, N is rounded to a multiple of `station_minutes`. With
//! `wave_coupling=` its frames also carry the sea state at every node
//! (`waves::SeaStateFields`: H_s, T_p, the mean direction, the spread, and
//! the power whitecapping and breaking take), linear in time between the
//! exchanges, for the viewer's wave colourings, arrows and photo view.
//!
//! `rx0=r` smooths the bed, keeping its volume, until the slope factor
//! r_x0 = |h₁ − h₂|/(h₁ + h₂) between neighbouring nodes at least
//! `rx0_min_depth` (3 m) deep is at most r (`Bathymetry2D::smooth_rx0`):
//! shoals a node wide otherwise carry spurious m/s currents.
//!
//! `wall_land=lower` lowers the land at the walls of a coastline mesh
//! (`mesh=`) in elements that are otherwise water to the water beside it
//! (`Bathymetry2D::lower_wall_land`): the simplified coastline leaves a strip
//! of the elevation model's land a fraction of a node spacing wide in front
//! of two thirds of its walls, and every element with such a strip is partly
//! dry at every water level. Only elements wet off their walls below the
//! lowest water (2 m below MSL) are lowered. Off by default.
//!
//! `slopes3d=on` (for 3D runs, `levels=N`; in a 2D run it shows what the 3D
//! bed does to the tide) smooths the bed, keeping its volume,
//! until no element's depth range lets a σ-level cross the summer
//! pycnocline between two of its nodes (`Bathymetry2D::smooth_element_slopes`,
//! TODO P1.3): within an element r_x0 ≤ 0.15 (`slopes3d=r` for another bound)
//! between columns that are not thin (`debug_3d=thin=`, 0.1 m) wherever the
//! deeper one is below `free_depth` (19 m, the pycnocline's bottom).
//! Shores need none (the 3D wetting and drying moves their layers on the 2D
//! kernel's subcells) and do not move. Off by default. With it, `cliffs3d=D`
//! raises to land (+2 m) the shore nodes of elements with a node deeper than
//! D m (20 by default; `cliffs3d=off`) that a tide would make resolved
//! columns beside the deep ones (`Bathymetry2D::raise_cliff_shores`): a
//! 0.74 m corner among 61–82 m held the 3D step below 0.4 s for hours at
//! each high water, and at 40 m two shores of 0.3–0.5 m beside 24–38 m
//! blew up the 72 h run with MET wind on day 2. And `ponds3d=L` fills to their sills the tidal ponds
//! of at most one element's nodes at water levels between L m (the lowest
//! astronomical tide, −1.43 m, by default; `ponds3d=off`) and +2 m: nodes
//! the open water does not reach along the grid lines
//! (`Bathymetry2D::fill_tidal_ponds`), where a 25 m single-node pond held a
//! 3.7 m/s jet through an ebb and a 2.2 m one set the 3D step.
//!
//! `bbox=west,south,east,north` runs a smaller box, e.g. the 22 × 19 km
//! around Mausund (a quarter of the cost), with its own mesh and boundary
//! atlas (use `tide_transport=3` there): see "Mausund sub-domain" in
//! `docs/gmsh-meshes.md`.
//!
//! `lts=N` (N > 0) steps the tidal run with local time stepping
//! (`Multirate`, up to N levels of halved time steps): every element at
//! the largest power-of-two fraction of the step its own CFL allows.
//!
//! `rk=43` (the default) steps with SSP-RK(4,3), `rk=3` with SSP-RK3. The
//! positivity bound of the wet/dry scheme sets the step where elements may
//! run dry, and SSP-RK(4,3)'s SSP coefficient of 2 doubles it for 4/3 of the
//! RHS work per step. The bound is relaxed per element by how its water is
//! distributed, up to `cfl=`, by default 0.9 of the integrator's linear
//! stability limit (`linear_cfl_swe_2d`: 1.36 at P2 with SSP-RK(4,3)), and
//! in shoreline elements up to 0.9 of the subcells' lower limit
//! (`linear_cfl_subcells_swe_2d`: 0.99 at P2).
//!
//! Harmonic validation needs the record after spin-up to resolve the main
//! constituents: 15 days separate M2/S2 and K1/O1 (`hours=384` with the
//! default spin-up); N2 needs 28.
//!
//! ## Data files in ./data/
//!
//! - froya_topobathy.tif (Kartverket's topobathy model: land heights and
//!   depths at 50 m, from `scripts/kartverket_topobathy.sh`; used when present)
//! - froya_smola_hitra.tif (bathymetry) and GSHHS_f_L1.shp (coastline):
//!   without the topobathy model, or with `dem=none`
//! - froya_boundary_tides.txt (tidal atlas, optional)
//! - tide_gauges/mausund_obs.txt, froya_station_tides.txt (validation, optional)

use std::cell::RefCell;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use dg_rs::analysis::{
    ADCPStation, ADCPValidationResult, AnalysisError, ConstituentComparison, CurrentTimeSeries,
    EllipseFit, HarmonicAnalysis, Inference, ReferenceConstant, ReferenceFit,
    StationValidationResult, TidalEllipse, TideGaugeStation, TimeSeries,
};
#[cfg(feature = "netcdf")]
use dg_rs::boundary::NestingOptions;
use dg_rs::boundary::{
    AtlasPoint, BCContext2D, BoundaryCoverage, BoundaryLevel, BoundaryTides, CharacteristicOBC,
    ExternalStateProvider, HarmonicTide, InverseBarometer, MultiBoundaryCondition2D,
    NestingRelaxation2D, OceanModelState, Reflective2D, SWEBoundaryCondition2D, TidalAtlas,
};
use dg_rs::equations::ShallowWater2D;
#[cfg(feature = "netcdf")]
use dg_rs::io::WaveSpectraFile;
#[cfg(feature = "netcdf")]
use dg_rs::io::{AtmosphereReader, NetCDFMeshInfo, NetCDFWriter, NetCDFWriterConfig};
use dg_rs::io::{
    BedRaster, CoastlineData, CoordinateProjection, GeoBoundingBox, GeoTiffBathymetry,
    LocalProjection, OceanModelReader, SnapshotLayout, SnapshotWriter, TideGaugeFile, east_axis,
    read_adcp_file, read_tide_gauge_file, write_adcp_file, write_tide_gauge_file, write_vtk_swe,
};
use dg_rs::mesh::{
    Bathymetry2D, BoundaryTag, ElementSlopeBound, Mesh2D, MeshPoint, PointLocator2D,
    inverse_bilinear, read_gmsh_mesh,
};
use dg_rs::operators::{DGOperators2D, GeometricFactors2D};
use dg_rs::physics::{
    Coare35, GlsMixing, JerlovWater, PhysicsBuilder, PhysicsModule, SWEPhysics2D,
    SWEPhysics2DBuilder,
};
use dg_rs::simulation::{Simulation, SimulationResult};
use dg_rs::solver::{
    LINEAR_CFL_SAFETY, Probe2D, SWESolution2D, SWEState2D, StandardLimiter2D, WetDryConfig,
    linear_cfl_swe_2d, positivity_cfl_swe_2d,
};
use dg_rs::source::{
    AtmosphericPressure2D, CoriolisSource2D, DragCoefficient, GriddedAtmosphere2D,
    GriddedSurfaceFluxes, GriddedWindStress, ManningFriction2D, WindStress2D,
};
use dg_rs::tides::canonical_name;
use dg_rs::time::{
    IntegratorInfo, ModelClock, Multirate, SSPRK3, SspScheme, StandardIntegrator, TimeIntegrator,
};
#[cfg(feature = "netcdf")]
use dg_rs::types::Depth;
use dg_rs::types::ElementIndex;
use dg_rs::waves::{BoundarySpectra, PointSpectra, WindSeries};
use dg_rs::waves::{
    CoupledWaves2D, CoupledWavesStats, DEFAULT_RATE_LIMITER, GrowthLimiter, SeaStateFields,
    SourceIntegration, SourceTerms, SpectralGrid, WaveForceForm, WaveModel2D, Wind,
};

/// Gravitational acceleration (m/s²)
const G: f64 = 9.81;
/// Coriolis parameter at 63.8°N (s⁻¹)
const F_CORIOLIS: f64 = 1.31e-4;
/// Manning roughness (s/m^{1/3}; `manning=`)
const MANNING_N: f64 = 0.025;
/// Default bed elevation given to land nodes of shoreline elements (m above
/// MSL; `land_elevation=`)
const LAND_ELEVATION: f64 = 5.0;
/// Lowest water for `wall_land=lower` (m above MSL): below the lowest
/// astronomical tide at Mausund (chart datum, 1.43 m below MSL), so that the
/// elements it lowers stay wet through the tide
const WALL_LOW_WATER: f64 = -2.0;
/// The thin-column depth of a 3D tide (m; `Options::thin_depth`)
const TIDE_3D_THIN_DEPTH: f64 = 1.0;
/// Highest water for `cliffs3d=` (m above MSL), above the highest
/// astronomical tide at Mausund; the shore nodes it raises go to this height
const CLIFF_HIGH_WATER: f64 = 2.0;
/// Default depth (m) of `cliffs3d=`: at 40 m (the x–z slices let shallow
/// water beside 21–40 m hold for a day) the 72 h run with MET wind blew up at
/// 37 h from shores of 0.3–0.5 m beside 24–38 m; at 20 m it held the 72 h
const CLIFF_DEEP: f64 = 20.0;
/// Longest step of a 3D tide (s) unless `dt_3d=` sets it: it does not bind
/// on the coastline mesh (10–18 s steps), and it bounds how far past the end
/// the last step reads its forcing (`ModeSplitIntegrator::FORCING_REACH`)
const MAX_DT_3D: f64 = 60.0;
/// Default level of `ponds3d=` (m above MSL): the lowest astronomical tide at
/// Mausund (chart datum), the lowest water any run sees
const POND_LOW_WATER: f64 = -1.43;
/// Default elevation model: Kartverket's topobathy model at 50 m
/// (`scripts/kartverket_topobathy.sh 8.0 63.6 9.2 64.0 50 data/froya_topobathy.tif`)
const DEM: &str = "data/froya_topobathy.tif";
/// Frøya–Smøla–Hitra: west, south, east, north (°)
const FROYA_BBOX: [f64; 4] = [8.0, 63.6, 9.2, 64.0];
/// Default depth (m) below which `rx0=` leaves the bed alone: the shore and
/// the dry area keep their shape
const RX0_MIN_DEPTH: f64 = 3.0;
/// Depth (m) of the bottom of the 3D runs' summer pycnocline (T and S step as
/// `tanh((z + 15)/4)`), for the bound of `slopes3d=`
const PYCNOCLINE_BOTTOM: f64 = 19.0;
/// Land-mask cells per bathymetry pixel and direction: the coastline is
/// rasterised at ≈ 25 × 58 m
const LAND_MASK_REFINEMENT: usize = 4;

/// M2 period (s) and a typical amplitude on this coast (m)
const M2_PERIOD: f64 = 12.420_601 * 3600.0;
const M2_AMPLITUDE: f64 = 0.8;
/// Default tidal ramp-up (h)
const TIDAL_RAMP_HOURS: f64 = 1.0;
/// Largest distance (m) from an open-boundary node to a tidal-atlas point
const ATLAS_COVERAGE: f64 = 5000.0;

/// Wind (m/s, from °) and atmospheric pressure gradient (Pa/m, from °)
const WIND_SPEED: f64 = 8.0;
const WIND_DIRECTION: f64 = 225.0;
const PRESSURE_GRADIENT: f64 = 1.5e-3;
const PRESSURE_DIRECTION: f64 = 225.0;

/// A station samples the model at its position if every node of its element
/// is at least this deep (m below MSL), so that it is in open water that
/// stays wet through the tide; else at the nearest point of such an element
const STATION_MIN_DEPTH: f64 = 3.0;
/// ... no farther than this from the station (m)
const STATION_MAX_OFFSET: f64 = 3000.0;
/// Constituents fitted at the stations, in priority order (a short record
/// keeps the first ones): the forcing's, without the long-period ones
const STATION_CONSTITUENTS: [&str; 12] = [
    "M2", "S2", "K1", "O1", "N2", "Q1", "K2", "P1", "M4", "MS4", "MN4", "M6",
];
/// Constituents fitted to a gauge's whole record: long-period ones too, so
/// that the seasonal and fortnightly signal does not leak into the tides
const GAUGE_CONSTITUENTS: [&str; 15] = [
    "M2", "S2", "K1", "O1", "N2", "Q1", "K2", "P1", "M4", "MS4", "MN4", "M6", "Mf", "Mm", "Ssa",
];

/// Command-line options (`key=value`, or a bare flag)
#[derive(Clone)]
struct Options {
    nx: usize,
    ny: usize,
    order: usize,
    hours: f64,
    rest_hours: f64,
    ramp_hours: f64,
    output_minutes: f64,
    /// Minutes between frames of the snapshot file (`snapshot_minutes=`; 0: none)
    snapshot_minutes: f64,
    wind: bool,
    start: String,
    tides: String,
    #[cfg_attr(not(feature = "netcdf"), allow(dead_code))]
    norkyst: Option<String>,
    /// Weather-model files (`met=a.nc,b.nc`), joined in time
    #[cfg_attr(not(feature = "netcdf"), allow(dead_code))]
    met: Vec<String>,
    /// Nesting: relaxation band width (km) and timescale at the boundary
    /// (min), whether to blend the bed to the parent's across the band, and
    /// whether to add the inverse-barometer level to the parent's ζ
    #[cfg_attr(not(feature = "netcdf"), allow(dead_code))]
    band_km: f64,
    #[cfg_attr(not(feature = "netcdf"), allow(dead_code))]
    band_minutes: f64,
    #[cfg_attr(not(feature = "netcdf"), allow(dead_code))]
    blend_bed: bool,
    /// The largest share of the open faces the atlas or the parent may
    /// leave uncovered (they become walls)
    max_closed: f64,
    nesting_ib: bool,
    /// In 3D, the bulk heat and freshwater fluxes of `met=` when it has the
    /// fields (`heat=0`: none)
    #[cfg_attr(not(feature = "netcdf"), allow(dead_code))]
    heat: bool,
    /// The optical water type of the shortwave absorption (`jerlov=III`)
    jerlov: JerlovWater,
    /// The GLS closure of the 3D columns (`mixing=k-epsilon`): its name and
    /// constructor
    mixing: (&'static str, fn() -> GlsMixing),
    /// Nesting: replace NorKyst's `gauge_gains` and `gauge_ratios`
    /// constituents by the gauge-corrected ones (`nest_tides=corrected`, the
    /// default) or keep its tides (`nest_tides=raw`)
    correct_nested_tides: bool,
    /// Added to NorKyst's ζ (m); default: minus the mean level `Z0` of the
    /// boundary atlas
    #[cfg_attr(not(feature = "netcdf"), allow(dead_code))]
    nesting_level: Option<f64>,
    gauges: Vec<String>,
    /// Current records (`currents=a.txt,b.txt`, ADCP file format:
    /// depth-averaged east/north velocity)
    currents: Vec<String>,
    station_atlas: String,
    station_minutes: f64,
    spinup_hours: f64,
    /// Atlas constituents scaled by the gauge's constants over NorKyst's at
    /// the gauge (`gauge_gains=K1,O1,S2`), before the ratio inference
    gauge_gains: Vec<&'static str>,
    /// Atlas constituents re-inferred from their neighbour with the gauge's
    /// ratio (`gauge_ratios=N2,Q1,P1,K2`)
    gauge_ratios: Vec<&'static str>,
    land_elevation: f64,
    /// Manning roughness of the 2D bed friction (`manning=`, s/m^{1/3})
    manning: f64,
    /// Elevation model with land heights (`dem=`, used if the file exists;
    /// `dem=none` for the GeoTIFF bathymetry and GSHHS coastline)
    dem: Option<String>,
    /// L2-project the bed onto the nodes (`bed=projected`, the default) or
    /// sample it there (`bed=point`). Point samples of the 50 m raster at
    /// nodes 500 m apart land on skerries at random and close sounds: at
    /// 1 km the M2 current near Mausund was 0.41 of NorKyst's with point
    /// samples and 0.63 projected, the gauge's centred RMSE 4.4 and 4.0 cm
    project_bed: bool,
    /// Lower the land at the coastline walls of elements that are otherwise
    /// water to their water (`wall_land=lower`, `Bathymetry2D::lower_wall_land`)
    /// or keep the elevation model's (`wall_land=keep`, the default)
    lower_wall_land: bool,
    /// Coastline-fitted Gmsh mesh (`mesh=`, from
    /// `scripts/gmsh_coastline_mesh.py`) instead of the `nx` × `ny` grid
    mesh: Option<String>,
    /// Local time stepping with up to this many levels (`lts=`; 0: global
    /// steps)
    lts: usize,
    /// SSP-RK(4,3) (`rk=43`, default) or SSP-RK3 (`rk=3`), globally or as
    /// the base of local time stepping
    integrator: StandardIntegrator,
    /// CFL number of the tidal run (`cfl=`): its linear stability bound. The
    /// default is `LINEAR_CFL_SAFETY` (0.9) times the integrator's linear
    /// limit (`linear_cfl_swe_2d`); the wet/dry positivity bound caps it
    /// where elements may run dry
    cfl: f64,
    /// Smooth the bed to a slope factor r_x0 ≤ this between neighbouring
    /// nodes deeper than `rx0_min_depth` (`rx0=`, e.g. 0.3; off by default)
    rx0: Option<f64>,
    rx0_min_depth: f64,
    /// For 3D runs (`levels=`), smooth the bed until every element is within
    /// this bound (`Bathymetry2D::smooth_element_slopes`): `slopes3d=on` (r_x0
    /// 0.15) or `slopes3d=r`, off by default; `free_depth=` sets the depth
    /// above which elements are free (1.5 × the summer pycnocline's bottom by
    /// default)
    slopes_3d: Option<ElementSlopeBound>,
    /// With `slopes3d`, raise the shore nodes of elements deeper than this
    /// (m) to land (`Bathymetry2D::raise_cliff_shores`): `cliffs3d=D`, 20 m
    /// by default, `cliffs3d=off`
    cliffs_3d: Option<f64>,
    /// With `slopes3d`, fill to their sills the tidal ponds of at most one
    /// element's nodes at water levels from this one (m above MSL) to +2 m
    /// (`Bathymetry2D::fill_tidal_ponds`): `ponds3d=L`, the lowest
    /// astronomical tide (−1.43 m) by default, `ponds3d=off`
    ponds_3d: Option<f64>,
    /// Domain box `west,south,east,north` (°; `bbox=`). A smaller box needs
    /// its own boundary atlas (`tides=`, from `norkyst_boundary_tides bbox=`)
    bbox: [f64; 4],
    /// Scale the atlas velocity by NorKyst's total depth over the child's,
    /// by up to this factor (`tide_transport=`; 0: off)
    tide_transport: f64,
    profile: usize,
    /// Time this many steps of the spectral wave model on the domain instead
    /// of the tidal run (`waves=N`; 0: off), on `wave_grid=frequencies,directions`
    /// with refraction capped at `turning=` rad/s (none by default)
    waves: usize,
    wave_grid: [usize; 2],
    turning: Option<f64>,
    /// Step refraction and frequency shifting implicitly (`implicit=1`)
    implicit_refraction: bool,
    /// The sources implicit in the DIA, WAM's integration
    /// (`wave_integration=implicit`), instead of SWAN's frozen rates
    /// (`frozen`); `wave_sources=wam` (the default) sets it with the limiter below
    wave_implicit_sources: bool,
    /// Hersbach & Janssen's growth limiter, proportional to the step
    /// (`wave_limiter=rate`; `rate2` caps the losses too, as ecWAM), instead
    /// of Ris's per step (`ris`, SWAN's sources keep it)
    wave_limiter: Option<GrowthLimiter>,
    /// Propagation steps per wave step (`wave_substeps=`)
    wave_substeps: usize,
    /// The waves' propagation CFL number (`wave_cfl=`)
    wave_cfl: f64,
    /// Dry land absorbs the waves (`wave_land=absorbing`, the default)
    /// instead of being water of the minimum depth (`floored`)
    wave_absorbing_land: bool,
    /// The waves' force on the circulation (`wave_force=dissipation|stress`)
    wave_force: WaveForceForm,
    /// Each wave node's sea held to `H_rms ≤ γ d` (`wave_depth_limit=1`, the
    /// default)
    wave_depth_limit: bool,
    /// With `waves=N`, run on after the N timed steps to this many model
    /// hours, the last step landing on it (`wave_hours=`; 0: off)
    wave_hours: f64,
    /// With `waves=N`, compare the sea at every node with an earlier run's
    /// `wave_nodes.txt` (`wave_reference=<file>`)
    wave_reference: Option<PathBuf>,
    /// The waves on a grid of their own (`wave_mesh=NX,NY[,ORDER]`, order 1
    /// by default) instead of the run's mesh, coupled to it by
    /// `WaveCoupling2D`
    wave_mesh: Option<[usize; 3]>,
    /// Run the waves with the tide, two-way coupled every this many minutes
    /// (`wave_coupling=`; 0: off)
    wave_coupling: f64,
    /// The sea at the open boundaries and, to start with, everywhere:
    /// JONSWAP of H_s (m) and T_p (s) coming from (degrees, nautical)
    /// (`wave_sea=HS,TP,FROM`)
    wave_sea: [f64; 3],
    /// A parent wave model's point spectra for the open boundary and its
    /// wind for the waves, instead of `wave_sea` (`wave_spectra=<file>`,
    /// from `met_wave_subset`)
    wave_spectra: Option<PathBuf>,
    /// Points where the waves are reported besides the parent's, with
    /// `wave_spectra=` (`wave_stations=NAME:LON,LAT;…`)
    #[cfg_attr(not(feature = "netcdf"), allow(dead_code))]
    wave_stations: Vec<(String, f64, f64)>,
    /// Each open-boundary node takes the inverse-distance mean of this many
    /// nearest parent points (`wave_neighbours=`)
    wave_neighbours: usize,
    /// Time the 3D model on the domain with this many σ-levels instead of
    /// the tidal run (`levels=N`; 0: off), for `steps_3d=` steps
    levels: usize,
    /// Run the tide in 3D with `levels` σ-levels (`tide3d=1`) instead of
    /// timing steps of it
    tide_3d: bool,
    steps_3d: usize,
    dt_3d: Option<f64>,
    /// Courant number of the 3D step (`cfl_3d=`, `Hydrostatic3D::compute_dt`;
    /// 2, see the module docs)
    cfl_3d: f64,
    /// Courant number of the 3D tide's barotropic substeps (`cfl_bt=`,
    /// `ModeSplitIntegrator::with_barotropic_cfl`)
    cfl_bt: f64,
    /// SSP scheme of the barotropic substeps (`rk_bt=3|43`)
    rk_bt: dg_rs::time::SspScheme,
    /// SSP scheme of the 3D stages (`rk_3d=3|43`)
    rk_3d: dg_rs::time::SspScheme,
    /// Most levels of local time stepping in the barotropic pass
    /// (`bt_levels=`, 2; 0: global substeps)
    bt_levels: usize,
    debug_3d: String,
    /// Write a restart of the 3D tide every this many model hours
    /// (`restart_hours=`; 0: none), at the first progress line at or after it
    restart_hours: f64,
    /// Resume the 3D tide from this restart file (`resume=`)
    resume: Option<PathBuf>,
    /// Rivers of the 3D tide (`rivers=<file>`, from `norkyst_river_subset`),
    /// snapped onto the mesh from up to `river_snap_km` away, entering over
    /// the top `river_top` of the column
    #[cfg_attr(not(feature = "netcdf"), allow(dead_code))]
    rivers: Option<PathBuf>,
    #[cfg_attr(not(feature = "netcdf"), allow(dead_code))]
    river_snap_km: f64,
    #[cfg_attr(not(feature = "netcdf"), allow(dead_code))]
    river_top: f64,
    output: Option<PathBuf>,
}

/// The 3D model of `levels=N`.
type Physics3D = dg_rs::physics::Hydrostatic3D<
    dg_rs::physics::LinearEOS,
    Box<dyn dg_rs::physics::VerticalMixing + Send + Sync>,
    Reflective2D,
>;

impl Options {
    /// The 3D thin-column depth (`debug_3d=thin=D`, else the model's
    /// default).
    /// The 3D thin-column depth (`debug_3d=thin=`): the model's default, or
    /// for a 3D tide 1 m. A column just over 0.1 m deep has millimetre layers,
    /// and as the tide floods the shore their `|Ω|/H_z` bound took the step
    /// from 4.6 s to 0.7 s within 20 minutes at Mausund; at 1 m it stays at
    /// 4–5 s (the shore films are the 2D module's)
    fn thin_depth(&self) -> f64 {
        let default = if self.tide_3d {
            TIDE_3D_THIN_DEPTH
        } else {
            Physics3D::DEFAULT_MIN_COLUMN_DEPTH
        };
        self.debug_3d
            .split(',')
            .find_map(|f| f.strip_prefix("thin=").and_then(|v| v.parse().ok()))
            .unwrap_or(default)
    }

    fn parse() -> Result<Self, String> {
        let args: HashMap<String, String> = std::env::args()
            .skip(1)
            .map(|a| match a.split_once('=') {
                Some((k, v)) => (k.to_string(), v.to_string()),
                None => (a, String::new()),
            })
            .collect();
        let get = |key: &str, default: f64| -> Result<f64, String> {
            args.get(key).map_or(Ok(default), |v| {
                v.parse().map_err(|_| format!("bad {key}={v}"))
            })
        };
        // WAM's integration of the sources: both parts below unless one is set
        let wam_sources = match args.get("wave_sources").map(String::as_str) {
            Some("swan") => false,
            None | Some("wam") => true,
            Some(other) => return Err(format!("wave_sources=swan|wam, not {other}")),
        };
        let constituents = |key: &str, default: &str| -> Result<Vec<&'static str>, String> {
            args.get(key)
                .map_or(default, String::as_str)
                .split(',')
                .filter(|n| !n.is_empty())
                .map(|n| canonical_name(n).ok_or(format!("{key}: unknown constituent {n}")))
                .collect()
        };
        let mut opts = Self {
            nx: get("nx", 120.0)? as usize,
            ny: get("ny", 90.0)? as usize,
            order: get("order", 2.0)? as usize,
            hours: get("hours", M2_PERIOD / 3600.0)?,
            rest_hours: get("rest_hours", 1.0)?,
            ramp_hours: get("ramp_hours", TIDAL_RAMP_HOURS)?,
            output_minutes: get("output_minutes", 60.0)?,
            snapshot_minutes: get("snapshot_minutes", 0.0)?,
            land_elevation: get("land_elevation", LAND_ELEVATION)?,
            manning: get("manning", MANNING_N)?,
            dem: match args.get("dem").map_or(DEM, String::as_str) {
                "none" => None,
                path => Some(path.to_string()),
            },
            project_bed: match args.get("bed").map_or("projected", String::as_str) {
                "projected" => true,
                "point" => false,
                other => return Err(format!("bad bed={other}: projected or point")),
            },
            lower_wall_land: match args.get("wall_land").map_or("keep", String::as_str) {
                "lower" => true,
                "keep" => false,
                other => return Err(format!("bad wall_land={other}: keep or lower")),
            },
            profile: get("profile", 0.0)? as usize,
            waves: get("waves", 0.0)? as usize,
            levels: get("levels", 0.0)? as usize,
            tide_3d: get("tide3d", 0.0)? != 0.0,
            steps_3d: get("steps_3d", 10.0)? as usize,
            dt_3d: args
                .get("dt_3d")
                .map(|v| v.parse::<f64>())
                .transpose()
                .map_err(|e| e.to_string())?,
            // Set below: its default follows the 3D stages' scheme
            cfl_3d: f64::NAN,
            cfl_bt: get("cfl_bt", 2.0)?,
            bt_levels: get("bt_levels", 2.0)? as usize,
            rk_bt: match args.get("rk_bt").map_or("43", String::as_str) {
                "3" => dg_rs::time::SspScheme::Rk3,
                "43" => dg_rs::time::SspScheme::Rk43,
                other => return Err(format!("rk_bt={other}: expected 3 or 43")),
            },
            rk_3d: match args.get("rk_3d").map_or("43", String::as_str) {
                "3" => dg_rs::time::SspScheme::Rk3,
                "43" => dg_rs::time::SspScheme::Rk43,
                other => return Err(format!("rk_3d={other}: expected 3 or 43")),
            },
            debug_3d: args.get("debug_3d").cloned().unwrap_or_default(),
            restart_hours: get("restart_hours", 0.0)?,
            resume: args.get("resume").map(PathBuf::from),
            rivers: args.get("rivers").map(PathBuf::from),
            river_snap_km: get("river_snap_km", 2.0)?,
            river_top: get("river_top", 0.2)?,
            wave_grid: {
                let text = args.get("wave_grid").map_or("25,36", String::as_str);
                let parts: Vec<usize> = text
                    .split(',')
                    .map(|v| v.parse().map_err(|_| format!("bad wave_grid={text}")))
                    .collect::<Result<_, _>>()?;
                match parts[..] {
                    [nf, nd] => [nf, nd],
                    _ => return Err(format!("wave_grid=frequencies,directions, not {text}")),
                }
            },
            wave_mesh: match args.get("wave_mesh") {
                None => None,
                Some(text) => {
                    let parts: Vec<usize> = text
                        .split(',')
                        .map(|v| v.parse().map_err(|_| format!("bad wave_mesh={text}")))
                        .collect::<Result<_, _>>()?;
                    match parts[..] {
                        [nx, ny] => Some([nx, ny, 1]),
                        [nx, ny, order] => Some([nx, ny, order]),
                        _ => return Err(format!("wave_mesh=NX,NY[,ORDER], not {text}")),
                    }
                }
            },
            turning: args
                .get("turning")
                .map(|v| v.parse().map_err(|_| format!("bad turning={v}")))
                .transpose()?,
            implicit_refraction: get("implicit", 0.0)? != 0.0,
            wave_implicit_sources: match args.get("wave_integration").map(String::as_str) {
                None => wam_sources,
                Some("frozen") => false,
                Some("implicit") => true,
                Some(other) => {
                    return Err(format!("wave_integration=frozen|implicit, not {other}"));
                }
            },
            wave_limiter: match args.get("wave_limiter").map(String::as_str) {
                None if wam_sources => Some(GrowthLimiter::Rate(DEFAULT_RATE_LIMITER)),
                None | Some("ris") => None,
                Some("rate") => Some(GrowthLimiter::Rate(DEFAULT_RATE_LIMITER)),
                Some("rate2") => Some(GrowthLimiter::RateBothSigns(DEFAULT_RATE_LIMITER)),
                Some(other) => return Err(format!("wave_limiter=ris|rate|rate2, not {other}")),
            },
            wave_substeps: get("wave_substeps", if wam_sources { 4.0 } else { 1.0 })? as usize,
            wave_cfl: get("wave_cfl", if wam_sources { 1.0 } else { 0.5 })?,
            wave_absorbing_land: match args.get("wave_land").map(String::as_str) {
                Some("floored") => false,
                None | Some("absorbing") => true,
                Some(other) => return Err(format!("wave_land=floored|absorbing, not {other}")),
            },
            wave_force: match args.get("wave_force").map(String::as_str) {
                Some("stress") => WaveForceForm::RadiationStress,
                None | Some("dissipation") => WaveForceForm::Dissipation,
                Some(other) => return Err(format!("wave_force=stress|dissipation, not {other}")),
            },
            wave_depth_limit: get("wave_depth_limit", 1.0)? != 0.0,
            wave_hours: get("wave_hours", 0.0)?,
            wave_reference: args.get("wave_reference").map(PathBuf::from),
            wave_coupling: get("wave_coupling", 0.0)?,
            wave_spectra: args.get("wave_spectra").map(PathBuf::from),
            wave_stations: args
                .get("wave_stations")
                .map_or("", String::as_str)
                .split(';')
                .filter(|s| !s.is_empty())
                .map(|s| {
                    let (name, at) = s.split_once(':').unwrap_or(("", s));
                    match at
                        .split(',')
                        .map(str::parse)
                        .collect::<Result<Vec<f64>, _>>()
                    {
                        Ok(v) if v.len() == 2 => {
                            let name = if name.is_empty() { at } else { name };
                            Ok((name.to_string(), v[0], v[1]))
                        }
                        _ => Err(format!("bad wave_stations entry {s} (NAME:LON,LAT)")),
                    }
                })
                .collect::<Result<_, _>>()?,
            wave_neighbours: get("wave_neighbours", 2.0)? as usize,
            wave_sea: {
                let text = args.get("wave_sea").map_or("2.5,10,285", String::as_str);
                let parts: Vec<f64> = text
                    .split(',')
                    .map(|v| v.parse().map_err(|_| format!("bad wave_sea={text}")))
                    .collect::<Result<_, _>>()?;
                match parts[..] {
                    [hs, tp, from] => [hs, tp, from],
                    _ => return Err(format!("wave_sea=HS,TP,FROM, not {text}")),
                }
            },
            lts: get("lts", 0.0)? as usize,
            cfl: get("cfl", f64::NAN)?,
            integrator: match args.get("rk").map_or("43", String::as_str) {
                "43" => StandardIntegrator::SSPRK43,
                "3" => StandardIntegrator::SSPRK3,
                other => return Err(format!("bad rk={other}: 43 or 3")),
            },
            rx0: args
                .get("rx0")
                .map(|v| v.parse().map_err(|_| format!("bad rx0={v}")))
                .transpose()?,
            rx0_min_depth: get("rx0_min_depth", RX0_MIN_DEPTH)?,
            slopes_3d: {
                let default = ElementSlopeBound::for_pycnocline(PYCNOCLINE_BOTTOM);
                let free_depth = get("free_depth", default.free_depth)?;
                match args.get("slopes3d").map(String::as_str) {
                    None | Some("off") => None,
                    Some("on") => Some(ElementSlopeBound {
                        free_depth,
                        ..default
                    }),
                    Some(r) => Some(ElementSlopeBound {
                        r_max: r.parse().map_err(|_| format!("bad slopes3d={r}"))?,
                        free_depth,
                    }),
                }
            },
            cliffs_3d: match args.get("cliffs3d").map(String::as_str) {
                None => Some(CLIFF_DEEP),
                Some("off") => None,
                Some(d) => Some(d.parse().map_err(|_| format!("bad cliffs3d={d}"))?),
            },
            ponds_3d: match args.get("ponds3d").map(String::as_str) {
                None => Some(POND_LOW_WATER),
                Some("off") => None,
                Some(l) => Some(l.parse().map_err(|_| format!("bad ponds3d={l}"))?),
            },
            bbox: match args.get("bbox") {
                None => FROYA_BBOX,
                Some(v) => v
                    .split(',')
                    .map(|c| c.trim().parse::<f64>())
                    .collect::<Result<Vec<_>, _>>()
                    .ok()
                    .and_then(|c| <[f64; 4]>::try_from(c).ok())
                    .filter(|[w, s, e, n]| w < e && s < n)
                    .ok_or(format!("bad bbox={v}: west,south,east,north"))?,
            },
            tide_transport: get("tide_transport", 0.0)?,
            mesh: args.get("mesh").cloned(),
            output: args.get("output").map(PathBuf::from),
            wind: args.contains_key("wind"),
            start: args
                .get("start")
                .cloned()
                .unwrap_or("2025-06-15T00:00:00Z".into()),
            tides: args
                .get("tides")
                .cloned()
                .unwrap_or("data/froya_boundary_tides.txt".into()),
            norkyst: args.get("norkyst").cloned(),
            met: args
                .get("met")
                .map_or("", String::as_str)
                .split(',')
                .filter(|f| !f.is_empty())
                .map(String::from)
                .collect(),
            band_km: get("band_km", 3.0)?,
            band_minutes: get("band_minutes", 30.0)?,
            blend_bed: get("blend", 1.0)? != 0.0,
            max_closed: get("max_closed", 0.05)?,
            nesting_ib: get("ib", 0.0)? != 0.0,
            heat: get("heat", 1.0)? != 0.0,
            jerlov: args.get("jerlov").map_or("III", String::as_str).parse()?,
            mixing: match args.get("mixing").map_or("k-epsilon", String::as_str) {
                "k-epsilon" => ("k-epsilon", GlsMixing::k_epsilon),
                "k-omega" => ("k-omega", GlsMixing::k_omega),
                "generic" => ("generic", GlsMixing::generic),
                other => {
                    return Err(format!("bad mixing={other}: k-epsilon, k-omega or generic"));
                }
            },
            correct_nested_tides: match args.get("nest_tides").map_or("corrected", String::as_str) {
                "corrected" => true,
                "raw" => false,
                other => return Err(format!("bad nest_tides={other}: corrected or raw")),
            },
            nesting_level: args
                .get("nest_level")
                .map(|v| v.parse().map_err(|_| format!("bad nest_level={v}")))
                .transpose()?,
            gauges: args
                .get("gauges")
                .map_or("data/tide_gauges/mausund_obs.txt", String::as_str)
                .split(',')
                .filter(|g| !g.is_empty())
                .map(String::from)
                .collect(),
            currents: args
                .get("currents")
                .map_or("", String::as_str)
                .split(',')
                .filter(|c| !c.is_empty())
                .map(String::from)
                .collect(),
            station_atlas: args
                .get("station_atlas")
                .cloned()
                .unwrap_or("data/froya_station_tides.txt".into()),
            station_minutes: get("station_minutes", 10.0)?,
            spinup_hours: get("spinup_hours", 24.0)?,
            gauge_gains: constituents("gauge_gains", "K1,O1,S2")?,
            gauge_ratios: constituents("gauge_ratios", "N2,Q1,P1,K2")?,
        };
        // SSP-RK(4,3)'s stages hold twice SSP-RK3's step where a forward-Euler
        // bound sets it (the shoreline elements' subcells)
        opts.cfl_3d = get(
            "cfl_3d",
            match opts.rk_3d {
                dg_rs::time::SspScheme::Rk3 => 2.0,
                dg_rs::time::SspScheme::Rk43 => 4.0,
            },
        )?;
        if opts.cfl.is_nan() {
            let scheme = opts.integrator.ssp_scheme().expect("an SSP integrator");
            opts.cfl = linear_cfl_swe_2d(opts.order, scheme)
                .map_or(1.0, |linear| LINEAR_CFL_SAFETY * linear);
        }
        Ok(opts)
    }
}

/// Surface range (and where the highest wet node is: x, y, h), largest speed
/// where h > 10 cm (and where: x, y, h), the same in open water (still-water
/// depth at least `STATION_MIN_DEPTH`, away from the foreshore films), and
/// the number of wet nodes (h > 1 mm).
struct Stats {
    eta: (f64, f64),
    highest: (f64, f64, f64),
    speed: f64,
    fastest: (f64, f64, f64),
    open_speed: f64,
    open_fastest: (f64, f64, f64),
    wet: usize,
}

/// A tide gauge's record (finite samples, Unix times) and its reference
/// constants.
struct Gauge {
    station: TideGaugeStation,
    observed: TimeSeries,
    observed_fit: Result<ReferenceFit, AnalysisError>,
}

/// A current record (Unix times, east/north) and its tidal ellipses.
struct CurrentRecord {
    station: ADCPStation,
    observed: CurrentTimeSeries,
    observed_fit: Result<EllipseFit, AnalysisError>,
}

/// A station, the point that samples the model there, what it is compared
/// with, and the sampled series.
struct Station {
    name: String,
    longitude: f64,
    latitude: f64,
    /// Tide gauge record (η) and current record, if any
    gauge: Option<Gauge>,
    current: Option<CurrentRecord>,
    /// The DG solution evaluated at the sampling point
    probe: Probe2D,
    /// Distance from the station to the sampling point (m; 0 at the station)
    offset: f64,
    /// Still-water depth at the sampling point (m)
    depth: f64,
    /// Local east in the mesh axes, `(cos θ, sin θ)`
    east: (f64, f64),
    /// Unix times, surface elevation, depth-averaged east and north velocity
    times: Vec<f64>,
    eta: Vec<f64>,
    u_east: Vec<f64>,
    v_north: Vec<f64>,
}

impl Station {
    /// Record the state of `q` at Unix time `t`.
    fn sample(&mut self, q: &SWESolution2D, bathymetry: &Bathymetry2D, t: f64) {
        let p = self
            .probe
            .sample_swe(q, Some(bathymetry), WetDryConfig::DEFAULT_H_DRY);
        let (c, s) = self.east;
        self.times.push(t);
        self.eta.push(p.eta);
        self.u_east.push(p.u * c + p.v * s);
        self.v_north.push(-p.u * s + p.v * c);
    }
}

/// Water-only mesh with nodal bathymetry.
struct Domain {
    name: &'static str,
    mesh: Arc<Mesh2D>,
    ops: Arc<DGOperators2D>,
    geom: Arc<GeometricFactors2D>,
    bathymetry: Arc<Bathymetry2D>,
    #[cfg_attr(not(feature = "netcdf"), allow(dead_code))]
    projection: Option<LocalProjection>,
    /// Manning roughness of the 2D bed friction (s/m^{1/3})
    manning: f64,
}

impl Domain {
    /// Grid `nx` × `ny` over `[x0, x1] × [y0, y1]` with open sides, bed
    /// elevation `bed(x, y)` (positive on land) sampled at the nodes (or
    /// projected onto them at `resolution` with `bed=projected`); keep the elements
    /// with a node below mean sea level.
    fn build(
        name: &'static str,
        grid: Mesh2D,
        opts: &Options,
        bed: impl Fn(f64, f64) -> f64,
        resolution: f64,
        projection: Option<LocalProjection>,
    ) -> Self {
        let ops = DGOperators2D::new(opts.order);
        let grid_geom = GeometricFactors2D::compute(&grid, &ops);
        let mut grid_bed = if opts.project_bed {
            Bathymetry2D::project(&grid, &ops, &grid_geom, bed, resolution)
        } else {
            Bathymetry2D::from_function(&grid, &ops, &grid_geom, bed)
        };
        // Land at the walls of elements that are otherwise water: a strip the
        // simplified coastline leaves in front of its walls
        if opts.lower_wall_land {
            let report = grid_bed.lower_wall_land(&grid, &ops, &grid_geom, WALL_LOW_WATER);
            println!(
                "  Wall land lowered in {} elements: {} nodes, by up to {:.1} m, {:.4} km³ of water added",
                report.elements,
                report.changed,
                report.max_change,
                report.volume / 1e9
            );
        }
        // Water one node wide is unresolved: make it shore
        let raised = grid_bed.raise_isolated_wet_nodes(&grid, &ops, &grid_geom, 0.0);
        println!("  {raised} isolated wet nodes raised to the lowest of their neighbours");
        // Shoals and pits a node wide: the depth-averaged velocity spikes over
        // them (a 6 m shoal among 15–45 m nodes carried 2.2–2.8 m/s)
        let min_depth = opts.rx0_min_depth;
        let describe = |bed: &Bathymetry2D| match bed.max_rx0(&grid, &ops, &grid_geom, min_depth) {
            Some(rx0) => {
                let at = |node: usize| {
                    let (k, i) = (ElementIndex::new(node / ops.n_nodes), node % ops.n_nodes);
                    let [x, y] = grid.reference_to_physical(k, ops.nodes_r[i], ops.nodes_s[i]);
                    (x / 1000.0, y / 1000.0, -bed.data[node])
                };
                let (s, d) = (at(rx0.nodes[0]), at(rx0.nodes[1]));
                format!(
                    "{:.2} between ({:.1}, {:.1}) km, {:.1} m and ({:.1}, {:.1}) km, {:.1} m",
                    rx0.value, s.0, s.1, s.2, d.0, d.1, d.2
                )
            }
            None => "none".into(),
        };
        println!(
            "  Largest slope factor r_x0 (nodes ≥ {min_depth} m deep): {}",
            describe(&grid_bed)
        );
        if let Some(r_max) = opts.rx0 {
            let report = grid_bed.smooth_rx0(&grid, &ops, &grid_geom, r_max, min_depth);
            println!(
                "  Smoothed to r_x0 ≤ {r_max} ({} sweeps): {} nodes changed, by up to {:.1} m; now {}",
                report.sweeps,
                report.changed,
                report.max_change,
                describe(&grid_bed)
            );
        }
        // 3D: no σ-level may cross the pycnocline between two nodes of one
        // element (TODO P1.3), between the columns that are not thin (the
        // shores are the wetting and drying's)
        if let Some(bound) = opts.slopes_3d {
            let thin = opts.thin_depth();
            let report = grid_bed.smooth_element_slopes(&grid, &ops, &grid_geom, bound, thin);
            println!(
                "  Smoothed for 3D (within an element r_x0 ≤ {} below {:.1} m, columns ≥ {thin} m, \
                 {} sweeps): {} elements over the bound (largest excess {:.1} m), {} nodes \
                 changed by up to {:.1} m, volume kept; {} elements left over",
                bound.r_max,
                bound.free_depth,
                report.sweeps,
                report.elements_before,
                report.excess_before,
                report.changed,
                report.max_change,
                report.elements_after
            );
            // Shore nodes of cliff elements: thin at low water, resolved at
            // high water beside the deep nodes, outside the bound
            if let Some(deep) = opts.cliffs_3d {
                let report = grid_bed.raise_cliff_shores(
                    &grid,
                    &ops,
                    &grid_geom,
                    deep,
                    thin,
                    CLIFF_HIGH_WATER,
                    CLIFF_HIGH_WATER,
                );
                let raised = grid_bed.raise_isolated_wet_nodes(&grid, &ops, &grid_geom, 0.0);
                println!(
                    "  Cliff shores raised to land (+{CLIFF_HIGH_WATER} m) in {} elements deeper \
                     than {deep} m: {} nodes, by up to {:.1} m, {:.3} km² of water at MSL; \
                     {raised} isolated wet nodes raised after",
                    report.elements,
                    report.nodes,
                    report.max_raise,
                    report.area / 1e6
                );
            }
            // Ponds below the lowest water: a node the open water does not
            // reach held a phantom current and set the 3D step
            if let Some(low_water) = opts.ponds_3d {
                let report = grid_bed.fill_tidal_ponds(
                    &grid,
                    &ops,
                    &grid_geom,
                    low_water,
                    CLIFF_HIGH_WATER,
                    ops.n_nodes,
                );
                let raised = grid_bed.raise_isolated_wet_nodes(&grid, &ops, &grid_geom, 0.0);
                println!(
                    "  Tidal ponds between {low_water} and +{CLIFF_HIGH_WATER} m filled to their sills: {} ponds, {} nodes, \
                     by up to {:.1} m, {:.3} km², {:.2e} m³; {} larger ponds left; {raised} isolated wet nodes raised after",
                    report.ponds,
                    report.nodes,
                    report.max_raise,
                    report.area / 1e6,
                    report.volume,
                    report.left
                );
            }
        }
        let has_water = |k: ElementIndex| grid_bed.element(k).iter().any(|&b| b < 0.0);
        let (mesh, kept) = grid.retain_elements(has_water, BoundaryTag::Wall);
        let bathymetry = grid_bed.select_elements(&kept);
        let geom = GeometricFactors2D::compute(&mesh, &ops);

        Self {
            name,
            mesh: Arc::new(mesh),
            ops: Arc::new(ops),
            geom: Arc::new(geom),
            bathymetry: Arc::new(bathymetry),
            projection,
            manning: opts.manning,
        }
    }

    /// Frøya–Smøla–Hitra from the data files, or `None` if they are missing.
    fn froya(opts: &Options) -> Result<Option<Self>, Box<dyn std::error::Error>> {
        let [west, south, east, north] = opts.bbox;
        let bbox = GeoBoundingBox::new(west, south, east, north);
        let (lat0, lon0) = bbox.center();
        let projection = LocalProjection::new(lat0, lon0);
        let dem_path = opts.dem.as_deref().map(Path::new).filter(|p| p.exists());
        let raster = if let Some(dem_path) = dem_path {
            // Kartverket's topobathy model: land heights and depths in one grid
            let dem = GeoTiffBathymetry::load(dem_path)?;
            let mut raster = BedRaster::elevation_model(&dem, &bbox)?;
            let (width, height) = raster.dimensions();
            // The 1 m level of the service fills unsurveyed sea with a flat 0
            // (`scripts/kartverket_topobathy.sh` refuses the cells that get it)
            let flat = (0..height)
                .flat_map(|row| (0..width).map(move |col| (row, col)))
                .filter(|&(row, col)| raster.pixel(row, col) == 0.0)
                .count();
            if flat * 20 > width * height {
                return Err(format!(
                    "{}: {:.0} % of the pixels are exactly 0, the flat sea of the 1 m level; \
                     fetch it at 50 m cells",
                    dem_path.display(),
                    100.0 * flat as f64 / (width * height) as f64
                )
                .into());
            }
            // Missing depths are an exact 0 too (the water surface): fill them
            // from their surroundings. Land above `land_elevation` is never
            // wet: cap it
            let filled = raster.fill_holes(|b| b == 0.0);
            println!(
                "  Elevation model: {} ({filled} zero pixels filled, land capped at {} m)",
                dem_path.display(),
                opts.land_elevation
            );
            raster.clamp_land(opts.land_elevation)
        } else {
            let bathy_path = Path::new("data/froya_smola_hitra.tif");
            let coast_path = Path::new("data/GSHHS_f_L1.shp");
            if !bathy_path.exists() || !coast_path.exists() {
                return Ok(None);
            }
            let geotiff = GeoTiffBathymetry::load(bathy_path)?;
            let coastline = CoastlineData::load(coast_path, &bbox)?;
            println!("  Bathymetry: {}", geotiff.statistics());
            println!(
                "  Coastline: {} polygons",
                coastline.statistics().polygon_count
            );
            BedRaster::from_geotiff(
                &geotiff,
                Some(&coastline),
                opts.land_elevation,
                &bbox,
                LAND_MASK_REFINEMENT,
            )
        };
        let (width, height) = raster.dimensions();
        println!(
            "  Bed raster: {width} × {height} pixels, {:.0} m resolution, {:.1} % water; {} onto the nodes",
            raster.pixel_size(),
            100.0 * raster.water_fraction(),
            if opts.project_bed {
                "projected"
            } else {
                "sampled"
            }
        );

        let grid = match &opts.mesh {
            Some(path) => {
                let mesh = read_gmsh_mesh(Path::new(path))?;
                println!(
                    "  Mesh: {path}, {} quadrilaterals, sizes {:.0}–{:.0} m",
                    mesh.n_elements,
                    mesh.h_min(),
                    mesh.h_max()
                );
                mesh
            }
            None => {
                // The sides are open sea wherever they cross water (land is
                // not meshed)
                let (x0, y0) = projection.geo_to_xy(bbox.min_lat, bbox.min_lon);
                let (x1, y1) = projection.geo_to_xy(bbox.max_lat, bbox.max_lon);
                Mesh2D::uniform_rectangle_with_bc(
                    x0,
                    x1,
                    y0,
                    y1,
                    opts.nx,
                    opts.ny,
                    BoundaryTag::Open,
                )
            }
        };
        Ok(Some(Self::build(
            "froya",
            grid,
            opts,
            raster.sampler(&projection),
            raster.pixel_size(),
            Some(projection),
        )))
    }

    /// 50 × 40 km basin shoaling to the east, with a beach along the east
    /// side and a round island.
    fn synthetic(opts: &Options) -> Self {
        let (lx, ly) = (50_000.0, 40_000.0);
        let bed = |x: f64, y: f64| {
            let island = (x - 0.4 * lx).hypot(y - 0.5 * ly) < 4_000.0;
            if island {
                opts.land_elevation
            } else {
                -100.0 + 105.0 * (x / lx).powi(2)
            }
        };
        let grid = Mesh2D::uniform_rectangle_with_bc(
            0.0,
            lx,
            0.0,
            ly,
            opts.nx,
            opts.ny,
            BoundaryTag::Open,
        );
        Self::build("synthetic", grid, opts, bed, 100.0, None)
    }

    /// Make walls of the open-boundary faces with a node farther than
    /// `ATLAS_COVERAGE` from the tidal atlas: water the atlas's parent model
    /// does not resolve (a narrow fjord arm crossing the domain edge) has no
    /// tide to force.
    fn close_uncovered_open_faces(
        &mut self,
        opts: &Options,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let atlas_path = Path::new(&opts.tides);
        if !atlas_path.exists() || opts.norkyst.is_some() {
            return Ok(());
        }
        let atlas = TidalAtlas::read(atlas_path)?;
        let closed =
            self.close_open_faces(&opts.tides, opts.max_closed, |mesh, ops, projection| {
                BoundaryCoverage::of_atlas(
                    mesh,
                    ops,
                    projection,
                    BoundaryTag::Open,
                    &atlas,
                    ATLAS_COVERAGE,
                )
            })?;
        if closed > 0 {
            println!(
                "  {closed} open-boundary faces lie farther than {:.0} km from the tidal atlas: walls",
                ATLAS_COVERAGE / 1000.0
            );
        }
        Ok(())
    }

    /// Make walls of the open-boundary faces `coverage` finds uncovered;
    /// returns how many. More than `max_closed` of the open faces is an
    /// error (`BoundaryCoverage::close`).
    fn close_open_faces(
        &mut self,
        source: &str,
        max_closed: f64,
        coverage: impl FnOnce(&Mesh2D, &DGOperators2D, &LocalProjection) -> BoundaryCoverage,
    ) -> Result<usize, String> {
        let Some(projection) = self.projection else {
            return Ok(0);
        };
        let coverage = coverage(&self.mesh, &self.ops, &projection);
        let mesh = Arc::get_mut(&mut self.mesh).expect("mesh not shared yet");
        coverage.close(mesh, max_closed).map_err(|e| {
            format!(
                "{source}: {e}: cut one for this domain (`norkyst_nesting_subset bbox=…`) or \
                 raise max_closed"
            )
        })
    }

    /// NorKyst nesting (`norkyst=`): read the parent, close the open faces it
    /// does not cover, sample it at the open boundary and the relaxation
    /// band, and blend the bed to its depth across the band.
    #[cfg(feature = "netcdf")]
    fn nesting(&mut self, opts: &Options) -> Result<Option<Nested>, Box<dyn std::error::Error>> {
        let (Some(path), Some(projection)) = (&opts.norkyst, self.projection) else {
            return Ok(None);
        };
        // A 3D tide also nests the parent's profiles
        let reader = Arc::new(if opts.levels > 0 && opts.tide_3d {
            OceanModelReader::from_file_with_profiles(Path::new(path))?
        } else {
            OceanModelReader::from_file(Path::new(path))?
        });
        println!("  NorKyst: {}", reader.summary());
        // NorKyst's ζ is not referenced to mean sea level: its 30-day mean
        // along the Frøya boundary is −0.28 m (−0.26 m at Mausund, where the
        // gauge's is 0). Shift it by the atlas's mean level.
        let level = match opts.nesting_level {
            Some(level) => level,
            None => {
                let atlas = Path::new(&opts.tides);
                let means: Vec<f64> = if atlas.exists() {
                    TidalAtlas::read(atlas)?
                        .points
                        .iter()
                        .filter_map(|p| p.mean)
                        .collect()
                } else {
                    Vec::new()
                };
                if means.is_empty() {
                    0.0
                } else {
                    -means.iter().sum::<f64>() / means.len() as f64
                }
            }
        };
        println!("  NorKyst ζ shifted by {level:+.3} m to mean sea level");
        let options = NestingOptions::default()
            .with_band(1000.0 * opts.band_km)
            .with_ramp_up(3600.0 * opts.ramp_hours)
            .with_reference_level(level);
        let closed = self.close_open_faces(path, opts.max_closed, |mesh, ops, projection| {
            BoundaryCoverage::of_parent(mesh, ops, projection, BoundaryTag::Open, &reader, &options)
        })?;
        if closed > 0 {
            println!("  {closed} open-boundary faces have no wet NorKyst point nearby: walls");
        }
        let clock = ModelClock::parse(&opts.start)?;
        let parent = OceanModelState::new(
            reader.clone(),
            &self.mesh,
            &self.ops,
            &projection,
            BoundaryTag::Open,
            clock,
            &options,
        )?;
        println!(
            "  Nesting: {} open-boundary nodes ({} snapped to a wet NorKyst point), {} nodes in a {} km band",
            parent.n_boundary_nodes(),
            parent.n_snapped(),
            parent.n_band_nodes(),
            opts.band_km
        );
        if let Some((lo, median, hi)) = parent.depth_ratios(&self.bathymetry) {
            println!(
                "  NorKyst/child depth at the open boundary: {lo:.2}–{hi:.2} (median {median:.2})"
            );
        }
        if opts.blend_bed && opts.band_km > 0.0 {
            let bathymetry = Arc::get_mut(&mut self.bathymetry).expect("bed not shared yet");
            // Not at shores that dry at low water: NorKyst's grid has none
            let blended =
                parent.blend_bathymetry(bathymetry, &self.ops, &self.geom, -WALL_LOW_WATER);
            println!(
                "  Bed blended to NorKyst's at {blended} nodes of the band (shores \
                 shallower than {} m kept)",
                -WALL_LOW_WATER
            );
        }
        Ok(Some((parent, reader)))
    }

    #[cfg(not(feature = "netcdf"))]
    fn nesting(&mut self, _opts: &Options) -> Result<Option<Nested>, Box<dyn std::error::Error>> {
        Ok(None)
    }

    fn builder<BC: SWEBoundaryCondition2D>(&self, bc: BC) -> SWEPhysics2DBuilder<BC> {
        self.builder_without_friction(bc)
            .with_implicit_friction(ManningFriction2D::new(G, self.manning))
    }

    /// The 2D module without its bed friction: the 3D model's, whose bottom
    /// drag's depth mean replaces it.
    fn builder_without_friction<BC: SWEBoundaryCondition2D>(
        &self,
        bc: BC,
    ) -> SWEPhysics2DBuilder<BC> {
        PhysicsBuilder::swe_2d(
            self.mesh.clone(),
            self.ops.clone(),
            self.geom.clone(),
            ShallowWater2D::new(G),
            bc,
        )
        .with_bathymetry(self.bathymetry.clone())
        .with_limiter(StandardLimiter2D::Positivity(WetDryConfig::DEFAULT_H_DRY))
        .with_wet_dry(WetDryConfig::default())
        .with_source(CoriolisSource2D::f_plane(F_CORIOLIS))
    }

    /// Still water at mean sea level: h = max(0, −B).
    fn at_rest(&self) -> SWESolution2D {
        let mut q = SWESolution2D::new(self.mesh.n_elements, self.ops.n_nodes);
        for k in ElementIndex::iter(self.mesh.n_elements) {
            for i in 0..self.ops.n_nodes {
                let h = (-self.bathymetry.get(k, i)).max(0.0);
                q.set_state(k, i, SWEState2D::new(h, 0.0, 0.0));
            }
        }
        q
    }

    /// Stations for the tide-gauge files `gauges` and the current records
    /// `currents` that lie in the domain (see [`Domain::probe`]), and one at
    /// each gauge's nearest point of the station atlas `atlas`: NorKyst's
    /// currents are compared there, at NorKyst's own position, since they
    /// vary over a few hundred metres among islands.
    fn stations(
        &self,
        gauges: &[String],
        currents: &[String],
        atlas: Option<&TidalAtlas>,
    ) -> Vec<Station> {
        let Some(projection) = &self.projection else {
            return Vec::new();
        };
        let locator = PointLocator2D::new(&self.mesh);
        let days = |t: &[f64]| {
            t.last()
                .zip(t.first())
                .map_or(0.0, |(b, a)| (b - a) / 86_400.0)
        };
        let mut stations = Vec::new();
        let mut add = |name: String,
                       longitude: f64,
                       latitude: f64,
                       gauge: Option<Gauge>,
                       current: Option<CurrentRecord>,
                       record_days: f64| {
            let (x, y) = projection.geo_to_xy(latitude, longitude);
            let Some((probe, offset, depth)) = self.probe(&locator, [x, y]) else {
                println!("  Station {name}: outside the domain (skipped)");
                return;
            };
            println!(
                "  Station {name}: sampled {}, {depth:.1} m deep; record {record_days:.0} days",
                if offset > 0.0 {
                    format!("{offset:.0} m from the station (its element is not submerged)")
                } else {
                    "at the station".to_string()
                }
            );
            stations.push(Station {
                name,
                longitude,
                latitude,
                gauge,
                current,
                probe,
                offset,
                depth,
                east: east_axis(projection, latitude, longitude),
                times: Vec::new(),
                eta: Vec::new(),
                u_east: Vec::new(),
                v_north: Vec::new(),
            });
        };
        for path in gauges {
            let gauge = match read_tide_gauge_file(Path::new(path)) {
                Ok(gauge) => gauge,
                Err(e) => {
                    println!("  Gauge {path}: {e} (skipped)");
                    continue;
                }
            };
            let Some(station) = gauge.station.clone() else {
                println!("  Gauge {path}: no station position (skipped)");
                continue;
            };
            let (times, values): (Vec<f64>, Vec<f64>) = gauge
                .time_series
                .times()
                .iter()
                .copied()
                .zip(gauge.time_series.values().iter().copied())
                .filter(|(_, v)| v.is_finite())
                .unzip();
            let observed_fit = fit_record(&times, &values, &GAUGE_CONSTITUENTS, None);
            let norkyst = atlas
                .and_then(|a| a.nearest(station.longitude, station.latitude))
                .filter(|&(p, d)| {
                    d <= STATION_MAX_OFFSET
                        && d > 0.0
                        && p.constituents.iter().all(|c| c.velocity.is_some())
                });
            let name = station.name.clone();
            add(
                name.clone(),
                station.longitude,
                station.latitude,
                Some(Gauge {
                    station,
                    observed: TimeSeries::new(&times, &values),
                    observed_fit,
                }),
                None,
                days(&times),
            );
            if let Some((p, _)) = norkyst {
                add(
                    format!("{name} NorKyst point"),
                    p.lon,
                    p.lat,
                    None,
                    None,
                    0.0,
                );
            }
        }
        for path in currents {
            let record = match read_adcp_file(Path::new(path)) {
                Ok(record) => record,
                Err(e) => {
                    println!("  Current record {path}: {e} (skipped)");
                    continue;
                }
            };
            let observed = record.time_series;
            let (times, u, v) = (observed.times(), observed.u_values(), observed.v_values());
            let observed_fit = fit_current_record(&times, &u, &v, &GAUGE_CONSTITUENTS, None);
            let station = record.station;
            add(
                station.name.clone(),
                station.longitude,
                station.latitude,
                None,
                Some(CurrentRecord {
                    station,
                    observed,
                    observed_fit,
                }),
                days(&times),
            );
        }
        stations
    }

    /// Where a station at `p` samples the model: at `p` if its element is
    /// submerged (every node at least `STATION_MIN_DEPTH` below mean sea
    /// level), else at the nearest point of a submerged element within
    /// `STATION_MAX_OFFSET`. Shoreline elements are avoided because
    /// a coarse mesh leaves pockets there that hold water above the tide.
    /// The probe, its distance from `p` and its still-water depth.
    fn probe(&self, locator: &PointLocator2D, p: [f64; 2]) -> Option<(Probe2D, f64, f64)> {
        let submerged = |k: ElementIndex| {
            self.bathymetry
                .element(k)
                .iter()
                .all(|&b| b <= -STATION_MIN_DEPTH)
        };
        let depth = |probe: &Probe2D| -probe.evaluate(self.bathymetry.element(probe.element()));
        if let Some(probe) = Probe2D::at(locator, &self.ops, p)
            && submerged(probe.element())
        {
            let d = depth(&probe);
            return Some((probe, 0.0, d));
        }
        // The station's reference coordinates clamped to the element: its
        // nearest point on a rectangle (the Frøya grid), close to it on any
        // convex quadrilateral
        let (distance, point) = ElementIndex::iter(self.mesh.n_elements)
            .filter(|&k| submerged(k))
            .filter_map(|k| {
                let [r, s] = inverse_bilinear(&self.mesh.element_vertices(k), p)
                    .unwrap_or([0.0, 0.0])
                    .map(|c| c.clamp(-1.0, 1.0));
                let [x, y] = self.mesh.reference_to_physical(k, r, s);
                let distance = (x - p[0]).hypot(y - p[1]);
                (distance <= STATION_MAX_OFFSET)
                    .then_some((distance, MeshPoint { element: k, r, s }))
            })
            .min_by(|a, b| a.0.total_cmp(&b.0))?;
        let probe = Probe2D::new(&self.mesh, &self.ops, point);
        let d = depth(&probe);
        Some((probe, distance, d))
    }

    fn volume(&self, q: &SWESolution2D) -> f64 {
        ElementIndex::iter(self.mesh.n_elements)
            .map(|k| {
                (0..self.ops.n_nodes)
                    .map(|i| {
                        self.geom.mass[self.geom.node_index(k.as_usize(), i)] * q.get_state(k, i).h
                    })
                    .sum::<f64>()
            })
            .sum()
    }

    fn stats(&self, q: &SWESolution2D) -> Stats {
        let mut stats = Stats {
            eta: (f64::MAX, f64::MIN),
            highest: (0.0, 0.0, 0.0),
            speed: 0.0,
            fastest: (0.0, 0.0, 0.0),
            open_speed: 0.0,
            open_fastest: (0.0, 0.0, 0.0),
            wet: 0,
        };
        for k in ElementIndex::iter(self.mesh.n_elements) {
            for i in 0..self.ops.n_nodes {
                let s = q.get_state(k, i);
                if s.h > WetDryConfig::DEFAULT_H_DRY {
                    stats.wet += 1;
                    let eta = s.h + self.bathymetry.get(k, i);
                    if eta > stats.eta.1 {
                        let [x, y] = self.mesh.reference_to_physical(
                            k,
                            self.ops.nodes_r[i],
                            self.ops.nodes_s[i],
                        );
                        stats.highest = (x, y, s.h);
                    }
                    stats.eta = (stats.eta.0.min(eta), stats.eta.1.max(eta));
                }
                let speed = s.hu.hypot(s.hv) / s.h;
                let at = || {
                    let [x, y] = self.mesh.reference_to_physical(
                        k,
                        self.ops.nodes_r[i],
                        self.ops.nodes_s[i],
                    );
                    (x, y, s.h)
                };
                if s.h > 0.1 && speed > stats.speed {
                    stats.speed = speed;
                    stats.fastest = at();
                }
                if -self.bathymetry.get(k, i) >= STATION_MIN_DEPTH && speed > stats.open_speed {
                    stats.open_speed = speed;
                    stats.open_fastest = at();
                }
            }
        }
        stats
    }

    fn print_summary(&self) {
        let depths: Vec<f64> = self.bathymetry.data.iter().map(|b| -b).collect();
        let n_nodes = depths.len();
        let dry = depths.iter().filter(|&&d| d <= 0.0).count();
        let shoreline = ElementIndex::iter(self.mesh.n_elements)
            .filter(|&k| (0..self.ops.n_nodes).any(|i| self.bathymetry.get(k, i) >= 0.0))
            .count();
        println!(
            "  Water-only mesh: {} elements (P{}), {} boundary faces; {shoreline} shoreline elements",
            self.mesh.n_elements, self.ops.order, self.mesh.n_boundary_edges
        );
        println!(
            "  Depth: max {:.0} m; {:.1} % of nodes dry at mean sea level",
            depths.iter().copied().fold(0.0, f64::max),
            100.0 * dry as f64 / n_nodes as f64
        );
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let opts = Options::parse()?;
    println!("Frøya–Smøla–Hitra tidal run (WetDry split form, Simulation)\n");
    println!("Setting up the domain...");
    let mut domain = match Domain::froya(&opts)? {
        Some(domain) => domain,
        None => {
            println!("  Data files not found in ./data/: synthetic basin instead");
            Domain::synthetic(&opts)
        }
    };
    domain.print_summary();
    let (parent, parent_reader) = match domain.nesting(&opts)? {
        Some((parent, reader)) => (Some(parent), Some(reader)),
        None => (None, None),
    };
    domain.close_uncovered_open_faces(&opts)?;
    let wave_domain = match opts.wave_mesh {
        Some([nx, ny, order]) if opts.waves > 0 || opts.wave_coupling > 0.0 => {
            println!("The waves' own grid, {nx} × {ny} at P{order}...");
            let mut wave_opts = opts.clone();
            (wave_opts.mesh, wave_opts.nx, wave_opts.ny, wave_opts.order) = (None, nx, ny, order);
            let wave_domain =
                Domain::froya(&wave_opts)?.ok_or("wave_mesh= needs the Frøya data files")?;
            wave_domain.print_summary();
            Some(wave_domain)
        }
        _ => None,
    };
    if opts.waves > 0 {
        wave_cost(wave_domain.as_ref().unwrap_or(&domain), &domain, &opts);
        return Ok(());
    }
    if opts.levels > 0 && opts.tide_3d {
        return tidal_run_3d(&domain, &opts, parent, parent_reader);
    }
    if opts.restart_hours > 0.0 || opts.resume.is_some() {
        return Err("restart_hours= and resume= are for the 3D tide (levels=N tide3d=1)".into());
    }
    if opts.rivers.is_some() {
        return Err("rivers= is for the 3D tide (levels=N tide3d=1)".into());
    }
    if opts.wave_coupling > 0.0 && opts.levels > 0 {
        return Err("wave_coupling= is for the 2D tide".into());
    }
    if opts.levels > 0 {
        cost_3d(&domain, &opts);
        return Ok(());
    }

    lake_at_rest(&domain, opts.rest_hours);
    let waves = (opts.wave_coupling > 0.0).then(|| wave_domain.as_ref().unwrap_or(&domain));
    tidal_run(&domain, &opts, parent, waves)
}

/// Walls everywhere and no forcing: the largest spurious current and surface
/// deviation where h > 10 cm.
fn lake_at_rest(domain: &Domain, hours: f64) {
    println!("\nLake at rest ({hours} h, walls only)...");
    let physics = domain.builder(Reflective2D::new()).build();
    let mut q = domain.at_rest();
    let rhs = physics.compute_rhs(&q, 0.0);
    println!("  max |dq/dt| at t = 0: {:.2e}", rhs.max_abs());

    let sim = Simulation::new(physics, SSPRK3).with_cfl(1.0);
    let result = sim.run(&mut q, 0.0, hours * 3600.0);
    let stats = domain.stats(&q);
    println!(
        "  after {} steps ({:.1} s wall): max |u| {:.2e} m/s, η in [{:.2e}, {:.2e}] m{}",
        result.n_steps,
        result.wall_time,
        stats.speed,
        stats.eta.0,
        stats.eta.1,
        if result.success { "" } else { " (FAILED)" }
    );
}

/// What a tidal run is forced and validated with, in 2D and 3D alike.
struct TideSetup {
    t_end: f64,
    clock: ModelClock,
    stations: Vec<Station>,
    station_atlas: Option<TidalAtlas>,
    /// The open boundary's condition, and a description of it
    open: Box<dyn SWEBoundaryCondition2D>,
    forcing: String,
    /// The nesting's relaxation band of the depth mean
    band: Option<NestingRelaxation2D>,
    /// The weather model's source term for the 2D module: wind and pressure
    /// in 2D, the pressure alone in 3D
    weather: Option<GriddedAtmosphere2D>,
    /// In 3D, the weather model's wind stress on the columns
    wind_3d: Option<GriddedWindStress>,
    /// In 3D, the weather model's heat and freshwater fluxes on the columns
    fluxes_3d: Option<GriddedSurfaceFluxes>,
}

/// The stations, the clock and the open boundary of a tidal run (the parent
/// model's, its tides corrected to the gauge, or the atlas's), and its band
/// and weather.
/// `<output>/clock.txt`: the run's `Clock:` line, so its VTU frames replay with their
/// dates in `dg-viz` without the run's log.
fn write_clock(output_dir: &Path, clock: &ModelClock) -> std::io::Result<()> {
    fs::write(
        output_dir.join("clock.txt"),
        format!("Clock: t = 0 at {} UTC\n", clock.format(0.0)),
    )
}

fn tide_setup(
    domain: &Domain,
    opts: &Options,
    parent: Option<OceanModelState>,
    three_d: bool,
) -> Result<TideSetup, Box<dyn std::error::Error>> {
    let t_end = opts.hours * 3600.0;
    // A 3D step's barotropic pass reads its forcing past the step's end, the
    // last one past `t_end`
    let t_forcing = if three_d {
        let reach = dg_rs::time::ModeSplitIntegrator::FORCING_REACH;
        t_end + (reach - 1.0) * opts.dt_3d.unwrap_or(MAX_DT_3D)
    } else {
        t_end
    };
    let station_atlas = Path::new(&opts.station_atlas);
    let station_atlas = station_atlas
        .exists()
        .then(|| TidalAtlas::read(station_atlas))
        .transpose()?;
    let stations = domain.stations(&opts.gauges, &opts.currents, station_atlas.as_ref());
    let clock = ModelClock::parse(&opts.start)?;
    println!("  Clock: t = 0 at {} UTC", clock.format(0.0));
    if opts.levels == 0 {
        println!("  Bed friction: Manning n = {}", domain.manning);
    }
    let weather = weather(domain, opts, &clock, t_forcing)?;
    // A 3D model's columns take the wind, its 2D module the pressure
    // (split before the level's clone shares the fields)
    let (weather, wind_3d) = match weather {
        Some(atmosphere) if three_d => {
            let (pressure, wind) = atmosphere.split_for_3d();
            (Some(pressure), Some(wind))
        }
        other => (other, None),
    };
    let fluxes_3d = match &weather {
        Some(atmosphere) if three_d && opts.heat && atmosphere.reader().has_heat_budget() => {
            let (_, lat_min, _, lat_max) = atmosphere.reader().grid().bbox();
            let bulk = Coare35::at_latitude(0.5 * (lat_min + lat_max));
            println!(
                "  Surface heat and freshwater: COARE 3.5 on the weather, shortwave into Jerlov {:?} water",
                opts.jerlov
            );
            Some(atmosphere.surface_fluxes(bulk)?)
        }
        Some(_) if three_d && opts.heat => {
            println!("  Surface heat and freshwater: none (the weather has no heat budget)");
            None
        }
        _ => None,
    };
    let level = match &weather {
        Some(gridded) => Some(Level::Gridded(gridded.clone())),
        None => opts.wind.then(|| Level::Uniform(pressure())),
    };
    let reference = AtlasReference::new(&stations, station_atlas.as_ref());
    let parent = match parent {
        Some(parent) if opts.correct_nested_tides => Some(correct_parent_tides(
            domain,
            opts,
            reference.as_ref(),
            parent,
            t_end,
        )?),
        other => other,
    };
    if let Some(parent) = &parent {
        parent.check_time_coverage(0.0, t_forcing).map_err(|e| {
            if t_forcing > t_end {
                format!(
                    "{e}: a 3D run reads its forcing up to a third of a step ({:.0} s) past its \
                     end; cut the parent an hour longer (`norkyst_nesting_subset hours=`) \
                     or end the run earlier",
                    t_forcing - t_end
                )
            } else {
                e
            }
        })?;
    }
    let band = parent
        .as_ref()
        .filter(|_| opts.band_km > 0.0)
        .map(|p| p.relaxation(60.0 * opts.band_minutes));
    let (open, forcing) = open_boundary(
        domain,
        opts,
        &clock,
        t_end,
        reference.as_ref(),
        parent,
        level,
    )?;
    Ok(TideSetup {
        t_end,
        clock,
        stations,
        station_atlas,
        open,
        forcing,
        band,
        weather,
        wind_3d,
        fluxes_3d,
    })
}

fn tidal_run(
    domain: &Domain,
    opts: &Options,
    parent: Option<OceanModelState>,
    wave_domain: Option<&Domain>,
) -> Result<(), Box<dyn std::error::Error>> {
    let TideSetup {
        t_end,
        clock,
        mut stations,
        station_atlas,
        open,
        forcing,
        band,
        weather,
        ..
    } = tide_setup(domain, opts, parent, false)?;
    let wall = Reflective2D::new();
    let bc = MultiBoundaryCondition2D::new(&wall).with_open(open.as_ref());

    let mut builder = domain.builder(bc);
    if let Some(band) = band {
        builder = builder.with_source(band);
    }
    // The waves sample the weather on their own mesh
    let wave_weather = weather.clone();
    if let Some(weather) = weather {
        builder = builder.with_source(weather);
    } else if opts.wind {
        builder = builder
            .with_source(
                WindStress2D::from_direction(WIND_SPEED, WIND_DIRECTION)
                    .with_drag(DragCoefficient::LargePond),
            )
            .with_source(pressure());
    }
    let physics: SWEPhysics2D<_> = builder.build();
    if opts.profile > 0 {
        profile_phases(domain, &physics, opts.profile);
        return Ok(());
    }
    let (mut waves, mut wave_points) = match wave_domain {
        Some(wave_domain) => {
            let (waves, points) =
                coupled_waves(wave_domain, &physics, wave_weather.as_ref(), opts, &clock)?;
            (Some(waves), points)
        }
        None => (None, None),
    };

    let output_dir = opts
        .output
        .clone()
        .unwrap_or_else(|| Path::new("output").join(domain.name));
    fs::create_dir_all(&output_dir)?;
    write_clock(&output_dir, &clock)?;
    #[cfg(feature = "netcdf")]
    let mut netcdf = match &domain.projection {
        Some(projection) => Some(NetCDFWriter::create(
            NetCDFWriterConfig::new(output_dir.join("froya.nc").to_string_lossy())
                .with_title("Frøya–Smøla–Hitra tidal run")
                .with_institution("dg-rs")
                .with_clock(clock),
            &netcdf_mesh_info(domain, projection),
        )?),
        None => None,
    };

    println!(
        "\nTides: {forcing}, {:.2} h{} → {}",
        opts.hours,
        if !opts.met.is_empty() {
            ", gridded wind and pressure"
        } else if opts.wind {
            ", wind and pressure"
        } else {
            ""
        },
        output_dir.display()
    );
    println!(
        "  Integrator: {}{}, CFL {:.3}, capped at {:.3} where elements may run dry",
        opts.integrator.name(),
        if opts.lts > 0 {
            format!(", local time stepping up to {} levels", opts.lts)
        } else {
            String::new()
        },
        opts.cfl,
        opts.integrator.ssp_coefficient() * positivity_cfl_swe_2d(opts.order),
    );
    println!(
        "  time  |  η range (m)     | η max at (x, y km; h m) | max |u| (m/s) at (x, y km; h m) | open water ≥ 3 m (m/s)         | wet nodes | volume change"
    );

    let mut q = domain.at_rest();
    let volume0 = domain.volume(&q);
    // Callbacks sample the stations; every `output_every`-th also writes
    // output, every `snapshot_every`-th a snapshot frame
    let snapshot_minutes = (opts.snapshot_minutes > 0.0).then_some(opts.snapshot_minutes);
    let base_minutes = if stations.is_empty() {
        opts.output_minutes
            .min(snapshot_minutes.unwrap_or(f64::INFINITY))
    } else {
        opts.station_minutes
    };
    let every = |minutes: f64| (minutes / base_minutes).round().max(1.0) as usize;
    let (interval, output_every) = (base_minutes * 60.0, every(opts.output_minutes));
    if waves.is_some() {
        let ratio = opts.wave_coupling / base_minutes;
        if ratio < 1.0 || (ratio - ratio.round()).abs() > 1e-9 * ratio {
            return Err(format!(
                "wave_coupling={} must be a multiple of the {base_minutes}-minute callback \
                 interval (station_minutes=, or output_minutes= without stations)",
                opts.wave_coupling
            )
            .into());
        }
    }
    let snapshot_every = snapshot_minutes.map(every);
    let mut snapshot = match snapshot_every {
        Some(every) => {
            let path = output_dir.join(format!("{}.dgsnap", domain.name));
            println!(
                "  Snapshot file: {} every {} min",
                path.display(),
                every as f64 * base_minutes
            );
            let metadata = snapshot_metadata(domain, &stations, "");
            let metadata: Vec<(&str, &str)> =
                metadata.iter().map(|(k, v)| (*k, v.as_str())).collect();
            let surface: &[&str] = match waves {
                Some(_) => &SeaStateFields::NAMES,
                None => &[],
            };
            Some(SnapshotWriter::create_with(
                path,
                &domain.mesh,
                &domain.ops,
                &domain.bathymetry.data,
                Some(&clock),
                &metadata,
                WetDryConfig::DEFAULT_H_DRY,
                SnapshotLayout {
                    surface,
                    levels: None,
                },
            )?)
        }
        None => None,
    };
    // The waves' sea state for the snapshot frames
    let sea = (waves.is_some() && snapshot.is_some()).then(|| RefCell::new(SeaStates::default()));
    let mut n_callbacks = 0;
    let mut frame = 0;
    let mut write_error = None;
    let start = Instant::now();
    let mut callback = |q: &SWESolution2D, t: f64| {
        for s in &mut stations {
            s.sample(q, &domain.bathymetry, clock.unix(t));
        }
        n_callbacks += 1;
        if let (Some(writer), Some(every)) = (snapshot.as_mut(), snapshot_every)
            && (n_callbacks - 1) % every == 0
        {
            let mut sea = sea.as_ref().map(RefCell::borrow_mut);
            let surface = sea.as_mut().map_or(Vec::new(), |sea| sea.at(t));
            if let Err(e) = writer.write_state(t, q, &surface) {
                write_error.get_or_insert(e.to_string());
            }
        }
        if (n_callbacks - 1) % output_every != 0 {
            return;
        }
        let stats = domain.stats(q);
        let (x, y, h) = stats.fastest;
        let (xo, yo, ho) = stats.open_fastest;
        println!(
            "{:6.2} h | [{:+.3}, {:+.3}] | ({:6.1}, {:6.1}; {:.3}) | {:5.2} at ({:6.1}, {:6.1}; {h:6.1}) | {:5.2} at ({:6.1}, {:6.1}; {ho:6.1}) | {:9} | {:+.3e}",
            t / 3600.0,
            stats.eta.0,
            stats.eta.1,
            stats.highest.0 / 1e3,
            stats.highest.1 / 1e3,
            stats.highest.2,
            stats.speed,
            x / 1e3,
            y / 1e3,
            stats.open_speed,
            xo / 1e3,
            yo / 1e3,
            stats.wet,
            domain.volume(q) / volume0 - 1.0
        );
        let path = output_dir.join(format!("{}_{frame:04}.vtu", domain.name));
        let written = write_vtk_swe(
            &path,
            &domain.mesh,
            &domain.ops,
            q,
            Some(&domain.bathymetry),
            t,
            WetDryConfig::DEFAULT_H_DRY,
        );
        if let Err(e) = written {
            write_error.get_or_insert(e.to_string());
        }
        #[cfg(feature = "netcdf")]
        if let Some(writer) = netcdf.as_mut() {
            let (h, eta, u, v) = netcdf_fields(domain, q);
            if let Err(e) = writer.write_timestep(t, &h, &eta, Some(&u), Some(&v)) {
                write_error.get_or_insert(e.to_string());
            }
        }
        frame += 1;
    };
    // The waves' steps and wall time since the last line
    let mut since = (0.0, CoupledWavesStats::default());
    let mut report = |waves: &CoupledWaves2D, t: f64| {
        report_waves(domain, waves, t, since);
        if let Some(points) = wave_points.as_mut() {
            points.report(waves, t);
        }
        since = (t, waves.stats());
    };
    let coupling = waves.as_mut().map(|waves| Coupling {
        waves,
        interval: 60.0 * opts.wave_coupling,
        report_every: 60.0 * opts.output_minutes,
        report: &mut report,
        sea: sea.as_ref(),
    });
    let (result, clips) = if opts.lts > 0 {
        let sim = Simulation::new(physics, Multirate::with_base(opts.integrator, opts.lts))
            .with_cfl(opts.cfl)
            .with_callback_interval(interval);
        run_tide(sim, &mut q, t_end, coupling, &mut callback)
    } else {
        let sim = Simulation::new(physics, opts.integrator)
            .with_cfl(opts.cfl)
            .with_callback_interval(interval);
        run_tide(sim, &mut q, t_end, coupling, &mut callback)
    };
    if let Some(e) = write_error {
        return Err(e.into());
    }

    let steps = result.n_steps.max(1);
    println!(
        "\n{} after {:.2} h: {} steps (mean dt {:.2} s), {:.1} s wall ({:.1} ms/step), {} negative-depth clips",
        if result.success { "Done" } else { "FAILED" },
        result.final_time / 3600.0,
        result.n_steps,
        result.final_time / steps as f64,
        start.elapsed().as_secs_f64(),
        1e3 * start.elapsed().as_secs_f64() / steps as f64,
        clips
    );
    if let Some(stats) = result.local_time_stepping {
        println!(
            "  Local time stepping: time-step ratio up to 2^{}, {:.2}x less RHS work than global steps at the finest step",
            stats.finest_level,
            stats.speedup()
        );
    }
    if let Some(waves) = &waves {
        let stats = waves.stats();
        let wall = start.elapsed().as_secs_f64();
        println!(
            "  Waves: {} exchanges every {} min, {} wave steps (mean {:.2} s); stepping {:.1} s \
             ({:.0} % of the wall time), exchanges {:.1} s ({:.1} %)",
            stats.exchanges,
            opts.wave_coupling,
            stats.wave_steps,
            result.final_time / stats.wave_steps.max(1) as f64,
            stats.stepping_time,
            100.0 * stats.stepping_time / wall,
            stats.exchange_time,
            100.0 * stats.exchange_time / wall
        );
    }
    if let Some(points) = &wave_points {
        points.summarise(&output_dir, opts.ramp_hours.max(1.0) * 3600.0 * 2.0)?;
    }
    // The sea at every wave node, and against an earlier run's
    if let Some(waves) = &waves {
        let model = waves.model();
        let params = model.parameters(waves.state());
        if let Err(e) = write_wave_nodes(model, &params, result.final_time, &output_dir) {
            eprintln!("  could not write the waves at the nodes: {e}");
        }
        if let Some(path) = &opts.wave_reference
            && let Err(e) = compare_wave_nodes(model, &params, path)
        {
            eprintln!("  could not compare with {}: {e}", path.display());
        }
    }
    report_stations(&stations, &output_dir, opts, &clock, station_atlas.as_ref())?;
    if let Some(e) = result.error {
        return Err(e.into());
    }
    println!("Visualize with ParaView: {}/*.vtu", output_dir.display());
    Ok(())
}

/// The waves of a tidal run with `wave_coupling=`: the model, its exchange
/// interval (s), the progress line every `report_every` seconds, and the sea
/// states for the snapshot frames, if any.
struct Coupling<'a> {
    waves: &'a mut CoupledWaves2D,
    interval: f64,
    report_every: f64,
    report: &'a mut dyn FnMut(&CoupledWaves2D, f64),
    sea: Option<&'a RefCell<SeaStates>>,
}

/// The coupled waves' sea state at the two ends of the coupling interval the
/// circulation is in, for the snapshot frames within it (linear in time
/// between them, as the force is).
#[derive(Default)]
struct SeaStates {
    before: Option<(f64, [Vec<f32>; 7])>,
    after: Option<(f64, [Vec<f32>; 7])>,
    /// The fields at a frame's time
    now: [Vec<f32>; 7],
}

impl SeaStates {
    /// The waves' sea state at their time `t`, the start of the next interval
    /// the first time and its end after that.
    fn push(&mut self, waves: &CoupledWaves2D, t: f64) {
        self.before = self.after.take();
        self.after = Some((t, waves.sea_state().as_f32()));
    }

    /// The fields at time `t`, in the order of `SeaStateFields::NAMES`.
    fn at(&mut self, t: f64) -> Vec<&[f32]> {
        let (Some((t1, b)), a) = (&self.after, &self.before) else {
            return Vec::new();
        };
        let (t0, a) = a.as_ref().map_or((*t1, b), |(t0, a)| (*t0, a));
        let w = if t1 > &t0 {
            ((t - t0) / (t1 - t0)).clamp(0.0, 1.0) as f32
        } else {
            1.0
        };
        for ((out, a), b) in self.now.iter_mut().zip(a).zip(b) {
            out.clear();
            out.extend(a.iter().zip(b).map(|(a, b)| a + w * (b - a)));
        }
        self.now.iter().map(Vec::as_slice).collect()
    }
}

/// Run the tide to `t_end`, with the waves of `coupling` exchanging every
/// interval if any. Returns the result and the negative-depth clips.
fn run_tide<BC, I>(
    mut sim: Simulation<SWESolution2D, SWEPhysics2D<BC>, I>,
    q: &mut SWESolution2D,
    t_end: f64,
    coupling: Option<Coupling<'_>>,
    callback: impl FnMut(&SWESolution2D, f64),
) -> (SimulationResult, usize)
where
    BC: SWEBoundaryCondition2D,
    I: TimeIntegrator<SWESolution2D>,
{
    let result = match coupling {
        None => sim.run_with_callback(q, 0.0, t_end, callback),
        Some(Coupling {
            waves,
            interval,
            report_every,
            report,
            sea,
        }) => sim.run_with_exchange(
            q,
            0.0,
            t_end,
            interval,
            |physics, q, t, t_next| {
                if let Some(sea) = sea
                    && sea.borrow().after.is_none()
                {
                    sea.borrow_mut().push(waves, t);
                }
                waves.exchange(physics, q, t, t_next);
                if let Some(sea) = sea {
                    sea.borrow_mut().push(waves, t_next);
                }
                let k = (t_next / report_every).round();
                if (t_next - k * report_every).abs() < 1e-6 * report_every {
                    report(waves, t_next);
                }
            },
            callback,
        ),
    };
    let clips = sim.physics().negative_depth_clips();
    (result, clips)
}

/// The spectral wave model on `domain` (TODO F.4): SWAN's default sources
/// (Komen, the DIA, JONSWAP friction, Battjes–Janssen) on `wave_grid=`
/// (0.04–0.5 Hz), refraction as `turning=` and `implicit=` say, under
/// `wind`; and the JONSWAP sea of `wave_sea=` that comes in through the open
/// boundaries.
fn wave_model(domain: &Domain, opts: &Options, wind: Option<Wind>) -> (WaveModel2D, Vec<f64>) {
    let [nf, nd] = opts.wave_grid;
    let grid = SpectralGrid::new(0.04, 0.5, nf, nd);
    let [hs, tp, from] = opts.wave_sea;
    // Coming from `from` (clockwise from north) is travelling to 270° − from
    // (counter-clockwise from east, the mesh's x)
    let sea = grid.jonswap(hs, tp, 3.3, (270.0 - from).to_radians(), 4.0);
    let mut model = WaveModel2D::new(
        domain.mesh.clone(),
        domain.ops.clone(),
        domain.geom.clone(),
        &domain.bathymetry,
        grid,
        G,
    )
    .with_sources({
        let sources = SourceTerms::swan_defaults(G).with_depth_limit(opts.wave_depth_limit);
        let sources = if opts.wave_implicit_sources {
            sources.with_integration(SourceIntegration::Implicit)
        } else {
            sources
        };
        match opts.wave_limiter {
            Some(limiter) => sources.with_limiter(Some(limiter)),
            None => sources,
        }
    })
    .with_boundary_spectrum(&sea)
    .with_turning_limit(opts.turning)
    .with_implicit_refraction(opts.implicit_refraction)
    .with_implicit_frequency_shift(opts.implicit_refraction)
    .with_substeps(opts.wave_substeps.max(1))
    .with_absorbing_land(opts.wave_absorbing_land);
    if let Some(wind) = wind {
        model = model.with_wind(wind);
    }
    (model, sea)
}

/// The waves of `wave_coupling=` on `domain`, coupled to the tide's
/// `physics`, their force and bed stress ramped up over `ramp_hours`. With
/// `wave_spectra=`, a parent wave model's spectra on the open boundary and
/// its wind, the mean of the boundary's spectra everywhere to start with
/// (and its points for comparison); else the sea of `wave_sea=` at the
/// boundary and everywhere, under the uniform wind of `wind` (none
/// otherwise). The `weather` of `met=`, if any, gives the wind instead: per
/// wave node, not ramped.
fn coupled_waves<BC: SWEBoundaryCondition2D>(
    domain: &Domain,
    physics: &SWEPhysics2D<BC>,
    weather: Option<&GriddedAtmosphere2D>,
    opts: &Options,
    clock: &ModelClock,
) -> Result<(CoupledWaves2D, Option<WavePoints>), Box<dyn std::error::Error>> {
    let weather = weather
        .map(|w| w.on_mesh(&domain.mesh, &domain.ops))
        .transpose()?;
    // Blowing from WIND_DIRECTION (clockwise from north) is blowing to
    // 90° − (WIND_DIRECTION + 180°) counter-clockwise from east
    let wind = (opts.wind && weather.is_none()).then(|| Wind {
        u10: WIND_SPEED,
        direction: (-90.0 - WIND_DIRECTION).to_radians(),
    });
    let parent = match &opts.wave_spectra {
        Some(path) => Some(parent_waves(domain, opts, clock, path)?),
        None => None,
    };
    let (model, sea) = wave_model(domain, opts, wind);
    let header = format!(
        "\nWaves with the tide, exchanging every {} min: {} elements (P{}), {} nodes, {} × {} \
         components",
        opts.wave_coupling,
        domain.mesh.n_elements,
        domain.ops.order,
        model.n_points(),
        opts.wave_grid[0],
        opts.wave_grid[1],
    );
    let (waves, points) = match parent {
        Some(parent) => {
            let boundary =
                BoundarySpectra::for_model(&model, &parent.spectra, opts.wave_neighbours);
            // The boundary's mean spectrum everywhere at the start
            let nc = model.grid.n_components();
            let at_start = boundary.at(0.0);
            let mut mean = vec![0.0; nc];
            for spectrum in at_start.chunks_exact(nc) {
                mean.iter_mut().zip(spectrum).for_each(|(m, x)| *m += x);
            }
            let n = boundary.n_targets().max(1) as f64;
            mean.iter_mut().for_each(|m| *m /= n);
            let start = model.grid.parameters(&mean);
            println!(
                "{header}; the boundary from {} parent points ({} nearest each, {} open-boundary \
                 nodes), H_s {:.2} m at the start on average{}",
                parent.spectra.positions.len(),
                opts.wave_neighbours,
                boundary.n_targets(),
                start.hs,
                match (&weather, &parent.wind) {
                    (Some(_), _) => ", the weather's wind (met=) per node",
                    (None, Some(_)) => ", the parent's wind",
                    (None, None) => ", no wind",
                }
            );
            let state = model.uniform_state(&mean);
            let mut waves = CoupledWaves2D::new(model, state, physics, 0.0)
                .with_ramp(3600.0 * opts.ramp_hours.max(1e-3))
                .with_cfl(opts.wave_cfl)
                .with_force_form(opts.wave_force)
                .with_boundary(boundary);
            match (weather, parent.wind.clone()) {
                (Some(weather), _) => waves = waves.with_gridded_wind(weather),
                (None, Some(series)) => waves = waves.with_wind(series),
                (None, None) => {}
            }
            (waves, Some(parent.points))
        }
        None => {
            let state = model.uniform_state(&sea);
            let [hs, tp, from] = opts.wave_sea;
            println!(
                "{header}; JONSWAP H_s {hs} m, T_p {tp} s from {from}°{}",
                match (&weather, wind) {
                    (Some(_), _) => ", the weather's wind (met=) per node".into(),
                    (None, Some(_)) => format!(", wind {WIND_SPEED} m/s from {WIND_DIRECTION}°"),
                    (None, None) => ", no wind".into(),
                }
            );
            let mut waves = CoupledWaves2D::new(model, state, physics, 0.0)
                .with_ramp(3600.0 * opts.ramp_hours.max(1e-3))
                .with_cfl(opts.wave_cfl)
                .with_force_form(opts.wave_force);
            if let Some(weather) = weather {
                waves = waves.with_gridded_wind(weather);
            }
            (waves, None)
        }
    };

    let (to_waves, to_circulation) = (
        waves.coupling().to_waves(),
        waves.coupling().to_circulation(),
    );
    println!(
        "  wave nodes outside the run's mesh: {} of {} (up to {:.0} m from it); run's nodes \
         outside the waves': {} of {} (up to {:.0} m)",
        to_waves.n_outside(),
        to_waves.n_target_points(),
        to_waves.largest_gap(),
        to_circulation.n_outside(),
        to_circulation.n_target_points(),
        to_circulation.largest_gap()
    );
    Ok((waves, points))
}

/// A parent wave model's spectra (`wave_spectra=`) for a wave mesh: on its
/// coordinates and the run's clock, its wind, and its points for comparison.
struct ParentWaves {
    spectra: PointSpectra,
    wind: Option<WindSeries>,
    points: WavePoints,
}

/// Read `path` (`io::WaveSpectraFile`, e.g. from `met_wave_subset`) for the
/// waves on `domain` under `clock`.
#[cfg(feature = "netcdf")]
fn parent_waves(
    domain: &Domain,
    opts: &Options,
    clock: &ModelClock,
    path: &Path,
) -> Result<ParentWaves, Box<dyn std::error::Error>> {
    let projection = domain
        .projection
        .ok_or("wave_spectra= needs a georeferenced domain (the Frøya data)")?;
    let file = WaveSpectraFile::from_file(path)?;
    let model_time = |unix: f64| clock.model_time(unix);
    let (first, last) = (
        model_time(file.times[0]),
        model_time(file.times[file.times.len() - 1]),
    );
    println!(
        "  Wave spectra {}: {} points, {} → {} UTC{}",
        path.display(),
        file.n_points(),
        clock.format(first),
        clock.format(last),
        file.forecast_reference_time
            .map_or(String::new(), |t| format!(
                " (forecast from {} UTC)",
                clock.format(model_time(t))
            ))
    );
    if first > 0.0 || last < 3600.0 * opts.hours {
        println!(
            "  warning: the run (0 → {:.1} h) reaches beyond the spectra ({:.1} → {:.1} h); \
             the boundary is held at the ends",
            opts.hours,
            first / 3600.0,
            last / 3600.0
        );
    }
    // Geographic east in the mesh axes, at the projection's centre
    let (ex, ey) = east_axis(&projection, projection.ref_lat(), projection.ref_lon());
    let east = ey.atan2(ex);
    let position = |lon: f64, lat: f64| {
        let (x, y) = projection.geo_to_xy(lat, lon);
        [x, y]
    };
    let spectra = file.point_spectra(position, model_time, east);
    let wind = file.wind_series(&[], model_time, east);
    let locator = PointLocator2D::new(&domain.mesh);
    let mut located = Vec::new();
    let mut labels = Vec::new();
    // The parent's points, then the stations of our own
    let points = (0..spectra.positions.len())
        .map(|p| {
            (
                format!("{:.2}°E {:.2}°N", file.longitude[p], file.latitude[p]),
                spectra.positions[p],
            )
        })
        .chain(opts.wave_stations.iter().map(|(name, lon, lat)| {
            (
                format!("{name} ({lon:.4}°E {lat:.4}°N)"),
                position(*lon, *lat),
            )
        }))
        .collect::<Vec<_>>();
    for (label, [x, y]) in points {
        let found = locator.locate([x, y]).map(|point| {
            (
                point.element.as_usize(),
                domain.ops.interpolation_weights(point.r, point.s),
            )
        });
        println!(
            "    {label} at ({:6.1}, {:6.1}) km{}",
            x / 1e3,
            y / 1e3,
            if found.is_none() {
                ", outside the wave mesh"
            } else {
                ""
            }
        );
        labels.push(label);
        located.push(found);
    }
    let n_points = labels.len();
    let points = WavePoints {
        labels,
        located,
        n_parent: spectra.positions.len(),
        east,
        times: spectra.times.clone(),
        hs: file.hs.clone(),
        series: vec![Vec::new(); n_points],
    };
    Ok(ParentWaves {
        spectra,
        wind,
        points,
    })
}

#[cfg(not(feature = "netcdf"))]
fn parent_waves(
    _domain: &Domain,
    _opts: &Options,
    _clock: &ModelClock,
    _path: &Path,
) -> Result<ParentWaves, Box<dyn std::error::Error>> {
    Err("wave_spectra= needs the netcdf feature".into())
}

/// The parent wave model's points: our H_s there against the parent's.
struct WavePoints {
    labels: Vec<String>,
    /// The wave element of each point and the basis weights there (none
    /// outside the wave mesh)
    located: Vec<Option<(usize, Vec<f64>)>>,
    /// The parent's points come first, then `wave_stations=`
    n_parent: usize,
    /// Geographic east in the mesh axes (rad from the x axis)
    east: f64,
    /// The parent's times (s of model time) and H_s, `[time][n_parent]`
    times: Vec<f64>,
    hs: Option<Vec<f64>>,
    /// Per point: t, our H_s, the parent's, our T_p and our mean direction
    /// (coming from, degrees clockwise from north) at the nearest node
    series: Vec<Vec<[f64; 5]>>,
}

impl WavePoints {
    /// The parent's H_s at point `p` and time `t`, linear in time.
    fn parent_hs(&self, p: usize, t: f64) -> f64 {
        let Some(hs) = self.hs.as_ref().filter(|_| p < self.n_parent) else {
            return f64::NAN;
        };
        let (np, times) = (self.n_parent, &self.times);
        let upper = times.partition_point(|&x| x <= t).clamp(1, times.len() - 1);
        if times.len() == 1 {
            return hs[p];
        }
        let (a, b) = (upper - 1, upper);
        let w = ((t - times[a]) / (times[b] - times[a])).clamp(0.0, 1.0);
        (1.0 - w) * hs[a * np + p] + w * hs[b * np + p]
    }

    /// Our H_s and the parent's at every point at `t`: a progress line, and
    /// the series.
    fn report(&mut self, waves: &CoupledWaves2D, t: f64) {
        let model = waves.model();
        let params = model.parameters(waves.state());
        let nn = model.ops.n_nodes;
        let mut line = String::from("    H_s ours/parent:");
        for p in 0..self.labels.len() {
            let parent = self.parent_hs(p, t);
            let ours = self.located[p].as_ref().map_or(f64::NAN, |(k, w)| {
                w.iter()
                    .enumerate()
                    .map(|(i, w)| w * params[k * nn + i].hs)
                    .sum()
            });
            // T_p and the direction at the nearest node (the largest weight)
            let (tp, from) = self.located[p]
                .as_ref()
                .map_or((f64::NAN, f64::NAN), |(k, w)| {
                    let i = (0..w.len())
                        .max_by(|&a, &b| w[a].total_cmp(&w[b]))
                        .unwrap_or(0);
                    let node = &params[k * nn + i];
                    // Travelling to `direction` (rad from the mesh x axis) is
                    // coming from the bearing 270° − (direction − east)
                    let from =
                        (270.0 - (node.direction - self.east).to_degrees()).rem_euclid(360.0);
                    (node.tp, from)
                });
            self.series[p].push([t, ours, parent, tp, from]);
            if parent.is_finite() {
                line += &format!(" {} {ours:.2}/{parent:.2} m;", self.labels[p]);
            } else {
                line += &format!(" {} {ours:.2} m;", self.labels[p]);
            }
        }
        println!("{}", line.trim_end_matches(';'));
    }

    /// Bias and RMSE of our H_s against the parent's at every point from
    /// `after` (s) on, and the series into `<dir>/wave_points.txt`.
    fn summarise(&self, dir: &Path, after: f64) -> std::io::Result<()> {
        println!(
            "\nH_s against the parent wave model at its points, from hour {:.0}:",
            after / 3600.0
        );
        for (label, series) in self.labels.iter().zip(&self.series) {
            let pairs: Vec<(f64, f64)> = series
                .iter()
                .filter(|s| s[0] >= after && s[1].is_finite() && s[2].is_finite())
                .map(|s| (s[1], s[2]))
                .collect();
            if pairs.is_empty() {
                continue;
            }
            let n = pairs.len() as f64;
            let bias = pairs.iter().map(|(a, b)| a - b).sum::<f64>() / n;
            let rmse = (pairs.iter().map(|(a, b)| (a - b).powi(2)).sum::<f64>() / n).sqrt();
            let mean = pairs.iter().map(|(_, b)| b).sum::<f64>() / n;
            println!(
                "  {label}: parent mean {mean:.2} m, ours {:.2} m; bias {bias:+.2} m, RMSE {rmse:.2} m \
                 ({:.0} % of the mean) over {} samples",
                mean + bias,
                100.0 * rmse / mean,
                pairs.len()
            );
        }
        let mut text = String::from(
            "# The waves at the parent wave model's points and the stations: t (s), then per point \
             our H_s (m), the parent's (NaN at a station), our T_p (s) and our mean direction \
             (coming from, degrees clockwise from north)\n# points:",
        );
        for label in &self.labels {
            text += &format!(" {label};");
        }
        text.push('\n');
        let rows = self.series.first().map_or(0, Vec::len);
        for r in 0..rows {
            text += &format!("{:.0}", self.series[0][r][0]);
            for series in &self.series {
                let [_, ours, parent, tp, from] = series[r];
                text += &format!(" {ours:.4} {parent:.4} {tp:.3} {from:.1}");
            }
            text.push('\n');
        }
        fs::write(dir.join("wave_points.txt"), text)
    }
}

/// The depth bands (m) of [`depth_limit_bands`].
const LIMIT_BANDS: [(f64, f64); 5] = [
    (0.0, 0.5),
    (0.5, 1.0),
    (1.0, 2.0),
    (2.0, 5.0),
    (5.0, f64::INFINITY),
];

/// The wave nodes at or above breaking's `H_rms = γ d` (to 1e-6), the edge
/// that `SourceTerms::with_depth_limit` holds them to, per band of
/// [`LIMIT_BANDS`], and of all the nodes in each band; none without breaking.
fn depth_limit_bands(
    model: &WaveModel2D,
    params: &[dg_rs::waves::WaveParameters],
) -> Option<[(usize, usize); 5]> {
    let (_, gamma) = model.sources.breaking?;
    let mut bands = [(0, 0); 5];
    for (w, &d) in params.iter().zip(model.depth()) {
        let Some(b) = LIMIT_BANDS
            .iter()
            .position(|&(lo, hi)| (lo..hi).contains(&d))
        else {
            continue;
        };
        bands[b].1 += 1;
        if w.hs / 2f64.sqrt() >= (1.0 - 1e-6) * gamma * d {
            bands[b].0 += 1;
        }
    }
    Some(bands)
}

/// A line of [`depth_limit_bands`].
fn report_depth_limit(model: &WaveModel2D, params: &[dg_rs::waves::WaveParameters]) {
    let Some(bands) = depth_limit_bands(model, params) else {
        return;
    };
    let total: usize = bands.iter().map(|b| b.0).sum();
    let mut line = format!(
        "    at breaking's H_rms = γd ({}): {total} nodes;",
        if model.sources.depth_limit {
            "the depth limit"
        } else {
            "or above, no limit"
        }
    );
    for ((lo, hi), (at, all)) in LIMIT_BANDS.iter().zip(bands) {
        let band = if hi.is_finite() {
            format!("{lo}–{hi} m")
        } else {
            format!("≥ {lo} m")
        };
        line += &format!(" {band} {at} of {all},");
    }
    println!("{}", line.trim_end_matches(','));
}

/// The progress line of the waves at `t`: the largest H_s and where, the
/// mean over wave nodes at least 3 m deep, and the largest force on the tide
/// and where.
fn report_waves(
    domain: &Domain,
    waves: &CoupledWaves2D,
    t: f64,
    (t_last, last): (f64, CoupledWavesStats),
) {
    let model = waves.model();
    let stats = waves.stats();
    let params = model.parameters(waves.state());
    let node = |mesh: &Mesh2D, ops: &DGOperators2D, p: usize| {
        let (k, i) = (p / ops.n_nodes, p % ops.n_nodes);
        mesh.reference_to_physical(ElementIndex::new(k), ops.nodes_r[i], ops.nodes_s[i])
    };
    let largest = |values: &mut dyn Iterator<Item = f64>| {
        values.enumerate().fold(
            (0, 0.0),
            |best, (p, v)| if v > best.1 { (p, v) } else { best },
        )
    };
    let (highest, hs_max) = largest(&mut params.iter().map(|p| p.hs));
    let (steepest, ratio_max) =
        largest(&mut params.iter().zip(model.depth()).map(|(p, &d)| p.hs / d));
    let deep: Vec<f64> = params
        .iter()
        .zip(model.depth())
        .filter(|&(_, &d)| d >= 3.0)
        .map(|(p, _)| p.hs)
        .collect();
    let hs_mean = deep.iter().sum::<f64>() / deep.len().max(1) as f64;
    let [x, y] = node(&model.mesh, &model.ops, highest);
    let [rx, ry] = node(&model.mesh, &model.ops, steepest);
    let force = waves.force().unwrap_or(&[]);
    let (strongest, f_max) = largest(&mut force.iter().map(|f| f[0].hypot(f[1])));
    let [fx, fy] = node(&domain.mesh, &domain.ops, strongest);
    println!(
        "  waves at {:6.2} h: H_s ≤ {hs_max:.2} m at ({:6.1}, {:6.1} km), mean {hs_mean:.2} m where \
         ≥ 3 m deep; H_s/d ≤ {ratio_max:.2} at ({:6.1}, {:6.1} km; {:.2} m deep); force ≤ {:.2e} \
         m²/s² at ({:6.1}, {:6.1} km); step {:.2} s, {:.0} s wall per model hour",
        t / 3600.0,
        x / 1e3,
        y / 1e3,
        rx / 1e3,
        ry / 1e3,
        model.depth()[steepest],
        f_max,
        fx / 1e3,
        fy / 1e3,
        (t - t_last) / (stats.wave_steps - last.wave_steps).max(1) as f64,
        3600.0
            * (stats.stepping_time + stats.exchange_time - last.stepping_time - last.exchange_time)
            / (t - t_last)
    );
    report_depth_limit(model, &params);
    // What sets the step in the last exchange's level and currents
    let limits = model.time_step_limits(waves.cfl());
    let (name, limit) = limits.binding();
    let [x, y] = node(&model.mesh, &model.ops, limit.point);
    let [u, v] = model.current()[limit.point];
    println!(
        "    step {:.2} s set by {name} at ({:6.1}, {:6.1} km; {:.1} m deep, current {:.2} m/s, \
         {:.3} Hz); propagation alone {:.2} s",
        limit.dt,
        x / 1e3,
        y / 1e3,
        model.depth()[limit.point],
        u.hypot(v),
        model.grid.sigma[limit.frequency] / std::f64::consts::TAU,
        limits.propagation.dt
    );
}

/// The snapshot file's metadata: the title (with `suffix`), the stations in
/// mesh coordinates and the mesh's local projection, for the viewer.
fn snapshot_metadata(
    domain: &Domain,
    stations: &[Station],
    suffix: &str,
) -> Vec<(&'static str, String)> {
    let title = match domain.projection {
        // The date is the header's clock
        Some(_) => "Frøya–Smøla–Hitra",
        None => "Synthetic basin with an island and a beach",
    };
    let mut metadata = vec![
        ("title", format!("{title}{suffix}")),
        ("source", "froya_real_data".to_string()),
    ];
    if let Some(projection) = &domain.projection {
        for s in stations {
            let (x, y) = projection.geo_to_xy(s.latitude, s.longitude);
            metadata.push(("station", format!("{},{x:.1},{y:.1}", s.name)));
        }
        // The mesh's local projection, for the viewer's land around the domain
        metadata.push((
            "projection",
            format!("local,{},{}", projection.ref_lat(), projection.ref_lon()),
        ));
    }
    metadata
}

/// Every station's record to `output_dir`, and its comparison with its gauge
/// and NorKyst after the spin-up.
fn report_stations(
    stations: &[Station],
    output_dir: &Path,
    opts: &Options,
    clock: &ModelClock,
    station_atlas: Option<&TidalAtlas>,
) -> Result<(), Box<dyn std::error::Error>> {
    let analysis_start = clock.unix(opts.spinup_hours * 3600.0);
    for s in stations {
        let slug = slug(&s.name);
        let path = output_dir.join(format!("station_{slug}.txt"));
        let series = TimeSeries::new(&s.times, &s.eta).with_name(s.name.clone());
        let mut file =
            TideGaugeFile::from_time_series(series).with_station(s.gauge.as_ref().map_or_else(
                || TideGaugeStation::new(s.name.clone(), s.longitude, s.latitude),
                |g| g.station.clone(),
            ));
        file.datum = Some("MSL".into());
        file.units = Some("m".into());
        write_tide_gauge_file(&path, &file)?;
        let current_path = output_dir.join(format!("currents_{slug}.txt"));
        let currents = CurrentTimeSeries::new(&s.times, &s.u_east, &s.v_north);
        let station =
            ADCPStation::new(s.name.clone(), s.longitude, s.latitude).with_water_depth(s.depth);
        write_adcp_file(&current_path, &station, &currents)?;
        println!(
            "\nStation {} → {}, {}",
            s.name,
            path.display(),
            current_path.display()
        );
        let sampled = if s.offset > 0.0 {
            format!("{:.0} m from the station", s.offset)
        } else {
            "at the station".into()
        };
        println!("  sampled {sampled}, {:.1} m deep", s.depth);
        let atlas_point = station_atlas
            .and_then(|a| a.nearest(s.longitude, s.latitude))
            .filter(|&(_, d)| d <= STATION_MAX_OFFSET);
        if let Some((p, d)) = atlas_point {
            println!(
                "  NorKyst: atlas point {d:.0} m from the station, {:.0} m deep",
                p.depth
            );
        }
        if let Some(gauge) = &s.gauge
            && let Err(e) = report_station(s, gauge, analysis_start, atlas_point.map(|(p, _)| p))
        {
            println!("  no harmonic comparison: {e}");
        }
        if let Err(e) = report_currents(s, analysis_start, atlas_point.map(|(p, _)| p)) {
            println!("  no current comparison: {e}");
        }
    }
    Ok(())
}

/// The domain's mean profile of T and S (°C, psu) in `z` (m), on a 1 m
/// grid: a horizontally uniform stratification for the references of the PGF,
/// the tracer limiter and the open faces.
///
/// At each height the mean of the columns that reach it, interpolated between
/// their layer centres, then made statically stable: adjacent heights where
/// the density decreases downwards are pooled into their weighted mean
/// (pool-adjacent-violators on `density(T, S)`, which is linear in T and S,
/// so a pool's density is its mean's). The columns that reach a height change
/// with depth (below 100 m only the deep holes), and that mean alone was
/// inverted in places: as a reference, and with `uniform` as the state, its
/// overturning drove a bed jet of 0.26 m/s within ten minutes in a 120 m hole
/// at Mausund (TODO P1.3). Taking every column at every height instead
/// (extended below its bed) is stable but puts the shallow columns' bottom
/// water in the deep holes: 10.1 °C at 100 m where the holes have 8.3 °C.
#[derive(Clone)]
struct MeanProfile {
    /// Height of the lowest grid point (m)
    z0: f64,
    temp: Vec<f64>,
    salt: Vec<f64>,
}

impl MeanProfile {
    fn of(
        state: &dg_rs::solver::state::Solution3D,
        bed: &[f64],
        sigma: &dg_rs::vertical::SigmaGrid,
        use_column: impl Fn(usize) -> bool,
        density: impl Fn(f64, f64) -> f64,
    ) -> Option<Self> {
        let nl = state.n_levels;
        let deepest = bed.iter().copied().fold(0.0, f64::min);
        let n = (-deepest).ceil() as usize + 1;
        let z0 = -((n - 1) as f64);
        let (mut sum_t, mut sum_s, mut count) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        for (idx, &b) in bed.iter().enumerate() {
            if b >= 0.0 || !use_column(idx) {
                continue;
            }
            // The layer centres at mean sea level, from the bed up
            let z = |l: usize| sigma.sigma_rho()[l] * -b;
            let (t, sal) = (
                &state.temp[idx * nl..(idx + 1) * nl],
                &state.salt[idx * nl..(idx + 1) * nl],
            );
            let mut l = 0;
            for i in 0..n {
                let zi = z0 + i as f64;
                // Within the column: its bed to the surface
                if zi < b {
                    continue;
                }
                while l + 1 < nl && z(l + 1) <= zi {
                    l += 1;
                }
                let (value_t, value_s) = if zi <= z(0) {
                    (t[0], sal[0])
                } else if l + 1 >= nl {
                    (t[nl - 1], sal[nl - 1])
                } else {
                    let w = (zi - z(l)) / (z(l + 1) - z(l));
                    (
                        t[l] + w * (t[l + 1] - t[l]),
                        sal[l] + w * (sal[l + 1] - sal[l]),
                    )
                };
                sum_t[i] += value_t;
                sum_s[i] += value_s;
                count[i] += 1.0;
            }
        }
        if count.iter().all(|&c| c == 0.0) {
            return None;
        }
        // Pools from the surface down: (Σ T, Σ S, weight, heights)
        let mut pools: Vec<(f64, f64, f64, usize)> = Vec::new();
        for i in (0..n).rev() {
            if count[i] == 0.0 {
                // Below every column: the pool above extends
                if let Some(last) = pools.last_mut() {
                    last.3 += 1;
                }
                continue;
            }
            pools.push((sum_t[i], sum_s[i], count[i], 1));
            while pools.len() >= 2 {
                let rho = |p: &(f64, f64, f64, usize)| density(p.0 / p.2, p.1 / p.2);
                let (lower, upper) = (pools[pools.len() - 1], pools[pools.len() - 2]);
                if rho(&lower) >= rho(&upper) {
                    break;
                }
                pools.pop();
                let merged = pools.last_mut().expect("two pools");
                *merged = (
                    merged.0 + lower.0,
                    merged.1 + lower.1,
                    merged.2 + lower.2,
                    merged.3 + lower.3,
                );
            }
        }
        // Back to heights, from the bottom up
        let (mut temp, mut salt) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for &(t, s, w, len) in pools.iter().rev() {
            temp.extend(std::iter::repeat_n(t / w, len));
            salt.extend(std::iter::repeat_n(s / w, len));
        }
        // Heights above the top layer centre of every column take the top
        let (t_top, s_top) = (*temp.last()?, *salt.last()?);
        temp.resize(n, t_top);
        salt.resize(n, s_top);
        Some(Self { z0, temp, salt })
    }

    /// T and S at height `z`, linear between grid points and held beyond them.
    fn at(&self, z: f64) -> (f64, f64) {
        let x = (z - self.z0).clamp(0.0, (self.temp.len() - 1) as f64);
        let i = (x.floor() as usize).min(self.temp.len().saturating_sub(2));
        let w = (x - i as f64).clamp(0.0, 1.0);
        let j = (i + 1).min(self.temp.len() - 1);
        (
            self.temp[i] + w * (self.temp[j] - self.temp[i]),
            self.salt[i] + w * (self.salt[j] - self.salt[i]),
        )
    }
}

/// The tide in 3D (`levels=N tide3d=1`, TODO P1.3): see the module docs.
/// The rivers of `rivers=` on the mesh, those whose mouth is within
/// `river_snap_km` of it (the others are listed and left out).
#[cfg(feature = "netcdf")]
fn rivers_3d(
    domain: &Domain,
    opts: &Options,
    clock: &ModelClock,
    t_end: f64,
) -> Result<Option<dg_rs::source::RiverSources>, Box<dyn std::error::Error>> {
    use dg_rs::io::RiverForcing;
    use dg_rs::source::{RiverProfile, RiverSources};

    let Some(path) = &opts.rivers else {
        return Ok(None);
    };
    let projection = domain
        .projection
        .ok_or("rivers= needs a georeferenced domain (the Frøya data files)")?;
    let forcing = RiverForcing::from_file(path)?;
    // Daily values: the run must lie within the file's days
    let (first, last) = forcing.time_range();
    let (start, end) = (clock.unix(0.0), clock.unix(t_end));
    if start < first || end > last {
        return Err(format!(
            "{}: the rivers run from {} to {}, the run from {} to {} \
             (`norkyst_river_subset start= days=`)",
            path.display(),
            clock.format(clock.model_time(first)),
            clock.format(clock.model_time(last)),
            clock.format(0.0),
            clock.format(t_end)
        )
        .into());
    }
    let rivers = forcing.rivers(
        &projection,
        clock,
        &RiverProfile::TopFraction(opts.river_top),
    )?;
    let (sources, skipped) = RiverSources::on_mesh(
        rivers,
        &domain.mesh,
        &domain.geom,
        1000.0 * opts.river_snap_km,
    );
    // Over the run, hourly
    let hours = (t_end / 3600.0).ceil().max(1.0) as usize;
    let mean = |i: usize| {
        (0..=hours)
            .map(|h| sources.discharge(i, (h as f64 * 3600.0).min(t_end)))
            .sum::<f64>()
            / (hours + 1) as f64
    };
    let total: f64 = (0..sources.len()).map(mean).sum();
    println!(
        "  Rivers: {} of {} from {} ({total:.1} m³/s over the run), over the top {:.0} % of \
         the column; {} more than {} km from the mesh left out",
        sources.len(),
        forcing.len(),
        path.display(),
        100.0 * opts.river_top,
        skipped.len(),
        opts.river_snap_km
    );
    let mut order: Vec<usize> = (0..sources.len()).collect();
    order.sort_by(|&a, &b| mean(b).total_cmp(&mean(a)));
    for &i in order.iter().take(5) {
        println!(
            "    river {}: {:.1} m³/s into element {} (snapped {:.0} m)",
            sources.river(i).name,
            mean(i),
            sources.element(i).as_usize(),
            sources.snap_distance(i)
        );
    }
    for error in skipped.iter().take(5) {
        println!("    left out: {error}");
    }
    Ok(Some(sources))
}

#[cfg(not(feature = "netcdf"))]
fn rivers_3d(
    _domain: &Domain,
    opts: &Options,
    _clock: &ModelClock,
    _t_end: f64,
) -> Result<Option<dg_rs::source::RiverSources>, Box<dyn std::error::Error>> {
    match opts.rivers {
        Some(_) => Err("rivers= needs the `netcdf` feature".into()),
        None => Ok(None),
    }
}

fn tidal_run_3d(
    domain: &Domain,
    opts: &Options,
    parent: Option<OceanModelState>,
    parent_reader: Option<Arc<OceanModelReader>>,
) -> Result<(), Box<dyn std::error::Error>> {
    use dg_rs::boundary::{
        ColumnContext3D, DeepReference, Nesting3D, NestingBand3D, OceanColumnsOptions,
        OceanModelColumns, ParentColumn, ParentColumns3D, ReferenceColumns,
    };
    // e-folding depth of the parent's bottom anomaly below its bed (m)
    const DEEP_DECAY: f64 = 10.0;
    use dg_rs::io::Restart3D;
    use dg_rs::physics::{
        BottomDrag3D, EquationOfState, Forcing, Hydrostatic3D, ImplicitVerticalAdvection,
        LinearEOS, TimeStepLimit, VerticalMixing,
    };
    use dg_rs::simulation::{RunContext3D, Simulation3D};
    use dg_rs::solver::state::Solution3D;
    use dg_rs::solver::{TracerLimiter3DConfig, TracerLimiterType3D, TracerReferenceProfile};
    use dg_rs::source::SpongeProfile;
    use dg_rs::time::ModeSplitIntegrator;
    use dg_rs::vertical::{SigmaGrid, SongHaidvogelStretching};

    const RHO0: f64 = 1025.0;
    let TideSetup {
        t_end,
        clock,
        mut stations,
        station_atlas,
        open,
        forcing,
        band,
        weather,
        wind_3d,
        fluxes_3d,
    } = tide_setup(domain, opts, parent, true)?;
    // `debug_3d=rest`: walls all round, no band and no nesting: the
    // stratification at rest, as `levels=N` alone runs it
    let rest = opts.debug_3d.split(',').any(|f| f == "rest");
    let wall = Reflective2D::new();
    let bc = if rest {
        MultiBoundaryCondition2D::new(&wall)
    } else {
        MultiBoundaryCondition2D::new(&wall).with_open(open.as_ref())
    };
    let mut builder = domain.builder_without_friction(bc);
    if let Some(band) = band.filter(|_| !rest) {
        builder = builder.with_source(band);
    }
    // The pressure gradient, a depth-uniform force, on the 2D module; the
    // wind stress on the columns (below)
    if let Some(weather) = weather {
        builder = builder.with_source(weather);
    } else if opts.wind {
        builder = builder.with_source(pressure());
    }
    let swe = builder.build();

    // The water at rest at mean sea level, stratified as the parent model at
    // the start where it has profiles, else as the summer pycnocline
    let nl = opts.levels;
    let sigma = Arc::new(SigmaGrid::new(
        nl,
        SongHaidvogelStretching::new(5.0, 0.4, 10.0),
    ));
    let eos = LinearEOS::default();
    let (ne, nn) = (domain.mesh.n_elements, domain.ops.n_nodes);
    let bed = &domain.bathymetry.data;
    let mut state = Solution3D::new(ne, nn, nl);
    // The parent's columns, with the tracers below its bed relaxing to
    // `deep` if given (`DeepReference`)
    let parent_columns = |deep: Option<DeepReference>| -> Option<Arc<dyn ParentColumns3D>> {
        match (&parent_reader, domain.projection) {
            (Some(reader), Some(projection)) if reader.has_tracer_profiles() => {
                let columns = OceanModelColumns::new(
                    reader.clone(),
                    &domain.mesh,
                    &domain.ops,
                    projection,
                    clock,
                    OceanColumnsOptions::default(),
                );
                Some(Arc::new(match deep {
                    Some(deep) => columns.with_deep_reference(deep),
                    None => columns,
                }))
            }
            _ => None,
        }
    };
    let mut columns = parent_columns(None);
    let summer = |z: f64| -> (f64, f64) {
        let step = 0.5 * (1.0 + ((z + 15.0) / 4.0).tanh());
        (
            eos.t0 + 4.0 * step + 0.002 * z.min(0.0),
            eos.s0 - 1.5 * step,
        )
    };
    // `debug_3d=` switches for diagnosing the 3D tide: `nonest` (the open
    // faces relax to the state at rest, not the parent), `constant` (constant
    // mixing instead of GLS), `nolimiter`, `uniform` (the parent's mean
    // profile everywhere: no horizontal density gradients), `flat` (no
    // stratification), `held` (the parent's tracers held below its bed),
    // `still` (no initial shear from the parent), `explicit` (no implicit
    // part of the vertical advection)
    let dbg = |flag: &str| opts.debug_3d.split(',').any(|f| f == flag);
    let mut covered = vec![false; ne * nn];
    // The parent's columns at every wet node at the start
    let sample = |columns: &Option<Arc<dyn ParentColumns3D>>,
                  state: &mut Solution3D,
                  covered: &mut [bool]| {
        let (mut u, mut v, mut t_col, mut s_col) =
            (vec![0.0; nl], vec![0.0; nl], vec![0.0; nl], vec![0.0; nl]);
        for idx in 0..ne * nn {
            let eta = bed[idx].max(0.0);
            state.eta.data[idx] = eta;
            covered[idx] = false;
            let Some(columns) = columns.as_ref().filter(|_| bed[idx] < 0.0) else {
                continue;
            };
            let (k, i) = (idx / nn, idx % nn);
            let [x, y] = domain.mesh.reference_to_physical(
                ElementIndex::new(k),
                domain.ops.nodes_r[i],
                domain.ops.nodes_s[i],
            );
            let ctx = ColumnContext3D {
                time: 0.0,
                node: idx,
                position: (x, y),
                bed: bed[idx],
                eta,
            };
            let out = ParentColumn {
                u: &mut u,
                v: &mut v,
                temp: &mut t_col,
                salt: &mut s_col,
            };
            if columns.column(&ctx, &sigma, out) {
                let column = idx * nl..(idx + 1) * nl;
                state.u[column.clone()].copy_from_slice(&u);
                state.v[column.clone()].copy_from_slice(&v);
                state.temp[column.clone()].copy_from_slice(&t_col);
                state.salt[column].copy_from_slice(&s_col);
                covered[idx] = true;
            }
        }
    };
    sample(&columns, &mut state, &mut covered);
    let n_covered = covered.iter().filter(|&&c| c).count();
    // The mean profile of the parent's columns, else the summer pycnocline's
    let mean = if n_covered > 0 {
        MeanProfile::of(
            &state,
            bed,
            &sigma,
            |idx| covered[idx],
            |t, s| eos.compute_density(t, s, 0.0),
        )
    } else {
        None
    };
    // Sampled again with the tracers below the parent's bed relaxing to the
    // mean profile: held, its bottom water ran down the deep holes' walls
    if let Some(mean) = mean.clone().filter(|_| !dbg("held")) {
        columns = parent_columns(Some(DeepReference {
            profile: Arc::new(move |d| mean.at(-d)),
            decay: DEEP_DECAY,
        }));
        sample(&columns, &mut state, &mut covered);
    }
    if dbg("uniform") || dbg("flat") {
        covered.fill(false);
    }
    let profile = |z: f64| {
        if dbg("flat") {
            (eos.t0, eos.s0)
        } else {
            mean.as_ref().map_or_else(|| summer(z), |m| m.at(z))
        }
    };
    for idx in (0..ne * nn).filter(|&idx| !covered[idx]) {
        let eta = state.eta.data[idx];
        for (l, &s) in sigma.sigma_rho().iter().enumerate() {
            let z = eta + s * (eta - bed[idx]);
            (state.temp[idx * nl + l], state.salt[idx * nl + l]) = profile(z);
        }
    }
    // The velocity: the parent's shear about the 2D module's depth mean,
    // zero at the start (the tide ramps up from rest), where the parent
    // covers a column at least the thin depth deep; else at rest
    // (`debug_3d=still`: at rest everywhere, the parent's density field
    // adjusting without its currents)
    let mut n_sheared = 0;
    for idx in 0..ne * nn {
        let column = idx * nl..(idx + 1) * nl;
        let sheared =
            covered[idx] && state.eta.data[idx] - bed[idx] >= opts.thin_depth() && !dbg("still");
        for field in [&mut state.u, &mut state.v] {
            let values = &mut field[column.clone()];
            let mean = if sheared {
                sigma.depth_average(values)
            } else {
                0.0
            };
            values
                .iter_mut()
                .for_each(|x| *x = if sheared { *x - mean } else { 0.0 });
        }
        n_sheared += usize::from(sheared);
    }
    let largest_shear = state
        .u
        .iter()
        .zip(&state.v)
        .map(|(u, v)| u.hypot(*v))
        .fold(0.0, f64::max);
    let (t_lo, t_hi) = state
        .temp
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &t| {
            (a.min(t), b.max(t))
        });
    let (s_lo, s_hi) = state
        .salt
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &t| {
            (a.min(t), b.max(t))
        });
    println!(
        "\n3D model: {ne} elements × {nn} nodes × {nl} levels = {:.2} M points, P{}",
        (ne * nn * nl) as f64 / 1e6,
        domain.ops.order
    );
    match &columns {
        Some(_) => println!(
            "  Initial T, S from NorKyst at {n_covered} of {} wet nodes (the others its mean \
             profile): T {t_lo:.2}–{t_hi:.2} °C, S {s_lo:.2}–{s_hi:.2}; its shear about a \
             depth mean at rest at {n_sheared} nodes (largest {largest_shear:.2} m/s)",
            bed.iter().filter(|&&b| b < 0.0).count()
        ),
        None => println!(
            "  Initial T, S: the summer pycnocline (no NorKyst profiles): T {t_lo:.2}–{t_hi:.2} °C"
        ),
    }
    let (t_top, s_top) = profile(-0.5);
    let (t_30, s_30) = profile(-30.0);
    let (t_100, s_100) = profile(-100.0);
    println!(
        "  Mean profile: {t_top:.2} °C, {s_top:.2} at 0.5 m; {t_30:.2} °C, {s_30:.2} at 30 m; \
         {t_100:.2} °C, {s_100:.2} at 100 m"
    );

    // The open faces relax to the parent's columns, or to the state at rest
    let band_3d = NestingBand3D {
        width: 1000.0 * opts.band_km,
        profile: SpongeProfile::default(),
        velocity_timescale: Some(60.0 * opts.band_minutes),
        tracer_timescale: Some(60.0 * opts.band_minutes),
    };
    let relaxed_to: Arc<dyn ParentColumns3D> = match &columns {
        Some(columns) if !dbg("nonest") => columns.clone(),
        _ => Arc::new(ReferenceColumns::from_state(&state)),
    };
    let nesting = Nesting3D::new(
        relaxed_to,
        &domain.mesh,
        &domain.ops,
        nl,
        &[BoundaryTag::Open],
        &band_3d,
    )
    .ok();
    println!(
        "  Open faces: {} over a {} km band, {} min",
        match (&nesting, &columns) {
            (None, _) => "none",
            (Some(_), Some(_)) if !dbg("nonest") => "nested in NorKyst's u, v, T, S",
            (Some(_), _) => "relaxed to the stratification at rest",
        },
        opts.band_km,
        opts.band_minutes
    );

    let deepest = bed.iter().copied().fold(0.0, f64::min);
    let limiter = TracerLimiter3DConfig {
        limiter_type: TracerLimiterType3D::HorizontalKuzmin { relaxation: 1.0 },
        ..TracerLimiter3DConfig::default()
    }
    .with_reference_profile(TracerReferenceProfile::from_fn(
        deepest - 1.0,
        1.0,
        4001,
        profile,
    ));
    let mut physics = Hydrostatic3D::new(
        domain.mesh.clone(),
        domain.ops.clone(),
        domain.geom.clone(),
        sigma.clone(),
        domain.bathymetry.clone(),
        Arc::new(CoriolisSource2D::f_plane(F_CORIOLIS)),
        eos,
        if dbg("constant") {
            Box::new(dg_rs::physics::ConstantMixing::new(1e-3, 0.0))
                as Box<dyn VerticalMixing + Send + Sync>
        } else {
            Box::new((opts.mixing.1)())
        },
        swe,
        Forcing {
            // The `wind` option's uniform stress (with `met=` the columns'
            // own, below)
            surface_stress: if wind_3d.is_none() && opts.wind {
                // From WIND_DIRECTION (meteorological), as `from_direction`
                let from = WIND_DIRECTION.to_radians();
                let (u_10, v_10) = (-WIND_SPEED * from.sin(), -WIND_SPEED * from.cos());
                let (tau_x, tau_y) = WindStress2D::constant(u_10, v_10)
                    .with_drag(DragCoefficient::LargePond)
                    .compute_stress(u_10, v_10);
                [tau_x, tau_y]
            } else {
                [0.0, 0.0]
            },
            bottom_stress: [0.0, 0.0],
            surface_heat_flux: 0.0,
        },
        G,
        RHO0,
    )
    .with_bottom_drag(BottomDrag3D::log_layer(0.003))
    .with_min_column_depth(opts.thin_depth())
    .with_smagorinsky_viscosity(0.1)
    .with_tracer_limiter(if dbg("nolimiter") {
        TracerLimiter3DConfig::none()
    } else {
        limiter
    });
    if let Some(nesting) = nesting.filter(|_| !rest) {
        physics = physics.with_nesting(nesting);
    }
    if let Some(wind) = wind_3d {
        physics = physics.with_surface_stress(wind);
    }
    if let Some(fluxes) = fluxes_3d {
        physics = physics
            .with_surface_fluxes(fluxes)
            .with_shortwave_absorption(opts.jerlov);
    }
    if let Some(rivers) = rivers_3d(domain, opts, &clock, t_end)? {
        physics = physics.with_rivers(rivers);
    }
    // The horizontal viscosity once per step (`visc_step`, as ROMS)
    if dbg("visc_step") {
        physics = physics.with_viscosity_once_per_step();
    }
    // The vertical advection beyond an explicit Courant number implicitly:
    // the columns just deeper than the thin depth set the step otherwise
    if !dbg("explicit") {
        physics = physics.with_implicit_vertical_advection(ImplicitVerticalAdvection::default());
    }
    physics.update_density(&mut state);
    // The PGF about the mean profile (Mellor et al. 1998)
    let physics = physics.with_reference_profile(&state, |z| {
        let (t, s) = profile(z);
        eos.compute_density(t, s, z)
    });

    let output_dir = opts
        .output
        .clone()
        .unwrap_or_else(|| Path::new("output").join(format!("{}_3d", domain.name)));
    fs::create_dir_all(&output_dir)?;
    write_clock(&output_dir, &clock)?;
    println!(
        "\nTides in 3D: {forcing}, {:.2} h{}, GLS {} → {}",
        opts.hours,
        if !opts.met.is_empty() {
            ", gridded wind and pressure"
        } else if opts.wind {
            ", wind and pressure"
        } else {
            ""
        },
        opts.mixing.0,
        output_dir.display()
    );
    // A resumed run (`resume=`): the physics above is built from the initial
    // state as the restarted run's was (the PGF's and the band's references
    // are of it); the state, the time and the records continue the file's
    let restart = match &opts.resume {
        Some(path) => {
            let restart = Restart3D::read(path)?;
            println!(
                "  Resuming from {} at {:.2} h (written by: {})",
                path.display(),
                restart.time / 3600.0,
                restart.metadata("command").unwrap_or("?")
            );
            for (j, station) in stations.iter_mut().enumerate() {
                if restart.metadata(&format!("station.{j}")) != Some(station.name.as_str()) {
                    return Err(format!(
                        "station {j} is {}, the restart's {:?}",
                        station.name,
                        restart.metadata(&format!("station.{j}"))
                    )
                    .into());
                }
                let record = |field: &str| {
                    restart
                        .extra(&format!("station.{j}.{field}"))
                        .map(<[f64]>::to_vec)
                        .ok_or(format!("the restart has no station.{j}.{field}"))
                };
                station.times = record("times")?;
                station.eta = record("eta")?;
                station.u_east = record("u_east")?;
                station.v_north = record("v_north")?;
            }
            if restart
                .metadata(&format!("station.{}", stations.len()))
                .is_some()
            {
                return Err("the restart has more stations than this run".into());
            }
            Some(restart)
        }
        None => None,
    };
    let t_start = restart.as_ref().map_or(0.0, |r| r.time);
    let snapshot_minutes = (opts.snapshot_minutes > 0.0).then_some(opts.snapshot_minutes);
    let base_minutes = if stations.is_empty() {
        opts.output_minutes
            .min(snapshot_minutes.unwrap_or(f64::INFINITY))
    } else {
        opts.station_minutes
    };
    let every = |minutes: f64| (minutes / base_minutes).round().max(1.0) as usize;
    let (interval, output_every) = (base_minutes * 60.0, every(opts.output_minutes));
    let snapshot_every = snapshot_minutes.map(every);
    let snapshot_path = output_dir.join(format!("{}.dgsnap", domain.name));
    let mut snapshot = match snapshot_every {
        // A resumed run continues its file, without the frames the stopped
        // run wrote after the restart
        Some(every) if restart.is_some() && snapshot_path.exists() => {
            println!(
                "  Snapshot file: {} every {} min, continued after {:.2} h",
                snapshot_path.display(),
                every as f64 * base_minutes,
                t_start / 3600.0
            );
            Some(SnapshotWriter::append(&snapshot_path, t_start, 0.0)?)
        }
        Some(every) => {
            let path = snapshot_path.clone();
            println!(
                "  Snapshot file: {} every {} min",
                path.display(),
                every as f64 * base_minutes
            );
            let metadata = snapshot_metadata(domain, &stations, " in 3D");
            let metadata: Vec<(&str, &str)> =
                metadata.iter().map(|(k, v)| (*k, v.as_str())).collect();
            Some(SnapshotWriter::create_3d(
                path,
                &domain.mesh,
                &domain.ops,
                bed,
                &sigma,
                Some(&clock),
                &metadata,
            )?)
        }
        None => None,
    };
    println!(
        "  time  |  η range (m)     | max |ū| (m/s) at (x, y km; h m) | max |u| (m/s), level at (x, y km; h m, element) | T range (°C), in thin columns | wall"
    );

    let mut q = domain.at_rest();
    let barotropic = |s: &Solution3D, q: &mut SWESolution2D| {
        for k in ElementIndex::iter(ne) {
            for i in 0..nn {
                let idx = k.as_usize() * nn + i;
                let h = (s.eta.data[idx] - bed[idx]).max(0.0);
                q.set_state(
                    k,
                    i,
                    SWEState2D::new(h, h * s.ubar.data[idx], h * s.vbar.data[idx]),
                );
            }
        }
    };
    // The callbacks so far and the next sample time, continued by a resumed
    // run (its first callback repeats the restart's time, already sampled)
    let (n_callbacks_done, next_sample_at) = match &restart {
        Some(restart) => match restart.extra("driver.counters") {
            Some(&[n, next]) => (n as usize, next),
            _ => return Err("the restart has no driver.counters".into()),
        },
        None => (0, 0.0),
    };
    let mut n_callbacks = n_callbacks_done;
    let mut write_error = None;
    let start = Instant::now();
    // Called every step: the steps between the samples, and their range
    let (mut last_t, mut next_sample) = (f64::NAN, next_sample_at);
    let (mut steps, mut dt_lo, mut since) = (0usize, f64::INFINITY, t_start);
    // What set the shortest step of the interval
    let mut binding: Option<(&str, TimeStepLimit)> = None;
    // A restart every `restart_hours`, at the first progress line at or after
    // each multiple (`resume=` continues from it)
    let restart_path = output_dir.join(format!("{}.restart", domain.name));
    let restart_interval = 3600.0 * opts.restart_hours;
    let mut next_restart = if restart_interval > 0.0 {
        println!(
            "  Restart file: {} every {} h",
            restart_path.display(),
            opts.restart_hours
        );
        ((t_start / restart_interval).floor() + 1.0) * restart_interval
    } else {
        f64::INFINITY
    };
    let command = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    let callback =
        |s: &Solution3D,
         t: f64,
         run: &RunContext3D<LinearEOS, Box<dyn VerticalMixing + Send + Sync>, _>| {
            let physics = run.physics;
            if last_t.is_finite() {
                steps += 1;
                if t - last_t < dt_lo {
                    dt_lo = t - last_t;
                    binding = physics.last_time_step_limits().map(|limits| {
                        let mut bounds = vec![
                            ("advection and internal waves", limits.advection),
                            ("Coriolis", limits.coriolis),
                            ("viscosity", limits.viscosity),
                        ];
                        if physics.implicit_vertical_advection.is_none() {
                            bounds.push(("vertical advection", limits.vertical));
                        }
                        bounds
                            .into_iter()
                            .min_by(|a, b| a.1.dt.total_cmp(&b.1.dt))
                            .expect("bounds")
                    });
                }
            }
            last_t = t;
            if t < next_sample - 1e-6 {
                return;
            }
            while next_sample <= t + 1e-6 {
                next_sample += interval;
            }
            barotropic(s, &mut q);
            for station in &mut stations {
                station.sample(&q, &domain.bathymetry, clock.unix(t));
            }
            n_callbacks += 1;
            if let (Some(writer), Some(every)) = (snapshot.as_mut(), snapshot_every)
                && (n_callbacks - 1) % every == 0
                && let Err(e) = writer.write_solution_3d(t, s)
            {
                write_error.get_or_insert(e.to_string());
            }
            if (n_callbacks - 1) % output_every != 0 {
                return;
            }
            // What set the shortest step, and the vertical bound now
            let element_place = |k: usize| {
                let (mut x, mut y, mut depth) = (0.0, 0.0, 0.0);
                for i in 0..nn {
                    let [px, py] = domain.mesh.reference_to_physical(
                        ElementIndex::new(k),
                        domain.ops.nodes_r[i],
                        domain.ops.nodes_s[i],
                    );
                    x += px / nn as f64;
                    y += py / nn as f64;
                    depth += (s.eta.data[k * nn + i] - bed[k * nn + i]) / nn as f64;
                }
                (x / 1e3, y / 1e3, depth)
            };
            if steps > 0 {
                let set_by = match binding {
                    Some((name, limit)) if limit.element < ne => {
                        let (x, y, d) = element_place(limit.element);
                        format!(
                            "; the shortest set by {name} in element {} at ({x:.1}, {y:.1}) km, {d:.1} m mean depth",
                            limit.element
                        )
                    }
                    _ => String::new(),
                };
                let vertical = physics
                    .last_time_step_limits()
                    .map_or(f64::NAN, |l| l.vertical.dt);
                let implicit = if physics.implicit_vertical_advection.is_some() {
                    let stats = physics.take_implicit_advection_stats();
                    format!(
                        "; implicit vertical advection in up to {} columns, outflow Courant number ≤ {:.2}",
                        stats.columns, stats.largest_courant
                    )
                } else {
                    String::new()
                };
                println!(
                    "         steps {steps}: dt {dt_lo:.2}–{:.2} s{set_by}; |Ω|/H_z bound now {vertical:.2} s{implicit}",
                    (t - since) / steps as f64,
                );
                if dbg("dt_dist") {
                    // Every element's own advective step now: what local time
                    // stepping in 3D could save, and who sets it
                    let mut own = physics.element_advection_steps(s, opts.cfl_3d);
                    own.retain(|e| e.dt.is_finite());
                    own.sort_by(|a, b| a.dt.total_cmp(&b.dt));
                    let smallest = own.first().map_or(f64::NAN, |e| e.dt);
                    let quantile = |q: f64| {
                        let i = (own.len().saturating_sub(1) as f64 * q) as usize;
                        own.get(i).map_or(f64::NAN, |e| e.dt)
                    };
                    let gain = |levels: u32| {
                        let work: f64 = own
                            .iter()
                            .map(|e| {
                                let l = ((e.dt / smallest).log2().floor() as u32).min(levels);
                                0.5_f64.powi(l as i32)
                            })
                            .sum();
                        own.len() as f64 / work
                    };
                    let without = |depth: f64| {
                        own.iter()
                            .find(|e| e.depth >= depth)
                            .map_or(f64::NAN, |e| e.dt)
                    };
                    println!(
                        "         own steps: min {smallest:.2}, p0.1 {:.2}, p1 {:.2}, p5 {:.2}, median {:.2} s; LTS gain L=1 {:.2}, 2 {:.2}, 3 {:.2}, 4 {:.2}; min over binding depth ≥ 2 m {:.2}, ≥ 5 m {:.2}",
                        quantile(0.001),
                        quantile(0.01),
                        quantile(0.05),
                        quantile(0.5),
                        gain(1),
                        gain(2),
                        gain(3),
                        gain(4),
                        without(2.0),
                        without(5.0),
                    );
                    let limits = physics.time_step_limits(s, opts.cfl_3d);
                    println!(
                        "           limits now: advection {:.2} (element {}), viscosity {:.2} (element {}), coriolis {:.0}, vertical {:.2}; last compute_dt's advection {:.2}",
                        limits.advection.dt,
                        limits.advection.element,
                        limits.viscosity.dt,
                        limits.viscosity.element,
                        limits.coriolis.dt,
                        limits.vertical.dt,
                        physics
                            .last_time_step_limits()
                            .map_or(f64::NAN, |l| l.advection.dt),
                    );
                    for e in own.iter().take(5) {
                        println!(
                            "           dt {:.2} s: |u| {:.2} m/s, c1 {:.2} m/s, h {:.1} m, depth {:.2} m",
                            e.dt, e.speed, e.c1, e.length, e.depth
                        );
                    }
                }
            }
            binding = None;
            (steps, dt_lo, since) = (0, f64::INFINITY, t);
            let stats = domain.stats(&q);
            let (x, y, h) = stats.fastest;
            // The fastest layer of a column at least the thin depth deep
            let (mut fastest, mut level, mut column) = (0.0_f64, 0, 0);
            let (mut t_lo, mut t_hi) = (f64::INFINITY, f64::NEG_INFINITY);
            // T in the thin (shore) columns too, where a drift is held at
            // first and shows in the deeper columns only once it spreads; not
            // the dry ones (under 1 μm), which keep their last value
            let (mut thin_lo, mut thin_hi) = (f64::INFINITY, f64::NEG_INFINITY);
            for (idx, &b) in bed.iter().enumerate() {
                let depth = s.eta.data[idx] - b;
                if depth < opts.thin_depth() {
                    if depth >= 1e-6 {
                        for &temp in &s.temp[idx * nl..(idx + 1) * nl] {
                            thin_lo = thin_lo.min(temp);
                            thin_hi = thin_hi.max(temp);
                        }
                    }
                    continue;
                }
                for l in 0..nl {
                    let j = idx * nl + l;
                    let speed = s.u[j].hypot(s.v[j]);
                    if speed.is_nan() || speed > fastest {
                        (fastest, level, column) = (speed, l, idx);
                    }
                    t_lo = t_lo.min(s.temp[j]);
                    t_hi = t_hi.max(s.temp[j]);
                }
            }
            let [fx, fy] = domain.mesh.reference_to_physical(
                ElementIndex::new(column / nn),
                domain.ops.nodes_r[column % nn],
                domain.ops.nodes_s[column % nn],
            );
            let thin = if thin_lo <= thin_hi {
                format!("{thin_lo:5.2}–{thin_hi:5.2}")
            } else {
                "none".to_string()
            };
            println!(
                "{:6.2} h | [{:+.3}, {:+.3}] | {:5.2} at ({:6.1}, {:6.1}; {h:6.1}) | {fastest:5.2}, level {level:2} at ({:6.1}, {:6.1}; {:5.1} m, element {}) | {t_lo:5.2}–{t_hi:5.2}, {thin} | {:.0} s",
                t / 3600.0,
                stats.eta.0,
                stats.eta.1,
                stats.speed,
                x / 1e3,
                y / 1e3,
                fx / 1e3,
                fy / 1e3,
                s.eta.data[column] - bed[column],
                column / nn,
                start.elapsed().as_secs_f64()
            );
            if t >= next_restart - 1e-6 {
                let mut restart = run.restart(s, t);
                restart.set_metadata("command", command.as_str());
                for (j, station) in stations.iter().enumerate() {
                    restart.set_metadata(&format!("station.{j}"), station.name.as_str());
                    for (field, values) in [
                        ("times", &station.times),
                        ("eta", &station.eta),
                        ("u_east", &station.u_east),
                        ("v_north", &station.v_north),
                    ] {
                        restart.set_extra(&format!("station.{j}.{field}"), values.clone());
                    }
                }
                restart.set_extra("driver.counters", vec![n_callbacks as f64, next_sample]);
                match restart.write(&restart_path) {
                    Ok(()) => println!(
                        "         restart at {:.2} h → {}",
                        t / 3600.0,
                        restart_path.display()
                    ),
                    // A long run goes on without it
                    Err(e) => println!("         WARNING: no restart at {:.2} h: {e}", t / 3600.0),
                }
                while next_restart <= t + 1e-6 {
                    next_restart += restart_interval;
                }
            }
        };
    let integrator = ModeSplitIntegrator::new()
        .with_barotropic_cfl(opts.cfl_bt)
        .with_barotropic_scheme(opts.rk_bt)
        .with_baroclinic_scheme(opts.rk_3d)
        .with_barotropic_levels(opts.bt_levels);
    let mut sim = Simulation3D::new(physics, integrator)
        .with_cfl(opts.cfl_3d)
        .with_dt_max(opts.dt_3d.unwrap_or(MAX_DT_3D));
    // The snapshots at their times exactly, so that runs with different
    // steps compare frame by frame (`snapshot_compare_3d`)
    if let Some(every) = snapshot_every {
        sim = sim.with_landing_interval(every as f64 * interval);
    }
    if let Some(restart) = restart {
        sim.resume(&restart)?;
        state = restart.state;
    }
    let result = sim.run_with_context_callback(&mut state, t_start, t_end, callback);
    if dg_rs::profiling::enabled() {
        println!(
            "
Phase profile:
{}",
            dg_rs::profiling::report()
        );
    }
    if let Some(e) = write_error {
        return Err(e.into());
    }
    let steps = result.n_steps.max(1);
    let wall = start.elapsed().as_secs_f64();
    let run_time = result.final_time - t_start;
    println!(
        "\n{} after {:.2} h: {} steps (mean dt {:.2} s, {:.2}–{:.2} s), {:.1} s wall ({:.0} ms/step, {:.0} s per model hour), {} barotropic substeps, {} negative-depth clips",
        if result.success { "Done" } else { "FAILED" },
        result.final_time / 3600.0,
        result.n_steps,
        run_time / steps as f64,
        result.dt_min,
        result.dt_max,
        wall,
        1e3 * wall / steps as f64,
        3600.0 * wall / run_time.max(1.0),
        sim.integrator().last_substeps(),
        sim.physics().swe_physics.negative_depth_clips()
    );
    if let Some(stats) = sim.integrator().barotropic_multirate_stats() {
        println!(
            "Barotropic local time steps: up to {} levels, {:.2}× less RHS work than global substeps at the finest step ({} coarse substeps)",
            stats.finest_level,
            stats.speedup(),
            stats.steps
        );
    }
    report_stations(&stations, &output_dir, opts, &clock, station_atlas.as_ref())?;
    if let Some(e) = result.error {
        return Err(e.into());
    }
    Ok(())
}

/// Wall time of each phase of an SSP-RK3 step (`profile=N`): after 15 min of
/// spin-up (so the flow and the wet/dry state are realistic), `n` calls of
/// each. A step is 3 RHS, 3 post-processing and 3 implicit-damping calls and
/// one dt.
fn profile_phases<P: PhysicsModule<SWESolution2D>>(domain: &Domain, physics: &P, n: usize) {
    let mut q = domain.at_rest();
    let spin_up = 900.0;
    let mut t = 0.0;
    let mut stages = dg_rs::time::StageWorkspace::new();
    while t < spin_up {
        let dt = physics
            .compute_dt_ssp(&q, 1.0, Some(SspScheme::Rk3))
            .min(spin_up - t);
        dg_rs::time::TimeIntegrator::step_with_relaxation(
            &SSPRK3,
            &mut q,
            dt,
            t,
            |s, time, out| physics.compute_rhs_into(s, time, out),
            |stage, from, dt| physics.implicit_damping(stage, from, dt),
            |s| physics.post_process(s),
            &mut stages,
        );
        t += dt;
    }
    let dt = physics.compute_dt_ssp(&q, 1.0, Some(SspScheme::Rk3));

    // Median of `n` calls (after a warm-up): robust to the boost and thermal
    // swings of a laptop
    let time = |f: &mut dyn FnMut()| {
        f();
        let mut ms: Vec<f64> = (0..n)
            .map(|_| {
                let start = Instant::now();
                f();
                1e3 * start.elapsed().as_secs_f64()
            })
            .collect();
        ms.sort_by(f64::total_cmp);
        ms[n / 2]
    };
    let measure = |threads: usize| {
        let mut out = q.clone();
        let mut scratch = q.clone();
        let rhs = time(&mut || physics.compute_rhs_into(&q, t, &mut out));
        let copy = time(&mut || scratch.clone_from(&q));
        let post = time(&mut || {
            scratch.clone_from(&q);
            physics.post_process(&mut scratch);
        }) - copy;
        let damping = time(&mut || {
            scratch.clone_from(&q);
            physics.implicit_damping(&mut scratch, &q, dt);
        }) - copy;
        let dt_ms = time(&mut || {
            std::hint::black_box(physics.compute_dt_ssp(&q, 1.0, Some(SspScheme::Rk3)));
        });
        let step = 3.0 * (rhs + post + damping) + dt_ms;
        println!("\nPhases after {spin_up} s, {threads} threads: ms per call, share of a step");
        for (name, ms, calls) in [
            ("RHS", rhs, 3.0),
            ("post_process (limiter, wet/dry)", post, 3.0),
            ("implicit damping", damping, 3.0),
            ("dt", dt_ms, 1.0),
        ] {
            println!(
                "  {name:32} {ms:7.3} ms  {:5.1} %",
                100.0 * calls * ms / step
            );
        }
        println!("  step (sum, without RK combinations) {step:.2} ms");
    };

    // One thread, then all of them
    #[cfg(feature = "parallel")]
    for threads in [
        1,
        std::thread::available_parallelism().map_or(1, |n| n.get()),
    ] {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("thread pool")
            .install(|| measure(threads));
    }
    #[cfg(not(feature = "parallel"))]
    measure(1);
}

/// The cost of the spectral wave model on this domain (`waves=N`, TODO F.4):
/// SWAN's default sources (Komen, the DIA, JONSWAP friction, Battjes–Janssen)
/// under a 10 m/s wind to the east, the JONSWAP sea of `wave_sea=` (by
/// default H_s 2.5 m and T_p 10 s from the west-north-west) through the open
/// boundaries and, to start with, over the whole domain. The median wall time of N steps after a warm-up, split into the
/// propagation and the sources, the step and what sets it, and the cost per
/// model hour. With the waves on a grid of their own (`wave_mesh=`), also the
/// coupling to the run's mesh `circulation` (`WaveCoupling2D`): its build and
/// each exchange, from a circulation at rest.
fn wave_cost(domain: &Domain, circulation: &Domain, opts: &Options) {
    use dg_rs::waves::{
        StokesDriftField, WaveCoupling2D, WaveWorkspace, group_velocity, wavenumber,
    };
    let [nf, nd] = opts.wave_grid;
    let wind = Wind {
        u10: 10.0,
        direction: 0.0,
    };
    let (mut model, sea) = wave_model(domain, opts, Some(wind));
    let n_components = model.grid.n_components();
    let np = model.n_points();
    println!(
        "\nSpectral wave model: {} elements, {np} nodes (P{}), {nf} frequencies (0.04–0.5 Hz) × \
         {nd} directions = {n_components} components, {:.1} M unknowns",
        domain.mesh.n_elements,
        domain.ops.order,
        (np * n_components) as f64 / 1e6
    );

    // The step, and the part of it the geographic propagation alone allows
    let cfl = opts.wave_cfl;
    let dt = model.compute_dt(cfl);
    let sigma_min = model.grid.sigma[0];
    let nn = domain.ops.n_nodes;
    let order_factor = (2 * domain.ops.order + 1) as f64;
    let element_dt: Vec<f64> = (0..domain.mesh.n_elements)
        .map(|k| {
            let cg = (k * nn..(k + 1) * nn)
                .map(|p| {
                    let d = model.depth()[p];
                    group_velocity(sigma_min, wavenumber(sigma_min, d, G), d)
                })
                .fold(0.0, f64::max);
            cfl * domain.geom.element_size(k) / (order_factor * cg)
        })
        .collect();
    let dt_geographic = element_dt.iter().cloned().fold(f64::INFINITY, f64::min);
    // Every element at its own step against all at the smallest: the most
    // local time stepping could save (power-of-two levels save a little less)
    let multirate =
        element_dt.len() as f64 / element_dt.iter().map(|dt| dt_geographic / dt).sum::<f64>();
    println!(
        "  elements' own geographic steps {dt_geographic:.3}–{:.1} s: local time stepping could \
         save up to {multirate:.1}×",
        element_dt.iter().cloned().fold(0.0, f64::max)
    );
    println!(
        "  step {dt:.3} s at CFL {cfl}; the geographic propagation alone would allow {dt_geographic:.3} s{}",
        if dt < 0.99 * dt_geographic {
            " (refraction or frequency shifting sets it)"
        } else {
            ""
        }
    );

    let mut n = model.uniform_state(&sea);
    let mut ws = WaveWorkspace::default();
    let mut t = 0.0;
    let median = |times: &mut Vec<f64>| {
        times.sort_by(f64::total_cmp);
        times[times.len() / 2]
    };
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    // Warm-up, then the steps timed
    for _ in 0..2 {
        model.step(&mut n, dt, &mut ws);
        t += dt;
    }
    let mut steps = Vec::with_capacity(opts.waves);
    let mut sources = Vec::with_capacity(opts.waves);
    for _ in 0..opts.waves {
        let start = Instant::now();
        model.step(&mut n, dt, &mut ws);
        steps.push(start.elapsed().as_secs_f64());
        t += dt;
        let mut scratch = n.clone();
        let start = Instant::now();
        model.apply_sources(&mut scratch, dt);
        sources.push(start.elapsed().as_secs_f64());
    }
    // The parts of a step that run node by node, each alone (with its own
    // transposes to node-major and back)
    let timed = |f: &mut dyn FnMut(&mut dg_rs::waves::WaveSolution)| {
        let mut times = Vec::with_capacity(opts.waves);
        for _ in 0..opts.waves {
            let mut scratch = n.clone();
            let start = Instant::now();
            f(&mut scratch);
            times.push(start.elapsed().as_secs_f64());
        }
        1e3 * median(&mut times)
    };
    let refraction = timed(&mut |s| model.apply_implicit_refraction(s, 0.5 * dt));
    let shift = timed(&mut |s| model.apply_implicit_frequency_shift(s, 0.5 * dt));
    let with_dia = model.sources.clone();
    model.sources.quadruplets = None;
    let without_dia = timed(&mut |s| model.apply_sources(s, dt));
    model.sources = with_dia;
    let (step, source) = (median(&mut steps), median(&mut sources));
    let per_hour = 3600.0 / dt * step;
    println!(
        "  {:.1} ms per step on {threads} threads: propagation {:.1} ms ({:.0} %), sources {:.1} ms \
         ({:.0} %)",
        1e3 * step,
        1e3 * (step - source),
        100.0 * (step - source) / step,
        1e3 * source,
        100.0 * source / step
    );
    println!(
        "  {:.0} ns per component and node per step ({:.1} ns of it propagation, three RK stages)",
        1e9 * step / (np * n_components) as f64,
        1e9 * (step - source) / (np * n_components) as f64,
    );
    println!(
        "  alone: the sources {:.1} ms ({:.1} ms of it the DIA); a half-step \
         of implicit refraction {refraction:.1} ms, of frequency shifting {shift:.1} ms",
        1e3 * source,
        1e3 * source - without_dia,
    );
    println!(
        "  {per_hour:.0} s of wall time per model hour ({:.2}× real time)",
        3600.0 / per_hour
    );
    // On to `wave_hours`, the last step landing on it
    let end = 3600.0 * opts.wave_hours;
    if end > t {
        let start = Instant::now();
        while t < end {
            let h = dt.min(end - t);
            model.step(&mut n, h, &mut ws);
            t = if end - t <= dt { end } else { t + dt };
        }
        println!(
            "  on to {:.2} h in {:.0} s of wall time",
            opts.wave_hours,
            start.elapsed().as_secs_f64()
        );
    }
    let params = model.parameters(&n);
    let hs_max = params.iter().map(|p| p.hs).fold(0.0, f64::max);
    println!(
        "  after {:.0} s: H_s ≤ {hs_max:.2} m, total action {:.3e}",
        t,
        model.total_action(&n)
    );
    let output_dir = opts
        .output
        .clone()
        .unwrap_or_else(|| Path::new("output").join(circulation.name));
    report_depth_limit(&model, &params);
    if let Err(e) = write_wave_nodes(&model, &params, t, &output_dir) {
        eprintln!("  could not write the waves at the nodes: {e}");
    }
    if let Some(path) = &opts.wave_reference
        && let Err(e) = compare_wave_nodes(&model, &params, path)
    {
        eprintln!("  could not compare with {}: {e}", path.display());
    }
    if Arc::ptr_eq(&domain.mesh, &circulation.mesh) {
        return;
    }

    // The coupling to the run's mesh: the circulation at rest
    let start = Instant::now();
    let coupling = WaveCoupling2D::new(
        &model,
        circulation.mesh.clone(),
        circulation.ops.clone(),
        circulation.geom.clone(),
    );
    let build = start.elapsed().as_secs_f64();
    let (to_waves, to_circulation) = (coupling.to_waves(), coupling.to_circulation());
    println!(
        "\nCoupling to the run's mesh ({} elements, P{}, {} nodes): built in {:.0} ms",
        circulation.mesh.n_elements,
        circulation.ops.order,
        to_circulation.n_target_points(),
        1e3 * build
    );
    println!(
        "  wave nodes outside the run's mesh: {} of {} (up to {:.0} m from it); run's nodes \
         outside the waves': {} of {} (up to {:.0} m)",
        to_waves.n_outside(),
        to_waves.n_target_points(),
        to_waves.largest_gap(),
        to_circulation.n_outside(),
        to_circulation.n_target_points(),
        to_circulation.largest_gap()
    );
    let nn = circulation.ops.n_nodes;
    let mut q = SWESolution2D::new(circulation.mesh.n_elements, nn);
    for k in ElementIndex::iter(circulation.mesh.n_elements) {
        for i in 0..nn {
            let h = (-circulation.bathymetry.get(k, i)).max(0.0);
            q.set_state(k, i, SWEState2D::new(h, 0.0, 0.0));
        }
    }
    let time = |f: &mut dyn FnMut()| {
        let mut times: Vec<f64> = (0..5)
            .map(|_| {
                let start = Instant::now();
                f();
                start.elapsed().as_secs_f64()
            })
            .collect();
        median(&mut times)
    };
    let to_model = time(&mut || coupling.update_waves(&mut model, &q, &circulation.bathymetry));
    let force = time(&mut || {
        std::hint::black_box(coupling.force(&model, &n));
    });
    let bed = time(&mut || {
        std::hint::black_box(coupling.bed_wave_stress(&model, &n, 1e-3));
    });
    let roughness = time(&mut || {
        std::hint::black_box(coupling.surface_roughness(&model, &n, 0.6));
    });
    let stokes = time(&mut || {
        std::hint::black_box(coupling.stokes_drift(&model, &n));
    });
    let stokes_own = time(&mut || {
        std::hint::black_box(StokesDriftField::new(&model, &n));
    });
    let exchange = to_model + force + bed + roughness + stokes;
    println!(
        "  one exchange {:.0} ms ({:.1} wave steps): level and currents to the waves {:.0} ms, \
         force {:.0} ms, bed stress {:.0} ms, roughness {:.0} ms, Stokes drift {:.0} ms (of it \
         {:.0} ms on the waves' own mesh)",
        1e3 * exchange,
        exchange / step,
        1e3 * to_model,
        1e3 * force,
        1e3 * bed,
        1e3 * roughness,
        1e3 * stokes,
        1e3 * stokes_own
    );
}

/// Each wave node's position (m), depth (m) and sea at time `t` (s), to
/// `<output_dir>/wave_nodes.txt`: the reference for a later run's
/// `wave_reference=`.
fn write_wave_nodes(
    model: &WaveModel2D,
    params: &[dg_rs::waves::WaveParameters],
    t: f64,
    output_dir: &Path,
) -> std::io::Result<()> {
    use std::fmt::Write as _;
    fs::create_dir_all(output_dir)?;
    let path = output_dir.join("wave_nodes.txt");
    let mut text = format!(
        "# The waves at every node after {t:.1} s\n# x (m), y (m), depth (m), H_s (m), T_m01 (s), \
         mean direction (deg, travelling to, counter-clockwise from east), spread (deg)\n"
    );
    for (p, (xy, w)) in wave_node_positions(model).zip(params).enumerate() {
        let _ = writeln!(
            text,
            "{:.3} {:.3} {:.4} {:.6} {:.5} {:.4} {:.4}",
            xy[0],
            xy[1],
            model.depth()[p],
            w.hs,
            w.tm01,
            w.direction.to_degrees(),
            w.spread.to_degrees()
        );
    }
    fs::write(&path, text)?;
    println!("  the waves at the nodes → {}", path.display());
    Ok(())
}

/// The wave nodes' positions (m), in the model's node order.
fn wave_node_positions(model: &WaveModel2D) -> impl Iterator<Item = [f64; 2]> + '_ {
    let (mesh, ops) = (&model.mesh, &model.ops);
    (0..mesh.n_elements).flat_map(move |k| {
        (0..ops.n_nodes).map(move |i| {
            mesh.reference_to_physical(ElementIndex::new(k), ops.nodes_r[i], ops.nodes_s[i])
        })
    })
}

/// This run's sea against an earlier run's `wave_nodes.txt` on the same
/// nodes (`wave_reference=`): H_s node by node (bias, RMS, the share of
/// nodes off by more than 2, 5 and 10 %, the largest difference and where,
/// by depth), T_m01 and the mean direction.
fn compare_wave_nodes(
    model: &WaveModel2D,
    params: &[dg_rs::waves::WaveParameters],
    path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    // x, y, depth, H_s, T_m01, direction (deg)
    let reference: Vec<[f64; 6]> = fs::read_to_string(path)?
        .lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .map(|line| {
            let v: Vec<f64> = line
                .split_whitespace()
                .map(str::parse)
                .collect::<Result<_, _>>()?;
            Ok::<_, std::num::ParseFloatError>([v[0], v[1], v[2], v[3], v[4], v[5]])
        })
        .collect::<Result<_, _>>()?;
    if reference.len() != params.len() {
        return Err(format!("{} nodes there, {} here", reference.len(), params.len()).into());
    }
    for (xy, r) in wave_node_positions(model).zip(&reference) {
        if (xy[0] - r[0]).abs() > 0.01 || (xy[1] - r[1]).abs() > 0.01 {
            return Err(format!("node at ({:.0}, {:.0}) m differs", r[0], r[1]).into());
        }
    }
    // The nodes with a sea worth comparing
    const HS_MIN: f64 = 0.1;
    let nodes: Vec<usize> = (0..params.len())
        .filter(|&p| reference[p][3] >= HS_MIN)
        .collect();
    let max_at = |hs: &dyn Fn(usize) -> f64| {
        (0..params.len())
            .max_by(|&a, &b| hs(a).total_cmp(&hs(b)))
            .unwrap()
    };
    let (here, there) = (max_at(&|p| params[p].hs), max_at(&|p| reference[p][3]));
    println!(
        "\nAgainst {} ({} of {} nodes with H_s ≥ {HS_MIN} m there):",
        path.display(),
        nodes.len(),
        params.len()
    );
    for (name, p) in [("here", here), ("there", there)] {
        println!(
            "  largest H_s {name}: {:.3} m here, {:.3} m there, at ({:.1}, {:.1}) km, {:.1} m deep",
            params[p].hs,
            reference[p][3],
            reference[p][0] / 1e3,
            reference[p][1] / 1e3,
            reference[p][2]
        );
    }
    let summary = |label: &str, nodes: &[usize]| {
        if nodes.is_empty() {
            return;
        }
        let n = nodes.len() as f64;
        let diff = |p: usize| params[p].hs - reference[p][3];
        let relative = |p: usize| diff(p) / reference[p][3];
        let bias = nodes.iter().map(|&p| diff(p)).sum::<f64>() / n;
        let rms = (nodes.iter().map(|&p| diff(p).powi(2)).sum::<f64>() / n).sqrt();
        let rms_relative = (nodes.iter().map(|&p| relative(p).powi(2)).sum::<f64>() / n).sqrt();
        let share = |bound: f64| {
            100.0 * nodes.iter().filter(|&&p| relative(p).abs() > bound).count() as f64 / n
        };
        println!(
            "  {label:>12} {:>6} nodes: ΔH_s bias {bias:+.4} m, RMS {rms:.4} m ({:.2} %), off by \
             > 2 / 5 / 10 %: {:.2} / {:.2} / {:.2} % of the nodes",
            nodes.len(),
            100.0 * rms_relative,
            share(0.02),
            share(0.05),
            share(0.10)
        );
    };
    summary("all", &nodes);
    for (low, high) in [
        (0.0, 1.0),
        (1.0, 3.0),
        (3.0, 10.0),
        (10.0, 30.0),
        (30.0, 100.0),
        (100.0, f64::INFINITY),
    ] {
        let band: Vec<usize> = nodes
            .iter()
            .copied()
            .filter(|&p| (low..high).contains(&reference[p][2]))
            .collect();
        summary(&format!("{low:.0}–{high:.0} m"), &band);
    }
    let worst = |key: &dyn Fn(usize) -> f64| {
        nodes
            .iter()
            .copied()
            .max_by(|&a, &b| key(a).total_cmp(&key(b)))
    };
    let largest = [
        ("in m", worst(&|p| (params[p].hs - reference[p][3]).abs())),
        (
            "relative",
            worst(&|p| ((params[p].hs - reference[p][3]) / reference[p][3]).abs()),
        ),
    ];
    for (name, p) in largest {
        let Some(p) = p else { continue };
        println!(
            "  largest difference {name}: {:.3} m here against {:.3} m there ({:+.1} %), at \
             ({:.1}, {:.1}) km, {:.1} m deep",
            params[p].hs,
            reference[p][3],
            100.0 * (params[p].hs / reference[p][3] - 1.0),
            reference[p][0] / 1e3,
            reference[p][1] / 1e3,
            reference[p][2]
        );
    }
    // Periods and directions where the sea is more than a ripple
    let seas: Vec<usize> = nodes
        .iter()
        .copied()
        .filter(|&p| reference[p][3] >= 0.5)
        .collect();
    if !seas.is_empty() {
        let n = seas.len() as f64;
        let period = (seas
            .iter()
            .map(|&p| (params[p].tm01 / reference[p][4] - 1.0).powi(2))
            .sum::<f64>()
            / n)
            .sqrt();
        let turn = |p: usize| {
            let d = params[p].direction.to_degrees() - reference[p][5];
            (d + 180.0).rem_euclid(360.0) - 180.0
        };
        let direction = (seas.iter().map(|&p| turn(p).powi(2)).sum::<f64>() / n).sqrt();
        let largest_turn = seas.iter().map(|&p| turn(p).abs()).fold(0.0, f64::max);
        println!(
            "  where H_s ≥ 0.5 m ({} nodes): T_m01 RMS {:.2} %, mean direction RMS {direction:.2}° \
             (largest {largest_turn:.1}°)",
            seas.len(),
            100.0 * period
        );
    }
    Ok(())
}

/// The cost of the 3D model on this domain (`levels=N`, TODO P1.3): the
/// mode-split step with GLS k-ε, log-layer drag, Smagorinsky viscosity of the
/// shear and the Kuzmin limiter, over a summer pycnocline at rest (T 4 °C
/// warmer above ≈ 15 m, S 1.5 psu fresher), with the run's 2D module (walls
/// everywhere: the open boundaries cost nothing next to the interior). The
/// median wall time of `steps_3d` steps after a warm-up; the baroclinic step
/// at rest and with 1 m/s currents everywhere, each element's own step (what
/// local time stepping could save), and the barotropic substeps per step.
fn cost_3d(domain: &Domain, opts: &Options) {
    use dg_rs::physics::{
        BottomDrag3D, ConstantMixing, Forcing, Hydrostatic3D, LinearEOS, VerticalMixing,
    };
    use dg_rs::solver::state::Solution3D;
    use dg_rs::solver::{TracerLimiter3DConfig, TracerLimiterType3D, TracerReferenceProfile};
    use dg_rs::time::ModeSplitIntegrator;
    use dg_rs::vertical::{SigmaGrid, SongHaidvogelStretching};

    const RHO0: f64 = 1025.0;
    // `debug_3d=around=K:R`: only the elements within R m of element K
    // (centres), walls around, to reproduce a local growth quickly
    let around = opts.debug_3d.split(',').find_map(|f| {
        let (k, r) = f.strip_prefix("around=")?.split_once(':')?;
        Some((k.parse::<usize>().ok()?, r.parse::<f64>().ok()?))
    });
    let patch;
    let domain = match around {
        None => domain,
        Some((k, radius)) => {
            let centre = |e: ElementIndex| domain.mesh.reference_to_physical(e, 0.0, 0.0);
            let [x0, y0] = centre(ElementIndex::new(k));
            let (mesh, kept) = domain.mesh.retain_elements(
                |e| {
                    let [x, y] = centre(e);
                    (x - x0).hypot(y - y0) <= radius
                },
                BoundaryTag::Wall,
            );
            let geom = GeometricFactors2D::compute(&mesh, &domain.ops);
            let bathymetry = domain.bathymetry.select_elements(&kept);
            println!(
                "  Patch of {} elements within {radius} m of element {k} (now element {})",
                kept.len(),
                kept.iter()
                    .position(|&e| e == k)
                    .expect("the element is in its patch")
            );
            // `debug_3d=export=PATH`: the patch as a text fixture for tests
            if let Some(path) = opts
                .debug_3d
                .split(',')
                .find_map(|f| f.strip_prefix("export="))
            {
                write_patch_fixture(path, &mesh, &bathymetry, k, radius);
            }
            patch = Domain {
                name: domain.name,
                mesh: Arc::new(mesh),
                ops: domain.ops.clone(),
                geom: Arc::new(geom),
                bathymetry: Arc::new(bathymetry),
                projection: domain.projection,
                manning: domain.manning,
            };
            &patch
        }
    };
    let nl = opts.levels;
    let sigma = Arc::new(SigmaGrid::new(
        nl,
        SongHaidvogelStretching::new(5.0, 0.4, 10.0),
    ));
    // The 2D module of the run, without its own friction: the 3D bottom
    // drag's depth mean replaces it
    let swe = PhysicsBuilder::swe_2d(
        domain.mesh.clone(),
        domain.ops.clone(),
        domain.geom.clone(),
        ShallowWater2D::new(G),
        Reflective2D::new(),
    )
    .with_bathymetry(domain.bathymetry.clone())
    .with_limiter(StandardLimiter2D::Positivity(WetDryConfig::DEFAULT_H_DRY))
    .with_wet_dry(WetDryConfig::default())
    .with_source(CoriolisSource2D::f_plane(F_CORIOLIS))
    .build();
    let eos = LinearEOS::default();
    let dbg = |flag: &str| opts.debug_3d.split(',').any(|f| f == flag);
    // `debug_3d=key=value`
    let dbg_value = |key: &str| {
        opts.debug_3d
            .split(',')
            .find_map(|f| f.strip_prefix(key)?.strip_prefix('='))
            .map(str::to_string)
    };
    // The temperature gradient below the pycnocline (`deep=`, °C/m)
    let deep_gradient: f64 = dbg_value("deep").map_or(0.002, |v| v.parse().expect("deep="));
    // The summer stratification, horizontally uniform: T 4 °C warmer and S
    // 1.5 psu fresher above ≈ 15 m (`linear`: 0.02 °C/m; `uniform`: none)
    let profile = |z: f64| -> (f64, f64) {
        if dbg("uniform") {
            (eos.t0, eos.s0)
        } else if dbg("linear") {
            (eos.t0 + 0.02 * z, eos.s0)
        } else {
            let step = 0.5 * (1.0 + ((z + 15.0) / 4.0).tanh());
            (
                eos.t0 + 4.0 * step + deep_gradient * z.min(0.0),
                eos.s0 - 1.5 * step,
            )
        }
    };
    let physics: Physics3D = Hydrostatic3D::new(
        domain.mesh.clone(),
        domain.ops.clone(),
        domain.geom.clone(),
        sigma.clone(),
        domain.bathymetry.clone(),
        Arc::new(CoriolisSource2D::f_plane(F_CORIOLIS)),
        eos,
        if opts.debug_3d.split(',').any(|f| f == "constant") {
            Box::new(ConstantMixing::new(1e-3, 0.0)) as Box<dyn VerticalMixing + Send + Sync>
        } else {
            Box::new((opts.mixing.1)())
        },
        swe,
        Forcing {
            surface_stress: [0.0, 0.0],
            bottom_stress: [0.0, 0.0],
            surface_heat_flux: 0.0,
        },
        G,
        RHO0,
    )
    .with_bottom_drag(BottomDrag3D::log_layer(0.003))
    .with_min_column_depth(opts.thin_depth())
    .with_smagorinsky_viscosity(0.1)
    // `tadv=none|centred`: the turbulence's advection (LimitedAkima by default)
    .with_turbulence_advection(match dbg_value("tadv").as_deref() {
        Some("none") => None,
        Some("centred") => Some(dg_rs::solver::rhs::VerticalAdvection::Centred),
        Some(other) => panic!("tadv={other}"),
        None => Some(dg_rs::solver::rhs::VerticalAdvection::LimitedAkima),
    })
    .with_horizontal_viscosity(
        opts.debug_3d
            .split(',')
            .find_map(|f| f.strip_prefix("nu=").and_then(|v| v.parse().ok()))
            .unwrap_or(0.0),
    )
    .with_vertical_advection({
        use dg_rs::solver::rhs::VerticalAdvection;
        match dbg_value("vadv").as_deref() {
            _ if dbg("vcentred") => VerticalAdvection::Centred,
            Some("centred") => VerticalAdvection::Centred,
            Some("akima") => VerticalAdvection::Akima,
            Some("tvd") => VerticalAdvection::Tvd,
            Some("upwind") => VerticalAdvection::Upwind,
            Some("hermite") => VerticalAdvection::HermiteMean,
            Some(other) => panic!("vadv={other}"),
            None => VerticalAdvection::default(),
        }
    })
    .with_tracer_limiter(if dbg("nolimiter") {
        TracerLimiter3DConfig::none()
    } else {
        // The stratification as the limiter's reference: without it the
        // limiter clips a curved profile where σ-levels cross it steeply
        let config = TracerLimiter3DConfig {
            limiter_type: TracerLimiterType3D::HorizontalKuzmin { relaxation: 1.0 },
            ..TracerLimiter3DConfig::default()
        };
        let deepest = domain.bathymetry.data.iter().copied().fold(0.0, f64::min);
        config.with_reference_profile(TracerReferenceProfile::from_fn(
            deepest - 1.0,
            1.0,
            4001,
            profile,
        ))
    });

    let (ne, nn) = (domain.mesh.n_elements, domain.ops.n_nodes);
    let mut state = Solution3D::new(ne, nn, nl);
    for idx in 0..ne * nn {
        let bed = domain.bathymetry.data[idx];
        let eta = bed.max(0.0);
        state.eta.data[idx] = eta;
        for (l, &s) in sigma.sigma_rho().iter().enumerate() {
            let z = eta + s * (eta - bed);
            (state.temp[idx * nl + l], state.salt[idx * nl + l]) = profile(z);
        }
    }
    physics.update_density(&mut state);
    let wet = (0..ne * nn)
        .filter(|&idx| state.eta.data[idx] - domain.bathymetry.data[idx] > 0.0)
        .count();
    println!(
        "\n3D model: {ne} elements × {nn} nodes × {nl} levels = {:.2} M points ({wet} wet columns), P{}",
        (ne * nn * nl) as f64 / 1e6,
        domain.ops.order
    );

    // Each element's own step from horizontal advection and internal waves
    // (the bound of `Hydrostatic3D::compute_dt` that sets it here), at rest
    // and with `speed` everywhere
    let cfl = opts.cfl_3d;
    let order_factor = (domain.ops.order as f64 + 1.0).powi(2);
    let element_dt = |speed: f64| -> Vec<f64> {
        (0..ne)
            .map(|k| {
                let mut rate = 0.0_f64;
                for i in 0..nn {
                    let idx = k * nn + i;
                    let depth = state.eta.data[idx] - domain.bathymetry.data[idx];
                    if depth < physics.min_column_depth {
                        continue;
                    }
                    let rho = &state.rho[idx * nl..(idx + 1) * nl];
                    let mut integral = 0.0;
                    for l in 1..nl {
                        let dz = (sigma.sigma_rho()[l] - sigma.sigma_rho()[l - 1]) * depth;
                        integral += (-G / RHO0 * (rho[l] - rho[l - 1]) * dz).max(0.0).sqrt();
                    }
                    let c1 =
                        (integral / std::f64::consts::PI).max(Physics3D::MIN_INTERNAL_WAVE_SPEED);
                    let ((rx, ry), (sx, sy)) = (domain.geom.grad_r(k, i), domain.geom.grad_s(k, i));
                    rate = rate.max((speed + c1) * 0.5 * (rx.hypot(ry) + sx.hypot(sy)));
                }
                if rate > 0.0 {
                    cfl / rate / order_factor
                } else {
                    f64::INFINITY
                }
            })
            .collect()
    };
    let report = |label: &str, dts: &[f64]| -> f64 {
        let mut sorted: Vec<f64> = dts.iter().copied().filter(|d| d.is_finite()).collect();
        sorted.sort_by(f64::total_cmp);
        let smallest = sorted[0];
        // Every element at its own step against all at the smallest
        let multirate = sorted.len() as f64 / sorted.iter().map(|d| smallest / d).sum::<f64>();
        let below = |factor: f64| sorted.iter().filter(|&&d| d < factor * smallest).count();
        println!(
            "  {label}: step {smallest:.2} s (median element {:.1} s; {} elements within 2× of \
             the smallest, {} within 4×); local time stepping could save up to {multirate:.1}×",
            sorted[sorted.len() / 2],
            below(2.0),
            below(4.0)
        );
        smallest
    };
    let dt_rest = physics.compute_dt(&state, cfl);
    let own_rest = report("at rest", &element_dt(0.0));
    let own_flow = report("1 m/s currents", &element_dt(1.0));
    println!(
        "  Hydrostatic3D::compute_dt at rest: {dt_rest:.2} s (the element bound alone: \
         {own_rest:.2} s)"
    );

    // One 2D RHS, for the share of the barotropic pass
    let q2d = domain.at_rest();
    let swe_rhs = {
        let mut times = Vec::new();
        for _ in 0..5 {
            let start = Instant::now();
            let _ = physics.swe_physics.compute_rhs(&q2d, 0.0);
            times.push(start.elapsed().as_secs_f64());
        }
        times.sort_by(f64::total_cmp);
        times[2]
    };

    // The stratification's profile as the PGF's reference, exact at rest
    // (`balanced`: the state as the balanced reference, which leaves the
    // constant-depth form's error at rest, 1.7e-4 m/s² at cliff shores;
    // `unbalanced`: none, the σ-pairs' error, 2.2e-2 m/s²)
    let physics = if dbg("unbalanced") {
        physics
    } else if dbg("balanced") {
        physics.with_balanced_reference(&state)
    } else {
        use dg_rs::physics::EquationOfState;
        physics.with_reference_profile(&state, |z| {
            let (t, s) = profile(z);
            eos.compute_density(t, s, z)
        })
    };
    // The vertical tracer advection about the state at rest, as
    // `Simulation3D` does (`noref`: the plain scheme)
    let physics = if dbg("noref") {
        physics
    } else {
        physics.with_vertical_reference(&state)
    };
    let mut integrator = ModeSplitIntegrator::new().with_baroclinic_scheme(opts.rk_3d);
    let ssp = opts.rk_3d.ssp_coefficient();
    let mut t = 0.0;
    // The step of `dt_3d=`, or as `Simulation3D` takes it, every step
    let mut timed_step = |state: &mut Solution3D, t: &mut f64| -> f64 {
        let dt = opts
            .dt_3d
            .unwrap_or_else(|| physics.compute_dt_ssp(state, cfl, ssp));
        physics.update_density(state);
        integrator.step(state, &physics, dt, *t);
        physics.post_process(state);
        *t += dt;
        dt
    };
    let init = state.clone();
    let dump = dbg_value("dump");
    if let Some(prefix) = &dump {
        write_mode_geometry(prefix, domain, &sigma, &state);
        write_mode_dump(&format!("{prefix}.init.bin"), &state);
    }
    for _ in 0..2 {
        timed_step(&mut state, &mut t);
    }
    // `seed=PREFIX:AMP`: add the perturbation PREFIX.end.bin − PREFIX.init.bin
    // of an earlier run, scaled to a largest speed of AMP m/s
    if let Some(seed) = dbg_value("seed") {
        let (prefix, amp) = seed.rsplit_once(':').expect("seed=PREFIX:AMP");
        let amp: f64 = amp.parse().expect("seed amplitude");
        let a = read_mode_dump(&format!("{prefix}.init.bin"), &state);
        let b = read_mode_dump(&format!("{prefix}.end.bin"), &state);
        let speed = (0..state.u.len())
            .map(|i| (b.u[i] - a.u[i]).hypot(b.v[i] - a.v[i]))
            .fold(0.0, f64::max);
        let scale = amp / speed;
        let add = |x: &mut [f64], b: &[f64], a: &[f64]| {
            for ((x, b), a) in x.iter_mut().zip(b).zip(a) {
                *x += scale * (b - a);
            }
        };
        add(&mut state.u, &b.u, &a.u);
        add(&mut state.v, &b.v, &a.v);
        add(&mut state.temp, &b.temp, &a.temp);
        add(&mut state.salt, &b.salt, &a.salt);
        add(&mut state.eta.data, &b.eta.data, &a.eta.data);
        add(&mut state.ubar.data, &b.ubar.data, &a.ubar.data);
        add(&mut state.vbar.data, &b.vbar.data, &a.vbar.data);
        physics.update_density(&mut state);
        println!("  Seeded with {prefix}'s perturbation at {amp:.1e} m/s (scale {scale:.3e})");
    }
    // `perturb=A`: a deterministic random temperature perturbation of up to
    // ±A/2 °C at every point of a column at least 0.1 m deep, as the terrace
    // fixture's gate seeds it, so slow modes emerge within hours rather than
    // from round-off
    if let Some(amp) = dbg_value("perturb") {
        let amp: f64 = amp.parse().expect("perturb amplitude");
        let mut seed = 0x2545_f491_4f6c_dd1d_u64;
        for (idx, t) in state.temp.iter_mut().enumerate() {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let column = idx / nl;
            if state.eta.data[column] - domain.bathymetry.data[column] > 0.1 {
                *t += amp * ((seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5);
            }
        }
        physics.update_density(&mut state);
        println!("  Perturbed T by up to ±{:.1e} °C", 0.5 * amp);
    }
    if dbg("trace") {
        let mut rhs = state.clone();
        physics.compute_momentum_rhs_into(&state, 0.0, &mut rhs);
        let (mut best, mut at) = (0.0_f64, 0);
        for (i, (u, v)) in rhs.u.iter().zip(&rhs.v).enumerate() {
            if u.hypot(*v) > best {
                best = u.hypot(*v);
                at = i;
            }
        }
        let k = at / nl / nn;
        println!(
            "  PGF + Coriolis + viscosity of the state: largest {best:.2e} m/s² at element {k} level {}; depths {:?}",
            at % nl,
            (0..nn)
                .map(
                    |i| ((state.eta.data[k * nn + i] - domain.bathymetry.data[k * nn + i]) * 10.0)
                        .round()
                        / 10.0
                )
                .collect::<Vec<_>>()
        );
        let every =
            dbg_value("every").map_or((opts.steps_3d / 24).max(1), |v| v.parse().expect("every="));
        let gap: usize = dbg_value("gap").map_or(opts.steps_3d / 8, |v| v.parse().expect("gap="));
        for n in 0..opts.steps_3d {
            let dt = timed_step(&mut state, &mut t);
            if let Some(prefix) = &dump
                && n + 1 + gap == opts.steps_3d
            {
                write_mode_dump(&format!("{prefix}.prev.bin"), &state);
            }
            if n % every != 0 && n + 1 != opts.steps_3d {
                continue;
            }
            // The perturbation about the initial state: kinetic energy
            // ½ Σ m H_z |u|² (per unit density), and its tracers and η
            let (mut ke, mut d_t, mut d_s, mut d_eta) = (0.0, 0.0_f64, 0.0_f64, 0.0_f64);
            for k in 0..ne {
                for i in 0..nn {
                    let idx = k * nn + i;
                    let depth = state.eta.data[idx] - domain.bathymetry.data[idx];
                    d_eta = d_eta.max((state.eta.data[idx] - init.eta.data[idx]).abs());
                    if depth <= 0.0 {
                        continue;
                    }
                    let m = domain.geom.node_mass(k, i);
                    for l in 0..nl {
                        let j = idx * nl + l;
                        let hz = sigma.d_sigma()[l] * depth;
                        ke += 0.5 * m * hz * (state.u[j].powi(2) + state.v[j].powi(2));
                        d_t = d_t.max((state.temp[j] - init.temp[j]).abs());
                        d_s = d_s.max((state.salt[j] - init.salt[j]).abs());
                    }
                }
            }
            println!(
                "  pert t {:.4} h: KE {ke:.4e}; max |dT| {d_t:.3e} |dS| {d_s:.3e} |deta| {d_eta:.3e}",
                t / 3600.0
            );
            let (mut best, mut at) = (0.0_f64, 0);
            for (i, (u, v)) in state.u.iter().zip(&state.v).enumerate() {
                let s = u.hypot(*v);
                if s.is_nan() || s > best {
                    best = s;
                    at = i;
                }
            }
            let column = at / nl;
            let (k, node) = (column / nn, column % nn);
            let depths: Vec<f64> = (0..nn)
                .map(|i| state.eta.data[k * nn + i] - domain.bathymetry.data[k * nn + i])
                .collect();
            let [x, y] = domain.mesh.reference_to_physical(
                ElementIndex::new(k),
                domain.ops.nodes_r[node],
                domain.ops.nodes_s[node],
            );
            println!(
                "  step {n} (t {:.2} h, dt {dt:.2} s): largest {best:.2e} m/s at element {k} node \
                 {node} level {} ({:.1}, {:.1}) km; depths {:?}; affine {}",
                t / 3600.0,
                at % nl,
                x / 1e3,
                y / 1e3,
                depths
                    .iter()
                    .map(|d| (d * 10.0).round() / 10.0)
                    .collect::<Vec<_>>(),
                domain.geom.element_is_affine(k)
            );
        }
        // The ten fastest elements at the end: how far a growing mode reaches
        let mut elements: Vec<(f64, usize, usize, usize)> = (0..state.n_elements)
            .map(|k| {
                (0..nn * nl)
                    .map(|j| {
                        let i = k * nn * nl + j;
                        (state.u[i].hypot(state.v[i]), k, j / nl, j % nl)
                    })
                    .fold((0.0, k, 0, 0), |a, b| if b.0 > a.0 { b } else { a })
            })
            .collect();
        elements.sort_by(|a, b| b.0.total_cmp(&a.0));
        if let Some(prefix) = &dump {
            write_mode_dump(&format!("{prefix}.end.bin"), &state);
        }
        println!("  Fastest elements at the end:");
        for &(speed, k, node, level) in elements.iter().take(10) {
            let depths: Vec<f64> = (0..nn)
                .map(|i| {
                    let d = state.eta.data[k * nn + i] - domain.bathymetry.data[k * nn + i];
                    (d * 10.0).round() / 10.0
                })
                .collect();
            println!(
                "    {speed:.2e} m/s element {k} node {node} level {level}; depths {depths:?}"
            );
        }
        return;
    }
    let mut times = Vec::with_capacity(opts.steps_3d);
    let mut dt = dt_rest;
    for _ in 0..opts.steps_3d {
        let start = Instant::now();
        dt = timed_step(&mut state, &mut t);
        times.push(start.elapsed().as_secs_f64());
    }
    times.sort_by(f64::total_cmp);
    let per_step = times[times.len() / 2];
    let n_bt = integrator.last_substeps();
    // The pass runs ≈ 1.3 n_bt SSP-RK3 substeps of three 2D RHS each
    let barotropic = 1.3 * n_bt as f64 * 3.0 * swe_rhs;
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    println!(
        "  {:.0} ms per step on {threads} threads at {dt:.2} s: {n_bt} barotropic substeps \
         (≈ {:.0} ms, {:.0} %, at {:.2} ms per 2D RHS); the 3D stages and the rest ≈ {:.0} ms",
        1e3 * per_step,
        1e3 * barotropic,
        100.0 * barotropic / per_step,
        1e3 * swe_rhs,
        1e3 * (per_step - barotropic)
    );
    for (label, dt) in [("at rest", dt), ("1 m/s currents", own_flow.min(dt))] {
        let per_hour = 3600.0 / dt * per_step;
        println!(
            "  {label}: {per_hour:.0} s of wall time per model hour ({:.2}× real time); \
             15 days ≈ {:.0} h",
            3600.0 / per_hour,
            per_hour * 360.0 / 3600.0
        );
    }
    let speed = state
        .u
        .iter()
        .zip(&state.v)
        .map(|(u, v)| u.hypot(*v))
        .fold(0.0, |m: f64, s| if s.is_nan() || s > m { s } else { m });
    println!("  after {t:.0} s at rest: largest layer speed {speed:.2e} m/s");
}

/// Write `state`'s η, ū, v̄ and per-level u, v, T, S, ρ as little-endian
/// f64 after a header of (elements, nodes, levels) as u64 (`debug_3d=dump=`).
fn write_mode_dump(path: &str, state: &dg_rs::solver::state::Solution3D) {
    let mut bytes = Vec::new();
    for n in [state.n_elements, state.n_nodes, state.n_levels] {
        bytes.extend_from_slice(&(n as u64).to_le_bytes());
    }
    for field in [
        &state.eta.data,
        &state.ubar.data,
        &state.vbar.data,
        &state.u,
        &state.v,
        &state.temp,
        &state.salt,
        &state.rho,
    ] {
        for x in field.iter() {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
    }
    fs::write(path, bytes).expect("write the mode dump");
}

/// Read a dump of [`write_mode_dump`] into a copy of `like`.
fn read_mode_dump(
    path: &str,
    like: &dg_rs::solver::state::Solution3D,
) -> dg_rs::solver::state::Solution3D {
    let bytes = fs::read(path).expect("read the mode dump");
    let mut words = bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|&c| u64::from_le_bytes(c));
    let header: Vec<usize> = (0..3).map(|_| words.next().unwrap() as usize).collect();
    assert_eq!(
        header,
        [like.n_elements, like.n_nodes, like.n_levels],
        "dump {path} is for another domain"
    );
    let mut out = like.clone();
    for field in [
        &mut out.eta.data,
        &mut out.ubar.data,
        &mut out.vbar.data,
        &mut out.u,
        &mut out.v,
        &mut out.temp,
        &mut out.salt,
        &mut out.rho,
    ] {
        for x in field.iter_mut() {
            *x = f64::from_bits(words.next().expect("dump too short"));
        }
    }
    out
}

/// The nodes' x, y (m), bed (m), mass weights, and the σ of the layer
/// centres and the layers' Δσ, for analysing mode dumps (`PREFIX.geom.bin`:
/// a header of (elements, nodes, levels) as u64, then f64).
fn write_mode_geometry(
    prefix: &str,
    domain: &Domain,
    sigma: &dg_rs::vertical::SigmaGrid,
    state: &dg_rs::solver::state::Solution3D,
) {
    let (ne, nn, nl) = (state.n_elements, state.n_nodes, state.n_levels);
    let mut bytes = Vec::new();
    for n in [ne, nn, nl] {
        bytes.extend_from_slice(&(n as u64).to_le_bytes());
    }
    let mut push = |x: f64| bytes.extend_from_slice(&x.to_le_bytes());
    let points: Vec<[f64; 2]> = (0..ne * nn)
        .map(|idx| {
            domain.mesh.reference_to_physical(
                ElementIndex::new(idx / nn),
                domain.ops.nodes_r[idx % nn],
                domain.ops.nodes_s[idx % nn],
            )
        })
        .collect();
    points.iter().for_each(|p| push(p[0]));
    points.iter().for_each(|p| push(p[1]));
    domain.bathymetry.data.iter().for_each(|&b| push(b));
    (0..ne * nn).for_each(|idx| push(domain.geom.node_mass(idx / nn, idx % nn)));
    sigma.sigma_rho().iter().for_each(|&s| push(s));
    sigma.d_sigma().iter().for_each(|&s| push(s));
    fs::write(format!("{prefix}.geom.bin"), bytes).expect("write the mode geometry");
}

/// Write a patch of the domain (`debug_3d=around=K:R,export=PATH`) as a text
/// fixture: the vertices (m, relative to the first), the quadrilaterals, and
/// per element node its bed elevation and the bed's gradient (all faces
/// walls). Read by the 3D tests (`tests/data/`).
fn write_patch_fixture(
    path: &str,
    mesh: &dg_rs::mesh::Mesh2D,
    bathymetry: &Bathymetry2D,
    element: usize,
    radius: f64,
) {
    use std::fmt::Write as _;
    let [x0, y0] = mesh.vertices[0];
    let mut text = format!(
        "# dg-rs patch fixture: the elements of froya_coast.msh within {radius} m of element \
         {element}, bed smoothed for 3D (slopes3d), walls around\n"
    );
    writeln!(text, "vertices {}", mesh.vertices.len()).unwrap();
    for &[x, y] in &mesh.vertices {
        writeln!(text, "{} {}", x - x0, y - y0).unwrap();
    }
    writeln!(text, "quads {}", mesh.elements.len()).unwrap();
    for quad in &mesh.elements {
        writeln!(text, "{} {} {} {}", quad[0], quad[1], quad[2], quad[3]).unwrap();
    }
    writeln!(text, "bed {}", bathymetry.n_nodes).unwrap();
    for idx in 0..bathymetry.data.len() {
        writeln!(
            text,
            "{} {} {}",
            bathymetry.data[idx], bathymetry.gradient_x[idx], bathymetry.gradient_y[idx]
        )
        .unwrap();
    }
    fs::write(path, text).expect("write the patch fixture");
}

/// Lower-case ASCII file-name form of a station name.
fn slug(name: &str) -> String {
    name.to_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

/// Reference constants of the constituents a record resolves; the others of
/// P1, K2, N2 and Q1 are inferred with the ratios of `reference` (a longer
/// record at the same place) where it has them, else at equilibrium.
fn fit_record(
    times: &[f64],
    values: &[f64],
    candidates: &[&'static str],
    reference: Option<&ReferenceFit>,
) -> Result<ReferenceFit, AnalysisError> {
    let inferences = Inference::EQUILIBRIUM.map(|i| i.or_from(reference));
    HarmonicAnalysis::resolvable(candidates, record_length(times), 1.0, &inferences)
        .fit(times, values)
}

/// Last minus first time (s); 0 for an empty record.
fn record_length(times: &[f64]) -> f64 {
    times
        .first()
        .zip(times.last())
        .map_or(0.0, |(t0, t1)| t1 - t0)
}

/// Tidal ellipses of the constituents a current record resolves; the others
/// of P1, K2, N2 and Q1 are inferred with the major-axis ratio and lag
/// difference of `reference` (a longer record at the same place) where it
/// has them, else at equilibrium, for both components alike.
fn fit_current_record(
    times: &[f64],
    u: &[f64],
    v: &[f64],
    candidates: &[&'static str],
    reference: Option<&EllipseFit>,
) -> Result<EllipseFit, AnalysisError> {
    let inferences = Inference::EQUILIBRIUM.map(|i| i.or_from_ellipses(reference));
    HarmonicAnalysis::resolvable(candidates, record_length(times), 1.0, &inferences)
        .fit_ellipses(times, u, v)
}

/// The model's reference constants at a station after spin-up (from
/// `analysis_start`, Unix s) against the gauge's (whole record) and
/// NorKyst's (`norkyst`, the station atlas); RMSE against the observations
/// and the gauge's tidal prediction.
fn report_station(
    s: &Station,
    gauge_record: &Gauge,
    analysis_start: f64,
    norkyst: Option<&AtlasPoint>,
) -> Result<(), Box<dyn std::error::Error>> {
    let first = s.times.partition_point(|&t| t < analysis_start);
    let (times, eta) = (&s.times[first..], &s.eta[first..]);
    let gauge_times = gauge_record.observed.times();
    let gauge = gauge_record
        .observed_fit
        .as_ref()
        .map_err(|e| format!("gauge fit: {e}"))?;
    let model = fit_record(times, eta, &STATION_CONSTITUENTS, Some(gauge))?;

    let days = |t: &[f64]| (t[t.len() - 1] - t[0]) / 86_400.0;
    println!(
        "  Surface: model {:.1} days after spin-up (R² {:.4}), gauge {:.1} days (R² {:.4})",
        days(times),
        model.r_squared,
        days(gauge_times),
        gauge.r_squared
    );
    println!(
        "  name |  model H   G    |  gauge H   G    | ratio   ΔG (°)  |ΔZ| (m) | NorKyst H   G    |ΔZ| (m)"
    );
    let cmp = |a: &ReferenceConstant, h: f64, g: f64| {
        ConstituentComparison::new(
            a.name,
            a.amplitude,
            a.lag_deg.to_radians(),
            h,
            g.to_radians(),
        )
    };
    let (mut rss_gauge, mut rss_norkyst) = (0.0, 0.0);
    for c in &model.constants {
        let Some(g) = gauge.get(c.name) else { continue };
        let vs_gauge = cmp(c, g.amplitude, g.lag_deg);
        rss_gauge += vs_gauge.complex_difference().powi(2);
        let mut line = format!(
            "  {:4} | {:7.4} {:6.1}{} | {:7.4} {:6.1} | {:5.3} {:+7.1}   {:.4}",
            c.name,
            c.amplitude,
            c.lag_deg,
            if c.inferred { "*" } else { " " },
            g.amplitude,
            g.lag_deg,
            vs_gauge.amplitude_ratio,
            vs_gauge.phase_error_degrees(),
            vs_gauge.complex_difference()
        );
        if let Some(n) = norkyst.and_then(|p| p.constituents.iter().find(|n| n.name == c.name)) {
            let vs_norkyst = cmp(c, n.eta.0, n.eta.1);
            rss_norkyst += vs_norkyst.complex_difference().powi(2);
            line += &format!(
                "  | {:7.4} {:6.1}   {:.4}",
                n.eta.0,
                n.eta.1,
                vs_norkyst.complex_difference()
            );
        }
        println!("{line}");
    }
    println!("  (* inferred with the gauge's ratio)");
    print!(
        "  root sum of squares of |ΔZ|: {:.4} m vs the gauge",
        rss_gauge.sqrt()
    );
    if norkyst.is_some() {
        print!(", {:.4} m vs NorKyst", rss_norkyst.sqrt());
    }
    println!();

    // Time series after spin-up. The forcing has no mean level, so the bias
    // is the gauge's mean over the window: see the centred RMSE
    let model_series = TimeSeries::new(times, eta);
    let prediction = TimeSeries::new(times, &gauge.predict(times));
    for (what, reference) in [
        ("observations", &gauge_record.observed),
        ("gauge tidal prediction", &prediction),
    ] {
        let v =
            match StationValidationResult::compute(&gauge_record.station, &model_series, reference)
            {
                Ok(v) => v,
                Err(e) => {
                    println!("  vs {what}: not compared ({e})");
                    continue;
                }
            };
        let centred = (v.metrics.rmse.powi(2) - v.metrics.bias.powi(2))
            .max(0.0)
            .sqrt();
        println!(
            "  vs {what}: RMSE {:.3} m (centred {centred:.3}), bias {:+.3} m, correlation {:.4}, \
             std model {:.3} / reference {:.3} m ({} samples)",
            v.metrics.rmse,
            v.metrics.bias,
            v.metrics.correlation,
            v.model_std,
            v.obs_std,
            v.metrics.n_points
        );
    }
    Ok(())
}

/// The model's tidal ellipses of the depth-averaged current at a station
/// after spin-up (from `analysis_start`, Unix s) against NorKyst's (the
/// station atlas, if it has currents) and a current record's (its whole
/// record fitted); time-series skill against the record.
fn report_currents(
    s: &Station,
    analysis_start: f64,
    norkyst: Option<&AtlasPoint>,
) -> Result<(), Box<dyn std::error::Error>> {
    let first = s.times.partition_point(|&t| t < analysis_start);
    let (times, u, v) = (&s.times[first..], &s.u_east[first..], &s.v_north[first..]);
    let observed = s
        .current
        .as_ref()
        .map(|c| {
            c.observed_fit
                .as_ref()
                .map_err(|e| format!("current record fit: {e}"))
        })
        .transpose()?;
    let model = fit_current_record(times, u, v, &STATION_CONSTITUENTS, observed)?;
    let norkyst: Vec<TidalEllipse> = norkyst
        .map(|p| {
            p.constituents
                .iter()
                .filter_map(|c| {
                    Some(TidalEllipse::from_components(
                        c.name,
                        c.velocity?.0,
                        c.velocity?.1,
                    ))
                })
                .collect()
        })
        .unwrap_or_default();
    println!(
        "  Depth-averaged current: model {:.1} days after spin-up (R² {:.4}), mean {:+.3} m/s east, \
         {:+.3} m/s north",
        (times[times.len() - 1] - times[0]) / 86_400.0,
        model.r_squared(u, v),
        model.mean.0,
        model.mean.1
    );
    let references: Vec<(&str, &[TidalEllipse])> = [
        (!norkyst.is_empty()).then_some(("NorKyst", norkyst.as_slice())),
        observed.map(|o| ("record", o.ellipses.as_slice())),
    ]
    .into_iter()
    .flatten()
    .collect();
    let mut header = "  name | model major  minor   inc     G   ".to_string();
    for (name, _) in &references {
        header += &format!("| {name:>7} major  minor   inc     G    |ΔW| ");
    }
    println!("{header}  (m/s, degrees: inclination from east, Greenwich lag)");
    let mut rss = vec![0.0; references.len()];
    for e in &model.ellipses {
        let mut line = format!(
            "  {:4} | {:11.4} {:+7.4} {:5.1} {:6.1}{} ",
            e.name,
            e.major,
            e.minor,
            e.inclination_deg,
            e.lag_deg,
            if e.inferred { "*" } else { " " }
        );
        for ((_, reference), rss) in references.iter().zip(&mut rss) {
            match reference.iter().find(|r| r.name == e.name) {
                Some(r) => {
                    let d = e.complex_difference(r);
                    *rss += d * d;
                    line += &format!(
                        "| {:13.4} {:+7.4} {:5.1} {:6.1}  {d:.4} ",
                        r.major, r.minor, r.inclination_deg, r.lag_deg
                    );
                }
                None => line += &format!("| {:>46} ", "-"),
            }
        }
        println!("{line}");
    }
    if !references.is_empty() {
        let summary: Vec<String> = references
            .iter()
            .zip(&rss)
            .map(|((name, _), rss)| format!("{:.4} m/s vs {name}", rss.sqrt()))
            .collect();
        println!(
            "  (* inferred) root sum of squares of |ΔW|: {}",
            summary.join(", ")
        );
    }

    // Time series against the record, paired by time
    if let Some(record) = &s.current {
        let model_series = CurrentTimeSeries::new(times, u, v);
        let (paired, _) = CurrentTimeSeries::paired_by_time(&model_series, &record.observed);
        if paired.len() < 2 {
            return Err("the current record does not overlap the run after spin-up".into());
        }
        let r = ADCPValidationResult::compute(&record.station, &model_series, &record.observed);
        let m = &r.metrics;
        println!(
            "  vs current record ({} samples): RMSE u {:.3}, v {:.3}, speed {:.3} m/s; \
             bias u {:+.3}, v {:+.3} m/s; vector correlation {:.3}; direction RMSE {:.1}°",
            m.u_metrics.n_points,
            m.u_metrics.rmse,
            m.v_metrics.rmse,
            m.speed_metrics.rmse,
            m.u_metrics.bias,
            m.v_metrics.bias,
            m.vector_correlation,
            m.direction_rmse
        );
    }
    Ok(())
}

/// The atmospheric pressure gradient of the `wind` option.
fn pressure() -> AtmosphericPressure2D {
    AtmosphericPressure2D::from_direction(PRESSURE_GRADIENT, PRESSURE_DIRECTION)
}

/// The inverse-barometer level of the pressure forcing.
enum Level {
    /// The uniform gradient of the `wind` option
    Uniform(AtmosphericPressure2D),
    /// A weather model's pressure (`met=`)
    Gridded(GriddedAtmosphere2D),
}

impl BoundaryLevel for Level {
    fn level(&self, ctx: &BCContext2D) -> f64 {
        match self {
            Self::Uniform(p) => p.inverse_barometer(ctx.position.0, ctx.position.1, ctx.time),
            Self::Gridded(g) => g.level(ctx),
        }
    }
}

/// Characteristic OBC with external data from `provider`, raised by the
/// inverse-barometer level of the pressure forcing, if any.
fn characteristic<P: ExternalStateProvider + 'static>(
    provider: P,
    level: Option<Level>,
) -> Box<dyn SWEBoundaryCondition2D> {
    match level {
        Some(level) => Box::new(CharacteristicOBC::new(InverseBarometer::new(
            provider, level,
        ))),
        None => Box::new(CharacteristicOBC::new(provider)),
    }
}

/// Weather-model wind and pressure (`met=`) on the mesh, ramped up like the
/// tide.
#[cfg(feature = "netcdf")]
fn weather(
    domain: &Domain,
    opts: &Options,
    clock: &ModelClock,
    t_end: f64,
) -> Result<Option<GriddedAtmosphere2D>, Box<dyn std::error::Error>> {
    let Some(projection) = domain.projection.filter(|_| !opts.met.is_empty()) else {
        return Ok(None);
    };
    let reader = Arc::new(AtmosphereReader::from_files(&opts.met)?);
    println!("  {}", reader.summary());
    let atmosphere =
        GriddedAtmosphere2D::new(reader, &domain.mesh, &domain.ops, projection, *clock)?
            .with_ramp_up(3600.0 * opts.ramp_hours);
    atmosphere.check_time_coverage(0.0, t_end)?;
    Ok(Some(atmosphere))
}

#[cfg(not(feature = "netcdf"))]
fn weather(
    _domain: &Domain,
    _opts: &Options,
    _clock: &ModelClock,
    _t_end: f64,
) -> Result<Option<GriddedAtmosphere2D>, Box<dyn std::error::Error>> {
    Ok(None)
}

type OpenBoundary = (Box<dyn SWEBoundaryCondition2D>, String);

/// The parent model of `norkyst=` at the open boundary, and its reader (with
/// the profiles for a 3D tide).
type Nested = (OceanModelState, Arc<OceanModelReader>);

/// What the boundary atlas is corrected with: the first gauge with a
/// whole-record fit, and NorKyst at it (the nearest `station_atlas` point
/// within `STATION_MAX_OFFSET`, and its distance in m).
struct AtlasReference<'a> {
    station: &'a str,
    fit: &'a ReferenceFit,
    norkyst: Option<(&'a AtlasPoint, f64)>,
}

impl<'a> AtlasReference<'a> {
    fn new(stations: &'a [Station], station_atlas: Option<&'a TidalAtlas>) -> Option<Self> {
        stations.iter().find_map(|s| {
            let fit = s.gauge.as_ref()?.observed_fit.as_ref().ok()?;
            let norkyst = station_atlas
                .and_then(|a| a.nearest(s.longitude, s.latitude))
                .filter(|&(_, d)| d <= STATION_MAX_OFFSET);
            Some(Self {
                station: &s.name,
                fit,
                norkyst,
            })
        })
    }
}

/// `atlas` corrected at the reference gauge (`TidalAtlas::infer`): first the
/// `gauge_gains` constituents are scaled by the complex ratio of the gauge's
/// constant to NorKyst's there, which keeps NorKyst's spatial structure and
/// fixes it at the gauge (the diurnals: NorKyst's K1 is 1.14×, +8.6° and its
/// O1 0.95×, −17.3° at Mausund); then the `gauge_ratios` constituents are
/// re-inferred from their neighbours with the gauge's ratios. Unchanged
/// without a gauge fit, and without the gains if NorKyst is not known at
/// the gauge.
fn corrected_atlas(
    atlas: &TidalAtlas,
    opts: &Options,
    reference: Option<&AtlasReference>,
) -> Result<TidalAtlas, Box<dyn std::error::Error>> {
    let mut atlas = atlas.clone();
    let signed = |lag: f64| (lag + 180.0).rem_euclid(360.0) - 180.0;
    let Some(&AtlasReference {
        station,
        fit,
        norkyst,
    }) = reference
    else {
        if !opts.gauge_gains.is_empty() || !opts.gauge_ratios.is_empty() {
            println!(
                "  No gauge fit: the atlas keeps its own {:?}",
                [&opts.gauge_gains[..], &opts.gauge_ratios[..]].concat()
            );
        }
        return Ok(atlas);
    };
    match norkyst {
        None if !opts.gauge_gains.is_empty() => println!(
            "  No NorKyst constants at {station} (station_atlas=): the atlas keeps its own {:?}",
            opts.gauge_gains
        ),
        None => {}
        Some((point, distance)) => {
            for &name in &opts.gauge_gains {
                let (Some(gauge), Some(source)) = (
                    fit.get(name),
                    point.constituents.iter().find(|c| c.name == name),
                ) else {
                    return Err(format!("gauge_gains: {station} or NorKyst lacks {name}").into());
                };
                let gain = gauge.amplitude / source.eta.0;
                let lag = signed(gauge.lag_deg - source.eta.1);
                atlas.infer(name, name, gain, lag)?;
                println!(
                    "  Atlas {name} × {gain:.3}, lag {lag:+.1}° (gauge {station} over NorKyst \
                     {distance:.0} m from it)"
                );
            }
        }
    }
    for &name in &opts.gauge_ratios {
        let Some(inference) = Inference::EQUILIBRIUM.iter().find(|i| i.name == name) else {
            return Err(format!("gauge_ratios: {name} is not P1, K2, N2 or Q1").into());
        };
        let from = inference.from;
        let i = Inference::from_reference(name, from, fit)
            .ok_or(format!("gauge_ratios: {station} lacks {name} or {from}"))?;
        atlas.infer(name, from, i.amplitude_ratio, i.lag_offset_deg)?;
        println!(
            "  Atlas {name} = {:.3} × {from}, lag {:+.1}° (ratio at {station})",
            i.amplitude_ratio,
            signed(i.lag_offset_deg)
        );
    }
    Ok(atlas)
}

/// NorKyst with its tides corrected (`nest_tides=corrected`): the boundary
/// atlas is NorKyst's own harmonic fit, so adding (corrected atlas − atlas)
/// keeps NorKyst's residual (coastal current, surge) and replaces its
/// `gauge_gains` and `gauge_ratios` constituents by the gauge-corrected
/// ones, at the boundary
/// and across the relaxation band.
fn correct_parent_tides(
    domain: &Domain,
    opts: &Options,
    reference: Option<&AtlasReference>,
    parent: OceanModelState,
    t_end: f64,
) -> Result<OceanModelState, Box<dyn std::error::Error>> {
    let atlas_path = Path::new(&opts.tides);
    let (Some(projection), true) = (&domain.projection, atlas_path.exists()) else {
        println!(
            "  No tidal atlas at {}: NorKyst's tides as they are",
            opts.tides
        );
        return Ok(parent);
    };
    let raw = TidalAtlas::read(atlas_path)?;
    let correction = corrected_atlas(&raw, opts, reference)?.difference(&raw)?;
    let largest = correction
        .points
        .iter()
        .flat_map(|p| p.constituents.iter().map(|c| (c.eta.0, c.name)))
        .fold((0.0, ""), |a, b| if b.0 > a.0 { b } else { a });
    if largest.0 == 0.0 {
        return Ok(parent);
    }
    println!(
        "  NorKyst tides corrected by (corrected − raw) atlas, largest {:.3} m ({})",
        largest.0, largest.1
    );
    Ok(parent.with_tidal_correction(
        &correction,
        projection,
        t_end,
        ATLAS_COVERAGE + 1000.0 * opts.band_km,
    )?)
}

/// Open-boundary condition and a description of the forcing: NorKyst
/// nesting (`norkyst=`), else atlas tides (`tides=`), else uniform M2; raised
/// by the inverse-barometer `level` of the pressure forcing (for nesting only
/// with `ib=1`).
fn open_boundary(
    domain: &Domain,
    opts: &Options,
    clock: &ModelClock,
    t_end: f64,
    reference: Option<&AtlasReference>,
    parent: Option<OceanModelState>,
    level: Option<Level>,
) -> Result<OpenBoundary, Box<dyn std::error::Error>> {
    if let Some(parent) = parent {
        let description = format!(
            "NorKyst nesting ({})",
            parent.reader().velocity_source().unwrap_or("no currents")
        );
        let level = level.filter(|_| opts.nesting_ib);
        return Ok((characteristic(parent, level), description));
    }

    let clock = *clock;
    let atlas_path = Path::new(&opts.tides);
    if let (Some(projection), true) = (&domain.projection, atlas_path.exists()) {
        let atlas = corrected_atlas(&TidalAtlas::read(atlas_path)?, opts, reference)?;
        let tides = atlas
            .boundary_tides(
                &domain.mesh,
                &domain.ops,
                projection,
                BoundaryTag::Open,
                &clock,
                t_end,
                ATLAS_COVERAGE,
            )?
            .with_ramp_up(3600.0 * opts.ramp_hours);
        let tides = if opts.tide_transport > 0.0 {
            tides.with_transport_scaling(opts.tide_transport)
        } else {
            tides
        };
        let mut description = describe(&tides, atlas_path);
        if opts.tide_transport > 0.0 {
            description.push_str(&format!(
                ", velocity transport-scaled (≤ {}×)",
                opts.tide_transport
            ));
        }
        return Ok((characteristic(tides, level), description));
    }

    println!("  No tidal atlas at {}: uniform M2", opts.tides);
    let tide = HarmonicTide::m2(M2_AMPLITUDE, 0.0)
        .with_ramp_up(3600.0 * opts.ramp_hours)
        .with_nodal_corrections(&clock, 0.5 * t_end);
    let description = format!("M2 of {M2_AMPLITUDE} m in one phase at the open boundaries");
    Ok((characteristic(tide, level), description))
}

/// Constituents, forced nodes, and the M2 amplitude range and phase spread
/// along the open boundary.
fn describe(tides: &BoundaryTides, path: &Path) -> String {
    let mut text = format!(
        "{} constituents from {} at {} open-boundary nodes",
        tides.names().len(),
        path.display(),
        tides.n_nodes()
    );
    if let Some(m2) = tides.names().iter().position(|&n| n == "M2") {
        let constants: Vec<(f64, f64)> = (0..tides.n_nodes())
            .map(|slot| tides.elevation_constants(slot, m2))
            .collect();
        let (lo, hi) = constants.iter().fold((f64::MAX, f64::MIN), |(lo, hi), c| {
            (lo.min(c.0), hi.max(c.0))
        });
        // Phases relative to the first node, in (−180°, 180°]
        let relative = constants
            .iter()
            .map(|c| (c.1 - constants[0].1 + 180.0).rem_euclid(360.0) - 180.0);
        let (p_lo, p_hi) =
            relative.fold((f64::MAX, f64::MIN), |(lo, hi), p| (lo.min(p), hi.max(p)));
        text += &format!("; M2 {lo:.3}–{hi:.3} m, phase spread {:.1}°", p_hi - p_lo);
    }
    text
}

#[cfg(feature = "netcdf")]
fn netcdf_mesh_info(domain: &Domain, projection: &LocalProjection) -> NetCDFMeshInfo {
    let (mut x, mut y, mut lat, mut lon) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for k in ElementIndex::iter(domain.mesh.n_elements) {
        for i in 0..domain.ops.n_nodes {
            let [xi, yi] =
                domain
                    .mesh
                    .reference_to_physical(k, domain.ops.nodes_r[i], domain.ops.nodes_s[i]);
            let (la, lo) = projection.xy_to_geo(xi, yi);
            x.push(xi);
            y.push(yi);
            lat.push(la);
            lon.push(lo);
        }
    }
    NetCDFMeshInfo::from_xy(x, y).with_latlon(lat, lon)
}

/// Depth, surface elevation and velocities per node
#[cfg(feature = "netcdf")]
fn netcdf_fields(domain: &Domain, q: &SWESolution2D) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let (mut h, mut eta, mut u, mut v) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for k in ElementIndex::iter(domain.mesh.n_elements) {
        for i in 0..domain.ops.n_nodes {
            let s = q.get_state(k, i);
            let (ui, vi) = s.velocity_simple(Depth::new(WetDryConfig::DEFAULT_H_DRY));
            h.push(s.h);
            eta.push(s.h + domain.bathymetry.get(k, i));
            u.push(ui);
            v.push(vi);
        }
    }
    (h, eta, u, v)
}

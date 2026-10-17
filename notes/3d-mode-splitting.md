# 3D: mode splitting (P4 intro, P4.1)

## Priority 4: 3D Rebuilt on the ROMS Recipe (`review-2026-09-25.md` §2, §3, §7 Phase D)

The 2026-02-11 plan (vertical infrastructure → mode splitting → mixing → physics → validation) is superseded. The scaffolding exists, but the coupling must be restructured before new physics is added. See `3D_TODO.md` for background (its checklists predate the current code).

### P4.1 Mode splitting

**Before PR 1:** SSP-RK3 wrapped a Forward Euler barotropic subcycle that ran in every stage. It was first-order, M2 lost 3–14 % per period, and it cost 6·n_bt 2D RHS per step (Shchepetkin & McWilliams 2005; `review-2026-09-25.md` §2).
**After PR 1:** one filtered pass per step, 0.02–0.1 % barotropic amplitude loss per period, ≈ 3.9·n_bt 2D RHS per step.
**After PR 2:** G holds only what the 2D module cannot compute: the baroclinic PGF, the shear dispersion and the 3D stresses. Its step average is AB3-extrapolated, and the slow coupling is second order.
**After PR 3:** each step also records DU_avg2, the transport that moved η (η̄ − ηⁿ = −Δt∇·DU_avg2), for the 3D continuity of P4.2.

**Target step n → n+1 (plan of 2026-09-26):**
1. One 3D RHS at tⁿ, after refreshing density. It is reused as the first 3D RK stage.
2. Slow forcing G = D·(⟨R₃D(u)⟩ − R_adv+Cor(ū)) + (τ_s − τ_b)/ρ₀ (see the design decisions). Its step average comes from Gⁿ, Gⁿ⁻¹, Gⁿ⁻² by AB3 with variable steps (lower order on the first two steps).
3. One barotropic pass over [t, ≈ t + 1.5·dt] with n_bt = ⌈dt/dt_bt,max⌉ from the 2D CFL. Accumulate:
   - the primary-weighted η̄ and transport (DU_avg1), which become the new barotropic state;
   - the secondary-weighted face-flux transport (DU_avg2), for P4.2.
4. 3D RK stages with Hz at each stage time from the pass, and density recomputed every stage.
5. Implicit vertical diffusion, then reset the depth mean of u to DU_avg1/D. The existing reconciliation does this.

**Design decisions (deviations from the ROMS recipe):**
- **Fast-mode sub-steps are SSP-RK3, not generalised forward-backward (AB3-AM4).**
  - FB needs continuity and momentum evaluated separately at staggered times. The DG upwind fluxes couple h and hu in both equations.
  - The Zhang–Shu positivity guarantee that wet/dry relies on needs an SSP scheme.
  - Cost is about 4.5·n_bt RHS per step (vs 6·n_bt today) at the DG CFL. It reuses `SWEPhysics2D` (wet/dry, limiter, `compute_rhs_into`).
  - FB can come later as an optimisation once the coupling is verified.
- **G is Thetis-style, not the ROMS `rufrc`** (revised 2026-09-26, PR 2). The plan was the full PGF in the 3D RHS with G = ∫R₃D dz − R₂D(q̄ⁿ) + stresses. It was dropped as structurally unsound here:
  - In that construction each step's fast pass adds only R₂D(q) − R₂D(qⁿ). The bulk of the pressure force on the slow barotropic mode (tides, seiches) would come from ∫R₃D.
  - The 3D PGF (`compute_pressure_gradient`) differentiates η within each element and never lifts the jumps at element faces (P4.3). So the slow mode would lose the DG pressure coupling between elements.
  - Instead, the 2D module owns the depth-mean flow: −g∇η with its DG face coupling, advection of ū, Coriolis on ū, and its friction on ū.
  - G = D·(⟨R₃D(u)⟩ − R_adv+Cor(ū)) + (τ_s − τ_b)/ρ₀: the same 3D advection + Coriolis operator applied to columns of uniform ū removes the mean-flow part. What is left is the baroclinic PGF, the shear dispersion −∇·⟨u′u′⟩ and the stresses. For unsheared flow G is exactly the stress.
  - The 3D PGF stays baroclinic-only (P0.6).
  - Rule: configure the 2D module with the same Coriolis parameter, and without wind/friction sources that duplicate `Forcing` or `with_surface_stress` (`Hydrostatic3D` module docs; `GriddedAtmosphere2D::split_for_3d` does it for weather-model forcing).
  - Revisit the ROMS form once P4.3 lifts the 3D pressure/η jumps at faces. (P4.3 does since 2026-09-30: the 3D PGF with `rho_ref = 0` has the DG face coupling of η.)

**PR 1: one barotropic pass** — done on `feat/p4-1-barotropic-pass`
- [x] The fast pass steps transport (h, hu, hv) in a `SWESolution2D` through `SWEPhysics2D`: positivity, limiter, wet/dry and implicit damping at every stage. The unguarded 1/h (`Hydrostatic3D::compute_rhs_2d`) is gone.
  - `Solution3D` still stores η/ū/v̄ as the slow barotropic state, with ū = D̄ū/D̄ after the pass (guarded for h̄ ≤ 0). The 3D kernels read η; moving them to h is P4.2 work.
- [x] One SSP-RK3 barotropic pass per baroclinic step. η/ū/v̄ leave the 3D RK combination: the 3D stages get constant rates, i.e. the barotropic state linearly interpolated to each stage time.
- [x] Shchepetkin & McWilliams (2005) power-law filter (p = 2, q = 4, r = 0.284) over ≈ 1.3·dt, replacing the Hann window over 2·dt.
  - **Its forcing reach** (2026-10-17): the pass runs the whole window, so a step from tⁿ evaluates boundaries and surface forcing up to tⁿ + (M*/n_bt)·Δt, at most 4/3 Δt (n_bt = 6; 1.25–1.33 for n_bt ≤ 11, → 1.32; `BarotropicFilter::reach`, `ModeSplitIntegrator::FORCING_REACH`, gate `the_filter_window_stays_within_the_forcing_reach` over n_bt 4–2000). A run's last step reads past `t_end`: the 72 h Frøya run on a 72 h parent panicked in its last step, 0.62 s past the parent's last snapshot, after 4.7 h of wall time. `froya_real_data` caps the 3D step at 60 s (`MAX_DT_3D`, or `dt_3d=`; it does not bind at the coastline mesh's 10–18 s) and checks the parent and the weather to `t_end + Δt_max/3` at setup.
  - The r term makes the second moment about t + dt almost zero. Measured per-period amplitude loss at 50 steps per period: Hann 5.0 %, power law with r = 0: 3.2 %, r = 0.284: 0.02–0.1 % (n_bt 40–7). r = 0.3 amplifies.
  - The centring iteration needs n_bt ≥ 4 (`MIN_BAROTROPIC_SUBSTEPS`).
- [x] Barotropic CFL control: n_bt = max(4, ⌈dt/dt_bt,max⌉) every step, at `min(barotropic_cfl, max_cfl)`.
- [x] Density refreshed at every 3D RK stage (`Simulation3D` stage hook).
- [x] One `step`. The 3D and 2D stage buffers and the field buffers are reused across steps, and `Solution3D::copy_from` copies in place. Still allocating: `compute_rhs_3d` internals (P4.5).
- [x] Gate: `seiche_amplitude_is_kept_by_the_mode_split`. It loses 0.035 % per period more than the 2D model at n_bt = 23 (was ≈ 23 % per period), and conserves volume to round-off.
  - The 100-period form was cut to 10 periods: the loss per period is constant, and the test must stay cheap in debug CI (16 s).

**PR 2: consistent slow forcing** — done on `feat/p4-1-slow-forcing`
- [x] G without the double count: D·(⟨R₃D(u)⟩ − R_adv+Cor(ū)) instead of D·⟨R₃D(u)⟩, with AB3 step averages (`step_average_weights`, variable steps; the history restarts when a run does not continue).
  - In a nonlinear unsheared seiche the old G drifted from the 2D model by 0.14 of the amplitude in two periods; the new G drifts by 1.8e-3, falling with dt.
- [x] Surface and bottom stress into G, as (τ_s − τ_b)/ρ₀ from `Forcing`. The vertical diffusion now also divides by the model's ρ₀, not 1025, so G and the columns agree.
  - τ_b is still the user's constant; a quadratic drag from the bottom-layer velocity, consistent with the 2D friction, is P4.4. (Done 2026-09-30: `with_bottom_drag`, see P4.4.)
- [x] ~~Full PGF in the 3D RHS~~: dropped, see the design decisions.
- [x] `ModeSplitPhysics` trait: the splitter's view of the 3D model (2D module, 3D RHS, slow forcing, implicit vertical terms, stage hook). `Hydrostatic3D` implements it; `ModeSplitIntegrator::step(state, physics, dt, t)`.

**PR 3: transport for 3D continuity** — done on `feat/p4-1-du-avg2`
- [x] DU_avg2 (`BarotropicTransport`, `ModeSplitIntegrator::barotropic_transport`): the nodal (hu, hv) and the face mass flux F*_h of every RK stage of the pass, weighted by W_j·b_s/n_bt (secondary filter weights W_j = Σ_{m≥j} w_m, SSP-RK3 weights b = (1/6, 1/6, 2/3)).
  - The 2D kernel writes F*_h per element face node on request (`compute_rhs_swe_2d_face_mass_into`, serial and parallel; `BarotropicPhysics`). The plain RHS is unchanged bit for bit.
  - `BarotropicTransport::divergence_into` is the kernel's strong-form divergence. η̄ − ηⁿ = −Δt·∇·DU_avg2 holds to 4e-13 of the η change for `Standard` and `EntropyStable`: the mass part of the two-point flux is {{hu}}, so the flux-differencing volume term is the nodal D·(hu, hv).
  - `WetDry` elements with a dry node use subcell finite volumes, and the positivity limiter and wet/dry correction change nodal depths. There only the element balance ∫(η̄ − ηⁿ) = −Δt∮F*_h holds, to round-off. P4.2's per-level correction has to work with that: in those elements, correct at the element (or subcell) level, not the nodal one.
- [x] Fixed on the way (PR 1 bug): η at the end of a step came from the RK combination of constant rates, which can leave a dry node a hair below its bed. The next pass then started from h ≈ −1e-16 and the positivity limiter "emptied" the element (15 clips in 10 steps on a beach). Now η = h̄ + B and ū = D̄ū/D̄ are set exactly from the filtered state; for h̄ ≥ 0, (h̄ + B) − B ≥ 0 in floating point.
- Gate tests: `barotropic_transport_moves_the_free_surface` (nodal identity, both formulations; with primary weights or without the face flux it fails by O(1)), `barotropic_transport_balances_every_element_with_wetting_and_drying` (beach, 0 clips), `face_mass_flux_balances_each_element` (kernel: RHS unchanged, serial = parallel, ∫dh/dt = −∮F*_h for all three formulations).

- [x] Phase floor of the splitting for internal waves (noted 2026-09-30, P4.6): with the levels converged, the mode-1 internal seiche's period was 0.05–0.1 % short of the free-surface speed, and the gap grew as the baroclinic step shrank (−0.056 / −0.049 / −0.077 / −0.099 % at 480 / 240 / 120 / 60 s on 80 levels; 65–520 steps per period).
  - **Answered 2026-09-30: mostly the wave's nonlinearity, not the splitting.** At 480 s steps the period error is −0.056 % at a = 0.5 m (a/H = 0.025), −0.011 % at 0.25 m, +0.004 % at 0.05 m (∝ a²). The barotropic substeps do not matter (64 minimum substeps: the same to 1e-4 %).
  - [ ] A small dt-dependent residual remains at any amplitude: +0.011 / −0.018 / −0.039 % at 240 / 120 / 60 s for a = 0.05 m (130–520 steps per period); ≈ 0.01 % at a realistic M2 internal-tide step. Per-step coupling (the filtered reset, or the stages' interpolated barotropic state) if a tighter gate ever needs it.

**Gate tests:**
- [x] 2D–3D equivalence: `unsheared_flow_matches_the_2d_model`. A nonlinear seiche (η/H = 0.05) over two periods: 1.8e-3 of the amplitude at 50 steps per period (old G: 0.14).
- [x] Seiche amplitude: `seiche_amplitude_is_kept_by_the_mode_split`. The filtered splitting loses 0.035 % per period more than the 2D model, measured over 10 periods (was ≈ 23 % per period).
  - The "< 1 % over 100 periods" target was too strict for any filtered-reset splitting. The loss is ≈ μ₂ω²/2 per step, first order in Δt with a constant ≈ 1e-3 of a plain average's. For M2 at Δt = 300 s it is ≈ 0.008 % per period.
- [x] Temporal convergence: `slow_forcing_is_integrated_to_second_order`. The AB3 steps are third order; the two starting steps set second order globally (measured ratios 4.5 → 4.1). The filter's μ₂ term is separate: see the previous item.
- [x] Wind setup: `wind_setup_balances_the_surface_stress`, slope 1.0001 × τ/(ρ₀gD).
- [x] Ekman: `wind_drives_the_ekman_inertial_transport`, 3.9e-4 of τ/(ρ₀f) from the inertial-Ekman solution.
- [x] Column momentum after one implicit diffusion step = Δt·(τ_s − τ_b)/ρ₀: `column_momentum_changes_by_the_stress_impulse`.

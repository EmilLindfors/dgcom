# Overview: state, strategy, metrics, references

# TODO - DG Coastal Ocean Model (dg-rs)

This roadmap is driven by the full-crate review of **2026-09-25** in `review-2026-09-25.md`, which supersedes the 2026-07-08 review (in git history). The 2026-02-11 reports (ROMS/NorKyst comparison, math evaluation, BC assessment, devil's advocate) were deleted on 2026-10-10 as superseded by that review; they are in git history.

**Current state.**
- **What is solid:** the DG core (GLL operators, strong-form DGSEM, Roe/HLL, SSP-RK3, SoA/rayon), the tidal astronomy module, and the NorKyst ingest readers. 1,024 lib + 61 integration + 59 doctests pass (`--no-default-features --features parallel,simd`).
- **What the tests don't cover:** the suite runs in under 2 s, and the physics tests only exercise easy configurations.
- **What the review found broken:**
  - Realistic bathymetry is not well-balanced: metre-per-second spurious currents within an hour. (Since fixed opt-in by the split-form DGSEM, P1.2.)
  - The cell-average workaround leaks mass. (Since fixed, P0.13.)
  - Flather-type open boundaries reflect about 1/3 of outgoing waves. (Since fixed, P0.12.)
  - Coastline-fitted meshes cannot be loaded.
  - Mode splitting is first-order and damps the barotropic mode.
  - The 3D layer is scaffolding with three blocker-level math errors. (Vertical advection ×D since fixed, P0.16; tracer constancy and the σ pressure gradient since fixed, P4.2/P4.3, 2026-09-30.)
- **Cost:** as configured, roughly ~50× ROMS core-hours for a 2D tidal run (cost-model estimate, `review-2026-09-25.md` §5).

**Strategic direction.**
1. **Make the 2D barotropic tide model correct, then cheap, then validated.** This is where DG's fjord-geometry and accuracy-per-DOF advantages can actually beat ROMS.
2. **Then rebuild 3D on the ROMS recipe.**

"Faster than ROMS" is a claim to be *proven* with a cost-vs-error benchmark (P3.2), plausible for 2D tides at P3+ with local time stepping or an implicit free surface. It should not be assumed.

## Key Metrics

| Metric | Current (2026-09-25) | Target (2D operational) |
|---|---|---|
| Tests | 1,090 lib + 88 integration + 59 doc (no-default) | + the P1.7 gating physics tests |
| Lake-at-rest, steep nodal B | Split forms: RHS ≤ 4e-12 for any nodal B, P1–P4. Real Frøya bathymetry (0–404 m, `WetDry`, P2): max \|u\| 1.5e-10 m/s after 1 h (the collocated scheme: 0.9–2.3 m/s or blow-up) | residual < 1e-10, p = 1–4 |
| Mass conservation, face-discontinuous B | Machine precision (P0.13; split forms by construction) | machine precision |
| Open-boundary reflection (Flather) | < 1e-5 (P0.12, `tests/open_boundary_flather_test.rs`) | < 1 % |
| SWE convergence | P1 1.67, P2 3.02 (linearised, flat, periodic; thresholds 1.5/2.5) | N+1, nonlinear, bathymetry, curved |
| Allocations per 2D step (production stepper) | `Simulation` path: 0 B warm (without limiter/viscosity); legacy `ssp_rk3_swe_2d`: ~33 full-field arrays | 0 |
| 2D tidal cost vs ROMS (model estimate) | ~47× core-hours | ≤ 1× per unit accuracy (P3.2) |
| Validated against observations | Mausund (Kartverket MSU), 15 days at 1 km: M2 1.03×, +5.4° (≈ +2.3° at 500 m); tidal-prediction RMSE 9.8 cm | 5+ tide gauges, ADCP |
| Multi-day stability | One M2 cycle on Frøya (9,653 P2 elements, 11.4 min wall, 0 clips) | 30+ days, real domain |
| 3D | Scaffolding; tracer constancy (P4.2) and the balanced PGF (P4.3) fixed 2026-09-30; vertical advection ×D (P0.16) and the tracer wall leak (P0.22) fixed; mode splitting: one filtered barotropic pass (0.035 % seiche damping per period), second-order slow coupling, wind setup/Ekman to < 0.1 % (P4.1 PR 1–2; DU_avg2 is PR 3) | lock exchange, mode-1 internal wave, seamount spin-up |
| GPU | Non-functional (P0.11) | fixed or replaced (P2.7) |

## Altitude pass on the TODO list (2026-11-15)

Asked: what to work on next for the most gain. Taken up from the next item to the outcome behind it: **farm-site currents someone can act on, with a measured error, on any coast** ("dgcom's current at site X is within Y of a measured record, and closer than NorKyst's").

- **The η score at a nested gauge measures the parent, not dgcom.** The 2D run follows NorKyst within 0.5–0.7 cm and the 3D run follows 2D within 0.2 cm (`validation.md`, 2026-06-12). The open fit refinements (K2's lead, the diurnal age at Vardø, nodal f/u per time, shelf waves, a quadratic nodal drift) move millimetres of a score the model does not own.
- **Where dgcom differs from NorKyst it has no reference.** The M2 current is 0.56–0.61 of NorKyst's where NorKyst is under 50 m; the island lee is 0.12–0.19 against 0.255 m/s; 3D/2D is 0.78 in 3–20 m; Kattholmen's mean current is 0.08 m/s. About ten items end in "settle with observed currents" (P3.1 ×3, P1.3 ×2, F.1 ×2, F.2's K, P4.4's roughness, P4.6 Frohavet), and the project holds none.
- **Chiloé's inner sea is where η tests dgcom.** It is forced by an atlas, and Puerto Montt is 16–18 cm, 22 % over at neaps.
- **Scoring was slow.** A change to the fit meant every scored run again.

Decided (the user, 2026-11-15): `dgcom score` first (done the same day, `case-files.md`), then the 15-day 2D runs that settle `nest_tides` (P3.4/P1.5; 2D parent files of 15 days should load whole, ≈ 22 MB per 3 days, so no lazy reading), with the search for measured currents beside it (P3.1, leads in `validation.md`). The proposed mechanism behind `dgcom score` is a scoreboard: every run scored the same way, against the observations, against its parent, and with the parent's own error over the window. Backtested on what the notes record: the walled parent file's 1.9–2.1 cm at Mausund (NorKyst's own 7.5) put the model at least 5.4 cm from its parent, against ≈ 0.5 cm for an honest nest, so a model-against-parent column flags it on the first run; the pre-registered 2026-06-12 day was NorKyst's worst of June (11.9 of 2.3–12.1 cm), which the parent's spread over the month beside the day would have shown at once.

### The triage of TODO.md (2026-11-15)

The user, after the pass: "I don't think it's realistic that we have 200 items that are still worth doing". There were 399, 38 of them hedged in their own words ("low priority", "only if", "not worth it"), because every session filed every opportunity it noticed as a line (the old CLAUDE.md rule). Each line was judged against the outcome above:

- **Kept (22):** on the path now (the 15-day nest score and the `nest_tides` default, measured currents, Chiloé's spring–neap score and its gauges, `dgcom score` against the parent, the farm's report and particles in case files, retiring `froya_real_data`, the 3D bed's levels from the case's tide, the topobathy datum, the cage layout), or a defect of the runs made at farm depths (the 3D bed layers over steep beds, 3D tracers out of range at dry and thin columns, the start-up cooling at the farms, films on land), plus the 3D defaults to decide on one run, the convergence gate CLAUDE.md asks for, the legacy and dead code in two deletions, and the viewer in CI. Ten of them are new lines that fold 38 old ones.
- **Parked (285):** moved verbatim to each notes file's "Out of TODO.md (2026-11-15)". Most of the waves (41 of 50), the performance (37 of 43), the 3D numerics (40 of 50) and the viewer (20 of 23) are there: real, but no run or decision waits on them now.
- **Dropped (64):** done (stage lists, the turbulence advection's cost, H_s at stations), superseded (one-day and 1 km reruns, by the 15-day case runs and the #231 sweep), duplicates, the surf zone, and the lines that said themselves they were not worth it.

The rule that grew the list changed with it (CLAUDE.md, `README.md` conventions): an opportunity goes under Parked in its notes file; a line goes into `TODO.md` only when it beats the ones there.

---

## References

- Hesthaven & Warburton (2008), Nodal DG Methods.
- Toro (2009), Riemann Solvers for Fluid Dynamics.
- Wintermeyer, Winters, Gassner & Kopriva (2017); Wintermeyer et al. (2018), entropy-stable well-balanced DGSEM for SWE; wet/dry extension.
- Audusse et al. (2004), hydrostatic reconstruction; Xing & Shu (2006); Xing, Zhang & Shu (2010).
- Zhang & Shu (2010), positivity-preserving limiters; Kuzmin (2010), vertex-based slope limiting; Vater, Beisiegel & Behrens (2019).
- Flather (1976); Chapman (1985); Kärnä et al. (2011), weak Flather in DG; Davies (1976), flow relaxation.
- Shchepetkin & McWilliams (2003), density-Jacobian PGF; (2005), ROMS split-explicit stepping.
- Umlauf & Burchard (2003); Warner et al. (2005), GLS vertical mixing.
- Kärnä et al. (2018), Thetis 3D.
- NorKyst v3 preprint (2025), egusphere-2025-3986.

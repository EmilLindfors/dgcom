# Notes

Context behind `TODO.md`, one file per area: the findings, measurements, decisions, dead ends and finished work that the one-line items in `TODO.md` point to. Read the file for the area you are working in; you rarely need more than one or two.

The files were split out of the old 1,500-line `TODO.md` on 2026-10-10, verbatim, so the item IDs (P1.3, F.4, …) still match the CHANGELOG, the code's doc comments and git history. Notes are point-in-time: each entry is dated, and the code wins where they disagree.

| File | Area | IDs |
|---|---|---|
| [overview.md](overview.md) | State and strategy at the 2026-09-25 review, key metrics, references | — |
| [validation.md](validation.md) | Frøya/Mausund runs, the tidal error budget at the gauge, current comparisons with NorKyst, the ROMS cost benchmark, long-run stability | P3 |
| [norkyst-issues.md](norkyst-issues.md) | NorKyst-800 v3 against 17 Kartverket gauges and its tidal forcing files: N2 (fixed 2026-03), K2/P1/Q1, datum, pressure, ubar/vbar; the report and draft message to MET | P3.1, P1.5 |
| [farm.md](farm.md) | The salmon-farm application target (stage A 2D, stage B 3D), cage drag in 2D and 3D, farm-site readers | F.1 |
| [particles.md](particles.md) | Lagrangian particle tracking, 2D and 3D, lice behaviour | F.2 |
| [viz.md](viz.md) | The Bevy viewer in `viz/` | F.3 |
| [waves.md](waves.md) | The spectral wave model in `src/waves/` and its coupling to the currents | F.4 |
| [2d-numerics.md](2d-numerics.md) | The 2D RHS kernel, well-balanced/entropy-stable/positivity-preserving DGSEM, wet/dry, gating tests | P1.1, P1.2, P1.7 |
| [geometry-mesh.md](geometry-mesh.md) | Coastline-fitted meshes, Gmsh, quadrilaterals, curved elements, the stratified rest state over steep beds | P1.3 |
| [boundaries-nesting.md](boundaries-nesting.md) | Characteristic open boundaries, tidal atlases, NorKyst/ROMS nesting | P1.4, P1.5 |
| [inputs-forcing.md](inputs-forcing.md) | Bathymetry, vertical datum, atmospheric forcing, rivers, shoreline handling | P1.6 |
| [performance.md](performance.md) | Time step, kernels, fused stepper, local time stepping, benchmarks, GPU/MPI | P2 |
| [3d-mode-splitting.md](3d-mode-splitting.md) | Barotropic/baroclinic mode splitting | P4.1 |
| [3d-numerics.md](3d-numerics.md) | 3D continuity, tracers, momentum, PGF and vertical grid, 3D wet/dry, viscosity, kernels | P4.2, P4.3, P4.5 |
| [3d-mixing.md](3d-mixing.md) | GLS, bottom and surface boundary layers | P4.4 |
| [3d-validation.md](3d-validation.md) | 3D test cases (estuary, seamount, lock exchange, …) | P4.6 |
| [operational.md](operational.md) | Rivers, ROMS-compatible output, restarts, the operational pipeline, future research | P5, P7 |
| [tech-debt.md](tech-debt.md) | Legacy integrators, config enums, re-exports, error handling, CI | P6 |
| [correctness-bugs.md](correctness-bugs.md) | The confirmed bugs from the 2026-09-25 review (all fixed except the GPU) | P0 |
| [history.md](history.md) | Items resolved before the 2026-09-25 review, completed phases | — |
| [review-2026-09-25.md](review-2026-09-25.md) | The full-crate review that set the current direction (historical; source comments cite its §s) | — |

Reference documents live in `docs/`: `architecture.md` (module tree, status by layer), `accuracy.md` (verified convergence and conservation), `gmsh-meshes.md`, `stratified-rest-over-steep-beds.md`, `windows-native-deps.md`.

## Conventions

- **`TODO.md` holds open items only**, one line each, starting with the item's ID (`**P1.3**`) under a section whose heading links its notes file. Search the notes file for the ID to find the context. No results, no history.
- **Finishing an item:** delete its line from `TODO.md`, and record the outcome (date, PR, the numbers) under the item's heading in its notes file. Notable changes also go in `CHANGELOG.md`.
- **A new opportunity or follow-up:** a one-line checkbox in `TODO.md` under the matching area; put the measurement, file/function pointer and reasoning in the notes file when it takes more than the line.
- **A new area:** add a file here and a row to the table above.

# A stratified rest state over steep beds in the 3D model

Literature and analysis for TODO P1.3's open blocker: a stratified fluid at
rest does not stay at rest where σ-levels cross the pycnocline inside an
element (cliffs, steep shores). The experiments and their numbers are in
`notes/geometry-mesh.md` (P1.3); this note collects what the literature says, why each cure
tried so far fails, and what a research-grade cure would have to do.
Written 2026-10-04.

## The problem in one paragraph

The 3D PGF is a sum over node pairs (σ-pairs, `solver::rhs::baroclinic`).
It is the exact discrete adjoint of the split-form tracer advection, so the
semi-discrete scheme conserves `KE + Σ gρz H_z` exactly. With the reference
profile subtracted, the force at rest is round-off. Yet round-off grows:
e-folding 15–17 min on the Frøya bed and ≈ 38–45 min in the x–z slice of
`a_pycnocline_over_a_cliff_stays_at_rest` (15 → 300 m inside one 1 km P2
element). A linear profile is stable.

## What the literature says

**The symptom is known, and it is an instability, not only an error.**

- Beckmann & Haidvogel (1993, JPO 23, doi:10.1175/1520-0485(1993)023<1736:NSOFAA>2.0.CO;2) found σ-models unstable over a wider range of parameters once forced. Their z-interpolated "corrected gradient" had far smaller errors but "a much more restrictive range of stable behavior". The constant-depth form here behaves the same way: 1e-2 m/s at 2 h, then NaN.
- Zängl (2012, MWR 140, doi:10.1175/MWR-D-12-00049.1) shows terrain-following PGFs are "prone to numerical instability if the height difference between adjacent grid points is much larger than the vertical layer spacing". That is the slice exactly: the mid levels sit at 2–3 m in the 15 m column and 40–130 m in the 300 m one.
- Haney (1991, JPO 21) states the consistency condition (rx1): a level at one point must lie between the neighbouring levels at the adjacent point. Mellor, Oey & Ezer (1998, JAOT 15) find the errors decay in prognostic runs with zero diffusivity. That held for their finite-difference schemes, and the slices here show it does not hold for an energy-exact non-dissipative one.
- Lemarié et al. (2012, Ocean Modelling 42, doi:10.1016/j.ocemod.2011.11.007) describe the feedback in which mixing along σ turns PGF-driven flow into real horizontal density gradients.

**Exact energy is not enough; a Casimir is needed.**

- Shepherd (1993, Atmosphere-Ocean 31, doi:10.1080/07055900.1993.9649460): "energy conservation alone is insufficient". Rest is stable when energy plus Casimirs (the pseudo-energy) has a minimum there. The non-kinetic part is the available potential energy.
- Holliday & McIntyre (1981, JFM 107, doi:10.1017/S0022112081001742) give the exact APE density. Its small-amplitude form, `½ g ρ′²/|∂_z ρ_r|`, has a weight `1/N²` that blows up in weakly stratified water.
- Ricardo, Duru & Lee (2024, SISC 46, doi:10.1137/24M1638938) report the same failure in thermal shallow water. Energy-bounded schemes "tend to exhibit numerical instability over relatively short time frames". Their cure is to also conserve, or upwind-dissipate, the buoyancy variance (a Casimir).
- Bell, Peixoto & Thuburn (2017, QJRMS 143, doi:10.1002/qj.2950) found energy- and enstrophy-conserving Boussinesq schemes linearly unstable. They prove the fixed versions stable by showing the linearised matrix is Hermitian under a positive weighting. That is the analysis tool for this problem.
- Tadmor (1987, 2003): for a scalar with convex entropy `C`, the entropy-conservative two-point flux is unique. With `C′(ρ) = −g Z_r(ρ)` (the APE Casimir) it is exactly the prototype's pair density `ρ*`. Upwind fluxes are entropy-stable for every convex `C` at once.

**What operational and DG models do.**

- NorKyst-800 (Albretsen et al. 2011, Fisken og Havet 2/2011): bed smoothed to rx0 ≈ 0.34 with a Haney number ≈ 6. They use θs = 8 and Tcline = 10 m, keeping levels near the surface "so less smoothing is necessary", and quote Shchepetkin: Haney < 3 is safe, > 8–10 is "insane".
- Norkyst v3 (Christensen et al. 2026, GMD 19, doi:10.5194/gmd-19-2785-2026): Laplacian smoothing and a 10 m minimum depth. "Volume-conserving smoothing methods … made it more difficult to control pressure gradient errors."
- Thetis (Kärnä et al. 2018, GMD 11, doi:10.5194/gmd-11-4359-2018): a DG σ-like model. A Lax–Friedrichs velocity penalty "is required to stabilize the internal pressure gradient", the baroclinic head is P2, and there is no 3D wetting and drying.
- SCHISM LSC² (Zhang et al. 2015, 2016, Ocean Modelling 85/102): per-node local σ grids with shaved cells, a z-based baroclinic gradient (cubic spline, constant extrapolation at the bed), and no bed smoothing. That claim comes from the SCHISM docs, not checked in the papers.
- Multi-envelope coordinates (Bruciaferri, Shapiro & Wobus 2018, Ocean Dynamics 68; Bruciaferri et al. 2024, JAMES 16; Wise et al. 2022, Ocean Modelling 170): levels follow smoothed "virtual bottoms", and the bed itself is not smoothed.
- ROMS wetting and drying (Warner et al. 2013, C&G 58): fluxes into dry cells are blocked, and the baroclinic PGF is masked on the same faces (`umask_wet` in `prsgrd3*.h`). The mass flux and the force are switched off together.

## Analysis: why each cure tried fails

**1. With pairs by level index, a stable rest state forces the pair density's stiffness.** Linearise about rest. The perturbation transport `δQ` advects the background, and the force responds to `δρ` through the pair's density `ρ̂`. The energy identity pairs the force with the advection for any symmetric `ρ̂`. A positive quadratic pseudo-energy `KE + ½ Σ m_i w_i δρ_i²` is then conserved only if each node's weight times the pair's background advection matches the pair's `∂ρ̂/∂ρ_i`. The vertical pairs of a resolved column pin `w_i` to `g/G_i`, the APE weight (`G = |∂_z ρ_r|`). For a straddling pair this forces `∂ρ̂/∂ρ_i ≈ (ρ_i − ρ̂)/(G_i Δz) ≈ chord/(2 G_i)`, which is what `ρ*` does.

Making the flux entropy-stable instead of entropy-conservative does not avoid this. At rest `Q = 0`, a flux continuous in `Q` must equal `ρ*` for both flow directions, so its linearisation about rest is `ρ*`'s. Only a flux discontinuous in `Q` (upwind) escapes, and at the PGF it acts as Coulomb friction of magnitude `g Δρ Δz/(2ρ₀Δx)`, ≈ 3e-3 m/s² on the slice's cliff.

The stiffness is large in the slice. For each σ-level, the ratio of the pair's chord to the stratification at the two nodes (`z` at the 15 m and 300 m columns):

| level | z (15 m) | z (300 m) | chord/G shallow | chord/G deep |
|---|---|---|---|---|
| 0 (bed) | −13.4 | −268 | 0.03 | 14.5 |
| 4 | −5.3 | −105 | 2.5 | 51 |
| 9 | −1.7 | −34 | 37 | 131 |
| 12 | −0.84 | −16.9 | 72 | 0.4 |
| 16 | −0.26 | −5.2 | 3.0 | 0.4 |
| 19 (top) | −0.03 | −0.56 | 1.1 | 0.9 |

Half the levels have a ratio above 10 at one node. The regularised reference (a floor on `G`) caps the ratio by making the Casimir inexact where the real water is weaker stratified than the floor. That is why 0.05 kg/m⁴ brought back the old growth at element 335.

**2. Diffusion of the anomaly about the reference can feed the APE.** A tracer tendency `T` changes the pseudo-energy at the rate `g Σ m (ρ′/G) T`. For `T = κ L ρ′` with `L` a conservative Laplacian this is not sign-definite when `G` differs across a pair. With two nodes at `G` = 0.25 and 4e-4 kg/m⁴, `sym(G⁻¹L)` has eigenvalues −3020 and +516. Hence "slows the growth, does not stop it". Diffusing the displacement `Z_r(ρ) − z` instead is provably dissipative, conservative and zero at rest, but it carries the same `1/G` weight and so the same stiffness.

**3. Stretching cannot remove the straddles at a shore.** Every level of a column shallower than the pycnocline pairs with levels of a deeper neighbour in the same element, under any σ-like map with the same level count per column. ROMS-style `hc` hybrids (unused in `SigmaGrid` today) and multi-envelope levels reduce the interior straddles only. Only pairing by depth (z-levels, shaved cells, LSC²) avoids them. Under mode splitting that cannot block the deep layers' flow toward the shallow node either: the layer transports must sum to the 2D pair flux, which knows no cliff.

**4. Shoreline elements leak through the wet–dry pairs.** See the experiments below. A dry node holds no volume, so the wet node's split-form exchange with it ends up in the wet column's own Ω. Its pressure work is exact only for linear profiles, and no PGF term on the horizontal pair can be its partner. ROMS's cure is to block the flux and the force together, which here would reach into the 2D `WetDry` mass flux as well.

## Experiments, 2026-10-04

The pair density (`exp/pair-density-wet-dry`, `DGRS_EC=1`) at time steps small enough for its stiffness, with GLS on the Frøya 1 km grid (20 levels):

- `thin=2`, 8 s step, GLS and the limiter: 3.4e-3 m/s at 0.26 h, 0.32 m/s at 0.5 h.
- The same without the limiter: 0.07 m/s at 0.38 h, 0.70 m/s at 0.63 h.
- Both grow in shoreline elements with dry nodes beside 80–135 m ones (553, 393: depths 3.8, 2.5, 0.2, 16.7, 0.5, 0.0, 80, 135, 82 m). The step was not the problem: constant mixing at 36 s held 6 h before the same elements grew.
- Even at the start, the limiter changes the rest state's force from 1e-9 to 3e-6 m/s².
- `thin=0.1`, 4 s step, GLS and the limiter: 4.2e-3 m/s at 0.19 h, 0.16 m/s at 0.31 h, again in a shoreline element with dry nodes (77: depths 0.0, 0.2, 0.0, 3.3, 0.0, 0.0, 13.4, 43, 1.3 m).
- So with GLS the pair density makes the shorelines far worse than the unchanged scheme, which reaches 6e-4 m/s at 0.7 h and grows first at a fully wet cliff (element 335). It cures fully wet cliffs, but not the bed Frøya has.

A pressure-work partner for wet–dry pairs (`DGRS_DRYBANK=1`: the σ-pairs difference with the wet column's density, matching the wet-side rule) makes the cliff-to-land slice worse. The e-folding drops from ≈ 5.8 h to ≈ 3.1 h (7.4e-11 → 1.7e-9 m/s over 6–24 h against 1.8e-10 → 5.8e-8).

## What a research-grade cure needs

Three independent pieces, none sufficient alone:

1. **A stability proof tool.** Linearise the slice's step map about rest (finite differences, ≈ 6000 unknowns) and take its spectrum, as Bell, Peixoto & Thuburn do. This gives every unstable mode, its growth and its location, and the stiffness (largest frequency) of a candidate, in seconds instead of 6 h runs.
2. **Wet–dry pairs consistent in 2D and 3D.** Block the wet node's exchange with a dry node in the layer continuity, the tracer advection and the PGF together (ROMS `umask_wet`). The 2D `WetDry` mass flux must block the same pairs, or the 3D layers cannot sum to it. This is the blocker on Frøya, at any time step.
3. **The pair density with its stiff part implicit.** The stiff coupling is local to one element and one level (the pair term of the force and of the advection): for P2, 9 nodes × (u, v, ρ) per element and level. That makes a linearly implicit (Rosenbrock/W-method or IMEX) stage solve of a 27 × 27 system per element-level feasible. It needs a reference profile, though, so it does not carry over to evolving stratifications (rivers, nesting, GLS mixing layers) without updating the Casimir.

The alternatives that avoid the problem do so by geometry: z-pairs or shaved cells as in SCHISM's LSC², or envelopes. They are a larger change here because the 2D/3D mode splitting pairs layer transports with the barotropic pair flux.

## Follow-up 2026-10-04: wetting and drying made consistent, and smoothing

Piece 2 is done, and it did not need the 2D kernel to change. Its wet/dry subcells already block a dry bank: Audusse et al.'s hydrostatic reconstruction gives `h* = 0` there. The 3D layers of those elements ignored that and moved their volume with the DG pairs. Now:

- The 2D kernel reports the mass flux through every subcell interface, and the mode splitter averages it like the face fluxes.
- Elements that took the subcells throughout the pass move their layers through the same interfaces (`Δσ_l F̄` plus the central baroclinic part between columns that are not thin), so the layers sum to the 2D update node by node.
- The pressure gradient's volume term in such an element is the subcell interfaces' central difference, the adjoint of that divergence.
- A thin column exchanges no baroclinic transport across faces either.
- 2D takes its subcells in exactly the elements where 3D has a thin column.

Which form the advection takes on the subcell interfaces matters. Upwind moved a steep shore's linear stratification by 8e-3 °C, the diapycnal mixing of diffusion along steep σ-levels (Marchesiello et al. 2009). Central keeps it to 5e-4 °C and is the PGF's energy partner.

With the shores consistent, the bed smoothing needs no shore rule. Pairs of columns that are not thin are bounded per element (r_x0 ≤ 0.2 below 1.5 × the pycnocline's bottom), volume-preserving, and the coastline does not move. Slices hold for a day at ≈ 2e-10 m/s: the 15 → 300 m cliff, 0.5 m beside 150 m, and land beside 300 m. Before, land beside 300 m reached 1.3e-3 m/s in 48 h unsmoothed.

Frøya at rest (20 levels, GLS and the limiter): on the 1 km grid, constant mixing without the limiter holds 5.8e-10 m/s for 24 h. With GLS and the limiter the speed creeps up, to 1.1e-3 m/s at 8 h and 1.1e-2 at 24 h, at the bed of shallow elements: mixing at the slopes, as the control shows. The coastline mesh with GLS: about 1e-3 m/s per hour for 1.5 h, then 8e-3 m/s at 2 h in a shore element at a wetting front. Before, it reached 0.16 m/s within 15 min.

The smoothing is still heavy on the coastline mesh: 23k of ≈ 50k nodes change, by up to 210 m. The ratio bound chains outward from every steep coast element, about a kilometre at 200 m elements. Pieces 1 and 3 (a spectrum harness, and an implicit pair density) remain the route to a lighter bound.

### Follow-up: baroclinic exchange inside shore elements

Longer runs on the coastline mesh (constant mixing, no limiter) showed a slower instability. Shore element 8722 grew with an e-folding of 38 min from 3.7 h on. It holds 28.5 and 16.6 m columns beside 2.9 and 0.3 m ones across a line of dry nodes. The baroclinic exchange through its subcells was energy-exact, yet it grew; the Casimir argument above has no reason to hold on this stencil either. A smaller thin depth or free depth, or upwind tracers, did not stop it. Shore elements now carry only their layers' share of the barotropic flux and feel no pressure gradient within themselves, which is the ROMS wet–dry mask in DG form. With that, a patch around the element holds 9e-10 m/s for 8.8 h. Their faces still couple them fully to wet neighbours.

## The 36-min mode: the vertical reconstruction of the background (2026-10-04)

After the r_x0 0.15 smoothing, the coastline mesh still grew from ≈ 4.5 h on, with an e-folding of ≈ 36 min, in the bed layers of near-shore terraces (TODO P1.3). Every earlier fix moved this mode without removing it. It is not the straddle mechanism above. It comes from how the vertical tracer advection reconstructs the *background* stratification at the σ-surfaces.

### Harness

`froya_real_data … debug_3d=…,dump=PREFIX` writes the state at rest, `gap=` steps before the end, and at the end (`PREFIX.{init,prev,end}.bin`, with `PREFIX.geom.bin`). `seed=PREFIX:AMP` starts a run from rest plus the grown mode, scaled to a largest speed of AMP. On the 95-element patch `around=11048:1500` a seeded test takes 39 s of wall time for 2 model hours. It shows the e-folding from the first step, instead of after 4.5 h of round-off. `deep=` sets the gradient below the pycnocline and `vadv=` the vertical tracer scheme. `scripts/mode3d_dump.py PREFIX` reads the dumps.

### The mode

- It is real: dumps an hour apart correlate to 0.997 in u, v, T and S, with a growth of 5.7× per hour (e-folding 34.5 min) and no oscillation. Neither the time step (half the step: 34.6 min) nor the stratification below the pycnocline (0.001, 0.002 or 0.004 °C/m: 34.4–34.8 min) changes the rate. It is a property of the semi-discrete scheme and of the pycnocline.
- It sits at one vertex on a terrace edge (30.6 m beside 22.6–23.3 m, r_x0 0.14) shared by shore element 91, which has one dry node and so is a subcell element, and its wet neighbours 90 and 92. The three elements' velocities at the vertex differ in direction: the mode is at the grid scale.
- It is largest in the bed layers 0–2 (z −27 to −13 m). In the 23 m columns these cross the pycnocline's lower flank (11–19 m). δρ alternates in sign from level to level from the bed up to ≈ −8 m.

### What sets it: the background's surface values

Seeded with the mode, 2 model hours, constant mixing and no limiter:

| vertical tracer scheme | e-folding | KE after 2 h (from 1.5e-9) |
|---|---|---|
| upwind | 26–36 min | 7.6e-6 |
| TVD | 29 min | 4.0e-6 |
| LimitedAkima (the default) | 34.6 min | 1.5e-6 |
| Akima | ≈ 50 min | 1.4e-7 |
| Hermite mean (the PGF's energy partner) | ≈ 75 min (transient, see the spectrum below) | 3.0e-8 |
| centred | decays, then ≥ 10 h | 5.0e-9 |
| LimitedAkima, TVD or upwind, about the rest state (`vref`) | as centred | 5.0e-9 |

From round-off, centred and the Hermite mean show no exponential mode in 8.8 h on the patch (1e-9 m/s, the noise floor), where LimitedAkima reaches 3.5e-6 m/s. The Hermite mean's growth when seeded is transient: the seed is LimitedAkima's mode, not its own, and its spectrum (below) has nothing faster than centred's.

**Why.** At rest `Ω̄ = 0`. To first order a scheme then acts only through the surface value `C̄*` it gives the *background* at each σ-surface, multiplied by the perturbation's `δΩ`. How the perturbation itself is reconstructed is second order. Centred takes the arithmetic mean of the two layers. Upwind takes the upwind layer's value, which is the mean plus `½|δΩ|ΔC̄`: it mixes the background stratification at a rate set by the perturbation's own vertical flow. The mixing makes a density anomaly, its pressure gradient drives flow, and that flow's `|Ω|` mixes more. In the continuum the mixing does not depend on the flow it causes, so this feedback has no physical counterpart. TVD and LimitedAkima clip where the profile curves, at the pycnocline's foot, and so act partly the same way. Akima is linear and non-dissipative, but it moves the surface value off the mean by a curvature term, `h(d_{l−1} − d_l)/6` on uniform levels, and grows as an oscillatory pair.

**Energy is not the edge, and there is no band (corrected 2026-10-10).** The exact vertical partner of the PGF's Hermite pressure integral is the Hermite mean, `½(C_{l−1} + C_l) + h(d_{l−1} − d_l)/12`: the mean of the pressure integral's cubic between the two layer centres. On uniform levels it is centred plus half of Akima's curvature correction. A first sweep of that correction's weight `c` (0 centred, ½ the Hermite mean, 1 Akima) seemed to show a stable band from 0 to 0.75. Its Ritz values were not converged (see the spectrum below). Converged, every `c` grows on the fixture, through two branches that cross near c ≈ 0.3:

| c | −1 | −0.5 | −0.25 | 0 | 0.25 | 0.5 | 0.75 | 1 | 1.5 |
|---|---|---|---|---|---|---|---|---|---|
| e-folding (min) | 45 | 60 | 78 | 116 | 158 | 91 | 65 | 52 | 39 |
| type | pair | pair | pair | pair | pair | real | real | real | real |

The real branch (bed levels 0–2 of the terrace element and the shore element beside it) is this section's mechanism, fed by the curvature term. The oscillatory branch (levels 2–4 of the terrace element and its deeper neighbour, period ≈ 3 h at c = 0) is damped as c rises but is there at every c tried. A flat column with the same levels, pycnocline, pressure integral and surface values (`scripts/column_modes.py`) is neutral for every c in [−1, 1.5]: its internal modes stay real and distinct, so the modal energy is a conserved norm at every c. A fully wet slice with the same terrace inside one element is stable at c = 0 and 1. So the weighted norm the earlier note asked for exists wherever the scheme acts alone. The growth needs the terrace beside a shore element, and its mechanism is open (TODO P1.3). Upwind and the limiters fail differently, through `|δΩ|`.

**Reconstruction about a reference: `Hydrostatic3D::with_vertical_reference`.** It stores T and S per node and level, at rest (`with_vertical_reference(state)`) or from a z-profile sampled at the levels (`with_vertical_reference_profile`). At every σ-surface the scheme's value is corrected by the share `ψ` of the reference's own departure from the centred value: `C_s = V(C)_s − ψ[V(C_ref)_s − ½(C_ref,l−1 + C_ref,l)]`, with `ψ = clamp(ΔC/ΔC_ref, 0, 1)`. The correction moves the value towards the column's centred one, never past it.
- **At rest** `ψ = 1`, so the background's flux is centred to first order.
- **Constancy:** a uniform tracer has `ψ = 0`, so it stays uniform whatever the reference. The first prototype (`centred(C_ref) + V(C − C_ref)`) did not keep it: in a mixed column under a stratified reference, upwind restratified it by `½|Ω|ΔC_ref`, ≈ 0.1 °C/h at tidal Ω.
- **Weakened stratification:** a column stratified like the reference at a fraction of its strength (`C = a + bC_ref`, 0 < b ≤ 1) stays exactly centred.
- **Fronts:** jumps beyond the reference's keep the limiter.
- **Moving references:** `set_vertical_reference` refreshes it, and `with_vertical_reference_timescale(τ)` relaxes it towards the state at every step of `Simulation3D`.

On the patch, upwind, TVD, Akima and LimitedAkima about the rest state all give centred's numbers, which removes the 36-min mode but not the slower oscillatory one (≈ 2 h, below). The gates are `a_terrace_beside_a_shore_element_stays_at_rest_about_its_reference` and the transport unit tests (centred at the reference, constancy, the scaled stratification, a bounded front).

### The spectrum

`probe_terrace_spectrum` (ignored test) runs Arnoldi on finite differences of the step map, about rest on a 22-element fixture: the elements within 400 m of 11048, which hold the mode on their own (`tests/data/froya_terrace_patch.txt`, 16k unknowns). It reports the leading Ritz values with their relative residuals and where the leading modes live. For upwind and the limited schemes the map is only positively homogeneous (`|Ω|`), so their values describe one linearisation and are indicative.

The first table here used a 600 s map and 24 vectors. Its residuals (printed since 2026-10-10) were 0.2–0.4: over 10 min the weakly damped internal waves crowd the unit circle, and the leading modes were missed. That table gave centred 8.8 h, the Hermite mean 7.2 h, and LimitedAkima about the rest state 8.8 h. From an 1800 s map the rates agree across horizons (600 s with 60 vectors, 1800 s, 3600 s) and with time integration of the seeded fixture. The defaults are now 1800 s and 30 vectors:

| vertical tracer scheme | leading growth (e-folding) |
|---|---|
| LimitedAkima | 33 min (real; on the 19 m free-depth fixture) |
| Akima weight 1 (`hermite`, `DGRS_WEIGHT=1`) | 52 min (real) |
| Hermite mean | 91 min (real), then real modes at 100 and 137 min |
| centred | 116 min (pair, period 3.2 h) |
| LimitedAkima about the rest state | 115 min (pair, as centred); 142 min on the 19 m free-depth fixture |

The integrated fixture agrees: about the reference it holds at the seed's transient (≈ 4e-7 m/s) for ≈ 5 h, then grows to 1.3e-5 m/s at 12 h. The gate `a_terrace_beside_a_shore_element_stays_at_rest_about_its_reference` ends at 3 h. Its 12 h companion is an ignored known failure. The same vertex grows on the 95-element patch of the coastline mesh when seeded (`froya_real_data … debug_3d=…,perturb=1e-6`), so it is not the fixture's walls. From round-off a 2 h mode needs a day to show, which is why the 8.8 h round-off runs saw nothing.

### Two mechanisms, not one

The unsmoothed 15 → 300 m cliff (`a_pycnocline_over_a_cliff_stays_at_rest`, 6 h) barely depends on the vertical scheme: 2.9e-8 m/s with LimitedAkima, 2.7e-8 with centred or `vref`, 2.8e-8 with Akima, 3.3e-8 with the Hermite mean, 4.0e-8 with TVD and 4.7e-8 with upwind. That is the straddle mechanism of the analysis above, which the per-element smoothing bounds. The 36-min mode of the smoothed coastline mesh is the vertical one. The 2D shore probes (islet, spit, corner) stay at round-off with every scheme (1e-11 m/s, e-folding ≥ 7 h at noise level). What makes the terrace vertex beside a subcell element the place where it grows is still open.

### The r_x0 0.2 failure was the same mode

The bound was tightened from 0.2 to 0.15 because element 12097 of the coastline mesh grew at 0.2. On the patch `around=12097:1500 slopes3d=0.2` (constant mixing, no limiter), the default grows from ≈ 5 h with an e-folding of 41 min, to 5.2e-5 m/s at 12.6 h. The mode has the same shape: shore element 103 (22.9 m beside 33.7 m, one dry node) and its neighbours 104 (12097) and 102, with dumps correlating to 0.9996. With `vref` the patch holds 7.6e-10 m/s for 12.6 h. With the reconstruction about a reference, the smoothing could be lighter.

### The full coastline mesh

With `vref` (r_x0 0.15, constant mixing, no limiter), the whole mesh stays at round-off (≤ 3e-9 m/s, in deep water) until ≈ 7 h. Without it the 36-min mode grows from ≈ 4.5 h, to 1.4e-6 m/s at 8.2 h. After 7 h a much slower growth appears: 3.3e-8 m/s at 8.6 h, oscillating (velocities an hour apart correlate 0.24), with density growing ×1.2 per hour. It sits in element 9381, which is fully wet but has 28.5 m beside 2.6 m inside it. The free-depth rule leaves that pair unsmoothed, because the deeper column is not below 28.5 m. It is probably the straddle mechanism, and it is open.

### The bound with the reference, and a trade-off

On the full mesh with the reference (constant mixing, no limiter, 8.6 h):
- **r_x0 0.15**: 3.3e-8 m/s, at element 9381.
- **r_x0 0.2**: 1.9e-8 m/s, at fully wet element 11163 (35–53 m, exactly at the bound, e-folding ≈ 55 min from 6.8 h) and at exempted shore elements.

On the patch `around=11163:1500 slopes3d=0.2` the default holds 8.6e-10 m/s for 9.1 h, but the reference grows with e-folding ≈ 59 min. So the vertical scheme's dissipation cuts both ways. LimitedAkima feeds the terrace mode beside shore elements and damps the straddle mode of fully wet elements at the bound; centred-type advection of the background does the opposite. At r_x0 0.15 the reference was strictly better. A lighter bound needs a cure for the straddle mode itself (the pair density with its stiff part implicit, piece 3 above, or a bound on the Haney number).

The 9381 patch (`around=9381:1500`, r_x0 0.15, the free-depth exemption: 28.5 m beside 2.6 m in one fully wet element) grows with e-folding ≈ 43 min from ≈ 7 h, with or without the reference, and saturates near 4e-2 m/s at 20 h: the straddle mechanism. The 1D calibration slices let shallow-to-28 m steps go unsmoothed. In 2D they do not hold. With `free_depth=19` (the pycnocline's bottom) the patch holds below 9e-10 m/s for 20.6 h, with or without the reference, at the cost of 32.1k smoothed nodes of the coastline mesh instead of 26.2k. On the full mesh with the reference (constant mixing, no limiter), `free_depth=19` holds ≤ 3.7e-9 m/s through 10 h and 8.4e-9 m/s at 11.7 h. The only rise is a faint one at a 19 m column beside 0.2–8.6 m ones, now at the new free depth. That is the best rest state of the coastline mesh so far. With GLS and the limiter it ends at 2.8e-3 m/s after 8.6 h, against 2.6e-2 m/s with the 28.5 m free depth, and it levels off (2.0e-3 m/s at 4 h, 2.7e-3 at 8.2 h). What is left is spread over the bed layers of shallow elements: the GLS creep at slopes, with no exponential growth.

The 2D tide on the Mausund sub-domain with the 3D bed (hours 24–72):

| bed | centred RMSE vs prediction / observations | M2 phase |
|---|---|---|
| unsmoothed | 4.1 / 6.3 cm | −8.7° |
| free depth 28.5 m | 2.5 / 4.4 cm | −12.3° |
| free depth 19 m | 3.1 / 4.5 cm | −13.9° |

The time series improve and the 2-day M2 phase worsens. At NorKyst's point near Mausund the M2 current rises towards NorKyst's (0.014 → 0.067 → 0.102 m/s, against 0.246): the smoothing opens part of the island lee.

### With GLS and the limiter

On the full coastline mesh with the reference (r_x0 0.15, 8.6 h), the largest speed is 1.9e-3 m/s at 3–4 h. Before the reference, this setup reached 8e-3 m/s at 2 h. It then grows slowly (e-folding ≈ 2.5 h) to 2.6e-2 m/s, at elements with 27–28.5 m beside 2–9 m: the shallow-deep pairs the free depth exempts.


On the 11048 patch from rest, 8.8 h:
- **Default**: 2.2e-2 m/s, in elements 90–92 (the mode).
- **`vref`**: 1.4e-3 m/s, still rising slowly and close to linearly, in shallow shore elements 13 and 63 (5–28 m). That is the GLS creep at the slopes.
- **Centred**: 1.5e-3 m/s, in the same elements as `vref`.

## The cure: a reference stratification, advected by the pressure gradient's adjoint (2026-11-20)

The rest state is now stable by construction, on beds that are not smoothed.
`Hydrostatic3D::with_reference_profile` takes a horizontally uniform `T̄(z)`,
`S̄(z)`; the baroclinic pressure gradient is that of the density anomaly about
it, and the tracers' advection moves the reference by the vertical velocity
that is that force's exact adjoint. The scheme linearised about rest in the
reference then conserves a positive pseudo-energy (kinetic energy plus the
available potential energy `½ g ρ′²/Γ`), with the stratification's own weights
at every node, and every element keeps its tracer inventories. The price is
in the time step: the scheme carries in full the frequency `N |∇_σ z|` of a
stratification displaced along a sloping level, and the step must resolve it
(a bound of `compute_dt` now; it binds on the exaggerated cliff slice, not on
the fixture cut from the Frøya bed). Code: `solver::rhs::baroclinic` ("About
a reference"), `ReferenceFields`, `add_reference_advection`. This section is
the derivation, the routes that were ruled out on the way, and the
measurements.

### The layers as a graph

Write `a = (i, l)` for node `i` of an element on level `l`, `m_a = w_i J_i`
for its horizontal mass, `H_a = D_i Δσ_l` for its layer thickness and
`z_a = η_i + σ_l D_i` for its height. The layers move their volume along the
edges of a graph, each with a volume flux `F_ab = −F_ba` (m³/s) from `a` to
`b`:

- **pairs**: two nodes `i ≠ j` of an element on one level, with
  `Φ_ij = m_i Dx_ij·Q_i − m_j Dx_ji·Q_j`, where `Q = H u` (corrected to the
  barotropic transport) and `Dx_ij` are the element's derivative coefficients
  with the pair's metric (on a parallelogram `Φ_ij = 2 m_i Dx_ij·{{Q}}_ij`,
  the split form's two-point flux; nonzero only for nodes on one GLL line);
- **σ-surfaces**: between the layers `l − 1` and `l` of a column, `m Ω`;
- **faces**: between the two sides of an element face, the layer's face flux;
- **subcell interfaces**: in a shore element (one with a thin column), the
  layer's share of the barotropic flux.

The layer continuity is `m_a ∂H_a/∂t = −Σ_b F_ab`, and the conservative tracer
advection, in advective form,

```text
    m_a H_a ∂C_a/∂t = −Σ_b F_ab (Ĉ_ab − C_a),
```

with the edge value `Ĉ_ab = ½(C_a + C_b)` on pairs and subcell interfaces
(the split form), the vertical scheme's surface value on σ-surfaces (the mean
of the two layers for the reference's share, since 2026-10-04), and the
upwind value on faces.

### Why the σ-pairs scheme is not stable: the chord

At rest in a profile `C̄(z)` the two sides of a face are at one height (the
bed is continuous), so a perturbation's flux `δF` advects the background only
along pairs and σ-surfaces:

```text
    m_a H_a ∂δC_a/∂t = −½ Σ_b δF_ab [C̄(z_b) − C̄(z_a)].          (chord)
```

The σ-pairs pressure gradient, summed against the transports and by parts
with the continuity (the pressure differences of the pairs and faces become
those across the σ-surfaces, and with the trapezoid rule
`p_l − p_{l+1} = ½(ρ′_l + ρ′_{l+1})(z_{l+1} − z_l)`), does the work

```text
    −(ρ₀/g) Σ_a m_a Q_a·F_a = Σ_edges F_ab ½(ρ′_a + ρ′_b)(z_b − z_a) =: b(F, ρ′)   (work)
```

over the pairs and the σ-surfaces. So the force responds to `ρ′_a` through
the *height* differences `z_b − z_a` of the node's edges, and the advection
makes `ρ′_a` from the *density* differences `ρ̄_b − ρ̄_a` along the same edges.
The operator `(δu, δρ) ↦ (−(g/ρ₀) B δρ, −A δu)` is skew-adjoint in a norm
`½ρ₀ Σ m H |δu|² + ½ Σ s_a δρ_a²` only if `s_a (ρ̄_b − ρ̄_a) = −g m_a H_a (z_b − z_a)`
for *every* edge of `a` at once: the chords of all the node's edges must have
one slope. That holds for a profile linear in `z` (the variance argument of
2026-10-03) and fails wherever an edge straddles the pycnocline. Without it
the spectrum of `A B` is not real and non-negative, and pairs of complex
eigenvalues appear: the modes of the cliff and of the terrace.

### The scheme

Five changes, all on exactly the edges the force acts along: the pairs of an
element without a thin column that moves its layers by the pairs, the subcell
interfaces between two columns that are not thin in an element with a thin
one (step 5), and the σ-surfaces of the columns that are not thin.

1. **The force is the anomaly's, and linear.** `ρ′_a = ρ_a − ρ̄(z_a)` with
   `ρ̄(z) = ρ(T̄(z), S̄(z))` at the node's own height at the stage's `η`, and
   the trapezoid rule for its pressure (the monotone cubic's harmonic-mean
   slopes are not linear in the density, so that force has no adjoint). At
   rest in the reference the force is zero, over any bed.
2. **The reference is advected by the tangent.** To the conservative
   advection the kernel adds, per edge and at both its ends,

   ```text
       ½ F_ab θ_ab [(C̄_b − C̄_a) − γ_a (z_b − z_a)] / m_a,
       γ_a = [C̄(z_{a+½}) − C̄(z_{a−½})] / H_a,
   ```

   `γ_a` the reference's mean gradient over the layer of `a` (between the
   σ-surfaces that bound it, so a column's gradients add up to the
   reference's change over it whatever its levels resolve),
   `θ_ab = clamp(ΔC_ab/ΔC̄_ab, 0, 1)` the share of the reference's jump that
   the tracer's own jump along the edge holds.
3. **The element's mean displacement.** The two ends' terms of an edge do not
   cancel: they add up to `F_ab τ_ab`,
   `τ_ab = (C̄_b − C̄_a) − ½(γ_a + γ_b)(z_b − z_a)`, the trapezoid rule's error
   of the reference's gradient along the edge. With `R = Σ F_ab τ_ab` over the
   element's edges and `N = Σ m H Γ` over its stably stratified layers
   (`Γ = −(ρ_T γ^T + ρ_S γ^S)`), every layer of the element gets `−H Γ R/N`:
   the element's stratification displaced as a whole at the rate `R/N`. And
   the force leaves out, along every edge of the element, the trapezoid
   rule's error of the anomaly of the element's mean displacement
   `κ = Σ m H ρ′/N`:

   ```text
       ½(ρ′_a + ρ′_b)(z_b − z_a) − κ [(ρ̄_b − ρ̄_a) + ½(Γ_a + Γ_b)(z_b − z_a)].
   ```

4. **The pressure's mean over a column is the anomaly's first moment about
   the bed.** Transports with a depth mean move the free surface, and a
   layer's centre rises through the reference (which is fixed in `z`) by
   `(1 + σ) ∂η/∂t`: the anomaly gets `Γ (1 + σ) ∂η/∂t` besides the
   advection's. Its partner in the force's work is the pressure's mean over
   the column, times the column's `∂η/∂t` (the proof below). So every
   column's pressure is shifted, by the same amount at every level, to

   ```text
       Σ_l Δσ_l p_l = Σ_l Δσ_l ρ′_l (z_l − B),
   ```

   the mean of the exact pressure of layers of uniform `ρ′` (the layer
   centres are the midpoints of their σ-surfaces, in every stretching of the
   crate). The trapezoid rule's own mean differs from it on stretched levels
   and by step 3's term. Only the depth-mean force changes, by a term of the
   size of the rule's error; a uniform anomaly still exerts none.
5. **An element with a thin column** (a shore element) moves its layers
   through the GLL subcell interfaces of the 2D wet/dry kernel, each with its
   share `Δσ_l F̄` of the barotropic flux, and exchanges no baroclinic
   transport within itself (since 2026-10-04). That flux moves the
   stratification along the element's sloping levels like any other. So an
   interface between two columns that are not thin is an edge of the graph:
   the reference is advected by the tangent along it (step 2, and step 3 for
   the element), and the force gets the adjoint of that, which for a flux
   that is the same share of one barotropic flux at every level is a force
   that is the same at every level: with `m` the interface's metric and `w`
   the line's quadrature weight, `∂p/∂x` at both its nodes gets
   `½ w m Σ_l Δσ_l Δp_l / m_node`, the mean over the column of the pair's
   pressure differences. The barotropic flow in a shore element then feels
   the anomaly it makes. The layers there still get no baroclinic force and
   exchange no baroclinic transport, and an interface with a thin column on
   one side stays outside (the wet/dry front; the 2D kernel's hydrostatic
   reconstruction decides its flux).

In a fluid at rest in the reference `θ = 1`, the chord of (chord) cancels, and
what is left of the advection is

```text
    H_a ∂δC_a/∂t = −γ^C_a (H w)_a − H_a Γ_a R^C/N,
    (H w)_a = (1/m_a) Σ_b ½ δF_ab (z_b − z_a).
```

`w` is the rate at which the layer's water rises relative to its node,
`u·∇_σ z + Ω`: the horizontal part is the split-form derivative of the level's
height along the flow, the vertical part `Ω` averaged to the layer centre with
the trapezoid rule's weights, `½(Ω_{l−½} Δz_{l−½} + Ω_{l+½} Δz_{l+½})/H`. It
is defined by the force: `Σ_a m_a ρ′_a (H w)_a` is the sum (work), term by
term. Step 3 reads the same way: `κ` is the projection of the anomaly on a
uniform displacement of the reference (whose anomaly is `κΓ`) in the
available potential energy's norm, and the force integrates that part of the
anomaly along each edge exactly, as `−κ(ρ̄_b − ρ̄_a)`, instead of by the
trapezoid rule. A stratification displaced as a whole within an element then
exerts no force within it (but for the surface half-layer), where the
trapezoid rule alone leaves the σ form's error of the displaced profile at
every straddle: the well-known limit of subtracting a reference profile, that
the error comes back once the isopycnals move, is gone for the element's mean
displacement.

### The statement and its proof

**Assumptions.** The reference is statically stable: with `ρ_T`, `ρ_S` the
equation of state's partials at the reference at node `a` (the code's density
depends on `T` and `S` alone), `Γ_a = −(ρ_T γ^T_a + ρ_S γ^S_a) ≥ 0`. The
equation of state is linear in `T` and `S`. The bed is continuous across
element faces. The state is at rest in the reference (`u = 0`, `η` flat,
`T = T̄(z_a)`, `S = S̄(z_a)` at every node). The force is the σ-pairs form.

**Claim.** For the semi-discrete scheme linearised about that state, between
walls, with any layer transports (with a depth mean or without; the free
surface follows them by the layers' continuity, and the layer fluxes are
their central ones),

```text
    ρ₀ Σ_a m_a H_a δu_a·(∂δu_a/∂t)_force + g Σ_{Γ_a > 0} m_a H_a (ρ′_a/Γ_a) ∂ρ′_a/∂t = 0,
    ρ′_a = ρ_T δT_a + ρ_S δS_a + Γ_a (1 + σ_a) δη_a :
```

the baroclinic force's work and the rate of the available potential energy
`½ g ρ′²/Γ` cancel, for the anomaly about the reference at the layer's own
height. Where `Γ_a = 0` the anomaly does not change. With the 2D module's own
exchange between the depth-mean flow and `½ρ₀ g η²`, and the viscosities'
dissipation `D`, the pseudo-energy

```text
    ½ρ₀ Σ_a m_a H_a |δu_a|² + ½ρ₀ g Σ_i m_i δη_i² + ½ g Σ_{Γ_a > 0} m_a H_a ρ′_a²/Γ_a
```

changes at `−D ≤ 0` less what the 2D module's Riemann fluxes dissipate. And
every element's inventories of `T` and `S` change only through its faces.

**Proof.** (i) From the scheme,
`m_a H_a ∂δρ_a/∂t = Γ_a [m_a (H w)_a − m_a H_a R/N]`, with
`R = Σ_edges δF_ab τ_ab` and `τ_ab = (ρ̄_b − ρ̄_a) + ½(Γ_a + Γ_b)(z_b − z_a)`
the density's: `δρ` is the same combination of `δT` and `δS` that `Γ` is of
their gradients, and `ρ̄`'s chords of theirs. (ii) The momentum advection is
quadratic in the perturbation, Coriolis does no work node by node, the bottom
drag is quadratic and Smagorinsky's viscosity vanishes with the strain; the
force's work, by (work) with step 3's term, is
`ρ₀ Σ m_a H_a δu_a·∂δu_a/∂t = −g [b(δF, δρ) − κ R] − D`. (iii) The potential
energy's rate is `g Σ_a (δρ_a/Γ_a) m_a H_a ∂δρ_a/∂t = g Σ_a m_a δρ_a (H w)_a − g (Σ_a m_a H_a δρ_a/N) R = g [b(δF, δρ) − κ R]`:
each edge gives `½ F_ab (z_b − z_a)` to `m_a (H w)_a` and
`½ F_ba (z_a − z_b)`, the same number, to `m_b (H w)_b`, and the second sum is
`κ` by its definition. The two rates cancel. (iv) The inventory: the edges'
terms add up to `R` in the element, and step 3 takes `Σ m H Γ R/N = R` back.
(v) A depth mean. The sum (work) is by parts with the layers' continuity,
`Σ_b F_ab = −m_a (Δσ ∂η/∂t + Ω_{a+½} − Ω_{a−½})`: the pressure differences
along the horizontal edges become `Σ_a p_a Σ_b F_ab`, whose `Ω` part is the
vertical edges' share of `(H w)` (and of `R`, for step 3's term in the
pressure), and whose `∂η/∂t` part is `−Σ_i m_i ∂η_i/∂t Σ_l Δσ_l p_l`. The
anomaly's rate has `Γ (1 + σ) ∂η/∂t` more, which gives the potential energy
`g Σ_i m_i ∂η_i/∂t Σ_l Δσ_l ρ′_l (z_l − B_i)`. Step 4 makes the two equal,
column by column. (vi) A subcell interface carries `F_l = Δσ_l w ½ m·(Q̄_a +
Q̄_b)`, `Q̄` the columns' transports; `Σ_l F_l Δp_l` is the work of step 5's
force on them. ∎

The quantity is positive definite on the dynamic variables, with the weights
of the available potential energy of Holliday & McIntyre (1981); the operator
is skew-adjoint in that norm, the route of Bell, Peixoto & Thuburn (2017).
It holds for any mesh, bed, level distribution and reference, in particular
at straddles: `Γ_a` is the node's own stratification, whatever its
neighbours'. No weight is singular: the fastest mode is an internal wave's,
`ω² ≤ (g/ρ₀) max Γ · ‖w‖²`, and the displacement's rate `R/N` is of the size
of `w` itself (at a straddle `τ` is the reference's range and `N` the columns'
area times that range). But `w` has the level's slope in it, and that sets a
frequency of its own: see "The frequency along sloping levels, and the time
step".

The whole exchange is checked to round-off, with the actual kernels, by the
gate `the_reference_advection_and_the_pressure_gradient_exchange_pseudo_energy_exactly`:
random transports, without a depth mean and with one, and random anomalies,
the tracers at rest in a thermocline, on parallelograms and distorted
elements in both metric forms, on stretched levels, over a 5 → 125 m cliff,
across faces, with a shore element with dry nodes. The rate of the potential
energy (from the conservative tracer kernel about the reference plus
`add_reference_advection`, and the layers' rise with the free surface) and
the force's work agree to 1e-11 of their terms, and the inventory's rate is
round-off. Without step 4 the two differ by 8 % of the work for transports
with a depth mean on those levels (3 % on uniform levels: step 3's term), and
without step 5 by another 4 %. In time it is checked
by `the_pseudo_energy_of_a_departure_from_rest_does_not_grow_over_a_cliff`
and by the spectrum probes in pseudo-energy coordinates (below).

### Conservation: what an edge cannot do, and the element can

*With single-valued fluxes between cells paired by level index, exact
conservation of a curved reference, the skew structure and bounded weights
exclude each other.* Let an edge treat its two ends with tangent increments
`t_ab` at `a` and `t_ba` at `b`. The tracer the edge creates is
`F_ab [ (C̄_b − C̄_a) − ½(t_ab − t_ba) ]`, zero only if the two tangents average
to the chord. The skew structure needs `t_ab = γ_a κ_ab` with `κ_ab` the
weight of `ρ′_a` in the force's term of that edge (`κ_ab + κ_ba = z_b − z_a`
for a consistent force). Both hold only for
`κ_ab = (z_b − z_a)(Ḡ_ab − γ_b)/(γ_a − γ_b)`, with `Ḡ_ab` the chord's slope:
between 0 and `z_b − z_a` where the chord's slope lies between the two ends'
gradients (a resolved profile), and unbounded where it does not (both ends in
weakly stratified water, the pycnocline between). That is the linearisation
of Tadmor's entropy-conservative pair density with the Casimir's weights, and
its stiffness (`chord/Γ` up to 100) is the result of 2026-10-04 again, from
the other side.

The way out is that conservation is needed per element, not per edge, and
every edge the force acts along lies within one element. The edges' errors
`F_ab τ_ab` are collected over the element and taken back as one more
vertical velocity, uniform over the element (step 3), which has a bounded
adjoint in the force. Without step 3 the scheme of steps 1 and 2 is stable
too (it was the first prototype), but its tracer inventories drift at the
trapezoid rule's error: under a 0.5 m seiche over a 12 → 4 m slope with a
6 °C thermocline (`SlopingTide`), the basin's mean temperature changed by
4.5e-3, 9.3e-4 and 2.6e-4 °C over one period on 3, 6 and 12 levels (second
order in the layers' thickness), most of it secular. With step 3 the
inventories of `T` and `S` hold to 2e-14 on the same runs
(`a_curved_reference_keeps_constancy_and_the_tracers_inventories`).

### The frequency along sloping levels, and the time step

The exchange between the two energies has a frequency. In the pseudo-energy's
coordinates, `r = ρ′ √(g m H/Γ)` and `v = u √(ρ₀ m H)`, the pairs' part of the
operator linearised about rest is `[[0, −Bᵀ], [B, 0]]` on each level of an
element, with

```text
    B[i; i]  =  ½ N_i Σ_j (z_j − z_i) D_ij              (½ N ∇_σ z at the node)
    B[i; j]  = −½ N_i (z_j − z_i) √(m_j D_j / m_i D_i) D_ji
```

(`N² = (g/ρ₀) Γ`, `D_ij` the pair's derivative coefficients, `D_i` the depth).
Its eigenvalues are `±i` times the singular values of `B`: frequencies
`N |∇_σ z|`, the buoyancy frequency times the slope of the level. The
element's mean displacement (step 3) is an orthogonal projection in these
coordinates, off the direction `r_a ∝ √(m H Γ)`, and does not raise them.

That frequency is the hydrostatic equations' own: water that moves along a
surface of slope `s` through a stratification `N` oscillates at `N s` (the
frequency of the internal waves whose rays run along the slope). Where the
levels resolve the stratification they cross, the scheme of the full density
carried it too. Where a level crosses the pycnocline between two nodes it did
not: the chord gave it `(g/ρ₀) Δρ̄/Δz` in place of `N²` (the reference's change
over the level's whole rise: across the 15 → 300 m cliff 1.06 kg/m³ over
142.5 m against 0.25 kg/m⁴ at the node, 34 times less), and no conserved
quantity. The tangent takes the node's own `N²` for the whole
rise, which is what the node's water does to first order, and what the
adjoint needs.

An explicit step must resolve the frequency: SSP-RK3 is stable on the
imaginary axis for `ω Δt ≤ √3`. `Hydrostatic3D::compute_dt` bounds it, with a
reference only (`TimeStepLimits::sloping_levels`,
`ReferenceFields::sloping_level_rate`; the local steps take it per element):
`ω Δt ≤ 1` with `ω = √(‖B‖₁ ‖B‖∞)`, the row and column sums of `B`, where
`z_j − z_i = (1 + σ)(η_j − η_i) − σ(B_j − B_i)` is bounded over the levels by
each column's largest `N` and largest `N |σ|`. That is `O(nodes²)` per
element and step, and the fields it reads are the first stage's (the step's
cost does not change). In a shore element the edges are step 5's subcell
interfaces and the partner is the depth-mean velocity, `v̄ = ū √(ρ₀ m D)`:
`B[a, l; a] = ¼ N_al √Δσ_l (z_b − z_a)_l w m/m_a` and the same times
`√(m_a D_b/(m_b D_a))` for the other end, bounded in the same way. The gate
`the_sloping_level_rate_bounds_the_pairs_frequency` assembles both `B`s from
the kernels, column by column, on parallelograms and distorted elements in
both metric forms over a 5 → 125 m cliff with a shore element: the bound is
1.22 to 1.54 times the largest singular value on the pairs and 3.0 to 3.1
times on the shore element's interfaces. With the margin of `√3` the model's
step is 2.1 to 2.7 times below the scheme's limit where the pairs' bound sets
it.

So the cure does need a shorter step, exactly where `N |∇_σ z| Δt` was above
the limit and the old scheme did not see it. The numbers are under
"Measurements": none on the fixture cut from the Frøya bed, a factor of 4 on
the cliff slice. What would give the step back is a force and an advection
that do not act along the levels at all (pairs by depth, below), not a
weight: scaling the pair's term of the force by a factor below one, with its
adjoint, keeps the skew structure but leaves a uniform anomaly `ρ′` with a
force `g ρ′ Δz/ρ₀` along every sloping level.

### Where the proof stops

- **A state that is not the reference.** Only the share `θ` of the
  reference's stratification that the tracers hold along an edge is advected
  by the tangent; the rest of their jump is advected conservatively, as
  before. A column at a fraction `b` of the reference's strength is exact
  (`θ = b`); a mixed column gets nothing (constancy, gate
  `the_reference_advection_keeps_constancy_a_linear_reference_and_the_elements_inventories`);
  another shape of profile is in between, and has no proof. The reference is
  fixed in `z`: under a tide the levels move through it, and the anomaly is
  the physical one (differential stretching over a slope makes real
  gradients, which a reference that followed the σ-levels would hide). The
  force's displacement term is linear in the anomaly whatever its size: for
  an anomaly that is not a displacement (a river plume over a cliff) it is a
  term of the size of the σ form's own error there, and vanishes with it
  under refinement.
- **The barotropic mode.** The exchange is exact for the central layer
  fluxes, whatever their depth mean (steps 4 and 5). The depth mean of the
  force reaches the 2D module as the slow forcing, and the layers carry their
  shares of the 2D mass flux. What has no partner is the dissipative part of
  the 2D Riemann flux (at faces and on the subcell interfaces: it moves the
  stratification with a flux proportional to the jump of `η`, not to a
  transport the force acts on), and the interfaces to a thin column. Both
  couple the two modes with a feedback of relative size `Δρ/ρ₀` times the 2D
  dissipation. The step maps below include the barotropic variables.
- **A discontinuous bed.** Across a face whose two sides are at different
  heights the upwind face value advects the background at a rate set by
  `|F|`, and the force's face term has a `ρ′ Δz` part: neither is covered.
  Every bed pipeline of the crate gives a continuous bed.
- **An equation of state that is not linear in T and S.** `Γ` is taken with
  the partials at the reference, and the reference density's chords differ
  from the tracers' at the equation's curvature: the exchange is exact up to
  that.
- **Time.** The 3D stages are SSP-RK3 (dissipative on the imaginary axis
  below `|ω|Δt = √3`), the barotropic pass is filtered and its slow forcing
  extrapolated by AB3. No proof; the step maps below contract, at steps
  within the bound of the section above. Above it the rest state blows up
  within the hour (the cliff at 72 s).
- **The implicit part of the vertical advection** (above an outflow Courant
  number of 0.4) is upwind in the full tracer, as before. It is zero at rest.
- **Limiters, mixing.** A limiter or a diffusion changes the tracers at rest
  only if it acts on the reference itself; give the limiter the same profile.
  GLS mixing changes the stratification, and the state then drifts from the
  reference (see the first item).
- **An element switching between the pairs and the subcells** within a step
  (a column crossing the thin depth): the force decides by the stage's
  depths, the layers by the barotropic pass of the step. At rest they agree.

### Routes ruled out on the way

- **The pair density with its stiff part implicit** (piece 3 of 2026-10-04).
  The stiff couplings are not local to an element and level: the vertical
  pairs are stiff too where 20 levels in 300 m do not resolve an 8 m
  pycnocline (`chord/Γ` ≈ 100 between adjacent levels), and they couple every
  level of a column through `Ω`. The solve would be 9 nodes × levels ×
  (u, v, T, S) per element and stage, with the reference's inverse map in it.
  The tangent scheme is that scheme with the stiff remainder (the chord minus
  the tangent, and its partner in the force) left out of the edges and its
  total kept by the element.
- **A weaker force at straddles** (the conservative chords kept, the force
  made their adjoint with node weights `g/γ̂_a`, `γ̂_a` the steepest chord at
  the node). Skew and conservative, but the internal waves of every column
  beside a cliff slow by `√(Γ/γ̂)`, up to 7×: the physics pays, not a
  truncation error.
- **Pairs by depth** (z-pairs, shaved cells, LSC²). A pair at one height
  advects no background at all, so nothing needs repair. It replaces the
  layer graph (a pair's flux lands in another level of the neighbouring
  column, and a deep layer beside a shallow column has no partner), the mode
  splitter's shares and the 2D/3D identity with it. Not needed for rest; it
  would also remove the σ form's error of the anomalies that are not a
  displacement (Parked, P1.3).
- **The redistribution with other weights.** Taking the element's error back
  in proportion to each tracer's own gradient makes the force's partner
  divide by `Γ` (stiff where `T` and `S` compensate in density); in
  proportion to `H Γ` for both tracers it is the uniform displacement above,
  and its partner is bounded.
- **Cancelling the reference's advection where the force has no term** (the
  first prototype did, on shore elements' subcell interfaces). It gives up
  conservation to first order under a tide, and the 30-vector spectra were
  the same with and without (terrace 0.9977 against 0.9979, shore slice
  0.9974 against 0.9973). What those interfaces needed was the force's
  partner (step 5).
- **A weight below one on the pair's term of the force, with its adjoint in
  the advection**, to take the frequency `N |∇_σ z|` out of the levels that
  do not resolve the stratification they cross. It keeps the skew structure,
  but the pair's term and the pressure difference cancel for a uniform
  anomaly only at weight one: a uniform `ρ′` of 1 kg/m³ across a level that
  rises 142 m would get a force of `g ρ′ Δz/ρ₀` times the weight's defect,
  1 m/s in minutes. A window-mean gradient at the node (so that the tangent
  never exceeds the reference's range over the pair) makes the node's
  vertical advection of the reference too weak by the same factor instead.

### What the probes showed on the way to steps 4 and 5

Steps 1 to 3 were the scheme of the first day, with a proof for transports
without a depth mean. Its 30-vector spectra contracted (terrace 0.99795 for
the pseudo-energy norm over the half hour). More Krylov vectors did not: on
the terrace the leading Ritz value was 1.00099 at 60 vectors and 1.01272 at
100 (real, residual 5e-2), in the bed layers' tracers of the shore elements
and in the barotropic fields, and the pseudo-energy of a random departure,
0.1477 of its start after 96 h, was 0.1507 after 144 h. Three things were in
that, and the harness and the scheme have each changed for them.

- **The conserved totals have neutral directions.** A departure that changes
  the volume, the heat or the salt ends in another state at rest, an
  eigenvalue of exactly 1. Arnoldi approaches it slowly and from both sides,
  because its tracers settle over days: on the day map of the fully wet
  cliff slice the leading Ritz value was 1.040 (residual 2.8e-2), and its
  vector's barotropic part was 99.9 % a uniform `η` over all eight elements.
  The step map keeps the three totals, so the departures that change none of
  them are an invariant subspace. The probes now take the map on it (a
  uniform `η`, `T` and `S` are taken out of every departure before it is
  advanced; `DGRS_DEFLATE=0` for the whole space), and the same day map gives
  1.0003 for the norm and 0.9995 for the leading Ritz value.
- **The anomaly is about the reference at the layer's own height**, which
  rises with the free surface: `ρ′ = ρ_T δT + ρ_S δS + Γ (1 + σ) δη`. The
  probe's coordinates had the first two terms only (`EnergyCoordinates`).
- **The identity with a depth mean.** With the gate's transports left with
  their depth mean, and the layers' rise in the anomaly's rate, the
  potential energy's rate and the force's work differed by 8 % of the work
  (3 % on uniform levels), and by 4 % more with the shore element in the
  mesh. The two terms of the first are in the proof's part (v): the
  trapezoid pressure's mean over a column is not the anomaly's first moment
  on stretched levels, and step 3's term in the pressure has a mean with no
  partner. The second is the barotropic flux on a shore element's subcell
  interfaces, which moved the stratification along the levels with the
  chord and with no force answering. Steps 4 and 5 are the cures, and the
  identity is exact with them.

On the terrace, with the totals taken out, 100 vectors gave 1.00987 for the
norm and a leading Ritz value of 1.00613 (e-folding 82 h, tracers) before
steps 4 and 5, and give 1.00028 and 0.99978 with them.

### Measurements

The probes run in pseudo-energy coordinates (`EnergyCoordinates`: the shear
times `√(ρ₀ m H)`, the depth mean times `√(ρ₀ m D)`, `η` times `√(ρ₀ g m)`,
the density anomaly about the reference at the layer's height times
`√(g m H/Γ)`; the squared norm is twice the pseudo-energy), on the departures
that keep the volume, the heat and the salt. Arnoldi's Ritz values then lie
in the step map's numerical range in that norm, whatever their residuals,
and the largest singular value of the Arnoldi matrix is the largest growth
of the norm over the horizon among the Krylov vectors. For a stable scheme
without tracer diffusion the Ritz values crowd the unit circle from inside
and do not converge one by one (the leading ones' residuals are 1e-2 to
1e-1), so three things are reported: that growth, the leading Ritz values at
30 and at 100 vectors, and the pseudo-energy of a random departure over days
(`probe_pseudo_energy`), which a growing mode of any shape would raise.

"Before" is the code of 2026-11-17 (σ-pairs of the full density with the
profile's force subtracted, LimitedAkima about the per-node vertical
reference); "after" this scheme, steps 1 to 5. Constant mixing
(ν 1e-3 m²/s), no tracer diffusion, no limiter. The terrace fixture is 22
elements of the smoothed Frøya bed around a terrace beside shore elements
with dry nodes, in 5 s steps; the slices are a step of the bed inside one
1 km P2 element of an 8 km channel, in 36 s steps, not smoothed.

**The spectrum of the step map** (largest growth of the norm over the
horizon / leading Ritz value):

| | before | after, 30 vectors | after, 100 vectors |
|---|---|---|---|
| terrace, 1800 s | a pair at 1.2974 (e-folding 115.2 min, ω 5.46e-4 rad/s, residual 8.7e-3), a second at 153 min | 0.99747 / 0.99531 | 1.00028 / 0.99978; then a pair at 0.97971, ω 5.72e-4 rad/s, velocity and anomaly in equal parts: the old mode's oscillation, decaying |
| 15 → 300 m cliff, 3600 s | pairs with e-folding 46.9, 47.6, 48.5 min (residuals ≤ 2e-7); the norm grows 4.62× | 0.99817 / 0.99485 | 0.99968 / 0.99907 |
| land beside 300 m, 3600 s | on the day map, tracer modes at 2.24, 1.73, 1.37 (e-folding 30, 44, 76 h; residuals 3e-14, 2e-9, 4e-4) | 0.99948 / 0.98955 | 1.00242 / 0.99808 |
| 0.5 m beside 150 m, 3600 s | | 1.00056 / 0.99731 | 1.00195 / 0.99965 |
| 15 → 60 m, 3600 s | | 0.99880 / 0.99793 | 0.99974 / 0.99960 |

No Ritz value is above 1. The norm's growth over the horizon is above 1 by
3e-4 to 2.4e-3 on the terrace and beside shallow or dry columns at 100
vectors: transient, in the parts that have no partner (the 2D Riemann
flux's dissipation moves the stratification too, an interface to a thin
column, the splitting in time).

**At rest** (largest layer speed):

| | before | after |
|---|---|---|
| terrace, seeded with 1e-6 °C | 4.2e-7 m/s at 3 h, 1.3e-5 at 12 h | 3.7e-7 at 3 h, 4.6e-7 at 12 h |
| the cliff from round-off, 6 h | 6.6e-10 m/s (2.3e-8 when its gate was written) | 1.4e-12 |
| unsmoothed steps from round-off, 12 → 48 h | 15 → 300 m: 2.4e-6 → NaN; 0.5 → 150 m: 4.1e-8 → NaN; 15 → 60 m: 6.1e-11 → 5.0e-2 (e-folding 1.8 h); land → 300 m: 1.8e-12 → 2.6e-12 | at 48 h: 9.3e-12, 2.4e-12, 2.2e-12, 2.2e-12 |
| the cliff at the model's own step (27.4 s), 6 h | | 4.2e-12 |

**The pseudo-energy of a random departure** (every field, 1e-6 at most, the
totals kept; fraction of its start):

| hours | 1 | 6 | 24 | 48 | 96 | 144 |
|---|---|---|---|---|---|---|
| 15 → 300 m cliff | 0.525 | 0.399 | 0.286 | 0.237 | 0.203 | |
| land beside 300 m | 0.723 | 0.546 | 0.366 | 0.272 | 0.206 | |
| 0.5 m beside 150 m | 0.472 | 0.381 | 0.301 | 0.268 | 0.248 | |
| terrace | | 0.233 | 0.168 | 0.154 | 0.146 | 0.142 |

On the slices it falls at every one of the 9600 steps. On the terrace it falls
from each sampled time to the next through 144 h (0.193 at 12 h), and over a
single step it rises by 1.3e-7 of itself at most (the splitting's exchanges
between the two modes). With steps 1 to 3 alone, and the totals not kept, the
terrace gave 0.229, 0.164, 0.151 and 0.1477 at 6, 24, 48 and 96 h, and 0.1507
at 144 h, with single steps rising by up to 1.4e-6.

**Conservation.** A 0.5 m seiche over a 12 → 4 m slope with a 6 °C
thermocline in the reference, one period, on 3, 6 and 12 levels
(`a_curved_reference_keeps_constancy_and_the_tracers_inventories`): the
volume to 1.9e-14, the inventories of `T` and `S` to 2.2e-14, a uniform
tracer uniform, while `T` moves by up to 2.6 °C at a node. With steps 1 and 2
alone the basin's mean `T` changed by 4.5e-3, 9.3e-4 and 2.6e-4 °C.

**The time step** (largest stable step: the hour's or two hours' map stays
finite; the model's bounds at Courant number 1):

| | before | after |
|---|---|---|
| terrace: largest stable step | 30 s (36 s blows up within the hour) | 30 s (36 s blows up) |
| terrace: the model's bounds | advection and internal waves 10.3 s | the same; sloping levels 53.6 s, not binding |
| 15 → 300 m cliff: largest stable step | 240 s (360 s blows up; its round-off grows at any step) | 60 s (72 s blows up) |
| 15 → 300 m cliff: the model's bounds | advection and internal waves 161.5 s | sloping levels 27.4 s |
| 0.5 m beside 150 m: the model's bounds | 213.4 s | sloping levels 177.9 s |
| 15 → 60 m: the model's bounds | 264.1 s | the same (sloping levels 303.9 s) |
| land beside 300 m: the model's bounds | 161.5 s | the same (sloping levels 617 s, the shore element's interfaces) |

So the step the cure needs is the same on the fixture cut from the Frøya bed,
and four times shorter on the cliff slice, where a level rises 142 m between
two nodes 500 m apart through `N = 0.049 s⁻¹`. The model takes 27 s there by
its own bound, a sixth of what it took before. Whether the bound sets the
step of a real run depends on its steepest stratified element against its
fastest current: the full-mesh run prints both (`notes/geometry-mesh.md`).

**The cost per step** (minima of alternating runs, 2026-11-21, 4 cores):

| | before | after |
|---|---|---|
| `profile_3d nx=64 ny=16 order=2 levels=20` (1024 elements, release), 1 thread | 353.6 ms | 388.0 ms (+9.7 %): the momentum stages 61.1 → 71.4, the tracer stages 75.1 → 92.0, `compute_dt` 8.9 → 14.4 (the reference's fields, which the first stage then finds filled), the barotropic pass and the rest unchanged |
| the same, 4 threads | 146.4 ms | 151.3 ms (+3.4 %) |
| terrace fixture (22 elements, 20 levels, test profile), 1 thread | 20.3 ms | 21.9 ms (+7.8 %) |

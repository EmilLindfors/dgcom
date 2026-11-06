# NorKyst-800 v3: tides, datum, pressure and velocities (P3.1, P1.5)

Measured 2026-10-16 against 17 Kartverket gauges from Tregde to Vardø, on NorKyst-800 v3 as published on MET's THREDDS: the operational "best estimate" aggregation `fou-hi/norkystv3_800m_m00_be` (2024-01 to now), the hindcast `romshindcast/norkyst_v3/norkyst_v3_sdepth_aggr.ncml` (2012-01 to 2025-08), and the two tidal forcing files MET publishes. Everything below comes out of one script, `./scripts/norkyst_tide_check.sh` (see "Reproducing"); the full tables it writes are `output/norkyst_issues/tables.md` and `output/norkyst_issues/forcing.md`.

## Summary

1. **Our analysis is not the problem.** A record built from UTide's astronomy (none of ours) is recovered by our 30-day fit to ≤ 5 % per constituent, N2 to 1 % (`tests/harmonic_analysis_test.rs`); under 0.10 m of red weather noise a 30-day N2 is 0.999 ± 0.016 of the truth. On the real series UTide gives the same NorKyst/gauge N2 as our fit (within 0.01–0.04). The gauges' June N2 is their year's within 0–3 %, and Kartverket's own prediction has 0.99–1.00 of the observed N2. NorKyst's ζ is instantaneous (`cell_methods: point`) on whole UTC hours, and its M2 phase agrees with the gauges within ±1.1° (2 min) at 11 of 17 gauges.
2. **NorKyst's N2 was 0.10–0.24 of the observed at every gauge** (median 0.19, phase −6° at Stavanger to −56° at Tromsø) **from January 2024 to February 2026, and is right since March 2026** (0.85–1.18, median 0.99, ±6°). The cause is in MET's tide file: the one that forces the hindcast (`forcing/tide.nc`, created 2023-03-31) has N2/M2 of 0.020–0.034 along the open boundaries against the gauges' 0.21; the one created 2026-02-25 (`static_nc_files/norkyst_tide.nc`) has 0.20–0.26. The 2012–2025 hindcast has N2 at 0.30–0.44 (median 0.38).
3. **Still wrong in the operational output in 2026:** K2 is 0.81 of the observed (0.78 in 2024–25), and K2/S2 relative to the gauges is 0.80 (0.78), as if K2's nodal factor (f ≈ 1.3 in 2024–26) were not applied; P1 is 1.7× and Q1 1.7× (1.35× in 2024–25), not explained by the forcing file's amplitudes; the minor semidiurnals (2N2, μ2, ν2, L2, λ2) are absent (mostly 0.01–0.13×; 2–7 cm of |ΔZ| together); M4 is 0.23×. The hindcast over 2024–25 differs from the forecasts on the same dates (K2 0.72×, N2 0.38×), and neither matches correct nodal factors; a static tide file used without them is wrong by an amount that changes through the 18.6-year nodal cycle.
4. **Datum.** NorKyst's mean ζ is 0.28 m below the gauges' mean (median; 0.20–0.40 m, deepest in Finnmark), so 0.22–0.52 m below NN2000 too; it is neither MSL nor NN2000.
5. **Pressure.** NorKyst's non-tidal ζ responds to sea-level pressure at a quarter of the gauges' rate (−0.22 against −0.80 cm/hPa, daily means against ERA5); the gauge-minus-NorKyst residual regresses on pressure at −0.58 cm/hPa (r = −0.8). So NorKyst's ζ carries little of the inverse barometer, although its configuration has `ATM_PRESS`.
6. **Velocities.** The operational output has no ubar/vbar (the hindcast has `ubar_eastward`/`vbar_northward`), and its z-levels stop at 300 m.

MET's answers (2026-11-05) are summarised below under "What MET confirmed". In short: the operational system switched to the new tide file on 2026-03-03; nodal corrections are probably not applied; ζ has no geodetic datum (subtract the model's own time mean); the inverse barometer is applied at the open boundary.

Correction to the brief: Mausund's K1 1.14× and O1 0.95× came from a year fit of the gauge against a 30-day fit of NorKyst; on the same 30 days (June 2025) they are K1 1.21×, O1 1.10× (the year fit, both sides: K1 1.06×, O1 0.95×). A 30-day record cannot separate P1 from K1 or K2 from S2 (182.6 days), so the diurnal and K2/S2 numbers below come from the year (2024-07 to 2025-07) and the 2026-04 to 2026-10 half year.

## Reproducing

```bash
./scripts/norkyst_tide_check.sh     # ≈ 1.5 h the first time (THREDDS), ≈ 2 min after
```

| Step | Script | Output |
|---|---|---|
| Gauges: Kartverket observations and predictions, 17 permanent gauges | `scripts/kartverket_gauge.sh`, list in `scripts/norkyst_tide_gauges.csv` | `data/tide_gauges/coast/<CODE>_{obs,pre}[_2026-04].txt`, `<CODE>_obs_<year>.txt` (BGO, MSU, BOO, 2024–26) |
| NorKyst ζ at the nearest wet cell to each gauge (92–1866 m away), hourly | `scripts/norkyst_gauge_series.py` (`url=` for the hindcast) | `data/norkyst_gauges/{2024-07_2025-07,2026-04_2026-10,2024-01_2026-10,hindcast_2024-07_2025-07}/<CODE>.txt` |
| Our fit of gauge and NorKyst on the instants both have, one analysis | `examples/tide_gauge_fit.rs <gauge> compare=<NorKyst> csv=` | `output/norkyst_issues/fits.csv` (and `fits_pre.csv`, gauge against Kartverket's prediction) |
| Tables, the pair test, UTide, datum, pressure, plots | `scripts/norkyst_tide_report.py` | `tables.md`, `ratios_latitude.png`, `phases_latitude.png`, `n2_monthly.png` |
| The forcing files themselves | `scripts/norkyst_tide_forcing.py` | `forcing.md` |
| The analysis gates | `cargo test --test harmonic_analysis_test` | `test_n2_q1_recovery_against_independent_astronomy`, `test_thirty_day_n2_q1_under_weather_noise` |

Every comparison uses the instants both series have (NorKyst's aggregation has five 25–27 h gaps in 2024, which are left out, never filled). Ratios are NorKyst over the gauge; ΔG = G_NorKyst − G_gauge (> 0: NorKyst later). Pressure is ERA5 `pressure_msl` from the Open-Meteo archive (cached in `data/pressure/`); MSL and NN2000 come from Kartverket's `locationlevels` (cached in `data/kartverket/`).

## 1. Is it our analysis? (no)

- **Independent astronomy.** `test_n2_q1_recovery_against_independent_astronomy` builds 2025-06-01 + 30 days hourly from the standard speeds and UTide's f, V₀ + u (`uv run scripts/norkyst_tide_report.py fuv 2025-06-01 2025-07-01` prints them) with Mausund-sized constants; the fit `tide_gauge_fit` runs recovers every constituent to ≤ 5 % of its amplitude (Q1's 4.4 % is the two astronomies' satellite terms: f 2.6 %, u 2° apart) and N2 to 1 %.
- **Weather noise.** `test_thirty_day_n2_q1_under_weather_noise`: 40 realizations of 0.10 m AR(1) noise (the gauges' 30-day residual is 0.075 m in June, 0.11–0.16 m in January) give N2 0.999 ± 0.016 (worst 6 %), Q1 1.01 ± 0.21 (worst 56 %). A single 30-day Q1 ratio is noise; N2 is not.
- **The gauges' own N2** (`tables.md` A1): June/year 1.00–1.03 at 16 gauges (Stavanger 0.94, N2 there 0.03 m); Kartverket's prediction over the year has 0.986–1.001 of our observed N2. N2/M2 at the gauges is 0.18–0.25 (0.207 median), normal.
  - January/year is 0.75–0.82 at 16 gauges (Tregde 1.07): over 30 days ν2 (≈ 0.18 of N2 here, 206 days from N2) cannot be separated and beats against N2. NorKyst has no ν2, so its monthly N2 ratio against the gauge swings ±20 % for this reason alone (Jan 2025 0.25, June 2025 0.17 at the median gauge; `n2_monthly.png`). Use the long windows.
- **UTide against our fit on the real series** (A2, June 2025): N2 ratio 0.075–0.239 ours against 0.111–0.238 UTide (UTide fits more constituents; Mausund 0.175 / 0.188, Bodø 0.188 / 0.193).
- **Time axis** (A3): every NorKyst sample is on a whole UTC hour; M2 ΔG over the year is within ±1.1° at 11 gauges, −1.7 to −6.6° at Tregde, Stavanger, Bergen, Måløy, Heimsjø, Tromsø (≤ 14 min, local).
- V₀, f, u are applied identically to both series by construction: `tide_gauge_fit compare=` restricts both to the shared instants and fits them with the same `HarmonicAnalysis`.

## 2. N2, coast-wide, and fixed in March 2026

Our fit, NorKyst against the gauge, ratio (ΔG °). The plots are `output/norkyst_issues/ratios_latitude.png` and `phases_latitude.png`; June 2025, January 2025 and June 2026 (30 days each) are in `tables.md` B.

**2024-07-01 to 2025-07-01 (operational, old tide file):**

| gauge | lat | N2 | Q1 | O1 | K1 | S2 | M2 | M4 | K2 | P1 | RSS (m) | Δmean (m) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TRG | 58.01 | 0.21 (-21) | 0.22 (-152) | 0.60 (-34) | 1.06 (+91) | 0.63 (+21) | 1.43 (-3) | 1.94 (+103) | 0.82 (+16) | 2.26 (+7) | 0.050 | -0.315 |
| SVG | 58.97 | 0.10 (-6) | 0.82 (+26) | 0.64 (-5) | 1.16 (+5) | 0.94 (-6) | 0.82 (-7) | 0.23 (+121) | 0.65 (+17) | 1.28 (+19) | 0.052 | -0.302 |
| BGO | 60.40 | 0.13 (-21) | 0.72 (-5) | 0.73 (-11) | 1.00 (+4) | 0.90 (-4) | 0.92 (-5) | 0.13 (+114) | 0.63 (+12) | 1.36 (+5) | 0.101 | -0.288 |
| MAY | 61.93 | 0.16 (-22) | 0.62 (-37) | 0.72 (-17) | 1.10 (-5) | 0.96 (+0) | 1.01 (-2) | 0.71 (-106) | 0.70 (+11) | 2.07 (-2) | 0.111 | -0.290 |
| AES | 62.47 | 0.17 (-21) | 1.00 (-49) | 0.90 (-17) | 1.43 (-4) | 0.95 (+1) | 1.02 (+0) | 0.27 (-45) | 0.72 (+11) | 2.92 (+10) | 0.121 | -0.280 |
| KSU | 63.11 | 0.17 (-25) | 1.25 (-40) | 0.97 (-14) | 1.42 (+6) | 0.94 (+1) | 1.01 (-0) | 0.17 (-36) | 0.73 (+9) | 2.50 (+27) | 0.131 | -0.271 |
| HEI | 63.43 | 0.18 (-32) | 1.35 (-37) | 1.00 (-12) | 1.24 (+10) | 0.96 (-0) | 1.02 (-2) | 0.26 (-65) | 0.75 (+7) | 2.13 (+29) | 0.150 | -0.259 |
| MSU | 63.87 | 0.18 (-31) | 1.36 (-33) | 0.95 (-9) | 1.06 (+9) | 0.97 (+1) | 1.03 (-1) | 0.16 (-9) | 0.76 (+9) | 1.89 (+27) | 0.138 | -0.275 |
| TRD | 63.44 | 0.18 (-32) | 1.37 (-34) | 0.99 (-10) | 1.15 (+7) | 0.99 (+1) | 1.05 (-0) | 0.38 (-56) | 0.78 (+8) | 1.96 (+28) | 0.182 | -0.201 |
| RVK | 64.86 | 0.19 (-36) | 1.40 (-31) | 0.98 (-12) | 0.99 (-0) | 0.99 (+3) | 1.04 (+0) | 0.17 (-35) | 0.77 (+9) | 1.68 (+20) | 0.151 | -0.258 |
| TAZ | 66.50 | 0.20 (-42) | 1.36 (-31) | 1.20 (-16) | 0.97 (-11) | 0.99 (+3) | 1.03 (+0) | 0.23 (-43) | 0.79 (+7) | 1.50 (+3) | 0.150 | -0.285 |
| BOO | 67.29 | 0.20 (-46) | 1.32 (-28) | 1.40 (-18) | 0.95 (-17) | 0.99 (+4) | 1.02 (+1) | 0.22 (-72) | 0.79 (+7) | 1.40 (-4) | 0.171 | -0.262 |
| KAB | 68.21 | 0.20 (-47) | 1.33 (-27) | 1.44 (-21) | 0.93 (-16) | 0.98 (+4) | 1.01 (+1) | 0.21 (-78) | 0.78 (+7) | 1.37 (-4) | 0.184 | -0.280 |
| NVK | 68.43 | 0.20 (-47) | 1.35 (-26) | 1.58 (-19) | 0.92 (-18) | 1.01 (+4) | 1.03 (+1) | 0.22 (-83) | 0.80 (+7) | 1.37 (-6) | 0.205 | -0.231 |
| TOS | 69.65 | 0.24 (-56) | 1.70 (-35) | 1.36 (-12) | 1.16 (-14) | 1.11 (-1) | 1.07 (-4) | 0.49 (+133) | 0.92 (-1) | 1.89 (-8) | 0.183 | -0.319 |
| HFT | 70.66 | 0.23 (-53) | 1.80 (-29) | 1.58 (-7) | 1.05 (-15) | 1.11 (+6) | 1.04 (-1) | 0.14 (+20) | 0.90 (+4) | 1.75 (-8) | 0.185 | -0.331 |
| VAW | 70.37 | 0.21 (-51) | 1.71 (-20) | 2.06 (+2) | 0.93 (-16) | 1.23 (+3) | 1.00 (+1) | 0.55 (-78) | 0.81 (+3) | 1.32 (-8) | 0.220 | -0.398 |

Median ratio: N2 0.19, Q1 1.35, O1 0.99, K1 1.06, S2 0.98, M2 1.02, M4 0.23, K2 0.78, P1 1.75. RSS median 0.150 m.

**2026-04-01 to 2026-10-01 (operational, new tide file):**

| gauge | lat | N2 | Q1 | O1 | K1 | S2 | M2 | M4 | K2 | P1 | RSS (m) | Δmean (m) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TRG | 58.01 | 1.14 (+4) | 1.15 (-78) | 0.53 (-31) | 1.00 (+69) | 0.80 (+27) | 1.53 (-11) | 6.68 (+13) | 0.83 (+22) | 0.86 (-107) | 0.056 | -0.281 |
| SVG | 58.97 | 0.85 (+2) | 1.32 (+27) | 0.67 (-4) | 1.16 (+13) | 0.96 (-6) | 0.77 (-4) | 0.30 (+89) | 0.66 (+23) | 1.43 (+37) | 0.051 | -0.260 |
| BGO | 60.40 | 0.92 (-2) | 1.00 (-0) | 0.74 (-13) | 0.97 (+9) | 0.92 (-5) | 0.90 (-4) | 0.13 (+53) | 0.66 (+17) | 1.45 (+25) | 0.065 | -0.251 |
| MAY | 61.93 | 0.98 (+0) | 0.91 (-35) | 0.74 (-21) | 1.06 (-0) | 0.97 (-0) | 0.99 (-1) | 0.82 (-117) | 0.73 (+19) | 2.18 (+9) | 0.039 | -0.244 |
| AES | 62.47 | 0.99 (+2) | 1.36 (-41) | 0.90 (-18) | 1.41 (-1) | 0.97 (+1) | 1.00 (+1) | 0.27 (-49) | 0.75 (+18) | 2.98 (+17) | 0.053 | -0.234 |
| KSU | 63.11 | 0.98 (+1) | 1.65 (-32) | 0.97 (-16) | 1.42 (+9) | 0.96 (+1) | 1.00 (+0) | 0.20 (-50) | 0.76 (+16) | 2.61 (+32) | 0.054 | -0.227 |
| HEI | 63.43 | 0.97 (-1) | 1.75 (-29) | 1.00 (-14) | 1.24 (+14) | 0.97 (-1) | 1.00 (-1) | 0.28 (-72) | 0.77 (+15) | 2.22 (+36) | 0.057 | -0.204 |
| MSU | 63.87 | 0.99 (+0) | 1.78 (-26) | 0.96 (-12) | 1.07 (+14) | 0.99 (+1) | 1.01 (+0) | 0.17 (-22) | 0.79 (+16) | 1.91 (+33) | 0.045 | -0.229 |
| TRD | 63.44 | 1.00 (+0) | 1.72 (-27) | 0.99 (-13) | 1.14 (+13) | 1.00 (+1) | 1.02 (-0) | 0.36 (-68) | 0.81 (+16) | 1.96 (+37) | 0.085 | -0.158 |
| RVK | 64.86 | 0.99 (+1) | 1.81 (-24) | 1.00 (-15) | 1.00 (+6) | 1.01 (+2) | 1.02 (+1) | 0.20 (-34) | 0.81 (+16) | 1.72 (+24) | 0.049 | -0.207 |
| TAZ | 66.50 | 0.99 (+0) | 1.76 (-21) | 1.21 (-17) | 0.97 (-7) | 1.01 (+3) | 1.02 (+1) | 0.22 (-45) | 0.83 (+15) | 1.58 (+5) | 0.045 | -0.239 |
| BOO | 67.29 | 0.98 (+1) | 1.73 (-17) | 1.37 (-18) | 0.95 (-13) | 1.02 (+3) | 1.01 (+1) | 0.23 (-71) | 0.83 (+15) | 1.50 (-2) | 0.061 | -0.209 |
| KAB | 68.21 | 0.97 (+1) | 1.74 (-17) | 1.40 (-20) | 0.93 (-13) | 1.01 (+3) | 1.00 (+1) | 0.21 (-76) | 0.83 (+14) | 1.46 (-2) | 0.070 | -0.238 |
| NVK | 68.43 | 0.98 (+1) | 1.71 (-16) | 1.53 (-19) | 0.92 (-14) | 1.02 (+3) | 1.01 (+1) | 0.22 (-82) | 0.84 (+14) | 1.46 (-3) | 0.093 | -0.188 |
| TOS | 69.65 | 1.15 (-5) | 2.05 (-29) | 1.36 (-14) | 1.16 (-9) | 1.13 (-1) | 1.05 (-3) | 0.24 (+123) | 0.94 (+7) | 2.18 (-2) | 0.092 | -0.273 |
| HFT | 70.66 | 1.17 (-4) | 2.00 (-22) | 1.58 (-10) | 1.06 (-11) | 1.14 (+5) | 1.03 (-1) | 0.23 (+31) | 0.92 (+10) | 2.04 (-5) | 0.076 | -0.288 |
| VAW | 70.37 | 1.18 (-6) | 1.80 (-16) | 2.09 (-1) | 0.94 (-12) | 1.27 (+1) | 0.99 (-0) | 0.49 (-76) | 0.83 (+7) | 1.54 (-8) | 0.104 | -0.340 |

Median ratio: N2 0.99, Q1 1.73, O1 1.00, K1 1.06, S2 1.00, M2 1.01, M4 0.23, K2 0.81, P1 1.72. RSS median 0.057 m: the new file removed about 60 % of NorKyst's tidal error at the gauges.

- **Coast-wide, not local:** the N2 deficit is the same at all 17 gauges from Tregde to Vardø and in both seasons (June 2025 median 0.17, January 2025 0.25, the year 0.19); the gauges' N2/M2 is normal everywhere. Tregde (an M2 amphidrome, M2 0.08 m) and Stavanger are the noisy ends of every table.
- **When it changed** (`tables.md` E, `n2_monthly.png`, calendar months at Bergen, Mausund and Bodø, 2024-01 to 2026-09): N2 0.07–0.25 in every month through 2026-02, 0.80–1.18 from 2026-03 (the ±20 % is the ν2 beat of section 1). The tide file is dated 2026-02-25; February is still wrong at all three gauges, so the switch came between 2026-02-25 and early March.
- **The hindcast** (`tables.md` B.hindcast, 2024-07..2025-07, the same instants): N2 0.30–0.44 (median 0.38, −4 to −28°), K2 0.72 (−14°), P1 1.73, Q1 1.28, M2 1.03, RSS median 0.126 m: better N2 than the forecasts of the same dates (0.19), worse K2 (0.78), so the two were not forced alike.

## 3. What the forcing files contain (`output/norkyst_issues/forcing.md`)

Both files: `title = Tidal forcing from met.no`, `history = Created by tide2roms from sequential files from met.no`, `constituents = K2 S2 M2 N2 K1 P1 O1 Q1`, periods correct and in that order (11.967235 … 26.868357 h): no period/order mix-up. Values on the whole grid (the atlas interpolated), of which ROMS uses the open boundaries.

| | hindcast `romshindcast/norkyst_v3/forcing/tide.nc` (2023-03-31) | operational `fou-hi/norkystv3_his_files/static_nc_files/norkyst_tide.nc` (2026-02-25) | the gauges |
|---|---|---|---|
| N2/M2 on the boundaries (S, N, W, E) | 0.020, 0.027, 0.031, 0.034 | 0.212, 0.204, 0.199, 0.255 | 0.208 |
| Q1/O1 | 0.43, 0.37, 0.35, 0.60 | 0.45, 0.38, 0.36, 0.63 | 0.44 |
| K2/S2 | 0.21, 0.20, 0.21, 0.20 | 0.22, 0.21, 0.22, 0.21 | 0.28 |
| P1/K1 | 0.26, 0.29, 0.35, 0.30 | 0.26, 0.28, 0.34, 0.29 | 0.27 |
| N2 at the gauges, file / gauge (Tromsø left out: its nearest cell has no atlas values, every constituent ≤ 0.2×) | 0.05–0.13 (Tregde 0.28) | 0.71–1.06 | |

- The new file has 9.05× the old one's N2 on the boundaries and the other amplitudes within 0–9 %; its phases are the old ones shifted uniformly per constituent (M2 +84.4°, O1 +68.4°, K2 +19.8°, S2 −0.2°), i.e. a different reference date; N2's shift scatters (+23 to +79°), the old N2 phases being wrong too.
- **The reference date.** ROMS forces ζ = A cos(ω(t − TIDE_START) − φ), so G − φ = V₀(T₀) + u(T₀) at every gauge. Searched hourly over 2010–2027 against the gauges' G: the new file fits **2017-01-01 00:00 UTC with V₀ + u** (weighted RMS 4.5° over the eight constituents; the next minimum more than 30 days away 6.3°), which is `TIDE_START` in the published configuration (Zenodo 10.5281/zenodo.16810677, `roms_norkyst.in`: `TIDE_START = 17167.0d0`, `TIME_REF = 19700101`). Its amplitudes carry 2017's nodal factors: file/gauge K2 0.752 against f(2017) = 0.764, O1 0.851 against 0.825. The old file fits 2014-01-01 00:00 (6.2°, but N2 alone is off by 57°).
- **So the static file is a 2017 snapshot.** Used as is from 2017-01-01, it gives the tide of 2017's nodal state at every later date: in 2025–26 K2 at 0.58–0.60 of the truth, O1 0.70, K1 0.80, M2 1.07 (f(2017)/f(2025); `tables.md` C, "frozen at 2017-01-01"). Neither output behaves like that: the forecasts have K2 0.78–0.81× and M2 1.01–1.02×, the hindcast K2 0.72× and M2 1.03× (K2/S2 0.72, against 0.58 for a file frozen at 2017 and 0.76 without nodal factors), so the runs apparently re-reference the files in a way the published configuration does not show.
- No tidal potential in the interior: the configuration (`norkyst.h`) defines `SSH_TIDES`, `UV_TIDES`, `ADD_FSOBC`, `ADD_M2OBC`, not `TIDE_GENERATING_FORCES`. The NorKyst v3 paper (Christensen et al. 2026, GMD 19, 2785, doi:10.5194/gmd-19-2785-2026) names TPXO9 as the source, without the constituent list or nodal handling. NorKyst v2 (Albretsen et al. 2011, Fisken og Havet 2/2011) used TPXO7.2's eight constituents with per-run phase shifts and nodal corrections (`tidenc2roms`).

## 4. What is still wrong in 2026: the frequency-pair test

A pair of constituents close in frequency sees the same shelf dynamics (resonance, friction, resolution), so (X/Y)_NorKyst / (X/Y)_gauge isolates the forcing. Our fit, median over 17 gauges (`tables.md` C):

| pair (Δfrequency) | 2024-07..2025-07 | 2026-04..2026-10 | no nodal factors (2025 / 2026) | nodal frozen at 2017-01-01 (2025 / 2026) |
|---|---|---|---|---|
| N2/M2 (3.6 %) | 0.175, −31° | 0.975, +0° | 1.00 / 1.00 | 1.01 / 1.00 |
| K2/S2 (0.27 %) | 0.779, +6° | 0.802, +14° | 0.762 / 0.785 | 0.583 / 0.601 |
| M2/S2 (3.5 %) | 1.036, −2° | 1.002, −1° | 1.040 / 1.035 | 1.078 / 1.074 |
| P1/K1 (0.55 %) | 1.661, +13° | 1.723, +12° | 1.126 / 1.112 | 1.274 / 1.257 |
| Q1/O1 (3.9 %) | 1.137, −21° | 1.508, −14° | 0.984 / 0.959 | 1.012 / 0.986 |

- **K2:** 0.78–0.80 of the gauges relative to S2 in both periods, 13 of 17 gauges between 0.73 and 0.83, matching a K2 forced without its nodal factor (f = 1.31 in 2025, so 0.76) and not the 2017 snapshot (0.58–0.60). In 2024–25 M2/S2 (1.036) matches the same hypothesis (1.040); in 2026 it does not (1.002), so the 2026 change may have altered M2 too (or the response, with the new grid `norkyst_grd_v31.nc`); K2/S2 is the cleaner test. K2's error is 1–2 cm at most gauges. The hindcast's K2/S2 is 0.72 (−16°), between the two hypotheses.
- **P1:** 1.7× K1's ratio in both periods and in the hindcast (P1/K1 relative 1.36–2.11 from Bergen north). P1 and K1 are 0.55 % apart, so this is forcing (or forcing-like, e.g. atmospheric), but the files' P1/K1 is normal (0.26–0.35 on the boundaries, file/gauge 1.03 for P1 and 0.95 for K1 at the gauges). Not explained. Not a stitching artefact of the best-estimate aggregation: the hindcast has the same P1, S1 is small (NorKyst 0.002–0.008 m against the gauges' 0.004–0.011, `tables.md` H), and the residual's step into 00 UTC is at most 1.17× the other hours' at 13 gauges (1.3–2.2× at Træna, Bodø, Kabelvåg and Hammerfest, a small restart jump worth mentioning to MET).
- **Q1:** 1.35× (2024–25) and 1.73× (2026) of the gauges (Q1/O1 relative 1.14 and 1.51, so Q1 got worse in 2026; the file's Q1 changed by ×1.09, −11.6°). The files' Q1/O1 matches the gauges' 0.44 (the equilibrium 0.19 is not the reference here). Not explained either.
- **Minor semidiurnals absent** (UTide, the year, `tables.md` D): 2N2 0.03–0.13×, μ2 0.01–0.13× (Tromsø 0.36), ν2 0.02–0.09× (Stavanger 0.28), L2 0.01–0.13×, λ2 0.02–0.26×, 0.004–0.043 m each at the gauges; root sum of squares 0.020–0.072 m per gauge. Expected with eight constituents; TPXO9 (2N2) and FES2014 (μ2, ν2, L2, 2N2, λ2) have them.
- **Overtides:** M4 0.13–0.55× (median 0.23) in both periods except at Tregde (1.9×, an M2 amphidrome) and Måløy (0.7×); the forcing has no M4, and an 800 m grid on a smoothed bed (2–3× deeper than Kartverket's topobathy in 3–60 m of water, `notes/validation.md`) generates little.

## 5. Datum (`tables.md` F)

| | median | range |
|---|---|---|
| NorKyst − gauge mean, 2024-07..2025-07 | −0.280 m | −0.398 (Vardø) to −0.202 m (Trondheim) |
| June 2025 / January 2025 (30 days) | −0.279 / −0.327 m | |
| 2026-04..2026-10 | −0.238 m | −0.340 to −0.158 m |
| NorKyst mean − NN2000 (year) | −0.307 m | −0.523 to −0.223 m |

The gauges' year mean is +0.04 to +0.11 m above Kartverket's MSL (1996–2014), and NN2000 lies 0.05–0.24 m above that MSL (more in the north), so NorKyst's ζ is referenced to neither; its offset deepens northward and varies by 4–10 cm between seasons and years. The attribute says "Sea surface height above geoid"; MET documents no reference level or mean dynamic topography. MET has since confirmed that there is no geodetic datum: ζ is relative to the model's own rest level ("What MET confirmed", item 5).

## 6. Pressure (`tables.md` G)

Daily means of the non-tidal residual (UTide year fit removed) against ERA5 sea-level pressure at each gauge, 2024-07..2025-07, 361–363 days:

- The gauges: −0.69 to −1.00 cm/hPa (median −0.80; the static inverse barometer is −0.99, the rest is wind and the dynamic response).
- NorKyst: −0.07 (Tregde) to −0.56 cm/hPa (Vardø), median −0.22.
- Gauge − NorKyst: median −0.58 cm/hPa, r −0.48 to −0.82: the pressure explains most of what NorKyst misses (the difference's σ is 9–10 cm).

So NorKyst's ζ carries roughly a quarter of the observed pressure response, least in the south, although the GMD paper says "the atmospheric pressure is also included in the forcing" (`ATM_PRESS`, with `PRESS_COMPENSATE` at the boundaries). Either the published ζ has the inverse barometer removed, or the open boundary holds the level to the parent's (which has no surge; ROMS's `PRESS_COMPENSATE` acts in the Shchepetkin ubar condition, while ζ uses Chapman). For us: `froya_real_data ib=1` adds an inverse barometer that NorKyst mostly lacks, so it does not count it twice (at most ≈ 0.2 of 0.8 cm/hPa).

## 7. Velocities

- The operational aggregation and its daily `_AN` files hold `zeta`, `u_eastward`, `v_northward`, `w`, T, S, `AKs` and wind on 15 z-levels (0–300 m), no `ubar`/`vbar`; we depth-average the z-levels (`io::depth_average_z`), and deeper columns repeat the 300 m value, so boundary transports are approximate.
- The hindcast (`romshindcast/norkyst_v3/norkyst_v3_sdepth_aggr.ncml`, hourly 2012-01-05 to 2025-08-02) has `ubar_eastward`, `vbar_northward` and s-level fields. ROMS writes ubar/vbar in the forecasts too (`Hout(idUbar) == T` in the configuration); only the published product lacks them.

## 8. Minor

- The point/NCSS endpoint answers 503 most of the time; OPeNDAP windows (this report's fetches) work, with occasional DAP failures that succeed on retry.
- The bed and resolution (complaint 4 of the brief) and the overtides (5) are limitations of an 800 m model, not defects; they are documented in `notes/validation.md` and listed here for context only.

## What this means for us

- The boundary atlases built from NorKyst for 2025 dates (`data/froya_boundary_tides.txt`, `data/bodo_boundary_tides.txt`, June 2025) carry the old forcing's N2 (and K2, P1, Q1): `gauge_ratios`/`gauge_gains` stay necessary for them, circular as they are. For dates from 2026-03, NorKyst's N2 is right and the N2 correction is no longer needed; K2 (≈ 0.8×), P1 (1.7×) and Q1 (1.7×) still are, and theory can replace the gauge for two of them without circularity: K2 = 0.272 S2 and P1 = 0.331 K1 at the equilibrium ratios (`Inference::EQUILIBRIUM`; the gauges' own ratios are 0.284 and 0.268).
- P3.4 (Bodø, pre-registered) uses 2025-06 atlases with `nest_tides=raw`: its raw-forcing score includes NorKyst's 2025 N2 defect (0.20× at Bodø). That is part of what "raw" measures, but it is now known to be fixed in later NorKyst.
- `ib=1` does not add a second full IB with NorKyst nesting (section 6), but it may overshoot by up to ≈ 0.4 cm/hPa. See "What MET confirmed", item 4.
- The datum offset varies along the coast (0.20–0.40 m) and by season (4–10 cm). MET confirms (2026-11-05) that ζ is relative to the model's own rest level, so the right default is the parent's own long-term mean at the boundary, as `froya_real_data` takes it from the atlas `Z0` now. The gauges' MSL and NN2000 do not match NorKyst.
- **After the N2 fix, NorKyst's day-to-day tide error still cycles with the spring–neap phase** (2026-11-05, `notes/validation.md` P3.4): its centred RMSE against the gauge over hours 3–24 of each June 2026 day runs 2.3–12.1 cm at Bodø and 2.1–9.6 cm at Mausund, on the same dates at both (worst 06-10..13 and 06-19..21, best near springs 06-17 and neaps), with high water up to 20 min late before springs and 15 min early after. Kartverket's prediction stays at 2–6 cm. The likely terms are K2 without its nodal factor and the missing minor semidiurnals (their phase against M2 turns over the cycle), plus the diurnals; nested in raw NorKyst the model inherits it (0.5–0.7 cm from the parent at both gauges on 2026-06-12).
- The hindcast's ubar/vbar can measure the error of our z-level depth average (`io::depth_average_z`, 300 m cap) for 2012–2025 dates.

## What MET confirmed (2026-11-05)

MET Norway answered the questions from this report. Here is what they confirmed, item by item, and what each answer means for us:

1. **N2:** the operational system switched to the corrected tide file on 2026-03-03. This matches section 2: February is still wrong, March is right. The 2012–2025 hindcast stays at N2 0.38×.
   - **For us:** NorKyst's N2 can be trusted in forecasts from 2026-03-03.
   - Start atlas and scoring windows a few days later (2026-03-06 or after), so the tide has spun up from the new boundary.
2. **K2 and nodal corrections:** nodal f and u are probably not applied in the current runs.
   - That fits the K2/S2 pair test (0.78–0.80 measured, 0.76–0.79 expected without f) and M2/S2 in 2024–25.
   - It does not fit M2/S2 in 2026 (1.002 measured, 1.035 expected).
   - It does not explain how phases dated 2017 come out right at today's dates.
   - **For us:** the equilibrium K2 = 0.272 S2 in NorKyst-derived atlases is the right fix.
   - Multiplying every NorKyst constituent by f(date) is *not* supported: it would make M2 worse in 2026.
   - If NorKyst starts applying nodal corrections, its K2 (and perhaps M2, O1 and K1) will change again. Re-run `scripts/norkyst_tide_check.sh` on later months before building new atlases.
3. **P1 and Q1:** not explained yet.
   - **For us:** take P1 at the equilibrium ratio (P1 = 0.331 K1). Q1 still needs the gauge ratio.
4. **Pressure:** the inverse barometer is applied to the open-boundary values (ROMS `PRESS_COMPENSATE`).
   - That does not explain our measurement. If NorKyst carried the full inverse barometer, its non-tidal ζ would respond at about −1 cm/hPa, as the gauges do (−0.80), not at −0.22.
   - Two explanations remain: the published ζ is not the level the model ran with, or the boundary correction does not reach ζ (a Chapman ζ condition held to the parent, with the compensation only in ubar; section 6).
   - **For us:** NorKyst's ζ lacks most of the inverse barometer, so `ib=1` does not count it twice in full.
   - But `ib=1` adds the full static −0.99 cm/hPa, while gauge minus NorKyst regresses at only −0.58, and the parent already carries ≈ −0.22.
   - So `ib=1` may overshoot the pressure response by up to ≈ 0.4 cm/hPa (≈ 8 cm in a 20 hPa low), if the whole slope is pressure. Part of it is the wind, which correlates with pressure.
   - The option for a fractional inverse barometer exists since 2026-11-05 (`froya_real_data ib=<factor>`, `InverseBarometer::with_factor`). Scoring it on a stormy week is open (TODO P1.5).
5. **Datum:** ζ has no geodetic datum. It is relative to the model's rest level over its own bed, so its time mean at a point can sit above or below 0 and drifts on long and short time scales. To compare with gauges, subtract the model's own time mean.
   - The −0.28 m (−0.20 to −0.40 m along the coast) is the model's own mean dynamic level.
   - **For us:** this settles P1.5's datum question. The default is to subtract the parent's own long-term mean ζ at the boundary, over a month or more around the run, as the atlas `Z0` that `froya_real_data` uses does.
   - Do not use a fixed offset, MSL or NN2000.
   - Do not take the mean from the run's own 1–3-day window: that would also remove a real surge in the window.
6. **Minor semidiurnals** (2N2, μ2, ν2, L2): still absent.
   - **For us:** the catalogue item (2N2, μ2, ν2 and L2 from N2/M2 ratios) stays.
   - If NorKyst adds them, our atlases can take them from NorKyst directly.
7. **ubar/vbar:** the operational files may later be published on s-levels with ubar/vbar.
   - **For us:** `OceanModelReader` already prefers `ubar`/`vbar` when present (`src/io/ocean_model*.rs`), so the z-level depth average with its 300 m cap goes away by itself if those files appear.
   - Check the variable names when they do (the hindcast uses `ubar_eastward`/`vbar_northward`).
8. **The 00 UTC step** at Træna, Bodø, Kabelvåg and Hammerfest: no answer.

Still open on the pressure point (item 4): whether the boundary correction reaches ζ, or only ubar.

## Caveats and dead ends

- 30-day Q1 and the K2/P1 separation are noise- or Rayleigh-limited (section 1); the coast-wide claims use the year and the half year.
- The research note that the files' Q1/O1 (0.35–0.63) is "2–3× too large" compared it with the equilibrium 0.19; the gauges here have 0.44, so the files' Q1 is normal.
- The best-estimate aggregation's daily stitching at 00 UTC leaves a small step at four northern gauges and none elsewhere (section 4); with the hindcast's identical P1 it does not explain P1/Q1.
- Nearest-wet-cell sampling puts some NorKyst points 1–1.9 km from the gauge (Kristiansund, Rørvik, Bodø, Ålesund); the pair test cancels that, and N2 is low at the close points too (Tromsø 92 m, Stavanger 129 m).

#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "utide",
#     "matplotlib",
# ]
# ///
"""
The NorKyst-800 tide report (notes/norkyst-issues.md), from the files that
scripts/norkyst_tide_check.sh makes.

    uv run scripts/norkyst_tide_report.py <gauges.csv> <gauge dir> <norkyst dir> <out dir>
    uv run scripts/norkyst_tide_report.py fuv 2025-06-01 2025-07-01 [lat=63.87]

Sections of <out>/tables.md:

A. Is it our analysis? The gauges' 30-day N2 against their year's; the
   gauges against Kartverket's own prediction; NorKyst/gauge ratios from
   UTide (Codiga 2011, an independent V0/f/u and fit) against ours; the
   time axis (whole hours, the M2 phase difference as a clock check).
B. Coast-wide ratios and phase differences, NorKyst against the gauge, from
   our fits (fits.csv, tide_gauge_fit compare=) for June 2025, January 2025
   and the year 2024-07 to 2025-07.
C. The frequency-pair test: ratio-of-ratios for pairs close in frequency
   (N2/M2, K2/S2, M2/S2, P1/K1, Q1/O1), which no shelf dynamics can split,
   against what a forcing without nodal corrections (f = 1, u = 0) predicts.
D. Minor constituents (UTide's year fit): 2N2, MU2, NU2, L2, LDA2, EPS2,
   J1, OO1.
E. Monthly history 2024-01 to 2026-09 at BGO, MSU, BOO (UTide, calendar
   months, the same instants).
F. Datum: NorKyst's mean against the gauge's, MSL and NN2000 (Kartverket's
   `locationlevels`).
G. Pressure: daily-mean non-tidal residuals of gauge and NorKyst regressed
   on ERA5 sea-level pressure (Open-Meteo archive API, cached in
   data/pressure/): an inverse barometer in NorKyst shows as a gauge−NorKyst
   slope near 0, its absence as ≈ −1 cm/hPa.

`fuv` prints UTide's f, V0 (first sample) and u (midpoint) for a window,
the numbers in tests/harmonic_analysis_test.rs.
"""

import csv
import json
import re
import sys
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

MAIN = ["N2", "Q1", "O1", "K1", "S2", "M2", "M4"]
YEAR_EXTRA = ["K2", "P1"]
LONG_PERIOD = {"Mf", "Mm", "Ssa"}
PAIRS = [("N2", "M2"), ("K2", "S2"), ("M2", "S2"), ("P1", "K1"), ("Q1", "O1")]
MINORS = ["2N2", "MU2", "NU2", "L2", "LDA2", "EPS2", "J1", "OO1"]
WINDOWS = {
    "june": ("2025-06-01", "2025-07-01"),
    "january": ("2025-01-01", "2025-01-31"),
    "year": ("2024-07-01", "2025-07-01"),
    "june2026": ("2026-06-01", "2026-07-01"),
    "2026": ("2026-04-01", "2026-10-01"),
    # The hindcast (romshindcast/norkyst_v3), the same year as "year"
    "hindcast": ("2024-07-01", "2025-07-01"),
}
# Windows long enough to separate K2 from S2 and P1 from K1
LONG_WINDOWS = ["year", "2026", "hindcast"]
# The operational tide file's reference date (scripts/norkyst_tide_forcing.py)
FROZEN = "2017-01-01"
LONG_CODES = ["BGO", "MSU", "BOO"]


def unix(day):
    return datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()


def read_series(path):
    """A gauge-format file: header dict, Unix times, values."""
    meta, t, v = {}, [], []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            key, _, value = line[1:].partition(":")
            meta[key.strip().lower()] = value.strip()
            continue
        a, b = line.split()[:2]
        t.append(datetime.strptime(a, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp())
        v.append(float(b))
    return meta, np.array(t), np.array(v)


def common(a, b, start=None, end=None):
    """(t, va, vb) on the instants both series have in [start, end)."""
    (ta, va), (tb, vb) = a, b
    t = np.intersect1d(np.round(ta), np.round(tb))
    if start is not None:
        t = t[(t >= unix(start)) & (t < unix(end))]
    ia = np.searchsorted(np.round(ta), t)
    ib = np.searchsorted(np.round(tb), t)
    return t, va[ia], vb[ib]


def utide_fit(t, v, lat, constit="auto"):
    import utide

    times = t.astype("int64").astype("datetime64[s]")
    c = utide.solve(times, v, lat=lat, constit=constit, method="ols", conf_int="none",
                    verbose=False, trend=False, nodal=True)
    return c, {n: (a, g) for n, a, g in zip(c.name, c.A, c.g)}


def utide_residual(t, v, lat):
    import utide

    c, _ = utide_fit(t, v, lat)
    times = t.astype("int64").astype("datetime64[s]")
    return v - utide.reconstruct(times, c, verbose=False).h


def cdiff(a, g, b, h):
    return abs(a * np.exp(-1j * np.radians(g)) - b * np.exp(-1j * np.radians(h)))


def wrap(d):
    return (d + 180.0) % 360.0 - 180.0


def utide_fuv(start, end, lat, names):
    """{name: (f at mid, u at mid °, V0 at start °)} from UTide."""
    from utide._ut_constants import ut_constants
    from utide.harmonics import FUV

    table = list(ut_constants.const.name)
    # utide.astronomy.ut_astron takes datetime.toordinal days
    day = lambda s: datetime.strptime(s, "%Y-%m-%d").toordinal()
    t0 = float(day(start))
    hours = (unix(end) - unix(start)) / 3600.0
    tmid = t0 + 0.5 * (hours - 1) / 24.0
    out = {}
    for n in names:
        i = np.array([table.index(n)])
        f, u, _ = FUV(np.array([tmid]), tmid, i, lat, [0, 0, 0, 0])
        _, _, v0 = FUV(np.array([t0]), t0, i, lat, [0, 0, 0, 0])
        out[n] = (f.item(), 360.0 * u.item(), (360.0 * v0.item()) % 360.0)
    return out


def fuv_command(args):
    pos = [a for a in args if "=" not in a]
    kw = dict(a.split("=", 1) for a in args if "=" in a)
    lat = float(kw.get("lat", 63.87))
    for n, (f, u, v0) in utide_fuv(pos[0], pos[1], lat, ["M2", "S2", "N2", "K1", "O1", "Q1", "K2", "P1"]).items():
        print(f"{n:3s}  f {f:.5f}  V0 {v0:8.3f}  u {u:+7.3f}  V0+u {(v0 + u) % 360:8.3f}")


def read_fits(path):
    """{(code, window): {constituent: row}} from tide_gauge_fit's CSV."""
    fits = defaultdict(dict)
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            code, window = r["label"].split(":")
            fits[(code, window)][r["constituent"]] = r
    return fits


def fetch_json(url, cache):
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    cache.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as r:
        text = r.read().decode("utf-8")
    cache.write_text(text, encoding="utf-8")
    return json.loads(text)


def levels(code, lat, lon, cache_dir):
    """Kartverket's reference levels (cm above chart datum) at a gauge."""
    cache = cache_dir / f"{code}_levels.xml"
    if not cache.exists():
        url = (f"https://vannstand.kartverket.no/tideapi.php?tide_request=locationlevels"
               f"&lat={lat}&lon={lon}&lang=en")
        with urllib.request.urlopen(url, timeout=120) as r:
            cache.write_bytes(r.read())
    xml = cache.read_text(encoding="utf-8")
    return {m.group(1): float(m.group(2)) for m in re.finditer(r'reflevel code="([^"]+)" value="([^"]+)"', xml)}


def era5_pressure(code, lat, lon, start, end, cache_dir):
    """Hourly ERA5 sea-level pressure (hPa) at a point: Unix times, values."""
    last = datetime.fromtimestamp(unix(end) - 86400, timezone.utc).strftime("%Y-%m-%d")
    url = (f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
           f"&start_date={start}&end_date={last}&hourly=pressure_msl&timezone=GMT")
    data = fetch_json(url, cache_dir / f"{code}_era5_{start}_{end}.json")
    t = np.array([datetime.strptime(s, "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc).timestamp()
                  for s in data["hourly"]["time"]])
    p = np.array([np.nan if x is None else x for x in data["hourly"]["pressure_msl"]], dtype=float)
    return t, p


def daily_means(t, v):
    day = np.floor(t / 86400.0)
    out_t, out_v = [], []
    for d in np.unique(day):
        m = day == d
        if m.sum() >= 20:
            out_t.append(d)
            out_v.append(np.nanmean(v[m]))
    return np.array(out_t), np.array(out_v)


def slope(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m] - x[m].mean(), y[m] - y[m].mean()
    b = (x * y).sum() / (x * x).sum()
    r = (x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum())
    return b, r


def table(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main(argv):
    if argv and argv[0] == "fuv":
        return fuv_command(argv[1:])
    if len(argv) != 4:
        print(__doc__)
        sys.exit(1)
    gauges_csv, gauge_dir, nk_dir, out = map(Path, argv)
    with open(gauges_csv, newline="", encoding="utf-8") as f:
        gauges = list(csv.DictReader(f))
    lat_of = {g["code"]: float(g["lat"]) for g in gauges}
    codes = [g["code"] for g in gauges]
    fits = read_fits(out / "fits.csv")
    fits_pre = read_fits(out / "fits_pre.csv")
    year_dir = nk_dir / "2024-07_2025-07"
    md = ["# NorKyst-800 v3 tides against Kartverket gauges",
          "",
          f"Generated by `scripts/norkyst_tide_report.py` on {datetime.now(timezone.utc):%Y-%m-%d}. "
          "Ratios and ΔG are NorKyst against the gauge (ΔG > 0: NorKyst later). "
          "Windows: june 2025-06-01..07-01, january 2025-01-01..01-31, year 2024-07-01..2025-07-01; "
          "every comparison on the instants both series have.",
          ""]

    def get(code, window, name, key, src=fits):
        r = src.get((code, window), {}).get(name)
        return float(r[key]) if r and r[key] != "" else np.nan

    series = {}
    for code in codes:
        series[code] = (read_series(gauge_dir / f"{code}_obs.txt"), read_series(year_dir / f"{code}.txt"))

    # A. Our analysis
    md += ["## A. Is it our analysis?", "",
           "### A1. Gauge N2 (m): 30 days against the year, and Kartverket's prediction", "",
           "Our fit of the observations; `pre/obs` is the same fit of Kartverket's own prediction "
           "(their harmonic constants) over the year, divided by the observations'.", ""]
    rows = []
    for code in codes:
        hy, hj, hw = (get(code, w, "N2", "h_ref") for w in ("year", "june", "january"))
        n2m2 = hy / get(code, "year", "M2", "h_ref")
        pre = get(code, "year", "N2", "ratio", fits_pre)
        rows.append([code, f"{lat_of[code]:.2f}", f"{hy:.4f}", f"{hj:.4f}", f"{hw:.4f}",
                     f"{hj / hy:.3f}", f"{hw / hy:.3f}", f"{n2m2:.3f}", f"{pre:.3f}"])
    md += [table(rows, ["gauge", "lat", "N2 year", "N2 june", "N2 january", "june/year",
                        "january/year", "N2/M2 year", "pre/obs N2"]), ""]

    md += ["### A2. UTide against our fit: NorKyst/gauge ratio, June 2025", "",
           "UTide with its automatic constituent choice (Rayleigh 1) on the same instants; "
           "ours from fits.csv.", ""]
    rows = []
    utide_june = {}
    for code in codes:
        (_, tg, vg), (_, tn, vn) = series[code]
        t, a, b = common((tg, vg), (tn, vn), *WINDOWS["june"])
        _, G = utide_fit(t, a, lat_of[code])
        _, N = utide_fit(t, b, lat_of[code])
        utide_june[code] = (G, N)
        cells = [code]
        for n in ["N2", "Q1", "M2", "K1", "O1", "S2"]:
            ours = get(code, "june", n, "ratio")
            theirs = N[n][0] / G[n][0]
            cells.append(f"{ours:.3f} / {theirs:.3f}")
        rows.append(cells)
    md += [table(rows, ["gauge", "N2 ours / UTide", "Q1", "M2", "K1", "O1", "S2"]), ""]

    md += ["### A3. Time axis", "",
           "NorKyst `zeta` has `cell_methods: ocean_time: point` (instants). Every sample below "
           "is on a whole UTC hour; the M2 phase difference is the clock check (1° = 2.07 min).", ""]
    rows = []
    for code in codes:
        (_, tg, _), (_, tn, _) = series[code]
        whole = bool(np.all(np.mod(tn, 3600) == 0) and np.all(np.mod(tg, 3600) == 0))
        rows.append([code, len(tn), "yes" if whole else "NO",
                     f"{get(code, 'year', 'M2', 'dg_deg'):+.1f}",
                     f"{get(code, 'year', 'M2', 'dg_deg') * 2.07:+.1f}"])
    md += [table(rows, ["gauge", "NorKyst samples (year)", "whole hours", "M2 ΔG year (°)", "≈ min"]), ""]

    # B. Coast-wide tables
    md += ["## B. Coast-wide: NorKyst against the gauge (our fit)", ""]
    for window in WINDOWS:
        names = MAIN + (YEAR_EXTRA if window in LONG_WINDOWS else [])
        md += [f"### B.{window} ({WINDOWS[window][0]} to {WINDOWS[window][1]})", "",
               "ratio (ΔG °) per constituent; RSS is the root sum of squares of |ΔZ| over the "
               "fitted tidal constituents (no long periods, no inferred); Δmean = NorKyst − gauge.", ""]
        rows, rss_all, dmean_all = [], [], []
        for code in codes:
            f = fits.get((code, window), {})
            rss = np.sqrt(sum(float(r["dz"]) ** 2 for n, r in f.items()
                              if n != "Z0" and n not in LONG_PERIOD and r["inferred"] == "0"))
            cells = [code, f"{lat_of[code]:.2f}"]
            for n in names:
                inferred = f.get(n, {}).get("inferred") == "1"
                cells.append(f"{get(code, window, n, 'ratio'):.2f} ({get(code, window, n, 'dg_deg'):+.0f})"
                             + ("*" if inferred else ""))
            cells += [f"{rss:.3f}", f"{float(f['Z0']['dz']):+.3f}"]
            rss_all.append(rss)
            dmean_all.append(float(f["Z0"]["dz"]))
            rows.append(cells)
        md += [table(rows, ["gauge", "lat"] + names + ["RSS (m)", "Δmean (m)"]), ""]
        if window not in LONG_WINDOWS:
            md += ["`*` inferred at the equilibrium ratio (identical in both fits).", ""]
        # Medians
        med = [f"{n} {np.nanmedian([get(c, window, n, 'ratio') for c in codes]):.2f}" for n in names]
        med_dg = [f"{n} {np.nanmedian([get(c, window, n, 'dg_deg') for c in codes]):+.0f}°" for n in names]
        md += ["Median ratio: " + ", ".join(med), "",
               "Median ΔG: " + ", ".join(med_dg), "",
               f"RSS median {np.median(rss_all):.3f} m (range {min(rss_all):.3f}–{max(rss_all):.3f}); "
               f"Δmean median {np.median(dmean_all):+.3f} m (range {min(dmean_all):+.3f} to {max(dmean_all):+.3f})", ""]

    # C. Pair test
    md += ["## C. The frequency-pair test", "",
           "For a pair X/Y close in frequency, (X/Y)_NorKyst / (X/Y)_gauge and ΔG_X − ΔG_Y cancel "
           "NorKyst's dynamics (resonance, friction, resolution act alike on both) and leave the "
           "forcing. Two hypotheses for the forcing, with UTide's f, u (65°N): no nodal factors "
           "(f = 1, u = 0) gives f_Y/f_X and u_X − u_Y; nodal factors frozen at "
           f"{FROZEN} (the operational file's reference date) gives "
           "[f_X(T₀)/f_X] / [f_Y(T₀)/f_Y] and [u_X − u_X(T₀)] − [u_Y − u_Y(T₀)], f and u at the "
           "window's midpoint.", ""]
    names8 = ["M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1"]
    frozen = utide_fuv(FROZEN, FROZEN, 65.0, names8)
    for window in LONG_WINDOWS:
        mid = utide_fuv(*WINDOWS[window], 65.0, names8)
        rows = []
        for x, y in PAIRS:
            rr = np.array([get(c, window, x, "ratio") / get(c, window, y, "ratio") for c in codes])
            dd = np.array([wrap(get(c, window, x, "dg_deg") - get(c, window, y, "dg_deg")) for c in codes])
            (fx, ux, _), (fy, uy, _) = mid[x], mid[y]
            (fx0, ux0, _), (fy0, uy0, _) = frozen[x], frozen[y]
            rows.append([f"{x}/{y}", f"{np.nanmedian(rr):.3f}", f"{np.nanmin(rr):.3f}–{np.nanmax(rr):.3f}",
                         f"{np.nanmedian(dd):+.1f}", f"{np.nanmin(dd):+.0f}…{np.nanmax(dd):+.0f}",
                         f"{fy / fx:.3f} ({ux - uy:+.1f}°)",
                         f"{(fx0 / fx) / (fy0 / fy):.3f} ({(ux - ux0) - (uy - uy0):+.1f}°)"])
        md += [f"### C.{window} ({WINDOWS[window][0]} to {WINDOWS[window][1]})", "",
               table(rows, ["pair", "ratio-of-ratios median", "range", "Δ(ΔG) median (°)", "range",
                            "no nodal factors", f"frozen at {FROZEN}"]), ""]
        rows = []
        for code in codes:
            rows.append([code] + [f"{get(code, window, x, 'ratio') / get(code, window, y, 'ratio'):.2f}"
                                  for x, y in PAIRS])
        md += ["Per gauge (ratio-of-ratios):", "", table(rows, ["gauge"] + [f"{x}/{y}" for x, y in PAIRS]), ""]
    md += ["Single constituents against the hypotheses (NorKyst/gauge median over the gauges; "
           "the hypotheses give 1/f (no nodal factors) and f(T₀)/f (frozen), times NorKyst's own response):", ""]
    rows = []
    for window in LONG_WINDOWS:
        mid = utide_fuv(*WINDOWS[window], 65.0, names8)
        for n in names8:
            r = np.nanmedian([get(c, window, n, "ratio") for c in codes])
            dg = np.nanmedian([get(c, window, n, "dg_deg") for c in codes])
            rows.append([window, n, f"{r:.3f} ({dg:+.0f}°)", f"{1 / mid[n][0]:.3f} ({mid[n][1]:+.0f}°)",
                         f"{frozen[n][0] / mid[n][0]:.3f} ({mid[n][1] - frozen[n][1]:+.0f}°)"])
    md += [table(rows, ["window", "constituent", "NorKyst/gauge (ΔG)", "no nodal factors", f"frozen at {FROZEN}"]), ""]

    # D. Minor constituents
    md += ["## D. Minor constituents (UTide, the year, same instants)", "",
           "NorKyst/gauge amplitude ratio; gauge amplitude (m) in brackets; RSS of |ΔZ| over these.", ""]
    rows = []
    year_utide = {}
    for code in codes:
        (_, tg, vg), (_, tn, vn) = series[code]
        t, a, b = common((tg, vg), (tn, vn), *WINDOWS["year"])
        _, G = utide_fit(t, a, lat_of[code])
        _, N = utide_fit(t, b, lat_of[code])
        year_utide[code] = (G, N, t, a, b)
        cells, rss = [code], 0.0
        for n in MINORS:
            if n in G and n in N:
                cells.append(f"{N[n][0] / G[n][0]:.2f} ({G[n][0]:.3f})")
                rss += cdiff(*G[n], *N[n]) ** 2
            else:
                cells.append("–")
        cells.append(f"{np.sqrt(rss):.3f}")
        rows.append(cells)
    md += [table(rows, ["gauge"] + MINORS + ["RSS (m)"]), ""]

    # E. Monthly history
    long_dir = nk_dir / "2024-01_2026-10"
    history = {}
    md += ["## E. Monthly history, 2024-01 to 2026-09 (UTide: M2 S2 N2 K1 O1 Q1 M4, calendar months)", ""]
    rows = []
    for code in LONG_CODES:
        nk = long_dir / f"{code}.txt"
        parts = sorted(gauge_dir.glob(f"{code}_obs_20??.txt"))
        if not nk.exists() or not parts:
            md += [f"{code}: no long record (run norkyst_tide_check.sh)", ""]
            continue
        _, tn, vn = read_series(nk)
        gs = [read_series(p) for p in parts]
        tg = np.concatenate([g[1] for g in gs])
        vg = np.concatenate([g[2] for g in gs])
        tg, keep = np.unique(tg, return_index=True)
        vg = vg[keep]
        months = []
        for year in (2024, 2025, 2026):
            for month in range(1, 13):
                start = f"{year}-{month:02d}-01"
                end = f"{year + month // 12}-{month % 12 + 1:02d}-01"
                if unix(end) > unix("2026-10-01"):
                    continue
                t, a, b = common((tg, vg), (tn, vn), start, end)
                if len(t) < 24 * 28:
                    continue
                const = ["M2", "S2", "N2", "K1", "O1", "Q1", "M4"]
                _, G = utide_fit(t, a, lat_of[code], const)
                _, N = utide_fit(t, b, lat_of[code], const)
                months.append((start[:7], N["N2"][0] / G["N2"][0], wrap(N["N2"][1] - G["N2"][1]),
                               N["M2"][0] / G["M2"][0], N["Q1"][0] / G["Q1"][0],
                               N["O1"][0] / G["O1"][0], wrap(N["M2"][1] - G["M2"][1])))
        history[code] = months
        n2 = np.array([m[1] for m in months])
        rows.append([code, f"{months[0][0]}…{months[-1][0]}", len(months), f"{np.median(n2):.3f}",
                     f"{n2.min():.3f}–{n2.max():.3f}",
                     f"{np.median([m[2] for m in months]):+.0f}",
                     f"{np.median([m[3] for m in months]):.3f}",
                     f"{np.median([m[4] for m in months]):.2f}"])
    md += [table(rows, ["gauge", "months", "n", "N2 ratio median", "range", "N2 ΔG median (°)",
                        "M2 ratio median", "Q1 ratio median"]), ""]
    for code, months in history.items():
        md += [f"{code} by month, N2 / M2 / O1 ratio and M2 ΔG: "
               + ", ".join(f"{m[0]} {m[1]:.2f}/{m[3]:.3f}/{m[5]:.2f}/{m[6]:+.0f}°" for m in months), ""]

    # F. Datum
    md += ["## F. Datum", "",
           "Means over the year (2024-07..2025-07, the same instants). Gauge series are relative "
           "to Kartverket's MSL (1996–2014); NN2000−MSL from `locationlevels`.", ""]
    rows = []
    cache = Path("data/kartverket")
    cache.mkdir(parents=True, exist_ok=True)
    offsets, to_nn = [], []
    for code in codes:
        g = next(x for x in gauges if x["code"] == code)
        lv = levels(code, g["lat"], g["lon"], cache)
        nn = (lv["NN2000"] - lv["MSL"]) / 100.0 if "NN2000" in lv and "MSL" in lv else np.nan
        G, N, t, a, b = year_utide[code]
        gmean, nmean = a.mean(), b.mean()
        offsets.append(nmean - gmean)
        to_nn.append(nmean - nn)
        rows.append([code, f"{gmean:+.3f}", f"{nmean:+.3f}", f"{nmean - gmean:+.3f}",
                     f"{get(code, 'june', 'Z0', 'dz'):+.3f}", f"{get(code, 'january', 'Z0', 'dz'):+.3f}",
                     f"{get(code, '2026', 'Z0', 'dz'):+.3f}",
                     f"{nn:+.3f}", f"{nmean - nn:+.3f}"])
    md += [table(rows, ["gauge", "gauge mean above MSL", "NorKyst mean", "NorKyst − gauge (year)",
                        "june 2025", "january 2025", "2026-04..10", "NN2000 − MSL", "NorKyst mean − NN2000"]), "",
           f"NorKyst − gauge over the year: median {np.median(offsets):+.3f} m, "
           f"range {min(offsets):+.3f} to {max(offsets):+.3f} m. NorKyst − NN2000: median "
           f"{np.median(to_nn):+.3f} m, range {min(to_nn):+.3f} to {max(to_nn):+.3f} m.", ""]

    # G. Pressure
    md += ["## G. Inverse barometer", "",
           "Daily means of the non-tidal residual (UTide year fit removed) against ERA5 sea-level "
           "pressure at the gauge (Open-Meteo archive). Slopes in cm/hPa; r the correlation. The "
           "static inverse barometer is −0.99 cm/hPa.", ""]
    rows = []
    pcache = Path("data/pressure")
    slopes = []
    for code in codes:
        g = next(x for x in gauges if x["code"] == code)
        G, N, t, a, b = year_utide[code]
        rg = utide_residual(t, a, lat_of[code])
        rn = utide_residual(t, b, lat_of[code])
        tp, p = era5_pressure(code, g["lat"], g["lon"], *WINDOWS["year"], pcache)
        ip = np.searchsorted(tp, t)
        ok = (ip < len(tp)) & (tp[np.minimum(ip, len(tp) - 1)] == t)
        pd_t, pd = daily_means(t[ok], p[ip[ok]])
        _, gd = daily_means(t[ok], rg[ok])
        _, nd = daily_means(t[ok], rn[ok])
        bg, r_g = slope(pd, 100 * gd)
        bn, r_n = slope(pd, 100 * nd)
        bd, r_d = slope(pd, 100 * (gd - nd))
        _, r_gn = slope(gd, nd)
        slopes.append((bg, bn, bd))
        rows.append([code, len(pd_t), f"{bg:+.2f} ({r_g:+.2f})", f"{bn:+.2f} ({r_n:+.2f})",
                     f"{bd:+.2f} ({r_d:+.2f})", f"{r_gn:.2f}",
                     f"{100 * np.std(gd):.1f}", f"{100 * np.std(nd):.1f}", f"{100 * np.std(gd - nd):.1f}"])
    s = np.array(slopes)
    md += [table(rows, ["gauge", "days", "gauge slope (r)", "NorKyst slope (r)", "gauge − NorKyst slope (r)",
                        "r(gauge, NorKyst)", "σ gauge (cm)", "σ NorKyst (cm)", "σ difference (cm)"]), "",
           f"Median slopes: gauge {np.median(s[:, 0]):+.2f}, NorKyst {np.median(s[:, 1]):+.2f}, "
           f"gauge − NorKyst {np.median(s[:, 2]):+.2f} cm/hPa.", ""]

    # H. The best-estimate aggregation's daily stitching
    import utide

    md += ["## H. Daily stitching of the best-estimate aggregation", "",
           "The aggregation joins daily forecasts at 00 UTC. A jump there would put energy at 24 h "
           "(S1) and its annual sidebands (K1, P1). Year fit with S1 (UTide: M2 S2 N2 K2 K1 O1 P1 Q1 "
           "S1 M4) on the same instants; RMS of the hourly step of the residual into 00 UTC against "
           "the median over the other hours.", ""]
    rows = []
    const = ["M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1", "S1", "M4"]
    for code in codes:
        G, N, t, a, b = year_utide[code]
        cells = [code]
        for v in (a, b):
            c, d = utide_fit(t, v, lat_of[code], const)
            r = v - utide.reconstruct(t.astype("int64").astype("datetime64[s]"), c, verbose=False).h
            ok = np.diff(t) == 3600
            hour = ((t[1:] % 86400) // 3600)[ok]
            step = np.diff(r)[ok]
            at0 = np.sqrt(np.mean(step[hour == 0] ** 2))
            other = np.median([np.sqrt(np.mean(step[hour == h] ** 2)) for h in range(1, 24)])
            cells += [f"{d['S1'][0]:.4f}", f"{100 * at0:.2f} / {100 * other:.2f}"]
        rows.append(cells)
    md += [table(rows, ["gauge", "gauge S1 (m)", "gauge step 00Z / other (cm)",
                        "NorKyst S1 (m)", "NorKyst step 00Z / other (cm)"]), ""]

    (out / "tables.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {out / 'tables.md'}")
    plots(codes, lat_of, get, history, out)


def plots(codes, lat_of, get, history, out):
    import matplotlib
    import matplotlib.ticker

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # The reference palette's first three categorical slots (validated
    # all-pairs, light mode); markers are the secondary encoding
    style = {
        "june": dict(color="#2a78d6", marker="o", label="June 2025 (30 d)"),
        "year": dict(color="#eb6834", marker="s", label="2024-07 to 2025-07"),
        "2026": dict(color="#1baf7a", marker="^", label="2026-04 to 2026-10 (new tide file)"),
    }
    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": muted, "axes.labelcolor": ink,
                         "xtick.color": muted, "ytick.color": muted, "axes.titlecolor": ink})
    lats = np.array([lat_of[c] for c in codes])
    for key, ylabel, fname, ref, log in [
        ("ratio", "NorKyst / gauge amplitude", "ratios_latitude.png", 1.0, True),
        ("dg_deg", "ΔG, NorKyst − gauge (°)", "phases_latitude.png", 0.0, False),
    ]:
        fig, axes = plt.subplots(2, 4, figsize=(11, 5.2), sharex=True, constrained_layout=True)
        for ax, name in zip(axes.flat, MAIN + ["K2"]):
            ax.axhline(ref, color=muted, lw=1)
            for window, st in style.items():
                if name == "K2" and window not in LONG_WINDOWS:
                    continue
                y = np.array([get(c, window, name, key) for c in codes])
                ax.plot(lats, y, ls="none", ms=5, mfc=st["color"], mec="#fcfcfb", mew=0.8,
                        marker=st["marker"], label=st["label"])
            ax.set_title(name + (" (≥ 6 months only)" if name == "K2" else ""), loc="left", fontsize=10)
            if log:
                ax.set_yscale("log")
                ax.set_ylim(0.1, 3.0)
                ax.set_yticks([0.1, 0.2, 0.5, 1, 2])
                ax.set_yticklabels(["0.1", "0.2", "0.5", "1", "2"])
                ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            else:
                ax.set_ylim(-90, 90)
            ax.grid(color=grid, lw=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        for ax in axes[1]:
            ax.set_xlabel("gauge latitude (°N)")
        for ax in axes[:, 0]:
            ax.set_ylabel(ylabel)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper right", ncol=3, frameon=False)
        fig.suptitle(f"NorKyst-800 v3 against {len(codes)} Kartverket gauges", x=0.01, ha="left",
                     fontsize=11, color=ink)
        fig.savefig(out / fname, dpi=150, facecolor="#fcfcfb")
        plt.close(fig)
        print(f"wrote {out / fname}")

    if history:
        fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True, constrained_layout=True)
        for ax, (k, name, lim) in zip(axes, [(1, "N2", (0, 1.2)), (3, "M2", (0.85, 1.15)),
                                             (5, "O1", (0.4, 2.3))]):
            ax.axhline(1.0, color=muted, lw=1)
            for (code, months), (color, marker) in zip(history.items(),
                                                       [("#2a78d6", "o"), ("#eb6834", "s"), ("#1baf7a", "^")]):
                # Mid-month, so the switch line falls between February and March
                x = [datetime.strptime(m[0] + "-15", "%Y-%m-%d") for m in months]
                ax.plot(x, [m[k] for m in months], color=color, lw=2, marker=marker, ms=5,
                        mec="#fcfcfb", label=code)
            ax.axvline(datetime(2026, 2, 25), color=muted, lw=1, ls="--")
            ax.set_ylim(*lim)
            ax.set_ylabel(f"{name}, NorKyst / gauge")
            ax.grid(color=grid, lw=0.6)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        axes[0].legend(frameon=False, ncol=3, loc="upper left")
        axes[0].text(datetime(2026, 2, 15), 0.55, "new tide file created 2026-02-25", color=muted,
                     fontsize=8, ha="right")
        axes[0].set_title("NorKyst-800 v3 by calendar month (UTide, the same instants)", loc="left", fontsize=10)
        fig.savefig(out / "n2_monthly.png", dpi=150, facecolor="#fcfcfb")
        plt.close(fig)
        print(f"wrote {out / 'n2_monthly.png'}")


if __name__ == "__main__":
    main(sys.argv[1:])

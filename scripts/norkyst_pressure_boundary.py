#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "utide",
#     "netCDF4",
# ]
# ///
"""
Does NorKyst-800's ζ carry the inverse barometer relative to its own open
boundary only? (notes/norkyst-issues.md §6 "Cause".)

    uv run scripts/norkyst_pressure_boundary.py [out=output/norkyst_issues/pressure_boundary.md]

NorKyst runs ROMS with Shchepetkin ubar/vbar and Chapman ζ open boundaries,
where `PRESS_COMPENSATE` does nothing (ROMS applies it in Flather only). If
that is why its ζ responds at only −0.22 cm/hPa, then quasi-statically

    ζ_NorKyst ≈ −(p_local − p_boundary)/(ρ₀ g),
    gauge − NorKyst ≈ −(p_boundary − p₀)/(ρ₀ g) + wind + dynamics,

so gauge − NorKyst should regress on the boundary pressure at ≈ −1 cm/hPa
and better than on the local pressure, and NorKyst on (p_local − p_boundary)
at ≈ −1 cm/hPa.

Inputs are what scripts/norkyst_tide_check.sh fetched (gauge and NorKyst
series for 2024-07..2025-07, the cached grid) plus:
- the open-boundary cells: the first interior row/column of the grid that
  is wet in `mask_rho` of norkyst_grd_v31.nc (the tide file has values on
  land too), every STRIDE-th one;
- ERA5 sea-level pressure there (Open-Meteo archive, cached in
  data/pressure/boundary/).
Non-tidal residuals are UTide year fits removed, regressed as daily means.
"""

import csv
import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True  # no __pycache__ in scripts/
sys.path.insert(0, str(Path(__file__).parent))
import norkyst_tide_report as rep  # noqa: E402

GRID = "https://thredds.met.no/thredds/dodsC/fou-hi/norkystv3_his_files/static_nc_files/norkyst_grd_v31.nc"
STRIDE = 60  # ≈ 48 km along the boundary; ERA5 is ≈ 31 km
IB = 100.0 / (1025.0 * 9.81) * 100.0  # static inverse barometer, cm/hPa
START, END = rep.WINDOWS["year"]


def boundary_points(cache):
    """Every STRIDE-th wet cell of the first interior row/column: (side, lat, lon)."""
    if cache.exists():
        return [tuple(x) for x in json.loads(cache.read_text(encoding="utf-8"))]
    import netCDF4

    g = np.load("data/norkyst_gauges/norkyst_grid.npz")
    lon, lat = g["lon"], g["lat"]
    mask = netCDF4.Dataset(GRID)["mask_rho"]
    ny, nx = mask.shape
    sides = {"south": (1, slice(None)), "north": (ny - 2, slice(None)),
             "west": (slice(None), 1), "east": (slice(None), nx - 2)}
    pts = []
    for side, (j, i) in sides.items():
        wet = np.flatnonzero(np.asarray(mask[j, i]) > 0)
        print(f"{side}: {wet.size} wet boundary cells")
        for k in wet[::STRIDE]:
            jj = j if isinstance(j, int) else k
            ii = i if isinstance(i, int) else k
            pts.append((side, round(float(lat[jj, ii]), 4), round(float(lon[jj, ii]), 4)))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(pts), encoding="utf-8")
    return pts


def era5_many(pts, cache_dir):
    """Hourly ERA5 sea-level pressure at many points: Unix times, [point][hour] (hPa)."""
    last = datetime.fromtimestamp(rep.unix(END) - 86400, timezone.utc).strftime("%Y-%m-%d")
    out = []
    t = None
    for b in range(0, len(pts), 10):
        batch = pts[b:b + 10]
        cache = cache_dir / f"batch{STRIDE}_{b:03d}_{START}_{END}.json"
        if not cache.exists():
            url = ("https://archive-api.open-meteo.com/v1/archive?"
                   f"latitude={','.join(str(p[1]) for p in batch)}"
                   f"&longitude={','.join(str(p[2]) for p in batch)}"
                   f"&start_date={START}&end_date={last}&hourly=pressure_msl&timezone=GMT")
            for attempt in range(5):
                try:
                    with urllib.request.urlopen(url, timeout=300) as r:
                        text = r.read().decode("utf-8")
                    break
                except Exception as e:  # rate limit or a slow archive
                    print(f"  batch {b}: {e}; retrying")
                    time.sleep(30 * (attempt + 1))
            else:
                raise SystemExit(f"Open-Meteo failed for batch {b}")
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(text, encoding="utf-8")
        data = json.loads(cache.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = [data]
        for d in data:
            tt = np.array([datetime.strptime(s, "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc).timestamp()
                           for s in d["hourly"]["time"]])
            t = tt if t is None else t
            assert np.array_equal(t, tt)
            out.append([np.nan if x is None else x for x in d["hourly"]["pressure_msl"]])
    return t, np.array(out, dtype=float)


def distance_km(lon1, lat1, lon2, lat2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def ols(y, *xs):
    """Least-squares coefficients of y on xs (with intercept) and R²."""
    X = np.column_stack([np.ones_like(y)] + list(xs))
    m = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    c, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
    res = y[m] - X[m] @ c
    r2 = 1 - res.var() / y[m].var()
    return c[1:], r2


def main(argv):
    kw = dict(a.split("=", 1) for a in argv if "=" in a)
    out = Path(kw.get("out", "output/norkyst_issues/pressure_boundary.md"))
    with open("scripts/norkyst_tide_gauges.csv", newline="", encoding="utf-8") as f:
        gauges = list(csv.DictReader(f))
    pts = boundary_points(Path("data/pressure/boundary/points.json"))
    print(f"{len(pts)} boundary points (every {STRIDE}th wet cell)")
    tb, pb = era5_many(pts, Path("data/pressure/boundary"))
    blat = np.array([p[1] for p in pts])
    blon = np.array([p[2] for p in pts])
    db_t, _ = rep.daily_means(tb, pb[0])
    pb_d = np.array([rep.daily_means(tb, row)[1] for row in pb])  # [point][day]
    p_mean = np.nanmean(pb_d, axis=0)
    side_of = np.array([p[0] for p in pts])
    # The south edge is wet in the Kattegat and by the Kola coast at the east
    # corner; the Kola stretch goes with the Barents mean
    kattegat = (side_of == "south") & (blon < 20)
    barents = (side_of == "east") | ((side_of == "south") & (blon > 20))
    side_means = {"North Sea (west)": np.nanmean(pb_d[side_of == "west"], axis=0),
                  "Norwegian Sea (north)": np.nanmean(pb_d[side_of == "north"], axis=0),
                  "Barents (east)": np.nanmean(pb_d[barents], axis=0),
                  "Kattegat (south)": np.nanmean(pb_d[kattegat], axis=0)}
    side_rows = []

    rows, summary = [], []
    for g in gauges:
        code, lat, lon = g["code"], float(g["lat"]), float(g["lon"])
        _, tg, vg = rep.read_series(Path("data/tide_gauges/coast") / f"{code}_obs.txt")
        _, tn, vn = rep.read_series(Path("data/norkyst_gauges/2024-07_2025-07") / f"{code}.txt")
        t, a, b = rep.common((tg, vg), (tn, vn), START, END)
        rg = rep.utide_residual(t, a, lat)
        rn = rep.utide_residual(t, b, lat)
        tp, p = rep.era5_pressure(code, lat, lon, START, END, Path("data/pressure"))
        ip = np.searchsorted(tp, t)
        ok = (ip < len(tp)) & (tp[np.minimum(ip, len(tp) - 1)] == t)
        day, pl = rep.daily_means(t[ok], p[ip[ok]])
        _, gd = rep.daily_means(t[ok], 100 * rg[ok])
        _, nd = rep.daily_means(t[ok], 100 * rn[ok])
        # Boundary pressure on the same days: the whole-boundary mean and
        # the nearest boundary points (inverse-distance weights over the 5)
        k = np.searchsorted(db_t, day)
        assert np.array_equal(db_t[k], day)
        dist = distance_km(lon, lat, blon, blat)
        near = np.argsort(dist)[:5]
        w = 1.0 / dist[near]
        pn = (w[:, None] * pb_d[near][:, k]).sum(axis=0) / w.sum()
        pm = p_mean[k]
        diff = gd - nd

        (s_loc,), r2_loc = ols(diff, pl)
        (s_mean,), r2_mean = ols(diff, pm)
        (s_near,), r2_near = ols(diff, pn)
        (c_loc, c_near), r2_both = ols(diff, pl, pn)
        (s_nk_loc,), r2_nk_loc = ols(nd, pl)
        (s_nk_rel,), r2_nk_rel = ols(nd, pl - pn)
        (s_nk_relm,), r2_nk_relm = ols(nd, pl - pm)
        # Control: the gauge itself on local and boundary pressure. If the
        # gauge also follows the boundary mean better, the gauge − NorKyst
        # comparison does not single out NorKyst's boundary.
        (s_g_loc,), r2_g_loc = ols(gd, pl)
        (s_g_bnd,), r2_g_bnd = ols(gd, pm)
        summary.append((s_loc, s_mean, s_near, c_loc, c_near, s_nk_loc, s_nk_rel, s_nk_relm))
        rows.append([code, f"{lat:.2f}", f"{dist[near[0]]:.0f}", len(day),
                     f"{s_loc:+.2f} ({r2_loc:.2f})", f"{s_mean:+.2f} ({r2_mean:.2f})",
                     f"{s_near:+.2f} ({r2_near:.2f})", f"{c_loc:+.2f} / {c_near:+.2f} ({r2_both:.2f})",
                     f"{s_nk_loc:+.2f} ({r2_nk_loc:.2f})", f"{s_nk_rel:+.2f} ({r2_nk_rel:.2f})",
                     f"{s_nk_relm:+.2f} ({r2_nk_relm:.2f})",
                     f"{np.nanstd(pl - pn):.1f}", f"{np.nanstd(pl - pm):.1f}", f"{np.nanstd(pl):.1f}",
                     f"{s_g_loc:+.2f} ({r2_g_loc:.2f})", f"{s_g_bnd:+.2f} ({r2_g_bnd:.2f})"])
        cells = [code]
        for ps in side_means.values():
            (sl,), r2 = ols(diff, ps[k])
            cells.append(f"{sl:+.2f} ({r2:.2f})")
        coefs, r2 = ols(diff, *(ps[k] for ps in side_means.values()))
        cells.append(" / ".join(f"{c:+.2f}" for c in coefs) + f" ({r2:.2f})")
        side_rows.append(cells)
        print(code, rows[-1][4], rows[-1][5], rows[-1][8], rows[-1][10], rows[-1][12:])

    s = np.array(summary)
    med = np.median(s, axis=0)
    md = ["# NorKyst ζ and the pressure on its open boundary", "",
          f"Generated by `scripts/norkyst_pressure_boundary.py` on {datetime.now(timezone.utc):%Y-%m-%d}. "
          f"Window {START}..{END}; daily means of the non-tidal residual (UTide year fit removed); "
          f"ERA5 sea-level pressure (Open-Meteo archive). {len(pts)} open-boundary points, every "
          f"{STRIDE}th wet cell of the first interior row/column. Slopes in cm/hPa, R² in brackets. "
          f"The static inverse barometer is −{IB:.2f} cm/hPa.", "",
          "- p_loc: at the gauge. p_bnd: the whole-boundary mean. p_near: inverse-distance mean of "
          "the 5 nearest boundary points.",
          "- If NorKyst's ζ ≈ −(p_loc − p_boundary)/(ρ₀g), gauge − NorKyst regresses on the boundary "
          "pressure at ≈ −1, and NorKyst on p_loc − p_boundary at ≈ −1.", "",
          rep.table(rows, ["gauge", "lat", "nearest bnd (km)", "days",
                           "gauge−NK on p_loc", "gauge−NK on p_bnd", "gauge−NK on p_near",
                           "gauge−NK on p_loc, p_near", "NK on p_loc", "NK on p_loc − p_near",
                           "NK on p_loc − p_bnd", "σ(p_loc − p_near) hPa", "σ(p_loc − p_bnd) hPa",
                           "σ(p_loc) hPa", "gauge on p_loc", "gauge on p_bnd"]), "",
          f"Medians: gauge − NorKyst on p_loc {med[0]:+.2f}, on p_bnd {med[1]:+.2f}, on p_near "
          f"{med[2]:+.2f}; both: p_loc {med[3]:+.2f}, p_near {med[4]:+.2f}. NorKyst on p_loc "
          f"{med[5]:+.2f}, on p_loc − p_near {med[6]:+.2f}, on p_loc − p_bnd {med[7]:+.2f}.", "",
          "## Per boundary", "",
          "gauge − NorKyst on the mean pressure of each open boundary (Kattegat: the wet stretch of "
          "the south edge; Barents: the east edge and the Kola stretch of the south edge), and on all "
          "four together (coefficients in the column order).", "",
          rep.table(side_rows, ["gauge"] + list(side_means) + ["all four"]), ""]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1:])

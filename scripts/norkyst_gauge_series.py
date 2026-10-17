#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "netCDF4",
# ]
# ///
"""
NorKyst-800 sea level at tide gauges, in the gauge-file format.

For each gauge in a CSV (`name,code,lat,lon`), snap to the nearest wet
NorKyst cell (as `examples/norkyst_boundary_tides.rs` does for `points=`)
and write NorKyst's hourly `zeta` over `[from, to)` as a text file that
`dg_rs::io::read_tide_gauge_file` reads, so that the gauge and NorKyst go
through the same reader and the same harmonic fit (`examples/tide_gauge_fit.rs
<gauge> compare=<this file>`).

    uv run scripts/norkyst_gauge_series.py scripts/norkyst_tide_gauges.csv \\
        2025-06-01 2025-07-01 data/norkyst_gauges/2025-06 [codes=BOO,MSU] [max_snap_m=3000] [url=<OPeNDAP>]

Output: `<out>/<code>.txt`, one line per hour `ISO-time zeta(m)`, with the
cell's position, depth and snap distance in the header. NorKyst's `zeta`
is an instantaneous value (`cell_methods: ocean_time: point`); the
aggregation is checked for gaps, which are reported and left out (never
interpolated). The lon/lat/h grid is cached in `<out>/../norkyst_grid.npz`.
"""

import csv
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import netCDF4
import numpy as np

URL = "https://thredds.met.no/thredds/dodsC/fou-hi/norkystv3_800m_m00_be"
FILL = -32767
# Hours per DAP request
BLOCK = 24 * 31


def retry(what, f, attempts=6):
    for k in range(1, attempts + 1):
        try:
            return f()
        except Exception as e:  # THREDDS answers 503 / drops connections under load
            if k == attempts:
                raise
            print(f"  {what}: {e}; retrying in {10 * k} s", file=sys.stderr)
            time.sleep(10 * k)


def distance(lon1, lat1, lon2, lat2):
    """Equirectangular distance (m), as in the Rust examples."""
    r = 6_371_000.0
    x = np.radians(lon2 - lon1) * np.cos(np.radians(0.5 * (lat1 + lat2)))
    y = np.radians(lat2 - lat1)
    return r * np.hypot(x, y)


def unix(day):
    return datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()


def iso(t):
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main():
    pos = [a for a in sys.argv[1:] if "=" not in a]
    kw = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    if len(pos) != 4:
        print(__doc__)
        sys.exit(1)
    gauges_csv, start, end, out = pos
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    codes = set(kw["codes"].split(",")) if "codes" in kw else None
    max_snap = float(kw.get("max_snap_m", 3000))
    url = kw.get("url", URL)

    with open(gauges_csv, newline="", encoding="utf-8") as f:
        gauges = [
            g for g in csv.DictReader(f) if not codes or g["code"] in codes
        ]

    ds = retry("open", lambda: netCDF4.Dataset(url))
    tvar = ds.variables["time"]
    assert tvar.units == "seconds since 1970-01-01 00:00:00", tvar.units
    times = retry("time", lambda: np.asarray(tvar[:], dtype=float))
    t0, t1 = unix(start), unix(end)
    sel = np.nonzero((times >= t0) & (times < t1))[0]
    if len(sel) == 0:
        sys.exit("no NorKyst times in the window")
    k0, k1 = sel[0], sel[-1] + 1
    window = times[k0:k1]
    steps = np.diff(window)
    expected = int(round((t1 - t0) / 3600))
    print(f"{url}\n  {start} to {end}: {len(window)} of {expected} hourly times")
    for k in np.nonzero(steps != 3600)[0]:
        print(f"  gap: {iso(window[k])} -> {iso(window[k + 1])} ({steps[k] / 3600:.0f} h)")

    # One cached grid per dataset (the default keeps its old name)
    tag = "" if url == URL else "_" + url.rstrip("/").split("/")[-1].split(".")[0]
    grid_file = out.parent / f"norkyst_grid{tag}.npz"
    if grid_file.exists():
        grid = np.load(grid_file)
        lon, lat, h = grid["lon"], grid["lat"], grid["h"]
    else:
        print("  reading the lon/lat/h grid")
        lon = retry("lon", lambda: np.asarray(ds.variables["lon"][:], dtype=float))
        lat = retry("lat", lambda: np.asarray(ds.variables["lat"][:], dtype=float))
        h = retry("h", lambda: np.ma.filled(ds.variables["h"][:], np.nan).astype(float))
        np.savez_compressed(grid_file, lon=lon, lat=lat, h=h)

    zeta = ds.variables["zeta"]
    zeta.set_auto_maskandscale(False)
    scale = float(zeta.scale_factor)
    offset = float(zeta.add_offset)
    for g in gauges:
        glat, glon = float(g["lat"]), float(g["lon"])
        # Window of ±20 cells around the nearest grid point, wet mask from
        # the first time step
        d_all = distance(glon, glat, lon, lat)
        jc, ic = np.unravel_index(np.argmin(d_all), d_all.shape)
        j0, i0 = max(jc - 20, 0), max(ic - 20, 0)
        j1, i1 = min(jc + 21, lon.shape[0]), min(ic + 21, lon.shape[1])
        first = retry("mask", lambda: zeta[k0, j0:j1, i0:i1])
        d = np.where(first != FILL, d_all[j0:j1, i0:i1], np.inf)
        jw, iw = np.unravel_index(np.argmin(d), d.shape)
        snap = d[jw, iw]
        if not math.isfinite(snap) or snap > max_snap:
            print(f"  {g['code']}: no wet cell within {max_snap:.0f} m, skipped")
            continue
        j, i = j0 + jw, i0 + iw
        started = time.time()
        raw = np.empty(k1 - k0, dtype=np.int16)
        for b in range(k0, k1, BLOCK):
            e = min(b + BLOCK, k1)
            raw[b - k0 : e - k0] = retry(g["code"], lambda: zeta[b:e, j, i])
        wet = raw != FILL
        values = offset + scale * raw.astype(float)
        path = out / f"{g['code']}.txt"
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(f"# NorKyst-800 v3 zeta at the nearest wet cell to {g['name']} ({g['code']}), {start} to {end}, hourly\n")
            f.write(f"# source: {url} zeta[time, Y={j}, X={i}]\n")
            f.write(f"# station: NorKyst {g['name']}\n")
            f.write(f"# id: NK-{g['code']}\n")
            f.write(f"# latitude: {lat[j, i]:.5f}\n")
            f.write(f"# longitude: {lon[j, i]:.5f}\n")
            f.write(f"# depth: {h[j, i]:.1f}\n")
            f.write(f"# snap: {snap:.0f} m from the gauge at {glat}, {glon}\n")
            f.write("# datum: NorKyst zeta (sea surface height above geoid)\n")
            f.write("# units: m\n")
            f.write("# columns: datetime water_level(m)\n")
            for t, v, ok in zip(window, values, wet):
                if ok:
                    f.write(f"{iso(t)} {v:.3f}\n")
        print(
            f"  {g['code']} {g['name']}: cell Y={j} X={i} ({lon[j, i]:.4f}, {lat[j, i]:.4f}),"
            f" {snap:.0f} m away, {h[j, i]:.0f} m deep; {wet.sum()} samples,"
            f" {time.time() - started:.0f} s -> {path}"
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "netCDF4"]
# ///
"""NorKyst v3 (best estimate) ISO-week mean temperature at the farms, at one depth.

    uv run scripts/norkyst_farm_weeks.py data/farm_temps/froya.txt <out.txt> [depth_index=3] [stride_hours=3]

The depth index is into the aggregation's levels (0 1 2 3 5 7 10 15 25 50 … m),
so 3 is 3 m. About 15 min for 2025-01 to now over OPeNDAP.
Reads the farms' `localityNo lat lon year week T name` lines, takes the nearest
wet NorKyst cell to each farm inside the Frøya window (Y 584+121, X 1104+121),
and writes `localityNo year week T_farm T_norkyst n_samples dist_km h`.
"""
import sys
from collections import defaultdict
from datetime import datetime, timezone

import netCDF4 as nc
import numpy as np

URL = "https://thredds.met.no/thredds/dodsC/fou-hi/norkystv3_800m_m00_be"
Y0, X0, N = 584, 1104, 121
farms_path, out_path = sys.argv[1], sys.argv[2]
k = int(sys.argv[3]) if len(sys.argv) > 3 else 3
stride = int(sys.argv[4]) if len(sys.argv) > 4 else 3

farm_obs = {}
farm_pos = {}
for line in open(farms_path, encoding="utf-8"):
    if line.startswith("#"):
        continue
    no, la, lo, y, w, t, _ = line.split(maxsplit=6)
    farm_obs[(int(no), int(y), int(w))] = float(t)
    farm_pos[int(no)] = (float(la), float(lo))

d = nc.Dataset(URL)
lon = d["lon"][Y0:Y0 + N, X0:X0 + N]
lat = d["lat"][Y0:Y0 + N, X0:X0 + N]
h = np.ma.filled(d["h"][Y0:Y0 + N, X0:X0 + N], np.nan)
print("depth", float(d["depth"][k]), "m")
# Wet cells: finite temperature at this depth at the first time.
t_first = np.ma.filled(d["temperature"][0, k, Y0:Y0 + N, X0:X0 + N].astype(float), np.nan)
wet = np.isfinite(t_first)
cell = {}
for no, (la, lo) in farm_pos.items():
    dist = np.hypot((lon - lo) * np.cos(np.radians(la)), lat - la) * 111.2
    dist = np.where(wet, dist, np.inf)
    j, i = np.unravel_index(np.argmin(dist), dist.shape)
    cell[no] = (j, i, float(dist[j, i]), float(h[j, i]))
jj = np.array([c[0] for c in cell.values()])
ii = np.array([c[1] for c in cell.values()])
nos = list(cell)

times = d["time"][:]
t0 = datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp()
idx = np.where(times >= t0)[0][::stride]
sums = defaultdict(float)
counts = defaultdict(int)
chunk = 240 // stride * stride  # about 10 days per request
for a in range(0, len(idx), chunk):
    sel = idx[a:a + chunk]
    lo_t, hi_t = sel[0], sel[-1]
    block = d["temperature"][lo_t:hi_t + 1:stride, k, Y0:Y0 + N, X0:X0 + N]
    block = np.ma.filled(block.astype(float), np.nan)
    for r, ti in enumerate(range(lo_t, hi_t + 1, stride)):
        iso = datetime.fromtimestamp(float(times[ti]), timezone.utc).isocalendar()
        vals = block[r, jj, ii]
        for no, v in zip(nos, vals):
            if np.isfinite(v):
                sums[(no, iso[0], iso[1])] += v
                counts[(no, iso[0], iso[1])] += 1
    print(datetime.fromtimestamp(float(times[hi_t]), timezone.utc).date(), flush=True)

with open(out_path, "w") as f:
    f.write(f"# localityNo year week T_farm T_norkyst n dist_km h  (NorKyst BE at {float(d['depth'][k])} m)\n")
    for key, t_farm in sorted(farm_obs.items()):
        n = counts.get(key, 0)
        if n == 0:
            continue
        no = key[0]
        f.write(f"{no} {key[1]} {key[2]} {t_farm:.3f} {sums[key] / n:.3f} {n} "
                f"{cell[no][2]:.2f} {cell[no][3]:.1f}\n")
print("->", out_path)

#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "netCDF4"]
# ///
"""Farm weekly temperatures as hourly point observations over a run's window,
for snapshot_sample_3d (unix_time lon lat value), and NorKyst at the same
points and hours at 0 and 3 m (nearest wet cell).

    uv run scripts/farm_obs_window.py <farm_temps.txt> <norkyst_nest.nc> <out_prefix> <start_iso> <hours> [skip_hours=6]
Writes <out_prefix>.obs.txt (value = the farm's ISO-week mean) and
<out_prefix>.norkyst.txt (unix lon lat farm nk0 nk3 localityNo dist_km).
"""
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
from netCDF4 import Dataset

farm_path, nk_path, prefix, start_iso, hours = sys.argv[1:6]
skip = float(sys.argv[6]) if len(sys.argv) > 6 else 6.0
start = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
hours = float(hours)

weekly = {}
where = {}
for line in open(farm_path):
    if line.startswith("#"):
        continue
    p = line.split()
    no, lat, lon, y, w, t = int(p[0]), float(p[1]), float(p[2]), int(p[3]), int(p[4]), float(p[5])
    weekly[(no, y, w)] = t
    where[no] = (lon, lat)

nk = Dataset(nk_path)
lon, lat = nk["lon"][:], nk["lat"][:]
t_nk = np.asarray(nk["time"][:], dtype=float)
units = nk["time"].units
assert units.startswith("seconds since 1970"), units
depths = np.asarray(nk["depth"][:], dtype=float)
k0, k3 = int(np.argmin(np.abs(depths - 0))), int(np.argmin(np.abs(depths - 3)))
temp = nk["temperature"]
wet = np.isfinite(np.ma.filled(temp[0, k3].astype(float), np.nan))

times = []
h = skip
while h <= hours + 1e-9:
    times.append(start + timedelta(hours=h))
    h += 1.0
unix = np.array([t.timestamp() for t in times])
i_lo = max(int(np.searchsorted(t_nk, unix[0])) - 1, 0)
i_hi = min(int(np.searchsorted(t_nk, unix[-1])) + 1, len(t_nk) - 1)
block0 = np.ma.filled(temp[i_lo:i_hi + 1, k0].astype(float), np.nan)
block3 = np.ma.filled(temp[i_lo:i_hi + 1, k3].astype(float), np.nan)
tt = t_nk[i_lo:i_hi + 1]

obs = open(f"{prefix}.obs.txt", "w")
nko = open(f"{prefix}.norkyst.txt", "w")
obs.write(f"# BarentsWatch farm weekly means as hourly observations from {start_iso} +{skip} h\n")
nko.write("# unix lon lat farm nk0 nk3 localityNo dist_km\n")
n_farms = 0
for no, (plon, plat) in sorted(where.items()):
    d = np.hypot((lon - plon) * np.cos(np.deg2rad(plat)), lat - plat) * 111.2
    d = np.where(wet, d, np.inf)
    j, i = np.unravel_index(np.argmin(d), d.shape)
    if d[j, i] > 1.2:
        continue
    rows = []
    for t, u in zip(times, unix):
        y, w, _ = t.isocalendar()
        val = weekly.get((no, y, w))
        if val is None:
            continue
        a = np.interp(u, tt, block0[:, j, i])
        b = np.interp(u, tt, block3[:, j, i])
        rows.append((u, val, a, b))
    if not rows:
        continue
    n_farms += 1
    for u, val, a, b in rows:
        obs.write(f"{u:.0f} {plon:.6f} {plat:.6f} {val:.3f}\n")
        nko.write(f"{u:.0f} {plon:.6f} {plat:.6f} {val:.3f} {a:.3f} {b:.3f} {no} {d[j, i]:.2f}\n")
print(f"{n_farms} farms within 1.2 km of a wet NorKyst cell, {len(times)} hours "
      f"(NorKyst levels {depths[k0]:g} and {depths[k3]:g} m)")

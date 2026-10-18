#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""Robust farm-against-NorKyst statistics from norkyst_farm_weeks.py's output.

    uv run scripts/farm_temperature_stats.py <norkyst_farm_weeks output> [max_dist_km=1.2] [min_samples=50]
Screens: whole weeks only (n >= min_samples of 56 three-hourly samples), farms
within max_dist of a wet NorKyst cell, and per week a 3-MAD cut on
d = T_farm - T_norkyst (at least 0.6 °C). Reports medians, not means.
"""
import sys
from collections import defaultdict
from datetime import date

import numpy as np

path = sys.argv[1]
max_dist = float(sys.argv[2]) if len(sys.argv) > 2 else 1.2
min_n = int(sys.argv[3]) if len(sys.argv) > 3 else 50

rows = []
for line in open(path):
    if line.startswith("#"):
        continue
    no, y, w, tf, tn, n, dist, h = line.split()
    rows.append((int(no), int(y), int(w), float(tf), float(tn), int(n), float(dist), float(h)))
print(f"{len(rows)} farm-weeks; ", end="")
rows = [r for r in rows if r[5] >= min_n and r[6] <= max_dist]
print(f"{len(rows)} after whole-week and distance <= {max_dist} km")

by_week = defaultdict(list)
for r in rows:
    by_week[(r[1], r[2])].append(r)
kept, dropped = [], 0
for key, rs in by_week.items():
    d = np.array([r[3] - r[4] for r in rs])
    med = np.median(d)
    mad = 1.4826 * np.median(np.abs(d - med))
    cut = max(3 * mad, 0.6)
    for r, di in zip(rs, d):
        if abs(di - med) <= cut:
            kept.append(r)
        else:
            dropped += 1
print(f"{dropped} farm-weeks dropped as outliers, {len(kept)} kept, "
      f"{len({r[0] for r in kept})} farms")


def month_of(y, w):
    return date.fromisocalendar(y, w, 4).strftime("%Y-%m")


def stats(rs):
    d = np.array([r[3] - r[4] for r in rs])
    q1, q3 = np.percentile(d, [25, 75])
    return len(d), np.median(d), q1, q3, np.median([r[3] for r in rs]), np.median([r[4] for r in rs])


print("\nmonth     n   median(farm-NK)  IQR            farm  NorKyst  (3 m, medians)")
months = defaultdict(list)
for r in kept:
    months[month_of(r[1], r[2])].append(r)
for m in sorted(months):
    n, md, q1, q3, tf, tn = stats(months[m])
    print(f"{m}  {n:4d}   {md:+6.2f}          [{q1:+5.2f},{q3:+5.2f}]  {tf:5.2f}  {tn:5.2f}")

print("\nThe windows:")
for y, w in ((2025, 24), (2025, 25), (2026, 23), (2026, 24)):
    rs = [r for r in kept if (r[1], r[2]) == (y, w)]
    if rs:
        n, md, q1, q3, tf, tn = stats(rs)
        print(f"{y} w{w} ({date.fromisocalendar(y, w, 1)}..): n={n} farm {tf:.2f} NorKyst {tn:.2f} "
              f"median diff {md:+.2f} [{q1:+.2f},{q3:+.2f}]")

# Week-to-week warming of the same farm: sensor offsets cancel.
key = {(r[0], r[1], r[2]): r for r in kept}
dd = defaultdict(list)
for (no, y, w), r in key.items():
    nxt = key.get((no, y, w + 1))
    if nxt:
        dd[month_of(y, w)].append(((nxt[3] - r[3]), (nxt[4] - r[4])))
print("\nWeek-to-week change, same farm: month  n  median dT_farm  median dT_NK  median(dfarm-dNK)  r")
for m in sorted(dd):
    a = np.array(dd[m])
    r = np.corrcoef(a[:, 0], a[:, 1])[0, 1] if len(a) > 2 else np.nan
    print(f"{m}  {len(a):4d}  {np.median(a[:, 0]):+6.2f}  {np.median(a[:, 1]):+6.2f}  "
          f"{np.median(a[:, 0] - a[:, 1]):+6.2f}  {r:5.2f}")

# Per-farm persistent offsets (site or sensor), and how they spread.
per_farm = defaultdict(list)
for r in kept:
    per_farm[r[0]].append(r[3] - r[4])
offs = np.array([np.median(v) for v in per_farm.values() if len(v) >= 20])
print(f"\nPer-farm median offset (farms with >= 20 weeks, n={len(offs)}): median {np.median(offs):+.2f}, "
      f"IQR [{np.percentile(offs, 25):+.2f},{np.percentile(offs, 75):+.2f}], "
      f"range [{offs.min():+.2f},{offs.max():+.2f}]")

#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""Score runs at the farms over a window: the farms' 3 m, the model's and
NorKyst's 0 and 3 m, per farm averaged over the window, medians across farms.

    uv run scripts/farm_window_stats.py <prefix.norkyst.txt> <label>,<model_d0.txt>,<model_d3.txt> ...
(model files from snapshot_sample_3d with depth=0 and depth=3 on <prefix>.obs.txt).
Farms whose farm - NorKyst(3 m) is more than max(3 MAD, 0.6 °C) from the
median are dropped (the farm data are noisy).
"""
import sys
from collections import defaultdict

import numpy as np


def key(u, lon, lat):
    return (int(round(float(u))), round(float(lon), 4), round(float(lat), 4))


nk = {}
farm_of = {}
for line in open(sys.argv[1]):
    if line.startswith("#"):
        continue
    u, lon, lat, farm, a, b, no, _ = line.split()
    k = key(u, lon, lat)
    nk[k] = (float(farm), float(a), float(b))
    farm_of[k] = int(no)

runs = []
for spec in sys.argv[2:]:
    label, p0, p3 = spec.split(",")
    m = defaultdict(dict)
    for which, path in (("0", p0), ("3", p3)):
        for line in open(path):
            if line.startswith("#"):
                continue
            u, lon, lat, _, model = line.split()
            m[key(u, lon, lat)][which] = float(model)
    runs.append((label, m))

# Keys every run sampled at both depths
common = [k for k in nk if all("0" in m.get(k, {}) and "3" in m.get(k, {}) for _, m in runs)]
per_farm = defaultdict(list)
for k in common:
    per_farm[farm_of[k]].append(k)

farm_mean = {no: np.mean([nk[k][0] for k in ks]) for no, ks in per_farm.items()}
nk3_mean = {no: np.mean([nk[k][2] for k in ks]) for no, ks in per_farm.items()}
d = np.array([farm_mean[n] - nk3_mean[n] for n in per_farm])
med = np.median(d)
cut = max(3 * 1.4826 * np.median(np.abs(d - med)), 0.6)
keep = [n for n in per_farm if abs(farm_mean[n] - nk3_mean[n] - med) <= cut]
print(f"{len(per_farm)} farms sampled by every run, {len(keep)} kept after the 3-MAD cut "
      f"({len(common)} farm-hours)")


def summary(name, values):
    v = np.array(values)
    q1, q3 = np.percentile(v, [25, 75])
    print(f"  {name:34s} median {np.median(v):+6.2f}  IQR [{q1:+5.2f},{q3:+5.2f}]")


print("\nNorKyst")
summary("farm - NorKyst 3 m", [farm_mean[n] - nk3_mean[n] for n in keep])
summary("NorKyst 0 m - 3 m", [np.mean([nk[k][1] - nk[k][2] for k in per_farm[n]]) for n in keep])
for label, m in runs:
    print(f"\n{label}")
    summary("farm - model 3 m", [farm_mean[n] - np.mean([m[k]["3"] for k in per_farm[n]]) for n in keep])
    summary("model 0 m - 3 m", [np.mean([m[k]["0"] - m[k]["3"] for k in per_farm[n]]) for n in keep])
    summary("model - NorKyst at 0 m", [np.mean([m[k]["0"] - nk[k][1] for k in per_farm[n]]) for n in keep])
    summary("model - NorKyst at 3 m", [np.mean([m[k]["3"] - nk[k][2] for k in per_farm[n]]) for n in keep])
    # The 0-3 m difference by hour of day (UTC), across farms and days
    by_hour = defaultdict(list)
    for n in keep:
        for k in per_farm[n]:
            by_hour[(k[0] // 3600) % 24].append(m[k]["0"] - m[k]["3"])
    print("  0-3 m by UTC hour: " + " ".join(f"{h:02d}:{np.median(by_hour[h]):+.2f}" for h in sorted(by_hour) if h % 3 == 0))

by_hour = defaultdict(list)
for n in keep:
    for k in per_farm[n]:
        by_hour[(k[0] // 3600) % 24].append(nk[k][1] - nk[k][2])
print("\nNorKyst 0-3 m by UTC hour: " + " ".join(f"{h:02d}:{np.median(by_hour[h]):+.2f}" for h in sorted(by_hour) if h % 3 == 0))

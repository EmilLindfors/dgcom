#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""Satellite pixels: runs (0 and 3 m) and NorKyst's surface on the same pixels.

    uv run scripts/sat_window_stats.py <matched_norkyst.txt> <start_unix> <label>,<d0.txt>,<d3.txt> ...
Per night (24 h from start): medians of model - satellite at 0 m, NorKyst -
satellite, and the model's 0-3 m difference, on the pixels every run sampled.
"""
import sys
from collections import defaultdict

import numpy as np


def key(u, lon, lat):
    return (int(round(float(u))), round(float(lon), 3), round(float(lat), 3))


nk = {}
for line in open(sys.argv[1]):
    if line.startswith("#"):
        continue
    u, lon, lat, sat, t = line.split()[:5]
    nk[key(u, lon, lat)] = (float(sat), float(t))
start = float(sys.argv[2])
runs = []
for spec in sys.argv[3:]:
    label, p0, p3 = spec.split(",")
    m = defaultdict(dict)
    sat = {}
    for which, path in (("0", p0), ("3", p3)):
        for line in open(path):
            if line.startswith("#"):
                continue
            u, lon, lat, obs, model = line.split()
            k = key(u, lon, lat)
            m[k][which] = float(model)
            sat[k] = float(obs)
    runs.append((label, m, sat))

common = [k for k in runs[0][2] if all("0" in m.get(k, {}) and "3" in m.get(k, {}) for _, m, _ in runs)]
sat = runs[0][2]
with_nk = [k for k in common if k in nk]
print(f"{len(common)} pixels sampled by every run at 0 and 3 m, {len(with_nk)} with NorKyst")
days = defaultdict(list)
for k in common:
    days[int((k[0] - start) // 86400)].append(k)
head = f"{'night':>5} {'n':>5} {'sat':>6} " + " ".join(f"{'NK-sat':>7}" if i == 0 else "" for i in range(1)) + "".join(
    f" | {label + ' 0m-sat':>14} {'0-3m':>6}" for label, _, _ in runs)
print(head)
for d in sorted(days):
    ks = days[d]
    nks = [nk[k][1] - sat[k] for k in ks if k in nk]
    line = f"{d:5d} {len(ks):5d} {np.median([sat[k] for k in ks]):6.2f} {np.median(nks) if nks else np.nan:+7.2f}"
    for label, m, _ in runs:
        line += f" | {np.median([m[k]['0'] - sat[k] for k in ks]):+14.2f} {np.median([m[k]['0'] - m[k]['3'] for k in ks]):+6.2f}"
    print(line)
ks = common
nks = [nk[k][1] - sat[k] for k in ks if k in nk]
line = f"{'all':>5} {len(ks):5d} {np.median([sat[k] for k in ks]):6.2f} {np.median(nks):+7.2f}"
for label, m, _ in runs:
    line += f" | {np.median([m[k]['0'] - sat[k] for k in ks]):+14.2f} {np.median([m[k]['0'] - m[k]['3'] for k in ks]):+6.2f}"
print(line)

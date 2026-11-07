#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "scipy",
#     "netCDF4",
# ]
# ///
"""
Compare the M2 tidal current of a froya_real_data run with NorKyst-800's,
map-wide: tide against tide, at every NorKyst grid point in the domain.

NorKyst's ū, v̄ are the total depth-mean flow (tides, coastal current,
wind), so peak speeds are not comparable with a tide-only run. Here both
are fitted identically (mean + M2 + K1 + M4 over hourly samples of the same
window), and the M2 ellipse semi-major axes and η amplitudes are compared at
the NorKyst points that have a wet model node within 300 m. Results are
binned by NorKyst depth, with the model/NorKyst depth ratio: NorKyst's
bathymetry is smoothed, so in shallow water its depth (and so its velocity
for a given transport) differs from the child's.

The run needs hourly VTU output over the window, e.g.

    cargo run --release --example froya_real_data -- nx=60 ny=45 hours=72 \\
        ramp_hours=3 output_minutes=60 output=output/m2
    uv run scripts/norkyst_current_comparison.py output/m2 [norkyst=data/froya_norkyst.nc] \\
        [start=24] [end=72] [lat0=63.8] [lon0=8.6] [station=8.665231,63.869331] \
        [points=<file.csv>] [nodes=<file.csv>]

`norkyst=` is a subset from examples/norkyst_nesting_subset.rs starting at
the run's `start`; `lat0`/`lon0` are the centre of the example's local
projection; `station` (lon,lat) adds the statistics within 3 km of it. `points=` writes
every compared NorKyst point (position, both depths, both M2 current
semi-major axes and η amplitudes, both M4 η amplitudes); `nodes=` the
model's M2 current and M2/M4 η amplitudes at every wet node, for maps of
where the overtides are made.
Hours are counted from the run's start (the frame number).
"""

import math
import re
import sys

import netCDF4
import numpy as np
from scipy.spatial import cKDTree

PERIODS_H = {"M2": 12.4206012, "K1": 23.93447213, "M4": 6.210300601}


def m2_amplitude(series, pinv, constituent=0):
    """Complex amplitude of `series` [time, ...] of the `constituent`-th of
    PERIODS_H (M2 by default): a cos ωt + b sin ωt → a − ib."""
    c = np.tensordot(pinv, series, axes=(1, 0))
    return c[1 + 2 * constituent] - 1j * c[2 + 2 * constituent]


def semi_major(u, v):
    """Semi-major axis |W+| + |W-| of the ellipse of complex amplitudes u, v."""
    return np.abs(0.5 * (u + 1j * v)) + np.abs(0.5 * (np.conj(u) + 1j * np.conj(v)))


def read_vtu(path, names=("points", "u", "v", "eta", "bathymetry", "h")):
    text = open(path).read()
    out = {}
    for m in re.finditer(r"<DataArray([^>]*)>(.*?)</DataArray>", text, re.S):
        name = re.search(r'Name="([^"]*)"', m.group(1))
        name = name.group(1) if name else "points"
        if name in names:
            out[name] = np.array(m.group(2).split(), float)
    return out


def local_projection(lat0, lon0):
    """dgcom `LocalProjection` (WGS84 radii of curvature at lat0)."""
    a, e2 = 6378137.0, 6.69437999014e-3
    s = math.sin(math.radians(lat0))
    per_lat = a * (1 - e2) / (1 - e2 * s * s) ** 1.5 * math.pi / 180
    per_lon = a / math.sqrt(1 - e2 * s * s) * math.cos(math.radians(lat0)) * math.pi / 180
    return lambda lon, lat: ((lon - lon0) * per_lon, (lat - lat0) * per_lat)


def main():
    run = sys.argv[1]
    opts = dict(a.split("=", 1) for a in sys.argv[2:])
    hours = np.arange(int(opts.get("start", 24)), int(opts.get("end", 72)) + 1)
    to_xy = local_projection(float(opts.get("lat0", 63.8)), float(opts.get("lon0", 8.6)))
    design = np.column_stack([np.ones(len(hours))] + sum(
        [[np.cos(2 * np.pi * hours / p), np.sin(2 * np.pi * hours / p)] for p in PERIODS_H.values()],
        [],
    ))
    pinv = np.linalg.pinv(design)

    nk = netCDF4.Dataset(opts.get("norkyst", "data/froya_norkyst.nc"))
    t = nk["time"][:]
    idx = [int(np.argmin(abs(t - (t[0] + 3600 * h)))) for h in hours]
    fill = lambda x: np.ma.filled(x, 0.0)
    land = np.ma.getmaskarray(nk["ubar_eastward"][idx[0]]).ravel()
    nk_major = semi_major(
        m2_amplitude(fill(nk["ubar_eastward"][idx]), pinv),
        m2_amplitude(fill(nk["vbar_northward"][idx]), pinv),
    ).ravel()
    nk_zeta = fill(nk["zeta"][idx])
    nk_eta = np.abs(m2_amplitude(nk_zeta, pinv)).ravel()
    nk_m4 = np.abs(m2_amplitude(nk_zeta, pinv, 2)).ravel()
    x, y = to_xy(nk["lon"][:], nk["lat"][:])
    x, y, h = np.asarray(x).ravel(), np.asarray(y).ravel(), np.asarray(nk["h"][:]).ravel()

    frames = [read_vtu(f"{run}/froya_{hr:04d}.vtu") for hr in hours]
    points = frames[0]["points"].reshape(-1, 3)[:, :2]
    depth = -frames[0]["bathymetry"]
    wet = np.flatnonzero(np.min([f["h"] for f in frames], axis=0) > 0.1)
    major = semi_major(
        m2_amplitude(np.array([f["u"] for f in frames]), pinv),
        m2_amplitude(np.array([f["v"] for f in frames]), pinv),
    )
    model_eta = np.array([f["eta"] for f in frames])
    eta = np.abs(m2_amplitude(model_eta, pinv))
    m4 = np.abs(m2_amplitude(model_eta, pinv, 2))
    distance, nearest = cKDTree(points[wet]).query(np.c_[x, y])
    nearest = wet[nearest]
    ok = (distance < 300) & ~land & (h > 3) & (nk_major > 0.02)
    ratio = major[nearest][ok] / nk_major[ok]
    depth_ratio = depth[nearest][ok] / h[ok]
    print(
        f"{ok.sum()} NorKyst points (hours {hours[0]}-{hours[-1]}): M2 eta amplitude model/NorKyst "
        f"median {np.median(eta[nearest][ok] / nk_eta[ok]):.3f}; M2 current semi-major model/NorKyst "
        f"median {np.median(ratio):.2f} (quartiles {np.percentile(ratio, 25):.2f}-{np.percentile(ratio, 75):.2f})"
    )
    for lo, hi in [(3, 30), (30, 60), (60, 120), (120, 400)]:
        sel = (h[ok] >= lo) & (h[ok] < hi)
        if sel.any():
            print(
                f"  NorKyst h {lo:3}-{hi:3} m: {sel.sum():5} points, current {np.median(ratio[sel]):.2f}, "
                f"depth {np.median(depth_ratio[sel]):.2f}, transport {np.median(ratio[sel] * depth_ratio[sel]):.2f}"
            )
    if "points" in opts:
        lon, lat = np.asarray(nk["lon"][:]).ravel(), np.asarray(nk["lat"][:]).ravel()
        n = nearest[ok]
        np.savetxt(
            opts["points"],
            np.c_[lon[ok], lat[ok], x[ok], y[ok], h[ok], depth[n], nk_major[ok], major[n],
                  nk_eta[ok], eta[n], nk_m4[ok], m4[n]],
            fmt="%.6f", delimiter=",",
            header="lon,lat,x,y,h_norkyst,depth_model,m2_major_norkyst,m2_major_model,"
            "m2_eta_norkyst,m2_eta_model,m4_eta_norkyst,m4_eta_model",
            comments="",
        )
    if "nodes" in opts:
        np.savetxt(
            opts["nodes"],
            np.c_[points[wet], depth[wet], major[wet], eta[wet], m4[wet]],
            fmt="%.6f", delimiter=",",
            header="x,y,depth,m2_major,m2_eta,m4_eta", comments="",
        )
    if "station" in opts:
        sx, sy = to_xy(*map(float, opts["station"].split(",")))
        sel = np.hypot(x[ok] - sx, y[ok] - sy) < 3000
        if sel.any():
            print(
                f"  within 3 km of {opts['station']}: {sel.sum()} points, current {np.median(ratio[sel]):.2f} "
                f"(NorKyst M2 {np.median(nk_major[ok][sel]):.3f} m/s), depth {np.median(depth_ratio[sel]):.2f}"
            )


if __name__ == "__main__":
    main()

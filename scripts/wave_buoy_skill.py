#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["netCDF4", "numpy"]
# ///
"""
H_s skill of the spectral wave model at wave buoys (TODO F.4), against the
buoys' hourly means (`scripts/e39_buoy_series.py`), for one or more runs of
`froya_real_data wave_coupling=… wave_spectra=… wave_stations=NAME:LON,LAT;…`
(their `wave_points.txt`), with the parent wave model at its nearest point as
the baseline (the spectra file of `scripts/nora3_wave_spectra.py`).

The model's samples (every `station_minutes`) are averaged over the same
hours [t, t + 1 h) as the buoys'. Scores from `after` hours of model time on:
bias, RMSE, scatter index SI = RMSE / mean observed, and the correlation r;
T_p and the mean direction (coming from) are reported as means and the mean
difference, not scored. The parent counts only within `max_km` of the buoy.

Usage:
    uv run scripts/wave_buoy_skill.py obs=data/wave_buoys/e39_20240131.txt \\
        start=2024-01-31T00:00:00Z runs=swan:output/alesund_swan,wam:output/alesund_wam \\
        [parent=data/alesund_nora3_spectra.nc] [after=12] [max_km=5]
"""
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import netCDF4
import numpy as np


def read_points(path: Path):
    """Station name → (times [s], H_s, T_p, from) from a wave_points.txt."""
    lines = path.read_text().splitlines()
    labels = next(l for l in lines if l.startswith("# points:"))[len("# points:"):]
    labels = [s.strip() for s in labels.split(";") if s.strip()]
    rows = np.array([[float(x) for x in l.split()] for l in lines if l and not l.startswith("#")])
    out = {}
    for i, label in enumerate(labels):
        name = label.split(" (")[0]
        c = 1 + 4 * i
        out[name] = (rows[:, 0], rows[:, c], rows[:, c + 2], rows[:, c + 3])
    return out


def hourly(times, values, hours, circular=False):
    """Means over [h, h + 1 h) of model time (s) for each hour h (s)."""
    out = []
    for h in hours:
        m = (times >= h) & (times < h + 3600.0) & np.isfinite(values)
        if not m.any():
            out.append(np.nan)
        elif circular:
            r = np.radians(values[m])
            out.append(math.degrees(math.atan2(np.sin(r).mean(), np.cos(r).mean())) % 360.0)
        else:
            out.append(values[m].mean())
    return np.array(out)


def scores(model, obs):
    m = np.isfinite(model) & np.isfinite(obs)
    if m.sum() < 3:
        return None
    d = model[m] - obs[m]
    rmse = math.sqrt((d ** 2).mean())
    return dict(n=int(m.sum()), obs=obs[m].mean(), model=model[m].mean(), bias=d.mean(),
                rmse=rmse, si=rmse / obs[m].mean(), r=float(np.corrcoef(model[m], obs[m])[0, 1]))


def angle_diff(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def main() -> None:
    args = dict(a.split("=", 1) for a in sys.argv[1:])
    start = datetime.fromisoformat(args["start"].replace("Z", "+00:00")).timestamp()
    after = float(args.get("after", "12")) * 3600.0
    max_km = float(args.get("max_km", "5"))
    obs = {}
    for line in Path(args["obs"]).read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        b, t, lon, lat, hm0, tp, tm02, mdir, n = line.split()
        name = b.split("_")[0]
        when = datetime.fromisoformat(t.replace("Z", "+00:00")).timestamp() - start
        obs.setdefault(name, {"lon": float(lon), "lat": float(lat), "rows": []})["rows"].append(
            (when, float(hm0), float(tp), float(mdir)))
    runs = [r.split(":", 1) for r in args["runs"].split(",")]
    series = {name: read_points(Path(d) / "wave_points.txt") for name, d in runs}

    parent = None
    if "parent" in args:
        nc = netCDF4.Dataset(args["parent"])
        parent = dict(t=np.asarray(nc["time"][:]) - start, lon=np.asarray(nc["longitude"][0, :]),
                      lat=np.asarray(nc["latitude"][0, :]), hs=np.asarray(nc["hs"][:, 0, :]),
                      tp=np.asarray(nc["tp"][:, 0, :]))

    print(f"H_s against the buoys' hourly means, from hour {after / 3600:.0f}"
          f" (bias, RMSE in m; SI = RMSE / mean observed)\n")
    print(f"{'buoy':5s} {'source':14s} {'n':>4s} {'obs':>6s} {'model':>6s} {'bias':>6s} {'RMSE':>6s} {'SI':>6s} {'r':>5s}"
          f"   {'Tp obs/mod':>11s}  {'dir diff':>8s}")
    summary = {}
    for name in sorted(obs, key=lambda n: "DABCFG".find(n[0])):
        rows = np.array(obs[name]["rows"])
        rows = rows[rows[:, 0] >= after]
        hours, o_hs, o_tp, o_dir = rows.T
        sources = []
        for run, points in series.items():
            if name in points:
                t, hs, tp, frm = points[name]
                sources.append((run, hourly(t, hs, hours), hourly(t, tp, hours), hourly(t, frm, hours, True)))
        if parent is not None:
            lon, lat = obs[name]["lon"], obs[name]["lat"]
            d = np.hypot((parent["lon"] - lon) * math.cos(math.radians(lat)), parent["lat"] - lat) * 111.0
            k = int(np.argmin(d))
            if d[k] <= max_km:
                p_hs = np.interp(hours + 1800.0, parent["t"], parent["hs"][:, k], left=np.nan, right=np.nan)
                p_tp = np.interp(hours + 1800.0, parent["t"], parent["tp"][:, k], left=np.nan, right=np.nan)
                sources.append((f"NORA3 {d[k]:.1f} km", p_hs, p_tp, np.full_like(p_hs, np.nan)))
        # Every source on the same hours: those all of them cover.
        common = np.logical_and.reduce([np.isfinite(hs) for _, hs, _, _ in sources]) if sources else None
        for source, hs, tp, frm in sources:
            hs = np.where(common, hs, np.nan)
            s = scores(hs, o_hs)
            if s is None:
                continue
            summary.setdefault(source.split(" ")[0], {})[name] = s
            m = np.isfinite(tp) & np.isfinite(o_tp)
            tps = f"{o_tp[m].mean():5.1f}/{tp[m].mean():<5.1f}" if m.any() else ""
            m = np.isfinite(frm) & np.isfinite(o_dir)
            dirs = f"{angle_diff(frm[m], o_dir[m]).mean():+7.0f}°" if m.any() else ""
            print(f"{name:5s} {source:14s} {s['n']:4d} {s['obs']:6.2f} {s['model']:6.2f} {s['bias']:+6.2f} "
                  f"{s['rmse']:6.2f} {s['si']:6.2f} {s['r']:5.2f}   {tps:>11s}  {dirs:>8s}")
        print()
    sheltered = ["A", "B", "C", "F"]
    print("Mean SI over the sheltered buoys A, B, C, F (where each source has them):")
    for source, by in summary.items():
        have = [by[b]["si"] for b in sheltered if b in by]
        print(f"  {source:8s} {np.mean(have):.3f} over {len(have)} ({', '.join(b for b in sheltered if b in by)})")


if __name__ == "__main__":
    main()

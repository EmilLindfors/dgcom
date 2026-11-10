#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pandas"]
# ///
"""
A case run's stations against tide gauges from the IOC Sea Level Station
Monitoring Facility (https://www.ioc-sealevelmonitoring.org: SHOA's Chilean
gauges, and most of the world's real-time ones), and optionally a global
tide model's own prediction at the gauges as the baseline to beat.

For each station it prints, per window of `window_h` hours after `skip_h`:
the observed and modelled range, the centred RMSE (both series with their
window mean removed: the gauges' datum is the station's own, and the record
holds the weather's surge too) and the lag that fits best.

The records are fetched once into `cache` (one JSON per gauge and period; the
service gives at most 30 days per request). With `model=` and `model_dir=`
(pyTMD's names, as `scripts/tide_model_boundary_atlas.py`) the tide model is
scored the same way; that needs `uv run --with pyTMD --with dask --with
xarray --with netCDF4 --with scipy`.

Usage:
    uv run scripts/ioc_gauge_skill.py stations=cases/chiloe/output/chiloe/stations.csv \\
        start=2025-11-12T00:00:00 \\
        gauges="Puerto Montt:pmon;Ancud:ancu;Castro:cstr;Melinka:pmel" \\
        [skip_h=12.42] [window_h=12.42] [cache=data/tide_gauges/ioc] \\
        [model=TPXO10-atlas-v2-nc model_dir=data/tide_models \\
         positions="Puerto Montt:-72.9608,-41.4850;…"]
"""
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

SERVICE = "https://www.ioc-sealevelmonitoring.org/service.php"
MAX_LAG_STEPS = 12


def parse_args(argv):
    args = {}
    for arg in argv:
        key, sep, value = arg.partition("=")
        if not sep:
            sys.exit(f"argument {arg!r} is not key=value")
        args[key] = value
    for key in ("stations", "start", "gauges"):
        if key not in args:
            sys.exit(f"missing {key}=; see the docstring")
    return args


def pairs(text):
    """`name:value;name:value` → {name: value}."""
    return dict(item.split(":", 1) for item in text.split(";") if item)


def gauge_record(code, start, stop, cache):
    """The gauge's record (seconds after `start`, level in m) from its most
    reporting sensor, fetched once."""
    path = Path(cache) / f"{code}_{start:%Y%m%dT%H%M}_{stop:%Y%m%dT%H%M}.json"
    if not path.exists():
        query = urllib.parse.urlencode(
            dict(query="data", code=code, format="json",
                 timestart=f"{start:%Y-%m-%dT%H:%M:%S}", timestop=f"{stop:%Y-%m-%dT%H:%M:%S}")
        )
        request = urllib.request.Request(f"{SERVICE}?{query}", headers={"User-Agent": "curl"})
        with urllib.request.urlopen(request, timeout=300) as response:
            body = response.read()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    record = pd.DataFrame(json.loads(path.read_text()))
    if record.empty:
        return None, None, None
    sensor = record["sensor"].value_counts().index[0]
    record = record[record["sensor"] == sensor]
    seconds = (pd.to_datetime(record["stime"]) - start).dt.total_seconds().to_numpy()
    return seconds, record["slevel"].to_numpy(float), sensor


def model_tide(model, directory, lon, lat, start, seconds):
    """The tide model's level at a point and times (NaN on its land)."""
    import pyTMD.compute

    since_2000 = (start - pd.Timestamp("2000-01-01")).total_seconds() + seconds
    tide = pyTMD.compute.tide_elevations(
        np.full(len(seconds), lon), np.full(len(seconds), lat), since_2000,
        directory=directory, model=model, crop=True,
        bounds=[lon - 1, lon + 1, lat - 1, lat + 1], chunks="auto",
        type="drift", standard="UTC", extrapolate=True, cutoff=10.0,
    )
    return np.ma.filled(np.ma.asarray(tide, dtype=float), np.nan)


def centred_rmse(a, b):
    return float(np.sqrt(np.mean(((a - a.mean()) - (b - b.mean())) ** 2)))


def lag_minutes(a, b, step):
    """Minutes `a` is late on `b`: the shift with the least centred error."""
    a, b = a - a.mean(), b - b.mean()
    n = min(MAX_LAG_STEPS, len(a) // 4)
    if n == 0:
        return 0.0
    error = lambda k: np.mean((np.roll(a, k)[n:-n] - b[n:-n]) ** 2)
    return -min(range(-n, n + 1), key=error) * step / 60.0


def main():
    args = parse_args(sys.argv[1:])
    start = pd.Timestamp(args["start"]).tz_localize(None)
    skip = float(args.get("skip_h", 12.42)) * 3600.0
    window = float(args.get("window_h", 12.42)) * 3600.0
    codes = pairs(args["gauges"])
    positions = {
        name: tuple(float(v) for v in value.split(","))
        for name, value in pairs(args.get("positions", "")).items()
    }
    stations = pd.read_csv(args["stations"])
    stop = start + pd.Timedelta(seconds=float(stations["time_s"].max()) + 3600.0)
    for name, series in stations.groupby("station", sort=False):
        if name not in codes:
            continue
        t = series["time_s"].to_numpy(float)
        eta = series["eta_m"].to_numpy(float)
        seconds, level, sensor = gauge_record(
            codes[name], start, stop, args.get("cache", "data/tide_gauges/ioc")
        )
        if seconds is None:
            print(f"{name}: no record at {codes[name]}")
            continue
        observed = np.interp(t, seconds, level)
        tide = None
        if "model" in args and name in positions:
            tide = model_tide(args["model"], args["model_dir"], *positions[name], start, t)
        print(f"{name} ({codes[name]}, sensor {sensor}):")
        step = t[1] - t[0]
        begin = skip
        while begin + window <= t[-1] + step:
            w = (t >= begin) & (t < begin + window)
            o, m = observed[w], eta[w]
            line = (
                f"  {begin / 3600:6.1f}-{(begin + window) / 3600:6.1f} h: range obs {np.ptp(o):.2f} m, "
                f"run {np.ptp(m):.2f} m, RMSE {centred_rmse(m, o):.3f} m, {lag_minutes(m, o, step):+.0f} min"
            )
            if tide is not None:
                x = tide[w]
                if np.isfinite(x).all():
                    line += (
                        f" | {args['model']}: range {np.ptp(x):.2f} m, "
                        f"RMSE {centred_rmse(x, o):.3f} m, {lag_minutes(x, o, step):+.0f} min"
                    )
                else:
                    line += f" | {args['model']}: no water within 10 km"
            print(line)
            begin += window


if __name__ == "__main__":
    main()

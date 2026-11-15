#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# ///
"""
An hourly tide-gauge record from the University of Hawaii Sea Level Center
(fast-delivery data, https://uhslc.soest.hawaii.edu) in the gauge format of
`scripts/kartverket_gauge.sh`, about the window's own mean.

    uv run scripts/uhslc_gauge.py <uhslc id> <from> <to> <out.txt> [cache dir]

e.g. `684 2022-01-01 2023-01-01 data/tide_gauges/world/h684_obs.txt` (Puerto
Montt). The station list (`meta.geojson`: ids, names, positions, spans) and
each station's whole record (`h<id>.csv`: year, month, day, hour, mm; -32767
where missing) are kept in the cache directory (default
`data/tide_gauges/uhslc`), so a second window of a station is not fetched
again.
"""

import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE = "https://uhslc.soest.hawaii.edu/data"


def cached(cache, name, url):
    path = cache / name
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(url, headers={"User-Agent": "curl"})
        with urllib.request.urlopen(request, timeout=300) as response:
            path.write_bytes(response.read())
    return path


def main():
    station, start, end, out = int(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
    cache = Path(sys.argv[5]) if len(sys.argv) > 5 else Path("data/tide_gauges/uhslc")
    day = lambda s: datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    t0, t1 = day(start), day(end)

    meta = json.load(open(cached(cache, "meta.geojson", f"{BASE}/meta.geojson"), encoding="utf-8"))
    feature = next(f for f in meta["features"] if f["properties"]["uhslc_id"] == station)
    lon, lat = feature["geometry"]["coordinates"]
    lon = (lon + 180.0) % 360.0 - 180.0
    name = feature["properties"]["name"]

    rows = []
    record = cached(cache, f"h{station:03d}.csv", f"{BASE}/csv/fast/hourly/h{station:03d}.csv")
    for line in open(record, encoding="utf-8"):
        y, m, d, h, mm = (int(x) for x in line.split(","))
        t = datetime(y, m, d, h, tzinfo=timezone.utc)
        if t0 <= t < t1 and mm > -30000:
            rows.append((t, mm / 1000.0))
    if not rows:
        sys.exit(f"{name}: no samples in {start} .. {end}")
    mean = sum(v for _, v in rows) / len(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"# UHSLC fast-delivery water level, {start} to {end}, every 60 min\n")
        f.write(f"# source: {BASE}/csv/fast/hourly/h{station:03d}.csv\n")
        f.write(f"# station: {name}\n# id: h{station:03d}\n# latitude: {lat:.4f}\n# longitude: {lon:.4f}\n")
        f.write("# datum: the record's mean\n# units: m\n# columns: datetime water_level(m)\n")
        for t, v in rows:
            f.write(f"{t:%Y-%m-%dT%H:%M:%SZ} {v - mean:.3f}\n")
    hours = (t1 - t0).total_seconds() / 3600.0
    print(f"{name} (h{station:03d}): {len(rows)} of {hours:.0f} hours, mean {mean:.3f} m removed")


if __name__ == "__main__":
    main()

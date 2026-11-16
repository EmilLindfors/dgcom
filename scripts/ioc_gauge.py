#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# ///
"""
A tide-gauge record from the IOC Sea Level Station Monitoring Facility
(https://www.ioc-sealevelmonitoring.org: SHOA's Chilean gauges, and most of
the world's real-time ones) in the gauge format of `scripts/kartverket_gauge.sh`,
about the window's own mean: what a case's `[[output.station]] gauge` reads
(`dgcom score`, `gauge_skill.csv`) and the viewer's `--gauge`.

    uv run scripts/ioc_gauge.py <code> <name> <lat> <lon> <from> <to> <out.txt> \
        [step_min=60] [cache=data/tide_gauges/ioc]

e.g. `pmon "Puerto Montt" -41.4850 -72.9608 2025-10-13 2025-12-12
data/tide_gauges/chiloe_pmon_obs.txt`. The service gives at most 30 days per
request; each request's JSON is kept in the cache, so a window is fetched once.
The record is the sensor that reports most over the window. With `step_min`
60 (the default) each hour's value is the median of the minutes within 5 of
it, with at least 6 of them, else the hour is left out: the median drops the
real-time records' single-sample spikes, and an 11-minute window takes 0.03 %
off M2 (an hour's mean would take 1.1 %, 2 cm of Puerto Montt's 2 m).
`step_min=0` writes every sample, as the viewer's trace wants.
"""

import json
import statistics
import sys
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

SERVICE = "https://www.ioc-sealevelmonitoring.org/service.php"
MAX_DAYS = 30
HALF_WINDOW_MIN = 5
MIN_SAMPLES = 6


def fetch(code, start, stop, cache):
    """The service's records of `code` between `start` and `stop`, in
    requests of at most `MAX_DAYS` days, each kept in `cache`."""
    rows = []
    t = start
    while t < stop:
        end = min(t + timedelta(days=MAX_DAYS), stop)
        path = cache / f"{code}_{t:%Y%m%dT%H%M}_{end:%Y%m%dT%H%M}.json"
        if not path.exists():
            query = urllib.parse.urlencode(dict(
                query="data", code=code, format="json",
                timestart=f"{t:%Y-%m-%dT%H:%M:%S}", timestop=f"{end:%Y-%m-%dT%H:%M:%S}",
            ))
            request = urllib.request.Request(f"{SERVICE}?{query}", headers={"User-Agent": "curl"})
            with urllib.request.urlopen(request, timeout=300) as response:
                body = response.read()
            cache.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
        rows += json.loads(path.read_text())
        t = end
    return rows


def main():
    positional = [a for a in sys.argv[1:] if "=" not in a]
    options = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    if len(positional) != 7:
        sys.exit(__doc__)
    code, name, lat, lon, first, last, out = positional
    step = int(options.get("step_min", 60))
    cache = Path(options.get("cache", "data/tide_gauges/ioc"))
    utc = lambda text: datetime.fromisoformat(text).replace(tzinfo=timezone.utc)
    start, stop = utc(first), utc(last)

    rows = fetch(code, start.replace(tzinfo=None), stop.replace(tzinfo=None), cache)
    if not rows:
        sys.exit(f"{code}: no record between {first} and {last}")
    sensor = Counter(r["sensor"] for r in rows).most_common(1)[0][0]
    samples = {}
    for r in rows:
        if r["sensor"] != sensor or r.get("slevel") is None:
            continue
        t = utc(r["stime"].strip())
        if start <= t < stop:
            samples[t] = float(r["slevel"])  # requests overlap at their ends
    times = sorted(samples)

    if step == 0:
        series = [(t, samples[t]) for t in times]
    else:
        by_slot = {}
        for t in times:
            # The nearest multiple of `step`, if within the half window
            minutes = (t - start).total_seconds() / 60.0
            slot = round(minutes / step)
            if abs(minutes - slot * step) <= HALF_WINDOW_MIN:
                by_slot.setdefault(slot, []).append(samples[t])
        series = [
            (start + timedelta(minutes=slot * step), statistics.median(values))
            for slot, values in sorted(by_slot.items())
            if len(values) >= MIN_SAMPLES
        ]
    mean = sum(v for _, v in series) / len(series)
    expected = int((stop - start).total_seconds() / 60 / step) if step else len(times)

    with open(out, "w", encoding="utf-8") as f:
        f.write(
            f"# IOC Sea Level Station Monitoring Facility, sensor {sensor}, {first} to {last}"
            + (f", the median of the minutes within {HALF_WINDOW_MIN} of each {step} min" if step else "")
            + "\n"
        )
        f.write(f"# source: https://www.ioc-sealevelmonitoring.org/station.php?code={code}\n")
        f.write(f"# station: {name}\n# id: {code}\n# latitude: {lat}\n# longitude: {lon}\n")
        f.write("# datum: the window's mean\n# units: m\n# columns: datetime water_level(m)\n")
        for t, v in series:
            f.write(f"{t:%Y-%m-%dT%H:%M:%SZ} {v - mean:.3f}\n")
    print(f"{name} ({code}, sensor {sensor}): {len(series)} of {expected} values "
          f"({100 * len(series) / max(expected, 1):.0f} %), {first} to {last} → {out}")


main()

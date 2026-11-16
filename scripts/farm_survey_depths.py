#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "rasterio",
# ]
# ///
"""Measured depths at fish farms against a topobathy raster.

Fiskeridirektoratet's site register (pub-aqua) gives the sites in a box, and
its environmental report register (envreportreg) each site's MOM-B surveys:
10-20+ stations under and around the farm with positions and the depth as
sounded (the tide not taken off). Each station is compared with the raster
sampled at it, the raster's sea part lowered by `offset` m (mean sea level
above chart datum: the topobathy's sea part is in chart datum,
notes/inputs-forcing.md P1.6, 2026-11-15).

usage: uv run scripts/farm_survey_depths.py <raster.tif> <cache_dir> \
           <min_lon,min_lat,max_lon,max_lat> <offset_m> <municipality code>...

e.g.   uv run scripts/farm_survey_depths.py data/froya_topobathy.tif output/farm_depths \
           8.0,63.6,9.2,64.0 1.434 5014 5056 1573 5055 5057
Writes <cache_dir>/stations.csv (site, report, date, lon, lat, measured, raster).
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import rasterio

AQUA = "https://api.fiskeridir.no/pub-aqua/api/v1/sites"
REPORTS = "https://api.fiskeridir.no/envreportreg-public/api/v2/report"


def get(url, path):
    """The JSON at `url`, cached in `path`; {} for a 404 (a site without reports)."""
    if not path.exists():
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                path.write_bytes(r.read())
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise
            path.write_text("{}", encoding="utf-8")
        time.sleep(0.2)
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    raster, cache = Path(sys.argv[1]), Path(sys.argv[2])
    lo0, la0, lo1, la1 = (float(x) for x in sys.argv[3].split(","))
    offset = float(sys.argv[4])
    cache.mkdir(parents=True, exist_ok=True)

    sites = {}
    for code in sys.argv[5:]:
        start = 0
        while True:
            page = get(f"{AQUA}?municipality-code={code}&range={start}-{start + 99}",
                       cache / f"sites_{code}_{start}.json")
            for s in page:
                if (s.get("waterType") == "Salt" and s.get("latitude")
                        and lo0 <= s["longitude"] <= lo1 and la0 <= s["latitude"] <= la1):
                    sites[s["siteNr"]] = s
            if len(page) < 100:
                break
            start += 100
    print(f"{len(sites)} salt-water sites in the box")

    rows = []
    for nr, s in sorted(sites.items()):
        found = get(f"{REPORTS}/search?siteNumber={nr}&size=100", cache / f"search_{nr}.json")
        mom_b = [r for r in found.get("content", []) if r.get("envExaminationType") == "b"]
        if not mom_b:
            continue
        latest = max(mom_b, key=lambda r: r.get("reportCreated") or "")
        rep = get(f"{REPORTS}/{latest['reportId']}", cache / f"report_{latest['reportId']}.json")
        for m in rep.get("measurements") or []:
            if m.get("depthMeters") is None or m.get("longitudeDecimal") is None:
                continue
            rows.append((nr, s["name"], latest["reportId"], rep.get("measurementDate"),
                         m["longitudeDecimal"], m["latitudeDecimal"], float(m["depthMeters"])))

    with rasterio.open(raster) as src:
        vals = np.array([v[0] for v in src.sample([(r[4], r[5]) for r in rows])], dtype=float)
    model = -vals + offset  # depth below mean sea level on the raster
    out = cache / "stations.csv"
    with out.open("w", encoding="utf-8") as f:
        f.write("site,name,report,date,lon,lat,measured_m,raster_m\n")
        for r, d in zip(rows, model):
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]},{r[6]},{d:.2f}\n")

    measured = np.array([r[6] for r in rows])
    diff = measured - model
    print(f"{len(rows)} stations at {len({r[0] for r in rows})} sites (latest MOM-B each): {out}")
    q = np.percentile(diff, [10, 25, 50, 75, 90])
    print(f"measured - raster: median {q[2]:+.1f} m, quartiles {q[1]:+.1f}, {q[3]:+.1f}, "
          f"10-90 % {q[0]:+.1f}, {q[4]:+.1f}")
    print(f"raster shallower by more than 5 m at {np.mean(diff > 5):.0%} of stations, "
          f"by more than half the measured depth at {np.mean(diff > 0.5 * measured):.0%}")
    print("per site (median measured, median raster, median difference, stations):")
    for nr in sorted({r[0] for r in rows}):
        k = np.array([r[0] == nr for r in rows])
        name = next(r[1] for r in rows if r[0] == nr)
        print(f"  {nr:6d} {name[:22]:22s} {np.median(measured[k]):6.1f} {np.median(model[k]):6.1f} "
              f"{np.median(diff[k]):+6.1f}  {k.sum()}")


main()

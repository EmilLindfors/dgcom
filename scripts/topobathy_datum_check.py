#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "pyproj",
#     "rasterio",
# ]
# ///
"""Topobathy sea-part datum: chart-datum depth points (Kartverket WFS
wfs.dybdedata, app:Dybdepunkt) against the 50 m topobathy raster.

If the raster's sea part is in chart datum, raster + chart depth ~ 0; if in
NN2000, ~ -(NN2000 - chart datum), 1.515 m at Mausund.

usage: uv run scripts/topobathy_datum_check.py <raster.tif> <cache_dir> <min_lon,min_lat,max_lon,max_lat> [...]

e.g.   uv run scripts/topobathy_datum_check.py data/froya_topobathy.tif output/datum_check 8.0,63.6,9.2,64.0
(notes/inputs-forcing.md, P1.6, 2026-11-15)
"""
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer

WFS = "https://wfs.geonorge.no/skwms1/wfs.dybdedata"
NS = {
    "gml": "http://www.opengis.net/gml/3.2",
    "app": "http://skjema.geonorge.no/SOSI/produktspesifikasjon/Dybdedata/20201001",
    "wfs": "http://www.opengis.net/wfs/2.0",
}


def fetch(box, out):
    lo0, la0, lo1, la1 = box
    pts = []
    start = 0
    while True:
        url = (
            f"{WFS}?service=WFS&version=2.0.0&request=GetFeature&typeNames=app:Dybdepunkt"
            f"&count=5000&startIndex={start}&srsName=urn:ogc:def:crs:EPSG::25833"
            f"&bbox={la0},{lo0},{la1},{lo1},urn:ogc:def:crs:EPSG::4258"
        )
        path = out / f"pts_{lo0}_{la0}_{start}.xml"
        if not path.exists():
            with urllib.request.urlopen(url, timeout=120) as r:
                path.write_bytes(r.read())
        root = ET.parse(path).getroot()
        members = root.findall("wfs:member", NS)
        for m in members:
            p = m.find(".//gml:pos", NS)
            d = m.find(".//app:dybde", NS)
            if p is not None and d is not None:
                e, n = map(float, p.text.split())
                pts.append((e, n, float(d.text)))
        if len(members) < 5000:
            return pts
        start += 5000


def main():
    raster, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    to_geo = Transformer.from_crs(25833, 4326, always_xy=True)
    with rasterio.open(raster) as src:
        z = src.read(1).astype(float)
        inv = ~src.transform
        nodata = src.nodata
    for spec in sys.argv[3:]:
        box = tuple(float(x) for x in spec.split(","))
        pts = np.array(fetch(box, out))
        if len(pts) == 0:
            print(f"{box}: no depth points")
            continue
        lon, lat = to_geo.transform(pts[:, 0], pts[:, 1])
        col, row = inv * (np.asarray(lon), np.asarray(lat))
        # Bilinear on pixel centres
        c, r = col - 0.5, row - 0.5
        i, j = np.floor(r).astype(int), np.floor(c).astype(int)
        ok = (i >= 1) & (j >= 1) & (i < z.shape[0] - 2) & (j < z.shape[1] - 2)
        i, j, r, c, depth = i[ok], j[ok], r[ok], c[ok], pts[ok, 2]
        fr, fc = r - i, c - j
        val = (
            z[i, j] * (1 - fr) * (1 - fc)
            + z[i, j + 1] * (1 - fr) * fc
            + z[i + 1, j] * fr * (1 - fc)
            + z[i + 1, j + 1] * fr * fc
        )
        # Flatness: the range of the 4 x 4 pixels around the point
        win = np.stack([z[i + a, j + b] for a in (-1, 0, 1, 2) for b in (-1, 0, 1, 2)])
        spread = win.max(axis=0) - win.min(axis=0)
        valid = np.isfinite(val) & (val < 0) & (depth > 0)
        if nodata is not None:
            valid &= (win != nodata).all(axis=0)
        s = val + depth  # raster elevation + chart depth
        print(f"box {box}: {len(pts)} points, {valid.sum()} under water in the raster")
        for lo, hi in [(0, 5), (5, 10), (10, 20), (20, 50), (50, 200)]:
            for flat in (None, 1.0, 2.0, 4.0):
                m = valid & (depth >= lo) & (depth < hi)
                if flat is not None:
                    m &= spread < flat
                if m.sum() < 5:
                    continue
                q = np.percentile(s[m], [25, 50, 75])
                print(
                    f"  chart depth {lo:3d}-{hi:3d} m"
                    f"{'' if flat is None else f', 4x4 range < {flat} m':<20}"
                    f"  n {m.sum():5d}  raster + depth: median {q[1]:+.2f} m (quartiles {q[0]:+.2f}, {q[2]:+.2f})"
                )


main()

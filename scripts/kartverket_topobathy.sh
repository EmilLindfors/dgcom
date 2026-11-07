#!/bin/bash
# Fetch Kartverket's national topobathy terrain model (land heights and sea
# depths in one seamless grid) as a GeoTIFF in longitude/latitude (EPSG:4326),
# for `dgcom::io::GeoTiffBathymetry` / `BedRaster`.
#
# Source: the WCS "Topobaty nasjonal detaljert terrengmodell 25833"
# (https://wcs.geonorge.no/skwms1/wcs.hoyde-dtm-nhm-topobathy-25833), free,
# served by hoydedata.no, which reprojects to EPSG:4326 on request. Values
# are bed elevations in metres, positive up (land heights to depths).
#
# Resolution: the service has two levels. At cells of about 42 m and coarser it
# serves the 50 m model, land and sea interpolated together. Finer requests
# get the 1 m level, which has depths only inside surveyed projects and the
# flat water surface (0) elsewhere (measured 2026-09-26: square cells of 40 m
# get the 1 m level, 42 m the 50 m model). So cells of 50 m and up give the
# 50 m model, for the solver; cells of 40 m and below give the 1 m level, with a
# warning: it is land detail for the viewer, layered over the 50 m model
# (`dg-viz --terrain coarse.tif,fine.tif`), not a bed (`froya_real_data` rejects
# it). Around the Kattholmen farm it is 17 % lidar land and 83 % flat 0.
# Cells between 40 and 50 m are refused: which level they get is not known.
# One request is at most 3840 x 2160 pixels.
#
# Vertical datum: land heights are NN2000 (within 2-24 cm of mean sea level
# at the Norwegian tide gauges). The seamless model has to share one datum,
# so the sea part is taken to be NN2000 too; the service documentation does
# not state it (see TODO.md P1.6).
#
# Usage:
#   ./scripts/kartverket_topobathy.sh <min_lon> <min_lat> <max_lon> <max_lat> [cell_m=50] [out]
#
# Examples (the Frøya-Smøla-Hitra domain at about 50 m, 1 request; the land
# around the Kattholmen farm at 2 m, 1500 x 1350 pixels):
#   ./scripts/kartverket_topobathy.sh 8.0 63.6 9.2 64.0 50 data/froya_topobathy.tif
#   ./scripts/kartverket_topobathy.sh 8.648 63.8575 8.710 63.8845 2 data/kattholmen_topobathy_2m.tif

set -euo pipefail

if [ $# -lt 4 ]; then
    sed -n '2,35p' "$0"
    exit 1
fi

MIN_LON="$1"
MIN_LAT="$2"
MAX_LON="$3"
MAX_LAT="$4"
CELL="${5:-50}"
OUT="${6:-data/topobathy_${MIN_LON}_${MIN_LAT}_${MAX_LON}_${MAX_LAT}.tif}"

# Width and height in pixels for square-ish cells of CELL metres at the
# central latitude, and a check against the service limits
read -r WIDTH HEIGHT < <(awk -v a="$MIN_LON" -v b="$MIN_LAT" -v c="$MAX_LON" -v d="$MAX_LAT" -v cell="$CELL" 'BEGIN {
    if (cell > 40 && cell < 50) { print "cells between 40 and 50 m may get either level: use >= 50 m (the 50 m model) or <= 40 m (the 1 m level)" > "/dev/stderr"; exit 1 }
    if (cell <= 40) { print "cells <= 40 m get the 1 m level: lidar land and surveyed depths, a flat 0 over unsurveyed sea (land detail for the viewer, not a bed)" > "/dev/stderr" }
    pi = atan2(0, -1); lat = (b + d) / 2 * pi / 180
    w = int((c - a) * 111320 * cos(lat) / cell + 0.5); h = int((d - b) * 111320 / cell + 0.5)
    if (w < 1 || h < 1) { print "empty box" > "/dev/stderr"; exit 1 }
    if (w > 3840 || h > 2160) { printf "%d x %d pixels exceeds the 3840 x 2160 limit: use a larger cell or a smaller box\n", w, h > "/dev/stderr"; exit 1 }
    print w, h
}')

URL="https://hoydedata.no/arcgis/services/NHM_DTM_TOPOBATHY_25833/ImageServer/WCSServer"
URL+="?service=WCS&version=1.0.0&request=GetCoverage&coverage=nhm_dtm_topobathy_25833"
URL+="&crs=EPSG:4326&bbox=${MIN_LON},${MIN_LAT},${MAX_LON},${MAX_LAT}"
URL+="&width=${WIDTH}&height=${HEIGHT}&format=GeoTIFF&interpolation=bilinear"

mkdir -p "$(dirname "$OUT")"
curl -sf -m 600 -o "$OUT" "$URL"
# An error comes back as XML with status 200
if [ "$(head -c 2 "$OUT")" != "II" ] && [ "$(head -c 2 "$OUT")" != "MM" ]; then
    echo "Not a GeoTIFF:" >&2
    head -c 500 "$OUT" >&2
    rm -f "$OUT"
    exit 1
fi
echo "$OUT: ${WIDTH} x ${HEIGHT} pixels over [${MIN_LON}, ${MAX_LON}] x [${MIN_LAT}, ${MAX_LAT}] (~${CELL} m)"
echo "source: ${URL}"

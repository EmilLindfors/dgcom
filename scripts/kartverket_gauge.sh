#!/bin/bash
# Fetch Kartverket water levels at a tide gauge into the text format of
# `dgcom::io::read_tide_gauge_file` (ISO times in UTC, metres above MSL).
#
# Source: Kartverket's tide API (https://vannstand.kartverket.no/tideapi_en.html),
# which serves observations (`obs`) or predictions (`pre`) for a location.
# Observations include the weather-driven surge. Timestamps in `pre`-flagged
# gap fills are dropped from an `obs` request.
#
# Usage:
#   ./scripts/kartverket_gauge.sh <name> <lat> <lon> <from> <to> [interval_min=60] [datatype=obs] [out]
#
# Example (Mausund, MSU, inside the Frøya domain; a year of hourly data):
#   ./scripts/kartverket_gauge.sh Mausund 63.869331 8.665231 2024-07-01 2025-07-01

set -euo pipefail

if [ $# -lt 5 ]; then
    sed -n '2,15p' "$0"
    exit 1
fi

NAME="$1"
LAT="$2"
LON="$3"
FROM="$4"
TO="$5"
INTERVAL="${6:-60}"
DATATYPE="${7:-obs}"
SLUG="$(echo "$NAME" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9\n' '_')"
OUT="${8:-data/tide_gauges/${SLUG}_${DATATYPE}.txt}"

URL="https://vannstand.kartverket.no/tideapi.php?tide_request=locationdata"
URL+="&lat=${LAT}&lon=${LON}&fromtime=${FROM}T00:00&totime=${TO}T00:00"
URL+="&datatype=${DATATYPE}&refcode=msl&lang=en&interval=${INTERVAL}&dst=0&tzone=0"

XML="$(curl -sf -m 300 "$URL")"
CODE="$(echo "$XML" | sed -n 's/.*<location [^>]*code="\([^"]*\)".*/\1/p' | head -1)"
if [ -z "$CODE" ]; then
    echo "No location in the response:" >&2
    echo "$XML" | head -5 >&2
    exit 1
fi

{
    echo "# Kartverket water level (${DATATYPE}), ${FROM} to ${TO}, every ${INTERVAL} min"
    echo "# source: ${URL}"
    echo "# station: ${NAME}"
    echo "# id: ${CODE}"
    echo "# latitude: ${LAT}"
    echo "# longitude: ${LON}"
    echo "# datum: MSL"
    echo "# units: m"
    echo "# columns: datetime water_level(m)"
    # <waterlevel value="-47.0" time="2024-07-01T00:00:00+00:00" flag="obs"/>, cm
    echo "$XML" | sed -n "s/.*<waterlevel value=\"\([^\"]*\)\" time=\"\([^\"+]*\)+00:00\" flag=\"${DATATYPE}\".*/\2Z \1/p" |
        awk '{ printf "%s %.3f\n", $1, $2 / 100 }'
} > "$OUT"

echo "$OUT: $(grep -vc '^#' "$OUT") samples (station ${CODE})"

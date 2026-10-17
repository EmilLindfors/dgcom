#!/bin/bash
# Reproduce the numbers in notes/norkyst-issues.md: NorKyst-800 v3's tide
# against Kartverket gauges along the coast, and NorKyst's tidal forcing files.
#
#   1. Kartverket observations (obs) and predictions (pre) at every gauge in
#      scripts/norkyst_tide_gauges.csv (scripts/kartverket_gauge.sh), into
#      data/tide_gauges/coast/: 2024-07-01 to 2025-07-01 (<code>_obs.txt) and
#      2026-04-01 to 2026-10-01 (<code>_obs_2026-04.txt); 2024-01 to 2026-09
#      at BGO, MSU, BOO for the monthly history (<code>_obs_<year>.txt).
#   2. NorKyst-800 zeta at the nearest wet cell of each gauge over the same
#      periods (scripts/norkyst_gauge_series.py), into data/norkyst_gauges/.
#   3. Our harmonic fit (examples/tide_gauge_fit.rs compare=) of gauge and
#      NorKyst on the instants both have, appended to
#      output/norkyst_issues/fits.csv, for June 2025, January 2025, the year
#      2024-07..2025-07, June 2026, 2026-04..2026-10 and the hindcast
#      (romshindcast/norkyst_v3) over 2024-07..2025-07; the gauge against
#      Kartverket's own prediction into fits_pre.csv.
#   4. The report: tables, the frequency-pair test, UTide as the outside
#      judge, the datum and pressure checks, plots
#      (scripts/norkyst_tide_report.py), and the forcing files themselves
#      (scripts/norkyst_tide_forcing.py), into output/norkyst_issues/.
#
# Downloads are skipped when the file exists. Needs curl, uv and cargo; the
# fits build without NetCDF (--no-default-features).
#
# Usage: ./scripts/norkyst_tide_check.sh

set -euo pipefail
cd "$(dirname "$0")/.."

GAUGES=scripts/norkyst_tide_gauges.csv
OUT=output/norkyst_issues
GAUGE_DIR=data/tide_gauges/coast
NK_DIR=data/norkyst_gauges
YEAR_DIR="$NK_DIR/2024-07_2025-07"
LONG_DIR="$NK_DIR/2024-01_2026-10"
NOW_DIR="$NK_DIR/2026-04_2026-10"
HINDCAST_DIR="$NK_DIR/hindcast_2024-07_2025-07"
HINDCAST=https://thredds.met.no/thredds/dodsC/romshindcast/norkyst_v3/norkyst_v3_sdepth_aggr.ncml
mkdir -p "$OUT" "$GAUGE_DIR" "$NK_DIR"

# 1. Gauges
fetch() { # name lat lon from to datatype file
    [ -s "$7" ] || ./scripts/kartverket_gauge.sh "$1" "$2" "$3" "$4" "$5" 60 "$6" "$7"
}
tail -n +2 "$GAUGES" | while IFS=, read -r name code lat lon; do
    for dt in obs pre; do
        fetch "$name" "$lat" "$lon" 2024-07-01 2025-07-01 "$dt" "$GAUGE_DIR/${code}_${dt}.txt"
        fetch "$name" "$lat" "$lon" 2026-04-01 2026-10-01 "$dt" "$GAUGE_DIR/${code}_${dt}_2026-04.txt"
    done
    # The long record for the monthly history, a year per request
    case "$code" in BGO|MSU|BOO)
        fetch "$name" "$lat" "$lon" 2024-01-01 2025-01-01 obs "$GAUGE_DIR/${code}_obs_2024.txt"
        fetch "$name" "$lat" "$lon" 2025-01-01 2026-01-01 obs "$GAUGE_DIR/${code}_obs_2025.txt"
        fetch "$name" "$lat" "$lon" 2026-01-01 2026-10-01 obs "$GAUGE_DIR/${code}_obs_2026.txt"
    esac
done

# 2. NorKyst at the gauges
[ -s "$YEAR_DIR/VAW.txt" ] || uv run scripts/norkyst_gauge_series.py "$GAUGES" 2024-07-01 2025-07-01 "$YEAR_DIR"
[ -s "$NOW_DIR/VAW.txt" ] || uv run scripts/norkyst_gauge_series.py "$GAUGES" 2026-04-01 2026-10-01 "$NOW_DIR"
[ -s "$HINDCAST_DIR/VAW.txt" ] || uv run scripts/norkyst_gauge_series.py "$GAUGES" 2024-07-01 2025-07-01 "$HINDCAST_DIR" url="$HINDCAST"
[ -s "$LONG_DIR/BOO.txt" ] || uv run scripts/norkyst_gauge_series.py "$GAUGES" 2024-01-01 2026-10-01 "$LONG_DIR" codes=BGO,MSU,BOO

# 3. Our fits on matched instants
cargo build --release --no-default-features --features parallel,simd --example tide_gauge_fit
FIT=target/release/examples/tide_gauge_fit
rm -f "$OUT/fits.csv" "$OUT/fits_pre.csv"
tail -n +2 "$GAUGES" | while IFS=, read -r name code lat lon; do
    for window in "june 2025-06-01 2025-07-01 obs.txt $YEAR_DIR" \
        "january 2025-01-01 2025-01-31 obs.txt $YEAR_DIR" \
        "year 2024-07-01 2025-07-01 obs.txt $YEAR_DIR" \
        "june2026 2026-06-01 2026-07-01 obs_2026-04.txt $NOW_DIR" \
        "2026 2026-04-01 2026-10-01 obs_2026-04.txt $NOW_DIR" \
        "hindcast 2024-07-01 2025-07-01 obs.txt $HINDCAST_DIR"; do
        set -- $window
        gauge="$GAUGE_DIR/${code}_$4"
        "$FIT" "$gauge" from="$2" to="$3" compare="$5/${code}.txt" \
            csv="$OUT/fits.csv" label="$code:$1" > "$OUT/fit_${code}_$1.txt"
        "$FIT" "$gauge" from="$2" to="$3" compare="${gauge/obs/pre}" \
            csv="$OUT/fits_pre.csv" label="$code:$1" > "$OUT/fit_${code}_$1_pre.txt"
    done
done
echo "fits: $OUT/fits.csv, $OUT/fits_pre.csv"

# 4. The report and the forcing files
uv run scripts/norkyst_tide_report.py "$GAUGES" "$GAUGE_DIR" "$NK_DIR" "$OUT"
uv run scripts/norkyst_tide_forcing.py "$GAUGES" "$OUT/fits.csv" out="$OUT/forcing.md"

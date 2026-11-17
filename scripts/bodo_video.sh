#!/usr/bin/env bash
# Fifteen days of tide at Bodø (about 70 s, 1920x1080, 30 fps): the pre-registered
# 15-day nested run (cases/bodo/scored_15d_2026-06.toml, NorKyst-800's tide corrected
# from theory) in current speed, with its clock and the water level at the Bodø gauge,
# model against Kartverket's observations, drawn over it.
#
#   ./scripts/bodo_video.sh RUN_DIR [out_dir]     # default output/video
#
# PREDICTED_CM [5.0] is the daily score predicted before the run (notes/validation.md,
# P3.4: the theory run's 5.0, the raw run's 7.8), for the end card.
#
# RUN_DIR is the run's output folder (its .dgsnap, stations.csv, gauge_daily.csv and
# gauge_skill.csv). The shot is a `dg-viz --record` along scripts/bodo_video/tide.cam;
# the cards, the clock and the gauge trace are added by scripts/bodo_video/assemble.py.
# `--window off` renders without a display (a GPU server over ssh); drop it to watch.
set -euo pipefail
cd "$(dirname "$0")/.."
run=$(cd "$1" && pwd)
out="${2:-output/video}"
frames="$PWD/$out/frames/bodo"
mkdir -p "$frames"
(cd viz && cargo build --release)
viewer="$PWD/viz/target/release/dg-viz"
cam="$PWD/scripts/bodo_video/tide.cam"
snap=$(ls "$run"/*.dgsnap | head -1)
rm -rf "${frames:?}"/frame_*.png
(cd viz && RUST_LOG=error "$viewer" --replay "$snap" --window off --minimap off \
    --record "$frames" --camera "$cam" \
    --terrain ../data/bodo_topobathy.tif --scale-max 0.35 --water-alpha 0.85 --ui legend) |
    grep -E "frames \(model" | tail -1
uv run --with pillow --with numpy --with imageio-ffmpeg python scripts/bodo_video/assemble.py \
    "$frames" scripts/bodo_video/tide.cam "$run/stations.csv" \
    data/tide_gauges/bodo_obs_202606.txt "$run/gauge_skill.csv" "$out/bodo_15_days.mp4" \
    "${PREDICTED_CM:-5.0}"

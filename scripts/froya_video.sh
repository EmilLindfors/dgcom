#!/usr/bin/env bash
# The Frøya showcase video (about 45 s, 1920x1080, 30 fps): the tide over
# Frøya-Hitra-Smøla in the nested 3D run, the currents through the Kattholmen farm,
# the storm of 2025-10-09 in wave height, and the storm at sea level by the farm.
#
#   ./scripts/froya_video.sh [out_dir]      # default output/video
#
# Each shot is a `dg-viz --record` of a saved run along a camera path in
# scripts/froya_video/*.cam (see viz/src/record.rs for the format); the captions,
# cards and crossfades are added by scripts/froya_video/assemble.py (ffmpeg).
# Needs the runs' snapshot files (output/froya_3day/3d_rk43, output/storm_viz) and the
# Kattholmen site file (data/sites/14042.txt). About two minutes after the viewer is built.
set -euo pipefail
cd "$(dirname "$0")/.."
out="${1:-output/video}"
paths="$PWD/scripts/froya_video"
frames="$PWD/$out/frames"
mkdir -p "$frames"
(cd viz && cargo build --release)
viewer="$PWD/viz/target/release/dg-viz"
tide=../output/froya_3day/3d_rk43/froya.dgsnap
storm=../output/storm_viz/froya.dgsnap
site=../data/sites/14042.txt

record() { # shot name, then the viewer's options
    local shot=$1
    shift
    rm -rf "${frames:?}/$shot"
    (cd viz && RUST_LOG=error "$viewer" --minimap off --record "$frames/$shot" --camera "$paths/$shot"_*.cam "$@") |
        grep -E "frames \(model" | tail -1
}
record 1 --replay $tide --view domain --sheet off --scale-max 0.35 --water-alpha 0.7
record 2 --replay $tide --site $site --view domain --sheet off --scale-max 0.12 --water-alpha 0.85
record 3 --replay $storm --colour hs --scale-max 8 --view domain
record 4 --replay $storm --site $site --photo on --wind 15,250 --vz 1 --ui off
uv run python scripts/froya_video/assemble.py "$frames" "$out/froya_showcase.mp4"
echo "wrote $out/froya_showcase.mp4"

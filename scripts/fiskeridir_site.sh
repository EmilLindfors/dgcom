#!/usr/bin/env bash
# Fetch a fish farm site from Fiskeridirektoratet into a farm site file
# (`dgcom::io::read_farm_site_file`, TODO F.1):
#
# - the site's register entry (Akvakulturregisteret: name, position,
#   capacity), from the public pub-aqua API;
# - the installation's certified geometry (NYTEK, the installation
#   certificate of an accredited inspection body): the site boundary polygon
#   of the mooring frame and the mooring lines, from Fiskeridirektoratet's
#   NYTEK feature service. NYTEK's coordinates are not quality controlled by
#   Fiskeridirektoratet.
#
# Neither source gives the cages themselves: the reader lays them out on the
# frame (`FarmSite::cage_layout`), or the file can be given `cage` lines by
# hand.
#
#   ./scripts/fiskeridir_site.sh 14042                  # → data/sites/14042.txt
#   ./scripts/fiskeridir_site.sh 14042 out/kattholmen.txt
#
# Needs curl and uv (Python for the JSON).
set -euo pipefail

site="${1:?usage: fiskeridir_site.sh <site number> [output file]}"
out="${2:-data/sites/${site}.txt}"
mkdir -p "$(dirname "$out")"

nytek="https://gis.fiskeridir.no/server/rest/services/NYTEK_FEATURE_API_OGC/FeatureServer"
query="query?where=loknr%3D${site}&outFields=*&outSR=4326&returnGeometry=true&f=json"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

curl -sf "https://api.fiskeridir.no/pub-aqua/api/v1/sites?nr=${site}" \
    -H "accept: application/json" >"$tmp/register.json"
curl -sf "$nytek/8/$query" >"$tmp/boundary.json"
curl -sf "$nytek/7/$query" >"$tmp/moorings.json"

uv run python - "$site" "$tmp" "$out" <<'PY'
import json
import sys

site, tmp, out = sys.argv[1], sys.argv[2], sys.argv[3]
register = [s for s in json.load(open(f"{tmp}/register.json", encoding="utf-8"))
            if str(s.get("siteNr")) == site]
if not register:
    sys.exit(f"site {site} is not in the register")
entry = register[0]
boundary = json.load(open(f"{tmp}/boundary.json", encoding="utf-8")).get("features", [])
moorings = json.load(open(f"{tmp}/moorings.json", encoding="utf-8")).get("features", [])

lines = [
    f"# Fiskeridirektoratet site {site}: Akvakulturregisteret (pub-aqua API) and the",
    "# NYTEK installation certificate (not quality controlled by Fiskeridirektoratet).",
    "# Coordinates are longitude latitude (WGS84, degrees).",
    f"site {site} {entry['name']}",
    f"position {entry['longitude']} {entry['latitude']}",
]
if entry.get("capacity") is not None:
    lines.append(f"capacity {entry['capacity']} {entry.get('capacityUnitType') or ''}".rstrip())
species = ", ".join(s["latinName"] for s in entry.get("speciesLimitations") or [])
if species:
    lines.append(f"# species: {species}")
if boundary:
    attributes = boundary[0]["attributes"]
    lines.append(
        f"# NYTEK certificate from {attributes.get('sertif_fra_dato')} by "
        f"{attributes.get('inspeksjonsorgan')}"
    )
    ring = boundary[0]["geometry"]["rings"][0]
    if ring[0] == ring[-1]:
        ring = ring[:-1]
    lines += [f"boundary {lon:.7f} {lat:.7f}" for lon, lat in ring]
else:
    print(f"warning: no NYTEK boundary for site {site}", file=sys.stderr)
seen = set()
for feature in moorings:
    a = feature["attributes"]
    key = (a["top_lng"], a["top_lat"], a["bottom_lng"], a["bottom_lat"])
    if key in seen:
        continue
    seen.add(key)
    kind = "raft" if "Raft" in (a.get("koord_besk") or "") else "farm"
    lines.append(
        f"mooring {a['top_lng']:.7f} {a['top_lat']:.7f} "
        f"{a['bottom_lng']:.7f} {a['bottom_lat']:.7f} {kind}"
    )
open(out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
print(f"{out}: {entry['name']}, {len(boundary and ring)} boundary points, {len(seen)} mooring lines")
PY

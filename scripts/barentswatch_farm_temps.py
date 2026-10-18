#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Weekly sea temperatures of the fish farms in a box, from BarentsWatch.

    uv run scripts/barentswatch_farm_temps.py .env data/farm_temps/froya.txt 2025 2026

Writes `localityNo lat lon year week seaTemperature name` lines for every
reported week of the sea localities inside 63.2–64.4° N, 7.2–10.0° E (the
Frøya NorKyst window). The fish-health API takes an OAuth client
(`BARENTSWATCH_CLIENT_ID` = `email:clientname`, `BARENTSWATCH_CLIENT_SECRET`
in the git-ignored `.env`), not a user login. The values are farm-reported
weekly means at about 3 m and not very accurate: use them across many farms
with robust statistics (`scripts/farm_temperature_stats.py`), never alone.
"""
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

BOX = (63.2, 64.4, 7.2, 10.0)
BASE = "https://www.barentswatch.no/bwapi"


def load_env(path):
    env = {}
    for line in Path(path).read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def token(env):
    body = urllib.parse.urlencode({
        "grant_type": "client_credentials", "scope": "api",
        "client_id": env["BARENTSWATCH_CLIENT_ID"],
        "client_secret": env["BARENTSWATCH_CLIENT_SECRET"],
    }).encode()
    req = urllib.request.Request("https://id.barentswatch.no/connect/token", data=body,
                                 headers={"Content-Type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)["access_token"]


def get(tok, path):
    req = urllib.request.Request(BASE + path, headers={"Authorization": f"Bearer {tok}"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2 + 3 * attempt)
    return None


def main():
    env = load_env(sys.argv[1])
    out, years = sys.argv[2], [int(y) for y in sys.argv[3:]]
    tok = token(env)
    locs = {}
    for y in years:
        for wk in (10, 23, 35):
            d = get(tok, f"/v1/geodata/fishhealth/locality/{y}/{wk}") or {}
            for l in d.get("localities", []):
                if (BOX[0] <= l["lat"] <= BOX[1] and BOX[2] <= l["lon"] <= BOX[3]
                        and not l.get("isOnLand")):
                    locs[l["localityNo"]] = l
    print(len(locs), "sea localities in the box")
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        f.write("# localityNo lat lon year week seaTemperature name (BarentsWatch fish health, weekly)\n")
        for no, l in sorted(locs.items()):
            for y in years:
                d = get(tok, f"/v1/geodata/fishhealth/locality/{no}/seatemperature/{y}")
                for w in (d or {}).get("data") or []:
                    t = w.get("seaTemperature")
                    if w.get("hasReported") and t is not None:
                        f.write(f'{no} {l["lat"]:.6f} {l["lon"]:.6f} {y} {w["week"]} {t:.3f} '
                                f'{l["name"].replace(" ", "_")}\n')
                        n += 1
    print(n, "weekly values ->", out)


if __name__ == "__main__":
    main()

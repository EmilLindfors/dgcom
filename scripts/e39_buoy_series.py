#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Wave observations from the Norwegian Public Roads Administration's E39 fjord
buoys (Sulafjorden, Breisundet, Vartdalsfjorden, Halsafjorden), from MET
Norway's THREDDS archive (`obs/buoy-svv-e39`, monthly files of 10-minute
values, 2016 to April 2024), as hourly means over a window, for scoring the
spectral wave model (TODO F.4).

Each output line is one buoy and hour:

    buoy  time(UTC ISO)  lon  lat  Hm0(m)  Tp(s)  Tm02(s)  Mdir(deg, from)  n

`Hm0`, `Tp`, `Tm02` are means of the 10-minute values in the hour [t, t+1h),
`Mdir` their circular mean (the direction the waves come *from*); `n` is how
many 10-minute values went in (hours without any are left out). Values outside
0 < Hm0 < 20 m are dropped as spikes (Breisundet logs 14.99 m on 2023-01-09).

The server rate-limits parallel requests (HTTP 429), so the files are fetched
one at a time with backoff.

Usage:
    uv run scripts/e39_buoy_series.py start=2024-02-01 hours=96 \\
        [buoys=D_Breisundet,A_Sulafjorden,B_Sulafjorden,C_Sulafjorden,F_Vartdalsfjorden] \\
        [out=data/wave_buoys/e39_20240201.txt]
"""
import math
import re
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE = "https://thredds.met.no/thredds/dodsC/obs/buoy-svv-e39"
VARIABLES = ["time", "longitude", "latitude", "Hm0", "Tp", "Tm02", "Mdir"]


def fetch(url: str) -> str:
    for attempt in range(8):
        try:
            return urllib.request.urlopen(url, timeout=180).read().decode()
        except Exception as e:  # noqa: BLE001 - retried, then raised
            if "429" in str(e) or "timed out" in str(e):
                time.sleep(5 * (attempt + 1))
                continue
            raise
    raise RuntimeError(f"{url}: still rate-limited")


def arrays(text: str) -> dict[str, list[float]]:
    """The 1D arrays of an OPeNDAP ASCII response."""
    out = {}
    text = text.split("-" * 20, 1)[-1].lstrip("-\n")
    for block in text.split("\n\n"):
        lines = block.strip().split("\n")
        m = re.match(r"^(?:\w+\.)?(\w+)\[\d+\]$", lines[0].strip()) if lines else None
        if m and len(lines) > 1:
            out[m.group(1)] = [float(x) for x in ",".join(lines[1:]).split(",") if x.strip()]
    return out


def month_file(buoy: str, year: int, month: int) -> dict[str, list[float]]:
    url = f"{BASE}/{year}/{month:02d}/{year}{month:02d}_E39_{buoy}_wave.nc"
    dds = fetch(url + ".dds")
    n = int(re.search(r"time\[time = (\d+)\]", dds).group(1))
    query = ",".join(f"{v}[0:1:{n - 1}]" for v in VARIABLES)
    return arrays(fetch(url + ".ascii?" + query.replace("[", "%5B").replace("]", "%5D")))


def main() -> None:
    args = dict(a.split("=", 1) for a in sys.argv[1:])
    start = datetime.fromisoformat(args.get("start", "2024-02-01")).replace(tzinfo=timezone.utc)
    hours = int(args.get("hours", "96"))
    end = start + timedelta(hours=hours)
    buoys = args.get(
        "buoys", "D_Breisundet,A_Sulafjorden,B_Sulafjorden,C_Sulafjorden,F_Vartdalsfjorden"
    ).split(",")
    out = Path(args.get("out", f"data/wave_buoys/e39_{start:%Y%m%d}.txt"))

    months = sorted({(t.year, t.month) for t in (start + timedelta(hours=h) for h in range(hours + 1))})
    lines = [
        f"# E39 buoys (Statens vegvesen, MET Norway THREDDS obs/buoy-svv-e39), hourly means from {start:%Y-%m-%dT%H:%MZ}, {hours} h",
        "# buoy time lon lat Hm0_m Tp_s Tm02_s Mdir_from_deg n",
    ]
    for buoy in buoys:
        records = {k: [] for k in VARIABLES}
        for year, month in months:
            try:
                data = month_file(buoy, year, month)
            except Exception as e:  # noqa: BLE001 - a missing month is reported, not fatal
                print(f"{buoy} {year}-{month:02d}: {e}", file=sys.stderr)
                continue
            for k in VARIABLES:
                records[k].extend(data.get(k, []))
        hourly: dict[datetime, list[tuple]] = {}
        for i, t in enumerate(records["time"]):
            when = datetime.fromtimestamp(t, timezone.utc)
            if not (start <= when < end):
                continue
            hm0 = records["Hm0"][i]
            if not (0.0 < hm0 < 20.0):
                continue
            hour = when.replace(minute=0, second=0, microsecond=0)
            hourly.setdefault(hour, []).append(
                tuple(records[k][i] for k in ["longitude", "latitude", "Hm0", "Tp", "Tm02", "Mdir"])
            )
        for hour in sorted(hourly):
            rows = hourly[hour]
            n = len(rows)
            mean = [sum(r[j] for r in rows) / n for j in range(5)]
            s = sum(math.sin(math.radians(r[5])) for r in rows)
            c = sum(math.cos(math.radians(r[5])) for r in rows)
            mdir = math.degrees(math.atan2(s, c)) % 360.0
            lines.append(
                f"{buoy} {hour:%Y-%m-%dT%H:%M:%SZ} {mean[0]:.5f} {mean[1]:.5f} "
                f"{mean[2]:.3f} {mean[3]:.2f} {mean[4]:.2f} {mdir:.1f} {n}"
            )
        print(f"{buoy}: {len(hourly)} hours", file=sys.stderr)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()

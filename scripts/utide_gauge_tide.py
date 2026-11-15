#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "utide",
# ]
# ///
"""
A gauge's tide from UTide (Codiga 2011): an analysis independent of ours (its
own astronomy, constituent selection and solver), as the judge of
`examples/tide_inference_skill.rs` (notes/validation.md, P3.1 "Inference of
the minor constituents").

    uv run scripts/utide_gauge_tide.py <out dir> <fit file> [<more files to predict> ...]

Fits the whole of <fit file> (gauge format, `scripts/kartverket_gauge.sh`)
with UTide's automatic constituent selection, and writes

- `<out dir>/<id>_tide.txt`: the fit's reconstruction at the instants of the
  fit file and of the other files, without the mean and without the
  long-period constituents (under 0.5 cycles a day: Sa, Ssa, Msm, Mm, Msf,
  Mf, which are mostly weather), in the gauge format;
- `<out dir>/<id>_constants.csv`: name, frequency (cph), H (m), G (°).
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import utide


def read_series(path):
    meta, t, v = {}, [], []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            key, _, value = line[1:].partition(":")
            meta.setdefault(key.strip().lower(), value.strip())
            continue
        a, b = line.split()[:2]
        t.append(datetime.strptime(a, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp())
        v.append(float(b))
    t, v = np.array(t), np.array(v)
    keep = np.isfinite(v)
    return meta, t[keep], v[keep]


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    meta, t, v = read_series(sys.argv[2])
    lat = float(meta["latitude"])
    code = meta["id"]
    dates = lambda t: t.astype("int64").astype("datetime64[s]")
    coef = utide.solve(dates(t), v, lat=lat, method="ols", conf_int="none", verbose=False,
                       trend=False, nodal=True)
    tidal = [n for n, f in zip(coef.name, coef.aux.frq) if f > 0.5 / 24.0]

    with open(out / f"{code}_constants.csv", "w", encoding="utf-8") as f:
        f.write("name,frequency_cph,h_m,g_deg\n")
        for n, frq, a, g in zip(coef.name, coef.aux.frq, coef.A, coef.g):
            f.write(f"{n},{frq:.8f},{a:.5f},{g:.2f}\n")

    times = [t]
    for path in sys.argv[3:]:
        times.append(read_series(path)[1])
    times = np.unique(np.concatenate(times))
    tide = utide.reconstruct(dates(times), coef, constit=tidal, verbose=False).h - coef.mean
    with open(out / f"{code}_tide.txt", "w", encoding="utf-8") as f:
        f.write(f"# UTide reconstruction of {Path(sys.argv[2]).name}, {len(coef.name)} constituents fitted, "
                f"{len(tidal)} kept (no mean, no long-period)\n")
        for key in ("station", "id", "latitude", "longitude"):
            f.write(f"# {key}: {meta[key]}\n")
        f.write("# units: m\n# columns: datetime water_level(m)\n")
        for ti, hi in zip(times, tide):
            stamp = datetime.fromtimestamp(ti, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            f.write(f"{stamp} {hi:.4f}\n")
    print(f"{code}: {len(t)} samples, {len(coef.name)} constituents, {len(tidal)} tidal, "
          f"tide RMS {np.sqrt(np.mean(tide**2)):.3f} m, residual RMS "
          f"{np.sqrt(np.mean((v - utide.reconstruct(dates(t), coef, verbose=False).h)**2)):.3f} m")


if __name__ == "__main__":
    main()

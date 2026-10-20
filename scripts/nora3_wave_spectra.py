#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["netCDF4", "numpy"]
# ///
"""
Boundary spectra for the spectral wave model from MET Norway's NORA3 wave
hindcast (WINDSURFER, WAM cycle 4.7 at 3 km forced by the NORA3 HARMONIE-AROME
hindcast; hourly 2D spectra at ≈ 28,700 points, densest along the Norwegian
coast, 1959 to now), in the layout of MET's MyWave WAM 800 m spectra files,
which `io::WaveSpectraFile` reads (`froya_real_data wave_spectra=<file>`).

For past windows, where the MyWave forecasts are gone, this is the parent: its
spectra at the points inside `bbox` (with `margin` degrees around it), and at
the point nearest each of `points=lon,lat;…` (e.g. wave buoys, so the run's
`wave_points.txt` compares the DG model with WAM there too).

The daily files (`windsurfer/mywavewam3km_spectra/YYYY/MM/SPCYYYYMMDD00.nc`,
24 hours each, ≈ 2 GB) are read over OPeNDAP only at the chosen points, one
request per contiguous run of point indices.

- `SPEC[time][y][x][freq][direction]`: variance density in m²/(Hz·rad), as
  MET writes it (`m**2 s`); 30 frequencies 0.0345–0.5476 Hz (ratio 1.1),
  24 directions, the direction the waves travel *to* (oceanographic, as the
  files' `direction_convention` says);
- `hs` = 4 √(∫∫ SPEC df dθ) with bin widths from the frequency centres and θ
  in radians, `tp` = 1/f at the peak of the frequency spectrum, both
  computed here (the spectra files carry neither); `depth` and the wind are
  not in them.

Usage:
    uv run scripts/nora3_wave_spectra.py bbox=5.45,62.15,6.40,62.65 \\
        start=2024-01-31 hours=120 [margin=0.1] \\
        [points=5.9315,62.4453;6.0456,62.4265] [out=data/alesund_nora3_spectra.nc]
"""
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import netCDF4
import numpy as np

BASE = "https://thredds.met.no/thredds/dodsC/windsurfer/mywavewam3km_spectra"


def url(day: datetime) -> str:
    return f"{BASE}/{day:%Y}/{day:%m}/SPC{day:%Y%m%d}00.nc"


def open_retry(u: str) -> netCDF4.Dataset:
    for attempt in range(8):
        try:
            return netCDF4.Dataset(u)
        except OSError as e:
            print(f"  {u}: {e}; retrying", file=sys.stderr)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"cannot open {u}")


def runs(indices: list[int]) -> list[tuple[int, int]]:
    """Contiguous runs [a, b) of sorted indices."""
    out = []
    for i in indices:
        if out and out[-1][1] == i:
            out[-1] = (out[-1][0], i + 1)
        else:
            out.append((i, i + 1))
    return out


def main() -> None:
    args = dict(a.split("=", 1) for a in sys.argv[1:])
    lon0, lat0, lon1, lat1 = map(float, args.get("bbox", "5.45,62.15,6.40,62.65").split(","))
    margin = float(args.get("margin", "0.1"))
    start = datetime.fromisoformat(args.get("start", "2024-01-31")).replace(tzinfo=timezone.utc)
    hours = int(args.get("hours", "120"))
    extra = [tuple(map(float, p.split(","))) for p in args.get("points", "").split(";") if p]
    out = Path(args.get("out", "data/alesund_nora3_spectra.nc"))
    end = start + timedelta(hours=hours)

    first = open_retry(url(start))
    lon = first["longitude"][0, :].astype(float)
    lat = first["latitude"][0, :].astype(float)
    freq = first["freq"][:].astype(float)
    direction = first["direction"][:].astype(float)
    convention = getattr(first, "direction_convention", "")
    first.close()

    inside = np.flatnonzero(
        (lon >= lon0 - margin) & (lon <= lon1 + margin) & (lat >= lat0 - margin) & (lat <= lat1 + margin)
    ).tolist()
    for plon, plat in extra:
        # Nearest by distance on the sphere's local tangent plane
        d = np.hypot((lon - plon) * np.cos(np.radians(plat)), lat - plat)
        k = int(np.argmin(d))
        print(f"  point {plon:.4f},{plat:.4f}: NORA3 {k} at {lon[k]:.4f},{lat[k]:.4f} "
              f"({d[k] * 111.0:.2f} km)", file=sys.stderr)
        inside.append(k)
    chosen = sorted(set(inside))
    print(f"{len(chosen)} points in {len(runs(chosen))} runs", file=sys.stderr)

    times, spec = [], []
    day = start.replace(hour=0)
    while day <= end:
        ds = open_retry(url(day))
        t = netCDF4.num2date(ds["time"][:], ds["time"].units, only_use_cftime_datetimes=False)
        t = np.array([x.replace(tzinfo=timezone.utc).timestamp() for x in t])
        keep = np.flatnonzero((t >= start.timestamp()) & (t <= end.timestamp()))
        if keep.size:
            a, b = int(keep[0]), int(keep[-1]) + 1
            parts = []
            for i0, i1 in runs(chosen):
                for attempt in range(8):
                    try:
                        parts.append(np.asarray(ds["SPEC"][a:b, 0, i0:i1, :, :], dtype=np.float64))
                        break
                    except (OSError, RuntimeError) as e:
                        print(f"  {day:%Y-%m-%d} [{i0}:{i1}]: {e}; retrying", file=sys.stderr)
                        time.sleep(5 * (attempt + 1))
                else:
                    raise RuntimeError("OPeNDAP keeps failing")
            times.extend(t[a:b])
            spec.append(np.concatenate(parts, axis=1))
            print(f"  {day:%Y-%m-%d}: {b - a} hours", file=sys.stderr)
        ds.close()
        day += timedelta(days=1)
    spec = np.concatenate(spec, axis=0)  # [time][point][freq][dir]
    spec = np.where(np.isfinite(spec) & (spec >= 0), spec, 0.0)

    # Bin widths from the frequency centres
    edges = np.concatenate([[freq[0] - 0.5 * (freq[1] - freq[0])],
                            0.5 * (freq[1:] + freq[:-1]),
                            [freq[-1] + 0.5 * (freq[-1] - freq[-2])]])
    df = np.diff(edges)
    dtheta = 2.0 * np.pi / direction.size
    e_f = spec.sum(axis=3) * dtheta  # [time][point][freq]
    hs = 4.0 * np.sqrt((e_f * df).sum(axis=2))
    tp = 1.0 / freq[np.argmax(e_f, axis=2)]

    nt, npnt = spec.shape[0], spec.shape[1]
    out.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(out, "w") as nc:
        nc.title = "NORA3 wave hindcast (WINDSURFER, WAM 3 km) 2D spectra, subset"
        nc.source = f"{BASE} (MET Norway, CC BY 4.0)"
        nc.direction_convention = convention or "propagating towards (oceanographic)"
        nc.history = f"scripts/nora3_wave_spectra.py {' '.join(sys.argv[1:])}"
        nc.createDimension("time", nt)
        nc.createDimension("y", 1)
        nc.createDimension("x", npnt)
        nc.createDimension("freq", freq.size)
        nc.createDimension("direction", direction.size)
        v = nc.createVariable("time", "f8", ("time",))
        v.units = "seconds since 1970-01-01 00:00:00"
        v.standard_name = "time"
        v[:] = np.array(times)
        v = nc.createVariable("freq", "f8", ("freq",))
        v.units = "1/s"
        v[:] = freq
        v = nc.createVariable("direction", "f8", ("direction",))
        v.units = "degree"
        v[:] = direction
        for name, values, units in [("latitude", lat, "degree_north"), ("longitude", lon, "degree_east")]:
            v = nc.createVariable(name, "f8", ("y", "x"))
            v.units = units
            v.standard_name = name
            v[0, :] = values[chosen]
        v = nc.createVariable("SPEC", "f8", ("time", "y", "x", "freq", "direction"))
        v.units = "m**2 s"
        v.long_name = "2-D spectrum of total sea"
        v[:] = spec[:, None, :, :, :]
        for name, values, units in [("hs", hs, "m"), ("tp", tp, "s")]:
            v = nc.createVariable(name, "f8", ("time", "y", "x"))
            v.units = units
            v[:] = values[:, None, :]
        v = nc.createVariable("nora3_index", "i4", ("x",))
        v[:] = np.array(chosen)
    print(f"wrote {out}: {nt} times × {npnt} points", file=sys.stderr)
    for i, k in enumerate(chosen):
        print(f"  {k:6d} {lon[k]:.4f} {lat[k]:.4f}  H_s max {hs[:, i].max():.2f} m", file=sys.stderr)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["pyTMD>=3.1", "numpy", "xarray", "netCDF4", "scipy", "dask"]
# ///
"""
A tidal atlas for an open boundary from a global tide model (TPXO, FES, GOT,
… : any model pyTMD's database knows), in the text format
`boundary::TidalAtlas` reads (`[boundary.atlas]` or `[boundary.nesting] atlas`
in a case file).

It is the global counterpart of `examples/norkyst_boundary_tides.rs`, which fits
the atlas to a month of NorKyst-800: for a coast without a regional model with
a known tide (Chiloé, `cases/chiloe`), or to correct a parent's tide against
an independent one.

The points lie along the sides of `bbox` every `spacing_km` (the open boundary
of a `[mesh.geographic]` box), plus any `points=lon,lat;…` (tide gauges: give
them `points_out=` to write them to their own file, as the station atlas of a
`[boundary.*.gauge]` correction). At each point it writes the model's
elevation amplitude and Greenwich phase lag and, where the model has them, the
depth-mean eastward and northward currents (transports over the model's
depth), for the constituents dgcom supports (`tides::CONSTITUENT_NAMES`;
others, e.g. TPXO's S1, are left out and listed).

Points on the model's land are filled from the nearest model water within
`extrapolate_km` (default 10; the model's coast is coarser than the mesh's),
and dropped beyond it. Their depth column is then the model's depth at the
nearest water point.

The models are not downloadable without registration: TPXO10-atlas from
https://www.tpxo.net (free for academic use), FES2022 from AVISO. Put them under
`model_dir` as pyTMD's database expects (e.g. `TPXO10_atlas_v2/*.nc` for
`TPXO10-atlas-v2-nc`; TPXO's own zip unpacks to `TPXO10_atlas_v2_nc/`, which
has to be renamed).

Usage:
    uv run scripts/tide_model_boundary_atlas.py bbox=-75.3,-44.2,-72.3,-41.3 \\
        model=TPXO10-atlas-v2-nc model_dir=data/tide_models \\
        [spacing_km=2] [extrapolate_km=10] \\
        [points=-72.9608,-41.4850;-73.8327,-41.8674] [points_out=data/chiloe_station_tides.txt] \\
        out=data/chiloe_boundary_tides.txt
"""
import math
import sys
from pathlib import Path

import numpy as np
import xarray as xr

# `tides::CONSTITUENT_NAMES`, in their conventional spelling
SUPPORTED = [
    "M2", "S2", "N2", "K2", "2N2", "MU2", "NU2", "LDA2", "L2", "K1", "O1", "P1",
    "Q1", "M4", "MS4", "MN4", "M6", "Mf", "Mm", "Ssa",
]
# pyTMD's spellings that differ from dgcom's beyond case
ALIASES = {"lambda2": "LDA2", "la2": "LDA2", "lam2": "LDA2", "sa": None}
EARTH_RADIUS_KM = 6371.0


def parse_args(argv):
    args = {}
    for arg in argv:
        key, sep, value = arg.partition("=")
        if not sep:
            sys.exit(f"argument {arg!r} is not key=value")
        args[key] = value
    for key in ("bbox", "model", "model_dir", "out"):
        if key not in args:
            sys.exit(f"missing {key}=; see the docstring")
    return args


def dgcom_name(name):
    """dgcom's spelling of a pyTMD constituent, or None if unsupported."""
    lower = name.lower()
    if lower in ALIASES:
        return ALIASES[lower]
    return next((c for c in SUPPORTED if c.lower() == lower), None)


def perimeter(bbox, spacing_km):
    """Points along the sides of bbox = (west, south, east, north), corners once."""
    west, south, east, north = bbox
    lat_mid = math.radians(0.5 * (south + north))
    km_per_deg_lat = math.pi * EARTH_RADIUS_KM / 180.0
    km_per_deg_lon = km_per_deg_lat * math.cos(lat_mid)

    def side(a, b, km_per_deg):
        """From a towards b, without b (the next side's first point)."""
        n = max(1, math.ceil(abs(b - a) * km_per_deg / spacing_km))
        return np.linspace(a, b, n + 1)[:-1]

    sides = [  # counter-clockwise from the south-west corner
        (side(west, east, km_per_deg_lon), south),  # south side
        (east, side(south, north, km_per_deg_lat)),  # east side
        (side(east, west, km_per_deg_lon), north),  # north side
        (west, side(north, south, km_per_deg_lat)),  # west side
    ]
    lons, lats = [], []
    for x, y in sides:
        x, y = np.broadcast_arrays(x, y)
        lons.extend(x)
        lats.extend(y)
    return np.array(lons, dtype=float), np.array(lats, dtype=float)


def at_points(ds, lon, lat, extrapolate_km):
    """The model's complex constants at the points: {name: complex array}, and
    the model depth (NaN if the model has none)."""
    x = xr.DataArray(lon, dims="point")
    y = xr.DataArray(lat, dims="point")
    other = ds.tmd.interp(
        x, y, extrapolate=True, cutoff=extrapolate_km, k=1  # km
    )
    constants = {c: other[c].values for c in other.tmd.constituents}
    depth = other["bathymetry"].values if "bathymetry" in other else np.full(len(lon), np.nan)
    return constants, depth


def to_units(ds, units):
    try:
        return ds.tmd.to_units(units)
    except Exception as e:  # pint cannot convert: the model's units are odd
        sys.exit(f"cannot convert the model to {units}: {e}")


def atlas_lines(lon, lat, elevation, currents, depth, names, skipped):
    """Atlas lines per point (constituents in `names` order); points where any
    constituent is missing are dropped and counted in `skipped`."""
    lines = []
    for k in range(len(lon)):
        point = []
        for source, name in names:
            eta = elevation[source][k]
            if not np.isfinite(eta):
                break
            fields = [lon[k], lat[k], depth[k], name, abs(eta), lag(eta)]
            if currents is not None:
                u, v = currents[0][source][k], currents[1][source][k]
                if not (np.isfinite(u) and np.isfinite(v)):
                    break
                fields += [abs(u), lag(u), abs(v), lag(v)]
            point.append(fields)
        else:
            lines.extend(point)
            continue
        skipped.append((lon[k], lat[k]))
    return lines


def lag(z):
    """Greenwich phase lag (degrees, 0–360) of a complex constant H e^{−iG}."""
    return math.degrees(math.atan2(-z.imag, z.real)) % 360.0


def write_atlas(path, header, lines):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for line in header:
            f.write(f"# {line}\n")
        f.write("# lon lat depth name eta_amp eta_lag [u_amp u_lag v_amp v_lag]\n")
        for fields in lines:
            lon, lat, depth, name, *rest = fields
            depth = "nan" if not np.isfinite(depth) else f"{depth:.1f}"
            values = " ".join(
                f"{v:.6f}" if i % 2 == 0 else f"{v:.2f}" for i, v in enumerate(rest)
            )
            f.write(f"{lon:.5f} {lat:.5f} {depth} {name} {values}\n")
    return len(lines)


def main():
    import pyTMD.io  # registers the `.tmd` accessors

    args = parse_args(sys.argv[1:])
    bbox = [float(v) for v in args["bbox"].split(",")]
    if len(bbox) != 4 or not (bbox[0] < bbox[2] and bbox[1] < bbox[3]):
        sys.exit("bbox must be west,south,east,north")
    spacing = float(args.get("spacing_km", 2.0))
    extrapolate_km = float(args.get("extrapolate_km", 10.0))
    lon, lat = perimeter(bbox, spacing)
    stations = []
    if args.get("points"):
        stations = [tuple(float(v) for v in p.split(",")) for p in args["points"].split(";")]

    model = args["model"]
    directory = args["model_dir"]
    margin = 1.0  # degrees around bbox and the points for the crop
    xs = [bbox[0], bbox[2]] + [p[0] for p in stations]
    ys = [bbox[1], bbox[3]] + [p[1] for p in stations]
    bounds = [min(xs) - margin, max(xs) + margin, min(ys) - margin, max(ys) + margin]

    def open_group(group):
        # Lazily (dask), so only the crop is read: TPXO10-atlas's global 1/30°
        # grids are 0.9 GB per constituent in memory, and 15 of them per
        # group do not fit
        try:
            ds = pyTMD.io.open_dataset(model, directory=directory, group=group, chunks="auto")
        except TypeError:  # a format whose reader takes no `chunks`
            ds = pyTMD.io.open_dataset(model, directory=directory, group=group)
        return ds.tmd.crop(bounds).compute()

    z = to_units(open_group("z"), "m")
    try:
        u = to_units(open_group("u"), "m/s")
        v = to_units(open_group("v"), "m/s")
    except (KeyError, ValueError, FileNotFoundError) as e:
        print(f"no currents in {model} ({e}): elevation only", file=sys.stderr)
        u = v = None

    names, unsupported = [], []
    for c in z.tmd.constituents:
        name = dgcom_name(c)
        if name is None:
            unsupported.append(c)
        elif u is None or (c in u.tmd.constituents and c in v.tmd.constituents):
            names.append((c, name))
    if unsupported:
        print(f"left out (not in dgcom's catalogue): {', '.join(unsupported)}", file=sys.stderr)
    if not names:
        sys.exit(f"{model} has no constituent dgcom supports")

    header = [
        f"{model} at the open boundary of bbox {args['bbox']} (spacing {spacing} km)",
        f"written by scripts/tide_model_boundary_atlas.py; land points filled from "
        f"model water within {extrapolate_km} km",
        f"constituents: {' '.join(n for _, n in names)}",
    ]

    def build(lon, lat):
        eta, depth = at_points(z, lon, lat, extrapolate_km)
        cur = None
        if u is not None:
            cur = (at_points(u, lon, lat, extrapolate_km)[0], at_points(v, lon, lat, extrapolate_km)[0])
        skipped = []
        return atlas_lines(lon, lat, eta, cur, depth, names, skipped), skipped

    lines, skipped = build(lon, lat)
    n = write_atlas(args["out"], header, lines)
    print(f"{args['out']}: {n // len(names)} points × {len(names)} constituents "
          f"({len(skipped)} of {len(lon)} on the model's land beyond {extrapolate_km} km, dropped)")
    if stations:
        out = args.get("points_out") or args["out"].replace(".txt", "_points.txt")
        s_lon = np.array([p[0] for p in stations])
        s_lat = np.array([p[1] for p in stations])
        lines, skipped = build(s_lon, s_lat)
        if skipped:
            print(f"stations without model water within {extrapolate_km} km: {skipped}", file=sys.stderr)
        n = write_atlas(out, header[:1] + ["at the given points"] + header[2:], lines)
        print(f"{out}: {n // len(names)} points")


if __name__ == "__main__":
    main()

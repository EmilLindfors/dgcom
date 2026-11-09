#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["copernicusmarine>=2.0", "xarray", "netCDF4", "numpy"]
# ///
"""
A parent-model file for nesting from Copernicus Marine's global ocean
analysis and forecast (GLOBAL_ANALYSISFORECAST_PHY_001_024, 1/12°), in one
NetCDF that `io::OceanModelReader::from_file` reads (`[boundary.nesting] file`
in a case file, or 3D nesting's profiles with `from_file_with_profiles`).

It is the global counterpart of `examples/norkyst_nesting_subset.rs`, for a
coast without a regional model: Chile (`cases/chiloe`), or anywhere else.
Copernicus Marine splits the product by variable, so this merges, on one
time axis:

- `uo`, `vo`, `thetao`, `so`: the 3D fields on z-levels (`depth`, positive
  down), every 6 hours (`period=6h`, instantaneous) or daily (`period=1d`,
  daily means);
- `zos`: the sea surface height above the geoid, from the hourly-mean 2D
  dataset at the same times (6h) or the daily mean (1d);
- `deptho` and `mask`: the model's depth and land mask (static).

What the reader does with it: `zos` is the level, `uo`/`vo` averaged over the
column to `deptho` the depth-mean current, the surface `thetao`/`so` the
tracers (and the whole profiles for 3D nesting).

Mind the tide: at 1/12° the model resolves no coastal tide (Chiloé's inner
sea, a fjord), and the daily means have none at all. Nest in it for the
residual (subtidal) flow and the tracers, with the tide from a tidal atlas
(`scripts/tide_model_boundary_atlas.py`); the nesting mode that adds an
atlas tide to a parent without one is #215. Its `zos` is above the geoid, not a tidal
datum: `[boundary.nesting] level_m` (or the atlas's `Z0`) sets the datum.

Needs a (free) Copernicus Marine account: `copernicusmarine login` once, or
COPERNICUSMARINE_SERVICE_USERNAME/PASSWORD in the environment (`.env` is
git-ignored).

Usage:
    uv run scripts/cmems_parent_subset.py bbox=-74.6,-44.2,-72.3,-41.3 \\
        start=2025-11-12 days=3 [period=6h] [max_depth=500] [margin=0.25] \\
        out=data/chiloe_cmems.nc
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import xarray as xr

PRODUCT = "GLOBAL_ANALYSISFORECAST_PHY_001_024"
DATASETS = {
    "6h": {
        "uo": "cmems_mod_glo_phy-cur_anfc_0.083deg_PT6H-i",
        "vo": "cmems_mod_glo_phy-cur_anfc_0.083deg_PT6H-i",
        "thetao": "cmems_mod_glo_phy-thetao_anfc_0.083deg_PT6H-i",
        "so": "cmems_mod_glo_phy-so_anfc_0.083deg_PT6H-i",
        "zos": "cmems_mod_glo_phy_anfc_0.083deg_PT1H-m",
    },
    "1d": {
        "uo": "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",
        "vo": "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",
        "thetao": "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m",
        "so": "cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m",
        "zos": "cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
    },
}
STATIC = "cmems_mod_glo_phy_anfc_0.083deg_static"


def parse_args(argv):
    args = {}
    for arg in argv:
        key, sep, value = arg.partition("=")
        if not sep:
            sys.exit(f"argument {arg!r} is not key=value")
        args[key] = value
    for key in ("bbox", "start", "out"):
        if key not in args:
            sys.exit(f"missing {key}=; see the docstring")
    return args


def main():
    import copernicusmarine as cm

    args = parse_args(sys.argv[1:])
    west, south, east, north = (float(v) for v in args["bbox"].split(","))
    margin = float(args.get("margin", 0.25))  # degrees: room for the stencils
    period = args.get("period", "6h")
    if period not in DATASETS:
        sys.exit(f"period must be one of {', '.join(DATASETS)}")
    start = datetime.fromisoformat(args["start"])
    end = start + timedelta(days=float(args.get("days", 3)))
    max_depth = float(args.get("max_depth", 500.0))
    region = dict(
        minimum_longitude=west - margin,
        maximum_longitude=east + margin,
        minimum_latitude=south - margin,
        maximum_latitude=north + margin,
    )

    def open_(dataset_id, variables, **extra):
        print(f"{dataset_id}: {', '.join(variables)}", file=sys.stderr)
        return cm.open_dataset(dataset_id=dataset_id, variables=variables, **region, **extra)

    when = dict(start_datetime=start.isoformat(), end_datetime=end.isoformat())
    depth = dict(minimum_depth=0.0, maximum_depth=max_depth)
    ids = DATASETS[period]
    by_dataset = {}
    for var in ("uo", "vo", "thetao", "so"):
        by_dataset.setdefault(ids[var], []).append(var)
    fields = [open_(ds_id, variables, **when, **depth) for ds_id, variables in by_dataset.items()]
    merged = xr.merge(fields, join="exact")

    # The level at the 3D fields' times (hourly means centred on the half
    # hour: the nearest within 1 h)
    zos = open_(ids["zos"], ["zos"], **when)["zos"]
    zos = zos.sel(time=merged.time, method="nearest", tolerance=np.timedelta64(1, "h"))
    merged["zos"] = zos.assign_coords(time=merged.time)

    static = open_(STATIC, ["deptho", "mask"], dataset_part="bathy")
    for name in ("deptho", "mask"):
        if name in static:
            field = static[name]
            if "depth" in field.dims:  # the 3D mask: its surface level
                field = field.isel(depth=0, drop=True)
            merged[name] = field.reindex_like(merged[["latitude", "longitude"]], method="nearest")

    merged.attrs.update(
        title=f"{PRODUCT} subset for dgcom nesting",
        source=", ".join(sorted({*by_dataset, ids['zos'], STATIC})),
        history=f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%MZ} scripts/cmems_parent_subset.py {' '.join(sys.argv[1:])}",
    )
    merged["depth"].attrs.setdefault("positive", "down")
    out = Path(args["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_netcdf(out)
    print(
        f"{out}: {merged.sizes['time']} times, {merged.sizes['depth']} levels, "
        f"{merged.sizes['latitude']} × {merged.sizes['longitude']} points"
    )


if __name__ == "__main__":
    main()

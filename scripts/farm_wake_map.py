#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "matplotlib",
# ]
# ///
"""
Maps of a farm case's wake and residual current (TODO F.1), from the
`farm_wake.csv` that `froya_real_data site=<file>` writes: the relative
deficit of the summed speeds 1 − Σ|u_cages|/Σ|u_open| for each direction of
the tide, and the residual current (the analysis window's mean) without the
cages and its change with them, with the cages of `farm_cages.csv`.

    uv run scripts/farm_wake_map.py output/kattholmen_farm [out=<file.png>] [extent=800]
"""

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    args = [a for a in sys.argv[1:] if "=" not in a]
    opts = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    if not args:
        sys.exit(__doc__)
    run = Path(args[0])
    out = Path(opts.get("out", run / "farm_wake.png"))
    extent = float(opts.get("extent", 800))

    with open(run / "farm_wake.csv", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = np.array([[float(v) for v in row] for row in reader])
    col = {name: i for i, name in enumerate(header)}
    x, y = rows[:, col["x_m"]], rows[:, col["y_m"]]
    xs, ys = np.unique(x), np.unique(y)
    shape = (len(ys), len(xs))

    def field(name: str) -> np.ndarray:
        return rows[:, col[name]].reshape(shape)

    deficits = [h for h in header if h.startswith("deficit_")]
    if not deficits:
        sys.exit("farm_wake.csv has no deficit columns: the run had no reference without cages")
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.6), constrained_layout=True)
    window = (xs >= -extent) & (xs <= extent)
    wy = (ys >= -extent) & (ys <= extent)
    sub = np.ix_(wy, window)
    X, Y = np.meshgrid(xs[window], ys[wy])

    for ax, name in zip(axes[:2], deficits):
        d = 100 * field(name)[sub]
        im = ax.pcolormesh(X, Y, d, cmap="RdBu_r", vmin=-20, vmax=20, shading="auto")
        ax.contour(X, Y, d, levels=[5, 10, 20], colors="k", linewidths=0.6)
        ax.set_title(f"Speed deficit (%), current toward {name.split('_')[1]}°")
        fig.colorbar(im, ax=ax, shrink=0.8)

    u0, v0 = field("residual_u_open")[sub], field("residual_v_open")[sub]
    u1, v1 = field("residual_u_cages")[sub], field("residual_v_cages")[sub]
    speed = np.hypot(u0, v0)
    # A drying shoal's metres per second would set the scales
    top = np.nanpercentile(speed, 98)
    im = axes[2].pcolormesh(X, Y, speed, cmap="viridis", vmin=0, vmax=top, shading="auto")
    step = max(1, len(xs[window]) // 25)
    q = (slice(None, None, step), slice(None, None, step))
    axes[2].quiver(X[q], Y[q], u0[q], v0[q], color="w", scale=20 * top, width=0.003)
    axes[2].set_title("Residual current without the cages (m/s)")
    fig.colorbar(im, ax=axes[2], shrink=0.8)

    change = np.hypot(u1 - u0, v1 - v0)
    top = np.nanpercentile(change, 99.5)
    im = axes[3].pcolormesh(
        X, Y, 1000 * change, cmap="magma", vmin=0, vmax=1000 * top, shading="auto"
    )
    axes[3].quiver(
        X[q], Y[q], (u1 - u0)[q], (v1 - v0)[q], color="c", scale=20 * top, width=0.003
    )
    axes[3].set_title("Change of the residual with the cages (mm/s)")
    fig.colorbar(im, ax=axes[3], shrink=0.8)

    cages = run / "farm_cages.csv"
    circles = np.loadtxt(cages, delimiter=",", skiprows=1, ndmin=2) if cages.exists() else []
    for ax in axes:
        for cx, cy, r in circles:
            ax.add_patch(plt.Circle((cx, cy), r, fill=False, color="k", linewidth=0.8))
        ax.set_aspect("equal")
        ax.set_xlabel("east of the farm's centre (m)")
        ax.set_facecolor("0.8")
    axes[0].set_ylabel("north of the farm's centre (m)")
    fig.savefig(out, dpi=110)
    print(out)


if __name__ == "__main__":
    main()

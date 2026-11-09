#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "gmsh>=4.11",
#     "netCDF4",
#     "numpy",
#     "scipy",
#     "rasterio",
#     "shapely>=2",
# ]
# ///
"""
Coastline-fitted all-quad mesh of a coastal domain from an elevation model
(Kartverket's topobathy model from scripts/kartverket_topobathy.sh), for
examples/froya_real_data.rs (`mesh=`).

1. Land is where the model is above 0 m. Water not connected to the edge of
   the box (lakes, and sea pockets cut off at this resolution) becomes land.
2. Land features narrower than `min_land` (m) are removed (an opening of the
   land polygons: erode, then dilate): skerries and thin points the mesh
   cannot resolve. They stay in the bed as shoals, which the wet/dry scheme
   dries. Removing land never closes a sound; the opposite choice (closing
   the water) would block the through-flow that the Mausund comparison showed
   a coarse bed already chokes (TODO P3.1).
3. Islands smaller than `min_island` (m²) are dropped, coastlines are
   simplified and then resampled at the coastal element size, so that no
   boundary segment is much shorter than the elements next to it.
4. Gmsh meshes the water with a size field growing from `coast` (m) at the
   coastline to `far` (m) `dist` metres offshore, optionally refined to
   `farm_size` within `farm_radius` of `farm=lon,lat`, and to `river_size`
   within `river_radius` of the river mouths of `rivers=<file>` (a river
   file as `examples/norkyst_river_subset.rs` writes it, `river_lon`,
   `river_lat`; the mouths inside the box). A river enters an element whose
   nodes are all under water (`source::river`), so on a coarse coast its
   mouth moves to the nearest such element, a kilometre or more away; at
   `river_size` it moves less than an element. Near the mouths the water
   narrower than `river_min_water` (default 2·`river_size`) is kept, not
   `min_water` (estuaries and fjord heads stay open), and the coastline is
   resampled at the local size. The size grows away from a mouth as fast as
   from the coast (`(far − coast)/dist`). Recombination and
   subdivision give an all-quad mesh (sizes here are after subdivision).
   Recombination only pairs triangles into quads of quality at least
   `recombine_quality` (0.3); the other triangles become three quads each on
   subdivision. Gmsh's default (0.01) accepts nearly degenerate quads, and
   the corners of up to 179° they leave after subdivision (mostly quads with
   two coastline edges) collapse the bilinear Jacobian and set the time
   step: 0.0096 s at Frøya, against 0.11 s with 0.3 and `smooth=5` Laplace
   passes.
5. Boundary curves on the box are the physical group "open", the rest
   "coast"; the water is "water".

Coordinates are the example's local projection (`io::LocalProjection` centred
on the box), in metres.

Usage:
    uv run scripts/gmsh_coastline_mesh.py [dem=data/froya_topobathy.tif] \\
        [bbox=8.0,63.6,9.2,64.0] [coast=200] [far=1500] [dist=6000] \\
        [min_land=2·coast] [min_water=2·coast] [min_island=40000] [farm=lon,lat] [farm_size=50] \\
        [farm_radius=1000] [rivers=data/froya_rivers.nc] [river_size=50] [river_radius=1000] \\
        [river_min_water=2·river_size] [recombine_quality=0.3] [smooth=5] [out=data/froya_coast.msh]
"""

import math
import sys

import gmsh
import numpy as np
import rasterio
import rasterio.features
import shapely
from scipy import ndimage
from shapely.geometry import LineString, Polygon, box, shape
from shapely.ops import unary_union


def local_projection(lat0, lon0):
    """`io::LocalProjection` (WGS84 radii of curvature at the centre)."""
    a, e2 = 6378137.0, 6.69437999014e-3
    s = math.sin(math.radians(lat0))
    per_lat = a * (1 - e2) / (1 - e2 * s * s) ** 1.5 * math.pi / 180
    per_lon = a / math.sqrt(1 - e2 * s * s) * math.cos(math.radians(lat0)) * math.pi / 180
    return lambda lon, lat: ((lon - lon0) * per_lon, (lat - lat0) * per_lat)


def land_polygons(dem, bbox, project):
    """Land of the elevation model inside `bbox` as polygons in metres, with
    unconnected water filled."""
    with rasterio.open(dem) as src:
        window = rasterio.windows.from_bounds(*bbox, transform=src.transform)
        z = src.read(1, window=window, boundless=True, fill_value=10.0)
        transform = src.window_transform(window)
    land = z > 0.0
    # Water not connected to the box edge is land
    labels, _ = ndimage.label(~land)
    edge = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    land |= ~np.isin(labels, edge[edge > 0])
    polygons = [
        shape(geom)
        for geom, value in rasterio.features.shapes(land.astype(np.uint8), transform=transform)
        if value == 1
    ]
    to_xy = lambda coords: np.column_stack(project(coords[:, 0], coords[:, 1]))
    return unary_union([shapely.transform(p, to_xy) for p in polygons])


def resample(line, spacing):
    """Points along `line` about `spacing` apart, keeping both ends.
    `spacing` is a number, or a `Spacing` of the position: the points are
    then equally spaced in `∫ ds / spacing`."""
    if not isinstance(spacing, Spacing):
        n = max(1, round(line.length / spacing))
        return [line.interpolate(i / n, normalized=True).coords[0] for i in range(n + 1)]
    # The spacing along the line, sampled at a quarter of its finest
    m = max(2, int(np.ceil(line.length / (0.25 * spacing.finest))) + 1)
    s = np.linspace(0.0, line.length, m)
    h = spacing(shapely.get_coordinates(shapely.line_interpolate_point(line, s)))
    count = np.concatenate([[0.0], np.cumsum(0.5 * (1 / h[1:] + 1 / h[:-1]) * np.diff(s))])
    n = max(1, round(count[-1]))
    at = np.interp(np.linspace(0.0, count[-1], n + 1), count, s)
    at[0], at[-1] = 0.0, line.length
    return [tuple(c) for c in shapely.get_coordinates(shapely.line_interpolate_point(line, at))]


class Spacing:
    """A coastline spacing varying with the position: `base` (m), or less
    near the points `centres`, `size(r)` at `r` metres from one."""

    def __init__(self, base, centres, size, finest):
        self.base, self.centres, self.size, self.finest = base, np.asarray(centres), size, finest

    def __call__(self, xy):
        r = np.hypot(xy[:, None, 0] - self.centres[None, :, 0], xy[:, None, 1] - self.centres[None, :, 1])
        return np.minimum(self.base, self.size(r.min(axis=1)))


def river_mouths(path, bbox, project):
    """The mouths (x, y) in metres of the river file's rivers inside `bbox`."""
    import netCDF4

    with netCDF4.Dataset(path) as d:
        lon, lat = d.variables["river_lon"][:], d.variables["river_lat"][:]
    inside = [
        (float(x), float(y))
        for x, y in zip(lon, lat)
        if bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]
    ]
    print(f"rivers: {len(inside)} of {len(lon)} mouths of {path} inside the box")
    return [project(x, y) for x, y in inside]


def main(opts):
    bbox = [float(v) for v in opts.get("bbox", "8.0,63.6,9.2,64.0").split(",")]
    coast, far = float(opts.get("coast", 200)), float(opts.get("far", 1500))
    dist = float(opts.get("dist", 6000))
    min_land = float(opts.get("min_land", 2 * coast))
    min_water = float(opts.get("min_water", 2 * coast))
    min_island = float(opts.get("min_island", 40_000))
    out = opts.get("out", "data/froya_coast.msh")
    lat0, lon0 = 0.5 * (bbox[1] + bbox[3]), 0.5 * (bbox[0] + bbox[2])
    project = local_projection(lat0, lon0)
    x0, y0 = project(bbox[0], bbox[1])
    x1, y1 = project(bbox[2], bbox[3])
    domain = box(x0, y0, x1, y1)

    mouths = river_mouths(opts["rivers"], bbox, project) if "rivers" in opts else []
    river_size = float(opts.get("river_size", 50))
    river_radius = float(opts.get("river_radius", 1000))
    river_min_water = float(opts.get("river_min_water", 2 * river_size))
    near_mouths = unary_union([shapely.Point(m).buffer(river_radius) for m in mouths])

    land = land_polygons(opts.get("dem", "data/froya_topobathy.tif"), bbox, project)
    # Land narrower than min_land: removed (erode, then dilate the land)
    r = 0.5 * min_land
    land = land.buffer(-r, quad_segs=2).buffer(r, quad_segs=2)
    # Water narrower than min_water: filled (erode, then dilate the water),
    # which also joins islands that (nearly) touch; near the river mouths
    # only water narrower than river_min_water
    water = domain.difference(land)
    opened = lambda w, width: w.buffer(-0.5 * width, quad_segs=2).buffer(0.5 * width, quad_segs=2)
    sea = opened(water, min_water)
    if mouths:
        sea = sea.union(opened(water, river_min_water).intersection(near_mouths))
    sea = sea.intersection(domain)
    sea = max(getattr(sea, "geoms", [sea]), key=lambda p: p.area)
    sea = Polygon(sea.exterior, [h for h in sea.interiors if Polygon(h).area >= min_island])
    sea = sea.simplify(0.1 * min(coast, river_size if mouths else coast), preserve_topology=True)
    rings = [LineString(sea.exterior.coords)] + [LineString(h.coords) for h in sea.interiors]
    gap = min(
        (rings[i].distance(rings[j]) for i in range(len(rings)) for j in range(i + 1, len(rings))),
        default=math.inf,
    )
    print(
        f"sea {sea.area / 1e6:.0f} km² of {domain.area / 1e6:.0f}; {len(sea.interiors)} islands; "
        f"coastline {sea.length / 1e3:.0f} km; closest two coastlines {gap:.0f} m apart"
    )

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("coast")
    geo = gmsh.model.geo
    eps = 1e-6
    side = lambda p: (
        0 if abs(p[1] - y0) < eps else 1 if abs(p[0] - x1) < eps else 2 if abs(p[1] - y1) < eps
        else 3 if abs(p[0] - x0) < eps else None
    )
    point_tags = {}

    def point(p):
        key = (round(p[0], 6), round(p[1], 6))
        if key not in point_tags:
            point_tags[key] = geo.addPoint(p[0], p[1], 0.0)
        return point_tags[key]

    open_lines, coast_lines = [], []
    # Coastline segments of the element size before subdivision (one
    # element each; features narrower than it were removed above)
    mesher = opts.get("mesher", "subdivide")
    scale = 2.0 if mesher == "subdivide" else 1.0
    # A mouth's size grows as fast as the coast's away from the coast
    river_reach = (far - river_size) * dist / (far - coast)

    def river_ball(r):
        """The size (before scaling) `r` metres from a mouth (Gmsh's Ball
        field below)."""
        grown = river_size + (far - river_size) * (r - river_radius) / river_reach
        return np.where(r <= river_radius, river_size, np.minimum(far, grown))

    if mouths:
        spacing = Spacing(
            scale * coast,
            mouths,
            lambda r: scale * river_ball(r),
            scale * min(coast, river_size),
        )
    else:
        spacing = scale * coast

    def ring_loop(ring):
        """Curve loop of a ring. Edges along one side of the box are "open",
        one straight line per side (meshed by the size field); coastline runs
        are resampled at the size before subdivision. A run shorter than
        half the coastal size takes its neighbour's type."""
        coords = list(ring.coords)[:-1]
        n = len(coords)
        on_side = lambda a, b: side(a) is not None and side(a) == side(b)
        kind = ["open" if on_side(coords[i], coords[(i + 1) % n]) else "coast" for i in range(n)]
        # Runs of one kind, and of one box side for open ones
        key = lambda i: (kind[i], side(coords[i]) if kind[i] == "open" else None)
        start = next((i for i in range(n) if key(i) != key(i - 1)), 0)
        order = [(start + k) % n for k in range(n)]
        runs = []
        for i in order:
            if runs and key(i) == runs[-1][0]:
                runs[-1][1].append(coords[(i + 1) % n])
            else:
                runs.append([key(i), [coords[i], coords[(i + 1) % n]]])
        for k, run in enumerate(runs):
            if len(runs) > 1 and LineString(run[1]).length < 0.5 * coast:
                run[0] = runs[k - 1][0]
        lines = []
        for (run_kind, _), pts in runs:
            if run_kind == "open":
                lines.append(geo.addLine(point(pts[0]), point(pts[-1])))
                open_lines.append(lines[-1])
            else:
                pts = resample(LineString(pts), spacing)
                for a, b in zip(pts, pts[1:]):
                    lines.append(geo.addLine(point(a), point(b)))
                    coast_lines.append(lines[-1])
        return geo.addCurveLoop(lines)

    loops = [ring_loop(sea.exterior)] + [ring_loop(h) for h in sea.interiors]
    surface = geo.addPlaneSurface(loops)
    geo.synchronize()
    gmsh.model.addPhysicalGroup(1, open_lines, name="open")
    gmsh.model.addPhysicalGroup(1, coast_lines, name="coast")
    gmsh.model.addPhysicalGroup(2, [surface], name="water")

    # Meshers:
    # - subdivide (default): recombination + subdivision at twice the size.
    #   The quads of subdivided triangles are smaller than the rest (1 %
    #   under 38 m, the smallest 19 m across, at a 200 m coast); local time
    #   stepping keeps them from setting the cost (`lts=` in the example).
    #   Gmsh's full-quad recombination halves every curve the same way.
    # - qs: quasi-structured quads (Gmsh algorithm 11) at the final size;
    #   more than 20 min on the Frøya coastline
    subdivide = mesher == "subdivide"
    field = gmsh.model.mesh.field
    distance = field.add("Distance")
    field.setNumbers(distance, "CurvesList", coast_lines)
    field.setNumber(distance, "Sampling", 4)
    threshold = field.add("Threshold")
    field.setNumber(threshold, "InField", distance)
    field.setNumber(threshold, "SizeMin", scale * coast)
    field.setNumber(threshold, "SizeMax", scale * far)
    field.setNumber(threshold, "DistMin", 0.5 * coast)
    field.setNumber(threshold, "DistMax", dist)
    fields = [threshold]
    if "farm" in opts:
        fx, fy = project(*[float(v) for v in opts["farm"].split(",")])
        farm = geo.addPoint(fx, fy, 0.0)
        geo.synchronize()
        gmsh.model.mesh.embed(0, [farm], 2, surface)
        farm_distance = field.add("Distance")
        field.setNumbers(farm_distance, "PointsList", [farm])
        farm_threshold = field.add("Threshold")
        field.setNumber(farm_threshold, "InField", farm_distance)
        field.setNumber(farm_threshold, "SizeMin", scale * float(opts.get("farm_size", 50)))
        field.setNumber(farm_threshold, "SizeMax", scale * far)
        field.setNumber(farm_threshold, "DistMin", float(opts.get("farm_radius", 1000)))
        field.setNumber(farm_threshold, "DistMax", dist)
        fields.append(farm_threshold)
    for mx, my in mouths:
        # river_ball: the river's size within river_radius, then growing
        ball = field.add("Ball")
        field.setNumber(ball, "XCenter", mx)
        field.setNumber(ball, "YCenter", my)
        field.setNumber(ball, "Radius", river_radius)
        field.setNumber(ball, "Thickness", river_reach)
        field.setNumber(ball, "VIn", scale * river_size)
        field.setNumber(ball, "VOut", scale * far)
        fields.append(ball)
    minimum = field.add("Min")
    field.setNumbers(minimum, "FieldsList", fields)
    field.setAsBackgroundMesh(minimum)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    if mesher == "qs":
        gmsh.option.setNumber("Mesh.Algorithm", 11)
    else:
        gmsh.option.setNumber("Mesh.Algorithm", 8)
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.Smoothing", 5)
        # Blossom (1): subdivision turns any triangle left into quads
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)
        gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
        # Pairs worse than this stay triangles (and become three good quads
        # on subdivision); see the module docs
        gmsh.option.setNumber(
            "Mesh.RecombineMinimumQuality", float(opts.get("recombine_quality", 0.3))
        )
    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    gmsh.model.mesh.generate(2)
    smoothing = int(opts.get("smooth", 5 if subdivide else 0))
    if smoothing:
        # Evens out the short edges subdivided triangles leave next to the
        # coastline (1st percentile 38 → 51 m at Frøya, smallest time step
        # 0.091 → 0.112 s), but can fold elements: checked below
        gmsh.model.mesh.optimize("Laplace2D", niter=smoothing)

    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.option.setNumber("Mesh.Binary", 1)
    gmsh.write(out)
    types, tags, _ = gmsh.model.mesh.getElements(2)
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    xyz = dict(zip(node_tags, coords.reshape(-1, 3)))
    _, _, quad_nodes = gmsh.model.mesh.getElements(2)
    quads = np.array([xyz[n][:2] for n in quad_nodes[0]]).reshape(-1, 4, 2)
    edges = np.linalg.norm(quads - np.roll(quads, 1, axis=1), axis=2).min(axis=1)
    # Every corner of every quad must turn the same way as the mesh's
    # majority: convex, and not folded over a neighbour
    a, b = np.roll(quads, -1, axis=1) - quads, np.roll(quads, 1, axis=1) - quads
    turn = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    sign = 1.0 if (turn > 0).all(axis=1).sum() >= (turn < 0).all(axis=1).sum() else -1.0
    bad = ~(sign * turn > 0).all(axis=1)
    if bad.any():
        centre = quads[np.argmax(bad)].mean(axis=0)
        sys.exit(f"{bad.sum()} non-convex quads, e.g. at ({centre[0]:.0f}, {centre[1]:.0f})")
    # A corner near 180° collapses the bilinear map's Jacobian there and sets
    # the time step, whatever the element's size
    cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    angles = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    print(
        f"{out}: {sum(len(t) for t in tags)} surface elements (types {list(types)}); shortest edge "
        f"min {edges.min():.0f} m, 1st percentile {np.percentile(edges, 1):.0f} m, median {np.median(edges):.0f} m; "
        f"largest corner {angles.max():.0f}°, {(angles > 160).sum()} corners over 160°"
    )
    gmsh.finalize()


if __name__ == "__main__":
    main(dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a))

# Gmsh meshes

The 2D solver takes meshes of first-order, straight-sided quadrilaterals.
`read_gmsh_mesh` loads them from Gmsh MSH 4.1 files (ASCII or binary; Gmsh's
default format) and from MSH 2.2 ASCII files. This page covers the Gmsh settings
that produce such meshes and how the physical groups become boundary conditions.

## Coordinates

Mesh in metres, in a projected system (UTM zone 32/33 N for the Norwegian coast;
see `io::UtmProjection`). The reader keeps x and y and ignores z.

## All-quadrilateral meshes

Recombination alone often leaves a few triangles, and the reader rejects those.
To get only quads, recombine and then subdivide:

```
Mesh.Algorithm = 8;               // Frontal-Delaunay for quads
Mesh.RecombinationAlgorithm = 3;  // Blossom full-quad
Mesh.RecombineAll = 1;
Mesh.SubdivisionAlgorithm = 1;    // split every element into quads: all-quad
Mesh.ElementOrder = 1;
```

Subdivision halves the element size, so set the size fields to twice the target
size.

Also set a minimum quality for recombination:

```
Mesh.RecombineMinimumQuality = 0.3;  // default 0.01
```

With the default, recombination accepts nearly degenerate quads, for example
two triangles along a straight stretch of coastline. After subdivision these
leave quads with a corner of up to 179°. There the bilinear map's Jacobian
nearly vanishes, and the time step collapses with it, whatever the element's
size. With 0.3, the poor pairs stay triangles and are subdivided into three
good quads each. The time step follows `CFL/(2N+1)·4/(λ_r + λ_s)` with
λ_r = c|∇r|, so check the largest corner angle of a new mesh as well as its
shortest edge.

The reader also rejects:
- degenerate or non-convex quads (their bilinear map has J ≤ 0);
- high-order elements;
- periodic and partitioned meshes.

Each error names the Gmsh element and node tags.

## Boundary groups

Put every boundary curve in a named physical curve, and put the water in a
physical surface. Once a model has any physical group, Gmsh saves only the
elements in physical groups, so without the surface the file has no quads.

| Group name (case-insensitive) | `BoundaryTag` |
|---|---|
| `coast`, `coastline`, `land`, `shore`, `shoreline`, `wall`, `walls`, `closed` | `Wall` |
| `open`, `ocean`, `sea`, `open_boundary` | `Open` |
| `tidal`, `tide`, `tides`, `tidal_forcing` | `TidalForcing` |
| `river`, `rivers` | `River` |
| any other name | `Custom(group number)` |

Groups without a name map by their number: 1 is a wall, 2 open, 3 tidal forcing,
4 river, 5 Dirichlet, 6 Neumann, and any other n is `Custom(n)`. A boundary edge
outside every group is a wall. A curve in two groups that map to different tags
is an error.

## Example: a fjord arm with a refined farm site

```
SetFactory("OpenCASCADE");
// Coastline polygon in UTM metres (e.g. from io::coastline), open to the sea
// along the western edge
Point(1) = {500000, 7050000, 0};  Point(2) = {512000, 7050000, 0};
Point(3) = {512000, 7056000, 0};  Point(4) = {500000, 7056000, 0};
Line(1) = {1, 2}; Line(2) = {2, 3}; Line(3) = {3, 4}; Line(4) = {4, 1};
Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

Physical Curve("coast") = {1, 2, 3};
Physical Curve("open") = {4};
Physical Surface("water") = {1};

// 40 m quads (80 m before subdivision) within 500 m of the farm, 800 m offshore
Point(10) = {506000, 7053000, 0};
Field[1] = Distance; Field[1].PointsList = {10};
Field[2] = Threshold; Field[2].InField = 1;
Field[2].SizeMin = 80; Field[2].SizeMax = 1600;
Field[2].DistMin = 500; Field[2].DistMax = 4000;
Background Field = 2;
Mesh.MeshSizeExtendFromBoundary = 0;

Mesh.Algorithm = 8;
Mesh.RecombinationAlgorithm = 3;
Mesh.RecombineAll = 1;
Mesh.RecombineMinimumQuality = 0.3;
Mesh.SubdivisionAlgorithm = 1;
```

Mesh and save it with `gmsh fjord.geo -2 -format msh41 -o fjord.msh`. Then load
it and dispatch the conditions by tag:

```rust
use dgcom::boundary::{CharacteristicOBC, MultiBoundaryCondition2D, Reflective2D};
use dgcom::mesh::read_gmsh_mesh;

let mesh = read_gmsh_mesh(std::path::Path::new("fjord.msh"))?;
let wall = Reflective2D::new();
let sea = CharacteristicOBC::new(tides); // e.g. BoundaryTides from a TidalAtlas
let bc = MultiBoundaryCondition2D::new(&wall).with_open(&sea);
```

The cages themselves do not need to be meshed. A cage is usually smaller than
an element, and `CageDrag2D` weights its drag onto the nodes it covers:

```rust
use dgcom::source::{CageDrag2D, NetCage};

// Two 50 m ring cages with 20 m nets, solidity 0.25 (clean netting)
let cages = [
    NetCage::circular([506000.0, 7053000.0], 25.0, 20.0, 0.25),
    NetCage::circular([506070.0, 7053000.0], 25.0, 20.0, 0.25),
];
let drag = CageDrag2D::new(&mesh, &ops, &cages);
let physics = builder.with_cage_drag(drag).build();
```

## Coastline-fitted meshes from an elevation model

`scripts/gmsh_coastline_mesh.py` meshes the water of a lon/lat box from an
elevation model with land heights and depths (Kartverket's topobathy model,
`scripts/kartverket_topobathy.sh`). `froya_real_data mesh=<file>` runs on the
result, with the bed L2-projected from the same model:

```bash
uv run scripts/gmsh_coastline_mesh.py [coast=200] [far=1500] [dist=6000] \
    [min_land=2·coast] [min_water=2·coast] [min_island=40000] [farm=lon,lat] \
    [farm_size=50] [farm_radius=1000] [out=data/froya_coast.msh]
cargo run --release --example froya_real_data -- mesh=data/froya_coast.msh lts=8
```

- **Coastline.** The 0 m contour of the model. Water not connected to the
  edge of the box becomes land.
- **Unresolvable features.** Land narrower than `min_land` is removed (an
  opening of the land); skerries and thin points stay in the bed as shoals,
  which the wet/dry scheme dries. Water narrower than `min_water` is filled
  (an opening of the water), which also joins islands that nearly touch.
  Both default to twice the coastal size. Coastline segments are that long
  before subdivision, and a narrower feature would be cut by them, or would
  force elements as small as itself.
- **Coastline segments.** They are resampled at the element size before
  subdivision. Sides on the box are one straight line each, tagged `open`,
  and the coastline is `coast`.
- **Mesher.** The default is recombination (minimum quality
  `recombine_quality=0.3`) and subdivision, then `smooth=5` Laplace passes,
  with sizes from `coast` at the shore to `far` beyond `dist`, optionally
  `farm_size` near `farm`. Gmsh's full-quad recombination halves every curve
  the same way. Its quasi-structured quads (`mesher=qs`) took over 20
  minutes on the Frøya coastline.
- **Check.** Every quad must be convex and oriented like the rest (Laplace
  smoothing can fold elements). The script prints the shortest edges and
  the largest corner angle.

**Frøya** (8.0–9.2°E, 63.6–64.0°N, the defaults): 12,490 quads, 33 islands,
479 km of coastline, coastlines at least 278 m apart; the largest corner is
153°. At rest, max |dq/dt| is 1e-11. The quads of subdivided triangles are
smaller than the rest: 1 % under 51 m and the smallest 19 m across, at a
200 m coast. The smallest time step is 0.11 s (P2). With local time
stepping (`lts=8`, which uses 5 levels here), a coarse step of 3.5 s costs
≈ 115 ms at 24 threads: 2 min per simulated hour, against 3.7 min with
global steps.

Before the minimum recombination quality (Gmsh's default 0.01, no
smoothing), the mesh had 12,206 quads and 110 corners over 160°. The worst
reached 178° in 50–110 m of water and set a time step of 0.0096 s. Local time
stepping then needed 8 levels and a 2.3 s coarse step at ≈ 140 ms: 3.6 min
per simulated hour, 1.84× the current mesh.

**Mausund sub-domain** (8.45–8.90°E, 63.78–63.95°N, 22 × 19 km around the
Mausund gauge): for quicker checks. The same coastal sizes give 2,929 quads
(2,799 in water, 9 islands), a quarter of Frøya, at ≈ 25 s per simulated
hour with `lts=8` on 24 threads (a 3-day gauge comparison in ≈ 30 min). A
sub-box needs its own boundary atlas, since the Frøya atlas only covers the
Frøya perimeter (≈ 10 min over OPeNDAP, netCDF build):

```bash
uv run scripts/gmsh_coastline_mesh.py bbox=8.45,63.78,8.90,63.95 out=data/mausund_coast.msh
cargo run --release --example norkyst_boundary_tides -- bbox=8.45,63.78,8.90,63.95 \
    start=2025-06-01 days=30 out=data/mausund_boundary_tides.txt
cargo run --release --no-default-features --features parallel,simd --example froya_real_data -- \
    mesh=data/mausund_coast.msh bbox=8.45,63.78,8.90,63.95 \
    tides=data/mausund_boundary_tides.txt tide_transport=3 lts=8 hours=72 ramp_hours=3
```

Use `tide_transport=3`: the sub-box's sides cross the skerries in 20–45 m of
water, where the bed is on average 0.6 of NorKyst's smoothed one, and
NorKyst's velocity unscaled pushes too little transport through them. At
Mausund (hours 24–72 from 2025-06-15) the centred RMSE against the gauge's
tidal prediction is 8.1 cm without scaling and 4.2 cm with it (M2 0.96× →
1.01× the gauge's), against 2.3 cm for the full Frøya coastline mesh. The
sub-domain is for relative comparisons (a bed or scheme change against a
baseline), not for the absolute skill.

The `bbox=` of the run must be the mesh's: both centre the local projection
on it. For `scripts/norkyst_current_comparison.py` pass the same centre
(`lat0=63.865 lon0=8.675`).

## Test meshes

`scripts/gmsh_fixtures.py` builds the meshes in `tests/data/gmsh/` with the Gmsh
Python API (`uv run scripts/gmsh_fixtures.py`). It meshes a bay with a curved
coastline and an island and saves the result in the three supported formats,
plus a quad-dominant version that the reader must reject.

`scripts/gmsh_farm_mesh.py` builds `tests/data/gmsh/fjord_farm.msh`: the fjord
arm above, in local metres, with ≈ 20 m quads at the farm growing to ≈ 450 m
(`uv run scripts/gmsh_farm_mesh.py`). It is the benchmark for local time
stepping (`MultirateSSPRK3`, `examples/local_time_stepping_farm.rs`). With
`farm=40 out=tests/data/gmsh/fjord_farm_3d.msh` it builds the same fjord with
40 m quads at the farm (2016 elements against 3584), the mesh the viewer's 3D
fjord runs on (`dg-viz --scenario fjord3d`): the 3D model has no local time
stepping, so the farm's quads set its one step.

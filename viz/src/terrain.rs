//! The land and the sea bed around the model domain, from elevation models, so that
//! the domain sits in its map instead of a cut-out block.
//!
//! An elevation model is a north-up longitude/latitude GeoTIFF of bed elevations with
//! land heights (Kartverket's topobathy model from `scripts/kartverket_topobathy.sh`;
//! `--terrain`), placed by the scenario's projection ([`Scenario::projection`]). The
//! first is the base: its exact-zero holes are filled as the solver's runs fill them
//! (`BedRaster::fill_holes`), and it is sampled bilinearly on a regular grid at its
//! resolution (coarser if that would exceed [`MAX_VERTICES`]) over the mesh's box and
//! a margin around it (`--terrain-margin`).
//!
//! Further models are finer patches over it, such as the service's 1 m level around a
//! farm: lidar land, and a flat 0 over the sea it has not surveyed. Where a patch has
//! land (or a surveyed depth) it replaces the coarser models; its exact 0 is the water
//! surface, and takes their depth there, capped at 0. Each patch is its own grid,
//! aligned to the base grid's lines with a whole number of its cells in each of the
//! base's, which are cut out under it; its border vertices follow the base's edges, so
//! the seam has no cracks. Patches should not overlap one another.
//!
//! The model draws its own bed and water ([`crate::surface`]), so the terrain stops at
//! the mesh boundary: cells wholly inside are left out, and the inside corners of the
//! cells across the boundary snap onto it (its nearest point). There the terrain takes
//! the land's height, the elevation model's or the model's bed where that is higher,
//! and the model's bed rises to it in a wall along the boundary ([`Terrain::coast`]):
//! the coast the model has. The wall matters: the coastline mesh's walls stand in deep
//! water (within a cell of the boundary the model's bed is −17 m at the 10th percentile
//! against +16 m land outside, at Frøya), and grid cells sloping across it instead
//! drew a row of teeth around every island.
//!
//! Land the mesh does not resolve is drawn inside it too: where the elevation model
//! stands [`LAND_CLEARANCE`] above both mean sea level and the model's bed, as islets
//! a coarse mesh floods (around the Kattholmen farm the Frøya mesh has water over
//! every skerry). The inside vertices beside it keep the elevation model's height
//! down to [`SHORE_DEPTH`], so the land slopes into the model's water at its real
//! coastline and stops just under it; the model's bed is the sea floor. Sunk under the
//! model's bed (tens of metres deep on a coarse element there), every coast hung a
//! curtain one cell wide; at the elevation model's own depth, a 50 m cell with one
//! land corner hung an icicle under each skerry. That hides the model's water over
//! such land: it is a map of the coast, not of the model's.
//!
//! Land is coloured by height ([`LAND`]), the sea bed on the model bed's depth scale
//! ([`BedScale`]), so the colours run on across the model's open boundaries. The sea
//! outside the model is still and flat at mean sea level, but on the mesh boundary,
//! where it meets the model's water at the surface there (so the two join within a
//! cell at open boundaries instead of a step of the tide's height): a plain tint, as
//! translucent as the model's water and hidden with it, so the model's coloured water
//! stands out in it. L toggles the terrain.
//!
//! In the photo view ([`crate::photo`]) the ground is seen through the still sea
//! over it (its depth below mean sea level in UV 1), and the sea carries the waves.

use std::collections::HashMap;
use std::error::Error;
use std::path::{Path, PathBuf};

use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, PrimitiveTopology, VertexAttributeValues};
use bevy::prelude::*;
use dg_rs::io::{BedRaster, CoordinateProjection, GeoBoundingBox, GeoTiffBathymetry};
use dg_rs::mesh::PointLocator2D;
use dg_rs::types::ElementIndex;
use rayon::prelude::*;

use crate::actions::{Act, Action};
use crate::colormap::{LAND, Lut, SEABED};
use crate::field::{Field, Frame, Nodes, Probe};
use crate::photo::PhotoPart;
use crate::scenario::Scenario;
use crate::solver::H_DRY;
use crate::surface::{BedScale, SurfaceStyle, WaterOpacity};

pub struct TerrainPlugin;

impl Plugin for TerrainPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn.after(crate::surface::spawn))
            .add_systems(
                Update,
                (
                    toggle,
                    sea_style,
                    sea_level.after(crate::field::interpolate),
                ),
            );
    }
}

/// Most vertices of one patch's grid (≈ 2000 × 1000 cells).
const MAX_VERTICES: usize = 2_000_000;
/// Height (m) that land inside the mesh must stand above mean sea level and above the
/// model's bed to be drawn there.
const LAND_CLEARANCE: f64 = 0.5;
/// Depth (m) below mean sea level to which such land's foot is drawn.
const SHORE_DEPTH: f64 = 2.0;
/// Height (m) at the top of the land's colour map.
const LAND_TOP: f32 = 300.0;
/// The still sea outside the model (sRGB).
const SEA: [f32; 3] = [0.10, 0.25, 0.34];

/// A regular grid in mesh coordinates: vertex `j * nx + i` at `(x0 + i dx, y0 + j dy)`
/// (row 0 south).
#[derive(Clone, Copy, Debug)]
struct Grid {
    x0: f64,
    y0: f64,
    dx: f64,
    dy: f64,
    nx: usize,
    ny: usize,
}

impl Grid {
    fn point(&self, g: usize) -> [f64; 2] {
        [
            self.x0 + (g % self.nx) as f64 * self.dx,
            self.y0 + (g / self.nx) as f64 * self.dy,
        ]
    }

    fn len(&self) -> usize {
        self.nx * self.ny
    }
}

/// A point of the mesh boundary: between the boundary nodes `a` and `b`, a fraction
/// `t` of the way from `a`.
#[derive(Clone, Copy, Debug)]
struct OnBoundary {
    a: usize,
    b: usize,
    t: f32,
}

/// A vertex of a patch: where it is (m, mesh coordinates), its height (m) and, snapped
/// onto the mesh boundary, where on it.
type Vertex = ([f64; 2], f64, Option<OnBoundary>);

/// One grid of the terrain, in the world.
struct Patch {
    grid: Grid,
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    heights: Vec<f32>,
    /// Whether the vertex is inside the mesh
    inside: Vec<bool>,
    /// Whether the vertex is drawn: outside the mesh, or land inside it; a cell with a
    /// drawn corner is drawn
    drawn: Vec<bool>,
    /// The vertices snapped onto the mesh boundary, and where
    snapped: Vec<(usize, OnBoundary)>,
    /// Cells left to finer patches, `[i0, i1, j0, j1]`: cells `i0 ≤ i < i1`,
    /// `j0 ≤ j < j1`
    holes: Vec<[usize; 4]>,
}

impl Patch {
    /// The corners of every cell that `keep` keeps by its corners, outside the holes.
    fn cells(&self, keep: impl Fn([usize; 4]) -> bool) -> Vec<u32> {
        let nx = self.grid.nx;
        let mut indices = Vec::new();
        for j in 0..self.grid.ny - 1 {
            for i in 0..nx - 1 {
                if self
                    .holes
                    .iter()
                    .any(|&[i0, i1, j0, j1]| (i0..i1).contains(&i) && (j0..j1).contains(&j))
                {
                    continue;
                }
                // South-west, south-east, north-east, north-west
                let v = [
                    j * nx + i,
                    j * nx + i + 1,
                    (j + 1) * nx + i + 1,
                    (j + 1) * nx + i,
                ];
                if keep(v) {
                    let [a, b, c, d] = v.map(|v| v as u32);
                    // The materials draw both sides, so the winding does not matter
                    indices.extend([a, b, c, a, c, d]);
                }
            }
        }
        indices
    }
}

/// An elevation model over the box (m, mesh coordinates) it covers.
struct Layer {
    raster: BedRaster,
    lo: [f64; 2],
    hi: [f64; 2],
}

impl Layer {
    /// The model `path` over the box `lo`–`hi` (m), cut to its coverage; `base` fills
    /// its holes.
    fn load<P: CoordinateProjection>(
        path: &Path,
        projection: &P,
        lo: [f64; 2],
        hi: [f64; 2],
        base: bool,
    ) -> Result<Self, Box<dyn Error>> {
        let dem = GeoTiffBathymetry::load(path)?;
        let geo = [
            [lo[0], lo[1]],
            [hi[0], lo[1]],
            [lo[0], hi[1]],
            [hi[0], hi[1]],
        ]
        .map(|[x, y]| projection.xy_to_geo(x, y));
        let full = *dem.bbox();
        let min = |v: &mut dyn Iterator<Item = f64>| v.fold(f64::INFINITY, f64::min);
        let max = |v: &mut dyn Iterator<Item = f64>| v.fold(f64::NEG_INFINITY, f64::max);
        let window = GeoBoundingBox::new(
            min(&mut geo.iter().map(|c| c.1)).max(full.min_lon),
            min(&mut geo.iter().map(|c| c.0)).max(full.min_lat),
            max(&mut geo.iter().map(|c| c.1)).min(full.max_lon),
            max(&mut geo.iter().map(|c| c.0)).min(full.max_lat),
        );
        if window.min_lon >= window.max_lon || window.min_lat >= window.max_lat {
            return Err(format!("{} does not cover the mesh", path.display()).into());
        }
        let mut raster = BedRaster::elevation_model(&dem, &window)?;
        if base {
            raster.fill_holes(|b| b == 0.0);
        }
        // The window in mesh coordinates, inside the box
        let xy = [
            [window.min_lat, window.min_lon],
            [window.min_lat, window.max_lon],
            [window.max_lat, window.min_lon],
            [window.max_lat, window.max_lon],
        ]
        .map(|[lat, lon]| projection.geo_to_xy(lat, lon));
        Ok(Self {
            raster,
            lo: [
                lo[0].max(min(&mut xy.iter().map(|p| p.0))),
                lo[1].max(min(&mut xy.iter().map(|p| p.1))),
            ],
            hi: [
                hi[0].min(max(&mut xy.iter().map(|p| p.0))),
                hi[1].min(max(&mut xy.iter().map(|p| p.1))),
            ],
        })
    }

    fn covers(&self, [x, y]: [f64; 2]) -> bool {
        x >= self.lo[0] && x <= self.hi[0] && y >= self.lo[1] && y <= self.hi[1]
    }
}

/// The elevation models on the terrain's grids, in the world.
#[derive(Resource)]
pub struct Terrain {
    /// The base grid first, then a patch per finer model
    patches: Vec<Patch>,
    /// The land's height (m) at every DG node on the mesh boundary
    coast: HashMap<usize, f32>,
    /// World (X, Z) corners of the base grid, a cell inside its edge
    extent: [Vec2; 2],
    /// Whether it is shown (L)
    pub on: bool,
}

impl Terrain {
    /// The elevation models `paths` (the base first, then finer patches) around the
    /// scenario's mesh, `margin` times the mesh's extent beyond it on every side (cut
    /// to the base's coverage), placed by `projection`.
    pub fn build<P: CoordinateProjection + Sync>(
        paths: &[PathBuf],
        projection: &P,
        margin: f64,
        scenario: &Scenario,
        nodes: &Nodes,
        locator: &PointLocator2D,
        frame: &Frame,
    ) -> Result<Self, Box<dyn Error>> {
        let (lo, hi) = scenario.mesh.vertices.iter().fold(
            ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]),
            |(lo, hi), v| {
                (
                    [lo[0].min(v[0]), lo[1].min(v[1])],
                    [hi[0].max(v[0]), hi[1].max(v[1])],
                )
            },
        );
        let pad = margin * (hi[0] - lo[0]).max(hi[1] - lo[1]);
        let (lo, hi) = ([lo[0] - pad, lo[1] - pad], [hi[0] + pad, hi[1] + pad]);
        let Some((base_path, patch_paths)) = paths.split_first() else {
            return Err("no elevation model".into());
        };
        let base = Layer::load(base_path, projection, lo, hi, true)?;
        let mut fine = Vec::new();
        for path in patch_paths {
            match Layer::load(path, projection, base.lo, base.hi, false) {
                Ok(layer) => fine.push(layer),
                Err(e) => eprintln!("no terrain patch from {}: {e}", path.display()),
            }
        }

        // The finest model at a point; a patch's exact 0 is the water surface over
        // sea it has not surveyed, which takes the coarser depth, at most 0
        let elevation = |[x, y]: [f64; 2]| {
            let (lat, lon) = projection.xy_to_geo(x, y);
            let mut z = base.raster.elevation(lat, lon);
            for layer in fine.iter().filter(|l| l.covers([x, y])) {
                let f = layer.raster.elevation(lat, lon);
                z = if f == 0.0 { z.min(0.0) } else { f };
            }
            z
        };

        // The base grid, at the base's resolution
        let (x0, y0, x1, y1) = (base.lo[0], base.lo[1], base.hi[0], base.hi[1]);
        let spacing = base
            .raster
            .pixel_size()
            .max(((x1 - x0) * (y1 - y0) / MAX_VERTICES as f64).sqrt());
        let nx = ((x1 - x0) / spacing).round() as usize + 1;
        let ny = ((y1 - y0) / spacing).round() as usize + 1;
        let base_grid = Grid {
            x0,
            y0,
            dx: (x1 - x0) / (nx - 1) as f64,
            dy: (y1 - y0) / (ny - 1) as f64,
            nx,
            ny,
        };
        // Each patch over the base cells it covers wholly, m × m cells in each
        let patch_grids: Vec<(Grid, [usize; 4], usize)> = fine
            .iter()
            .filter_map(|layer| {
                let g = &base_grid;
                let i0 = ((layer.lo[0] - g.x0) / g.dx - 1e-9).ceil().max(0.0) as usize;
                let j0 = ((layer.lo[1] - g.y0) / g.dy - 1e-9).ceil().max(0.0) as usize;
                let i1 = (((layer.hi[0] - g.x0) / g.dx + 1e-9).floor() as usize).min(g.nx - 1);
                let j1 = (((layer.hi[1] - g.y0) / g.dy + 1e-9).floor() as usize).min(g.ny - 1);
                if i1 <= i0 || j1 <= j0 {
                    eprintln!("a terrain patch covers no whole cell of the base: not drawn");
                    return None;
                }
                let size = |m: usize| ((i1 - i0) * m + 1) * ((j1 - j0) * m + 1);
                let mut m = (g.dx.min(g.dy) / layer.raster.pixel_size())
                    .round()
                    .max(1.0) as usize;
                while m > 1 && size(m) > MAX_VERTICES {
                    m -= 1;
                }
                let grid = Grid {
                    x0: g.x0 + i0 as f64 * g.dx,
                    y0: g.y0 + j0 as f64 * g.dy,
                    dx: g.dx / m as f64,
                    dy: g.dy / m as f64,
                    nx: (i1 - i0) * m + 1,
                    ny: (j1 - j0) * m + 1,
                };
                Some((grid, [i0, i1, j0, j1], m))
            })
            .collect();

        // The mesh boundary: segments between consecutive nodes of every boundary face
        let (mesh, ops) = (&scenario.mesh, &scenario.ops);
        let node_xy = |g: usize| {
            let (k, i) = (g / ops.n_nodes, g % ops.n_nodes);
            mesh.reference_to_physical(ElementIndex::new(k), ops.nodes_r[i], ops.nodes_s[i])
        };
        let segments: Vec<[usize; 2]> = nodes
            .boundary
            .iter()
            .flat_map(|&(k, face)| {
                let ends: Vec<usize> = ops.face_nodes[face]
                    .iter()
                    .map(|&i| k * ops.n_nodes + i)
                    .collect();
                ends.windows(2).map(|w| [w[0], w[1]]).collect::<Vec<_>>()
            })
            .collect();
        // The land at the boundary: the elevation model, or the model's bed where that
        // is higher (the coast wall of `crate::surface` rises to it)
        let top = |p: [f64; 2], bed: f64| elevation(p).max(bed);
        let coast: HashMap<usize, f32> = segments
            .iter()
            .flatten()
            .map(|&g| (g, top(node_xy(g), nodes.bed[g] as f64) as f32))
            .collect();

        // A grid's vertices: outside the mesh, and on land inside it, the elevation
        // model; the inside corners of the cells across the boundary snap onto it, at
        // the land's height there
        let vertices = |grid: &Grid| -> (Vec<Vertex>, Vec<bool>, Vec<bool>) {
            let samples: Vec<(f64, Option<f64>)> = (0..grid.len())
                .into_par_iter()
                .map(|g| {
                    let p = grid.point(g);
                    let bed = Probe::at(locator, scenario, p).map(|pr| pr.eval(&nodes.bed));
                    (elevation(p), bed.map(f64::from))
                })
                .collect();
            let inside: Vec<bool> = samples.iter().map(|s| s.1.is_some()).collect();
            let land: Vec<bool> = samples
                .iter()
                .map(|&(z, bed)| bed.is_some_and(|b| z > LAND_CLEARANCE && z > b + LAND_CLEARANCE))
                .collect();
            let (nx, ny) = (grid.nx as isize, grid.ny as isize);
            let across = |g: usize| {
                let (i, j) = ((g % grid.nx) as isize, (g / grid.nx) as isize);
                (-1..=1).any(|dj| {
                    (-1..=1).any(|di| {
                        let (i, j) = (i + di, j + dj);
                        (0..nx).contains(&i)
                            && (0..ny).contains(&j)
                            && !inside[(j * nx + i) as usize]
                    })
                })
            };
            let points = (0..grid.len())
                .into_par_iter()
                .map(|g| {
                    let p = grid.point(g);
                    let (z, bed) = samples[g];
                    if bed.is_none() || land[g] {
                        return (p, z, None);
                    }
                    if !across(g) {
                        return (p, z.max(-SHORE_DEPTH), None);
                    }
                    // The nearest point of the boundary, and the model's bed there
                    let mut best = (f64::INFINITY, p, 0.0, OnBoundary { a: 0, b: 0, t: 0.0 });
                    for &[a, b] in &segments {
                        let (pa, pb) = (node_xy(a), node_xy(b));
                        let ab = [pb[0] - pa[0], pb[1] - pa[1]];
                        let t = (((p[0] - pa[0]) * ab[0] + (p[1] - pa[1]) * ab[1])
                            / (ab[0] * ab[0] + ab[1] * ab[1]).max(1e-12))
                        .clamp(0.0, 1.0);
                        let q = [pa[0] + t * ab[0], pa[1] + t * ab[1]];
                        let d = (q[0] - p[0]).powi(2) + (q[1] - p[1]).powi(2);
                        if d < best.0 {
                            let bed =
                                nodes.bed[a] as f64 + t * (nodes.bed[b] - nodes.bed[a]) as f64;
                            best = (d, q, bed, OnBoundary { a, b, t: t as f32 });
                        }
                    }
                    (best.1, top(best.1, best.2), Some(best.3))
                })
                .collect();
            let drawn = inside.iter().zip(&land).map(|(&i, &l)| !i || l).collect();
            (points, inside, drawn)
        };

        let (base_points, base_inside, base_drawn) = vertices(&base_grid);
        let mut patches = vec![finish(
            base_grid,
            base_points.clone(),
            base_inside.clone(),
            base_drawn,
            patch_grids.iter().map(|p| p.1).collect(),
            frame,
        )];
        for &(grid, [i0, _, j0, _], m) in &patch_grids {
            let (mut points, inside, drawn) = vertices(&grid);
            // The border follows the base's edges, linear between its vertices
            let base_at = |i: usize, j: usize| {
                let g = j * base_grid.nx + i;
                (!base_inside[g]).then_some(base_points[g].1)
            };
            for g in 0..grid.len() {
                let (i, j) = (g % grid.nx, g / grid.nx);
                let on_border = i == 0 || j == 0 || i == grid.nx - 1 || j == grid.ny - 1;
                if !on_border || inside[g] {
                    continue;
                }
                // Along x (bottom and top rows) or along y (left and right columns)
                let (along, fixed, by_x) = if j == 0 || j == grid.ny - 1 {
                    (i, j / m, true)
                } else {
                    (j, i / m, false)
                };
                let (q, r) = (along / m, along % m);
                let corner = |q: usize| {
                    if by_x {
                        base_at(i0 + q, j0 + fixed)
                    } else {
                        base_at(i0 + fixed, j0 + q)
                    }
                };
                let blended = if r == 0 {
                    corner(q)
                } else {
                    corner(q).zip(corner(q + 1)).map(|(a, b)| {
                        let t = r as f64 / m as f64;
                        a + t * (b - a)
                    })
                };
                if let Some(z) = blended {
                    points[g].1 = z;
                }
            }
            patches.push(finish(grid, points, inside, drawn, Vec::new(), frame));
        }
        // A cell in from the edge, so that a mesh boundary on the base's edge is
        // not covered
        let inset = [base_grid.dx, base_grid.dy];
        let corner = |x: f64, y: f64| {
            let w = frame.world([x, y], 0.0);
            Vec2::new(w.x, w.z)
        };
        let (a, b) = (
            corner(x0 + inset[0], y0 + inset[1]),
            corner(x1 - inset[0], y1 - inset[1]),
        );
        let extent = [a.min(b), a.max(b)];
        Ok(Self {
            patches,
            coast,
            extent,
            on: true,
        })
    }

    /// Whether the terrain covers the world point (X, Z) with a cell to spare.
    pub fn covers(&self, xz: Vec2) -> bool {
        xz.cmpge(self.extent[0]).all() && xz.cmple(self.extent[1]).all()
    }

    /// Each grid's size and spacing (m), the base first.
    pub fn grids(&self) -> Vec<([usize; 2], f64)> {
        self.patches
            .iter()
            .map(|p| ([p.grid.nx, p.grid.ny], p.grid.dx.max(p.grid.dy)))
            .collect()
    }

    /// The land's height (m) at node `g` on the mesh boundary.
    pub fn coast(&self, g: usize) -> Option<f32> {
        self.coast.get(&g).copied()
    }
}

/// A patch of the vertices `points` on `grid`, in the world.
fn finish(
    grid: Grid,
    points: Vec<Vertex>,
    inside: Vec<bool>,
    drawn: Vec<bool>,
    holes: Vec<[usize; 4]>,
    frame: &Frame,
) -> Patch {
    let (nx, ny) = (grid.nx, grid.ny);
    let height = |i: usize, j: usize| points[j * nx + i].1;
    let mut positions = Vec::with_capacity(grid.len());
    let mut normals = Vec::with_capacity(grid.len());
    for j in 0..ny {
        for i in 0..nx {
            let (p, _, _) = points[j * nx + i];
            positions.push(frame.world(p, height(i, j) as f32).to_array());
            // Central differences, one-sided at the edges
            let (il, ir) = (i.saturating_sub(1), (i + 1).min(nx - 1));
            let (jl, jr) = (j.saturating_sub(1), (j + 1).min(ny - 1));
            let zx = (height(ir, j) - height(il, j)) / ((ir - il) as f64 * grid.dx);
            let zy = (height(i, jr) - height(i, jl)) / ((jr - jl) as f64 * grid.dy);
            let vz = frame.vz as f64;
            // World X = x, Z = −y (as `Nodes::normals`)
            normals.push(
                Vec3::new((-vz * zx) as f32, 1.0, (vz * zy) as f32)
                    .normalize()
                    .to_array(),
            );
        }
    }
    Patch {
        grid,
        positions,
        normals,
        heights: points.iter().map(|p| p.1 as f32).collect(),
        snapped: points
            .iter()
            .enumerate()
            .filter_map(|(g, p)| p.2.map(|on| (g, on)))
            .collect(),
        inside,
        drawn,
        holes,
    }
}

/// Colours of the ground by height: land by [`LAND`], the sea bed on the model bed's
/// scale.
pub struct GroundColours {
    land: Lut,
    seabed: Lut,
    scale: BedScale,
}

impl GroundColours {
    pub fn new(scale: BedScale) -> Self {
        Self {
            land: Lut::new(LAND),
            seabed: Lut::new(SEABED),
            scale,
        }
    }

    pub fn at(&self, z: f32) -> [f32; 4] {
        if z >= 0.0 {
            self.land.at(z / LAND_TOP)
        } else {
            let BedScale { shallow, deep } = self.scale;
            self.seabed.at((-z - shallow) / (deep - shallow).max(1e-6))
        }
    }
}

/// The terrain's entities (L toggles them).
#[derive(Component)]
struct TerrainPart;

/// The sea outside the model, and its vertices on the mesh boundary.
#[derive(Component)]
struct OuterSea {
    material: Handle<StandardMaterial>,
    on_boundary: Vec<(u32, OnBoundary)>,
}

pub(crate) fn spawn(
    mut commands: Commands,
    terrain: Option<Res<Terrain>>,
    scale: Res<BedScale>,
    opacity: Res<WaterOpacity>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let Some(terrain) = terrain else { return };
    let colouring = GroundColours::new(*scale);
    let ground_material = materials.add(StandardMaterial {
        perceptual_roughness: 0.95,
        reflectance: 0.15,
        double_sided: true,
        cull_mode: None,
        ..default()
    });
    let sea_material = materials.add(StandardMaterial {
        base_color: Color::srgba(SEA[0], SEA[1], SEA[2], opacity.0),
        alpha_mode: AlphaMode::Blend,
        perceptual_roughness: 0.3,
        reflectance: 0.35,
        double_sided: true,
        cull_mode: None,
        ..default()
    });
    for patch in &terrain.patches {
        let colours: Vec<[f32; 4]> = patch.heights.iter().map(|&z| colouring.at(z)).collect();
        let ground = patch.cells(|v| v.iter().any(|&g| patch.drawn[g]));
        // The photo view's depth of the still sea over the ground
        let n_vertices = patch.positions.len();
        let depth: Vec<[f32; 2]> = patch
            .heights
            .iter()
            .map(|&z| [(-z).max(0.0), 0.0])
            .collect();
        commands.spawn((
            Name::new("Terrain"),
            TerrainPart,
            PhotoPart::Bed,
            Mesh3d(
                meshes.add(
                    Mesh::new(
                        PrimitiveTopology::TriangleList,
                        RenderAssetUsages::default(),
                    )
                    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, patch.positions.clone())
                    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, patch.normals.clone())
                    .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, colours)
                    // Unused, but Bevy's PBR fragment shader with a second UV set and
                    // no first reads an undeclared `uv` ("cannot find declaration of
                    // uv" on every start, Bevy 0.20.0)
                    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, vec![[0.0f32; 2]; n_vertices])
                    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_1, depth)
                    .with_inserted_indices(Indices::U32(ground)),
                ),
            ),
            MeshMaterial3d(ground_material.clone()),
        ));

        // The sea: cells with water outside the model, at z = 0, on their own vertices
        let wet = |v: [usize; 4]| {
            !v.iter().all(|&g| patch.inside[g]) && v.iter().any(|&g| patch.heights[g] < 0.0)
        };
        let mut index = vec![u32::MAX; patch.positions.len()];
        let mut positions = Vec::new();
        let mut indices = patch.cells(wet);
        for g in &mut indices {
            let v = &mut index[*g as usize];
            if *v == u32::MAX {
                *v = positions.len() as u32;
                let [x, _, z] = patch.positions[*g as usize];
                positions.push([x, 0.0, z]);
            }
            *g = *v;
        }
        if indices.is_empty() {
            continue;
        }
        let on_boundary = patch
            .snapped
            .iter()
            .filter(|(g, _)| index[*g] != u32::MAX)
            .map(|&(g, on)| (index[g], on))
            .collect();
        let normals = vec![[0.0, 1.0, 0.0]; positions.len()];
        commands.spawn((
            Name::new("Sea"),
            TerrainPart,
            PhotoPart::Sea,
            OuterSea {
                material: sea_material.clone(),
                on_boundary,
            },
            Mesh3d(
                meshes.add(
                    Mesh::new(
                        PrimitiveTopology::TriangleList,
                        RenderAssetUsages::default(),
                    )
                    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
                    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
                    .with_inserted_indices(Indices::U32(indices)),
                ),
            ),
            MeshMaterial3d(sea_material.clone()),
        ));
    }
}

fn toggle(
    mut acts: MessageReader<Act>,
    terrain: Option<ResMut<Terrain>>,
    mut parts: Query<(&mut Visibility, Has<OuterSea>), With<TerrainPart>>,
    style: Res<SurfaceStyle>,
) {
    let Some(mut terrain) = terrain else { return };
    let toggles = acts.read().filter(|a| a.0 == Action::Land).count();
    if toggles % 2 == 1 {
        terrain.on = !terrain.on;
    } else if !style.is_changed() {
        return;
    }
    let on = terrain.on;
    for (mut visibility, sea) in &mut parts {
        // The sea goes with the model's water
        let visible = on && !(sea && *style == SurfaceStyle::Hidden);
        *visibility = if visible {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
    }
}

/// The sea outside the model as translucent as the model's water, or opaque with it.
fn sea_style(
    style: Res<SurfaceStyle>,
    opacity: Res<WaterOpacity>,
    seas: Query<&OuterSea>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    if !(style.is_changed() || opacity.is_changed()) {
        return;
    }
    // The patches share one material
    if let Some(sea) = seas.iter().next()
        && let Some(mut material) = materials.get_mut(&sea.material)
    {
        let (alpha, mode) = match *style {
            SurfaceStyle::Opaque => (1.0, AlphaMode::Opaque),
            _ => (opacity.0, AlphaMode::Blend),
        };
        material.base_color.set_alpha(alpha);
        material.alpha_mode = mode;
    }
}

/// The sea's vertices on the mesh boundary at the model's surface there (0 where the
/// model is dry).
fn sea_level(
    field: Res<Field>,
    nodes: Res<Nodes>,
    frame: Res<Frame>,
    seas: Query<(&OuterSea, &Mesh3d)>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    if field.t.is_none() || !field.is_changed() {
        return;
    }
    let level = |g: usize| {
        let eta = field.eta[g];
        if eta - nodes.bed[g] > H_DRY { eta } else { 0.0 }
    };
    for (sea, mesh) in &seas {
        let Some(mut mesh) = meshes.get_mut(&mesh.0) else {
            continue;
        };
        if let Some(VertexAttributeValues::Float32x3(positions)) =
            mesh.attribute_mut(Mesh::ATTRIBUTE_POSITION)
        {
            for &(v, OnBoundary { a, b, t }) in &sea.on_boundary {
                let z = level(a) + t * (level(b) - level(a));
                positions[v as usize][1] = z * frame.vz;
            }
        }
    }
}

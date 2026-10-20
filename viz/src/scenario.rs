//! What is simulated: the mesh, the bed, the cages and the forcing.
//!
//! - [`Scenario::fjord_farm`] is the fjord arm of `examples/local_time_stepping_farm.rs`
//!   (`docs/gmsh-meshes.md`): 12 × 6 km, open to the sea in the west, quads refined
//!   from ≈ 450 m to ≈ 20 m at a fish farm in the middle. The bed deepens from 30 m at
//!   the head to 150 m at the mouth; an M2 tide enters through a characteristic open
//!   boundary, and Coriolis, Manning friction and two net cages act on the flow.
//! - [`Scenario::froya`] is Frøya–Smøla–Hitra as `examples/froya_real_data.rs` runs it
//!   on the coastline-fitted mesh (`mesh=`): the bed from Kartverket's topobathy model,
//!   projected onto the nodes, and NorKyst-800 boundary tides from the tidal atlas
//!   (`BoundaryTides`), on a clock starting 2025-06-15. The data files are those of the
//!   example (untracked, in `data/`; see `notes/inputs-forcing.md` and
//!   `notes/validation.md` for how to fetch them).
//! - [`Scenario::farm_channel`] is the stratified tidal channel of `examples/farm_3d.rs`,
//!   run by the 3D model ([`ThreeD`]): 6 km long (periodic along it), 600 m wide and
//!   40 m deep, 2 °C warmer over the top 10 m, an M2 current of ≈ 0.5 m/s driven by a
//!   depth-uniform body force, two 50 m cages with 20 m nets across the flow, GLS k-ε
//!   mixing and a log-layer bottom drag.
//! - [`Scenario::fjord_farm_3d`] is the fjord farm run by the 3D model: the same mesh,
//!   bed and M2 tide (through the open boundary, driving the barotropic mode), a
//!   brackish surface layer over the sill water and a summer thermocline
//!   ([`fjord_profile`]), on surface-stretched σ-levels; T and S relax to that
//!   stratification at rest in a band along the open boundary.

use std::error::Error;
use std::f64::consts::PI;
use std::path::Path;
use std::sync::Arc;

use dg_rs::boundary::{BoundaryTides, TidalAtlas};
use dg_rs::io::{
    BedRaster, CoordinateProjection, GeoBoundingBox, GeoTiffBathymetry, LocalProjection,
    SnapshotHeader,
};
use dg_rs::mesh::{Bathymetry2D, BoundaryTag, ElementSlopeBound, Mesh2D, read_gmsh_mesh};
use dg_rs::operators::{DGOperators2D, GeometricFactors2D};
use dg_rs::particles::ClearSkyLight;
use dg_rs::solver::{SWESolution2D, SWEState2D};
use dg_rs::source::{CageFootprint, NetCage};
use dg_rs::time::ModelClock;
use dg_rs::types::ElementIndex;
use dg_rs::vertical::{SigmaGrid, SongHaidvogelStretching, UniformStretching};

pub const G: f64 = 9.81;

/// The tide at the open boundary.
#[derive(Clone, Debug)]
pub enum Tide {
    /// M2 of this amplitude (m), in one phase along the whole boundary
    UniformM2(f64),
    /// Tides of a boundary atlas, varying along the boundary
    Atlas(BoundaryTides),
    /// A depth-uniform M2 body force along mesh x that drives a current of this
    /// amplitude (m/s) without friction: the tidal surface slope of a periodic channel
    BodyForceM2(f64),
}

/// Tidal forcing and friction of a scenario.
#[derive(Clone, Debug)]
pub struct Forcing {
    pub tide: Tide,
    /// Manning coefficient (s/m^(1/3))
    pub manning: f64,
    /// Coriolis parameter (1/s)
    pub coriolis: f64,
    /// Time over which the tide ramps up from rest (s); 0 switches it on at once.
    /// Switched on at once, the tide sends a start-up surge through the fjord
    /// (≈ 0.13 m/s at the farm) whose drag wake outlasts the tidal current there.
    pub ramp: f64,
}

/// The close-up view (F): its camera and the fine grid of current arrows around the
/// point of interest.
#[derive(Clone, Copy, Debug)]
pub struct CloseUp {
    /// Camera distance (world units)
    pub distance: f32,
    /// Camera direction: angle about the vertical from world +Z, and above the
    /// horizontal (rad)
    pub yaw: f32,
    pub pitch: f32,
    /// Spacing of the fine arrows (m)
    pub arrow_spacing: f64,
    /// Radius of the fine arrow grid (m)
    pub arrow_radius: f64,
}

/// The 3D model of a scenario (`Hydrostatic3D` with mode splitting).
#[derive(Clone, Debug)]
pub struct ThreeD {
    pub sigma: Arc<SigmaGrid>,
    /// Temperature (°C) and salinity at rest at height z (m, negative below the
    /// surface), the same in every column
    pub profile: fn(f64) -> (f64, f64),
    /// Open faces: the width (m) of the band along them in which T and S relax to
    /// the stratification at rest, and its time scale on the boundary (s)
    pub open_band: (f64, f64),
    /// The vertical mixing
    pub mixing: Mixing,
    /// Bed roughness z₀ of the log-layer drag (m)
    pub roughness: f64,
    /// Horizontal viscosity: constant background (m²/s) and Smagorinsky coefficient
    pub viscosity: f64,
    pub smagorinsky: f64,
    /// Surface light over the site, for the lice larvae
    pub light: ClearSkyLight,
}

/// The vertical mixing of a 3D model.
#[derive(Clone, Copy, Debug)]
pub enum Mixing {
    /// GLS k-ε, with the bed's roughness
    Gls,
    /// Constant eddy viscosity and diffusivity (m²/s): with no diffusivity a
    /// stratification at rest stays as it is, as the rest-state gates need
    #[cfg_attr(not(test), expect(dead_code, reason = "the gates' mixing"))]
    Constant { viscosity: f64, diffusivity: f64 },
}

pub struct Scenario {
    pub name: String,
    pub mesh: Arc<Mesh2D>,
    pub ops: Arc<DGOperators2D>,
    pub geom: Arc<GeometricFactors2D>,
    pub bathymetry: Arc<Bathymetry2D>,
    pub cages: Vec<NetCage>,
    /// Point of interest, where the close-up view looks: the farm, or a tide gauge
    /// (m, mesh coordinates)
    pub farm: [f64; 2],
    pub close_up: CloseUp,
    /// Named points the number keys frame as the close-up does (m, mesh
    /// coordinates): the stations of a run, the farm sites, `--place`s
    pub places: Vec<(String, [f64; 2])>,
    pub forcing: Forcing,
    /// The 3D model, for a scenario run in 3D
    pub three_d: Option<ThreeD>,
    /// The mesh's periods along x and y (m), for a periodic mesh
    pub periodic: Option<[f64; 2]>,
    /// UTC of model time 0, for a scenario with a date (its tides, its daylight)
    pub clock: Option<ModelClock>,
    /// Mesh coordinates to longitude/latitude, for a scenario on a real coast: places
    /// the elevation model's land around it ([`crate::terrain`])
    pub projection: Option<LocalProjection>,
}

impl Scenario {
    pub fn fjord_farm(mesh_path: &Path, order: usize) -> Result<Self, Box<dyn Error>> {
        const LX: f64 = 12_000.0;
        const FARM: [f64; 2] = [6_000.0, 3_000.0];
        let bed =
            |x: f64, y: f64| -(30.0 + 120.0 * (1.0 - x / LX)) - 10.0 * (PI * y / 6_000.0).sin();

        let mesh = Arc::new(read_gmsh_mesh(mesh_path)?);
        let ops = Arc::new(DGOperators2D::new(order));
        let geom = Arc::new(GeometricFactors2D::compute(&mesh, &ops));
        let bathymetry = Arc::new(Bathymetry2D::from_function(&mesh, &ops, &geom, bed));
        Ok(Self {
            name: "Fjord farm: M2 0.8 m, two 50 m cages".into(),
            cages: vec![
                NetCage::circular([FARM[0] - 40.0, FARM[1]], 25.0, 20.0, 0.25),
                NetCage::circular([FARM[0] + 40.0, FARM[1]], 25.0, 20.0, 0.25),
            ],
            farm: FARM,
            places: Vec::new(),
            close_up: CloseUp {
                distance: 420.0,
                yaw: 0.7,
                pitch: 0.5,
                arrow_spacing: 25.0,
                arrow_radius: 500.0,
            },
            forcing: Forcing {
                tide: Tide::UniformM2(0.8),
                manning: 0.025,
                coriolis: 1.2e-4,
                ramp: 3600.0,
            },
            three_d: None,
            periodic: None,
            clock: None,
            projection: None,
            mesh,
            ops,
            geom,
            bathymetry,
        })
    }

    /// The fjord farm (see [`Self::fjord_farm`]) run by the 3D model on `levels`
    /// surface-stretched σ-levels, stratified as [`fjord_profile`], with the larvae's
    /// light at Mausund on `clock`. The tide enters through the open boundary as in
    /// 2D, for the barotropic mode; the bed's element slopes are bounded for the
    /// pycnocline (`Bathymetry2D::smooth_element_slopes`), which the analytic fjord
    /// bed is within already (r_x0 ≲ 0.05 everywhere), so it is a guard for other
    /// meshes.
    pub fn fjord_farm_3d(
        mesh_path: &Path,
        order: usize,
        levels: usize,
        clock: ModelClock,
    ) -> Result<Self, Box<dyn Error>> {
        /// Mausund, off Frøya (longitude, latitude): where the sun is
        const SITE: [f64; 2] = [8.67, 63.87];
        /// The bottom of the pycnocline of [`fjord_profile`] (m)
        const PYCNOCLINE_BOTTOM: f64 = 20.0;
        /// The 3D model's thin-column depth (`Hydrostatic3D::DEFAULT_MIN_COLUMN_DEPTH`,
        /// m): shallower pairs need no bound
        const THIN_COLUMN: f64 = 0.1;

        let mut scenario = Self::fjord_farm(mesh_path, order)?;
        let mut bed = Bathymetry2D::clone(&scenario.bathymetry);
        let smoothed = bed.smooth_element_slopes(
            &scenario.mesh,
            &scenario.ops,
            &scenario.geom,
            ElementSlopeBound::for_pycnocline(PYCNOCLINE_BOTTOM),
            THIN_COLUMN,
        );
        if smoothed.changed > 0 {
            println!(
                "The bed smoothed for the 3D model: {} elements over the slope bound, \
                 {} nodes moved by up to {:.2} m",
                smoothed.elements_before, smoothed.changed, smoothed.max_change
            );
        }
        scenario.bathymetry = Arc::new(bed);
        scenario.name = "Fjord farm, 3D: M2 0.8 m, brackish layer over sill water, \
                         two 50 m cages"
            .into();
        scenario.three_d = Some(ThreeD {
            // Refined at the surface for the brackish layer, as Frøya's 3D tide
            sigma: Arc::new(SigmaGrid::new(
                levels,
                SongHaidvogelStretching::new(5.0, 0.4, 10.0),
            )),
            profile: fjord_profile,
            // A band a sixth of the fjord's length
            open_band: (2_000.0, 1_800.0),
            mixing: Mixing::Gls,
            roughness: 0.005,
            viscosity: 1.0,
            smagorinsky: 0.2,
            light: ClearSkyLight::new(clock, SITE[0], SITE[1]),
        });
        // Low, from the side
        scenario.close_up = CloseUp {
            distance: 900.0,
            yaw: 2.6,
            pitch: 0.28,
            ..scenario.close_up
        };
        scenario.clock = Some(clock);
        Ok(scenario)
    }

    /// The stratified tidal channel of `examples/farm_3d.rs` (see the module docs) on
    /// `levels` σ-levels, with the larvae's light at Mausund on `clock`.
    pub fn farm_channel(order: usize, levels: usize, clock: ModelClock) -> Self {
        const LX: f64 = 6_000.0;
        const LY: f64 = 600.0;
        const DX: f64 = 60.0;
        const DEPTH: f64 = 40.0;
        const FARM: [f64; 2] = [3_000.0, 300.0];
        /// Mausund, off Frøya (longitude, latitude): where the sun is
        const SITE: [f64; 2] = [8.67, 63.87];

        let (nx, ny) = ((LX / DX).round() as usize, (LY / DX).round() as usize);
        let mesh = Arc::new(Mesh2D::channel_periodic_x(0.0, LX, 0.0, LY, nx, ny));
        let ops = Arc::new(DGOperators2D::new(order));
        let geom = Arc::new(GeometricFactors2D::compute(&mesh, &ops));
        let bathymetry = Arc::new(Bathymetry2D::constant(mesh.n_elements, ops.n_nodes, -DEPTH));
        let cages = [-60.0, 60.0]
            .map(|dy| NetCage::circular([FARM[0], FARM[1] + dy], 25.0, 20.0, 0.25))
            .to_vec();
        Self {
            name: "Farm channel, 3D: M2 0.5 m/s, 2 C thermocline at 10 m, two 50 m cages".into(),
            three_d: Some(ThreeD {
                sigma: Arc::new(SigmaGrid::new(levels, UniformStretching)),
                // 10 °C below, 12 °C above, a smooth step at 10 m
                profile: |z| (11.0 + (0.5 * (z + 10.0)).tanh(), 35.0),
                // Periodic: no open faces
                open_band: (0.0, 0.0),
                mixing: Mixing::Gls,
                roughness: 0.005,
                viscosity: 1.0,
                smagorinsky: 0.2,
                light: ClearSkyLight::new(clock, SITE[0], SITE[1]),
            }),
            cages,
            // Periodic along the channel only
            periodic: Some([LX, f64::INFINITY]),
            clock: Some(clock),
            projection: None,
            farm: FARM,
            places: Vec::new(),
            // Low, from the second cage's side
            close_up: CloseUp {
                distance: 800.0,
                yaw: 2.6,
                pitch: 0.28,
                arrow_spacing: 25.0,
                arrow_radius: 600.0,
            },
            forcing: Forcing {
                tide: Tide::BodyForceM2(0.5),
                manning: 0.0,
                coriolis: 1.2e-4,
                ramp: 3600.0,
            },
            mesh,
            ops,
            geom,
            bathymetry,
        }
    }

    /// Frøya–Smøla–Hitra on the coastline mesh `mesh_path`, with the elevation model
    /// `dem_path` and the boundary tidal atlas `atlas_path` (see the module docs),
    /// run for `duration` seconds (for the tides' nodal corrections).
    pub fn froya(
        mesh_path: &Path,
        dem_path: &Path,
        atlas_path: &Path,
        order: usize,
        duration: f64,
    ) -> Result<Self, Box<dyn Error>> {
        /// Land above this height (m) is never wet: capped, so that shoreline cliffs
        /// do not drive films up the land
        const LAND_ELEVATION: f64 = 5.0;
        /// Open faces farther than this (m) from an atlas point become walls
        const ATLAS_COVERAGE: f64 = 5000.0;
        /// Mausund tide gauge (Kartverket MSU), the close-up view
        const MAUSUND: (f64, f64) = (63.869331, 8.665231);

        let bbox = GeoBoundingBox::new(8.0, 63.6, 9.2, 64.0);
        let (lat0, lon0) = bbox.center();
        let projection = LocalProjection::new(lat0, lon0);

        // Kartverket's topobathy model: land heights and depths in one grid; its
        // missing depths are an exact 0, filled from their surroundings
        let dem = GeoTiffBathymetry::load(dem_path)?;
        let mut raster = BedRaster::elevation_model(&dem, &bbox)?;
        raster.fill_holes(|b| b == 0.0);
        let raster = raster.clamp_land(LAND_ELEVATION);

        // The bed L2-projected onto the nodes; water one node wide is unresolved
        // and becomes shore; the elements without water are dropped
        let grid = read_gmsh_mesh(mesh_path)?;
        let ops = DGOperators2D::new(order);
        let grid_geom = GeometricFactors2D::compute(&grid, &ops);
        let mut grid_bed = Bathymetry2D::project(
            &grid,
            &ops,
            &grid_geom,
            raster.sampler(&projection),
            raster.pixel_size(),
        );
        grid_bed.raise_isolated_wet_nodes(&grid, &ops, &grid_geom, 0.0);
        let has_water = |k: ElementIndex| grid_bed.element(k).iter().any(|&b| b < 0.0);
        let (mut mesh, kept) = grid.retain_elements(has_water, BoundaryTag::Wall);
        let bathymetry = grid_bed.select_elements(&kept);

        // Water the atlas's parent model does not resolve (a fjord arm crossing the
        // domain edge far from any atlas point) has no tide to force: wall it off
        let atlas = TidalAtlas::read(atlas_path)?;
        for e in 0..mesh.edges.len() {
            let edge = &mesh.edges[e];
            if edge.right.is_some() || edge.boundary_tag != Some(BoundaryTag::Open) {
                continue;
            }
            let (k, face) = (ElementIndex::new(edge.left.element), edge.left.face);
            let uncovered = ops.face_nodes[face].iter().any(|&i| {
                let [x, y] = mesh.reference_to_physical(k, ops.nodes_r[i], ops.nodes_s[i]);
                let (lat, lon) = projection.xy_to_geo(x, y);
                atlas
                    .nearest(lon, lat)
                    .is_none_or(|(_, d)| d > ATLAS_COVERAGE)
            });
            if uncovered {
                mesh.edges[e].boundary_tag = Some(BoundaryTag::Wall);
            }
        }

        let clock = ModelClock::parse("2025-06-15T00:00:00Z")?;
        let tides = atlas.boundary_tides(
            &mesh,
            &ops,
            &projection,
            BoundaryTag::Open,
            &clock,
            duration,
            ATLAS_COVERAGE,
        )?;
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let (x, y) = projection.geo_to_xy(MAUSUND.0, MAUSUND.1);
        Ok(Self {
            name: "Frøya–Smøla–Hitra: NorKyst-800 tides from 2025-06-15".into(),
            cages: Vec::new(),
            farm: [x, y],
            places: vec![("Mausund".into(), [x, y])],
            close_up: CloseUp {
                distance: 6_000.0,
                yaw: 0.7,
                pitch: 0.5,
                arrow_spacing: 250.0,
                arrow_radius: 5_000.0,
            },
            forcing: Forcing {
                tide: Tide::Atlas(tides),
                manning: 0.025,
                coriolis: 1.31e-4,
                // A spring tide squeezed into one hour overshoots (TODO P1.4)
                ramp: 3.0 * 3600.0,
            },
            three_d: None,
            periodic: None,
            clock: Some(clock),
            projection: Some(projection),
            mesh: Arc::new(mesh),
            ops: Arc::new(ops),
            geom: Arc::new(geom),
            bathymetry: Arc::new(bathymetry),
        })
    }

    /// The domain of a snapshot file's header, for a replay: its mesh and bed, a
    /// close-up at its point of interest (`point_of_interest=x,y`, else its first
    /// `station=name,x,y`, else the domain's centre) framed for the domain's size,
    /// every station a place the number keys frame,
    /// the cages (`cage=x,y,radius,net_depth,drag_per_length`) and the periods of a
    /// periodic mesh (`periodic=x,y`), and the projection of a mesh on a real coast
    /// (`projection=local,lat,lon`: [`LocalProjection`] about that point). A 3D file's σ-grid makes it a 3D scenario. It runs nothing, so its forcing and 3D physics are none.
    pub fn from_snapshot(header: SnapshotHeader) -> Self {
        let (lo, hi) = header.mesh.vertices.iter().fold(
            ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]),
            |(lo, hi), v| {
                (
                    [lo[0].min(v[0]), lo[1].min(v[1])],
                    [hi[0].max(v[0]), hi[1].max(v[1])],
                )
            },
        );
        let extent = (hi[0] - lo[0]).max(hi[1] - lo[1]);
        let point = |s: &str| -> Option<[f64; 2]> {
            let mut c = s.rsplitn(3, ',').map(|c| c.trim().parse::<f64>());
            let (y, x) = (c.next()?.ok()?, c.next()?.ok()?);
            Some([x, y])
        };
        let farm = header
            .metadata("point_of_interest")
            .and_then(|p| point(&format!("_,{p}")))
            .or_else(|| header.metadata("station").and_then(point))
            .unwrap_or([0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])]);
        let name = header
            .metadata("title")
            .unwrap_or("Snapshot file")
            .to_string();
        // Every station, by its name
        let places = header
            .metadata
            .iter()
            .filter(|(k, _)| k == "station")
            .filter_map(|(_, v)| {
                let at = point(v)?;
                let name = v.rsplitn(3, ',').nth(2)?.trim();
                Some((name.to_string(), at))
            })
            .collect();
        let numbers = |s: &str| -> Option<Vec<f64>> {
            s.split(',').map(|c| c.trim().parse::<f64>().ok()).collect()
        };
        let cages = header
            .metadata
            .iter()
            .filter(|(k, _)| k == "cage")
            .filter_map(|(_, v)| match numbers(v)?.as_slice() {
                &[x, y, radius, depth, drag] => Some(NetCage::new(
                    CageFootprint::Circle {
                        center: [x, y],
                        radius,
                    },
                    depth,
                    drag,
                )),
                _ => None,
            })
            .collect();
        let projection = header.metadata("projection").and_then(parse_projection);
        let periodic = header
            .metadata("periodic")
            .and_then(numbers)
            .and_then(|p| <[f64; 2]>::try_from(p).ok());
        let three_d = header.levels.as_ref().map(|levels| {
            ThreeD {
                sigma: Arc::new(levels.sigma.clone()),
                // A replay runs no model: these are not used
                profile: |_| (0.0, 0.0),
                open_band: (0.0, 0.0),
                mixing: Mixing::Gls,
                roughness: 0.0,
                viscosity: 0.0,
                smagorinsky: 0.0,
                light: ClearSkyLight::new(header.clock.unwrap_or_default(), 0.0, 0.0),
            }
        });

        let ops = DGOperators2D::new(header.order);
        let mesh = header.mesh;
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let mut bathymetry = Bathymetry2D::constant(mesh.n_elements, ops.n_nodes, 0.0);
        bathymetry.data.copy_from_slice(&header.bathymetry);
        bathymetry.compute_gradients(&ops, &geom);
        // As Frøya's for its ≈ 60 km; in 3D low, to look along the water column
        let (yaw, pitch) = if three_d.is_some() {
            (2.6, 0.28)
        } else {
            (0.7, 0.5)
        };
        Self {
            name,
            cages,
            farm,
            places,
            close_up: CloseUp {
                distance: (extent / 10.0) as f32,
                yaw,
                pitch,
                arrow_spacing: extent / 240.0,
                arrow_radius: extent / 12.0,
            },
            forcing: Forcing {
                tide: Tide::UniformM2(0.0),
                manning: 0.0,
                coriolis: 0.0,
                ramp: 0.0,
            },
            three_d,
            periodic,
            clock: header.clock,
            projection,
            mesh: Arc::new(mesh),
            ops: Arc::new(ops),
            geom: Arc::new(geom),
            bathymetry: Arc::new(bathymetry),
        }
    }

    /// Still water at z = 0 over the bed.
    pub fn at_rest(&self) -> SWESolution2D {
        let mut q = SWESolution2D::new(self.mesh.n_elements, self.ops.n_nodes);
        for k in ElementIndex::iter(self.mesh.n_elements) {
            for i in 0..self.ops.n_nodes {
                let h = (-self.bathymetry.get(k, i)).max(0.0);
                q.set_state(k, i, SWEState2D::new(h, 0.0, 0.0));
            }
        }
        q
    }
}

/// The fjord's stratification at rest, temperature (°C) and salinity at height `z`
/// (m, negative below the surface): a brackish surface layer from river runoff, S 25
/// at the surface over S 33 sill water, the step centred at 5 m and 3 m thick; and a
/// summer thermocline, 14 °C over 8 °C, centred at 12 m and 8 m thick. Salinity sets
/// most of the density step (≈ 6 kg/m³ against ≈ 1 for temperature).
pub fn fjord_profile(z: f64) -> (f64, f64) {
    let above = |centre: f64, thickness: f64| 0.5 * (1.0 + ((z + centre) / thickness).tanh());
    (8.0 + 6.0 * above(12.0, 4.0), 33.0 - 8.0 * above(5.0, 1.5))
}

/// A projection as snapshot files write it: `local,lat,lon`, a [`LocalProjection`]
/// about that point (degrees).
pub fn parse_projection(text: &str) -> Option<LocalProjection> {
    match text
        .split(',')
        .map(str::trim)
        .collect::<Vec<_>>()
        .as_slice()
    {
        ["local", lat, lon] => Some(LocalProjection::new(lat.parse().ok()?, lon.parse().ok()?)),
        _ => None,
    }
}

/// `projection` as [`parse_projection`] reads it.
pub fn format_projection(projection: &LocalProjection) -> String {
    format!("local,{},{}", projection.ref_lat(), projection.ref_lon())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Brackish over sill water, warm over cold, and salinity sets most of the
    /// density step.
    #[test]
    fn the_fjord_is_brackish_over_sill_water() {
        use dg_rs::physics::{EquationOfState, LinearEOS};
        let close = |a: f64, b: f64| (a - b).abs() < 0.02;
        let ((t_top, s_top), (t_sill, s_sill)) = (fjord_profile(0.0), fjord_profile(-40.0));
        assert!(
            close(s_top, 25.0) && close(s_sill, 33.0),
            "{s_top}, {s_sill}"
        );
        assert!(
            close(t_top, 14.0) && close(t_sill, 8.0),
            "{t_top}, {t_sill}"
        );
        assert!(close(fjord_profile(-5.0).1, 29.0) && close(fjord_profile(-12.0).0, 11.0));
        let eos = LinearEOS::default();
        let rho = |t, s| eos.compute_density(t, s, 0.0);
        let by_salt = rho(t_sill, s_sill) - rho(t_sill, s_top);
        let by_heat = rho(t_sill, s_sill) - rho(t_top, s_sill);
        assert!(by_salt > 5.0 * by_heat, "{by_salt} against {by_heat} kg/m³");
    }

    #[test]
    fn a_projection_reads_back_as_written() {
        let projection = LocalProjection::new(63.8, 8.675);
        let read = parse_projection(&format_projection(&projection)).unwrap();
        assert_eq!(
            (read.ref_lat(), read.ref_lon()),
            (projection.ref_lat(), projection.ref_lon())
        );
        // The same point maps to the same mesh coordinates
        assert_eq!(
            read.geo_to_xy(63.87, 8.68),
            projection.geo_to_xy(63.87, 8.68)
        );
        assert!(parse_projection("utm,33").is_none());
        assert!(parse_projection("local,63.8").is_none());
    }
}

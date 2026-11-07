//! The discretized domain of a case: the mesh, placed on the Earth or not,
//! its bed, and the land dropped.
//!
//! The order follows `examples/froya_real_data.rs`, so that a case
//! reproduces its runs: build the whole mesh, put the bed on its nodes
//! (a raster's projected or sampled, then its isolated wet nodes raised),
//! smooth the slopes if asked, drop the elements with no node below mean
//! sea level, and wall the open faces a tidal atlas does not cover.

use std::sync::Arc;

use super::schema::{AtlasCorrection, RasterKind, Sampling, SideTag};
use super::{CaseError, CaseFile};
use crate::boundary::{BoundaryCoverage, TidalAtlas};
use crate::io::{
    BedRaster, CoastlineData, CoordinateProjection, GeoBoundingBox, GeoTiffBathymetry,
    LocalProjection,
};
use crate::mesh::{Bathymetry2D, BoundaryTag, Mesh2D, read_gmsh_mesh};
use crate::operators::{DGOperators2D, GeometricFactors2D};
use crate::types::ElementIndex;

/// Mask cells per raster pixel side of a `"bathymetry"` raster's land mask.
const LAND_MASK_REFINEMENT: usize = 4;

/// The discretized domain of a case.
pub struct Domain {
    /// The mesh
    pub mesh: Arc<Mesh2D>,
    /// The reference element's operators
    pub ops: Arc<DGOperators2D>,
    /// The geometric factors
    pub geom: Arc<GeometricFactors2D>,
    /// The bed elevation B (negative under water)
    pub bathymetry: Arc<Bathymetry2D>,
    /// The boundary tags the mesh uses, walls left out
    pub open_tags: Vec<SideTag>,
    /// The projection that places the mesh, if it is placed
    pub projection: Option<LocalProjection>,
    /// The tidal atlas of each `[[boundary]]` that has one, read and
    /// corrected
    pub atlases: Vec<(SideTag, TidalAtlas)>,
    /// What was done to build it, line by line, for a log
    pub notes: Vec<String>,
}

impl Domain {
    /// Positions of the nodes, element-major.
    pub fn node_positions(&self) -> Vec<[f64; 2]> {
        ElementIndex::iter(self.mesh.n_elements)
            .flat_map(|k| {
                (0..self.ops.n_nodes).map(move |i| {
                    self.mesh
                        .reference_to_physical(k, self.ops.nodes_r[i], self.ops.nodes_s[i])
                })
            })
            .collect()
    }
}

impl CaseFile {
    /// The mesh, operators and bed of the case, with its boundary tags
    /// checked against the `[[boundary]]` entries.
    pub fn build_domain(&self) -> Result<Domain, CaseError> {
        let case = &self.case;
        let projection = case.projection();
        let mut notes = Vec::new();
        let grid = self.grid(projection.as_ref())?;
        let ops = DGOperators2D::new(case.mesh.order);
        let grid_geom = GeometricFactors2D::compute(&grid, &ops);

        let b = &case.bathymetry;
        let mut bed = if let Some(depth) = b.depth_m {
            Bathymetry2D::from_function(&grid, &ops, &grid_geom, |_, _| -depth)
        } else if let Some(ramp) = &b.ramp {
            Bathymetry2D::from_function(&grid, &ops, &grid_geom, |x, y| -ramp.depth(x, y))
        } else {
            let projection = projection
                .as_ref()
                .expect("validated: a raster needs a placement");
            let raster = self.raster(&grid, projection, &mut notes)?;
            let sampler = raster.sampler(projection);
            let mut bed = match b.raster.as_ref().expect("validated").sampling {
                Sampling::Projected => {
                    Bathymetry2D::project(&grid, &ops, &grid_geom, sampler, raster.pixel_size())
                }
                Sampling::Point => Bathymetry2D::from_function(&grid, &ops, &grid_geom, sampler),
            };
            let raised = bed.raise_isolated_wet_nodes(&grid, &ops, &grid_geom, 0.0);
            notes.push(format!(
                "{raised} isolated wet nodes raised to the lowest of their neighbours"
            ));
            bed
        };
        if let Some(r_max) = b.smooth_rx0 {
            let s = bed.smooth_rx0(&grid, &ops, &grid_geom, r_max, b.rx0_min_depth_m);
            notes.push(format!(
                "slope factor r_x0 {:.2} → {:.2} (nodes ≥ {} m deep): {} nodes changed by up to {:.1} m",
                s.before, s.after, b.rx0_min_depth_m, s.changed, s.max_change
            ));
        }

        let (mut mesh, bathymetry, geom) = if case.mesh.drop_land {
            let has_water = |k: ElementIndex| bed.element(k).iter().any(|&z| z < 0.0);
            let (mesh, kept) = grid.retain_elements(has_water, BoundaryTag::Wall);
            if kept.is_empty() {
                return Err(CaseError::Invalid(
                    "no element has a node below mean sea level: the domain is all land".into(),
                ));
            }
            notes.push(format!(
                "{} of {} elements kept (a node below mean sea level)",
                kept.len(),
                grid.n_elements
            ));
            let bathymetry = bed.select_elements(&kept);
            let geom = GeometricFactors2D::compute(&mesh, &ops);
            (mesh, bathymetry, geom)
        } else {
            (grid, bed, grid_geom)
        };

        // Tidal atlases: the open faces they do not cover become walls
        let mut atlases = Vec::new();
        for b in &case.boundaries {
            let Some(spec) = &b.atlas else { continue };
            let path = self.resolve(&spec.file);
            let mut atlas = TidalAtlas::read(&path)
                .map_err(|e| CaseError::Io(format!("{}: {e}", path.display())))?;
            if spec.correction == AtlasCorrection::Theory {
                atlas
                    .infer_from_theory()
                    .map_err(|e| CaseError::Invalid(format!("{}: {e}", path.display())))?;
            }
            let projection = projection.as_ref().expect("validated: placed");
            let coverage = BoundaryCoverage::of_atlas(
                &mesh,
                &ops,
                projection,
                b.tag.boundary_tag(),
                &atlas,
                spec.coverage_m,
            );
            let closed = coverage.close(&mut mesh, spec.max_closed).map_err(|e| {
                CaseError::Invalid(format!(
                    "{}: {e}: an atlas for this domain is needed, or a larger \
                     boundary.atlas.max_closed",
                    path.display()
                ))
            })?;
            notes.push(format!(
                "tidal atlas {} ({} constituents{}): {closed} of {} {:?} faces farther than \
                 {:.1} km from it made walls",
                path.display(),
                atlas.names().len(),
                match spec.correction {
                    AtlasCorrection::None => "",
                    AtlasCorrection::Theory => ", corrected from theory",
                },
                coverage.n_faces(),
                b.tag.name(),
                spec.coverage_m / 1000.0
            ));
            atlases.push((b.tag, atlas));
        }

        let open_tags = open_tags(&mesh)?;
        for tag in &open_tags {
            if !case.boundaries.iter().any(|b| b.tag == *tag) {
                return Err(CaseError::Invalid(format!(
                    "the mesh has {tag:?} boundary faces but the case no [[boundary]] with \
                     tag = {:?}",
                    tag.name()
                )));
            }
        }
        for b in &case.boundaries {
            if !open_tags.contains(&b.tag) {
                return Err(CaseError::Invalid(format!(
                    "[[boundary]] tag = {:?}: the mesh has no faces with that tag",
                    b.tag.name()
                )));
            }
        }
        Ok(Domain {
            mesh: Arc::new(mesh),
            ops: Arc::new(ops),
            geom: Arc::new(geom),
            bathymetry: Arc::new(bathymetry),
            open_tags,
            projection,
            atlases,
            notes,
        })
    }

    /// The whole mesh, before land is dropped.
    fn grid(&self, projection: Option<&LocalProjection>) -> Result<Mesh2D, CaseError> {
        let m = &self.case.mesh;
        if let Some(r) = &m.rectangle {
            Ok(Mesh2D::uniform_rectangle_with_sides(
                r.x_m[0],
                r.x_m[1],
                r.y_m[0],
                r.y_m[1],
                r.nx,
                r.ny,
                [r.south, r.east, r.north, r.west].map(SideTag::boundary_tag),
            ))
        } else if let Some(path) = &m.gmsh {
            let path = self.resolve(path);
            read_gmsh_mesh(&path)
                .map_err(|e| CaseError::Io(format!("mesh.gmsh {}: {e}", path.display())))
        } else {
            let g = m.geographic.as_ref().expect("validated");
            let projection = projection.expect("a geographic mesh is placed");
            let [west, south, east, north] = g.bbox;
            let (x0, y0) = projection.geo_to_xy(south, west);
            let (x1, y1) = projection.geo_to_xy(north, east);
            // The sides are open sea wherever they cross water
            Ok(Mesh2D::uniform_rectangle_with_bc(
                x0,
                x1,
                y0,
                y1,
                g.nx,
                g.ny,
                BoundaryTag::Open,
            ))
        }
    }

    /// The bed raster, cropped to the mesh.
    fn raster(
        &self,
        grid: &Mesh2D,
        projection: &LocalProjection,
        notes: &mut Vec<String>,
    ) -> Result<BedRaster, CaseError> {
        let spec = self.case.bathymetry.raster.as_ref().expect("validated");
        let path = self.resolve(&spec.file);
        let io = |e: &dyn std::fmt::Display| CaseError::Io(format!("{}: {e}", path.display()));
        let window = match &self.case.mesh.geographic {
            Some(g) => {
                let [w, s, e, n] = g.bbox;
                GeoBoundingBox::new(w, s, e, n)
            }
            None => mesh_window(grid, projection),
        };
        let tiff = GeoTiffBathymetry::load(&path).map_err(|e| io(&e))?;
        let have = tiff.bbox();
        let overlaps = window.min_lon < have.max_lon
            && window.max_lon > have.min_lon
            && window.min_lat < have.max_lat
            && window.max_lat > have.min_lat;
        if !overlaps {
            return Err(CaseError::Invalid(format!(
                "{}: the raster covers {have:?}, not the mesh's {window:?}",
                path.display()
            )));
        }
        let raster = match spec.kind {
            RasterKind::Topobathy => {
                let mut raster = BedRaster::elevation_model(&tiff, &window).map_err(|e| io(&e))?;
                if spec.fill_zero_holes {
                    let filled = raster.fill_holes(|z| z == 0.0);
                    notes.push(format!(
                        "{filled} zero pixels filled from their surroundings"
                    ));
                }
                raster.clamp_land(spec.land_elevation_m)
            }
            RasterKind::Bathymetry => {
                let coastline = spec
                    .coastline
                    .as_ref()
                    .map(|c| {
                        let c = self.resolve(c);
                        CoastlineData::load(&c, &window)
                            .map_err(|e| CaseError::Io(format!("{}: {e}", c.display())))
                    })
                    .transpose()?;
                BedRaster::from_geotiff(
                    &tiff,
                    coastline.as_ref(),
                    spec.land_elevation_m,
                    &window,
                    LAND_MASK_REFINEMENT,
                )
            }
        };
        let (width, height) = raster.dimensions();
        notes.push(format!(
            "bed raster {}: {width} × {height} pixels, {:.0} m, {:.1} % water",
            path.display(),
            raster.pixel_size(),
            100.0 * raster.water_fraction()
        ));
        Ok(raster)
    }
}

/// The longitude/latitude box around a placed mesh, with a margin of 2 % of
/// its size.
fn mesh_window(mesh: &Mesh2D, projection: &LocalProjection) -> GeoBoundingBox {
    let (mut lat_min, mut lat_max) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut lon_min, mut lon_max) = (f64::INFINITY, f64::NEG_INFINITY);
    for &[x, y] in &mesh.vertices {
        let (lat, lon) = projection.xy_to_geo(x, y);
        lat_min = lat_min.min(lat);
        lat_max = lat_max.max(lat);
        lon_min = lon_min.min(lon);
        lon_max = lon_max.max(lon);
    }
    let (dlat, dlon) = (0.02 * (lat_max - lat_min), 0.02 * (lon_max - lon_min));
    GeoBoundingBox::new(
        lon_min - dlon,
        lat_min - dlat,
        lon_max + dlon,
        lat_max + dlat,
    )
}

/// The tags of the mesh's open boundary faces.
fn open_tags(mesh: &Mesh2D) -> Result<Vec<SideTag>, CaseError> {
    let mut tags = Vec::new();
    for k in ElementIndex::iter(mesh.n_elements) {
        for face in 0..4 {
            let Some(tag) = mesh.boundary_tag(k, face) else {
                continue;
            };
            let side = SideTag::from_boundary_tag(tag).ok_or_else(|| {
                CaseError::Invalid(format!(
                    "the mesh has boundary faces tagged {tag:?}; a case names wall, open, \
                     tidal and river faces only"
                ))
            })?;
            if side != SideTag::Wall && !tags.contains(&side) {
                tags.push(side);
            }
        }
    }
    Ok(tags)
}

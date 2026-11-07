//! The discretized domain of a case: the mesh, placed on the Earth or not,
//! its bed, and the land dropped.
//!
//! The order follows `examples/froya_real_data.rs`, so that a case
//! reproduces its runs: build the whole mesh, put the bed on its nodes
//! (a raster's projected or sampled), lower the land at the walls if asked,
//! raise a raster's isolated wet nodes, smooth the slopes if asked, make the
//! bed a 3D model's if asked (element slopes, cliff shores raised, tidal
//! ponds filled), drop the elements with no node below mean
//! sea level, wall the open faces a tidal atlas does not cover, and nest
//! in a parent model: wall the faces it has no water at, blend the bed to
//! its across the relaxation band, and correct its tide. Last, the stations
//! are placed, in open water as the example's are
//! (`Probe2D::in_open_water`).

use std::sync::Arc;

use super::schema::{
    AtlasCorrection, GaugeCorrection, Nesting, RasterKind, Sampling, SideTag, Slopes3D,
};
use super::{CaseError, CaseFile};
use crate::analysis::HarmonicAnalysis;
use crate::boundary::{
    BoundaryCoverage, GaugeAdjustment, NestingOptions, OceanModelState, ParentDatum, TidalAtlas,
};
use crate::io::{
    BedRaster, CoastlineData, CoordinateProjection, GeoBoundingBox, GeoTiffBathymetry,
    LocalProjection, OceanModelReader, read_tide_gauge_file,
};
use crate::mesh::{Bathymetry2D, BoundaryTag, Mesh2D, PointLocator2D, read_gmsh_mesh};
use crate::operators::{DGOperators2D, GeometricFactors2D};
use crate::solver::Probe2D;
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
    /// The parent model of each `[[boundary]]` nested in one, sampled at the
    /// boundary and its relaxation band, its tide corrected
    pub parents: Vec<(SideTag, OceanModelState)>,
    /// Where each `[[output.station]]` samples the model
    pub stations: Vec<StationProbe>,
    /// What was done to build it, line by line, for a log
    pub notes: Vec<String>,
}

/// Where a `[[output.station]]` samples the model.
#[derive(Clone, Debug)]
pub struct StationProbe {
    /// Its name
    pub name: String,
    /// The sampling point
    pub probe: Probe2D,
    /// How far the point is from the station's position (m): 0 unless it
    /// was moved into open water
    pub offset_m: f64,
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
            match b.raster.as_ref().expect("validated").sampling {
                Sampling::Projected => {
                    Bathymetry2D::project(&grid, &ops, &grid_geom, sampler, raster.pixel_size())
                }
                Sampling::Point => Bathymetry2D::from_function(&grid, &ops, &grid_geom, sampler),
            }
        };
        if b.lower_wall_land {
            let w = bed.lower_wall_land(&grid, &ops, &grid_geom, b.wall_low_water_m);
            notes.push(format!(
                "wall land lowered in {} elements: {} nodes, by up to {:.1} m, {:.4} km³ of water added",
                w.elements,
                w.changed,
                w.max_change,
                w.volume / 1e9
            ));
        }
        if b.raster.is_some() {
            let raised = bed.raise_isolated_wet_nodes(&grid, &ops, &grid_geom, 0.0);
            notes.push(format!(
                "{raised} isolated wet nodes raised to the lowest of their neighbours"
            ));
        }
        if let Some(r_max) = b.smooth_rx0 {
            let s = bed.smooth_rx0(&grid, &ops, &grid_geom, r_max, b.rx0_min_depth_m);
            notes.push(format!(
                "slope factor r_x0 {:.2} → {:.2} (nodes ≥ {} m deep): {} nodes changed by up to {:.1} m",
                s.before, s.after, b.rx0_min_depth_m, s.changed, s.max_change
            ));
        }
        if let Some(spec) = &b.slopes_3d {
            Self::bed_for_3d(spec, &grid, &ops, &grid_geom, &mut bed, &mut notes);
        }

        let (mut mesh, mut bathymetry, geom) = if case.mesh.drop_land {
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
            if let Some(gauge) = &spec.gauge {
                for change in self.correct_at_gauge(&mut atlas, gauge)? {
                    notes.push(format!("tidal atlas {}: {change}", path.display()));
                }
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

        // Parent models: the open faces they have no water at become walls,
        // and the bed is blended to theirs across the band
        let mut parents = Vec::new();
        for b in &case.boundaries {
            let Some(spec) = &b.nesting else { continue };
            let projection = projection.as_ref().expect("validated: placed");
            let parent = self.nest(
                b.tag,
                spec,
                &mut mesh,
                &ops,
                &geom,
                &mut bathymetry,
                projection,
                &mut notes,
            )?;
            parents.push((b.tag, parent));
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
        let stations = self.stations(&mesh, &ops, &bathymetry, projection.as_ref(), &mut notes)?;
        Ok(Domain {
            mesh: Arc::new(mesh),
            ops: Arc::new(ops),
            geom: Arc::new(geom),
            bathymetry: Arc::new(bathymetry),
            open_tags,
            projection,
            atlases,
            parents,
            stations,
            notes,
        })
    }

    /// Where each `[[output.station]]` samples the model: in open water
    /// (`Probe2D::in_open_water`), or where it is with
    /// `open_water_depth_m = 0`.
    fn stations(
        &self,
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        bathymetry: &Bathymetry2D,
        projection: Option<&LocalProjection>,
        notes: &mut Vec<String>,
    ) -> Result<Vec<StationProbe>, CaseError> {
        let case = &self.case;
        let locator = PointLocator2D::new(mesh);
        let mut stations = Vec::new();
        for s in &case.output.stations {
            let (x, y) = match (s.x_m, s.y_m, s.lon, s.lat) {
                (Some(x), Some(y), _, _) => (x, y),
                (_, _, Some(lon), Some(lat)) => {
                    projection.expect("validated: placed").geo_to_xy(lat, lon)
                }
                _ => unreachable!("validated"),
            };
            let (depth, reach) = (s.open_water_depth_m, s.max_offset_m);
            let at = Probe2D::at(&locator, ops, [x, y]);
            let found = if depth > 0.0 {
                Probe2D::in_open_water(&locator, ops, bathymetry, [x, y], depth, reach)
            } else {
                at.clone().map(|probe| (probe, 0.0))
            };
            let Some((probe, offset_m)) = found else {
                return Err(CaseError::Invalid(match at {
                    None => format!(
                        "station {:?} at ({x:.1}, {y:.1}) m is outside the mesh{}",
                        s.name,
                        if case.mesh.drop_land {
                            " (or on land that was dropped)"
                        } else {
                            ""
                        }
                    ),
                    Some(_) => format!(
                        "station {:?} at ({x:.1}, {y:.1}) m has no element at least {depth} m \
                         deep within {reach} m (`open_water_depth_m = 0` samples it where it is)",
                        s.name
                    ),
                }));
            };
            if offset_m > 0.0 {
                let bed = probe.evaluate(bathymetry.element(probe.element()));
                notes.push(format!(
                    "station {:?} sampled {offset_m:.0} m from its position, in the nearest \
                     element at least {depth} m deep ({:.1} m deep there)",
                    s.name, -bed
                ));
            }
            stations.push(StationProbe {
                name: s.name.clone(),
                probe,
                offset_m,
            });
        }
        Ok(stations)
    }

    /// `[bathymetry.slopes_3d]`: smooth the element slopes, then raise the
    /// cliff shores and fill the tidal ponds, raising the wet nodes either
    /// leaves isolated after each (`froya_real_data slopes3d=on`'s order).
    fn bed_for_3d(
        spec: &Slopes3D,
        grid: &Mesh2D,
        ops: &DGOperators2D,
        geom: &GeometricFactors2D,
        bed: &mut Bathymetry2D,
        notes: &mut Vec<String>,
    ) {
        let thin = spec.thin_depth_m;
        let s = bed.smooth_element_slopes(grid, ops, geom, spec.bound(), thin);
        notes.push(format!(
            "smoothed for 3D (within an element r_x0 ≤ {} below {} m, columns ≥ {thin} m, \
             {} sweeps): {} elements over the bound (largest excess {:.1} m), {} nodes changed \
             by up to {:.1} m, volume kept; {} elements left over",
            spec.r_max,
            spec.free_depth_m,
            s.sweeps,
            s.elements_before,
            s.excess_before,
            s.changed,
            s.max_change,
            s.elements_after
        ));
        let high = spec.high_water_m;
        if spec.raise_cliffs {
            let deep = spec.cliff_depth_m;
            let c = bed.raise_cliff_shores(grid, ops, geom, deep, thin, high, high);
            let raised = bed.raise_isolated_wet_nodes(grid, ops, geom, 0.0);
            notes.push(format!(
                "cliff shores raised to land (+{high} m) in {} elements deeper than {deep} m: \
                 {} nodes, by up to {:.1} m, {:.3} km² of water at mean sea level; {raised} \
                 isolated wet nodes raised after",
                c.elements,
                c.nodes,
                c.max_raise,
                c.area / 1e6
            ));
        }
        if spec.fill_ponds {
            let low = spec.pond_low_water_m;
            let p = bed.fill_tidal_ponds(grid, ops, geom, low, high, ops.n_nodes);
            let raised = bed.raise_isolated_wet_nodes(grid, ops, geom, 0.0);
            notes.push(format!(
                "tidal ponds between {low} and +{high} m filled to their sills: {} ponds, {} \
                 nodes, by up to {:.1} m, {:.3} km², {:.2e} m³; {} larger ponds left; {raised} \
                 isolated wet nodes raised after",
                p.ponds,
                p.nodes,
                p.max_raise,
                p.area / 1e6,
                p.volume,
                p.left
            ));
        }
    }

    /// Nest the faces tagged `tag` in the parent of `spec`: wall the faces
    /// it has no water at, sample it at the boundary and the band, blend
    /// the bed, and correct its tide.
    #[allow(clippy::too_many_arguments)]
    fn nest(
        &self,
        tag: SideTag,
        spec: &Nesting,
        mesh: &mut Mesh2D,
        ops: &DGOperators2D,
        geom: &GeometricFactors2D,
        bathymetry: &mut Bathymetry2D,
        projection: &LocalProjection,
        notes: &mut Vec<String>,
    ) -> Result<OceanModelState, CaseError> {
        let path = self.resolve(&spec.file);
        let reader = Arc::new(read_parent(&path)?);
        notes.push(format!("parent {}: {}", path.display(), reader.summary()));
        let invalid =
            |e: &dyn std::fmt::Display| CaseError::Invalid(format!("{}: {e}", path.display()));
        let atlas = spec
            .atlas
            .as_ref()
            .map(|a| {
                let a = self.resolve(a);
                TidalAtlas::read(&a).map_err(|e| CaseError::Io(format!("{}: {e}", a.display())))
            })
            .transpose()?;

        // The parent's ζ has no datum: its mean comes off
        let atlas_mean = atlas.as_ref().and_then(TidalAtlas::mean_level);
        let (datum, datum_source) = match (spec.level_m, atlas_mean) {
            (Some(level), _) => (ParentDatum::Shift(level), "level_m"),
            (None, Some(mean)) => (ParentDatum::Shift(-mean), "minus the atlas's mean level"),
            (None, None) => (
                ParentDatum::RecordMean,
                "minus the parent's own mean at the boundary",
            ),
        };
        let mut options = NestingOptions::default()
            .with_band(1000.0 * spec.band_km)
            .with_ramp_up(3600.0 * spec.ramp_hours)
            .with_datum(datum);
        if spec.transport_scaling == 1.0 {
            options = options.with_transport_scaling(false);
        } else {
            options.max_transport_ratio = spec.transport_scaling;
        }

        let boundary_tag = tag.boundary_tag();
        let coverage =
            BoundaryCoverage::of_parent(mesh, ops, projection, boundary_tag, &reader, &options);
        let closed = coverage.close(mesh, spec.max_closed).map_err(|e| {
            CaseError::Invalid(format!(
                "{}: {e}: a parent cut for this domain is needed \
                 (`norkyst_nesting_subset bbox=…`), or a larger boundary.nesting.max_closed",
                path.display()
            ))
        })?;
        notes.push(format!(
            "parent {}: {closed} of {} {:?} faces without parent water made walls",
            path.display(),
            coverage.n_faces(),
            tag.name()
        ));
        let clock = self
            .case
            .clock()?
            .expect("validated: nesting needs a start");
        let parent =
            OceanModelState::new(reader, mesh, ops, projection, boundary_tag, clock, &options)
                .map_err(|e| invalid(&e))?;
        notes.push(format!(
            "parent ζ shifted by {:+.3} m to mean sea level ({datum_source}); {} boundary nodes \
             ({} snapped to a wet parent point), {} nodes in a {} km band",
            parent.reference_level(),
            parent.n_boundary_nodes(),
            parent.n_snapped(),
            parent.n_band_nodes(),
            spec.band_km
        ));
        if let Some((lo, median, hi)) = parent.depth_ratios(bathymetry) {
            notes.push(format!(
                "parent/child depth at the boundary: {lo:.2}–{hi:.2} (median {median:.2})"
            ));
        }
        if spec.blend_bed && spec.band_km > 0.0 {
            let blended = parent.blend_bathymetry(bathymetry, ops, geom, spec.blend_shore_depth_m);
            notes.push(format!(
                "bed blended to the parent's at {blended} nodes of the band (elements with a \
                 node shallower than {} m kept)",
                spec.blend_shore_depth_m
            ));
        }

        // The parent's tide corrected by (corrected − raw) atlas: the atlas
        // is the parent's own harmonic fit, so its residual flow stays
        let parent = match atlas {
            Some(raw) if spec.correction != AtlasCorrection::None || spec.gauge.is_some() => {
                let mut corrected = raw.clone();
                if spec.correction == AtlasCorrection::Theory {
                    corrected.infer_from_theory().map_err(|e| invalid(&e))?;
                }
                if let Some(gauge) = &spec.gauge {
                    for change in self.correct_at_gauge(&mut corrected, gauge)? {
                        notes.push(format!("parent's tide: {change}"));
                    }
                }
                let correction = corrected.difference(&raw).map_err(|e| invalid(&e))?;
                let keeps = |name: &str| {
                    correction.points.iter().all(|p| {
                        p.constituents
                            .iter()
                            .filter(|c| c.name == name)
                            .all(|c| c.eta.0 < 1e-3)
                    })
                };
                if spec.correction == AtlasCorrection::Theory && keeps("K2") && keeps("P1") {
                    notes.push(
                        "the atlas has K2 and P1 at the equilibrium ratios already, so theory \
                         adds the minor semidiurnals only (an atlas made with \
                         `norkyst_boundary_tides ratios=` has the parent's own)"
                            .into(),
                    );
                }
                let (largest, name) = correction
                    .points
                    .iter()
                    .flat_map(|p| p.constituents.iter().map(|c| (c.eta.0, c.name)))
                    .fold((0.0, ""), |a, b| if b.0 > a.0 { b } else { a });
                if largest == 0.0 {
                    parent
                } else {
                    notes.push(format!(
                        "parent's tide corrected by (corrected − raw) atlas, largest \
                         {largest:.3} m ({name})"
                    ));
                    parent
                        .with_tidal_correction(
                            &correction,
                            projection,
                            spec.atlas_coverage_m + 1000.0 * spec.band_km,
                        )
                        .map_err(|e| invalid(&e))?
                }
            }
            _ => parent,
        };
        let t_end = 3600.0 * self.case.run.duration_hours;
        parent
            .check_time_coverage(0.0, t_end)
            .map_err(|e| invalid(&e))?;
        Ok(parent)
    }

    /// Correct `atlas` at the gauge of `spec`; what changed.
    fn correct_at_gauge(
        &self,
        atlas: &mut TidalAtlas,
        spec: &GaugeCorrection,
    ) -> Result<Vec<GaugeAdjustment>, CaseError> {
        let record_path = self.resolve(&spec.record);
        let io = |path: &std::path::Path, e: &dyn std::fmt::Display| {
            CaseError::Io(format!("{}: {e}", path.display()))
        };
        let record = read_tide_gauge_file(&record_path).map_err(|e| io(&record_path, &e))?;
        let station = record.station.as_ref().ok_or_else(|| {
            CaseError::Invalid(format!(
                "{}: the record has no station position",
                record_path.display()
            ))
        })?;
        let (times, values): (Vec<f64>, Vec<f64>) = record
            .time_series
            .times()
            .iter()
            .copied()
            .zip(record.time_series.values().iter().copied())
            .filter(|(_, v)| v.is_finite())
            .unzip();
        let fit = HarmonicAnalysis::fit_gauge_record(&times, &values)
            .map_err(|e| CaseError::Invalid(format!("{}: {e}", record_path.display())))?;
        let source_path = self.resolve(&spec.source);
        let source = TidalAtlas::read(&source_path).map_err(|e| io(&source_path, &e))?;
        let (point, distance) = source
            .nearest(station.longitude, station.latitude)
            .filter(|&(_, d)| d <= spec.max_offset_m)
            .ok_or_else(|| {
                CaseError::Invalid(format!(
                    "{}: no point within {} m of the gauge {}",
                    source_path.display(),
                    spec.max_offset_m,
                    station.name
                ))
            })?;
        let gains: Vec<&str> = spec.gains.iter().map(String::as_str).collect();
        let ratios: Vec<&str> = spec.ratios.iter().map(String::as_str).collect();
        let changes = atlas
            .correct_at_gauge(&fit, Some(point), &gains, &ratios)
            .map_err(|e| {
                CaseError::Invalid(format!(
                    "gauge {} ({distance:.0} m from the source point): {e}",
                    station.name
                ))
            })?;
        Ok(changes)
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

/// The parent model's output at `path`.
#[cfg(feature = "netcdf")]
fn read_parent(path: &std::path::Path) -> Result<OceanModelReader, CaseError> {
    OceanModelReader::from_file(path).map_err(|e| CaseError::Io(format!("{}: {e}", path.display())))
}

#[cfg(not(feature = "netcdf"))]
fn read_parent(path: &std::path::Path) -> Result<OceanModelReader, CaseError> {
    Err(CaseError::Invalid(format!(
        "{}: boundary.nesting reads NetCDF, which needs dgcom built with the `netcdf` feature \
         (on by default)",
        path.display()
    )))
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

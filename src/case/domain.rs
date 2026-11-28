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
//! are placed: their surface in open water as the example's are
//! (`Probe2D::in_open_water`), their current where the bed is that deep
//! (`Probe2D::nearest_at_depth`).

use std::sync::Arc;

use super::patch::{DomainPatch, WholeMesh};
use super::schema::{
    AtlasCorrection, AtmosphereSection, BathymetrySection, BedFor3D, BoundarySection, Case,
    FarmSection, GaugeCorrection, MeshSection, Nesting, RasterKind, RiversSection, Sampling,
    SideTag, Slopes3D, Station,
};
use super::{CaseError, CaseFile};
use crate::analysis::{GaugeRecord, HarmonicAnalysis, TideGaugeStation};
use crate::boundary::{
    BoundaryCoverage, GaugeAdjustment, NestingOptions, OceanModelState, ParentDatum,
    ParentSeries2D, RunParent, TidalAtlas,
};
use crate::io::{
    AtmosphereReader, BedRaster, CoastlineData, CoordinateProjection, GeoBoundingBox,
    GeoTiffBathymetry, LocalProjection, OceanModelReader, SoundingCorrection, read_farm_site_file,
    read_soundings_file, read_tide_gauge_file,
};
use crate::mesh::{Bathymetry2D, BoundaryTag, Mesh2D, PointLocator2D, read_gmsh_mesh};
use crate::operators::{DGOperators2D, GeometricFactors2D};
use crate::physics::Coare35;
use crate::solver::Probe2D;
use crate::source::{CageFootprint, NetCage, RiverSources, SpongeProfile};
use crate::source::{GriddedAtmosphere2D, GriddedSurfaceFluxes, GriddedWindStress};
use crate::tides::canonical_name;
use crate::types::ElementIndex;
use crate::waves::{PointSpectra, WindSeries};

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
    /// The tidal constituents the boundaries carry: those of each
    /// `[boundary.tide]` and of each atlas after its correction; for a
    /// parent model the eight major ones ([`HarmonicAnalysis::MAJOR`]) and
    /// those of its atlas after its correction. A station's record is
    /// fitted with them ([`HarmonicAnalysis::station_candidates`]).
    pub tide_constituents: Vec<&'static str>,
    /// Where each `[[output.station]]` samples the model
    pub stations: Vec<StationProbe>,
    /// `[atmosphere]`'s weather on the mesh, if any: in a 3D run its
    /// pressure only (the depth mean's), the wind in `wind_3d`
    pub atmosphere: Option<GriddedAtmosphere2D>,
    /// In a 3D run, the weather's wind stress on the columns
    pub wind_3d: Option<GriddedWindStress>,
    /// In a 3D run with `heat`, the weather's heat and freshwater fluxes
    pub fluxes_3d: Option<GriddedSurfaceFluxes>,
    /// The parent model's output, with its profiles in a 3D run
    pub parent_reader: Option<Arc<OceanModelReader>>,
    /// `[waves]`: their mesh, bed, sea and weather
    pub waves: Option<WaveDomain>,
    /// `[farm]`: the cages, their drag scaled by `drag_share`
    pub cages: Vec<NetCage>,
    /// `[rivers]`: the rivers placed on the mesh
    pub rivers: Option<RiverSources>,
    /// `[mesh.patch]`: what was cut out of the whole mesh
    pub patch: Option<DomainPatch>,
    /// `mesh.patch.whole_run`: the whole run's state at the patch's open
    /// faces and band
    pub run_parent: Option<RunParent>,
    /// What was done to build it, line by line, for a log
    pub notes: Vec<String>,
}

/// The waves' mesh and inputs (`[waves]`).
pub struct WaveDomain {
    /// The waves' mesh: their own (`[waves.mesh]`) or the run's
    pub mesh: Arc<Mesh2D>,
    /// Its operators
    pub ops: Arc<DGOperators2D>,
    /// Its geometric factors
    pub geom: Arc<GeometricFactors2D>,
    /// The case's bed on it
    pub bathymetry: Arc<Bathymetry2D>,
    /// A parent wave model's spectra on the waves' coordinates and the run's
    /// clock, and its wind (`[waves.spectra]`)
    pub parent: Option<(PointSpectra, Option<WindSeries>)>,
    /// The parent's own H_s at its points, to compare the waves with, if its
    /// file has it
    pub parent_points: Option<ParentWavePoints>,
    /// `[atmosphere]`'s weather on the waves' nodes (its wind, not ramped)
    pub weather: Option<GriddedAtmosphere2D>,
}

/// A parent wave model's points and its own significant wave height there.
#[derive(Clone, Debug)]
pub struct ParentWavePoints {
    /// Each point's longitude and latitude (degrees)
    pub lon_lat: Vec<[f64; 2]>,
    /// Their positions in the mesh's coordinates (m)
    pub positions: Vec<[f64; 2]>,
    /// Times (s of model time), ascending
    pub times: Vec<f64>,
    /// H_s (m), `[time][point]`
    pub hs: Vec<f64>,
}

impl ParentWavePoints {
    /// The parent's H_s at point `p` at time `t`, linear in time (held
    /// beyond its times).
    pub fn hs_at(&self, p: usize, t: f64) -> f64 {
        let (np, times) = (self.positions.len(), &self.times);
        if times.len() == 1 {
            return self.hs[p];
        }
        let upper = times.partition_point(|&x| x <= t).clamp(1, times.len() - 1);
        let (a, b) = (upper - 1, upper);
        let w = ((t - times[a]) / (times[b] - times[a])).clamp(0.0, 1.0);
        (1.0 - w) * self.hs[a * np + p] + w * self.hs[b * np + p]
    }
}

/// Where a `[[output.station]]` samples the model: the surface in an element
/// that stays under water (`Probe2D::in_open_water`), the depth-mean
/// current, the depth and the sea state where the bed itself is that deep
/// (`Probe2D::nearest_at_depth`), so the current is sampled at a point that
/// does not move with the mesh and the order. Both are at the station where
/// it is in open water, and with `open_water_depth_m = 0`.
#[derive(Clone, Debug)]
pub struct StationProbe {
    /// Its name
    pub name: String,
    /// Where the surface is sampled
    pub level: Probe2D,
    /// How far `level` is from the station's position (m)
    pub level_offset_m: f64,
    /// Where the current, the depth and the sea state are sampled
    pub current: Probe2D,
    /// How far `current` is from the station's position (m)
    pub current_offset_m: f64,
    /// The station's position (x, y in m)
    pub position: [f64; 2],
    /// The tide gauge it is scored against, read and fitted
    pub gauge: Option<GaugeRecord>,
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
        let (mut mesh, ops, mut geom, mut bathymetry) =
            self.mesh_and_bed(&case.mesh, projection.as_ref(), &mut notes)?;

        // `[mesh.patch]`: cut it out of the whole mesh, which is kept to
        // place the stations and rivers on as the whole run does
        let (mut patch, mut whole) = (None, None);
        if let Some(spec) = &case.mesh.patch {
            let full = WholeMesh {
                mesh,
                geom,
                bathymetry,
            };
            let cut = self.cut_patch(spec, full, &ops, projection.as_ref(), &mut notes)?;
            (mesh, geom, bathymetry) = (cut.mesh, cut.geom, cut.bathymetry);
            (patch, whole) = (Some(cut.patch), Some(cut.whole));
        }
        // A patch leaves out the `[[boundary]]` entries whose faces it cut
        // away; nested in the whole run, all of them
        let cut_tags = open_tags(&mesh)?;
        let in_run = case.mesh.patch.as_ref().and_then(|p| p.whole_run.as_ref());
        let in_use = |b: &BoundarySection| {
            patch.is_none() || (in_run.is_none() && cut_tags.contains(&b.tag))
        };
        for b in case.boundaries.iter().filter(|b| !in_use(b)) {
            notes.push(match in_run {
                Some(_) => format!(
                    "[[boundary]] tag = {:?}: the whole run's state instead, at every open face \
                     of the patch",
                    b.tag.name()
                ),
                None => format!(
                    "[[boundary]] tag = {:?}: none of its faces is in the patch, left out",
                    b.tag.name()
                ),
            });
        }

        // What the boundaries' tides carry, for the fit of a station's record
        let tide_constituents = self.tide_constituents()?;

        // Tidal atlases: the open faces they do not cover become walls
        let mut atlases = Vec::new();
        for b in case.boundaries.iter().filter(|b| in_use(b)) {
            let Some(spec) = &b.atlas else { continue };
            let path = self.resolve(&spec.file);
            let mut atlas = read_atlas(&path)?;
            let (inferred, changes) =
                self.force_atlas(&mut atlas, &path, spec.correction, spec.gauge.as_ref())?;
            let correction = match spec.correction {
                AtlasCorrection::None => String::new(),
                AtlasCorrection::Theory => ", corrected from theory".into(),
                AtlasCorrection::Minors => {
                    format!(
                        ", {} of them inferred: {}",
                        inferred.len(),
                        inferred.join(" ")
                    )
                }
            };
            for change in changes {
                notes.push(format!("tidal atlas {}: {change}", path.display()));
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
                 {:.1} km from it made walls ({})",
                path.display(),
                atlas.names().len(),
                correction,
                coverage.n_faces(),
                b.tag.name(),
                spec.coverage_m / 1000.0,
                closed_length(&coverage)
            ));
            atlases.push((b.tag, atlas));
        }
        Self::check_3d_bed_tide(case, &atlases, &mut notes);

        // Parent models: the open faces they have no water at become walls,
        // and the bed is blended to theirs across the band
        let mut parents = Vec::new();
        let mut parent_reader = None;
        for b in case.boundaries.iter().filter(|b| in_use(b)) {
            let Some(spec) = &b.nesting else { continue };
            let projection = projection.as_ref().expect("validated: placed");
            let (parent, reader) = self.nest(
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
            parent_reader = Some(reader);
        }
        // A 3D patch whose cut left its nested boundary out still starts
        // its columns from the parent, as the whole run does
        if parent_reader.is_none()
            && case.three_d.is_some()
            && let Some(spec) = case.boundaries.iter().find_map(|b| b.nesting.as_ref())
        {
            let path = self.resolve(&spec.file);
            parent_reader = Some(Arc::new(read_parent(&path, true)?));
            notes.push(format!(
                "parent {}: read for the 3D start only (the patch does not nest in it)",
                path.display()
            ));
        }

        let open_tags = open_tags(&mesh)?;
        for tag in open_tags.iter().filter(|_| in_run.is_none()) {
            if !case.boundaries.iter().any(|b| b.tag == *tag) {
                return Err(CaseError::Invalid(format!(
                    "the mesh has {tag:?} boundary faces but the case no [[boundary]] with \
                     tag = {:?}{}",
                    tag.name(),
                    match patch {
                        Some(_) => " (mesh.patch.faces = \"wall\" closes the patch's cut)",
                        None => "",
                    }
                )));
            }
        }
        for b in case.boundaries.iter().filter(|b| in_use(b)) {
            if !open_tags.contains(&b.tag) {
                return Err(CaseError::Invalid(format!(
                    "[[boundary]] tag = {:?}: the mesh has no faces with that tag",
                    b.tag.name()
                )));
            }
        }
        // Nested in the whole run's snapshots, from the restart's time (or
        // the start) to the end
        let run_parent = match (in_run, &patch) {
            (Some(path), Some(cut)) => {
                let spec = case.mesh.patch.as_ref().expect("a patch");
                let path = self.resolve(path);
                let t_start = match self.resume_path() {
                    Some(restart) => crate::io::restart_time(&restart)
                        .map_err(|e| CaseError::Io(format!("{}: {e}", restart.display())))?,
                    None => 0.0,
                };
                let t_end = 3600.0 * case.run.duration_hours;
                let tags: Vec<_> = open_tags.iter().map(|t| t.boundary_tag()).collect();
                let parent = RunParent::new(
                    &path,
                    &mesh,
                    &ops,
                    &cut.whole_elements,
                    &tags,
                    (1000.0 * spec.band_km, SpongeProfile::Cosine),
                    (t_start, t_end),
                )
                .map_err(|e| {
                    CaseError::Invalid(format!("mesh.patch.whole_run {}: {e}", path.display()))
                })?;
                let (frames, first, last) = parent.frames();
                notes.push(format!(
                    "patch: nested in the whole run's snapshots {}: {frames} frames over \
                     {:.3}–{:.3} h at {} nodes (its open faces and a {} km band, {} min)",
                    path.display(),
                    first / 3600.0,
                    last / 3600.0,
                    parent.n_slots(),
                    spec.band_km,
                    spec.band_minutes
                ));
                Some(parent)
            }
            _ => None,
        };
        // A patch's stations: placed on the whole mesh, kept where they fall
        // in it
        let stations = match (&patch, &whole) {
            (Some(patch), Some(w)) => {
                let all = self.stations(
                    &w.mesh,
                    &ops,
                    &w.bathymetry,
                    projection.as_ref(),
                    &mut notes,
                )?;
                let (kept, left_out) = patch.keep_stations(all, &mesh, &ops);
                if !left_out.is_empty() {
                    notes.push(format!(
                        "patch: {} of {} stations outside it, left out: {}",
                        left_out.len(),
                        left_out.len() + kept.len(),
                        left_out.join(", ")
                    ));
                }
                kept
            }
            _ => self.stations(&mesh, &ops, &bathymetry, projection.as_ref(), &mut notes)?,
        };
        if stations.iter().any(|s| s.gauge.is_some()) {
            notes.push(format!(
                "gauged stations: the model's record is fitted with the constituents the \
                 boundaries carry and the overtides: {}",
                HarmonicAnalysis::station_candidates(&tide_constituents).join(" ")
            ));
        }
        let atmosphere = match &case.atmosphere {
            Some(spec) => Some(self.atmosphere(spec, &mesh, &ops, projection, &mut notes)?),
            None => None,
        };
        // A 3D run: the wind on the columns, the pressure on the depth mean,
        // and the heat budget's fluxes if the weather has it
        let (atmosphere, wind_3d, fluxes_3d) = match (atmosphere, &case.three_d) {
            (Some(weather), Some(d)) => {
                let (pressure, wind) = weather.split_for_3d();
                let fluxes = if d.heat && pressure.reader().has_heat_budget() {
                    let (_, lat_min, _, lat_max) = pressure.reader().grid().bbox();
                    let bulk = Coare35::at_latitude(0.5 * (lat_min + lat_max));
                    notes.push(format!(
                        "surface heat and freshwater: COARE 3.5 on the weather, shortwave into \
                         Jerlov {} water",
                        d.jerlov
                    ));
                    Some(
                        pressure
                            .surface_fluxes(bulk)
                            .map_err(|e| CaseError::Invalid(format!("[atmosphere]: {e}")))?,
                    )
                } else {
                    if d.heat {
                        notes.push(
                            "surface heat and freshwater: none (the weather has no heat budget)"
                                .into(),
                        );
                    }
                    None
                };
                (Some(pressure), Some(wind), fluxes)
            }
            (weather, _) => (weather, None, None),
        };
        let mut domain = Domain {
            mesh: Arc::new(mesh),
            ops: Arc::new(ops),
            geom: Arc::new(geom),
            bathymetry: Arc::new(bathymetry),
            open_tags,
            projection,
            atlases,
            parents,
            tide_constituents,
            stations,
            atmosphere,
            wind_3d,
            fluxes_3d,
            parent_reader,
            waves: None,
            cages: Vec::new(),
            rivers: None,
            patch,
            run_parent,
            notes,
        };
        if let Some(farm) = &case.farm {
            let projection = projection.as_ref().expect("validated: placed");
            let mut notes = Vec::new();
            domain.cages = self.cages(farm, &domain, projection, &mut notes)?;
            domain.notes.extend(notes);
        }
        if case.waves.is_some() {
            let mut notes = Vec::new();
            domain.waves = Some(self.wave_domain(&domain, projection.as_ref(), &mut notes)?);
            domain.notes.extend(notes);
        }
        if let Some(spec) = &case.rivers {
            let projection = projection.as_ref().expect("validated: placed");
            let mut notes = Vec::new();
            fn on(d: &Domain) -> (&Mesh2D, &GeometricFactors2D, &Bathymetry2D) {
                (&d.mesh, &d.geom, &d.bathymetry)
            }
            domain.rivers = Some(match (&domain.patch, &whole) {
                // Placed on the whole mesh, kept where they enter the patch
                (Some(patch), Some(w)) => {
                    let all = (&w.mesh, &w.geom, &w.bathymetry);
                    let all = self.rivers(spec, all, projection, &mut notes)?;
                    let n = all.len();
                    let (mesh, geom, bed) = on(&domain);
                    let (kept, left_out) =
                        all.retain_elements(&patch.whole_elements, mesh, geom, bed);
                    notes.push(format!(
                        "patch: {} of the {n} rivers enter it (their elements above are the \
                         whole mesh's){}",
                        kept.len(),
                        match left_out.len() {
                            0 => String::new(),
                            _ => format!("; left out: {}", left_out.join(", ")),
                        }
                    ));
                    kept
                }
                _ => self.rivers(spec, on(&domain), projection, &mut notes)?,
            });
            domain.notes.extend(notes);
        }
        Ok(domain)
    }

    /// `[rivers]`: the file's rivers on the mesh, those whose mouth is
    /// within `snap_km` of its water (the others listed and left out), as
    /// `froya_real_data rivers=` places them.
    fn rivers(
        &self,
        spec: &RiversSection,
        (mesh, geom, bathymetry): (&Mesh2D, &GeometricFactors2D, &Bathymetry2D),
        projection: &LocalProjection,
        notes: &mut Vec<String>,
    ) -> Result<RiverSources, CaseError> {
        let path = self.resolve(&spec.file);
        let forcing = read_rivers(&path)?;
        let clock = self.case.clock()?.expect("validated: a start");
        // Daily values: the run must lie within the file's days
        let t_end = 3600.0 * self.case.run.duration_hours;
        let (first, last) = forcing.time_range();
        if clock.unix(0.0) < first || clock.unix(t_end) > last {
            return Err(CaseError::Invalid(format!(
                "{}: the rivers run from {} to {}, the run from {} to {} \
                 (`norkyst_river_subset start= days=`)",
                path.display(),
                clock.format(clock.model_time(first)),
                clock.format(clock.model_time(last)),
                clock.format(0.0),
                clock.format(t_end)
            )));
        }
        let profile: crate::source::RiverProfile = spec.profile.parse().expect("validated");
        let rivers = forcing
            .rivers(projection, &clock, &profile)
            .map_err(|e| CaseError::Invalid(format!("{}: {e}", path.display())))?;
        // Mean discharges over the run
        let offered: f64 = rivers
            .iter()
            .map(|r| r.discharge.mean_over(0.0, t_end))
            .sum();
        // The faces nested in a parent, whose water carries the rivers
        // beyond them in
        let parent_tags: Vec<_> = self
            .case
            .boundaries
            .iter()
            .filter(|b| b.nesting.is_some())
            .map(|b| b.tag.boundary_tag())
            .collect();
        let (sources, skipped) = RiverSources::on_mesh(
            rivers,
            mesh,
            geom,
            bathymetry,
            1000.0 * spec.snap_km,
            &parent_tags,
        );
        let mean = |i: usize| sources.river(i).discharge.mean_over(0.0, t_end);
        let total: f64 = (0..sources.len()).map(mean).sum();
        notes.push(format!(
            "rivers: {} of {} from {} ({total:.1} of {offered:.1} m³/s over the run), entering \
             over {}; {} left out (beyond a nested boundary, or more than {} km from the mesh's \
             water)",
            sources.len(),
            forcing.len(),
            path.display(),
            spec.profile,
            skipped.len(),
            spec.snap_km
        ));
        let mut order: Vec<usize> = (0..sources.len()).collect();
        order.sort_by(|&a, &b| mean(b).total_cmp(&mean(a)));
        for &i in order.iter().take(5) {
            notes.push(format!(
                "  river {}: {:.1} m³/s into element {} ({:.1}–{:.1} m deep; snapped {:.0} m)",
                sources.river(i).name,
                mean(i),
                sources.element(i).as_usize(),
                sources.depth_range(i).0,
                sources.depth_range(i).1,
                sources.snap_distance(i)
            ));
        }
        let farthest = (0..sources.len()).max_by(|&a, &b| {
            sources
                .snap_distance(a)
                .total_cmp(&sources.snap_distance(b))
        });
        if let Some(i) = farthest {
            notes.push(format!(
                "  farthest moved: river {}, {:.0} m",
                sources.river(i).name,
                sources.snap_distance(i)
            ));
        }
        for error in skipped.iter().take(5) {
            notes.push(format!("  left out: {error}"));
        }
        Ok(sources)
    }

    /// `[farm]`: the site's cages, their drag scaled by `drag_share`.
    fn cages(
        &self,
        farm: &FarmSection,
        domain: &Domain,
        projection: &LocalProjection,
        notes: &mut Vec<String>,
    ) -> Result<Vec<NetCage>, CaseError> {
        let path = self.resolve(&farm.site);
        let site = read_farm_site_file(&path)
            .map_err(|e| CaseError::Io(format!("{}: {e}", path.display())))?;
        let cages = site.cage_layout(projection, &farm.grid());
        let circles: Vec<([f64; 2], f64)> = cages
            .iter()
            .filter_map(|c| match c.footprint {
                CageFootprint::Circle { center, radius } => Some((center, radius)),
                _ => None,
            })
            .collect();
        if circles.is_empty() {
            return Err(CaseError::Invalid(format!(
                "{}: no cages (no `cage` lines and no frame)",
                path.display()
            )));
        }
        let locator = PointLocator2D::new(&domain.mesh);
        if let Some((c, _)) = circles.iter().find(|(c, _)| locator.locate(*c).is_none()) {
            return Err(CaseError::Invalid(format!(
                "{}: a cage at ({:.1}, {:.1}) km is outside the mesh",
                path.display(),
                c[0] / 1e3,
                c[1] / 1e3
            )));
        }
        let n = circles.len() as f64;
        let centre = [0, 1].map(|d| circles.iter().map(|(c, _)| c[d]).sum::<f64>() / n);
        let (lat, lon) = projection.xy_to_geo(centre[0], centre[1]);
        notes.push(format!(
            "farm: site {} {}, {} cages of radius {:.1} m, nets {:.0} m deep, solidity {} \
             (C_d·a {:.4} 1/m), centred at {lat:.5}°N {lon:.5}°E; drag share {}",
            site.number,
            site.name,
            cages.len(),
            circles[0].1,
            cages[0].net_depth,
            farm.solidity,
            cages[0].drag_per_length,
            farm.drag_share
        ));
        Ok(cages
            .into_iter()
            .map(|c| NetCage {
                drag_per_length: farm.drag_share * c.drag_per_length,
                ..c
            })
            .collect())
    }

    /// `[waves]`: the waves' mesh (their own or the run's), the parent's
    /// spectra, and the weather on the waves' nodes.
    fn wave_domain(
        &self,
        domain: &Domain,
        projection: Option<&LocalProjection>,
        notes: &mut Vec<String>,
    ) -> Result<WaveDomain, CaseError> {
        let spec = self.case.waves.as_ref().expect("a waves case");
        let (mesh, ops, geom, bathymetry) = match &spec.mesh {
            Some(m) => {
                let mut own = Vec::new();
                let (mesh, ops, geom, bed) = self.mesh_and_bed(m, projection, &mut own)?;
                notes.push(format!(
                    "waves' own mesh: {} elements of order {} ({} nodes)",
                    mesh.n_elements,
                    ops.order,
                    mesh.n_elements * ops.n_nodes
                ));
                notes.extend(own.into_iter().map(|n| format!("waves' mesh: {n}")));
                (Arc::new(mesh), Arc::new(ops), Arc::new(geom), Arc::new(bed))
            }
            None => (
                domain.mesh.clone(),
                domain.ops.clone(),
                domain.geom.clone(),
                domain.bathymetry.clone(),
            ),
        };
        let weather = match &domain.atmosphere {
            Some(atmosphere) => Some(
                atmosphere
                    .on_mesh(&mesh, &ops)
                    .map_err(|e| CaseError::Invalid(format!("[atmosphere] on the waves: {e}")))?,
            ),
            None => None,
        };
        let mut parent_points = None;
        let parent = match &spec.spectra {
            Some(spectra) => {
                let path = self.resolve(&spectra.file);
                let projection = projection.expect("validated: placed");
                let clock = self.case.clock()?.expect("validated: a start");
                let (spectra, wind, points, line) = read_wave_spectra(&path, projection, &clock)?;
                parent_points = points;
                let t_end = 3600.0 * self.case.run.duration_hours;
                let (first, last) = (
                    spectra.times.first().copied().unwrap_or(0.0),
                    spectra.times.last().copied().unwrap_or(0.0),
                );
                notes.push(line);
                if first > 0.0 || last < t_end {
                    notes.push(format!(
                        "warning: the run (0 → {:.1} h) reaches beyond the spectra ({:.1} → \
                         {:.1} h); the boundary is held at the ends",
                        t_end / 3600.0,
                        first / 3600.0,
                        last / 3600.0
                    ));
                }
                Some((spectra, wind))
            }
            None => None,
        };
        Ok(WaveDomain {
            mesh,
            ops,
            geom,
            bathymetry,
            parent,
            parent_points,
            weather,
        })
    }

    /// `[atmosphere]`: the weather sampled at every node of the mesh,
    /// ramped up, and checked to cover the run.
    fn atmosphere(
        &self,
        spec: &AtmosphereSection,
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        projection: Option<LocalProjection>,
        notes: &mut Vec<String>,
    ) -> Result<GriddedAtmosphere2D, CaseError> {
        let case = &self.case;
        let paths: Vec<_> = spec.files.iter().map(|f| self.resolve(f)).collect();
        let reader = read_weather(&paths)?;
        notes.push(format!("weather: {}", reader.summary()));
        let projection = projection.expect("validated: placed");
        let clock = case.clock()?.expect("validated: a start");
        let invalid = |e: String| CaseError::Invalid(format!("[atmosphere]: {e}"));
        let mut atmosphere =
            GriddedAtmosphere2D::new(Arc::new(reader), mesh, ops, projection, clock)
                .map_err(invalid)?
                .with_ramp_up(3600.0 * spec.ramp_hours);
        if !spec.wind {
            atmosphere = atmosphere.without_wind();
        }
        if !spec.pressure {
            atmosphere = atmosphere.without_pressure();
        }
        atmosphere
            .check_time_coverage(0.0, case.forcing_end_s())
            .map_err(invalid)?;
        Ok(atmosphere)
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
                Probe2D::in_open_water(&locator, ops, bathymetry, [x, y], depth, reach).zip(
                    Probe2D::nearest_at_depth(&locator, ops, bathymetry, [x, y], depth, reach),
                )
            } else {
                at.clone().map(|probe| ((probe.clone(), 0.0), (probe, 0.0)))
            };
            let Some(((level, level_offset_m), (current, current_offset_m))) = found else {
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
            if level_offset_m > 0.0 || current_offset_m > 0.0 {
                let bed = |p: &Probe2D| -p.evaluate(bathymetry.element(p.element()));
                notes.push(format!(
                    "station {:?} sampled its surface {level_offset_m:.0} m from its position, \
                     in the nearest element at least {depth} m deep ({:.1} m deep there), and \
                     its current {current_offset_m:.0} m from it, {:.1} m deep",
                    s.name,
                    bed(&level),
                    bed(&current)
                ));
            }
            let gauge = self.station_gauge(s)?.map(|(gauge, note)| {
                notes.push(note);
                gauge
            });
            stations.push(StationProbe {
                name: s.name.clone(),
                level,
                level_offset_m,
                current,
                current_offset_m,
                position: [x, y],
                gauge,
            });
        }
        Ok(stations)
    }

    /// The case's tide against a 3D bed taken by default
    /// ([`default_3d_bed_tide_warning`]).
    fn check_3d_bed_tide(case: &Case, atlases: &[(SideTag, TidalAtlas)], notes: &mut Vec<String>) {
        let atlases: Vec<&TidalAtlas> = atlases.iter().map(|(_, a)| a).collect();
        notes.extend(default_3d_bed_tide_warning(
            &case.bathymetry,
            case.three_d.is_some(),
            &atlases,
        ));
    }

    /// A 3D run's bed against the slope bound of `[bathymetry.slopes_3d]`
    /// (its own, or the defaults' when it is left out): how many elements
    /// let a σ-level cross the pycnocline between two of their resolved
    /// nodes (at least `thin` deep), and the largest of them. σ-levels
    /// across such slopes drive spurious currents (the hydrostatic
    /// inconsistency of the pressure gradient, Beckmann & Haidvogel 1993;
    /// ROMS reports r_x0 for every grid). Plain r_x0 does not tell a treated
    /// bed from one that blows up: the refined Frøya mesh has r_x0 0.97 (a
    /// 1.1 m node beside an 83.6 m one) untreated and 0.90 (1.0 m beside
    /// 19.0 m, above the free depth) treated. Untreated, its 9193 elements
    /// over the bound grew σ-level currents of 21 m/s within 400 s and blew
    /// up at `cfl_3d = 4` (TODO P1.6, 2026-11-09). A bed over the bound
    /// that was not smoothed (`spec` none: `slopes_3d = false`, or an
    /// analytic bed) is warned about.
    #[allow(clippy::too_many_arguments)]
    fn report_3d_slopes(
        b: &BathymetrySection,
        spec: Option<&Slopes3D>,
        thin: f64,
        grid: &Mesh2D,
        ops: &DGOperators2D,
        geom: &GeometricFactors2D,
        bed: &Bathymetry2D,
        notes: &mut Vec<String>,
    ) {
        let bound = spec.cloned().unwrap_or_default().bound();
        let excess = bed.element_slope_excess(grid, ops, geom, bound, thin);
        let over = excess.iter().filter(|&&e| e > bound.tolerance()).count();
        let (worst, largest) =
            excess.iter().enumerate().fold(
                (0, 0.0),
                |(k, m), (j, &e)| if e > m { (j, e) } else { (k, m) },
            );
        let rule = format!(
            "r_x0 ≤ {} within an element below {} m, nodes ≥ {thin} m deep",
            bound.r_max, bound.free_depth
        );
        if over == 0 {
            notes.push(format!(
                "3D bed: every element within the σ-levels' slope bound ({rule})"
            ));
            return;
        }
        let corners = grid.element_vertices(ElementIndex::new(worst));
        let [x, y] = corners
            .iter()
            .fold([0.0, 0.0], |c, v| [c[0] + 0.25 * v[0], c[1] + 0.25 * v[1]]);
        notes.push(format!(
            "3D bed: {over} elements over the σ-levels' slope bound ({rule}), the largest \
             {largest:.1} m too deep, at ({:.1}, {:.1}) km",
            x / 1e3,
            y / 1e3
        ));
        if spec.is_none() {
            let instead = if b.slopes_3d == BedFor3D::Off {
                "leave out `slopes_3d = false` to smooth the bed for 3D"
            } else {
                "[bathymetry.slopes_3d] smooths the bed for 3D"
            };
            notes.push(format!(
                "warning: σ-levels across these slopes drive spurious currents that can blow \
                 the run up; {instead}"
            ));
        }
    }

    /// `[bathymetry.slopes_3d]`: smooth the element slopes, then raise the
    /// cliff shores and fill the tidal ponds, raising the wet nodes either
    /// leaves isolated after each (`froya_real_data slopes3d=on`'s order).
    fn bed_for_3d(
        spec: &Slopes3D,
        thin: f64,
        grid: &Mesh2D,
        ops: &DGOperators2D,
        geom: &GeometricFactors2D,
        bed: &mut Bathymetry2D,
        notes: &mut Vec<String>,
    ) {
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
    ) -> Result<(OceanModelState, Arc<OceanModelReader>), CaseError> {
        let path = self.resolve(&spec.file);
        let reader = Arc::new(read_parent(&path, self.case.three_d.is_some())?);
        notes.push(format!("parent {}: {}", path.display(), reader.summary()));
        // A patch's cut lies inside the domain, away from a boundary
        // atlas's points on its perimeter
        let hint = match (&self.case.mesh.patch, &spec.atlas) {
            (Some(_), Some(_)) => {
                "; a patch's cut is inside the domain, away from the atlas's points on its \
                 perimeter: boundary.<n>.nesting.atlas_coverage_m takes them from farther"
            }
            _ => "",
        };
        let invalid = |e: &dyn std::fmt::Display| {
            CaseError::Invalid(format!("{}: {e}{hint}", path.display()))
        };
        let atlas = spec
            .atlas
            .as_ref()
            .map(|a| {
                let a = self.resolve(a);
                read_atlas(&a).map(|atlas| (atlas, a))
            })
            .transpose()?;

        // The parent's ζ has no datum: its mean comes off
        let atlas_mean = atlas.as_ref().and_then(|(a, _)| a.mean_level());
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
            "parent {}: {closed} of {} {:?} faces without parent water made walls ({})",
            path.display(),
            coverage.n_faces(),
            tag.name(),
            closed_length(&coverage)
        ));
        let clock = self
            .case
            .clock()?
            .expect("validated: nesting needs a start");
        let parent = OceanModelState::new(
            reader.clone(),
            mesh,
            ops,
            projection,
            boundary_tag,
            clock,
            &options,
        )
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
            Some((raw, atlas_path))
                if spec.correction != AtlasCorrection::None || spec.gauge.is_some() =>
            {
                let mut corrected = raw.clone();
                let (_, changes) = self.force_atlas(
                    &mut corrected,
                    &atlas_path,
                    spec.correction,
                    spec.gauge.as_ref(),
                )?;
                for change in changes {
                    notes.push(format!("parent's tide: {change}"));
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
        let (t_end, t_forcing) = (
            3600.0 * self.case.run.duration_hours,
            self.case.forcing_end_s(),
        );
        parent.check_time_coverage(0.0, t_forcing).map_err(|e| {
            if t_forcing > t_end {
                invalid(&format!(
                    "{e}: a 3D run reads its forcing up to {:.0} s past its end (a third of \
                     three_d.dt_max_s); cut the parent an hour longer or end the run earlier",
                    t_forcing - t_end
                ))
            } else {
                invalid(&e)
            }
        })?;
        Ok((parent, reader))
    }

    /// The tidal constituents the boundaries carry, as
    /// [`Domain::tide_constituents`] holds them: each `[boundary.tide]`'s,
    /// then each atlas's as the run forces with it, then for each parent
    /// model the eight major ones and its atlas's; each once, in that order.
    /// It reads the atlases and the gauges that correct them, not the mesh.
    pub fn tide_constituents(&self) -> Result<Vec<&'static str>, CaseError> {
        let boundaries = &self.case.boundaries;
        let mut carried: Vec<&'static str> = Vec::new();
        let mut carry = |names: &[&'static str]| {
            for name in names {
                if !carried.contains(name) {
                    carried.push(name);
                }
            }
        };
        for tide in boundaries.iter().filter_map(|b| b.tide.as_ref()) {
            let names: Vec<&'static str> = tide
                .constituents
                .iter()
                .filter_map(|c| canonical_name(&c.name))
                .collect();
            carry(&names);
        }
        let forced = |file: &std::path::Path, correction, gauge| {
            let path = self.resolve(file);
            let mut atlas = read_atlas(&path)?;
            self.force_atlas(&mut atlas, &path, correction, gauge)?;
            Ok::<_, CaseError>(atlas.names())
        };
        for spec in boundaries.iter().filter_map(|b| b.atlas.as_ref()) {
            carry(&forced(&spec.file, spec.correction, spec.gauge.as_ref())?);
        }
        for spec in boundaries.iter().filter_map(|b| b.nesting.as_ref()) {
            carry(&HarmonicAnalysis::MAJOR);
            if let Some(file) = &spec.atlas {
                carry(&forced(file, spec.correction, spec.gauge.as_ref())?);
            }
        }
        Ok(carried)
    }

    /// `atlas` (read from `path`) as the run forces with it: `correction`
    /// applied, then corrected at `gauge`'s record. The constituents the
    /// correction inferred (`Minors`'), and what the gauge changed.
    fn force_atlas(
        &self,
        atlas: &mut TidalAtlas,
        path: &std::path::Path,
        correction: AtlasCorrection,
        gauge: Option<&GaugeCorrection>,
    ) -> Result<(Vec<&'static str>, Vec<GaugeAdjustment>), CaseError> {
        let invalid = |e| CaseError::Invalid(format!("{}: {e}", path.display()));
        let inferred = match correction {
            AtlasCorrection::None => Vec::new(),
            AtlasCorrection::Theory => {
                atlas.infer_from_theory().map_err(invalid)?;
                Vec::new()
            }
            AtlasCorrection::Minors => atlas.infer_missing_minors().map_err(invalid)?,
        };
        let changes = match gauge {
            Some(gauge) => self.correct_at_gauge(atlas, gauge)?,
            None => Vec::new(),
        };
        Ok((inferred, changes))
    }

    /// The tide gauge record of a `[[output.station]]`, read and fitted, and
    /// what the log says of it; `None` if it has no `gauge`.
    pub(super) fn station_gauge(
        &self,
        s: &Station,
    ) -> Result<Option<(GaugeRecord, String)>, CaseError> {
        let Some(file) = &s.gauge else {
            return Ok(None);
        };
        let path = self.resolve(file);
        let record = read_tide_gauge_file(&path)
            .map_err(|e| CaseError::Io(format!("{}: {e}", path.display())))?;
        let station = record.station.clone().unwrap_or_else(|| {
            TideGaugeStation::new(
                s.name.clone(),
                s.lon.unwrap_or(f64::NAN),
                s.lat.unwrap_or(f64::NAN),
            )
        });
        let gauge = GaugeRecord::new(
            station,
            record.time_series.times(),
            record.time_series.values(),
        );
        let note = format!(
            "station {:?}: gauge record of {:.0} days, {}",
            s.name,
            gauge.days(),
            match &gauge.fit {
                Ok(fit) => format!(
                    "{} constituents fitted (R² {:.4})",
                    fit.constants.len(),
                    fit.r_squared
                ),
                Err(e) => format!("no fit ({e}): skill against the observations only"),
            }
        );
        Ok(Some((gauge, note)))
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
    /// The mesh `m` and the case's bed on it, the way the example builds
    /// it (see the module docs): the whole grid, the bed on its nodes and
    /// treated, the land dropped. For the run's mesh and the waves' own.
    fn mesh_and_bed(
        &self,
        m: &MeshSection,
        projection: Option<&LocalProjection>,
        notes: &mut Vec<String>,
    ) -> Result<(Mesh2D, DGOperators2D, GeometricFactors2D, Bathymetry2D), CaseError> {
        let case = &self.case;
        let grid = self.grid(m, projection)?;
        let ops = DGOperators2D::new(m.order);
        let grid_geom = GeometricFactors2D::compute(&grid, &ops);

        let b = &case.bathymetry;
        let mut bed = if let Some(depth) = b.depth_m {
            Bathymetry2D::from_function(&grid, &ops, &grid_geom, |_, _| -depth)
        } else if let Some(ramp) = &b.ramp {
            Bathymetry2D::from_function(&grid, &ops, &grid_geom, |x, y| -ramp.depth(x, y))
        } else {
            let projection = projection.expect("validated: a raster needs a placement");
            let raster = self.raster(m, &grid, projection, notes)?;
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
        let slopes_3d = b.slopes_3d_for(case.three_d.is_some());
        if let Some(spec) = &slopes_3d {
            if b.slopes_3d == BedFor3D::Default {
                notes.push(format!(
                    "3D bed: [bathymetry.slopes_3d]'s defaults, as for every 3D run on a raster \
                     bed: r_x0 ≤ {} below the summer pycnocline's {} m, cliff shores raised to \
                     +{} m and ponds filled from {} m, Frøya's summer and Mausund's tides (set \
                     them for another coast; `slopes_3d = false` keeps the bed as it is)",
                    spec.r_max, spec.free_depth_m, spec.high_water_m, spec.pond_low_water_m
                ));
            }
            let thin = spec.thin_depth(case.three_d.as_ref());
            Self::bed_for_3d(spec, thin, &grid, &ops, &grid_geom, &mut bed, notes);
        }
        if let Some(three_d) = &case.three_d {
            Self::report_3d_slopes(
                b,
                slopes_3d.as_ref(),
                three_d.thin_depth_m,
                &grid,
                &ops,
                &grid_geom,
                &bed,
                notes,
            );
        }

        let (mesh, bathymetry, geom) = if m.drop_land {
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
        Ok((mesh, ops, geom, bathymetry))
    }

    fn grid(
        &self,
        m: &MeshSection,
        projection: Option<&LocalProjection>,
    ) -> Result<Mesh2D, CaseError> {
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
        m: &MeshSection,
        grid: &Mesh2D,
        projection: &LocalProjection,
        notes: &mut Vec<String>,
    ) -> Result<BedRaster, CaseError> {
        let spec = self.case.bathymetry.raster.as_ref().expect("validated");
        let path = self.resolve(&spec.file);
        let io = |e: &dyn std::fmt::Display| CaseError::Io(format!("{}: {e}", path.display()));
        // `crop`, else the mesh's own box (a geographic mesh's), else its
        // extent with a margin
        let window = match (spec.crop, &m.geographic) {
            (Some([w, s, e, n]), _) => GeoBoundingBox::new(w, s, e, n),
            (None, Some(g)) => {
                let [w, s, e, n] = g.bbox;
                GeoBoundingBox::new(w, s, e, n)
            }
            (None, None) => mesh_window(grid, projection),
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
                if spec.sea_datum_m > 0.0 {
                    let moved = raster.lower_sea_to_mean_sea_level(spec.sea_datum_m);
                    notes.push(format!(
                        "sea part lowered to mean sea level, {} m below its datum: {moved} pixels",
                        spec.sea_datum_m
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
        let mut raster = raster;
        if let Some(soundings) = &spec.soundings {
            let file = self.resolve(&soundings.file);
            let stations = read_soundings_file(&file)
                .map_err(|e| CaseError::Io(format!("{}: {e}", file.display())))?;
            let report =
                raster.correct_to_soundings(&stations, soundings.range_m, soundings.noise_m);
            notes.push(sounding_note(&file, soundings, &report));
        }
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

/// The log's line for a raster corrected to soundings.
fn sounding_note(
    file: &std::path::Path,
    spec: &super::schema::Soundings,
    r: &SoundingCorrection,
) -> String {
    let mut line = format!(
        "soundings {}: {} of {} stations on the raster's sea ({} off it, {} on its land)",
        file.display(),
        r.used,
        r.stations,
        r.off_raster,
        r.on_land
    );
    if r.used == 0 {
        return line + ", nothing corrected";
    }
    let [q1, m, q3] = SoundingCorrection::quartiles(&r.deeper_than_raster);
    let [a1, am, a3] = SoundingCorrection::quartiles(&r.deeper_than_corrected);
    line += &format!(
        ", measured less the raster's depth {m:+.1} m (quartiles {q1:+.1}, {q3:+.1}), \
         {am:+.1} m ({a1:+.1}, {a3:+.1}) after; field σ {:.1} m, noise {} m, range {} m: \
         {} pixels moved, deepened by up to {:.1} m, made shallower by up to {:.1} m",
        r.signal, spec.noise_m, spec.range_m, r.pixels, r.deepened_by, r.shoaled_by
    );
    if r.held > 0 {
        line += &format!(
            " ({} held {} m below mean sea level)",
            r.held,
            BedRaster::SHALLOWEST_SOUNDED
        );
    }
    line
}

/// What [`read_wave_spectra`] reads: the spectra, the wind, the parent's own
/// H_s at its points and a line for the log.
type WaveSpectraRead = (
    PointSpectra,
    Option<WindSeries>,
    Option<ParentWavePoints>,
    String,
);

/// A parent wave model's spectra at `path` on the coordinates of
/// `projection` and the time of `clock`, its wind, its own H_s at its points
/// (if the file has it) and a line for the log.
#[cfg(feature = "netcdf")]
fn read_wave_spectra(
    path: &std::path::Path,
    projection: &LocalProjection,
    clock: &crate::time::ModelClock,
) -> Result<WaveSpectraRead, CaseError> {
    let file = crate::io::WaveSpectraFile::from_file(path)
        .map_err(|e| CaseError::Io(format!("{}: {e}", path.display())))?;
    let model_time = |unix: f64| clock.model_time(unix);
    // Geographic east in the mesh axes, at the projection's centre
    let (ex, ey) = crate::io::east_axis(projection, projection.ref_lat(), projection.ref_lon());
    let east = ey.atan2(ex);
    let position = |lon: f64, lat: f64| {
        let (x, y) = projection.geo_to_xy(lat, lon);
        [x, y]
    };
    let spectra = file.point_spectra(position, model_time, east);
    let wind = file.wind_series(&[], model_time, east);
    let points = file.hs.as_ref().map(|hs| ParentWavePoints {
        lon_lat: file
            .longitude
            .iter()
            .zip(&file.latitude)
            .map(|(&lon, &lat)| [lon, lat])
            .collect(),
        positions: spectra.positions.clone(),
        times: spectra.times.clone(),
        hs: hs.clone(),
    });
    let line = format!(
        "wave spectra {}: {} points, {} → {} UTC{}",
        path.display(),
        file.n_points(),
        clock.format(model_time(file.times[0])),
        clock.format(model_time(file.times[file.times.len() - 1])),
        file.forecast_reference_time
            .map_or(String::new(), |t| format!(
                " (forecast from {} UTC)",
                clock.format(model_time(t))
            ))
    );
    Ok((spectra, wind, points, line))
}

#[cfg(not(feature = "netcdf"))]
fn read_wave_spectra(
    path: &std::path::Path,
    _projection: &LocalProjection,
    _clock: &crate::time::ModelClock,
) -> Result<WaveSpectraRead, CaseError> {
    Err(CaseError::Invalid(format!(
        "{}: [waves.spectra] reads NetCDF, which needs dgcom built with the `netcdf` feature",
        path.display()
    )))
}

/// The river file at `path`.
#[cfg(feature = "netcdf")]
fn read_rivers(path: &std::path::Path) -> Result<crate::io::RiverForcing, CaseError> {
    crate::io::RiverForcing::from_file(path)
        .map_err(|e| CaseError::Io(format!("{}: {e}", path.display())))
}

#[cfg(not(feature = "netcdf"))]
fn read_rivers(path: &std::path::Path) -> Result<crate::io::RiverForcing, CaseError> {
    Err(CaseError::Invalid(format!(
        "{}: [rivers] reads NetCDF, which needs dgcom built with the `netcdf` feature (on by \
         default)",
        path.display()
    )))
}

/// The tidal atlas at `path`.
fn read_atlas(path: &std::path::Path) -> Result<TidalAtlas, CaseError> {
    TidalAtlas::read(path).map_err(|e| CaseError::Io(format!("{}: {e}", path.display())))
}

/// The weather files `paths`, joined in time.
#[cfg(feature = "netcdf")]
fn read_weather(paths: &[std::path::PathBuf]) -> Result<AtmosphereReader, CaseError> {
    AtmosphereReader::from_files(paths).map_err(|e| {
        let files: Vec<_> = paths.iter().map(|p| p.display().to_string()).collect();
        CaseError::Io(format!("{}: {e}", files.join(", ")))
    })
}

#[cfg(not(feature = "netcdf"))]
fn read_weather(_paths: &[std::path::PathBuf]) -> Result<AtmosphereReader, CaseError> {
    Err(CaseError::Invalid(
        "[atmosphere] reads NetCDF, which needs dgcom built with the `netcdf` feature (on by default)"
            .into(),
    ))
}

/// The parent model's output at `path`, with its profiles for a 3D run.
#[cfg(feature = "netcdf")]
fn read_parent(path: &std::path::Path, profiles: bool) -> Result<OceanModelReader, CaseError> {
    if profiles {
        OceanModelReader::from_file_with_profiles(path)
    } else {
        OceanModelReader::from_file(path)
    }
    .map_err(|e| CaseError::Io(format!("{}: {e}", path.display())))
}

#[cfg(not(feature = "netcdf"))]
fn read_parent(path: &std::path::Path, _profiles: bool) -> Result<OceanModelReader, CaseError> {
    Err(CaseError::Invalid(format!(
        "{}: boundary.nesting reads NetCDF, which needs dgcom built with the `netcdf` feature \
         (on by default)",
        path.display()
    )))
}

/// The longitude/latitude box around a placed mesh, with a margin of 2 % of
/// its size.
/// The length made walls against the boundary's, as the notes give it
/// (`max_closed` is a share of the length).
fn closed_length(coverage: &BoundaryCoverage) -> String {
    format!(
        "{:.1} of {:.1} km, {:.1} %",
        coverage.uncovered_length() / 1000.0,
        coverage.length() / 1000.0,
        100.0 * coverage.uncovered_fraction()
    )
}

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

/// A 3D bed taken by default (`slopes_3d` left out) against the case's
/// tide. Its defaults raise cliff shores to +2 m and fill ponds from
/// −1.43 m (Mausund's tides), a band of 3.43 m. A tide whose range (twice
/// its tidal atlases' largest sum of amplitudes) is wider than that floods
/// the raised shores at high water, or leaves ponds below the band, so the
/// run is warned to set the levels. Frøya's and Bodø's NorKyst atlases give
/// 2.8–3.3 m. A case without an atlas (a parent model's tide alone) is not
/// checked: the run's note gives the levels.
fn default_3d_bed_tide_warning(
    b: &BathymetrySection,
    three_d: bool,
    atlases: &[&TidalAtlas],
) -> Option<String> {
    if b.slopes_3d != BedFor3D::Default {
        return None;
    }
    let spec = b.slopes_3d_for(three_d)?;
    let range = 2.0
        * atlases
            .iter()
            .filter_map(|a| a.largest_amplitude_sum())
            .fold(0.0, f64::max);
    (range > spec.high_water_m - spec.pond_low_water_m).then(|| {
        format!(
            "warning: the tide's range is up to {range:.2} m (twice the tidal atlas's largest sum \
             of amplitudes), wider than the default 3D bed's {} to +{} m (Mausund's tides): set \
             [bathymetry.slopes_3d] pond_low_water_m and high_water_m to this coast's lowest and \
             highest astronomical tides",
            spec.pond_low_water_m, spec.high_water_m
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression (a side review of the 3D bed's default, 2026-11-10): the
    /// default's tide levels are Mausund's, and a coast with a larger tide
    /// (Chiloé's metres) is warned to set its own.
    #[test]
    fn a_default_3d_bed_is_checked_against_the_tide() {
        let case = |bathymetry: &str| {
            Case::parse(
                &format!(
                    "[run]\nduration_hours = 1.0\n[mesh.rectangle]\nx_m = [0.0, 1000.0]\n\
                     y_m = [0.0, 100.0]\nnx = 4\nny = 1\n[bathymetry]\n{bathymetry}\n\
                     [bathymetry.raster]\nfile = \"bed.tif\"\nkind = \"topobathy\"\n"
                ),
                false,
                &[],
            )
            .unwrap()
            .bathymetry
        };
        let atlas = |amplitudes: &[(&str, f64)]| {
            let mut text = String::from("# lon lat depth name eta_amp eta_lag\n");
            for (lon, scale) in [(8.0, 0.5), (8.1, 1.0)] {
                for (name, amp) in amplitudes {
                    text += &format!("{lon} 63.6 50.0 {name} {} 0.0\n", amp * scale);
                }
            }
            TidalAtlas::parse(&text).unwrap()
        };
        // Frøya's boundary: M2 0.70, S2 0.25, … sums to 1.40 m at most
        let froya = atlas(&[
            ("M2", 0.70),
            ("S2", 0.25),
            ("N2", 0.25),
            ("K1", 0.12),
            ("O1", 0.08),
        ]);
        assert!((froya.largest_amplitude_sum().unwrap() - 1.40).abs() < 1e-12);
        // A macrotidal coast: M2 1.9 m
        let large = atlas(&[("M2", 1.9), ("S2", 0.6), ("N2", 0.4)]);

        let default = case("");
        assert_eq!(default_3d_bed_tide_warning(&default, true, &[&froya]), None);
        let warning = default_3d_bed_tide_warning(&default, true, &[&froya, &large]).unwrap();
        assert!(warning.contains("up to 5.80 m"), "{warning}");
        assert!(warning.contains("-1.43 to +2 m"), "{warning}");
        // No 3D bed, or one the case set: nothing to say
        assert_eq!(
            default_3d_bed_tide_warning(&default, false, &[&large]),
            None
        );
        let set = case("slopes_3d = { high_water_m = 3.5, pond_low_water_m = -3.5 }");
        assert_eq!(default_3d_bed_tide_warning(&set, true, &[&large]), None);
        let off = case("slopes_3d = false");
        assert_eq!(default_3d_bed_tide_warning(&off, true, &[&large]), None);
    }
}

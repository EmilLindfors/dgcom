//! `[mesh.patch]`: a case run on a cut-out of its mesh ([`Patch`]).
//!
//! The patch is cut from the whole mesh and bed after they are built, so
//! its elements, their order and their bed are the whole run's. The whole
//! mesh is kept while the stations and rivers are placed: each is placed on
//! it as in the whole run and kept if it falls in the patch, in the same
//! element. A restart of the whole run starts the patch from its state
//! ([`DomainPatch::seed_2d`], [`DomainPatch::seed_3d`]).

use super::schema::{Patch, SideTag};
use super::{CaseError, CaseFile};
use crate::io::{CoordinateProjection, LocalProjection, Restart3D};
use crate::mesh::{Bathymetry2D, Mesh2D, PointLocator2D};
use crate::operators::{DGOperators2D, GeometricFactors2D};
use crate::solver::{Probe2D, SWESolution2D};
use crate::types::ElementIndex;

/// What `[mesh.patch]` cut out of the whole mesh.
#[derive(Clone, Debug)]
pub struct DomainPatch {
    /// For each element of the patch, its index in the whole mesh
    /// (ascending)
    pub whole_elements: Vec<usize>,
    /// The number of elements of the whole mesh
    pub whole_n_elements: usize,
    /// The patch's bed as cut, before its own open boundaries blended it to
    /// a parent's across their band: the whole run's bed, but within the
    /// band of the whole domain's nested boundary, which the whole run
    /// blended and the patch does not
    pub bed_as_cut: Vec<f64>,
}

impl DomainPatch {
    /// The index in the whole mesh of the patch's element `k`.
    pub fn whole_element(&self, k: usize) -> usize {
        self.whole_elements[k]
    }

    /// The patch's index of the whole mesh's element `whole`, if the patch
    /// has it.
    pub fn element_of_whole(&self, whole: usize) -> Option<usize> {
        self.whole_elements.binary_search(&whole).ok()
    }

    /// Whether a restart of `n_elements` is the whole run's, to start the
    /// patch from, rather than the patch's own.
    pub fn is_whole(&self, n_elements: usize) -> bool {
        n_elements == self.whole_n_elements && n_elements != self.whole_elements.len()
    }

    /// The patch's 2D state from the whole run's `whole`: its elements'
    /// h, hu and hv. Where the patch's bed `bed` is not the whole run's
    /// (its own band's blending), the surface and the velocity are kept:
    /// h = max(η − B, 0).
    pub fn seed_2d(&self, whole: &SWESolution2D, bed: &Bathymetry2D) -> SWESolution2D {
        let mut q = whole.select_elements(&self.whole_elements);
        for (i, (&b, &cut)) in bed.data.iter().zip(&self.bed_as_cut).enumerate() {
            if b == cut {
                continue;
            }
            let h = q.data[0][i];
            let h_new = (h + cut - b).max(0.0);
            let ratio = if h > 0.0 { h_new / h } else { 0.0 };
            q.data[0][i] = h_new;
            q.data[1][i] *= ratio;
            q.data[2][i] *= ratio;
        }
        q
    }

    /// The patch's 3D restart from the whole run's `whole`, on the patch's
    /// domain of fingerprint `domain` ([`Restart3D::select_elements`]). The
    /// 3D state keeps η, so a bed the patch's band blended changes the
    /// columns' depth, not the surface.
    pub fn seed_3d(&self, whole: &Restart3D, domain: u64) -> Restart3D {
        whole.select_elements(&self.whole_elements, domain)
    }

    /// The station probes of the whole mesh that fall in the patch, both
    /// their level and their current in its elements, on the patch's mesh;
    /// the others' names.
    pub(super) fn keep_stations(
        &self,
        stations: Vec<super::StationProbe>,
        mesh: &Mesh2D,
        ops: &DGOperators2D,
    ) -> (Vec<super::StationProbe>, Vec<String>) {
        let (mut kept, mut left_out) = (Vec::new(), Vec::new());
        let on_patch = |p: &Probe2D| {
            let k = self.element_of_whole(p.element().as_usize())?;
            let mut point = p.point();
            point.element = ElementIndex::new(k);
            Some(Probe2D::new(mesh, ops, point))
        };
        for mut s in stations {
            match (on_patch(&s.level), on_patch(&s.current)) {
                (Some(level), Some(current)) => {
                    s.level = level;
                    s.current = current;
                    kept.push(s);
                }
                _ => left_out.push(s.name),
            }
        }
        (kept, left_out)
    }
}

/// The whole mesh a patch was cut from, with its geometry and bed, kept
/// while the stations and rivers are placed on it.
pub(super) struct WholeMesh {
    pub mesh: Mesh2D,
    pub geom: GeometricFactors2D,
    pub bathymetry: Bathymetry2D,
}

/// A cut: the patch's mesh, geometry and bed, and what it was cut from.
pub(super) struct Cut {
    pub patch: DomainPatch,
    pub whole: WholeMesh,
    pub mesh: Mesh2D,
    pub geom: GeometricFactors2D,
    pub bathymetry: Bathymetry2D,
}

impl CaseFile {
    /// The width (km) of the relaxation band at the faces tagged `tag`: a
    /// patch's own nested in the whole run, its `[[boundary]]`'s nesting
    /// band, or a 3D run's band where no parent is nested; `None` for walls
    /// and for a 2D boundary that nests in none.
    fn band_km_at(&self, tag: SideTag) -> Option<f64> {
        if let Some(p) = self
            .case
            .mesh
            .patch
            .as_ref()
            .filter(|p| p.whole_run.is_some())
        {
            return Some(p.band_km);
        }
        let b = self.case.boundaries.iter().find(|b| b.tag == tag)?;
        match (&b.nesting, &self.case.three_d) {
            (Some(n), _) => Some(n.band_km),
            (None, Some(d)) => Some(d.band_km),
            (None, None) => None,
        }
    }

    /// Cut `spec`'s patch out of the whole `mesh` and its bed.
    pub(super) fn cut_patch(
        &self,
        spec: &Patch,
        whole: WholeMesh,
        ops: &DGOperators2D,
        projection: Option<&LocalProjection>,
        notes: &mut Vec<String>,
    ) -> Result<Cut, CaseError> {
        let invalid = |m: String| CaseError::Invalid(m);
        let mesh = &whole.mesh;
        let n = mesh.n_elements;
        let centre = |e: ElementIndex| mesh.reference_to_physical(e, 0.0, 0.0);
        let point = |c: &super::GeoPoint| {
            let (x, y) = projection
                .expect("validated: placed")
                .geo_to_xy(c.lat, c.lon);
            [x, y]
        };
        let (kept, region) = match (spec.element, spec.center, spec.bbox, spec.radius_km) {
            (Some(k), _, _, Some(r)) => {
                if k >= n {
                    return Err(invalid(format!(
                        "mesh.patch.element {k}: the mesh has {n} elements (0–{})",
                        n.saturating_sub(1)
                    )));
                }
                let start = ElementIndex::new(k);
                let [x0, y0] = centre(start);
                let radius = 1000.0 * r;
                let kept = mesh.connected_elements(start, |e| {
                    let [x, y] = centre(e);
                    (x - x0).hypot(y - y0) <= radius
                });
                (kept, format!("within {r} km of element {k}"))
            }
            (_, Some(c), _, Some(r)) => {
                let [x0, y0] = point(&c);
                let start = match PointLocator2D::new(mesh).locate([x0, y0]) {
                    Some(p) => p.element,
                    // On land or off the mesh: the element nearest by centre
                    None => ElementIndex::iter(n)
                        .min_by(|&a, &b| {
                            let d = |e| {
                                let [x, y] = centre(e);
                                (x - x0).hypot(y - y0)
                            };
                            d(a).total_cmp(&d(b))
                        })
                        .ok_or_else(|| invalid("mesh.patch: the mesh is empty".into()))?,
                };
                let radius = 1000.0 * r;
                // Its own element always, even if its centre is farther
                let kept = mesh.connected_elements(start, |e| {
                    let [x, y] = centre(e);
                    e == start || (x - x0).hypot(y - y0) <= radius
                });
                (
                    kept,
                    format!(
                        "within {r} km of {} (element {} of the whole mesh)",
                        c.hemispheres(),
                        start.as_usize()
                    ),
                )
            }
            (_, _, Some([w, s, e, nn]), None) => {
                let projection = projection.expect("validated: placed");
                let kept: Vec<usize> = ElementIndex::iter(n)
                    .filter(|&k| {
                        let [x, y] = centre(k);
                        let (lat, lon) = projection.xy_to_geo(x, y);
                        (w..=e).contains(&lon) && (s..=nn).contains(&lat)
                    })
                    .map(ElementIndex::as_usize)
                    .collect();
                (kept, format!("with their centre in [{w}, {s}, {e}, {nn}]"))
            }
            _ => unreachable!("validated"),
        };
        if kept.is_empty() {
            return Err(invalid(format!("mesh.patch: no element {region}")));
        }
        let mut inside = vec![false; n];
        kept.iter().for_each(|&k| inside[k] = true);
        let tag = spec.faces.boundary_tag();
        let (patch_mesh, kept) = mesh.retain_elements(|e| inside[e.as_usize()], tag);
        // The cut's faces: those whose neighbour in the whole mesh was
        // dropped, by their midpoints
        let mut cut_length = 0.0;
        let mut midpoints = Vec::new();
        for &k in &kept {
            let e = ElementIndex::new(k);
            for face in 0..4 {
                if let Some(nb) = mesh.neighbor(e, face)
                    && !inside[nb.element]
                {
                    let v = mesh.element_vertices(e);
                    let ([x0, y0], [x1, y1]) = (v[face], v[(face + 1) % 4]);
                    cut_length += (x1 - x0).hypot(y1 - y0);
                    midpoints.push([0.5 * (x0 + x1), 0.5 * (y0 + y1)]);
                }
            }
        }
        let geom = GeometricFactors2D::compute(&patch_mesh, ops);
        let bathymetry = whole.bathymetry.select_elements(&kept);
        notes.push(format!(
            "patch: {} of {n} elements ({:.1} %) {region}; its {} cut faces ({:.1} km) made {}",
            kept.len(),
            100.0 * kept.len() as f64 / n as f64,
            midpoints.len(),
            cut_length / 1000.0,
            match spec.faces {
                SideTag::Wall => "walls".to_string(),
                t => format!("{:?} boundary faces", t.name()),
            }
        ));
        // An open cut's band: the share of the patch it relaxes to the
        // parent (or the start), which a band meant for the whole domain's
        // sides can cover whole
        if let Some(band_km) = self.band_km_at(spec.faces)
            && band_km > 0.0
            && !midpoints.is_empty()
        {
            let band = 1000.0 * band_km;
            let in_band = kept
                .iter()
                .filter(|&&k| {
                    let [x, y] = centre(ElementIndex::new(k));
                    midpoints
                        .iter()
                        .any(|[mx, my]| (x - mx).hypot(y - my) < band)
                })
                .count();
            notes.push(format!(
                "patch: its cut's {band_km} km band reaches {in_band} of its {} elements' \
                 centres ({:.0} %; the band's width narrows it)",
                kept.len(),
                100.0 * in_band as f64 / kept.len() as f64
            ));
        }
        let patch = DomainPatch {
            bed_as_cut: bathymetry.data.clone(),
            whole_elements: kept,
            whole_n_elements: n,
        };
        Ok(Cut {
            patch,
            whole,
            mesh: patch_mesh,
            geom,
            bathymetry,
        })
    }
}

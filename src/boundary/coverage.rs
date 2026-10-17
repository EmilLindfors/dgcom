//! Which open-boundary faces a forcing source covers.
//!
//! A parent model or tidal atlas cut for a smaller domain leaves part of a
//! larger domain's open boundary without data. Forcing those faces anyway is
//! an error ([`OceanModelState::new`](crate::boundary::OceanModelState::new)
//! refuses them); making them walls is the usual cure for a narrow fjord
//! arm crossing the domain edge, which the parent does not resolve. Closing
//! a large share of the boundary, though, silently turns the domain into a
//! closed box: the Frøya 3D day nested in a parent cut for the Mausund
//! sub-domain closed 120 of its 223 open faces and halved the M2 current
//! (TODO P1.3, `notes/geometry-mesh.md`). [`BoundaryCoverage`] finds the
//! uncovered faces, and [`BoundaryCoverage::close`] makes walls of them only
//! while they stay below a given share.
//!
//! A face is covered when every one of its nodes is: for a parent, the node
//! lies in a parent cell with a wet corner or within `max_snap` of a wet
//! parent point (as `OceanModelState::new` needs); for an atlas, within a
//! distance of an atlas point.

use thiserror::Error;

use crate::boundary::{NestingOptions, TidalAtlas};
use crate::io::{CoordinateProjection, OceanModelReader};
use crate::mesh::{BoundaryTag, Mesh2D};
use crate::operators::DGOperators2D;
use crate::types::ElementIndex;

/// More of the boundary is uncovered than [`BoundaryCoverage::close`] allows.
#[derive(Debug, Error, PartialEq)]
#[error(
    "{uncovered} of {faces} boundary faces tagged {tag:?} are not covered by the forcing \
     (more than {max_fraction} of them)"
)]
pub struct CoverageError {
    /// The boundary tag checked
    pub tag: BoundaryTag,
    /// Faces with that tag
    pub faces: usize,
    /// Faces with a node the source does not cover
    pub uncovered: usize,
    /// The largest share allowed
    pub max_fraction: f64,
}

/// The faces of one boundary tag a forcing source does not cover.
#[derive(Clone, Debug, PartialEq)]
pub struct BoundaryCoverage {
    tag: BoundaryTag,
    faces: usize,
    uncovered: Vec<usize>,
}

impl BoundaryCoverage {
    /// The faces tagged `tag` with a node where `covered(lon, lat)` is
    /// false; mesh coordinates map to longitude/latitude with `projection`.
    pub fn of<P: CoordinateProjection>(
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        projection: &P,
        tag: BoundaryTag,
        covered: impl Fn(f64, f64) -> bool,
    ) -> Self {
        let mut faces = 0;
        let mut uncovered = Vec::new();
        for (e, edge) in mesh.edges.iter().enumerate() {
            if edge.right.is_some() || edge.boundary_tag != Some(tag) {
                continue;
            }
            faces += 1;
            let (k, face) = (ElementIndex::new(edge.left.element), edge.left.face);
            let missed = ops.face_nodes[face].iter().any(|&i| {
                let [x, y] = mesh.reference_to_physical(k, ops.nodes_r[i], ops.nodes_s[i]);
                let (lat, lon) = projection.xy_to_geo(x, y);
                !covered(lon, lat)
            });
            if missed {
                uncovered.push(e);
            }
        }
        Self {
            tag,
            faces,
            uncovered,
        }
    }

    /// The faces tagged `tag` that `reader` cannot nest under `options`: a
    /// node outside every parent cell with a wet corner and farther than
    /// `options.max_snap` from a wet parent point.
    pub fn of_parent<P: CoordinateProjection>(
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        projection: &P,
        tag: BoundaryTag,
        reader: &OceanModelReader,
        options: &NestingOptions,
    ) -> Self {
        Self::of(mesh, ops, projection, tag, |lon, lat| {
            reader.stencil(lon, lat).is_some()
                || reader
                    .grid()
                    .nearest(lon, lat, options.max_snap, |k| reader.is_wet(k))
                    .is_some()
        })
    }

    /// The faces tagged `tag` with a node farther than `max_distance` (m)
    /// from every point of `atlas`.
    pub fn of_atlas<P: CoordinateProjection>(
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        projection: &P,
        tag: BoundaryTag,
        atlas: &TidalAtlas,
        max_distance: f64,
    ) -> Self {
        Self::of(mesh, ops, projection, tag, |lon, lat| {
            atlas
                .nearest(lon, lat)
                .is_some_and(|(_, d)| d <= max_distance)
        })
    }

    /// The boundary tag checked.
    pub fn tag(&self) -> BoundaryTag {
        self.tag
    }

    /// Faces with the tag.
    pub fn n_faces(&self) -> usize {
        self.faces
    }

    /// Indices into `mesh.edges` of the uncovered faces.
    pub fn uncovered(&self) -> &[usize] {
        &self.uncovered
    }

    /// Share of the faces that are uncovered (0 with no faces).
    pub fn uncovered_fraction(&self) -> f64 {
        if self.faces == 0 {
            0.0
        } else {
            self.uncovered.len() as f64 / self.faces as f64
        }
    }

    /// Make walls of the uncovered faces of `mesh` (the mesh the coverage
    /// was computed on) and return how many, or refuse, leaving the mesh
    /// unchanged, when more than `max_fraction` of the faces are uncovered.
    pub fn close(&self, mesh: &mut Mesh2D, max_fraction: f64) -> Result<usize, CoverageError> {
        if self.uncovered_fraction() > max_fraction {
            return Err(CoverageError {
                tag: self.tag,
                faces: self.faces,
                uncovered: self.uncovered.len(),
                max_fraction,
            });
        }
        for &e in &self.uncovered {
            debug_assert_eq!(mesh.edges[e].boundary_tag, Some(self.tag));
            mesh.edges[e].boundary_tag = Some(BoundaryTag::Wall);
        }
        Ok(self.uncovered.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::AtlasPoint;
    use crate::io::{FieldSeries, GeoGrid, LocalProjection};

    /// A parent on a 2.5 km grid over x, y ∈ [−20, 20] km, dry in the corner
    /// x, y > 7 km: the corner a subset cut for a smaller domain misses.
    fn parent(projection: &LocalProjection) -> OceanModelReader {
        let (lat0, lon0) = projection.xy_to_geo(-20_000.0, -20_000.0);
        let (lat1, lon1) = projection.xy_to_geo(20_000.0, 20_000.0);
        let n = 17;
        let axis = |a: f64, b: f64| -> Vec<f64> {
            (0..n)
                .map(|i| a + (b - a) * i as f64 / (n - 1) as f64)
                .collect()
        };
        let grid = GeoGrid::regular(axis(lon0, lon1), axis(lat0, lat1)).unwrap();
        let depth = (0..grid.len())
            .map(|k| {
                let (lon, lat) = grid.position(k);
                let (x, y) = projection.geo_to_xy(lat, lon);
                if x > 7000.0 && y > 7000.0 { 0.0 } else { 50.0 }
            })
            .collect();
        let m = grid.len();
        OceanModelReader::new(grid, vec![0.0, 3600.0])
            .unwrap()
            .with_ssh(FieldSeries::new(m, vec![0.0; 2 * m]))
            .with_depth(depth)
    }

    /// A 30 × 30 km box of 5 km elements, all open.
    fn mesh() -> Mesh2D {
        Mesh2D::uniform_rectangle_with_bc(
            -15_000.0,
            15_000.0,
            -15_000.0,
            15_000.0,
            6,
            6,
            BoundaryTag::Open,
        )
    }

    /// The parent misses the box's north-east corner: the two faces next to
    /// it on each of the east and north sides have a node more than
    /// `max_snap` from wet water; the rest of the boundary is covered.
    /// Closing them is refused above the allowed share and makes walls of
    /// exactly those faces below it.
    #[test]
    fn a_parent_missing_a_corner_leaves_those_faces_uncovered() {
        let projection = LocalProjection::new(63.5, 8.5);
        let (mut mesh, ops) = (mesh(), DGOperators2D::new(2));
        let coverage = BoundaryCoverage::of_parent(
            &mesh,
            &ops,
            &projection,
            BoundaryTag::Open,
            &parent(&projection),
            &NestingOptions::default(),
        );
        assert_eq!(coverage.n_faces(), 24);
        assert_eq!(coverage.uncovered().len(), 4, "{:?}", coverage.uncovered());
        for &e in coverage.uncovered() {
            let edge = &mesh.edges[e];
            let k = ElementIndex::new(edge.left.element);
            let [x, y] = mesh
                .element_vertices(k)
                .iter()
                .fold([0.0, 0.0], |a, v| [a[0] + v[0] / 4.0, a[1] + v[1] / 4.0]);
            assert!(x > 5000.0 && y > 5000.0, "face {e} at ({x}, {y})");
        }
        let error = coverage.close(&mut mesh, 0.1).unwrap_err();
        assert_eq!((error.faces, error.uncovered), (24, 4));
        assert!(
            mesh.edges
                .iter()
                .all(|e| e.boundary_tag != Some(BoundaryTag::Wall))
        );
        assert_eq!(coverage.close(&mut mesh, 0.2), Ok(4));
        let walls: Vec<usize> = (0..mesh.edges.len())
            .filter(|&e| mesh.edges[e].boundary_tag == Some(BoundaryTag::Wall))
            .collect();
        assert_eq!(walls, coverage.uncovered());
    }

    /// An atlas covers the faces whose nodes all lie within the distance of
    /// one of its points: a single point at the box's centre reaches the
    /// corners (21.2 km away) only with a distance beyond that.
    #[test]
    fn atlas_coverage_follows_the_distance() {
        let projection = LocalProjection::new(63.5, 8.5);
        let (mesh, ops) = (mesh(), DGOperators2D::new(1));
        let (lat, lon) = projection.xy_to_geo(0.0, 0.0);
        let atlas = TidalAtlas {
            points: vec![AtlasPoint {
                lon,
                lat,
                depth: 50.0,
                mean: None,
                constituents: Vec::new(),
            }],
            header: Vec::new(),
        };
        let of = |d: f64| {
            BoundaryCoverage::of_atlas(&mesh, &ops, &projection, BoundaryTag::Open, &atlas, d)
        };
        assert_eq!(of(22_000.0).uncovered_fraction(), 0.0);
        // Within 16 km: only the faces with every node within 16 km of the
        // centre, the middle two of each side (nodes 15 and 15.8 km away)
        assert_eq!(of(16_000.0).uncovered().len(), 16);
        assert_eq!(of(14_000.0).uncovered_fraction(), 1.0);
    }
}

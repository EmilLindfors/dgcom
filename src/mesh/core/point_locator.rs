//! Point location on a quadrilateral mesh.
//!
//! [`PointLocator2D::locate`] finds the element holding a point and the
//! point's reference coordinates `(r, s)`, which is what sampling a DG field
//! at an arbitrary position needs (stations, particles): the nodal basis is
//! then evaluated at `(r, s)`
//! ([`DGOperators2D::interpolation_weights`](crate::operators::DGOperators2D::interpolation_weights)).
//!
//! # Inverse map
//!
//! Elements are bilinear quadrilaterals,
//!
//! ```text
//! x(r, s) = ¼[(1−r)(1−s) x₀ + (1+r)(1−s) x₁ + (1+r)(1+s) x₂ + (1−r)(1+s) x₃],
//! ```
//!
//! so `x(r, s) = p` is solved by Newton's method from the element centre
//! ([`inverse_bilinear`]). On a convex quadrilateral the map is a bijection
//! of `[−1, 1]²` onto the element with a nonsingular Jacobian, and the
//! iteration converges quadratically (exactly in one step on a
//! parallelogram, where the map is affine). A point is in the element when
//! its `(r, s)` lies in `[−1, 1]²` up to round-off.
//!
//! # Search
//!
//! A uniform bucket grid covers the mesh's bounding box with about one bucket
//! per element; each bucket lists the elements whose bounding box overlaps
//! it (compressed rows). A query tests only the elements of its bucket whose
//! bounding box contains the point, so it costs O(1) on meshes whose element
//! sizes do not vary by orders of magnitude, and a few more candidates on
//! graded ones (a large element spans several buckets).

use std::ops::Deref;
use std::sync::Arc;

use super::Mesh2D;
use crate::types::ElementIndex;

/// Round-off allowance on `[−1, 1]²` when deciding that a point is in an
/// element (points on a shared face belong to either element).
const INSIDE_TOLERANCE: f64 = 1e-9;

/// A point inside the mesh: its element and reference coordinates in
/// `[−1, 1]²`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshPoint {
    /// Element holding the point.
    pub element: ElementIndex,
    /// Reference coordinate r.
    pub r: f64,
    /// Reference coordinate s.
    pub s: f64,
}

/// Reference coordinates `(r, s)` of `p` under the bilinear map of the
/// quadrilateral `vertices` (counter-clockwise from `(−1, −1)`), by Newton's
/// method from the centre; `None` if the iteration does not converge (a
/// degenerate element, or a point far outside a non-parallelogram one). The
/// result may lie outside `[−1, 1]²`: the point is then outside the element.
pub fn inverse_bilinear(vertices: &[[f64; 2]; 4], p: [f64; 2]) -> Option<[f64; 2]> {
    let [[x0, y0], [x1, y1], [x2, y2], [x3, y3]] = *vertices;
    // x(r, s) = a + b r + c s + d r s
    let (ax, ay) = (0.25 * (x0 + x1 + x2 + x3), 0.25 * (y0 + y1 + y2 + y3));
    let (bx, by) = (0.25 * (-x0 + x1 + x2 - x3), 0.25 * (-y0 + y1 + y2 - y3));
    let (cx, cy) = (0.25 * (-x0 - x1 + x2 + x3), 0.25 * (-y0 - y1 + y2 + y3));
    let (dx, dy) = (0.25 * (x0 - x1 + x2 - x3), 0.25 * (y0 - y1 + y2 - y3));
    // Newton converges down to the round-off of x(r, s) − p, ≈ ε·|p| in space
    // and so ≈ ε·|p|/√J in (r, s): on small elements far from the origin
    // (projected coordinates) that is above any fixed tolerance, and a step
    // bouncing at round-off must count as converged.
    let noise = 16.0 * f64::EPSILON * p[0].abs().max(p[1].abs()).max(ax.abs()).max(ay.abs());
    let (mut r, mut s) = (0.0, 0.0);
    for _ in 0..50 {
        let fx = ax + bx * r + cx * s + dx * r * s - p[0];
        let fy = ay + by * r + cy * s + dy * r * s - p[1];
        // Jacobian [[∂x/∂r, ∂x/∂s], [∂y/∂r, ∂y/∂s]]
        let (xr, xs) = (bx + dx * s, cx + dx * r);
        let (yr, ys) = (by + dy * s, cy + dy * r);
        let det = xr * ys - xs * yr;
        if det == 0.0 || !det.is_finite() {
            return None;
        }
        let dr = (ys * fx - xs * fy) / det;
        let ds = (-yr * fx + xr * fy) / det;
        r -= dr;
        s -= ds;
        if dr.abs().max(ds.abs()) < 1e-14 * (1.0 + r.abs().max(s.abs())) + noise / det.abs().sqrt()
        {
            return Some([r, s]);
        }
        if r.abs().max(s.abs()) > 1e6 {
            return None;
        }
    }
    None
}

/// Finds the element holding a point (see the module docs).
#[derive(Clone)]
pub struct PointLocator2D<'a> {
    mesh: MeshRef<'a>,
    /// Lower-left corner of the bucket grid
    origin: [f64; 2],
    /// Buckets per unit length in x and y
    inv_cell: [f64; 2],
    /// Buckets in x and y
    dims: [usize; 2],
    /// Compressed rows: the elements of bucket `b` are
    /// `elements[offsets[b]..offsets[b + 1]]`
    offsets: Vec<usize>,
    elements: Vec<ElementIndex>,
}

/// Axis-aligned bounding box `[x_min, x_max, y_min, y_max]` of an element.
fn bounding_box(vertices: &[[f64; 2]; 4]) -> [f64; 4] {
    vertices.iter().fold(
        [
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ],
        |[x0, x1, y0, y1], &[x, y]| [x0.min(x), x1.max(x), y0.min(y), y1.max(y)],
    )
}

/// The mesh a locator searches: borrowed ([`PointLocator2D::new`]) or shared
/// ([`PointLocator2D::shared`]).
#[derive(Clone)]
enum MeshRef<'a> {
    Borrowed(&'a Mesh2D),
    Shared(Arc<Mesh2D>),
}

impl Deref for MeshRef<'_> {
    type Target = Mesh2D;

    fn deref(&self) -> &Mesh2D {
        match self {
            MeshRef::Borrowed(mesh) => mesh,
            MeshRef::Shared(mesh) => mesh,
        }
    }
}

impl PointLocator2D<'static> {
    /// Bucket the elements of a shared mesh: the locator holds the mesh, so it
    /// can outlive the scope that built it (e.g. live in an application's state).
    pub fn shared(mesh: Arc<Mesh2D>) -> Self {
        Self::build(MeshRef::Shared(mesh))
    }
}

impl<'a> PointLocator2D<'a> {
    /// Bucket the elements of `mesh`.
    pub fn new(mesh: &'a Mesh2D) -> Self {
        Self::build(MeshRef::Borrowed(mesh))
    }

    fn build(mesh: MeshRef<'a>) -> Self {
        let boxes: Vec<[f64; 4]> = ElementIndex::iter(mesh.n_elements)
            .map(|k| bounding_box(&mesh.element_vertices(k)))
            .collect();
        let [x0, x1, y0, y1] = boxes.iter().fold(
            [
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::INFINITY,
                f64::NEG_INFINITY,
            ],
            |[a, b, c, d], bx| [a.min(bx[0]), b.max(bx[1]), c.min(bx[2]), d.max(bx[3])],
        );
        let (width, height) = ((x1 - x0).max(0.0), (y1 - y0).max(0.0));
        // About one bucket per element, square buckets
        let n = mesh.n_elements.max(1) as f64;
        let cell = (width * height / n).sqrt();
        let cell = if cell > 0.0 {
            cell
        } else {
            width.max(height).max(1.0) / n
        };
        let buckets = |extent: f64| ((extent / cell).ceil() as usize).clamp(1, 4 * n as usize);
        let dims = [buckets(width), buckets(height)];
        let mut locator = Self {
            mesh,
            origin: [x0, y0],
            inv_cell: [
                dims[0] as f64 / width.max(cell),
                dims[1] as f64 / height.max(cell),
            ],
            dims,
            offsets: vec![0; dims[0] * dims[1] + 1],
            elements: Vec::new(),
        };
        // Count, prefix-sum, fill
        let ranges: Vec<[usize; 4]> = boxes
            .iter()
            .map(|b| {
                let (i0, j0) = locator.bucket(b[0], b[2]);
                let (i1, j1) = locator.bucket(b[1], b[3]);
                [i0, i1, j0, j1]
            })
            .collect();
        for &[i0, i1, j0, j1] in &ranges {
            for j in j0..=j1 {
                for i in i0..=i1 {
                    locator.offsets[j * dims[0] + i + 1] += 1;
                }
            }
        }
        for b in 0..dims[0] * dims[1] {
            locator.offsets[b + 1] += locator.offsets[b];
        }
        let mut next = locator.offsets.clone();
        locator.elements = vec![ElementIndex::new(0); locator.offsets[dims[0] * dims[1]]];
        for (k, &[i0, i1, j0, j1]) in ranges.iter().enumerate() {
            for j in j0..=j1 {
                for i in i0..=i1 {
                    let b = j * dims[0] + i;
                    locator.elements[next[b]] = ElementIndex::new(k);
                    next[b] += 1;
                }
            }
        }
        locator
    }

    /// The mesh being searched.
    pub fn mesh(&self) -> &Mesh2D {
        &self.mesh
    }

    /// Bucket `(i, j)` holding `(x, y)`, clamped to the grid.
    fn bucket(&self, x: f64, y: f64) -> (usize, usize) {
        let index = |v: f64, axis: usize| {
            let f = ((v - self.origin[axis]) * self.inv_cell[axis]).floor();
            (f.max(0.0) as usize).min(self.dims[axis] - 1)
        };
        (index(x, 0), index(y, 1))
    }

    /// The element holding `p` and its reference coordinates, or `None` if
    /// `p` is outside the mesh. A point on a face shared by two elements is
    /// given to either.
    pub fn locate(&self, p: [f64; 2]) -> Option<MeshPoint> {
        if self.mesh.n_elements == 0 || !p[0].is_finite() || !p[1].is_finite() {
            return None;
        }
        let (i, j) = self.bucket(p[0], p[1]);
        let b = j * self.dims[0] + i;
        self.elements[self.offsets[b]..self.offsets[b + 1]]
            .iter()
            .find_map(|&k| self.in_element(k, p))
    }

    /// The point of the mesh closest to `p` and its distance from `p`: `p`
    /// itself (distance 0) if it is inside, else the nearest point on the
    /// faces of the elements around it. `None` for an empty mesh or a
    /// non-finite `p`.
    ///
    /// The buckets are searched in square rings around `p`'s; an element not
    /// yet tested lies in buckets of a later ring, at least `R` bucket widths
    /// away after ring `R`, so the search stops once the best distance is
    /// below that.
    pub fn nearest(&self, p: [f64; 2]) -> Option<(MeshPoint, f64)> {
        if self.mesh.n_elements == 0 || !p[0].is_finite() || !p[1].is_finite() {
            return None;
        }
        if let Some(found) = self.locate(p) {
            return Some((found, 0.0));
        }
        let (ci, cj) = self.bucket(p[0], p[1]);
        let width = (1.0 / self.inv_cell[0]).min(1.0 / self.inv_cell[1]);
        let mut best: Option<(MeshPoint, f64)> = None;
        let rings = self.dims[0].max(self.dims[1]);
        for ring in 0..=rings {
            let (i0, i1) = (ci.saturating_sub(ring), (ci + ring).min(self.dims[0] - 1));
            let (j0, j1) = (cj.saturating_sub(ring), (cj + ring).min(self.dims[1] - 1));
            for j in j0..=j1 {
                for i in i0..=i1 {
                    // Only the ring itself: the inner buckets were tested
                    if i.abs_diff(ci) != ring && j.abs_diff(cj) != ring {
                        continue;
                    }
                    let b = j * self.dims[0] + i;
                    for &k in &self.elements[self.offsets[b]..self.offsets[b + 1]] {
                        let (point, distance) = self.nearest_in_element(k, p);
                        if best.is_none_or(|(_, d)| distance < d) {
                            best = Some((point, distance));
                        }
                    }
                }
            }
            if best.is_some_and(|(_, d)| d <= ring as f64 * width) {
                break;
            }
        }
        best
    }

    /// The point of element `k` closest to `p`, which is outside it, and its
    /// distance: the nearest point of its four straight faces, whose
    /// reference coordinates are those of the face's parameter.
    fn nearest_in_element(&self, k: ElementIndex, p: [f64; 2]) -> (MeshPoint, f64) {
        let v = self.mesh.element_vertices(k);
        let mut best = (
            MeshPoint {
                element: k,
                r: 0.0,
                s: 0.0,
            },
            f64::INFINITY,
        );
        for face in 0..4 {
            let (a, b) = (v[face], v[(face + 1) % 4]);
            let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
            let length2 = dx * dx + dy * dy;
            let t = if length2 > 0.0 {
                (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length2).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let distance = (a[0] + t * dx - p[0]).hypot(a[1] + t * dy - p[1]);
            if distance < best.1 {
                // Faces counter-clockwise from (−1, −1): s = −1, r = 1, s = 1, r = −1
                let u = -1.0 + 2.0 * t;
                let (r, s) = match face {
                    0 => (u, -1.0),
                    1 => (1.0, u),
                    2 => (-u, 1.0),
                    _ => (-1.0, -u),
                };
                best = (MeshPoint { element: k, r, s }, distance);
            }
        }
        best
    }

    /// `p` in element `k`, if it is there.
    pub fn in_element(&self, k: ElementIndex, p: [f64; 2]) -> Option<MeshPoint> {
        let vertices = self.mesh.element_vertices(k);
        let [x0, x1, y0, y1] = bounding_box(&vertices);
        let slack = INSIDE_TOLERANCE * (x1 - x0).max(y1 - y0);
        if p[0] < x0 - slack || p[0] > x1 + slack || p[1] < y0 - slack || p[1] > y1 + slack {
            return None;
        }
        let [r, s] = inverse_bilinear(&vertices, p)?;
        (r.abs() <= 1.0 + INSIDE_TOLERANCE && s.abs() <= 1.0 + INSIDE_TOLERANCE).then(|| {
            MeshPoint {
                element: k,
                r: r.clamp(-1.0, 1.0),
                s: s.clamp(-1.0, 1.0),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A structured mesh with its interior vertices displaced, so the
    /// elements are general (non-parallelogram) convex quadrilaterals.
    fn distorted_mesh(nx: usize, ny: usize) -> Mesh2D {
        let mut mesh = Mesh2D::uniform_rectangle(0.0, 3.0, -1.0, 1.0, nx, ny);
        let (hx, hy) = (3.0 / nx as f64, 2.0 / ny as f64);
        for v in &mut mesh.vertices {
            let [x, y] = *v;
            let interior = x > 1e-12 && x < 3.0 - 1e-12 && y > -1.0 + 1e-12 && y < 1.0 - 1e-12;
            if interior {
                v[0] += 0.2 * hx * (7.0 * x + 3.0 * y).sin();
                v[1] += 0.2 * hy * (5.0 * x - 4.0 * y).cos();
            }
        }
        mesh
    }

    /// Points mapped from known `(k, r, s)` are located back to the same
    /// element and coordinates.
    #[test]
    fn locates_mapped_points_on_a_distorted_mesh() {
        let mesh = distorted_mesh(13, 7);
        let locator = PointLocator2D::new(&mesh);
        let mut checked = 0;
        for k in ElementIndex::iter(mesh.n_elements) {
            for (r, s) in [(0.0, 0.0), (0.37, -0.81), (-0.93, 0.62), (0.99, 0.99)] {
                let p = mesh.reference_to_physical(k, r, s);
                let found = locator.locate(p).expect("inside the mesh");
                assert_eq!(found.element, k);
                assert!((found.r - r).abs() < 1e-12 && (found.s - s).abs() < 1e-12);
                checked += 1;
            }
        }
        assert_eq!(checked, 4 * 13 * 7);
    }

    /// Vertices and face points belong to one of their elements, and the
    /// coordinates map back to the point.
    /// A locator holding its mesh finds what one borrowing it finds, after the
    /// scope that built it has dropped its own handle on the mesh.
    #[test]
    fn a_shared_locator_holds_its_mesh() {
        let mesh = distorted_mesh(13, 7);
        let borrowed = PointLocator2D::new(&mesh);
        let shared = PointLocator2D::shared(std::sync::Arc::new(mesh.clone()));
        for k in ElementIndex::iter(mesh.n_elements) {
            let p = mesh.reference_to_physical(k, 0.37, -0.81);
            let (a, b) = (borrowed.locate(p).unwrap(), shared.locate(p).unwrap());
            assert_eq!((a.element, a.r, a.s), (b.element, b.r, b.s));
        }
        assert_eq!(shared.mesh().n_elements, mesh.n_elements);
    }

    #[test]
    fn locates_vertices_and_face_points() {
        let mesh = distorted_mesh(6, 5);
        let locator = PointLocator2D::new(&mesh);
        for (v, &p) in mesh.vertices.iter().enumerate() {
            let found = locator.locate(p).expect("vertex in the mesh");
            assert!(
                mesh.elements_at_vertex(v)
                    .contains(&found.element.as_usize())
            );
            let q = mesh.reference_to_physical(found.element, found.r, found.s);
            assert!((q[0] - p[0]).abs() < 1e-12 && (q[1] - p[1]).abs() < 1e-12);
        }
    }

    #[test]
    fn points_outside_the_mesh_are_not_located() {
        let mesh = distorted_mesh(6, 5);
        let locator = PointLocator2D::new(&mesh);
        for p in [
            [-0.01, 0.0],
            [3.01, 0.5],
            [1.0, 1.001],
            [1.0, -1.2],
            [f64::NAN, 0.0],
        ] {
            assert!(locator.locate(p).is_none(), "{p:?}");
        }
    }

    /// The nearest point agrees with a search over every face of every
    /// element, for points around, inside and far from a mesh with a hole,
    /// and maps to a point at the distance it reports.
    #[test]
    fn the_nearest_point_is_the_closest_of_all_faces() {
        let full = distorted_mesh(9, 6);
        let (mesh, _) = full.retain_elements(
            |k| {
                let [x, y] = full.reference_to_physical(k, 0.0, 0.0);
                (x - 1.5).hypot(y) > 0.45
            },
            crate::mesh::BoundaryTag::Wall,
        );
        let locator = PointLocator2D::new(&mesh);
        let brute = |p: [f64; 2]| {
            ElementIndex::iter(mesh.n_elements)
                .map(|k| {
                    if locator.in_element(k, p).is_some() {
                        0.0
                    } else {
                        locator.nearest_in_element(k, p).1
                    }
                })
                .fold(f64::INFINITY, f64::min)
        };
        let mut outside = 0;
        for i in 0..41 {
            for j in 0..29 {
                let p = [-1.0 + 0.125 * i as f64, -2.3 + 0.1625 * j as f64];
                let (point, distance) = locator.nearest(p).expect("a point");
                let expected = brute(p);
                assert!(
                    (distance - expected).abs() < 1e-12,
                    "{p:?}: {distance} against {expected}"
                );
                let q = mesh.reference_to_physical(point.element, point.r, point.s);
                assert!(
                    ((q[0] - p[0]).hypot(q[1] - p[1]) - distance).abs() < 1e-12,
                    "{p:?} maps to {q:?}"
                );
                outside += usize::from(distance > 0.0);
            }
        }
        assert!(outside > 300, "{outside} points outside");
        let (_, far) = locator.nearest([40.0, -30.0]).unwrap();
        assert!((far - brute([40.0, -30.0])).abs() < 1e-12);
        assert!(locator.nearest([f64::NAN, 0.0]).is_none());
    }

    /// Holes: points in elements removed by `retain_elements` are outside.
    #[test]
    fn points_in_removed_elements_are_outside() {
        let full = Mesh2D::uniform_rectangle(0.0, 4.0, 0.0, 4.0, 4, 4);
        let (mesh, kept) = full.retain_elements(
            |k| {
                let [x, y] = full.reference_to_physical(k, 0.0, 0.0);
                !(1.0..3.0).contains(&x) || !(1.0..3.0).contains(&y)
            },
            crate::mesh::BoundaryTag::Wall,
        );
        assert_eq!(kept.len(), 12);
        let locator = PointLocator2D::new(&mesh);
        assert!(locator.locate([2.0, 2.0]).is_none());
        assert!(locator.locate([1.5, 2.5]).is_none());
        let found = locator.locate([0.5, 2.5]).expect("kept element");
        let c = mesh.reference_to_physical(found.element, 0.0, 0.0);
        assert!((c[0] - 0.5).abs() < 1e-12 && (c[1] - 2.5).abs() < 1e-12);
    }

    /// Newton is exact in one step on a parallelogram (the map is affine)
    /// and converges on a strongly skewed trapezoid.
    #[test]
    fn inverse_bilinear_on_skewed_elements() {
        let parallelogram = [[0.0, 0.0], [2.0, 0.5], [2.5, 1.5], [0.5, 1.0]];
        let trapezoid = [[0.0, 0.0], [4.0, 0.0], [2.2, 1.0], [1.8, 1.0]];
        for vertices in [parallelogram, trapezoid] {
            for (r, s) in [(0.3, -0.7), (-1.0, 1.0), (0.95, 0.9), (1.5, 0.2)] {
                let n = [
                    0.25 * (1.0 - r) * (1.0 - s),
                    0.25 * (1.0 + r) * (1.0 - s),
                    0.25 * (1.0 + r) * (1.0 + s),
                    0.25 * (1.0 - r) * (1.0 + s),
                ];
                let p = [0, 1].map(|d| (0..4).map(|v| n[v] * vertices[v][d]).sum::<f64>());
                let [rr, ss] = inverse_bilinear(&vertices, p).unwrap();
                assert!(
                    (rr - r).abs() < 1e-12 && (ss - s).abs() < 1e-12,
                    "{vertices:?}"
                );
            }
        }
    }

    /// Regression: Newton's step settles at the round-off of x(r, s) − p,
    /// which on a small element far from the origin is above 1e-14 in (r, s).
    /// A fixed tolerance then ran out of iterations and reported points inside
    /// as outside: the cage centres of the fjord-farm mesh (a 20 m quad at
    /// x ≈ 6 km, found by dg-viz) and most points on UTM-scale meshes.
    #[test]
    fn inverse_bilinear_converges_at_round_off_far_from_the_origin() {
        let farm = [
            [5980.243973360533, 2996.022050984966],
            [5973.01191486553, 3014.8088120528337],
            [5952.808833700457, 3010.6876645752254],
            [5960.487946721068, 2992.0441019699315],
        ];
        let [r, s] = inverse_bilinear(&farm, [5960.0, 3000.0]).expect("converges");
        assert!(
            (r + 0.2024).abs() < 1e-3 && (s - 0.7417).abs() < 1e-3,
            "{r} {s}"
        );

        // A skewed 20 m quad at UTM northings: every point of a grid over it.
        let (x0, y0) = (512_345.678, 7_045_678.901);
        let utm =
            [[0.0, 0.0], [21.0, 1.5], [19.5, 18.0], [-1.0, 20.5]].map(|[x, y]| [x0 + x, y0 + y]);
        for i in 0..=20 {
            for j in 0..=20 {
                let (r, s) = (-1.0 + 0.1 * i as f64, -1.0 + 0.1 * j as f64);
                let n = [
                    0.25 * (1.0 - r) * (1.0 - s),
                    0.25 * (1.0 + r) * (1.0 - s),
                    0.25 * (1.0 + r) * (1.0 + s),
                    0.25 * (1.0 - r) * (1.0 + s),
                ];
                let p = [0, 1].map(|d| (0..4).map(|v| n[v] * utm[v][d]).sum::<f64>());
                let [rr, ss] = inverse_bilinear(&utm, p).expect("converges");
                assert!(
                    (rr - r).abs() < 1e-8 && (ss - s).abs() < 1e-8,
                    "({r}, {s}) → ({rr}, {ss})"
                );
            }
        }
    }
}

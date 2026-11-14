//! Sampling a nodal DG solution at an arbitrary point.
//!
//! A [`Probe2D`] fixes a point in the mesh: its element, its reference
//! coordinates and the values of the element's nodal basis there
//! ([`DGOperators2D::interpolation_weights`]), so each sample is one dot
//! product per field with the element's nodal values. This evaluates the DG
//! polynomial itself, not the nearest node: a station samples the model where
//! the instrument is, at the order of the discretisation.
//!
//! [`Probe2D::sample_swe`] evaluates the conserved variables `h`, `hu`, `hv`
//! and the bed `B`, and derives the surface `η = h + B` and the depth-averaged
//! velocity `(u, v) = (hu, hv)/h` from them: the velocity is the ratio of the
//! interpolated transport and depth (the transport-weighted mean of the
//! element's velocity), not an interpolation of nodal velocities, which
//! would weight a nearly dry node's `hu/h` as much as a deep one's.
//!
//! Velocities are in the mesh axes; rotate them to east/north with the
//! projection's local grid convergence ([`crate::io::east_axis`]).

use crate::mesh::{Bathymetry2D, Mesh2D, MeshPoint, PointLocator2D, inverse_bilinear};
use crate::operators::DGOperators2D;
use crate::solver::SWESolution2D;
use crate::types::ElementIndex;

/// A fixed sampling point (see the module docs).
#[derive(Clone, Debug)]
pub struct Probe2D {
    point: MeshPoint,
    position: [f64; 2],
    weights: Vec<f64>,
}

/// The shallow-water state at a probe.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SWEPointSample {
    /// Water depth h (m)
    pub h: f64,
    /// Transport hu (m²/s, mesh axes)
    pub hu: f64,
    /// Transport hv (m²/s, mesh axes)
    pub hv: f64,
    /// Bed elevation B (m, negative under water; 0 without bathymetry)
    pub bed: f64,
    /// Surface elevation η = h + B (m)
    pub eta: f64,
    /// Depth-averaged velocity u = hu/h (m/s, mesh axes; 0 where h ≤ h_min)
    pub u: f64,
    /// Depth-averaged velocity v = hv/h (m/s, mesh axes; 0 where h ≤ h_min)
    pub v: f64,
}

impl Probe2D {
    /// A probe at `point` of `mesh`.
    pub fn new(mesh: &Mesh2D, ops: &DGOperators2D, point: MeshPoint) -> Self {
        Self {
            position: mesh.reference_to_physical(point.element, point.r, point.s),
            weights: ops.interpolation_weights(point.r, point.s),
            point,
        }
    }

    /// A probe at the physical position `p`, or `None` if it is outside the
    /// mesh.
    pub fn at(locator: &PointLocator2D, ops: &DGOperators2D, p: [f64; 2]) -> Option<Self> {
        locator
            .locate(p)
            .map(|point| Self::new(locator.mesh(), ops, point))
    }

    /// Element holding the probe.
    pub fn element(&self) -> ElementIndex {
        self.point.element
    }

    /// Element and reference coordinates.
    pub fn point(&self) -> MeshPoint {
        self.point
    }

    /// Physical position `(x, y)`.
    pub fn position(&self) -> [f64; 2] {
        self.position
    }

    /// Nodal basis values at the probe, in node order.
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// A nodal field of the probe's element evaluated at the probe.
    #[inline]
    pub fn evaluate(&self, element_values: &[f64]) -> f64 {
        debug_assert_eq!(element_values.len(), self.weights.len());
        self.weights
            .iter()
            .zip(element_values)
            .map(|(w, v)| w * v)
            .sum()
    }

    /// A field stored element by element (`[n_elements × n_nodes]`, like
    /// the SoA solution fields) evaluated at the probe.
    #[inline]
    pub fn evaluate_field(&self, field: &[f64]) -> f64 {
        let n = self.weights.len();
        let start = self.point.element.as_usize() * n;
        self.evaluate(&field[start..start + n])
    }

    /// The shallow-water state at the probe (see the module docs);
    /// `bathymetry` `None` means a flat bed at 0.
    pub fn sample_swe(
        &self,
        q: &SWESolution2D,
        bathymetry: Option<&Bathymetry2D>,
        h_min: f64,
    ) -> SWEPointSample {
        let k = self.point.element;
        let h = self.evaluate(q.element_h(k));
        let hu = self.evaluate(q.element_hu(k));
        let hv = self.evaluate(q.element_hv(k));
        let bed = bathymetry.map_or(0.0, |b| self.evaluate(b.element(k)));
        let (u, v) = if h > h_min {
            (hu / h, hv / h)
        } else {
            (0.0, 0.0)
        };
        SWEPointSample {
            h,
            hu,
            hv,
            bed,
            eta: h + bed,
            u,
            v,
        }
    }

    /// A probe for a station at `p` in open water, with its distance from
    /// `p` (m): at `p` if every node of its element has its bed at least
    /// `min_depth` (m) below mean sea level, else at the nearest point of
    /// such an element no farther than `max_offset` (m) from `p`. `None` if
    /// there is none.
    ///
    /// # Why
    ///
    /// A tide gauge stands at the shore, often in an element that is partly
    /// land in the mesh. There `η = h + B` mixes the land nodes' bed into the
    /// surface through the basis: at rest, `h = 0` and `B > 0` at a land node,
    /// whose weight at the gauge can be negative on a P2 element, so the
    /// surface reads wrong by a fraction of the land's height (−0.31 m at
    /// Bodø's gauge, with land at +5 m). In an element that stays wet
    /// through the tide the surface is the water's.
    ///
    /// The nearest point of an element is its reference coordinates of `p`
    /// clamped to the element: exact on parallelograms, close to it on other
    /// convex quadrilaterals. The search is over every element, once per
    /// station.
    pub fn in_open_water(
        locator: &PointLocator2D,
        ops: &DGOperators2D,
        bathymetry: &Bathymetry2D,
        p: [f64; 2],
        min_depth: f64,
        max_offset: f64,
    ) -> Option<(Self, f64)> {
        let mesh = locator.mesh();
        let open = |k: ElementIndex| bathymetry.element(k).iter().all(|&b| b <= -min_depth);
        if let Some(probe) = Self::at(locator, ops, p)
            && open(probe.element())
        {
            return Some((probe, 0.0));
        }
        let (distance, point) = ElementIndex::iter(mesh.n_elements)
            .filter(|&k| open(k))
            .filter_map(|k| {
                let [r, s] = inverse_bilinear(&mesh.element_vertices(k), p)
                    .unwrap_or([0.0, 0.0])
                    .map(|c| c.clamp(-1.0, 1.0));
                let [x, y] = mesh.reference_to_physical(k, r, s);
                let distance = (x - p[0]).hypot(y - p[1]);
                (distance <= max_offset).then_some((distance, MeshPoint { element: k, r, s }))
            })
            .min_by(|a, b| a.0.total_cmp(&b.0))?;
        Some((Self::new(mesh, ops, point), distance))
    }

    /// A probe at `p` if the bed there is at least `min_depth` (m) below
    /// mean sea level, else at the nearest point where it is, no farther
    /// than `max_offset` (m) from `p`, with its distance from `p` (m). `None`
    /// if there is none.
    ///
    /// # Why
    ///
    /// [`Self::in_open_water`] moves a station into the nearest element whose
    /// every node is that deep, which is right for the surface (see there)
    /// but depends on the elements: the nearer such element is farther out on
    /// a coarser mesh, so a station's current there is sampled in deeper
    /// water (Melinka 4.4 to 14.5 m deep over meshes and orders), and differs
    /// between them by as much as the current. The depth-mean current needs
    /// no wet element, only water: `hu` and `h` are 0 at a dry node, and
    /// their ratio is the element's transport-weighted velocity. So the
    /// current is sampled where the bed polynomial itself reaches the depth,
    /// a point that converges with the bed, not with the elements.
    ///
    /// # How
    ///
    /// Each element within `max_offset` of `p` is sampled on a grid of
    /// reference points no coarser than about 10 m and `2/(4N)` in `r` and
    /// `s`. From the nearest deep grid points, nearest first, a 9 × 9 grid
    /// over a window about the nearest deep point so far, halved each time
    /// down to 1e-12 in `r` and `s`, closes in on the nearest point of the
    /// element where the bed is that deep: on the contour of the bed
    /// polynomial at `−min_depth`, wherever the elements' edges are. The
    /// search is over every element, once per station.
    pub fn nearest_at_depth(
        locator: &PointLocator2D,
        ops: &DGOperators2D,
        bathymetry: &Bathymetry2D,
        p: [f64; 2],
        min_depth: f64,
        max_offset: f64,
    ) -> Option<(Self, f64)> {
        /// The largest spacing of the search grid (m)
        const SPACING: f64 = 10.0;
        /// The most grid points along an element's side
        const MAX_POINTS: usize = 801;

        let mesh = locator.mesh();
        // The bed polynomial of element k at (r, s); orders up to 15
        let bed_at = |k: ElementIndex, r: f64, s: f64| -> f64 {
            let mut weights = [0.0; 256];
            let weights = &mut weights[..ops.n_nodes];
            ops.interpolation_weights_into(r, s, weights);
            weights
                .iter()
                .zip(bathymetry.element(k))
                .map(|(w, b)| w * b)
                .sum()
        };
        let deep = |b: f64| b <= -min_depth;
        if let Some(probe) = Self::at(locator, ops, p)
            && deep(probe.evaluate(bathymetry.element(probe.element())))
        {
            return Some((probe, 0.0));
        }
        let distance = |k: ElementIndex, r: f64, s: f64| {
            let [x, y] = mesh.reference_to_physical(k, r, s);
            (x - p[0]).hypot(y - p[1])
        };
        // Each element's nearest deep grid point: (distance, element, r, s,
        // the grid's spacing in r and s, its spacing in metres at most)
        let mut candidates = Vec::new();
        for k in ElementIndex::iter(mesh.n_elements) {
            let corners = mesh.element_vertices(k);
            let axis = |d: usize| corners.map(|c| c[d]);
            let lo = [0, 1].map(|d| axis(d).into_iter().fold(f64::INFINITY, f64::min));
            let hi = [0, 1].map(|d| axis(d).into_iter().fold(f64::NEG_INFINITY, f64::max));
            // No point of the element is nearer than its bounding box
            let gap = |d: usize| (lo[d] - p[d]).max(p[d] - hi[d]).max(0.0);
            if gap(0).hypot(gap(1)) > max_offset {
                continue;
            }
            let side = (hi[0] - lo[0]).max(hi[1] - lo[1]);
            // An odd count, so the grid holds the element's centre lines
            let m = ((side / SPACING).ceil() as usize + 1)
                .max(4 * ops.order + 1)
                .min(MAX_POINTS)
                | 1;
            let step = 2.0 / (m - 1) as f64;
            let mut nearest: Option<(f64, f64, f64)> = None;
            for i in 0..m {
                for j in 0..m {
                    let (r, s) = (-1.0 + i as f64 * step, -1.0 + j as f64 * step);
                    if !deep(bed_at(k, r, s)) {
                        continue;
                    }
                    let d = distance(k, r, s);
                    if nearest.is_none_or(|(e, _, _)| d < e) {
                        nearest = Some((d, r, s));
                    }
                }
            }
            // The grid's spacing in metres, at most
            let spacing = 0.5 * side * step;
            let narrow = (hi[0] - lo[0]).min(hi[1] - lo[1]);
            if let Some((d, r, s)) = nearest
                && d <= max_offset + 2.0 * spacing
            {
                candidates.push((d, k, r, s, spacing, narrow));
            }
        }
        candidates.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut best: Option<(f64, MeshPoint)> = None;
        for (d, k, r, s, spacing, narrow) in candidates {
            // The nearest point of an element is within a grid cell of a
            // deep grid point: no nearer than this one less two spacings
            if best.is_some_and(|(e, _)| d - 2.0 * spacing > e) {
                break;
            }
            // The grid's nearest point may be this far along the contour
            // from the element's nearest point (in r and s): where the
            // distance has grown by a spacing
            let reach = ((2.0 * d * spacing).sqrt() + 2.0 * spacing) * 2.0 / narrow;
            let (d, [r, s]) = along_the_contour(
                &|r, s| r.abs() <= 1.0 && s.abs() <= 1.0 && deep(bed_at(k, r, s)),
                &|r, s| bed_at(k, r, s),
                &|r, s| distance(k, r, s),
                (d, [r, s]),
                reach.min(2.0),
            );
            if d <= max_offset && best.is_none_or(|(e, _)| d < e) {
                best = Some((d, MeshPoint { element: k, r, s }));
            }
        }
        let (distance, point) = best?;
        Some((Self::new(mesh, ops, point), distance))
    }
}

/// The nearest point to a station along the boundary of the `deep` set near
/// `start` (its distance and its reference coordinates), for
/// [`Probe2D::nearest_at_depth`]: `start` is a deep point at distance
/// `start.0`, `bed` the bed polynomial and `distance` the distance from the
/// station, all in the element's reference coordinates.
///
/// The distance changes only to second order along the contour near its
/// nearest point, so a grid finds that point's distance well and its
/// position poorly (a 10 m grid 550 m from the station leaves it ≈ 100 m
/// along the shore). Here the contour is followed instead: the point `τ`
/// along the bed's level line at `start`, moved across it to the last deep
/// point (by bisection, so it is deep however the bed rounds on the
/// contour), and the distance minimised over `τ ∈ [−reach, reach]` by golden
/// section. The nearest of the points seen and `start` is returned.
fn along_the_contour(
    deep: &dyn Fn(f64, f64) -> bool,
    bed: &dyn Fn(f64, f64) -> f64,
    distance: &dyn Fn(f64, f64) -> f64,
    start: (f64, [f64; 2]),
    reach: f64,
) -> (f64, [f64; 2]) {
    let (_, [r0, s0]) = start;
    // Across the contour: the direction the bed rises in at the start
    let h = 1e-6;
    let g = [
        (bed(r0 + h, s0) - bed(r0 - h, s0)) / (2.0 * h),
        (bed(r0, s0 + h) - bed(r0, s0 - h)) / (2.0 * h),
    ];
    let norm = g[0].hypot(g[1]);
    if norm.is_nan() || norm == 0.0 {
        return start;
    }
    let across = [g[0] / norm, g[1] / norm];
    let along = [-across[1], across[0]];
    let on_contour = |tau: f64| -> Option<[f64; 2]> {
        let at = |l: f64| {
            [
                r0 + tau * along[0] + l * across[0],
                s0 + tau * along[1] + l * across[1],
            ]
        };
        let deep_at = |l: f64| {
            let [r, s] = at(l);
            deep(r, s)
        };
        let mut lo = if deep_at(0.0) {
            0.0
        } else if deep_at(-reach) {
            -reach
        } else {
            return None;
        };
        let mut hi = reach;
        if deep_at(hi) {
            return Some(at(hi));
        }
        for _ in 0..60 {
            let mid = 0.5 * (lo + hi);
            if deep_at(mid) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        Some(at(lo))
    };
    let mut best = start;
    let mut cost = |tau: f64| -> f64 {
        let Some([r, s]) = on_contour(tau) else {
            return f64::INFINITY;
        };
        let d = distance(r, s);
        if d < best.0 {
            best = (d, [r, s]);
        }
        d
    };
    let golden = 0.5 * (5.0_f64.sqrt() - 1.0);
    let (mut a, mut b) = (-reach, reach);
    let (mut x1, mut x2) = (b - golden * (b - a), a + golden * (b - a));
    let (mut f1, mut f2) = (cost(x1), cost(x2));
    cost(0.0);
    for _ in 0..80 {
        if f1 < f2 {
            b = x2;
            (x2, f2) = (x1, f1);
            x1 = b - golden * (b - a);
            f1 = cost(x1);
        } else {
            a = x1;
            (x1, f1) = (x2, f2);
            x2 = a + golden * (b - a);
            f2 = cost(x2);
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::GeometricFactors2D;
    use crate::solver::SWEState2D;

    /// A general quadrilateral mesh (interior vertices displaced).
    fn distorted_mesh(nx: usize, ny: usize, lx: f64, ly: f64) -> Mesh2D {
        let mut mesh = Mesh2D::uniform_rectangle(0.0, lx, 0.0, ly, nx, ny);
        let (hx, hy) = (lx / nx as f64, ly / ny as f64);
        for v in &mut mesh.vertices {
            let [x, y] = *v;
            if x > 1e-9 && x < lx - 1e-9 && y > 1e-9 && y < ly - 1e-9 {
                v[0] += 0.15 * hx * (3.1 * x / lx + 2.0 * y / ly).sin();
                v[1] += 0.15 * hy * (2.3 * x / lx - 1.7 * y / ly).cos();
            }
        }
        mesh
    }

    /// Samples of fields that are polynomials in each element's (r, s) are
    /// exact at every point; η, u and v follow from them.
    #[test]
    fn samples_are_exact_for_element_polynomials() {
        let mesh = distorted_mesh(5, 4, 1000.0, 800.0);
        let ops = DGOperators2D::new(3);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let locator = PointLocator2D::new(&mesh);
        // Degree 3 in r and s, different in every element
        let field = |k: usize, r: f64, s: f64, c: f64| {
            c + 0.1 * k as f64 + r * r * r * s - 0.5 * s * s + 0.25 * r * s * s * s
        };
        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        let bed =
            Bathymetry2D::from_function(&mesh, &ops, &geom, |x, y| -20.0 + 0.01 * x - 0.002 * y);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let (r, s) = (ops.nodes_r[i], ops.nodes_s[i]);
                let h = field(k.as_usize(), r, s, 10.0);
                q.set_state(
                    k,
                    i,
                    SWEState2D::new(
                        h,
                        field(k.as_usize(), r, s, 1.0),
                        -field(k.as_usize(), r, s, 2.0),
                    ),
                );
            }
        }
        for k in ElementIndex::iter(mesh.n_elements) {
            for (r, s) in [(0.3, -0.2), (-0.77, 0.91), (1.0, -1.0)] {
                let p = mesh.reference_to_physical(k, r, s);
                let probe = Probe2D::at(&locator, &ops, p).unwrap();
                // A vertex or face point may land in the neighbour, whose
                // polynomial differs: compare in the probe's own element
                let (kk, rr, ss) = (probe.element().as_usize(), probe.point().r, probe.point().s);
                let sample = probe.sample_swe(&q, Some(&bed), 1e-6);
                let h = field(kk, rr, ss, 10.0);
                let (hu, hv) = (field(kk, rr, ss, 1.0), -field(kk, rr, ss, 2.0));
                assert!((sample.h - h).abs() < 1e-11);
                assert!((sample.hu - hu).abs() < 1e-11);
                assert!((sample.hv - hv).abs() < 1e-11);
                assert!((sample.u - hu / h).abs() < 1e-12);
                assert!((sample.v - hv / h).abs() < 1e-12);
                // The bed is linear in x and y, which Q1 ⊂ Q3 holds exactly
                let [x, y] = probe.position();
                let b = -20.0 + 0.01 * x - 0.002 * y;
                assert!((sample.bed - b).abs() < 1e-10);
                assert!((sample.eta - (h + b)).abs() < 1e-10);
                assert_eq!(probe.evaluate_field(q.h_data()), sample.h);
            }
        }
    }

    /// Smooth fields converge at N + 1 at points between the nodes (the
    /// interpolation error of the nodal polynomial).
    #[test]
    fn point_interpolation_converges_at_order_n_plus_one() {
        let f = |x: f64, y: f64| (2.0 * x).sin() * (1.5 * y).cos() + 0.3 * x * y;
        let points: Vec<[f64; 2]> = (0..40)
            .map(|i| {
                let t = i as f64 + 0.5;
                [
                    (0.618_033_988_7 * t).fract() * 3.0,
                    (0.414_213_562_4 * t).fract() * 2.0,
                ]
            })
            .collect();
        for order in 1..=4 {
            let ops = DGOperators2D::new(order);
            let error = |n: usize| {
                let mesh = distorted_mesh(3 * n, 2 * n, 3.0, 2.0);
                let locator = PointLocator2D::new(&mesh);
                let mut field = vec![0.0; mesh.n_elements * ops.n_nodes];
                for k in ElementIndex::iter(mesh.n_elements) {
                    for i in 0..ops.n_nodes {
                        let [x, y] = mesh.reference_to_physical(k, ops.nodes_r[i], ops.nodes_s[i]);
                        field[k.as_usize() * ops.n_nodes + i] = f(x, y);
                    }
                }
                points
                    .iter()
                    .map(|&p| {
                        let probe = Probe2D::at(&locator, &ops, p).unwrap();
                        (probe.evaluate_field(&field) - f(p[0], p[1])).abs()
                    })
                    .fold(0.0, f64::max)
            };
            let (coarse, fine) = (error(4), error(8));
            let rate = (coarse / fine).log2();
            assert!(
                rate > order as f64 + 0.6,
                "P{order}: rate {rate:.2} ({coarse:.2e} → {fine:.2e})"
            );
        }
    }

    #[test]
    fn dry_points_have_zero_velocity() {
        let mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 1, 1);
        let ops = DGOperators2D::new(2);
        let mut q = SWESolution2D::new(1, ops.n_nodes);
        for i in 0..ops.n_nodes {
            q.set_state(ElementIndex::new(0), i, SWEState2D::new(0.0, 1e-9, 0.0));
        }
        let probe = Probe2D::at(&PointLocator2D::new(&mesh), &ops, [0.4, 0.6]).unwrap();
        let sample = probe.sample_swe(&q, None, 1e-3);
        assert_eq!((sample.u, sample.v), (0.0, 0.0));
        assert_eq!(sample.eta, sample.h);
    }

    /// A station in an element that is partly land reads the land's bed in
    /// its surface at rest; moved to the nearest element under water, it
    /// reads the water's.
    #[test]
    fn a_station_on_the_shore_is_moved_into_open_water() {
        // Four 1 km elements: two 10 m deep, then the shore (its east nodes
        // on land at +5 m), then land
        let mesh = Mesh2D::uniform_rectangle(0.0, 4000.0, 0.0, 1000.0, 4, 1);
        let ops = DGOperators2D::new(2);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let bed =
            Bathymetry2D::from_function(
                &mesh,
                &ops,
                &geom,
                |x, _| {
                    if x <= 2000.0 { -10.0 } else { 5.0 }
                },
            );
        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for (i, &b) in bed.element(k).iter().enumerate() {
                q.set_state(k, i, SWEState2D::new((-b).max(0.0), 0.0, 0.0));
            }
        }
        let locator = PointLocator2D::new(&mesh);
        let shore = [2250.0, 500.0];
        let at = Probe2D::at(&locator, &ops, shore).unwrap();
        let eta_at = at.sample_swe(&q, Some(&bed), 1e-3).eta;
        assert!(
            eta_at.abs() > 1.0,
            "the shore element reads its land: {eta_at}"
        );

        let (moved, offset) =
            Probe2D::in_open_water(&locator, &ops, &bed, shore, 3.0, 1000.0).unwrap();
        assert!((offset - 250.0).abs() < 1e-9, "{offset}");
        let [x, y] = moved.position();
        assert!((x - 2000.0).abs() < 1e-9 && (y - 500.0).abs() < 1e-9);
        assert_eq!(moved.element().as_usize(), 1);
        assert_eq!(moved.sample_swe(&q, Some(&bed), 1e-3).eta, 0.0);

        // In open water it stays; too far from open water there is none
        let (stays, offset) =
            Probe2D::in_open_water(&locator, &ops, &bed, [1500.0, 500.0], 3.0, 1000.0).unwrap();
        assert_eq!((offset, stays.element().as_usize()), (0.0, 1));
        assert!(Probe2D::in_open_water(&locator, &ops, &bed, shore, 3.0, 200.0).is_none());
        assert!(Probe2D::in_open_water(&locator, &ops, &bed, shore, 20.0, 1e6).is_none());
    }

    /// A station on a sloping shore: the point at a depth is where the bed
    /// reaches it, the same on every mesh and order, while the nearest
    /// element that deep everywhere moves out with the element size.
    #[test]
    fn the_point_at_a_depth_does_not_move_with_the_mesh() {
        // The bed rises 1 cm per metre to the shore at x = 2000 m: 3 m deep
        // at x = 1700 m, land beyond
        let slope = |x: f64, _y: f64| 0.01 * (x - 2000.0);
        let station = [2250.0, 500.0];
        let mut open_water = Vec::new();
        for (nx, ny, order, distorted) in [
            (4, 1, 1, false),
            (8, 2, 2, false),
            (6, 3, 3, false),
            (12, 5, 2, true),
            (16, 4, 4, false),
        ] {
            let mesh = if distorted {
                distorted_mesh(nx, ny, 4000.0, 1000.0)
            } else {
                Mesh2D::uniform_rectangle(0.0, 4000.0, 0.0, 1000.0, nx, ny)
            };
            let ops = DGOperators2D::new(order);
            let geom = GeometricFactors2D::compute(&mesh, &ops);
            let bed = Bathymetry2D::from_function(&mesh, &ops, &geom, slope);
            let locator = PointLocator2D::new(&mesh);
            let (probe, offset) =
                Probe2D::nearest_at_depth(&locator, &ops, &bed, station, 3.0, 3000.0).unwrap();
            let [x, y] = probe.position();
            let depth = -probe.evaluate(bed.element(probe.element()));
            println!(
                "{nx}×{ny} P{order}{}: ({x:.2}, {y:.2}), {offset:.3} m off, {depth:.6} m deep",
                if distorted { " distorted" } else { "" }
            );
            assert!((depth - 3.0).abs() < 1e-6, "{depth}");
            assert!((x - 1700.0).abs() < 1e-6 && (y - 500.0).abs() < 1e-3);
            assert!((offset - 550.0).abs() < 1e-6, "{offset}");
            let (level, _) =
                Probe2D::in_open_water(&locator, &ops, &bed, station, 3.0, 3000.0).unwrap();
            open_water.push(level.position()[0]);

            // In deep enough water it stays; with no water that deep within
            // reach there is none
            let (stays, offset) =
                Probe2D::nearest_at_depth(&locator, &ops, &bed, [1200.0, 300.0], 3.0, 10.0)
                    .unwrap();
            assert_eq!(offset, 0.0);
            assert!((stays.position()[0] - 1200.0).abs() < 1e-9);
            assert!(Probe2D::nearest_at_depth(&locator, &ops, &bed, station, 3.0, 500.0).is_none());
            assert!(Probe2D::nearest_at_depth(&locator, &ops, &bed, station, 25.0, 1e6).is_none());
        }
        // The whole-element rule samples 1000 m from the shore on the 1 km
        // mesh and 500 m on the 500 m one
        assert!((open_water[0] - 1000.0).abs() < 1e-9 && (open_water[1] - 1500.0).abs() < 1e-9);
    }

    /// A basin whose 3 m contour is a circle (the bed is biquadratic, which
    /// Q2 and above hold exactly, on distorted elements too): the point is
    /// where the station's ray from the centre crosses it, on any mesh.
    #[test]
    fn the_point_at_a_depth_follows_a_curved_contour() {
        let (centre, curvature) = ([2000.0, 500.0], 1e-5);
        let bowl = |x: f64, y: f64| {
            -20.0 + curvature * ((x - centre[0]).powi(2) + (y - centre[1]).powi(2))
        };
        let radius = (17.0 / curvature).sqrt();
        let station = [3800.0, 900.0];
        let (dx, dy) = (station[0] - centre[0], station[1] - centre[1]);
        let reach = dx.hypot(dy);
        let expected = [
            centre[0] + radius * dx / reach,
            centre[1] + radius * dy / reach,
        ];
        for (nx, ny, order, distorted) in [
            (4, 1, 2, false),
            (8, 2, 3, false),
            (12, 5, 2, true),
            (10, 3, 4, true),
        ] {
            let mesh = if distorted {
                distorted_mesh(nx, ny, 4000.0, 1000.0)
            } else {
                Mesh2D::uniform_rectangle(0.0, 4000.0, 0.0, 1000.0, nx, ny)
            };
            let ops = DGOperators2D::new(order);
            let geom = GeometricFactors2D::compute(&mesh, &ops);
            let bed = Bathymetry2D::from_function(&mesh, &ops, &geom, bowl);
            let locator = PointLocator2D::new(&mesh);
            let (probe, offset) =
                Probe2D::nearest_at_depth(&locator, &ops, &bed, station, 3.0, 3000.0).unwrap();
            let [x, y] = probe.position();
            println!(
                "{nx}×{ny} P{order}: ({x:.3}, {y:.3}) against ({:.3}, {:.3}), {offset:.4} m off",
                expected[0], expected[1]
            );
            assert!(
                (x - expected[0]).hypot(y - expected[1]) < 1e-3,
                "{nx}×{ny} P{order}"
            );
            assert!((offset - (reach - radius)).abs() < 1e-6, "{offset}");
            assert!(bowl(x, y) <= -3.0 + 1e-9);
        }
    }
}

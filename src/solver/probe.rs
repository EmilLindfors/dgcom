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
}

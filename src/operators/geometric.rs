//! Geometric factors for 2D quadrilateral elements, at every node.
//!
//! Each element is the bilinear (isoparametric, straight-sided) image of the
//! reference square [-1, 1]² through its four vertices,
//!
//! ```text
//! x(r, s) = ¼[(1−r)(1−s) x₀ + (1+r)(1−s) x₁ + (1+r)(1+s) x₂ + (1−r)(1+s) x₃]
//! ```
//!
//! so the Jacobian varies over a general quadrilateral and is constant only on
//! parallelograms. The factors are therefore stored per node (Hesthaven &
//! Warburton 2008, §6.6; Kopriva 2009, ch. 6):
//!
//! - the inverse metric `(r_x, r_y, s_x, s_y)` and `J = x_r y_s − x_s y_r`,
//!   so that `∂u/∂x = r_x ∂u/∂r + s_x ∂u/∂s` (chain rule) at each node;
//! - the contravariant vectors `J∇r = (y_s, −x_s)` and `J∇s = (−y_r, x_r)`,
//!   used by the conservative (divergence) form of the volume terms,
//!   `∇·F = J⁻¹[∂_r(J∇r·F) + ∂_s(J∇s·F)]`;
//! - at every face node, the outward unit normal `n` and surface Jacobian
//!   `sJ`, taken from the same metric: `sJ n = ±J∇r` on the faces `r = ±1`
//!   and `±J∇s` on `s = ±1`, so the volume and surface terms agree at the
//!   element boundary.
//!
//! # Metric identities
//!
//! The bilinear map gives `J∇r` linear in `r` only and `J∇s` linear in `s` only
//! (`∂_r(y_s) = −∂_s(−y_r)`), so the discrete metric identities
//! `D_r(J r_x) + D_s(J s_x) = 0`, `D_r(J r_y) + D_s(J s_y) = 0` hold to round-off
//! for any order `N ≥ 1`. That is what makes the conservative form preserve a
//! uniform flow (free-stream preservation) and the split forms well-balanced
//! on non-affine elements (Kopriva 2006; Wintermeyer et al. 2017).
//!
//! Curved (higher-order) element boundaries would compute the same fields from
//! the derivatives of an interpolated high-order map; the layout does not change.

use crate::mesh::Mesh2D;
use crate::operators::DGOperators2D;
use crate::types::ElementIndex;

/// Geometric factors for 2D quadrilateral elements, per node.
///
/// Volume fields are indexed `[k * n_nodes + i]` (see [`Self::node_index`]),
/// face fields `[(k * 4 + face) * n_face_nodes + fi]`, with `fi` in the face-node
/// order of [`DGOperators2D::face_nodes`] (see [`Self::face_node_index`]).
#[derive(Clone)]
pub struct GeometricFactors2D {
    /// ∂r/∂x at every node
    pub rx: Vec<f64>,

    /// ∂r/∂y at every node
    pub ry: Vec<f64>,

    /// ∂s/∂x at every node
    pub sx: Vec<f64>,

    /// ∂s/∂y at every node
    pub sy: Vec<f64>,

    /// Jacobian determinant at every node: J = x_r y_s − x_s y_r
    pub det_j: Vec<f64>,

    /// 1/J at every node
    pub det_j_inv: Vec<f64>,

    /// Diagonal of the element mass matrix: GLL weight × J at every node, so
    /// that `∫_k u ≈ Σ_i mass[k·n + i] u_i` (exact for polynomials of degree
    /// 2N − 1 times J)
    pub mass: Vec<f64>,

    /// Element areas `Σ_i mass`, one per element
    pub area: Vec<f64>,

    /// Surface Jacobian (physical / reference edge length) at every face node
    pub surface_j: Vec<f64>,

    /// Outward unit normal at every face node
    pub normals: Vec<(f64, f64)>,

    /// Number of elements
    pub n_elements: usize,

    /// Nodes per element
    pub n_nodes: usize,

    /// Nodes per face
    pub n_face_nodes: usize,

    /// Per element, densely packed: whether it is a parallelogram, and its
    /// metric and face geometry at node 0 / face node 0. That is the whole
    /// geometry of a parallelogram, which the affine fast paths read instead
    /// of the (strided) node-0 entries of the per-node fields.
    elements: Vec<ElementGeometry>,

    /// The corners of every element, counter-clockwise from `(r, s) =
    /// (−1, −1)`: all of a bilinear element's geometry in one cache line
    /// ([`Self::corners`]).
    corners: Vec<[[f64; 2]; 4]>,

    /// Whether each element is a parallelogram, one byte each
    /// ([`Self::element_is_affine`]).
    parallelograms: Vec<bool>,

    /// Whether every element is a parallelogram
    affine: bool,
}

/// The geometry of one element at node 0 and face node 0, which is all of it
/// for a parallelogram; see [`GeometricFactors2D::element_geometry`].
#[derive(Clone, Copy, Debug)]
pub struct ElementGeometry {
    /// Whether the element is a parallelogram (constant metric)
    pub affine: bool,
    /// Metric at node 0
    pub metric: AffineMetric,
    /// Outward unit normal and surface Jacobian of each face at its node 0
    pub faces: [((f64, f64), f64); 4],
}

impl GeometricFactors2D {
    /// Compute the geometric factors of every element of `mesh` at the nodes
    /// of `ops`.
    ///
    /// # Panics
    /// If the Jacobian is not positive and finite at every node (a degenerate,
    /// non-convex or clockwise element).
    pub fn compute(mesh: &Mesh2D, ops: &DGOperators2D) -> Self {
        let n_elements = mesh.n_elements;
        let n_nodes = ops.n_nodes;
        let n_face_nodes = ops.n_face_nodes;

        let n_vol = n_elements * n_nodes;
        let n_face = n_elements * 4 * n_face_nodes;
        let mut geom = Self {
            rx: Vec::with_capacity(n_vol),
            ry: Vec::with_capacity(n_vol),
            sx: Vec::with_capacity(n_vol),
            sy: Vec::with_capacity(n_vol),
            det_j: Vec::with_capacity(n_vol),
            det_j_inv: Vec::with_capacity(n_vol),
            mass: Vec::with_capacity(n_vol),
            area: Vec::with_capacity(n_elements),
            surface_j: Vec::with_capacity(n_face),
            normals: Vec::with_capacity(n_face),
            n_elements,
            n_nodes,
            n_face_nodes,
            elements: Vec::with_capacity(n_elements),
            corners: Vec::with_capacity(n_elements),
            parallelograms: Vec::with_capacity(n_elements),
            affine: true,
        };

        for k in ElementIndex::iter(n_elements) {
            let verts = mesh.element_vertices(k);
            let affine = is_parallelogram(&verts);
            geom.affine &= affine;
            geom.corners.push(verts);
            geom.parallelograms.push(affine);

            let mut area = 0.0;
            for i in 0..n_nodes {
                let m = BilinearMetric::at(&verts, ops.nodes_r[i], ops.nodes_s[i]);
                assert!(
                    m.det.is_finite() && m.det > 0.0,
                    "GeometricFactors2D requires a positive finite Jacobian; element {} has \
                     det(J) = {} at node {} (degenerate, non-convex or clockwise element)",
                    k.as_usize(),
                    m.det,
                    i
                );
                geom.rx.push(m.y_s / m.det);
                geom.ry.push(-m.x_s / m.det);
                geom.sx.push(-m.y_r / m.det);
                geom.sy.push(m.x_r / m.det);
                geom.det_j.push(m.det);
                geom.det_j_inv.push(1.0 / m.det);
                let mass = ops.weights[i] * m.det;
                geom.mass.push(mass);
                area += mass;
            }
            geom.area.push(area);

            for face in 0..4 {
                for &node in &ops.face_nodes[face] {
                    let m = BilinearMetric::at(&verts, ops.nodes_r[node], ops.nodes_s[node]);
                    // sJ n = outward contravariant vector of the face
                    let (vx, vy) = match face {
                        0 => (m.y_r, -m.x_r), // −J∇s
                        1 => (m.y_s, -m.x_s), // +J∇r
                        2 => (-m.y_r, m.x_r), // +J∇s
                        _ => (-m.y_s, m.x_s), // −J∇r
                    };
                    let s_j = (vx * vx + vy * vy).sqrt();
                    geom.surface_j.push(s_j);
                    geom.normals.push((vx / s_j, vy / s_j));
                }
            }

            let ki = k.as_usize();
            let n0 = geom.node_index(ki, 0);
            let metric = AffineMetric {
                rx: geom.rx[n0],
                ry: geom.ry[n0],
                sx: geom.sx[n0],
                sy: geom.sy[n0],
                det_j: geom.det_j[n0],
                det_j_inv: geom.det_j_inv[n0],
            };
            let faces = std::array::from_fn(|face| {
                let f0 = geom.face_node_index(ki, face, 0);
                (geom.normals[f0], geom.surface_j[f0])
            });
            geom.elements.push(ElementGeometry {
                affine,
                metric,
                faces,
            });
        }

        geom
    }

    /// Whether every element is a parallelogram, i.e. every metric term is
    /// constant within its element.
    #[inline]
    pub fn is_affine(&self) -> bool {
        self.affine
    }

    /// Whether element `k` is a parallelogram: its metric is constant (to
    /// round-off), so kernels may use the values at node 0 throughout.
    #[inline]
    pub fn element_is_affine(&self, k: usize) -> bool {
        self.parallelograms[k]
    }

    /// The corners of element `k`, counter-clockwise from `(r, s) = (−1, −1)`.
    ///
    /// Every per-node field of this struct is a function of them, computed
    /// by `BilinearMetric::at`: with `x_r, x_s, y_r, y_s` the derivatives of
    /// the bilinear map at a node and `J = x_r y_s − x_s y_r`,
    /// `(r_x, r_y, s_x, s_y) = (y_s, −x_s, −y_r, x_r) / J`; at a face node
    /// `sJ n` is `(y_r, −x_r)`, `(y_s, −x_s)`, `(−y_r, x_r)`, `(−y_s, x_s)`
    /// on faces 0–3. A kernel that evaluates these itself, operation for
    /// operation, reads 64 bytes per element instead of the per-node fields
    /// (≈ 1 kB at P2) and gets the same values to the bit. Along a face they
    /// do not depend on the node: `sJ n` is half the face's vertex
    /// difference, turned.
    #[inline]
    pub fn corners(&self, k: usize) -> &[[f64; 2]; 4] {
        &self.corners[k]
    }

    /// Node-0 geometry of element `k` from a dense per-element array: for a
    /// parallelogram ([`ElementGeometry::affine`]) its whole geometry.
    #[inline]
    pub fn element_geometry(&self, k: usize) -> &ElementGeometry {
        &self.elements[k]
    }

    /// Index of node `i` of element `k` in the volume fields.
    #[inline]
    pub fn node_index(&self, k: usize, i: usize) -> usize {
        k * self.n_nodes + i
    }

    /// Index of face node `fi` of face `face` of element `k` in the face fields.
    #[inline]
    pub fn face_node_index(&self, k: usize, face: usize, fi: usize) -> usize {
        (k * 4 + face) * self.n_face_nodes + fi
    }

    /// Jacobian determinant at node `i` of element `k`.
    #[inline]
    pub fn jacobian(&self, k: usize, i: usize) -> f64 {
        self.det_j[self.node_index(k, i)]
    }

    /// Inverse Jacobian determinant at node `i` of element `k`.
    #[inline]
    pub fn jacobian_inv(&self, k: usize, i: usize) -> f64 {
        self.det_j_inv[self.node_index(k, i)]
    }

    /// `∇r = (r_x, r_y)` at node `i` of element `k`.
    #[inline]
    pub fn grad_r(&self, k: usize, i: usize) -> (f64, f64) {
        let n = self.node_index(k, i);
        (self.rx[n], self.ry[n])
    }

    /// `∇s = (s_x, s_y)` at node `i` of element `k`.
    #[inline]
    pub fn grad_s(&self, k: usize, i: usize) -> (f64, f64) {
        let n = self.node_index(k, i);
        (self.sx[n], self.sy[n])
    }

    /// Contravariant vectors `(J∇r, J∇s)` at node `i` of element `k`.
    #[inline]
    pub fn contravariant(&self, k: usize, i: usize) -> ((f64, f64), (f64, f64)) {
        let n = self.node_index(k, i);
        let j = self.det_j[n];
        (
            (j * self.rx[n], j * self.ry[n]),
            (j * self.sx[n], j * self.sy[n]),
        )
    }

    /// Surface Jacobian at face node `fi` of face `face` of element `k`.
    #[inline]
    pub fn surface_jacobian(&self, k: usize, face: usize, fi: usize) -> f64 {
        self.surface_j[self.face_node_index(k, face, fi)]
    }

    /// Outward unit normal at face node `fi` of face `face` of element `k`.
    #[inline]
    pub fn normal(&self, k: usize, face: usize, fi: usize) -> (f64, f64) {
        self.normals[self.face_node_index(k, face, fi)]
    }

    /// Scale of the GLL lift at face node `fi`: `sJ / J` at the face node,
    /// with `J` at the volume node `node` the face node coincides with.
    #[inline]
    pub fn lift_scale(&self, k: usize, face: usize, fi: usize, node: usize) -> f64 {
        self.surface_jacobian(k, face, fi) * self.jacobian_inv(k, node)
    }

    /// Transform reference derivatives at node `i` of element `k` to physical
    /// derivatives: `(∂u/∂x, ∂u/∂y)` from `(∂u/∂r, ∂u/∂s)`.
    #[inline]
    pub fn transform_derivatives(&self, k: usize, i: usize, du_dr: f64, du_ds: f64) -> (f64, f64) {
        let n = self.node_index(k, i);
        let du_dx = self.rx[n] * du_dr + self.sx[n] * du_ds;
        let du_dy = self.ry[n] * du_dr + self.sy[n] * du_ds;
        (du_dx, du_dy)
    }

    /// Mass-weighted mean of the nodal values `u` of element `k`:
    /// `Σ_i w_i J_i u_i / Σ_i w_i J_i`, the element mean of the discrete solution.
    #[inline]
    pub fn element_mean(&self, k: usize, u: &[f64]) -> f64 {
        let mass = &self.mass[self.node_index(k, 0)..][..self.n_nodes];
        mass.iter().zip(u).map(|(m, u)| m * u).sum::<f64>() / self.area[k]
    }

    /// Quadrature mass `w_i J_i` of node `i` of element `k`.
    #[inline]
    pub fn node_mass(&self, k: usize, i: usize) -> f64 {
        self.mass[self.node_index(k, i)]
    }

    /// `∫_k u = Σ_i w_i J_i u_i` of the nodal values `u` of element `k`.
    #[inline]
    pub fn integrate_element(&self, k: usize, u: &[f64]) -> f64 {
        let mass = &self.mass[self.node_index(k, 0)..][..self.n_nodes];
        mass.iter().zip(u).map(|(m, u)| m * u).sum()
    }

    /// Size of element `k` from its area, `√area` (for affine elements
    /// `2√J`).
    #[inline]
    pub fn element_size(&self, k: usize) -> f64 {
        self.area[k].sqrt()
    }

    /// The constant metric of element `k`, for kernels that still assume
    /// parallelogram elements (the batched SIMD and Burn prototypes). Their
    /// callers must check [`Self::is_affine`] (see [`Self::assert_affine`]);
    /// on other elements it is the value at node 0.
    #[inline]
    pub fn affine_metric(&self, k: usize) -> AffineMetric {
        debug_assert!(
            self.elements[k].affine,
            "affine_metric on a non-affine element"
        );
        self.elements[k].metric
    }

    /// The constant outward normal and surface Jacobian of face `face` of
    /// element `k` (a straight face); see [`Self::affine_metric`].
    #[inline]
    pub fn affine_face(&self, k: usize, face: usize) -> ((f64, f64), f64) {
        self.elements[k].faces[face]
    }

    /// Panic unless every element is a parallelogram, naming the component
    /// that still needs constant per-element metrics.
    pub fn assert_affine(&self, component: &str) {
        assert!(
            self.affine,
            "{component} assumes parallelogram (affine) elements, but the mesh has general \
             quadrilaterals; the 2D and 3D model kernels support them, the batched SIMD and \
             Burn prototypes not yet (TODO P1.3)"
        );
    }
}

/// Constant metric of a parallelogram element; see
/// [`GeometricFactors2D::affine_metric`].
#[derive(Clone, Copy, Debug)]
pub struct AffineMetric {
    pub rx: f64,
    pub ry: f64,
    pub sx: f64,
    pub sy: f64,
    pub det_j: f64,
    pub det_j_inv: f64,
}

impl AffineMetric {
    /// Contravariant vectors `(J∇r, J∇s)`.
    #[inline]
    pub fn contravariant(&self) -> ((f64, f64), (f64, f64)) {
        let j = self.det_j;
        ((j * self.rx, j * self.ry), (j * self.sx, j * self.sy))
    }
}

/// Jacobian determinant `x_r y_s − x_s y_r` of the bilinear map of the
/// quadrilateral `verts` (counter-clockwise from `(r, s) = (−1, −1)`) at the
/// reference point `(r, s)`.
pub(crate) fn bilinear_jacobian(verts: &[[f64; 2]; 4], r: f64, s: f64) -> f64 {
    BilinearMetric::at(verts, r, s).det
}

/// Derivatives of the bilinear map at a reference point.
struct BilinearMetric {
    x_r: f64,
    x_s: f64,
    y_r: f64,
    y_s: f64,
    det: f64,
}

impl BilinearMetric {
    /// Vertices counter-clockwise from `(r, s) = (−1, −1)`.
    fn at(verts: &[[f64; 2]; 4], r: f64, s: f64) -> Self {
        let [[x0, y0], [x1, y1], [x2, y2], [x3, y3]] = *verts;
        let x_r = 0.25 * ((1.0 - s) * (x1 - x0) + (1.0 + s) * (x2 - x3));
        let y_r = 0.25 * ((1.0 - s) * (y1 - y0) + (1.0 + s) * (y2 - y3));
        let x_s = 0.25 * ((1.0 - r) * (x3 - x0) + (1.0 + r) * (x2 - x1));
        let y_s = 0.25 * ((1.0 - r) * (y3 - y0) + (1.0 + r) * (y2 - y1));
        Self {
            x_r,
            x_s,
            y_r,
            y_s,
            det: x_r * y_s - x_s * y_r,
        }
    }
}

/// Whether the quadrilateral is a parallelogram (zero bilinear term
/// `x₀ − x₁ + x₂ − x₃`), relative to its size.
fn is_parallelogram(verts: &[[f64; 2]; 4]) -> bool {
    let [[x0, y0], [x1, y1], [x2, y2], [x3, y3]] = *verts;
    let (bx, by) = (x0 - x1 + x2 - x3, y0 - y1 + y2 - y3);
    let mut scale: f64 = 0.0;
    for i in 0..4 {
        let [xa, ya] = verts[i];
        let [xb, yb] = verts[(i + 1) % 4];
        scale = scale.max((xb - xa).hypot(yb - ya));
    }
    bx.hypot(by) <= 1.0e-12 * scale
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A general (non-parallelogram) quadrilateral mesh: the interior vertices
    /// of a uniform `n × n` mesh of the unit square moved by a smooth field.
    fn distorted_mesh(n: usize, amplitude: f64) -> Mesh2D {
        let mut mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, n, n);
        for v in &mut mesh.vertices {
            let [x, y] = *v;
            let bump =
                amplitude * (std::f64::consts::PI * x).sin() * (std::f64::consts::PI * y).sin();
            *v = [x + bump * (2.0 * y - 1.0), y + 0.7 * bump * (1.0 - 2.0 * x)];
        }
        mesh
    }

    fn setup(mesh: &Mesh2D, order: usize) -> (DGOperators2D, GeometricFactors2D) {
        let ops = DGOperators2D::new(order);
        let geom = GeometricFactors2D::compute(mesh, &ops);
        (ops, geom)
    }

    #[test]
    fn test_geometric_factors_dimensions() {
        let mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 2, 2);
        let (ops, geom) = setup(&mesh, 2);

        let n_vol = mesh.n_elements * ops.n_nodes;
        assert_eq!(geom.n_elements, mesh.n_elements);
        assert_eq!(geom.rx.len(), n_vol);
        assert_eq!(geom.det_j.len(), n_vol);
        assert_eq!(geom.mass.len(), n_vol);
        assert_eq!(geom.area.len(), mesh.n_elements);
        assert_eq!(geom.surface_j.len(), mesh.n_elements * 4 * ops.n_face_nodes);
        assert_eq!(geom.normals.len(), mesh.n_elements * 4 * ops.n_face_nodes);
        assert!(geom.is_affine());
    }

    #[test]
    fn test_unit_square_jacobian() {
        // [0, 1]² from [-1, 1]²: x = (1 + r)/2, so J = 1/4, rx = sy = 2
        let mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 1, 1);
        let (ops, geom) = setup(&mesh, 3);

        for i in 0..ops.n_nodes {
            assert!((geom.jacobian(0, i) - 0.25).abs() < 1e-14);
            assert!((geom.grad_r(0, i).0 - 2.0).abs() < 1e-14);
            assert!(geom.grad_r(0, i).1.abs() < 1e-14);
            assert!(geom.grad_s(0, i).0.abs() < 1e-14);
            assert!((geom.grad_s(0, i).1 - 2.0).abs() < 1e-14);
        }
        assert!((geom.area[0] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_scaled_rectangle_jacobian() {
        // [0, 2] × [0, 1]: J = 1 · 1/2
        let mesh = Mesh2D::uniform_rectangle(0.0, 2.0, 0.0, 1.0, 1, 1);
        let (_, geom) = setup(&mesh, 2);
        assert!(geom.det_j.iter().all(|j| (j - 0.5).abs() < 1e-14));
    }

    #[test]
    fn test_surface_jacobian_and_normals_unit_square() {
        let mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 1, 1);
        let (ops, geom) = setup(&mesh, 2);
        let expected = [(0.0, -1.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)];

        for (face, &(ex, ey)) in expected.iter().enumerate() {
            for fi in 0..ops.n_face_nodes {
                // Edge length 1 over reference length 2
                assert!((geom.surface_jacobian(0, face, fi) - 0.5).abs() < 1e-14);
                let (nx, ny) = geom.normal(0, face, fi);
                assert!((nx - ex).abs() < 1e-14 && (ny - ey).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn test_transform_derivatives() {
        // Unit square: du/dx = 2 du/dr, du/dy = 2 du/ds
        let mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 1, 1);
        let (_, geom) = setup(&mesh, 1);
        let (du_dx, du_dy) = geom.transform_derivatives(0, 0, 1.0, 2.0);
        assert!((du_dx - 2.0).abs() < 1e-14);
        assert!((du_dy - 4.0).abs() < 1e-14);
    }

    #[test]
    fn test_jacobian_inverse_consistency() {
        let mesh = distorted_mesh(3, 0.1);
        let (_, geom) = setup(&mesh, 3);
        for (j, j_inv) in geom.det_j.iter().zip(&geom.det_j_inv) {
            assert!(*j > 0.0);
            assert!((j * j_inv - 1.0).abs() < 1e-14);
        }
    }

    #[test]
    fn test_skewed_parallelogram_is_affine() {
        let mut mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 1, 1);
        mesh.vertices[2] = [0.25, 1.0];
        mesh.vertices[3] = [1.25, 1.0];

        let (_, geom) = setup(&mesh, 2);
        assert!(geom.is_affine());
        assert!(geom.det_j.iter().all(|j| (j - 0.25).abs() < 1e-14));
    }

    /// A trapezoid: the Jacobian varies over the element, and the element
    /// area is its exact polygon area (J is bilinear, integrated exactly).
    #[test]
    fn test_non_affine_quad_has_varying_jacobian() {
        let mut mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 1, 1);
        // Vertices (0,0), (1,0), (1.25,1), (0,1): area = (1 + 1.25)/2
        let top_right = mesh.elements[0][2];
        mesh.vertices[top_right] = [1.25, 1.0];
        let (ops, geom) = setup(&mesh, 2);

        assert!(!geom.is_affine());
        let (j_min, j_max) = geom
            .det_j
            .iter()
            .fold((f64::MAX, f64::MIN), |(lo, hi), &j| (lo.min(j), hi.max(j)));
        assert!(j_max - j_min > 0.05, "J should vary: {j_min} .. {j_max}");
        assert!((geom.area[0] - 1.125).abs() < 1e-14);

        // The right face is the segment (1,0)–(1.25,1): length √(1 + 1/16)
        let (nx, ny) = geom.normal(0, 1, 0);
        let len = (1.0f64 + 0.0625).sqrt();
        assert!((nx - 1.0 / len).abs() < 1e-14 && (ny + 0.25 / len).abs() < 1e-14);
        for fi in 0..ops.n_face_nodes {
            assert!((geom.surface_jacobian(0, 1, fi) - 0.5 * len).abs() < 1e-14);
        }
    }

    /// The element areas of a distorted mesh sum to the domain area.
    #[test]
    fn test_distorted_mesh_areas_sum_to_domain() {
        let mesh = distorted_mesh(4, 0.15);
        for order in 1..=4 {
            let (_, geom) = setup(&mesh, order);
            assert!(!geom.is_affine());
            let total: f64 = geom.area.iter().sum();
            assert!((total - 1.0).abs() < 1e-13, "P{order}: area {total}");
        }
    }

    /// Discrete metric identities: D_r(J r_x) + D_s(J s_x) = 0 and the same for
    /// y, at every node, for every order. They make the conservative form
    /// preserve a uniform state on non-affine elements.
    #[test]
    fn test_discrete_metric_identities() {
        // The metric pair (∂/∂r, ∂/∂s) of a quantity at element k, node i
        type MetricPair = fn(&GeometricFactors2D, usize, usize) -> ((f64, f64), (f64, f64));
        let mesh = distorted_mesh(3, 0.15);
        for order in 1..=5 {
            let (ops, geom) = setup(&mesh, order);
            let n = ops.n_nodes;
            for k in 0..mesh.n_elements {
                let comp = |f: MetricPair, which: usize| {
                    let (mut ar, mut as_) = (vec![0.0; n], vec![0.0; n]);
                    for i in 0..n {
                        let (a_r, a_s) = f(&geom, k, i);
                        ar[i] = if which == 0 { a_r.0 } else { a_r.1 };
                        as_[i] = if which == 0 { a_s.0 } else { a_s.1 };
                    }
                    (ops.apply_dr(&ar), ops.apply_ds(&as_))
                };
                for which in 0..2 {
                    let (dr, ds) = comp(GeometricFactors2D::contravariant, which);
                    for i in 0..n {
                        assert!(
                            (dr[i] + ds[i]).abs() < 1e-13,
                            "P{order} element {k} node {i}: metric identity {}",
                            dr[i] + ds[i]
                        );
                    }
                }
            }
        }
    }

    /// At every face node, sJ·n equals the outward contravariant vector of
    /// the volume node there, so the volume and surface terms telescope.
    #[test]
    fn test_face_normals_match_contravariant_vectors() {
        let mesh = distorted_mesh(3, 0.15);
        let (ops, geom) = setup(&mesh, 3);
        for k in 0..mesh.n_elements {
            for face in 0..4 {
                for (fi, &node) in ops.face_nodes[face].iter().enumerate() {
                    let (a_r, a_s) = geom.contravariant(k, node);
                    let (vx, vy) = match face {
                        0 => (-a_s.0, -a_s.1),
                        1 => a_r,
                        2 => a_s,
                        _ => (-a_r.0, -a_r.1),
                    };
                    let (nx, ny) = geom.normal(k, face, fi);
                    let s_j = geom.surface_jacobian(k, face, fi);
                    assert!((s_j * nx - vx).abs() < 1e-14 && (s_j * ny - vy).abs() < 1e-14);
                }
            }
        }
    }

    /// Neighbours see the same face: equal sJ and opposite normals at the
    /// matching (reversed) face nodes.
    #[test]
    fn test_shared_faces_agree_between_neighbours() {
        let mesh = distorted_mesh(4, 0.15);
        let (ops, geom) = setup(&mesh, 3);
        let nfp = ops.n_face_nodes;
        for k in ElementIndex::iter(mesh.n_elements) {
            for face in 0..4 {
                let Some(nb) = mesh.neighbor(k, face) else {
                    continue;
                };
                for fi in 0..nfp {
                    let nfi = nfp - 1 - fi;
                    let (nx, ny) = geom.normal(k.as_usize(), face, fi);
                    let (mx, my) = geom.normal(nb.element, nb.face, nfi);
                    assert!((nx + mx).abs() < 1e-13 && (ny + my).abs() < 1e-13);
                    let sj = geom.surface_jacobian(k.as_usize(), face, fi);
                    let sj_nb = geom.surface_jacobian(nb.element, nb.face, nfi);
                    assert!((sj - sj_nb).abs() < 1e-14);
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "positive finite Jacobian")]
    fn test_non_convex_quad_is_rejected() {
        let mut mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 1, 1);
        // Pull the top-right vertex past the diagonal: a non-convex "dart"
        let top_right = mesh.elements[0][2];
        mesh.vertices[top_right] = [0.2, 0.2];
        let ops = DGOperators2D::new(2);
        let _ = GeometricFactors2D::compute(&mesh, &ops);
    }

    /// What the element kernels take from the corners instead of the
    /// per-node fields (`swe_2d_split_form_batch.rs`): a face's normal and
    /// surface Jacobian are the same at every node of it, to the bit (they
    /// are half the face's vertex difference, turned), and across a face
    /// whose two sides share their vertices the normals are opposite and the
    /// surface Jacobians equal, to the bit.
    #[test]
    fn test_face_geometry_is_constant_along_a_face_and_opposite_across_it() {
        let mesh = distorted_mesh(5, 0.04);
        for order in 1..=4 {
            let ops = DGOperators2D::new(order);
            let geom = GeometricFactors2D::compute(&mesh, &ops);
            assert!(!geom.is_affine());
            let bits = |(x, y): (f64, f64)| (x.to_bits(), y.to_bits());
            for k in 0..mesh.n_elements {
                assert_eq!(
                    *geom.corners(k),
                    mesh.element_vertices(ElementIndex::new(k))
                );
                for face in 0..4 {
                    let (normal, s_j) = geom.element_geometry(k).faces[face];
                    for fi in 0..ops.n_face_nodes {
                        assert_eq!(bits(geom.normal(k, face, fi)), bits(normal));
                        assert_eq!(geom.surface_jacobian(k, face, fi).to_bits(), s_j.to_bits());
                    }
                    let link = mesh.face_links[k][face];
                    if link.is_boundary() {
                        continue;
                    }
                    assert!(link.shares_vertices());
                    let (other, s_j_other) =
                        geom.element_geometry(link.element()).faces[link.face()];
                    assert_eq!(
                        bits(other),
                        bits((-normal.0, -normal.1)),
                        "P{order}, {k}/{face}"
                    );
                    assert_eq!(s_j_other.to_bits(), s_j.to_bits(), "P{order}, {k}/{face}");
                }
            }
        }
    }
}

//! DG operators for 2D tensor-product quadrilateral elements.
//!
//! For tensor-product elements, operators can be expressed as Kronecker products:
//! - Dr = I_s ⊗ Dr_1d (differentiate in r, identity in s)
//! - Ds = Ds_1d ⊗ I_r (identity in r, differentiate in s)
//!
//! The mass matrix remains diagonal for GLL collocation.
//!
//! Face convention (counter-clockwise):
//! - Face 0 (bottom): s = -1, r varies from -1 to +1
//! - Face 1 (right):  r = +1, s varies from -1 to +1
//! - Face 2 (top):    s = +1, r varies from +1 to -1 (reversed)
//! - Face 3 (left):   r = -1, s varies from +1 to -1 (reversed)

use crate::basis::{Vandermonde, Vandermonde2D};
use crate::polynomial::{
    gauss_lobatto_nodes, gauss_lobatto_weights, lagrange_basis, lagrange_basis_derivative,
    node_index_1d_to_2d, tensor_product_gll_nodes, tensor_product_gll_weights,
};
use faer::Mat;

/// Outward unit normal for each face of the reference quadrilateral.
pub const FACE_NORMALS: [(f64, f64); 4] = [
    (0.0, -1.0), // Face 0 (bottom): -s direction
    (1.0, 0.0),  // Face 1 (right): +r direction
    (0.0, 1.0),  // Face 2 (top): +s direction
    (-1.0, 0.0), // Face 3 (left): -r direction
];

/// All DG operators for 2D tensor-product quadrilateral elements.
#[derive(Clone)]
pub struct DGOperators2D {
    /// Polynomial order
    pub order: usize,

    /// Number of nodes per element = (order+1)²
    pub n_nodes: usize,

    /// Number of nodes per face = order+1
    pub n_face_nodes: usize,

    /// Number of nodes in 1D = order+1
    pub n_1d: usize,

    /// Reference r-coordinates of all nodes
    pub nodes_r: Vec<f64>,

    /// Reference s-coordinates of all nodes
    pub nodes_s: Vec<f64>,

    /// 2D quadrature weights (length n_nodes)
    pub weights: Vec<f64>,

    /// 1D reference nodes
    pub nodes_1d: Vec<f64>,

    /// 1D quadrature weights
    pub weights_1d: Vec<f64>,

    /// Differentiation matrix w.r.t. r: (∂u/∂r)_k = Σ_j Dr[k,j] * u_j
    /// Shape: (n_nodes, n_nodes)
    pub dr: Mat<f64>,

    /// Differentiation matrix w.r.t. s: (∂u/∂s)_k = Σ_j Ds[k,j] * u_j
    /// Shape: (n_nodes, n_nodes)
    pub ds: Mat<f64>,

    /// 1D differentiation matrix for sum-factorization
    /// Shape: (n_1d, n_1d)
    pub dr_1d: Mat<f64>,

    /// Mass matrix (diagonal for GLL)
    /// Shape: (n_nodes, n_nodes)
    pub mass: Mat<f64>,

    /// Inverse mass matrix
    /// Shape: (n_nodes, n_nodes)
    pub mass_inv: Mat<f64>,

    /// LIFT matrix for each face: maps face flux to volume contribution
    /// lift[face] has shape (n_nodes, n_face_nodes)
    pub lift: [Mat<f64>; 4],

    /// Face interpolation matrices: extract volume values at face nodes
    /// face_interp[face] has shape (n_face_nodes, n_nodes)
    pub face_interp: [Mat<f64>; 4],

    /// Indices of volume nodes on each face (for direct extraction)
    /// face_nodes[face] contains n_face_nodes indices into the volume node array
    pub face_nodes: [Vec<usize>; 4],

    /// 2D Vandermonde matrix
    pub vandermonde: Vandermonde2D,

    /// Row-major cache for Dr matrix (for SIMD kernels)
    /// Layout: dr_row_major[i * n_nodes + j] = dr[(i, j)]
    pub dr_row_major: Vec<f64>,

    /// Row-major cache for Ds matrix (for SIMD kernels)
    /// Layout: ds_row_major[i * n_nodes + j] = ds[(i, j)]
    pub ds_row_major: Vec<f64>,

    /// Row-major cache for LIFT matrices (for SIMD kernels)
    /// Layout: lift_row_major[face][i * n_face_nodes + fi] = lift[face][(i, fi)]
    pub lift_row_major: [Vec<f64>; 4],

    /// Row-major cache for the 1D differentiation matrix (split-form kernel)
    /// Layout: dr_1d_row_major[a * n_1d + c] = dr_1d[(a, c)]
    pub dr_1d_row_major: Vec<f64>,
}

impl DGOperators2D {
    /// Create DG operators for a given polynomial order.
    pub fn new(order: usize) -> Self {
        let n_1d = order + 1;
        let n_nodes = n_1d * n_1d;
        let n_face_nodes = n_1d;

        // Generate 1D and 2D nodes/weights
        let nodes_1d = gauss_lobatto_nodes(order);
        let weights_1d = gauss_lobatto_weights(order, &nodes_1d);
        let nodes_2d = tensor_product_gll_nodes(order);
        let weights = tensor_product_gll_weights(order);

        // Extract r and s coordinates
        let nodes_r: Vec<f64> = nodes_2d.iter().map(|(r, _)| *r).collect();
        let nodes_s: Vec<f64> = nodes_2d.iter().map(|(_, s)| *s).collect();

        // Build 1D Vandermonde for differentiation matrix
        let vander_1d = Vandermonde::new(order, &nodes_1d);

        // Build 1D differentiation matrix
        let dr_1d = differentiation_matrix_1d(&vander_1d);

        // Build 2D Vandermonde
        let vandermonde = Vandermonde2D::new(order);

        // Build 2D differentiation matrices: Dr = Vr * V^{-1}, Ds = Vs * V^{-1}
        let dr = differentiation_matrix_2d(&vandermonde.vr, &vandermonde.v_inv);
        let ds = differentiation_matrix_2d(&vandermonde.vs, &vandermonde.v_inv);

        // Build mass matrices (diagonal for GLL)
        let mass = mass_matrix_2d(&weights);
        let mass_inv = mass_matrix_inv_2d(&weights);

        // Build face node indices
        let face_nodes = compute_face_nodes(n_1d);

        // Build face interpolation matrices
        let face_interp = compute_face_interpolation(&face_nodes, n_nodes, n_face_nodes);

        // Build LIFT matrices
        let lift = compute_lift_matrices(&mass_inv, &face_interp, &weights_1d);

        // Build row-major caches for SIMD kernels
        let dr_row_major = mat_to_row_major(&dr);
        let ds_row_major = mat_to_row_major(&ds);
        let dr_1d_row_major = mat_to_row_major(&dr_1d);
        let lift_row_major = [
            mat_to_row_major(&lift[0]),
            mat_to_row_major(&lift[1]),
            mat_to_row_major(&lift[2]),
            mat_to_row_major(&lift[3]),
        ];

        Self {
            order,
            n_nodes,
            n_face_nodes,
            n_1d,
            nodes_r,
            nodes_s,
            weights,
            nodes_1d,
            weights_1d,
            dr,
            ds,
            dr_1d,
            mass,
            mass_inv,
            lift,
            face_interp,
            face_nodes,
            vandermonde,
            dr_row_major,
            ds_row_major,
            lift_row_major,
            dr_1d_row_major,
        }
    }

    /// Get the outward unit normal for a face.
    #[inline]
    pub fn face_normal(&self, face: usize) -> (f64, f64) {
        FACE_NORMALS[face]
    }

    /// Apply the r-differentiation matrix to nodal values.
    pub fn apply_dr(&self, u: &[f64]) -> Vec<f64> {
        assert_eq!(u.len(), self.n_nodes);
        let mut result = vec![0.0; self.n_nodes];
        for i in 0..self.n_nodes {
            for j in 0..self.n_nodes {
                result[i] += self.dr[(i, j)] * u[j];
            }
        }
        result
    }

    /// Apply the s-differentiation matrix to nodal values.
    pub fn apply_ds(&self, u: &[f64]) -> Vec<f64> {
        assert_eq!(u.len(), self.n_nodes);
        let mut result = vec![0.0; self.n_nodes];
        for i in 0..self.n_nodes {
            for j in 0..self.n_nodes {
                result[i] += self.ds[(i, j)] * u[j];
            }
        }
        result
    }

    /// Extract face values from volume values using direct indexing.
    /// Returns values at the n_face_nodes points on the specified face.
    pub fn extract_face_values(&self, u: &[f64], face: usize) -> Vec<f64> {
        assert_eq!(u.len(), self.n_nodes);
        self.face_nodes[face].iter().map(|&k| u[k]).collect()
    }

    /// Apply LIFT operator for a single face.
    /// Takes flux values at face nodes and returns volume contributions.
    pub fn apply_lift(&self, flux: &[f64], face: usize) -> Vec<f64> {
        assert_eq!(flux.len(), self.n_face_nodes);
        let mut result = vec![0.0; self.n_nodes];
        let lift = &self.lift[face];
        for i in 0..self.n_nodes {
            for j in 0..self.n_face_nodes {
                result[i] += lift[(i, j)] * flux[j];
            }
        }
        result
    }

    /// Get the 1D quadrature weight for a face node.
    #[inline]
    pub fn face_weight(&self, face_node: usize) -> f64 {
        self.weights_1d[face_node]
    }

    /// Values of the nodal basis `ℓᵢ(r) ℓⱼ(s)` at the reference point
    /// `(r, s)`, in node order (`k = j·n_1d + i`), into `out`: a nodal field
    /// `u` of the element is `Σₖ out[k] u[k]` there. Exact for every
    /// polynomial of degree ≤ N in each of r and s.
    pub fn interpolation_weights_into(&self, r: f64, s: f64, out: &mut [f64]) {
        const MAX_1D: usize = 16;
        assert!(self.n_1d <= MAX_1D, "order {} above 15", self.order);
        assert_eq!(out.len(), self.n_nodes);
        let (mut lr, mut ls) = ([0.0; MAX_1D], [0.0; MAX_1D]);
        let n = self.n_1d;
        lagrange_basis(&self.nodes_1d, r, &mut lr[..n]);
        lagrange_basis(&self.nodes_1d, s, &mut ls[..n]);
        for (row, &l_s) in out.chunks_exact_mut(n).zip(&ls[..n]) {
            for (w, &l_r) in row.iter_mut().zip(&lr[..n]) {
                *w = l_r * l_s;
            }
        }
    }

    /// Derivatives of the nodal basis along r and s at the reference point
    /// `(r, s)`, `ℓᵢ'(r) ℓⱼ(s)` into `d_r` and `ℓᵢ(r) ℓⱼ'(s)` into `d_s`, in
    /// node order: `∂u/∂r = Σₖ d_r[k] u[k]` there. Exact for every
    /// polynomial of degree ≤ N in each of r and s.
    pub fn interpolation_gradient_weights_into(
        &self,
        r: f64,
        s: f64,
        d_r: &mut [f64],
        d_s: &mut [f64],
    ) {
        const MAX_1D: usize = 16;
        assert!(self.n_1d <= MAX_1D, "order {} above 15", self.order);
        assert_eq!(d_r.len(), self.n_nodes);
        assert_eq!(d_s.len(), self.n_nodes);
        let n = self.n_1d;
        let (mut lr, mut ls) = ([0.0; MAX_1D], [0.0; MAX_1D]);
        let (mut dlr, mut dls) = ([0.0; MAX_1D], [0.0; MAX_1D]);
        lagrange_basis(&self.nodes_1d, r, &mut lr[..n]);
        lagrange_basis(&self.nodes_1d, s, &mut ls[..n]);
        lagrange_basis_derivative(&self.nodes_1d, r, &mut dlr[..n]);
        lagrange_basis_derivative(&self.nodes_1d, s, &mut dls[..n]);
        for (j, (row_r, row_s)) in d_r
            .chunks_exact_mut(n)
            .zip(d_s.chunks_exact_mut(n))
            .enumerate()
        {
            for i in 0..n {
                row_r[i] = dlr[i] * ls[j];
                row_s[i] = lr[i] * dls[j];
            }
        }
    }

    /// [`interpolation_weights_into`](Self::interpolation_weights_into) into a
    /// new vector.
    pub fn interpolation_weights(&self, r: f64, s: f64) -> Vec<f64> {
        let mut out = vec![0.0; self.n_nodes];
        self.interpolation_weights_into(r, s, &mut out);
        out
    }
}

/// Compute 1D differentiation matrix Dr = Vr * V^{-1}.
fn differentiation_matrix_1d(vander: &Vandermonde) -> Mat<f64> {
    let n = vander.order + 1;
    let mut dr = Mat::zeros(n, n);

    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += vander.vr[(i, k)] * vander.v_inv[(k, j)];
            }
            dr[(i, j)] = sum;
        }
    }

    dr
}

/// Compute 2D differentiation matrix from derivative Vandermonde and inverse Vandermonde.
fn differentiation_matrix_2d(vd: &Mat<f64>, v_inv: &Mat<f64>) -> Mat<f64> {
    let n = vd.nrows();
    let mut d = Mat::zeros(n, n);

    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += vd[(i, k)] * v_inv[(k, j)];
            }
            d[(i, j)] = sum;
        }
    }

    d
}

/// Convert a faer matrix to row-major Vec<f64> for SIMD kernels.
fn mat_to_row_major(m: &Mat<f64>) -> Vec<f64> {
    let nrows = m.nrows();
    let ncols = m.ncols();
    let mut result = vec![0.0; nrows * ncols];
    for i in 0..nrows {
        for j in 0..ncols {
            result[i * ncols + j] = m[(i, j)];
        }
    }
    result
}

/// Compute diagonal mass matrix from 2D weights.
fn mass_matrix_2d(weights: &[f64]) -> Mat<f64> {
    let n = weights.len();
    let mut m = Mat::zeros(n, n);
    for i in 0..n {
        m[(i, i)] = weights[i];
    }
    m
}

/// Compute inverse diagonal mass matrix.
fn mass_matrix_inv_2d(weights: &[f64]) -> Mat<f64> {
    let n = weights.len();
    let mut m_inv = Mat::zeros(n, n);
    for i in 0..n {
        m_inv[(i, i)] = 1.0 / weights[i];
    }
    m_inv
}

/// Compute volume node indices for each face.
///
/// Node ordering in 2D: k = j * n_1d + i (s varies slowest, r varies fastest)
///
/// Face 0 (bottom, s=-1): j=0, i=0..n_1d  -> indices [0, 1, ..., n_1d-1]
/// Face 1 (right, r=+1):  i=n_1d-1, j=0..n_1d -> indices [n_1d-1, 2*n_1d-1, ..., n_1d*n_1d-1]
/// Face 2 (top, s=+1):    j=n_1d-1, i=n_1d-1..0 (reversed) -> indices [n_1d*n_1d-1, ..., n_1d*(n_1d-1)]
/// Face 3 (left, r=-1):   i=0, j=n_1d-1..0 (reversed) -> indices [(n_1d-1)*n_1d, ..., 0]
fn compute_face_nodes(n_1d: usize) -> [Vec<usize>; 4] {
    let mut face_nodes = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];

    // Face 0 (bottom): s = -1, j = 0, r goes from -1 to +1
    for i in 0..n_1d {
        face_nodes[0].push(node_index_1d_to_2d(i, 0, n_1d));
    }

    // Face 1 (right): r = +1, i = n_1d-1, s goes from -1 to +1
    for j in 0..n_1d {
        face_nodes[1].push(node_index_1d_to_2d(n_1d - 1, j, n_1d));
    }

    // Face 2 (top): s = +1, j = n_1d-1, r goes from +1 to -1 (reversed)
    for i in (0..n_1d).rev() {
        face_nodes[2].push(node_index_1d_to_2d(i, n_1d - 1, n_1d));
    }

    // Face 3 (left): r = -1, i = 0, s goes from +1 to -1 (reversed)
    for j in (0..n_1d).rev() {
        face_nodes[3].push(node_index_1d_to_2d(0, j, n_1d));
    }

    face_nodes
}

/// Compute face interpolation matrices.
///
/// face_interp[f] has shape (n_face_nodes, n_nodes) and extracts values at face nodes.
/// For GLL nodes, this is just a permutation/extraction matrix.
fn compute_face_interpolation(
    face_nodes: &[Vec<usize>; 4],
    n_nodes: usize,
    n_face_nodes: usize,
) -> [Mat<f64>; 4] {
    let mut interp = [
        Mat::zeros(n_face_nodes, n_nodes),
        Mat::zeros(n_face_nodes, n_nodes),
        Mat::zeros(n_face_nodes, n_nodes),
        Mat::zeros(n_face_nodes, n_nodes),
    ];

    for face in 0..4 {
        for (local, &global) in face_nodes[face].iter().enumerate() {
            interp[face][(local, global)] = 1.0;
        }
    }

    interp
}

/// Compute LIFT matrices for surface contributions.
///
/// LIFT[f] = M^{-1} * E_f^T * M_f
///
/// where E_f is the face interpolation matrix and M_f is the face mass matrix.
/// For GLL with diagonal mass, this simplifies considerably.
fn compute_lift_matrices(
    mass_inv: &Mat<f64>,
    face_interp: &[Mat<f64>; 4],
    weights_1d: &[f64],
) -> [Mat<f64>; 4] {
    let n_nodes = mass_inv.nrows();
    let n_face_nodes = weights_1d.len();

    let mut lift = [
        Mat::zeros(n_nodes, n_face_nodes),
        Mat::zeros(n_nodes, n_face_nodes),
        Mat::zeros(n_nodes, n_face_nodes),
        Mat::zeros(n_nodes, n_face_nodes),
    ];

    // Face mass matrix (diagonal) = diag(weights_1d)
    // E_f^T * M_f = transpose of face_interp times face mass
    // Then multiply by M^{-1}

    for face in 0..4 {
        let e = &face_interp[face];

        // Compute E^T * M_f first (n_nodes × n_face_nodes)
        // E^T[i, j] = E[j, i], then multiply by M_f[j, j] = weights_1d[j]
        let mut et_mf = Mat::zeros(n_nodes, n_face_nodes);
        for i in 0..n_nodes {
            for j in 0..n_face_nodes {
                et_mf[(i, j)] = e[(j, i)] * weights_1d[j];
            }
        }

        // LIFT = M^{-1} * (E^T * M_f)
        // For diagonal M^{-1}, this is: LIFT[i, j] = M^{-1}[i, i] * et_mf[i, j]
        for i in 0..n_nodes {
            for j in 0..n_face_nodes {
                lift[face][(i, j)] = mass_inv[(i, i)] * et_mf[(i, j)];
            }
        }
    }

    lift
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The point weights reproduce every polynomial of degree ≤ N in r and in
    /// s (Q_N) at arbitrary points, and are the identity at the nodes.
    #[test]
    fn interpolation_weights_are_exact_on_q_n() {
        for order in 1..=5 {
            let ops = DGOperators2D::new(order);
            let p = |r: f64, s: f64| {
                (0..=order)
                    .flat_map(|a| (0..=order).map(move |b| (a, b)))
                    .map(|(a, b)| {
                        (1.0 + a as f64 - 0.3 * b as f64) * r.powi(a as i32) * s.powi(b as i32)
                    })
                    .sum::<f64>()
            };
            let values: Vec<f64> = (0..ops.n_nodes)
                .map(|k| p(ops.nodes_r[k], ops.nodes_s[k]))
                .collect();
            for k in 0..ops.n_nodes {
                let w = ops.interpolation_weights(ops.nodes_r[k], ops.nodes_s[k]);
                for (j, &wj) in w.iter().enumerate() {
                    assert_eq!(wj, if j == k { 1.0 } else { 0.0 });
                }
            }
            for (r, s) in [
                (-1.0, 0.3),
                (0.17, -0.81),
                (0.5, 0.5),
                (0.99, 1.0),
                (-0.4, -1.0),
            ] {
                let w = ops.interpolation_weights(r, s);
                let u: f64 = w.iter().zip(&values).map(|(a, b)| a * b).sum();
                assert!(
                    (u - p(r, s)).abs() < 1e-11,
                    "order {order} at ({r}, {s}): {u} vs {}",
                    p(r, s)
                );
            }
        }
    }

    /// The point gradient weights differentiate every Q_N polynomial exactly
    /// at arbitrary points, and are the rows of Dr and Ds at the nodes.
    #[test]
    fn interpolation_gradient_weights_are_exact_on_q_n() {
        for order in 1..=5 {
            let ops = DGOperators2D::new(order);
            let term = |a: usize, b: usize| 1.0 + a as f64 - 0.3 * b as f64;
            let terms = || (0..=order).flat_map(|a| (0..=order).map(move |b| (a, b)));
            let p = |r: f64, s: f64| {
                terms()
                    .map(|(a, b)| term(a, b) * r.powi(a as i32) * s.powi(b as i32))
                    .sum::<f64>()
            };
            let p_r = |r: f64, s: f64| {
                terms()
                    .filter(|&(a, _)| a > 0)
                    .map(|(a, b)| term(a, b) * a as f64 * r.powi(a as i32 - 1) * s.powi(b as i32))
                    .sum::<f64>()
            };
            let p_s = |r: f64, s: f64| {
                terms()
                    .filter(|&(_, b)| b > 0)
                    .map(|(a, b)| term(a, b) * b as f64 * r.powi(a as i32) * s.powi(b as i32 - 1))
                    .sum::<f64>()
            };
            let values: Vec<f64> = (0..ops.n_nodes)
                .map(|k| p(ops.nodes_r[k], ops.nodes_s[k]))
                .collect();
            let (mut d_r, mut d_s) = (vec![0.0; ops.n_nodes], vec![0.0; ops.n_nodes]);
            for k in 0..ops.n_nodes {
                ops.interpolation_gradient_weights_into(
                    ops.nodes_r[k],
                    ops.nodes_s[k],
                    &mut d_r,
                    &mut d_s,
                );
                for j in 0..ops.n_nodes {
                    assert!((d_r[j] - ops.dr[(k, j)]).abs() < 1e-11);
                    assert!((d_s[j] - ops.ds[(k, j)]).abs() < 1e-11);
                }
            }
            for (r, s) in [(-1.0, 0.3), (0.17, -0.81), (0.5, 0.5), (0.99, 1.0)] {
                ops.interpolation_gradient_weights_into(r, s, &mut d_r, &mut d_s);
                let dot = |w: &[f64]| w.iter().zip(&values).map(|(a, b)| a * b).sum::<f64>();
                let scale = 1.0 + p_r(r, s).abs() + p_s(r, s).abs();
                assert!(
                    (dot(&d_r) - p_r(r, s)).abs() < 1e-10 * scale,
                    "order {order} ∂r"
                );
                assert!(
                    (dot(&d_s) - p_s(r, s)).abs() < 1e-10 * scale,
                    "order {order} ∂s"
                );
            }
        }
    }

    #[test]
    fn test_operators_2d_dimensions() {
        for order in 1..=4 {
            let ops = DGOperators2D::new(order);
            let n = (order + 1) * (order + 1);
            let n_1d = order + 1;

            assert_eq!(ops.order, order);
            assert_eq!(ops.n_nodes, n);
            assert_eq!(ops.n_face_nodes, n_1d);
            assert_eq!(ops.n_1d, n_1d);
            assert_eq!(ops.dr.nrows(), n);
            assert_eq!(ops.dr.ncols(), n);
            assert_eq!(ops.ds.nrows(), n);
            assert_eq!(ops.ds.ncols(), n);
            assert_eq!(ops.mass.nrows(), n);
            assert_eq!(ops.mass_inv.nrows(), n);

            for face in 0..4 {
                assert_eq!(ops.lift[face].nrows(), n);
                assert_eq!(ops.lift[face].ncols(), n_1d);
                assert_eq!(ops.face_interp[face].nrows(), n_1d);
                assert_eq!(ops.face_interp[face].ncols(), n);
                assert_eq!(ops.face_nodes[face].len(), n_1d);
            }
        }
    }

    #[test]
    fn test_differentiation_constant() {
        let ops = DGOperators2D::new(3);
        let constant = vec![2.5; ops.n_nodes];

        let dr = ops.apply_dr(&constant);
        let ds = ops.apply_ds(&constant);

        for i in 0..ops.n_nodes {
            assert!(
                dr[i].abs() < 1e-12,
                "∂(const)/∂r should be 0, got {} at node {}",
                dr[i],
                i
            );
            assert!(
                ds[i].abs() < 1e-12,
                "∂(const)/∂s should be 0, got {} at node {}",
                ds[i],
                i
            );
        }
    }

    #[test]
    fn test_differentiation_linear() {
        let ops = DGOperators2D::new(3);

        // f(r, s) = 2*r + 3*s
        let f: Vec<f64> = ops
            .nodes_r
            .iter()
            .zip(ops.nodes_s.iter())
            .map(|(&r, &s)| 2.0 * r + 3.0 * s)
            .collect();

        let dr = ops.apply_dr(&f);
        let ds = ops.apply_ds(&f);

        for i in 0..ops.n_nodes {
            assert!(
                (dr[i] - 2.0).abs() < 1e-12,
                "∂(2r+3s)/∂r should be 2, got {} at node {}",
                dr[i],
                i
            );
            assert!(
                (ds[i] - 3.0).abs() < 1e-12,
                "∂(2r+3s)/∂s should be 3, got {} at node {}",
                ds[i],
                i
            );
        }
    }

    #[test]
    fn test_differentiation_polynomial_exactness() {
        let order = 3;
        let ops = DGOperators2D::new(order);

        // f(r, s) = r³ + 2*r*s² + s³
        // ∂f/∂r = 3*r² + 2*s²
        // ∂f/∂s = 4*r*s + 3*s²
        let f: Vec<f64> = ops
            .nodes_r
            .iter()
            .zip(ops.nodes_s.iter())
            .map(|(&r, &s)| r.powi(3) + 2.0 * r * s * s + s.powi(3))
            .collect();

        let expected_dr: Vec<f64> = ops
            .nodes_r
            .iter()
            .zip(ops.nodes_s.iter())
            .map(|(&r, &s)| 3.0 * r * r + 2.0 * s * s)
            .collect();

        let expected_ds: Vec<f64> = ops
            .nodes_r
            .iter()
            .zip(ops.nodes_s.iter())
            .map(|(&r, &s)| 4.0 * r * s + 3.0 * s * s)
            .collect();

        let dr = ops.apply_dr(&f);
        let ds = ops.apply_ds(&f);

        for i in 0..ops.n_nodes {
            assert!(
                (dr[i] - expected_dr[i]).abs() < 1e-10,
                "∂f/∂r at node {}: got {}, expected {}",
                i,
                dr[i],
                expected_dr[i]
            );
            assert!(
                (ds[i] - expected_ds[i]).abs() < 1e-10,
                "∂f/∂s at node {}: got {}, expected {}",
                i,
                ds[i],
                expected_ds[i]
            );
        }
    }

    #[test]
    fn test_mass_matrix_is_diagonal() {
        let ops = DGOperators2D::new(3);

        for i in 0..ops.n_nodes {
            for j in 0..ops.n_nodes {
                if i != j {
                    assert!(
                        ops.mass[(i, j)].abs() < 1e-14,
                        "Mass matrix should be diagonal"
                    );
                    assert!(
                        ops.mass_inv[(i, j)].abs() < 1e-14,
                        "Mass inverse should be diagonal"
                    );
                } else {
                    assert!(
                        (ops.mass[(i, i)] * ops.mass_inv[(i, i)] - 1.0).abs() < 1e-14,
                        "M * M^{{-1}} diagonal should be 1"
                    );
                }
            }
        }
    }

    #[test]
    fn test_face_nodes_count() {
        for order in 1..=4 {
            let ops = DGOperators2D::new(order);
            for face in 0..4 {
                assert_eq!(
                    ops.face_nodes[face].len(),
                    ops.n_face_nodes,
                    "Face {} should have {} nodes",
                    face,
                    ops.n_face_nodes
                );
            }
        }
    }

    #[test]
    fn test_face_nodes_are_unique() {
        let ops = DGOperators2D::new(3);

        for face in 0..4 {
            let nodes = &ops.face_nodes[face];
            for (i, &n1) in nodes.iter().enumerate() {
                for (j, &n2) in nodes.iter().enumerate() {
                    if i != j {
                        assert_ne!(n1, n2, "Face {} has duplicate node indices", face);
                    }
                }
            }
        }
    }

    #[test]
    fn test_face_extraction_polynomial() {
        let ops = DGOperators2D::new(2);

        // f(r, s) = r + s
        let f: Vec<f64> = ops
            .nodes_r
            .iter()
            .zip(ops.nodes_s.iter())
            .map(|(&r, &s)| r + s)
            .collect();

        // Face 0 (bottom): s = -1, so f = r - 1
        let face0 = ops.extract_face_values(&f, 0);
        for (i, &v) in face0.iter().enumerate() {
            let expected = ops.nodes_1d[i] - 1.0;
            assert!(
                (v - expected).abs() < 1e-14,
                "Face 0 node {}: got {}, expected {}",
                i,
                v,
                expected
            );
        }

        // Face 1 (right): r = +1, so f = 1 + s
        let face1 = ops.extract_face_values(&f, 1);
        for (i, &v) in face1.iter().enumerate() {
            let expected = 1.0 + ops.nodes_1d[i];
            assert!(
                (v - expected).abs() < 1e-14,
                "Face 1 node {}: got {}, expected {}",
                i,
                v,
                expected
            );
        }
    }

    #[test]
    fn test_lift_sparse_structure() {
        // For GLL nodes with diagonal mass, LIFT should only have non-zero
        // entries at the face nodes
        let ops = DGOperators2D::new(2);

        for face in 0..4 {
            let face_node_set: std::collections::HashSet<_> =
                ops.face_nodes[face].iter().cloned().collect();

            for i in 0..ops.n_nodes {
                let row_sum: f64 = (0..ops.n_face_nodes)
                    .map(|j| ops.lift[face][(i, j)].abs())
                    .sum();

                if face_node_set.contains(&i) {
                    // Face nodes should have non-zero LIFT entries
                    assert!(
                        row_sum > 1e-14,
                        "LIFT[{}] row {} should be non-zero (face node)",
                        face,
                        i
                    );
                } else {
                    // Non-face nodes should have zero LIFT entries
                    assert!(
                        row_sum < 1e-14,
                        "LIFT[{}] row {} should be zero (interior node)",
                        face,
                        i
                    );
                }
            }
        }
    }

    #[test]
    fn test_face_normals() {
        let ops = DGOperators2D::new(2);

        // Check normals are unit vectors
        for face in 0..4 {
            let (nx, ny) = ops.face_normal(face);
            let length = (nx * nx + ny * ny).sqrt();
            assert!(
                (length - 1.0).abs() < 1e-14,
                "Face {} normal should be unit vector",
                face
            );
        }

        // Check normals point outward
        assert_eq!(ops.face_normal(0), (0.0, -1.0)); // bottom: -y
        assert_eq!(ops.face_normal(1), (1.0, 0.0)); // right: +x
        assert_eq!(ops.face_normal(2), (0.0, 1.0)); // top: +y
        assert_eq!(ops.face_normal(3), (-1.0, 0.0)); // left: -x
    }
}

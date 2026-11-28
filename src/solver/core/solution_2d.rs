//! DG solution storage for 2D problems.
//!
//! Provides contiguous storage for nodal values on 2D quadrilateral meshes.
//! Layout is element-major: data[k * n_nodes + i] for element k, node i.

use crate::mesh::Mesh2D;
use crate::operators::{DGOperators2D, GeometricFactors2D};
use crate::types::ElementIndex;

/// Solution storage for 2D DG discretization.
///
/// Stores nodal values in a contiguous array with layout [n_elements, n_nodes].
/// For tensor-product elements with order p, n_nodes = (p+1)².
///
/// # Example
///
/// ```ignore
/// let sol = DGSolution2D::new(mesh.n_elements, ops.n_nodes);
/// sol.set_from_function(&mesh, &ops, |x, y| f64::sin(x) * f64::cos(y));
/// ```
#[derive(Clone)]
pub struct DGSolution2D {
    /// Nodal values, stored as data[k * n_nodes + i] for element k, node i
    pub data: Vec<f64>,
    /// Number of elements
    pub n_elements: usize,
    /// Number of nodes per element
    pub n_nodes: usize,
}

impl DGSolution2D {
    /// Create a new solution storage initialized to zero.
    pub fn new(n_elements: usize, n_nodes: usize) -> Self {
        Self {
            data: vec![0.0; n_elements * n_nodes],
            n_elements,
            n_nodes,
        }
    }

    /// The values of the elements `kept` (indices into this one, e.g. those
    /// [`crate::mesh::Mesh2D::retain_elements`] returns), in that order.
    pub fn select_elements(&self, kept: &[usize]) -> Self {
        Self {
            data: super::select_rows(&self.data, self.n_nodes, kept),
            n_elements: kept.len(),
            n_nodes: self.n_nodes,
        }
    }

    /// Create a solution storage from existing data.
    pub fn from_data(data: Vec<f64>, n_elements: usize, n_nodes: usize) -> Self {
        debug_assert_eq!(data.len(), n_elements * n_nodes);
        Self {
            data,
            n_elements,
            n_nodes,
        }
    }

    /// Get the value at element k, node i.
    #[inline]
    pub fn get(&self, k: usize, i: usize) -> f64 {
        self.data[k * self.n_nodes + i]
    }

    /// Set the value at element k, node i.
    #[inline]
    pub fn set(&mut self, k: usize, i: usize, value: f64) {
        self.data[k * self.n_nodes + i] = value;
    }

    /// Get the nodal values for element k as a slice.
    #[inline]
    pub fn element(&self, k: ElementIndex) -> &[f64] {
        let start = k.as_usize() * self.n_nodes;
        &self.data[start..start + self.n_nodes]
    }

    /// Get mutable nodal values for element k.
    #[inline]
    pub fn element_mut(&mut self, k: ElementIndex) -> &mut [f64] {
        let start = k.as_usize() * self.n_nodes;
        &mut self.data[start..start + self.n_nodes]
    }

    /// Set the solution from a function of physical coordinates.
    ///
    /// Evaluates f(x, y) at each physical node location.
    pub fn set_from_function<F>(&mut self, mesh: &Mesh2D, ops: &DGOperators2D, f: F)
    where
        F: Fn(f64, f64) -> f64,
    {
        let n_nodes = self.n_nodes;
        for k in ElementIndex::iter(mesh.n_elements) {
            let start = k.as_usize() * n_nodes;
            for i in 0..n_nodes {
                let (r, s) = (ops.nodes_r[i], ops.nodes_s[i]);
                let [x, y] = mesh.reference_to_physical(k, r, s);
                self.data[start + i] = f(x, y);
            }
        }
    }

    /// Set all values to a constant.
    pub fn fill(&mut self, value: f64) {
        self.data.fill(value);
    }

    /// Compute the L2 error against an exact solution.
    ///
    /// Uses the quadrature weights and Jacobian for accurate integration:
    /// ||u - u_exact||_L2 = sqrt( ∫∫ (u - u_exact)² dA )
    pub fn l2_error<F>(
        &self,
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        geom: &GeometricFactors2D,
        exact: F,
    ) -> f64
    where
        F: Fn(f64, f64) -> f64,
    {
        let mut error_sq = 0.0;

        for k in ElementIndex::iter(mesh.n_elements) {
            let u_k = self.element(k);

            for i in 0..self.n_nodes {
                let j = geom.jacobian(k.as_usize(), i);
                let (r, s) = (ops.nodes_r[i], ops.nodes_s[i]);
                let [x, y] = mesh.reference_to_physical(k, r, s);
                let u_exact = exact(x, y);
                let diff = u_k[i] - u_exact;
                error_sq += ops.weights[i] * diff * diff * j;
            }
        }

        error_sq.sqrt()
    }

    /// Compute the L∞ (maximum) error against an exact solution.
    pub fn linf_error<F>(&self, mesh: &Mesh2D, ops: &DGOperators2D, exact: F) -> f64
    where
        F: Fn(f64, f64) -> f64,
    {
        let mut max_error: f64 = 0.0;

        for k in ElementIndex::iter(mesh.n_elements) {
            let u_k = self.element(k);

            for i in 0..self.n_nodes {
                let (r, s) = (ops.nodes_r[i], ops.nodes_s[i]);
                let [x, y] = mesh.reference_to_physical(k, r, s);
                let u_exact = exact(x, y);
                let error = (u_k[i] - u_exact).abs();
                max_error = max_error.max(error);
            }
        }

        max_error
    }

    /// Scale all values by a constant.
    pub fn scale(&mut self, c: f64) {
        for v in &mut self.data {
            *v *= c;
        }
    }

    /// Add c * other to self (axpy operation).
    pub fn axpy(&mut self, c: f64, other: &DGSolution2D) {
        debug_assert_eq!(self.data.len(), other.data.len());
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a += c * *b;
        }
    }

    /// Add other to self.
    pub fn add(&mut self, other: &DGSolution2D) {
        self.axpy(1.0, other);
    }

    /// Copy from another solution.
    pub fn copy_from(&mut self, other: &DGSolution2D) {
        debug_assert_eq!(self.data.len(), other.data.len());
        self.data.copy_from_slice(&other.data);
    }

    /// Get maximum absolute value.
    pub fn max_abs(&self) -> f64 {
        self.data.iter().map(|&x| x.abs()).fold(0.0, f64::max)
    }

    /// Get minimum value.
    pub fn min(&self) -> f64 {
        self.data.iter().copied().fold(f64::INFINITY, f64::min)
    }

    /// Get maximum value.
    pub fn max(&self) -> f64 {
        self.data.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    }

    /// Compute the integral of the solution over the domain.
    ///
    /// Uses quadrature weights and Jacobian for accurate integration:
    /// ∫∫ u dA = Σ_k Σ_i w_i * u_{k,i} * J_k
    pub fn integrate(&self, ops: &DGOperators2D, geom: &GeometricFactors2D) -> f64 {
        let mut integral = 0.0;

        for k in ElementIndex::iter(self.n_elements) {
            let u_k = self.element(k);

            for (i, &w) in ops.weights.iter().enumerate() {
                let j = geom.jacobian(k.as_usize(), i);
                integral += w * u_k[i] * j;
            }
        }

        integral
    }

    /// Extract face values for a specific element and face.
    ///
    /// Returns a vector of nodal values along the face.
    pub fn extract_face_values(
        &self,
        k: ElementIndex,
        face: usize,
        ops: &DGOperators2D,
    ) -> Vec<f64> {
        let u_k = self.element(k);
        ops.face_nodes[face].iter().map(|&i| u_k[i]).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::DGOperators2D;

    fn create_test_setup() -> (Mesh2D, DGOperators2D, GeometricFactors2D) {
        let mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 2, 2);
        let ops = DGOperators2D::new(2);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        (mesh, ops, geom)
    }

    #[test]
    fn test_solution_storage() {
        let n_elem = 4;
        let n_nodes = 9; // Order 2: (2+1)² = 9
        let mut sol = DGSolution2D::new(n_elem, n_nodes);

        assert_eq!(sol.data.len(), n_elem * n_nodes);

        let k0 = ElementIndex::new(0);
        let k1 = ElementIndex::new(1);

        // Set values
        sol.element_mut(k0)[0] = 1.0;
        sol.element_mut(k0)[1] = 2.0;
        sol.element_mut(k1)[0] = 3.0;

        // Read back
        assert!((sol.element(k0)[0] - 1.0).abs() < 1e-14);
        assert!((sol.element(k0)[1] - 2.0).abs() < 1e-14);
        assert!((sol.element(k1)[0] - 3.0).abs() < 1e-14);
    }

    #[test]
    fn test_set_from_function() {
        let (mesh, ops, _geom) = create_test_setup();
        let mut sol = DGSolution2D::new(mesh.n_elements, ops.n_nodes);

        // Set u(x, y) = x + 2*y
        sol.set_from_function(&mesh, &ops, |x, y| x + 2.0 * y);

        // Verify at node 0 of element 0
        let k0 = ElementIndex::new(0);
        let [x0, y0] = mesh.reference_to_physical(k0, ops.nodes_r[0], ops.nodes_s[0]);
        assert!((sol.element(k0)[0] - (x0 + 2.0 * y0)).abs() < 1e-14);
    }

    #[test]
    fn test_axpy() {
        let n_elem = 2;
        let n_nodes = 9;

        let mut a = DGSolution2D::new(n_elem, n_nodes);
        let mut b = DGSolution2D::new(n_elem, n_nodes);

        for i in 0..a.data.len() {
            a.data[i] = 1.0;
            b.data[i] = 2.0;
        }

        a.axpy(0.5, &b); // a = a + 0.5 * b = 1 + 1 = 2

        for &v in &a.data {
            assert!((v - 2.0).abs() < 1e-14);
        }
    }

    #[test]
    fn test_scale() {
        let mut sol = DGSolution2D::new(2, 9);
        sol.data.fill(3.0);
        sol.scale(2.0);
        for &v in &sol.data {
            assert!((v - 6.0).abs() < 1e-14);
        }
    }

    #[test]
    fn test_l2_error_exact() {
        let (mesh, ops, geom) = create_test_setup();
        let mut sol = DGSolution2D::new(mesh.n_elements, ops.n_nodes);

        // Set exact solution
        sol.set_from_function(&mesh, &ops, |x, y| x * y);

        // Error against itself should be zero
        let error = sol.l2_error(&mesh, &ops, &geom, |x, y| x * y);
        assert!(error < 1e-14, "L2 error should be zero, got {}", error);
    }

    #[test]
    fn test_linf_error_exact() {
        let (mesh, ops, _geom) = create_test_setup();
        let mut sol = DGSolution2D::new(mesh.n_elements, ops.n_nodes);

        sol.set_from_function(&mesh, &ops, |x, _y| x);
        let error = sol.linf_error(&mesh, &ops, |x, _y| x);
        assert!(error < 1e-14, "Linf error should be zero, got {}", error);
    }

    #[test]
    fn test_integrate_constant() {
        let (mesh, ops, geom) = create_test_setup();
        let mut sol = DGSolution2D::new(mesh.n_elements, ops.n_nodes);

        // Constant function: integral over [0,1]×[0,1] should be the constant
        sol.fill(3.0);
        let integral = sol.integrate(&ops, &geom);
        assert!(
            (integral - 3.0).abs() < 1e-12,
            "Integral of 3 over [0,1]^2 should be 3, got {}",
            integral
        );
    }

    #[test]
    fn test_integrate_linear() {
        let (mesh, ops, geom) = create_test_setup();
        let mut sol = DGSolution2D::new(mesh.n_elements, ops.n_nodes);

        // u(x, y) = x: integral over [0,1]×[0,1] = 1/2
        sol.set_from_function(&mesh, &ops, |x, _y| x);
        let integral = sol.integrate(&ops, &geom);
        assert!(
            (integral - 0.5).abs() < 1e-12,
            "Integral of x over [0,1]^2 should be 0.5, got {}",
            integral
        );
    }

    #[test]
    fn test_max_min() {
        let mut sol = DGSolution2D::new(2, 4);
        sol.data = vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0];

        assert!((sol.max() - 7.0).abs() < 1e-14);
        assert!((sol.min() - (-8.0)).abs() < 1e-14);
        assert!((sol.max_abs() - 8.0).abs() < 1e-14);
    }

    #[test]
    fn test_copy_from() {
        let mut a = DGSolution2D::new(2, 4);
        let mut b = DGSolution2D::new(2, 4);

        for (i, v) in b.data.iter_mut().enumerate() {
            *v = i as f64;
        }

        a.copy_from(&b);

        for (i, &v) in a.data.iter().enumerate() {
            assert!((v - i as f64).abs() < 1e-14);
        }
    }

    #[test]
    fn test_extract_face_values() {
        let ops = DGOperators2D::new(2);
        let n_nodes = ops.n_nodes;
        let n_face_nodes = ops.n_face_nodes;
        let mut sol = DGSolution2D::new(1, n_nodes);
        let k0 = ElementIndex::new(0);

        // Fill with node indices
        for i in 0..n_nodes {
            sol.element_mut(k0)[i] = i as f64;
        }

        // Extract face values
        for face in 0..4 {
            let face_vals = sol.extract_face_values(k0, face, &ops);
            assert_eq!(face_vals.len(), n_face_nodes);

            // Verify values match the face node indices
            for (i, &val) in face_vals.iter().enumerate() {
                let expected = ops.face_nodes[face][i] as f64;
                assert!((val - expected).abs() < 1e-14);
            }
        }
    }
}

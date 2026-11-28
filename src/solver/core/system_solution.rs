//! Generic solution storage for systems of conservation laws.
//!
//! Provides multi-variable solution containers using Structure-of-Arrays (SoA) layout
//! for efficient SIMD operations and cache-friendly access to individual variables.
//!
//! # Storage Layout
//!
//! Each variable is stored in a separate contiguous array:
//! ```text
//! data[0]: [var0_e0_n0, var0_e0_n1, ..., var0_eK_nM]  // All values for variable 0
//! data[1]: [var1_e0_n0, var1_e0_n1, ..., var1_eK_nM]  // All values for variable 1
//! ...
//! ```
//!
//! This enables efficient per-variable operations (SIMD, batched GEMM) without
//! memory layout conversion overhead.

use crate::types::ElementIndex;
use std::array;

/// The rows `kept` of `values`, `row` values each, in that order: an
/// element-major field of a state on a submesh. Empty stays empty.
pub(crate) fn select_rows(values: &[f64], row: usize, kept: &[usize]) -> Vec<f64> {
    kept.iter()
        .flat_map(|&k| values[k * row..][..row].iter().copied())
        .collect()
}

/// Solution storage for systems of N conservation laws (1D).
///
/// Stores nodal values in interleaved layout:
/// `data[k * n_nodes * N + i * N + var]` for element k, node i, variable var.
///
/// This layout provides good cache locality when accessing all variables at a node,
/// which is the common pattern in flux computations.
pub struct SystemSolution<const N: usize> {
    /// Nodal values in interleaved layout
    pub data: Vec<f64>,
    /// Number of elements
    pub n_elements: usize,
    /// Number of nodes per element
    pub n_nodes: usize,
}

impl<const N: usize> SystemSolution<N> {
    /// Create a new solution storage initialized to zero.
    pub fn new(n_elements: usize, n_nodes: usize) -> Self {
        Self {
            data: vec![0.0; n_elements * n_nodes * N],
            n_elements,
            n_nodes,
        }
    }

    /// Create a new solution storage initialized to zero (alias for `new`).
    pub fn zeros(n_elements: usize, n_nodes: usize) -> Self {
        Self::new(n_elements, n_nodes)
    }

    /// Get the state at node i in element k.
    pub fn get(&self, k: ElementIndex, i: usize) -> [f64; N] {
        let base = (k.as_usize() * self.n_nodes + i) * N;
        let mut result = [0.0; N];
        result[..N].copy_from_slice(&self.data[base..base + N]);
        result
    }

    /// Set the state at node i in element k.
    pub fn set(&mut self, k: ElementIndex, i: usize, values: [f64; N]) {
        let base = (k.as_usize() * self.n_nodes + i) * N;
        self.data[base..base + N].copy_from_slice(&values[..N]);
    }

    /// Get a single variable at node i in element k.
    pub fn get_var(&self, k: ElementIndex, i: usize, var: usize) -> f64 {
        let base = (k.as_usize() * self.n_nodes + i) * N;
        self.data[base + var]
    }

    /// Set a single variable at node i in element k.
    pub fn set_var(&mut self, k: ElementIndex, i: usize, var: usize, value: f64) {
        let base = (k.as_usize() * self.n_nodes + i) * N;
        self.data[base + var] = value;
    }

    /// Get all nodal values for a single variable in element k.
    ///
    /// Returns a vector since we can't return a strided slice.
    pub fn element_var(&self, k: ElementIndex, var: usize) -> Vec<f64> {
        let mut result = Vec::with_capacity(self.n_nodes);
        for i in 0..self.n_nodes {
            result.push(self.get_var(k, i, var));
        }
        result
    }

    /// Scale all values by a constant.
    pub fn scale(&mut self, c: f64) {
        for v in &mut self.data {
            *v *= c;
        }
    }

    /// Add c * other to self (axpy operation).
    pub fn axpy(&mut self, c: f64, other: &Self) {
        assert_eq!(self.data.len(), other.data.len());
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a += c * *b;
        }
    }

    /// Add other to self.
    pub fn add(&mut self, other: &Self) {
        self.axpy(1.0, other);
    }

    /// Get maximum absolute value across all variables.
    pub fn max_abs(&self) -> f64 {
        self.data.iter().map(|&x| x.abs()).fold(0.0, f64::max)
    }

    /// Get maximum absolute value for a specific variable.
    pub fn max_abs_var(&self, var: usize) -> f64 {
        let mut max_val: f64 = 0.0;
        for k in ElementIndex::iter(self.n_elements) {
            for i in 0..self.n_nodes {
                max_val = max_val.max(self.get_var(k, i, var).abs());
            }
        }
        max_val
    }
}

/// Solution storage for 2D systems of N conservation laws.
///
/// Uses Structure-of-Arrays (SoA) layout for efficient SIMD operations:
/// `data[var][k * n_nodes + i]` for element k, node i, variable var.
///
/// Each variable is stored in a separate contiguous array, enabling:
/// - Efficient per-variable operations (SIMD vectorization)
/// - Batched GEMM without memory layout conversion
/// - Cache-friendly access to individual variables
pub struct SystemSolution2D<const N: usize> {
    /// Nodal values in SoA layout: data[var][k * n_nodes + i]
    pub data: [Vec<f64>; N],
    /// Number of elements
    pub n_elements: usize,
    /// Number of nodes per element
    pub n_nodes: usize,
}

impl<const N: usize> SystemSolution2D<N> {
    /// Create a new solution storage initialized to zero.
    pub fn new(n_elements: usize, n_nodes: usize) -> Self {
        let total = n_elements * n_nodes;
        Self {
            data: array::from_fn(|_| vec![0.0; total]),
            n_elements,
            n_nodes,
        }
    }

    /// The state of the elements `kept` (indices into this one, e.g. those
    /// [`crate::mesh::Mesh2D::retain_elements`] returns), in that order.
    pub fn select_elements(&self, kept: &[usize]) -> Self {
        let n = self.n_nodes;
        Self {
            data: array::from_fn(|v| select_rows(&self.data[v], n, kept)),
            n_elements: kept.len(),
            n_nodes: n,
        }
    }

    /// Create a new solution storage initialized to zero (alias for `new`).
    pub fn zeros(n_elements: usize, n_nodes: usize) -> Self {
        Self::new(n_elements, n_nodes)
    }

    /// Create a solution from SoA data arrays.
    ///
    /// Each array in `data` should have `n_elements * n_nodes` values for one variable.
    pub fn from_soa_data(data: [Vec<f64>; N], n_elements: usize, n_nodes: usize) -> Self {
        let expected = n_elements * n_nodes;
        for (var, arr) in data.iter().enumerate() {
            debug_assert_eq!(
                arr.len(),
                expected,
                "Variable {} size mismatch: expected {}, got {}",
                var,
                expected,
                arr.len()
            );
        }
        Self {
            data,
            n_elements,
            n_nodes,
        }
    }

    /// Get the state at node i in element k.
    #[inline(always)]
    pub fn get(&self, k: ElementIndex, i: usize) -> [f64; N] {
        let idx = k.as_usize() * self.n_nodes + i;
        array::from_fn(|var| self.data[var][idx])
    }

    /// Set the state at node i in element k.
    #[inline(always)]
    pub fn set(&mut self, k: ElementIndex, i: usize, values: [f64; N]) {
        let idx = k.as_usize() * self.n_nodes + i;
        for var in 0..N {
            self.data[var][idx] = values[var];
        }
    }

    /// Get a single variable at node i in element k.
    #[inline(always)]
    pub fn get_var(&self, k: ElementIndex, i: usize, var: usize) -> f64 {
        let idx = k.as_usize() * self.n_nodes + i;
        self.data[var][idx]
    }

    /// Set a single variable at node i in element k.
    #[inline(always)]
    pub fn set_var(&mut self, k: ElementIndex, i: usize, var: usize, value: f64) {
        let idx = k.as_usize() * self.n_nodes + i;
        self.data[var][idx] = value;
    }

    /// Get all nodal values for a single variable in element k.
    ///
    /// Returns a slice to the contiguous data for this element's variable.
    /// This is O(1) with SoA storage - no allocation needed.
    #[inline(always)]
    pub fn element_var(&self, k: ElementIndex, var: usize) -> &[f64] {
        let start = k.as_usize() * self.n_nodes;
        &self.data[var][start..start + self.n_nodes]
    }

    /// Get mutable slice for a single variable in element k.
    #[inline(always)]
    pub fn element_var_mut(&mut self, k: ElementIndex, var: usize) -> &mut [f64] {
        let start = k.as_usize() * self.n_nodes;
        &mut self.data[var][start..start + self.n_nodes]
    }

    /// Get the entire array for a variable (all elements).
    #[inline(always)]
    pub fn var_data(&self, var: usize) -> &[f64] {
        &self.data[var]
    }

    /// Get mutable access to the entire array for a variable.
    #[inline(always)]
    pub fn var_data_mut(&mut self, var: usize) -> &mut [f64] {
        &mut self.data[var]
    }

    /// Scale all values by a constant.
    pub fn scale(&mut self, c: f64) {
        for var_data in &mut self.data {
            for v in var_data {
                *v *= c;
            }
        }
    }

    /// Add c * other to self (axpy operation).
    pub fn axpy(&mut self, c: f64, other: &Self) {
        debug_assert_eq!(self.n_elements, other.n_elements);
        debug_assert_eq!(self.n_nodes, other.n_nodes);
        for var in 0..N {
            for (a, b) in self.data[var].iter_mut().zip(other.data[var].iter()) {
                *a += c * *b;
            }
        }
    }

    /// Add other to self.
    pub fn add(&mut self, other: &Self) {
        self.axpy(1.0, other);
    }

    /// Get maximum absolute value across all variables.
    pub fn max_abs(&self) -> f64 {
        let mut max_val: f64 = 0.0;
        for var_data in &self.data {
            for &v in var_data {
                max_val = max_val.max(v.abs());
            }
        }
        max_val
    }

    /// Get maximum absolute value for a specific variable.
    pub fn max_abs_var(&self, var: usize) -> f64 {
        self.data[var].iter().map(|&x| x.abs()).fold(0.0, f64::max)
    }

    /// Copy from another solution.
    pub fn copy_from(&mut self, other: &Self) {
        debug_assert_eq!(self.n_elements, other.n_elements);
        debug_assert_eq!(self.n_nodes, other.n_nodes);
        for var in 0..N {
            self.data[var].copy_from_slice(&other.data[var]);
        }
    }

    /// Fill all values with a constant.
    pub fn fill(&mut self, value: f64) {
        for var_data in &mut self.data {
            var_data.fill(value);
        }
    }

    /// Get total number of values per variable.
    #[inline(always)]
    pub fn total_nodes(&self) -> usize {
        self.n_elements * self.n_nodes
    }
}

// Manual `Clone` so that `clone_from` reuses the buffers: the time
// integrators copy the state into persistent stage storage every stage.
impl<const N: usize> Clone for SystemSolution<N> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            n_elements: self.n_elements,
            n_nodes: self.n_nodes,
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.data.clone_from(&source.data);
        self.n_elements = source.n_elements;
        self.n_nodes = source.n_nodes;
    }
}

impl<const N: usize> Clone for SystemSolution2D<N> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            n_elements: self.n_elements,
            n_nodes: self.n_nodes,
        }
    }

    fn clone_from(&mut self, source: &Self) {
        for (dst, src) in self.data.iter_mut().zip(&source.data) {
            dst.clone_from(src);
        }
        self.n_elements = source.n_elements;
        self.n_nodes = source.n_nodes;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_solution_basic() {
        let n_elem = 4;
        let n_nodes = 3;
        let mut sol: SystemSolution<2> = SystemSolution::new(n_elem, n_nodes);

        assert_eq!(sol.data.len(), n_elem * n_nodes * 2);

        // Set and get values
        sol.set(ElementIndex::new(0), 0, [1.0, 2.0]);
        let vals = sol.get(ElementIndex::new(0), 0);
        assert!((vals[0] - 1.0).abs() < 1e-14);
        assert!((vals[1] - 2.0).abs() < 1e-14);
    }

    #[test]
    fn test_system_solution_axpy() {
        let n_elem = 2;
        let n_nodes = 3;

        let mut a: SystemSolution<2> = SystemSolution::new(n_elem, n_nodes);
        let mut b: SystemSolution<2> = SystemSolution::new(n_elem, n_nodes);

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
    fn test_system_solution_element_var() {
        let n_elem = 2;
        let n_nodes = 3;
        let mut sol: SystemSolution<2> = SystemSolution::new(n_elem, n_nodes);

        let k0 = ElementIndex::new(0);

        // Set some values
        for i in 0..n_nodes {
            sol.set(k0, i, [i as f64, (i + 10) as f64]);
        }

        // Get first variable values for element 0
        let var0_vals = sol.element_var(k0, 0);
        assert_eq!(var0_vals.len(), n_nodes);
        for i in 0..n_nodes {
            assert!((var0_vals[i] - i as f64).abs() < 1e-14);
        }

        // Get second variable values for element 0
        let var1_vals = sol.element_var(k0, 1);
        for i in 0..n_nodes {
            assert!((var1_vals[i] - (i + 10) as f64).abs() < 1e-14);
        }
    }

    #[test]
    fn test_system_solution_2d_basic() {
        let n_elem = 4;
        let n_nodes = 9; // (2+1)²
        let mut sol: SystemSolution2D<3> = SystemSolution2D::new(n_elem, n_nodes);

        // SoA layout: 3 arrays of n_elem * n_nodes each
        assert_eq!(sol.data.len(), 3);
        assert_eq!(sol.data[0].len(), n_elem * n_nodes);
        assert_eq!(sol.data[1].len(), n_elem * n_nodes);
        assert_eq!(sol.data[2].len(), n_elem * n_nodes);

        // Set and get values
        let k0 = ElementIndex::new(0);
        sol.set(k0, 0, [1.0, 2.0, 3.0]);
        let vals = sol.get(k0, 0);
        assert!((vals[0] - 1.0).abs() < 1e-14);
        assert!((vals[1] - 2.0).abs() < 1e-14);
        assert!((vals[2] - 3.0).abs() < 1e-14);
    }

    #[test]
    fn test_system_solution_2d_axpy() {
        let n_elem = 2;
        let n_nodes = 4;

        let mut a: SystemSolution2D<3> = SystemSolution2D::new(n_elem, n_nodes);
        let mut b: SystemSolution2D<3> = SystemSolution2D::new(n_elem, n_nodes);

        // Fill all arrays with SoA layout
        for var in 0..3 {
            for i in 0..a.data[var].len() {
                a.data[var][i] = 1.0;
                b.data[var][i] = 2.0;
            }
        }

        a.axpy(0.5, &b); // a = a + 0.5 * b = 1 + 1 = 2

        for var in 0..3 {
            for &v in &a.data[var] {
                assert!((v - 2.0).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn test_system_solution_2d_element_var() {
        let n_elem = 2;
        let n_nodes = 3;
        let mut sol: SystemSolution2D<3> = SystemSolution2D::new(n_elem, n_nodes);

        let k0 = ElementIndex::new(0);

        // Set some values
        for i in 0..n_nodes {
            sol.set(k0, i, [i as f64, (i + 10) as f64, (i + 20) as f64]);
        }

        // Get first variable values for element 0 (now returns slice)
        let var0_vals = sol.element_var(k0, 0);
        assert_eq!(var0_vals.len(), n_nodes);
        for i in 0..n_nodes {
            assert!((var0_vals[i] - i as f64).abs() < 1e-14);
        }

        // Get second variable values for element 0
        let var1_vals = sol.element_var(k0, 1);
        for i in 0..n_nodes {
            assert!((var1_vals[i] - (i + 10) as f64).abs() < 1e-14);
        }

        // Get third variable values for element 0
        let var2_vals = sol.element_var(k0, 2);
        for i in 0..n_nodes {
            assert!((var2_vals[i] - (i + 20) as f64).abs() < 1e-14);
        }
    }

    #[test]
    fn test_system_solution_2d_var_data() {
        let n_elem = 2;
        let n_nodes = 3;
        let mut sol: SystemSolution2D<3> = SystemSolution2D::new(n_elem, n_nodes);

        // Fill variable 0 with sequential values
        let var0 = sol.var_data_mut(0);
        for (i, v) in var0.iter_mut().enumerate() {
            *v = i as f64;
        }

        // Verify via var_data
        let var0_read = sol.var_data(0);
        assert_eq!(var0_read.len(), n_elem * n_nodes);
        for i in 0..var0_read.len() {
            assert!((var0_read[i] - i as f64).abs() < 1e-14);
        }
    }
}

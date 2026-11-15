//! 2D Shallow water state types.
//!
//! For 2D shallow water equations, we have state (h, hu, hv) where:
//! - h = water depth
//! - hu = x-momentum (h * u)
//! - hv = y-momentum (h * v)

use std::ops::{Add, Mul, Neg, Sub};

use crate::mesh::Mesh2D;
use crate::operators::{DGOperators2D, GeometricFactors2D};
use crate::types::{Depth, ElementIndex, Lane};

use super::super::core::SystemSolution2D;

/// 2D Shallow water state: (h, hu, hv).
///
/// Stores the conserved variables for 2D shallow water equations, or a flux
/// of them. `T` is `f64` except in the pointwise physics shared by the scalar,
/// batched and GPU kernels ([`Lane`]), where it may hold one value per SIMD
/// lane.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SWEState2D<T = f64> {
    /// Water depth h (must be non-negative)
    pub h: T,
    /// x-momentum hu = h * u
    pub hu: T,
    /// y-momentum hv = h * v
    pub hv: T,
}

impl SWEState2D {
    /// Create a new 2D SWE state.
    #[inline(always)]
    pub fn new(h: f64, hu: f64, hv: f64) -> Self {
        Self { h, hu, hv }
    }

    /// Create a state from primitive variables (h, u, v).
    #[inline(always)]
    pub fn from_primitives(h: f64, u: f64, v: f64) -> Self {
        Self {
            h,
            hu: h * u,
            hv: h * v,
        }
    }

    /// Compute velocity components (u, v) with dry cell protection.
    ///
    /// Uses desingularization to avoid division by zero:
    /// u = 2 * h * hu / (h² + max(h, h_min)²)
    #[inline(always)]
    pub fn velocity(&self, h_min: Depth) -> (f64, f64) {
        let h_min = h_min.meters();
        let h_reg = self.h.max(h_min);
        let denom = self.h * self.h + h_reg * h_reg;
        let inv_denom = 1.0 / denom;
        let factor = 2.0 * self.h * inv_denom;
        (self.hu * factor, self.hv * factor)
    }

    /// Compute velocity without desingularization (for wet cells).
    ///
    /// Returns (0, 0) if h <= h_min to avoid NaN.
    #[inline(always)]
    pub fn velocity_simple(&self, h_min: Depth) -> (f64, f64) {
        let h_min = h_min.meters();
        if self.h > h_min {
            let h_inv = 1.0 / self.h;
            (self.hu * h_inv, self.hv * h_inv)
        } else {
            (0.0, 0.0)
        }
    }

    /// Compute velocity magnitude |u| = sqrt(u² + v²).
    pub fn velocity_magnitude(&self, h_min: Depth) -> f64 {
        let (u, v) = self.velocity(h_min);
        (u * u + v * v).sqrt()
    }

    /// Total water surface elevation eta = h + B.
    pub fn surface_elevation(&self, bathymetry: f64) -> f64 {
        self.h + bathymetry
    }

    /// Check if this cell is "dry" (h < h_min).
    pub fn is_dry(&self, h_min: Depth) -> bool {
        self.h < h_min.meters()
    }

    /// Create a zero state.
    #[inline(always)]
    pub fn zero() -> Self {
        Self {
            h: 0.0,
            hu: 0.0,
            hv: 0.0,
        }
    }

    /// Convert to array representation [h, hu, hv].
    #[inline(always)]
    pub fn to_array(&self) -> [f64; 3] {
        [self.h, self.hu, self.hv]
    }

    /// Create from array representation [h, hu, hv].
    #[inline(always)]
    pub fn from_array(arr: [f64; 3]) -> Self {
        Self {
            h: arr[0],
            hu: arr[1],
            hv: arr[2],
        }
    }

    /// Compute the wave celerity c = sqrt(g * h).
    pub fn celerity(&self, g: f64) -> f64 {
        (g * self.h.max(0.0)).sqrt()
    }

    /// Compute the maximum wave speed |u| + c.
    pub fn max_wave_speed(&self, g: f64, h_min: Depth) -> f64 {
        let (u, v) = self.velocity(h_min);
        let speed = (u * u + v * v).sqrt();
        let c = self.celerity(g);
        speed + c
    }

    /// Compute the Froude number Fr = |u| / c.
    pub fn froude_number(&self, g: f64, h_min: Depth) -> f64 {
        let c = self.celerity(g);
        if c > h_min.meters() {
            self.velocity_magnitude(h_min) / c
        } else {
            0.0
        }
    }
}

impl<T: Lane> Add for SWEState2D<T> {
    type Output = Self;

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        Self {
            h: self.h + other.h,
            hu: self.hu + other.hu,
            hv: self.hv + other.hv,
        }
    }
}

impl<T: Lane> Sub for SWEState2D<T> {
    type Output = Self;

    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        Self {
            h: self.h - other.h,
            hu: self.hu - other.hu,
            hv: self.hv - other.hv,
        }
    }
}

impl<T: Lane> Neg for SWEState2D<T> {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self {
        Self {
            h: -self.h,
            hu: -self.hu,
            hv: -self.hv,
        }
    }
}

impl Mul<f64> for SWEState2D {
    type Output = Self;

    fn mul(self, scalar: f64) -> Self {
        Self {
            h: self.h * scalar,
            hu: self.hu * scalar,
            hv: self.hv * scalar,
        }
    }
}

impl Mul<SWEState2D> for f64 {
    type Output = SWEState2D;

    fn mul(self, state: SWEState2D) -> SWEState2D {
        SWEState2D {
            h: self * state.h,
            hu: self * state.hu,
            hv: self * state.hv,
        }
    }
}

/// Specialized solution for 2D SWE (3-variable system: h, hu, hv).
///
/// Uses Structure-of-Arrays (SoA) layout:
/// - `data[0]`: All h (water depth) values
/// - `data[1]`: All hu (x-momentum) values
/// - `data[2]`: All hv (y-momentum) values
pub type SWESolution2D = SystemSolution2D<3>;

/// Variable indices for SWE equations.
pub const SWE_VAR_H: usize = 0;
pub const SWE_VAR_HU: usize = 1;
pub const SWE_VAR_HV: usize = 2;

impl SWESolution2D {
    /// Get SWE state at node i in element k.
    #[inline(always)]
    pub fn get_state(&self, k: ElementIndex, i: usize) -> SWEState2D {
        let idx = k.as_usize() * self.n_nodes + i;
        SWEState2D {
            h: self.data[SWE_VAR_H][idx],
            hu: self.data[SWE_VAR_HU][idx],
            hv: self.data[SWE_VAR_HV][idx],
        }
    }

    /// Set SWE state at node i in element k.
    #[inline(always)]
    pub fn set_state(&mut self, k: ElementIndex, i: usize, state: SWEState2D) {
        let idx = k.as_usize() * self.n_nodes + i;
        self.data[SWE_VAR_H][idx] = state.h;
        self.data[SWE_VAR_HU][idx] = state.hu;
        self.data[SWE_VAR_HV][idx] = state.hv;
    }

    /// Get slice of h (water depth) values for element k.
    #[inline(always)]
    pub fn element_h(&self, k: ElementIndex) -> &[f64] {
        self.element_var(k, SWE_VAR_H)
    }

    /// Get slice of hu (x-momentum) values for element k.
    #[inline(always)]
    pub fn element_hu(&self, k: ElementIndex) -> &[f64] {
        self.element_var(k, SWE_VAR_HU)
    }

    /// Get slice of hv (y-momentum) values for element k.
    #[inline(always)]
    pub fn element_hv(&self, k: ElementIndex) -> &[f64] {
        self.element_var(k, SWE_VAR_HV)
    }

    /// Get mutable slice of h (water depth) values for element k.
    #[inline(always)]
    pub fn element_h_mut(&mut self, k: ElementIndex) -> &mut [f64] {
        self.element_var_mut(k, SWE_VAR_H)
    }

    /// Get mutable slice of hu (x-momentum) values for element k.
    #[inline(always)]
    pub fn element_hu_mut(&mut self, k: ElementIndex) -> &mut [f64] {
        self.element_var_mut(k, SWE_VAR_HU)
    }

    /// Get mutable slice of hv (y-momentum) values for element k.
    #[inline(always)]
    pub fn element_hv_mut(&mut self, k: ElementIndex) -> &mut [f64] {
        self.element_var_mut(k, SWE_VAR_HV)
    }

    /// Get slice of all h (water depth) values across all elements.
    #[inline(always)]
    pub fn h_data(&self) -> &[f64] {
        self.var_data(SWE_VAR_H)
    }

    /// Get slice of all hu (x-momentum) values across all elements.
    #[inline(always)]
    pub fn hu_data(&self) -> &[f64] {
        self.var_data(SWE_VAR_HU)
    }

    /// Get slice of all hv (y-momentum) values across all elements.
    #[inline(always)]
    pub fn hv_data(&self) -> &[f64] {
        self.var_data(SWE_VAR_HV)
    }

    /// Get mutable slice of all h values across all elements.
    #[inline(always)]
    pub fn h_data_mut(&mut self) -> &mut [f64] {
        self.var_data_mut(SWE_VAR_H)
    }

    /// Get mutable slice of all hu values across all elements.
    #[inline(always)]
    pub fn hu_data_mut(&mut self) -> &mut [f64] {
        self.var_data_mut(SWE_VAR_HU)
    }

    /// Get mutable slice of all hv values across all elements.
    #[inline(always)]
    pub fn hv_data_mut(&mut self) -> &mut [f64] {
        self.var_data_mut(SWE_VAR_HV)
    }

    /// Initialize from functions for h, u, and v.
    ///
    /// Evaluates h(x, y), u(x, y), v(x, y) at each physical node location
    /// and stores (h, h*u, h*v).
    pub fn set_from_functions<H, U, V>(
        &mut self,
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        h_fn: H,
        u_fn: U,
        v_fn: V,
    ) where
        H: Fn(f64, f64) -> f64,
        U: Fn(f64, f64) -> f64,
        V: Fn(f64, f64) -> f64,
    {
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let (r, s) = (ops.nodes_r[i], ops.nodes_s[i]);
                let [x, y] = mesh.reference_to_physical(k, r, s);
                let h = h_fn(x, y);
                let u = u_fn(x, y);
                let v = v_fn(x, y);
                self.set_state(k, i, SWEState2D::from_primitives(h, u, v));
            }
        }
    }

    /// Initialize from a single state function.
    pub fn set_from_state_function<F>(&mut self, mesh: &Mesh2D, ops: &DGOperators2D, f: F)
    where
        F: Fn(f64, f64) -> SWEState2D,
    {
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let (r, s) = (ops.nodes_r[i], ops.nodes_s[i]);
                let [x, y] = mesh.reference_to_physical(k, r, s);
                self.set_state(k, i, f(x, y));
            }
        }
    }

    /// Compute the integral of water depth h over the domain.
    ///
    /// Uses quadrature weights and Jacobian for accurate integration:
    /// ∫∫ h dA = Σ_k Σ_i w_i * h_{k,i} * J_k
    pub fn integrate_depth(&self, ops: &DGOperators2D, geom: &GeometricFactors2D) -> f64 {
        let mut integral = 0.0;
        for k in ElementIndex::iter(self.n_elements) {
            for (i, &w) in ops.weights.iter().enumerate() {
                let j = geom.jacobian(k.as_usize(), i);
                let h = self.get_var(k, i, 0);
                integral += w * h * j;
            }
        }
        integral
    }

    /// Compute the integral of x-momentum hu over the domain.
    pub fn integrate_x_momentum(&self, ops: &DGOperators2D, geom: &GeometricFactors2D) -> f64 {
        let mut integral = 0.0;
        for k in ElementIndex::iter(self.n_elements) {
            for (i, &w) in ops.weights.iter().enumerate() {
                let j = geom.jacobian(k.as_usize(), i);
                let hu = self.get_var(k, i, 1);
                integral += w * hu * j;
            }
        }
        integral
    }

    /// Compute the integral of y-momentum hv over the domain.
    pub fn integrate_y_momentum(&self, ops: &DGOperators2D, geom: &GeometricFactors2D) -> f64 {
        let mut integral = 0.0;
        for k in ElementIndex::iter(self.n_elements) {
            for (i, &w) in ops.weights.iter().enumerate() {
                let j = geom.jacobian(k.as_usize(), i);
                let hv = self.get_var(k, i, 2);
                integral += w * hv * j;
            }
        }
        integral
    }

    /// Compute L2 error for water depth against exact solution.
    pub fn l2_error_depth<F>(
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
            for i in 0..self.n_nodes {
                let j = geom.jacobian(k.as_usize(), i);
                let (r, s) = (ops.nodes_r[i], ops.nodes_s[i]);
                let [x, y] = mesh.reference_to_physical(k, r, s);
                let h = self.get_var(k, i, 0);
                let h_exact = exact(x, y);
                let diff = h - h_exact;
                error_sq += ops.weights[i] * diff * diff * j;
            }
        }
        error_sq.sqrt()
    }

    /// Compute L-infinity error for water depth.
    pub fn linf_error_depth<F>(&self, mesh: &Mesh2D, ops: &DGOperators2D, exact: F) -> f64
    where
        F: Fn(f64, f64) -> f64,
    {
        let mut max_error: f64 = 0.0;
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..self.n_nodes {
                let (r, s) = (ops.nodes_r[i], ops.nodes_s[i]);
                let [x, y] = mesh.reference_to_physical(k, r, s);
                let h = self.get_var(k, i, 0);
                let h_exact = exact(x, y);
                max_error = max_error.max((h - h_exact).abs());
            }
        }
        max_error
    }

    /// Check if any cell has negative depth.
    pub fn has_negative_depth(&self) -> bool {
        for k in ElementIndex::iter(self.n_elements) {
            for i in 0..self.n_nodes {
                if self.get_var(k, i, 0) < 0.0 {
                    return true;
                }
            }
        }
        false
    }

    /// Get minimum depth across the domain.
    pub fn min_depth(&self) -> f64 {
        let mut min_h = f64::INFINITY;
        for k in ElementIndex::iter(self.n_elements) {
            for i in 0..self.n_nodes {
                min_h = min_h.min(self.get_var(k, i, 0));
            }
        }
        min_h
    }

    /// Get maximum depth across the domain.
    pub fn max_depth(&self) -> f64 {
        let mut max_h = f64::NEG_INFINITY;
        for k in ElementIndex::iter(self.n_elements) {
            for i in 0..self.n_nodes {
                max_h = max_h.max(self.get_var(k, i, 0));
            }
        }
        max_h
    }

    /// Extract face values for a specific element and face.
    ///
    /// Returns a vector of SWEState2D for each node on the face.
    pub fn extract_face_states(
        &self,
        k: ElementIndex,
        face: usize,
        ops: &DGOperators2D,
    ) -> Vec<SWEState2D> {
        ops.face_nodes[face]
            .iter()
            .map(|&i| self.get_state(k, i))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Depth;

    fn k(idx: usize) -> ElementIndex {
        ElementIndex::new(idx)
    }

    #[test]
    fn test_swe_state_2d_basic() {
        let state = SWEState2D::new(2.0, 3.0, 4.0);
        assert!((state.h - 2.0).abs() < 1e-14);
        assert!((state.hu - 3.0).abs() < 1e-14);
        assert!((state.hv - 4.0).abs() < 1e-14);
    }

    #[test]
    fn test_swe_state_2d_from_primitives() {
        let state = SWEState2D::from_primitives(2.0, 1.5, 2.0);
        assert!((state.h - 2.0).abs() < 1e-14);
        assert!((state.hu - 3.0).abs() < 1e-14); // hu = h * u = 2 * 1.5
        assert!((state.hv - 4.0).abs() < 1e-14); // hv = h * v = 2 * 2.0
    }

    #[test]
    fn test_swe_state_2d_velocity() {
        let state = SWEState2D::new(2.0, 3.0, 4.0);
        let h_min = Depth::new(1e-6);

        // Simple velocity
        let (u, v) = state.velocity_simple(h_min);
        assert!((u - 1.5).abs() < 1e-14);
        assert!((v - 2.0).abs() < 1e-14);

        // Regularized velocity should be very close for wet cells
        let (u_reg, v_reg) = state.velocity(h_min);
        assert!((u_reg - 1.5).abs() < 1e-10);
        assert!((v_reg - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_swe_state_2d_velocity_dry() {
        let state = SWEState2D::new(1e-10, 1e-10, 1e-10);
        let h_min = Depth::new(1e-6);

        // Simple velocity returns (0, 0) for dry cells
        let (u, v) = state.velocity_simple(h_min);
        assert!((u - 0.0).abs() < 1e-14);
        assert!((v - 0.0).abs() < 1e-14);

        // Regularized velocity should be finite
        let (u_reg, v_reg) = state.velocity(h_min);
        assert!(u_reg.is_finite());
        assert!(v_reg.is_finite());
    }

    #[test]
    fn test_swe_state_2d_wave_speed() {
        let state = SWEState2D::new(1.0, 1.0, 0.0); // h=1, u=1, v=0
        let g = 9.81;
        let h_min = Depth::new(1e-6);

        let c = state.celerity(g);
        assert!((c - g.sqrt()).abs() < 1e-14);

        let max_speed = state.max_wave_speed(g, h_min);
        assert!((max_speed - (1.0 + g.sqrt())).abs() < 1e-10);
    }

    #[test]
    fn test_swe_state_2d_arithmetic() {
        let a = SWEState2D::new(1.0, 2.0, 3.0);
        let b = SWEState2D::new(4.0, 5.0, 6.0);

        let sum = a + b;
        assert!((sum.h - 5.0).abs() < 1e-14);
        assert!((sum.hu - 7.0).abs() < 1e-14);
        assert!((sum.hv - 9.0).abs() < 1e-14);

        let diff = a - b;
        assert!((diff.h - (-3.0)).abs() < 1e-14);
        assert!((diff.hu - (-3.0)).abs() < 1e-14);
        assert!((diff.hv - (-3.0)).abs() < 1e-14);

        let scaled = a * 2.0;
        assert!((scaled.h - 2.0).abs() < 1e-14);
        assert!((scaled.hu - 4.0).abs() < 1e-14);
        assert!((scaled.hv - 6.0).abs() < 1e-14);

        let scaled2 = 3.0 * a;
        assert!((scaled2.h - 3.0).abs() < 1e-14);
        assert!((scaled2.hu - 6.0).abs() < 1e-14);
        assert!((scaled2.hv - 9.0).abs() < 1e-14);
    }

    #[test]
    fn test_swe_state_2d_array_conversion() {
        let state = SWEState2D::new(1.5, 2.5, 3.5);
        let arr = state.to_array();
        assert!((arr[0] - 1.5).abs() < 1e-14);
        assert!((arr[1] - 2.5).abs() < 1e-14);
        assert!((arr[2] - 3.5).abs() < 1e-14);

        let state2 = SWEState2D::from_array(arr);
        assert_eq!(state, state2);
    }

    #[test]
    fn test_system_solution_2d_basic() {
        let n_elem = 4;
        let n_nodes = 9; // (2+1)²
        let mut sol = SWESolution2D::new(n_elem, n_nodes);

        // SoA layout: 3 arrays of n_elem * n_nodes each
        assert_eq!(sol.data.len(), 3);
        assert_eq!(sol.data[0].len(), n_elem * n_nodes);

        // Set and get values
        sol.set(k(0), 0, [1.0, 2.0, 3.0]);
        let vals = sol.get(k(0), 0);
        assert!((vals[0] - 1.0).abs() < 1e-14);
        assert!((vals[1] - 2.0).abs() < 1e-14);
        assert!((vals[2] - 3.0).abs() < 1e-14);

        // Set and get state
        let state = SWEState2D::new(4.0, 5.0, 6.0);
        sol.set_state(k(1), 2, state);
        let state2 = sol.get_state(k(1), 2);
        assert_eq!(state, state2);
    }

    #[test]
    fn test_system_solution_2d_axpy() {
        let n_elem = 2;
        let n_nodes = 4;

        let mut a = SWESolution2D::new(n_elem, n_nodes);
        let mut b = SWESolution2D::new(n_elem, n_nodes);

        // SoA layout: fill each variable array
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
        let mut sol = SWESolution2D::new(n_elem, n_nodes);

        // Set some values
        for i in 0..n_nodes {
            sol.set(k(0), i, [i as f64, (i + 10) as f64, (i + 20) as f64]);
        }

        // Get depth values for element 0
        let h_vals = sol.element_var(k(0), 0);
        assert_eq!(h_vals.len(), n_nodes);
        for i in 0..n_nodes {
            assert!((h_vals[i] - i as f64).abs() < 1e-14);
        }

        // Get x-momentum values for element 0
        let hu_vals = sol.element_var(k(0), 1);
        for i in 0..n_nodes {
            assert!((hu_vals[i] - (i + 10) as f64).abs() < 1e-14);
        }

        // Get y-momentum values for element 0
        let hv_vals = sol.element_var(k(0), 2);
        for i in 0..n_nodes {
            assert!((hv_vals[i] - (i + 20) as f64).abs() < 1e-14);
        }
    }

    #[test]
    fn test_swe_solution_2d_negative_depth() {
        let mut sol = SWESolution2D::new(2, 3);
        assert!(!sol.has_negative_depth());

        sol.set_var(k(0), 0, 0, -0.1);
        assert!(sol.has_negative_depth());
    }

    #[test]
    fn test_swe_solution_2d_min_max_depth() {
        let mut sol = SWESolution2D::new(2, 3);
        sol.set_var(k(0), 0, 0, 1.0);
        sol.set_var(k(0), 1, 0, 2.0);
        sol.set_var(k(1), 0, 0, 0.5);
        sol.set_var(k(1), 1, 0, 3.0);

        assert!((sol.min_depth() - 0.0).abs() < 1e-14); // Other nodes are still 0
        assert!((sol.max_depth() - 3.0).abs() < 1e-14);
    }

    #[test]
    fn test_swe_state_2d_froude_number() {
        // Subcritical flow: Fr < 1
        let state = SWEState2D::from_primitives(1.0, 1.0, 0.0); // h=1, u=1, v=0
        let g = 9.81;
        let h_min = Depth::new(1e-6);
        let fr = state.froude_number(g, h_min);
        assert!(fr < 1.0);
        assert!((fr - 1.0 / g.sqrt()).abs() < 1e-10);

        // Supercritical flow: Fr > 1
        let state_super = SWEState2D::from_primitives(0.1, 5.0, 0.0); // shallow, fast
        let fr_super = state_super.froude_number(g, h_min);
        assert!(fr_super > 1.0);
    }
}

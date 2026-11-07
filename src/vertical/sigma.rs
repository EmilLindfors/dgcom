//! Sigma coordinate grid for terrain-following vertical discretization.
//!
//! Sigma coordinates are terrain-following coordinates commonly used in
//! coastal ocean models (ROMS, SCHISM, Thetis). The coordinate σ ranges
//! from -1 (bottom) to 0 (surface).
//!
//! # Performance
//!
//! This module is designed for high performance:
//! - Contiguous `Vec<f64>` storage for cache-friendly access
//! - Batch operations on slices for LLVM auto-vectorization (SIMD)
//! - All coordinate transforms are `#[inline]` for zero-cost abstraction
//! - `*_into` methods avoid allocations in hot paths
//!
//! # Coordinate Transform
//!
//! The physical depth z is computed from sigma via:
//!
//! ```text
//! z = η + (η + H) × σ
//! ```
//!
//! where:
//! - η is the free surface elevation
//! - H is the water depth (positive downward from reference)
//! - σ ∈ [-1, 0]
//!
//! # Example
//!
//! ```
//! use dgcom::vertical::{SigmaGrid, UniformStretching, SongHaidvogelStretching};
//!
//! // Create a 35-level sigma grid with uniform spacing
//! let grid = SigmaGrid::new(35, UniformStretching);
//!
//! // Convert sigma to physical depth at a point
//! let eta = 0.5;   // Surface elevation (m)
//! let h = 200.0;   // Water depth (m)
//! let z = grid.sigma_to_z(grid.sigma_rho()[0], eta, h);
//! ```

use super::stretching::Stretching;
use crate::types::{Depth, Elevation, PhysicalZ, Sigma};

/// Sigma coordinate grid for terrain-following vertical discretization.
///
/// σ ∈ [-1, 0] where:
/// - σ = -1 at bottom (z = -H)
/// - σ = 0 at surface (z = η)
///
/// The grid stores sigma values at cell centers (rho-points) and
/// cell faces (w-points), following ROMS convention.
///
/// # Memory Layout
///
/// All arrays use contiguous `Vec<f64>` for:
/// - Cache-friendly sequential access patterns
/// - LLVM auto-vectorization of batch operations
/// - Zero-copy slicing for batch operations
#[derive(Clone)]
pub struct SigmaGrid {
    /// Number of vertical levels (cells).
    n_levels: usize,

    /// σ values at cell centers (rho-points), length = n_levels.
    /// Index 0 is nearest to bottom, index n_levels-1 is nearest to surface.
    sigma_rho: Vec<f64>,

    /// σ values at cell faces (w-points), length = n_levels + 1.
    /// sigma_w[0] = bottom, sigma_w[n_levels] = surface.
    sigma_w: Vec<f64>,

    /// Layer thicknesses in sigma space: Δσ[k] = sigma_w[k+1] - sigma_w[k].
    d_sigma: Vec<f64>,

    /// Name of the stretching function used.
    stretching_name: String,

    /// Description of stretching parameters.
    stretching_description: String,
}

impl SigmaGrid {
    /// Create a new sigma grid with the specified stretching.
    ///
    /// # Arguments
    ///
    /// * `n_levels` - Number of vertical levels (typically 20-50 for coastal)
    /// * `stretching` - Stretching function (uniform, Song-Haidvogel, etc.)
    ///
    /// # Example
    ///
    /// ```
    /// use dgcom::vertical::{SigmaGrid, UniformStretching};
    ///
    /// let grid = SigmaGrid::new(35, UniformStretching);
    /// assert_eq!(grid.n_levels(), 35);
    /// ```
    pub fn new(n_levels: usize, stretching: impl Stretching) -> Self {
        let (sigma_rho, sigma_w) = stretching.compute_sigma(n_levels);

        // Compute layer thicknesses
        let d_sigma: Vec<f64> = (0..n_levels).map(|k| sigma_w[k + 1] - sigma_w[k]).collect();

        Self {
            n_levels,
            sigma_rho,
            sigma_w,
            d_sigma,
            stretching_name: stretching.name().to_string(),
            stretching_description: stretching.description(),
        }
    }

    /// A grid of given levels, e.g. as stored in an output file: `sigma_w` the
    /// n + 1 faces from −1 (bed) to 0 (surface), increasing, and `sigma_rho` the n
    /// centres, each between its faces. `stretching` names where they came from.
    pub fn from_levels(
        sigma_rho: Vec<f64>,
        sigma_w: Vec<f64>,
        stretching: &str,
    ) -> Result<Self, String> {
        let n_levels = sigma_rho.len();
        if n_levels == 0 || sigma_w.len() != n_levels + 1 {
            return Err(format!(
                "{n_levels} centres need {} faces, not {}",
                n_levels + 1,
                sigma_w.len()
            ));
        }
        if (sigma_w[0] + 1.0).abs() > 1e-12 || sigma_w[n_levels].abs() > 1e-12 {
            return Err("the faces must run from -1 to 0".into());
        }
        if let Some(k) =
            (0..n_levels).find(|&k| !(sigma_w[k] < sigma_rho[k] && sigma_rho[k] < sigma_w[k + 1]))
        {
            return Err(format!("level {k}'s centre is not between its faces"));
        }
        let d_sigma = (0..n_levels).map(|k| sigma_w[k + 1] - sigma_w[k]).collect();
        Ok(Self {
            n_levels,
            sigma_rho,
            sigma_w,
            d_sigma,
            stretching_name: stretching.to_string(),
            stretching_description: format!("{stretching} (from stored levels)"),
        })
    }

    /// Create a uniform sigma grid (convenience constructor).
    #[inline]
    pub fn uniform(n_levels: usize) -> Self {
        Self::new(n_levels, super::stretching::UniformStretching)
    }

    // =========================================================================
    // Accessors (return slices for zero-copy access)
    // =========================================================================

    /// Number of vertical levels.
    #[inline]
    pub fn n_levels(&self) -> usize {
        self.n_levels
    }

    /// Sigma values at cell centers (rho-points) as a slice.
    #[inline]
    pub fn sigma_rho(&self) -> &[f64] {
        &self.sigma_rho
    }

    /// Sigma values at cell faces (w-points) as a slice.
    #[inline]
    pub fn sigma_w(&self) -> &[f64] {
        &self.sigma_w
    }

    /// Layer thicknesses in sigma space as a slice.
    #[inline]
    pub fn d_sigma(&self) -> &[f64] {
        &self.d_sigma
    }

    /// Name of the stretching function.
    #[inline]
    pub fn stretching_name(&self) -> &str {
        &self.stretching_name
    }

    /// Description of stretching parameters.
    #[inline]
    pub fn stretching_description(&self) -> &str {
        &self.stretching_description
    }

    // =========================================================================
    // Scalar Coordinate Transforms (for single points)
    // =========================================================================

    /// Convert sigma to physical depth z.
    ///
    /// ```text
    /// z = η + (η + H) × σ
    /// ```
    ///
    /// # Arguments
    ///
    /// * `sigma` - Sigma coordinate (-1 to 0)
    /// * `eta` - Free surface elevation (positive upward)
    /// * `h` - Water depth below reference (positive downward, so h > 0)
    ///
    /// # Returns
    ///
    /// Physical depth z (negative below reference, positive above).
    ///
    /// # Example
    ///
    /// ```
    /// use dgcom::vertical::{SigmaGrid, UniformStretching};
    ///
    /// let grid = SigmaGrid::new(10, UniformStretching);
    ///
    /// // At flat surface with 100m depth
    /// let z_surface = grid.sigma_to_z(0.0, 0.0, 100.0);
    /// assert!((z_surface - 0.0).abs() < 1e-10);
    ///
    /// let z_bottom = grid.sigma_to_z(-1.0, 0.0, 100.0);
    /// assert!((z_bottom - (-100.0)).abs() < 1e-10);
    ///
    /// let z_mid = grid.sigma_to_z(-0.5, 0.0, 100.0);
    /// assert!((z_mid - (-50.0)).abs() < 1e-10);
    /// ```
    #[inline]
    pub fn sigma_to_z(&self, sigma: f64, eta: f64, h: f64) -> f64 {
        // z = η + (η + H) × σ
        eta + (eta + h) * sigma
    }

    /// Convert physical depth z to sigma.
    ///
    /// ```text
    /// σ = (z - η) / (η + H)
    /// ```
    #[inline]
    pub fn z_to_sigma(&self, z: f64, eta: f64, h: f64) -> f64 {
        (z - eta) / (eta + h)
    }

    /// Compute layer thickness in physical space at a given point.
    ///
    /// ```text
    /// Δz[k] = (η + H) × Δσ[k]
    /// ```
    #[inline]
    pub fn layer_thickness(&self, level: usize, eta: f64, h: f64) -> f64 {
        (eta + h) * self.d_sigma[level]
    }

    /// Compute total water column height.
    ///
    /// ```text
    /// H_total = η + H
    /// ```
    #[inline]
    pub fn total_depth(&self, eta: f64, h: f64) -> f64 {
        eta + h
    }

    // =========================================================================
    // Typed Coordinate Transforms (using newtypes)
    // =========================================================================

    /// Convert sigma to physical depth z using typed parameters.
    ///
    /// # Example
    ///
    /// ```
    /// use dgcom::vertical::{SigmaGrid, UniformStretching};
    /// use dgcom::types::{Sigma, Elevation, Depth};
    ///
    /// let grid = SigmaGrid::new(10, UniformStretching);
    /// let sigma = Sigma::new(-0.5);
    /// let eta = Elevation::new(0.0);
    /// let h = Depth::new(100.0);
    ///
    /// let z = grid.sigma_to_z_typed(sigma, eta, h);
    /// assert!((z.meters() - (-50.0)).abs() < 1e-10);
    /// ```
    #[inline]
    pub fn sigma_to_z_typed(&self, sigma: Sigma, eta: Elevation, h: Depth) -> PhysicalZ {
        sigma.to_z(eta, h)
    }

    /// Convert physical depth z to sigma using typed parameters.
    #[inline]
    pub fn z_to_sigma_typed(&self, z: PhysicalZ, eta: Elevation, h: Depth) -> Sigma {
        Sigma::from_z(z, eta, h)
    }

    /// Compute layer thickness in physical space using typed parameters.
    #[inline]
    pub fn layer_thickness_typed(&self, level: usize, eta: Elevation, h: Depth) -> f64 {
        (eta.meters() + h.meters()) * self.d_sigma[level]
    }

    /// Compute total water column height using typed parameters.
    #[inline]
    pub fn total_depth_typed(&self, eta: Elevation, h: Depth) -> f64 {
        eta.meters() + h.meters()
    }

    /// Get sigma at a level as a typed Sigma value.
    #[inline]
    pub fn sigma_at_level_typed(&self, level: usize) -> Sigma {
        Sigma::new_unchecked(self.sigma_rho[level])
    }

    // =========================================================================
    // Metric Terms (for pressure gradient correction)
    // =========================================================================

    /// Compute ∂z/∂σ (vertical metric term).
    ///
    /// For terrain-following coordinates:
    /// ```text
    /// ∂z/∂σ = η + H
    /// ```
    ///
    /// This is constant in the vertical for simple sigma coordinates.
    #[inline]
    pub fn dz_dsigma(&self, eta: f64, h: f64) -> f64 {
        eta + h
    }

    /// Compute ∂z/∂x at constant σ (horizontal metric term).
    ///
    /// ```text
    /// ∂z/∂x|_σ = ∂η/∂x + σ × (∂η/∂x + ∂H/∂x)
    ///          = (1 + σ) × ∂η/∂x + σ × ∂H/∂x
    /// ```
    ///
    /// This is needed for computing horizontal pressure gradients
    /// in sigma coordinates.
    #[inline]
    pub fn dz_dx_at_sigma(&self, sigma: f64, d_eta_dx: f64, d_h_dx: f64) -> f64 {
        (1.0 + sigma) * d_eta_dx + sigma * d_h_dx
    }

    /// Compute ∂z/∂y at constant σ.
    #[inline]
    pub fn dz_dy_at_sigma(&self, sigma: f64, d_eta_dy: f64, d_h_dy: f64) -> f64 {
        (1.0 + sigma) * d_eta_dy + sigma * d_h_dy
    }

    /// Compute ∂z/∂x at constant σ for a batch of columns (e.g. all nodes in an element).
    ///
    /// Useful for SIMD-accelerated pressure gradient computation.
    #[inline]
    pub fn dz_dx_at_sigma_batch(
        &self,
        level: usize,
        d_eta_dx: &[f64],
        d_h_dx: &[f64],
        dz_dx_out: &mut [f64],
    ) {
        debug_assert_eq!(d_eta_dx.len(), d_h_dx.len());
        debug_assert_eq!(d_eta_dx.len(), dz_dx_out.len());

        let sigma = self.sigma_rho[level];
        let c1 = 1.0 + sigma;
        let c2 = sigma;

        for i in 0..d_eta_dx.len() {
            dz_dx_out[i] = c1 * d_eta_dx[i] + c2 * d_h_dx[i];
        }
    }

    /// Compute ∂z/∂y at constant σ for a batch of columns.
    #[inline]
    pub fn dz_dy_at_sigma_batch(
        &self,
        level: usize,
        d_eta_dy: &[f64],
        d_h_dy: &[f64],
        dz_dy_out: &mut [f64],
    ) {
        debug_assert_eq!(d_eta_dy.len(), d_h_dy.len());
        debug_assert_eq!(d_eta_dy.len(), dz_dy_out.len());

        let sigma = self.sigma_rho[level];
        let c1 = 1.0 + sigma;
        let c2 = sigma;

        for i in 0..d_eta_dy.len() {
            dz_dy_out[i] = c1 * d_eta_dy[i] + c2 * d_h_dy[i];
        }
    }

    // =========================================================================
    // Vectorized Operations (write to pre-allocated buffers)
    // =========================================================================

    /// Compute z-coordinates at all cell centers for a water column.
    ///
    /// This is the vectorized version - writes directly to output buffer
    /// to avoid allocation. Use this in hot paths.
    ///
    /// # Arguments
    ///
    /// * `eta` - Free surface elevation
    /// * `h` - Water depth
    /// * `z_out` - Output buffer, must have length >= n_levels
    ///
    /// # Panics
    ///
    /// Panics in debug mode if `z_out.len() < n_levels`.
    #[inline]
    pub fn z_at_levels_into(&self, eta: f64, h: f64, z_out: &mut [f64]) {
        debug_assert!(
            z_out.len() >= self.n_levels,
            "Output buffer too small: {} < {}",
            z_out.len(),
            self.n_levels
        );

        let depth = eta + h;
        // zip pattern for idiomatic Rust; auto-vectorizes with LLVM
        for (z, &sigma) in z_out.iter_mut().zip(self.sigma_rho.iter()) {
            *z = eta + depth * sigma;
        }
    }

    /// Compute z-coordinates at all cell faces for a water column.
    ///
    /// # Arguments
    ///
    /// * `eta` - Free surface elevation
    /// * `h` - Water depth
    /// * `z_out` - Output buffer, must have length >= n_levels + 1
    #[inline]
    pub fn z_at_faces_into(&self, eta: f64, h: f64, z_out: &mut [f64]) {
        debug_assert!(z_out.len() > self.n_levels, "Output buffer too small");

        let depth = eta + h;
        for (z, &sigma) in z_out.iter_mut().zip(self.sigma_w.iter()) {
            *z = eta + depth * sigma;
        }
    }

    /// Compute layer thicknesses for a water column.
    ///
    /// # Arguments
    ///
    /// * `eta` - Free surface elevation
    /// * `h` - Water depth
    /// * `dz_out` - Output buffer, must have length >= n_levels
    #[inline]
    pub fn layer_thicknesses_into(&self, eta: f64, h: f64, dz_out: &mut [f64]) {
        debug_assert!(dz_out.len() >= self.n_levels, "Output buffer too small");

        let depth = eta + h;
        for (dz, &ds) in dz_out.iter_mut().zip(self.d_sigma.iter()) {
            *dz = depth * ds;
        }
    }

    /// Compute z-coordinates at all levels (allocating version).
    ///
    /// For hot paths, prefer `z_at_levels_into` to avoid allocation.
    pub fn z_at_levels(&self, eta: f64, h: f64) -> Vec<f64> {
        let mut z = vec![0.0; self.n_levels];
        self.z_at_levels_into(eta, h, &mut z);
        z
    }

    /// Compute z-coordinates at all faces (allocating version).
    pub fn z_at_faces(&self, eta: f64, h: f64) -> Vec<f64> {
        let mut z = vec![0.0; self.n_levels + 1];
        self.z_at_faces_into(eta, h, &mut z);
        z
    }

    /// Compute layer thicknesses (allocating version).
    pub fn layer_thicknesses(&self, eta: f64, h: f64) -> Vec<f64> {
        let mut dz = vec![0.0; self.n_levels];
        self.layer_thicknesses_into(eta, h, &mut dz);
        dz
    }

    // =========================================================================
    // Batch Operations for Multiple Water Columns (SIMD-friendly)
    // =========================================================================

    /// Compute z at a specific level for multiple water columns.
    ///
    /// This is optimized for the common case in 3D simulations where
    /// we need z[k] for all horizontal points at once. The simple
    /// loop structure allows LLVM to auto-vectorize.
    ///
    /// # Arguments
    ///
    /// * `level` - Vertical level index
    /// * `eta` - Surface elevations, length = n_columns
    /// * `h` - Water depths, length = n_columns
    /// * `z_out` - Output buffer, length = n_columns
    ///
    /// # Formula
    ///
    /// For each column i: z[i] = eta[i] + (eta[i] + h[i]) * sigma_rho[level]
    #[inline]
    pub fn z_at_level_batch(&self, level: usize, eta: &[f64], h: &[f64], z_out: &mut [f64]) {
        debug_assert_eq!(eta.len(), h.len());
        debug_assert_eq!(eta.len(), z_out.len());
        debug_assert!(level < self.n_levels);

        let sigma = self.sigma_rho[level];

        // Simple loop structure for LLVM auto-vectorization
        for i in 0..eta.len() {
            let depth = eta[i] + h[i];
            z_out[i] = eta[i] + depth * sigma;
        }
    }

    /// Compute layer thickness at a specific level for multiple water columns.
    ///
    /// # Arguments
    ///
    /// * `level` - Vertical level index
    /// * `eta` - Surface elevations, length = n_columns
    /// * `h` - Water depths, length = n_columns
    /// * `dz_out` - Output buffer, length = n_columns
    #[inline]
    pub fn layer_thickness_batch(&self, level: usize, eta: &[f64], h: &[f64], dz_out: &mut [f64]) {
        debug_assert_eq!(eta.len(), h.len());
        debug_assert_eq!(eta.len(), dz_out.len());
        debug_assert!(level < self.n_levels);

        let d_sigma = self.d_sigma[level];

        for i in 0..eta.len() {
            dz_out[i] = (eta[i] + h[i]) * d_sigma;
        }
    }

    /// Compute ∂z/∂σ for multiple water columns.
    ///
    /// # Arguments
    ///
    /// * `eta` - Surface elevations
    /// * `h` - Water depths
    /// * `hz_out` - Output buffer for Hz = eta + h
    #[inline]
    pub fn dz_dsigma_batch(&self, eta: &[f64], h: &[f64], hz_out: &mut [f64]) {
        debug_assert_eq!(eta.len(), h.len());
        debug_assert_eq!(eta.len(), hz_out.len());

        for i in 0..eta.len() {
            hz_out[i] = eta[i] + h[i];
        }
    }

    // =========================================================================
    // Grid Information
    // =========================================================================

    /// Get the sigma value at the center of level k.
    #[inline]
    pub fn sigma_at_level(&self, level: usize) -> f64 {
        self.sigma_rho[level]
    }

    /// Get the sigma value at the top face of level k.
    #[inline]
    pub fn sigma_at_top_face(&self, level: usize) -> f64 {
        self.sigma_w[level + 1]
    }

    /// Get the sigma value at the bottom face of level k.
    #[inline]
    pub fn sigma_at_bottom_face(&self, level: usize) -> f64 {
        self.sigma_w[level]
    }

    /// Find the level index containing a given sigma value.
    ///
    /// Uses binary search for O(log n) lookup.
    ///
    /// Returns None if sigma is outside [-1, 0].
    pub fn find_level(&self, sigma: f64) -> Option<usize> {
        let bottom = self.sigma_w[0];
        let top = self.sigma_w[self.n_levels];

        if sigma < bottom || sigma > top {
            return None;
        }

        // Binary search for the level containing sigma
        let mut lo = 0;
        let mut hi = self.n_levels;

        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if sigma < self.sigma_w[mid] {
                hi = mid;
            } else if sigma > self.sigma_w[mid + 1] {
                lo = mid + 1;
            } else {
                return Some(mid);
            }
        }

        Some(lo.min(self.n_levels - 1))
    }

    // =========================================================================
    // Diagnostics
    // =========================================================================

    /// Print grid information for debugging.
    pub fn print_info(&self) {
        println!("Sigma Grid: {} levels", self.n_levels);
        println!("  Stretching: {}", self.stretching_description);
        println!(
            "  Sigma range: [{:.6}, {:.6}]",
            self.sigma_w[0], self.sigma_w[self.n_levels]
        );

        // Layer thickness statistics
        let min_ds = self.min_d_sigma();
        let max_ds = self.max_d_sigma();
        let mean_ds = self.mean_d_sigma();

        println!(
            "  Δσ: min={:.6}, max={:.6}, mean={:.6}",
            min_ds, max_ds, mean_ds
        );
        println!("  Refinement ratio: {:.2}", max_ds / min_ds);
    }

    /// Get minimum layer thickness in sigma space.
    pub fn min_d_sigma(&self) -> f64 {
        self.d_sigma.iter().copied().fold(f64::INFINITY, f64::min)
    }

    /// Get maximum layer thickness in sigma space.
    pub fn max_d_sigma(&self) -> f64 {
        self.d_sigma.iter().copied().fold(0.0, f64::max)
    }

    /// Get mean layer thickness in sigma space.
    pub fn mean_d_sigma(&self) -> f64 {
        self.d_sigma.iter().sum::<f64>() / self.n_levels as f64
    }

    /// Get the stretching refinement ratio (max_ds / min_ds).
    pub fn refinement_ratio(&self) -> f64 {
        self.max_d_sigma() / self.min_d_sigma()
    }

    /// Compute the depth-weighted average of a vertical column.
    ///
    /// Integral is computed as sum(q_k * d_sigma_k).
    /// Since sum(d_sigma) = 1, this is the average.
    ///
    /// # Arguments
    /// * `column` - Vertical column of values at rho-points
    #[inline]
    pub fn depth_average(&self, column: &[f64]) -> f64 {
        debug_assert_eq!(column.len(), self.n_levels);
        let mut sum = 0.0;
        // Auto-vectorizable loop
        for k in 0..self.n_levels {
            sum += column[k] * self.d_sigma[k];
        }
        sum
    }

    /// Compute the depth-weighted average of a vertical column (SIMD optimized).
    ///
    /// # Arguments
    /// * `column` - Vertical column of values
    /// * `out` - Mutable reference to store result (useful for accumulation)
    #[inline]
    pub fn depth_average_accumulate(&self, column: &[f64], out: &mut f64) {
        *out += self.depth_average(column);
    }
}

// =============================================================================
// Display
// =============================================================================

impl std::fmt::Debug for SigmaGrid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SigmaGrid")
            .field("n_levels", &self.n_levels)
            .field("stretching", &self.stretching_name)
            .finish()
    }
}

impl std::fmt::Display for SigmaGrid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SigmaGrid({} levels, {})",
            self.n_levels, self.stretching_description
        )
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vertical::stretching::{SongHaidvogelStretching, UniformStretching};

    const TOL: f64 = 1e-10;

    #[test]
    fn a_grid_is_rebuilt_from_its_levels() {
        let grid = SigmaGrid::new(7, SongHaidvogelStretching::new(5.0, 0.4, 10.0));
        let rebuilt = SigmaGrid::from_levels(
            grid.sigma_rho().to_vec(),
            grid.sigma_w().to_vec(),
            grid.stretching_name(),
        )
        .unwrap();
        assert_eq!(rebuilt.d_sigma(), grid.d_sigma());
        assert_eq!(
            rebuilt.sigma_to_z(-0.3, 0.2, 40.0),
            grid.sigma_to_z(-0.3, 0.2, 40.0)
        );
        // Faces not from −1 to 0, a centre outside its layer, or lengths that disagree
        assert!(SigmaGrid::from_levels(vec![-0.5], vec![-1.0, 0.1], "x").is_err());
        assert!(SigmaGrid::from_levels(vec![-0.9, -0.95], vec![-1.0, -0.5, 0.0], "x").is_err());
        assert!(SigmaGrid::from_levels(vec![-0.5], vec![-1.0, -0.5, 0.0], "x").is_err());
        assert!(SigmaGrid::from_levels(vec![], vec![-1.0], "x").is_err());
    }

    #[test]
    fn test_depth_average() {
        let n_levels = 10;
        let grid = SigmaGrid::uniform(n_levels);

        // Constant function: average should be the constant
        let col_const = vec![5.0; n_levels];
        let avg = grid.depth_average(&col_const);
        assert!((avg - 5.0).abs() < 1e-14);

        // Linear function: f(s) = s + 0.5 (goes from -0.5 to 0.5)
        // Average should be 0.0
        let col_lin: Vec<f64> = grid.sigma_rho().iter().map(|s| s + 0.5).collect();
        let avg_lin = grid.depth_average(&col_lin);
        // Riemann sum error is O(h^2) for midpoint, so exact for linear?
        // sigma_rho are midpoints. Yes.
        assert!(avg_lin.abs() < 1e-14);
    }

    #[test]
    fn test_sigma_grid_creation() {
        let grid = SigmaGrid::new(35, UniformStretching);

        assert_eq!(grid.n_levels(), 35);
        assert_eq!(grid.sigma_rho().len(), 35);
        assert_eq!(grid.sigma_w().len(), 36);
        assert_eq!(grid.d_sigma().len(), 35);
    }

    #[test]
    fn test_sigma_to_z_flat_surface() {
        let grid = SigmaGrid::uniform(10);

        // Flat surface, 100m depth
        let eta = 0.0;
        let h = 100.0;

        // Surface: sigma = 0 -> z = 0
        assert!((grid.sigma_to_z(0.0, eta, h) - 0.0).abs() < TOL);

        // Bottom: sigma = -1 -> z = -100
        assert!((grid.sigma_to_z(-1.0, eta, h) - (-100.0)).abs() < TOL);

        // Mid-depth: sigma = -0.5 -> z = -50
        assert!((grid.sigma_to_z(-0.5, eta, h) - (-50.0)).abs() < TOL);
    }

    #[test]
    fn test_sigma_to_z_with_elevation() {
        let grid = SigmaGrid::uniform(10);

        // Elevated surface, 100m depth
        let eta = 1.0;
        let h = 100.0;

        // Surface: sigma = 0 -> z = eta = 1.0
        assert!((grid.sigma_to_z(0.0, eta, h) - 1.0).abs() < TOL);

        // Bottom: sigma = -1 -> z = eta - (eta + h) = -100
        assert!((grid.sigma_to_z(-1.0, eta, h) - (-100.0)).abs() < TOL);
    }

    #[test]
    fn test_z_to_sigma_roundtrip() {
        let grid = SigmaGrid::uniform(10);

        let eta = 0.5;
        let h = 200.0;

        for k in 0..grid.n_levels() {
            let sigma = grid.sigma_rho()[k];
            let z = grid.sigma_to_z(sigma, eta, h);
            let sigma_back = grid.z_to_sigma(z, eta, h);
            assert!(
                (sigma - sigma_back).abs() < TOL,
                "Roundtrip failed: {} -> {} -> {}",
                sigma,
                z,
                sigma_back
            );
        }
    }

    #[test]
    fn test_layer_thickness_uniform() {
        let grid = SigmaGrid::uniform(10);

        let eta = 0.0;
        let h = 100.0;

        // All layers should have equal thickness
        let expected_dz = 10.0;
        for k in 0..grid.n_levels() {
            let dz = grid.layer_thickness(k, eta, h);
            assert!(
                (dz - expected_dz).abs() < TOL,
                "Layer {} thickness {} != {}",
                k,
                dz,
                expected_dz
            );
        }
    }

    #[test]
    fn test_layer_thickness_sum() {
        let grid = SigmaGrid::new(35, SongHaidvogelStretching::default());

        let eta = 0.5;
        let h = 200.0;
        let total = grid.total_depth(eta, h);

        let sum_dz: f64 = (0..grid.n_levels())
            .map(|k| grid.layer_thickness(k, eta, h))
            .sum();

        assert!(
            (sum_dz - total).abs() < TOL * total,
            "Layer thicknesses should sum to total depth: {} != {}",
            sum_dz,
            total
        );
    }

    #[test]
    fn test_metric_terms() {
        let grid = SigmaGrid::uniform(10);

        let eta = 0.5;
        let h = 100.0;

        // ∂z/∂σ = η + H
        let dz_ds = grid.dz_dsigma(eta, h);
        assert!((dz_ds - 100.5).abs() < TOL);
    }

    #[test]
    fn test_find_level() {
        let grid = SigmaGrid::uniform(10);

        // Bottom level
        assert_eq!(grid.find_level(-0.95), Some(0));

        // Top level
        assert_eq!(grid.find_level(-0.05), Some(9));

        // Middle level
        assert_eq!(grid.find_level(-0.45), Some(5));

        // Out of bounds
        assert_eq!(grid.find_level(-1.5), None);
        assert_eq!(grid.find_level(0.5), None);
    }

    #[test]
    fn test_z_at_levels() {
        let grid = SigmaGrid::uniform(5);

        let eta = 0.0;
        let h = 100.0;

        let z_levels = grid.z_at_levels(eta, h);
        assert_eq!(z_levels.len(), 5);

        // Check that z values are increasing (bottom to surface)
        for i in 1..5 {
            assert!(z_levels[i] > z_levels[i - 1]);
        }
    }

    #[test]
    fn test_z_at_levels_into() {
        let grid = SigmaGrid::uniform(5);

        let eta = 0.0;
        let h = 100.0;

        let mut z_out = vec![0.0; 5];
        grid.z_at_levels_into(eta, h, &mut z_out);

        let z_alloc = grid.z_at_levels(eta, h);

        for k in 0..5 {
            assert!(
                (z_out[k] - z_alloc[k]).abs() < TOL,
                "Mismatch at level {}",
                k
            );
        }
    }

    #[test]
    fn test_batch_operations() {
        let grid = SigmaGrid::uniform(10);

        let n_cols = 100;
        let eta: Vec<f64> = (0..n_cols).map(|i| 0.1 * i as f64).collect();
        let h: Vec<f64> = (0..n_cols).map(|i| 100.0 + i as f64).collect();
        let mut z_out = vec![0.0; n_cols];

        // Test z_at_level_batch
        let level = 5;
        grid.z_at_level_batch(level, &eta, &h, &mut z_out);

        // Verify against scalar version
        for i in 0..n_cols {
            let expected = grid.sigma_to_z(grid.sigma_at_level(level), eta[i], h[i]);
            assert!(
                (z_out[i] - expected).abs() < TOL,
                "Batch mismatch at column {}",
                i
            );
        }
    }

    #[test]
    fn test_layer_thickness_batch() {
        let grid = SigmaGrid::uniform(10);

        let n_cols = 50;
        let eta: Vec<f64> = (0..n_cols).map(|i| 0.1 * i as f64).collect();
        let h: Vec<f64> = (0..n_cols).map(|i| 100.0 + i as f64).collect();
        let mut dz_out = vec![0.0; n_cols];

        let level = 3;
        grid.layer_thickness_batch(level, &eta, &h, &mut dz_out);

        // Verify against scalar version
        for i in 0..n_cols {
            let expected = grid.layer_thickness(level, eta[i], h[i]);
            assert!(
                (dz_out[i] - expected).abs() < TOL,
                "Batch thickness mismatch at column {}",
                i
            );
        }
    }

    #[test]
    fn test_refinement_ratio() {
        // Uniform should have ratio ~1
        let uniform = SigmaGrid::uniform(10);
        assert!((uniform.refinement_ratio() - 1.0).abs() < 0.01);

        // Stretched should have ratio > 1
        let stretched = SigmaGrid::new(35, SongHaidvogelStretching::new(7.0, 0.1, 250.0));
        assert!(stretched.refinement_ratio() > 1.0);
    }

    #[test]
    fn test_display() {
        let grid = SigmaGrid::uniform(35);
        let s = format!("{}", grid);
        assert!(s.contains("35 levels"));
        assert!(s.contains("uniform"));
    }

    // =========================================================================
    // Tests for typed API
    // =========================================================================

    #[test]
    fn test_sigma_to_z_typed() {
        use crate::types::{Depth, Elevation, Sigma as SigmaType};

        let grid = SigmaGrid::uniform(10);

        let sigma = SigmaType::new(-0.5);
        let eta = Elevation::new(0.0);
        let h = Depth::new(100.0);

        let z = grid.sigma_to_z_typed(sigma, eta, h);
        assert!((z.meters() - (-50.0)).abs() < TOL);

        // Surface
        let z_surf = grid.sigma_to_z_typed(SigmaType::SURFACE, eta, h);
        assert!((z_surf.meters() - 0.0).abs() < TOL);

        // Bottom
        let z_bot = grid.sigma_to_z_typed(SigmaType::BOTTOM, eta, h);
        assert!((z_bot.meters() - (-100.0)).abs() < TOL);
    }

    #[test]
    fn test_z_to_sigma_typed_roundtrip() {
        use crate::types::{Depth, Elevation};

        let grid = SigmaGrid::uniform(10);

        let eta = Elevation::new(0.5);
        let h = Depth::new(200.0);

        for k in 0..grid.n_levels() {
            let sigma = grid.sigma_at_level_typed(k);
            let z = grid.sigma_to_z_typed(sigma, eta, h);
            let sigma_back = grid.z_to_sigma_typed(z, eta, h);

            assert!(
                (sigma.value() - sigma_back.value()).abs() < TOL,
                "Typed roundtrip failed at level {}: {} -> {} -> {}",
                k,
                sigma.value(),
                z.meters(),
                sigma_back.value()
            );
        }
    }

    #[test]
    fn test_layer_thickness_typed() {
        use crate::types::{Depth, Elevation};

        let grid = SigmaGrid::uniform(10);

        let eta = Elevation::new(0.0);
        let h = Depth::new(100.0);

        // All layers should have equal thickness (10m for 10 uniform levels)
        let expected_dz = 10.0;
        for k in 0..grid.n_levels() {
            let dz = grid.layer_thickness_typed(k, eta, h);
            assert!(
                (dz - expected_dz).abs() < TOL,
                "Typed layer {} thickness {} != {}",
                k,
                dz,
                expected_dz
            );
        }
    }

    #[test]
    fn test_total_depth_typed() {
        use crate::types::{Depth, Elevation};

        let grid = SigmaGrid::uniform(10);

        let eta = Elevation::new(1.5);
        let h = Depth::new(100.0);

        let total = grid.total_depth_typed(eta, h);
        assert!((total - 101.5).abs() < TOL);
    }
}

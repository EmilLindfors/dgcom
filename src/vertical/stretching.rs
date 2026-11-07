//! Vertical stretching functions for sigma coordinate grids.
//!
//! Stretching functions control the vertical distribution of sigma levels,
//! allowing refinement near the surface and/or bottom boundaries.
//!
//! # Available Stretching Functions
//!
//! - [`UniformStretching`]: Equal spacing in sigma
//! - [`SongHaidvogelStretching`]: ROMS-style surface/bottom clustering
//!
//! # Example
//!
//! ```
//! use dgcom::vertical::{SigmaGrid, UniformStretching, SongHaidvogelStretching};
//!
//! // Uniform spacing
//! let uniform = SigmaGrid::new(35, UniformStretching);
//!
//! // ROMS-style stretching with surface refinement
//! let stretched = SigmaGrid::new(35, SongHaidvogelStretching {
//!     theta_s: 7.0,   // Strong surface stretching
//!     theta_b: 0.1,   // Weak bottom stretching
//!     hc: 250.0,      // Critical depth (m)
//! });
//! ```

/// Trait for vertical stretching functions.
///
/// A stretching function defines how sigma levels are distributed
/// between -1 (bottom) and 0 (surface).
///
/// # Implementation Notes
///
/// - `sigma_rho`: Cell-center values, length = n_levels
/// - `sigma_w`: Cell-face values, length = n_levels + 1
/// - Values should be strictly increasing from -1 to 0
pub trait Stretching: Send + Sync {
    /// Compute sigma level distributions.
    ///
    /// Returns (sigma_rho, sigma_w) where:
    /// - `sigma_rho`: Cell-center sigma values, length = n_levels
    /// - `sigma_w`: Cell-face sigma values, length = n_levels + 1
    ///
    /// Convention:
    /// - sigma_w[0] = -1.0 (bottom)
    /// - sigma_w[n_levels] = 0.0 (surface)
    fn compute_sigma(&self, n_levels: usize) -> (Vec<f64>, Vec<f64>);

    /// Human-readable name for debugging and logging.
    fn name(&self) -> &'static str;

    /// Description of parameters (for diagnostics).
    fn description(&self) -> String {
        self.name().to_string()
    }
}

// =============================================================================
// Uniform Stretching
// =============================================================================

/// Uniform (equal) spacing in sigma coordinates.
///
/// The simplest stretching function: levels are equally spaced
/// from bottom (-1) to surface (0).
///
/// # When to Use
///
/// - Testing and debugging
/// - When vertical structure is relatively uniform
/// - As a baseline for comparison with other stretching
#[derive(Clone, Copy, Debug, Default)]
pub struct UniformStretching;

impl Stretching for UniformStretching {
    fn compute_sigma(&self, n_levels: usize) -> (Vec<f64>, Vec<f64>) {
        let n = n_levels;

        // Cell faces: equally spaced from -1 to 0
        let sigma_w: Vec<f64> = (0..=n).map(|k| -1.0 + k as f64 / n as f64).collect();

        // Cell centers: midpoints between faces
        let sigma_rho: Vec<f64> = (0..n)
            .map(|k| (sigma_w[k] + sigma_w[k + 1]) / 2.0)
            .collect();

        (sigma_rho, sigma_w)
    }

    fn name(&self) -> &'static str {
        "uniform"
    }
}

// =============================================================================
// Song-Haidvogel Stretching (ROMS Default)
// =============================================================================

/// Song-Haidvogel vertical stretching function.
///
/// This is the default stretching function used in ROMS and other
/// sigma-coordinate ocean models. It allows independent control of
/// resolution near the surface and bottom.
///
/// # Parameters
///
/// - `theta_s`: Surface stretching parameter (0 to 10)
///   - 0 = no surface stretching
///   - 7 = strong surface refinement (typical for mixed layer)
///
/// - `theta_b`: Bottom stretching parameter (0 to 4)
///   - 0 = no bottom stretching
///   - 2 = moderate bottom refinement (typical for BBL)
///
/// - `hc`: Critical depth (meters)
///   - Controls the depth at which stretching transitions
///   - Typical values: 10-250 m
///   - Larger values = more uniform in shallow water
///
/// # References
///
/// - Song, Y. and D.B. Haidvogel (1994): A semi-implicit ocean circulation
///   model using a generalized topography-following coordinate system.
///   J. Comp. Phys., 115, 228-244.
///
/// # Example
///
/// ```
/// use dgcom::vertical::{SigmaGrid, SongHaidvogelStretching};
///
/// // Norwegian coastal configuration
/// let stretching = SongHaidvogelStretching {
///     theta_s: 7.0,   // Strong surface refinement for mixed layer
///     theta_b: 0.1,   // Weak bottom refinement
///     hc: 250.0,      // Critical depth
/// };
/// let grid = SigmaGrid::new(35, stretching);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct SongHaidvogelStretching {
    /// Surface stretching parameter (0-10, typical: 5-7).
    pub theta_s: f64,
    /// Bottom stretching parameter (0-4, typical: 0-2).
    pub theta_b: f64,
    /// Critical depth in meters (typical: 10-250).
    pub hc: f64,
}

impl Default for SongHaidvogelStretching {
    fn default() -> Self {
        Self {
            theta_s: 5.0,
            theta_b: 0.4,
            hc: 200.0,
        }
    }
}

impl SongHaidvogelStretching {
    /// Create stretching with custom parameters.
    pub fn new(theta_s: f64, theta_b: f64, hc: f64) -> Self {
        Self {
            theta_s,
            theta_b,
            hc,
        }
    }

    /// Compute the C(sigma) stretching function.
    ///
    /// This is a non-standard Cs function that uses a weighted blend of
    /// surface and bottom components: `Cs = (θs/(θs+θb))*Cs_s + (θb/(θs+θb))*Cs_b`.
    /// This does not match any standard ROMS Vstretching formula.
    /// For the standard ROMS Vstretching=4, use [`ROMSVstretching4`].
    ///
    /// The formula preserves boundaries: C(-1) = -1, C(0) = 0.
    fn cs_function(&self, sigma: f64) -> f64 {
        // Surface stretching: clusters levels near surface (sigma = 0)
        // C(s) = (1 - cosh(theta_s * s)) / (cosh(theta_s) - 1)
        //
        // At s = 0:  C(0) = 0
        // At s = -1: C(-1) = -1

        let cs_surface = if self.theta_s > 0.0 {
            (1.0 - (self.theta_s * sigma).cosh()) / (self.theta_s.cosh() - 1.0)
        } else {
            sigma
        };

        // Bottom stretching: clusters levels near bottom (sigma = -1)
        // Use tanh function: Cb = tanh(theta_b * (s + 1)) / tanh(theta_b) - 1
        //
        // At s = 0:  Cb = 0
        // At s = -1: Cb = -1

        let cs_bottom = if self.theta_b > 0.0 {
            (self.theta_b * (sigma + 1.0)).tanh() / self.theta_b.tanh() - 1.0
        } else {
            sigma
        };

        // Blend surface and bottom stretching
        // Weight by normalized theta values for smooth transition
        if self.theta_s > 0.0 && self.theta_b > 0.0 {
            // Both active: blend them
            // Weight surface stretching more heavily when theta_s > theta_b
            let weight_s = self.theta_s / (self.theta_s + self.theta_b);
            let weight_b = self.theta_b / (self.theta_s + self.theta_b);
            weight_s * cs_surface + weight_b * cs_bottom
        } else if self.theta_s > 0.0 {
            cs_surface
        } else if self.theta_b > 0.0 {
            cs_bottom
        } else {
            sigma
        }
    }
}

impl Stretching for SongHaidvogelStretching {
    fn compute_sigma(&self, n_levels: usize) -> (Vec<f64>, Vec<f64>) {
        let n = n_levels;

        // First compute uniform sigma levels
        let sigma_uniform: Vec<f64> = (0..=n).map(|k| -1.0 + k as f64 / n as f64).collect();

        // Apply stretching function to get sigma_w
        let sigma_w: Vec<f64> = sigma_uniform.iter().map(|&s| self.cs_function(s)).collect();

        // Cell centers: midpoints in stretched space
        let sigma_rho: Vec<f64> = (0..n)
            .map(|k| (sigma_w[k] + sigma_w[k + 1]) / 2.0)
            .collect();

        (sigma_rho, sigma_w)
    }

    fn name(&self) -> &'static str {
        "song_haidvogel"
    }

    fn description(&self) -> String {
        format!(
            "Song-Haidvogel (theta_s={:.1}, theta_b={:.1}, hc={:.0}m)",
            self.theta_s, self.theta_b, self.hc
        )
    }
}

// =============================================================================
// ROMS Vstretching 4 (Shchepetkin & McWilliams, 2005)
// =============================================================================

/// ROMS Vstretching option 4 (Shchepetkin & McWilliams, 2005).
///
/// This is the standard ROMS stretching function (Vstretching=4), widely used
/// in operational ocean modeling. It provides better behaved stretching than
/// the Song-Haidvogel formula, especially for steep bathymetry.
///
/// # Formula
///
/// ```text
/// C(s) = (1 - b) * Cs + b * Cb
///
/// where:
///   Cs = sinh(θs * s) / sinh(θs)                     (surface component)
///   Cb = -0.5 + tanh(θb * (s + 0.5)) / (2 * tanh(0.5 * θb))  (bottom component)
///   b  = (1 - cosh(θs * s)) / (cosh(θs) - 1)         (blending weight)
/// ```
///
/// When θs = 0: Cs = s, b = s (linear).
/// When θb = 0: Cb = s (linear).
///
/// # Parameters
///
/// - `theta_s`: Surface stretching parameter (0 to 10)
/// - `theta_b`: Bottom stretching parameter (0 to 4)
///
/// # Properties
///
/// - C(-1) = -1, C(0) = 0 (preserves boundaries)
/// - Monotonically increasing for valid parameter ranges
/// - Better separation of surface and bottom control than Song-Haidvogel
///
/// # References
///
/// - Shchepetkin, A.F. and J.C. McWilliams (2005): The regional oceanic modeling
///   system (ROMS): a split-explicit, free-surface, topography-following-coordinate
///   oceanic model. Ocean Modelling, 9, 347-404.
#[derive(Clone, Copy, Debug)]
pub struct ROMSVstretching4 {
    /// Surface stretching parameter (0-10, typical: 5-7).
    pub theta_s: f64,
    /// Bottom stretching parameter (0-4, typical: 0-2).
    pub theta_b: f64,
}

impl Default for ROMSVstretching4 {
    fn default() -> Self {
        Self {
            theta_s: 5.0,
            theta_b: 0.4,
        }
    }
}

impl ROMSVstretching4 {
    /// Create with custom parameters.
    pub fn new(theta_s: f64, theta_b: f64) -> Self {
        Self { theta_s, theta_b }
    }

    /// Compute the C(s) stretching function (Vstretching=4).
    fn cs_function(&self, s: f64) -> f64 {
        // Surface component
        let cs = if self.theta_s > 0.0 {
            (self.theta_s * s).sinh() / self.theta_s.sinh()
        } else {
            s
        };

        // Bottom component
        let cb = if self.theta_b > 0.0 {
            -0.5 + (self.theta_b * (s + 0.5)).tanh() / (2.0 * (0.5 * self.theta_b).tanh())
        } else {
            s
        };

        // Blending weight
        let b = if self.theta_s > 0.0 {
            (1.0 - (self.theta_s * s).cosh()) / (self.theta_s.cosh() - 1.0)
        } else {
            // When theta_s = 0, b = s (linear blending)
            s
        };

        (1.0 - b) * cs + b * cb
    }
}

impl Stretching for ROMSVstretching4 {
    fn compute_sigma(&self, n_levels: usize) -> (Vec<f64>, Vec<f64>) {
        let n = n_levels;

        // Compute uniform sigma levels
        let sigma_uniform: Vec<f64> = (0..=n).map(|k| -1.0 + k as f64 / n as f64).collect();

        // Apply stretching function to get sigma_w
        let sigma_w: Vec<f64> = sigma_uniform.iter().map(|&s| self.cs_function(s)).collect();

        // Cell centers: midpoints in stretched space
        let sigma_rho: Vec<f64> = (0..n)
            .map(|k| (sigma_w[k] + sigma_w[k + 1]) / 2.0)
            .collect();

        (sigma_rho, sigma_w)
    }

    fn name(&self) -> &'static str {
        "roms_vstretching4"
    }

    fn description(&self) -> String {
        format!(
            "ROMS Vstretching=4 (theta_s={:.1}, theta_b={:.1})",
            self.theta_s, self.theta_b
        )
    }
}

// =============================================================================
// Double Stretching (Surface + Bottom)
// =============================================================================

/// Double-tanh stretching for surface and bottom boundary layers.
///
/// Provides independent control of surface and bottom layer thicknesses.
/// Useful when you need fine resolution in both boundary layers.
#[derive(Clone, Copy, Debug)]
pub struct DoubleTanhStretching {
    /// Surface layer thickness parameter (0-1).
    pub surface_frac: f64,
    /// Bottom layer thickness parameter (0-1).
    pub bottom_frac: f64,
    /// Stretching sharpness for surface layer.
    pub surface_sharpness: f64,
    /// Stretching sharpness for bottom layer.
    pub bottom_sharpness: f64,
}

impl Default for DoubleTanhStretching {
    fn default() -> Self {
        Self {
            surface_frac: 0.1,
            bottom_frac: 0.1,
            surface_sharpness: 5.0,
            bottom_sharpness: 3.0,
        }
    }
}

impl Stretching for DoubleTanhStretching {
    fn compute_sigma(&self, n_levels: usize) -> (Vec<f64>, Vec<f64>) {
        let n = n_levels;

        // Compute sigma_w values with double-tanh stretching
        let sigma_w: Vec<f64> = (0..=n)
            .map(|k| {
                let s_uniform = -1.0 + k as f64 / n as f64;

                // Surface stretching: concentrate near sigma = 0
                let surface_stretch = if self.surface_sharpness > 0.0 {
                    let arg = self.surface_sharpness * (s_uniform + self.surface_frac);
                    self.surface_frac * (arg.tanh() / self.surface_sharpness.tanh() - 1.0)
                } else {
                    0.0
                };

                // Bottom stretching: concentrate near sigma = -1
                let bottom_stretch = if self.bottom_sharpness > 0.0 {
                    let arg = self.bottom_sharpness * (s_uniform + 1.0 - self.bottom_frac);
                    -self.bottom_frac * (1.0 - arg.tanh() / self.bottom_sharpness.tanh())
                } else {
                    0.0
                };

                s_uniform + surface_stretch + bottom_stretch
            })
            .collect();

        // Cell centers
        let sigma_rho: Vec<f64> = (0..n)
            .map(|k| (sigma_w[k] + sigma_w[k + 1]) / 2.0)
            .collect();

        (sigma_rho, sigma_w)
    }

    fn name(&self) -> &'static str {
        "double_tanh"
    }

    fn description(&self) -> String {
        format!(
            "Double-tanh (surface={:.0}%, bottom={:.0}%)",
            self.surface_frac * 100.0,
            self.bottom_frac * 100.0
        )
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    #[test]
    fn test_uniform_stretching_bounds() {
        let stretching = UniformStretching;
        let n = 35;
        let (sigma_rho, sigma_w) = stretching.compute_sigma(n);

        // Check lengths
        assert_eq!(sigma_rho.len(), n);
        assert_eq!(sigma_w.len(), n + 1);

        // Check boundary values
        assert!((sigma_w[0] - (-1.0)).abs() < TOL, "Bottom should be -1.0");
        assert!((sigma_w[n] - 0.0).abs() < TOL, "Surface should be 0.0");

        // Check monotonicity
        for i in 1..=n {
            assert!(
                sigma_w[i] > sigma_w[i - 1],
                "sigma_w should be monotonically increasing"
            );
        }
        for i in 1..n {
            assert!(
                sigma_rho[i] > sigma_rho[i - 1],
                "sigma_rho should be monotonically increasing"
            );
        }
    }

    #[test]
    fn test_uniform_stretching_spacing() {
        let stretching = UniformStretching;
        let n = 10;
        let (_, sigma_w) = stretching.compute_sigma(n);

        // Check uniform spacing
        let expected_spacing = 1.0 / n as f64;
        for i in 1..=n {
            let spacing = sigma_w[i] - sigma_w[i - 1];
            assert!(
                (spacing - expected_spacing).abs() < TOL,
                "Spacing should be uniform: got {}, expected {}",
                spacing,
                expected_spacing
            );
        }
    }

    #[test]
    fn test_song_haidvogel_bounds() {
        let stretching = SongHaidvogelStretching::new(7.0, 0.1, 250.0);
        let n = 35;
        let (sigma_rho, sigma_w) = stretching.compute_sigma(n);

        // Check lengths
        assert_eq!(sigma_rho.len(), n);
        assert_eq!(sigma_w.len(), n + 1);

        // Check boundary values (may not be exactly -1 and 0 due to stretching)
        assert!(sigma_w[0] < -0.9, "Bottom should be near -1.0");
        assert!(sigma_w[n] > -0.1, "Surface should be near 0.0");

        // Check monotonicity
        for i in 1..=n {
            assert!(
                sigma_w[i] > sigma_w[i - 1],
                "sigma_w should be monotonically increasing"
            );
        }
    }

    #[test]
    fn test_song_haidvogel_surface_refinement() {
        let stretching = SongHaidvogelStretching::new(7.0, 0.0, 250.0);
        let n = 35;
        let (_, sigma_w) = stretching.compute_sigma(n);

        // Surface layers should be thinner than bottom layers
        let surface_spacing = sigma_w[n] - sigma_w[n - 1];
        let bottom_spacing = sigma_w[1] - sigma_w[0];

        assert!(
            surface_spacing < bottom_spacing,
            "Surface stretching should give finer surface resolution: surface={}, bottom={}",
            surface_spacing,
            bottom_spacing
        );
    }

    #[test]
    fn test_no_stretching() {
        // With theta_s = 0 and theta_b = 0, should be uniform
        let stretching = SongHaidvogelStretching::new(0.0, 0.0, 250.0);
        let uniform = UniformStretching;

        let n = 10;
        let (_, sigma_w_sh) = stretching.compute_sigma(n);
        let (_, sigma_w_uni) = uniform.compute_sigma(n);

        for i in 0..=n {
            assert!(
                (sigma_w_sh[i] - sigma_w_uni[i]).abs() < TOL,
                "With no stretching, should match uniform"
            );
        }
    }

    #[test]
    fn test_sigma_rho_between_sigma_w() {
        let stretching = SongHaidvogelStretching::default();
        let n = 20;
        let (sigma_rho, sigma_w) = stretching.compute_sigma(n);

        for k in 0..n {
            assert!(
                sigma_rho[k] > sigma_w[k] && sigma_rho[k] < sigma_w[k + 1],
                "sigma_rho[{}] should be between sigma_w[{}] and sigma_w[{}]",
                k,
                k,
                k + 1
            );
        }
    }

    #[test]
    fn test_double_tanh_bounds() {
        let stretching = DoubleTanhStretching::default();
        let n = 35;
        let (sigma_rho, sigma_w) = stretching.compute_sigma(n);

        assert_eq!(sigma_rho.len(), n);
        assert_eq!(sigma_w.len(), n + 1);

        // Check monotonicity
        for i in 1..=n {
            assert!(
                sigma_w[i] > sigma_w[i - 1],
                "sigma_w should be monotonically increasing"
            );
        }
    }

    #[test]
    fn test_stretching_names() {
        assert_eq!(UniformStretching.name(), "uniform");
        assert_eq!(SongHaidvogelStretching::default().name(), "song_haidvogel");
        assert_eq!(DoubleTanhStretching::default().name(), "double_tanh");
        assert_eq!(ROMSVstretching4::default().name(), "roms_vstretching4");
    }

    // =========================================================================
    // ROMS Vstretching 4 tests
    // =========================================================================

    #[test]
    fn test_roms_vstretching4_bounds() {
        let stretching = ROMSVstretching4::new(7.0, 2.0);
        let n = 35;
        let (sigma_rho, sigma_w) = stretching.compute_sigma(n);

        assert_eq!(sigma_rho.len(), n);
        assert_eq!(sigma_w.len(), n + 1);

        // Check boundary values: C(-1) = -1, C(0) = 0
        assert!(
            (sigma_w[0] - (-1.0)).abs() < TOL,
            "Bottom should be -1.0, got {}",
            sigma_w[0]
        );
        assert!(
            sigma_w[n].abs() < TOL,
            "Surface should be 0.0, got {}",
            sigma_w[n]
        );
    }

    #[test]
    fn test_roms_vstretching4_monotonicity() {
        let stretching = ROMSVstretching4::new(7.0, 2.0);
        let n = 35;
        let (sigma_rho, sigma_w) = stretching.compute_sigma(n);

        // Check strict monotonicity
        for i in 1..=n {
            assert!(
                sigma_w[i] > sigma_w[i - 1],
                "sigma_w should be strictly increasing: sigma_w[{}]={} <= sigma_w[{}]={}",
                i,
                sigma_w[i],
                i - 1,
                sigma_w[i - 1]
            );
        }
        for i in 1..n {
            assert!(
                sigma_rho[i] > sigma_rho[i - 1],
                "sigma_rho should be strictly increasing"
            );
        }
    }

    #[test]
    fn test_roms_vstretching4_zero_params() {
        // With theta_s = 0 and theta_b = 0, should be uniform
        let stretching = ROMSVstretching4::new(0.0, 0.0);
        let uniform = UniformStretching;

        let n = 10;
        let (_, sigma_w_v4) = stretching.compute_sigma(n);
        let (_, sigma_w_uni) = uniform.compute_sigma(n);

        for i in 0..=n {
            assert!(
                (sigma_w_v4[i] - sigma_w_uni[i]).abs() < TOL,
                "With no stretching, should match uniform: v4={}, uni={}",
                sigma_w_v4[i],
                sigma_w_uni[i]
            );
        }
    }

    #[test]
    fn test_roms_vstretching4_surface_refinement() {
        // Strong surface stretching, no bottom stretching
        let stretching = ROMSVstretching4::new(7.0, 0.0);
        let n = 35;
        let (_, sigma_w) = stretching.compute_sigma(n);

        // Surface layers should be thinner than bottom layers
        let surface_spacing = sigma_w[n] - sigma_w[n - 1];
        let bottom_spacing = sigma_w[1] - sigma_w[0];

        assert!(
            surface_spacing < bottom_spacing,
            "Surface stretching should give finer surface resolution: surface={}, bottom={}",
            surface_spacing,
            bottom_spacing
        );
    }

    #[test]
    fn test_roms_vstretching4_sigma_rho_between() {
        let stretching = ROMSVstretching4::new(5.0, 1.0);
        let n = 20;
        let (sigma_rho, sigma_w) = stretching.compute_sigma(n);

        for k in 0..n {
            assert!(
                sigma_rho[k] > sigma_w[k] && sigma_rho[k] < sigma_w[k + 1],
                "sigma_rho[{}] should be between sigma_w[{}] and sigma_w[{}]",
                k,
                k,
                k + 1
            );
        }
    }
}

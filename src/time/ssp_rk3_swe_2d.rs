//! SSP-RK3 time integration for 2D shallow water equations with limiters.
//!
//! This module provides time integration for 2D SWE that applies slope limiters
//! after each RK stage, essential for stability with:
//! - Steep bathymetry gradients
//! - Wetting/drying fronts
//! - Strong forcing (tidal, wind)
//!
//! # Example
//!
//! ```ignore
//! use dgcom::time::{SWE2DTimeConfig, ssp_rk3_swe_2d_step_limited};
//!
//! let config = SWE2DTimeConfig::new(0.3, 9.81, 0.01)
//!     .with_kuzmin_limiters(1.0);
//!
//! ssp_rk3_swe_2d_step_limited(&mut state, dt, t, &mesh, &ops, &geom, &rhs_fn, &config);
//! ```

use crate::mesh::Mesh2D;
use crate::operators::{DGOperators2D, GeometricFactors2D};
use crate::solver::{ImplicitDamping2D, KuzminParameter2D, SWESolution2D, WetDryConfig};
#[cfg(not(feature = "parallel"))]
use crate::solver::{
    apply_implicit_damping_2d as implicit_damping, apply_swe_limiters_kuzmin_2d,
    apply_wet_dry_correction_all, swe_positivity_limiter_2d,
};
#[cfg(feature = "parallel")]
use crate::solver::{
    apply_implicit_damping_2d_parallel as implicit_damping, apply_swe_limiters_kuzmin_2d_parallel,
    apply_wet_dry_correction_all_parallel, swe_positivity_limiter_2d_parallel,
};
use crate::time::{SSPRK3, StageWorkspace, TimeIntegrator};
use crate::types::Depth;

/// Type of limiter to use for 2D SWE.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum SWELimiterType {
    /// No limiting (not recommended for production)
    #[default]
    None,
    /// Kuzmin vertex-based limiter + positivity
    Kuzmin,
    /// Positivity only (no slope limiting)
    PositivityOnly,
}

/// Configuration for 2D SWE time integration with limiters.
#[derive(Clone)]
pub struct SWE2DTimeConfig {
    /// CFL number
    pub cfl: f64,
    /// Gravitational acceleration
    pub g: f64,
    /// Dry threshold (m): the positivity limiter enforces h ≥ 0 and zeroes the
    /// momentum of elements with a smaller mean depth; the wet/dry treatment
    /// desingularizes velocities below it
    pub h_min: f64,
    /// Type of limiter to apply
    pub limiter_type: SWELimiterType,
    /// Kuzmin parameter (if using Kuzmin limiter)
    pub kuzmin: KuzminParameter2D,
    /// Wetting/drying configuration (None = disabled)
    pub wet_dry: Option<WetDryConfig>,
}

impl SWE2DTimeConfig {
    /// Create a new SWE 2D time configuration.
    ///
    /// # Arguments
    /// * `cfl` - CFL number (typically 0.1-0.5)
    /// * `g` - Gravitational acceleration (m/s²)
    /// * `h_min` - Minimum water depth (m)
    pub fn new(cfl: f64, g: f64, h_min: f64) -> Self {
        Self {
            cfl,
            g,
            h_min,
            limiter_type: SWELimiterType::None,
            kuzmin: KuzminParameter2D::default(),
            wet_dry: None,
        }
    }

    /// Enable Kuzmin vertex-based limiters.
    ///
    /// # Arguments
    /// * `relaxation` - Relaxation factor (1.0 = strict, 1.1 = 10% relaxed)
    pub fn with_kuzmin_limiters(mut self, relaxation: f64) -> Self {
        self.limiter_type = SWELimiterType::Kuzmin;
        self.kuzmin = KuzminParameter2D::relaxed(relaxation);
        self
    }

    /// Enable positivity-only limiting (no slope limiting).
    pub fn with_positivity_only(mut self) -> Self {
        self.limiter_type = SWELimiterType::PositivityOnly;
        self
    }

    /// Disable all limiters.
    pub fn without_limiters(mut self) -> Self {
        self.limiter_type = SWELimiterType::None;
        self
    }

    /// Enable the wetting/drying treatment with `h_dry = h_min` (see
    /// [`WetDryConfig`]).
    ///
    /// This applies:
    /// - Positivity (h ≥ 0) and Kurganov–Petrova velocity desingularization
    ///   after every stage
    /// - Velocity capping (default 20 m/s)
    /// - Point-implicit thin-layer relaxation of the momentum in every stage
    pub fn with_wet_dry_treatment(mut self) -> Self {
        self.wet_dry = Some(WetDryConfig::new(Depth::new(self.h_min), self.g));
        self
    }

    /// Enable improved wetting/drying with custom maximum velocity.
    pub fn with_wet_dry_treatment_custom(mut self, max_velocity: f64) -> Self {
        self.wet_dry =
            Some(WetDryConfig::new(Depth::new(self.h_min), self.g).with_max_velocity(max_velocity));
        self
    }
}

/// Apply the configured limiter and wet/dry treatment to the SWE solution.
fn apply_configured_limiter(
    swe: &mut SWESolution2D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    config: &SWE2DTimeConfig,
) {
    // Apply slope/positivity limiters first
    // Use parallel versions when the feature is enabled
    #[cfg(feature = "parallel")]
    match config.limiter_type {
        SWELimiterType::None => {}
        SWELimiterType::Kuzmin => {
            apply_swe_limiters_kuzmin_2d_parallel(
                swe,
                mesh,
                ops,
                geom,
                &config.kuzmin,
                config.h_min,
            );
        }
        SWELimiterType::PositivityOnly => {
            swe_positivity_limiter_2d_parallel(swe, geom, config.h_min);
        }
    }

    #[cfg(not(feature = "parallel"))]
    match config.limiter_type {
        SWELimiterType::None => {}
        SWELimiterType::Kuzmin => {
            apply_swe_limiters_kuzmin_2d(swe, mesh, ops, geom, &config.kuzmin, config.h_min);
        }
        SWELimiterType::PositivityOnly => {
            swe_positivity_limiter_2d(swe, geom, config.h_min);
        }
    }

    // Apply wet/dry treatment (positivity, desingularization, velocity cap)
    if let Some(ref wet_dry) = config.wet_dry {
        #[cfg(feature = "parallel")]
        apply_wet_dry_correction_all_parallel(swe, geom, wet_dry);

        #[cfg(not(feature = "parallel"))]
        apply_wet_dry_correction_all(swe, geom, wet_dry);
    }
}

/// Perform one step of SSP-RK3 for 2D SWE with limiters applied after each stage.
///
/// Stage times for SSP-RK3:
/// - Stage 1: t
/// - Stage 2: t + dt
/// - Stage 3: t + dt/2
///
/// Limiters are applied after each stage to maintain stability; the wet/dry
/// thin-layer relaxation is implicit in each stage
/// (`SSPRK3::step_with_relaxation`). Allocates stage storage every call; the
/// `Simulation` path does not.
///
/// # Arguments
/// * `state` - SWE solution to update (modified in place)
/// * `dt` - Time step
/// * `t` - Current time
/// * `mesh` - 2D mesh
/// * `ops` - DG operators
/// * `geom` - Geometric factors (element means for the limiters)
/// * `rhs_fn` - Function that computes RHS given solution and time
/// * `config` - Time configuration with limiter settings
pub fn ssp_rk3_swe_2d_step_limited<F>(
    state: &mut SWESolution2D,
    dt: f64,
    t: f64,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    rhs_fn: F,
    config: &SWE2DTimeConfig,
) where
    F: Fn(&SWESolution2D, f64) -> SWESolution2D,
{
    let damping = ImplicitDamping2D {
        friction: None,
        cages: None,
        wet_dry: config.wet_dry.as_ref(),
        h_min: Depth::new(config.h_min),
    };
    SSPRK3.step_with_relaxation(
        state,
        dt,
        t,
        |q, time, out: &mut SWESolution2D| *out = rhs_fn(q, time),
        |stage, from, dt| implicit_damping(stage, from, dt, &damping),
        |q| apply_configured_limiter(q, mesh, ops, geom, config),
        &mut StageWorkspace::new(),
    );
}

/// Run a complete 2D SWE simulation with limiters.
///
/// # Arguments
/// * `state` - Initial SWE solution (modified in place during simulation)
/// * `t_end` - Final simulation time
/// * `mesh` - 2D mesh
/// * `ops` - DG operators
/// * `geom` - Geometric factors (element means for the limiters)
/// * `rhs_fn` - Function computing RHS given solution and time
/// * `dt_fn` - Function computing time step from current state
/// * `config` - Time configuration with limiter settings
/// * `callback` - Optional callback called after each step with (t, state)
///
/// # Returns
/// (final_time, number_of_steps)
pub fn run_swe_2d_simulation<F, D, C>(
    state: &mut SWESolution2D,
    t_end: f64,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    rhs_fn: F,
    dt_fn: D,
    config: &SWE2DTimeConfig,
    mut callback: Option<C>,
) -> (f64, usize)
where
    F: Fn(&SWESolution2D, f64) -> SWESolution2D,
    D: Fn(&SWESolution2D) -> f64,
    C: FnMut(f64, &SWESolution2D),
{
    let mut t = 0.0;
    let mut n_steps = 0;

    while t < t_end {
        let mut dt = dt_fn(state);

        if t + dt > t_end {
            dt = t_end - t;
        }

        ssp_rk3_swe_2d_step_limited(state, dt, t, mesh, ops, geom, &rhs_fn, config);

        t += dt;
        n_steps += 1;

        if let Some(ref mut cb) = callback {
            cb(t, state);
        }
    }

    (t, n_steps)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swe_2d_time_config_default() {
        let config = SWE2DTimeConfig::new(0.3, 9.81, 0.01);
        assert!((config.cfl - 0.3).abs() < 1e-10);
        assert!((config.g - 9.81).abs() < 1e-10);
        assert!((config.h_min - 0.01).abs() < 1e-10);
        assert_eq!(config.limiter_type, SWELimiterType::None);
    }

    #[test]
    fn test_swe_2d_time_config_with_kuzmin() {
        let config = SWE2DTimeConfig::new(0.3, 9.81, 0.01).with_kuzmin_limiters(1.1);
        assert_eq!(config.limiter_type, SWELimiterType::Kuzmin);
        assert!((config.kuzmin.relaxation - 1.1).abs() < 1e-10);
    }
}

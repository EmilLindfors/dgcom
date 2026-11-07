//! Tidal simulation configuration builder.
//!
//! Provides a convenient builder for configuring tidal simulations with
//! recommended settings for boundary conditions and sponge layers.
//!
//! # Example
//!
//! ```ignore
//! use dgcom::boundary::{TidalSimulationBuilder, SpongeConfig};
//!
//! // For closed/semi-closed basins (recommended)
//! let builder = TidalSimulationBuilder::closed_basin_stable(0.5, 50.0, 5000.0);
//! let bc = builder.build_bc();
//!
//! // For open ocean boundaries
//! let builder = TidalSimulationBuilder::open_ocean(0.5, 50.0);
//! let bc = builder.build_bc();
//! ```

use crate::boundary::tidal::TidalConstituent;
use crate::boundary::{CharacteristicOBC, HarmonicTidal2D, HarmonicTide, SWEBoundaryCondition2D};
use crate::mesh::Bathymetry2D;
use crate::solver::SWEState2D;
use crate::source::{SpongeLayer2D, SpongeProfile};

use super::bathymetry_validation::{
    BathymetryValidationConfig, BathymetryValidationResult, validate_bathymetry_convention,
};

/// Configuration for sponge layer damping zones.
#[derive(Debug, Clone)]
pub struct SpongeConfig {
    /// Width of sponge layer (m).
    pub width: f64,
    /// Maximum damping coefficient (1/s).
    pub gamma_max: f64,
    /// Damping profile shape.
    pub profile: SpongeProfile,
    /// Which boundaries have sponge layers: [left, right, bottom, top].
    pub boundaries: [bool; 4],
}

impl Default for SpongeConfig {
    fn default() -> Self {
        Self {
            width: 5000.0,
            gamma_max: 0.01,
            profile: SpongeProfile::Cosine,
            boundaries: [false, true, false, false], // Right boundary only (outflow)
        }
    }
}

impl SpongeConfig {
    /// Create a new sponge configuration.
    pub fn new(width: f64, gamma_max: f64) -> Self {
        Self {
            width,
            gamma_max,
            ..Default::default()
        }
    }

    /// Apply sponge to all boundaries.
    pub fn all_boundaries(mut self) -> Self {
        self.boundaries = [true, true, true, true];
        self
    }

    /// Set which boundaries have sponge layers.
    pub fn with_boundaries(mut self, left: bool, right: bool, bottom: bool, top: bool) -> Self {
        self.boundaries = [left, right, bottom, top];
        self
    }

    /// Set the damping profile.
    pub fn with_profile(mut self, profile: SpongeProfile) -> Self {
        self.profile = profile;
        self
    }
}

/// Type of boundary condition to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TidalBCType {
    /// Characteristic (Flather) OBC: outgoing waves leave, the tide comes in.
    Flather,
    /// Dirichlet BC for elevation (stable, doesn't absorb outgoing waves).
    Dirichlet,
}

/// Builder for tidal simulation configuration.
///
/// Provides presets for common scenarios and validates configuration.
#[derive(Debug, Clone)]
pub struct TidalSimulationBuilder {
    /// Tidal constituents.
    constituents: Vec<TidalConstituent>,
    /// Mean surface elevation (m).
    mean_elevation: f64,
    /// Reference depth (m, below MSL).
    h_ref: f64,
    /// BC type to use.
    bc_type: TidalBCType,
    /// Ramp-up duration (s).
    ramp_duration: Option<f64>,
    /// Sponge layer configuration.
    sponge_config: Option<SpongeConfig>,
    /// Minimum depth threshold.
    h_min: f64,
}

impl TidalSimulationBuilder {
    /// Create a builder with M2 tide only.
    ///
    /// # Arguments
    /// * `amplitude` - M2 tidal amplitude (m)
    /// * `phase` - M2 tidal phase (radians)
    /// * `h_ref` - Reference depth below MSL (m)
    pub fn m2(amplitude: f64, phase: f64, h_ref: f64) -> Self {
        Self {
            constituents: vec![TidalConstituent::m2(amplitude, phase)],
            mean_elevation: 0.0,
            h_ref,
            bc_type: TidalBCType::Flather,
            ramp_duration: None,
            sponge_config: None,
            h_min: 1e-6,
        }
    }

    /// Create a builder with custom constituents.
    pub fn with_constituents(constituents: Vec<TidalConstituent>, h_ref: f64) -> Self {
        Self {
            constituents,
            mean_elevation: 0.0,
            h_ref,
            bc_type: TidalBCType::Flather,
            ramp_duration: None,
            sponge_config: None,
            h_min: 1e-6,
        }
    }

    /// Preset for closed or semi-closed basins (e.g., fjords, bays).
    ///
    /// Uses Dirichlet BC (HarmonicTidal2D) to avoid velocity feedback resonance,
    /// plus sponge layer to absorb outgoing waves.
    ///
    /// # Arguments
    /// * `amplitude` - M2 amplitude (m)
    /// * `h_ref` - Reference depth (m)
    /// * `sponge_width` - Width of sponge layer (m)
    pub fn closed_basin_stable(amplitude: f64, h_ref: f64, sponge_width: f64) -> Self {
        Self {
            constituents: vec![TidalConstituent::m2(amplitude, 0.0)],
            mean_elevation: 0.0,
            h_ref,
            bc_type: TidalBCType::Dirichlet,
            ramp_duration: Some(3600.0), // 1 hour ramp
            sponge_config: Some(SpongeConfig::new(sponge_width, 0.01)),
            h_min: 1e-6,
        }
    }

    /// Preset for open ocean boundaries.
    ///
    /// Uses Flather BC (HarmonicFlather2D) for best wave absorption.
    /// Suitable when there are no reflecting walls nearby.
    pub fn open_ocean(amplitude: f64, h_ref: f64) -> Self {
        Self {
            constituents: vec![TidalConstituent::m2(amplitude, 0.0)],
            mean_elevation: 0.0,
            h_ref,
            bc_type: TidalBCType::Flather,
            ramp_duration: Some(3600.0),
            sponge_config: None,
            h_min: 1e-6,
        }
    }

    /// Preset for fjord mouth boundaries.
    ///
    /// Uses Flather BC with sponge layer for characteristic separation
    /// and absorption of reflected waves.
    pub fn fjord_mouth(amplitude: f64, h_ref: f64, sponge_width: f64) -> Self {
        Self {
            constituents: vec![TidalConstituent::m2(amplitude, 0.0)],
            mean_elevation: 0.0,
            h_ref,
            bc_type: TidalBCType::Flather,
            ramp_duration: Some(3600.0),
            sponge_config: Some(SpongeConfig::new(sponge_width, 0.005)),
            h_min: 1e-6,
        }
    }

    /// Add additional tidal constituents.
    pub fn add_constituent(mut self, constituent: TidalConstituent) -> Self {
        self.constituents.push(constituent);
        self
    }

    /// Add S2 constituent (semi-diurnal solar).
    pub fn with_s2(self, amplitude: f64, phase: f64) -> Self {
        self.add_constituent(TidalConstituent::s2(amplitude, phase))
    }

    /// Add K1 constituent (diurnal).
    pub fn with_k1(self, amplitude: f64, phase: f64) -> Self {
        self.add_constituent(TidalConstituent::k1(amplitude, phase))
    }

    /// Set mean surface elevation.
    pub fn with_mean_elevation(mut self, mean: f64) -> Self {
        self.mean_elevation = mean;
        self
    }

    /// Set BC type explicitly.
    pub fn with_bc_type(mut self, bc_type: TidalBCType) -> Self {
        self.bc_type = bc_type;
        self
    }

    /// Set ramp-up duration.
    pub fn with_ramp_up(mut self, duration: f64) -> Self {
        self.ramp_duration = Some(duration);
        self
    }

    /// Disable ramp-up.
    pub fn without_ramp_up(mut self) -> Self {
        self.ramp_duration = None;
        self
    }

    /// Set sponge layer configuration.
    pub fn with_sponge(mut self, config: SpongeConfig) -> Self {
        self.sponge_config = Some(config);
        self
    }

    /// Disable sponge layer.
    pub fn without_sponge(mut self) -> Self {
        self.sponge_config = None;
        self
    }

    /// Set minimum depth threshold.
    pub fn with_h_min(mut self, h_min: f64) -> Self {
        self.h_min = h_min;
        self
    }

    /// The configured tide.
    fn tide(&self) -> HarmonicTide {
        let tide =
            HarmonicTide::new(self.constituents.clone()).with_mean_elevation(self.mean_elevation);
        match self.ramp_duration {
            Some(duration) => tide.with_ramp_up(duration),
            None => tide,
        }
    }

    /// Build the boundary condition.
    pub fn build_bc(&self) -> Box<dyn SWEBoundaryCondition2D> {
        match self.bc_type {
            TidalBCType::Flather => Box::new(CharacteristicOBC::new(self.tide())),
            TidalBCType::Dirichlet => {
                Box::new(HarmonicTidal2D::from_tide(self.tide()).with_h_min(self.h_min))
            }
        }
    }

    /// Build sponge layer if configured.
    ///
    /// # Arguments
    /// * `x_range` - Domain extent in x: (x_min, x_max)
    /// * `y_range` - Domain extent in y: (y_min, y_max)
    ///
    /// Returns `None` if no sponge is configured.
    pub fn build_sponge(
        &self,
        x_range: (f64, f64),
        y_range: (f64, f64),
    ) -> Option<SpongeLayer2D<impl Fn(f64, f64, f64) -> SWEState2D + Clone>> {
        let config = self.sponge_config.as_ref()?;
        let h_ref = self.h_ref;

        // Reference state: calm water at rest
        let reference_state =
            move |_x: f64, _y: f64, _t: f64| SWEState2D::from_primitives(h_ref, 0.0, 0.0);

        // Domain as (x_min, x_max, y_min, y_max)
        let domain = (x_range.0, x_range.1, y_range.0, y_range.1);

        Some(
            SpongeLayer2D::rectangular(
                reference_state,
                config.gamma_max,
                config.width,
                domain,
                config.boundaries,
            )
            .with_profile(config.profile),
        )
    }

    /// Validate bathymetry configuration before simulation.
    ///
    /// Returns validation result. If invalid, suggests correct bathymetry value.
    pub fn validate_bathymetry(&self, bathymetry: &Bathymetry2D) -> BathymetryValidationResult {
        // Sample bathymetry at first element, first node
        let b_sample = if !bathymetry.data.is_empty() {
            bathymetry.data[0]
        } else {
            0.0
        };

        // Expected: bathymetry should be approximately -h_ref for surface at MSL
        let config = BathymetryValidationConfig::default();
        validate_bathymetry_convention(
            self.h_ref, // Expected water depth
            b_sample,
            self.mean_elevation, // Expected surface elevation
            &config,
        )
    }

    /// Get recommended bathymetry value for this configuration.
    pub fn recommended_bathymetry(&self) -> f64 {
        -self.h_ref
    }

    /// Get the BC type being used.
    pub fn bc_type(&self) -> TidalBCType {
        self.bc_type
    }

    /// Get the reference depth.
    pub fn h_ref(&self) -> f64 {
        self.h_ref
    }

    /// Check if sponge layer is configured.
    pub fn has_sponge(&self) -> bool {
        self.sponge_config.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_m2_builder() {
        let builder = TidalSimulationBuilder::m2(0.5, 0.0, 50.0);
        assert_eq!(builder.constituents.len(), 1);
        assert_eq!(builder.h_ref, 50.0);
        assert_eq!(builder.bc_type, TidalBCType::Flather);
    }

    #[test]
    fn test_closed_basin_preset() {
        let builder = TidalSimulationBuilder::closed_basin_stable(0.5, 50.0, 5000.0);
        assert_eq!(builder.bc_type, TidalBCType::Dirichlet);
        assert!(builder.sponge_config.is_some());
        assert!(builder.ramp_duration.is_some());
    }

    #[test]
    fn test_open_ocean_preset() {
        let builder = TidalSimulationBuilder::open_ocean(0.5, 50.0);
        assert_eq!(builder.bc_type, TidalBCType::Flather);
        assert!(builder.sponge_config.is_none());
    }

    #[test]
    fn test_add_constituents() {
        let builder = TidalSimulationBuilder::m2(0.5, 0.0, 50.0)
            .with_s2(0.2, 0.0)
            .with_k1(0.1, 0.0);
        assert_eq!(builder.constituents.len(), 3);
    }

    #[test]
    fn test_build_bc_flather() {
        let builder = TidalSimulationBuilder::m2(0.5, 0.0, 50.0);
        let bc = builder.build_bc();
        assert_eq!(bc.name(), "characteristic_obc");
    }

    #[test]
    fn test_build_bc_dirichlet() {
        let builder =
            TidalSimulationBuilder::m2(0.5, 0.0, 50.0).with_bc_type(TidalBCType::Dirichlet);
        let bc = builder.build_bc();
        assert_eq!(bc.name(), "harmonic_tidal_2d");
    }

    #[test]
    fn test_recommended_bathymetry() {
        let builder = TidalSimulationBuilder::m2(0.5, 0.0, 50.0);
        assert!((builder.recommended_bathymetry() - (-50.0)).abs() < 1e-10);
    }
}

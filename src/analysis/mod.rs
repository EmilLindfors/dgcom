//! Tidal harmonic analysis, comparison tools, and tide gauge validation.
//!
//! This module provides tools for:
//! - Decomposing tidal time series into constituent harmonics using least-squares fitting
//! - Comparing model results against observations
//! - Validating model output against tide gauge observations
//!
//! # Harmonic analysis
//!
//! [`HarmonicAnalysis`] fits tidal reference constants (amplitude `H`,
//! Greenwich phase lag `G`) by least squares on an absolute (Unix) time
//! axis, the astronomical arguments and nodal corrections in the basis:
//!
//! ```text
//! η(t) = Z₀ + Σⱼ fⱼ Hⱼ cos(ωⱼ (t − t₀) + V₀ⱼ + uⱼ − Gⱼ)
//! ```
//!
//! Pairs the record cannot separate (Rayleigh criterion) are an error;
//! [`HarmonicAnalysis::resolvable`] picks what a record resolves and infers
//! the rest ([`Inference`]). Failures are [`AnalysisError`]s.
//!
//! ```ignore
//! use dgcom::analysis::{HarmonicAnalysis, Inference};
//!
//! // Hourly water levels at Unix times
//! let analysis = HarmonicAnalysis::resolvable(
//!     &["M2", "S2", "N2", "K1", "O1", "M4"],
//!     times[times.len() - 1] - times[0],
//!     1.0,
//!     &Inference::EQUILIBRIUM,
//! );
//! let fit = analysis.fit(&times, &values)?;
//! let m2 = fit.get("M2").unwrap();
//! println!("M2: H {:.3} m, G {:.1}°", m2.amplitude, m2.lag_deg);
//! ```
//!
//! # Example - Tide Gauge Validation
//!
//! ```ignore
//! use dgcom::analysis::{TideGaugeStation, StationValidationResult, TimeSeries};
//!
//! let station = TideGaugeStation::new("Bergen", 5.32, 60.39);
//! let model = TimeSeries::new(&times, &model_values);
//! let obs = TimeSeries::new(&times, &obs_values);
//!
//! let result = StationValidationResult::compute(&station, &model, &obs)?;
//! println!("RMSE: {:.3} m", result.metrics.rmse);
//! println!("Skill: {:.2}", result.metrics.skill_score);
//! ```

mod adcp;
mod ellipse;
mod error;
mod harmonic;
mod metrics;
mod stability;
mod tide_gauge;

pub use adcp::{
    ADCPStation, ADCPValidationResult, ADCPValidationSummary, CurrentPoint, CurrentTimeSeries,
    CurrentValidationMetrics, norwegian_stations as adcp_norwegian_stations,
};
pub use ellipse::{EllipseFit, TidalEllipse};
pub use error::AnalysisError;
pub use harmonic::{HarmonicAnalysis, Inference, ReferenceConstant, ReferenceFit};
pub use metrics::{
    ComparisonMetrics, ConstituentComparison, ConstituentComparisonSummary, compare_harmonics,
};
pub use stability::{StabilityMonitor, StabilityStatus, StabilityThresholds, StabilityWarning};
pub use tide_gauge::{
    GaugeComparison, GaugeRecord, ModelExtractor, PrecomputedExtractor, SkillScore,
    StationValidationResult, TideGaugeStation, ValidationSummary, norwegian_stations,
    validate_stations,
};

use std::f64::consts::PI;

/// A time series: times (s, often Unix) and values, as two arrays, with an
/// optional location and name.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TimeSeries {
    times: Vec<f64>,
    values: Vec<f64>,
    /// Optional (x, y) location coordinates
    pub location: Option<(f64, f64)>,
    /// Optional name/identifier
    pub name: Option<String>,
}

impl TimeSeries {
    /// A time series from parallel arrays of times and values.
    ///
    /// # Panics
    ///
    /// Panics if `times` and `values` have different lengths.
    pub fn new(times: &[f64], values: &[f64]) -> Self {
        Self::from_vecs(times.to_vec(), values.to_vec())
    }

    /// A time series that takes ownership of its arrays.
    ///
    /// # Panics
    ///
    /// Panics if `times` and `values` have different lengths.
    pub fn from_vecs(times: Vec<f64>, values: Vec<f64>) -> Self {
        assert_eq!(
            times.len(),
            values.len(),
            "times and values must have same length"
        );
        Self {
            times,
            values,
            location: None,
            name: None,
        }
    }

    /// Create a time series with location information.
    pub fn with_location(mut self, x: f64, y: f64) -> Self {
        self.location = Some((x, y));
        self
    }

    /// Create a time series with a name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Number of data points.
    pub fn len(&self) -> usize {
        self.times.len()
    }

    /// Check if the time series is empty.
    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }

    /// Duration of the time series (last time - first time).
    pub fn duration(&self) -> f64 {
        match (self.times.first(), self.times.last()) {
            (Some(first), Some(last)) => last - first,
            _ => 0.0,
        }
    }

    /// The times.
    pub fn times(&self) -> &[f64] {
        &self.times
    }

    /// The values.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// The same times with each value mapped by `f`.
    pub fn map_values(&self, f: impl Fn(f64) -> f64) -> Self {
        Self {
            times: self.times.clone(),
            values: self.values.iter().map(|&v| f(v)).collect(),
            location: self.location,
            name: self.name.clone(),
        }
    }

    /// Compute the mean value.
    pub fn mean(&self) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }
        self.values.iter().sum::<f64>() / self.values.len() as f64
    }

    /// Compute the variance.
    pub fn variance(&self) -> f64 {
        if self.values.len() < 2 {
            return 0.0;
        }
        let mean = self.mean();
        let sum_sq: f64 = self.values.iter().map(|v| (v - mean).powi(2)).sum();
        sum_sq / (self.values.len() - 1) as f64
    }

    /// Compute the standard deviation.
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
}

/// Wrap a phase angle to the range [0, 2π).
pub fn wrap_phase(phase: f64) -> f64 {
    let mut p = phase % (2.0 * PI);
    if p < 0.0 {
        p += 2.0 * PI;
    }
    p
}

/// Compute phase difference wrapped to [-π, π].
pub fn phase_difference(phase1: f64, phase2: f64) -> f64 {
    let diff = phase1 - phase2;
    let mut wrapped = diff % (2.0 * PI);
    if wrapped > PI {
        wrapped -= 2.0 * PI;
    } else if wrapped < -PI {
        wrapped += 2.0 * PI;
    }
    wrapped
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_series_creation() {
        let times = vec![0.0, 1.0, 2.0, 3.0];
        let values = vec![1.0, 2.0, 1.5, 2.5];
        let ts = TimeSeries::new(&times, &values);

        assert_eq!(ts.len(), 4);
        assert!((ts.duration() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_time_series_statistics() {
        let times = vec![0.0, 1.0, 2.0, 3.0];
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let ts = TimeSeries::new(&times, &values);

        assert!((ts.mean() - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_wrap_phase() {
        assert!((wrap_phase(0.0) - 0.0).abs() < 1e-10);
        assert!((wrap_phase(PI) - PI).abs() < 1e-10);
        assert!((wrap_phase(-PI) - PI).abs() < 1e-10);
        assert!((wrap_phase(3.0 * PI) - PI).abs() < 1e-10);
    }

    #[test]
    fn test_phase_difference() {
        assert!((phase_difference(0.1, 0.0) - 0.1).abs() < 1e-10);
        assert!((phase_difference(0.0, 0.1) - (-0.1)).abs() < 1e-10);

        // Wrap around
        let diff = phase_difference(0.1, 2.0 * PI - 0.1);
        assert!((diff - 0.2).abs() < 1e-10);
    }
}

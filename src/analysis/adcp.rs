//! ADCP (Acoustic Doppler Current Profiler) validation for comparing model currents against observations.
//!
//! This module provides tools for validating 2D shallow water model velocity results
//! against ADCP current observations.
//!
//! # Features
//!
//! - Station metadata management (location, name, depth)
//! - Model velocity extraction at station locations
//! - Time series comparison for u and v components (RMSE, bias, correlation)
//! - Vector validation (speed, direction errors)
//! - Multi-station validation summaries
//!
//! # Example
//!
//! ```ignore
//! use dgcom::analysis::{ADCPStation, ADCPValidationResult, CurrentTimeSeries};
//!
//! // Define station
//! let station = ADCPStation::new("Froya_ADCP", 8.85, 63.75)
//!     .with_depth(25.0);
//!
//! // Create observed current time series
//! let obs = CurrentTimeSeries::new(&times, &u_obs, &v_obs)
//!     .with_name("Froya_ADCP");
//!
//! // Create model time series (extracted at station location)
//! let model = CurrentTimeSeries::new(&times, &u_model, &v_model)
//!     .with_name("Froya_ADCP");
//!
//! // Validate
//! let result = ADCPValidationResult::compute(&station, &model, &obs);
//! println!("Speed RMSE: {:.3} m/s", result.speed_metrics.rmse);
//! println!("Direction RMSE: {:.1}°", result.direction_rmse_degrees);
//! ```

use super::ComparisonMetrics;
use std::f64::consts::PI;

/// Metadata for an ADCP current measurement station.
///
/// Stores location, name, and depth information.
#[derive(Clone, Debug)]
pub struct ADCPStation {
    /// Station identifier/name
    pub name: String,
    /// Longitude (degrees East)
    pub longitude: f64,
    /// Latitude (degrees North)
    pub latitude: f64,
    /// Local x-coordinate in model domain (computed from projection)
    pub x: Option<f64>,
    /// Local y-coordinate in model domain (computed from projection)
    pub y: Option<f64>,
    /// Instrument depth (m below surface, positive downward)
    pub depth: Option<f64>,
    /// Water depth at station (m)
    pub water_depth: Option<f64>,
    /// Station ID in external database
    pub external_id: Option<String>,
}

impl ADCPStation {
    /// Create a new ADCP station.
    ///
    /// # Arguments
    /// * `name` - Station name/identifier
    /// * `longitude` - Longitude in degrees East
    /// * `latitude` - Latitude in degrees North
    pub fn new(name: impl Into<String>, longitude: f64, latitude: f64) -> Self {
        Self {
            name: name.into(),
            longitude,
            latitude,
            x: None,
            y: None,
            depth: None,
            water_depth: None,
            external_id: None,
        }
    }

    /// Set local model coordinates.
    pub fn with_local_coords(mut self, x: f64, y: f64) -> Self {
        self.x = Some(x);
        self.y = Some(y);
        self
    }

    /// Set instrument depth (meters below surface).
    pub fn with_depth(mut self, depth: f64) -> Self {
        self.depth = Some(depth);
        self
    }

    /// Set water depth at station.
    pub fn with_water_depth(mut self, water_depth: f64) -> Self {
        self.water_depth = Some(water_depth);
        self
    }

    /// Set external database ID.
    pub fn with_external_id(mut self, id: impl Into<String>) -> Self {
        self.external_id = Some(id.into());
        self
    }

    /// Check if local coordinates are set.
    pub fn has_local_coords(&self) -> bool {
        self.x.is_some() && self.y.is_some()
    }

    /// Get local coordinates (x, y).
    ///
    /// # Panics
    /// Panics if local coordinates are not set.
    pub fn local_coords(&self) -> (f64, f64) {
        (
            self.x.expect("Local x-coordinate not set"),
            self.y.expect("Local y-coordinate not set"),
        )
    }
}

/// A single current measurement point (u, v at time t).
#[derive(Clone, Copy, Debug)]
pub struct CurrentPoint {
    /// Time in seconds
    pub time: f64,
    /// Eastward velocity component (m/s)
    pub u: f64,
    /// Northward velocity component (m/s)
    pub v: f64,
}

impl CurrentPoint {
    /// Create a new current measurement point.
    pub fn new(time: f64, u: f64, v: f64) -> Self {
        Self { time, u, v }
    }

    /// Compute current speed.
    #[inline]
    pub fn speed(&self) -> f64 {
        (self.u * self.u + self.v * self.v).sqrt()
    }

    /// Compute current direction in radians (0 = East, π/2 = North).
    #[inline]
    pub fn direction(&self) -> f64 {
        self.v.atan2(self.u)
    }

    /// Compute current direction in degrees (0 = East, 90 = North).
    #[inline]
    pub fn direction_degrees(&self) -> f64 {
        self.direction().to_degrees()
    }

    /// Compute current direction in oceanographic convention (0 = North, 90 = East).
    #[inline]
    pub fn direction_oceanographic(&self) -> f64 {
        let dir = 90.0 - self.direction_degrees();
        if dir < 0.0 {
            dir + 360.0
        } else if dir >= 360.0 {
            dir - 360.0
        } else {
            dir
        }
    }
}

/// Time series of current velocity measurements.
///
/// Stores (time, u, v) triplets with optional location and name.
#[derive(Clone, Debug)]
pub struct CurrentTimeSeries {
    /// The current data points
    pub data: Vec<CurrentPoint>,
    /// Optional (x, y) location coordinates
    pub location: Option<(f64, f64)>,
    /// Optional name/identifier
    pub name: Option<String>,
}

impl CurrentTimeSeries {
    /// Create a new current time series from parallel arrays.
    ///
    /// # Arguments
    /// * `times` - Time values in seconds
    /// * `u` - Eastward velocity component (m/s)
    /// * `v` - Northward velocity component (m/s)
    ///
    /// # Panics
    /// Panics if arrays have different lengths.
    pub fn new(times: &[f64], u: &[f64], v: &[f64]) -> Self {
        assert_eq!(times.len(), u.len(), "times and u must have same length");
        assert_eq!(times.len(), v.len(), "times and v must have same length");

        let data = times
            .iter()
            .zip(u.iter())
            .zip(v.iter())
            .map(|((&t, &u), &v)| CurrentPoint::new(t, u, v))
            .collect();

        Self {
            data,
            location: None,
            name: None,
        }
    }

    /// Set location coordinates.
    pub fn with_location(mut self, x: f64, y: f64) -> Self {
        self.location = Some((x, y));
        self
    }

    /// Set name/identifier.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Number of data points.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Duration of the time series (last time - first time).
    pub fn duration(&self) -> f64 {
        if self.data.is_empty() {
            return 0.0;
        }
        self.data.last().unwrap().time - self.data.first().unwrap().time
    }

    /// Get times as a vector.
    pub fn times(&self) -> Vec<f64> {
        self.data.iter().map(|p| p.time).collect()
    }

    /// Get u-component as a vector.
    pub fn u_values(&self) -> Vec<f64> {
        self.data.iter().map(|p| p.u).collect()
    }

    /// Get v-component as a vector.
    pub fn v_values(&self) -> Vec<f64> {
        self.data.iter().map(|p| p.v).collect()
    }

    /// Get speeds as a vector.
    pub fn speeds(&self) -> Vec<f64> {
        self.data.iter().map(|p| p.speed()).collect()
    }

    /// Get directions as a vector (radians, mathematical convention).
    pub fn directions(&self) -> Vec<f64> {
        self.data.iter().map(|p| p.direction()).collect()
    }

    /// Mean u-component.
    pub fn mean_u(&self) -> f64 {
        if self.data.is_empty() {
            return 0.0;
        }
        self.data.iter().map(|p| p.u).sum::<f64>() / self.data.len() as f64
    }

    /// Mean v-component.
    pub fn mean_v(&self) -> f64 {
        if self.data.is_empty() {
            return 0.0;
        }
        self.data.iter().map(|p| p.v).sum::<f64>() / self.data.len() as f64
    }

    /// Mean speed.
    pub fn mean_speed(&self) -> f64 {
        if self.data.is_empty() {
            return 0.0;
        }
        self.data.iter().map(|p| p.speed()).sum::<f64>() / self.data.len() as f64
    }

    /// The observations `obs` that lie inside the time span of `model`
    /// (sorted times), and the model linearly interpolated to their times:
    /// two series on one time axis, for [`CurrentValidationMetrics`].
    pub fn paired_by_time(model: &Self, obs: &Self) -> (Self, Self) {
        let (Some(first), Some(last)) = (model.data.first(), model.data.last()) else {
            return (Self::new(&[], &[], &[]), Self::new(&[], &[], &[]));
        };
        let (first, last) = (first.time, last.time);
        let (paired_model, paired_obs): (Vec<CurrentPoint>, Vec<CurrentPoint>) = obs
            .data
            .iter()
            .filter(|o| (first..=last).contains(&o.time))
            .map(|o| {
                let i = model.data.partition_point(|m| m.time < o.time);
                let at = if model.data[i].time == o.time || i == 0 {
                    model.data[i]
                } else {
                    let (a, b) = (model.data[i - 1], model.data[i]);
                    let w = (o.time - a.time) / (b.time - a.time);
                    CurrentPoint::new(o.time, a.u + w * (b.u - a.u), a.v + w * (b.v - a.v))
                };
                (CurrentPoint { time: o.time, ..at }, *o)
            })
            .unzip();
        let with = |data, source: &Self| Self {
            data,
            location: source.location,
            name: source.name.clone(),
        };
        (with(paired_model, model), with(paired_obs, obs))
    }
}

/// Validation metrics for current velocity comparison.
///
/// Includes component-wise metrics and vector metrics.
#[derive(Clone, Debug)]
pub struct CurrentValidationMetrics {
    /// Metrics for u-component (eastward velocity)
    pub u_metrics: ComparisonMetrics,
    /// Metrics for v-component (northward velocity)
    pub v_metrics: ComparisonMetrics,
    /// Metrics for speed (magnitude)
    pub speed_metrics: ComparisonMetrics,
    /// Vector correlation coefficient (complex correlation)
    pub vector_correlation: f64,
    /// Mean direction error (degrees, signed)
    pub mean_direction_error: f64,
    /// RMSE of direction error (degrees)
    pub direction_rmse: f64,
    /// Mean angular deviation (unsigned direction error, degrees)
    pub mean_angular_deviation: f64,
}

impl CurrentValidationMetrics {
    /// Compute validation metrics between model and observed currents.
    ///
    /// # Arguments
    /// * `model` - Model current time series
    /// * `obs` - Observed current time series
    ///
    /// # Panics
    /// Panics if time series have different lengths or are empty.
    pub fn compute(model: &CurrentTimeSeries, obs: &CurrentTimeSeries) -> Self {
        assert_eq!(
            model.len(),
            obs.len(),
            "Model and observation must have same length"
        );
        assert!(!model.is_empty(), "Time series must not be empty");

        let n = model.len();

        // Component-wise metrics
        let u_model = model.u_values();
        let u_obs = obs.u_values();
        let u_metrics = ComparisonMetrics::compute(&u_model, &u_obs);

        let v_model = model.v_values();
        let v_obs = obs.v_values();
        let v_metrics = ComparisonMetrics::compute(&v_model, &v_obs);

        // Speed metrics
        let speed_model = model.speeds();
        let speed_obs = obs.speeds();
        let speed_metrics = ComparisonMetrics::compute(&speed_model, &speed_obs);

        // Vector correlation (complex correlation coefficient)
        // ρ = |<model, obs*>| / sqrt(<model, model*><obs, obs*>)
        // where complex velocity w = u + iv
        let mut sum_model_obs_real = 0.0;
        let mut sum_model_obs_imag = 0.0;
        let mut sum_model_sq = 0.0;
        let mut sum_obs_sq = 0.0;

        for i in 0..n {
            let m = &model.data[i];
            let o = &obs.data[i];

            // model * conj(obs) = (u_m + i*v_m)(u_o - i*v_o) = (u_m*u_o + v_m*v_o) + i(v_m*u_o - u_m*v_o)
            sum_model_obs_real += m.u * o.u + m.v * o.v;
            sum_model_obs_imag += m.v * o.u - m.u * o.v;

            sum_model_sq += m.u * m.u + m.v * m.v;
            sum_obs_sq += o.u * o.u + o.v * o.v;
        }

        let vector_correlation = if sum_model_sq > 1e-10 && sum_obs_sq > 1e-10 {
            let inner_prod_mag = (sum_model_obs_real * sum_model_obs_real
                + sum_model_obs_imag * sum_model_obs_imag)
                .sqrt();
            inner_prod_mag / (sum_model_sq.sqrt() * sum_obs_sq.sqrt())
        } else {
            0.0
        };

        // Direction errors
        let mut direction_errors = Vec::with_capacity(n);
        let mut valid_dir_count = 0;

        for i in 0..n {
            let m = &model.data[i];
            let o = &obs.data[i];

            // Only compute direction error for non-zero velocities
            let m_speed = m.speed();
            let o_speed = o.speed();

            if m_speed > 0.01 && o_speed > 0.01 {
                // Speed threshold 1 cm/s
                let m_dir = m.direction();
                let o_dir = o.direction();

                // Wrap difference to [-π, π]
                let mut diff = m_dir - o_dir;
                if diff > PI {
                    diff -= 2.0 * PI;
                } else if diff < -PI {
                    diff += 2.0 * PI;
                }

                direction_errors.push(diff);
                valid_dir_count += 1;
            }
        }

        let (mean_direction_error, direction_rmse, mean_angular_deviation) = if valid_dir_count > 0
        {
            let mean_err =
                direction_errors.iter().sum::<f64>() / valid_dir_count as f64 * 180.0 / PI;

            let mse = direction_errors.iter().map(|e| e * e).sum::<f64>() / valid_dir_count as f64;
            let rmse_deg = mse.sqrt() * 180.0 / PI;

            let mad = direction_errors.iter().map(|e| e.abs()).sum::<f64>()
                / valid_dir_count as f64
                * 180.0
                / PI;

            (mean_err, rmse_deg, mad)
        } else {
            (0.0, 0.0, 0.0)
        };

        Self {
            u_metrics,
            v_metrics,
            speed_metrics,
            vector_correlation,
            mean_direction_error,
            direction_rmse,
            mean_angular_deviation,
        }
    }
}

/// Validation result for a single ADCP station.
#[derive(Clone, Debug)]
pub struct ADCPValidationResult {
    /// Station metadata
    pub station: ADCPStation,
    /// Current validation metrics
    pub metrics: CurrentValidationMetrics,
    /// Model mean velocity (u, v)
    pub model_mean: (f64, f64),
    /// Observed mean velocity (u, v)
    pub obs_mean: (f64, f64),
    /// Model mean speed (m/s)
    pub model_mean_speed: f64,
    /// Observed mean speed (m/s)
    pub obs_mean_speed: f64,
}

impl ADCPValidationResult {
    /// Compute validation result from model and observation time series.
    pub fn compute(
        station: &ADCPStation,
        model: &CurrentTimeSeries,
        obs: &CurrentTimeSeries,
    ) -> Self {
        let (model, obs) = &CurrentTimeSeries::paired_by_time(model, obs);
        assert!(
            model.len() >= 2,
            "{}: fewer than two observations inside the model's time span",
            station.name
        );
        let metrics = CurrentValidationMetrics::compute(model, obs);

        Self {
            station: station.clone(),
            metrics,
            model_mean: (model.mean_u(), model.mean_v()),
            obs_mean: (obs.mean_u(), obs.mean_v()),
            model_mean_speed: model.mean_speed(),
            obs_mean_speed: obs.mean_speed(),
        }
    }

    /// Check if model passes basic validation criteria.
    ///
    /// Criteria:
    /// - Speed correlation > 0.8
    /// - Vector correlation > 0.7
    /// - Speed RMSE < 0.15 m/s
    /// - Direction RMSE < 30°
    pub fn passes_basic_validation(&self) -> bool {
        self.metrics.speed_metrics.correlation > 0.8
            && self.metrics.vector_correlation > 0.7
            && self.metrics.speed_metrics.rmse < 0.15
            && self.metrics.direction_rmse < 30.0
    }

    /// Check if model passes strict validation criteria.
    ///
    /// Criteria:
    /// - Speed correlation > 0.9
    /// - Vector correlation > 0.85
    /// - Speed RMSE < 0.10 m/s
    /// - Direction RMSE < 20°
    pub fn passes_strict_validation(&self) -> bool {
        self.metrics.speed_metrics.correlation > 0.9
            && self.metrics.vector_correlation > 0.85
            && self.metrics.speed_metrics.rmse < 0.10
            && self.metrics.direction_rmse < 20.0
    }

    /// Format validation result as a summary string.
    pub fn summary(&self) -> String {
        format!(
            "Station: {}\n\
             Location: ({:.2}°E, {:.2}°N)\n\
             --------------------------\n\
             Component Metrics:\n\
             U RMSE:      {:.4} m/s\n\
             U Bias:      {:.4} m/s\n\
             U Corr:      {:.4}\n\
             V RMSE:      {:.4} m/s\n\
             V Bias:      {:.4} m/s\n\
             V Corr:      {:.4}\n\
             --------------------------\n\
             Speed Metrics:\n\
             RMSE:        {:.4} m/s\n\
             Bias:        {:.4} m/s\n\
             Correlation: {:.4}\n\
             Skill Score: {:.4}\n\
             --------------------------\n\
             Vector Metrics:\n\
             Vector Corr: {:.4}\n\
             Dir RMSE:    {:.1}°\n\
             Mean Dir Err:{:.1}°\n\
             --------------------------\n\
             Mean Values:\n\
             Model:       ({:.3}, {:.3}) m/s, {:.3} m/s\n\
             Observed:    ({:.3}, {:.3}) m/s, {:.3} m/s\n",
            self.station.name,
            self.station.longitude,
            self.station.latitude,
            self.metrics.u_metrics.rmse,
            self.metrics.u_metrics.bias,
            self.metrics.u_metrics.correlation,
            self.metrics.v_metrics.rmse,
            self.metrics.v_metrics.bias,
            self.metrics.v_metrics.correlation,
            self.metrics.speed_metrics.rmse,
            self.metrics.speed_metrics.bias,
            self.metrics.speed_metrics.correlation,
            self.metrics.speed_metrics.skill_score,
            self.metrics.vector_correlation,
            self.metrics.direction_rmse,
            self.metrics.mean_direction_error,
            self.model_mean.0,
            self.model_mean.1,
            self.model_mean_speed,
            self.obs_mean.0,
            self.obs_mean.1,
            self.obs_mean_speed,
        )
    }
}

/// Summary of validation results across multiple ADCP stations.
#[derive(Clone, Debug)]
pub struct ADCPValidationSummary {
    /// Individual station results
    pub station_results: Vec<ADCPValidationResult>,
    /// Mean speed RMSE across stations (m/s)
    pub mean_speed_rmse: f64,
    /// Mean vector correlation
    pub mean_vector_correlation: f64,
    /// Mean direction RMSE (degrees)
    pub mean_direction_rmse: f64,
    /// Number of stations passing basic validation
    pub n_passing_basic: usize,
    /// Number of stations passing strict validation
    pub n_passing_strict: usize,
}

impl ADCPValidationSummary {
    /// Create a summary from individual station results.
    pub fn from_results(results: Vec<ADCPValidationResult>) -> Self {
        let n = results.len();

        if n == 0 {
            return Self {
                station_results: results,
                mean_speed_rmse: 0.0,
                mean_vector_correlation: 0.0,
                mean_direction_rmse: 0.0,
                n_passing_basic: 0,
                n_passing_strict: 0,
            };
        }

        let mean_speed_rmse = results
            .iter()
            .map(|r| r.metrics.speed_metrics.rmse)
            .sum::<f64>()
            / n as f64;

        let mean_vector_correlation = results
            .iter()
            .map(|r| r.metrics.vector_correlation)
            .sum::<f64>()
            / n as f64;

        let mean_direction_rmse = results
            .iter()
            .map(|r| r.metrics.direction_rmse)
            .sum::<f64>()
            / n as f64;

        let n_passing_basic = results
            .iter()
            .filter(|r| r.passes_basic_validation())
            .count();
        let n_passing_strict = results
            .iter()
            .filter(|r| r.passes_strict_validation())
            .count();

        Self {
            station_results: results,
            mean_speed_rmse,
            mean_vector_correlation,
            mean_direction_rmse,
            n_passing_basic,
            n_passing_strict,
        }
    }

    /// Format summary as string.
    pub fn summary(&self) -> String {
        format!(
            "ADCP Validation Summary ({} stations)\n\
             ================================\n\
             Mean Speed RMSE:       {:.4} m/s\n\
             Mean Vector Corr:      {:.4}\n\
             Mean Direction RMSE:   {:.1}°\n\
             Passing Basic:         {}/{}\n\
             Passing Strict:        {}/{}\n",
            self.station_results.len(),
            self.mean_speed_rmse,
            self.mean_vector_correlation,
            self.mean_direction_rmse,
            self.n_passing_basic,
            self.station_results.len(),
            self.n_passing_strict,
            self.station_results.len(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adcp_station_creation() {
        let station = ADCPStation::new("Test", 8.5, 63.5)
            .with_depth(20.0)
            .with_water_depth(50.0)
            .with_local_coords(1000.0, 2000.0);

        assert_eq!(station.name, "Test");
        assert_eq!(station.longitude, 8.5);
        assert_eq!(station.latitude, 63.5);
        assert_eq!(station.depth, Some(20.0));
        assert_eq!(station.water_depth, Some(50.0));
        assert!(station.has_local_coords());
        assert_eq!(station.local_coords(), (1000.0, 2000.0));
    }

    #[test]
    fn test_current_point() {
        let p = CurrentPoint::new(0.0, 1.0, 0.0);
        assert!((p.speed() - 1.0).abs() < 1e-10);
        assert!(p.direction().abs() < 1e-10); // 0 = East

        let p2 = CurrentPoint::new(0.0, 0.0, 1.0);
        assert!((p2.speed() - 1.0).abs() < 1e-10);
        assert!((p2.direction() - PI / 2.0).abs() < 1e-10); // π/2 = North

        let p3 = CurrentPoint::new(0.0, 1.0, 1.0);
        assert!((p3.speed() - 2.0_f64.sqrt()).abs() < 1e-10);
        assert!((p3.direction() - PI / 4.0).abs() < 1e-10); // 45°
    }

    #[test]
    fn test_current_time_series() {
        let times = vec![0.0, 3600.0, 7200.0];
        let u = vec![1.0, 0.5, 0.0];
        let v = vec![0.0, 0.5, 1.0];

        let ts = CurrentTimeSeries::new(&times, &u, &v).with_name("Test");

        assert_eq!(ts.len(), 3);
        assert!((ts.duration() - 7200.0).abs() < 1e-10);
        assert!((ts.mean_u() - 0.5).abs() < 1e-10);
        assert!((ts.mean_v() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_identical_currents_perfect_metrics() {
        let times = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let u = vec![0.5, 0.3, 0.1, -0.1, 0.2];
        let v = vec![0.1, 0.2, 0.3, 0.2, 0.1];

        let model = CurrentTimeSeries::new(&times, &u, &v);
        let obs = CurrentTimeSeries::new(&times, &u, &v);

        let metrics = CurrentValidationMetrics::compute(&model, &obs);

        assert!(metrics.u_metrics.rmse < 1e-10);
        assert!(metrics.v_metrics.rmse < 1e-10);
        assert!(metrics.speed_metrics.rmse < 1e-10);
        assert!((metrics.u_metrics.correlation - 1.0).abs() < 1e-10);
        assert!((metrics.vector_correlation - 1.0).abs() < 1e-10);
        assert!(metrics.direction_rmse < 1e-10);
    }

    #[test]
    fn test_opposite_currents_negative_correlation() {
        let times = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let u1 = vec![0.5, 0.3, 0.1, -0.1, 0.2];
        let v1 = vec![0.1, 0.2, 0.3, 0.2, 0.1];
        let u2: Vec<f64> = u1.iter().map(|&x| -x).collect();
        let v2: Vec<f64> = v1.iter().map(|&x| -x).collect();

        let model = CurrentTimeSeries::new(&times, &u1, &v1);
        let obs = CurrentTimeSeries::new(&times, &u2, &v2);

        let metrics = CurrentValidationMetrics::compute(&model, &obs);

        // Speeds should be the same (same magnitude)
        assert!(metrics.speed_metrics.rmse < 1e-10);

        // But components are opposite → correlation = -1
        assert!((metrics.u_metrics.correlation - (-1.0)).abs() < 1e-10);
        assert!((metrics.v_metrics.correlation - (-1.0)).abs() < 1e-10);

        // Direction error should be ~180° for all points
        assert!((metrics.direction_rmse - 180.0).abs() < 1.0);
    }

    #[test]
    fn test_validation_result() {
        let station = ADCPStation::new("Test", 8.5, 63.5);

        let times = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let u = vec![0.5, 0.3, 0.1, -0.1, 0.2];
        let v = vec![0.1, 0.2, 0.3, 0.2, 0.1];

        let model = CurrentTimeSeries::new(&times, &u, &v);
        let obs = CurrentTimeSeries::new(&times, &u, &v);

        let result = ADCPValidationResult::compute(&station, &model, &obs);

        assert!(result.passes_basic_validation());
        assert!(result.passes_strict_validation());

        let summary = result.summary();
        assert!(summary.contains("Test"));
        assert!(summary.contains("U RMSE"));
    }

    /// Observations between and outside the model's samples: the model is
    /// interpolated to the observation times inside its span, the rest are
    /// dropped.
    #[test]
    fn validation_pairs_by_time() {
        let model = CurrentTimeSeries::new(&[0.0, 10.0, 20.0], &[0.0, 1.0, 3.0], &[1.0, 1.0, -1.0]);
        let obs = CurrentTimeSeries::new(
            &[-5.0, 0.0, 5.0, 15.0, 20.0, 25.0],
            &[9.0, 0.0, 0.5, 2.0, 3.0, 9.0],
            &[9.0, 1.0, 1.0, 0.0, -1.0, 9.0],
        );
        let (m, o) = CurrentTimeSeries::paired_by_time(&model, &obs);
        assert_eq!(m.times(), vec![0.0, 5.0, 15.0, 20.0]);
        assert_eq!(o.times(), m.times());
        assert_eq!(m.u_values(), vec![0.0, 0.5, 2.0, 3.0]);
        assert_eq!(m.v_values(), vec![1.0, 1.0, 0.0, -1.0]);
        let result = ADCPValidationResult::compute(&ADCPStation::new("T", 0.0, 0.0), &model, &obs);
        assert!(result.metrics.u_metrics.rmse < 1e-15 && result.metrics.v_metrics.rmse < 1e-15);
    }

    #[test]
    fn test_validation_summary() {
        let station1 = ADCPStation::new("Station1", 8.5, 63.5);
        let station2 = ADCPStation::new("Station2", 9.0, 64.0);

        let times = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let u = vec![0.5, 0.3, 0.1, -0.1, 0.2];
        let v = vec![0.1, 0.2, 0.3, 0.2, 0.1];

        let model = CurrentTimeSeries::new(&times, &u, &v);
        let obs = CurrentTimeSeries::new(&times, &u, &v);

        let result1 = ADCPValidationResult::compute(&station1, &model, &obs);
        let result2 = ADCPValidationResult::compute(&station2, &model, &obs);

        let summary = ADCPValidationSummary::from_results(vec![result1, result2]);

        assert_eq!(summary.station_results.len(), 2);
        assert_eq!(summary.n_passing_basic, 2);
        assert_eq!(summary.n_passing_strict, 2);
    }
}

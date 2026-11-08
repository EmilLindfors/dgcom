//! Tide gauge validation for comparing model output against observations.
//!
//! This module provides tools for validating 2D shallow water model results
//! against tide gauge observations from the Norwegian coast.
//!
//! # Features
//!
//! - Station metadata management (location, name, datum)
//! - Model value extraction at station locations
//! - Time series comparison (RMSE, bias, correlation, skill score)
//! - Harmonic constituent comparison (amplitude, phase errors)
//! - Multi-station validation summaries
//!
//! # Example
//!
//! ```ignore
//! use dgcom::analysis::{TideGaugeStation, StationValidationResult};
//! use dgcom::analysis::TimeSeries;
//!
//! // Define stations
//! let bergen = TideGaugeStation::new("Bergen", 5.32, 60.39)
//!     .with_datum_offset(0.0);
//!
//! // Create observed time series
//! let obs = TimeSeries::new(&times, &water_levels)
//!     .with_name("Bergen");
//!
//! // Create model time series (extracted at station location)
//! let model = TimeSeries::new(&times, &model_eta)
//!     .with_name("Bergen");
//!
//! // Validate
//! let result = StationValidationResult::compute(&bergen, &model, &obs)?;
//! println!("RMSE: {:.3} m", result.metrics.rmse);
//! println!("Skill: {:.2}", result.metrics.skill_score);
//! ```

use super::{
    AnalysisError, ComparisonMetrics, ConstituentComparison, ConstituentComparisonSummary,
    HarmonicAnalysis, ReferenceFit, TimeSeries, compare_harmonics,
};

/// Metadata for a tide gauge station.
///
/// Stores location, name, and reference datum information.
#[derive(Clone, Debug)]
pub struct TideGaugeStation {
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
    /// Vertical datum offset (m) - add to observations to match model datum
    pub datum_offset: f64,
    /// Station ID in external database (e.g., PSMSL, UHSLC)
    pub external_id: Option<String>,
}

impl TideGaugeStation {
    /// Create a new tide gauge station.
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
            datum_offset: 0.0,
            external_id: None,
        }
    }

    /// Set local model coordinates.
    pub fn with_local_coords(mut self, x: f64, y: f64) -> Self {
        self.x = Some(x);
        self.y = Some(y);
        self
    }

    /// Set vertical datum offset.
    ///
    /// This offset is added to observations to match the model's vertical datum.
    pub fn with_datum_offset(mut self, offset: f64) -> Self {
        self.datum_offset = offset;
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

/// Norwegian tide gauge stations.
///
/// Standard stations for validation along the Norwegian coast.
/// Coordinates are approximate and should be verified against official sources.
pub mod norwegian_stations {
    use super::TideGaugeStation;

    /// Bergen tide gauge (Kartverket station).
    pub fn bergen() -> TideGaugeStation {
        TideGaugeStation::new("Bergen", 5.32, 60.39)
    }

    /// Stavanger tide gauge.
    pub fn stavanger() -> TideGaugeStation {
        TideGaugeStation::new("Stavanger", 5.73, 58.97)
    }

    /// Trondheim tide gauge.
    pub fn trondheim() -> TideGaugeStation {
        TideGaugeStation::new("Trondheim", 10.39, 63.44)
    }

    /// Kristiansund tide gauge.
    pub fn kristiansund() -> TideGaugeStation {
        TideGaugeStation::new("Kristiansund", 7.73, 63.11)
    }

    /// Ålesund tide gauge.
    pub fn alesund() -> TideGaugeStation {
        TideGaugeStation::new("Ålesund", 6.15, 62.47)
    }

    /// Bodø tide gauge.
    pub fn bodo() -> TideGaugeStation {
        TideGaugeStation::new("Bodø", 14.39, 67.29)
    }

    /// Tromsø tide gauge.
    pub fn tromso() -> TideGaugeStation {
        TideGaugeStation::new("Tromsø", 18.96, 69.65)
    }

    /// Hammerfest tide gauge.
    pub fn hammerfest() -> TideGaugeStation {
        TideGaugeStation::new("Hammerfest", 23.68, 70.66)
    }

    /// Vardo tide gauge.
    pub fn vardo() -> TideGaugeStation {
        TideGaugeStation::new("Vardø", 31.11, 70.37)
    }

    /// Heimsjø tide gauge (Frøya region).
    pub fn heimsjo() -> TideGaugeStation {
        TideGaugeStation::new("Heimsjø", 9.10, 63.43)
    }

    /// Get all standard Norwegian coast stations.
    pub fn all_stations() -> Vec<TideGaugeStation> {
        vec![
            bergen(),
            stavanger(),
            trondheim(),
            kristiansund(),
            alesund(),
            bodo(),
            tromso(),
            hammerfest(),
            vardo(),
        ]
    }

    /// Get stations in the Trøndelag region (around Frøya).
    pub fn trondelag_stations() -> Vec<TideGaugeStation> {
        vec![trondheim(), kristiansund(), heimsjo()]
    }
}

/// Model values linearly interpolated to the observation times that lie
/// inside the model's time span, paired with those observations.
fn paired_by_time(model: &TimeSeries, obs: &TimeSeries) -> (Vec<f64>, Vec<f64>) {
    let (mt, mv) = (model.times(), model.values());
    let (Some(&first), Some(&last)) = (mt.first(), mt.last()) else {
        return (Vec::new(), Vec::new());
    };
    obs.times()
        .iter()
        .zip(obs.values())
        .filter(|(t, _)| (first..=last).contains(*t))
        .map(|(&t, v)| {
            let i = mt.partition_point(|&m| m < t);
            let model_at = if mt[i] == t || i == 0 {
                mv[i]
            } else {
                let w = (t - mt[i - 1]) / (mt[i] - mt[i - 1]);
                mv[i - 1] + w * (mv[i] - mv[i - 1])
            };
            (model_at, v)
        })
        .unzip()
}

/// A tide gauge's record: its finite samples (Unix times) and their
/// reference constants over the whole record.
#[derive(Clone, Debug)]
pub struct GaugeRecord {
    /// The gauge
    pub station: TideGaugeStation,
    /// The finite samples
    pub observed: TimeSeries,
    /// [`HarmonicAnalysis::fit_gauge_record`] of them
    pub fit: Result<ReferenceFit, AnalysisError>,
}

impl GaugeRecord {
    /// `station`'s samples `(times[i], values[i])` (Unix times), the
    /// non-finite ones (gaps) left out, and their whole-record fit.
    pub fn new(station: TideGaugeStation, times: &[f64], values: &[f64]) -> Self {
        let (times, values): (Vec<f64>, Vec<f64>) = times
            .iter()
            .zip(values)
            .filter(|(_, v)| v.is_finite())
            .map(|(&t, &v)| (t, v))
            .unzip();
        let fit = HarmonicAnalysis::fit_gauge_record(&times, &values);
        Self {
            observed: TimeSeries::new(&times, &values).with_name(station.name.clone()),
            station,
            fit,
        }
    }

    /// Length of the record (days).
    pub fn days(&self) -> f64 {
        let t = self.observed.times();
        t.first()
            .zip(t.last())
            .map_or(0.0, |(t0, t1)| (t1 - t0) / 86_400.0)
    }
}

/// The skill of a series against a reference over their paired samples
/// ([`StationValidationResult::compute`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SkillScore {
    /// Paired samples
    pub n: usize,
    /// Root-mean-square difference (m)
    pub rmse: f64,
    /// Root-mean-square difference about the mean difference (m): the
    /// error with the bias, often the datum's, taken out
    pub centred_rmse: f64,
    /// Mean difference, series minus reference (m)
    pub bias: f64,
    /// Pearson correlation
    pub correlation: f64,
    /// Standard deviations of the series and of the reference (m)
    pub std: (f64, f64),
}

impl SkillScore {
    fn of(result: &StationValidationResult) -> Self {
        let m = &result.metrics;
        Self {
            n: m.n_points,
            rmse: m.rmse,
            centred_rmse: (m.rmse.powi(2) - m.bias.powi(2)).max(0.0).sqrt(),
            bias: m.bias,
            correlation: m.correlation,
            std: (result.model_std, result.obs_std),
        }
    }
}

/// A model's surface at a tide gauge against the gauge's record: the skill
/// against the observations and against the gauge's own tidal prediction
/// (its whole-record fit, which takes the weather out), and the reference
/// constants constituent by constituent.
#[derive(Clone, Debug)]
pub struct GaugeComparison {
    /// The model's reference constants
    /// ([`HarmonicAnalysis::fit_with_reference`] of
    /// [`HarmonicAnalysis::STATION_RECORD`] with the gauge's ratios). A
    /// record too short for any is an error, and the rest stands.
    pub model_fit: Result<ReferenceFit, AnalysisError>,
    /// Model against gauge for each constituent of the model's fit that the
    /// gauge's has, in the model's order
    pub constituents: Vec<ConstituentComparison>,
    /// Against the observations
    pub vs_observations: Result<SkillScore, AnalysisError>,
    /// Against the gauge's tidal prediction at the model's times
    pub vs_prediction: Result<SkillScore, AnalysisError>,
}

impl GaugeComparison {
    /// Compare `model` (the surface at the gauge, Unix times, the spin-up
    /// already cut) with `gauge`.
    pub fn compute(model: &TimeSeries, gauge: &GaugeRecord) -> Self {
        let gauge_fit = gauge.fit.as_ref().ok();
        let model_fit = HarmonicAnalysis::fit_with_reference(
            &HarmonicAnalysis::STATION_RECORD,
            model.times(),
            model.values(),
            gauge_fit,
        );
        let constituents = match (&model_fit, gauge_fit) {
            (Ok(m), Some(g)) => m
                .constants
                .iter()
                .filter_map(|c| {
                    let o = g.get(c.name)?;
                    Some(ConstituentComparison::new(
                        c.name,
                        c.amplitude,
                        c.lag_deg.to_radians(),
                        o.amplitude,
                        o.lag_deg.to_radians(),
                    ))
                })
                .collect(),
            _ => Vec::new(),
        };
        let skill = |reference: &TimeSeries| {
            StationValidationResult::compute(&gauge.station, model, reference)
                .map(|r| SkillScore::of(&r))
        };
        let vs_prediction = match &gauge.fit {
            Ok(fit) => skill(&TimeSeries::new(model.times(), &fit.predict(model.times()))),
            Err(e) => Err(e.clone()),
        };
        Self {
            model_fit,
            constituents,
            vs_observations: skill(&gauge.observed),
            vs_prediction,
        }
    }

    /// Root sum of squares of the constituents' complex differences |ΔZ|
    /// (m).
    pub fn rss_difference(&self) -> f64 {
        self.constituents
            .iter()
            .map(|c| c.complex_difference().powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

/// Validation result for a single tide gauge station.
#[derive(Clone, Debug)]
pub struct StationValidationResult {
    /// Station metadata
    pub station: TideGaugeStation,
    /// Time series comparison metrics
    pub metrics: ComparisonMetrics,
    /// Reference constants of the model series
    pub model_harmonics: Option<ReferenceFit>,
    /// Reference constants of the observed series (datum offset applied)
    pub obs_harmonics: Option<ReferenceFit>,
    /// Constituent-by-constituent comparison
    pub constituent_comparisons: Vec<ConstituentComparison>,
    /// Summary of constituent comparisons
    pub constituent_summary: Option<ConstituentComparisonSummary>,
    /// Model mean water level (m)
    pub model_mean: f64,
    /// Observed mean water level (m)
    pub obs_mean: f64,
    /// Model standard deviation (m)
    pub model_std: f64,
    /// Observed standard deviation (m)
    pub obs_std: f64,
}

impl StationValidationResult {
    /// Compute validation result from model and observation time series.
    ///
    /// The series are paired by time, not by index: the model is linearly
    /// interpolated to each observation time inside the model's time span.
    /// Both must therefore be on the same time axis; put model output on the
    /// observations' Unix axis with [`ModelClock::unix`](crate::time::ModelClock::unix).
    /// Means, spreads and metrics are over the paired samples.
    ///
    /// # Arguments
    /// * `station` - Station metadata
    /// * `model` - Model water level time series (increasing times)
    /// * `obs` - Observed water level time series (increasing times; will have
    ///   datum_offset applied)
    ///
    /// # Errors
    /// [`AnalysisError::TooFewPairedSamples`] if fewer than two observations
    /// fall inside the model's time span (a run shorter than the gauge's
    /// sampling interval after its spin-up, or a record that misses the run).
    pub fn compute(
        station: &TideGaugeStation,
        model: &TimeSeries,
        obs: &TimeSeries,
    ) -> Result<Self, AnalysisError> {
        let (model_values, obs_values) = paired_by_time(model, obs);
        if model_values.len() < 2 {
            return Err(AnalysisError::TooFewPairedSamples {
                station: station.name.clone(),
            });
        }

        // Apply datum offset to observations
        let obs_adjusted: Vec<f64> = obs_values
            .iter()
            .map(|&v| v + station.datum_offset)
            .collect();

        let metrics = ComparisonMetrics::compute(&model_values, &obs_adjusted);

        let mean_std = |values: &[f64]| {
            let n = values.len() as f64;
            let mean = values.iter().sum::<f64>() / n;
            let var = values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
            (mean, var.sqrt())
        };
        let (model_mean, model_std) = mean_std(&model_values);
        let (obs_mean, obs_std) = mean_std(&obs_adjusted);

        Ok(Self {
            station: station.clone(),
            metrics,
            model_harmonics: None,
            obs_harmonics: None,
            constituent_comparisons: Vec::new(),
            constituent_summary: None,
            model_mean,
            obs_mean,
            model_std,
            obs_std,
        })
    }

    /// [`Self::compute`] and the reference constants of both series (on
    /// the same absolute time axis, so they compare whatever the records'
    /// spans), compared constituent by constituent.
    ///
    /// # Errors
    /// As [`Self::compute`] and [`HarmonicAnalysis::fit`] (e.g. a record too
    /// short to separate the analysis's constituents).
    pub fn compute_with_harmonics(
        station: &TideGaugeStation,
        model: &TimeSeries,
        obs: &TimeSeries,
        analysis: &HarmonicAnalysis,
    ) -> Result<Self, AnalysisError> {
        let mut result = Self::compute(station, model, obs)?;
        let model_harmonics = analysis.fit_series(model)?;
        let obs_harmonics = analysis.fit_series(&obs.map_values(|v| v + station.datum_offset))?;

        // Compare constituents
        let comparisons = compare_harmonics(&model_harmonics, &obs_harmonics);
        let summary = ConstituentComparisonSummary::from_comparisons(&comparisons);

        result.model_harmonics = Some(model_harmonics);
        result.obs_harmonics = Some(obs_harmonics);
        result.constituent_comparisons = comparisons;
        result.constituent_summary = Some(summary);

        Ok(result)
    }

    /// Check if model passes basic validation criteria.
    ///
    /// Criteria:
    /// - Correlation > 0.9
    /// - Skill score > 0.8
    /// - Bias < 0.1 m
    pub fn passes_basic_validation(&self) -> bool {
        self.metrics.correlation > 0.9
            && self.metrics.skill_score > 0.8
            && self.metrics.bias.abs() < 0.1
    }

    /// Check if model passes strict validation criteria.
    ///
    /// Criteria:
    /// - Correlation > 0.95
    /// - Skill score > 0.9
    /// - Bias < 0.05 m
    /// - RMSE < 0.1 m
    pub fn passes_strict_validation(&self) -> bool {
        self.metrics.correlation > 0.95
            && self.metrics.skill_score > 0.9
            && self.metrics.bias.abs() < 0.05
            && self.metrics.rmse < 0.1
    }

    /// Format validation result as a summary string.
    pub fn summary(&self) -> String {
        let mut s = format!(
            "Station: {}\n\
             Location: ({:.2}°E, {:.2}°N)\n\
             --------------------------\n\
             Time Series Metrics:\n\
             RMSE:        {:.4} m\n\
             Bias:        {:.4} m\n\
             MAE:         {:.4} m\n\
             Correlation: {:.4}\n\
             Skill Score: {:.4}\n\
             Max Error:   {:.4} m\n\
             N Points:    {}\n\
             --------------------------\n\
             Statistics:\n\
             Model Mean:  {:.4} m\n\
             Obs Mean:    {:.4} m\n\
             Model Std:   {:.4} m\n\
             Obs Std:     {:.4} m\n",
            self.station.name,
            self.station.longitude,
            self.station.latitude,
            self.metrics.rmse,
            self.metrics.bias,
            self.metrics.mae,
            self.metrics.correlation,
            self.metrics.skill_score,
            self.metrics.max_error,
            self.metrics.n_points,
            self.model_mean,
            self.obs_mean,
            self.model_std,
            self.obs_std,
        );

        if let Some(ref summary) = self.constituent_summary {
            s.push_str(&format!(
                "--------------------------\n\
                 Harmonic Comparison:\n\
                 N Constituents:      {}\n\
                 Mean Amp Ratio:      {:.3}\n\
                 RMS Amp Error:       {:.4} m\n\
                 RMS Phase Error:     {:.1}°\n",
                summary.n_constituents,
                summary.mean_amplitude_ratio,
                summary.rms_amplitude_error,
                summary.rms_phase_error_degrees(),
            ));
        }

        s
    }
}

/// Summary of validation across multiple stations.
#[derive(Clone, Debug)]
pub struct ValidationSummary {
    /// Number of stations validated
    pub n_stations: usize,
    /// Mean RMSE across all stations
    pub mean_rmse: f64,
    /// Mean bias across all stations
    pub mean_bias: f64,
    /// Mean correlation across all stations
    pub mean_correlation: f64,
    /// Mean skill score across all stations
    pub mean_skill_score: f64,
    /// Worst (highest) RMSE
    pub worst_rmse: f64,
    /// Station with worst RMSE
    pub worst_station: String,
    /// Best (lowest) RMSE
    pub best_rmse: f64,
    /// Station with best RMSE
    pub best_station: String,
    /// Number of stations passing basic validation
    pub n_passing_basic: usize,
    /// Number of stations passing strict validation
    pub n_passing_strict: usize,
}

impl ValidationSummary {
    /// Compute summary from a list of station validation results.
    pub fn from_results(results: &[StationValidationResult]) -> Self {
        if results.is_empty() {
            return Self {
                n_stations: 0,
                mean_rmse: 0.0,
                mean_bias: 0.0,
                mean_correlation: 0.0,
                mean_skill_score: 0.0,
                worst_rmse: 0.0,
                worst_station: String::new(),
                best_rmse: 0.0,
                best_station: String::new(),
                n_passing_basic: 0,
                n_passing_strict: 0,
            };
        }

        let n = results.len() as f64;

        let mean_rmse = results.iter().map(|r| r.metrics.rmse).sum::<f64>() / n;
        let mean_bias = results.iter().map(|r| r.metrics.bias).sum::<f64>() / n;
        let mean_correlation = results.iter().map(|r| r.metrics.correlation).sum::<f64>() / n;
        let mean_skill_score = results.iter().map(|r| r.metrics.skill_score).sum::<f64>() / n;

        // Use partial_cmp with unwrap_or to handle NaN values gracefully
        let worst = results
            .iter()
            .max_by(|a, b| {
                a.metrics
                    .rmse
                    .partial_cmp(&b.metrics.rmse)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("results is not empty");
        let best = results
            .iter()
            .min_by(|a, b| {
                a.metrics
                    .rmse
                    .partial_cmp(&b.metrics.rmse)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("results is not empty");

        let n_passing_basic = results
            .iter()
            .filter(|r| r.passes_basic_validation())
            .count();
        let n_passing_strict = results
            .iter()
            .filter(|r| r.passes_strict_validation())
            .count();

        Self {
            n_stations: results.len(),
            mean_rmse,
            mean_bias,
            mean_correlation,
            mean_skill_score,
            worst_rmse: worst.metrics.rmse,
            worst_station: worst.station.name.clone(),
            best_rmse: best.metrics.rmse,
            best_station: best.station.name.clone(),
            n_passing_basic,
            n_passing_strict,
        }
    }
}

/// The summary as a table.
impl std::fmt::Display for ValidationSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Validation Summary ({} stations)\n\
             ================================\n\
             Mean RMSE:        {:.4} m\n\
             Mean Bias:        {:.4} m\n\
             Mean Correlation: {:.4}\n\
             Mean Skill:       {:.4}\n\
             --------------------------------\n\
             Best RMSE:   {:.4} m ({})\n\
             Worst RMSE:  {:.4} m ({})\n\
             --------------------------------\n\
             Passing Basic:  {}/{}\n\
             Passing Strict: {}/{}\n",
            self.n_stations,
            self.mean_rmse,
            self.mean_bias,
            self.mean_correlation,
            self.mean_skill_score,
            self.best_rmse,
            self.best_station,
            self.worst_rmse,
            self.worst_station,
            self.n_passing_basic,
            self.n_stations,
            self.n_passing_strict,
            self.n_stations,
        )
    }
}

/// Extract model values at tide gauge stations.
///
/// This trait allows extracting water level time series from different
/// model output formats at specific station locations.
pub trait ModelExtractor {
    /// Extract water level at a single point for all time steps.
    ///
    /// # Arguments
    /// * `x` - Local x-coordinate
    /// * `y` - Local y-coordinate
    /// * `times` - Time values (seconds)
    ///
    /// # Returns
    /// Water level (surface elevation) at each time step, or None if point is outside domain.
    fn extract_at_point(&self, x: f64, y: f64, times: &[f64]) -> Option<Vec<f64>>;

    /// Extract time series at a tide gauge station.
    fn extract_at_station(&self, station: &TideGaugeStation, times: &[f64]) -> Option<TimeSeries> {
        let (x, y) = station.local_coords();
        self.extract_at_point(x, y, times).map(|values| {
            TimeSeries::new(times, &values)
                .with_name(&station.name)
                .with_location(x, y)
        })
    }
}

/// Simple model extractor that stores pre-computed time series at station locations.
///
/// Use this when model output has already been extracted during simulation.
#[derive(Clone, Debug, Default)]
pub struct PrecomputedExtractor {
    /// Map from station name to time series
    data: std::collections::HashMap<String, TimeSeries>,
}

impl PrecomputedExtractor {
    /// Create a new empty extractor.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a time series for a station.
    pub fn add_series(&mut self, station_name: impl Into<String>, series: TimeSeries) {
        self.data.insert(station_name.into(), series);
    }

    /// Get time series for a station.
    pub fn get_series(&self, station_name: &str) -> Option<&TimeSeries> {
        self.data.get(station_name)
    }
}

impl ModelExtractor for PrecomputedExtractor {
    fn extract_at_point(&self, _x: f64, _y: f64, _times: &[f64]) -> Option<Vec<f64>> {
        // This extractor doesn't support arbitrary point extraction
        None
    }

    fn extract_at_station(&self, station: &TideGaugeStation, _times: &[f64]) -> Option<TimeSeries> {
        self.data.get(&station.name).cloned()
    }
}

/// Validate the model at several stations.
///
/// # Arguments
/// * `stations` - List of tide gauge stations
/// * `model_extractor` - Model output extractor
/// * `observations` - Map from station name to observed time series
/// * `times` - Common time vector for model extraction
/// * `analysis` - Optional harmonic analysis
///
/// # Returns
/// Each station's result or why it could not be validated (no model series,
/// no observations, a record too short), in the order of `stations`, and
/// the summary over the stations validated.
pub fn validate_stations(
    stations: &[TideGaugeStation],
    model_extractor: &dyn ModelExtractor,
    observations: &std::collections::HashMap<String, TimeSeries>,
    times: &[f64],
    analysis: Option<&HarmonicAnalysis>,
) -> (
    Vec<Result<StationValidationResult, AnalysisError>>,
    ValidationSummary,
) {
    let results: Vec<_> = stations
        .iter()
        .map(|station| {
            let model = model_extractor
                .extract_at_station(station, times)
                .ok_or_else(|| AnalysisError::NoModelSeries(station.name.clone()))?;
            let obs = observations
                .get(&station.name)
                .ok_or_else(|| AnalysisError::NoObservations(station.name.clone()))?;
            match analysis {
                Some(ha) => {
                    StationValidationResult::compute_with_harmonics(station, &model, obs, ha)
                }
                None => StationValidationResult::compute(station, &model, obs),
            }
        })
        .collect();
    let validated: Vec<StationValidationResult> = results
        .iter()
        .filter_map(|r| r.as_ref().ok().cloned())
        .collect();
    let summary = ValidationSummary::from_results(&validated);
    (results, summary)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    #[test]
    fn test_station_creation() {
        let station = TideGaugeStation::new("Bergen", 5.32, 60.39)
            .with_datum_offset(0.05)
            .with_local_coords(1000.0, 2000.0);

        assert_eq!(station.name, "Bergen");
        assert!((station.longitude - 5.32).abs() < TOL);
        assert!((station.latitude - 60.39).abs() < TOL);
        assert!((station.datum_offset - 0.05).abs() < TOL);
        assert!(station.has_local_coords());
        assert_eq!(station.local_coords(), (1000.0, 2000.0));
    }

    #[test]
    fn test_norwegian_stations() {
        let stations = norwegian_stations::all_stations();
        assert!(!stations.is_empty());

        let bergen = norwegian_stations::bergen();
        assert_eq!(bergen.name, "Bergen");
    }

    #[test]
    fn test_perfect_validation() {
        let station = TideGaugeStation::new("Test", 0.0, 0.0);
        let times: Vec<f64> = (0..100).map(|i| i as f64 * 3600.0).collect();
        let values: Vec<f64> = times.iter().map(|&t| (t / 44712.0).sin()).collect();

        let model = TimeSeries::new(&times, &values);
        let obs = TimeSeries::new(&times, &values);

        let result = StationValidationResult::compute(&station, &model, &obs).unwrap();

        assert!(result.metrics.rmse.abs() < TOL);
        assert!(result.metrics.correlation > 0.99);
        assert!(result.metrics.skill_score > 0.99);
        assert!(result.passes_basic_validation());
        assert!(result.passes_strict_validation());
    }

    /// Series with different sampling and spans are compared at the same
    /// instants (this used to require equal lengths and pair by index).
    #[test]
    fn test_series_are_paired_by_time() {
        let station = TideGaugeStation::new("Test", 0.0, 0.0);
        let signal = |t: f64| 0.3 + 2e-6 * t;
        let t0 = 1.75e9;
        let model_t: Vec<f64> = (0..100).map(|i| t0 + i as f64 * 3600.0).collect();
        let obs_t: Vec<f64> = (0..300)
            .map(|i| t0 + 36_000.0 + i as f64 * 1800.0)
            .collect();
        let model = TimeSeries::new(
            &model_t,
            &model_t.iter().map(|&t| signal(t)).collect::<Vec<_>>(),
        );
        let obs = TimeSeries::new(
            &obs_t,
            &obs_t.iter().map(|&t| signal(t)).collect::<Vec<_>>(),
        );

        let result = StationValidationResult::compute(&station, &model, &obs).unwrap();
        assert!(result.metrics.rmse < 1e-9, "rmse {}", result.metrics.rmse);
        // Observations after the model's last output are not compared
        let paired = paired_by_time(&model, &obs).0.len();
        assert_eq!(paired, 1 + (99 - 10) * 2);
    }

    /// A run that ends before the gauge's second sample after its spin-up
    /// has nothing to compare (`froya_real_data` panicked on a 4 h run with
    /// a 3 h spin-up and hourly observations).
    #[test]
    fn too_few_paired_observations_are_an_error() {
        let station = TideGaugeStation::new("Test", 0.0, 0.0);
        let model_t: Vec<f64> = (0..7).map(|i| i as f64 * 600.0).collect();
        let model = TimeSeries::new(&model_t, &vec![0.1; model_t.len()]);
        let obs_t: Vec<f64> = (0..10).map(|i| i as f64 * 3600.0).collect();
        let obs = TimeSeries::new(&obs_t, &vec![0.2; obs_t.len()]);

        // One hourly observation inside a 0–3000 s run…
        let one = TimeSeries::new(&model_t[..6], &[0.1; 6]);
        assert_eq!(
            StationValidationResult::compute(&station, &one, &obs).unwrap_err(),
            AnalysisError::TooFewPairedSamples {
                station: "Test".into()
            }
        );
        // …but two inside 0–3600 s are enough
        let two = StationValidationResult::compute(&station, &model, &obs)
            .expect("two paired observations");
        assert_eq!(two.metrics.n_points, 2);
        assert!((two.metrics.bias + 0.1).abs() < TOL);
    }

    #[test]
    fn test_validation_with_bias() {
        let station = TideGaugeStation::new("Test", 0.0, 0.0);
        let times: Vec<f64> = (0..100).map(|i| i as f64 * 3600.0).collect();
        let values: Vec<f64> = times.iter().map(|&t| (t / 44712.0).sin()).collect();
        let biased: Vec<f64> = values.iter().map(|&v| v + 0.5).collect();

        let model = TimeSeries::new(&times, &biased);
        let obs = TimeSeries::new(&times, &values);

        let result = StationValidationResult::compute(&station, &model, &obs).unwrap();

        assert!((result.metrics.bias - 0.5).abs() < TOL);
        assert!(result.metrics.correlation > 0.99); // High correlation despite bias
    }

    #[test]
    fn test_datum_offset_applied() {
        let station = TideGaugeStation::new("Test", 0.0, 0.0).with_datum_offset(0.1);
        let times: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let model_values = vec![1.1; 10]; // Model at 1.1
        let obs_values = vec![1.0; 10]; // Obs at 1.0, will become 1.1 with offset

        let model = TimeSeries::new(&times, &model_values);
        let obs = TimeSeries::new(&times, &obs_values);

        let result = StationValidationResult::compute(&station, &model, &obs).unwrap();

        // After applying datum offset, model and adjusted obs should match
        assert!(result.metrics.rmse.abs() < TOL);
    }

    #[test]
    fn test_validation_summary() {
        let station1 = TideGaugeStation::new("Good", 0.0, 0.0);
        let station2 = TideGaugeStation::new("Bad", 1.0, 1.0);

        let times: Vec<f64> = (0..100).map(|i| i as f64 * 3600.0).collect();
        let good_values: Vec<f64> = times.iter().map(|&t| (t / 44712.0).sin()).collect();
        let bad_model: Vec<f64> = good_values.iter().map(|&v| v + 0.2).collect();

        let model1 = TimeSeries::new(&times, &good_values);
        let obs1 = TimeSeries::new(&times, &good_values);
        let model2 = TimeSeries::new(&times, &bad_model);
        let obs2 = TimeSeries::new(&times, &good_values);

        let result1 = StationValidationResult::compute(&station1, &model1, &obs1).unwrap();
        let result2 = StationValidationResult::compute(&station2, &model2, &obs2).unwrap();

        let summary = ValidationSummary::from_results(&[result1, result2]);

        assert_eq!(summary.n_stations, 2);
        assert_eq!(summary.best_station, "Good");
        assert_eq!(summary.worst_station, "Bad");
    }

    #[test]
    fn test_precomputed_extractor() {
        let mut extractor = PrecomputedExtractor::new();

        let times: Vec<f64> = vec![0.0, 1.0, 2.0];
        let values: Vec<f64> = vec![1.0, 1.5, 1.2];
        let series = TimeSeries::new(&times, &values);

        extractor.add_series("Bergen", series.clone());

        let station = TideGaugeStation::new("Bergen", 5.32, 60.39);
        let extracted = extractor.extract_at_station(&station, &times);

        assert!(extracted.is_some());
        assert_eq!(extracted.unwrap().len(), 3);
    }

    /// Two days separate M2 from K1: the model's 10 % larger M2 shows in
    /// the comparison. The four standard constituents need 15 days, so the
    /// same records with them are an error (the old record-time fit gave
    /// numbers for them anyway).
    #[test]
    fn test_validation_with_harmonics() {
        let station = TideGaugeStation::new("Test", 0.0, 0.0);
        let t0 = 1.75e9;
        let times: Vec<f64> = (0..288).map(|i| t0 + i as f64 * 600.0).collect();
        let tide = |scale: f64| -> Vec<f64> {
            let constants = ReferenceFit {
                mean: 0.0,
                constants: vec![
                    crate::analysis::ReferenceConstant {
                        name: "M2",
                        amplitude: 0.5 * scale,
                        lag_deg: 300.0,
                        inferred: false,
                    },
                    crate::analysis::ReferenceConstant {
                        name: "K1",
                        amplitude: 0.07,
                        lag_deg: 165.0,
                        inferred: false,
                    },
                ],
                r_squared: 1.0,
                residual_rms: 0.0,
            };
            constants.predict(&times)
        };
        let model = TimeSeries::new(&times, &tide(1.1));
        let obs = TimeSeries::new(&times, &tide(1.0));

        let analysis = HarmonicAnalysis::new(&["M2", "K1"]);
        let result =
            StationValidationResult::compute_with_harmonics(&station, &model, &obs, &analysis)
                .unwrap();
        let m2 = result.constituent_comparisons[0];
        assert_eq!(m2.name, "M2");
        assert!((m2.amplitude_ratio - 1.1).abs() < 1e-9, "{m2:?}");
        assert!(m2.phase_error.abs() < 1e-9, "{m2:?}");
        assert!(result.constituent_summary.is_some());

        let standard = StationValidationResult::compute_with_harmonics(
            &station,
            &model,
            &obs,
            &HarmonicAnalysis::standard(),
        );
        assert!(matches!(standard, Err(AnalysisError::NotSeparable { .. })));
    }

    #[test]
    fn test_summary_output() {
        let station = TideGaugeStation::new("Test", 5.0, 60.0);
        let times: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let values = vec![1.0; 10];

        let model = TimeSeries::new(&times, &values);
        let obs = TimeSeries::new(&times, &values);

        let result = StationValidationResult::compute(&station, &model, &obs).unwrap();
        let summary = result.summary();

        assert!(summary.contains("Test"));
        assert!(summary.contains("RMSE"));
        assert!(summary.contains("Skill Score"));
    }

    /// The tide of `constants` (name, amplitude, Greenwich lag in degrees)
    /// about `mean` at `times` (Unix s), as a fit predicts it.
    fn tide(mean: f64, constants: &[(&'static str, f64, f64)], times: &[f64]) -> Vec<f64> {
        let fit = ReferenceFit {
            mean,
            constants: constants
                .iter()
                .map(
                    |&(name, amplitude, lag_deg)| crate::analysis::ReferenceConstant {
                        name,
                        amplitude,
                        lag_deg,
                        inferred: false,
                    },
                )
                .collect(),
            r_squared: 1.0,
            residual_rms: 0.0,
        };
        fit.predict(times)
    }

    /// A gauge of M2 and K1 (with P1 at equilibrium) for 60 days, with a
    /// gap; a 3-day model with 1.2
    /// times its M2, 2.5 times its K1 and 5 cm above it. The comparison
    /// finds those ratios with no phase error, and scores the model
    /// against the gauge's prediction over every model sample.
    #[test]
    fn a_comparison_finds_the_models_constants_against_the_gauges() {
        let t0 = 1_749_945_600.0;
        let gauge_times: Vec<f64> = (0..60 * 24).map(|h| t0 + 3600.0 * f64::from(h)).collect();
        // P1 at its equilibrium ratio to K1, as the fit infers it
        let p1 = crate::analysis::Inference::P1_FROM_K1.amplitude_ratio;
        let mut gauge_values = tide(
            0.0,
            &[
                ("M2", 1.0, 60.0),
                ("K1", 0.2, 200.0),
                ("P1", 0.2 * p1, 200.0),
            ],
            &gauge_times,
        );
        gauge_values[100] = f64::NAN;
        let gauge = GaugeRecord::new(
            TideGaugeStation::new("Test", 8.0, 63.0),
            &gauge_times,
            &gauge_values,
        );
        assert_eq!(gauge.observed.len(), 60 * 24 - 1, "the gap is left out");
        let fit = gauge.fit.as_ref().unwrap();
        assert!(
            (fit.get("M2").unwrap().amplitude - 1.0).abs() < 1e-6,
            "{:?}",
            fit.constants
        );

        let start = t0 + 20.0 * 86_400.0;
        let model_times: Vec<f64> = (0..3 * 24 * 6)
            .map(|i| start + 600.0 * f64::from(i))
            .collect();
        let model_values = tide(
            0.05,
            &[
                ("M2", 1.2, 60.0),
                ("K1", 0.5, 200.0),
                ("P1", 0.5 * p1, 200.0),
            ],
            &model_times,
        );
        let comparison =
            GaugeComparison::compute(&TimeSeries::new(&model_times, &model_values), &gauge);
        for (name, ratio) in [("M2", 1.2), ("K1", 2.5)] {
            let c = comparison
                .constituents
                .iter()
                .find(|c| c.name == name)
                .unwrap();
            assert!(
                (c.amplitude_ratio - ratio).abs() < 1e-6,
                "{name}: {}",
                c.amplitude_ratio
            );
            assert!(
                c.phase_error_degrees().abs() < 1e-4,
                "{name}: {}",
                c.phase_error_degrees()
            );
        }
        let prediction = comparison.vs_prediction.as_ref().unwrap();
        assert_eq!(prediction.n, model_times.len());
        // The offset, and the 3-day mean of the larger tide
        assert!((prediction.bias - 0.05).abs() < 0.01, "{}", prediction.bias);
        let observed = comparison.vs_observations.as_ref().unwrap();
        assert_eq!(
            observed.n,
            3 * 24,
            "the hourly samples within the model's span, 0 to 71 h"
        );
        assert!(comparison.rss_difference() > 0.2);
    }
}

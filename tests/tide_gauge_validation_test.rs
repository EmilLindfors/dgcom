//! Integration tests for tide gauge validation.
//!
//! Tests the full workflow from synthetic model output to validation metrics.

use dgcom::analysis::{
    HarmonicAnalysis, PrecomputedExtractor, StationValidationResult, TideGaugeStation, TimeSeries,
    ValidationSummary, validate_stations,
};
use std::collections::HashMap;
use std::f64::consts::PI;

/// M2 tidal period in seconds (~12.42 hours).
const M2_PERIOD: f64 = 12.42 * 3600.0;

/// S2 tidal period in seconds (12.00 hours).
const S2_PERIOD: f64 = 12.0 * 3600.0;

/// Generate synthetic tidal signal.
fn synthetic_tidal_signal(
    times: &[f64],
    m2_amplitude: f64,
    m2_phase: f64,
    s2_amplitude: f64,
    s2_phase: f64,
    noise_std: f64,
) -> Vec<f64> {
    let m2_omega = 2.0 * PI / M2_PERIOD;
    let s2_omega = 2.0 * PI / S2_PERIOD;

    times
        .iter()
        .map(|&t| {
            let m2 = m2_amplitude * (m2_omega * t + m2_phase).cos();
            let s2 = s2_amplitude * (s2_omega * t + s2_phase).cos();
            let noise = if noise_std > 0.0 {
                // Simple deterministic "noise" for reproducibility
                0.01 * ((t * 0.001).sin() + (t * 0.0023).cos())
            } else {
                0.0
            };
            m2 + s2 + noise
        })
        .collect()
}

#[test]
fn test_perfect_model_validation() {
    // Simulate 48 hours at 10-minute intervals
    let dt = 600.0; // 10 minutes
    let n_points = 288; // 48 hours
    let times: Vec<f64> = (0..n_points).map(|i| i as f64 * dt).collect();

    // Generate identical model and observation data
    let values = synthetic_tidal_signal(&times, 0.5, 0.0, 0.15, 0.3, 0.0);

    let station = TideGaugeStation::new("Test", 5.0, 60.0);
    let model = TimeSeries::new(&times, &values);
    let obs = TimeSeries::new(&times, &values);

    let result = StationValidationResult::compute(&station, &model, &obs).unwrap();

    assert!(result.metrics.rmse < 1e-10, "RMSE should be ~0");
    assert!(
        result.metrics.correlation > 0.999,
        "Correlation should be ~1"
    );
    assert!(result.metrics.skill_score > 0.999, "Skill should be ~1");
    assert!(result.passes_strict_validation());
}

#[test]
fn test_model_with_bias() {
    let dt = 600.0;
    let n_points = 288;
    let times: Vec<f64> = (0..n_points).map(|i| i as f64 * dt).collect();

    let obs_values = synthetic_tidal_signal(&times, 0.5, 0.0, 0.15, 0.3, 0.0);
    let model_values: Vec<f64> = obs_values.iter().map(|&v| v + 0.1).collect(); // 10 cm bias

    let station = TideGaugeStation::new("Test", 5.0, 60.0);
    let model = TimeSeries::new(&times, &model_values);
    let obs = TimeSeries::new(&times, &obs_values);

    let result = StationValidationResult::compute(&station, &model, &obs).unwrap();

    assert!(
        (result.metrics.bias - 0.1).abs() < 1e-10,
        "Bias should be 0.1 m"
    );
    assert!(
        result.metrics.correlation > 0.999,
        "Correlation should still be high"
    );
    assert!(
        !result.passes_strict_validation(),
        "Should fail strict due to bias"
    );
}

#[test]
fn test_model_with_amplitude_error() {
    let dt = 600.0;
    let n_points = 288;
    let times: Vec<f64> = (0..n_points).map(|i| i as f64 * dt).collect();

    // Model underestimates amplitude by 20%
    let obs_values = synthetic_tidal_signal(&times, 0.5, 0.0, 0.15, 0.3, 0.0);
    let model_values = synthetic_tidal_signal(&times, 0.4, 0.0, 0.12, 0.3, 0.0);

    let station = TideGaugeStation::new("Test", 5.0, 60.0);
    let model = TimeSeries::new(&times, &model_values);
    let obs = TimeSeries::new(&times, &obs_values);

    let result = StationValidationResult::compute(&station, &model, &obs).unwrap();

    assert!(
        result.metrics.correlation > 0.999,
        "Correlation should be high"
    );
    assert!(
        result.metrics.rmse > 0.05,
        "RMSE should reflect amplitude error"
    );
}

#[test]
fn test_model_with_phase_error() {
    let dt = 600.0;
    let n_points = 288;
    let times: Vec<f64> = (0..n_points).map(|i| i as f64 * dt).collect();

    // Model has 30-minute (π/12 rad) phase error
    let phase_error = PI / 12.0;
    let obs_values = synthetic_tidal_signal(&times, 0.5, 0.0, 0.15, 0.3, 0.0);
    let model_values =
        synthetic_tidal_signal(&times, 0.5, phase_error, 0.15, 0.3 + phase_error, 0.0);

    let station = TideGaugeStation::new("Test", 5.0, 60.0);
    let model = TimeSeries::new(&times, &model_values);
    let obs = TimeSeries::new(&times, &obs_values);

    let result = StationValidationResult::compute(&station, &model, &obs).unwrap();

    // Phase error reduces correlation
    assert!(
        result.metrics.correlation < 0.999,
        "Phase error should reduce correlation"
    );
    assert!(
        result.metrics.correlation > 0.9,
        "But correlation still reasonable"
    );
}

#[test]
fn test_harmonic_comparison() {
    // 16 days at 10 minutes, on a Unix time axis: M2 and S2 need 14.8 days
    // to separate (the standard analysis refuses shorter records)
    let dt = 600.0;
    let n_points = 16 * 144;
    let times: Vec<f64> = (0..n_points).map(|i| 1.75e9 + i as f64 * dt).collect();

    let m2_amp = 0.5;
    let m2_phase = 0.5;
    let s2_amp = 0.15;
    let s2_phase = 1.0;

    let obs_values = synthetic_tidal_signal(&times, m2_amp, m2_phase, s2_amp, s2_phase, 0.0);
    let model_values = synthetic_tidal_signal(
        &times,
        m2_amp * 0.95,
        m2_phase + 0.1,
        s2_amp * 1.05,
        s2_phase - 0.05,
        0.0,
    );

    let station = TideGaugeStation::new("Test", 5.0, 60.0);
    let model = TimeSeries::new(&times, &model_values);
    let obs = TimeSeries::new(&times, &obs_values);

    let analysis = HarmonicAnalysis::standard();
    let result =
        StationValidationResult::compute_with_harmonics(&station, &model, &obs, &analysis).unwrap();

    assert!(result.model_harmonics.is_some());
    assert!(result.obs_harmonics.is_some());
    assert!(!result.constituent_comparisons.is_empty());

    // Check that M2 constituent comparison is reasonable
    if let Some(m2_comp) = result
        .constituent_comparisons
        .iter()
        .find(|c| c.name == "M2")
    {
        assert!(
            m2_comp.amplitude_ratio > 0.9 && m2_comp.amplitude_ratio < 1.1,
            "M2 amplitude ratio should be close to 1"
        );
        assert!(
            m2_comp.phase_error.abs() < 0.2,
            "M2 phase error should be small"
        );
    }
}

#[test]
fn test_multi_station_validation() {
    let dt = 600.0;
    let n_points = 288;
    let times: Vec<f64> = (0..n_points).map(|i| i as f64 * dt).collect();

    // Create test stations
    let stations = vec![
        TideGaugeStation::new("Station1", 5.0, 60.0).with_local_coords(0.0, 0.0),
        TideGaugeStation::new("Station2", 6.0, 61.0).with_local_coords(100.0, 100.0),
        TideGaugeStation::new("Station3", 7.0, 62.0).with_local_coords(200.0, 200.0),
    ];

    // Generate different accuracy levels for each station
    let obs1 = synthetic_tidal_signal(&times, 0.5, 0.0, 0.15, 0.3, 0.0);
    let obs2 = synthetic_tidal_signal(&times, 0.6, 0.2, 0.18, 0.5, 0.0);
    let obs3 = synthetic_tidal_signal(&times, 0.4, 0.1, 0.12, 0.4, 0.0);

    // Model is perfect for station 1, has errors for others
    let model1 = obs1.clone();
    let model2: Vec<f64> = obs2.iter().map(|&v| v + 0.05).collect(); // 5 cm bias
    let model3 = synthetic_tidal_signal(&times, 0.35, 0.15, 0.10, 0.45, 0.0); // Amplitude error

    // Set up precomputed extractor
    let mut extractor = PrecomputedExtractor::new();
    extractor.add_series("Station1", TimeSeries::new(&times, &model1));
    extractor.add_series("Station2", TimeSeries::new(&times, &model2));
    extractor.add_series("Station3", TimeSeries::new(&times, &model3));

    // Set up observations
    let mut observations = HashMap::new();
    observations.insert("Station1".to_string(), TimeSeries::new(&times, &obs1));
    observations.insert("Station2".to_string(), TimeSeries::new(&times, &obs2));
    observations.insert("Station3".to_string(), TimeSeries::new(&times, &obs3));

    // Run validation
    let (results, summary) = validate_stations(&stations, &extractor, &observations, &times, None);

    assert_eq!(results.len(), 3);
    assert_eq!(summary.n_stations, 3);

    // Station 1 should be best (perfect match)
    assert_eq!(summary.best_station, "Station1");
    assert!(summary.best_rmse < 1e-10);

    // Check that results can be retrieved for each station
    for result in &results {
        let result = result
            .as_ref()
            .expect("every station has a model and observations");
        assert!(result.metrics.rmse >= 0.0);
        assert!(result.metrics.correlation.abs() <= 1.0);
    }
}

#[test]
fn test_validation_summary_statistics() {
    let station1 = TideGaugeStation::new("Good", 0.0, 0.0);
    let station2 = TideGaugeStation::new("Medium", 1.0, 1.0);
    let station3 = TideGaugeStation::new("Poor", 2.0, 2.0);

    let times: Vec<f64> = (0..100).map(|i| i as f64 * 3600.0).collect();
    let good_values: Vec<f64> = times
        .iter()
        .map(|&t| (t / M2_PERIOD * 2.0 * PI).sin())
        .collect();

    // Good: perfect match
    let model1 = TimeSeries::new(&times, &good_values);
    let obs1 = TimeSeries::new(&times, &good_values);

    // Medium: small bias
    let medium_values: Vec<f64> = good_values.iter().map(|&v| v + 0.03).collect();
    let model2 = TimeSeries::new(&times, &medium_values);
    let obs2 = TimeSeries::new(&times, &good_values);

    // Poor: large errors
    let poor_values: Vec<f64> = good_values.iter().map(|&v| v * 0.5 + 0.2).collect();
    let model3 = TimeSeries::new(&times, &poor_values);
    let obs3 = TimeSeries::new(&times, &good_values);

    let result1 = StationValidationResult::compute(&station1, &model1, &obs1).unwrap();
    let result2 = StationValidationResult::compute(&station2, &model2, &obs2).unwrap();
    let result3 = StationValidationResult::compute(&station3, &model3, &obs3).unwrap();

    let summary = ValidationSummary::from_results(&[result1, result2, result3]);

    assert_eq!(summary.n_stations, 3);
    assert_eq!(summary.best_station, "Good");
    assert_eq!(summary.worst_station, "Poor");
    assert!(summary.mean_rmse > 0.0);
    assert!(summary.mean_correlation > 0.0);

    // At least one should pass basic validation
    assert!(summary.n_passing_basic >= 1);
}

#[test]
fn test_datum_offset_correction() {
    let station = TideGaugeStation::new("Test", 5.0, 60.0).with_datum_offset(0.15); // 15 cm datum difference

    let times: Vec<f64> = (0..100).map(|i| i as f64 * 3600.0).collect();
    let model_values = vec![1.0; 100]; // Model at 1.0 m
    let obs_values = vec![0.85; 100]; // Obs at 0.85 m (before offset)

    let model = TimeSeries::new(&times, &model_values);
    let obs = TimeSeries::new(&times, &obs_values);

    let result = StationValidationResult::compute(&station, &model, &obs).unwrap();

    // After applying 0.15 m offset, obs becomes 1.0 m, matching model
    assert!(
        result.metrics.rmse < 1e-10,
        "RMSE should be ~0 after datum correction"
    );
    assert!((result.metrics.bias).abs() < 1e-10, "Bias should be ~0");
}

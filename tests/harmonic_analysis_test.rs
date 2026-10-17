//! Integration tests for the harmonic analysis: fitting reference constants,
//! prediction, inference, the Rayleigh guard, and comparison metrics.

use dg_rs::{
    AnalysisError, ComparisonMetrics, HarmonicAnalysis, Inference, ReferenceConstant, ReferenceFit,
    TimeSeries, compare_harmonics,
};

/// 2025-06-01 00:00 UTC
const T0: f64 = 1_748_736_000.0;

/// `hours` hourly Unix times from `T0`.
fn hourly(hours: usize) -> Vec<f64> {
    (0..hours).map(|i| T0 + i as f64 * 3600.0).collect()
}

/// The tide of reference constants `(name, H, G°)` about `mean` at `times`,
/// in the fit's own convention (`ReferenceFit::predict`).
fn tide(times: &[f64], mean: f64, constants: &[(&'static str, f64, f64)]) -> Vec<f64> {
    ReferenceFit {
        mean,
        constants: constants
            .iter()
            .map(|&(name, amplitude, lag_deg)| ReferenceConstant {
                name,
                amplitude,
                lag_deg,
                inferred: false,
            })
            .collect(),
        r_squared: 1.0,
        residual_rms: 0.0,
    }
    .predict(times)
}

fn lag_error(a: f64, b: f64) -> f64 {
    ((a - b + 180.0).rem_euclid(360.0) - 180.0).abs()
}

#[test]
fn test_m2_recovery() {
    // ~42 days of M2 alone; the other standard constituents fit to nothing
    let times = hourly(1000);
    let values = tide(&times, 0.0, &[("M2", 1.0, 28.6)]);

    let fit = HarmonicAnalysis::standard().fit(&times, &values).unwrap();
    let m2 = fit.get("M2").unwrap();
    assert!((m2.amplitude - 1.0).abs() < 1e-9, "M2 H {}", m2.amplitude);
    assert!(lag_error(m2.lag_deg, 28.6) < 1e-6, "M2 G {}", m2.lag_deg);
    for name in ["S2", "K1", "O1"] {
        assert!(fit.get(name).unwrap().amplitude < 1e-9, "{name}");
    }
}

#[test]
fn test_m2_s2_separation() {
    // 400 hours (~17 days) separate M2 and S2 (14.8 days)
    let times = hourly(400);
    let values = tide(&times, 2.0, &[("M2", 1.0, 0.0), ("S2", 0.5, 45.0)]);

    let fit = HarmonicAnalysis::new(&["M2", "S2"])
        .fit(&times, &values)
        .unwrap();
    assert!((fit.mean - 2.0).abs() < 1e-9, "mean {}", fit.mean);
    let (m2, s2) = (fit.get("M2").unwrap(), fit.get("S2").unwrap());
    assert!((m2.amplitude - 1.0).abs() < 1e-9);
    assert!((s2.amplitude - 0.5).abs() < 1e-9);
    assert!(lag_error(s2.lag_deg, 45.0) < 1e-6, "S2 G {}", s2.lag_deg);
}

/// Ten days cannot separate M2 from S2: an error, where the old
/// record-time fit returned an ill-conditioned answer.
#[test]
fn test_short_record_is_an_error() {
    let times = hourly(240);
    let values = tide(&times, 0.0, &[("M2", 1.0, 0.0), ("S2", 0.5, 45.0)]);
    let err = HarmonicAnalysis::standard()
        .fit(&times, &values)
        .unwrap_err();
    assert!(
        matches!(
            err,
            AnalysisError::NotSeparable {
                a: "M2",
                b: "S2",
                ..
            }
        ),
        "{err}"
    );
    // What 10 days do resolve
    let analysis =
        HarmonicAnalysis::resolvable(&["M2", "S2", "K1", "O1"], times[239] - times[0], 1.0, &[]);
    assert_eq!(analysis.names(), ["M2", "K1"]);
}

#[test]
fn test_norwegian_coast_constituents() {
    // 33 days: M2, S2, N2, K1, O1 fitted, P1 inferred from K1 at its
    // equilibrium ratio (separating it needs half a year)
    let truth = [
        ("M2", 1.2, 5.7),
        ("S2", 0.4, 17.2),
        ("N2", 0.2, 28.6),
        ("K1", 0.15, 40.1),
        ("O1", 0.1, 51.6),
        ("P1", 0.331 * 0.15, 40.1),
    ];
    let times = hourly(800);
    let values = tide(&times, 1.5, &truth);

    let fit = HarmonicAnalysis::norwegian_coast()
        .fit(&times, &values)
        .unwrap();
    for (name, h, g) in truth {
        let c = fit.get(name).unwrap();
        assert!(
            (c.amplitude - h).abs() < 1e-8,
            "{name}: H {} vs {h}",
            c.amplitude
        );
        assert!(
            lag_error(c.lag_deg, g) < 1e-6,
            "{name}: G {} vs {g}",
            c.lag_deg
        );
    }
    assert!(fit.get("P1").unwrap().inferred);
    assert!(fit.r_squared > 1.0 - 1e-12, "R² {}", fit.r_squared);
}

#[test]
fn test_reconstruction() {
    let times = hourly(500);
    let original = tide(&times, 5.0, &[("M2", 1.0, 0.0)]);

    let fit = HarmonicAnalysis::new(&["M2"])
        .fit_series(&TimeSeries::new(&times, &original))
        .unwrap();
    let metrics = ComparisonMetrics::compute(&fit.predict(&times), &original);
    assert!(metrics.rmse < 1e-9, "RMSE {}", metrics.rmse);
}

#[test]
fn test_comparison_metrics() {
    // Perfect model
    let obs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let model_perfect = obs.clone();

    let metrics = ComparisonMetrics::compute(&model_perfect, &obs);
    assert!(metrics.rmse < 1e-10);
    assert!((metrics.correlation - 1.0).abs() < 1e-10);
    assert!((metrics.skill_score - 1.0).abs() < 1e-10);

    // Biased model
    let model_biased: Vec<f64> = obs.iter().map(|&x| x + 0.1).collect();
    let metrics_biased = ComparisonMetrics::compute(&model_biased, &obs);
    assert!((metrics_biased.bias - 0.1).abs() < 1e-10);
    assert!((metrics_biased.correlation - 1.0).abs() < 1e-10); // Still perfect correlation
}

/// Model and observation over different spans compare constituent by
/// constituent: reference constants do not depend on the record.
#[test]
fn test_harmonic_comparison() {
    let model_times = hourly(500);
    let obs_times: Vec<f64> = hourly(700)
        .into_iter()
        .map(|t| t + 9.0 * 86_400.0)
        .collect();
    let model = tide(&model_times, 0.0, &[("M2", 1.0, 30.0)]);
    let obs = tide(&obs_times, 0.0, &[("M2", 1.1, 31.0)]);

    let analysis = HarmonicAnalysis::new(&["M2"]);
    let comparisons = compare_harmonics(
        &analysis.fit(&model_times, &model).unwrap(),
        &analysis.fit(&obs_times, &obs).unwrap(),
    );
    assert_eq!(comparisons.len(), 1);
    let m2 = &comparisons[0];
    assert!((m2.amplitude_error - 0.1).abs() < 1e-9);
    assert!((m2.amplitude_ratio - 1.0 / 1.1).abs() < 1e-9);
    assert!((m2.phase_error_degrees() + 1.0).abs() < 1e-6, "{m2:?}");
}

#[test]
fn test_minimum_record_length() {
    // M2 and S2 set it for the standard four: 354.4 hours
    let min_hours = HarmonicAnalysis::standard().minimum_record_length() / 3600.0;
    assert!(
        (min_hours - 354.37).abs() < 0.1,
        "Minimum record length should be ~354 hours, got {min_hours:.1} hours"
    );
}

#[test]
fn test_time_series_methods() {
    let times = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let series = TimeSeries::new(&times, &values)
        .with_location(100.0, 200.0)
        .with_name("Test Station");

    assert_eq!(series.len(), 5);
    assert!((series.duration() - 4.0).abs() < 1e-10);
    assert!((series.mean() - 3.0).abs() < 1e-10);
    assert!(series.location.is_some());
    assert_eq!(series.name.as_deref(), Some("Test Station"));
    assert_eq!(series.times(), times.as_slice());

    // Variance of [1,2,3,4,5] = 2.5 (sample variance)
    assert!((series.variance() - 2.5).abs() < 1e-10, "Variance error");

    let shifted = series.map_values(|v| v + 0.5);
    assert!((shifted.mean() - 3.5).abs() < 1e-10);
    assert_eq!(shifted.name.as_deref(), Some("Test Station"));
}

#[test]
fn test_noisy_signal_recovery() {
    let times = hourly(1000);
    let mut values = tide(&times, 0.0, &[("M2", 1.0, 0.0)]);
    // Deterministic "noise" (small amplitude)
    for (i, v) in values.iter_mut().enumerate() {
        *v += 0.01 * (i as f64 * 0.1).sin();
    }

    let fit = HarmonicAnalysis::new(&["M2"]).fit(&times, &values).unwrap();
    let m2 = fit.get("M2").unwrap();
    assert!((m2.amplitude - 1.0).abs() < 0.02, "M2 H {}", m2.amplitude);
    assert!(fit.r_squared > 0.99, "R² {}", fit.r_squared);
}

#[test]
fn test_phase_wraparound() {
    // Lags near 360° and 0° are the same; the fit returns [0, 360)
    let times = hourly(500);
    let values = tide(&times, 0.0, &[("M2", 1.0, 354.3)]);

    let fit = HarmonicAnalysis::new(&["M2"]).fit(&times, &values).unwrap();
    let g = fit.get("M2").unwrap().lag_deg;
    assert!((0.0..360.0).contains(&g), "G {g} not in [0, 360)");
    assert!(lag_error(g, 354.3) < 1e-6, "G {g}");
}

/// An inference whose ratio comes from a reference fit (a longer record at
/// the same place) reproduces that reference's ratio in a short fit.
#[test]
fn test_inference_from_a_reference_fit() {
    let truth = [("M2", 0.74, 301.2), ("N2", 0.154, 279.1)];
    let year = hourly(24 * 366);
    let reference = HarmonicAnalysis::new(&["M2", "N2"])
        .fit(&year, &tide(&year, 0.0, &truth))
        .unwrap();

    let fortnight = hourly(24 * 15);
    let values = tide(&fortnight, 0.0, &truth);
    let fit = HarmonicAnalysis::resolvable(
        &["M2", "N2"],
        fortnight[fortnight.len() - 1] - fortnight[0],
        1.0,
        &[Inference::N2_FROM_M2.or_from(Some(&reference))],
    )
    .fit(&fortnight, &values)
    .unwrap();
    let n2 = fit.get("N2").unwrap();
    assert!(n2.inferred);
    assert!((n2.amplitude - 0.154).abs() < 1e-8, "N2 H {}", n2.amplitude);
    assert!(lag_error(n2.lag_deg, 279.1) < 1e-6, "N2 G {}", n2.lag_deg);
}

/// (name, speed °/h, f, V₀ + u °) on 2025-06-01 + 30 days hourly, from UTide
/// (`utide.harmonics.FUV`, Codiga 2011; V₀ at the first sample, f and u at
/// the record midpoint, as `HarmonicAnalysis::fit` takes them;
/// `uv run scripts/norkyst_tide_report.py fuv` prints them) and the standard
/// speeds: none of the crate's astronomy.
const JUNE_2025_UTIDE: [(&str, f64, f64, f64); 8] = [
    ("M2", 28.984_104_2, 0.964_41, 243.042 + 0.164),
    ("S2", 30.000_000_0, 1.002_35, 0.000 - 0.021),
    ("N2", 28.439_729_5, 0.963_36, 152.282 + 0.463),
    ("K1", 15.041_068_6, 1.112_17, 159.737 + 0.928),
    ("O1", 13.943_035_6, 1.178_45, 83.305 - 1.362),
    ("Q1", 13.398_660_9, 1.213_28, 352.544 + 0.876),
    ("K2", 30.082_137_3, 1.310_91, 139.475 + 1.857),
    ("P1", 14.958_931_4, 0.989_38, 200.263 + 0.001),
];

/// Mausund-sized constants (m, °), with P1 and K2 at the fit's equilibrium
/// inference ratios.
const MAUSUND_LIKE: [(&str, f64, f64); 8] = [
    ("M2", 0.740, 300.7),
    ("S2", 0.238, 342.4),
    ("N2", 0.156, 269.1),
    ("K1", 0.058, 157.7),
    ("O1", 0.044, 34.6),
    ("Q1", 0.019, 329.4),
    ("K2", 0.272 * 0.238, 342.4),
    ("P1", 0.331 * 0.058, 157.7),
];

/// The tide of `MAUSUND_LIKE` from `JUNE_2025_UTIDE`, plus `noise`.
fn june_2025_independent(times: &[f64], noise: &[f64]) -> Vec<f64> {
    times
        .iter()
        .zip(noise)
        .map(|(&t, &n)| {
            let hours = (t - times[0]) / 3600.0;
            n + MAUSUND_LIKE
                .iter()
                .map(|&(name, h, g)| {
                    let &(_, speed, f, v0u) = JUNE_2025_UTIDE.iter().find(|a| a.0 == name).unwrap();
                    f * h * (speed * hours + v0u - g).to_radians().cos()
                })
                .sum::<f64>()
        })
        .collect()
}

/// `n` hourly samples of AR(1) red noise (φ = 0.98 per hour, weather-like:
/// most of its energy below the tidal bands) scaled to `rms`, from a fixed
/// LCG seeded by `seed`.
fn red_noise(n: usize, rms: f64, seed: u64) -> Vec<f64> {
    let mut state = seed.wrapping_mul(0x2545_f491_4f6c_dd1d) | 1;
    let mut weather = 0.0;
    let mut noise: Vec<f64> = (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            weather = 0.98 * weather + ((state >> 11) as f64 / (1u64 << 53) as f64 - 0.5);
            weather
        })
        .collect();
    let scale = rms / (noise.iter().map(|v| v * v).sum::<f64>() / n as f64).sqrt();
    noise.iter_mut().for_each(|v| *v *= scale);
    noise
}

/// The analysis `tide_gauge_fit` runs on 30 days.
fn thirty_day_analysis(times: &[f64]) -> HarmonicAnalysis {
    let analysis = HarmonicAnalysis::resolvable(
        &["M2", "S2", "K1", "O1", "N2", "Q1", "K2", "P1"],
        times[times.len() - 1] - times[0],
        1.0,
        &Inference::EQUILIBRIUM,
    );
    assert_eq!(analysis.names(), ["M2", "S2", "K1", "O1", "N2", "Q1"]);
    analysis
}

/// |H e^{−iG} − H' e^{−iG'}| of a fitted constant against `MAUSUND_LIKE`.
fn complex_error(fit: &ReferenceFit, name: &str) -> f64 {
    let c = fit.get(name).unwrap();
    let &(_, h, g) = MAUSUND_LIKE.iter().find(|t| t.0 == name).unwrap();
    let (a, b) = (c.lag_deg.to_radians(), g.to_radians());
    (c.amplitude * a.cos() - h * b.cos()).hypot(c.amplitude * a.sin() - h * b.sin())
}

/// The outside judge for the June 2025 NorKyst/gauge comparisons
/// (`notes/norkyst-issues.md`), part 1: on a noise-free 30-day record built
/// from UTide's astronomy, the fit `tide_gauge_fit` runs recovers every
/// constituent to within the two astronomies' satellite terms (≤ 2.6 % in
/// f, ≤ 2° in u), N2 and Q1 included. A fit that lost N2 (NorKyst shows
/// 0.18× the gauge) or doubled Q1 would fail by an order of magnitude.
#[test]
fn test_n2_q1_recovery_against_independent_astronomy() {
    let times = hourly(720);
    let values = june_2025_independent(&times, &vec![0.0; times.len()]);
    let fit = thirty_day_analysis(&times).fit(&times, &values).unwrap();
    for name in ["M2", "S2", "N2", "K1", "O1", "Q1"] {
        let c = fit.get(name).unwrap();
        let &(_, h, _) = MAUSUND_LIKE.iter().find(|t| t.0 == name).unwrap();
        // 5 % of the amplitude (Q1: f 2.6 % and u 2° apart in the two
        // astronomies, 4.4 % in complex terms), or 0.6 mm
        let tolerance = (0.05 * h).max(6e-4);
        assert!(
            complex_error(&fit, name) < tolerance,
            "{name}: H {:.4} G {:.1}, |ΔZ| {:.5} m",
            c.amplitude,
            c.lag_deg,
            complex_error(&fit, name)
        );
    }
    let (n2, q1) = (fit.get("N2").unwrap(), fit.get("Q1").unwrap());
    assert!(
        (n2.amplitude / 0.156 - 1.0).abs() < 0.01,
        "N2 H {}",
        n2.amplitude
    );
    assert!(
        (q1.amplitude / 0.019 - 1.0).abs() < 0.04,
        "Q1 H {}",
        q1.amplitude
    );
}

/// Part 2, the same record with 0.10 m of red weather noise (the gauges'
/// residual after a 30-day fit is 0.075 m in June, 0.11–0.16 m in January)
/// over 40 realizations: N2 stays unbiased and within a few per cent, while
/// Q1 (0.019 m) scatters by tens of per cent, so a single 30-day Q1 ratio is
/// noise-limited and needs the year fit.
#[test]
fn test_thirty_day_n2_q1_under_weather_noise() {
    let times = hourly(720);
    let analysis = thirty_day_analysis(&times);
    let (mut n2, mut q1) = (Vec::new(), Vec::new());
    for seed in 1..=40 {
        let values = june_2025_independent(&times, &red_noise(times.len(), 0.10, seed));
        let fit = analysis.fit(&times, &values).unwrap();
        n2.push(fit.get("N2").unwrap().amplitude / 0.156);
        q1.push(fit.get("Q1").unwrap().amplitude / 0.019);
    }
    let stats = |r: &[f64]| {
        let mean = r.iter().sum::<f64>() / r.len() as f64;
        let sd = (r.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (r.len() - 1) as f64).sqrt();
        let max = r.iter().fold(0.0f64, |m, x| m.max((x - 1.0).abs()));
        (mean, sd, max)
    };
    let (n2_mean, n2_sd, n2_max) = stats(&n2);
    let (q1_mean, q1_sd, q1_max) = stats(&q1);
    eprintln!(
        "30-day ratios over 40 noise realizations: N2 {n2_mean:.3} ± {n2_sd:.3} (worst {n2_max:.3}), \
         Q1 {q1_mean:.3} ± {q1_sd:.3} (worst {q1_max:.3})"
    );
    // N2: unbiased, a few per cent; never anywhere near NorKyst's 0.18×
    assert!((n2_mean - 1.0).abs() < 0.01, "N2 mean ratio {n2_mean}");
    assert!(n2_sd < 0.04, "N2 ratio spread {n2_sd}");
    assert!(n2_max < 0.12, "N2 worst ratio error {n2_max}");
    // Q1: unbiased on average, but a single 30-day value is uncertain
    assert!((q1_mean - 1.0).abs() < 0.1, "Q1 mean ratio {q1_mean}");
    assert!(q1_sd < 0.4, "Q1 ratio spread {q1_sd}");
}

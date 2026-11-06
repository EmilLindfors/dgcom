//! A parent model's tide corrected from theory, scored at a tide gauge
//! without running the nested model.
//!
//! The nested model passes its parent's tide through within a centimetre
//! (`notes/validation.md` P3.4), so what a correction of the parent's tide
//! does at a gauge can be measured on the parent's own series there:
//!
//! 1. Fit the parent's series (e.g. NorKyst-800 ζ at the gauge's cell,
//!    `scripts/norkyst_gauge_series.py`) over `ratios_from..ratios_to` (half a
//!    year or more), which separates K2 from S2 and P1 from K1: the ratios
//!    the parent carries.
//! 2. Fit it over `from..to` (a month, as `norkyst_boundary_tides` builds a
//!    boundary atlas) with K2 and P1 inferred at those ratios: the raw atlas
//!    at the gauge.
//! 3. The theory atlas (`TidalAtlas::infer_from_theory`: K2 and P1 at the
//!    equilibrium ratios, the minor semidiurnals by admittance) minus the
//!    raw one is the correction `nest_tides=theory` adds to the parent
//!    (without the `skip=` constituents, to see what each contributes).
//! 4. Per day of `from..to`, the centred RMSE over hours `hours` (default
//!    3–24) against the gauge of the parent, the corrected parent and, with
//!    `pre=`, Kartverket's prediction.
//!
//! ## Run
//!
//! ```bash
//! cargo run --release --example parent_tide_check -- \
//!     data/tide_gauges/coast/BOO_obs_2026-04.txt \
//!     data/norkyst_gauges/2026-04_2026-10/BOO.txt \
//!     [from=2026-06-01] [to=2026-07-01] [ratios_from=2026-04-01] [ratios_to=2026-10-01] \
//!     [hours=3,24] [pre=data/tide_gauges/coast/BOO_pre_2026-04.txt] [skip=MU2,2N2]
//! ```

use std::collections::HashMap;
use std::path::Path;

use dg_rs::analysis::{HarmonicAnalysis, Inference, ReferenceConstant, ReferenceFit};
use dg_rs::boundary::{AtlasConstituent, AtlasPoint, TidalAtlas};
use dg_rs::io::{TideGaugeFile, read_tide_gauge_file};
use dg_rs::time::ModelClock;

/// What a month separates (as `norkyst_boundary_tides`)
const MONTH: [&str; 10] = ["M2", "S2", "N2", "K1", "O1", "Q1", "M4", "MS4", "MN4", "M6"];
/// Half a year adds K2 and P1
const HALF_YEAR: [&str; 12] = [
    "M2", "S2", "N2", "K1", "O1", "Q1", "K2", "P1", "M4", "MS4", "MN4", "M6",
];

/// Hourly samples of `file` in `[from, to)`, keyed by the whole second.
fn series(file: &TideGaugeFile, from: f64, to: f64) -> HashMap<i64, f64> {
    file.time_series
        .times()
        .iter()
        .zip(file.time_series.values())
        .filter(|&(&t, v)| v.is_finite() && t >= from && t < to)
        .map(|(&t, &v)| (t.round() as i64, v))
        .collect()
}

fn sorted(map: &HashMap<i64, f64>) -> (Vec<f64>, Vec<f64>) {
    let mut pairs: Vec<(i64, f64)> = map.iter().map(|(&t, &v)| (t, v)).collect();
    pairs.sort_by_key(|p| p.0);
    pairs.into_iter().map(|(t, v)| (t as f64, v)).unzip()
}

/// The fit as a one-point atlas (and back), to use the atlas's inference.
fn to_atlas(fit: &ReferenceFit) -> TidalAtlas {
    TidalAtlas {
        points: vec![AtlasPoint {
            lon: 0.0,
            lat: 0.0,
            depth: f64::NAN,
            mean: Some(fit.mean),
            constituents: fit
                .constants
                .iter()
                .map(|c| AtlasConstituent {
                    name: c.name,
                    eta: (c.amplitude, c.lag_deg),
                    velocity: None,
                })
                .collect(),
        }],
        header: Vec::new(),
    }
}

fn to_fit(atlas: &TidalAtlas) -> ReferenceFit {
    let p = &atlas.points[0];
    ReferenceFit {
        mean: 0.0,
        constants: p
            .constituents
            .iter()
            .map(|c| ReferenceConstant {
                name: c.name,
                amplitude: c.eta.0,
                lag_deg: c.eta.1,
                inferred: false,
            })
            .collect(),
        r_squared: f64::NAN,
        residual_rms: f64::NAN,
    }
}

/// Centred RMSE of `a − b` (the mean difference removed).
fn centred_rmse(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let d: Vec<f64> = a.iter().zip(b).map(|(x, y)| x - y).collect();
    let mean = d.iter().sum::<f64>() / n;
    (d.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n).sqrt()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args: HashMap<String, String> = HashMap::new();
    let mut files = Vec::new();
    for a in std::env::args().skip(1) {
        match a.split_once('=') {
            Some((k, v)) => {
                args.insert(k.into(), v.into());
            }
            None => files.push(a),
        }
    }
    let [gauge_path, parent_path] = files.as_slice() else {
        return Err(
            "usage: parent_tide_check <gauge file> <parent series> [from=] [to=] \
                    [ratios_from=] [ratios_to=] [hours=3,24] [pre=] [skip=]"
                .into(),
        );
    };
    let time = |key: &str, default: &str| {
        ModelClock::parse(args.get(key).map_or(default, String::as_str)).map(|c| c.epoch_unix)
    };
    let (from, to) = (time("from", "2026-06-01")?, time("to", "2026-07-01")?);
    let (r_from, r_to) = (
        time("ratios_from", "2026-04-01")?,
        time("ratios_to", "2026-10-01")?,
    );
    let hours: Vec<f64> = args
        .get("hours")
        .map_or("3,24", String::as_str)
        .split(',')
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    let [h0, h1] = hours[..] else {
        return Err("hours=first,last".into());
    };

    let gauge = read_tide_gauge_file(Path::new(gauge_path))?;
    let parent = read_tide_gauge_file(Path::new(parent_path))?;

    // 1. The parent's own K2/S2 and P1/K1
    let (t, v) = sorted(&series(&parent, r_from, r_to));
    let long = HarmonicAnalysis::new(&HALF_YEAR).fit(&t, &v)?;
    let carried = [
        Inference::from_reference("K2", "S2", &long).ok_or("no K2/S2")?,
        Inference::from_reference("P1", "K1", &long).ok_or("no P1/K1")?,
    ];
    println!(
        "Parent over {} .. {} ({:.0} days):",
        ModelClock::new(r_from).format(0.0),
        ModelClock::new(r_to).format(0.0),
        (t[t.len() - 1] - t[0]) / 86_400.0
    );
    for i in &carried {
        println!(
            "  {}/{} = {:.3}, lag {:+.1}° (equilibrium {:.3})",
            i.name,
            i.from,
            i.amplitude_ratio,
            i.lag_offset_deg,
            Inference::EQUILIBRIUM
                .iter()
                .find(|e| e.name == i.name)
                .map_or(f64::NAN, |e| e.amplitude_ratio)
        );
    }

    // 2. The raw atlas at the gauge, 3. the correction
    let parent_month = series(&parent, from, to);
    let (t, v) = sorted(&parent_month);
    let raw = HarmonicAnalysis::new(&MONTH)
        .with_inferences(&carried)
        .fit(&t, &v)?;
    let raw_atlas = to_atlas(&raw);
    let mut theory = raw_atlas.clone();
    theory.infer_from_theory()?;
    let skip: Vec<String> = args
        .get("skip")
        .map_or("", String::as_str)
        .split(',')
        .filter(|s| !s.is_empty())
        .map(str::to_uppercase)
        .collect();
    let mut correction = to_fit(&theory.difference(&raw_atlas)?);
    correction
        .constants
        .retain(|c| !skip.contains(&c.name.to_uppercase()));
    println!("\nCorrection (theory − raw) at the gauge:");
    for c in correction.constants.iter().filter(|c| c.amplitude > 1e-9) {
        println!("  {:4}  {:.4} m  {:6.1}°", c.name, c.amplitude, c.lag_deg);
    }
    let rss = correction
        .constants
        .iter()
        .map(|c| c.amplitude.powi(2))
        .sum::<f64>()
        .sqrt();
    println!("  root sum of squares {rss:.4} m");

    // 4. Daily scores on the instants all series have
    let observed = series(&gauge, from, to);
    let predicted = args
        .get("pre")
        .map(|p| read_tide_gauge_file(Path::new(p)).map(|f| series(&f, from, to)))
        .transpose()?;
    let shift = correction.predict(&t);
    let corrected: HashMap<i64, f64> = t
        .iter()
        .zip(&shift)
        .map(|(&t, &dz)| (t as i64, parent_month[&(t as i64)] + dz))
        .collect();
    println!(
        "\nCentred RMSE (cm) against the gauge, hours {h0}–{h1} of each day:\n  day          parent  corrected{}",
        if predicted.is_some() {
            "  prediction"
        } else {
            ""
        }
    );
    let mut totals = [0.0; 3];
    let mut n_days = 0;
    let mut day = from;
    while day < to {
        let keys: Vec<i64> = (0..)
            .map(|k| day as i64 + 3600 * k)
            .take_while(|&s| (s as f64) < day + 86_400.0)
            .filter(|&s| {
                let h = (s as f64 - day) / 3600.0;
                h >= h0
                    && h <= h1
                    && observed.contains_key(&s)
                    && parent_month.contains_key(&s)
                    && predicted.as_ref().is_none_or(|p| p.contains_key(&s))
            })
            .collect();
        if keys.len() >= 12 {
            let pick = |m: &HashMap<i64, f64>| keys.iter().map(|k| m[k]).collect::<Vec<_>>();
            let obs = pick(&observed);
            let scores = [
                centred_rmse(&pick(&parent_month), &obs),
                centred_rmse(&pick(&corrected), &obs),
                predicted
                    .as_ref()
                    .map_or(f64::NAN, |p| centred_rmse(&pick(p), &obs)),
            ];
            println!(
                "  {}  {:6.1}  {:9.1}{}",
                &ModelClock::new(day).format(0.0)[..10],
                100.0 * scores[0],
                100.0 * scores[1],
                if predicted.is_some() {
                    format!("  {:10.1}", 100.0 * scores[2])
                } else {
                    String::new()
                }
            );
            for (t, s) in totals.iter_mut().zip(scores) {
                *t += s * s;
            }
            n_days += 1;
        }
        day += 86_400.0;
    }
    let rms = |s: f64| 100.0 * (s / n_days as f64).sqrt();
    println!(
        "  RMS over {n_days} days  {:6.1}  {:9.1}{}",
        rms(totals[0]),
        rms(totals[1]),
        if predicted.is_some() {
            format!("  {:10.1}", rms(totals[2]))
        } else {
            String::new()
        }
    );
    Ok(())
}

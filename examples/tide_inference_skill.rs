//! What inference is worth in a short record's harmonic fit, judged by an
//! analysis that is not ours (`notes/validation.md`, P3.1 "Inference of the
//! minor constituents").
//!
//! A window of `days` is cut from each gauge's year, every `step` days, and
//! fitted by each variant below. A variant's fit is scored against UTide's
//! fit of the whole year (`scripts/utide_gauge_tide.py`: its own astronomy,
//! constituent selection and solver):
//!
//! - the main constituents against UTide's constants, as amplitude ratio,
//!   lag difference and complex difference `|ΔZ|`;
//! - the tide it predicts (no mean, no long-period constituents) against
//!   UTide's over the rest of the year, which the fit never saw, and against
//!   the observations there (the weather stays in that one);
//! - with `later=<suffix>` (e.g. `_2026`): against the observations of
//!   `<id>_obs<suffix>.txt`, a record neither fit has seen.
//!
//! Variants:
//!
//! - `none`: the major constituents the window resolves, the unresolved of
//!   P1, K2, N2 and Q1 at their equilibrium ratios, the minor semidiurnals
//!   left out (the gauge fit until 2026-11-14);
//! - `nu2`: with ν2 from N2 and M2 by linear admittance;
//! - `nu2 t2`: with ν2 and T2;
//! - `inferred`: with all six by admittance, none fitted;
//! - `equal lag`: the minor ones fitted where the window resolves them and
//!   inferred by `Inference::theory` where not, an unresolved N2 at M2's
//!   lag;
//! - `theory`: `HarmonicAnalysis::fit_gauge_record`, which is `equal lag`
//!   with N2's lag from the age of the tide (the same from 27.6 days, where
//!   N2 is fitted);
//! - `own`: `HarmonicAnalysis::fit_with_reference` with this gauge's own
//!   year as the reference (not out of sample: the bound a perfect
//!   inference would reach);
//! - `later`: the same with the `later=` record as the reference, out of
//!   sample.
//!
//! ## Run
//!
//! ```bash
//! for f in data/tide_gauges/coast/*_obs.txt; do
//!     uv run scripts/utide_gauge_tide.py output/tide_inference/utide $f; done
//! cargo run --release --example tide_inference_skill -- gauges=data/tide_gauges/coast \
//!     utide=output/tide_inference/utide [days=30] [step=30] [ids=MSU,BOO] [later=_2026] \
//!     [csv=output/tide_inference/windows.csv]
//! ```

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use dgcom::analysis::{AnalysisError, HarmonicAnalysis, Inference, ReferenceFit};
use dgcom::io::read_tide_gauge_file;
use dgcom::tides::{AdmittanceInference, constituent_period};
use dgcom::time::ModelClock;

#[path = "common/arguments.rs"]
mod arguments;

/// The constituents scored against UTide's
const SCORED: [&str; 5] = ["M2", "S2", "N2", "K1", "O1"];

/// Constituents with a period over two days: fitted, never predicted
fn long_period(name: &str) -> bool {
    constituent_period(name).is_some_and(|p| p > 2.0 * 86_400.0)
}

/// `fit`'s tide: no mean, no long-period constituents.
fn tide_of(fit: &ReferenceFit) -> ReferenceFit {
    ReferenceFit {
        mean: 0.0,
        constants: fit
            .constants
            .iter()
            .filter(|c| !long_period(c.name))
            .copied()
            .collect(),
        ..fit.clone()
    }
}

/// The finite samples of a gauge-format file.
fn series(path: &Path) -> Result<(Vec<f64>, Vec<f64>), Box<dyn std::error::Error>> {
    let file = read_tide_gauge_file(path)?;
    Ok(file
        .time_series
        .times()
        .iter()
        .copied()
        .zip(file.time_series.values().iter().copied())
        .filter(|(_, v)| v.is_finite())
        .unzip())
}

/// UTide's constants `name → (H, G°)` from `scripts/utide_gauge_tide.py`.
fn utide_constants(path: &Path) -> Result<HashMap<String, (f64, f64)>, Box<dyn std::error::Error>> {
    let mut constants = HashMap::new();
    for line in std::fs::read_to_string(path)?.lines().skip(1) {
        let fields: Vec<&str> = line.split(',').collect();
        if let [name, _, h, g] = fields[..] {
            constants.insert(name.to_string(), (h.parse()?, g.parse()?));
        }
    }
    Ok(constants)
}

/// The gauge fit's constituents before the minor semidiurnals joined them
const MAJOR: [&str; 15] = [
    "M2", "S2", "K1", "O1", "N2", "Q1", "K2", "P1", "M4", "MS4", "MN4", "M6", "Mf", "Mm", "Ssa",
];

/// Fits a window `(times, values)`
type Fit<'a> = Box<dyn Fn(&[f64], &[f64]) -> Result<ReferenceFit, AnalysisError> + 'a>;

/// A way to fit a window `(times, values)`.
struct Variant<'a> {
    name: &'static str,
    fit: Fit<'a>,
}

/// Fit what the window resolves of `candidates`, with `inferences` for the
/// rest.
fn resolved<'a>(
    name: &'static str,
    candidates: &'a [&'static str],
    inferences: Vec<Inference>,
) -> Variant<'a> {
    Variant {
        name,
        fit: Box::new(move |t, v| {
            HarmonicAnalysis::resolvable(candidates, t[t.len() - 1] - t[0], 1.0, &inferences)
                .fit(t, v)
        }),
    }
}

/// A window's score for one variant.
struct Score {
    id: String,
    start: f64,
    variant: &'static str,
    /// (ratio, lag difference °, |ΔZ| m) per `SCORED`, NaN where UTide or
    /// the fit lacks it
    constants: Vec<(f64, f64, f64)>,
    /// RMS of the predicted tide against UTide's inside and outside the
    /// window (m)
    inside: f64,
    outside: f64,
    /// RMS against the observations outside the window, each series about
    /// its own mean there (m)
    outside_obs: f64,
    /// The same on the later record (m; NaN without one)
    later_obs: f64,
}

fn rms(differences: impl Iterator<Item = f64>) -> f64 {
    let (sum, n) = differences.fold((0.0, 0usize), |(s, n), d| (s + d * d, n + 1));
    if n == 0 {
        f64::NAN
    } else {
        (sum / n as f64).sqrt()
    }
}

/// RMS of `a − b` with the mean difference taken out.
fn centred_rms(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let mean = a.iter().zip(b).map(|(a, b)| a - b).sum::<f64>() / n;
    rms(a.iter().zip(b).map(|(a, b)| a - b - mean))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = arguments::Arguments::from_env();
    let gauges = PathBuf::from(args.get("gauges").ok_or("gauges=<dir of <id>_obs.txt>")?);
    let utide = PathBuf::from(
        args.get("utide")
            .ok_or("utide=<dir of utide_gauge_tide.py>")?,
    );
    let days: f64 = args.get("days").map_or(Ok(30.0), |v| v.parse())?;
    let step: f64 = args.get("step").map_or(Ok(30.0), |v| v.parse())?;
    let later = args.get("later").cloned();
    let only: Option<Vec<String>> = args
        .get("ids")
        .map(|v| v.split(',').map(str::to_string).collect());
    let csv = args.get("csv").cloned();
    args.refuse_unread()?;

    let mut ids: Vec<String> = std::fs::read_dir(&utide)?
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter_map(|n| n.strip_suffix("_tide.txt").map(str::to_string))
        .filter(|id| only.as_ref().is_none_or(|o| o.contains(id)))
        .collect();
    ids.sort();

    let admittance = |names: &[&str]| -> Vec<Inference> {
        AdmittanceInference::MINORS
            .iter()
            .filter(|a| names.contains(&a.name))
            .filter_map(Inference::by_admittance)
            .chain(Inference::EQUILIBRIUM)
            .collect()
    };
    let mut scores: Vec<Score> = Vec::new();
    for id in &ids {
        let (t_obs, v_obs) = series(&gauges.join(format!("{id}_obs.txt")))?;
        let (t_tide, v_tide) = series(&utide.join(format!("{id}_tide.txt")))?;
        let truth_at: HashMap<i64, f64> = t_tide
            .iter()
            .map(|t| t.round() as i64)
            .zip(v_tide.iter().copied())
            .collect();
        let truth: Vec<f64> = t_obs
            .iter()
            .map(|t| truth_at[&(t.round() as i64)])
            .collect();
        let constants = utide_constants(&utide.join(format!("{id}_constants.csv")))?;
        let later_record = later
            .as_ref()
            .map(|suffix| gauges.join(format!("{id}_obs{suffix}.txt")))
            .filter(|p| p.exists())
            .map(|p| series(&p))
            .transpose()?;

        let own = HarmonicAnalysis::fit_gauge_record(&t_obs, &v_obs)?;
        let later_fit = later_record
            .as_ref()
            .map(|(t, v)| HarmonicAnalysis::fit_gauge_record(t, v))
            .transpose()?;
        let with_reference = |name, reference: &'_ ReferenceFit| {
            let reference = reference.clone();
            Variant {
                name,
                fit: Box::new(move |t, v| {
                    HarmonicAnalysis::fit_with_reference(
                        &HarmonicAnalysis::GAUGE_RECORD,
                        t,
                        v,
                        Some(&reference),
                    )
                }),
            }
        };
        let mut variants = vec![
            resolved("none", &MAJOR, Inference::EQUILIBRIUM.to_vec()),
            resolved("nu2", &MAJOR, admittance(&["NU2"])),
            resolved("nu2 t2", &MAJOR, admittance(&["NU2", "T2"])),
            resolved("inferred", &MAJOR, Inference::theory()),
            resolved(
                "equal lag",
                &HarmonicAnalysis::GAUGE_RECORD,
                Inference::theory(),
            ),
            Variant {
                name: "theory",
                fit: Box::new(HarmonicAnalysis::fit_gauge_record),
            },
            with_reference("own", &own),
        ];
        if let Some(later_fit) = &later_fit {
            variants.push(with_reference("later", later_fit));
        }

        let (first, last) = (t_obs[0], t_obs[t_obs.len() - 1]);
        let mut start = first;
        while start + days * 86_400.0 <= last + 1.0 {
            let end = start + days * 86_400.0;
            let inside: Vec<bool> = t_obs.iter().map(|&t| t >= start && t < end).collect();
            let (t, v): (Vec<f64>, Vec<f64>) = t_obs
                .iter()
                .zip(&v_obs)
                .zip(&inside)
                .filter(|(_, inside)| **inside)
                .map(|((t, v), _)| (*t, *v))
                .unzip();
            // A window with a gap of more than a tenth of it is skipped
            if (t.len() as f64) < 0.9 * days * 24.0 {
                start += step * 86_400.0;
                continue;
            }
            for variant in &variants {
                let fit = (variant.fit)(&t, &v)?;
                let tide = tide_of(&fit);
                let predicted = tide.predict(&t_obs);
                let pick = |values: &[f64], want: bool| -> Vec<f64> {
                    values
                        .iter()
                        .zip(&inside)
                        .filter(|(_, inside)| **inside == want)
                        .map(|(v, _)| *v)
                        .collect()
                };
                let against = |want: bool| {
                    rms(pick(&predicted, want)
                        .iter()
                        .zip(pick(&truth, want))
                        .map(|(p, t)| p - t))
                };
                let later_obs = later_record
                    .as_ref()
                    .map_or(f64::NAN, |(t, v)| centred_rms(&tide.predict(t), v));
                scores.push(Score {
                    id: id.clone(),
                    start,
                    variant: variant.name,
                    constants: SCORED
                        .iter()
                        .map(|name| {
                            let (Some(c), Some(&(h, g))) = (fit.get(name), constants.get(*name))
                            else {
                                return (f64::NAN, f64::NAN, f64::NAN);
                            };
                            let lag = (c.lag_deg - g + 180.0).rem_euclid(360.0) - 180.0;
                            let (a, b) = (c.lag_deg.to_radians(), g.to_radians());
                            let dz = (c.amplitude * a.cos() - h * b.cos())
                                .hypot(c.amplitude * a.sin() - h * b.sin());
                            (c.amplitude / h, lag, dz)
                        })
                        .collect(),
                    inside: against(true),
                    outside: against(false),
                    outside_obs: centred_rms(&pick(&predicted, false), &pick(&v_obs, false)),
                    later_obs,
                });
            }
            start += step * 86_400.0;
        }
    }

    if let Some(csv) = &csv {
        let mut out = std::fs::File::create(csv)?;
        write!(out, "id,start,variant")?;
        for name in SCORED {
            write!(out, ",{name}_ratio,{name}_dg_deg,{name}_dz_m")?;
        }
        writeln!(out, ",inside_m,outside_m,outside_obs_m,later_obs_m")?;
        for s in &scores {
            write!(
                out,
                "{},{},{}",
                s.id,
                &ModelClock::new(s.start).format(0.0)[..10],
                s.variant
            )?;
            for (ratio, lag, dz) in &s.constants {
                write!(out, ",{ratio:.4},{lag:.2},{dz:.5}")?;
            }
            writeln!(
                out,
                ",{:.5},{:.5},{:.5},{:.5}",
                s.inside, s.outside, s.outside_obs, s.later_obs
            )?;
        }
    }

    // The windows of all gauges together, by variant
    let mut names: Vec<&'static str> = Vec::new();
    for s in &scores {
        if !names.contains(&s.variant) {
            names.push(s.variant);
        }
    }
    let quantile = |values: &mut Vec<f64>, q: f64| -> f64 {
        values.retain(|v| v.is_finite());
        if values.is_empty() {
            return f64::NAN;
        }
        values.sort_by(|a, b| a.total_cmp(b));
        values[((values.len() - 1) as f64 * q).round() as usize]
    };
    let pooled = |values: &[f64]| rms(values.iter().copied().filter(|v| v.is_finite()));
    println!(
        "{} gauges, windows of {days} days every {step} days\n",
        ids.len()
    );
    println!(
        "variant    windows  N2 ratio: min    5 %   median   95 %    max   N2 ΔG ° 5 % .. 95 %"
    );
    let n2 = SCORED.iter().position(|&n| n == "N2").expect("scored");
    for &name in &names {
        let of: Vec<&Score> = scores.iter().filter(|s| s.variant == name).collect();
        let mut ratio: Vec<f64> = of.iter().map(|s| s.constants[n2].0).collect();
        let mut lag: Vec<f64> = of.iter().map(|s| s.constants[n2].1).collect();
        println!(
            "{name:9}  {:7}          {:6.3} {:6.3} {:7.3} {:6.3} {:6.3}   {:+7.1} .. {:+.1}",
            of.len(),
            quantile(&mut ratio, 0.0),
            quantile(&mut ratio, 0.05),
            quantile(&mut ratio, 0.5),
            quantile(&mut ratio, 0.95),
            quantile(&mut ratio, 1.0),
            quantile(&mut lag, 0.05),
            quantile(&mut lag, 0.95),
        );
    }
    println!(
        "\n|ΔZ| against UTide's year (mm, RMS over the windows), and the predicted tide (cm, RMS)"
    );
    println!(
        "variant      M2     S2     N2     K1     O1   | inside  outside  outside obs  later obs"
    );
    for &name in &names {
        let of: Vec<&Score> = scores.iter().filter(|s| s.variant == name).collect();
        print!("{name:9}");
        for j in 0..SCORED.len() {
            let dz: Vec<f64> = of.iter().map(|s| s.constants[j].2).collect();
            print!(" {:6.1}", 1e3 * pooled(&dz));
        }
        let column =
            |f: fn(&Score) -> f64| 1e2 * pooled(&of.iter().map(|s| f(s)).collect::<Vec<_>>());
        println!(
            "   | {:6.2}  {:7.2}  {:11.2}  {:9.2}",
            column(|s| s.inside),
            column(|s| s.outside),
            column(|s| s.outside_obs),
            column(|s| s.later_obs),
        );
    }

    // Per gauge: the spread of N2 and the tide outside the window
    println!("\nper gauge: N2 ratio min .. max, and the tide outside the window (cm, RMS)");
    print!("gauge");
    for &name in &names {
        print!("  {name:>20}");
    }
    println!();
    for id in &ids {
        print!("{id:5}");
        for &name in &names {
            let of: Vec<&Score> = scores
                .iter()
                .filter(|s| s.variant == name && &s.id == id)
                .collect();
            if of.is_empty() {
                print!("  {:>20}", "");
                continue;
            }
            let mut ratio: Vec<f64> = of.iter().map(|s| s.constants[n2].0).collect();
            let outside: Vec<f64> = of.iter().map(|s| s.outside).collect();
            print!(
                "  {:5.2} ..{:5.2} {:6.2}",
                quantile(&mut ratio, 0.0),
                quantile(&mut ratio, 1.0),
                1e2 * pooled(&outside)
            );
        }
        println!();
    }
    Ok(())
}

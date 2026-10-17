//! Harmonic reference constants of a tide-gauge record, optionally compared
//! with a tidal atlas or with a second series (e.g. NorKyst-800 at the gauge).
//!
//! 1. Read a gauge file (`dg_rs::io::read_tide_gauge_file`, ISO times, e.g.
//!    from `scripts/kartverket_gauge.sh`), keep `[from, to)`.
//! 2. Fit reference constants (`HarmonicAnalysis::fit`) of every constituent
//!    the record resolves (`HarmonicAnalysis::resolvable`, Rayleigh factor
//!    `rayleigh`); unresolved P1, K2, N2 and Q1 are inferred at their
//!    equilibrium ratios.
//! 3. With `atlas=<file>`: the atlas point nearest the gauge and, per
//!    constituent, the amplitude ratio, phase difference and complex
//!    difference, plus their root sum of squares.
//! 4. With `compare=<file>`: a second series in the same format (e.g.
//!    `scripts/norkyst_gauge_series.py`), both restricted to the instants
//!    they share in `[from, to)` and fitted by the same analysis, so that the
//!    two differ only in their data. Per constituent the ratio, phase
//!    difference and complex difference of the second against the gauge,
//!    the difference of their means (the datum), and with `csv=<file>` the
//!    rows appended to a CSV (header written when the file is new).
//!
//! ## Run
//!
//! ```bash
//! ./scripts/kartverket_gauge.sh Mausund 63.869331 8.665231 2024-07-01 2025-07-01
//! cargo run --release --example tide_gauge_fit -- data/tide_gauges/mausund_obs.txt \
//!     [from=2025-06-01] [to=2025-07-01] [rayleigh=1] [atlas=data/froya_station_tides.txt] \
//!     [compare=data/norkyst_gauges/2024-07_2025-07/MSU.txt] [csv=<file>] [label=MSU]
//! ```

use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::Path;

use dg_rs::analysis::{ConstituentComparison, HarmonicAnalysis, Inference, ReferenceFit};
use dg_rs::boundary::TidalAtlas;
use dg_rs::io::{TideGaugeFile, read_tide_gauge_file};
use dg_rs::time::ModelClock;

/// Candidates in priority order: a short record keeps the first ones.
const CANDIDATES: [&str; 15] = [
    "M2", "S2", "K1", "O1", "N2", "Q1", "K2", "P1", "M4", "MS4", "MN4", "M6", "Mf", "Mm", "Ssa",
];

/// Long-period constituents (mostly weather), left out of the comparison's
/// root sum of squares
const LONG_PERIOD: [&str; 3] = ["Mf", "Mm", "Ssa"];

/// Finite samples of `file` in `[from, to)`, and only at `keep` if given.
fn samples(
    file: &TideGaugeFile,
    from: Option<f64>,
    to: Option<f64>,
    keep: Option<&HashSet<i64>>,
) -> (Vec<f64>, Vec<f64>) {
    file.time_series
        .times()
        .iter()
        .copied()
        .zip(file.time_series.values().iter().copied())
        .filter(|&(t, v)| {
            v.is_finite()
                && from.is_none_or(|f| t >= f)
                && to.is_none_or(|e| t < e)
                && keep.is_none_or(|k| k.contains(&(t.round() as i64)))
        })
        .unzip()
}

fn print_fit(name: &str, times: &[f64], fit: &ReferenceFit) {
    println!(
        "{name}: {} samples, {} to {} ({:.1} days)",
        times.len(),
        ModelClock::new(times[0]).format(0.0),
        ModelClock::new(times[times.len() - 1]).format(0.0),
        (times[times.len() - 1] - times[0]) / 86_400.0
    );
    println!(
        "  Z0 {:+.3} m, R² {:.4}, residual RMS {:.3} m\n",
        fit.mean, fit.r_squared, fit.residual_rms
    );
    println!("  name    H (m)    G (°)");
    for c in &fit.constants {
        println!(
            "  {:4}  {:7.4}  {:7.2}{}",
            c.name,
            c.amplitude,
            c.lag_deg,
            if c.inferred { "  (inferred)" } else { "" }
        );
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args: HashMap<String, String> = HashMap::new();
    let mut file = None;
    for a in std::env::args().skip(1) {
        match a.split_once('=') {
            Some((k, v)) => {
                args.insert(k.into(), v.into());
            }
            None => file = Some(a),
        }
    }
    let file = file.ok_or(
        "usage: tide_gauge_fit <gauge file> [from=] [to=] [rayleigh=] [atlas=] [compare=] [csv=] [label=]",
    )?;
    let bound = |key: &str| -> Result<Option<f64>, String> {
        args.get(key)
            .map(|v| ModelClock::parse(v).map(|c| c.epoch_unix))
            .transpose()
    };
    let (from, to) = (bound("from")?, bound("to")?);
    let rayleigh: f64 = args.get("rayleigh").map_or(Ok(1.0), |v| v.parse())?;

    let gauge = read_tide_gauge_file(Path::new(&file))?;
    let other = args
        .get("compare")
        .map(|path| read_tide_gauge_file(Path::new(path)))
        .transpose()?;
    // With a second series, only the instants both have
    let shared: Option<HashSet<i64>> = other.as_ref().map(|other| {
        let (t, _) = samples(other, from, to, None);
        t.iter().map(|t| t.round() as i64).collect()
    });
    let (times, values) = samples(&gauge, from, to, shared.as_ref());
    let (Some(&t0), Some(&t1)) = (times.first(), times.last()) else {
        return Err("no samples in the window".into());
    };
    let name = gauge
        .station
        .as_ref()
        .map_or(file.clone(), |s| s.name.clone());

    // Every constituent the record resolves, the equilibrium inferences for
    // the rest
    let analysis =
        HarmonicAnalysis::resolvable(&CANDIDATES, t1 - t0, rayleigh, &Inference::EQUILIBRIUM);
    let fit = analysis.fit(&times, &values)?;
    print_fit(&name, &times, &fit);

    if let Some(path) = args.get("atlas") {
        let station = gauge
            .station
            .as_ref()
            .ok_or("the gauge file has no position")?;
        let atlas = TidalAtlas::read(Path::new(path))?;
        let (point, distance) = atlas
            .nearest(station.longitude, station.latitude)
            .ok_or("empty atlas")?;
        println!(
            "\nvs {path}: point ({:.4}, {:.4}), {:.0} m from the gauge, depth {:.0} m",
            point.lon, point.lat, distance, point.depth
        );
        println!("  name  H atlas  H gauge  ratio   ΔG (°)   |ΔZ| (m)");
        let mut rss = 0.0;
        for c in &point.constituents {
            let Some(g) = fit.get(c.name) else { continue };
            let cmp = ConstituentComparison::new(
                c.name,
                c.eta.0,
                c.eta.1.to_radians(),
                g.amplitude,
                g.lag_deg.to_radians(),
            );
            rss += cmp.complex_difference().powi(2);
            println!(
                "  {:4}  {:7.4}  {:7.4}  {:5.3}  {:+7.2}   {:.4}{}",
                c.name,
                c.eta.0,
                g.amplitude,
                cmp.amplitude_ratio,
                cmp.phase_error_degrees(),
                cmp.complex_difference(),
                if g.inferred { "  (gauge inferred)" } else { "" }
            );
        }
        println!("  root sum of squares of |ΔZ|: {:.4} m", rss.sqrt());
    }

    if let Some(other) = &other {
        let keep: HashSet<i64> = times.iter().map(|t| t.round() as i64).collect();
        let (t_other, v_other) = samples(other, from, to, Some(&keep));
        let other_name = other
            .station
            .as_ref()
            .map_or(args["compare"].clone(), |s| s.name.clone());
        let fit_other = analysis.fit(&t_other, &v_other)?;
        println!();
        print_fit(&other_name, &t_other, &fit_other);
        println!(
            "\n{other_name} against {name}, the same {} instants and analysis:",
            times.len()
        );
        println!(
            "  mean: {:+.3} against {:+.3} m, difference {:+.3} m",
            fit_other.mean,
            fit.mean,
            fit_other.mean - fit.mean
        );
        println!("  name  H other  H gauge  ratio   ΔG (°)   |ΔZ| (m)");
        let mut rss = 0.0;
        let mut rows = Vec::new();
        for g in &fit.constants {
            let c = fit_other.get(g.name).expect("the same analysis");
            let cmp = ConstituentComparison::new(
                g.name,
                c.amplitude,
                c.lag_deg.to_radians(),
                g.amplitude,
                g.lag_deg.to_radians(),
            );
            if !g.inferred && !LONG_PERIOD.contains(&g.name) {
                rss += cmp.complex_difference().powi(2);
            }
            println!(
                "  {:4}  {:7.4}  {:7.4}  {:5.3}  {:+7.2}   {:.4}{}",
                g.name,
                c.amplitude,
                g.amplitude,
                cmp.amplitude_ratio,
                cmp.phase_error_degrees(),
                cmp.complex_difference(),
                if g.inferred { "  (inferred)" } else { "" }
            );
            rows.push(format!(
                "{},{:.5},{:.2},{:.5},{:.2},{:.4},{:.2},{:.5},{}",
                g.name,
                g.amplitude,
                g.lag_deg,
                c.amplitude,
                c.lag_deg,
                cmp.amplitude_ratio,
                cmp.phase_error_degrees(),
                cmp.complex_difference(),
                g.inferred as u8
            ));
        }
        println!(
            "  root sum of squares of |ΔZ| (fitted, without Mf/Mm/Ssa): {:.4} m",
            rss.sqrt()
        );

        if let Some(csv) = args.get("csv") {
            let label = args.get("label").cloned().unwrap_or(name.clone());
            let new = !Path::new(csv).exists();
            let mut out = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(csv)?;
            if new {
                writeln!(
                    out,
                    "label,from,to,samples,constituent,h_ref,g_ref,h_cmp,g_cmp,ratio,dg_deg,dz,inferred"
                )?;
            }
            let window = format!(
                "{},{},{},",
                label,
                ModelClock::new(t0).format(0.0),
                ModelClock::new(t1).format(0.0)
            );
            writeln!(
                out,
                "{window}{},Z0,{:.5},0,{:.5},0,,,{:.5},0",
                times.len(),
                fit.mean,
                fit_other.mean,
                fit_other.mean - fit.mean
            )?;
            for row in rows {
                writeln!(out, "{window}{},{row}", times.len())?;
            }
            println!("  appended to {csv}");
        }
    }
    Ok(())
}

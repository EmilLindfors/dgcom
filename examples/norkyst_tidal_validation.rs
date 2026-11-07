//! End-to-end tidal validation against NorKyst-800, via the text-ingest reader.
//!
//! This wires the full P1.5 comparison path:
//!
//! 1. Read `norkyst-client --format text` point output with
//!    [`dgcom::io::read_norkyst_text_file`] → a sea-surface-height (`zeta`) series.
//! 2. Fit reference constants `(H, G)` with [`HarmonicAnalysis`] (the
//!    astronomy and nodal corrections are in its basis, so they compare
//!    directly with a catalogue): the constituents the record resolves, the
//!    equilibrium inferences for the rest.
//! 3. Compare `(H, G)` against a published harmonic catalogue and print skill numbers.
//!
//! ## Run
//!
//! Synthetic self-check (no data needed) — synthesizes a tide from a known
//! catalogue (with the nodal corrections), writes it as a NorKyst text file,
//! reads it back through the real reader, and confirms the fit recovers the
//! catalogue:
//!
//! ```bash
//! cargo run --example norkyst_tidal_validation --no-default-features --features parallel,simd
//! ```
//!
//! Real data. Point series (text) fetched with
//! `norkyst-client … --format text -o series.txt`:
//!
//! ```bash
//! cargo run --example norkyst_tidal_validation -- series.txt
//! ```
//!
//! Grid parquet (the reliable OPeNDAP path) fetched with
//! `norkyst-client … --bbox … --format parquet -o dataset/` — pass the dataset
//! directory (or a single `.parquet`) plus optional `lon lat` to pick the
//! nearest cell (defaults to Bergen). Requires the `parquet` feature:
//!
//! ```bash
//! cargo run --example norkyst_tidal_validation \
//!   --features parquet -- dataset/ 5.32 60.39
//! ```
//!
//! The reference constants it prints are directly comparable to a catalogue
//! (Kartverket / ROMS / NorKyst harmonic constants).

use dgcom::analysis::{HarmonicAnalysis, Inference, ReferenceConstant, ReferenceFit, TimeSeries};
use dgcom::io::read_norkyst_text_file;
use dgcom::time::ModelClock;

/// Constituents to fit in priority order: a short record keeps the first.
const CANDIDATES: [&str; 11] = [
    "M2", "S2", "K1", "O1", "N2", "Q1", "K2", "P1", "M4", "MS4", "M6",
];

/// A published harmonic constant: amplitude `h` (m) and Greenwich phase lag `g` (deg).
struct Catalogue {
    name: &'static str,
    h: f64,
    g_deg: f64,
}

/// Synthetic Bergen-like west-coast catalogue (NOT official values — for the
/// self-check only). Amplitudes in metres, Greenwich lags in degrees.
const BERGEN_LIKE: &[Catalogue] = &[
    Catalogue {
        name: "M2",
        h: 0.55,
        g_deg: 175.0,
    },
    Catalogue {
        name: "S2",
        h: 0.18,
        g_deg: 200.0,
    },
    Catalogue {
        name: "N2",
        h: 0.11,
        g_deg: 155.0,
    },
    Catalogue {
        name: "K1",
        h: 0.065,
        g_deg: 100.0,
    },
    Catalogue {
        name: "O1",
        h: 0.055,
        g_deg: 330.0,
    },
    // At K1's equilibrium ratio, as a month's fit infers it
    Catalogue {
        name: "P1",
        h: 0.331 * 0.065,
        g_deg: 100.0,
    },
];

fn main() {
    match std::env::args().nth(1) {
        Some(path) => {
            // Optional station coords (used to pick the nearest grid cell for
            // gridded parquet input); default to Bergen.
            let lon = std::env::args()
                .nth(2)
                .and_then(|s| s.parse().ok())
                .unwrap_or(5.32);
            let lat = std::env::args()
                .nth(3)
                .and_then(|s| s.parse().ok())
                .unwrap_or(60.39);
            real_data_mode(&path, lon, lat);
        }
        None => synthetic_self_check(),
    }
}

// ---------------------------------------------------------------------------
// Real-data mode
// ---------------------------------------------------------------------------

fn real_data_mode(path: &str, lon: f64, lat: f64) {
    let zeta = match load_zeta(path, lon, lat) {
        Some(z) => z,
        None => std::process::exit(1),
    };

    let analysis =
        HarmonicAnalysis::resolvable(&CANDIDATES, zeta.duration(), 1.0, &Inference::EQUILIBRIUM);
    let fit = match analysis.fit_series(&zeta) {
        Ok(fit) => fit,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };

    println!("\nReference constants:");
    println!(
        "  fit R² = {:.4}, record length = {:.1} days",
        fit.r_squared,
        zeta.duration() / 86_400.0
    );
    println!("  {:<4} {:>10} {:>10}", "name", "H (m)", "G (deg)");
    for c in &fit.constants {
        println!(
            "  {:<4} {:>10.4} {:>10.1}{}",
            c.name,
            c.amplitude,
            c.lag_deg,
            if c.inferred { "  (inferred)" } else { "" }
        );
    }
    println!(
        "\nCompare these against the station's published catalogue to score the model \
         (RMS amplitude error, RMS phase error)."
    );
}

/// Load a sea-surface-height series from either the text point format or, with
/// the `parquet` feature, a norkyst-client **grid** parquet file/dataset
/// (picking the wet cell nearest `(lon, lat)`). A multi-site text file is
/// reduced to the site nearest `(lon, lat)`. Returns `None` (after printing
/// why) if nothing usable is found.
fn load_zeta(path: &str, lon: f64, lat: f64) -> Option<TimeSeries> {
    let p = std::path::Path::new(path);
    let looks_parquet = p.is_dir() || p.extension().is_some_and(|e| e == "parquet");

    if looks_parquet {
        #[cfg(feature = "parquet")]
        {
            println!("Reading NorKyst grid parquet: {path}");
            let data = match dgcom::io::read_norkyst_parquet_glob(p) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("error: {e}");
                    return None;
                }
            };
            let zeta = data.sea_surface_height_series_nearest(lon, lat);
            println!(
                "  {} grid rows; nearest wet cell to ({lon}, {lat}) → {} zeta samples",
                data.len(),
                zeta.len()
            );
            if zeta.is_empty() {
                eprintln!("error: no wet cell with sea-surface height near ({lon}, {lat}).");
                return None;
            }
            return Some(zeta);
        }
        #[cfg(not(feature = "parquet"))]
        {
            eprintln!(
                "error: {path} looks like parquet, but this build lacks the `parquet` feature.\n\
                 Rebuild with `--features parquet` to read grid parquet natively."
            );
            return None;
        }
    }

    println!("Reading NorKyst text output: {path}");
    let data = match read_norkyst_text_file(p) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("error: {e}");
            return None;
        }
    };
    // Site-mode files interleave several stations; keep the one nearest (lon, lat).
    let data = match data.site_ids().as_slice() {
        [] | [_] => data,
        ids => {
            let dist2 = |id: i64| {
                let r = data.records_for_site(id).next().expect("site has records");
                let dlon = (r.lon - lon) * lat.to_radians().cos();
                dlon * dlon + (r.lat - lat).powi(2)
            };
            let id = *ids
                .iter()
                .min_by(|&&a, &&b| dist2(a).total_cmp(&dist2(b)))
                .expect("non-empty");
            println!(
                "  {} sites; using site {id}, nearest ({lon}, {lat})",
                ids.len()
            );
            data.for_site(id)
        }
    };
    let zeta = match data.sea_surface_height_series() {
        Ok(z) => z,
        Err(e) => {
            eprintln!("error: {e}");
            return None;
        }
    };
    println!(
        "  {} snapshots, {} with sea-surface height",
        data.len(),
        zeta.len()
    );
    if zeta.is_empty() {
        eprintln!(
            "error: no sea-surface height in the record. This needs a norkyst-client build \
             whose text writer emits the `surface …` line."
        );
        return None;
    }
    Some(zeta)
}

// ---------------------------------------------------------------------------
// Synthetic self-check
// ---------------------------------------------------------------------------

fn synthetic_self_check() {
    println!("Synthetic round-trip self-check (no external data)\n");

    // 30 days of hourly zeta from the catalogue, written in the
    // norkyst-client text format, so the real reader (not a shortcut) parses
    // it back
    let epoch_unix = ModelClock::at_datetime(2024, 6, 1, 0, 0).epoch_unix;
    let n_hours = 30 * 24;
    let times: Vec<f64> = (0..n_hours)
        .map(|i| epoch_unix + i as f64 * 3600.0)
        .collect();
    let catalogue = ReferenceFit {
        mean: 0.30,
        constants: BERGEN_LIKE
            .iter()
            .map(|c| ReferenceConstant {
                name: c.name,
                amplitude: c.h,
                lag_deg: c.g_deg,
                inferred: false,
            })
            .collect(),
        r_squared: 1.0,
        residual_rms: 0.0,
    };
    let mut text = String::new();
    for (&t, zeta) in times.iter().zip(catalogue.predict(&times)) {
        let stamp = format_utc(t as i64);
        text.push_str("site_id: 999\n");
        text.push_str(&format!("time={stamp} lat=60.39 lon=5.32\n"));
        text.push_str(&format!(
            "surface sea_surface_height=Some({zeta}) bottom_depth=Some(300.0)\n"
        ));
    }

    // Round-trip through a real file and the real reader.
    let file = std::env::temp_dir().join("norkyst_selfcheck_series.txt");
    std::fs::write(&file, &text).expect("write temp series");
    println!("Wrote synthetic series: {}", file.display());
    let data = read_norkyst_text_file(&file).expect("read back series");
    let zeta_series = data
        .sea_surface_height_series()
        .expect("single-site series");
    println!(
        "Read back {} snapshots ({} with zeta)\n",
        data.len(),
        zeta_series.len()
    );

    // A month resolves M2, S2, N2, K1 and O1; P1 is inferred from K1
    let fit = HarmonicAnalysis::norwegian_coast()
        .fit_series(&zeta_series)
        .expect("a month of hourly samples");
    print_comparison(&fit);
    let _ = std::fs::remove_file(&file);
}

/// Print the inferred reference constants next to the catalogue they came from.
fn print_comparison(fit: &ReferenceFit) {
    println!("Recovered catalogue (fit R² = {:.6}):", fit.r_squared);
    println!(
        "  {:<4} {:>8} {:>8} {:>8}   {:>8} {:>8} {:>8}",
        "name", "H_cat", "H_fit", "ΔH", "G_cat", "G_fit", "ΔG"
    );

    let mut sum_dh2 = 0.0_f64;
    let mut sum_dg2 = 0.0_f64;
    let mut n = 0.0;
    for cat in BERGEN_LIKE {
        let Some(c) = fit.get(cat.name) else {
            continue;
        };
        let h_fit = c.amplitude;
        let g_fit = c.lag_deg;
        let dh = h_fit - cat.h;
        let dg = wrap_deg_signed(g_fit - cat.g_deg);
        sum_dh2 += dh * dh;
        sum_dg2 += dg * dg;
        n += 1.0;
        println!(
            "  {:<4} {:>8.4} {:>8.4} {:>8.4}   {:>8.1} {:>8.1} {:>8.2}",
            cat.name, cat.h, h_fit, dh, cat.g_deg, g_fit, dg
        );
    }

    if n > 0.0 {
        println!(
            "\n  RMS amplitude error: {:.5} m    RMS phase error: {:.3}°",
            (sum_dh2 / n).sqrt(),
            (sum_dg2 / n).sqrt()
        );
        println!(
            "  (Both should be ~0: the reader and the fit reproduce the catalogue the \
             synthetic tide was built from.)"
        );
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Wrap a degree difference to `(−180, 180]`.
fn wrap_deg_signed(d: f64) -> f64 {
    let mut x = d.rem_euclid(360.0);
    if x > 180.0 {
        x -= 360.0;
    }
    x
}

/// Format Unix seconds as `YYYY-MM-DDTHH:MM:SSZ` (the reader accepts RFC3339).
fn format_utc(unix_s: i64) -> String {
    let days = unix_s.div_euclid(86_400);
    let secs = unix_s.rem_euclid(86_400);
    let (y, m, d) = civil_from_days(days);
    let (hh, mm, ss) = (secs / 3600, (secs % 3600) / 60, secs % 60);
    format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}Z")
}

/// Inverse of Howard Hinnant's `days_from_civil`: days-since-Unix-epoch → `(y, m, d)`.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

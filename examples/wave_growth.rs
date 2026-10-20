//! Wind-sea growth of the spectral wave model against the empirical curves.
//!
//! - `mode=duration`: one node in deep water under a steady wind from calm, the
//!   sources alone; prints `E* = g² m_0/U₁₀⁴` and `f_p* = f_p U₁₀/g` against
//!   `t* = g t/U₁₀` and the Pierson–Moskowitz limit (E* = 3.64e-3, f_p* = 0.13).
//! - `mode=fetch` (default): a deep strip downwind of a straight coast, periodic
//!   along the coast, run to the steady state; prints E* and f_p* against
//!   `X* = g X/U₁₀²` and Kahma & Calkoen's (1992) composite fit,
//!   `E* = 5.2e-7 X*^0.9`, `f_p* = 2.18 X*^−0.27`.
//!
//! ```text
//! cargo run --release --no-default-features --features parallel,simd --example wave_growth -- \
//!     mode=fetch u10=10 fetch_km=120 elements=60 order=1 hours=12 n_freq=30 n_dir=24 tail=4
//! ```
//!
//! Also `f_min=0.06 f_max=1 cfl=0.5`, `dt=10` (duration), `tail=0` for no tail and
//! `dia=0` for no quadruplets. The sources' time integration:
//! `integration=frozen|midpoint|implicit`, `limiter=ris|rate[:C]|rate2[:C]|none`
//! (`rate2` caps the losses too), and
//! `substeps=N` propagation steps per step (fetch). The gates are in
//! `tests/wave_model_test.rs`.

use std::collections::HashMap;
use std::f64::consts::TAU;
use std::sync::Arc;

use dg_rs::mesh::{Bathymetry2D, BoundaryTag, Mesh2D};
use dg_rs::operators::{DGOperators2D, GeometricFactors2D};
use dg_rs::types::ElementIndex;
use dg_rs::waves::{
    DEFAULT_LIMITER, DEFAULT_RATE_LIMITER, GrowthLimiter, SourceIntegration, SourceTerms,
    SpectralGrid, WaveModel2D, WaveWorkspace, Wind, wavenumber,
};

const G: f64 = 9.81;
const DEPTH: f64 = 1000.0;

fn main() {
    let args: HashMap<String, String> = std::env::args()
        .skip(1)
        .filter_map(|a| {
            a.split_once('=')
                .map(|(k, v)| (k.to_string(), v.to_string()))
        })
        .collect();
    let num = |key: &str, default: f64| -> f64 {
        args.get(key).map_or(default, |v| v.parse().expect(key))
    };
    let u10 = num("u10", 10.0);
    let n_freq = num("n_freq", 30.0) as usize;
    let n_dir = num("n_dir", 24.0) as usize;
    let f_min = num("f_min", 0.06);
    let f_max = num("f_max", 1.0);
    let tail = num("tail", 4.0);
    let grid = SpectralGrid::new(f_min, f_max, n_freq, n_dir);
    let sources = SourceTerms::swan_defaults(G)
        .with_tail((tail > 0.0).then_some(tail))
        .with_bottom_friction(None)
        .with_breaking(None);
    let sources = if num("dia", 1.0) > 0.0 {
        sources
    } else {
        sources.with_quadruplets(None)
    };
    let text = |key: &str, default: &str| args.get(key).map_or(default.to_string(), String::clone);
    let sources = sources
        .with_integration(match text("integration", "frozen").as_str() {
            "midpoint" => SourceIntegration::Midpoint,
            "implicit" => SourceIntegration::Implicit,
            _ => SourceIntegration::FrozenRates,
        })
        .with_limiter(match text("limiter", "ris").as_str() {
            "none" => None,
            rate if rate.starts_with("rate") => {
                let (kind, coefficient) = rate.split_once(':').unwrap_or((rate, ""));
                let coefficient = if coefficient.is_empty() {
                    DEFAULT_RATE_LIMITER
                } else {
                    coefficient.parse().expect("limiter=rate[2]:C")
                };
                Some(match kind {
                    "rate2" => GrowthLimiter::RateBothSigns(coefficient),
                    _ => GrowthLimiter::Rate(coefficient),
                })
            }
            _ => Some(GrowthLimiter::PerStep(DEFAULT_LIMITER)),
        });
    let wind = Wind {
        u10,
        direction: TAU / 4.0,
    };
    match args.get("mode").map_or("fetch", |s| s.as_str()) {
        "duration" => duration(&grid, &sources, wind, num("hours", 72.0), num("dt", 10.0)),
        _ => fetch(&grid, sources, wind, &num),
    }
}

fn dimensionless(grid: &SpectralGrid, e: &[f64], u10: f64) -> (f64, f64) {
    let p = grid.parameters(e);
    (G * G * p.m0 / u10.powi(4), u10 / (p.tp.max(1e-9) * G))
}

fn duration(grid: &SpectralGrid, sources: &SourceTerms, wind: Wind, hours: f64, dt: f64) {
    let (nc, nd) = (grid.n_components(), grid.n_dir());
    let k: Vec<f64> = grid
        .sigma
        .iter()
        .map(|&s| wavenumber(s, DEPTH, G))
        .collect();
    let mut n = vec![0.0; nc];
    let (mut e, mut a, mut b) = (vec![0.0; nc], vec![0.0; nc], vec![0.0; nc]);
    let steps = (hours * 3600.0 / dt).round() as usize;
    let mut next = 0.5;
    println!("t*       E*         f_p*     (PM: 3.64e-3, 0.13)");
    for s in 1..=steps {
        sources.integrate(grid, &mut n, &k, DEPTH, wind, dt, &mut e, &mut a, &mut b);
        let t = s as f64 * dt;
        if t / 3600.0 >= next || s == steps {
            next *= 2f64.sqrt();
            let e: Vec<f64> = (0..nc).map(|c| n[c] * grid.sigma[c / nd]).collect();
            let (es, fs) = dimensionless(grid, &e, wind.u10);
            println!(
                "{:<8.0} {:<10.3e} {:.4}  ({:.1} h)",
                G * t / wind.u10,
                es,
                fs,
                t / 3600.0
            );
        }
    }
}

fn fetch(grid: &SpectralGrid, sources: SourceTerms, wind: Wind, num: &dyn Fn(&str, f64) -> f64) {
    let length = num("fetch_km", 120.0) * 1e3;
    let n_el = num("elements", 60.0) as usize;
    let order = num("order", 1.0) as usize;
    let hours = num("hours", 12.0);
    let cfl = num("cfl", 0.5);
    // The coast at y = 0 (a wall: nothing comes in), the open sea at y = L
    let mut mesh = Mesh2D::channel_periodic_x(0.0, 1000.0, 0.0, length, 1, n_el);
    for edge in mesh.edges.iter_mut().filter(|e| e.right.is_none()) {
        let y = mesh.vertices[edge.vertices.0][1];
        if y > 0.5 * length {
            edge.boundary_tag = Some(BoundaryTag::Open);
        }
    }
    let ops = DGOperators2D::new(order);
    let geom = GeometricFactors2D::compute(&mesh, &ops);
    let bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, |_, _| -DEPTH);
    let model = WaveModel2D::new(
        Arc::new(mesh),
        Arc::new(ops),
        Arc::new(geom),
        &bathymetry,
        grid.clone(),
        G,
    )
    .with_sources(sources)
    .with_wind(wind)
    .with_substeps(num("substeps", 1.0) as usize);
    let mut n = model.zero_state();
    let mut ws = WaveWorkspace::default();
    let dt = model.compute_dt(cfl);
    let steps = (hours * 3600.0 / dt).ceil() as usize;
    let dt = hours * 3600.0 / steps as f64;
    println!("dt {dt:.2} s, {steps} steps, {} points", model.n_points());
    let clock = std::time::Instant::now();
    for _ in 0..steps {
        model.step(&mut n, dt, &mut ws);
    }
    println!("{:.1} s", clock.elapsed().as_secs_f64());
    let xy: Vec<[f64; 2]> = ElementIndex::iter(model.mesh.n_elements)
        .flat_map(|k| {
            let (mesh, ops) = (&model.mesh, &model.ops);
            (0..ops.n_nodes)
                .map(move |i| mesh.reference_to_physical(k, ops.nodes_r[i], ops.nodes_s[i]))
        })
        .collect();
    let mut e = vec![0.0; grid.n_components()];
    let mut seen = Vec::new();
    println!("X*       E*         KC92       ratio  f_p*    KC92    ratio  H_s (m) T_p (s)");
    let u10 = wind.u10;
    let mut by_fetch: Vec<usize> = (0..model.n_points()).collect();
    by_fetch.sort_by(|&a, &b| xy[a][1].total_cmp(&xy[b][1]));
    for p in by_fetch {
        let y = xy[p][1];
        if y <= 0.0 || seen.iter().any(|&s: &f64| (s - y).abs() < 1.0) {
            continue;
        }
        seen.push(y);
        model.energy_spectrum_into(&n, p, &mut e);
        let params = grid.parameters(&e);
        let (es, fs) = dimensionless(grid, &e, u10);
        let x = G * y / (u10 * u10);
        let (e_kc, f_kc) = (5.2e-7 * x.powf(0.9), 2.18 * x.powf(-0.27));
        println!(
            "{x:<8.0} {es:<10.3e} {e_kc:<10.3e} {:<6.2} {fs:<7.4} {f_kc:<7.4} {:<6.2} {:<7.3} {:.2}",
            es / e_kc,
            fs / f_kc,
            params.hs,
            params.tp
        );
    }
}

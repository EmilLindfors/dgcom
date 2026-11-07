//! Gate tests for case files (`dgcom::case`, TODO P8.1).
//!
//! 1. The shipped example, `cases/tidal_channel/case.toml`, runs (shortened)
//!    and writes the case it ran, which reads back the same.
//! 2. A tide forced through a case's open boundary into a frictionless
//!    channel closed at its head forms the standing wave of linear theory:
//!    amplitude ∝ cos k(L − x), the whole channel in phase.
//! 3. A case whose mesh and boundaries disagree, or whose station lies
//!    outside the mesh, is refused before it runs.

use std::f64::consts::PI;
use std::path::Path;

use dgcom::case::{Case, CaseError, CaseFile, StationSeries};
use dgcom::tides::constituent_period;

const EXAMPLE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/cases/tidal_channel/case.toml");

/// The example, shortened, with its output in `dir`.
fn example(dir: &Path, hours: f64) -> CaseFile {
    let overrides = [
        format!("run.duration_hours={hours}"),
        format!("output.directory={:?}", dir.display().to_string()),
    ];
    CaseFile::read(Path::new(EXAMPLE), &overrides).unwrap()
}

#[test]
fn the_example_case_runs_and_records_what_it_ran() {
    let dir = tempfile::tempdir().unwrap();
    let file = example(dir.path(), 2.0);
    let report = file.run(&mut |_| {}).unwrap();
    assert!((report.final_time_s - 7200.0).abs() < 1e-6);
    assert_eq!(report.stations.len(), 3);
    for s in &report.stations {
        // 0, 30, 60, 90, 120 minutes
        assert_eq!(s.time_s.len(), 5, "{}: {:?}", s.name, s.time_s);
        assert!(s.eta_m.iter().chain(&s.u_m_s).all(|v| v.is_finite()));
    }
    let mouth = &report.stations[0];
    assert!(
        mouth.eta_m.last().unwrap().abs() > 0.1,
        "the tide reached the mouth: {:?}",
        mouth.eta_m
    );
    let resolved = std::fs::read_to_string(dir.path().join("case.resolved.toml")).unwrap();
    assert_eq!(Case::parse(&resolved, false, &[]).unwrap(), file.case);
    let csv = std::fs::read_to_string(dir.path().join("stations.csv")).unwrap();
    assert_eq!(csv.lines().count(), 1 + 3 * 5);
    assert!(dir.path().join("tidal_channel.dgsnap").exists());
}

/// Amplitude and phase (rad) of the M2 fit to `s` from time `from` on.
fn m2_fit(s: &StationSeries, from: f64) -> (f64, f64) {
    let omega = 2.0 * PI / constituent_period("M2").unwrap();
    // Least squares for η = a cos ωt + b sin ωt + c
    let mut m = [[0.0; 3]; 3];
    let mut r = [0.0; 3];
    for (&t, &eta) in s.time_s.iter().zip(&s.eta_m).filter(|(t, _)| **t >= from) {
        let basis = [(omega * t).cos(), (omega * t).sin(), 1.0];
        for i in 0..3 {
            r[i] += basis[i] * eta;
            for j in 0..3 {
                m[i][j] += basis[i] * basis[j];
            }
        }
    }
    // Gaussian elimination on the 3 × 3 normal equations
    for p in 0..3 {
        for row in p + 1..3 {
            let f = m[row][p] / m[p][p];
            for col in p..3 {
                m[row][col] -= f * m[p][col];
            }
            r[row] -= f * r[p];
        }
    }
    let mut x = [0.0; 3];
    for p in (0..3).rev() {
        let tail: f64 = (p + 1..3).map(|c| m[p][c] * x[c]).sum();
        x[p] = (r[p] - tail) / m[p][p];
    }
    (x[0].hypot(x[1]), x[1].atan2(x[0]))
}

/// The M2 amplitude at the mouth and the head station, and their phase
/// difference (degrees), of a tide of `amplitude` forced with
/// `boundary.velocity = velocity` into a frictionless channel `length` long
/// and `depth` deep, closed at its head.
fn standing_wave(
    velocity: &str,
    length: f64,
    depth: f64,
    amplitude: f64,
    x: [f64; 2],
) -> (f64, f64, f64) {
    let dir = tempfile::tempdir().unwrap();
    let text = format!(
        r#"
        [run]
        duration_hours = 62.0

        [mesh.rectangle]
        x_m = [0.0, {length}]
        y_m = [0.0, 2000.0]
        nx = 30
        ny = 1
        west = "open"

        [bathymetry]
        depth_m = {depth}

        [physics]
        wetting_drying = false

        [[boundary]]
        tag = "open"
        velocity = "{velocity}"
        tide = {{ constituents = [{{ name = "M2", amplitude_m = {amplitude} }}], ramp_hours = 3.0 }}

        [output]
        directory = {dir:?}
        interval_minutes = 20.0
        snapshot = false

        [[output.station]]
        name = "mouth"
        x_m = {x0}
        y_m = 1000.0

        [[output.station]]
        name = "head"
        x_m = {x1}
        y_m = 1000.0
        "#,
        dir = dir.path().display().to_string(),
        x0 = x[0],
        x1 = x[1],
    );
    let case = Case::parse(&text, false, &[]).unwrap();
    case.validate().unwrap();
    let file = CaseFile {
        case,
        base_dir: dir.path().to_path_buf(),
    };
    let report = file.run(&mut |_| {}).unwrap();
    // The last two M2 periods, after five of spin-up
    let from = 62.0 * 3600.0 - 2.0 * constituent_period("M2").unwrap();
    let (a_mouth, phase_mouth) = m2_fit(&report.stations[0], from);
    let (a_head, phase_head) = m2_fit(&report.stations[1], from);
    (a_mouth, a_head, (phase_head - phase_mouth).to_degrees())
}

/// Linear theory: a wave entering at amplitude `a_in` and reflected at the
/// head stands as `2 a_in cos k(L − x)`; the at-rest boundary admits half
/// the given amplitude (`a_in = A/2`), the incoming-wave one all of it.
#[test]
fn a_tide_through_a_case_boundary_forms_the_standing_wave_of_theory() {
    let (length, depth, amplitude) = (60_000.0, 10.0, 0.1);
    let x = [3_000.0, 59_000.0];
    let k = 2.0 * PI / constituent_period("M2").unwrap() / (9.81_f64 * depth).sqrt();
    let shape = |x: f64| (k * (length - x)).cos();
    for (velocity, a_in) in [("at_rest", 0.5 * amplitude), ("incoming_wave", amplitude)] {
        let (a_mouth, a_head, lag) = standing_wave(velocity, length, depth, amplitude, x);
        let (mouth, head) = (2.0 * a_in * shape(x[0]), 2.0 * a_in * shape(x[1]));
        println!(
            "standing wave, {velocity}: mouth {a_mouth:.4} m (theory {mouth:.4}), \
             head {a_head:.4} m (theory {head:.4}), phase difference {lag:.2}°"
        );
        for (model, theory) in [(a_mouth, mouth), (a_head, head)] {
            assert!(
                (model / theory - 1.0).abs() < 0.02,
                "{velocity}: {model} m against {theory} m"
            );
        }
        assert!(lag.abs() < 3.0, "{velocity}: head and mouth {lag}° apart");
    }
}

/// The example with `edit` applied to its text, as a case file in `dir`.
fn edited_example(dir: &Path, edit: impl Fn(String) -> String) -> Result<CaseFile, CaseError> {
    let text = edit(std::fs::read_to_string(EXAMPLE).unwrap());
    let path = dir.join("case.toml");
    std::fs::write(&path, text).unwrap();
    CaseFile::read(&path, &[])
}

#[test]
fn open_faces_without_a_boundary_are_refused() {
    let dir = tempfile::tempdir().unwrap();
    let file = edited_example(dir.path(), |t| {
        t.replace("west = \"open\"", "west = \"open\"\nnorth = \"tidal\"")
    })
    .unwrap();
    let err = file.build_domain().err().unwrap().to_string();
    assert!(
        err.contains("Tidal") && err.contains("[[boundary]]"),
        "{err}"
    );
}

#[test]
fn a_boundary_for_faces_the_mesh_lacks_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let file = edited_example(dir.path(), |t| {
        format!(
            "{t}
[[boundary]]
tag = \"river\"
still = {{}}
"
        )
    })
    .unwrap();
    let err = file.build_domain().err().unwrap().to_string();
    assert!(err.contains("river"), "{err}");
}

#[test]
fn a_station_outside_the_mesh_is_refused_before_the_run() {
    let dir = tempfile::tempdir().unwrap();
    let file = edited_example(dir.path(), |t| {
        t.replace("x_m = 34000.0", "x_m = 50000.0")
            .replace("duration_hours = 25.0", "duration_hours = 1000.0")
    })
    .unwrap();
    let err = file.run(&mut |_| {}).err().unwrap().to_string();
    assert!(err.contains("head") && err.contains("outside"), "{err}");
}

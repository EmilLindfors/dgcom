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
    assert_eq!(
        Case::parse(&resolved, false, &[]).unwrap(),
        file.case.resolved()
    );
    let csv = std::fs::read_to_string(dir.path().join("stations.csv")).unwrap();
    assert_eq!(csv.lines().count(), 1 + 3 * 5);
    assert!(dir.path().join("tidal_channel.dgsnap").exists());
}

/// Stations every 10 minutes with the fields every 30: the stations are
/// recorded every 10 minutes, the snapshot every 30, and the run is the
/// one that writes everything every 10 minutes, value for value (its steps
/// land on the station times).
#[test]
fn stations_can_be_recorded_more_often_than_the_fields() {
    let run = |overrides: &[&str]| {
        let dir = tempfile::tempdir().unwrap();
        let mut all = vec![
            "run.duration_hours=2.0".to_string(),
            format!("output.directory={:?}", dir.path().display().to_string()),
        ];
        all.extend(overrides.iter().map(|s| s.to_string()));
        let file = CaseFile::read(Path::new(EXAMPLE), &all).unwrap();
        let report = file.run(&mut |_| {}).unwrap();
        let frames = dgcom::io::SnapshotReader::open(dir.path().join("tidal_channel.dgsnap"))
            .unwrap()
            .n_frames()
            .unwrap();
        (report, frames)
    };
    let (split, split_frames) = run(&["output.station_interval_minutes=10.0"]);
    let (every, every_frames) = run(&["output.interval_minutes=10.0"]);
    assert_eq!((split_frames, every_frames), (5, 13));
    for (a, b) in split.stations.iter().zip(&every.stations) {
        assert_eq!(a.time_s.len(), 13, "{}", a.name);
        assert_eq!(
            (&a.time_s, &a.eta_m, &a.u_m_s),
            (&b.time_s, &b.eta_m, &b.u_m_s)
        );
    }
    assert_eq!(split.n_steps, every.n_steps);

    let dir = tempfile::tempdir().unwrap();
    let err = CaseFile::read(
        Path::new(EXAMPLE),
        &[
            "output.station_interval_minutes=20.0".to_string(),
            format!("output.directory={:?}", dir.path().display().to_string()),
        ],
    )
    .err()
    .unwrap()
    .to_string();
    assert!(err.contains("whole multiple"), "{err}");
}

/// A station is scored against its gauge after the spin-up (by default the
/// tide's 2 h ramp): here a gauge that read the station's own surface 10 cm
/// high through the run, and a pure M2 for a month on either side. The
/// skill against it is a bias of −10 cm with nothing else, over the hourly
/// samples from 2 h to the end (14 h), and both score files are written.
#[test]
fn a_station_is_scored_against_its_gauge() {
    let hours = 14.0;
    let dir = tempfile::tempdir().unwrap();
    let plain = example(dir.path(), hours).run(&mut |_| {}).unwrap();
    let mouth = &plain.stations[0];

    // 2025-06-15T00:00:00Z, the example's run.start
    let start_unix = 1_749_945_600.0;
    let mut record = String::from(
        "# station: mouth\n# longitude: 8.5\n# latitude: 63.5\n# columns: time(s) water_level(m)\n",
    );
    for h in -720..=744 {
        let t = 3600.0 * f64::from(h);
        let level = match mouth.time_s.iter().position(|&s| (s - t).abs() < 1e-6) {
            Some(i) => mouth.eta_m[i] + 0.1,
            None => 0.8 * (2.0 * std::f64::consts::PI * t / (12.420_601 * 3600.0)).cos(),
        };
        record += &format!("{} {level}\n", start_unix + t);
    }
    let gauge = dir.path().join("gauge.txt");
    std::fs::write(&gauge, record).unwrap();

    let dir = tempfile::tempdir().unwrap();
    let file = CaseFile::read(
        Path::new(EXAMPLE),
        &[
            format!("run.duration_hours={hours}"),
            format!("output.directory={:?}", dir.path().display().to_string()),
            format!("output.station.0.gauge={:?}", gauge.display().to_string()),
        ],
    )
    .unwrap();
    assert_eq!(file.case.spinup_hours(), 2.0);
    let report = file.run(&mut |_| {}).unwrap();
    // The gauge changes nothing in the run
    assert_eq!(report.stations[0].eta_m, mouth.eta_m);
    let g = report.stations[0].gauge.as_ref().unwrap();
    let v = g.vs_observations.as_ref().unwrap();
    println!(
        "mouth vs its gauge: bias {:+.4} m, centred {:.1e} m, r {:.6}, {} samples; vs prediction {:?}",
        v.bias,
        v.centred_rmse,
        v.correlation,
        v.n,
        g.vs_prediction.as_ref().map(|p| p.centred_rmse)
    );
    assert_eq!(v.n, 13, "hours 2 to 14");
    assert!((v.bias + 0.1).abs() < 1e-9 && v.centred_rmse < 1e-9);
    assert!((v.correlation - 1.0).abs() < 1e-9);
    assert!(g.vs_prediction.is_ok());
    assert!(report.stations[1].gauge.is_none());
    let names: Vec<_> = g.constituents.iter().map(|c| c.name).collect();
    assert!(names.contains(&"M2"), "{names:?}");

    let skill = std::fs::read_to_string(dir.path().join("gauge_skill.csv")).unwrap();
    assert_eq!(skill.lines().count(), 3, "{skill}");
    assert!(
        skill
            .lines()
            .nth(1)
            .unwrap()
            .starts_with("mouth,observations,13,")
    );
    let constants = std::fs::read_to_string(dir.path().join("gauge_constituents.csv")).unwrap();
    assert_eq!(constants.lines().count(), 1 + names.len(), "{constants}");
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

// ---------------------------------------------------------------------------
// Placed meshes and bed rasters
// ---------------------------------------------------------------------------

/// Write a geographic (EPSG:4326) Float32 GeoTIFF of `width` × `height`
/// pixels over `[west, south, east, north]`, pixel (row, col) holding
/// `elevation(lon, lat)` at its centre.
fn write_raster(
    path: &Path,
    [west, south, east, north]: [f64; 4],
    (width, height): (u32, u32),
    elevation: impl Fn(f64, f64) -> f64,
) {
    use tiff::encoder::{TiffEncoder, colortype::Gray32Float};
    use tiff::tags::Tag;
    let (dlon, dlat) = (
        (east - west) / f64::from(width),
        (north - south) / f64::from(height),
    );
    let data: Vec<f32> = (0..height)
        .flat_map(|row| (0..width).map(move |col| (row, col)))
        .map(|(row, col)| {
            let lon = west + (f64::from(col) + 0.5) * dlon;
            let lat = north - (f64::from(row) + 0.5) * dlat;
            elevation(lon, lat) as f32
        })
        .collect();
    let mut encoder = TiffEncoder::new(std::fs::File::create(path).unwrap()).unwrap();
    let mut image = encoder.new_image::<Gray32Float>(width, height).unwrap();
    let scale = [dlon, dlat, 0.0];
    let tiepoint = [0.0, 0.0, 0.0, west, north, 0.0];
    // Geographic model, PixelIsArea, WGS 84
    let keys: [u16; 16] = [1, 1, 0, 3, 1024, 0, 1, 2, 1025, 0, 1, 1, 2048, 0, 1, 4326];
    let e = image.encoder();
    e.write_tag(Tag::ModelPixelScaleTag, &scale[..]).unwrap();
    e.write_tag(Tag::ModelTiepointTag, &tiepoint[..]).unwrap();
    e.write_tag(Tag::GeoKeyDirectoryTag, &keys[..]).unwrap();
    image.write_data(&data).unwrap();
}

const BAY: [f64; 4] = [8.0, 63.5, 8.2, 63.6];

/// A bay over `BAY`: 20 m deep at its west side, shoaling eastwards to land
/// beyond 8.133°E (up to +10 m), with a round island near its middle.
fn bay_elevation(lon: f64, lat: f64) -> f64 {
    let island = ((lon - 8.07) * 0.445).hypot(lat - 63.55) < 0.006;
    if island {
        3.0
    } else {
        -20.0 + 30.0 * (lon - 8.0) / 0.2
    }
}

/// A case on the bay raster in `dir`, with `extra` appended.
fn bay_case(dir: &Path, extra: &str) -> Result<CaseFile, CaseError> {
    write_raster(&dir.join("bay.tif"), BAY, (60, 30), bay_elevation);
    let text = format!(
        r#"
        [run]
        duration_hours = 1.0

        [mesh.geographic]
        bbox = {BAY:?}
        nx = 12
        ny = 8

        [bathymetry.raster]
        file = "bay.tif"
        kind = "topobathy"

        [physics]
        coriolis = "beta_plane"
        friction = {{ manning = 0.025 }}

        [[boundary]]
        tag = "open"
        still = {{}}

        [output]
        directory = "out"
        interval_minutes = 15.0

        [[output.station]]
        name = "bay"
        lon = 8.05
        lat = 63.53
        {extra}
        "#
    );
    let path = dir.join("case.toml");
    std::fs::write(&path, text).unwrap();
    CaseFile::read(&path, &[])
}

/// Largest speed and surface departure over a run's outputs.
fn rest_departure(file: &CaseFile) -> (f64, f64, usize) {
    let domain = file.build_domain().unwrap();
    let (mut speed, mut eta) = (0.0_f64, 0.0_f64);
    let report = file
        .run_on(&domain, &mut |p| {
            speed = speed.max(p.max_speed);
            eta = eta.max(p.eta_range.0.abs()).max(p.eta_range.1.abs());
        })
        .unwrap();
    assert!(report.stations[0].eta_m.iter().all(|e| e.abs() < 1e-9));
    (speed, eta, domain.mesh.n_elements)
}

/// A lake at rest over a raster bed with land, shores that are partly dry,
/// an island and open sides holds still water to round-off: the bed reaches
/// the nodes, the land is dropped and the coastline walled without upsetting
/// the well-balanced wet/dry scheme; with local time stepping too.
#[test]
fn a_placed_raster_bay_at_rest_stays_at_rest() {
    let dir = tempfile::tempdir().unwrap();
    let file = bay_case(dir.path(), "").unwrap();
    let domain = file.build_domain().unwrap();
    assert!(
        domain.mesh.n_elements < 12 * 8,
        "the land east of 8.133°E is dropped: {} elements",
        domain.mesh.n_elements
    );
    assert!(domain.bathymetry.max() > 0.0, "shore nodes on land remain");
    let (speed, eta, _) = rest_departure(&file);
    println!("bay at rest: |u| ≤ {speed:.2e} m/s, |η| ≤ {eta:.2e} m");
    assert!(speed < 1e-9 && eta < 1e-9, "|u| {speed}, |η| {eta}");

    let dir = tempfile::tempdir().unwrap();
    let file = bay_case(
        dir.path(),
        "\n[numerics]\nlocal_time_stepping_levels = 2\nintegrator = \"ssp_rk3\"\n",
    )
    .unwrap();
    let (speed, eta, _) = rest_departure(&file);
    println!("bay at rest, local time stepping: |u| ≤ {speed:.2e} m/s, |η| ≤ {eta:.2e} m");
    assert!(speed < 1e-9 && eta < 1e-9, "|u| {speed}, |η| {eta}");
}

#[test]
fn a_raster_that_misses_the_mesh_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let file = bay_case(dir.path(), "").unwrap();
    // The same raster, somewhere else
    write_raster(
        &dir.path().join("bay.tif"),
        [10.0, 60.0, 10.2, 60.1],
        (60, 30),
        bay_elevation,
    );
    let err = file.build_domain().err().unwrap().to_string();
    assert!(err.contains("covers"), "{err}");
}

#[test]
fn a_station_on_dropped_land_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let file = bay_case(
        dir.path(),
        "\n[[output.station]]\nname = \"inland\"\nlon = 8.19\nlat = 63.55\n",
    )
    .unwrap();
    let err = file.run(&mut |_| {}).err().unwrap().to_string();
    assert!(err.contains("inland") && err.contains("dropped"), "{err}");
}

/// A gauge on a quay is on land in the mesh: it samples the nearest element
/// in open water (at least 3 m deep), whose surface is the water's, and
/// with `open_water_depth_m = 0` it is refused where it is.
#[test]
fn a_gauge_on_land_beside_the_water_is_moved_into_open_water() {
    let quay = "\n[[output.station]]\nname = \"quay\"\nlon = 8.14\nlat = 63.53\n";
    let dir = tempfile::tempdir().unwrap();
    let file = bay_case(dir.path(), quay).unwrap();
    let domain = file.build_domain().unwrap();
    let moved = &domain.stations[1];
    println!("quay: sampled {:.0} m from its position", moved.offset_m);
    assert!(
        moved.offset_m > 1000.0 && moved.offset_m < 3000.0,
        "{}",
        moved.offset_m
    );
    assert_eq!(
        domain.stations[0].offset_m, 0.0,
        "the bay's is in open water"
    );
    assert!(
        domain.notes.iter().any(|n| n.contains("\"quay\" sampled")),
        "{:?}",
        domain.notes
    );
    let report = file.run_on(&domain, &mut |_| {}).unwrap();
    assert!(report.stations[1].eta_m.iter().all(|e| e.abs() < 1e-9));

    let dir = tempfile::tempdir().unwrap();
    let file = bay_case(dir.path(), &format!("{quay}open_water_depth_m = 0\n")).unwrap();
    let err = file.build_domain().err().unwrap().to_string();
    assert!(err.contains("quay") && err.contains("dropped"), "{err}");
}

// ---------------------------------------------------------------------------
// Tidal atlases
// ---------------------------------------------------------------------------

/// Where the placed channels below lie: x = y = 0 here.
const ORIGIN: (f64, f64) = (63.7, 8.5);

/// An atlas of the level only, with `constituents` (name, amplitude, lag)
/// at points (x, y) of the placed mesh.
fn write_atlas(path: &Path, points: &[(f64, f64)], constituents: &[(&str, f64, f64)]) {
    use dgcom::io::{CoordinateProjection, LocalProjection};
    let projection = LocalProjection::new(ORIGIN.0, ORIGIN.1);
    let mut text = String::from("# test atlas\n# lon lat depth name eta_amp eta_lag\n");
    for &(x, y) in points {
        let (lat, lon) = projection.xy_to_geo(x, y);
        for (name, amp, lag) in constituents {
            text += &format!("{lon:.8} {lat:.8} 10.0 {name} {amp} {lag}\n");
        }
    }
    std::fs::write(path, text).unwrap();
}

/// A placed channel 30 km long, 2 km wide and 10 m deep, open at its west
/// end, with the given boundary entry; its stations' records over 6 h.
fn placed_channel(dir: &Path, boundary: &str) -> Result<Vec<StationSeries>, CaseError> {
    let text = format!(
        r#"
        [run]
        start = "2025-06-15T00:00:00Z"
        duration_hours = 6.0

        [mesh]
        origin = {{ lat = {}, lon = {} }}

        [mesh.rectangle]
        x_m = [0.0, 30000.0]
        y_m = [0.0, 2000.0]
        nx = 15
        ny = 1
        west = "open"

        [bathymetry]
        depth_m = 10.0

        [[boundary]]
        tag = "open"
        {boundary}

        [output]
        directory = "out"
        interval_minutes = 30.0
        snapshot = false

        [[output.station]]
        name = "mouth"
        x_m = 1000.0
        y_m = 1000.0

        [[output.station]]
        name = "head"
        x_m = 29000.0
        y_m = 1000.0
        "#,
        ORIGIN.0, ORIGIN.1
    );
    let path = dir.join("case.toml");
    std::fs::write(&path, text).unwrap();
    Ok(CaseFile::read(&path, &[])?.run(&mut |_| {})?.stations)
}

/// An atlas holding one tide at every point gives the boundary the same
/// tide as `tide` with those constants: both are nodal-corrected from
/// `run.start`, with the lags as Greenwich lags.
#[test]
fn a_uniform_atlas_forces_the_same_tide_as_its_harmonic_constants() {
    let constituents = [("M2", 0.3, 40.0), ("S2", 0.1, 75.0), ("K1", 0.05, 200.0)];
    let dir = tempfile::tempdir().unwrap();
    let points: Vec<(f64, f64)> = (0..=4).map(|i| (0.0, 500.0 * i as f64)).collect();
    write_atlas(&dir.path().join("atlas.txt"), &points, &constituents);
    let atlas = placed_channel(dir.path(), "atlas = { file = \"atlas.txt\" }").unwrap();

    let dir = tempfile::tempdir().unwrap();
    let list: Vec<String> = constituents
        .iter()
        .map(|(n, a, g)| format!("{{ name = \"{n}\", amplitude_m = {a}, phase_deg = {g} }}"))
        .collect();
    let tide = placed_channel(
        dir.path(),
        &format!("tide = {{ constituents = [{}] }}", list.join(", ")),
    )
    .unwrap();

    let mut largest = 0.0_f64;
    for (a, t) in atlas.iter().zip(&tide) {
        for (ea, et) in a.eta_m.iter().zip(&t.eta_m) {
            largest = largest.max((ea - et).abs());
        }
    }
    let range = tide[1].eta_m.iter().fold(0.0_f64, |m, e| m.max(e.abs()));
    println!(
        "atlas against harmonic tide: largest difference {largest:.2e} m (|η| up to {range:.3} m)"
    );
    assert!(range > 0.1, "the tide reached the head: {range}");
    assert!(largest < 1e-9, "{largest} m");
}

/// Open faces farther than `coverage_m` from the atlas become walls, up to
/// `max_closed` of them; beyond, the case is refused.
#[test]
fn an_atlas_walls_the_open_faces_it_does_not_cover() {
    let dir = tempfile::tempdir().unwrap();
    // Points along the lower quarter of the mouth only
    let points: Vec<(f64, f64)> = (0..=2).map(|i| (0.0, 250.0 * i as f64)).collect();
    write_atlas(&dir.path().join("atlas.txt"), &points, &[("M2", 0.3, 40.0)]);
    let text = |max_closed: f64| {
        format!(
            r#"
            [run]
            start = "2025-06-15T00:00:00Z"
            duration_hours = 1.0

            [mesh]
            origin = {{ lat = {}, lon = {} }}

            [mesh.rectangle]
            x_m = [0.0, 8000.0]
            y_m = [0.0, 8000.0]
            nx = 4
            ny = 8
            west = "open"

            [bathymetry]
            depth_m = 10.0

            [[boundary]]
            tag = "open"
            atlas = {{ file = "atlas.txt", coverage_m = 1500.0, max_closed = {max_closed} }}
            "#,
            ORIGIN.0, ORIGIN.1
        )
    };
    let path = dir.path().join("case.toml");
    std::fs::write(&path, text(0.05)).unwrap();
    let err = CaseFile::read(&path, &[])
        .unwrap()
        .build_domain()
        .err()
        .unwrap();
    assert!(err.to_string().contains("max_closed"), "{err}");

    std::fs::write(&path, text(0.9)).unwrap();
    let domain = CaseFile::read(&path, &[]).unwrap().build_domain().unwrap();
    let open = (0..domain.mesh.n_elements)
        .flat_map(|k| (0..4).map(move |f| (k, f)))
        .filter(|&(k, f)| {
            domain
                .mesh
                .boundary_tag(dgcom::types::ElementIndex::new(k), f)
                == Some(dgcom::mesh::BoundaryTag::Open)
        })
        .count();
    // The faces of 1 km along the mouth whose nodes lie within 1.5 km of a
    // point at y ≤ 500 m: y ∈ [0, 2000] m
    assert_eq!(open, 2, "{:?}", domain.notes);
}

/// A year's hourly gauge record (2025) of M2 and S2 (amplitudes, lags) at
/// (x, y) of the placed mesh, in the format of `scripts/kartverket_gauge.sh`.
/// A year separates every constituent of the gauge fit, so none is inferred
/// at a ratio the record lacks.
fn write_gauge(path: &Path, (x, y): (f64, f64), m2: (f64, f64), s2: (f64, f64)) {
    use dgcom::io::{CoordinateProjection, LocalProjection};
    use dgcom::time::ModelClock;
    let (lat, lon) = LocalProjection::new(ORIGIN.0, ORIGIN.1).xy_to_geo(x, y);
    // Greenwich lags need V₀ + u, which the clock gives
    let clock = ModelClock::parse("2025-01-01T00:00:00Z").unwrap();
    let mut text = format!(
        "# station: Test\n# id: TST\n# latitude: {lat}\n# longitude: {lon}\n# datum: MSL\n# units: m\n\
         # columns: datetime water_level(m)\n"
    );
    const DAYS: [u32; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    for hour in 0..365 * 24 {
        let t = 3600.0 * f64::from(hour);
        let level: f64 = [("M2", m2), ("S2", s2)]
            .iter()
            .map(|&(name, (amp, lag))| {
                let drift = clock.nodal_drift(name).unwrap().at(t);
                let omega = 2.0 * PI / constituent_period(name).unwrap();
                drift.f * amp * (omega * t + drift.phase_offset_rad() - f64::to_radians(lag)).cos()
            })
            .sum();
        let (mut day, h) = (hour / 24, hour % 24);
        let mut month = 0;
        while day >= DAYS[month] {
            day -= DAYS[month];
            month += 1;
        }
        text += &format!(
            "2025-{:02}-{:02}T{h:02}:00:00Z {level:.5}\n",
            month + 1,
            day + 1
        );
    }
    std::fs::write(path, text).unwrap();
}

/// An atlas corrected at a gauge takes the gauge's M2 and S2 where its
/// source is the gauge's, and keeps its structure: the gain is the gauge's
/// constant over the source's.
#[test]
fn an_atlas_is_corrected_at_a_gauge() {
    let dir = tempfile::tempdir().unwrap();
    let mouth: Vec<(f64, f64)> = (0..=4).map(|i| (0.0, 500.0 * i as f64)).collect();
    // The boundary atlas and the source at the gauge, both from "the model":
    // M2 0.3 m, 40°; S2 0.1 m, 75°
    let model = [("M2", 0.3, 40.0), ("S2", 0.1, 75.0)];
    write_atlas(&dir.path().join("atlas.txt"), &mouth, &model);
    write_atlas(&dir.path().join("source.txt"), &[(5000.0, 1000.0)], &model);
    // The gauge sees M2 1.2× and 10° later, S2 0.9× and 5° earlier
    write_gauge(
        &dir.path().join("gauge.txt"),
        (5200.0, 1000.0),
        (0.36, 50.0),
        (0.09, 70.0),
    );
    let boundary = r#"[boundary.atlas]
        file = "atlas.txt"
        [boundary.atlas.gauge]
        record = "gauge.txt"
        source = "source.txt"
        gains = ["M2", "S2"]
        ratios = []"#;
    std::fs::write(dir.path().join("case.toml"), "").unwrap();
    // `placed_channel` runs it; read its notes from a domain build first
    let records = placed_channel(dir.path(), boundary).unwrap();
    let file = CaseFile::read(&dir.path().join("case.toml"), &[]).unwrap();
    let notes = file.build_domain().unwrap().notes.join("\n");
    println!("{notes}");
    // "M2 × 1.199, lag +10.0°": the gain and lag the domain's log records
    let gain = |name: &str| -> (f64, f64) {
        let rest = notes.split(&format!("{name} × ")).nth(1).expect(name);
        let (gain, rest) = rest.split_once(", lag ").unwrap();
        let lag = rest.split('°').next().unwrap();
        (gain.parse().unwrap(), lag.parse().unwrap())
    };
    // The fit's nodal treatment over the year costs about 0.1 %
    for (name, expected) in [("M2", (1.2, 10.0)), ("S2", (0.9, -5.0))] {
        let (g, lag) = gain(name);
        assert!(
            (g / expected.0 - 1.0).abs() < 0.003 && (lag - expected.1).abs() < 0.2,
            "{name}: × {g}, {lag}° against × {}, {}°",
            expected.0,
            expected.1
        );
    }
    assert!(records[1].eta_m.iter().any(|e| e.abs() > 0.2));

    // A source with no point near the gauge is refused
    write_atlas(&dir.path().join("source.txt"), &[(20000.0, 1000.0)], &model);
    let err = CaseFile::read(&dir.path().join("case.toml"), &[])
        .unwrap()
        .build_domain()
        .err()
        .unwrap();
    assert!(err.to_string().contains("no point within"), "{err}");
}

// ---------------------------------------------------------------------------
// The bed treatments of `froya_real_data`: wall land, the 3D bed
// ---------------------------------------------------------------------------

/// A fjord over `BAY`, south to north: a 25 m shelf, a 60 m basin, a strip
/// of shore at +0.5 m (dry at mean sea level, wet at high water) and land
/// at +3 m with a dip to +0.5 m in it, a pond at high water.
fn fjord_elevation(lon: f64, lat: f64) -> f64 {
    let dip = ((lon - 8.1) / 0.012).hypot((lat - 63.585) / 0.006) < 1.0;
    match lat {
        _ if dip => 0.5,
        l if l < 63.53 => -25.0,
        l if l < 63.56 => -60.0,
        l if l < 63.57 => 0.5,
        _ => 3.0,
    }
}

/// A case on the fjord raster in `dir`, sampled at the nodes, its land
/// kept, with `bathymetry` appended to its `[bathymetry]`.
fn fjord_case(dir: &Path, bathymetry: &str) -> CaseFile {
    write_raster(&dir.join("fjord.tif"), BAY, (200, 100), fjord_elevation);
    let text = format!(
        r#"
        [run]
        duration_hours = 1.0

        [mesh]
        drop_land = false
        [mesh.geographic]
        bbox = {BAY:?}
        nx = 12
        ny = 8

        [bathymetry]
        {bathymetry}
        [bathymetry.raster]
        file = "fjord.tif"
        kind = "topobathy"
        sampling = "point"

        [physics]
        friction = {{ manning = 0.025 }}

        [[boundary]]
        tag = "open"
        still = {{}}
        "#
    );
    let path = dir.join("case.toml");
    std::fs::write(&path, text).unwrap();
    CaseFile::read(&path, &[]).unwrap()
}

/// `[bathymetry.slopes_3d]` is `froya_real_data slopes3d=on`: the library's
/// element slopes, cliff shores and tidal ponds, in its order, each
/// followed by the isolated wet nodes raised, on the raster's bed.
#[test]
fn the_3d_bed_is_the_librarys_in_the_examples_order() {
    let dir = tempfile::tempdir().unwrap();
    let plain = fjord_case(dir.path(), "").build_domain().unwrap();
    let (mesh, ops, geom) = (&plain.mesh, &plain.ops, &plain.geom);
    let mut bed = (*plain.bathymetry).clone();
    let s = bed.smooth_element_slopes(
        mesh,
        ops,
        geom,
        dgcom::mesh::ElementSlopeBound::for_pycnocline(19.0),
        0.1,
    );
    let c = bed.raise_cliff_shores(mesh, ops, geom, 20.0, 0.1, 2.0, 2.0);
    bed.raise_isolated_wet_nodes(mesh, ops, geom, 0.0);
    let p = bed.fill_tidal_ponds(mesh, ops, geom, -1.43, 2.0, ops.n_nodes);
    bed.raise_isolated_wet_nodes(mesh, ops, geom, 0.0);
    println!(
        "fjord: {} nodes smoothed, {} cliff shore nodes raised, {} ponds ({} nodes) filled",
        s.changed, c.nodes, p.ponds, p.nodes
    );
    assert!(
        s.changed > 0 && c.nodes > 0 && p.nodes > 0,
        "every step acts on the fjord"
    );

    let dir = tempfile::tempdir().unwrap();
    let treated = fjord_case(dir.path(), "[bathymetry.slopes_3d]")
        .build_domain()
        .unwrap();
    assert_eq!(treated.mesh.n_elements, mesh.n_elements);
    assert_eq!(treated.bathymetry.data, bed.data);
    for step in ["smoothed for 3D", "cliff shores", "tidal ponds"] {
        assert!(
            treated.notes.iter().any(|n| n.contains(step)),
            "{step}: {:?}",
            treated.notes
        );
    }

    // Each step can be left out
    let dir = tempfile::tempdir().unwrap();
    let smoothed = fjord_case(
        dir.path(),
        "[bathymetry.slopes_3d]\nraise_cliffs = false\nfill_ponds = false",
    )
    .build_domain()
    .unwrap();
    let mut only = (*plain.bathymetry).clone();
    only.smooth_element_slopes(
        mesh,
        ops,
        geom,
        dgcom::mesh::ElementSlopeBound::for_pycnocline(19.0),
        0.1,
    );
    assert_eq!(smoothed.bathymetry.data, only.data);
}

/// `lower_wall_land` is `froya_real_data wall_land=lower`: a channel whose
/// bed rises to +1 m at its north wall has the wall's land lowered to the
/// water beside it, before anything else is done to the bed.
#[test]
fn wall_land_is_lowered_to_the_water_beside_it() {
    let case = |extra: &str| {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("case.toml");
        std::fs::write(
            &path,
            format!(
                r#"
                [run]
                duration_hours = 1.0
                [mesh]
                drop_land = false
                [mesh.rectangle]
                x_m = [0.0, 4000.0]
                y_m = [0.0, 1000.0]
                nx = 4
                ny = 2
                [bathymetry]
                ramp = {{ from_m = [0.0, 0.0], to_m = [0.0, 1000.0], depth_m = [20.0, -1.0] }}
                {extra}
                "#
            ),
        )
        .unwrap();
        CaseFile::read(&path, &[]).unwrap().build_domain().unwrap()
    };
    let plain = case("");
    let lowered = case("lower_wall_land = true");
    let mut bed = (*plain.bathymetry).clone();
    let w = bed.lower_wall_land(&plain.mesh, &plain.ops, &plain.geom, -2.0);
    assert!(w.changed > 0 && w.volume > 0.0, "{w:?}");
    assert_eq!(lowered.bathymetry.data, bed.data);
    assert!(
        lowered.bathymetry.max() < 0.0,
        "the wall's land is under water: {}",
        lowered.bathymetry.max()
    );
    assert!(
        lowered
            .notes
            .iter()
            .any(|n| n.contains("wall land lowered"))
    );
}

// ---------------------------------------------------------------------------
// Nesting in a parent model (NetCDF)
// ---------------------------------------------------------------------------

/// 2025-06-15 00:00 UTC, the placed channels' `run.start`, in Unix seconds.
#[cfg(feature = "netcdf")]
const START_UNIX: f64 = 1_749_945_600.0;

/// A parent model's output on a regular longitude/latitude grid over x in
/// `x_m` and y in [-10, 12] km of the placed mesh: hourly from `run.start`
/// for `hours`, ζ = `zeta` and the current at rest everywhere, `depth_m`
/// deep.
#[cfg(feature = "netcdf")]
fn write_parent(path: &Path, x_m: [f64; 2], hours: usize, zeta: f64, depth_m: f64) {
    use dgcom::io::{CoordinateProjection, LocalProjection};
    let projection = LocalProjection::new(ORIGIN.0, ORIGIN.1);
    let (lat0, lon0) = projection.xy_to_geo(x_m[0], -10_000.0);
    let (lat1, lon1) = projection.xy_to_geo(x_m[1], 12_000.0);
    let n = 12;
    let axis = |a: f64, b: f64| -> Vec<f64> {
        (0..n)
            .map(|i| a + (b - a) * i as f64 / (n - 1) as f64)
            .collect()
    };
    let mut file = netcdf::create(path).unwrap();
    file.add_dimension("time", hours + 1).unwrap();
    file.add_dimension("lat", n).unwrap();
    file.add_dimension("lon", n).unwrap();
    let mut var = file.add_variable::<f64>("lat", &["lat"]).unwrap();
    var.put_values(&axis(lat0, lat1), ..).unwrap();
    let mut var = file.add_variable::<f64>("lon", &["lon"]).unwrap();
    var.put_values(&axis(lon0, lon1), ..).unwrap();
    let mut var = file.add_variable::<f64>("time", &["time"]).unwrap();
    var.put_attribute("units", "seconds since 1970-01-01 00:00:00")
        .unwrap();
    let times: Vec<f64> = (0..=hours)
        .map(|h| START_UNIX + 3600.0 * h as f64)
        .collect();
    var.put_values(&times, ..).unwrap();
    let mut var = file.add_variable::<f32>("h", &["lat", "lon"]).unwrap();
    var.put_values(&vec![depth_m as f32; n * n], ..).unwrap();
    let field = |value: f64| vec![value as f32; (hours + 1) * n * n];
    for (name, value) in [
        ("zeta", zeta),
        ("ubar_eastward", 0.0),
        ("vbar_northward", 0.0),
    ] {
        let mut var = file
            .add_variable::<f32>(name, &["time", "lat", "lon"])
            .unwrap();
        var.put_values(&field(value), ..).unwrap();
    }
}

/// A placed channel 10 km long, 2 km wide and 10 m deep, nested at its west
/// end with `nesting` (a `[boundary.nesting]` body), run for `hours`.
#[cfg(feature = "netcdf")]
fn nested_channel(dir: &Path, nesting: &str, hours: f64) -> CaseFile {
    let text = format!(
        r#"
        [run]
        start = "2025-06-15T00:00:00Z"
        duration_hours = {hours}

        [mesh]
        origin = {{ lat = {}, lon = {} }}

        [mesh.rectangle]
        x_m = [0.0, 10000.0]
        y_m = [0.0, 2000.0]
        nx = 10
        ny = 2
        west = "open"

        [bathymetry]
        depth_m = 10.0

        [physics]
        friction = {{ manning = 0.025 }}

        [[boundary]]
        tag = "open"
        [boundary.nesting]
        {nesting}

        [output]
        directory = "out"
        interval_minutes = 30.0
        snapshot = false

        [[output.station]]
        name = "head"
        x_m = 9500.0
        y_m = 1000.0
        "#,
        ORIGIN.0, ORIGIN.1
    );
    let path = dir.join("case.toml");
    std::fs::write(&path, text).unwrap();
    CaseFile::read(&path, &[]).unwrap()
}

/// The channel fills to the parent's level less its mean: `level_m`, else
/// minus the mean level `Z0` of the parent's atlas. Without either, the
/// parent's own mean needs a month of record, and a few hours are refused.
#[cfg(feature = "netcdf")]
#[test]
fn a_nested_channel_fills_to_its_parents_level_less_its_mean() {
    let dir = tempfile::tempdir().unwrap();
    write_parent(
        &dir.path().join("parent.nc"),
        [-10_000.0, 25_000.0],
        8,
        0.3,
        10.0,
    );
    let mouth: Vec<(f64, f64)> = (0..=4).map(|i| (0.0, 500.0 * i as f64)).collect();
    write_atlas(
        &dir.path().join("atlas.txt"),
        &mouth,
        &[("Z0", 0.1, 0.0), ("M2", 0.0, 0.0)],
    );
    // A slow ramp next to the channel's quarter-wave period (≈ 1.1 h)
    for datum in ["level_m = -0.1", "atlas = \"atlas.txt\""] {
        let file = nested_channel(
            dir.path(),
            &format!("file = \"parent.nc\"\nramp_hours = 2.0\n{datum}"),
            5.0,
        );
        let report = file.run(&mut |_| {}).unwrap();
        let head = &report.stations[0].eta_m;
        let last = head[head.len() - 1];
        println!("{datum}: η at the head {last:.4} m");
        assert!((last - 0.2).abs() < 0.01, "{datum}: η {last} m");
        let filled = report.volume_m3.1 - report.volume_m3.0;
        assert!(
            (filled / (0.2 * 10_000.0 * 2_000.0) - 1.0).abs() < 0.05,
            "{filled} m³"
        );
    }
    let err = nested_channel(dir.path(), "file = \"parent.nc\"", 1.0)
        .build_domain()
        .err()
        .unwrap();
    assert!(err.to_string().contains("days"), "{err}");
}

/// A parent with no water near the open faces walls them, up to
/// `max_closed`; past it, the case is refused. A run longer than the
/// parent's record is refused before it starts.
#[cfg(feature = "netcdf")]
#[test]
fn a_parent_that_misses_the_boundary_or_the_run_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    // Cut for a domain 5 km further east: nothing within 2 km of the mouth
    write_parent(
        &dir.path().join("east.nc"),
        [5_000.0, 25_000.0],
        8,
        0.3,
        10.0,
    );
    let err = nested_channel(dir.path(), "file = \"east.nc\"\nlevel_m = 0.0", 1.0)
        .build_domain()
        .err()
        .unwrap();
    assert!(err.to_string().contains("max_closed"), "{err}");

    write_parent(
        &dir.path().join("short.nc"),
        [-10_000.0, 25_000.0],
        2,
        0.3,
        10.0,
    );
    let file = nested_channel(dir.path(), "file = \"short.nc\"\nlevel_m = 0.0", 3.0);
    let err = file.build_domain().err().unwrap();
    println!("{err}");
    assert!(err.to_string().contains("short.nc"), "{err}");
    nested_channel(dir.path(), "file = \"short.nc\"\nlevel_m = 0.0", 2.0)
        .build_domain()
        .unwrap();
}

/// The bed is blended to the parent's across the band: the parent's depth
/// on the boundary, the case's own beyond the band, and in between a
/// weight that falls with the distance. `blend_bed = false` keeps it.
#[cfg(feature = "netcdf")]
#[test]
fn the_bed_is_blended_to_the_parents_across_the_band() {
    let dir = tempfile::tempdir().unwrap();
    write_parent(
        &dir.path().join("parent.nc"),
        [-10_000.0, 25_000.0],
        4,
        0.0,
        20.0,
    );
    let nesting = "file = \"parent.nc\"\nlevel_m = 0.0\nband_km = 3.0";
    let domain = nested_channel(dir.path(), nesting, 1.0)
        .build_domain()
        .unwrap();
    let positions = domain.node_positions();
    let (mut at_mouth, mut beyond, mut between) = (Vec::new(), Vec::new(), Vec::new());
    for (&[x, _], &b) in positions.iter().zip(&domain.bathymetry.data) {
        match x {
            x if x < 1.0 => at_mouth.push(b),
            x if x > 3_001.0 => beyond.push(b),
            x if (1_400.0..1_600.0).contains(&x) => between.push(b),
            _ => {}
        }
    }
    assert!(
        at_mouth.iter().all(|&b| (b + 20.0).abs() < 1e-9),
        "{at_mouth:?}"
    );
    assert!(beyond.iter().all(|&b| b == -10.0), "{beyond:?}");
    assert!(
        between.iter().all(|&b| b < -10.5 && b > -19.5),
        "{between:?}"
    );

    let kept = nested_channel(dir.path(), &format!("{nesting}\nblend_bed = false"), 1.0)
        .build_domain()
        .unwrap();
    assert!(kept.bathymetry.data.iter().all(|&b| b == -10.0));
}

// ---------------------------------------------------------------------------
// The atmosphere (NetCDF)
// ---------------------------------------------------------------------------

/// Hourly weather from `run.start` for `hours` on a regular
/// longitude/latitude grid over the placed channel: a uniform 10 m wind of
/// `u10` m/s towards the east and a uniform sea-level pressure `msl_pa`.
#[cfg(feature = "netcdf")]
fn write_weather(path: &Path, hours: usize, u10: f64, msl_pa: f64) {
    use dgcom::io::{CoordinateProjection, LocalProjection};
    let projection = LocalProjection::new(ORIGIN.0, ORIGIN.1);
    let (lat0, lon0) = projection.xy_to_geo(-10_000.0, -10_000.0);
    let (lat1, lon1) = projection.xy_to_geo(20_000.0, 12_000.0);
    let n = 8;
    let axis = |a: f64, b: f64| -> Vec<f64> {
        (0..n)
            .map(|i| a + (b - a) * i as f64 / (n - 1) as f64)
            .collect()
    };
    let mut file = netcdf::create(path).unwrap();
    file.add_dimension("time", hours + 1).unwrap();
    file.add_dimension("lat", n).unwrap();
    file.add_dimension("lon", n).unwrap();
    let mut var = file.add_variable::<f64>("lat", &["lat"]).unwrap();
    var.put_values(&axis(lat0, lat1), ..).unwrap();
    let mut var = file.add_variable::<f64>("lon", &["lon"]).unwrap();
    var.put_values(&axis(lon0, lon1), ..).unwrap();
    let mut var = file.add_variable::<f64>("time", &["time"]).unwrap();
    var.put_attribute("units", "seconds since 1970-01-01 00:00:00")
        .unwrap();
    let times: Vec<f64> = (0..=hours)
        .map(|h| START_UNIX + 3600.0 * h as f64)
        .collect();
    var.put_values(&times, ..).unwrap();
    let field = |value: f64| vec![value as f32; (hours + 1) * n * n];
    for (name, value, units) in [
        ("u10", u10, "m s-1"),
        ("v10", 0.0, "m s-1"),
        ("msl", msl_pa, "Pa"),
    ] {
        let mut var = file
            .add_variable::<f32>(name, &["time", "lat", "lon"])
            .unwrap();
        var.put_attribute("units", units).unwrap();
        var.put_values(&field(value), ..).unwrap();
    }
}

/// A placed channel 10 km long, 2 km wide and 10 m deep, still water at its
/// open west end (`boundary` appended to its `[[boundary]]`), under the
/// weather of `weather.nc` for `hours`, with stations at its mouth and head.
#[cfg(feature = "netcdf")]
fn weathered_channel(dir: &Path, boundary: &str, hours: f64) -> Vec<StationSeries> {
    let text = format!(
        r#"
        [run]
        start = "2025-06-15T00:00:00Z"
        duration_hours = {hours}

        [mesh]
        origin = {{ lat = {}, lon = {} }}

        [mesh.rectangle]
        x_m = [0.0, 10000.0]
        y_m = [0.0, 2000.0]
        nx = 10
        ny = 2
        west = "open"

        [bathymetry]
        depth_m = 10.0

        [physics]
        friction = {{ manning = 0.025 }}

        [[boundary]]
        tag = "open"
        still = {{}}
        {boundary}

        [atmosphere]
        files = ["weather.nc"]

        [output]
        directory = "out"
        interval_minutes = 30.0
        snapshot = false

        [[output.station]]
        name = "mouth"
        x_m = 500.0
        y_m = 1000.0

        [[output.station]]
        name = "head"
        x_m = 9500.0
        y_m = 1000.0
        "#,
        ORIGIN.0, ORIGIN.1
    );
    let path = dir.join("case.toml");
    std::fs::write(&path, text).unwrap();
    CaseFile::read(&path, &[])
        .unwrap()
        .run(&mut |_| {})
        .unwrap()
        .stations
}

/// Under a uniform low the open boundary takes the inverse-barometer level
/// and the channel settles to it, −Δp/(ρg); with `inverse_barometer = 0`
/// the boundary holds mean sea level, and a uniform pressure, which has no
/// gradient, moves nothing.
#[cfg(feature = "netcdf")]
#[test]
fn a_low_lowers_the_channel_to_its_inverse_barometer_level() {
    let dir = tempfile::tempdir().unwrap();
    let dp = 1000.0;
    write_weather(&dir.path().join("weather.nc"), 13, 0.0, 101_325.0 + dp);
    let ib = -dp / (1025.0 * 9.81);
    let full = weathered_channel(dir.path(), "", 12.0);
    let half = weathered_channel(dir.path(), "inverse_barometer = 0.5", 12.0);
    let none = weathered_channel(dir.path(), "inverse_barometer = 0.0", 12.0);
    let last = |s: &[StationSeries]| *s[1].eta_m.last().unwrap();
    println!(
        "head after 12 h under +10 hPa: {:.5} m (theory {ib:.5}), half {:.5}, none {:.2e}",
        last(&full),
        last(&half),
        last(&none)
    );
    assert!(
        (last(&full) - ib).abs() < 0.005 * ib.abs(),
        "{}",
        last(&full)
    );
    assert!(
        (last(&half) - 0.5 * ib).abs() < 0.005 * ib.abs(),
        "{}",
        last(&half)
    );
    assert!(last(&none).abs() < 1e-9, "{}", last(&none));
}

/// A uniform wind towards the closed head sets the sea up against it: at
/// rest, the slope balances the stress, ∂η/∂x = τ/(ρ g h), so the head
/// stands τ L/(ρ g h) above the mouth (L the 9 km between the stations).
#[cfg(feature = "netcdf")]
#[test]
fn a_wind_sets_the_channel_up_against_its_head() {
    use dgcom::source::{DragCoefficient, RHO_AIR, RHO_WATER};
    let dir = tempfile::tempdir().unwrap();
    let wind = 10.0;
    write_weather(&dir.path().join("weather.nc"), 25, wind, 101_325.0);
    let stations = weathered_channel(dir.path(), "", 24.0);
    let tau = RHO_AIR * DragCoefficient::LargePond.compute(wind) * wind * wind;
    let theory = tau * 9000.0 / (RHO_WATER * 9.81 * 10.0);
    let setup = stations[1].eta_m.last().unwrap() - stations[0].eta_m.last().unwrap();
    println!("set-up over 9 km under {wind} m/s: {setup:.5} m (theory {theory:.5})");
    assert!(
        (setup - theory).abs() < 0.01 * theory,
        "{setup} against {theory}"
    );
}

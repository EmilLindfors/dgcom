//! Gate tests for `[mesh.patch]`: a case run on a cut-out of its mesh.
//!
//! 1. The patch is the whole mesh's elements within the radius, in their
//!    order, with the whole bed bit for bit; the stations outside it and a
//!    boundary whose faces it cut away are left out, the others kept in the
//!    same element.
//! 2. Started from the whole run's 2D restart, the patch's first record is
//!    the whole run's at that time, bit for bit; walled, it keeps its
//!    volume; a restart of another case is refused by key, and the patch's
//!    own restart resumes as any run's.
//! 3. The same from a 3D run's restart: the columns, their history and the
//!    first record.

use std::path::Path;

use dgcom::case::{CaseFile, Domain, StationSeries};
use dgcom::types::ElementIndex;

const EXAMPLE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/cases/tidal_channel/case.toml");

fn example(dir: &Path, extra: &[String]) -> CaseFile {
    let mut overrides = vec![format!("output.directory={:?}", dir.display().to_string())];
    overrides.extend(extra.iter().cloned());
    CaseFile::read(Path::new(EXAMPLE), &overrides).unwrap()
}

/// The element whose centre is nearest `(x, y)`.
fn element_near(domain: &Domain, [x, y]: [f64; 2]) -> usize {
    let mesh = &domain.mesh;
    (0..mesh.n_elements)
        .min_by(|&a, &b| {
            let d = |k| {
                let [cx, cy] = mesh.reference_to_physical(ElementIndex::new(k), 0.0, 0.0);
                (cx - x).hypot(cy - y)
            };
            d(a).total_cmp(&d(b))
        })
        .unwrap()
}

fn patch_overrides(element: usize, radius_km: f64, faces: &str) -> Vec<String> {
    vec![
        format!("mesh.patch.element={element}"),
        format!("mesh.patch.radius_km={radius_km}"),
        format!("mesh.patch.faces={faces:?}"),
    ]
}

#[test]
fn a_patch_is_the_whole_meshs_elements_and_bed() {
    let dir = tempfile::tempdir().unwrap();
    let whole = example(dir.path(), &[]).build_domain().unwrap();
    // At the head: the flats, 3 km round the element at (34.5, 1.5) km
    let k = element_near(&whole, [34500.0, 1500.0]);
    let file = example(dir.path(), &patch_overrides(k, 3.0, "wall"));
    let domain = file.build_domain().unwrap();
    let patch = domain.patch.as_ref().unwrap();

    let centre =
        |d: &Domain, e: usize| d.mesh.reference_to_physical(ElementIndex::new(e), 0.0, 0.0);
    let [x0, y0] = centre(&whole, k);
    let expected: Vec<usize> = (0..whole.mesh.n_elements)
        .filter(|&e| {
            let [x, y] = centre(&whole, e);
            (x - x0).hypot(y - y0) <= 3000.0
        })
        .collect();
    assert_eq!(patch.whole_elements, expected);
    assert_eq!(patch.whole_n_elements, whole.mesh.n_elements);
    assert_eq!(domain.mesh.n_elements, expected.len());
    let nn = domain.ops.n_nodes;
    for (i, &e) in expected.iter().enumerate() {
        assert_eq!(
            domain.mesh.element_vertices(ElementIndex::new(i)),
            whole.mesh.element_vertices(ElementIndex::new(e))
        );
        let bed = |d: &Domain, k: usize| d.bathymetry.data[k * nn..][..nn].to_vec();
        let (a, b) = (bed(&domain, i), bed(&whole, e));
        assert!(
            a.iter().zip(&b).all(|(a, b)| a.to_bits() == b.to_bits()),
            "element {e}: {a:?} against {b:?}"
        );
    }

    // Walled: the mouth's open boundary is not in it, nor the two stations
    // down the channel
    assert!(domain.open_tags.is_empty(), "{:?}", domain.open_tags);
    let noted = |what: &str| domain.notes.iter().any(|n| n.contains(what));
    assert!(noted(
        "[[boundary]] tag = \"open\": none of its faces is in the patch"
    ));
    assert!(noted("2 of 3 stations outside it, left out: mouth, middle"));
    assert!(noted(&format!("patch: {} of 160 elements", expected.len())));
    let (head, whole_head) = (&domain.stations[0], &whole.stations[2]);
    assert_eq!(head.name, "head");
    assert_eq!(
        patch.whole_element(head.level.element().as_usize()),
        whole_head.level.element().as_usize()
    );
    assert_eq!(head.level.position(), whole_head.level.position());
    assert_eq!(head.level.weights(), whole_head.level.weights());

    // Open by default: the cut takes the case's open boundary
    let open = example(dir.path(), &patch_overrides(k, 3.0, "open"));
    let domain = open.build_domain().unwrap();
    assert_eq!(domain.open_tags.len(), 1);
    assert!(!domain.notes.iter().any(|n| n.contains("[[boundary]] tag")));
    // Its own folder, inside the whole run's, without output.directory
    let unset = CaseFile::read(Path::new(EXAMPLE), &patch_overrides(k, 3.0, "open")).unwrap();
    assert!(
        unset
            .output_dir()
            .ends_with(format!("output/tidal_channel/patch_e{k}_3km")),
        "{}",
        unset.output_dir().display()
    );
}

/// The record of `name` at `t` in `stations`: η, u, v.
fn record_at(stations: &[StationSeries], name: &str, t: f64) -> [f64; 3] {
    let s = stations.iter().find(|s| s.name == name).unwrap();
    let i = s.time_s.iter().position(|&x| x == t).unwrap();
    [s.eta_m[i], s.u_m_s[i], s.v_m_s[i]]
}

#[test]
fn a_patch_starts_from_the_whole_runs_restart() {
    let whole_dir = tempfile::tempdir().unwrap();
    let whole_file = example(
        whole_dir.path(),
        &[
            "run.duration_hours=1.0".into(),
            "restart.interval_hours=1.0".into(),
        ],
    );
    let whole = whole_file.run(&mut |_| {}).unwrap();
    let restart = whole_dir.path().join("tidal_channel.restart");
    let from = format!("restart.from={:?}", restart.display().to_string());
    let k = element_near(&whole_file.build_domain().unwrap(), [34500.0, 1500.0]);

    let patch = |dir: &Path, extra: &[&str]| {
        let mut o = patch_overrides(k, 3.0, "wall");
        o.extend(extra.iter().map(|s| s.to_string()));
        example(dir, &o)
    };
    let dir = tempfile::tempdir().unwrap();
    let report = patch(
        dir.path(),
        &[
            &from,
            "run.duration_hours=1.5",
            "restart.interval_hours=0.5",
        ],
    )
    .run(&mut |_| {})
    .unwrap();
    assert!(
        report.notes.iter().any(
            |n| n.starts_with("patch started from the whole run's restart")
                && n.contains("at 1.00 h")
        ),
        "{:?}",
        report.notes
    );
    // Its records start at the restart's time, the first the whole run's
    let head = &report.stations[0];
    assert_eq!(head.time_s, vec![3600.0, 5400.0]);
    let (a, b) = (
        record_at(&report.stations, "head", 3600.0),
        record_at(&whole.stations, "head", 3600.0),
    );
    assert!(
        a.iter().zip(&b).all(|(a, b)| a.to_bits() == b.to_bits()),
        "{a:?} against {b:?}"
    );
    // Walled, it keeps its water
    let (v0, v1) = report.volume_m3;
    assert!((v1 - v0).abs() <= 1e-12 * v0, "{v0} → {v1}");
    assert!(head.eta_m.iter().all(|e| e.is_finite()));

    // A restart of another case is refused, by its key
    let other = tempfile::tempdir().unwrap();
    let err = patch(
        other.path(),
        &[
            &from,
            "run.duration_hours=1.5",
            "physics.viscosity_m2_s=1.0",
        ],
    )
    .run(&mut |_| {})
    .unwrap_err()
    .to_string();
    assert!(err.contains("physics.viscosity_m2_s = 1.0"), "{err}");

    // The patch's own restart (at 1.5 h) resumes as any run's
    let own = dir.path().join("tidal_channel.restart");
    let resumed = patch(
        dir.path(),
        &[
            &format!("restart.from={:?}", own.display().to_string()),
            "run.duration_hours=2.0",
        ],
    )
    .run(&mut |_| {})
    .unwrap();
    assert!(
        resumed.notes[0].starts_with("resumed from ") && resumed.notes[0].contains("at 1.50 h"),
        "{:?}",
        resumed.notes
    );
    assert_eq!(*resumed.stations[0].time_s.last().unwrap(), 7200.0);
}

/// The largest |difference| of η and of the speed between `name`'s records
/// in `a` and `b` at their common times.
fn record_gap(a: &[StationSeries], b: &[StationSeries], name: &str) -> (f64, f64, usize) {
    let find = |s: &[StationSeries]| s.iter().find(|s| s.name == name).unwrap().clone();
    let (sa, sb) = (find(a), find(b));
    let (mut eta, mut speed, mut n) = (0.0_f64, 0.0_f64, 0);
    for (i, t) in sb.time_s.iter().enumerate() {
        let Some(j) = sa.time_s.iter().position(|x| x == t) else {
            continue;
        };
        eta = eta.max((sb.eta_m[i] - sa.eta_m[j]).abs());
        let du = sb.u_m_s[i] - sa.u_m_s[j];
        let dv = sb.v_m_s[i] - sa.v_m_s[j];
        speed = speed.max(du.hypot(dv));
        n += 1;
    }
    (eta, speed, n)
}

/// A patch of the tidal channel's middle nested in the whole run's own
/// snapshots (`mesh.patch.whole_run`) follows the whole run there: the
/// middle station's η and current against the whole run's, where the same
/// patch with the case's tide at its cut (the mouth's tide, the wrong one
/// there) is far off. From the start and from the whole run's restart.
#[test]
fn a_patch_nested_in_its_whole_run_replays_it() {
    let whole_dir = tempfile::tempdir().unwrap();
    let whole_file = example(
        whole_dir.path(),
        &[
            "run.duration_hours=3.0".into(),
            "output.interval_minutes=5.0".into(),
            "restart.interval_hours=1.0".into(),
            "restart.keep=true".into(),
        ],
    );
    let whole = whole_file.run(&mut |_| {}).unwrap();
    let k = element_near(&whole_file.build_domain().unwrap(), [20500.0, 1500.0]);
    let snapshot = whole_dir.path().join("tidal_channel.dgsnap");
    let in_run = format!("mesh.patch.whole_run={:?}", snapshot.display().to_string());

    let run = |extra: &[&str]| {
        let dir = tempfile::tempdir().unwrap();
        let mut o = patch_overrides(k, 6.0, "open");
        o.push("run.duration_hours=3.0".into());
        o.push("output.interval_minutes=5.0".into());
        o.extend(extra.iter().map(|s| s.to_string()));
        let file = example(dir.path(), &o);
        let domain = file.build_domain().unwrap();
        let report = file.run_on(&domain, &mut |_| {}).unwrap();
        (report, domain.notes, dir)
    };
    let (nested, notes, _d1) = run(&[&in_run]);
    assert!(
        notes
            .iter()
            .any(|n| n.starts_with("patch: nested in the whole run's snapshots")),
        "{notes:?}"
    );
    let (tide, _, _d2) = run(&[]);
    let (eta, speed, n) = record_gap(&whole.stations, &nested.stations, "middle");
    let (eta_tide, speed_tide, _) = record_gap(&whole.stations, &tide.stations, "middle");
    println!(
        "middle over 3 h ({n} records): nested in the run |Δη| ≤ {:.2} mm, |Δu| ≤ {:.2} mm/s; \
         the case's tide at the cut {:.1} mm, {:.1} mm/s",
        1e3 * eta,
        1e3 * speed,
        1e3 * eta_tide,
        1e3 * speed_tide
    );
    assert!(n >= 36, "{n} common records");
    // 0.66 mm and 0.58 mm/s against 427 mm and 285 mm/s (2026-11-28)
    assert!(eta < 2e-3 && speed < 2e-3, "{eta}, {speed}");
    assert!(
        eta_tide > 0.1 && speed_tide > 0.1,
        "{eta_tide}, {speed_tide}"
    );

    // From the whole run's restart at 1 h
    let restart = whole_dir.path().join("tidal_channel_001.00h.restart");
    let (seeded, _, _d3) = run(&[
        &in_run,
        &format!("restart.from={:?}", restart.display().to_string()),
    ]);
    let (eta_seeded, speed_seeded, n) = record_gap(&whole.stations, &seeded.stations, "middle");
    println!(
        "from the 1 h restart ({n} records): |Δη| ≤ {:.2} mm, |Δu| ≤ {:.2} mm/s",
        1e3 * eta_seeded,
        1e3 * speed_seeded
    );
    assert_eq!(seeded.stations[0].time_s[0], 3600.0);
    assert!(
        eta_seeded < 2e-3 && speed_seeded < 2e-3,
        "{eta_seeded}, {speed_seeded}"
    );
}

/// A 3D channel's M2 tide (as `case_test`'s resumed 3D run), its restarts
/// every half hour, in `dir`.
fn channel_3d(dir: &Path, minutes: f64, extra: &[String]) -> CaseFile {
    let path = dir.join("case.toml");
    std::fs::write(
        &path,
        format!(
            r#"
            [run]
            name = "channel"
            duration_hours = {:?}

            [mesh.rectangle]
            x_m = [0.0, 10000.0]
            y_m = [0.0, 2000.0]
            nx = 10
            ny = 2
            west = "open"

            [bathymetry]
            depth_m = 20.0

            [[boundary]]
            tag = "open"
            [boundary.tide]
            ramp_hours = 0.25
            constituents = [{{ name = "M2", amplitude_m = 0.5 }}]

            [three_d]
            levels = 6
            dt_max_s = 47.0
            profile = [[-6.0, 14.0, 32.0], [-14.0, 10.0, 34.0]]

            [output]
            directory = "out"
            interval_minutes = 10.0
            station_interval_minutes = 5.0

            [[output.station]]
            name = "head"
            x_m = 9500.0
            y_m = 1000.0

            [restart]
            interval_hours = 0.5
            "#,
            minutes / 60.0
        ),
    )
    .unwrap();
    CaseFile::read(&path, extra).unwrap()
}

/// A 3D channel's tide (as `case_test`'s resumed 3D run), its last 3 km
/// started from the whole run's restart at 30 minutes, walled.
#[test]
fn a_patch_starts_from_the_whole_3d_runs_restart() {
    let whole_dir = tempfile::tempdir().unwrap();
    let whole_file = channel_3d(whole_dir.path(), 30.0, &[]);
    let whole = whole_file.run(&mut |_| {}).unwrap();
    let restart = whole_dir.path().join("out").join("channel.restart");
    assert_eq!(dgcom::io::Restart3D::read(&restart).unwrap().time, 1800.0);
    let k = element_near(&whole_file.build_domain().unwrap(), [9500.0, 500.0]);

    let dir = tempfile::tempdir().unwrap();
    let mut extra = patch_overrides(k, 2.5, "wall");
    extra.push(format!("restart.from={:?}", restart.display().to_string()));
    extra.push("output.directory=\"patch\"".into());
    let file = channel_3d(dir.path(), 45.0, &extra);
    let domain = file.build_domain().unwrap();
    assert_eq!(
        domain.mesh.n_elements, 6,
        "x = 7.5, 8.5, 9.5 km in two rows"
    );
    let report = file.run_on(&domain, &mut |_| {}).unwrap();
    assert!(
        report
            .notes
            .iter()
            .any(|n| n.starts_with("patch started from the whole run's restart")),
        "{:?}",
        report.notes
    );
    let head = &report.stations[0];
    assert_eq!(head.time_s, vec![1800.0, 2100.0, 2400.0, 2700.0]);
    let (a, b) = (
        record_at(&report.stations, "head", 1800.0),
        record_at(&whole.stations, "head", 1800.0),
    );
    assert!(
        a.iter().zip(&b).all(|(a, b)| a.to_bits() == b.to_bits()),
        "{a:?} against {b:?}"
    );
    assert!(head.eta_m.iter().chain(&head.u_m_s).all(|v| v.is_finite()));
    let (v0, v1) = report.volume_m3;
    assert!((v1 - v0).abs() <= 1e-11 * v0, "{v0} → {v1}");
}

/// The 3D channel's last 3 km nested in the whole run's snapshots follows
/// it at the head (η, the depth mean), where the same patch walled at its
/// cut stops the tide.
#[test]
fn a_3d_patch_nested_in_its_whole_run_replays_it() {
    let whole_dir = tempfile::tempdir().unwrap();
    // Frames every minute: at 5 the tide's ramped front, which sweeps the
    // channel in 15 minutes, is 20 mm off at the head (0.9 mm at 1)
    let every = ["output.interval_minutes=1.0".to_string()];
    let whole_file = channel_3d(whole_dir.path(), 60.0, &every);
    let whole = whole_file.run(&mut |_| {}).unwrap();
    let k = element_near(&whole_file.build_domain().unwrap(), [9500.0, 500.0]);
    let snapshot = whole_dir.path().join("out").join("channel.dgsnap");
    let patch = |faces: &str, extra: &[String]| {
        let dir = tempfile::tempdir().unwrap();
        let mut o = patch_overrides(k, 2.5, faces);
        o.extend(every.iter().cloned());
        o.extend(extra.iter().cloned());
        let report = channel_3d(dir.path(), 60.0, &o).run(&mut |_| {}).unwrap();
        (report, dir)
    };
    let in_run = format!("mesh.patch.whole_run={:?}", snapshot.display().to_string());
    let (nested, _d1) = patch("open", &[in_run]);
    assert!(
        nested
            .notes
            .iter()
            .any(|n| n.starts_with("open faces: nested in the whole run's u, v, T, S")),
        "{:?}",
        nested.notes
    );
    let (walled, _d2) = patch("wall", &[]);
    let (eta, speed, n) = record_gap(&whole.stations, &nested.stations, "head");
    let (eta_wall, speed_wall, _) = record_gap(&whole.stations, &walled.stations, "head");
    println!(
        "3D head over 1 h ({n} records): nested in the run |Δη| ≤ {:.3} mm, |Δū| ≤ {:.3} mm/s; \
         walled {:.1} mm, {:.1} mm/s",
        1e3 * eta,
        1e3 * speed,
        1e3 * eta_wall,
        1e3 * speed_wall
    );
    assert!(n >= 12, "{n} records");
    // 0.86 mm and 0.10 mm/s against 492 mm and 20 mm/s (2026-11-28)
    assert!(eta < 2e-3 && speed < 1e-3, "{eta}, {speed}");
    assert!(
        eta_wall > 0.2 && speed_wall > 0.01,
        "{eta_wall}, {speed_wall}"
    );
}

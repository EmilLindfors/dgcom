//! A 3D run at point observations: the run's snapshot file sampled where and
//! when each observation was made (satellite SST pixels, farm temperatures,
//! FerryBox records, CTD casts).
//!
//! ```text
//! cargo run --release --example snapshot_sample_3d -- <run.dgsnap> <observations.txt> \
//!     [field=temp] [depth=0] [max_gap_hours=1] [out=<file>]
//! ```
//!
//! `observations.txt` holds one observation per line, `unix_time lon lat
//! value` (whitespace-separated, `#` comments). Each is matched with the
//! run's `field` (a layered field of the snapshot: `temp`, `salt`, `u`, `v`)
//! at `depth` metres below the surface: the top layer's value above its
//! centre, linear between layer centres below (as `norkyst_profiles_3d`),
//! within the element holding the point (its nodal basis) and linear in time
//! between frames. Observations up to `max_gap_hours` past the last frame
//! take the last frame; earlier ones before the first, points outside the
//! mesh and columns shallower than `depth` are left out.
//!
//! Written to `out` (default `<observations>.model.txt`): `unix_time lon lat
//! value model`. Printed: the count, the mean difference model − observation,
//! its RMS and standard deviation, and the correlation, over all and per day.

use std::collections::HashMap;
use std::error::Error;
use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

use dg_rs::io::{CoordinateProjection, LocalProjection, SnapshotFrame, SnapshotReader};
use dg_rs::mesh::PointLocator2D;
use dg_rs::operators::DGOperators2D;
use dg_rs::solver::Probe2D;

/// One observation and where it sits in the mesh.
struct Observation {
    unix: f64,
    lon: f64,
    lat: f64,
    value: f64,
    probe: Probe2D,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: snapshot_sample_3d <run.dgsnap> <observations.txt> \
                 [field=temp] [depth=0] [max_gap_hours=1] [out=<file>]";
    let positional: Vec<&String> = args.iter().filter(|a| !a.contains('=')).collect();
    let [snapshot_path, obs_path] =
        [positional.first(), positional.get(1)].map(|a| a.map(PathBuf::from).ok_or(usage));
    let (snapshot_path, obs_path) = (snapshot_path?, obs_path?);
    let options: HashMap<&str, &str> = args.iter().filter_map(|a| a.split_once('=')).collect();
    let field_name = options.get("field").copied().unwrap_or("temp");
    let depth: f64 = options.get("depth").map_or(Ok(0.0), |s| s.parse())?;
    let max_gap = 3600.0
        * options
            .get("max_gap_hours")
            .map_or(Ok(1.0), |s| s.parse())?;
    let out_path = options
        .get("out")
        .map_or_else(|| obs_path.with_extension("model.txt"), PathBuf::from);

    let mut snapshot = SnapshotReader::open(&snapshot_path)?;
    let header = snapshot.header().clone();
    let levels = header.levels.clone().ok_or("not a 3D snapshot")?;
    let field = levels
        .field(field_name)
        .ok_or(format!("no layered field {field_name}"))?;
    let clock = header.clock.ok_or("the snapshot has no clock")?;
    let projection = match header
        .metadata("projection")
        .and_then(|p| p.strip_prefix("local,"))
        .and_then(|p| p.split_once(','))
    {
        Some((lat, lon)) => LocalProjection::new(lat.parse()?, lon.parse()?),
        None => return Err("the snapshot has no local projection".into()),
    };
    let ops = DGOperators2D::new(header.order);
    let (nn, nl) = (ops.n_nodes, header.n_levels());
    // Layer centres' depths below the surface per unit column depth, from the
    // surface down
    let unit_depths: Vec<f64> = levels.sigma.sigma_rho().iter().rev().map(|s| -s).collect();
    let locator = PointLocator2D::new(&header.mesh);

    // The observations inside the mesh, in time order
    let text = fs::read_to_string(&obs_path)?;
    let (mut read, mut outside) = (0usize, 0usize);
    let mut observations = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let values: Vec<f64> = line
            .split_whitespace()
            .take(4)
            .map(str::parse)
            .collect::<Result<_, _>>()?;
        let [unix, lon, lat, value] = values[..] else {
            return Err(format!("{line:?}: unix_time lon lat value").into());
        };
        read += 1;
        let (x, y) = projection.geo_to_xy(lat, lon);
        match Probe2D::at(&locator, &ops, [x, y]) {
            Some(probe) => observations.push(Observation {
                unix,
                lon,
                lat,
                value,
                probe,
            }),
            None => outside += 1,
        }
    }
    observations.sort_by(|a, b| a.unix.total_cmp(&b.unix));

    let times = snapshot.times()?;
    let (first, last) = (
        clock.unix(*times.first().ok_or("no frames")?),
        clock.unix(*times.last().ok_or("no frames")?),
    );
    // The field at `depth` at one frame and observation
    let sample = |frame: &SnapshotFrame, obs: &Observation, element: &mut [f64]| -> Option<f64> {
        let k = obs.probe.element().as_usize();
        let nodes = k * nn..(k + 1) * nn;
        for (e, j) in element.iter_mut().zip(nodes.clone()) {
            *e = f64::from(frame.eta[j]) - header.bathymetry[j];
        }
        let column_depth = obs.probe.evaluate(element);
        if column_depth < depth.max(0.0) {
            return None;
        }
        let layer = |l: usize, element: &mut [f64]| {
            for (e, j) in element.iter_mut().zip(nodes.clone()) {
                *e = f64::from(frame.layers[field][j * nl + l]);
            }
            obs.probe.evaluate(element)
        };
        // From the surface down: layer index nl − 1 − i at unit_depths[i]
        let at = |i: usize| unit_depths[i] * column_depth;
        if depth <= at(0) {
            return Some(layer(nl - 1, element));
        }
        let i = (1..nl).find(|&i| at(i) >= depth).unwrap_or(nl - 1);
        let w = ((depth - at(i - 1)) / (at(i) - at(i - 1))).clamp(0.0, 1.0);
        let (above, below) = (layer(nl - i, element), layer(nl - 1 - i, element));
        Some((1.0 - w) * above + w * below)
    };

    let mut rows: Vec<(&Observation, f64)> = Vec::new();
    let (mut before, mut after) = (0usize, 0usize);
    let mut frames = [SnapshotFrame::default(), SnapshotFrame::default()];
    let mut loaded: [Option<usize>; 2] = [None, None];
    let mut element = vec![0.0; nn];
    for obs in &observations {
        if obs.unix < first {
            before += 1;
            continue;
        }
        if obs.unix > last + max_gap {
            after += 1;
            continue;
        }
        // The frame pair around the observation (the last frame alone past it)
        let unix = obs.unix.min(last);
        let i = times
            .partition_point(|&t| clock.unix(t) <= unix)
            .saturating_sub(1)
            .min(times.len().saturating_sub(2));
        let j = (i + 1).min(times.len() - 1);
        for (slot, index) in [i, j].into_iter().enumerate() {
            if loaded[slot] != Some(index) {
                snapshot.read_frame_into(index, &mut frames[slot])?;
                loaded[slot] = Some(index);
            }
        }
        let (ti, tj) = (clock.unix(times[i]), clock.unix(times[j]));
        let w = if tj > ti {
            ((unix - ti) / (tj - ti)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let (Some(a), Some(b)) = (
            sample(&frames[0], obs, &mut element),
            sample(&frames[1], obs, &mut element),
        ) else {
            continue;
        };
        rows.push((obs, (1.0 - w) * a + w * b));
    }

    let mut out = format!(
        "# {} at {depth} m below the surface from {}, at the observations of {}\n\
         # columns: unix_time lon lat value model\n",
        field_name,
        snapshot_path.display(),
        obs_path.display()
    );
    for (obs, model) in &rows {
        writeln!(
            out,
            "{:.0} {:.5} {:.5} {:.4} {:.4}",
            obs.unix, obs.lon, obs.lat, obs.value, model
        )?;
    }
    fs::write(&out_path, out)?;

    println!(
        "{read} observations: {outside} outside the mesh, {before} before the run, {after} more than \
         {:.1} h after it, {} matched ({field_name} at {depth} m)",
        max_gap / 3600.0,
        rows.len()
    );
    let summary = |label: &str, pairs: &[(f64, f64)]| {
        if pairs.is_empty() {
            return;
        }
        let n = pairs.len() as f64;
        let diff: Vec<f64> = pairs.iter().map(|(o, m)| m - o).collect();
        let mean = diff.iter().sum::<f64>() / n;
        let rms = (diff.iter().map(|d| d * d).sum::<f64>() / n).sqrt();
        let std = (rms * rms - mean * mean).max(0.0).sqrt();
        let (mo, mm) = (
            pairs.iter().map(|p| p.0).sum::<f64>() / n,
            pairs.iter().map(|p| p.1).sum::<f64>() / n,
        );
        let cov: f64 = pairs.iter().map(|(o, m)| (o - mo) * (m - mm)).sum();
        let var_o: f64 = pairs.iter().map(|(o, _)| (o - mo).powi(2)).sum();
        let var_m: f64 = pairs.iter().map(|(_, m)| (m - mm).powi(2)).sum();
        println!(
            "  {label:<12} n {:>6} | observed {mo:>8.3}, model {mm:>8.3} | model − observed {mean:+.3} \
             (RMS {rms:.3}, std {std:.3}), r {:.3}",
            pairs.len(),
            cov / (var_o * var_m).sqrt()
        );
    };
    let all: Vec<(f64, f64)> = rows.iter().map(|(o, m)| (o.value, *m)).collect();
    summary("all", &all);
    let mut days: Vec<i64> = rows
        .iter()
        .map(|(o, _)| (o.unix / 86400.0).floor() as i64)
        .collect();
    days.dedup();
    for day in days {
        let pairs: Vec<(f64, f64)> = rows
            .iter()
            .filter(|(o, _)| (o.unix / 86400.0).floor() as i64 == day)
            .map(|(o, m)| (o.value, *m))
            .collect();
        let hours = (86400.0 * day as f64 - first) / 3600.0;
        summary(&format!("day {hours:+.0} h"), &pairs);
    }
    println!("  → {}", out_path.display());
    Ok(())
}

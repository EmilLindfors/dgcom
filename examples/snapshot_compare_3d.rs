//! Compare two 3D snapshot files frame by frame: how far a run that changed
//! the numerics' round-off (or more) is from a reference run.
//!
//! ```text
//! cargo run --release --example snapshot_compare_3d -- <reference.dgsnap> <run.dgsnap> [min_depth=10]
//! ```
//!
//! For every time both files hold, over the nodes at least `min_depth` (m)
//! deep in the reference: η, the depth-mean velocity, the surface layer's
//! velocity and T, as the RMS and the largest difference (with where the
//! largest is). Both runs must be on the same mesh and levels.

use dgcom::io::SnapshotReader;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [reference, run] = [args.first(), args.get(1)]
        .map(|a| a.ok_or("usage: snapshot_compare_3d <reference> <run> [min_depth=10]"));
    let min_depth = match args.get(2) {
        Some(a) => a.trim_start_matches("min_depth=").parse()?,
        None => 10.0,
    };
    let mut a = SnapshotReader::open(reference?)?;
    let mut b = SnapshotReader::open(run?)?;
    let (n, nl) = (a.header().n_points(), a.header().n_levels());
    if (b.header().n_points(), b.header().n_levels()) != (n, nl) {
        return Err("the files have different meshes or levels".into());
    }
    let levels = a.header().levels.clone().ok_or("not a 3D snapshot")?;
    let field = |name: &str| levels.field(name).ok_or(format!("no layered field {name}"));
    let (fu, fv, ft) = (field("u")?, field("v")?, field("temp")?);
    let bed = a.header().bathymetry.clone();
    let times_b = b.times()?;
    println!(
        "nodes ≥ {min_depth} m deep; RMS / max (node) of run − reference\n\
         {:>8} | {:>7} | {:>19} | {:>19} | {:>19} | {:>19}",
        "t (h)", "nodes", "η (mm)", "ū (mm/s)", "surface u (mm/s)", "T (m°C)"
    );
    for i in 0..a.n_frames()? {
        let t = a.time(i)?;
        let Some(j) = times_b.iter().position(|&tb| (tb - t).abs() < 1e-3) else {
            continue;
        };
        let (fa, fb) = (a.read_frame(i)?, b.read_frame(j)?);
        let deep: Vec<usize> = (0..n)
            .filter(|&p| f64::from(fa.eta[p]) - bed[p] >= min_depth)
            .collect();
        // RMS and largest |difference| (with its node) over `samples`
        let stats = |samples: &mut dyn Iterator<Item = (usize, f64)>| {
            let (mut sum, mut count, mut max, mut at) = (0.0, 0usize, 0.0_f64, 0);
            for (p, d) in samples {
                sum += d * d;
                count += 1;
                if d.abs() > max {
                    (max, at) = (d.abs(), p);
                }
            }
            format!(
                "{:>8.3} {:>7.2} ({at:>6})",
                (sum / count.max(1) as f64).sqrt(),
                max
            )
        };
        let diff = |x: &[f32], y: &[f32], k: usize| f64::from(y[k]) - f64::from(x[k]);
        let top = |p: usize| p * nl + nl - 1;
        println!(
            "{:>8.3} | {:>7} | {} | {} | {} | {}",
            t / 3600.0,
            deep.len(),
            stats(&mut deep.iter().map(|&p| (p, 1e3 * diff(&fa.eta, &fb.eta, p)))),
            stats(&mut deep.iter().map(|&p| {
                let d = (diff(&fa.u, &fb.u, p)).hypot(diff(&fa.v, &fb.v, p));
                (p, 1e3 * d)
            })),
            stats(&mut deep.iter().map(|&p| {
                let (ua, va) = (&fa.layers[fu], &fa.layers[fv]);
                let (ub, vb) = (&fb.layers[fu], &fb.layers[fv]);
                let d = (diff(ua, ub, top(p))).hypot(diff(va, vb, top(p)));
                (p, 1e3 * d)
            })),
            stats(&mut deep.iter().flat_map(|&p| {
                let (ta, tb) = (&fa.layers[ft], &fb.layers[ft]);
                (0..nl).map(move |l| (p, 1e3 * diff(ta, tb, p * nl + l)))
            })),
        );
    }
    Ok(())
}

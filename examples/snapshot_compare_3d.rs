//! Compare two snapshot files frame by frame: how far a run that changed
//! the numerics' round-off (or more) is from a reference run, or whether two
//! runs that should agree bit for bit do (a case file against the example it
//! replaces, TODO P8.2).
//!
//! ```text
//! cargo run --release --example snapshot_compare_3d -- <reference.dgsnap> <run.dgsnap> \
//!     [min_depth=10] [points=name:lon,lat;name:xy:x_km,y_km;… | points=@<file>] [from_hours=0]
//! ```
//!
//! For every time both files hold, over the nodes at least `min_depth` (m)
//! deep in the reference: η, the depth-mean velocity and, for 3D files, the
//! surface layer's velocity and T, as the RMS and the largest difference
//! (with where the largest is), and how many values differ at all, the
//! surface fields (a coupled run's sea state) included. Both runs must be on
//! the same mesh and levels.
//!
//! `points=` adds the same differences at named points (farm sites, river
//! mouths), interpolated within the element holding each (its nodal basis),
//! over the common frames after `from_hours` (by default every frame but
//! the initial state, which runs from the same start share) where the
//! reference's column there is at least 1 m deep: η, the depth-mean
//! velocity beside the reference's RMS speed,
//! the surface layer's velocity, every level's velocity and T. A point is
//! `name:lon,lat` (the snapshot's local projection) or `name:xy:x_km,y_km`
//! (mesh axes); `@<file>` reads one point per line (`#` starts a comment).

use dgcom::io::{CoordinateProjection, LocalProjection, SnapshotHeader, SnapshotReader};
use dgcom::mesh::PointLocator2D;
use dgcom::operators::DGOperators2D;
use dgcom::solver::Probe2D;

/// A point's column shallower than this (m) in the reference is left out of
/// its frame
const POINT_MIN_DEPTH: f64 = 1.0;

/// RMS and largest |difference| of a running sample
#[derive(Default)]
struct Stat {
    squares: f64,
    count: usize,
    max: f64,
}

impl Stat {
    fn add(&mut self, d: f64) {
        self.squares += d * d;
        self.count += 1;
        self.max = self.max.max(d.abs());
    }

    fn rms(&self) -> f64 {
        (self.squares / self.count.max(1) as f64).sqrt()
    }

    fn show(&self) -> String {
        if self.count == 0 {
            return format!("{:>16}", "—");
        }
        format!("{:>7.2} {:>8.2}", self.rms(), self.max)
    }
}

/// A named point of `points=` and its differences over the frames.
struct Point {
    name: String,
    probe: Probe2D,
    /// The reference's depth there, summed over the frames
    depth: f64,
    /// The reference's depth-mean speed (mm/s)
    speed: Stat,
    eta: Stat,
    mean: Stat,
    surface: Stat,
    layers: Stat,
    temperature: Stat,
}

/// The points of a `points=` spec (`@<file>` reads one per line).
fn parse_points(
    spec: &str,
    header: &SnapshotHeader,
) -> Result<Vec<Point>, Box<dyn std::error::Error>> {
    let spec = match spec.strip_prefix('@') {
        Some(file) => std::fs::read_to_string(file)?
            .lines()
            .map(|l| l.split('#').next().unwrap_or("").trim())
            .filter(|l| !l.is_empty())
            .collect::<Vec<_>>()
            .join(";"),
        None => spec.to_string(),
    };
    let projection = match header
        .metadata("projection")
        .and_then(|p| p.strip_prefix("local,"))
        .and_then(|p| p.split_once(','))
    {
        Some((lat, lon)) => Some(LocalProjection::new(lat.parse()?, lon.parse()?)),
        None => None,
    };
    let ops = DGOperators2D::new(header.order);
    let locator = PointLocator2D::new(&header.mesh);
    let mut points = Vec::new();
    for item in spec.split(';').map(str::trim).filter(|s| !s.is_empty()) {
        let (name, place) = item
            .split_once(':')
            .ok_or(format!("point {item:?}: name:lon,lat or name:xy:x_km,y_km"))?;
        let (xy, place) = match place.strip_prefix("xy:") {
            Some(rest) => (true, rest),
            None => (false, place),
        };
        let (a, b) = place
            .split_once(',')
            .ok_or(format!("point {item:?}: two coordinates"))?;
        let (a, b): (f64, f64) = (a.trim().parse()?, b.trim().parse()?);
        let (x, y) = match (xy, &projection) {
            (true, _) => (1e3 * a, 1e3 * b),
            (false, Some(projection)) => projection.geo_to_xy(b, a),
            (false, None) => {
                return Err(format!("point {item:?}: the snapshot has no local projection").into());
            }
        };
        let name = name.trim().to_string();
        match Probe2D::at(&locator, &ops, [x, y]) {
            Some(probe) => points.push(Point {
                name,
                probe,
                depth: 0.0,
                speed: Stat::default(),
                eta: Stat::default(),
                mean: Stat::default(),
                surface: Stat::default(),
                layers: Stat::default(),
                temperature: Stat::default(),
            }),
            None => println!("{name}: outside the mesh, left out"),
        }
    }
    Ok(points)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage =
        "usage: snapshot_compare_3d <reference> <run> [min_depth=10] [points=…] [from_hours=0]";
    let [reference, run] = [args.first(), args.get(1)].map(|a| a.ok_or(usage));
    let mut min_depth = 10.0;
    let mut points_spec = None;
    let mut from_hours = 0.0;
    for a in args.iter().skip(2) {
        if let Some(spec) = a.strip_prefix("points=") {
            points_spec = Some(spec);
        } else if let Some(hours) = a.strip_prefix("from_hours=") {
            from_hours = hours.parse()?;
        } else {
            min_depth = a.trim_start_matches("min_depth=").parse()?;
        }
    }
    let mut a = SnapshotReader::open(reference?)?;
    let mut b = SnapshotReader::open(run?)?;
    let (n, nl) = (a.header().n_points(), a.header().n_levels());
    if (b.header().n_points(), b.header().n_levels()) != (n, nl) {
        return Err("the files have different meshes or levels".into());
    }
    let mut points = match points_spec {
        Some(spec) => parse_points(spec, a.header())?,
        None => Vec::new(),
    };
    let nn = n / a.header().mesh.n_elements;
    // The layered fields of a 3D file
    let layered = match a.header().levels.clone() {
        Some(levels) => {
            let field = |name: &str| levels.field(name).ok_or(format!("no layered field {name}"));
            Some((field("u")?, field("v")?, field("temp")?))
        }
        None => None,
    };
    let bed = a.header().bathymetry.clone();
    let times_b = b.times()?;
    println!(
        "nodes ≥ {min_depth} m deep; RMS / max (node) of run − reference\n\
         {:>8} | {:>7} | {:>19} | {:>19} | {:>19} | {:>19} | {:>9}",
        "t (h)", "nodes", "η (mm)", "ū (mm/s)", "surface u (mm/s)", "T (m°C)", "differ"
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
        let (surface, temperature) = match layered {
            Some((fu, fv, ft)) => (
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
            ),
            None => ("—".into(), "—".into()),
        };
        // Values that differ in any bit, over every node and field (the
        // surface fields too: a coupled run's sea state)
        let differ = [(&fa.eta, &fb.eta), (&fa.u, &fb.u), (&fa.v, &fb.v)]
            .into_iter()
            .chain(fa.layers.iter().zip(&fb.layers))
            .chain(fa.surface.iter().zip(&fb.surface))
            .map(|(x, y)| {
                x.iter()
                    .zip(y)
                    .filter(|(p, q)| p.to_bits() != q.to_bits())
                    .count()
            })
            .sum::<usize>();
        println!(
            "{:>8.3} | {:>7} | {} | {} | {:>19} | {:>19} | {:>9}",
            t / 3600.0,
            deep.len(),
            stats(&mut deep.iter().map(|&p| (p, 1e3 * diff(&fa.eta, &fb.eta, p)))),
            stats(&mut deep.iter().map(|&p| {
                let d = (diff(&fa.u, &fb.u, p)).hypot(diff(&fa.v, &fb.v, p));
                (p, 1e3 * d)
            })),
            surface,
            temperature,
            differ,
        );

        // The points: a nodal field interpolated at the probe (`stride`
        // values per node, the `offset`-th of them)
        if t <= 3600.0 * from_hours + 1e-3 {
            continue;
        }
        for point in &mut points {
            let first = point.probe.element().as_usize() * nn;
            let at = |values: &[f32], stride: usize, offset: usize| -> f64 {
                let weights = point.probe.weights();
                weights
                    .iter()
                    .enumerate()
                    .map(|(i, w)| w * f64::from(values[(first + i) * stride + offset]))
                    .sum()
            };
            let point_bed: f64 = point
                .probe
                .weights()
                .iter()
                .enumerate()
                .map(|(i, w)| w * bed[first + i])
                .sum();
            let depth = at(&fa.eta, 1, 0) - point_bed;
            if depth < POINT_MIN_DEPTH {
                continue;
            }
            point.depth += depth;
            point
                .speed
                .add(1e3 * at(&fa.u, 1, 0).hypot(at(&fa.v, 1, 0)));
            point.eta.add(1e3 * (at(&fb.eta, 1, 0) - at(&fa.eta, 1, 0)));
            let du = at(&fb.u, 1, 0) - at(&fa.u, 1, 0);
            let dv = at(&fb.v, 1, 0) - at(&fa.v, 1, 0);
            point.mean.add(1e3 * du.hypot(dv));
            if let Some((fu, fv, ft)) = layered {
                for l in 0..nl {
                    let du = at(&fb.layers[fu], nl, l) - at(&fa.layers[fu], nl, l);
                    let dv = at(&fb.layers[fv], nl, l) - at(&fa.layers[fv], nl, l);
                    point.layers.add(1e3 * du.hypot(dv));
                    if l == nl - 1 {
                        point.surface.add(1e3 * du.hypot(dv));
                    }
                    let dt = at(&fb.layers[ft], nl, l) - at(&fa.layers[ft], nl, l);
                    point.temperature.add(1e3 * dt);
                }
            }
        }
    }

    if points.is_empty() {
        return Ok(());
    }
    println!(
        "\npoints, over the common frames after {from_hours} h where the reference is \
         ≥ {POINT_MIN_DEPTH} m deep: RMS and largest |run − reference|; `speed` is the \
         reference's depth-mean speed\n\
         {:<20} | {:>9} | {:>6} | {:>16} | {:>16} | {:>16} | {:>16} | {:>16} | {:>16}",
        "point",
        "depth (m)",
        "frames",
        "speed (mm/s)",
        "η (mm)",
        "ū (mm/s)",
        "surface u (mm/s)",
        "levels' u (mm/s)",
        "T (m°C)"
    );
    for p in &points {
        println!(
            "{:<20} | {:>9.1} | {:>6} | {} | {} | {} | {} | {} | {}",
            p.name,
            p.depth / p.speed.count.max(1) as f64,
            p.speed.count,
            p.speed.show(),
            p.eta.show(),
            p.mean.show(),
            p.surface.show(),
            p.layers.show(),
            p.temperature.show(),
        );
    }
    // The depth-mean difference against the reference's speed, point by point
    let mut ratios: Vec<(f64, &str)> = points
        .iter()
        .filter(|p| p.speed.count > 0 && p.speed.rms() > 0.0)
        .map(|p| (p.mean.rms() / p.speed.rms(), p.name.as_str()))
        .collect();
    ratios.sort_by(|x, y| x.0.total_cmp(&y.0));
    if let (Some(median), Some(largest)) = (ratios.get(ratios.len() / 2), ratios.last()) {
        println!(
            "ū's RMS difference over the reference's RMS speed, {} points: median {:.1} %, \
             largest {:.1} % ({})",
            ratios.len(),
            100.0 * median.0,
            100.0 * largest.0,
            largest.1
        );
    }
    Ok(())
}

//! Does a patch see a change as the whole run sees it? A change (a code
//! change, a switch, a parameter) run on the whole domain and on a patch of
//! it (`[mesh.patch]` nested in the unchanged whole run, `whole_run =`):
//! its effect at named points in each, and how far the patch's is from the
//! whole run's.
//!
//! ```text
//! cargo run --release --example patch_effect_3d -- \
//!     <whole_a.dgsnap> <whole_b.dgsnap> <patch_a.dgsnap> <patch_b.dgsnap> \
//!     points=name:lon,lat;… | points=@<file>  [from_hours=0]
//! ```
//!
//! `a` is the reference (the code or case as it is), `b` the change; both
//! patches nest in `whole_a`, the only whole run a change on a patch needs.
//! For every frame all four files hold after `from_hours` and every point in
//! both meshes whose reference column is at least 1 m deep: the whole run's
//! effect `B − A` and the patch's `b − a` of η, the depth-mean velocity, the
//! surface layer's velocity and T over the levels, as RMS over the frames,
//! and the RMS of `(b − a) − (B − A)`: what the patch gets wrong of the
//! change. The patch's own error against the whole run (`a − A`) is the
//! floor under which no effect shows; `snapshot_compare_3d` prints it.

use dgcom::io::{CoordinateProjection, LocalProjection, SnapshotFrame, SnapshotReader};
use dgcom::mesh::PointLocator2D;
use dgcom::operators::DGOperators2D;
use dgcom::solver::Probe2D;

/// A point's column shallower than this (m) in the reference is left out of
/// its frame
const POINT_MIN_DEPTH: f64 = 1.0;

/// Running sums of squares: the whole run's effect, the patch's, their
/// difference.
#[derive(Default, Clone, Copy)]
struct Effect {
    whole: f64,
    patch: f64,
    error: f64,
    count: usize,
}

impl Effect {
    fn add(&mut self, whole: f64, patch: f64) {
        self.whole += whole * whole;
        self.patch += patch * patch;
        self.error += (patch - whole) * (patch - whole);
        self.count += 1;
    }

    fn show(&self) -> String {
        if self.count == 0 {
            return format!("{:>26}", "—");
        }
        let rms = |s: f64| (s / self.count as f64).sqrt();
        format!(
            "{:>8.3} {:>8.3} {:>8.3}",
            rms(self.whole),
            rms(self.patch),
            rms(self.error)
        )
    }
}

struct Point {
    name: String,
    /// On the whole mesh and on the patch's
    whole: Probe2D,
    patch: Probe2D,
    eta: Effect,
    mean: [Effect; 2],
    surface: [Effect; 2],
    temperature: Effect,
}

/// A nodal field interpolated at `probe` (`stride` values per node, the
/// `offset`-th of them).
fn at(values: &[f32], probe: &Probe2D, nn: usize, stride: usize, offset: usize) -> f64 {
    let first = probe.element().as_usize() * nn;
    probe
        .weights()
        .iter()
        .enumerate()
        .map(|(i, w)| w * f64::from(values[(first + i) * stride + offset]))
        .sum()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: patch_effect_3d <whole_a> <whole_b> <patch_a> <patch_b> points=… \
                 [from_hours=0]";
    if args.len() < 5 {
        return Err(usage.into());
    }
    let mut spec = None;
    let mut from_hours = 0.0;
    for a in &args[4..] {
        if let Some(s) = a.strip_prefix("points=") {
            spec = Some(s.to_string());
        } else if let Some(h) = a.strip_prefix("from_hours=") {
            from_hours = h.parse()?;
        } else {
            return Err(format!("unknown argument {a}\n{usage}").into());
        }
    }
    let spec = spec.ok_or(usage)?;
    let mut files = args[..4]
        .iter()
        .map(SnapshotReader::open)
        .collect::<Result<Vec<_>, _>>()?;
    let (nl, order) = (files[0].header().n_levels(), files[0].header().order);
    if files.iter().any(|f| f.header().n_levels() != nl) {
        return Err("the files have different levels".into());
    }
    let nn = (order + 1) * (order + 1);
    let fields = match files[0].header().levels.clone() {
        Some(levels) => {
            let field = |name: &str| levels.field(name).ok_or(format!("no layered field {name}"));
            Some((field("u")?, field("v")?, field("temp")?))
        }
        None => None,
    };
    let projection = files[0]
        .header()
        .metadata("projection")
        .and_then(|p| p.strip_prefix("local,"))
        .and_then(|p| p.split_once(','))
        .map(|(lat, lon)| -> Result<_, std::num::ParseFloatError> {
            Ok(LocalProjection::new(lat.parse()?, lon.parse()?))
        })
        .transpose()?;

    // The points, on the whole mesh and on the patch's
    let spec = match spec.strip_prefix('@') {
        Some(file) => std::fs::read_to_string(file)?
            .lines()
            .map(|l| l.split('#').next().unwrap_or("").trim().to_string())
            .filter(|l| !l.is_empty())
            .collect::<Vec<_>>()
            .join(";"),
        None => spec,
    };
    let ops = DGOperators2D::new(order);
    let whole_mesh = files[0].header().mesh.clone();
    let patch_mesh = files[2].header().mesh.clone();
    let (whole_at, patch_at) = (
        PointLocator2D::new(&whole_mesh),
        PointLocator2D::new(&patch_mesh),
    );
    let mut points = Vec::new();
    for item in spec.split(';').map(str::trim).filter(|s| !s.is_empty()) {
        let (name, place) = item
            .split_once(':')
            .ok_or(format!("point {item:?}: name:lon,lat"))?;
        let (a, b) = place
            .split_once(',')
            .ok_or(format!("point {item:?}: lon,lat"))?;
        let (lon, lat): (f64, f64) = (a.trim().parse()?, b.trim().parse()?);
        let (x, y) = projection
            .as_ref()
            .ok_or("the snapshot has no local projection")?
            .geo_to_xy(lat, lon);
        if let (Some(whole), Some(patch)) = (
            Probe2D::at(&whole_at, &ops, [x, y]),
            Probe2D::at(&patch_at, &ops, [x, y]),
        ) {
            points.push(Point {
                name: name.trim().to_string(),
                whole,
                patch,
                eta: Effect::default(),
                mean: [Effect::default(); 2],
                surface: [Effect::default(); 2],
                temperature: Effect::default(),
            });
        }
    }
    if points.is_empty() {
        return Err("no point lies in the patch".into());
    }
    let bed = files[0].header().bathymetry.clone();

    // The frames all four hold
    let times: Vec<Vec<f64>> = files
        .iter_mut()
        .map(|f| f.times())
        .collect::<Result<_, _>>()?;
    let mut frames = 0;
    for &t in &times[0] {
        if t <= 3600.0 * from_hours + 1e-3 {
            continue;
        }
        let index: Option<Vec<usize>> = times
            .iter()
            .map(|ts| ts.iter().position(|&x| (x - t).abs() < 1e-3))
            .collect();
        let Some(index) = index else { continue };
        let f: Vec<SnapshotFrame> = files
            .iter_mut()
            .zip(&index)
            .map(|(file, &j)| file.read_frame(j))
            .collect::<Result<_, _>>()?;
        frames += 1;
        for p in &mut points {
            let first = p.whole.element().as_usize() * nn;
            let point_bed: f64 = p
                .whole
                .weights()
                .iter()
                .enumerate()
                .map(|(i, w)| w * bed[first + i])
                .sum();
            if at(&f[0].eta, &p.whole, nn, 1, 0) - point_bed < POINT_MIN_DEPTH {
                continue;
            }
            // b − a of a field on each pair (whole: 0, 1; patch: 2, 3)
            let effect = |pick: &dyn Fn(&SnapshotFrame) -> &[f32], stride: usize, offset: usize| {
                let w = at(pick(&f[1]), &p.whole, nn, stride, offset)
                    - at(pick(&f[0]), &p.whole, nn, stride, offset);
                let q = at(pick(&f[3]), &p.patch, nn, stride, offset)
                    - at(pick(&f[2]), &p.patch, nn, stride, offset);
                (1e3 * w, 1e3 * q)
            };
            let (w, q) = effect(&|s| &s.eta, 1, 0);
            p.eta.add(w, q);
            let (w, q) = effect(&|s| &s.u, 1, 0);
            p.mean[0].add(w, q);
            let (w, q) = effect(&|s| &s.v, 1, 0);
            p.mean[1].add(w, q);
            if let Some((fu, fv, ft)) = fields {
                for (c, field) in [fu, fv].into_iter().enumerate() {
                    let (w, q) = effect(&|s| &s.layers[field], nl, nl - 1);
                    p.surface[c].add(w, q);
                }
                for l in 0..nl {
                    let (w, q) = effect(&|s| &s.layers[ft], nl, l);
                    p.temperature.add(w, q);
                }
            }
        }
    }
    println!(
        "{frames} common frames after {from_hours} h; RMS over them of the whole run's effect \
         (B − A), the patch's (b − a) and their difference\n{:<20} | {:>26} | {:>26} | {:>26} | \
         {:>26}\n{:<20} | {:>26} | {:>26} | {:>26} | {:>26}",
        "point",
        "η (mm)",
        "ū (mm/s, |·| of u and v)",
        "surface u (mm/s)",
        "T (m°C)",
        "",
        "whole    patch    error",
        "whole    patch    error",
        "whole    patch    error",
        "whole    patch    error",
    );
    // The velocity's two components as one vector's RMS
    let vector = |e: &[Effect; 2]| Effect {
        whole: e[0].whole + e[1].whole,
        patch: e[0].patch + e[1].patch,
        error: e[0].error + e[1].error,
        count: e[0].count,
    };
    for p in &points {
        println!(
            "{:<20} | {} | {} | {} | {}",
            p.name,
            p.eta.show(),
            vector(&p.mean).show(),
            vector(&p.surface).show(),
            p.temperature.show()
        );
    }
    Ok(())
}

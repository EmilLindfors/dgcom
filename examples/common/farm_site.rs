//! A real fish farm in `froya_real_data`'s 2D tide (TODO F.1): the cages of
//! a Fiskeridirektoratet site file (`site=`, `io::farm_site`), their wake
//! and their effect on the residual current, the drag on the farm, and the
//! pen-to-pen connectivity of particles released in the cages, with and
//! without the random walk's depth drift (TODO F.2).
//!
//! The site's cages ([`FarmSite::cage_layout`], [`CageGrid::RING_160M`]
//! unless `cage_spacing= cage_radius= net_depth= solidity=` say otherwise)
//! drag on the depth mean (`CageDrag2D`). Unless `farm_runs=cages`, the
//! same tide runs first without them (into `<output>/no_cages`), and every
//! callback samples the depth-mean current of both runs on a grid of
//! `farm_grid` m (default 20) over `farm_window` m (default 1500) around the
//! farm. Over the analysis window (the last whole number of M2 periods after
//! `spinup_hours`) the report gives
//! - the current at the farm: speed, principal axis, residual;
//! - the wake: the relative speed deficit `1 − |u_cages|/|u_open|` along the
//!   current through the farm's centre, up- and down-current, for each
//!   direction of the tide, and the area where it exceeds 5, 10 and 20 %;
//! - the residual current (the window's mean) with and without the cages;
//! - the drag on the farm, `F = ρ₀ ∫ Λ h u dA`.
//!
//! `<output>/farm_wake.csv` has the grid's fields for maps, and
//! `farm_cages.csv` the cages (`scripts/farm_wake_map.py` draws them).
//!
//! With `particles=N`, the run with cages releases N particles in each cage
//! (a sunflower pattern over its footprint) at `release` hours (default
//! `spinup_hours`) and every `release_every` hours (default a quarter M2
//! period) for `releases` batches (default 4), at the first callback at or
//! after each time, and tracks them online between the callbacks' states in
//! steps of at most `particle_seconds` (default 20), with a random walk of
//! each `kh` (m²/s, default 0.1; a list tracks each) in `seeds` seeds
//! (default 4) and for each walk tracer of `walks` (default
//! `depth,surface`): `depth` is a tracer mixed through the water column
//! (`WalkTracer::DepthIntegrated`, the drift `K∇h/h`), `surface` one per
//! unit area (`WalkTracer::Surface`, no depth drift). All walks see the same
//! flow and the same seeds, so their difference is the drift's. Particles
//! strand in water under 5 cm and leave through the open boundary.
//!
//! Connectivity (`particles::connectivity`): each cage's footprint is a
//! contact zone, and the farm as a whole (a disc round the cages) another,
//! which a particle reaches again only after it has been `farm_return` m
//! (default 500) beyond it. Per walk, over all batches and per batch: the
//! share returning to their own cage (after a cage radius beyond it), the
//! share reaching another cage, the exposure to the other cages per
//! particle, the share back at the farm, the median age at first contact
//! with another cage; the particles' fates and the depth they are in at the
//! end. Means over the seeds with their standard errors, and the difference
//! depth − surface in standard errors (`z`; the walks share their seeds, so
//! the independent errors overstate the difference's). The cage-by-cage
//! matrix of the first walk, and `<output>/connectivity.csv` with every
//! seed's matrix per batch.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write as _};
use std::path::{Path, PathBuf};
use std::time::Instant;

use dg_rs::io::{CageGrid, CoordinateProjection, FarmSite, read_farm_site_file};
use dg_rs::mesh::data::Bathymetry2D;
use dg_rs::mesh::{Mesh2D, PointLocator2D};
use dg_rs::operators::{DGOperators2D, GeometricFactors2D};
use dg_rs::particles::{
    ConnectivityMatrix, ConnectivityRecorder, ContactZone, Particle2D, ParticleStatus,
    ParticleTracker2D, SWEVelocity2D, WalkTracer,
};
use dg_rs::physics::cage_drag::for_each_caged_node;
use dg_rs::solver::{Probe2D, SWESolution2D, WetDryConfig};
use dg_rs::source::{CageDrag2D, CageFootprint, NetCage};
use dg_rs::time::ModelClock;
use dg_rs::types::ElementIndex;

const RHO0: f64 = 1025.0;
/// Below this speed (m/s) at the farm's centre a sample is slack water and
/// does not enter the wake's statistics, nor does a point slower than
/// `POINT_MIN_SPEED` its deficit
const MIN_SPEED: f64 = 0.02;
const POINT_MIN_SPEED: f64 = 0.01;
/// Distances (m) from the farm's centre along the current where the wake is
/// reported (negative: up-current)
const RAY: [f64; 15] = [
    -400.0, -200.0, -100.0, 0.0, 50.0, 100.0, 150.0, 200.0, 300.0, 400.0, 600.0, 800.0, 1000.0,
    1200.0, 1400.0,
];
/// Particles strand in water shallower than this (m)
const STRANDING_DEPTH: f64 = 0.05;
const M2_PERIOD: f64 = 12.420_601 * 3600.0;

/// What `site=` and its companions ask for (see the module docs).
#[derive(Clone, Debug)]
pub struct FarmOptions {
    site: PathBuf,
    /// Run the tide without the cages first
    reference: bool,
    grid: CageGrid,
    /// Half-width (m) and spacing (m) of the sampled window
    window: f64,
    spacing: f64,
    farm_return: f64,
    particles: usize,
    /// Release of the first batch (s), the interval between batches (s) and
    /// their number
    release: f64,
    release_every: f64,
    releases: usize,
    particle_seconds: f64,
    kh: Vec<f64>,
    seeds: usize,
    tracers: Vec<WalkTracer>,
}

impl FarmOptions {
    /// The farm options of the command line, if it has `site=`; the first
    /// batch of particles is released at `spinup_hours` unless `release=`
    /// says otherwise.
    pub fn parse(
        args: &HashMap<String, String>,
        spinup_hours: f64,
    ) -> Result<Option<Self>, String> {
        let Some(site) = args.get("site") else {
            return Ok(None);
        };
        let get = |key: &str, default: f64| -> Result<f64, String> {
            args.get(key).map_or(Ok(default), |v| {
                v.parse().map_err(|_| format!("bad {key}={v}"))
            })
        };
        let ring = CageGrid::RING_160M;
        let grid = CageGrid::new(
            get("cage_spacing", ring.spacing)?,
            get("cage_radius", ring.radius)?,
            get("net_depth", ring.net_depth)?,
        )
        .with_solidity(get("solidity", ring.solidity)?);
        let list = |key: &str, default: &str| -> Vec<String> {
            args.get(key)
                .map_or(default, String::as_str)
                .split(',')
                .filter(|s| !s.is_empty())
                .map(String::from)
                .collect()
        };
        let kh = list("kh", "0.1")
            .iter()
            .map(|v| v.parse().map_err(|_| format!("bad kh={v}")))
            .collect::<Result<Vec<f64>, _>>()?;
        let tracers = list("walks", "depth,surface")
            .iter()
            .map(|w| match w.as_str() {
                "depth" => Ok(WalkTracer::DepthIntegrated),
                "surface" => Ok(WalkTracer::Surface),
                other => Err(format!("bad walks={other}: depth and/or surface")),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let options = Self {
            site: PathBuf::from(site),
            reference: match args.get("farm_runs").map_or("both", String::as_str) {
                "both" => true,
                "cages" => false,
                other => return Err(format!("bad farm_runs={other}: both or cages")),
            },
            grid,
            window: get("farm_window", 1500.0)?,
            spacing: get("farm_grid", 20.0)?,
            farm_return: get("farm_return", 500.0)?,
            particles: get("particles", 0.0)? as usize,
            release: 3600.0 * get("release", spinup_hours)?,
            release_every: 3600.0 * get("release_every", M2_PERIOD / 4.0 / 3600.0)?,
            releases: get("releases", 4.0)? as usize,
            particle_seconds: get("particle_seconds", 20.0)?,
            kh,
            seeds: (get("seeds", 4.0)? as usize).max(1),
            tracers,
        };
        if options.kh.is_empty() || options.tracers.is_empty() {
            return Err("kh= and walks= need at least one value".into());
        }
        if !(options.spacing > 0.0 && options.window > options.spacing) {
            return Err("farm_grid= must be positive and below farm_window=".into());
        }
        Ok(Some(options))
    }

    /// Whether the tide runs without the cages first.
    pub fn reference(&self) -> bool {
        self.reference
    }
}

/// A farm site on a mesh: its cages, and the probes the runs sample.
pub struct FarmCase {
    options: FarmOptions,
    site: FarmSite,
    cages: Vec<NetCage>,
    /// Centre and radius of each cage, and the cages' mean centre
    circles: Vec<([f64; 2], f64)>,
    centre: [f64; 2],
    /// Radius of the disc round every cage
    farm_radius: f64,
    /// Bed depth below mean sea level at each cage's centre (m)
    depths: Vec<f64>,
    /// Number, smallest and largest edge (m) of the elements over the farm
    elements: (usize, f64, f64),
    /// The sampled grid: `n × n` points from `origin`, row by row (`None`
    /// outside the mesh)
    n: usize,
    origin: [f64; 2],
    probes: Vec<Option<Probe2D>>,
    centre_probe: Probe2D,
    cage_probes: Vec<Probe2D>,
}

impl FarmCase {
    pub fn new(
        options: FarmOptions,
        projection: &impl CoordinateProjection,
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        bathymetry: &Bathymetry2D,
    ) -> Result<Self, String> {
        let site = read_farm_site_file(&options.site)
            .map_err(|e| format!("{}: {e}", options.site.display()))?;
        let cages = site.cage_layout(projection, &options.grid);
        if cages.is_empty() {
            return Err(format!(
                "{}: no cages (no `cage` lines and no frame)",
                options.site.display()
            ));
        }
        let circles: Vec<([f64; 2], f64)> = cages
            .iter()
            .map(|c| match c.footprint {
                CageFootprint::Circle { center, radius } => (center, radius),
                _ => unreachable!("cage_layout gives circular cages"),
            })
            .collect();
        let n_cages = circles.len() as f64;
        let centre = [0, 1].map(|d| circles.iter().map(|(c, _)| c[d]).sum::<f64>() / n_cages);
        let farm_radius = circles
            .iter()
            .map(|(c, r)| (c[0] - centre[0]).hypot(c[1] - centre[1]) + r)
            .fold(0.0, f64::max);
        let locator = PointLocator2D::new(mesh);
        let probe = |p: [f64; 2]| Probe2D::at(&locator, ops, p);
        let centre_probe = probe(centre).ok_or("the farm is outside the mesh")?;
        let cage_probes: Vec<Probe2D> = circles
            .iter()
            .map(|(c, _)| probe(*c).ok_or("a cage is outside the mesh"))
            .collect::<Result<_, _>>()?;
        let depths = cage_probes
            .iter()
            .map(|p| -p.evaluate(bathymetry.element(p.element())))
            .collect();
        // The elements with a vertex over the farm, and their edges
        let mut elements = (0, f64::INFINITY, 0.0f64);
        for k in ElementIndex::iter(mesh.n_elements) {
            let v = mesh.element_vertices(k);
            let over = v
                .iter()
                .any(|p| (p[0] - centre[0]).hypot(p[1] - centre[1]) <= farm_radius);
            if !over {
                continue;
            }
            elements.0 += 1;
            for i in 0..4 {
                let (a, b) = (v[i], v[(i + 1) % 4]);
                let edge = (b[0] - a[0]).hypot(b[1] - a[1]);
                elements.1 = elements.1.min(edge);
                elements.2 = elements.2.max(edge);
            }
        }
        let n = (2.0 * options.window / options.spacing).round() as usize + 1;
        let origin = [centre[0] - options.window, centre[1] - options.window];
        let probes = (0..n * n)
            .map(|i| {
                let (row, col) = (i / n, i % n);
                probe([
                    origin[0] + col as f64 * options.spacing,
                    origin[1] + row as f64 * options.spacing,
                ])
            })
            .collect();
        Ok(Self {
            options,
            site,
            cages,
            circles,
            centre,
            farm_radius,
            depths,
            elements,
            n,
            origin,
            probes,
            centre_probe,
            cage_probes,
        })
    }

    pub fn options(&self) -> &FarmOptions {
        &self.options
    }

    /// The cages' drag on `mesh` at the order of `ops`.
    pub fn cage_drag(&self, mesh: &Mesh2D, ops: &DGOperators2D) -> CageDrag2D {
        CageDrag2D::new(mesh, ops, &self.cages)
    }

    /// The site and its cages.
    pub fn describe(&self, projection: &impl CoordinateProjection) {
        let grid = &self.options.grid;
        let (lat, lon) = projection.xy_to_geo(self.centre[0], self.centre[1]);
        let cage = &self.cages[0];
        println!(
            "  Farm: site {} {} ({}), {} cages of radius {:.1} m, nets {:.0} m deep, solidity {} \
             (C_d·a {:.4} 1/m), centred at {lat:.5}° N, {lon:.5}° E, ({:.2}, {:.2}) km; \
             within {:.0} m of the centre",
            self.site.number,
            self.site.name,
            self.options.site.display(),
            self.cages.len(),
            self.circles[0].1,
            cage.net_depth,
            grid.solidity,
            cage.drag_per_length,
            self.centre[0] / 1e3,
            self.centre[1] / 1e3,
            self.farm_radius
        );
        let depths: Vec<String> = self.depths.iter().map(|d| format!("{d:.0}")).collect();
        println!(
            "  Farm: bed at the cage centres {} m below mean sea level; {} elements over the farm,              edges {:.0}–{:.0} m; sampled every {} m over ±{} m",
            depths.join(", "),
            self.elements.0,
            self.elements.1,
            self.elements.2,
            self.options.spacing,
            self.options.window
        );
    }

    /// What a run records at the farm, with the cages (and the particles) or
    /// without them.
    pub fn record<'a>(
        &'a self,
        mesh: &'a Mesh2D,
        ops: &'a DGOperators2D,
        geom: &'a GeometricFactors2D,
        cages: bool,
    ) -> FarmRecord<'a> {
        let particles =
            (cages && self.options.particles > 0).then(|| FarmParticles::new(self, mesh, ops));
        FarmRecord {
            case: self,
            cages,
            drag: cages.then(|| (self.cage_drag(mesh, ops), geom)),
            n_nodes: ops.n_nodes,
            times: Vec::new(),
            grid: Vec::new(),
            centre: Vec::new(),
            cage_speed: Vec::new(),
            force: Vec::new(),
            particles,
        }
    }

    /// Position of grid point `i`.
    fn point(&self, i: usize) -> [f64; 2] {
        [
            self.origin[0] + (i % self.n) as f64 * self.options.spacing,
            self.origin[1] + (i / self.n) as f64 * self.options.spacing,
        ]
    }

    /// The grid field `field` (one sample) at `p`, bilinear between the four
    /// points round it if all four are wet.
    fn at(&self, field: &[[f32; 2]], p: [f64; 2]) -> Option<[f64; 2]> {
        let fx = (p[0] - self.origin[0]) / self.options.spacing;
        let fy = (p[1] - self.origin[1]) / self.options.spacing;
        if fx < 0.0 || fy < 0.0 {
            return None;
        }
        let (i, j) = (fx.floor() as usize, fy.floor() as usize);
        if i + 1 >= self.n || j + 1 >= self.n {
            return None;
        }
        let (a, b) = (fx - i as f64, fy - j as f64);
        let mut out = [0.0; 2];
        for (dj, wj) in [(0, 1.0 - b), (1, b)] {
            for (di, wi) in [(0, 1.0 - a), (1, a)] {
                let value = field[(j + dj) * self.n + i + di];
                if value[0].is_nan() {
                    return None;
                }
                out[0] += wi * wj * value[0] as f64;
                out[1] += wi * wj * value[1] as f64;
            }
        }
        Some(out)
    }
}

/// What one run records at the farm at every callback.
pub struct FarmRecord<'a> {
    case: &'a FarmCase,
    cages: bool,
    drag: Option<(CageDrag2D, &'a GeometricFactors2D)>,
    n_nodes: usize,
    times: Vec<f64>,
    /// The depth-mean current on the grid, `[time × point]`, NaN where dry
    grid: Vec<[f32; 2]>,
    /// The current at the farm's centre, and the speed at each cage centre
    /// (`[time × cage]`)
    centre: Vec<[f64; 2]>,
    cage_speed: Vec<f64>,
    /// The drag on the farm (N), with the cages
    force: Vec<[f64; 2]>,
    particles: Option<FarmParticles<'a>>,
}

impl FarmRecord<'_> {
    /// Sample the state `q` at `t` (and move the particles to it).
    pub fn sample(&mut self, q: &SWESolution2D, t: f64) {
        let h_dry = WetDryConfig::DEFAULT_H_DRY;
        let velocity = |probe: &Probe2D| {
            let s = probe.sample_swe(q, None, h_dry);
            (s.h > h_dry).then_some([s.u, s.v])
        };
        self.times.push(t);
        self.grid.extend(self.case.probes.iter().map(
            |probe| match probe.as_ref().and_then(velocity) {
                Some([u, v]) => [u as f32, v as f32],
                None => [f32::NAN; 2],
            },
        ));
        self.centre
            .push(velocity(&self.case.centre_probe).unwrap_or([0.0; 2]));
        self.cage_speed.extend(
            self.case
                .cage_probes
                .iter()
                .map(|p| velocity(p).map_or(0.0, |[u, v]| u.hypot(v))),
        );
        if let Some((drag, geom)) = &self.drag {
            let nn = self.n_nodes;
            let mut force = [0.0; 2];
            for k in 0..q.n_elements {
                for_each_caged_node(drag, k, nn, |i, entries| {
                    let idx = k * nn + i;
                    let s = q.get_state(ElementIndex::new(k), i);
                    let (h, hu, hv) = (s.h, s.hu, s.hv);
                    if h <= h_dry {
                        return;
                    }
                    let rate = CageDrag2D::damping_rate(entries, h, hu.hypot(hv) / h);
                    force[0] += RHO0 * geom.mass[idx] * rate * hu;
                    force[1] += RHO0 * geom.mass[idx] * rate * hv;
                });
            }
            self.force.push(force);
        }
        if let Some(particles) = self.particles.as_mut() {
            particles.advance(q, t);
        }
    }

    /// The samples in `[t0, t1]` (indices).
    fn window(&self, [t0, t1]: [f64; 2]) -> std::ops::Range<usize> {
        let first = self.times.partition_point(|&t| t < t0 - 1e-6);
        let end = self.times.partition_point(|&t| t <= t1 + 1e-6);
        first..end
    }

    fn sample_grid(&self, k: usize) -> &[[f32; 2]] {
        let n = self.case.probes.len();
        &self.grid[k * n..(k + 1) * n]
    }

    /// The mean current over samples `range` at each grid point (NaN where
    /// any sample was dry).
    fn residual(&self, range: std::ops::Range<usize>) -> Vec<[f64; 2]> {
        let count = range.len() as f64;
        let mut mean = vec![[0.0; 2]; self.case.probes.len()];
        for k in range {
            for (m, v) in mean.iter_mut().zip(self.sample_grid(k)) {
                m[0] += v[0] as f64 / count;
                m[1] += v[1] as f64 / count;
            }
        }
        mean
    }
}

/// Heading (degrees clockwise from north) the vector `[east, north]` points to.
fn heading([u, v]: [f64; 2]) -> f64 {
    u.atan2(v).to_degrees().rem_euclid(360.0)
}

fn speed([u, v]: [f64; 2]) -> f64 {
    u.hypot(v)
}

/// Mean and standard error of `values` (NaN error for one value).
fn mean_error(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    if values.len() < 2 {
        return (mean, f64::NAN);
    }
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, (var / n).sqrt())
}

fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

impl FarmCase {
    /// The farm's report (see the module docs) over the analysis window
    /// `window` (s), from the run with cages and, if any, the one without;
    /// the CSV files go to `output`.
    pub fn report(
        &self,
        reference: Option<&FarmRecord>,
        caged: &FarmRecord,
        window: [f64; 2],
        clock: &ModelClock,
        output: &Path,
    ) -> std::io::Result<()> {
        assert!(caged.cages);
        let range = caged.window(window);
        println!(
            "\nFarm {} {}: analysis over {:.2}–{:.2} h ({} samples, {:.2} M2 periods)",
            self.site.number,
            self.site.name,
            window[0] / 3600.0,
            window[1] / 3600.0,
            range.len(),
            (window[1] - window[0]) / M2_PERIOD
        );
        if range.len() < 2 {
            println!("  too few samples in the window");
            return Ok(());
        }
        let open = reference.unwrap_or(caged);
        let current = |record: &FarmRecord, label: &str| {
            let centre = &record.centre[record.window(window)];
            let n = centre.len() as f64;
            let mean = [0, 1].map(|d| centre.iter().map(|c| c[d]).sum::<f64>() / n);
            let speeds: Vec<f64> = centre.iter().map(|&c| speed(c)).collect();
            let mean_speed = speeds.iter().sum::<f64>() / n;
            let max = speeds.iter().copied().fold(0.0, f64::max);
            println!(
                "  Current at the farm's centre ({label}): mean speed {mean_speed:.3} m/s, \
                 largest {max:.3} m/s; residual {:.4} m/s toward {:.0}°",
                speed(mean),
                heading(mean)
            );
            mean
        };
        if let Some(reference) = reference {
            current(reference, "no cages");
        }
        current(caged, "cages");
        // Principal axis of the open flow at the centre
        let centre = &open.centre[open.window(window)];
        let n = centre.len() as f64;
        let mean = [0, 1].map(|d| centre.iter().map(|c| c[d]).sum::<f64>() / n);
        let (mut cuu, mut cvv, mut cuv) = (0.0, 0.0, 0.0);
        for c in centre {
            let (u, v) = (c[0] - mean[0], c[1] - mean[1]);
            cuu += u * u;
            cvv += v * v;
            cuv += u * v;
        }
        let theta = 0.5 * (2.0 * cuv).atan2(cuu - cvv);
        let axis = [theta.cos(), theta.sin()];
        let major = ((cuu + cvv) / 2.0 / n
            + (((cuu - cvv) / 2.0 / n).powi(2) + (cuv / n).powi(2)).sqrt())
        .sqrt();
        let minor = ((cuu + cvv) / n - major * major).max(0.0).sqrt();
        println!(
            "  Principal axis {:.0}°/{:.0}°, RMS speed along it {:.3} m/s, across {:.3} m/s",
            heading(axis),
            heading([-axis[0], -axis[1]]),
            major,
            minor
        );

        // Through the farm: the time-mean speed at the cage centres
        let n_cages = self.cages.len();
        let mean_speeds = |record: &FarmRecord| -> Vec<f64> {
            let r = record.window(window);
            let count = r.len() as f64;
            (0..n_cages)
                .map(|c| {
                    r.clone()
                        .map(|k| record.cage_speed[k * n_cages + c])
                        .sum::<f64>()
                        / count
                })
                .collect()
        };
        if let Some(reference) = reference {
            let (with, without) = (mean_speeds(caged), mean_speeds(reference));
            let ratios: Vec<String> = with
                .iter()
                .zip(&without)
                .map(|(a, b)| format!("{:.2}", a / b))
                .collect();
            println!(
                "  Mean speed at the cage centres, cages over no cages: {} (farm {:.3} against {:.3} m/s)",
                ratios.join(", "),
                with.iter().sum::<f64>() / n_cages as f64,
                without.iter().sum::<f64>() / n_cages as f64
            );
        }

        // Drag on the farm
        let forces: Vec<(f64, usize)> = range.clone().map(|k| (speed(caged.force[k]), k)).collect();
        let mean_force = forces.iter().map(|f| f.0).sum::<f64>() / forces.len() as f64;
        let (max_force, at) = forces
            .iter()
            .copied()
            .fold((0.0, range.start), |a, b| if b.0 > a.0 { b } else { a });
        println!(
            "  Drag on the farm: mean {:.1} kN, largest {:.1} kN at {:.2} h ({} UTC, {:.3} m/s at the centre)",
            mean_force / 1e3,
            max_force / 1e3,
            caged.times[at] / 3600.0,
            clock.format(caged.times[at]),
            speed(caged.centre[at])
        );

        let mut csv = BufWriter::new(File::create(output.join("farm_wake.csv"))?);
        if let Some(reference) = reference {
            self.report_wake(reference, caged, window, axis, &mut csv)?;
        } else {
            let residual = caged.residual(range.clone());
            writeln!(csv, "x_m,y_m,residual_u,residual_v")?;
            for (i, r) in residual.iter().enumerate() {
                let p = self.point(i);
                writeln!(
                    csv,
                    "{:.1},{:.1},{:.5},{:.5}",
                    p[0] - self.centre[0],
                    p[1] - self.centre[1],
                    r[0],
                    r[1]
                )?;
            }
        }
        csv.flush()?;
        let mut cages = BufWriter::new(File::create(output.join("farm_cages.csv"))?);
        writeln!(cages, "x_m,y_m,radius_m")?;
        for (c, r) in &self.circles {
            writeln!(
                cages,
                "{:.1},{:.1},{r:.1}",
                c[0] - self.centre[0],
                c[1] - self.centre[1]
            )?;
        }
        cages.flush()?;
        println!(
            "  Maps: {} (cages: farm_cages.csv; `scripts/farm_wake_map.py`)",
            output.join("farm_wake.csv").display()
        );

        if let Some(particles) = &caged.particles {
            particles.report(self, caged, clock, output)?;
        }
        Ok(())
    }

    /// The wake and the residual, cages against no cages.
    fn report_wake(
        &self,
        reference: &FarmRecord,
        caged: &FarmRecord,
        window: [f64; 2],
        axis: [f64; 2],
        csv: &mut impl std::io::Write,
    ) -> std::io::Result<()> {
        let range = caged.window(window);
        let open_range = reference.window(window);
        if open_range != range
            || reference.times[range.clone()]
                .iter()
                .zip(&caged.times[range.clone()])
                .any(|(a, b)| (a - b).abs() > 1e-6)
        {
            println!("  the runs were sampled at different times: no wake");
            return Ok(());
        }
        let n_points = self.probes.len();
        let spacing = self.options.spacing;
        // Per direction of the tide (along the axis, against it): the sum
        // and count of the relative deficit along the ray, the samples, the
        // mean direction, the area over 5/10/20 %, and the mean deficit map
        let mut ray = [[(0.0, 0usize); RAY.len()]; 2];
        let mut samples = [0usize; 2];
        let mut direction = [[0.0; 2]; 2];
        let mut edge = [0.0; 2];
        let mut area = [[0.0; 3]; 2];
        let mut deficit_map = vec![[(0.0f64, 0u32); 2]; n_points];
        let mut mean_speed = vec![[0.0f64; 2]; n_points];
        for k in range.clone() {
            let open = reference.sample_grid(k);
            let cages = caged.sample_grid(k);
            for (m, (a, b)) in mean_speed.iter_mut().zip(open.iter().zip(cages)) {
                m[0] += (a[0] as f64).hypot(a[1] as f64) / range.len() as f64;
                m[1] += (b[0] as f64).hypot(b[1] as f64) / range.len() as f64;
            }
            let u = reference.centre[k];
            let s = speed(u);
            if s < MIN_SPEED {
                continue;
            }
            let d = [u[0] / s, u[1] / s];
            let class = usize::from(d[0] * axis[0] + d[1] * axis[1] < 0.0);
            samples[class] += 1;
            direction[class][0] += d[0];
            direction[class][1] += d[1];
            edge[class] += self
                .circles
                .iter()
                .map(|(c, r)| (c[0] - self.centre[0]) * d[0] + (c[1] - self.centre[1]) * d[1] + r)
                .fold(f64::NEG_INFINITY, f64::max);
            for (j, &distance) in RAY.iter().enumerate() {
                let p = [
                    self.centre[0] + distance * d[0],
                    self.centre[1] + distance * d[1],
                ];
                if let (Some(a), Some(b)) = (self.at(open, p), self.at(cages, p))
                    && speed(a) >= POINT_MIN_SPEED
                {
                    ray[class][j].0 += 1.0 - speed(b) / speed(a);
                    ray[class][j].1 += 1;
                }
            }
            for (i, (a, b)) in open.iter().zip(cages).enumerate() {
                let (a, b) = ([a[0] as f64, a[1] as f64], [b[0] as f64, b[1] as f64]);
                if a[0].is_nan() || b[0].is_nan() || speed(a) < POINT_MIN_SPEED {
                    continue;
                }
                let r = 1.0 - speed(b) / speed(a);
                for (level, threshold) in [0.05, 0.1, 0.2].iter().enumerate() {
                    if r > *threshold {
                        area[class][level] += spacing * spacing;
                    }
                }
                deficit_map[i][class].0 += r;
                deficit_map[i][class].1 += 1;
            }
        }
        let headings = [0, 1].map(|c| heading(direction[c]));
        println!(
            "  Wake: relative speed deficit 1 − |u_cages|/|u_open| along the current through the \
             farm's centre (samples with ≥ {MIN_SPEED} m/s there)"
        );
        println!(
            "    toward {:.0}°: {} samples, farm edge {:.0} m down-current; toward {:.0}°: {} samples, edge {:.0} m",
            headings[0],
            samples[0],
            edge[0] / samples[0].max(1) as f64,
            headings[1],
            samples[1],
            edge[1] / samples[1].max(1) as f64
        );
        println!(
            "    distance (m) | toward {:.0}° | toward {:.0}°",
            headings[0], headings[1]
        );
        for (j, distance) in RAY.iter().enumerate() {
            let cell = |c: usize| {
                let (sum, count) = ray[c][j];
                if count == 0 {
                    "      —".to_string()
                } else {
                    format!("{:6.2} %", 100.0 * sum / count as f64)
                }
            };
            println!("    {distance:12.0} | {:>11} | {:>11}", cell(0), cell(1));
        }
        for c in 0..2 {
            let n = samples[c].max(1) as f64;
            println!(
                "    toward {:.0}°: mean area of the wake over 5 / 10 / 20 % {:.1} / {:.1} / {:.1} ha",
                headings[c],
                area[c][0] / n / 1e4,
                area[c][1] / n / 1e4,
                area[c][2] / n / 1e4
            );
        }

        // The residual current with and without the cages
        let open_residual = reference.residual(range.clone());
        let caged_residual = caged.residual(range.clone());
        let at_centre = |field: &[[f64; 2]]| {
            let f: Vec<[f32; 2]> = field.iter().map(|v| [v[0] as f32, v[1] as f32]).collect();
            self.at(&f, self.centre).unwrap_or([f64::NAN; 2])
        };
        let (a, b) = (at_centre(&open_residual), at_centre(&caged_residual));
        println!(
            "  Residual current at the farm's centre: {:.4} m/s toward {:.0}° without the cages, \
             {:.4} m/s toward {:.0}° with them",
            speed(a),
            heading(a),
            speed(b),
            heading(b)
        );
        for radius in [250.0, 500.0, 1000.0] {
            let (mut sum, mut sum_open, mut count, mut largest) = (0.0, 0.0, 0, (0.0, [0.0; 2]));
            for (i, (a, b)) in open_residual.iter().zip(&caged_residual).enumerate() {
                let p = self.point(i);
                let r = [p[0] - self.centre[0], p[1] - self.centre[1]];
                if a[0].is_nan() || b[0].is_nan() || speed(r) > radius {
                    continue;
                }
                let change = speed([b[0] - a[0], b[1] - a[1]]);
                sum += change;
                sum_open += speed(*a);
                count += 1;
                if change > largest.0 {
                    largest = (change, r);
                }
            }
            if count > 0 {
                println!(
                    "    within {radius:.0} m: residual speed {:.4} m/s without the cages, mean \
                     change {:.4} m/s, largest {:.4} m/s at ({:+.0}, {:+.0}) m",
                    sum_open / count as f64,
                    sum / count as f64,
                    largest.0,
                    largest.1[0],
                    largest.1[1]
                );
            }
        }

        writeln!(
            csv,
            "x_m,y_m,residual_u_open,residual_v_open,residual_u_cages,residual_v_cages,\
             speed_open,speed_cages,deficit_{:.0},deficit_{:.0}",
            headings[0], headings[1]
        )?;
        for i in 0..n_points {
            let p = self.point(i);
            let deficit = |c: usize| {
                let (sum, count) = deficit_map[i][c];
                if count == 0 {
                    f64::NAN
                } else {
                    sum / count as f64
                }
            };
            writeln!(
                csv,
                "{:.1},{:.1},{:.5},{:.5},{:.5},{:.5},{:.4},{:.4},{:.4},{:.4}",
                p[0] - self.centre[0],
                p[1] - self.centre[1],
                open_residual[i][0],
                open_residual[i][1],
                caged_residual[i][0],
                caged_residual[i][1],
                mean_speed[i][0],
                mean_speed[i][1],
                deficit(0),
                deficit(1)
            )?;
        }
        Ok(())
    }
}

/// One random walk of the particles: its tracer, `K` and seed, and its
/// particles' contacts with the cages and with the farm.
struct Walk<'a> {
    tracer: WalkTracer,
    kh: f64,
    tracker: ParticleTracker2D<'a>,
    particles: Vec<Particle2D>,
    cages: ConnectivityRecorder,
    farm: ConnectivityRecorder,
}

fn tracer_name(tracer: WalkTracer) -> &'static str {
    match tracer {
        WalkTracer::DepthIntegrated => "depth",
        WalkTracer::Surface => "surface",
    }
}

/// The farm's particles, tracked online in every walk (see the module docs).
struct FarmParticles<'a> {
    ops: &'a DGOperators2D,
    walks: Vec<Walk<'a>>,
    seeds: usize,
    /// Particles per cage and batch, and the cages they are released in
    n: usize,
    circles: Vec<([f64; 2], f64)>,
    /// The longest tracking step (s)
    step: f64,
    /// When the batches are due, and when each was released
    due: Vec<f64>,
    released: Vec<f64>,
    previous: Option<(f64, SWESolution2D)>,
    steps: usize,
    seconds: f64,
}

impl<'a> FarmParticles<'a> {
    fn new(case: &FarmCase, mesh: &'a Mesh2D, ops: &'a DGOperators2D) -> Self {
        let options = &case.options;
        let zones: Vec<ContactZone> = case
            .circles
            .iter()
            .map(|&(c, r)| ContactZone::circular(c, r, [0.0, f64::INFINITY]))
            .collect();
        // A return to its own cage only after a cage radius beyond it
        let cages = ConnectivityRecorder::new(zones).with_return_distance(case.circles[0].1);
        let farm = ConnectivityRecorder::new(vec![ContactZone::circular(
            case.centre,
            case.farm_radius,
            [0.0, f64::INFINITY],
        )])
        .with_return_distance(options.farm_return);
        let mut walks = Vec::new();
        for &tracer in &options.tracers {
            for &kh in &options.kh {
                for seed in 0..options.seeds as u64 {
                    walks.push(Walk {
                        tracer,
                        kh,
                        tracker: ParticleTracker2D::new(mesh, ops)
                            .with_diffusivity(kh)
                            .with_walk_tracer(tracer)
                            .with_stranding_depth(STRANDING_DEPTH)
                            .with_seed(seed),
                        particles: Vec::new(),
                        cages: cages.clone(),
                        farm: farm.clone(),
                    });
                }
            }
        }
        Self {
            ops,
            walks,
            seeds: options.seeds,
            n: options.particles,
            circles: case.circles.clone(),
            step: options.particle_seconds,
            due: (0..options.releases)
                .map(|b| options.release + b as f64 * options.release_every)
                .collect(),
            released: Vec::new(),
            previous: None,
            steps: 0,
            seconds: 0.0,
        }
    }

    /// Move every walk's particles from the previous state to `q` at `t`
    /// (linear in time between the two), then release the batches due.
    fn advance(&mut self, q: &SWESolution2D, t: f64) {
        let start = Instant::now();
        if let Some((t0, q0)) = &self.previous
            && !self.released.is_empty()
        {
            let field = SWEVelocity2D::between(*t0, q0, t, q, WetDryConfig::DEFAULT_H_DRY);
            let n = ((t - t0) / self.step).ceil().max(1.0) as usize;
            let dt = (t - t0) / n as f64;
            for s in 0..n {
                let ts = t0 + s as f64 * dt;
                for walk in &mut self.walks {
                    walk.tracker.step(&mut walk.particles, &field, ts, dt);
                    for (i, p) in walk.particles.iter().enumerate() {
                        if p.status() != ParticleStatus::Active {
                            continue;
                        }
                        walk.cages.record(i, ts + dt, dt, p.position(), || 0.0, 1.0);
                        walk.farm.record(i, ts + dt, dt, p.position(), || 0.0, 1.0);
                    }
                }
            }
            self.steps += n;
        }
        while self
            .due
            .get(self.released.len())
            .is_some_and(|&due| due <= t + 1e-6)
        {
            self.release(t);
        }
        match &mut self.previous {
            Some((tp, qp)) => {
                *tp = t;
                qp.clone_from(q);
            }
            None => self.previous = Some((t, q.clone())),
        }
        self.seconds += start.elapsed().as_secs_f64();
    }

    /// The next batch at `t`, in every walk: the same points in each.
    fn release(&mut self, t: f64) {
        let batch = self.released.len();
        let golden_angle = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
        for walk in &mut self.walks {
            for (c, &(centre, radius)) in self.circles.iter().enumerate() {
                for i in 0..self.n {
                    let r = radius * ((i as f64 + 0.5) / self.n as f64).sqrt();
                    let angle = i as f64 * golden_angle;
                    let p = [centre[0] + r * angle.cos(), centre[1] + r * angle.sin()];
                    let id = walk.particles.len() as u64;
                    let particle = walk
                        .tracker
                        .release(id, p)
                        .expect("the cages are in the mesh");
                    walk.particles.push(particle);
                    walk.cages.add(c, batch, t);
                    walk.farm.add(0, batch, t);
                }
            }
        }
        self.released.push(t);
    }

    /// The water depth at `p` in the last state.
    fn depth_at(&self, p: &Particle2D, weights: &mut [f64]) -> f64 {
        let Some((_, q)) = &self.previous else {
            return f64::NAN;
        };
        let point = p.point();
        self.ops
            .interpolation_weights_into(point.r, point.s, weights);
        weights
            .iter()
            .zip(q.element_h(point.element))
            .map(|(w, h)| w * h)
            .sum()
    }
}

/// One walk's connectivity over the batches a filter passes.
#[derive(Clone, Copy, Default)]
struct Shares {
    /// Returned to their own cage
    own: f64,
    /// Reached another cage
    other: f64,
    /// Exposure to the other cages per released particle (s)
    exposure: f64,
    /// Back at the farm from `farm_return` m beyond it
    farm: f64,
    /// Median age at the first contact with another cage (s)
    first: f64,
}

impl Shares {
    fn of(walk: &Walk, groups: impl Fn(usize) -> bool + Copy) -> Self {
        let matrix = walk.cages.matrix(0.0, groups);
        let n = matrix.n_zones();
        let released: usize = (0..n).map(|i| matrix.released(i)).sum();
        if released == 0 {
            return Self::default();
        }
        let total = released as f64;
        let own = (0..n).map(|i| matrix.reached(i, i)).sum::<usize>() as f64 / total;
        let exposure = (0..n)
            .flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j)))
            .map(|(i, j)| matrix.mean_exposure(i, j) * matrix.released(i) as f64)
            .sum::<f64>()
            / total;
        let mut firsts = Vec::new();
        for index in 0..walk.cages.len() {
            let (source, group, _) = walk.cages.particle(index);
            if !groups(group) {
                continue;
            }
            let first = walk
                .cages
                .particle_contacts(index)
                .iter()
                .filter(|c| c.zone != source)
                .map(|c| c.first)
                .fold(f64::INFINITY, f64::min);
            if first.is_finite() {
                firsts.push(first);
            }
        }
        Self {
            own,
            other: firsts.len() as f64 / total,
            exposure,
            farm: walk.farm.matrix(0.0, groups).share(0, 0),
            first: median(&mut firsts),
        }
    }
}

/// Where one walk's particles are at the end.
#[derive(Clone, Copy, Default)]
struct Fates {
    active: f64,
    stranded: f64,
    exited: f64,
    /// Of those in the domain: mean distance from the farm's centre (m), the
    /// share within 1 km of it, and the mean depth of the active ones (m)
    distance: f64,
    near: f64,
    depth: f64,
}

impl FarmParticles<'_> {
    fn fates(&self, walk: &Walk, centre: [f64; 2]) -> Fates {
        let total = walk.particles.len().max(1) as f64;
        let mut weights = vec![0.0; self.ops.n_nodes];
        let (mut fates, mut inside, mut active) = (Fates::default(), 0usize, 0usize);
        for p in &walk.particles {
            match p.status() {
                ParticleStatus::Active => fates.active += 1.0 / total,
                ParticleStatus::Stranded => fates.stranded += 1.0 / total,
                _ => {}
            }
            if !p.in_domain() {
                fates.exited += 1.0 / total;
                continue;
            }
            let [x, y] = p.position();
            let distance = (x - centre[0]).hypot(y - centre[1]);
            fates.distance += distance;
            fates.near += f64::from(distance <= 1000.0);
            inside += 1;
            if p.status() == ParticleStatus::Active {
                fates.depth += self.depth_at(p, &mut weights);
                active += 1;
            }
        }
        fates.distance /= inside.max(1) as f64;
        fates.near /= total;
        fates.depth /= active.max(1) as f64;
        fates
    }

    /// The connectivity and the particles' fates (see the module docs).
    fn report(
        &self,
        case: &FarmCase,
        caged: &FarmRecord,
        clock: &ModelClock,
        output: &Path,
    ) -> std::io::Result<()> {
        let per_walk = self.walks.first().map_or(0, |w| w.particles.len());
        println!(
            "\n  Particles: {} per cage in each of {} batches ({} per walk), {} walks \
             ({} seeds each); {} tracking steps (≤ {} s), {:.1} s tracking",
            self.n,
            self.released.len(),
            per_walk,
            self.walks.len(),
            self.seeds,
            self.steps,
            self.step,
            self.seconds
        );
        if self.released.is_empty() {
            println!("    none released");
            return Ok(());
        }
        for (b, &t) in self.released.iter().enumerate() {
            let k = caged.times.partition_point(|&s| s < t - 1e-6);
            let u = caged.centre.get(k).copied().unwrap_or([0.0; 2]);
            println!(
                "    batch {b}: released at {:.2} h ({} UTC), {:.3} m/s toward {:.0}° at the farm, tracked {:.1} h",
                t / 3600.0,
                clock.format(t),
                speed(u),
                heading(u),
                (caged.times.last().copied().unwrap_or(t) - t) / 3600.0
            );
        }
        // The walks by tracer and K, `seeds` each
        let groups: Vec<&[Walk]> = self.walks.chunks(self.seeds).collect();
        let n_batches = self.released.len();
        let all = |_: usize| true;
        println!(
            "    over all batches, mean ± standard error of {} seeds: own cage again | another \
             cage | exposure to the other cages per particle | back at the farm (from {:.0} m) \
             | median age at the first contact with another cage",
            self.seeds, case.options.farm_return
        );
        let summary =
            |walks: &[Walk], f: &dyn Fn(&Shares) -> f64, groups: &dyn Fn(usize) -> bool| {
                let values: Vec<f64> = walks.iter().map(|w| f(&Shares::of(w, groups))).collect();
                mean_error(&values)
            };
        let fields: [ShareField; 4] = [
            ("own", &|s| 100.0 * s.own),
            ("other", &|s| 100.0 * s.other),
            ("exposure", &|s| s.exposure),
            ("farm", &|s| 100.0 * s.farm),
        ];
        let pm = |(m, e): (f64, f64), unit: &str| {
            if e.is_nan() {
                format!("{m:.2}{unit}")
            } else {
                format!("{m:.2} ± {e:.2}{unit}")
            }
        };
        for walks in &groups {
            let walk = &walks[0];
            let s: Vec<(f64, f64)> = fields
                .iter()
                .map(|(_, f)| summary(walks, *f, &all))
                .collect();
            let first = summary(walks, &|s| s.first / 3600.0, &all);
            println!(
                "    {:>7} K = {}: {} | {} | {} | {} | {:.2} h",
                tracer_name(walk.tracer),
                walk.kh,
                pm(s[0], " %"),
                pm(s[1], " %"),
                pm(s[2], " s"),
                pm(s[3], " %"),
                first.0
            );
            for b in 0..n_batches {
                let batch = move |g: usize| g == b;
                let own = summary(walks, fields[0].1, &batch);
                let other = summary(walks, fields[1].1, &batch);
                let farm = summary(walks, fields[3].1, &batch);
                println!(
                    "            batch {b}: own {} | another {} | farm {}",
                    pm(own, " %"),
                    pm(other, " %"),
                    pm(farm, " %")
                );
            }
            let fates: Vec<Fates> = walks.iter().map(|w| self.fates(w, case.centre)).collect();
            let fate =
                |f: &dyn Fn(&Fates) -> f64| mean_error(&fates.iter().map(f).collect::<Vec<_>>());
            println!(
                "            at the end: {} active, {} stranded, {} out; {} within 1 km of the \
                 farm, mean distance {}; mean depth of the active ones {}",
                pm(fate(&|f| 100.0 * f.active), " %"),
                pm(fate(&|f| 100.0 * f.stranded), " %"),
                pm(fate(&|f| 100.0 * f.exited), " %"),
                pm(fate(&|f| 100.0 * f.near), " %"),
                pm(fate(&|f| f.distance), " m"),
                pm(fate(&|f| f.depth), " m")
            );
        }
        // The depth drift: each `K`'s depth walk against its surface walk
        for depth in groups
            .iter()
            .filter(|w| w[0].tracer == WalkTracer::DepthIntegrated)
        {
            let Some(surface) = groups
                .iter()
                .find(|w| w[0].tracer == WalkTracer::Surface && w[0].kh == depth[0].kh)
            else {
                continue;
            };
            let difference = |a: (f64, f64), b: (f64, f64)| {
                let error = a.1.hypot(b.1);
                if error > 0.0 {
                    format!("{:+.2} (z {:+.1})", a.0 - b.0, (a.0 - b.0) / error)
                } else {
                    format!("{:+.2}", a.0 - b.0)
                }
            };
            let mut parts = Vec::new();
            for (name, f) in fields {
                let (a, b) = (summary(depth, f, &all), summary(surface, f, &all));
                parts.push(format!("{name} {}", difference(a, b)));
            }
            let fate = |walks: &[Walk], f: &dyn Fn(&Fates) -> f64| {
                mean_error(
                    &walks
                        .iter()
                        .map(|w| f(&self.fates(w, case.centre)))
                        .collect::<Vec<_>>(),
                )
            };
            for (name, f) in [
                (
                    "stranded %",
                    &(|f: &Fates| 100.0 * f.stranded) as &dyn Fn(&Fates) -> f64,
                ),
                ("depth m", &|f: &Fates| f.depth),
                ("distance m", &|f: &Fates| f.distance),
            ] {
                let (a, b) = (fate(depth, f), fate(surface, f));
                parts.push(format!("{name} {}", difference(a, b)));
            }
            println!(
                "    depth − surface walk, K = {}: {}",
                depth[0].kh,
                parts.join(", ")
            );
        }
        // The cage-by-cage matrix of the first walk, mean over its seeds
        let first = groups[0];
        let matrices: Vec<ConnectivityMatrix> =
            first.iter().map(|w| w.cages.matrix(0.0, all)).collect();
        let n = case.circles.len();
        println!(
            "    {} walk, K = {}: share (%) from each cage (rows) to each cage (columns), mean of the seeds",
            tracer_name(first[0].tracer),
            first[0].kh
        );
        println!(
            "      {}",
            (0..n).map(|j| format!("{j:>6}")).collect::<String>()
        );
        for i in 0..n {
            let row: String = (0..n)
                .map(|j| {
                    let share =
                        matrices.iter().map(|m| m.share(i, j)).sum::<f64>() / matrices.len() as f64;
                    format!("{:6.1}", 100.0 * share)
                })
                .collect();
            println!("    {i:>2}{row}");
        }

        let path = output.join("connectivity.csv");
        let mut csv = BufWriter::new(File::create(&path)?);
        writeln!(
            csv,
            "walk,kh,seed,batch,source,receiver,released,reached,mean_exposure_s"
        )?;
        for walks in &groups {
            for (seed, walk) in walks.iter().enumerate() {
                for b in 0..n_batches {
                    let m = walk.cages.matrix(0.0, |g| g == b);
                    for i in 0..n {
                        for j in 0..n {
                            writeln!(
                                csv,
                                "{},{},{seed},{b},{i},{j},{},{},{:.1}",
                                tracer_name(walk.tracer),
                                walk.kh,
                                m.released(i),
                                m.reached(i, j),
                                m.mean_exposure(i, j)
                            )?;
                        }
                    }
                }
            }
        }
        csv.flush()?;
        println!("    Connectivity per seed and batch: {}", path.display());
        Ok(())
    }
}

/// A connectivity statistic by name.
type ShareField<'f> = (&'static str, &'f dyn Fn(&Shares) -> f64);

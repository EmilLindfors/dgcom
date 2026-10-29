//! Local time stepping on a farm-refined mesh (TODO P2.5).
//!
//! The fjord arm of `docs/gmsh-meshes.md`: 12 × 6 km, open to the sea in the
//! west, quads refined from ≈ 450 m to ≈ 20 m at a fish farm in the middle
//! (`tests/data/gmsh/fjord_farm.msh`, from `scripts/gmsh_farm_mesh.py`). The
//! bed deepens from 30 m at the head to 150 m at the mouth. An M2 tide of
//! 0.8 m enters through a characteristic open boundary; Coriolis, Manning
//! friction and two net cages act on the flow.
//!
//! The tide ramps up from rest over `ramp` seconds (default one hour). Switched
//! on at full amplitude (`ramp=0`), it sends a ≈ 0.13 m/s surge past the farm,
//! whose drag wake then outlasts the few mm/s of tidal current there unless
//! horizontal viscosity (`nu`, a constant in m²/s; default none) removes it
//! (TODO F.1). The step is set by the gravity wave on the 20 m elements, so
//! neither the ramp nor a viscosity of a few m²/s changes it.
//!
//! The same run is taken twice from rest, with global SSP-RK3 and with
//! `MultirateSSPRK3` (`run=global` or `run=local` for one of them). It
//! reports the wall time, the RHS work and the difference between the two
//! solutions: η and speed at the farm and over the domain.
//!
//! With `particles=N`, each run also releases N particles in each cage at
//! t = 0 and tracks them online (TODO F.2): between the solver's snapshots
//! every `particle_seconds` (default 60), linear in time, with a horizontal
//! random walk of diffusivity `kh` (m²/s, default 0.1). It reports where the
//! clouds went, how far they spread, and what the tracking cost.
//!
//! ```bash
//! cargo run --release --no-default-features --features parallel,simd \
//!     --example local_time_stepping_farm -- [hours=1] [order=2] [levels=8] [ramp=3600] \
//!     [nu=0] [run=both] [particles=0] [particle_seconds=60] [kh=0.1]
//! ```

#[path = "common/arguments.rs"]
mod arguments;

use std::f64::consts::PI;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use dg_rs::boundary::{CharacteristicOBC, HarmonicTide, MultiBoundaryCondition2D, Reflective2D};
use dg_rs::equations::ShallowWater2D;
use dg_rs::mesh::{Bathymetry2D, read_gmsh_mesh};
use dg_rs::operators::{DGOperators2D, GeometricFactors2D};
use dg_rs::particles::{Particle2D, ParticleStatus, ParticleTracker2D, SWEVelocity2D};
use dg_rs::physics::PhysicsBuilder;
use dg_rs::simulation::{Simulation, SimulationResult};
use dg_rs::solver::{SWESolution2D, SWEState2D, StandardLimiter2D, WetDryConfig};
use dg_rs::source::{
    CageDrag2D, CageFootprint, CoriolisSource2D, HorizontalViscosity2D, ManningFriction2D, NetCage,
};
use dg_rs::time::{MultirateSSPRK3, SSPRK3};
use dg_rs::types::ElementIndex;

const G: f64 = 9.81;
const LX: f64 = 12_000.0;
const FARM: [f64; 2] = [6_000.0, 3_000.0];

fn bed(x: f64, y: f64) -> f64 {
    -(30.0 + 120.0 * (1.0 - x / LX)) - 10.0 * (PI * y / 6_000.0).sin()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = arguments::Arguments::from_env();
    let get = |key: &str, default: f64| args.get(key).map_or(Ok(default), |v| v.parse());
    let hours: f64 = get("hours", 1.0)?;
    let order = get("order", 2.0)? as usize;
    let levels = get("levels", 8.0)? as usize;
    let ramp: f64 = get("ramp", 3600.0)?;
    let nu: f64 = get("nu", 0.0)?;
    let run_which = args.get("run").map_or("both", String::as_str);
    let particles_per_cage = get("particles", 0.0)? as usize;
    let particle_seconds: f64 = get("particle_seconds", 60.0)?;
    let kh: f64 = get("kh", 0.1)?;
    args.refuse_unread()?;

    let mesh = Arc::new(read_gmsh_mesh(Path::new("tests/data/gmsh/fjord_farm.msh"))?);
    let ops = Arc::new(DGOperators2D::new(order));
    let geom = Arc::new(GeometricFactors2D::compute(&mesh, &ops));
    let bathymetry = Arc::new(Bathymetry2D::from_function(&mesh, &ops, &geom, bed));
    let sizes: Vec<f64> = ElementIndex::iter(mesh.n_elements)
        .map(|k| geom.area[k.as_usize()].sqrt())
        .collect();
    println!(
        "Mesh: {} quads (P{order}), element size {:.0}–{:.0} m",
        mesh.n_elements,
        sizes.iter().copied().fold(f64::INFINITY, f64::min),
        sizes.iter().copied().fold(0.0, f64::max)
    );

    let wall = Reflective2D::new();
    let tide = HarmonicTide::m2(0.8, 0.0);
    let sea = CharacteristicOBC::new(if ramp > 0.0 {
        tide.with_ramp_up(ramp)
    } else {
        tide
    });
    let cages = [
        NetCage::circular([FARM[0] - 40.0, FARM[1]], 25.0, 20.0, 0.25),
        NetCage::circular([FARM[0] + 40.0, FARM[1]], 25.0, 20.0, 0.25),
    ];
    let physics = || {
        let builder = PhysicsBuilder::swe_2d(
            mesh.clone(),
            ops.clone(),
            geom.clone(),
            ShallowWater2D::new(G),
            MultiBoundaryCondition2D::new(&wall).with_open(&sea),
        )
        .with_bathymetry(bathymetry.clone())
        .with_limiter(StandardLimiter2D::Positivity(WetDryConfig::DEFAULT_H_DRY))
        .with_wet_dry(WetDryConfig::default())
        .with_implicit_friction(ManningFriction2D::new(G, 0.025))
        .with_cage_drag(CageDrag2D::new(&mesh, &ops, &cages))
        .with_source(CoriolisSource2D::f_plane(1.2e-4));
        if nu > 0.0 {
            builder.with_viscosity(HorizontalViscosity2D::constant(nu))
        } else {
            builder
        }
        .build()
    };
    let at_rest = || {
        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let h = -bathymetry.get(k, i);
                q.set_state(k, i, SWEState2D::new(h, 0.0, 0.0));
            }
        }
        q
    };
    let t_end = hours * 3600.0;

    let run = |label: &str, q: &mut SWESolution2D, local: bool| -> SimulationResult {
        let mut tracking = (particles_per_cage > 0)
            .then(|| FarmParticles::release(&mesh, &ops, &cages, particles_per_cage, kh));
        let mut callback = |q: &SWESolution2D, t: f64| {
            if let Some(tracking) = tracking.as_mut() {
                tracking.advance(q, t);
            }
        };
        let interval = (particles_per_cage > 0).then_some(particle_seconds);
        let start = Instant::now();
        let result = if local {
            let sim = Simulation::new(physics(), MultirateSSPRK3::new(levels)).with_cfl(1.0);
            match interval {
                Some(i) => sim.with_callback_interval(i),
                None => sim,
            }
            .run_with_callback(q, 0.0, t_end, &mut callback)
        } else {
            let sim = Simulation::new(physics(), SSPRK3).with_cfl(1.0);
            match interval {
                Some(i) => sim.with_callback_interval(i),
                None => sim,
            }
            .run_with_callback(q, 0.0, t_end, &mut callback)
        };
        let wall = start.elapsed().as_secs_f64();
        if let Some(tracking) = &tracking {
            tracking.report(label, &cages);
        }
        println!(
            "{label:>9}: {:6} steps, dt {:.3}–{:.3} s, {wall:6.1} s wall{}",
            result.n_steps,
            result.dt_min,
            result.dt_max,
            result
                .local_time_stepping
                .map_or(String::new(), |s| format!(
                    ", levels 0–{}, {:.2}x less RHS work",
                    s.finest_level,
                    s.speedup()
                ))
        );
        assert!(result.success, "{label}: {:?}", result.error);
        result
    };

    let viscosity = if nu > 0.0 {
        format!(", ν = {nu} m²/s")
    } else {
        String::new()
    };
    println!("\nM2 from rest, {hours} h{viscosity}:");
    let mut global = at_rest();
    let mut local = at_rest();
    match run_which {
        "global" => {
            run("global", &mut global, false);
            return Ok(());
        }
        "local" => {
            run("local", &mut local, true);
            return Ok(());
        }
        _ => {
            run("global", &mut global, false);
            run("local", &mut local, true);
        }
    }

    // Differences near the farm (within 500 m) and everywhere
    let mut near = (0.0_f64, 0.0_f64, 0.0_f64);
    let mut all = (0.0_f64, 0.0_f64, 0.0_f64);
    for k in ElementIndex::iter(mesh.n_elements) {
        for i in 0..ops.n_nodes {
            let [x, y] = mesh.reference_to_physical(k, ops.nodes_r[i], ops.nodes_s[i]);
            let (a, b) = (global.get_state(k, i), local.get_state(k, i));
            let speed = |s: SWEState2D| (s.hu * s.hu + s.hv * s.hv).sqrt() / s.h;
            let (d_eta, d_speed) = ((a.h - b.h).abs(), (speed(a) - speed(b)).abs());
            let near_farm = (x - FARM[0]).hypot(y - FARM[1]) < 500.0;
            for d in [&mut all].into_iter().chain(near_farm.then_some(&mut near)) {
                d.0 = d.0.max(d_eta);
                d.1 = d.1.max(d_speed);
                d.2 = d.2.max(speed(a));
            }
        }
    }
    println!(
        "\nmax |local − global|: near the farm η {:.2e} m, speed {:.2e} m/s (max speed {:.2} m/s)",
        near.0, near.1, near.2
    );
    println!(
        "                      everywhere    η {:.2e} m, speed {:.2e} m/s (max speed {:.2} m/s)",
        all.0, all.1, all.2
    );
    Ok(())
}

/// Particles released in the cages and tracked online (TODO F.2).
struct FarmParticles<'a> {
    tracker: ParticleTracker2D<'a>,
    particles: Vec<Particle2D>,
    /// Release point of each particle and the cage it came from
    origins: Vec<([f64; 2], usize)>,
    /// The solver's previous snapshot
    previous: Option<(f64, SWESolution2D)>,
    seconds: f64,
    steps: usize,
}

impl<'a> FarmParticles<'a> {
    /// `n` particles in each cage footprint (a sunflower pattern), with a
    /// random walk of diffusivity `kh`; they strand in water under 5 cm.
    fn release(
        mesh: &'a dg_rs::mesh::Mesh2D,
        ops: &'a DGOperators2D,
        cages: &[NetCage],
        n: usize,
        kh: f64,
    ) -> Self {
        let tracker = ParticleTracker2D::new(mesh, ops)
            .with_diffusivity(kh)
            .with_stranding_depth(0.05);
        let golden_angle = PI * (3.0 - 5.0_f64.sqrt());
        let mut particles = Vec::new();
        let mut origins = Vec::new();
        for (c, cage) in cages.iter().enumerate() {
            let CageFootprint::Circle { center, radius } = cage.footprint else {
                panic!("the farm's cages are circles");
            };
            for i in 0..n {
                let r = radius * ((i as f64 + 0.5) / n as f64).sqrt();
                let angle = i as f64 * golden_angle;
                let p = [center[0] + r * angle.cos(), center[1] + r * angle.sin()];
                let particle = tracker
                    .release(particles.len() as u64, p)
                    .expect("cages are in the water");
                particles.push(particle);
                origins.push((p, c));
            }
        }
        Self {
            tracker,
            particles,
            origins,
            previous: None,
            seconds: 0.0,
            steps: 0,
        }
    }

    /// Move the particles from the previous snapshot to `q` at `t`, in
    /// steps of at most 10 s.
    fn advance(&mut self, q: &SWESolution2D, t: f64) {
        let start = Instant::now();
        if let Some((t0, q0)) = &self.previous {
            let field = SWEVelocity2D::between(*t0, q0, t, q, WetDryConfig::DEFAULT_H_DRY);
            let n = ((t - t0) / 10.0).ceil().max(1.0) as usize;
            let dt = (t - t0) / n as f64;
            for s in 0..n {
                self.tracker
                    .step(&mut self.particles, &field, t0 + s as f64 * dt, dt);
            }
            self.steps += n;
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

    /// Per cage: particles still in the domain, stranded and out through
    /// the open boundary; mean drift and RMS spread about the mean.
    fn report(&self, label: &str, cages: &[NetCage]) {
        println!(
            "{label:>9}: {} particles, {} tracking steps, {:.2} s tracking",
            self.particles.len(),
            self.steps,
            self.seconds
        );
        for c in 0..cages.len() {
            let cloud: Vec<(&Particle2D, [f64; 2])> = self
                .particles
                .iter()
                .zip(&self.origins)
                .filter(|(_, (_, cage))| *cage == c)
                .map(|(p, (origin, _))| (p, *origin))
                .collect();
            let count =
                |status: ParticleStatus| cloud.iter().filter(|(p, _)| p.status() == status).count();
            let exited = cloud.iter().filter(|(p, _)| !p.in_domain()).count();
            let inside: Vec<[f64; 2]> = cloud
                .iter()
                .filter(|(p, _)| p.in_domain())
                .map(|(p, o)| [p.position()[0] - o[0], p.position()[1] - o[1]])
                .collect();
            let n = inside.len().max(1) as f64;
            let mean = [0, 1].map(|d| inside.iter().map(|x| x[d]).sum::<f64>() / n);
            let spread = (inside
                .iter()
                .map(|x| (x[0] - mean[0]).powi(2) + (x[1] - mean[1]).powi(2))
                .sum::<f64>()
                / n)
                .sqrt();
            println!(
                "           cage {c}: {} active, {} stranded, {exited} out; drift ({:+.0}, {:+.0}) m, spread {spread:.0} m",
                count(ParticleStatus::Active),
                count(ParticleStatus::Stranded),
                mean[0],
                mean[1]
            );
        }
    }
}

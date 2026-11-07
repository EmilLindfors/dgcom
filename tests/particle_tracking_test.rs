//! Gates for Lagrangian particle tracking (`dgcom::particles`, TODO F.2).
//!
//! - Solid-body rotation, a field the bilinear elements hold exactly: one
//!   revolution returns to the start to the RK4 error, converging at fourth
//!   order in the step.
//! - No particle crosses a wall, however far the random walk throws it.
//! - Visser's well-mixed condition: a uniform distribution in a closed basin
//!   stays uniform under the random walk, and in open water the spread grows
//!   as 2Kt.
//! - Open boundaries let particles out where they cross; periodic faces
//!   carry them across.
//! - Particles strand on drying ground and float again when it floods.
//! - Reproducible random walk: a particle's path does not depend on which
//!   other particles are tracked with it.
//! - Online tracking in a running `Simulation` between callback snapshots
//!   converges at second order in the snapshot interval.

use std::sync::Arc;

use dgcom::boundary::Reflective2D;
use dgcom::equations::ShallowWater2D;
use dgcom::mesh::{BoundaryTag, Mesh2D, PointLocator2D};
use dgcom::operators::{DGOperators2D, GeometricFactors2D};
use dgcom::particles::{
    NodalVelocity2D, Particle2D, ParticleStatus, ParticleTracker2D, SWEVelocity2D,
};
use dgcom::physics::PhysicsBuilder;
use dgcom::simulation::Simulation;
use dgcom::solver::{SWESolution2D, SWEState2D};
use dgcom::source::{SourceContext2D, SourceTerm2D};
use dgcom::time::SSPRK3;
use dgcom::types::ElementIndex;

/// `nx × ny` elements over `[x0, x1] × [y0, y1]` with the interior vertices
/// displaced: general convex quadrilaterals, straight outer walls.
fn distorted_mesh(x0: f64, x1: f64, y0: f64, y1: f64, nx: usize, ny: usize) -> Mesh2D {
    let mut mesh = Mesh2D::uniform_rectangle(x0, x1, y0, y1, nx, ny);
    let (hx, hy) = ((x1 - x0) / nx as f64, (y1 - y0) / ny as f64);
    let eps = 1e-9 * (x1 - x0).max(y1 - y0);
    for v in &mut mesh.vertices {
        let [x, y] = *v;
        if x > x0 + eps && x < x1 - eps && y > y0 + eps && y < y1 - eps {
            let (sx, sy) = ((x - x0) / (x1 - x0), (y - y0) / (y1 - y0));
            v[0] += 0.2 * hx * (7.0 * sx + 3.0 * sy).sin();
            v[1] += 0.2 * hy * (5.0 * sx - 4.0 * sy).cos();
        }
    }
    mesh
}

/// Nodal values of `f(x, y)`, element by element.
fn nodal(mesh: &Mesh2D, ops: &DGOperators2D, f: impl Fn(f64, f64) -> f64) -> Vec<f64> {
    let mut values = Vec::with_capacity(mesh.n_elements * ops.n_nodes);
    for k in ElementIndex::iter(mesh.n_elements) {
        for i in 0..ops.n_nodes {
            let [x, y] = mesh.reference_to_physical(k, ops.nodes_r[i], ops.nodes_s[i]);
            values.push(f(x, y));
        }
    }
    values
}

/// Quasi-random points (an R2 sequence) in `[x0, x1] × [y0, y1]`.
fn scattered(n: usize, [x0, x1, y0, y1]: [f64; 4]) -> Vec<[f64; 2]> {
    (0..n)
        .map(|i| {
            let t = i as f64 + 0.5;
            [
                x0 + (x1 - x0) * (0.754_877_666_2 * t).fract(),
                y0 + (y1 - y0) * (0.569_840_290_9 * t).fract(),
            ]
        })
        .collect()
}

fn release_all(tracker: &ParticleTracker2D, points: &[[f64; 2]]) -> Vec<Particle2D> {
    points
        .iter()
        .enumerate()
        .map(|(i, &p)| tracker.release(i as u64, p).expect("inside the mesh"))
        .collect()
}

/// The particle's element and reference coordinates map back to its
/// position.
fn assert_consistent(mesh: &Mesh2D, p: &Particle2D) {
    let point = p.point();
    let [x, y] = mesh.reference_to_physical(point.element, point.r, point.s);
    let [px, py] = p.position();
    assert!(
        (x - px).abs() < 1e-10 && (y - py).abs() < 1e-10,
        "particle {} at {:?}, its point maps to {:?}",
        p.id(),
        p.position(),
        [x, y]
    );
}

#[test]
fn solid_body_rotation_converges_at_fourth_order() {
    let mesh = distorted_mesh(-1.0, 1.0, -1.0, 1.0, 9, 7);
    let ops = DGOperators2D::new(2);
    let omega = 0.7;
    let (u, v) = (
        nodal(&mesh, &ops, |_, y| -omega * y),
        nodal(&mesh, &ops, |x, _| omega * x),
    );
    let field = NodalVelocity2D::new(&u, &v, ops.n_nodes);
    let tracker = ParticleTracker2D::new(&mesh, &ops);
    // Circles of radius 0.1–0.9, which stay inside the square
    let start: Vec<[f64; 2]> = (0..24)
        .map(|i| {
            let (radius, angle) = (0.1 + 0.8 * i as f64 / 23.0, 2.4 * i as f64);
            [radius * angle.cos(), radius * angle.sin()]
        })
        .collect();
    let period = std::f64::consts::TAU / omega;
    let error = |steps: usize| {
        let mut particles = release_all(&tracker, &start);
        let dt = period / steps as f64;
        for n in 0..steps {
            tracker.step(&mut particles, &field, n as f64 * dt, dt);
        }
        particles
            .iter()
            .zip(&start)
            .map(|(p, s)| {
                assert_eq!(p.status(), ParticleStatus::Active);
                assert_consistent(&mesh, p);
                let [x, y] = p.position();
                ((x - s[0]).powi(2) + (y - s[1]).powi(2)).sqrt()
            })
            .fold(0.0, f64::max)
    };
    let (coarse, fine) = (error(50), error(100));
    let rate = (coarse / fine).log2();
    println!("rotation: {coarse:.3e} → {fine:.3e}, rate {rate:.2}");
    assert!(rate > 3.8, "rate {rate:.2}");
    assert!(fine < 2e-6, "error {fine:.2e} after one revolution");
}

/// A random walk much wider than an element, plus a flow into the corner:
/// every particle stays in the basin, and its element holds it.
#[test]
fn no_particle_crosses_a_wall() {
    let mesh = distorted_mesh(0.0, 1.0, 0.0, 1.0, 6, 6);
    let ops = DGOperators2D::new(3);
    let (u, v) = (
        nodal(&mesh, &ops, |_, _| 0.3),
        nodal(&mesh, &ops, |_, _| 0.2),
    );
    let field = NodalVelocity2D::new(&u, &v, ops.n_nodes);
    // √(2KΔt) = 0.45: several elements, often several walls, per step
    let tracker = ParticleTracker2D::new(&mesh, &ops)
        .with_diffusivity(0.1)
        .with_seed(7);
    let mut particles = release_all(&tracker, &scattered(2000, [0.0, 1.0, 0.0, 1.0]));
    for n in 0..40 {
        tracker.step(&mut particles, &field, n as f64, 1.0);
    }
    for p in &particles {
        assert_eq!(p.status(), ParticleStatus::Active);
        let [x, y] = p.position();
        assert!(
            (0.0..=1.0).contains(&x) && (0.0..=1.0).contains(&y),
            "particle {} outside the basin at {:?}",
            p.id(),
            p.position()
        );
        assert_consistent(&mesh, p);
    }
}

/// Visser (1997): a uniform distribution in a closed basin stays uniform
/// (no piling up at the walls), for steps smaller and larger than the
/// elements.
#[test]
fn random_walk_keeps_a_uniform_distribution_uniform() {
    let mesh = distorted_mesh(0.0, 1.0, 0.0, 1.0, 8, 8);
    let ops = DGOperators2D::new(2);
    let zero = vec![0.0; mesh.n_elements * ops.n_nodes];
    let field = NodalVelocity2D::new(&zero, &zero, ops.n_nodes);
    let n = 16_000;
    for (diffusivity, steps) in [(2e-3, 60), (0.05, 10)] {
        let tracker = ParticleTracker2D::new(&mesh, &ops)
            .with_diffusivity(diffusivity)
            .with_seed(11);
        let mut particles = release_all(&tracker, &scattered(n, [0.0, 1.0, 0.0, 1.0]));
        for step in 0..steps {
            tracker.step(&mut particles, &field, step as f64, 1.0);
        }
        // 8 × 8 bins, the outer ones against the walls
        let bins = 8;
        let mut counts = vec![0usize; bins * bins];
        for p in &particles {
            let [x, y] = p.position();
            let (i, j) = (
                ((x * bins as f64) as usize).min(bins - 1),
                ((y * bins as f64) as usize).min(bins - 1),
            );
            counts[j * bins + i] += 1;
        }
        let expected = n as f64 / (bins * bins) as f64;
        let chi2: f64 = counts
            .iter()
            .map(|&c| (c as f64 - expected).powi(2) / expected)
            .sum();
        // χ² with 63 degrees of freedom: mean 63, 99.9 % below 104
        println!("K = {diffusivity}: χ² = {chi2:.1}");
        assert!(
            chi2 < 110.0,
            "K = {diffusivity}: χ² = {chi2:.1}, {counts:?}"
        );
    }
}

/// Away from walls the spread is 2Kt per axis, and the mean does not move.
#[test]
fn random_walk_spreads_at_the_diffusivity() {
    let mesh = Mesh2D::uniform_rectangle(-100.0, 100.0, -100.0, 100.0, 10, 10);
    let ops = DGOperators2D::new(1);
    let zero = vec![0.0; mesh.n_elements * ops.n_nodes];
    let field = NodalVelocity2D::new(&zero, &zero, ops.n_nodes);
    let (diffusivity, dt, steps) = (0.5, 2.0, 10);
    let tracker = ParticleTracker2D::new(&mesh, &ops)
        .with_diffusivity(diffusivity)
        .with_seed(3);
    let n = 20_000;
    let mut particles = release_all(&tracker, &vec![[1.0, -2.0]; n]);
    for step in 0..steps {
        tracker.step(&mut particles, &field, step as f64 * dt, dt);
    }
    let expected = 2.0 * diffusivity * dt * steps as f64;
    for axis in 0..2 {
        let origin = [1.0, -2.0][axis];
        let d: Vec<f64> = particles
            .iter()
            .map(|p| p.position()[axis] - origin)
            .collect();
        let mean = d.iter().sum::<f64>() / n as f64;
        let variance = d.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        // Standard errors: mean √(20/n) = 0.03, variance 20·√(2/n) = 0.2
        assert!(mean.abs() < 0.15, "axis {axis}: mean {mean}");
        assert!(
            (variance / expected - 1.0).abs() < 0.05,
            "axis {axis}: variance {variance} against {expected}"
        );
    }
}

#[test]
fn particles_leave_through_open_boundaries() {
    let mesh = Mesh2D::uniform_rectangle_with_sides(
        0.0,
        4.0,
        0.0,
        1.0,
        8,
        2,
        [
            BoundaryTag::Wall,
            BoundaryTag::Open,
            BoundaryTag::Wall,
            BoundaryTag::Wall,
        ],
    );
    let ops = DGOperators2D::new(2);
    let (u, v) = (
        nodal(&mesh, &ops, |_, _| 1.0),
        nodal(&mesh, &ops, |_, _| 0.1),
    );
    let field = NodalVelocity2D::new(&u, &v, ops.n_nodes);
    let tracker = ParticleTracker2D::new(&mesh, &ops);
    let start = [[0.5, 0.2], [2.0, 0.3], [3.9, 0.1]];
    let mut particles = release_all(&tracker, &start);
    let dt = 0.25;
    for step in 0..20 {
        tracker.step(&mut particles, &field, step as f64 * dt, dt);
    }
    for (p, s) in particles.iter().zip(&start) {
        assert_eq!(p.status(), ParticleStatus::Exited(Some(BoundaryTag::Open)));
        assert!(!p.in_domain());
        // Left where the straight path crosses x = 4, then stopped
        let [x, y] = p.position();
        let expected_y = s[1] + 0.1 * (4.0 - s[0]);
        assert!(
            (x - 4.0).abs() < 1e-12 && (y - expected_y).abs() < 1e-12,
            "{:?}",
            p.position()
        );
    }
    // Walls still reflect: a particle aimed at the north wall stays in
    let mut particle = vec![tracker.release(9, [0.5, 0.5]).unwrap()];
    let (u, v) = (
        nodal(&mesh, &ops, |_, _| 0.0),
        nodal(&mesh, &ops, |_, _| 1.0),
    );
    let field = NodalVelocity2D::new(&u, &v, ops.n_nodes);
    tracker.step(&mut particle, &field, 0.0, 0.8);
    assert_eq!(particle[0].status(), ParticleStatus::Active);
    assert!((particle[0].position()[1] - 0.7).abs() < 1e-12);
}

#[test]
fn periodic_faces_carry_particles_across() {
    let mesh = Mesh2D::channel_periodic_x(0.0, 4.0, 0.0, 1.0, 8, 2);
    let ops = DGOperators2D::new(2);
    let (u, v) = (
        nodal(&mesh, &ops, |_, _| 1.0),
        nodal(&mesh, &ops, |_, _| 0.0),
    );
    let field = NodalVelocity2D::new(&u, &v, ops.n_nodes);
    let tracker = ParticleTracker2D::new(&mesh, &ops);
    let mut particles = release_all(&tracker, &[[0.3, 0.2], [3.9, 0.7]]);
    let dt = 0.5;
    for step in 0..21 {
        tracker.step(&mut particles, &field, step as f64 * dt, dt);
    }
    // 10.5 along the channel
    for (p, x0) in particles.iter().zip([0.3, 3.9]) {
        let expected = (x0 + 10.5_f64).rem_euclid(4.0);
        assert!(
            (p.position()[0] - expected).abs() < 1e-12,
            "{:?} against {expected}",
            p.position()
        );
        assert_consistent(&mesh, p);
    }
}

/// A particle carried up a beach strands where the water gets shallower
/// than the stranding depth, stays there, and floats again when the water
/// rises.
#[test]
fn particles_strand_on_drying_ground_and_refloat() {
    let mesh = Mesh2D::uniform_rectangle(0.0, 2.0, 0.0, 1.0, 8, 2);
    let ops = DGOperators2D::new(2);
    let state = |h: &dyn Fn(f64) -> f64, u: f64| {
        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let [x, _] = mesh.reference_to_physical(k, ops.nodes_r[i], ops.nodes_s[i]);
                q.set_state(k, i, SWEState2D::new(h(x), h(x) * u, 0.0));
            }
        }
        q
    };
    // Depth 1 − x (dry from x = 1), flowing shorewards at 1 m/s
    let ebb = state(&|x| (1.0 - x).max(0.0), 1.0);
    let flood = state(&|_| 1.0, 0.0);
    let tracker = ParticleTracker2D::new(&mesh, &ops).with_stranding_depth(0.1);
    let mut particles = vec![tracker.release(0, [0.2, 0.5]).unwrap()];
    let dt = 0.05;
    let field = SWEVelocity2D::steady(&ebb, 1e-6);
    for step in 0..40 {
        tracker.step(&mut particles, &field, step as f64 * dt, dt);
    }
    let p = &particles[0];
    assert_eq!(p.status(), ParticleStatus::Stranded);
    let x = p.position()[0];
    // The first step to reach water shallower than 0.1 m (x > 0.9)
    assert!(x > 0.9 - 1e-9 && x < 0.95 + 1e-9, "stranded at x = {x}");
    let field = SWEVelocity2D::steady(&flood, 1e-6);
    tracker.step(&mut particles, &field, 2.0, dt);
    assert_eq!(particles[0].status(), ParticleStatus::Active);
    assert_eq!(particles[0].position()[0], x);
}

/// Each particle has its own random stream: its path is the same tracked
/// alone or among others, and a different seed gives a different path.
#[test]
fn random_walk_is_reproducible_per_particle() {
    let mesh = distorted_mesh(0.0, 1.0, 0.0, 1.0, 5, 5);
    let ops = DGOperators2D::new(2);
    let zero = vec![0.0; mesh.n_elements * ops.n_nodes];
    let field = NodalVelocity2D::new(&zero, &zero, ops.n_nodes);
    let tracker = ParticleTracker2D::new(&mesh, &ops)
        .with_diffusivity(1e-3)
        .with_seed(5);
    let points = scattered(300, [0.0, 1.0, 0.0, 1.0]);
    let run = |tracker: &ParticleTracker2D, particles: &mut Vec<Particle2D>| {
        for step in 0..20 {
            tracker.step(particles, &field, step as f64, 1.0);
        }
    };
    let mut all = release_all(&tracker, &points);
    run(&tracker, &mut all);
    let mut alone = vec![tracker.release(137, points[137]).unwrap()];
    run(&tracker, &mut alone);
    assert_eq!(alone[0], all[137]);
    let reseeded = tracker.clone().with_seed(6);
    let mut other = vec![reseeded.release(137, points[137]).unwrap()];
    run(&reseeded, &mut other);
    assert_ne!(other[0].position(), all[137].position());
}

/// Uniform body force `(h·A cos ωt, 0)` on a periodic basin at rest: the
/// flow stays uniform with u = (A/ω) sin ωt, and a particle moves by
/// (A/ω²)(1 − cos ωt).
struct OscillatingForce {
    amplitude: f64,
    omega: f64,
}

impl SourceTerm2D for OscillatingForce {
    fn evaluate(&self, ctx: &SourceContext2D) -> SWEState2D {
        let force = self.amplitude * (self.omega * ctx.time).cos();
        SWEState2D::new(0.0, ctx.state.h * force, 0.0)
    }

    fn name(&self) -> &'static str {
        "oscillating_force"
    }
}

/// Tracking online, between the snapshots a running `Simulation` hands its
/// callback, follows the flow to the error of the linear time
/// interpolation: second order in the snapshot interval.
#[test]
fn online_tracking_between_snapshots_converges_at_second_order() {
    let mesh = Arc::new(Mesh2D::uniform_periodic(0.0, 1000.0, 0.0, 500.0, 8, 4));
    let ops = Arc::new(DGOperators2D::new(2));
    let geom = Arc::new(GeometricFactors2D::compute(&mesh, &ops));
    let (amplitude, omega) = (1e-3, std::f64::consts::TAU / 600.0);
    let depth = 10.0;
    let t_end = 0.35 * 600.0;
    let start = [[100.0, 50.0], [990.0, 250.0], [500.0, 499.0]];
    let error = |interval: f64| {
        let physics = PhysicsBuilder::swe_2d(
            mesh.clone(),
            ops.clone(),
            geom.clone(),
            ShallowWater2D::new(9.81),
            Reflective2D::new(),
        )
        .with_source(OscillatingForce { amplitude, omega })
        .build();
        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                q.set_state(k, i, SWEState2D::new(depth, 0.0, 0.0));
            }
        }
        let tracker = ParticleTracker2D::new(&mesh, &ops);
        let mut particles = release_all(&tracker, &start);
        let mut previous: Option<(f64, SWESolution2D)> = None;
        let mut worst: f64 = 0.0;
        let result = Simulation::new(physics, SSPRK3)
            .with_callback_interval(interval)
            .run_with_callback(&mut q, 0.0, t_end, |q, t| {
                if let Some((t0, q0)) = &previous {
                    let field = SWEVelocity2D::between(*t0, q0, t, q, 1e-6);
                    tracker.step(&mut particles, &field, *t0, t - t0);
                    let shift = amplitude / (omega * omega) * (1.0 - (omega * t).cos());
                    for (p, s) in particles.iter().zip(&start) {
                        let x = (s[0] + shift).rem_euclid(1000.0);
                        worst = worst
                            .max((p.position()[0] - x).abs())
                            .max((p.position()[1] - s[1]).abs());
                    }
                }
                previous = Some((t, q.clone()));
            });
        assert!(result.success);
        worst
    };
    let (coarse, fine) = (error(30.0), error(15.0));
    let rate = (coarse / fine).log2();
    println!("online: {coarse:.3e} → {fine:.3e} m, rate {rate:.2}");
    assert!(rate > 1.8, "rate {rate:.2}");
    // The displacement is 9.1 m; the interpolation error a few cm
    assert!(fine < 0.05, "error {fine:.3e} m");
}

/// The tracker's locator is the mesh's bucket search.
#[test]
fn release_outside_the_mesh_is_refused() {
    let mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 2, 2);
    let ops = DGOperators2D::new(1);
    let tracker = ParticleTracker2D::new(&mesh, &ops);
    assert!(tracker.release(0, [1.5, 0.5]).is_none());
    let p = tracker.release(1, [0.25, 0.75]).unwrap();
    let located = PointLocator2D::new(&mesh).locate([0.25, 0.75]).unwrap();
    assert_eq!(p.point(), located);
}

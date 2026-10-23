//! Gates for the random walk's depth drift `K∇h/h` (Dimou & Adams 1993,
//! TODO F.2): a tracer mixed through a water column of varying depth stays
//! well mixed.
//!
//! - 2D, depth-integrated: particles in proportion to the depth in a closed
//!   basin whose depth varies ninefold stay so, under a `K(x, y)` varying
//!   tenfold as well (drifts `∇K` and `K∇h/h`). A surface walk
//!   ([`WalkTracer::Surface`]) spreads them evenly over the area instead.
//! - 3D: particles uniform in volume over the same bed stay uniform, in the
//!   horizontal (in proportion to the depth along each σ-surface) and in σ;
//!   without the drift they leave the deep water.

use dg_rs::mesh::{Bathymetry2D, Mesh2D};
use dg_rs::operators::{DGOperators2D, GeometricFactors2D};
use dg_rs::particles::{
    DiffusivityInTime, HorizontalDiffusivity, Particle2D, Particle3D, ParticleTracker2D,
    ParticleTracker3D, ParticleVelocity3D, SWEVelocity2D, Solution3DVelocity, WalkTracer,
    WithHorizontalDiffusivity2D,
};
use dg_rs::solver::state::Solution3D;
use dg_rs::solver::{SWESolution2D, SWEState2D};
use dg_rs::types::ElementIndex;
use dg_rs::vertical::{SigmaGrid, UniformStretching};

const BINS: usize = 8;

/// The depth (m): 0.2 to 1.8 over the unit square.
fn depth(x: f64, y: f64) -> f64 {
    let pi = std::f64::consts::PI;
    1.0 + 0.8 * (pi * x).cos() * (0.5 * pi * (y + 0.3)).sin()
}

/// The unit square in 8 × 8 general quadrilaterals (interior vertices
/// displaced), at order 2.
fn basin() -> (Mesh2D, DGOperators2D) {
    let mut mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 8, 8);
    for v in &mut mesh.vertices {
        let [x, y] = *v;
        if x > 1e-9 && x < 1.0 - 1e-9 && y > 1e-9 && y < 1.0 - 1e-9 {
            v[0] += 0.025 * (7.0 * x + 3.0 * y).sin();
            v[1] += 0.025 * (5.0 * x - 4.0 * y).cos();
        }
    }
    (mesh, DGOperators2D::new(2))
}

/// Nodal values of `f(x, y)`, element by element.
fn nodal(mesh: &Mesh2D, ops: &DGOperators2D, f: impl Fn(f64, f64) -> f64) -> Vec<f64> {
    ElementIndex::iter(mesh.n_elements)
        .flat_map(|k| (0..ops.n_nodes).map(move |i| (k, i)))
        .map(|(k, i)| {
            let [x, y] = mesh.reference_to_physical(k, ops.nodes_r[i], ops.nodes_s[i]);
            f(x, y)
        })
        .collect()
}

/// `n` positions in proportion to the depth: a Kronecker sequence in three
/// dimensions, the third accepting a point with probability `h/h_max`.
fn in_proportion_to_depth(n: usize) -> Vec<[f64; 2]> {
    let mut points = Vec::with_capacity(n);
    let mut t = 0.0_f64;
    while points.len() < n {
        t += 1.0;
        let p = [(0.754_877_666_2 * t).fract(), (0.569_840_290_9 * t).fract()];
        if (0.438_744_337_3 * t).fract() * 1.8 < depth(p[0], p[1]) {
            points.push(p);
        }
    }
    points
}

/// χ² of `positions` over 8 × 8 bins of the unit square, against counts in
/// proportion to `weight` (its integral over each bin, by the midpoint rule
/// on 64 × 64 points per bin).
fn chi2(positions: impl Iterator<Item = [f64; 2]>, weight: impl Fn(f64, f64) -> f64) -> f64 {
    let mut counts = vec![0.0; BINS * BINS];
    let mut n = 0.0;
    for [x, y] in positions {
        let (i, j) = (
            ((x * BINS as f64) as usize).min(BINS - 1),
            ((y * BINS as f64) as usize).min(BINS - 1),
        );
        counts[j * BINS + i] += 1.0;
        n += 1.0;
    }
    let sub = 64;
    let h = 1.0 / (BINS * sub) as f64;
    let mut mass = vec![0.0; BINS * BINS];
    for a in 0..BINS * sub {
        for b in 0..BINS * sub {
            mass[(b / sub) * BINS + a / sub] += weight((a as f64 + 0.5) * h, (b as f64 + 0.5) * h);
        }
    }
    let total: f64 = mass.iter().sum();
    counts
        .iter()
        .zip(&mass)
        .map(|(c, m)| {
            let expected = n * m / total;
            (c - expected).powi(2) / expected
        })
        .sum()
}

#[test]
fn a_depth_integrated_tracer_stays_well_mixed_over_a_varying_bed() {
    let (mesh, ops) = basin();
    let geom = GeometricFactors2D::compute(&mesh, &ops);
    // Still water over the bed, and K from 1e-4 to 1.9e-3 m²/s
    let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
    for (n, h) in nodal(&mesh, &ops, depth).into_iter().enumerate() {
        q.set_state(
            ElementIndex::new(n / ops.n_nodes),
            n % ops.n_nodes,
            SWEState2D::new(h, 0.0, 0.0),
        );
    }
    let tau = std::f64::consts::TAU;
    let k = nodal(&mesh, &ops, |x, y| {
        1e-3 * (1.0 + 0.9 * (tau * x).sin() * (tau * y).sin())
    });
    let field = HorizontalDiffusivity::new(&mesh, &ops, &geom).from_nodal(k, Vec::new());
    let flow = WithHorizontalDiffusivity2D::new(
        SWEVelocity2D::steady(&q, 1e-6),
        DiffusivityInTime::steady(&field),
    );
    let run = |tracer: WalkTracer| {
        let tracker = ParticleTracker2D::new(&mesh, &ops)
            .with_walk_tracer(tracer)
            .with_seed(5);
        let mut particles: Vec<Particle2D> = in_proportion_to_depth(16_000)
            .into_iter()
            .enumerate()
            .map(|(i, p)| tracker.release(i as u64, p).unwrap())
            .collect();
        // 400 steps of 0.5 s: ≈ 6 diffusion times of a bin at the mean K
        let dt = 0.5;
        for step in 0..400 {
            tracker.step(&mut particles, &flow, step as f64 * dt, dt);
        }
        particles
    };
    let integrated = run(WalkTracer::DepthIntegrated);
    let surface = run(WalkTracer::Surface);
    let good = chi2(integrated.iter().map(Particle2D::position), depth);
    let bad = chi2(surface.iter().map(Particle2D::position), depth);
    let surface_even = chi2(surface.iter().map(Particle2D::position), |_, _| 1.0);
    // χ² with 63 degrees of freedom: mean 63, 99.9 % below 104
    println!(
        "χ² against h: {good:.1} depth-integrated, {bad:.1} surface walk \
         (the surface walk against an even spread: {surface_even:.1})"
    );
    assert!(good < 110.0, "depth-integrated: χ² = {good:.1}");
    assert!(bad > 500.0, "surface walk: χ² = {bad:.1}");
}

/// A 3D flow whose depth has no gradient: the walk without its depth drift.
struct WithoutDepthDrift<'a>(Solution3DVelocity<'a>);

impl ParticleVelocity3D for WithoutDepthDrift<'_> {
    fn velocity(&self, element: ElementIndex, weights: &[f64], sigma: f64, t: f64) -> [f64; 3] {
        self.0.velocity(element, weights, sigma, t)
    }

    fn depth(&self, element: ElementIndex, weights: &[f64], t: f64) -> f64 {
        self.0.depth(element, weights, t)
    }

    fn diffusivity(
        &self,
        element: ElementIndex,
        weights: &[f64],
        sigma: f64,
        t: f64,
    ) -> Option<(f64, f64)> {
        self.0.diffusivity(element, weights, sigma, t)
    }
}

/// χ² of the particles' σ over 10 bins against an even spread.
fn chi2_sigma(particles: &[Particle3D]) -> f64 {
    let bins = 10;
    let mut counts = vec![0.0; bins];
    for p in particles {
        counts[((-p.sigma() * bins as f64) as usize).min(bins - 1)] += 1.0;
    }
    let expected = particles.len() as f64 / bins as f64;
    counts
        .iter()
        .map(|c| (c - expected).powi(2) / expected)
        .sum()
}

#[test]
fn a_well_mixed_population_stays_well_mixed_over_a_varying_bed_in_3d() {
    let (mesh, ops) = basin();
    let geom = GeometricFactors2D::compute(&mesh, &ops);
    let nl = 8;
    let sigma = SigmaGrid::new(nl, UniformStretching);
    let bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, y| -depth(x, y));
    // Still water, a uniform vertical eddy diffusivity of 1e-4 m²/s
    let mut state = Solution3D::new(mesh.n_elements, ops.n_nodes, nl);
    state.eddy_diffusivity.fill(1e-4);
    let flow = Solution3DVelocity::steady(&state, &sigma, &bathymetry, 1e-3);
    let tracker = ParticleTracker3D::new(&mesh, &ops)
        .with_horizontal_diffusivity(1e-3)
        .with_vertical_random_walk()
        .with_seed(11);
    let n = 16_000;
    let start: Vec<Particle3D> = in_proportion_to_depth(n)
        .into_iter()
        .enumerate()
        .map(|(i, p)| {
            let s = -((0.414_213_562_4 * (i as f64 + 0.5)).fract());
            tracker.release(i as u64, p, s, 0.0).unwrap()
        })
        .collect();
    let dt = 0.5;
    let (mut with, mut without) = (start.clone(), start);
    for step in 0..400 {
        let t = step as f64 * dt;
        tracker.step(&mut with, &flow, t, dt);
        tracker.step(&mut without, &WithoutDepthDrift(flow), t, dt);
    }
    let good = chi2(with.iter().map(Particle3D::position), depth);
    let bad = chi2(without.iter().map(Particle3D::position), depth);
    let (good_sigma, bad_sigma) = (chi2_sigma(&with), chi2_sigma(&without));
    println!(
        "χ² against h: {good:.1} with the drift, {bad:.1} without; \
         over σ: {good_sigma:.1} and {bad_sigma:.1}"
    );
    assert!(good < 110.0, "with the drift: χ² = {good:.1}");
    assert!(bad > 500.0, "without the drift: χ² = {bad:.1}");
    // 9 degrees of freedom: 99.9 % below 27.9
    assert!(good_sigma < 30.0, "σ with the drift: χ² = {good_sigma:.1}");
}

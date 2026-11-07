//! Gates for the random walk in a horizontal diffusivity that varies in
//! space (`dgcom::particles::diffusivity`, TODO F.2).
//!
//! - Visser's well-mixed condition in 2D: a uniform distribution in a closed
//!   basin stays uniform under a `K(x, y)` varying tenfold, with the drift
//!   `∇K`; without it the particles collect where `K` is small.
//! - In 3D the field's `K` at the particle's σ-level (linear between the
//!   levels) adds to the tracker's constant: the spread is `2(K_field +
//!   K_tracker)t`.

use dgcom::mesh::Mesh2D;
use dgcom::operators::{DGOperators2D, GeometricFactors2D};
use dgcom::particles::{
    DiffusivityInTime, HorizontalDiffusivity, NodalVelocity2D, Particle2D, ParticleTracker2D,
    ParticleTracker3D, ParticleVelocity2D, ParticleVelocity3D, WithHorizontalDiffusivity2D,
    WithHorizontalDiffusivity3D,
};
use dgcom::types::ElementIndex;

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

/// A flow whose walk has the field's `K` but not its drift: the negative
/// control.
struct WithoutDrift<V>(V);

impl<V: ParticleVelocity2D> ParticleVelocity2D for WithoutDrift<V> {
    fn velocity(&self, element: ElementIndex, weights: &[f64], t: f64) -> [f64; 2] {
        self.0.velocity(element, weights, t)
    }

    fn horizontal_diffusivity(
        &self,
        element: ElementIndex,
        weights: &[f64],
        t: f64,
    ) -> Option<(f64, [f64; 2])> {
        self.0
            .horizontal_diffusivity(element, weights, t)
            .map(|(k, _)| (k, [0.0; 2]))
    }
}

/// χ² of the particles' positions over 8 × 8 bins of the unit square
/// against a uniform distribution.
fn chi2(particles: &[Particle2D]) -> f64 {
    let bins = 8;
    let mut counts = vec![0usize; bins * bins];
    for p in particles {
        let [x, y] = p.position();
        let (i, j) = (
            ((x * bins as f64) as usize).min(bins - 1),
            ((y * bins as f64) as usize).min(bins - 1),
        );
        counts[j * bins + i] += 1;
    }
    let expected = particles.len() as f64 / (bins * bins) as f64;
    counts
        .iter()
        .map(|&c| (c as f64 - expected).powi(2) / expected)
        .sum()
}

#[test]
fn variable_diffusivity_keeps_a_uniform_distribution_uniform() {
    let mut mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 8, 8);
    // Interior vertices displaced: general quadrilaterals
    for v in &mut mesh.vertices {
        let [x, y] = *v;
        if x > 1e-9 && x < 1.0 - 1e-9 && y > 1e-9 && y < 1.0 - 1e-9 {
            v[0] += 0.025 * (7.0 * x + 3.0 * y).sin();
            v[1] += 0.025 * (5.0 * x - 4.0 * y).cos();
        }
    }
    let ops = DGOperators2D::new(2);
    let geom = GeometricFactors2D::compute(&mesh, &ops);
    let tau = std::f64::consts::TAU;
    // K from 1e-4 to 1.9e-3 m²/s
    let k = nodal(&mesh, &ops, |x, y| {
        1e-3 * (1.0 + 0.9 * (tau * x).sin() * (tau * y).sin())
    });
    let field = HorizontalDiffusivity::new(&mesh, &ops, &geom).from_nodal(k, Vec::new());
    let zero = vec![0.0; mesh.n_elements * ops.n_nodes];
    let still = NodalVelocity2D::new(&zero, &zero, ops.n_nodes);
    let flow = WithHorizontalDiffusivity2D::new(still, DiffusivityInTime::steady(&field));
    let tracker = ParticleTracker2D::new(&mesh, &ops).with_seed(5);
    let n = 16_000;
    let start: Vec<Particle2D> = (0..n)
        .map(|i| {
            let t = i as f64 + 0.5;
            let p = [(0.754_877_666_2 * t).fract(), (0.569_840_290_9 * t).fract()];
            tracker.release(i as u64, p).unwrap()
        })
        .collect();
    // 400 steps of 0.5 s: ≈ 6 diffusion times of a bin (L²/K = 15 s at
    // the mean K)
    let (dt, steps) = (0.5, 400);
    let mut with = start.clone();
    let mut without = start;
    for step in 0..steps {
        let t = step as f64 * dt;
        tracker.step(&mut with, &flow, t, dt);
        tracker.step(&mut without, &WithoutDrift(flow), t, dt);
    }
    let (good, bad) = (chi2(&with), chi2(&without));
    // χ² with 63 degrees of freedom: mean 63, 99.9 % below 104
    println!("χ² over 64 bins: {good:.1} with the drift ∇K, {bad:.1} without");
    assert!(good < 110.0, "with the drift: χ² = {good:.1}");
    assert!(bad > 500.0, "without the drift: χ² = {bad:.1}");
}

/// Still water 10 m deep, without vertical motion or diffusivity.
#[derive(Clone, Copy)]
struct StillColumn;

impl ParticleVelocity3D for StillColumn {
    fn velocity(&self, _: ElementIndex, _: &[f64], _: f64, _: f64) -> [f64; 3] {
        [0.0; 3]
    }

    fn depth(&self, _: ElementIndex, _: &[f64], _: f64) -> f64 {
        10.0
    }
}

#[test]
fn diffusivity_field_adds_to_the_trackers_constant_in_3d() {
    let mesh = Mesh2D::uniform_rectangle(-200.0, 200.0, -200.0, 200.0, 8, 8);
    let ops = DGOperators2D::new(1);
    let geom = GeometricFactors2D::compute(&mesh, &ops);
    // K = 1 at σ = −0.75 and 3 at σ = −0.25: 2 m²/s at σ = −0.5
    let n_total = mesh.n_elements * ops.n_nodes;
    let mut k = vec![1.0; 2 * n_total];
    k[n_total..].fill(3.0);
    let field = HorizontalDiffusivity::new(&mesh, &ops, &geom).from_nodal(k, vec![-0.75, -0.25]);
    let flow = WithHorizontalDiffusivity3D::new(StillColumn, DiffusivityInTime::steady(&field));
    let tracker = ParticleTracker3D::new(&mesh, &ops)
        .with_horizontal_diffusivity(0.25)
        .with_seed(9);
    let n = 20_000;
    let mut particles: Vec<_> = (0..n)
        .map(|i| tracker.release(i, [3.0, -4.0], -0.5, 0.0).unwrap())
        .collect();
    let (dt, steps) = (2.0, 10);
    for step in 0..steps {
        tracker.step(&mut particles, &flow, step as f64 * dt, dt);
    }
    let expected = 2.0 * (2.0 + 0.25) * dt * steps as f64;
    for (axis, origin) in [3.0, -4.0].into_iter().enumerate() {
        let d: Vec<f64> = particles
            .iter()
            .map(|p| p.position()[axis] - origin)
            .collect();
        let mean = d.iter().sum::<f64>() / n as f64;
        let variance = d.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        println!("axis {axis}: variance {variance:.1} against {expected:.1}");
        // Standard error of the variance: √(2/n) = 1 %
        assert!((variance / expected - 1.0).abs() < 0.04);
        assert!(particles.iter().all(|p| p.sigma() == -0.5));
    }
}

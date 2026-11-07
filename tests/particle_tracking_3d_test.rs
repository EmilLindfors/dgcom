//! Gates for 3D particle tracking (`ParticleTracker3D`, TODO F.2 Stage B).
//!
//! - Visser's (1997) well-mixed test: under a diffusivity that varies with
//!   depth, a uniform population stays uniform with the drift `K'Δt`; without
//!   it, particles collect where `K` is small.
//! - Advection through a sheared flow with a σ-velocity is followed exactly
//!   (RK4 is exact for the quadratic-in-time trajectory).
//! - Sinking particles reach the bed when they should, and settle there or
//!   reflect.
//! - Swimming larvae (`SalmonLice`) under constant mixing reach the analytic
//!   steady profiles of the light threshold and of a halocline, and develop
//!   and die in degree-days from the model's temperature.

use dgcom::mesh::Mesh2D;
use dgcom::operators::DGOperators2D;
use dgcom::particles::{
    ConstantLight, LiceStage, Particle3D, ParticleBehaviour3D, ParticleStatus, ParticleTracker3D,
    ParticleVelocity3D, SalmonLice,
};
use dgcom::types::ElementIndex;

/// `K(z, D)` and `∂K/∂z` of a diffusivity profile.
type Profile = fn(f64, f64) -> (f64, f64);

/// One tracking step of a tracker through some field.
type StepFn<'a> = dyn Fn(&ParticleTracker3D, &mut [Particle3D], f64, f64) + 'a;

/// Temperature and salinity at `z` in a column of depth `D`.
type Tracers = fn(f64, f64) -> [f64; 2];

/// A horizontally uniform water column: depth `depth`, velocity
/// `u(σ) = u0 + u1 σ`, `v`, a constant σ-velocity, a diffusivity `K(z)`
/// with its derivative, and temperature and salinity.
struct Column {
    depth: f64,
    u: [f64; 2],
    v: f64,
    sigma_rate: f64,
    diffusivity: Option<Profile>,
    tracers: Option<Tracers>,
}

impl Column {
    fn still(depth: f64) -> Self {
        Self {
            depth,
            u: [0.0, 0.0],
            v: 0.0,
            sigma_rate: 0.0,
            diffusivity: None,
            tracers: None,
        }
    }
}

impl ParticleVelocity3D for Column {
    fn velocity(&self, _: ElementIndex, _: &[f64], sigma: f64, _: f64) -> [f64; 3] {
        [self.u[0] + self.u[1] * sigma, self.v, self.sigma_rate]
    }

    fn depth(&self, _: ElementIndex, _: &[f64], _: f64) -> f64 {
        self.depth
    }

    fn diffusivity(&self, _: ElementIndex, _: &[f64], sigma: f64, _: f64) -> Option<(f64, f64)> {
        self.diffusivity.map(|k| k(sigma * self.depth, self.depth))
    }

    fn tracers(&self, _: ElementIndex, _: &[f64], sigma: f64, _: f64) -> Option<[f64; 2]> {
        self.tracers.map(|f| f(sigma * self.depth, self.depth))
    }
}

/// Visser-type profile over depth `H` (ζ = −z the depth below the surface):
/// `K = 10⁻³ + 0.02 (ζ/H)(1 − ζ/H)²`, smallest at the surface and the bed,
/// largest at H/3, with `K' ≠ 0` at the surface.
fn visser_profile(z: f64, depth: f64) -> (f64, f64) {
    let x = -z / depth;
    let k = 1e-3 + 0.02 * x * (1.0 - x) * (1.0 - x);
    // dK/dz = −dK/dζ
    let dk_dz = -(0.02 / depth) * (1.0 - x) * (1.0 - 3.0 * x);
    (k, dk_dz)
}

/// The same `K` with its derivative dropped: the naive walk.
fn visser_profile_without_drift(z: f64, depth: f64) -> (f64, f64) {
    (visser_profile(z, depth).0, 0.0)
}

fn mesh() -> (Mesh2D, DGOperators2D) {
    (
        Mesh2D::uniform_periodic(0.0, 10_000.0, 0.0, 10_000.0, 4, 4),
        DGOperators2D::new(1),
    )
}

/// χ² of the particles' σ-levels against a uniform distribution, `bins`
/// bins.
fn chi_squared(particles: &[Particle3D], bins: usize) -> f64 {
    let mut counts = vec![0.0; bins];
    for p in particles {
        let b = ((-p.sigma() * bins as f64) as usize).min(bins - 1);
        counts[b] += 1.0;
    }
    let expected = particles.len() as f64 / bins as f64;
    counts
        .iter()
        .map(|c| (c - expected) * (c - expected) / expected)
        .sum()
}

/// Gate: Visser's well-mixed condition. 2000 particles spread evenly over
/// a 10 m column, 12 h (≈ 1.2 vertical mixing times H²/K̄) at 20 s steps
/// (`Δt·max|K''|` ≈ 0.02). With the drift the population stays uniform:
/// χ² over 20 bins below the 0.1 % critical value of 43.8 (19 degrees of
/// freedom). Without the drift it is far from uniform.
#[test]
fn a_well_mixed_population_stays_well_mixed() {
    let (mesh, ops) = mesh();
    let run = |profile: Profile| {
        let tracker = ParticleTracker3D::new(&mesh, &ops)
            .with_vertical_random_walk()
            .with_seed(7);
        let field = Column {
            diffusivity: Some(profile),
            ..Column::still(10.0)
        };
        let n = 2000;
        let mut particles: Vec<Particle3D> = (0..n)
            .map(|i| {
                let sigma = -(i as f64 + 0.5) / n as f64;
                tracker
                    .release(i as u64, [5_000.0, 5_000.0], sigma, 0.0)
                    .unwrap()
            })
            .collect();
        let dt = 20.0;
        for step in 0..(12 * 3600 / 20) {
            tracker.step(&mut particles, &field, step as f64 * dt, dt);
        }
        for p in &particles {
            assert_eq!(p.status(), ParticleStatus::Active);
            assert!((-1.0..=0.0).contains(&p.sigma()), "σ {}", p.sigma());
            // No horizontal motion in still water
            assert_eq!(p.position(), [5_000.0, 5_000.0]);
        }
        chi_squared(&particles, 20)
    };
    let with_drift = run(visser_profile);
    let without = run(visser_profile_without_drift);
    // Measured χ² 7.3 with the drift, 417 without
    assert!(
        with_drift < 43.8,
        "the well-mixed population lost uniformity: χ² = {with_drift:.1}"
    );
    assert!(
        without > 5.0 * 43.8,
        "the naive walk should collect particles where K is small: χ² = {without:.1}"
    );
}

/// Gate: a sheared flow `u = u₀ + u₁σ` with a constant σ-velocity `c`
/// carries a particle to `σ(t) = σ₀ + ct`,
/// `x(t) = x₀ + u₀t + u₁(σ₀t + ct²/2)`, `y(t) = y₀ + vt`, which RK4
/// follows exactly (across element faces and the periodic boundary).
#[test]
fn a_sheared_flow_with_a_sigma_velocity_is_followed_exactly() {
    let (mesh, ops) = mesh();
    let tracker = ParticleTracker3D::new(&mesh, &ops);
    let field = Column {
        depth: 30.0,
        u: [0.4, 0.3],
        v: -0.15,
        sigma_rate: 1e-4,
        diffusivity: None,
        tracers: None,
    };
    let (x0, y0, s0) = (1_234.0, 8_765.0, -0.9);
    let mut particles = vec![tracker.release(1, [x0, y0], s0, 0.0).unwrap()];
    let (dt, steps) = (100.0, 60);
    for n in 0..steps {
        tracker.step(&mut particles, &field, n as f64 * dt, dt);
    }
    let t = dt * steps as f64;
    let sigma = s0 + 1e-4 * t;
    let x = (x0 + 0.4 * t + 0.3 * (s0 * t + 0.5 * 1e-4 * t * t)).rem_euclid(10_000.0);
    let y = (y0 - 0.15 * t).rem_euclid(10_000.0);
    let p = &particles[0];
    let [px, py] = p.position();
    assert!(
        (p.sigma() - sigma).abs() < 1e-12,
        "σ {} vs {sigma}",
        p.sigma()
    );
    assert!((px - x).abs() < 1e-8, "x {px} vs {x}");
    assert!((py - y).abs() < 1e-8, "y {py} vs {y}");
}

/// Gate: particles sinking at 1 cm/s from 1 m below the surface of a 10 m
/// column reach the bed after 900 s. With bed settling they stay there
/// ([`ParticleStatus::Settled`], at σ = −1, no further motion); without,
/// they reflect off the bed (and, still sinking, bounce along it).
#[test]
fn sinking_particles_settle_on_the_bed() {
    let (mesh, ops) = mesh();
    let field = Column {
        u: [0.1, 0.0],
        ..Column::still(10.0)
    };
    let release = |tracker: &ParticleTracker3D| {
        vec![tracker.release(3, [5_000.0, 5_000.0], -0.1, -0.01).unwrap()]
    };
    let dt = 60.0;

    let settling = ParticleTracker3D::new(&mesh, &ops).with_bed_settling();
    let mut particles = release(&settling);
    let mut landed = None;
    for n in 0..30 {
        settling.step(&mut particles, &field, n as f64 * dt, dt);
        if landed.is_none() && particles[0].status() == ParticleStatus::Settled {
            landed = Some((n + 1) as f64 * dt);
        }
    }
    let p = &particles[0];
    let landed = landed.expect("the particle never settled");
    // σ reaches −1 exactly at 900 s; the step that would pass it settles
    assert!((900.0..=960.0).contains(&landed), "settled at {landed} s");
    assert_eq!(p.sigma(), -1.0);
    assert!((p.depth_below_surface(10.0) - 10.0).abs() < 1e-12);
    // It drifted with the current until it landed, then stopped
    let x = p.position()[0];
    assert!((x - (5_000.0 + 0.1 * landed)).abs() < 1e-6, "x {x}");

    let reflecting = ParticleTracker3D::new(&mesh, &ops);
    let mut particles = release(&reflecting);
    for n in 0..16 {
        reflecting.step(&mut particles, &field, n as f64 * dt, dt);
    }
    // 960 s: 0.6 m below the bed, reflected to 0.6 m above it
    let p = &particles[0];
    assert_eq!(p.status(), ParticleStatus::Active);
    assert!((p.sigma() + 0.94).abs() < 1e-12, "σ {}", p.sigma());
    for n in 16..40 {
        reflecting.step(&mut particles, &field, n as f64 * dt, dt);
        assert!((-1.0..=-0.94 + 1e-12).contains(&particles[0].sigma()));
    }
}

/// Gate: the well-mixed condition with the model's own fields. A
/// `Solution3D` at rest on 20 σ-levels holds the parabolic diffusivity of
/// a log-layer channel, `K = κu*|z|(1 − |z|/H) + K₀`, at its w-points
/// (`Solution3D::eddy_diffusivity`), sampled through
/// `Solution3DVelocity`: piecewise linear in z, so the drift `K'` jumps at
/// every w-point. A uniform population stays uniform; with the same field
/// stripped of its drift it does not.
#[test]
fn the_model_diffusivity_keeps_a_well_mixed_population_well_mixed() {
    use dgcom::mesh::Bathymetry2D;
    use dgcom::particles::Solution3DVelocity;
    use dgcom::solver::state::Solution3D;
    use dgcom::vertical::{SigmaGrid, UniformStretching};

    let (mesh, ops) = mesh();
    let (depth, n_levels, u_star) = (10.0, 20, 0.01);
    let sigma = SigmaGrid::new(n_levels, UniformStretching);
    let bathymetry = Bathymetry2D::constant(mesh.n_elements, ops.n_nodes, -depth);
    let mut state = Solution3D::new(mesh.n_elements, ops.n_nodes, n_levels);
    for column in state.eddy_diffusivity.chunks_exact_mut(n_levels + 1) {
        for (k, &s) in column.iter_mut().zip(sigma.sigma_w()) {
            let z = -s * depth;
            *k = 0.41 * u_star * z * (1.0 - z / depth) + 1e-5;
        }
    }
    let field = Solution3DVelocity::steady(&state, &sigma, &bathymetry, 0.01);

    /// `field` with the drift dropped: the naive walk
    struct WithoutDrift<'a>(Solution3DVelocity<'a>);
    impl ParticleVelocity3D for WithoutDrift<'_> {
        fn velocity(&self, k: ElementIndex, w: &[f64], s: f64, t: f64) -> [f64; 3] {
            self.0.velocity(k, w, s, t)
        }
        fn depth(&self, k: ElementIndex, w: &[f64], t: f64) -> f64 {
            self.0.depth(k, w, t)
        }
        fn diffusivity(&self, k: ElementIndex, w: &[f64], s: f64, t: f64) -> Option<(f64, f64)> {
            self.0.diffusivity(k, w, s, t).map(|(k, _)| (k, 0.0))
        }
    }

    let run = |field: &StepFn<'_>| {
        let tracker = ParticleTracker3D::new(&mesh, &ops)
            .with_vertical_random_walk()
            .with_seed(11);
        let n = 2000;
        let mut particles: Vec<Particle3D> = (0..n)
            .map(|i| {
                let s = -(i as f64 + 0.5) / n as f64;
                tracker
                    .release(i as u64, [3_000.0, 7_000.0], s, 0.0)
                    .unwrap()
            })
            .collect();
        // Mixing time H²/K̄ ≈ 1.5e4 s (K̄ = κu*H/6); 6 h at 5 s steps
        let dt = 5.0;
        for step in 0..(6 * 3600 / 5) {
            field(&tracker, &mut particles, step as f64 * dt, dt);
        }
        chi_squared(&particles, 20)
    };
    let with_drift = run(&|tracker, particles, t, dt| tracker.step(particles, &field, t, dt));
    let naive = WithoutDrift(field);
    let without = run(&|tracker, particles, t, dt| tracker.step(particles, &naive, t, dt));
    // Measured χ² 5.6 with the drift, 5435 without
    assert!(
        with_drift < 43.8,
        "the well-mixed population lost uniformity: χ² = {with_drift:.1}"
    );
    assert!(
        without > 5.0 * 43.8,
        "the naive walk should collect particles where K is small: χ² = {without:.1}"
    );
}

/// Steady state of larvae swimming under a constant diffusivity in a still
/// 10 m column, from a uniform start: 20 000 particles, 20 bins of 0.5 m.
/// Returns the χ² of the final σ-levels against the density `expected`
/// (unnormalised, of the depth ζ below the surface) and against a uniform
/// distribution.
fn swimming_steady_state(
    field: &Column,
    behaviour: &impl ParticleBehaviour3D,
    expected: impl Fn(f64) -> f64,
) -> (f64, f64) {
    let (mesh, ops) = mesh();
    let tracker = ParticleTracker3D::new(&mesh, &ops)
        .with_vertical_random_walk()
        .with_seed(23);
    let n = 20_000;
    let mut particles: Vec<Particle3D> = (0..n)
        .map(|i| {
            let s = -(i as f64 + 0.5) / n as f64;
            tracker
                .release(i as u64, [5_000.0, 5_000.0], s, 0.0)
                .unwrap()
        })
        .collect();
    // The slowest diffusive mode decays at Kπ²/H² ≈ 1/2000 s: 10 e-folds.
    // The steps resolve K/w² = 800 s and the profile's scale K/w = 2 m
    // (the walk's step is 0.32 m, the swim's 0.025 m). The swim is chosen
    // at the start of the step, so a cue's jump is smeared over a walk
    // step: at 20 s the light gate's χ² rises to 41, at 5 s it is 13
    let dt = 10.0;
    for step in 0..2000 {
        tracker.step_with(&mut particles, field, behaviour, step as f64 * dt, dt);
    }
    let bins = 20;
    let mut counts = vec![0.0; bins];
    for p in &particles {
        assert_eq!(p.status(), ParticleStatus::Active);
        let b = ((-p.sigma() * bins as f64) as usize).min(bins - 1);
        counts[b] += 1.0;
    }
    // Expected bin contents: the density integrated over each bin (Simpson
    // on 16 panels)
    let width = field.depth / bins as f64;
    let integral = |a: f64| {
        let m = 16;
        let h = width / m as f64;
        let sum: f64 = (0..=m)
            .map(|j| {
                let c = if j == 0 || j == m {
                    1.0
                } else if j % 2 == 1 {
                    4.0
                } else {
                    2.0
                };
                c * expected(a + j as f64 * h)
            })
            .sum();
        sum * h / 3.0
    };
    let mass: Vec<f64> = (0..bins).map(|b| integral(b as f64 * width)).collect();
    let total: f64 = mass.iter().sum();
    let chi2 = |expect: &dyn Fn(usize) -> f64| -> f64 {
        (0..bins)
            .map(|b| (counts[b] - expect(b)).powi(2) / expect(b))
            .sum()
    };
    let against_theory = chi2(&|b| n as f64 * mass[b] / total);
    let against_uniform = chi2(&|_| n as f64 / bins as f64);
    (against_theory, against_uniform)
}

/// Gate: light-seeking larvae. With the surface light set so that a
/// nauplius's threshold (0.39 µmol photons m⁻² s⁻¹, Johnsen et al. 2014)
/// lies at ζ* = 5 m, larvae swim up at `w` above it and drift passively
/// below. With a constant `K` the zero-flux steady state is
/// `c ∝ e^{−wζ/K}` above ζ* and constant below. χ² over 20 bins against
/// the 0.1 % critical value of 43.8 (19 degrees of freedom): measured 23.4
/// (16.6–23.9 over other seeds), 20 400 against a uniform column; with the
/// swim 5 % slower or faster than the theory's, 64 and 56.
#[test]
fn light_seeking_larvae_reach_the_analytic_profile() {
    let (k, w, threshold_depth) = (5e-3, 2.5e-3, 5.0);
    let surface = 0.39 * (0.2f64 * threshold_depth).exp();
    let mut lice = SalmonLice::johnsen_2014(ConstantLight(surface));
    lice.swimming_speed = w;
    let field = Column {
        diffusivity: Some(|_, _| (5e-3, 0.0)),
        ..Column::still(10.0)
    };
    let (theory, uniform) = swimming_steady_state(&field, &lice, |zeta| {
        (-w * zeta.min(threshold_depth) / k).exp()
    });
    assert!(theory < 43.8, "χ² {theory:.1} against the analytic profile");
    assert!(uniform > 20.0 * 43.8, "χ² {uniform:.1} against uniform");
}

/// Gate: a halocline traps the larvae. Over fresh water (S = 15) above
/// ζ_h = 3 m and sea water (33) below, with light everywhere, larvae swim
/// down above the halocline (salinity below 20 wins over light) and up
/// below it. The steady state under a constant `K` is the Laplace profile
/// `c ∝ e^{−w|ζ − ζ_h|/K}`. Measured χ² 11.4 (16–25 over other seeds),
/// 11 600 against uniform; 41 with the swim 5 % slow.
#[test]
fn a_halocline_traps_swimming_larvae() {
    let (k, w, halocline) = (5e-3, 2.5e-3, 3.0);
    let mut lice = SalmonLice::johnsen_2014(ConstantLight(1000.0));
    lice.swimming_speed = w;
    let field = Column {
        diffusivity: Some(|_, _| (5e-3, 0.0)),
        tracers: Some(|z, _| [10.0, if -z < 3.0 { 15.0 } else { 33.0 }]),
        ..Column::still(10.0)
    };
    let (theory, uniform) = swimming_steady_state(&field, &lice, |zeta| {
        (-w * (zeta - halocline).abs() / k).exp()
    });
    assert!(theory < 43.8, "χ² {theory:.1} against the analytic profile");
    assert!(uniform > 20.0 * 43.8, "χ² {uniform:.1} against uniform");
}

/// Gate: development from the model's temperature. A `Solution3D` at rest
/// holds 12 °C over 8 °C (the top and bottom two of 4 σ-levels), sampled at
/// the particles through `Solution3DVelocity`. Larvae at the top and bottom
/// (no light, no mixing) accrue `T` degree-days per day, become copepodids
/// at 40 and die at 170 (LADiM), on the step that reaches them. A field
/// without temperature does not develop them.
#[test]
fn larvae_develop_and_die_in_degree_days() {
    use dgcom::mesh::Bathymetry2D;
    use dgcom::particles::Solution3DVelocity;
    use dgcom::solver::state::Solution3D;
    use dgcom::vertical::{SigmaGrid, UniformStretching};

    let (mesh, ops) = mesh();
    let n_levels = 4;
    let sigma = SigmaGrid::new(n_levels, UniformStretching);
    let bathymetry = Bathymetry2D::constant(mesh.n_elements, ops.n_nodes, -20.0);
    let mut state = Solution3D::new(mesh.n_elements, ops.n_nodes, n_levels);
    // Levels bottom to surface: 8, 8, 12, 12 °C
    for (i, t) in state.temp.iter_mut().enumerate() {
        *t = if i % n_levels < 2 { 8.0 } else { 12.0 };
    }
    state.salt.fill(34.0);
    let field = Solution3DVelocity::steady(&state, &sigma, &bathymetry, 0.01);
    let lice = SalmonLice::ladim(ConstantLight(0.0));
    let tracker = ParticleTracker3D::new(&mesh, &ops);
    // Beyond the top and bottom layer centres, where the field is held
    let mut particles: Vec<Particle3D> = [-0.05, -0.95]
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            tracker
                .release(i as u64, [2_000.0, 6_000.0], s, 0.0)
                .unwrap()
        })
        .collect();
    let dt = 3600.0;
    let mut copepodid_at = [None; 2];
    let mut died_at = [None; 2];
    for step in 0..(24 * 22) {
        tracker.step_with(&mut particles, &field, &lice, step as f64 * dt, dt);
        let hours = (step + 1) as f64;
        for (j, p) in particles.iter().enumerate() {
            if copepodid_at[j].is_none() && lice.stage(p.development()) == LiceStage::Copepodid {
                copepodid_at[j] = Some(hours);
            }
            if died_at[j].is_none() && p.status() == ParticleStatus::Dead {
                died_at[j] = Some(hours);
            }
            if p.status() == ParticleStatus::Active {
                let temp = if j == 0 { 12.0 } else { 8.0 };
                assert!((p.development() - temp * hours / 24.0).abs() < 1e-9);
                assert_eq!(p.swimming_speed(), 0.0);
            }
        }
    }
    // 40 degree-days: 80 h at 12 °C, 120 h at 8 °C; 170: 340 h, 510 h
    assert_eq!(copepodid_at, [Some(80.0), Some(120.0)]);
    assert_eq!(died_at, [Some(340.0), Some(510.0)]);
    assert_eq!(particles[0].age(), 340.0 * 3600.0);

    // The analytic column has no temperature: no development
    let column = Column::still(20.0);
    let mut larva = vec![tracker.release(9, [2_000.0, 6_000.0], -0.5, 0.0).unwrap()];
    for step in 0..48 {
        tracker.step_with(&mut larva, &column, &lice, step as f64 * dt, dt);
    }
    assert_eq!(larva[0].development(), 0.0);
    assert_eq!(larva[0].status(), ParticleStatus::Active);
}

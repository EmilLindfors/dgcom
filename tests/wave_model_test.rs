//! Gates of the spectral wave model (`dg_rs::waves`, TODO F.4): the DG
//! propagation converges at its order, the propagation with refraction keeps the
//! total action, and the steady states of shoaling, refraction on a sloping shelf
//! (Snell's law) and a following current (conservation of the absolute frequency)
//! are the analytic ones (stage 1). With the full source terms (stage 2: the DIA,
//! the diagnostic tail and the limiter), a wind sea grows with fetch as Kahma &
//! Calkoen (1992) observed, and with duration towards Pierson–Moskowitz.

use std::f64::consts::{PI, TAU};
use std::sync::Arc;

use dg_rs::mesh::{Bathymetry2D, BoundaryTag, Mesh2D};
use dg_rs::operators::{DGOperators2D, GeometricFactors2D};
use dg_rs::types::ElementIndex;
use dg_rs::waves::{
    DEFAULT_RATE_LIMITER, GrowthLimiter, SourceIntegration, SourceTerms, SpectralAdvection,
    SpectralGrid, WaveModel2D, WaveSolution, WaveWorkspace, Wind, group_velocity, wavenumber,
};

const G: f64 = 9.81;

/// A model on `mesh` at order `order` over the bed `bed(x, y)` (negative under water).
fn model(
    mesh: Mesh2D,
    order: usize,
    bed: impl Fn(f64, f64) -> f64,
    grid: SpectralGrid,
) -> WaveModel2D {
    let ops = DGOperators2D::new(order);
    let geom = GeometricFactors2D::compute(&mesh, &ops);
    let bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, bed);
    WaveModel2D::new(
        Arc::new(mesh),
        Arc::new(ops),
        Arc::new(geom),
        &bathymetry,
        grid,
        G,
    )
}

/// Mesh coordinates of every node.
fn nodes(model: &WaveModel2D) -> Vec<[f64; 2]> {
    let (mesh, ops) = (&model.mesh, &model.ops);
    ElementIndex::iter(mesh.n_elements)
        .flat_map(|k| {
            (0..ops.n_nodes)
                .map(move |i| mesh.reference_to_physical(k, ops.nodes_r[i], ops.nodes_s[i]))
        })
        .collect()
}

/// Run from `t = 0` to `t_end` at Courant number `cfl`, landing on `t_end`.
fn run(model: &WaveModel2D, n: &mut WaveSolution, t_end: f64, cfl: f64) {
    let mut ws = WaveWorkspace::default();
    let dt_max = model.compute_dt(cfl);
    let steps = (t_end / dt_max).ceil() as usize;
    let dt = t_end / steps as f64;
    for _ in 0..steps {
        model.step(n, dt, &mut ws);
    }
}

/// L2 error of component `c` against `exact(x, y)`, by the GLL quadrature.
fn l2_error(
    model: &WaveModel2D,
    n: &WaveSolution,
    c: usize,
    exact: impl Fn(f64, f64) -> f64,
) -> f64 {
    let (ops, geom) = (&model.ops, &model.geom);
    let xy = nodes(model);
    let field: Vec<f64> = n.component(c).collect();
    let mut sum = 0.0;
    for k in 0..model.mesh.n_elements {
        for a in 0..ops.n_nodes {
            let p = k * ops.n_nodes + a;
            let e = field[p] - exact(xy[p][0], xy[p][1]);
            sum += ops.weights[a] * geom.jacobian(k, a) * e * e;
        }
    }
    sum.sqrt()
}

/// One component of a uniform deep sea (1000 m) carries a smooth periodic field
/// along x and along the diagonal at its group velocity; the L2 error after a
/// quarter period falls at the DG order + 1.
#[test]
fn propagation_converges_at_the_dg_order() {
    const L: f64 = 1000.0;
    for (j, label) in [(0, "along x"), (1, "diagonal")] {
        let mut errors = Vec::new();
        for n_el in [4, 8, 16] {
            let grid = SpectralGrid::new(0.08, 0.2, 2, 8);
            let m = model(
                Mesh2D::uniform_periodic(0.0, L, 0.0, L, n_el, n_el),
                2,
                |_, _| -1000.0,
                grid,
            );
            let sigma = m.grid.sigma[0];
            let cg = group_velocity(sigma, wavenumber(sigma, 1000.0, G), 1000.0);
            let theta = m.grid.theta[j];
            // A field periodic on the square, constant across the direction of travel
            let (a, b) = if j == 0 { (1.0, 0.0) } else { (1.0, 1.0) };
            let speed = cg * (a * theta.cos() + b * theta.sin());
            let f =
                |x: f64, y: f64, t: f64| 1.0 + 0.5 * (TAU * (a * x + b * y - speed * t) / L).sin();
            let c = m.grid.component(0, j);
            let mut n = m.zero_state();
            let xy = nodes(&m);
            for (p, [x, y]) in xy.iter().enumerate() {
                n.spectrum_mut(p)[c] = f(*x, *y, 0.0);
            }
            let t_end = 0.25 * L / cg;
            run(&m, &mut n, t_end, 0.3);
            errors.push(l2_error(&m, &n, c, |x, y| f(x, y, t_end)));
        }
        let rates: Vec<f64> = errors.windows(2).map(|e| (e[0] / e[1]).log2()).collect();
        println!("{label}: errors {errors:?}, rates {rates:?}");
        assert!(rates[1] > 2.7, "{label}: P2 converges at {rates:?}");
    }
}

/// Refraction over a periodic shoal moves energy between directions and places,
/// but the total action `Σ Δσ Δθ ∫ N dA` stays to round-off.
#[test]
fn refraction_keeps_the_total_action() {
    const L: f64 = 2000.0;
    let grid = SpectralGrid::new(0.06, 0.4, 10, 24);
    let m = model(
        Mesh2D::uniform_periodic(0.0, L, 0.0, L, 6, 6),
        2,
        |x, y| -(10.0 + 5.0 * (TAU * x / L).sin() * (TAU * y / L).cos()),
        grid,
    );
    let e = m.grid.jonswap(1.0, 6.0, 3.3, PI / 6.0, 2.0);
    let mut n = m.uniform_state(&e);
    let initial = n.clone();
    let total = m.total_action(&n);
    run(&m, &mut n, 300.0, 0.5);
    let change = (m.total_action(&n) / total - 1.0).abs();
    let moved = n
        .data
        .iter()
        .zip(&initial.data)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    println!("relative change of the total action {change:.2e}, largest change of N {moved:.3e}");
    assert!(moved > 1e-3 * initial.data.iter().fold(0.0f64, |a, &b| a.max(b)));
    assert!(change < 1e-12, "total action changed by {change:e}");
}

/// Normal incidence on a slope from 20 to 3 m: in the steady state the action
/// flux `c_g N` is the same everywhere (Green's law for a spectrum without
/// refraction or sources).
#[test]
fn shoaling_keeps_the_action_flux() {
    const L: f64 = 2000.0;
    let depth = |x: f64| 20.0 - 17.0 * x / L;
    // Normal incidence turns no wave out of its bin, so the fewest bins do
    let grid = SpectralGrid::new(0.1, 0.15, 3, 4);
    let i = 1;
    let mut e = vec![0.0; grid.n_components()];
    e[grid.component(i, 0)] = 1.0;
    let m = model(
        Mesh2D::uniform_rectangle_with_sides(
            0.0,
            L,
            0.0,
            200.0,
            40,
            1,
            [
                BoundaryTag::Wall,
                BoundaryTag::Open,
                BoundaryTag::Wall,
                BoundaryTag::Open,
            ],
        ),
        2,
        |x, _| -depth(x),
        grid,
    )
    .with_boundary_spectrum(&e);
    let sigma = m.grid.sigma[i];
    let cg_at = |d: f64| group_velocity(sigma, wavenumber(sigma, d, G), d);
    let mut n = m.zero_state();
    run(&m, &mut n, 4.0 * L / cg_at(3.0), 0.5);
    let (c, xy) = (m.grid.component(i, 0), nodes(&m));
    let flux0 = cg_at(depth(0.0)) * e[c] / sigma;
    let worst = (0..m.n_points())
        .map(|p| (cg_at(depth(xy[p][0])) * n.spectrum(p)[c] / flux0 - 1.0).abs())
        .fold(0.0, f64::max);
    println!("largest relative departure of c_g N from the boundary's: {worst:.2e}");
    assert!(worst < 2e-3, "c_g N departs by {worst:e}");
}

/// Oblique incidence (30° from the shore normal) on a shelf from 20 to 4 m deep:
/// the steady mean direction follows Snell's law, `sin α / c` constant, to the
/// first order in the direction bins. The incoming sea is one direction bin, a
/// spike the bins do not resolve: it leaves its bin through a face whose
/// turning rate is the face's, O(Δθ) off the bin's, and a limited (TVD)
/// reconstruction is first order at the extremum, so MUSCL does not help here
/// (2.13 / 1.04 / 0.52° against upwind's 2.06 / 1.03 / 0.51°). A resolved
/// spread is second order: [`a_spread_sea_refracts_to_the_exact_steady_state`].
#[test]
fn refraction_follows_snells_law() {
    const LX: f64 = 400.0;
    const LY: f64 = 2000.0;
    let depth = |y: f64| 20.0 - 16.0 * y / LY;
    let mut errors = Vec::new();
    for n_dir in [36, 72, 144] {
        // Uniform along x: one element across; 20 along the shelf (4 × 40
        // gives the same departures to 0.002°, at 30× the cost)
        let mut mesh = Mesh2D::channel_periodic_x(0.0, LX, 0.0, LY, 1, 20);
        for edge in mesh.edges.iter_mut().filter(|e| e.right.is_none()) {
            edge.boundary_tag = Some(BoundaryTag::Open);
        }
        let grid = SpectralGrid::new(0.12, 0.15, 2, n_dir);
        // θ = 60°: 30° off +y, the shore normal
        let (i, j0) = (0, n_dir / 6);
        let mut e = vec![0.0; grid.n_components()];
        e[grid.component(i, j0)] = 1.0;
        let m = model(mesh, 2, |_, y| -depth(y), grid).with_boundary_spectrum(&e);
        let sigma = m.grid.sigma[i];
        let celerity = |d: f64| sigma / wavenumber(sigma, d, G);
        let alpha0 = 30.0f64.to_radians();
        let mut n = m.zero_state();
        let cg_min = group_velocity(sigma, wavenumber(sigma, 4.0, G), 4.0);
        run(&m, &mut n, 3.0 * LY / (cg_min * alpha0.cos()), 0.5);
        let (xy, params) = (nodes(&m), m.parameters(&n));
        let mut worst: f64 = 0.0;
        for p in (0..m.n_points()).filter(|&p| (200.0..1800.0).contains(&xy[p][1])) {
            let d = depth(xy[p][1]);
            let alpha = (alpha0.sin() * celerity(d) / celerity(depth(0.0))).asin();
            let expected = 90.0 - alpha.to_degrees();
            worst = worst.max((params[p].direction.to_degrees() - expected).abs());
        }
        errors.push(worst);
    }
    let rates: Vec<f64> = errors.windows(2).map(|e| (e[0] / e[1]).log2()).collect();
    println!(
        "largest departure of the mean direction from Snell's law: {errors:?}°, rates {rates:?}"
    );
    assert!(
        rates.iter().all(|&r| r > 0.8),
        "first order in Δθ: {rates:?}"
    );
    assert!(errors[1] < 1.5, "{}° off Snell with 5° bins", errors[1]);
}

/// A spread sea (cos⁸ about 20° off the shore normal) refracting on the shelf
/// of [`refraction_follows_snells_law`]. In the steady state of a shelf uniform
/// along the shore, `k cos θ` is constant along a ray (Snell) and so is the
/// action density in wavenumber space, `N c_g/k` (Liouville), so the exact
/// directional spectrum at depth d is `N₀(θ₀) (k c_g₀)/(k₀ c_g)` with
/// `k₀ cos θ₀ = k cos θ`. The mean direction converges to it at second order
/// in the direction bins with MUSCL (0.37 / 0.072 / 0.012° at 10 / 5 / 2.5°,
/// rates 2.4 and 2.5) and at first order with upwind (1.38 / 0.70 / 0.35°).
#[test]
fn a_spread_sea_refracts_to_the_exact_steady_state() {
    const LX: f64 = 400.0;
    const LY: f64 = 2000.0;
    let depth = |y: f64| 20.0 - 16.0 * y / LY;
    let (mean, m) = (70f64.to_radians(), 8.0);
    let incoming = |theta: f64| {
        let c = (theta - mean).cos();
        if c > 0.0 { c.powf(m) } else { 0.0 }
    };
    let errors = |scheme: SpectralAdvection| -> Vec<f64> {
        let mut errors = Vec::new();
        for n_dir in [36, 72, 144] {
            let mut mesh = Mesh2D::channel_periodic_x(0.0, LX, 0.0, LY, 1, 20);
            for edge in mesh.edges.iter_mut().filter(|e| e.right.is_none()) {
                edge.boundary_tag = Some(BoundaryTag::Open);
            }
            let grid = SpectralGrid::new(0.12, 0.15, 2, n_dir);
            let mut e = vec![0.0; grid.n_components()];
            for j in 0..n_dir {
                e[grid.component(0, j)] = incoming(grid.theta[j]);
            }
            let m = model(mesh, 2, |_, y| -depth(y), grid)
                .with_boundary_spectrum(&e)
                .with_spectral_advection(scheme);
            let sigma = m.grid.sigma[0];
            let (k0, cg0) = {
                let k = wavenumber(sigma, depth(0.0), G);
                (k, group_velocity(sigma, k, depth(0.0)))
            };
            let mut n = m.zero_state();
            // Steady for every direction 20° or more off the shore (those below
            // carry 1e-3 of the energy)
            let cg_min = group_velocity(sigma, wavenumber(sigma, 4.0, G), 4.0);
            run(
                &m,
                &mut n,
                3.0 * LY / (cg_min * 20f64.to_radians().sin()),
                0.5,
            );
            let (xy, params) = (nodes(&m), m.parameters(&n));
            let mut worst: f64 = 0.0;
            for p in (0..m.n_points()).filter(|&p| (200.0..1800.0).contains(&xy[p][1])) {
                let d = depth(xy[p][1]);
                let k = wavenumber(sigma, d, G);
                let cg = group_velocity(sigma, k, d);
                // The exact mean direction, by a fine midpoint rule over (0, π)
                let (mut a, mut b) = (0.0, 0.0);
                let fine = 20_000;
                for q in 0..fine {
                    let theta = PI * (q as f64 + 0.5) / fine as f64;
                    let c0 = k / k0 * theta.cos();
                    if c0.abs() < 1.0 {
                        let w = incoming(c0.acos()) * k * cg0 / (k0 * cg);
                        a += w * theta.cos();
                        b += w * theta.sin();
                    }
                }
                let exact = b.atan2(a);
                worst = worst.max((params[p].direction - exact).to_degrees().abs());
            }
            errors.push(worst);
        }
        errors
    };
    let rates = |e: &[f64]| -> Vec<f64> { e.windows(2).map(|w| (w[0] / w[1]).log2()).collect() };
    let (muscl, upwind) = (
        errors(SpectralAdvection::VanLeer),
        errors(SpectralAdvection::Upwind),
    );
    println!("mean direction off the exact steady state at 10/5/2.5° bins:");
    println!("  MUSCL {muscl:?}°, rates {:?}", rates(&muscl));
    println!("  upwind {upwind:?}°, rates {:?}", rates(&upwind));
    assert!(
        rates(&muscl).iter().all(|&r| r > 1.7),
        "MUSCL: {:?}",
        rates(&muscl)
    );
    assert!(
        rates(&upwind).iter().all(|&r| r > 0.8),
        "upwind: {:?}",
        rates(&upwind)
    );
    assert!(
        muscl[1] < 0.25 * upwind[1],
        "MUSCL {muscl:?} against upwind {upwind:?}"
    );
}

/// A following current accelerating from 0 to 0.6 m/s over deep water stretches
/// the waves: the absolute frequency `σ + k U` is conserved, so the intrinsic
/// frequency falls as `σ + σ² U/g = σ_0`. One frequency bin comes in, a spike,
/// so the schemes are first order here (0.34 % upwind, 0.53 % MUSCL with 8
/// direction bins; 0.54 % MUSCL with the test's 4, whose wider bins the current
/// turns a little less energy across): see
/// [`a_spread_sea_on_a_current_has_the_exact_doppler_shift`] for a resolved
/// spectrum.
#[test]
fn a_following_current_shifts_the_frequency_doppler() {
    const L: f64 = 2000.0;
    // One direction bin carries the sea and no current gradient turns it:
    // the fewest bins do
    let grid = SpectralGrid::new(0.15, 0.35, 41, 4);
    let i0 = (0..grid.n_freq())
        .min_by(|&a, &b| {
            let d = |i: usize| (grid.sigma[i] / TAU - 0.3).abs();
            d(a).total_cmp(&d(b))
        })
        .unwrap();
    let mut e = vec![0.0; grid.n_components()];
    e[grid.component(i0, 0)] = 1.0;
    let mut m = model(
        Mesh2D::uniform_rectangle_with_sides(
            0.0,
            L,
            0.0,
            100.0,
            40,
            1,
            [
                BoundaryTag::Wall,
                BoundaryTag::Open,
                BoundaryTag::Wall,
                BoundaryTag::Open,
            ],
        ),
        2,
        |_, _| -1000.0,
        grid,
    )
    .with_boundary_spectrum(&e);
    let xy = nodes(&m);
    let current = |x: f64| 0.6 * x / L;
    let u: Vec<f64> = xy.iter().map(|p| current(p[0])).collect();
    m.set_currents(&u, &vec![0.0; u.len()]);
    let sigma0 = m.grid.sigma[i0];
    let mut n = m.zero_state();
    run(&m, &mut n, 4.0 * L / (G / (2.0 * sigma0)), 0.5);
    let (nd, mut worst) = (m.grid.n_dir(), 0.0f64);
    for p in (0..m.n_points()).filter(|&p| xy[p][0] > 200.0) {
        let (mut num, mut den) = (0.0, 0.0);
        for ii in 0..m.grid.n_freq() {
            let w = n.spectrum(p)[ii * nd] * m.grid.d_sigma[ii];
            num += m.grid.sigma[ii] * w;
            den += w;
        }
        let uu = current(xy[p][0]);
        let expected = (-1.0 + (1.0 + 4.0 * uu / G * sigma0).sqrt()) / (2.0 * uu / G);
        worst = worst.max((num / den / expected - 1.0).abs());
    }
    println!("largest relative departure of the mean σ from σ + kU = σ₀: {worst:.2e}");
    assert!(worst < 0.01, "mean frequency off by {worst:e}");
}

/// A sea smooth in frequency and direction (Gaussian in σ about 0.3 Hz, cos⁸
/// about +y) on a following current accelerating along y, 0 → 0.6 m/s over
/// 500 m of deep water, uniform along x (periodic). Along a ray `k_x` and
/// `ω = σ + k_y V` are constant, and so is the action density in wavenumber
/// space, `N c_g/k` (Liouville), so the exact steady spectrum is
/// `N₀(σ₀, θ₀)(σ/σ₀)³` with `σ₀ = ω`, `k₀ cos θ₀ = k cos θ`. The mean
/// frequency of the model converges to it at second order with MUSCL (9.2e-3 /
/// 2.0e-3 / 3.3e-4, rates 2.2 and 2.6) and first order with upwind (1.8e-2 /
/// 9.0e-3 / 4.5e-3, rates 1.00), refining frequencies (γ = 1.105 / 1.051 /
/// 1.025) and directions (30 / 15 / 7.5°) together.
#[test]
fn a_spread_sea_on_a_current_has_the_exact_doppler_shift() {
    const L: f64 = 500.0;
    let current = |y: f64| 0.6 * y / L;
    let errors = |scheme: SpectralAdvection| -> Vec<f64> {
        [(13, 12), (25, 24), (49, 48)]
            .into_iter()
            .map(|(n_freq, n_dir)| {
                let m = spread_sea_on_a_current(n_freq, n_dir, current, |m| {
                    m.with_spectral_advection(scheme)
                });
                spread_sea_doppler_error(&m, current, 0.5, 50.0).0
            })
            .collect()
    };
    let rates = |e: &[f64]| -> Vec<f64> { e.windows(2).map(|w| (w[0] / w[1]).log2()).collect() };
    let (muscl, upwind) = (
        errors(SpectralAdvection::VanLeer),
        errors(SpectralAdvection::Upwind),
    );
    println!(
        "relative error of the mean σ at (γ, Δθ) = (1.105, 30°), (1.051, 15°), (1.025, 7.5°):"
    );
    println!("  MUSCL {muscl:?}, rates {:?}", rates(&muscl));
    println!("  upwind {upwind:?}, rates {:?}", rates(&upwind));
    assert!(
        rates(&muscl).iter().all(|&r| r > 1.7),
        "MUSCL: {:?}",
        rates(&muscl)
    );
    assert!(
        rates(&upwind).iter().all(|&r| r > 0.8),
        "upwind: {:?}",
        rates(&upwind)
    );
    assert!(
        muscl[1] < 0.25 * upwind[1],
        "MUSCL {muscl:?} against upwind {upwind:?}"
    );
}

/// The incoming action density of the spread sea on a current: Gaussian in σ
/// about 0.3 Hz, cos⁸ about +y.
fn spread_sea(s: f64, theta: f64) -> f64 {
    let (centre, width) = (TAU * 0.3, TAU * 0.04);
    let c = (theta - PI / 2.0).cos();
    let spread = if c > 0.0 { c.powi(8) } else { 0.0 };
    (-((s - centre) / width).powi(2)).exp() * spread
}

/// The exact steady mean σ of [`spread_sea`] in deep water where the current
/// along +y is `v`, by the midpoint rule over (σ, θ): along a ray `k_x` and
/// `ω = σ + k_y V` are constant, and so is `N c_g/k`, so the spectrum is
/// `N₀(σ₀, θ₀)(σ/σ₀)³` with `σ₀ = ω`, `k₀ cos θ₀ = k cos θ`. It depends on the
/// local current only.
fn spread_sea_mean_frequency(v: f64) -> f64 {
    let (lo, hi, n_s, n_t) = (TAU * 0.08, TAU * 0.6, 400, 720);
    let (mut num, mut den) = (0.0, 0.0);
    for a in 0..n_s {
        let s = lo + (hi - lo) * (a as f64 + 0.5) / n_s as f64;
        let k = s * s / G;
        for b in 0..n_t {
            let theta = TAU * (b as f64 + 0.5) / n_t as f64;
            let (kx, ky) = (k * theta.cos(), k * theta.sin());
            if ky <= 0.0 {
                continue;
            }
            let s0 = s + ky * v;
            let k0 = s0 * s0 / G;
            if k0 <= kx.abs() {
                continue;
            }
            let theta0 = (k0 * k0 - kx * kx).sqrt().atan2(kx);
            let w = spread_sea(s0, theta0) * (s / s0).powi(3);
            num += s * w;
            den += w;
        }
    }
    num / den
}

/// The model of the spread sea entering from y = 0 over 500 m of deep water
/// (five P2 elements of 100 m along y, periodic in x) on the current
/// `current(y)` along +y, on `n_freq` frequencies (0.15–0.5 Hz) and `n_dir`
/// directions, configured by `configure`.
fn spread_sea_on_a_current(
    n_freq: usize,
    n_dir: usize,
    current: impl Fn(f64) -> f64,
    configure: impl FnOnce(WaveModel2D) -> WaveModel2D,
) -> WaveModel2D {
    let mut mesh = Mesh2D::channel_periodic_x(0.0, 100.0, 0.0, 500.0, 1, 5);
    for edge in mesh.edges.iter_mut().filter(|e| e.right.is_none()) {
        edge.boundary_tag = Some(BoundaryTag::Open);
    }
    let grid = SpectralGrid::new(0.15, 0.5, n_freq, n_dir);
    let mut e = vec![0.0; grid.n_components()];
    for i in 0..n_freq {
        for j in 0..n_dir {
            // Variance density E = σ N of the incoming action
            let s = grid.sigma[i];
            e[grid.component(i, j)] = s * spread_sea(s, grid.theta[j]);
        }
    }
    let mut m = configure(model(mesh, 2, |_, _| -1000.0, grid).with_boundary_spectrum(&e));
    let v: Vec<f64> = nodes(&m).iter().map(|p| current(p[1])).collect();
    m.set_currents(&vec![0.0; v.len()], &v);
    m
}

/// Run `m` of [`spread_sea_on_a_current`] to steady at Courant number `cfl`
/// and return the largest relative error of the mean σ beyond y = 50 m
/// against [`spread_sea_mean_frequency`], and the step.
fn spread_sea_doppler_error(
    m: &WaveModel2D,
    current: impl Fn(f64) -> f64,
    cfl: f64,
    from_y: f64,
) -> (f64, f64) {
    let xy = nodes(m);
    let mut n = m.zero_state();
    // Steady for every direction 30° or more off the coast
    let slowest = G / (2.0 * TAU * 0.5) * 0.5;
    run(m, &mut n, 3.0 * 500.0 / slowest, cfl);
    let (nd, mut worst) = (m.grid.n_dir(), 0.0f64);
    let mut exact_at: Vec<(f64, f64)> = Vec::new();
    for p in (0..m.n_points()).filter(|&p| xy[p][1] > from_y) {
        let (mut num, mut den) = (0.0, 0.0);
        for c in 0..m.grid.n_components() {
            let i = c / nd;
            let w = n.spectrum(p)[c] * m.grid.d_sigma[i];
            num += m.grid.sigma[i] * w;
            den += w;
        }
        let y = xy[p][1];
        let exact = match exact_at.iter().find(|(yy, _)| (yy - y).abs() < 1e-9) {
            Some(&(_, e)) => e,
            None => {
                let e = spread_sea_mean_frequency(current(y));
                exact_at.push((y, e));
                e
            }
        };
        worst = worst.max((num / den / exact - 1.0).abs());
    }
    (worst, m.compute_dt(cfl))
}

/// The spread sea of [`a_spread_sea_on_a_current_has_the_exact_doppler_shift`]
/// on a following current that rises from 0 to 2 m/s within one 100 m element
/// (`2 clamp((y − 100)/100)`, exact at P2), on 49 frequencies: frequency
/// shifting limits the explicit MUSCL step to 0.63 s, against the geographic
/// 1.39 s. Implicit frequency shifting runs at the geographic step. Beyond the
/// ramp (y > 250 m; inside it the error is the ramp's spatial resolution, 3.5 %
/// for every scheme) the mean σ is off the exact steady state by 2.1e-3,
/// against explicit MUSCL's 3.7e-3: the deferred correction makes the fixed
/// point MUSCL's. Without it, 1.4e-2, as explicit upwind's 1.3e-2.
#[test]
fn implicit_frequency_shifting_steps_past_the_shifting_limit() {
    let current = |y: f64| 2.0 * ((y - 100.0) / 100.0).clamp(0.0, 1.0);
    let (n_freq, n_dir) = (49, 24);
    let solve = |scheme: SpectralAdvection, implicit: bool, cfl: f64| {
        let m = spread_sea_on_a_current(n_freq, n_dir, current, |m| {
            m.with_spectral_advection(scheme)
                .with_implicit_frequency_shift(implicit)
        });
        let limits = m.time_step_limits(cfl);
        let (error, dt) = spread_sea_doppler_error(&m, current, cfl, 250.0);
        (error, dt, limits.propagation.dt)
    };
    let (muscl, dt_muscl, geographic) = solve(SpectralAdvection::VanLeer, false, 0.5);
    let (upwind, dt_upwind, _) = solve(SpectralAdvection::Upwind, false, 0.5);
    let (implicit, dt_implicit, _) = solve(SpectralAdvection::VanLeer, true, 0.5);
    let (implicit_upwind, _, _) = solve(SpectralAdvection::Upwind, true, 0.5);
    println!(
        "a current rising 2 m/s within 100 m, mean σ off the exact steady state beyond it: \
         explicit MUSCL {muscl:.2e} at Δt {dt_muscl:.3} s, explicit upwind {upwind:.2e} at \
         {dt_upwind:.3} s (geographic {geographic:.3} s); implicit {implicit:.2e} at \
         {dt_implicit:.3} s, without the correction {implicit_upwind:.2e}"
    );
    assert!(
        dt_implicit > 1.5 * dt_muscl,
        "implicit Δt {dt_implicit} against explicit {dt_muscl}"
    );
    assert!((dt_implicit / geographic - 1.0).abs() < 1e-12);
    assert!(
        implicit < 1.3 * muscl,
        "implicit {implicit:e} against MUSCL's {muscl:e}"
    );
    assert!(
        implicit_upwind > 3.0 * implicit,
        "the correction does nothing: {implicit_upwind:e} against {implicit:e}"
    );
}

/// `(E*, f_p*) = (g² m_0/U₁₀⁴, f_p U₁₀/g)` of the variance density `e`.
fn dimensionless(grid: &SpectralGrid, e: &[f64], u10: f64) -> (f64, f64) {
    let p = grid.parameters(e);
    (G * G * p.m0 / u10.powi(4), u10 / (p.tp * G))
}

/// The deep-water sources of SWAN's GEN3 KOMEN set (wind, whitecapping, the DIA,
/// the f⁻⁴ tail and the limiter; no bed or breaking in 1000 m).
fn deep_water_sources() -> SourceTerms {
    SourceTerms::swan_defaults(G)
        .with_bottom_friction(None)
        .with_breaking(None)
}

/// Fetch-limited growth: a steady 10 m/s wind blowing off a straight coast over
/// deep water (a strip periodic along the coast, 120 km offshore, P1). In the
/// steady state the peak frequency follows Kahma & Calkoen's (1992) composite
/// `f_p* = 2.18 X*^−0.27` to a few per cent from X* = 10³ to 1.2·10⁴, and the
/// energy `E* = 5.2e-7 X*^0.9` within the known deficit of Komen whitecapping at
/// long fetch: measured 1.3× at X* = 1200 falling to 0.60× at 1.2·10⁴ (E* ∝
/// X*^0.56; f_p within 0.90–1.05×). The curve is converged: P2, twice the
/// elements, 36 directions, a 2 Hz grid or no tail move it by ≤ 4.5 %.
#[test]
fn fetch_limited_growth_follows_kahma_and_calkoen() {
    let curve = fetch_curve(deep_water_sources(), 1);
    assert_follows_kahma_and_calkoen(&curve);
}

/// The fetch-limited strip of [`fetch_limited_growth_follows_kahma_and_calkoen`]
/// with `sources` and `substeps` propagation steps per step, run to the steady
/// state: `(X*, E*, f_p*)` at the nodes from X* = 10³ to 1.25·10⁴.
fn fetch_curve(sources: SourceTerms, substeps: usize) -> Vec<(f64, f64, f64)> {
    const LENGTH: f64 = 120e3;
    let u10 = 10.0;
    let mut mesh = Mesh2D::channel_periodic_x(0.0, 1000.0, 0.0, LENGTH, 1, 30);
    // The coast at y = 0 lets nothing in; the open sea at y = L lets waves out
    for edge in mesh.edges.iter_mut().filter(|e| e.right.is_none()) {
        if mesh.vertices[edge.vertices.0][1] > 0.5 * LENGTH {
            edge.boundary_tag = Some(BoundaryTag::Open);
        }
    }
    let wind = Wind {
        u10,
        direction: PI / 2.0,
    };
    let m = model(
        mesh,
        1,
        |_, _| -1000.0,
        SpectralGrid::new(0.06, 1.0, 25, 24),
    )
    .with_sources(sources)
    .with_wind(wind)
    .with_substeps(substeps);
    let mut n = m.zero_state();
    run(&m, &mut n, 10.0 * 3600.0, 0.5);
    let xy = nodes(&m);
    let mut e = vec![0.0; m.grid.n_components()];
    let mut curve: Vec<(f64, f64, f64)> = Vec::new();
    for (p, [_, y]) in xy.into_iter().enumerate() {
        let x = G * y / (u10 * u10);
        if !(1000.0..12500.0).contains(&x) || curve.iter().any(|c| (c.0 - x).abs() < 1.0) {
            continue;
        }
        m.energy_spectrum_into(&n, p, &mut e);
        let (es, fs) = dimensionless(&m.grid, &e, u10);
        curve.push((x, es, fs));
    }
    curve.sort_by(|a, b| a.0.total_cmp(&b.0));
    curve
}

/// The bounds of [`fetch_limited_growth_follows_kahma_and_calkoen`].
fn assert_follows_kahma_and_calkoen(curve: &[(f64, f64, f64)]) {
    for &(x, es, fs) in curve {
        let (e_kc, f_kc) = (5.2e-7 * x.powf(0.9), 2.18 * x.powf(-0.27));
        println!(
            "X* {x:7.0}: E* {es:.3e} ({:.2}× KC92), f_p* {fs:.4} ({:.2}× KC92)",
            es / e_kc,
            fs / f_kc
        );
        assert!(
            (0.85..1.12).contains(&(fs / f_kc)),
            "X* {x:.0}: f_p* {fs:.4} against {f_kc:.4}"
        );
        assert!(
            (0.55..1.4).contains(&(es / e_kc)),
            "X* {x:.0}: E* {es:.3e} against {e_kc:.3e}"
        );
    }
    // Growing with fetch, slower than KC92's 0.9 but not stalled
    let (first, last) = (curve[0], curve[curve.len() - 1]);
    let exponent = (last.1 / first.1).ln() / (last.0 / first.0).ln();
    println!("E* ∝ X*^{exponent:.2}");
    assert!(curve.windows(2).all(|w| w[1].1 > w[0].1));
    assert!((0.5..1.0).contains(&exponent), "E* ∝ X*^{exponent}");
}

/// The sources implicit in the DIA with Hersbach & Janssen's rate limiter
/// (WAM's integration) at long steps. The fetch-limited strip of
/// [`fetch_limited_growth_follows_kahma_and_calkoen`] meets the same bounds
/// with the sources once per 8 propagation steps (an outer step of 145 s),
/// and its curve is that of one propagation step per step (18 s) to 1.8 % in
/// E* (at the shortest fetches; ≤ 0.2 % beyond X* = 5000) and 0.5 % in f_p*.
/// With frozen rates and Ris's limiter, which caps the growth per step, the
/// shortest fetch's E* drops 9 % between the two steps (1.17 → 1.06× KC92).
#[test]
fn implicit_sources_keep_the_fetch_curve_at_long_steps() {
    let sources = deep_water_sources()
        .with_integration(SourceIntegration::Implicit)
        .with_limiter(Some(GrowthLimiter::Rate(DEFAULT_RATE_LIMITER)));
    let (short, long) = (fetch_curve(sources.clone(), 1), fetch_curve(sources, 8));
    assert_follows_kahma_and_calkoen(&long);
    for (a, b) in short.iter().zip(&long) {
        assert_eq!(a.0, b.0);
        assert!(
            (b.1 / a.1 - 1.0).abs() < 0.025,
            "X* {:.0}: E* {:e} against {:e}",
            a.0,
            b.1,
            a.1
        );
        assert!(
            (b.2 / a.2 - 1.0).abs() < 0.01,
            "X* {:.0}: f_p* {} against {}",
            a.0,
            b.2,
            a.2
        );
    }
}

/// Duration-limited growth at one node: a 10 m/s wind over a calm deep sea, the
/// sources alone. The energy grows and the peak moves to lower frequencies
/// monotonically, towards Pierson–Moskowitz (E* = 3.64e-3, f_p* = 0.13): 0.85×
/// and 0.95× of it after 96 h (t* = 3.4·10⁵). The step does not matter: 10, 60
/// and 300 s agree to 0.3 % at 96 h. (Run on, the sea passes Pierson–Moskowitz
/// after ≈ 180 h and reaches 1.5× at 1000 h: Komen whitecapping with δ = 1 and
/// the δ = 0 constant C_ds, as SWAN warns.)
#[test]
fn duration_limited_growth_approaches_pierson_moskowitz() {
    let grid = SpectralGrid::new(0.06, 1.0, 30, 24);
    let (u10, depth, dt) = (10.0, 1000.0, 300.0);
    let wind = Wind {
        u10,
        direction: 0.0,
    };
    let sources = deep_water_sources();
    let k: Vec<f64> = grid
        .sigma
        .iter()
        .map(|&s| wavenumber(s, depth, G))
        .collect();
    let (nc, nd) = (grid.n_components(), grid.n_dir());
    let mut n = vec![0.0; nc];
    let (mut e, mut a, mut b) = (vec![0.0; nc], vec![0.0; nc], vec![0.0; nc]);
    let mut history = Vec::new();
    for step in 1..=(96 * 3600) / 300 {
        sources.integrate(&grid, &mut n, &k, depth, wind, dt, &mut e, &mut a, &mut b);
        if step % 12 == 0 {
            let e: Vec<f64> = (0..nc).map(|c| n[c] * grid.sigma[c / nd]).collect();
            history.push(dimensionless(&grid, &e, u10));
        }
    }
    let (es, fs) = *history.last().unwrap();
    println!(
        "after 96 h: E* {es:.3e} ({:.2}× PM), f_p* {fs:.4}",
        es / 3.64e-3
    );
    assert!(history.windows(2).all(|w| w[1].0 > w[0].0));
    // The parabolic peak may sit still between hourly samples
    assert!(history.windows(2).all(|w| w[1].1 <= w[0].1 * (1.0 + 1e-9)));
    assert!((0.75..1.0).contains(&(es / 3.64e-3)), "E* {es:e}");
    assert!((fs / 0.13 - 1.0).abs() < 0.1, "f_p* {fs}");
}

/// Duration-limited growth with the sources implicit in the DIA and Hersbach &
/// Janssen's rate limiter: the sea at 96 h is within the bounds of
/// [`duration_limited_growth_approaches_pierson_moskowitz`], and steps of 300
/// and 900 s give the sea of 60 s steps to 2.2 % in E* at 6 h (young, where
/// the limiter acts; f_p* to 3.5 %, the peak moving a bin) and to 0.15 % at
/// 96 h. With frozen rates and Ris's limiter E* at 6 h drops 8 % from 60 to
/// 300 s steps.
#[test]
fn implicit_sources_grow_a_sea_whatever_the_step() {
    let grid = SpectralGrid::new(0.06, 1.0, 30, 24);
    let (u10, depth) = (10.0, 1000.0);
    let wind = Wind {
        u10,
        direction: 0.0,
    };
    let sources = deep_water_sources()
        .with_integration(SourceIntegration::Implicit)
        .with_limiter(Some(GrowthLimiter::Rate(DEFAULT_RATE_LIMITER)));
    let k: Vec<f64> = grid
        .sigma
        .iter()
        .map(|&s| wavenumber(s, depth, G))
        .collect();
    let (nc, nd) = (grid.n_components(), grid.n_dir());
    let grow = |dt: f64| -> [(f64, f64); 2] {
        let mut n = vec![0.0; nc];
        let (mut e, mut a, mut b) = (vec![0.0; nc], vec![0.0; nc], vec![0.0; nc]);
        let mut at = [(0.0, 0.0); 2];
        for step in 1..=(96.0 * 3600.0 / dt) as usize {
            sources.integrate(&grid, &mut n, &k, depth, wind, dt, &mut e, &mut a, &mut b);
            let t = step as f64 * dt;
            for (slot, hours) in at.iter_mut().zip([6.0, 96.0]) {
                if t == hours * 3600.0 {
                    let e: Vec<f64> = (0..nc).map(|c| n[c] * grid.sigma[c / nd]).collect();
                    *slot = dimensionless(&grid, &e, u10);
                }
            }
        }
        at
    };
    let reference = grow(60.0);
    let (es, fs) = reference[1];
    println!(
        "after 96 h: E* {es:.3e} ({:.2}× PM), f_p* {fs:.4}",
        es / 3.64e-3
    );
    assert!((0.75..1.0).contains(&(es / 3.64e-3)), "E* {es:e}");
    assert!((fs / 0.13 - 1.0).abs() < 0.1, "f_p* {fs}");
    for dt in [300.0, 900.0] {
        for (hours, (a, b)) in [6, 96].iter().zip(reference.iter().zip(grow(dt))) {
            assert!(
                (b.0 / a.0 - 1.0).abs() < if *hours == 6 { 0.03 } else { 0.005 },
                "{dt} s, {hours} h: E* {:e} against {:e}",
                b.0,
                a.0
            );
            assert!(
                (b.1 / a.1 - 1.0).abs() < if *hours == 6 { 0.05 } else { 0.005 },
                "{dt} s, {hours} h: f_p* {} against {}",
                b.1,
                a.1
            );
        }
    }
}

/// The spread sea of [`a_spread_sea_refracts_to_the_exact_steady_state`] on a
/// steep shelf, 20 to 2 m over 300 m in four P2 elements, where refraction
/// limits the explicit MUSCL step to a third of the geographic one (0.36 s
/// against 1.11 s). Implicit refraction runs at the geographic step. With the
/// deferred correction its steady state is MUSCL's up to the splitting: the
/// mean direction is 0.195° off the exact one against explicit MUSCL's 0.167°,
/// and 0.174° at a quarter of the step (the splitting error falls about
/// linearly: backward Euler half-steps around the stages). Without the
/// correction it is first-order upwind's (0.91° against 0.88° explicitly).
#[test]
fn implicit_refraction_steps_past_the_turning_limit() {
    const LX: f64 = 100.0;
    const LY: f64 = 300.0;
    let depth = |y: f64| 20.0 - 18.0 * y / LY;
    let (mean, m) = (70f64.to_radians(), 8.0);
    let incoming = |theta: f64| {
        let c = (theta - mean).cos();
        if c > 0.0 { c.powf(m) } else { 0.0 }
    };
    let n_dir = 72;
    let solve = |scheme: SpectralAdvection, implicit: bool, cfl: f64| -> (f64, f64) {
        let mut mesh = Mesh2D::channel_periodic_x(0.0, LX, 0.0, LY, 1, 4);
        for edge in mesh.edges.iter_mut().filter(|e| e.right.is_none()) {
            edge.boundary_tag = Some(BoundaryTag::Open);
        }
        let grid = SpectralGrid::new(0.12, 0.15, 2, n_dir);
        let mut e = vec![0.0; grid.n_components()];
        for j in 0..n_dir {
            e[grid.component(0, j)] = incoming(grid.theta[j]);
        }
        let m = model(mesh, 2, |_, y| -depth(y), grid)
            .with_boundary_spectrum(&e)
            .with_spectral_advection(scheme)
            .with_implicit_refraction(implicit);
        let sigma = m.grid.sigma[0];
        let (k0, cg0) = {
            let k = wavenumber(sigma, depth(0.0), G);
            (k, group_velocity(sigma, k, depth(0.0)))
        };
        let mut n = m.zero_state();
        let cg_min = group_velocity(sigma, wavenumber(sigma, 2.0, G), 2.0);
        let dt = m.compute_dt(cfl);
        run(
            &m,
            &mut n,
            3.0 * LY / (cg_min * 20f64.to_radians().sin()),
            cfl,
        );
        let (xy, params) = (nodes(&m), m.parameters(&n));
        let mut worst: f64 = 0.0;
        for p in (0..m.n_points()).filter(|&p| (30.0..270.0).contains(&xy[p][1])) {
            let d = depth(xy[p][1]);
            let k = wavenumber(sigma, d, G);
            let cg = group_velocity(sigma, k, d);
            let (mut a, mut b) = (0.0, 0.0);
            let fine = 20_000;
            for q in 0..fine {
                let theta = PI * (q as f64 + 0.5) / fine as f64;
                let c0 = k / k0 * theta.cos();
                if c0.abs() < 1.0 {
                    let w = incoming(c0.acos()) * k * cg0 / (k0 * cg);
                    a += w * theta.cos();
                    b += w * theta.sin();
                }
            }
            let exact = b.atan2(a);
            worst = worst.max((params[p].direction - exact).to_degrees().abs());
        }
        (worst, dt)
    };
    let (muscl, dt_muscl) = solve(SpectralAdvection::VanLeer, false, 0.5);
    let (upwind, dt_upwind) = solve(SpectralAdvection::Upwind, false, 0.5);
    let (implicit, dt_implicit) = solve(SpectralAdvection::VanLeer, true, 0.5);
    let (finer, dt_finer) = solve(SpectralAdvection::VanLeer, true, 0.125);
    let (implicit_upwind, _) = solve(SpectralAdvection::Upwind, true, 0.5);
    println!(
        "steep shelf, 5° bins, mean direction off the exact steady state: explicit MUSCL \
         {muscl:.3}° at Δt {dt_muscl:.3} s, explicit upwind {upwind:.3}° at {dt_upwind:.3} s; \
         implicit {implicit:.3}° at {dt_implicit:.3} s, {finer:.3}° at {dt_finer:.3} s; \
         implicit without the correction {implicit_upwind:.3}°"
    );
    assert!(
        dt_implicit > 2.5 * dt_muscl,
        "implicit Δt {dt_implicit} against explicit {dt_muscl}"
    );
    assert!(
        implicit < 1.3 * muscl,
        "implicit {implicit}° against MUSCL's {muscl}°"
    );
    assert!(finer < 1.1 * muscl, "{finer}° at the smaller step");
    assert!(
        (implicit_upwind / upwind - 1.0).abs() < 0.2,
        "implicit upwind {implicit_upwind}° against explicit {upwind}°"
    );
}

/// A boundary spectrum that changes along the open face
/// (`WaveModel2D::set_boundary_spectra`, one per node): one component
/// travelling straight in from y = 0 over deep water, its density rising
/// linearly along x, reaches the steady state `N(x, y) = N_b(x)` downstream,
/// exact at P2: 9e-15 after eight crossings (1.4e-6 after four, the
/// transient). The same spectrum at every node is the uniform boundary bit
/// for bit.
#[test]
fn a_boundary_spectrum_varying_along_the_face_is_carried_straight_in() {
    const LX: f64 = 400.0;
    const LY: f64 = 600.0;
    let mesh = || {
        Mesh2D::uniform_rectangle_with_sides(
            0.0,
            LX,
            0.0,
            LY,
            4,
            3,
            [
                BoundaryTag::Open,
                BoundaryTag::Wall,
                BoundaryTag::Wall,
                BoundaryTag::Wall,
            ],
        )
    };
    let grid = SpectralGrid::new(0.1, 0.15, 2, 8);
    let (i, j) = (0, 2);
    assert!((grid.theta[j] - PI / 2.0).abs() < 1e-12);
    let c = grid.component(i, j);
    let nc = grid.n_components();
    let mut m = model(mesh(), 2, |_, _| -1000.0, grid.clone());
    let xy = nodes(&m);
    let points = m.open_boundary_points();
    assert!(points.iter().all(|&p| xy[p][1].abs() < 1e-9));
    assert_eq!(
        points.len(),
        4 * 3,
        "the nodes of four P2 faces (DG: per element)"
    );
    let profile = |x: f64| 1.0 + x / LX;
    let mut e = vec![0.0; points.len() * nc];
    for (s, &p) in points.iter().enumerate() {
        e[s * nc + c] = profile(xy[p][0]);
    }
    m.set_boundary_spectra(&e);
    let sigma = m.grid.sigma[i];
    let cg = group_velocity(sigma, wavenumber(sigma, 1000.0, G), 1000.0);
    let mut n = m.zero_state();
    run(&m, &mut n, 8.0 * LY / cg, 0.5);
    let worst = (0..m.n_points())
        .map(|p| (n.spectrum(p)[c] * sigma / profile(xy[p][0]) - 1.0).abs())
        .fold(0.0, f64::max);
    println!("largest relative departure from the boundary's profile: {worst:.2e}");
    assert!(worst < 1e-12, "departs by {worst:e}");

    // A constant spectrum per node is the uniform boundary
    let uniform: Vec<f64> = (0..nc).map(|k| if k == c { 2.0 } else { 0.0 }).collect();
    let m_uniform = model(mesh(), 2, |_, _| -1000.0, grid.clone()).with_boundary_spectrum(&uniform);
    let mut m_nodal = model(mesh(), 2, |_, _| -1000.0, grid);
    m_nodal.set_boundary_spectra(&uniform.repeat(points.len()));
    let (mut a, mut b) = (m_uniform.zero_state(), m_nodal.zero_state());
    run(&m_uniform, &mut a, 0.5 * LY / cg, 0.5);
    run(&m_nodal, &mut b, 0.5 * LY / cg, 0.5);
    assert!(a.component(c).eq(b.component(c)));
}

/// Substeps keep what one short step gives where refraction and breaking are
/// fast (TODO F.4, 2026-10-08). A spread swell (cos⁸, 30° off the shore
/// normal, H_s 1.5 m) refracts and breaks on a shelf from 10 to 0.5 m deep,
/// with implicit refraction, Battjes–Janssen breaking and bottom friction.
/// Stepped 8 propagation steps at a time, the steady H_s in the surf zone and
/// the mean direction stay those of the single steps: breaking follows every
/// propagation step, and refraction takes its halves around each (0.53 % and
/// 0.001°). With both once per outer step, as before, the surf zone kept half
/// an outer step of undissipated inflow (17 % and 0.26°).
#[test]
fn substeps_keep_the_surf_zone_and_the_refraction_of_a_single_step() {
    const LX: f64 = 100.0;
    const LY: f64 = 500.0;
    let depth = |y: f64| 10.0 - 9.5 * y / LY;
    let grid = SpectralGrid::new(0.08, 0.16, 4, 36);
    let mean = 60f64.to_radians();
    let mut e = grid.jonswap(1.5, 9.0, 3.3, mean, 8.0);
    let hs = grid.parameters(&e).hs;
    e.iter_mut().for_each(|x| *x *= (1.5 / hs).powi(2));
    let solve = |substeps: usize| {
        let mut mesh = Mesh2D::channel_periodic_x(0.0, LX, 0.0, LY, 1, 10);
        for edge in mesh.edges.iter_mut().filter(|e| e.right.is_none()) {
            edge.boundary_tag = Some(if mesh.vertices[edge.vertices.0][1] < 0.5 * LY {
                BoundaryTag::Open
            } else {
                BoundaryTag::Wall
            });
        }
        let m = model(mesh, 2, |_, y| -depth(y), grid.clone())
            .with_sources(
                SourceTerms::none(G)
                    .with_breaking(Some((1.0, 0.73)))
                    .with_bottom_friction(Some(0.038)),
            )
            .with_boundary_spectrum(&e)
            .with_implicit_refraction(true)
            .with_implicit_frequency_shift(true)
            .with_substeps(substeps);
        let mut n = m.zero_state();
        let cg_min = group_velocity(m.grid.sigma[0], wavenumber(m.grid.sigma[0], 0.5, G), 0.5);
        run(&m, &mut n, 4.0 * LY / cg_min, 0.5);
        let xy = nodes(&m);
        let params = m.parameters(&n);
        (0..m.n_points())
            .map(|p| (depth(xy[p][1]), params[p].hs, params[p].direction))
            .collect::<Vec<_>>()
    };
    let (single, sub) = (solve(1), solve(8));
    let (mut surf, mut turn, mut hs_min_ratio): (f64, f64, f64) = (0.0, 0.0, f64::INFINITY);
    for (a, b) in single.iter().zip(&sub) {
        turn = turn.max((b.2 - a.2).to_degrees().abs());
        if a.0 < 3.0 {
            surf = surf.max((b.1 / a.1 - 1.0).abs());
            hs_min_ratio = hs_min_ratio.min(a.1 / (0.73 * a.0));
        }
    }
    println!(
        "8 substeps against single steps: H_s in water under 3 m off by up to {:.2} %, the \
         mean direction by up to {turn:.3}°; H_s/(γ d) there ≥ {hs_min_ratio:.2}",
        100.0 * surf
    );
    assert!(
        hs_min_ratio > 0.4,
        "the surf zone breaks: H_s/(γ d) {hs_min_ratio}"
    );
    assert!(surf < 0.01, "H_s in the surf zone off by {surf:e}");
    assert!(turn < 0.1, "the mean direction off by {turn}°");
}

/// The two-sided rate limiter (`GrowthLimiter::RateBothSigns`, ecWAM's cap on
/// `|ΔF|`) leaves depth-induced breaking whole: the model steps breaking on a
/// pass of its own under it, also in one step. On the shelf of
/// `substeps_keep_the_surf_zone_and_the_refraction_of_a_single_step` (a swell
/// breaking from 10 to 0.5 m, no wind, so the limit is `C g² f*_PM f⁻⁴ Δt`),
/// the steady H_s in the surf zone is that of no limiter to 0.81 %: 0.24 %
/// from stepping breaking after the friction instead of with it, the rest
/// the cap on the bottom friction's losses (ecWAM caps them too). With
/// breaking inside the cap the surf zone kept up to 5.9× the H_s.
#[test]
fn the_two_sided_limiter_leaves_breaking_whole() {
    const LX: f64 = 100.0;
    const LY: f64 = 500.0;
    let depth = |y: f64| 10.0 - 9.5 * y / LY;
    let grid = SpectralGrid::new(0.08, 0.16, 4, 36);
    let mut e = grid.jonswap(1.5, 9.0, 3.3, 60f64.to_radians(), 8.0);
    let hs = grid.parameters(&e).hs;
    e.iter_mut().for_each(|x| *x *= (1.5 / hs).powi(2));
    let solve = |limiter: Option<GrowthLimiter>| {
        let mut mesh = Mesh2D::channel_periodic_x(0.0, LX, 0.0, LY, 1, 10);
        for edge in mesh.edges.iter_mut().filter(|e| e.right.is_none()) {
            edge.boundary_tag = Some(if mesh.vertices[edge.vertices.0][1] < 0.5 * LY {
                BoundaryTag::Open
            } else {
                BoundaryTag::Wall
            });
        }
        let m = model(mesh, 2, |_, y| -depth(y), grid.clone())
            .with_sources(
                SourceTerms::none(G)
                    .with_breaking(Some((1.0, 0.73)))
                    .with_bottom_friction(Some(0.038))
                    .with_limiter(limiter),
            )
            .with_boundary_spectrum(&e)
            .with_implicit_refraction(true)
            .with_implicit_frequency_shift(true);
        let mut n = m.zero_state();
        let cg_min = group_velocity(m.grid.sigma[0], wavenumber(m.grid.sigma[0], 0.5, G), 0.5);
        run(&m, &mut n, 4.0 * LY / cg_min, 0.5);
        let xy = nodes(&m);
        let params = m.parameters(&n);
        (0..m.n_points())
            .map(|p| (depth(xy[p][1]), params[p].hs))
            .collect::<Vec<_>>()
    };
    let free = solve(None);
    let capped = solve(Some(GrowthLimiter::RateBothSigns(DEFAULT_RATE_LIMITER)));
    let (mut surf, mut hs_min_ratio): (f64, f64) = (0.0, f64::INFINITY);
    for (a, b) in free.iter().zip(&capped) {
        if a.0 < 3.0 {
            surf = surf.max((b.1 / a.1 - 1.0).abs());
            hs_min_ratio = hs_min_ratio.min(a.1 / (0.73 * a.0));
        }
    }
    println!(
        "two-sided limiter against none: H_s in water under 3 m off by up to {:.3} %; \
         H_s/(γ d) there ≥ {hs_min_ratio:.2}",
        100.0 * surf
    );
    assert!(
        hs_min_ratio > 0.4,
        "the surf zone breaks: H_s/(γ d) {hs_min_ratio}"
    );
    assert!(surf < 0.012, "H_s in the surf zone off by {surf:e}");
}

/// Absorbing land is a coast: on a shelf with an island of whole dry
/// elements, a stormy sea (wind, the full sources, implicit refraction and
/// shifting, swell through open boundaries) evolves in the water exactly, bit
/// for bit, as on the mesh without the island's elements, whose faces there
/// are walls; the island holds no action. Once the tide covers the island the
/// model is the one without absorbing land, bit for bit.
#[test]
fn absorbing_land_is_the_coast_of_a_mesh_without_it() {
    const LX: f64 = 6000.0;
    const LY: f64 = 4000.0;
    let water = |x: f64, _: f64| -(30.0 - 4e-3 * x);
    let full = Mesh2D::uniform_rectangle_with_bc(0.0, LX, 0.0, LY, 6, 4, BoundaryTag::Open);
    let centre = |mesh: &Mesh2D, k: ElementIndex| {
        let c = mesh.reference_to_physical(k, 0.0, 0.0);
        (3000.0..5000.0).contains(&c[0]) && (1000.0..3000.0).contains(&c[1])
    };
    let island: Vec<bool> = ElementIndex::iter(full.n_elements)
        .map(|k| centre(&full, k))
        .collect();
    let (cut, kept) = full.retain_elements(|k| !island[k.as_usize()], BoundaryTag::Wall);
    assert_eq!(kept.len(), full.n_elements - 4);
    let grid = SpectralGrid::new(0.05, 0.4, 10, 18);
    let mut sea = grid.jonswap(2.5, 9.0, 3.3, 0.3, 4.0);
    let hs = grid.parameters(&sea).hs;
    sea.iter_mut().for_each(|x| *x *= (2.5 / hs).powi(2));
    let build = |mesh: Mesh2D, land: &dyn Fn(usize) -> bool| {
        let ops = DGOperators2D::new(2);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let mut bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, water);
        let nn = ops.n_nodes;
        for k in (0..mesh.n_elements).filter(|&k| land(k)) {
            bathymetry.data[k * nn..(k + 1) * nn].fill(2.0);
        }
        WaveModel2D::new(
            Arc::new(mesh),
            Arc::new(ops),
            Arc::new(geom),
            &bathymetry,
            grid.clone(),
            G,
        )
        .with_sources(SourceTerms::swan_defaults(G))
        .with_wind(Wind {
            u10: 18.0,
            direction: 0.5,
        })
        .with_boundary_spectrum(&sea)
        .with_implicit_refraction(true)
        .with_implicit_frequency_shift(true)
        .with_substeps(2)
    };
    let mut with_island = build(full.clone(), &|k| island[k]).with_absorbing_land(true);
    let without = build(cut, &|_| false);
    let dt = without.compute_dt(0.5);
    assert_eq!(with_island.compute_dt(0.5), dt);
    let steps = |model: &WaveModel2D, n: &mut WaveSolution| {
        let mut ws = WaveWorkspace::default();
        for _ in 0..6 {
            model.step(n, dt, &mut ws);
        }
    };
    let (mut a, mut b) = (with_island.uniform_state(&sea), without.uniform_state(&sea));
    steps(&with_island, &mut a);
    steps(&without, &mut b);
    let nn = with_island.ops.n_nodes;
    for c in 0..grid.n_components() {
        let (a, b): (Vec<f64>, Vec<f64>) = (a.component(c).collect(), b.component(c).collect());
        for (new, &old) in kept.iter().enumerate() {
            assert!(
                a[old * nn..(old + 1) * nn] == b[new * nn..(new + 1) * nn],
                "component {c}, element {old}"
            );
        }
        for k in (0..full.n_elements).filter(|&k| island[k]) {
            assert!(a[k * nn..(k + 1) * nn].iter().all(|&x| x == 0.0));
        }
    }
    let hs_max = with_island
        .parameters(&a)
        .iter()
        .fold(0.0f64, |m, p| m.max(p.hs));
    assert!(hs_max > 2.0, "a sea ran: H_s ≤ {hs_max}");

    // The tide over the island: nothing is dry
    let floored = build(full.clone(), &|k| island[k]);
    let high = vec![3.0; with_island.n_points()];
    with_island.set_water_level(&high);
    let mut floored = floored;
    floored.set_water_level(&high);
    let (mut a, mut b) = (with_island.uniform_state(&sea), floored.uniform_state(&sea));
    steps(&with_island, &mut a);
    steps(&floored, &mut b);
    assert!(a.data == b.data, "nothing dry, nothing absorbed");
}

/// Swell running up a beach onto land (the depth from 10 m to 2 m above the
/// water, the shoreline between two nodes, no breaking). On floored land
/// (0.1 m, the default) the waves slow to the floor's group velocity and
/// pile up, as much as the sinks there allow, so H_s on land depends on the
/// floor and on the sinks. Absorbing land holds nothing, and the sea in the
/// water hardly depends on either.
#[test]
fn absorbing_land_keeps_the_waves_off_it() {
    const LX: f64 = 100.0;
    const LY: f64 = 500.0;
    let depth = |y: f64| 10.0 - 0.024 * y;
    let grid = SpectralGrid::new(0.08, 0.16, 4, 36);
    let mut e = grid.jonswap(1.0, 9.0, 3.3, 75f64.to_radians(), 8.0);
    let hs = grid.parameters(&e).hs;
    e.iter_mut().for_each(|x| *x *= (1.0 / hs).powi(2));
    let solve = |absorbing: bool, depth_min: f64, friction: Option<f64>| {
        let mesh = Mesh2D::channel_periodic_x_with_sides(
            0.0,
            LX,
            0.0,
            LY,
            1,
            10,
            [BoundaryTag::Open, BoundaryTag::Wall],
        );
        let m = model(mesh, 2, |_, y| -depth(y), grid.clone())
            .with_sources(SourceTerms::none(G).with_bottom_friction(friction))
            .with_boundary_spectrum(&e)
            .with_implicit_refraction(true)
            .with_implicit_frequency_shift(true)
            .with_depth_min(depth_min)
            .with_absorbing_land(absorbing);
        let mut n = m.zero_state();
        let cg_min = group_velocity(m.grid.sigma[0], wavenumber(m.grid.sigma[0], 0.4, G), 0.4);
        run(&m, &mut n, 4.0 * LY / cg_min, 0.5);
        let xy = nodes(&m);
        let params = m.parameters(&n);
        (0..m.n_points())
            .map(|p| (depth(xy[p][1]), params[p].hs))
            .collect::<Vec<_>>()
    };
    // The largest H_s on land and in the water, and the largest relative
    // difference in the water between two runs
    let on_land = |r: &[(f64, f64)]| {
        r.iter()
            .filter(|x| x.0 <= 0.0)
            .fold(0.0f64, |m, x| m.max(x.1))
    };
    let in_water = |r: &[(f64, f64)]| {
        r.iter()
            .filter(|x| x.0 > 0.0)
            .fold(0.0f64, |m, x| m.max(x.1))
    };
    let apart = |a: &[(f64, f64)], b: &[(f64, f64)]| {
        a.iter()
            .zip(b)
            .filter(|(x, _)| x.0 > 0.0)
            .fold(0.0f64, |m, (x, y)| m.max((y.1 / x.1 - 1.0).abs()))
    };
    let mut runs = Vec::new();
    for absorbing in [false, true] {
        let base = solve(absorbing, 0.1, None);
        let thin = solve(absorbing, 0.01, None);
        let rough = solve(absorbing, 0.1, Some(0.038));
        println!(
            "absorbing {absorbing}: H_s ≤ {:.3} m in the water, ≤ {:.3} / {:.3} / {:.3} m on land \
             (floor 0.1 m, 0.01 m, 0.1 m with friction); in the water the floor changes H_s by up \
             to {:.2} %",
            in_water(&base),
            on_land(&base),
            on_land(&thin),
            on_land(&rough),
            100.0 * apart(&base, &thin),
        );
        runs.push((base, thin, rough));
    }
    let (floored, absorbing) = (&runs[0], &runs[1]);
    // Floored land: 2.62 m on land against 1.85 m in the water, 6.86 m on a
    // 0.01 m floor, 1.77 m with friction; the floor moves the water's H_s
    // by 12.6 %
    assert!(on_land(&floored.0) > 1.3 * in_water(&floored.0));
    assert!(on_land(&floored.1) > 2.0 * on_land(&floored.0));
    assert!(apart(&floored.0, &floored.1) > 0.1);
    // Absorbing land: nothing on it, and the floor moves the water's H_s by
    // 0.79 %, in the shoreline element
    for run in [&absorbing.0, &absorbing.1, &absorbing.2] {
        assert_eq!(on_land(run), 0.0);
    }
    assert!(apart(&absorbing.0, &absorbing.1) < 0.01);
    // Offshore of the shoreline element the land does not matter
    let offshore = floored
        .0
        .iter()
        .zip(&absorbing.0)
        .filter(|(x, _)| x.0 > 1.0)
        .fold(0.0f64, |m, (x, y)| m.max((y.1 / x.1 - 1.0).abs()));
    println!("absorbing against floored land, H_s over 1 m deep: up to {offshore:.2e}");
    assert!(offshore < 1e-3);
}

/// A coarse P1 element from 20 m of water to 0.15 m at the shore, as the
/// 1 km wave grid at Frøya has: the shallow node gets the deep node's sea
/// through the element's polynomial, and Battjes–Janssen breaking, saturated
/// at `H_rms = γ d`, cannot remove it. The depth limit holds every node to
/// `H_s ≤ √2 γ d` and leaves the water offshore of the element alone.
#[test]
fn the_depth_limit_keeps_metres_of_sea_out_of_decimetres_of_water() {
    const LX: f64 = 100.0;
    const LY: f64 = 500.0;
    const SHORE: f64 = 0.15;
    // 20 m to the last element, then the shore within it
    let depth = |y: f64| {
        if y <= 450.0 {
            20.0
        } else {
            20.0 + (SHORE - 20.0) * (y - 450.0) / 50.0
        }
    };
    let grid = SpectralGrid::new(0.08, 0.16, 4, 36);
    let mut e = grid.jonswap(1.0, 9.0, 3.3, 75f64.to_radians(), 8.0);
    let hs = grid.parameters(&e).hs;
    e.iter_mut().for_each(|x| *x *= (2.0 / hs).powi(2));
    let solve = |limit: bool| {
        let mesh = Mesh2D::channel_periodic_x_with_sides(
            0.0,
            LX,
            0.0,
            LY,
            1,
            10,
            [BoundaryTag::Open, BoundaryTag::Wall],
        );
        let sources = SourceTerms::none(G)
            .with_breaking(Some((1.0, 0.73)))
            .with_depth_limit(limit);
        let m = model(mesh, 1, |_, y| -depth(y), grid.clone())
            .with_sources(sources)
            .with_boundary_spectrum(&e)
            .with_implicit_refraction(true)
            .with_implicit_frequency_shift(true)
            .with_absorbing_land(true);
        let mut n = m.zero_state();
        let cg_min = group_velocity(m.grid.sigma[0], wavenumber(m.grid.sigma[0], 20.0, G), 20.0);
        run(&m, &mut n, 4.0 * LY / cg_min, 0.5);
        let xy = nodes(&m);
        let params = m.parameters(&n);
        (0..m.n_points())
            .map(|p| (xy[p][1], m.depth()[p], params[p].hs))
            .collect::<Vec<_>>()
    };
    let (free, limited) = (solve(false), solve(true));
    let at_shore = |r: &[(f64, f64, f64)]| {
        r.iter()
            .filter(|x| x.0 == LY)
            .fold(0.0f64, |m, x| m.max(x.2))
    };
    // Offshore of the shoreline element, the largest relative change
    let offshore = free
        .iter()
        .zip(&limited)
        .filter(|(x, _)| x.0 < 450.0)
        .fold(0.0f64, |m, (x, y)| m.max((y.2 / x.2 - 1.0).abs()));
    let ratio = |r: &[(f64, f64, f64)]| r.iter().fold(0.0f64, |m, x| m.max(x.2 / x.1));
    // The shoreline element's deep node
    let inner = |r: &[(f64, f64, f64)]| {
        r.iter()
            .filter(|x| x.0 == 450.0)
            .fold(0.0f64, |m, x| m.max(x.2))
    };
    println!(
        "H_s at the shore ({SHORE} m deep): {:.3} m free, {:.4} m limited; largest H_s/d {:.2} \
         against {:.3}; the element's deep node {:.4} against {:.4} m; offshore of the \
         element H_s changes by up to {offshore:.2e}",
        at_shore(&free),
        at_shore(&limited),
        ratio(&free),
        ratio(&limited),
        inner(&free),
        inner(&limited),
    );
    // Free: 4.83 m in 0.15 m of water (H_s/d 32); limited: 0.155 m
    let bound = 2f64.sqrt() * 0.73;
    assert!(at_shore(&free) > 20.0 * bound * SHORE);
    assert!(ratio(&limited) <= bound * (1.0 + 1e-12));
    // Upwind faces: nothing the limit removes at the shore comes back out
    // (measured 0)
    assert!(offshore < 1e-12);
}

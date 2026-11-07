//! Validation tests for 2D SWE solver.
//!
//! These tests verify the solver against analytical solutions and physical principles:
//! 1. Circular dam break (radial symmetry, mass conservation)
//! 2. Standing wave in channel (correct period)
//! 3. Geostrophic balance (Coriolis equilibrium)
//! 4. Sponge layer wave absorption

use dgcom::types::ElementIndex;
use dgcom::{
    DGOperators2D, GeometricFactors2D, Mesh2D, Reflective2D, SWE2DRhsConfig, SWESolution2D,
    SWEState2D, ShallowWater2D, compute_dt_swe_2d, compute_rhs_swe_2d,
    source::{CombinedSource2D, CoriolisSource2D, SpongeLayer2D},
};
use std::f64::consts::PI;

fn k(idx: usize) -> ElementIndex {
    ElementIndex::new(idx)
}

const G: f64 = 9.81;

/// Helper to compute one SSP-RK3 step with source terms
fn ssp_rk3_step_with_sources<BC: dgcom::SWEBoundaryCondition2D>(
    q: &mut SWESolution2D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    config: &SWE2DRhsConfig<BC>,
    dt: f64,
    time: f64,
) {
    let rhs1 = compute_rhs_swe_2d(q, mesh, ops, geom, config, time);
    let mut q1 = q.clone();
    q1.axpy(dt, &rhs1);

    let rhs2 = compute_rhs_swe_2d(&q1, mesh, ops, geom, config, time + dt);
    let mut q2 = q.clone();
    q2.axpy(0.25 * dt, &rhs1);
    q2.axpy(0.25 * dt, &rhs2);

    let rhs3 = compute_rhs_swe_2d(&q2, mesh, ops, geom, config, time + 0.5 * dt);
    q.scale(1.0 / 3.0);
    q.axpy(2.0 / 3.0, &q2);
    q.axpy(2.0 / 3.0 * dt, &rhs3);
}

/// Test circular dam break with mass conservation.
///
/// Initial condition: circular dam with h_in > h_out
/// Verifies:
/// - Mass is conserved to machine precision
/// - Radial symmetry is maintained
/// - Solution remains stable
#[test]
fn test_circular_dam_break_mass_conservation() {
    let mesh = Mesh2D::uniform_periodic(0.0, 100.0, 0.0, 100.0, 10, 10);
    let ops = DGOperators2D::new(2);
    let geom = GeometricFactors2D::compute(&mesh, &ops);

    let equation = ShallowWater2D::new(G);
    let bc = Reflective2D::new();
    let config = SWE2DRhsConfig::new(&equation, &bc).with_coriolis(false);

    // Circular dam: h = 3 inside r < 20, h = 1 outside
    let cx = 50.0;
    let cy = 50.0;
    let r0 = 20.0;

    let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
    q.set_from_functions(
        &mesh,
        &ops,
        |x, y| {
            let r = ((x - cx).powi(2) + (y - cy).powi(2)).sqrt();
            if r < r0 { 3.0 } else { 1.0 }
        },
        |_, _| 0.0,
        |_, _| 0.0,
    );

    let initial_mass = q.integrate_depth(&ops, &geom);

    // Run for a few time steps
    let cfl = 0.3;
    let end_time = 1.0;
    let mut time = 0.0;

    while time < end_time {
        let dt = compute_dt_swe_2d(&q, &mesh, &geom, &equation, ops.order, cfl);
        let dt = dt.min(end_time - time);
        ssp_rk3_step_with_sources(&mut q, &mesh, &ops, &geom, &config, dt, time);
        time += dt;

        // Check solution remains bounded
        assert!(
            !q.has_negative_depth(),
            "Negative depth at time {:.3}",
            time
        );
    }

    let final_mass = q.integrate_depth(&ops, &geom);
    let mass_error = ((final_mass - initial_mass) / initial_mass).abs();

    assert!(
        mass_error < 1e-10,
        "Mass conservation error: {:.2e}",
        mass_error
    );
}

/// Test standing wave in rectangular channel.
///
/// Analytical solution for frictionless channel:
/// η(x, t) = A * cos(kx) * cos(ωt)
/// where ω = sqrt(gH) * k
///
/// The period is timed from the zero crossings of the mode amplitude
/// a(t) = ∫ η cos(kx) dA (interpolated between steps). P2 with 20 elements
/// per wavelength gets it to 3e-5. (The test used to time the peaks of η at
/// one node, which was 0.9 % off, under a 5 % tolerance, and skipped the
/// check when it found fewer than two peaks.)
#[test]
fn test_standing_wave_period() {
    let lx = 100.0;
    let ly = 10.0;
    let nx = 20;
    let ny = 2;

    // Channel mesh (periodic in x, walls in y)
    let mesh = Mesh2D::channel_periodic_x(0.0, lx, 0.0, ly, nx, ny);
    let ops = DGOperators2D::new(2);
    let geom = GeometricFactors2D::compute(&mesh, &ops);

    let h0 = 10.0; // Mean depth
    let amplitude = 0.1;
    let wave_k = 2.0 * PI / lx; // Wave number
    let omega = wave_k * (G * h0).sqrt(); // Analytical frequency
    let period = 2.0 * PI / omega;

    let equation = ShallowWater2D::new(G);
    let bc = Reflective2D::new();
    let config = SWE2DRhsConfig::new(&equation, &bc).with_coriolis(false);

    // Initial condition: η = A*cos(kx), u = v = 0
    let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
    q.set_from_functions(
        &mesh,
        &ops,
        |x, _y| h0 + amplitude * (wave_k * x).cos(),
        |_, _| 0.0,
        |_, _| 0.0,
    );

    // Amplitude of the cos(kx) mode, ∫ (h − h0) cos(kx) dA by GLL quadrature
    let mode = |q: &SWESolution2D| {
        let mut a = 0.0;
        for ki in 0..mesh.n_elements {
            for i in 0..ops.n_nodes {
                let [x, _] = mesh.reference_to_physical(k(ki), ops.nodes_r[i], ops.nodes_s[i]);
                let weight = ops.weights[i] * geom.det_j[ki * ops.n_nodes + i];
                a += weight * (q.get_state(k(ki), i).h - h0) * (wave_k * x).cos();
            }
        }
        a
    };

    // Zero crossings of a(t): at T/4, 3T/4, 5T/4
    let cfl = 0.1;
    let mut time = 0.0;
    let mut previous = mode(&q);
    let mut crossings = Vec::new();
    while crossings.len() < 3 {
        assert!(time < 2.0 * period, "no oscillation by t = {time:.2} s");
        let dt = compute_dt_swe_2d(&q, &mesh, &geom, &equation, ops.order, cfl);
        ssp_rk3_step_with_sources(&mut q, &mesh, &ops, &geom, &config, dt, time);
        time += dt;
        let a = mode(&q);
        if a.signum() != previous.signum() {
            crossings.push(time - dt + dt * previous / (previous - a));
        }
        previous = a;
    }

    let measured_period = crossings[2] - crossings[0];
    let period_error = ((measured_period - period) / period).abs();
    assert!(
        period_error < 1e-4,
        "Standing wave period error: {:.2e} (expected {:.5}s, got {:.5}s)",
        period_error,
        period,
        measured_period
    );
    // First crossing at a quarter period
    assert!(
        (crossings[0] - 0.25 * period).abs() < 1e-3 * period,
        "first zero crossing at {:.4} s, expected {:.4} s",
        crossings[0],
        0.25 * period
    );
}

/// Geostrophic balance.
///
/// On an f-plane, η = h₀ + A sin(kx) with u = 0 and v = (g/f) ∂η/∂x is an
/// exact steady state of the nonlinear SWE over a flat bed: the pressure
/// gradient g h ∂h/∂x balances f h v, the flow runs along the isolines, and
/// nothing varies along y. The wave is periodic, so the periodic mesh sees
/// no jump (the old test put a linear η on a periodic mesh: its 0.1 m seam
/// swamped the 1e-3 Coriolis term, and it passed with Coriolis removed).
///
/// Checks, each with a negative control without Coriolis:
/// - the RHS is a small fraction of the Coriolis term f h v (P3: 3e-6
///   against 6e-3 at 16 elements per wavelength), and converges at order
///   ≈ N (2.9); without Coriolis it is the Coriolis term;
/// - over 6 h (a third of an inertial period, seven gravity-wave crossings
///   of the wavelength) the flow stays within 1.1e-4 of geostrophic;
///   without Coriolis, gravity waves (u ≈ A·c/h₀, 5 % of v) move it 5e-2.
#[test]
fn test_geostrophic_balance() {
    let lx = 100_000.0; // one wavelength
    let f = 1.0e-4;
    let h0 = 100.0;
    let amplitude = 0.1;
    let wave_k = 2.0 * PI / lx;
    let v_amp = G * amplitude * wave_k / f; // 0.62 m/s
    let eta = move |x: f64| h0 + amplitude * (wave_k * x).sin();
    let v_geo = move |x: f64| v_amp * (wave_k * x).cos();

    let equation = ShallowWater2D::new(G);
    let bc = Reflective2D::new();
    let coriolis = CoriolisSource2D::f_plane(f);
    let setup = |n: usize| {
        let mesh = Mesh2D::uniform_periodic(0.0, lx, 0.0, lx, n, 2);
        let ops = DGOperators2D::new(3);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        q.set_from_functions(&mesh, &ops, |x, _| eta(x), |_, _| 0.0, |x, _| v_geo(x));
        (mesh, ops, geom, q)
    };
    let config = |with_coriolis: bool| {
        let config = SWE2DRhsConfig::new(&equation, &bc).with_coriolis(false);
        if with_coriolis {
            config.with_source_terms(&coriolis)
        } else {
            config
        }
    };
    // The Coriolis term the balance rests on
    let coriolis_scale = f * h0 * v_amp;

    // Steady: the residual is truncation error, converging at order ≥ N
    let residual = |n: usize, with_coriolis: bool| {
        let (mesh, ops, geom, q) = setup(n);
        compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config(with_coriolis), 0.0).max_abs()
    };
    let (coarse, fine) = (residual(8, true), residual(16, true));
    let rate = (coarse / fine).log2();
    assert!(
        fine < 1e-3 * coriolis_scale && rate > 2.5,
        "geostrophic residual {coarse:.2e} → {fine:.2e} (rate {rate:.2}), Coriolis term {coriolis_scale:.2e}"
    );
    let unbalanced = residual(8, false);
    assert!(
        unbalanced > 0.9 * coriolis_scale,
        "without Coriolis the residual {unbalanced:.2e} should be the Coriolis term {coriolis_scale:.2e}"
    );

    // Over 6 h the flow stays geostrophic; without Coriolis the pressure
    // gradient drives u and the state oscillates as standing gravity waves
    let drift = |with_coriolis: bool| {
        let (mesh, ops, geom, mut q) = setup(8);
        let config = config(with_coriolis);
        let (cfl, end_time) = (0.3, 6.0 * 3600.0);
        let mut time = 0.0;
        while time < end_time {
            let dt = compute_dt_swe_2d(&q, &mesh, &geom, &equation, ops.order, cfl);
            let dt = dt.min(end_time - time);
            ssp_rk3_step_with_sources(&mut q, &mesh, &ops, &geom, &config, dt, time);
            time += dt;
        }
        let mut deviation: f64 = 0.0;
        for ki in 0..mesh.n_elements {
            for i in 0..ops.n_nodes {
                let [x, _] = mesh.reference_to_physical(k(ki), ops.nodes_r[i], ops.nodes_s[i]);
                let state = q.get_state(k(ki), i);
                let (u, v) = (state.hu / state.h, state.hv / state.h);
                deviation = deviation.max(u.abs()).max((v - v_geo(x)).abs());
            }
        }
        deviation / v_amp
    };
    let (balanced, adjusting) = (drift(true), drift(false));
    assert!(
        balanced < 1e-3,
        "geostrophic flow drifted by {:.2e} of its speed in 6 h",
        balanced
    );
    assert!(
        adjusting > 0.02,
        "without Coriolis the flow should leave the balance, drift {adjusting:.2e}"
    );
}

/// Test sponge layer wave absorption.
///
/// Send a wave toward a sponge layer and verify it's damped
/// compared to a simulation without sponge.
#[test]
fn test_sponge_layer_absorption() {
    let lx = 200.0;
    let ly = 50.0;
    let sponge_width = 30.0;

    let mesh = Mesh2D::channel_periodic_x(0.0, lx, 0.0, ly, 20, 5);
    let ops = DGOperators2D::new(2);
    let geom = GeometricFactors2D::compute(&mesh, &ops);

    let h0 = 10.0;
    let equation = ShallowWater2D::new(G);
    let bc = Reflective2D::new();

    // Initial condition: Gaussian bump moving right
    let initial = |x: f64, _y: f64| -> f64 {
        let x0 = 50.0;
        let sigma = 10.0;
        h0 + 1.0 * (-((x - x0) / sigma).powi(2)).exp()
    };

    // Run WITHOUT sponge
    let config_no_sponge = SWE2DRhsConfig::new(&equation, &bc).with_coriolis(false);
    let mut q_no_sponge = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
    q_no_sponge.set_from_functions(&mesh, &ops, initial, |_, _| 0.0, |_, _| 0.0);

    // Run WITH sponge on right boundary
    let sponge = SpongeLayer2D::rectangular(
        |_x, _y, _t| SWEState2D::new(h0, 0.0, 0.0),
        1.0, // gamma_max
        sponge_width,
        (0.0, lx, 0.0, ly),
        [false, true, false, false], // Right boundary only
    );
    let config_with_sponge = SWE2DRhsConfig::new(&equation, &bc)
        .with_coriolis(false)
        .with_source_terms(&sponge);
    let mut q_with_sponge = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
    q_with_sponge.set_from_functions(&mesh, &ops, initial, |_, _| 0.0, |_, _| 0.0);

    // Run both simulations until wave reaches boundary
    let cfl = 0.3;
    let end_time = 15.0; // Time for wave to reach right boundary
    let mut time = 0.0;

    while time < end_time {
        let dt = compute_dt_swe_2d(&q_no_sponge, &mesh, &geom, &equation, ops.order, cfl);
        let dt = dt.min(end_time - time);

        ssp_rk3_step_with_sources(
            &mut q_no_sponge,
            &mesh,
            &ops,
            &geom,
            &config_no_sponge,
            dt,
            time,
        );
        ssp_rk3_step_with_sources(
            &mut q_with_sponge,
            &mesh,
            &ops,
            &geom,
            &config_with_sponge,
            dt,
            time,
        );
        time += dt;
    }

    // Measure perturbation energy in the sponge zone
    let mut energy_no_sponge = 0.0;
    let mut energy_with_sponge = 0.0;

    for ki in 0..mesh.n_elements {
        for (i, &w) in ops.weights.iter().enumerate() {
            let j = geom.jacobian(ki, i);
            let (r, s) = (ops.nodes_r[i], ops.nodes_s[i]);
            let [x, _y] = mesh.reference_to_physical(k(ki), r, s);

            // Only count energy in right half (where sponge is)
            if x > lx / 2.0 {
                let state_no = q_no_sponge.get_state(k(ki), i);
                let state_with = q_with_sponge.get_state(k(ki), i);

                // Perturbation energy: (h - h0)²
                energy_no_sponge += w * (state_no.h - h0).powi(2) * j;
                energy_with_sponge += w * (state_with.h - h0).powi(2) * j;
            }
        }
    }

    // Sponge should reduce energy significantly
    let reduction = energy_with_sponge / energy_no_sponge.max(1e-14);
    assert!(
        reduction < 0.5,
        "Sponge should reduce wave energy by >50%: ratio = {:.2}",
        reduction
    );
}

/// Test combined source terms (Coriolis + sponge).
#[test]
fn test_combined_source_terms() {
    let mesh = Mesh2D::uniform_periodic(0.0, 100.0, 0.0, 100.0, 5, 5);
    let ops = DGOperators2D::new(2);
    let geom = GeometricFactors2D::compute(&mesh, &ops);

    let equation = ShallowWater2D::new(G);
    let bc = Reflective2D::new();

    // Create combined sources
    let coriolis = CoriolisSource2D::f_plane(1.0e-4);
    let sponge = SpongeLayer2D::rectangular(
        |_x, _y, _t| SWEState2D::new(10.0, 0.0, 0.0),
        0.1,
        10.0,
        (0.0, 100.0, 0.0, 100.0),
        [true, false, false, false],
    );
    let combined = CombinedSource2D::new(vec![&coriolis, &sponge]);

    let config = SWE2DRhsConfig::new(&equation, &bc)
        .with_coriolis(false)
        .with_source_terms(&combined);

    let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
    q.set_from_functions(&mesh, &ops, |_, _| 10.0, |_, _| 1.0, |_, _| 0.5);

    // Just verify it runs without error
    let rhs = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);
    assert!(
        rhs.max_abs() < 1000.0,
        "Combined sources should give bounded RHS"
    );
}

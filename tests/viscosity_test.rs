//! Integration tests for horizontal viscosity in 2D SWE.
//!
//! Tests:
//! 1. Exponential decay: sinusoidal velocity perturbation decays as exp(-ν·k²·t)
//! 2. Smagorinsky smoke test: short simulation doesn't blow up
//! 3. Parallel consistency: parallel RHS matches sequential

use dgcom::{
    DGOperators2D, GeometricFactors2D, HorizontalViscosity2D, Mesh2D, Reflective2D, SWE2DRhsConfig,
    SWEFluxType2D, SWESolution2D, ShallowWater2D, compute_dt_swe_2d, compute_dt_viscosity,
    compute_rhs_swe_2d,
};
use std::f64::consts::PI;

/// SSP-RK3 step for SWE with viscosity.
fn ssp_rk3_step<BC: dgcom::SWEBoundaryCondition2D>(
    q: &mut SWESolution2D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    config: &SWE2DRhsConfig<BC>,
    dt: f64,
    time: f64,
) {
    let n = q.data[0].len(); // total nodes per variable (SoA layout)

    // Stage 1
    let rhs1 = compute_rhs_swe_2d(q, mesh, ops, geom, config, time);
    let mut u1 = SWESolution2D::new(q.n_elements, q.n_nodes);
    u1.copy_from(q);
    u1.axpy(dt, &rhs1);

    // Stage 2
    let rhs2 = compute_rhs_swe_2d(&u1, mesh, ops, geom, config, time + dt);
    u1.axpy(dt, &rhs2);
    let mut u2 = SWESolution2D::new(q.n_elements, q.n_nodes);
    for var in 0..3 {
        for i in 0..n {
            u2.data[var][i] = 0.75 * q.data[var][i] + 0.25 * u1.data[var][i];
        }
    }

    // Stage 3
    let rhs3 = compute_rhs_swe_2d(&u2, mesh, ops, geom, config, time + 0.5 * dt);
    u2.axpy(dt, &rhs3);
    for var in 0..3 {
        for i in 0..n {
            q.data[var][i] = (1.0 / 3.0) * q.data[var][i] + (2.0 / 3.0) * u2.data[var][i];
        }
    }
}

/// Exponential decay test: a sinusoidal velocity perturbation on constant depth
/// with constant viscosity should decay as exp(-ν·k²·t).
///
/// Setup: periodic domain [0, 2π]², h = 1 (uniform), hu = ε·sin(x)·sin(y), hv = 0.
/// With ν constant and no other physics, the x-momentum perturbation decays as
/// ε·exp(-ν·(kx² + ky²)·t) where kx = ky = 1, so k² = 2.
#[test]
fn test_viscosity_exponential_decay() {
    let l = 2.0 * PI;
    let nx = 8;
    let ny = 8;
    let order = 3;
    let nu = 0.1;
    let eps = 0.01; // small perturbation to keep linearized regime

    let mesh = Mesh2D::uniform_periodic(0.0, l, 0.0, l, nx, ny);
    let ops = DGOperators2D::new(order);
    let geom = GeometricFactors2D::compute(&mesh, &ops);

    // Use low gravity to minimize wave effects relative to diffusion
    let equation = ShallowWater2D::new(0.1);
    let bc = Reflective2D::new(); // periodic mesh, BC won't be hit

    let visc = HorizontalViscosity2D::constant(nu);
    let config = SWE2DRhsConfig::new(&equation, &bc)
        .with_coriolis(false)
        .with_flux_type(SWEFluxType2D::Roe)
        .with_viscosity(&visc);

    // Initialize: h = 1, hu = ε·sin(x)·sin(y), hv = 0
    let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
    q.set_from_functions(
        &mesh,
        &ops,
        |_x, _y| 1.0,
        |x, y| eps * x.sin() * y.sin(), // u = hu/h = ε·sin(x)·sin(y)
        |_x, _y| 0.0,
    );

    // Measure initial amplitude (SoA: data[1] = all hu values)
    let initial_max_hu = q.data[1].iter().map(|v| v.abs()).fold(0.0_f64, f64::max);

    // Time integration
    let t_final = 1.0;
    let mut time = 0.0;
    let cfl_adv = 0.3;
    let cfl_visc = 0.2;

    let min_h_elem = (0..mesh.n_elements)
        .map(|k_elem| geom.element_size(k_elem))
        .fold(f64::INFINITY, f64::min);

    while time < t_final {
        let dt_adv = compute_dt_swe_2d(&q, &mesh, &geom, &equation, order, cfl_adv);
        let dt_visc = compute_dt_viscosity(nu, min_h_elem, order, cfl_visc);
        let dt = dt_adv.min(dt_visc).min(t_final - time);
        ssp_rk3_step(&mut q, &mesh, &ops, &geom, &config, dt, time);
        time += dt;
    }

    // Measure final amplitude (SoA: data[1] = all hu values)
    let final_max_hu = q.data[1].iter().map(|v| v.abs()).fold(0.0_f64, f64::max);

    // Theoretical decay factor: exp(-ν·k²·T) = exp(-0.1·2·1.0) = exp(-0.2)
    let theoretical_decay = (-nu * 2.0 * t_final).exp();
    let measured_decay = final_max_hu / initial_max_hu;

    println!("Initial max |hu| = {initial_max_hu:.6e}");
    println!("Final max |hu|   = {final_max_hu:.6e}");
    println!("Measured decay    = {measured_decay:.6}");
    println!("Theoretical decay = {theoretical_decay:.6}");

    // The amplitude should have decreased
    assert!(
        final_max_hu < initial_max_hu,
        "Viscosity should damp the perturbation: initial={initial_max_hu:.6e}, final={final_max_hu:.6e}"
    );

    // Allow generous tolerance since we have wave coupling and numerical diffusion
    // The measured decay should be in the right ballpark (within factor of 2)
    let relative_error = ((measured_decay - theoretical_decay) / theoretical_decay).abs();
    println!("Relative error    = {relative_error:.4}");
    assert!(
        relative_error < 0.5,
        "Decay rate should approximate exp(-ν·k²·t): measured={measured_decay:.6}, theory={theoretical_decay:.6}, rel_err={relative_error:.4}"
    );
}

/// Smoke test for Smagorinsky viscosity: verify no blow-up.
#[test]
fn test_smagorinsky_smoke() {
    let mesh = Mesh2D::uniform_periodic(0.0, 10.0, 0.0, 10.0, 6, 6);
    let ops = DGOperators2D::new(2);
    let geom = GeometricFactors2D::compute(&mesh, &ops);
    let equation = ShallowWater2D::new(9.81);
    let bc = Reflective2D::new();

    let visc = HorizontalViscosity2D::smagorinsky(0.15);
    let config = SWE2DRhsConfig::new(&equation, &bc)
        .with_coriolis(false)
        .with_viscosity(&visc);

    // Initialize with a dam-break-like perturbation
    let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
    q.set_from_functions(
        &mesh,
        &ops,
        |x, y| {
            if (x - 5.0).powi(2) + (y - 5.0).powi(2) < 4.0 {
                2.0
            } else {
                1.0
            }
        },
        |_, _| 0.0,
        |_, _| 0.0,
    );

    let initial_mass = q.integrate_depth(&ops, &geom);

    // Take 20 time steps
    let cfl = 0.2;
    let min_h_elem = (0..mesh.n_elements)
        .map(|k_elem| geom.element_size(k_elem))
        .fold(f64::INFINITY, f64::min);

    let mut time = 0.0;
    for _ in 0..20 {
        let dt_adv = compute_dt_swe_2d(&q, &mesh, &geom, &equation, ops.order, cfl);
        // Use a generous viscosity estimate for dt
        let dt_visc = compute_dt_viscosity(10.0, min_h_elem, ops.order, 0.1);
        let dt = dt_adv.min(dt_visc);
        ssp_rk3_step(&mut q, &mesh, &ops, &geom, &config, dt, time);
        time += dt;
    }

    // Check no blow-up
    let max_val = q.max_abs();
    assert!(
        max_val < 100.0,
        "Smagorinsky should not blow up: max_abs = {max_val:.6e}"
    );

    // Mass should be roughly conserved (viscosity only acts on momentum)
    let final_mass = q.integrate_depth(&ops, &geom);
    let mass_error = ((final_mass - initial_mass) / initial_mass).abs();
    assert!(
        mass_error < 1e-8,
        "Mass should be conserved: initial={initial_mass:.10}, final={final_mass:.10}, error={mass_error:.2e}"
    );
}

/// Verify that viscosity doesn't affect continuity (mass conservation).
/// With viscosity on, the h equation should get zero contribution from viscous terms.
#[test]
fn test_viscosity_mass_conservation() {
    let l = 2.0 * PI;
    let mesh = Mesh2D::uniform_periodic(0.0, l, 0.0, l, 4, 4);
    let ops = DGOperators2D::new(2);
    let geom = GeometricFactors2D::compute(&mesh, &ops);
    let equation = ShallowWater2D::new(9.81);
    let bc = Reflective2D::new();

    let visc = HorizontalViscosity2D::constant(1.0);
    let config = SWE2DRhsConfig::new(&equation, &bc)
        .with_coriolis(false)
        .with_viscosity(&visc);

    // Initialize with non-trivial velocity
    let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
    q.set_from_functions(
        &mesh,
        &ops,
        |_x, _y| 1.0,
        |x, _y| 0.1 * x.sin(),
        |_x, y| 0.1 * y.cos(),
    );

    let initial_mass = q.integrate_depth(&ops, &geom);

    // Take a few steps
    let min_h_elem = (0..mesh.n_elements)
        .map(|k_elem| geom.element_size(k_elem))
        .fold(f64::INFINITY, f64::min);

    let mut time = 0.0;
    for _ in 0..5 {
        let dt_adv = compute_dt_swe_2d(&q, &mesh, &geom, &equation, ops.order, 0.2);
        let dt_visc = compute_dt_viscosity(1.0, min_h_elem, ops.order, 0.1);
        let dt = dt_adv.min(dt_visc);
        ssp_rk3_step(&mut q, &mesh, &ops, &geom, &config, dt, time);
        time += dt;
    }

    let final_mass = q.integrate_depth(&ops, &geom);
    let mass_error = ((final_mass - initial_mass) / initial_mass).abs();
    assert!(
        mass_error < 1e-10,
        "Viscosity must not affect continuity: mass error = {mass_error:.2e}"
    );
}

/// Test the diffusive CFL helper.
#[test]
fn test_compute_dt_viscosity() {
    // Basic sanity
    let dt = compute_dt_viscosity(1.0, 1.0, 2, 0.5);
    // (2*2+1)^2 = 25, dt = 0.5 * 1.0 / (1.0 * 25) = 0.02
    assert!((dt - 0.02).abs() < 1e-14, "dt = {dt}");

    // Zero viscosity → infinite dt
    let dt_zero = compute_dt_viscosity(0.0, 1.0, 2, 0.5);
    assert!(dt_zero.is_infinite());

    // Smaller element → smaller dt (quadratic)
    let dt_small = compute_dt_viscosity(1.0, 0.5, 2, 0.5);
    assert!(
        (dt_small / dt - 0.25).abs() < 1e-14,
        "dt_small = {dt_small}"
    );
}

/// Test parallel RHS matches sequential when viscosity is enabled.
#[cfg(all(feature = "parallel", feature = "simd"))]
#[test]
fn test_viscosity_parallel_consistency() {
    use dgcom::compute_rhs_swe_2d_parallel;

    let mesh = Mesh2D::uniform_periodic(0.0, 10.0, 0.0, 10.0, 4, 4);
    let ops = DGOperators2D::new(2);
    let geom = GeometricFactors2D::compute(&mesh, &ops);
    let equation = ShallowWater2D::new(9.81);
    let bc = Reflective2D::new();

    let visc = HorizontalViscosity2D::constant(0.5);
    let config = SWE2DRhsConfig::new(&equation, &bc)
        .with_coriolis(false)
        .with_viscosity(&visc);

    // Non-trivial initial condition
    let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
    q.set_from_functions(
        &mesh,
        &ops,
        |x, y| 1.0 + 0.1 * (2.0 * PI * x / 10.0).sin() * (2.0 * PI * y / 10.0).cos(),
        |x, _y| 0.05 * (2.0 * PI * x / 10.0).cos(),
        |_x, y| 0.05 * (2.0 * PI * y / 10.0).sin(),
    );

    let rhs_serial = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);
    let rhs_parallel = compute_rhs_swe_2d_parallel(&q, &mesh, &ops, &geom, &config, 0.0);

    let max_diff = (0..3)
        .flat_map(|var| {
            rhs_serial.data[var]
                .iter()
                .zip(rhs_parallel.data[var].iter())
                .map(|(a, b)| (a - b).abs())
        })
        .fold(0.0_f64, f64::max);

    assert!(
        max_diff < 1e-10,
        "Parallel RHS with viscosity should match serial: max diff = {max_diff:.2e}"
    );
}

/// `SWEPhysics2DBuilder::with_viscosity` reaches the `Simulation` path.
///
/// A shear flow u = ε·sin(y), v = 0 over a flat bed on a periodic domain is an
/// exact solution with no pressure gradient and no advection (u·∇u = 0), so the
/// viscous momentum equation reduces to ∂u/∂t = ν ∂²u/∂y²: hu decays as
/// ε·sin(y)·exp(−νt). The same run without viscosity must not decay.
#[test]
fn test_physics_builder_viscosity_decays_shear_flow() {
    use dgcom::Simulation;
    use dgcom::physics::PhysicsBuilder;
    use dgcom::time::SSPRK3;
    use std::sync::Arc;

    let (nu, eps, t_end) = (0.1, 0.01, 1.0);
    let mesh = Arc::new(Mesh2D::uniform_periodic(0.0, 2.0 * PI, 0.0, 2.0 * PI, 8, 8));
    let ops = Arc::new(DGOperators2D::new(3));
    let geom = Arc::new(GeometricFactors2D::compute(&mesh, &ops));
    let amplitude = |viscous: bool| {
        let mut builder = PhysicsBuilder::swe_2d(
            mesh.clone(),
            ops.clone(),
            geom.clone(),
            ShallowWater2D::new(9.81),
            Reflective2D::new(),
        );
        if viscous {
            builder = builder.with_viscosity(HorizontalViscosity2D::constant(nu));
        }
        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        q.set_from_functions(&mesh, &ops, |_, _| 1.0, |_, y| eps * y.sin(), |_, _| 0.0);
        let result = Simulation::new(builder.build(), SSPRK3)
            .with_dt_max(0.005)
            .run(&mut q, 0.0, t_end);
        assert!(result.success, "{:?}", result.error);
        q.data[1].iter().map(|v| v.abs()).fold(0.0_f64, f64::max) / eps
    };

    let viscous = amplitude(true);
    let expected = (-nu * t_end).exp();
    assert!(
        (viscous - expected).abs() < 1e-3,
        "decay {viscous:.6} against exp(-νt) = {expected:.6}"
    );
    let inviscid = amplitude(false);
    assert!(
        (inviscid - 1.0).abs() < 1e-6,
        "inviscid amplitude {inviscid:.8}"
    );
}

/// Shear flow u = ε·sin(y) on a periodic 2π domain (the exact solution
/// ε·sin(y)·exp(−νt) of `test_physics_builder_viscosity_decays_shear_flow`)
/// run through `Simulation` at CFL 1 with the time step it chooses.
fn simulated_shear_decay(nu: f64, t_end: f64, local: bool, grade: f64) -> f64 {
    use dgcom::Simulation;
    use dgcom::physics::PhysicsBuilder;
    use dgcom::time::{MultirateSSPRK3, SSPRK3};
    use std::sync::Arc;

    let eps = 0.01;
    let mut mesh = Mesh2D::uniform_periodic(0.0, 1.0, 0.0, 2.0 * PI, 12, 8);
    // Columns growing geometrically along x (the flow does not vary along x)
    for v in &mut mesh.vertices {
        v[0] = 2.0
            * PI
            * if grade == 1.0 {
                v[0]
            } else {
                (grade.powf(12.0 * v[0]) - 1.0) / (grade.powf(12.0) - 1.0)
            };
    }
    let mesh = Arc::new(mesh);
    let ops = Arc::new(DGOperators2D::new(3));
    let geom = Arc::new(GeometricFactors2D::compute(&mesh, &ops));
    let physics = PhysicsBuilder::swe_2d(
        mesh.clone(),
        ops.clone(),
        geom,
        ShallowWater2D::new(9.81),
        Reflective2D::new(),
    )
    .with_viscosity(HorizontalViscosity2D::constant(nu))
    .build();
    let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
    q.set_from_functions(&mesh, &ops, |_, _| 1.0, |_, y| eps * y.sin(), |_, _| 0.0);
    let result = if local {
        let result = Simulation::new(physics, MultirateSSPRK3::new(8))
            .with_cfl(1.0)
            .run(&mut q, 0.0, t_end);
        let stats = result.local_time_stepping.expect("multirate stats");
        assert!(stats.finest_level >= 2, "{stats:?}");
        result
    } else {
        Simulation::new(physics, SSPRK3)
            .with_cfl(1.0)
            .run(&mut q, 0.0, t_end)
    };
    assert!(result.success, "{:?}", result.error);
    q.data[1].iter().map(|v| v.abs()).fold(0.0_f64, f64::max) / eps
}

/// With a viscosity strong enough that the diffusive limit, not the gravity
/// wave, sets the time step (ν = 5 m²/s on elements of 0.5 m at P3: ≈ 20×
/// below the advective step), `Simulation` stays stable at CFL 1 and follows
/// exp(−νt). Without the viscous bound in `compute_dt` it took the advective
/// step and blew up.
#[test]
fn test_viscous_time_step_bound_keeps_simulation_stable() {
    let (nu, t_end) = (5.0, 0.2);
    let decay = simulated_shear_decay(nu, t_end, false, 1.0);
    let expected = (-nu * t_end).exp();
    assert!(
        (decay - expected).abs() < 1e-3 * expected,
        "decay {decay:.6} against exp(-νt) = {expected:.6}"
    );
}

/// The same under local time stepping on columns graded by 1.2 (element
/// widths 7× apart, levels set by the viscous limit ∝ width²): stable, and
/// the decay is still exp(−νt).
#[test]
fn test_viscous_decay_under_local_time_stepping() {
    let (nu, t_end) = (5.0, 0.2);
    let decay = simulated_shear_decay(nu, t_end, true, 1.2);
    let expected = (-nu * t_end).exp();
    assert!(
        (decay - expected).abs() < 1e-3 * expected,
        "decay {decay:.6} against exp(-νt) = {expected:.6}"
    );
}

/// `element_dt_viscous_swe_2d` bounds the spectral radius ρ of the BR1
/// viscous operator (power iteration on periodic meshes of square and 4:1
/// elements, P1–P4): ρ·Δt ≤ 2.5127 (SSP-RK3's real-axis limit) at CFL 1,
/// and not far below it.
#[test]
fn test_viscous_dt_bounds_the_br1_spectral_radius() {
    use dgcom::solver::{SWEFormulation2D, element_dt_viscous_swe_2d};

    let nu = 1.0;
    let visc = HorizontalViscosity2D::constant(nu);
    let equation = ShallowWater2D::new(9.81);
    let bc = Reflective2D::new();
    // Entropy-conservative fluxes: no interface dissipation, so at rest over
    // a flat bed the momentum RHS of a tiny velocity is the viscous term
    let config = SWE2DRhsConfig::new(&equation, &bc)
        .with_coriolis(false)
        .with_formulation(SWEFormulation2D::EntropyConservative)
        .with_viscosity(&visc);
    for (nx, ny) in [(4, 4), (8, 2)] {
        let mesh = Mesh2D::uniform_periodic(0.0, 8.0, 0.0, 8.0, nx, ny);
        for order in 1..=4 {
            let ops = DGOperators2D::new(order);
            let geom = GeometricFactors2D::compute(&mesh, &ops);
            let n = mesh.n_elements * ops.n_nodes;
            let mut v: Vec<f64> = (0..2 * n)
                .map(|i| ((i * 7919) % 101) as f64 - 50.0)
                .collect();
            let eps = 1e-7;
            let mut rho = 0.0;
            for _ in 0..200 {
                let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
                v.iter_mut().for_each(|x| *x /= norm);
                let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
                q.data[0].fill(1.0);
                for (var, part) in [1, 2].into_iter().zip(v.chunks_exact(n)) {
                    for (x, &p) in q.data[var].iter_mut().zip(part) {
                        *x = eps * p;
                    }
                }
                let rhs = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);
                let w: Vec<f64> = rhs.data[1]
                    .iter()
                    .chain(&rhs.data[2])
                    .map(|x| x / eps)
                    .collect();
                rho = w.iter().zip(&v).map(|(a, b)| a * b).sum::<f64>().abs();
                v = w;
            }
            let dt = element_dt_viscous_swe_2d(&mesh, &ops, &geom, nu, order, 1.0, 0);
            let product = rho * dt;
            println!("{nx}x{ny} P{order}: rho {rho:.3}, rho·dt {product:.3}");
            assert!(product <= 2.5127, "{nx}x{ny} P{order}: ρ·Δt = {product:.4}");
            assert!(
                product > 0.7 * 2.5127,
                "{nx}x{ny} P{order}: ρ·Δt = {product:.4}"
            );
        }
    }
}

/// A Smagorinsky viscosity is in the time step: it is bounded from the
/// current strain at every step. A grid-scale shear u = U·sin(4y) (two
/// elements per wavelength) with an exaggerated `C_s` = 1 and a slow gravity
/// wave makes the viscous limit the binding one (≈ 1/10 of the advective
/// step). At CFL 1, `Simulation` stays stable and the shear decays
/// monotonically; with the advective step alone it blows up in the first
/// step. As the shear decays, ν falls with it and the step grows.
#[test]
fn smagorinsky_viscosity_bounds_the_time_step() {
    use dgcom::Simulation;
    use dgcom::physics::{PhysicsBuilder, PhysicsModule};
    use dgcom::time::SSPRK3;
    use std::sync::Arc;

    let mesh = Arc::new(Mesh2D::uniform_periodic(0.0, 2.0 * PI, 0.0, 2.0 * PI, 8, 8));
    let ops = Arc::new(DGOperators2D::new(3));
    let geom = Arc::new(GeometricFactors2D::compute(&mesh, &ops));
    let physics = |viscosity: Option<HorizontalViscosity2D>| {
        let builder = PhysicsBuilder::swe_2d(
            mesh.clone(),
            ops.clone(),
            geom.clone(),
            ShallowWater2D::new(1e-4),
            Reflective2D::new(),
        );
        match viscosity {
            Some(viscosity) => builder.with_viscosity(viscosity),
            None => builder,
        }
        .build()
    };
    let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
    q.set_from_functions(&mesh, &ops, |_, _| 1.0, |_, y| (4.0 * y).sin(), |_, _| 0.0);

    let dt_advective = physics(None).compute_dt(&q, 1.0);
    let viscous = physics(Some(HorizontalViscosity2D::smagorinsky(1.0)));
    let dt = viscous.compute_dt(&q, 1.0);
    assert!(dt < 0.2 * dt_advective, "{dt} against {dt_advective}");

    let amplitude = |q: &SWESolution2D| q.data[1].iter().map(|v| v.abs()).fold(0.0, f64::max);
    let (mut previous, mut previous_dt) = (amplitude(&q), dt);
    let sim = Simulation::new(viscous, SSPRK3).with_cfl(1.0);
    for step in 1..=5 {
        let result = sim.run(&mut q, 0.25 * (step - 1) as f64, 0.25 * step as f64);
        assert!(result.success, "{:?}", result.error);
        let now = amplitude(&q);
        assert!(
            now < previous,
            "step {step}: amplitude {now} after {previous}"
        );
        assert!(
            result.dt_max >= previous_dt,
            "step {step}: dt {} after {previous_dt}",
            result.dt_max
        );
        (previous, previous_dt) = (now, result.dt_max);
    }
    assert!(
        previous_dt > 1.5 * dt,
        "dt {previous_dt} after starting at {dt}"
    );
}

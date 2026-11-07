//! Convergence test for the DG advection solver.
//!
//! Verifies that the solver achieves the expected order of accuracy (N+1)
//! for smooth solutions, where N is the polynomial order.

use dgcom::{
    BoundaryCondition, DGOperators1D, DGSolution1D, Mesh1D, compute_dt, compute_rhs, ssp_rk3_step,
    ssp_rk3_step_timed,
};
// 2D imports
use dgcom::{
    Advection2D, AdvectionFluxType, DGOperators2D, DGSolution2D, GeometricFactors2D, Mesh2D,
    PeriodicBC2D, compute_dt_advection_2d, compute_rhs_advection_2d, ssp_rk3_step_2d_timed,
};
// SWE 2D imports
use dgcom::{
    Reflective2D, SWE2DRhsConfig, SWEFluxType2D, SWESolution2D, ShallowWater2D, compute_dt_swe_2d,
    compute_rhs_swe_2d,
};
// SWE 1D imports
use dgcom::boundary::ReflectiveBC;
use dgcom::source::SourceTerm;
use dgcom::time::{SSPRK3, TimeIntegrator};
use dgcom::{
    SWEFluxType, SWERhsConfig, SWESolution, SWEState, ShallowWater1D, compute_dt_swe,
    compute_rhs_swe,
};
use std::f64::consts::PI;

/// Run a single advection simulation and return the L2 error.
fn run_advection(n_elements: usize, order: usize, t_final: f64, a: f64, cfl: f64) -> f64 {
    let mesh = Mesh1D::uniform(0.0, 2.0, n_elements);
    let ops = DGOperators1D::new(order);

    // Initial condition: sin(pi * x)
    let initial_condition = |x: f64| (PI * x).sin();

    let mut u = DGSolution1D::new(n_elements, ops.n_nodes);
    u.set_from_function(&mesh, &ops, initial_condition);

    // Exact solution at time t
    let exact_solution = |t: f64| move |x: f64| (PI * (x - a * t)).sin();

    // Time stepping
    let dt = compute_dt(&mesh, a, order, cfl);
    let n_steps = (t_final / dt).ceil() as usize;
    let dt = t_final / n_steps as f64;

    let mut t = 0.0;
    for _ in 0..n_steps {
        // Use time-aware RK3 with BCs evaluated at correct stage times
        ssp_rk3_step_timed(
            &mut u,
            |u_, stage_time| {
                let bc = BoundaryCondition {
                    left: exact_solution(stage_time)(0.0),
                    right: 0.0,
                };
                compute_rhs(u_, &mesh, &ops, a, &bc)
            },
            t,
            dt,
        );
        t += dt;
    }

    u.l2_error(&mesh, &ops, exact_solution(t_final))
}

#[test]
fn test_convergence_p1() {
    // P1 -> expect 2nd order convergence
    let order = 1;
    let t_final = 0.5;
    let a = 1.0;
    let cfl = 0.3;

    let resolutions = [10, 20, 40, 80];
    let errors: Vec<f64> = resolutions
        .iter()
        .map(|&n| run_advection(n, order, t_final, a, cfl))
        .collect();

    println!("P1 convergence:");
    for (i, (&n, &err)) in resolutions.iter().zip(errors.iter()).enumerate() {
        if i > 0 {
            let ratio = errors[i - 1] / err;
            let observed_order = ratio.log2();
            println!(
                "  n={:3}: error={:.4e}, ratio={:.2}, order={:.2}",
                n, err, ratio, observed_order
            );
        } else {
            println!("  n={:3}: error={:.4e}", n, err);
        }
    }

    // Check convergence rate for last refinement
    let ratio = errors[errors.len() - 2] / errors[errors.len() - 1];
    let observed_order = ratio.log2();

    assert!(
        observed_order > 1.8,
        "P1 should be at least 2nd order, observed {:.2}",
        observed_order
    );
}

#[test]
fn test_convergence_p3() {
    let order = 3; // P3 -> expect 4th order convergence
    let t_final = 0.5;
    let a = 1.0;
    let cfl = 0.3;

    let resolutions = [5, 10, 20, 40];
    let errors: Vec<f64> = resolutions
        .iter()
        .map(|&n| run_advection(n, order, t_final, a, cfl))
        .collect();

    println!("P3 convergence:");
    for (i, (&n, &err)) in resolutions.iter().zip(errors.iter()).enumerate() {
        if i > 0 {
            let ratio = errors[i - 1] / err;
            let observed_order = ratio.log2();
            println!(
                "  n={:3}: error={:.4e}, ratio={:.2}, order={:.2}",
                n, err, ratio, observed_order
            );
        } else {
            println!("  n={:3}: error={:.4e}", n, err);
        }
    }

    // Check convergence rate for last refinement
    let ratio = errors[errors.len() - 2] / errors[errors.len() - 1];
    let observed_order = ratio.log2();

    assert!(
        observed_order > 3.5,
        "P3 should be at least 4th order, observed {:.2}",
        observed_order
    );
}

#[test]
fn test_convergence_p5() {
    let order = 5; // P5 -> expect 6th order convergence
    let t_final = 0.1; // Short time to minimize temporal error
    let a = 1.0;
    let cfl = 0.05; // Small CFL for accurate time integration

    // Use coarser resolutions to stay in spatial-error-dominated regime
    // (temporal error is O(dt^3) which limits convergence on fine meshes)
    let resolutions = [3, 6, 12, 24];
    let errors: Vec<f64> = resolutions
        .iter()
        .map(|&n| run_advection(n, order, t_final, a, cfl))
        .collect();

    println!("P5 convergence:");
    for (i, (&n, &err)) in resolutions.iter().zip(errors.iter()).enumerate() {
        if i > 0 {
            let ratio = errors[i - 1] / err;
            let observed_order = ratio.log2();
            println!(
                "  n={:3}: error={:.4e}, ratio={:.2}, order={:.2}",
                n, err, ratio, observed_order
            );
        } else {
            println!("  n={:3}: error={:.4e}", n, err);
        }
    }

    // Check convergence rate for last refinement
    let ratio = errors[errors.len() - 2] / errors[errors.len() - 1];
    let observed_order = ratio.log2();

    assert!(
        observed_order > 5.0,
        "P5 should be at least 6th order, observed {:.2}",
        observed_order
    );
}

#[test]
fn test_negative_velocity() {
    // Test with negative velocity (rightward to leftward flow)
    let order = 3;
    let t_final = 0.5;
    let a = -1.0; // Negative velocity
    let cfl = 0.3;

    let resolutions = [10, 20, 40];
    let errors: Vec<f64> = resolutions
        .iter()
        .map(|&n| {
            let mesh = Mesh1D::uniform(0.0, 2.0, n);
            let ops = DGOperators1D::new(order);

            let initial_condition = |x: f64| (PI * x).sin();
            let mut u = DGSolution1D::new(n, ops.n_nodes);
            u.set_from_function(&mesh, &ops, initial_condition);

            let exact_solution = |t: f64| move |x: f64| (PI * (x - a * t)).sin();

            let dt = compute_dt(&mesh, a, order, cfl);
            let n_steps = (t_final / dt).ceil() as usize;
            let dt = t_final / n_steps as f64;

            let mut t = 0.0;
            for _ in 0..n_steps {
                // Use time-aware RK3 with BCs evaluated at correct stage times
                ssp_rk3_step_timed(
                    &mut u,
                    |u_, stage_time| {
                        // For a < 0, inflow is at right boundary
                        let bc = BoundaryCondition {
                            left: 0.0,                              // Not used (outflow)
                            right: exact_solution(stage_time)(2.0), // Dirichlet at inflow
                        };
                        compute_rhs(u_, &mesh, &ops, a, &bc)
                    },
                    t,
                    dt,
                );
                t += dt;
            }

            u.l2_error(&mesh, &ops, exact_solution(t_final))
        })
        .collect();

    println!("Negative velocity convergence:");
    for (i, (&n, &err)) in resolutions.iter().zip(errors.iter()).enumerate() {
        if i > 0 {
            let ratio = errors[i - 1] / err;
            println!("  n={:3}: error={:.4e}, ratio={:.2}", n, err, ratio);
        } else {
            println!("  n={:3}: error={:.4e}", n, err);
        }
    }

    // Should still converge
    assert!(errors.last().unwrap() < &0.01, "Solution should converge");
}

#[test]
fn test_convergence_periodic() {
    // Test with periodic boundary conditions
    // This avoids time-dependent BC issues and should achieve optimal convergence
    let order = 3;
    let t_final = 2.0; // Full period for domain [0, 2] with a=1
    let a = 1.0;
    let cfl = 0.3;

    let resolutions = [5, 10, 20, 40];
    let errors: Vec<f64> = resolutions
        .iter()
        .map(|&n| {
            let mesh = Mesh1D::uniform_periodic(0.0, 2.0, n);
            let ops = DGOperators1D::new(order);

            // Initial condition: sin(pi * x) - periodic on [0, 2]
            let initial_condition = |x: f64| (PI * x).sin();
            let mut u = DGSolution1D::new(n, ops.n_nodes);
            u.set_from_function(&mesh, &ops, initial_condition);

            // Exact solution at time t (same as initial after t=2 with a=1)
            let exact_solution = |t: f64| move |x: f64| (PI * (x - a * t)).sin();

            let dt = compute_dt(&mesh, a, order, cfl);
            let n_steps = (t_final / dt).ceil() as usize;
            let dt = t_final / n_steps as f64;

            // Periodic BCs: no boundary values needed, use default
            let bc = BoundaryCondition::default();

            for _ in 0..n_steps {
                // Can use simple ssp_rk3_step since there's no time-dependent BC
                ssp_rk3_step(&mut u, |u_| compute_rhs(u_, &mesh, &ops, a, &bc), dt);
            }

            u.l2_error(&mesh, &ops, exact_solution(t_final))
        })
        .collect();

    println!("Periodic BC convergence (P3):");
    for (i, (&n, &err)) in resolutions.iter().zip(errors.iter()).enumerate() {
        if i > 0 {
            let ratio = errors[i - 1] / err;
            let observed_order = ratio.log2();
            println!(
                "  n={:3}: error={:.4e}, ratio={:.2}, order={:.2}",
                n, err, ratio, observed_order
            );
        } else {
            println!("  n={:3}: error={:.4e}", n, err);
        }
    }

    // Check convergence rate for last refinement
    let ratio = errors[errors.len() - 2] / errors[errors.len() - 1];
    let observed_order = ratio.log2();
    assert!(
        observed_order > 3.5,
        "P3 with periodic BC should be at least 4th order, observed {:.2}",
        observed_order
    );
}

#[test]
fn test_conservation_periodic() {
    // With periodic BCs, the total integral should be conserved
    let order = 3;
    let n_elements = 20;
    let t_final = 2.0; // Full period
    let a = 1.0;
    let cfl = 0.3;

    let mesh = Mesh1D::uniform_periodic(0.0, 2.0, n_elements);
    let ops = DGOperators1D::new(order);

    // Initial condition: sin(pi * x) - has zero integral on [0, 2]
    let initial_condition = |x: f64| (PI * x).sin();
    let mut u = DGSolution1D::new(n_elements, ops.n_nodes);
    u.set_from_function(&mesh, &ops, initial_condition);

    let initial_integral = u.integrate(&mesh, &ops);
    println!("Initial integral: {:.6e}", initial_integral);

    // Time stepping
    let dt = compute_dt(&mesh, a, order, cfl);
    let n_steps = (t_final / dt).ceil() as usize;
    let dt = t_final / n_steps as f64;

    let bc = BoundaryCondition::default();

    for _ in 0..n_steps {
        ssp_rk3_step(&mut u, |u_| compute_rhs(u_, &mesh, &ops, a, &bc), dt);
    }

    let final_integral = u.integrate(&mesh, &ops);
    println!("Final integral:   {:.6e}", final_integral);
    println!(
        "Change:           {:.6e}",
        (final_integral - initial_integral).abs()
    );

    // Integral should be conserved (up to machine precision)
    assert!(
        (final_integral - initial_integral).abs() < 1e-10,
        "Integral should be conserved: initial={:.6e}, final={:.6e}",
        initial_integral,
        final_integral
    );
}

#[test]
fn test_conservation_nonzero_integral() {
    // Test with a function that has non-zero integral
    let order = 3;
    let n_elements = 20;
    let t_final = 1.0;
    let a = 1.0;
    let cfl = 0.3;

    let mesh = Mesh1D::uniform_periodic(0.0, 2.0, n_elements);
    let ops = DGOperators1D::new(order);

    // Initial condition: 1 + 0.5*sin(pi * x) - has integral = 2
    let initial_condition = |x: f64| 1.0 + 0.5 * (PI * x).sin();
    let mut u = DGSolution1D::new(n_elements, ops.n_nodes);
    u.set_from_function(&mesh, &ops, initial_condition);

    let initial_integral = u.integrate(&mesh, &ops);
    println!("Initial integral (should be ~2): {:.6e}", initial_integral);

    // Time stepping
    let dt = compute_dt(&mesh, a, order, cfl);
    let n_steps = (t_final / dt).ceil() as usize;
    let dt = t_final / n_steps as f64;

    let bc = BoundaryCondition::default();

    for _ in 0..n_steps {
        ssp_rk3_step(&mut u, |u_| compute_rhs(u_, &mesh, &ops, a, &bc), dt);
    }

    let final_integral = u.integrate(&mesh, &ops);
    println!("Final integral:   {:.6e}", final_integral);

    // Integral should be conserved
    assert!(
        (final_integral - initial_integral).abs() < 1e-10,
        "Integral should be conserved: initial={:.6e}, final={:.6e}",
        initial_integral,
        final_integral
    );
}

// ============================================================================
// 2D Advection Convergence Tests
// ============================================================================

/// Run a 2D advection simulation with periodic BCs and return L2 error.
fn run_advection_2d(
    n: usize,
    distorted: bool,
    order: usize,
    t_final: f64,
    a: (f64, f64),
    cfl: f64,
) -> f64 {
    let mesh = periodic_mesh(2.0, n, distorted);
    let ops = DGOperators2D::new(order);
    let geom = GeometricFactors2D::compute(&mesh, &ops);
    let equation = Advection2D::new(a.0, a.1);

    // Initial condition: sin(π x) * sin(π y)
    let initial_condition = |x: f64, y: f64| (PI * x).sin() * (PI * y).sin();

    let mut u = DGSolution2D::new(mesh.n_elements, ops.n_nodes);
    u.set_from_function(&mesh, &ops, initial_condition);

    // Exact solution at time t (advected initial condition)
    let exact_solution =
        |t: f64| move |x: f64, y: f64| (PI * (x - a.0 * t)).sin() * (PI * (y - a.1 * t)).sin();

    // Time stepping
    let dt = compute_dt_advection_2d(&mesh, &geom, &equation, order, cfl);
    let n_steps = (t_final / dt).ceil() as usize;
    let dt = t_final / n_steps as f64;

    let bc = PeriodicBC2D;

    let mut t = 0.0;
    for _ in 0..n_steps {
        ssp_rk3_step_2d_timed(
            &mut u,
            |u_, stage_time| {
                compute_rhs_advection_2d(
                    u_,
                    &mesh,
                    &ops,
                    &geom,
                    &equation,
                    &bc,
                    AdvectionFluxType::Upwind,
                    stage_time,
                )
            },
            t,
            dt,
        );
        t += dt;
    }

    u.l2_error(&mesh, &ops, &geom, exact_solution(t_final))
}

#[test]
fn test_convergence_2d_p2() {
    // P2 (order 2) -> expect 3rd order convergence
    let order = 2;
    let t_final = 0.5;
    let a = (1.0, 0.5); // Diagonal velocity
    let cfl = 0.2;

    let resolutions = [4, 8, 16];
    let errors: Vec<f64> = resolutions
        .iter()
        .map(|&n| run_advection_2d(n, false, order, t_final, a, cfl))
        .collect();

    println!("\n2D P2 convergence:");
    for (i, (&n, &err)) in resolutions.iter().zip(errors.iter()).enumerate() {
        if i > 0 {
            let ratio = errors[i - 1] / err;
            let observed_order = ratio.log2();
            println!(
                "  n={:3}: error={:.4e}, ratio={:.2}, order={:.2}",
                n, err, ratio, observed_order
            );
        } else {
            println!("  n={:3}: error={:.4e}", n, err);
        }
    }

    // Check convergence rate for last refinement
    let ratio = errors[errors.len() - 2] / errors[errors.len() - 1];
    let observed_order = ratio.log2();

    assert!(
        observed_order > 2.5,
        "2D P2 should be at least 3rd order, observed {:.2}",
        observed_order
    );
}

#[test]
fn test_convergence_2d_p3() {
    // P3 (order 3) -> expect 4th order convergence
    let order = 3;
    let t_final = 0.2; // Short time to minimize temporal error
    let a = (1.0, 1.0);
    let cfl = 0.1;

    let resolutions = [4, 8, 16];
    let errors: Vec<f64> = resolutions
        .iter()
        .map(|&n| run_advection_2d(n, false, order, t_final, a, cfl))
        .collect();

    println!("\n2D P3 convergence:");
    for (i, (&n, &err)) in resolutions.iter().zip(errors.iter()).enumerate() {
        if i > 0 {
            let ratio = errors[i - 1] / err;
            let observed_order = ratio.log2();
            println!(
                "  n={:3}: error={:.4e}, ratio={:.2}, order={:.2}",
                n, err, ratio, observed_order
            );
        } else {
            println!("  n={:3}: error={:.4e}", n, err);
        }
    }

    // Check convergence rate for last refinement
    let ratio = errors[errors.len() - 2] / errors[errors.len() - 1];
    let observed_order = ratio.log2();

    assert!(
        observed_order > 3.5,
        "2D P3 should be at least 4th order, observed {:.2}",
        observed_order
    );
}

#[test]
fn test_conservation_2d_periodic() {
    // Test mass conservation with periodic BCs in 2D
    let order = 3;
    let n = 8;
    let t_final = 1.0;
    let a = (1.0, 0.5);
    let cfl = 0.2;

    let mesh = Mesh2D::uniform_periodic(0.0, 2.0, 0.0, 2.0, n, n);
    let ops = DGOperators2D::new(order);
    let geom = GeometricFactors2D::compute(&mesh, &ops);
    let equation = Advection2D::new(a.0, a.1);

    // Initial condition with non-zero integral
    let mut u = DGSolution2D::new(mesh.n_elements, ops.n_nodes);
    u.set_from_function(&mesh, &ops, |x, y| {
        1.0 + 0.5 * (PI * x).sin() * (PI * y).sin()
    });

    let initial_integral = u.integrate(&ops, &geom);
    println!("\n2D conservation test:");
    println!("  Initial integral: {:.6e}", initial_integral);

    // Time stepping
    let dt = compute_dt_advection_2d(&mesh, &geom, &equation, order, cfl);
    let n_steps = (t_final / dt).ceil() as usize;
    let dt = t_final / n_steps as f64;

    let bc = PeriodicBC2D;

    for _ in 0..n_steps {
        ssp_rk3_step_2d_timed(
            &mut u,
            |u_, t| {
                compute_rhs_advection_2d(
                    u_,
                    &mesh,
                    &ops,
                    &geom,
                    &equation,
                    &bc,
                    AdvectionFluxType::Upwind,
                    t,
                )
            },
            0.0, // Time doesn't matter for periodic BC
            dt,
        );
    }

    let final_integral = u.integrate(&ops, &geom);
    println!("  Final integral:   {:.6e}", final_integral);
    println!(
        "  Change:           {:.6e}",
        (final_integral - initial_integral).abs()
    );

    // Integral should be conserved (with some tolerance for numerical errors)
    assert!(
        (final_integral - initial_integral).abs() < 1e-10,
        "2D integral should be conserved: initial={:.6e}, final={:.6e}",
        initial_integral,
        final_integral
    );
}

#[test]
fn test_advection_2d_x_direction_only() {
    // Pure x-direction advection (easier case)
    let order = 2;
    let n = 8;
    let t_final = 0.5;
    let a = (1.0, 0.0);
    let cfl = 0.2;

    let mesh = Mesh2D::uniform_periodic(0.0, 2.0, 0.0, 2.0, n, n);
    let ops = DGOperators2D::new(order);
    let geom = GeometricFactors2D::compute(&mesh, &ops);
    let equation = Advection2D::new(a.0, a.1);

    // Initial condition: sin(π x) (varies only in x)
    let mut u = DGSolution2D::new(mesh.n_elements, ops.n_nodes);
    u.set_from_function(&mesh, &ops, |x, _y| (PI * x).sin());

    // Exact solution: sin(π (x - t))
    let exact = |x: f64, _y: f64| (PI * (x - t_final)).sin();

    // Time stepping
    let dt = compute_dt_advection_2d(&mesh, &geom, &equation, order, cfl);
    let n_steps = (t_final / dt).ceil() as usize;
    let dt = t_final / n_steps as f64;

    let bc = PeriodicBC2D;

    for _ in 0..n_steps {
        ssp_rk3_step_2d_timed(
            &mut u,
            |u_, t| {
                compute_rhs_advection_2d(
                    u_,
                    &mesh,
                    &ops,
                    &geom,
                    &equation,
                    &bc,
                    AdvectionFluxType::Upwind,
                    t,
                )
            },
            0.0,
            dt,
        );
    }

    let error = u.l2_error(&mesh, &ops, &geom, exact);
    println!("\n2D x-direction advection error: {:.4e}", error);

    assert!(
        error < 0.1,
        "Pure x-direction advection should have small error, got {}",
        error
    );
}

// ============================================================================
// 2D Shallow Water Equations Convergence Tests
// ============================================================================

/// Run a 2D SWE simulation with linearized wave on periodic domain.
///
/// Uses a small-amplitude wave (ε=1e-4) so the linearized SWE solution
/// is accurate to O(ε²)=O(1e-8), well below spatial discretization error.
fn run_swe_2d_convergence(nx: usize, ny: usize, order: usize, t_final: f64, cfl: f64) -> f64 {
    let g: f64 = 9.81;
    let h0: f64 = 10.0; // Base depth
    let eps: f64 = 1e-4; // Small perturbation amplitude
    let l: f64 = 2.0; // Domain size

    // Wave parameters
    let kx = 2.0 * PI / l;
    let ky = 2.0 * PI / l;
    let k = (kx * kx + ky * ky).sqrt();
    let omega = (g * h0).sqrt() * k; // Dispersion relation

    // Setup
    let mesh = Mesh2D::uniform_periodic(0.0, l, 0.0, l, nx, ny);
    let ops = DGOperators2D::new(order);
    let geom = GeometricFactors2D::compute(&mesh, &ops);
    let equation = ShallowWater2D::new(g);
    let bc = Reflective2D::new(); // Never called on periodic mesh
    let config = SWE2DRhsConfig::new(&equation, &bc)
        .with_flux_type(SWEFluxType2D::Roe)
        .with_coriolis(false);

    // Exact solution (linearized SWE)
    let exact_h = |x: f64, y: f64, t: f64| h0 + eps * (kx * x + ky * y - omega * t).sin();
    let exact_u =
        |x: f64, y: f64, t: f64| eps * g * kx / omega * (kx * x + ky * y - omega * t).sin();
    let exact_v =
        |x: f64, y: f64, t: f64| eps * g * ky / omega * (kx * x + ky * y - omega * t).sin();

    // Initialize
    let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
    q.set_from_functions(
        &mesh,
        &ops,
        |x, y| exact_h(x, y, 0.0),
        |x, y| exact_u(x, y, 0.0),
        |x, y| exact_v(x, y, 0.0),
    );

    // Compute initial time step
    let dt_init = compute_dt_swe_2d(&q, &mesh, &geom, &equation, order, cfl);
    let n_steps = (t_final / dt_init).ceil() as usize;
    let dt = t_final / n_steps as f64;

    // SSP-RK3 time integration (manual, no limiters for smooth solution)
    for _ in 0..n_steps {
        let q0 = q.clone();

        // Stage 1: q = q + dt * L(q)
        let rhs = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);
        q.axpy(dt, &rhs);

        // Stage 2: q = 3/4*q0 + 1/4*(q + dt*L(q))
        let rhs = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);
        q.axpy(dt, &rhs);
        q.scale(0.25);
        q.axpy(0.75, &q0);

        // Stage 3: q = 1/3*q0 + 2/3*(q + dt*L(q))
        let rhs = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, 0.0);
        q.axpy(dt, &rhs);
        q.scale(2.0 / 3.0);
        q.axpy(1.0 / 3.0, &q0);
    }

    // Compute L2 error in depth
    q.l2_error_depth(&mesh, &ops, &geom, |x, y| exact_h(x, y, t_final))
}

#[test]
fn test_convergence_swe_2d_p1() {
    // P1 (order 1) → expect at least 2nd order convergence (threshold 1.5)
    let order = 1;
    let t_final = 0.05;
    let cfl = 0.2;

    let resolutions = [4, 8, 16];
    let errors: Vec<f64> = resolutions
        .iter()
        .map(|&n| run_swe_2d_convergence(n, n, order, t_final, cfl))
        .collect();

    println!("\nSWE 2D P1 convergence:");
    for (i, (&n, &err)) in resolutions.iter().zip(errors.iter()).enumerate() {
        if i > 0 {
            let ratio = errors[i - 1] / err;
            let observed_order = ratio.log2();
            println!(
                "  n={:3}: error={:.4e}, ratio={:.2}, order={:.2}",
                n, err, ratio, observed_order
            );
        } else {
            println!("  n={:3}: error={:.4e}", n, err);
        }
    }

    // Check convergence rate for last refinement
    let ratio = errors[errors.len() - 2] / errors[errors.len() - 1];
    let observed_order = ratio.log2();

    assert!(
        observed_order > 1.5,
        "SWE 2D P1 should converge at order > 1.5, observed {:.2}",
        observed_order
    );
}

#[test]
fn test_convergence_swe_2d_p2() {
    // P2 (order 2) → expect at least 3rd order convergence (threshold 2.5)
    let order = 2;
    let t_final = 0.02;
    let cfl = 0.1;

    let resolutions = [4, 8, 16];
    let errors: Vec<f64> = resolutions
        .iter()
        .map(|&n| run_swe_2d_convergence(n, n, order, t_final, cfl))
        .collect();

    println!("\nSWE 2D P2 convergence:");
    for (i, (&n, &err)) in resolutions.iter().zip(errors.iter()).enumerate() {
        if i > 0 {
            let ratio = errors[i - 1] / err;
            let observed_order = ratio.log2();
            println!(
                "  n={:3}: error={:.4e}, ratio={:.2}, order={:.2}",
                n, err, ratio, observed_order
            );
        } else {
            println!("  n={:3}: error={:.4e}", n, err);
        }
    }

    // Check convergence rate for last refinement
    let ratio = errors[errors.len() - 2] / errors[errors.len() - 1];
    let observed_order = ratio.log2();

    assert!(
        observed_order > 2.5,
        "SWE 2D P2 should converge at order > 2.5, observed {:.2}",
        observed_order
    );
}

// ============================================================================
// 1D Shallow Water Equations Convergence Tests
// ============================================================================

/// Manufactured solution for the nonlinear 1D SWE on a flat bottom:
/// h = H0 + A·sin(k(x − C·t)), u = U (constant), so hu = U·h.
///
/// With h_t = −C·h_x, the forcing that makes it exact is
/// S_h  = h_t + (hu)_x               = (U − C)·h_x
/// S_hu = (hu)_t + (hu² + ½gh²)_x    = (U² − U·C + g·h)·h_x
struct SweManufactured1D {
    g: f64,
    h0: f64,
    amplitude: f64,
    k: f64,
    c: f64,
    u: f64,
}

impl SweManufactured1D {
    fn depth(&self, x: f64, t: f64) -> f64 {
        self.h0 + self.amplitude * (self.k * (x - self.c * t)).sin()
    }
}

impl SourceTerm for SweManufactured1D {
    fn evaluate(&self, _state: &SWEState, _db_dx: f64, x: f64, t: f64) -> SWEState {
        let h = self.depth(x, t);
        let h_x = self.amplitude * self.k * (self.k * (x - self.c * t)).cos();
        SWEState::new(
            (self.u - self.c) * h_x,
            (self.u * self.u - self.u * self.c + self.g * h) * h_x,
        )
    }

    fn name(&self) -> &'static str {
        "swe_1d_manufactured"
    }
}

/// Run the manufactured 1D SWE problem on a periodic domain and return the L2
/// error in depth at `t_final`.
///
/// The solution is exact for the nonlinear equations, so there is no
/// linearization error floor.
fn run_swe_1d_convergence(
    n_elements: usize,
    order: usize,
    flux_type: SWEFluxType,
    t_final: f64,
    cfl: f64,
) -> f64 {
    let l: f64 = 2.0;
    let mms = SweManufactured1D {
        g: 9.81,
        h0: 2.0,
        amplitude: 0.2,
        k: 2.0 * PI / l,
        c: 1.0,
        u: 0.5,
    };

    let mesh = Mesh1D::uniform_periodic(0.0, l, n_elements);
    let ops = DGOperators1D::new(order);
    let equation = ShallowWater1D::new(mms.g);
    let bc = ReflectiveBC::new(); // Never called on a periodic mesh
    let config = SWERhsConfig::new(&equation, &bc, &bc)
        .with_flux_type(flux_type)
        .with_source(&mms);

    let mut q = SWESolution::new(mesh.n_elements, ops.n_nodes);
    q.set_from_functions(&mesh, &ops, |x| mms.depth(x, 0.0), |_| mms.u);

    let dt_init = compute_dt_swe(&q, &mesh, &equation, order, cfl);
    let n_steps = (t_final / dt_init).ceil() as usize;
    let dt = t_final / n_steps as f64;

    let mut t = 0.0;
    for _ in 0..n_steps {
        SSPRK3.step(&mut q, dt, t, |q, t| {
            compute_rhs_swe(q, &mesh, &ops, &config, t)
        });
        t += dt;
    }

    q.l2_error_depth(&mesh, &ops, |x| mms.depth(x, t_final))
}

/// Assert that the 1D SWE solver converges at least at `min_order` for `order`.
fn check_swe_1d_convergence(order: usize, t_final: f64, cfl: f64, min_order: f64) {
    for flux_type in [
        SWEFluxType::Roe,
        SWEFluxType::Hll,
        SWEFluxType::LaxFriedrichs,
    ] {
        let resolutions = [8, 16, 32];
        let errors: Vec<f64> = resolutions
            .iter()
            .map(|&n| run_swe_1d_convergence(n, order, flux_type, t_final, cfl))
            .collect();

        println!("\nSWE 1D P{order} convergence ({flux_type:?}):");
        for (i, (&n, &err)) in resolutions.iter().zip(errors.iter()).enumerate() {
            if i > 0 {
                let observed_order = (errors[i - 1] / err).log2();
                println!("  n={n:3}: error={err:.4e}, order={observed_order:.2}");
            } else {
                println!("  n={n:3}: error={err:.4e}");
            }
        }

        let observed_order = (errors[errors.len() - 2] / errors[errors.len() - 1]).log2();
        assert!(
            observed_order > min_order,
            "SWE 1D P{order} ({flux_type:?}) should converge at order > {min_order}, \
             observed {observed_order:.2} (errors {errors:?})"
        );
    }
}

#[test]
fn test_convergence_swe_1d_p1() {
    check_swe_1d_convergence(1, 0.5, 0.3, 1.8);
}

#[test]
fn test_convergence_swe_1d_p2() {
    check_swe_1d_convergence(2, 0.5, 0.2, 2.7);
}

#[test]
fn test_convergence_swe_1d_p3() {
    check_swe_1d_convergence(3, 0.5, 0.1, 3.7);
}

// ============================================================================
// 2D SWE split-form (Wintermeyer et al. 2017) convergence with bathymetry
// ============================================================================

/// Manufactured nonlinear solution over variable bathymetry on [0, √2]²:
/// H = h + b = 7 + cos(ω x)·sin(ω y)·cos(2πt), ω = 2π√2,
/// b = 2 + ½ sin(√2 π x) + ½ sin(√2 π y), (u, v) = (½, 3/2).
mod manufactured {
    use std::f64::consts::{PI, SQRT_2};

    pub const G: f64 = 9.81;
    pub const U: f64 = 0.5;
    pub const V: f64 = 1.5;
    const OMEGA_X: f64 = 2.0 * PI * SQRT_2;
    const OMEGA_T: f64 = 2.0 * PI;

    pub fn bed(x: f64, y: f64) -> f64 {
        2.0 + 0.5 * (SQRT_2 * PI * x).sin() + 0.5 * (SQRT_2 * PI * y).sin()
    }

    fn bed_gradient(x: f64, y: f64) -> (f64, f64) {
        let c = 0.5 * SQRT_2 * PI;
        (c * (SQRT_2 * PI * x).cos(), c * (SQRT_2 * PI * y).cos())
    }

    fn surface(x: f64, y: f64, t: f64) -> f64 {
        7.0 + (OMEGA_X * x).cos() * (OMEGA_X * y).sin() * (OMEGA_T * t).cos()
    }

    pub fn depth(x: f64, y: f64, t: f64) -> f64 {
        surface(x, y, t) - bed(x, y)
    }

    /// S = ∂q/∂t + ∇·F(q) + g h (0, ∇b) for the exact solution:
    /// S_h = H_t + u h_x + v h_y, S_hu = u S_h + g h H_x, S_hv = v S_h + g h H_y.
    pub fn source(x: f64, y: f64, t: f64) -> (f64, f64, f64) {
        let (cx, sx) = ((OMEGA_X * x).cos(), (OMEGA_X * x).sin());
        let (cy, sy) = ((OMEGA_X * y).cos(), (OMEGA_X * y).sin());
        let (ct, st) = ((OMEGA_T * t).cos(), (OMEGA_T * t).sin());
        let h_t = -OMEGA_T * cx * sy * st;
        let surface_x = -OMEGA_X * sx * sy * ct;
        let surface_y = OMEGA_X * cx * cy * ct;
        let (b_x, b_y) = bed_gradient(x, y);
        let h = depth(x, y, t);

        let s_h = h_t + U * (surface_x - b_x) + V * (surface_y - b_y);
        (
            s_h,
            U * s_h + G * h * surface_x,
            V * s_h + G * h * surface_y,
        )
    }
}

struct ManufacturedSource;

impl dgcom::SourceTerm2D for ManufacturedSource {
    fn evaluate(&self, ctx: &dgcom::SourceContext2D) -> dgcom::SWEState2D {
        let (x, y) = ctx.position;
        let (s_h, s_hu, s_hv) = manufactured::source(x, y, ctx.time);
        dgcom::SWEState2D::new(s_h, s_hu, s_hv)
    }

    fn name(&self) -> &'static str {
        "manufactured_swe_2d"
    }
}

/// An n×n periodic mesh of [0, l]², uniform or with its vertices moved by a
/// smooth periodic field (general quadrilaterals, TODO P1.3).
fn periodic_mesh(l: f64, n: usize, distorted: bool) -> Mesh2D {
    let mut mesh = Mesh2D::uniform_periodic(0.0, l, 0.0, l, n, n);
    if distorted {
        let tau = 2.0 * PI / l;
        let a = 0.04 * l;
        for v in &mut mesh.vertices {
            let [x, y] = *v;
            let (sx, sy) = ((tau * x).sin(), (tau * y).sin());
            *v = [
                x + a * sx * sy + 0.5 * a * (tau * y).sin(),
                y - 0.7 * a * sx * sy + 0.4 * a * (tau * x).sin(),
            ];
        }
    }
    mesh
}

/// L2 depth error of a split form on an n×n periodic mesh (`WetDry`: subcell
/// finite volumes in elements with a node shallower than `subcell_depth`).
fn run_swe_2d_split_form_manufactured(
    formulation: dgcom::SWEFormulation2D,
    subcell_depth: f64,
    distorted: bool,
    n: usize,
    order: usize,
    t_final: f64,
    cfl: f64,
) -> f64 {
    use dgcom::mesh::Bathymetry2D;
    use manufactured::{G, U, V, bed, depth};

    let l = std::f64::consts::SQRT_2;
    let mesh = periodic_mesh(l, n, distorted);
    let ops = DGOperators2D::new(order);
    let geom = GeometricFactors2D::compute(&mesh, &ops);
    let equation = ShallowWater2D::new(G);
    let bc = Reflective2D::new(); // Never called on periodic mesh
    let bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, bed);
    let source = ManufacturedSource;
    let config = SWE2DRhsConfig::new(&equation, &bc)
        .with_coriolis(false)
        .with_formulation(formulation)
        .with_bathymetry(&bathymetry)
        .with_source_terms(&source)
        .with_subcell_depth(subcell_depth);

    let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
    q.set_from_functions(&mesh, &ops, |x, y| depth(x, y, 0.0), |_, _| U, |_, _| V);

    let dt_init = compute_dt_swe_2d(&q, &mesh, &geom, &equation, order, cfl);
    let n_steps = (t_final / dt_init).ceil() as usize;
    let dt = t_final / n_steps as f64;

    // SSP-RK3 with the correct stage times for the time-dependent source
    let mut t = 0.0;
    for _ in 0..n_steps {
        let q0 = q.clone();

        let rhs = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, t);
        q.axpy(dt, &rhs);

        let rhs = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, t + dt);
        q.axpy(dt, &rhs);
        q.scale(0.25);
        q.axpy(0.75, &q0);

        let rhs = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config, t + 0.5 * dt);
        q.axpy(dt, &rhs);
        q.scale(2.0 / 3.0);
        q.axpy(1.0 / 3.0, &q0);

        t += dt;
    }

    q.l2_error_depth(&mesh, &ops, &geom, |x, y| depth(x, y, t_final))
}

fn check_split_form_convergence(
    formulation: dgcom::SWEFormulation2D,
    order: usize,
    cfl: f64,
    min_order: f64,
) {
    check_split_form_convergence_with(formulation, 0.0, false, &[4, 8, 16], order, cfl, min_order);
}

fn check_split_form_convergence_with(
    formulation: dgcom::SWEFormulation2D,
    subcell_depth: f64,
    distorted: bool,
    resolutions: &[usize],
    order: usize,
    cfl: f64,
    min_order: f64,
) {
    let t_final = 0.1;
    let errors: Vec<f64> = resolutions
        .iter()
        .map(|&n| {
            run_swe_2d_split_form_manufactured(
                formulation,
                subcell_depth,
                distorted,
                n,
                order,
                t_final,
                cfl,
            )
        })
        .collect();

    println!(
        "\nSWE 2D split form {formulation:?} P{order}, subcell depth {subcell_depth:e}, distorted = {distorted} \
         (manufactured, bathymetry):"
    );
    for (i, (&n, &err)) in resolutions.iter().zip(errors.iter()).enumerate() {
        if i > 0 {
            let observed = (errors[i - 1] / err).log2();
            println!("  n={n:3}: error={err:.4e}, order={observed:.2}");
        } else {
            println!("  n={n:3}: error={err:.4e}");
        }
    }

    let observed = (errors[errors.len() - 2] / errors[errors.len() - 1]).log2();
    assert!(
        observed > min_order,
        "Split-form SWE 2D {formulation:?} P{order} should converge at order > {min_order}, observed {observed:.2}"
    );
}

#[test]
fn test_convergence_swe_2d_split_form_p2() {
    // P2 → 3rd order expected
    check_split_form_convergence(dgcom::SWEFormulation2D::EntropyStable, 2, 0.1, 2.5);
}

#[test]
fn test_convergence_swe_2d_split_form_p3() {
    // P3 → 4th order expected
    check_split_form_convergence(dgcom::SWEFormulation2D::EntropyStable, 3, 0.05, 3.5);
}

#[test]
fn test_convergence_swe_2d_wet_dry_split_form() {
    // Fully wet: flux differencing with hydrostatic-HLL faces keeps N+1
    check_split_form_convergence(dgcom::SWEFormulation2D::WetDry, 2, 0.1, 2.5);
    check_split_form_convergence(dgcom::SWEFormulation2D::WetDry, 3, 0.05, 3.5);
}

#[test]
fn test_convergence_swe_2d_wet_dry_subcells() {
    // Subcell finite volumes forced in every element (an infinite subcell
    // depth; the nodes are wet, so the surface is reconstructed): the limited linear reconstruction on the
    // GLL subcells converges at second order for any N. Measured 8 → 16:
    // P1 1.90, P2 1.87, P3 1.92 (1.96 / 1.90 / 2.04 from 16 → 32). Without the
    // reconstruction: P2 1.37, then 0.68 from 16 → 32.
    for order in 1..=3 {
        check_split_form_convergence_with(
            dgcom::SWEFormulation2D::WetDry,
            f64::INFINITY,
            false,
            &[4, 8, 16],
            order,
            0.2,
            1.7,
        );
    }
}

// ---------------------------------------------------------------------------
// General (non-parallelogram) quadrilaterals, TODO P1.3
// ---------------------------------------------------------------------------

#[test]
fn test_convergence_swe_2d_split_form_distorted_mesh() {
    // Metric at every node, curvilinear split form (Wintermeyer et al. 2017):
    // N+1 on distorted periodic meshes as on uniform ones. Measured 8 → 16:
    // EntropyStable P2 2.95, P3 3.88; WetDry (fully wet) P2 2.95, P3 3.89
    for (formulation, order, cfl, min_order) in [
        (dgcom::SWEFormulation2D::EntropyStable, 2, 0.1, 2.5),
        (dgcom::SWEFormulation2D::EntropyStable, 3, 0.05, 3.5),
        (dgcom::SWEFormulation2D::WetDry, 2, 0.1, 2.5),
        (dgcom::SWEFormulation2D::WetDry, 3, 0.05, 3.5),
    ] {
        check_split_form_convergence_with(
            formulation,
            0.0,
            true,
            &[4, 8, 16],
            order,
            cfl,
            min_order,
        );
    }
}

#[test]
fn test_convergence_swe_2d_wet_dry_subcells_distorted_mesh() {
    // Subcells in every element with the interface metrics and the metric
    // balance term: second order on distorted meshes. Measured 8 → 16:
    // P1 1.92, P2 1.74, P3 1.86; P2 16 → 32: 1.91 (uniform mesh: 1.90)
    for order in 1..=3 {
        check_split_form_convergence_with(
            dgcom::SWEFormulation2D::WetDry,
            f64::INFINITY,
            true,
            &[4, 8, 16],
            order,
            0.2,
            1.7,
        );
    }
}

#[test]
fn test_convergence_advection_2d_distorted_mesh() {
    // Conservative form with the metric at every node: N+1 on distorted
    // periodic meshes
    for (order, t_final, cfl, min_order) in [(2, 0.5, 0.2, 2.5), (3, 0.2, 0.1, 3.5)] {
        let resolutions = [4, 8, 16];
        let errors: Vec<f64> = resolutions
            .iter()
            .map(|&n| run_advection_2d(n, true, order, t_final, (1.0, 0.5), cfl))
            .collect();
        println!("\n2D advection P{order}, distorted mesh:");
        for (i, (&n, &err)) in resolutions.iter().zip(errors.iter()).enumerate() {
            let rate = if i > 0 {
                (errors[i - 1] / err).log2()
            } else {
                f64::NAN
            };
            println!("  n={n:3}: error={err:.4e}, order={rate:.2}");
        }
        let observed = (errors[1] / errors[2]).log2();
        assert!(
            observed > min_order,
            "2D advection P{order} on a distorted mesh: order {observed:.2} ≤ {min_order}"
        );
    }
}

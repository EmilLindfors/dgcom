//! 1D advection example using DG.
//!
//! Solves du/dt + a * du/dx = 0 on [0, 2] with:
//! - Initial condition: u(x, 0) = sin(pi * x)
//! - Advection velocity: a = 1
//! - Inflow BC at left (Dirichlet), outflow BC at right
//!
//! After time t, exact solution is u(x, t) = sin(pi * (x - a*t))

use dgcom::{
    BoundaryCondition, DGOperators1D, DGSolution1D, Mesh1D, compute_dt, compute_rhs,
    ssp_rk3_step_timed,
};
use std::f64::consts::PI;

fn main() {
    // Parameters
    let order = 3; // P3 (4th order accuracy)
    let n_elements = 20;
    let x_min = 0.0;
    let x_max = 2.0;
    let a = 1.0; // Advection velocity
    let t_final = 1.0; // Simulate until t = 1
    let cfl = 0.3;

    println!("1D DG Advection Solver");
    println!("======================");
    println!("Order: P{}", order);
    println!("Elements: {}", n_elements);
    println!("Domain: [{}, {}]", x_min, x_max);
    println!("Advection velocity: {}", a);
    println!("Final time: {}", t_final);
    println!();

    // Setup mesh and operators
    let mesh = Mesh1D::uniform(x_min, x_max, n_elements);
    let ops = DGOperators1D::new(order);

    println!("Nodes per element: {}", ops.n_nodes);
    println!("Total DOFs: {}", n_elements * ops.n_nodes);
    println!();

    // Initial condition: sin(pi * x)
    let initial_condition = |x: f64| (PI * x).sin();

    let mut u = DGSolution1D::new(n_elements, ops.n_nodes);
    u.set_from_function(&mesh, &ops, initial_condition);

    // Exact solution at time t
    let exact_solution = |t: f64| move |x: f64| (PI * (x - a * t)).sin();

    // Compute initial error
    let error_0 = u.l2_error(&mesh, &ops, exact_solution(0.0));
    println!("Initial L2 error: {:.2e}", error_0);

    // Time stepping
    let dt = compute_dt(&mesh, a, order, cfl);
    let n_steps = (t_final / dt).ceil() as usize;
    let dt = t_final / n_steps as f64; // Adjust to hit t_final exactly

    println!("Time step: {:.4e}", dt);
    println!("Number of steps: {}", n_steps);
    println!();

    // Time stepping with time-aware RK3 for proper high-order accuracy
    // For a > 0, inflow is at left boundary
    let mut t = 0.0;

    for step in 0..n_steps {
        // SSP-RK3 step with BCs evaluated at correct stage times
        ssp_rk3_step_timed(
            &mut u,
            |u_, stage_time| {
                let bc = BoundaryCondition {
                    left: exact_solution(stage_time)(x_min), // Dirichlet at inflow
                    right: 0.0,                              // Not used (outflow)
                };
                compute_rhs(u_, &mesh, &ops, a, &bc)
            },
            t,
            dt,
        );

        t += dt;

        // Print progress
        if (step + 1) % (n_steps / 5).max(1) == 0 || step == n_steps - 1 {
            let error = u.l2_error(&mesh, &ops, exact_solution(t));
            println!(
                "Step {:5} / {:5}: t = {:.4}, L2 error = {:.4e}",
                step + 1,
                n_steps,
                t,
                error
            );
        }
    }

    // Final error
    println!();
    let final_error = u.l2_error(&mesh, &ops, exact_solution(t_final));
    let linf_error = u.linf_error(&mesh, &ops, exact_solution(t_final));

    println!("Final L2 error:   {:.4e}", final_error);
    println!("Final L∞ error:   {:.4e}", linf_error);
}

//! Integration tests for 2D Shallow Water Equations solver.
//!
//! These tests verify:
//! - Lake-at-rest (well-balanced property)
//! - Mass conservation
//! - Dam break evolution
//! - Coriolis effects

use dgcom::types::ElementIndex;
use dgcom::{
    DGOperators2D, GeometricFactors2D, Mesh2D, Reflective2D, SWE2DRhsConfig, SWEFluxType2D,
    SWESolution2D, SWEState2D, ShallowWater2D, compute_dt_swe_2d, compute_rhs_swe_2d,
};

fn k(idx: usize) -> ElementIndex {
    ElementIndex::new(idx)
}

const G: f64 = 10.0;

/// Run SSP-RK3 step for 2D SWE.
fn ssp_rk3_swe_step<BC: dgcom::SWEBoundaryCondition2D>(
    q: &mut SWESolution2D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    config: &SWE2DRhsConfig<BC>,
    dt: f64,
    time: f64,
) {
    // Stage 1: u1 = u + dt * L(u)
    let rhs1 = compute_rhs_swe_2d(q, mesh, ops, geom, config, time);
    let mut u1 = SWESolution2D::new(q.n_elements, q.n_nodes);
    u1.copy_from(q);
    u1.axpy(dt, &rhs1);

    // Stage 2: u2 = 3/4 * u + 1/4 * (u1 + dt * L(u1))
    let rhs2 = compute_rhs_swe_2d(&u1, mesh, ops, geom, config, time + dt);
    u1.axpy(dt, &rhs2);
    let mut u2 = SWESolution2D::new(q.n_elements, q.n_nodes);
    // u2 = 0.75 * q + 0.25 * u1 (using axpy twice)
    u2.copy_from(q);
    u2.scale(0.75);
    u2.axpy(0.25, &u1);

    // Stage 3: u_new = 1/3 * u + 2/3 * (u2 + dt * L(u2))
    let rhs3 = compute_rhs_swe_2d(&u2, mesh, ops, geom, config, time + 0.5 * dt);
    u2.axpy(dt, &rhs3);
    // q = 1/3 * q + 2/3 * u2 (using scale and axpy)
    q.scale(1.0 / 3.0);
    q.axpy(2.0 / 3.0, &u2);
}

/// Lake at rest over a steep, nodal bed (TODO P1.7): a fjord sill and
/// continental-slope profile from 30 m to 400 m over 10 km, plus bumps, so
/// the bed is far from linear inside the elements. The split forms
/// (`EntropyStable`, `WetDry`) must keep η = 0, u = 0 to round-off at P1–P4
/// with walls, serial and parallel. The collocated `Standard` form with
/// hydrostatic reconstruction is balanced only for beds of degree ≤ p/2 and
/// serves as the negative control. (This test used a flat bottom.)
#[test]
fn test_lake_at_rest() {
    use dgcom::mesh::Bathymetry2D;
    use dgcom::solver::SWEFormulation2D;
    use dgcom::source::BathymetrySource2D;

    let bed = |x: f64, y: f64| {
        let slope = 0.5 * (1.0 + ((x - 5_000.0) / 1_200.0).tanh()); // 0 → 1 across the slope
        -30.0 - 370.0 * slope + 20.0 * (x / 700.0).sin() * (y / 900.0).cos()
    };
    let equation = ShallowWater2D::new(G);
    let bc = Reflective2D::new();
    let bathy_source = BathymetrySource2D::new(G);

    for order in 1..=4 {
        let mesh = Mesh2D::uniform_rectangle(0.0, 10_000.0, 0.0, 4_000.0, 10, 4);
        let ops = DGOperators2D::new(order);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, bed);
        let (depths, _) = bathymetry
            .data
            .iter()
            .fold((f64::INFINITY, 0.0), |(lo, hi): (f64, f64), &b| {
                (lo.min(-b), hi.max(-b))
            });
        assert!(depths < 40.0, "shallowest {depths}");
        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for ki in 0..mesh.n_elements {
            for i in 0..ops.n_nodes {
                let h = -bathymetry.get(k(ki), i);
                q.set_state(k(ki), i, SWEState2D::new(h, 0.0, 0.0));
            }
        }
        let rates = |config: &SWE2DRhsConfig<Reflective2D>| {
            let serial = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, config, 0.0);
            #[cfg(feature = "parallel")]
            {
                let parallel =
                    dgcom::compute_rhs_swe_2d_parallel(&q, &mesh, &ops, &geom, config, 0.0);
                assert_eq!(serial.data, parallel.data, "p={order}: serial ≠ parallel");
            }
            let max = |v: &[f64]| v.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
            (
                max(serial.h_data()),
                max(serial.hu_data()).max(max(serial.hv_data())),
            )
        };

        for formulation in [SWEFormulation2D::EntropyStable, SWEFormulation2D::WetDry] {
            let config = SWE2DRhsConfig::new(&equation, &bc)
                .with_coriolis(false)
                .with_formulation(formulation)
                .with_bathymetry(&bathymetry);
            let (dh, dm) = rates(&config);
            // d(hu)/dt of 1e-10 m²/s² is 2.5e-13 m/s² of spurious
            // acceleration in 400 m of water
            assert!(
                dh < 1e-12 && dm < 1e-10,
                "p={order}, {formulation:?}: max |dh/dt| = {dh:.2e}, max |d(hu)/dt| = {dm:.2e}"
            );
        }

        // Negative control: the collocated form is not balanced on this bed
        let config = SWE2DRhsConfig::new(&equation, &bc)
            .with_coriolis(false)
            .with_bathymetry(&bathymetry)
            .with_source_terms(&bathy_source)
            .with_well_balanced(true);
        let (_, dm) = rates(&config);
        assert!(dm > 1e-4, "p={order}, Standard: max |d(hu)/dt| = {dm:.2e}");
    }
}

/// Test lake-at-rest with perturbation.
///
/// Verifies that small perturbations evolve correctly.
#[test]
fn test_lake_at_rest_perturbation() {
    let mesh = Mesh2D::uniform_periodic(0.0, 10.0, 0.0, 10.0, 8, 8);
    let ops = DGOperators2D::new(2);
    let geom = GeometricFactors2D::compute(&mesh, &ops);
    let equation = ShallowWater2D::new(G);
    let bc = Reflective2D::new();
    let config = SWE2DRhsConfig::new(&equation, &bc).with_coriolis(false);

    // Initialize: h = 2.0 + perturbation
    let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
    q.set_from_functions(
        &mesh,
        &ops,
        |x, y| {
            2.0 + 0.1
                * (2.0 * std::f64::consts::PI * x / 10.0).sin()
                * (2.0 * std::f64::consts::PI * y / 10.0).sin()
        },
        |_, _| 0.0,
        |_, _| 0.0,
    );

    let initial_mass = q.integrate_depth(&ops, &geom);

    // Take a few time steps
    let cfl = 0.3;
    let mut time = 0.0;
    for _ in 0..10 {
        let dt = compute_dt_swe_2d(&q, &mesh, &geom, &equation, ops.order, cfl);
        ssp_rk3_swe_step(&mut q, &mesh, &ops, &geom, &config, dt, time);
        time += dt;
    }

    // Check mass conservation
    let final_mass = q.integrate_depth(&ops, &geom);
    let mass_error = ((final_mass - initial_mass) / initial_mass).abs();
    assert!(
        mass_error < 1e-12,
        "Mass should be conserved: error = {:.2e}",
        mass_error
    );

    // Check no negative depths
    assert!(!q.has_negative_depth(), "Depth should remain positive");
}

/// Test mass conservation for smooth initial condition.
///
/// Uses a smooth initial condition to test conservation properties
/// without the complications of discontinuities.
#[test]
fn test_mass_conservation_smooth() {
    let mesh = Mesh2D::uniform_periodic(0.0, 10.0, 0.0, 10.0, 8, 8);
    let ops = DGOperators2D::new(2);
    let geom = GeometricFactors2D::compute(&mesh, &ops);
    let equation = ShallowWater2D::new(G);
    let bc = Reflective2D::new();
    let config = SWE2DRhsConfig::new(&equation, &bc).with_coriolis(false);

    // Smooth Gaussian bump initial condition
    let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
    q.set_from_functions(
        &mesh,
        &ops,
        |x, y| {
            let cx = 5.0;
            let cy = 5.0;
            let dist_sq = (x - cx).powi(2) + (y - cy).powi(2);
            2.0 + 0.5 * (-dist_sq / 2.0).exp()
        },
        |_, _| 0.0,
        |_, _| 0.0,
    );

    let initial_mass = q.integrate_depth(&ops, &geom);
    let initial_x_momentum = q.integrate_x_momentum(&ops, &geom);
    let initial_y_momentum = q.integrate_y_momentum(&ops, &geom);

    // Run simulation for a short time
    let cfl = 0.2;
    let mut time = 0.0;
    let end_time = 0.5;
    while time < end_time {
        let dt = compute_dt_swe_2d(&q, &mesh, &geom, &equation, ops.order, cfl);
        let dt = dt.min(end_time - time);
        ssp_rk3_swe_step(&mut q, &mesh, &ops, &geom, &config, dt, time);
        time += dt;

        // Early termination if solution has issues
        if q.max_abs() > 100.0 || q.has_negative_depth() {
            panic!(
                "Solution unstable at t = {}: max = {}, min_h = {}",
                time,
                q.max_abs(),
                q.min_depth()
            );
        }
    }

    // Check mass conservation (should be exact for periodic)
    let final_mass = q.integrate_depth(&ops, &geom);
    let mass_error = ((final_mass - initial_mass) / initial_mass).abs();
    assert!(
        mass_error < 1e-11,
        "Mass should be conserved: error = {:.2e}",
        mass_error
    );

    // Check x-momentum conservation (should be conserved for periodic without Coriolis)
    let final_x_momentum = q.integrate_x_momentum(&ops, &geom);
    let x_mom_error = (final_x_momentum - initial_x_momentum).abs();
    assert!(
        x_mom_error < 1e-10,
        "X-momentum should be conserved: error = {:.2e}",
        x_mom_error
    );

    // Check y-momentum conservation
    let final_y_momentum = q.integrate_y_momentum(&ops, &geom);
    let y_mom_error = (final_y_momentum - initial_y_momentum).abs();
    assert!(
        y_mom_error < 1e-10,
        "Y-momentum should be conserved: error = {:.2e}",
        y_mom_error
    );
}

/// Test circular dam break.
///
/// A circular dam break tests radial symmetry and 2D behavior.
#[test]
fn test_circular_dam_break() {
    let mesh = Mesh2D::uniform_periodic(0.0, 10.0, 0.0, 10.0, 8, 8);
    let ops = DGOperators2D::new(2);
    let geom = GeometricFactors2D::compute(&mesh, &ops);
    let equation = ShallowWater2D::new(G);
    let bc = Reflective2D::new();
    let config = SWE2DRhsConfig::new(&equation, &bc).with_coriolis(false);

    // Circular dam: higher water in center
    let cx = 5.0;
    let cy = 5.0;
    let r = 2.0;
    let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
    q.set_from_functions(
        &mesh,
        &ops,
        |x, y| {
            let dist = ((x - cx).powi(2) + (y - cy).powi(2)).sqrt();
            if dist < r { 3.0 } else { 1.0 }
        },
        |_, _| 0.0,
        |_, _| 0.0,
    );

    let initial_mass = q.integrate_depth(&ops, &geom);

    // Run simulation
    let cfl = 0.2;
    let mut time = 0.0;
    let end_time = 0.3;
    while time < end_time {
        let dt = compute_dt_swe_2d(&q, &mesh, &geom, &equation, ops.order, cfl);
        let dt = dt.min(end_time - time);
        ssp_rk3_swe_step(&mut q, &mesh, &ops, &geom, &config, dt, time);
        time += dt;

        // Early termination if solution blows up
        if q.max_abs() > 100.0 {
            panic!("Solution blew up at t = {}", time);
        }
    }

    // Check mass conservation
    let final_mass = q.integrate_depth(&ops, &geom);
    let mass_error = ((final_mass - initial_mass) / initial_mass).abs();
    assert!(
        mass_error < 1e-11,
        "Mass should be conserved: error = {:.2e}",
        mass_error
    );

    // Check no negative depths
    assert!(
        !q.has_negative_depth(),
        "Depth should remain positive, min = {}",
        q.min_depth()
    );
}

/// Test Coriolis effect on uniform flow.
///
/// Verifies that Coriolis source term is computed correctly
/// by checking that it deflects flow as expected.
#[test]
fn test_coriolis_effect() {
    // Small domain for simplicity
    let mesh = Mesh2D::uniform_periodic(0.0, 10.0, 0.0, 10.0, 4, 4);
    let ops = DGOperators2D::new(2);
    let geom = GeometricFactors2D::compute(&mesh, &ops);

    // Norwegian coast Coriolis
    let f = 1.2e-4;
    let equation = ShallowWater2D::with_coriolis(G, f);
    let bc = Reflective2D::new();

    // Uniform flow in x-direction: should be deflected to the right (positive y)
    let h = 10.0;
    let u = 1.0;
    let v = 0.0;

    let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
    q.set_from_functions(&mesh, &ops, |_, _| h, |_, _| u, |_, _| v);

    // Compute RHS with Coriolis
    let config_with = SWE2DRhsConfig::new(&equation, &bc).with_coriolis(true);
    let rhs_with = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config_with, 0.0);

    // Compute RHS without Coriolis
    let config_without = SWE2DRhsConfig::new(&equation, &bc).with_coriolis(false);
    let rhs_without = compute_rhs_swe_2d(&q, &mesh, &ops, &geom, &config_without, 0.0);

    // The difference should be the Coriolis source term
    // d(hu)/dt += f * hv = f * h * v = 0 (since v = 0)
    // d(hv)/dt -= f * hu = -f * h * u = -1.2e-4 * 10 * 1 = -1.2e-3
    let expected_hv_source = -f * h * u;

    // Check that the hv difference is approximately the expected source
    let diff_hv = rhs_with.get_var(k(0), 0, 2) - rhs_without.get_var(k(0), 0, 2);
    assert!(
        (diff_hv - expected_hv_source).abs() < 1e-8,
        "Coriolis hv source: got {:.2e}, expected {:.2e}",
        diff_hv,
        expected_hv_source
    );

    // Run a few time steps and verify momentum changes direction
    let mut q_with = q.clone();
    let cfl = 0.3;
    let mut time = 0.0;
    let end_time = 100.0; // Need long enough for Coriolis to have visible effect
    while time < end_time {
        let dt = compute_dt_swe_2d(&q_with, &mesh, &geom, &equation, ops.order, cfl);
        let dt = dt.min(end_time - time);
        ssp_rk3_swe_step(&mut q_with, &mesh, &ops, &geom, &config_with, dt, time);
        time += dt;
    }

    // After evolving, should have developed v-momentum (rightward deflection)
    let final_y_momentum = q_with.integrate_y_momentum(&ops, &geom);
    // Coriolis should have deflected momentum to negative v (in Northern Hemisphere)
    assert!(
        final_y_momentum < -1e-6,
        "Coriolis should deflect x-momentum to negative v: got y-momentum = {:.2e}",
        final_y_momentum
    );
}

/// Test different flux types give similar results for smooth solutions.
#[test]
fn test_flux_type_comparison() {
    let mesh = Mesh2D::uniform_periodic(0.0, 10.0, 0.0, 10.0, 4, 4);
    let ops = DGOperators2D::new(2);
    let geom = GeometricFactors2D::compute(&mesh, &ops);
    let equation = ShallowWater2D::new(G);
    let bc = Reflective2D::new();

    // Smooth initial condition
    let create_ic = || {
        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        q.set_from_functions(
            &mesh,
            &ops,
            |x, y| {
                2.0 + 0.2
                    * (2.0 * std::f64::consts::PI * x / 10.0).sin()
                    * (2.0 * std::f64::consts::PI * y / 10.0).sin()
            },
            |_, _| 0.0,
            |_, _| 0.0,
        );
        q
    };

    let flux_types = [
        SWEFluxType2D::Roe,
        SWEFluxType2D::HLL,
        SWEFluxType2D::Rusanov,
    ];

    // Run a few steps with each flux type
    let end_time = 0.1;
    let cfl = 0.2;
    let mut final_masses = Vec::new();
    let mut final_max_h = Vec::new();

    for flux_type in &flux_types {
        let mut q = create_ic();
        let config = SWE2DRhsConfig::new(&equation, &bc)
            .with_flux_type(*flux_type)
            .with_coriolis(false);

        let mut time = 0.0;
        while time < end_time {
            let dt = compute_dt_swe_2d(&q, &mesh, &geom, &equation, ops.order, cfl);
            let dt = dt.min(end_time - time);
            ssp_rk3_swe_step(&mut q, &mesh, &ops, &geom, &config, dt, time);
            time += dt;
        }

        final_masses.push(q.integrate_depth(&ops, &geom));
        final_max_h.push(q.max_depth());
    }

    // All flux types should conserve mass
    let initial_mass = create_ic().integrate_depth(&ops, &geom);
    for (i, &mass) in final_masses.iter().enumerate() {
        let error = ((mass - initial_mass) / initial_mass).abs();
        assert!(
            error < 1e-12,
            "{:?} flux: mass error = {:.2e}",
            flux_types[i],
            error
        );
    }

    // Results should be similar for smooth solution
    let max_diff = final_max_h.iter().fold(0.0_f64, |acc, &h| {
        final_max_h
            .iter()
            .fold(acc, |inner, &h2| inner.max((h - h2).abs()))
    });
    assert!(
        max_diff < 0.1,
        "Different flux types should give similar results for smooth solutions: diff = {}",
        max_diff
    );
}

/// Test reflective boundary conditions.
#[test]
fn test_reflective_boundary() {
    let mesh = Mesh2D::uniform_rectangle(0.0, 10.0, 0.0, 10.0, 4, 4);
    let ops = DGOperators2D::new(2);
    let geom = GeometricFactors2D::compute(&mesh, &ops);
    let equation = ShallowWater2D::new(G);
    let bc = Reflective2D::new();
    let config = SWE2DRhsConfig::new(&equation, &bc).with_coriolis(false);

    // Wave hitting wall
    let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
    q.set_from_functions(
        &mesh,
        &ops,
        |x, _y| {
            // Gaussian bump near left boundary
            let x0 = 2.0;
            2.0 + 0.5 * (-(x - x0).powi(2) / 1.0).exp()
        },
        |x, _y| {
            // Moving right
            let x0 = 2.0;
            0.5 * (-(x - x0).powi(2) / 1.0).exp()
        },
        |_, _| 0.0,
    );

    let initial_mass = q.integrate_depth(&ops, &geom);

    // Run simulation
    let cfl = 0.2;
    let mut time = 0.0;
    let end_time = 2.0;
    while time < end_time {
        let dt = compute_dt_swe_2d(&q, &mesh, &geom, &equation, ops.order, cfl);
        let dt = dt.min(end_time - time);
        ssp_rk3_swe_step(&mut q, &mesh, &ops, &geom, &config, dt, time);
        time += dt;

        // Check stability
        if q.max_abs() > 100.0 || q.has_negative_depth() {
            panic!(
                "Solution unstable at t = {}: max = {}, min_h = {}",
                time,
                q.max_abs(),
                q.min_depth()
            );
        }
    }

    // Mass should be approximately conserved (some error due to wall reflection)
    let final_mass = q.integrate_depth(&ops, &geom);
    let mass_error = ((final_mass - initial_mass) / initial_mass).abs();
    assert!(
        mass_error < 1e-6,
        "Mass should be approximately conserved: error = {:.2e}",
        mass_error
    );
}

/// Test stability for long simulation.
#[test]
fn test_long_term_stability() {
    let mesh = Mesh2D::uniform_periodic(0.0, 10.0, 0.0, 10.0, 4, 4);
    let ops = DGOperators2D::new(2);
    let geom = GeometricFactors2D::compute(&mesh, &ops);
    let equation = ShallowWater2D::new(G);
    let bc = Reflective2D::new();
    let config = SWE2DRhsConfig::new(&equation, &bc).with_coriolis(false);

    // Smooth initial perturbation
    let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
    q.set_from_functions(
        &mesh,
        &ops,
        |x, y| {
            1.0 + 0.1
                * (2.0 * std::f64::consts::PI * x / 10.0).sin()
                * (2.0 * std::f64::consts::PI * y / 10.0).sin()
        },
        |_, _| 0.0,
        |_, _| 0.0,
    );

    let initial_mass = q.integrate_depth(&ops, &geom);

    // Run for many time steps
    let cfl = 0.3;
    let mut time = 0.0;
    let end_time = 10.0;
    let mut step = 0;
    while time < end_time {
        let dt = compute_dt_swe_2d(&q, &mesh, &geom, &equation, ops.order, cfl);
        let dt = dt.min(end_time - time);
        ssp_rk3_swe_step(&mut q, &mesh, &ops, &geom, &config, dt, time);
        time += dt;
        step += 1;

        // Check stability every 100 steps
        if step % 100 == 0 {
            assert!(
                !q.has_negative_depth(),
                "Negative depth at t = {}, step {}",
                time,
                step
            );
            let current_mass = q.integrate_depth(&ops, &geom);
            let mass_error = ((current_mass - initial_mass) / initial_mass).abs();
            assert!(
                mass_error < 1e-10,
                "Mass drift at t = {}: {:.2e}",
                time,
                mass_error
            );
        }
    }

    // Final checks
    assert!(
        !q.has_negative_depth(),
        "Should remain positive for {} steps",
        step
    );
    let final_mass = q.integrate_depth(&ops, &geom);
    let final_error = ((final_mass - initial_mass) / initial_mass).abs();
    assert!(
        final_error < 1e-10,
        "Final mass error after {} steps: {:.2e}",
        step,
        final_error
    );
}

/// Multi-day tidal run on the production path (TODO P1.7): three days of
/// M2 through an open boundary into a 40 × 20 km basin that shoals from
/// 200 m at the mouth to 20 m at the head, with Coriolis, point-implicit
/// Manning friction and the split-form operator (`Simulation` + `SSPRK3` +
/// `SWEPhysics2D`).
///
/// After a 3 h ramp and a day of spin-up the response must be periodic:
/// the basin volume and the surface at the head repeat from one tidal cycle
/// to the next (no secular drift of mass, mean level or amplitude), and the
/// head amplitude is the forcing's (the basin is 3 % of a tidal wavelength
/// long, so the tide is nearly uniform across it). Measured: cycle to cycle
/// the volume changes by 5e-7 m of basin-mean surface and the head surface
/// by 1.4e-6 m; the head amplitude is 0.498 m for 0.5 m forced.
#[test]
fn test_multi_day_tidal_run() {
    use std::sync::Arc;

    use dgcom::boundary::{
        CharacteristicOBC, HarmonicTide, MultiBoundaryCondition2D, TidalConstituent,
    };
    use dgcom::mesh::{Bathymetry2D, BoundaryTag};
    use dgcom::physics::PhysicsBuilder;
    use dgcom::simulation::Simulation;
    use dgcom::solver::SWEFormulation2D;
    use dgcom::source::{CoriolisSource2D, ManningFriction2D};
    use dgcom::time::SSPRK3;

    let (lx, ly, amplitude) = (40_000.0, 20_000.0, 0.5);
    let period = TidalConstituent::m2(amplitude, 0.0).period;
    let bed = move |x: f64, y: f64| {
        -(20.0 + 180.0 * (1.0 - x / lx)) - 10.0 * (std::f64::consts::PI * y / ly).sin()
    };
    let mesh = Arc::new(Mesh2D::uniform_rectangle_with_sides(
        0.0,
        lx,
        0.0,
        ly,
        8,
        4,
        [
            BoundaryTag::Wall,
            BoundaryTag::Wall,
            BoundaryTag::Wall,
            BoundaryTag::Open,
        ],
    ));
    let ops = Arc::new(DGOperators2D::new(2));
    let geom = Arc::new(GeometricFactors2D::compute(&mesh, &ops));
    let bathymetry = Arc::new(Bathymetry2D::from_function(&mesh, &ops, &geom, bed));

    let wall = Reflective2D::new();
    let sea = CharacteristicOBC::new(HarmonicTide::m2(amplitude, 0.0).with_ramp_up(3.0 * 3600.0));
    let physics = PhysicsBuilder::swe_2d(
        mesh.clone(),
        ops.clone(),
        geom.clone(),
        ShallowWater2D::new(G),
        MultiBoundaryCondition2D::new(&wall).with_open(&sea),
    )
    .with_bathymetry(bathymetry.clone())
    .with_formulation(SWEFormulation2D::EntropyStable)
    .with_implicit_friction(ManningFriction2D::new(G, 0.025))
    .with_source(CoriolisSource2D::f_plane(1.2e-4))
    .build();

    let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
    for ki in 0..mesh.n_elements {
        for i in 0..ops.n_nodes {
            q.set_state(
                k(ki),
                i,
                SWEState2D::new(-bathymetry.get(k(ki), i), 0.0, 0.0),
            );
        }
    }

    // Every hour of the tidal cycle (24 samples per period): the basin's
    // volume and the mean surface of the head column of elements
    let samples_per_period = 24;
    let head: Vec<usize> = (0..mesh.n_elements)
        .filter(|&ki| {
            mesh.element_vertices(k(ki))
                .iter()
                .all(|v| v[0] >= lx - 5_000.0 - 1e-6)
        })
        .collect();
    let mut volume = Vec::new();
    let mut head_eta = Vec::new();
    let mut max_speed: f64 = 0.0;
    let end = 3.0 * 86_400.0;
    Simulation::new(physics, SSPRK3)
        .with_callback_interval(period / samples_per_period as f64)
        .run_with_callback(&mut q, 0.0, end, |q, _| {
            let (mut v, mut eta, mut area) = (0.0, 0.0, 0.0);
            for ki in 0..mesh.n_elements {
                for i in 0..ops.n_nodes {
                    let weight = ops.weights[i] * geom.det_j[ki * ops.n_nodes + i];
                    let state = q.get_state(k(ki), i);
                    v += weight * state.h;
                    max_speed = max_speed.max(state.hu.hypot(state.hv) / state.h);
                    if head.contains(&ki) {
                        eta += weight * (state.h + bathymetry.get(k(ki), i));
                        area += weight;
                    }
                }
            }
            volume.push(v);
            head_eta.push(eta / area);
        });
    assert!(q.h_data().iter().all(|h| h.is_finite() && *h > 0.0));
    assert!(max_speed < 0.5, "max speed {max_speed:.3} m/s");

    // Compare the last full cycle with the one before (volume in metres of
    // basin-mean surface)
    let n = samples_per_period;
    let last = volume.len() - n..volume.len();
    let before = volume.len() - 2 * n..volume.len() - n;
    let area = lx * ly;
    let volume_change = last
        .clone()
        .zip(before.clone())
        .map(|(a, b)| ((volume[a] - volume[b]) / area).abs())
        .fold(0.0_f64, f64::max);
    let head_change = last
        .clone()
        .zip(before)
        .map(|(a, b)| (head_eta[a] - head_eta[b]).abs())
        .fold(0.0_f64, f64::max);
    let range = |r: std::ops::Range<usize>| {
        let (lo, hi) = r.fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), i| {
            (lo.min(head_eta[i]), hi.max(head_eta[i]))
        });
        (0.5 * (hi - lo), 0.5 * (hi + lo))
    };
    let (head_amplitude, head_mean) = range(last);
    assert!(
        volume_change < 1e-5 && head_change < 2e-5,
        "not periodic: volume changed by {volume_change:.2e} m, head η by {head_change:.2e} m"
    );
    assert!(
        (0.98..1.02).contains(&(head_amplitude / amplitude)),
        "head amplitude {head_amplitude:.4} m against {amplitude} m forced"
    );
    assert!(head_mean.abs() < 1e-3, "head mean level {head_mean:.3e} m");
}

/// Long-run mass conservation: 100 wave-crossing times of a periodic domain
/// (316 s, > 1000 steps) with a smooth perturbation, checking mass, positivity
/// and boundedness along the way. For a run over days of model time see
/// `test_multi_day_tidal_run`.
#[test]
fn test_long_run_mass_conservation() {
    let mesh = Mesh2D::uniform_periodic(0.0, 10.0, 0.0, 10.0, 4, 4);
    let ops = DGOperators2D::new(1); // P1 for speed
    let geom = GeometricFactors2D::compute(&mesh, &ops);
    let equation = ShallowWater2D::new(G);
    let bc = Reflective2D::new(); // Never called on periodic mesh
    let config = SWE2DRhsConfig::new(&equation, &bc).with_coriolis(false);

    // Smooth Gaussian perturbation centered at domain midpoint
    let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
    q.set_from_functions(
        &mesh,
        &ops,
        |x, y| {
            let r2 = (x - 5.0).powi(2) + (y - 5.0).powi(2);
            1.0 + 0.05 * (-r2 / 2.0).exp()
        },
        |_, _| 0.0,
        |_, _| 0.0,
    );

    let initial_mass = q.integrate_depth(&ops, &geom);
    let initial_max_h = q.max_depth();

    // Wave crossing time: L/c = 10/sqrt(10) ≈ 3.16s
    // Run for 100 crossing times ≈ 316s
    let end_time = 100.0 * 10.0 / (G * 1.0_f64).sqrt();
    let cfl = 0.3;
    let mut time = 0.0;
    let mut step = 0;
    let mut max_h_ever = initial_max_h;

    while time < end_time {
        let dt = compute_dt_swe_2d(&q, &mesh, &geom, &equation, ops.order, cfl);
        let dt = dt.min(end_time - time);
        if dt <= 0.0 {
            break;
        }
        ssp_rk3_swe_step(&mut q, &mesh, &ops, &geom, &config, dt, time);
        time += dt;
        step += 1;

        let current_max = q.max_depth();
        if current_max > max_h_ever {
            max_h_ever = current_max;
        }

        // Periodic checks for stability
        if step % 500 == 0 {
            assert!(
                !q.has_negative_depth(),
                "Negative depth at t = {:.1}, step {}",
                time,
                step
            );

            // Solution should stay bounded (no blow-up)
            assert!(
                current_max < 3.0 * initial_max_h,
                "Solution blowing up at t = {:.1}: max_h = {:.4} (initial {:.4})",
                time,
                current_max,
                initial_max_h
            );

            // Mass conservation
            let current_mass = q.integrate_depth(&ops, &geom);
            let mass_error = ((current_mass - initial_mass) / initial_mass).abs();
            assert!(
                mass_error < 1e-10,
                "Mass drift at t = {:.1}: {:.2e}",
                time,
                mass_error
            );
        }
    }

    // Final verification
    assert!(step > 1000, "Should have run many steps, got {}", step);
    let final_mass = q.integrate_depth(&ops, &geom);
    let final_error = ((final_mass - initial_mass) / initial_mass).abs();
    assert!(
        final_error < 1e-10,
        "Final mass error after {} steps ({:.0}s): {:.2e}",
        step,
        time,
        final_error
    );
    assert!(
        !q.has_negative_depth(),
        "Negative depth after {} steps",
        step
    );
}

/// Long-time conservation test.
///
/// Verifies mass AND momentum conservation over many oscillation periods
/// on a fully periodic domain with no source terms.
///
/// On a periodic domain without friction/Coriolis, the DG scheme should
/// conserve total mass, x-momentum, and y-momentum to machine precision.
#[test]
fn test_long_time_conservation() {
    use std::f64::consts::PI;

    let mesh = Mesh2D::uniform_periodic(0.0, 10.0, 0.0, 10.0, 4, 4);
    let ops = DGOperators2D::new(1); // P1 for speed
    let geom = GeometricFactors2D::compute(&mesh, &ops);
    let equation = ShallowWater2D::new(G);
    let bc = Reflective2D::new(); // Never called on periodic mesh
    let config = SWE2DRhsConfig::new(&equation, &bc).with_coriolis(false);

    // Initial condition: standing wave with both h and velocity perturbations
    // to exercise momentum conservation
    let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
    q.set_from_functions(
        &mesh,
        &ops,
        |x, y| 1.0 + 0.05 * (2.0 * PI * x / 10.0).sin() * (2.0 * PI * y / 10.0).cos(),
        |x, _y| 0.02 * (2.0 * PI * x / 10.0).cos(), // small u perturbation
        |_x, y| 0.02 * (2.0 * PI * y / 10.0).sin(), // small v perturbation
    );

    let initial_mass = q.integrate_depth(&ops, &geom);
    let initial_hu = q.integrate_x_momentum(&ops, &geom);
    let initial_hv = q.integrate_y_momentum(&ops, &geom);

    // Run for 50 oscillation periods
    // Standing wave period ≈ L / c = 10 / sqrt(10) ≈ 3.16s
    // 50 periods ≈ 158s
    let period = 10.0 / (G * 1.0_f64).sqrt();
    let end_time = 50.0 * period;
    let cfl = 0.3;
    let mut time = 0.0;
    let mut step = 0;

    while time < end_time {
        let dt = compute_dt_swe_2d(&q, &mesh, &geom, &equation, ops.order, cfl);
        let dt = dt.min(end_time - time);
        if dt <= 0.0 {
            break;
        }
        ssp_rk3_swe_step(&mut q, &mesh, &ops, &geom, &config, dt, time);
        time += dt;
        step += 1;
    }

    // Verify conservation of all three quantities
    let final_mass = q.integrate_depth(&ops, &geom);
    let final_hu = q.integrate_x_momentum(&ops, &geom);
    let final_hv = q.integrate_y_momentum(&ops, &geom);

    let mass_error = ((final_mass - initial_mass) / initial_mass).abs();
    assert!(
        mass_error < 1e-10,
        "Mass conservation failed after {} steps ({:.0}s, {:.0} periods): {:.2e}",
        step,
        time,
        time / period,
        mass_error
    );

    // Momentum conservation: absolute error (initial momentum may be near zero)
    let hu_error = (final_hu - initial_hu).abs();
    assert!(
        hu_error < 1e-10,
        "X-momentum conservation failed after {} steps: initial={:.6e}, final={:.6e}, error={:.2e}",
        step,
        initial_hu,
        final_hu,
        hu_error
    );

    let hv_error = (final_hv - initial_hv).abs();
    assert!(
        hv_error < 1e-10,
        "Y-momentum conservation failed after {} steps: initial={:.6e}, final={:.6e}, error={:.2e}",
        step,
        initial_hv,
        final_hv,
        hv_error
    );

    // Verify solution stays bounded
    assert!(
        !q.has_negative_depth(),
        "Negative depth after {} periods",
        time / period
    );
    assert!(
        q.max_depth() < 5.0,
        "Solution blew up: max_h = {:.4}",
        q.max_depth()
    );
}

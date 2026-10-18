//! Minimal 3D Simulation Example
//!
//! Simulates a standing wave in a closed basin (seiche) using the 3D solver.
//! Demonstrates the setup of the Simulation3D runner.

use dg_rs::boundary::Reflective2D;
use dg_rs::equations::ShallowWater2D;
use dg_rs::mesh::Mesh2D;
use dg_rs::mesh::data::Bathymetry2D;
use dg_rs::operators::{DGOperators2D, GeometricFactors2D};
use dg_rs::physics::{ConstantMixing, Forcing, Hydrostatic3D, LinearEOS, PhysicsBuilder};
use dg_rs::simulation::Simulation3D;
use dg_rs::solver::state::Solution3D;
use dg_rs::time::ModeSplitIntegrator;
use dg_rs::types::ElementIndex;
use dg_rs::vertical::{SigmaGrid, UniformStretching};
use std::sync::Arc;

fn main() -> Result<(), String> {
    println!("Starting Minimal 3D Simulation...");

    // 1. Setup Mesh (Simple 2x1 rectangle)
    // 1000m x 100m domain
    let mesh = Arc::new(Mesh2D::uniform_rectangle(0.0, 1000.0, 0.0, 100.0, 2, 1));
    let ops = Arc::new(DGOperators2D::new(1)); // Order 1 (bilinear)
    let geom = Arc::new(GeometricFactors2D::compute(&mesh, &ops));

    println!(
        "Mesh: {} elements, {} nodes/element",
        mesh.n_elements, ops.n_nodes
    );

    // 2. Setup Vertical Grid (3 levels)
    let sigma = Arc::new(SigmaGrid::new(3, UniformStretching));

    // 3. Setup Physics Components
    // Flat bed at -10 m gives 10 m depth when eta = 0.
    let bathymetry = Arc::new(Bathymetry2D::from_function(&mesh, &ops, &geom, |_, _| {
        -10.0
    }));
    let coriolis = Arc::new(dg_rs::source::CoriolisSource2D::f_plane(0.0)); // No rotation
    let eos = LinearEOS::default();
    let mixing = ConstantMixing::new(1e-4, 1e-4);
    let forcing = Forcing {
        surface_stress: [0.0, 0.0],
        bottom_stress: [0.0, 0.0],
        surface_heat_flux: 0.0,
    };

    // Build 2D sub-model
    let swe_eq = ShallowWater2D::new(9.81);
    let swe_bc = Reflective2D::default();
    let swe_physics = PhysicsBuilder::swe_2d(
        mesh.clone(),
        ops.clone(),
        geom.clone(),
        swe_eq,
        swe_bc.clone(),
    )
    .with_bathymetry(bathymetry.clone())
    .build();

    // 4. Create Hydrostatic3D Physics
    let physics = Hydrostatic3D::new(
        mesh.clone(),
        ops.clone(),
        geom.clone(),
        sigma.clone(),
        bathymetry.clone(),
        coriolis.clone(),
        eos,
        mixing,
        swe_physics,
        forcing,
        9.81,   // g
        1025.0, // rho0
    );

    // 5. Initial State (Surface Perturbation)
    let mut state = Solution3D::new(mesh.n_elements, ops.n_nodes, 3);

    // Apply linear slope to free surface: eta = 0.1 * (x/L - 0.5)
    // This creates a pressure gradient that drives flow.
    for k in 0..mesh.n_elements {
        for i in 0..ops.n_nodes {
            let r = ops.nodes_r[i];
            let s = ops.nodes_s[i];
            let k_idx = ElementIndex::new(k);
            let [x, _y] = mesh.reference_to_physical(k_idx, r, s);

            let eta = 0.1 * (x / 1000.0 - 0.5);

            // Set 2D state
            state.eta.data[k * ops.n_nodes + i] = eta;
            state.ubar.data[k * ops.n_nodes + i] = 0.0;
            state.vbar.data[k * ops.n_nodes + i] = 0.0;

            // Set 3D state (u, v, w, rho, tracers)
            // Initial velocity 0
            // Initial density constant
            // Initial T, S constant
            // We rely on Solution3D::new zero-initialization for velocities.
            // We should set T, S if needed, but 0 is fine for this test.
        }
    }

    // Update density based on initial T,S (0,0)
    physics.update_density(&mut state);

    // 6. Setup Integrator
    // Barotropic substeps per baroclinic step follow from the 2D CFL
    let integrator = ModeSplitIntegrator::new();

    // 7. Run Simulation
    let mut sim = Simulation3D::new(physics, integrator)
        .with_cfl(0.5)
        .with_max_steps(5) // Run 5 steps
        .verbose();

    println!("Running simulation...");
    let result = sim.run(&mut state, 0.0, 10.0);

    println!("Simulation finished.");
    println!("Result: {:?}", result);

    if result.success {
        Ok(())
    } else {
        Err(result.error.unwrap_or_else(|| "Unknown error".to_string()))
    }
}

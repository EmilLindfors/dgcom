//! High-Resolution Benchmark
//!
//! Tests DG solver performance at 50m effective resolution with different polynomial orders.
//! Uses a 5km × 5km domain to keep computational cost manageable.
//!
//! ## Run
//!
//! ```bash
//! # Serial baseline
//! cargo run --release --example high_res_benchmark
//!
//! # With parallel processing
//! cargo run --release --example high_res_benchmark --features parallel
//!
//! # With SIMD optimization
//! cargo run --release --example high_res_benchmark --features simd
//!
//! # With both parallel and SIMD
//! cargo run --release --example high_res_benchmark --features "parallel simd"
//! ```

use std::time::Instant;

use dgcom::boundary::{CharacteristicOBC, HarmonicTide, MultiBoundaryCondition2D, Reflective2D};
use dgcom::equations::ShallowWater2D;
use dgcom::mesh::{Bathymetry2D, BoundaryTag, Mesh2D};
use dgcom::operators::{DGOperators2D, GeometricFactors2D};
#[cfg(not(all(feature = "parallel", feature = "simd")))]
use dgcom::solver::compute_dt_swe_2d;
#[cfg(all(feature = "parallel", feature = "simd"))]
use dgcom::solver::compute_dt_swe_2d_parallel;
use dgcom::solver::{SWE2DRhsConfig, SWESolution2D};

// RHS functions based on enabled features
#[cfg(not(all(feature = "parallel", feature = "simd")))]
use dgcom::solver::compute_rhs_swe_2d;
#[cfg(all(feature = "parallel", feature = "simd"))]
use dgcom::solver::compute_rhs_swe_2d_parallel;
use dgcom::source::{CombinedSource2D, CoriolisSource2D, ManningFriction2D};
use dgcom::time::{SWE2DTimeConfig, ssp_rk3_swe_2d_step_limited};
use dgcom::types::{Depth, ElementIndex};
use dgcom::{SWEFluxType2D, SWEState2D};

fn k(idx: usize) -> ElementIndex {
    ElementIndex::new(idx)
}

// Physical constants
const G: f64 = 9.81;
const H_MIN: f64 = 1.0;
const F_CORIOLIS: f64 = 1.26e-4; // 64°N
const MANNING_N: f64 = 0.025;

// Domain size (10km × 10km for larger test)
const DOMAIN_SIZE: f64 = 10000.0;

// Target effective resolution
const TARGET_RESOLUTION: f64 = 50.0;

// Simulation duration (5 minutes for benchmark)
const T_END: f64 = 300.0;

// Tidal parameters
const M2_AMPLITUDE: f64 = 0.5;
const RAMP_DURATION: f64 = 300.0;

fn main() {
    println!("=================================================================");
    println!("  High-Resolution DG Benchmark (50m effective resolution)");
    println!("=================================================================");
    println!();

    // Show enabled features
    print!("Features enabled: ");
    let features: Vec<&str> = [
        ("parallel", cfg!(feature = "parallel")),
        ("simd", cfg!(feature = "simd")),
    ]
    .into_iter()
    .filter_map(|(name, on)| on.then_some(name))
    .collect();
    if features.is_empty() {
        println!("none (serial baseline)");
    } else {
        println!("{}", features.join(", "));
    }

    #[cfg(feature = "parallel")]
    println!("CPU threads: {}", rayon::current_num_threads());

    println!();
    println!(
        "Domain: {:.1} km × {:.1} km",
        DOMAIN_SIZE / 1000.0,
        DOMAIN_SIZE / 1000.0
    );
    println!("Target effective resolution: {} m", TARGET_RESOLUTION);
    println!("Simulation duration: {:.1} minutes", T_END / 60.0);
    println!();

    // Test different polynomial orders
    let orders = [2, 3, 4, 5];

    println!(
        "{:<6} {:<10} {:<12} {:<10} {:<12} {:<10} {:<12}",
        "Order", "Elements", "Total DOFs", "Eff. Res", "Avg dt (s)", "Steps", "Time (s)"
    );
    println!("{}", "-".repeat(80));

    let mut results = Vec::new();

    for &order in &orders {
        match run_benchmark(order) {
            Ok(result) => {
                println!(
                    "{:<6} {:<10} {:<12} {:<10.1} {:<12.4} {:<10} {:<12.2}",
                    format!("P{}", order),
                    result.n_elements,
                    result.total_dofs,
                    result.effective_resolution,
                    result.avg_dt,
                    result.total_steps,
                    result.wall_time
                );
                results.push(result);
            }
            Err(e) => {
                println!("P{}: FAILED - {}", order, e);
            }
        }
    }

    println!();
    println!("=================================================================");
    println!("  Analysis");
    println!("=================================================================");
    println!();

    if results.len() >= 2 {
        let baseline = &results[0]; // P2
        println!("Relative to P2 baseline:");
        println!();
        for result in &results {
            let speedup = baseline.wall_time / result.wall_time;
            let dof_ratio = result.total_dofs as f64 / baseline.total_dofs as f64;
            let step_ratio = result.total_steps as f64 / baseline.total_steps as f64;
            println!(
                "  P{}: {:.2}× DOFs, {:.2}× steps, {:.2}× wall time (speedup: {:.2}×)",
                result.order,
                dof_ratio,
                step_ratio,
                result.wall_time / baseline.wall_time,
                speedup
            );
        }
    }

    println!();
    println!("Recommendation for 50m resolution coastal modeling:");
    if let Some(best) = results
        .iter()
        .min_by(|a, b| a.wall_time.partial_cmp(&b.wall_time).unwrap())
    {
        println!("  → P{} offers best wall-clock performance", best.order);
        println!(
            "  → {} elements, {} DOFs, {:.4}s avg time step",
            best.n_elements, best.total_dofs, best.avg_dt
        );
    }
}

struct BenchmarkResult {
    order: usize,
    n_elements: usize,
    total_dofs: usize,
    effective_resolution: f64,
    avg_dt: f64,
    total_steps: usize,
    wall_time: f64,
}

fn run_benchmark(order: usize) -> Result<BenchmarkResult, String> {
    // Calculate number of elements needed for target resolution
    // Effective resolution ≈ element_size / (order + 1)
    // So element_size = target_resolution * (order + 1)
    let element_size = TARGET_RESOLUTION * (order + 1) as f64;
    let n_elements_1d = (DOMAIN_SIZE / element_size).ceil() as usize;
    let n_elements_1d = n_elements_1d.max(4); // Minimum 4 elements per direction

    // Create mesh
    let mut mesh = Mesh2D::uniform_rectangle_with_bc(
        0.0,
        DOMAIN_SIZE,
        0.0,
        DOMAIN_SIZE,
        n_elements_1d,
        n_elements_1d,
        BoundaryTag::Wall,
    );

    // Tag west boundary as open (tidal forcing)
    for edge in mesh.edges.iter_mut() {
        if edge.is_boundary() {
            let (v0, v1) = edge.vertices;
            let x0 = mesh.vertices[v0][0];
            let x1 = mesh.vertices[v1][0];
            let x_mid = (x0 + x1) / 2.0;

            if x_mid < 10.0 {
                // West boundary
                edge.boundary_tag = Some(BoundaryTag::Open);
            }
        }
    }

    // Create operators
    let ops = DGOperators2D::new(order);
    let geom = GeometricFactors2D::compute(&mesh, &ops);

    let n_elements = mesh.n_elements;
    let total_dofs = n_elements * ops.n_nodes * 3; // h, hu, hv
    let actual_element_size = DOMAIN_SIZE / n_elements_1d as f64;
    let effective_resolution = actual_element_size / (order + 1) as f64;

    // Create bathymetry (sloping from 50m at west to 20m at east)
    let bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, _y| {
        let t = x / DOMAIN_SIZE;
        -50.0 + 30.0 * t // -50m to -20m
    });

    // Initialize state (lake at rest, η = 0)
    let mut state = SWESolution2D::new(n_elements, ops.n_nodes);
    for ki in 0..n_elements {
        for i in 0..ops.n_nodes {
            let b = bathymetry.get(k(ki), i);
            let h = (0.0 - b).max(H_MIN); // η = 0, so h = -B
            state.set_state(k(ki), i, SWEState2D::from_primitives(h, 0.0, 0.0));
        }
    }

    // Physics
    let coriolis = CoriolisSource2D::f_plane(F_CORIOLIS);
    let friction = ManningFriction2D::new(G, MANNING_N);
    let sources = CombinedSource2D::new(vec![&coriolis, &friction]);

    // Boundary conditions
    let wall_bc = Reflective2D::new();
    let tidal_bc =
        CharacteristicOBC::new(HarmonicTide::m2(M2_AMPLITUDE, 0.0).with_ramp_up(RAMP_DURATION));
    let bc = MultiBoundaryCondition2D::new(&wall_bc).with_open(&tidal_bc);

    // Equation and time config
    let equation = ShallowWater2D::new(G);
    let cfl = 0.1; // Conservative CFL for stability
    let time_config = SWE2DTimeConfig::new(cfl, G, H_MIN)
        .with_kuzmin_limiters(1.0)
        .with_wet_dry_treatment();

    // Run simulation
    let start = Instant::now();
    let mut t = 0.0;
    let mut step = 0;
    let mut max_velocity = 0.0_f64;

    while t < T_END {
        #[cfg(all(feature = "parallel", feature = "simd"))]
        let dt = compute_dt_swe_2d_parallel(&state, &mesh, &geom, &equation, order, cfl);
        #[cfg(not(all(feature = "parallel", feature = "simd")))]
        let dt = compute_dt_swe_2d(&state, &mesh, &geom, &equation, order, cfl);
        let dt = dt.min(T_END - t);

        if dt < 1e-10 {
            return Err(format!("Time step too small: {:.2e}", dt));
        }

        let rhs_fn = |s: &SWESolution2D, time: f64| {
            let config = SWE2DRhsConfig::new(&equation, &bc)
                .with_flux_type(SWEFluxType2D::Rusanov)
                .with_source_terms(&sources)
                .with_bathymetry(&bathymetry)
                .with_well_balanced(true);

            // Use the best available implementation
            #[cfg(all(feature = "parallel", feature = "simd"))]
            {
                compute_rhs_swe_2d_parallel(s, &mesh, &ops, &geom, &config, time)
            }
            #[cfg(not(all(feature = "parallel", feature = "simd")))]
            {
                compute_rhs_swe_2d(s, &mesh, &ops, &geom, &config, time)
            }
        };

        ssp_rk3_swe_2d_step_limited(&mut state, dt, t, &mesh, &ops, &geom, rhs_fn, &time_config);

        t += dt;
        step += 1;

        // Check for blow-up
        let mut current_max_vel = 0.0_f64;
        for ki in 0..n_elements {
            for i in 0..ops.n_nodes {
                let s = state.get_state(k(ki), i);
                if !s.h.is_finite() || !s.hu.is_finite() || !s.hv.is_finite() {
                    return Err(format!("NaN detected at step {}", step));
                }
                let (u, v) = s.velocity_simple(Depth::new(H_MIN));
                current_max_vel = current_max_vel.max((u * u + v * v).sqrt());
            }
        }
        max_velocity = max_velocity.max(current_max_vel);

        if max_velocity > 50.0 {
            return Err(format!(
                "Velocity blow-up: {:.1} m/s at step {}",
                max_velocity, step
            ));
        }
    }

    let wall_time = start.elapsed().as_secs_f64();
    let avg_dt = t / step as f64;

    Ok(BenchmarkResult {
        order,
        n_elements,
        total_dofs,
        effective_resolution,
        avg_dt,
        total_steps: step,
        wall_time,
    })
}

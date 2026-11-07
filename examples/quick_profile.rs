//! Quick profiling benchmark - realistic element count
//!
//! Run with different backends:
//! ```bash
//! # CPU serial
//! cargo run --release --example quick_profile
//!
//! # CPU parallel (rayon)
//! cargo run --release --features parallel --example quick_profile
//!
//! # GPU CUDA
//! cargo run --release --features burn-cuda --example quick_profile
//!
//! # GPU WGPU (cross-platform)
//! cargo run --release --features burn-wgpu --example quick_profile
//! ```

use std::time::Instant;

use dgcom::SWEState2D;
use dgcom::mesh::{BoundaryTag, Mesh2D};
use dgcom::operators::{DGOperators2D, GeometricFactors2D};
use dgcom::solver::SWESolution2D;
use dgcom::types::ElementIndex;

// CPU imports
#[cfg(not(any(feature = "burn-cuda", feature = "burn-wgpu")))]
use dgcom::SWEFluxType2D;
#[cfg(not(any(feature = "burn-cuda", feature = "burn-wgpu")))]
use dgcom::boundary::{MultiBoundaryCondition2D, Reflective2D};
#[cfg(not(any(feature = "burn-cuda", feature = "burn-wgpu")))]
use dgcom::equations::ShallowWater2D;
#[cfg(not(any(feature = "burn-cuda", feature = "burn-wgpu")))]
use dgcom::mesh::Bathymetry2D;
#[cfg(not(any(feature = "burn-cuda", feature = "burn-wgpu")))]
use dgcom::solver::SWE2DRhsConfig;
#[cfg(not(any(feature = "burn-cuda", feature = "burn-wgpu")))]
use dgcom::time::{SWE2DTimeConfig, ssp_rk3_swe_2d_step_limited};

#[cfg(all(
    not(all(feature = "parallel", feature = "simd")),
    not(any(feature = "burn-cuda", feature = "burn-wgpu"))
))]
use dgcom::solver::{compute_dt_swe_2d, compute_rhs_swe_2d};
#[cfg(all(
    all(feature = "parallel", feature = "simd"),
    not(any(feature = "burn-cuda", feature = "burn-wgpu"))
))]
use dgcom::solver::{compute_dt_swe_2d_parallel, compute_rhs_swe_2d_parallel};

// GPU imports
#[cfg(any(feature = "burn-cuda", feature = "burn-wgpu"))]
use dgcom::solver::burn::{
    BurnConnectivity, BurnGeometricFactors2D, BurnOperators2D, BurnRhsConfig, BurnSWESolution2D,
};
#[cfg(any(feature = "burn-cuda", feature = "burn-wgpu"))]
use dgcom::time::{compute_dt_burn, ssp_rk3_step_burn};

#[cfg(feature = "burn-cuda")]
use burn_cuda::{Cuda, CudaDevice};
#[cfg(all(feature = "burn-wgpu", not(feature = "burn-cuda")))]
use burn_wgpu::{Wgpu, WgpuDevice};

fn k(idx: usize) -> ElementIndex {
    ElementIndex::new(idx)
}

const G: f64 = 9.81;
const H_MIN: f64 = 1.0;

fn main() {
    // Fjord-scale: 20km x 5km at ~50m resolution -> ~40,000 elements
    let order = 3;
    let nx = 400; // 20km / 50m
    let ny = 100; // 5km / 50m

    let mesh = Mesh2D::uniform_rectangle_with_bc(
        0.0,
        20000.0,
        0.0,
        5000.0, // 20km x 5km fjord
        nx,
        ny,
        BoundaryTag::Wall,
    );

    let ops = DGOperators2D::new(order);
    let geom = GeometricFactors2D::compute(&mesh, &ops);
    let n_elements = mesh.n_elements;

    // Determine backend name for output
    #[cfg(feature = "burn-cuda")]
    let backend_name = "GPU (CUDA)";
    #[cfg(all(feature = "burn-wgpu", not(feature = "burn-cuda")))]
    let backend_name = "GPU (WGPU)";
    #[cfg(all(
        feature = "parallel",
        not(any(feature = "burn-cuda", feature = "burn-wgpu"))
    ))]
    let backend_name = "CPU (parallel/rayon)";
    #[cfg(all(
        not(feature = "parallel"),
        not(any(feature = "burn-cuda", feature = "burn-wgpu"))
    ))]
    let backend_name = "CPU (serial)";

    println!("Backend: {}", backend_name);
    println!(
        "Elements: {}, DOFs: {}, Order: P{}",
        n_elements,
        n_elements * ops.n_nodes * 3,
        order
    );

    // Initialize state: h = 20m (flat surface at z=0, bathymetry at -20m)
    let mut state = SWESolution2D::new(n_elements, ops.n_nodes);
    for ki in 0..n_elements {
        for i in 0..ops.n_nodes {
            let h = 20.0; // Constant 20m depth
            state.set_state(k(ki), i, SWEState2D::from_primitives(h, 0.0, 0.0));
        }
    }

    // Run benchmark based on feature
    #[cfg(any(feature = "burn-cuda", feature = "burn-wgpu"))]
    run_gpu_benchmark(&mesh, &ops, &geom, state, n_elements);

    #[cfg(not(any(feature = "burn-cuda", feature = "burn-wgpu")))]
    run_cpu_benchmark(&mesh, &ops, &geom, state, n_elements);
}

/// GPU benchmark using Burn
#[cfg(any(feature = "burn-cuda", feature = "burn-wgpu"))]
fn run_gpu_benchmark(
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    state: SWESolution2D,
    n_elements: usize,
) {
    // Use f64 precision for numerical stability (coastal ocean modeling requires f64)
    // Note: We use default-features=false for burn-cuda/burn-wgpu to disable fusion
    // because fusion doesn't support f64
    #[cfg(feature = "burn-cuda")]
    type B = Cuda<f64, i64>;
    #[cfg(all(feature = "burn-wgpu", not(feature = "burn-cuda")))]
    type B = Wgpu<f64, i64, u32>;

    // Get device
    #[cfg(feature = "burn-cuda")]
    let device = CudaDevice::default();
    #[cfg(all(feature = "burn-wgpu", not(feature = "burn-cuda")))]
    let device = WgpuDevice::default();

    println!("Uploading data to GPU...");
    let upload_start = Instant::now();

    // Upload operators, geometry, and connectivity to GPU
    let burn_ops = BurnOperators2D::<B>::from_cpu(ops, &device);
    let burn_geom = BurnGeometricFactors2D::<B>::from_cpu(geom, &device);
    let connectivity =
        BurnConnectivity::<B>::from_mesh(mesh, geom, &ops.face_nodes, ops.n_face_nodes, &device);

    // Upload initial state
    let mut burn_state = BurnSWESolution2D::<B>::from_cpu(&state, &device);

    let upload_time = upload_start.elapsed().as_secs_f64();
    println!("Upload time: {:.3}s", upload_time);

    // RHS config (no Coriolis/friction for simplicity)
    let config = BurnRhsConfig {
        g: G,
        h_min: H_MIN,
        coriolis_f: None,
        manning_g_n2: None,
    };

    println!("Running 20 time steps on GPU...");
    let start = Instant::now();
    let mut t = 0.0;

    for step in 0..20 {
        // Compute time step on GPU
        let dt = compute_dt_burn(&burn_state, &burn_geom, G, H_MIN, 0.2);

        // SSP-RK3 step on GPU
        ssp_rk3_step_burn(
            &mut burn_state,
            dt,
            &burn_ops,
            &burn_geom,
            &connectivity,
            &config,
        );
        t += dt;

        if step % 5 == 0 {
            println!("Step {}/20: t = {:.4}s, dt = {:.6}s", step, t, dt);
        }
    }

    let elapsed = start.elapsed().as_secs_f64();

    // Download result to verify
    println!("Downloading result from GPU...");
    let download_start = Instant::now();
    let result = burn_state.to_cpu();
    let download_time = download_start.elapsed().as_secs_f64();
    println!("Download time: {:.3}s", download_time);

    // Check result validity
    let (mut min_h, mut max_h) = (f64::INFINITY, f64::NEG_INFINITY);
    for ki in 0..n_elements {
        for i in 0..ops.n_nodes {
            let h = result.get_state(k(ki), i).h;
            min_h = min_h.min(h);
            max_h = max_h.max(h);
        }
    }
    println!("Result: h in [{:.4}, {:.4}]", min_h, max_h);

    println!(
        "\nGPU Performance: {} elements x 20 steps in {:.2}s ({:.1} steps/s)",
        n_elements,
        elapsed,
        20.0 / elapsed
    );
    println!(
        "Total time (incl. transfer): {:.2}s",
        upload_time + elapsed + download_time
    );
}

/// CPU benchmark (parallel or serial)
#[cfg(not(any(feature = "burn-cuda", feature = "burn-wgpu")))]
fn run_cpu_benchmark(
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    mut state: SWESolution2D,
    n_elements: usize,
) {
    // Flat bathymetry at -20m
    let bathymetry = Bathymetry2D::from_function(mesh, ops, geom, |_, _| -20.0);

    let wall_bc = Reflective2D::new();
    let bc = MultiBoundaryCondition2D::new(&wall_bc);
    let equation = ShallowWater2D::new(G);
    let time_config = SWE2DTimeConfig::new(0.2, G, H_MIN)
        .with_kuzmin_limiters(1.0)
        .with_wet_dry_treatment();

    let start = Instant::now();
    let mut t = 0.0;

    // Run 20 steps
    for step in 0..20 {
        #[cfg(all(feature = "parallel", feature = "simd"))]
        let dt = compute_dt_swe_2d_parallel(&state, mesh, geom, &equation, ops.order, 0.2);
        #[cfg(not(all(feature = "parallel", feature = "simd")))]
        let dt = compute_dt_swe_2d(&state, mesh, geom, &equation, ops.order, 0.2);

        let rhs_fn = |s: &SWESolution2D, time: f64| {
            let config = SWE2DRhsConfig::new(&equation, &bc)
                .with_flux_type(SWEFluxType2D::Rusanov)
                .with_bathymetry(&bathymetry)
                .with_well_balanced(true);

            #[cfg(all(feature = "parallel", feature = "simd"))]
            {
                compute_rhs_swe_2d_parallel(s, mesh, ops, geom, &config, time)
            }
            #[cfg(not(all(feature = "parallel", feature = "simd")))]
            {
                compute_rhs_swe_2d(s, mesh, ops, geom, &config, time)
            }
        };

        ssp_rk3_swe_2d_step_limited(&mut state, dt, t, mesh, ops, geom, rhs_fn, &time_config);
        t += dt;

        if step % 5 == 0 {
            println!("Step {}/20: t = {:.4}s, dt = {:.6}s", step, t, dt);
        }
    }

    let elapsed = start.elapsed().as_secs_f64();

    // Check result validity
    let (mut min_h, mut max_h) = (f64::INFINITY, f64::NEG_INFINITY);
    for ki in 0..n_elements {
        for i in 0..ops.n_nodes {
            let h = state.get_state(k(ki), i).h;
            min_h = min_h.min(h);
            max_h = max_h.max(h);
        }
    }
    println!("Result: h in [{:.4}, {:.4}]", min_h, max_h);

    println!(
        "\nCPU Performance: {} elements x 20 steps in {:.2}s ({:.1} steps/s)",
        n_elements,
        elapsed,
        20.0 / elapsed
    );
}

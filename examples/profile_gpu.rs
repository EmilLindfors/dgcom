//! GPU profiling benchmark - identify bottlenecks
//!
//! Run with:
//! ```bash
//! cargo run --release --features burn-wgpu --example profile_gpu
//! ```

use std::time::Instant;

use dgcom::SWEState2D;
use dgcom::mesh::{BoundaryTag, Mesh2D};
use dgcom::operators::{DGOperators2D, GeometricFactors2D};
use dgcom::solver::SWESolution2D;
use dgcom::solver::burn::{
    BurnConnectivity, BurnGeometricFactors2D, BurnOperators2D, BurnRhsConfig, BurnSWESolution2D,
};
use dgcom::time::{compute_dt_burn, ssp_rk3_step_burn};
use dgcom::types::ElementIndex;

#[cfg(feature = "burn-cuda")]
use burn_cuda::{Cuda, CudaDevice};
#[cfg(all(
    feature = "burn-ndarray",
    not(any(feature = "burn-cuda", feature = "burn-wgpu"))
))]
use burn_ndarray::{NdArray, NdArrayDevice};
#[cfg(all(feature = "burn-wgpu", not(feature = "burn-cuda")))]
use burn_wgpu::{Wgpu, WgpuDevice};

fn k(idx: usize) -> ElementIndex {
    ElementIndex::new(idx)
}

const G: f64 = 9.81;
const H_MIN: f64 = 1.0;

fn main() {
    #[cfg(feature = "burn-cuda")]
    type B = Cuda<f64, i64>;
    #[cfg(all(feature = "burn-wgpu", not(feature = "burn-cuda")))]
    type B = Wgpu<f64, i64, u32>;
    #[cfg(all(
        feature = "burn-ndarray",
        not(any(feature = "burn-cuda", feature = "burn-wgpu"))
    ))]
    type B = NdArray<f64>;

    #[cfg(feature = "burn-cuda")]
    let device = CudaDevice::default();
    #[cfg(all(feature = "burn-wgpu", not(feature = "burn-cuda")))]
    let device = WgpuDevice::default();
    #[cfg(all(
        feature = "burn-ndarray",
        not(any(feature = "burn-cuda", feature = "burn-wgpu"))
    ))]
    let device = NdArrayDevice::default();

    // Smaller problem for faster iteration
    let order = 3;
    let nx = 100;
    let ny = 50;

    println!("=== GPU Profiling ===");
    println!("Mesh: {}x{} = {} elements, P{}", nx, ny, nx * ny, order);

    let mesh =
        Mesh2D::uniform_rectangle_with_bc(0.0, 10000.0, 0.0, 5000.0, nx, ny, BoundaryTag::Wall);

    let ops = DGOperators2D::new(order);
    let geom = GeometricFactors2D::compute(&mesh, &ops);
    let n_elements = mesh.n_elements;
    let n_nodes = ops.n_nodes;

    // Initialize state
    let mut state = SWESolution2D::new(n_elements, n_nodes);
    for ki in 0..n_elements {
        for i in 0..n_nodes {
            state.set_state(k(ki), i, SWEState2D::from_primitives(20.0, 0.0, 0.0));
        }
    }

    println!("\n--- Upload timing ---");

    let t0 = Instant::now();
    let burn_ops = BurnOperators2D::<B>::from_cpu(&ops, &device);
    println!("  Operators upload: {:?}", t0.elapsed());

    let t0 = Instant::now();
    let burn_geom = BurnGeometricFactors2D::<B>::from_cpu(&geom, &device);
    println!("  Geometry upload: {:?}", t0.elapsed());

    let t0 = Instant::now();
    let connectivity =
        BurnConnectivity::<B>::from_mesh(&mesh, &geom, &ops.face_nodes, ops.n_face_nodes, &device);
    println!("  Connectivity upload: {:?}", t0.elapsed());

    let t0 = Instant::now();
    let mut burn_state = BurnSWESolution2D::<B>::from_cpu(&state, &device);
    println!("  State upload: {:?}", t0.elapsed());

    let config = BurnRhsConfig {
        g: G,
        h_min: H_MIN,
        coriolis_f: None,
        manning_g_n2: None,
    };

    println!("\n--- compute_dt_burn timing (GPU->CPU sync!) ---");
    for i in 0..3 {
        let t0 = Instant::now();
        let dt = compute_dt_burn(&burn_state, &burn_geom, G, H_MIN, 0.2);
        println!("  dt computation {}: {:?} (dt={:.6})", i, t0.elapsed(), dt);
    }

    println!("\n--- ssp_rk3_step_burn timing ---");
    let dt = 0.1; // Fixed dt to avoid sync in dt computation
    for i in 0..5 {
        let t0 = Instant::now();
        ssp_rk3_step_burn(
            &mut burn_state,
            dt,
            &burn_ops,
            &burn_geom,
            &connectivity,
            &config,
        );
        println!("  RK3 step {}: {:?}", i, t0.elapsed());
    }

    println!("\n--- Individual RHS timing ---");
    // Time just the RHS computation
    use dgcom::solver::burn::compute_rhs_swe_2d_burn;
    for i in 0..5 {
        let t0 = Instant::now();
        let _rhs =
            compute_rhs_swe_2d_burn(&burn_state, &burn_ops, &burn_geom, &connectivity, &config);
        println!("  RHS {}: {:?}", i, t0.elapsed());
    }

    println!("\n--- Download timing ---");
    let t0 = Instant::now();
    let result = burn_state.to_cpu();
    println!("  State download: {:?}", t0.elapsed());

    // Verify result
    let h = result.get_state(k(0), 0).h;
    println!("\nResult check: h[0,0] = {:.6}", h);

    println!("\n=== Summary ===");
    println!("The compute_dt_burn calls force GPU->CPU sync.");
    println!("For real performance, use fixed dt or compute dt less frequently.");
}

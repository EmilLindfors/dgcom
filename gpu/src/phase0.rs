//! Phase 0's gate: the fully wet RHS on the GPU ([`GpuSWE2D`]) against
//! `compute_rhs_swe_2d_parallel_into` on the CPU, node for node, and timed.

use std::sync::Arc;
use std::time::Instant;

use dgcom::SWEState2D;
use dgcom::boundary::Reflective2D;
use dgcom::compute_rhs_swe_2d_parallel_into;
use dgcom::equations::ShallowWater2D;
use dgcom::mesh::{Bathymetry2D, BoundaryTag, ElementFace, Mesh2D};
use dgcom::operators::{DGOperators2D, GeometricFactors2D};
use dgcom::physics::PhysicsModule;
use dgcom::solver::{SWE2DRhsConfig, SWEFormulation2D, SWESolution2D};
use dgcom::time::Integrable;
use dgcom::types::ElementIndex;

use crate::swe2d::{DeviceState, Gpu, GpuSWE2D, NN, ORDER};

const G: f64 = 9.81;

/// Deterministic pseudo-random numbers in [0, 1) (64-bit LCG).
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Element size (m).
const DX: f64 = 100.0;

/// A doubly periodic `nx × ny` mesh of 100 m quadrilaterals, its interior
/// vertices moved by up to `jitter · DX` (0: parallelograms throughout).
fn periodic_mesh(nx: usize, ny: usize, jitter: f64) -> Mesh2D {
    let mut rng = Lcg(2026_11_15);
    let mut vertices = Vec::with_capacity((nx + 1) * (ny + 1));
    for j in 0..=ny {
        for i in 0..=nx {
            let (mut x, mut y) = (i as f64 * DX, j as f64 * DX);
            if i > 0 && i < nx && j > 0 && j < ny {
                x += jitter * DX * (2.0 * rng.next() - 1.0);
                y += jitter * DX * (2.0 * rng.next() - 1.0);
            }
            vertices.push([x, y]);
        }
    }
    let elements = (0..ny)
        .flat_map(|j| {
            (0..nx).map(move |i| {
                let v0 = j * (nx + 1) + i;
                [v0, v0 + 1, v0 + nx + 2, v0 + nx + 1]
            })
        })
        .collect();
    let periodic: Vec<_> = (0..ny)
        .map(|j| {
            (
                ElementFace::new(j * nx + nx - 1, 1),
                ElementFace::new(j * nx, 3),
            )
        })
        .chain((0..nx).map(|i| {
            (
                ElementFace::new((ny - 1) * nx + i, 2),
                ElementFace::new(i, 0),
            )
        }))
        .collect();
    Mesh2D::from_quads(vertices, elements, &periodic, |_| BoundaryTag::Wall)
        .expect("a structured grid is a valid quad mesh")
}

/// A mesh with its operators, geometry, a smooth periodic bed 12–28 m deep
/// and a smooth wet state on it.
pub struct Case {
    pub mesh: Arc<Mesh2D>,
    pub ops: Arc<DGOperators2D>,
    pub geom: Arc<GeometricFactors2D>,
    pub bathymetry: Arc<Bathymetry2D>,
    pub equation: ShallowWater2D,
    pub q: SWESolution2D,
}

impl Case {
    pub fn new(nx: usize, ny: usize, jitter: f64) -> Self {
        let mesh = periodic_mesh(nx, ny, jitter);
        let ops = DGOperators2D::new(ORDER);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let (kx, ky) = (
            std::f64::consts::TAU / (nx as f64 * DX),
            std::f64::consts::TAU / (ny as f64 * DX),
        );
        let bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, y| {
            -20.0 - 8.0 * (kx * x).sin() * (ky * y).cos()
        });
        let mut q = SWESolution2D::new(mesh.n_elements, NN);
        for k in 0..mesh.n_elements {
            let ki = ElementIndex::new(k);
            for i in 0..NN {
                let [x, y] = mesh.reference_to_physical(ki, ops.nodes_r[i], ops.nodes_s[i]);
                let eta = 0.3 * (2.0 * kx * x).cos() * (ky * y).sin();
                let h = eta - bathymetry.get(ki, i);
                let (u, v) = (0.6 * (ky * y).sin(), 0.4 * (kx * x).cos());
                q.set_state(ki, i, SWEState2D::new(h, h * u, h * v));
            }
        }
        Self {
            mesh: Arc::new(mesh),
            ops: Arc::new(ops),
            geom: Arc::new(geom),
            bathymetry: Arc::new(bathymetry),
            equation: ShallowWater2D::new(G),
            q,
        }
    }

    pub fn n_elements(&self) -> usize {
        self.mesh.n_elements
    }

    pub fn affine_elements(&self) -> usize {
        (0..self.n_elements())
            .filter(|&k| self.geom.element_geometry(k).affine)
            .count()
    }

    /// The GPU physics module of this case.
    pub fn gpu_physics(&self, gpu: &Arc<Gpu>) -> GpuSWE2D {
        GpuSWE2D::new(
            gpu,
            self.mesh.clone(),
            self.ops.clone(),
            self.geom.clone(),
            &self.bathymetry,
            &self.equation,
        )
        .expect("GPU physics")
    }
}

/// The CPU kernel the gates compare against.
pub fn cpu_kernel() -> &'static str {
    if cfg!(feature = "simd") {
        "batched (simd)"
    } else {
        "scalar"
    }
}

/// Distance in units in the last place between two finite f64.
fn ulps(a: f64, b: f64) -> u64 {
    let key = |x: f64| {
        let bits = x.to_bits() as i64;
        if bits < 0 { i64::MIN - bits } else { bits }
    };
    key(a).abs_diff(key(b))
}

/// Agreement of two states: nodes identical in all three fields, the largest
/// difference in ulp and relative to each field's largest value.
pub fn agreement(cpu: &SWESolution2D, gpu: &SWESolution2D) -> (usize, u64, f64) {
    let n_nodes = cpu.data[0].len();
    let identical = (0..n_nodes)
        .filter(|&n| (0..3).all(|v| cpu.data[v][n].to_bits() == gpu.data[v][n].to_bits()))
        .count();
    let (mut worst_ulps, mut worst_rel) = (0u64, 0.0f64);
    for v in 0..3 {
        let scale = cpu.data[v].iter().fold(0.0f64, |m, x| m.max(x.abs()));
        for n in 0..n_nodes {
            let (c, g) = (cpu.data[v][n], gpu.data[v][n]);
            assert!(
                g.is_finite(),
                "non-finite GPU value at node {n}, variable {v}"
            );
            worst_ulps = worst_ulps.max(ulps(c, g));
            worst_rel = worst_rel.max((c - g).abs() / scale);
        }
    }
    (identical, worst_ulps, worst_rel)
}

/// One case on both: agreement and, if `iterations > 0`, the time per
/// element evaluation.
fn run_case(
    label: &str,
    case: &Case,
    gpu: &Arc<Gpu>,
    iterations: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let wall = Reflective2D::new();
    let config = SWE2DRhsConfig::new(&case.equation, &wall)
        .with_formulation(SWEFormulation2D::EntropyStable)
        .with_bathymetry(&case.bathymetry);
    let n_el = case.n_elements();
    let rhs_cpu = |out: &mut SWESolution2D| {
        compute_rhs_swe_2d_parallel_into(
            &case.q, &case.mesh, &case.ops, &case.geom, &config, 0.0, out,
        )
    };

    let mut cpu = SWESolution2D::new(n_el, NN);
    rhs_cpu(&mut cpu);

    let physics = case.gpu_physics(gpu);
    let state = DeviceState::upload(gpu, &case.q);
    let mut out = state.zeros_like();
    physics.compute_rhs_into(&state, 0.0, &mut out);
    let (identical, worst_ulps, worst_rel) = agreement(&cpu, &out.download());
    println!(
        "{label}: {n_el} P{ORDER} elements ({} affine), {identical} of {} nodes bit-identical \
         to the {} CPU kernel, largest difference {worst_ulps} ulp, {worst_rel:.1e} of the \
         field's largest",
        case.affine_elements(),
        n_el * NN,
        cpu_kernel()
    );

    if iterations > 0 {
        for _ in 0..3 {
            rhs_cpu(&mut cpu);
        }
        let start = Instant::now();
        for _ in 0..iterations {
            rhs_cpu(&mut cpu);
        }
        let cpu_ns = start.elapsed().as_secs_f64() * 1e9 / (iterations * n_el) as f64;

        for _ in 0..3 {
            physics.compute_rhs_into(&state, 0.0, &mut out);
        }
        gpu.stream.synchronize()?;
        let gpu_iterations = 5 * iterations;
        let start = Instant::now();
        for _ in 0..gpu_iterations {
            physics.compute_rhs_into(&state, 0.0, &mut out);
        }
        gpu.stream.synchronize()?;
        let gpu_ns = start.elapsed().as_secs_f64() * 1e9 / (gpu_iterations * n_el) as f64;
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
        println!(
            "  per element evaluation: CPU {cpu_ns:.1} ns ({threads} threads), GPU {gpu_ns:.1} ns \
             ({:.2}x)",
            cpu_ns / gpu_ns
        );
    }
    Ok(())
}

pub fn run(gpu: &Arc<Gpu>) -> Result<(), Box<dyn std::error::Error>> {
    println!("== Phase 0: the fully wet RHS ==");
    run_case("parallelograms", &Case::new(48, 40, 0.0), gpu, 0)?;
    run_case("jittered", &Case::new(48, 40, 0.15), gpu, 0)?;
    run_case("jittered, timing", &Case::new(320, 300, 0.15), gpu, 20)?;
    run_case("parallelograms, timing", &Case::new(320, 300, 0.0), gpu, 20)?;
    Ok(())
}

//! Benchmarks for time stepping and diagnostics.
//!
//! Run with: `cargo bench --bench time_stepping_bench`
//!
//! Benchmarks SSP-RK3 time integration and diagnostic computations.

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use dgcom::boundary::Reflective2D;
use dgcom::equations::ShallowWater2D;
use dgcom::mesh::Mesh2D;
use dgcom::operators::{DGOperators2D, GeometricFactors2D};
use dgcom::physics::{PhysicsBuilder, SWEPhysics2D};
use dgcom::simulation::Simulation;
use dgcom::solver::{
    DiagnosticsTracker, SWE2DRhsConfig, SWEDiagnostics2D, SWESolution2D, SWEState2D,
    compute_dt_swe_2d, compute_rhs_swe_2d,
};
use dgcom::time::SSPRK3;
use dgcom::types::ElementIndex;
use std::sync::Arc;

/// Setup a test problem.
fn setup_problem(
    nx: usize,
    ny: usize,
    order: usize,
) -> (
    Mesh2D,
    DGOperators2D,
    GeometricFactors2D,
    SWESolution2D,
    ShallowWater2D,
) {
    let mesh = Mesh2D::uniform_rectangle(0.0, 1000.0, 0.0, 1000.0, nx, ny);
    let ops = DGOperators2D::new(order);
    let geom = GeometricFactors2D::compute(&mesh, &ops);
    let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);

    // Initialize with uniform flow
    let h0 = 10.0;
    let u0 = 0.5;
    let v0 = 0.3;
    for k in ElementIndex::iter(q.n_elements) {
        for i in 0..ops.n_nodes {
            q.set_state(k, i, SWEState2D::new(h0, h0 * u0, h0 * v0));
        }
    }

    let equation = ShallowWater2D::new(9.81);

    (mesh, ops, geom, q, equation)
}

/// Benchmark CFL-based time step computation.
fn bench_compute_dt(c: &mut Criterion) {
    let mut group = c.benchmark_group("compute_dt");

    for (nx, ny) in [(8, 8), (16, 16), (32, 32)] {
        let n_elements = nx * ny;
        let (mesh, ops, geom, q, equation) = setup_problem(nx, ny, 3);

        group.bench_with_input(
            BenchmarkId::new("swe_2d", format!("{}_elements", n_elements)),
            &n_elements,
            |b, _| {
                b.iter(|| {
                    compute_dt_swe_2d(
                        black_box(&q),
                        black_box(&mesh),
                        black_box(&geom),
                        black_box(&equation),
                        black_box(ops.order),
                        black_box(0.5),
                    )
                });
            },
        );
    }

    group.finish();
}

/// Production physics (`SWEPhysics2D`, walls, no limiter) for a problem.
fn physics(
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
) -> SWEPhysics2D<Reflective2D> {
    PhysicsBuilder::swe_2d(
        Arc::new(mesh.clone()),
        Arc::new(ops.clone()),
        Arc::new(geom.clone()),
        ShallowWater2D::new(9.81),
        Reflective2D::new(),
    )
    .build()
}

/// Benchmark SSP-RK3 steps on the production path (`Simulation` with
/// `SSPRK3` and `SWEPhysics2D`): one step, which includes allocating the
/// stage workspace, and runs of 10–100 steps, which reuse it.
fn bench_ssp_rk3_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("ssp_rk3_step");
    group.sample_size(30);

    for (nx, ny) in [(8, 8), (16, 16)] {
        let n_elements = nx * ny;
        let (mesh, ops, geom, q, _) = setup_problem(nx, ny, 3);
        let sim = Simulation::new(physics(&mesh, &ops, &geom), SSPRK3).with_dt_max(0.1);
        let mut q_work = q.clone();

        group.bench_with_input(
            BenchmarkId::new("step", format!("{}_elements", n_elements)),
            &n_elements,
            |b, _| {
                b.iter(|| {
                    q_work.clone_from(&q);
                    sim.run(black_box(&mut q_work), 0.0, 0.1)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark multiple time steps (short simulation).
fn bench_multiple_steps(c: &mut Criterion) {
    let mut group = c.benchmark_group("multiple_steps");
    group.sample_size(20);

    let (nx, ny) = (10, 10);
    let (mesh, ops, geom, q, _) = setup_problem(nx, ny, 3);
    let dt = 0.1;
    let sim = Simulation::new(physics(&mesh, &ops, &geom), SSPRK3).with_dt_max(dt);
    let mut q_work = q.clone();

    for n_steps in [10, 50, 100] {
        group.bench_with_input(
            BenchmarkId::new("steps", n_steps.to_string()),
            &n_steps,
            |b, &n_steps| {
                b.iter(|| {
                    q_work.clone_from(&q);
                    let result = sim.run(black_box(&mut q_work), 0.0, n_steps as f64 * dt);
                    assert_eq!(result.n_steps, n_steps);
                    result
                });
            },
        );
    }

    group.finish();
}

/// Benchmark diagnostics computation.
fn bench_diagnostics(c: &mut Criterion) {
    let mut group = c.benchmark_group("diagnostics");

    for (nx, ny) in [(8, 8), (16, 16), (32, 32)] {
        let n_elements = nx * ny;
        let (mesh, ops, geom, q, _) = setup_problem(nx, ny, 3);
        let g = 9.81;
        let dt = 0.1;

        group.bench_with_input(
            BenchmarkId::new("compute", format!("{}_elements", n_elements)),
            &n_elements,
            |b, _| {
                b.iter(|| {
                    SWEDiagnostics2D::compute(
                        black_box(&q),
                        black_box(&mesh),
                        black_box(&ops),
                        black_box(&geom),
                        black_box(g),
                        black_box(dt),
                    )
                });
            },
        );
    }

    group.finish();
}

/// Benchmark diagnostics tracker update.
fn bench_diagnostics_tracker(c: &mut Criterion) {
    let mut group = c.benchmark_group("diagnostics_tracker");

    let (mesh, ops, geom, q, _) = setup_problem(16, 16, 3);
    let g = 9.81;
    let dt = 0.1;

    let initial = SWEDiagnostics2D::compute(&q, &mesh, &ops, &geom, g, dt);
    let mut tracker = DiagnosticsTracker::new(initial.clone());

    // Pre-compute some diagnostics to update with
    let diags: Vec<_> = (0..100)
        .map(|i| {
            // Slightly modify the diagnostics
            let mut d = initial.clone();
            d.total_mass *= 1.0 + (i as f64) * 1e-6;
            d.total_energy *= 1.0 - (i as f64) * 1e-7;
            d
        })
        .collect();

    group.bench_function("update", |b| {
        let mut idx = 0;
        b.iter(|| {
            let diag = &diags[idx % diags.len()];
            tracker.update(black_box(idx as f64), black_box(diag.clone()));
            idx += 1;
        });
    });

    group.bench_function("is_stable", |b| {
        b.iter(|| tracker.is_stable());
    });

    group.bench_function("mass_error", |b| {
        b.iter(|| tracker.mass_error());
    });

    group.finish();
}

/// Benchmark RHS computation as baseline for time stepping.
fn bench_rhs_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("rhs_baseline");
    group.sample_size(50);

    let (nx, ny) = (16, 16);
    let (mesh, ops, geom, q, equation) = setup_problem(nx, ny, 3);
    let bc = Reflective2D::new();
    let config = SWE2DRhsConfig::new(&equation, &bc);

    group.bench_function("single_rhs", |b| {
        b.iter(|| {
            compute_rhs_swe_2d(
                black_box(&q),
                black_box(&mesh),
                black_box(&ops),
                black_box(&geom),
                black_box(&config),
                black_box(0.0),
            )
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_compute_dt,
    bench_ssp_rk3_step,
    bench_multiple_steps,
    bench_diagnostics,
    bench_diagnostics_tracker,
    bench_rhs_baseline
);
criterion_main!(benches);

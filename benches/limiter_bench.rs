//! Benchmarks for slope limiters.
//!
//! Run with: `cargo bench --bench limiter_bench`
//!
//! Benchmarks Kuzmin and positivity limiters for 2D SWE.

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use dgcom::mesh::Mesh2D;
use dgcom::operators::{DGOperators2D, GeometricFactors2D};
use dgcom::solver::{
    KuzminParameter2D, SWEState2D, SystemSolution2D, apply_swe_limiters_kuzmin_2d,
    swe_cell_averages_2d, swe_kuzmin_limiter_2d, swe_positivity_limiter_2d,
};
use dgcom::types::ElementIndex;

/// Setup a test problem with oscillatory solution (needs limiting).
fn setup_problem(
    nx: usize,
    ny: usize,
    order: usize,
) -> (
    Mesh2D,
    DGOperators2D,
    GeometricFactors2D,
    SystemSolution2D<3>,
) {
    let mesh = Mesh2D::uniform_rectangle(0.0, 1000.0, 0.0, 1000.0, nx, ny);
    let ops = DGOperators2D::new(order);
    let geom = GeometricFactors2D::compute(&mesh, &ops);
    let mut q = SystemSolution2D::new(mesh.n_elements, ops.n_nodes);

    // Initialize with solution that has oscillations (needs limiting)
    let n_1d = order + 1;
    for k in ElementIndex::iter(q.n_elements) {
        let verts = mesh.element_vertices(k);
        let cx = (verts[0][0] + verts[2][0]) / 2.0;

        for i in 0..ops.n_nodes {
            let i_r = i % n_1d;
            let i_s = i / n_1d;
            // Add node-level variation to create oscillations
            let r = ops.nodes_1d[i_r];
            let s = ops.nodes_1d[i_s];

            let h = 10.0 + 0.5 * (cx / 100.0).sin() + 0.1 * r * s;
            let u = 0.5 + 0.1 * r;
            let v = 0.3 + 0.05 * s;

            q.set_state(k, i, SWEState2D::new(h, h * u, h * v));
        }
    }

    (mesh, ops, geom, q)
}

/// Benchmark cell average computation.
fn bench_cell_averages(c: &mut Criterion) {
    let mut group = c.benchmark_group("cell_averages");

    for (nx, ny) in [(8, 8), (16, 16), (32, 32)] {
        let n_elements = nx * ny;
        let (_, _, geom, q) = setup_problem(nx, ny, 3);

        group.bench_with_input(
            BenchmarkId::new("swe", format!("{}_elements", n_elements)),
            &n_elements,
            |b, _| {
                b.iter(|| swe_cell_averages_2d(black_box(&q), black_box(&geom)));
            },
        );
    }

    group.finish();
}

/// Benchmark Kuzmin limiter.
fn bench_kuzmin_limiter(c: &mut Criterion) {
    let mut group = c.benchmark_group("kuzmin_limiter");

    for (nx, ny) in [(8, 8), (16, 16), (32, 32)] {
        let n_elements = nx * ny;
        let (mesh, ops, geom, q) = setup_problem(nx, ny, 3);
        let kuzmin = KuzminParameter2D::strict();
        let q_backup = q.clone();

        group.bench_with_input(
            BenchmarkId::new("apply", format!("{}_elements", n_elements)),
            &n_elements,
            |b, _| {
                let mut q_work = q_backup.clone();
                b.iter(|| {
                    q_work.clone_from(&q_backup);
                    swe_kuzmin_limiter_2d(
                        black_box(&mut q_work),
                        black_box(&mesh),
                        black_box(&ops),
                        black_box(&geom),
                        black_box(&kuzmin),
                    )
                });
            },
        );
    }

    group.finish();
}

/// Benchmark positivity limiter.
fn bench_positivity_limiter(c: &mut Criterion) {
    let mut group = c.benchmark_group("positivity_limiter");

    for (nx, ny) in [(8, 8), (16, 16), (32, 32)] {
        let n_elements = nx * ny;
        let (_, _, geom, q) = setup_problem(nx, ny, 3);
        let h_min = 1e-6;
        let q_backup = q.clone();

        group.bench_with_input(
            BenchmarkId::new("apply", format!("{}_elements", n_elements)),
            &n_elements,
            |b, _| {
                let mut q_work = q_backup.clone();
                b.iter(|| {
                    q_work.clone_from(&q_backup);
                    swe_positivity_limiter_2d(
                        black_box(&mut q_work),
                        black_box(&geom),
                        black_box(h_min),
                    )
                });
            },
        );
    }

    group.finish();
}

/// Benchmark combined limiters (Kuzmin + positivity).
fn bench_combined_kuzmin_positivity(c: &mut Criterion) {
    let mut group = c.benchmark_group("combined_kuzmin_positivity");

    for (nx, ny) in [(8, 8), (16, 16), (32, 32)] {
        let n_elements = nx * ny;
        let (mesh, ops, geom, q) = setup_problem(nx, ny, 3);
        let kuzmin = KuzminParameter2D::strict();
        let h_min = 1e-6;
        let q_backup = q.clone();

        group.bench_with_input(
            BenchmarkId::new("apply", format!("{}_elements", n_elements)),
            &n_elements,
            |b, _| {
                let mut q_work = q_backup.clone();
                b.iter(|| {
                    q_work.clone_from(&q_backup);
                    apply_swe_limiters_kuzmin_2d(
                        black_box(&mut q_work),
                        black_box(&mesh),
                        black_box(&ops),
                        black_box(&geom),
                        black_box(&kuzmin),
                        black_box(h_min),
                    )
                });
            },
        );
    }

    group.finish();
}

/// Compare Kuzmin vs positivity-only limiters at same mesh size.
fn bench_limiter_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("limiter_comparison");

    let (nx, ny) = (16, 16);
    let (mesh, ops, geom, q) = setup_problem(nx, ny, 3);
    let kuzmin = KuzminParameter2D::strict();
    let h_min = 1e-6;

    // Reset solution before each iteration
    let q_backup = q.clone();

    group.bench_function("kuzmin_full", |b| {
        let mut q_work = q_backup.clone();
        b.iter(|| {
            q_work.clone_from(&q_backup);
            apply_swe_limiters_kuzmin_2d(
                black_box(&mut q_work),
                black_box(&mesh),
                black_box(&ops),
                black_box(&geom),
                black_box(&kuzmin),
                black_box(h_min),
            )
        });
    });

    group.bench_function("positivity_only", |b| {
        let mut q_work = q_backup.clone();
        b.iter(|| {
            q_work.clone_from(&q_backup);
            swe_positivity_limiter_2d(black_box(&mut q_work), black_box(&geom), black_box(h_min))
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_cell_averages,
    bench_kuzmin_limiter,
    bench_positivity_limiter,
    bench_combined_kuzmin_positivity,
    bench_limiter_comparison
);
criterion_main!(benches);

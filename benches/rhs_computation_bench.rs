//! Benchmarks for full 2D SWE RHS computation.
//!
//! Run with: `cargo bench --bench rhs_computation_bench`
//!
//! Benchmarks the complete RHS computation at various mesh sizes and polynomial orders.

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use dgcom::ElementIndex;
use dgcom::boundary::Reflective2D;
use dgcom::equations::ShallowWater2D;
use dgcom::mesh::Mesh2D;
use dgcom::operators::{DGOperators2D, GeometricFactors2D};
use dgcom::solver::{SWE2DRhsConfig, SWESolution2D, compute_rhs_swe_2d};

#[cfg(all(feature = "parallel", feature = "simd"))]
use dgcom::solver::compute_rhs_swe_2d_parallel;

/// Setup a test problem with uniform flow.
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
    for k in 0..q.n_elements {
        for i in 0..ops.n_nodes {
            q.set_state(
                ElementIndex::new(k),
                i,
                dgcom::solver::SWEState2D::new(h0, h0 * u0, h0 * v0),
            );
        }
    }

    let equation = ShallowWater2D::new(9.81);

    (mesh, ops, geom, q, equation)
}

/// Benchmark RHS computation at different mesh sizes.
fn bench_rhs_mesh_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("rhs_mesh_size");
    group.sample_size(50);

    let order = 3; // P3 elements (16 nodes per element)

    // Include larger mesh sizes to test parallel scaling
    for (nx, ny) in [(4, 4), (8, 8), (16, 16), (32, 32), (64, 64), (100, 100)] {
        let n_elements = nx * ny;
        let (mesh, ops, geom, q, equation) = setup_problem(nx, ny, order);
        let bc = Reflective2D::new();
        let config = SWE2DRhsConfig::new(&equation, &bc);

        group.bench_with_input(
            BenchmarkId::new("serial", format!("{}x{}_{}", nx, ny, n_elements)),
            &n_elements,
            |b, _| {
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
            },
        );

        #[cfg(all(feature = "parallel", feature = "simd"))]
        group.bench_with_input(
            BenchmarkId::new("parallel", format!("{}x{}_{}", nx, ny, n_elements)),
            &n_elements,
            |b, _| {
                b.iter(|| {
                    compute_rhs_swe_2d_parallel(
                        black_box(&q),
                        black_box(&mesh),
                        black_box(&ops),
                        black_box(&geom),
                        black_box(&config),
                        black_box(0.0),
                    )
                });
            },
        );
    }

    group.finish();
}

/// Benchmark RHS computation at different polynomial orders.
fn bench_rhs_polynomial_order(c: &mut Criterion) {
    let mut group = c.benchmark_group("rhs_polynomial_order");
    group.sample_size(50);

    let (nx, ny) = (10, 10); // 100 elements

    for order in [1, 2, 3, 4, 5] {
        let n_nodes = (order + 1) * (order + 1);
        let (mesh, ops, geom, q, equation) = setup_problem(nx, ny, order);
        let bc = Reflective2D::new();
        let config = SWE2DRhsConfig::new(&equation, &bc);

        group.bench_with_input(
            BenchmarkId::new("P", format!("{}_{}_nodes", order, n_nodes)),
            &order,
            |b, _| {
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
            },
        );
    }

    group.finish();
}

/// Benchmark RHS with source terms enabled.
fn bench_rhs_with_sources(c: &mut Criterion) {
    use dgcom::mesh::Bathymetry2D;
    use dgcom::source::{
        BathymetrySource2D, CombinedSource2D, CoriolisSource2D, ManningFriction2D,
    };

    let mut group = c.benchmark_group("rhs_with_sources");
    group.sample_size(50);

    let (nx, ny) = (10, 10);
    let order = 3;
    let (mesh, ops, geom, q, equation) = setup_problem(nx, ny, order);
    let bc = Reflective2D::new();

    // Create bathymetry
    let bathy = Bathymetry2D::constant(mesh.n_elements, ops.n_nodes, 50.0);

    // Create source terms
    let coriolis = CoriolisSource2D::f_plane(1.2e-4);
    let friction = ManningFriction2D::new(0.025, 1e-3);
    let bathy_source = BathymetrySource2D::new(9.81);

    // Benchmark without sources
    let config_no_src = SWE2DRhsConfig::new(&equation, &bc);
    group.bench_function("no_sources", |b| {
        b.iter(|| {
            compute_rhs_swe_2d(
                black_box(&q),
                black_box(&mesh),
                black_box(&ops),
                black_box(&geom),
                black_box(&config_no_src),
                black_box(0.0),
            )
        });
    });

    // Benchmark with Coriolis only
    let config_coriolis = SWE2DRhsConfig::new(&equation, &bc).with_source_terms(&coriolis);
    group.bench_function("coriolis", |b| {
        b.iter(|| {
            compute_rhs_swe_2d(
                black_box(&q),
                black_box(&mesh),
                black_box(&ops),
                black_box(&geom),
                black_box(&config_coriolis),
                black_box(0.0),
            )
        });
    });

    // Benchmark with friction
    let config_friction = SWE2DRhsConfig::new(&equation, &bc).with_source_terms(&friction);
    group.bench_function("friction", |b| {
        b.iter(|| {
            compute_rhs_swe_2d(
                black_box(&q),
                black_box(&mesh),
                black_box(&ops),
                black_box(&geom),
                black_box(&config_friction),
                black_box(0.0),
            )
        });
    });

    // Benchmark with bathymetry
    let config_bathy = SWE2DRhsConfig::new(&equation, &bc)
        .with_bathymetry(&bathy)
        .with_source_terms(&bathy_source);
    group.bench_function("bathymetry", |b| {
        b.iter(|| {
            compute_rhs_swe_2d(
                black_box(&q),
                black_box(&mesh),
                black_box(&ops),
                black_box(&geom),
                black_box(&config_bathy),
                black_box(0.0),
            )
        });
    });

    // Benchmark with all sources combined
    let combined = CombinedSource2D::new(vec![&coriolis, &friction, &bathy_source]);
    let config_all = SWE2DRhsConfig::new(&equation, &bc)
        .with_bathymetry(&bathy)
        .with_source_terms(&combined);
    group.bench_function("all_sources", |b| {
        b.iter(|| {
            compute_rhs_swe_2d(
                black_box(&q),
                black_box(&mesh),
                black_box(&ops),
                black_box(&geom),
                black_box(&config_all),
                black_box(0.0),
            )
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_rhs_mesh_size,
    bench_rhs_polynomial_order,
    bench_rhs_with_sources
);
criterion_main!(benches);

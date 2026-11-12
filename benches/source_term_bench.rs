//! Benchmarks for source term computations.
//!
//! Run with: `cargo bench --bench source_term_bench`
//!
//! Benchmarks Coriolis, Manning friction, bathymetry, and combined source terms.

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use dgcom::mesh::{Bathymetry2D, Mesh2D};
use dgcom::operators::{DGOperators2D, GeometricFactors2D};
use dgcom::solver::{SWESolution2D, SWEState2D};
use dgcom::source::{
    BathymetrySource2D, CombinedSource2D, CoriolisSource2D, ManningFriction2D, SourceContext2D,
    SourceTerm2D,
};
use dgcom::types::ElementIndex;

const G: f64 = 9.81;
const H_MIN: f64 = 1e-6;

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
    Bathymetry2D,
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

    // Create sloped bathymetry
    let mut bathy = Bathymetry2D::constant(mesh.n_elements, ops.n_nodes, 50.0);
    // Add slope in x-direction
    for k in ElementIndex::iter(mesh.n_elements) {
        let verts = mesh.element_vertices(k);
        let cx = (verts[0][0] + verts[2][0]) / 2.0;
        for i in 0..ops.n_nodes {
            let idx = k.as_usize() * ops.n_nodes + i;
            bathy.data[idx] = 50.0 - 0.01 * cx; // 1% slope
        }
    }
    bathy.compute_gradients(&ops, &geom);

    (mesh, ops, geom, q, bathy)
}

/// Create a source context for benchmarking.
fn make_context(
    state: &SWEState2D,
    x: f64,
    y: f64,
    bathy: f64,
    grad: (f64, f64),
) -> SourceContext2D {
    SourceContext2D::new(0.0, (x, y), *state, bathy, grad, G, H_MIN)
}

/// Benchmark Coriolis source term.
fn bench_coriolis(c: &mut Criterion) {
    let mut group = c.benchmark_group("coriolis");

    for (nx, ny) in [(8, 8), (16, 16), (32, 32)] {
        let n_elements = nx * ny;
        let (mesh, ops, _, q, _) = setup_problem(nx, ny, 3);
        let coriolis = CoriolisSource2D::f_plane(1.2e-4);

        // Pre-allocate source output
        let mut results = vec![SWEState2D::zero(); q.n_elements * ops.n_nodes];

        group.bench_with_input(
            BenchmarkId::new("compute", format!("{}_elements", n_elements)),
            &n_elements,
            |b, _| {
                b.iter(|| {
                    for k in ElementIndex::iter(q.n_elements) {
                        let verts = mesh.element_vertices(k);
                        let cx = (verts[0][0] + verts[2][0]) / 2.0;
                        let cy = (verts[0][1] + verts[2][1]) / 2.0;

                        for i in 0..ops.n_nodes {
                            let idx = k.as_usize() * ops.n_nodes + i;
                            let state = q.get_state(k, i);
                            let ctx = make_context(&state, cx, cy, 0.0, (0.0, 0.0));
                            results[idx] = coriolis.evaluate(black_box(&ctx));
                        }
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark Manning friction source term.
fn bench_manning_friction(c: &mut Criterion) {
    let mut group = c.benchmark_group("manning_friction");

    for (nx, ny) in [(8, 8), (16, 16), (32, 32)] {
        let n_elements = nx * ny;
        let (_, ops, _, q, _) = setup_problem(nx, ny, 3);
        let friction = ManningFriction2D::new(G, 0.025);

        // Pre-allocate source output
        let mut results = vec![SWEState2D::zero(); q.n_elements * ops.n_nodes];

        group.bench_with_input(
            BenchmarkId::new("compute", format!("{}_elements", n_elements)),
            &n_elements,
            |b, _| {
                b.iter(|| {
                    for k in ElementIndex::iter(q.n_elements) {
                        for i in 0..ops.n_nodes {
                            let idx = k.as_usize() * ops.n_nodes + i;
                            let state = q.get_state(k, i);
                            let ctx = make_context(&state, 0.0, 0.0, 0.0, (0.0, 0.0));
                            results[idx] = friction.evaluate(black_box(&ctx));
                        }
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark bathymetry source term.
fn bench_bathymetry_source(c: &mut Criterion) {
    let mut group = c.benchmark_group("bathymetry_source");

    for (nx, ny) in [(8, 8), (16, 16), (32, 32)] {
        let n_elements = nx * ny;
        let (_, ops, _, q, bathy) = setup_problem(nx, ny, 3);
        let bathy_source = BathymetrySource2D::new(G);

        // Pre-allocate source output
        let mut results = vec![SWEState2D::zero(); q.n_elements * ops.n_nodes];

        group.bench_with_input(
            BenchmarkId::new("compute", format!("{}_elements", n_elements)),
            &n_elements,
            |b, _| {
                b.iter(|| {
                    for k in ElementIndex::iter(q.n_elements) {
                        for i in 0..ops.n_nodes {
                            let idx = k.as_usize() * ops.n_nodes + i;
                            let state = q.get_state(k, i);
                            let bathy_val = bathy.data[idx];
                            let grad = (bathy.gradient_x[idx], bathy.gradient_y[idx]);
                            let ctx = make_context(&state, 0.0, 0.0, bathy_val, grad);
                            results[idx] = bathy_source.evaluate(black_box(&ctx));
                        }
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark combined source terms.
fn bench_combined_sources(c: &mut Criterion) {
    let mut group = c.benchmark_group("combined_sources");

    let (mesh, ops, _, q, bathy) = setup_problem(16, 16, 3);
    let coriolis = CoriolisSource2D::f_plane(1.2e-4);
    let friction = ManningFriction2D::new(G, 0.025);
    let bathy_source = BathymetrySource2D::new(G);

    // Pre-allocate source output
    let mut results = vec![SWEState2D::zero(); q.n_elements * ops.n_nodes];

    // Individual sources for comparison
    group.bench_function("coriolis_only", |b| {
        b.iter(|| {
            for k in ElementIndex::iter(q.n_elements) {
                let verts = mesh.element_vertices(k);
                let cx = (verts[0][0] + verts[2][0]) / 2.0;
                let cy = (verts[0][1] + verts[2][1]) / 2.0;
                for i in 0..ops.n_nodes {
                    let idx = k.as_usize() * ops.n_nodes + i;
                    let state = q.get_state(k, i);
                    let ctx = make_context(&state, cx, cy, 0.0, (0.0, 0.0));
                    results[idx] = coriolis.evaluate(black_box(&ctx));
                }
            }
        });
    });

    group.bench_function("friction_only", |b| {
        b.iter(|| {
            for k in ElementIndex::iter(q.n_elements) {
                for i in 0..ops.n_nodes {
                    let idx = k.as_usize() * ops.n_nodes + i;
                    let state = q.get_state(k, i);
                    let ctx = make_context(&state, 0.0, 0.0, 0.0, (0.0, 0.0));
                    results[idx] = friction.evaluate(black_box(&ctx));
                }
            }
        });
    });

    group.bench_function("bathymetry_only", |b| {
        b.iter(|| {
            for k in ElementIndex::iter(q.n_elements) {
                for i in 0..ops.n_nodes {
                    let idx = k.as_usize() * ops.n_nodes + i;
                    let state = q.get_state(k, i);
                    let bathy_val = bathy.data[idx];
                    let grad = (bathy.gradient_x[idx], bathy.gradient_y[idx]);
                    let ctx = make_context(&state, 0.0, 0.0, bathy_val, grad);
                    results[idx] = bathy_source.evaluate(black_box(&ctx));
                }
            }
        });
    });

    // Combined using CombinedSource2D
    let combined = CombinedSource2D::new(vec![&coriolis, &friction, &bathy_source]);
    group.bench_function("combined_all", |b| {
        b.iter(|| {
            for k in ElementIndex::iter(q.n_elements) {
                let verts = mesh.element_vertices(k);
                let cx = (verts[0][0] + verts[2][0]) / 2.0;
                let cy = (verts[0][1] + verts[2][1]) / 2.0;
                for i in 0..ops.n_nodes {
                    let idx = k.as_usize() * ops.n_nodes + i;
                    let state = q.get_state(k, i);
                    let bathy_val = bathy.data[idx];
                    let grad = (bathy.gradient_x[idx], bathy.gradient_y[idx]);
                    let ctx = make_context(&state, cx, cy, bathy_val, grad);
                    results[idx] = combined.evaluate(black_box(&ctx));
                }
            }
        });
    });

    group.finish();
}

/// Benchmark source term overhead in varying flow conditions.
fn bench_source_flow_conditions(c: &mut Criterion) {
    let mut group = c.benchmark_group("source_flow_conditions");

    let (mesh, ops, _, _, _) = setup_problem(16, 16, 3);

    // Test different flow conditions
    let conditions = [
        ("uniform_slow", 10.0, 0.1, 0.1),
        ("uniform_fast", 10.0, 2.0, 1.5),
        ("shallow_slow", 1.0, 0.1, 0.1),
        ("shallow_fast", 1.0, 1.0, 0.5),
        ("deep_fast", 50.0, 3.0, 2.0),
    ];

    let friction = ManningFriction2D::new(G, 0.025);

    for (name, h, u, v) in conditions {
        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(q.n_elements) {
            for i in 0..ops.n_nodes {
                q.set_state(k, i, SWEState2D::new(h, h * u, h * v));
            }
        }

        let mut results = vec![SWEState2D::zero(); q.n_elements * ops.n_nodes];

        group.bench_function(name, |b| {
            b.iter(|| {
                for k in ElementIndex::iter(q.n_elements) {
                    for i in 0..ops.n_nodes {
                        let idx = k.as_usize() * ops.n_nodes + i;
                        let state = q.get_state(k, i);
                        let ctx = make_context(&state, 0.0, 0.0, 0.0, (0.0, 0.0));
                        results[idx] = friction.evaluate(black_box(&ctx));
                    }
                }
            });
        });
    }

    group.finish();
}

/// Benchmark bathymetry gradient computation (expensive operation).
fn bench_bathymetry_gradients(c: &mut Criterion) {
    let mut group = c.benchmark_group("bathymetry_gradients");

    for (nx, ny) in [(8, 8), (16, 16), (32, 32)] {
        let n_elements = nx * ny;
        let (mesh, ops, geom, _, _) = setup_problem(nx, ny, 3);

        // Create bathymetry without gradients
        let mut bathy = Bathymetry2D::flat(mesh.n_elements, ops.n_nodes);
        // Add some variation
        for k in ElementIndex::iter(mesh.n_elements) {
            let verts = mesh.element_vertices(k);
            let cx = (verts[0][0] + verts[2][0]) / 2.0;
            let cy = (verts[0][1] + verts[2][1]) / 2.0;
            for i in 0..ops.n_nodes {
                let idx = k.as_usize() * ops.n_nodes + i;
                bathy.data[idx] = 50.0 - 0.01 * cx + 0.005 * (cy / 100.0).sin();
            }
        }

        group.bench_with_input(
            BenchmarkId::new("compute_gradients", format!("{}_elements", n_elements)),
            &n_elements,
            |b, _| {
                b.iter(|| {
                    bathy.compute_gradients(black_box(&ops), black_box(&geom));
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_coriolis,
    bench_manning_friction,
    bench_bathymetry_source,
    bench_combined_sources,
    bench_source_flow_conditions,
    bench_bathymetry_gradients
);
criterion_main!(benches);

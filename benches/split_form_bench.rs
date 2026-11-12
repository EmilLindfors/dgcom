//! The production 2D SWE RHS: the `WetDry` split form (Wintermeyer et al.
//! 2017 flux differencing, hydrostatic HLL faces, subcells at shorelines).
//!
//! Run with: `cargo bench --bench split_form --no-default-features --features parallel,simd`
//!
//! A periodic 128 × 128 P2 mesh (16,384 elements, 147k nodes) over a bed with
//! islands (≈ 3 % of the elements take the subcells), η = 0.3 m with a
//! current. `affine` keeps the uniform parallelograms; `curved` moves the
//! vertices so every element is a general quadrilateral, as on a Gmsh
//! coastline mesh. `shoreline` is `curved` over tidal flats where 33 % of
//! the elements take the subcells (none of them dry), close to the Frøya
//! coastline mesh (38.5 %). Serial and parallel full RHS evaluations.
//!
//! `split_form_subset_p2` is the kernels' cost as local time stepping runs
//! them (issue #229): the subset RHS over lists that skip through the
//! `shoreline` case's elements in Hilbert order (a rectangle, since the
//! reordering keeps no periodic pairs apart from the mesh's own), in ns per
//! element evaluation. Compare runs made back to back: a laptop's clock
//! falls within seconds of load, by 2× on the one these were written on.

use std::f64::consts::PI;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use dgcom::boundary::Reflective2D;
use dgcom::equations::ShallowWater2D;
use dgcom::mesh::{Bathymetry2D, Mesh2D};
use dgcom::operators::{DGOperators2D, GeometricFactors2D};
use dgcom::solver::{
    SWE2DRhsConfig, SWEFormulation2D, SWESolution2D, SWEState2D, compute_rhs_swe_2d_into,
    compute_rhs_swe_2d_subset_then,
};
use dgcom::types::ElementIndex;

const G: f64 = 9.81;
const L: f64 = 20_000.0;
const N: usize = 128;
const ORDER: usize = 2;

struct Case {
    mesh: Mesh2D,
    ops: DGOperators2D,
    geom: GeometricFactors2D,
    bathymetry: Bathymetry2D,
    q: SWESolution2D,
}

fn case(curved: bool, shoreline: bool) -> Case {
    case_on(
        Mesh2D::uniform_periodic(0.0, L, 0.0, L, N, N),
        curved,
        shoreline,
    )
}

fn case_on(mut mesh: Mesh2D, curved: bool, shoreline: bool) -> Case {
    let tau = 2.0 * PI / L;
    if curved {
        // About 10 % of an element's size: general quadrilaterals everywhere
        let a = 0.1 * L / N as f64;
        for v in &mut mesh.vertices {
            let [x, y] = *v;
            let (sx, sy) = ((9.0 * tau * x).sin(), (7.0 * tau * y).sin());
            *v = [x + a * sx * sy, y - 0.7 * a * sx * sy];
        }
    }
    let ops = DGOperators2D::new(ORDER);
    let geom = GeometricFactors2D::compute(&mesh, &ops);
    let bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, y| {
        if shoreline {
            // Shoals a few elements across, their tops above η = 0.3
            -0.4 + (24.0 * tau * x).cos() * (20.0 * tau * y).cos()
        } else {
            // Deep channels with a few islands reaching above η = 0.3
            -15.0 + 17.0 * (3.0 * tau * x).cos() * (2.0 * tau * y).cos()
        }
    });
    let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
    for k in ElementIndex::iter(mesh.n_elements) {
        for i in 0..ops.n_nodes {
            let [x, y] = mesh.reference_to_physical(k, ops.nodes_r[i], ops.nodes_s[i]);
            let h = (0.3 - bathymetry.get(k, i)).max(0.0);
            let (u, v) = if h > 0.0 {
                (0.5 * (tau * y).cos(), 0.3 * (tau * x).sin())
            } else {
                (0.0, 0.0)
            };
            q.set_state(k, i, SWEState2D::from_primitives(h, u, v));
        }
    }
    Case {
        mesh,
        ops,
        geom,
        bathymetry,
        q,
    }
}

fn bench_split_form(c: &mut Criterion) {
    let equation = ShallowWater2D::new(G);
    let bc = Reflective2D::new();
    let mut group = c.benchmark_group("split_form_wet_dry_p2");
    group.sample_size(30);
    for (name, curved, shoreline) in [
        ("affine", false, false),
        ("curved", true, false),
        ("shoreline", true, true),
    ] {
        let Case {
            mesh,
            ops,
            geom,
            bathymetry,
            q,
        } = case(curved, shoreline);
        let config = SWE2DRhsConfig::new(&equation, &bc)
            .with_coriolis(false)
            .with_formulation(SWEFormulation2D::WetDry)
            .with_bathymetry(&bathymetry)
            .with_dry_threshold(1e-3);
        let mut out = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        group.throughput(Throughput::Elements((mesh.n_elements * ops.n_nodes) as u64));
        group.bench_function(BenchmarkId::new("serial", name), |b| {
            b.iter(|| {
                compute_rhs_swe_2d_into(black_box(&q), &mesh, &ops, &geom, &config, 0.0, &mut out)
            })
        });
        #[cfg(feature = "parallel")]
        group.bench_function(BenchmarkId::new("parallel", name), |b| {
            b.iter(|| {
                dgcom::solver::compute_rhs_swe_2d_parallel_into(
                    black_box(&q),
                    &mesh,
                    &ops,
                    &geom,
                    &config,
                    0.0,
                    &mut out,
                )
            })
        });
    }
    group.finish();
}

/// The subset RHS of local time stepping: the listed elements only, each
/// with the stage's continuation left empty.
fn bench_subset(c: &mut Criterion) {
    let equation = ShallowWater2D::new(G);
    let bc = Reflective2D::new();
    let mut group = c.benchmark_group("split_form_subset_p2");
    group.sample_size(30);
    // Vertices moved first, so that the curve orders the elements as built
    let Case {
        mesh,
        ops,
        geom,
        bathymetry,
        q,
    } = case_on(
        Mesh2D::uniform_rectangle(0.0, L, 0.0, L, N, N)
            .in_hilbert_order()
            .0,
        true,
        true,
    );
    let config = SWE2DRhsConfig::new(&equation, &bc)
        .with_coriolis(false)
        .with_formulation(SWEFormulation2D::WetDry)
        .with_bathymetry(&bathymetry)
        .with_dry_threshold(1e-3);
    let mut out = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
    let all = 0..mesh.n_elements as u32;
    for (name, elements) in [
        ("all", all.clone().collect::<Vec<_>>()),
        ("two_of_three", all.clone().filter(|k| k % 3 != 0).collect()),
        ("runs_of_40_in_100", all.filter(|k| k % 100 < 40).collect()),
    ] {
        group.throughput(Throughput::Elements(elements.len() as u64));
        group.bench_function(BenchmarkId::new("subset", name), |b| {
            b.iter(|| {
                compute_rhs_swe_2d_subset_then(
                    black_box(&q),
                    &mesh,
                    &ops,
                    &geom,
                    &config,
                    &elements,
                    &|_| 0.0,
                    &mut out,
                    None,
                    &|_, _, _| {},
                )
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_split_form, bench_subset);
criterion_main!(benches);

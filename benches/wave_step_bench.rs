//! The spectral wave model's step and its parts.
//!
//! Run with: `cargo bench --bench wave_step --no-default-features --features parallel,simd`
//! (one thread: `RAYON_NUM_THREADS=1`).
//!
//! As the Frøya wave runs (`froya_real_data waves=… wave_mesh=60,45
//! implicit=1`): a 60 × 45 km P1 grid of 1 km elements (2,700 elements,
//! 10,800 nodes, open boundaries) over a bed shoaling from 150 m to 5 m with
//! banks, 25 × 36 components (0.04–0.5 Hz), SWAN's default sources under a
//! 15 m/s wind, a sheared current, implicit refraction and frequency
//! shifting, and a JONSWAP sea everywhere and on the boundary.
//!
//! `step` is the production path; the node passes alone (`sources`,
//! `implicit_*`) are the public `apply_*` functions. `sources_without_dia`
//! leaves out the quadruplets, so the DIA's share is `sources` less it.

use std::f64::consts::PI;
use std::sync::Arc;

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use dg_rs::mesh::{Bathymetry2D, BoundaryTag, Mesh2D};
use dg_rs::operators::{DGOperators2D, GeometricFactors2D};
use dg_rs::types::ElementIndex;
use dg_rs::waves::{SourceTerms, SpectralGrid, WaveModel2D, WaveSolution, WaveWorkspace, Wind};

const G: f64 = 9.81;
const LX: f64 = 60_000.0;
const LY: f64 = 45_000.0;

fn build(sources: SourceTerms) -> (WaveModel2D, WaveSolution) {
    let mesh = Mesh2D::uniform_rectangle_with_bc(0.0, LX, 0.0, LY, 60, 45, BoundaryTag::Open);
    let ops = DGOperators2D::new(1);
    let geom = GeometricFactors2D::compute(&mesh, &ops);
    let bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, y| {
        let shoaling = 150.0 - 145.0 * x / LX;
        let banks =
            0.4 * shoaling * (2.0 * PI * y / 15_000.0).sin() * (2.0 * PI * x / 20_000.0).cos();
        -(shoaling - banks).max(2.0)
    });
    let grid = SpectralGrid::new(0.04, 0.5, 25, 36);
    let sea = grid.jonswap(3.0, 10.0, 3.3, 0.3, 4.0);
    let mut model = WaveModel2D::new(
        Arc::new(mesh),
        Arc::new(ops),
        Arc::new(geom),
        &bathymetry,
        grid,
        G,
    )
    .with_sources(sources)
    .with_boundary_spectrum(&sea)
    .with_wind(Wind {
        u10: 15.0,
        direction: 0.3,
    })
    .with_implicit_refraction(true)
    .with_implicit_frequency_shift(true);
    // A sheared current, 0.5 m/s at most
    let (mesh, ops) = (model.mesh.clone(), model.ops.clone());
    let n = mesh.n_elements * ops.n_nodes;
    let (mut u, mut v) = (vec![0.0; n], vec![0.0; n]);
    for k in ElementIndex::iter(mesh.n_elements) {
        for a in 0..ops.n_nodes {
            let [x, y] = mesh.reference_to_physical(k, ops.nodes_r[a], ops.nodes_s[a]);
            let p = k.as_usize() * ops.n_nodes + a;
            u[p] = 0.5 * (2.0 * PI * y / LY).sin();
            v[p] = 0.2 * (2.0 * PI * x / LX).cos();
        }
    }
    model.set_currents(&u, &v);
    let state = model.uniform_state(&sea);
    (model, state)
}

fn bench_wave_step(c: &mut Criterion) {
    let (model, state) = build(SourceTerms::swan_defaults(G));
    let dt = model.compute_dt(0.5);
    let mut group = c.benchmark_group("wave_step_p1");
    group.sample_size(10);
    group.throughput(Throughput::Elements(state.data.len() as u64));
    let mut ws = WaveWorkspace::default();

    let mut n = state.clone();
    group.bench_function("step", |b| {
        b.iter(|| model.step(black_box(&mut n), dt, &mut ws))
    });
    let mut n = state.clone();
    group.bench_function("sources", |b| {
        b.iter(|| model.apply_sources(black_box(&mut n), dt))
    });
    // The sources less the DIA: the DIA's share is the difference
    let (without_dia, _) = build(SourceTerms::swan_defaults(G).with_quadruplets(None));
    let mut n = state.clone();
    group.bench_function("sources_without_dia", |b| {
        b.iter(|| without_dia.apply_sources(black_box(&mut n), dt))
    });
    let mut n = state.clone();
    group.bench_function("implicit_refraction", |b| {
        b.iter(|| model.apply_implicit_refraction(black_box(&mut n), 0.5 * dt))
    });
    let mut n = state.clone();
    group.bench_function("implicit_frequency_shift", |b| {
        b.iter(|| model.apply_implicit_frequency_shift(black_box(&mut n), 0.5 * dt))
    });
    group.finish();
}

criterion_group!(benches, bench_wave_step);
criterion_main!(benches);

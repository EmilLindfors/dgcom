//! Benchmarks for 2D SWE RHS computation comparing scalar and SIMD implementations.
//!
//! Run with: `cargo bench --features simd`
//!
//! The benchmarks compare:
//! - Scalar vs SIMD differentiation matrix application
//! - Scalar vs SIMD LIFT matrix application
//! - Scalar vs SIMD Coriolis source term
//! - Scalar vs SIMD derivative combination

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use dgcom::solver::{
    apply_diff_matrix_scalar, apply_lift_scalar, combine_derivatives_scalar, coriolis_source_scalar,
};

#[cfg(feature = "simd")]
use dgcom::solver::{apply_diff_matrix, apply_lift, combine_derivatives, coriolis_source};

/// Generate deterministic pseudo-random data for benchmarks.
fn random_vec(n: usize, seed: u64) -> Vec<f64> {
    let mut v = Vec::with_capacity(n);
    let mut x = seed;
    for _ in 0..n {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
        let val = (x as f64) / (u64::MAX as f64) * 2.0 - 1.0;
        v.push(val);
    }
    v
}

/// Benchmark differentiation matrix application.
fn bench_diff_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("diff_matrix");

    // Test different polynomial orders (n_nodes = (order+1)²)
    for (label, n_nodes) in [("P2_9", 9), ("P3_16", 16), ("P4_25", 25), ("P5_36", 36)] {
        // Setup test data
        let d = random_vec(n_nodes * n_nodes, 42);
        let flux_h = random_vec(n_nodes, 1);
        let flux_hu = random_vec(n_nodes, 2);
        let flux_hv = random_vec(n_nodes, 3);

        group.throughput(Throughput::Elements(n_nodes as u64));

        // Scalar benchmark
        group.bench_with_input(
            BenchmarkId::new("scalar", label),
            &n_nodes,
            |b, &n_nodes| {
                let mut out_h = vec![0.0; n_nodes];
                let mut out_hu = vec![0.0; n_nodes];
                let mut out_hv = vec![0.0; n_nodes];

                b.iter(|| {
                    apply_diff_matrix_scalar(
                        black_box(&d),
                        black_box(&flux_h),
                        black_box(&flux_hu),
                        black_box(&flux_hv),
                        black_box(&mut out_h),
                        black_box(&mut out_hu),
                        black_box(&mut out_hv),
                        black_box(n_nodes),
                    );
                });
            },
        );

        // SIMD benchmark
        #[cfg(feature = "simd")]
        group.bench_with_input(BenchmarkId::new("simd", label), &n_nodes, |b, &n_nodes| {
            let mut out_h = vec![0.0; n_nodes];
            let mut out_hu = vec![0.0; n_nodes];
            let mut out_hv = vec![0.0; n_nodes];

            b.iter(|| {
                apply_diff_matrix(
                    black_box(&d),
                    black_box(&flux_h),
                    black_box(&flux_hu),
                    black_box(&flux_hv),
                    black_box(&mut out_h),
                    black_box(&mut out_hu),
                    black_box(&mut out_hv),
                    black_box(n_nodes),
                );
            });
        });
    }

    group.finish();
}

/// Benchmark LIFT matrix application.
fn bench_lift_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("lift_matrix");

    // Test different polynomial orders
    for (label, n_nodes, n_face_nodes) in
        [("P2", 9, 3), ("P3", 16, 4), ("P4", 25, 5), ("P5", 36, 6)]
    {
        let lift = random_vec(n_nodes * n_face_nodes, 42);
        let flux_diff_h = random_vec(n_face_nodes, 1);
        let flux_diff_hu = random_vec(n_face_nodes, 2);
        let flux_diff_hv = random_vec(n_face_nodes, 3);
        let scale = 0.5;

        group.throughput(Throughput::Elements(n_nodes as u64));

        // Scalar benchmark
        group.bench_with_input(
            BenchmarkId::new("scalar", label),
            &(n_nodes, n_face_nodes),
            |b, &(n_nodes, n_face_nodes)| {
                let mut out_h = vec![0.0; n_nodes];
                let mut out_hu = vec![0.0; n_nodes];
                let mut out_hv = vec![0.0; n_nodes];

                b.iter(|| {
                    apply_lift_scalar(
                        black_box(&lift),
                        black_box(&flux_diff_h),
                        black_box(&flux_diff_hu),
                        black_box(&flux_diff_hv),
                        black_box(&mut out_h),
                        black_box(&mut out_hu),
                        black_box(&mut out_hv),
                        black_box(n_nodes),
                        black_box(n_face_nodes),
                        black_box(scale),
                    );
                });
            },
        );

        // SIMD benchmark
        #[cfg(feature = "simd")]
        group.bench_with_input(
            BenchmarkId::new("simd", label),
            &(n_nodes, n_face_nodes),
            |b, &(n_nodes, n_face_nodes)| {
                let mut out_h = vec![0.0; n_nodes];
                let mut out_hu = vec![0.0; n_nodes];
                let mut out_hv = vec![0.0; n_nodes];

                b.iter(|| {
                    apply_lift(
                        black_box(&lift),
                        black_box(&flux_diff_h),
                        black_box(&flux_diff_hu),
                        black_box(&flux_diff_hv),
                        black_box(&mut out_h),
                        black_box(&mut out_hu),
                        black_box(&mut out_hv),
                        black_box(n_nodes),
                        black_box(n_face_nodes),
                        black_box(scale),
                    );
                });
            },
        );
    }

    group.finish();
}

/// Benchmark Coriolis source term.
fn bench_coriolis(c: &mut Criterion) {
    let mut group = c.benchmark_group("coriolis");

    for (label, n_nodes) in [("P2_9", 9), ("P3_16", 16), ("P4_25", 25), ("P5_36", 36)] {
        let hu = random_vec(n_nodes, 1);
        let hv = random_vec(n_nodes, 2);
        let f = 1.2e-4;

        group.throughput(Throughput::Elements(n_nodes as u64));

        // Scalar benchmark
        group.bench_with_input(
            BenchmarkId::new("scalar", label),
            &n_nodes,
            |b, &n_nodes| {
                let mut out_hu = vec![0.0; n_nodes];
                let mut out_hv = vec![0.0; n_nodes];

                b.iter(|| {
                    coriolis_source_scalar(
                        black_box(&hu),
                        black_box(&hv),
                        black_box(&mut out_hu),
                        black_box(&mut out_hv),
                        black_box(f),
                        black_box(n_nodes),
                    );
                });
            },
        );

        // SIMD benchmark
        #[cfg(feature = "simd")]
        group.bench_with_input(BenchmarkId::new("simd", label), &n_nodes, |b, &n_nodes| {
            let mut out_hu = vec![0.0; n_nodes];
            let mut out_hv = vec![0.0; n_nodes];

            b.iter(|| {
                coriolis_source(
                    black_box(&hu),
                    black_box(&hv),
                    black_box(&mut out_hu),
                    black_box(&mut out_hv),
                    black_box(f),
                    black_box(n_nodes),
                );
            });
        });
    }

    group.finish();
}

/// Benchmark derivative combination with geometric factors.
fn bench_combine_derivatives(c: &mut Criterion) {
    let mut group = c.benchmark_group("combine_derivatives");

    for (label, n_nodes) in [("P2_9", 9), ("P3_16", 16), ("P4_25", 25), ("P5_36", 36)] {
        let dfx_dr_h = random_vec(n_nodes, 1);
        let dfx_dr_hu = random_vec(n_nodes, 2);
        let dfx_dr_hv = random_vec(n_nodes, 3);
        let dfx_ds_h = random_vec(n_nodes, 4);
        let dfx_ds_hu = random_vec(n_nodes, 5);
        let dfx_ds_hv = random_vec(n_nodes, 6);
        let dfy_dr_h = random_vec(n_nodes, 7);
        let dfy_dr_hu = random_vec(n_nodes, 8);
        let dfy_dr_hv = random_vec(n_nodes, 9);
        let dfy_ds_h = random_vec(n_nodes, 10);
        let dfy_ds_hu = random_vec(n_nodes, 11);
        let dfy_ds_hv = random_vec(n_nodes, 12);
        let (rx, sx, ry, sy) = (0.5, 0.3, -0.2, 0.4);

        group.throughput(Throughput::Elements(n_nodes as u64));

        // Scalar benchmark
        group.bench_with_input(
            BenchmarkId::new("scalar", label),
            &n_nodes,
            |b, &n_nodes| {
                let mut out_h = vec![0.0; n_nodes];
                let mut out_hu = vec![0.0; n_nodes];
                let mut out_hv = vec![0.0; n_nodes];

                b.iter(|| {
                    combine_derivatives_scalar(
                        black_box(&dfx_dr_h),
                        black_box(&dfx_dr_hu),
                        black_box(&dfx_dr_hv),
                        black_box(&dfx_ds_h),
                        black_box(&dfx_ds_hu),
                        black_box(&dfx_ds_hv),
                        black_box(&dfy_dr_h),
                        black_box(&dfy_dr_hu),
                        black_box(&dfy_dr_hv),
                        black_box(&dfy_ds_h),
                        black_box(&dfy_ds_hu),
                        black_box(&dfy_ds_hv),
                        black_box(&mut out_h),
                        black_box(&mut out_hu),
                        black_box(&mut out_hv),
                        black_box(rx),
                        black_box(sx),
                        black_box(ry),
                        black_box(sy),
                        black_box(n_nodes),
                    );
                });
            },
        );

        // SIMD benchmark
        #[cfg(feature = "simd")]
        group.bench_with_input(BenchmarkId::new("simd", label), &n_nodes, |b, &n_nodes| {
            let mut out_h = vec![0.0; n_nodes];
            let mut out_hu = vec![0.0; n_nodes];
            let mut out_hv = vec![0.0; n_nodes];

            b.iter(|| {
                combine_derivatives(
                    black_box(&dfx_dr_h),
                    black_box(&dfx_dr_hu),
                    black_box(&dfx_dr_hv),
                    black_box(&dfx_ds_h),
                    black_box(&dfx_ds_hu),
                    black_box(&dfx_ds_hv),
                    black_box(&dfy_dr_h),
                    black_box(&dfy_dr_hu),
                    black_box(&dfy_dr_hv),
                    black_box(&dfy_ds_h),
                    black_box(&dfy_ds_hu),
                    black_box(&dfy_ds_hv),
                    black_box(&mut out_h),
                    black_box(&mut out_hu),
                    black_box(&mut out_hv),
                    black_box(rx),
                    black_box(sx),
                    black_box(ry),
                    black_box(sy),
                    black_box(n_nodes),
                );
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_diff_matrix,
    bench_lift_matrix,
    bench_coriolis,
    bench_combine_derivatives
);
criterion_main!(benches);

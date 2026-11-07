//! Benchmarks for numerical flux functions.
//!
//! Run with: `cargo bench --bench flux_bench`
//!
//! Compares performance of different numerical flux schemes for 2D SWE.

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use dgcom::flux::{hll_flux_swe_2d, roe_flux_swe_2d, rusanov_flux_swe_2d};
use dgcom::solver::SWEState2D;

const H_MIN: f64 = 1e-6;

/// Generate test states for flux computation.
fn generate_test_states(n: usize) -> Vec<(SWEState2D, SWEState2D, (f64, f64))> {
    let mut states = Vec::with_capacity(n);
    for i in 0..n {
        let phase = (i as f64) * 0.1;

        // Left state
        let h_l = 10.0 + 2.0 * phase.sin();
        let u_l = 0.5 + 0.3 * phase.cos();
        let v_l = 0.2 - 0.1 * phase.sin();
        let left = SWEState2D::new(h_l, h_l * u_l, h_l * v_l);

        // Right state (slightly different)
        let h_r = 10.0 + 1.5 * (phase + 0.5).sin();
        let u_r = 0.4 + 0.2 * (phase + 0.3).cos();
        let v_r = 0.3 - 0.15 * (phase + 0.2).sin();
        let right = SWEState2D::new(h_r, h_r * u_r, h_r * v_r);

        // Normal direction (unit vector)
        let angle = phase * 0.5;
        let normal = (angle.cos(), angle.sin());

        states.push((left, right, normal));
    }
    states
}

/// Benchmark individual flux functions.
fn bench_flux_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("flux_functions");

    let g = 9.81;
    let states = generate_test_states(1000);

    // Benchmark Roe flux
    group.bench_function("roe", |b| {
        b.iter(|| {
            let mut total_h = 0.0;
            for (left, right, normal) in &states {
                let flux = roe_flux_swe_2d(
                    black_box(left),
                    black_box(right),
                    black_box(*normal),
                    black_box(g),
                    black_box(H_MIN),
                );
                total_h += flux.h;
            }
            total_h
        });
    });

    // Benchmark HLL flux
    group.bench_function("hll", |b| {
        b.iter(|| {
            let mut total_h = 0.0;
            for (left, right, normal) in &states {
                let flux = hll_flux_swe_2d(
                    black_box(left),
                    black_box(right),
                    black_box(*normal),
                    black_box(g),
                    black_box(H_MIN),
                );
                total_h += flux.h;
            }
            total_h
        });
    });

    // Benchmark Rusanov flux
    group.bench_function("rusanov", |b| {
        b.iter(|| {
            let mut total_h = 0.0;
            for (left, right, normal) in &states {
                let flux = rusanov_flux_swe_2d(
                    black_box(left),
                    black_box(right),
                    black_box(*normal),
                    black_box(g),
                    black_box(H_MIN),
                );
                total_h += flux.h;
            }
            total_h
        });
    });

    group.finish();
}

/// Benchmark flux computation with varying depth ratios.
fn bench_flux_depth_ratio(c: &mut Criterion) {
    let mut group = c.benchmark_group("flux_depth_ratio");

    let g = 9.81;
    let normal = (1.0, 0.0);

    // Test different depth ratios (h_right / h_left)
    for ratio in [0.5, 0.9, 1.0, 1.1, 2.0] {
        let h_l = 10.0;
        let h_r = h_l * ratio;
        let left = SWEState2D::new(h_l, h_l * 0.5, 0.0);
        let right = SWEState2D::new(h_r, h_r * 0.3, 0.0);

        group.bench_with_input(
            BenchmarkId::new("roe", format!("ratio_{:.1}", ratio)),
            &ratio,
            |b, _| {
                b.iter(|| {
                    roe_flux_swe_2d(
                        black_box(&left),
                        black_box(&right),
                        black_box(normal),
                        black_box(g),
                        black_box(H_MIN),
                    )
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("hll", format!("ratio_{:.1}", ratio)),
            &ratio,
            |b, _| {
                b.iter(|| {
                    hll_flux_swe_2d(
                        black_box(&left),
                        black_box(&right),
                        black_box(normal),
                        black_box(g),
                        black_box(H_MIN),
                    )
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("rusanov", format!("ratio_{:.1}", ratio)),
            &ratio,
            |b, _| {
                b.iter(|| {
                    rusanov_flux_swe_2d(
                        black_box(&left),
                        black_box(&right),
                        black_box(normal),
                        black_box(g),
                        black_box(H_MIN),
                    )
                });
            },
        );
    }

    group.finish();
}

/// Benchmark flux with dry states (wetting/drying scenarios).
fn bench_flux_dry_states(c: &mut Criterion) {
    let mut group = c.benchmark_group("flux_dry_states");

    let g = 9.81;
    let normal = (1.0, 0.0);

    // Wet-wet case
    let wet_wet_l = SWEState2D::new(10.0, 5.0, 0.0);
    let wet_wet_r = SWEState2D::new(10.0, 4.0, 0.0);

    // Wet-dry case (shallow right)
    let wet_dry_l = SWEState2D::new(10.0, 5.0, 0.0);
    let wet_dry_r = SWEState2D::new(0.001, 0.0, 0.0);

    // Dry-wet case (shallow left)
    let dry_wet_l = SWEState2D::new(0.001, 0.0, 0.0);
    let dry_wet_r = SWEState2D::new(10.0, 4.0, 0.0);

    group.bench_function("roe_wet_wet", |b| {
        b.iter(|| {
            roe_flux_swe_2d(
                black_box(&wet_wet_l),
                black_box(&wet_wet_r),
                black_box(normal),
                black_box(g),
                black_box(H_MIN),
            )
        });
    });

    group.bench_function("roe_wet_dry", |b| {
        b.iter(|| {
            roe_flux_swe_2d(
                black_box(&wet_dry_l),
                black_box(&wet_dry_r),
                black_box(normal),
                black_box(g),
                black_box(H_MIN),
            )
        });
    });

    group.bench_function("roe_dry_wet", |b| {
        b.iter(|| {
            roe_flux_swe_2d(
                black_box(&dry_wet_l),
                black_box(&dry_wet_r),
                black_box(normal),
                black_box(g),
                black_box(H_MIN),
            )
        });
    });

    group.bench_function("hll_wet_wet", |b| {
        b.iter(|| {
            hll_flux_swe_2d(
                black_box(&wet_wet_l),
                black_box(&wet_wet_r),
                black_box(normal),
                black_box(g),
                black_box(H_MIN),
            )
        });
    });

    group.bench_function("hll_wet_dry", |b| {
        b.iter(|| {
            hll_flux_swe_2d(
                black_box(&wet_dry_l),
                black_box(&wet_dry_r),
                black_box(normal),
                black_box(g),
                black_box(H_MIN),
            )
        });
    });

    group.bench_function("rusanov_wet_wet", |b| {
        b.iter(|| {
            rusanov_flux_swe_2d(
                black_box(&wet_wet_l),
                black_box(&wet_wet_r),
                black_box(normal),
                black_box(g),
                black_box(H_MIN),
            )
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_flux_functions,
    bench_flux_depth_ratio,
    bench_flux_dry_states
);
criterion_main!(benches);

//! The 3D tracer limiter's gate (P2.7 Phase 4.3b, `notes/gpu.md`): dgcom's
//! `apply_tracer_limiters_3d` on the CPU against [`Limiter3D`] on the GPU,
//! value for value, in three configurations:
//!
//! - the case runner's: the horizontal Kuzmin limiter about the reference
//!   stratification, then the default bounds, conserving;
//! - Kuzmin relaxed (1.5) without a reference, then bounds that cut through
//!   the fixtures' stratification, the out-of-bounds layer means clamped;
//! - those bounds alone, conserving;
//!
//! on the walled beach (thin columns; a cubic reference), the open channel
//! (jittered and stretched; a reference linear between uneven samples) and
//! the doubly periodic wet fixture (vertex patches across the seams; a
//! reference linear between even samples), their tracers as a stage leaves
//! them (every value moved by up to 1 %). Mutants it must catch, and timed
//! at Chiloé's size (96,000 P2 elements × 20 levels). (`stage3d` has the
//! limiter in a stage's values.)

use std::sync::Arc;
use std::time::Instant;

use dgcom::solver::state::Solution3D;
use dgcom::solver::{
    Blocks, TracerAveragePolicy3D, TracerBounds, TracerLimiter3DConfig, TracerLimiter3DStats,
    TracerLimiterType3D, apply_tracer_limiters_3d,
};

use crate::gate::{self, Bar};
use crate::hooks::State3D;
use crate::limiter::{Limiter3D, VertexPatches};
use crate::listed::On;
use crate::stage3d::{Snapshots, report, report_mutant};
use crate::swe2d::{Gpu, NN, ORDER};
use crate::three_d::Statics3D;
use crate::transport3d::{Fixture, LEVELS, Shape};

/// The case runner's limiter (`case::run::three_d`): Kuzmin about the
/// fixture's reference stratification, then the default bounds.
pub(crate) fn case_runner_config(fx: &Fixture) -> TracerLimiter3DConfig {
    let config = TracerLimiter3DConfig {
        limiter_type: TracerLimiterType3D::HorizontalKuzmin { relaxation: 1.0 },
        ..TracerLimiter3DConfig::default()
    };
    match &fx.profile {
        Some(profile) => config.with_reference_profile(profile.clone()),
        None => config,
    }
}

/// Bounds that cut through the fixtures' stratification (T 5–11, S
/// 31.5–34.5): layers above or below them, and layers across them.
const NARROW: TracerBounds = TracerBounds {
    t_min: 7.0,
    t_max: 10.0,
    s_min: 32.6,
    s_max: 33.8,
};

/// `fx`'s state with its tracers as a stage leaves them: every value moved
/// by up to 1 % (T) and 0.4 % (S).
fn stage_state(fx: &Fixture) -> Solution3D {
    let mut state = fx.state.clone();
    for (t, (temp, salt)) in state.temp.iter_mut().zip(state.salt.iter_mut()).enumerate() {
        *temp *= 1.0 + 0.01 * (0.731 * t as f64).sin();
        *salt *= 1.0 + 0.004 * (0.377 * t as f64 + 1.0).sin();
    }
    state
}

/// The CPU's limiting of `state` with `config` (and its statistics).
fn on_cpu(
    fx: &Fixture,
    state: &Solution3D,
    config: &TracerLimiter3DConfig,
) -> (Snapshots, TracerLimiter3DStats) {
    let mut limited = state.clone();
    let stats = apply_tracer_limiters_3d(
        &mut limited,
        &fx.mesh,
        &fx.ops,
        &fx.geom,
        &fx.bathymetry,
        &fx.sigma,
        config,
        fx.min_column_depth,
        Blocks::All(fx.n_elements()),
    );
    (snapshots(&limited), stats)
}

fn snapshots(s: &Solution3D) -> Snapshots {
    vec![
        ("limited T".into(), s.temp.clone()),
        ("limited S".into(), s.salt.clone()),
    ]
}

/// The device's limiting of `state` with `config` over `patches`, columns
/// shallower than `thin_depth` left alone.
fn on_device(
    gpu: &Arc<Gpu>,
    statics: &Statics3D,
    patches: &VertexPatches,
    config: &TracerLimiter3DConfig,
    thin_depth: f64,
    state: &Solution3D,
) -> Result<Snapshots, Box<dyn std::error::Error>> {
    let mut limiter = Limiter3D::new(gpu, statics, patches, config, thin_depth)?;
    let mut values = State3D::upload(gpu, state)?;
    limiter.apply(statics, &mut values, On::ALL)?;
    let mut host = state.clone();
    values.download_into(gpu, &mut host)?;
    Ok(snapshots(&host))
}

/// `patches` with `edit` applied to each vertex's elements (the union kept).
fn edited_vertex_patches(
    patches: &VertexPatches,
    edit: impl Fn(&[u32]) -> Vec<u32>,
) -> VertexPatches {
    let mut offsets = vec![0];
    let mut elements = Vec::new();
    for w in patches.vertex_offsets.windows(2) {
        elements.extend(edit(&patches.vertex_elements[w[0] as usize..w[1] as usize]));
        offsets.push(elements.len() as u32);
    }
    VertexPatches {
        vertex_offsets: offsets,
        vertex_elements: elements,
        ..patches.clone()
    }
}

/// One fixture in every configuration, and (if `mutants`) the mutants.
/// Whether all are the same to the bit (and every mutant caught).
fn gate(
    gpu: &Arc<Gpu>,
    label: &str,
    shape: Shape,
    mutants: bool,
) -> Result<bool, Box<dyn std::error::Error>> {
    let fx = Fixture::new(shape);
    let ne = fx.n_elements();
    let state = stage_state(&fx);
    let statics = fx.statics(gpu)?;
    let patches = VertexPatches::of(&fx.mesh);
    let n_thin = (0..ne * NN)
        .filter(|&idx| state.eta.data[idx] - fx.bathymetry.data[idx] < fx.min_column_depth)
        .count();
    let reference = match &fx.profile {
        None => "no reference",
        Some(p) if p.slopes().is_some() => "a cubic reference",
        Some(p) if p.view().inverse_spacing.is_some() => "a reference linear between even samples",
        Some(_) => "a reference linear between uneven samples",
    };
    println!(
        "{label}: {ne} P{ORDER} elements × {LEVELS} levels, {n_thin} thin columns, {reference}"
    );
    let case_runner = case_runner_config(&fx);
    let configs = [
        ("the case runner's", case_runner.clone()),
        (
            "Kuzmin 1.5 without a reference, narrow bounds enforced",
            TracerLimiter3DConfig {
                limiter_type: TracerLimiterType3D::HorizontalKuzmin { relaxation: 1.5 },
                bounds: NARROW,
                average_policy: TracerAveragePolicy3D::EnforceBounds,
                ..TracerLimiter3DConfig::default()
            },
        ),
        (
            "narrow bounds alone, conserving",
            TracerLimiter3DConfig::bounds(NARROW),
        ),
    ];
    let mut same = true;
    let mut cpus = Vec::new();
    for (name, config) in &configs {
        let (cpu, stats) = on_cpu(&fx, &state, config);
        println!(
            "  {name}: {} T and {} S element layers limited ({} and {} means out of bounds)",
            stats.limited_temperature_cells,
            stats.limited_salinity_cells,
            stats.temperature_average_violations,
            stats.salinity_average_violations
        );
        let device = on_device(gpu, &statics, &patches, config, fx.min_column_depth, &state)?;
        same &= report(&cpu, &device);
        cpus.push(cpu);
    }
    if mutants {
        let run = |config: &TracerLimiter3DConfig, patches: &VertexPatches, thin: f64| {
            on_device(gpu, &statics, patches, config, thin, &state)
        };
        let thin = fx.min_column_depth;
        let relaxed = TracerLimiter3DConfig {
            limiter_type: TracerLimiterType3D::HorizontalKuzmin { relaxation: 1.2 },
            ..case_runner.clone()
        };
        let unreferenced = TracerLimiter3DConfig {
            reference: None,
            ..case_runner.clone()
        };
        let lone = VertexPatches {
            union_offsets: (0..=ne as u32).collect(),
            union_elements: (0..ne as u32).collect(),
            ..patches.clone()
        };
        let short = edited_vertex_patches(&patches, |elements| {
            elements[..elements.len().saturating_sub(1).max(1)].to_vec()
        });
        let mut wide = NARROW;
        wide.t_max += 0.2;
        let mut runs = vec![
            (
                "Kuzmin relaxed 1.2",
                &cpus[0],
                run(&relaxed, &patches, thin)?,
            ),
            (
                "no reference",
                &cpus[0],
                run(&unreferenced, &patches, thin)?,
            ),
            (
                "the union of the vertex patches the element alone",
                &cpus[0],
                run(&case_runner, &lone, thin)?,
            ),
            (
                "each vertex patch without its last element",
                &cpus[0],
                run(&case_runner, &short, thin)?,
            ),
            (
                "the bounds' T max 0.2 higher",
                &cpus[2],
                run(&TracerLimiter3DConfig::bounds(wide), &patches, thin)?,
            ),
            (
                "the out-of-bounds means clamped",
                &cpus[2],
                run(
                    &TracerLimiter3DConfig::bounds(NARROW)
                        .with_average_policy(TracerAveragePolicy3D::EnforceBounds),
                    &patches,
                    thin,
                )?,
            ),
        ];
        if n_thin > 0 {
            runs.push((
                "the thin columns limited",
                &cpus[0],
                run(&case_runner, &patches, 0.0)?,
            ));
        }
        for (what, cpu, device) in runs {
            same &= report_mutant(what, cpu, &device);
        }
    }
    println!("  {}", gate::verdict(label, same));
    Ok(same)
}

/// Time the case runner's limiter on the CPU and the GPU at `shape`'s size,
/// per element and level, on the fixture's tracers as a stage leaves them
/// and on its own (calmer: a few per cent of the element layers limited),
/// from the same tracers every time.
fn time(
    gpu: &Arc<Gpu>,
    label: &str,
    shape: Shape,
    iterations: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let fx = Fixture::new(shape);
    let stage = stage_state(&fx);
    for (which, state) in [("as a stage leaves them", &stage), ("its own", &fx.state)] {
        time_on(
            gpu,
            &fx,
            &format!("{label}, its tracers {which}"),
            state,
            iterations,
        )?;
    }
    Ok(())
}

fn time_on(
    gpu: &Arc<Gpu>,
    fx: &Fixture,
    label: &str,
    state: &Solution3D,
    iterations: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let ne = fx.n_elements();
    let points = (ne * LEVELS) as f64;
    let config = case_runner_config(fx);
    let mut cpu_state = state.clone();
    let mut cpu_time = 0.0;
    let mut stats = TracerLimiter3DStats::default();
    for it in 0..iterations + 1 {
        cpu_state.temp.copy_from_slice(&state.temp);
        cpu_state.salt.copy_from_slice(&state.salt);
        let t0 = Instant::now();
        stats = apply_tracer_limiters_3d(
            &mut cpu_state,
            &fx.mesh,
            &fx.ops,
            &fx.geom,
            &fx.bathymetry,
            &fx.sigma,
            &config,
            fx.min_column_depth,
            Blocks::All(ne),
        );
        if it >= 1 {
            cpu_time += t0.elapsed().as_secs_f64();
        }
    }
    println!(
        "timing, {label}: {ne} P{ORDER} elements × {LEVELS} levels ({:.1} M points), the case \
         runner's limiter: {} T and {} S of {} element layers limited, {iterations} stages",
        (ne * NN * LEVELS) as f64 / 1e6,
        stats.limited_temperature_cells,
        stats.limited_salinity_cells,
        ne * LEVELS
    );
    let statics = fx.statics(gpu)?;
    let mut limiter = Limiter3D::new(
        gpu,
        &statics,
        &VertexPatches::of(&fx.mesh),
        &config,
        fx.min_column_depth,
    )?;
    let start = State3D::upload(gpu, state)?;
    let mut values = State3D::upload(gpu, state)?;
    let gpu_iterations = 5 * iterations;
    let mut gpu_time = 0.0;
    for it in 0..gpu_iterations + 2 {
        values.copy_from(gpu, &start)?;
        gpu.stream.synchronize()?;
        let t0 = Instant::now();
        limiter.apply(&statics, &mut values, On::ALL)?;
        gpu.stream.synchronize()?;
        if it >= 2 {
            gpu_time += t0.elapsed().as_secs_f64();
        }
    }
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let (c, g) = (
        cpu_time * 1e9 / (iterations as f64 * points),
        gpu_time * 1e9 / (gpu_iterations as f64 * points),
    );
    println!(
        "  the limiter, per element and level: CPU ({threads} threads) {c:.2} ns | GPU {g:.2} ns \
         ({:.1}×); per stage {:.1} ms | {:.2} ms",
        c / g,
        c * points / 1e6,
        g * points / 1e6
    );
    Ok(())
}

pub fn run(gpu: &Arc<Gpu>) -> Result<(), Box<dyn std::error::Error>> {
    println!("== The 3D tracer limiter: Kuzmin about the reference, the bounds (Phase 4.3b) ==");
    gate::start("limiter3d", Bar::MIRRORED);
    let cases = [
        ("beach", Shape::BEACH, true),
        ("channel, jittered, stretched", Shape::CHANNEL, true),
        (
            "wet, doubly periodic",
            Shape {
                reference: true,
                ..Shape::WET
            },
            false,
        ),
    ];
    let mut failed = Vec::new();
    for (label, shape, mutants) in cases {
        if !gate(gpu, label, shape, mutants)? {
            failed.push(label);
        }
    }
    time(
        gpu,
        "beach",
        Shape {
            nx: 320,
            ny: 300,
            jitter: 0.15,
            stretched: true,
            ..Shape::BEACH
        },
        10,
    )?;
    if !failed.is_empty() {
        println!("(failed: {})", failed.join("; "));
    }
    Ok(())
}

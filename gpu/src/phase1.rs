//! Phase 1's `Simulation` gate: dgcom's own `Simulation` and SSP-RK(4,3)
//! stepping the GPU state ([`DeviceState`]) with the GPU physics
//! ([`GpuSWE2D`]), against the same run on the CPU (`SWEPhysics2D` built as
//! the case runner builds it), and the time per step of both: the fully wet
//! split form on periodic beds, and the wet/dry stage (`WetDry`, the
//! positivity-bounded step, the wet/dry correction, Manning friction and the
//! thin-layer relaxation) on a walled drying beach.

use std::sync::Arc;
use std::time::Instant;

use dgcom::simulation::Simulation;
use dgcom::time::SSPRK43;

use crate::gate::{self, Bar};
use crate::phase0::{Case, agreement, cpu_kernel, holds};
use crate::swe2d::{DeviceState, Gpu, NN};

/// The CFL number of both runs.
const CFL: f64 = 0.5;

/// Runs `case` to `t_end` on both and reports the steps, the agreement of the
/// final states and the time per step.
fn run_case(
    label: &str,
    case: &Case,
    gpu: &Arc<Gpu>,
    t_end: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let n_el = case.n_elements();
    if case.subcell_elements() > 0 {
        println!(
            "  {label}: {} of {n_el} elements on the subcells at the start",
            case.subcell_elements()
        );
    }

    let cpu_sim = Simulation::new(case.cpu_physics(), SSPRK43).with_cfl(CFL);
    let mut cpu = case.q.clone();
    let start = Instant::now();
    let cpu_result = cpu_sim.run(&mut cpu, 0.0, t_end);
    let cpu_seconds = start.elapsed().as_secs_f64();
    assert!(cpu_result.success, "CPU run: {cpu_result:?}");

    let twin = case.cpu_physics();
    let gpu_sim = Simulation::new(case.gpu_physics(gpu, &twin), SSPRK43).with_cfl(CFL);
    let mut state = DeviceState::upload(gpu, &case.q);
    gpu.stream.synchronize()?;
    let start = Instant::now();
    let gpu_result = gpu_sim.run(&mut state, 0.0, t_end);
    gpu.stream.synchronize()?;
    let gpu_seconds = start.elapsed().as_secs_f64();
    assert!(gpu_result.success, "GPU run: {gpu_result:?}");

    let (identical, worst_ulps, worst_rel) = agreement(&cpu, &state.download());
    println!(
        "{label}: {n_el} P2 elements ({} affine), {} steps on the CPU and {} on the GPU to \
         t = {t_end} s; after them {identical} of {} nodes bit-identical to the {} CPU kernel, \
         largest difference {worst_ulps} ulp, {worst_rel:.1e} of the field's largest: {}",
        case.affine_elements(),
        cpu_result.n_steps,
        gpu_result.n_steps,
        n_el * NN,
        cpu_kernel(),
        gate::verdict(
            label,
            cpu_result.n_steps == gpu_result.n_steps && holds(identical, n_el * NN, worst_rel)
        )
    );
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let per_step = |seconds: f64, steps: usize| seconds * 1e3 / steps as f64;
    let per_element_stage = |seconds: f64, steps: usize| seconds * 1e9 / (steps * 4 * n_el) as f64;
    println!(
        "  per step: CPU {:.2} ms ({threads} threads), GPU {:.2} ms ({:.2}x); per element \
         and stage: CPU {:.1} ns, GPU {:.1} ns",
        per_step(cpu_seconds, cpu_result.n_steps),
        per_step(gpu_seconds, gpu_result.n_steps),
        cpu_seconds / gpu_seconds * gpu_result.n_steps as f64 / cpu_result.n_steps as f64,
        per_element_stage(cpu_seconds, cpu_result.n_steps),
        per_element_stage(gpu_seconds, gpu_result.n_steps),
    );
    Ok(())
}

/// The wet/dry beach of 96,000 elements alone (for a profiler).
pub fn run_beach(gpu: &Arc<Gpu>) -> Result<(), Box<dyn std::error::Error>> {
    run_case(
        "beach, jittered, timing",
        &Case::beach(320, 300, 0.2, false),
        gpu,
        30.0,
    )
}

pub fn run(gpu: &Arc<Gpu>) -> Result<(), Box<dyn std::error::Error>> {
    println!("== Phase 1: Simulation with SSP-RK(4,3) on the GPU ==");
    // Whole runs: contraction moves these by 1e-14 to 4e-11
    gate::start("phase1", Bar::within(1e-8));
    run_case("parallelograms", &Case::new(48, 40, 0.0), gpu, 30.0)?;
    run_case("jittered", &Case::new(48, 40, 0.15), gpu, 30.0)?;
    run_case("jittered, timing", &Case::new(320, 300, 0.15), gpu, 30.0)?;
    println!("== Phase 1: Simulation of a wet/dry beach on the GPU ==");
    run_case(
        "beach, parallelograms",
        &Case::beach(36, 20, 0.0, false),
        gpu,
        120.0,
    )?;
    run_case(
        "beach, jittered",
        &Case::beach(36, 20, 0.2, false),
        gpu,
        120.0,
    )?;
    run_case(
        "beach with films, jittered",
        &Case::beach(36, 20, 0.2, true),
        gpu,
        30.0,
    )?;
    run_case(
        "beach, jittered, timing",
        &Case::beach(320, 300, 0.2, false),
        gpu,
        30.0,
    )?;
    Ok(())
}

//! The 3D step's gate (P2.7 Phase 4.5b, `notes/gpu.md`): whole steps of
//! dgcom's own `ModeSplitIntegrator` with the device's model ([`Model3D`])
//! against the same steps with a `Hydrostatic3D` on the CPU, every field
//! after every step.
//!
//! The model has the case runner's terms: the wet/dry 2D module of a 3D run
//! (Coriolis on a β-plane, walls), k-ε, the log-layer bottom drag, a farm
//! of net cages, uniform surface stresses and a wind stress field, rivers,
//! the reference stratification with its pressure gradient, its advection
//! and the Kuzmin limiter about it, implicit vertical advection, the
//! shear's Smagorinsky viscosity. The integrator is the case runner's: the
//! pass on two levels of local time stepping, SSP-RK(4,3) in both. The
//! walled beach (films, dry and thin columns, merged cells) starts with no
//! turbulence, as a case does, so the first step allocates it.
//!
//! The mutants give the device's side another input than the CPU's: a term
//! left out of its physics, or another integrator.

use std::error::Error;
use std::sync::Arc;
use std::time::Instant;

use dgcom::Reflective2D;
use dgcom::physics::vertical_mixing::Forcing;
use dgcom::physics::{
    BottomDrag3D, GlsMixing, Hydrostatic3D, ImplicitVerticalAdvection, LinearEOS,
};
use dgcom::solver::state::Solution3D;
use dgcom::solver::{TracerLimiter3DConfig, TracerLimiterType3D};
use dgcom::time::{ModeSplitIntegrator, SspScheme};

use crate::forcing3d::{cages, coriolis, wind};
use crate::hooks::State3D;
use crate::model3d::{Model3D, Work3D};
use crate::pass3d::{beach_physics, beach_rivers};
use crate::stage3d::{Snapshots, report, report_mutant, start_state};
use crate::swe2d::Gpu;
use crate::transport3d::{Fixture, G, RHO0, Shape};

type Physics = Hydrostatic3D<LinearEOS, GlsMixing, Reflective2D>;

/// The steps compared.
const STEPS: usize = 4;
/// The case runner's 3D CFL number with SSP-RK(4,3), and its cap (s).
const CFL: f64 = 4.0;
const DT_MAX: f64 = 60.0;

/// The terms of a model (a mutant leaves one out).
#[derive(Clone, Copy)]
struct Terms {
    wind: bool,
    rivers: bool,
    cages: bool,
    implicit: bool,
    limiter: bool,
    reference: bool,
    smagorinsky: f64,
    roughness: f64,
}

impl Terms {
    const CASE_RUNNER: Self = Self {
        wind: true,
        rivers: true,
        cages: true,
        implicit: true,
        limiter: true,
        reference: true,
        smagorinsky: 0.1,
        roughness: 0.003,
    };
}

/// A `Hydrostatic3D` of `fx` with `terms` (its boundaries the fixture's).
fn physics(fx: &Fixture, terms: Terms) -> Physics {
    let mut limiter = TracerLimiter3DConfig {
        limiter_type: TracerLimiterType3D::HorizontalKuzmin { relaxation: 1.0 },
        ..TracerLimiter3DConfig::default()
    };
    if let Some(profile) = &fx.profile {
        limiter = limiter.with_reference_profile(profile.clone());
    }
    if !terms.limiter {
        limiter = TracerLimiter3DConfig::none();
    }
    let mut physics = Hydrostatic3D::new(
        Arc::new(fx.mesh.clone()),
        Arc::new(fx.ops.clone()),
        Arc::new(fx.geom.clone()),
        Arc::new(fx.sigma.clone()),
        Arc::new(fx.bathymetry.clone()),
        Arc::new(coriolis()),
        LinearEOS::default(),
        GlsMixing::k_epsilon(),
        beach_physics(fx),
        Forcing {
            surface_stress: [0.03, -0.01],
            bottom_stress: [0.0, 0.0],
            surface_heat_flux: 0.0,
        },
        G,
        RHO0,
    )
    .with_bottom_drag(BottomDrag3D::log_layer(terms.roughness))
    .with_min_column_depth(fx.min_column_depth)
    .with_smagorinsky_viscosity(terms.smagorinsky)
    .with_tracer_limiter(limiter);
    if terms.cages {
        physics = physics.with_cage_drag(cages(fx));
    }
    if terms.wind {
        physics = physics.with_surface_stress(wind(fx));
    }
    if terms.rivers {
        physics = physics.with_rivers(beach_rivers(fx));
    }
    if terms.implicit {
        physics = physics.with_implicit_vertical_advection(ImplicitVerticalAdvection::default());
    }
    physics.boundaries = fx.boundaries.clone();
    match &fx.profile {
        Some(profile) if terms.reference => physics.with_reference_profile(profile.clone()),
        _ => physics.without_reference_profile(),
    }
}

/// The integrator of a run (a mutant changes it).
#[derive(Clone, Copy)]
struct Run {
    barotropic_levels: usize,
    scheme: SspScheme,
    min_substeps: usize,
}

impl Run {
    /// The case runner's: the pass on two levels, SSP-RK(4,3) in both.
    const CASE_RUNNER: Self = Self {
        barotropic_levels: 2,
        scheme: SspScheme::Rk43,
        min_substeps: dgcom::time::MIN_BAROTROPIC_SUBSTEPS,
    };

    fn integrator(self) -> ModeSplitIntegrator {
        ModeSplitIntegrator::new()
            .with_barotropic_cfl(2.0)
            .with_barotropic_scheme(SspScheme::Rk43)
            .with_baroclinic_scheme(self.scheme)
            .with_barotropic_levels(self.barotropic_levels)
            .with_min_substeps(self.min_substeps)
    }
}

/// `state`'s fields after step `n`, with the splitter's counts.
fn take(out: &mut Snapshots, n: usize, state: &Solution3D, merged: [usize; 2], substeps: usize) {
    let at = |field: &str| format!("step {n} {field}");
    out.push((
        at("counts"),
        vec![merged[0] as f64, merged[1] as f64, substeps as f64],
    ));
    for (name, field) in [
        ("eta", &state.eta.data),
        ("ubar", &state.ubar.data),
        ("vbar", &state.vbar.data),
        ("u", &state.u),
        ("v", &state.v),
        ("T", &state.temp),
        ("S", &state.salt),
        ("k", &state.tke),
        ("psi", &state.gls),
        ("rho", &state.rho),
    ] {
        out.push((at(name), field.clone()));
    }
}

/// `STEPS` steps of `dt` from `start` on the CPU: the snapshots and the
/// time (s).
fn on_cpu(physics: &Physics, run: Run, start: &Solution3D, dt: f64) -> (Snapshots, f64) {
    let mut state = start.clone();
    let mut integrator = run.integrator();
    let mut snapshots = Snapshots::new();
    let clock = Instant::now();
    for n in 0..STEPS {
        physics.update_density(&mut state);
        integrator.step(&mut state, physics, dt, n as f64 * dt);
        take(
            &mut snapshots,
            n,
            &state,
            integrator.merged_cells(),
            integrator.last_substeps(),
        );
    }
    (snapshots, clock.elapsed().as_secs_f64())
}

/// The same steps on the device: the snapshots and the time (s, the
/// snapshots' downloads not counted).
fn on_device(
    gpu: &Arc<Gpu>,
    physics: &Physics,
    run: Run,
    start: &Solution3D,
    dt: f64,
) -> Result<(Snapshots, f64), Box<dyn Error>> {
    let model = Model3D::new(gpu, physics)?;
    let mut state = State3D::upload(gpu, start)?;
    let mut integrator = run.integrator().on::<Work3D>();
    let mut snapshots = Snapshots::new();
    let mut host = start.clone();
    let mut time = 0.0;
    for n in 0..STEPS {
        let clock = Instant::now();
        model.update_density(&mut state);
        integrator.step(&mut state, &model, dt, n as f64 * dt);
        gpu.stream.synchronize()?;
        time += clock.elapsed().as_secs_f64();
        state.download_into(gpu, &mut host)?;
        let merged = integrator.merged_cells();
        take(&mut snapshots, n, &host, merged, integrator.last_substeps());
    }
    Ok((snapshots, time))
}

/// One fixture on both sides, and if `mutants` the mutants; whether every
/// field of every step is the same to the bit (and every mutant caught).
fn gate(gpu: &Arc<Gpu>, label: &str, shape: Shape, mutants: bool) -> Result<bool, Box<dyn Error>> {
    let fx = Fixture::new(shape);
    let base = Terms::CASE_RUNNER;
    let physics = physics(&fx, base);
    // As a case starts: no turbulence (the closure allocates it at the
    // first step's end), the density of the start's T and S
    let mut start = start_state(&fx);
    start.tke.clear();
    start.gls.clear();
    physics.update_density(&mut start);
    let dt = physics
        .compute_dt_ssp(&start, CFL, SspScheme::Rk43)
        .min(DT_MAX);
    println!(
        "  {label}: {} elements × {} levels, {STEPS} steps of {dt:.2} s",
        fx.n_elements(),
        fx.sigma.n_levels()
    );
    let run = Run::CASE_RUNNER;
    let (cpu, cpu_time) = on_cpu(&physics, run, &start, dt);
    let (device, device_time) = on_device(gpu, &physics, run, &start, dt)?;
    let mut same = report(&cpu, &device);
    let last = &cpu[cpu.len() - 11].1;
    println!(
        "    after {STEPS} steps: {} cells merged and {} kept so far, {} barotropic substeps",
        last[0], last[1], last[2]
    );
    println!(
        "    {:.1} ms a step on the CPU, {:.1} ms on the GPU",
        1e3 * cpu_time / STEPS as f64,
        1e3 * device_time / STEPS as f64
    );
    if !mutants {
        return Ok(same);
    }
    let terms: [(&str, Terms); 8] = [
        (
            "no wind stress field",
            Terms {
                wind: false,
                ..base
            },
        ),
        (
            "no rivers",
            Terms {
                rivers: false,
                ..base
            },
        ),
        (
            "no cages",
            Terms {
                cages: false,
                ..base
            },
        ),
        (
            "explicit vertical advection",
            Terms {
                implicit: false,
                ..base
            },
        ),
        (
            "no tracer limiter",
            Terms {
                limiter: false,
                ..base
            },
        ),
        (
            "no reference",
            Terms {
                reference: false,
                ..base
            },
        ),
        (
            "Smagorinsky 0.12",
            Terms {
                smagorinsky: 0.12,
                ..base
            },
        ),
        (
            "bed roughness 4 mm",
            Terms {
                roughness: 0.004,
                ..base
            },
        ),
    ];
    for (what, terms) in terms {
        let (mutant, _) = on_device(gpu, &self::physics(&fx, terms), run, &start, dt)?;
        same &= report_mutant(what, &cpu, &mutant);
    }
    let runs: [(&str, Run); 3] = [
        (
            "global substeps",
            Run {
                barotropic_levels: 0,
                ..run
            },
        ),
        (
            "SSP-RK3 stages",
            Run {
                scheme: SspScheme::Rk3,
                ..run
            },
        ),
        (
            "at least 40 substeps",
            Run {
                min_substeps: 40,
                ..run
            },
        ),
    ];
    for (what, mutated) in runs {
        let (mutant, _) = on_device(gpu, &physics, mutated, &start, dt)?;
        same &= report_mutant(what, &cpu, &mutant);
    }
    Ok(same)
}

/// Whole steps of the case runner's model on `fx`, after one untimed step
/// on each side: the CPU's 24 threads against the device.
fn time(gpu: &Arc<Gpu>, fx: &Fixture, steps: usize) -> Result<(), Box<dyn Error>> {
    let physics = physics(fx, Terms::CASE_RUNNER);
    let mut start = start_state(fx);
    start.tke.clear();
    start.gls.clear();
    physics.update_density(&mut start);
    let dt = physics
        .compute_dt_ssp(&start, CFL, SspScheme::Rk43)
        .min(DT_MAX);
    let run = Run::CASE_RUNNER;
    let mut cpu = start.clone();
    let mut integrator = run.integrator();
    let mut cpu_time = 0.0;
    for n in 0..=steps {
        let clock = Instant::now();
        physics.update_density(&mut cpu);
        integrator.step(&mut cpu, &physics, dt, n as f64 * dt);
        if n > 0 {
            cpu_time += clock.elapsed().as_secs_f64();
        }
    }
    let substeps = integrator.last_substeps();
    drop(cpu);
    let model = Model3D::new(gpu, &physics)?;
    let mut state = State3D::upload(gpu, &start)?;
    let mut integrator = run.integrator().on::<Work3D>();
    let mut gpu_time = 0.0;
    for n in 0..=steps {
        gpu.stream.synchronize()?;
        let clock = Instant::now();
        model.update_density(&mut state);
        integrator.step(&mut state, &model, dt, n as f64 * dt);
        gpu.stream.synchronize()?;
        if n > 0 {
            gpu_time += clock.elapsed().as_secs_f64();
        }
    }
    let (cpu_ms, gpu_ms) = (1e3 * cpu_time / steps as f64, 1e3 * gpu_time / steps as f64);
    println!(
        "timing, the whole 3D step: {} elements × {} levels, steps of {dt:.1} s with \
         {substeps} barotropic substeps: {cpu_ms:.0} ms a step on the CPU's threads, \
         {gpu_ms:.0} ms on the GPU ({:.1}×)",
        fx.n_elements(),
        fx.sigma.n_levels(),
        cpu_ms / gpu_ms
    );
    Ok(())
}

pub fn run(gpu: &Arc<Gpu>) -> Result<(), Box<dyn Error>> {
    println!("3D steps, whole: ModeSplitIntegrator with the device's model against the CPU's");
    let mut same = gate(gpu, "beach", Shape::BEACH, true)?;
    same &= gate(
        gpu,
        "beach, jittered and stretched",
        Shape {
            jitter: 0.2,
            stretched: true,
            ..Shape::BEACH
        },
        false,
    )?;
    // 48,000 elements by default: the model takes ≈ 136 kB an element
    // (6.5 GB at 48,000), and a card that cannot hold it pages to the
    // host's memory (Chiloé's 96,000 took 53 s a step on the laptop's 8 GB);
    // `DGCOM_STEP3D_ELEMENTS` sets the size (`time.sh`: 96,000)
    let elements: usize = std::env::var("DGCOM_STEP3D_ELEMENTS")
        .ok()
        .and_then(|n| n.parse().ok())
        .unwrap_or(48_000);
    let ny = 300.min(elements / 2).max(1);
    let large = Fixture::new(Shape {
        nx: elements / ny,
        ny,
        jitter: 0.15,
        stretched: true,
        ..Shape::BEACH
    });
    time(gpu, &large, 3)?;
    if !same {
        return Err("the 3D steps differ, or a mutant went uncaught".into());
    }
    Ok(())
}

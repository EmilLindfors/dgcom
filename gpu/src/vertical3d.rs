//! The 3D vertical step's gate (P2.7 Phase 4.4a, `notes/gpu.md`): the drag
//! rates of a step's start and `Hydrostatic3D::vertical_implicit` on the GPU
//! ([`Vertical3D`]) against dgcom on the CPU (`BottomDrag3D::column_rate`,
//! `cage_rates`, `apply_vertical_diffusion`, `thin_columns_to_depth_mean`),
//! value for value: the rates, `u`, `v`, `T`, `S`, the GLS turbulence `k`,
//! `ψ` and the mixing's viscosity and diffusivity, in three configurations:
//!
//! - the case runner's: k-ε, the log-layer drag, net cages, a wind stress
//!   field, heat and fresh-water fluxes with the shortwave absorbed through
//!   the column (Jerlov III), the thin columns' fluxes into their bed;
//! - the generic model with breaking waves, Charnock's roughness and a
//!   roughness field, a constant drag coefficient, fluxes in clear water
//!   (Jerlov I), the thin columns without fluxes;
//! - k-ω with nothing but its own turbulence (no drag, no surface fields);
//!
//! on the walled beach (thin columns and films) and the open channel
//! (jittered, stretched). Mutants it must catch, and timed at Chiloé's size
//! (96,000 P2 elements × 20 levels).

use std::sync::Arc;
use std::time::Instant;

use cuda_core::DeviceBuffer;
use dgcom::physics::cage_drag::{cage_rates, for_each_caged_node};
use dgcom::physics::vertical_mixing::Forcing;
use dgcom::physics::{
    BottomDrag3D, GlsMixing, JerlovWater, SurfaceFields, SurfaceTracerFluxes,
    apply_vertical_diffusion, thin_columns_to_depth_mean,
};
use dgcom::solver::Blocks;
use dgcom::solver::state::Solution3D;
use dgcom::source::{CageDrag2D, NetCage};
use dgcom::time::StepDrag;

use crate::gate::{self, Bar};
use crate::hooks::State3D;
use crate::listed::On;
use crate::stage3d::{Snapshots, report, report_mutant, start_state};
use crate::swe2d::{Gpu, NN, ORDER};
use crate::three_d::Statics3D;
use crate::transport3d::{Fixture, G, LEVELS, RHO0, Shape};
use crate::vertical::{SurfaceDevice, Vertical3D, VerticalSetup};

/// The baroclinic step the vertical terms take (s).
const STEP: f64 = 120.0;

/// One configuration of the vertical step.
#[derive(Clone)]
struct Config {
    mixing: GlsMixing,
    forcing: Forcing,
    bottom_drag: Option<BottomDrag3D>,
    cages: bool,
    /// The wind stress field, the roughness field, the tracer fluxes
    stress: bool,
    roughness: bool,
    fluxes: Option<JerlovWater>,
    thin_bed: Option<f64>,
}

impl Config {
    /// The case runner's (`case::run::three_d`), with every forcing it takes.
    fn case_runner() -> Self {
        Self {
            mixing: GlsMixing::k_epsilon(),
            forcing: Forcing {
                surface_stress: [0.0, 0.0],
                bottom_stress: [0.0, 0.0],
                surface_heat_flux: 0.0,
            },
            bottom_drag: Some(BottomDrag3D::log_layer(0.01)),
            cages: true,
            stress: true,
            roughness: false,
            fluxes: Some(JerlovWater::III),
            thin_bed: Some(0.1),
        }
    }
}

/// The surface fields of a step on `fx`'s columns: a wind stress of
/// 0.05–0.15 N/m² turning over the domain, a roughness of 0.05–0.5 m,
/// non-solar heat −150…+50 W/m², shortwave 0–400 W/m² (none in a fifth of
/// the columns), evaporation minus precipitation ±2e-5 kg m⁻² s⁻¹.
struct Surface {
    stress: [Vec<f64>; 2],
    roughness: Vec<f64>,
    fluxes: [Vec<f64>; 3],
}

impl Surface {
    fn new(fx: &Fixture) -> Self {
        let n = fx.n_elements() * NN;
        let at = |f: &dyn Fn(f64) -> f64| (0..n).map(|i| f(i as f64)).collect::<Vec<f64>>();
        Self {
            stress: [
                at(&|i| 0.1 * (0.0013 * i).cos() + 0.05 * (0.37 * i).sin()),
                at(&|i| 0.1 * (0.0013 * i).sin() - 0.03 * (0.29 * i).cos()),
            ],
            roughness: at(&|i| 0.05 + 0.45 * (0.5 + 0.5 * (0.011 * i).sin())),
            fluxes: [
                at(&|i| -50.0 + 100.0 * (0.017 * i).sin()),
                at(&|i| {
                    let s = 200.0 + 200.0 * (0.0071 * i).sin();
                    if (i as usize).is_multiple_of(5) {
                        0.0
                    } else {
                        s
                    }
                }),
                at(&|i| 2e-5 * (0.023 * i).cos()),
            ],
        }
    }
}

/// A farm of four net cages of 50 m radius, 3 to 20 m deep (two of them
/// overlapping), where `fx` is deep.
fn cages(fx: &Fixture) -> CageDrag2D {
    let cages = [
        NetCage::circular([300.0, 500.0], 50.0, 3.0, 0.25),
        NetCage::circular([420.0, 500.0], 50.0, 8.0, 0.2),
        NetCage::circular([300.0, 650.0], 50.0, 12.0, 0.3),
        NetCage::circular([340.0, 680.0], 50.0, 20.0, 0.25),
    ];
    CageDrag2D::new(&fx.mesh, &fx.ops, &cages)
}

/// The CPU's step of `start` in `config`: the rates, then the vertical step.
fn on_cpu(
    fx: &Fixture,
    start: &Solution3D,
    config: &Config,
    surface: &Surface,
    cages: &CageDrag2D,
) -> Snapshots {
    let mut state = start.clone();
    let (bottom, layers) = cpu_step(fx, &mut state, config, surface, cages);
    let mut out = Snapshots::new();
    if let Some(bottom) = bottom {
        out.push(("rates bottom".into(), bottom));
    }
    if let Some(layers) = layers {
        out.push(("rates layers".into(), layers));
    }
    take(&mut out, &state);
    out
}

/// The step's drag rates of `state`, then its vertical step in `config`
/// (dgcom's functions as `ModeSplitIntegrator::step` and `Hydrostatic3D`
/// call them); the rates.
#[allow(clippy::type_complexity)]
fn cpu_step(
    fx: &Fixture,
    state: &mut Solution3D,
    config: &Config,
    surface: &Surface,
    cages: &CageDrag2D,
) -> (Option<Vec<f64>>, Option<Vec<f64>>) {
    let ne = fx.n_elements();
    let all = Blocks::All(ne);
    let (nl, thin) = (LEVELS, fx.min_column_depth);
    let bed = &fx.bathymetry.data;
    let sigma = &fx.sigma;
    // The step's drag rates, as `bottom_drag_into` and `layer_drag_into`
    let bottom: Option<Vec<f64>> = config.bottom_drag.map(|drag| {
        (0..ne * NN)
            .map(|idx| {
                drag.column_rate(
                    state.eta.data[idx] - bed[idx],
                    thin,
                    sigma.sigma_rho()[0],
                    [state.u[idx * nl], state.v[idx * nl]],
                    [state.ubar.data[idx], state.vbar.data[idx]],
                )
            })
            .collect()
    });
    let layers: Option<Vec<f64>> = config.cages.then(|| {
        let mut rate = vec![0.0; ne * NN * nl];
        for k in 0..ne {
            for_each_caged_node(cages, k, NN, |i, entries| {
                let idx = k * NN + i;
                let column = idx * nl..(idx + 1) * nl;
                cage_rates(
                    entries.len(),
                    |e| (entries[e].coefficient, entries[e].net_depth),
                    [sigma.sigma_w(), sigma.d_sigma()],
                    state.eta.data[idx] - bed[idx],
                    thin,
                    [&state.u[column.clone()], &state.v[column.clone()]],
                    [state.ubar.data[idx], state.vbar.data[idx]],
                    &mut rate[column],
                );
            });
        }
        rate
    });
    // `vertical_implicit`
    apply_vertical_diffusion(
        state,
        sigma,
        &fx.bathymetry,
        STEP,
        &config.mixing,
        &config.forcing,
        SurfaceFields {
            stress: config
                .stress
                .then(|| [&surface.stress[0][..], &surface.stress[1][..]]),
            roughness: config.roughness.then_some(&surface.roughness[..]),
            tracers: config.fluxes.map(|water| SurfaceTracerFluxes {
                nonsolar: &surface.fluxes[0],
                shortwave: &surface.fluxes[1],
                freshwater: &surface.fluxes[2],
                water,
            }),
            thin_bed: config.thin_bed,
        },
        G,
        RHO0,
        thin,
        StepDrag {
            bottom: bottom.as_deref(),
            layers: layers.as_deref(),
        },
        all,
    );
    thin_columns_to_depth_mean(state, bed, thin, all);
    (bottom, layers)
}

fn take(out: &mut Snapshots, s: &Solution3D) {
    for (name, field) in [
        ("u", &s.u),
        ("v", &s.v),
        ("T", &s.temp),
        ("S", &s.salt),
        ("k", &s.tke),
        ("psi", &s.gls),
        ("av", &s.eddy_viscosity),
        ("kt", &s.eddy_diffusivity),
    ] {
        out.push((format!("step {name}"), field.clone()));
    }
}

/// The device's step of `start` in `config`, its uploads `uploaded`.
#[allow(clippy::too_many_arguments)]
fn on_device(
    gpu: &Arc<Gpu>,
    fx: &Fixture,
    statics: &Statics3D,
    start: &Solution3D,
    config: &Config,
    uploaded: &Uploaded,
    cages: &CageDrag2D,
    thin_depth: f64,
) -> Result<Snapshots, Box<dyn std::error::Error>> {
    let mut vertical = Vertical3D::new(
        gpu,
        statics,
        &fx.sigma,
        VerticalSetup {
            mixing: &config.mixing,
            forcing: config.forcing,
            bottom_drag: config.bottom_drag,
            cages: config.cages.then_some(cages),
            water: config.fluxes.unwrap_or_default(),
            thin_bed: config.thin_bed,
            g: G,
            rho0: RHO0,
            min_column_depth: thin_depth,
        },
    )?;
    let mut state = State3D::upload(gpu, start)?;
    vertical.drag_rates(statics, &state)?;
    let surface = SurfaceDevice {
        stress: config
            .stress
            .then(|| [&uploaded.stress[0], &uploaded.stress[1]]),
        roughness: config.roughness.then_some(&uploaded.roughness),
        tracers: config.fluxes.map(|_| {
            [
                &uploaded.fluxes[0],
                &uploaded.fluxes[1],
                &uploaded.fluxes[2],
            ]
        }),
    };
    vertical.step(statics, &mut state, surface, STEP, On::ALL)?;
    let st = &gpu.stream;
    let mut out = Snapshots::new();
    if config.bottom_drag.is_some() {
        out.push(("rates bottom".into(), vertical.bottom_rate.to_host_vec(st)?));
    }
    if config.cages {
        out.push(("rates layers".into(), vertical.layer_rate.to_host_vec(st)?));
    }
    let mut host = start.clone();
    state.download_into(gpu, &mut host)?;
    host.eddy_viscosity = vertical.eddy_viscosity.to_host_vec(st)?;
    host.eddy_diffusivity = vertical.eddy_diffusivity.to_host_vec(st)?;
    take(&mut out, &host);
    Ok(out)
}

/// [`Surface`] on the device.
struct Uploaded {
    stress: [DeviceBuffer<f64>; 2],
    roughness: DeviceBuffer<f64>,
    fluxes: [DeviceBuffer<f64>; 3],
}

impl Uploaded {
    fn new(gpu: &Gpu, s: &Surface) -> Result<Self, Box<dyn std::error::Error>> {
        let up = |v: &Vec<f64>| DeviceBuffer::from_host(&gpu.stream, v);
        Ok(Self {
            stress: [up(&s.stress[0])?, up(&s.stress[1])?],
            roughness: up(&s.roughness)?,
            fluxes: [up(&s.fluxes[0])?, up(&s.fluxes[1])?, up(&s.fluxes[2])?],
        })
    }
}

/// `fx`'s state for a step: its stage state with GLS turbulence, the eddy
/// coefficients at zero (the CPU leaves the thin columns' as they are).
fn step_state(fx: &Fixture) -> Solution3D {
    let mut state = start_state(fx);
    state.eddy_viscosity.fill(0.0);
    state.eddy_diffusivity.fill(0.0);
    state
}

/// One fixture in every configuration, and (if `mutants`) the mutants.
fn gate(
    gpu: &Arc<Gpu>,
    label: &str,
    shape: Shape,
    mutants: bool,
) -> Result<bool, Box<dyn std::error::Error>> {
    let fx = Fixture::new(shape);
    let ne = fx.n_elements();
    let start = step_state(&fx);
    let statics = fx.statics(gpu)?;
    let surface = Surface::new(&fx);
    let uploaded = Uploaded::new(gpu, &surface)?;
    let cages = cages(&fx);
    let n_thin = (0..ne * NN)
        .filter(|&idx| start.eta.data[idx] - fx.bathymetry.data[idx] < fx.min_column_depth)
        .count();
    let caged = cages.nodes().len();
    println!(
        "{label}: {ne} P{ORDER} elements × {LEVELS} levels, {n_thin} thin columns, {caged} cage \
         entries, steps of {STEP} s"
    );
    let case_runner = Config::case_runner();
    let configs = [
        (
            "the case runner's (k-ε, log-layer drag, cages, wind, fluxes)",
            case_runner.clone(),
        ),
        (
            "generic, breaking waves, Charnock, roughness, constant drag, clear water",
            Config {
                mixing: GlsMixing::generic()
                    .with_wave_breaking(GlsMixing::DEFAULT_WAVE_BREAKING)
                    .with_charnock_roughness(GlsMixing::DEFAULT_CHARNOCK),
                forcing: Forcing {
                    surface_stress: [0.02, -0.01],
                    bottom_stress: [0.0, 0.0],
                    surface_heat_flux: -20.0,
                },
                bottom_drag: Some(BottomDrag3D::quadratic(2.5e-3)),
                cages: false,
                stress: true,
                roughness: true,
                fluxes: Some(JerlovWater::I),
                thin_bed: None,
            },
        ),
        (
            "k-ω alone",
            Config {
                mixing: GlsMixing::k_omega(),
                bottom_drag: None,
                cages: false,
                stress: false,
                fluxes: None,
                thin_bed: None,
                ..Config::case_runner()
            },
        ),
    ];
    let mut same = true;
    let mut cpus = Vec::new();
    let thin = fx.min_column_depth;
    for (name, config) in &configs {
        println!("  {name}");
        let cpu = on_cpu(&fx, &start, config, &surface, &cages);
        let device = on_device(gpu, &fx, &statics, &start, config, &uploaded, &cages, thin)?;
        same &= report(&cpu, &device);
        cpus.push(cpu);
    }
    if mutants {
        let run = |config: &Config, thin| {
            on_device(gpu, &fx, &statics, &start, config, &uploaded, &cages, thin)
        };
        let base = &case_runner;
        let runs = [
            (
                "no bottom drag",
                run(
                    &Config {
                        bottom_drag: None,
                        ..base.clone()
                    },
                    thin,
                )?,
            ),
            (
                "no cages",
                run(
                    &Config {
                        cages: false,
                        ..base.clone()
                    },
                    thin,
                )?,
            ),
            (
                "the shortwave absorbed in clear water (Jerlov I)",
                run(
                    &Config {
                        fluxes: Some(JerlovWater::I),
                        ..base.clone()
                    },
                    thin,
                )?,
            ),
            (
                "k-ω, not k-ε",
                run(
                    &Config {
                        mixing: GlsMixing::k_omega(),
                        ..base.clone()
                    },
                    thin,
                )?,
            ),
            (
                "the wind stress field left out",
                run(
                    &Config {
                        stress: false,
                        ..base.clone()
                    },
                    thin,
                )?,
            ),
            (
                "no fluxes into the thin columns",
                run(
                    &Config {
                        thin_bed: None,
                        ..base.clone()
                    },
                    thin,
                )?,
            ),
            ("the thin depth halved", run(base, 0.5 * thin)?),
        ];
        for (what, device) in runs {
            same &= report_mutant(what, &cpus[0], &pad(&cpus[0], device));
        }
    }
    println!(
        "  {}",
        gate::verdict(label, same)
    );
    Ok(same)
}

/// A mutant's snapshots in the gate's order: those it lacks (its rates
/// when it has no drag) as zeros.
fn pad(like: &Snapshots, device: Snapshots) -> Snapshots {
    like.iter()
        .map(|(name, values)| {
            device
                .iter()
                .find(|(n, _)| n == name)
                .cloned()
                .unwrap_or_else(|| (name.clone(), vec![0.0; values.len()]))
        })
        .collect()
}

/// Time the case runner's vertical step (its rates included) on the CPU and
/// the GPU, per element and level.
fn time(
    gpu: &Arc<Gpu>,
    label: &str,
    shape: Shape,
    iterations: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let fx = Fixture::new(shape);
    let ne = fx.n_elements();
    let points = (ne * LEVELS) as f64;
    let start = step_state(&fx);
    let surface = Surface::new(&fx);
    let cages = cages(&fx);
    let config = Config::case_runner();
    let mut cpu_time = 0.0;
    let mut state = start.clone();
    for it in 0..iterations + 1 {
        for (to, from) in [
            (&mut state.u, &start.u),
            (&mut state.v, &start.v),
            (&mut state.temp, &start.temp),
            (&mut state.salt, &start.salt),
            (&mut state.tke, &start.tke),
            (&mut state.gls, &start.gls),
        ] {
            to.copy_from_slice(from);
        }
        let t0 = Instant::now();
        let rates = cpu_step(&fx, &mut state, &config, &surface, &cages);
        if it >= 1 {
            cpu_time += t0.elapsed().as_secs_f64();
        }
        drop(rates);
    }
    drop(state);
    let statics = fx.statics(gpu)?;
    let uploaded = Uploaded::new(gpu, &surface)?;
    let mut vertical = Vertical3D::new(
        gpu,
        &statics,
        &fx.sigma,
        VerticalSetup {
            mixing: &config.mixing,
            forcing: config.forcing,
            bottom_drag: config.bottom_drag,
            cages: Some(&cages),
            water: JerlovWater::III,
            thin_bed: config.thin_bed,
            g: G,
            rho0: RHO0,
            min_column_depth: fx.min_column_depth,
        },
    )?;
    let initial = State3D::upload(gpu, &start)?;
    let mut state = State3D::upload(gpu, &start)?;
    let surface = SurfaceDevice {
        stress: Some([&uploaded.stress[0], &uploaded.stress[1]]),
        roughness: None,
        tracers: Some([
            &uploaded.fluxes[0],
            &uploaded.fluxes[1],
            &uploaded.fluxes[2],
        ]),
    };
    let gpu_iterations = 5 * iterations;
    let mut gpu_time = 0.0;
    for it in 0..gpu_iterations + 2 {
        state.copy_from(gpu, &initial)?;
        gpu.stream.synchronize()?;
        let t0 = Instant::now();
        vertical.drag_rates(&statics, &state)?;
        vertical.step(&statics, &mut state, surface, STEP, On::ALL)?;
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
        "timing, {label}: {ne} P{ORDER} elements × {LEVELS} levels ({:.1} M points), the case \
         runner's vertical step, {iterations} steps",
        (ne * NN * LEVELS) as f64 / 1e6
    );
    println!(
        "  the vertical step, per element and level: CPU ({threads} threads) {c:.2} ns | GPU \
         {g:.2} ns ({:.1}×); per step {:.1} ms | {:.2} ms",
        c / g,
        c * points / 1e6,
        g * points / 1e6
    );
    Ok(())
}

/// `dgcom::math` on the device against the CPU, 1 M pairs over the ranges
/// the column physics takes and beyond (`x` 1e-8–1e2 on a log scale, `y`
/// −3–3); with the platform's (`f64::powf` …) for contrast, which the
/// CPU's column physics took before. Whether every value is the same.
fn math_gate(gpu: &Gpu) -> Result<bool, Box<dyn std::error::Error>> {
    let n = 1_000_000;
    let x: Vec<f64> = (0..n)
        .map(|i| 10f64.powf(-8.0 + 10.0 * ((i as f64 * 0.618_034) % 1.0)))
        .collect();
    let y: Vec<f64> = (0..n)
        .map(|i| -3.0 + 6.0 * ((i as f64 * 0.414_214 + 0.1) % 1.0))
        .collect();
    let device = crate::vertical::elementary(gpu, &x, &y)?;
    let mut same = true;
    println!("dgcom::math, {n} pairs:");
    for (j, name) in ["pow", "exp", "ln", "hypot"].into_iter().enumerate() {
        let (mut differ, mut platform) = (0, 0);
        for i in 0..n {
            let (a, b) = (x[i], y[i]);
            let [cpu, std] = match j {
                0 => [dgcom::math::pow(a, b), a.powf(b)],
                1 => [dgcom::math::exp(b), b.exp()],
                2 => [dgcom::math::ln(a), a.ln()],
                _ => [dgcom::math::hypot(a, b), a.hypot(b)],
            };
            differ += usize::from(cpu.to_bits() != device[i][j].to_bits());
            platform += usize::from(cpu.to_bits() != std.to_bits());
        }
        println!(
            "    {name:<6} device against CPU: {differ} differ (the platform's differs from it in \
             {platform})"
        );
        same &= differ == 0;
    }
    Ok(same)
}

pub fn run(gpu: &Arc<Gpu>) -> Result<(), Box<dyn std::error::Error>> {
    println!("== The 3D vertical step: GLS, the implicit diffusion, the drags (Phase 4.4a) ==");
    // The closure's powers: 1e-10 with contraction, the mutants ≥ 4e-3
    gate::start("vertical3d", Bar::within(1e-8));
    let mut failed = Vec::new();
    if !math_gate(gpu)? {
        failed.push("dgcom::math");
    }
    let cases = [
        ("beach", Shape::BEACH, true),
        ("channel, jittered, stretched", Shape::CHANNEL, false),
    ];
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
        5,
    )?;
    if !failed.is_empty() {
        println!(
            "(failed: {})",
            failed.join("; ")
        );
    }
    Ok(())
}

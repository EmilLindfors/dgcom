//! The implicit vertical advection's gate (P2.7 Phase 4.4b, `notes/gpu.md`):
//! [`Implicit3D`] on the GPU against dgcom on the CPU, value for value:
//!
//! - the split of a stage's `Ω` into its explicit and implicit parts
//!   (`split_vertical_velocity`, which `Hydrostatic3D` applies to the layer
//!   transport's `Ω` before the stage's advection), with the case runner's
//!   `ImplicitVerticalAdvection::default()` (explicit below an outflow
//!   Courant number of 0.4, the explicit part below 0.8);
//! - the relaxation of the stage's inventories of u, v, T, S and the
//!   turbulence's k, ψ with the implicit part
//!   (`apply_implicit_vertical_advection`, which the mode splitter applies
//!   after each stage's combination);
//!
//! on the walled beaches with 40 s forward-Euler steps, whose columns just
//! deeper than the thin depth have centimetre layers and Courant numbers in
//! the tens, and the open channel. Mutants it must catch, and timed at
//! Chiloé's size (96,000 P2 elements × 20 levels).

use std::sync::Arc;
use std::time::Instant;

use cuda_core::DeviceBuffer;
use dgcom::physics::ImplicitVerticalAdvection;
use dgcom::physics::implicit_advection::{
    apply_implicit_vertical_advection, split_vertical_velocity,
};
use dgcom::solver::Blocks;
use dgcom::solver::rhs::{LayerTransport, MetricForm, to_inventory, w_cell_thicknesses};
use dgcom::solver::state::Solution3D;

use crate::gate::{self, Bar};
use crate::hooks::State3D;
use crate::listed::On;
use crate::stage3d::{Snapshots, report, report_mutant, start_state};
use crate::swe2d::{Gpu, NN, ORDER};
use crate::three_d::Statics3D;
use crate::transport3d::{Fixture, LEVELS, Shape, cpu_layers};
use crate::vertical::Implicit3D;

/// The stage's step a relaxation takes, as a share of the elements'
/// forward-Euler step (`β_i Δt`).
const STAGE_SHARE: f64 = 0.5;

/// `fx`'s stage: its state's inventories (`to_inventory` of every field),
/// the layer transport's `Ω` and the elements' relaxation steps.
struct Stage {
    inventories: Solution3D,
    omega: Vec<f64>,
    dt: Vec<f64>,
    d_sigma_w: Vec<f64>,
}

impl Stage {
    fn new(fx: &Fixture) -> Self {
        let ne = fx.n_elements();
        let mut layers = LayerTransport::new(ne, &fx.ops, LEVELS).with_metric(MetricForm::Averaged);
        cpu_layers(fx, &mut layers);
        let mut d_sigma_w = vec![0.0; LEVELS + 1];
        w_cell_thicknesses(fx.sigma.d_sigma(), &mut d_sigma_w);
        let mut inventories = start_state(fx);
        let all = Blocks::All(ne);
        let eta = fx.state.eta.data.clone();
        let Solution3D {
            u,
            v,
            temp,
            salt,
            tke,
            gls,
            ..
        } = &mut inventories;
        for field in [u, v, temp, salt] {
            to_inventory(field, &eta, fx.sigma.d_sigma(), &fx.bathymetry, all);
        }
        for field in [tke, gls] {
            to_inventory(field, &eta, &d_sigma_w, &fx.bathymetry, all);
        }
        Self {
            inventories,
            omega: layers.omega.clone(),
            dt: fx.euler_dt.iter().map(|dt| STAGE_SHARE * dt).collect(),
            d_sigma_w,
        }
    }
}

/// The CPU's split of `stage`'s `Ω` with `split` and relaxation: the
/// snapshots, the columns with an implicit part and the largest Courant
/// number.
fn on_cpu(
    fx: &Fixture,
    stage: &Stage,
    split: &ImplicitVerticalAdvection,
) -> (Snapshots, usize, f64) {
    let ne = fx.n_elements();
    let all = Blocks::All(ne);
    let mut omega = stage.omega.clone();
    let mut implicit = vec![0.0; omega.len()];
    let (columns, courant) = split_vertical_velocity(
        split,
        [&fx.state.eta.data, &fx.bathymetry.data],
        fx.min_column_depth,
        fx.sigma.d_sigma(),
        &fx.euler_dt,
        NN,
        [&mut omega, &mut implicit],
        all,
    );
    let mut relaxed = stage.inventories.clone();
    apply_implicit_vertical_advection(
        &mut relaxed,
        &implicit,
        fx.sigma.d_sigma(),
        &stage.d_sigma_w,
        &fx.bathymetry,
        all,
        &stage.dt,
    );
    let mut out = vec![
        ("split omega".to_string(), omega),
        ("split implicit".to_string(), implicit),
        ("split columns".to_string(), vec![columns as f64]),
    ];
    take(&mut out, &relaxed);
    (out, columns, courant)
}

fn take(out: &mut Snapshots, s: &Solution3D) {
    for (name, field) in [
        ("u", &s.u),
        ("v", &s.v),
        ("T", &s.temp),
        ("S", &s.salt),
        ("k", &s.tke),
        ("psi", &s.gls),
    ] {
        out.push((format!("relaxed {name}"), field.clone()));
    }
}

/// What the device takes: the split, its thin depth, the forward-Euler
/// steps and the relaxation steps.
#[derive(Clone, Copy)]
struct DeviceRun<'a> {
    split: ImplicitVerticalAdvection,
    thin_depth: f64,
    euler_dt: &'a [f64],
    dt: &'a [f64],
}

/// The device's split of `stage`'s `Ω` and relaxation.
fn on_device(
    gpu: &Arc<Gpu>,
    statics: &Statics3D,
    stage: &Stage,
    eta: &DeviceBuffer<f64>,
    run: DeviceRun,
) -> Result<Snapshots, Box<dyn std::error::Error>> {
    let st = &gpu.stream;
    let mut implicit = Implicit3D::new(gpu, statics, run.split, run.thin_depth)?;
    let mut omega = DeviceBuffer::from_host(st, &stage.omega)?;
    let euler_dt = DeviceBuffer::from_host(st, run.euler_dt)?;
    implicit.split(statics, eta, &euler_dt, &mut omega)?;
    let mut inventories = State3D::upload(gpu, &stage.inventories)?;
    let dt = DeviceBuffer::from_host(st, run.dt)?;
    implicit.relax(statics, &mut inventories, &dt, On::ALL)?;
    let mut host = stage.inventories.clone();
    inventories.download_into(gpu, &mut host)?;
    let mut out = vec![
        ("split omega".to_string(), omega.to_host_vec(st)?),
        (
            "split implicit".to_string(),
            implicit.implicit.to_host_vec(st)?,
        ),
        (
            "split columns".to_string(),
            vec![implicit.columns()? as f64],
        ),
    ];
    take(&mut out, &host);
    Ok(out)
}

/// One fixture, and (if `mutants`) the mutants. Whether all are the same to
/// the bit (and every mutant caught).
fn gate(
    gpu: &Arc<Gpu>,
    label: &str,
    shape: Shape,
    mutants: bool,
) -> Result<bool, Box<dyn std::error::Error>> {
    let fx = Fixture::new(shape);
    let ne = fx.n_elements();
    let stage = Stage::new(&fx);
    let split = ImplicitVerticalAdvection::default();
    let (cpu, columns, courant) = on_cpu(&fx, &stage, &split);
    println!(
        "{label}: {ne} P{ORDER} elements × {LEVELS} levels, forward-Euler steps of {} s: {columns} \
         of {} columns with an implicit part, outflow Courant numbers up to {courant:.1}",
        shape.euler_dt,
        ne * NN
    );
    let statics = fx.statics(gpu)?;
    let eta = DeviceBuffer::from_host(&gpu.stream, &fx.state.eta.data)?;
    let thin = fx.min_column_depth;
    let device = |run: DeviceRun| on_device(gpu, &statics, &stage, &eta, run);
    let mut same = report(
        &cpu,
        &device(DeviceRun {
            split,
            thin_depth: thin,
            euler_dt: &fx.euler_dt,
            dt: &stage.dt,
        })?,
    );
    if columns == 0 {
        println!("    the fixture has no implicit part");
        same = false;
    }
    if mutants {
        let half_euler: Vec<f64> = fx.euler_dt.iter().map(|dt| 0.5 * dt).collect();
        let half_dt: Vec<f64> = stage.dt.iter().map(|dt| 0.5 * dt).collect();
        let base = DeviceRun {
            split,
            thin_depth: thin,
            euler_dt: &fx.euler_dt,
            dt: &stage.dt,
        };
        let runs = [
            (
                "explicit up to a Courant number of 0.3",
                DeviceRun {
                    split: ImplicitVerticalAdvection::new(0.3, 0.8),
                    ..base
                },
            ),
            (
                "the split at half the forward-Euler step",
                DeviceRun {
                    euler_dt: &half_euler,
                    ..base
                },
            ),
            (
                "the relaxation over half the stage's step",
                DeviceRun {
                    dt: &half_dt,
                    ..base
                },
            ),
            (
                "the thin depth halved",
                DeviceRun {
                    thin_depth: 0.5 * thin,
                    ..base
                },
            ),
        ];
        for (what, run) in runs {
            same &= report_mutant(what, &cpu, &device(run)?);
        }
    }
    println!(
        "  {}",
        gate::verdict(label, same)
    );
    Ok(same)
}

/// Time the split and the relaxation on the CPU and the GPU, per element and
/// level.
fn time(
    gpu: &Arc<Gpu>,
    label: &str,
    shape: Shape,
    iterations: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let fx = Fixture::new(shape);
    let ne = fx.n_elements();
    let points = (ne * LEVELS) as f64;
    let stage = Stage::new(&fx);
    let split = ImplicitVerticalAdvection::default();
    let all = Blocks::All(ne);
    let mut omega = stage.omega.clone();
    let mut implicit = vec![0.0; omega.len()];
    let mut relaxed = stage.inventories.clone();
    let mut cpu_time = 0.0;
    let mut columns = 0;
    for it in 0..iterations + 1 {
        omega.copy_from_slice(&stage.omega);
        for (to, from) in [
            (&mut relaxed.u, &stage.inventories.u),
            (&mut relaxed.v, &stage.inventories.v),
            (&mut relaxed.temp, &stage.inventories.temp),
            (&mut relaxed.salt, &stage.inventories.salt),
            (&mut relaxed.tke, &stage.inventories.tke),
            (&mut relaxed.gls, &stage.inventories.gls),
        ] {
            to.copy_from_slice(from);
        }
        let t0 = Instant::now();
        columns = split_vertical_velocity(
            &split,
            [&fx.state.eta.data, &fx.bathymetry.data],
            fx.min_column_depth,
            fx.sigma.d_sigma(),
            &fx.euler_dt,
            NN,
            [&mut omega, &mut implicit],
            all,
        )
        .0;
        apply_implicit_vertical_advection(
            &mut relaxed,
            &implicit,
            fx.sigma.d_sigma(),
            &stage.d_sigma_w,
            &fx.bathymetry,
            all,
            &stage.dt,
        );
        if it >= 1 {
            cpu_time += t0.elapsed().as_secs_f64();
        }
    }
    let statics = fx.statics(gpu)?;
    let st = &gpu.stream;
    let mut device = Implicit3D::new(gpu, &statics, split, fx.min_column_depth)?;
    let eta = DeviceBuffer::from_host(st, &fx.state.eta.data)?;
    let euler_dt = DeviceBuffer::from_host(st, &fx.euler_dt)?;
    let dt = DeviceBuffer::from_host(st, &stage.dt)?;
    let start_omega = DeviceBuffer::from_host(st, &stage.omega)?;
    let mut omega = DeviceBuffer::from_host(st, &stage.omega)?;
    let initial = State3D::upload(gpu, &stage.inventories)?;
    let mut inventories = State3D::upload(gpu, &stage.inventories)?;
    let gpu_iterations = 5 * iterations;
    let mut gpu_time = 0.0;
    for it in 0..gpu_iterations + 2 {
        omega.copy_from_device_async(&start_omega, st)?;
        inventories.copy_from(gpu, &initial)?;
        st.synchronize()?;
        let t0 = Instant::now();
        device.split(&statics, &eta, &euler_dt, &mut omega)?;
        device.relax(&statics, &mut inventories, &dt, On::ALL)?;
        st.synchronize()?;
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
        "timing, {label}: {ne} P{ORDER} elements × {LEVELS} levels ({:.1} M points), {columns} \
         columns with an implicit part, {iterations} stages",
        (ne * NN * LEVELS) as f64 / 1e6
    );
    println!(
        "  the split and the relaxation, per element and level: CPU ({threads} threads) {c:.2} ns \
         | GPU {g:.2} ns ({:.1}×); per stage {:.1} ms | {:.2} ms",
        c / g,
        c * points / 1e6,
        g * points / 1e6
    );
    Ok(())
}

pub fn run(gpu: &Arc<Gpu>) -> Result<(), Box<dyn std::error::Error>> {
    println!("== Implicit vertical advection: the split of Ω, the relaxation (Phase 4.4b) ==");
    gate::start("implicit3d", Bar::MIRRORED);
    let long = Shape {
        euler_dt: 40.0,
        ..Shape::BEACH
    };
    let cases = [
        ("beach, parallelograms", long, true),
        (
            "beach, jittered, stretched",
            Shape {
                jitter: 0.2,
                stretched: true,
                ..long
            },
            false,
        ),
        ("channel, jittered, stretched", Shape::CHANNEL, false),
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
            ..long
        },
        10,
    )?;
    if !failed.is_empty() {
        println!(
            "(failed: {})",
            failed.join("; ")
        );
    }
    Ok(())
}

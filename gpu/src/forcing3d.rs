//! The slow forcing's gate (P2.7 Phase 4.4c1, `notes/gpu.md`):
//! [`SlowForcing3D`] on the GPU against `Hydrostatic3D::slow_forcing_into`
//! on the CPU, value for value, for a model with the case runner's terms: a
//! β-plane, the log-layer bottom drag, a farm of net cages, uniform
//! surface and bottom stresses and a wind stress field. On the walled beach
//! (thin columns moving with made-up depth means, which `G` takes at rest)
//! and the open channel. Mutants it must catch, and timed at Chiloé's size
//! (96,000 P2 elements × 20 levels).

use std::sync::Arc;
use std::time::Instant;

use cuda_core::DeviceBuffer;
use dgcom::Reflective2D;
use dgcom::equations::ShallowWater2D;
use dgcom::physics::JerlovWater;
use dgcom::physics::vertical_mixing::Forcing;
use dgcom::physics::{
    AnalyticSurfaceStress, BottomDrag3D, GlsMixing, Hydrostatic3D, LinearEOS, PhysicsBuilder,
    SurfaceStress3D,
};
use dgcom::solver::rhs::{MetricForm, VerticalAdvection};
use dgcom::solver::state::Solution3D;
use dgcom::solver::{SWEFormulation2D, SWESolution2D};
use dgcom::source::{CageDrag2D, CoriolisSource2D, NetCage};
use dgcom::time::ModeSplitPhysics;
use dgcom::vertical::SigmaGrid;

use crate::gate::{self, Bar};
use crate::forcing::{ForcingSetup, SlowForcing3D};
use crate::hooks::State3D;
use crate::stage3d::{Snapshots, report, report_mutant, start_state};
use crate::swe2d::{Gpu, NN, ORDER};
use crate::three_d::Statics3D;
use crate::transport3d::{Fixture, G, LEVELS, RHO0, Shape};
use crate::vertical::{Vertical3D, VerticalSetup, caged_columns};

/// The step's time (s): the wind field's.
const TIME: f64 = 1800.0;
/// The uniform surface and bottom stresses (N/m²).
const STRESS: [f64; 4] = [0.06, -0.02, 0.004, 0.001];

/// A wind stress field over `fx`'s mesh, turning with time.
pub(crate) fn wind(fx: &Fixture) -> AnalyticSurfaceStress {
    AnalyticSurfaceStress::new(&fx.mesh, &fx.ops, |x, y, t| {
        let turn = 1e-4 * t + 1e-3 * x;
        [
            0.08 * turn.cos() + 0.02 * (2e-3 * y).sin(),
            0.08 * turn.sin(),
        ]
    })
}

/// A farm of three net cages where `fx` is deep.
pub(crate) fn cages(fx: &Fixture) -> CageDrag2D {
    let cages = [
        NetCage::circular([300.0, 500.0], 50.0, 6.0, 0.25),
        NetCage::circular([420.0, 520.0], 50.0, 12.0, 0.2),
        NetCage::circular([340.0, 660.0], 50.0, 20.0, 0.3),
    ];
    CageDrag2D::new(&fx.mesh, &fx.ops, &cages)
}

/// A momentum RHS of every layer of `like`'s shape.
fn momentum_rhs(like: &Solution3D) -> Solution3D {
    let mut rhs = like.clone();
    for (t, (u, v)) in rhs.u.iter_mut().zip(rhs.v.iter_mut()).enumerate() {
        *u = 1e-4 * (0.13 * t as f64).sin();
        *v = 1e-4 * (0.17 * t as f64).cos();
    }
    rhs
}

/// A `Hydrostatic3D` of `fx` with the case runner's terms (its boundaries
/// the fixture's).
fn cpu_physics(fx: &Fixture) -> Hydrostatic3D<LinearEOS, GlsMixing, Reflective2D> {
    let swe = PhysicsBuilder::swe_2d(
        Arc::new(fx.mesh.clone()),
        Arc::new(fx.ops.clone()),
        Arc::new(fx.geom.clone()),
        ShallowWater2D::new(G),
        Reflective2D::new(),
    )
    .with_bathymetry(Arc::new(fx.bathymetry.clone()))
    .with_formulation(SWEFormulation2D::EntropyStable)
    .build();
    let physics = Hydrostatic3D::new(
        Arc::new(fx.mesh.clone()),
        Arc::new(fx.ops.clone()),
        Arc::new(fx.geom.clone()),
        Arc::new(fx.sigma.clone()),
        Arc::new(fx.bathymetry.clone()),
        Arc::new(coriolis()),
        LinearEOS::default(),
        GlsMixing::k_epsilon(),
        swe,
        Forcing {
            surface_stress: [STRESS[0], STRESS[1]],
            bottom_stress: [STRESS[2], STRESS[3]],
            surface_heat_flux: 0.0,
        },
        G,
        RHO0,
    )
    .with_min_column_depth(fx.min_column_depth)
    .with_bottom_drag(BottomDrag3D::log_layer(0.01))
    .with_cage_drag(cages(fx))
    .with_surface_stress(wind(fx));
    let mut physics = physics;
    physics.boundaries = fx.boundaries.clone();
    physics
}

/// The CPU's `G` of `state` with `rhs` by `physics`.
fn on_cpu(
    fx: &Fixture,
    physics: &Hydrostatic3D<LinearEOS, GlsMixing, Reflective2D>,
    state: &Solution3D,
    rhs: &Solution3D,
) -> Snapshots {
    let mut g = SWESolution2D::new(fx.n_elements(), NN);
    physics.slow_forcing_into(state, rhs, TIME, &mut g);
    let [h, hu, hv] = g.data;
    vec![("G h".into(), h), ("G hu".into(), hu), ("G hv".into(), hv)]
}

pub(crate) fn coriolis() -> CoriolisSource2D {
    CoriolisSource2D::beta_plane(1.3e-4, 2e-11, 0.0)
}

/// What the device's run takes (a mutant changes one).
#[derive(Clone, Copy)]
struct Run {
    drag: Option<BottomDrag3D>,
    cages: bool,
    wind: bool,
    stress: [f64; 4],
    vertical: VerticalAdvection,
}

impl Run {
    const CASE_RUNNER: Self = Self {
        drag: Some(BottomDrag3D::LogLayer {
            z0: 0.01,
            cd_min: BottomDrag3D::DEFAULT_CD_MIN,
            cd_max: BottomDrag3D::DEFAULT_CD_MAX,
        }),
        cages: true,
        wind: true,
        stress: STRESS,
        vertical: VerticalAdvection::Centred,
    };
}

/// The device's side of `fx`.
struct Device {
    statics: Statics3D,
    forcing: SlowForcing3D,
    vertical: Vertical3D,
    caged: DeviceBuffer<u32>,
    wind: [DeviceBuffer<f64>; 2],
}

impl Device {
    fn new(gpu: &Arc<Gpu>, fx: &Fixture, run: Run) -> Result<Self, Box<dyn std::error::Error>> {
        let st = &gpu.stream;
        let bar_statics = Statics3D::upload(
            gpu,
            &fx.mesh,
            &fx.ops,
            &fx.geom,
            &fx.bathymetry,
            &SigmaGrid::uniform(1),
            &fx.boundaries,
            MetricForm::Averaged,
        )?;
        let cages = cages(fx);
        let forcing = SlowForcing3D::new(
            gpu,
            fx.statics(gpu)?,
            bar_statics,
            &fx.mesh,
            &fx.ops,
            fx.sigma.sigma_rho()[0],
            ForcingSetup {
                coriolis: &coriolis(),
                stress: run.stress,
                bottom_drag: run.drag,
                rho0: RHO0,
                min_column_depth: fx.min_column_depth,
                vertical: run.vertical,
            },
        )?;
        let statics = fx.statics(gpu)?;
        let vertical = Vertical3D::new(
            gpu,
            &statics,
            &fx.sigma,
            VerticalSetup {
                mixing: &GlsMixing::k_epsilon(),
                forcing: Forcing {
                    surface_stress: [run.stress[0], run.stress[1]],
                    bottom_stress: [run.stress[2], run.stress[3]],
                    surface_heat_flux: 0.0,
                },
                bottom_drag: run.drag,
                cages: run.cages.then_some(&cages),
                water: JerlovWater::default(),
                thin_bed: None,
                g: G,
                rho0: RHO0,
                min_column_depth: fx.min_column_depth,
            },
        )?;
        let n = fx.n_elements() * NN;
        let (mut tx, mut ty) = (vec![0.0; n], vec![0.0; n]);
        wind(fx).surface_stress_into(TIME, &mut tx, &mut ty);
        Ok(Self {
            caged: DeviceBuffer::from_host(st, &caged_columns(&cages, n))?,
            wind: [
                DeviceBuffer::from_host(st, &tx)?,
                DeviceBuffer::from_host(st, &ty)?,
            ],
            statics,
            forcing,
            vertical,
        })
    }

    /// `G` of `state` (uploaded) with `rhs` (uploaded).
    fn run(
        &mut self,
        gpu: &Gpu,
        state: &State3D,
        rhs: [&DeviceBuffer<f64>; 2],
        run: Run,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.vertical.drag_rates(&self.statics, state)?;
        let cages = run.cages.then_some((&self.vertical, &self.caged));
        let wind = run.wind.then(|| [&self.wind[0], &self.wind[1]]);
        self.forcing.compute(state, rhs, wind, cages)?;
        let _ = gpu;
        Ok(())
    }

    fn snapshots(&self, gpu: &Gpu) -> Result<Snapshots, Box<dyn std::error::Error>> {
        let st = &gpu.stream;
        let [h, hu, hv] = &self.forcing.g;
        Ok(vec![
            ("G h".into(), h.to_host_vec(st)?),
            ("G hu".into(), hu.to_host_vec(st)?),
            ("G hv".into(), hv.to_host_vec(st)?),
        ])
    }
}

/// `fx`'s device `G` with `run`.
fn on_device(
    gpu: &Arc<Gpu>,
    fx: &Fixture,
    state: &Solution3D,
    rhs: &Solution3D,
    run: Run,
) -> Result<Snapshots, Box<dyn std::error::Error>> {
    let mut device = Device::new(gpu, fx, run)?;
    let values = State3D::upload(gpu, state)?;
    let st = &gpu.stream;
    let (ru, rv) = (
        DeviceBuffer::from_host(st, &rhs.u)?,
        DeviceBuffer::from_host(st, &rhs.v)?,
    );
    device.run(gpu, &values, [&ru, &rv], run)?;
    device.snapshots(gpu)
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
    let state = start_state(&fx);
    let rhs = momentum_rhs(&state);
    let thin = (0..ne * NN)
        .filter(|&idx| state.eta.data[idx] - fx.bathymetry.data[idx] < fx.min_column_depth)
        .count();
    println!(
        "{label}: {ne} P{ORDER} elements × {LEVELS} levels, {thin} thin columns, {} cage entries",
        cages(&fx).nodes().len()
    );
    let cpu = on_cpu(&fx, &cpu_physics(&fx), &state, &rhs);
    let mut same = report(&cpu, &on_device(gpu, &fx, &state, &rhs, Run::CASE_RUNNER)?);
    if mutants {
        let base = Run::CASE_RUNNER;
        let runs = [
            ("no bottom drag", Run { drag: None, ..base }),
            (
                "no cages",
                Run {
                    cages: false,
                    ..base
                },
            ),
            (
                "the wind field left out",
                Run {
                    wind: false,
                    ..base
                },
            ),
            (
                "no bottom stress",
                Run {
                    stress: [STRESS[0], STRESS[1], 0.0, 0.0],
                    ..base
                },
            ),
            (
                "the momentum upwind in the vertical",
                Run {
                    vertical: VerticalAdvection::Upwind,
                    ..base
                },
            ),
        ];
        for (what, run) in runs {
            same &= report_mutant(what, &cpu, &on_device(gpu, &fx, &state, &rhs, run)?);
        }
    }
    println!(
        "  {}",
        gate::verdict(label, same)
    );
    Ok(same)
}

/// Time `G` on the CPU and the GPU at `shape`'s size, per element and level.
fn time(
    gpu: &Arc<Gpu>,
    label: &str,
    shape: Shape,
    iterations: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let fx = Fixture::new(shape);
    let ne = fx.n_elements();
    let points = (ne * LEVELS) as f64;
    let state = start_state(&fx);
    let rhs = momentum_rhs(&state);
    let physics = cpu_physics(&fx);
    let mut cpu_time = 0.0;
    for it in 0..iterations + 1 {
        // A changed state each time: `G` keeps the layers' advection of
        // the state `update_vertical_velocity` last saw
        let mut state = state.clone();
        state.u[0] += 1e-9 * it as f64;
        let t0 = Instant::now();
        let g = on_cpu(&fx, &physics, &state, &rhs);
        if it >= 1 {
            cpu_time += t0.elapsed().as_secs_f64();
        }
        drop(g);
    }
    let mut device = Device::new(gpu, &fx, Run::CASE_RUNNER)?;
    let values = State3D::upload(gpu, &state)?;
    let st = &gpu.stream;
    let (ru, rv) = (
        DeviceBuffer::from_host(st, &rhs.u)?,
        DeviceBuffer::from_host(st, &rhs.v)?,
    );
    let gpu_iterations = 5 * iterations;
    let mut gpu_time = 0.0;
    for it in 0..gpu_iterations + 2 {
        st.synchronize()?;
        let t0 = Instant::now();
        device.run(gpu, &values, [&ru, &rv], Run::CASE_RUNNER)?;
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
        "timing, {label}: {ne} P{ORDER} elements × {LEVELS} levels ({:.1} M points), the case \
         runner's slow forcing (the drag rates included on the GPU), {iterations} steps",
        (ne * NN * LEVELS) as f64 / 1e6
    );
    println!(
        "  G, per element and level: CPU ({threads} threads) {c:.2} ns | GPU {g:.2} ns ({:.1}×); \
         per step {:.1} ms | {:.2} ms",
        c / g,
        c * points / 1e6,
        g * points / 1e6
    );
    Ok(())
}

pub fn run(gpu: &Arc<Gpu>) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "== The slow forcing G: the layers' and the depth mean's advection, Coriolis, stresses, drags (Phase 4.4c1) =="
    );
    gate::start("forcing3d", Bar::MIRRORED);
    let cases = [
        ("beach", Shape::BEACH, true),
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
            ..Shape::BEACH
        },
        3,
    )?;
    if !failed.is_empty() {
        println!(
            "(failed: {})",
            failed.join("; ")
        );
    }
    Ok(())
}

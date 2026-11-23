//! The barotropic pass's gate (P2.7 Phase 4.4c2, `notes/gpu.md`):
//! [`BarotropicPass3D`] on the GPU against dgcom's `barotropic_pass` and
//! what follows it in `ModeSplitIntegrator::step` (the 2D module's
//! post-processing of the average, `barotropic_rates`,
//! `BarotropicTransport::divergence_into`, `pooled_elements`), value for
//! value:
//!
//! - the walled beach with the case runner's 2D module of a 3D run (the
//!   wet/dry stage, Coriolis, no bed friction): its shore, films and dry
//!   columns, three rivers (two in one element, one with a discharge that
//!   changes through the pass), a bottom drag and a layer drag;
//! - the channel of the nesting gate (`nest.rs`): its open boundary from the
//!   parent and the band relaxing the level, the 2D module's own volume
//!   sources, its drying flats and Manning friction.
//!
//! The slow forcing `G` and the drag rates are made up (4.4c1 and 4.4a gate
//! them). Mutants it must catch, and timed at Chiloé's size (96,000 P2
//! elements).

use std::sync::Arc;
use std::time::Instant;

use cuda_core::DeviceBuffer;
use dgcom::boundary::{
    CharacteristicOBC, MultiBoundaryCondition2D, Reflective2D, SWEBoundaryCondition2D,
};
use dgcom::equations::ShallowWater2D;
use dgcom::mesh::Bathymetry2D;
use dgcom::physics::{PhysicsBuilder, PhysicsModule, SWEPhysics2D};
use dgcom::solver::rhs::{face_mass_len, subcell_mass_len};
use dgcom::solver::state::Solution3D;
use dgcom::solver::{DGSolution2D, SWESolution2D, StandardLimiter2D, WetDryConfig};
use dgcom::source::{CoriolisSource2D, River, RiverInflow, RiverSeries, RiverSources};
use dgcom::time::{
    BarotropicFilter, BarotropicPhysics, BarotropicTransport, GlobalPass, MIN_BAROTROPIC_SUBSTEPS,
    PassDrag, PassWork, SspScheme, barotropic_pass, barotropic_rates, column_drag_rates,
    pooled_elements, to_transport,
};
use dgcom::types::ElementIndex;
use dgcom::vertical::SigmaGrid;

use crate::hooks::State3D;
use crate::nest::{Domain, Nest};
use crate::pass::{BarotropicPass3D, DevicePass};
use crate::phase0::DX;
use crate::stage3d::{Snapshots, report, report_mutant, start_state};
use crate::swe2d::{DeviceState, Gpu, GpuSWE2D, NN, ORDER};
use crate::transport3d::{Fixture, G, LEVELS, Shape};

/// The barotropic CFL number of both sides.
const CFL: f64 = 0.5;

/// What the pass takes besides the 2D module and the case (a mutant changes
/// one on the device).
#[derive(Clone, Copy)]
struct Run {
    scheme: SspScheme,
    /// Substeps beyond the CFL's
    extra: usize,
    forcing: bool,
    bottom: bool,
    layers: bool,
    rivers: bool,
}

impl Run {
    const fn with(scheme: SspScheme) -> Self {
        Self {
            scheme,
            extra: 0,
            forcing: true,
            bottom: true,
            layers: true,
            rivers: true,
        }
    }
}

/// One fixture's pass: the 3D state at the step's start (its `η`, `ū`,
/// `v̄`), the bed, the rivers, the step, the made-up `G` and drag rates.
struct Case<'a> {
    state: Solution3D,
    bed: &'a Bathymetry2D,
    rivers: Option<RiverSources>,
    t: f64,
    dt: f64,
    g: SWESolution2D,
    bottom: Vec<f64>,
    d_sigma: Vec<f64>,
    layers: Vec<f64>,
}

impl<'a> Case<'a> {
    /// `state` over `bed` with made-up `G` and drags.
    fn new(
        state: Solution3D,
        bed: &'a Bathymetry2D,
        rivers: Option<RiverSources>,
        t: f64,
        dt: f64,
    ) -> Self {
        let n = state.eta.data.len();
        let mut g = SWESolution2D::new(n / NN, NN);
        for idx in 0..n {
            let x = idx as f64;
            let depth = (state.eta.data[idx] - bed.data[idx]).max(0.0);
            g.data[1][idx] = 2e-5 * depth * (0.37 * x).sin();
            g.data[2][idx] = 1.5e-5 * depth * (0.53 * x).cos();
        }
        let bottom = (0..n)
            .map(|idx| 2e-3 * (1.0 + 0.5 * (0.11 * idx as f64).sin()))
            .collect();
        // A layer drag in every seventh column, on its upper half
        let sigma = SigmaGrid::uniform(LEVELS);
        let mut layers = vec![0.0; n * LEVELS];
        for idx in (0..n).step_by(7) {
            for l in LEVELS / 2..LEVELS {
                layers[idx * LEVELS + l] = 4e-4 * (1.0 + 0.3 * (0.7 * (idx + l) as f64).cos());
            }
        }
        Self {
            state,
            bed,
            rivers,
            t,
            dt,
            g,
            bottom,
            d_sigma: sigma.d_sigma().to_vec(),
            layers,
        }
    }

    /// The pass's substeps from the step's largest substep `dt_bt_max`, as
    /// `ModeSplitIntegrator::step` takes them.
    fn substeps(&self, dt_bt_max: f64, run: Run) -> usize {
        ((self.dt / dt_bt_max).ceil() as usize).max(MIN_BAROTROPIC_SUBSTEPS) + run.extra
    }
}

/// NaN as one value (the CPU's propagated NaN, the device's constant).
fn canonical(v: Vec<f64>) -> Vec<f64> {
    v.into_iter()
        .map(|x| if x.is_nan() { f64::NAN } else { x })
        .collect()
}

/// The CPU's pass of `case` with `physics`, and what follows it.
fn on_cpu<BC: SWEBoundaryCondition2D + Sync>(
    physics: &SWEPhysics2D<BC>,
    case: &Case,
    run: Run,
) -> Snapshots {
    let (mesh, ops, geom) = (physics.mesh(), physics.operators(), physics.geometry());
    let (ne, nn) = (mesh.n_elements, ops.n_nodes);
    let n_rivers = case.rivers.as_ref().map_or(0, RiverSources::len);
    let mut q = SWESolution2D::new(ne, nn);
    to_transport(&case.state, case.bed, &mut q);
    let n_bt = case.substeps(physics.compute_dt_ssp(&q, CFL, Some(run.scheme)), run);
    let filter = BarotropicFilter::new(n_bt);
    let mut column = vec![0.0; ne * nn];
    column_drag_rates(&case.d_sigma, &case.layers, &mut column);
    let mut work = PassWork::new(ne, ops);
    let mut q_avg = SWESolution2D::new(ne, nn);
    let mut transport = BarotropicTransport::new(
        ne,
        nn,
        face_mass_len(mesh, ops),
        subcell_mass_len(mesh, ops),
        n_rivers,
    );
    let clips_before = physics.negative_depth_clips();
    barotropic_pass(
        physics,
        GlobalPass {
            filter: &filter,
            scheme: run.scheme,
            t: case.t,
            dt_bt: case.dt / n_bt as f64,
            forcing: &case.g,
            drag: PassDrag {
                bottom: Some(&case.bottom),
                column: Some(&column),
            },
            rivers: case.rivers.as_ref(),
        },
        &mut work,
        &mut q,
        &mut q_avg,
        &mut transport,
    );
    physics.post_process(&mut q_avg);
    let mut rates: [DGSolution2D; 5] = std::array::from_fn(|_| DGSolution2D::new(ne, nn));
    let [a, b, c, d, e] = &mut rates;
    barotropic_rates(&case.state, case.bed, &q_avg, case.dt, [a, b, c, d, e]);
    let mut divergence = DGSolution2D::new(ne, nn);
    transport.divergence_into(ops, geom, physics.metric_form(), &mut divergence);
    let mut column_source = vec![0.0; ne * nn];
    let mut pooled = vec![false; ne];
    let inflow = case.rivers.as_ref().map(|rivers| RiverInflow {
        rivers,
        discharge: &transport.discharge,
    });
    pooled_elements(
        &case.state.eta.data,
        case.bed,
        &rates[0],
        &divergence,
        &transport,
        inflow,
        case.dt,
        &mut column_source,
        &mut pooled,
    );
    let clips = physics.negative_depth_clips() - clips_before;
    let [h, hu, hv] = q.data;
    let [ah, ahu, ahv] = q_avg.data;
    let [r_eta, r_u, r_v, r_hu, r_hv] = rates.map(|r| r.data);
    vec![
        (
            "pass substeps/clips".into(),
            vec![n_bt as f64, clips as f64],
        ),
        ("last substep h".into(), h),
        ("last substep hu".into(), hu),
        ("last substep hv".into(), hv),
        ("average h".into(), ah),
        ("average hu".into(), ahu),
        ("average hv".into(), ahv),
        ("DU_avg2 hu".into(), transport.hu.data),
        ("DU_avg2 hv".into(), transport.hv.data),
        ("DU_avg2 faces".into(), transport.face),
        ("DU_avg2 subcells".into(), canonical(transport.subcell)),
        ("means volume-sources".into(), transport.mass_source.data),
        ("means discharge".into(), transport.discharge),
        ("rate eta".into(), r_eta),
        ("rate ubar".into(), r_u),
        ("rate vbar".into(), r_v),
        ("rate Dubar".into(), r_hu),
        ("rate Dvbar".into(), r_hv),
        ("balance divergence".into(), divergence.data),
        ("balance column-sources".into(), column_source),
        (
            "balance pooled".into(),
            pooled.iter().map(|&p| f64::from(u8::from(p))).collect(),
        ),
    ]
}

/// The device's pass of `case` with the GPU's twin of `physics`.
fn on_device<BC: SWEBoundaryCondition2D + Sync>(
    gpu: &Arc<Gpu>,
    physics: &SWEPhysics2D<BC>,
    case: &Case,
    run: Run,
) -> Result<Snapshots, Box<dyn std::error::Error>> {
    let form = physics.device_form()?;
    let module = GpuSWE2D::from_form(gpu, &form)?;
    let mut device = Device::new(gpu, &module, physics.metric_form(), case)?;
    device.snapshots(gpu, &module, case, run)
}

/// The device's side of a case: the pass and the uploads.
struct Device {
    pass: BarotropicPass3D,
    state: State3D,
    q: DeviceState,
    g: DeviceState,
    zero_g: DeviceState,
    bottom: DeviceBuffer<f64>,
    d_sigma: DeviceBuffer<f64>,
    layers: DeviceBuffer<f64>,
    column: DeviceBuffer<f64>,
}

impl Device {
    fn new(
        gpu: &Arc<Gpu>,
        module: &GpuSWE2D,
        metric: dgcom::solver::rhs::MetricForm,
        case: &Case,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let st = &gpu.stream;
        let ne = case.state.n_elements;
        Ok(Self {
            pass: BarotropicPass3D::new(gpu, module, case.bed, metric, case.rivers.as_ref())?,
            state: State3D::upload(gpu, &case.state)?,
            q: DeviceState::upload(gpu, &SWESolution2D::new(ne, NN)),
            g: DeviceState::upload(gpu, &case.g),
            zero_g: DeviceState::upload(gpu, &SWESolution2D::new(ne, NN)),
            bottom: DeviceBuffer::from_host(st, &case.bottom)?,
            d_sigma: DeviceBuffer::from_host(st, &case.d_sigma)?,
            layers: DeviceBuffer::from_host(st, &case.layers)?,
            column: DeviceBuffer::zeroed(st, ne * NN)?,
        })
    }

    /// The pass with `run`, and its snapshots.
    fn snapshots(
        &mut self,
        gpu: &Gpu,
        module: &GpuSWE2D,
        case: &Case,
        run: Run,
    ) -> Result<Snapshots, Box<dyn std::error::Error>> {
        let st = &gpu.stream;
        let clips_before = module.negative_depth_clips();
        let n_bt = self.step(module, case, run)?;
        let clips = module.negative_depth_clips() - clips_before;
        let pass = &self.pass;
        let down = |b: &DeviceBuffer<f64>| b.to_host_vec(st);
        let q = self.q.download();
        let avg = pass.q_avg.download();
        let [h, hu, hv] = q.data;
        let [ah, ahu, ahv] = avg.data;
        let f = &pass.flux;
        let [r_u, r_v, r_hu, r_hv] = &pass.rates;
        let pooled = f.pooled.to_host_vec(st)?;
        Ok(vec![
            (
                "pass substeps/clips".into(),
                vec![n_bt as f64, clips as f64],
            ),
            ("last substep h".into(), h),
            ("last substep hu".into(), hu),
            ("last substep hv".into(), hv),
            ("average h".into(), ah),
            ("average hu".into(), ahu),
            ("average hv".into(), ahv),
            ("DU_avg2 hu".into(), down(&f.hu)?),
            ("DU_avg2 hv".into(), down(&f.hv)?),
            ("DU_avg2 faces".into(), down(&f.face)?),
            ("DU_avg2 subcells".into(), canonical(down(&f.subcell)?)),
            ("means volume-sources".into(), down(&pass.mass_source)?),
            ("means discharge".into(), pass.discharge.clone()),
            ("rate eta".into(), down(&f.eta_rate)?),
            ("rate ubar".into(), down(r_u)?),
            ("rate vbar".into(), down(r_v)?),
            ("rate Dubar".into(), down(r_hu)?),
            ("rate Dvbar".into(), down(r_hv)?),
            ("balance divergence".into(), down(&pass.divergence)?),
            ("balance column-sources".into(), down(&pass.column_source)?),
            (
                "balance pooled".into(),
                pooled
                    .iter()
                    .map(|&p| f64::from(u8::from(p != 0)))
                    .collect(),
            ),
        ])
    }

    /// One step's pass on the device: the transport of the state, the
    /// substeps, `Λ̄`, the pass and what follows it. Its substeps.
    fn step(
        &mut self,
        module: &GpuSWE2D,
        case: &Case,
        run: Run,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        let pass = &mut self.pass;
        pass.to_transport(&self.state, &mut self.q)?;
        let n_bt = case.substeps(module.compute_dt_ssp(&self.q, CFL, Some(run.scheme)), run);
        let filter = BarotropicFilter::new(n_bt);
        pass.column_drag(&self.d_sigma, &self.layers, &mut self.column)?;
        pass.run(
            module,
            DevicePass {
                filter: &filter,
                scheme: run.scheme,
                t: case.t,
                dt_bt: case.dt / n_bt as f64,
                forcing: if run.forcing { &self.g } else { &self.zero_g },
                bottom: run.bottom.then_some(&self.bottom),
                column: run.layers.then_some(&self.column),
                rivers: case.rivers.as_ref().filter(|_| run.rivers),
            },
            &mut self.q,
        )?;
        pass.finish(module, &self.state, case.dt)?;
        Ok(n_bt)
    }
}

/// One case on both sides with `base`, and (if `mutants`) the mutants:
/// `others` are the mutants of a changed 2D module. Whether all are the same
/// to the bit (and every mutant caught).
fn gate<BC: SWEBoundaryCondition2D + Sync>(
    gpu: &Arc<Gpu>,
    physics: &SWEPhysics2D<BC>,
    case: &Case,
    base: Run,
    mutants: bool,
    others: &[(&str, &SWEPhysics2D<BC>)],
) -> Result<bool, Box<dyn std::error::Error>> {
    let cpu = on_cpu(physics, case, base);
    let device = on_device(gpu, physics, case, base)?;
    let ne = case.state.n_elements;
    let subcells = cpu[10]
        .1
        .chunks(NSUB_OF_P2)
        .filter(|s| s[0].is_finite())
        .count();
    let pooled = cpu
        .last()
        .map_or(0, |(_, p)| p.iter().filter(|&&p| p > 0.0).count());
    let sourced = cpu[11].1.iter().filter(|&&s| s != 0.0).count();
    println!(
        "  {} substeps of SSP-RK{} ({} clips), {subcells} of {ne} elements on the subcells \
         throughout, {pooled} balanced only as a whole, {sourced} nodes with a volume source",
        cpu[0].1[0],
        match base.scheme {
            SspScheme::Rk3 => "3",
            SspScheme::Rk43 => "(4,3)",
        },
        cpu[0].1[1],
    );
    let mut same = report(&cpu, &device);
    if mutants {
        let other_scheme = match base.scheme {
            SspScheme::Rk3 => SspScheme::Rk43,
            SspScheme::Rk43 => SspScheme::Rk3,
        };
        let mut runs = vec![
            (
                "no slow forcing",
                Run {
                    forcing: false,
                    ..base
                },
            ),
            (
                "no bottom drag",
                Run {
                    bottom: false,
                    ..base
                },
            ),
            (
                "no layer drag",
                Run {
                    layers: false,
                    ..base
                },
            ),
            ("one substep more", Run { extra: 1, ..base }),
            (
                "the other scheme's substeps",
                Run {
                    scheme: other_scheme,
                    ..base
                },
            ),
        ];
        if case.rivers.is_some() {
            runs.push((
                "the rivers left out",
                Run {
                    rivers: false,
                    ..base
                },
            ));
        }
        for (what, run) in runs {
            same &= report_mutant(what, &cpu, &on_device(gpu, physics, case, run)?);
        }
        for &(what, other) in others {
            same &= report_mutant(what, &cpu, &on_device(gpu, other, case, base)?);
        }
    }
    println!(
        "  {}",
        if same {
            "bit for bit"
        } else {
            "NOT bit for bit"
        }
    );
    Ok(same)
}

/// Subcell interfaces of a P2 element.
const NSUB_OF_P2: usize = crate::three_d::NSUB;

/// The case runner's 2D module of a 3D run on `fx`: the wet/dry stage,
/// Coriolis on a β-plane, walls, no bed friction.
fn beach_physics(fx: &Fixture) -> SWEPhysics2D<Reflective2D> {
    PhysicsBuilder::swe_2d(
        Arc::new(fx.mesh.clone()),
        Arc::new(fx.ops.clone()),
        Arc::new(fx.geom.clone()),
        ShallowWater2D::new(G),
        Reflective2D::new(),
    )
    .with_bathymetry(Arc::new(fx.bathymetry.clone()))
    .with_limiter(StandardLimiter2D::Positivity(WetDryConfig::DEFAULT_H_DRY))
    .with_wet_dry(WetDryConfig::default())
    .with_source(CoriolisSource2D::beta_plane(1.3e-4, 2e-11, 0.0))
    .build()
}

/// Three rivers on `fx`'s wet side, two of them in one element, the third's
/// discharge rising through the pass.
fn beach_rivers(fx: &Fixture) -> RiverSources {
    let (lx, ly) = (36.0 * DX, 20.0 * DX);
    let at = |fx: f64, fy: f64| [fx * lx, fy * ly];
    let rising = RiverSeries::new(vec![0.0, 600.0], vec![5.0, 9.0]).expect("a series");
    let list = vec![
        River::new("a", at(0.21, 0.47), 30.0),
        River::new("b", at(0.212, 0.472), 12.0),
        River::new("c", at(0.33, 0.21), 6.0).with_discharge(rising),
    ];
    let rivers = RiverSources::new(list, &fx.mesh, &fx.geom, &fx.bathymetry, 300.0, &[])
        .expect("the rivers are on the beach");
    assert_eq!(
        rivers.element(0),
        rivers.element(1),
        "two rivers share an element"
    );
    rivers
}

/// The beach's case: its stage gate's start state (thin columns moving),
/// with rivers if `rivers`.
fn beach_case(fx: &Fixture, rivers: bool) -> Case<'_> {
    Case::new(
        start_state(fx),
        &fx.bathymetry,
        rivers.then(|| beach_rivers(fx)),
        120.0,
        20.0,
    )
}

/// The nested channel's 3D state (one level): still water but on the
/// flats, a slow current.
fn nest_state(domain: &Domain) -> Solution3D {
    let (ne, nn) = (domain.mesh.n_elements, domain.ops.n_nodes);
    let mut state = Solution3D::new(ne, nn, 1);
    for k in 0..ne {
        for i in 0..nn {
            let idx = k * nn + i;
            let b = domain.bed.get(ElementIndex::new(k), i);
            let x = idx as f64;
            state.eta.data[idx] = (0.02 * (0.013 * x).sin()).max(b);
            state.ubar.data[idx] = 0.15 * (0.021 * x).cos();
            state.vbar.data[idx] = 0.05 * (0.017 * x).sin();
        }
    }
    state
}

/// Time the pass and what follows it on the CPU and the GPU at `fx`'s size.
fn time(gpu: &Arc<Gpu>, fx: &Fixture, iterations: usize) -> Result<(), Box<dyn std::error::Error>> {
    let physics = beach_physics(fx);
    let case = beach_case(fx, true);
    let run = Run::with(SspScheme::Rk3);
    let ne = fx.n_elements();
    let mut n_bt = 0.0;
    let mut cpu_time = 0.0;
    for it in 0..iterations + 1 {
        let t0 = Instant::now();
        let snapshots = on_cpu(&physics, &case, run);
        if it >= 1 {
            cpu_time += t0.elapsed().as_secs_f64();
        }
        n_bt = snapshots[0].1[0];
    }
    let form = physics.device_form()?;
    let module = GpuSWE2D::from_form(gpu, &form)?;
    let mut device = Device::new(gpu, &module, physics.metric_form(), &case)?;
    let gpu_iterations = 3 * iterations;
    let mut gpu_time = 0.0;
    let st = &gpu.stream;
    for it in 0..gpu_iterations + 1 {
        st.synchronize()?;
        let t0 = Instant::now();
        device.step(&module, &case, run)?;
        st.synchronize()?;
        if it >= 1 {
            gpu_time += t0.elapsed().as_secs_f64();
        }
    }
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let (c, g) = (
        cpu_time / iterations as f64,
        gpu_time / gpu_iterations as f64,
    );
    let stages = 3.0 * n_bt * BarotropicFilter::new(n_bt as usize).reach();
    println!(
        "timing, beach: {ne} P{ORDER} elements, a pass of {n_bt} substeps of SSP-RK3 \
         ({stages:.0} stages) with three rivers and what follows it"
    );
    println!(
        "  per pass: CPU ({threads} threads) {:.1} ms | GPU {:.1} ms ({:.1}×); per stage \
         and element {:.1} ns | {:.1} ns",
        c * 1e3,
        g * 1e3,
        c / g,
        c * 1e9 / (stages * ne as f64),
        g * 1e9 / (stages * ne as f64),
    );
    Ok(())
}

pub fn run(gpu: &Arc<Gpu>) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "== The barotropic pass on global substeps: G, the drags, DU_avg2, the filter, the rivers, the band's sources (Phase 4.4c2) =="
    );
    let mut failed = Vec::new();
    let fx = Fixture::new(Shape::BEACH);
    let physics = beach_physics(&fx);
    for (scheme, mutants) in [(SspScheme::Rk3, true), (SspScheme::Rk43, false)] {
        let label = match scheme {
            SspScheme::Rk3 => "beach, rivers, SSP-RK3",
            SspScheme::Rk43 => "beach, rivers, SSP-RK(4,3)",
        };
        println!("{label}: {} P{ORDER} elements", fx.n_elements());
        let case = beach_case(&fx, true);
        if !gate(gpu, &physics, &case, Run::with(scheme), mutants, &[])? {
            failed.push(label);
        }
    }

    // The nested channel: its open boundary and the band's volume sources
    let nest = Nest::cubic();
    let domain = Domain::new(&nest)?;
    let wall = Reflective2D::new();
    let open = CharacteristicOBC::new(domain.parent.clone());
    let bc = || MultiBoundaryCondition2D::new(&wall).with_open(&open);
    let physics = domain.physics(&nest, bc());
    let momentum_only = Nest {
        momentum_only: true,
        ..Nest::cubic()
    };
    let other = domain.physics(&momentum_only, bc());
    let state = nest_state(&domain);
    let case = Case::new(state, &domain.bed, None, 7200.0, 60.0);
    let label = "nested channel, its band's sources, SSP-RK(4,3)";
    println!(
        "{label}: {} P{ORDER} elements, the band's level relaxed",
        domain.mesh.n_elements
    );
    let others = [("the band relaxing the momentum alone", &other)];
    if !gate(
        gpu,
        &physics,
        &case,
        Run::with(SspScheme::Rk43),
        true,
        &others,
    )? {
        failed.push(label);
    }

    time(
        gpu,
        &Fixture::new(Shape {
            nx: 320,
            ny: 300,
            jitter: 0.15,
            stretched: true,
            ..Shape::BEACH
        }),
        3,
    )?;
    if !failed.is_empty() {
        println!(
            "(not bit for bit: {}; expected only in a build with multiply–add contraction)",
            failed.join("; ")
        );
    }
    Ok(())
}

//! The barotropic pass's gate (P2.7 Phase 4.4c2–c3, `notes/gpu.md`):
//! [`BarotropicPass3D`] on the GPU against dgcom's `barotropic_pass` on
//! global substeps and `barotropic_pass_local` on local time steps (the
//! levels from each side's own element steps), and what follows them in
//! `ModeSplitIntegrator::step` (the 2D module's post-processing of the
//! average, `barotropic_rates`, `BarotropicTransport::divergence_into`,
//! `pooled_elements`), value for value:
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
//! them). Then the step's merged cells from what each side's pass moved
//! (Phase 4.4d: [`Merged3D::find`] against `MergedCells::find`), with
//! made-up forward-Euler steps of the 3D stages. Mutants it must catch, and
//! timed at Chiloé's size (96,000 P2 elements).

use std::sync::Arc;
use std::time::Instant;

use cuda_core::DeviceBuffer;
use dgcom::boundary::{
    CharacteristicOBC, MultiBoundaryCondition2D, Reflective2D, SWEBoundaryCondition2D,
};
use dgcom::equations::ShallowWater2D;
use dgcom::mesh::Bathymetry2D;
use dgcom::physics::{PhysicsBuilder, PhysicsModule, SWEPhysics2D};
use dgcom::solver::rhs::{MergedCells, StepWater, face_mass_len, subcell_mass_len};
use dgcom::solver::state::Solution3D;
use dgcom::solver::{DGSolution2D, SWESolution2D, StandardLimiter2D, WetDryConfig};
use dgcom::source::{CoriolisSource2D, River, RiverInflow, RiverSeries, RiverSources};
use dgcom::time::{
    BarotropicFilter, BarotropicPhysics, BarotropicTransport, LocalPass, MIN_BAROTROPIC_SUBSTEPS,
    PassDrag, PassInputs, PassWork, SspScheme, barotropic_pass, barotropic_pass_local,
    barotropic_rates, column_drag_rates, pooled_elements, to_transport,
};
use dgcom::types::ElementIndex;
use dgcom::vertical::SigmaGrid;

use crate::gate::{self, Bar};
use crate::hooks::{Merged3D, MergedLists, State3D};
use crate::nest::{Domain, Nest};
use crate::pass::{BarotropicPass3D, DevicePass, LocalPass3D};
use crate::phase0::DX;
use crate::stage3d::{Snapshots, report, report_mutant, start_state};
use crate::swe2d::{DeviceState, Gpu, GpuSWE2D, N1, NN, ORDER};
use crate::three_d::SLOTS;
use crate::transport3d::{Fixture, G, LEVELS, Shape};

/// What the pass takes besides the 2D module and the case (a mutant changes
/// one on the device).
#[derive(Clone, Copy)]
struct Run {
    scheme: SspScheme,
    /// The barotropic CFL number, and the levels of local time stepping
    /// (0: global substeps)
    cfl: f64,
    levels: usize,
    /// Substeps beyond the CFL's
    extra: usize,
    forcing: bool,
    bottom: bool,
    layers: bool,
    rivers: bool,
    /// A change to the merged cells' search on the device
    find: Find,
}

/// What a mutant changes in the merged cells' search on the device.
#[derive(Clone, Copy, PartialEq)]
enum Find {
    /// Nothing
    Same,
    /// The cells' water without the face fluxes, the subcell fluxes, the
    /// column sources
    NoFaces,
    NoSubcells,
    NoSources,
    /// No column thin (a thin depth of zero)
    NoneThin,
    /// Every element at the coarse level's forward-Euler step
    CoarseEuler,
    /// The forward-Euler steps halved
    HalfEuler,
    /// The cells' water without `∂η/∂t`
    NoRate,
}

/// The merged cells' mutants. The thin depth itself moves no group on these
/// fixtures (a hundredth of it neither): a cell drains as a film, or dries
/// within the step, where any depth is thin. Only the beach's groups move
/// with the forward-Euler steps and `∂η/∂t` (the last three): the nested
/// channel's few drain many times over within any of these steps.
const FIND_MUTANTS: [(&str, Find); 7] = [
    ("the cells' water without the face fluxes", Find::NoFaces),
    (
        "the cells' water without the subcell fluxes",
        Find::NoSubcells,
    ),
    (
        "the cells' water without the column sources",
        Find::NoSources,
    ),
    ("no column thin", Find::NoneThin),
    (
        "every element at the coarse forward-Euler step",
        Find::CoarseEuler,
    ),
    ("the forward-Euler steps halved", Find::HalfEuler),
    ("the cells' water without ∂η/∂t", Find::NoRate),
];

impl Run {
    /// Global substeps of `scheme` at CFL 0.5.
    const fn global(scheme: SspScheme) -> Self {
        Self {
            scheme,
            cfl: 0.5,
            levels: 0,
            extra: 0,
            forcing: true,
            bottom: true,
            layers: true,
            rivers: true,
            find: Find::Same,
        }
    }

    /// The case runner's 3D pass: SSP-RK(4,3) at CFL 2 on two levels of
    /// local time stepping (`[three_d]`'s defaults).
    const CASE_RUNNER: Self = Self {
        cfl: 2.0,
        levels: 2,
        ..Self::global(SspScheme::Rk43)
    };
}

/// One fixture's pass: the 3D state at the step's start (its `η`, `ū`,
/// `v̄`), the bed, the rivers, the step, the made-up `G` and drag rates; the
/// 3D stages' forward-Euler steps and the thin depth of the merged cells'
/// search.
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
    euler_dt: Vec<f64>,
    thin_depth: f64,
}

impl<'a> Case<'a> {
    /// `state` over `bed` with made-up `G` and drags; columns shallower than
    /// `thin_depth` thin, and the 3D stages those of SSP-RK3 on two levels
    /// (every other element's step halved: its forward-Euler step).
    fn new(
        state: Solution3D,
        bed: &'a Bathymetry2D,
        rivers: Option<RiverSources>,
        t: f64,
        dt: f64,
        thin_depth: f64,
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
        let euler_dt = (0..n / NN).map(|k| dt / (1 + k % 2) as f64).collect();
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
            euler_dt,
            thin_depth,
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

/// The CPU's pass of `case` with `physics`, and what follows it to the
/// step's merged cells (their search's time added to `find_time`).
fn on_cpu<BC: SWEBoundaryCondition2D + Sync>(
    physics: &SWEPhysics2D<BC>,
    case: &Case,
    run: Run,
    find_time: &mut f64,
) -> Snapshots {
    let (mesh, ops, geom) = (physics.mesh(), physics.operators(), physics.geometry());
    let (ne, nn) = (mesh.n_elements, ops.n_nodes);
    let n_rivers = case.rivers.as_ref().map_or(0, RiverSources::len);
    let mut q = SWESolution2D::new(ne, nn);
    to_transport(&case.state, case.bed, &mut q);
    // The substeps, as `ModeSplitIntegrator::step` takes them: global, or
    // the coarse substep of local time steps at most Δt over the fewest
    let mut local = (run.levels > 0).then(|| LocalPass::new(run.levels, run.scheme, &q, n_rivers));
    let dt_bt_max = match &mut local {
        Some(local) => {
            let steps = physics
                .local_time_stepping()
                .filter(|l| l.is_element_local())
                .expect("the 2D module steps locally");
            let cap = case.dt / MIN_BAROTROPIC_SUBSTEPS as f64;
            local.assign_levels(steps, mesh, &q, run.cfl, cap)
        }
        None => physics.compute_dt_ssp(&q, run.cfl, Some(run.scheme)),
    };
    let n_bt = case.substeps(dt_bt_max, run);
    let filter = BarotropicFilter::new(n_bt);
    let mut column = vec![0.0; ne * nn];
    column_drag_rates(&case.d_sigma, &case.layers, &mut column);
    let mut q_avg = SWESolution2D::new(ne, nn);
    let mut transport = BarotropicTransport::new(
        ne,
        nn,
        face_mass_len(mesh, ops),
        subcell_mass_len(mesh, ops),
        n_rivers,
    );
    let clips_before = physics.negative_depth_clips();
    let inputs = PassInputs {
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
    };
    let levels: Vec<f64> = match &mut local {
        Some(local) => {
            barotropic_pass_local(physics, inputs, local, &mut q, &mut q_avg, &mut transport);
            local
                .stepper()
                .levels()
                .iter()
                .map(|&l| f64::from(l))
                .collect()
        }
        None => {
            let mut work = PassWork::new(ne, ops);
            barotropic_pass(
                physics,
                inputs,
                &mut work,
                &mut q,
                &mut q_avg,
                &mut transport,
            );
            // Every element on the one level
            vec![0.0; ne]
        }
    };
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
    // The thin cells that lose more than their water in a stage
    let t0 = Instant::now();
    let mut merged = MergedCells::new(ne, nn);
    merged.find(&StepWater {
        mesh,
        ops,
        geom,
        bathymetry: case.bed,
        eta: &case.state.eta.data,
        eta_rate: &rates[0].data,
        dt: case.dt,
        euler_dt: &case.euler_dt,
        face: &transport.face,
        subcell: &transport.subcell,
        source: &column_source,
        thin_depth: case.thin_depth,
    });
    *find_time += t0.elapsed().as_secs_f64();
    let merged_cells = merged_snapshots(
        &MergedLists::from_tables(merged.tables()),
        merged.counts(),
        ne,
    );
    let clips = physics.negative_depth_clips() - clips_before;
    let [h, hu, hv] = q.data;
    let [ah, ahu, ahv] = q_avg.data;
    let [r_eta, r_u, r_v, r_hu, r_hv] = rates.map(|r| r.data);
    let mut snapshots: Snapshots = vec![
        (
            "pass substeps/clips".into(),
            vec![n_bt as f64, clips as f64],
        ),
        ("pass levels".into(), levels),
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
    ];
    snapshots.extend(merged_cells);
    snapshots
}

/// The step's merged cells as snapshots of fixed lengths (so that a
/// mutant's other lists compare too): every node's group and whether it is
/// joined; every element's group (whether it has one), whether it is
/// flushed, its place in the flushed list (from 1; 0: not flushed), the
/// water entering and leaving it, and its inflows in order (from `k·SLOTS`:
/// the node the water comes from plus 1, the rate); the cells merged and
/// flushed.
fn merged_snapshots(lists: &MergedLists, counts: [usize; 2], ne: usize) -> Snapshots {
    let as_f64 = |v: &[u32]| v.iter().map(|&x| f64::from(x)).collect::<Vec<f64>>();
    let mut grouped = vec![0.0; ne];
    for &k in &lists.elements {
        grouped[k as usize] = 1.0;
    }
    let (mut order, mut entering, mut leaving) = (vec![0.0; ne], vec![0.0; ne], vec![0.0; ne]);
    for (j, &(k, i, o)) in lists.flushed_elements.iter().enumerate() {
        let k = k as usize;
        order[k] = (j + 1) as f64;
        entering[k] = i;
        leaving[k] = o;
    }
    let (mut from, mut rate) = (vec![0.0; ne * SLOTS], vec![0.0; ne * SLOTS]);
    let mut seen = vec![0; ne];
    for &(k, node, r) in &lists.inflows {
        let k = k as usize;
        let at = k * SLOTS + seen[k];
        seen[k] += 1;
        from[at] = f64::from(node) + 1.0;
        rate[at] = r;
    }
    vec![
        ("merged group".into(), as_f64(&lists.group)),
        ("merged joined".into(), as_f64(&lists.joined)),
        ("merged grouped".into(), grouped),
        ("merged flushed".into(), as_f64(&lists.flushed)),
        ("merged order".into(), order),
        ("merged entering".into(), entering),
        ("merged leaving".into(), leaving),
        ("merged inflow-from".into(), from),
        ("merged inflow-rate".into(), rate),
        ("merged cells".into(), counts.map(|c| c as f64).to_vec()),
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
    device.snapshots(&module, case, run)
}

/// The device's side of a case: the pass, the merged cells and the uploads.
struct Device {
    gpu: Arc<Gpu>,
    pass: BarotropicPass3D,
    merged: Merged3D,
    state: State3D,
    q: DeviceState,
    g: DeviceState,
    zero_g: DeviceState,
    bottom: DeviceBuffer<f64>,
    d_sigma: DeviceBuffer<f64>,
    layers: DeviceBuffer<f64>,
    column: DeviceBuffer<f64>,
    /// Zeros in place of the face fluxes, the subcell fluxes, and a value
    /// per node (mutants)
    zero_faces: DeviceBuffer<f64>,
    zero_subcells: DeviceBuffer<f64>,
    zero_nodes: DeviceBuffer<f64>,
    /// The pass on local time steps, from the first step that takes it
    local: Option<LocalPass3D>,
    /// The merged cells' search's time, summed (s)
    find_time: f64,
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
        let pass = BarotropicPass3D::new(gpu, module, case.bed, metric, case.rivers.as_ref())?;
        Ok(Self {
            merged: Merged3D::none(gpu, pass.statics())?,
            pass,
            state: State3D::upload(gpu, &case.state)?,
            q: DeviceState::upload(gpu, &SWESolution2D::new(ne, NN)),
            g: DeviceState::upload(gpu, &case.g),
            zero_g: DeviceState::upload(gpu, &SWESolution2D::new(ne, NN)),
            bottom: DeviceBuffer::from_host(st, &case.bottom)?,
            d_sigma: DeviceBuffer::from_host(st, &case.d_sigma)?,
            layers: DeviceBuffer::from_host(st, &case.layers)?,
            column: DeviceBuffer::zeroed(st, ne * NN)?,
            zero_faces: DeviceBuffer::zeroed(st, ne * 4 * N1)?,
            zero_subcells: DeviceBuffer::zeroed(st, ne * NSUB_OF_P2)?,
            zero_nodes: DeviceBuffer::zeroed(st, ne * NN)?,
            local: None,
            find_time: 0.0,
            gpu: gpu.clone(),
        })
    }

    /// The pass with `run` and the merged cells, and their snapshots.
    fn snapshots(
        &mut self,
        module: &GpuSWE2D,
        case: &Case,
        run: Run,
    ) -> Result<Snapshots, Box<dyn std::error::Error>> {
        let gpu = self.gpu.clone();
        let st = &gpu.stream;
        let clips_before = module.negative_depth_clips();
        let (n_bt, levels) = self.step(module, case, run)?;
        let clips = module.negative_depth_clips() - clips_before;
        let merged_cells = merged_snapshots(
            &self.merged.download(&gpu)?,
            self.merged.counts(),
            case.state.n_elements,
        );
        let pass = &self.pass;
        let down = |b: &DeviceBuffer<f64>| b.to_host_vec(st);
        let q = self.q.download();
        let avg = pass.q_avg.download();
        let [h, hu, hv] = q.data;
        let [ah, ahu, ahv] = avg.data;
        let f = &pass.flux;
        let [r_u, r_v, r_hu, r_hv] = &pass.rates;
        let pooled = f.pooled.to_host_vec(st)?;
        let mut snapshots: Snapshots = vec![
            (
                "pass substeps/clips".into(),
                vec![n_bt as f64, clips as f64],
            ),
            ("pass levels".into(), levels),
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
        ];
        snapshots.extend(merged_cells);
        Ok(snapshots)
    }

    /// One step's pass on the device: the transport of the state, the
    /// substeps (and the levels), `Λ̄`, the pass and what follows it to the
    /// step's merged cells. Its substeps and every element's level (0 on
    /// global substeps).
    fn step(
        &mut self,
        module: &GpuSWE2D,
        case: &Case,
        run: Run,
    ) -> Result<(usize, Vec<f64>), Box<dyn std::error::Error>> {
        let Self {
            gpu,
            pass,
            merged,
            state,
            q,
            g,
            zero_g,
            bottom,
            d_sigma,
            layers,
            column,
            zero_faces,
            zero_subcells,
            zero_nodes,
            local,
            find_time,
        } = self;
        pass.to_transport(state, q)?;
        let mut local = (run.levels > 0)
            .then(|| local.get_or_insert_with(|| LocalPass3D::new(run.levels, run.scheme, q)));
        let dt_bt_max = match &mut local {
            Some(local) => {
                let cap = case.dt / MIN_BAROTROPIC_SUBSTEPS as f64;
                local.assign_levels(module, q, run.cfl, cap)
            }
            None => module.compute_dt_ssp(q, run.cfl, Some(run.scheme)),
        };
        let n_bt = case.substeps(dt_bt_max, run);
        let filter = BarotropicFilter::new(n_bt);
        pass.column_drag(d_sigma, layers, column)?;
        let inputs = DevicePass {
            filter: &filter,
            scheme: run.scheme,
            t: case.t,
            dt_bt: case.dt / n_bt as f64,
            forcing: if run.forcing { g } else { zero_g },
            bottom: run.bottom.then_some(&*bottom),
            column: run.layers.then_some(&*column),
            rivers: case.rivers.as_ref().filter(|_| run.rivers),
        };
        let levels = match local {
            Some(local) => {
                pass.run_local(module, inputs, local, q)?;
                let levels = local.stepper().levels();
                levels.iter().map(|&l| f64::from(l)).collect()
            }
            None => {
                pass.run(module, inputs, q)?;
                vec![0.0; case.state.n_elements]
            }
        };
        pass.post_process_average(module);
        pass.finish(state, case.dt)?;
        // The thin cells that lose more than their water in a stage, at the
        // 3D stages' forward-Euler steps (the driver's to set)
        let st = &gpu.stream;
        let euler: Vec<f64> = match run.find {
            Find::CoarseEuler => vec![case.dt; case.euler_dt.len()],
            Find::HalfEuler => case.euler_dt.iter().map(|&e| 0.5 * e).collect(),
            _ => case.euler_dt.clone(),
        };
        pass.flux.euler_dt.copy_from_host(st, &euler)?;
        let thin_depth = match run.find {
            Find::NoneThin => 0.0,
            _ => case.thin_depth,
        };
        let mut water = pass.step_water(state, case.dt, thin_depth);
        match run.find {
            Find::NoFaces => water.face = zero_faces,
            Find::NoSubcells => water.subcell = zero_subcells,
            Find::NoSources => water.source = zero_nodes,
            Find::NoRate => water.eta_rate = zero_nodes,
            _ => {}
        }
        st.synchronize()?;
        let t0 = Instant::now();
        merged.find(gpu, pass.statics(), &water)?;
        *find_time += t0.elapsed().as_secs_f64();
        Ok((n_bt, levels))
    }
}

/// One case on both sides with `base`, and (if `mutants`) the mutants:
/// `others` are the mutants of a changed 2D module, `finds` those of the
/// merged cells' search. Whether all are the same to the bit (and every
/// mutant caught).
#[allow(clippy::too_many_arguments)]
fn gate<BC: SWEBoundaryCondition2D + Sync>(
    gpu: &Arc<Gpu>,
    label: &str,
    physics: &SWEPhysics2D<BC>,
    case: &Case,
    base: Run,
    mutants: bool,
    others: &[(&str, &SWEPhysics2D<BC>)],
    finds: &[(&'static str, Find)],
) -> Result<bool, Box<dyn std::error::Error>> {
    let cpu = on_cpu(physics, case, base, &mut 0.0);
    let device = on_device(gpu, physics, case, base)?;
    let ne = case.state.n_elements;
    let field = |name: &str| {
        &cpu.iter()
            .find(|(n, _)| n == name)
            .expect("a snapshot of that name")
            .1
    };
    let subcells = field("DU_avg2 subcells")
        .chunks(NSUB_OF_P2)
        .filter(|s| s[0].is_finite())
        .count();
    let pooled = field("balance pooled").iter().filter(|&&p| p > 0.0).count();
    let sourced = field("means volume-sources")
        .iter()
        .filter(|&&s| s != 0.0)
        .count();
    let counts = field("pass substeps/clips");
    let levels = field("pass levels");
    let stepping = if base.levels == 0 {
        "global substeps".to_string()
    } else {
        let per_level: Vec<String> = (0..=base.levels)
            .map(|l| {
                let n = levels.iter().filter(|&&x| x == l as f64).count();
                format!("{n} on level {l}")
            })
            .collect();
        format!("coarse substeps, elements {}", per_level.join(", "))
    };
    println!(
        "  {} {stepping} of SSP-RK{} ({} clips), {subcells} of {ne} elements on the subcells \
         throughout, {pooled} balanced only as a whole, {sourced} nodes with a volume source",
        counts[0],
        match base.scheme {
            SspScheme::Rk3 => "3",
            SspScheme::Rk43 => "(4,3)",
        },
        counts[1],
    );
    let count = |name: &str| field(name).iter().filter(|&&x| x != 0.0).count();
    let cells = field("merged cells");
    println!(
        "  merged cells at forward-Euler steps of {}–{} s, thin below {} m: {} elements with a \
         group ({} cells), {} flushed ({} cells, {} inflows)",
        case.dt / 2.0,
        case.dt,
        case.thin_depth,
        count("merged grouped"),
        cells[0],
        count("merged flushed"),
        cells[1],
        count("merged inflow-rate"),
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
        if base.levels > 0 {
            runs.push(("global substeps", Run { levels: 0, ..base }));
        }
        // The merged cells' search
        for &(what, find) in finds {
            runs.push((what, Run { find, ..base }));
        }
        for (what, run) in runs {
            same &= report_mutant(what, &cpu, &on_device(gpu, physics, case, run)?);
        }
        for &(what, other) in others {
            same &= report_mutant(what, &cpu, &on_device(gpu, other, case, base)?);
        }
    }
    println!("  {}", gate::verdict(label, same));
    Ok(same)
}

/// Subcell interfaces of a P2 element.
const NSUB_OF_P2: usize = crate::three_d::NSUB;

/// The case runner's 2D module of a 3D run on `fx`: the wet/dry stage,
/// Coriolis on a β-plane, walls, no bed friction.
pub(crate) fn beach_physics(fx: &Fixture) -> SWEPhysics2D<Reflective2D> {
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
pub(crate) fn beach_rivers(fx: &Fixture) -> RiverSources {
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
        fx.min_column_depth,
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

/// Time the pass with `run` and what follows it on the CPU and the GPU at
/// `fx`'s size.
fn time(
    gpu: &Arc<Gpu>,
    fx: &Fixture,
    run: Run,
    iterations: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let physics = beach_physics(fx);
    let case = beach_case(fx, true);
    let ne = fx.n_elements();
    let mut n_bt = 0.0;
    let mut levels = Vec::new();
    let mut merged = Vec::new();
    let (mut cpu_time, mut cpu_find) = (0.0, 0.0);
    for it in 0..iterations + 1 {
        let t0 = Instant::now();
        let mut find = 0.0;
        let snapshots = on_cpu(&physics, &case, run, &mut find);
        if it >= 1 {
            cpu_time += t0.elapsed().as_secs_f64();
            cpu_find += find;
        }
        n_bt = snapshots[0].1[0];
        levels = snapshots[1].1.clone();
        merged = snapshots
            .iter()
            .find(|(name, _)| name == "merged cells")
            .expect("the merged cells")
            .1
            .clone();
    }
    let form = physics.device_form()?;
    let module = GpuSWE2D::from_form(gpu, &form)?;
    let mut device = Device::new(gpu, &module, physics.metric_form(), &case)?;
    let gpu_iterations = 3 * iterations;
    let (mut gpu_time, mut gpu_find) = (0.0, 0.0);
    let st = &gpu.stream;
    for it in 0..gpu_iterations + 1 {
        st.synchronize()?;
        let t0 = Instant::now();
        let find_before = device.find_time;
        device.step(&module, &case, run)?;
        st.synchronize()?;
        if it >= 1 {
            gpu_time += t0.elapsed().as_secs_f64();
            gpu_find += device.find_time - find_before;
        }
    }
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let (c, g) = (
        cpu_time / iterations as f64,
        gpu_time / gpu_iterations as f64,
    );
    let n_stages = run.scheme.n_stages() as f64;
    let substeps = n_bt * BarotropicFilter::new(n_bt as usize).reach();
    let scheme = match run.scheme {
        SspScheme::Rk3 => "SSP-RK3",
        SspScheme::Rk43 => "SSP-RK(4,3)",
    };
    if run.levels == 0 {
        let stages = n_stages * substeps;
        println!(
            "timing, beach: {ne} P{ORDER} elements, a pass of {n_bt} substeps of {scheme} \
             ({stages:.0} stages) at CFL {} with three rivers and what follows it",
            run.cfl
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
    } else {
        let finest = levels.iter().copied().fold(0.0, f64::max);
        let fine = levels.iter().filter(|&&l| l > 0.0).count();
        println!(
            "timing, beach: {ne} P{ORDER} elements, a pass of {n_bt} coarse substeps of {scheme} \
             at CFL {} on {} levels ({fine} elements finer than the coarse step, finest level \
             {finest}) with three rivers and what follows it",
            run.cfl, run.levels
        );
        println!(
            "  per pass: CPU ({threads} threads) {:.1} ms | GPU {:.1} ms ({:.1}×)",
            c * 1e3,
            g * 1e3,
            c / g,
        );
    }
    let (c, g) = (
        cpu_find / iterations as f64,
        gpu_find / gpu_iterations as f64,
    );
    println!(
        "  of which the merged cells ({} merged, {} flushed): CPU (one thread) {:.2} ms | GPU \
         {:.2} ms ({:.1}×)",
        merged[0],
        merged[1],
        c * 1e3,
        g * 1e3,
        c / g,
    );
    Ok(())
}

pub fn run(gpu: &Arc<Gpu>) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "== The barotropic pass, global substeps and local time steps: G, the drags, DU_avg2, the filter, the rivers, the band's sources; the step's merged cells (Phase 4.4c2–d) =="
    );
    // A whole pass and its merged cells: where a threshold decides (a cell
    // merged, an element pooled) contraction flips up to 0.5 % of a field's
    // flags; the mutants move thousands of values by ≥ 6e-2
    gate::start("pass3d", Bar::MIRRORED.with_outliers(0.01));
    let mut failed = Vec::new();
    let fx = Fixture::new(Shape::BEACH);
    let physics = beach_physics(&fx);
    let beach_runs = [
        ("beach, rivers, SSP-RK3", Run::global(SspScheme::Rk3), true),
        (
            "beach, rivers, SSP-RK(4,3)",
            Run::global(SspScheme::Rk43),
            false,
        ),
        (
            "beach, rivers, the case runner's pass on local time steps",
            Run::CASE_RUNNER,
            true,
        ),
        (
            "beach, rivers, SSP-RK3 on three levels",
            Run {
                levels: 3,
                ..Run::global(SspScheme::Rk3)
            },
            false,
        ),
    ];
    for (label, run, mutants) in beach_runs {
        println!("{label}: {} P{ORDER} elements", fx.n_elements());
        let case = beach_case(&fx, true);
        if !gate(
            gpu,
            label,
            &physics,
            &case,
            run,
            mutants,
            &[],
            &FIND_MUTANTS,
        )? {
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
    // Thin below the case runner's 1 m (`[three_d] thin_depth_m`)
    let case = Case::new(state, &domain.bed, None, 7200.0, 60.0, 1.0);
    let others = [("the band relaxing the momentum alone", &other)];
    for (label, run) in [
        (
            "nested channel, its band's sources, SSP-RK(4,3)",
            Run::global(SspScheme::Rk43),
        ),
        (
            "nested channel, its band's sources, the case runner's pass on local time steps",
            Run::CASE_RUNNER,
        ),
    ] {
        println!(
            "{label}: {} P{ORDER} elements, the band's level relaxed",
            domain.mesh.n_elements
        );
        if !gate(
            gpu,
            label,
            &physics,
            &case,
            run,
            true,
            &others,
            &FIND_MUTANTS[..4],
        )? {
            failed.push(label);
        }
    }

    let large = Fixture::new(Shape {
        nx: 320,
        ny: 300,
        jitter: 0.15,
        stretched: true,
        ..Shape::BEACH
    });
    time(gpu, &large, Run::global(SspScheme::Rk3), 3)?;
    time(gpu, &large, Run::CASE_RUNNER, 3)?;
    if !failed.is_empty() {
        println!("(failed: {})", failed.join("; "));
    }
    Ok(())
}

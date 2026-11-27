//! The device's own judges (P2.7, the rule of 2026-11-25 in `notes/gpu.md`):
//! what the GPU's numerics must reach whatever the CPU's bits are. The other
//! gates hold the device to the CPU, bit for bit or within a tolerance; these
//! hold it to the scheme's properties, as dgcom's tests hold the CPU
//! (`tests/convergence_test.rs`, `tests/curvilinear_3d_test.rs`,
//! `src/simulation/simulation_3d.rs`), so a kernel may leave the CPU's loop
//! order and still be judged. GitHub's runners have no GPU: `dgcom-gpu
//! judges` (also in `all`, and in `time.sh`) is where they run.
//!
//! 2D, dgcom's own `Simulation` with SSP-RK(4,3) over the device's module:
//! - **Convergence.** A steady vortex in gradient-wind balance over a round
//!   hill, an exact solution of the nonlinear equations with the bed and the
//!   Coriolis force and no made-up source (a device has a closed set of
//!   sources, so the CPU's manufactured solution cannot run there), on
//!   doubly periodic meshes distorted by a smooth field: order N + 1 = 3 of
//!   the depth's L2 error for both split forms.
//! - **Lake at rest with a shoreline**, on a jittered walled beach: no flow.
//! - **Mass** to round-off and **positivity** on the drying beach, with
//!   films and dry nodes.
//!
//! 3D, dgcom's own `ModeSplitIntegrator` over the device's model
//! (`Model3D`), the case runner's terms on the gates' wet/dry beach:
//! - **Rest about the reference**: the pycnocline at rest over the sloping
//!   bed, across the σ-levels and the shoreline's thin columns, and over a
//!   doubly periodic bed of bumps on general quadrilaterals (the seamount
//!   test): no flow.
//! - **Constancy**: uniform T and S stay uniform under a sheared flow, a
//!   wind, films, merged cells and rivers of the same water.
//! - **Conservation**: the inventories of T and S of the stratified water.
//!
//! Both 3D judges run on global 3D steps and on local ones of two levels,
//! to the same bars.
//!
//! Each judge has a mutant it must catch: the device given another input
//! (an operator table off by a part in a thousand, another bed, no wet/dry
//! correction, no reference), which the judge must fail.

use std::error::Error;
use std::sync::Arc;

use dgcom::SWEState2D;
use dgcom::equations::ShallowWater2D;
use dgcom::mesh::{Bathymetry2D, Mesh2D};
use dgcom::operators::{DGOperators2D, GeometricFactors2D};
use dgcom::physics::vertical_mixing::Forcing;
use dgcom::physics::{GlsMixing, JerlovWater};
use dgcom::simulation::{RunModel3D, Simulation};
use dgcom::solver::SWESolution2D;
use dgcom::solver::state::Solution3D;
use dgcom::source::{CoriolisSource2D, River, RiverSeries, RiverSources};
use dgcom::time::{ModeSplitPhysics, SSPRK43, SspScheme};
use dgcom::types::ElementIndex;

use crate::hooks::State3D;
use crate::model3d::{Model3D, Work3D};
use crate::phase0::{Case, DX};
use crate::stage3d::start_state;
use crate::step3d::{CFL as CFL_3D, DT_MAX, Physics, Run, Terms, physics_with};
use crate::swe2d::{DeviceState, Formulation, Gpu, GpuSWE2D, NN, ORDER};
use crate::transport3d::{Fixture, LEVELS, Shape};

const G: f64 = 9.81;
/// The wet/dry depth of the case runner's 2D module (`WetDryConfig`'s).
const H_DRY: f64 = 1e-3;
/// The CFL number of the 2D runs (the case runner's default with
/// SSP-RK(4,3) is 0.5).
const CFL: f64 = 0.5;

/// What the judges found: every line printed, the failures kept.
#[derive(Default)]
pub(crate) struct Tally {
    failed: Vec<String>,
    judged: usize,
    mutants: usize,
}

impl Tally {
    /// A judge's line: `measured` against its bar, and whether it holds.
    pub(crate) fn judge(&mut self, name: &str, measured: &str, pass: bool) {
        self.judged += 1;
        println!(
            "  {name}: {measured}: {}",
            if pass { "ok" } else { "FAILED" }
        );
        if !pass {
            self.failed.push(name.to_string());
        }
    }

    /// A mutant's line: what the judge measured of it, and whether the judge
    /// fails it (`caught`).
    pub(crate) fn mutant(&mut self, name: &str, measured: &str, caught: bool) {
        self.mutants += 1;
        println!(
            "    mutant ({name}): {measured}{}",
            if caught { ", caught" } else { ": NOT CAUGHT" }
        );
        if !caught {
            self.failed.push(format!("mutant {name}"));
        }
    }
}

/// What a mutant gives the device in place of the case's own.
#[derive(Default)]
struct Device2D {
    /// Other operator tables
    ops: Option<Arc<DGOperators2D>>,
    /// Another bed
    bed: Option<Arc<Bathymetry2D>>,
    /// The depth under which an element takes the subcells (infinite: every
    /// element; zero: none)
    subcell_depth: Option<f64>,
    /// No wet/dry correction after a stage
    no_correction: bool,
}

/// A 2D run on the device: its steps, end state and negative-depth clips.
struct Run2D {
    steps: usize,
    state: SWESolution2D,
    clips: usize,
}

/// `case` from its state to `t_end` on the device as `device` changes it,
/// with dgcom's `Simulation` and SSP-RK(4,3).
fn run_2d(
    gpu: &Arc<Gpu>,
    case: &Case,
    device: &Device2D,
    t_end: f64,
) -> Result<Run2D, Box<dyn Error>> {
    let twin = case.cpu_physics();
    let mut form = twin.device_form()?;
    if let Some(ops) = &device.ops {
        form.ops = ops;
    }
    if let Some(bed) = &device.bed {
        form.bathymetry = Some(bed);
    }
    if let Some(depth) = device.subcell_depth {
        let (_, h_dry) = form.subcells.ok_or("the subcells need the WetDry form")?;
        form.subcells = Some((depth, h_dry));
    }
    if device.no_correction {
        form.wet_dry = None;
    }
    let physics = GpuSWE2D::from_form(gpu, &form)?;
    let sim = Simulation::new(physics, SSPRK43).with_cfl(CFL);
    let mut state = DeviceState::upload(gpu, &case.q);
    let result = sim.run(&mut state, 0.0, t_end);
    gpu.stream.synchronize()?;
    if !result.success {
        return Err(format!("the device's run failed: {:?}", result.error).into());
    }
    Ok(Run2D {
        steps: result.n_steps,
        state: state.download(),
        clips: sim.physics().negative_depth_clips(),
    })
}

/// A steady vortex in gradient-wind balance over a round hill, in a doubly
/// periodic square of side `L`, of compact support: with `s = r / r₀` from
/// the centre and `c(s) = 1 − s²/S²` inside `s < S` (zero beyond),
/// the azimuthal speed `v_θ = a s c⁵`, the surface
/// `η = −(a² S²/22g) c¹¹ − (f a r₀ S²/12g) c⁶` (so that
/// `v_θ²/r + f v_θ = g ∂η/∂r`) and the bed `−20 m + 6 m c⁵`. The depth
/// depends on `r` alone and the flow is along its circles, so the transport
/// has no divergence: the state is steady for the nonlinear equations, with
/// the bed and the Coriolis force and no made-up source.
///
/// It fills the square (`S r₀ = 0.47 L`), so that 16 elements a side
/// resolve it and the orders are the schemes' asymptotic ones: a Gaussian
/// vortex narrow enough to vanish at the edge gave 2.69 for the split form
/// and 2.59 for the second-order subcells at 32–128 elements a side, on the
/// CPU's kernels as on the device's, and told nothing apart. Its fields are
/// four times continuously differentiable at its edge and exactly at rest
/// beyond it.
struct Vortex {
    side: f64,
    r0: f64,
    /// `S`: the vortex's edge in units of `r₀`
    edge: f64,
    /// `a` (m/s): the speed's scale (its largest is 0.56 a, at s = S/√11)
    speed: f64,
    f: f64,
}

impl Vortex {
    const HERE: Self = Self {
        side: 16_000.0,
        r0: 2_500.0,
        edge: 3.0,
        speed: 2.7,
        f: 1.3e-4,
    };

    /// `c = 1 − s²/S²` at (x, y), zero outside the vortex.
    fn c(&self, x: f64, y: f64) -> f64 {
        let centre = 0.5 * self.side;
        let s2 = ((x - centre) * (x - centre) + (y - centre) * (y - centre)) / (self.r0 * self.r0);
        (1.0 - s2 / (self.edge * self.edge)).max(0.0)
    }

    fn eta(&self, x: f64, y: f64) -> f64 {
        let c = self.c(x, y);
        let s2 = self.edge * self.edge;
        -self.speed * self.speed * s2 / (22.0 * G) * c.powi(11)
            - self.f * self.speed * self.r0 * s2 / (12.0 * G) * c.powi(6)
    }

    fn bed(&self, x: f64, y: f64) -> f64 {
        -20.0 + 6.0 * self.c(x, y).powi(5)
    }

    fn depth(&self, x: f64, y: f64) -> f64 {
        self.eta(x, y) - self.bed(x, y)
    }

    fn velocity(&self, x: f64, y: f64) -> (f64, f64) {
        let centre = 0.5 * self.side;
        let rate = self.speed / self.r0 * self.c(x, y).powi(5);
        (-rate * (y - centre), rate * (x - centre))
    }

    /// The vortex on an `n × n` mesh of the square, its vertices moved by
    /// the smooth periodic field of `tests/convergence_test.rs`'s
    /// `periodic_mesh` (general quadrilaterals, the metric at every node).
    fn case(&self, n: usize, formulation: Formulation) -> Case {
        let l = self.side;
        let mut mesh = Mesh2D::uniform_periodic(0.0, l, 0.0, l, n, n);
        let tau = std::f64::consts::TAU / l;
        let a = 0.04 * l;
        for v in &mut mesh.vertices {
            let [x, y] = *v;
            let (sx, sy) = ((tau * x).sin(), (tau * y).sin());
            *v = [
                x + a * sx * sy + 0.5 * a * (tau * y).sin(),
                y - 0.7 * a * sx * sy + 0.4 * a * (tau * x).sin(),
            ];
        }
        let ops = DGOperators2D::new(ORDER);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, y| self.bed(x, y));
        let mut q = SWESolution2D::new(mesh.n_elements, NN);
        for k in 0..mesh.n_elements {
            let ki = ElementIndex::new(k);
            for i in 0..NN {
                let [x, y] = mesh.reference_to_physical(ki, ops.nodes_r[i], ops.nodes_s[i]);
                // The depth over the nodal bed, as the kernels take the surface
                let h = self.eta(x, y) - bathymetry.get(ki, i);
                let (u, v) = self.velocity(x, y);
                q.set_state(ki, i, SWEState2D::new(h, h * u, h * v));
            }
        }
        Case {
            mesh: Arc::new(mesh),
            ops: Arc::new(ops),
            geom: Arc::new(geom),
            bathymetry: Arc::new(bathymetry),
            equation: ShallowWater2D::new(G),
            q,
            formulation,
            manning: None,
            coriolis: CoriolisSource2D::f_plane(self.f),
        }
    }

    /// The depth's L2 error (rms over the square, m) of `state` on `case`.
    fn error(&self, case: &Case, state: &SWESolution2D) -> f64 {
        state.l2_error_depth(&case.mesh, &case.ops, &case.geom, |x, y| self.depth(x, y)) / self.side
    }
}

/// The vortex's errors on a sequence of meshes, the orders between them and
/// the runs' steps.
type Orders = (Vec<f64>, Vec<f64>, Vec<usize>);

/// The vortex's errors on `resolutions` after `t_end`, on the device as
/// `device` changes it, and the orders between them.
fn vortex_orders(
    gpu: &Arc<Gpu>,
    formulation: Formulation,
    device: impl Fn(&Case) -> Device2D,
    resolutions: &[usize],
    t_end: f64,
) -> Result<Orders, Box<dyn Error>> {
    let vortex = Vortex::HERE;
    let mut errors = Vec::new();
    let mut steps = Vec::new();
    for &n in resolutions {
        let case = vortex.case(n, formulation);
        let run = run_2d(gpu, &case, &device(&case), t_end)?;
        errors.push(vortex.error(&case, &run.state));
        steps.push(run.steps);
    }
    let orders = errors.windows(2).map(|e| (e[0] / e[1]).log2()).collect();
    Ok((errors, orders, steps))
}

/// The vortex's errors on `resolutions` on the CPU (the kernels real runs
/// use), for the reader: what dgcom's own scheme reaches on this fixture.
fn vortex_orders_cpu(formulation: Formulation, resolutions: &[usize], t_end: f64) -> Vec<f64> {
    let vortex = Vortex::HERE;
    let errors: Vec<f64> = resolutions
        .iter()
        .map(|&n| {
            let case = vortex.case(n, formulation);
            let sim = Simulation::new(case.cpu_physics(), SSPRK43).with_cfl(CFL);
            let mut q = case.q.clone();
            let result = sim.run(&mut q, 0.0, t_end);
            assert!(result.success, "the CPU's run: {result:?}");
            vortex.error(&case, &q)
        })
        .collect();
    errors.windows(2).map(|e| (e[0] / e[1]).log2()).collect()
}

/// The order the split forms must exceed at P2 (N + 1 = 3; dgcom's tests
/// ask 2.5 of the CPU on their manufactured solution and measure 2.95). The
/// line prints the CPU's kernels' orders on this vortex beside the
/// device's; the second-order subcells must fall under this bar.
const ORDER_BAR: f64 = 2.5;
/// The order the subcells must exceed (second order; dgcom's tests ask 1.7).
const SUBCELL_ORDER_BAR: f64 = 1.7;

fn convergence(gpu: &Arc<Gpu>, tally: &mut Tally) -> Result<(), Box<dyn Error>> {
    println!(
        "2D convergence: a steady vortex of compact support over a hill, distorted periodic \
         meshes, P{ORDER}, to 600 s (the depth's rms error)"
    );
    let resolutions = [32, 64, 128];
    let t_end = 600.0;
    let wet_dry = Formulation::WetDry {
        subcell_depth: H_DRY,
        h_dry: H_DRY,
    };
    let line = |errors: &[f64], orders: &[f64], steps: &[usize]| {
        format!(
            "errors {:.3e}, {:.3e}, {:.3e} m at n = 32, 64, 128 ({}, {}, {} steps), orders \
             {:.2}, {:.2}",
            errors[0], errors[1], errors[2], steps[0], steps[1], steps[2], orders[0], orders[1]
        )
    };
    for (name, formulation) in [
        ("EntropyStable", Formulation::EntropyStable),
        ("WetDry, fully wet", wet_dry),
    ] {
        let (errors, orders, steps) = vortex_orders(
            gpu,
            formulation,
            |_| Device2D::default(),
            &resolutions,
            t_end,
        )?;
        let cpu = vortex_orders_cpu(formulation, &resolutions, t_end);
        tally.judge(
            name,
            &format!(
                "{} (the CPU's kernels: {:.2}, {:.2}); the last must exceed {ORDER_BAR}",
                line(&errors, &orders, &steps),
                cpu[0],
                cpu[1]
            ),
            orders[1] > ORDER_BAR,
        );
    }
    // The 1D derivative matrix off by a part in a thousand in one entry
    let (errors, orders, steps) = vortex_orders(
        gpu,
        Formulation::EntropyStable,
        |case| {
            let mut ops = (*case.ops).clone();
            ops.dr_1d_row_major[1] *= 1.001;
            Device2D {
                ops: Some(Arc::new(ops)),
                ..Device2D::default()
            }
        },
        &resolutions,
        t_end,
    )?;
    tally.mutant(
        "an entry of the device's D off by 1e-3",
        &line(&errors, &orders, &steps),
        orders[1] <= ORDER_BAR,
    );
    // The second-order subcells in every element: their own bar
    let (errors, orders, steps) = vortex_orders(
        gpu,
        wet_dry,
        |_| Device2D {
            subcell_depth: Some(f64::INFINITY),
            ..Device2D::default()
        },
        &resolutions,
        t_end,
    )?;
    tally.judge(
        "WetDry, the subcells in every element",
        &format!(
            "{}; the last must exceed {SUBCELL_ORDER_BAR}",
            line(&errors, &orders, &steps)
        ),
        orders[1] > SUBCELL_ORDER_BAR,
    );
    // ... and under the split form's bar: the judge tells second order from
    // third
    tally.mutant(
        "the subcells in every element, judged as the split form",
        &format!("order {:.2} against {ORDER_BAR}", orders[1]),
        orders[1] <= ORDER_BAR,
    );
    Ok(())
}

/// The beach of the gates (`Case::beach`: a jittered walled mesh, the bed
/// from −8 m to +2 m with bumps, Manning friction, a β-plane, `WetDry`) with
/// the water at rest at mean sea level and the land dry.
fn lake(nx: usize, ny: usize) -> Case {
    let mut case = Case::beach(nx, ny, 0.2, false);
    for k in 0..case.mesh.n_elements {
        let ki = ElementIndex::new(k);
        for i in 0..NN {
            let h = (-case.bathymetry.get(ki, i)).max(0.0);
            case.q.set_state(ki, i, SWEState2D::new(h, 0.0, 0.0));
        }
    }
    case
}

/// The largest speed (m/s) at the nodes deeper than `H_DRY`, the largest
/// transport (m²/s) anywhere and the largest surface elevation (m) at those
/// nodes.
fn motion(case: &Case, state: &SWESolution2D) -> (f64, f64, f64) {
    let (mut speed, mut transport, mut level) = (0.0f64, 0.0f64, 0.0f64);
    for k in 0..case.mesh.n_elements {
        let ki = ElementIndex::new(k);
        for i in 0..NN {
            let q = state.get_state(ki, i);
            let flow = q.hu.hypot(q.hv);
            transport = transport.max(flow);
            if q.h > H_DRY {
                speed = speed.max(flow / q.h);
                level = level.max((q.h + case.bathymetry.get(ki, i)).abs());
            }
        }
    }
    (speed, transport, level)
}

/// The sum of `terms`, compensated (Kahan), so that the sum's own round-off
/// is not judged.
fn compensated(terms: impl Iterator<Item = f64>) -> f64 {
    let (mut sum, mut carry) = (0.0f64, 0.0f64);
    for term in terms {
        let y = term - carry;
        let t = sum + y;
        carry = (t - sum) - y;
        sum = t;
    }
    sum
}

/// The water's volume (m³) of `state` on `case`: the nodal masses' sum.
fn volume(case: &Case, state: &SWESolution2D) -> f64 {
    compensated((0..case.mesh.n_elements * NN).map(|flat| {
        let (k, i) = (flat / NN, flat % NN);
        case.geom.mass[case.geom.node_index(k, i)] * state.get_state(ElementIndex::new(k), i).h
    }))
}

/// The smallest depth of `state` (m).
fn least_depth(state: &SWESolution2D) -> f64 {
    state.data[0].iter().fold(f64::INFINITY, |m, &h| m.min(h))
}

/// `ops` with the lift of face 1's middle node into its own node larger by
/// `1 + by`: the two sides of an edge then exchange fluxes that differ.
fn lift_off(ops: &DGOperators2D, by: f64) -> Arc<DGOperators2D> {
    let mut ops = ops.clone();
    let node = ops.face_nodes[1][1];
    ops.lift_row_major[1][node * ops.n_face_nodes + 1] *= 1.0 + by;
    Arc::new(ops)
}

/// The largest speed a lake at rest may show (m/s): dgcom's CPU kernels
/// keep it at zero exactly; multiply–add contraction leaves round-off
/// (1e-13 after 800 steps).
const REST_SPEED_BAR: f64 = 1e-12;
/// The largest relative change of the water's volume (2e-14 after 400
/// steps).
const VOLUME_BAR: f64 = 1e-12;

fn lake_at_rest(gpu: &Arc<Gpu>, tally: &mut Tally) -> Result<(), Box<dyn Error>> {
    let case = lake(36, 20);
    let t_end = 600.0;
    let shore = (0..case.mesh.n_elements)
        .filter(|&k| {
            let h = &case.q.data[0][k * NN..(k + 1) * NN];
            h.contains(&0.0) && h.iter().any(|&h| h > 0.0)
        })
        .count();
    println!(
        "2D lake at rest with a shoreline: {} P{ORDER} elements, jittered, {shore} of them across \
         the shoreline, to {t_end} s",
        case.mesh.n_elements
    );
    let measure = |device: &Device2D| -> Result<(usize, f64, f64, f64), Box<dyn Error>> {
        let run = run_2d(gpu, &case, device, t_end)?;
        let (speed, transport, level) = motion(&case, &run.state);
        Ok((run.steps, speed, transport, level))
    };
    let (steps, speed, transport, level) = measure(&Device2D::default())?;
    tally.judge(
        "at rest",
        &format!(
            "after {steps} steps the largest speed {speed:.2e} m/s (transport {transport:.2e} \
             m²/s, surface {level:.2e} m); the speed must stay under {REST_SPEED_BAR:.0e}"
        ),
        speed <= REST_SPEED_BAR && level <= REST_SPEED_BAR,
    );
    // A bed on the device 1 mm higher at one node of every seventh element
    let mut bumped = (*case.bathymetry).clone();
    for k in (0..case.mesh.n_elements).step_by(7) {
        bumped.data[k * NN + 4] += 1e-3;
    }
    let (_, speed, ..) = measure(&Device2D {
        bed: Some(Arc::new(bumped)),
        ..Device2D::default()
    })?;
    tally.mutant(
        "the device's bed 1 mm higher at a node of every seventh element",
        &format!("the largest speed {speed:.2e} m/s"),
        speed > REST_SPEED_BAR,
    );
    // An entry of the 1D derivative matrix off by a part in a million: a
    // level surface then has a slope
    let mut ops = (*case.ops).clone();
    ops.dr_1d_row_major[1] *= 1.0 + 1e-6;
    let (_, speed, ..) = measure(&Device2D {
        ops: Some(Arc::new(ops)),
        ..Device2D::default()
    })?;
    tally.mutant(
        "an entry of the device's D off by 1e-6",
        &format!("the largest speed {speed:.2e} m/s"),
        speed > REST_SPEED_BAR,
    );
    Ok(())
}

fn mass_and_positivity(gpu: &Arc<Gpu>, tally: &mut Tally) -> Result<(), Box<dyn Error>> {
    println!(
        "2D mass and positivity: the drying beach (720 P{ORDER} elements, jittered, flow towards \
         the shore), to 300 s"
    );
    let t_end = 300.0;
    for (name, films) in [("shoreline", false), ("films and dry nodes", true)] {
        let case = Case::beach(36, 20, 0.2, films);
        let before = volume(&case, &case.q);
        let run = run_2d(gpu, &case, &Device2D::default(), t_end)?;
        let change = (volume(&case, &run.state) - before) / before;
        let least = least_depth(&run.state);
        tally.judge(
            &format!("{name}, mass"),
            &format!(
                "after {} steps the volume changed by {change:.2e} of itself ({} clips); it \
                 must stay within {VOLUME_BAR:.0e} with no clip",
                run.steps, run.clips
            ),
            change.abs() <= VOLUME_BAR && run.clips == 0,
        );
        tally.judge(
            &format!("{name}, positivity"),
            &format!("the smallest depth {least:.2e} m; none may be negative"),
            least >= 0.0,
        );
        if films {
            continue;
        }
        let run = run_2d(
            gpu,
            &case,
            &Device2D {
                ops: Some(lift_off(&case.ops, 1e-3)),
                ..Device2D::default()
            },
            t_end,
        )?;
        let change = (volume(&case, &run.state) - before) / before;
        tally.mutant(
            "a face node's lift off by 1e-3",
            &format!("the volume changed by {change:.2e} of itself"),
            change.abs() > VOLUME_BAR,
        );
        // Neither the subcells nor the wet/dry correction at the shoreline:
        // the split form over its dry nodes leaves depths negative (the
        // beach keeps them positive with either alone), or the run stops
        let run = run_2d(
            gpu,
            &case,
            &Device2D {
                subcell_depth: Some(0.0),
                no_correction: true,
                ..Device2D::default()
            },
            t_end,
        );
        let (measured, caught) = match run {
            Ok(run) => {
                let least = least_depth(&run.state);
                (
                    format!(
                        "{} steps, {} clips, the smallest depth {least:.2e} m",
                        run.steps, run.clips
                    ),
                    run.clips > 0 || least.is_nan() || least < 0.0,
                )
            }
            Err(e) => (format!("the run stopped ({e})"), true),
        };
        tally.mutant(
            "neither subcells nor the wet/dry correction at the shoreline",
            &measured,
            caught,
        );
    }
    Ok(())
}

// =============================================================================
// 3D
// =============================================================================

/// The 3D judges' fixture: the gates' walled beach (720 P2 elements × 20
/// levels, the bed from −8 m to +2 m with bumps), jittered by 0.2 on ROMS's
/// Vstretching 4, with its cubic reference stratification.
fn beach_3d() -> Fixture {
    Fixture::new(Shape {
        jitter: 0.2,
        stretched: true,
        ..Shape::BEACH
    })
}

/// The case runner's terms without the surface fluxes, the wind field and
/// the rivers (a judge adds what it needs).
const QUIET: Terms = Terms {
    wind: false,
    rivers: false,
    fluxes: None,
    ..Terms::CASE_RUNNER
};

const NO_STRESS: Forcing = Forcing {
    surface_stress: [0.0, 0.0],
    bottom_stress: [0.0, 0.0],
    surface_heat_flux: 0.0,
};

/// `steps` steps of the case runner's integrator (`Run::CASE_RUNNER`) from
/// `start` on the device, as a run takes them (`Simulation3D`): on global 3D
/// steps (`levels` 0) each of the CPU's own bound of the device's state, on
/// local ones the coarse step of the levels the device's own element steps
/// give (at most the case runner's cap either way); then the end of a step
/// (`RunModel3D::post_process`: the tracer limiter, the density, `w`).
/// `each` sees the state after every step. The time reached, and the finest
/// level a step used.
fn run_3d(
    gpu: &Arc<Gpu>,
    physics: &Physics,
    start: &Solution3D,
    steps: usize,
    levels: usize,
    mut each: impl FnMut(&Solution3D),
) -> Result<(f64, usize), Box<dyn Error>> {
    let mut host = start.clone();
    physics.update_density(&mut host);
    let mut model = Model3D::new(gpu, physics)?;
    model.start_from(&host);
    let mut state = State3D::upload(gpu, &host)?;
    let run = Run {
        baroclinic_levels: levels,
        ..Run::CASE_RUNNER
    };
    let mut integrator = run.integrator().on::<Work3D>();
    let (mut t, mut finest) = (0.0, 0);
    for _ in 0..steps {
        model.update_density(&mut state);
        let dt = if levels == 0 {
            physics
                .compute_dt_ssp(&host, CFL_3D, SspScheme::Rk43)
                .min(DT_MAX)
        } else {
            let mut element_dt = vec![0.0; physics.mesh.n_elements];
            let work = integrator.work_for(&model, &mut state);
            model.element_steps(&state, work, CFL_3D, SspScheme::Rk43, &mut element_dt);
            let stencils = ModeSplitPhysics::stage_stencils(physics);
            integrator.assign_baroclinic_levels(
                &element_dt,
                &physics.mesh,
                &stencils,
                Some(DT_MAX),
                None,
            )
        };
        integrator.step(&mut state, &model, dt, t);
        if let Some(stats) = integrator.baroclinic_multirate_stats() {
            finest = finest.max(stats.finest_level);
        }
        let work = integrator.work_mut().expect("a step made the work");
        RunModel3D::post_process(&model, &mut state, work);
        t += dt;
        state.download_into(gpu, &mut host)?;
        each(&host);
    }
    Ok((t, finest))
}

/// [`run_3d`]'s steps on the CPU (its own element steps on local 3D
/// steps): the CPU's kernels on a judge's fixture, for the line it prints.
fn run_3d_cpu(
    physics: &Physics,
    start: &Solution3D,
    steps: usize,
    levels: usize,
    mut each: impl FnMut(&Solution3D),
) {
    let mut state = start.clone();
    let run = Run {
        baroclinic_levels: levels,
        ..Run::CASE_RUNNER
    };
    let mut integrator = run.integrator();
    let mut t = 0.0;
    let mut element_dt = vec![0.0; physics.mesh.n_elements];
    for _ in 0..steps {
        physics.update_density(&mut state);
        let dt = if levels == 0 {
            physics
                .compute_dt_ssp(&state, CFL_3D, SspScheme::Rk43)
                .min(DT_MAX)
        } else {
            physics.element_steps_into(&state, CFL_3D, SspScheme::Rk43, &mut element_dt);
            let stencils = ModeSplitPhysics::stage_stencils(physics);
            integrator.assign_baroclinic_levels(
                &element_dt,
                &physics.mesh,
                &stencils,
                Some(DT_MAX),
                None,
            )
        };
        integrator.step(&mut state, physics, dt, t);
        physics.post_process(&mut state);
        t += dt;
        each(&state);
    }
}

/// The largest layer speed over `steps` global steps from `start` on the
/// CPU (m/s).
fn rest_on_cpu(physics: &Physics, start: &Solution3D, steps: usize) -> f64 {
    let mut speed = 0.0f64;
    run_3d_cpu(physics, start, steps, 0, |s| {
        let now = largest_speed(s);
        speed = if now.is_nan() { now } else { speed.max(now) };
    });
    speed
}

/// The largest layer speed of `state` (m/s), NaN if a value is not finite.
fn largest_speed(state: &Solution3D) -> f64 {
    let mut speed = 0.0f64;
    for (u, v) in state.u.iter().zip(&state.v) {
        let s = u.hypot(*v);
        if !s.is_finite() {
            return f64::NAN;
        }
        speed = speed.max(s);
    }
    speed
}

/// `fx`'s water at rest at mean sea level, the land dry, its T and S the
/// reference stratification's at every layer centre's height, no
/// turbulence yet (as a case starts).
fn at_rest(fx: &Fixture) -> Solution3D {
    let profile = fx.profile.as_ref().expect("the fixture has a reference");
    let mut state = fx.state.clone();
    let sigma = fx.sigma.sigma_rho();
    for idx in 0..state.eta.data.len() {
        let bed = fx.bathymetry.data[idx];
        let eta = bed.max(0.0);
        state.eta.data[idx] = eta;
        state.ubar.data[idx] = 0.0;
        state.vbar.data[idx] = 0.0;
        for (l, &s) in sigma.iter().enumerate() {
            let at = idx * LEVELS + l;
            let (temp, salt) = profile.at(eta + s * (eta - bed));
            state.temp[at] = temp;
            state.salt[at] = salt;
            state.u[at] = 0.0;
            state.v[at] = 0.0;
        }
    }
    state.w.iter_mut().for_each(|w| *w = 0.0);
    state.tke.clear();
    state.gls.clear();
    state
}

/// `fx`'s water at rest as [`at_rest`], its temperature linear in height
/// (0.25 °C/m from 12 °C at mean sea level) and its salinity uniform: with
/// the linear equation of state a density linear in z, whose pressure
/// gradient along sloping levels the σ-pairs cancel without a reference.
fn at_rest_linear(fx: &Fixture) -> Solution3D {
    let mut state = at_rest(fx);
    let sigma = fx.sigma.sigma_rho();
    for idx in 0..state.eta.data.len() {
        let (eta, bed) = (state.eta.data[idx], fx.bathymetry.data[idx]);
        for (l, &s) in sigma.iter().enumerate() {
            let at = idx * LEVELS + l;
            state.temp[at] = 12.0 + 0.25 * (eta + s * (eta - bed));
            state.salt[at] = 33.0;
        }
    }
    state
}

/// The largest layer speed water at rest about its reference may reach
/// (m/s): dgcom's `stratified_lake_with_dry_land_stays_at_rest` asks this
/// of the CPU and measures 1.3e-13.
const REST_3D_BAR: f64 = 1e-10;

fn rest_3d(
    gpu: &Arc<Gpu>,
    tally: &mut Tally,
    fx: &Fixture,
    what: &str,
    quiet: Terms,
) -> Result<(), Box<dyn Error>> {
    let steps = 40;
    println!(
        "3D rest about the reference: {what}, {} P{ORDER} elements × {LEVELS} levels, jittered \
         and stretched, the pycnocline crossing its σ-levels, {steps} steps",
        fx.mesh.n_elements
    );
    // No mixing of the stratification (a closure's background diffusion of
    // it over the sloping bed is a flow the water at rest would be right to
    // start): no background, and turbulence at a floor that gives none
    let still = || {
        GlsMixing::k_epsilon()
            .with_background(0.0, 0.0)
            .with_minimum(1e-30, 1e-30)
    };
    let start = at_rest(fx);
    let run = |physics: &Physics, start: &Solution3D| -> Result<(f64, f64), Box<dyn Error>> {
        let mut speed = 0.0f64;
        let (t, _) = run_3d(gpu, physics, start, steps, 0, |s| {
            let now = largest_speed(s);
            speed = if now.is_nan() { now } else { speed.max(now) };
        })?;
        Ok((speed, t))
    };
    let measure = |terms: Terms| -> Result<(f64, f64), Box<dyn Error>> {
        run(&physics_with(fx, terms, NO_STRESS, still(), None), &start)
    };
    let (speed, t) = measure(quiet)?;
    tally.judge(
        "at rest",
        &format!(
            "over {steps} steps ({t:.0} s) the largest layer speed {speed:.2e} m/s; it must \
             stay under {REST_3D_BAR:.0e}"
        ),
        speed <= REST_3D_BAR,
    );
    let (speed, _) = measure(Terms {
        reference: false,
        ..quiet
    })?;
    tally.mutant(
        "no reference on the device: the pressure gradient of the whole density",
        &format!("the largest layer speed {speed:.2e} m/s"),
        speed.is_nan() || speed > REST_3D_BAR,
    );
    // Without a reference, a density linear in height: the pressure
    // gradient's kernels on pressures of the whole column, which must cancel
    // pair by pair (about the reference above they work on an anomaly of
    // zero). dgcom's `stratified_lake_with_dry_land_stays_at_rest` and the
    // seamount test ask this of the CPU.
    let linear = at_rest_linear(fx);
    // (and without the Kuzmin limiter, which with no reference to take
    // the departure from clips the stratification along the sloping levels)
    let bare = Terms {
        reference: false,
        limiter: false,
        ..quiet
    };
    let physics = physics_with(fx, bare, NO_STRESS, still(), None);
    let (speed, t) = run(&physics, &linear)?;
    let on_cpu = rest_on_cpu(&physics, &linear, steps);
    tally.judge(
        "at rest on a density linear in height, no reference",
        &format!(
            "over {steps} steps ({t:.0} s) the largest layer speed {speed:.2e} m/s (the CPU's \
             kernels: {on_cpu:.2e}); it must stay under {REST_3D_BAR:.0e}"
        ),
        speed <= REST_3D_BAR,
    );
    // The pairs cancel one by one at rest, so no error in D or in the lift
    // shows here (an entry of D off by 1e-3 leaves 6e-13 m/s). What shows
    // is the stratification no longer linear: the Kuzmin limiter on, which
    // with no reference clips it along the sloping levels
    let clipped = physics_with(
        fx,
        Terms {
            limiter: true,
            ..bare
        },
        NO_STRESS,
        still(),
        None,
    );
    let (speed, _) = run(&clipped, &linear)?;
    tally.mutant(
        "the Kuzmin limiter on, about no reference",
        &format!("the largest layer speed {speed:.2e} m/s"),
        speed.is_nan() || speed > REST_3D_BAR,
    );
    Ok(())
}

/// The beach's three rivers (`pass3d`'s places and discharges) with water
/// of `temperature` and `salinity`.
fn rivers_of(fx: &Fixture, temperature: f64, salinity: f64) -> RiverSources {
    let (lx, ly) = (36.0 * DX, 20.0 * DX);
    let river = |name: &str, fx: f64, fy: f64, discharge: f64| {
        River::new(name, [fx * lx, fy * ly], discharge)
            .with_temperature(RiverSeries::constant(temperature))
            .with_salinity(RiverSeries::constant(salinity))
    };
    RiverSources::new(
        vec![
            river("a", 0.21, 0.47, 30.0),
            river("b", 0.212, 0.472, 12.0),
            river("c", 0.33, 0.21, 6.0),
        ],
        &fx.mesh,
        &fx.geom,
        &fx.bathymetry,
        300.0,
        &[],
    )
    .expect("the rivers are on the beach")
}

/// How far uniform tracers may drift (their units): dgcom's
/// `beach_tracers_are_constant_and_conserved` asks this of the CPU and
/// measures 4.7e-9, the 2D pass's round-off over nearly dry elements.
const CONSTANCY_BAR: f64 = 5e-8;
/// How far the tracers' inventories may drift, of themselves: the same
/// test's bar is 5e-11 (2.3e-11 measured: elements too dry to define a
/// concentration keep their last one).
const INVENTORY_BAR: f64 = 5e-11;

/// `tracer`'s inventory in `state` (its units × m³): every node's column
/// sum over its depth, times its mass.
fn inventory(fx: &Fixture, state: &Solution3D, tracer: &[f64]) -> f64 {
    let d_sigma = fx.sigma.d_sigma();
    compensated((0..state.eta.data.len()).map(|idx| {
        let depth = state.eta.data[idx] - fx.bathymetry.data[idx];
        let column: f64 = (0..LEVELS)
            .map(|l| depth * d_sigma[l] * tracer[idx * LEVELS + l])
            .sum();
        fx.geom.mass[fx.geom.node_index(idx / NN, idx % NN)] * column
    }))
}

/// The tracers' judges on global 3D steps (`levels` 0) or on local ones of
/// up to `levels` levels (the passes on their elements' lists, the element
/// steps the device's own).
fn tracers_3d(gpu: &Arc<Gpu>, tally: &mut Tally, levels: usize) -> Result<(), Box<dyn Error>> {
    let fx = beach_3d();
    let steps = 40;
    let stepping = if levels == 0 {
        "global 3D steps".to_string()
    } else {
        format!("local 3D steps on up to {levels} levels")
    };
    println!(
        "3D tracers, {stepping}: the beach under its sheared flow, a wind and its uniform \
         stress, films and thin columns, {steps} steps"
    );
    let mut finest = 0;
    let forcing = Forcing {
        surface_stress: [0.03, -0.01],
        ..NO_STRESS
    };
    let moving = Terms {
        wind: true,
        ..QUIET
    };
    // Constancy: uniform T and S, rivers of the same water, advected about
    // the stratified reference. Without the Kuzmin limiter about it: that
    // limiter takes a tracer's departure from the reference, which for a
    // uniform tracer is the reference's own shape, and clips its extrema
    // (by 0.24 °C here, on the CPU as on the device)
    let (temp, salt) = (12.3, 33.1);
    let mut uniform = start_state(&fx);
    uniform.temp.fill(temp);
    uniform.salt.fill(salt);
    uniform.tke.clear();
    uniform.gls.clear();
    let drift_in = |s: &Solution3D, drift: &mut f64| {
        for (&t, &x) in s.temp.iter().zip(&s.salt) {
            let d = (t - temp).abs().max((x - salt).abs());
            *drift = if d.is_nan() { d } else { drift.max(d) };
        }
    };
    // (the device's drift and the finest level used; the CPU's kernels' if
    // asked)
    let drift_with = |rivers: Option<f64>,
                      fluxes: Option<JerlovWater>,
                      cpu: bool|
     -> Result<(f64, usize, f64), Box<dyn Error>> {
        let physics = physics_with(
            &fx,
            Terms {
                limiter: false,
                fluxes,
                ..moving
            },
            forcing,
            GlsMixing::k_epsilon(),
            rivers.map(|t| rivers_of(&fx, t, salt)),
        );
        let mut drift = 0.0f64;
        let (_, finest) = run_3d(gpu, &physics, &uniform, steps, levels, |s| {
            drift_in(s, &mut drift)
        })?;
        let mut on_cpu = f64::NAN;
        if cpu {
            on_cpu = 0.0;
            run_3d_cpu(&physics, &uniform, steps, levels, |s| {
                drift_in(s, &mut on_cpu)
            });
        }
        Ok((drift, finest, on_cpu))
    };
    // The same bar on local steps as on global ones. Until 2026-11-26 local
    // steps drifted by 2.7e-6 here (4.3e-6 with the rivers), the CPU's
    // kernels too: GLS lengths of kilometres that advection left behind
    // gave diffusivities of 10⁷ m²/s, whose implicit solve lost a uniform
    // tracer to round-off; the length is bounded by the wall and the solve
    // is for the increment since (TODO P4.5, notes/gpu.md "Local 3D steps")
    let (drift, used, on_cpu) = drift_with(Some(temp), None, true)?;
    finest = finest.max(used);
    tally.judge(
        "uniform T and S stay uniform (rivers of the same water)",
        &format!(
            "the largest drift {drift:.2e} (the CPU's kernels: {on_cpu:.2e}); it must stay \
             under {CONSTANCY_BAR:.0e}"
        ),
        drift <= CONSTANCY_BAR,
    );
    let (drift, _, _) = drift_with(Some(temp + 1e-3), None, false)?;
    tally.mutant(
        "the rivers' water 1e-3 °C warmer",
        &format!("the largest drift {drift:.2e}"),
        drift.is_nan() || drift > CONSTANCY_BAR,
    );
    // The inventories of the stratified water, nothing entering
    let mut stratified = start_state(&fx);
    stratified.tke.clear();
    stratified.gls.clear();
    let drift_of = |terms: Terms| -> Result<(f64, f64, usize), Box<dyn Error>> {
        let physics = physics_with(&fx, terms, forcing, GlsMixing::k_epsilon(), None);
        let before = [
            inventory(&fx, &stratified, &stratified.temp),
            inventory(&fx, &stratified, &stratified.salt),
        ];
        let (mut worst, mut end) = (0.0f64, [0.0; 2]);
        let (_, finest) = run_3d(gpu, &physics, &stratified, steps, levels, |s| {
            end = [inventory(&fx, s, &s.temp), inventory(&fx, s, &s.salt)];
            for (now, was) in end.iter().zip(&before) {
                let d = ((now - was) / was).abs();
                worst = if d.is_nan() { d } else { worst.max(d) };
            }
        })?;
        Ok((worst, ((end[0] - before[0]) / before[0]).abs(), finest))
    };
    let (worst, _, used) = drift_of(moving)?;
    finest = finest.max(used);
    tally.judge(
        "the inventories of T and S are kept",
        &format!(
            "the largest drift {worst:.2e} of an inventory; it must stay under \
             {INVENTORY_BAR:.0e}"
        ),
        worst <= INVENTORY_BAR,
    );
    let (_, heat, _) = drift_of(Terms {
        fluxes: Some(JerlovWater::III),
        ..moving
    })?;
    tally.mutant(
        "the surface heat flux on",
        &format!("T's inventory changed by {heat:.2e} of itself"),
        heat.is_nan() || heat > INVENTORY_BAR,
    );
    // On local steps the judges above say nothing of them unless a pass
    // listed a part of the mesh
    if levels > 0 {
        tally.judge(
            "the steps were local",
            &format!("the finest level used was {finest} (a time-step ratio of 2^{finest})"),
            finest > 0,
        );
    }
    Ok(())
}

/// Every judge; an error if one fails or a mutant is not caught.
pub fn run(gpu: &Arc<Gpu>) -> Result<(), Box<dyn Error>> {
    println!("== The device's own judges (convergence, rest, conservation, positivity) ==");
    let mut tally = Tally::default();
    convergence(gpu, &mut tally)?;
    lake_at_rest(gpu, &mut tally)?;
    mass_and_positivity(gpu, &mut tally)?;
    rest_3d(gpu, &mut tally, &beach_3d(), "the wet/dry beach", QUIET)?;
    // The seamount test: a doubly periodic bed of bumps 12 to 28 m deep under
    // the pycnocline, every element a general quadrilateral, all of it wet
    rest_3d(
        gpu,
        &mut tally,
        &Fixture::new(Shape {
            jitter: 0.15,
            stretched: true,
            reference: true,
            ..Shape::WET
        }),
        "bumps of the bed under the pycnocline, doubly periodic (the seamount test)",
        Terms {
            cages: false,
            ..QUIET
        },
    )?;
    tracers_3d(gpu, &mut tally, 0)?;
    tracers_3d(gpu, &mut tally, 2)?;
    println!(
        "  {} judges, {} mutants: {}",
        tally.judged,
        tally.mutants,
        if tally.failed.is_empty() {
            "all hold".to_string()
        } else {
            format!("FAILED: {}", tally.failed.join("; "))
        }
    );
    if tally.failed.is_empty() {
        Ok(())
    } else {
        Err(format!("the device's judges failed: {}", tally.failed.join("; ")).into())
    }
}

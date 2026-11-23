//! The 2D SWE on the GPU (P2.7, `notes/gpu.md`): a device-resident state
//! ([`DeviceState`], dgcom's [`Integrable`]) and a physics module
//! ([`GpuSWE2D`], dgcom's [`PhysicsModule`]), so that dgcom's own
//! `Simulation` and SSP-RK steppers drive the GPU.
//!
//! The split-form RHS of `SWEFormulation2D::EntropyStable` and of `WetDry`
//! (the subcells in elements with a node shallower than the subcell depth,
//! the hydrostatic HLL at faces) with a bed, walls (`Reflective2D`, called
//! as the CPU calls it), characteristic open boundaries with an external
//! state uniform along each (a harmonic tide, still water), atlas tides
//! node by node (`BoundaryTides`: each constituent's time factors from the
//! host once per stage time, the sums per node on the device) or a parent
//! model node by node (`OceanModelState`: its series per node on the device,
//! the time stencil from the host), Coriolis, the nesting's relaxation band
//! (`NestingRelaxation2D`, a kernel after the element kernels); global
//! steps through the fused per-element stages, and local time stepping
//! (dgcom's `Multirate`, [`LocalTimeStepping`]: each pass a prefix of the
//! coarse step's elements and edges sorted by level, each element's stage
//! from dgcom's `element_stage`); built from dgcom's own physics module
//! ([`GpuSWE2D::from_form`], `SWEPhysics2D::device_form`). The kernels follow
//! `SplitFormSWE2D` (`src/solver/rhs/swe_2d_split_form.rs`) operation for
//! operation and call dgcom's pointwise physics (the two-point flux, the
//! hydrostatic HLL, the subcell reconstruction, the interface metrics, the
//! wall's ghost state), so a build without multiply–add contraction
//! (`cargo oxide run --no-fmad`) gives the CPU's bits. One thread per edge and one per element; every loop unrolls (P2's
//! bounds are constants, the face nodes a formula), so an element's nodes and
//! RHS stay in registers.

// The device code's loops index arrays by the loop variable: with `#[unroll]`
// and constant bounds the indices are constants and the arrays stay in
// registers (an iterator would not unroll). Its helpers take the arrays one
// by one (a struct of them would be copied), and `#[device]` hides an item's
// own `allow` from clippy.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use std::sync::{Arc, Mutex};

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig, Stream};
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32};
use cuda_device::{DisjointSlice, device, kernel, thread, warp};
use cuda_host::cuda_module;
use dgcom::SWEState2D;
use dgcom::boundary::{
    BCContext2D, DeviceBoundary2D, DeviceExternal2D, ElevationOnly, ExternalState,
    NestingRelaxation2D, OceanModelState, Reflective2D, SWEBoundaryCondition2D,
    characteristic_boundary_state, parent_external_state, parent_state, parent_term,
    relaxation_source, tide_state, tide_term,
};
use dgcom::flux::{
    SWENodeState2D, entropy_stable_dissipation_2d, wintermeyer_bed_interface_term_2d,
    wintermeyer_flux_2d,
};
use dgcom::mesh::{Bathymetry2D, BoundaryTag, Mesh2D};
use dgcom::operators::{DGOperators2D, GeometricFactors2D};
use dgcom::physics::builder::LINE_LOCAL_HEADROOM;
use dgcom::physics::{PhysicsModule, PhysicsModuleInfo, SWEDeviceForm2D};
use dgcom::solver::rhs::subcells::telescoped_interfaces;
use dgcom::solver::{
    PositivityBound, SUBCELL_CAP_DEPTH_FACTOR, SWEFormulation2D, SWESide2D, SWESolution2D,
    WetDryConfig, damp_node, dt_from_reference_rate, hydrostatic_hll, node_reference_rate,
    reconstruct_subcell, wet_dry_element,
};
use dgcom::source::{
    CageNode, CoriolisSource2D, DeviceFriction2D, DeviceSource2D, ManningFriction2D,
};
use dgcom::time::{
    Integrable, LocalTimeStepping, NodeLines, ShuOsherStage, SspScheme, StagePlan, element_stage,
};
use dgcom::types::{Depth, ElementIndex};

/// Polynomial order of the kernels (P2).
pub const ORDER: usize = 2;
/// Nodes per GLL line.
pub const N1: usize = ORDER + 1;
/// Nodes per element.
pub const NN: usize = N1 * N1;
/// Face-buffer slots per edge: the left side's `N1`, then the right side's.
pub const NF: usize = 2 * N1;
/// Per-element geometry of a parallelogram: `J∇r` (2), `J∇s` (2), `1/J`,
/// each face's outward normal (2) and surface Jacobian, then `J`.
const EG: usize = 18;

/// The right side of a boundary edge (`edges`).
const NO_NEIGHBOUR: u32 = u32::MAX;

/// A boundary edge's condition in the place of its right face (`edges`): a
/// wall, else the index of its open boundary in [`OpenStates`].
const WALL: u32 = u32::MAX;

/// Open boundaries (boundary tags with a characteristic condition) the
/// kernels take.
pub const MAX_OPEN: usize = 4;

/// The open boundaries at the time of a stage on one level: the external
/// state of each boundary with a uniform one, evaluated on the host
/// (`ExternalStateProvider::uniform_state`), and every boundary's flags.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct OpenStates {
    /// η, u, v of each boundary's uniform external state (unused for
    /// atlas tides, [`OpenForm`])
    pub state: [[f64; 3]; MAX_OPEN],
    /// Bit 0: the state has a velocity (else elevation only); bit 1:
    /// elevation-only data enter as a wave (`ElevationOnly::IncomingWave`)
    pub flags: [u32; MAX_OPEN],
}

/// [`OpenForm::kind`]: a uniform state, its level's [`OpenStates`] entry.
const UNIFORM: u32 = 0;
/// [`OpenForm::kind`]: atlas tides per face node (`BoundaryTides`).
const TIDES: u32 = 1;
/// [`OpenForm::kind`]: a parent model per face node (`OceanModelState`).
const PARENT: u32 = 2;

/// An open boundary's external data as the boundary kernel evaluates it
/// (`DeviceExternal2D`), in a device buffer: a uniform state, atlas tides
/// (the constants of a face node's row of the tide tables, the time factors
/// of its level's row of the moments) or a parent (the slot of a face
/// node's row, the parent's [`ParentForm`]).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct OpenForm {
    /// [`UNIFORM`], [`TIDES`] or [`PARENT`]
    pub kind: u32,
    /// The tides' constituents
    pub constituents: u32,
    /// The tides' moment in a level's row of the moments
    /// (`BoundaryTides::moment`: the ramp, then 3 per constituent)
    pub moment: u32,
    /// The tides carry a velocity
    pub velocity: u32,
    /// The parent's index in the parent forms
    pub parent: u32,
    /// The tides' largest transport-scaling factor (0: not scaled)
    pub max_transport_ratio: f64,
}

// SAFETY: plain old data (u32s and an f64, `#[repr(C)]`).
unsafe impl cuda_core::DeviceCopy for OpenForm {}

/// A parent model on the device (`OceanModelState`), in a device buffer:
/// where its slots' series and data start in the parent tables, and its
/// moment in a level's row of the moments (`OceanModelState::moment`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ParentForm {
    /// Snapshots per slot
    pub n_times: u32,
    /// Its slot 0's first snapshot in the series
    pub series: u32,
    /// Its slot 0 in the slot data (`[parent depth, band weight]`)
    pub slots: u32,
    /// Its moment in a level's row of the moments
    pub moment: u32,
    /// The parent has a depth-mean velocity
    pub velocity: u32,
    /// The largest transport-scaling factor (0: not scaled)
    pub max_transport_ratio: f64,
    /// Added to ζ, ramped with it
    pub reference_level: f64,
}

// SAFETY: plain old data (u32s and f64s, `#[repr(C)]`).
unsafe impl cuda_core::DeviceCopy for ParentForm {}

/// The parent tables as the kernels read them ([`Forcing`]): the parents'
/// forms, every slot's snapshots and its `[parent depth, band weight]`.
type ParentRows<'a> = (&'a [ParentForm], &'a [[f32; 3]], &'a [[f64; 2]]);

/// Parent `form`'s `(η, velocity)` at its slot `slot` over bed `bed`, the
/// time weights and ramp from `moment` (a level's row): dgcom's
/// `parent_term` and `parent_state`, as `OceanModelState::evaluate`.
#[inline]
fn parent_at(
    form: &ParentForm,
    slot: usize,
    moment: &[f64],
    series: &[[f32; 3]],
    slots: &[[f64; 2]],
    bed: f64,
) -> (f64, Option<(f64, f64)>) {
    let m = form.moment as usize;
    let first = form.series as usize + slot * form.n_times as usize;
    let mut sum = [0.0; 3];
    for i in 0..moment[m + 1] as usize {
        parent_term(
            &mut sum,
            moment[m + 6 + i],
            &series[first + moment[m + 2 + i] as usize],
        );
    }
    let transport = if form.max_transport_ratio > 0.0 {
        Some((
            form.max_transport_ratio,
            slots[form.slots as usize + slot][0],
        ))
    } else {
        None
    };
    parent_state(
        sum,
        moment[m],
        form.reference_level,
        form.velocity != 0,
        transport,
        bed,
    )
}

/// Local time stepping's levels below the coarsest that the kernels take
/// (a stage's boundary data has an entry per level, [`StageLevel`]).
pub const MAX_DEVICE_LEVELS: usize = 8;

/// Entries of a stage's boundary data: one per level `0..=MAX_DEVICE_LEVELS`.
const LEVELS: usize = MAX_DEVICE_LEVELS + 1;

/// A stage's time and open boundaries' external states on one level: with
/// local time stepping the elements of one pass are at different times, the
/// same time for all elements of a level (`element_stage`); global steps use
/// level 0 only. In a device buffer ([`BoundaryUploads`]): a kernel copies a
/// by-value aggregate parameter into every thread's local memory.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct StageLevel {
    pub time: f64,
    pub open: OpenStates,
}

// SAFETY: plain old data (f64s and u32s, `#[repr(C)]`).
unsafe impl cuda_core::DeviceCopy for StageLevel {}

/// The point-implicit damping of a stage (`damp_node`): Manning friction if
/// `friction`, the thin-layer relaxation if `wet_dry` (its `WetDryConfig`
/// also the wet/dry correction's).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Damping {
    pub h_min: f64,
    pub friction: u32,
    pub g: f64,
    pub manning_n: f64,
    pub friction_h_min: f64,
    pub wet_dry: u32,
    pub h_dry: f64,
    pub relaxation_time: f64,
    pub max_velocity: f64,
    pub wet_dry_g: f64,
}

impl Damping {
    /// Whether it damps at all (`ImplicitDamping2D::is_active`).
    #[inline]
    fn is_active(&self) -> bool {
        self.friction != 0 || self.wet_dry != 0
    }

    #[inline]
    fn wet_dry_config(&self) -> WetDryConfig {
        WetDryConfig {
            h_dry: Depth::new(self.h_dry),
            relaxation_time: self.relaxation_time,
            max_velocity: self.max_velocity,
            g: self.wet_dry_g,
        }
    }

    /// `damp_node` of mesh node `n` with depth `h`, from `from`, over `dt`.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn node(&self, n: usize, h: f64, u: &mut f64, v: &mut f64, from: SWEState2D, dt: f64) {
        let law = ManningFriction2D {
            g: self.g,
            manning_n: self.manning_n,
            h_min: self.friction_h_min,
        };
        let config = self.wet_dry_config();
        let no_cages: &[CageNode] = &[];
        damp_node(
            if self.friction != 0 { Some(&law) } else { None },
            no_cages,
            if self.wet_dry != 0 {
                Some(&config)
            } else {
                None
            },
            Depth::new(self.h_min),
            n,
            h,
            u,
            v,
            from,
            dt,
        );
    }
}

/// Thread `i`'s item of a launch over `n` items: `list[i]`, or `i` itself
/// where the list is empty (every item); `None` past the end.
#[inline]
fn item(i: usize, list: &[u32], n: u32) -> Option<usize> {
    if i >= n as usize {
        None
    } else if list.is_empty() {
        Some(i)
    } else {
        Some(list[i] as usize)
    }
}

/// Thread `i`'s element of a split list (`split_elements`): `list[i]` for
/// the first `counts[kind]` threads, `None` past them. The count is the
/// device's, so a launch covers the pass and the threads past the kind's
/// elements return at once.
#[inline]
fn split_item(i: usize, list: &[u32], counts: &[u32], kind: usize) -> Option<usize> {
    if i >= counts[kind] as usize {
        None
    } else {
        Some(list[i] as usize)
    }
}

/// The wet elements' slot of `split_elements`' counts (the flux-differencing
/// kernel's) and the subcell elements' (the subcell kernels').
const WET: usize = 0;
const SHALLOW: usize = 1;

/// The level of element `k` (`level` empty: global steps, level 0).
#[inline]
fn level_of(level: &[u32], k: usize) -> usize {
    if level.is_empty() {
        0
    } else {
        level[k] as usize
    }
}

/// Threads per block of the element kernels (their registers bound the block).
const ELEMENT_BLOCK: u32 = 128;

/// Threads per block of `subcell_terms`: at its 186 registers an SM holds
/// 10 warps, which 256-thread blocks leave at 8.
const SUBCELL_BLOCK: u32 = 64;

/// Volume node of face node `fi` of `face` (`DGOperators2D::face_nodes`:
/// faces 2 and 3 list theirs in reverse). A formula, not a table, so that
/// with the loops unrolled the element's nodes are indexed by constants and
/// stay in registers.
#[inline]
pub(crate) const fn face_node(face: usize, fi: usize) -> usize {
    match face {
        0 => fi,
        1 => fi * N1 + N1 - 1,
        2 => (N1 - 1) * N1 + N1 - 1 - fi,
        _ => (N1 - 1 - fi) * N1,
    }
}

// =============================================================================
// Device helpers: dgcom's pointwise physics, arranged as SplitFormSWE2D does
// =============================================================================

/// Node `n` (`k·NN + i`) of the state, as `SplitFormSWE2D` reads it.
#[inline]
fn node_state(
    h: &[f64],
    hu: &[f64],
    hv: &[f64],
    bed: &[f64],
    n: usize,
    h_min: f64,
) -> SWENodeState2D {
    SWENodeState2D::new(&SWEState2D::new(h[n], hu[n], hv[n]), bed[n], h_min)
}

/// `SplitFormSWE2D::surface_flux`: `F*·n` seen from side a, and `F*·(−n)`
/// from side b; the hydrostatic HLL of `WetDry` if `hll`, else the
/// entropy-stable flux.
#[inline]
fn surface_flux(
    q_a: &SWENodeState2D,
    q_b: &SWENodeState2D,
    n: (f64, f64),
    g: f64,
    hll: bool,
    h_min: f64,
) -> (SWEState2D, SWEState2D) {
    if hll {
        let (f_a, f_b) = hydrostatic_hll(&q_a.into(), &q_b.into(), n, g, h_min);
        return (f_a, -f_b);
    }
    let mut shared = wintermeyer_flux_2d(q_a, q_b, n, g);
    shared = shared + entropy_stable_dissipation_2d(q_a, q_b, n, g);
    let minus_n = (-n.0, -n.1);
    (
        shared + wintermeyer_bed_interface_term_2d(q_a.h, q_a.b, q_b.b, n, g),
        -shared + wintermeyer_bed_interface_term_2d(q_b.h, q_b.b, q_a.b, minus_n, g),
    )
}

/// `SplitFormSWE2D::line_volume` with a bed: the flux-differencing volume
/// term (times J) along the line of nodes `idx(0..N1)`.
#[device]
fn line_volume<const AFFINE: bool, I: Fn(usize) -> usize>(
    nodes: &[SWENodeState2D; NN],
    rhs: &mut [SWEState2D; NN],
    metric: &[(f64, f64); N1],
    idx: I,
    affine_metric: (f64, f64),
    d1: &[f64],
    g: f64,
) {
    #[unroll]
    for a in 0..N1 {
        let ia = idx(a);
        let q_a = nodes[ia];
        let m_a = if AFFINE { affine_metric } else { metric[a] };

        let mut acc = (2.0 * d1[a * N1 + a]) * wintermeyer_flux_2d(&q_a, &q_a, m_a, g);
        // c = a + 1..N1, as a constant trip count so that it unrolls
        #[unroll]
        for c in 0..N1 {
            if c <= a {
                continue;
            }
            let ic = idx(c);
            let m = if AFFINE {
                m_a
            } else {
                let m_c = metric[c];
                (0.5 * (m_a.0 + m_c.0), 0.5 * (m_a.1 + m_c.1))
            };
            let f = wintermeyer_flux_2d(&q_a, &nodes[ic], m, g);
            acc = acc + (2.0 * d1[a * N1 + c]) * f;
            rhs[ic] = rhs[ic] - (2.0 * d1[c * N1 + a]) * f;
        }

        let (mut db, mut db_x, mut db_y) = (0.0, 0.0, 0.0);
        #[unroll]
        for c in 0..N1 {
            let d_b = d1[a * N1 + c] * nodes[idx(c)].b;
            db += d_b;
            if !AFFINE {
                db_x += d_b * metric[c].0;
                db_y += d_b * metric[c].1;
            }
        }
        let (db_x, db_y) = if AFFINE {
            (db * m_a.0, db * m_a.1)
        } else {
            (0.5 * (db * m_a.0 + db_x), 0.5 * (db * m_a.1 + db_y))
        };
        let force = g * q_a.h;
        acc = acc + SWEState2D::new(0.0, force * db_x, force * db_y);

        rhs[ia] = rhs[ia] - acc;
    }
}

/// The terms per node of a line's subcells: 3 for each of its nodes, in the
/// order `SplitFormSWE2D::line_subcells` adds them to the node's RHS.
pub const LINE_TERMS: usize = 3;

/// `SplitFormSWE2D::line_subcells`: the subcell finite-volume update (times
/// J) along one GLL line of an element, its nodes `q` in order, the metric
/// at them `metric`, its end subcells taking their outer slope neighbour
/// from `ends` (the node, its distance, whether there is one); as the terms
/// it adds to each node, in the CPU's order: node `a` gets `terms[a][0]`,
/// then `[1]`, then `[2]` (the end flux or an interface's HLL flux, an
/// interface's HLL flux, the bed and metric balance). A subtraction on the
/// CPU is the negated term's sum here, the same bits. Not generic, so that
/// it inlines (a generic `#[device]` function stays a call), and the GLL
/// data are arrays, so that its loops have constant trip counts.
#[allow(clippy::too_many_arguments)]
#[device]
// The device backend makes this alwaysinline (no call in the PTX); the host
// compiler, which sees an exported symbol, ignores it.
#[allow(unused_attributes)]
#[inline(always)]
fn line_terms(
    q: &[SWENodeState2D; N1],
    metric: &[(f64, f64); N1],
    ends: [(SWENodeState2D, f64, bool); 2],
    affine: bool,
    w: &[f64; N1],
    xi: &[f64; N1],
    d1: &[f64; NN],
    g: f64,
    h_min: f64,
    h_dry: f64,
) -> [[SWEState2D; LINE_TERMS]; N1] {
    let mut terms = [[SWEState2D::zero(); LINE_TERMS]; N1];
    // Interface metrics (the telescoping sum of the flux-differencing term)
    let mut interfaces = [(0.0, 0.0); N1 + 1];
    telescoped_interfaces(w, d1, metric, affine, &mut interfaces);

    let (q_first, q_last) = (q[0], q[N1 - 1]);
    let (m_first, m_last) = (metric[0], metric[N1 - 1]);
    terms[0][0] = (1.0 / w[0]) * wintermeyer_flux_2d(&q_first, &q_first, m_first, g);
    terms[N1 - 1][0] = -((1.0 / w[N1 - 1]) * wintermeyer_flux_2d(&q_last, &q_last, m_last, g));

    // Face states. Subcell a spans [x_a, x_a + w_a] with x_0 = −1
    let mut faces = [(SWESide2D::default(), SWESide2D::default()); N1];
    let mut x_left = -1.0;
    #[unroll]
    for a in 0..N1 {
        let side = SWESide2D::from(&q[a]);
        let x_right = x_left + w[a];
        let (left, x_l, has_left) = if a > 0 {
            (SWESide2D::from(&q[a - 1]), xi[a - 1], true)
        } else {
            (SWESide2D::from(&ends[0].0), -1.0 - ends[0].1, ends[0].2)
        };
        let (right, x_r, has_right) = if a + 1 < N1 {
            (SWESide2D::from(&q[a + 1]), xi[a + 1], true)
        } else {
            (SWESide2D::from(&ends[1].0), 1.0 + ends[1].1, ends[1].2)
        };
        faces[a] = if has_left && has_right {
            reconstruct_subcell(
                [left, side, right],
                [x_l, xi[a], x_r],
                (x_left, x_right),
                h_dry,
            )
        } else {
            (side, side)
        };
        x_left = x_right;
    }

    // Interface a + 1/2: node a's term after its first, node a + 1's after
    // the end flux or the interface before it
    #[unroll]
    for a in 0..N1 - 1 {
        let m = interfaces[a + 1];
        let (f_a, f_b) = hydrostatic_hll(&faces[a].1, &faces[a + 1].0, m, g, h_min);
        terms[a][1] = -((1.0 / w[a]) * f_a);
        terms[a + 1][if a + 1 == N1 - 1 { 1 } else { 0 }] = (1.0 / w[a + 1]) * f_b;
    }

    // Bed-slope term, zero in subcells without a slope (B_L = B_R), and the
    // metric balance term, zero on affine elements
    #[unroll]
    for a in 0..N1 {
        let (left, right) = faces[a];
        let w_a = w[a];
        let bed = -0.5 * g * (left.h + right.h) * (right.b - left.b) / w_a;
        let (force_x, force_y) = if affine {
            let m = metric[0];
            (bed * m.0, bed * m.1)
        } else {
            let (m_l, m_r) = (interfaces[a], interfaces[a + 1]);
            let (h_a, b_a) = (q[a].h, q[a].b);
            let balance =
                0.25 * g * ((right.h + h_a) * (b_a - right.b) + (left.h + h_a) * (b_a - left.b))
                    / w_a;
            (
                bed * 0.5 * (m_l.0 + m_r.0) + balance * (m_r.0 - m_l.0),
                bed * 0.5 * (m_l.1 + m_r.1) + balance * (m_r.1 - m_l.1),
            )
        };
        terms[a][2] = SWEState2D::new(0.0, force_x, force_y);
    }
    terms
}

/// `SplitFormSWE2D::outer_node`: of face node `fi` of `face` of element `k`,
/// the neighbour's first interior node on the GLL line through it, its
/// distance from the face in `k`'s reference coordinate, and whether there
/// is a neighbour.
#[inline]
#[allow(clippy::too_many_arguments)]
fn outer_node(
    k: usize,
    face: usize,
    fi: usize,
    state: (&[f64], &[f64], &[f64], &[f64]),
    outer_distance: &[f64],
    element_edges: &[[u32; 4]],
    edges: &[[u32; 4]],
    h_min: f64,
) -> (SWENodeState2D, f64, bool) {
    let edge = edges[element_edges[k][face] as usize];
    if edge[2] == NO_NEIGHBOUR {
        return (SWENodeState2D::default(), 0.0, false);
    }
    let (nb, nb_face) = if edge[0] as usize == k && edge[1] as usize == face {
        (edge[2] as usize, edge[3] as usize)
    } else {
        (edge[0] as usize, edge[1] as usize)
    };
    let nb_node = face_node(nb_face, N1 - 1 - fi);
    // One node into the neighbour, normal to its face
    let inner = match nb_face {
        0 => nb_node + N1,
        1 => nb_node - 1,
        2 => nb_node - N1,
        _ => nb_node + 1,
    };
    // Static: `Statics::upload` computes it as the CPU does
    let distance = outer_distance[(k * 4 + face) * N1 + fi];
    let (h, hu, hv, bed) = state;
    (
        node_state(h, hu, hv, bed, nb * NN + inner, h_min),
        distance,
        true,
    )
}

/// `SplitFormSWE2D::boundary_flux` at a wall: `Reflective2D`'s ghost state
/// (dgcom's own), through the surface flux.
#[inline]
#[allow(clippy::too_many_arguments)]
fn wall_flux(
    q_int: &SWENodeState2D,
    state: SWEState2D,
    bed: f64,
    normal: (f64, f64),
    time: f64,
    g: f64,
    h_min: f64,
    wall_h_min: f64,
    hll: bool,
) -> SWEState2D {
    // The position is not read by a wall
    let ctx = BCContext2D::new(time, (0.0, 0.0), state, bed, normal, g, h_min);
    let ghost = Reflective2D::with_h_min(wall_h_min).ghost_state(&ctx);
    let q_ext = SWENodeState2D::new(&ghost, q_int.b, h_min);
    surface_flux(q_int, &q_ext, normal, g, hll, h_min).0
}

/// The external state of open boundary `i` at a face node over bed `bed`:
/// its level's uniform state (`open`); or from the node's row `row` of its
/// form's table and the level's moments (`moment`), its atlas tides
/// (`tides`: `coefficients` and `nodes`, `[mean, source depth]`, with
/// `max_constituents` per row; dgcom's `tide_term` and `tide_state`) or its
/// parent (the row's slot, `parents`; [`parent_at`]).
#[inline]
#[allow(clippy::too_many_arguments)]
fn open_external(
    open: &OpenStates,
    form: &OpenForm,
    i: usize,
    bed: f64,
    row: usize,
    moment: &[f64],
    (_, coefficients, nodes, max_constituents): TideRows<'_>,
    (parents, series, slots): ParentRows<'_>,
    parent_slot: &[u32],
) -> ExternalState {
    if form.kind == UNIFORM {
        let [eta, u, v] = open.state[i];
        return if open.flags[i] & 1 != 0 {
            ExternalState::new(eta, u, v)
        } else {
            ExternalState::elevation(eta)
        };
    }
    if form.kind == PARENT {
        let parent = &parents[form.parent as usize];
        let slot = parent_slot[row] as usize;
        return parent_external_state(parent_at(parent, slot, moment, series, slots, bed));
    }
    let m = form.moment as usize;
    let mut sum = [0.0; 3];
    for j in 0..form.constituents as usize {
        let p = m + 1 + 3 * j;
        tide_term(
            &mut sum,
            &coefficients[row * max_constituents + j],
            moment[p],
            moment[p + 1],
            moment[p + 2],
        );
    }
    let [mean, source_depth] = nodes[row];
    let transport = if form.max_transport_ratio > 0.0 {
        Some((form.max_transport_ratio, source_depth))
    } else {
        None
    };
    tide_state(sum, moment[m], mean, form.velocity != 0, transport, bed)
}

/// `SplitFormSWE2D::boundary_flux` at a characteristic open boundary:
/// dgcom's `characteristic_boundary_state` from the interior trace and the
/// external state `external` of open boundary `i` (its elevation-only data's
/// velocity from `open`), its physical flux `F(q_b)·n`.
#[inline]
#[allow(clippy::too_many_arguments)]
fn open_flux(
    q_int: &SWENodeState2D,
    state: SWEState2D,
    bed: f64,
    normal: (f64, f64),
    time: f64,
    g: f64,
    h_min: f64,
    external: ExternalState,
    open: &OpenStates,
    i: usize,
) -> SWEState2D {
    let flags = open.flags[i];
    let elevation_only = if flags & 2 != 0 {
        ElevationOnly::IncomingWave
    } else {
        ElevationOnly::AtRest
    };
    // The position is not read with the external state given
    let ctx = BCContext2D::new(time, (0.0, 0.0), state, bed, normal, g, h_min);
    let q = characteristic_boundary_state(&ctx, external, elevation_only);
    let q_b = SWENodeState2D::new(&q, q_int.b, h_min);
    wintermeyer_flux_2d(&q_b, &q_b, normal, g)
}

/// `(∇r, ∇s, |∇r|, |∇s|)` of a node from `[r_x, r_y, s_x, s_y]`, as
/// `element_max_reference_rate` forms it.
#[inline]
fn rate_metric(grads: [f64; 4]) -> ((f64, f64), (f64, f64), f64, f64) {
    let norm_r = (grads[0] * grads[0] + grads[1] * grads[1]).sqrt();
    let norm_s = (grads[2] * grads[2] + grads[3] * grads[3]).sqrt();
    ((grads[0], grads[1]), (grads[2], grads[3]), norm_r, norm_s)
}

/// `SplitFormSWE2D::edge_fluxes` of interior edge `e`, into `slot`: both
/// sides' F* (left first, each in its own face-node order), the hydrostatic
/// HLL if `HLL`, else the entropy-stable flux.
#[allow(clippy::too_many_arguments)]
#[device]
fn edge_body<const HLL: bool>(
    e: usize,
    g: f64,
    h_min: f64,
    (h, hu, hv, bed): (&[f64], &[f64], &[f64], &[f64]),
    affine: &[u32],
    egeom: &[[f64; EG]],
    fgeom: &[[f64; 3]],
    edges: &[[u32; 4]],
    slot: &mut [[f64; 3]; NF],
) {
    let [le, lf, re, rf] = edges[e];
    let (le, lf, re, rf) = (le as usize, lf as usize, re as usize, rf as usize);
    #[unroll]
    for fi in 0..N1 {
        let normal = if affine[le] != 0 {
            (egeom[le][5 + 3 * lf], egeom[le][6 + 3 * lf])
        } else {
            let f = fgeom[(le * 4 + lf) * N1 + fi];
            (f[0], f[1])
        };
        let left = le * NN + face_node(lf, fi);
        let q_left = node_state(h, hu, hv, bed, left, h_min);
        let rfi = N1 - 1 - fi;
        let right = re * NN + face_node(rf, rfi);
        let (f_left, f_right) = surface_flux(
            &q_left,
            &node_state(h, hu, hv, bed, right, h_min),
            normal,
            g,
            HLL,
            h_min,
        );
        slot[fi] = [f_left.h, f_left.hu, f_left.hv];
        slot[N1 + rfi] = [f_right.h, f_right.hu, f_right.hv];
    }
}

/// The tide tables as the boundary kernel reads them ([`Forcing`]): every
/// edge's first row in its form's table, each tide row's constants and
/// `[mean, source depth]`, and the constituents per row.
type TideRows<'a> = (&'a [u32], &'a [[f64; 6]], &'a [[f64; 2]], usize);

/// `SplitFormSWE2D::boundary_flux` at face node `fi` of boundary edge `e`:
/// a wall's ghost state or an open boundary's state at its element's stage
/// time (its level's entry of `boundary`, and of the tides' `moments`, rows
/// of `width`), through the hydrostatic HLL if `hll`. An open boundary's
/// external data is its form's (`opens`); atlas tides read the face node's
/// row of the tide tables (`tides`: every edge's first row, the constants,
/// `[mean, source depth]` and the constituents per row).
#[allow(clippy::too_many_arguments)]
#[device]
fn boundary_node(
    e: usize,
    fi: usize,
    hll: bool,
    g: f64,
    h_min: f64,
    wall_h_min: f64,
    boundary: &[StageLevel],
    (opens, moments, width): (&[OpenForm], &[f64], usize),
    tides: TideRows<'_>,
    parents: ParentRows<'_>,
    parent_slot: &[u32],
    level: &[u32],
    (h, hu, hv, bed): (&[f64], &[f64], &[f64], &[f64]),
    affine: &[u32],
    egeom: &[[f64; EG]],
    fgeom: &[[f64; 3]],
    edges: &[[u32; 4]],
) -> [f64; 3] {
    let [le, lf, _, rf] = edges[e];
    let (le, lf) = (le as usize, lf as usize);
    let l = level_of(level, le);
    // By reference: a copy, indexed by the boundary, would be local memory
    let StageLevel { time, open } = &boundary[l];
    let time = *time;
    let normal = if affine[le] != 0 {
        (egeom[le][5 + 3 * lf], egeom[le][6 + 3 * lf])
    } else {
        let f = fgeom[(le * 4 + lf) * N1 + fi];
        (f[0], f[1])
    };
    let left = le * NN + face_node(lf, fi);
    let q_left = node_state(h, hu, hv, bed, left, h_min);
    let state = SWEState2D::new(h[left], hu[left], hv[left]);
    let f = if rf == WALL {
        wall_flux(
            &q_left, state, bed[left], normal, time, g, h_min, wall_h_min, hll,
        )
    } else {
        let i = rf as usize;
        let form = &opens[i];
        // The edge's rows in its form's table (none for a uniform state)
        let row = if form.kind == UNIFORM {
            0
        } else {
            tides.0[e] as usize + fi
        };
        let external = open_external(
            open,
            form,
            i,
            bed[left],
            row,
            &moments[l * width..(l + 1) * width],
            tides,
            parents,
            parent_slot,
        );
        open_flux(
            &q_left, state, bed[left], normal, time, g, h_min, external, open, i,
        )
    };
    [f.h, f.hu, f.hv]
}

/// Whether element `k` has a node shallower than `depth` (its volume term
/// is then the subcells', `SplitFormSWE2D::element_rhs`).
#[device]
// The device backend makes this alwaysinline (no call in the PTX); the host
// compiler, which sees an exported symbol, ignores it.
#[allow(unused_attributes)]
#[inline(always)]
fn is_subcell_element(k: usize, h: &[f64], depth: f64) -> bool {
    let mut shallow = false;
    #[unroll]
    for i in 0..NN {
        if h[k * NN + i] < depth {
            shallow = true;
        }
    }
    shallow
}

/// The end of `SplitFormSWE2D::element_rhs` for element `k`: the lifted face
/// terms `LIFT sJ (F(q⁻)·n − F*)` with F* from `faces`, then `1/J`, into
/// `out` (`[h, hu, hv]`); then the Coriolis force as `CoriolisSource2D`
/// adds it (`coriolis` 1: f = `f0`, 2: f of each node from `f_node`).
#[allow(clippy::too_many_arguments)]
#[device]
// The device backend makes this alwaysinline (no call in the PTX); the host
// compiler, which sees an exported symbol, ignores it.
#[allow(unused_attributes)]
#[inline(always)]
fn finish_element(
    k: usize,
    nodes: &[SWENodeState2D; NN],
    mut rhs: [SWEState2D; NN],
    g: f64,
    aff: bool,
    eg: &[f64; EG],
    jinv: &[f64],
    fgeom: &[[f64; 3]],
    lift: &[f64],
    element_edges: &[[u32; 4]],
    edges: &[[u32; 4]],
    faces: &[[[f64; 3]; NF]],
    (coriolis, f0, f_node): (u32, f64, &[f64]),
    out_h: &mut DisjointSlice<f64>,
    out_hu: &mut DisjointSlice<f64>,
    out_hv: &mut DisjointSlice<f64>,
) {
    #[unroll]
    for face in 0..4 {
        let e = element_edges[k][face] as usize;
        let edge = edges[e];
        // A wall's F* is in its left (own) side
        let side = usize::from(edge[0] as usize != k || edge[1] as usize != face);
        #[unroll]
        for fi in 0..N1 {
            let node = face_node(face, fi);
            let (normal, scale) = if aff {
                ((eg[5 + 3 * face], eg[6 + 3 * face]), eg[7 + 3 * face])
            } else {
                let f = fgeom[(k * 4 + face) * N1 + fi];
                ((f[0], f[1]), f[2])
            };
            let q_int = nodes[node];
            let s = faces[e][side * N1 + fi];
            let f_star = SWEState2D::new(s[0], s[1], s[2]);
            let flux_diff = wintermeyer_flux_2d(&q_int, &q_int, normal, g) - f_star;
            rhs[node] = rhs[node] + (scale * lift[(face * NN + node) * N1 + fi]) * flux_diff;
        }
    }

    #[unroll]
    for i in 0..NN {
        let r = rhs[i];
        let j_inv = if aff { eg[4] } else { jinv[k * NN + i] };
        let n = k * NN + i;
        let (mut out_u, mut out_v) = (j_inv * r.hu, j_inv * r.hv);
        if coriolis != 0 {
            let f = if coriolis == 1 { f0 } else { f_node[n] };
            out_u += f * nodes[i].hv;
            out_v += -f * nodes[i].hu;
        }
        // SAFETY: thread k writes only its own element's rows
        // (k·NN..(k + 1)·NN; the listed elements are distinct), and
        // k < n_elements.
        unsafe {
            *out_h.get_unchecked_mut(n) = j_inv * r.h;
            *out_hu.get_unchecked_mut(n) = out_u;
            *out_hv.get_unchecked_mut(n) = out_v;
        }
    }
}

// =============================================================================
// Kernels
// =============================================================================

#[cuda_module]
pub mod kernels {
    use super::*;

    /// The entropy-stable F* of the listed interior edges (one thread per
    /// edge).
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn edge_fluxes_entropy_stable(
        g: f64,
        h_min: f64,
        list: &[u32],
        n: u32,
        h: &[f64],
        hu: &[f64],
        hv: &[f64],
        bed: &[f64],
        affine: &[u32],
        egeom: &[[f64; EG]],
        fgeom: &[[f64; 3]],
        edges: &[[u32; 4]],
        mut out: DisjointSlice<[[f64; 3]; NF]>,
    ) {
        let Some(e) = item(thread::index_1d().get(), list, n) else {
            return;
        };
        // SAFETY: the listed edges are distinct, so thread i alone writes
        // edge e's slots, and e < n_edges (the buffer's length).
        let slot = unsafe { out.get_unchecked_mut(e) };
        edge_body::<false>(
            e,
            g,
            h_min,
            (h, hu, hv, bed),
            affine,
            egeom,
            fgeom,
            edges,
            slot,
        );
    }

    /// The hydrostatic HLL F* of the listed interior edges (`WetDry`, one
    /// thread per edge).
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn edge_fluxes_hll(
        g: f64,
        h_min: f64,
        list: &[u32],
        n: u32,
        h: &[f64],
        hu: &[f64],
        hv: &[f64],
        bed: &[f64],
        affine: &[u32],
        egeom: &[[f64; EG]],
        fgeom: &[[f64; 3]],
        edges: &[[u32; 4]],
        mut out: DisjointSlice<[[f64; 3]; NF]>,
    ) {
        let Some(e) = item(thread::index_1d().get(), list, n) else {
            return;
        };
        // SAFETY: the listed edges are distinct, so thread i alone writes
        // edge e's slots, and e < n_edges (the buffer's length).
        let slot = unsafe { out.get_unchecked_mut(e) };
        edge_body::<true>(
            e,
            g,
            h_min,
            (h, hu, hv, bed),
            affine,
            egeom,
            fgeom,
            edges,
            slot,
        );
    }

    /// The F* of the listed boundary edges (one thread per face node): walls
    /// and open boundaries at their element's stage time, `boundary`'s
    /// entry of its level (and the tides' `moments`, a row of `width` per
    /// level); an open boundary's data is its form's (`opens`): atlas tides
    /// or a parent from the face node's row of its form's table (`row` of
    /// its edge plus the face node).
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn boundary_fluxes(
        hll: u32,
        g: f64,
        h_min: f64,
        wall_h_min: f64,
        boundary: &[StageLevel],
        opens: &[OpenForm],
        moments: &[f64],
        width: u32,
        row: &[u32],
        tide_coefficients: &[[f64; 6]],
        tide_nodes: &[[f64; 2]],
        max_constituents: u32,
        parents: &[ParentForm],
        parent_series: &[[f32; 3]],
        parent_slots: &[[f64; 2]],
        parent_row_slot: &[u32],
        level: &[u32],
        list: &[u32],
        n: u32,
        h: &[f64],
        hu: &[f64],
        hv: &[f64],
        bed: &[f64],
        affine: &[u32],
        egeom: &[[f64; EG]],
        fgeom: &[[f64; 3]],
        edges: &[[u32; 4]],
        mut out: DisjointSlice<[[f64; 3]; NF]>,
    ) {
        // One thread per face node: few boundary edges, each a long FP64
        // sequence, and a GPU has many SMs to keep busy
        let t = thread::index_1d().get();
        let (i, fi) = (t / N1, t % N1);
        let Some(e) = item(i, list, n) else {
            return;
        };
        let f = boundary_node(
            e,
            fi,
            hll != 0,
            g,
            h_min,
            wall_h_min,
            boundary,
            (opens, moments, width as usize),
            (
                row,
                tide_coefficients,
                tide_nodes,
                max_constituents as usize,
            ),
            (parents, parent_series, parent_slots),
            parent_row_slot,
            level,
            (h, hu, hv, bed),
            affine,
            egeom,
            fgeom,
            edges,
        );
        // SAFETY: the listed edges are distinct and e < n_edges; thread
        // (i, fi) alone writes slot fi of edge e (a boundary edge has only
        // its left side), through a raw pointer, no reference to the edge's
        // other slots.
        unsafe {
            let slots = out.as_mut_ptr().add(e) as *mut [f64; 3];
            *slots.add(fi) = f;
        }
    }

    /// The nesting's relaxation band towards parent `p` (`NestingRelaxation2D`
    /// with `rate`, `wet_depth`, `momentum_only`), added to the RHS `out`
    /// of the listed elements (one thread per element), after the element
    /// kernels: as the CPU adds its sources after the Coriolis force, its
    /// `add_element` node by node over the band's elements (`band`, every
    /// node with a slot, `slot_of_node`; both parent `p`'s of
    /// `n_elements` elements), dgcom's `relaxation_source` at the element's
    /// level's time ([`parent_at`]).
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn nesting_relaxation(
        p: u32,
        rate: f64,
        wet_depth: f64,
        momentum_only: u32,
        parents: &[ParentForm],
        series: &[[f32; 3]],
        slots: &[[f64; 2]],
        slot_of_node: &[u32],
        band: &[u32],
        n_elements: u32,
        moments: &[f64],
        width: u32,
        level: &[u32],
        list: &[u32],
        n: u32,
        h: &[f64],
        hu: &[f64],
        hv: &[f64],
        bed: &[f64],
        mut out_h: DisjointSlice<f64>,
        mut out_hu: DisjointSlice<f64>,
        mut out_hv: DisjointSlice<f64>,
    ) {
        let Some(k) = item(thread::index_1d().get(), list, n) else {
            return;
        };
        let p = p as usize;
        let n_el = n_elements as usize;
        if band[p * n_el + k] == 0 {
            return;
        }
        let form = &parents[p];
        let width = width as usize;
        let l = level_of(level, k);
        let moment = &moments[l * width..(l + 1) * width];
        for i in 0..NN {
            let node = k * NN + i;
            let slot = slot_of_node[p * n_el * NN + node];
            if slot == u32::MAX {
                continue;
            }
            let slot = slot as usize;
            let q = SWEState2D::new(h[node], hu[node], hv[node]);
            let b = bed[node];
            let s = relaxation_source(
                slots[form.slots as usize + slot][1],
                rate,
                wet_depth,
                momentum_only != 0,
                &q,
                b,
                || parent_at(form, slot, moment, series, slots, b),
            );
            // SAFETY: thread k writes only its own element's rows (the
            // listed elements are distinct), and k < n_elements.
            unsafe {
                *out_h.get_unchecked_mut(node) += s.h;
                *out_hu.get_unchecked_mut(node) += s.hu;
                *out_hv.get_unchecked_mut(node) += s.hv;
            }
        }
    }

    /// The listed elements (every element if `list` is empty) split by
    /// kind: those without a node shallower than `subcell_depth` appended to
    /// `wet`, the others to `shallow`, `counts[WET]` and `counts[SHALLOW]`
    /// counting them (zero before the launch). Each element kernel then runs
    /// over its own kind only: in a pass's own order the kinds share warps,
    /// and a warp with one subcell element pays the whole subcell chain on
    /// the FP64 units (Chiloé: 20 % of the elements on the subcells kept 48 %
    /// of the line warps and 84 % of the element warps busy). One atomic per
    /// warp and kind; the order within a list is the atomics', which no
    /// result depends on (each element writes only its own rows).
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn split_elements(
        subcell_depth: f64,
        list: &[u32],
        n: u32,
        h: &[f64],
        counts: *mut u32,
        mut wet: DisjointSlice<u32>,
        mut shallow: DisjointSlice<u32>,
    ) {
        // Every lane reaches the votes: no early return
        let k = item(thread::index_1d().get(), list, n);
        let (is_wet, is_shallow) = match k {
            Some(k) => {
                let sub = is_subcell_element(k, h, subcell_depth);
                (!sub, sub)
            }
            None => (false, false),
        };
        let k = k.unwrap_or(0) as u32;
        let (below, lane) = (warp::lanemask_lt(), warp::lane_id());
        let wet_mask = warp::ballot(is_wet);
        let shallow_mask = warp::ballot(is_shallow);
        // The warp's first lane of each kind reserves the warp's slots
        let mut base = [0u32; 2];
        if wet_mask != 0 && lane == wet_mask.trailing_zeros() {
            // SAFETY: `counts` holds two u32 that only atomics touch here.
            let count = unsafe { DeviceAtomicU32::from_ptr(counts.add(WET)) };
            base[WET] = count.fetch_add(wet_mask.count_ones(), AtomicOrdering::Relaxed);
        }
        if shallow_mask != 0 && lane == shallow_mask.trailing_zeros() {
            // SAFETY: as above.
            let count = unsafe { DeviceAtomicU32::from_ptr(counts.add(SHALLOW)) };
            base[SHALLOW] = count.fetch_add(shallow_mask.count_ones(), AtomicOrdering::Relaxed);
        }
        let wet_base = warp::shuffle(base[WET], wet_mask.trailing_zeros() & 31);
        let shallow_base = warp::shuffle(base[SHALLOW], shallow_mask.trailing_zeros() & 31);
        // SAFETY: each lane of a kind writes the slot its warp reserved plus
        // its rank among the warp's lanes of that kind: distinct slots, within
        // the pass's n ≤ the lists' length.
        unsafe {
            if is_wet {
                *wet.get_unchecked_mut((wet_base + (wet_mask & below).count_ones()) as usize) = k;
            }
            if is_shallow {
                *shallow.get_unchecked_mut(
                    (shallow_base + (shallow_mask & below).count_ones()) as usize,
                ) = k;
            }
        }
    }

    /// `SplitFormSWE2D::element_rhs` of the wet elements of a pass (`list`,
    /// `split_elements`' first `counts[WET]`: no node shallower than the
    /// subcell depth; one thread per element; the others are
    /// `element_rhs_subcells`'):
    /// the flux-differencing volume term and the lifted face terms, divided
    /// by J. One thread per element, not per line as the subcells: split
    /// in two kernels through a buffer of each line's terms (2026-11-16), a
    /// short pass took 25 + 34 µs against 66, but Chiloé's half hour 27.7 s
    /// against 25.6 and the fully wet RHS 1.5× as long (`notes/gpu.md`).
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn element_rhs_volume(
        g: f64,
        h_min: f64,
        list: &[u32],
        counts: &[u32],
        h: &[f64],
        hu: &[f64],
        hv: &[f64],
        bed: &[f64],
        affine: &[u32],
        egeom: &[[f64; EG]],
        contra: &[[f64; 4]],
        jinv: &[f64],
        fgeom: &[[f64; 3]],
        d1: &[f64],
        lift: &[f64],
        element_edges: &[[u32; 4]],
        edges: &[[u32; 4]],
        faces: &[[[f64; 3]; NF]],
        coriolis: u32,
        f0: f64,
        f_node: &[f64],
        mut out_h: DisjointSlice<f64>,
        mut out_hu: DisjointSlice<f64>,
        mut out_hv: DisjointSlice<f64>,
    ) {
        let Some(k) = split_item(thread::index_1d().get(), list, counts, WET) else {
            return;
        };
        let mut nodes = [SWENodeState2D::default(); NN];
        #[unroll]
        for i in 0..NN {
            nodes[i] = node_state(h, hu, hv, bed, k * NN + i, h_min);
        }
        let mut rhs = [SWEState2D::zero(); NN];

        let aff = affine[k] != 0;
        let eg = egeom[k];
        let (affine_r, affine_s) = ((eg[0], eg[1]), (eg[2], eg[3]));
        let mut metric = [(0.0, 0.0); N1];
        #[unroll]
        for line in 0..N1 {
            let r_line = |a: usize| line * N1 + a;
            let s_line = |j: usize| j * N1 + line;
            #[unroll]
            for a in 0..N1 {
                metric[a] = if aff {
                    affine_r
                } else {
                    let c = contra[k * NN + r_line(a)];
                    (c[0], c[1])
                };
            }
            if aff {
                line_volume::<true, _>(&nodes, &mut rhs, &metric, r_line, affine_r, d1, g);
            } else {
                line_volume::<false, _>(&nodes, &mut rhs, &metric, r_line, affine_r, d1, g);
            }
            #[unroll]
            for j in 0..N1 {
                metric[j] = if aff {
                    affine_s
                } else {
                    let c = contra[k * NN + s_line(j)];
                    (c[2], c[3])
                };
            }
            if aff {
                line_volume::<true, _>(&nodes, &mut rhs, &metric, s_line, affine_s, d1, g);
            } else {
                line_volume::<false, _>(&nodes, &mut rhs, &metric, s_line, affine_s, d1, g);
            }
        }

        finish_element(
            k,
            &nodes,
            rhs,
            g,
            aff,
            &eg,
            jinv,
            fgeom,
            lift,
            element_edges,
            edges,
            faces,
            (coriolis, f0, f_node),
            &mut out_h,
            &mut out_hu,
            &mut out_hv,
        );
    }

    /// The subcell terms of the subcell elements of a pass (`list`,
    /// `split_elements`' first `counts[SHALLOW]`: a node shallower than the
    /// subcell depth): one thread per
    /// element and GLL line (`2·N1` per element, line `l` of direction `d`
    /// the element's `2l + d`), `line_terms` of the line into `terms`
    /// (`LINE_TERMS` per node, at `((k·2N1 + 2l + d)·N1 + a)·LINE_TERMS`),
    /// which `element_rhs_subcells` sums. A line is a sixth of an element's
    /// subcell work: the chain each thread runs, which bounds a short pass
    /// of local time stepping, is a sixth as long.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn subcell_terms(
        g: f64,
        h_min: f64,
        h_dry: f64,
        list: &[u32],
        counts: &[u32],
        h: &[f64],
        hu: &[f64],
        hv: &[f64],
        bed: &[f64],
        affine: &[u32],
        egeom: &[[f64; EG]],
        contra: &[[f64; 4]],
        outer_distance: &[f64],
        d1: &[f64],
        w: &[f64],
        xi: &[f64],
        element_edges: &[[u32; 4]],
        edges: &[[u32; 4]],
        mut terms: DisjointSlice<[f64; 3]>,
    ) {
        let t = thread::index_1d().get();
        let (i, j) = (t / (2 * N1), t % (2 * N1));
        let Some(k) = split_item(i, list, counts, SHALLOW) else {
            return;
        };
        let (line, direction) = (j / 2, j % 2);
        let node = |a: usize| {
            if direction == 0 {
                line * N1 + a
            } else {
                a * N1 + line
            }
        };
        // The GLL weights, nodes and D as arrays: constant lengths for the
        // loops of the subcells (telescoped_interfaces reads their length)
        let (mut w_a, mut xi_a, mut d1_a) = ([0.0; N1], [0.0; N1], [0.0; NN]);
        #[unroll]
        for a in 0..N1 {
            w_a[a] = w[a];
            xi_a[a] = xi[a];
        }
        #[unroll]
        for a in 0..NN {
            d1_a[a] = d1[a];
        }
        let aff = affine[k] != 0;
        let eg = &egeom[k];
        let mut q = [SWENodeState2D::default(); N1];
        let mut metric = [(0.0, 0.0); N1];
        #[unroll]
        for a in 0..N1 {
            q[a] = node_state(h, hu, hv, bed, k * NN + node(a), h_min);
            metric[a] = if aff {
                if direction == 0 {
                    (eg[0], eg[1])
                } else {
                    (eg[2], eg[3])
                }
            } else {
                let c = contra[k * NN + node(a)];
                if direction == 0 {
                    (c[0], c[1])
                } else {
                    (c[2], c[3])
                }
            };
        }
        let outer = |face: usize, fi: usize| {
            outer_node(
                k,
                face,
                fi,
                (h, hu, hv, bed),
                outer_distance,
                element_edges,
                edges,
                h_min,
            )
        };
        // Line ends on faces 3/1 (r-lines) and 0/2 (s-lines); faces 2 and 3
        // list their nodes in reverse
        let (rev, fwd) = (N1 - 1 - line, line);
        let ends = if direction == 0 {
            [outer(3, rev), outer(1, fwd)]
        } else {
            [outer(0, fwd), outer(2, rev)]
        };
        let line_t = line_terms(&q, &metric, ends, aff, &w_a, &xi_a, &d1_a, g, h_min, h_dry);
        let first = (k * 2 * N1 + j) * N1 * LINE_TERMS;
        #[unroll]
        for a in 0..N1 {
            #[unroll]
            for t in 0..LINE_TERMS {
                let s = line_t[a][t];
                // SAFETY: thread (k, j) writes only its own line's terms
                // (the listed elements are distinct), within the buffer's
                // n_elements · 2N1 · N1 · LINE_TERMS.
                unsafe {
                    *terms.get_unchecked_mut(first + a * LINE_TERMS + t) = [s.h, s.hu, s.hv];
                }
            }
        }
    }

    /// `SplitFormSWE2D::element_rhs` of the subcell elements of a pass
    /// (`list`, `split_elements`' first `counts[SHALLOW]`; one thread per
    /// element; the others are `element_rhs_volume`'s): the
    /// subcell terms of its lines (`subcell_terms`) summed into each node in
    /// the CPU's order (line by line, its r-line then its s-line, a node's
    /// terms in turn), then the lifted face terms, divided by J, and the
    /// Coriolis force.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn element_rhs_subcells(
        g: f64,
        h_min: f64,
        list: &[u32],
        counts: &[u32],
        h: &[f64],
        hu: &[f64],
        hv: &[f64],
        bed: &[f64],
        affine: &[u32],
        egeom: &[[f64; EG]],
        jinv: &[f64],
        fgeom: &[[f64; 3]],
        lift: &[f64],
        element_edges: &[[u32; 4]],
        edges: &[[u32; 4]],
        faces: &[[[f64; 3]; NF]],
        terms: &[[f64; 3]],
        coriolis: u32,
        f0: f64,
        f_node: &[f64],
        mut out_h: DisjointSlice<f64>,
        mut out_hu: DisjointSlice<f64>,
        mut out_hv: DisjointSlice<f64>,
    ) {
        let Some(k) = split_item(thread::index_1d().get(), list, counts, SHALLOW) else {
            return;
        };
        let mut nodes = [SWENodeState2D::default(); NN];
        #[unroll]
        for i in 0..NN {
            nodes[i] = node_state(h, hu, hv, bed, k * NN + i, h_min);
        }
        let mut rhs = [SWEState2D::zero(); NN];
        #[unroll]
        for line in 0..N1 {
            #[unroll]
            for direction in 0..2 {
                let first = (k * 2 * N1 + 2 * line + direction) * N1 * LINE_TERMS;
                #[unroll]
                for a in 0..N1 {
                    let node = if direction == 0 {
                        line * N1 + a
                    } else {
                        a * N1 + line
                    };
                    #[unroll]
                    for t in 0..LINE_TERMS {
                        let [th, thu, thv] = terms[first + a * LINE_TERMS + t];
                        rhs[node] = rhs[node] + SWEState2D::new(th, thu, thv);
                    }
                }
            }
        }

        let aff = affine[k] != 0;
        let eg = egeom[k];
        finish_element(
            k,
            &nodes,
            rhs,
            g,
            aff,
            &eg,
            jinv,
            fgeom,
            lift,
            element_edges,
            edges,
            faces,
            (coriolis, f0, f_node),
            &mut out_h,
            &mut out_hu,
            &mut out_hv,
        );
    }

    /// `apply_wet_dry_correction_all` of element `k` (one thread per
    /// element): dgcom's `wet_dry_element` on its rows, with its nodal
    /// masses and 1/area; an element it empties is counted in `clips`.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn wet_dry_correction(
        h_dry: f64,
        relaxation_time: f64,
        max_velocity: f64,
        g: f64,
        mass: &[f64],
        inv_area: &[f64],
        mut h: DisjointSlice<f64>,
        mut hu: DisjointSlice<f64>,
        mut hv: DisjointSlice<f64>,
        mut clips: DisjointSlice<u32>,
    ) {
        let k = thread::index_1d().get();
        if k >= inv_area.len() {
            return;
        }
        let config = WetDryConfig {
            h_dry: Depth::new(h_dry),
            relaxation_time,
            max_velocity,
            g,
        };
        let (mut eh, mut ehu, mut ehv, mut m) = ([0.0; NN], [0.0; NN], [0.0; NN], [0.0; NN]);
        #[unroll]
        for i in 0..NN {
            let n = k * NN + i;
            // SAFETY: thread k reads and writes only its own element's rows.
            unsafe {
                eh[i] = *h.get_unchecked_mut(n);
                ehu[i] = *hu.get_unchecked_mut(n);
                ehv[i] = *hv.get_unchecked_mut(n);
            }
            m[i] = mass[n];
        }
        if wet_dry_element(&mut eh, &mut ehu, &mut ehv, &m, inv_area[k], &config) {
            // SAFETY: thread k writes only its own element's count.
            unsafe {
                *clips.get_unchecked_mut(k) += 1;
            }
        }
        #[unroll]
        for i in 0..NN {
            let n = k * NN + i;
            // SAFETY: as above.
            unsafe {
                *h.get_unchecked_mut(n) = eh[i];
                *hu.get_unchecked_mut(n) = ehu[i];
                *hv.get_unchecked_mut(n) = ehv[i];
            }
        }
    }

    /// `apply_implicit_damping_2d` of node `n` (one thread per node):
    /// dgcom's `damp_node` with Manning friction and the thin-layer
    /// relaxation as `damping` says, no cages.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn implicit_damping(
        dt: f64,
        damping: Damping,
        h: &[f64],
        from_h: &[f64],
        from_hu: &[f64],
        from_hv: &[f64],
        mut hu: DisjointSlice<f64>,
        mut hv: DisjointSlice<f64>,
    ) {
        let n = thread::index_1d().get();
        if n >= h.len() {
            return;
        }
        // SAFETY: thread n reads and writes only node n.
        let (mut u, mut v) = unsafe { (*hu.get_unchecked_mut(n), *hv.get_unchecked_mut(n)) };
        let from = SWEState2D::new(from_h[n], from_hu[n], from_hv[n]);
        damping.node(n, h[n], &mut u, &mut v, from, dt);
        // SAFETY: as above.
        unsafe {
            *hu.get_unchecked_mut(n) = u;
            *hv.get_unchecked_mut(n) = v;
        }
    }

    /// `SWEPhysics2D::stage_where`'s stage of every listed element k (every
    /// element if `list` is empty; one thread per element), whose RHS is in
    /// its rows of `out`: the stage `element_stage` gives it (its level and,
    /// for the last stage, its rate), `out = a·base + b·input + c·F`, the
    /// damping over `c`, then either `acc ← w·out + keep·acc` (the last
    /// stage) or the wet/dry correction. A line-local element that went
    /// shallower than `shallow` in a stage value the next stage reads is
    /// counted in `breaks`.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn stage_elements(
        t: f64,
        dt: f64,
        finest: u32,
        substep: u32,
        depth: u32,
        last: u32,
        start: f64,
        input: f64,
        beta: f64,
        c: f64,
        level: &[u32],
        rate: &[u32],
        list: &[u32],
        n: u32,
        base_h: &[f64],
        base_hu: &[f64],
        base_hv: &[f64],
        in_h: &[f64],
        in_hu: &[f64],
        in_hv: &[f64],
        damping: Damping,
        line_local: &[u32],
        shallow: f64,
        mass: &[f64],
        inv_area: &[f64],
        mut out_h: DisjointSlice<f64>,
        mut out_hu: DisjointSlice<f64>,
        mut out_hv: DisjointSlice<f64>,
        mut acc_h: DisjointSlice<f64>,
        mut acc_hu: DisjointSlice<f64>,
        mut acc_hv: DisjointSlice<f64>,
        mut clips: DisjointSlice<u32>,
        mut breaks: DisjointSlice<u32>,
    ) {
        let Some(k) = item(thread::index_1d().get(), list, n) else {
            return;
        };
        let stage = element_stage(
            (t, dt),
            finest as usize,
            substep as usize,
            depth as usize,
            ShuOsherStage {
                start,
                input,
                beta,
                c,
            },
            last != 0,
            level_of(level, k) as u8,
            level_of(rate, k) as u8,
        );
        let (mut eh, mut ehu, mut ehv) = ([0.0; NN], [0.0; NN], [0.0; NN]);
        // out = a·base + b·input + c·F, F in the rows (as scale + axpy
        // round it)
        let combine = |s: f64, x: f64, f: f64| {
            let mut v = if stage.a != 0.0 { stage.a * s } else { 0.0 };
            if stage.b != 0.0 {
                v += stage.b * x;
            }
            if stage.c != 0.0 {
                v += stage.c * f;
            }
            v
        };
        #[unroll]
        for i in 0..NN {
            let m = k * NN + i;
            // SAFETY: thread k reads and writes only its own element's rows
            // (the listed elements are distinct).
            let (fh, fhu, fhv) = unsafe {
                (
                    *out_h.get_unchecked_mut(m),
                    *out_hu.get_unchecked_mut(m),
                    *out_hv.get_unchecked_mut(m),
                )
            };
            eh[i] = combine(base_h[m], in_h[m], fh);
            ehu[i] = combine(base_hu[m], in_hu[m], fhu);
            ehv[i] = combine(base_hv[m], in_hv[m], fhv);
        }
        if damping.is_active() {
            #[unroll]
            for i in 0..NN {
                let m = k * NN + i;
                let from = SWEState2D::new(in_h[m], in_hu[m], in_hv[m]);
                damping.node(m, eh[i], &mut ehu[i], &mut ehv[i], from, stage.c);
            }
        }
        if stage.accumulate.is_none() && !line_local.is_empty() && line_local[k] != 0 {
            let mut shallower = false;
            #[unroll]
            for i in 0..NN {
                shallower |= eh[i] < shallow;
            }
            if shallower {
                // SAFETY: thread k writes only its own element's count.
                unsafe {
                    *breaks.get_unchecked_mut(k) += 1;
                }
            }
        }
        match stage.accumulate {
            Some((w, keep)) => {
                #[unroll]
                for i in 0..NN {
                    let m = k * NN + i;
                    // SAFETY: as above, the element's rows of the sum.
                    unsafe {
                        let sum = |s: &mut f64, v: f64| {
                            *s = if keep != 0.0 {
                                w * v + keep * *s
                            } else {
                                w * v
                            };
                        };
                        sum(acc_h.get_unchecked_mut(m), eh[i]);
                        sum(acc_hu.get_unchecked_mut(m), ehu[i]);
                        sum(acc_hv.get_unchecked_mut(m), ehv[i]);
                    }
                }
            }
            None => {
                if damping.wet_dry != 0 {
                    let mut em = [0.0; NN];
                    #[unroll]
                    for i in 0..NN {
                        em[i] = mass[k * NN + i];
                    }
                    let config = damping.wet_dry_config();
                    if wet_dry_element(&mut eh, &mut ehu, &mut ehv, &em, inv_area[k], &config) {
                        // SAFETY: thread k writes only its own element's count.
                        unsafe {
                            *clips.get_unchecked_mut(k) += 1;
                        }
                    }
                }
            }
        }
        #[unroll]
        for i in 0..NN {
            let m = k * NN + i;
            // SAFETY: as above.
            unsafe {
                *out_h.get_unchecked_mut(m) = eh[i];
                *out_hu.get_unchecked_mut(m) = ehu[i];
                *out_hv.get_unchecked_mut(m) = ehv[i];
            }
        }
    }

    /// `SWEPhysics2D::finish_where` of every listed element k (one thread
    /// per element): its rows of the state from `acc`, the line-local check
    /// as in `stage_elements`, then the wet/dry correction (if `wet_dry`).
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn finish_elements(
        list: &[u32],
        n: u32,
        acc_h: &[f64],
        acc_hu: &[f64],
        acc_hv: &[f64],
        damping: Damping,
        line_local: &[u32],
        shallow: f64,
        mass: &[f64],
        inv_area: &[f64],
        mut h: DisjointSlice<f64>,
        mut hu: DisjointSlice<f64>,
        mut hv: DisjointSlice<f64>,
        mut clips: DisjointSlice<u32>,
        mut breaks: DisjointSlice<u32>,
    ) {
        let Some(k) = item(thread::index_1d().get(), list, n) else {
            return;
        };
        let (mut eh, mut ehu, mut ehv, mut em) = ([0.0; NN], [0.0; NN], [0.0; NN], [0.0; NN]);
        let mut shallower = false;
        #[unroll]
        for i in 0..NN {
            let m = k * NN + i;
            eh[i] = acc_h[m];
            ehu[i] = acc_hu[m];
            ehv[i] = acc_hv[m];
            em[i] = mass[m];
            if eh[i] < shallow {
                shallower = true;
            }
        }
        if shallower && !line_local.is_empty() && line_local[k] != 0 {
            // SAFETY: thread k writes only its own element's count.
            unsafe {
                *breaks.get_unchecked_mut(k) += 1;
            }
        }
        if damping.wet_dry != 0 {
            let config = damping.wet_dry_config();
            if wet_dry_element(&mut eh, &mut ehu, &mut ehv, &em, inv_area[k], &config) {
                // SAFETY: as above.
                unsafe {
                    *clips.get_unchecked_mut(k) += 1;
                }
            }
        }
        #[unroll]
        for i in 0..NN {
            let m = k * NN + i;
            // SAFETY: thread k writes only its own element's rows (the
            // listed elements are distinct).
            unsafe {
                *h.get_unchecked_mut(m) = eh[i];
                *hu.get_unchecked_mut(m) = ehu[i];
                *hv.get_unchecked_mut(m) = ehv[i];
            }
        }
    }

    /// Bit j of element k's flag: every node of k deeper than `dj`
    /// (one thread per element; for `line_local` and `coarse_step_bound`).
    #[kernel]
    pub fn depth_flags(d0: f64, d1: f64, d2: f64, h: &[f64], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let k = idx.get();
        let depths = [d0, d1, d2];
        if let Some(flags) = out.get_mut(idx) {
            let mut bits = 0u32;
            #[unroll]
            for j in 0..3 {
                // Every node deeper (false at a NaN, as `all(h > depth)`)
                let mut deeper = true;
                #[unroll]
                for i in 0..NN {
                    deeper &= h[k * NN + i] > depths[j];
                }
                bits |= u32::from(deeper) << j;
            }
            *flags = bits;
        }
    }

    /// `element_dt` of `min_element_dt_swe_2d` (one thread per element): the
    /// largest reference rate of the element's nodes and of its neighbours'
    /// face nodes (with the element's metric there), and the CFL capped by
    /// the positivity bound (if `positive`).
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn element_dt(
        g: f64,
        h_min: f64,
        cfl: f64,
        order: u32,
        positive: u32,
        bound_cfl: f64,
        bound_max_cfl: f64,
        bound_max_cfl_subcells: f64,
        bound_h_dry: f64,
        h: &[f64],
        hu: &[f64],
        hv: &[f64],
        affine: &[u32],
        affine_grads: &[[f64; 4]],
        grads: &[[f64; 4]],
        mass: &[f64],
        element_edges: &[[u32; 4]],
        edges: &[[u32; 4]],
        mut out: DisjointSlice<f64>,
    ) {
        let idx = thread::index_1d();
        let k = idx.get();
        if let Some(dt) = out.get_mut(idx) {
            let h_min_depth = Depth::new(h_min);
            // element_max_reference_rate
            let aff = affine[k] != 0;
            let affine_metric = rate_metric(affine_grads[k]);
            let mut rate = 0.0f64;
            #[unroll]
            for i in 0..NN {
                let n = k * NN + i;
                let state = SWEState2D::new(h[n], hu[n], hv[n]);
                let r = if state.h <= h_min {
                    0.0
                } else {
                    let metric = if aff {
                        affine_metric
                    } else {
                        rate_metric(grads[n])
                    };
                    node_reference_rate(&state, metric, g, h_min_depth)
                };
                rate = rate.max(r);
            }
            // The neighbours' face nodes, with this element's metric there
            #[unroll]
            for face in 0..4 {
                let edge = edges[element_edges[k][face] as usize];
                if edge[2] != NO_NEIGHBOUR {
                    let (nb, nb_face) = if edge[0] as usize == k && edge[1] as usize == face {
                        (edge[2] as usize, edge[3] as usize)
                    } else {
                        (edge[0] as usize, edge[1] as usize)
                    };
                    #[unroll]
                    for fi in 0..N1 {
                        let i = face_node(face, fi);
                        let n = nb * NN + face_node(nb_face, N1 - 1 - fi);
                        let state = SWEState2D::new(h[n], hu[n], hv[n]);
                        if state.h > h_min {
                            let metric = rate_metric(grads[k * NN + i]);
                            rate = rate.max(node_reference_rate(&state, metric, g, h_min_depth));
                        }
                    }
                }
            }
            let cfl = if positive != 0 {
                let bound = PositivityBound {
                    cfl: bound_cfl,
                    max_cfl: bound_max_cfl,
                    max_cfl_subcells: bound_max_cfl_subcells,
                    h_dry: bound_h_dry,
                };
                let (mut eh, mut m) = ([0.0; NN], [0.0; NN]);
                #[unroll]
                for i in 0..NN {
                    eh[i] = h[k * NN + i];
                    m[i] = mass[k * NN + i];
                }
                cfl.min(bound.element_cfl(&eh, &m, N1))
            } else {
                cfl
            };
            *dt = dt_from_reference_rate(rate, order as usize, cfl);
        }
    }

    /// `Integrable::combine` with a base: `out = a·base + c₁x₁ (+ c₂x₂)`,
    /// in the order of `combine_values` (copy, scale unless `a = 1`, one
    /// multiply–add per term).
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn combine_from(
        a: f64,
        terms: u32,
        c1: f64,
        c2: f64,
        base: &[f64],
        x1: &[f64],
        x2: &[f64],
        mut out: DisjointSlice<f64>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(v) = out.get_mut(idx) {
            let mut value = base[i];
            if a != 1.0 {
                value *= a;
            }
            if terms > 0 {
                value += c1 * x1[i];
            }
            if terms > 1 {
                value += c2 * x2[i];
            }
            *v = value;
        }
    }

    /// `Integrable::combine` without a base: `out = a·out + c₁x₁ (+ c₂x₂)`.
    #[kernel]
    pub fn combine_in_place(
        a: f64,
        terms: u32,
        c1: f64,
        c2: f64,
        x1: &[f64],
        x2: &[f64],
        mut out: DisjointSlice<f64>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(v) = out.get_mut(idx) {
            let mut value = *v;
            if a != 1.0 {
                value *= a;
            }
            if terms > 0 {
                value += c1 * x1[i];
            }
            if terms > 1 {
                value += c2 * x2[i];
            }
            *v = value;
        }
    }

    /// `element_max_reference_rate`: the largest `node_reference_rate` of
    /// element `k`'s wet nodes (zero at the dry ones), one thread per element.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn element_rates(
        g: f64,
        h_min: f64,
        h: &[f64],
        hu: &[f64],
        hv: &[f64],
        affine: &[u32],
        affine_grads: &[[f64; 4]],
        grads: &[[f64; 4]],
        mut out: DisjointSlice<f64>,
    ) {
        let idx = thread::index_1d();
        let k = idx.get();
        if let Some(rate) = out.get_mut(idx) {
            let aff = affine[k] != 0;
            let affine_metric = rate_metric(affine_grads[k]);
            let mut max = 0.0f64;
            #[unroll]
            for i in 0..NN {
                let n = k * NN + i;
                let state = SWEState2D::new(h[n], hu[n], hv[n]);
                let r = if state.h <= h_min {
                    0.0
                } else {
                    let metric = if aff {
                        affine_metric
                    } else {
                        rate_metric(grads[n])
                    };
                    node_reference_rate(&state, metric, g, Depth::new(h_min))
                };
                max = max.max(r);
            }
            *rate = max;
        }
    }

    /// 1 where element `k` has a value that is not finite, else 0.
    #[kernel]
    pub fn element_nonfinite(h: &[f64], hu: &[f64], hv: &[f64], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let k = idx.get();
        if let Some(flag) = out.get_mut(idx) {
            let mut bad = 0u32;
            #[unroll]
            for i in 0..NN {
                let n = k * NN + i;
                if !(h[n].is_finite() && hu[n].is_finite() && hv[n].is_finite()) {
                    bad = 1;
                }
            }
            *flag = bad;
        }
    }
}

// =============================================================================
// Host: the device, the static data, the state
// =============================================================================

/// The GPU: the stream every operation is ordered on (it holds the context),
/// and the kernels.
pub struct Gpu {
    pub stream: Arc<Stream>,
    module: kernels::LoadedModule,
    /// The 3D kernels (`three_d.rs`)
    pub(crate) module3d: crate::three_d::kernels3d::LoadedModule,
    /// The 3D momentum terms' kernels (`baroclinic.rs`)
    pub(crate) module_baroclinic: crate::baroclinic::kernels_baroclinic::LoadedModule,
    /// The 3D shear's viscosity (`viscosity.rs`)
    pub(crate) module_viscosity: crate::viscosity::kernels_viscosity::LoadedModule,
    /// The 3D stages' values and hooks (`hooks.rs`)
    pub(crate) module_hooks: crate::hooks::kernels_hooks::LoadedModule,
    /// The 3D tracer limiter (`limiter.rs`)
    pub(crate) module_limiter: crate::limiter::kernels_limiter::LoadedModule,
    /// The device's name and compute capability
    pub name: String,
    /// Its streaming multiprocessors
    sms: u32,
}

impl Gpu {
    /// CUDA device `ordinal`, its default stream and the kernels.
    pub fn on(ordinal: usize) -> Result<Arc<Self>, Box<dyn std::error::Error>> {
        let ctx = CudaContext::new(ordinal)?;
        let (major, minor) = ctx.compute_capability()?;
        let name = format!("{} (sm_{major}{minor})", ctx.device_name()?);
        let stream = ctx.default_stream();
        let module = kernels::load(&ctx)?;
        let module3d = crate::three_d::kernels3d::load(&ctx)?;
        let module_baroclinic = crate::baroclinic::kernels_baroclinic::load(&ctx)?;
        let module_viscosity = crate::viscosity::kernels_viscosity::load(&ctx)?;
        let module_hooks = crate::hooks::kernels_hooks::load(&ctx)?;
        let module_limiter = crate::limiter::kernels_limiter::load(&ctx)?;
        let sms = ctx.multiprocessor_count()?;
        Ok(Arc::new(Self {
            stream,
            module,
            module3d,
            module_baroclinic,
            module_viscosity,
            module_hooks,
            module_limiter,
            name,
            sms,
        }))
    }

    /// A launch over `n` threads in blocks of `max_block`, halved (down to
    /// a warp) while the grid would give fewer than two blocks per SM. A
    /// short pass of local time stepping (a few hundred elements) is bound by
    /// the FP64 rate of the SMs it occupies, and a block runs on one: 128
    /// elements in one block took Chiloé's volume kernel 70 µs, as long as
    /// 4,352 in 34 blocks (2026-11-16, RTX 3090: 2 FP64 units per SM).
    fn spread(&self, n: usize, max_block: u32) -> LaunchConfig {
        let n = n as u32;
        let mut block = max_block;
        while block > 32 && n.div_ceil(block) < 2 * self.sms {
            block /= 2;
        }
        LaunchConfig {
            grid_dim: (n.div_ceil(block).max(1), 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        }
    }
}

fn one_dimensional(n: usize) -> LaunchConfig {
    LaunchConfig::for_num_elems(n as u32)
}

fn per_element(n_elements: usize) -> LaunchConfig {
    let n = n_elements as u32;
    LaunchConfig {
        grid_dim: (n.div_ceil(ELEMENT_BLOCK), 1, 1),
        block_dim: (ELEMENT_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// The SWE state `[h, hu, hv]` on the device (`[k·NN + i]` per field, as
/// `SWESolution2D`). Every operation is enqueued on the GPU's stream.
pub struct DeviceState {
    pub h: DeviceBuffer<f64>,
    pub hu: DeviceBuffer<f64>,
    pub hv: DeviceBuffer<f64>,
    gpu: Arc<Gpu>,
}

impl DeviceState {
    /// `q` on the device.
    pub fn upload(gpu: &Arc<Gpu>, q: &SWESolution2D) -> Self {
        assert_eq!(q.n_nodes, NN, "the kernels are compiled for P{ORDER}");
        let up = |field: &[f64]| DeviceBuffer::from_host(&gpu.stream, field).expect("upload");
        Self {
            h: up(&q.data[0]),
            hu: up(&q.data[1]),
            hv: up(&q.data[2]),
            gpu: gpu.clone(),
        }
    }

    /// The state on the host.
    pub fn download(&self) -> SWESolution2D {
        let n_elements = self.h.len() / NN;
        let mut q = SWESolution2D::new(n_elements, NN);
        for (field, buffer) in q.data.iter_mut().zip([&self.h, &self.hu, &self.hv]) {
            buffer
                .copy_to_host(&self.gpu.stream, field)
                .expect("download");
        }
        q
    }

    fn fields(&self) -> [&DeviceBuffer<f64>; 3] {
        [&self.h, &self.hu, &self.hv]
    }

    fn fields_mut(&mut self) -> [&mut DeviceBuffer<f64>; 3] {
        [&mut self.h, &mut self.hu, &mut self.hv]
    }

    fn zeroed_like(&self) -> Self {
        let zero = || DeviceBuffer::zeroed(&self.gpu.stream, self.h.len()).expect("allocation");
        Self {
            h: zero(),
            hu: zero(),
            hv: zero(),
            gpu: self.gpu.clone(),
        }
    }

    /// `combine` of up to two terms, field by field, on the device.
    fn combine_terms(&mut self, base: Option<&Self>, a: f64, terms: &[(f64, &Self)]) {
        assert!(terms.len() <= 2, "the device combines at most two terms");
        if base.is_none() && terms.is_empty() && a == 1.0 {
            return;
        }
        let gpu = self.gpu.clone();
        let cfg = one_dimensional(self.h.len());
        let n_terms = terms.len() as u32;
        let c = |j: usize| terms.get(j).map_or(0.0, |t| t.0);
        // The kernels read a term's slot only if it is used
        let empty = DeviceBuffer::<f64>::zeroed(&gpu.stream, 0).expect("allocation");
        for f in 0..3 {
            let x = |j: usize| terms.get(j).map_or(&empty, |t| t.1.fields()[f]);
            let out = match f {
                0 => &mut self.h,
                1 => &mut self.hu,
                _ => &mut self.hv,
            };
            // SAFETY: every buffer is the state's length (or unused); `out`
            // is none of the inputs (`&mut self` against `&Self`).
            unsafe {
                match base {
                    Some(base) => gpu.module.combine_from(
                        &gpu.stream,
                        cfg,
                        a,
                        n_terms,
                        c(0),
                        c(1),
                        base.fields()[f],
                        x(0),
                        x(1),
                        out,
                    ),
                    None => gpu.module.combine_in_place(
                        &gpu.stream,
                        cfg,
                        a,
                        n_terms,
                        c(0),
                        c(1),
                        x(0),
                        x(1),
                        out,
                    ),
                }
            }
            .expect("combine");
        }
    }
}

impl Clone for DeviceState {
    fn clone(&self) -> Self {
        let mut copy = self.zeroed_like();
        copy.copy_from(self);
        copy
    }
}

impl Integrable for DeviceState {
    fn scale(&mut self, c: f64) {
        self.combine_terms(None, c, &[]);
    }

    fn axpy(&mut self, c: f64, other: &Self) {
        self.combine_terms(None, 1.0, &[(c, other)]);
    }

    fn zeros_like(&self) -> Self {
        self.zeroed_like()
    }

    fn copy_from(&mut self, other: &Self) {
        let stream = self.gpu.stream.clone();
        for (to, from) in self.fields_mut().into_iter().zip(other.fields()) {
            to.copy_from_device_async(from, &stream)
                .expect("device copy");
        }
    }

    fn combine(&mut self, base: Option<&Self>, a: f64, terms: &[(f64, &Self)]) {
        if terms.len() <= 2 {
            self.combine_terms(base, a, terms);
            return;
        }
        // The plain sequence (the SSP schemes never need it)
        if let Some(base) = base {
            self.copy_from(base);
        }
        if a != 1.0 {
            self.scale(a);
        }
        for &(c, x) in terms {
            self.axpy(c, x);
        }
    }

    fn is_finite(&self) -> bool {
        let n_elements = self.h.len() / NN;
        let gpu = &self.gpu;
        let mut flags = DeviceBuffer::<u32>::zeroed(&gpu.stream, n_elements).expect("allocation");
        // SAFETY: one flag per element, the state's length.
        unsafe {
            gpu.module.element_nonfinite(
                &gpu.stream,
                one_dimensional(n_elements),
                &self.h,
                &self.hu,
                &self.hv,
                &mut flags,
            )
        }
        .expect("is_finite");
        let flags = flags.to_host_vec(&gpu.stream).expect("download");
        flags.iter().all(|&f| f == 0)
    }
}

/// The boundary edges of `mesh` (`boundary`) or its interior edges, in
/// order.
fn edge_kinds(mesh: &Mesh2D, boundary: bool) -> Vec<u32> {
    (0..mesh.n_edges as u32)
        .filter(|&e| mesh.edges[e as usize].right.is_none() == boundary)
        .collect()
}

/// The static data of a mesh, packed as the kernels read them, on the device.
struct Statics {
    bed: DeviceBuffer<f64>,
    affine: DeviceBuffer<u32>,
    egeom: DeviceBuffer<[f64; EG]>,
    contra: DeviceBuffer<[f64; 4]>,
    jinv: DeviceBuffer<f64>,
    fgeom: DeviceBuffer<[f64; 3]>,
    /// Per face node (`(k·4 + face)·N1 + fi`): the distance of the
    /// neighbour's first interior node on the GLL line through it from the
    /// face, in `k`'s reference coordinate (zero at a boundary): static, so
    /// `outer_node` reads it rather than divide three times per line end
    outer_distance: DeviceBuffer<f64>,
    d1: DeviceBuffer<f64>,
    /// The GLL weights and nodes on [−1, 1]
    w: DeviceBuffer<f64>,
    xi: DeviceBuffer<f64>,
    lift: DeviceBuffer<f64>,
    element_edges: DeviceBuffer<[u32; 4]>,
    /// Left element and face, right element and face (`NO_NEIGHBOUR` at a
    /// wall)
    edges: DeviceBuffer<[u32; 4]>,
    /// The interior edges and the boundary edges, each in order (the edge
    /// lists of global steps)
    interior_edges: DeviceBuffer<u32>,
    boundary_edges: DeviceBuffer<u32>,
    /// `[r_x, r_y, s_x, s_y]` of every node, and of every parallelogram
    grads: DeviceBuffer<[f64; 4]>,
    affine_grads: DeviceBuffer<[f64; 4]>,
    /// Nodal masses (GLL weight × J) and every element's 1/area
    mass: DeviceBuffer<f64>,
    inv_area: DeviceBuffer<f64>,
    /// The Coriolis parameter of every node on a β-plane (empty otherwise)
    f_node: DeviceBuffer<f64>,
}

impl Statics {
    /// The static data, with the condition of each boundary edge from its
    /// tag (`boundary`: [`WALL`] or an open boundary's index) and the
    /// Coriolis parameter of every node on a β-plane.
    fn upload(
        gpu: &Gpu,
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        geom: &GeometricFactors2D,
        bathymetry: Option<&Bathymetry2D>,
        boundary: impl Fn(Option<BoundaryTag>) -> u32,
        coriolis: Option<&CoriolisSource2D>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let stream = &gpu.stream;
        let n_el = mesh.n_elements;
        assert_eq!(ops.n_1d, N1, "the kernels are compiled for P{ORDER}");
        for (face, nodes) in ops.face_nodes.iter().enumerate() {
            for (fi, &node) in nodes.iter().enumerate() {
                assert_eq!(face_node(face, fi), node, "face node {fi} of face {face}");
            }
        }

        let mut affine = Vec::with_capacity(n_el);
        let mut egeom = Vec::with_capacity(n_el);
        let mut affine_grads = Vec::with_capacity(n_el);
        let mut contra = Vec::with_capacity(n_el * NN);
        let mut grads = Vec::with_capacity(n_el * NN);
        let mut jinv = Vec::with_capacity(n_el * NN);
        let mut jac = Vec::with_capacity(n_el * NN);
        let mut fgeom = Vec::with_capacity(n_el * 4 * N1);
        let mut bed = Vec::with_capacity(n_el * NN);
        for k in 0..n_el {
            let element = geom.element_geometry(k);
            affine.push(u32::from(element.affine));
            let (r, s) = element.metric.contravariant();
            let mut eg = [0.0; EG];
            eg[..5].copy_from_slice(&[r.0, r.1, s.0, s.1, element.metric.det_j_inv]);
            for (face, &((nx, ny), sj)) in element.faces.iter().enumerate() {
                eg[5 + 3 * face..8 + 3 * face].copy_from_slice(&[nx, ny, sj]);
            }
            eg[17] = element.metric.det_j;
            egeom.push(eg);
            let m = &element.metric;
            affine_grads.push([m.rx, m.ry, m.sx, m.sy]);
            for i in 0..NN {
                let (r, s) = geom.contravariant(k, i);
                contra.push([r.0, r.1, s.0, s.1]);
                let ((rx, ry), (sx, sy)) = (geom.grad_r(k, i), geom.grad_s(k, i));
                grads.push([rx, ry, sx, sy]);
                jinv.push(geom.jacobian_inv(k, i));
                jac.push(geom.jacobian(k, i));
                bed.push(bathymetry.map_or(0.0, |b| b.get(ElementIndex::new(k), i)));
            }
            for face in 0..4 {
                for fi in 0..N1 {
                    let (nx, ny) = geom.normal(k, face, fi);
                    fgeom.push([nx, ny, geom.surface_jacobian(k, face, fi)]);
                }
            }
        }
        let lift: Vec<f64> = ops.lift_row_major.iter().flatten().copied().collect();
        let mass: Vec<f64> = (0..n_el)
            .flat_map(|k| (0..NN).map(move |i| (k, i)))
            .map(|(k, i)| geom.mass[geom.node_index(k, i)])
            .collect();
        let inv_area: Vec<f64> = geom.area.iter().map(|a| 1.0 / a).collect();
        let element_edges: Vec<[u32; 4]> = mesh
            .element_edges
            .iter()
            .map(|e| e.map(|e| e as u32))
            .collect();
        let edges: Vec<[u32; 4]> = mesh
            .edges
            .iter()
            .map(|e| {
                let (right, right_face) = e.right.map_or_else(
                    || (NO_NEIGHBOUR, boundary(e.boundary_tag)),
                    |r| (r.element as u32, r.face as u32),
                );
                [e.left.element as u32, e.left.face as u32, right, right_face]
            })
            .collect();
        // `SplitFormSWE2D::outer_node`'s distance, in its arithmetic: the
        // element height normal to a face at a face node, J / sJ there (in
        // physical units / 2, constant on a parallelogram), of the neighbour
        // over this element's, times the GLL spacing next to the face
        let height = |e: usize, f: usize, fi: usize, node: usize| {
            if affine[e] != 0 {
                egeom[e][17] / egeom[e][7 + 3 * f]
            } else {
                jac[e * NN + node] / fgeom[(e * 4 + f) * N1 + fi][2]
            }
        };
        let xi = &ops.nodes_1d;
        let mut outer_distance = vec![0.0; n_el * 4 * N1];
        for k in 0..n_el {
            for face in 0..4 {
                let edge = edges[element_edges[k][face] as usize];
                if edge[2] == NO_NEIGHBOUR {
                    continue;
                }
                let (nb, nb_face) = if edge[0] as usize == k && edge[1] as usize == face {
                    (edge[2] as usize, edge[3] as usize)
                } else {
                    (edge[0] as usize, edge[1] as usize)
                };
                for fi in 0..N1 {
                    let nb_node = face_node(nb_face, N1 - 1 - fi);
                    let own_node = face_node(face, fi);
                    outer_distance[(k * 4 + face) * N1 + fi] = (xi[1] - xi[0])
                        * height(nb, nb_face, N1 - 1 - fi, nb_node)
                        / height(k, face, fi, own_node);
                }
            }
        }
        // `CoriolisSource2D::add_element` on a β-plane: f at each node's y
        let f_node: Vec<f64> = match coriolis {
            Some(c) if c.beta != 0.0 => (0..n_el)
                .flat_map(|k| {
                    let corners = mesh.element_vertices(ElementIndex::new(k));
                    (0..NN).map(move |i| {
                        let (r, s) = (ops.nodes_r[i], ops.nodes_s[i]);
                        c.f_at(Mesh2D::bilinear(&corners, r, s)[1])
                    })
                })
                .collect(),
            _ => Vec::new(),
        };
        Ok(Self {
            bed: DeviceBuffer::from_host(stream, &bed)?,
            affine: DeviceBuffer::from_host(stream, &affine)?,
            egeom: DeviceBuffer::from_host(stream, &egeom)?,
            contra: DeviceBuffer::from_host(stream, &contra)?,
            jinv: DeviceBuffer::from_host(stream, &jinv)?,
            fgeom: DeviceBuffer::from_host(stream, &fgeom)?,
            outer_distance: DeviceBuffer::from_host(stream, &outer_distance)?,
            d1: DeviceBuffer::from_host(stream, &ops.dr_1d_row_major)?,
            w: DeviceBuffer::from_host(stream, &ops.weights_1d)?,
            xi: DeviceBuffer::from_host(stream, &ops.nodes_1d)?,
            lift: DeviceBuffer::from_host(stream, &lift)?,
            element_edges: DeviceBuffer::from_host(stream, &element_edges)?,
            edges: DeviceBuffer::from_host(stream, &edges)?,
            interior_edges: DeviceBuffer::from_host(stream, &edge_kinds(mesh, false))?,
            boundary_edges: DeviceBuffer::from_host(stream, &edge_kinds(mesh, true))?,
            grads: DeviceBuffer::from_host(stream, &grads)?,
            affine_grads: DeviceBuffer::from_host(stream, &affine_grads)?,
            mass: DeviceBuffer::from_host(stream, &mass)?,
            inv_area: DeviceBuffer::from_host(stream, &inv_area)?,
            f_node: DeviceBuffer::from_host(stream, &f_node)?,
        })
    }
}

/// A pass's elements split by kind (`split_elements`): the wet ones (the
/// flux-differencing kernel's) and those with a node shallower than the
/// subcell depth (the subcell kernels'), `counts[WET]` and `counts[SHALLOW]`
/// of them, on the device.
struct Split {
    wet: DeviceBuffer<u32>,
    shallow: DeviceBuffer<u32>,
    counts: DeviceBuffer<u32>,
}

/// The split form the GPU runs (`SWEFormulation2D`).
#[derive(Clone, Copy, Debug)]
pub enum Formulation {
    /// Flux differencing everywhere, the entropy-stable interface flux
    EntropyStable,
    /// Subcells in elements with a node shallower than `subcell_depth`, the
    /// hydrostatic HLL at faces, `h_dry` for the reconstruction
    WetDry { subcell_depth: f64, h_dry: f64 },
}

/// The 2D SWE physics module on the GPU: dgcom's `SWEPhysics2D` as a device
/// runs it ([`Self::from_form`]): a split formulation ([`Formulation`]), a
/// bed, walls (`Reflective2D`) and characteristic open boundaries
/// (`CharacteristicOBC` with a uniform external state, atlas tides or a
/// parent per node, `DeviceExternal2D`), the Coriolis force, the nesting's
/// relaxation band,
/// the wet/dry correction after every stage and the thin-layer relaxation
/// (its positivity limiter the redundant one of the case runner,
/// `Positivity(h_dry)`), Manning friction applied point-implicitly, the
/// positivity-bounded time step (Phase 1, `notes/gpu.md`).
pub struct GpuSWE2D<'a> {
    gpu: Arc<Gpu>,
    mesh: Arc<Mesh2D>,
    ops: Arc<DGOperators2D>,
    geom: Arc<GeometricFactors2D>,
    statics: Statics,
    g: f64,
    h_min: f64,
    formulation: Formulation,
    /// The walls' `Reflective2D::h_min`
    wall_h_min: f64,
    /// The open boundaries' external data and the velocity of their
    /// elevation-only data, in the order of [`OpenStates`], and their forms
    /// and atlas tides on the device
    opens: Vec<(DeviceExternal2D<'a>, ElevationOnly)>,
    /// The parents of the open boundaries and the relaxation bands (each
    /// once), and the bands, each with its parent's index, in the order the
    /// CPU adds them (after the Coriolis force)
    parents: Vec<&'a OceanModelState>,
    relaxations: Vec<(usize, &'a NestingRelaxation2D)>,
    forcing: Forcing,
    coriolis: Option<CoriolisSource2D>,
    wet_dry: Option<WetDryConfig>,
    friction: Option<ManningFriction2D>,
    /// Every element's count of the wet/dry correction's clips
    clips: Mutex<DeviceBuffer<u32>>,
    /// Every element's time step (`compute_dt_ssp` of a wet/dry run)
    element_dts: Mutex<DeviceBuffer<f64>>,
    /// Both sides' F* of every edge, written by the edge kernel for the
    /// element kernel
    faces: Mutex<DeviceBuffer<[[f64; 3]; NF]>>,
    /// The subcell terms of every element's lines (`subcell_terms`), for
    /// `element_rhs_subcells`; empty for the entropy-stable form
    terms: Mutex<DeviceBuffer<[f64; 3]>>,
    /// A pass's elements split by kind for the element kernels
    split: Mutex<Split>,
    /// Every element's largest reference rate (the time step)
    rates: Mutex<DeviceBuffer<f64>>,
    /// Local time stepping: the line-local depth and the positivity bound's
    /// `h_dry` (`SWEDeviceForm2D`), the coarse step's lists
    /// ([`Schedule`]), the line-local flags of the coarse step (bit 0, empty
    /// while none are marked) and every element's count of broken promises
    line_local_depth: Option<f64>,
    positivity_h_dry: Option<f64>,
    schedule: Mutex<Schedule>,
    line_local: Mutex<DeviceBuffer<u32>>,
    breaks: Mutex<DeviceBuffer<u32>>,
    /// The stage's boundary data on the device
    boundary: Mutex<BoundaryUploads>,
    /// An empty list (every element or edge) and empty rows (the sum of a
    /// stage that does not accumulate)
    no_list: DeviceBuffer<u32>,
    no_rows: Mutex<[DeviceBuffer<f64>; 3]>,
}

/// The open boundaries' external data and the parents on the device
/// (`DeviceExternal2D`, `DeviceSource2D::NestingRelaxation`): every open
/// boundary's form; a row per face node of a tide- or parent-forced
/// boundary edge, in its form's table: a tide row's constants
/// (`BoundaryTides::coefficients`, padded to the most constituents of any
/// boundary) and `[mean, source depth]`, a parent row's slot; and every
/// parent's form, its slots' snapshots and `[parent depth, band weight]`,
/// its slot of every mesh node and its band elements. A level's moments
/// are the tides' (`1 + 3 n_c` each), then the parents' (`OceanModelState::MOMENT`
/// each).
struct Forcing {
    /// Every open boundary's form, on the host and on the device
    host: Vec<OpenForm>,
    forms: DeviceBuffer<OpenForm>,
    /// Every edge's first row in its form's table (`u32::MAX` but on a
    /// tide- or parent-forced boundary; empty without either)
    row: DeviceBuffer<u32>,
    coefficients: DeviceBuffer<[f64; 6]>,
    nodes: DeviceBuffer<[f64; 2]>,
    max_constituents: usize,
    /// Every parent's form, on the host and on the device
    parents_host: Vec<ParentForm>,
    parents: DeviceBuffer<ParentForm>,
    series: DeviceBuffer<[f32; 3]>,
    slots: DeviceBuffer<[f64; 2]>,
    /// Each parent's slot of every mesh node and its band elements (a
    /// mesh's worth per parent)
    slot_of_node: DeviceBuffer<u32>,
    band: DeviceBuffer<u32>,
    /// A parent row's slot
    parent_row_slot: DeviceBuffer<u32>,
    /// The values of a level's moments
    width: usize,
}

impl Forcing {
    /// The tables of `opens` (by index, as the edges' codes; `code` gives
    /// each boundary tag's) and of `parents`, on `mesh` with `ops`.
    fn upload(
        gpu: &Gpu,
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        opens: &[(DeviceExternal2D<'_>, ElevationOnly)],
        code: impl Fn(Option<BoundaryTag>) -> u32,
        parents: &[&OceanModelState],
    ) -> Result<Self, String> {
        let stream = &gpu.stream;
        let device = |e: cuda_core::DriverError| e.to_string();
        let parent_of = |parent: &OceanModelState| {
            parents
                .iter()
                .position(|p| p.is(parent))
                .expect("every open boundary's parent is listed")
        };
        // The moments: the tides', then the parents'
        let mut width = 0;
        let mut max_constituents = 0;
        let mut forms = Vec::with_capacity(opens.len());
        for (external, _) in opens {
            forms.push(match external {
                DeviceExternal2D::Uniform(_) => OpenForm::default(),
                DeviceExternal2D::Tides(tides) => {
                    let n = tides.n_constituents();
                    let form = OpenForm {
                        kind: TIDES,
                        constituents: n as u32,
                        moment: width as u32,
                        velocity: u32::from(tides.has_velocity()),
                        max_transport_ratio: tides.max_transport_ratio().unwrap_or(0.0),
                        ..OpenForm::default()
                    };
                    width += 1 + 3 * n;
                    max_constituents = max_constituents.max(n);
                    form
                }
                DeviceExternal2D::Parent(parent) => OpenForm {
                    kind: PARENT,
                    parent: parent_of(parent) as u32,
                    ..OpenForm::default()
                },
            });
        }
        let n_flat = mesh.n_elements * NN;
        let mut parent_forms = Vec::with_capacity(parents.len());
        let (mut series, mut slots) = (Vec::new(), Vec::new());
        let (mut slot_of_node, mut band) = (Vec::new(), Vec::new());
        for parent in parents {
            parent_forms.push(ParentForm {
                n_times: parent.reader().n_times() as u32,
                series: u32::try_from(series.len()).map_err(|_| "a parent too long")?,
                slots: slots.len() as u32,
                moment: width as u32,
                velocity: u32::from(parent.has_velocity()),
                max_transport_ratio: parent.max_transport_ratio().unwrap_or(0.0),
                reference_level: parent.reference_level(),
            });
            width += OceanModelState::MOMENT;
            for slot in 0..parent.n_slots() {
                series.extend_from_slice(parent.slot_series(slot));
                slots.push([parent.parent_depth(slot), parent.band_weight(slot)]);
            }
            assert_eq!(
                parent.slot_of_node().len(),
                n_flat,
                "the parent of this mesh"
            );
            slot_of_node.extend_from_slice(parent.slot_of_node());
            band.extend(parent.band_elements().iter().map(|&b| u32::from(b)));
        }

        // A row per face node of a tide- or parent-forced edge
        let (mut row, mut coefficients, mut nodes, mut parent_row_slot) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        if forms.iter().any(|f| f.kind != UNIFORM) {
            row = vec![u32::MAX; mesh.n_edges];
        }
        for (e, edge) in mesh.edges.iter().enumerate() {
            if edge.right.is_some() {
                continue;
            }
            let Some((external, _)) = opens.get(code(edge.boundary_tag) as usize) else {
                continue;
            };
            let k = ElementIndex::new(edge.left.element);
            // The node and position of the CPU's boundary context
            // (`boundary_state`) at face node `fi`
            let node = |fi: usize| {
                let node = face_node(edge.left.face, fi);
                let [x, y] = mesh.reference_to_physical(k, ops.nodes_r[node], ops.nodes_s[node]);
                (Some(k.as_usize() * ops.n_nodes + node), (x, y))
            };
            match external {
                DeviceExternal2D::Uniform(_) => {}
                DeviceExternal2D::Tides(tides) => {
                    row[e] = nodes.len() as u32;
                    for fi in 0..N1 {
                        let (index, position) = node(fi);
                        let slot = tides.slot_of(index, position);
                        let constants = tides.coefficients(slot).as_chunks::<6>().0;
                        coefficients.extend_from_slice(constants);
                        coefficients
                            .resize(nodes.len() * max_constituents + max_constituents, [0.0; 6]);
                        nodes.push([tides.node_mean(slot), tides.source_depths()[slot]]);
                    }
                }
                DeviceExternal2D::Parent(parent) => {
                    row[e] = parent_row_slot.len() as u32;
                    for fi in 0..N1 {
                        let (index, position) = node(fi);
                        parent_row_slot.push(parent.slot_of(index, position) as u32);
                    }
                }
            }
        }
        Ok(Self {
            forms: DeviceBuffer::from_host(stream, &forms).map_err(device)?,
            host: forms,
            row: DeviceBuffer::from_host(stream, &row).map_err(device)?,
            coefficients: DeviceBuffer::from_host(stream, &coefficients).map_err(device)?,
            nodes: DeviceBuffer::from_host(stream, &nodes).map_err(device)?,
            max_constituents,
            parents: DeviceBuffer::from_host(stream, &parent_forms).map_err(device)?,
            parents_host: parent_forms,
            series: DeviceBuffer::from_host(stream, &series).map_err(device)?,
            slots: DeviceBuffer::from_host(stream, &slots).map_err(device)?,
            slot_of_node: DeviceBuffer::from_host(stream, &slot_of_node).map_err(device)?,
            band: DeviceBuffer::from_host(stream, &band).map_err(device)?,
            parent_row_slot: DeviceBuffer::from_host(stream, &parent_row_slot).map_err(device)?,
            width,
        })
    }
}

/// A stage's boundary data on the device: [`StageLevel`] per level and the
/// moments per level (rows of [`Forcing::width`]), each stage's
/// copied from a host slot of a ring, which outlives its asynchronous copy
/// (the stream is synchronized each time the ring wraps).
struct BoundaryUploads {
    device: DeviceBuffer<StageLevel>,
    host: Vec<[StageLevel; LEVELS]>,
    moments: DeviceBuffer<f64>,
    moments_host: Vec<f64>,
    width: usize,
    next: usize,
}

impl BoundaryUploads {
    /// Host slots: stage passes between two synchronizations.
    const SLOTS: usize = 1024;

    /// The ring of a run whose tides' moments are `width` values per level.
    fn new(gpu: &Gpu, width: usize) -> Result<Self, String> {
        let zeroed = |n| DeviceBuffer::zeroed(&gpu.stream, n).map_err(|e| e.to_string());
        Ok(Self {
            device: DeviceBuffer::zeroed(&gpu.stream, LEVELS).map_err(|e| e.to_string())?,
            host: vec![[StageLevel::default(); LEVELS]; Self::SLOTS],
            moments: zeroed(LEVELS * width)?,
            moments_host: vec![0.0; Self::SLOTS * LEVELS * width],
            width,
            next: 0,
        })
    }

    /// A stage's boundary data on the device, for the kernels enqueued next
    /// on `stream`, as `fill` writes them into a cleared slot: the levels
    /// and the moments.
    fn upload(
        &mut self,
        stream: &Stream,
        fill: impl FnOnce(&mut [StageLevel; LEVELS], &mut [f64]),
    ) -> (&DeviceBuffer<StageLevel>, &DeviceBuffer<f64>) {
        if self.next == Self::SLOTS {
            // Every copy from the slots has completed
            stream.synchronize().expect("synchronize");
            self.next = 0;
        }
        let n = LEVELS * self.width;
        let slot = &mut self.host[self.next];
        let moments = &mut self.moments_host[self.next * n..][..n];
        *slot = [StageLevel::default(); LEVELS];
        moments.fill(0.0);
        fill(slot, moments);
        self.next += 1;
        // SAFETY: the slot is not written again before the stream is
        // synchronized (above), so it outlives the copies; the buffers are
        // LEVELS and LEVELS × width long. Copies and kernels are ordered on
        // the stream, so the kernels of an earlier stage have read the
        // buffers before they change.
        unsafe {
            self.device
                .copy_from_host_async_unchecked(stream, slot)
                .expect("boundary upload");
            self.moments
                .copy_from_host_async_unchecked(stream, moments)
                .expect("moment upload");
        }
        (&self.device, &self.moments)
    }
}

/// The order of a coarse step's passes on the device
/// ([`LocalTimeStepping::begin_coarse_step`]): for every key (the stage
/// rates `rates[s]`, then the levels) the elements sorted by it, largest
/// first, so that a pass's elements (those whose key is at least its depth)
/// are a prefix; for every stage the edges by the larger key of their two
/// elements (the edges whose fluxes the pass's elements read). With one
/// level (global steps through the fused stages) there are no lists: every
/// pass is every element.
struct Schedule {
    finest: usize,
    /// The levels and stage rates of the lists, as uploaded
    level_host: Vec<u8>,
    rates_host: Vec<Vec<u8>>,
    /// Every element's level and last-stage rate
    level: DeviceBuffer<u32>,
    last_rate: DeviceBuffer<u32>,
    elements: Vec<DeviceBuffer<u32>>,
    element_counts: Vec<Vec<usize>>,
    /// Per stage, the interior edges (`[0]`) and the boundary edges (`[1]`)
    /// by key
    edges: Vec<[DeviceBuffer<u32>; 2]>,
    edge_counts: Vec<[Vec<usize>; 2]>,
}

impl Schedule {
    fn empty(gpu: &Gpu) -> Result<Self, String> {
        let none = || DeviceBuffer::zeroed(&gpu.stream, 0).map_err(|e| e.to_string());
        Ok(Self {
            finest: 0,
            level_host: Vec::new(),
            rates_host: Vec::new(),
            level: none()?,
            last_rate: none()?,
            elements: Vec::new(),
            element_counts: Vec::new(),
            edges: Vec::new(),
            edge_counts: Vec::new(),
        })
    }

    /// Items `0..keys.len()` by key, largest first (in index order within a
    /// key), and how many have a key of at least `d`, for `d` in
    /// `0..=finest`.
    fn order(keys: impl Fn(usize) -> u8, n: usize, finest: usize) -> (Vec<u32>, Vec<usize>) {
        let mut count = vec![0usize; finest + 1];
        for i in 0..n {
            count[keys(i) as usize] += 1;
        }
        // at_least[d] = Σ_{v ≥ d} count[v]; key v starts after the larger keys
        let mut at_least = vec![0usize; finest + 2];
        for d in (0..=finest).rev() {
            at_least[d] = at_least[d + 1] + count[d];
        }
        let mut next: Vec<usize> = (0..=finest).map(|v| at_least[v + 1]).collect();
        let mut order = vec![0u32; n];
        for i in 0..n {
            let v = keys(i) as usize;
            order[next[v]] = i as u32;
            next[v] += 1;
        }
        at_least.truncate(finest + 1);
        (order, at_least)
    }

    /// The lists of the levels `level` and stage rates `rates` on `mesh`,
    /// uploaded unless they are the last ones.
    fn update(
        &mut self,
        gpu: &Gpu,
        mesh: &Mesh2D,
        level: &[u8],
        rates: &[Vec<u8>],
    ) -> Result<(), Box<dyn std::error::Error>> {
        if self.level_host == level && self.rates_host == rates {
            return Ok(());
        }
        self.level_host = level.to_vec();
        self.rates_host = rates.to_vec();
        let finest = level
            .iter()
            .chain(rates.iter().flatten())
            .copied()
            .max()
            .unwrap_or(0) as usize;
        assert!(
            finest <= MAX_DEVICE_LEVELS,
            "{finest} levels: the device takes at most {MAX_DEVICE_LEVELS}"
        );
        self.finest = finest;
        if finest == 0 {
            return Ok(());
        }
        let n = mesh.n_elements;
        let stream = &gpu.stream;
        let upload = |buffer: &mut DeviceBuffer<u32>, host: &[u32]| {
            if buffer.len() == host.len() {
                buffer.copy_from_host(stream, host)
            } else {
                *buffer = DeviceBuffer::from_host(stream, host)?;
                Ok(())
            }
        };
        let widen = |keys: &[u8]| keys.iter().map(|&k| u32::from(k)).collect::<Vec<_>>();
        upload(&mut self.level, &widen(level))?;
        upload(&mut self.last_rate, &widen(rates.last().expect("a stage")))?;
        let keys: Vec<&[u8]> = rates.iter().map(Vec::as_slice).chain([level]).collect();
        self.elements.resize_with(keys.len(), || {
            DeviceBuffer::zeroed(stream, 0).expect("allocation")
        });
        self.element_counts.resize(keys.len(), Vec::new());
        for (j, key) in keys.iter().enumerate() {
            let (order, counts) = Self::order(|k| key[k], n, finest);
            upload(&mut self.elements[j], &order)?;
            self.element_counts[j] = counts;
        }
        let kinds = [edge_kinds(mesh, false), edge_kinds(mesh, true)];
        self.edges.resize_with(rates.len(), || {
            [0, 1].map(|_| DeviceBuffer::zeroed(stream, 0).expect("allocation"))
        });
        self.edge_counts
            .resize(rates.len(), [Vec::new(), Vec::new()]);
        for (s, rate) in rates.iter().enumerate() {
            for (kind, edges) in kinds.iter().enumerate() {
                let edge_key = |i: usize| {
                    let edge = &mesh.edges[edges[i] as usize];
                    let left = rate[edge.left.element];
                    edge.right.map_or(left, |r| left.max(rate[r.element]))
                };
                let (order, counts) = Self::order(edge_key, edges.len(), finest);
                let order: Vec<u32> = order.iter().map(|&i| edges[i as usize]).collect();
                upload(&mut self.edges[s][kind], &order)?;
                self.edge_counts[s][kind] = counts;
            }
        }
        Ok(())
    }
}

impl<'a> GpuSWE2D<'a> {
    /// dgcom's physics module as its device form describes it
    /// (`SWEPhysics2D::device_form`), on `gpu`; what the kernels do not run
    /// is refused, named.
    pub fn from_form(gpu: &Arc<Gpu>, form: &SWEDeviceForm2D<'a>) -> Result<Self, String> {
        let (mesh, ops) = (form.mesh, form.ops);
        if ops.n_1d != N1 {
            return Err(format!(
                "a mesh of order {}: the kernels are compiled for P{ORDER}",
                ops.order
            ));
        }
        let formulation = match (form.formulation, form.subcells) {
            (SWEFormulation2D::EntropyStable, _) => Formulation::EntropyStable,
            (SWEFormulation2D::WetDry, Some((subcell_depth, h_dry))) => Formulation::WetDry {
                subcell_depth,
                h_dry,
            },
            (other, _) => return Err(format!("the {other:?} formulation")),
        };
        // Each boundary tag's condition: walls (one h_min), open boundaries
        // by index
        let mut wall_h_min = None;
        let mut opens = Vec::new();
        let mut codes = Vec::new();
        for &(tag, boundary) in &form.boundaries {
            let code = match boundary {
                DeviceBoundary2D::Wall { h_min } => {
                    if wall_h_min.is_some_and(|w| w != h_min) {
                        return Err("walls of two h_min".into());
                    }
                    wall_h_min = Some(h_min);
                    WALL
                }
                DeviceBoundary2D::Characteristic {
                    external,
                    elevation_only,
                } => {
                    if opens.len() == MAX_OPEN {
                        return Err(format!("more than {MAX_OPEN} open boundaries"));
                    }
                    opens.push((external, elevation_only));
                    (opens.len() - 1) as u32
                }
            };
            codes.push((tag, code));
        }
        let boundary = |tag| {
            codes
                .iter()
                .find_map(|&(t, code)| (t == tag).then_some(code))
                .expect("every boundary tag has a form")
        };
        // The sources: the Coriolis force (in the element kernels) first,
        // then the relaxation bands (a kernel each, after them)
        let mut coriolis = None;
        let mut relaxations: Vec<&NestingRelaxation2D> = Vec::new();
        for source in &form.sources {
            match *source {
                DeviceSource2D::Coriolis(c) => {
                    if coriolis.is_some() || !relaxations.is_empty() {
                        return Err(
                            "a Coriolis force after another source (the device adds it first)"
                                .into(),
                        );
                    }
                    coriolis = Some(c);
                }
                DeviceSource2D::NestingRelaxation(r) => relaxations.push(r),
            }
        }
        // Every parent once
        let mut parents: Vec<&OceanModelState> = Vec::new();
        let mut parent_index =
            |parent: &'a OceanModelState| match parents.iter().position(|p| p.is(parent)) {
                Some(i) => i,
                None => {
                    parents.push(parent);
                    parents.len() - 1
                }
            };
        for (external, _) in &opens {
            if let DeviceExternal2D::Parent(parent) = external {
                parent_index(parent);
            }
        }
        let relaxations: Vec<(usize, &NestingRelaxation2D)> = relaxations
            .into_iter()
            .map(|r| (parent_index(r.parent()), r))
            .collect();
        let friction = form.friction.as_ref().map(|law| match law {
            DeviceFriction2D::Manning(m) => m.clone(),
        });
        let device = |e: Box<dyn std::error::Error>| e.to_string();
        let statics = Statics::upload(
            gpu,
            mesh,
            ops,
            form.geom,
            form.bathymetry,
            boundary,
            coriolis.as_ref(),
        )
        .map_err(device)?;
        let forcing = Forcing::upload(gpu, mesh, ops, &opens, boundary, &parents)?;
        let stream = &gpu.stream;
        let zeroed = |n| DeviceBuffer::zeroed(stream, n).map_err(|e| e.to_string());
        Ok(Self {
            gpu: gpu.clone(),
            mesh: mesh.clone(),
            ops: ops.clone(),
            geom: form.geom.clone(),
            statics,
            g: form.equation.g,
            h_min: form.equation.h_min.meters(),
            formulation,
            wall_h_min: wall_h_min.unwrap_or(Reflective2D::new().h_min),
            opens,
            coriolis,
            wet_dry: form.wet_dry.clone(),
            friction,
            clips: Mutex::new(
                DeviceBuffer::zeroed(stream, mesh.n_elements).map_err(|e| e.to_string())?,
            ),
            element_dts: Mutex::new(zeroed(mesh.n_elements)?),
            faces: Mutex::new(
                DeviceBuffer::zeroed(stream, mesh.n_edges).map_err(|e| e.to_string())?,
            ),
            terms: Mutex::new(
                DeviceBuffer::zeroed(
                    stream,
                    match formulation {
                        Formulation::EntropyStable => 0,
                        Formulation::WetDry { .. } => mesh.n_elements * 2 * N1 * N1 * LINE_TERMS,
                    },
                )
                .map_err(|e| e.to_string())?,
            ),
            rates: Mutex::new(zeroed(mesh.n_elements)?),
            split: Mutex::new(Split {
                wet: DeviceBuffer::zeroed(stream, mesh.n_elements).map_err(|e| e.to_string())?,
                shallow: DeviceBuffer::zeroed(stream, mesh.n_elements)
                    .map_err(|e| e.to_string())?,
                counts: DeviceBuffer::zeroed(stream, 2).map_err(|e| e.to_string())?,
            }),
            line_local_depth: form.line_local_depth,
            positivity_h_dry: form.positivity_h_dry,
            schedule: Mutex::new(Schedule::empty(gpu)?),
            line_local: Mutex::new(DeviceBuffer::zeroed(stream, 0).map_err(|e| e.to_string())?),
            breaks: Mutex::new(
                DeviceBuffer::zeroed(stream, mesh.n_elements).map_err(|e| e.to_string())?,
            ),
            no_list: DeviceBuffer::zeroed(stream, 0).map_err(|e| e.to_string())?,
            no_rows: Mutex::new([zeroed(0)?, zeroed(0)?, zeroed(0)?]),
            boundary: Mutex::new(BoundaryUploads::new(gpu, forcing.width)?),
            parents,
            relaxations,
            forcing,
        })
    }

    /// Elements the wet/dry correction emptied, over every stage so far
    /// (`SWEPhysics2D::negative_depth_clips`).
    pub fn negative_depth_clips(&self) -> usize {
        let clips = self.clips.lock().expect("clip counts");
        let clips = clips.to_host_vec(&self.gpu.stream).expect("download");
        clips.iter().map(|&c| c as usize).sum()
    }

    /// The open boundaries' data at `time`: the uniform external states
    /// and every boundary's elevation-only velocity, and the tides' and the
    /// parents' moments into `moments` (a level's row).
    fn open_states(&self, time: f64, moments: &mut [f64]) -> OpenStates {
        let mut open = OpenStates::default();
        for (i, (external, elevation_only)) in self.opens.iter().enumerate() {
            let incoming = u32::from(*elevation_only == ElevationOnly::IncomingWave) << 1;
            match external {
                DeviceExternal2D::Uniform(provider) => {
                    let state = provider
                        .uniform_state(time)
                        .expect("a uniform external state (DeviceExternal2D::Uniform)");
                    let (u, v) = state.velocity.unwrap_or((0.0, 0.0));
                    open.state[i] = [state.eta, u, v];
                    open.flags[i] = u32::from(state.velocity.is_some()) | incoming;
                }
                DeviceExternal2D::Tides(tides) => {
                    let m = self.forcing.host[i].moment as usize;
                    tides.moment(time, &mut moments[m..m + 1 + 3 * tides.n_constituents()]);
                    open.flags[i] = incoming;
                }
                DeviceExternal2D::Parent(_) => open.flags[i] = incoming,
            }
        }
        for (parent, form) in self.parents.iter().zip(&self.forcing.parents_host) {
            let m = form.moment as usize;
            parent.moment(time, &mut moments[m..m + OceanModelState::MOMENT]);
        }
        open
    }

    /// The Coriolis parameters of the element kernels: none (0), f-plane (1)
    /// or a β-plane's f per node (2), and f₀.
    fn coriolis_args(&self) -> (u32, f64) {
        match &self.coriolis {
            None => (0, 0.0),
            Some(c) if c.beta == 0.0 => (1, c.f0),
            Some(c) => (2, c.f0),
        }
    }

    /// The point-implicit damping of a stage and the wet/dry correction.
    fn damping(&self) -> Damping {
        let (g, manning_n, friction_h_min) = self
            .friction
            .as_ref()
            .map_or((0.0, 0.0, 0.0), |f| (f.g, f.manning_n, f.h_min));
        let (h_dry, relaxation_time, max_velocity, wet_dry_g) =
            self.wet_dry.as_ref().map_or((0.0, 0.0, 0.0, 0.0), |c| {
                (c.h_dry.meters(), c.relaxation_time, c.max_velocity, c.g)
            });
        Damping {
            h_min: self.h_min,
            friction: u32::from(self.friction.is_some()),
            g,
            manning_n,
            friction_h_min,
            wet_dry: u32::from(self.wet_dry.is_some()),
            h_dry,
            relaxation_time,
            max_velocity,
            wet_dry_g,
        }
    }

    /// The RHS of the listed elements into their rows of `out` (`edges` and
    /// `elements`: the first `n` of a list, every edge or element where the
    /// list is empty): the F* of the listed edges, then the element
    /// kernels. The boundary data of level `l` are at its stage time
    /// `times[l]`; `level` gives a boundary edge its element's level (empty:
    /// level 0).
    fn rhs_where(
        &self,
        state: &DeviceState,
        times: &[f64],
        level: &DeviceBuffer<u32>,
        [(interior, n_interior), (boundary_list, n_boundary)]: [(&DeviceBuffer<u32>, usize); 2],
        (element_list, n_elements): (&DeviceBuffer<u32>, usize),
        out: &mut DeviceState,
    ) {
        let (gpu, s) = (&self.gpu, &self.statics);
        // The element kernels split at the subcell depth (none for the
        // entropy-stable form: no depth is below −∞)
        let (hll, subcell_depth, h_dry) = match self.formulation {
            Formulation::EntropyStable => (false, f64::NEG_INFINITY, 0.0),
            Formulation::WetDry {
                subcell_depth,
                h_dry,
            } => (true, subcell_depth, h_dry),
        };
        let mut faces = self.faces.lock().expect("face buffer");
        let mut uploads = self.boundary.lock().expect("boundary data");
        let width = self.forcing.width;
        let (boundary, moments) = uploads.upload(&gpu.stream, |levels, moments| {
            for (l, &time) in times.iter().enumerate() {
                let open = self.open_states(time, &mut moments[l * width..(l + 1) * width]);
                levels[l] = StageLevel { time, open };
            }
        });
        let (coriolis, f0) = self.coriolis_args();
        let elements_cfg = gpu.spread(n_elements, ELEMENT_BLOCK);
        let mut split = self.split.lock().expect("split lists");
        let Split {
            wet,
            shallow,
            counts,
        } = &mut *split;
        let n_pass = n_elements;
        let n_elements = n_elements as u32;
        // SAFETY: every buffer is sized for this mesh (`Statics::upload`, the
        // state's), the lists hold distinct edges and elements of it, and the
        // kernels index within its elements, edges and nodes.
        unsafe {
            // The pass's elements by kind, for the element kernels
            counts
                .zero_async(&gpu.stream)
                .expect("clear the split counts");
            gpu.module
                .split_elements(
                    &gpu.stream,
                    gpu.spread(n_pass, 256),
                    subcell_depth,
                    element_list,
                    n_elements,
                    &state.h,
                    counts.cu_deviceptr() as *mut u32,
                    &mut *wet,
                    &mut *shallow,
                )
                .expect("split elements");
            if n_interior > 0 {
                let cfg = gpu.spread(n_interior, 256);
                let n = n_interior as u32;
                if hll {
                    gpu.module.edge_fluxes_hll(
                        &gpu.stream,
                        cfg,
                        self.g,
                        self.h_min,
                        interior,
                        n,
                        &state.h,
                        &state.hu,
                        &state.hv,
                        &s.bed,
                        &s.affine,
                        &s.egeom,
                        &s.fgeom,
                        &s.edges,
                        &mut faces,
                    )
                } else {
                    gpu.module.edge_fluxes_entropy_stable(
                        &gpu.stream,
                        cfg,
                        self.g,
                        self.h_min,
                        interior,
                        n,
                        &state.h,
                        &state.hu,
                        &state.hv,
                        &s.bed,
                        &s.affine,
                        &s.egeom,
                        &s.fgeom,
                        &s.edges,
                        &mut faces,
                    )
                }
                .expect("edge fluxes");
            }
            if n_boundary > 0 {
                gpu.module
                    .boundary_fluxes(
                        &gpu.stream,
                        gpu.spread(n_boundary * N1, 256),
                        u32::from(hll),
                        self.g,
                        self.h_min,
                        self.wall_h_min,
                        boundary,
                        &self.forcing.forms,
                        moments,
                        width as u32,
                        &self.forcing.row,
                        &self.forcing.coefficients,
                        &self.forcing.nodes,
                        self.forcing.max_constituents as u32,
                        &self.forcing.parents,
                        &self.forcing.series,
                        &self.forcing.slots,
                        &self.forcing.parent_row_slot,
                        level,
                        boundary_list,
                        n_boundary as u32,
                        &state.h,
                        &state.hu,
                        &state.hv,
                        &s.bed,
                        &s.affine,
                        &s.egeom,
                        &s.fgeom,
                        &s.edges,
                        &mut faces,
                    )
                    .expect("boundary fluxes");
            }
            let [out_h, out_hu, out_hv] = out.fields_mut();
            gpu.module
                .element_rhs_volume(
                    &gpu.stream,
                    elements_cfg,
                    self.g,
                    self.h_min,
                    &*wet,
                    &*counts,
                    &state.h,
                    &state.hu,
                    &state.hv,
                    &s.bed,
                    &s.affine,
                    &s.egeom,
                    &s.contra,
                    &s.jinv,
                    &s.fgeom,
                    &s.d1,
                    &s.lift,
                    &s.element_edges,
                    &s.edges,
                    &faces,
                    coriolis,
                    f0,
                    &s.f_node,
                    out_h,
                    out_hu,
                    out_hv,
                )
                .expect("element RHS (volume)");
            if hll {
                // The subcells: each line's terms (one thread per line), then
                // each element's sum in the CPU's order (one per element)
                let mut terms = self.terms.lock().expect("subcell terms");
                gpu.module
                    .subcell_terms(
                        &gpu.stream,
                        gpu.spread(n_pass * 2 * N1, SUBCELL_BLOCK),
                        self.g,
                        self.h_min,
                        h_dry,
                        &*shallow,
                        &*counts,
                        &state.h,
                        &state.hu,
                        &state.hv,
                        &s.bed,
                        &s.affine,
                        &s.egeom,
                        &s.contra,
                        &s.outer_distance,
                        &s.d1,
                        &s.w,
                        &s.xi,
                        &s.element_edges,
                        &s.edges,
                        &mut terms,
                    )
                    .expect("subcell terms");
                let [out_h, out_hu, out_hv] = out.fields_mut();
                gpu.module
                    .element_rhs_subcells(
                        &gpu.stream,
                        elements_cfg,
                        self.g,
                        self.h_min,
                        &*shallow,
                        &*counts,
                        &state.h,
                        &state.hu,
                        &state.hv,
                        &s.bed,
                        &s.affine,
                        &s.egeom,
                        &s.jinv,
                        &s.fgeom,
                        &s.lift,
                        &s.element_edges,
                        &s.edges,
                        &faces,
                        &terms,
                        coriolis,
                        f0,
                        &s.f_node,
                        out_h,
                        out_hu,
                        out_hv,
                    )
                    .expect("element RHS (subcells)");
            }
            // The relaxation bands, in the CPU's order, after the Coriolis
            // force of the element kernels
            let f = &self.forcing;
            for &(p, relaxation) in &self.relaxations {
                let [out_h, out_hu, out_hv] = out.fields_mut();
                gpu.module
                    .nesting_relaxation(
                        &gpu.stream,
                        gpu.spread(n_pass, 256),
                        p as u32,
                        relaxation.rate(),
                        relaxation.wet_depth(),
                        u32::from(relaxation.momentum_only()),
                        &f.parents,
                        &f.series,
                        &f.slots,
                        &f.slot_of_node,
                        &f.band,
                        self.mesh.n_elements as u32,
                        moments,
                        width as u32,
                        level,
                        element_list,
                        n_elements,
                        &state.h,
                        &state.hu,
                        &state.hv,
                        &s.bed,
                        out_h,
                        out_hu,
                        out_hv,
                    )
                    .expect("nesting relaxation");
            }
        }
    }

    /// Every element's flags of `depth_flags` (bit j: every node deeper
    /// than `depths[j]`), on the host.
    fn depth_flags(&self, state: &DeviceState, depths: [f64; 3]) -> Vec<u32> {
        let gpu = &self.gpu;
        let n = self.mesh.n_elements;
        let mut flags = DeviceBuffer::<u32>::zeroed(&gpu.stream, n).expect("allocation");
        // SAFETY: one flag per element of the state.
        unsafe {
            gpu.module.depth_flags(
                &gpu.stream,
                one_dimensional(n),
                depths[0],
                depths[1],
                depths[2],
                &state.h,
                &mut flags,
            )
        }
        .expect("depth flags");
        flags.to_host_vec(&gpu.stream).expect("download")
    }
}

impl PhysicsModuleInfo for GpuSWE2D<'_> {
    fn name(&self) -> &'static str {
        "swe-2d-gpu"
    }

    fn description(&self) -> &str {
        "2D Shallow Water Equations on the GPU"
    }

    fn n_variables(&self) -> usize {
        3
    }

    fn variable_names(&self) -> &[&'static str] {
        &["h", "hu", "hv"]
    }
}

impl PhysicsModule<DeviceState> for GpuSWE2D<'_> {
    fn compute_rhs(&self, state: &DeviceState, time: f64) -> DeviceState {
        let mut out = state.zeroed_like();
        self.compute_rhs_into(state, time, &mut out);
        out
    }

    fn compute_rhs_into(&self, state: &DeviceState, time: f64, out: &mut DeviceState) {
        let s = &self.statics;
        self.rhs_where(
            state,
            &[time],
            &self.no_list,
            [
                (&s.interior_edges, s.interior_edges.len()),
                (&s.boundary_edges, s.boundary_edges.len()),
            ],
            (&self.no_list, self.mesh.n_elements),
            out,
        );
    }

    /// `compute_dt_swe_2d`: the largest element rate on the device, reduced
    /// on the host (a max is exact in any order).
    fn compute_dt(&self, state: &DeviceState, cfl: f64) -> f64 {
        let (gpu, s) = (&self.gpu, &self.statics);
        let mut rates = self.rates.lock().expect("rate buffer");
        // SAFETY: one rate per element; the state and statics are this mesh's.
        unsafe {
            gpu.module.element_rates(
                &gpu.stream,
                one_dimensional(self.mesh.n_elements),
                self.g,
                self.h_min,
                &state.h,
                &state.hu,
                &state.hv,
                &s.affine,
                &s.affine_grads,
                &s.grads,
                &mut rates,
            )
        }
        .expect("element rates");
        let rates = rates.to_host_vec(&gpu.stream).expect("download");
        let max_rate = rates.iter().copied().fold(0.0_f64, f64::max);
        dt_from_reference_rate(max_rate, self.ops.order, cfl)
    }

    /// The wet/dry correction (`SWEPhysics2D::post_process` with a
    /// redundant positivity limiter).
    fn post_process(&self, state: &mut DeviceState) {
        let Some(config) = &self.wet_dry else {
            return;
        };
        let (gpu, s) = (&self.gpu, &self.statics);
        let [h, hu, hv] = state.fields_mut();
        let mut clips = self.clips.lock().expect("clip counts");
        // SAFETY: one thread per element of this mesh; the state's rows.
        unsafe {
            gpu.module.wet_dry_correction(
                &gpu.stream,
                per_element(self.mesh.n_elements),
                config.h_dry.meters(),
                config.relaxation_time,
                config.max_velocity,
                config.g,
                &s.mass,
                &s.inv_area,
                h,
                hu,
                hv,
                &mut clips,
            )
        }
        .expect("wet/dry correction");
    }

    /// Point-implicit Manning friction and thin-layer relaxation.
    fn implicit_damping(&self, stage: &mut DeviceState, from: &DeviceState, dt: f64) {
        let damping = self.damping();
        if !damping.is_active() {
            return;
        }
        let gpu = &self.gpu;
        let n_nodes = stage.h.len();
        let DeviceState { h, hu, hv, .. } = stage;
        // SAFETY: one thread per node of the state; `from` is another state.
        unsafe {
            gpu.module.implicit_damping(
                &gpu.stream,
                one_dimensional(n_nodes),
                dt,
                damping,
                &*h,
                &from.h,
                &from.hu,
                &from.hv,
                hu,
                hv,
            )
        }
        .expect("implicit damping");
    }

    /// With wetting and drying, `min_element_dt_swe_2d` under the positivity
    /// bound of `scheme`, the smallest element step reduced on the host
    /// (exact in any order); else `compute_dt`.
    fn compute_dt_ssp(&self, state: &DeviceState, cfl: f64, scheme: Option<SspScheme>) -> f64 {
        let Some(config) = &self.wet_dry else {
            return self.compute_dt(state, cfl);
        };
        let bound = PositivityBound::new(self.ops.order, scheme, config.h_dry.meters());
        let (gpu, s) = (&self.gpu, &self.statics);
        let mut dts = self.element_dts.lock().expect("time step buffer");
        // SAFETY: one step per element; the state and statics are this mesh's.
        unsafe {
            gpu.module.element_dt(
                &gpu.stream,
                one_dimensional(self.mesh.n_elements),
                self.g,
                self.h_min,
                cfl,
                self.ops.order as u32,
                1,
                bound.cfl,
                bound.max_cfl,
                bound.max_cfl_subcells,
                bound.h_dry,
                &state.h,
                &state.hu,
                &state.hv,
                &s.affine,
                &s.affine_grads,
                &s.grads,
                &s.mass,
                &s.element_edges,
                &s.edges,
                &mut dts,
            )
        }
        .expect("element time steps");
        let dts = dts.to_host_vec(&gpu.stream).expect("download");
        dts.iter().copied().fold(f64::INFINITY, f64::min)
    }

    fn local_time_stepping(&self) -> Option<&dyn LocalTimeStepping<DeviceState>> {
        Some(self)
    }

    fn mesh(&self) -> &Mesh2D {
        &self.mesh
    }

    fn operators(&self) -> &DGOperators2D {
        &self.ops
    }

    fn geometry(&self) -> &GeometricFactors2D {
        &self.geom
    }

    fn order(&self) -> usize {
        self.ops.order
    }
}

/// Local time stepping (dgcom's `Multirate`) and the fused stages of global
/// steps, as `SWEPhysics2D` does them: every pass a list of the coarse
/// step's [`Schedule`] (a prefix of the elements and edges by key), the
/// stage of each element from its level ([`element_stage`], the CPU's
/// arithmetic), the open boundaries' states per level.
impl LocalTimeStepping<DeviceState> for GpuSWE2D<'_> {
    /// `element_dt_swe_2d` under the positivity bound of `scheme` with
    /// wetting and drying, on the device; the steps to the host.
    fn element_dt(&self, state: &DeviceState, cfl: f64, scheme: SspScheme, out: &mut [f64]) {
        let (gpu, s) = (&self.gpu, &self.statics);
        let bound = self
            .positivity_h_dry
            .map(|h_dry| PositivityBound::new(self.ops.order, Some(scheme), h_dry));
        let (positive, bound) = match bound {
            Some(bound) => (1, bound),
            None => (0, PositivityBound::new(self.ops.order, Some(scheme), 0.0)),
        };
        let mut dts = self.element_dts.lock().expect("time step buffer");
        // SAFETY: one step per element; the state and statics are this mesh's.
        unsafe {
            gpu.module.element_dt(
                &gpu.stream,
                one_dimensional(self.mesh.n_elements),
                self.g,
                self.h_min,
                cfl,
                self.ops.order as u32,
                positive,
                bound.cfl,
                bound.max_cfl,
                bound.max_cfl_subcells,
                bound.h_dry,
                &state.h,
                &state.hu,
                &state.hv,
                &s.affine,
                &s.affine_grads,
                &s.grads,
                &s.mass,
                &s.element_edges,
                &s.edges,
                &mut dts,
            )
        }
        .expect("element time steps");
        dts.copy_to_host(&gpu.stream, out).expect("download");
    }

    /// The elements whose every node is `LINE_LOCAL_HEADROOM` deeper than
    /// the line-local depth (`SWEPhysics2D::line_local`).
    fn line_local(&self, state: &DeviceState, out: &mut [bool]) -> Option<NodeLines<'_>> {
        let mut kept = self.line_local.lock().expect("line-local flags");
        let none = DeviceBuffer::zeroed(&self.gpu.stream, 0).expect("allocation");
        let shallow = match self.line_local_depth {
            Some(shallow) => shallow,
            None => {
                *kept = none;
                return None;
            }
        };
        let depth = if shallow == f64::NEG_INFINITY {
            shallow
        } else {
            shallow + LINE_LOCAL_HEADROOM
        };
        let flags = self.depth_flags(state, [depth; 3]);
        for (flag, &bits) in out.iter_mut().zip(&flags) {
            *flag = bits & 1 != 0;
        }
        *kept = DeviceBuffer::from_host(&self.gpu.stream, &flags).expect("upload");
        Some(NodeLines {
            n_1d: self.ops.n_1d,
            face_nodes: &self.ops.face_nodes,
        })
    }

    fn line_local_breaks(&self) -> u64 {
        let breaks = self.breaks.lock().expect("break counts");
        let breaks = breaks.to_host_vec(&self.gpu.stream).expect("download");
        breaks.iter().map(|&b| u64::from(b)).sum()
    }

    /// `SWEPhysics2D::coarse_step_bound`: the largest step of an element
    /// whose every node is deeper than `SUBCELL_CAP_DEPTH_FACTOR` times
    /// `h_dry`, else than `h_dry`.
    fn coarse_step_bound(&self, state: &DeviceState, element_dt: &[f64]) -> f64 {
        let Some(h_dry) = self.positivity_h_dry else {
            return f64::INFINITY;
        };
        let flags = self.depth_flags(state, [SUBCELL_CAP_DEPTH_FACTOR * h_dry, h_dry, h_dry]);
        let longest = |bit: u32| {
            element_dt
                .iter()
                .zip(&flags)
                .filter(|&(dt, &f)| dt.is_finite() && f & bit != 0)
                .map(|(&dt, _)| dt)
                .fold(0.0, f64::max)
        };
        [1, 2]
            .into_iter()
            .map(longest)
            .find(|&dt| dt > 0.0)
            .unwrap_or(f64::INFINITY)
    }

    fn begin_coarse_step(&self, level: &[u8], rates: &[Vec<u8>]) {
        self.schedule
            .lock()
            .expect("schedule")
            .update(&self.gpu, &self.mesh, level, rates)
            .expect("the coarse step's lists");
    }

    /// The RHS of the pass's elements (a prefix of the schedule's order),
    /// then per element the stage combination, the damping, and the sum
    /// into `acc` or the wet/dry correction (`stage_elements`).
    fn stage_where(
        &self,
        base: &DeviceState,
        input: &DeviceState,
        elements: &[u32],
        plan: &StagePlan<'_>,
        out: &mut DeviceState,
        acc: Option<&mut DeviceState>,
    ) {
        if elements.is_empty() {
            return;
        }
        let schedule = self.schedule.lock().expect("schedule");
        let finest = schedule.finest;
        assert_eq!(plan.finest, finest, "the plan is the coarse step's");
        // Every level's stage time
        let mut times = [0.0; LEVELS];
        for (l, time) in times.iter_mut().enumerate().take(finest + 1) {
            *time = element_stage(
                plan.step,
                plan.finest,
                plan.substep,
                plan.depth,
                plan.weights,
                false,
                l as u8,
                l as u8,
            )
            .time;
        }
        let (s, d) = (plan.stage, plan.depth);
        let all = finest == 0;
        let n_elements = if all {
            self.mesh.n_elements
        } else {
            schedule.element_counts[s][d]
        };
        assert_eq!(
            elements.len(),
            n_elements,
            "the pass's elements are the schedule's"
        );
        // The first n of the schedule's orders (every element, and the
        // static edge lists, with one level)
        let element_list = if all {
            &self.no_list
        } else {
            &schedule.elements[s]
        };
        let st = &self.statics;
        let edges = if all {
            [
                (&st.interior_edges, st.interior_edges.len()),
                (&st.boundary_edges, st.boundary_edges.len()),
            ]
        } else {
            [0, 1].map(|kind| (&schedule.edges[s][kind], schedule.edge_counts[s][kind][d]))
        };
        let level = if all { &self.no_list } else { &schedule.level };
        let last_rate = if all {
            &self.no_list
        } else {
            &schedule.last_rate
        };
        self.rhs_where(
            input,
            &times[..=finest],
            level,
            edges,
            (element_list, n_elements),
            out,
        );

        let gpu = &self.gpu;
        let line_local = self.line_local.lock().expect("line-local flags");
        let mut clips = self.clips.lock().expect("clip counts");
        let mut breaks = self.breaks.lock().expect("break counts");
        let mut no_rows = self.no_rows.lock().expect("empty rows");
        let [no_h, no_hu, no_hv] = &mut *no_rows;
        assert!(
            acc.is_some() || !plan.last,
            "an accumulating stage needs the sum buffer"
        );
        let (acc_h, acc_hu, acc_hv) = match acc {
            Some(acc) => {
                let DeviceState { h, hu, hv, .. } = acc;
                (h, hu, hv)
            }
            None => (no_h, no_hu, no_hv),
        };
        let w = plan.weights;
        let shallow = self.line_local_depth.unwrap_or(f64::NEG_INFINITY);
        let [out_h, out_hu, out_hv] = out.fields_mut();
        // SAFETY: the list holds distinct elements of this mesh; `out`,
        // `acc` and the counts are this mesh's (or, for a stage that does not
        // accumulate, `acc` is empty and unread), `base` and `input` other
        // states.
        unsafe {
            gpu.module.stage_elements(
                &gpu.stream,
                gpu.spread(n_elements, ELEMENT_BLOCK),
                plan.step.0,
                plan.step.1,
                plan.finest as u32,
                plan.substep as u32,
                plan.depth as u32,
                u32::from(plan.last),
                w.start,
                w.input,
                w.beta,
                w.c,
                level,
                last_rate,
                element_list,
                n_elements as u32,
                &base.h,
                &base.hu,
                &base.hv,
                &input.h,
                &input.hu,
                &input.hv,
                self.damping(),
                &line_local,
                shallow,
                &st.mass,
                &st.inv_area,
                out_h,
                out_hu,
                out_hv,
                acc_h,
                acc_hu,
                acc_hv,
                &mut clips,
                &mut breaks,
            )
        }
        .expect("stage");
    }

    fn finish_where(
        &self,
        state: &mut DeviceState,
        acc: &DeviceState,
        elements: &[u32],
        depth: usize,
    ) {
        if elements.is_empty() {
            return;
        }
        let schedule = self.schedule.lock().expect("schedule");
        let all = schedule.finest == 0;
        let key = schedule.element_counts.len().saturating_sub(1);
        let n = if all {
            self.mesh.n_elements
        } else {
            schedule.element_counts[key][depth]
        };
        assert_eq!(elements.len(), n, "the pass's elements are the schedule's");
        let list = if all {
            &self.no_list
        } else {
            &schedule.elements[key]
        };
        let (gpu, st) = (&self.gpu, &self.statics);
        let line_local = self.line_local.lock().expect("line-local flags");
        let mut clips = self.clips.lock().expect("clip counts");
        let mut breaks = self.breaks.lock().expect("break counts");
        let shallow = self.line_local_depth.unwrap_or(f64::NEG_INFINITY);
        let [h, hu, hv] = state.fields_mut();
        // SAFETY: the list holds distinct elements of this mesh; the state,
        // the sum and the counts are this mesh's.
        unsafe {
            gpu.module.finish_elements(
                &gpu.stream,
                gpu.spread(n, ELEMENT_BLOCK),
                list,
                n as u32,
                &acc.h,
                &acc.hu,
                &acc.hv,
                self.damping(),
                &line_local,
                shallow,
                &st.mass,
                &st.inv_area,
                h,
                hu,
                hv,
                &mut clips,
                &mut breaks,
            )
        }
        .expect("finish");
    }
}

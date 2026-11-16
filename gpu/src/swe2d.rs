//! The 2D SWE on the GPU (P2.7, `notes/gpu.md`): a device-resident state
//! ([`DeviceState`], dgcom's [`Integrable`]) and a physics module
//! ([`GpuSWE2D`], dgcom's [`PhysicsModule`]), so that dgcom's own
//! `Simulation` and SSP-RK steppers drive the GPU.
//!
//! The split-form RHS of `SWEFormulation2D::EntropyStable` and of `WetDry`
//! (the subcells in elements with a node shallower than the subcell depth,
//! the hydrostatic HLL at faces) with a bed, walls (`Reflective2D`, called
//! as the CPU calls it), characteristic open boundaries with an external
//! state uniform along each (a harmonic tide, still water), Coriolis; global
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
use cuda_device::{DisjointSlice, device, kernel, thread};
use cuda_host::cuda_module;
use dgcom::SWEState2D;
use dgcom::boundary::{
    BCContext2D, DeviceBoundary2D, ElevationOnly, ExternalState, ExternalStateProvider,
    Reflective2D, SWEBoundaryCondition2D, characteristic_boundary_state,
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

/// The external state of every open boundary at the time of a stage,
/// evaluated on the host (`ExternalStateProvider::uniform_state`) and passed
/// to the edge kernels by value.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct OpenStates {
    /// η, u, v of each boundary's external state
    pub state: [[f64; 3]; MAX_OPEN],
    /// Bit 0: the state has a velocity (else elevation only); bit 1:
    /// elevation-only data enter as a wave (`ElevationOnly::IncomingWave`)
    pub flags: [u32; MAX_OPEN],
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

/// Volume node of face node `fi` of `face` (`DGOperators2D::face_nodes`:
/// faces 2 and 3 list theirs in reverse). A formula, not a table, so that
/// with the loops unrolled the element's nodes are indexed by constants and
/// stay in registers.
#[inline]
const fn face_node(face: usize, fi: usize) -> usize {
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

/// `SplitFormSWE2D::line_subcells`: the subcell finite-volume update (times
/// J) along `line` of `direction` (0: an r-line, its nodes `line·N1 + a`;
/// 1: an s-line, `a·N1 + line`), its end subcells taking their outer slope
/// neighbour from `ends` (the node, its distance, whether there is one).
/// Not generic, so that it inlines (a generic `#[device]` function stays a
/// call), and the GLL data are arrays, so that its loops have constant trip
/// counts.
#[allow(clippy::too_many_arguments)]
#[device]
// The device backend makes this alwaysinline (no call in the PTX); the host
// compiler, which sees an exported symbol, ignores it.
#[allow(unused_attributes)]
#[inline(always)]
fn line_subcells(
    nodes: &[SWENodeState2D; NN],
    rhs: &mut [SWEState2D; NN],
    metric: &[(f64, f64); N1],
    line: usize,
    direction: usize,
    ends: [(SWENodeState2D, f64, bool); 2],
    affine: bool,
    w: &[f64; N1],
    xi: &[f64; N1],
    d1: &[f64; NN],
    g: f64,
    h_min: f64,
    h_dry: f64,
) {
    let idx = |a: usize| {
        if direction == 0 {
            line * N1 + a
        } else {
            a * N1 + line
        }
    };
    // Interface metrics (the telescoping sum of the flux-differencing term)
    let mut interfaces = [(0.0, 0.0); N1 + 1];
    telescoped_interfaces(w, d1, metric, affine, &mut interfaces);

    let (first, last) = (idx(0), idx(N1 - 1));
    let (q_first, q_last) = (nodes[first], nodes[last]);
    let (m_first, m_last) = (metric[0], metric[N1 - 1]);
    rhs[first] = rhs[first] + (1.0 / w[0]) * wintermeyer_flux_2d(&q_first, &q_first, m_first, g);
    rhs[last] = rhs[last] - (1.0 / w[N1 - 1]) * wintermeyer_flux_2d(&q_last, &q_last, m_last, g);

    // Face states. Subcell a spans [x_a, x_a + w_a] with x_0 = −1
    let mut faces = [(SWESide2D::default(), SWESide2D::default()); N1];
    let mut x_left = -1.0;
    #[unroll]
    for a in 0..N1 {
        let q = SWESide2D::from(&nodes[idx(a)]);
        let x_right = x_left + w[a];
        let (left, x_l, has_left) = if a > 0 {
            (SWESide2D::from(&nodes[idx(a - 1)]), xi[a - 1], true)
        } else {
            (SWESide2D::from(&ends[0].0), -1.0 - ends[0].1, ends[0].2)
        };
        let (right, x_r, has_right) = if a + 1 < N1 {
            (SWESide2D::from(&nodes[idx(a + 1)]), xi[a + 1], true)
        } else {
            (SWESide2D::from(&ends[1].0), 1.0 + ends[1].1, ends[1].2)
        };
        faces[a] = if has_left && has_right {
            reconstruct_subcell(
                [left, q, right],
                [x_l, xi[a], x_r],
                (x_left, x_right),
                h_dry,
            )
        } else {
            (q, q)
        };
        x_left = x_right;
    }

    #[unroll]
    for a in 0..N1 - 1 {
        let (ia, ib) = (idx(a), idx(a + 1));
        let m = interfaces[a + 1];
        let (f_a, f_b) = hydrostatic_hll(&faces[a].1, &faces[a + 1].0, m, g, h_min);
        rhs[ia] = rhs[ia] - (1.0 / w[a]) * f_a;
        rhs[ib] = rhs[ib] + (1.0 / w[a + 1]) * f_b;
    }

    // Bed-slope term, zero in subcells without a slope (B_L = B_R), and the
    // metric balance term, zero on affine elements
    #[unroll]
    for a in 0..N1 {
        let ia = idx(a);
        let (left, right) = faces[a];
        let w_a = w[a];
        let bed = -0.5 * g * (left.h + right.h) * (right.b - left.b) / w_a;
        let (force_x, force_y) = if affine {
            let m = metric[0];
            (bed * m.0, bed * m.1)
        } else {
            let (m_l, m_r) = (interfaces[a], interfaces[a + 1]);
            let (h_a, b_a) = (nodes[ia].h, nodes[ia].b);
            let balance =
                0.25 * g * ((right.h + h_a) * (b_a - right.b) + (left.h + h_a) * (b_a - left.b))
                    / w_a;
            (
                bed * 0.5 * (m_l.0 + m_r.0) + balance * (m_r.0 - m_l.0),
                bed * 0.5 * (m_l.1 + m_r.1) + balance * (m_r.1 - m_l.1),
            )
        };
        rhs[ia] = rhs[ia] + SWEState2D::new(0.0, force_x, force_y);
    }
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
    affine: &[u32],
    egeom: &[[f64; EG]],
    jac: &[f64],
    fgeom: &[[f64; 3]],
    element_edges: &[[u32; 4]],
    edges: &[[u32; 4]],
    xi: &[f64],
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
    // Element height normal to a face at a face node, in physical units / 2:
    // J / sJ there (constant on a parallelogram)
    let height = |e: usize, f: usize, fi: usize, node: usize| {
        if affine[e] != 0 {
            egeom[e][17] / egeom[e][7 + 3 * f]
        } else {
            jac[e * NN + node] / fgeom[(e * 4 + f) * N1 + fi][2]
        }
    };
    let nb_node = face_node(nb_face, N1 - 1 - fi);
    // One node into the neighbour, normal to its face
    let inner = match nb_face {
        0 => nb_node + N1,
        1 => nb_node - 1,
        2 => nb_node - N1,
        _ => nb_node + 1,
    };
    let own_node = face_node(face, fi);
    let distance =
        (xi[1] - xi[0]) * height(nb, nb_face, N1 - 1 - fi, nb_node) / height(k, face, fi, own_node);
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

/// `SplitFormSWE2D::boundary_flux` at a characteristic open boundary:
/// dgcom's `characteristic_boundary_state` from the interior trace and the
/// external state of open boundary `i`, its physical flux `F(q_b)·n`.
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
    open: &OpenStates,
    i: usize,
) -> SWEState2D {
    let [eta, u, v] = open.state[i];
    let flags = open.flags[i];
    let external = if flags & 1 != 0 {
        ExternalState::new(eta, u, v)
    } else {
        ExternalState::elevation(eta)
    };
    let elevation_only = if flags & 2 != 0 {
        ElevationOnly::IncomingWave
    } else {
        ElevationOnly::AtRest
    };
    // The position is not read by a uniform external state
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

/// `SplitFormSWE2D::boundary_flux` at face node `fi` of boundary edge `e`:
/// a wall's ghost state or an open boundary's state at its element's stage
/// time (its level's entry of `boundary`), through the hydrostatic HLL if
/// `hll`.
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
    level: &[u32],
    (h, hu, hv, bed): (&[f64], &[f64], &[f64], &[f64]),
    affine: &[u32],
    egeom: &[[f64; EG]],
    fgeom: &[[f64; 3]],
    edges: &[[u32; 4]],
) -> [f64; 3] {
    let [le, lf, _, rf] = edges[e];
    let (le, lf) = (le as usize, lf as usize);
    let StageLevel { time, open } = boundary[level_of(level, le)];
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
        open_flux(
            &q_left,
            state,
            bed[left],
            normal,
            time,
            g,
            h_min,
            &open,
            rf as usize,
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
    /// entry of its level.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn boundary_fluxes(
        hll: u32,
        g: f64,
        h_min: f64,
        wall_h_min: f64,
        boundary: &[StageLevel],
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

    /// `SplitFormSWE2D::element_rhs` of the listed elements (every element
    /// if `list` is empty) without a node shallower than `subcell_depth`
    /// (one thread per element; the others are `element_rhs_subcells`'):
    /// the flux-differencing volume term and the lifted face terms, divided
    /// by J.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn element_rhs_volume(
        g: f64,
        h_min: f64,
        subcell_depth: f64,
        list: &[u32],
        n: u32,
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
        let Some(k) = item(thread::index_1d().get(), list, n) else {
            return;
        };
        if is_subcell_element(k, h, subcell_depth) {
            return;
        }
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

    /// `SplitFormSWE2D::element_rhs` of the listed elements (every element
    /// if `list` is empty) with a node shallower than `subcell_depth`
    /// (`WetDry`, one thread per element; the others return at once): the
    /// subcell finite volumes, with the neighbours' outer nodes, and the
    /// lifted face terms, divided by J.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn element_rhs_subcells(
        g: f64,
        h_min: f64,
        subcell_depth: f64,
        h_dry: f64,
        list: &[u32],
        n: u32,
        h: &[f64],
        hu: &[f64],
        hv: &[f64],
        bed: &[f64],
        affine: &[u32],
        egeom: &[[f64; EG]],
        contra: &[[f64; 4]],
        jinv: &[f64],
        jac: &[f64],
        fgeom: &[[f64; 3]],
        d1: &[f64],
        w: &[f64],
        xi: &[f64],
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
        let Some(k) = item(thread::index_1d().get(), list, n) else {
            return;
        };
        if !is_subcell_element(k, h, subcell_depth) {
            return;
        }
        let mut nodes = [SWENodeState2D::default(); NN];
        #[unroll]
        for i in 0..NN {
            nodes[i] = node_state(h, hu, hv, bed, k * NN + i, h_min);
        }
        let mut rhs = [SWEState2D::zero(); NN];

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
        // The outer node of face node `fi` of `face`, read where its line
        // needs it (each is the end of one line only): the 12 of an element
        // need not be held at once
        let outer = |face: usize, fi: usize| {
            outer_node(
                k,
                face,
                fi,
                (h, hu, hv, bed),
                affine,
                egeom,
                jac,
                fgeom,
                element_edges,
                edges,
                &xi_a,
                h_min,
            )
        };

        let aff = affine[k] != 0;
        let eg = egeom[k];
        let (affine_r, affine_s) = ((eg[0], eg[1]), (eg[2], eg[3]));
        let mut metric = [(0.0, 0.0); N1];
        #[unroll]
        for line in 0..N1 {
            let r_line = |a: usize| line * N1 + a;
            let s_line = |j: usize| j * N1 + line;
            // Line ends on faces 3/1 (r-lines) and 0/2 (s-lines); faces 2
            // and 3 list their nodes in reverse
            let (rev, fwd) = (N1 - 1 - line, line);
            #[unroll]
            for a in 0..N1 {
                metric[a] = if aff {
                    affine_r
                } else {
                    let c = contra[k * NN + r_line(a)];
                    (c[0], c[1])
                };
            }
            let ends = [outer(3, rev), outer(1, fwd)];
            line_subcells(
                &nodes, &mut rhs, &metric, line, 0, ends, aff, &w_a, &xi_a, &d1_a, g, h_min, h_dry,
            );
            #[unroll]
            for j in 0..N1 {
                metric[j] = if aff {
                    affine_s
                } else {
                    let c = contra[k * NN + s_line(j)];
                    (c[2], c[3])
                };
            }
            let ends = [outer(0, fwd), outer(2, rev)];
            line_subcells(
                &nodes, &mut rhs, &metric, line, 1, ends, aff, &w_a, &xi_a, &d1_a, g, h_min, h_dry,
            );
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
    /// The device's name and compute capability
    pub name: String,
}

impl Gpu {
    /// CUDA device `ordinal`, its default stream and the kernels.
    pub fn on(ordinal: usize) -> Result<Arc<Self>, Box<dyn std::error::Error>> {
        let ctx = CudaContext::new(ordinal)?;
        let (major, minor) = ctx.compute_capability()?;
        let name = format!("{} (sm_{major}{minor})", ctx.device_name()?);
        let stream = ctx.default_stream();
        let module = kernels::load(&ctx)?;
        Ok(Arc::new(Self {
            stream,
            module,
            name,
        }))
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
    /// `J` at every node
    jac: DeviceBuffer<f64>,
    fgeom: DeviceBuffer<[f64; 3]>,
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
            jac: DeviceBuffer::from_host(stream, &jac)?,
            fgeom: DeviceBuffer::from_host(stream, &fgeom)?,
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
/// (`CharacteristicOBC` with a uniform external state), the Coriolis force,
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
    /// elevation-only data, in the order of [`OpenStates`]
    opens: Vec<(&'a dyn ExternalStateProvider, ElevationOnly)>,
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

/// A stage's boundary data ([`StageLevel`] per level) on the device: each
/// stage's copied from a host slot of a ring, which outlives its
/// asynchronous copy (the stream is synchronized each time the ring wraps).
struct BoundaryUploads {
    device: DeviceBuffer<StageLevel>,
    host: Vec<[StageLevel; LEVELS]>,
    next: usize,
}

impl BoundaryUploads {
    /// Host slots: stage passes between two synchronizations.
    const SLOTS: usize = 1024;

    fn new(gpu: &Gpu) -> Result<Self, String> {
        Ok(Self {
            device: DeviceBuffer::zeroed(&gpu.stream, LEVELS).map_err(|e| e.to_string())?,
            host: vec![[StageLevel::default(); LEVELS]; Self::SLOTS],
            next: 0,
        })
    }

    /// `levels` on the device, for the kernels enqueued next on `stream`.
    fn upload(
        &mut self,
        stream: &Stream,
        levels: &[StageLevel; LEVELS],
    ) -> &DeviceBuffer<StageLevel> {
        if self.next == Self::SLOTS {
            // Every copy from the slots has completed
            stream.synchronize().expect("synchronize");
            self.next = 0;
        }
        let slot = &mut self.host[self.next];
        *slot = *levels;
        self.next += 1;
        // SAFETY: the slot is not written again before the stream is
        // synchronized (above), so it outlives the copy; the buffer is
        // LEVELS long. Copies and kernels are ordered on the stream, so the
        // kernels of an earlier stage have read the buffer before it changes.
        unsafe { self.device.copy_from_host_async_unchecked(stream, slot) }
            .expect("boundary upload");
        &self.device
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
                    provider,
                    elevation_only,
                } => {
                    if opens.len() == MAX_OPEN {
                        return Err(format!("more than {MAX_OPEN} open boundaries"));
                    }
                    opens.push((provider, elevation_only));
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
        let coriolis = form.source.map(|source| match source {
            DeviceSource2D::Coriolis(c) => c,
        });
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
            rates: Mutex::new(zeroed(mesh.n_elements)?),
            line_local_depth: form.line_local_depth,
            positivity_h_dry: form.positivity_h_dry,
            schedule: Mutex::new(Schedule::empty(gpu)?),
            line_local: Mutex::new(DeviceBuffer::zeroed(stream, 0).map_err(|e| e.to_string())?),
            breaks: Mutex::new(
                DeviceBuffer::zeroed(stream, mesh.n_elements).map_err(|e| e.to_string())?,
            ),
            no_list: DeviceBuffer::zeroed(stream, 0).map_err(|e| e.to_string())?,
            no_rows: Mutex::new([zeroed(0)?, zeroed(0)?, zeroed(0)?]),
            boundary: Mutex::new(BoundaryUploads::new(gpu)?),
        })
    }

    /// Elements the wet/dry correction emptied, over every stage so far
    /// (`SWEPhysics2D::negative_depth_clips`).
    pub fn negative_depth_clips(&self) -> usize {
        let clips = self.clips.lock().expect("clip counts");
        let clips = clips.to_host_vec(&self.gpu.stream).expect("download");
        clips.iter().map(|&c| c as usize).sum()
    }

    /// The open boundaries' external states at `time`.
    fn open_states(&self, time: f64) -> OpenStates {
        let mut open = OpenStates::default();
        for (i, (provider, elevation_only)) in self.opens.iter().enumerate() {
            let state = provider
                .uniform_state(time)
                .expect("a uniform external state (DeviceBoundary2D::Characteristic)");
            let (u, v) = state.velocity.unwrap_or((0.0, 0.0));
            open.state[i] = [state.eta, u, v];
            open.flags[i] = u32::from(state.velocity.is_some())
                | (u32::from(*elevation_only == ElevationOnly::IncomingWave) << 1);
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
    /// kernels. `level` gives a boundary edge its element's entry of
    /// `boundary` (empty: level 0).
    fn rhs_where(
        &self,
        state: &DeviceState,
        boundary: &[StageLevel; LEVELS],
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
        let uploaded = uploads.upload(&gpu.stream, boundary);
        let (coriolis, f0) = self.coriolis_args();
        let elements_cfg = per_element(n_elements);
        let n_elements = n_elements as u32;
        // SAFETY: every buffer is sized for this mesh (`Statics::upload`, the
        // state's), the lists hold distinct edges and elements of it, and the
        // kernels index within its elements, edges and nodes.
        unsafe {
            if n_interior > 0 {
                let cfg = one_dimensional(n_interior);
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
                        one_dimensional(n_boundary * N1),
                        u32::from(hll),
                        self.g,
                        self.h_min,
                        self.wall_h_min,
                        uploaded,
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
                    subcell_depth,
                    element_list,
                    n_elements,
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
                let [out_h, out_hu, out_hv] = out.fields_mut();
                gpu.module
                    .element_rhs_subcells(
                        &gpu.stream,
                        elements_cfg,
                        self.g,
                        self.h_min,
                        subcell_depth,
                        h_dry,
                        element_list,
                        n_elements,
                        &state.h,
                        &state.hu,
                        &state.hv,
                        &s.bed,
                        &s.affine,
                        &s.egeom,
                        &s.contra,
                        &s.jinv,
                        &s.jac,
                        &s.fgeom,
                        &s.d1,
                        &s.w,
                        &s.xi,
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
                    .expect("element RHS (subcells)");
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
        let mut boundary = [StageLevel::default(); LEVELS];
        boundary[0] = StageLevel {
            time,
            open: self.open_states(time),
        };
        let s = &self.statics;
        self.rhs_where(
            state,
            &boundary,
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
        // Every level's stage time and open boundary states
        let mut boundary = [StageLevel::default(); LEVELS];
        for (l, entry) in boundary.iter_mut().enumerate().take(finest + 1) {
            let time = element_stage(
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
            *entry = StageLevel {
                time,
                open: self.open_states(time),
            };
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
            &boundary,
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
                per_element(n_elements),
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
                per_element(n),
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

//! The 2D SWE on the GPU (P2.7, `notes/gpu.md`): a device-resident state
//! ([`DeviceState`], dgcom's [`Integrable`]) and a physics module
//! ([`GpuSWE2D`], dgcom's [`PhysicsModule`]), so that dgcom's own
//! `Simulation` and SSP-RK steppers drive the GPU.
//!
//! The split-form RHS of `SWEFormulation2D::EntropyStable` and of `WetDry`
//! (the subcells in elements with a node shallower than the subcell depth,
//! the hydrostatic HLL at faces) with a bed, walls (`Reflective2D`, called
//! as the CPU calls it), no sources yet, global steps. The kernels follow
//! `SplitFormSWE2D` (`src/solver/rhs/swe_2d_split_form.rs`) operation for
//! operation and call dgcom's pointwise physics (the two-point flux, the
//! hydrostatic HLL, the subcell reconstruction, the interface metrics, the
//! wall's ghost state), so a build without multiply–add contraction
//! (`cargo oxide run --no-fmad`) gives the CPU's bits. One thread per edge and one per element; every loop unrolls (P2's
//! bounds are constants, the face nodes a formula), so an element's nodes and
//! RHS stay in registers.

use std::sync::{Arc, Mutex};

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig, Stream};
use cuda_device::{DisjointSlice, device, kernel, thread};
use cuda_host::cuda_module;
use dgcom::SWEState2D;
use dgcom::boundary::{BCContext2D, Reflective2D, SWEBoundaryCondition2D};
use dgcom::equations::ShallowWater2D;
use dgcom::flux::{
    SWENodeState2D, entropy_stable_dissipation_2d, wintermeyer_bed_interface_term_2d,
    wintermeyer_flux_2d,
};
use dgcom::mesh::{Bathymetry2D, BoundaryTag, Mesh2D};
use dgcom::operators::{DGOperators2D, GeometricFactors2D};
use dgcom::physics::{PhysicsModule, PhysicsModuleInfo};
use dgcom::solver::rhs::subcells::telescoped_interfaces;
use dgcom::solver::{
    PositivityBound, SWESide2D, SWESolution2D, WetDryConfig, damp_node, dt_from_reference_rate,
    hydrostatic_hll, node_reference_rate, reconstruct_subcell, wet_dry_element,
};
use dgcom::source::{CageNode, ManningFriction2D};
use dgcom::time::Integrable;
use dgcom::time::SspScheme;
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
/// J) along the line of nodes `idx(0..N1)`, its end subcells taking their
/// outer slope neighbour from `ends` (the node, its distance, whether there
/// is one).
#[device]
#[allow(clippy::too_many_arguments)]
fn line_subcells<I: Fn(usize) -> usize>(
    nodes: &[SWENodeState2D; NN],
    rhs: &mut [SWEState2D; NN],
    metric: &[(f64, f64); N1],
    idx: I,
    ends: [(SWENodeState2D, f64, bool); 2],
    affine: bool,
    w: &[f64],
    xi: &[f64],
    d1: &[f64],
    g: f64,
    h_min: f64,
    h_dry: f64,
) {
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

/// `(∇r, ∇s, |∇r|, |∇s|)` of a node from `[r_x, r_y, s_x, s_y]`, as
/// `element_max_reference_rate` forms it.
#[inline]
fn rate_metric(grads: [f64; 4]) -> ((f64, f64), (f64, f64), f64, f64) {
    let norm_r = (grads[0] * grads[0] + grads[1] * grads[1]).sqrt();
    let norm_s = (grads[2] * grads[2] + grads[3] * grads[3]).sqrt();
    ((grads[0], grads[1]), (grads[2], grads[3]), norm_r, norm_s)
}

/// `SplitFormSWE2D::edge_fluxes` of edge `e`, into `slot`: both sides' F*
/// of an interior edge (left first, each in its own face-node order), the
/// hydrostatic HLL if `HLL`, else the entropy-stable flux; at a wall
/// (`boundary_flux`) the left side's F* from the wall's ghost state.
#[device]
#[allow(clippy::too_many_arguments)]
fn edge_body<const HLL: bool>(
    e: usize,
    g: f64,
    h_min: f64,
    wall_h_min: f64,
    time: f64,
    (h, hu, hv, bed): (&[f64], &[f64], &[f64], &[f64]),
    affine: &[u32],
    egeom: &[[f64; EG]],
    fgeom: &[[f64; 3]],
    edges: &[[u32; 4]],
    slot: &mut [[f64; 3]; NF],
) {
    let [le, lf, re, rf] = edges[e];
    let (le, lf) = (le as usize, lf as usize);
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
        if re == NO_NEIGHBOUR {
            let state = SWEState2D::new(h[left], hu[left], hv[left]);
            let f = wall_flux(
                &q_left, state, bed[left], normal, time, g, h_min, wall_h_min, HLL,
            );
            slot[fi] = [f.h, f.hu, f.hv];
        } else {
            let (re, rf) = (re as usize, rf as usize);
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
/// `out` (`[h, hu, hv]`).
#[device]
// The device backend makes this alwaysinline (no call in the PTX); the host
// compiler, which sees an exported symbol, ignores it.
#[allow(unused_attributes)]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
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
        // SAFETY: thread k writes only its own element's rows
        // (k·NN..(k + 1)·NN), and k < n_elements.
        unsafe {
            *out_h.get_unchecked_mut(n) = j_inv * r.h;
            *out_hu.get_unchecked_mut(n) = j_inv * r.hu;
            *out_hv.get_unchecked_mut(n) = j_inv * r.hv;
        }
    }
}

// =============================================================================
// Kernels
// =============================================================================

#[cuda_module]
pub mod kernels {
    use super::*;

    /// The entropy-stable F* of every edge (one thread per edge).
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn edge_fluxes_entropy_stable(
        g: f64,
        h_min: f64,
        wall_h_min: f64,
        time: f64,
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
        let idx = thread::index_1d();
        let e = idx.get();
        if let Some(slot) = out.get_mut(idx) {
            edge_body::<false>(
                e,
                g,
                h_min,
                wall_h_min,
                time,
                (h, hu, hv, bed),
                affine,
                egeom,
                fgeom,
                edges,
                slot,
            );
        }
    }

    /// The hydrostatic HLL F* of every edge (`WetDry`, one thread per edge).
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn edge_fluxes_hll(
        g: f64,
        h_min: f64,
        wall_h_min: f64,
        time: f64,
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
        let idx = thread::index_1d();
        let e = idx.get();
        if let Some(slot) = out.get_mut(idx) {
            edge_body::<true>(
                e,
                g,
                h_min,
                wall_h_min,
                time,
                (h, hu, hv, bed),
                affine,
                egeom,
                fgeom,
                edges,
                slot,
            );
        }
    }

    /// `SplitFormSWE2D::element_rhs` of the elements without a node
    /// shallower than `subcell_depth` (one thread per element; the others
    /// are `element_rhs_subcells`'): the flux-differencing volume term and
    /// the lifted face terms, divided by J.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn element_rhs_volume(
        g: f64,
        h_min: f64,
        subcell_depth: f64,
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
        mut out_h: DisjointSlice<f64>,
        mut out_hu: DisjointSlice<f64>,
        mut out_hv: DisjointSlice<f64>,
    ) {
        let k = thread::index_1d().get();
        if k >= element_edges.len() || is_subcell_element(k, h, subcell_depth) {
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
            &mut out_h,
            &mut out_hu,
            &mut out_hv,
        );
    }

    /// `SplitFormSWE2D::element_rhs` of the elements with a node shallower
    /// than `subcell_depth` (`WetDry`, one thread per element; the others
    /// return at once): the subcell finite volumes, with the neighbours'
    /// outer nodes, and the lifted face terms, divided by J.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn element_rhs_subcells(
        g: f64,
        h_min: f64,
        subcell_depth: f64,
        h_dry: f64,
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
        mut out_h: DisjointSlice<f64>,
        mut out_hu: DisjointSlice<f64>,
        mut out_hv: DisjointSlice<f64>,
    ) {
        let k = thread::index_1d().get();
        if k >= element_edges.len() || !is_subcell_element(k, h, subcell_depth) {
            return;
        }
        let mut nodes = [SWENodeState2D::default(); NN];
        #[unroll]
        for i in 0..NN {
            nodes[i] = node_state(h, hu, hv, bed, k * NN + i, h_min);
        }
        let mut rhs = [SWEState2D::zero(); NN];

        let mut outer = [(SWENodeState2D::default(), 0.0, false); 4 * N1];
        #[unroll]
        for face in 0..4 {
            #[unroll]
            for fi in 0..N1 {
                outer[face * N1 + fi] = outer_node(
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
                    xi,
                    h_min,
                );
            }
        }

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
            let ends = [outer[3 * N1 + rev], outer[N1 + fwd]];
            line_subcells(
                &nodes, &mut rhs, &metric, r_line, ends, aff, w, xi, d1, g, h_min, h_dry,
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
            let ends = [outer[fwd], outer[2 * N1 + rev]];
            line_subcells(
                &nodes, &mut rhs, &metric, s_line, ends, aff, w, xi, d1, g, h_min, h_dry,
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
            &mut out_h,
            &mut out_hu,
            &mut out_hv,
        );
    }

    /// `apply_wet_dry_correction_all` of element `k` (one thread per
    /// element): dgcom's `wet_dry_element` on its rows, with its nodal
    /// masses and 1/area.
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
        wet_dry_element(&mut eh, &mut ehu, &mut ehv, &m, inv_area[k], &config);
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
    /// dgcom's `damp_node` with Manning friction (if `friction`) and the
    /// thin-layer relaxation (if `wet_dry`), no cages.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn implicit_damping(
        dt: f64,
        h_min: f64,
        friction: u32,
        g: f64,
        manning_n: f64,
        friction_h_min: f64,
        wet_dry: u32,
        h_dry: f64,
        relaxation_time: f64,
        max_velocity: f64,
        wet_dry_g: f64,
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
        let law = ManningFriction2D {
            g,
            manning_n,
            h_min: friction_h_min,
        };
        let config = WetDryConfig {
            h_dry: Depth::new(h_dry),
            relaxation_time,
            max_velocity,
            g: wet_dry_g,
        };
        let no_cages: &[CageNode] = &[];
        // SAFETY: thread n reads and writes only node n.
        let (mut u, mut v) = unsafe { (*hu.get_unchecked_mut(n), *hv.get_unchecked_mut(n)) };
        damp_node(
            if friction != 0 { Some(&law) } else { None },
            no_cages,
            if wet_dry != 0 { Some(&config) } else { None },
            Depth::new(h_min),
            n,
            h[n],
            &mut u,
            &mut v,
            SWEState2D::new(from_h[n], from_hu[n], from_hv[n]),
            dt,
        );
        // SAFETY: as above.
        unsafe {
            *hu.get_unchecked_mut(n) = u;
            *hv.get_unchecked_mut(n) = v;
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
}

impl Gpu {
    /// Device 0, its default stream and the kernels.
    pub fn new() -> Result<Arc<Self>, Box<dyn std::error::Error>> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let module = kernels::load(&ctx)?;
        Ok(Arc::new(Self { stream, module }))
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
    /// `[r_x, r_y, s_x, s_y]` of every node, and of every parallelogram
    grads: DeviceBuffer<[f64; 4]>,
    affine_grads: DeviceBuffer<[f64; 4]>,
    /// Nodal masses (GLL weight × J) and every element's 1/area
    mass: DeviceBuffer<f64>,
    inv_area: DeviceBuffer<f64>,
}

impl Statics {
    fn upload(
        gpu: &Gpu,
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        geom: &GeometricFactors2D,
        bathymetry: &Bathymetry2D,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let stream = &gpu.stream;
        let n_el = mesh.n_elements;
        assert_eq!(ops.n_1d, N1, "the kernels are compiled for P{ORDER}");
        assert!(
            mesh.edges
                .iter()
                .all(|e| e.right.is_some()
                    || e.boundary_tag.unwrap_or_default() == BoundaryTag::Wall),
            "the kernels take walls only so far: no open boundaries"
        );
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
                bed.push(bathymetry.get(ElementIndex::new(k), i));
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
                let (right, right_face) = e.right.map_or((NO_NEIGHBOUR, NO_NEIGHBOUR), |r| {
                    (r.element as u32, r.face as u32)
                });
                [e.left.element as u32, e.left.face as u32, right, right_face]
            })
            .collect();
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
            grads: DeviceBuffer::from_host(stream, &grads)?,
            affine_grads: DeviceBuffer::from_host(stream, &affine_grads)?,
            mass: DeviceBuffer::from_host(stream, &mass)?,
            inv_area: DeviceBuffer::from_host(stream, &inv_area)?,
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

/// The 2D SWE physics module on the GPU: `SWEPhysics2D` with a split
/// formulation ([`Formulation`]), a bed, walls (`Reflective2D`), the wet/dry
/// correction after every stage and the thin-layer relaxation
/// ([`Self::with_wet_dry`]; its positivity limiter is the redundant one of
/// the case runner, `Positivity(h_dry)`), Manning friction applied
/// point-implicitly ([`Self::with_friction`]), the positivity-bounded time
/// step; no sources yet (Phase 1, `notes/gpu.md`).
pub struct GpuSWE2D {
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
    wet_dry: Option<WetDryConfig>,
    friction: Option<ManningFriction2D>,
    /// Every element's time step (`compute_dt_ssp` of a wet/dry run)
    element_dts: Mutex<DeviceBuffer<f64>>,
    /// Both sides' F* of every edge, written by the edge kernel for the
    /// element kernel
    faces: Mutex<DeviceBuffer<[[f64; 3]; NF]>>,
    /// Every element's largest reference rate (the time step)
    rates: Mutex<DeviceBuffer<f64>>,
}

impl GpuSWE2D {
    pub fn new(
        gpu: &Arc<Gpu>,
        mesh: Arc<Mesh2D>,
        ops: Arc<DGOperators2D>,
        geom: Arc<GeometricFactors2D>,
        bathymetry: &Bathymetry2D,
        equation: &ShallowWater2D,
        formulation: Formulation,
        wall: &Reflective2D,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let statics = Statics::upload(gpu, &mesh, &ops, &geom, bathymetry)?;
        let faces = DeviceBuffer::zeroed(&gpu.stream, mesh.n_edges)?;
        let rates = DeviceBuffer::zeroed(&gpu.stream, mesh.n_elements)?;
        let element_dts = DeviceBuffer::zeroed(&gpu.stream, mesh.n_elements)?;
        Ok(Self {
            gpu: gpu.clone(),
            mesh,
            ops,
            geom,
            statics,
            g: equation.g,
            h_min: equation.h_min.meters(),
            formulation,
            wall_h_min: wall.h_min,
            wet_dry: None,
            friction: None,
            element_dts: Mutex::new(element_dts),
            faces: Mutex::new(faces),
            rates: Mutex::new(rates),
        })
    }
}

impl GpuSWE2D {
    /// Wetting and drying: the wet/dry correction after every stage, the
    /// thin-layer relaxation in the implicit damping, the time step under
    /// the positivity bound (`SWEPhysics2D::with_wet_dry` with the
    /// positivity limiter at `config.h_dry`).
    pub fn with_wet_dry(mut self, config: WetDryConfig) -> Self {
        self.wet_dry = Some(config);
        self
    }

    /// Manning bottom friction, point-implicit in every stage
    /// (`SWEPhysics2D::with_implicit_friction`).
    pub fn with_friction(mut self, friction: ManningFriction2D) -> Self {
        self.friction = Some(friction);
        self
    }
}

impl PhysicsModuleInfo for GpuSWE2D {
    fn name(&self) -> &'static str {
        "swe-2d-gpu"
    }

    fn description(&self) -> &str {
        "2D Shallow Water Equations on the GPU (fully wet split form)"
    }

    fn n_variables(&self) -> usize {
        3
    }

    fn variable_names(&self) -> &[&'static str] {
        &["h", "hu", "hv"]
    }
}

impl PhysicsModule<DeviceState> for GpuSWE2D {
    fn compute_rhs(&self, state: &DeviceState, time: f64) -> DeviceState {
        let mut out = state.zeroed_like();
        self.compute_rhs_into(state, time, &mut out);
        out
    }

    fn compute_rhs_into(&self, state: &DeviceState, time: f64, out: &mut DeviceState) {
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
        let edges_cfg = one_dimensional(self.mesh.n_edges);
        let elements_cfg = per_element(self.mesh.n_elements);
        // SAFETY: every buffer is sized for this mesh (`Statics::upload`, the
        // state's), and the kernels index within its elements, edges and nodes.
        unsafe {
            if hll {
                gpu.module.edge_fluxes_hll(
                    &gpu.stream,
                    edges_cfg,
                    self.g,
                    self.h_min,
                    self.wall_h_min,
                    time,
                    &state.h,
                    &state.hu,
                    &state.hv,
                    &s.bed,
                    &s.affine,
                    &s.egeom,
                    &s.fgeom,
                    &s.edges,
                    &mut *faces,
                )
            } else {
                gpu.module.edge_fluxes_entropy_stable(
                    &gpu.stream,
                    edges_cfg,
                    self.g,
                    self.h_min,
                    self.wall_h_min,
                    time,
                    &state.h,
                    &state.hu,
                    &state.hv,
                    &s.bed,
                    &s.affine,
                    &s.egeom,
                    &s.fgeom,
                    &s.edges,
                    &mut *faces,
                )
            }
            .expect("edge fluxes");
            let [out_h, out_hu, out_hv] = out.fields_mut();
            gpu.module
                .element_rhs_volume(
                    &gpu.stream,
                    elements_cfg,
                    self.g,
                    self.h_min,
                    subcell_depth,
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
                    &*faces,
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
                        &*faces,
                        out_h,
                        out_hu,
                        out_hv,
                    )
                    .expect("element RHS (subcells)");
            }
        }
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
                &mut *rates,
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
            )
        }
        .expect("wet/dry correction");
    }

    /// Point-implicit Manning friction and thin-layer relaxation.
    fn implicit_damping(&self, stage: &mut DeviceState, from: &DeviceState, dt: f64) {
        if self.friction.is_none() && self.wet_dry.is_none() {
            return;
        }
        let gpu = &self.gpu;
        let (g, manning_n, friction_h_min) = self
            .friction
            .as_ref()
            .map_or((0.0, 0.0, 0.0), |f| (f.g, f.manning_n, f.h_min));
        let (h_dry, relaxation_time, max_velocity, wd_g) =
            self.wet_dry.as_ref().map_or((0.0, 0.0, 0.0, 0.0), |c| {
                (c.h_dry.meters(), c.relaxation_time, c.max_velocity, c.g)
            });
        let n_nodes = stage.h.len();
        let DeviceState { h, hu, hv, .. } = stage;
        // SAFETY: one thread per node of the state; `from` is another state.
        unsafe {
            gpu.module.implicit_damping(
                &gpu.stream,
                one_dimensional(n_nodes),
                dt,
                self.h_min,
                u32::from(self.friction.is_some()),
                g,
                manning_n,
                friction_h_min,
                u32::from(self.wet_dry.is_some()),
                h_dry,
                relaxation_time,
                max_velocity,
                wd_g,
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
                &mut *dts,
            )
        }
        .expect("element time steps");
        let dts = dts.to_host_vec(&gpu.stream).expect("download");
        dts.iter().copied().fold(f64::INFINITY, f64::min)
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

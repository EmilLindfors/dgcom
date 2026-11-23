//! The 3D stages' transport on the GPU (P2.7 Phase 4, `notes/gpu.md`): the
//! layer transports and their Ω (dgcom's `LayerTransport::compute`), and the
//! advection of a pair of fields by them (`apply_tracer_transport_3d` for T
//! and S, `apply_momentum_transport_3d` for u and v,
//! `apply_w_point_transport_3d` for the turbulence's k and ψ at the
//! w-points) with the water no transport carried
//! (`LayerTransport::add_volume_sources`) and the rivers' water
//! (`RiverInflow`).
//!
//! - **Wetting and drying** (Phase 4.1a): the subcell elements (those with a
//!   column thinner than the minimum column depth, and those the barotropic
//!   pass moved on the GLL subcells throughout) move their layers through
//!   the subcell interfaces, each layer with its share of the pass's flux
//!   (or of the column transport's central value, none beside a thin
//!   column, where the pass did not: `fallback`), and their fields cross
//!   those interfaces at the donor's value beside a thin column (the
//!   tracers, the turbulence) or the central one (the momentum); the volume
//!   sources of a fallback or pooled element move water between its nodes
//!   at the mean value of what the losing nodes lost.
//! - **Open faces** (Phase 4.1b): the layers carry the interior's profile
//!   (central with a nesting parent's `H_z u` where it has one), and water
//!   flowing in brings the parent's values or the tracer's boundary
//!   condition's (`DeviceTracerBC3D`), the momentum the interior's.
//! - **Rivers** (Phase 4.1b): their layer volume sources in Ω, the river
//!   water's tracers, and the turbulence the river water brings at the
//!   column's own value.
//! - **The w-points** (Phase 4.1b): the w-cells' transports averaged from
//!   the layers' as a kernel reads them, the end w-points taken as their
//!   neighbours' in the vertical reconstruction.
//!
//! The kernels follow `src/solver/rhs/transport_3d.rs` operation for
//! operation and call dgcom's own column kernel (`column_vertical_flux`),
//! boundary conditions (`DeviceTracerBC3D::exterior_value`) and subcell
//! numbering (`subcell_slot`, `line_node`); the subcell interfaces' metrics
//! are dgcom's `line_metric` and `telescoped_interfaces`, tabled once. So a
//! build without multiply–add contraction (`--no-fmad`) gives the CPU's
//! bits (`transport3d.rs`, the gate).
//!
//! The 3D fields keep the CPU's layout, `[element][node][level]`. The
//! horizontal kernels take one thread per element and level (the element's
//! nodes of that level in registers; consecutive threads read consecutive
//! levels), the column kernels one thread per column.

// As `swe2d.rs`: the device loops index arrays by the loop variable.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use std::sync::Arc;

use cuda_core::{DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, device, kernel, thread};
use cuda_host::cuda_module;
use dgcom::mesh::{Bathymetry2D, Mesh2D};
use dgcom::operators::{DGOperators2D, GeometricFactors2D};
use dgcom::solver::rhs::subcells::{
    line_metric, line_node, subcell_interfaces, subcell_slot, telescoped_interfaces,
};
use dgcom::solver::rhs::{
    Boundaries3D, ColumnScratch, DeviceTracerBC3D, EndLayerShed, ExteriorField, FaceExterior,
    MetricForm, UNBOUNDED, VerticalAdvection, column_vertical_flux, w_cell_thicknesses,
};
use dgcom::source::RiverSources;
use dgcom::types::ElementIndex;
use dgcom::vertical::SigmaGrid;

use crate::swe2d::{Gpu, N1, NN, ORDER, face_node};

/// Most σ-levels a column kernel takes (its scratch is local arrays).
pub const MAX_LEVELS: usize = 64;

/// The other side of a face node (`Statics3D::across`) at a wall ...
pub(crate) const WALL: u32 = u32::MAX;
/// ... and at an open face.
pub(crate) const OPEN: u32 = u32::MAX - 1;
/// No slot (an exterior field's at a face node without its values).
const NONE: u32 = u32::MAX;

/// Face-node slots per element.
const SLOTS: usize = 4 * N1;

/// Subcell interfaces per element (`subcell_interfaces`).
pub const NSUB: usize = 2 * N1 * (N1 - 1);

/// An element's kind (`Transport3D::kinds`): its layers move through the
/// subcell interfaces (`LayerTransport::subcell_elements`) ...
pub(crate) const SUBCELL: u32 = 1;
/// ... without the pass's interface fluxes (`LayerTransport::fallback`) ...
const FALLBACK: u32 = 2;
/// ... and its volume sources move water between its nodes (fallback, or
/// balanced only as a whole by the pass: `BarotropicFlux::pooled`).
pub(crate) const POOLING: u32 = 4;

/// A vertical scheme as the column kernel takes it (a kernel parameter is a
/// number).
fn scheme_code(scheme: VerticalAdvection) -> u32 {
    match scheme {
        VerticalAdvection::Centred => 0,
        VerticalAdvection::Upwind => 1,
        VerticalAdvection::Akima => 2,
        VerticalAdvection::Tvd => 3,
        VerticalAdvection::LimitedAkima => 4,
        VerticalAdvection::HermiteMean => 5,
    }
}

#[inline]
fn scheme_of(code: u32) -> VerticalAdvection {
    match code {
        0 => VerticalAdvection::Centred,
        1 => VerticalAdvection::Upwind,
        2 => VerticalAdvection::Akima,
        3 => VerticalAdvection::Tvd,
        4 => VerticalAdvection::LimitedAkima,
        _ => VerticalAdvection::HermiteMean,
    }
}

/// A tracer boundary condition as a kernel takes it (two numbers).
fn bc_code(bc: DeviceTracerBC3D) -> (u32, f64) {
    match bc {
        DeviceTracerBC3D::Extrapolation => (0, 0.0),
        DeviceTracerBC3D::Fixed(value) => (1, value),
        DeviceTracerBC3D::Upwind(value) => (2, value),
    }
}

#[inline]
fn bc_of(kind: u32, value: f64) -> DeviceTracerBC3D {
    match kind {
        0 => DeviceTracerBC3D::Extrapolation,
        1 => DeviceTracerBC3D::Fixed(value),
        _ => DeviceTracerBC3D::Upwind(value),
    }
}

/// Level `j` of row `row` of a layer field (`nl` per row), or of its w-cells
/// if `w`: half the bed layer's, the mean of two layers', half the top
/// layer's (`WCellBlock::stagger`, `nl + 1` per row).
#[device]
// The device backend makes this alwaysinline (no call in the PTX); the host
// compiler, which sees an exported symbol, ignores it.
#[allow(unused_attributes)]
#[inline(always)]
fn layer_value(values: &[f64], row: usize, nl: usize, j: usize, w: bool) -> f64 {
    let at = row * nl;
    if !w {
        values[at + j]
    } else if j == 0 {
        0.5 * values[at]
    } else if j == nl {
        0.5 * values[at + nl - 1]
    } else {
        0.5 * (values[at + j - 1] + values[at + j])
    }
}

/// The value flowing in through face slot `s` (level `j` of `nf`): an
/// exterior field's where it has one (`ExteriorField::at`), else the
/// boundary condition's (`kind`, `value`) beside `interior` with the flux
/// `flux`.
#[device]
#[allow(unused_attributes)]
#[inline(always)]
fn inflow(
    s: usize,
    j: usize,
    nf: usize,
    ext_slots: &[u32],
    ext: &[f64],
    kind: u32,
    value: f64,
    interior: f64,
    flux: f64,
) -> f64 {
    if !ext_slots.is_empty() {
        let slot = ext_slots[s];
        if slot != NONE {
            return ext[slot as usize * nf + j];
        }
    }
    bc_of(kind, value).exterior_value(interior, flux)
}

/// `subcell_divergence_element` of one level of element `k`: the nodal
/// fluxes `(fx, fy)` (the layer transports, or those times a field), the
/// fluxes through the subcell interfaces (`interface`, along their
/// contravariant directions) and out of the face nodes (`f_star`); the
/// lines' end metrics from `line_ends` (`line_metric`'s first and last).
#[device]
#[allow(unused_attributes)]
#[inline(always)]
fn subcell_divergence(
    k: usize,
    fx: &[f64; NN],
    fy: &[f64; NN],
    interface: &[f64; NSUB],
    f_star: &[f64; SLOTS],
    line_ends: &[[f64; 2]],
    jinv: &[f64],
    w: &[f64],
    fnode: &[[f64; 3]],
    lift: &[f64],
) -> [f64; NN] {
    let mut d = [0.0; NN];
    #[unroll]
    for dir in 0..2 {
        #[unroll]
        for line in 0..N1 {
            let ends = (k * 2 * N1 + dir * N1 + line) * 2;
            let (first, last) = (line_ends[ends], line_ends[ends + 1]);
            #[unroll]
            for a in 0..N1 {
                let node = line_node(N1, dir, line, a);
                let right = if a + 1 < N1 {
                    interface[subcell_slot(N1, dir, line, a)]
                } else {
                    last[0] * fx[node] + last[1] * fy[node]
                };
                let left = if a > 0 {
                    interface[subcell_slot(N1, dir, line, a - 1)]
                } else {
                    first[0] * fx[node] + first[1] * fy[node]
                };
                d[node] += (right - left) / w[a];
            }
        }
    }
    #[unroll]
    for i in 0..NN {
        d[i] *= jinv[k * NN + i];
    }
    #[unroll]
    for f in 0..4 {
        #[unroll]
        for fi in 0..N1 {
            let node = face_node(f, fi);
            let g = fnode[(k * 4 + f) * N1 + fi];
            let jump = g[0] * fx[node] + g[1] * fy[node] - f_star[f * N1 + fi];
            #[unroll]
            for i in 0..NN {
                let lift = g[2] * lift[(f * NN + i) * N1 + fi];
                d[i] -= lift * jump;
            }
        }
    }
    d
}

// =============================================================================
// Kernels
// =============================================================================

#[cuda_module]
pub mod kernels3d {
    use super::*;

    /// The kind of every element (one thread per element; `LayerTransport::compute`'s
    /// start): a subcell element if the pass moved it on the subcells
    /// throughout (its interface fluxes in `bar_subcell` all finite) or it
    /// has a thin column; a fallback element if a subcell element without
    /// them; pooling if a fallback element or `pooled` (empty for none). The
    /// thin columns of every element into `thin`.
    #[kernel]
    pub fn layer_kinds(
        n_elements: u32,
        min_column_depth: f64,
        eta: &[f64],
        bed: &[f64],
        bar_subcell: &[f64],
        pooled: &[u32],
        mut kinds: DisjointSlice<u32>,
        mut thin: DisjointSlice<u32>,
    ) {
        let k = thread::index_1d().get();
        if k >= n_elements as usize {
            return;
        }
        // Without a barotropic flux (`bar_subcell` empty) no pass moved it
        let pass = !bar_subcell.is_empty()
            && bar_subcell[k * NSUB..(k + 1) * NSUB]
                .iter()
                .all(|f| f.is_finite());
        let mut any_thin = false;
        // SAFETY: thread k alone writes element k's kind and nodes.
        unsafe {
            #[unroll]
            for i in 0..NN {
                let flat = k * NN + i;
                let is_thin = eta[flat] - bed[flat] < min_column_depth;
                *thin.get_unchecked_mut(flat) = u32::from(is_thin);
                any_thin |= is_thin;
            }
            let subcell = pass || any_thin;
            let fallback = subcell && !pass;
            let pooling = fallback || (!pooled.is_empty() && pooled[k] != 0);
            *kinds.get_unchecked_mut(k) = u32::from(subcell) * SUBCELL
                + u32::from(fallback) * FALLBACK
                + u32::from(pooling) * POOLING;
        }
    }

    /// `LayerTransport::compute`'s step 1, one thread per column: the layer
    /// transports `H_z u` (`depth·Δσ_l·u`), corrected by each layer's share
    /// `Δσ_l` of the difference between the barotropic transport and their
    /// sum (in level order), and the column's `Δt_E/D`.
    #[kernel]
    pub fn layer_nodal(
        nl: u32,
        n_columns: u32,
        eta: &[f64],
        bed: &[f64],
        u: &[f64],
        v: &[f64],
        d_sigma: &[f64],
        euler_dt: &[f64],
        bar_hu: &[f64],
        bar_hv: &[f64],
        mut hu: DisjointSlice<f64>,
        mut hv: DisjointSlice<f64>,
        mut dt_over_depth: DisjointSlice<f64>,
    ) {
        let idx = thread::index_1d().get();
        if idx >= n_columns as usize {
            return;
        }
        let nl = nl as usize;
        let depth = eta[idx] - bed[idx];
        // Without a barotropic flux (`bar_hu` empty) no correction and no
        // step
        let corrected = !bar_hu.is_empty();
        let dt = if corrected && depth > 0.0 {
            euler_dt[idx / NN] / depth
        } else {
            0.0
        };
        let (mut sum_u, mut sum_v) = (0.0, 0.0);
        // SAFETY: thread idx alone writes column idx (idx < n_columns, the
        // buffers' columns).
        unsafe {
            *dt_over_depth.get_unchecked_mut(idx) = dt;
            for l in 0..nl {
                let ds = d_sigma[l];
                let a = depth * ds * u[idx * nl + l];
                let b = depth * ds * v[idx * nl + l];
                *hu.get_unchecked_mut(idx * nl + l) = a;
                *hv.get_unchecked_mut(idx * nl + l) = b;
                sum_u += a;
                sum_v += b;
            }
            if !corrected {
                return;
            }
            let (corr_u, corr_v) = (bar_hu[idx] - sum_u, bar_hv[idx] - sum_v);
            for l in 0..nl {
                let ds = d_sigma[l];
                *hu.get_unchecked_mut(idx * nl + l) += ds * corr_u;
                *hv.get_unchecked_mut(idx * nl + l) += ds * corr_v;
            }
        }
    }

    /// `LayerTransport::compute`'s step 2, one thread per face node of every
    /// element (`(k·4 + face)·N1 + fi`): the central average of the two
    /// sides' layer transports normal to the face (at an open face the
    /// interior's own, or with a nesting parent's `H_z u` where `parent_u`'s
    /// slot has one; none at a wall, none beside a thin column), corrected
    /// by each layer's share of the difference to the barotropic face flux.
    #[kernel]
    pub fn layer_faces(
        nl: u32,
        n_slots: u32,
        eta: &[f64],
        bed: &[f64],
        thin: &[u32],
        fnode: &[[f64; 3]],
        across: &[u32],
        parent_u_slots: &[u32],
        parent_u: &[f64],
        parent_v_slots: &[u32],
        parent_v: &[f64],
        hu: &[f64],
        hv: &[f64],
        d_sigma: &[f64],
        bar_face: &[f64],
        mut face: DisjointSlice<f64>,
    ) {
        let s = thread::index_1d().get();
        if s >= n_slots as usize {
            return;
        }
        let nl = nl as usize;
        let (k, f, fi) = (s / SLOTS, (s / N1) % 4, s % N1);
        let flat = k * NN + face_node(f, fi);
        let interior = flat * nl;
        let g = fnode[s];
        let (nx, ny) = (g[0], g[1]);
        let nb = across[s];
        let parent = nb == OPEN && !parent_u_slots.is_empty() && parent_u_slots[s] != NONE;
        // The other side's node: the neighbour's, or the interior's own at
        // an open face without a parent
        let other = if nb == OPEN { flat } else { nb as usize };
        let blocked = thin[flat] != 0 || (nb != WALL && !parent && thin[other] != 0);
        let depth = eta[flat] - bed[flat];
        let mut sum = 0.0;
        // SAFETY: thread s alone writes slot s's levels (s < n_slots).
        unsafe {
            for l in 0..nl {
                let q_in = nx * hu[interior + l] + ny * hv[interior + l];
                let flux = if blocked || nb == WALL {
                    0.0
                } else if parent {
                    let u = parent_u[parent_u_slots[s] as usize * nl + l];
                    let v = parent_v[parent_v_slots[s] as usize * nl + l];
                    0.5 * (q_in + depth * d_sigma[l] * (nx * u + ny * v))
                } else {
                    let e = other * nl;
                    0.5 * (q_in + nx * hu[e + l] + ny * hv[e + l])
                };
                *face.get_unchecked_mut(s * nl + l) = flux;
                sum += flux;
            }
            // Without a barotropic flux (`bar_face` empty) no correction
            if bar_face.is_empty() {
                return;
            }
            let corr = bar_face[s] - sum;
            for l in 0..nl {
                *face.get_unchecked_mut(s * nl + l) += d_sigma[l] * corr;
            }
        }
    }

    /// `LayerTransport::compute`'s step 2b, one thread per subcell interface
    /// of every element (`k·NSUB + slot`), the subcell elements' only: each
    /// layer's share `Δσ_l` of the flux through the interface, the pass's
    /// where it moved the element on the subcells (else the column
    /// transports' central value along the interface, none beside a thin
    /// column). The other elements' slots are left as they are (nothing
    /// reads them).
    #[kernel]
    pub fn layer_subcells(
        nl: u32,
        n: u32,
        kinds: &[u32],
        thin: &[u32],
        sub_metric: &[[f64; 2]],
        bar_subcell: &[f64],
        hu: &[f64],
        hv: &[f64],
        d_sigma: &[f64],
        mut sub: DisjointSlice<f64>,
    ) {
        let t = thread::index_1d().get();
        if t >= n as usize {
            return;
        }
        let nl = nl as usize;
        let (k, slot) = (t / NSUB, t % NSUB);
        let kind = kinds[k];
        if kind & SUBCELL == 0 {
            return;
        }
        let a = slot % (N1 - 1);
        let (dir, line) = ((slot / (N1 - 1)) / N1, (slot / (N1 - 1)) % N1);
        let (ia, ib) = (
            k * NN + line_node(N1, dir, line, a),
            k * NN + line_node(N1, dir, line, a + 1),
        );
        let m = sub_metric[t];
        let bar = if kind & FALLBACK == 0 {
            bar_subcell[t]
        } else if thin[ia] != 0 || thin[ib] != 0 {
            0.0
        } else {
            // The column transport along the interface, as the CPU sums it
            let column = |flat: usize| {
                (0..nl)
                    .map(|l| m[0] * hu[flat * nl + l] + m[1] * hv[flat * nl + l])
                    .sum::<f64>()
            };
            0.5 * (column(ia) + column(ib))
        };
        // SAFETY: thread t alone writes interface t's levels (t < n).
        unsafe {
            for l in 0..nl {
                *sub.get_unchecked_mut(t * nl + l) = d_sigma[l] * bar;
            }
        }
    }

    /// The divergence of every layer's transport, one thread per element
    /// and level (`k·nl + l`): `transport_divergence_levels` (the averaged
    /// metric where `averaged[k]`, else the conservative one) in a wet
    /// element, `subcell_divergence_element` in a subcell element, less the
    /// lifted face jumps, into `div` (`[element][node][level]`).
    #[kernel]
    pub fn layer_divergence(
        nl: u32,
        n: u32,
        kinds: &[u32],
        averaged: &[u32],
        contra: &[[f64; 4]],
        jinv: &[f64],
        fnode: &[[f64; 3]],
        dr: &[f64],
        ds: &[f64],
        lift: &[f64],
        line_ends: &[[f64; 2]],
        w: &[f64],
        hu: &[f64],
        hv: &[f64],
        face: &[f64],
        sub: &[f64],
        mut div: DisjointSlice<f64>,
    ) {
        let t = thread::index_1d().get();
        if t >= n as usize {
            return;
        }
        let nl = nl as usize;
        let (k, l) = (t / nl, t % nl);
        let mut q_u = [0.0; NN];
        let mut q_v = [0.0; NN];
        #[unroll]
        for i in 0..NN {
            q_u[i] = hu[(k * NN + i) * nl + l];
            q_v[i] = hv[(k * NN + i) * nl + l];
        }
        let mut d = [0.0; NN];
        if kinds[k] & SUBCELL != 0 {
            let mut interface = [0.0; NSUB];
            #[unroll]
            for s in 0..NSUB {
                interface[s] = sub[(k * NSUB + s) * nl + l];
            }
            let mut f_star = [0.0; SLOTS];
            #[unroll]
            for s in 0..SLOTS {
                f_star[s] = face[(k * SLOTS + s) * nl + l];
            }
            d = subcell_divergence(
                k, &q_u, &q_v, &interface, &f_star, line_ends, jinv, w, fnode, lift,
            );
        } else {
            if averaged[k] != 0 {
                #[unroll]
                for i in 0..NN {
                    let ci = contra[k * NN + i];
                    let (mut a, mut acc) = (0.0, 0.0);
                    #[unroll]
                    for j in 0..NN {
                        let cj = contra[k * NN + j];
                        let (rx, ry) = (ci[0] + cj[0], ci[1] + cj[1]);
                        let (sx, sy) = (ci[2] + cj[2], ci[3] + cj[3]);
                        let (dr, ds) = (dr[i * NN + j], ds[i * NN + j]);
                        let (qx, qy) = (q_u[i] + q_u[j], q_v[i] + q_v[j]);
                        a += dr * (rx * qx + ry * qy);
                        acc += ds * (sx * qx + sy * qy);
                    }
                    let scale = 0.5 * jinv[k * NN + i];
                    d[i] = scale * (a + acc);
                }
            } else {
                let mut fr = [0.0; NN];
                let mut fs = [0.0; NN];
                #[unroll]
                for j in 0..NN {
                    let c = contra[k * NN + j];
                    fr[j] = c[0] * q_u[j] + c[1] * q_v[j];
                    fs[j] = c[2] * q_u[j] + c[3] * q_v[j];
                }
                #[unroll]
                for i in 0..NN {
                    let (mut a, mut acc) = (0.0, 0.0);
                    #[unroll]
                    for j in 0..NN {
                        a += dr[i * NN + j] * fr[j];
                        acc += ds[i * NN + j] * fs[j];
                    }
                    let scale = jinv[k * NN + i];
                    d[i] = scale * (a + acc);
                }
            }
            #[unroll]
            for f in 0..4 {
                #[unroll]
                for fi in 0..N1 {
                    let node = face_node(f, fi);
                    let s = (k * 4 + f) * N1 + fi;
                    let g = fnode[s];
                    let jump = g[0] * q_u[node] + g[1] * q_v[node] - face[s * nl + l];
                    #[unroll]
                    for i in 0..NN {
                        let lift = g[2] * lift[(f * NN + i) * N1 + fi];
                        d[i] -= lift * jump;
                    }
                }
            }
        }
        // SAFETY: thread (k, l) alone writes level l of element k's nodes.
        unsafe {
            #[unroll]
            for i in 0..NN {
                *div.get_unchecked_mut((k * NN + i) * nl + l) = d[i];
            }
        }
    }

    /// `LayerTransport::compute`'s step 3, one thread per column: Ω from the
    /// bed up (the divergence of each layer, its share of `∂η/∂t` and the
    /// rivers' layer sources, `RiverInflow::add_layer_rates`, with no rivers
    /// if `river_offsets` is empty), the surface residual spread over the
    /// column in proportion to `σ_w + 1` and kept as the column's volume
    /// source (negated), and its magnitude in `residual` (the largest is
    /// taken on the host).
    #[kernel]
    pub fn layer_omega(
        nl: u32,
        n_columns: u32,
        div: &[f64],
        eta_rate: &[f64],
        d_sigma: &[f64],
        sigma_w: &[f64],
        river_offsets: &[u32],
        river_by_element: &[u32],
        river_q: &[f64],
        river_share: &[f64],
        river_weights: &[f64],
        mut omega: DisjointSlice<f64>,
        mut volume_source: DisjointSlice<f64>,
        mut residual: DisjointSlice<f64>,
    ) {
        let idx = thread::index_1d().get();
        if idx >= n_columns as usize {
            return;
        }
        let nl = nl as usize;
        let (k, node) = (idx / NN, idx % NN);
        // Without a barotropic flux (`eta_rate` empty) the column's own
        // divergence, summed as the CPU sums it
        let rate = if eta_rate.is_empty() {
            -div[idx * nl..(idx + 1) * nl].iter().sum::<f64>()
        } else {
            eta_rate[idx]
        };
        let base = idx * (nl + 1);
        let rivers = if river_offsets.is_empty() {
            0..0
        } else {
            river_offsets[k] as usize..river_offsets[k + 1] as usize
        };
        // SAFETY: thread idx alone writes column idx (idx < n_columns).
        unsafe {
            let mut w = 0.0;
            *omega.get_unchecked_mut(base) = w;
            for l in 0..nl {
                // The rivers' layer source, summed as `add_layer_rates` sums
                // it (from zero, river by river)
                let mut source = 0.0;
                for r in rivers.clone() {
                    let i = river_by_element[r] as usize;
                    let river_rate = river_q[i] * river_share[i * NN + node];
                    source += river_weights[(i * NN + node) * nl + l] * river_rate;
                }
                w = w - div[idx * nl + l] - d_sigma[l] * rate + source;
                *omega.get_unchecked_mut(base + l + 1) = w;
            }
            let r = w;
            for l in 0..=nl {
                *omega.get_unchecked_mut(base + l) -= (sigma_w[l] + 1.0) * r;
            }
            *volume_source.get_unchecked_mut(idx) = -r;
            *residual.get_unchecked_mut(idx) = r.abs();
        }
    }

    /// The horizontal advection of two fields `a`, `b` by the layer
    /// transports (by their w-cells' if `w_points`: the fields then have
    /// `nl + 1` values per column), one thread per element and level: the
    /// face flux `F φ↑` upwind of the face flux (the interior's at a wall;
    /// flowing in through an open face, the field's exterior value where it
    /// has one, else its boundary condition's, `inflow`); in a wet element
    /// the split-form volume term in the averaged or the conservative metric
    /// form (`advective_divergence_levels`), in a subcell element the
    /// subcell interfaces' fluxes of `φ` (the donor's value beside a thin
    /// column if `donor`, else the central one) and
    /// `subcell_divergence_element` of `q φ` (`LayerContext::flux_divergence`);
    /// the lifted face jumps. With `overwrite` the tendencies are
    /// `−∇·(Q φ)` (the tracers', the turbulence's), else it is subtracted
    /// from them (the momentum's).
    #[kernel]
    pub fn advect_horizontal(
        nl: u32,
        w_points: u32,
        n: u32,
        overwrite: u32,
        donor: u32,
        kinds: &[u32],
        thin: &[u32],
        averaged: &[u32],
        contra: &[[f64; 4]],
        jinv: &[f64],
        fnode: &[[f64; 3]],
        across: &[u32],
        dr: &[f64],
        ds: &[f64],
        lift: &[f64],
        line_ends: &[[f64; 2]],
        w: &[f64],
        hu: &[f64],
        hv: &[f64],
        face: &[f64],
        sub: &[f64],
        phi_a: &[f64],
        phi_b: &[f64],
        ext_slots_a: &[u32],
        ext_a: &[f64],
        bc_kind_a: u32,
        bc_value_a: f64,
        ext_slots_b: &[u32],
        ext_b: &[f64],
        bc_kind_b: u32,
        bc_value_b: f64,
        mut rhs_a: DisjointSlice<f64>,
        mut rhs_b: DisjointSlice<f64>,
    ) {
        let t = thread::index_1d().get();
        if t >= n as usize {
            return;
        }
        let nl = nl as usize;
        let wp = w_points != 0;
        let nf = nl + usize::from(wp);
        let (k, l) = (t / nf, t % nf);
        let mut q_u = [0.0; NN];
        let mut q_v = [0.0; NN];
        let mut pa = [0.0; NN];
        let mut pb = [0.0; NN];
        #[unroll]
        for i in 0..NN {
            q_u[i] = layer_value(hu, k * NN + i, nl, l, wp);
            q_v[i] = layer_value(hv, k * NN + i, nl, l, wp);
            let at = (k * NN + i) * nf + l;
            pa[i] = phi_a[at];
            pb[i] = phi_b[at];
        }
        // The face fluxes of both fields, `F φ↑`
        let mut f_star_a = [0.0; SLOTS];
        let mut f_star_b = [0.0; SLOTS];
        #[unroll]
        for f in 0..4 {
            #[unroll]
            for fi in 0..N1 {
                let node = face_node(f, fi);
                let s = (k * 4 + f) * N1 + fi;
                let flux = layer_value(face, s, nl, l, wp);
                let nb = across[s];
                let (upwind_a, upwind_b) = if flux >= 0.0 || nb == WALL {
                    (pa[node], pb[node])
                } else if nb == OPEN {
                    (
                        inflow(
                            s,
                            l,
                            nf,
                            ext_slots_a,
                            ext_a,
                            bc_kind_a,
                            bc_value_a,
                            pa[node],
                            flux,
                        ),
                        inflow(
                            s,
                            l,
                            nf,
                            ext_slots_b,
                            ext_b,
                            bc_kind_b,
                            bc_value_b,
                            pb[node],
                            flux,
                        ),
                    )
                } else {
                    let e = nb as usize * nf + l;
                    (phi_a[e], phi_b[e])
                };
                f_star_a[f * N1 + fi] = flux * upwind_a;
                f_star_b[f * N1 + fi] = flux * upwind_b;
            }
        }
        let mut da = [0.0; NN];
        let mut db = [0.0; NN];
        if kinds[k] & SUBCELL != 0 {
            // Each field on its own, as `LayerContext::flux_divergence`
            #[unroll]
            for field in 0..2 {
                let (phi, f_star) = if field == 0 {
                    (&pa, &f_star_a)
                } else {
                    (&pb, &f_star_b)
                };
                let mut interface = [0.0; NSUB];
                #[unroll]
                for dir in 0..2 {
                    #[unroll]
                    for line in 0..N1 {
                        #[unroll]
                        for a in 0..N1 - 1 {
                            let slot = subcell_slot(N1, dir, line, a);
                            let flux = layer_value(sub, k * NSUB + slot, nl, l, wp);
                            let (ia, ib) =
                                (line_node(N1, dir, line, a), line_node(N1, dir, line, a + 1));
                            interface[slot] = if donor != 0
                                && (thin[k * NN + ia] != 0 || thin[k * NN + ib] != 0)
                            {
                                flux * if flux >= 0.0 { phi[ia] } else { phi[ib] }
                            } else {
                                flux * 0.5 * (phi[ia] + phi[ib])
                            };
                        }
                    }
                }
                let mut fx = [0.0; NN];
                let mut fy = [0.0; NN];
                #[unroll]
                for i in 0..NN {
                    fx[i] = q_u[i] * phi[i];
                    fy[i] = q_v[i] * phi[i];
                }
                let d = subcell_divergence(
                    k, &fx, &fy, &interface, f_star, line_ends, jinv, w, fnode, lift,
                );
                if field == 0 {
                    da = d;
                } else {
                    db = d;
                }
            }
        } else {
            if averaged[k] != 0 {
                // 2{{Ja}}·{{q}} = ½(Ja_i + Ja_j)·(q_i + q_j)
                #[unroll]
                for i in 0..NN {
                    let ci = contra[k * NN + i];
                    #[unroll]
                    for j in 0..NN {
                        let cj = contra[k * NN + j];
                        let (rx, ry) = (ci[0] + cj[0], ci[1] + cj[1]);
                        let (sx, sy) = (ci[2] + cj[2], ci[3] + cj[3]);
                        let (dr, ds) = (dr[i * NN + j], ds[i * NN + j]);
                        let (qx, qy) = (q_u[i] + q_u[j], q_v[i] + q_v[j]);
                        let flux = 0.5 * (dr * (rx * qx + ry * qy) + ds * (sx * qx + sy * qy));
                        da[i] += flux * (pa[i] + pa[j]);
                        db[i] += flux * (pb[i] + pb[j]);
                    }
                    let scale = 0.5 * jinv[k * NN + i];
                    da[i] *= scale;
                    db[i] *= scale;
                }
            } else {
                let mut fr = [0.0; NN];
                let mut fs = [0.0; NN];
                #[unroll]
                for j in 0..NN {
                    let c = contra[k * NN + j];
                    fr[j] = c[0] * q_u[j] + c[1] * q_v[j];
                    fs[j] = c[2] * q_u[j] + c[3] * q_v[j];
                }
                #[unroll]
                for i in 0..NN {
                    #[unroll]
                    for j in 0..NN {
                        let flux =
                            dr[i * NN + j] * (fr[i] + fr[j]) + ds[i * NN + j] * (fs[i] + fs[j]);
                        da[i] += flux * (pa[i] + pa[j]);
                        db[i] += flux * (pb[i] + pb[j]);
                    }
                    let scale = 0.5 * jinv[k * NN + i];
                    da[i] *= scale;
                    db[i] *= scale;
                }
            }
            // The face terms, field a's then field b's (each a node's in the
            // CPU's order of faces and face nodes)
            #[unroll]
            for field in 0..2 {
                let (phi, f_star) = if field == 0 {
                    (&pa, &f_star_a)
                } else {
                    (&pb, &f_star_b)
                };
                #[unroll]
                for f in 0..4 {
                    #[unroll]
                    for fi in 0..N1 {
                        let node = face_node(f, fi);
                        let s = (k * 4 + f) * N1 + fi;
                        let g = fnode[s];
                        let normal = g[0] * q_u[node] + g[1] * q_v[node];
                        let jump = normal * phi[node] - f_star[f * N1 + fi];
                        #[unroll]
                        for i in 0..NN {
                            let lift = g[2] * lift[(f * NN + i) * N1 + fi];
                            if field == 0 {
                                da[i] -= lift * jump;
                            } else {
                                db[i] -= lift * jump;
                            }
                        }
                    }
                }
            }
        }
        // SAFETY: thread (k, l) alone writes level l of element k's nodes.
        unsafe {
            #[unroll]
            for i in 0..NN {
                let at = (k * NN + i) * nf + l;
                if overwrite != 0 {
                    *rhs_a.get_unchecked_mut(at) = -da[i];
                    *rhs_b.get_unchecked_mut(at) = -db[i];
                } else {
                    *rhs_a.get_unchecked_mut(at) -= da[i];
                    *rhs_b.get_unchecked_mut(at) -= db[i];
                }
            }
        }
    }

    /// The vertical advection of one field of the layers, one thread per
    /// column: dgcom's `column_vertical_flux` (the scheme `scheme`, the end
    /// layers' water kept within `(range_min, range_max)` at the column's
    /// `Δt_E/D`, about `reference` if it is not empty), subtracted from
    /// `rhs`; then, if `volume_source` is not empty and the column's element
    /// is not pooling, the water no transport carried at the field's own
    /// value, `Δσ_l s C` (`LayerTransport::add_volume_sources`), added, as
    /// the CPU adds it after the advection (`pooled_sources` adds the
    /// pooling elements').
    #[kernel]
    pub fn advect_vertical(
        nl: u32,
        n_columns: u32,
        scheme: u32,
        range_min: f64,
        range_max: f64,
        kinds: &[u32],
        field: &[f64],
        reference: &[f64],
        omega: &[f64],
        d_sigma: &[f64],
        dt_over_depth: &[f64],
        volume_source: &[f64],
        mut rhs: DisjointSlice<f64>,
    ) {
        let idx = thread::index_1d().get();
        if idx >= n_columns as usize {
            return;
        }
        let nl = nl as usize;
        let column = &field[idx * nl..(idx + 1) * nl];
        let reference = if reference.is_empty() {
            None
        } else {
            Some(&reference[idx * nl..(idx + 1) * nl])
        };
        let mut slope = [0.0; MAX_LEVELS];
        let mut surface = [0.0; MAX_LEVELS + 1];
        let mut reference_surface = [0.0; MAX_LEVELS + 1];
        // SAFETY: thread idx alone writes column idx's tendencies
        // (idx < n_columns, nl levels each).
        let rhs = unsafe { core::slice::from_raw_parts_mut(rhs.as_mut_ptr().add(idx * nl), nl) };
        column_vertical_flux(
            rhs,
            column,
            reference,
            &omega[idx * (nl + 1)..(idx + 1) * (nl + 1)],
            &d_sigma[..nl],
            EndLayerShed::new((range_min, range_max), dt_over_depth[idx]),
            scheme_of(scheme),
            ColumnScratch {
                slope: &mut slope[..nl],
                surface: &mut surface[..nl + 1],
                reference_surface: &mut reference_surface[..nl + 1],
            },
        );
        if !volume_source.is_empty() && kinds[idx / NN] & POOLING == 0 {
            let s = volume_source[idx];
            for c in 0..nl {
                rhs[c] += d_sigma[c] * s * column[c];
            }
        }
    }

    /// The vertical advection of one field at the w-points (`nl + 1` per
    /// column, bed first), one thread per column, as
    /// `apply_w_point_transport_3d` takes it: the w-cells' Ω (the mean of
    /// the layer's two, none through the bed and the surface), their
    /// σ-thicknesses `d_sigma_w`, the end values taken as their neighbours'
    /// (three w-points or more), no range; then the rivers' water at the
    /// column's own value (`RiverInflow::add_w_point_sources_at_own_value`,
    /// none if `river_offsets` is empty) and, if `volume_source` is not
    /// empty and the column's element is not pooling, the water no
    /// transport carried (the pooling elements' in `pooled_sources`).
    #[kernel]
    pub fn advect_vertical_w(
        nl: u32,
        n_columns: u32,
        scheme: u32,
        kinds: &[u32],
        field: &[f64],
        omega: &[f64],
        d_sigma_w: &[f64],
        river_offsets: &[u32],
        river_by_element: &[u32],
        river_q: &[f64],
        river_share: &[f64],
        river_weights: &[f64],
        volume_source: &[f64],
        mut rhs: DisjointSlice<f64>,
    ) {
        let idx = thread::index_1d().get();
        if idx >= n_columns as usize {
            return;
        }
        let nl = nl as usize;
        let nw = nl + 1;
        let (k, node) = (idx / NN, idx % NN);
        let field = &field[idx * nw..(idx + 1) * nw];
        let mut own = [0.0; MAX_LEVELS + 1];
        own[..nw].copy_from_slice(field);
        if nw >= 3 {
            own[0] = own[1];
            own[nw - 1] = own[nw - 2];
        }
        // The w-cells' Ω: `WCellBlock::stagger`
        let layers = &omega[idx * (nl + 1)..(idx + 1) * (nl + 1)];
        let mut omega_w = [0.0; MAX_LEVELS + 2];
        for l in 0..nl {
            omega_w[l + 1] = 0.5 * (layers[l] + layers[l + 1]);
        }
        let mut slope = [0.0; MAX_LEVELS + 1];
        let mut surface = [0.0; MAX_LEVELS + 2];
        let mut reference_surface = [0.0; MAX_LEVELS + 2];
        // SAFETY: thread idx alone writes column idx's tendencies
        // (idx < n_columns, nw w-points each).
        let rhs = unsafe { core::slice::from_raw_parts_mut(rhs.as_mut_ptr().add(idx * nw), nw) };
        column_vertical_flux(
            rhs,
            &own[..nw],
            None,
            &omega_w[..nw + 1],
            &d_sigma_w[..nw],
            EndLayerShed::new(UNBOUNDED, 0.0),
            scheme_of(scheme),
            ColumnScratch {
                slope: &mut slope[..nw],
                surface: &mut surface[..nw + 1],
                reference_surface: &mut reference_surface[..nw + 1],
            },
        );
        if !river_offsets.is_empty() {
            for r in river_offsets[k] as usize..river_offsets[k + 1] as usize {
                let i = river_by_element[r] as usize;
                let weights = &river_weights[(i * NN + node) * nl..(i * NN + node + 1) * nl];
                let rate = river_q[i] * river_share[i * NN + node];
                for j in 0..nw {
                    let below = if j > 0 { weights[j - 1] } else { 0.0 };
                    let above = weights.get(j).copied().unwrap_or(0.0);
                    rhs[j] += 0.5 * (below + above) * rate * field[j];
                }
            }
        }
        if !volume_source.is_empty() && kinds[k] & POOLING == 0 {
            let s = volume_source[idx];
            for c in 0..nw {
                rhs[c] += d_sigma_w[c] * s * field[c];
            }
        }
    }

    /// `LayerTransport::add_volume_sources` in a pooling element, one thread
    /// per element and cell (`nc` cells per column of σ-thickness `cells`:
    /// the layers, or the w-cells), the others returning at once: the water
    /// its nodes lose leaves at their own value, and the water they gain
    /// comes at the mean value of what the others lost (what they gain
    /// beyond it, at their own), `Δσ_c s C` added to `rhs`.
    #[kernel]
    pub fn pooled_sources(
        nc: u32,
        n: u32,
        kinds: &[u32],
        mass: &[f64],
        volume_source: &[f64],
        cells: &[f64],
        field: &[f64],
        mut rhs: DisjointSlice<f64>,
    ) {
        let t = thread::index_1d().get();
        if t >= n as usize {
            return;
        }
        let nc = nc as usize;
        let (k, c) = (t / nc, t % nc);
        if kinds[k] & POOLING == 0 {
            return;
        }
        let mut s = [0.0; NN];
        let (mut lost, mut gained) = (0.0f64, 0.0f64);
        #[unroll]
        for i in 0..NN {
            s[i] = volume_source[k * NN + i];
            let rate = mass[k * NN + i] * s[i];
            lost += (-rate).max(0.0);
            gained += rate.max(0.0);
        }
        let dc = cells[c];
        // SAFETY: thread (k, c) alone writes cell c of element k's nodes.
        unsafe {
            if lost <= 0.0 || gained <= 0.0 {
                #[unroll]
                for i in 0..NN {
                    let at = (k * NN + i) * nc + c;
                    *rhs.get_unchecked_mut(at) += dc * s[i] * field[at];
                }
                return;
            }
            // The share of a gain that is the others' loss
            let share = (lost / gained).min(1.0);
            let mut pool = 0.0;
            #[unroll]
            for i in 0..NN {
                if s[i] < 0.0 {
                    pool -= mass[k * NN + i] * s[i] * field[(k * NN + i) * nc + c];
                }
            }
            let pool = pool / lost;
            #[unroll]
            for i in 0..NN {
                let at = (k * NN + i) * nc + c;
                let own = field[at];
                let value = if s[i] > 0.0 {
                    share * pool + (1.0 - share) * own
                } else {
                    own
                };
                *rhs.get_unchecked_mut(at) += dc * s[i] * value;
            }
        }
    }

    /// The river water's tracer (`RiverInflow::add_tracer_sources`), one
    /// thread per node and level of each element with rivers
    /// (`river_elements`): each river's layer source times its water's
    /// concentration `conc`, added to `rhs` river by river.
    #[kernel]
    pub fn river_sources(
        nl: u32,
        n: u32,
        river_elements: &[u32],
        river_offsets: &[u32],
        river_by_element: &[u32],
        river_q: &[f64],
        river_share: &[f64],
        river_weights: &[f64],
        conc: &[f64],
        mut rhs: DisjointSlice<f64>,
    ) {
        let t = thread::index_1d().get();
        if t >= n as usize {
            return;
        }
        let nl = nl as usize;
        let (e, rest) = (t / (NN * nl), t % (NN * nl));
        let (node, l) = (rest / nl, rest % nl);
        let k = river_elements[e] as usize;
        // SAFETY: thread t alone writes level l of node `node` of element k
        // (the elements are distinct).
        unsafe {
            let r = rhs.get_unchecked_mut((k * NN + node) * nl + l);
            for slot in river_offsets[k] as usize..river_offsets[k + 1] as usize {
                let i = river_by_element[slot] as usize;
                let rate = river_q[i] * river_share[i * NN + node];
                *r += river_weights[(i * NN + node) * nl + l] * rate * conc[i];
            }
        }
    }
}

// =============================================================================
// Host side
// =============================================================================

/// A launch over `n` threads in blocks of `block`.
pub(crate) fn launch(n: usize, block: u32) -> LaunchConfig {
    let n = n as u32;
    LaunchConfig {
        grid_dim: (n.div_ceil(block).max(1), 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Threads per block of the column kernels.
pub(crate) const COLUMN_BLOCK: u32 = 128;
/// Threads per block of the element-and-level kernels.
pub(crate) const LEVEL_BLOCK: u32 = 128;

/// What the 3D transport kernels read of a mesh and its σ-levels, packed as
/// they read it, on the device; every value from the CPU's own functions.
pub struct Statics3D {
    pub n_elements: usize,
    pub n_levels: usize,
    pub(crate) bed: DeviceBuffer<f64>,
    /// Whether element `k`'s volume terms take the averaged metric
    /// (`MetricForm::Averaged` on an element that is not a parallelogram)
    pub(crate) averaged: DeviceBuffer<u32>,
    /// `J∇r`, `J∇s` of every node (`GeometricFactors2D::contravariant`)
    pub(crate) contra: DeviceBuffer<[f64; 4]>,
    pub(crate) jinv: DeviceBuffer<f64>,
    /// Nodal masses (GLL weight × J, `GeometricFactors2D::node_mass`)
    pub(crate) mass: DeviceBuffer<f64>,
    /// Per face node (`(k·4 + face)·N1 + fi`): the outward normal and the
    /// lift's scale (`GeometricFactors2D::lift_scale`)
    pub(crate) fnode: DeviceBuffer<[f64; 3]>,
    /// Per face node: the flat node on the other side, [`WALL`] at a wall,
    /// [`OPEN`] at an open face
    pub(crate) across: DeviceBuffer<u32>,
    pub(crate) dr: DeviceBuffer<f64>,
    pub(crate) ds: DeviceBuffer<f64>,
    pub(crate) lift: DeviceBuffer<f64>,
    /// The GLL weights on [−1, 1]
    pub(crate) w: DeviceBuffer<f64>,
    /// Per subcell interface (`k·NSUB + slot`): its contravariant direction
    /// (`telescoped_interfaces` of the line's `line_metric`)
    pub(crate) sub_metric: DeviceBuffer<[f64; 2]>,
    /// Per GLL line (`(k·2 + dir)·N1 + line`): `line_metric` at its first
    /// and last node
    pub(crate) line_ends: DeviceBuffer<[f64; 2]>,
    pub(crate) d_sigma: DeviceBuffer<f64>,
    pub(crate) sigma_w: DeviceBuffer<f64>,
    /// The layer centres' σ
    pub(crate) sigma_rho: DeviceBuffer<f64>,
    /// The w-cells' σ-thicknesses (`w_cell_thicknesses`)
    pub(crate) d_sigma_w: DeviceBuffer<f64>,
    /// Each face node's open boundary tag (`None` but at open faces), for
    /// [`ExteriorDevice::new`]
    open_tags: Vec<Option<dgcom::mesh::BoundaryTag>>,
}

impl Statics3D {
    /// The static data of `mesh` with its bed under `sigma`'s levels, the
    /// volume terms in `metric`'s form, the faces classified by
    /// `boundaries`; refuses what the kernels do not take (another order
    /// than P2, more than [`MAX_LEVELS`] levels).
    pub fn upload(
        gpu: &Gpu,
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        geom: &GeometricFactors2D,
        bathymetry: &Bathymetry2D,
        sigma: &SigmaGrid,
        boundaries: &Boundaries3D,
        metric: MetricForm,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let stream = &gpu.stream;
        let n_el = mesh.n_elements;
        let nl = sigma.d_sigma().len();
        if ops.n_1d != N1 {
            return Err(format!("the 3D kernels are compiled for P{ORDER}").into());
        }
        if nl > MAX_LEVELS {
            return Err(format!("{nl} σ-levels: the 3D kernels take {MAX_LEVELS}").into());
        }
        assert_eq!(
            subcell_interfaces(N1),
            NSUB,
            "subcell interfaces per element"
        );
        for (face, nodes) in ops.face_nodes.iter().enumerate() {
            for (fi, &node) in nodes.iter().enumerate() {
                assert_eq!(face_node(face, fi), node, "face node {fi} of face {face}");
                // The momentum kernels lift a face node's jump into that
                // node alone
                for i in 0..NN {
                    let entry = ops.lift_row_major[face][i * N1 + fi];
                    assert_eq!(
                        entry == 0.0,
                        i != node,
                        "LIFT of face {face} at ({i}, {fi})"
                    );
                }
            }
        }
        let mut averaged = Vec::with_capacity(n_el);
        let mut contra = Vec::with_capacity(n_el * NN);
        let mut jinv = Vec::with_capacity(n_el * NN);
        let mut mass = Vec::with_capacity(n_el * NN);
        let mut fnode = Vec::with_capacity(n_el * SLOTS);
        let mut across = Vec::with_capacity(n_el * SLOTS);
        let mut open_tags = Vec::with_capacity(n_el * SLOTS);
        let mut sub_metric = vec![[0.0; 2]; n_el * NSUB];
        let mut line_ends = vec![[0.0; 2]; n_el * 2 * N1 * 2];
        let mut metric_line = [(0.0, 0.0); N1];
        let mut interfaces = [(0.0, 0.0); N1 + 1];
        for k in 0..n_el {
            averaged.push(u32::from(
                metric == MetricForm::Averaged && !geom.element_is_affine(k),
            ));
            for i in 0..NN {
                let (r, s) = geom.contravariant(k, i);
                contra.push([r.0, r.1, s.0, s.1]);
                jinv.push(geom.jacobian_inv(k, i));
                mass.push(geom.node_mass(k, i));
            }
            for f in 0..4 {
                let exterior = boundaries.exterior(mesh, ElementIndex::new(k), f);
                for fi in 0..N1 {
                    let node = ops.face_nodes[f][fi];
                    let (nx, ny) = geom.normal(k, f, fi);
                    fnode.push([nx, ny, geom.lift_scale(k, f, fi, node)]);
                    let (other, tag) = match exterior {
                        FaceExterior::Element(nb) => (
                            (nb.element * NN + ops.face_nodes[nb.face][N1 - 1 - fi]) as u32,
                            None,
                        ),
                        FaceExterior::Wall => (WALL, None),
                        FaceExterior::Open(tag) => (OPEN, Some(tag)),
                    };
                    across.push(other);
                    open_tags.push(tag);
                }
            }
            // The subcell lines, as `LayerTransport::compute` and
            // `subcell_divergence_element` take them
            for dir in 0..2 {
                for line in 0..N1 {
                    let affine = line_metric(ops, geom, k, dir, line, &mut metric_line);
                    telescoped_interfaces(
                        &ops.weights_1d,
                        &ops.dr_1d_row_major,
                        &metric_line,
                        affine,
                        &mut interfaces,
                    );
                    for a in 0..N1 - 1 {
                        let m = interfaces[a + 1];
                        sub_metric[k * NSUB + subcell_slot(N1, dir, line, a)] = [m.0, m.1];
                    }
                    let ends = (k * 2 * N1 + dir * N1 + line) * 2;
                    let (first, last) = (metric_line[0], metric_line[N1 - 1]);
                    line_ends[ends] = [first.0, first.1];
                    line_ends[ends + 1] = [last.0, last.1];
                }
            }
        }
        let lift: Vec<f64> = ops.lift_row_major.iter().flatten().copied().collect();
        let mut d_sigma_w = vec![0.0; nl + 1];
        w_cell_thicknesses(sigma.d_sigma(), &mut d_sigma_w);
        Ok(Self {
            n_elements: n_el,
            n_levels: nl,
            bed: DeviceBuffer::from_host(stream, &bathymetry.data)?,
            averaged: DeviceBuffer::from_host(stream, &averaged)?,
            contra: DeviceBuffer::from_host(stream, &contra)?,
            jinv: DeviceBuffer::from_host(stream, &jinv)?,
            mass: DeviceBuffer::from_host(stream, &mass)?,
            fnode: DeviceBuffer::from_host(stream, &fnode)?,
            across: DeviceBuffer::from_host(stream, &across)?,
            dr: DeviceBuffer::from_host(stream, &ops.dr_row_major)?,
            ds: DeviceBuffer::from_host(stream, &ops.ds_row_major)?,
            lift: DeviceBuffer::from_host(stream, &lift)?,
            w: DeviceBuffer::from_host(stream, &ops.weights_1d)?,
            sub_metric: DeviceBuffer::from_host(stream, &sub_metric)?,
            line_ends: DeviceBuffer::from_host(stream, &line_ends)?,
            d_sigma: DeviceBuffer::from_host(stream, sigma.d_sigma())?,
            sigma_w: DeviceBuffer::from_host(stream, sigma.sigma_w())?,
            sigma_rho: DeviceBuffer::from_host(stream, sigma.sigma_rho())?,
            d_sigma_w: DeviceBuffer::from_host(stream, &d_sigma_w)?,
            open_tags,
        })
    }

    pub(crate) fn n_columns(&self) -> usize {
        self.n_elements * NN
    }
}

/// A field's values outside the open faces (dgcom's `ExteriorField`: a
/// nesting parent's profiles), on the device: each face node's slot (static)
/// and the values of every slot (`[slot][level]`, uploaded when they
/// change).
pub struct ExteriorDevice {
    slots: DeviceBuffer<u32>,
    pub values: DeviceBuffer<f64>,
}

impl ExteriorDevice {
    /// `field` at the face nodes of `statics`' open faces: a face node's
    /// slot where `field.at` has a value there.
    pub fn new(
        gpu: &Gpu,
        statics: &Statics3D,
        field: &ExteriorField,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        assert_eq!(
            field.n_levels, statics.n_levels,
            "an exterior field of the layers"
        );
        let slots: Vec<u32> = statics
            .open_tags
            .iter()
            .enumerate()
            .map(|(s, tag)| {
                let (k, f, fi) = (s / SLOTS, (s / N1) % 4, s % N1);
                let flat = k * NN + face_node(f, fi);
                match tag {
                    Some(tag) if field.at(*tag, flat, 0).is_some() => field.slot_of_node[flat],
                    _ => NONE,
                }
            })
            .collect();
        Ok(Self {
            slots: DeviceBuffer::from_host(&gpu.stream, &slots)?,
            values: DeviceBuffer::from_host(&gpu.stream, field.values)?,
        })
    }
}

/// Where dgcom's `RiverSources` put the rivers, on the device: the rivers
/// of every element in index order (`in_element`, `offsets[k]..offsets[k +
/// 1]` of `by_element`), the elements with rivers, each once, and each
/// river's share of its discharge at its element's nodes (`share`, `i·NN +
/// node`).
pub struct RiverTables {
    pub(crate) n_rivers: usize,
    pub(crate) offsets: DeviceBuffer<u32>,
    pub(crate) by_element: DeviceBuffer<u32>,
    pub(crate) elements: DeviceBuffer<u32>,
    pub(crate) n_elements_with: usize,
    pub(crate) share: DeviceBuffer<f64>,
}

impl RiverTables {
    /// `rivers`' tables over `n_elements` elements.
    pub fn new(
        gpu: &Gpu,
        n_elements: usize,
        rivers: &RiverSources,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let stream = &gpu.stream;
        let mut offsets = vec![0u32; n_elements + 1];
        let mut by_element = Vec::with_capacity(rivers.len());
        let mut elements = Vec::new();
        for k in 0..n_elements {
            let here = rivers.in_element(k);
            by_element.extend(here.iter().map(|&i| i as u32));
            offsets[k + 1] = by_element.len() as u32;
            if !here.is_empty() {
                elements.push(k as u32);
            }
        }
        let share: Vec<f64> = (0..rivers.len())
            .flat_map(|i| rivers.share(i).to_vec())
            .collect();
        Ok(Self {
            n_rivers: rivers.len(),
            offsets: DeviceBuffer::from_host(stream, &offsets)?,
            by_element: DeviceBuffer::from_host(stream, &by_element)?,
            n_elements_with: elements.len(),
            elements: DeviceBuffer::from_host(stream, &elements)?,
            share: DeviceBuffer::from_host(stream, &share)?,
        })
    }
}

/// The rivers on the device (dgcom's `RiverSources` with its levels set):
/// their [`RiverTables`], each river's level weights (static), and its
/// step-mean discharge (`RiverInflow`, [`Self::set_discharge`]).
pub struct RiversDevice {
    tables: RiverTables,
    weights: DeviceBuffer<f64>,
    q: DeviceBuffer<f64>,
}

impl RiversDevice {
    pub fn new(
        gpu: &Gpu,
        statics: &Statics3D,
        rivers: &RiverSources,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let stream = &gpu.stream;
        assert_eq!(
            rivers.n_levels(),
            statics.n_levels,
            "the rivers' levels are the grid's"
        );
        let weights: Vec<f64> = (0..rivers.len())
            .flat_map(|i| (0..NN).flat_map(move |node| rivers.level_weights(i, node).to_vec()))
            .collect();
        Ok(Self {
            tables: RiverTables::new(gpu, statics.n_elements, rivers)?,
            weights: DeviceBuffer::from_host(stream, &weights)?,
            q: DeviceBuffer::zeroed(stream, rivers.len())?,
        })
    }

    /// Every river's step-mean discharge (m³/s; `RiverInflow::discharge`).
    pub fn set_discharge(
        &mut self,
        gpu: &Gpu,
        q: &[f64],
    ) -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(q.len(), self.tables.n_rivers, "a discharge per river");
        self.q = DeviceBuffer::from_host(&gpu.stream, q)?;
        Ok(())
    }
}

/// The barotropic side of a baroclinic step as the layer transports take it
/// (dgcom's `BarotropicFlux`; its rivers are [`Transport3D`]'s), on the
/// device.
pub struct BarotropicDevice {
    /// `DU_avg2`, `[element][node]`
    pub hu: DeviceBuffer<f64>,
    pub hv: DeviceBuffer<f64>,
    /// The mass flux out of every face node, `(k·4 + face)·N1 + fi`
    pub face: DeviceBuffer<f64>,
    /// The mass flux through every subcell interface, `k·NSUB + slot`; NaN
    /// in the elements the pass did not move on the subcells throughout
    pub subcell: DeviceBuffer<f64>,
    /// `∂η/∂t` over the step, `[element][node]`
    pub eta_rate: DeviceBuffer<f64>,
    /// The longest forward-Euler step of every element
    pub euler_dt: DeviceBuffer<f64>,
    /// The elements the pass balanced only as a whole (non-zero; empty for
    /// none)
    pub pooled: DeviceBuffer<u32>,
}

/// A field's advection as [`Transport3D::advect`] takes it: the field, its
/// reference stratification (if any), its range, its values outside the
/// open faces (if any) and its boundary condition there.
pub struct Advected<'a> {
    pub field: &'a DeviceBuffer<f64>,
    pub reference: Option<&'a DeviceBuffer<f64>>,
    pub range: (f64, f64),
    pub exterior: Option<&'a ExteriorDevice>,
    pub bc: DeviceTracerBC3D,
}

/// How the fields cross the subcell interfaces of a subcell element
/// (dgcom's `InterfaceValue`).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Interface {
    /// The central value (the momentum)
    Central,
    /// The donor's value beside a thin column (the tracers, the turbulence)
    DonorBesideThin,
}

/// The layer transports of a 3D stage on the device (dgcom's
/// `LayerTransport`), and the advection by them.
pub struct Transport3D {
    gpu: Arc<Gpu>,
    statics: Statics3D,
    /// Columns thinner than this exchange no baroclinic transport
    min_column_depth: f64,
    /// The rivers, if any
    pub rivers: Option<RiversDevice>,
    /// Every element's kind ([`SUBCELL`], [`FALLBACK`], [`POOLING`])
    pub kinds: DeviceBuffer<u32>,
    /// The thin columns (every element's; the CPU sets the subcell
    /// elements' only)
    pub thin: DeviceBuffer<u32>,
    /// `Q_l` at the nodes, `[element][node][level]`
    pub hu: DeviceBuffer<f64>,
    pub hv: DeviceBuffer<f64>,
    /// Layer fluxes out of every face node, `[face node][level]`
    pub face: DeviceBuffer<f64>,
    /// Layer fluxes through the subcell elements' interfaces,
    /// `[element][slot][level]`
    pub subcell: DeviceBuffer<f64>,
    /// Ω at the w-points, `[element][node][level + 1]`
    pub omega: DeviceBuffer<f64>,
    pub volume_source: DeviceBuffer<f64>,
    pub dt_over_depth: DeviceBuffer<f64>,
    /// Each column's surface residual's magnitude
    residual: DeviceBuffer<f64>,
    /// The layers' divergence, `[element][node][level]`
    div: DeviceBuffer<f64>,
    /// An empty buffer for an absent field
    none: DeviceBuffer<f64>,
    none_u32: DeviceBuffer<u32>,
}

impl Transport3D {
    pub fn new(
        gpu: &Arc<Gpu>,
        statics: Statics3D,
        min_column_depth: f64,
        rivers: Option<RiversDevice>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let stream = &gpu.stream;
        let (ne, nl) = (statics.n_elements, statics.n_levels);
        let zeros = |n: usize| DeviceBuffer::<f64>::zeroed(stream, n);
        Ok(Self {
            gpu: gpu.clone(),
            min_column_depth,
            rivers,
            kinds: DeviceBuffer::zeroed(stream, ne)?,
            thin: DeviceBuffer::zeroed(stream, ne * NN)?,
            hu: zeros(ne * NN * nl)?,
            hv: zeros(ne * NN * nl)?,
            face: zeros(ne * SLOTS * nl)?,
            subcell: zeros(ne * NSUB * nl)?,
            omega: zeros(ne * NN * (nl + 1))?,
            volume_source: zeros(ne * NN)?,
            dt_over_depth: zeros(ne * NN)?,
            residual: zeros(ne * NN)?,
            div: zeros(ne * NN * nl)?,
            none: zeros(0)?,
            none_u32: DeviceBuffer::zeroed(stream, 0)?,
            statics,
        })
    }

    /// The mesh's and the levels' tables.
    pub fn statics(&self) -> &Statics3D {
        &self.statics
    }

    /// The rivers' tables, or empty buffers for none.
    fn river_tables(&self) -> [&DeviceBuffer<u32>; 2] {
        match &self.rivers {
            Some(r) => [&r.tables.offsets, &r.tables.by_element],
            None => [&self.none_u32, &self.none_u32],
        }
    }

    fn river_values(&self) -> [&DeviceBuffer<f64>; 3] {
        match &self.rivers {
            Some(r) => [&r.q, &r.tables.share, &r.weights],
            None => [&self.none, &self.none, &self.none],
        }
    }

    /// The layer transports and Ω of the state `(eta, u, v)` corrected to
    /// `barotropic` (`LayerTransport::compute` with `Some(barotropic)`, its
    /// rivers this transport's), with a nesting parent's velocity `parent`
    /// (`Exterior3D::velocity`) outside the open faces it covers; the thin
    /// columns' velocities must be at rest, as the mode splitter's stage
    /// values are (`Hydrostatic3D`'s mask).
    pub fn compute(
        &mut self,
        eta: &DeviceBuffer<f64>,
        u: &DeviceBuffer<f64>,
        v: &DeviceBuffer<f64>,
        barotropic: Option<&BarotropicDevice>,
        parent: Option<[&ExteriorDevice; 2]>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let gpu = self.gpu.clone();
        let (m, st) = (&gpu.module3d, &gpu.stream);
        let s = &self.statics;
        let nl = s.n_levels as u32;
        let n_columns = s.n_columns();
        let n_slots = s.n_elements * SLOTS;
        let n_sub = s.n_elements * NSUB;
        let n_levels = s.n_elements * s.n_levels;
        // Without a barotropic flux (`LayerTransport::compute` with `None`:
        // the layers' own transports, the slow forcing's) every pass array
        // is empty, and so are the rivers (the pass's)
        let none = &self.none;
        let (bar_subcell, bar_hu, bar_hv, bar_face, eta_rate, euler_dt) = match barotropic {
            Some(b) => (&b.subcell, &b.hu, &b.hv, &b.face, &b.eta_rate, &b.euler_dt),
            None => (none, none, none, none, none, none),
        };
        let pooled = barotropic.map_or(&self.none_u32, |b| &b.pooled);
        // Field by field, so that the outputs below can be borrowed apart
        let rivers = self.rivers.as_ref().filter(|_| barotropic.is_some());
        let (offsets, by_element, q, share, weights) = match rivers {
            Some(r) => (
                &r.tables.offsets,
                &r.tables.by_element,
                &r.q,
                &r.tables.share,
                &r.weights,
            ),
            None => (
                &self.none_u32,
                &self.none_u32,
                &self.none,
                &self.none,
                &self.none,
            ),
        };
        let (pu_slots, pu, pv_slots, pv) = match parent {
            Some([u, v]) => (&u.slots, &u.values, &v.slots, &v.values),
            None => (&self.none_u32, &self.none, &self.none_u32, &self.none),
        };
        // SAFETY: every buffer is sized for this mesh and these levels
        // (`Statics3D::upload`, `Self::new`, the caller's state and pass);
        // no output is an input.
        unsafe {
            m.layer_kinds(
                st,
                launch(s.n_elements, COLUMN_BLOCK),
                s.n_elements as u32,
                self.min_column_depth,
                eta,
                &s.bed,
                bar_subcell,
                pooled,
                &mut self.kinds,
                &mut self.thin,
            )?;
            m.layer_nodal(
                st,
                launch(n_columns, COLUMN_BLOCK),
                nl,
                n_columns as u32,
                eta,
                &s.bed,
                u,
                v,
                &s.d_sigma,
                euler_dt,
                bar_hu,
                bar_hv,
                &mut self.hu,
                &mut self.hv,
                &mut self.dt_over_depth,
            )?;
            m.layer_faces(
                st,
                launch(n_slots, COLUMN_BLOCK),
                nl,
                n_slots as u32,
                eta,
                &s.bed,
                &self.thin,
                &s.fnode,
                &s.across,
                pu_slots,
                pu,
                pv_slots,
                pv,
                &self.hu,
                &self.hv,
                &s.d_sigma,
                bar_face,
                &mut self.face,
            )?;
            m.layer_subcells(
                st,
                launch(n_sub, COLUMN_BLOCK),
                nl,
                n_sub as u32,
                &self.kinds,
                &self.thin,
                &s.sub_metric,
                bar_subcell,
                &self.hu,
                &self.hv,
                &s.d_sigma,
                &mut self.subcell,
            )?;
            m.layer_divergence(
                st,
                launch(n_levels, LEVEL_BLOCK),
                nl,
                n_levels as u32,
                &self.kinds,
                &s.averaged,
                &s.contra,
                &s.jinv,
                &s.fnode,
                &s.dr,
                &s.ds,
                &s.lift,
                &s.line_ends,
                &s.w,
                &self.hu,
                &self.hv,
                &self.face,
                &self.subcell,
                &mut self.div,
            )?;
            m.layer_omega(
                st,
                launch(n_columns, COLUMN_BLOCK),
                nl,
                n_columns as u32,
                &self.div,
                eta_rate,
                &s.d_sigma,
                &s.sigma_w,
                offsets,
                by_element,
                q,
                share,
                weights,
                &mut self.omega,
                &mut self.volume_source,
                &mut self.residual,
            )?;
        }
        Ok(())
    }

    /// The largest `|Ω|` at the surface before it was spread over the column
    /// (`LayerTransport::surface_residual`), from the device.
    pub fn surface_residual(&self) -> Result<f64, Box<dyn std::error::Error>> {
        let residual = self.residual.to_host_vec(&self.gpu.stream)?;
        Ok(residual.iter().fold(0.0f64, |m, &r| m.max(r)).max(0.0))
    }

    /// The horizontal advection of a pair of fields (`w_points`: at the
    /// w-points), as [`Self::advect`] and [`Self::advect_w`] start.
    fn advect_horizontal(
        &self,
        fields: [&Advected; 2],
        rhs: [&mut DeviceBuffer<f64>; 2],
        interface: Interface,
        overwrite: bool,
        w_points: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let gpu = self.gpu.clone();
        let s = &self.statics;
        let nf = s.n_levels + usize::from(w_points);
        let n = s.n_elements * nf;
        assert!(
            !w_points || fields.iter().all(|a| a.exterior.is_none()),
            "exterior values are the layers'"
        );
        let (ext_slots_a, ext_a) = match fields[0].exterior {
            Some(e) => (&e.slots, &e.values),
            None => (&self.none_u32, &self.none),
        };
        let (ext_slots_b, ext_b) = match fields[1].exterior {
            Some(e) => (&e.slots, &e.values),
            None => (&self.none_u32, &self.none),
        };
        let (bc_kind_a, bc_value_a) = bc_code(fields[0].bc);
        let (bc_kind_b, bc_value_b) = bc_code(fields[1].bc);
        let [rhs_a, rhs_b] = rhs;
        // SAFETY: as in `compute`; `rhs` are none of the inputs.
        unsafe {
            gpu.module3d.advect_horizontal(
                &gpu.stream,
                launch(n, LEVEL_BLOCK),
                s.n_levels as u32,
                u32::from(w_points),
                n as u32,
                u32::from(overwrite),
                u32::from(interface == Interface::DonorBesideThin),
                &self.kinds,
                &self.thin,
                &s.averaged,
                &s.contra,
                &s.jinv,
                &s.fnode,
                &s.across,
                &s.dr,
                &s.ds,
                &s.lift,
                &s.line_ends,
                &s.w,
                &self.hu,
                &self.hv,
                &self.face,
                &self.subcell,
                fields[0].field,
                fields[1].field,
                ext_slots_a,
                ext_a,
                bc_kind_a,
                bc_value_a,
                ext_slots_b,
                ext_b,
                bc_kind_b,
                bc_value_b,
                rhs_a,
                rhs_b,
            )?;
        }
        Ok(())
    }

    /// The advection of the pair `fields` of the layers by these transports
    /// into `rhs`: overwritten with `−∇·(Q φ) − δ(Ω φ)` if `overwrite` (the
    /// tracers', `apply_tracer_transport_3d`), else that added to it (the
    /// momentum's, `apply_momentum_transport_3d`), the vertical
    /// reconstruction by `vertical`, the subcell interfaces crossed at
    /// `interface`'s value; with `sources`, then the water no transport
    /// carried at each field's value (`LayerTransport::add_volume_sources`),
    /// as `Hydrostatic3D::compute_transport_rhs_into` adds it to T, S, u and
    /// v.
    pub fn advect(
        &self,
        fields: [&Advected; 2],
        rhs: [&mut DeviceBuffer<f64>; 2],
        vertical: VerticalAdvection,
        interface: Interface,
        overwrite: bool,
        sources: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let [rhs_a, rhs_b] = rhs;
        self.advect_horizontal(
            fields,
            [&mut *rhs_a, &mut *rhs_b],
            interface,
            overwrite,
            false,
        )?;
        let gpu = self.gpu.clone();
        let (m, st) = (&gpu.module3d, &gpu.stream);
        let s = &self.statics;
        let nl = s.n_levels as u32;
        let n_columns = s.n_columns();
        // SAFETY: as in `compute`; `rhs` are none of the inputs.
        unsafe {
            for (advected, rhs) in fields.iter().zip([rhs_a, rhs_b]) {
                m.advect_vertical(
                    st,
                    launch(n_columns, COLUMN_BLOCK),
                    nl,
                    n_columns as u32,
                    scheme_code(vertical),
                    advected.range.0,
                    advected.range.1,
                    &self.kinds,
                    advected.field,
                    advected.reference.unwrap_or(&self.none),
                    &self.omega,
                    &s.d_sigma,
                    &self.dt_over_depth,
                    if sources {
                        &self.volume_source
                    } else {
                        &self.none
                    },
                    &mut *rhs,
                )?;
                if sources {
                    self.pooled_sources(rhs, advected.field, false)?;
                }
            }
        }
        Ok(())
    }

    /// The advection of a pair of fields at the w-points (the turbulence's
    /// k and ψ, `apply_w_point_transport_3d` with their zero-gradient
    /// boundary condition), overwriting `rhs`, the vertical reconstruction
    /// by `vertical`; with `sources`, then the rivers' water at each
    /// column's own value and the water no transport carried, as
    /// `Hydrostatic3D`'s turbulence advection adds them.
    pub fn advect_w(
        &self,
        fields: [&DeviceBuffer<f64>; 2],
        rhs: [&mut DeviceBuffer<f64>; 2],
        vertical: VerticalAdvection,
        sources: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let advected = fields.map(|field| Advected {
            field,
            reference: None,
            range: UNBOUNDED,
            exterior: None,
            bc: DeviceTracerBC3D::Extrapolation,
        });
        let [rhs_a, rhs_b] = rhs;
        self.advect_horizontal(
            [&advected[0], &advected[1]],
            [&mut *rhs_a, &mut *rhs_b],
            Interface::DonorBesideThin,
            true,
            true,
        )?;
        let gpu = self.gpu.clone();
        let (m, st) = (&gpu.module3d, &gpu.stream);
        let s = &self.statics;
        let n_columns = s.n_columns();
        let [offsets, by_element] = self.river_tables();
        let [q, share, weights] = self.river_values();
        let none_u32 = &self.none_u32;
        let (offsets, by_element) = if sources {
            (offsets, by_element)
        } else {
            (none_u32, none_u32)
        };
        // SAFETY: as in `compute`; `rhs` are none of the inputs.
        unsafe {
            for (field, rhs) in fields.into_iter().zip([rhs_a, rhs_b]) {
                m.advect_vertical_w(
                    st,
                    launch(n_columns, COLUMN_BLOCK),
                    s.n_levels as u32,
                    n_columns as u32,
                    scheme_code(vertical),
                    &self.kinds,
                    field,
                    &self.omega,
                    &s.d_sigma_w,
                    offsets,
                    by_element,
                    q,
                    share,
                    weights,
                    if sources {
                        &self.volume_source
                    } else {
                        &self.none
                    },
                    &mut *rhs,
                )?;
                if sources {
                    self.pooled_sources(rhs, field, true)?;
                }
            }
        }
        Ok(())
    }

    /// `pooled_sources` of `field` into `rhs`, over the layers or the
    /// w-cells.
    pub(crate) fn pooled_sources(
        &self,
        rhs: &mut DeviceBuffer<f64>,
        field: &DeviceBuffer<f64>,
        w_points: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let gpu = self.gpu.clone();
        let s = &self.statics;
        let nc = s.n_levels + usize::from(w_points);
        let n = s.n_elements * nc;
        // SAFETY: as in `compute`; `rhs` is not `field`.
        unsafe {
            gpu.module3d.pooled_sources(
                &gpu.stream,
                launch(n, LEVEL_BLOCK),
                nc as u32,
                n as u32,
                &self.kinds,
                &s.mass,
                &self.volume_source,
                if w_points { &s.d_sigma_w } else { &s.d_sigma },
                field,
                rhs,
            )?;
        }
        Ok(())
    }

    /// The river water's tracer of concentration `conc` (one per river, at
    /// the stage's time) added to `rhs` (`RiverInflow::add_tracer_sources`);
    /// nothing without rivers.
    pub fn river_sources(
        &self,
        rhs: &mut DeviceBuffer<f64>,
        conc: &DeviceBuffer<f64>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let Some(rivers) = &self.rivers else {
            return Ok(());
        };
        let gpu = self.gpu.clone();
        let nl = self.statics.n_levels;
        let t = &rivers.tables;
        let n = t.n_elements_with * NN * nl;
        if n == 0 {
            return Ok(());
        }
        // SAFETY: as in `compute`; the listed elements are distinct.
        unsafe {
            gpu.module3d.river_sources(
                &gpu.stream,
                launch(n, LEVEL_BLOCK),
                nl as u32,
                n as u32,
                &t.elements,
                &t.offsets,
                &t.by_element,
                &rivers.q,
                &t.share,
                &rivers.weights,
                conc,
                rhs,
            )?;
        }
        Ok(())
    }
}

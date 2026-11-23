//! The 3D stages' momentum terms on the GPU (P2.7 Phase 4.2, `notes/gpu.md`):
//! the reference stratification at a stage's levels and every element's
//! displacement about it (dgcom's `ReferenceFields::fill`,
//! `ReferenceFields::set_displacements`), the baroclinic pressure gradient in
//! the σ-pairs form (`compute_pressure_gradient`) with Coriolis
//! (`apply_coriolis_3d`), as `compute_momentum_rhs_3d` takes them, and the
//! reference's advection of T and S, the pressure gradient's adjoint
//! (`add_reference_advection`), with the water no transport carried after it
//! as `Hydrostatic3D::compute_transport_rhs_into` adds it.
//!
//! The column and pointwise functions are dgcom's own: the profile's lookup
//! (`ReferenceProfileView::at`), the linear equation of state,
//! `reference_column`, `element_displacement`, `column_pressure`,
//! `sigma_pair_difference`, `pair_coefficients`, `reference_edge`,
//! `taken_back_weight`. The loops over an element's node pairs, subcell
//! interfaces, faces and σ-surfaces mirror `baroclinic.rs` over
//! [`Statics3D`]'s tables. So a build without multiply–add contraction
//! (`--no-fmad`) gives the CPU's bits (`momentum3d.rs` and `transport3d.rs`,
//! the gates).
//!
//! Device forms exist for the σ-pairs form (the default; the constant-depth
//! one is for diagnostics) and the linear equation of state (the case
//! runner's). The kernels take one thread per column (the reference, the
//! pressure integrals), per element (the displacements, the reference
//! advection's totals), per subcell interface (the depth-mean force's
//! column sums) or per element and level (the force, the reference's
//! advection), as `three_d.rs`.

// As `three_d.rs`: the device loops index arrays by the loop variable.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use std::sync::Arc;

use cuda_core::DeviceBuffer;
use cuda_device::{DisjointSlice, device, kernel, thread};
use cuda_host::cuda_module;
use dgcom::mesh::Mesh2D;
use dgcom::operators::DGOperators2D;
use dgcom::physics::{EquationOfState, LinearEOS};
use dgcom::solver::rhs::baroclinic::{
    BalancedReference, ColumnPressure, ColumnReference, ElementReference, PairEnd, ReferenceColumn,
    column_pressure, element_displacement, inverse_layer_thicknesses, pair_coefficients,
    reference_column, reference_edge, sigma_pair_difference, taken_back_weight,
};
use dgcom::solver::rhs::subcells::{line_node, subcell_slot};
use dgcom::solver::{ReferenceProfileView, TracerReferenceProfile};
use dgcom::source::CoriolisSource2D;
use dgcom::types::ElementIndex;
use dgcom::vertical::SigmaGrid;

use crate::swe2d::{Gpu, N1, NN, face_node};
use crate::three_d::{
    COLUMN_BLOCK, LEVEL_BLOCK, MAX_LEVELS, NSUB, OPEN, POOLING, SUBCELL, Statics3D, Transport3D,
    WALL, launch,
};

/// The end of a pair at level `l` of column `c` (flat): its pressure, its
/// density anomaly (about the reference `density` at its own levels, or
/// `rho_ref` without one: `density` empty), its height, and the reference
/// there, as `Columns::fill` and `ColumnView::end` take them.
#[device]
#[allow(unused_attributes)]
#[inline(always)]
fn pair_end(
    c: usize,
    nl: usize,
    l: usize,
    eta: &[f64],
    bed: &[f64],
    sigma_rho: &[f64],
    rho: &[f64],
    rho_ref: f64,
    density: &[f64],
    stratification: &[f64],
    at_level: &[f64],
) -> PairEnd {
    let (e, b) = (eta[c], bed[c]);
    let depth = e - b;
    let at = c * nl + l;
    let (reference, gamma, anomaly) = if density.is_empty() {
        (0.0, 0.0, rho[at] - rho_ref)
    } else {
        (density[at], stratification[at], rho[at] - density[at])
    };
    PairEnd {
        pressure: at_level[at],
        rho: anomaly,
        z: e + sigma_rho[l] * depth,
        reference,
        stratification: gamma,
    }
}

/// The derivative coefficients of the pair `(i, j)` of element `k`
/// (`pair_derivatives`) from the tables.
#[device]
#[allow(unused_attributes)]
#[inline(always)]
fn pair_at(
    k: usize,
    i: usize,
    j: usize,
    averaged: bool,
    contra: &[[f64; 4]],
    jinv: &[f64],
    dr: &[f64],
    ds: &[f64],
) -> [(f64, f64); 2] {
    let (ci, cj) = (contra[k * NN + i], contra[k * NN + j]);
    pair_coefficients(
        [
            ((ci[0], ci[1]), (ci[2], ci[3])),
            ((cj[0], cj[1]), (cj[2], cj[3])),
        ],
        [jinv[k * NN + i], jinv[k * NN + j]],
        [
            (dr[i * NN + j], ds[i * NN + j]),
            (dr[j * NN + i], ds[j * NN + i]),
        ],
        averaged,
    )
}

/// The nodes `(i, j)` (of the element) of subcell interface `slot`.
#[device]
#[allow(unused_attributes)]
#[inline(always)]
fn interface_nodes(slot: usize) -> (usize, usize, usize) {
    let a = slot % (N1 - 1);
    let (dir, line) = ((slot / (N1 - 1)) / N1, (slot / (N1 - 1)) % N1);
    (
        line_node(N1, dir, line, a),
        line_node(N1, dir, line, a + 1),
        line,
    )
}

// =============================================================================
// Kernels
// =============================================================================

#[cuda_module]
pub mod kernels_baroclinic {
    use super::*;

    /// `ReferenceFields::fill` of every column, one thread per column: the
    /// reference profile (its samples `heights`, `temp`, `salt`, their slopes
    /// if cubic, else empty; `inverse_spacing` if `even`; `inverse_widths`)
    /// at the column's levels, with the linear equation of state's density
    /// and partials (`eos`: ρ₀, T₀, S₀, α, β), by dgcom's `reference_column`.
    /// A column whose `η` is the one it was last filled at (`filled_at`) is
    /// left as it is, as the CPU leaves an element.
    #[kernel]
    pub fn reference_fields(
        nl: u32,
        n_columns: u32,
        heights: &[f64],
        temp: &[f64],
        salt: &[f64],
        slopes_t: &[f64],
        slopes_s: &[f64],
        even: u32,
        inverse_spacing: f64,
        inverse_widths: &[f64],
        eos: &[f64],
        eta: &[f64],
        bed: &[f64],
        sigma_rho: &[f64],
        sigma_w: &[f64],
        d_sigma: &[f64],
        inverse_d_sigma: &[f64],
        mut filled_at: DisjointSlice<f64>,
        mut out_t: DisjointSlice<f64>,
        mut out_s: DisjointSlice<f64>,
        mut gradient_t: DisjointSlice<f64>,
        mut gradient_s: DisjointSlice<f64>,
        mut density: DisjointSlice<f64>,
        mut stratification: DisjointSlice<f64>,
        mut residual: DisjointSlice<f64>,
    ) {
        let idx = thread::index_1d().get();
        if idx >= n_columns as usize {
            return;
        }
        let nl = nl as usize;
        // SAFETY: thread idx alone writes column idx (idx < n_columns, nl
        // levels each).
        unsafe {
            let filled = filled_at.get_unchecked_mut(idx);
            if *filled == eta[idx] {
                return;
            }
            *filled = eta[idx];
            let column = |out: *mut f64| core::slice::from_raw_parts_mut(out.add(idx * nl), nl);
            let view = ReferenceProfileView {
                heights,
                temp,
                salt,
                slopes: if slopes_t.is_empty() {
                    None
                } else {
                    Some([slopes_t, slopes_s])
                },
                inverse_spacing: if even != 0 {
                    Some(inverse_spacing)
                } else {
                    None
                },
                inverse_widths,
            };
            let eos = LinearEOS {
                rho0: eos[0],
                t0: eos[1],
                s0: eos[2],
                alpha: eos[3],
                beta: eos[4],
            };
            reference_column(
                |z| view.at(z),
                |t, s| eos.compute_density(t, s, 0.0),
                |t, s| eos.density_partials(t, s),
                eta[idx],
                bed[idx],
                [&sigma_rho[..nl], &sigma_w[..nl + 1], &d_sigma[..nl]],
                &inverse_d_sigma[..nl],
                ReferenceColumn {
                    tracers: [column(out_t.as_mut_ptr()), column(out_s.as_mut_ptr())],
                    gradients: [
                        column(gradient_t.as_mut_ptr()),
                        column(gradient_s.as_mut_ptr()),
                    ],
                    density: column(density.as_mut_ptr()),
                    stratification: column(stratification.as_mut_ptr()),
                    residual: column(residual.as_mut_ptr()),
                },
            );
        }
    }

    /// `ReferenceFields::set_displacements`, one thread per element: dgcom's
    /// `element_displacement` of the element's columns.
    #[kernel]
    pub fn displacements(
        nl: u32,
        n_elements: u32,
        min_column_depth: f64,
        eta: &[f64],
        bed: &[f64],
        mass: &[f64],
        rho: &[f64],
        density: &[f64],
        stratification: &[f64],
        residual: &[f64],
        d_sigma: &[f64],
        mut displacement: DisjointSlice<f64>,
    ) {
        let k = thread::index_1d().get();
        if k >= n_elements as usize {
            return;
        }
        let nl = nl as usize;
        let (nodes, block) = (k * NN..(k + 1) * NN, k * NN * nl..(k + 1) * NN * nl);
        let kappa = element_displacement(
            &eta[nodes.clone()],
            &bed[nodes.clone()],
            &mass[nodes],
            &rho[block.clone()],
            ElementReference {
                density: &density[block.clone()],
                stratification: &stratification[block.clone()],
                residual: &residual[block],
            },
            &d_sigma[..nl],
            min_column_depth,
        );
        // SAFETY: thread k alone writes element k's displacement.
        unsafe {
            *displacement.get_unchecked_mut(k) = kappa;
        }
    }

    /// The pressure (per `g`) at the levels of every column, one thread per
    /// column: dgcom's `column_pressure` of the density anomaly about the
    /// reference (`density`, `stratification` and the element's
    /// `displacement`; empty for none, then about `rho_ref`), its `at_level`
    /// into `at_level`.
    #[kernel]
    pub fn column_pressures(
        nl: u32,
        n_columns: u32,
        rho_ref: f64,
        eta: &[f64],
        bed: &[f64],
        rho: &[f64],
        density: &[f64],
        stratification: &[f64],
        displacement: &[f64],
        sigma_rho: &[f64],
        d_sigma: &[f64],
        mut at_level: DisjointSlice<f64>,
    ) {
        let idx = thread::index_1d().get();
        if idx >= n_columns as usize {
            return;
        }
        let nl = nl as usize;
        let column = idx * nl..(idx + 1) * nl;
        let reference = if density.is_empty() {
            None
        } else {
            Some(ColumnReference {
                density: &density[column.clone()],
                stratification: &stratification[column.clone()],
                displacement: displacement[idx / NN],
            })
        };
        let mut z = [0.0; MAX_LEVELS];
        let mut anomaly = [0.0; MAX_LEVELS];
        let mut slope = [0.0; MAX_LEVELS];
        let mut pressure = [0.0; MAX_LEVELS];
        // SAFETY: thread idx alone writes column idx's levels (idx <
        // n_columns).
        let out =
            unsafe { core::slice::from_raw_parts_mut(at_level.as_mut_ptr().add(idx * nl), nl) };
        column_pressure(
            eta[idx],
            bed[idx],
            &sigma_rho[..nl],
            &d_sigma[..nl],
            &rho[column],
            rho_ref,
            reference,
            ColumnPressure {
                z: &mut z[..nl],
                rho: &mut anomaly[..nl],
                slope: &mut slope[..nl],
                pressure: &mut pressure[..nl],
                at_level: out,
            },
        );
    }

    /// The column sums `Σ_l Δσ_l Δp_l` of the pressure differences along
    /// every subcell interface between two columns that are not thin, in an
    /// element with a thin column (`subcell_mean_term`'s; about the
    /// reference, with the element's displacement), one thread per
    /// interface (`k·NSUB + slot`); the others' are left as they are
    /// (nothing reads them).
    #[kernel]
    pub fn subcell_means(
        nl: u32,
        n: u32,
        min_column_depth: f64,
        eta: &[f64],
        bed: &[f64],
        rho: &[f64],
        density: &[f64],
        stratification: &[f64],
        displacement: &[f64],
        at_level: &[f64],
        sigma_rho: &[f64],
        d_sigma: &[f64],
        mut sub_mean: DisjointSlice<f64>,
    ) {
        let t = thread::index_1d().get();
        if t >= n as usize {
            return;
        }
        let nl = nl as usize;
        let (k, slot) = (t / NSUB, t % NSUB);
        let wet = |c: usize| eta[c] - bed[c] >= min_column_depth;
        if (k * NN..(k + 1) * NN).all(wet) {
            return;
        }
        let (i, j, _) = interface_nodes(slot);
        let (i, j) = (k * NN + i, k * NN + j);
        if !(wet(i) && wet(j)) {
            return;
        }
        let kappa = displacement[k];
        let end = |c: usize, l: usize| {
            pair_end(
                c,
                nl,
                l,
                eta,
                bed,
                sigma_rho,
                rho,
                0.0,
                density,
                stratification,
                at_level,
            )
        };
        let mean = (0..nl)
            .map(|l| sigma_pair_difference(end(i, l), end(j, l), kappa) * d_sigma[l])
            .sum::<f64>();
        // SAFETY: thread t alone writes interface t.
        unsafe {
            *sub_mean.get_unchecked_mut(t) = mean;
        }
    }

    /// `compute_momentum_rhs_3d`'s velocity tendencies, one thread per
    /// element and level (`k·nl + l`): the σ-pairs pressure gradient
    /// (`pressure_gradient_element`: the volume term in an element without a
    /// thin column, about a reference the subcell interfaces' depth-mean
    /// term in one with, the lifted pressure jumps across its faces to other
    /// elements) times `scale` (`−g/ρ₀`), the balanced reference's
    /// correction (empty: none), and Coriolis (`f` per node), overwriting
    /// `rhs_u`, `rhs_v`. Without a reference (`density` empty) the anomaly
    /// is about `rho_ref`.
    #[kernel]
    pub fn pressure_gradient(
        nl: u32,
        n: u32,
        min_column_depth: f64,
        scale: f64,
        rho_ref: f64,
        eta: &[f64],
        bed: &[f64],
        rho: &[f64],
        density: &[f64],
        stratification: &[f64],
        displacement: &[f64],
        at_level: &[f64],
        sigma_rho: &[f64],
        averaged: &[u32],
        contra: &[[f64; 4]],
        jinv: &[f64],
        dr: &[f64],
        ds: &[f64],
        mass: &[f64],
        w: &[f64],
        sub_metric: &[[f64; 2]],
        fnode: &[[f64; 3]],
        across: &[u32],
        lift: &[f64],
        sub_mean: &[f64],
        balanced_x: &[f64],
        balanced_y: &[f64],
        coriolis: &[f64],
        u: &[f64],
        v: &[f64],
        mut rhs_u: DisjointSlice<f64>,
        mut rhs_v: DisjointSlice<f64>,
    ) {
        let t = thread::index_1d().get();
        if t >= n as usize {
            return;
        }
        let nl = nl as usize;
        let (k, l) = (t / nl, t % nl);
        let reference = !density.is_empty();
        let end = |c: usize| {
            pair_end(
                c,
                nl,
                l,
                eta,
                bed,
                sigma_rho,
                rho,
                rho_ref,
                density,
                stratification,
                at_level,
            )
        };
        let wet_at = |c: usize| eta[c] - bed[c] >= min_column_depth;
        let mut wet = [false; NN];
        let mut all_wet = true;
        #[unroll]
        for i in 0..NN {
            wet[i] = wet_at(k * NN + i);
            all_wet &= wet[i];
        }
        let mut px = [0.0; NN];
        let mut py = [0.0; NN];
        if all_wet {
            // `dg_volume_term`: every pair once, `Δp_ji = −Δp_ij`, in the
            // CPU's order (all of them: P2's `D_r`, `D_s` carry round-off off
            // the GLL lines)
            let kappa = if reference { displacement[k] } else { 0.0 };
            let avg = averaged[k] != 0;
            let mut ends = [end(k * NN); NN];
            #[unroll]
            for i in 1..NN {
                ends[i] = end(k * NN + i);
            }
            // j over all nodes, so that the trip count is constant and the
            // loops unroll
            #[unroll]
            for i in 0..NN {
                #[unroll]
                for j in 0..NN {
                    if j > i {
                        let [(dx_ij, dy_ij), (dx_ji, dy_ji)] =
                            pair_at(k, i, j, avg, contra, jinv, dr, ds);
                        let dp = sigma_pair_difference(ends[i], ends[j], kappa);
                        px[i] += dx_ij * dp;
                        py[i] += dy_ij * dp;
                        px[j] -= dx_ji * dp;
                        py[j] -= dy_ji * dp;
                    }
                }
            }
        } else if reference {
            // `subcell_mean_term`: the interfaces between columns that are
            // not thin, in the CPU's order
            #[unroll]
            for dir in 0..2 {
                #[unroll]
                for line in 0..N1 {
                    #[unroll]
                    for a in 0..N1 - 1 {
                        let (i, j) = (line_node(N1, dir, line, a), line_node(N1, dir, line, a + 1));
                        if wet[i] && wet[j] {
                            let slot = k * NSUB + subcell_slot(N1, dir, line, a);
                            let m = sub_metric[slot];
                            let half = 0.5 * w[line] * sub_mean[slot];
                            let si = half / mass[k * NN + i];
                            px[i] += si * m[0];
                            py[i] += si * m[1];
                            let sj = half / mass[k * NN + j];
                            px[j] += sj * m[0];
                            py[j] += sj * m[1];
                        }
                    }
                }
            }
        }
        // The faces to other elements: LIFT (n·(p* − p⁻)), p* − p⁻ = ½Δp,
        // no displacement term across a face; into the face node alone. The
        // LIFT's other entries are zero (`Statics3D::upload` checks it), and
        // a sum from +0 is never −0, so the CPU's ±0 terms there change none
        // of these sums
        #[unroll]
        for f in 0..4 {
            #[unroll]
            for fi in 0..N1 {
                let s = (k * 4 + f) * N1 + fi;
                let nb = across[s];
                let node = face_node(f, fi);
                if nb != WALL && nb != OPEN && wet[node] && wet_at(nb as usize) {
                    let dp = sigma_pair_difference(end(k * NN + node), end(nb as usize), 0.0);
                    let jump = 0.5 * dp;
                    if jump != 0.0 {
                        let g = fnode[s];
                        let (jx, jy) = (g[2] * g[0] * jump, g[2] * g[1] * jump);
                        let lf = lift[(f * NN + node) * N1 + fi];
                        px[node] += lf * jx;
                        py[node] += lf * jy;
                    }
                }
            }
        }
        // SAFETY: thread (k, l) alone writes level l of element k's nodes.
        unsafe {
            #[unroll]
            for i in 0..NN {
                let at = (k * NN + i) * nl + l;
                let mut a = scale * px[i];
                let mut b = scale * py[i];
                if !balanced_x.is_empty() {
                    a += balanced_x[at];
                    b += balanced_y[at];
                }
                let f = coriolis[k * NN + i];
                a += f * v[at];
                b -= f * u[at];
                *rhs_u.get_unchecked_mut(at) = a;
                *rhs_v.get_unchecked_mut(at) = b;
            }
        }
    }

    /// `add_reference_advection`'s edges, one thread per element and level
    /// (`k·nl + l`), T and S together: the pairs of the split form in an
    /// element without a thin column that moves its layers by them, or the
    /// subcell interfaces between columns that are not thin in a subcell
    /// element, then the σ-surfaces of every column that is not thin,
    /// added to `rhs_t`, `rhs_s` in the CPU's order; what they make of each
    /// tracer on the level into `made` (`[element][level][tracer]`).
    #[kernel]
    pub fn reference_edges(
        nl: u32,
        n: u32,
        min_column_depth: f64,
        kinds: &[u32],
        averaged: &[u32],
        contra: &[[f64; 4]],
        jinv: &[f64],
        dr: &[f64],
        ds: &[f64],
        mass: &[f64],
        w: &[f64],
        eta: &[f64],
        bed: &[f64],
        sigma_rho: &[f64],
        hu: &[f64],
        hv: &[f64],
        sub: &[f64],
        omega: &[f64],
        reference_t: &[f64],
        reference_s: &[f64],
        gradient_t: &[f64],
        gradient_s: &[f64],
        temp: &[f64],
        salt: &[f64],
        mut rhs_t: DisjointSlice<f64>,
        mut rhs_s: DisjointSlice<f64>,
        mut made: DisjointSlice<f64>,
    ) {
        let t = thread::index_1d().get();
        if t >= n as usize {
            return;
        }
        let nl = nl as usize;
        let (k, l) = (t / nl, t % nl);
        let mut e = [0.0; NN];
        let mut depth = [0.0; NN];
        let mut wet = [false; NN];
        let mut all_wet = true;
        let mut out_t = [0.0; NN];
        let mut out_s = [0.0; NN];
        // SAFETY: thread (k, l) alone reads and writes level l of element
        // k's nodes, and its own `made` entries.
        unsafe {
            #[unroll]
            for i in 0..NN {
                let c = k * NN + i;
                e[i] = eta[c];
                depth[i] = eta[c] - bed[c];
                wet[i] = depth[i] >= min_column_depth;
                all_wet &= wet[i];
                out_t[i] = *rhs_t.get_unchecked_mut(c * nl + l);
                out_s[i] = *rhs_s.get_unchecked_mut(c * nl + l);
            }
        }
        // One edge's terms at its ends `i`, `j` (rows `ci`, `cj` of the
        // level), over the ends' masses
        let edge = |half: f64,
                    dz: f64,
                    ci: usize,
                    cj: usize,
                    reference: &[f64],
                    gradient: &[f64],
                    tracer: &[f64]| {
            reference_edge(
                half,
                dz,
                [reference[ci], reference[cj]],
                [gradient[ci], gradient[cj]],
                [tracer[ci], tracer[cj]],
            )
        };
        let (mut made_t, mut made_s) = (0.0, 0.0);
        let subcell = kinds[k] & SUBCELL != 0;
        // The nodes' `1/m`, as the CPU divides per edge
        let mut inverse_mass = [0.0; NN];
        #[unroll]
        for i in 0..NN {
            inverse_mass[i] = 1.0 / mass[k * NN + i];
        }
        if !subcell && all_wet {
            let avg = averaged[k] != 0;
            #[unroll]
            for i in 0..NN {
                #[unroll]
                for j in 0..NN {
                    // Only nodes on one GLL line are a pair (the CPU's test:
                    // P2's `D_r`, `D_s` carry round-off off the lines, so
                    // few pairs are apart)
                    let apart_ij = dr[i * NN + j] == 0.0 && ds[i * NN + j] == 0.0;
                    let apart_ji = dr[j * NN + i] == 0.0 && ds[j * NN + i] == 0.0;
                    if j > i && !(apart_ij && apart_ji) {
                        let [(dx_ij, dy_ij), (dx_ji, dy_ji)] =
                            pair_at(k, i, j, avg, contra, jinv, dr, ds);
                        let (m_i, m_j) = (mass[k * NN + i], mass[k * NN + j]);
                        let (ax, ay) = (m_i * dx_ij, m_i * dy_ij);
                        let (bx, by) = (m_j * dx_ji, m_j * dy_ji);
                        let (d_eta, d_depth) = (e[j] - e[i], depth[j] - depth[i]);
                        let (ci, cj) = ((k * NN + i) * nl + l, (k * NN + j) * nl + l);
                        let half =
                            0.5 * ((ax * hu[ci] + ay * hv[ci]) - (bx * hu[cj] + by * hv[cj]));
                        let dz = d_eta + sigma_rho[l] * d_depth;
                        let (inv_i, inv_j) = (inverse_mass[i], inverse_mass[j]);
                        let [a, b] = edge(half, dz, ci, cj, reference_t, gradient_t, temp);
                        out_t[i] += a * inv_i;
                        out_t[j] += b * inv_j;
                        made_t += a + b;
                        let [a, b] = edge(half, dz, ci, cj, reference_s, gradient_s, salt);
                        out_s[i] += a * inv_i;
                        out_s[j] += b * inv_j;
                        made_s += a + b;
                    }
                }
            }
        } else if subcell {
            #[unroll]
            for dir in 0..2 {
                #[unroll]
                for line in 0..N1 {
                    #[unroll]
                    for a in 0..N1 - 1 {
                        let (i, j) = (line_node(N1, dir, line, a), line_node(N1, dir, line, a + 1));
                        if wet[i] && wet[j] {
                            let slot = k * NSUB + subcell_slot(N1, dir, line, a);
                            let q = sub[slot * nl + l];
                            let (d_eta, d_depth) = (e[j] - e[i], depth[j] - depth[i]);
                            let half = 0.5 * w[line] * q;
                            let dz = d_eta + sigma_rho[l] * d_depth;
                            let (ci, cj) = ((k * NN + i) * nl + l, (k * NN + j) * nl + l);
                            let (inv_i, inv_j) = (inverse_mass[i], inverse_mass[j]);
                            let [a, b] = edge(half, dz, ci, cj, reference_t, gradient_t, temp);
                            out_t[i] += a * inv_i;
                            out_t[j] += b * inv_j;
                            made_t += a + b;
                            let [a, b] = edge(half, dz, ci, cj, reference_s, gradient_s, salt);
                            out_s[i] += a * inv_i;
                            out_s[j] += b * inv_j;
                            made_s += a + b;
                        }
                    }
                }
            }
        }
        // The σ-surfaces of every column that is not thin: the one below
        // the level (its upper end), then the one above (its lower end),
        // each through the CPU's zeroed scratch (`0.0 + …`, `… * 1.0`)
        #[unroll]
        for i in 0..NN {
            if wet[i] {
                let c = k * NN + i;
                let (base, row) = (c * (nl + 1), c * nl);
                if l >= 1 {
                    let half = 0.5 * omega[base + l];
                    let dz = (sigma_rho[l] - sigma_rho[l - 1]) * depth[i];
                    let (lo, hi) = (row + l - 1, row + l);
                    let [_, b] = edge(half, dz, lo, hi, reference_t, gradient_t, temp);
                    out_t[i] += 0.0 + b * 1.0;
                    let [_, b] = edge(half, dz, lo, hi, reference_s, gradient_s, salt);
                    out_s[i] += 0.0 + b * 1.0;
                }
                if l + 1 < nl {
                    let half = 0.5 * omega[base + l + 1];
                    let dz = (sigma_rho[l + 1] - sigma_rho[l]) * depth[i];
                    let (lo, hi) = (row + l, row + l + 1);
                    let m = mass[c];
                    let [a, b] = edge(half, dz, lo, hi, reference_t, gradient_t, temp);
                    out_t[i] += 0.0 + a * 1.0;
                    made_t += m * (0.0 + (a + b));
                    let [a, b] = edge(half, dz, lo, hi, reference_s, gradient_s, salt);
                    out_s[i] += 0.0 + a * 1.0;
                    made_s += m * (0.0 + (a + b));
                }
            }
        }
        // SAFETY: as above.
        unsafe {
            #[unroll]
            for i in 0..NN {
                let at = (k * NN + i) * nl + l;
                *rhs_t.get_unchecked_mut(at) = out_t[i];
                *rhs_s.get_unchecked_mut(at) = out_s[i];
            }
            *made.get_unchecked_mut(t * 2) = made_t;
            *made.get_unchecked_mut(t * 2 + 1) = made_s;
        }
    }

    /// `add_reference_advection`'s totals, one thread per element: what the
    /// edges made of T and of S over the element (summed over its levels as
    /// the CPU sums them), the weight `N` it is taken back over (dgcom's
    /// `taken_back_weight`) and how, into `totals` (`[element][4]`: T, S,
    /// `N`, 0 for none taken back, 1 in the residual's shape, 2 uniform).
    #[kernel]
    pub fn reference_totals(
        nl: u32,
        n_elements: u32,
        min_column_depth: f64,
        eta: &[f64],
        bed: &[f64],
        mass: &[f64],
        density: &[f64],
        stratification: &[f64],
        residual: &[f64],
        d_sigma: &[f64],
        made: &[f64],
        mut totals: DisjointSlice<f64>,
    ) {
        let k = thread::index_1d().get();
        if k >= n_elements as usize {
            return;
        }
        let nl = nl as usize;
        let made_t = (0..nl).map(|l| made[(k * nl + l) * 2]).sum::<f64>();
        let made_s = (0..nl).map(|l| made[(k * nl + l) * 2 + 1]).sum::<f64>();
        let (how, weight) = if made_t == 0.0 && made_s == 0.0 {
            (0.0, 0.0)
        } else {
            let (nodes, block) = (k * NN..(k + 1) * NN, k * NN * nl..(k + 1) * NN * nl);
            let (shaped, weight) = taken_back_weight(
                &eta[nodes.clone()],
                &bed[nodes.clone()],
                &mass[nodes],
                ElementReference {
                    density: &density[block.clone()],
                    stratification: &stratification[block.clone()],
                    residual: &residual[block],
                },
                &d_sigma[..nl],
                min_column_depth,
            );
            (if shaped { 1.0 } else { 2.0 }, weight)
        };
        // SAFETY: thread k alone writes element k's totals.
        unsafe {
            *totals.get_unchecked_mut(k * 4) = made_t;
            *totals.get_unchecked_mut(k * 4 + 1) = made_s;
            *totals.get_unchecked_mut(k * 4 + 2) = weight;
            *totals.get_unchecked_mut(k * 4 + 3) = how;
        }
    }

    /// `add_reference_advection`'s correction, one thread per element and
    /// level: the element's tracer error taken back as a displacement of
    /// its stratification, `−H_z Γ g R/N` (`g = 1` uniform) in every column
    /// that is not thin; then, if `volume_source` is not empty and the
    /// element is not pooling, the water no transport carried at T's and
    /// S's own values (`LayerTransport::add_volume_sources`, after the
    /// reference's advection as on the CPU; `pooled_sources` adds the
    /// pooling elements').
    #[kernel]
    pub fn reference_correction(
        nl: u32,
        n: u32,
        min_column_depth: f64,
        kinds: &[u32],
        eta: &[f64],
        bed: &[f64],
        stratification: &[f64],
        residual: &[f64],
        d_sigma: &[f64],
        totals: &[f64],
        volume_source: &[f64],
        temp: &[f64],
        salt: &[f64],
        mut rhs_t: DisjointSlice<f64>,
        mut rhs_s: DisjointSlice<f64>,
    ) {
        let t = thread::index_1d().get();
        if t >= n as usize {
            return;
        }
        let nl = nl as usize;
        let (k, l) = (t / nl, t % nl);
        let (made_t, made_s, weight, how) = (
            totals[k * 4],
            totals[k * 4 + 1],
            totals[k * 4 + 2],
            totals[k * 4 + 3],
        );
        // None taken back as the CPU returns early (a NaN weight is taken)
        let none_taken = how == 0.0 || weight <= 0.0;
        let sources = !volume_source.is_empty() && kinds[k] & POOLING == 0;
        let ds = d_sigma[l];
        // SAFETY: thread (k, l) alone writes level l of element k's nodes.
        unsafe {
            #[unroll]
            for i in 0..NN {
                let c = k * NN + i;
                let at = c * nl + l;
                let rt = rhs_t.get_unchecked_mut(at);
                let rs = rhs_s.get_unchecked_mut(at);
                let depth = eta[c] - bed[c];
                if !none_taken && depth >= min_column_depth {
                    let share = depth / weight;
                    let gamma = stratification[at].max(0.0);
                    let shape = if how == 1.0 { residual[at] } else { 1.0 };
                    let taken = share * made_t;
                    *rt -= taken * ds * gamma * shape;
                    let taken = share * made_s;
                    *rs -= taken * ds * gamma * shape;
                }
                if sources {
                    let s = volume_source[c];
                    *rt += ds * s * temp[at];
                    *rs += ds * s * salt[at];
                }
            }
        }
    }
}

// =============================================================================
// Host side
// =============================================================================

/// A reference profile's samples on the device (`ReferenceProfileView`'s
/// slices).
struct ProfileDevice {
    heights: DeviceBuffer<f64>,
    temp: DeviceBuffer<f64>,
    salt: DeviceBuffer<f64>,
    /// `[dT/dz, dS/dz]` if cubic, else empty
    slopes: [DeviceBuffer<f64>; 2],
    inverse_spacing: Option<f64>,
    inverse_widths: DeviceBuffer<f64>,
}

/// The reference stratification at a stage's levels on the device (dgcom's
/// `ReferenceFields`), `[element][node][level]` but the displacements.
pub struct ReferenceFieldsDevice {
    /// `T̄`, `S̄` at the layer centres
    pub tracers: [DeviceBuffer<f64>; 2],
    /// Their gradients over each layer
    pub gradients: [DeviceBuffer<f64>; 2],
    /// `ρ̄`
    pub density: DeviceBuffer<f64>,
    /// `Γ`
    pub stratification: DeviceBuffer<f64>,
    /// `Γ⁺`'s residual about each column's line in `z`
    pub residual: DeviceBuffer<f64>,
    /// `κ` of every element
    pub displacement: DeviceBuffer<f64>,
    /// The `η` every column was last filled at (NaN before the first)
    filled_at: DeviceBuffer<f64>,
}

/// The 3D momentum terms on the device that do not move with the layer
/// fluxes (the baroclinic pressure gradient and Coriolis), and the
/// reference's advection.
pub struct Baroclinic3D {
    gpu: Arc<Gpu>,
    /// `−g/ρ₀`
    scale: f64,
    rho0: f64,
    min_column_depth: f64,
    /// The linear equation of state: ρ₀, T₀, S₀, α, β
    eos: DeviceBuffer<f64>,
    profile: Option<ProfileDevice>,
    /// The reference at the levels of the last [`Self::reference`]
    pub fields: Option<ReferenceFieldsDevice>,
    inverse_d_sigma: DeviceBuffer<f64>,
    /// `f` at every node
    coriolis: DeviceBuffer<f64>,
    /// A balanced reference state's correction (`BalancedReference`)
    balanced: Option<[DeviceBuffer<f64>; 2]>,
    /// Every column's pressure at its levels
    at_level: DeviceBuffer<f64>,
    /// The subcell interfaces' column sums of the pressure differences
    pub(crate) sub_mean: DeviceBuffer<f64>,
    /// What the reference's edges make of T and S, per element and level
    made: DeviceBuffer<f64>,
    totals: DeviceBuffer<f64>,
    none: DeviceBuffer<f64>,
}

/// What [`Baroclinic3D::new`] takes of a 3D physics module besides the mesh.
pub struct BaroclinicSetup<'a> {
    pub coriolis: &'a CoriolisSource2D,
    pub eos: LinearEOS,
    /// The reference stratification, if any (`Hydrostatic3D`'s active one)
    pub profile: Option<&'a TracerReferenceProfile>,
    /// A balanced reference state, if any
    pub balanced: Option<&'a BalancedReference>,
    pub g: f64,
    pub rho0: f64,
    pub min_column_depth: f64,
}

impl Baroclinic3D {
    pub fn new(
        gpu: &Arc<Gpu>,
        statics: &Statics3D,
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        sigma: &SigmaGrid,
        setup: BaroclinicSetup,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let stream = &gpu.stream;
        let up = |v: &[f64]| DeviceBuffer::from_host(stream, v);
        let (ne, nl) = (statics.n_elements, statics.n_levels);
        let n = ne * NN * nl;
        let zeros = |n: usize| DeviceBuffer::<f64>::zeroed(stream, n);
        let eos = setup.eos;
        let profile = match setup.profile {
            Some(profile) => {
                let view = profile.view();
                Some(ProfileDevice {
                    heights: up(view.heights)?,
                    temp: up(view.temp)?,
                    salt: up(view.salt)?,
                    slopes: match view.slopes {
                        Some([t, s]) => [up(t)?, up(s)?],
                        None => [zeros(0)?, zeros(0)?],
                    },
                    inverse_spacing: view.inverse_spacing,
                    inverse_widths: up(view.inverse_widths)?,
                })
            }
            None => None,
        };
        let fields = match setup.profile {
            Some(_) => Some(ReferenceFieldsDevice {
                tracers: [zeros(n)?, zeros(n)?],
                gradients: [zeros(n)?, zeros(n)?],
                density: zeros(n)?,
                stratification: zeros(n)?,
                residual: zeros(n)?,
                displacement: zeros(ne)?,
                filled_at: up(&vec![f64::NAN; ne * NN])?,
            }),
            None => None,
        };
        let mut inverse_d_sigma = vec![0.0; nl];
        inverse_layer_thicknesses(sigma.sigma_w(), &mut inverse_d_sigma);
        // f at every node, as `apply_coriolis_3d` takes it
        let mut f = Vec::with_capacity(ne * NN);
        for k in 0..ne {
            let ki = ElementIndex::new(k);
            for i in 0..NN {
                let [_x, y] = mesh.reference_to_physical(ki, ops.nodes_r[i], ops.nodes_s[i]);
                f.push(setup.coriolis.f_at(y));
            }
        }
        let balanced = match setup.balanced {
            Some(b) => Some([up(&b.correction[0])?, up(&b.correction[1])?]),
            None => None,
        };
        Ok(Self {
            gpu: gpu.clone(),
            scale: -setup.g / setup.rho0,
            rho0: setup.rho0,
            min_column_depth: setup.min_column_depth,
            eos: up(&[eos.rho0, eos.t0, eos.s0, eos.alpha, eos.beta])?,
            profile,
            fields,
            inverse_d_sigma: up(&inverse_d_sigma)?,
            coriolis: up(&f)?,
            balanced,
            at_level: zeros(n)?,
            sub_mean: zeros(ne * NSUB)?,
            made: zeros(ne * nl * 2)?,
            totals: zeros(ne * 4)?,
            none: zeros(0)?,
        })
    }

    /// The reference at the levels of `eta` (the columns whose `η` moved)
    /// and every element's displacement about it from the density `rho`
    /// (`ReferenceFields::fill`, `ReferenceFields::set_displacements`);
    /// nothing without a reference. Before [`Self::momentum`] and
    /// [`Self::advect_reference`] of a state.
    pub fn reference(
        &mut self,
        statics: &Statics3D,
        eta: &DeviceBuffer<f64>,
        rho: &DeviceBuffer<f64>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let (Some(profile), Some(fields)) = (&self.profile, &mut self.fields) else {
            return Ok(());
        };
        let gpu = &self.gpu;
        let (m, st) = (&gpu.module_baroclinic, &gpu.stream);
        let s = statics;
        let nl = s.n_levels as u32;
        let n_columns = s.n_columns();
        let ReferenceFieldsDevice {
            tracers: [temp, salt],
            gradients: [gradient_t, gradient_s],
            density,
            stratification,
            residual,
            displacement,
            filled_at,
        } = fields;
        // SAFETY: every buffer is sized for this mesh and these levels
        // (`Statics3D::upload`, `Self::new`, the caller's state); no output
        // is an input.
        unsafe {
            m.reference_fields(
                st,
                launch(n_columns, COLUMN_BLOCK),
                nl,
                n_columns as u32,
                &profile.heights,
                &profile.temp,
                &profile.salt,
                &profile.slopes[0],
                &profile.slopes[1],
                u32::from(profile.inverse_spacing.is_some()),
                profile.inverse_spacing.unwrap_or(0.0),
                &profile.inverse_widths,
                &self.eos,
                eta,
                &s.bed,
                &s.sigma_rho,
                &s.sigma_w,
                &s.d_sigma,
                &self.inverse_d_sigma,
                filled_at,
                temp,
                salt,
                gradient_t,
                gradient_s,
                density,
                stratification,
                residual,
            )?;
            m.displacements(
                st,
                launch(s.n_elements, COLUMN_BLOCK),
                nl,
                s.n_elements as u32,
                self.min_column_depth,
                eta,
                &s.bed,
                &s.mass,
                rho,
                density,
                stratification,
                residual,
                &s.d_sigma,
                displacement,
            )?;
        }
        Ok(())
    }

    /// `compute_momentum_rhs_3d` of the state `(eta, rho, u, v)`: `rhs_u`,
    /// `rhs_v` overwritten with the baroclinic pressure gradient (σ-pairs,
    /// about the reference of the last [`Self::reference`] if any, else
    /// about ρ₀), the balanced reference's correction and Coriolis.
    pub fn momentum(
        &mut self,
        statics: &Statics3D,
        eta: &DeviceBuffer<f64>,
        rho: &DeviceBuffer<f64>,
        u: &DeviceBuffer<f64>,
        v: &DeviceBuffer<f64>,
        rhs_u: &mut DeviceBuffer<f64>,
        rhs_v: &mut DeviceBuffer<f64>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.pressures(statics, eta, rho)?;
        self.gradient(statics, eta, rho, u, v, rhs_u, rhs_v)
    }

    /// The (element, node, level) arrays the reference's fields, or empty
    /// buffers without one.
    fn reference_buffers(&self) -> [&DeviceBuffer<f64>; 3] {
        match &self.fields {
            Some(f) => [&f.density, &f.stratification, &f.displacement],
            None => [&self.none; 3],
        }
    }

    /// [`Self::momentum`]'s first part: every column's pressure at its
    /// levels, and (about a reference) the subcell interfaces' column sums.
    pub(crate) fn pressures(
        &mut self,
        statics: &Statics3D,
        eta: &DeviceBuffer<f64>,
        rho: &DeviceBuffer<f64>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let gpu = self.gpu.clone();
        let (m, st) = (&gpu.module_baroclinic, &gpu.stream);
        let s = statics;
        let nl = s.n_levels as u32;
        let n_columns = s.n_columns();
        let referenced = self.fields.is_some();
        let (density, stratification, displacement) = match &self.fields {
            Some(f) => (&f.density, &f.stratification, &f.displacement),
            None => (&self.none, &self.none, &self.none),
        };
        // SAFETY: as in `reference`; the outputs are none of the inputs.
        unsafe {
            m.column_pressures(
                st,
                launch(n_columns, COLUMN_BLOCK),
                nl,
                n_columns as u32,
                self.rho0,
                eta,
                &s.bed,
                rho,
                density,
                stratification,
                displacement,
                &s.sigma_rho,
                &s.d_sigma,
                &mut self.at_level,
            )?;
            if referenced {
                let n_sub = s.n_elements * NSUB;
                m.subcell_means(
                    st,
                    launch(n_sub, COLUMN_BLOCK),
                    nl,
                    n_sub as u32,
                    self.min_column_depth,
                    eta,
                    &s.bed,
                    rho,
                    density,
                    stratification,
                    displacement,
                    &self.at_level,
                    &s.sigma_rho,
                    &s.d_sigma,
                    &mut self.sub_mean,
                )?;
            }
        }
        Ok(())
    }

    /// [`Self::momentum`]'s second part, after [`Self::pressures`]: the
    /// force and Coriolis into `rhs_u`, `rhs_v`.
    pub(crate) fn gradient(
        &self,
        statics: &Statics3D,
        eta: &DeviceBuffer<f64>,
        rho: &DeviceBuffer<f64>,
        u: &DeviceBuffer<f64>,
        v: &DeviceBuffer<f64>,
        rhs_u: &mut DeviceBuffer<f64>,
        rhs_v: &mut DeviceBuffer<f64>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let gpu = &self.gpu;
        let (m, st) = (&gpu.module_baroclinic, &gpu.stream);
        let s = statics;
        let nl = s.n_levels as u32;
        let [density, stratification, displacement] = self.reference_buffers();
        let none = &self.none;
        let (balanced_x, balanced_y) = match &self.balanced {
            Some([x, y]) => (x, y),
            None => (none, none),
        };
        // SAFETY: as in `reference`; `rhs_u`, `rhs_v` are none of the inputs.
        unsafe {
            let n = s.n_elements * s.n_levels;
            m.pressure_gradient(
                st,
                launch(n, LEVEL_BLOCK),
                nl,
                n as u32,
                self.min_column_depth,
                self.scale,
                self.rho0,
                eta,
                &s.bed,
                rho,
                density,
                stratification,
                displacement,
                &self.at_level,
                &s.sigma_rho,
                &s.averaged,
                &s.contra,
                &s.jinv,
                &s.dr,
                &s.ds,
                &s.mass,
                &s.w,
                &s.sub_metric,
                &s.fnode,
                &s.across,
                &s.lift,
                &self.sub_mean,
                balanced_x,
                balanced_y,
                &self.coriolis,
                u,
                v,
                rhs_u,
                rhs_v,
            )?;
        }
        Ok(())
    }

    /// `add_reference_advection` of T and S (`temp`, `salt`) by `transport`'s
    /// layer transports, at the levels of `eta` (the reference of the last
    /// [`Self::reference`]), added to `rhs_t`, `rhs_s` after their advection
    /// (`Transport3D::advect` without sources); with `sources` then the water
    /// no transport carried, as `Transport3D::advect` adds it with them.
    /// Nothing but the sources without a reference.
    pub fn advect_reference(
        &mut self,
        transport: &Transport3D,
        eta: &DeviceBuffer<f64>,
        temp: &DeviceBuffer<f64>,
        salt: &DeviceBuffer<f64>,
        rhs_t: &mut DeviceBuffer<f64>,
        rhs_s: &mut DeviceBuffer<f64>,
        sources: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let gpu = self.gpu.clone();
        let (m, st) = (&gpu.module_baroclinic, &gpu.stream);
        let s = transport.statics();
        let nl = s.n_levels as u32;
        let n = s.n_elements * s.n_levels;
        let Some(fields) = &self.fields else {
            assert!(
                !sources,
                "without a reference `Transport3D::advect` adds the sources"
            );
            return Ok(());
        };
        // SAFETY: as in `reference`; `rhs_t`, `rhs_s` are none of the
        // inputs.
        unsafe {
            m.reference_edges(
                st,
                launch(n, LEVEL_BLOCK),
                nl,
                n as u32,
                self.min_column_depth,
                &transport.kinds,
                &s.averaged,
                &s.contra,
                &s.jinv,
                &s.dr,
                &s.ds,
                &s.mass,
                &s.w,
                eta,
                &s.bed,
                &s.sigma_rho,
                &transport.hu,
                &transport.hv,
                &transport.subcell,
                &transport.omega,
                &fields.tracers[0],
                &fields.tracers[1],
                &fields.gradients[0],
                &fields.gradients[1],
                temp,
                salt,
                &mut *rhs_t,
                &mut *rhs_s,
                &mut self.made,
            )?;
            m.reference_totals(
                st,
                launch(s.n_elements, COLUMN_BLOCK),
                nl,
                s.n_elements as u32,
                self.min_column_depth,
                eta,
                &s.bed,
                &s.mass,
                &fields.density,
                &fields.stratification,
                &fields.residual,
                &s.d_sigma,
                &self.made,
                &mut self.totals,
            )?;
            m.reference_correction(
                st,
                launch(n, LEVEL_BLOCK),
                nl,
                n as u32,
                self.min_column_depth,
                &transport.kinds,
                eta,
                &s.bed,
                &fields.stratification,
                &fields.residual,
                &s.d_sigma,
                &self.totals,
                if sources {
                    &transport.volume_source
                } else {
                    &self.none
                },
                temp,
                salt,
                &mut *rhs_t,
                &mut *rhs_s,
            )?;
        }
        if sources {
            transport.pooled_sources(rhs_t, temp, false)?;
            transport.pooled_sources(rhs_s, salt, false)?;
        }
        Ok(())
    }
}

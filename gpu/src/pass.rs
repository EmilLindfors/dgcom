//! The 3D step's barotropic pass on the GPU (P2.7 Phase 4.4c2–c3,
//! `notes/gpu.md`): dgcom's `barotropic_pass` and `barotropic_pass_local`,
//! the filtered pass of the depth-mean transport `(h, hu, hv)` through the
//! step that the mode splitter runs between the slow forcing and the 3D
//! stages, and what the 3D stages take from it.
//!
//! - **Every stage** on global substeps ([`BarotropicPass3D::run`], in the
//!   order of dgcom's own stage table, `barotropic_stages`): the 2D module's RHS
//!   ([`GpuSWE2D::rhs_pass_fluxes_into`], which also keeps the subcell
//!   interfaces' mass fluxes and the nesting bands' volume sources), their
//!   mean ([`kernels_pass::mean_source`]), the rivers' discharge
//!   ([`kernels_pass::river_stage`]), then [`kernels_pass::pass_stage`], an
//!   element a thread: the stage's combination with the slow forcing `G`,
//!   the drags' depth-mean part point-implicitly, the sums of `DU_avg2` (the
//!   stage value's transport, the face nodes' F* from the 2D module's face
//!   buffer, the subcell fluxes, NaN where the stage took the volume term)
//!   and of the filter's average, then the 2D module's damping and wet/dry
//!   correction (`barotropic_stage` with `finish_stage_element`).
//! - **Every stage pass** on local time steps ([`BarotropicPass3D::run_local`]
//!   with [`LocalPass3D`], dgcom's own `MultirateStepper` over the device's
//!   state): the 2D module's RHS of the pass's elements at their levels'
//!   stage times ([`GpuSWE2D::stage_pass_where`]), the rivers'
//!   ([`kernels_pass::river_lts`]), then [`kernels_pass::pass_stage_lts`]:
//!   each element's stage from `element_stage`, the same terms with the sums
//!   weighted by the element's stage rate, and either the wet/dry correction
//!   or the substep's sum (`coarse_step`); the substeps' ends are the 2D
//!   module's `finish_where`.
//! - **After it** ([`BarotropicPass3D::finish`]): the 2D module's wet/dry
//!   correction of the average, the constant rates of `η`, `ū`, `v̄`, `Dū`,
//!   `Dv̄` over the step (`barotropic_node_rates`), the divergence of
//!   `DU_avg2` (4.0's `layer_divergence` on one level), and the elements the
//!   pass balanced only as a whole with every column's water that no flux
//!   carried (`pooled_element`).
//!
//! The filter's weights, the stage times and the rivers' discharge are the
//! host's, as the CPU evaluates them. Each kernel follows dgcom's code
//! operation for operation, so a build without multiply–add contraction
//! gives the CPU's bits (`pass3d.rs`, the gate).

// As `three_d.rs`: the device loops index arrays by the loop variable.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use std::sync::Arc;

use cuda_core::DeviceBuffer;
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;
use dgcom::SWEState2D;
use dgcom::mesh::Bathymetry2D;
use dgcom::physics::PhysicsModule;
use dgcom::solver::rhs::{Boundaries3D, MetricForm};
use dgcom::solver::{SWESolution2D, wet_dry_element};
use dgcom::source::RiverSources;
use dgcom::time::{
    BarotropicFilter, Combination, LocalTimeStepping, MultiratePass, MultirateStepper,
    ShuOsherStage, Slot, SspScheme, barotropic_node_rates, barotropic_stages, element_stage,
    pooled_element,
};
use dgcom::vertical::SigmaGrid;

use crate::hooks::State3D;
use crate::swe2d::{
    Damping, DeviceState, Gpu, GpuSWE2D, N1, NF, NN, PassFluxes, is_subcell_element, item, level_of,
};
use crate::three_d::{
    BarotropicDevice, COLUMN_BLOCK, LEVEL_BLOCK, NSUB, RiverTables, Statics3D, launch,
};

/// [`kernels_pass::pass_stage`]'s combinations (dgcom's `Combination`).
const ADVANCE: u32 = 0;
const RK3_SECOND: u32 = 1;
const RK3_THIRD: u32 = 2;
const RK43_THIRD: u32 = 3;

/// The kernels' code of a combination.
fn combination_code(combination: Combination) -> u32 {
    match combination {
        Combination::Advance => ADVANCE,
        Combination::Rk3Second => RK3_SECOND,
        Combination::Rk3Third => RK3_THIRD,
        Combination::Rk43Third => RK43_THIRD,
    }
}

// =============================================================================
// Kernels
// =============================================================================

#[cuda_module]
pub mod kernels_pass {
    use super::*;

    /// `to_transport`, a thread per node: `h = η − B` and `(h ū, h v̄)`.
    #[kernel]
    pub fn to_transport(
        n: u32,
        eta: &[f64],
        bed: &[f64],
        ubar: &[f64],
        vbar: &[f64],
        mut h: DisjointSlice<f64>,
        mut hu: DisjointSlice<f64>,
        mut hv: DisjointSlice<f64>,
    ) {
        let idx = thread::index_1d().get();
        if idx >= n as usize {
            return;
        }
        let d = eta[idx] - bed[idx];
        // SAFETY: thread idx alone writes node idx (idx < n).
        unsafe {
            *h.get_unchecked_mut(idx) = d;
            *hu.get_unchecked_mut(idx) = d * ubar[idx];
            *hv.get_unchecked_mut(idx) = d * vbar[idx];
        }
    }

    /// `column_drag_rates`, a thread per column: `Λ̄ = Σ_l λ_l Δσ_l`,
    /// summed as `f64`'s `Sum` does (from −0).
    #[kernel]
    pub fn column_drag(
        nl: u32,
        n: u32,
        d_sigma: &[f64],
        rates: &[f64],
        mut out: DisjointSlice<f64>,
    ) {
        let idx = thread::index_1d().get();
        if idx >= n as usize {
            return;
        }
        let nl = nl as usize;
        let mut sum = -0.0;
        for l in 0..nl {
            sum += rates[idx * nl + l] * d_sigma[l];
        }
        // SAFETY: thread idx alone writes column idx (idx < n).
        unsafe {
            *out.get_unchecked_mut(idx) = sum;
        }
    }

    /// The mean of the 2D module's volume sources over the pass
    /// (`mass_source += c·S_h`), a thread per node.
    #[kernel]
    pub fn mean_source(n: u32, c: f64, s: &[f64], mut mean: DisjointSlice<f64>) {
        let idx = thread::index_1d().get();
        if idx >= n as usize {
            return;
        }
        // SAFETY: thread idx alone touches node idx (idx < n).
        unsafe {
            *mean.get_unchecked_mut(idx) += c * s[idx];
        }
    }

    /// The rivers' discharge added to a stage's RHS, a thread per element
    /// with rivers (`elements`): each of its rivers in index order,
    /// `h += Q·share` at its nodes, `Q` the river's discharge at the stage's
    /// time (`discharge[first + i]`).
    #[kernel]
    pub fn river_stage(
        n: u32,
        first: u32,
        elements: &[u32],
        offsets: &[u32],
        by_element: &[u32],
        discharge: &[f64],
        share: &[f64],
        mut h: DisjointSlice<f64>,
    ) {
        let t = thread::index_1d().get();
        if t >= n as usize {
            return;
        }
        let k = elements[t] as usize;
        for j in offsets[k] as usize..offsets[k + 1] as usize {
            let i = by_element[j] as usize;
            let q = discharge[first as usize + i];
            for node in 0..NN {
                // SAFETY: thread t alone writes element k's rows (the
                // listed elements are distinct).
                unsafe {
                    *h.get_unchecked_mut(k * NN + node) += q * share[i * NN + node];
                }
            }
        }
    }

    /// `barotropic_stage` of element k with `finish_stage_element`, a
    /// thread per element: the stage value from `x` (and `qⁿ`, `q`, for the
    /// combinations that read it; the third stage of SSP-RK3 reads the old
    /// value of its target) with the rate `RHS + G` as `combination` says,
    /// the drags' depth-mean part over `dt_fraction·dt` (the bottom rate
    /// `r/h` where `bottom` is not empty, the column mean `Λ̄` where
    /// `column` is not), the sums of `DU_avg2` with weight `c` (the
    /// transport of `x`, the face nodes' F* of `faces`, the subcell fluxes
    /// `sub_mass` where `x` took the subcells, else NaN), the average with
    /// weight `average` if `has_average`, then the damping from `x` and the
    /// wet/dry correction of the new value, whose clips go to `clips`.
    #[kernel]
    pub fn pass_stage(
        n_elements: u32,
        combination: u32,
        dt_fraction: f64,
        dt: f64,
        c: f64,
        has_average: u32,
        average: f64,
        subcell_depth: f64,
        damping: Damping,
        x_h: &[f64],
        x_hu: &[f64],
        x_hv: &[f64],
        q_h: &[f64],
        q_hu: &[f64],
        q_hv: &[f64],
        rhs_h: &[f64],
        rhs_hu: &[f64],
        rhs_hv: &[f64],
        g_h: &[f64],
        g_hu: &[f64],
        g_hv: &[f64],
        bottom: &[f64],
        column: &[f64],
        faces: &[[[f64; 3]; NF]],
        element_edges: &[[u32; 4]],
        edges: &[[u32; 4]],
        sub_mass: &[f64],
        mass: &[f64],
        inv_area: &[f64],
        mut out_h: DisjointSlice<f64>,
        mut out_hu: DisjointSlice<f64>,
        mut out_hv: DisjointSlice<f64>,
        mut tr_hu: DisjointSlice<f64>,
        mut tr_hv: DisjointSlice<f64>,
        mut tr_face: DisjointSlice<f64>,
        mut tr_sub: DisjointSlice<f64>,
        mut avg_h: DisjointSlice<f64>,
        mut avg_hu: DisjointSlice<f64>,
        mut avg_hv: DisjointSlice<f64>,
        mut clips: DisjointSlice<u32>,
    ) {
        let k = thread::index_1d().get();
        if k >= n_elements as usize {
            return;
        }
        let step = dt_fraction * dt;
        // The combination as `barotropic_stage` writes it
        let combine = |x: f64, q: f64, old: f64, rate: f64| match combination {
            ADVANCE => x + step * rate,
            RK3_SECOND => q * 0.75 + 0.25 * x + 0.25 * dt * rate,
            RK3_THIRD => old * (1.0 / 3.0) + 2.0 / 3.0 * x + 2.0 / 3.0 * dt * rate,
            _ => q * (2.0 / 3.0) + 1.0 / 3.0 * x + step * rate,
        };
        let reads_q = combination == RK3_SECOND || combination == RK43_THIRD;
        let reads_old = combination == RK3_THIRD;
        let (mut eh, mut ehu, mut ehv) = ([0.0; NN], [0.0; NN], [0.0; NN]);
        #[unroll]
        for i in 0..NN {
            let m = k * NN + i;
            let (qh, qhu, qhv) = if reads_q {
                (q_h[m], q_hu[m], q_hv[m])
            } else {
                (0.0, 0.0, 0.0)
            };
            // SAFETY: thread k reads and writes only its own element's rows.
            let (oh, ohu, ohv) = if reads_old {
                unsafe {
                    (
                        *out_h.get_unchecked_mut(m),
                        *out_hu.get_unchecked_mut(m),
                        *out_hv.get_unchecked_mut(m),
                    )
                }
            } else {
                (0.0, 0.0, 0.0)
            };
            eh[i] = combine(x_h[m], qh, oh, rhs_h[m] + g_h[m]);
            ehu[i] = combine(x_hu[m], qhu, ohu, rhs_hu[m] + g_hu[m]);
            ehv[i] = combine(x_hv[m], qhv, ohv, rhs_hv[m] + g_hv[m]);
        }
        // The drags' depth-mean part, point-implicitly with the new depth
        let (has_bottom, has_column) = (!bottom.is_empty(), !column.is_empty());
        if has_bottom || has_column {
            #[unroll]
            for i in 0..NN {
                let h = eh[i];
                if h > 0.0 {
                    let m = k * NN + i;
                    let rate = (if has_bottom { bottom[m] / h } else { 0.0 })
                        + (if has_column { column[m] } else { 0.0 });
                    if rate > 0.0 {
                        let factor = 1.0 / (1.0 + step * rate);
                        ehu[i] *= factor;
                        ehv[i] *= factor;
                    }
                }
            }
        }
        // The sums: DU_avg2 and the average
        #[unroll]
        for i in 0..NN {
            let m = k * NN + i;
            // SAFETY: as above, the element's rows of the sums.
            unsafe {
                *tr_hu.get_unchecked_mut(m) += c * x_hu[m];
                *tr_hv.get_unchecked_mut(m) += c * x_hv[m];
                if has_average != 0 {
                    *avg_h.get_unchecked_mut(m) += average * x_h[m];
                    *avg_hu.get_unchecked_mut(m) += average * x_hu[m];
                    *avg_hv.get_unchecked_mut(m) += average * x_hv[m];
                }
            }
        }
        #[unroll]
        for f in 0..4 {
            let e = element_edges[k][f] as usize;
            let edge = edges[e];
            // A wall's F* is in its left (own) side
            let side = usize::from(edge[0] as usize != k || edge[1] as usize != f);
            #[unroll]
            for fi in 0..N1 {
                // SAFETY: as above, the element's face nodes.
                unsafe {
                    *tr_face.get_unchecked_mut((k * 4 + f) * N1 + fi) +=
                        c * faces[e][side * N1 + fi][0];
                }
            }
        }
        // NaN (a stage without subcells) stays NaN
        let subcells = is_subcell_element(k, x_h, subcell_depth);
        #[unroll]
        for s in 0..NSUB {
            let j = k * NSUB + s;
            // SAFETY: as above, the element's interfaces.
            unsafe {
                let sum = tr_sub.get_unchecked_mut(j);
                *sum = if subcells {
                    *sum + c * sub_mass[j]
                } else {
                    f64::NAN
                };
            }
        }
        // The 2D module's damping from `x`, then its wet/dry correction
        if damping.is_active() {
            #[unroll]
            for i in 0..NN {
                let m = k * NN + i;
                let from = SWEState2D::new(x_h[m], x_hu[m], x_hv[m]);
                damping.node(m, eh[i], &mut ehu[i], &mut ehv[i], from, step);
            }
        }
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

    /// The filter's weight of the last substep's state, `avg += w·q`, a
    /// thread per node.
    #[kernel]
    pub fn average_last(
        n: u32,
        w: f64,
        h: &[f64],
        hu: &[f64],
        hv: &[f64],
        mut avg_h: DisjointSlice<f64>,
        mut avg_hu: DisjointSlice<f64>,
        mut avg_hv: DisjointSlice<f64>,
    ) {
        let idx = thread::index_1d().get();
        if idx >= n as usize {
            return;
        }
        // SAFETY: thread idx alone touches node idx (idx < n).
        unsafe {
            *avg_h.get_unchecked_mut(idx) += w * h[idx];
            *avg_hu.get_unchecked_mut(idx) += w * hu[idx];
            *avg_hv.get_unchecked_mut(idx) += w * hv[idx];
        }
    }

    /// `barotropic_rates`, a thread per node: `barotropic_node_rates` from
    /// `(η, ū, v̄)` to the filtered transport over a step `dt`.
    #[kernel]
    pub fn pass_rates(
        n: u32,
        dt: f64,
        eta: &[f64],
        ubar: &[f64],
        vbar: &[f64],
        bed: &[f64],
        avg_h: &[f64],
        avg_hu: &[f64],
        avg_hv: &[f64],
        mut rate_eta: DisjointSlice<f64>,
        mut rate_ubar: DisjointSlice<f64>,
        mut rate_vbar: DisjointSlice<f64>,
        mut rate_hu: DisjointSlice<f64>,
        mut rate_hv: DisjointSlice<f64>,
    ) {
        let idx = thread::index_1d().get();
        if idx >= n as usize {
            return;
        }
        let r = barotropic_node_rates(
            [eta[idx], ubar[idx], vbar[idx]],
            [avg_h[idx], avg_hu[idx], avg_hv[idx]],
            bed[idx],
            dt,
        );
        // SAFETY: thread idx alone writes node idx (idx < n).
        unsafe {
            *rate_eta.get_unchecked_mut(idx) = r[0];
            *rate_ubar.get_unchecked_mut(idx) = r[1];
            *rate_vbar.get_unchecked_mut(idx) = r[2];
            *rate_hu.get_unchecked_mut(idx) = r[3];
            *rate_hv.get_unchecked_mut(idx) = r[4];
        }
    }

    /// `pooled_elements`, a thread per element: dgcom's `pooled_element`
    /// with the rivers' volume rate at each node (`RiverInflow::volume_rate`
    /// with the pass's mean discharge, summed as `f64`'s `Sum` does; none
    /// unless `has_rivers`).
    #[kernel]
    pub fn pooled_marks(
        n_elements: u32,
        dt: f64,
        has_rivers: u32,
        eta_rate: &[f64],
        div: &[f64],
        mass_source: &[f64],
        eta: &[f64],
        bed: &[f64],
        offsets: &[u32],
        by_element: &[u32],
        discharge: &[f64],
        share: &[f64],
        mut column_source: DisjointSlice<f64>,
        mut pooled: DisjointSlice<u32>,
    ) {
        let k = thread::index_1d().get();
        if k >= n_elements as usize {
            return;
        }
        let node = |i: usize| {
            let m = k * NN + i;
            [eta_rate[m], div[m], mass_source[m], eta[m], bed[m]]
        };
        let source = |i: usize| {
            if has_rivers == 0 {
                0.0
            } else {
                let mut sum = -0.0;
                for j in offsets[k] as usize..offsets[k + 1] as usize {
                    let r = by_element[j] as usize;
                    sum += discharge[r] * share[r * NN + i];
                }
                sum
            }
        };
        let mut values = [0.0; NN];
        let mark = pooled_element(NN, node, source, dt, |i, s| values[i] = s);
        // SAFETY: thread k alone writes element k's columns and mark.
        unsafe {
            for i in 0..NN {
                *column_source.get_unchecked_mut(k * NN + i) = values[i];
            }
            *pooled.get_unchecked_mut(k) = u32::from(mark);
        }
    }

    /// The rivers' discharge added to a stage pass's RHS on local time
    /// steps, a thread per element of the pass (the first `n` of `list`,
    /// every element where it is empty): each of its rivers in index order,
    /// `h += Q·share`, `Q` the river's discharge at the element's stage time
    /// (`discharge[first + i]`, the host's).
    #[kernel]
    pub fn river_lts(
        n: u32,
        list: &[u32],
        first: u32,
        offsets: &[u32],
        by_element: &[u32],
        discharge: &[f64],
        share: &[f64],
        mut h: DisjointSlice<f64>,
    ) {
        let Some(k) = item(thread::index_1d().get(), list, n) else {
            return;
        };
        for j in offsets[k] as usize..offsets[k + 1] as usize {
            let i = by_element[j] as usize;
            let q = discharge[first as usize + i];
            for node in 0..NN {
                // SAFETY: thread k alone writes element k's rows (the
                // pass's elements are distinct).
                unsafe {
                    *h.get_unchecked_mut(k * NN + node) += q * share[i * NN + node];
                }
            }
        }
    }

    /// `coarse_step`'s stage of element k (the pass's barotropic stage on
    /// local time steps), a thread per element of the pass: its stage from
    /// `element_stage` (its level, this stage's rate), the value
    /// `a·S + b·Y + c·(RHS + G)` from the rows' RHS, the drags' depth-mean
    /// part over `c`, the sums of `DU_avg2` and of the 2D module's volume
    /// sources (if `has_sources`) with weight `W_m b_s / (n_bt 2^r)`, `r`
    /// the element's rate, then either its damping and wet/dry correction
    /// (a stage the next reads) or its damping and the substep's sum into
    /// `acc` (the last stage).
    #[kernel]
    pub fn pass_stage_lts(
        n: u32,
        list: &[u32],
        t: f64,
        dt: f64,
        finest: u32,
        substep: u32,
        depth: u32,
        last: u32,
        start: f64,
        input: f64,
        beta: f64,
        c_stage: f64,
        level: &[u32],
        rate: &[u32],
        w_secondary: f64,
        b_s: f64,
        n_bt: f64,
        subcell_depth: f64,
        damping: Damping,
        has_sources: u32,
        base_h: &[f64],
        base_hu: &[f64],
        base_hv: &[f64],
        x_h: &[f64],
        x_hu: &[f64],
        x_hv: &[f64],
        g_h: &[f64],
        g_hu: &[f64],
        g_hv: &[f64],
        bottom: &[f64],
        column: &[f64],
        faces: &[[[f64; 3]; NF]],
        element_edges: &[[u32; 4]],
        edges: &[[u32; 4]],
        sub_mass: &[f64],
        sources: &[f64],
        mass: &[f64],
        inv_area: &[f64],
        mut out_h: DisjointSlice<f64>,
        mut out_hu: DisjointSlice<f64>,
        mut out_hv: DisjointSlice<f64>,
        mut tr_hu: DisjointSlice<f64>,
        mut tr_hv: DisjointSlice<f64>,
        mut tr_face: DisjointSlice<f64>,
        mut tr_sub: DisjointSlice<f64>,
        mut tr_mass: DisjointSlice<f64>,
        mut acc_h: DisjointSlice<f64>,
        mut acc_hu: DisjointSlice<f64>,
        mut acc_hv: DisjointSlice<f64>,
        mut clips: DisjointSlice<u32>,
    ) {
        let Some(k) = item(thread::index_1d().get(), list, n) else {
            return;
        };
        let r = level_of(rate, k);
        let stage = element_stage(
            (t, dt),
            finest as usize,
            substep as usize,
            depth as usize,
            ShuOsherStage {
                start,
                input,
                beta,
                c: c_stage,
            },
            last != 0,
            level_of(level, k) as u8,
            r as u8,
        );
        let c = w_secondary * b_s / (n_bt * (1u64 << r) as f64);
        // The stage value, as `SSPRK3`/`SSPRK43` combine it, from the rate
        // RHS + G in the rows
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
        let (mut eh, mut ehu, mut ehv) = ([0.0; NN], [0.0; NN], [0.0; NN]);
        #[unroll]
        for i in 0..NN {
            let m = k * NN + i;
            // SAFETY: thread k reads and writes only its own element's rows
            // (the pass's elements are distinct).
            let (fh, fhu, fhv) = unsafe {
                (
                    *out_h.get_unchecked_mut(m),
                    *out_hu.get_unchecked_mut(m),
                    *out_hv.get_unchecked_mut(m),
                )
            };
            eh[i] = combine(base_h[m], x_h[m], fh + g_h[m]);
            ehu[i] = combine(base_hu[m], x_hu[m], fhu + g_hu[m]);
            ehv[i] = combine(base_hv[m], x_hv[m], fhv + g_hv[m]);
        }
        // The drags' depth-mean part, point-implicitly with the new depth
        let (has_bottom, has_column) = (!bottom.is_empty(), !column.is_empty());
        if has_bottom || has_column {
            #[unroll]
            for i in 0..NN {
                let h = eh[i];
                if h > 0.0 {
                    let m = k * NN + i;
                    let rate = (if has_bottom { bottom[m] / h } else { 0.0 })
                        + (if has_column { column[m] } else { 0.0 });
                    if rate > 0.0 {
                        let factor = 1.0 / (1.0 + stage.c * rate);
                        ehu[i] *= factor;
                        ehv[i] *= factor;
                    }
                }
            }
        }
        // The sums of DU_avg2 and of the volume sources
        #[unroll]
        for i in 0..NN {
            let m = k * NN + i;
            // SAFETY: as above, the element's rows of the sums.
            unsafe {
                *tr_hu.get_unchecked_mut(m) += c * x_hu[m];
                *tr_hv.get_unchecked_mut(m) += c * x_hv[m];
                if has_sources != 0 {
                    *tr_mass.get_unchecked_mut(m) += c * sources[m];
                }
            }
        }
        #[unroll]
        for f in 0..4 {
            let e = element_edges[k][f] as usize;
            let edge = edges[e];
            // A wall's F* is in its left (own) side
            let side = usize::from(edge[0] as usize != k || edge[1] as usize != f);
            #[unroll]
            for fi in 0..N1 {
                // SAFETY: as above, the element's face nodes.
                unsafe {
                    *tr_face.get_unchecked_mut((k * 4 + f) * N1 + fi) +=
                        c * faces[e][side * N1 + fi][0];
                }
            }
        }
        // NaN (a stage without subcells) stays NaN
        let subcells = is_subcell_element(k, x_h, subcell_depth);
        #[unroll]
        for s in 0..NSUB {
            let j = k * NSUB + s;
            // SAFETY: as above, the element's interfaces.
            unsafe {
                let sum = tr_sub.get_unchecked_mut(j);
                *sum = if subcells {
                    *sum + c * sub_mass[j]
                } else {
                    f64::NAN
                };
            }
        }
        // The 2D module's damping from the stage's input
        if damping.is_active() {
            #[unroll]
            for i in 0..NN {
                let m = k * NN + i;
                let from = SWEState2D::new(x_h[m], x_hu[m], x_hv[m]);
                damping.node(m, eh[i], &mut ehu[i], &mut ehv[i], from, stage.c);
            }
        }
        match stage.accumulate {
            // The last stage: the substep's sum
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
            // A stage the next reads: the wet/dry correction
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
}

// =============================================================================
// Host side
// =============================================================================

/// A barotropic pass on global substeps on the device (dgcom's
/// `GlobalPass`).
#[derive(Clone, Copy)]
pub struct DevicePass<'a> {
    /// The filter: `n_bt` substeps per baroclinic step, the window's weights.
    pub filter: &'a BarotropicFilter,
    /// The substeps' scheme.
    pub scheme: SspScheme,
    /// The pass's start `tⁿ` and its substep `Δt/n_bt` (s).
    pub t: f64,
    pub dt_bt: f64,
    /// The step average of the slow forcing, `(0, G_hu, G_hv)`.
    pub forcing: &'a DeviceState,
    /// The bottom-drag rate `r` of every column, if any.
    pub bottom: Option<&'a DeviceBuffer<f64>>,
    /// The column mean `Λ̄` of the layer drag, if any
    /// ([`BarotropicPass3D::column_drag`]).
    pub column: Option<&'a DeviceBuffer<f64>>,
    /// The rivers (their discharge series; the tables are the pass's).
    pub rivers: Option<&'a RiverSources>,
}

/// The barotropic pass on local time steps on the device (dgcom's
/// `LocalPass`): the levels and the stage buffers of dgcom's own
/// `MultirateStepper::schedule`.
pub struct LocalPass3D {
    scheme: SspScheme,
    stepper: MultirateStepper<DeviceState>,
    element_dt: Vec<f64>,
    values: Vec<DeviceState>,
    last: DeviceState,
    acc: DeviceState,
}

impl LocalPass3D {
    /// Up to `max_levels` levels of `scheme`'s substeps, stage buffers like
    /// `like`.
    pub fn new(max_levels: usize, scheme: SspScheme, like: &DeviceState) -> Self {
        Self {
            scheme,
            stepper: MultirateStepper::new(max_levels, scheme),
            element_dt: Vec::new(),
            values: (0..scheme.n_stages() - 1).map(|_| like.clone()).collect(),
            last: like.clone(),
            acc: like.clone(),
        }
    }

    /// `LocalPass::assign_levels` from `q`'s element steps under `module`
    /// (on the device) at the barotropic CFL number `cfl`; the coarse
    /// substep, at most `dt_cap`.
    pub fn assign_levels(
        &mut self,
        module: &GpuSWE2D,
        q: &DeviceState,
        cfl: f64,
        dt_cap: f64,
    ) -> f64 {
        let mesh = module.mesh();
        self.element_dt.resize(mesh.n_elements, 0.0);
        module.element_dt(q, cfl, self.scheme, &mut self.element_dt);
        self.stepper.assign_levels_from(
            &self.element_dt,
            mesh,
            &[module.stencil()],
            Some(dt_cap),
            None,
        )
    }

    /// The levels, the stage rates and the work counts.
    pub fn stepper(&self) -> &MultirateStepper<DeviceState> {
        &self.stepper
    }
}

/// The barotropic pass of the 3D model on the GPU (see the module docs): its
/// work buffers and what it leaves for the 3D stages.
pub struct BarotropicPass3D {
    gpu: Arc<Gpu>,
    /// The mesh under one σ-level, in the 2D module's metric form: the
    /// divergence of `DU_avg2`
    statics: Statics3D,
    u1: DeviceState,
    u2: DeviceState,
    k_2d: DeviceState,
    /// A stage's subcell fluxes and the 2D module's volume sources
    sub_mass: DeviceBuffer<f64>,
    sources: DeviceState,
    rivers: Option<RiverTables>,
    /// Every stage's discharge of every river, `stage·n_rivers + i`, and
    /// their means ([`Self::discharge`])
    discharges: DeviceBuffer<f64>,
    mean_discharge: DeviceBuffer<f64>,
    kinds: DeviceBuffer<u32>,
    thin: DeviceBuffer<u32>,
    none: DeviceBuffer<f64>,
    none_u32: DeviceBuffer<u32>,
    /// The filtered transport
    pub q_avg: DeviceState,
    /// `DU_avg2` (its face and subcell fluxes), `∂η/∂t` and the elements
    /// balanced only as a whole, as the 3D transport takes them (its
    /// `euler_dt` the 3D stages', the driver's to set)
    pub flux: BarotropicDevice,
    /// The 2D module's volume sources averaged with the transport's weights
    pub mass_source: DeviceBuffer<f64>,
    /// Every river's discharge averaged so (on the host, as the CPU sums
    /// it)
    pub discharge: Vec<f64>,
    /// The constant rates of `ū`, `v̄`, `Dū`, `Dv̄` over the step (`η`'s is
    /// `flux.eta_rate`)
    pub rates: [DeviceBuffer<f64>; 4],
    /// The divergence of `DU_avg2`
    pub divergence: DeviceBuffer<f64>,
    /// Every column's water that no flux of the pass carried (m/s)
    pub column_source: DeviceBuffer<f64>,
}

/// The stage's input `x`, its target and `qⁿ` among the pass's buffers
/// (dgcom's `stage_buffers`).
fn stage_buffers<'a>(
    x: Slot,
    target: Slot,
    q: &'a mut DeviceState,
    u1: &'a mut DeviceState,
    u2: &'a mut DeviceState,
) -> (
    &'a DeviceState,
    &'a mut DeviceState,
    Option<&'a DeviceState>,
) {
    match (x, target) {
        (Slot::Q, Slot::U1) => (&*q, u1, None),
        (Slot::U1, Slot::U2) => (&*u1, u2, Some(&*q)),
        (Slot::U2, Slot::Q) => (&*u2, q, None),
        (Slot::U2, Slot::U1) => (&*u2, u1, Some(&*q)),
        (Slot::U1, Slot::Q) => (&*u1, q, None),
        (x, target) => unreachable!("no barotropic stage writes {target:?} from {x:?}"),
    }
}

impl BarotropicPass3D {
    /// For the 2D module `module` over the bed `bathymetry`, its volume
    /// form `metric` (`BarotropicPhysics::metric_form`), and the 3D model's
    /// rivers, if any.
    pub fn new(
        gpu: &Arc<Gpu>,
        module: &GpuSWE2D,
        bathymetry: &Bathymetry2D,
        metric: MetricForm,
        rivers: Option<&RiverSources>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let st = &gpu.stream;
        let (mesh, ops, geom) = (module.mesh(), module.operators(), module.geometry());
        let ne = mesh.n_elements;
        let n = ne * NN;
        let statics = Statics3D::upload(
            gpu,
            mesh,
            ops,
            geom,
            bathymetry,
            &SigmaGrid::uniform(1),
            &Boundaries3D::default(),
            metric,
        )?;
        let zeros = |n: usize| DeviceBuffer::<f64>::zeroed(st, n);
        let state = || DeviceState::upload(gpu, &SWESolution2D::new(ne, NN));
        Ok(Self {
            u1: state(),
            u2: state(),
            k_2d: state(),
            sources: state(),
            q_avg: state(),
            sub_mass: zeros(ne * NSUB)?,
            rivers: rivers.map(|r| RiverTables::new(gpu, ne, r)).transpose()?,
            discharges: zeros(0)?,
            mean_discharge: zeros(0)?,
            kinds: DeviceBuffer::zeroed(st, ne)?,
            thin: DeviceBuffer::zeroed(st, n)?,
            none: zeros(0)?,
            none_u32: DeviceBuffer::zeroed(st, 0)?,
            flux: BarotropicDevice {
                hu: zeros(n)?,
                hv: zeros(n)?,
                face: zeros(ne * 4 * N1)?,
                subcell: zeros(ne * NSUB)?,
                eta_rate: zeros(n)?,
                euler_dt: zeros(ne)?,
                pooled: DeviceBuffer::zeroed(st, ne)?,
            },
            mass_source: zeros(n)?,
            discharge: vec![0.0; rivers.map_or(0, RiverSources::len)],
            rates: [zeros(n)?, zeros(n)?, zeros(n)?, zeros(n)?],
            divergence: zeros(n)?,
            column_source: zeros(n)?,
            statics,
            gpu: gpu.clone(),
        })
    }

    /// `to_transport` of `state` into `q`.
    pub fn to_transport(
        &self,
        state: &State3D,
        q: &mut DeviceState,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let (gpu, s) = (&self.gpu, &self.statics);
        let n = s.n_elements * NN;
        // SAFETY: every buffer holds the mesh's nodes.
        unsafe {
            gpu.module_pass.to_transport(
                &gpu.stream,
                launch(n, COLUMN_BLOCK),
                n as u32,
                &state.eta,
                &s.bed,
                &state.ubar,
                &state.vbar,
                &mut q.h,
                &mut q.hu,
                &mut q.hv,
            )?;
        }
        Ok(())
    }

    /// `column_drag_rates` of the layer-drag rates `rates` (`[element]
    /// [node][level]` of `d_sigma`'s levels) into `out` (a column each).
    pub fn column_drag(
        &self,
        d_sigma: &DeviceBuffer<f64>,
        rates: &DeviceBuffer<f64>,
        out: &mut DeviceBuffer<f64>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let gpu = &self.gpu;
        let n = self.statics.n_elements * NN;
        assert_eq!(out.len(), n, "a column mean per column");
        assert_eq!(rates.len(), n * d_sigma.len(), "a rate per layer");
        // SAFETY: `rates` holds `d_sigma.len()` levels of every column.
        unsafe {
            gpu.module_pass.column_drag(
                &gpu.stream,
                launch(n, COLUMN_BLOCK),
                d_sigma.len() as u32,
                n as u32,
                d_sigma,
                rates,
                out,
            )?;
        }
        Ok(())
    }

    /// The pass's sums from zero: the average, `DU_avg2`, the volume sources
    /// and the rivers' discharge.
    fn clear_sums(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let st = &self.gpu.stream;
        for field in [&mut self.q_avg.h, &mut self.q_avg.hu, &mut self.q_avg.hv] {
            field.zero_async(st)?;
        }
        let f = &mut self.flux;
        for field in [&mut f.hu, &mut f.hv, &mut f.face, &mut f.subcell] {
            field.zero_async(st)?;
        }
        self.mass_source.zero_async(st)?;
        self.discharge.fill(0.0);
        Ok(())
    }

    /// dgcom's `barotropic_pass_local` of `q` with `module` and `local`'s
    /// levels ([`LocalPass3D::assign_levels`]): `q`, [`Self::q_avg`],
    /// [`Self::flux`]'s `DU_avg2`, [`Self::mass_source`] and
    /// [`Self::discharge`] end as [`Self::run`] leaves them.
    pub fn run_local(
        &mut self,
        module: &GpuSWE2D,
        pass: DevicePass<'_>,
        local: &mut LocalPass3D,
        q: &mut DeviceState,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let gpu = self.gpu.clone();
        let (m, st) = (&gpu.module_pass, &gpu.stream);
        let DevicePass {
            filter,
            scheme,
            t,
            dt_bt,
            forcing,
            bottom,
            column,
            rivers,
        } = pass;
        assert_eq!(scheme, local.scheme, "the levels are the pass's scheme's");
        let n = self.statics.n_elements * NN;
        let n_bt = filter.n_bt();
        let stages = barotropic_stages(scheme);
        let n_stages = stages.len();
        self.clear_sums()?;
        module.begin_coarse_step(local.stepper.levels(), local.stepper.stage_rates());
        // The rivers' discharge at their elements' stage times in every
        // pass, and their means, on the host in the CPU's order: a river's
        // term where its element is in the pass (its stage's rate at least
        // the pass's depth), else zero
        let n_rivers = self.discharge.len();
        let rivers = rivers.filter(|_| self.rivers.is_some());
        if let Some(rivers) = rivers {
            let mut all = Vec::new();
            let means = &mut self.discharge;
            for (mi, &w_secondary) in filter.secondary_weights().iter().enumerate() {
                let t_m = t + mi as f64 * dt_bt;
                local.stepper.schedule(t_m, dt_bt, |p| {
                    let MultiratePass::Stage { stage: s, plan, .. } = p else {
                        return;
                    };
                    let b_s = stages[s].weight;
                    for (i, mean) in means.iter_mut().enumerate() {
                        let k = rivers.element(i).as_usize();
                        let discharge = rivers.discharge(i, plan.at(k).time);
                        all.push(discharge);
                        let r = plan.rate[k];
                        let term = if r as usize >= plan.depth {
                            let c = w_secondary * b_s / (n_bt as f64 * (1u64 << r) as f64);
                            c * discharge
                        } else {
                            0.0
                        };
                        *mean += term;
                    }
                });
            }
            self.discharges = DeviceBuffer::from_host(st, &all)?;
        }
        if self.rivers.is_some() {
            self.mean_discharge = DeviceBuffer::from_host(st, &self.discharge)?;
        }
        let has_sources = u32::from(module.has_mass_sources());
        let tables = module.pass_tables();
        let (subcell_depth, damping) = (tables.subcell_depth, tables.damping);
        drop(tables);
        let Self {
            sub_mass,
            sources,
            rivers: river_tables,
            discharges,
            none,
            flux,
            mass_source,
            q_avg,
            ..
        } = self;
        let none: &DeviceBuffer<f64> = none;
        let (bottom, column) = (bottom.unwrap_or(none), column.unwrap_or(none));
        let discharges: &DeviceBuffer<f64> = discharges;
        let river_tables = river_tables.as_ref().filter(|_| rivers.is_some());
        let LocalPass3D {
            stepper,
            values,
            last,
            acc,
            ..
        } = local;
        let mut pass_index = 0;
        let mut failure: Option<Box<dyn std::error::Error>> = None;
        for (mi, &w_secondary) in filter.secondary_weights().iter().enumerate() {
            let t_m = t + mi as f64 * dt_bt;
            stepper.schedule(t_m, dt_bt, |p| match p {
                MultiratePass::Stage {
                    stage: s,
                    elements,
                    plan,
                } => {
                    let (done, rest) = values.split_at_mut(s);
                    let input: &DeviceState = if s == 0 { &*q } else { &done[s - 1] };
                    let out: &mut DeviceState = if s + 1 < n_stages {
                        &mut rest[0]
                    } else {
                        &mut *last
                    };
                    let first = (pass_index * n_rivers) as u32;
                    pass_index += 1;
                    let base: &DeviceState = q;
                    let fluxes = PassFluxes {
                        subcell: &mut *sub_mass,
                        sources: &mut *sources,
                    };
                    let b_s = stages[s].weight;
                    module.stage_pass_where(
                        input,
                        elements,
                        plan,
                        out,
                        fluxes,
                        |lists, out, fluxes| {
                            let mut launch_all = || -> Result<(), Box<dyn std::error::Error>> {
                                // SAFETY: every buffer is sized for this mesh (or
                                // the rivers' passes); the pass's elements are
                                // distinct; `out` and `acc` are neither `base`
                                // nor `input` (the stepper's buffers).
                                unsafe {
                                    if let Some(r) = river_tables {
                                        m.river_lts(
                                            st,
                                            launch(lists.n, COLUMN_BLOCK),
                                            lists.n as u32,
                                            lists.list,
                                            first,
                                            &r.offsets,
                                            &r.by_element,
                                            discharges,
                                            &r.share,
                                            &mut out.h,
                                        )?;
                                    }
                                    let mut tables = module.pass_tables();
                                    let w = plan.weights;
                                    m.pass_stage_lts(
                                        st,
                                        launch(lists.n, COLUMN_BLOCK),
                                        lists.n as u32,
                                        lists.list,
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
                                        lists.level,
                                        lists.rate,
                                        w_secondary,
                                        b_s,
                                        n_bt as f64,
                                        subcell_depth,
                                        damping,
                                        has_sources,
                                        &base.h,
                                        &base.hu,
                                        &base.hv,
                                        &input.h,
                                        &input.hu,
                                        &input.hv,
                                        &forcing.h,
                                        &forcing.hu,
                                        &forcing.hv,
                                        bottom,
                                        column,
                                        &tables.faces,
                                        tables.element_edges,
                                        tables.edges,
                                        fluxes.subcell,
                                        &fluxes.sources.h,
                                        tables.mass,
                                        tables.inv_area,
                                        &mut out.h,
                                        &mut out.hu,
                                        &mut out.hv,
                                        &mut flux.hu,
                                        &mut flux.hv,
                                        &mut flux.face,
                                        &mut flux.subcell,
                                        &mut *mass_source,
                                        &mut acc.h,
                                        &mut acc.hu,
                                        &mut acc.hv,
                                        &mut tables.clips,
                                    )?;
                                }
                                Ok(())
                            };
                            if let Err(e) = launch_all() {
                                failure.get_or_insert(e);
                            }
                        },
                    );
                }
                MultiratePass::Finish { elements, depth } => {
                    module.finish_where(q, acc, elements, depth)
                }
            });
            stepper.count_step(dt_bt);
            if let Some(e) = failure.take() {
                return Err(e);
            }
            let avg = &mut *q_avg;
            // SAFETY: every buffer holds the mesh's nodes.
            unsafe {
                m.average_last(
                    st,
                    launch(n, COLUMN_BLOCK),
                    n as u32,
                    filter.weights()[mi],
                    &q.h,
                    &q.hu,
                    &q.hv,
                    &mut avg.h,
                    &mut avg.hu,
                    &mut avg.hv,
                )?;
            }
        }
        Ok(())
    }

    /// dgcom's `barotropic_pass` of `q` with `module`: `q` ends as the last
    /// substep's state, [`Self::q_avg`] as the filtered average (without the
    /// 2D module's wet/dry correction, [`Self::finish`]'s), [`Self::flux`]'s
    /// `DU_avg2`, [`Self::mass_source`] and [`Self::discharge`] as the
    /// pass's sums.
    pub fn run(
        &mut self,
        module: &GpuSWE2D,
        pass: DevicePass<'_>,
        q: &mut DeviceState,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let gpu = self.gpu.clone();
        let (m, st) = (&gpu.module_pass, &gpu.stream);
        let DevicePass {
            filter,
            scheme,
            t,
            dt_bt,
            forcing,
            bottom,
            column,
            rivers,
        } = pass;
        let ne = self.statics.n_elements;
        let n = ne * NN;
        let n_bt = filter.n_bt();
        let stages = barotropic_stages(scheme);
        self.clear_sums()?;
        // The rivers' discharge at every stage time and their means, on the
        // host in the CPU's order; the stages' values uploaded at once (and
        // zero means for a pass without them)
        let n_rivers = self.discharge.len();
        self.discharge.fill(0.0);
        let rivers = rivers.filter(|_| self.rivers.is_some());
        if let Some(rivers) = rivers {
            let mut all = Vec::with_capacity(filter.weights().len() * stages.len() * n_rivers);
            for (mi, &w_secondary) in filter.secondary_weights().iter().enumerate() {
                let t_m = t + mi as f64 * dt_bt;
                for stage in stages {
                    let time = t_m + stage.time * dt_bt;
                    let c = w_secondary * stage.weight / n_bt as f64;
                    for (i, mean) in self.discharge.iter_mut().enumerate() {
                        let discharge = rivers.discharge(i, time);
                        all.push(discharge);
                        *mean += c * discharge;
                    }
                }
            }
            self.discharges = DeviceBuffer::from_host(st, &all)?;
        }
        if self.rivers.is_some() {
            self.mean_discharge = DeviceBuffer::from_host(st, &self.discharge)?;
        }
        let none = &self.none;
        let (bottom, column) = (bottom.unwrap_or(none), column.unwrap_or(none));
        let mass_sources = module.has_mass_sources();
        // The filter's weight of the state the next substep starts from
        let mut pending_weight = None;
        let mut stage_index = 0;
        for (mi, &w_secondary) in filter.secondary_weights().iter().enumerate() {
            let t_m = t + mi as f64 * dt_bt;
            for stage in stages {
                let time = t_m + stage.time * dt_bt;
                let (x, target, start) =
                    stage_buffers(stage.x, stage.target, q, &mut self.u1, &mut self.u2);
                module.rhs_pass_fluxes_into(
                    x,
                    time,
                    &mut self.k_2d,
                    PassFluxes {
                        subcell: &mut self.sub_mass,
                        sources: &mut self.sources,
                    },
                );
                let c = w_secondary * stage.weight / n_bt as f64;
                // SAFETY: every buffer holds the mesh's nodes (or the
                // rivers' stages); no output is an input.
                unsafe {
                    if mass_sources {
                        m.mean_source(
                            st,
                            launch(n, COLUMN_BLOCK),
                            n as u32,
                            c,
                            &self.sources.h,
                            &mut self.mass_source,
                        )?;
                    }
                    if let Some(tables) = self.rivers.as_ref().filter(|_| rivers.is_some())
                        && tables.n_elements_with > 0
                    {
                        m.river_stage(
                            st,
                            launch(tables.n_elements_with, COLUMN_BLOCK),
                            tables.n_elements_with as u32,
                            (stage_index * n_rivers) as u32,
                            &tables.elements,
                            &tables.offsets,
                            &tables.by_element,
                            &self.discharges,
                            &tables.share,
                            &mut self.k_2d.h,
                        )?;
                    }
                }
                let average = pending_weight.take();
                let mut tables = module.pass_tables();
                let start = start.unwrap_or(x);
                let f = &mut self.flux;
                let avg = &mut self.q_avg;
                // SAFETY: as above; the stage's target is neither its input
                // nor `qⁿ` (dgcom's stage table).
                unsafe {
                    m.pass_stage(
                        st,
                        launch(ne, COLUMN_BLOCK),
                        ne as u32,
                        combination_code(stage.combination),
                        stage.dt_fraction,
                        dt_bt,
                        c,
                        u32::from(average.is_some()),
                        average.unwrap_or(0.0),
                        tables.subcell_depth,
                        tables.damping,
                        &x.h,
                        &x.hu,
                        &x.hv,
                        &start.h,
                        &start.hu,
                        &start.hv,
                        &self.k_2d.h,
                        &self.k_2d.hu,
                        &self.k_2d.hv,
                        &forcing.h,
                        &forcing.hu,
                        &forcing.hv,
                        bottom,
                        column,
                        &tables.faces,
                        tables.element_edges,
                        tables.edges,
                        &self.sub_mass,
                        tables.mass,
                        tables.inv_area,
                        &mut target.h,
                        &mut target.hu,
                        &mut target.hv,
                        &mut f.hu,
                        &mut f.hv,
                        &mut f.face,
                        &mut f.subcell,
                        &mut avg.h,
                        &mut avg.hu,
                        &mut avg.hv,
                        &mut tables.clips,
                    )?;
                }
                stage_index += 1;
            }
            pending_weight = Some(filter.weights()[mi]);
        }
        if let Some(w) = pending_weight {
            let avg = &mut self.q_avg;
            // SAFETY: as above.
            unsafe {
                m.average_last(
                    st,
                    launch(n, COLUMN_BLOCK),
                    n as u32,
                    w,
                    &q.h,
                    &q.hu,
                    &q.hv,
                    &mut avg.h,
                    &mut avg.hu,
                    &mut avg.hv,
                )?;
            }
        }
        Ok(())
    }

    /// What follows the pass in the step, from the state at the step's
    /// start `state` and its length `dt`: the 2D module's wet/dry correction
    /// of [`Self::q_avg`], the constant rates ([`Self::rates`],
    /// `flux.eta_rate`), the divergence of `DU_avg2`, and
    /// [`Self::column_source`] with `flux.pooled`.
    pub fn finish(
        &mut self,
        module: &GpuSWE2D,
        state: &State3D,
        dt: f64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // The early weights are negative: the average's positivity
        module.post_process(&mut self.q_avg);
        let gpu = self.gpu.clone();
        let (m, m3, st) = (&gpu.module_pass, &gpu.module3d, &gpu.stream);
        let s = &self.statics;
        let ne = s.n_elements;
        let n = ne * NN;
        let [rate_ubar, rate_vbar, rate_hu, rate_hv] = &mut self.rates;
        let f = &mut self.flux;
        let (offsets, by_element, share) = match &self.rivers {
            Some(r) => (&r.offsets, &r.by_element, &r.share),
            None => (&self.none_u32, &self.none_u32, &self.none),
        };
        // SAFETY: every buffer is sized for this mesh (one level); no output
        // is an input.
        unsafe {
            m.pass_rates(
                st,
                launch(n, COLUMN_BLOCK),
                n as u32,
                dt,
                &state.eta,
                &state.ubar,
                &state.vbar,
                &s.bed,
                &self.q_avg.h,
                &self.q_avg.hu,
                &self.q_avg.hv,
                &mut f.eta_rate,
                rate_ubar,
                rate_vbar,
                rate_hu,
                rate_hv,
            )?;
            // The divergence: the subcell form where every stage took the
            // subcells (no column is thinner than −∞)
            m3.layer_kinds(
                st,
                launch(ne, COLUMN_BLOCK),
                ne as u32,
                f64::NEG_INFINITY,
                &state.eta,
                &s.bed,
                &f.subcell,
                &self.none_u32,
                &mut self.kinds,
                &mut self.thin,
            )?;
            m3.layer_divergence(
                st,
                launch(ne, LEVEL_BLOCK),
                1,
                ne as u32,
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
                &f.hu,
                &f.hv,
                &f.face,
                &f.subcell,
                &mut self.divergence,
            )?;
            m.pooled_marks(
                st,
                launch(ne, COLUMN_BLOCK),
                ne as u32,
                dt,
                u32::from(self.rivers.is_some()),
                &f.eta_rate,
                &self.divergence,
                &self.mass_source,
                &state.eta,
                &s.bed,
                offsets,
                by_element,
                &self.mean_discharge,
                share,
                &mut self.column_source,
                &mut f.pooled,
            )?;
        }
        Ok(())
    }
}

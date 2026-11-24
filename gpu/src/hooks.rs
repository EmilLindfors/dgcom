//! The mode splitter's 3D stage values on the GPU (P2.7 Phase 4.3a,
//! `notes/gpu.md`): what `ModeSplitIntegrator::step` does to a stage's
//! inventories to make the values the next stage's RHS reads, and the hooks
//! of `Hydrostatic3D` it calls there:
//!
//! - inventories ↔ values: dgcom's `to_inventory`, `from_inventory_after`
//!   (the last values where a node has no water, the stash's velocities in
//!   the masked columns), `inventory_of`, `dry_inventories`;
//! - the step's merged cells (`MergedCells`): their groups, found once per
//!   step from the pass's water on the device ([`Merged3D::find`], Phase
//!   4.4d: dgcom's `find_element`, a thread per element; the flushed
//!   elements put in the CPU's order and in waves on the host) or uploaded
//!   from the host's `MergedCells::find` ([`Merged3D::upload`]), set to
//!   their means at the step's start (`MergedCells::start`), converted as
//!   one cell (`convert_element`), and the flushed elements' values passed
//!   through (`pass_through`);
//! - the thin columns at rest in a stage's values and back
//!   (`Hydrostatic3D::mask_stage` with `StageStash::mask`, `forget`,
//!   `restore_into`), their momentum tendencies zeroed
//!   (`zero_thin_momentum`);
//! - the tracer limiter of `post_stage` (Phase 4.3b, `limiter.rs`), then
//!   the density of the linear equation of state (`update_density_on`;
//!   `end_density` alone at the step's end);
//! - the nesting band's relaxation of the shear and the tracers to the
//!   parent's columns (`NestingColumns::relax_shear`, `relax_tracers`,
//!   Phase 4.3c): the band's slots static ([`Band3D`]), the parent's columns
//!   at them evaluated on the host and uploaded as they change.
//!
//! Each kernel follows its CPU loop operation for operation and calls
//! dgcom's own pointwise functions (`layer_volume_of`, `layer_thickness_of`,
//! `dry_volume_of`, `LinearEOS::compute_density`), so a build without
//! multiply–add contraction (`--no-fmad`) gives the CPU's bits
//! (`stage3d.rs`, the gate). The fields keep the CPU's layout,
//! `[element][node][cell]`; the pointwise kernels take a thread per cell
//! (consecutive threads on consecutive levels), the groups' a thread per
//! element with a group and cell, the flushed elements' a thread per cell
//! (their order matters: a flushed element reads the values of a flushed
//! neighbour converted before it).

// As `three_d.rs`: the device loops index arrays by the loop variable.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use std::sync::Arc;

use cuda_core::DeviceBuffer;
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32};
use cuda_device::{DisjointSlice, kernel, thread, warp};
use cuda_host::cuda_module;
use dgcom::boundary::{NestingSlots, Supplied};
use dgcom::physics::{EquationOfState, LinearEOS};
use dgcom::solver::rhs::merged_cells::dry_volume_of;
use dgcom::solver::rhs::transport_3d::{DRY_DEPTH, layer_thickness_of, layer_volume_of};
use dgcom::solver::rhs::{CellWater, MergedTables, face_volume_rate, find_element};
use dgcom::solver::state::Solution3D;
use dgcom::vertical::depth_average;

use crate::limiter::Limiter3D;
use crate::swe2d::{Gpu, N1, NN, face_node};
use crate::three_d::{COLUMN_BLOCK, LEVEL_BLOCK, NSUB, OPEN, SLOTS, Statics3D, WALL, launch};

// =============================================================================
// Kernels
// =============================================================================

#[cuda_module]
pub mod kernels_hooks {
    use super::*;

    /// `to_inventory`, one thread per cell: the concentration (or
    /// velocity) times its cell's volume `D·Δσ` under `eta`, zero without
    /// water (`cells` the σ-thicknesses of a column's cells, `nc` of them).
    #[kernel]
    pub fn to_inventory(
        nc: u32,
        n: u32,
        eta: &[f64],
        bed: &[f64],
        cells: &[f64],
        mut field: DisjointSlice<f64>,
    ) {
        let t = thread::index_1d().get();
        if t >= n as usize {
            return;
        }
        let nc = nc as usize;
        let (col, c) = (t / nc, t % nc);
        // SAFETY: thread t alone touches cell t.
        unsafe {
            *field.get_unchecked_mut(t) *= layer_volume_of(eta[col] - bed[col], cells[c]);
        }
    }

    /// `from_inventory_after`'s nodes, one thread per cell: the inventory
    /// `q` over its cell's thickness under `eta`; at a node shallower than
    /// `DRY_DEPTH`, or carried with its group (`joined`, empty for none),
    /// the last value: `last`'s (empty: `out`'s own), or the stash's `kept`
    /// in a `masked` column (empty: none).
    #[kernel]
    pub fn from_inventory(
        nc: u32,
        n: u32,
        eta: &[f64],
        bed: &[f64],
        cells: &[f64],
        q: &[f64],
        last: &[f64],
        masked: &[u32],
        kept: &[f64],
        joined: &[u32],
        mut out: DisjointSlice<f64>,
    ) {
        let t = thread::index_1d().get();
        if t >= n as usize {
            return;
        }
        let nc = nc as usize;
        let (col, c) = (t / nc, t % nc);
        let depth = eta[col] - bed[col];
        let carried = !joined.is_empty() && joined[col] != 0;
        // SAFETY: thread t alone touches cell t.
        unsafe {
            let out = out.get_unchecked_mut(t);
            *out = if depth < DRY_DEPTH || carried {
                if !masked.is_empty() && masked[col] != 0 {
                    kept[t]
                } else if !last.is_empty() {
                    last[t]
                } else {
                    *out
                }
            } else {
                q[t] / layer_thickness_of(depth, cells[c])
            };
        }
    }

    /// `MergedCells::convert_element`, one thread per element with a group
    /// (`listed`) and cell: every group of two or more (not of a flushed
    /// element) takes its inventory over its volume, if it holds more than
    /// `dry_volume_of`'s.
    #[kernel]
    pub fn merged_convert(
        nc: u32,
        n: u32,
        listed: &[u32],
        group: &[u32],
        joined: &[u32],
        flushed: &[u32],
        mass: &[f64],
        eta: &[f64],
        bed: &[f64],
        cells: &[f64],
        q: &[f64],
        mut out: DisjointSlice<f64>,
    ) {
        let t = thread::index_1d().get();
        if t >= n as usize {
            return;
        }
        let nc = nc as usize;
        let (j, c) = (t / nc, t % nc);
        let k = listed[j] as usize;
        if flushed[k] != 0 {
            return;
        }
        let dc = cells[c];
        // `MergedCells::for_each_group`: each group by its first node
        for root in 0..NN {
            let r = k * NN + root;
            if group[r] as usize != root || joined[r] == 0 {
                continue;
            }
            let (mut inventory, mut volume, mut area) = (0.0, 0.0, -0.0);
            for i in 0..NN {
                let idx = k * NN + i;
                if group[idx] as usize != root {
                    continue;
                }
                let m = mass[idx];
                inventory += m * q[idx * nc + c];
                volume += m * layer_volume_of(eta[idx] - bed[idx], dc);
                area += m;
            }
            if volume > dry_volume_of(area, dc) {
                for i in 0..NN {
                    let idx = k * NN + i;
                    if group[idx] as usize == root {
                        // SAFETY: thread (j, c) alone touches cell c of
                        // element k's nodes.
                        unsafe {
                            *out.get_unchecked_mut(idx * nc + c) = inventory / volume;
                        }
                    }
                }
            }
        }
    }

    /// `MergedCells::start`, one thread per element with a group and cell:
    /// every group's members (not a flushed element's) to their mean of
    /// `field` weighted by their volumes under `eta`, if they hold more than
    /// `dry_volume_of`'s.
    #[kernel]
    pub fn merged_start(
        nc: u32,
        n: u32,
        listed: &[u32],
        group: &[u32],
        joined: &[u32],
        flushed: &[u32],
        mass: &[f64],
        eta: &[f64],
        bed: &[f64],
        cells: &[f64],
        mut field: DisjointSlice<f64>,
    ) {
        let t = thread::index_1d().get();
        if t >= n as usize {
            return;
        }
        let nc = nc as usize;
        let (j, c) = (t / nc, t % nc);
        let k = listed[j] as usize;
        if flushed[k] != 0 {
            return;
        }
        let dc = cells[c];
        for root in 0..NN {
            let r = k * NN + root;
            if group[r] as usize != root || joined[r] == 0 {
                continue;
            }
            let (mut inventory, mut volume, mut area) = (0.0, 0.0, -0.0);
            for i in 0..NN {
                let idx = k * NN + i;
                if group[idx] as usize != root {
                    continue;
                }
                let weight = mass[idx] * layer_volume_of(eta[idx] - bed[idx], dc);
                // SAFETY: thread (j, c) alone touches cell c of element k's
                // nodes.
                inventory += weight * unsafe { *field.get_unchecked_mut(idx * nc + c) };
                volume += weight;
                area += mass[idx];
            }
            if volume > dry_volume_of(area, dc) {
                for i in 0..NN {
                    let idx = k * NN + i;
                    if group[idx] as usize == root {
                        // SAFETY: as above.
                        unsafe {
                            *field.get_unchecked_mut(idx * nc + c) = inventory / volume;
                        }
                    }
                }
            }
        }
    }

    /// `MergedCells::pass_through` of one wave of the flushed elements, one
    /// thread per element of the wave and cell: at a step's `end` each
    /// element's inventory over its volume, before that the value that
    /// passes its inflow through (`inflow_from`, `inflow_rate` of flushed
    /// element `j` from `inflow_offsets[j]` to `inflow_offsets[j + 1]`), from
    /// its neighbours' values in `out`. The wave's elements are the flushed
    /// elements `wave[first..first + n / nc]` (indices into the flushed
    /// list); none of them reads another's nodes, and the waves run in the
    /// order that gives each the values the list's order would
    /// ([`Merged3D::upload`]).
    #[kernel]
    pub fn merged_pass_through(
        nc: u32,
        n: u32,
        first: u32,
        end: u32,
        wave: &[u32],
        flushed_k: &[u32],
        flushed_io: &[[f64; 2]],
        inflow_offsets: &[u32],
        inflow_from: &[u32],
        inflow_rate: &[f64],
        mass: &[f64],
        eta: &[f64],
        bed: &[f64],
        cells: &[f64],
        q: &[f64],
        mut out: DisjointSlice<f64>,
    ) {
        let t = thread::index_1d().get();
        if t >= n as usize {
            return;
        }
        let nc = nc as usize;
        let (w, c) = (t / nc, t % nc);
        let j = wave[first as usize + w] as usize;
        let dc = cells[c];
        let k = flushed_k[j] as usize;
        let [entering, leaving] = flushed_io[j];
        let inflows = inflow_offsets[j] as usize..inflow_offsets[j + 1] as usize;
        // From other elements (`Iterator::sum`, from −0.0)
        let mut from_elements = -0.0;
        for e in inflows.clone() {
            from_elements += inflow_rate[e];
        }
        let passing = if leaving > 0.0 && from_elements > 0.0 {
            (entering.min(leaving) / leaving).min(1.0)
        } else {
            0.0
        };
        let (mut inventory, mut volume, mut area) = (0.0, 0.0, -0.0);
        for i in 0..NN {
            let idx = k * NN + i;
            let m = mass[idx];
            inventory += m * q[idx * nc + c];
            volume += m * layer_volume_of(eta[idx] - bed[idx], dc);
            area += m;
        }
        let own = volume > dry_volume_of(area, dc);
        let value = if end != 0 || passing == 0.0 {
            if !own {
                return;
            }
            inventory / volume
        } else {
            let mut sum = -0.0;
            for e in inflows {
                // SAFETY: no thread of this wave writes a node it reads
                // (none of the wave's elements reads another's).
                let value = unsafe { *out.get_unchecked_mut(inflow_from[e] as usize * nc + c) };
                sum += inflow_rate[e] * value;
            }
            let entering = sum / from_elements;
            if own {
                passing * entering + (1.0 - passing) * (inventory / volume)
            } else {
                entering
            }
        };
        for i in 0..NN {
            // SAFETY: thread (w, c) alone writes cell c of element k's nodes.
            unsafe {
                *out.get_unchecked_mut((k * NN + i) * nc + c) = value;
            }
        }
    }

    /// `inventory_of`, one thread per cell: the value times its cell's
    /// volume under `eta`, but in the elements `flushed` lists (empty:
    /// none).
    #[kernel]
    pub fn inventory_of(
        nc: u32,
        n: u32,
        eta: &[f64],
        bed: &[f64],
        cells: &[f64],
        values: &[f64],
        flushed: &[u32],
        mut q: DisjointSlice<f64>,
    ) {
        let t = thread::index_1d().get();
        if t >= n as usize {
            return;
        }
        let nc = nc as usize;
        let (col, c) = (t / nc, t % nc);
        if !flushed.is_empty() && flushed[col / NN] != 0 {
            return;
        }
        // SAFETY: thread t alone touches cell t.
        unsafe {
            *q.get_unchecked_mut(t) = values[t] * layer_volume_of(eta[col] - bed[col], cells[c]);
        }
    }

    /// `dry_inventories`, one thread per cell: [`inventory_of`] at the nodes
    /// shallower than `DRY_DEPTH` only.
    #[kernel]
    pub fn dry_inventories(
        nc: u32,
        n: u32,
        eta: &[f64],
        bed: &[f64],
        cells: &[f64],
        values: &[f64],
        flushed: &[u32],
        mut q: DisjointSlice<f64>,
    ) {
        let t = thread::index_1d().get();
        if t >= n as usize {
            return;
        }
        let nc = nc as usize;
        let (col, c) = (t / nc, t % nc);
        if !flushed.is_empty() && flushed[col / NN] != 0 {
            return;
        }
        let depth = eta[col] - bed[col];
        if depth < DRY_DEPTH {
            // SAFETY: thread t alone touches cell t.
            unsafe {
                *q.get_unchecked_mut(t) = values[t] * layer_volume_of(depth, cells[c]);
            }
        }
    }

    /// `update_density_on` with the linear equation of state (`eos`: ρ₀, T₀,
    /// S₀, α, β), one thread per layer of the elements `listed` (empty:
    /// every element), `per_element` layers each.
    #[kernel]
    pub fn density(
        n: u32,
        per_element: u32,
        listed: &[u32],
        eos: &[f64],
        temp: &[f64],
        salt: &[f64],
        mut rho: DisjointSlice<f64>,
    ) {
        let t = thread::index_1d().get();
        if t >= n as usize {
            return;
        }
        let idx = if listed.is_empty() {
            t
        } else {
            let per = per_element as usize;
            listed[t / per] as usize * per + t % per
        };
        let eos = LinearEOS {
            rho0: eos[0],
            t0: eos[1],
            s0: eos[2],
            alpha: eos[3],
            beta: eos[4],
        };
        // SAFETY: thread t alone touches layer idx (the listed elements are
        // distinct).
        unsafe {
            *rho.get_unchecked_mut(idx) = eos.compute_density(temp[idx], salt[idx], 0.0);
        }
    }

    /// `StageStash::forget` and `mask` with `Hydrostatic3D::mask_stage`'s
    /// test, one thread per layer: a column shallower than `thin_depth`
    /// under `eta` is masked, its `u`, `v` (and, by its bed layer's thread,
    /// `ū`, `v̄`) kept and set to zero; the others unmasked.
    #[kernel]
    pub fn mask_thin(
        nl: u32,
        n: u32,
        thin_depth: f64,
        eta: &[f64],
        bed: &[f64],
        mut u: DisjointSlice<f64>,
        mut v: DisjointSlice<f64>,
        mut ubar: DisjointSlice<f64>,
        mut vbar: DisjointSlice<f64>,
        mut masked: DisjointSlice<u32>,
        mut kept_u: DisjointSlice<f64>,
        mut kept_v: DisjointSlice<f64>,
        mut kept_ubar: DisjointSlice<f64>,
        mut kept_vbar: DisjointSlice<f64>,
    ) {
        let t = thread::index_1d().get();
        if t >= n as usize {
            return;
        }
        let nl = nl as usize;
        let (col, l) = (t / nl, t % nl);
        let thin = eta[col] - bed[col] < thin_depth;
        // SAFETY: thread t alone touches layer t, and the bed layer's
        // thread column col.
        unsafe {
            if l == 0 {
                *masked.get_unchecked_mut(col) = u32::from(thin);
            }
            if thin {
                let u = u.get_unchecked_mut(t);
                *kept_u.get_unchecked_mut(t) = *u;
                *u = 0.0;
                let v = v.get_unchecked_mut(t);
                *kept_v.get_unchecked_mut(t) = *v;
                *v = 0.0;
                if l == 0 {
                    let ubar = ubar.get_unchecked_mut(col);
                    *kept_ubar.get_unchecked_mut(col) = *ubar;
                    *ubar = 0.0;
                    let vbar = vbar.get_unchecked_mut(col);
                    *kept_vbar.get_unchecked_mut(col) = *vbar;
                    *vbar = 0.0;
                }
            }
        }
    }

    /// `StageStash::restore_into`, one thread per layer: the masked
    /// columns' kept `u`, `v` (and `ū`, `v̄`) back.
    #[kernel]
    pub fn restore_thin(
        nl: u32,
        n: u32,
        masked: &[u32],
        kept_u: &[f64],
        kept_v: &[f64],
        kept_ubar: &[f64],
        kept_vbar: &[f64],
        mut u: DisjointSlice<f64>,
        mut v: DisjointSlice<f64>,
        mut ubar: DisjointSlice<f64>,
        mut vbar: DisjointSlice<f64>,
    ) {
        let t = thread::index_1d().get();
        if t >= n as usize {
            return;
        }
        let nl = nl as usize;
        let (col, l) = (t / nl, t % nl);
        if masked[col] == 0 {
            return;
        }
        // SAFETY: as `mask_thin`.
        unsafe {
            *u.get_unchecked_mut(t) = kept_u[t];
            *v.get_unchecked_mut(t) = kept_v[t];
            if l == 0 {
                *ubar.get_unchecked_mut(col) = kept_ubar[col];
                *vbar.get_unchecked_mut(col) = kept_vbar[col];
            }
        }
    }

    /// `zero_thin_momentum`, one thread per layer: the momentum tendencies
    /// of the columns shallower than `thin_depth` under `eta` to zero.
    #[kernel]
    pub fn zero_thin(
        nl: u32,
        n: u32,
        thin_depth: f64,
        eta: &[f64],
        bed: &[f64],
        mut rhs_u: DisjointSlice<f64>,
        mut rhs_v: DisjointSlice<f64>,
    ) {
        let t = thread::index_1d().get();
        if t >= n as usize {
            return;
        }
        let col = t / nl as usize;
        if eta[col] - bed[col] < thin_depth {
            // SAFETY: thread t alone touches layer t.
            unsafe {
                *rhs_u.get_unchecked_mut(t) = 0.0;
                *rhs_v.get_unchecked_mut(t) = 0.0;
            }
        }
    }

    /// `NestingColumns::relax_shear`, one thread per band slot: the shear of
    /// `(u, v)` relaxed to the parent's (`parent_u`, `parent_v`,
    /// `[slot][level]`) at the slot's `rate`, `rate (p_l − x_l − (⟨p⟩ −
    /// ⟨x⟩))` with dgcom's `depth_average`, added to `rhs_u`, `rhs_v`; not
    /// where the rate is zero, nor in a column shallower than `thin_depth`.
    #[kernel]
    pub fn relax_shear(
        nl: u32,
        n_slots: u32,
        thin_depth: f64,
        nodes: &[u32],
        rate: &[f64],
        eta: &[f64],
        bed: &[f64],
        d_sigma: &[f64],
        parent_u: &[f64],
        parent_v: &[f64],
        u: &[f64],
        v: &[f64],
        mut rhs_u: DisjointSlice<f64>,
        mut rhs_v: DisjointSlice<f64>,
    ) {
        let slot = thread::index_1d().get();
        if slot >= n_slots as usize {
            return;
        }
        let (r, flat) = (rate[slot], nodes[slot] as usize);
        if r == 0.0 || eta[flat] - bed[flat] < thin_depth {
            return;
        }
        let nl = nl as usize;
        let (parent, column) = (slot * nl..(slot + 1) * nl, flat * nl..(flat + 1) * nl);
        let ds = &d_sigma[..nl];
        let offset_u =
            depth_average(&parent_u[parent.clone()], ds) - depth_average(&u[column.clone()], ds);
        let offset_v = depth_average(&parent_v[parent], ds) - depth_average(&v[column], ds);
        // SAFETY: thread slot alone touches column flat (a node has one
        // slot).
        unsafe {
            for l in 0..nl {
                let (p, x) = (slot * nl + l, flat * nl + l);
                *rhs_u.get_unchecked_mut(x) += r * (parent_u[p] - u[x] - offset_u);
            }
            for l in 0..nl {
                let (p, x) = (slot * nl + l, flat * nl + l);
                *rhs_v.get_unchecked_mut(x) += r * (parent_v[p] - v[x] - offset_v);
            }
        }
    }

    /// `NestingColumns::relax_tracers`, one thread per band slot: T and S
    /// relaxed to the parent's at the slot's `rate` as inventories,
    /// `D Δσ_l rate (C_p − C)`, added to `rhs_t`, `rhs_s`; not where the rate
    /// is zero, nor in a column shallower than `thin_depth`.
    #[kernel]
    pub fn relax_tracers(
        nl: u32,
        n_slots: u32,
        thin_depth: f64,
        nodes: &[u32],
        rate: &[f64],
        eta: &[f64],
        bed: &[f64],
        d_sigma: &[f64],
        parent_t: &[f64],
        parent_s: &[f64],
        temp: &[f64],
        salt: &[f64],
        mut rhs_t: DisjointSlice<f64>,
        mut rhs_s: DisjointSlice<f64>,
    ) {
        let slot = thread::index_1d().get();
        if slot >= n_slots as usize {
            return;
        }
        let (r, flat) = (rate[slot], nodes[slot] as usize);
        let depth = eta[flat] - bed[flat];
        if r == 0.0 || depth < thin_depth {
            return;
        }
        let nl = nl as usize;
        // SAFETY: as `relax_shear`.
        unsafe {
            for l in 0..nl {
                let (p, x) = (slot * nl + l, flat * nl + l);
                *rhs_t.get_unchecked_mut(x) += depth * d_sigma[l] * r * (parent_t[p] - temp[x]);
            }
            for l in 0..nl {
                let (p, x) = (slot * nl + l, flat * nl + l);
                *rhs_s.get_unchecked_mut(x) += depth * d_sigma[l] * r * (parent_s[p] - salt[x]);
            }
        }
    }

    /// A slot for every lane that `takes` one in the list counted at
    /// `count`: one atomic per warp, the slots in lane order within it.
    /// Every lane of the warp calls it.
    ///
    /// # Safety
    ///
    /// `count` points to a u32 that only atomics touch while the kernel runs.
    #[inline]
    unsafe fn warp_slot(takes: bool, count: *mut u32) -> usize {
        let mask = warp::ballot(takes);
        let lane = warp::lane_id();
        let mut base = 0u32;
        if mask != 0 && lane == mask.trailing_zeros() {
            // SAFETY: the caller's.
            let count = unsafe { DeviceAtomicU32::from_ptr(count) };
            base = count.fetch_add(mask.count_ones(), AtomicOrdering::Relaxed);
        }
        let base = warp::shuffle(base, mask.trailing_zeros() & 31);
        (base + (mask & warp::lanemask_lt()).count_ones()) as usize
    }

    /// `MergedCells::find`, a thread per element: dgcom's `find_element` of
    /// its cells (`eta` at the step's start and its rate `eta_rate` over the
    /// step `dt`, `source` the water no flux carried, over `bed`), the
    /// pass's mean mass fluxes out of its face nodes (`face`) and through its
    /// subcell interfaces (`subcell`), its stages' forward-Euler step
    /// `euler_dt`, with columns shallower than `thin_depth` thin. Every
    /// node's `group` and `joined` and every element's `flushed`; the
    /// elements with a group listed in `elements` (`counts[0]` of them), the
    /// flushed ones in `flushed_k` with the water entering and leaving
    /// (`flushed_io`, `counts[1]` of them) and their inflows from other
    /// elements (flushed element `j`'s `inflow_n[j]` from `j·SLOTS` in
    /// `inflow_from`, `inflow_rate`, in the CPU's order); the cells merged
    /// and flushed summed in `counts[2]`, `counts[3]`. `counts` zeroed
    /// before: one atomic per warp and list, the lists' order the atomics'
    /// (the host sorts the flushed one; no result depends on the other's).
    #[kernel]
    pub fn merged_find(
        ne: u32,
        dt: f64,
        thin_depth: f64,
        eta: &[f64],
        bed: &[f64],
        eta_rate: &[f64],
        source: &[f64],
        euler_dt: &[f64],
        mass: &[f64],
        fnode: &[[f64; 3]],
        lift: &[f64],
        w: &[f64],
        face: &[f64],
        subcell: &[f64],
        across: &[u32],
        counts: *mut u32,
        mut group: DisjointSlice<u32>,
        mut joined: DisjointSlice<u32>,
        mut flushed: DisjointSlice<u32>,
        mut elements: DisjointSlice<u32>,
        mut flushed_k: DisjointSlice<u32>,
        mut flushed_io: DisjointSlice<[f64; 2]>,
        mut inflow_n: DisjointSlice<u32>,
        mut inflow_from: DisjointSlice<u32>,
        mut inflow_rate: DisjointSlice<f64>,
    ) {
        // Every lane reaches the votes: no early return
        let k = thread::index_1d().get();
        // The volume rate out of face node `fi` of face `f`
        let face_rate = |f: usize, fi: usize| {
            let (node, s) = (face_node(f, fi), (k * 4 + f) * N1 + fi);
            face_volume_rate(
                mass[k * NN + node],
                fnode[s][2],
                lift[(f * NN + node) * N1 + fi],
                face[s],
            )
        };
        let mut found = None;
        if k < ne as usize {
            found = find_element::<NN>(
                NN,
                N1,
                dt,
                euler_dt[k],
                thin_depth,
                |i| {
                    let idx = k * NN + i;
                    CellWater {
                        eta: eta[idx],
                        eta_rate: eta_rate[idx],
                        bed: bed[idx],
                        mass: mass[idx],
                        source: source[idx],
                    }
                },
                &face_rate,
                face_node,
                |slot| subcell[k * NSUB + slot],
                |line| w[line],
            );
            for i in 0..NN {
                let (g, j) = match &found {
                    Some(found) => (found.group[i] as u32, u32::from(found.joined[i])),
                    None => (i as u32, 0),
                };
                // SAFETY: thread k alone touches element k's nodes.
                unsafe {
                    *group.get_unchecked_mut(k * NN + i) = g;
                    *joined.get_unchecked_mut(k * NN + i) = j;
                }
            }
            // SAFETY: as above.
            unsafe {
                *flushed.get_unchecked_mut(k) = u32::from(found.is_some_and(|f| f.flushed));
            }
        }
        let (n_joined, is_flushed, entering, leaving) = match &found {
            Some(found) => (
                found.joined.iter().filter(|&&j| j).count() as u32,
                found.flushed,
                found.entering,
                found.leaving,
            ),
            None => (0, false, 0.0, 0.0),
        };
        if n_joined > 0 {
            // SAFETY: `counts` holds four u32 that only atomics touch here.
            let cells =
                unsafe { DeviceAtomicU32::from_ptr(counts.add(2 + usize::from(is_flushed))) };
            cells.fetch_add(n_joined, AtomicOrdering::Relaxed);
        }
        // SAFETY: as above; each lane that takes a slot writes it alone.
        unsafe {
            let at = warp_slot(n_joined > 0, counts);
            if n_joined > 0 {
                *elements.get_unchecked_mut(at) = k as u32;
            }
            let j = warp_slot(is_flushed, counts.add(1));
            if is_flushed {
                *flushed_k.get_unchecked_mut(j) = k as u32;
                *flushed_io.get_unchecked_mut(j) = [entering, leaving];
                // What flows in from other elements, face by face
                let mut n = 0;
                for f in 0..4 {
                    for fi in 0..N1 {
                        let from = across[(k * 4 + f) * N1 + fi];
                        if from == WALL || from == OPEN {
                            continue;
                        }
                        let rate = -face_rate(f, fi);
                        if rate > 0.0 {
                            *inflow_from.get_unchecked_mut(j * SLOTS + n) = from;
                            *inflow_rate.get_unchecked_mut(j * SLOTS + n) = rate;
                            n += 1;
                        }
                    }
                }
                *inflow_n.get_unchecked_mut(j) = n as u32;
            }
        }
    }
}

// =============================================================================
// Host side
// =============================================================================

/// The fields of a 3D state as the stages carry them (dgcom's `Solution3D`:
/// `η`, `ū`, `v̄` per column; `u`, `v`, T, S per layer and k, ψ per w-point,
/// as values or inventories; the density, empty in an inventory buffer), on
/// the device.
pub struct State3D {
    pub eta: DeviceBuffer<f64>,
    pub ubar: DeviceBuffer<f64>,
    pub vbar: DeviceBuffer<f64>,
    pub u: DeviceBuffer<f64>,
    pub v: DeviceBuffer<f64>,
    pub temp: DeviceBuffer<f64>,
    pub salt: DeviceBuffer<f64>,
    /// Empty without a prognostic closure
    pub tke: DeviceBuffer<f64>,
    pub gls: DeviceBuffer<f64>,
    pub rho: DeviceBuffer<f64>,
}

/// The inventory fields in the mode splitter's order (`inventory_fields`):
/// u, v, T, S on the layers, then k, ψ on the w-points.
pub const N_FIELDS: usize = 6;
/// The fields before this one are the momentum's (each node's own: no
/// groups), ...
const N_MOMENTUM_FIELDS: usize = 2;
/// ... before this one on the layers, from it on the w-points.
const N_LAYER_FIELDS: usize = 4;

impl State3D {
    /// `state`'s fields, uploaded.
    pub fn upload(gpu: &Gpu, state: &Solution3D) -> Result<Self, Box<dyn std::error::Error>> {
        let up = |v: &[f64]| DeviceBuffer::from_host(&gpu.stream, v);
        Ok(Self {
            eta: up(&state.eta.data)?,
            ubar: up(&state.ubar.data)?,
            vbar: up(&state.vbar.data)?,
            u: up(&state.u)?,
            v: up(&state.v)?,
            temp: up(&state.temp)?,
            salt: up(&state.salt)?,
            tke: up(&state.tke)?,
            gls: up(&state.gls)?,
            rho: up(&state.rho)?,
        })
    }

    /// The device's fields into `state` (sized as they are).
    pub fn download_into(
        &self,
        gpu: &Gpu,
        state: &mut Solution3D,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let st = &gpu.stream;
        self.eta.copy_to_host(st, &mut state.eta.data)?;
        self.ubar.copy_to_host(st, &mut state.ubar.data)?;
        self.vbar.copy_to_host(st, &mut state.vbar.data)?;
        for (to, from) in [
            (&mut state.u, &self.u),
            (&mut state.v, &self.v),
            (&mut state.temp, &self.temp),
            (&mut state.salt, &self.salt),
            (&mut state.tke, &self.tke),
            (&mut state.gls, &self.gls),
            (&mut state.rho, &self.rho),
        ] {
            to.resize(from.len(), 0.0);
            from.copy_to_host(st, to)?;
        }
        Ok(())
    }

    /// The inventory fields (`inventory_fields`).
    pub fn fields(&self) -> [&DeviceBuffer<f64>; N_FIELDS] {
        [
            &self.u, &self.v, &self.temp, &self.salt, &self.tke, &self.gls,
        ]
    }

    pub fn fields_mut(&mut self) -> [&mut DeviceBuffer<f64>; N_FIELDS] {
        [
            &mut self.u,
            &mut self.v,
            &mut self.temp,
            &mut self.salt,
            &mut self.tke,
            &mut self.gls,
        ]
    }

    /// `η`, `ū`, `v̄` of `from` (`copy_barotropic_rows` on every element).
    pub fn copy_barotropic(
        &mut self,
        gpu: &Gpu,
        from: &Self,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let st = &gpu.stream;
        self.eta.copy_from_device_async(&from.eta, st)?;
        self.ubar.copy_from_device_async(&from.ubar, st)?;
        self.vbar.copy_from_device_async(&from.vbar, st)?;
        Ok(())
    }

    /// Every field of `from` (sized alike).
    pub fn copy_from(&mut self, gpu: &Gpu, from: &Self) -> Result<(), Box<dyn std::error::Error>> {
        self.copy_barotropic(gpu, from)?;
        for (to, from) in self.fields_mut().into_iter().zip(from.fields()) {
            if !to.is_empty() {
                to.copy_from_device_async(from, &gpu.stream)?;
            }
        }
        if !self.rho.is_empty() {
            self.rho.copy_from_device_async(&from.rho, &gpu.stream)?;
        }
        Ok(())
    }
}

/// What `Hydrostatic3D::mask_stage` took from a stage's values (dgcom's
/// `StageStash`): the masked columns, and their `u`, `v`, `ū`, `v̄`.
pub struct Stash3D {
    pub masked: DeviceBuffer<u32>,
    pub u: DeviceBuffer<f64>,
    pub v: DeviceBuffer<f64>,
    pub ubar: DeviceBuffer<f64>,
    pub vbar: DeviceBuffer<f64>,
}

impl Stash3D {
    /// Nothing masked, on `statics`' columns.
    pub fn new(gpu: &Gpu, statics: &Statics3D) -> Result<Self, Box<dyn std::error::Error>> {
        let st = &gpu.stream;
        let (n_columns, n) = (statics.n_columns(), statics.n_columns() * statics.n_levels);
        Ok(Self {
            masked: DeviceBuffer::zeroed(st, n_columns)?,
            u: DeviceBuffer::zeroed(st, n)?,
            v: DeviceBuffer::zeroed(st, n)?,
            ubar: DeviceBuffer::zeroed(st, n_columns)?,
            vbar: DeviceBuffer::zeroed(st, n_columns)?,
        })
    }

    /// Mark every column unmasked (`StageStash::forget`).
    pub fn forget(&mut self, gpu: &Gpu) -> Result<(), Box<dyn std::error::Error>> {
        self.masked.zero_async(&gpu.stream)?;
        Ok(())
    }
}

/// A stage's values with what the mask took from them (the mode splitter's
/// `StageValues`).
pub struct StageValues3D {
    pub values: State3D,
    pub stash: Stash3D,
}

/// What the barotropic pass of a step moved, as [`Merged3D::find`] reads it
/// (dgcom's `StepWater`, on the device).
pub struct StepWater3D<'a> {
    /// `η` at the step's start
    pub eta: &'a DeviceBuffer<f64>,
    /// `∂η/∂t` over the step
    pub eta_rate: &'a DeviceBuffer<f64>,
    /// The step (s)
    pub dt: f64,
    /// Every element's longest forward-Euler step in the stages (s)
    pub euler_dt: &'a DeviceBuffer<f64>,
    /// The mean mass flux out of every face node, `(k·4 + face)·N1 + fi`
    pub face: &'a DeviceBuffer<f64>,
    /// The mean mass flux through every subcell interface, `k·NSUB + slot`;
    /// NaN where the pass did not move the element on its subcells
    /// throughout
    pub subcell: &'a DeviceBuffer<f64>,
    /// The water of every column that no flux carried (m/s)
    pub source: &'a DeviceBuffer<f64>,
    /// Columns shallower than this are thin (m)
    pub thin_depth: f64,
}

/// The step's merged cells on the device (dgcom's `MergedCells`): found
/// there from the pass's water ([`Merged3D::find`]), or uploaded from
/// [`MergedCells::tables`](dgcom::solver::rhs::MergedCells::tables)
/// ([`Merged3D::upload`]).
pub struct Merged3D {
    /// The elements with a group (found on the device: in the atomics'
    /// order), and how many
    elements: DeviceBuffer<u32>,
    n_grouped: usize,
    /// Per node: its group's first node, and whether it is in a group of two
    /// or more
    group: DeviceBuffer<u32>,
    joined: DeviceBuffer<u32>,
    /// Per element: flushed
    flushed: DeviceBuffer<u32>,
    /// The flushed elements in order, with the water entering and leaving
    /// each, and where their inflows start in the inflow lists
    flushed_k: DeviceBuffer<u32>,
    flushed_io: DeviceBuffer<[f64; 2]>,
    n_flushed: usize,
    inflow_offsets: DeviceBuffer<u32>,
    inflow_from: DeviceBuffer<u32>,
    inflow_rate: DeviceBuffer<f64>,
    /// The flushed elements (indices into their list) wave by wave, and
    /// where each wave starts in it
    waves: DeviceBuffer<u32>,
    wave_offsets: Vec<usize>,
    /// [`Self::find`]'s lists as its kernel leaves them
    found: Option<Found>,
    /// Cells merged and flushed, summed over the steps found on the device
    counts: [usize; 2],
}

/// `merged_find`'s lists: its counts (the elements with a group, the
/// flushed ones, the cells merged and flushed), and the flushed elements in
/// the atomics' order with the water entering and leaving and their inflows
/// (`n` of them from `j·SLOTS` in `from`, `rate`).
struct Found {
    counts: DeviceBuffer<u32>,
    k: DeviceBuffer<u32>,
    io: DeviceBuffer<[f64; 2]>,
    n: DeviceBuffer<u32>,
    from: DeviceBuffer<u32>,
    rate: DeviceBuffer<f64>,
}

/// A step's groups on the host, as [`MergedTables`] holds them (the
/// elements with a group in order): what [`Merged3D::download`] gives.
pub struct MergedLists {
    pub group: Vec<u32>,
    pub joined: Vec<u32>,
    pub flushed: Vec<u32>,
    pub elements: Vec<u32>,
    pub flushed_elements: Vec<(u32, f64, f64)>,
    pub inflows: Vec<(u32, u32, f64)>,
}

impl MergedLists {
    /// The CPU's groups `tables`.
    pub fn from_tables(tables: MergedTables) -> Self {
        let flags = |v: &[bool]| v.iter().map(|&b| u32::from(b)).collect();
        Self {
            group: tables.group.iter().map(|&g| u32::from(g)).collect(),
            joined: flags(tables.joined),
            flushed: flags(tables.flushed),
            elements: tables.elements.to_vec(),
            flushed_elements: tables.flushed_elements.to_vec(),
            inflows: tables.inflows.to_vec(),
        }
    }
}

/// The first `n` values of `buf` (waits for the device): what
/// `DeviceBuffer::copy_to_host` does, on a prefix.
fn prefix_to_host<T: Default + Clone>(
    gpu: &Gpu,
    buf: &DeviceBuffer<T>,
    n: usize,
) -> Result<Vec<T>, Box<dyn std::error::Error>> {
    let mut host = vec![T::default(); n];
    if n == 0 {
        return Ok(host);
    }
    let st = &gpu.stream;
    let view = buf.slice(0..n)?;
    let lease = view.storage().acquire(st, false, false)?;
    view.context().bind_to_thread()?;
    // SAFETY: the view holds `n` values of `T` on the device, `host` room
    // for `n`; the stream is waited for before `host` is read.
    let submitted = unsafe {
        cuda_core::memory::memcpy_dtoh_async(
            host.as_mut_ptr(),
            view.cu_deviceptr(),
            view.num_bytes(),
            st.cu_stream(),
        )
    };
    let completed = st.synchronize();
    if completed.is_err() {
        std::mem::forget(lease);
    }
    submitted.and(completed)?;
    Ok(host)
}

impl Merged3D {
    /// The step's groups `tables` (`MergedCells::find`'s).
    pub fn upload(gpu: &Gpu, tables: MergedTables) -> Result<Self, Box<dyn std::error::Error>> {
        let st = &gpu.stream;
        let flags = |v: &[bool]| v.iter().map(|&b| u32::from(b)).collect::<Vec<u32>>();
        let group: Vec<u32> = tables.group.iter().map(|&g| u32::from(g)).collect();
        let mut merged = Self::with(
            gpu,
            DeviceBuffer::from_host(st, tables.elements)?,
            tables.elements.len(),
            DeviceBuffer::from_host(st, &group)?,
            DeviceBuffer::from_host(st, &flags(tables.joined))?,
            DeviceBuffer::from_host(st, &flags(tables.flushed))?,
        )?;
        merged.set_flushed(
            gpu,
            tables.flushed_elements,
            tables.inflows,
            tables.flushed.len(),
        )?;
        Ok(merged)
    }

    /// The groups `group`, `joined`, `flushed` with the elements listed in
    /// `elements` (`n_grouped` of them), no element flushed.
    fn with(
        gpu: &Gpu,
        elements: DeviceBuffer<u32>,
        n_grouped: usize,
        group: DeviceBuffer<u32>,
        joined: DeviceBuffer<u32>,
        flushed: DeviceBuffer<u32>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let st = &gpu.stream;
        Ok(Self {
            elements,
            n_grouped,
            group,
            joined,
            flushed,
            flushed_k: DeviceBuffer::zeroed(st, 0)?,
            flushed_io: DeviceBuffer::zeroed(st, 0)?,
            n_flushed: 0,
            inflow_offsets: DeviceBuffer::from_host(st, &[0u32])?,
            inflow_from: DeviceBuffer::zeroed(st, 0)?,
            inflow_rate: DeviceBuffer::zeroed(st, 0)?,
            waves: DeviceBuffer::zeroed(st, 0)?,
            wave_offsets: vec![0],
            found: None,
            counts: [0; 2],
        })
    }

    /// The flushed elements `flushed_elements` in order, with the water
    /// entering and leaving each, and their inflows `inflows` (element by
    /// element in that order: the element, the node the water comes from,
    /// the rate), on a mesh of `n_elements`: the lists and their waves.
    fn set_flushed(
        &mut self,
        gpu: &Gpu,
        flushed_elements: &[(u32, f64, f64)],
        inflows: &[(u32, u32, f64)],
        n_elements: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let st = &gpu.stream;
        let flushed_k: Vec<u32> = flushed_elements.iter().map(|f| f.0).collect();
        let flushed_io: Vec<[f64; 2]> = flushed_elements
            .iter()
            .map(|&(_, entering, leaving)| [entering, leaving])
            .collect();
        // Each flushed element's inflows, as `pass_through` splits them off
        // the list in turn
        let mut inflow_offsets = Vec::with_capacity(flushed_k.len() + 1);
        let mut at = 0;
        inflow_offsets.push(0u32);
        for &k in &flushed_k {
            at += inflows[at..].iter().take_while(|i| i.0 == k).count();
            inflow_offsets.push(at as u32);
        }
        assert_eq!(at, inflows.len(), "every inflow is a flushed element's");
        let inflow_from: Vec<u32> = inflows.iter().map(|i| i.1).collect();
        let inflow_rate: Vec<f64> = inflows.iter().map(|i| i.2).collect();
        let (waves, wave_offsets) =
            flushed_waves(&flushed_k, &inflow_offsets, &inflow_from, n_elements);
        self.n_flushed = flushed_k.len();
        self.flushed_k = DeviceBuffer::from_host(st, &flushed_k)?;
        self.flushed_io = DeviceBuffer::from_host(st, &flushed_io)?;
        self.inflow_offsets = DeviceBuffer::from_host(st, &inflow_offsets)?;
        self.inflow_from = DeviceBuffer::from_host(st, &inflow_from)?;
        self.inflow_rate = DeviceBuffer::from_host(st, &inflow_rate)?;
        self.waves = DeviceBuffer::from_host(st, &waves)?;
        self.wave_offsets = wave_offsets;
        Ok(())
    }

    /// `MergedCells::find` on the device: the step's groups from the pass's
    /// water `water` on `s`'s mesh, replacing the last step's. Waits for the
    /// device: the host takes the counts, and the flushed elements to put
    /// them in the CPU's order (by element) and in waves.
    pub fn find(
        &mut self,
        gpu: &Gpu,
        s: &Statics3D,
        water: &StepWater3D,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let st = &gpu.stream;
        let ne = s.n_elements;
        assert_eq!(self.flushed.len(), ne, "groups for this mesh");
        if self.elements.len() < ne {
            self.elements = DeviceBuffer::zeroed(st, ne)?;
        }
        if self.found.is_none() {
            self.found = Some(Found {
                counts: DeviceBuffer::zeroed(st, 4)?,
                k: DeviceBuffer::zeroed(st, ne)?,
                io: DeviceBuffer::zeroed(st, ne)?,
                n: DeviceBuffer::zeroed(st, ne)?,
                from: DeviceBuffer::zeroed(st, ne * SLOTS)?,
                rate: DeviceBuffer::zeroed(st, ne * SLOTS)?,
            });
        }
        let found = self.found.as_mut().expect("made above");
        found.counts.zero_async(st)?;
        // SAFETY: every buffer is sized for this mesh; `counts` holds the
        // four u32 the kernel's atomics touch.
        unsafe {
            gpu.module_hooks.merged_find(
                st,
                launch(ne, COLUMN_BLOCK),
                ne as u32,
                water.dt,
                water.thin_depth,
                water.eta,
                &s.bed,
                water.eta_rate,
                water.source,
                water.euler_dt,
                &s.mass,
                &s.fnode,
                &s.lift,
                &s.w,
                water.face,
                water.subcell,
                &s.across,
                found.counts.cu_deviceptr() as *mut u32,
                &mut self.group,
                &mut self.joined,
                &mut self.flushed,
                &mut self.elements,
                &mut found.k,
                &mut found.io,
                &mut found.n,
                &mut found.from,
                &mut found.rate,
            )?;
        }
        let counts = found.counts.to_host_vec(st)?;
        let [n_grouped, n_flushed, merged, flushed] = [0, 1, 2, 3].map(|i| counts[i] as usize);
        self.n_grouped = n_grouped;
        self.counts[0] += merged;
        self.counts[1] += flushed;
        if n_flushed == 0 && self.n_flushed == 0 {
            return Ok(());
        }
        // The flushed elements in the CPU's order, each with its inflows
        let k = prefix_to_host(gpu, &found.k, n_flushed)?;
        let io = prefix_to_host(gpu, &found.io, n_flushed)?;
        let n = prefix_to_host(gpu, &found.n, n_flushed)?;
        let from = prefix_to_host(gpu, &found.from, n_flushed * SLOTS)?;
        let rate = prefix_to_host(gpu, &found.rate, n_flushed * SLOTS)?;
        let mut order: Vec<usize> = (0..n_flushed).collect();
        order.sort_unstable_by_key(|&j| k[j]);
        let mut flushed_elements = Vec::with_capacity(n_flushed);
        let mut inflows = Vec::new();
        for j in order {
            let [entering, leaving] = io[j];
            flushed_elements.push((k[j], entering, leaving));
            for e in j * SLOTS..j * SLOTS + n[j] as usize {
                inflows.push((k[j], from[e], rate[e]));
            }
        }
        self.set_flushed(gpu, &flushed_elements, &inflows, ne)
    }

    /// The cells merged with others and the cells of flushed elements,
    /// summed over the steps [`Self::find`] found (`MergedCells::counts`).
    pub fn counts(&self) -> [usize; 2] {
        self.counts
    }

    /// The groups on the host, the elements with a group in order.
    pub fn download(&self, gpu: &Gpu) -> Result<MergedLists, Box<dyn std::error::Error>> {
        let st = &gpu.stream;
        let mut elements = prefix_to_host(gpu, &self.elements, self.n_grouped)?;
        elements.sort_unstable();
        let k = self.flushed_k.to_host_vec(st)?;
        let io = self.flushed_io.to_host_vec(st)?;
        let offsets = self.inflow_offsets.to_host_vec(st)?;
        let from = self.inflow_from.to_host_vec(st)?;
        let rate = self.inflow_rate.to_host_vec(st)?;
        let mut inflows = Vec::with_capacity(from.len());
        for (j, &k) in k.iter().enumerate() {
            for e in offsets[j] as usize..offsets[j + 1] as usize {
                inflows.push((k, from[e], rate[e]));
            }
        }
        Ok(MergedLists {
            group: self.group.to_host_vec(st)?,
            joined: self.joined.to_host_vec(st)?,
            flushed: self.flushed.to_host_vec(st)?,
            elements,
            flushed_elements: k
                .iter()
                .zip(&io)
                .map(|(&k, io)| (k, io[0], io[1]))
                .collect(),
            inflows,
        })
    }

    /// How many waves the flushed elements take.
    pub fn n_waves(&self) -> usize {
        self.wave_offsets.len() - 1
    }

    /// The waves in reverse order (the gate's mutant: of two flushed
    /// elements one of which reads the other's nodes, the earlier in the
    /// list converts after the later one).
    pub(crate) fn reverse_waves(&mut self, gpu: &Gpu) -> Result<(), Box<dyn std::error::Error>> {
        let waves = self.waves.to_host_vec(&gpu.stream)?;
        let mut order = Vec::with_capacity(waves.len());
        let mut offsets = vec![0];
        for w in self.wave_offsets.windows(2).rev() {
            order.extend_from_slice(&waves[w[0]..w[1]]);
            offsets.push(order.len());
        }
        self.waves = DeviceBuffer::from_host(&gpu.stream, &order)?;
        self.wave_offsets = offsets;
        Ok(())
    }

    /// No groups (a step without thin columns), on `statics`' mesh.
    pub fn none(gpu: &Gpu, statics: &Statics3D) -> Result<Self, Box<dyn std::error::Error>> {
        let st = &gpu.stream;
        let ne = statics.n_elements;
        let group: Vec<u32> = (0..ne * NN).map(|idx| (idx % NN) as u32).collect();
        Self::with(
            gpu,
            DeviceBuffer::zeroed(st, 0)?,
            0,
            DeviceBuffer::from_host(st, &group)?,
            DeviceBuffer::zeroed(st, ne * NN)?,
            DeviceBuffer::zeroed(st, ne)?,
        )
    }

    /// Whether any element has a group.
    pub fn any(&self) -> bool {
        self.n_grouped > 0
    }
}

/// The flushed elements `flushed_k` (with their inflows' source nodes,
/// `inflow_from` from `inflow_offsets`, on a mesh of `n_elements`) in waves
/// that give each the values `MergedCells::pass_through`'s order does: it
/// converts them one after another, and each reads its neighbours' nodes,
/// so of two flushed elements of which one reads the other's nodes, the one
/// earlier in the list must convert in an earlier wave (the later one sees
/// it converted, or it sees the later one not yet converted). Each element's
/// wave is one past the latest of such partners before it in the list; the
/// rest are independent. The elements wave by wave (indices into the list,
/// in list order within a wave) and where each wave starts.
fn flushed_waves(
    flushed_k: &[u32],
    inflow_offsets: &[u32],
    inflow_from: &[u32],
    n_elements: usize,
) -> (Vec<u32>, Vec<usize>) {
    let n = flushed_k.len();
    let mut position = vec![u32::MAX; n_elements];
    for (j, &k) in flushed_k.iter().enumerate() {
        position[k as usize] = j as u32;
    }
    // For each flushed element, its partners earlier in the list
    let mut earlier: Vec<Vec<usize>> = vec![Vec::new(); n];
    for j in 0..n {
        for &from in &inflow_from[inflow_offsets[j] as usize..inflow_offsets[j + 1] as usize] {
            let i = position[from as usize / NN];
            if i == u32::MAX || i as usize == j {
                continue;
            }
            let (a, b) = (j.min(i as usize), j.max(i as usize));
            earlier[b].push(a);
        }
    }
    let mut wave = vec![0usize; n];
    for j in 0..n {
        wave[j] = earlier[j].iter().map(|&a| wave[a] + 1).max().unwrap_or(0);
    }
    let n_waves = wave.iter().map(|&w| w + 1).max().unwrap_or(0);
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_by_key(|&j| (wave[j as usize], j));
    let mut offsets = vec![0usize; n_waves + 1];
    for &w in &wave {
        offsets[w + 1] += 1;
    }
    for w in 0..n_waves {
        offsets[w + 1] += offsets[w];
    }
    (order, offsets)
}

/// A nesting band on the device (dgcom's `Nesting3D`, from its
/// [`slots`](dgcom::boundary::Nesting3D::slots)): each slot's node and
/// relaxation rates (static), and the parent's columns at them
/// (`NestingColumns::values`, uploaded as they change: [`Self::set_columns`]).
pub struct Band3D {
    n_slots: usize,
    supplied: Supplied,
    nodes: DeviceBuffer<u32>,
    velocity_rate: DeviceBuffer<f64>,
    tracer_rate: DeviceBuffer<f64>,
    /// The parent's u, v, T, S at every slot, `[slot][level]`
    pub values: [DeviceBuffer<f64>; 4],
}

impl Band3D {
    /// The band of `slots` on `statics`' levels.
    pub fn new(
        gpu: &Gpu,
        statics: &Statics3D,
        slots: NestingSlots,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let st = &gpu.stream;
        let nodes: Vec<u32> = slots.nodes.iter().map(|&n| n as u32).collect();
        let n = nodes.len() * statics.n_levels;
        Ok(Self {
            n_slots: nodes.len(),
            supplied: slots.supplied,
            nodes: DeviceBuffer::from_host(st, &nodes)?,
            velocity_rate: DeviceBuffer::from_host(st, slots.velocity_rate)?,
            tracer_rate: DeviceBuffer::from_host(st, slots.tracer_rate)?,
            values: [
                DeviceBuffer::zeroed(st, n)?,
                DeviceBuffer::zeroed(st, n)?,
                DeviceBuffer::zeroed(st, n)?,
                DeviceBuffer::zeroed(st, n)?,
            ],
        })
    }

    /// The parent's columns at the slots (`NestingColumns::values`: u, v,
    /// T, S).
    pub fn set_columns(
        &mut self,
        gpu: &Gpu,
        values: [&[f64]; 4],
    ) -> Result<(), Box<dyn std::error::Error>> {
        for (to, from) in self.values.iter_mut().zip(values) {
            to.copy_from_host(&gpu.stream, from)?;
        }
        Ok(())
    }
}

/// The mode splitter's stage values and `Hydrostatic3D`'s hooks there on the
/// device.
pub struct StageHooks3D {
    gpu: Arc<Gpu>,
    /// Columns shallower than this are thin
    min_column_depth: f64,
    /// The linear equation of state: ρ₀, T₀, S₀, α, β
    eos: DeviceBuffer<f64>,
    /// The tracer limiter of `post_stage`, if any
    limiter: Option<Limiter3D>,
    none: DeviceBuffer<f64>,
    none_u32: DeviceBuffer<u32>,
}

impl StageHooks3D {
    /// The hooks with the linear equation of state `eos` and no tracer
    /// limiter; columns shallower than `min_column_depth` are thin.
    pub fn new(
        gpu: &Arc<Gpu>,
        eos: LinearEOS,
        min_column_depth: f64,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let st = &gpu.stream;
        Ok(Self {
            gpu: gpu.clone(),
            min_column_depth,
            eos: DeviceBuffer::from_host(st, &[eos.rho0, eos.t0, eos.s0, eos.alpha, eos.beta])?,
            limiter: None,
            none: DeviceBuffer::zeroed(st, 0)?,
            none_u32: DeviceBuffer::zeroed(st, 0)?,
        })
    }

    /// The hooks with `limiter` limiting every stage's tracers
    /// (`Hydrostatic3D::post_stage`; its thin depth should be the hooks').
    pub fn with_limiter(mut self, limiter: Limiter3D) -> Self {
        self.limiter = Some(limiter);
        self
    }

    /// A stage buffer for `statics`' mesh: values like `like`'s (sized as
    /// they are, the density included), nothing masked.
    pub fn values_like(
        &self,
        statics: &Statics3D,
        like: &Solution3D,
    ) -> Result<StageValues3D, Box<dyn std::error::Error>> {
        Ok(StageValues3D {
            values: State3D::upload(&self.gpu, like)?,
            stash: Stash3D::new(&self.gpu, statics)?,
        })
    }

    /// The σ-thicknesses of inventory field `field`'s cells, and how many a
    /// column has.
    fn cells_of(s: &Statics3D, field: usize) -> (&DeviceBuffer<f64>, usize) {
        if field < N_LAYER_FIELDS {
            (&s.d_sigma, s.n_levels)
        } else {
            (&s.d_sigma_w, s.n_levels + 1)
        }
    }

    /// `to_inventory` of `field` (inventory field number `which`) under
    /// `eta`.
    pub fn to_inventory(
        &self,
        s: &Statics3D,
        which: usize,
        eta: &DeviceBuffer<f64>,
        field: &mut DeviceBuffer<f64>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if field.is_empty() {
            return Ok(());
        }
        let (cells, nc) = Self::cells_of(s, which);
        let n = s.n_columns() * nc;
        // SAFETY: the buffers are sized for this mesh and these levels.
        unsafe {
            self.gpu.module_hooks.to_inventory(
                &self.gpu.stream,
                launch(n, LEVEL_BLOCK),
                nc as u32,
                n as u32,
                eta,
                &s.bed,
                cells,
                field,
            )?;
        }
        Ok(())
    }

    /// Every inventory field of `state` to inventories under its `η`
    /// (the splitter's `to_inventories`).
    pub fn to_inventories(
        &self,
        s: &Statics3D,
        state: &mut State3D,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let State3D {
            eta,
            u,
            v,
            temp,
            salt,
            tke,
            gls,
            ..
        } = state;
        for (which, field) in [u, v, temp, salt, tke, gls].into_iter().enumerate() {
            self.to_inventory(s, which, eta, field)?;
        }
        Ok(())
    }

    /// The step's start (the splitter's, before its first stage): every
    /// group's members of `state`'s T, S, k, ψ to their means
    /// (`MergedCells::start`), and the density of the elements with a group
    /// (`end_density` on them).
    pub fn start(
        &self,
        s: &Statics3D,
        merged: &Merged3D,
        state: &mut State3D,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if !merged.any() {
            return Ok(());
        }
        let (m, st) = (&self.gpu.module_hooks, &self.gpu.stream);
        let State3D {
            eta,
            temp,
            salt,
            tke,
            gls,
            ..
        } = state;
        for (which, field) in [(2, temp), (3, salt), (4, tke), (5, gls)] {
            if field.is_empty() {
                continue;
            }
            let (cells, nc) = Self::cells_of(s, which);
            let n = merged.n_grouped * nc;
            // SAFETY: the buffers are sized for this mesh and these levels,
            // the groups for this mesh (`Merged3D::upload`).
            unsafe {
                m.merged_start(
                    st,
                    launch(n, LEVEL_BLOCK),
                    nc as u32,
                    n as u32,
                    &merged.elements,
                    &merged.group,
                    &merged.joined,
                    &merged.flushed,
                    &s.mass,
                    eta,
                    &s.bed,
                    cells,
                    field,
                )?;
            }
        }
        self.density(s, state, Some(merged))
    }

    /// The density of `state`'s T and S (`update_density_on`), on the
    /// elements with a group of `merged` if given, else on every element.
    pub fn density(
        &self,
        s: &Statics3D,
        state: &mut State3D,
        merged: Option<&Merged3D>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let per = NN * s.n_levels;
        let (listed, n) = match merged {
            Some(merged) => (&merged.elements, merged.n_grouped * per),
            None => (&self.none_u32, s.n_elements * per),
        };
        if n == 0 {
            return Ok(());
        }
        // SAFETY: the buffers are sized for this mesh and these levels.
        unsafe {
            self.gpu.module_hooks.density(
                &self.gpu.stream,
                launch(n, LEVEL_BLOCK),
                n as u32,
                per as u32,
                listed,
                &self.eos,
                &state.temp,
                &state.salt,
                &mut state.rho,
            )?;
        }
        Ok(())
    }

    /// The flushed elements' values of inventory field `which` (`q` under
    /// `eta`) into `out` (`MergedCells::pass_through` on every element).
    pub fn pass_through(
        &self,
        s: &Statics3D,
        merged: &Merged3D,
        which: usize,
        q: &DeviceBuffer<f64>,
        eta: &DeviceBuffer<f64>,
        out: &mut DeviceBuffer<f64>,
        end: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if merged.n_flushed == 0 || q.is_empty() {
            return Ok(());
        }
        let (cells, nc) = Self::cells_of(s, which);
        for w in merged.wave_offsets.windows(2) {
            let n = (w[1] - w[0]) * nc;
            // SAFETY: the buffers are sized for this mesh and these levels,
            // the groups for this mesh.
            unsafe {
                self.gpu.module_hooks.merged_pass_through(
                    &self.gpu.stream,
                    launch(n, LEVEL_BLOCK),
                    nc as u32,
                    n as u32,
                    w[0] as u32,
                    u32::from(end),
                    &merged.waves,
                    &merged.flushed_k,
                    &merged.flushed_io,
                    &merged.inflow_offsets,
                    &merged.inflow_from,
                    &merged.inflow_rate,
                    &s.mass,
                    eta,
                    &s.bed,
                    cells,
                    q,
                    out,
                )?;
            }
        }
        Ok(())
    }

    /// Inventories `inv` → values in `out` (the splitter's `values_of`), the
    /// last values where a node has no water from `from` (its stash's
    /// velocities in its masked columns), or, without `from`, `out`'s own
    /// (its stash's); the groups as one cell, the flushed elements passed
    /// through; then `η`, `ū`, `v̄` of `inv`. The density is left to the
    /// caller.
    pub fn values_of(
        &self,
        s: &Statics3D,
        merged: &Merged3D,
        inv: &State3D,
        out: &mut StageValues3D,
        from: Option<&StageValues3D>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.to_values(s, merged, inv, out, from, false)?;
        out.values.copy_barotropic(&self.gpu, inv)
    }

    /// [`Self::values_of`] but `η`, `ū`, `v̄`: at a step's `end` the
    /// flushed elements take their own inventories' values (the splitter's
    /// `to_values`).
    pub fn to_values(
        &self,
        s: &Statics3D,
        merged: &Merged3D,
        inv: &State3D,
        out: &mut StageValues3D,
        from: Option<&StageValues3D>,
        end: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let (m, st) = (&self.gpu.module_hooks, &self.gpu.stream);
        let StageValues3D { values, stash } = out;
        let stash = match from {
            Some(from) => &from.stash,
            None => &*stash,
        };
        for (which, (q, out_field)) in inv
            .fields()
            .into_iter()
            .zip(values.fields_mut())
            .enumerate()
        {
            if q.is_empty() {
                continue;
            }
            let (cells, nc) = Self::cells_of(s, which);
            let n = s.n_columns() * nc;
            let last = from.map_or(&self.none, |from| from.values.fields()[which]);
            let (masked, kept) = match which {
                0 => (&stash.masked, &stash.u),
                1 => (&stash.masked, &stash.v),
                _ => (&self.none_u32, &self.none),
            };
            let grouped = which >= N_MOMENTUM_FIELDS && merged.any();
            let joined = if grouped {
                &merged.joined
            } else {
                &self.none_u32
            };
            // SAFETY: the buffers are sized for this mesh and these levels,
            // the groups for this mesh; `out_field` is no input.
            unsafe {
                m.from_inventory(
                    st,
                    launch(n, LEVEL_BLOCK),
                    nc as u32,
                    n as u32,
                    &inv.eta,
                    &s.bed,
                    cells,
                    q,
                    last,
                    masked,
                    kept,
                    joined,
                    out_field,
                )?;
                if grouped {
                    let n = merged.n_grouped * nc;
                    m.merged_convert(
                        st,
                        launch(n, LEVEL_BLOCK),
                        nc as u32,
                        n as u32,
                        &merged.elements,
                        &merged.group,
                        &merged.joined,
                        &merged.flushed,
                        &s.mass,
                        &inv.eta,
                        &s.bed,
                        cells,
                        q,
                        out_field,
                    )?;
                }
            }
            if grouped {
                self.pass_through(s, merged, which, q, &inv.eta, out_field, end)?;
            }
        }
        Ok(())
    }

    /// `inventory_of` (`all`: every inventory field, else T and S) of
    /// `values` into `inv` under `inv`'s `η`: not the flushed elements' of
    /// the fields with groups.
    pub fn inventories_of(
        &self,
        s: &Statics3D,
        merged: &Merged3D,
        values: &State3D,
        inv: &mut State3D,
        all: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.inventories(
            s,
            merged,
            values,
            inv,
            if all { 0..N_FIELDS } else { 2..4 },
            false,
        )
    }

    /// `dry_inventories` of `values` into `inv` (u, v, k, ψ).
    pub fn dry_inventories(
        &self,
        s: &Statics3D,
        merged: &Merged3D,
        values: &State3D,
        inv: &mut State3D,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.inventories(s, merged, values, inv, 0..2, true)?;
        self.inventories(s, merged, values, inv, 4..N_FIELDS, true)
    }

    fn inventories(
        &self,
        s: &Statics3D,
        merged: &Merged3D,
        values: &State3D,
        inv: &mut State3D,
        fields: std::ops::Range<usize>,
        dry: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let (m, st) = (&self.gpu.module_hooks, &self.gpu.stream);
        let State3D {
            eta,
            u,
            v,
            temp,
            salt,
            tke,
            gls,
            ..
        } = inv;
        let eta: &DeviceBuffer<f64> = eta;
        let targets = [u, v, temp, salt, tke, gls];
        for which in fields {
            let q = &mut *targets[which];
            if q.is_empty() {
                continue;
            }
            let c = values.fields()[which];
            let (cells, nc) = Self::cells_of(s, which);
            let n = s.n_columns() * nc;
            let flushed = if which >= N_MOMENTUM_FIELDS {
                &merged.flushed
            } else {
                &self.none_u32
            };
            // SAFETY: the buffers are sized for this mesh and these levels.
            unsafe {
                let config = launch(n, LEVEL_BLOCK);
                if dry {
                    m.dry_inventories(
                        st, config, nc as u32, n as u32, eta, &s.bed, cells, c, flushed, q,
                    )?;
                } else {
                    m.inventory_of(
                        st, config, nc as u32, n as u32, eta, &s.bed, cells, c, flushed, q,
                    )?;
                }
            }
        }
        Ok(())
    }

    /// The thin columns of `out`'s values at rest, what they held into its
    /// stash (`StageStash::forget`, then `Hydrostatic3D::mask_stage`).
    pub fn mask(
        &self,
        s: &Statics3D,
        out: &mut StageValues3D,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let n = s.n_columns() * s.n_levels;
        let StageValues3D { values, stash } = out;
        // SAFETY: the buffers are sized for this mesh and these levels.
        unsafe {
            self.gpu.module_hooks.mask_thin(
                &self.gpu.stream,
                launch(n, LEVEL_BLOCK),
                s.n_levels as u32,
                n as u32,
                self.min_column_depth,
                &values.eta,
                &s.bed,
                &mut values.u,
                &mut values.v,
                &mut values.ubar,
                &mut values.vbar,
                &mut stash.masked,
                &mut stash.u,
                &mut stash.v,
                &mut stash.ubar,
                &mut stash.vbar,
            )?;
        }
        Ok(())
    }

    /// The masked columns of `out`'s values back (`StageStash::restore_into`
    /// on every element).
    pub fn restore(
        &self,
        s: &Statics3D,
        out: &mut StageValues3D,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let n = s.n_columns() * s.n_levels;
        let StageValues3D { values, stash } = out;
        // SAFETY: the buffers are sized for this mesh and these levels.
        unsafe {
            self.gpu.module_hooks.restore_thin(
                &self.gpu.stream,
                launch(n, LEVEL_BLOCK),
                s.n_levels as u32,
                n as u32,
                &stash.masked,
                &stash.u,
                &stash.v,
                &stash.ubar,
                &stash.vbar,
                &mut values.u,
                &mut values.v,
                &mut values.ubar,
                &mut values.vbar,
            )?;
        }
        Ok(())
    }

    /// A stage's values from its inventories `inv` and its RHS argument's
    /// values `from` (the splitter's `stage_values`): [`Self::values_of`],
    /// the tracer limiter (`post_stage`, but at the step's `end`: then
    /// `end_density`), the density, the limited T and S back into `inv`,
    /// the dry nodes' inventories of u, v, k, ψ from their last values, and
    /// the thin columns masked for the next RHS.
    pub fn stage_values(
        &mut self,
        s: &Statics3D,
        merged: &Merged3D,
        inv: &mut State3D,
        out: &mut StageValues3D,
        from: Option<&StageValues3D>,
        end: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.values_of(s, merged, inv, out, from)?;
        if let (false, Some(limiter)) = (end, &mut self.limiter) {
            limiter.apply(s, &mut out.values)?;
        }
        self.density(s, &mut out.values, None)?;
        self.inventories_of(s, merged, &out.values, inv, false)?;
        self.dry_inventories(s, merged, &out.values, inv)?;
        self.mask(s, out)
    }

    /// `Hydrostatic3D::post_process`'s tracer limiter and density of a
    /// step's end state `values` (its `w` is the caller's).
    pub fn post_process(
        &mut self,
        s: &Statics3D,
        values: &mut State3D,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(limiter) = &mut self.limiter {
            limiter.apply(s, values)?;
        }
        self.density(s, values, None)
    }

    /// The momentum tendencies `rhs_u`, `rhs_v` of the columns thin under
    /// `eta` to zero (`zero_thin_momentum`).
    pub fn zero_thin_momentum(
        &self,
        s: &Statics3D,
        eta: &DeviceBuffer<f64>,
        rhs_u: &mut DeviceBuffer<f64>,
        rhs_v: &mut DeviceBuffer<f64>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let n = s.n_columns() * s.n_levels;
        // SAFETY: the buffers are sized for this mesh and these levels.
        unsafe {
            self.gpu.module_hooks.zero_thin(
                &self.gpu.stream,
                launch(n, LEVEL_BLOCK),
                s.n_levels as u32,
                n as u32,
                self.min_column_depth,
                eta,
                &s.bed,
                rhs_u,
                rhs_v,
            )?;
        }
        Ok(())
    }

    /// The nesting band's relaxation of the shear of `(u, v)` under `eta`
    /// to the parent's columns of `band`, added to the velocity tendencies
    /// `rhs_u`, `rhs_v` (`NestingColumns::relax_shear`, the thin columns
    /// skipped); nothing if the parent supplies no velocity.
    pub fn relax_shear(
        &self,
        s: &Statics3D,
        band: &Band3D,
        eta: &DeviceBuffer<f64>,
        u: &DeviceBuffer<f64>,
        v: &DeviceBuffer<f64>,
        rhs_u: &mut DeviceBuffer<f64>,
        rhs_v: &mut DeviceBuffer<f64>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if !band.supplied.velocity || band.n_slots == 0 {
            return Ok(());
        }
        // SAFETY: the buffers are sized for this mesh and these levels, the
        // band's for its slots.
        unsafe {
            self.gpu.module_hooks.relax_shear(
                &self.gpu.stream,
                launch(band.n_slots, LEVEL_BLOCK),
                s.n_levels as u32,
                band.n_slots as u32,
                self.min_column_depth,
                &band.nodes,
                &band.velocity_rate,
                eta,
                &s.bed,
                &s.d_sigma,
                &band.values[0],
                &band.values[1],
                u,
                v,
                rhs_u,
                rhs_v,
            )?;
        }
        Ok(())
    }

    /// The nesting band's relaxation of T and S under `eta` to the parent's
    /// columns of `band`, added to their inventory tendencies `rhs_t`,
    /// `rhs_s` (`NestingColumns::relax_tracers`, the thin columns skipped);
    /// nothing if the parent supplies no tracers.
    pub fn relax_tracers(
        &self,
        s: &Statics3D,
        band: &Band3D,
        eta: &DeviceBuffer<f64>,
        temp: &DeviceBuffer<f64>,
        salt: &DeviceBuffer<f64>,
        rhs_t: &mut DeviceBuffer<f64>,
        rhs_s: &mut DeviceBuffer<f64>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if !band.supplied.tracers || band.n_slots == 0 {
            return Ok(());
        }
        // SAFETY: as `relax_shear`.
        unsafe {
            self.gpu.module_hooks.relax_tracers(
                &self.gpu.stream,
                launch(band.n_slots, LEVEL_BLOCK),
                s.n_levels as u32,
                band.n_slots as u32,
                self.min_column_depth,
                &band.nodes,
                &band.tracer_rate,
                eta,
                &s.bed,
                &s.d_sigma,
                &band.values[2],
                &band.values[3],
                temp,
                salt,
                rhs_t,
                rhs_s,
            )?;
        }
        Ok(())
    }
}

//! A parent model's columns on the GPU (P2.7 Phase 4.5k, `notes/gpu.md`):
//! the nesting band's parent (`OceanModelColumns`: NorKyst at Frøya, Bodø,
//! Mausund) sampled on the device, where 4.5c gathered the band's columns,
//! evaluated `Nesting3D::columns` on the host and uploaded them at every
//! stage.
//!
//! What a slot samples is static and goes up once, from the parent's device
//! form (`DeviceParent3D::Model`, dgcom's `ParentModel3D`): each slot's
//! horizontal stencil, east axis and parent bed (`ParentNode`), the profiles
//! of the parent points the stencils use at every snapshot (u, v, T, S,
//! `[time][point][level]` f32 as the parent's own, with their z-levels or
//! the points' s-level fractions), and the deep reference's table. Per call
//! the host turns the stage times of the listed elements into a few
//! *moments* (a time and its snapshots' weights, the parent's
//! `ParentModel3D::moment`), passed as a kernel argument, and every slot's
//! code (its moment, or not listed), kept on the device by content: on
//! global steps one buffer, on local steps one per kind of pass.
//!
//! Two kernels then do what `Nesting3D::columns` does per slot:
//!
//! - `parent_check`, a thread per slot: the cache by time and `η` (a slot
//!   evaluated at the same time and free surface keeps its column, as the
//!   CPU's does), and whether the parent covers the node
//!   (`profile_covers` of each field it supplies);
//! - `parent_columns`, a thread per slot and level: at the child's layer
//!   centre, dgcom's `profile_sample_at` of each field, the rotation to the
//!   mesh axes, `deep_tracers` below the parent's bed; where the parent does
//!   not cover the node, the child's own column (at rest in a thin column).
//!
//! The columns land in the band's buffers ([`Band3D`]), and from there in
//! the open faces' exterior values. The arithmetic is dgcom's own functions
//! on the same inputs, so the bits are the CPU's (`--no-fmad`).
//!
//! That kernel was the largest of a nested run (22.7 ms a stage on Frøya's
//! 20,224 slots, a fifth of the run, 2026-11-25): every thread walks the
//! parent's levels once per field, point and snapshot. Where the parent's
//! valid levels do not change in time and deepen with their index (a model's
//! bed mask: `steady_field` checks it when the tables go up), two kernels
//! take its place, the device's own form of the same sampling (the GPU may
//! lead, `notes/gpu.md`):
//!
//! - `parent_moments`, a thread per parent value and moment: the snapshots
//!   weighed once, for every slot, level and field that reads the value;
//! - `parent_columns_steady`, a thread per slot: each point of its stencil
//!   walked once for all the child's levels, the level spacings'
//!   reciprocals tabled.
//!
//! They differ from `profile_sample_at` by round-off (the weights before the
//! interpolation in depth, a product for a division): 6e-16 of a field's
//! largest value on the gate's parents, which holds them within 1e-12
//! (`parent3d`), beside the general kernel bit for bit.

// As `three_d.rs`: the device loops index arrays by the loop variable.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use cuda_core::DeviceBuffer;
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;
use dgcom::boundary::{Nesting3D, ParentModel3D, Supplied, deep_tracers};
use dgcom::io::{ProfileLevels, ProfileSeries, Profiles, TimeStencil};
use dgcom::io::{profile_covers, profile_sample_at};
use dgcom::solver::Blocks;
use dgcom::time::StageTime;

use crate::hooks::{Band3D, State3D};
use crate::swe2d::{Gpu, NN};
use crate::three_d::{COLUMN_BLOCK, LEVEL_BLOCK, MAX_LEVELS, Statics3D, launch};

/// Moments a launch takes ([`Moments`]); more go in several launches.
pub const MAX_MOMENTS: usize = 8;

/// A slot's code when its element is not listed.
const NOT_LISTED: u32 = u32::MAX;

/// What `parent_check` found for a slot.
const SKIP: u32 = 0;
const COVERED: u32 = 1;
const OWN: u32 = 2;

/// One field's profiles in the parent tables: where its values and level
/// positions start, its points and levels, and how its levels lie.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct FieldForm {
    /// Whether the parent has the field
    pub present: u32,
    /// Whether the levels are fractions of the column (s-levels)
    pub fractions: u32,
    pub n_points: u32,
    pub n_levels: u32,
    /// Its first value (`[time][point][level]`) and level position
    /// (`[level]`, or `[point][level]` for fractions)
    pub values: u64,
    pub positions: u64,
}

/// The parent and the band as both kernels take them (by value).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ParentArgs {
    /// u, v (east, north), T, S
    pub fields: [FieldForm; 4],
    /// What the parent supplies
    pub velocity: u32,
    pub tracers: u32,
    /// Whether a deep reference acts below the parent's bed, its table's
    /// lowest height and its decay depth
    pub deep: u32,
    pub z0: f64,
    pub decay: f64,
    /// The band's slots and the child's levels
    pub n: u32,
    pub nl: u32,
    /// Columns shallower than this are thin
    pub min_column_depth: f64,
}

/// Up to [`MAX_MOMENTS`] moments of one launch: the slots with codes
/// `base..base + count` take moment `code − base`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Moments {
    pub base: u32,
    pub count: u32,
    /// The stage time (s) and its snapshots' `TimeStencil`
    pub time: [f64; MAX_MOMENTS],
    pub len: [u32; MAX_MOMENTS],
    pub idx: [[u32; 4]; MAX_MOMENTS],
    pub w: [[f64; 4]; MAX_MOMENTS],
}

impl Moments {
    /// Slot code `code`'s moment in this launch, if it has one.
    #[inline]
    fn of(&self, code: u32) -> Option<usize> {
        (code != NOT_LISTED && code >= self.base && code - self.base < self.count)
            .then(|| (code - self.base) as usize)
    }

    /// Moment `m`'s snapshots and weights.
    #[inline]
    fn stencil(&self, m: usize) -> TimeStencil {
        let idx = self.idx[m];
        TimeStencil {
            idx: [
                idx[0] as usize,
                idx[1] as usize,
                idx[2] as usize,
                idx[3] as usize,
            ],
            w: self.w[m],
            len: self.len[m] as usize,
        }
    }
}

/// A field's profiles in the parent tables, as dgcom's sampling reads them.
#[derive(Clone, Copy)]
pub struct FieldView<'a> {
    values: &'a [f32],
    positions: &'a [f64],
    form: FieldForm,
}

impl Profiles for FieldView<'_> {
    #[inline]
    fn n_levels(&self) -> usize {
        self.form.n_levels as usize
    }

    #[inline]
    fn value(&self, t: usize, k: usize, l: usize) -> f32 {
        let (np, nl) = (self.form.n_points as usize, self.form.n_levels as usize);
        self.values[self.form.values as usize + (t * np + k) * nl + l]
    }

    #[inline]
    fn position(&self, k: usize, l: usize) -> f64 {
        let at = if self.form.fractions != 0 {
            k * self.form.n_levels as usize + l
        } else {
            l
        };
        self.positions[self.form.positions as usize + at]
    }

    #[inline]
    fn fractions(&self) -> bool {
        self.form.fractions != 0
    }
}

/// Field `f`'s view of the parent tables.
#[inline]
fn field<'a>(
    args: &ParentArgs,
    f: usize,
    values: &'a [f32],
    positions: &'a [f64],
) -> FieldView<'a> {
    FieldView {
        values,
        positions,
        form: args.fields[f],
    }
}

/// The slot code of `slot` (an empty `codes`: every slot listed, at the
/// first moment).
#[inline]
fn code_of(codes: &[u32], slot: usize) -> u32 {
    if codes.is_empty() { 0 } else { codes[slot] }
}

/// Where the steady sampling's tables lie ([`Steady`]): each field's first
/// value in a moment's weighed profiles (past every value for a field the
/// parent lacks) and a moment's length; each field's first entry in the
/// tables per point and level (`inv_dz`) and per point (`first_valid`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct SteadyForm {
    pub offsets: [u32; 4],
    pub per_moment: u32,
    pub tables: [u32; 4],
    pub firsts: [u32; 4],
}

/// Where point `k`'s level `l` of the field `form` lies (a depth, or a
/// fraction of the column), in the parent tables' `positions`.
#[inline(always)]
fn level_position(form: &FieldForm, positions: &[f64], k: usize, l: usize) -> f64 {
    let at = if form.fractions != 0 {
        k * form.n_levels as usize + l
    } else {
        l
    };
    positions[form.positions as usize + at]
}

// =============================================================================
// Kernels
// =============================================================================

#[cuda_module]
pub mod kernels_parent {
    use super::*;

    /// A thread per slot: whether its column is evaluated now and how. A
    /// slot of this launch's moments whose last evaluation was at the same
    /// time and free surface `η` is skipped (the CPU's cache); the others
    /// record their time and `η`, and are [`COVERED`] where the parent
    /// covers the node (a stencil, and `profile_covers` of every field it
    /// supplies), else take the child's own column ([`OWN`]).
    #[kernel]
    pub fn parent_check(
        args: ParentArgs,
        moments: Moments,
        codes: &[u32],
        nodes: &[u32],
        eta: &[f64],
        has: &[u32],
        idx: &[[u32; 4]],
        w: &[[f64; 4]],
        values: &[f32],
        positions: &[f64],
        mut cache_time: DisjointSlice<f64>,
        mut cache_eta: DisjointSlice<f64>,
        mut flags: DisjointSlice<u32>,
    ) {
        let slot = thread::index_1d().get();
        if slot >= args.n as usize {
            return;
        }
        // SAFETY: thread slot alone reads and writes slot `slot` (< n).
        unsafe {
            let flag = flags.get_unchecked_mut(slot);
            let Some(m) = moments.of(code_of(codes, slot)) else {
                *flag = SKIP;
                return;
            };
            let (t, e) = (moments.time[m], eta[nodes[slot] as usize]);
            let (last_t, last_e) = (
                cache_time.get_unchecked_mut(slot),
                cache_eta.get_unchecked_mut(slot),
            );
            if *last_t == t && *last_e == e {
                *flag = SKIP;
                return;
            }
            *last_t = t;
            *last_e = e;
            let time = moments.stencil(m);
            let covers = |f: usize| {
                args.fields[f].present != 0
                    && profile_covers(
                        &field(&args, f, values, positions),
                        &time,
                        &idx[slot],
                        &w[slot],
                    )
            };
            let covered = has[slot] != 0
                && (args.velocity == 0 || (covers(0) && covers(1)))
                && (args.tracers == 0 || (covers(2) && covers(3)));
            *flag = if covered { COVERED } else { OWN };
        }
    }

    /// A thread per slot and level: the column the slot's flag asks for at
    /// that level, into the band's `out_*` (`[slot][level]`). [`COVERED`]:
    /// the layer centre's depth `−σ_l max(η − B, 0)`, each supplied field's
    /// `profile_sample_at` there, the velocity rotated from east and north
    /// to the mesh axes (`east`), the tracers below the parent's bed
    /// `deep_tracers`; [`OWN`]: the child's own values, the velocity at rest
    /// in a thin column.
    #[kernel]
    pub fn parent_columns(
        args: ParentArgs,
        moments: Moments,
        codes: &[u32],
        flags: &[u32],
        nodes: &[u32],
        eta: &[f64],
        bed: &[f64],
        sigma_rho: &[f64],
        idx: &[[u32; 4]],
        w: &[[f64; 4]],
        east_bed: &[[f64; 3]],
        values: &[f32],
        positions: &[f64],
        deep_temp: &[f64],
        deep_salt: &[f64],
        u: &[f64],
        v: &[f64],
        temp: &[f64],
        salt: &[f64],
        mut out_u: DisjointSlice<f64>,
        mut out_v: DisjointSlice<f64>,
        mut out_t: DisjointSlice<f64>,
        mut out_s: DisjointSlice<f64>,
    ) {
        let i = thread::index_1d().get();
        let nl = args.nl as usize;
        if i >= args.n as usize * nl {
            return;
        }
        let (slot, l) = (i / nl, i % nl);
        let flag = flags[slot];
        if flag == SKIP {
            return;
        }
        let node = nodes[slot] as usize;
        let (e, b) = (eta[node], bed[node]);
        // SAFETY: thread i alone writes value i (= slot·nl + l).
        unsafe {
            if flag == OWN {
                let at = node * nl + l;
                let thin = e - b < args.min_column_depth;
                *out_u.get_unchecked_mut(i) = if thin { 0.0 } else { u[at] };
                *out_v.get_unchecked_mut(i) = if thin { 0.0 } else { v[at] };
                *out_t.get_unchecked_mut(i) = temp[at];
                *out_s.get_unchecked_mut(i) = salt[at];
                return;
            }
            let m = moments.of(code_of(codes, slot)).unwrap_or(0);
            let time = moments.stencil(m);
            let column_depth = (e - b).max(0.0);
            let d = -sigma_rho[l] * column_depth;
            let (idx, w) = (&idx[slot], &w[slot]);
            let sample = |f: usize| {
                profile_sample_at(
                    &field(&args, f, values, positions),
                    &time,
                    idx,
                    w,
                    d,
                    column_depth,
                )
            };
            let [c, s, h] = east_bed[slot];
            if args.velocity != 0 {
                let (east, north) = (sample(0), sample(1));
                *out_u.get_unchecked_mut(i) = east * c - north * s;
                *out_v.get_unchecked_mut(i) = east * s + north * c;
            }
            if args.tracers != 0 {
                let (mut t, mut sa) = (sample(2), sample(3));
                if args.deep != 0 && h.is_finite() {
                    (t, sa) =
                        deep_tracers((args.z0, deep_temp, deep_salt), args.decay, h, d, t, sa);
                }
                *out_t.get_unchecked_mut(i) = t;
                *out_s.get_unchecked_mut(i) = sa;
            }
        }
    }

    /// The profiles of the parent's points at a launch's moments, a thread
    /// per moment and value (`[moment][field][point][level]`, each field at
    /// `steady.offsets` within a moment): the snapshots of the moment's
    /// time stencil weighed, as `profile_sample_at` weighs them after its
    /// interpolation in depth. Where every level is valid at all snapshots
    /// or at none (`steady_field`) the two orders are the same sum but for
    /// round-off, and one weighing per parent value serves every slot,
    /// level and field that reads it.
    #[kernel]
    pub fn parent_moments(
        args: ParentArgs,
        steady: SteadyForm,
        moments: Moments,
        values: &[f32],
        mut out: DisjointSlice<f64>,
    ) {
        let i = thread::index_1d().get();
        let per = steady.per_moment as usize;
        if i >= moments.count as usize * per {
            return;
        }
        let (m, r) = (i / per, i % per);
        // The field whose profiles hold r (an absent field's offset is past
        // every value)
        let mut f = 0;
        #[unroll]
        for g in 0..4 {
            if r >= steady.offsets[g] as usize {
                f = g;
            }
        }
        let form = args.fields[f];
        let j = r - steady.offsets[f] as usize;
        let per_time = form.n_points as usize * form.n_levels as usize;
        let mut sum = 0.0;
        for s in 0..moments.len[m] as usize {
            let at = form.values as usize + moments.idx[m][s] as usize * per_time + j;
            sum += moments.w[m][s] * values[at] as f64;
        }
        // SAFETY: thread i alone writes value i.
        unsafe {
            *out.get_unchecked_mut(i) = sum;
        }
    }

    /// [`parent_columns`] where the parent's valid levels do not change in
    /// time (`steady_field`), a thread per slot: each supplied field from
    /// the moment's weighed profiles (`weighted`, [`parent_moments`]), every
    /// point of the slot's stencil walked once for all the child's levels
    /// (their depths in order, so the bracketing parent levels only move
    /// down), the level spacings' reciprocals tabled (`inv_dz`, NaN at a
    /// level that is not valid; `first_valid` a point's first valid level,
    /// `u32::MAX` without one). The same sampling as `profile_sample_at`
    /// in another order of sums: the device's own form (`notes/gpu.md`,
    /// where the backends differ).
    #[kernel]
    pub fn parent_columns_steady(
        args: ParentArgs,
        steady: SteadyForm,
        moments: Moments,
        codes: &[u32],
        flags: &[u32],
        nodes: &[u32],
        eta: &[f64],
        bed: &[f64],
        sigma_rho: &[f64],
        idx: &[[u32; 4]],
        w: &[[f64; 4]],
        east_bed: &[[f64; 3]],
        weighted: &[f64],
        positions: &[f64],
        inv_dz: &[f64],
        first_valid: &[u32],
        deep_temp: &[f64],
        deep_salt: &[f64],
        u: &[f64],
        v: &[f64],
        temp: &[f64],
        salt: &[f64],
        mut out_u: DisjointSlice<f64>,
        mut out_v: DisjointSlice<f64>,
        mut out_t: DisjointSlice<f64>,
        mut out_s: DisjointSlice<f64>,
    ) {
        let slot = thread::index_1d().get();
        let nl = args.nl as usize;
        if slot >= args.n as usize {
            return;
        }
        let flag = flags[slot];
        if flag == SKIP {
            return;
        }
        let node = nodes[slot] as usize;
        let (e, b) = (eta[node], bed[node]);
        // SAFETY: thread `slot` alone writes its column, values
        // `slot·nl .. (slot + 1)·nl` of each output.
        unsafe {
            if flag == OWN {
                let thin = e - b < args.min_column_depth;
                for l in 0..nl {
                    let (i, at) = (slot * nl + l, node * nl + l);
                    *out_u.get_unchecked_mut(i) = if thin { 0.0 } else { u[at] };
                    *out_v.get_unchecked_mut(i) = if thin { 0.0 } else { v[at] };
                    *out_t.get_unchecked_mut(i) = temp[at];
                    *out_s.get_unchecked_mut(i) = salt[at];
                }
                return;
            }
            let m = moments.of(code_of(codes, slot)).unwrap_or(0);
            let column_depth = (e - b).max(0.0);
            let (idx, w) = (&idx[slot], &w[slot]);
            // The child's levels from the surface down
            let surface_first = sigma_rho[0] > sigma_rho[nl - 1];
            let mut acc = [0.0f64; MAX_LEVELS];
            for f in 0..4 {
                let supplied = if f < 2 {
                    args.velocity != 0
                } else {
                    args.tracers != 0
                };
                if !supplied {
                    continue;
                }
                let form = args.fields[f];
                let nlp = form.n_levels as usize;
                let fractions = form.fractions != 0;
                let base = m * steady.per_moment as usize + steady.offsets[f] as usize;
                let (table, firsts) = (steady.tables[f] as usize, steady.firsts[f] as usize);
                let (mut total, mut first) = (0.0, true);
                for i in 0..4 {
                    let (k, wk) = (idx[i] as usize, w[i]);
                    if wk == 0.0 {
                        continue;
                    }
                    let start = first_valid[firsts + k];
                    if start == u32::MAX {
                        continue;
                    }
                    total += wk;
                    // The first valid level not above the target, and the
                    // valid one before it
                    let mut p = start as usize;
                    let (mut above_z, mut above_v, mut has_above) = (0.0, 0.0, false);
                    for step in 0..nl {
                        let l = if surface_first { step } else { nl - 1 - step };
                        let d = -sigma_rho[l] * column_depth;
                        let target = if fractions {
                            d / column_depth.max(f64::MIN_POSITIVE)
                        } else {
                            d
                        };
                        while p < nlp {
                            if !inv_dz[table + k * nlp + p].is_nan() {
                                let z = level_position(&form, positions, k, p);
                                if z >= target {
                                    break;
                                }
                                above_z = z;
                                above_v = weighted[base + k * nlp + p];
                                has_above = true;
                            }
                            p += 1;
                        }
                        let value = if p < nlp {
                            let below = weighted[base + k * nlp + p];
                            if has_above && level_position(&form, positions, k, p) > above_z {
                                above_v
                                    + (below - above_v)
                                        * ((target - above_z) * inv_dz[table + k * nlp + p])
                            } else {
                                below
                            }
                        } else {
                            above_v
                        };
                        acc[l] = if first {
                            wk * value
                        } else {
                            acc[l] + wk * value
                        };
                    }
                    first = false;
                }
                let scale = if first { f64::NAN } else { 1.0 / total };
                for l in 0..nl {
                    let (i, x) = (slot * nl + l, acc[l] * scale);
                    match f {
                        0 => *out_u.get_unchecked_mut(i) = x,
                        1 => *out_v.get_unchecked_mut(i) = x,
                        2 => *out_t.get_unchecked_mut(i) = x,
                        _ => *out_s.get_unchecked_mut(i) = x,
                    }
                }
            }
            // East and north to the mesh axes; the tracers below the
            // parent's bed
            let [c, s, h] = east_bed[slot];
            for l in 0..nl {
                let i = slot * nl + l;
                if args.velocity != 0 {
                    let (east, north) = (*out_u.get_unchecked_mut(i), *out_v.get_unchecked_mut(i));
                    *out_u.get_unchecked_mut(i) = east * c - north * s;
                    *out_v.get_unchecked_mut(i) = east * s + north * c;
                }
                if args.tracers != 0 && args.deep != 0 && h.is_finite() {
                    let d = -sigma_rho[l] * column_depth;
                    let (t, sa) = (*out_t.get_unchecked_mut(i), *out_s.get_unchecked_mut(i));
                    let (t, sa) =
                        deep_tracers((args.z0, deep_temp, deep_salt), args.decay, h, d, t, sa);
                    *out_t.get_unchecked_mut(i) = t;
                    *out_s.get_unchecked_mut(i) = sa;
                }
            }
        }
    }
}

// =============================================================================
// Host
// =============================================================================

/// The slot codes of the calls so far, on the device by content.
struct Codes {
    entries: Vec<(Vec<u32>, DeviceBuffer<u32>, u64)>,
    clock: u64,
    none: DeviceBuffer<u32>,
}

/// The slot codes kept ([`Codes`]): a local coarse step's passes repeat.
const MAX_CODES: usize = 96;

impl Codes {
    /// The device's buffer of `codes` (uploaded unless kept: the copy waits
    /// for the queued kernels); `None`, every slot at the first moment, is
    /// the empty buffer.
    fn of(&mut self, gpu: &Gpu, codes: Option<&[u32]>) -> &DeviceBuffer<u32> {
        let Some(codes) = codes else {
            return &self.none;
        };
        self.clock += 1;
        let i = match self.entries.iter().position(|(host, ..)| host == codes) {
            Some(i) => i,
            None => {
                let buffer = DeviceBuffer::from_host(&gpu.stream, codes).expect("slot codes");
                let entry = (codes.to_vec(), buffer, self.clock);
                if self.entries.len() < MAX_CODES {
                    self.entries.push(entry);
                    self.entries.len() - 1
                } else {
                    let oldest = (0..self.entries.len())
                        .min_by_key(|&i| self.entries[i].2)
                        .expect("a code buffer is kept");
                    self.entries[oldest] = entry;
                    oldest
                }
            }
        };
        self.entries[i].2 = self.clock;
        &self.entries[i].1
    }
}

/// The steady sampling's tables on the device ([`steady_field`]): a
/// parent whose levels are valid at every snapshot or at none, deepening
/// with their index, is sampled by `parent_moments` and
/// `parent_columns_steady`.
struct Steady {
    form: SteadyForm,
    /// The weighed profiles of a launch's moments
    /// (`[moment][field][point][level]`), and the times they were made for
    weighted: DeviceBuffer<f64>,
    made_for: Vec<f64>,
    /// `1/(z_l − z_above)` of every point's level (0 at its first valid
    /// level and where the two coincide, NaN where the level is not valid),
    /// and every point's first valid level
    inv_dz: DeviceBuffer<f64>,
    first_valid: DeviceBuffer<u32>,
}

/// One field's part of the steady tables: every point's levels' `inv_dz`
/// and its first valid level (`u32::MAX` without one), from `form` in the
/// tables `values` (of `n_times` snapshots) and `positions`. `None` where
/// the steady sampling would not be `profile_sample_at`'s: a level valid at
/// one snapshot and not at another, a valid value at a position that is not
/// finite, or valid levels that do not deepen with their index.
fn steady_field(
    form: &FieldForm,
    n_times: usize,
    values: &[f32],
    positions: &[f64],
) -> Option<(Vec<f64>, Vec<u32>)> {
    let (np, nl) = (form.n_points as usize, form.n_levels as usize);
    let value = |t: usize, k: usize, l: usize| values[form.values as usize + (t * np + k) * nl + l];
    let mut inv_dz = vec![f64::NAN; np * nl];
    let mut first_valid = vec![u32::MAX; np];
    for k in 0..np {
        let mut above: Option<f64> = None;
        for l in 0..nl {
            let valid = value(0, k, l).is_finite();
            if (1..n_times).any(|t| value(t, k, l).is_finite() != valid) {
                return None;
            }
            if !valid {
                continue;
            }
            let z = level_position(form, positions, k, l);
            if !z.is_finite() {
                return None;
            }
            inv_dz[k * nl + l] = match above {
                Some(z0) if z < z0 => return None,
                Some(z0) if z > z0 => 1.0 / (z - z0),
                Some(_) => 0.0,
                None => {
                    first_valid[k] = l as u32;
                    0.0
                }
            };
            above = Some(z);
        }
    }
    Some((inv_dz, first_valid))
}

/// A parent model on the device (see the module docs).
pub struct Parent3D {
    args: ParentArgs,
    /// Every slot's node, whether the parent covers it, its stencil (in the
    /// tables' points), east axis and parent bed `[cos, sin, h]`
    nodes: DeviceBuffer<u32>,
    has: DeviceBuffer<u32>,
    idx: DeviceBuffer<[u32; 4]>,
    w: DeviceBuffer<[f64; 4]>,
    east_bed: DeviceBuffer<[f64; 3]>,
    /// The four fields' profiles and level positions
    values: DeviceBuffer<f32>,
    positions: DeviceBuffer<f64>,
    /// The deep reference's T and S (one zero each without one)
    deep_temp: DeviceBuffer<f64>,
    deep_salt: DeviceBuffer<f64>,
    /// Every slot's last time and `η`, and what `parent_check` found
    cache_time: DeviceBuffer<f64>,
    cache_eta: DeviceBuffer<f64>,
    flags: DeviceBuffer<u32>,
    codes: Codes,
    /// The steady sampling's tables, where the parent allows it
    steady: Option<Steady>,
    /// Every slot's element, and scratch for a call's codes, times and
    /// listed elements
    elements: Vec<usize>,
    host_codes: Vec<u32>,
    times: Vec<f64>,
    listed: Vec<bool>,
    /// Bytes on the device
    pub bytes: usize,
}

impl Parent3D {
    /// The parent `model` of `nesting`'s band on `statics`' levels; columns
    /// shallower than `min_column_depth` are thin.
    pub fn new(
        gpu: &Gpu,
        statics: &Statics3D,
        nesting: &Nesting3D,
        model: &dyn ParentModel3D,
        min_column_depth: f64,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let st = &gpu.stream;
        let slots = nesting.slots();
        let n = slots.nodes.len();
        let Supplied { velocity, tracers } = slots.supplied;
        // Every slot's sampling, and the parent points its stencil reads
        // (the points of weight zero are never read: point 0)
        let profiles = model.profiles();
        let n_parent = profiles
            .iter()
            .flatten()
            .map(|p| p.n_points())
            .next()
            .unwrap_or(0);
        let mut point_of = vec![u32::MAX; n_parent];
        let mut used: Vec<usize> = Vec::new();
        let (mut has, mut idx, mut w, mut east_bed) = (
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
        );
        for (&node, &position) in slots.nodes.iter().zip(slots.positions) {
            match model.node(node, position) {
                Some(p) => {
                    let mut compact = [0u32; 4];
                    for i in 0..4 {
                        if p.stencil.w[i] == 0.0 {
                            continue;
                        }
                        let k = p.stencil.idx[i] as usize;
                        if point_of[k] == u32::MAX {
                            point_of[k] = used.len() as u32;
                            used.push(k);
                        }
                        compact[i] = point_of[k];
                    }
                    has.push(1u32);
                    idx.push(compact);
                    w.push(p.stencil.w);
                    east_bed.push([p.east.0, p.east.1, p.bed]);
                }
                None => {
                    has.push(0);
                    idx.push([0; 4]);
                    w.push([0.0; 4]);
                    east_bed.push([1.0, 0.0, f64::NAN]);
                }
            }
        }
        // The fields' profiles at the used points, every snapshot
        let (mut values, mut positions) = (Vec::<f32>::new(), Vec::<f64>::new());
        let mut fields = [FieldForm::default(); 4];
        // The steady sampling's tables, while every field allows it
        let mut steady = Some((SteadyForm::default(), Vec::<f64>::new(), Vec::<u32>::new()));
        let mut per_moment = 0usize;
        for (f, (form, series)) in fields.iter_mut().zip(profiles).enumerate() {
            if let Some((tables, ..)) = &mut steady {
                tables.offsets[f] = u32::MAX;
            }
            let Some(series) = series else { continue };
            *form = upload_form(series, &used, &mut values, &mut positions)?;
            let field = steady_field(form, series.n_times(), &values, &positions);
            steady = match (steady, field) {
                (Some((mut tables, mut inv_dz, mut first_valid)), Some((inv, first))) => {
                    tables.offsets[f] = u32::try_from(per_moment)?;
                    tables.tables[f] = u32::try_from(inv_dz.len())?;
                    tables.firsts[f] = u32::try_from(first_valid.len())?;
                    per_moment += inv.len();
                    inv_dz.extend(inv);
                    first_valid.extend(first);
                    Some((tables, inv_dz, first_valid))
                }
                _ => None,
            };
        }
        let steady = match steady {
            Some((mut form, inv_dz, first_valid))
                if per_moment > 0 && statics.n_levels <= MAX_LEVELS =>
            {
                form.per_moment = u32::try_from(per_moment)?;
                Some(Steady {
                    form,
                    weighted: DeviceBuffer::zeroed(st, MAX_MOMENTS * per_moment)?,
                    made_for: Vec::new(),
                    inv_dz: DeviceBuffer::from_host(st, &inv_dz)?,
                    first_valid: DeviceBuffer::from_host(st, &first_valid)?,
                })
            }
            _ => None,
        };
        let deep = model.deep_reference();
        let (z0, deep_temp, deep_salt, decay) = match deep {
            Some(deep) => {
                let (z0, t, s) = deep.profile.table();
                (z0, t.to_vec(), s.to_vec(), deep.decay)
            }
            None => (0.0, vec![0.0], vec![0.0], 1.0),
        };
        let args = ParentArgs {
            fields,
            velocity: u32::from(velocity),
            tracers: u32::from(tracers),
            deep: u32::from(deep.is_some()),
            z0,
            decay,
            n: n as u32,
            nl: statics.n_levels as u32,
            min_column_depth,
        };
        let nodes: Vec<u32> = slots.nodes.iter().map(|&k| k as u32).collect();
        let padded = |v: Vec<f32>| if v.is_empty() { vec![0.0] } else { v };
        let values = padded(values);
        let positions = if positions.is_empty() {
            vec![0.0]
        } else {
            positions
        };
        let bytes = values.len() * 4
            + positions.len() * 8
            + steady.as_ref().map_or(0, |s| {
                (s.weighted.len() + s.inv_dz.len()) * 8 + s.first_valid.len() * 4
            })
            + n * (4 + 4 + 16 + 32 + 24 + 8 + 8 + 4)
            + (deep_temp.len() + deep_salt.len()) * 8;
        Ok(Self {
            args,
            nodes: DeviceBuffer::from_host(st, &nodes)?,
            has: DeviceBuffer::from_host(st, &has)?,
            idx: DeviceBuffer::from_host(st, &idx)?,
            w: DeviceBuffer::from_host(st, &w)?,
            east_bed: DeviceBuffer::from_host(st, &east_bed)?,
            values: DeviceBuffer::from_host(st, &values)?,
            positions: DeviceBuffer::from_host(st, &positions)?,
            deep_temp: DeviceBuffer::from_host(st, &deep_temp)?,
            deep_salt: DeviceBuffer::from_host(st, &deep_salt)?,
            cache_time: DeviceBuffer::from_host(st, &vec![f64::NAN; n])?,
            cache_eta: DeviceBuffer::from_host(st, &vec![f64::NAN; n])?,
            flags: DeviceBuffer::zeroed(st, n)?,
            codes: Codes {
                entries: Vec::new(),
                clock: 0,
                none: DeviceBuffer::zeroed(st, 0)?,
            },
            steady,
            elements: slots.nodes.iter().map(|&k| k / NN).collect(),
            host_codes: vec![0; n],
            times: Vec::new(),
            listed: Vec::new(),
            bytes,
        })
    }

    /// This parent sampled by the general kernel (`parent_columns`: dgcom's
    /// `profile_sample_at` at every slot and level, the CPU's own order),
    /// which a parent whose valid levels change in time takes anyway; and
    /// whether the steady kernels would have sampled it.
    pub fn general(mut self) -> (Self, bool) {
        let steady = self.steady.take().is_some();
        (self, steady)
    }

    /// The parent's columns at the band's slots of `elements` for `state`,
    /// each at its element's `time`, into `band`'s values
    /// (`Nesting3D::columns` on the device).
    pub fn evaluate(
        &mut self,
        gpu: &Gpu,
        statics: &Statics3D,
        model: &dyn ParentModel3D,
        state: &State3D,
        time: StageTime<'_>,
        elements: Blocks<'_>,
        band: &mut Band3D,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let n = self.args.n as usize;
        if n == 0 {
            return Ok(());
        }
        // Every slot's moment: the distinct stage times of the listed
        // elements, in the slots' order
        let all = elements.is_all();
        if !all {
            self.listed.clear();
            self.listed.resize(statics.n_elements, false);
            elements.iter().for_each(|k| self.listed[k] = true);
        }
        self.times.clear();
        let mut listed_all = true;
        for (code, &k) in self.host_codes.iter_mut().zip(&self.elements) {
            if !all && !self.listed[k] {
                *code = NOT_LISTED;
                listed_all = false;
                continue;
            }
            let t = time.at(k);
            *code = match self.times.iter().position(|&s| s == t) {
                Some(m) => m as u32,
                None => {
                    self.times.push(t);
                    (self.times.len() - 1) as u32
                }
            };
        }
        if self.times.is_empty() {
            return Ok(());
        }
        let one = listed_all && self.times.len() == 1;
        let codes = self.codes.of(gpu, (!one).then_some(&self.host_codes[..]));
        let st = &gpu.stream;
        let m = &gpu.module_parent;
        let [out_u, out_v, out_t, out_s] = &mut band.values;
        for (chunk, times) in self.times.chunks(MAX_MOMENTS).enumerate() {
            let mut moments = Moments {
                base: (chunk * MAX_MOMENTS) as u32,
                count: times.len() as u32,
                ..Moments::default()
            };
            for (i, &t) in times.iter().enumerate() {
                let stencil = model.moment(t);
                moments.time[i] = t;
                moments.len[i] = stencil.len as u32;
                moments.idx[i] = stencil.idx.map(|k| k as u32);
                moments.w[i] = stencil.w;
            }
            // SAFETY: the slots' buffers are `n` long, the band's `n·nl`;
            // the nodes are the mesh's, the stencils the tables' points.
            unsafe {
                m.parent_check(
                    st,
                    launch(n, COLUMN_BLOCK),
                    self.args,
                    moments,
                    codes,
                    &self.nodes,
                    &state.eta,
                    &self.has,
                    &self.idx,
                    &self.w,
                    &self.values,
                    &self.positions,
                    &mut self.cache_time,
                    &mut self.cache_eta,
                    &mut self.flags,
                )?;
                let Some(steady) = &mut self.steady else {
                    m.parent_columns(
                        st,
                        launch(n * statics.n_levels, LEVEL_BLOCK),
                        self.args,
                        moments,
                        codes,
                        &self.flags,
                        &self.nodes,
                        &state.eta,
                        &statics.bed,
                        &statics.sigma_rho,
                        &self.idx,
                        &self.w,
                        &self.east_bed,
                        &self.values,
                        &self.positions,
                        &self.deep_temp,
                        &self.deep_salt,
                        &state.u,
                        &state.v,
                        &state.temp,
                        &state.salt,
                        out_u,
                        out_v,
                        out_t,
                        out_s,
                    )?;
                    continue;
                };
                // The moments' weighed profiles, unless the last launch
                // made these (a stage's second call, a cached pass)
                if steady.made_for != times {
                    m.parent_moments(
                        st,
                        launch(times.len() * steady.form.per_moment as usize, LEVEL_BLOCK),
                        self.args,
                        steady.form,
                        moments,
                        &self.values,
                        &mut steady.weighted,
                    )?;
                    steady.made_for.clear();
                    steady.made_for.extend_from_slice(times);
                }
                m.parent_columns_steady(
                    st,
                    launch(n, COLUMN_BLOCK),
                    self.args,
                    steady.form,
                    moments,
                    codes,
                    &self.flags,
                    &self.nodes,
                    &state.eta,
                    &statics.bed,
                    &statics.sigma_rho,
                    &self.idx,
                    &self.w,
                    &self.east_bed,
                    &steady.weighted,
                    &self.positions,
                    &steady.inv_dz,
                    &steady.first_valid,
                    &self.deep_temp,
                    &self.deep_salt,
                    &state.u,
                    &state.v,
                    &state.temp,
                    &state.salt,
                    out_u,
                    out_v,
                    out_t,
                    out_s,
                )?;
            }
        }
        Ok(())
    }
}

/// `series` at the points `used` (every snapshot) appended to the tables
/// `values` and `positions`, and its form there.
fn upload_form(
    series: &ProfileSeries,
    used: &[usize],
    values: &mut Vec<f32>,
    positions: &mut Vec<f64>,
) -> Result<FieldForm, Box<dyn std::error::Error>> {
    let nl = series.n_levels();
    let form = FieldForm {
        present: 1,
        fractions: u32::from(series.fractions()),
        n_points: u32::try_from(used.len())?,
        n_levels: u32::try_from(nl)?,
        values: values.len() as u64,
        positions: positions.len() as u64,
    };
    for t in 0..series.n_times() {
        for &k in used {
            values.extend_from_slice(series.column(t, k));
        }
    }
    match series.levels() {
        ProfileLevels::Depth(d) => positions.extend_from_slice(d),
        ProfileLevels::Fraction(f) => {
            for &k in used {
                positions.extend_from_slice(&f[k * nl..(k + 1) * nl]);
            }
        }
    }
    Ok(form)
}

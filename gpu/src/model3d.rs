//! The 3D model on the GPU (P2.7 Phase 4.5b, `notes/gpu.md`): dgcom's
//! `SplitModel` over the device modules of Phases 4.0–4.4, so that dgcom's
//! own `ModeSplitIntegrator` steps a 3D state that stays on the device.
//!
//! [`Model3D`] is a `Hydrostatic3D` as the device runs it (its 2D module on
//! [`GpuSWE2D`]); [`Work3D`] holds the device modules and the step's
//! buffers, which the splitter addresses by slot. Each method of the trait
//! is the CPU's section of the step (`ModeSplitPhysics` on `CpuWork`) in the
//! same order, on the modules each phase gated against it:
//!
//! - the drags' rates and `G` ([`Vertical3D::drag_rates`],
//!   [`SlowForcing3D`]), the momentum terms ([`Baroclinic3D`],
//!   [`Viscosity3D`]);
//! - the pass and what follows it ([`BarotropicPass3D`], [`LocalPass3D`]),
//!   the merged cells ([`Merged3D`]);
//! - a stage's transport and advection ([`Transport3D`], the reference's
//!   advection, [`Implicit3D`]), its values ([`StageHooks3D`] with the
//!   tracer limiter);
//! - the vertical terms ([`Vertical3D::step`]).
//!
//! What the splitter does around them that no module did is in the kernels
//! here: a stage's combination (`shu_osher`), the column sums of a stage's
//! momentum tendency, the filtered barotropic state at the step's end, and
//! the depth means of the end state.
//!
//! The 3D stages take global steps (the case runner's default) or local
//! time steps (4.5d): the splitter's passes then hand each section a list
//! of elements at their own stage times. What a pass writes into the stage
//! buffers that outlive it (a combination, the implicit advection, a stage's
//! values with the limiter and the mask, the substeps' sums and results,
//! the vertical step of a substep's end) acts on the listed elements alone,
//! the list kept on the device ([`Lists`]) and every element's stage weights
//! dgcom's `element_stage` of its level and rate ([`StagePlanDevice`]). A
//! stage's RHS is computed for every element, into buffers whose other rows
//! nothing reads; what depends on an element's stage time is evaluated at
//! the listed elements' own (the rivers' tracers, the nesting band's
//! columns). The forcing a trait object evaluates on the host is evaluated
//! there at the CPU's times and uploaded: a surface stress field, the
//! rivers' tracers, the surface heat and fresh-water fluxes (over the sea
//! surface gathered from the device, 4.5h). A nesting band's parent model
//! is sampled on the device ([`Parent3D`], 4.5k), a fixed parent evaluated
//! once (4.5g), any other parent on the host (`Nesting3D::columns` on the
//! band's nodes, gathered from the device, its cache the CPU's). The
//! viscosity once per step keeps the step start's tendency on the device and
//! adds it at every stage (4.5j; global steps only, as on the CPU). Not yet
//! on the device:
//! the conservative momentum form, the constant-depth pressure gradient;
//! [`Model3D::new`] refuses them by name.
//!
//! As `Simulation3D`'s model ([`RunModel3D`], 4.5c) it runs the step's
//! bound (4.5e), `post_process` and the blow-up check on the device. Its
//! copy of the state on the host is copied as a run's callback asks for it
//! ([`HostView3D`], 4.5f): the depth means for the stations, the whole state
//! at outputs.

// As `three_d.rs`: the device loops index arrays by the loop variable.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use std::sync::{Arc, Mutex, MutexGuard};

use cuda_core::DeviceBuffer;
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;
use dgcom::boundary::{DeviceParent3D, Nesting3D, SWEBoundaryCondition2D, Supplied};
use dgcom::case::StateView3D;
use dgcom::mesh::Mesh2D;
use dgcom::physics::{
    GlsMixing, Hydrostatic3D, LinearEOS, PhysicsModule, SWEPhysics2D, SeaSurface,
    SurfaceFluxFields, TimeStepLimits, VerticalMixing,
};
use dgcom::profiling;
use dgcom::simulation::{NonFinite, RunModel3D, non_finite_of_masks};
use dgcom::solver::rhs::{MomentumAdvectionForm, PressureGradientForm};
use dgcom::solver::state::Solution3D;
use dgcom::solver::{Blocks, SWESolution2D};
use dgcom::time::{
    Columns, DragKinds, ElementStage, Integrable, ModeSplitPhysics, MultirateStats, RhsStencil,
    ShuOsherStage, SplitModel, SplitPass, SplitWork, SspScheme, StageElements, StagePlan, StageRhs,
    StageSlots, StageTime, element_stage, filtered_state, shu_osher,
};
use dgcom::vertical::{SigmaGrid, depth_average};

use crate::baroclinic::{Baroclinic3D, BaroclinicSetup};
use crate::bound3d::Bound3D;
use crate::forcing::{ForcingSetup, SlowForcing3D};
use crate::hooks::{
    Band3D, Merged3D, StageHooks3D, StageValues3D, Stash3D, State3D, prefix_to_host,
};
use crate::limiter::{Limiter3D, VertexPatches};
use crate::listed::{ListId, Lists, On, listed_index};
use crate::parent::Parent3D;
use crate::pass::{BarotropicPass3D, DevicePass, LocalPass3D};
use crate::swe2d::{DeviceState, Gpu, GpuSWE2D, NN, level_of};
use crate::three_d::{
    Advected, COLUMN_BLOCK, ExteriorDevice, Interface, RiversDevice, Statics3D, Transport3D, launch,
};
use crate::vertical::{Implicit3D, SurfaceDevice, Vertical3D, VerticalSetup, caged_columns};
use crate::viscosity::Viscosity3D;

/// One stage pass of the 3D stages' schedule as a kernel takes it: dgcom's
/// `StagePlan` without its tables of levels and rates, which the kernel
/// takes as buffers ([`Levels3D`]).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct StagePlanDevice {
    /// The coarse step's start and length (s)
    pub t: f64,
    pub dt: f64,
    /// The stage's Shu–Osher weights (`ShuOsherStage`)
    pub start: f64,
    pub input: f64,
    pub beta: f64,
    pub c: f64,
    pub finest: u32,
    pub substep: u32,
    pub depth: u32,
    /// Whether it is the scheme's last stage
    pub last: u32,
}

impl StagePlanDevice {
    fn of(plan: &StagePlan<'_>) -> Self {
        let (t, dt) = plan.step;
        let w = plan.weights;
        Self {
            t,
            dt,
            start: w.start,
            input: w.input,
            beta: w.beta,
            c: w.c,
            finest: plan.finest as u32,
            substep: plan.substep as u32,
            depth: plan.depth as u32,
            last: u32::from(plan.last),
        }
    }

    /// Element `k`'s stage (`StagePlan::at`) with its `level` and `rate`
    /// (empty: 0).
    #[inline]
    pub fn at(&self, level: &[u32], rate: &[u32], k: usize) -> ElementStage {
        element_stage(
            (self.t, self.dt),
            self.finest as usize,
            self.substep as usize,
            self.depth as usize,
            ShuOsherStage {
                start: self.start,
                input: self.input,
                beta: self.beta,
                c: self.c,
            },
            self.last != 0,
            level_of(level, k) as u8,
            level_of(rate, k) as u8,
        )
    }
}

// =============================================================================
// Kernels
// =============================================================================

#[cuda_module]
pub mod kernels_model {
    use super::*;

    /// `combine_stage` of one field on the elements `listed` (empty: every
    /// element), `per` values each, a thread per value: element k's stage
    /// of the pass (`plan`: its `[t, dt, start, input, beta, c]`, `[finest,
    /// substep, depth, last]`) as `element_stage` gives it from its `level`
    /// and `rate` (empty: 0), `out = a·base + b·input + c·rate` as
    /// `shu_osher` sums it.
    #[kernel]
    pub fn combine(
        per: u32,
        n: u32,
        listed: &[u32],
        plan: StagePlanDevice,
        level: &[u32],
        rate: &[u32],
        base: &[f64],
        input: &[f64],
        rhs: &[f64],
        mut out: DisjointSlice<f64>,
    ) {
        let t = thread::index_1d().get();
        if t >= n as usize {
            return;
        }
        let i = listed_index(t, listed, per as usize);
        let stage = plan.at(level, rate, i / per as usize);
        // SAFETY: thread t alone writes value i (the listed elements are
        // distinct).
        unsafe {
            *out.get_unchecked_mut(i) =
                shu_osher(base[i], input[i], rhs[i], stage.a, stage.b, stage.c);
        }
    }

    /// `accumulate_stage` of one field on the elements `listed`, `per`
    /// values each, a thread per value: the last stage `last` into its
    /// substep's sum, `acc ← w·last + keep·acc` (the `acc` term only if
    /// `keep ≠ 0`) for element k's `element_stage` of the pass's last
    /// stage.
    #[kernel]
    pub fn accumulate(
        per: u32,
        n: u32,
        listed: &[u32],
        plan: StagePlanDevice,
        level: &[u32],
        rate: &[u32],
        last: &[f64],
        mut acc: DisjointSlice<f64>,
    ) {
        let t = thread::index_1d().get();
        if t >= n as usize {
            return;
        }
        let i = listed_index(t, listed, per as usize);
        let Some((w, keep)) = plan.at(level, rate, i / per as usize).accumulate else {
            return;
        };
        // SAFETY: as `combine`.
        unsafe {
            let s = acc.get_unchecked_mut(i);
            *s = if keep != 0.0 {
                w * last[i] + keep * *s
            } else {
                w * last[i]
            };
        }
    }

    /// The first half of `set_column_sums`, a thread per column: the sums of
    /// every column of `u` and `v` (`f64`'s `Sum`, as the CPU sums them).
    #[kernel]
    pub fn column_sums(
        nl: u32,
        n: u32,
        u: &[f64],
        v: &[f64],
        mut sum_u: DisjointSlice<f64>,
        mut sum_v: DisjointSlice<f64>,
    ) {
        let idx = thread::index_1d().get();
        if idx >= n as usize {
            return;
        }
        let nl = nl as usize;
        let column = idx * nl..(idx + 1) * nl;
        // SAFETY: thread idx alone writes column idx (idx < n).
        unsafe {
            *sum_u.get_unchecked_mut(idx) = u[column.clone()].iter().sum::<f64>();
            *sum_v.get_unchecked_mut(idx) = v[column].iter().sum::<f64>();
        }
    }

    /// The second half of `set_column_sums`, a thread per column: every
    /// column's sum replaced by the transport's rate (`rate_hu`, `rate_hv`),
    /// the difference shared out by `Δσ_l`.
    #[kernel]
    pub fn set_sums(
        nl: u32,
        n: u32,
        d_sigma: &[f64],
        rate_hu: &[f64],
        rate_hv: &[f64],
        sum_u: &[f64],
        sum_v: &[f64],
        mut u: DisjointSlice<f64>,
        mut v: DisjointSlice<f64>,
    ) {
        let idx = thread::index_1d().get();
        if idx >= n as usize {
            return;
        }
        let nl = nl as usize;
        let (du, dv) = (rate_hu[idx] - sum_u[idx], rate_hv[idx] - sum_v[idx]);
        for l in 0..nl {
            // SAFETY: thread idx alone writes column idx (idx < n).
            unsafe {
                *u.get_unchecked_mut(idx * nl + l) += d_sigma[l] * du;
                *v.get_unchecked_mut(idx * nl + l) += d_sigma[l] * dv;
            }
        }
    }

    /// The step's barotropic state from the filtered transport, a thread
    /// per node (`filtered_state` over the bed).
    #[kernel]
    pub fn end_barotropic(
        n: u32,
        bed: &[f64],
        h: &[f64],
        hu: &[f64],
        hv: &[f64],
        mut eta: DisjointSlice<f64>,
        mut ubar: DisjointSlice<f64>,
        mut vbar: DisjointSlice<f64>,
    ) {
        let idx = thread::index_1d().get();
        if idx >= n as usize {
            return;
        }
        let (e, u, v) = filtered_state([h[idx], hu[idx], hv[idx]], bed[idx]);
        // SAFETY: thread idx alone writes node idx (idx < n).
        unsafe {
            *eta.get_unchecked_mut(idx) = e;
            *ubar.get_unchecked_mut(idx) = u;
            *vbar.get_unchecked_mut(idx) = v;
        }
    }

    /// The depth means of the columns of `u` and `v` (`depth_average`) of
    /// the elements `listed` (empty: every element), a thread per column.
    #[kernel]
    pub fn depth_means(
        nl: u32,
        n: u32,
        listed: &[u32],
        d_sigma: &[f64],
        u: &[f64],
        v: &[f64],
        mut mean_u: DisjointSlice<f64>,
        mut mean_v: DisjointSlice<f64>,
    ) {
        let idx = thread::index_1d().get();
        if idx >= n as usize {
            return;
        }
        let idx = listed_index(idx, listed, NN);
        let nl = nl as usize;
        let column = idx * nl..(idx + 1) * nl;
        // SAFETY: thread idx alone writes column idx (idx < n).
        unsafe {
            *mean_u.get_unchecked_mut(idx) = depth_average(&u[column.clone()], d_sigma);
            *mean_v.get_unchecked_mut(idx) = depth_average(&v[column], d_sigma);
        }
    }

    /// The nesting band's columns (`η`, and u, v, T, S at its levels) at its
    /// slots' nodes `nodes`, a thread per slot.
    #[kernel]
    pub fn gather_band(
        nl: u32,
        n: u32,
        nodes: &[u32],
        eta: &[f64],
        u: &[f64],
        v: &[f64],
        temp: &[f64],
        salt: &[f64],
        mut out_eta: DisjointSlice<f64>,
        mut out_u: DisjointSlice<f64>,
        mut out_v: DisjointSlice<f64>,
        mut out_t: DisjointSlice<f64>,
        mut out_s: DisjointSlice<f64>,
    ) {
        let slot = thread::index_1d().get();
        if slot >= n as usize {
            return;
        }
        let (nl, node) = (nl as usize, nodes[slot] as usize);
        // SAFETY: thread slot alone writes slot `slot` (slot < n).
        unsafe {
            *out_eta.get_unchecked_mut(slot) = eta[node];
            for l in 0..nl {
                let (to, from) = (slot * nl + l, node * nl + l);
                *out_u.get_unchecked_mut(to) = u[from];
                *out_v.get_unchecked_mut(to) = v[from];
                *out_t.get_unchecked_mut(to) = temp[from];
                *out_s.get_unchecked_mut(to) = salt[from];
            }
        }
    }

    /// The sea surface (`surface_fluxes_at`): every column's top layer of
    /// T and S, a thread per column.
    #[kernel]
    pub fn sea_surface(
        nl: u32,
        n: u32,
        temp: &[f64],
        salt: &[f64],
        mut sst: DisjointSlice<f64>,
        mut sss: DisjointSlice<f64>,
    ) {
        let col = thread::index_1d().get();
        if col >= n as usize {
            return;
        }
        let top = (col + 1) * nl as usize - 1;
        // SAFETY: thread col alone writes column col (col < n).
        unsafe {
            *sst.get_unchecked_mut(col) = temp[top];
            *sss.get_unchecked_mut(col) = salt[top];
        }
    }

    /// `a·b + c` per value, as this build compiles it: the gates' probe of
    /// multiply–add contraction (`gate::probe`).
    #[kernel]
    pub fn multiply_add(n: u32, a: &[f64], b: &[f64], c: &[f64], mut out: DisjointSlice<f64>) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        // SAFETY: thread i alone writes value i (i < n).
        unsafe {
            *out.get_unchecked_mut(i) = a[i] * b[i] + c[i];
        }
    }

    /// `out += x`, a thread per value (`update_with`'s `*r += x`).
    #[kernel]
    pub fn add_into(n: u32, x: &[f64], mut out: DisjointSlice<f64>) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        // SAFETY: thread i alone writes value i (i < n).
        unsafe {
            *out.get_unchecked_mut(i) += x[i];
        }
    }

    /// `w` at the layer centres from Ω at the σ-surfaces
    /// (`update_vertical_velocity`: the mean of the two), a thread per
    /// column.
    #[kernel]
    pub fn w_of_omega(nl: u32, n: u32, omega: &[f64], mut w: DisjointSlice<f64>) {
        let idx = thread::index_1d().get();
        if idx >= n as usize {
            return;
        }
        let nl = nl as usize;
        for l in 0..nl {
            let (lo, hi) = (omega[idx * (nl + 1) + l], omega[idx * (nl + 1) + l + 1]);
            // SAFETY: thread idx alone writes column idx (idx < n).
            unsafe {
                *w.get_unchecked_mut(idx * nl + l) = 0.5 * (lo + hi);
            }
        }
    }

    /// `shift_columns` of `u` and `v` from their depth means to `ū`, `v̄`, of
    /// the elements `listed` (empty: every element), a thread per column.
    #[kernel]
    pub fn shift_means(
        nl: u32,
        n: u32,
        listed: &[u32],
        ubar: &[f64],
        vbar: &[f64],
        mean_u: &[f64],
        mean_v: &[f64],
        mut u: DisjointSlice<f64>,
        mut v: DisjointSlice<f64>,
    ) {
        let idx = thread::index_1d().get();
        if idx >= n as usize {
            return;
        }
        let idx = listed_index(idx, listed, NN);
        let nl = nl as usize;
        let (su, sv) = (ubar[idx] - mean_u[idx], vbar[idx] - mean_v[idx]);
        for l in 0..nl {
            // SAFETY: thread idx alone writes column idx (idx < n).
            unsafe {
                *u.get_unchecked_mut(idx * nl + l) += su;
                *v.get_unchecked_mut(idx * nl + l) += sv;
            }
        }
    }

    /// The blow-up check's masks (`non_finite_field`), a thread per layer
    /// value (`n` of them, `per_element` an element's): in `masks[k]` the
    /// bit of every field (`NON_FINITE_FIELDS`: η, u, v, T, S as bits 0–4)
    /// with a value that is not finite in element `k`, and in `count[0]` how
    /// many values were (both zeroed before). Only a blow-up's values touch
    /// the atomics.
    #[kernel]
    pub fn non_finite_masks(
        per_element: u32,
        nn: u32,
        n: u32,
        eta: &[f64],
        u: &[f64],
        v: &[f64],
        temp: &[f64],
        salt: &[f64],
        masks: *mut u32,
        count: *mut u32,
    ) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        let bit = |x: f64, bit: u32| if x.is_finite() { 0 } else { bit };
        let layers = bit(u[i], 2) | bit(v[i], 4) | bit(temp[i], 8) | bit(salt[i], 16);
        // η's values are the first n / levels threads'
        let n_eta = (n / per_element * nn) as usize;
        let surface = if i < n_eta { bit(eta[i], 1) } else { 0 };
        // SAFETY: `masks` holds a u32 per element, `count` one; only atomics
        // touch them here.
        unsafe {
            if layers != 0 {
                DeviceAtomicU32::from_ptr(masks.add(i / per_element as usize))
                    .fetch_or(layers, AtomicOrdering::Relaxed);
            }
            if surface != 0 {
                DeviceAtomicU32::from_ptr(masks.add(i / nn as usize))
                    .fetch_or(surface, AtomicOrdering::Relaxed);
            }
            if layers | surface != 0 {
                DeviceAtomicU32::from_ptr(count).fetch_add(1, AtomicOrdering::Relaxed);
            }
        }
    }
}

// =============================================================================
// Host side
// =============================================================================

/// `out = a·b + c` value by value on the device, as this build compiles it
/// (`gate::probe`).
pub(crate) fn multiply_add(
    gpu: &Gpu,
    a: &DeviceBuffer<f64>,
    b: &DeviceBuffer<f64>,
    c: &DeviceBuffer<f64>,
    out: &mut DeviceBuffer<f64>,
) -> Result<(), Box<dyn std::error::Error>> {
    let n = out.len();
    // SAFETY: the four buffers hold n values; the output is none of the
    // inputs.
    unsafe {
        gpu.module_model.multiply_add(
            &gpu.stream,
            launch(n, COLUMN_BLOCK),
            n as u32,
            a,
            b,
            c,
            out,
        )?;
    }
    Ok(())
}

/// The splitter's failure on the device: a CUDA call failed.
fn cuda<T>(result: Result<T, Box<dyn std::error::Error>>) -> T {
    result.unwrap_or_else(|e| panic!("CUDA: {e}"))
}

/// `list` among `n` elements, all of them as [`Blocks::All`] (the
/// splitter's `blocks_of`).
fn blocks_of(n: usize, list: &[u32]) -> Blocks<'_> {
    if list.len() == n {
        Blocks::All(n)
    } else {
        Blocks::list(n, list)
    }
}

/// A 3D state's fields on the device, zero: `n` values per layer field and
/// `n_columns` per column, the turbulence `n_w` (0 without one) and the
/// density if `rho`.
fn zeroed_state(gpu: &Gpu, n_columns: usize, n: usize, n_w: usize, rho: bool) -> State3D {
    let zeros = |n: usize| cuda(DeviceBuffer::zeroed(&gpu.stream, n).map_err(Into::into));
    State3D {
        eta: zeros(n_columns),
        ubar: zeros(n_columns),
        vbar: zeros(n_columns),
        u: zeros(n),
        v: zeros(n),
        temp: zeros(n),
        salt: zeros(n),
        tke: zeros(n_w),
        gls: zeros(n_w),
        rho: zeros(if rho { n } else { 0 }),
    }
}

/// `state`'s turbulence sized `n_w` (zero where it grows, as the CPU's
/// `resize` does).
fn fit_turbulence(gpu: &Gpu, state: &mut State3D, n_w: usize) {
    if state.tke.len() != n_w {
        let zeros = || cuda(DeviceBuffer::zeroed(&gpu.stream, n_w).map_err(Into::into));
        state.tke = zeros();
        state.gls = zeros();
    }
}

/// The 3D stages' buffers on the device, by slot (the CPU's
/// `StageBuffers`).
struct Stages3D {
    slots: StageSlots,
    inventories: Vec<State3D>,
    values: Vec<StageValues3D>,
    /// The substeps' results (inventories; several levels only)
    acc: Option<State3D>,
    /// One stage's RHS
    rate: State3D,
}

/// The fields a 3D stage combines (`stage_fields`): the inventories, then
/// `η`, `ū`, `v̄`; with how many values a node has of each.
fn stage_fields(s: &State3D, nl: usize) -> [(&DeviceBuffer<f64>, usize); 9] {
    [
        (&s.u, nl),
        (&s.v, nl),
        (&s.temp, nl),
        (&s.salt, nl),
        (&s.tke, nl + 1),
        (&s.gls, nl + 1),
        (&s.eta, 1),
        (&s.ubar, 1),
        (&s.vbar, 1),
    ]
}

/// [`stage_fields`], mutably.
fn stage_fields_mut(s: &mut State3D, nl: usize) -> [(&mut DeviceBuffer<f64>, usize); 9] {
    [
        (&mut s.u, nl),
        (&mut s.v, nl),
        (&mut s.temp, nl),
        (&mut s.salt, nl),
        (&mut s.tke, nl + 1),
        (&mut s.gls, nl + 1),
        (&mut s.eta, 1),
        (&mut s.ubar, 1),
        (&mut s.vbar, 1),
    ]
}

/// The levels and stage rates of the 3D stages' local time steps on the
/// device, as the passes' plans give them (`StagePlan::level`, `rate`),
/// uploaded as they change (at most once a coarse step).
struct Levels3D {
    level_host: Vec<u8>,
    level: DeviceBuffer<u32>,
    rates_host: Vec<Vec<u8>>,
    rates: Vec<DeviceBuffer<u32>>,
    none: DeviceBuffer<u32>,
}

impl Levels3D {
    fn new(gpu: &Gpu) -> Self {
        let none = || cuda(DeviceBuffer::zeroed(&gpu.stream, 0).map_err(Into::into));
        Self {
            level_host: Vec::new(),
            level: none(),
            rates_host: Vec::new(),
            rates: Vec::new(),
            none: none(),
        }
    }

    /// The levels and the stage's rates of `plan` (empty with one level:
    /// every element's are 0), and an empty list.
    fn of(&mut self, gpu: &Gpu, plan: &StagePlan<'_>) -> [&DeviceBuffer<u32>; 3] {
        if plan.finest == 0 {
            return [&self.none, &self.none, &self.none];
        }
        let st = &gpu.stream;
        let widen = |keys: &[u8]| keys.iter().map(|&k| u32::from(k)).collect::<Vec<u32>>();
        let up = |v: &[u8]| cuda(DeviceBuffer::from_host(st, &widen(v)).map_err(Into::into));
        if self.level_host != plan.level {
            self.level_host = plan.level.to_vec();
            self.level = up(plan.level);
        }
        let s = plan.stage;
        if self.rates_host.len() <= s {
            self.rates_host.resize(s + 1, Vec::new());
            self.rates.resize_with(s + 1, || {
                cuda(DeviceBuffer::zeroed(st, 0).map_err(Into::into))
            });
        }
        if self.rates_host[s] != plan.rate {
            self.rates_host[s] = plan.rate.to_vec();
            self.rates[s] = up(plan.rate);
        }
        [&self.level, &self.rates[s], &self.none]
    }
}

/// A nesting band on the device: the parent's columns at its slots, for the
/// band's relaxation ([`Band3D`]) and the open faces' exterior values. A
/// parent model (`DeviceParent3D::Model`) is sampled on the device
/// ([`Parent3D`], 4.5k); a fixed parent (`DeviceParent3D::Fixed`) is
/// evaluated once; any other parent on the host, by the physics' own
/// `Nesting3D` (`columns`, whose cache by time and `η` is then the CPU's)
/// over the slots' columns gathered from a state, and uploaded.
struct Nest3D {
    band: Band3D,
    supplied: Supplied,
    /// The parent model sampled on the device, if it is one
    model: Option<Parent3D>,
    /// Whether the parent is fixed, and its columns are set
    fixed: bool,
    set: bool,
    /// Every slot's node
    nodes: DeviceBuffer<u32>,
    /// The gathered `η` (a value a slot) and u, v, T, S (`[slot][level]`)
    gathered: [DeviceBuffer<f64>; 5],
    /// The same on the host, and a state holding them at their nodes (the
    /// rest unread: `columns` reads the slots' nodes only)
    host: [Vec<f64>; 5],
    state: Solution3D,
    /// The exterior values of u, v, T, S, from the first evaluation on
    exterior: Option<[ExteriorDevice; 4]>,
}

impl Nest3D {
    fn new(
        gpu: &Gpu,
        s: &Statics3D,
        nesting: &Nesting3D,
        n_nodes: usize,
        min_column_depth: f64,
    ) -> Self {
        let slots = nesting.slots();
        let st = &gpu.stream;
        let nodes: Vec<u32> = slots.nodes.iter().map(|&n| n as u32).collect();
        let (n, nl) = (nodes.len(), s.n_levels);
        let zeros = |n: usize| cuda(DeviceBuffer::<f64>::zeroed(st, n).map_err(Into::into));
        let model = match nesting.device_form() {
            Some(DeviceParent3D::Model(model)) => Some(cuda(Parent3D::new(
                gpu,
                s,
                nesting,
                model,
                min_column_depth,
            ))),
            _ => None,
        };
        // A model's columns are the device's: the exterior values from the
        // start (a field not supplied is never read)
        let exterior = model.as_ref().map(|_| {
            let zeros = vec![0.0; n * nl];
            let field = nesting.exterior_field(&zeros);
            std::array::from_fn(|_| cuda(ExteriorDevice::new(gpu, s, &field)))
        });
        Self {
            band: cuda(Band3D::new(gpu, s, slots)),
            supplied: slots.supplied,
            model,
            fixed: nesting.device_form().is_some_and(|f| f.is_fixed()),
            set: false,
            nodes: cuda(DeviceBuffer::from_host(st, &nodes).map_err(Into::into)),
            gathered: [
                zeros(n),
                zeros(n * nl),
                zeros(n * nl),
                zeros(n * nl),
                zeros(n * nl),
            ],
            host: [
                vec![0.0; n],
                vec![0.0; n * nl],
                vec![0.0; n * nl],
                vec![0.0; n * nl],
                vec![0.0; n * nl],
            ],
            state: Solution3D::new(s.n_elements, n_nodes, nl),
            exterior,
        }
    }

    /// The parent's columns at the band's slots of `elements` for `values`,
    /// each at its element's `time` (`Nesting3D::columns`), on the device: a
    /// fixed parent's once.
    fn evaluate<BC>(
        &mut self,
        gpu: &Gpu,
        s: &Statics3D,
        physics: &Hydrostatic3D<LinearEOS, GlsMixing, BC>,
        values: &State3D,
        time: StageTime<'_>,
        elements: Blocks<'_>,
    ) where
        BC: SWEBoundaryCondition2D + Clone + Send + Sync,
        SWEPhysics2D<BC>: PhysicsModule<SWESolution2D>,
    {
        if self.fixed && self.set {
            return;
        }
        self.set = true;
        let nesting = physics.nesting.as_ref().expect("a nested model");
        if let Some(parent) = self.model.as_mut() {
            let Some(DeviceParent3D::Model(model)) = nesting.device_form() else {
                unreachable!("the parent model's device form");
            };
            cuda(parent.evaluate(gpu, s, model, values, time, elements, &mut self.band));
            let exterior = self.exterior.as_mut().expect("a model's exterior values");
            for (to, from) in exterior.iter_mut().zip(&self.band.values) {
                cuda(
                    to.values
                        .copy_from_device_async(from, &gpu.stream)
                        .map_err(Into::into),
                );
            }
            return;
        }
        let (st, nl) = (&gpu.stream, s.n_levels);
        let n = self.host[0].len();
        if n > 0 {
            let [eta, u, v, t, sa] = &mut self.gathered;
            // SAFETY: the gathered buffers hold the slots' values; the nodes
            // are the mesh's.
            unsafe {
                cuda(
                    gpu.module_model
                        .gather_band(
                            st,
                            launch(n, COLUMN_BLOCK),
                            nl as u32,
                            n as u32,
                            &self.nodes,
                            &values.eta,
                            &values.u,
                            &values.v,
                            &values.temp,
                            &values.salt,
                            eta,
                            u,
                            v,
                            t,
                            sa,
                        )
                        .map_err(Into::into),
                );
            }
            for (to, from) in self.host.iter_mut().zip(&self.gathered) {
                cuda(from.copy_to_host(st, to).map_err(Into::into));
            }
        }
        // The gathered columns at their nodes
        let state = &mut self.state;
        let [eta, u, v, t, sa] = &self.host;
        for (slot, &node) in nesting.slots().nodes.iter().enumerate() {
            state.eta.data[node] = eta[slot];
            let (to, from) = (node * nl..(node + 1) * nl, slot * nl..(slot + 1) * nl);
            state.u[to.clone()].copy_from_slice(&u[from.clone()]);
            state.v[to.clone()].copy_from_slice(&v[from.clone()]);
            state.temp[to.clone()].copy_from_slice(&t[from.clone()]);
            state.salt[to].copy_from_slice(&sa[from]);
        }
        let columns = nesting.columns(
            state,
            &physics.bathymetry,
            &physics.sigma,
            time,
            elements,
            physics.min_column_depth,
        );
        cuda(self.band.set_columns(gpu, columns.values()));
        let exterior = columns.exterior();
        let fields = [
            exterior.velocity.as_ref().map(|[u, _]| u),
            exterior.velocity.as_ref().map(|[_, v]| v),
            exterior.temp.as_ref(),
            exterior.salt.as_ref(),
        ];
        match &mut self.exterior {
            None => {
                // Every field's slots are the band's: build each from the
                // first field supplied (a field not supplied is never read)
                let any = fields.iter().flatten().next().copied();
                if let Some(any) = any {
                    self.exterior = Some(
                        fields.map(|field| cuda(ExteriorDevice::new(gpu, s, field.unwrap_or(any)))),
                    );
                }
            }
            Some(devices) => {
                for (device, field) in devices.iter_mut().zip(fields) {
                    if let Some(field) = field {
                        cuda(
                            device
                                .values
                                .copy_from_host(st, field.values)
                                .map_err(Into::into),
                        );
                    }
                }
            }
        }
    }

    /// The exterior values of u and v (if the parent supplies a velocity)
    /// and of T and S (if it supplies tracers).
    #[allow(clippy::type_complexity)]
    fn exterior(&self) -> ([Option<&ExteriorDevice>; 2], [Option<&ExteriorDevice>; 2]) {
        match &self.exterior {
            Some([u, v, t, s]) => (
                [u, v].map(|e| self.supplied.velocity.then_some(e)),
                [t, s].map(|e| self.supplied.tracers.then_some(e)),
            ),
            None => ([None; 2], [None; 2]),
        }
    }
}

/// The surface heat and fresh-water fluxes' fields of every column: the sea
/// surface's T and S gathered on the device, both on the host with the
/// fluxes the physics evaluates there, and the fluxes on the device.
struct SurfaceFluxWork {
    surface: [DeviceBuffer<f64>; 2],
    /// SST, SSS, non-solar, shortwave, E − P
    host: [Vec<f64>; 5],
    fluxes: [DeviceBuffer<f64>; 3],
}

impl SurfaceFluxWork {
    fn new(gpu: &Gpu, n_columns: usize) -> Self {
        let zeros =
            || cuda(DeviceBuffer::<f64>::zeroed(&gpu.stream, n_columns).map_err(Into::into));
        Self {
            surface: [zeros(), zeros()],
            host: std::array::from_fn(|_| vec![0.0; n_columns]),
            fluxes: [zeros(), zeros(), zeros()],
        }
    }
}

/// The step's buffers and the device modules ([`SplitModel::Work`] of
/// [`Model3D`]).
pub struct Work3D {
    gpu: Arc<Gpu>,
    /// The mesh under the model's σ-levels
    statics: Statics3D,
    transport: Transport3D,
    baroclinic: Baroclinic3D,
    viscosity: Viscosity3D,
    hooks: StageHooks3D,
    vertical: Vertical3D,
    implicit: Option<Implicit3D>,
    forcing: SlowForcing3D,
    pass: BarotropicPass3D,
    /// The pass on local time steps, with its levels and scheme
    local: Option<(LocalPass3D, usize, SspScheme)>,
    merged: Merged3D,
    /// The columns under a cage (`caged_columns`), with cages
    caged: Option<DeviceBuffer<u32>>,
    /// `R₃D` at the step's start (u, v)
    rhs_n: [DeviceBuffer<f64>; 2],
    /// With the viscosity once per step, its tendency at the step's start
    /// (u, v), which every later stage adds
    kept_viscosity: Option<[DeviceBuffer<f64>; 2]>,
    /// `G` of the last three steps, newest first, and their step average
    history: [DeviceState; 3],
    average: DeviceState,
    /// The barotropic transport during the pass
    q: DeviceState,
    /// The column mean `Λ̄` of the layer drag
    column_drag: DeviceBuffer<f64>,
    /// The stages' `βΔt` of every element, by content: each stage's is the
    /// same over a step's passes
    implicit_dt: Vec<(Vec<f64>, DeviceBuffer<f64>)>,
    stages: Option<Stages3D>,
    /// The columns' sums or means of u and v
    sums: [DeviceBuffer<f64>; 2],
    /// The surface heat and fresh-water fluxes, if the physics has them
    surface_fluxes: Option<SurfaceFluxWork>,
    /// The surface stress field (host and device), the rivers' T and S
    stress_host: [Vec<f64>; 2],
    stress: Option<[DeviceBuffer<f64>; 2]>,
    river_conc: Option<[DeviceBuffer<f64>; 2]>,
    roughness: Option<DeviceBuffer<f64>>,
    /// The nesting band, if the model is nested
    nest: Option<Nest3D>,
    /// `w` of the end state (`post_process`), or of the start
    w: DeviceBuffer<f64>,
    /// The step's bound
    bound: Bound3D,
    /// The blow-up check's masks (an element's fields with a value not
    /// finite) and count
    non_finite: [DeviceBuffer<u32>; 2],
    /// The elements of the 3D stages' passes on local time steps, and
    /// their levels and rates
    lists: Lists,
    levels: Levels3D,
}

impl Work3D {
    /// The vertical terms' eddy viscosity and diffusivity into `host` (the
    /// device's, as of the last step).
    pub fn eddy_coefficients_into(&self, host: &mut Solution3D) {
        let st = &self.gpu.stream;
        for (to, from) in [
            (&mut host.eddy_viscosity, &self.vertical.eddy_viscosity),
            (&mut host.eddy_diffusivity, &self.vertical.eddy_diffusivity),
        ] {
            to.resize(from.len(), 0.0);
            cuda(from.copy_to_host(st, to).map_err(Into::into));
        }
    }

    /// The device's list of `elements` ([`Lists::upload`]).
    fn list(&mut self, elements: Blocks<'_>) -> ListId {
        cuda(self.lists.upload(&self.gpu, elements))
    }
}

impl SplitWork for Work3D {
    fn merged_cells(&self) -> [usize; 2] {
        self.merged.counts()
    }

    fn barotropic_multirate_stats(&self) -> Option<MultirateStats> {
        self.local
            .as_ref()
            .map(|(local, ..)| local.stepper().stats())
    }

    fn slow_forcing(&self, n: usize) -> Vec<SWESolution2D> {
        self.history[..n]
            .iter()
            .map(DeviceState::download)
            .collect()
    }

    fn restore_slow_forcing(&mut self, g: Vec<SWESolution2D>) {
        for (slot, g) in self.history.iter_mut().zip(g) {
            *slot = DeviceState::upload(&self.gpu, &g);
        }
    }
}

/// A `Hydrostatic3D` as the device runs it: its 2D module on the device,
/// and what the host evaluates for it (the forcing its trait objects give).
pub struct Model3D<'a, BC: SWEBoundaryCondition2D> {
    gpu: Arc<Gpu>,
    physics: &'a Hydrostatic3D<LinearEOS, GlsMixing, BC>,
    module: GpuSWE2D<'a>,
    /// Whether the pass's 2D module has element-local operations (the CPU's
    /// test for local time steps of the pass)
    element_local: bool,
    /// The mesh under the σ-levels and the equation of state, for the
    /// density between steps
    statics: Statics3D,
    density: StageHooks3D,
    /// The state on the host, as much of it as a run's callbacks asked for
    /// since the last step ([`HostView3D`])
    host: Mutex<HostCopy>,
}

/// How much of the state on the host is of the device's state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Copied {
    Nothing,
    /// η, ū, v̄
    DepthMeans,
    /// Every field, `w` and the eddy coefficients
    All,
}

/// The state on the host and how much of it is the device's.
struct HostCopy {
    state: Solution3D,
    copied: Copied,
}

impl<'a, BC> Model3D<'a, BC>
where
    BC: SWEBoundaryCondition2D + Clone + Send + Sync,
    SWEPhysics2D<BC>: PhysicsModule<SWESolution2D>,
{
    /// `physics` on the device `gpu`, or what it has that the device does
    /// not run.
    pub fn new(
        gpu: &Arc<Gpu>,
        physics: &'a Hydrostatic3D<LinearEOS, GlsMixing, BC>,
    ) -> Result<Self, String> {
        let refuse = |what: &str| Err(format!("{what} is not on the device yet"));
        if physics.momentum_advection_form != MomentumAdvectionForm::Split {
            return refuse("the momentum's conservative advection form");
        }
        if physics.pressure_gradient != PressureGradientForm::SigmaPairs {
            return refuse("the constant-depth pressure gradient");
        }
        for (name, bc) in [("T", &physics.temp_bc), ("S", &physics.salt_bc)] {
            if bc.device_form().is_none() {
                return Err(format!("{name}'s boundary condition has no device form"));
            }
        }
        let element_local = physics
            .swe_physics
            .local_time_stepping()
            .is_some_and(|local| local.is_element_local());
        let form = physics.swe_physics.device_form()?;
        // The 2D module's form borrows the physics, which outlives the model
        let module = GpuSWE2D::from_form(gpu, &form)?;
        let p = physics;
        let statics = Statics3D::upload(
            gpu,
            &p.mesh,
            &p.ops,
            &p.geom,
            &p.bathymetry,
            &p.sigma,
            &p.boundaries,
            p.metric_form(),
        )
        .map_err(|e| e.to_string())?;
        let density =
            StageHooks3D::new(gpu, p.eos, p.min_column_depth).map_err(|e| e.to_string())?;
        let (ne, nn, nl) = (p.mesh.n_elements, p.ops.n_nodes, p.sigma.n_levels());
        Ok(Self {
            gpu: gpu.clone(),
            physics,
            module,
            element_local,
            statics,
            density,
            host: Mutex::new(HostCopy {
                state: Solution3D::new(ne, nn, nl),
                copied: Copied::Nothing,
            }),
        })
    }

    /// Start a run from `state` (the host's copy, its `w` and eddy
    /// coefficients included, which the device does not hold).
    pub fn start_from(&mut self, state: &Solution3D) {
        *self.host_copy() = HostCopy {
            state: state.clone(),
            copied: Copied::All,
        };
    }

    /// The device state `device` on the host as a run's callback asks for
    /// it, `work` the integrator's (its `w` and eddy coefficients; none
    /// before the first step, when the host's are the start's).
    pub fn view<'m>(&'m self, device: &'m State3D, work: Option<&'m Work3D>) -> HostView3D<'m> {
        HostView3D {
            gpu: &self.gpu,
            device,
            work,
            host: self.host_copy(),
        }
    }

    fn host_copy(&self) -> MutexGuard<'_, HostCopy> {
        self.host.lock().expect("the host's copy")
    }

    /// `Hydrostatic3D::time_step_limits` of the device state `s` (its `w`
    /// the last step's, or the start's), on the device (`Bound3D`): and how
    /// many elements had their sloping levels' frequency measured.
    pub fn time_step_limits(
        &self,
        s: &State3D,
        work: &mut Work3D,
        cfl: f64,
        scheme: SspScheme,
    ) -> (TimeStepLimits, usize) {
        let Work3D {
            statics,
            baroclinic,
            viscosity,
            bound,
            w,
            ..
        } = work;
        cuda(bound.limits(
            statics,
            self.physics,
            baroclinic,
            viscosity,
            s,
            w,
            cfl,
            scheme,
        ))
    }

    /// The 2D module's negative-depth clips.
    pub fn negative_depth_clips(&self) -> usize {
        self.module.negative_depth_clips()
    }

    /// The density of `state`'s T and S (`Hydrostatic3D::update_density`,
    /// before every step).
    pub fn update_density(&self, state: &mut State3D) {
        cuda(self.density.density(&self.statics, state, On::ALL));
    }

    /// The mesh under `sigma`'s levels on the device.
    fn statics(&self, sigma: &SigmaGrid) -> Statics3D {
        let p = self.physics;
        cuda(Statics3D::upload(
            &self.gpu,
            &p.mesh,
            &p.ops,
            &p.geom,
            &p.bathymetry,
            sigma,
            &p.boundaries,
            p.metric_form(),
        ))
    }

    /// The surface heat and fresh-water fluxes at `t` over `state`'s sea
    /// surface into `work`'s device buffers (`Hydrostatic3D`'s
    /// `surface_fluxes_at`: the physics' own fluxes on the host, from the
    /// top layer of T and S), if the physics has them.
    fn surface_fluxes_at(&self, state: &State3D, t: f64, work: &mut Work3D) {
        let (Some(fluxes), Some(w)) = (&self.physics.surface_fluxes, &mut work.surface_fluxes)
        else {
            return;
        };
        let (gpu, s) = (&self.gpu, &work.statics);
        let (st, n) = (&gpu.stream, s.n_columns());
        let [sst_d, sss_d] = &mut w.surface;
        // SAFETY: T and S hold every column's levels, the surface a value a
        // column.
        unsafe {
            cuda(
                gpu.module_model
                    .sea_surface(
                        st,
                        launch(n, COLUMN_BLOCK),
                        s.n_levels as u32,
                        n as u32,
                        &state.temp,
                        &state.salt,
                        sst_d,
                        sss_d,
                    )
                    .map_err(Into::into),
            );
        }
        profiling::scope("device: surface fluxes", || {
            let [sst, sss, nonsolar, shortwave, freshwater] = &mut w.host;
            cuda(sst_d.copy_to_host(st, sst).map_err(Into::into));
            cuda(sss_d.copy_to_host(st, sss).map_err(Into::into));
            fluxes.surface_fluxes_into(
                t,
                SeaSurface {
                    temperature: sst,
                    salinity: sss,
                },
                SurfaceFluxFields {
                    nonsolar,
                    shortwave,
                    freshwater,
                },
            );
            for (to, from) in w
                .fluxes
                .iter_mut()
                .zip([&*nonsolar, &*shortwave, &*freshwater])
            {
                cuda(to.copy_from_host(st, from).map_err(Into::into));
            }
        });
    }

    /// A prognostic closure's turbulence at its initial values where
    /// `state` has none (`start_turbulence`, and the CPU's vertical
    /// diffusion at its first call).
    fn allocate_turbulence(&self, state: &mut State3D) {
        let p = self.physics;
        let Some([k0, psi0]) = p.mixing.initial_turbulence() else {
            return;
        };
        let n_w = p.mesh.n_elements * NN * (p.sigma.n_levels() + 1);
        if state.tke.len() != n_w {
            let st = &self.gpu.stream;
            let up = |v: f64| cuda(DeviceBuffer::from_host(st, &vec![v; n_w]).map_err(Into::into));
            state.tke = up(k0);
            state.gls = up(psi0);
        }
    }

    /// The surface stress field at `t` into `work`'s device buffers, if the
    /// physics has one.
    fn stress_at(&self, t: f64, work: &mut Work3D) {
        let Some(field) = &self.physics.surface_stress else {
            return;
        };
        let n = self.physics.mesh.n_elements * NN;
        let [x, y] = &mut work.stress_host;
        x.resize(n, 0.0);
        y.resize(n, 0.0);
        field.surface_stress_into(t, x, y);
        let st = &self.gpu.stream;
        let up = |v: &[f64]| cuda(DeviceBuffer::from_host(st, v).map_err(Into::into));
        work.stress = Some([up(x), up(y)]);
    }
}

/// How a momentum RHS takes the shear's viscosity: of its own values, or
/// with the viscosity once per step (`Hydrostatic3D`'s `momentum_rhs`) the
/// step start's, kept: computed into the buffers at the step's start
/// (`R₃D`), added at a later stage.
enum StepViscosity<'b> {
    Own,
    Keep(&'b mut [DeviceBuffer<f64>; 2]),
    Kept(&'b [DeviceBuffer<f64>; 2]),
}

/// `Hydrostatic3D::momentum_rhs` of `values` on the elements of `on` into
/// `rhs_u`, `rhs_v` (whose other rows are not read): the reference at their
/// levels, the pressure gradient and Coriolis, the shear's viscosity
/// (`step_viscosity`), the nesting band's relaxation of the shear, the thin
/// columns' tendencies zero. The pre-passes that the elements' faces read
/// of their neighbours (the reference, the columns' pressures, the
/// viscosity's gradients) take every element.
fn momentum(
    gpu: &Gpu,
    s: &Statics3D,
    baroclinic: &mut Baroclinic3D,
    viscosity: &mut Viscosity3D,
    hooks: &StageHooks3D,
    band: Option<&Band3D>,
    values: &State3D,
    step_viscosity: StepViscosity<'_>,
    rhs_u: &mut DeviceBuffer<f64>,
    rhs_v: &mut DeviceBuffer<f64>,
    on: On<'_>,
) {
    cuda(baroclinic.reference(s, &values.eta, &values.rho));
    cuda(baroclinic.momentum(
        s,
        &values.eta,
        &values.rho,
        &values.u,
        &values.v,
        rhs_u,
        rhs_v,
        on,
    ));
    match step_viscosity {
        StepViscosity::Own => {
            cuda(viscosity.apply(s, &values.eta, &values.u, &values.v, rhs_u, rhs_v, on))
        }
        StepViscosity::Keep(kept) => {
            // From zero, as the CPU's kept tendency, then added (the step's
            // start: every element)
            let [ku, kv] = &mut *kept;
            cuda(ku.zero_async(&gpu.stream).map_err(Into::into));
            cuda(kv.zero_async(&gpu.stream).map_err(Into::into));
            cuda(viscosity.apply(s, &values.eta, &values.u, &values.v, ku, kv, on));
            add_kept(gpu, kept, rhs_u, rhs_v);
        }
        StepViscosity::Kept(kept) => add_kept(gpu, kept, rhs_u, rhs_v),
    }
    // The nesting band's relaxation of the shear (its columns evaluated
    // for these values: `Nest3D::evaluate`)
    if let Some(band) = band {
        cuda(hooks.relax_shear(s, band, &values.eta, &values.u, &values.v, rhs_u, rhs_v));
    }
    cuda(hooks.zero_thin_momentum(s, &values.eta, rhs_u, rhs_v));
}

/// `rhs += kept` for u and v (`update_with`).
fn add_kept(
    gpu: &Gpu,
    [ku, kv]: &[DeviceBuffer<f64>; 2],
    rhs_u: &mut DeviceBuffer<f64>,
    rhs_v: &mut DeviceBuffer<f64>,
) {
    let n = rhs_u.len();
    for (x, out) in [(ku, rhs_u), (kv, rhs_v)] {
        // SAFETY: both hold every layer value.
        unsafe {
            cuda(
                gpu.module_model
                    .add_into(&gpu.stream, launch(n, COLUMN_BLOCK), n as u32, x, out)
                    .map_err(Into::into),
            );
        }
    }
}

impl<'a, BC> SplitModel for Model3D<'a, BC>
where
    BC: SWEBoundaryCondition2D + Clone + Send + Sync,
    SWEPhysics2D<BC>: PhysicsModule<SWESolution2D>,
{
    type State = State3D;
    type Work = Work3D;

    fn n_elements(&self) -> usize {
        self.physics.mesh.n_elements
    }

    fn new_work(&self, state: &State3D) -> Work3D {
        let (gpu, p) = (&self.gpu, self.physics);
        let st = &gpu.stream;
        let (ne, nl) = (p.mesh.n_elements, p.sigma.n_levels());
        let (n_columns, n) = (ne * NN, ne * NN * nl);
        let zeros = |n: usize| cuda(DeviceBuffer::<f64>::zeroed(st, n).map_err(Into::into));
        let up = |v: &[f64]| cuda(DeviceBuffer::from_host(st, v).map_err(Into::into));
        let statics = self.statics(&p.sigma);
        let thin = p.min_column_depth;
        let rivers = p.rivers.as_deref();
        let transport = cuda(Transport3D::new(
            gpu,
            self.statics(&p.sigma),
            thin,
            cuda(
                rivers
                    .map(|r| RiversDevice::new(gpu, &statics, r))
                    .transpose(),
            ),
        ));
        let reference = p.active_reference();
        let baroclinic = cuda(Baroclinic3D::new(
            gpu,
            &statics,
            &p.mesh,
            &p.ops,
            &p.sigma,
            BaroclinicSetup {
                coriolis: &p.coriolis,
                eos: p.eos,
                profile: reference,
                balanced: p.balanced_reference.as_deref(),
                g: p.g,
                rho0: p.rho0,
                min_column_depth: thin,
            },
        ));
        let viscosity = cuda(Viscosity3D::new(
            gpu,
            &statics,
            &p.ops,
            &p.geom,
            p.horizontal_viscosity,
            thin,
        ));
        let bound = cuda(Bound3D::new(gpu, &statics, p, reference.is_some()));
        let limiter = cuda(Limiter3D::new(
            gpu,
            &statics,
            &VertexPatches::of(&p.mesh),
            &p.tracer_limiter,
            thin,
        ));
        let hooks = cuda(StageHooks3D::new(gpu, p.eos, thin)).with_limiter(limiter);
        let vertical = cuda(Vertical3D::new(
            gpu,
            &statics,
            &p.sigma,
            VerticalSetup {
                mixing: &p.mixing,
                forcing: p.forcing,
                bottom_drag: p.bottom_drag,
                cages: p.cage_drag.as_ref(),
                water: p.shortwave_absorption,
                thin_bed: p.thin_column_bed,
                g: p.g,
                rho0: p.rho0,
                min_column_depth: thin,
            },
        ));
        let implicit = p
            .implicit_vertical_advection
            .map(|split| cuda(Implicit3D::new(gpu, &statics, split, thin)));
        let f = p.forcing;
        let forcing = cuda(SlowForcing3D::new(
            gpu,
            self.statics(&p.sigma),
            self.statics(&SigmaGrid::uniform(1)),
            &p.mesh,
            &p.ops,
            p.sigma.sigma_rho()[0],
            ForcingSetup {
                coriolis: &p.coriolis,
                stress: [
                    f.surface_stress[0],
                    f.surface_stress[1],
                    f.bottom_stress[0],
                    f.bottom_stress[1],
                ],
                bottom_drag: p.bottom_drag,
                rho0: p.rho0,
                min_column_depth: thin,
                vertical: p.momentum_vertical_advection,
            },
        ));
        let pass = cuda(BarotropicPass3D::new(
            gpu,
            &self.module,
            &p.bathymetry,
            p.metric_form(),
            rivers,
        ));
        let merged = cuda(Merged3D::none(gpu, pass.statics()));
        let caged = p
            .cage_drag
            .as_ref()
            .filter(|cages| !cages.is_empty())
            .map(|cages| {
                cuda(
                    DeviceBuffer::from_host(st, &caged_columns(cages, n_columns))
                        .map_err(Into::into),
                )
            });
        let like = SWESolution2D::new(ne, NN);
        let history = std::array::from_fn(|_| DeviceState::upload(gpu, &like));
        let _ = state;
        Work3D {
            gpu: gpu.clone(),
            transport,
            baroclinic,
            viscosity,
            hooks,
            vertical,
            implicit,
            forcing,
            pass,
            local: None,
            merged,
            caged,
            rhs_n: [zeros(n), zeros(n)],
            kept_viscosity: (p.viscosity_once_per_step() && !p.horizontal_viscosity.is_zero())
                .then(|| [zeros(n), zeros(n)]),
            history,
            average: DeviceState::upload(gpu, &like),
            q: DeviceState::upload(gpu, &like),
            column_drag: zeros(n_columns),
            implicit_dt: Vec::new(),
            stages: None,
            sums: [zeros(n_columns), zeros(n_columns)],
            surface_fluxes: p
                .surface_fluxes
                .as_ref()
                .map(|_| SurfaceFluxWork::new(gpu, n_columns)),
            stress_host: [Vec::new(), Vec::new()],
            stress: None,
            river_conc: rivers.map(|r| [zeros(r.len()), zeros(r.len())]),
            roughness: p.surface_roughness.as_deref().map(up),
            nest: p
                .nesting
                .as_ref()
                .map(|nesting| Nest3D::new(gpu, &statics, nesting, NN, p.min_column_depth)),
            // The start's (its bound reads it before the first step)
            w: up(&self.host_copy().state.w),
            bound,
            non_finite: [ne, 1].map(|n| cuda(DeviceBuffer::zeroed(st, n).map_err(Into::into))),
            lists: Lists::new(ne),
            levels: Levels3D::new(gpu),
            statics,
        }
    }

    fn start_turbulence(&self, state: &mut State3D) {
        self.allocate_turbulence(state);
    }

    fn drag_rates(&self, state: &State3D, _t: f64, work: &mut Work3D) -> DragKinds {
        cuda(work.vertical.drag_rates(&work.statics, state));
        let layers = work.caged.is_some();
        if layers {
            let Work3D {
                statics,
                vertical,
                pass,
                column_drag,
                ..
            } = work;
            cuda(pass.column_drag(&statics.d_sigma, &vertical.layer_rate, column_drag));
        }
        DragKinds {
            bottom: self.physics.bottom_drag.is_some(),
            layers,
        }
    }

    fn slow_forcing(&self, state: &State3D, t: f64, work: &mut Work3D) {
        work.history.rotate_right(1);
        {
            let Work3D {
                gpu,
                statics,
                baroclinic,
                viscosity,
                hooks,
                nest,
                rhs_n: [rhs_u, rhs_v],
                kept_viscosity,
                ..
            } = work;
            // The nesting band's columns of the step's start at `t`
            if let Some(nest) = nest.as_mut() {
                let all = Blocks::All(statics.n_elements);
                nest.evaluate(
                    gpu,
                    statics,
                    self.physics,
                    state,
                    StageTime::Uniform(t),
                    all,
                );
            }
            let band = nest.as_ref().map(|nest| &nest.band);
            let step_viscosity = match kept_viscosity {
                Some(kept) => StepViscosity::Keep(kept),
                None => StepViscosity::Own,
            };
            momentum(
                gpu,
                statics,
                baroclinic,
                viscosity,
                hooks,
                band,
                state,
                step_viscosity,
                rhs_u,
                rhs_v,
                On::ALL,
            );
        }
        self.stress_at(t, work);
        let Work3D {
            forcing,
            rhs_n,
            stress,
            vertical,
            caged,
            history,
            gpu,
            ..
        } = work;
        let stress = stress.as_ref().map(|[x, y]| [x, y]);
        let cages = caged.as_ref().map(|caged| (&*vertical, caged));
        cuda(forcing.compute(state, [&rhs_n[0], &rhs_n[1]], stress, cages));
        let newest = &mut history[0];
        for (to, from) in [&mut newest.h, &mut newest.hu, &mut newest.hv]
            .into_iter()
            .zip(&forcing.g)
        {
            cuda(
                to.copy_from_device_async(from, &gpu.stream)
                    .map_err(Into::into),
            );
        }
    }

    fn average_slow_forcing(&self, work: &mut Work3D, weights: &[f64]) {
        let Work3D {
            average,
            history,
            gpu,
            ..
        } = work;
        for field in [&mut average.h, &mut average.hu, &mut average.hv] {
            cuda(field.zero_async(&gpu.stream).map_err(Into::into));
        }
        for (g, &w) in history.iter().zip(weights) {
            average.axpy(w, g);
        }
    }

    fn to_transport(&self, state: &State3D, work: &mut Work3D) {
        cuda(work.pass.to_transport(state, &mut work.q));
    }

    fn assign_pass_levels(
        &self,
        work: &mut Work3D,
        levels: usize,
        scheme: SspScheme,
        cfl: f64,
        dt_cap: f64,
    ) -> Option<f64> {
        if !self.element_local {
            return None;
        }
        let Work3D { local, q, .. } = work;
        if local
            .as_ref()
            .is_none_or(|&(_, l, s)| l != levels || s != scheme)
        {
            *local = Some((LocalPass3D::new(levels, scheme, q), levels, scheme));
        }
        let (local, ..) = local.as_mut().expect("the pass was just set");
        Some(local.assign_levels(&self.module, q, cfl, dt_cap))
    }

    fn pass_step(&self, work: &Work3D, cfl: f64, scheme: SspScheme) -> f64 {
        self.module.compute_dt_ssp(&work.q, cfl, Some(scheme))
    }

    fn barotropic_pass(&self, work: &mut Work3D, pass: SplitPass<'_>) {
        let Work3D {
            pass: device,
            local,
            q,
            average,
            vertical,
            column_drag,
            ..
        } = work;
        let inputs = DevicePass {
            filter: pass.filter,
            scheme: pass.scheme,
            t: pass.t,
            dt_bt: pass.dt_bt,
            forcing: average,
            bottom: pass.drag.bottom.then_some(&vertical.bottom_rate),
            column: pass.drag.layers.then_some(&*column_drag),
            rivers: self.physics.rivers.as_deref(),
        };
        if pass.local {
            let (local, ..) = local
                .as_mut()
                .expect("the pass's levels are assigned (assign_pass_levels)");
            cuda(device.run_local(&self.module, inputs, local, q));
        } else {
            cuda(device.run(&self.module, inputs, q));
        }
        // The early weights are negative: the average's positivity
        device.post_process_average(&self.module);
    }

    fn after_pass(&self, state: &State3D, dt: f64, work: &mut Work3D) {
        cuda(work.pass.finish(state, dt));
        if let Some(rivers) = &mut work.transport.rivers {
            cuda(rivers.set_discharge(&self.gpu, &work.pass.discharge));
        }
    }

    fn merge_cells(&self, state: &mut State3D, dt: f64, euler_dt: &[f64], work: &mut Work3D) {
        let Work3D {
            gpu,
            statics,
            pass,
            merged,
            hooks,
            ..
        } = work;
        cuda(
            pass.flux
                .euler_dt
                .copy_from_host(&gpu.stream, euler_dt)
                .map_err(Into::into),
        );
        let water = pass.step_water(state, dt, self.physics.min_column_depth);
        cuda(merged.find(gpu, pass.statics(), &water));
        cuda(hooks.start(statics, merged, state));
    }

    fn start_stages(
        &self,
        state: &mut State3D,
        work: &mut Work3D,
        slots: StageSlots,
        start: usize,
    ) {
        let Work3D {
            gpu,
            statics: s,
            hooks,
            merged,
            stages,
            ..
        } = work;
        let (n_columns, nl) = (s.n_columns(), s.n_levels);
        let n_w = state.tke.len();
        if stages.as_ref().is_none_or(|stages| stages.slots != slots) {
            // The combinations carry no density; the values do
            let inventory = || zeroed_state(gpu, n_columns, n_columns * nl, n_w, false);
            *stages = Some(Stages3D {
                slots,
                inventories: (0..slots.inventories).map(|_| inventory()).collect(),
                values: (0..slots.values)
                    .map(|_| StageValues3D {
                        values: zeroed_state(gpu, n_columns, n_columns * nl, n_w, true),
                        stash: cuda(Stash3D::new(gpu, s)),
                    })
                    .collect(),
                acc: slots.acc.then(inventory),
                rate: inventory(),
            });
        }
        let stages = stages.as_mut().expect("just sized");
        for buffer in stages
            .inventories
            .iter_mut()
            .chain(stages.acc.as_mut())
            .chain(stages.values.iter_mut().map(|v| &mut v.values))
            .chain([&mut stages.rate])
        {
            fit_turbulence(gpu, buffer, n_w);
        }
        // The values the substeps' first RHS reads: the step's start, masked
        let out = &mut stages.values[start];
        cuda(out.values.copy_from(gpu, state));
        cuda(out.stash.forget(gpu));
        cuda(hooks.mask(s, out, On::ALL));
        cuda(hooks.to_inventories(s, state));
        // What a flushed element's nodes show to the first stage: what
        // passes through it
        if merged.any() {
            let fields = state.fields();
            let outs = out.values.fields_mut();
            for (which, (q, out)) in fields.into_iter().zip(outs).enumerate().skip(2) {
                let eta = &state.eta;
                cuda(hooks.pass_through(s, merged, which, q, eta, out, false, On::ALL));
            }
        }
    }

    fn stage_momentum(
        &self,
        work: &mut Work3D,
        values: usize,
        on: StageElements<'_>,
        reuse_at: Option<f64>,
    ) {
        let (elements, time) = (on.elements, on.time);
        let n = elements.n();
        // At a step's first stage the elements at tⁿ take R₃D; the others
        // (a substep that starts later) their own
        let (reused, fresh): (Vec<u32>, Vec<u32>) = match reuse_at {
            Some(t) => elements
                .iter()
                .map(|k| k as u32)
                .partition(|&k| time.at(k as usize) == t),
            None => (Vec::new(), elements.iter().map(|k| k as u32).collect()),
        };
        if !fresh.is_empty() {
            let fresh = blocks_of(n, &fresh);
            let id = work.list(fresh);
            let Work3D {
                gpu,
                statics,
                baroclinic,
                viscosity,
                hooks,
                stages,
                nest,
                kept_viscosity,
                lists,
                ..
            } = work;
            let Stages3D {
                values: all_values,
                rate,
                ..
            } = stages.as_mut().expect("sized");
            let values = &all_values[values].values;
            // The nesting band's columns of the stage's values, each at its
            // element's stage time
            if let Some(nest) = nest.as_mut() {
                nest.evaluate(gpu, statics, self.physics, values, time, fresh);
            }
            let band = nest.as_ref().map(|nest| &nest.band);
            let (u, v) = (&mut rate.u, &mut rate.v);
            let step_viscosity = match kept_viscosity {
                Some(kept) => {
                    assert!(
                        fresh.is_all(),
                        "the viscosity once per step needs global steps (no common step start \
                         otherwise)"
                    );
                    StepViscosity::Kept(kept)
                }
                None => StepViscosity::Own,
            };
            momentum(
                gpu,
                statics,
                baroclinic,
                viscosity,
                hooks,
                band,
                values,
                step_viscosity,
                u,
                v,
                lists.on(id),
            );
        }
        if !reused.is_empty() {
            let id = work.list(blocks_of(n, &reused));
            let Work3D {
                statics,
                hooks,
                rhs_n,
                stages,
                lists,
                ..
            } = work;
            let rate = &mut stages.as_mut().expect("sized").rate;
            let nl = statics.n_levels;
            for (to, from) in [&mut rate.u, &mut rate.v].into_iter().zip(&*rhs_n) {
                cuda(hooks.copy_rows(statics, nl, from, to, lists.on(id)));
            }
        }
    }

    /// The advection's kernels on `rhs.on`'s elements, their pre-passes (the
    /// layer transports, Ω, the reference) and the tendencies' cheap tails
    /// on every element (the rate's rows of the others are not read); the
    /// rivers' T and S and the nesting band's columns at the listed
    /// elements' own stage times.
    fn stage_transport(&self, state: &State3D, work: &mut Work3D, rhs: StageRhs<'_>) {
        let p = self.physics;
        let (time, elements) = (rhs.on.time, rhs.on.elements);
        let id = work.list(elements);
        // The rivers' T and S at their elements' stage times, as the CPU
        // evaluates them (another element's river is not read)
        if let (Some(rivers), Some([conc_t, conc_s])) = (p.rivers.as_deref(), &mut work.river_conc)
        {
            let at = |i: usize| time.at(rivers.element(i).as_usize());
            let temp: Vec<f64> = (0..rivers.len())
                .map(|i| rivers.river(i).temperature.at(at(i)))
                .collect();
            let salt: Vec<f64> = (0..rivers.len())
                .map(|i| rivers.river(i).salinity.at(at(i)))
                .collect();
            let st = &self.gpu.stream;
            cuda(conc_t.copy_from_host(st, &temp).map_err(Into::into));
            cuda(conc_s.copy_from_host(st, &salt).map_err(Into::into));
        }
        let Work3D {
            statics: s,
            transport,
            baroclinic,
            hooks,
            implicit,
            pass,
            stages,
            sums,
            river_conc,
            gpu,
            nest,
            lists,
            ..
        } = work;
        let on = lists.on(id);
        let Stages3D {
            inventories,
            values,
            rate,
            ..
        } = stages.as_mut().expect("sized");
        // Velocity tendencies → inventory tendencies under the input's η
        let input_eta = match rhs.input {
            None => &state.eta,
            Some(slot) => &inventories[slot].eta,
        };
        cuda(hooks.to_inventory(s, 0, input_eta, &mut rate.u));
        cuda(hooks.to_inventory(s, 1, input_eta, &mut rate.v));
        // The layer transports of the stage's values (their thin columns at
        // rest: masked), the implicit part of Ω split off, the reference at
        // their levels
        let vals = &values[rhs.values].values;
        // The nesting parent's columns of the values: its values outside the
        // open faces it covers, its relaxation of the tracers below
        if let Some(nest) = nest.as_mut() {
            profiling::scope("device: nesting columns", || {
                nest.evaluate(gpu, s, p, vals, time, elements)
            });
        }
        let (ext_velocity, ext_tracers) = nest
            .as_ref()
            .map_or(([None; 2], [None; 2]), Nest3D::exterior);
        let parent = ext_velocity[0].zip(ext_velocity[1]).map(|(u, v)| [u, v]);
        cuda(transport.compute(&vals.eta, &vals.u, &vals.v, Some(&pass.flux), parent));
        if let Some(implicit) = implicit {
            cuda(implicit.split(s, &vals.eta, &pass.flux.euler_dt, &mut transport.omega));
        }
        cuda(baroclinic.reference(s, &vals.eta, &vals.rho));
        // The advection with the water no transport carried, field by field
        let velocity = |field, exterior| Advected {
            field,
            reference: None,
            range: dgcom::solver::rhs::UNBOUNDED,
            exterior,
            bc: dgcom::solver::rhs::DeviceTracerBC3D::Extrapolation,
        };
        cuda(transport.advect(
            [
                &velocity(&vals.u, ext_velocity[0]),
                &velocity(&vals.v, ext_velocity[1]),
            ],
            [&mut rate.u, &mut rate.v],
            p.momentum_vertical_advection,
            Interface::Central,
            false,
            true,
            on,
        ));
        let ranges = p.tracer_limiter.advection_ranges();
        let form = |bc: &dyn dgcom::solver::rhs::TracerBoundaryCondition3D| {
            bc.device_form().expect("checked in Model3D::new")
        };
        let referenced = baroclinic.fields.is_some();
        {
            let reference = |i: usize| baroclinic.fields.as_ref().map(|f| &f.tracers[i]);
            cuda(transport.advect(
                [
                    &Advected {
                        field: &vals.temp,
                        reference: reference(0),
                        range: ranges[0],
                        exterior: ext_tracers[0],
                        bc: form(&*p.temp_bc),
                    },
                    &Advected {
                        field: &vals.salt,
                        reference: reference(1),
                        range: ranges[1],
                        exterior: ext_tracers[1],
                        bc: form(&*p.salt_bc),
                    },
                ],
                [&mut rate.temp, &mut rate.salt],
                p.vertical_advection,
                Interface::DonorBesideThin,
                true,
                !referenced,
                on,
            ));
        }
        if referenced {
            cuda(baroclinic.advect_reference(
                transport,
                &vals.eta,
                &vals.temp,
                &vals.salt,
                &mut rate.temp,
                &mut rate.salt,
                true,
                on,
            ));
        }
        if !vals.tke.is_empty() {
            match p.turbulence_advection {
                Some(scheme) => cuda(transport.advect_w(
                    [&vals.tke, &vals.gls],
                    [&mut rate.tke, &mut rate.gls],
                    scheme,
                    true,
                    on,
                )),
                None => {
                    for field in [&mut rate.tke, &mut rate.gls] {
                        cuda(field.zero_async(&gpu.stream).map_err(Into::into));
                    }
                }
            }
        }
        // The river water's tracers
        if let Some([conc_t, conc_s]) = river_conc {
            cuda(transport.river_sources(&mut rate.temp, conc_t));
            cuda(transport.river_sources(&mut rate.salt, conc_s));
        }
        // The nesting band's relaxation of the tracers
        if let Some(nest) = nest.as_ref() {
            cuda(hooks.relax_tracers(
                s,
                &nest.band,
                &vals.eta,
                &vals.temp,
                &vals.salt,
                &mut rate.temp,
                &mut rate.salt,
            ));
        }
        cuda(hooks.zero_thin_momentum(s, &vals.eta, &mut rate.u, &mut rate.v));
        // The column sums of the momentum the barotropic transport's rate,
        // and the barotropic rates
        let (m, st) = (&gpu.module_model, &gpu.stream);
        let (nl, n) = (s.n_levels as u32, s.n_columns());
        let [rate_ubar, rate_vbar, rate_hu, rate_hv] = &pass.rates;
        let [sum_u, sum_v] = sums;
        // SAFETY: every buffer holds the mesh's columns (and levels); no
        // output is an input.
        unsafe {
            cuda(
                m.column_sums(
                    st,
                    launch(n, COLUMN_BLOCK),
                    nl,
                    n as u32,
                    &rate.u,
                    &rate.v,
                    sum_u,
                    sum_v,
                )
                .map_err(Into::into),
            );
            cuda(
                m.set_sums(
                    st,
                    launch(n, COLUMN_BLOCK),
                    nl,
                    n as u32,
                    &s.d_sigma,
                    rate_hu,
                    rate_hv,
                    sum_u,
                    sum_v,
                    &mut rate.u,
                    &mut rate.v,
                )
                .map_err(Into::into),
            );
        }
        for (to, from) in [
            (&mut rate.eta, &pass.flux.eta_rate),
            (&mut rate.ubar, rate_ubar),
            (&mut rate.vbar, rate_vbar),
        ] {
            cuda(to.copy_from_device_async(from, st).map_err(Into::into));
        }
    }

    fn combine_stage(
        &self,
        state: &State3D,
        work: &mut Work3D,
        input: Option<usize>,
        out: usize,
        elements: Blocks<'_>,
        plan: &StagePlan<'_>,
    ) {
        let id = work.list(elements);
        let Work3D {
            gpu,
            statics,
            stages,
            lists,
            levels,
            ..
        } = work;
        let (ne, nl) = (statics.n_elements, statics.n_levels);
        let on = lists.on(id);
        let device_plan = StagePlanDevice::of(plan);
        let [level, rate_of, none] = levels.of(gpu, plan);
        let Stages3D {
            inventories, rate, ..
        } = stages.as_mut().expect("sized");
        let (input, out): (&State3D, &mut State3D) = match input {
            None => (state, &mut inventories[out]),
            Some(input) => {
                assert_ne!(input, out, "a stage reads and writes distinct buffers");
                let (low, high) = inventories.split_at_mut(input.max(out));
                if input < out {
                    (&low[input], &mut high[0])
                } else {
                    (&high[0], &mut low[out])
                }
            }
        };
        let base = state;
        let (bases, inputs, rates) = (
            stage_fields(base, nl),
            stage_fields(input, nl),
            stage_fields(rate, nl),
        );
        for (f, (out, per_node)) in stage_fields_mut(out, nl).into_iter().enumerate() {
            if out.is_empty() {
                continue;
            }
            let (x, y, r) = (bases[f].0, inputs[f].0, rates[f].0);
            let per = NN * per_node;
            let n = on.count(ne) * per;
            // SAFETY: every field holds `per` values of every element (the
            // same shape); `out` is another buffer than its inputs.
            unsafe {
                cuda(
                    gpu.module_model
                        .combine(
                            &gpu.stream,
                            launch(n, COLUMN_BLOCK),
                            per as u32,
                            n as u32,
                            on.listed(none),
                            device_plan,
                            level,
                            rate_of,
                            x,
                            y,
                            r,
                            out,
                        )
                        .map_err(Into::into),
                );
            }
        }
    }

    fn implicit_advection(&self, work: &mut Work3D, out: usize, elements: Blocks<'_>, dt: &[f64]) {
        if work.implicit.is_none() {
            return;
        }
        let id = work.list(elements);
        let Work3D {
            gpu,
            statics,
            implicit,
            implicit_dt,
            stages,
            lists,
            ..
        } = work;
        let implicit = implicit.as_ref().expect("checked above");
        // A stage's steps are the same in every pass of a coarse step: up
        // only when new (a copy waits for the queued kernels), the oldest
        // overwritten in stream order past one a stage of SSP-RK(4,3) twice
        let st = &gpu.stream;
        let at = match implicit_dt.iter().position(|(host, _)| host[..] == *dt) {
            Some(at) => at,
            None if implicit_dt.len() < 8 => {
                let device = cuda(DeviceBuffer::from_host(st, dt).map_err(Into::into));
                implicit_dt.push((dt.to_vec(), device));
                implicit_dt.len() - 1
            }
            None => {
                implicit_dt.rotate_left(1);
                let (host, device) = implicit_dt.last_mut().expect("eight kept");
                host.copy_from_slice(dt);
                cuda(device.copy_from_host(st, dt).map_err(Into::into));
                implicit_dt.len() - 1
            }
        };
        let stage = &mut stages.as_mut().expect("sized").inventories[out];
        cuda(implicit.relax(statics, stage, &implicit_dt[at].1, lists.on(id)));
    }

    fn stage_values(
        &self,
        work: &mut Work3D,
        inventories: usize,
        (read, write): (usize, usize),
        elements: Blocks<'_>,
        end: bool,
    ) {
        let id = work.list(elements);
        let Work3D {
            statics,
            hooks,
            merged,
            stages,
            lists,
            ..
        } = work;
        let on = lists.on(id);
        let Stages3D {
            inventories: all_inventories,
            values,
            ..
        } = stages.as_mut().expect("sized");
        let inv = &mut all_inventories[inventories];
        if read == write {
            let out = &mut values[write];
            cuda(hooks.stage_values(statics, merged, inv, out, None, end, on));
        } else {
            let (from, to) = read_write(values, read, write);
            cuda(hooks.stage_values(statics, merged, inv, to, Some(from), end, on));
        }
    }

    fn accumulate_stage(
        &self,
        work: &mut Work3D,
        last: usize,
        elements: Blocks<'_>,
        plan: &StagePlan<'_>,
    ) {
        let id = work.list(elements);
        let Work3D {
            gpu,
            statics,
            stages,
            lists,
            levels,
            ..
        } = work;
        let (ne, nl) = (statics.n_elements, statics.n_levels);
        let on = lists.on(id);
        let device_plan = StagePlanDevice::of(plan);
        let [level, rate, none] = levels.of(gpu, plan);
        let Stages3D {
            inventories, acc, ..
        } = stages.as_mut().expect("sized");
        let acc = acc
            .as_mut()
            .expect("the substeps' results (several levels)");
        let lasts = stage_fields(&inventories[last], nl);
        for (f, (acc, per_node)) in stage_fields_mut(acc, nl).into_iter().enumerate() {
            if acc.is_empty() {
                continue;
            }
            let per = NN * per_node;
            let n = on.count(ne) * per;
            // SAFETY: both hold `per` values of every element.
            unsafe {
                cuda(
                    gpu.module_model
                        .accumulate(
                            &gpu.stream,
                            launch(n, COLUMN_BLOCK),
                            per as u32,
                            n as u32,
                            on.listed(none),
                            device_plan,
                            level,
                            rate,
                            lasts[f].0,
                            acc,
                        )
                        .map_err(Into::into),
                );
            }
        }
    }

    fn finish_substep(
        &self,
        state: &mut State3D,
        work: &mut Work3D,
        result: Option<usize>,
        elements: Blocks<'_>,
    ) {
        let id = work.list(elements);
        let Work3D {
            statics,
            hooks,
            stages,
            lists,
            ..
        } = work;
        let nl = statics.n_levels;
        let Stages3D {
            inventories, acc, ..
        } = stages.as_mut().expect("sized");
        let from = match result {
            Some(slot) => &inventories[slot],
            None => acc
                .as_ref()
                .expect("the substeps' results (several levels)"),
        };
        // The stage fields (`copy_stage_rows`): not the density
        for ((to, per_node), (from, _)) in stage_fields_mut(state, nl)
            .into_iter()
            .zip(stage_fields(from, nl))
        {
            cuda(hooks.copy_rows(statics, per_node, from, to, lists.on(id)));
        }
    }

    fn substep_values(
        &self,
        state: &State3D,
        work: &mut Work3D,
        (last, start): (usize, usize),
        elements: Blocks<'_>,
    ) {
        let id = work.list(elements);
        let Work3D {
            statics,
            hooks,
            merged,
            stages,
            lists,
            ..
        } = work;
        let on = lists.on(id);
        let values = &mut stages.as_mut().expect("sized").values;
        let (from, to) = read_write(values, last, start);
        cuda(hooks.values_of(statics, merged, state, to, Some(from), on));
        cuda(hooks.density(statics, &mut to.values, on));
    }

    /// The vertical step of the state (the step's end) or of a values slot
    /// (a substep's end within the step), with the surface fluxes over its
    /// own sea surface and the stress at the middle of `[t, t + dt]`.
    fn vertical_terms(
        &self,
        columns: Columns,
        state: &mut State3D,
        work: &mut Work3D,
        elements: Blocks<'_>,
        t: f64,
        dt: f64,
        _drag: DragKinds,
    ) {
        // The values slot out of the work while its columns step
        let mut stages = work.stages.take();
        let target = match columns {
            Columns::State => {
                // The closure's turbulence from its first call on, as the
                // CPU's vertical diffusion allocates it
                self.allocate_turbulence(state);
                state
            }
            Columns::Values(slot) => &mut stages.as_mut().expect("sized").values[slot].values,
        };
        // The surface fluxes and stress at the middle of the step, the fluxes
        // over the columns' own sea surface
        self.surface_fluxes_at(target, t + 0.5 * dt, work);
        self.stress_at(t + 0.5 * dt, work);
        let id = work.list(elements);
        let Work3D {
            statics,
            vertical,
            stress,
            roughness,
            surface_fluxes,
            lists,
            ..
        } = work;
        let surface = SurfaceDevice {
            stress: stress.as_ref().map(|[x, y]| [x, y]),
            roughness: roughness.as_ref(),
            tracers: surface_fluxes
                .as_ref()
                .map(|w| [&w.fluxes[0], &w.fluxes[1], &w.fluxes[2]]),
        };
        cuda(vertical.step(statics, target, surface, dt, lists.on(id)));
        work.stages = stages;
    }

    fn restart_substep(
        &self,
        state: &mut State3D,
        work: &mut Work3D,
        start: usize,
        elements: Blocks<'_>,
    ) {
        let id = work.list(elements);
        let Work3D {
            gpu,
            statics: s,
            hooks,
            merged,
            stages,
            sums,
            lists,
            levels,
            ..
        } = work;
        let on = lists.on(id);
        let out = &mut stages.as_mut().expect("sized").values[start];
        // The depth means of u and v back to ū, v̄ (`reset_depth_means`)
        let values = &mut out.values;
        let (m, st) = (&gpu.module_model, &gpu.stream);
        let (nl, n) = (s.n_levels as u32, on.count(s.n_elements) * NN);
        let listed = on.listed(&levels.none);
        let [mean_u, mean_v] = sums;
        // SAFETY: every buffer holds the mesh's columns (and levels); no
        // output is an input.
        unsafe {
            cuda(
                m.depth_means(
                    st,
                    launch(n, COLUMN_BLOCK),
                    nl,
                    n as u32,
                    listed,
                    &s.d_sigma,
                    &values.u,
                    &values.v,
                    mean_u,
                    mean_v,
                )
                .map_err(Into::into),
            );
            cuda(
                m.shift_means(
                    st,
                    launch(n, COLUMN_BLOCK),
                    nl,
                    n as u32,
                    listed,
                    &values.ubar,
                    &values.vbar,
                    mean_u,
                    mean_v,
                    &mut values.u,
                    &mut values.v,
                )
                .map_err(Into::into),
            );
        }
        // Limited, its inventories the state's, masked for the next
        // substep's RHS
        cuda(hooks.post_stage(s, values, on));
        cuda(hooks.inventories_of(s, merged, values, state, true, on));
        cuda(hooks.mask(s, out, on));
    }

    fn end_step(&self, state: &mut State3D, work: &mut Work3D, last: usize) {
        let Work3D {
            gpu,
            statics: s,
            hooks,
            merged,
            pass,
            stages,
            ..
        } = work;
        let out = &mut stages.as_mut().expect("sized").values[last];
        // The last stage's values unmasked, their density the state's
        cuda(hooks.restore(s, out));
        cuda(out.stash.forget(gpu));
        std::mem::swap(&mut state.rho, &mut out.values.rho);
        // The filtered barotropic state, exactly
        let n = s.n_columns();
        let q = &pass.q_avg;
        // SAFETY: every buffer holds the mesh's nodes; no output is an input.
        unsafe {
            cuda(
                gpu.module_model
                    .end_barotropic(
                        &gpu.stream,
                        launch(n, COLUMN_BLOCK),
                        n as u32,
                        &s.bed,
                        &q.h,
                        &q.hu,
                        &q.hv,
                        &mut state.eta,
                        &mut state.ubar,
                        &mut state.vbar,
                    )
                    .map_err(Into::into),
            );
        }
        // Inventories back to velocities and concentrations with the new H_z
        cuda(hooks.to_values(s, merged, state, out, None, true, On::ALL));
        for (x, y) in state.fields_mut().into_iter().zip(out.values.fields_mut()) {
            std::mem::swap(x, y);
        }
    }

    fn restore_depth_means(&self, state: &mut State3D, work: &mut Work3D) {
        let Work3D {
            gpu,
            statics: s,
            sums,
            levels,
            ..
        } = work;
        let (m, st) = (&gpu.module_model, &gpu.stream);
        let (nl, n) = (s.n_levels as u32, s.n_columns());
        let [mean_u, mean_v] = sums;
        let all = &levels.none;
        // SAFETY: every buffer holds the mesh's columns (and levels); no
        // output is an input.
        unsafe {
            cuda(
                m.depth_means(
                    st,
                    launch(n, COLUMN_BLOCK),
                    nl,
                    n as u32,
                    all,
                    &s.d_sigma,
                    &state.u,
                    &state.v,
                    mean_u,
                    mean_v,
                )
                .map_err(Into::into),
            );
            cuda(
                m.shift_means(
                    st,
                    launch(n, COLUMN_BLOCK),
                    nl,
                    n as u32,
                    all,
                    &state.ubar,
                    &state.vbar,
                    mean_u,
                    mean_v,
                    &mut state.u,
                    &mut state.v,
                )
                .map_err(Into::into),
            );
        }
    }
}

/// `items[read]` and `items[write]` (distinct; the splitter's
/// `read_write`).
fn read_write<T>(items: &mut [T], read: usize, write: usize) -> (&T, &mut T) {
    assert_ne!(read, write, "a stage reads and writes distinct buffers");
    if read < write {
        let (low, high) = items.split_at_mut(write);
        (&low[read], &mut high[0])
    } else {
        let (low, high) = items.split_at_mut(read);
        (&high[0], &mut low[write])
    }
}

/// The device's model as `Simulation3D` runs it (4.5c): the step's bound,
/// the blow-up check and a run's callbacks on the host's copy of the state,
/// downloaded after every step's `post_process`; the density and the end
/// state's tracer limiter, density and `w` on the device.
impl<'a, BC> RunModel3D for Model3D<'a, BC>
where
    BC: SWEBoundaryCondition2D + Clone + Send + Sync,
    SWEPhysics2D<BC>: PhysicsModule<SWESolution2D>,
{
    /// Nothing: the start on the host is [`Model3D::start_from`]'s, the
    /// reference the physics' own, set before the model was built.
    fn prepare(&mut self, _state: &State3D) {}

    /// `Hydrostatic3D::compute_dt_ssp` with the bounds from the device
    /// (`Self::time_step_limits`), recorded as the CPU records them.
    fn compute_dt_ssp(
        &self,
        state: &State3D,
        cfl: f64,
        scheme: SspScheme,
        work: &mut Work3D,
    ) -> f64 {
        let (limits, _) = self.time_step_limits(state, work, cfl, scheme);
        self.physics.step_of(limits, scheme)
    }

    /// The CPU's of the whole state on the host: the local 3D steps' levels
    /// need it once a coarse step (Parked: on the device, as the bound).
    fn element_steps_into(
        &self,
        state: &State3D,
        cfl: f64,
        scheme: SspScheme,
        out: &mut [f64],
        work: &mut Work3D,
    ) {
        let mut view = self.view(state, Some(work));
        self.physics
            .element_steps_into(view.full(), cfl, scheme, out);
    }

    fn stencils(&self) -> Vec<RhsStencil> {
        ModeSplitPhysics::stage_stencils(self.physics)
    }

    fn mesh(&self) -> &Mesh2D {
        &self.physics.mesh
    }

    fn update_density(&self, state: &mut State3D) {
        Model3D::update_density(self, state);
    }

    /// `Hydrostatic3D::post_process` on the device: the tracer limiter and
    /// the density; `w` from the state's own layer transports, their thin
    /// columns at rest (`update_vertical_velocity`). The host's copy is then
    /// of the last step.
    fn post_process(&self, state: &mut State3D, work: &mut Work3D) {
        cuda(work.hooks.post_process(&work.statics, state));
        cuda(work.forcing.layer_transports(state));
        let (gpu, s) = (&self.gpu, &work.statics);
        let (nl, n) = (s.n_levels as u32, s.n_columns());
        // SAFETY: Ω holds every column's σ-surfaces, `w` its layers.
        unsafe {
            cuda(
                gpu.module_model
                    .w_of_omega(
                        &gpu.stream,
                        launch(n, COLUMN_BLOCK),
                        nl,
                        n as u32,
                        &work.forcing.layers.omega,
                        &mut work.w,
                    )
                    .map_err(Into::into),
            );
        }
        self.host_copy().copied = Copied::Nothing;
    }

    /// `non_finite_field` on the device: every element's mask of the fields
    /// with a value that is not finite there, and how many values; the masks
    /// to the host only after a blow-up, for `non_finite_of_masks`.
    fn non_finite(&self, state: &State3D, work: &mut Work3D) -> Option<NonFinite> {
        let (gpu, s) = (&self.gpu, &work.statics);
        let st = &gpu.stream;
        let [masks, count] = &mut work.non_finite;
        cuda(masks.zero_async(st).map_err(Into::into));
        cuda(count.zero_async(st).map_err(Into::into));
        let per_element = NN * s.n_levels;
        let n = s.n_elements * per_element;
        // SAFETY: the state's fields are this mesh's; `masks` holds a u32 per
        // element, `count` one.
        unsafe {
            cuda(
                gpu.module_model
                    .non_finite_masks(
                        st,
                        launch(n, COLUMN_BLOCK),
                        per_element as u32,
                        NN as u32,
                        n as u32,
                        &state.eta,
                        &state.u,
                        &state.v,
                        &state.temp,
                        &state.salt,
                        masks.cu_deviceptr() as *mut u32,
                        count.cu_deviceptr() as *mut u32,
                    )
                    .map_err(Into::into),
            );
        }
        // The step's kernels, then the count (`DG_PROFILE`: apart)
        profiling::scope("device: wait for the step", || {
            cuda(st.synchronize().map_err(Into::into))
        });
        if cuda(prefix_to_host(gpu, count, 1))[0] == 0 {
            return None;
        }
        non_finite_of_masks(&cuda(prefix_to_host(gpu, masks, s.n_elements)))
    }
}

/// The device's state on the host as a run's callback asks for it
/// ([`Model3D::view`]): each part copied the first time it is asked for
/// after a step, and kept for the rest of the step's callbacks.
pub struct HostView3D<'m> {
    gpu: &'m Gpu,
    device: &'m State3D,
    work: Option<&'m Work3D>,
    host: MutexGuard<'m, HostCopy>,
}

impl HostView3D<'_> {
    /// At least `what` of the device's state on the host.
    fn copy(&mut self, what: Copied) {
        if self.host.copied >= what {
            return;
        }
        let (gpu, device, work) = (self.gpu, self.device, self.work);
        let host = &mut self.host.state;
        profiling::scope("device: state to the host", || {
            let st = &gpu.stream;
            match what {
                Copied::Nothing => {}
                Copied::DepthMeans => {
                    for (to, from) in [
                        (&mut host.eta.data, &device.eta),
                        (&mut host.ubar.data, &device.ubar),
                        (&mut host.vbar.data, &device.vbar),
                    ] {
                        cuda(from.copy_to_host(st, to).map_err(Into::into));
                    }
                }
                Copied::All => {
                    cuda(device.download_into(gpu, host));
                    if let Some(work) = work {
                        cuda(work.w.copy_to_host(st, &mut host.w).map_err(Into::into));
                        work.eddy_coefficients_into(host);
                    }
                }
            }
        });
        self.host.copied = what;
    }
}

impl StateView3D for HostView3D<'_> {
    fn depth_means(&mut self) -> &Solution3D {
        self.copy(Copied::DepthMeans);
        &self.host.state
    }

    fn full(&mut self) -> &Solution3D {
        self.copy(Copied::All);
        &self.host.state
    }
}

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
//! The device takes global 3D steps (the case runner's default; local time
//! steps of the 3D stages are 4.5d). The forcing a trait object evaluates
//! on the host (a surface stress field, the rivers' tracers) is evaluated
//! there at the CPU's times and uploaded. Not yet on the device: a nesting
//! band and the surface heat and fresh-water fluxes (4.5c), the viscosity
//! once per step; [`Model3D::new`] refuses them by name.

// As `three_d.rs`: the device loops index arrays by the loop variable.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use std::sync::Arc;

use cuda_core::DeviceBuffer;
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;
use dgcom::boundary::SWEBoundaryCondition2D;
use dgcom::physics::{
    GlsMixing, Hydrostatic3D, LinearEOS, PhysicsModule, SWEPhysics2D, VerticalMixing,
};
use dgcom::solver::rhs::{MomentumAdvectionForm, PressureGradientForm};
use dgcom::solver::{Blocks, SWESolution2D};
use dgcom::time::{
    Columns, DragKinds, Integrable, MultirateStats, SplitModel, SplitPass, SplitWork, SspScheme,
    StageElements, StagePlan, StageRhs, StageSlots, filtered_state, shu_osher,
};
use dgcom::vertical::{SigmaGrid, depth_average};

use crate::baroclinic::{Baroclinic3D, BaroclinicSetup};
use crate::forcing::{ForcingSetup, SlowForcing3D};
use crate::hooks::{Merged3D, StageHooks3D, StageValues3D, Stash3D, State3D};
use crate::limiter::{Limiter3D, VertexPatches};
use crate::pass::{BarotropicPass3D, DevicePass, LocalPass3D};
use crate::swe2d::{DeviceState, Gpu, GpuSWE2D, NN};
use crate::three_d::{
    Advected, COLUMN_BLOCK, Interface, RiversDevice, Statics3D, Transport3D, launch,
};
use crate::vertical::{Implicit3D, SurfaceDevice, Vertical3D, VerticalSetup, caged_columns};
use crate::viscosity::Viscosity3D;

// =============================================================================
// Kernels
// =============================================================================

#[cuda_module]
pub mod kernels_model {
    use super::*;

    /// `combine_stage` of one field on every element at one stage, a thread
    /// per value: `out = a·base + b·input + c·rate` as `shu_osher` sums it.
    #[kernel]
    pub fn combine(
        n: u32,
        a: f64,
        b: f64,
        c: f64,
        base: &[f64],
        input: &[f64],
        rate: &[f64],
        mut out: DisjointSlice<f64>,
    ) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        // SAFETY: thread i alone writes value i (i < n).
        unsafe {
            *out.get_unchecked_mut(i) = shu_osher(base[i], input[i], rate[i], a, b, c);
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

    /// The depth means of every column of `u` and `v` (`depth_average`), a
    /// thread per column.
    #[kernel]
    pub fn depth_means(
        nl: u32,
        n: u32,
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
        let nl = nl as usize;
        let column = idx * nl..(idx + 1) * nl;
        // SAFETY: thread idx alone writes column idx (idx < n).
        unsafe {
            *mean_u.get_unchecked_mut(idx) = depth_average(&u[column.clone()], d_sigma);
            *mean_v.get_unchecked_mut(idx) = depth_average(&v[column], d_sigma);
        }
    }

    /// `shift_columns` of `u` and `v` from their depth means to `ū`, `v̄`, a
    /// thread per column.
    #[kernel]
    pub fn shift_means(
        nl: u32,
        n: u32,
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
}

// =============================================================================
// Host side
// =============================================================================

/// The splitter's failure on the device: a CUDA call failed.
fn cuda<T>(result: Result<T, Box<dyn std::error::Error>>) -> T {
    result.unwrap_or_else(|e| panic!("CUDA: {e}"))
}

/// Global steps only: `elements` is every element.
fn all(elements: Blocks<'_>) {
    assert!(
        elements.is_all(),
        "the device takes global 3D steps (local time steps of the 3D stages: P2.7 4.5d)"
    );
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
    /// One stage's RHS
    rate: State3D,
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
    /// `G` of the last three steps, newest first, and their step average
    history: [DeviceState; 3],
    average: DeviceState,
    /// The barotropic transport during the pass
    q: DeviceState,
    /// The column mean `Λ̄` of the layer drag
    column_drag: DeviceBuffer<f64>,
    /// A stage's `βΔt` of every element
    implicit_dt: DeviceBuffer<f64>,
    stages: Option<Stages3D>,
    /// The columns' sums or means of u and v
    sums: [DeviceBuffer<f64>; 2],
    /// The surface stress field (host and device), the rivers' T and S
    stress_host: [Vec<f64>; 2],
    stress: Option<[DeviceBuffer<f64>; 2]>,
    river_conc: Option<[DeviceBuffer<f64>; 2]>,
    roughness: Option<DeviceBuffer<f64>>,
}

impl Work3D {
    fn stages(&mut self) -> &mut Stages3D {
        self.stages
            .as_mut()
            .expect("the stages' buffers are sized at the step's start (start_stages)")
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
        if physics.nesting.is_some() {
            return refuse("a nesting band (P2.7 4.5c)");
        }
        if physics.surface_fluxes.is_some() {
            return refuse("the surface heat and fresh-water fluxes (P2.7 4.5c)");
        }
        if physics.viscosity_once_per_step() {
            return refuse("the viscosity once per step");
        }
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
        Ok(Self {
            gpu: gpu.clone(),
            physics,
            module,
            element_local,
            statics,
            density,
        })
    }

    /// The density of `state`'s T and S (`Hydrostatic3D::update_density`,
    /// before every step).
    pub fn update_density(&self, state: &mut State3D) {
        cuda(self.density.density(&self.statics, state, None));
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

/// `Hydrostatic3D::momentum_rhs` of `values` on every element into `rhs_u`,
/// `rhs_v`: the reference at their levels, the pressure gradient and
/// Coriolis, the shear's viscosity, the thin columns' tendencies zero.
fn momentum(
    s: &Statics3D,
    baroclinic: &mut Baroclinic3D,
    viscosity: &mut Viscosity3D,
    hooks: &StageHooks3D,
    values: &State3D,
    rhs_u: &mut DeviceBuffer<f64>,
    rhs_v: &mut DeviceBuffer<f64>,
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
    ));
    cuda(viscosity.apply(s, &values.eta, &values.u, &values.v, rhs_u, rhs_v));
    cuda(hooks.zero_thin_momentum(s, &values.eta, rhs_u, rhs_v));
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
            history,
            average: DeviceState::upload(gpu, &like),
            q: DeviceState::upload(gpu, &like),
            column_drag: zeros(n_columns),
            implicit_dt: zeros(ne),
            stages: None,
            sums: [zeros(n_columns), zeros(n_columns)],
            stress_host: [Vec::new(), Vec::new()],
            stress: None,
            river_conc: rivers.map(|r| [zeros(r.len()), zeros(r.len())]),
            roughness: p.surface_roughness.as_deref().map(up),
            statics,
        }
    }

    fn start_turbulence(&self, _state: &mut State3D) {
        panic!("the device takes global 3D steps (local time steps of the 3D stages: P2.7 4.5d)");
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
                statics,
                baroclinic,
                viscosity,
                hooks,
                rhs_n: [rhs_u, rhs_v],
                ..
            } = work;
            momentum(statics, baroclinic, viscosity, hooks, state, rhs_u, rhs_v);
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
        assert!(!slots.acc, "the device takes global 3D steps (P2.7 4.5d)");
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
            *stages = Some(Stages3D {
                slots,
                // The combinations carry no density; the values do
                inventories: (0..slots.inventories)
                    .map(|_| zeroed_state(gpu, n_columns, n_columns * nl, n_w, false))
                    .collect(),
                values: (0..slots.values)
                    .map(|_| StageValues3D {
                        values: zeroed_state(gpu, n_columns, n_columns * nl, n_w, true),
                        stash: cuda(Stash3D::new(gpu, s)),
                    })
                    .collect(),
                rate: zeroed_state(gpu, n_columns, n_columns * nl, n_w, false),
            });
        }
        let stages = stages.as_mut().expect("just sized");
        for buffer in stages
            .inventories
            .iter_mut()
            .chain(stages.values.iter_mut().map(|v| &mut v.values))
            .chain([&mut stages.rate])
        {
            fit_turbulence(gpu, buffer, n_w);
        }
        // The values the substeps' first RHS reads: the step's start, masked
        let out = &mut stages.values[start];
        cuda(out.values.copy_from(gpu, state));
        cuda(out.stash.forget(gpu));
        cuda(hooks.mask(s, out));
        cuda(hooks.to_inventories(s, state));
        // What a flushed element's nodes show to the first stage: what
        // passes through it
        if merged.any() {
            let fields = state.fields();
            let outs = out.values.fields_mut();
            for (which, (q, out)) in fields.into_iter().zip(outs).enumerate().skip(2) {
                cuda(hooks.pass_through(s, merged, which, q, &state.eta, out, false));
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
        all(on.elements);
        match reuse_at {
            Some(t) => {
                assert_eq!(
                    on.time.at(0),
                    t,
                    "a global step's first stage is at its start"
                );
                let Work3D {
                    gpu, rhs_n, stages, ..
                } = work;
                let rate = &mut stages.as_mut().expect("sized").rate;
                for (to, from) in [&mut rate.u, &mut rate.v].into_iter().zip(&*rhs_n) {
                    cuda(
                        to.copy_from_device_async(from, &gpu.stream)
                            .map_err(Into::into),
                    );
                }
            }
            None => {
                let Work3D {
                    statics,
                    baroclinic,
                    viscosity,
                    hooks,
                    stages,
                    ..
                } = work;
                let Stages3D {
                    values: all_values,
                    rate,
                    ..
                } = stages.as_mut().expect("sized");
                let values = &all_values[values].values;
                let (u, v) = (&mut rate.u, &mut rate.v);
                momentum(statics, baroclinic, viscosity, hooks, values, u, v);
            }
        }
    }

    fn stage_transport(&self, state: &State3D, work: &mut Work3D, rhs: StageRhs<'_>) {
        all(rhs.on.elements);
        let p = self.physics;
        let time = rhs.on.time.at(0);
        // The rivers' T and S at the stage's time, as the CPU evaluates them
        if let (Some(rivers), Some([conc_t, conc_s])) = (p.rivers.as_deref(), &mut work.river_conc)
        {
            let temp: Vec<f64> = (0..rivers.len())
                .map(|i| rivers.river(i).temperature.at(time))
                .collect();
            let salt: Vec<f64> = (0..rivers.len())
                .map(|i| rivers.river(i).salinity.at(time))
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
            ..
        } = work;
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
        cuda(transport.compute(&vals.eta, &vals.u, &vals.v, Some(&pass.flux), None));
        if let Some(implicit) = implicit {
            cuda(implicit.split(s, &vals.eta, &pass.flux.euler_dt, &mut transport.omega));
        }
        cuda(baroclinic.reference(s, &vals.eta, &vals.rho));
        // The advection with the water no transport carried, field by field
        let velocity = |field| Advected {
            field,
            reference: None,
            range: dgcom::solver::rhs::UNBOUNDED,
            exterior: None,
            bc: dgcom::solver::rhs::DeviceTracerBC3D::Extrapolation,
        };
        cuda(transport.advect(
            [&velocity(&vals.u), &velocity(&vals.v)],
            [&mut rate.u, &mut rate.v],
            p.momentum_vertical_advection,
            Interface::Central,
            false,
            true,
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
                        exterior: None,
                        bc: form(&*p.temp_bc),
                    },
                    &Advected {
                        field: &vals.salt,
                        reference: reference(1),
                        range: ranges[1],
                        exterior: None,
                        bc: form(&*p.salt_bc),
                    },
                ],
                [&mut rate.temp, &mut rate.salt],
                p.vertical_advection,
                Interface::DonorBesideThin,
                true,
                !referenced,
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
            ));
        }
        if !vals.tke.is_empty() {
            match p.turbulence_advection {
                Some(scheme) => cuda(transport.advect_w(
                    [&vals.tke, &vals.gls],
                    [&mut rate.tke, &mut rate.gls],
                    scheme,
                    true,
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
        all(elements);
        let stage = plan.at(0);
        let gpu = work.gpu.clone();
        let Stages3D {
            inventories, rate, ..
        } = work.stages();
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
        let fields = |s: &State3D| -> [*const DeviceBuffer<f64>; 9] {
            [
                &s.u, &s.v, &s.temp, &s.salt, &s.tke, &s.gls, &s.eta, &s.ubar, &s.vbar,
            ]
        };
        let (bases, inputs, rates) = (fields(base), fields(input), fields(rate));
        let outs = [
            &mut out.u,
            &mut out.v,
            &mut out.temp,
            &mut out.salt,
            &mut out.tke,
            &mut out.gls,
            &mut out.eta,
            &mut out.ubar,
            &mut out.vbar,
        ];
        for (f, out) in outs.into_iter().enumerate() {
            if out.is_empty() {
                continue;
            }
            // SAFETY: the pointers are the fields of `base`, `input` and
            // `rate`, alive and not written through `out` (another buffer).
            let (x, y, r) = unsafe { (&*bases[f], &*inputs[f], &*rates[f]) };
            let n = out.len();
            // SAFETY: every field holds `n` values (the same shape).
            unsafe {
                cuda(
                    gpu.module_model
                        .combine(
                            &gpu.stream,
                            launch(n, COLUMN_BLOCK),
                            n as u32,
                            stage.a,
                            stage.b,
                            stage.c,
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
        all(elements);
        let Work3D {
            gpu,
            statics,
            implicit,
            implicit_dt,
            stages,
            ..
        } = work;
        let Some(implicit) = implicit else {
            return;
        };
        cuda(
            implicit_dt
                .copy_from_host(&gpu.stream, dt)
                .map_err(Into::into),
        );
        let stage = &mut stages.as_mut().expect("sized").inventories[out];
        cuda(implicit.relax(statics, stage, implicit_dt));
    }

    fn stage_values(
        &self,
        work: &mut Work3D,
        inventories: usize,
        (read, write): (usize, usize),
        elements: Blocks<'_>,
        end: bool,
    ) {
        all(elements);
        assert_eq!(read, write, "global steps update their values in place");
        let Work3D {
            statics,
            hooks,
            merged,
            stages,
            ..
        } = work;
        let Stages3D {
            inventories: all_inventories,
            values,
            ..
        } = stages.as_mut().expect("sized");
        let inv = &mut all_inventories[inventories];
        cuda(hooks.stage_values(statics, merged, inv, &mut values[write], None, end));
    }

    fn accumulate_stage(
        &self,
        _work: &mut Work3D,
        _last: usize,
        _elements: Blocks<'_>,
        _plan: &StagePlan<'_>,
    ) {
        panic!("the device takes global 3D steps (local time steps of the 3D stages: P2.7 4.5d)");
    }

    fn finish_substep(
        &self,
        state: &mut State3D,
        work: &mut Work3D,
        result: Option<usize>,
        elements: Blocks<'_>,
    ) {
        all(elements);
        let slot = result.expect("global steps: the last stage is the result");
        let gpu = work.gpu.clone();
        let from = &work.stages().inventories[slot];
        let st = &gpu.stream;
        // The stage fields (`copy_stage_rows`): not the density
        for (to, from) in state.fields_mut().into_iter().zip(from.fields()) {
            if !to.is_empty() {
                cuda(to.copy_from_device_async(from, st).map_err(Into::into));
            }
        }
        cuda(state.copy_barotropic(&gpu, from));
    }

    fn substep_values(
        &self,
        _state: &State3D,
        _work: &mut Work3D,
        _last_start: (usize, usize),
        _elements: Blocks<'_>,
    ) {
        panic!("the device takes global 3D steps (local time steps of the 3D stages: P2.7 4.5d)");
    }

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
        all(elements);
        assert_eq!(columns, Columns::State, "global steps end in the state");
        let p = self.physics;
        // The closure's turbulence from its first call on, as the CPU's
        // vertical diffusion allocates it
        if let Some([k0, psi0]) = p.mixing.initial_turbulence() {
            let n_w = p.mesh.n_elements * NN * (p.sigma.n_levels() + 1);
            if state.tke.len() != n_w {
                let st = &self.gpu.stream;
                let up =
                    |v: f64| cuda(DeviceBuffer::from_host(st, &vec![v; n_w]).map_err(Into::into));
                state.tke = up(k0);
                state.gls = up(psi0);
            }
        }
        // The surface stress at the middle of the step
        self.stress_at(t + 0.5 * dt, work);
        let Work3D {
            statics,
            vertical,
            stress,
            roughness,
            ..
        } = work;
        let surface = SurfaceDevice {
            stress: stress.as_ref().map(|[x, y]| [x, y]),
            roughness: roughness.as_ref(),
            tracers: None,
        };
        cuda(vertical.step(statics, state, surface, dt));
    }

    fn restart_substep(
        &self,
        _state: &mut State3D,
        _work: &mut Work3D,
        _start: usize,
        _elements: Blocks<'_>,
    ) {
        panic!("the device takes global 3D steps (local time steps of the 3D stages: P2.7 4.5d)");
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
        cuda(hooks.to_values(s, merged, state, out, None, true));
        for (x, y) in state.fields_mut().into_iter().zip(out.values.fields_mut()) {
            std::mem::swap(x, y);
        }
    }

    fn restore_depth_means(&self, state: &mut State3D, work: &mut Work3D) {
        let Work3D {
            gpu,
            statics: s,
            sums,
            ..
        } = work;
        let (m, st) = (&gpu.module_model, &gpu.stream);
        let (nl, n) = (s.n_levels as u32, s.n_columns());
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

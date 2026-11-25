//! The 3D step's bound on the GPU (P2.7 Phase 4.5e, `notes/gpu.md`):
//! dgcom's `Hydrostatic3D::time_step_limits` of a device state, so a run
//! needs no copy of the state on the host to take its next step.
//!
//! Each element's rates are dgcom's own functions, called by a thread per
//! element: the advective and vertical ones of its columns
//! (`element_column_rates`, with `math::hypot` for the speeds: the same bits
//! everywhere) and the bound on its sloping levels' frequency
//! (`sloping_level_rate_of` over [`DeviceElement`], the element's view of
//! [`Statics3D`]'s tables). The elements whose bound comes near the step the
//! others allow have their frequency measured (`sloping_level_frequency_of`,
//! a power iteration): they are listed on the device and taken by a fixed
//! grid, each thread with its own scratch in global memory (as a thread's
//! stack the 5·NN·levels numbers would reserve gigabytes over the card). The
//! static bounds (Coriolis, a viscosity without Smagorinsky) are the host's,
//! once; Smagorinsky's largest viscosity of every element is the device's
//! viscosity module's (`Viscosity3D::largest`). Every per-element step comes
//! to the host, which takes the smallest with its element in the CPU's order
//! (`TimeStepLimit::min`: exact in any order) and the step from them
//! (`TimeStepLimits::step`). So a build without multiply–add contraction
//! (`--no-fmad`) gives the CPU's bits (`bound3d` gate in `step3d.rs`).

// As `three_d.rs`: the device loops index arrays by the loop variable.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use std::sync::Arc;

use cuda_core::DeviceBuffer;
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;
use dgcom::boundary::SWEBoundaryCondition2D;
use dgcom::physics::{
    ColumnState, GlsMixing, Hydrostatic3D, LinearEOS, PhysicsModule, SWEPhysics2D, TimeStepLimit,
    TimeStepLimits, element_column_rates, inverse_length, sloping_level_limit,
    sloping_level_step_unmeasured,
};
use dgcom::solver::SWESolution2D;
use dgcom::solver::rhs::baroclinic::{
    ElementReference, SlopingElement, sloping_level_frequency_of, sloping_level_rate_of,
};
use dgcom::solver::rhs::subcells::subcell_slot;
use dgcom::solver::rhs::{viscous_dt, viscous_metric};
use dgcom::time::SspScheme;
use dgcom::types::ElementIndex;

use crate::baroclinic::{Baroclinic3D, ReferenceFieldsDevice, pair_at};
use crate::hooks::State3D;
use crate::swe2d::{Gpu, N1, NN};
use crate::three_d::{COLUMN_BLOCK, NSUB, Statics3D, launch};
use crate::viscosity::Viscosity3D;

/// Threads of the frequencies' grid: each takes every this many'th listed
/// element, with scratch for one.
const FREQUENCY_THREADS: usize = 2048;

/// Element `k` as dgcom's sloping-level bounds read it, from [`Statics3D`]'s
/// tables and the state's free surface.
pub struct DeviceElement<'a> {
    pub k: usize,
    pub averaged: bool,
    pub eta: &'a [f64],
    pub bed: &'a [f64],
    pub mass: &'a [f64],
    pub contra: &'a [[f64; 4]],
    pub jinv: &'a [f64],
    pub dr: &'a [f64],
    pub ds: &'a [f64],
    pub weights: &'a [f64],
    pub sub_metric: &'a [[f64; 2]],
}

impl SlopingElement for DeviceElement<'_> {
    #[inline(always)]
    fn n_nodes(&self) -> usize {
        NN
    }

    #[inline(always)]
    fn n_1d(&self) -> usize {
        N1
    }

    #[inline(always)]
    fn eta(&self, i: usize) -> f64 {
        self.eta[self.k * NN + i]
    }

    #[inline(always)]
    fn bed(&self, i: usize) -> f64 {
        self.bed[self.k * NN + i]
    }

    #[inline(always)]
    fn node_mass(&self, i: usize) -> f64 {
        self.mass[self.k * NN + i]
    }

    #[inline(always)]
    fn pair(&self, i: usize, j: usize) -> [(f64, f64); 2] {
        pair_at(
            self.k,
            i,
            j,
            self.averaged,
            self.contra,
            self.jinv,
            self.dr,
            self.ds,
        )
    }

    #[inline(always)]
    fn apart(&self, a: usize, b: usize) -> bool {
        self.dr[a * NN + b] == 0.0 && self.ds[a * NN + b] == 0.0
    }

    #[inline(always)]
    fn weight_1d(&self, at: usize) -> f64 {
        self.weights[at]
    }

    /// `Statics3D::sub_metric` holds `telescoped_interfaces` of each line's
    /// `line_metric`, as `CpuElement` computes them.
    #[inline(always)]
    fn line_interfaces(&self, dir: usize, at: usize, out: &mut [(f64, f64)]) {
        for a in 0..N1 - 1 {
            let m = self.sub_metric[self.k * NSUB + subcell_slot(N1, dir, at, a)];
            out[a + 1] = (m[0], m[1]);
        }
    }
}

// =============================================================================
// Kernels
// =============================================================================

#[cuda_module]
pub mod kernels_bound {
    use super::*;

    /// Every element's advective step `cfl / wave / order_factor`, its
    /// vertical one `1 / vertical` (`element_column_rates`) and the bound on
    /// its sloping levels' frequency (`sloping_level_rate_of` about the
    /// reference's `stratification`; zero without one: `stratification`
    /// empty), one thread per element (`Hydrostatic3D::element_rates`).
    #[kernel]
    pub fn bound_rates(
        ne: u32,
        nl: u32,
        cfl: f64,
        order_factor: f64,
        g: f64,
        rho0: f64,
        buoyancy: f64,
        min_column_depth: f64,
        eta: &[f64],
        u: &[f64],
        v: &[f64],
        w: &[f64],
        rho: &[f64],
        inverse_lengths: &[f64],
        bed: &[f64],
        mass: &[f64],
        averaged: &[u32],
        contra: &[[f64; 4]],
        jinv: &[f64],
        dr: &[f64],
        ds: &[f64],
        weights: &[f64],
        sub_metric: &[[f64; 2]],
        sigma_rho: &[f64],
        d_sigma: &[f64],
        stratification: &[f64],
        mut advection: DisjointSlice<f64>,
        mut vertical: DisjointSlice<f64>,
        mut sloping: DisjointSlice<f64>,
    ) {
        let k = thread::index_1d().get();
        if k >= ne as usize {
            return;
        }
        let nl = nl as usize;
        let (sigma_rho, d_sigma) = (&sigma_rho[..nl], &d_sigma[..nl]);
        let column = |i: usize| {
            let idx = k * NN + i;
            // `Hydrostatic3D::is_thin`
            if eta[idx] - bed[idx] < min_column_depth {
                return None;
            }
            let layers = idx * nl..(idx + 1) * nl;
            Some(ColumnState {
                depth: eta[idx] - bed[idx],
                u: &u[layers.clone()],
                v: &v[layers.clone()],
                w: &w[layers.clone()],
                rho: &rho[layers],
                inverse_length: inverse_lengths[idx],
            })
        };
        let rates = element_column_rates(NN, sigma_rho, d_sigma, g, rho0, column);
        let rate = if stratification.is_empty() {
            0.0
        } else {
            let element = DeviceElement {
                k,
                averaged: averaged[k] != 0,
                eta,
                bed,
                mass,
                contra,
                jinv,
                dr,
                ds,
                weights,
                sub_metric,
            };
            let mut scratch = [0.0; 5 * NN];
            sloping_level_rate_of(
                &element,
                &stratification[k * NN * nl..(k + 1) * NN * nl],
                sigma_rho,
                d_sigma,
                buoyancy,
                min_column_depth,
                &mut scratch,
            )
        };
        // SAFETY: thread k alone writes element k.
        unsafe {
            *advection.get_unchecked_mut(k) = cfl / rates.wave / order_factor;
            *vertical.get_unchecked_mut(k) = 1.0 / rates.vertical;
            *sloping.get_unchecked_mut(k) = rate;
        }
    }

    /// Every element's step on its sloping levels that needs no measured
    /// frequency (`sloping_level_step_unmeasured` of its bound `sloping`,
    /// against the step `needed` of the others, or the element's own of
    /// `needed_each` if that is not empty, the scheme's `limit`); the
    /// others listed in `candidates` (`count[0]` of them, zeroed before; the
    /// list's order the atomics', which no result depends on), their steps
    /// left to [`sloping_frequencies`].
    #[kernel]
    pub fn sloping_steps(
        ne: u32,
        limit: f64,
        needed: f64,
        needed_each: &[f64],
        min_column_depth: f64,
        eta: &[f64],
        bed: &[f64],
        sloping: &[f64],
        count: *mut u32,
        mut candidates: DisjointSlice<u32>,
        mut steps: DisjointSlice<f64>,
    ) {
        let k = thread::index_1d().get();
        if k >= ne as usize {
            return;
        }
        let mut thin = false;
        for i in 0..NN {
            let idx = k * NN + i;
            thin |= eta[idx] - bed[idx] < min_column_depth;
        }
        let needed = if needed_each.is_empty() {
            needed
        } else {
            needed_each[k]
        };
        match sloping_level_step_unmeasured(sloping[k], thin, limit, needed) {
            // SAFETY: thread k alone writes element k.
            Some(dt) => unsafe { *steps.get_unchecked_mut(k) = dt },
            None => {
                // SAFETY: `count` points to a u32 that only atomics touch
                // here; every slot it hands out is below `ne`.
                unsafe {
                    let count = DeviceAtomicU32::from_ptr(count);
                    let slot = count.fetch_add(1, AtomicOrdering::Relaxed) as usize;
                    *candidates.get_unchecked_mut(slot) = k as u32;
                }
            }
        }
    }

    /// The listed elements' steps `limit / frequency`
    /// (`sloping_level_frequency_of` about the reference `density`,
    /// `stratification`, `residual`), a fixed grid of `threads` taking every
    /// `threads`'th of them, thread `t` with the scratch from
    /// `t·5·NN·nl` of `scratch`.
    #[kernel]
    pub fn sloping_frequencies(
        threads: u32,
        nl: u32,
        limit: f64,
        buoyancy: f64,
        min_column_depth: f64,
        count: &[u32],
        candidates: &[u32],
        eta: &[f64],
        bed: &[f64],
        mass: &[f64],
        averaged: &[u32],
        contra: &[[f64; 4]],
        jinv: &[f64],
        dr: &[f64],
        ds: &[f64],
        weights: &[f64],
        sub_metric: &[[f64; 2]],
        sigma_rho: &[f64],
        d_sigma: &[f64],
        density: &[f64],
        stratification: &[f64],
        residual: &[f64],
        scratch: *mut f64,
        mut steps: DisjointSlice<f64>,
    ) {
        let t = thread::index_1d().get();
        let (threads, nl) = (threads as usize, nl as usize);
        let (sigma_rho, d_sigma) = (&sigma_rho[..nl], &d_sigma[..nl]);
        let per_element = NN * nl;
        // SAFETY: `scratch` holds `5·NN·nl` numbers for each of `threads`
        // threads; thread t alone touches its own.
        let scratch = unsafe {
            core::slice::from_raw_parts_mut(scratch.add(t * 5 * per_element), 5 * per_element)
        };
        let mut c = t;
        while c < count[0] as usize {
            let k = candidates[c] as usize;
            let element = DeviceElement {
                k,
                averaged: averaged[k] != 0,
                eta,
                bed,
                mass,
                contra,
                jinv,
                dr,
                ds,
                weights,
                sub_metric,
            };
            let block = k * per_element..(k + 1) * per_element;
            let frequency = sloping_level_frequency_of(
                &element,
                ElementReference {
                    density: &density[block.clone()],
                    stratification: &stratification[block.clone()],
                    residual: &residual[block],
                },
                sigma_rho,
                d_sigma,
                buoyancy,
                min_column_depth,
                scratch,
            );
            // A listed element has no thin column (`sloping_steps`)
            let dt = match frequency {
                Some(frequency) => limit / frequency,
                None => f64::NAN,
            };
            // SAFETY: an element is listed once.
            unsafe { *steps.get_unchecked_mut(k) = dt };
            c += threads;
        }
    }
}

// =============================================================================
// Host side
// =============================================================================

/// The viscosity's bound of [`Bound3D`].
enum ViscosityBound {
    /// None
    Zero,
    /// A background alone: the CPU's, once per Courant number
    Background(Option<(f64, TimeStepLimit)>),
    /// Smagorinsky's: every element's static metric (`viscous_metric`), the
    /// device's largest viscosity per element and level and per element,
    /// and the host's copy
    Smagorinsky {
        metric: Vec<f64>,
        levels: DeviceBuffer<f64>,
        largest: DeviceBuffer<f64>,
        host: Vec<f64>,
    },
}

/// The step's bound of a device state ([`Self::limits`]): its buffers, and
/// what of it is static.
pub struct Bound3D {
    gpu: Arc<Gpu>,
    /// `1/h` at every node (`inverse_length`)
    inverse_lengths: DeviceBuffer<f64>,
    /// Every element's advective and vertical steps, its sloping levels'
    /// bound and step
    advection: DeviceBuffer<f64>,
    vertical: DeviceBuffer<f64>,
    sloping: DeviceBuffer<f64>,
    steps: DeviceBuffer<f64>,
    /// The elements whose frequency is measured, and how many
    candidates: DeviceBuffer<u32>,
    count: DeviceBuffer<u32>,
    /// The frequencies' scratch (empty without a reference)
    scratch: DeviceBuffer<f64>,
    /// Coriolis's bound (static), and every element's
    coriolis: TimeStepLimit,
    coriolis_each: Vec<f64>,
    /// Every element's step from its other bounds, which its sloping
    /// levels' step is weighed against (the element steps)
    needed: DeviceBuffer<f64>,
    viscosity: ViscosityBound,
    /// The steps on the host
    host: [Vec<f64>; 3],
}

impl Bound3D {
    /// The bound of `physics` on `statics`' mesh, with or without a
    /// reference (`reference`: the frequencies' scratch).
    pub fn new<BC: SWEBoundaryCondition2D>(
        gpu: &Arc<Gpu>,
        statics: &Statics3D,
        physics: &Hydrostatic3D<LinearEOS, GlsMixing, BC>,
        reference: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let st = &gpu.stream;
        let (ne, nl) = (statics.n_elements, statics.n_levels);
        let (mesh, geom) = (&physics.mesh, &physics.geom);
        let lengths: Vec<f64> = (0..ne * NN)
            .map(|idx| inverse_length(geom, idx / NN, idx % NN))
            .collect();
        // `Hydrostatic3D::time_step_limits`' Coriolis bound, element by
        // element
        let coriolis_each: Vec<f64> = (0..ne)
            .map(|k| {
                let f = mesh
                    .element_vertices(ElementIndex::new(k))
                    .iter()
                    .map(|&[_, y]| physics.coriolis.f_at(y).abs())
                    .fold(0.0, f64::max);
                1.0 / f
            })
            .collect();
        let coriolis = smallest(&coriolis_each);
        let nu = physics.horizontal_viscosity;
        let viscosity = if nu.is_zero() {
            ViscosityBound::Zero
        } else if nu.smagorinsky == 0.0 {
            ViscosityBound::Background(None)
        } else {
            ViscosityBound::Smagorinsky {
                metric: (0..ne)
                    .map(|k| viscous_metric(mesh, &physics.ops, geom, k))
                    .collect(),
                levels: DeviceBuffer::zeroed(st, ne * nl)?,
                largest: DeviceBuffer::zeroed(st, ne)?,
                host: vec![0.0; ne],
            }
        };
        let zeros = |n: usize| DeviceBuffer::<f64>::zeroed(st, n);
        Ok(Self {
            gpu: gpu.clone(),
            inverse_lengths: DeviceBuffer::from_host(st, &lengths)?,
            advection: zeros(ne)?,
            vertical: zeros(ne)?,
            sloping: zeros(ne)?,
            steps: zeros(ne)?,
            candidates: DeviceBuffer::zeroed(st, ne)?,
            count: DeviceBuffer::zeroed(st, 1)?,
            scratch: zeros(if reference {
                FREQUENCY_THREADS * 5 * NN * nl
            } else {
                0
            })?,
            coriolis,
            coriolis_each,
            needed: zeros(ne)?,
            viscosity,
            host: [vec![0.0; ne], vec![0.0; ne], vec![0.0; ne]],
        })
    }

    /// `Hydrostatic3D::time_step_limits` of the device state `s` (its `w`
    /// in `w`) at Courant number `cfl` in `scheme`: the reference at its
    /// levels first (`baroclinic`'s, as a stage takes it), Smagorinsky's
    /// viscosity from `viscosity`; and how many elements had their
    /// sloping levels' frequency measured.
    pub fn limits<BC: SWEBoundaryCondition2D>(
        &mut self,
        statics: &Statics3D,
        physics: &Hydrostatic3D<LinearEOS, GlsMixing, BC>,
        baroclinic: &mut Baroclinic3D,
        viscosity: &mut Viscosity3D,
        s: &State3D,
        w: &DeviceBuffer<f64>,
        cfl: f64,
        scheme: SspScheme,
    ) -> Result<(TimeStepLimits, usize), Box<dyn std::error::Error>> {
        let gpu = self.gpu.clone();
        let st = &gpu.stream;
        self.rates(statics, physics, baroclinic, s, w, cfl)?;
        let viscosity = self.viscosity_limit(statics, physics, viscosity, s, cfl)?;
        let [advection_h, vertical_h, _] = &mut self.host;
        self.advection.copy_to_host(st, advection_h)?;
        self.vertical.copy_to_host(st, vertical_h)?;
        let advection = smallest(advection_h);
        let vertical = smallest(vertical_h);
        let mut limits = TimeStepLimits {
            advection,
            vertical,
            coriolis: self.coriolis,
            sloping_levels: TimeStepLimit::NONE,
            viscosity,
        };
        let Some(fields) = baroclinic.fields.as_ref() else {
            return Ok((limits, 0));
        };
        let explicit_vertical = physics.implicit_vertical_advection.is_none();
        let needed = limits.needed(explicit_vertical, scheme);
        let measured = self.sloping(statics, physics, fields, s, scheme, needed, false)?;
        self.steps.copy_to_host(st, &mut self.host[2])?;
        limits.sloping_levels = smallest(&self.host[2]);
        Ok((limits, measured))
    }

    /// `Hydrostatic3D::element_steps_into` of the device state `s` (its `w`
    /// in `w`) into `out`: every element's step for local time stepping,
    /// from the device's own rates. The advective bound is the least over
    /// the element and its face neighbours; then Coriolis's, the vertical
    /// advection's (at `LOCAL_STEP_IMPLICIT_COURANT` times its explicit
    /// bound where its excess is implicit), the sloping levels' against the
    /// element's own step so far, the viscosity's with the face neighbours'
    /// `ν`: the CPU's arithmetic on the per-element numbers, which come to
    /// the host (three arrays of `ne`, where the CPU's function took the
    /// whole state down). How many frequencies were measured.
    pub fn element_steps<BC>(
        &mut self,
        statics: &Statics3D,
        physics: &Hydrostatic3D<LinearEOS, GlsMixing, BC>,
        baroclinic: &mut Baroclinic3D,
        viscosity: &mut Viscosity3D,
        s: &State3D,
        w: &DeviceBuffer<f64>,
        cfl: f64,
        scheme: SspScheme,
        out: &mut [f64],
    ) -> Result<usize, Box<dyn std::error::Error>>
    where
        BC: SWEBoundaryCondition2D + Clone + Send + Sync,
        SWEPhysics2D<BC>: PhysicsModule<SWESolution2D>,
    {
        let gpu = self.gpu.clone();
        let st = &gpu.stream;
        let mesh = &physics.mesh;
        let ne = statics.n_elements;
        assert_eq!(out.len(), ne, "one step per element");
        self.rates(statics, physics, baroclinic, s, w, cfl)?;
        let [advection_h, vertical_h, _] = &mut self.host;
        self.advection.copy_to_host(st, advection_h)?;
        self.vertical.copy_to_host(st, vertical_h)?;
        // The vertical advection's Courant number: one where it is
        // explicit, a few where its excess is implicit
        let courant = if physics.implicit_vertical_advection.is_none() {
            1.0
        } else {
            Hydrostatic3D::<LinearEOS, GlsMixing, BC>::LOCAL_STEP_IMPLICIT_COURANT
        };
        let ssp = scheme.ssp_coefficient();
        for (k, dt) in out.iter_mut().enumerate() {
            let bound = (0..4)
                .filter_map(|face| mesh.neighbor(ElementIndex::new(k), face))
                .map(|nb| advection_h[nb.element])
                .fold(advection_h[k], f64::min);
            *dt = bound
                .min(self.coriolis_each[k])
                .min(courant * ssp * vertical_h[k]);
        }
        let mut measured = 0;
        if let Some(fields) = baroclinic.fields.as_ref() {
            self.needed.copy_from_host(st, out)?;
            measured = self.sloping(statics, physics, fields, s, scheme, 0.0, true)?;
            self.steps.copy_to_host(st, &mut self.host[2])?;
            for (dt, &sloping) in out.iter_mut().zip(&self.host[2]) {
                *dt = dt.min(sloping);
            }
        }
        // The viscosity's, with the face neighbours' (BR1 couples them in)
        let order = physics.ops.order;
        let neighbours_largest = |largest: &[f64], k: usize| {
            (0..4)
                .filter_map(|face| mesh.neighbor(ElementIndex::new(k), face))
                .map(|nb| largest[nb.element])
                .fold(largest[k], f64::max)
        };
        match &mut self.viscosity {
            ViscosityBound::Zero => {}
            ViscosityBound::Background(_) => {
                let nu = physics.horizontal_viscosity.background;
                for (k, dt) in out.iter_mut().enumerate() {
                    let metric = viscous_metric(mesh, &physics.ops, &physics.geom, k);
                    *dt = dt.min(viscous_dt(nu, metric, order, cfl));
                }
            }
            ViscosityBound::Smagorinsky {
                metric,
                levels,
                largest,
                host,
            } => {
                viscosity.largest(statics, &s.eta, &s.u, &s.v, levels, largest)?;
                largest.copy_to_host(st, host)?;
                for (k, dt) in out.iter_mut().enumerate() {
                    let nu = neighbours_largest(host, k);
                    *dt = dt.min(viscous_dt(nu, metric[k], order, cfl));
                }
            }
        }
        Ok(measured)
    }

    /// Every element's advective and vertical steps and its sloping levels'
    /// bound (`bound_rates`) into this bound's buffers, the reference at
    /// `s`'s levels first (`baroclinic`'s, as a stage takes it).
    fn rates<BC: SWEBoundaryCondition2D>(
        &mut self,
        statics: &Statics3D,
        physics: &Hydrostatic3D<LinearEOS, GlsMixing, BC>,
        baroclinic: &mut Baroclinic3D,
        s: &State3D,
        w: &DeviceBuffer<f64>,
        cfl: f64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let gpu = self.gpu.clone();
        let (m, st) = (&gpu.module_bound, &gpu.stream);
        let t = statics;
        let (ne, nl) = (t.n_elements, t.n_levels as u32);
        baroclinic.reference(t, &s.eta, &s.rho)?;
        let none = DeviceBuffer::<f64>::zeroed(st, 0)?;
        let fields = baroclinic.fields.as_ref();
        let stratification = fields.map_or(&none, |f| &f.stratification);
        // SAFETY: every buffer is sized for this mesh and these levels
        // (`Statics3D::upload`, `Self::new`, the state); no output is an
        // input.
        unsafe {
            m.bound_rates(
                st,
                launch(ne, COLUMN_BLOCK),
                ne as u32,
                nl,
                cfl,
                (physics.ops.order as f64 + 1.0).powi(2),
                physics.g,
                physics.rho0,
                physics.g / physics.rho0,
                physics.min_column_depth,
                &s.eta,
                &s.u,
                &s.v,
                w,
                &s.rho,
                &self.inverse_lengths,
                &t.bed,
                &t.mass,
                &t.averaged,
                &t.contra,
                &t.jinv,
                &t.dr,
                &t.ds,
                &t.w,
                &t.sub_metric,
                &t.sigma_rho,
                &t.d_sigma,
                stratification,
                &mut self.advection,
                &mut self.vertical,
                &mut self.sloping,
            )?;
        }
        Ok(())
    }

    /// Every element's step on its sloping levels into `self.steps`
    /// (`sloping_steps` of the bounds [`Self::rates`] left, then the
    /// measured frequencies of the elements that need one), each against
    /// the step `needed` of the other bounds, or (`each`) against its own in
    /// `self.needed`; how many frequencies were measured.
    fn sloping<BC: SWEBoundaryCondition2D>(
        &mut self,
        statics: &Statics3D,
        physics: &Hydrostatic3D<LinearEOS, GlsMixing, BC>,
        fields: &ReferenceFieldsDevice,
        s: &State3D,
        scheme: SspScheme,
        needed: f64,
        each: bool,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        let gpu = self.gpu.clone();
        let (m, st) = (&gpu.module_bound, &gpu.stream);
        let t = statics;
        let (ne, nl) = (t.n_elements, t.n_levels as u32);
        let buoyancy = physics.g / physics.rho0;
        let mcd = physics.min_column_depth;
        let limit = sloping_level_limit(scheme);
        let none = DeviceBuffer::<f64>::zeroed(st, 0)?;
        let needed_each = if each { &self.needed } else { &none };
        self.count.copy_from_host(st, &[0])?;
        // SAFETY: as above; `count` is a u32 the atomics alone touch, and
        // `scratch` holds `FREQUENCY_THREADS` threads' (`Self::new`).
        unsafe {
            m.sloping_steps(
                st,
                launch(ne, COLUMN_BLOCK),
                ne as u32,
                limit,
                needed,
                needed_each,
                mcd,
                &s.eta,
                &t.bed,
                &self.sloping,
                self.count.cu_deviceptr() as *mut u32,
                &mut self.candidates,
                &mut self.steps,
            )?;
            m.sloping_frequencies(
                st,
                launch(FREQUENCY_THREADS, COLUMN_BLOCK),
                FREQUENCY_THREADS as u32,
                nl,
                limit,
                buoyancy,
                mcd,
                &self.count,
                &self.candidates,
                &s.eta,
                &t.bed,
                &t.mass,
                &t.averaged,
                &t.contra,
                &t.jinv,
                &t.dr,
                &t.ds,
                &t.w,
                &t.sub_metric,
                &t.sigma_rho,
                &t.d_sigma,
                &fields.density,
                &fields.stratification,
                &fields.residual,
                self.scratch.cu_deviceptr() as *mut f64,
                &mut self.steps,
            )?;
        }
        Ok(self.count.to_host_vec(st)?[0] as usize)
    }

    /// The viscosity's limit of `Hydrostatic3D::time_step_limits`: the
    /// background's from the CPU's own functions (once per `cfl`), or with
    /// Smagorinsky every element's largest viscosity from the device and
    /// the step from it on the host (`element_dt_viscous_swe_2d`'s
    /// arithmetic, its metric static).
    fn viscosity_limit<BC: SWEBoundaryCondition2D>(
        &mut self,
        statics: &Statics3D,
        physics: &Hydrostatic3D<LinearEOS, GlsMixing, BC>,
        viscosity: &mut Viscosity3D,
        s: &State3D,
        cfl: f64,
    ) -> Result<TimeStepLimit, Box<dyn std::error::Error>> {
        let (mesh, order) = (&physics.mesh, physics.ops.order);
        let ne = statics.n_elements;
        let neighbours_largest = |largest: &[f64], k: usize| {
            (0..4)
                .filter_map(|face| mesh.neighbor(ElementIndex::new(k), face))
                .map(|nb| largest[nb.element])
                .fold(largest[k], f64::max)
        };
        Ok(match &mut self.viscosity {
            ViscosityBound::Zero => TimeStepLimit::NONE,
            ViscosityBound::Background(Some((at, limit))) if *at == cfl => *limit,
            ViscosityBound::Background(cached) => {
                let background = physics.horizontal_viscosity.background;
                let largest = vec![background; ne];
                let limit = (0..ne)
                    .map(|k| TimeStepLimit {
                        dt: viscous_dt(
                            neighbours_largest(&largest, k),
                            viscous_metric(mesh, &physics.ops, &physics.geom, k),
                            order,
                            cfl,
                        ),
                        element: k,
                    })
                    .fold(TimeStepLimit::NONE, TimeStepLimit::min);
                *cached = Some((cfl, limit));
                limit
            }
            ViscosityBound::Smagorinsky {
                metric,
                levels,
                largest,
                host,
            } => {
                viscosity.largest(statics, &s.eta, &s.u, &s.v, levels, largest)?;
                largest.copy_to_host(&self.gpu.stream, host)?;
                (0..ne)
                    .map(|k| TimeStepLimit {
                        dt: viscous_dt(neighbours_largest(host, k), metric[k], order, cfl),
                        element: k,
                    })
                    .fold(TimeStepLimit::NONE, TimeStepLimit::min)
            }
        })
    }
}

/// The smallest of every element's `dts`, with its element (the CPU's
/// `TimeStepLimit::min`: exact in any order).
fn smallest(dts: &[f64]) -> TimeStepLimit {
    dts.iter()
        .enumerate()
        .map(|(element, &dt)| TimeStepLimit { dt, element })
        .fold(TimeStepLimit::NONE, TimeStepLimit::min)
}

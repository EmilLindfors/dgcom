//! The 3D step's slow forcing on the GPU (P2.7 Phase 4.4c1, `notes/gpu.md`):
//! `Hydrostatic3D::slow_forcing_into`, the `G` the barotropic pass adds to
//! the 2D module's rate so that the depth mean feels what the 3D model's
//! columns do,
//!
//! ```text
//!     G = D⟨R⟩ + Σ_l A_l(u) − A(ū) − D R_Cor(ū) + (τ_s − τ_b)/ρ₀ − r (u_b − ū) − D Σ_l Δσ_l λ_l (u_l − ū),
//! ```
//!
//! zero in the thin columns:
//!
//! - `⟨R⟩` the depth mean of the momentum RHS (`R` the caller's: the
//!   pressure gradient, Coriolis, the viscosity, the nesting band's
//!   relaxation);
//! - `Σ_l A_l(u)` the layers' advection by their own transports, the thin
//!   columns at rest ([`kernels_forcing::mask_flow`], then
//!   [`Transport3D::compute`] without a barotropic flux and
//!   [`Transport3D::advect`]);
//! - `A(ū) + D R_Cor(ū)` the same of the depth mean as one layer (a
//!   one-level [`Transport3D`], [`kernels_forcing::depth_mean_coriolis`]);
//! - the surface and bottom stresses (the uniform ones and a stress field,
//!   uploaded), the shear parts of the bottom and the cage drags (whose
//!   depth-mean parts the pass applies), from the step's drag rates
//!   ([`crate::vertical::Vertical3D::drag_rates`]).
//!
//! Each kernel follows `slow_forcing_into` operation for operation, so a
//! build without multiply–add contraction gives the CPU's bits
//! (`forcing3d.rs`, the gate).

// As `three_d.rs`: the device loops index arrays by the loop variable.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use std::sync::Arc;

use cuda_core::DeviceBuffer;
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;
use dgcom::mesh::Mesh2D;
use dgcom::operators::DGOperators2D;
use dgcom::physics::BottomDrag3D;
use dgcom::solver::rhs::{UNBOUNDED, VerticalAdvection};
use dgcom::source::CoriolisSource2D;
use dgcom::types::ElementIndex;
use dgcom::vertical::depth_average;

use crate::hooks::State3D;
use crate::swe2d::{Gpu, NN};
use crate::three_d::{Advected, COLUMN_BLOCK, Interface, Statics3D, Transport3D, launch};
use crate::vertical::Vertical3D;
use dgcom::solver::rhs::DeviceTracerBC3D;

// =============================================================================
// Kernels
// =============================================================================

#[cuda_module]
pub mod kernels_forcing {
    use super::*;

    /// `Hydrostatic3D::with_thin_columns_at_rest` for the slow forcing, a
    /// thread per column: the layers' `u`, `v` into `mu`, `mv`, and the depth
    /// mean `ū`, `v̄` into `bu`, `bv`, all zero in a column shallower than
    /// `thin_depth`.
    #[kernel]
    pub fn mask_flow(
        nl: u32,
        n: u32,
        thin_depth: f64,
        eta: &[f64],
        bed: &[f64],
        u: &[f64],
        v: &[f64],
        ubar: &[f64],
        vbar: &[f64],
        mut mu: DisjointSlice<f64>,
        mut mv: DisjointSlice<f64>,
        mut bu: DisjointSlice<f64>,
        mut bv: DisjointSlice<f64>,
    ) {
        let idx = thread::index_1d().get();
        if idx >= n as usize {
            return;
        }
        let nl = nl as usize;
        let thin = eta[idx] - bed[idx] < thin_depth;
        // SAFETY: thread idx alone writes column idx.
        unsafe {
            for l in 0..nl {
                let j = idx * nl + l;
                *mu.get_unchecked_mut(j) = if thin { 0.0 } else { u[j] };
                *mv.get_unchecked_mut(j) = if thin { 0.0 } else { v[j] };
            }
            *bu.get_unchecked_mut(idx) = if thin { 0.0 } else { ubar[idx] };
            *bv.get_unchecked_mut(idx) = if thin { 0.0 } else { vbar[idx] };
        }
    }

    /// The depth mean's Coriolis tendency as `slow_forcing_into` forms it, a
    /// thread per column: `apply_coriolis_3d` of the one-level state `(bu,
    /// bv)` from zero (`f` every node's), times the column's depth.
    #[kernel]
    pub fn depth_mean_coriolis(
        n: u32,
        f: &[f64],
        eta: &[f64],
        bed: &[f64],
        bu: &[f64],
        bv: &[f64],
        mut ru: DisjointSlice<f64>,
        mut rv: DisjointSlice<f64>,
    ) {
        let idx = thread::index_1d().get();
        if idx >= n as usize {
            return;
        }
        let (mut du, mut dv) = (0.0, 0.0);
        du += f[idx] * bv[idx];
        dv -= f[idx] * bu[idx];
        let depth = eta[idx] - bed[idx];
        // SAFETY: thread idx alone writes column idx.
        unsafe {
            *ru.get_unchecked_mut(idx) = du * depth;
            *rv.get_unchecked_mut(idx) = dv * depth;
        }
    }

    /// `slow_forcing_into`'s `G` of every column, a thread per column: zero
    /// in a thin column; else `D⟨R⟩ + Σ_l A_l − (A(ū) + D R_Cor(ū)) +
    /// (τ_s + field − τ_b)/ρ₀ − r (u_b − ū)` (`uniform`: `[τ_sx, τ_sy, τ_bx,
    /// τ_by]`; the field and the drag empty for none; the drag `kind` 0 a
    /// constant `C_d = law[0]`, 1 the log layer's `[z0, cd_min, cd_max]`),
    /// less the cages' shear part `D Σ_l Δσ_l λ_l (u_l − ū)` in the caged
    /// columns (`caged`, empty for none; `λ` the step's `layer_rate`).
    #[kernel]
    pub fn slow_forcing(
        nl: u32,
        n: u32,
        thin_depth: f64,
        rho0: f64,
        uniform: &[f64],
        kind: u32,
        law: &[f64],
        sigma_bottom: f64,
        eta: &[f64],
        bed: &[f64],
        d_sigma: &[f64],
        u: &[f64],
        v: &[f64],
        ubar: &[f64],
        vbar: &[f64],
        rhs_u: &[f64],
        rhs_v: &[f64],
        advection_u: &[f64],
        advection_v: &[f64],
        bar_u: &[f64],
        bar_v: &[f64],
        stress_x: &[f64],
        stress_y: &[f64],
        caged: &[u32],
        layer_rate: &[f64],
        mut g_h: DisjointSlice<f64>,
        mut g_hu: DisjointSlice<f64>,
        mut g_hv: DisjointSlice<f64>,
    ) {
        let idx = thread::index_1d().get();
        if idx >= n as usize {
            return;
        }
        let nl = nl as usize;
        let column = idx * nl..(idx + 1) * nl;
        let depth = eta[idx] - bed[idx];
        let (mut gu, mut gv) = (0.0, 0.0);
        if depth >= thin_depth {
            let mean_u = depth_average(&rhs_u[column.clone()], &d_sigma[..nl]);
            let mean_v = depth_average(&rhs_v[column.clone()], &d_sigma[..nl]);
            let advection_x: f64 = advection_u[column.clone()].iter().sum();
            let advection_y: f64 = advection_v[column.clone()].iter().sum();
            let (field_x, field_y) = if stress_x.is_empty() {
                (0.0, 0.0)
            } else {
                (stress_x[idx], stress_y[idx])
            };
            let sx = (uniform[0] + field_x - uniform[2]) / rho0;
            let sy = (uniform[1] + field_y - uniform[3]) / rho0;
            let (drag_x, drag_y) = if law.is_empty() {
                (0.0, 0.0)
            } else {
                let drag = if kind == 0 {
                    BottomDrag3D::Quadratic { cd: law[0] }
                } else {
                    BottomDrag3D::LogLayer {
                        z0: law[0],
                        cd_min: law[1],
                        cd_max: law[2],
                    }
                };
                let bottom = idx * nl;
                let r = drag.column_rate(
                    depth,
                    thin_depth,
                    sigma_bottom,
                    [u[bottom], v[bottom]],
                    [ubar[idx], vbar[idx]],
                );
                (r * (u[bottom] - ubar[idx]), r * (v[bottom] - vbar[idx]))
            };
            gu = depth * mean_u + advection_x - bar_u[idx] + sx - drag_x;
            gv = depth * mean_v + advection_y - bar_v[idx] + sy - drag_y;
            if !caged.is_empty() && caged[idx] != 0 {
                let (ub, vb) = (ubar[idx], vbar[idx]);
                let (mut cx, mut cy) = (0.0, 0.0);
                for l in 0..nl {
                    let j = idx * nl + l;
                    let rate = layer_rate[j];
                    cx += d_sigma[l] * rate * (u[j] - ub);
                    cy += d_sigma[l] * rate * (v[j] - vb);
                }
                gu -= depth * cx;
                gv -= depth * cy;
            }
        }
        // SAFETY: thread idx alone writes column idx.
        unsafe {
            *g_h.get_unchecked_mut(idx) = 0.0;
            *g_hu.get_unchecked_mut(idx) = gu;
            *g_hv.get_unchecked_mut(idx) = gv;
        }
    }
}

// =============================================================================
// Host side
// =============================================================================

/// What [`SlowForcing3D::new`] takes of a 3D physics module.
pub struct ForcingSetup<'a> {
    pub coriolis: &'a CoriolisSource2D,
    /// `[τ_sx, τ_sy, τ_bx, τ_by]` (N/m²): the uniform surface and bottom
    /// stresses (`Forcing`)
    pub stress: [f64; 4],
    pub bottom_drag: Option<BottomDrag3D>,
    pub rho0: f64,
    pub min_column_depth: f64,
    /// The momentum's vertical reconstruction
    /// (`Hydrostatic3D::momentum_vertical_advection`)
    pub vertical: VerticalAdvection,
}

/// The slow forcing `G` of a state on the device.
pub struct SlowForcing3D {
    gpu: Arc<Gpu>,
    /// The layers' own transports and the depth mean's (one level)
    pub layers: Transport3D,
    pub bar: Transport3D,
    vertical: VerticalAdvection,
    rho0: f64,
    min_column_depth: f64,
    sigma_bottom: f64,
    uniform: DeviceBuffer<f64>,
    drag: Option<(u32, DeviceBuffer<f64>)>,
    /// `f` at every node
    coriolis: DeviceBuffer<f64>,
    /// The state's flow with its thin columns at rest
    masked_u: DeviceBuffer<f64>,
    masked_v: DeviceBuffer<f64>,
    bar_u: DeviceBuffer<f64>,
    bar_v: DeviceBuffer<f64>,
    /// `Σ`'s terms: the layers' advection, `A(ū) + D R_Cor(ū)`
    pub advection_u: DeviceBuffer<f64>,
    pub advection_v: DeviceBuffer<f64>,
    pub bar_rhs_u: DeviceBuffer<f64>,
    pub bar_rhs_v: DeviceBuffer<f64>,
    /// `G`: `[h, hu, hv]` per node
    pub g: [DeviceBuffer<f64>; 3],
    none: DeviceBuffer<f64>,
    none_u32: DeviceBuffer<u32>,
}

impl SlowForcing3D {
    /// The slow forcing of `setup` on `statics`' mesh (the layers' tables)
    /// and `bar_statics` (the same mesh under one level).
    pub fn new(
        gpu: &Arc<Gpu>,
        statics: Statics3D,
        bar_statics: Statics3D,
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        sigma_bottom: f64,
        setup: ForcingSetup,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let st = &gpu.stream;
        let (ne, nl) = (statics.n_elements, statics.n_levels);
        let n = ne * NN;
        let zeros = |n: usize| DeviceBuffer::<f64>::zeroed(st, n);
        let drag = match setup.bottom_drag {
            None => None,
            Some(BottomDrag3D::Quadratic { cd }) => {
                Some((0, DeviceBuffer::from_host(st, &[cd, 0.0, 0.0])?))
            }
            Some(BottomDrag3D::LogLayer { z0, cd_min, cd_max }) => {
                Some((1, DeviceBuffer::from_host(st, &[z0, cd_min, cd_max])?))
            }
        };
        // f at every node, as `apply_coriolis_3d` takes it
        let mut f = Vec::with_capacity(n);
        for k in 0..ne {
            let ki = ElementIndex::new(k);
            for i in 0..NN {
                let [_x, y] = mesh.reference_to_physical(ki, ops.nodes_r[i], ops.nodes_s[i]);
                f.push(setup.coriolis.f_at(y));
            }
        }
        let thin = setup.min_column_depth;
        Ok(Self {
            gpu: gpu.clone(),
            layers: Transport3D::new(gpu, statics, thin, None)?,
            bar: Transport3D::new(gpu, bar_statics, thin, None)?,
            vertical: setup.vertical,
            rho0: setup.rho0,
            min_column_depth: thin,
            sigma_bottom,
            uniform: DeviceBuffer::from_host(st, &setup.stress)?,
            drag,
            coriolis: DeviceBuffer::from_host(st, &f)?,
            masked_u: zeros(n * nl)?,
            masked_v: zeros(n * nl)?,
            bar_u: zeros(n)?,
            bar_v: zeros(n)?,
            advection_u: zeros(n * nl)?,
            advection_v: zeros(n * nl)?,
            bar_rhs_u: zeros(n)?,
            bar_rhs_v: zeros(n)?,
            g: [zeros(n)?, zeros(n)?, zeros(n)?],
            none: zeros(0)?,
            none_u32: DeviceBuffer::zeroed(st, 0)?,
        })
    }

    /// `slow_forcing_into` of `state` with the momentum RHS `(rhs_u, rhs_v)`
    /// of it, the surface stress field `stress` (`[τ_x, τ_y]` per node at
    /// the step's time, if any) and the step's drag rates of `vertical`
    /// (its cages' if `caged`, the columns under a cage): into [`Self::g`].
    pub fn compute(
        &mut self,
        state: &State3D,
        rhs: [&DeviceBuffer<f64>; 2],
        stress: Option<[&DeviceBuffer<f64>; 2]>,
        cages: Option<(&Vertical3D, &DeviceBuffer<u32>)>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let gpu = self.gpu.clone();
        let (m, st) = (&gpu.module_forcing, &gpu.stream);
        let s = self.layers.statics();
        let (nl, n) = (s.n_levels, s.n_columns());
        // SAFETY: every buffer is sized for this mesh and these levels; no
        // output is an input.
        unsafe {
            m.mask_flow(
                st,
                launch(n, COLUMN_BLOCK),
                nl as u32,
                n as u32,
                self.min_column_depth,
                &state.eta,
                &s.bed,
                &state.u,
                &state.v,
                &state.ubar,
                &state.vbar,
                &mut self.masked_u,
                &mut self.masked_v,
                &mut self.bar_u,
                &mut self.bar_v,
            )?;
        }
        // Σ_l A_l(u): the layers' advection by their own transports, from
        // zero
        self.layers
            .compute(&state.eta, &self.masked_u, &self.masked_v, None, None)?;
        self.advection_u.zero_async(st)?;
        self.advection_v.zero_async(st)?;
        let velocity = |field| Advected {
            field,
            reference: None,
            range: UNBOUNDED,
            exterior: None,
            bc: DeviceTracerBC3D::Extrapolation,
        };
        self.layers.advect(
            [&velocity(&self.masked_u), &velocity(&self.masked_v)],
            [&mut self.advection_u, &mut self.advection_v],
            self.vertical,
            Interface::Central,
            false,
            false,
        )?;
        // A(ū) + D R_Cor(ū): the depth mean as one layer
        self.bar
            .compute(&state.eta, &self.bar_u, &self.bar_v, None, None)?;
        let s = self.layers.statics();
        // SAFETY: as above.
        unsafe {
            m.depth_mean_coriolis(
                st,
                launch(n, COLUMN_BLOCK),
                n as u32,
                &self.coriolis,
                &state.eta,
                &s.bed,
                &self.bar_u,
                &self.bar_v,
                &mut self.bar_rhs_u,
                &mut self.bar_rhs_v,
            )?;
        }
        self.bar.advect(
            [&velocity(&self.bar_u), &velocity(&self.bar_v)],
            [&mut self.bar_rhs_u, &mut self.bar_rhs_v],
            self.vertical,
            Interface::Central,
            false,
            false,
        )?;
        let none = &self.none;
        let [stress_x, stress_y] = stress.unwrap_or([none, none]);
        let (kind, law) = match &self.drag {
            Some((kind, law)) => (*kind, law),
            None => (0, none),
        };
        let (caged, layer_rate) = match cages {
            Some((vertical, caged)) => (caged, &vertical.layer_rate),
            None => (&self.none_u32, none),
        };
        let [g_h, g_hu, g_hv] = &mut self.g;
        // SAFETY: as above.
        unsafe {
            m.slow_forcing(
                st,
                launch(n, COLUMN_BLOCK),
                nl as u32,
                n as u32,
                self.min_column_depth,
                self.rho0,
                &self.uniform,
                kind,
                law,
                self.sigma_bottom,
                &state.eta,
                &s.bed,
                &s.d_sigma,
                &state.u,
                &state.v,
                &state.ubar,
                &state.vbar,
                rhs[0],
                rhs[1],
                &self.advection_u,
                &self.advection_v,
                &self.bar_rhs_u,
                &self.bar_rhs_v,
                stress_x,
                stress_y,
                caged,
                layer_rate,
                g_h,
                g_hu,
                g_hv,
            )?;
        }
        Ok(())
    }
}

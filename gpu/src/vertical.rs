//! The 3D step's vertical terms on the GPU (P2.7 Phase 4.4a, `notes/gpu.md`):
//! `Hydrostatic3D::vertical_implicit`, which the mode splitter calls once a
//! step after the stages, and the drag rates it takes, frozen at the step's
//! start (`bottom_drag_into`, `layer_drag_into`):
//!
//! - **The drag rates**: a column's bottom drag `C_d(z_b)|u_b|`
//!   (`BottomDrag3D::column_rate`, the log layer's or a constant `C_d`), a
//!   thread per column; the cages' `λ_l = c_l |u_l|` on every layer of a
//!   caged column (`cage_rates`), a thread per caged node.
//! - **The vertical step** of every column, a thread per column
//!   (`vertical_columns`): the GLS closure's `k` and `ψ` over the step
//!   (`GlsMixing::step_column`), then the implicit diffusion of `u`, `v`
//!   with the surface stress and the drags and of `T`, `S` with the surface
//!   fluxes and the shortwave absorbed through the column (`diffuse_column`);
//!   a thin column takes its fluxes with the bed under it
//!   (`thin_column_fluxes`) and the depth mean in every layer
//!   (`thin_columns_to_depth_mean`).
//!
//! The kernels call those dgcom functions themselves: their powers,
//! logarithms, exponentials and `hypot` are `dgcom::math`'s, plain
//! arithmetic, so a build without multiply–add contraction (`--no-fmad`)
//! gives the CPU's bits (`vertical3d.rs`, the gate). The surface fields
//! (the wind stress, the roughness, the heat and fresh-water fluxes) are the
//! host's, uploaded per step ([`SurfaceDevice`]).

// As `three_d.rs`: the device loops index arrays by the loop variable.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use std::sync::Arc;

use cuda_core::{DeviceBuffer, DeviceCopy};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;
use dgcom::physics::cage_drag::cage_rates;
use dgcom::physics::vertical_diffusion::{
    ColumnBuffers, ColumnForcing, diffuse_column, thin_column_fluxes,
};
use dgcom::physics::vertical_mixing::Forcing;
use dgcom::physics::{BottomDrag3D, GlsMixing, JerlovWater};
use dgcom::solver::rhs::transport_3d::DRY_DEPTH;
use dgcom::source::CageDrag2D;
use dgcom::vertical::SigmaGrid;

use crate::hooks::State3D;
use crate::swe2d::Gpu;
use crate::three_d::{COLUMN_BLOCK, MAX_LEVELS, Statics3D, launch};

/// A GLS closure on the device: dgcom's own, copied as it is.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct GlsDevice(pub GlsMixing);

// SAFETY: `GlsMixing` is `Copy`, of f64s, their arrays and options; the
// device reads it as the same Rust type.
unsafe impl DeviceCopy for GlsDevice {}

/// The GLS closure's work space for [`MAX_LEVELS`].
const WORK: usize = GlsMixing::work_len(MAX_LEVELS);

// =============================================================================
// Kernels
// =============================================================================

#[cuda_module]
pub mod kernels_vertical {
    use super::*;

    /// `dgcom::math`'s `pow(x, y)`, `exp(y)`, `ln(x)`, `hypot(x, y)` of
    /// every pair, four outputs each (the gate's check of the functions
    /// themselves).
    #[kernel]
    pub fn elementary(n: u32, x: &[f64], y: &[f64], mut out: DisjointSlice<f64>) {
        let t = thread::index_1d().get();
        if t >= n as usize {
            return;
        }
        let (a, b) = (x[t], y[t]);
        let values = [
            dgcom::math::pow(a, b),
            dgcom::math::exp(b),
            dgcom::math::ln(a),
            dgcom::math::hypot(a, b),
        ];
        // SAFETY: thread t alone writes its four outputs.
        unsafe {
            for (j, value) in values.into_iter().enumerate() {
                *out.get_unchecked_mut(t * 4 + j) = value;
            }
        }
    }

    /// `Hydrostatic3D::bottom_drag_into`, a thread per column: the drag
    /// (`kind` 0: `Quadratic { cd: law[0] }`, 1: `LogLayer { z0, cd_min,
    /// cd_max }` from `law`) of the bottom layer's velocity, or the depth
    /// mean's in a thin column, at the bottom layer's centre (`σ =
    /// sigma_bottom`).
    #[kernel]
    pub fn bottom_rates(
        nl: u32,
        n: u32,
        kind: u32,
        law: &[f64],
        thin_depth: f64,
        sigma_bottom: f64,
        eta: &[f64],
        bed: &[f64],
        u: &[f64],
        v: &[f64],
        ubar: &[f64],
        vbar: &[f64],
        mut rate: DisjointSlice<f64>,
    ) {
        let col = thread::index_1d().get();
        if col >= n as usize {
            return;
        }
        let drag = if kind == 0 {
            BottomDrag3D::Quadratic { cd: law[0] }
        } else {
            BottomDrag3D::LogLayer {
                z0: law[0],
                cd_min: law[1],
                cd_max: law[2],
            }
        };
        let bottom = col * nl as usize;
        // SAFETY: thread col alone writes column col's rate.
        unsafe {
            *rate.get_unchecked_mut(col) = drag.column_rate(
                eta[col] - bed[col],
                thin_depth,
                sigma_bottom,
                [u[bottom], v[bottom]],
                [ubar[col], vbar[col]],
            );
        }
    }

    /// `Hydrostatic3D::layer_drag_into` at the caged nodes, a thread per
    /// caged node (`nodes`, its entries `offsets[j]..offsets[j + 1]` of
    /// `coefficients` and `net_depths`): every layer's `λ_l` by dgcom's
    /// `cage_rates`. The other columns' rates stay zero.
    #[kernel]
    pub fn layer_rates(
        nl: u32,
        n: u32,
        nodes: &[u32],
        offsets: &[u32],
        coefficients: &[f64],
        net_depths: &[f64],
        thin_depth: f64,
        sigma_w: &[f64],
        d_sigma: &[f64],
        eta: &[f64],
        bed: &[f64],
        u: &[f64],
        v: &[f64],
        ubar: &[f64],
        vbar: &[f64],
        mut rate: DisjointSlice<f64>,
    ) {
        let j = thread::index_1d().get();
        if j >= n as usize {
            return;
        }
        let nl = nl as usize;
        let col = nodes[j] as usize;
        let column = col * nl..(col + 1) * nl;
        let first = offsets[j] as usize;
        let n_entries = offsets[j + 1] as usize - first;
        // SAFETY: the caged nodes are distinct: thread j alone writes
        // column col's rates.
        unsafe {
            let rate = core::slice::from_raw_parts_mut(rate.as_mut_ptr().add(col * nl), nl);
            cage_rates(
                n_entries,
                |e| (coefficients[first + e], net_depths[first + e]),
                [&sigma_w[..nl + 1], &d_sigma[..nl]],
                eta[col] - bed[col],
                thin_depth,
                [&u[column.clone()], &v[column]],
                [ubar[col], vbar[col]],
                rate,
            );
        }
    }

    /// `Hydrostatic3D::vertical_implicit` of every column, a thread per
    /// column: a column of water by dgcom's `diffuse_column` with the GLS
    /// closure's `step_column` (`gls[0]`), the uniform forcing `uniform`
    /// (`[τ_sx, τ_sy, τ_bx, τ_by, Q]`) plus the column's stress, roughness
    /// and fluxes (each empty for none; the water's type `water`, an index
    /// of `JerlovWater::ALL`), the drag rates (empty for none); a thin
    /// column its fluxes with `thin_bed` m of water under it (if
    /// `thin_bed_on`), then the depth mean in its layers. The viscosity and
    /// diffusivity of the columns of water into `av`, `kt`.
    #[kernel]
    pub fn vertical_columns(
        nl: u32,
        n: u32,
        gls: &[GlsDevice],
        uniform: &[f64],
        stress_x: &[f64],
        stress_y: &[f64],
        roughness: &[f64],
        nonsolar: &[f64],
        shortwave: &[f64],
        freshwater: &[f64],
        water: u32,
        thin_bed_on: u32,
        thin_bed: f64,
        bottom_rate: &[f64],
        layer_rate: &[f64],
        g: f64,
        rho0: f64,
        dt: f64,
        min_column_depth: f64,
        eta: &[f64],
        bed: &[f64],
        ubar: &[f64],
        vbar: &[f64],
        sigma_rho: &[f64],
        sigma_w: &[f64],
        d_sigma: &[f64],
        rho: &[f64],
        mut u: DisjointSlice<f64>,
        mut v: DisjointSlice<f64>,
        mut temp: DisjointSlice<f64>,
        mut salt: DisjointSlice<f64>,
        mut tke: DisjointSlice<f64>,
        mut psi: DisjointSlice<f64>,
        mut av_out: DisjointSlice<f64>,
        mut kt_out: DisjointSlice<f64>,
    ) {
        let idx = thread::index_1d().get();
        if idx >= n as usize {
            return;
        }
        let (nl, nw) = (nl as usize, nl as usize + 1);
        let h = -bed[idx];
        let fluxes = if nonsolar.is_empty() {
            None
        } else {
            Some([nonsolar[idx], shortwave[idx], freshwater[idx]])
        };
        // SAFETY: thread idx alone touches column idx of every field.
        unsafe {
            let column =
                |p: *mut f64, len: usize| core::slice::from_raw_parts_mut(p.add(idx * len), len);
            let (u, v) = (column(u.as_mut_ptr(), nl), column(v.as_mut_ptr(), nl));
            let (temp, salt) = (column(temp.as_mut_ptr(), nl), column(salt.as_mut_ptr(), nl));
            if eta[idx] + h < min_column_depth {
                // A thin column: its fluxes over the whole of it and the
                // bed, if asked for; then the depth mean in every layer
                let depth = eta[idx] + h;
                if thin_bed_on != 0 && depth >= DRY_DEPTH {
                    thin_column_fluxes([temp, salt], depth, thin_bed, uniform[4], fluxes, rho0, dt);
                }
                u.fill(ubar[idx]);
                v.fill(vbar[idx]);
                return;
            }
            let (mut tau_sx, mut tau_sy) = (uniform[0], uniform[1]);
            if !stress_x.is_empty() {
                tau_sx += stress_x[idx];
                tau_sy += stress_y[idx];
            }
            let (tke, psi) = (column(tke.as_mut_ptr(), nw), column(psi.as_mut_ptr(), nw));
            let mut a = [0.0; MAX_LEVELS];
            let mut b = [0.0; MAX_LEVELS];
            let mut c = [0.0; MAX_LEVELS];
            let mut d = [0.0; MAX_LEVELS];
            let mut x = [0.0; MAX_LEVELS];
            let mut c_prime = [0.0; MAX_LEVELS];
            let mut d_prime = [0.0; MAX_LEVELS];
            let mut z_r = [0.0; MAX_LEVELS];
            let mut z_w = [0.0; MAX_LEVELS + 1];
            let mut dz = [0.0; MAX_LEVELS];
            let mut av = [0.0; MAX_LEVELS + 1];
            let mut kt = [0.0; MAX_LEVELS + 1];
            let mut work = [0.0; WORK];
            let closure = gls[0].0;
            diffuse_column(
                [&sigma_rho[..nl], &sigma_w[..nw], &d_sigma[..nl]],
                eta[idx],
                h,
                [u, v, temp, salt],
                &rho[idx * nl..(idx + 1) * nl],
                &ColumnForcing {
                    forcing: Forcing {
                        surface_stress: [tau_sx, tau_sy],
                        bottom_stress: [uniform[2], uniform[3]],
                        surface_heat_flux: uniform[4],
                    },
                    roughness: if roughness.is_empty() {
                        None
                    } else {
                        Some(roughness[idx])
                    },
                    tracers: fluxes.map(|f| (f, JerlovWater::ALL[water as usize])),
                    bottom_drag: if bottom_rate.is_empty() {
                        0.0
                    } else {
                        bottom_rate[idx]
                    },
                },
                if layer_rate.is_empty() {
                    None
                } else {
                    Some(&layer_rate[idx * nl..(idx + 1) * nl])
                },
                [g, rho0, dt],
                ColumnBuffers {
                    a: &mut a[..nl],
                    b: &mut b[..nl],
                    c: &mut c[..nl],
                    d: &mut d[..nl],
                    x: &mut x[..nl],
                    c_prime: &mut c_prime[..nl],
                    d_prime: &mut d_prime[..nl],
                    z_r: &mut z_r[..nl],
                    z_w: &mut z_w[..nw],
                    dz: &mut dz[..nl],
                    av: &mut av[..nw],
                    kt: &mut kt[..nw],
                },
                |column, _forcing, av, kt| {
                    closure.step_column(column, dt, tke, psi, &mut work, av, kt);
                },
            );
            let (av_col, kt_col) = (
                column(av_out.as_mut_ptr(), nw),
                column(kt_out.as_mut_ptr(), nw),
            );
            av_col.copy_from_slice(&av[..nw]);
            kt_col.copy_from_slice(&kt[..nw]);
        }
    }
}

// =============================================================================
// Host side
// =============================================================================

/// The cages' entries by caged node, as [`kernels_vertical::layer_rates`]
/// reads them.
struct CagesDevice {
    n_nodes: usize,
    nodes: DeviceBuffer<u32>,
    offsets: DeviceBuffer<u32>,
    coefficients: DeviceBuffer<f64>,
    net_depths: DeviceBuffer<f64>,
}

/// What [`Vertical3D::new`] takes of a 3D physics module.
pub struct VerticalSetup<'a> {
    pub mixing: &'a GlsMixing,
    /// The uniform surface and bottom stresses and heat flux
    pub forcing: Forcing,
    pub bottom_drag: Option<BottomDrag3D>,
    pub cages: Option<&'a CageDrag2D>,
    /// The optical water type of the shortwave absorption
    pub water: JerlovWater,
    /// The thin columns' bed (`SurfaceFields::thin_bed`)
    pub thin_bed: Option<f64>,
    pub g: f64,
    pub rho0: f64,
    pub min_column_depth: f64,
}

/// The surface fields of a step on the device, each per column
/// (`[element][node]`), each optional (`SurfaceFields`).
#[derive(Clone, Copy, Default)]
pub struct SurfaceDevice<'a> {
    /// `[τ_x, τ_y]` (N/m²), added to the uniform stress
    pub stress: Option<[&'a DeviceBuffer<f64>; 2]>,
    /// The surface roughness (m)
    pub roughness: Option<&'a DeviceBuffer<f64>>,
    /// `[non-solar, shortwave, fresh water]` (`SurfaceTracerFluxes`)
    pub tracers: Option<[&'a DeviceBuffer<f64>; 3]>,
}

/// `dgcom::math`'s `[pow(x, y), exp(y), ln(x), hypot(x, y)]` of every pair
/// on the device.
pub fn elementary(
    gpu: &Gpu,
    x: &[f64],
    y: &[f64],
) -> Result<Vec<[f64; 4]>, Box<dyn std::error::Error>> {
    let st = &gpu.stream;
    let n = x.len();
    let (dx, dy) = (
        DeviceBuffer::from_host(st, x)?,
        DeviceBuffer::from_host(st, y)?,
    );
    let mut out = DeviceBuffer::<f64>::zeroed(st, 4 * n)?;
    // SAFETY: the buffers hold n pairs and 4n outputs.
    unsafe {
        gpu.module_vertical.elementary(
            st,
            launch(n, COLUMN_BLOCK),
            n as u32,
            &dx,
            &dy,
            &mut out,
        )?;
    }
    let flat = out.to_host_vec(st)?;
    Ok(flat.as_chunks::<4>().0.to_vec())
}

/// `Hydrostatic3D::vertical_implicit` and its drag rates on the device.
pub struct Vertical3D {
    gpu: Arc<Gpu>,
    gls: DeviceBuffer<GlsDevice>,
    /// `[τ_sx, τ_sy, τ_bx, τ_by, Q]`
    uniform: DeviceBuffer<f64>,
    /// `(kind, [z0 or cd, cd_min, cd_max])` of the bottom drag
    drag: Option<(u32, DeviceBuffer<f64>)>,
    cages: Option<CagesDevice>,
    water: u32,
    thin_bed: Option<f64>,
    g: f64,
    rho0: f64,
    min_column_depth: f64,
    sigma_bottom: f64,
    /// The step's bottom-drag rate of every column (`StepDrag::bottom`)
    pub bottom_rate: DeviceBuffer<f64>,
    /// The step's cage-drag rate of every layer (`StepDrag::layers`), zero
    /// outside the cages
    pub layer_rate: DeviceBuffer<f64>,
    /// The mixing's viscosity and diffusivity at every w-point
    /// (`Solution3D::eddy_viscosity`, `eddy_diffusivity`)
    pub eddy_viscosity: DeviceBuffer<f64>,
    pub eddy_diffusivity: DeviceBuffer<f64>,
    none: DeviceBuffer<f64>,
}

impl Vertical3D {
    /// The vertical step of `setup` on `statics`' mesh and `sigma`'s levels;
    /// refuses more levels than the kernel's arrays hold.
    pub fn new(
        gpu: &Arc<Gpu>,
        statics: &Statics3D,
        sigma: &SigmaGrid,
        setup: VerticalSetup,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let st = &gpu.stream;
        let (n_columns, nl) = (statics.n_columns(), statics.n_levels);
        if nl > MAX_LEVELS {
            return Err(format!("{nl} σ-levels: the vertical kernel takes {MAX_LEVELS}").into());
        }
        let f = setup.forcing;
        let drag = match setup.bottom_drag {
            None => None,
            Some(BottomDrag3D::Quadratic { cd }) => {
                Some((0, DeviceBuffer::from_host(st, &[cd, 0.0, 0.0])?))
            }
            Some(BottomDrag3D::LogLayer { z0, cd_min, cd_max }) => {
                Some((1, DeviceBuffer::from_host(st, &[z0, cd_min, cd_max])?))
            }
        };
        let cages = match setup.cages {
            Some(cages) if !cages.is_empty() => {
                let (mut nodes, mut offsets) = (Vec::new(), vec![0u32]);
                let (mut coefficients, mut net_depths) = (Vec::new(), Vec::new());
                for (e, entry) in cages.nodes().iter().enumerate() {
                    if nodes.last() != Some(&(entry.node as u32)) {
                        if e > 0 {
                            offsets.push(e as u32);
                        }
                        nodes.push(entry.node as u32);
                    }
                    coefficients.push(entry.coefficient);
                    net_depths.push(entry.net_depth);
                }
                offsets.push(cages.nodes().len() as u32);
                Some(CagesDevice {
                    n_nodes: nodes.len(),
                    nodes: DeviceBuffer::from_host(st, &nodes)?,
                    offsets: DeviceBuffer::from_host(st, &offsets)?,
                    coefficients: DeviceBuffer::from_host(st, &coefficients)?,
                    net_depths: DeviceBuffer::from_host(st, &net_depths)?,
                })
            }
            _ => None,
        };
        let zeros = |n: usize| DeviceBuffer::<f64>::zeroed(st, n);
        let water = JerlovWater::ALL
            .iter()
            .position(|&w| w == setup.water)
            .expect("a Jerlov type") as u32;
        Ok(Self {
            gpu: gpu.clone(),
            gls: DeviceBuffer::from_host(st, &[GlsDevice(*setup.mixing)])?,
            uniform: DeviceBuffer::from_host(
                st,
                &[
                    f.surface_stress[0],
                    f.surface_stress[1],
                    f.bottom_stress[0],
                    f.bottom_stress[1],
                    f.surface_heat_flux,
                ],
            )?,
            bottom_rate: zeros(if drag.is_some() { n_columns } else { 0 })?,
            layer_rate: zeros(if cages.is_some() { n_columns * nl } else { 0 })?,
            drag,
            cages,
            water,
            thin_bed: setup.thin_bed,
            g: setup.g,
            rho0: setup.rho0,
            min_column_depth: setup.min_column_depth,
            sigma_bottom: sigma.sigma_rho()[0],
            eddy_viscosity: zeros(n_columns * (nl + 1))?,
            eddy_diffusivity: zeros(n_columns * (nl + 1))?,
            none: zeros(0)?,
        })
    }

    /// The step's drag rates of `state` (its start's): the bottom drag's of
    /// every column and the cages' of every caged layer.
    pub fn drag_rates(
        &mut self,
        s: &Statics3D,
        state: &State3D,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let (m, st) = (&self.gpu.module_vertical, &self.gpu.stream);
        let nl = s.n_levels as u32;
        // SAFETY: the buffers are sized for this mesh and these levels.
        unsafe {
            if let Some((kind, law)) = &self.drag {
                let n = s.n_columns();
                m.bottom_rates(
                    st,
                    launch(n, COLUMN_BLOCK),
                    nl,
                    n as u32,
                    *kind,
                    law,
                    self.min_column_depth,
                    self.sigma_bottom,
                    &state.eta,
                    &s.bed,
                    &state.u,
                    &state.v,
                    &state.ubar,
                    &state.vbar,
                    &mut self.bottom_rate,
                )?;
            }
            if let Some(cages) = &self.cages {
                m.layer_rates(
                    st,
                    launch(cages.n_nodes, COLUMN_BLOCK),
                    nl,
                    cages.n_nodes as u32,
                    &cages.nodes,
                    &cages.offsets,
                    &cages.coefficients,
                    &cages.net_depths,
                    self.min_column_depth,
                    &s.sigma_w,
                    &s.d_sigma,
                    &state.eta,
                    &s.bed,
                    &state.u,
                    &state.v,
                    &state.ubar,
                    &state.vbar,
                    &mut self.layer_rate,
                )?;
            }
        }
        Ok(())
    }

    /// `vertical_implicit` of `state` over `dt` with the step's surface
    /// fields and the drag rates of the last [`Self::drag_rates`].
    pub fn step(
        &mut self,
        s: &Statics3D,
        state: &mut State3D,
        surface: SurfaceDevice,
        dt: f64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let gpu = &self.gpu;
        let (m, st) = (&gpu.module_vertical, &gpu.stream);
        let none = &self.none;
        let n = s.n_columns();
        let [stress_x, stress_y] = surface.stress.unwrap_or([none, none]);
        let [nonsolar, shortwave, freshwater] = surface.tracers.unwrap_or([none, none, none]);
        let State3D {
            eta,
            ubar,
            vbar,
            u,
            v,
            temp,
            salt,
            tke,
            gls,
            rho,
        } = state;
        // SAFETY: the buffers are sized for this mesh and these levels; no
        // output is an input.
        unsafe {
            m.vertical_columns(
                st,
                launch(n, COLUMN_BLOCK),
                s.n_levels as u32,
                n as u32,
                &self.gls,
                &self.uniform,
                stress_x,
                stress_y,
                surface.roughness.unwrap_or(none),
                nonsolar,
                shortwave,
                freshwater,
                self.water,
                u32::from(self.thin_bed.is_some()),
                self.thin_bed.unwrap_or(0.0),
                &self.bottom_rate,
                &self.layer_rate,
                self.g,
                self.rho0,
                dt,
                self.min_column_depth,
                eta,
                &s.bed,
                ubar,
                vbar,
                &s.sigma_rho,
                &s.sigma_w,
                &s.d_sigma,
                rho,
                u,
                v,
                temp,
                salt,
                tke,
                gls,
                &mut self.eddy_viscosity,
                &mut self.eddy_diffusivity,
            )?;
        }
        Ok(())
    }
}

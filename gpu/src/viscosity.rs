//! The horizontal viscosity of the 3D shear on the GPU (P2.7 Phase 4.2b,
//! `notes/gpu.md`): dgcom's `apply_horizontal_viscosity_3d`, the Laplacian
//! along σ-surfaces of every layer's shear `u′_l = u_l − ⟨u⟩` by BR1, with a
//! constant background `ν₀` and Smagorinsky's `(C_s Δ)²|S_l|` from the
//! layer's own strain rate (its shear's gradient and the depth mean's),
//! added to the velocity tendencies as `∇·(ν_l D∇u′_l)/D`.
//!
//! The passes are the CPU's (`viscosity_3d.rs`, with `diffusion_2d.rs`'s
//! `br1_gradient_element` and `br1_diffusion_element`), each mirrored over
//! [`Statics3D`]'s tables: the columns' depth and depth mean (dgcom's
//! `depth_average`), the depth means' BR1 gradients (with Smagorinsky), each
//! layer's shear gradients with `ν_l D` (stored as the fluxes `ν_l D ∇u′_l`),
//! then their BR1 divergence. The CPU computes a block of levels as lanes,
//! each as alone, so a thread per element and level gives its bits. Walls
//! mirror the velocity (dgcom's `mirrored_velocity`) and pass the normal
//! stress alone (`wall_stress`); open faces copy it and pass none; thin
//! columns carry no shear and no mean, and get nothing. So a build without
//! multiply–add contraction (`--no-fmad`) gives the CPU's bits
//! (`momentum3d.rs`, the gate).

// As `three_d.rs`: the device loops index arrays by the loop variable.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use std::sync::Arc;

use cuda_core::DeviceBuffer;
use cuda_device::{DisjointSlice, device, kernel, thread};
use cuda_host::cuda_module;
use dgcom::operators::{DGOperators2D, GeometricFactors2D};
use dgcom::solver::rhs::HorizontalViscosity3D;
use dgcom::solver::rhs::viscosity_3d::{mirrored_velocity, smagorinsky_areas, wall_stress};
use dgcom::source::swe_2d::viscosity::strain_rate_magnitude;
use dgcom::vertical::depth_average;

use crate::listed::{On, listed_index};
use crate::swe2d::{Gpu, N1, NN, face_node};
use crate::three_d::{COLUMN_BLOCK, LEVEL_BLOCK, OPEN, Statics3D, WALL, launch};

/// The BR1 gradients of a pair `(u, v)` at the nodes of element `k`
/// (`br1_gradient_element` of two scalars): the chain rule of `D_r`, `D_s`
/// on `own` (the element's values), then each face's lifted jump to the
/// values beyond it, `exterior(s, node)` at face node slot `s` (the
/// neighbour's, or the interior's mirrored at a wall and copied at an open
/// face). `[u_x, u_y, v_x, v_y]` per node.
#[device]
#[allow(unused_attributes)]
#[inline(always)]
fn pair_gradients<F: Fn(usize, usize) -> [f64; 2]>(
    k: usize,
    own: &[[f64; 2]; NN],
    exterior: F,
    metric: &[[f64; 4]],
    dr: &[f64],
    ds: &[f64],
    fnode: &[[f64; 3]],
    across: &[u32],
    lift: &[f64],
) -> [[f64; 4]; NN] {
    let mut out = [[0.0; 4]; NN];
    #[unroll]
    for i in 0..NN {
        let (mut du_dr, mut du_ds) = ([0.0; 2], [0.0; 2]);
        #[unroll]
        for j in 0..NN {
            let (r, s) = (dr[i * NN + j], ds[i * NN + j]);
            #[unroll]
            for c in 0..2 {
                du_dr[c] += r * own[j][c];
                du_ds[c] += s * own[j][c];
            }
        }
        // `GeometricFactors2D::transform_derivatives`
        let [rx, ry, sx, sy] = metric[k * NN + i];
        #[unroll]
        for c in 0..2 {
            out[i][2 * c] = rx * du_dr[c] + sx * du_ds[c];
            out[i][2 * c + 1] = ry * du_dr[c] + sy * du_ds[c];
        }
    }
    #[unroll]
    for f in 0..4 {
        #[unroll]
        for fi in 0..N1 {
            let s = (k * 4 + f) * N1 + fi;
            let node = face_node(f, fi);
            let g = fnode[s];
            let interior = own[node];
            let nb = across[s];
            let outside = if nb == WALL {
                mirrored_velocity(interior, (g[0], g[1]))
            } else if nb == OPEN {
                interior
            } else {
                exterior(s, nb as usize)
            };
            let entry = lift[(f * NN + node) * N1 + fi];
            #[unroll]
            for c in 0..2 {
                let correction = 0.5 * (outside[c] - interior[c]);
                let lifted = entry * g[2] * correction;
                out[node][2 * c] += lifted * g[0];
                out[node][2 * c + 1] += lifted * g[1];
            }
        }
    }
    out
}

// =============================================================================
// Kernels
// =============================================================================

#[cuda_module]
pub mod kernels_viscosity {
    use super::*;

    /// Every column's depth (zero in a thin one) and depth means of u and v
    /// (zero in a thin one), one thread per column (`Columns::prepare`).
    #[kernel]
    pub fn viscosity_columns(
        nl: u32,
        n_columns: u32,
        min_column_depth: f64,
        eta: &[f64],
        bed: &[f64],
        u: &[f64],
        v: &[f64],
        d_sigma: &[f64],
        mut depth: DisjointSlice<f64>,
        mut mean: DisjointSlice<[f64; 2]>,
    ) {
        let idx = thread::index_1d().get();
        if idx >= n_columns as usize {
            return;
        }
        let nl = nl as usize;
        let h = eta[idx] - bed[idx];
        let d = if h < min_column_depth { 0.0 } else { h };
        let column = idx * nl..(idx + 1) * nl;
        let m = if d == 0.0 {
            [0.0; 2]
        } else {
            [
                depth_average(&u[column.clone()], &d_sigma[..nl]),
                depth_average(&v[column], &d_sigma[..nl]),
            ]
        };
        // SAFETY: thread idx alone writes column idx.
        unsafe {
            *depth.get_unchecked_mut(idx) = d;
            *mean.get_unchecked_mut(idx) = m;
        }
    }

    /// The depth means' BR1 gradients, one thread per element (with
    /// Smagorinsky's viscosity), `[u_x, u_y, v_x, v_y]` per node.
    #[kernel]
    pub fn viscosity_mean_gradients(
        n_elements: u32,
        mean: &[[f64; 2]],
        metric: &[[f64; 4]],
        dr: &[f64],
        ds: &[f64],
        fnode: &[[f64; 3]],
        across: &[u32],
        lift: &[f64],
        mut gradient: DisjointSlice<[f64; 4]>,
    ) {
        let k = thread::index_1d().get();
        if k >= n_elements as usize {
            return;
        }
        let mut own = [[0.0; 2]; NN];
        #[unroll]
        for i in 0..NN {
            own[i] = mean[k * NN + i];
        }
        let out = pair_gradients(
            k,
            &own,
            |_, nb| mean[nb],
            metric,
            dr,
            ds,
            fnode,
            across,
            lift,
        );
        // SAFETY: thread k alone writes element k's nodes.
        unsafe {
            #[unroll]
            for i in 0..NN {
                *gradient.get_unchecked_mut(k * NN + i) = out[i];
            }
        }
    }

    /// Every layer's shear gradients and viscosity, one thread per element
    /// and level (`Columns::layers`): the BR1 gradients of `(u′, v′)` (zero
    /// in thin columns), `ν = ν₀ + (C_s Δ)²|S|` with the strain rate of the
    /// shear's and the mean's gradients (`ν₀` alone without Smagorinsky:
    /// `mean_gradient` empty), and the fluxes `ν D ∇u′`, `ν D ∇v′` into
    /// `flux` (`[element][node][level]`, four each).
    #[kernel]
    pub fn viscosity_layer_gradients(
        nl: u32,
        n: u32,
        listed: &[u32],
        background: f64,
        depth: &[f64],
        mean: &[[f64; 2]],
        mean_gradient: &[[f64; 4]],
        area: &[f64],
        u: &[f64],
        v: &[f64],
        metric: &[[f64; 4]],
        dr: &[f64],
        ds: &[f64],
        fnode: &[[f64; 3]],
        across: &[u32],
        lift: &[f64],
        mut flux: DisjointSlice<[f64; 4]>,
    ) {
        let t = thread::index_1d().get();
        if t >= n as usize {
            return;
        }
        let nl = nl as usize;
        // (of the elements `listed`; empty: of every element)
        let t = listed_index(t, listed, nl);
        let (k, l) = (t / nl, t % nl);
        let shear = |idx: usize| {
            if depth[idx] == 0.0 {
                [0.0; 2]
            } else {
                let [mu, mv] = mean[idx];
                [u[idx * nl + l] - mu, v[idx * nl + l] - mv]
            }
        };
        let mut own = [[0.0; 2]; NN];
        #[unroll]
        for i in 0..NN {
            own[i] = shear(k * NN + i);
        }
        let g = pair_gradients(
            k,
            &own,
            |_, nb| shear(nb),
            metric,
            dr,
            ds,
            fnode,
            across,
            lift,
        );
        let smagorinsky = !mean_gradient.is_empty();
        // SAFETY: thread (k, l) alone writes level l of element k's nodes.
        unsafe {
            #[unroll]
            for i in 0..NN {
                let idx = k * NN + i;
                let [ux, uy, vx, vy] = g[i];
                let nu = if smagorinsky {
                    let m = mean_gradient[idx];
                    let strain = strain_rate_magnitude(ux + m[0], uy + m[1], vx + m[2], vy + m[3]);
                    background + area[k] * strain
                } else {
                    background
                };
                let c = nu * depth[idx];
                *flux.get_unchecked_mut(idx * nl + l) = [c * ux, c * uy, c * vx, c * vy];
            }
        }
    }

    /// The largest viscosity of each element's nodes that are not thin at
    /// each level, from the background up, into `largest`
    /// (`[element][level]`): `largest_horizontal_viscosity_3d`'s `ν D / D`
    /// with Smagorinsky's `ν` as [`viscosity_layer_gradients`] takes it, one
    /// thread per element and level.
    #[kernel]
    pub fn viscosity_largest(
        nl: u32,
        n: u32,
        background: f64,
        depth: &[f64],
        mean: &[[f64; 2]],
        mean_gradient: &[[f64; 4]],
        area: &[f64],
        u: &[f64],
        v: &[f64],
        metric: &[[f64; 4]],
        dr: &[f64],
        ds: &[f64],
        fnode: &[[f64; 3]],
        across: &[u32],
        lift: &[f64],
        mut largest: DisjointSlice<f64>,
    ) {
        let t = thread::index_1d().get();
        if t >= n as usize {
            return;
        }
        let nl = nl as usize;
        let (k, l) = (t / nl, t % nl);
        let shear = |idx: usize| {
            if depth[idx] == 0.0 {
                [0.0; 2]
            } else {
                let [mu, mv] = mean[idx];
                [u[idx * nl + l] - mu, v[idx * nl + l] - mv]
            }
        };
        let mut own = [[0.0; 2]; NN];
        #[unroll]
        for i in 0..NN {
            own[i] = shear(k * NN + i);
        }
        let g = pair_gradients(
            k,
            &own,
            |_, nb| shear(nb),
            metric,
            dr,
            ds,
            fnode,
            across,
            lift,
        );
        let mut most = background;
        #[unroll]
        for i in 0..NN {
            let idx = k * NN + i;
            let d = depth[idx];
            if d > 0.0 {
                let [ux, uy, vx, vy] = g[i];
                let m = mean_gradient[idx];
                let strain = strain_rate_magnitude(ux + m[0], uy + m[1], vx + m[2], vy + m[3]);
                let nu = background + area[k] * strain;
                most = most.max(nu * d / d);
            }
        }
        // SAFETY: thread (k, l) alone writes level l of element k.
        unsafe { *largest.get_unchecked_mut(t) = most };
    }

    /// Every element's largest of its levels' (`viscosity_largest`), one
    /// thread per element.
    #[kernel]
    pub fn viscosity_element_largest(
        nl: u32,
        ne: u32,
        levels: &[f64],
        mut largest: DisjointSlice<f64>,
    ) {
        let k = thread::index_1d().get();
        if k >= ne as usize {
            return;
        }
        let nl = nl as usize;
        let mut most = levels[k * nl];
        for l in 1..nl {
            most = most.max(levels[k * nl + l]);
        }
        // SAFETY: thread k alone writes element k.
        unsafe { *largest.get_unchecked_mut(k) = most };
    }

    /// The BR1 divergence of every layer's fluxes, one thread per element and
    /// level (`br1_diffusion_element`): the conservative volume term, the
    /// central face flux (at a wall the normal stress alone, at an open face
    /// none), added to `rhs_u`, `rhs_v` over the column's depth (none in a
    /// thin column).
    #[kernel]
    pub fn viscosity_diffusion(
        nl: u32,
        n: u32,
        listed: &[u32],
        depth: &[f64],
        flux: &[[f64; 4]],
        contra: &[[f64; 4]],
        jinv: &[f64],
        dr: &[f64],
        ds: &[f64],
        fnode: &[[f64; 3]],
        across: &[u32],
        lift: &[f64],
        mut rhs_u: DisjointSlice<f64>,
        mut rhs_v: DisjointSlice<f64>,
    ) {
        let t = thread::index_1d().get();
        if t >= n as usize {
            return;
        }
        let nl = nl as usize;
        let t = listed_index(t, listed, nl);
        let (k, l) = (t / nl, t % nl);
        // The nodes' fluxes `[F_u, F_v]` and their contravariant components
        let mut fx = [[0.0; 2]; NN];
        let mut fy = [[0.0; 2]; NN];
        let mut fr = [[0.0; 2]; NN];
        let mut fs = [[0.0; 2]; NN];
        #[unroll]
        for i in 0..NN {
            let f = flux[(k * NN + i) * nl + l];
            let a = contra[k * NN + i];
            #[unroll]
            for c in 0..2 {
                let (x, y) = (f[2 * c], f[2 * c + 1]);
                fx[i][c] = x;
                fy[i][c] = y;
                fr[i][c] = a[0] * x + a[1] * y;
                fs[i][c] = a[2] * x + a[3] * y;
            }
        }
        let mut out = [[0.0; 2]; NN];
        #[unroll]
        for i in 0..NN {
            let (mut dfr_dr, mut dfs_ds) = ([0.0; 2], [0.0; 2]);
            #[unroll]
            for j in 0..NN {
                let (r, s) = (dr[i * NN + j], ds[i * NN + j]);
                #[unroll]
                for c in 0..2 {
                    dfr_dr[c] += r * fr[j][c];
                    dfs_ds[c] += s * fs[j][c];
                }
            }
            let j_inv = jinv[k * NN + i];
            #[unroll]
            for c in 0..2 {
                out[i][c] = j_inv * (dfr_dr[c] + dfs_ds[c]);
            }
        }
        #[unroll]
        for f in 0..4 {
            #[unroll]
            for fi in 0..N1 {
                let s = (k * 4 + f) * N1 + fi;
                let node = face_node(f, fi);
                let g = fnode[s];
                let (nx, ny) = (g[0], g[1]);
                let entry = lift[(f * NN + node) * N1 + fi];
                let interior = [
                    fx[node][0] * nx + fy[node][0] * ny,
                    fx[node][1] * nx + fy[node][1] * ny,
                ];
                let nb = across[s];
                let star = if nb == WALL {
                    wall_stress(interior, (nx, ny))
                } else if nb == OPEN {
                    [0.0; 2]
                } else {
                    let e = flux[nb as usize * nl + l];
                    [
                        0.5 * (interior[0] + e[0] * nx + e[1] * ny),
                        0.5 * (interior[1] + e[2] * nx + e[3] * ny),
                    ]
                };
                #[unroll]
                for c in 0..2 {
                    out[node][c] += entry * g[2] * (star[c] - interior[c]);
                }
            }
        }
        // SAFETY: thread (k, l) alone writes level l of element k's nodes.
        unsafe {
            #[unroll]
            for i in 0..NN {
                let idx = k * NN + i;
                let d = depth[idx];
                if d > 0.0 {
                    let at = idx * nl + l;
                    *rhs_u.get_unchecked_mut(at) += out[i][0] / d;
                    *rhs_v.get_unchecked_mut(at) += out[i][1] / d;
                }
            }
        }
    }
}

// =============================================================================
// Host side
// =============================================================================

/// The horizontal viscosity of the 3D shear on the device (dgcom's
/// `apply_horizontal_viscosity_3d`).
pub struct Viscosity3D {
    gpu: Arc<Gpu>,
    viscosity: HorizontalViscosity3D,
    min_column_depth: f64,
    /// `(r_x, r_y, s_x, s_y)` of every node (`GeometricFactors2D::rx`, …)
    metric: DeviceBuffer<[f64; 4]>,
    /// `(C_s Δ)²` of every element (empty without Smagorinsky)
    area: DeviceBuffer<f64>,
    /// Every column's depth, zero in thin ones
    pub(crate) depth: DeviceBuffer<f64>,
    /// Every column's depth mean of u and v
    pub(crate) mean: DeviceBuffer<[f64; 2]>,
    /// Their BR1 gradients (empty without Smagorinsky)
    pub(crate) mean_gradient: DeviceBuffer<[f64; 4]>,
    /// Every layer's fluxes `ν D ∇u′`, `ν D ∇v′`
    flux: DeviceBuffer<[f64; 4]>,
    /// Every element, as a list
    none_u32: DeviceBuffer<u32>,
}

impl Viscosity3D {
    /// `viscosity` on `statics`' mesh (`ops`, `geom` its operators and
    /// geometry); columns shallower than `min_column_depth` are thin.
    pub fn new(
        gpu: &Arc<Gpu>,
        statics: &Statics3D,
        ops: &DGOperators2D,
        geom: &GeometricFactors2D,
        viscosity: HorizontalViscosity3D,
        min_column_depth: f64,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let stream = &gpu.stream;
        let (ne, nl) = (statics.n_elements, statics.n_levels);
        let metric: Vec<[f64; 4]> = (0..ne * NN)
            .map(|n| [geom.rx[n], geom.ry[n], geom.sx[n], geom.sy[n]])
            .collect();
        let smagorinsky = viscosity.smagorinsky > 0.0;
        let mut area = vec![0.0; if smagorinsky { ne } else { 0 }];
        if smagorinsky {
            smagorinsky_areas(viscosity, ops, geom, &mut area);
        }
        Ok(Self {
            gpu: gpu.clone(),
            viscosity,
            min_column_depth,
            metric: DeviceBuffer::from_host(stream, &metric)?,
            area: DeviceBuffer::from_host(stream, &area)?,
            depth: DeviceBuffer::zeroed(stream, ne * NN)?,
            mean: DeviceBuffer::zeroed(stream, ne * NN)?,
            mean_gradient: DeviceBuffer::zeroed(stream, if smagorinsky { ne * NN } else { 0 })?,
            flux: DeviceBuffer::zeroed(stream, ne * NN * nl)?,
            none_u32: DeviceBuffer::zeroed(stream, 0)?,
        })
    }

    /// The viscosity's velocity tendency of the state `(eta, u, v)` added to
    /// `rhs_u`, `rhs_v` (`apply_horizontal_viscosity_3d`) on the elements of
    /// `on` (the gradients, which their faces read, of every element).
    pub fn apply(
        &mut self,
        statics: &Statics3D,
        eta: &DeviceBuffer<f64>,
        u: &DeviceBuffer<f64>,
        v: &DeviceBuffer<f64>,
        rhs_u: &mut DeviceBuffer<f64>,
        rhs_v: &mut DeviceBuffer<f64>,
        on: On<'_>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if self.viscosity.is_zero() {
            return Ok(());
        }
        self.means(statics, eta, u, v)?;
        self.layers(statics, u, v, rhs_u, rhs_v, on)
    }

    /// [`Self::apply`]'s first part: the columns' depths and means, and with
    /// Smagorinsky the means' gradients.
    pub(crate) fn means(
        &mut self,
        statics: &Statics3D,
        eta: &DeviceBuffer<f64>,
        u: &DeviceBuffer<f64>,
        v: &DeviceBuffer<f64>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let gpu = self.gpu.clone();
        let (m, st) = (&gpu.module_viscosity, &gpu.stream);
        let s = statics;
        let n_columns = s.n_columns();
        // SAFETY: every buffer is sized for this mesh and these levels
        // (`Statics3D::upload`, `Self::new`, the caller's state); no output
        // is an input.
        unsafe {
            m.viscosity_columns(
                st,
                launch(n_columns, COLUMN_BLOCK),
                s.n_levels as u32,
                n_columns as u32,
                self.min_column_depth,
                eta,
                &s.bed,
                u,
                v,
                &s.d_sigma,
                &mut self.depth,
                &mut self.mean,
            )?;
            if self.viscosity.smagorinsky > 0.0 {
                m.viscosity_mean_gradients(
                    st,
                    launch(s.n_elements, COLUMN_BLOCK),
                    s.n_elements as u32,
                    &self.mean,
                    &self.metric,
                    &s.dr,
                    &s.ds,
                    &s.fnode,
                    &s.across,
                    &s.lift,
                    &mut self.mean_gradient,
                )?;
            }
        }
        Ok(())
    }

    /// [`Self::apply`]'s second part, after [`Self::means`]: every layer's
    /// gradients and viscosity, then their divergence into `rhs_u`, `rhs_v`.
    pub(crate) fn layers(
        &mut self,
        statics: &Statics3D,
        u: &DeviceBuffer<f64>,
        v: &DeviceBuffer<f64>,
        rhs_u: &mut DeviceBuffer<f64>,
        rhs_v: &mut DeviceBuffer<f64>,
        on: On<'_>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let gpu = self.gpu.clone();
        let (m, st) = (&gpu.module_viscosity, &gpu.stream);
        let s = statics;
        let nl = s.n_levels as u32;
        // The gradients of the listed elements and their neighbours (the
        // diffusion's faces read both sides' fluxes)
        let halo = on.halo();
        let n = halo.count(s.n_elements) * s.n_levels;
        let n_listed = on.count(s.n_elements) * s.n_levels;
        // SAFETY: as in `means`; `rhs_u`, `rhs_v` are none of the inputs.
        unsafe {
            m.viscosity_layer_gradients(
                st,
                launch(n, LEVEL_BLOCK),
                nl,
                n as u32,
                halo.listed(&self.none_u32),
                self.viscosity.background,
                &self.depth,
                &self.mean,
                &self.mean_gradient,
                &self.area,
                u,
                v,
                &self.metric,
                &s.dr,
                &s.ds,
                &s.fnode,
                &s.across,
                &s.lift,
                &mut self.flux,
            )?;
            m.viscosity_diffusion(
                st,
                launch(n_listed, LEVEL_BLOCK),
                nl,
                n_listed as u32,
                on.listed(&self.none_u32),
                &self.depth,
                &self.flux,
                &s.contra,
                &s.jinv,
                &s.dr,
                &s.ds,
                &s.fnode,
                &s.across,
                &s.lift,
                rhs_u,
                rhs_v,
            )?;
        }
        Ok(())
    }

    /// `largest_horizontal_viscosity_3d` of the state `(eta, u, v)` with
    /// Smagorinsky: every element's largest viscosity into `largest`
    /// (`[element]`), each level's into `levels` (`[element][level]`).
    pub(crate) fn largest(
        &mut self,
        statics: &Statics3D,
        eta: &DeviceBuffer<f64>,
        u: &DeviceBuffer<f64>,
        v: &DeviceBuffer<f64>,
        levels: &mut DeviceBuffer<f64>,
        largest: &mut DeviceBuffer<f64>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        assert!(self.viscosity.smagorinsky > 0.0, "Smagorinsky's viscosity");
        self.means(statics, eta, u, v)?;
        let gpu = self.gpu.clone();
        let (m, st) = (&gpu.module_viscosity, &gpu.stream);
        let s = statics;
        let nl = s.n_levels as u32;
        let n = s.n_elements * s.n_levels;
        // SAFETY: as in `means`; `levels` holds a value per element and
        // level, `largest` one per element.
        unsafe {
            m.viscosity_largest(
                st,
                launch(n, LEVEL_BLOCK),
                nl,
                n as u32,
                self.viscosity.background,
                &self.depth,
                &self.mean,
                &self.mean_gradient,
                &self.area,
                u,
                v,
                &self.metric,
                &s.dr,
                &s.ds,
                &s.fnode,
                &s.across,
                &s.lift,
                levels,
            )?;
            m.viscosity_element_largest(
                st,
                launch(s.n_elements, COLUMN_BLOCK),
                nl,
                s.n_elements as u32,
                levels,
                largest,
            )?;
        }
        Ok(())
    }
}

//! The 3D tracer limiter on the GPU (P2.7 Phase 4.3b, `notes/gpu.md`):
//! dgcom's `apply_tracer_limiters_3d`, which `Hydrostatic3D::post_stage`
//! applies to every stage's values before their density.
//!
//! - **The horizontal Kuzmin limiter** (`TracerLimiterType3D::
//!   HorizontalKuzmin`) of T's and S's departures from the reference
//!   stratification, bounded at each node's height by its vertex patches'
//!   columns. Its pre-passes: every layer centre's height and the reference
//!   there ([`kernels_limiter::kuzmin_centres`], a thread per value); every
//!   element layer's mean height, the departures' layer means and the end
//!   layers' ranges ([`kernels_limiter::kuzmin_means`], a thread per element
//!   and level). Then a thread per element and level reads only those of its
//!   vertex patches' elements and writes its own nodes of both tracers
//!   ([`kernels_limiter::kuzmin_limit`]): what `apply_horizontal_kuzmin_fields`
//!   does to an element's layer, with dgcom's `HeightBracket`,
//!   `relaxed_bounds` and `kuzmin_layer_alpha`.
//! - **The bounds** (`TracerBounds` with its `TracerAveragePolicy3D`), a
//!   thread per element and level ([`kernels_limiter::bounds`]), with
//!   dgcom's `compute_theta`.
//!
//! The thin columns (shallower than the model's minimum column depth) are
//! left alone and weigh nothing, as on the CPU. The vertical column bounds
//! are not on the device (the case runner does not set them). The limiter
//! reports no statistics: `post_stage` updates the density whether or not
//! it changed anything, so the device does it always ([`StageHooks3D`]).
//!
//! [`StageHooks3D`]: crate::hooks::StageHooks3D

// As `three_d.rs`: the device loops index arrays by the loop variable.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use std::sync::Arc;

use cuda_core::DeviceBuffer;
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;
use dgcom::mesh::Mesh2D;
use dgcom::solver::limiters::tracer_3d::{
    HeightBracket, LIMITER_EPS, MIN_INTEGRAL_WEIGHT, compute_theta, kuzmin_layer_alpha,
    node_to_vertex, relaxed_bounds,
};
use dgcom::solver::rhs::transport_3d::layer_thickness_of;
use dgcom::solver::{
    ReferenceProfileView, TracerAveragePolicy3D, TracerLimiter3DConfig, TracerLimiterType3D,
};
use dgcom::types::ElementIndex;

use crate::baroclinic::ProfileDevice;
use crate::hooks::State3D;
use crate::listed::{On, listed_index};
use crate::swe2d::{Gpu, N1, NN};
use crate::three_d::{LEVEL_BLOCK, Statics3D, launch};

// =============================================================================
// Kernels
// =============================================================================

#[cuda_module]
pub mod kernels_limiter {
    use super::*;

    /// `fill_centre_heights` and the reference there, one thread per value:
    /// the layer centre's height `η + max(D, 0) σ`, and the reference
    /// profile's T and S at it (its samples `heights`, `temp`, `salt`, their
    /// slopes if cubic, else empty; `inverse_spacing` if `even`;
    /// `inverse_widths`), by dgcom's `ReferenceProfileView::at`; zero without
    /// a profile (`heights` empty).
    #[kernel]
    pub fn kuzmin_centres(
        nl: u32,
        n: u32,
        heights: &[f64],
        temp: &[f64],
        salt: &[f64],
        slopes_t: &[f64],
        slopes_s: &[f64],
        even: u32,
        inverse_spacing: f64,
        inverse_widths: &[f64],
        eta: &[f64],
        bed: &[f64],
        sigma_rho: &[f64],
        mut centres: DisjointSlice<f64>,
        mut ref_t: DisjointSlice<f64>,
        mut ref_s: DisjointSlice<f64>,
    ) {
        let t = thread::index_1d().get();
        if t >= n as usize {
            return;
        }
        let nl = nl as usize;
        let (col, level) = (t / nl, t % nl);
        let eta = eta[col];
        let depth = (eta - bed[col]).max(0.0);
        let z = eta + depth * sigma_rho[level];
        let (rt, rs) = if heights.is_empty() {
            (0.0, 0.0)
        } else {
            ReferenceProfileView {
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
            }
            .at(z)
        };
        // SAFETY: thread t alone writes value t.
        unsafe {
            *centres.get_unchecked_mut(t) = z;
            *ref_t.get_unchecked_mut(t) = rt;
            *ref_s.get_unchecked_mut(t) = rs;
        }
    }

    /// `LayerColumns::layer_mean_heights` and `KuzminTracer::new`'s layer
    /// means, one thread per element and level: the inventory-weighted mean
    /// of the layer centres' heights and of T's and S's departures from the
    /// reference (NaN for a layer without water); the bed layer's thread the
    /// range of the departures over the element's nodes in the bed layer,
    /// the surface layer's in the surface layer (`end_t`, `end_s`: `[bottom
    /// min, bottom max, top min, top max]` per element).
    #[kernel]
    pub fn kuzmin_means(
        nl: u32,
        n: u32,
        eta: &[f64],
        bed: &[f64],
        mass: &[f64],
        d_sigma: &[f64],
        centres: &[f64],
        temp: &[f64],
        salt: &[f64],
        ref_t: &[f64],
        ref_s: &[f64],
        mut heights: DisjointSlice<f64>,
        mut avg_t: DisjointSlice<f64>,
        mut avg_s: DisjointSlice<f64>,
        mut end_t: DisjointSlice<f64>,
        mut end_s: DisjointSlice<f64>,
    ) {
        let t = thread::index_1d().get();
        if t >= n as usize {
            return;
        }
        let nl = nl as usize;
        let (k, level) = (t / nl, t % nl);
        let ds = d_sigma[level];
        let mut weight_sum = 0.0;
        let mut inventory = [0.0; 3];
        for i in 0..NN {
            let col = k * NN + i;
            let j = col * nl + level;
            let weight = mass[col] * layer_thickness_of(eta[col] - bed[col], ds);
            weight_sum += weight;
            inventory[0] += weight * centres[j];
            inventory[1] += weight * (temp[j] - ref_t[j]);
            inventory[2] += weight * (salt[j] - ref_s[j]);
        }
        let mean = |inventory: f64| {
            if weight_sum > MIN_INTEGRAL_WEIGHT {
                inventory / weight_sum
            } else {
                f64::NAN
            }
        };
        // SAFETY: thread t alone writes element layer t, and the bed (top)
        // layer's thread alone its element's bottom (top) range.
        unsafe {
            *heights.get_unchecked_mut(t) = mean(inventory[0]);
            *avg_t.get_unchecked_mut(t) = mean(inventory[1]);
            *avg_s.get_unchecked_mut(t) = mean(inventory[2]);
            for (end, at) in [(0, 0), (nl - 1, 2)] {
                if level != end {
                    continue;
                }
                let mut range = [[f64::INFINITY, f64::NEG_INFINITY]; 2];
                for i in 0..NN {
                    let j = (k * NN + i) * nl + level;
                    let d = [temp[j] - ref_t[j], salt[j] - ref_s[j]];
                    for c in 0..2 {
                        range[c] = [range[c][0].min(d[c]), range[c][1].max(d[c])];
                    }
                }
                for (c, out) in [end_t.as_mut_ptr(), end_s.as_mut_ptr()]
                    .into_iter()
                    .enumerate()
                {
                    *out.add(k * 4 + at) = range[c][0];
                    *out.add(k * 4 + at + 1) = range[c][1];
                }
            }
        }
    }

    /// `apply_horizontal_kuzmin_fields` on an element's layer, one thread per
    /// element and level, both tracers: each node's departure from the
    /// reference, bounded at its height by its vertex patch's columns (a
    /// vertex's own patch, `element_vertices` into `vertex_offsets`,
    /// `vertex_elements`; any other node the union of the element's,
    /// `union_offsets`, `union_elements`), scaled about the element's own
    /// column shifted to keep the inventory (`kuzmin_layer_alpha`). Reads the
    /// pre-passes of the patches' elements and writes the element's own
    /// nodes of the level; the thin columns are left alone. Only the
    /// elements `listed` (empty: every element).
    #[kernel]
    pub fn kuzmin_limit(
        nl: u32,
        n: u32,
        listed: &[u32],
        relaxation: f64,
        thin_depth: f64,
        eta: &[f64],
        bed: &[f64],
        mass: &[f64],
        d_sigma: &[f64],
        centres: &[f64],
        heights: &[f64],
        ref_t: &[f64],
        ref_s: &[f64],
        avg_t: &[f64],
        avg_s: &[f64],
        end_t: &[f64],
        end_s: &[f64],
        element_vertices: &[u32],
        vertex_offsets: &[u32],
        vertex_elements: &[u32],
        union_offsets: &[u32],
        union_elements: &[u32],
        mut temp: DisjointSlice<f64>,
        mut salt: DisjointSlice<f64>,
    ) {
        let t = thread::index_1d().get();
        if t >= n as usize {
            return;
        }
        let nl = nl as usize;
        let t = listed_index(t, listed, nl);
        let (k, level) = (t / nl, t % nl);
        let mut thin = [false; NN];
        let mut all_thin = true;
        for i in 0..NN {
            let col = k * NN + i;
            thin[i] = eta[col] - bed[col] < thin_depth;
            all_thin &= thin[i];
        }
        if all_thin {
            return;
        }
        let averages = [avg_t, avg_s];
        let references = [ref_t, ref_s];
        let ends = [end_t, end_s];
        let end_range = |c: usize, e: usize| {
            let r = &ends[c][e * 4..e * 4 + 4];
            [r[0], r[1], r[2], r[3]]
        };
        let present = [0, 1].map(|c| averages[c][k * nl + level].is_finite());
        if !present[0] && !present[1] {
            return;
        }
        let fields = [temp.as_mut_ptr(), salt.as_mut_ptr()];
        let own_column = k * nl..(k + 1) * nl;
        let ds = d_sigma[level];
        // The element's own column at each node's height, `T̂_i`
        let mut height = [0.0; NN];
        let mut weight = [0.0; NN];
        let mut reference = [[0.0; NN]; 2];
        let mut value = [[0.0; NN]; 2];
        let mut base = [[0.0; NN]; 2];
        let mut scale = [[0.0; NN]; 2];
        let mut weight_sum = [0.0; 2];
        let mut departure = [0.0; 2];
        let mut within_own = [true; 2];
        for i in 0..NN {
            let col = k * NN + i;
            let j = col * nl + level;
            let z = centres[j];
            let bracket = HeightBracket::find(&heights[own_column.clone()], z, level);
            height[i] = z;
            // A thin column weighs nothing and is not limited
            let node_weight = if thin[i] {
                0.0
            } else {
                mass[col] * layer_thickness_of(eta[col] - bed[col], ds)
            };
            weight[i] = node_weight;
            for c in 0..2 {
                if !present[c] {
                    continue;
                }
                let (own, lower, upper) =
                    bracket.column_value(&averages[c][own_column.clone()], end_range(c, k));
                // SAFETY: thread t alone reads and writes element layer t
                // of either tracer.
                let field_value = unsafe { *fields[c].add(j) };
                let node_reference = references[c][j];
                let node_value = field_value - node_reference;
                let node_scale = field_value.abs().max(node_reference.abs());
                let (lo, hi) = relaxed_bounds(
                    own.min(lower).min(upper),
                    own.max(lower).max(upper),
                    relaxation,
                    node_scale,
                );
                within_own[c] &= thin[i] || (lo..=hi).contains(&node_value);
                weight_sum[c] += node_weight;
                departure[c] += node_weight * (node_value - own);
                reference[c][i] = node_reference;
                value[c][i] = node_value;
                base[c][i] = own;
                scale[c][i] = node_scale;
            }
        }
        let limit = [0, 1].map(|c| present[c] && !within_own[c]);
        if !limit[0] && !limit[1] {
            return;
        }
        // The bounds: every vertex within its own patch's, every other node
        // within the union's
        let mut bounds = [[(0.0, 0.0); NN]; 2];
        for i in 0..NN {
            let (patch, start, end) = match node_to_vertex(i, N1) {
                Some(local_vertex) => {
                    let vertex = element_vertices[k * 4 + local_vertex] as usize;
                    (
                        vertex_elements,
                        vertex_offsets[vertex] as usize,
                        vertex_offsets[vertex + 1] as usize,
                    )
                }
                None => (
                    union_elements,
                    union_offsets[k] as usize,
                    union_offsets[k + 1] as usize,
                ),
            };
            let mut range = [(f64::INFINITY, f64::NEG_INFINITY); 2];
            for p in start..end {
                let e = patch[p] as usize;
                let column = e * nl..(e + 1) * nl;
                let bracket = HeightBracket::find(&heights[column.clone()], height[i], level);
                for c in 0..2 {
                    if !limit[c] || !averages[c][e * nl + level].is_finite() {
                        continue;
                    }
                    let (value, a, b) =
                        bracket.column_value(&averages[c][column.clone()], end_range(c, e));
                    let (lo, hi) = range[c];
                    range[c] = (lo.min(value.min(a).min(b)), hi.max(value.max(a).max(b)));
                }
            }
            for c in 0..2 {
                if limit[c] {
                    let (bound_min, bound_max) = range[c];
                    bounds[c][i] = relaxed_bounds(bound_min, bound_max, relaxation, scale[c][i]);
                }
            }
        }
        for c in 0..2 {
            if !limit[c] {
                continue;
            }
            let alpha = kuzmin_layer_alpha(
                departure[c],
                weight_sum[c],
                &mut base[c],
                &value[c],
                &weight,
                &bounds[c],
                |i| thin[i],
            );
            if alpha >= 1.0 - LIMITER_EPS {
                continue;
            }
            for i in 0..NN {
                if thin[i] {
                    continue;
                }
                let j = (k * NN + i) * nl + level;
                // SAFETY: as above.
                unsafe {
                    *fields[c].add(j) =
                        reference[c][i] + base[c][i] + alpha * (value[c][i] - base[c][i]);
                }
            }
        }
    }

    /// `apply_horizontal_bounds_field` on an element's layer, one thread per
    /// element and level, T then S: `bounds` (`[T min, T max, S min, S
    /// max]`), a layer whose mean is outside them its mean (`enforce`: the
    /// mean clamped into them), else its deviations from it scaled into them
    /// (`compute_theta`); the thin columns left alone, weighing nothing.
    /// Only the elements `listed` (empty: every element).
    #[kernel]
    pub fn bounds(
        nl: u32,
        n: u32,
        listed: &[u32],
        thin_depth: f64,
        bounds: &[f64],
        enforce: u32,
        eta: &[f64],
        bed: &[f64],
        mass: &[f64],
        d_sigma: &[f64],
        mut temp: DisjointSlice<f64>,
        mut salt: DisjointSlice<f64>,
    ) {
        let t = thread::index_1d().get();
        if t >= n as usize {
            return;
        }
        let nl = nl as usize;
        let t = listed_index(t, listed, nl);
        let (k, level) = (t / nl, t % nl);
        let mut thin = [false; NN];
        for i in 0..NN {
            let col = k * NN + i;
            thin[i] = eta[col] - bed[col] < thin_depth;
        }
        let ds = d_sigma[level];
        let fields = [temp.as_mut_ptr(), salt.as_mut_ptr()];
        for c in 0..2 {
            let (bound_min, bound_max) = (bounds[2 * c], bounds[2 * c + 1]);
            let field = fields[c];
            // SAFETY: thread t alone reads and writes element layer t of
            // either tracer.
            unsafe {
                let mut weight_sum = 0.0;
                let mut inventory = 0.0;
                let mut min_value = f64::INFINITY;
                let mut max_value = f64::NEG_INFINITY;
                for i in 0..NN {
                    if thin[i] {
                        continue;
                    }
                    let col = k * NN + i;
                    let value = *field.add(col * nl + level);
                    let weight = mass[col] * layer_thickness_of(eta[col] - bed[col], ds);
                    weight_sum += weight;
                    inventory += weight * value;
                    min_value = min_value.min(value);
                    max_value = max_value.max(value);
                }
                if weight_sum <= MIN_INTEGRAL_WEIGHT {
                    continue;
                }
                let avg = inventory / weight_sum;
                if avg < bound_min - LIMITER_EPS || avg > bound_max + LIMITER_EPS {
                    let replacement = if enforce != 0 {
                        avg.clamp(bound_min, bound_max)
                    } else {
                        avg
                    };
                    for i in 0..NN {
                        if !thin[i] {
                            *field.add((k * NN + i) * nl + level) = replacement;
                        }
                    }
                    continue;
                }
                if min_value >= bound_min - LIMITER_EPS && max_value <= bound_max + LIMITER_EPS {
                    continue;
                }
                let theta = compute_theta(avg, min_value, max_value, bound_min, bound_max);
                if theta >= 1.0 - LIMITER_EPS {
                    continue;
                }
                for i in 0..NN {
                    if !thin[i] {
                        let at = field.add((k * NN + i) * nl + level);
                        *at = avg + theta * (*at - avg);
                    }
                }
            }
        }
    }
}

// =============================================================================
// Host side
// =============================================================================

/// A mesh's vertex patches as the Kuzmin limiter reads them: every element's
/// vertices, every vertex's elements (`Mesh2D::elements_at_vertex`) and the
/// union of every element's vertex patches in the CPU's order, as
/// offsets into lists.
#[derive(Clone, Debug)]
pub struct VertexPatches {
    pub element_vertices: Vec<u32>,
    pub vertex_offsets: Vec<u32>,
    pub vertex_elements: Vec<u32>,
    pub union_offsets: Vec<u32>,
    pub union_elements: Vec<u32>,
}

impl VertexPatches {
    pub fn of(mesh: &Mesh2D) -> Self {
        let mut vertex_offsets = vec![0];
        let mut vertex_elements = Vec::new();
        for vertex in 0..mesh.n_vertices {
            vertex_elements.extend(mesh.elements_at_vertex(vertex).iter().map(|&e| e as u32));
            vertex_offsets.push(vertex_elements.len() as u32);
        }
        let mut element_vertices = Vec::with_capacity(4 * mesh.n_elements);
        let mut union_offsets = vec![0];
        let mut union_elements = Vec::new();
        for k in 0..mesh.n_elements {
            let vertices = mesh.element_vertex_indices(ElementIndex::new(k));
            element_vertices.extend(vertices.map(|v| v as u32));
            // As `apply_horizontal_kuzmin_fields` gathers it
            let start = union_elements.len();
            for vertex in vertices {
                for &e in mesh.elements_at_vertex(vertex) {
                    if !union_elements[start..].contains(&(e as u32)) {
                        union_elements.push(e as u32);
                    }
                }
            }
            union_offsets.push(union_elements.len() as u32);
        }
        Self {
            element_vertices,
            vertex_offsets,
            vertex_elements,
            union_offsets,
            union_elements,
        }
    }
}

/// [`VertexPatches`] on the device.
struct PatchesDevice {
    element_vertices: DeviceBuffer<u32>,
    vertex_offsets: DeviceBuffer<u32>,
    vertex_elements: DeviceBuffer<u32>,
    union_offsets: DeviceBuffer<u32>,
    union_elements: DeviceBuffer<u32>,
}

/// The horizontal Kuzmin limiter's data on the device: its relaxation, the
/// reference profile, the vertex patches, and its pre-passes' buffers.
struct Kuzmin3D {
    relaxation: f64,
    profile: Option<ProfileDevice>,
    patches: PatchesDevice,
    /// Every value's layer-centre height, and T's and S's reference there
    centres: DeviceBuffer<f64>,
    ref_t: DeviceBuffer<f64>,
    ref_s: DeviceBuffer<f64>,
    /// Every element layer's mean height and T's and S's mean departures
    heights: DeviceBuffer<f64>,
    avg_t: DeviceBuffer<f64>,
    avg_s: DeviceBuffer<f64>,
    /// Every element's end layers' ranges of T's and S's departures
    end_t: DeviceBuffer<f64>,
    end_s: DeviceBuffer<f64>,
    none: DeviceBuffer<f64>,
}

/// dgcom's `apply_tracer_limiters_3d` with a `TracerLimiter3DConfig` on the
/// device, on every element.
pub struct Limiter3D {
    gpu: Arc<Gpu>,
    /// Columns shallower than this are left alone
    thin_depth: f64,
    /// `[T min, T max, S min, S max]`, if any limiter is on
    bounds: Option<DeviceBuffer<f64>>,
    /// Out-of-bounds layer means clamped (`TracerAveragePolicy3D::
    /// EnforceBounds`)
    enforce: bool,
    kuzmin: Option<Kuzmin3D>,
    /// Every element, as a list
    none_u32: DeviceBuffer<u32>,
}

impl Limiter3D {
    /// `config` on `statics`' mesh with its vertex `patches`, the columns
    /// shallower than `thin_depth` left alone; refuses the vertical column
    /// bounds.
    pub fn new(
        gpu: &Arc<Gpu>,
        statics: &Statics3D,
        patches: &VertexPatches,
        config: &TracerLimiter3DConfig,
        thin_depth: f64,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if config.vertical_column_bounds {
            return Err("the vertical column bounds are not on the device".into());
        }
        let st = &gpu.stream;
        let b = config.bounds;
        let bounds = match config.limiter_type {
            TracerLimiterType3D::None => None,
            _ => Some(DeviceBuffer::from_host(
                st,
                &[b.t_min, b.t_max, b.s_min, b.s_max],
            )?),
        };
        let kuzmin = match config.limiter_type {
            TracerLimiterType3D::HorizontalKuzmin { relaxation } => {
                let up = |v: &[u32]| DeviceBuffer::from_host(st, v);
                let (ne, nl) = (statics.n_elements, statics.n_levels);
                let zeros = |n: usize| DeviceBuffer::<f64>::zeroed(st, n);
                Some(Kuzmin3D {
                    relaxation,
                    profile: match &config.reference {
                        Some(profile) => Some(ProfileDevice::upload(gpu, profile)?),
                        None => None,
                    },
                    patches: PatchesDevice {
                        element_vertices: up(&patches.element_vertices)?,
                        vertex_offsets: up(&patches.vertex_offsets)?,
                        vertex_elements: up(&patches.vertex_elements)?,
                        union_offsets: up(&patches.union_offsets)?,
                        union_elements: up(&patches.union_elements)?,
                    },
                    centres: zeros(ne * NN * nl)?,
                    ref_t: zeros(ne * NN * nl)?,
                    ref_s: zeros(ne * NN * nl)?,
                    heights: zeros(ne * nl)?,
                    avg_t: zeros(ne * nl)?,
                    avg_s: zeros(ne * nl)?,
                    end_t: zeros(ne * 4)?,
                    end_s: zeros(ne * 4)?,
                    none: zeros(0)?,
                })
            }
            _ => None,
        };
        Ok(Self {
            gpu: gpu.clone(),
            thin_depth,
            bounds,
            enforce: config.average_policy == TracerAveragePolicy3D::EnforceBounds,
            kuzmin,
            none_u32: DeviceBuffer::zeroed(st, 0)?,
        })
    }

    /// Limit `values`' T and S (under their `η`) on the elements of `on`:
    /// the Kuzmin limiter's bounds from every element's values as they are.
    pub fn apply(
        &mut self,
        s: &Statics3D,
        values: &mut State3D,
        on: On<'_>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let Some(bounds) = &self.bounds else {
            return Ok(());
        };
        let (m, st) = (&self.gpu.module_limiter, &self.gpu.stream);
        let nl = s.n_levels as u32;
        let n_layers = s.n_elements * s.n_levels;
        let listed = on.listed(&self.none_u32);
        let n_limited = on.count(s.n_elements) * s.n_levels;
        let State3D {
            eta, temp, salt, ..
        } = values;
        if let Some(kz) = &mut self.kuzmin {
            let n_values = s.n_columns() * s.n_levels;
            let none = &kz.none;
            let (heights, temp_p, salt_p, slopes, even, inverse_spacing, widths) = match &kz.profile
            {
                Some(p) => (
                    &p.heights,
                    &p.temp,
                    &p.salt,
                    [&p.slopes[0], &p.slopes[1]],
                    u32::from(p.inverse_spacing.is_some()),
                    p.inverse_spacing.unwrap_or(0.0),
                    &p.inverse_widths,
                ),
                None => (none, none, none, [none, none], 0, 0.0, none),
            };
            // SAFETY: every buffer is sized for this mesh and these levels
            // (`Statics3D::upload`, `Self::new`, the caller's state); no
            // output is an input of the same kernel.
            unsafe {
                m.kuzmin_centres(
                    st,
                    launch(n_values, LEVEL_BLOCK),
                    nl,
                    n_values as u32,
                    heights,
                    temp_p,
                    salt_p,
                    slopes[0],
                    slopes[1],
                    even,
                    inverse_spacing,
                    widths,
                    eta,
                    &s.bed,
                    &s.sigma_rho,
                    &mut kz.centres,
                    &mut kz.ref_t,
                    &mut kz.ref_s,
                )?;
                m.kuzmin_means(
                    st,
                    launch(n_layers, LEVEL_BLOCK),
                    nl,
                    n_layers as u32,
                    eta,
                    &s.bed,
                    &s.mass,
                    &s.d_sigma,
                    &kz.centres,
                    temp,
                    salt,
                    &kz.ref_t,
                    &kz.ref_s,
                    &mut kz.heights,
                    &mut kz.avg_t,
                    &mut kz.avg_s,
                    &mut kz.end_t,
                    &mut kz.end_s,
                )?;
                let p = &kz.patches;
                m.kuzmin_limit(
                    st,
                    launch(n_limited, LEVEL_BLOCK),
                    nl,
                    n_limited as u32,
                    listed,
                    kz.relaxation,
                    self.thin_depth,
                    eta,
                    &s.bed,
                    &s.mass,
                    &s.d_sigma,
                    &kz.centres,
                    &kz.heights,
                    &kz.ref_t,
                    &kz.ref_s,
                    &kz.avg_t,
                    &kz.avg_s,
                    &kz.end_t,
                    &kz.end_s,
                    &p.element_vertices,
                    &p.vertex_offsets,
                    &p.vertex_elements,
                    &p.union_offsets,
                    &p.union_elements,
                    temp,
                    salt,
                )?;
            }
        }
        // SAFETY: as above.
        unsafe {
            m.bounds(
                st,
                launch(n_limited, LEVEL_BLOCK),
                nl,
                n_limited as u32,
                listed,
                self.thin_depth,
                bounds,
                u32::from(self.enforce),
                eta,
                &s.bed,
                &s.mass,
                &s.d_sigma,
                temp,
                salt,
            )?;
        }
        Ok(())
    }
}

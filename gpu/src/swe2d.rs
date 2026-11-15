//! The 2D SWE on the GPU (P2.7, `notes/gpu.md`): a device-resident state
//! ([`DeviceState`], dgcom's [`Integrable`]) and a physics module
//! ([`GpuSWE2D`], dgcom's [`PhysicsModule`]), so that dgcom's own
//! `Simulation` and SSP-RK steppers drive the GPU.
//!
//! Phase 1's first scope: the fully wet split-form RHS
//! (`SWEFormulation2D::EntropyStable`) with a bed, on meshes without
//! boundary faces (periodic), no sources, global steps. The kernels follow
//! `SplitFormSWE2D` (`src/solver/rhs/swe_2d_split_form.rs`) operation for
//! operation and call dgcom's pointwise physics, so a build without
//! multiply–add contraction (`cargo oxide run --no-fmad`) gives the CPU's
//! bits. One thread per edge and one per element; every loop unrolls (P2's
//! bounds are constants, the face nodes a formula), so an element's nodes and
//! RHS stay in registers.

use std::sync::{Arc, Mutex};

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig, Stream};
use cuda_device::{DisjointSlice, device, kernel, thread};
use cuda_host::cuda_module;
use dgcom::SWEState2D;
use dgcom::equations::ShallowWater2D;
use dgcom::flux::{
    SWENodeState2D, entropy_stable_dissipation_2d, wintermeyer_bed_interface_term_2d,
    wintermeyer_flux_2d,
};
use dgcom::mesh::{Bathymetry2D, Mesh2D};
use dgcom::operators::{DGOperators2D, GeometricFactors2D};
use dgcom::physics::{PhysicsModule, PhysicsModuleInfo};
use dgcom::solver::{SWESolution2D, dt_from_reference_rate, node_reference_rate};
use dgcom::time::Integrable;
use dgcom::types::{Depth, ElementIndex};

/// Polynomial order of the kernels (P2).
pub const ORDER: usize = 2;
/// Nodes per GLL line.
pub const N1: usize = ORDER + 1;
/// Nodes per element.
pub const NN: usize = N1 * N1;
/// Face-buffer slots per edge: the left side's `N1`, then the right side's.
pub const NF: usize = 2 * N1;
/// Per-element geometry of a parallelogram: `J∇r` (2), `J∇s` (2), `1/J`,
/// then each face's outward normal (2) and surface Jacobian.
const EG: usize = 17;

/// Threads per block of the element kernels (their registers bound the block).
const ELEMENT_BLOCK: u32 = 128;

/// Volume node of face node `fi` of `face` (`DGOperators2D::face_nodes`:
/// faces 2 and 3 list theirs in reverse). A formula, not a table, so that
/// with the loops unrolled the element's nodes are indexed by constants and
/// stay in registers.
#[inline]
const fn face_node(face: usize, fi: usize) -> usize {
    match face {
        0 => fi,
        1 => fi * N1 + N1 - 1,
        2 => (N1 - 1) * N1 + N1 - 1 - fi,
        _ => (N1 - 1 - fi) * N1,
    }
}

// =============================================================================
// Device helpers: dgcom's pointwise physics, arranged as SplitFormSWE2D does
// =============================================================================

/// Node `n` (`k·NN + i`) of the state, as `SplitFormSWE2D` reads it.
#[inline]
fn node_state(
    h: &[f64],
    hu: &[f64],
    hv: &[f64],
    bed: &[f64],
    n: usize,
    h_min: f64,
) -> SWENodeState2D {
    SWENodeState2D::new(&SWEState2D::new(h[n], hu[n], hv[n]), bed[n], h_min)
}

/// `SplitFormSWE2D::surface_flux` for `SurfaceFlux::EntropyStable`: `F*·n`
/// seen from side a, and `F*·(−n)` from side b.
#[inline]
fn surface_flux(
    q_a: &SWENodeState2D,
    q_b: &SWENodeState2D,
    n: (f64, f64),
    g: f64,
) -> (SWEState2D, SWEState2D) {
    let mut shared = wintermeyer_flux_2d(q_a, q_b, n, g);
    shared = shared + entropy_stable_dissipation_2d(q_a, q_b, n, g);
    let minus_n = (-n.0, -n.1);
    (
        shared + wintermeyer_bed_interface_term_2d(q_a.h, q_a.b, q_b.b, n, g),
        -shared + wintermeyer_bed_interface_term_2d(q_b.h, q_b.b, q_a.b, minus_n, g),
    )
}

/// `SplitFormSWE2D::line_volume` with a bed: the flux-differencing volume
/// term (times J) along the line of nodes `idx(0..N1)`.
#[device]
fn line_volume<const AFFINE: bool, I: Fn(usize) -> usize>(
    nodes: &[SWENodeState2D; NN],
    rhs: &mut [SWEState2D; NN],
    metric: &[(f64, f64); N1],
    idx: I,
    affine_metric: (f64, f64),
    d1: &[f64],
    g: f64,
) {
    #[unroll]
    for a in 0..N1 {
        let ia = idx(a);
        let q_a = nodes[ia];
        let m_a = if AFFINE { affine_metric } else { metric[a] };

        let mut acc = (2.0 * d1[a * N1 + a]) * wintermeyer_flux_2d(&q_a, &q_a, m_a, g);
        // c = a + 1..N1, as a constant trip count so that it unrolls
        #[unroll]
        for c in 0..N1 {
            if c <= a {
                continue;
            }
            let ic = idx(c);
            let m = if AFFINE {
                m_a
            } else {
                let m_c = metric[c];
                (0.5 * (m_a.0 + m_c.0), 0.5 * (m_a.1 + m_c.1))
            };
            let f = wintermeyer_flux_2d(&q_a, &nodes[ic], m, g);
            acc = acc + (2.0 * d1[a * N1 + c]) * f;
            rhs[ic] = rhs[ic] - (2.0 * d1[c * N1 + a]) * f;
        }

        let (mut db, mut db_x, mut db_y) = (0.0, 0.0, 0.0);
        #[unroll]
        for c in 0..N1 {
            let d_b = d1[a * N1 + c] * nodes[idx(c)].b;
            db += d_b;
            if !AFFINE {
                db_x += d_b * metric[c].0;
                db_y += d_b * metric[c].1;
            }
        }
        let (db_x, db_y) = if AFFINE {
            (db * m_a.0, db * m_a.1)
        } else {
            (0.5 * (db * m_a.0 + db_x), 0.5 * (db * m_a.1 + db_y))
        };
        let force = g * q_a.h;
        acc = acc + SWEState2D::new(0.0, force * db_x, force * db_y);

        rhs[ia] = rhs[ia] - acc;
    }
}

/// `(∇r, ∇s, |∇r|, |∇s|)` of a node from `[r_x, r_y, s_x, s_y]`, as
/// `element_max_reference_rate` forms it.
#[inline]
fn rate_metric(grads: [f64; 4]) -> ((f64, f64), (f64, f64), f64, f64) {
    let norm_r = (grads[0] * grads[0] + grads[1] * grads[1]).sqrt();
    let norm_s = (grads[2] * grads[2] + grads[3] * grads[3]).sqrt();
    ((grads[0], grads[1]), (grads[2], grads[3]), norm_r, norm_s)
}

// =============================================================================
// Kernels
// =============================================================================

#[cuda_module]
pub mod kernels {
    use super::*;

    /// `SplitFormSWE2D::edge_fluxes`: F* of both sides of interior edge `e`
    /// (one thread per edge), left side first, each in its own face-node
    /// order.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn edge_fluxes(
        g: f64,
        h_min: f64,
        h: &[f64],
        hu: &[f64],
        hv: &[f64],
        bed: &[f64],
        affine: &[u32],
        egeom: &[[f64; EG]],
        fgeom: &[[f64; 3]],
        edges: &[[u32; 4]],
        mut out: DisjointSlice<[[f64; 3]; NF]>,
    ) {
        let idx = thread::index_1d();
        let e = idx.get();
        if let Some(slot) = out.get_mut(idx) {
            let [le, lf, re, rf] = edges[e];
            let (le, lf, re, rf) = (le as usize, lf as usize, re as usize, rf as usize);
            #[unroll]
            for fi in 0..N1 {
                let normal = if affine[le] != 0 {
                    (egeom[le][5 + 3 * lf], egeom[le][6 + 3 * lf])
                } else {
                    let f = fgeom[(le * 4 + lf) * N1 + fi];
                    (f[0], f[1])
                };
                let rfi = N1 - 1 - fi;
                let left = le * NN + face_node(lf, fi);
                let right = re * NN + face_node(rf, rfi);
                let (f_left, f_right) = surface_flux(
                    &node_state(h, hu, hv, bed, left, h_min),
                    &node_state(h, hu, hv, bed, right, h_min),
                    normal,
                    g,
                );
                slot[fi] = [f_left.h, f_left.hu, f_left.hv];
                slot[N1 + rfi] = [f_right.h, f_right.hu, f_right.hv];
            }
        }
    }

    /// `SplitFormSWE2D::element_rhs` of a fully wet element (one thread per
    /// element): the flux-differencing volume term, the lifted face terms
    /// with F* from `faces`, divided by J.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn element_rhs(
        g: f64,
        h_min: f64,
        h: &[f64],
        hu: &[f64],
        hv: &[f64],
        bed: &[f64],
        affine: &[u32],
        egeom: &[[f64; EG]],
        contra: &[[f64; 4]],
        jinv: &[f64],
        fgeom: &[[f64; 3]],
        d1: &[f64],
        lift: &[f64],
        element_edges: &[[u32; 4]],
        edges: &[[u32; 4]],
        faces: &[[[f64; 3]; NF]],
        mut out_h: DisjointSlice<f64>,
        mut out_hu: DisjointSlice<f64>,
        mut out_hv: DisjointSlice<f64>,
    ) {
        let k = thread::index_1d().get();
        if k >= element_edges.len() {
            return;
        }
        let mut nodes = [SWENodeState2D::default(); NN];
        #[unroll]
        for i in 0..NN {
            nodes[i] = node_state(h, hu, hv, bed, k * NN + i, h_min);
        }
        let mut rhs = [SWEState2D::zero(); NN];

        let aff = affine[k] != 0;
        let eg = egeom[k];
        let (affine_r, affine_s) = ((eg[0], eg[1]), (eg[2], eg[3]));
        let mut metric = [(0.0, 0.0); N1];
        #[unroll]
        for line in 0..N1 {
            let r_line = |a: usize| line * N1 + a;
            let s_line = |j: usize| j * N1 + line;
            #[unroll]
            for a in 0..N1 {
                metric[a] = if aff {
                    affine_r
                } else {
                    let c = contra[k * NN + r_line(a)];
                    (c[0], c[1])
                };
            }
            if aff {
                line_volume::<true, _>(&nodes, &mut rhs, &metric, r_line, affine_r, d1, g);
            } else {
                line_volume::<false, _>(&nodes, &mut rhs, &metric, r_line, affine_r, d1, g);
            }
            #[unroll]
            for j in 0..N1 {
                metric[j] = if aff {
                    affine_s
                } else {
                    let c = contra[k * NN + s_line(j)];
                    (c[2], c[3])
                };
            }
            if aff {
                line_volume::<true, _>(&nodes, &mut rhs, &metric, s_line, affine_s, d1, g);
            } else {
                line_volume::<false, _>(&nodes, &mut rhs, &metric, s_line, affine_s, d1, g);
            }
        }

        #[unroll]
        for face in 0..4 {
            let e = element_edges[k][face] as usize;
            let edge = edges[e];
            let side = usize::from(edge[0] as usize != k || edge[1] as usize != face);
            #[unroll]
            for fi in 0..N1 {
                let node = face_node(face, fi);
                let (normal, scale) = if aff {
                    ((eg[5 + 3 * face], eg[6 + 3 * face]), eg[7 + 3 * face])
                } else {
                    let f = fgeom[(k * 4 + face) * N1 + fi];
                    ((f[0], f[1]), f[2])
                };
                let q_int = nodes[node];
                let s = faces[e][side * N1 + fi];
                let f_star = SWEState2D::new(s[0], s[1], s[2]);
                let flux_diff = wintermeyer_flux_2d(&q_int, &q_int, normal, g) - f_star;
                rhs[node] = rhs[node] + (scale * lift[(face * NN + node) * N1 + fi]) * flux_diff;
            }
        }

        #[unroll]
        for i in 0..NN {
            let r = rhs[i];
            let j_inv = if aff { eg[4] } else { jinv[k * NN + i] };
            let n = k * NN + i;
            // SAFETY: thread k writes only its own element's rows
            // (k·NN..(k + 1)·NN), and k < n_elements.
            unsafe {
                *out_h.get_unchecked_mut(n) = j_inv * r.h;
                *out_hu.get_unchecked_mut(n) = j_inv * r.hu;
                *out_hv.get_unchecked_mut(n) = j_inv * r.hv;
            }
        }
    }

    /// `Integrable::combine` with a base: `out = a·base + c₁x₁ (+ c₂x₂)`,
    /// in the order of `combine_values` (copy, scale unless `a = 1`, one
    /// multiply–add per term).
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn combine_from(
        a: f64,
        terms: u32,
        c1: f64,
        c2: f64,
        base: &[f64],
        x1: &[f64],
        x2: &[f64],
        mut out: DisjointSlice<f64>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(v) = out.get_mut(idx) {
            let mut value = base[i];
            if a != 1.0 {
                value *= a;
            }
            if terms > 0 {
                value += c1 * x1[i];
            }
            if terms > 1 {
                value += c2 * x2[i];
            }
            *v = value;
        }
    }

    /// `Integrable::combine` without a base: `out = a·out + c₁x₁ (+ c₂x₂)`.
    #[kernel]
    pub fn combine_in_place(
        a: f64,
        terms: u32,
        c1: f64,
        c2: f64,
        x1: &[f64],
        x2: &[f64],
        mut out: DisjointSlice<f64>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(v) = out.get_mut(idx) {
            let mut value = *v;
            if a != 1.0 {
                value *= a;
            }
            if terms > 0 {
                value += c1 * x1[i];
            }
            if terms > 1 {
                value += c2 * x2[i];
            }
            *v = value;
        }
    }

    /// `element_max_reference_rate`: the largest `node_reference_rate` of
    /// element `k`'s wet nodes (zero at the dry ones), one thread per element.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn element_rates(
        g: f64,
        h_min: f64,
        h: &[f64],
        hu: &[f64],
        hv: &[f64],
        affine: &[u32],
        affine_grads: &[[f64; 4]],
        grads: &[[f64; 4]],
        mut out: DisjointSlice<f64>,
    ) {
        let idx = thread::index_1d();
        let k = idx.get();
        if let Some(rate) = out.get_mut(idx) {
            let aff = affine[k] != 0;
            let affine_metric = rate_metric(affine_grads[k]);
            let mut max = 0.0f64;
            #[unroll]
            for i in 0..NN {
                let n = k * NN + i;
                let state = SWEState2D::new(h[n], hu[n], hv[n]);
                let r = if state.h <= h_min {
                    0.0
                } else {
                    let metric = if aff {
                        affine_metric
                    } else {
                        rate_metric(grads[n])
                    };
                    node_reference_rate(&state, metric, g, Depth::new(h_min))
                };
                max = max.max(r);
            }
            *rate = max;
        }
    }

    /// 1 where element `k` has a value that is not finite, else 0.
    #[kernel]
    pub fn element_nonfinite(h: &[f64], hu: &[f64], hv: &[f64], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let k = idx.get();
        if let Some(flag) = out.get_mut(idx) {
            let mut bad = 0u32;
            #[unroll]
            for i in 0..NN {
                let n = k * NN + i;
                if !(h[n].is_finite() && hu[n].is_finite() && hv[n].is_finite()) {
                    bad = 1;
                }
            }
            *flag = bad;
        }
    }
}

// =============================================================================
// Host: the device, the static data, the state
// =============================================================================

/// The GPU: the stream every operation is ordered on (it holds the context),
/// and the kernels.
pub struct Gpu {
    pub stream: Arc<Stream>,
    module: kernels::LoadedModule,
}

impl Gpu {
    /// Device 0, its default stream and the kernels.
    pub fn new() -> Result<Arc<Self>, Box<dyn std::error::Error>> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let module = kernels::load(&ctx)?;
        Ok(Arc::new(Self { stream, module }))
    }
}

fn one_dimensional(n: usize) -> LaunchConfig {
    LaunchConfig::for_num_elems(n as u32)
}

fn per_element(n_elements: usize) -> LaunchConfig {
    let n = n_elements as u32;
    LaunchConfig {
        grid_dim: (n.div_ceil(ELEMENT_BLOCK), 1, 1),
        block_dim: (ELEMENT_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// The SWE state `[h, hu, hv]` on the device (`[k·NN + i]` per field, as
/// `SWESolution2D`). Every operation is enqueued on the GPU's stream.
pub struct DeviceState {
    pub h: DeviceBuffer<f64>,
    pub hu: DeviceBuffer<f64>,
    pub hv: DeviceBuffer<f64>,
    gpu: Arc<Gpu>,
}

impl DeviceState {
    /// `q` on the device.
    pub fn upload(gpu: &Arc<Gpu>, q: &SWESolution2D) -> Self {
        assert_eq!(q.n_nodes, NN, "the kernels are compiled for P{ORDER}");
        let up = |field: &[f64]| DeviceBuffer::from_host(&gpu.stream, field).expect("upload");
        Self {
            h: up(&q.data[0]),
            hu: up(&q.data[1]),
            hv: up(&q.data[2]),
            gpu: gpu.clone(),
        }
    }

    /// The state on the host.
    pub fn download(&self) -> SWESolution2D {
        let n_elements = self.h.len() / NN;
        let mut q = SWESolution2D::new(n_elements, NN);
        for (field, buffer) in q.data.iter_mut().zip([&self.h, &self.hu, &self.hv]) {
            buffer
                .copy_to_host(&self.gpu.stream, field)
                .expect("download");
        }
        q
    }

    fn fields(&self) -> [&DeviceBuffer<f64>; 3] {
        [&self.h, &self.hu, &self.hv]
    }

    fn fields_mut(&mut self) -> [&mut DeviceBuffer<f64>; 3] {
        [&mut self.h, &mut self.hu, &mut self.hv]
    }

    fn zeroed_like(&self) -> Self {
        let zero = || DeviceBuffer::zeroed(&self.gpu.stream, self.h.len()).expect("allocation");
        Self {
            h: zero(),
            hu: zero(),
            hv: zero(),
            gpu: self.gpu.clone(),
        }
    }

    /// `combine` of up to two terms, field by field, on the device.
    fn combine_terms(&mut self, base: Option<&Self>, a: f64, terms: &[(f64, &Self)]) {
        assert!(terms.len() <= 2, "the device combines at most two terms");
        if base.is_none() && terms.is_empty() && a == 1.0 {
            return;
        }
        let gpu = self.gpu.clone();
        let cfg = one_dimensional(self.h.len());
        let n_terms = terms.len() as u32;
        let c = |j: usize| terms.get(j).map_or(0.0, |t| t.0);
        // The kernels read a term's slot only if it is used
        let empty = DeviceBuffer::<f64>::zeroed(&gpu.stream, 0).expect("allocation");
        for f in 0..3 {
            let x = |j: usize| terms.get(j).map_or(&empty, |t| t.1.fields()[f]);
            let out = match f {
                0 => &mut self.h,
                1 => &mut self.hu,
                _ => &mut self.hv,
            };
            // SAFETY: every buffer is the state's length (or unused); `out`
            // is none of the inputs (`&mut self` against `&Self`).
            unsafe {
                match base {
                    Some(base) => gpu.module.combine_from(
                        &gpu.stream,
                        cfg,
                        a,
                        n_terms,
                        c(0),
                        c(1),
                        base.fields()[f],
                        x(0),
                        x(1),
                        out,
                    ),
                    None => gpu.module.combine_in_place(
                        &gpu.stream,
                        cfg,
                        a,
                        n_terms,
                        c(0),
                        c(1),
                        x(0),
                        x(1),
                        out,
                    ),
                }
            }
            .expect("combine");
        }
    }
}

impl Clone for DeviceState {
    fn clone(&self) -> Self {
        let mut copy = self.zeroed_like();
        copy.copy_from(self);
        copy
    }
}

impl Integrable for DeviceState {
    fn scale(&mut self, c: f64) {
        self.combine_terms(None, c, &[]);
    }

    fn axpy(&mut self, c: f64, other: &Self) {
        self.combine_terms(None, 1.0, &[(c, other)]);
    }

    fn zeros_like(&self) -> Self {
        self.zeroed_like()
    }

    fn copy_from(&mut self, other: &Self) {
        let stream = self.gpu.stream.clone();
        for (to, from) in self.fields_mut().into_iter().zip(other.fields()) {
            to.copy_from_device_async(from, &stream)
                .expect("device copy");
        }
    }

    fn combine(&mut self, base: Option<&Self>, a: f64, terms: &[(f64, &Self)]) {
        if terms.len() <= 2 {
            self.combine_terms(base, a, terms);
            return;
        }
        // The plain sequence (the SSP schemes never need it)
        if let Some(base) = base {
            self.copy_from(base);
        }
        if a != 1.0 {
            self.scale(a);
        }
        for &(c, x) in terms {
            self.axpy(c, x);
        }
    }

    fn is_finite(&self) -> bool {
        let n_elements = self.h.len() / NN;
        let gpu = &self.gpu;
        let mut flags = DeviceBuffer::<u32>::zeroed(&gpu.stream, n_elements).expect("allocation");
        // SAFETY: one flag per element, the state's length.
        unsafe {
            gpu.module.element_nonfinite(
                &gpu.stream,
                one_dimensional(n_elements),
                &self.h,
                &self.hu,
                &self.hv,
                &mut flags,
            )
        }
        .expect("is_finite");
        let flags = flags.to_host_vec(&gpu.stream).expect("download");
        flags.iter().all(|&f| f == 0)
    }
}

/// The static data of a mesh, packed as the kernels read them, on the device.
struct Statics {
    bed: DeviceBuffer<f64>,
    affine: DeviceBuffer<u32>,
    egeom: DeviceBuffer<[f64; EG]>,
    contra: DeviceBuffer<[f64; 4]>,
    jinv: DeviceBuffer<f64>,
    fgeom: DeviceBuffer<[f64; 3]>,
    d1: DeviceBuffer<f64>,
    lift: DeviceBuffer<f64>,
    element_edges: DeviceBuffer<[u32; 4]>,
    edges: DeviceBuffer<[u32; 4]>,
    /// `[r_x, r_y, s_x, s_y]` of every node, and of every parallelogram
    grads: DeviceBuffer<[f64; 4]>,
    affine_grads: DeviceBuffer<[f64; 4]>,
}

impl Statics {
    fn upload(
        gpu: &Gpu,
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        geom: &GeometricFactors2D,
        bathymetry: &Bathymetry2D,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let stream = &gpu.stream;
        let n_el = mesh.n_elements;
        assert_eq!(ops.n_1d, N1, "the kernels are compiled for P{ORDER}");
        assert!(
            mesh.edges.iter().all(|e| e.right.is_some()),
            "the kernels take interior faces only so far (a periodic mesh)"
        );
        for (face, nodes) in ops.face_nodes.iter().enumerate() {
            for (fi, &node) in nodes.iter().enumerate() {
                assert_eq!(face_node(face, fi), node, "face node {fi} of face {face}");
            }
        }

        let mut affine = Vec::with_capacity(n_el);
        let mut egeom = Vec::with_capacity(n_el);
        let mut affine_grads = Vec::with_capacity(n_el);
        let mut contra = Vec::with_capacity(n_el * NN);
        let mut grads = Vec::with_capacity(n_el * NN);
        let mut jinv = Vec::with_capacity(n_el * NN);
        let mut fgeom = Vec::with_capacity(n_el * 4 * N1);
        let mut bed = Vec::with_capacity(n_el * NN);
        for k in 0..n_el {
            let element = geom.element_geometry(k);
            affine.push(u32::from(element.affine));
            let (r, s) = element.metric.contravariant();
            let mut eg = [0.0; EG];
            eg[..5].copy_from_slice(&[r.0, r.1, s.0, s.1, element.metric.det_j_inv]);
            for (face, &((nx, ny), sj)) in element.faces.iter().enumerate() {
                eg[5 + 3 * face..8 + 3 * face].copy_from_slice(&[nx, ny, sj]);
            }
            egeom.push(eg);
            let m = &element.metric;
            affine_grads.push([m.rx, m.ry, m.sx, m.sy]);
            for i in 0..NN {
                let (r, s) = geom.contravariant(k, i);
                contra.push([r.0, r.1, s.0, s.1]);
                let ((rx, ry), (sx, sy)) = (geom.grad_r(k, i), geom.grad_s(k, i));
                grads.push([rx, ry, sx, sy]);
                jinv.push(geom.jacobian_inv(k, i));
                bed.push(bathymetry.get(ElementIndex::new(k), i));
            }
            for face in 0..4 {
                for fi in 0..N1 {
                    let (nx, ny) = geom.normal(k, face, fi);
                    fgeom.push([nx, ny, geom.surface_jacobian(k, face, fi)]);
                }
            }
        }
        let lift: Vec<f64> = ops.lift_row_major.iter().flatten().copied().collect();
        let element_edges: Vec<[u32; 4]> = mesh
            .element_edges
            .iter()
            .map(|e| e.map(|e| e as u32))
            .collect();
        let edges: Vec<[u32; 4]> = mesh
            .edges
            .iter()
            .map(|e| {
                let right = e.right.expect("interior edge");
                [
                    e.left.element as u32,
                    e.left.face as u32,
                    right.element as u32,
                    right.face as u32,
                ]
            })
            .collect();
        Ok(Self {
            bed: DeviceBuffer::from_host(stream, &bed)?,
            affine: DeviceBuffer::from_host(stream, &affine)?,
            egeom: DeviceBuffer::from_host(stream, &egeom)?,
            contra: DeviceBuffer::from_host(stream, &contra)?,
            jinv: DeviceBuffer::from_host(stream, &jinv)?,
            fgeom: DeviceBuffer::from_host(stream, &fgeom)?,
            d1: DeviceBuffer::from_host(stream, &ops.dr_1d_row_major)?,
            lift: DeviceBuffer::from_host(stream, &lift)?,
            element_edges: DeviceBuffer::from_host(stream, &element_edges)?,
            edges: DeviceBuffer::from_host(stream, &edges)?,
            grads: DeviceBuffer::from_host(stream, &grads)?,
            affine_grads: DeviceBuffer::from_host(stream, &affine_grads)?,
        })
    }
}

/// The 2D SWE physics module on the GPU: `SWEPhysics2D` with
/// `SWEFormulation2D::EntropyStable`, a bed, no boundary faces and no sources
/// (Phase 1's first scope, `notes/gpu.md`).
pub struct GpuSWE2D {
    gpu: Arc<Gpu>,
    mesh: Arc<Mesh2D>,
    ops: Arc<DGOperators2D>,
    geom: Arc<GeometricFactors2D>,
    statics: Statics,
    g: f64,
    h_min: f64,
    /// Both sides' F* of every edge, written by the edge kernel for the
    /// element kernel
    faces: Mutex<DeviceBuffer<[[f64; 3]; NF]>>,
    /// Every element's largest reference rate (the time step)
    rates: Mutex<DeviceBuffer<f64>>,
}

impl GpuSWE2D {
    pub fn new(
        gpu: &Arc<Gpu>,
        mesh: Arc<Mesh2D>,
        ops: Arc<DGOperators2D>,
        geom: Arc<GeometricFactors2D>,
        bathymetry: &Bathymetry2D,
        equation: &ShallowWater2D,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let statics = Statics::upload(gpu, &mesh, &ops, &geom, bathymetry)?;
        let faces = DeviceBuffer::zeroed(&gpu.stream, mesh.n_edges)?;
        let rates = DeviceBuffer::zeroed(&gpu.stream, mesh.n_elements)?;
        Ok(Self {
            gpu: gpu.clone(),
            mesh,
            ops,
            geom,
            statics,
            g: equation.g,
            h_min: equation.h_min.meters(),
            faces: Mutex::new(faces),
            rates: Mutex::new(rates),
        })
    }
}

impl PhysicsModuleInfo for GpuSWE2D {
    fn name(&self) -> &'static str {
        "swe-2d-gpu"
    }

    fn description(&self) -> &str {
        "2D Shallow Water Equations on the GPU (fully wet split form)"
    }

    fn n_variables(&self) -> usize {
        3
    }

    fn variable_names(&self) -> &[&'static str] {
        &["h", "hu", "hv"]
    }
}

impl PhysicsModule<DeviceState> for GpuSWE2D {
    fn compute_rhs(&self, state: &DeviceState, time: f64) -> DeviceState {
        let mut out = state.zeroed_like();
        self.compute_rhs_into(state, time, &mut out);
        out
    }

    fn compute_rhs_into(&self, state: &DeviceState, _time: f64, out: &mut DeviceState) {
        let (gpu, s) = (&self.gpu, &self.statics);
        let mut faces = self.faces.lock().expect("face buffer");
        let [out_h, out_hu, out_hv] = out.fields_mut();
        // SAFETY: every buffer is sized for this mesh (`Statics::upload`, the
        // state's), and the kernels index within its elements, edges and nodes.
        unsafe {
            gpu.module
                .edge_fluxes(
                    &gpu.stream,
                    one_dimensional(self.mesh.n_edges),
                    self.g,
                    self.h_min,
                    &state.h,
                    &state.hu,
                    &state.hv,
                    &s.bed,
                    &s.affine,
                    &s.egeom,
                    &s.fgeom,
                    &s.edges,
                    &mut *faces,
                )
                .expect("edge fluxes");
            gpu.module
                .element_rhs(
                    &gpu.stream,
                    per_element(self.mesh.n_elements),
                    self.g,
                    self.h_min,
                    &state.h,
                    &state.hu,
                    &state.hv,
                    &s.bed,
                    &s.affine,
                    &s.egeom,
                    &s.contra,
                    &s.jinv,
                    &s.fgeom,
                    &s.d1,
                    &s.lift,
                    &s.element_edges,
                    &s.edges,
                    &*faces,
                    out_h,
                    out_hu,
                    out_hv,
                )
                .expect("element RHS");
        }
    }

    /// `compute_dt_swe_2d`: the largest element rate on the device, reduced
    /// on the host (a max is exact in any order).
    fn compute_dt(&self, state: &DeviceState, cfl: f64) -> f64 {
        let (gpu, s) = (&self.gpu, &self.statics);
        let mut rates = self.rates.lock().expect("rate buffer");
        // SAFETY: one rate per element; the state and statics are this mesh's.
        unsafe {
            gpu.module.element_rates(
                &gpu.stream,
                one_dimensional(self.mesh.n_elements),
                self.g,
                self.h_min,
                &state.h,
                &state.hu,
                &state.hv,
                &s.affine,
                &s.affine_grads,
                &s.grads,
                &mut *rates,
            )
        }
        .expect("element rates");
        let rates = rates.to_host_vec(&gpu.stream).expect("download");
        let max_rate = rates.iter().copied().fold(0.0_f64, f64::max);
        dt_from_reference_rate(max_rate, self.ops.order, cfl)
    }

    fn mesh(&self) -> &Mesh2D {
        &self.mesh
    }

    fn operators(&self) -> &DGOperators2D {
        &self.ops
    }

    fn geometry(&self) -> &GeometricFactors2D {
        &self.geom
    }

    fn order(&self) -> usize {
        self.ops.order
    }
}

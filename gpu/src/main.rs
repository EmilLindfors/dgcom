//! Phase 0 of the GPU port (P2.7, `notes/gpu.md`): the fully wet split-form
//! 2D RHS (`SWEFormulation2D::EntropyStable`) on the GPU, calling dgcom's own
//! pointwise physics, against `compute_rhs_swe_2d_parallel_into` on the CPU.
//!
//! The kernels follow `SplitFormSWE2D` (`src/solver/rhs/swe_2d_split_form.rs`)
//! operation for operation: `edge_fluxes` for the interior faces, then
//! `element_rhs` without subcells (every node wet). Built with `--no-fmad`
//! (no multiply–add contraction, as the CPU) the RHS is the CPU's bit for bit;
//! that is the gate. The default build fuses multiply–adds: within a
//! tolerance, and the timing.
//!
//! ```text
//! cargo oxide run --no-fmad      # the gate
//! cargo oxide run                # the timing
//! ```
//!
//! One thread per edge and one per element. Every loop unrolls (P2's bounds
//! are constants and the face nodes a formula), so an element's nodes and
//! RHS stay in registers: the PTX has no local memory.

use std::time::Instant;

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, device, kernel, thread};
use cuda_host::cuda_module;
use dgcom::boundary::Reflective2D;
use dgcom::equations::ShallowWater2D;
use dgcom::flux::{
    SWENodeState2D, entropy_stable_dissipation_2d, wintermeyer_bed_interface_term_2d,
    wintermeyer_flux_2d,
};
use dgcom::mesh::{Bathymetry2D, BoundaryTag, ElementFace, Mesh2D};
use dgcom::operators::{DGOperators2D, GeometricFactors2D};
use dgcom::solver::{SWE2DRhsConfig, SWEFormulation2D, SWESolution2D};
use dgcom::types::ElementIndex;
use dgcom::{SWEState2D, compute_rhs_swe_2d_parallel_into};

/// Polynomial order of the kernels (P2).
const ORDER: usize = 2;
/// Nodes per GLL line.
const N1: usize = ORDER + 1;
/// Nodes per element.
const NN: usize = N1 * N1;
/// Face-buffer slots per edge: the left side's `N1`, then the right side's.
const NF: usize = 2 * N1;
/// Per-element geometry of a parallelogram: `J∇r` (2), `J∇s` (2), `1/J`,
/// then each face's outward normal (2) and surface Jacobian.
const EG: usize = 17;

const G: f64 = 9.81;

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
// (shared with it from Phase 1, notes/gpu.md)
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
/// term (times J) along the line of nodes `idx(0..N1)`. Its loops unroll
/// (P2's bounds are constants), so the element's nodes stay in registers.
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

// =============================================================================
// Kernels
// =============================================================================

#[cuda_module]
mod kernels {
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
}

// =============================================================================
// Host: the mesh, the state, the device copies
// =============================================================================

/// Deterministic pseudo-random numbers in [0, 1) (64-bit LCG).
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Element size (m).
const DX: f64 = 100.0;

/// A doubly periodic `nx × ny` mesh of 100 m quadrilaterals, its interior
/// vertices moved by up to `jitter · DX` (0: parallelograms throughout).
fn periodic_mesh(nx: usize, ny: usize, jitter: f64) -> Mesh2D {
    let mut rng = Lcg(2026_11_15);
    let mut vertices = Vec::with_capacity((nx + 1) * (ny + 1));
    for j in 0..=ny {
        for i in 0..=nx {
            let (mut x, mut y) = (i as f64 * DX, j as f64 * DX);
            if i > 0 && i < nx && j > 0 && j < ny {
                x += jitter * DX * (2.0 * rng.next() - 1.0);
                y += jitter * DX * (2.0 * rng.next() - 1.0);
            }
            vertices.push([x, y]);
        }
    }
    let elements = (0..ny)
        .flat_map(|j| {
            (0..nx).map(move |i| {
                let v0 = j * (nx + 1) + i;
                [v0, v0 + 1, v0 + nx + 2, v0 + nx + 1]
            })
        })
        .collect();
    let periodic: Vec<_> = (0..ny)
        .map(|j| {
            (
                ElementFace::new(j * nx + nx - 1, 1),
                ElementFace::new(j * nx, 3),
            )
        })
        .chain((0..nx).map(|i| {
            (
                ElementFace::new((ny - 1) * nx + i, 2),
                ElementFace::new(i, 0),
            )
        }))
        .collect();
    Mesh2D::from_quads(vertices, elements, &periodic, |_| BoundaryTag::Wall)
        .expect("a structured grid is a valid quad mesh")
}

/// A mesh with its operators, geometry, a smooth periodic bed 12–28 m deep
/// and a smooth wet state on it.
struct Case {
    mesh: Mesh2D,
    ops: DGOperators2D,
    geom: GeometricFactors2D,
    bathymetry: Bathymetry2D,
    q: SWESolution2D,
}

impl Case {
    fn new(nx: usize, ny: usize, jitter: f64) -> Self {
        let mesh = periodic_mesh(nx, ny, jitter);
        let ops = DGOperators2D::new(ORDER);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let (kx, ky) = (
            std::f64::consts::TAU / (nx as f64 * DX),
            std::f64::consts::TAU / (ny as f64 * DX),
        );
        let bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, y| {
            -20.0 - 8.0 * (kx * x).sin() * (ky * y).cos()
        });
        let mut q = SWESolution2D::new(mesh.n_elements, NN);
        for k in 0..mesh.n_elements {
            let ki = ElementIndex::new(k);
            for i in 0..NN {
                let [x, y] = mesh.reference_to_physical(ki, ops.nodes_r[i], ops.nodes_s[i]);
                let eta = 0.3 * (2.0 * kx * x).cos() * (ky * y).sin();
                let h = eta - bathymetry.get(ki, i);
                let (u, v) = (0.6 * (ky * y).sin(), 0.4 * (kx * x).cos());
                q.set_state(ki, i, SWEState2D::new(h, h * u, h * v));
            }
        }
        Self {
            mesh,
            ops,
            geom,
            bathymetry,
            q,
        }
    }

    fn n_elements(&self) -> usize {
        self.mesh.n_elements
    }

    fn affine_elements(&self) -> usize {
        (0..self.n_elements())
            .filter(|&k| self.geom.element_geometry(k).affine)
            .count()
    }
}

/// The case's static data and state, packed as the kernels read them, on
/// the device.
struct Device {
    h: DeviceBuffer<f64>,
    hu: DeviceBuffer<f64>,
    hv: DeviceBuffer<f64>,
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
    faces: DeviceBuffer<[[f64; 3]; NF]>,
    out: [DeviceBuffer<f64>; 3],
}

impl Device {
    fn upload(case: &Case, stream: &cuda_core::Stream) -> Result<Self, Box<dyn std::error::Error>> {
        let (mesh, ops, geom) = (&case.mesh, &case.ops, &case.geom);
        let n_el = mesh.n_elements;
        assert_eq!(ops.n_1d, N1, "the kernels are compiled for P{ORDER}");
        assert!(
            mesh.edges.iter().all(|e| e.right.is_some()),
            "Phase 0 kernels take interior faces only (a periodic mesh)"
        );

        let mut affine = Vec::with_capacity(n_el);
        let mut egeom = Vec::with_capacity(n_el);
        let mut contra = Vec::with_capacity(n_el * NN);
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
            for i in 0..NN {
                let (r, s) = geom.contravariant(k, i);
                contra.push([r.0, r.1, s.0, s.1]);
                jinv.push(geom.jacobian_inv(k, i));
                bed.push(case.bathymetry.get(ElementIndex::new(k), i));
            }
            for face in 0..4 {
                for fi in 0..N1 {
                    let (nx, ny) = geom.normal(k, face, fi);
                    fgeom.push([nx, ny, geom.surface_jacobian(k, face, fi)]);
                }
            }
        }
        let lift: Vec<f64> = ops.lift_row_major.iter().flatten().copied().collect();
        for (face, nodes) in ops.face_nodes.iter().enumerate() {
            for (fi, &node) in nodes.iter().enumerate() {
                assert_eq!(face_node(face, fi), node, "face node {fi} of face {face}");
            }
        }
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

        let n_nodes = n_el * NN;
        Ok(Self {
            h: DeviceBuffer::from_host(stream, &case.q.data[0])?,
            hu: DeviceBuffer::from_host(stream, &case.q.data[1])?,
            hv: DeviceBuffer::from_host(stream, &case.q.data[2])?,
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
            faces: DeviceBuffer::zeroed(stream, mesh.n_edges)?,
            out: [
                DeviceBuffer::zeroed(stream, n_nodes)?,
                DeviceBuffer::zeroed(stream, n_nodes)?,
                DeviceBuffer::zeroed(stream, n_nodes)?,
            ],
        })
    }
}

/// Threads per block of the element kernel (its registers bound the block).
const ELEMENT_BLOCK: u32 = 128;

/// One RHS evaluation on the device: the edge kernel, then the element kernel.
fn gpu_rhs(
    module: &kernels::LoadedModule,
    stream: &cuda_core::Stream,
    dev: &mut Device,
    g: f64,
    h_min: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let n_edges = dev.edges.len() as u32;
    let n_el = dev.element_edges.len() as u32;
    let element_cfg = LaunchConfig {
        grid_dim: (n_el.div_ceil(ELEMENT_BLOCK), 1, 1),
        block_dim: (ELEMENT_BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let [out_h, out_hu, out_hv] = &mut dev.out;
    // SAFETY: every buffer is sized for this case (`Device::upload`), and the
    // kernels index within the mesh's elements, edges and nodes.
    unsafe {
        module.edge_fluxes(
            stream,
            LaunchConfig::for_num_elems(n_edges),
            g,
            h_min,
            &dev.h,
            &dev.hu,
            &dev.hv,
            &dev.bed,
            &dev.affine,
            &dev.egeom,
            &dev.fgeom,
            &dev.edges,
            &mut dev.faces,
        )?;
        module.element_rhs(
            stream,
            element_cfg,
            g,
            h_min,
            &dev.h,
            &dev.hu,
            &dev.hv,
            &dev.bed,
            &dev.affine,
            &dev.egeom,
            &dev.contra,
            &dev.jinv,
            &dev.fgeom,
            &dev.d1,
            &dev.lift,
            &dev.element_edges,
            &dev.edges,
            &dev.faces,
            out_h,
            out_hu,
            out_hv,
        )?;
    }
    Ok(())
}

/// Distance in units in the last place between two finite f64.
fn ulps(a: f64, b: f64) -> u64 {
    let key = |x: f64| {
        let bits = x.to_bits() as i64;
        if bits < 0 { i64::MIN - bits } else { bits }
    };
    key(a).abs_diff(key(b))
}

/// Runs one case on both and reports agreement and, if `iterations > 0`,
/// the time per element evaluation.
fn run(
    label: &str,
    case: &Case,
    ctx: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    iterations: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let stream = ctx.default_stream();
    let equation = ShallowWater2D::new(G);
    let h_min = equation.h_min.meters();
    let wall = Reflective2D::new();
    let config = SWE2DRhsConfig::new(&equation, &wall)
        .with_formulation(SWEFormulation2D::EntropyStable)
        .with_bathymetry(&case.bathymetry);
    let n_el = case.n_elements();

    let mut cpu = SWESolution2D::new(n_el, NN);
    compute_rhs_swe_2d_parallel_into(
        &case.q, &case.mesh, &case.ops, &case.geom, &config, 0.0, &mut cpu,
    );

    let mut dev = Device::upload(case, &stream)?;
    gpu_rhs(module, &stream, &mut dev, G, h_min)?;
    let gpu: Vec<Vec<f64>> = dev
        .out
        .iter()
        .map(|b| b.to_host_vec(&stream))
        .collect::<Result<_, _>>()?;

    let n_nodes = n_el * NN;
    let identical = (0..n_nodes)
        .filter(|&n| (0..3).all(|v| cpu.data[v][n].to_bits() == gpu[v][n].to_bits()))
        .count();
    let (mut worst_ulps, mut worst_rel) = (0u64, 0.0f64);
    for v in 0..3 {
        let scale = cpu.data[v].iter().fold(0.0f64, |m, x| m.max(x.abs()));
        for n in 0..n_nodes {
            let (c, g) = (cpu.data[v][n], gpu[v][n]);
            assert!(
                g.is_finite(),
                "non-finite GPU RHS at node {n}, variable {v}"
            );
            worst_ulps = worst_ulps.max(ulps(c, g));
            worst_rel = worst_rel.max((c - g).abs() / scale);
        }
    }
    println!(
        "{label}: {n_el} P{ORDER} elements ({} affine), {identical} of {n_nodes} nodes \
         bit-identical to the {} CPU kernel, largest difference {worst_ulps} ulp, \
         {worst_rel:.1e} of the field's largest",
        case.affine_elements(),
        if cfg!(feature = "simd") {
            "batched (simd)"
        } else {
            "scalar"
        }
    );

    if iterations > 0 {
        for _ in 0..3 {
            compute_rhs_swe_2d_parallel_into(
                &case.q, &case.mesh, &case.ops, &case.geom, &config, 0.0, &mut cpu,
            );
        }
        let start = Instant::now();
        for _ in 0..iterations {
            compute_rhs_swe_2d_parallel_into(
                &case.q, &case.mesh, &case.ops, &case.geom, &config, 0.0, &mut cpu,
            );
        }
        let cpu_ns = start.elapsed().as_secs_f64() * 1e9 / (iterations * n_el) as f64;

        for _ in 0..3 {
            gpu_rhs(module, &stream, &mut dev, G, h_min)?;
        }
        stream.synchronize()?;
        let gpu_iterations = 5 * iterations;
        let start = Instant::now();
        for _ in 0..gpu_iterations {
            gpu_rhs(module, &stream, &mut dev, G, h_min)?;
        }
        stream.synchronize()?;
        let gpu_ns = start.elapsed().as_secs_f64() * 1e9 / (gpu_iterations * n_el) as f64;
        println!(
            "  per element evaluation: CPU {cpu_ns:.1} ns ({} threads), GPU {gpu_ns:.1} ns ({:.2}x)",
            rayon_threads(),
            cpu_ns / gpu_ns
        );
    }
    Ok(())
}

fn rayon_threads() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let module = kernels::load(&ctx)?;

    run("parallelograms", &Case::new(48, 40, 0.0), &ctx, &module, 0)?;
    run("jittered", &Case::new(48, 40, 0.15), &ctx, &module, 0)?;
    run(
        "jittered, timing",
        &Case::new(320, 300, 0.15),
        &ctx,
        &module,
        20,
    )?;
    run(
        "parallelograms, timing",
        &Case::new(320, 300, 0.0),
        &ctx,
        &module,
        20,
    )?;
    Ok(())
}

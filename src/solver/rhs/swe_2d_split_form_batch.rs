//! [`SplitFormSWE2D::element_rhs`] for [`VOLUME_BATCH`] elements at once,
//! one per SIMD lane (`fearless_simd`, `f64x8` at the CPU's best level): the
//! flux-differencing volume term of [`SplitFormSWE2D::line_volume`], the
//! shoreline subcells of [`SplitFormSWE2D::line_subcells`], the surface
//! terms with the interior-face fluxes of [`SplitFormSWE2D::edge_fluxes`]
//! (the hydrostatic HLL of `WetDry`), and the division by J.
//!
//! A tensor-product element's lines are 2–5 nodes long at P1–P4, too short to
//! fill a vector, but every element runs the same arithmetic: lanes are
//! elements. [`SplitFormSWE2D::chunk_rhs`] takes [`ELEMENT_CHUNK`] elements,
//! sorts them into the two kinds (a node shallower than the subcell depth or
//! not) so that both fill their lanes, and evaluates each kind a batch at a
//! time. The node states and bed of a batch are transposed into
//! `[node][lane]` vectors, the loops run on them in the scalar order, and
//! each element's terms go to [`SplitFormWorkspace::batch`], from where
//! [`SplitFormSWE2D::element_rhs`] copies them out. The subcells' branches
//! (incomplete stencils at boundaries, the limiter's extrema, the dry-node
//! rule for `η`, dry HLL sides, upwind wave speeds) become selects.
//!
//! **What a batch reads.** Arithmetic is the smaller part of an element's
//! cost: the kernel is bound by what it reads (`notes/performance.md`, P2.8,
//! 2026-11-12). So a batch reads, per element, its three rows of the state,
//! its bed, its four corners and its four face links, and its neighbours'
//! face nodes of the state and the bed:
//! - **Geometry from the corners.** Every element is a bilinear
//!   quadrilateral, and `GeometricFactors2D::compute` derives each per-node
//!   field from its four corners. A batch does the same in vectors, operation
//!   for operation ([`Corners::at`]): 64 bytes per element instead of the
//!   per-node metric, Jacobian, normals and surface Jacobians (≈ 1 kB at P2).
//! - **Face fluxes where they are used.** An interior face's flux is
//!   evaluated in the batch of each of its two elements, from the edge's left
//!   side to its right with the left side's normal, as
//!   [`SplitFormSWE2D::edge_fluxes`] does: both get the same flux to the bit
//!   and exchange exactly opposite fluxes. There is no face pass and no face
//!   buffer: a pass over the edges wrote every flux to memory for a second
//!   pass over the elements to read back, and read the states twice.
//! - **Connectivity in four bytes a face** ([`FaceLink`]), not through the
//!   edge list.
//!
//! **Bit for bit.** Every lane performs the scalar kernel's operations in the
//! scalar kernel's order (multiplies and adds, no fused multiply-add; `sqrt`
//! and division are correctly rounded in both), so each element's RHS is
//! identical to the per-element evaluation. Parallelogram and curved elements
//! share a batch: a parallelogram's lanes carry its constant contravariant
//! vector at every node, whose pair average `½(m + m) = m` is exact, and
//! where the two forms associate differently (the bed term, the subcell
//! interface metrics and balance term) both are computed and selected per
//! lane. `f64::max`/`min`/`signum` become compare-and-select, which agrees
//! with them except for NaN arguments and a tie of +0 with −0.
//!
//! Boundary faces take their flux from the scalar
//! [`SplitFormSWE2D::boundary_flux`]; the unused lanes of a short batch
//! repeat its first element and are dropped.

use fearless_simd::{Level, dispatch, f64x8, mask64x8, prelude::*};

use super::{ELEMENT_CHUNK, SplitFormSWE2D, SplitFormWorkspace, SurfaceFlux, VOLUME_BATCH};
use crate::boundary::SWEBoundaryCondition2D;
use crate::flux::SWENodeState2D;
use crate::mesh::FaceLink;
use crate::solver::SWEState2D;
use crate::solver::rhs::subcells::line_node;
use crate::types::ElementIndex;

const L: usize = VOLUME_BATCH;
const _: () = assert!(L == 8 && u8::BITS as usize == L);

#[cfg(test)]
thread_local! {
    /// Turns the batching off on this thread (the reference for the
    /// bit-for-bit gate)
    pub(super) static UNBATCHED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

impl<BC: SWEBoundaryCondition2D> SplitFormSWE2D<'_, '_, BC> {
    /// Whether [`Self::chunk_rhs`] evaluates the elements: P1–P4.
    pub(in crate::solver::rhs) fn batches(&self) -> bool {
        #[cfg(test)]
        if UNBATCHED.with(std::cell::Cell::get) {
            return false;
        }
        (2..=5).contains(&self.ops.n_1d)
    }

    /// Whether [`Self::chunk_rhs`] also evaluates the interior-face fluxes
    /// (the hydrostatic HLL of `WetDry`), so that no face pass
    /// ([`Self::edge_fluxes`]) is needed.
    pub(in crate::solver::rhs) fn fuses_faces(&self) -> bool {
        self.batches() && self.surface == SurfaceFlux::HydrostaticHll
    }

    /// [`Self::element_rhs`] (volume and surface terms) of the elements `ks`
    /// (at most [`ELEMENT_CHUNK`]) into `ws.batch[..][j · n_nodes..]` for
    /// `ks[j]`, the subcell interface mass fluxes of subcell elements into
    /// `ws.batch_mass[j · subcell_interfaces(n_1d)..]` and, if `face_mass`,
    /// the mass component of `F*` at the face nodes into
    /// `ws.batch_face_mass[j · 4 n_1d..]`. The elements are sorted into the
    /// flux-differencing and the subcell kind and each kind is evaluated
    /// [`VOLUME_BATCH`] at a time, one per lane. Bit `j` of the result is
    /// set if `ks[j]`'s terms are there: all of them for P1–P4, none
    /// otherwise.
    ///
    /// `faces`: the interior-face fluxes of [`Self::edge_fluxes`] (not read
    /// if [`Self::fuses_faces`]); `time_of(k)`: the time of element `k`'s
    /// boundary faces. `ahead`: the elements of the chunk this thread
    /// evaluates next, whose data is prefetched meanwhile
    /// ([`Self::prefetch`]).
    pub(in crate::solver::rhs) fn chunk_rhs(
        &self,
        ks: &[usize],
        ahead: &[u32],
        faces: &[SWEState2D],
        time_of: &dyn Fn(usize) -> f64,
        face_mass: bool,
        ws: &mut SplitFormWorkspace,
    ) -> u64 {
        debug_assert!(ks.len() <= ELEMENT_CHUNK && ELEMENT_CHUNK <= 64);
        if !self.batches() || ks.is_empty() {
            return 0;
        }
        let n = self.ops.n_nodes;
        let h = &self.q.data[0];
        let (mut wet, mut n_wet) = ([0; ELEMENT_CHUNK], 0);
        let (mut shore, mut n_shore) = ([0; ELEMENT_CHUNK], 0);
        for (j, &k) in ks.iter().enumerate() {
            let subcells = self
                .subcells
                .is_some_and(|(depth, _)| h[k * n..(k + 1) * n].iter().any(|&h| h < depth));
            if subcells {
                shore[n_shore] = j;
                n_shore += 1;
            } else {
                wet[n_wet] = j;
                n_wet += 1;
            }
        }
        ws.batch_subcells = shore[..n_shore].iter().fold(0, |bits, &j| bits | 1 << j);
        let level = Level::new();
        // A batch's worth of the next chunk per batch of this one
        let n_batches = n_wet.div_ceil(L) + n_shore.div_ceil(L);
        let mut ahead = ahead.chunks(ahead.len().div_ceil(n_batches).max(1));
        for (slots, subcells) in wet[..n_wet]
            .chunks(L)
            .map(|s| (s, false))
            .chain(shore[..n_shore].chunks(L).map(|s| (s, true)))
        {
            for &k in ahead.next().unwrap_or_default() {
                self.prefetch(k as usize);
            }
            // Lanes without an element repeat the first one (and are dropped)
            let mut lanes = [ks[slots[0]]; L];
            for (lane, &j) in lanes.iter_mut().zip(slots) {
                *lane = ks[j];
            }
            let at = Lanes {
                lanes: &lanes,
                slots,
                subcells,
                faces,
                time_of,
                face_mass,
            };
            match self.ops.n_1d {
                2 => dispatch!(level, simd => self.batch::<_, 2, 4>(simd, &at, ws)),
                3 => dispatch!(level, simd => self.batch::<_, 3, 9>(simd, &at, ws)),
                4 => dispatch!(level, simd => self.batch::<_, 4, 16>(simd, &at, ws)),
                _ => dispatch!(level, simd => self.batch::<_, 5, 25>(simd, &at, ws)),
            }
        }
        if ks.len() == 64 {
            u64::MAX
        } else {
            (1 << ks.len()) - 1
        }
    }

    /// Ask for the cache lines element `k`'s batch will read: its node
    /// states, bed and corners, and its neighbours' face nodes.
    ///
    /// On an unstructured mesh an element has face neighbours far from it
    /// in memory whatever the numbering (in Hilbert order 1 in 13 faces of
    /// the Chiloé coastline mesh is more than 64 elements away), and a pass
    /// of local time stepping skips through the elements, which the
    /// hardware's stream prefetch does not follow. Asked for a chunk early,
    /// the lines arrive before the lane loops read them one after another:
    /// 2–3 % of the Chiloé tide's wall time on 24 threads (32.7 against
    /// 33.2–33.8 s per half model hour).
    #[inline]
    fn prefetch(&self, k: usize) {
        let n = self.ops.n_nodes;
        let [q_h, q_hu, q_hv] = &self.q.data;
        let bed = self.config.bathymetry.map_or(&[][..], |b| &b.data[..]);
        let fields = [&q_h[..], q_hu, q_hv, bed];
        for field in fields {
            prefetch(field, k * n);
            prefetch(field, k * n + n - 1);
        }
        let corners = self.geom.corners(k).as_flattened();
        prefetch(corners, 0);
        prefetch(corners, 7);
        for link in self.mesh.face_links[k] {
            // A neighbour next in memory comes with the element's own rows
            if link.is_boundary() || link.element().abs_diff(k) <= 2 {
                continue;
            }
            let nodes = &self.ops.face_nodes[link.face()];
            let first = link.element() * n;
            for field in fields {
                prefetch(field, first + nodes[0]);
                prefetch(field, first + nodes[nodes.len() - 1]);
            }
        }
    }

    /// The terms of the elements `at.lanes` (the first `at.slots.len()` of
    /// them, into `ws.batch` at `at.slots`): the volume term (flux
    /// differencing, or the subcells), the surface terms, and the division
    /// by J.
    #[inline(always)]
    fn batch<S: Simd, const N1: usize, const NN: usize>(
        &self,
        simd: S,
        at: &Lanes<'_>,
        ws: &mut SplitFormWorkspace,
    ) {
        let (lanes, slots, subcells) = (at.lanes, at.slots, at.subcells);
        let data = self.gather::<S, NN>(simd, lanes);
        let zero = f64x8::<S>::splat(simd, 0.0);
        let mut rhs = [[zero; NN]; 3];
        let mut mass = [zero; MAX_SUBCELL_INTERFACES];
        if subcells {
            self.subcell_lines::<S, N1, NN>(simd, &data, lanes, &mut rhs, &mut mass);
        } else {
            self.lines::<S, N1, NN>(simd, &data, &mut rhs);
        }
        let face_mass = at.face_mass.then_some(&mut ws.batch_face_mass[..]);
        self.surface::<S, N1, NN>(simd, &data, at, &mut rhs, face_mass);
        for (rhs, batch) in rhs.iter_mut().zip(&mut ws.batch) {
            for (r, j_inv) in rhs.iter_mut().zip(&data.j_inv) {
                *r = *j_inv * *r;
            }
            scatter::<S, NN>(rhs, slots, batch);
        }
        if subcells {
            let interfaces = 2 * N1 * (N1 - 1);
            for (s, mass) in mass[..interfaces].iter().enumerate() {
                let mass = mass.as_slice();
                for (l, &j) in slots.iter().enumerate() {
                    ws.batch_mass[j * interfaces + s] = mass[l];
                }
            }
        }
    }

    /// The surface terms of [`Self::element_rhs`] for a batch, added to
    /// `rhs` (times J) face by face in the scalar order: `LIFT sJ (F(q⁻)·n −
    /// F*)`. `face_mass`, if given, receives the mass component of `F*` (the
    /// workspace's `batch_face_mass`).
    ///
    /// `F*` of an interior face is the hydrostatic HLL flux between the
    /// element's node and the neighbour's, evaluated here as
    /// [`Self::edge_fluxes`] evaluates it: from the edge's left side to its
    /// right, with the left side's normal. Both sides of a face therefore
    /// compute the same flux, bit for bit, each from its own batch, and
    /// exchange exactly opposite fluxes with no face buffer between them: a
    /// face's two node states and its flux stay in one element's cache
    /// lines. The left side's normal is minus the right side's to the bit
    /// where they share their vertices ([`FaceLink::shares_vertices`]); a
    /// periodic pair's is read. The other split forms take `F*` from
    /// `at.faces`, boundary faces from the scalar [`Self::boundary_flux`].
    #[inline(always)]
    fn surface<S: Simd, const N1: usize, const NN: usize>(
        &self,
        simd: S,
        data: &Batch<S, NN>,
        at: &Lanes<'_>,
        rhs: &mut [[f64x8<S>; NN]; 3],
        mut face_mass: Option<&mut [f64]>,
    ) {
        let g = self.config.equation.g;
        let h_min = self.config.equation.h_min.meters();
        let (mesh, geom) = (self.mesh, self.geom);
        let fused = self.surface == SurfaceFlux::HydrostaticHll;
        let [q_h, q_hu, q_hv] = &self.q.data;
        let bed = self.config.bathymetry.map(|b| &b.data[..]);
        let nodes = data.nodes(simd, g);
        let n_lanes = at.slots.len();
        let zero = f64x8::<S>::splat(simd, 0.0);
        let h_min_v = f64x8::<S>::splat(simd, h_min);
        let [rh, rhu, rhv] = rhs;
        // The left side's normal of a face node, from the stored geometry
        let stored_normal = |k: usize, face: usize, fi: usize| {
            let element = geom.element_geometry(k);
            if element.affine {
                element.faces[face].0
            } else {
                geom.normal(k, face, fi)
            }
        };
        for face in 0..4 {
            let links = at.lanes.map(|k| mesh.face_links[k][face]);
            let flag = |f: fn(FaceLink) -> bool| links.map(|link| f64::from(u8::from(f(link))));
            let interior = |f: fn(FaceLink) -> bool| links.map(|l| !l.is_boundary() && f(l));
            let boundary = flag(FaceLink::is_boundary);
            let left = interior(FaceLink::is_left).map(|x| f64::from(u8::from(x)));
            // Right sides of periodic pairs: the left side's normal is read
            let apart = interior(|link| !link.is_left() && !link.shares_vertices());
            let any_apart = fused && apart.iter().any(|&x| x);
            let apart = apart.map(|x| f64::from(u8::from(x)));
            let is_boundary = f64x8::<S>::from_slice(simd, &boundary).simd_gt(zero);
            let is_left = f64x8::<S>::from_slice(simd, &left).simd_gt(zero);
            let is_apart = f64x8::<S>::from_slice(simd, &apart).simd_gt(zero);
            let (nx, ny, s_j) = data.faces[face];
            // Per face node and lane: F* of a boundary face or a buffered
            // flux in `star`, or the neighbour's node (h, hu, hv, B) in
            // `other`; each lane's link is read once for the face's nodes
            let mut star = [[[0.0; L]; 3]; MAX_FACE_NODES];
            let mut other = [[[0.0; L]; 4]; MAX_FACE_NODES];
            let mut normal = [[[0.0; L]; 2]; MAX_FACE_NODES];
            for l in 0..L {
                let k = at.lanes[l];
                let link = links[l];
                if link.is_boundary() {
                    // Lanes without an element repeat the first one
                    if l < n_lanes {
                        let k_idx = ElementIndex::new(k);
                        let time = (at.time_of)(k);
                        for (fi, &node) in self.ops.face_nodes[face].iter().enumerate() {
                            let q_int = SWENodeState2D::new(
                                &self.q.get_state(k_idx, node),
                                self.bed(k_idx, node),
                                h_min,
                            );
                            let n = stored_normal(k, face, fi);
                            let f = self.boundary_flux(k_idx, face, node, &q_int, n, time);
                            (star[fi][0][l], star[fi][1][l], star[fi][2][l]) = (f.h, f.hu, f.hv);
                        }
                    }
                } else if fused {
                    let (nb, nb_face) = (link.element(), link.face());
                    let nb_nodes = &self.ops.face_nodes[nb_face][..N1];
                    let row = nb * NN..(nb + 1) * NN;
                    let (nb_h, nb_hu, nb_hv) =
                        (&q_h[row.clone()], &q_hu[row.clone()], &q_hv[row.clone()]);
                    let nb_bed = bed.map(|b| &b[row]);
                    let read_normal = any_apart && !link.is_left() && !link.shares_vertices();
                    for fi in 0..N1 {
                        let nb_fi = N1 - 1 - fi;
                        let j = nb_nodes[nb_fi];
                        (other[fi][0][l], other[fi][1][l]) = (nb_h[j], nb_hu[j]);
                        (other[fi][2][l], other[fi][3][l]) =
                            (nb_hv[j], nb_bed.map_or(0.0, |b| b[j]));
                        if read_normal {
                            (normal[fi][0][l], normal[fi][1][l]) =
                                stored_normal(nb, nb_face, nb_fi);
                        }
                    }
                } else {
                    let side = usize::from(!link.is_left());
                    let first = (2 * mesh.element_edges[k][face] + side) * N1;
                    for (fi, f) in at.faces[first..first + N1].iter().enumerate() {
                        (star[fi][0][l], star[fi][1][l], star[fi][2][l]) = (f.h, f.hu, f.hv);
                    }
                }
            }
            for (fi, &node) in self.ops.face_nodes[face].iter().enumerate() {
                let (star, other, normal) = (&star[fi], &other[fi], &normal[fi]);
                let mut star = Flux {
                    h: f64x8::<S>::from_slice(simd, &star[0]),
                    hu: f64x8::<S>::from_slice(simd, &star[1]),
                    hv: f64x8::<S>::from_slice(simd, &star[2]),
                };
                if fused {
                    // SWENodeState2D::new of the neighbour's node
                    let h = f64x8::<S>::from_slice(simd, &other[0]);
                    let wet = h.simd_gt(h_min_v);
                    let neighbour = Side {
                        h,
                        u: wet.select(f64x8::<S>::from_slice(simd, &other[1]) / h, zero),
                        v: wet.select(f64x8::<S>::from_slice(simd, &other[2]) / h, zero),
                        b: f64x8::<S>::from_slice(simd, &other[3]),
                    };
                    let this = data.side(node);
                    let (a, b) = (
                        neighbour.select(is_left, this),
                        this.select(is_left, neighbour),
                    );
                    // The edge's normal is its left side's
                    let (mut mx, mut my) = (is_left.select(nx, -nx), is_left.select(ny, -ny));
                    if any_apart {
                        mx = is_apart.select(f64x8::<S>::from_slice(simd, &normal[0]), mx);
                        my = is_apart.select(f64x8::<S>::from_slice(simd, &normal[1]), my);
                    }
                    // surface_flux: (F_a, −F_b); boundary lanes keep their F*
                    let (f_a, f_b) = hydrostatic_hll(simd, a, b, (mx, my), g, h_min);
                    star = Flux {
                        h: is_boundary.select(star.h, is_left.select(f_a.h, -f_b.h)),
                        hu: is_boundary.select(star.hu, is_left.select(f_a.hu, -f_b.hu)),
                        hv: is_boundary.select(star.hv, is_left.select(f_a.hv, -f_b.hv)),
                    };
                }
                if let Some(face_mass) = face_mass.as_deref_mut() {
                    let mass = star.h.as_slice();
                    for (l, &j) in at.slots.iter().enumerate() {
                        face_mass[j * 4 * N1 + face * N1 + fi] = mass[l];
                    }
                }
                let (fh, fu, fv) = nodes.flux(node, node, nx, ny);
                let scale = s_j * f64x8::<S>::splat(simd, self.ops.lift[face][(node, fi)]);
                rh[node] += scale * (fh - star.h);
                rhu[node] += scale * (fu - star.hu);
                rhv[node] += scale * (fv - star.hv);
            }
        }
    }

    /// Node states and bed of the elements `lanes`, and their geometry from
    /// their corners ([`GeometricFactors2D::corners`]): the contravariant
    /// vectors and `1/J` at the nodes and the faces' normals and `sJ`, each
    /// the value `GeometricFactors2D::compute` stored, operation for
    /// operation (a parallelogram's at its node 0, as the scalar kernel
    /// reads them).
    #[inline(always)]
    fn gather<S: Simd, const NN: usize>(&self, simd: S, lanes: &[usize; L]) -> Batch<S, NN> {
        // No closure below touches a vector: built without `target-cpu`
        // flags, a closure that is not inlined runs outside the vector
        // context `dispatch!` sets up (6 % slower than scalar, measured).
        let h_min = self.config.equation.h_min.meters();
        let [q_h, q_hu, q_hv] = &self.q.data;
        let zero = f64x8::<S>::splat(simd, 0.0);
        let one = f64x8::<S>::splat(simd, 1.0);
        let (geom, ops) = (self.geom, self.ops);

        let h = transpose::<S, NN>(simd, lanes, q_h);
        let hu = transpose::<S, NN>(simd, lanes, q_hu);
        let hv = transpose::<S, NN>(simd, lanes, q_hv);
        let b = match self.config.bathymetry {
            Some(bathymetry) => transpose::<S, NN>(simd, lanes, &bathymetry.data),
            None => [zero; NN],
        };
        // SWENodeState2D::new: zero velocity at h ≤ h_min
        let h_min_v = f64x8::<S>::splat(simd, h_min);
        let mut u = [zero; NN];
        let mut v = [zero; NN];
        for i in 0..NN {
            let wet = h[i].simd_gt(h_min_v);
            u[i] = wet.select(hu[i] / h[i], zero);
            v[i] = wet.select(hv[i] / h[i], zero);
        }

        let mut xy = [[0.0; L]; 8];
        let mut affine = [0.0; L];
        for (l, &k) in lanes.iter().enumerate() {
            for (c, corner) in geom.corners(k).iter().enumerate() {
                (xy[2 * c][l], xy[2 * c + 1][l]) = (corner[0], corner[1]);
            }
            affine[l] = f64::from(u8::from(geom.element_is_affine(k)));
        }
        let affine = f64x8::<S>::from_slice(simd, &affine).simd_gt(zero);
        let corners = Corners::new(simd, &xy);

        // Contravariant vectors J∇r (r-lines) and J∇s (s-lines) and 1/J at
        // every node: `GeometricFactors2D::contravariant` and `jacobian_inv`,
        // or a parallelogram's `AffineMetric` (node 0's)
        let (mut mrx, mut mry, mut msx, mut msy) = ([zero; NN], [zero; NN], [zero; NN], [zero; NN]);
        let (mut j_inv, mut det) = ([zero; NN], [zero; NN]);
        let at_node = |i: usize| (ops.nodes_r[i], ops.nodes_s[i]);
        let (r0, s0) = at_node(0);
        let (r0, s0) = (f64x8::<S>::splat(simd, r0), f64x8::<S>::splat(simd, s0));
        for i in 0..NN {
            let (r, s) = at_node(i);
            let r = affine.select(r0, f64x8::<S>::splat(simd, r));
            let s = affine.select(s0, f64x8::<S>::splat(simd, s));
            let m = corners.at(simd, r, s);
            let (rx, ry) = (m.y_s / m.det, -m.x_s / m.det);
            let (sx, sy) = (-m.y_r / m.det, m.x_r / m.det);
            mrx[i] = m.det * rx;
            mry[i] = m.det * ry;
            msx[i] = m.det * sx;
            msy[i] = m.det * sy;
            j_inv[i] = one / m.det;
            det[i] = m.det;
        }
        // sJ n = the outward contravariant vector of the face, the same at
        // every node of it
        let mut faces = [(zero, zero, zero); 4];
        for (face, out) in faces.iter_mut().enumerate() {
            let (r, s) = at_node(ops.face_nodes[face][0]);
            let m = corners.at(simd, f64x8::<S>::splat(simd, r), f64x8::<S>::splat(simd, s));
            let (vx, vy) = match face {
                0 => (m.y_r, -m.x_r),
                1 => (m.y_s, -m.x_s),
                2 => (-m.y_r, m.x_r),
                _ => (-m.y_s, m.x_s),
            };
            let s_j = (vx * vx + vy * vy).sqrt();
            *out = (vx / s_j, vy / s_j, s_j);
        }
        Batch {
            j_inv,
            det,
            faces,
            h,
            hu,
            hv,
            u,
            v,
            b,
            metric: [[mrx, mry], [msx, msy]],
            affine,
        }
    }

    /// [`Self::line_volume`] over every line of a batch (`N1` nodes per
    /// line, `NN = N1²` per element), lane by lane in the scalar order, into
    /// `rhs` (zero on entry).
    #[inline(always)]
    fn lines<S: Simd, const N1: usize, const NN: usize>(
        &self,
        simd: S,
        data: &Batch<S, NN>,
        rhs: &mut [[f64x8<S>; NN]; 3],
    ) {
        debug_assert_eq!(N1 * N1, NN);
        let g = self.config.equation.g;
        let d1 = &self.ops.dr_1d_row_major;
        let zero = f64x8::<S>::splat(simd, 0.0);
        let half = f64x8::<S>::splat(simd, 0.5);
        let g_v = f64x8::<S>::splat(simd, g);
        let nodes = data.nodes(simd, g);
        let [rh, rhu, rhv] = rhs;

        for line in 0..N1 {
            for direction in 0..2 {
                let idx = |a: usize| line_node(N1, direction, line, a);
                let [mx, my] = &data.metric[direction];
                for a in 0..N1 {
                    let ia = idx(a);
                    let (mxa, mya) = (mx[ia], my[ia]);
                    let (fh, fu, fv) = nodes.flux(ia, ia, mxa, mya);
                    let s = f64x8::<S>::splat(simd, 2.0 * d1[a * N1 + a]);
                    let (mut ah, mut au, mut av) = (s * fh, s * fu, s * fv);
                    for c in (a + 1)..N1 {
                        let ic = idx(c);
                        let m = (half * (mxa + mx[ic]), half * (mya + my[ic]));
                        let (fh, fu, fv) = nodes.flux(ia, ic, m.0, m.1);
                        let s = f64x8::<S>::splat(simd, 2.0 * d1[a * N1 + c]);
                        ah += s * fh;
                        au += s * fu;
                        av += s * fv;
                        let s = f64x8::<S>::splat(simd, 2.0 * d1[c * N1 + a]);
                        rh[ic] -= s * fh;
                        rhu[ic] -= s * fu;
                        rhv[ic] -= s * fv;
                    }

                    if self.config.bathymetry.is_some() {
                        let (mut db, mut db_x, mut db_y) = (zero, zero, zero);
                        for c in 0..N1 {
                            let ic = idx(c);
                            let d_b = f64x8::<S>::splat(simd, d1[a * N1 + c]) * data.b[ic];
                            db += d_b;
                            db_x += d_b * mx[ic];
                            db_y += d_b * my[ic];
                        }
                        let db_x = data.affine.select(db * mxa, half * (db * mxa + db_x));
                        let db_y = data.affine.select(db * mya, half * (db * mya + db_y));
                        let force = g_v * data.h[ia];
                        // The scalar kernel adds SWEState2D::new(0.0, ..): −0 + 0 = +0
                        ah += zero;
                        au += force * db_x;
                        av += force * db_y;
                    }

                    rh[ia] -= ah;
                    rhu[ia] -= au;
                    rhv[ia] -= av;
                }
            }
        }
    }

    /// The outer nodes of [`Self::outer_nodes`] for a batch: per face node
    /// (`face · N1 + fi`), the neighbour's first interior node on the GLL
    /// line through it and its distance, where the face has a neighbour.
    ///
    /// The distance is the ratio of the two elements' heights `J / sJ`
    /// normal to the face. The neighbour's `J` there comes from its corners
    /// as the element's own does ([`Self::gather`]), and its `sJ` is the
    /// element's own where the two faces share their vertices (a periodic
    /// pair's is read): the values [`Self::outer_node`] reads, to the bit.
    #[inline(always)]
    fn gather_outer<S: Simd, const N1: usize, const NN: usize>(
        &self,
        simd: S,
        data: &Batch<S, NN>,
        lanes: &[usize; L],
    ) -> [Outer<S>; MAX_FACE_NODES * 4] {
        let (mesh, geom, ops) = (self.mesh, self.geom, self.ops);
        let [q_h, q_hu, q_hv] = &self.q.data;
        let bed = self.config.bathymetry.map(|b| &b.data[..]);
        let zero = f64x8::<S>::splat(simd, 0.0);
        let h_min = f64x8::<S>::splat(simd, self.config.equation.h_min.meters());
        let spacing = f64x8::<S>::splat(simd, ops.nodes_1d[1] - ops.nodes_1d[0]);
        let r0 = f64x8::<S>::splat(simd, ops.nodes_r[0]);
        let s0 = f64x8::<S>::splat(simd, ops.nodes_s[0]);
        let empty = Outer {
            node: Side {
                h: zero,
                u: zero,
                v: zero,
                b: zero,
            },
            distance: zero,
            present: zero.simd_gt(zero),
        };
        let mut outer = [empty; MAX_FACE_NODES * 4];
        for face in 0..4 {
            let links = lanes.map(|k| mesh.face_links[k][face]);
            // Each lane's neighbour and its face; at a boundary the element
            // itself, whose values are then not used
            let across: [(usize, usize); L] = std::array::from_fn(|l| {
                if links[l].is_boundary() {
                    (lanes[l], face)
                } else {
                    (links[l].element(), links[l].face())
                }
            });
            let mut xy = [[0.0; L]; 8];
            let mut flags = [[0.0; L]; 3];
            let apart = links.map(|link| !link.is_boundary() && !link.shares_vertices());
            let any_apart = apart.iter().any(|&x| x);
            for l in 0..L {
                for (c, corner) in geom.corners(across[l].0).iter().enumerate() {
                    (xy[2 * c][l], xy[2 * c + 1][l]) = (corner[0], corner[1]);
                }
                flags[0][l] = f64::from(u8::from(!links[l].is_boundary()));
                flags[1][l] = f64::from(u8::from(geom.element_is_affine(across[l].0)));
                flags[2][l] = f64::from(u8::from(apart[l]));
            }
            let corners = Corners::new(simd, &xy);
            let present = f64x8::<S>::from_slice(simd, &flags[0]).simd_gt(zero);
            let affine = f64x8::<S>::from_slice(simd, &flags[1]).simd_gt(zero);
            let is_apart = f64x8::<S>::from_slice(simd, &flags[2]).simd_gt(zero);
            let s_j = data.faces[face].2;
            for fi in 0..N1 {
                let nb_fi = N1 - 1 - fi;
                // The neighbour's inner node (h, hu, hv, B), the reference
                // position of its face node, and a periodic pair's sJ
                let mut t = [[0.0; L]; 7];
                for l in 0..L {
                    let (e, f) = across[l];
                    let nb_node = ops.face_nodes[f][nb_fi];
                    // One node into the neighbour, normal to its face
                    let inner = match f {
                        0 => nb_node + N1,
                        1 => nb_node - 1,
                        2 => nb_node - N1,
                        _ => nb_node + 1,
                    };
                    let j = e * NN + inner;
                    (t[0][l], t[1][l], t[2][l]) = (q_h[j], q_hu[j], q_hv[j]);
                    t[3][l] = bed.map_or(0.0, |b| b[j]);
                    (t[4][l], t[5][l]) = (ops.nodes_r[nb_node], ops.nodes_s[nb_node]);
                    if apart[l] {
                        let element = geom.element_geometry(e);
                        t[6][l] = if element.affine {
                            element.faces[f].1
                        } else {
                            geom.surface_jacobian(e, f, nb_fi)
                        };
                    }
                }
                // SWENodeState2D::new
                let h = f64x8::<S>::from_slice(simd, &t[0]);
                let wet = h.simd_gt(h_min);
                let node = Side {
                    h,
                    u: wet.select(f64x8::<S>::from_slice(simd, &t[1]) / h, zero),
                    v: wet.select(f64x8::<S>::from_slice(simd, &t[2]) / h, zero),
                    b: f64x8::<S>::from_slice(simd, &t[3]),
                };
                // Heights J / sJ normal to the face (a parallelogram's J at
                // its node 0)
                let r = affine.select(r0, f64x8::<S>::from_slice(simd, &t[4]));
                let s = affine.select(s0, f64x8::<S>::from_slice(simd, &t[5]));
                let det = corners.at(simd, r, s).det;
                let mut s_j_across = s_j;
                if any_apart {
                    s_j_across = is_apart.select(f64x8::<S>::from_slice(simd, &t[6]), s_j);
                }
                let own = data.det[ops.face_nodes[face][fi]] / s_j;
                outer[face * N1 + fi] = Outer {
                    node,
                    distance: spacing * (det / s_j_across) / own,
                    present,
                };
            }
        }
        outer
    }

    /// [`Self::line_subcells`] over every line of a batch of subcell
    /// elements (`N1` nodes per line), lane by lane in the scalar order, into
    /// `rhs` (zero on entry) and the interface mass fluxes into `mass`
    /// (`subcell_slot` order).
    #[inline(always)]
    fn subcell_lines<S: Simd, const N1: usize, const NN: usize>(
        &self,
        simd: S,
        data: &Batch<S, NN>,
        lanes: &[usize; L],
        rhs: &mut [[f64x8<S>; NN]; 3],
        mass: &mut [f64x8<S>; MAX_SUBCELL_INTERFACES],
    ) {
        let g = self.config.equation.g;
        let h_min = self.config.equation.h_min.meters();
        let h_dry = self.subcells.map_or(0.0, |(_, h_dry)| h_dry);
        let w = &self.ops.weights_1d;
        let xi = &self.ops.nodes_1d;
        let d1 = &self.ops.dr_1d_row_major;
        let zero = f64x8::<S>::splat(simd, 0.0);
        let half = f64x8::<S>::splat(simd, 0.5);
        let nodes = data.nodes(simd, g);
        let outer = self.gather_outer::<S, N1, NN>(simd, data, lanes);
        let [rh, rhu, rhv] = rhs;

        for line in 0..N1 {
            // Line ends on faces 3/1 (r-lines) and 0/2 (s-lines); faces 2
            // and 3 list their nodes in reverse
            let (rev, fwd) = (N1 - 1 - line, line);
            for direction in 0..2 {
                let idx = |a: usize| line_node(N1, direction, line, a);
                let [mx, my] = &data.metric[direction];
                let ends = if direction == 0 {
                    [outer[3 * N1 + rev], outer[N1 + fwd]]
                } else {
                    [outer[fwd], outer[2 * N1 + rev]]
                };
                let mut metric = [(zero, zero); MAX_FACE_NODES];
                for a in 0..N1 {
                    metric[a] = (mx[idx(a)], my[idx(a)]);
                }

                // telescoped_interfaces: the line's first metric on
                // parallelograms, the telescoping sum on curved elements
                let mut interfaces = [(zero, zero); MAX_FACE_NODES + 1];
                interfaces[0] = metric[0];
                let mut m = (zero, zero);
                for a in 0..N1 - 1 {
                    for c in (0..N1).filter(|&c| c != a) {
                        let q2 = f64x8::<S>::splat(simd, 2.0 * w[a] * d1[a * N1 + c] * 0.5);
                        m.0 += q2 * (metric[a].0 + metric[c].0);
                        m.1 += q2 * (metric[a].1 + metric[c].1);
                    }
                    interfaces[a + 1] = (
                        data.affine.select(metric[0].0, m.0),
                        data.affine.select(metric[0].1, m.1),
                    );
                }
                interfaces[N1] = (
                    data.affine.select(metric[0].0, metric[N1 - 1].0),
                    data.affine.select(metric[0].1, metric[N1 - 1].1),
                );

                // The end nodes' physical fluxes
                let (first, last) = (idx(0), idx(N1 - 1));
                let (fh, fu, fv) = nodes.flux(first, first, metric[0].0, metric[0].1);
                let s = f64x8::<S>::splat(simd, 1.0 / w[0]);
                rh[first] += s * fh;
                rhu[first] += s * fu;
                rhv[first] += s * fv;
                let (m_last_x, m_last_y) = metric[N1 - 1];
                let (fh, fu, fv) = nodes.flux(last, last, m_last_x, m_last_y);
                let s = f64x8::<S>::splat(simd, 1.0 / w[N1 - 1]);
                rh[last] -= s * fh;
                rhu[last] -= s * fu;
                rhv[last] -= s * fv;

                // Face states of the subcells: limited reconstruction where
                // the stencil is complete, the node value otherwise
                let mut faces = [(data.side(0), data.side(0)); MAX_FACE_NODES];
                let mut x_left = -1.0;
                for a in 0..N1 {
                    let q = data.side(idx(a));
                    let x_right = x_left + w[a];
                    let (left, x_l, left_present) = if a > 0 {
                        let x = f64x8::<S>::splat(simd, xi[a - 1]);
                        (data.side(idx(a - 1)), x, zero.simd_ge(zero))
                    } else {
                        let end = ends[0];
                        let x = f64x8::<S>::splat(simd, -1.0) - end.distance;
                        (end.node, x, end.present)
                    };
                    let (right, x_r, right_present) = if a + 1 < N1 {
                        let x = f64x8::<S>::splat(simd, xi[a + 1]);
                        (data.side(idx(a + 1)), x, zero.simd_ge(zero))
                    } else {
                        let end = ends[1];
                        let x = f64x8::<S>::splat(simd, 1.0) + end.distance;
                        (end.node, x, end.present)
                    };
                    let x_c = f64x8::<S>::splat(simd, xi[a]);
                    let (face_l, face_r) = reconstruct_subcell(
                        simd,
                        [left, q, right],
                        [x_l, x_c, x_r],
                        (x_left, x_right),
                        h_dry,
                    );
                    let complete = left_present & right_present;
                    faces[a] = (q.select(complete, face_l), q.select(complete, face_r));
                    x_left = x_right;
                }

                // Interior subcell interfaces
                for a in 0..N1 - 1 {
                    let (ia, ib) = (idx(a), idx(a + 1));
                    let m = interfaces[a + 1];
                    let (f_a, f_b) =
                        hydrostatic_hll(simd, faces[a].1, faces[a + 1].0, (m.0, m.1), g, h_min);
                    mass[(direction * N1 + line) * (N1 - 1) + a] = f_a.h;
                    let s = f64x8::<S>::splat(simd, 1.0 / w[a]);
                    rh[ia] -= s * f_a.h;
                    rhu[ia] -= s * f_a.hu;
                    rhv[ia] -= s * f_a.hv;
                    let s = f64x8::<S>::splat(simd, 1.0 / w[a + 1]);
                    rh[ib] += s * f_b.h;
                    rhu[ib] += s * f_b.hu;
                    rhv[ib] += s * f_b.hv;
                }

                // Bed-slope term and the metric balance term (zero on
                // parallelograms)
                let minus_half_g = f64x8::<S>::splat(simd, -0.5 * g);
                let quarter_g = f64x8::<S>::splat(simd, 0.25 * g);
                for a in 0..N1 {
                    let ia = idx(a);
                    let (left, right) = faces[a];
                    let w_a = f64x8::<S>::splat(simd, w[a]);
                    let bed = minus_half_g * (left.h + right.h) * (right.b - left.b) / w_a;
                    let m0 = metric[0];
                    let (m_l, m_r) = (interfaces[a], interfaces[a + 1]);
                    let (h_a, b_a) = (data.h[ia], data.b[ia]);
                    let balance = quarter_g
                        * ((right.h + h_a) * (b_a - right.b) + (left.h + h_a) * (b_a - left.b))
                        / w_a;
                    let force_x = data.affine.select(
                        bed * m0.0,
                        bed * half * (m_l.0 + m_r.0) + balance * (m_r.0 - m_l.0),
                    );
                    let force_y = data.affine.select(
                        bed * m0.1,
                        bed * half * (m_l.1 + m_r.1) + balance * (m_r.1 - m_l.1),
                    );
                    rh[ia] += zero;
                    rhu[ia] += force_x;
                    rhv[ia] += force_y;
                }
            }
        }
    }
}

/// Ask for the cache line of `field[i]` (nothing past the end).
#[inline(always)]
fn prefetch(field: &[f64], i: usize) {
    #[cfg(target_arch = "x86_64")]
    if let Some(x) = field.get(i) {
        use core::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
        // SAFETY: a prefetch reads nothing; the address is in bounds
        unsafe { _mm_prefetch::<_MM_HINT_T0>(std::ptr::from_ref(x).cast()) };
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = (field, i);
}

/// Largest per-element count of subcell interfaces, `2 N1 (N1 − 1)` at P4.
const MAX_SUBCELL_INTERFACES: usize = 2 * MAX_FACE_NODES * (MAX_FACE_NODES - 1);

/// The node data of a batch of elements, one per lane.
struct Batch<S: Simd, const NN: usize> {
    h: [f64x8<S>; NN],
    hu: [f64x8<S>; NN],
    hv: [f64x8<S>; NN],
    u: [f64x8<S>; NN],
    v: [f64x8<S>; NN],
    b: [f64x8<S>; NN],
    /// Contravariant vector of each line direction at every node,
    /// `[[J r_x, J r_y], [J s_x, J s_y]]`
    metric: [[[f64x8<S>; NN]; 2]; 2],
    /// Lanes of parallelograms
    affine: mask64x8<S>,
    /// `1/J` at every node
    j_inv: [f64x8<S>; NN],
    /// `J` at every node
    det: [f64x8<S>; NN],
    /// Outward unit normal `(n_x, n_y)` and `sJ` of each face
    faces: [(f64x8<S>, f64x8<S>, f64x8<S>); 4],
}

/// The corners of a batch of bilinear elements, as the differences
/// `BilinearMetric::at` takes of them.
struct Corners<S: Simd> {
    /// `x₁ − x₀`, `x₂ − x₃`, and the same of `y`
    along_r: [[f64x8<S>; 2]; 2],
    /// `x₃ − x₀`, `x₂ − x₁`, and the same of `y`
    along_s: [[f64x8<S>; 2]; 2],
}

/// Derivatives of the bilinear map at a reference point (`BilinearMetric`).
struct Metric<S: Simd> {
    x_r: f64x8<S>,
    x_s: f64x8<S>,
    y_r: f64x8<S>,
    y_s: f64x8<S>,
    det: f64x8<S>,
}

impl<S: Simd> Corners<S> {
    /// `xy[2c]`, `xy[2c + 1]`: x and y of corner `c`, one element per lane.
    #[inline(always)]
    fn new(simd: S, xy: &[[f64; L]; 8]) -> Self {
        let [x0, y0, x1, y1, x2, y2, x3, y3] = [
            f64x8::<S>::from_slice(simd, &xy[0]),
            f64x8::<S>::from_slice(simd, &xy[1]),
            f64x8::<S>::from_slice(simd, &xy[2]),
            f64x8::<S>::from_slice(simd, &xy[3]),
            f64x8::<S>::from_slice(simd, &xy[4]),
            f64x8::<S>::from_slice(simd, &xy[5]),
            f64x8::<S>::from_slice(simd, &xy[6]),
            f64x8::<S>::from_slice(simd, &xy[7]),
        ];
        Self {
            along_r: [[x1 - x0, x2 - x3], [y1 - y0, y2 - y3]],
            along_s: [[x3 - x0, x2 - x1], [y3 - y0, y2 - y1]],
        }
    }

    /// `BilinearMetric::at(r, s)`, operation for operation.
    #[inline(always)]
    fn at(&self, simd: S, r: f64x8<S>, s: f64x8<S>) -> Metric<S> {
        let one = f64x8::<S>::splat(simd, 1.0);
        let quarter = f64x8::<S>::splat(simd, 0.25);
        let [[x_lo, x_hi], [y_lo, y_hi]] = self.along_r;
        let x_r = quarter * ((one - s) * x_lo + (one + s) * x_hi);
        let y_r = quarter * ((one - s) * y_lo + (one + s) * y_hi);
        let [[x_lo, x_hi], [y_lo, y_hi]] = self.along_s;
        let x_s = quarter * ((one - r) * x_lo + (one + r) * x_hi);
        let y_s = quarter * ((one - r) * y_lo + (one + r) * y_hi);
        Metric {
            x_r,
            x_s,
            y_r,
            y_s,
            det: x_r * y_s - x_s * y_r,
        }
    }
}

/// The elements of one batch and what their terms take from outside it.
struct Lanes<'a> {
    /// The element of each lane (unused lanes repeat the first)
    lanes: &'a [usize; L],
    /// The workspace slot of each used lane
    slots: &'a [usize],
    /// Whether the elements take the subcell volume term
    subcells: bool,
    /// Interior-face fluxes ([`SplitFormSWE2D::edge_fluxes`])
    faces: &'a [SWEState2D],
    /// The time of an element's boundary faces
    time_of: &'a dyn Fn(usize) -> f64,
    /// Whether the mass component of `F*` is asked for
    face_mass: bool,
}

impl<S: Simd, const NN: usize> Batch<S, NN> {
    /// The state of node `i` as a face state.
    #[inline(always)]
    fn side(&self, i: usize) -> Side<S> {
        Side {
            h: self.h[i],
            u: self.u[i],
            v: self.v[i],
            b: self.b[i],
        }
    }

    #[inline(always)]
    fn nodes(&self, simd: S, g: f64) -> Nodes<'_, S, NN> {
        Nodes {
            h: &self.h,
            hu: &self.hu,
            hv: &self.hv,
            u: &self.u,
            v: &self.v,
            half: f64x8::<S>::splat(simd, 0.5),
            half_g: f64x8::<S>::splat(simd, 0.5 * g),
        }
    }
}

/// An outer node of a batch: the state, its distance from the face in the
/// element's reference coordinate, and the lanes where it exists.
#[derive(Clone, Copy)]
struct Outer<S: Simd> {
    node: Side<S>,
    distance: f64x8<S>,
    present: mask64x8<S>,
}

/// `f64::signum(x) · f64::min(|x|, bound)` for a nonzero `x`.
#[inline(always)]
fn signed_min<S: Simd>(simd: S, x: f64x8<S>, bound: f64x8<S>) -> f64x8<S> {
    let zero = f64x8::<S>::splat(simd, 0.0);
    let one = f64x8::<S>::splat(simd, 1.0);
    let sign = x.simd_lt(zero).select(-one, one);
    sign * min(x.abs(), bound)
}

/// `limited_slope`: the central slope of `q` at `xi[1]`, capped so that the
/// face values stay between the neighbouring node values; zero at an
/// extremum.
#[inline(always)]
fn limited_slope<S: Simd>(
    simd: S,
    q: [f64x8<S>; 3],
    xi: [f64x8<S>; 3],
    (x_l, x_r): (f64x8<S>, f64x8<S>),
) -> f64x8<S> {
    let zero = f64x8::<S>::splat(simd, 0.0);
    let half = f64x8::<S>::splat(simd, 0.5);
    let (dm, dp) = (q[1] - q[0], q[2] - q[1]);
    let extremum = (dm * dp).simd_le(zero);
    let central = (q[2] - q[0]) / (xi[2] - xi[0]);
    // bound(dq, face, node) = |dq / (face > 0 ? face : node / 2)|
    let (face_m, node_m) = (xi[1] - x_l, xi[1] - xi[0]);
    let (face_p, node_p) = (x_r - xi[1], xi[2] - xi[1]);
    let bound_m = (dm / face_m.simd_gt(zero).select(face_m, half * node_m)).abs();
    let bound_p = (dp / face_p.simd_gt(zero).select(face_p, half * node_p)).abs();
    let bound = min(bound_m, bound_p);
    extremum.select(zero, signed_min(simd, central, bound))
}

/// `reconstruct_subcell`: the face states of the subcell `[x_l, x_r]`
/// around `nodes[1]` from a limited linear reconstruction of `h`, `η`, `u`
/// and `v` (no `η` slope where a dry node lies below a wet surface of the
/// stencil).
#[inline(always)]
fn reconstruct_subcell<S: Simd>(
    simd: S,
    nodes: [Side<S>; 3],
    xi: [f64x8<S>; 3],
    (x_l, x_r): (f64, f64),
    h_dry: f64,
) -> (Side<S>, Side<S>) {
    let zero = f64x8::<S>::splat(simd, 0.0);
    let h_dry = f64x8::<S>::splat(simd, h_dry);
    let subcell = (f64x8::<S>::splat(simd, x_l), f64x8::<S>::splat(simd, x_r));
    let [m, c, p] = nodes;
    let eta = [m.h + m.b, c.h + c.b, p.h + p.b];
    let h = [m.h, c.h, p.h];
    let mut wet_surface = f64x8::<S>::splat(simd, f64::NEG_INFINITY);
    for (h, eta) in h.iter().zip(&eta) {
        // top.max(η) over the wet nodes
        let higher = eta.simd_gt(wet_surface);
        wet_surface = (h.simd_ge(h_dry) & higher).select(*eta, wet_surface);
    }
    let mut flooding = zero.simd_gt(zero);
    for (h, eta) in h.iter().zip(&eta) {
        flooding |= h.simd_lt(h_dry) & eta.simd_lt(wet_surface);
    }
    let s_h = limited_slope(simd, h, xi, subcell);
    let s_eta = flooding.select(zero, limited_slope(simd, eta, xi, subcell));
    let s_u = limited_slope(simd, [m.u, c.u, p.u], xi, subcell);
    let s_v = limited_slope(simd, [m.v, c.v, p.v], xi, subcell);
    let slopes = [s_h, s_eta, s_u, s_v];
    (
        reconstructed(simd, c, eta[1], slopes, subcell.0 - xi[1]),
        reconstructed(simd, c, eta[1], slopes, subcell.1 - xi[1]),
    )
}

/// `[node][lane]` vectors of the nodal `field` of the elements `lanes`: each
/// element's row is loaded eight nodes at a time and the eight rows are
/// transposed in registers ([`transpose8`]); the nodes left over (one at P2)
/// go value by value.
#[inline(always)]
fn transpose<S: Simd, const NN: usize>(
    simd: S,
    lanes: &[usize; L],
    field: &[f64],
) -> [f64x8<S>; NN] {
    let zero = f64x8::<S>::splat(simd, 0.0);
    let mut out = [zero; NN];
    let whole = NN - NN % L;
    for first in (0..whole).step_by(L) {
        let mut rows = [zero; L];
        for (row, &k) in rows.iter_mut().zip(lanes) {
            *row = f64x8::<S>::from_slice(simd, &field[k * NN + first..k * NN + first + L]);
        }
        out[first..first + L].copy_from_slice(&transpose8(rows));
    }
    for (i, out) in out.iter_mut().enumerate().skip(whole) {
        let mut t = [0.0; L];
        for (t, &k) in t.iter_mut().zip(lanes) {
            *t = field[k * NN + i];
        }
        *out = f64x8::<S>::from_slice(simd, &t);
    }
    out
}

/// The rows `slots[l] · NN..` of `rows` from the `[node][lane]` vectors
/// `values`, lane `l` to `slots[l]`: [`transpose`] backwards.
#[inline(always)]
fn scatter<S: Simd, const NN: usize>(values: &[f64x8<S>; NN], slots: &[usize], rows: &mut [f64]) {
    let whole = NN - NN % L;
    for first in (0..whole).step_by(L) {
        let mut nodes = [values[0]; L];
        nodes.copy_from_slice(&values[first..first + L]);
        for (lane, &j) in transpose8(nodes).iter().zip(slots) {
            lane.store_slice(&mut rows[j * NN + first..j * NN + first + L]);
        }
    }
    for (i, value) in values.iter().enumerate().skip(whole) {
        let value = value.as_slice();
        for (l, &j) in slots.iter().enumerate() {
            rows[j * NN + i] = value[l];
        }
    }
}

/// The transpose of eight vectors of eight values, `out[i][l] = v[l][i]`:
/// three rounds of interleaving each vector with the one four places on.
#[inline(always)]
fn transpose8<S: Simd>(v: [f64x8<S>; L]) -> [f64x8<S>; L] {
    interleave4(interleave4(interleave4(v)))
}

/// `v[i]` interleaved with `v[i + 4]`: the low halves, then the high halves.
#[inline(always)]
fn interleave4<S: Simd>(v: [f64x8<S>; L]) -> [f64x8<S>; L] {
    [
        v[0].zip_low(v[4]),
        v[0].zip_high(v[4]),
        v[1].zip_low(v[5]),
        v[1].zip_high(v[5]),
        v[2].zip_low(v[6]),
        v[2].zip_high(v[6]),
        v[3].zip_low(v[7]),
        v[3].zip_high(v[7]),
    ]
}

/// The node states of a batch of elements for the two-point flux.
struct Nodes<'a, S: Simd, const NN: usize> {
    h: &'a [f64x8<S>; NN],
    hu: &'a [f64x8<S>; NN],
    hv: &'a [f64x8<S>; NN],
    u: &'a [f64x8<S>; NN],
    v: &'a [f64x8<S>; NN],
    half: f64x8<S>,
    half_g: f64x8<S>,
}

impl<S: Simd, const NN: usize> Nodes<'_, S, NN> {
    /// `wintermeyer_flux_2d(q_l, q_r, m)`, operation for operation.
    #[inline(always)]
    fn flux(
        &self,
        l: usize,
        r: usize,
        mx: f64x8<S>,
        my: f64x8<S>,
    ) -> (f64x8<S>, f64x8<S>, f64x8<S>) {
        let (h, hu, hv, u, v) = (self.h, self.hu, self.hv, self.u, self.v);
        let mass = self.half * ((hu[l] + hu[r]) * mx + (hv[l] + hv[r]) * my);
        let pressure = self.half_g * (h[l] * h[r]);
        let u_avg = self.half * (u[l] + u[r]);
        let v_avg = self.half * (v[l] + v[r]);
        (
            mass,
            mass * u_avg + pressure * mx,
            mass * v_avg + pressure * my,
        )
    }
}

/// One side of a face for the batched hydrostatic HLL: the node states
/// (`SWENodeState2D`) of [`VOLUME_BATCH`] face nodes, one per lane.
#[derive(Clone, Copy)]
struct Side<S: Simd> {
    h: f64x8<S>,
    u: f64x8<S>,
    v: f64x8<S>,
    b: f64x8<S>,
}

impl<S: Simd> Side<S> {
    /// `other` in the lanes of `mask`, `self` elsewhere.
    #[inline(always)]
    fn select(self, mask: mask64x8<S>, other: Self) -> Self {
        Self {
            h: mask.select(other.h, self.h),
            u: mask.select(other.u, self.u),
            v: mask.select(other.v, self.v),
            b: mask.select(other.b, self.b),
        }
    }
}

/// A side of the hydrostatic HLL after the Audusse et al. (2004)
/// reconstruction, in face-aligned components (as `HllSide`).
#[derive(Clone, Copy)]
struct Star<S: Simd> {
    h: f64x8<S>,
    hun: f64x8<S>,
    hut: f64x8<S>,
    un: f64x8<S>,
    ut: f64x8<S>,
}

impl<S: Simd> Star<S> {
    /// The `side` of `hydrostatic_hll`: `h* = max(0, h + B − b_face)` with
    /// the velocity kept (zero on a side at or below `h_min`).
    #[inline(always)]
    fn new(
        simd: S,
        q: Side<S>,
        b_face: f64x8<S>,
        (nx, ny): (f64x8<S>, f64x8<S>),
        h_min: f64,
    ) -> Self {
        let zero = f64x8::<S>::splat(simd, 0.0);
        let h_min = f64x8::<S>::splat(simd, h_min);
        let h_star = max(q.h + q.b - b_face, zero);
        let wet = q.h.simd_gt(h_min);
        let (u, v) = (wet.select(q.u, zero), wet.select(q.v, zero));
        let (un, ut) = (u * nx + v * ny, -u * ny + v * nx);
        let (hun, hut) = (h_star * un, h_star * ut);
        let star_wet = h_star.simd_gt(h_min);
        Self {
            h: h_star,
            hun,
            hut,
            un: star_wet.select(un, zero),
            ut: star_wet.select(ut, zero),
        }
    }

    /// The physical flux in face-aligned components.
    #[inline(always)]
    fn flux(self, half_g: f64x8<S>) -> Flux<S> {
        Flux {
            h: self.h * self.un,
            hu: self.h * self.un * self.un + half_g * self.h * self.h,
            hv: self.h * self.un * self.ut,
        }
    }
}

/// The components of [`VOLUME_BATCH`] fluxes, one per lane.
#[derive(Clone, Copy)]
struct Flux<S: Simd> {
    h: f64x8<S>,
    hu: f64x8<S>,
    hv: f64x8<S>,
}

/// Largest face of P1–P4.
const MAX_FACE_NODES: usize = 5;

/// `f64::max(a, b)` for a non-NaN `b`: `a` if `a > b`, else `b` (a tie of
/// ±0 gives `b`).
#[inline(always)]
fn max<S: Simd>(a: f64x8<S>, b: f64x8<S>) -> f64x8<S> {
    a.simd_gt(b).select(a, b)
}

/// `f64::min(a, b)` for a non-NaN `b`: `a` if `a < b`, else `b`.
#[inline(always)]
fn min<S: Simd>(a: f64x8<S>, b: f64x8<S>) -> f64x8<S> {
    a.simd_lt(b).select(a, b)
}

/// [`SplitFormSWE2D::hydrostatic_hll`] with `hll_flux_face_aligned`,
/// `einfeldt_speeds_2d` and `rotate_from_normal`, lane by lane in the scalar
/// order: every branch is evaluated and selected.
#[inline(always)]
fn hydrostatic_hll<S: Simd>(
    simd: S,
    q_a: Side<S>,
    q_b: Side<S>,
    m: (f64x8<S>, f64x8<S>),
    g: f64,
    h_min: f64,
) -> (Flux<S>, Flux<S>) {
    let zero = f64x8::<S>::splat(simd, 0.0);
    let half = f64x8::<S>::splat(simd, 0.5);
    let two = f64x8::<S>::splat(simd, 2.0);
    let one = f64x8::<S>::splat(simd, 1.0);
    let g_v = f64x8::<S>::splat(simd, g);
    let half_g = f64x8::<S>::splat(simd, 0.5 * g);
    let h_min_v = f64x8::<S>::splat(simd, h_min);
    let norm = (m.0 * m.0 + m.1 * m.1).sqrt();
    let (nx, ny) = (m.0 / norm, m.1 / norm);
    let b_face = max(q_a.b, q_b.b);
    let a = Star::new(simd, q_a, b_face, (nx, ny), h_min);
    let b = Star::new(simd, q_b, b_face, (nx, ny), h_min);
    let (h_l, h_r) = (a.h, b.h);

    // hll_flux_face_aligned, with einfeldt_speeds_2d
    let hll_dry = h_l.simd_le(h_min_v) & h_r.simd_le(h_min_v);
    let c_l = (g_v * max(h_l, zero)).sqrt();
    let c_r = (g_v * max(h_r, zero)).sqrt();
    let sqrt_h_l = max(h_l, zero).sqrt();
    let sqrt_h_r = max(h_r, zero).sqrt();
    let roe = (sqrt_h_l + sqrt_h_r).simd_gt(f64x8::<S>::splat(simd, 1e-10));
    let h_roe = half * (h_l + h_r);
    let un_roe = roe.select(
        (sqrt_h_l * a.un + sqrt_h_r * b.un) / (sqrt_h_l + sqrt_h_r),
        zero,
    );
    let c_roe = roe.select((g_v * h_roe).sqrt(), zero);
    let s_l = h_l
        .simd_gt(h_min_v)
        .select(min(a.un - c_l, un_roe - c_roe), b.un - two * c_r);
    let s_r = h_r
        .simd_gt(h_min_v)
        .select(max(b.un + c_r, un_roe + c_roe), a.un + two * c_l);
    let (f_l, f_r) = (a.flux(half_g), b.flux(half_g));
    let inv_ds = f64x8::<S>::splat(simd, 1.0) / (s_r - s_l);
    let s_lr = s_l * s_r;
    let star = Flux {
        h: inv_ds * (s_r * f_l.h - s_l * f_r.h + s_lr * (h_r - h_l)),
        hu: inv_ds * (s_r * f_l.hu - s_l * f_r.hu + s_lr * (b.hun - a.hun)),
        hv: inv_ds * (s_r * f_l.hv - s_l * f_r.hv + s_lr * (b.hut - a.hut)),
    };
    // s_l >= 0: F_l; else s_r <= 0: F_r; else the star flux; zero if dry
    let (from_l, from_r) = (s_l.simd_ge(zero), s_r.simd_le(zero));
    let hll = Flux {
        h: hll_dry.select(zero, from_l.select(f_l.h, from_r.select(f_r.h, star.h))),
        hu: hll_dry.select(zero, from_l.select(f_l.hu, from_r.select(f_r.hu, star.hu))),
        hv: hll_dry.select(zero, from_l.select(f_l.hv, from_r.select(f_r.hv, star.hv))),
    };
    // |m| rotate_from_normal
    let flux = Flux {
        h: norm * hll.h,
        hu: norm * (hll.hu * nx - hll.hv * ny),
        hv: norm * (hll.hu * ny + hll.hv * nx),
    };
    // + g/2 (h^2 - h*^2)(0, m), all of g h^2 / 2 where HLL is dry, and a
    // wet side's wall_ram_pressure there (the wall along +m from a, −m from b)
    let (h_star_a, h_star_b) = (hll_dry.select(zero, h_l), hll_dry.select(zero, h_r));
    let wall = |q: Side<S>, towards: f64| {
        let un = f64x8::<S>::splat(simd, towards) * (q.u * nx + q.v * ny);
        let un = max(un, zero);
        let head = un * un / (two * g_v);
        let sill = b_face - (q.h + q.b);
        let ram = (one - head / sill) * q.h * un * (two * un + (g_v * q.h).sqrt());
        (hll_dry & q.h.simd_gt(h_min_v) & sill.simd_gt(head)).select(ram, zero)
    };
    // (zero in every lane unless one is dry: two divisions and a square
    // root per side that most batches do not need)
    let (wall_a, wall_b) = if hll_dry.any_true() {
        (wall(q_a, 1.0), wall(q_b, -1.0))
    } else {
        (zero, zero)
    };
    let dp_a = half_g * (q_a.h * q_a.h - h_star_a * h_star_a) + wall_a;
    let dp_b = half_g * (q_b.h * q_b.h - h_star_b * h_star_b) + wall_b;
    (
        Flux {
            h: flux.h + zero,
            hu: flux.hu + dp_a * m.0,
            hv: flux.hv + dp_a * m.1,
        },
        Flux {
            h: flux.h + zero,
            hu: flux.hu + dp_b * m.0,
            hv: flux.hv + dp_b * m.1,
        },
    )
}

/// The state at distance `d` from the node `c` (with surface `eta`) along
/// the limited slopes `[h, η, u, v]`, the depth kept ≥ 0 and the bed `η − h`.
#[inline(always)]
fn reconstructed<S: Simd>(
    simd: S,
    c: Side<S>,
    eta: f64x8<S>,
    [s_h, s_eta, s_u, s_v]: [f64x8<S>; 4],
    d: f64x8<S>,
) -> Side<S> {
    let h = max(c.h + s_h * d, f64x8::<S>::splat(simd, 0.0));
    Side {
        h,
        u: c.u + s_u * d,
        v: c.v + s_v * d,
        b: eta + s_eta * d - h,
    }
}

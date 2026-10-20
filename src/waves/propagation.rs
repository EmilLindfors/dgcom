//! The propagation right-hand side and positivity limiter on the node-major
//! state ([`super::WaveSolution`], `[point][component]`, component
//! `c = i n_θ + j`), the layout of the node passes (implicit refraction and
//! frequency shifting, sources) too, so a step reads and writes the state in
//! place without transposing it.
//!
//! Node-major, a frequency's directions are contiguous at every node, and
//! they share everything but `cos θ_j`, `sin θ_j` and the action: the
//! group velocity, the current and the element geometry. With the `simd`
//! feature the geographic term takes [`LANES`] directions per vector
//! (`fearless_simd`), the last vector of a frequency padded.
//!
//! Every component is computed with the operations, in the order, of the
//! component-major scalar kernel it replaces, so the step is the same bit for
//! bit (`the_simd_propagation_is_the_scalar_one`).
//!
//! A Runge–Kutta stage is one pass over the elements
//! ([`WaveModel2D::propagation_stage`]): each element's right-hand side is
//! written into its rows of the stage value, combined there with the step
//! start and the previous stage (Shu–Osher form, [`ShuOsherStage`]) and
//! limited for positivity, so no right-hand side is stored and no separate
//! combination or limiter pass reads the state again. The state is ≈ 70 MB
//! at Frøya (10,040 nodes × 900 components), so the passes over it are the
//! stage's cost: the stage reads its input and the step start and writes the
//! stage value, where a stored right-hand side, a pass per combination term
//! and the limiter's read took ≈ 4× the traffic (`notes/waves.md`, F.4).

use crate::mesh::BoundaryTag;
use crate::time::ShuOsherStage;
use crate::types::ElementIndex;

use super::{SpectralAdvection, WaveModel2D, face_flux, for_each_chunk};

/// Directions per vector of the geographic term.
#[cfg(feature = "simd")]
pub(super) const LANES: usize = 8;

/// The face nodes of one element face, with what the upwind flux needs.
#[derive(Clone, Copy)]
struct FaceNode {
    /// The element's node on the face
    a: usize,
    normal: (f64, f64),
    surface_jacobian: f64,
    /// The neighbour's coincident node (global), or none at a boundary
    neighbour: Option<usize>,
    /// A boundary face that lets the boundary spectrum in
    open: bool,
}

impl WaveModel2D {
    /// `out = −∇·((c_g e_θ + U) N) − ∂(c_θ N)/∂θ − ∂(c_σ N)/∂σ` for every
    /// component, node-major (no sources).
    pub(super) fn propagation_rhs_node_major(&self, n: &[f64], out: &mut [f64]) {
        let nc = self.grid.n_components();
        let nn = self.ops.n_nodes;
        for_each_chunk(
            out,
            nn * nc,
            || (),
            |_, k, rows| self.geographic_element(n, k, rows),
        );
        if !(self.implicit_refraction && self.implicit_frequency_shift) {
            for_each_chunk(
                out,
                nc,
                || (),
                |_, p, row| self.spectral_node(&n[p * nc..(p + 1) * nc], p, row),
            );
        }
    }

    /// The upwind face nodes of element `k`, face by face.
    fn face_nodes(&self, k: usize) -> impl Iterator<Item = (usize, FaceNode)> + '_ {
        let (ops, geom, mesh) = (&*self.ops, &*self.geom, &*self.mesh);
        let (nn, nfn) = (ops.n_nodes, ops.n_face_nodes);
        let element = ElementIndex::new(k);
        (0..4).flat_map(move |face| {
            let neighbour = mesh.neighbor(element, face);
            let open =
                neighbour.is_none() && mesh.boundary_tag(element, face) == Some(BoundaryTag::Open);
            // A dry neighbour (absorbing land) is coast
            let neighbour = neighbour.filter(|nb| !self.dry_element(nb.element));
            (0..nfn).map(move |fi| {
                (
                    face * nfn + fi,
                    FaceNode {
                        a: ops.face_nodes[face][fi],
                        normal: geom.normal(k, face, fi),
                        surface_jacobian: geom.surface_jacobian(k, face, fi),
                        neighbour: neighbour
                            .map(|nb| nb.element * nn + ops.face_nodes[nb.face][nfn - 1 - fi]),
                        open,
                    },
                )
            })
        })
    }

    /// The DG geographic term of element `k` for every component, into its
    /// node-major rows `out` (overwritten); zero in a dry element.
    fn geographic_element(&self, n: &[f64], k: usize, out: &mut [f64]) {
        if self.dry_element(k) {
            out.fill(0.0);
            return;
        }
        #[cfg(feature = "simd")]
        if super::vector_kernels() && self.geographic_element_simd(n, k, out) {
            return;
        }
        self.geographic_element_scalar(n, k, out);
    }

    /// [`Self::geographic_element`] one component at a time (the reference of
    /// the vector kernel, and the path without the `simd` feature).
    pub(super) fn geographic_element_scalar(&self, n: &[f64], k: usize, out: &mut [f64]) {
        let (ops, geom) = (&*self.ops, &*self.geom);
        let (nn, nfn) = (ops.n_nodes, ops.n_face_nodes);
        let np = self.n_points();
        let (nc, nd) = (self.grid.n_components(), self.grid.n_dir());
        let base = k * nn;
        let faces: Vec<(usize, FaceNode)> = self.face_nodes(k).collect();
        let (mut fr, mut fs) = (vec![0.0; nn], vec![0.0; nn]);
        for c in 0..nc {
            let (i, j) = (c / nd, c % nd);
            let (cos, sin) = (self.grid.cos_theta[j], self.grid.sin_theta[j]);
            let cg = &self.cg[i * np..(i + 1) * np];
            let field = |p: usize| n[p * nc + c];
            let velocity = |p: usize| {
                let [u, v] = self.current[p];
                [cg[p] * cos + u, cg[p] * sin + v]
            };
            // Volume: −J⁻¹ (D_r F̃_r + D_s F̃_s), F̃ the contravariant fluxes
            for a in 0..nn {
                let p = base + a;
                let [vx, vy] = velocity(p);
                let ((jrx, jry), (jsx, jsy)) = geom.contravariant(k, a);
                fr[a] = (jrx * vx + jry * vy) * field(p);
                fs[a] = (jsx * vx + jsy * vy) * field(p);
            }
            for a in 0..nn {
                let (dr, ds) = (
                    &ops.dr_row_major[a * nn..(a + 1) * nn],
                    &ops.ds_row_major[a * nn..(a + 1) * nn],
                );
                let mut div = 0.0;
                for b in 0..nn {
                    div += dr[b] * fr[b] + ds[b] * fs[b];
                }
                out[a * nc + c] = -div * geom.jacobian_inv(k, a);
            }
            // Faces: lift (F⁻·n − F*) with the upwind flux F*
            for &(slot, face) in &faces {
                let (face_index, fi) = (slot / nfn, slot % nfn);
                let p = base + face.a;
                let (nx, ny) = face.normal;
                let [vx, vy] = velocity(p);
                let un_in = vx * nx + vy * ny;
                let n_in = field(p);
                let flux = match face.neighbour {
                    Some(q) => {
                        let [wx, wy] = velocity(q);
                        let un_out = wx * nx + wy * ny;
                        if un_in + un_out >= 0.0 {
                            un_in * n_in
                        } else {
                            un_out * field(q)
                        }
                    }
                    // Out through every boundary; in only through open ones
                    None if un_in >= 0.0 => un_in * n_in,
                    None if face.open => un_in * self.boundary.action(c, p),
                    None => 0.0,
                };
                let jump = (un_in * n_in - flux) * face.surface_jacobian;
                let lift = &ops.lift_row_major[face_index];
                for b in 0..nn {
                    let l = lift[b * nfn + fi];
                    if l != 0.0 {
                        out[b * nc + c] += l * jump * geom.jacobian_inv(k, b);
                    }
                }
            }
        }
    }

    /// Subtract the direction and frequency flux divergences at node `p`,
    /// from its spectrum `here` (one value per component), from `out`.
    fn spectral_node(&self, here: &[f64], p: usize, out: &mut [f64]) {
        let (nf, nd) = (self.grid.n_freq(), self.grid.n_dir());
        let grid = &self.grid;
        let second_order = self.spectral_advection == SpectralAdvection::VanLeer;
        let explicit_refraction = !self.implicit_refraction;
        let inv_dtheta = 1.0 / grid.d_theta;
        for (c, out) in out.iter_mut().enumerate() {
            let (i, j) = (c / nd, c % nd);
            // Directions: periodic, faces at θ_j ± Δθ/2; the bins two away for
            // the reconstruction
            let dir = |offset: isize| {
                let jj = (j as isize + offset).rem_euclid(nd as isize) as usize;
                here[i * nd + jj]
            };
            if explicit_refraction {
                let (theta_lo, theta_hi) = (
                    grid.theta[j] - 0.5 * grid.d_theta,
                    grid.theta[j] + 0.5 * grid.d_theta,
                );
                let (below, above) = (dir(-1), dir(1));
                let (below2, above2) =
                    (second_order.then(|| dir(-2)), second_order.then(|| dir(2)));
                let f_hi = face_flux(
                    self.c_theta(i, theta_hi, p),
                    below2.is_some().then_some(below),
                    here[c],
                    above,
                    above2,
                );
                let f_lo = face_flux(
                    self.c_theta(i, theta_lo, p),
                    below2,
                    below,
                    here[c],
                    above2.is_some().then_some(above),
                );
                *out -= (f_hi - f_lo) * inv_dtheta;
            }
            if self.implicit_frequency_shift {
                continue;
            }
            // Frequencies: faces between bins; nothing enters at the ends
            let freq = |offset: isize| {
                let ii = i as isize + offset;
                (0..nf as isize)
                    .contains(&ii)
                    .then(|| here[ii as usize * nd + j])
            };
            let (lower, upper) = (freq(-1), freq(1));
            let (lower2, upper2) = (
                freq(-2).filter(|_| second_order),
                freq(2).filter(|_| second_order),
            );
            let inv_dsigma = 1.0 / grid.d_sigma[i];
            let theta = grid.theta[j];
            let cs = self.c_sigma(i, theta, p);
            let g_hi = match upper {
                Some(up) => face_flux(
                    0.5 * (cs + self.c_sigma(i + 1, theta, p)),
                    lower.filter(|_| second_order),
                    here[c],
                    up,
                    upper2,
                ),
                None => cs.max(0.0) * here[c],
            };
            let g_lo = match lower {
                Some(lo) => face_flux(
                    0.5 * (cs + self.c_sigma(i - 1, theta, p)),
                    lower2,
                    lo,
                    here[c],
                    upper.filter(|_| second_order),
                ),
                None => cs.min(0.0) * here[c],
            };
            *out -= (g_hi - g_lo) * inv_dsigma;
        }
    }

    /// One Runge–Kutta stage of the propagation over `dt`, node-major:
    /// `out = start·u⁽⁰⁾ + input·u⁽ⁱ⁻¹⁾ + beta·dt·L(u⁽ⁱ⁻¹⁾)`, limited for
    /// positivity, element by element in one pass (see the module docs).
    /// `u0` is the step start, `previous` the stage before (the right-hand
    /// side's argument; `u0` in the first stage).
    ///
    /// The combination takes the operations of `Integrable::combine` in its
    /// order, `((a·base) + c₁x₁) + c₂x₂` with `base` the step start unless its
    /// weight is zero, so it gives the bits of the generic integrators.
    pub(super) fn propagation_stage(
        &self,
        u0: &[f64],
        previous: &[f64],
        stage: ShuOsherStage,
        dt: f64,
        out: &mut [f64],
    ) {
        let (nn, nc) = (self.ops.n_nodes, self.grid.n_components());
        let spectral = !(self.implicit_refraction && self.implicit_frequency_shift);
        let (base, a, with_previous) = if stage.start != 0.0 {
            (u0, stage.start, stage.input != 0.0)
        } else {
            (previous, stage.input, false)
        };
        let beta_dt = stage.beta * dt;
        for_each_chunk(
            out,
            nn * nc,
            || (),
            |_, k, rows| {
                // L(u⁽ⁱ⁻¹⁾) into the element's rows
                self.geographic_element(previous, k, rows);
                if spectral {
                    for (a, row) in rows.chunks_exact_mut(nc).enumerate() {
                        let p = k * nn + a;
                        self.spectral_node(&previous[p * nc..(p + 1) * nc], p, row);
                    }
                }
                let at = k * nn * nc..(k + 1) * nn * nc;
                let (base, previous) = (&base[at.clone()], &previous[at]);
                for (q, x) in rows.iter_mut().enumerate() {
                    let mut value = base[q];
                    if a != 1.0 {
                        value *= a;
                    }
                    if with_previous {
                        value += stage.input * previous[q];
                    }
                    *x = value + beta_dt * *x;
                }
                self.limit_element(k, rows);
            },
        );
    }

    /// [`WaveModel2D::limit_positivity`] on the node-major state.
    pub(super) fn limit_positivity_node_major(&self, n: &mut [f64]) {
        let (nn, nc) = (self.ops.n_nodes, self.grid.n_components());
        for_each_chunk(n, nn * nc, || (), |_, k, rows| self.limit_element(k, rows));
    }

    /// The positivity limiter on element `k`'s node-major rows.
    fn limit_element(&self, k: usize, rows: &mut [f64]) {
        let (nn, nc) = (self.ops.n_nodes, self.grid.n_components());
        let geom = &*self.geom;
        let weights = &self.ops.weights;
        for c in 0..nc {
            let min = (0..nn)
                .map(|a| rows[a * nc + c])
                .fold(f64::INFINITY, f64::min);
            if min >= 0.0 {
                continue;
            }
            let (mut mass, mut area) = (0.0, 0.0);
            for a in 0..nn {
                let w = weights[a] * geom.jacobian(k, a);
                mass += w * rows[a * nc + c];
                area += w;
            }
            let mean = mass / area;
            if mean <= 0.0 {
                (0..nn).for_each(|a| rows[a * nc + c] = 0.0);
                continue;
            }
            let theta = mean / (mean - min);
            for a in 0..nn {
                let x = &mut rows[a * nc + c];
                *x = (mean + theta * (*x - mean)).max(0.0);
            }
        }
    }
}

#[cfg(feature = "simd")]
mod simd {
    use fearless_simd::{Level, dispatch, f64x8, prelude::*};

    use super::{FaceNode, LANES};
    use crate::waves::WaveModel2D;

    /// Largest face count of nodes handled by the vector kernel (P4).
    const MAX_FACE_NODES: usize = 4 * 5;

    impl WaveModel2D {
        /// [`Self::geographic_element_scalar`] with [`LANES`] directions per
        /// vector, lane by lane in the scalar order. False (nothing written)
        /// for orders above P4.
        #[inline(never)] // keeps its large frame out of rayon's recursive split frames
        pub(in crate::waves) fn geographic_element_simd(
            &self,
            n: &[f64],
            k: usize,
            out: &mut [f64],
        ) -> bool {
            let level = Level::new();
            match self.ops.n_nodes {
                1 => false,
                4 => dispatch!(level, simd => self.geographic_lanes::<_, 4>(simd, n, k, out)),
                9 => dispatch!(level, simd => self.geographic_lanes::<_, 9>(simd, n, k, out)),
                16 => dispatch!(level, simd => self.geographic_lanes::<_, 16>(simd, n, k, out)),
                25 => dispatch!(level, simd => self.geographic_lanes::<_, 25>(simd, n, k, out)),
                _ => false,
            }
        }

        #[inline(always)]
        fn geographic_lanes<S: Simd, const NN: usize>(
            &self,
            simd: S,
            n: &[f64],
            k: usize,
            out: &mut [f64],
        ) -> bool {
            let (ops, geom) = (&*self.ops, &*self.geom);
            let nfn = ops.n_face_nodes;
            let np = self.n_points();
            let (nf, nd, nc) = (
                self.grid.n_freq(),
                self.grid.n_dir(),
                self.grid.n_components(),
            );
            let base = k * NN;
            let n_faces = 4 * nfn;
            if n_faces > MAX_FACE_NODES {
                return false;
            }

            // Everything but the direction is shared by the lanes: per node
            // the contravariant vectors, J⁻¹ and the current; per face node
            // the normal, sJ and the neighbour
            let mut metric = [((0.0, 0.0), (0.0, 0.0)); NN];
            let mut jinv = [0.0; NN];
            let mut current = [[0.0; 2]; NN];
            for a in 0..NN {
                metric[a] = geom.contravariant(k, a);
                jinv[a] = geom.jacobian_inv(k, a);
                current[a] = self.current[base + a];
            }
            let empty = FaceNode {
                a: 0,
                normal: (0.0, 0.0),
                surface_jacobian: 0.0,
                neighbour: None,
                open: false,
            };
            let mut faces = [empty; MAX_FACE_NODES];
            for (slot, face) in self.face_nodes(k) {
                faces[slot] = face;
            }
            let zero = f64x8::<S>::splat(simd, 0.0);
            let chunks = nd.div_ceil(LANES);

            for i in 0..nf {
                let cg = &self.cg[i * np..(i + 1) * np];
                for m in 0..chunks {
                    let j0 = m * LANES;
                    let width = LANES.min(nd - j0);
                    let c0 = i * nd + j0;
                    let cos = load(simd, &self.grid.cos_theta[j0..j0 + width]);
                    let sin = load(simd, &self.grid.sin_theta[j0..j0 + width]);
                    let field = |p: usize| p * nc + c0;

                    // Velocities and action at the element's nodes
                    let mut vx = [zero; NN];
                    let mut vy = [zero; NN];
                    let mut action = [zero; NN];
                    for a in 0..NN {
                        let p = base + a;
                        let cg = f64x8::<S>::splat(simd, cg[p]);
                        vx[a] = cg * cos + f64x8::<S>::splat(simd, current[a][0]);
                        vy[a] = cg * sin + f64x8::<S>::splat(simd, current[a][1]);
                        action[a] = load(simd, &n[field(p)..field(p) + width]);
                    }
                    // Volume: −J⁻¹ (D_r F̃_r + D_s F̃_s)
                    let mut fr = [zero; NN];
                    let mut fs = [zero; NN];
                    for a in 0..NN {
                        let ((jrx, jry), (jsx, jsy)) = metric[a];
                        let (jrx, jry) =
                            (f64x8::<S>::splat(simd, jrx), f64x8::<S>::splat(simd, jry));
                        let (jsx, jsy) =
                            (f64x8::<S>::splat(simd, jsx), f64x8::<S>::splat(simd, jsy));
                        fr[a] = (jrx * vx[a] + jry * vy[a]) * action[a];
                        fs[a] = (jsx * vx[a] + jsy * vy[a]) * action[a];
                    }
                    let mut acc = [zero; NN];
                    for a in 0..NN {
                        let (dr, ds) = (
                            &ops.dr_row_major[a * NN..(a + 1) * NN],
                            &ops.ds_row_major[a * NN..(a + 1) * NN],
                        );
                        let mut div = zero;
                        for b in 0..NN {
                            div += f64x8::<S>::splat(simd, dr[b]) * fr[b]
                                + f64x8::<S>::splat(simd, ds[b]) * fs[b];
                        }
                        acc[a] = -div * f64x8::<S>::splat(simd, jinv[a]);
                    }
                    // Faces: lift (F⁻·n − F*) with the upwind flux F*
                    for (slot, face) in faces[..n_faces].iter().enumerate() {
                        let (face_index, fi) = (slot / nfn, slot % nfn);
                        let a = face.a;
                        let p = base + a;
                        let nx = f64x8::<S>::splat(simd, face.normal.0);
                        let ny = f64x8::<S>::splat(simd, face.normal.1);
                        let un_in = vx[a] * nx + vy[a] * ny;
                        let n_in = action[a];
                        let flux = match face.neighbour {
                            Some(q) => {
                                let cg_q = f64x8::<S>::splat(simd, cg[q]);
                                let [u, v] = self.current[q];
                                let wx = cg_q * cos + f64x8::<S>::splat(simd, u);
                                let wy = cg_q * sin + f64x8::<S>::splat(simd, v);
                                let un_out = wx * nx + wy * ny;
                                let n_out = load(simd, &n[field(q)..field(q) + width]);
                                (un_in + un_out)
                                    .simd_ge(zero)
                                    .select(un_in * n_in, un_out * n_out)
                            }
                            None => {
                                let inflow = if face.open {
                                    let mut t = [0.0; LANES];
                                    for (l, t) in t[..width].iter_mut().enumerate() {
                                        *t = self.boundary.action(c0 + l, p);
                                    }
                                    un_in * f64x8::<S>::from_slice(simd, &t)
                                } else {
                                    zero
                                };
                                un_in.simd_ge(zero).select(un_in * n_in, inflow)
                            }
                        };
                        let jump =
                            (un_in * n_in - flux) * f64x8::<S>::splat(simd, face.surface_jacobian);
                        let lift = &ops.lift_row_major[face_index];
                        for b in 0..NN {
                            let l = lift[b * nfn + fi];
                            if l != 0.0 {
                                acc[b] += f64x8::<S>::splat(simd, l)
                                    * jump
                                    * f64x8::<S>::splat(simd, jinv[b]);
                            }
                        }
                    }
                    for (a, acc) in acc.iter().enumerate() {
                        let row = a * nc + c0;
                        out[row..row + width].copy_from_slice(&acc.as_slice()[..width]);
                    }
                }
            }
            true
        }
    }

    /// The first `values.len()` (at most [`LANES`]) lanes from `values`, the
    /// rest zero.
    #[inline(always)]
    fn load<S: Simd>(simd: S, values: &[f64]) -> f64x8<S> {
        if values.len() == LANES {
            f64x8::<S>::from_slice(simd, values)
        } else {
            let mut t = [0.0; LANES];
            t[..values.len()].copy_from_slice(values);
            f64x8::<S>::from_slice(simd, &t)
        }
    }
}

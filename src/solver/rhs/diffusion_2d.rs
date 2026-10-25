//! Conservative DG diffusion helpers for 2D quadrilateral elements.
//!
//! The gradient is taken with the chain rule at every node; the divergence of
//! the diffusive flux in conservative form, `J⁻¹[∂_r(J∇r·F) + ∂_s(J∇s·F)]`,
//! so that the element-integrated diffusion telescopes to the face fluxes on
//! any (also non-affine) element.
//!
//! Both steps are written per element ([`br1_gradient_element`],
//! [`br1_diffusion_element`]) and read the other elements through closures,
//! so kernels can run them in their own element loops (in parallel, or on a
//! subset of the elements for local time stepping) without whole-mesh
//! temporaries.
//!
//! The GLL LIFT of a face is nonzero only in the rows of the face's own
//! nodes (`LIFT_f[i, fi] = δ(i, node(fi))/w_i`: diagonal mass, and the face
//! interpolation just picks the node), so a face correction touches only its
//! own nodes; the kernels apply just those entries.

use crate::mesh::Mesh2D;
use crate::operators::{DGOperators2D, GeometricFactors2D};
use crate::types::ElementIndex;

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct ScalarGradient2D {
    pub dx: f64,
    pub dy: f64,
}

#[cfg(test)]
#[inline]
fn idx(k: ElementIndex, node: usize, n_nodes: usize) -> usize {
    k.as_usize() * n_nodes + node
}

/// BR1 gradients of `N` scalars on element `k` into `out` (`n_nodes`
/// values per scalar).
///
/// The element-local derivative is corrected with
/// `LIFT * sJ/J * n * (u* - u-)`, u* the mean of the two face values, so
/// discontinuous values on adjacent elements contribute to the gradient
/// reconstruction. `value(j, node)` are the scalars at a node of element `k`
/// or of a face neighbour; `boundary_value(k, face, fi, node, interior)` the
/// exterior values on a boundary face, and on a face whose neighbour `j` is
/// not `coupled(j)` (the caller tells the two apart by the mesh's neighbour).
/// `own` is scratch for the element's `n_nodes` values.
///
/// The scalars share each node's `value` and `boundary_value` call (e.g. one
/// division by `h` and one boundary-condition evaluation for both velocity
/// components); each is computed with the same operations as alone.
pub(super) fn br1_gradient_element<const N: usize>(
    k: ElementIndex,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    value: impl Fn(ElementIndex, usize) -> [f64; N],
    boundary_value: impl Fn(ElementIndex, usize, usize, usize, [f64; N]) -> [f64; N],
    coupled: impl Fn(ElementIndex) -> bool,
    own: &mut [[f64; N]],
    out: [&mut [ScalarGradient2D]; N],
) {
    let n_face_nodes = ops.n_face_nodes;
    let k_usize = k.as_usize();
    debug_assert_eq!(own.len(), ops.n_nodes);
    debug_assert!(out.iter().all(|out| out.len() == ops.n_nodes));

    for (j, u) in own.iter_mut().enumerate() {
        *u = value(k, j);
    }

    let n_nodes = ops.n_nodes;
    let rows = ops.dr_row_major.chunks_exact(n_nodes);
    for (i, (dr, ds)) in rows.zip(ops.ds_row_major.chunks_exact(n_nodes)).enumerate() {
        let mut du_dr = [0.0; N];
        let mut du_ds = [0.0; N];

        for ((u_j, &dr), &ds) in own.iter().zip(dr).zip(ds) {
            for c in 0..N {
                du_dr[c] += dr * u_j[c];
                du_ds[c] += ds * u_j[c];
            }
        }

        for c in 0..N {
            let (dx, dy) = geom.transform_derivatives(k_usize, i, du_dr[c], du_ds[c]);
            out[c][i] = ScalarGradient2D { dx, dy };
        }
    }

    for face in 0..4 {
        let face_nodes = &ops.face_nodes[face];
        let neighbor = mesh
            .neighbor(k, face)
            .filter(|nb| coupled(ElementIndex::new(nb.element)));

        for fi in 0..n_face_nodes {
            let node = face_nodes[fi];
            let normal = geom.normal(k_usize, face, fi);
            let scale = geom.lift_scale(k_usize, face, fi, node);
            let value_int = own[node];
            let value_ext = if let Some(neighbor) = &neighbor {
                let neighbor_nodes = &ops.face_nodes[neighbor.face];
                let neighbor_node = neighbor_nodes[n_face_nodes - 1 - fi];
                value(ElementIndex::new(neighbor.element), neighbor_node)
            } else {
                boundary_value(k, face, fi, node, value_int)
            };

            let lift_entry = ops.lift_row_major[face][node * n_face_nodes + fi];
            for c in 0..N {
                let correction = 0.5 * (value_ext[c] - value_int[c]);
                let lift = lift_entry * scale * correction;
                out[c][node].dx += lift * normal.0;
                out[c][node].dy += lift * normal.1;
            }
        }
    }
}

/// Scratch of [`br1_diffusion_element`] for `N` fluxes: four arrays of
/// `n_nodes`.
pub(super) struct DiffusionScratch<const N: usize> {
    flux_x: Vec<[f64; N]>,
    flux_y: Vec<[f64; N]>,
    flux_r: Vec<[f64; N]>,
    flux_s: Vec<[f64; N]>,
}

impl<const N: usize> DiffusionScratch<N> {
    pub(super) fn new(n_nodes: usize) -> Self {
        Self {
            flux_x: vec![[0.0; N]; n_nodes],
            flux_y: vec![[0.0; N]; n_nodes],
            flux_r: vec![[0.0; N]; n_nodes],
            flux_s: vec![[0.0; N]; n_nodes],
        }
    }
}

/// `div(F)` of `N` diffusive fluxes on element `k` into `out` (overwritten),
/// for the fluxes `F = flux(j, node)` (e.g. `coeff · grad(value)`) at the
/// nodes of `k` and of its face neighbours, with the central face flux
/// `(F⁻ + F⁺)/2 · n`. The fluxes share each node's `flux` call; each is
/// computed with the same operations as alone.
///
/// The interface flux is single-valued and opposite on neighbouring
/// elements, so the element-integrated diffusion is conservative across
/// interior faces.
///
/// On a boundary face, and on a face whose neighbour `j` is not
/// `coupled(j)`, the normal flux is `boundary_flux(k, face, fi, normal,
/// F·n)`, from the interior's `F·n` per scalar. It must match the
/// exterior state the gradient took ([`br1_gradient_element`]'s
/// `boundary_value`), or the operator is not dissipative: with the strong
/// form's summation by parts, `Σ u·div(c∇u)` is `−Σ c|∇u|²` plus, on the
/// boundary, `(u* − u)·(F·n) + u·F*` with `u*` the gradient's face value
/// and `F*` this flux. For an exterior state `R u` (linear, `u* = ½(I + R)u`)
/// the flux `F* = ½(I − R)(F·n)` cancels it:
/// - `R = −I` (zero Dirichlet): the interior flux;
/// - `R = I` (zero gradient, a copy): no flux;
/// - a free-slip wall's mirror of a velocity, `R = I − 2nnᵀ`: the normal
///   stress only, `F* = n (n·(F·n))`.
///
/// The interior flux at a mirrored wall made the 3D viscosity grow with an
/// e-folding of 1.5 min at ν = 10 m²/s on the Frøya coastline mesh (TODO
/// P1.3). An inhomogeneous exterior state (`R u + g`, e.g. an open
/// boundary's external data) adds the forcing of `g` only.
pub(super) fn br1_diffusion_element<const N: usize>(
    k: ElementIndex,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    flux: impl Fn(ElementIndex, usize) -> [(f64, f64); N],
    boundary_flux: impl Fn(ElementIndex, usize, usize, (f64, f64), [f64; N]) -> [f64; N],
    coupled: impl Fn(ElementIndex) -> bool,
    scratch: &mut DiffusionScratch<N>,
    out: [&mut [f64]; N],
) {
    let n_face_nodes = ops.n_face_nodes;
    let k_usize = k.as_usize();
    debug_assert!(out.iter().all(|out| out.len() == ops.n_nodes));
    let DiffusionScratch {
        flux_x,
        flux_y,
        flux_r,
        flux_s,
    } = scratch;

    // Contravariant fluxes J∇r·F and J∇s·F
    for i in 0..ops.n_nodes {
        let fluxes = flux(k, i);
        let ((ar_x, ar_y), (as_x, as_y)) = geom.contravariant(k_usize, i);
        for (c, (fx, fy)) in fluxes.into_iter().enumerate() {
            flux_x[i][c] = fx;
            flux_y[i][c] = fy;
            flux_r[i][c] = ar_x * fx + ar_y * fy;
            flux_s[i][c] = as_x * fx + as_y * fy;
        }
    }

    let n_nodes = ops.n_nodes;
    let rows = ops.dr_row_major.chunks_exact(n_nodes);
    for (i, (dr, ds)) in rows.zip(ops.ds_row_major.chunks_exact(n_nodes)).enumerate() {
        let mut dfr_dr = [0.0; N];
        let mut dfs_ds = [0.0; N];

        for (((fr, fs), &dr), &ds) in flux_r.iter().zip(flux_s.iter()).zip(dr).zip(ds) {
            for c in 0..N {
                dfr_dr[c] += dr * fr[c];
                dfs_ds[c] += ds * fs[c];
            }
        }

        let jacobian_inv = geom.jacobian_inv(k_usize, i);
        for c in 0..N {
            out[c][i] = jacobian_inv * (dfr_dr[c] + dfs_ds[c]);
        }
    }

    for face in 0..4 {
        let face_nodes = &ops.face_nodes[face];
        let neighbor = mesh
            .neighbor(k, face)
            .filter(|nb| coupled(ElementIndex::new(nb.element)));

        for fi in 0..n_face_nodes {
            let node = face_nodes[fi];
            let normal = geom.normal(k_usize, face, fi);
            let lift_scale = geom.lift_scale(k_usize, face, fi, node);
            let lift_entry = ops.lift_row_major[face][node * n_face_nodes + fi];
            let exterior = neighbor.as_ref().map(|neighbor| {
                let neighbor_nodes = &ops.face_nodes[neighbor.face];
                flux(
                    ElementIndex::new(neighbor.element),
                    neighbor_nodes[n_face_nodes - 1 - fi],
                )
            });

            let flux_int_n: [f64; N] =
                std::array::from_fn(|c| flux_x[node][c] * normal.0 + flux_y[node][c] * normal.1);
            let flux_star_n = match exterior {
                Some(fluxes) => std::array::from_fn(|c| {
                    0.5 * (flux_int_n[c] + fluxes[c].0 * normal.0 + fluxes[c].1 * normal.1)
                }),
                None => boundary_flux(k, face, fi, normal, flux_int_n),
            };
            for c in 0..N {
                out[c][node] += lift_entry * lift_scale * (flux_star_n[c] - flux_int_n[c]);
            }
        }
    }
}

/// [`br1_gradient_element`] of every element, for `values` given at every
/// mesh node (whole-mesh arrays: the reference the kernels are tested
/// against), the faces to elements not `coupled` handled by
/// `boundary_value`.
#[cfg(test)]
pub(super) fn compute_br1_gradient_2d<F>(
    values: &[f64],
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    boundary_value: F,
    coupled: impl Fn(ElementIndex) -> bool,
) -> Vec<ScalarGradient2D>
where
    F: Fn(ElementIndex, usize, usize, usize, f64) -> f64,
{
    let n_nodes = ops.n_nodes;
    debug_assert_eq!(values.len(), mesh.n_elements * n_nodes);

    let mut gradients = vec![ScalarGradient2D::default(); values.len()];
    let mut own = vec![[0.0]; n_nodes];
    for (k, out) in ElementIndex::iter(mesh.n_elements).zip(gradients.chunks_exact_mut(n_nodes)) {
        br1_gradient_element(
            k,
            mesh,
            ops,
            geom,
            |j, node| [values[idx(j, node, n_nodes)]],
            |k, face, fi, node, [interior]| [boundary_value(k, face, fi, node, interior)],
            &coupled,
            &mut own,
            [out],
        );
    }
    gradients
}

#[cfg(test)]
mod tests {
    use crate::operators::DGOperators2D;

    /// The kernels apply only the LIFT entry of each face node's own row:
    /// exact only if every other entry is zero.
    #[test]
    fn lift_touches_only_the_face_nodes() {
        for order in 1..=5 {
            let ops = DGOperators2D::new(order);
            let nf = ops.n_face_nodes;
            for face in 0..4 {
                for fi in 0..nf {
                    let node = ops.face_nodes[face][fi];
                    for i in 0..ops.n_nodes {
                        let entry = ops.lift_row_major[face][i * nf + fi];
                        assert_eq!(
                            entry == 0.0,
                            i != node,
                            "P{order}, face {face}, ({i}, {fi})"
                        );
                    }
                }
            }
        }
    }
}

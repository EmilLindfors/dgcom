//! Horizontal eddy viscosity of the 2D SWE momentum, `∇·(νh∇u)` and
//! `∇·(νh∇v)`, discretised with BR1 (Bassi & Rebay 1997; Hesthaven &
//! Warburton 2008, §7.2).
//!
//! Two passes, both element by element:
//! 1. [`ViscousTerm::gradients`]: the BR1 gradients of u = hu/h and v = hv/h
//!    of an element (central face values; the boundary condition's ghost
//!    velocity on boundary faces), into a [`ViscousWorkspace`].
//! 2. [`ViscousTerm::add`], in the RHS kernel's element loop: the divergence
//!    of νh∇u from the element's own and its face neighbours' gradients, with
//!    the central face flux.
//!
//! So the term of element k reads the state beyond its face neighbours j:
//! j's gradient at the shared face nodes reads j's neighbours. With the
//! diagonal GLL LIFT (a face's lift touches only its own nodes) it reads them
//! only at the corner nodes, i.e. the elements across the two faces of j that
//! meet the shared face. The whole-mesh RHS runs pass 1 over every element
//! first; the element-subset RHS of local time stepping runs it over the
//! listed elements and their face neighbours, each at its own time, and the
//! multirate stepper follows the wider stencil
//! ([`RhsStencil::FacesAndCorners`](crate::time::RhsStencil::FacesAndCorners)).
//!
//! The face flux is the same on both sides of every interior face, so the
//! viscous term conserves momentum; it adds nothing to the mass equation.
//!
//! It acts on the water only: an element with a node shallower than the
//! subcell depth (`SWEFormulation2D::WetDry`'s shoreline elements, which run
//! the subcell update) or at or below the viscosity's `h_min` gets no viscous
//! term, and a face between it and a wet element is an edge of the viscous
//! domain: the wet side's gradient copies its own velocity there (zero
//! gradient) and no viscous flux crosses. That pair is dissipative (see
//! `diffusion_2d::br1_diffusion_element`) and conserves momentum. Before, the
//! shoreline elements took the term with their dry nodes' velocity at zero,
//! and the divergence of the wet nodes' stress put momentum into dry nodes:
//! at ν = 10 m²/s it pushed films 4 m up a cliff shore of the Frøya coastline
//! mesh within 2 h (TODO F.1, the Kattholmen ν sweep).

use crate::boundary::SWEBoundaryCondition2D;
use crate::mesh::Mesh2D;
use crate::operators::{DGOperators2D, GeometricFactors2D};
use crate::solver::SWESolution2D;
use crate::solver::core::blocks::{Pooled, for_each_block};
use crate::source::swe_2d::viscosity::HorizontalViscosity2D;
use crate::types::ElementIndex;

use super::diffusion_2d::{
    DiffusionScratch, ScalarGradient2D, br1_diffusion_element, br1_gradient_element,
};
use super::swe_2d::{SWE2DRhsConfig, SWEFormulation2D, boundary_state};

/// Per-element scratch of the viscous term (part of the RHS kernel's
/// per-thread element workspace).
pub(super) struct ViscousScratch {
    own: Vec<[f64; 2]>,
    diffusion: DiffusionScratch<2>,
    diff: [Vec<f64>; 2],
}

impl ViscousScratch {
    pub(super) fn new(n_nodes: usize) -> Self {
        Self {
            own: vec![[0.0; 2]; n_nodes],
            diffusion: DiffusionScratch::new(n_nodes),
            diff: [vec![0.0; n_nodes], vec![0.0; n_nodes]],
        }
    }
}

/// Velocity gradients of every mesh node (valid where pass 1 ran), and the
/// element-neighbourhood scratch of the subset RHS. Reused between RHS
/// evaluations.
#[derive(Default)]
pub(super) struct ViscousWorkspace {
    pub(super) grad_u: Vec<ScalarGradient2D>,
    pub(super) grad_v: Vec<ScalarGradient2D>,
    /// The listed elements and their face neighbours, distinct
    pub(super) neighbourhood: Vec<u32>,
    stamp: Vec<u32>,
    pass: u32,
}

impl ViscousWorkspace {
    /// Gradient storage for `n_total` nodes.
    pub(super) fn resize(&mut self, n_total: usize) {
        self.grad_u.resize(n_total, ScalarGradient2D::default());
        self.grad_v.resize(n_total, ScalarGradient2D::default());
    }

    /// `elements` and their face neighbours into `self.neighbourhood`, each
    /// once, without sorting (an element is taken when its stamp is not the
    /// current pass).
    pub(super) fn collect_neighbourhood(&mut self, mesh: &Mesh2D, elements: &[u32]) {
        if self.stamp.len() != mesh.n_elements || self.pass == u32::MAX {
            self.stamp.clear();
            self.stamp.resize(mesh.n_elements, 0);
            self.pass = 0;
        }
        self.pass += 1;
        self.neighbourhood.clear();
        let (stamp, pass, list) = (&mut self.stamp, self.pass, &mut self.neighbourhood);
        let mut take = |k: usize| {
            if stamp[k] != pass {
                stamp[k] = pass;
                list.push(k as u32);
            }
        };
        for &k in elements {
            take(k as usize);
            let k_idx = ElementIndex::new(k as usize);
            for face in 0..4 {
                if let Some(nb) = mesh.neighbor(k_idx, face) {
                    take(nb.element);
                }
            }
        }
    }
}

/// The viscous term of one RHS evaluation (see the [module documentation](self)).
pub(super) struct ViscousTerm<'a, 'c, BC: SWEBoundaryCondition2D> {
    q: &'a SWESolution2D,
    mesh: &'a Mesh2D,
    ops: &'a DGOperators2D,
    geom: &'a GeometricFactors2D,
    config: &'a SWE2DRhsConfig<'c, BC>,
    visc: &'c HorizontalViscosity2D,
    /// Depth (m) below which a node takes its element out of the viscous
    /// domain: the subcell depth under `WetDry`, else none
    subcell_depth: f64,
}

impl<'a, 'c, BC: SWEBoundaryCondition2D> ViscousTerm<'a, 'c, BC> {
    /// `None` when `config` has no viscosity.
    pub(super) fn new(
        q: &'a SWESolution2D,
        mesh: &'a Mesh2D,
        ops: &'a DGOperators2D,
        geom: &'a GeometricFactors2D,
        config: &'a SWE2DRhsConfig<'c, BC>,
    ) -> Option<Self> {
        let visc = config.viscosity?;
        let subcell_depth = match config.formulation {
            SWEFormulation2D::WetDry => config.subcell_depth.unwrap_or(config.h_dry),
            _ => 0.0,
        };
        Some(Self {
            q,
            mesh,
            ops,
            geom,
            config,
            visc,
            subcell_depth,
        })
    }

    /// Whether element `k` is in the viscous domain: every node deeper than
    /// the viscosity's `h_min` and not shallower than the subcell depth (see
    /// the [module documentation](self)).
    pub(super) fn in_water(&self, k: ElementIndex) -> bool {
        in_viscous_domain(self.q, self.visc, self.subcell_depth, k)
    }

    /// Velocity of the boundary condition's exterior state at boundary node
    /// `node` (face node `fi` of `face`) of element `k`, at `time`; zero
    /// where its depth is at or below the viscosity's `h_min`.
    fn boundary_velocity(
        &self,
        time: f64,
        k: ElementIndex,
        face: usize,
        fi: usize,
        node: usize,
    ) -> [f64; 2] {
        let normal = self.geom.normal(k.as_usize(), face, fi);
        let ghost = boundary_state(
            self.q,
            self.mesh,
            self.ops,
            self.config,
            time,
            k,
            face,
            node,
            normal,
        )
        .state();
        let h_min = self.visc.h_min;
        if ghost.h <= h_min {
            return [0.0; 2];
        }
        let h_safe = ghost.h.max(h_min);
        [ghost.hu / h_safe, ghost.hv / h_safe]
    }

    /// Pass 1: the BR1 gradients of u and v of element `k` into its rows
    /// `grad_u`, `grad_v` (`n_nodes` each), with the boundary condition
    /// evaluated at `time`; zero outside the viscous domain.
    pub(super) fn gradients(
        &self,
        k: usize,
        time: f64,
        scratch: &mut ViscousScratch,
        grad_u: &mut [ScalarGradient2D],
        grad_v: &mut [ScalarGradient2D],
    ) {
        let k = ElementIndex::new(k);
        if !self.in_water(k) {
            grad_u.fill(ScalarGradient2D::default());
            grad_v.fill(ScalarGradient2D::default());
            return;
        }
        br1_gradient_element(
            k,
            self.mesh,
            self.ops,
            self.geom,
            |j, node| node_velocity(self.q, self.visc, j, node),
            |k, face, fi, node, interior| {
                // The edge of the viscous domain: zero gradient
                if self.mesh.neighbor(k, face).is_some() {
                    interior
                } else {
                    self.boundary_velocity(time, k, face, fi, node)
                }
            },
            |j| self.in_water(j),
            &mut scratch.own,
            [grad_u, grad_v],
        );
    }

    /// Pass 2: `∇·(νh∇u)`, `∇·(νh∇v)` of element `k` added to its rows `hu`,
    /// `hv`, from the gradients `grad` of pass 1 (whole-mesh arrays, valid for
    /// `k` and its face neighbours).
    pub(super) fn add(
        &self,
        k: usize,
        grad: [&[ScalarGradient2D]; 2],
        scratch: &mut ViscousScratch,
        hu: &mut [f64],
        hv: &mut [f64],
    ) {
        if !self.in_water(ElementIndex::new(k)) {
            return;
        }
        let n_nodes = self.ops.n_nodes;
        // νh at a node; zero where (nearly) dry
        let coefficient = |j: ElementIndex, node: usize| {
            let flat = j.as_usize() * n_nodes + node;
            let h = self.q.get_state(j, node).h;
            if h <= self.visc.h_min {
                return 0.0;
            }
            let delta =
                HorizontalViscosity2D::filter_width(self.geom.area[j.as_usize()], self.ops.order);
            let (gu, gv) = (grad[0][flat], grad[1][flat]);
            self.visc
                .compute_viscosity(gu.dx, gu.dy, gv.dx, gv.dy, delta)
                * h
        };
        let ViscousScratch {
            diffusion, diff, ..
        } = scratch;
        let [diff_u, diff_v] = diff;
        br1_diffusion_element(
            ElementIndex::new(k),
            self.mesh,
            self.ops,
            self.geom,
            |j, node| {
                let coeff = coefficient(j, node).max(0.0);
                let flat = j.as_usize() * n_nodes + node;
                let (gu, gv) = (grad[0][flat], grad[1][flat]);
                [
                    (coeff * gu.dx, coeff * gu.dy),
                    (coeff * gv.dx, coeff * gv.dy),
                ]
            },
            // The fluxes that match the gradients' exterior states (see
            // `br1_diffusion_element`): the edge of the viscous domain copies
            // the velocity, so it passes nothing; a wall mirrors it, so it
            // passes the normal stress only; other boundaries prescribe an
            // exterior state and take the interior flux
            |k, face, _fi, (nx, ny), [fu, fv]| {
                if self.mesh.neighbor(k, face).is_some() {
                    return [0.0; 2];
                }
                let tag = self.mesh.boundary_tag(k, face);
                let wall = self
                    .config
                    .bc
                    .is_wall(tag)
                    .unwrap_or(matches!(tag, None | Some(crate::mesh::BoundaryTag::Wall)));
                if wall {
                    let normal = nx * fu + ny * fv;
                    [nx * normal, ny * normal]
                } else {
                    [fu, fv]
                }
            },
            |j| self.in_water(j),
            diffusion,
            [diff_u, diff_v],
        );
        for (row, diff) in [hu, hv].into_iter().zip(diff.iter()) {
            for (r, d) in row.iter_mut().zip(diff) {
                *r += d;
            }
        }
    }
}

/// Whether element `k` of `q` is in the viscous domain of `visc` with the
/// subcell depth `subcell_depth`: every node deeper than `visc.h_min` and
/// not shallower than `subcell_depth`.
pub(super) fn in_viscous_domain(
    q: &SWESolution2D,
    visc: &HorizontalViscosity2D,
    subcell_depth: f64,
    k: ElementIndex,
) -> bool {
    q.element_h(k)
        .iter()
        .all(|&h| h > visc.h_min && h >= subcell_depth)
}

/// Velocity at node `i` of element `k`; zero where the depth is at or below
/// the viscosity's `h_min`.
#[inline]
fn node_velocity(
    q: &SWESolution2D,
    visc: &HorizontalViscosity2D,
    k: ElementIndex,
    i: usize,
) -> [f64; 2] {
    let state = q.get_state(k, i);
    if state.h > visc.h_min {
        let h_safe = state.h.max(visc.h_min);
        [state.hu / h_safe, state.hv / h_safe]
    } else {
        [0.0; 2]
    }
}

/// The largest viscosity ν (m²/s) of every element's nodes into `largest`
/// (`[element]`), for the time step: the background, plus Smagorinsky's from
/// the strain of `q`'s BR1 velocity gradients at the nodes deeper than
/// `h_min` (the others carry no viscous flux).
///
/// The gradients are the RHS's except on boundary faces, which take the
/// interior value (no jump): the boundary condition's exterior state needs a
/// time, and a time step is chosen before the stages' times are known. In
/// parallel with the `parallel` feature, the same result either way.
pub fn largest_viscosity_swe_2d(
    largest: &mut [f64],
    q: &SWESolution2D,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    visc: &HorizontalViscosity2D,
) {
    assert_eq!(largest.len(), mesh.n_elements, "one value per element");
    if !visc.is_strain_dependent() {
        largest.fill(visc.background);
        return;
    }
    struct Scratch {
        own: Vec<[f64; 2]>,
        gradient: [Vec<ScalarGradient2D>; 2],
    }
    let nn = ops.n_nodes;
    for_each_block(
        mesh.n_elements,
        [largest],
        || {
            Pooled::take(
                |s: &Scratch| s.own.len() == nn,
                || Scratch {
                    own: vec![[0.0; 2]; nn],
                    gradient: std::array::from_fn(|_| vec![ScalarGradient2D::default(); nn]),
                },
            )
        },
        |scratch, k, [out]| {
            let k = ElementIndex::new(k);
            let Scratch { own, gradient } = &mut **scratch;
            let [grad_u, grad_v] = gradient;
            br1_gradient_element(
                k,
                mesh,
                ops,
                geom,
                |j, node| node_velocity(q, visc, j, node),
                |_, _, _, _, interior| interior,
                |_| true,
                own,
                [grad_u, grad_v],
            );
            let delta = HorizontalViscosity2D::filter_width(geom.area[k.as_usize()], ops.order);
            out[0] = (0..nn)
                .filter(|&i| q.get_state(k, i).h > visc.h_min)
                .map(|i| {
                    let (gu, gv) = (gradient[0][i], gradient[1][i]);
                    visc.compute_viscosity(gu.dx, gu.dy, gv.dx, gv.dy, delta)
                })
                .fold(visc.background, f64::max);
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::swe_2d::viscosity::strain_rate_magnitude;

    /// On a linear shear `u = a y + b x`, `v = c x` the BR1 gradients are
    /// exact (continuous across faces, and boundary faces have no jump), so
    /// the largest ν is the background plus `(C_s Δ)²|S|` with the constant
    /// strain, on every element and order. Dry nodes carry none.
    #[test]
    fn largest_viscosity_is_exact_on_a_linear_shear() {
        let (a, b, c) = (0.3, -0.1, 0.05);
        let visc = HorizontalViscosity2D::smagorinsky(0.4).with_background(0.25);
        let strain = strain_rate_magnitude(b, a, c, 0.0);
        for order in 1..=3 {
            let mesh = Mesh2D::uniform_rectangle(0.0, 300.0, 0.0, 200.0, 3, 2);
            let ops = DGOperators2D::new(order);
            let geom = GeometricFactors2D::compute(&mesh, &ops);
            let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
            q.set_from_functions(&mesh, &ops, |_, _| 2.0, |x, y| a * y + b * x, |x, _| c * x);
            let mut largest = vec![0.0; mesh.n_elements];
            largest_viscosity_swe_2d(&mut largest, &q, &mesh, &ops, &geom, &visc);
            // 100 m × 100 m elements: Δ = 100/N
            let delta = 100.0 / order as f64;
            let expected = 0.25 + (0.4 * delta).powi(2) * strain;
            for (k, nu) in largest.iter().enumerate() {
                assert!(
                    (nu - expected).abs() < 1e-11 * expected,
                    "P{order}, element {k}: {nu} against {expected}"
                );
            }

            // A dry element has the background only
            for var in 0..3 {
                q.data[var][..ops.n_nodes].fill(0.0);
            }
            largest_viscosity_swe_2d(&mut largest, &q, &mesh, &ops, &geom, &visc);
            assert_eq!(largest[0], 0.25, "P{order}: dry element");
        }
    }

    /// Without Smagorinsky's part the largest ν is the background, with no
    /// gradient pass.
    #[test]
    fn largest_viscosity_of_a_constant_is_the_constant() {
        let mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 2, 2);
        let ops = DGOperators2D::new(2);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        let mut largest = vec![0.0; mesh.n_elements];
        let visc = HorizontalViscosity2D::constant(3.0);
        largest_viscosity_swe_2d(&mut largest, &q, &mesh, &ops, &geom, &visc);
        assert_eq!(largest, vec![3.0; 4]);
    }
}

//! A horizontal diffusivity that varies in space, for the particles' random
//! walk (TODO F.2): from the circulation model's own Smagorinsky viscosity,
//! or any nodal field.
//!
//! # The walk with a variable `K`
//!
//! The particle form of `∂c/∂t = ∇·(K∇c)` is the Itô process
//!
//! ```text
//! dx = (u + ∇K) dt + √(2K) dW,
//! ```
//!
//! stepped by Euler–Maruyama with `K` and `∇K` at the start of the step
//! (Visser 1997 in 1D; Spivakovskaya et al. 2007 in 2D). Without the drift
//! `∇K` particles collect where `K` is small, and a well-mixed population
//! unmixes. The drift needs a gradient that exists, so [`HorizontalDiffusivityField`]
//! makes `K` continuous: the nodes that coincide across element faces
//! (periodic faces too) take their mean, and `∇K` is the element polynomial's
//! gradient, bounded and jumping at faces like Visser's piecewise-linear
//! profiles. The step must resolve `K`'s variation, `√(2KΔt) ≪ K/|∇K|`.
//!
//! A 3D flow's `K` along each σ-layer is held at the layer centres and linear
//! in σ between them, like `u`; the drift is the gradient along the
//! σ-surface. Where the depth varies the trackers add the drift `K∇D/D` as
//! well (Dimou & Adams 1993; see the [particles module docs](super)).
//!
//! # Smagorinsky's diffusivity
//!
//! [`HorizontalDiffusivity::smagorinsky_2d`] and
//! [`HorizontalDiffusivity::smagorinsky_3d`] give `K = ν/Pr_t` from the
//! model's eddy viscosity `ν = ν₀ + (C_s Δ)²|S|` ([`HorizontalViscosity2D`]),
//! with `|S|` the strain rate of the depth-mean velocity (2D) or of each
//! layer's velocity (3D), from its element polynomial, and `Δ = √(area)/N` the
//! node spacing. `Pr_t` is the turbulent Prandtl (Schmidt) number: 1 hands
//! the momentum's mixing to the particles as it is.
//!
//! # References
//!
//! - Visser, A. W. (1997). Using random walk models to simulate the vertical
//!   distribution of particles in a turbulent water column. *Mar. Ecol.
//!   Prog. Ser.* 158, 275–281.
//! - Spivakovskaya, D., Heemink, A. W. & Deleersnijder, E. (2007). The
//!   backward Itô method for the Lagrangian simulation of transport
//!   processes with large space variations of the diffusivity. *Ocean Sci.*
//!   3, 525–535.
//! - Dimou, K. N. & Adams, E. E. (1993). A random-walk, particle tracking
//!   model for well-mixed estuaries and coastal waters. *Estuar. Coast.
//!   Shelf Sci.* 37, 99–110.
//! - Smagorinsky, J. (1963). General circulation experiments with the
//!   primitive equations. *Mon. Weather Rev.* 91, 99–164.

use super::velocity_3d::bracket;
use super::{ParticleVelocity2D, ParticleVelocity3D};
use crate::mesh::Mesh2D;
use crate::operators::{DGOperators2D, GeometricFactors2D};
use crate::solver::state::Solution3D;
use crate::source::HorizontalViscosity2D;
use crate::types::ElementIndex;
use crate::vertical::SigmaGrid;

/// Builds [`HorizontalDiffusivityField`]s on one mesh: the mesh, basis and
/// geometry, and which nodes coincide (see the [module docs](self)).
pub struct HorizontalDiffusivity<'a> {
    ops: &'a DGOperators2D,
    geom: &'a GeometricFactors2D,
    /// For every node `k·n + i`, the first node of its group of coinciding
    /// nodes, and the group's size there
    group: Vec<usize>,
    group_size: Vec<usize>,
}

/// Root of `i` in the union-find `parent`, halving the path.
fn root(parent: &mut [usize], mut i: usize) -> usize {
    while parent[i] != i {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    i
}

impl<'a> HorizontalDiffusivity<'a> {
    /// The builder on `mesh` with basis `ops` and geometry `geom`.
    pub fn new(mesh: &Mesh2D, ops: &'a DGOperators2D, geom: &'a GeometricFactors2D) -> Self {
        let (nn, nf) = (ops.n_nodes, ops.n_face_nodes);
        let n_total = mesh.n_elements * nn;
        let mut parent: Vec<usize> = (0..n_total).collect();
        for k in ElementIndex::iter(mesh.n_elements) {
            for face in 0..4 {
                let Some(neighbour) = mesh.neighbor(k, face) else {
                    continue;
                };
                // The neighbour's face runs the other way
                for i in 0..nf {
                    let a = k.as_usize() * nn + ops.face_nodes[face][i];
                    let b = neighbour.element * nn + ops.face_nodes[neighbour.face][nf - 1 - i];
                    let (ra, rb) = (root(&mut parent, a), root(&mut parent, b));
                    if ra != rb {
                        parent[ra.max(rb)] = ra.min(rb);
                    }
                }
            }
        }
        let group: Vec<usize> = (0..n_total).map(|i| root(&mut parent, i)).collect();
        let mut group_size = vec![0; n_total];
        for &g in &group {
            group_size[g] += 1;
        }
        Self {
            ops,
            geom,
            group,
            group_size,
        }
    }

    /// The field of the nodal values `values` (m²/s, `[level][element][node]`)
    /// on the σ-levels `levels` (increasing; empty for one depth-independent
    /// level): made continuous, negative values set to 0, with its gradient.
    pub fn from_nodal(&self, mut values: Vec<f64>, levels: Vec<f64>) -> HorizontalDiffusivityField {
        let n_total = self.group.len();
        let n_levels = levels.len().max(1);
        assert_eq!(
            values.len(),
            n_total * n_levels,
            "one value per node and level"
        );
        assert!(
            levels.windows(2).all(|w| w[0] < w[1]),
            "levels must increase"
        );
        let mut sum = vec![0.0; n_total];
        for level in values.chunks_exact_mut(n_total) {
            sum.fill(0.0);
            for (i, &v) in level.iter().enumerate() {
                sum[self.group[i]] += v;
            }
            for (i, v) in level.iter_mut().enumerate() {
                let g = self.group[i];
                *v = (sum[g] / self.group_size[g] as f64).max(0.0);
            }
        }
        let nn = self.ops.n_nodes;
        let mut gradient = vec![[0.0; 2]; values.len()];
        for (level, grad) in values
            .chunks_exact(n_total)
            .zip(gradient.chunks_exact_mut(n_total))
        {
            for (k, (kv, kg)) in level
                .chunks_exact(nn)
                .zip(grad.chunks_exact_mut(nn))
                .enumerate()
            {
                for (i, g) in kg.iter_mut().enumerate() {
                    let (mut dr, mut ds) = (0.0, 0.0);
                    for (j, &v) in kv.iter().enumerate() {
                        dr += self.ops.dr[(i, j)] * v;
                        ds += self.ops.ds[(i, j)] * v;
                    }
                    let (dx, dy) = self.geom.transform_derivatives(k, i, dr, ds);
                    *g = [dx, dy];
                }
            }
        }
        HorizontalDiffusivityField {
            n_nodes: nn,
            n_total,
            levels,
            values,
            gradient,
        }
    }

    /// `ν/Pr_t` at every node of the velocity `(u, v)` (`[element][node]`
    /// each), from its element polynomial's strain.
    fn smagorinsky_level(
        &self,
        viscosity: &HorizontalViscosity2D,
        prandtl: f64,
        u: impl Fn(usize) -> f64,
        v: impl Fn(usize) -> f64,
        out: &mut [f64],
    ) {
        let nn = self.ops.n_nodes;
        for (k, out) in out.chunks_exact_mut(nn).enumerate() {
            let delta = HorizontalViscosity2D::filter_width(self.geom.area[k], self.ops.order);
            for (i, o) in out.iter_mut().enumerate() {
                let (mut ur, mut us, mut vr, mut vs) = (0.0, 0.0, 0.0, 0.0);
                for j in 0..nn {
                    let (dr, ds) = (self.ops.dr[(i, j)], self.ops.ds[(i, j)]);
                    let (uj, vj) = (u(k * nn + j), v(k * nn + j));
                    ur += dr * uj;
                    us += ds * uj;
                    vr += dr * vj;
                    vs += ds * vj;
                }
                let (ux, uy) = self.geom.transform_derivatives(k, i, ur, us);
                let (vx, vy) = self.geom.transform_derivatives(k, i, vr, vs);
                *o = viscosity.compute_viscosity(ux, uy, vx, vy, delta) / prandtl;
            }
        }
    }

    /// `K = ν/Pr_t` of the depth-mean velocity `(u, v)` (`[element][node]`)
    /// with the eddy viscosity `viscosity` (see the [module docs](self)).
    pub fn smagorinsky_2d(
        &self,
        u: &[f64],
        v: &[f64],
        viscosity: &HorizontalViscosity2D,
        prandtl: f64,
    ) -> HorizontalDiffusivityField {
        assert!(prandtl > 0.0, "the turbulent Prandtl number is positive");
        let mut values = vec![0.0; self.group.len()];
        self.smagorinsky_level(viscosity, prandtl, |n| u[n], |n| v[n], &mut values);
        self.from_nodal(values, Vec::new())
    }

    /// `K = ν/Pr_t` on every σ-layer of `state`, from the layer's velocity,
    /// at the layer centres of `sigma` (see the [module docs](self)).
    pub fn smagorinsky_3d(
        &self,
        state: &Solution3D,
        sigma: &SigmaGrid,
        viscosity: &HorizontalViscosity2D,
        prandtl: f64,
    ) -> HorizontalDiffusivityField {
        assert!(prandtl > 0.0, "the turbulent Prandtl number is positive");
        let n_total = self.group.len();
        let n_levels = state.n_levels;
        let mut values = vec![0.0; n_total * n_levels];
        for (l, level) in values.chunks_exact_mut(n_total).enumerate() {
            self.smagorinsky_level(
                viscosity,
                prandtl,
                |n| state.u[n * n_levels + l],
                |n| state.v[n * n_levels + l],
                level,
            );
        }
        self.from_nodal(values, sigma.sigma_rho().to_vec())
    }
}

/// A continuous nodal horizontal diffusivity and its gradient, on one level
/// or on σ-levels (see the [module docs](self)).
#[derive(Clone, Debug, PartialEq)]
pub struct HorizontalDiffusivityField {
    n_nodes: usize,
    n_total: usize,
    levels: Vec<f64>,
    /// `K` (m²/s), `[level][element][node]`
    values: Vec<f64>,
    /// `∇K` (m/s), the same layout
    gradient: Vec<[f64; 2]>,
}

impl HorizontalDiffusivityField {
    /// The nodal values (m²/s), `[level][element][node]`.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// The nodal gradients (m/s), `[level][element][node]`.
    pub fn gradients(&self) -> &[[f64; 2]] {
        &self.gradient
    }

    /// `K` and `∇K` of one level at the point.
    fn level(&self, l: usize, element: ElementIndex, weights: &[f64]) -> (f64, [f64; 2]) {
        let start = l * self.n_total + element.as_usize() * self.n_nodes;
        let (mut k, mut g) = (0.0, [0.0; 2]);
        for (j, w) in weights.iter().enumerate() {
            k += w * self.values[start + j];
            g[0] += w * self.gradient[start + j][0];
            g[1] += w * self.gradient[start + j][1];
        }
        (k, g)
    }

    /// `K` (m²/s, ≥ 0) and `∇K` (m/s) at σ-level `sigma` of the point of
    /// element `element` where the nodal basis takes the values `weights`
    /// (linear in σ between the levels, held beyond them).
    pub fn at(&self, element: ElementIndex, weights: &[f64], sigma: f64) -> (f64, [f64; 2]) {
        if self.levels.len() < 2 {
            let (k, g) = self.level(0, element, weights);
            return (k.max(0.0), g);
        }
        let (l, a) = bracket(&self.levels, sigma);
        let (k0, g0) = self.level(l, element, weights);
        if a == 0.0 {
            return (k0.max(0.0), g0);
        }
        let (k1, g1) = self.level(l + 1, element, weights);
        (
            ((1.0 - a) * k0 + a * k1).max(0.0),
            [0, 1].map(|d| (1.0 - a) * g0[d] + a * g1[d]),
        )
    }
}

/// One or two diffusivity fields, linear in time between them and held
/// outside them.
#[derive(Clone, Copy, Debug)]
pub struct DiffusivityInTime<'a> {
    before: (f64, &'a HorizontalDiffusivityField),
    after: (f64, &'a HorizontalDiffusivityField),
}

impl<'a> DiffusivityInTime<'a> {
    /// One field, the same at all times.
    pub fn steady(field: &'a HorizontalDiffusivityField) -> Self {
        Self {
            before: (0.0, field),
            after: (0.0, field),
        }
    }

    /// Linear in time between `f0` at `t0` and `f1` at `t1 > t0`.
    pub fn between(
        t0: f64,
        f0: &'a HorizontalDiffusivityField,
        t1: f64,
        f1: &'a HorizontalDiffusivityField,
    ) -> Self {
        assert!(t1 > t0, "fields must be in time order: {t0} → {t1}");
        assert_eq!(f0.values.len(), f1.values.len());
        Self {
            before: (t0, f0),
            after: (t1, f1),
        }
    }

    /// `K` and `∇K` at time `t` (see [`HorizontalDiffusivityField::at`]).
    pub fn at(
        &self,
        element: ElementIndex,
        weights: &[f64],
        sigma: f64,
        t: f64,
    ) -> (f64, [f64; 2]) {
        let (t0, t1) = (self.before.0, self.after.0);
        let a = if t1 > t0 {
            ((t - t0) / (t1 - t0)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let (k0, g0) = self.before.1.at(element, weights, sigma);
        if a == 0.0 {
            return (k0, g0);
        }
        let (k1, g1) = self.after.1.at(element, weights, sigma);
        (
            (1.0 - a) * k0 + a * k1,
            [0, 1].map(|d| (1.0 - a) * g0[d] + a * g1[d]),
        )
    }
}

/// A 2D flow with a horizontal diffusivity field for the walk (see the
/// [module docs](self)); the tracker adds its own constant diffusivity.
#[derive(Clone, Copy)]
pub struct WithHorizontalDiffusivity2D<'a, V> {
    flow: V,
    diffusivity: DiffusivityInTime<'a>,
}

impl<'a, V: ParticleVelocity2D> WithHorizontalDiffusivity2D<'a, V> {
    /// `flow` with the walk's diffusivity `diffusivity`.
    pub fn new(flow: V, diffusivity: DiffusivityInTime<'a>) -> Self {
        Self { flow, diffusivity }
    }
}

impl<V: ParticleVelocity2D> ParticleVelocity2D for WithHorizontalDiffusivity2D<'_, V> {
    fn velocity(&self, element: ElementIndex, weights: &[f64], t: f64) -> [f64; 2] {
        self.flow.velocity(element, weights, t)
    }

    fn depth(&self, element: ElementIndex, weights: &[f64], t: f64) -> Option<f64> {
        self.flow.depth(element, weights, t)
    }

    fn depth_gradient(
        &self,
        element: ElementIndex,
        gradient_weights: [&[f64]; 2],
        t: f64,
    ) -> Option<[f64; 2]> {
        self.flow.depth_gradient(element, gradient_weights, t)
    }

    fn horizontal_diffusivity(
        &self,
        element: ElementIndex,
        weights: &[f64],
        t: f64,
    ) -> Option<(f64, [f64; 2])> {
        Some(self.diffusivity.at(element, weights, 0.0, t))
    }
}

/// A 3D flow with a horizontal diffusivity field for the walk, at each
/// particle's σ-level (see the [module docs](self)); the tracker adds its
/// own constant diffusivity.
#[derive(Clone, Copy)]
pub struct WithHorizontalDiffusivity3D<'a, V> {
    flow: V,
    diffusivity: DiffusivityInTime<'a>,
}

impl<'a, V: ParticleVelocity3D> WithHorizontalDiffusivity3D<'a, V> {
    /// `flow` with the walk's diffusivity `diffusivity`.
    pub fn new(flow: V, diffusivity: DiffusivityInTime<'a>) -> Self {
        Self { flow, diffusivity }
    }
}

impl<V: ParticleVelocity3D> ParticleVelocity3D for WithHorizontalDiffusivity3D<'_, V> {
    fn velocity(&self, element: ElementIndex, weights: &[f64], sigma: f64, t: f64) -> [f64; 3] {
        self.flow.velocity(element, weights, sigma, t)
    }

    fn depth(&self, element: ElementIndex, weights: &[f64], t: f64) -> f64 {
        self.flow.depth(element, weights, t)
    }

    fn depth_gradient(
        &self,
        element: ElementIndex,
        gradient_weights: [&[f64]; 2],
        t: f64,
    ) -> Option<[f64; 2]> {
        self.flow.depth_gradient(element, gradient_weights, t)
    }

    fn diffusivity(
        &self,
        element: ElementIndex,
        weights: &[f64],
        sigma: f64,
        t: f64,
    ) -> Option<(f64, f64)> {
        self.flow.diffusivity(element, weights, sigma, t)
    }

    fn tracers(
        &self,
        element: ElementIndex,
        weights: &[f64],
        sigma: f64,
        t: f64,
    ) -> Option<[f64; 2]> {
        self.flow.tracers(element, weights, sigma, t)
    }

    fn horizontal_diffusivity(
        &self,
        element: ElementIndex,
        weights: &[f64],
        sigma: f64,
        t: f64,
    ) -> Option<(f64, [f64; 2])> {
        Some(self.diffusivity.at(element, weights, sigma, t))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Coinciding nodes take one value, a polynomial of the element's degree
    /// is kept with its exact gradient, and negative values are cut to 0.
    #[test]
    fn field_is_continuous_and_differentiates_exactly() {
        let mut mesh = Mesh2D::uniform_rectangle(0.0, 4.0, 0.0, 3.0, 4, 3);
        // Displace an interior vertex: bilinear elements, not parallelograms
        for v in &mut mesh.vertices {
            if (v[0] - 2.0).abs() < 1e-12 && (v[1] - 1.0).abs() < 1e-12 {
                *v = [2.2, 1.1];
            }
        }
        let ops = DGOperators2D::new(2);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let builder = HorizontalDiffusivity::new(&mesh, &ops, &geom);
        let nn = ops.n_nodes;
        let xy: Vec<[f64; 2]> = ElementIndex::iter(mesh.n_elements)
            .flat_map(|k| (0..nn).map(move |i| (k, i)))
            .map(|(k, i)| mesh.reference_to_physical(k, ops.nodes_r[i], ops.nodes_s[i]))
            .collect();
        // A field linear in x and y: in the elements' (bilinear) space
        let f = |[x, y]: [f64; 2]| 2.0 + 0.3 * x - 0.2 * y;
        let field = builder.from_nodal(xy.iter().map(|&p| f(p)).collect(), Vec::new());
        for (n, &p) in xy.iter().enumerate() {
            assert!((field.values()[n] - f(p)).abs() < 1e-12);
            let [gx, gy] = field.gradients()[n];
            assert!(
                (gx - 0.3).abs() < 1e-11 && (gy + 0.2).abs() < 1e-11,
                "{gx} {gy}"
            );
        }
        // A field discontinuous between elements: equal where nodes coincide
        let jumpy: Vec<f64> = (0..xy.len()).map(|n| (n / nn) as f64 - 3.0).collect();
        let field = builder.from_nodal(jumpy, Vec::new());
        for a in 0..xy.len() {
            for b in 0..xy.len() {
                let d = (xy[a][0] - xy[b][0]).hypot(xy[a][1] - xy[b][1]);
                if d < 1e-9 {
                    assert_eq!(field.values()[a], field.values()[b]);
                }
            }
            assert!(field.values()[a] >= 0.0);
        }
    }

    /// In a linear shear `u = αy` the strain rate is `|α|` everywhere, so
    /// on a uniform mesh `K = (ν₀ + (C_s Δ)²|α|)/Pr_t` exactly, without a
    /// gradient; between two levels it is linear in σ.
    #[test]
    fn smagorinsky_of_a_linear_shear() {
        let mesh = Mesh2D::uniform_rectangle(0.0, 300.0, 0.0, 200.0, 6, 4);
        let ops = DGOperators2D::new(3);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let builder = HorizontalDiffusivity::new(&mesh, &ops, &geom);
        let nn = ops.n_nodes;
        let alpha = 2e-3;
        let u: Vec<f64> = ElementIndex::iter(mesh.n_elements)
            .flat_map(|k| (0..nn).map(move |i| (k, i)))
            .map(|(k, i)| alpha * mesh.reference_to_physical(k, ops.nodes_r[i], ops.nodes_s[i])[1])
            .collect();
        let v = vec![0.0; u.len()];
        let viscosity = HorizontalViscosity2D::smagorinsky(0.2).with_background(0.05);
        let field = builder.smagorinsky_2d(&u, &v, &viscosity, 2.0);
        let delta: f64 = 50.0 / 3.0;
        let exact = (0.05 + (0.2 * delta).powi(2) * alpha) / 2.0;
        for (k, g) in field.values().iter().zip(field.gradients()) {
            assert!((k - exact).abs() < 1e-12 * exact, "{k} {exact}");
            assert!(g[0].abs() < 1e-12 && g[1].abs() < 1e-12);
        }
        // Two levels at σ = −0.75 and −0.25: K = 1 and 3, linear between
        let n_total = u.len();
        let mut values = vec![1.0; 2 * n_total];
        values[n_total..].fill(3.0);
        let field = builder.from_nodal(values, vec![-0.75, -0.25]);
        let w = ops.interpolation_weights(0.1, -0.3);
        let k = ElementIndex::new(5);
        assert!((field.at(k, &w, -0.5).0 - 2.0).abs() < 1e-12);
        assert!((field.at(k, &w, -0.9).0 - 1.0).abs() < 1e-12);
        assert!((field.at(k, &w, 0.0).0 - 3.0).abs() < 1e-12);
    }
}

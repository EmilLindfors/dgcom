//! Velocity fields that particles are advected by.
//!
//! A [`ParticleVelocity2D`] evaluates a velocity at a point of the mesh,
//! given the point's element and the values of the element's nodal basis
//! there ([`DGOperators2D::interpolation_weights_into`](crate::operators::DGOperators2D::interpolation_weights_into)),
//! so a DG field is sampled as its own polynomial, not at the nearest node.
//!
//! - [`SWEVelocity2D`]: the depth-averaged velocity of shallow-water
//!   solutions, steady or linear in time between two snapshots (output
//!   files, or the previous and current state of a running model).
//! - [`NodalVelocity2D`]: a steady nodal velocity field.

use crate::solver::SWESolution2D;
use crate::types::ElementIndex;

/// A velocity field for particle tracking (see the module docs).
pub trait ParticleVelocity2D: Sync {
    /// Velocity (mesh axes) at time `t` at the point of element `element`
    /// where the nodal basis takes the values `weights`.
    fn velocity(&self, element: ElementIndex, weights: &[f64], t: f64) -> [f64; 2];

    /// Water depth at the same point, for stranding particles on drying
    /// ground; `None` when the field has no depth (particles never strand).
    fn depth(&self, _element: ElementIndex, _weights: &[f64], _t: f64) -> Option<f64> {
        None
    }

    /// Gradient `∇h` of the water depth at the same point, where the nodal
    /// basis has the physical gradients `gradient_weights` (`∂ℓ/∂x`,
    /// `∂ℓ/∂y`), for the walk's drift `K∇h/h`; `None` (the default) when the
    /// field has no depth.
    fn depth_gradient(
        &self,
        _element: ElementIndex,
        _gradient_weights: [&[f64]; 2],
        _t: f64,
    ) -> Option<[f64; 2]> {
        None
    }

    /// Horizontal diffusivity `K` (m²/s) of the walk and its gradient `∇K`
    /// (m/s) at the point; `None` (the default) for the tracker's constant
    /// alone (see [`super::diffusivity`]).
    fn horizontal_diffusivity(
        &self,
        _element: ElementIndex,
        _weights: &[f64],
        _t: f64,
    ) -> Option<(f64, [f64; 2])> {
        None
    }
}

/// Dot product of the basis values with an element's nodal values.
#[inline]
fn evaluate(weights: &[f64], values: &[f64]) -> f64 {
    debug_assert_eq!(weights.len(), values.len());
    weights.iter().zip(values).map(|(w, v)| w * v).sum()
}

/// A steady nodal velocity field, stored element by element
/// (`[n_elements × n_nodes]`, like the SoA solution fields).
#[derive(Clone, Copy, Debug)]
pub struct NodalVelocity2D<'a> {
    u: &'a [f64],
    v: &'a [f64],
    n_nodes: usize,
}

impl<'a> NodalVelocity2D<'a> {
    /// The field with components `u` and `v`, `n_nodes` per element.
    pub fn new(u: &'a [f64], v: &'a [f64], n_nodes: usize) -> Self {
        assert_eq!(u.len(), v.len());
        assert_eq!(u.len() % n_nodes, 0);
        Self { u, v, n_nodes }
    }
}

impl ParticleVelocity2D for NodalVelocity2D<'_> {
    fn velocity(&self, element: ElementIndex, weights: &[f64], _t: f64) -> [f64; 2] {
        let nodes = element.as_usize() * self.n_nodes..(element.as_usize() + 1) * self.n_nodes;
        [
            evaluate(weights, &self.u[nodes.clone()]),
            evaluate(weights, &self.v[nodes]),
        ]
    }
}

/// The depth-averaged velocity of shallow-water solutions, linear in time
/// between two snapshots.
///
/// At each snapshot the velocity at a point is `(hu, hv)/h` of the
/// interpolated conserved variables (as [`Probe2D::sample_swe`](crate::solver::Probe2D::sample_swe)), and zero
/// where `h ≤ h_min`. Between the snapshots velocity and depth are
/// interpolated linearly in time, and outside them held at the nearer one.
#[derive(Clone, Copy)]
pub struct SWEVelocity2D<'a> {
    before: (f64, &'a SWESolution2D),
    after: (f64, &'a SWESolution2D),
    h_min: f64,
}

impl<'a> SWEVelocity2D<'a> {
    /// The velocity of one state, the same at all times.
    pub fn steady(q: &'a SWESolution2D, h_min: f64) -> Self {
        Self {
            before: (0.0, q),
            after: (0.0, q),
            h_min,
        }
    }

    /// Linear in time between `q0` at `t0` and `q1` at `t1 > t0`.
    pub fn between(
        t0: f64,
        q0: &'a SWESolution2D,
        t1: f64,
        q1: &'a SWESolution2D,
        h_min: f64,
    ) -> Self {
        assert!(t1 > t0, "snapshots must be in time order: {t0} → {t1}");
        assert_eq!(q0.n_elements, q1.n_elements);
        assert_eq!(q0.n_nodes, q1.n_nodes);
        Self {
            before: (t0, q0),
            after: (t1, q1),
            h_min,
        }
    }

    /// Weight of the later snapshot at time `t`.
    fn fraction(&self, t: f64) -> f64 {
        let (t0, t1) = (self.before.0, self.after.0);
        if t1 > t0 {
            ((t - t0) / (t1 - t0)).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    /// Depth and velocity of one snapshot at the point.
    fn sample(&self, q: &SWESolution2D, k: ElementIndex, weights: &[f64]) -> (f64, [f64; 2]) {
        let h = evaluate(weights, q.element_h(k));
        if h > self.h_min {
            let hu = evaluate(weights, q.element_hu(k));
            let hv = evaluate(weights, q.element_hv(k));
            (h, [hu / h, hv / h])
        } else {
            (h, [0.0, 0.0])
        }
    }
}

impl ParticleVelocity2D for SWEVelocity2D<'_> {
    fn velocity(&self, element: ElementIndex, weights: &[f64], t: f64) -> [f64; 2] {
        let a = self.fraction(t);
        let (_, u0) = self.sample(self.before.1, element, weights);
        if a == 0.0 {
            return u0;
        }
        let (_, u1) = self.sample(self.after.1, element, weights);
        [(1.0 - a) * u0[0] + a * u1[0], (1.0 - a) * u0[1] + a * u1[1]]
    }

    fn depth(&self, element: ElementIndex, weights: &[f64], t: f64) -> Option<f64> {
        let a = self.fraction(t);
        let h0 = evaluate(weights, self.before.1.element_h(element));
        if a == 0.0 {
            return Some(h0);
        }
        let h1 = evaluate(weights, self.after.1.element_h(element));
        Some((1.0 - a) * h0 + a * h1)
    }

    fn depth_gradient(
        &self,
        element: ElementIndex,
        gradient_weights: [&[f64]; 2],
        t: f64,
    ) -> Option<[f64; 2]> {
        let a = self.fraction(t);
        let g0 = gradient_weights.map(|g| evaluate(g, self.before.1.element_h(element)));
        if a == 0.0 {
            return Some(g0);
        }
        let g1 = gradient_weights.map(|g| evaluate(g, self.after.1.element_h(element)));
        Some([0, 1].map(|d| (1.0 - a) * g0[d] + a * g1[d]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::DGOperators2D;
    use crate::solver::SWEState2D;

    /// Velocity and depth are linear in time between the snapshots and held
    /// outside them; dry points have no velocity.
    #[test]
    fn snapshots_are_interpolated_linearly_in_time() {
        let ops = DGOperators2D::new(2);
        let k = ElementIndex::new(0);
        let uniform = |h: f64, u: f64, v: f64| {
            let mut q = SWESolution2D::new(1, ops.n_nodes);
            for i in 0..ops.n_nodes {
                q.set_state(k, i, SWEState2D::new(h, h * u, h * v));
            }
            q
        };
        let (q0, q1) = (uniform(2.0, 1.0, -0.5), uniform(4.0, 3.0, 0.5));
        let field = SWEVelocity2D::between(10.0, &q0, 20.0, &q1, 1e-6);
        let w = ops.interpolation_weights(0.3, -0.4);
        for (t, u, v, h) in [
            (10.0, 1.0, -0.5, 2.0),
            (12.5, 1.5, -0.25, 2.5),
            (20.0, 3.0, 0.5, 4.0),
            (5.0, 1.0, -0.5, 2.0),
            (25.0, 3.0, 0.5, 4.0),
        ] {
            let [uu, vv] = field.velocity(k, &w, t);
            assert!((uu - u).abs() < 1e-12 && (vv - v).abs() < 1e-12, "t = {t}");
            assert!((field.depth(k, &w, t).unwrap() - h).abs() < 1e-12);
        }
        let dry = uniform(1e-9, 1.0, 1.0);
        let field = SWEVelocity2D::steady(&dry, 1e-6);
        assert_eq!(field.velocity(k, &w, 3.0), [0.0, 0.0]);
    }
}

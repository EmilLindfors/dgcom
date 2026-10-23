//! Velocity and diffusivity fields that 3D particles move in.
//!
//! A [`ParticleVelocity3D`] evaluates, at a point of the mesh and a σ-level,
//! the horizontal velocity along the σ-surface and the σ-velocity `dσ/dt`,
//! the water depth, and optionally the vertical eddy diffusivity. As in 2D
//! the point is given by its element and the values of the element's nodal
//! basis there, so each layer is sampled as its own polynomial.
//!
//! [`Solution3DVelocity`] samples [`Solution3D`] snapshots, linear in time
//! between two of them:
//! - `u`, `v` at the layer centres `σ_l`, linear in σ between them and held
//!   above the top and below the bottom centre;
//! - `dσ/dt = Ω/D` from `Solution3D::w` (Ω at the layer centres) and
//!   `Ω = 0` at the surface and the bed, linear in σ between them;
//! - `K` from `Solution3D::eddy_diffusivity` at the w-points, linear in σ,
//!   so `∂K/∂z` is constant within each layer;
//! - temperature and salinity at the layer centres like `u`, `v`.

use crate::mesh::Bathymetry2D;
use crate::solver::state::Solution3D;
use crate::types::ElementIndex;
use crate::vertical::SigmaGrid;

/// A field for 3D particle tracking (see the module docs).
pub trait ParticleVelocity3D: Sync {
    /// `[u, v, dσ/dt]` (m/s, m/s, 1/s; mesh axes) at time `t` at σ-level
    /// `sigma` ∈ [−1, 0] of the point of element `element` where the nodal
    /// basis takes the values `weights`.
    fn velocity(&self, element: ElementIndex, weights: &[f64], sigma: f64, t: f64) -> [f64; 3];

    /// Water depth `D = η − B` (m) at the point.
    fn depth(&self, element: ElementIndex, weights: &[f64], t: f64) -> f64;

    /// Gradient `∇D` of the water depth at the point, where the nodal basis
    /// has the physical gradients `gradient_weights` (`∂ℓ/∂x`, `∂ℓ/∂y`), for
    /// the walk's drift `K∇D/D`; `None` (the default) for none.
    fn depth_gradient(
        &self,
        _element: ElementIndex,
        _gradient_weights: [&[f64]; 2],
        _t: f64,
    ) -> Option<[f64; 2]> {
        None
    }

    /// Vertical eddy diffusivity `K` (m²/s) and its derivative `∂K/∂z`
    /// (m/s) at the point and σ-level; `None` (the default) for none.
    fn diffusivity(
        &self,
        _element: ElementIndex,
        _weights: &[f64],
        _sigma: f64,
        _t: f64,
    ) -> Option<(f64, f64)> {
        None
    }

    /// Horizontal diffusivity `K` (m²/s) of the walk and its gradient along
    /// the σ-surface `∇K` (m/s) at the point and σ-level; `None` (the
    /// default) for the tracker's constant alone (see
    /// [`super::diffusivity`]).
    fn horizontal_diffusivity(
        &self,
        _element: ElementIndex,
        _weights: &[f64],
        _sigma: f64,
        _t: f64,
    ) -> Option<(f64, [f64; 2])> {
        None
    }

    /// Temperature (°C) and salinity at the point and σ-level, for the
    /// particles' behaviour ([`super::ParticleBehaviour3D`]); `None` (the
    /// default) for a field without them.
    fn tracers(
        &self,
        _element: ElementIndex,
        _weights: &[f64],
        _sigma: f64,
        _t: f64,
    ) -> Option<[f64; 2]> {
        None
    }
}

/// Dot product of the basis values with an element's nodal values.
#[inline]
fn evaluate(weights: &[f64], values: &[f64]) -> f64 {
    weights.iter().zip(values).map(|(w, v)| w * v).sum()
}

/// `Σ_i w_i field[(k·n + i)·stride + level]`: one level of a column field
/// at the point.
#[inline]
fn evaluate_level(weights: &[f64], field: &[f64], k: usize, stride: usize, level: usize) -> f64 {
    let n = weights.len();
    weights
        .iter()
        .enumerate()
        .map(|(i, w)| w * field[(k * n + i) * stride + level])
        .sum()
}

/// Bracketing index and weight of `x` in the increasing `grid`: `x` lies
/// at `(1 − a)·grid[j] + a·grid[j + 1]`, held at the ends.
#[inline]
pub(super) fn bracket(grid: &[f64], x: f64) -> (usize, f64) {
    let last = grid.len() - 1;
    if x <= grid[0] {
        return (0, 0.0);
    }
    if x >= grid[last] {
        return (last.saturating_sub(1), if last == 0 { 0.0 } else { 1.0 });
    }
    let j = grid.partition_point(|&g| g <= x) - 1;
    (j, (x - grid[j]) / (grid[j + 1] - grid[j]))
}

/// The fields of one snapshot at a point.
#[derive(Clone, Copy, Debug, Default)]
struct Sample {
    depth: f64,
    velocity: [f64; 3],
    diffusivity: Option<(f64, f64)>,
}

/// The velocity and diffusivity of [`Solution3D`] snapshots (see the module
/// docs), linear in time between two and held outside them.
#[derive(Clone, Copy)]
pub struct Solution3DVelocity<'a> {
    before: (f64, &'a Solution3D),
    after: (f64, &'a Solution3D),
    sigma: &'a SigmaGrid,
    bathymetry: &'a Bathymetry2D,
    min_depth: f64,
}

impl<'a> Solution3DVelocity<'a> {
    /// The fields of one state, the same at all times, on the σ-levels
    /// `sigma` over the bed `bathymetry`. Where the water is shallower than
    /// `min_depth` (m) the velocity and diffusivity are zero.
    pub fn steady(
        state: &'a Solution3D,
        sigma: &'a SigmaGrid,
        bathymetry: &'a Bathymetry2D,
        min_depth: f64,
    ) -> Self {
        Self::between(0.0, state, 0.0, state, sigma, bathymetry, min_depth)
    }

    /// Linear in time between `s0` at `t0` and `s1` at `t1 ≥ t0`.
    pub fn between(
        t0: f64,
        s0: &'a Solution3D,
        t1: f64,
        s1: &'a Solution3D,
        sigma: &'a SigmaGrid,
        bathymetry: &'a Bathymetry2D,
        min_depth: f64,
    ) -> Self {
        assert!(t1 >= t0, "snapshots must be in time order: {t0} → {t1}");
        for s in [s0, s1] {
            assert_eq!(s.n_levels, sigma.n_levels(), "σ-levels of the state");
            assert_eq!(s.eta.data.len(), bathymetry.data.len(), "nodes of the bed");
        }
        Self {
            before: (t0, s0),
            after: (t1, s1),
            sigma,
            bathymetry,
            min_depth,
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

    /// `∇D = ∇η − ∇B` of snapshot `s` at the point.
    fn depth_gradient_of(
        &self,
        s: &Solution3D,
        k: usize,
        gradient_weights: [&[f64]; 2],
    ) -> [f64; 2] {
        let n = gradient_weights[0].len();
        gradient_weights.map(|g| {
            evaluate(g, &s.eta.data[k * n..(k + 1) * n])
                - evaluate(g, self.bathymetry.element(ElementIndex::new(k)))
        })
    }

    fn depth_of(&self, s: &Solution3D, k: usize, weights: &[f64]) -> f64 {
        let n = weights.len();
        evaluate(weights, &s.eta.data[k * n..(k + 1) * n])
            - evaluate(weights, self.bathymetry.element(ElementIndex::new(k)))
    }

    /// The fields of snapshot `s` at the point and σ-level.
    fn sample(&self, s: &Solution3D, k: usize, weights: &[f64], sigma: f64) -> Sample {
        let depth = self.depth_of(s, k, weights);
        if depth <= self.min_depth {
            return Sample {
                depth,
                ..Sample::default()
            };
        }
        let nl = s.n_levels;
        // Horizontal velocity between the layer centres
        let (l, a) = bracket(self.sigma.sigma_rho(), sigma);
        let upper = (l + 1).min(nl - 1);
        let at = |field: &[f64]| {
            (1.0 - a) * evaluate_level(weights, field, k, nl, l)
                + a * evaluate_level(weights, field, k, nl, upper)
        };
        let (u, v) = (at(&s.u), at(&s.v));
        // Ω at the layer centres, zero at the bed and the surface
        let sigma_rho = self.sigma.sigma_rho();
        let omega = if sigma <= sigma_rho[0] {
            let below = evaluate_level(weights, &s.w, k, nl, 0);
            below * (sigma + 1.0) / (sigma_rho[0] + 1.0)
        } else if sigma >= sigma_rho[nl - 1] {
            let above = evaluate_level(weights, &s.w, k, nl, nl - 1);
            above * sigma / sigma_rho[nl - 1]
        } else {
            at(&s.w)
        };
        // K at the w-points
        let diffusivity = (s.eddy_diffusivity.len() == s.u.len() / nl * (nl + 1)).then(|| {
            let sigma_w = self.sigma.sigma_w();
            let j = (sigma_w.partition_point(|&g| g <= sigma).max(1) - 1).min(nl - 1);
            let a = ((sigma - sigma_w[j]) / (sigma_w[j + 1] - sigma_w[j])).clamp(0.0, 1.0);
            let k0 = evaluate_level(weights, &s.eddy_diffusivity, k, nl + 1, j);
            let k1 = evaluate_level(weights, &s.eddy_diffusivity, k, nl + 1, j + 1);
            let slope = (k1 - k0) / ((sigma_w[j + 1] - sigma_w[j]) * depth);
            ((1.0 - a) * k0 + a * k1, slope)
        });
        Sample {
            depth,
            velocity: [u, v, omega / depth],
            diffusivity,
        }
    }

    /// Temperature and salinity of snapshot `s` at the point and σ-level,
    /// between the layer centres and held outside them (in any depth: a
    /// thin column keeps its water's properties).
    fn tracers_of(&self, s: &Solution3D, k: usize, weights: &[f64], sigma: f64) -> [f64; 2] {
        let nl = s.n_levels;
        let (l, a) = bracket(self.sigma.sigma_rho(), sigma);
        let upper = (l + 1).min(nl - 1);
        [&s.temp, &s.salt].map(|field| {
            (1.0 - a) * evaluate_level(weights, field, k, nl, l)
                + a * evaluate_level(weights, field, k, nl, upper)
        })
    }

    fn blend(&self, k: usize, weights: &[f64], sigma: f64, t: f64) -> Sample {
        let a = self.fraction(t);
        let s0 = self.sample(self.before.1, k, weights, sigma);
        if a == 0.0 {
            return s0;
        }
        let s1 = self.sample(self.after.1, k, weights, sigma);
        let mix = |x: f64, y: f64| (1.0 - a) * x + a * y;
        Sample {
            depth: mix(s0.depth, s1.depth),
            velocity: [0, 1, 2].map(|d| mix(s0.velocity[d], s1.velocity[d])),
            diffusivity: match (s0.diffusivity, s1.diffusivity) {
                (Some((k0, d0)), Some((k1, d1))) => Some((mix(k0, k1), mix(d0, d1))),
                _ => None,
            },
        }
    }
}

impl ParticleVelocity3D for Solution3DVelocity<'_> {
    fn velocity(&self, element: ElementIndex, weights: &[f64], sigma: f64, t: f64) -> [f64; 3] {
        self.blend(element.as_usize(), weights, sigma, t).velocity
    }

    fn depth(&self, element: ElementIndex, weights: &[f64], t: f64) -> f64 {
        let a = self.fraction(t);
        let k = element.as_usize();
        let d0 = self.depth_of(self.before.1, k, weights);
        if a == 0.0 {
            return d0;
        }
        (1.0 - a) * d0 + a * self.depth_of(self.after.1, k, weights)
    }

    fn depth_gradient(
        &self,
        element: ElementIndex,
        gradient_weights: [&[f64]; 2],
        t: f64,
    ) -> Option<[f64; 2]> {
        let (a, k) = (self.fraction(t), element.as_usize());
        let g0 = self.depth_gradient_of(self.before.1, k, gradient_weights);
        if a == 0.0 {
            return Some(g0);
        }
        let g1 = self.depth_gradient_of(self.after.1, k, gradient_weights);
        Some([0, 1].map(|d| (1.0 - a) * g0[d] + a * g1[d]))
    }

    fn diffusivity(
        &self,
        element: ElementIndex,
        weights: &[f64],
        sigma: f64,
        t: f64,
    ) -> Option<(f64, f64)> {
        self.blend(element.as_usize(), weights, sigma, t)
            .diffusivity
    }

    fn tracers(
        &self,
        element: ElementIndex,
        weights: &[f64],
        sigma: f64,
        t: f64,
    ) -> Option<[f64; 2]> {
        let (a, k) = (self.fraction(t), element.as_usize());
        let before = self.tracers_of(self.before.1, k, weights, sigma);
        if a == 0.0 {
            return Some(before);
        }
        let after = self.tracers_of(self.after.1, k, weights, sigma);
        Some([0, 1].map(|i| (1.0 - a) * before[i] + a * after[i]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::DGOperators2D;
    use crate::vertical::SongHaidvogelStretching;

    /// Linear profiles in σ are sampled exactly between the layer centres
    /// (and K between the w-points, with its z-derivative), on stretched
    /// levels, and so are temperature and salinity; Ω vanishes at the
    /// surface and the bed; snapshots blend linearly in time.
    #[test]
    fn linear_profiles_are_sampled_exactly() {
        let ops = DGOperators2D::new(2);
        let nn = ops.n_nodes;
        let nl = 6;
        let sigma = SigmaGrid::new(nl, SongHaidvogelStretching::new(3.0, 0.4, 5.0));
        let depth = 20.0;
        let bathymetry = Bathymetry2D::constant(1, nn, -depth);
        let state = |scale: f64| {
            let mut s = Solution3D::new(1, nn, nl);
            for i in 0..nn {
                for (l, &sr) in sigma.sigma_rho().iter().enumerate() {
                    s.u[i * nl + l] = scale * (0.3 + 0.2 * sr);
                    s.v[i * nl + l] = scale * (-0.1 * sr);
                    s.w[i * nl + l] = scale * 1e-3 * sr * (1.0 + sr);
                    s.temp[i * nl + l] = scale * (8.0 + 2.0 * sr);
                    s.salt[i * nl + l] = 30.0 + scale * sr;
                }
                for (j, &sw) in sigma.sigma_w().iter().enumerate() {
                    s.eddy_diffusivity[i * (nl + 1) + j] = scale * (1e-3 - 2e-3 * sw);
                }
            }
            s
        };
        let (s0, s1) = (state(1.0), state(3.0));
        let field = Solution3DVelocity::between(10.0, &s0, 20.0, &s1, &sigma, &bathymetry, 0.01);
        let w = ops.interpolation_weights(0.2, -0.7);
        let k = ElementIndex::new(0);
        let (lo, hi) = (sigma.sigma_rho()[0], sigma.sigma_rho()[nl - 1]);
        for t in [10.0, 15.0, 20.0] {
            let scale = 1.0 + 2.0 * (t - 10.0) / 10.0;
            for s in [-0.97, lo, -0.6, -0.31, hi, -0.01] {
                let [u, v, sdot] = field.velocity(k, &w, s, t);
                let held = s.clamp(lo, hi);
                assert!((u - scale * (0.3 + 0.2 * held)).abs() < 1e-13, "u at σ {s}");
                assert!((v + scale * 0.1 * held).abs() < 1e-13, "v at σ {s}");
                if (lo..=hi).contains(&s) {
                    // Ω linear between the centres only where it is linear
                    // in the centres' values (they are a parabola)
                    assert!(sdot.is_finite());
                }
                let [temp, salt] = field.tracers(k, &w, s, t).unwrap();
                assert!(
                    (temp - scale * (8.0 + 2.0 * held)).abs() < 1e-13,
                    "T at σ {s}"
                );
                assert!((salt - 30.0 - scale * held).abs() < 1e-13, "S at σ {s}");
                let (kz, dkdz) = field.diffusivity(k, &w, s, t).unwrap();
                assert!((kz - scale * (1e-3 - 2e-3 * s)).abs() < 1e-15, "K at σ {s}");
                assert!((dkdz + scale * 2e-3 / depth).abs() < 1e-15, "K' at σ {s}");
            }
            assert!((field.depth(k, &w, t) - depth).abs() < 1e-12);
            assert_eq!(field.velocity(k, &w, 0.0, t)[2], 0.0);
            assert!(field.velocity(k, &w, -1.0, t)[2].abs() < 1e-18);
        }
    }

    #[test]
    fn bracket_holds_at_the_ends() {
        let grid = [-0.9, -0.5, -0.1];
        assert_eq!(bracket(&grid, -1.0), (0, 0.0));
        assert_eq!(bracket(&grid, 0.0), (1, 1.0));
        let (j, a) = bracket(&grid, -0.3);
        assert_eq!(j, 1);
        assert!((a - 0.5).abs() < 1e-15);
    }
}

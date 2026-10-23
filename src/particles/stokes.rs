//! Particles carried by the waves' Stokes drift as well as the current.
//!
//! A phase-averaged circulation model without wave forcing computes the
//! Eulerian current; a particle at the surface also moves with the Stokes drift
//! of the waves, the mean of the orbital motion it rides (Kenyon 1969;
//! McWilliams & Restrepo 1999). Offline trackers add it to the current
//! (OpenDrift's `stokes_drift`, LADiM), and so do these wrappers:
//!
//! - [`WithStokesDrift2D`]: the drift at a fixed depth under the surface (0 for
//!   surface drifters and floating material) added to a 2D flow;
//! - [`WithStokesDrift3D`]: the drift at each particle's own depth `−σ D` added
//!   to the horizontal velocity of a 3D flow. The drift decays as `e^{2kz}`, so
//!   larvae in the top metres feel it and particles below a wavelength or so do
//!   not. Its vertical part (`∫∇·u_s dz`, nonzero only where the sea state
//!   varies) is left out, as in those trackers.
//!
//! The drift comes from [`StokesDriftField`]s of the wave model, on the same
//! mesh and order as the flow, steady or linear in time between two wave states
//! ([`StokesDrift`]). Depth, diffusivity and tracers are the flow's.
//!
//! A circulation model forced by the waves through the vortex force (TODO F.4
//! coupling) still computes the Eulerian current, so the drift is added there
//! too; one forced by the radiation-stress gradient alone is ambiguous, and the
//! drift would be added as here.

use crate::types::ElementIndex;
use crate::waves::StokesDriftField;

use super::{ParticleVelocity2D, ParticleVelocity3D};

/// The Stokes drift of one or two wave states, linear in time between them and
/// held outside them.
#[derive(Clone, Copy, Debug)]
pub struct StokesDrift<'a> {
    before: (f64, &'a StokesDriftField),
    after: (f64, &'a StokesDriftField),
}

impl<'a> StokesDrift<'a> {
    /// The drift of one wave state, the same at all times.
    pub fn steady(field: &'a StokesDriftField) -> Self {
        Self {
            before: (0.0, field),
            after: (0.0, field),
        }
    }

    /// Linear in time between `f0` at `t0` and `f1` at `t1 > t0`.
    pub fn between(t0: f64, f0: &'a StokesDriftField, t1: f64, f1: &'a StokesDriftField) -> Self {
        assert!(t1 > t0, "wave states must be in time order: {t0} → {t1}");
        assert_eq!(f0.n_points(), f1.n_points());
        Self {
            before: (t0, f0),
            after: (t1, f1),
        }
    }

    /// The drift (m/s) at `depth_below` (m) under the surface at time `t` at the
    /// point of element `element` where the nodal basis takes the values
    /// `weights`.
    pub fn at(&self, element: ElementIndex, weights: &[f64], depth_below: f64, t: f64) -> [f64; 2] {
        let (t0, t1) = (self.before.0, self.after.0);
        let a = if t1 > t0 {
            ((t - t0) / (t1 - t0)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let u0 = self.before.1.at(element, weights, depth_below);
        if a == 0.0 {
            return u0;
        }
        let u1 = self.after.1.at(element, weights, depth_below);
        [0, 1].map(|d| (1.0 - a) * u0[d] + a * u1[d])
    }
}

/// A 2D flow plus the Stokes drift at a fixed depth (see the module docs).
#[derive(Clone, Copy)]
pub struct WithStokesDrift2D<'a, V> {
    flow: V,
    stokes: StokesDrift<'a>,
    depth_below: f64,
}

impl<'a, V: ParticleVelocity2D> WithStokesDrift2D<'a, V> {
    /// `flow` plus the drift at `depth_below` (m, ≥ 0) under the surface.
    pub fn new(flow: V, stokes: StokesDrift<'a>, depth_below: f64) -> Self {
        assert!(
            depth_below >= 0.0,
            "a depth under the surface: {depth_below}"
        );
        Self {
            flow,
            stokes,
            depth_below,
        }
    }
}

impl<V: ParticleVelocity2D> ParticleVelocity2D for WithStokesDrift2D<'_, V> {
    fn velocity(&self, element: ElementIndex, weights: &[f64], t: f64) -> [f64; 2] {
        let [u, v] = self.flow.velocity(element, weights, t);
        let [us, vs] = self.stokes.at(element, weights, self.depth_below, t);
        [u + us, v + vs]
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
        self.flow.horizontal_diffusivity(element, weights, t)
    }
}

/// A 3D flow plus the Stokes drift at each particle's depth (see the module
/// docs).
#[derive(Clone, Copy)]
pub struct WithStokesDrift3D<'a, V> {
    flow: V,
    stokes: StokesDrift<'a>,
}

impl<'a, V: ParticleVelocity3D> WithStokesDrift3D<'a, V> {
    pub fn new(flow: V, stokes: StokesDrift<'a>) -> Self {
        Self { flow, stokes }
    }
}

impl<V: ParticleVelocity3D> ParticleVelocity3D for WithStokesDrift3D<'_, V> {
    fn velocity(&self, element: ElementIndex, weights: &[f64], sigma: f64, t: f64) -> [f64; 3] {
        let [u, v, sigma_dot] = self.flow.velocity(element, weights, sigma, t);
        let depth = self.flow.depth(element, weights, t).max(0.0);
        let [us, vs] = self.stokes.at(element, weights, -sigma * depth, t);
        [u + us, v + vs, sigma_dot]
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
        self.flow.horizontal_diffusivity(element, weights, sigma, t)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::mesh::{Bathymetry2D, Mesh2D};
    use crate::operators::{DGOperators2D, GeometricFactors2D};
    use crate::particles::{
        NodalVelocity2D, ParticleTracker2D, ParticleTracker3D, Solution3DVelocity,
    };
    use crate::solver::state::Solution3D;
    use crate::vertical::SigmaGrid;
    use crate::waves::{SpectralGrid, WaveModel2D, WaveSolution};

    const G: f64 = 9.81;
    const DEPTH: f64 = 40.0;

    /// A swell of 1.5 m and 8 s towards 30° over a flat 40 m bed: the wave
    /// model, its state and its drift field.
    fn swell(mesh: &Mesh2D, ops: &DGOperators2D) -> (WaveModel2D, WaveSolution, StokesDriftField) {
        let geom = GeometricFactors2D::compute(mesh, ops);
        let bathymetry = Bathymetry2D::from_function(mesh, ops, &geom, |_, _| -DEPTH);
        let model = WaveModel2D::new(
            Arc::new(mesh.clone()),
            Arc::new(ops.clone()),
            Arc::new(geom),
            &bathymetry,
            SpectralGrid::new(0.05, 0.6, 24, 24),
            G,
        );
        let direction = 30f64.to_radians();
        let n = model.uniform_state(&model.grid.jonswap(1.5, 8.0, 3.3, direction, 10.0));
        let field = StokesDriftField::new(&model, &n);
        (model, n, field)
    }

    /// Over still water a particle moves with the drift alone: surface
    /// drifters at the surface's, and 3D particles at their own depth's (their
    /// σ unchanged), to round-off over many steps. Added to a current, the two
    /// velocities sum.
    #[test]
    fn particles_drift_with_the_waves_at_their_depth() {
        let mesh = Mesh2D::uniform_periodic(0.0, 2000.0, 0.0, 2000.0, 4, 4);
        let ops = DGOperators2D::new(2);
        let (model, state, field) = swell(&mesh, &ops);
        // The model's own spectral sum at a node, at a depth under the surface
        let drift_at = |depth_below: f64| model.stokes_drift(&state, -depth_below)[0];
        let stokes = StokesDrift::steady(&field);
        let (t_end, dt) = (600.0, 30.0);
        let start = [700.0, 900.0];

        // 2D: at the surface and 1 m down, still water, then a current
        let nn = ops.n_nodes;
        let zero = vec![0.0; mesh.n_elements * nn];
        let current = vec![0.2; mesh.n_elements * nn];
        let tracker = ParticleTracker2D::new(&mesh, &ops);
        for (depth_below, u) in [(0.0, &zero), (1.0, &zero), (0.0, &current)] {
            let flow =
                WithStokesDrift2D::new(NodalVelocity2D::new(u, &zero, nn), stokes, depth_below);
            let mut p = [tracker.release(1, start).unwrap()];
            let mut t = 0.0;
            while t < t_end {
                tracker.step(&mut p, &flow, t, dt);
                t += dt;
            }
            let [us, vs] = drift_at(depth_below);
            let expected = [start[0] + (us + u[0]) * t_end, start[1] + vs * t_end];
            let got = p[0].position();
            assert!(
                (got[0] - expected[0]).abs() < 1e-9 && (got[1] - expected[1]).abs() < 1e-9,
                "z {depth_below}, u {}: {got:?} against {expected:?}",
                u[0]
            );
            // The swell's drift is along its direction, cm/s at the surface
            assert!(us > 0.0 && (vs / us - 30f64.to_radians().tan()).abs() < 1e-9);
        }

        // 3D: a column at rest; each particle at its own depth
        let nl = 10;
        let sigma = SigmaGrid::uniform(nl);
        let bed = Bathymetry2D::constant(mesh.n_elements, nn, -DEPTH);
        let rest = Solution3D::new(mesh.n_elements, nn, nl);
        let flow = WithStokesDrift3D::new(
            Solution3DVelocity::steady(&rest, &sigma, &bed, 0.05),
            stokes,
        );
        let tracker = ParticleTracker3D::new(&mesh, &ops);
        let levels = [0.0, -0.025, -0.1, -0.5];
        let mut particles: Vec<_> = levels
            .iter()
            .enumerate()
            .map(|(i, &s)| tracker.release(i as u64, start, s, 0.0).unwrap())
            .collect();
        let mut t = 0.0;
        while t < t_end {
            tracker.step(&mut particles, &flow, t, dt);
            t += dt;
        }
        let mut drifts = Vec::new();
        for (p, &s) in particles.iter().zip(&levels) {
            let [us, vs] = drift_at(-s * DEPTH);
            let got = p.position();
            let moved = [got[0] - start[0], got[1] - start[1]];
            assert!(
                (moved[0] - us * t_end).abs() < 1e-9 && (moved[1] - vs * t_end).abs() < 1e-9,
                "σ {s}: moved {moved:?}, drift {:?}",
                [us * t_end, vs * t_end]
            );
            assert!((p.sigma() - s).abs() < 1e-12, "σ {} from {s}", p.sigma());
            drifts.push(us.hypot(vs));
        }
        println!("drift at 0, 1, 4, 20 m: {drifts:?} m/s");
        // Decaying with depth: an 8 s swell's e-folding depth 1/(2k) is ≈ 8 m
        assert!(drifts.windows(2).all(|w| w[1] < w[0]));
        assert!(drifts[3] < 0.1 * drifts[0]);
    }
}

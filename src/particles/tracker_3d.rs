//! Particle tracking in the 3D flow (TODO F.2, Stage B).
//!
//! A particle has a horizontal position, tracked as in 2D
//! ([`super::ParticleTracker2D`]: the element polynomial at the particle,
//! the face-by-face walk, walls, open exits, stranding), and a σ-level
//! `σ ∈ [−1, 0]` in its water column, so it rides the free surface and the
//! bed as the column changes.
//!
//! # Advection
//!
//! RK4 on `(x, y, σ)` with
//!
//! ```text
//! dx/dt = u(x, σ, t),    dσ/dt = Ω(x, σ, t)/D + w_p/D,
//! ```
//!
//! the horizontal velocity along the σ-surface, the σ-velocity of the flow
//! (`Ω = D dσ/dt` of the 3D continuity) and the particle's own vertical
//! speed `w_p` (m/s, positive up: sinking feed and faeces, swimming larvae).
//! The stage points reflect off the surface and the bed.
//!
//! # Vertical dispersion
//!
//! With [`ParticleTracker3D::with_vertical_random_walk`], each step adds
//! Visser's (1997) random displacement in `z` for a diffusivity `K(z)`,
//!
//! ```text
//! z' = z + K'(z) Δt + √(2 K(z + ½K'(z) Δt) Δt) ξ,
//! ```
//!
//! the Itô form of `∂c/∂t = ∂/∂z(K ∂c/∂z)`: the drift `K'` keeps a
//! well-mixed population well mixed where `K` varies (without it particles
//! collect where `K` is small), and `K` is taken half a drift step away.
//! It needs `Δt ≪ min 1/|K''|`. The surface and the bed reflect (or the bed
//! takes the particle with [`ParticleTracker3D::with_bed_settling`]).
//! Horizontal dispersion is the 2D tracker's walk along the σ-surface, with
//! the drift `K∇D/D` of a concentration in layers `D Δσ` thick (see the
//! [particles module docs](super)).
//!
//! # Behaviour
//!
//! [`ParticleTracker3D::step_with`] takes a [`ParticleBehaviour3D`]: at the
//! start of each step a particle senses its surroundings (depth, and
//! temperature and salinity through [`ParticleVelocity3D::tracers`]),
//! chooses a swimming speed held over the step, develops, and dies when the
//! behaviour says so (swimming salmon-lice larvae: [`super::SalmonLice`]).
//! [`ParticleTracker3D::step`] is the passive particle.
//!
//! # References
//!
//! - Visser, A. W. (1997). Using random walk models to simulate the vertical
//!   distribution of particles in a turbulent water column. *Mar. Ecol.
//!   Prog. Ser.* 158, 275–281.
//! - Gräwe, U. (2011). Implementation of high-order particle-tracking
//!   schemes in a water column model. *Ocean Modelling* 36, 80–89.

use super::behaviour::{ParticleBehaviour3D, Passive, Surroundings};
use super::tracker::{
    MAX_NODES, Particle2D, ParticleStatus, ParticleTracker2D, standard_normal_pair, uniform,
};
use super::velocity_3d::ParticleVelocity3D;
use super::walk::WalkEnd;
use crate::mesh::{BoundaryTag, Mesh2D, MeshPoint};
use crate::operators::DGOperators2D;

/// A particle in the 3D flow: a horizontal particle (position, element,
/// status, random stream) with a σ-level, its own vertical speed, and the
/// state of its behaviour (age, development, current swimming speed).
#[derive(Clone, Debug, PartialEq)]
pub struct Particle3D {
    horizontal: Particle2D,
    sigma: f64,
    vertical_speed: f64,
    age: f64,
    development: f64,
    swimming_speed: f64,
}

impl Particle3D {
    /// Identifier given at release; it also seeds the particle's random
    /// numbers.
    pub fn id(&self) -> u64 {
        self.horizontal.id()
    }

    /// Horizontal position (mesh coordinates).
    pub fn position(&self) -> [f64; 2] {
        self.horizontal.position()
    }

    /// Element and reference coordinates of the horizontal position.
    pub fn point(&self) -> MeshPoint {
        self.horizontal.point()
    }

    /// σ-level in the water column: 0 at the surface, −1 at the bed.
    pub fn sigma(&self) -> f64 {
        self.sigma
    }

    /// Depth below the surface (m) in a column of depth `depth`.
    pub fn depth_below_surface(&self, depth: f64) -> f64 {
        -self.sigma * depth
    }

    /// The particle's own vertical speed (m/s, positive up).
    pub fn vertical_speed(&self) -> f64 {
        self.vertical_speed
    }

    /// Time (s) since release, until it exits, settles or dies.
    pub fn age(&self) -> f64 {
        self.age
    }

    /// Development accrued by its behaviour (e.g. degree-days; see
    /// [`ParticleBehaviour3D::development_rate`]).
    pub fn development(&self) -> f64 {
        self.development
    }

    /// Swimming speed (m/s, positive up) its behaviour chose for the last
    /// step, on top of [`Self::vertical_speed`].
    pub fn swimming_speed(&self) -> f64 {
        self.swimming_speed
    }

    /// What the particle is doing.
    pub fn status(&self) -> ParticleStatus {
        self.horizontal.status()
    }

    /// Whether the particle is still in the domain (not exited).
    pub fn in_domain(&self) -> bool {
        self.horizontal.in_domain()
    }
}

/// `x` reflected into `[lo, hi]` (as often as needed).
#[inline]
fn reflect(mut x: f64, lo: f64, hi: f64) -> f64 {
    let width = hi - lo;
    if width <= 0.0 {
        return lo;
    }
    // Fold onto one period of the reflection, [lo, lo + 2·width)
    x = (x - lo).rem_euclid(2.0 * width);
    if x > width {
        lo + 2.0 * width - x
    } else {
        lo + x
    }
}

/// Advects and disperses particles through a 3D flow (see the module docs).
#[derive(Clone)]
pub struct ParticleTracker3D<'a> {
    horizontal: ParticleTracker2D<'a>,
    vertical_walk: bool,
    bed_settling: bool,
}

impl<'a> ParticleTracker3D<'a> {
    /// A tracker on `mesh` with basis `ops`: pure advection, walls
    /// reflecting and every other boundary open, no stranding, no vertical
    /// random walk, the bed reflecting.
    pub fn new(mesh: &'a Mesh2D, ops: &'a DGOperators2D) -> Self {
        Self {
            horizontal: ParticleTracker2D::new(mesh, ops),
            vertical_walk: false,
            bed_settling: false,
        }
    }

    /// Horizontal diffusivity (m²/s) of the random walk (see
    /// [`ParticleTracker2D::with_diffusivity`]).
    pub fn with_horizontal_diffusivity(mut self, diffusivity: f64) -> Self {
        self.horizontal = self.horizontal.with_diffusivity(diffusivity);
        self
    }

    /// Visser's vertical random walk with the field's eddy diffusivity
    /// ([`ParticleVelocity3D::diffusivity`]; see the module docs).
    pub fn with_vertical_random_walk(mut self) -> Self {
        self.vertical_walk = true;
        self
    }

    /// Particles that reach the bed stay there
    /// ([`ParticleStatus::Settled`]), instead of reflecting.
    pub fn with_bed_settling(mut self) -> Self {
        self.bed_settling = true;
        self
    }

    /// Particles in water shallower than `depth` (m) strand (see
    /// [`ParticleTracker2D::with_stranding_depth`]).
    pub fn with_stranding_depth(mut self, depth: f64) -> Self {
        self.horizontal = self.horizontal.with_stranding_depth(depth);
        self
    }

    /// The boundary tags that reflect particles (see
    /// [`ParticleTracker2D::with_reflecting`]).
    pub fn with_reflecting(mut self, tags: impl IntoIterator<Item = BoundaryTag>) -> Self {
        self.horizontal = self.horizontal.with_reflecting(tags);
        self
    }

    /// Seed of the random walks; each particle draws from its own stream.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.horizontal = self.horizontal.with_seed(seed);
        self
    }

    /// A particle at `position` and σ-level `sigma` (clamped to [−1, 0])
    /// with vertical speed `vertical_speed` (m/s, positive up), or `None` if
    /// the position is outside the mesh. `id` seeds its random numbers.
    pub fn release(
        &self,
        id: u64,
        position: [f64; 2],
        sigma: f64,
        vertical_speed: f64,
    ) -> Option<Particle3D> {
        assert!(vertical_speed.is_finite());
        Some(Particle3D {
            horizontal: self.horizontal.release(id, position)?,
            sigma: sigma.clamp(-1.0, 0.0),
            vertical_speed,
            age: 0.0,
            development: 0.0,
            swimming_speed: 0.0,
        })
    }

    /// Advance passive `particles` from `t` to `t + dt` through `field` (in
    /// parallel with the `parallel` feature; the result does not depend on
    /// the number of threads or the order of the particles).
    pub fn step(
        &self,
        particles: &mut [Particle3D],
        field: &impl ParticleVelocity3D,
        t: f64,
        dt: f64,
    ) {
        self.step_with(particles, field, &Passive, t, dt);
    }

    /// Advance `particles` from `t` to `t + dt` through `field`, swimming,
    /// developing and dying by `behaviour` (see the module docs; parallel
    /// and reproducible like [`Self::step`]).
    pub fn step_with(
        &self,
        particles: &mut [Particle3D],
        field: &impl ParticleVelocity3D,
        behaviour: &impl ParticleBehaviour3D,
        t: f64,
        dt: f64,
    ) {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            particles
                .par_iter_mut()
                .with_min_len(64)
                .for_each(|p| self.advance(p, field, behaviour, t, dt));
        }
        #[cfg(not(feature = "parallel"))]
        particles
            .iter_mut()
            .for_each(|p| self.advance(p, field, behaviour, t, dt));
    }

    /// The particle's behaviour over the step from `t`: it senses its
    /// surroundings at the start, chooses its swimming speed, ages and
    /// develops. Returns whether it is still alive.
    fn live(
        &self,
        p: &mut Particle3D,
        field: &impl ParticleVelocity3D,
        behaviour: &impl ParticleBehaviour3D,
        t: f64,
        dt: f64,
    ) -> bool {
        p.age += dt;
        if !behaviour.senses() {
            return true;
        }
        let (point, sigma) = (p.horizontal.point, p.sigma);
        let (column_depth, tracers) = self.with_weights(point, |w| {
            (
                field.depth(point.element, w, t),
                field.tracers(point.element, w, sigma, t),
            )
        });
        let surroundings = Surroundings {
            time: t,
            position: p.horizontal.position,
            depth: -sigma * column_depth.max(0.0),
            column_depth,
            temperature: tracers.map(|[temp, _]| temp),
            salinity: tracers.map(|[_, salt]| salt),
        };
        let draw = uniform(&mut p.horizontal.rng);
        p.swimming_speed = behaviour.swimming_speed(p.development, &surroundings, draw);
        p.development += behaviour.development_rate(&surroundings) * dt;
        if behaviour.expired(p.development, p.age) {
            p.swimming_speed = 0.0;
            p.horizontal.status = ParticleStatus::Dead;
            return false;
        }
        true
    }

    /// Call `f` with the nodal basis values at `point`.
    fn with_weights<R>(&self, point: MeshPoint, f: impl FnOnce(&[f64]) -> R) -> R {
        let ops = self.horizontal.ops();
        let mut weights = [0.0; MAX_NODES];
        let weights = &mut weights[..ops.n_nodes];
        ops.interpolation_weights_into(point.r, point.s, weights);
        f(weights)
    }

    /// `[u, v, dσ/dt]` of the flow plus the particle's own vertical speed.
    fn velocity(
        &self,
        field: &impl ParticleVelocity3D,
        point: MeshPoint,
        sigma: f64,
        w_p: f64,
        t: f64,
    ) -> [f64; 3] {
        self.with_weights(point, |w| {
            let [u, v, sdot] = field.velocity(point.element, w, sigma, t);
            let depth = field.depth(point.element, w, t);
            let own = if depth > 0.0 { w_p / depth } else { 0.0 };
            [u, v, sdot + own]
        })
    }

    /// One step of one particle (see the module docs).
    fn advance(
        &self,
        p: &mut Particle3D,
        field: &impl ParticleVelocity3D,
        behaviour: &impl ParticleBehaviour3D,
        t: f64,
        dt: f64,
    ) {
        let depth_at = |point: MeshPoint, time: f64| {
            self.with_weights(point, |w| field.depth(point.element, w, time))
        };
        match p.horizontal.status {
            ParticleStatus::Exited(_) | ParticleStatus::Settled | ParticleStatus::Dead => return,
            ParticleStatus::Stranded => {
                // Aground it still lives (on a drying flat), but does not swim
                if self.live(p, field, behaviour, t, dt) {
                    p.swimming_speed = 0.0;
                    let h = &mut p.horizontal;
                    if !self.horizontal.too_shallow(Some(depth_at(h.point, t + dt))) {
                        h.status = ParticleStatus::Active;
                    }
                }
                return;
            }
            ParticleStatus::Active => {}
        }
        if !self.live(p, field, behaviour, t, dt) {
            return;
        }
        let h = &mut p.horizontal;
        // RK4 on (x, y, σ); the stage points walk from the particle and
        // reflect off walls, the surface and the bed
        let (half, w_p, sigma) = (0.5 * dt, p.vertical_speed + p.swimming_speed, p.sigma);
        let stage = |k: [f64; 3], step: f64| {
            let point = self
                .horizontal
                .walk_by(h, [step * k[0], step * k[1]])
                .point();
            (point, reflect(sigma + step * k[2], -1.0, 0.0))
        };
        let k1 = self.velocity(field, h.point, sigma, w_p, t);
        let (point, s) = stage(k1, half);
        let k2 = self.velocity(field, point, s, w_p, t + half);
        let (point, s) = stage(k2, half);
        let k3 = self.velocity(field, point, s, w_p, t + half);
        let (point, s) = stage(k3, dt);
        let k4 = self.velocity(field, point, s, w_p, t + dt);
        let mut displacement =
            [0, 1, 2].map(|d| dt / 6.0 * (k1[d] + 2.0 * k2[d] + 2.0 * k3[d] + k4[d]));
        let element = h.point.element;
        let field_k = self.with_weights(h.point, |w| {
            field.horizontal_diffusivity(element, w, sigma, t)
        });
        let depth = self.horizontal.walk_depth(
            self.horizontal.walk_diffusivity(field_k),
            |w, g| {
                Some((
                    field.depth(element, w, t),
                    field.depth_gradient(element, g, t)?,
                ))
            },
            h.point,
        );
        self.horizontal
            .add_walk(&mut displacement[..2], &mut h.rng, field_k, depth, dt);
        match self
            .horizontal
            .walk_by(h, [displacement[0], displacement[1]])
        {
            WalkEnd::Inside { position, point } => {
                h.position = position;
                h.point = point;
            }
            WalkEnd::Exited {
                position,
                point,
                tag,
            } => {
                h.position = position;
                h.point = point;
                h.status = ParticleStatus::Exited(tag);
                return;
            }
        }
        let mut sigma = sigma + displacement[2];
        if sigma < -1.0 && self.bed_settling {
            p.sigma = -1.0;
            h.status = ParticleStatus::Settled;
            return;
        }
        sigma = reflect(sigma, -1.0, 0.0);

        // Visser's random walk in z = σD at the new position
        let point = h.point;
        let depth = depth_at(point, t + dt);
        if self.vertical_walk && depth > 0.0 {
            let k_at = |s: f64| {
                self.with_weights(point, |w| field.diffusivity(point.element, w, s, t + dt))
            };
            if let Some((k, dk)) = k_at(sigma) {
                let z = sigma * depth;
                let z_mid = reflect(z + 0.5 * dk * dt, -depth, 0.0);
                let k_mid = k_at(z_mid / depth).map_or(k, |(k, _)| k).max(0.0);
                let xi = standard_normal_pair(&mut h.rng)[0];
                let z_new = z + dk * dt + (2.0 * k_mid * dt).sqrt() * xi;
                if z_new < -depth && self.bed_settling {
                    p.sigma = -1.0;
                    h.status = ParticleStatus::Settled;
                    return;
                }
                sigma = reflect(z_new, -depth, 0.0) / depth;
            }
        }
        p.sigma = sigma;
        if self.horizontal.too_shallow(Some(depth)) {
            h.status = ParticleStatus::Stranded;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflection_folds_into_the_interval() {
        for (x, expect) in [
            (-0.5, -0.5),
            (0.2, -0.2),
            (-1.3, -0.7),
            (-2.5, -0.5),
            (2.2, -0.2),
            (-1.0, -1.0),
            (0.0, 0.0),
        ] {
            let got = reflect(x, -1.0, 0.0);
            assert!((got - expect).abs() < 1e-14, "{x} → {got}, not {expect}");
        }
    }
}

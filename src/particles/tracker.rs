//! Particle advection and dispersion (see the [module docs](super)).

use super::velocity::ParticleVelocity2D;
use super::walk::{WalkEnd, walk};
use crate::mesh::{BoundaryTag, Mesh2D, MeshPoint, PointLocator2D};
use crate::operators::DGOperators2D;

/// Largest element basis the tracker evaluates on the stack (order 15).
pub(super) const MAX_NODES: usize = 256;

/// What a particle is doing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParticleStatus {
    /// Moving with the flow.
    Active,
    /// On ground shallower than the stranding depth: held in place until
    /// the water there is deep enough again.
    Stranded,
    /// Left the domain through a boundary face that does not reflect (an
    /// open boundary), with that face's tag. It no longer moves.
    Exited(Option<BoundaryTag>),
    /// Reached the bed and stays there (3D tracking with bed settling,
    /// [`super::ParticleTracker3D::with_bed_settling`]): deposited feed or
    /// faeces. It no longer moves.
    Settled,
    /// Past the lifespan of its behaviour (3D tracking,
    /// [`super::ParticleBehaviour3D::expired`]): a larva dead of senescence
    /// or starvation. It no longer moves.
    Dead,
}

/// A particle: its position, where that is in the mesh, and its own random
/// number stream.
#[derive(Clone, Debug, PartialEq)]
pub struct Particle2D {
    id: u64,
    pub(super) position: [f64; 2],
    pub(super) point: MeshPoint,
    pub(super) status: ParticleStatus,
    /// SplitMix64 state
    pub(super) rng: u64,
}

impl Particle2D {
    /// Identifier given at release; it also seeds the particle's random
    /// numbers.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Physical position (mesh coordinates). For an exited particle, where
    /// it crossed the boundary.
    pub fn position(&self) -> [f64; 2] {
        self.position
    }

    /// Element and reference coordinates of the position.
    pub fn point(&self) -> MeshPoint {
        self.point
    }

    /// What the particle is doing.
    pub fn status(&self) -> ParticleStatus {
        self.status
    }

    /// Whether the particle is still in the domain (active or stranded).
    pub fn in_domain(&self) -> bool {
        !matches!(self.status, ParticleStatus::Exited(_))
    }
}

/// SplitMix64 (Steele, Lea & Flood 2014): the state advances by the golden
/// ratio and the output is a bijective mix of it.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Uniform in (0, 1): 53 random bits, offset by half a unit.
pub(super) fn uniform(state: &mut u64) -> f64 {
    ((splitmix64(state) >> 11) as f64 + 0.5) * (1.0 / (1u64 << 53) as f64)
}

/// Two independent standard normal numbers (Box–Muller).
pub(super) fn standard_normal_pair(state: &mut u64) -> [f64; 2] {
    let (u1, u2) = (uniform(state), uniform(state));
    let radius = (-2.0 * u1.ln()).sqrt();
    let (sin, cos) = (std::f64::consts::TAU * u2).sin_cos();
    [radius * cos, radius * sin]
}

/// What the 2D random walk spreads, which sets its drift where the depth
/// varies (see the [module docs](super)).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WalkTracer {
    /// A tracer mixed through the water column, `∂(hc)/∂t = ∇·(hK∇c)`: the
    /// walk drifts by `K∇h/h` (Dimou & Adams 1993), so a population in
    /// proportion to the depth stays so. The particles of a depth-averaged
    /// flow.
    #[default]
    DepthIntegrated,
    /// A tracer per unit area of the surface, `∂c/∂t = ∇·(K∇c)` (floating
    /// matter): no depth drift, so a uniform areal population stays uniform.
    Surface,
}

/// Advects and disperses particles through a DG velocity field (see the
/// [module docs](super)).
#[derive(Clone)]
pub struct ParticleTracker2D<'a> {
    ops: &'a DGOperators2D,
    locator: PointLocator2D<'a>,
    diffusivity: f64,
    walk_tracer: WalkTracer,
    stranding_depth: Option<f64>,
    reflecting: Vec<BoundaryTag>,
    seed: u64,
}

impl<'a> ParticleTracker2D<'a> {
    /// A tracker on `mesh` with basis `ops`: pure advection, walls
    /// ([`BoundaryTag::Wall`]) reflecting and every other boundary open, no
    /// stranding.
    pub fn new(mesh: &'a Mesh2D, ops: &'a DGOperators2D) -> Self {
        assert!(
            ops.n_nodes <= MAX_NODES,
            "order {} above the tracker's 15",
            ops.order
        );
        Self {
            ops,
            locator: PointLocator2D::new(mesh),
            diffusivity: 0.0,
            walk_tracer: WalkTracer::default(),
            stranding_depth: None,
            reflecting: vec![BoundaryTag::Wall],
            seed: 0,
        }
    }

    /// Horizontal diffusivity K (m²/s) of the random walk; 0 (the default)
    /// is pure advection.
    pub fn with_diffusivity(mut self, diffusivity: f64) -> Self {
        assert!(diffusivity >= 0.0 && diffusivity.is_finite());
        self.diffusivity = diffusivity;
        self
    }

    /// What the walk spreads: a depth-integrated tracer (the default) or a
    /// surface one (see [`WalkTracer`]).
    pub fn with_walk_tracer(mut self, tracer: WalkTracer) -> Self {
        self.walk_tracer = tracer;
        self
    }

    /// Particles on water shallower than `depth` (m) strand, and float
    /// again when it is deeper (needs a field with a depth, e.g.
    /// [`SWEVelocity2D`](super::SWEVelocity2D)).
    pub fn with_stranding_depth(mut self, depth: f64) -> Self {
        assert!(depth > 0.0);
        self.stranding_depth = Some(depth);
        self
    }

    /// The boundary tags that reflect particles; particles leave through
    /// every other boundary face. Faces without a tag always reflect.
    pub fn with_reflecting(mut self, tags: impl IntoIterator<Item = BoundaryTag>) -> Self {
        self.reflecting = tags.into_iter().collect();
        self
    }

    /// Seed of the random walk; each particle draws from its own stream,
    /// seeded by this and its id.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// The point locator the tracker releases particles with.
    pub fn locator(&self) -> &PointLocator2D<'a> {
        &self.locator
    }

    /// A particle at `position`, or `None` if that is outside the mesh.
    /// `id` seeds its random numbers, so give each particle its own.
    pub fn release(&self, id: u64, position: [f64; 2]) -> Option<Particle2D> {
        let point = self.locator.locate(position)?;
        let mut rng = self.seed ^ 0x6A09_E667_F3BC_C909;
        let mix = splitmix64(&mut rng);
        let mut rng = mix ^ id.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        splitmix64(&mut rng);
        Some(Particle2D {
            id,
            position,
            point,
            status: ParticleStatus::Active,
            rng,
        })
    }

    /// Advance `particles` from `t` to `t + dt` through `field` (in
    /// parallel with the `parallel` feature; the result does not depend on
    /// the number of threads or the order of the particles).
    pub fn step(
        &self,
        particles: &mut [Particle2D],
        field: &impl ParticleVelocity2D,
        t: f64,
        dt: f64,
    ) {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            particles
                .par_iter_mut()
                .with_min_len(64)
                .for_each(|p| self.advance(p, field, t, dt));
        }
        #[cfg(not(feature = "parallel"))]
        particles
            .iter_mut()
            .for_each(|p| self.advance(p, field, t, dt));
    }

    /// The walk's diffusivity: the tracker's constant plus the field's `K`.
    pub(super) fn walk_diffusivity(&self, field: Option<(f64, [f64; 2])>) -> f64 {
        self.diffusivity + field.map_or(0.0, |(k, _)| k.max(0.0))
    }

    /// Add the random walk over `dt` to `displacement`: the tracker's
    /// constant diffusivity plus the field's `K` with its drift `∇K Δt`, and
    /// with `depth` (`h` and `∇h` at the particle) the drift `K∇h/h Δt`
    /// (Euler–Maruyama at the start of the step; see
    /// [`super::diffusivity`]).
    pub(super) fn add_walk(
        &self,
        displacement: &mut [f64],
        rng: &mut u64,
        field: Option<(f64, [f64; 2])>,
        depth: Option<(f64, [f64; 2])>,
        dt: f64,
    ) {
        let k = self.walk_diffusivity(field);
        let drift = field.map_or([0.0; 2], |(_, gradient)| gradient);
        displacement[0] += drift[0] * dt;
        displacement[1] += drift[1] * dt;
        if let Some((h, gradient)) = depth
            && h > 0.0
        {
            displacement[0] += k * gradient[0] / h * dt;
            displacement[1] += k * gradient[1] / h * dt;
        }
        if k > 0.0 {
            let scale = (2.0 * k * dt).sqrt();
            let xi = standard_normal_pair(rng);
            displacement[0] += scale * xi[0];
            displacement[1] += scale * xi[1];
        }
    }

    /// Call `f` with the nodal basis values at `point`.
    pub(super) fn with_weights<R>(&self, point: MeshPoint, f: impl FnOnce(&[f64]) -> R) -> R {
        let mut weights = [0.0; MAX_NODES];
        let weights = &mut weights[..self.ops.n_nodes];
        self.ops
            .interpolation_weights_into(point.r, point.s, weights);
        f(weights)
    }

    /// Call `f` with the nodal basis values at `point` and their physical
    /// gradients `[∂ℓ/∂x, ∂ℓ/∂y]` there, through the element's bilinear map.
    pub(super) fn with_gradient_weights<R>(
        &self,
        point: MeshPoint,
        f: impl FnOnce(&[f64], [&[f64]; 2]) -> R,
    ) -> R {
        let n = self.ops.n_nodes;
        let mut weights = [0.0; MAX_NODES];
        let (mut d_x, mut d_y) = ([0.0; MAX_NODES], [0.0; MAX_NODES]);
        let (weights, d_x, d_y) = (&mut weights[..n], &mut d_x[..n], &mut d_y[..n]);
        self.ops
            .interpolation_weights_into(point.r, point.s, weights);
        self.ops
            .interpolation_gradient_weights_into(point.r, point.s, d_x, d_y);
        // x(r, s) = a + b r + c s + d r s
        let [[x0, y0], [x1, y1], [x2, y2], [x3, y3]] =
            self.locator.mesh().element_vertices(point.element);
        let (bx, by) = (0.25 * (-x0 + x1 + x2 - x3), 0.25 * (-y0 + y1 + y2 - y3));
        let (cx, cy) = (0.25 * (-x0 - x1 + x2 + x3), 0.25 * (-y0 - y1 + y2 + y3));
        let (dx, dy) = (0.25 * (x0 - x1 + x2 - x3), 0.25 * (y0 - y1 + y2 - y3));
        let (xr, xs) = (bx + dx * point.s, cx + dx * point.r);
        let (yr, ys) = (by + dy * point.s, cy + dy * point.r);
        let det = xr * ys - xs * yr;
        // From ∂/∂r, ∂/∂s to ∂/∂x = (y_s ∂/∂r − y_r ∂/∂s)/J and
        // ∂/∂y = (x_r ∂/∂s − x_s ∂/∂r)/J, in place
        for (gx, gy) in d_x.iter_mut().zip(d_y.iter_mut()) {
            let (dr, ds) = (*gx, *gy);
            *gx = (ys * dr - yr * ds) / det;
            *gy = (xr * ds - xs * dr) / det;
        }
        f(weights, [d_x, d_y])
    }

    /// `h` and `∇h` at `point` for the walk's depth drift (`depth` given the
    /// basis values and gradients there), when the walk spreads (`k > 0`) a
    /// depth-integrated tracer.
    pub(super) fn walk_depth(
        &self,
        k: f64,
        depth: impl FnOnce(&[f64], [&[f64]; 2]) -> Option<(f64, [f64; 2])>,
        point: MeshPoint,
    ) -> Option<(f64, [f64; 2])> {
        if k > 0.0 && self.walk_tracer == WalkTracer::DepthIntegrated {
            self.with_gradient_weights(point, depth)
        } else {
            None
        }
    }

    /// Basis of the tracked field.
    pub(super) fn ops(&self) -> &DGOperators2D {
        self.ops
    }

    /// Whether a boundary face with `tag` reflects.
    fn reflects(&self, tag: Option<BoundaryTag>) -> bool {
        tag.is_none_or(|tag| self.reflecting.contains(&tag))
    }

    /// Velocity of `field` at `point` and time `t`.
    fn velocity(&self, field: &impl ParticleVelocity2D, point: MeshPoint, t: f64) -> [f64; 2] {
        let mut weights = [0.0; MAX_NODES];
        let weights = &mut weights[..self.ops.n_nodes];
        self.ops
            .interpolation_weights_into(point.r, point.s, weights);
        field.velocity(point.element, weights, t)
    }

    /// Whether the water at `point` is too shallow to float in at `t`.
    fn aground(&self, field: &impl ParticleVelocity2D, point: MeshPoint, t: f64) -> bool {
        let mut weights = [0.0; MAX_NODES];
        let weights = &mut weights[..self.ops.n_nodes];
        self.ops
            .interpolation_weights_into(point.r, point.s, weights);
        self.too_shallow(field.depth(point.element, weights, t))
    }

    /// Whether a particle strands in water of depth `depth` (none: never).
    pub(super) fn too_shallow(&self, depth: Option<f64>) -> bool {
        self.stranding_depth
            .is_some_and(|limit| depth.is_some_and(|h| h < limit))
    }

    /// Walk from the particle's position by `displacement`.
    pub(super) fn walk_by(&self, p: &Particle2D, displacement: [f64; 2]) -> WalkEnd {
        let [x, y] = p.position;
        walk(
            self.locator.mesh(),
            p.point.element,
            p.position,
            [x + displacement[0], y + displacement[1]],
            |tag| self.reflects(tag),
        )
    }

    /// One step of one particle (see the module docs).
    fn advance(&self, p: &mut Particle2D, field: &impl ParticleVelocity2D, t: f64, dt: f64) {
        match p.status {
            ParticleStatus::Exited(_) | ParticleStatus::Settled | ParticleStatus::Dead => return,
            ParticleStatus::Stranded => {
                if !self.aground(field, p.point, t + dt) {
                    p.status = ParticleStatus::Active;
                }
                return;
            }
            ParticleStatus::Active => {}
        }
        // Classical RK4; the stage points are found by walking from the
        // particle, so they reflect off walls like the particle does
        let half = 0.5 * dt;
        let k1 = self.velocity(field, p.point, t);
        let stage = |k: [f64; 2], h: f64| self.walk_by(p, [h * k[0], h * k[1]]).point();
        let k2 = self.velocity(field, stage(k1, half), t + half);
        let k3 = self.velocity(field, stage(k2, half), t + half);
        let k4 = self.velocity(field, stage(k3, dt), t + dt);
        let mut displacement =
            [0, 1].map(|d| dt / 6.0 * (k1[d] + 2.0 * k2[d] + 2.0 * k3[d] + k4[d]));
        let element = p.point.element;
        let field_k = self.with_weights(p.point, |w| field.horizontal_diffusivity(element, w, t));
        let depth = self.walk_depth(
            self.walk_diffusivity(field_k),
            |w, g| {
                Some((
                    field.depth(element, w, t)?,
                    field.depth_gradient(element, g, t)?,
                ))
            },
            p.point,
        );
        self.add_walk(&mut displacement, &mut p.rng, field_k, depth, dt);
        match self.walk_by(p, displacement) {
            WalkEnd::Inside { position, point } => {
                p.position = position;
                p.point = point;
                if self.aground(field, point, t + dt) {
                    p.status = ParticleStatus::Stranded;
                }
            }
            WalkEnd::Exited {
                position,
                point,
                tag,
            } => {
                p.position = position;
                p.point = point;
                p.status = ParticleStatus::Exited(tag);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The physical gradient weights differentiate a linear field exactly on
    /// general (not parallelogram) quadrilaterals, and the value weights
    /// come with them.
    #[test]
    fn gradient_weights_are_exact_on_distorted_elements() {
        let mut mesh = Mesh2D::uniform_rectangle(0.0, 3.0, 0.0, 2.0, 3, 2);
        for v in &mut mesh.vertices {
            if (v[0] - 1.0).abs() < 1e-12 && (v[1] - 1.0).abs() < 1e-12 {
                *v = [1.3, 0.8];
            }
        }
        let ops = DGOperators2D::new(3);
        let tracker = ParticleTracker2D::new(&mesh, &ops);
        let f = |[x, y]: [f64; 2]| 3.0 * x - 2.0 * y + 1.0;
        for k in 0..mesh.n_elements {
            let k = crate::types::ElementIndex::new(k);
            let values: Vec<f64> = (0..ops.n_nodes)
                .map(|i| f(mesh.reference_to_physical(k, ops.nodes_r[i], ops.nodes_s[i])))
                .collect();
            for (r, s) in [(-0.6, 0.2), (0.9, -0.95), (0.0, 0.0)] {
                let point = MeshPoint { element: k, r, s };
                let (value, [gx, gy]) = tracker.with_gradient_weights(point, |w, [wx, wy]| {
                    let dot = |w: &[f64]| w.iter().zip(&values).map(|(a, b)| a * b).sum::<f64>();
                    (dot(w), [dot(wx), dot(wy)])
                });
                assert!((value - f(mesh.reference_to_physical(k, r, s))).abs() < 1e-12);
                assert!(
                    (gx - 3.0).abs() < 1e-11 && (gy + 2.0).abs() < 1e-11,
                    "{gx} {gy}"
                );
            }
        }
    }

    /// Box–Muller on SplitMix64: mean 0, variance 1, uncorrelated pairs.
    #[test]
    fn normal_numbers_have_unit_variance() {
        let mut state = 12345;
        let n = 200_000;
        let (mut sum, mut sum2, mut cross) = (0.0, 0.0, 0.0);
        for _ in 0..n / 2 {
            let [a, b] = standard_normal_pair(&mut state);
            sum += a + b;
            sum2 += a * a + b * b;
            cross += a * b;
        }
        let n = n as f64;
        assert!((sum / n).abs() < 0.01, "mean {}", sum / n);
        assert!((sum2 / n - 1.0).abs() < 0.01, "variance {}", sum2 / n);
        assert!(
            (cross / (0.5 * n)).abs() < 0.01,
            "correlation {}",
            cross / (0.5 * n)
        );
    }
}

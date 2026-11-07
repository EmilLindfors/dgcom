//! Particles of a 3D run, tracked on the solver thread in the 3D flow.
//!
//! [`Cloud3D`] releases three kinds from every cage, as `examples/farm_3d.rs` does
//! ([`KINDS`]): sea-lice larvae over the top 5 m, swimming by `SalmonLice` (up towards
//! the light of [`crate::scenario::ThreeD::light`], down out of fresh water) unless
//! passive; faeces sinking at 3 cm/s over the net's depth; feed pellets sinking at
//! 10 cm/s over the top 2 m. They move with `dgcom::particles::ParticleTracker3D`
//! (RK4 in the 3D velocity, linear in time between the solver's states, a horizontal
//! random walk and Visser's vertical walk in the model's eddy diffusivity); faeces and
//! feed settle on the bed.
//!
//! Each kind is stepped as one slice (a behaviour belongs to a species); the snapshot
//! lists the particles in release order, so particle `i` is the same in every snapshot.

use dgcom::mesh::{Bathymetry2D, Mesh2D, MeshPoint};
use dgcom::operators::DGOperators2D;
use dgcom::particles::{
    ClearSkyLight, Particle3D, ParticleTracker3D, SalmonLice, Solution3DVelocity,
};
use dgcom::solver::state::Solution3D;
use dgcom::source::NetCage;
use dgcom::vertical::SigmaGrid;

use crate::particles::{Lice, MAX_STEP, ParticleConfig, ParticleSnapshot, state_of};

/// A kind of particle: name, vertical speed (m/s, positive up), the depths it is
/// released over (m below the surface) and its colour (linear RGB).
pub struct Kind {
    pub name: &'static str,
    pub vertical_speed: f64,
    pub release_depths: [f64; 2],
    pub colour: [f32; 3],
}

pub const KINDS: [Kind; 3] = [
    Kind {
        name: "lice larvae",
        vertical_speed: 0.0,
        release_depths: [0.0, 5.0],
        colour: [1.0, 0.12, 0.45],
    },
    Kind {
        name: "faeces",
        vertical_speed: -0.03,
        release_depths: [0.0, 20.0],
        colour: [0.35, 0.16, 0.05],
    },
    Kind {
        name: "feed",
        vertical_speed: -0.10,
        release_depths: [0.0, 2.0],
        colour: [1.0, 0.75, 0.05],
    },
];
const LARVAE: usize = 0;

/// Below this depth (m) the field has no velocity or mixing.
const MIN_DEPTH: f64 = 0.05;

/// The particles of a 3D run (see the module docs).
pub struct Cloud3D<'a> {
    tracker: ParticleTracker3D<'a>,
    ops: &'a DGOperators2D,
    sigma: &'a SigmaGrid,
    bathymetry: &'a Bathymetry2D,
    lice: Option<SalmonLice<ClearSkyLight>>,
    /// The particles of each kind, and every particle's (kind, index) in release order
    kinds: [Vec<Particle3D>; 3],
    order: Vec<(u8, u32)>,
    born: Vec<f32>,
    /// Centre and radius of each cage's release disc
    sources: Vec<([f64; 2], f64)>,
    config: ParticleConfig,
    next_release: f64,
    previous: Option<(f64, Solution3D)>,
    weights: Vec<f64>,
}

impl<'a> Cloud3D<'a> {
    pub fn new(
        mesh: &'a Mesh2D,
        ops: &'a DGOperators2D,
        sigma: &'a SigmaGrid,
        bathymetry: &'a Bathymetry2D,
        cages: &[NetCage],
        light: ClearSkyLight,
        config: ParticleConfig,
    ) -> Self {
        let sources = cages
            .iter()
            .map(|cage| {
                let (lo, hi) = cage.footprint.bounding_box();
                let centre = [0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])];
                (centre, 0.5 * (hi[0] - lo[0]).min(hi[1] - lo[1]))
            })
            .collect();
        let lice = match config.lice {
            Lice::Ladim => Some(SalmonLice::ladim(light)),
            Lice::Johnsen => Some(SalmonLice::johnsen_2014(light)),
            Lice::Passive => None,
        };
        Self {
            tracker: ParticleTracker3D::new(mesh, ops)
                .with_horizontal_diffusivity(config.kh)
                .with_vertical_random_walk()
                .with_bed_settling(),
            ops,
            sigma,
            bathymetry,
            lice,
            kinds: Default::default(),
            order: Vec::new(),
            born: Vec::new(),
            sources,
            config,
            next_release: 0.0,
            previous: None,
            weights: vec![0.0; ops.n_nodes],
        }
    }

    /// Move the particles to `state` at `t`, then release the batches due.
    pub fn advance(&mut self, state: &Solution3D, t: f64) {
        if let Some((t0, s0)) = &self.previous
            && t > *t0
        {
            let field = Solution3DVelocity::between(
                *t0,
                s0,
                t,
                state,
                self.sigma,
                self.bathymetry,
                MIN_DEPTH,
            );
            let steps = ((t - t0) / MAX_STEP).ceil().max(1.0) as usize;
            let dt = (t - t0) / steps as f64;
            for s in 0..steps {
                let at = t0 + s as f64 * dt;
                for (kind, particles) in self.kinds.iter_mut().enumerate() {
                    match &self.lice {
                        Some(lice) if kind == LARVAE => {
                            self.tracker.step_with(particles, &field, lice, at, dt)
                        }
                        _ => self.tracker.step(particles, &field, at, dt),
                    }
                }
            }
        }
        match &mut self.previous {
            Some((tp, sp)) => {
                *tp = t;
                sp.clone_from(state);
            }
            None => self.previous = Some((t, state.clone())),
        }
        while self.next_release <= t + 1e-9 {
            self.release(state, t);
            self.next_release += self.config.release_every;
        }
    }

    /// One batch of every kind in each cage: spread over its disc (a sunflower pattern
    /// that turns from batch to batch) and evenly over the kind's release depths.
    fn release(&mut self, state: &Solution3D, t: f64) {
        const GOLDEN_ANGLE: f64 = 2.399_963_229_728_653;
        const PHI: f64 = 0.618_033_988_749_895;
        let n = self.config.per_release;
        for c in 0..self.sources.len() {
            let (centre, radius) = self.sources[c];
            for (kind, spec) in KINDS.iter().enumerate() {
                for i in 0..n {
                    let j = self.order.len();
                    let r = radius * ((j as f64 + 0.5) * PHI).fract().sqrt();
                    let angle = j as f64 * GOLDEN_ANGLE;
                    let p = [centre[0] + r * angle.cos(), centre[1] + r * angle.sin()];
                    let [top, bottom] = spec.release_depths;
                    let below = top + (bottom - top) * (i as f64 + 0.5) / n as f64;
                    // Placed at the surface first, to find the depth there
                    let Some(probe) = self.tracker.release(j as u64, p, 0.0, 0.0) else {
                        continue;
                    };
                    let (_, depth) = column_at(
                        self.ops,
                        self.bathymetry,
                        &mut self.weights,
                        state,
                        probe.point(),
                    );
                    let sigma = if depth > 0.0 { -below / depth } else { 0.0 };
                    let Some(particle) =
                        self.tracker
                            .release(j as u64, p, sigma, spec.vertical_speed)
                    else {
                        continue;
                    };
                    self.order.push((kind as u8, self.kinds[kind].len() as u32));
                    self.kinds[kind].push(particle);
                    self.born.push(t as f32);
                }
            }
        }
    }

    /// Where every particle is: its position and height in the water column.
    pub fn snapshot(&mut self, state: &Solution3D) -> ParticleSnapshot {
        let len = self.order.len();
        let mut out = ParticleSnapshot {
            xy: Vec::with_capacity(len),
            z: Vec::with_capacity(len),
            state: Vec::with_capacity(len),
            born: self.born.clone(),
            kind: Vec::with_capacity(len),
        };
        for o in 0..len {
            let (kind, index) = self.order[o];
            let p = &self.kinds[kind as usize][index as usize];
            let [x, y] = p.position();
            let (eta, depth) = column_at(
                self.ops,
                self.bathymetry,
                &mut self.weights,
                state,
                p.point(),
            );
            out.xy.push([x as f32, y as f32]);
            out.z.push((eta + p.sigma() * depth.max(0.0)) as f32);
            out.state.push(state_of(p.status()));
            out.kind.push(kind);
        }
        out
    }
}

/// Surface elevation and water depth of `state` at `point`.
fn column_at(
    ops: &DGOperators2D,
    bathymetry: &Bathymetry2D,
    weights: &mut [f64],
    state: &Solution3D,
    point: MeshPoint,
) -> (f64, f64) {
    ops.interpolation_weights_into(point.r, point.s, weights);
    let k = point.element.as_usize();
    let n = ops.n_nodes;
    let eta = &state.eta.data[k * n..(k + 1) * n];
    let bed = bathymetry.element(point.element);
    let (mut e, mut b) = (0.0, 0.0);
    for i in 0..n {
        e += weights[i] * eta[i];
        b += weights[i] * bed[i];
    }
    (e, e - b)
}

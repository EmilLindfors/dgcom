//! Particles released from the cages (sea-lice larvae, feed, faeces): tracked on the
//! solver thread, drawn riding the water surface (2D) or at their depth (3D).
//!
//! [`Cloud`] lives on the solver thread of a 2D run. At every snapshot it moves its particles from
//! the previous snapshot to this one with `dg_rs::particles::ParticleTracker2D` (RK4
//! on the depth-averaged current, linear in time between the two states, plus a
//! horizontal random walk), then releases a new batch spread over each cage's
//! footprint, and reports every particle's position, the surface η there, its state
//! and its release time ([`ParticleSnapshot`]). Particles are only ever appended, so
//! particle `i` is the same in every snapshot that has it. A 3D run's cloud
//! ([`crate::cloud_3d::Cloud3D`]) reports each particle's height in the water column and
//! its kind as well.
//!
//! The viewer interpolates the positions linearly between the snapshots around the
//! shown time and draws each particle as a small octahedron: in 2D just above the
//! surface, coloured by age (young bright, old dark; stranded grey); in 3D at its
//! height, coloured by kind (lice larvae pink, faeces brown, feed yellow; dark on the
//! bed, grey when dead). Particles that left through the open boundary are not drawn.
//! Their size follows the camera's distance, so the cloud stays visible from the farm
//! to the whole fjord. P toggles them.

use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::NoFrustumCulling;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;
use dg_rs::mesh::Mesh2D;
use dg_rs::operators::DGOperators2D;
use dg_rs::particles::{Particle2D, ParticleStatus, ParticleTracker2D, SWEVelocity2D};
use dg_rs::solver::{SWESolution2D, WetDryConfig};
use dg_rs::source::NetCage;

use crate::actions::{Act, Action};
use crate::camera::OrbitCamera;
use crate::cloud_3d::KINDS;
use crate::colormap::{Lut, PLASMA};
use crate::field::Frame;
use crate::playback::Playback;
use crate::screen::{RUN, RunResources, Screen, Viewing};

/// Longest tracking step (s); a snapshot interval is split into steps no longer.
pub const MAX_STEP: f64 = 10.0;

/// The mesh's periods along x and y (m) if it is periodic: a particle carried across
/// is not interpolated through the domain.
#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct Periodic(pub Option<[f32; 2]>);
/// Particles in water shallower than this (m) strand.
const STRANDING_DEPTH: f64 = 0.05;

/// Particle states as sent to the viewer.
pub const ACTIVE: u8 = 0;
pub const STRANDED: u8 = 1;
pub const EXITED: u8 = 2;
/// On the bed (3D: sunk feed and faeces)
pub const SETTLED: u8 = 3;
/// Past its lifespan (3D: lice larvae)
pub const DEAD: u8 = 4;
pub const STATES: usize = 5;

/// What the lice larvae of a 3D run do (`dg_rs::particles::SalmonLice`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lice {
    /// The operational IMR model's parameters
    Ladim,
    /// Johnsen et al. (2014)
    Johnsen,
    /// Neutrally buoyant, no swimming
    Passive,
}

/// What is released and how it disperses.
#[derive(Clone, Copy, Debug)]
pub struct ParticleConfig {
    /// Particles released in each cage per release (in 3D, of each kind); 0 for none
    pub per_release: usize,
    /// Model time between releases (s)
    pub release_every: f64,
    /// Horizontal diffusivity of the random walk (m²/s)
    pub kh: f64,
    /// The lice larvae's behaviour (3D)
    pub lice: Lice,
}

/// Every particle at one snapshot, in release order.
#[derive(Default)]
pub struct ParticleSnapshot {
    /// Position, mesh coordinates (m)
    pub xy: Vec<[f32; 2]>,
    /// Height (m): the surface at the particle in 2D, the particle's own in 3D
    pub z: Vec<f32>,
    /// [`ACTIVE`], [`STRANDED`], [`EXITED`], [`SETTLED`] or [`DEAD`]
    pub state: Vec<u8>,
    /// Model time of release (s)
    pub born: Vec<f32>,
    /// Kind of every particle (3D, an index into `crate::cloud_3d::KINDS`); empty in 2D
    pub kind: Vec<u8>,
}

impl ParticleSnapshot {
    pub fn len(&self) -> usize {
        self.xy.len()
    }

    pub fn bytes(&self) -> usize {
        self.len() * (8 + 4 + 1 + 4) + self.kind.len()
    }

    /// Particles in each state.
    pub fn counts(&self) -> [usize; STATES] {
        let mut counts = [0; STATES];
        for &s in &self.state {
            counts[s as usize] += 1;
        }
        counts
    }
}

/// The 2D tracker's status as sent to the viewer.
pub fn state_of(status: ParticleStatus) -> u8 {
    match status {
        ParticleStatus::Active => ACTIVE,
        ParticleStatus::Stranded => STRANDED,
        ParticleStatus::Exited(_) => EXITED,
        ParticleStatus::Settled => SETTLED,
        ParticleStatus::Dead => DEAD,
    }
}

/// The particles, on the solver thread (see the module docs).
pub struct Cloud<'a> {
    tracker: ParticleTracker2D<'a>,
    ops: &'a DGOperators2D,
    particles: Vec<Particle2D>,
    born: Vec<f32>,
    /// Centre and radius of each cage's release disc
    sources: Vec<([f64; 2], f64)>,
    config: ParticleConfig,
    next_release: f64,
    /// The previous snapshot's state
    previous: Option<(f64, SWESolution2D)>,
    weights: Vec<f64>,
}

impl<'a> Cloud<'a> {
    pub fn new(
        mesh: &'a Mesh2D,
        ops: &'a DGOperators2D,
        cages: &[NetCage],
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
        Self {
            tracker: ParticleTracker2D::new(mesh, ops)
                .with_diffusivity(config.kh)
                .with_stranding_depth(STRANDING_DEPTH),
            ops,
            particles: Vec::new(),
            born: Vec::new(),
            sources,
            config,
            next_release: 0.0,
            previous: None,
            weights: vec![0.0; ops.n_nodes],
        }
    }

    /// Move the particles to the state `q` at `t`, then release the batches due.
    pub fn advance(&mut self, q: &SWESolution2D, t: f64) {
        if let Some((t0, q0)) = &self.previous
            && t > *t0
        {
            let field = SWEVelocity2D::between(*t0, q0, t, q, WetDryConfig::DEFAULT_H_DRY);
            let steps = ((t - t0) / MAX_STEP).ceil().max(1.0) as usize;
            let dt = (t - t0) / steps as f64;
            for s in 0..steps {
                self.tracker
                    .step(&mut self.particles, &field, t0 + s as f64 * dt, dt);
            }
        }
        match &mut self.previous {
            Some((tp, qp)) => {
                *tp = t;
                qp.clone_from(q);
            }
            None => self.previous = Some((t, q.clone())),
        }
        while self.next_release <= t + 1e-9 {
            self.release(t);
            self.next_release += self.config.release_every;
        }
    }

    /// One batch in each cage, spread over its disc (a sunflower pattern that turns
    /// from batch to batch).
    fn release(&mut self, t: f64) {
        const GOLDEN_ANGLE: f64 = 2.399_963_229_728_653;
        const PHI: f64 = 0.618_033_988_749_895;
        let n = self.config.per_release;
        for &(centre, radius) in &self.sources {
            let first = self.particles.len();
            for j in first..first + n {
                let r = radius * ((j as f64 + 0.5) * PHI).fract().sqrt();
                let angle = j as f64 * GOLDEN_ANGLE;
                let p = [centre[0] + r * angle.cos(), centre[1] + r * angle.sin()];
                if let Some(particle) = self.tracker.release(self.particles.len() as u64, p) {
                    self.particles.push(particle);
                    self.born.push(t as f32);
                }
            }
        }
    }

    /// Where every particle is, and the surface there.
    pub fn snapshot(&mut self, q: &SWESolution2D, bed: &[f64]) -> ParticleSnapshot {
        let n = self.ops.n_nodes;
        let mut out = ParticleSnapshot {
            xy: Vec::with_capacity(self.particles.len()),
            z: Vec::with_capacity(self.particles.len()),
            state: Vec::with_capacity(self.particles.len()),
            born: self.born.clone(),
            kind: Vec::new(),
        };
        for p in &self.particles {
            let [x, y] = p.position();
            out.xy.push([x as f32, y as f32]);
            let point = p.point();
            self.ops
                .interpolation_weights_into(point.r, point.s, &mut self.weights);
            let k = point.element.as_usize();
            let (h, b) = (q.element_h(point.element), &bed[k * n..(k + 1) * n]);
            let eta: f64 = self
                .weights
                .iter()
                .zip(h.iter().zip(b))
                .map(|(w, (h, b))| w * (h + b))
                .sum();
            out.z.push(eta as f32);
            out.state.push(state_of(p.status()));
        }
        out
    }
}

/// The drawn cloud.
#[derive(Resource)]
pub struct Particles {
    pub on: bool,
    /// Particles in each state at the shown time
    pub counts: [usize; STATES],
    /// Active particles of each kind (3D) at the shown time
    pub kinds: Vec<usize>,
    /// Age (s) at the dark end of the colour map
    pub age_scale: f32,
    lut: Lut,
    mesh: Handle<Mesh>,
}

#[derive(Component)]
struct Cloudlet;

pub struct ParticlesPlugin;

impl Plugin for ParticlesPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(OnEnter(Screen::Viewer), spawn)
            .add_systems(
                Update,
                (toggle, draw.after(crate::field::interpolate)).in_set(Viewing),
            )
            .run_resource::<Particles>()
            .run_resource::<Periodic>();
    }
}

fn spawn(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mesh = meshes.add(
        Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, Vec::<[f32; 3]>::new())
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, Vec::<[f32; 3]>::new())
        .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, Vec::<[f32; 4]>::new())
        .with_inserted_indices(Indices::U32(Vec::new())),
    );
    commands.spawn((
        RUN,
        Name::new("Particles"),
        Cloudlet,
        Mesh3d(mesh.clone()),
        MeshMaterial3d(materials.add(StandardMaterial {
            unlit: true,
            ..default()
        })),
        // The cloud moves: a bounding box from its first frame would cull it wrongly.
        NoFrustumCulling,
        Visibility::Hidden,
    ));
    commands.insert_resource(Particles {
        on: true,
        counts: [0; STATES],
        kinds: Vec::new(),
        age_scale: 600.0,
        lut: Lut::new(PLASMA),
        mesh,
    });
}

fn toggle(mut acts: MessageReader<Act>, mut particles: ResMut<Particles>) {
    for _ in acts.read().filter(|a| a.0 == Action::Particles) {
        particles.on = !particles.on;
    }
}

/// The six corners of an octahedron and its eight faces.
const CORNERS: [Vec3; 6] = [
    Vec3::X,
    Vec3::NEG_X,
    Vec3::Y,
    Vec3::NEG_Y,
    Vec3::Z,
    Vec3::NEG_Z,
];
const FACES: [[u32; 3]; 8] = [
    [0, 2, 4],
    [4, 2, 1],
    [1, 2, 5],
    [5, 2, 0],
    [4, 3, 0],
    [1, 3, 4],
    [5, 3, 1],
    [0, 3, 5],
];

#[allow(clippy::too_many_arguments)] // a Bevy system's parameters are its queries
fn draw(
    mut particles: ResMut<Particles>,
    playback: Res<Playback>,
    frame: Res<Frame>,
    periodic: Res<Periodic>,
    camera: Query<&OrbitCamera>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut visibility: Query<&mut Visibility, With<Cloudlet>>,
) {
    let Ok(mut visible) = visibility.single_mut() else {
        return;
    };
    let Some((a, b, w)) = playback.bracket() else {
        return;
    };
    let (a, b) = (&a.particles, &b.particles);
    let t = playback.t as f32;
    let oldest = b.born.first().copied().unwrap_or(0.0);
    let age_scale = (t - oldest).max(600.0);
    let counts = b.counts();
    let mut kinds = vec![0; KINDS.len()];
    for (&kind, &state) in b.kind.iter().zip(&b.state) {
        if state == ACTIVE {
            kinds[kind as usize] += 1;
        }
    }
    if b.kind.is_empty() {
        kinds.clear();
    }
    if particles.counts != counts || particles.age_scale != age_scale || particles.kinds != kinds {
        particles.counts = counts;
        particles.age_scale = age_scale;
        particles.kinds = kinds;
    }
    if !particles.on || b.len() == 0 {
        visible.set_if_neq(Visibility::Hidden);
        return;
    }
    visible.set_if_neq(Visibility::Inherited);

    // A few pixels across at any distance: about 1.3 m at the farm view, 30 m over
    // the whole fjord.
    let size = camera
        .single()
        .map_or(1.5, |c| (0.003 * c.distance).clamp(0.4, 60.0));
    let grey = [0.55, 0.55, 0.55, 1.0];
    let in_3d = !b.kind.is_empty();
    // In 2D the particles ride just above the surface; in 3D they sit at their height
    let lift = if in_3d { 0.0 } else { 0.6 * size };

    let n = b.len();
    let mut positions = Vec::with_capacity(6 * n);
    let mut normals = Vec::with_capacity(6 * n);
    let mut colours = Vec::with_capacity(6 * n);
    let mut indices = Vec::with_capacity(24 * n);
    for i in 0..n {
        if b.state[i] == EXITED || b.born[i] > t + 1e-3 {
            continue;
        }
        // Released since the earlier snapshot, or carried across a periodic boundary:
        // at its later position.
        let wrapped = periodic.0.is_some_and(|[lx, ly]| {
            i < a.len()
                && ((a.xy[i][0] - b.xy[i][0]).abs() > 0.5 * lx
                    || (a.xy[i][1] - b.xy[i][1]).abs() > 0.5 * ly)
        });
        let (xy, z) = if i < a.len() && a.state[i] != EXITED && !wrapped {
            let lerp = |x: f32, y: f32| x + w * (y - x);
            (
                [lerp(a.xy[i][0], b.xy[i][0]), lerp(a.xy[i][1], b.xy[i][1])],
                lerp(a.z[i], b.z[i]),
            )
        } else {
            (b.xy[i], b.z[i])
        };
        let centre = frame.world([xy[0] as f64, xy[1] as f64], z) + Vec3::Y * lift;
        let colour = match b.state[i] {
            STRANDED | DEAD => grey,
            state if in_3d => {
                let [r, g, bl] = KINDS[b.kind[i] as usize].colour;
                // On the bed: darker
                let k = if state == SETTLED { 0.35 } else { 1.0 };
                [k * r, k * g, k * bl, 1.0]
            }
            _ => particles.lut.at(1.0 - (t - b.born[i]) / age_scale),
        };
        let base = positions.len() as u32;
        for corner in CORNERS {
            positions.push((centre + corner * size).to_array());
            normals.push(corner.to_array());
            colours.push(colour);
        }
        indices.extend(FACES.iter().flatten().map(|&v| base + v));
    }
    let Some(mut mesh) = meshes.get_mut(&particles.mesh) else {
        return;
    };
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colours);
    mesh.insert_indices(Indices::U32(indices));
}

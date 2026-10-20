//! The solver, run on a thread of its own while the viewer draws.
//!
//! [`spawn`] builds the scenario's physics and runs `Simulation::run_with_callback`
//! with a callback every `interval` seconds of model time. Each callback reduces the
//! state to what the viewer draws, a [`Snapshot`] of the surface η and the
//! depth-averaged velocity (u, v) at every node, plus the particles released from the
//! cages ([`crate::particles::Cloud`], tracked here between snapshots), and sends it
//! down a channel. A scenario with a 3D model runs `Simulation3D` instead
//! (`Hydrostatic3D` with mode splitting), and its snapshots carry the layers too
//! ([`Layers`]: u, v and temperature on every σ-level) and particles tracked in 3D
//! ([`crate::cloud_3d::Cloud3D`]). The solver's own rayon pool leaves a couple of cores
//! to Bevy.

use std::sync::mpsc::{Receiver, channel};
use std::time::Instant;

use std::f64::consts::PI;
use std::sync::Arc;

use crate::replay::SnapshotSink;
use dg_rs::boundary::{
    CharacteristicOBC, HarmonicTide, MultiBoundaryCondition2D, Nesting3D, NestingBand3D,
    ReferenceColumns, Reflective2D, SWEBoundaryCondition2D,
};
use dg_rs::equations::ShallowWater2D;
use dg_rs::mesh::BoundaryTag;
use dg_rs::physics::{
    BottomDrag3D, ConstantMixing, EquationOfState, Forcing as Forcing3D, GlsMixing, Hydrostatic3D,
    ImplicitVerticalAdvection, LinearEOS, PhysicsBuilder, VerticalMixing,
};
use dg_rs::simulation::{Simulation, Simulation3D};
use dg_rs::solver::state::Solution3D;
use dg_rs::solver::{
    SWESolution2D, SWEState2D, StandardLimiter2D, TracerLimiter3DConfig, TracerLimiterType3D,
    TracerReferenceProfile, WetDryConfig,
};
use dg_rs::source::{
    CageDrag2D, CoriolisSource2D, HorizontalViscosity2D, ManningFriction2D, SourceContext2D,
    SourceTerm2D, SpongeProfile,
};
use dg_rs::time::{ModeSplitIntegrator, MultirateSSPRK3, SSPRK3};

use crate::cloud_3d::Cloud3D;
use crate::particles::{Cloud, ParticleConfig, ParticleSnapshot};
use crate::scenario::{Forcing, G, Mixing, Scenario, Tide};

/// Reference density of the 3D model (kg/m³).
const RHO0: f64 = 1025.0;
/// Spacing of the tracer limiter's reference profile's samples (m).
const REFERENCE_SPACING: f64 = 0.02;
/// Angular frequency of M2 (1/s).
const M2: f64 = 2.0 * PI / 44_714.16;

/// The tidal surface slope of a periodic channel as a depth-uniform body force along
/// mesh x, `∂(hu)/∂t = h·a(t)`, `a = U ω r(t) cos ωt`, with a smooth ramp `r` over
/// `ramp` seconds: without friction it drives the current `U sin ωt`.
struct TidalForce {
    current: f64,
    ramp: f64,
}

impl TidalForce {
    fn acceleration(&self, t: f64) -> f64 {
        let ramp = if t < self.ramp {
            0.5 * (1.0 - (PI * t / self.ramp).cos())
        } else {
            1.0
        };
        self.current * M2 * ramp * (M2 * t).cos()
    }
}

impl SourceTerm2D for TidalForce {
    fn evaluate(&self, ctx: &SourceContext2D) -> SWEState2D {
        SWEState2D::new(0.0, ctx.state.h * self.acceleration(ctx.time), 0.0)
    }

    fn name(&self) -> &'static str {
        "tidal_force"
    }
}

/// The condition at the open faces: a characteristic one with the tide (ramped up
/// over `forcing.ramp`), or a wall where a body force drives the flow.
fn open_boundary(forcing: &Forcing) -> Box<dyn SWEBoundaryCondition2D + Send + Sync> {
    let ramp = (forcing.ramp > 0.0).then_some(forcing.ramp);
    match &forcing.tide {
        Tide::UniformM2(amplitude) => {
            let tide = HarmonicTide::m2(*amplitude, 0.0);
            Box::new(CharacteristicOBC::new(match ramp {
                Some(r) => tide.with_ramp_up(r),
                None => tide,
            }))
        }
        Tide::Atlas(tides) => {
            let tides = tides.clone();
            Box::new(CharacteristicOBC::new(match ramp {
                Some(r) => tides.with_ramp_up(r),
                None => tides,
            }))
        }
        Tide::BodyForceM2(_) => Box::new(Reflective2D::new()),
    }
}

/// The snapshot file a run is saved to, if any, and the time of its last frame. A
/// snapshot no later than that (the run's first callback repeats t = 0) is not
/// written again; on the first error it says so and stops writing, and the run goes
/// on.
struct Save(Option<SnapshotSink>, f64);

impl Save {
    fn new(writer: Option<SnapshotSink>) -> Self {
        Self(writer, f64::NEG_INFINITY)
    }

    fn write(&mut self, snapshot: &Snapshot) {
        let Some(writer) = self.0.as_mut().filter(|_| snapshot.t > self.1) else {
            return;
        };
        match writer.write(snapshot) {
            Ok(()) => self.1 = snapshot.t,
            Err(e) => {
                eprintln!(
                    "cannot save the snapshot at t = {} s, no more saved: {e}",
                    snapshot.t
                );
                self.0 = None;
            }
        }
    }
}

/// Below this depth a node is dry: no velocity, and the viewer hides its surface.
pub const H_DRY: f32 = 0.01;

/// The model state at one instant, node by node in the solver's element-major order.
pub struct Snapshot {
    /// Model time (s)
    pub t: f64,
    /// Surface elevation η = h + B (m)
    pub eta: Vec<f32>,
    /// Depth-averaged velocity, mesh x (m/s)
    pub u: Vec<f32>,
    /// Depth-averaged velocity, mesh y (m/s)
    pub v: Vec<f32>,
    /// The particles released so far
    pub particles: ParticleSnapshot,
    /// The layers of a 3D run
    pub layers: Option<Layers>,
    /// The sea state of a run with waves
    pub waves: Option<Waves>,
}

/// The sea state at one instant, node by node: a coupled wave run's
/// (`dg_rs::waves::SeaStateFields`, a snapshot file's surface fields).
#[derive(Clone, Debug, Default)]
pub struct Waves {
    /// Significant wave height H_s (m)
    pub hs: Vec<f32>,
    /// Peak period T_p (s), 0 without waves
    pub tp: Vec<f32>,
    /// Mean direction travelling to, a unit vector in mesh x and y (0 without waves)
    pub dir_x: Vec<f32>,
    pub dir_y: Vec<f32>,
    /// Directional spread (rad)
    pub spread: Vec<f32>,
    /// The power whitecapping and depth-induced breaking take from the waves (W/m²)
    pub whitecapping: Vec<f32>,
    pub breaking: Vec<f32>,
}

impl Waves {
    /// The fields in the order of `dg_rs::waves::SeaStateFields::NAMES`.
    pub fn fields(&self) -> [&Vec<f32>; 7] {
        [
            &self.hs,
            &self.tp,
            &self.dir_x,
            &self.dir_y,
            &self.spread,
            &self.whitecapping,
            &self.breaking,
        ]
    }

    pub fn fields_mut(&mut self) -> [&mut Vec<f32>; 7] {
        [
            &mut self.hs,
            &mut self.tp,
            &mut self.dir_x,
            &mut self.dir_y,
            &mut self.spread,
            &mut self.whitecapping,
            &mut self.breaking,
        ]
    }

    fn bytes(&self) -> usize {
        self.fields().iter().map(|f| 4 * f.len()).sum()
    }
}

/// The 3D fields at one instant, `[node][level]` in the solver's element-major node
/// order, levels from the bed up.
pub struct Layers {
    pub n_levels: usize,
    /// Velocity along the σ-layers, mesh x and y (m/s)
    pub u: Vec<f32>,
    pub v: Vec<f32>,
    /// Temperature (°C)
    pub temp: Vec<f32>,
    /// Salinity; empty if the run's file has none
    pub salt: Vec<f32>,
}

impl Layers {
    fn bytes(&self) -> usize {
        4 * (self.u.len() + self.v.len() + self.temp.len() + self.salt.len())
    }
}

impl Snapshot {
    fn of(q: &SWESolution2D, bed: &[f64], t: f64, particles: ParticleSnapshot) -> Self {
        let [h, hu, hv] = &q.data;
        let velocity = |m: &[f64]| {
            h.iter()
                .zip(m)
                .map(|(&h, &m)| {
                    if h > H_DRY as f64 {
                        (m / h) as f32
                    } else {
                        0.0
                    }
                })
                .collect()
        };
        Self {
            t,
            eta: h.iter().zip(bed).map(|(h, b)| (h + b) as f32).collect(),
            u: velocity(hu),
            v: velocity(hv),
            particles,
            layers: None,
            waves: None,
        }
    }

    /// The surface, the depth-averaged velocity and the layers of a 3D state.
    fn of_3d(state: &Solution3D, t: f64, particles: ParticleSnapshot) -> Self {
        let f32s = |v: &[f64]| v.iter().map(|&x| x as f32).collect::<Vec<f32>>();
        Self {
            t,
            eta: f32s(&state.eta.data),
            u: f32s(&state.ubar.data),
            v: f32s(&state.vbar.data),
            particles,
            layers: Some(Layers {
                n_levels: state.n_levels,
                u: f32s(&state.u),
                v: f32s(&state.v),
                temp: f32s(&state.temp),
                salt: f32s(&state.salt),
            }),
            waves: None,
        }
    }

    pub fn bytes(&self) -> usize {
        4 * (self.eta.len() + self.u.len() + self.v.len())
            + self.particles.bytes()
            + self.layers.as_ref().map_or(0, Layers::bytes)
            + self.waves.as_ref().map_or(0, Waves::bytes)
    }
}

pub enum SolverMessage {
    Snapshot(Box<Snapshot>),
    /// A frame of a snapshot file read on demand ([`crate::replay::serve`])
    Frame(Box<Snapshot>),
    /// (model time, η) at the trace's probe of the next frames of a file read on
    /// demand
    Series(Vec<(f64, f32)>),
    /// Model times of frames a run has appended to a file read on demand
    Grown(Vec<f64>),
    Finished {
        steps: usize,
        wall: f64,
        error: Option<String>,
    },
}

#[derive(Clone, Copy, Debug)]
pub struct SolverConfig {
    /// End of the run (s of model time)
    pub t_end: f64,
    /// Model time between snapshots (s)
    pub interval: f64,
    /// Local time stepping levels (`MultirateSSPRK3`); 0 steps every element globally
    pub levels: usize,
    /// Threads of the solver's rayon pool
    pub threads: usize,
    /// Whether the cages' nets drag on the flow (they are drawn either way)
    pub drag: bool,
    /// Particles released from the cages
    pub particles: ParticleConfig,
}

/// Start the run; snapshots arrive on the returned channel, the first at t = 0.
/// With `save`, every snapshot is also written to that snapshot file
/// ([`crate::replay::writer`]).
pub fn spawn(
    scenario: &Scenario,
    config: SolverConfig,
    save: Option<SnapshotSink>,
) -> Receiver<SolverMessage> {
    if scenario.three_d.is_some() {
        return spawn_3d(scenario, config, save);
    }
    let (tx, rx) = channel();
    let mesh = scenario.mesh.clone();
    let ops = scenario.ops.clone();
    let geom = scenario.geom.clone();
    let bathymetry = scenario.bathymetry.clone();
    let cages = scenario.cages.clone();
    let forcing = scenario.forcing.clone();
    let mut q = scenario.at_rest();
    std::thread::Builder::new()
        .name("dg-rs solver".into())
        .spawn(move || {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(config.threads)
                .thread_name(|i| format!("dg-rs rayon {i}"))
                .build()
                .expect("solver thread pool");
            pool.install(|| {
                let wall = Reflective2D::new();
                let body_force = match forcing.tide {
                    Tide::BodyForceM2(current) => current,
                    _ => 0.0,
                };
                let sea = open_boundary(&forcing);
                let physics = PhysicsBuilder::swe_2d(
                    mesh.clone(),
                    ops.clone(),
                    geom.clone(),
                    ShallowWater2D::new(G),
                    MultiBoundaryCondition2D::new(&wall).with_open(sea.as_ref()),
                )
                .with_bathymetry(bathymetry.clone())
                .with_limiter(StandardLimiter2D::Positivity(WetDryConfig::DEFAULT_H_DRY))
                .with_wet_dry(WetDryConfig::default())
                .with_implicit_friction(ManningFriction2D::new(G, forcing.manning))
                .with_cage_drag(CageDrag2D::new(
                    &mesh,
                    &ops,
                    if config.drag { &cages } else { &[] },
                ))
                .with_source(CoriolisSource2D::f_plane(forcing.coriolis))
                .with_source(TidalForce {
                    current: body_force,
                    ramp: forcing.ramp,
                })
                .build();

                let started = Instant::now();
                let mut save = Save::new(save);
                let mut cloud = (config.particles.per_release > 0)
                    .then(|| Cloud::new(&mesh, &ops, &cages, config.particles));
                // A send fails only once the viewer has closed; the run then ends with the process.
                let mut send = |q: &SWESolution2D, t: f64| {
                    let particles = match cloud.as_mut() {
                        Some(cloud) => {
                            cloud.advance(q, t);
                            cloud.snapshot(q, &bathymetry.data)
                        }
                        None => ParticleSnapshot::default(),
                    };
                    let snapshot = Snapshot::of(q, &bathymetry.data, t, particles);
                    save.write(&snapshot);
                    let _ = tx.send(SolverMessage::Snapshot(Box::new(snapshot)));
                };
                send(&q, 0.0);
                let result = if config.levels > 0 {
                    Simulation::new(physics, MultirateSSPRK3::new(config.levels))
                        .with_cfl(1.0)
                        .with_callback_interval(config.interval)
                        .run_with_callback(&mut q, 0.0, config.t_end, &mut send)
                } else {
                    Simulation::new(physics, SSPRK3)
                        .with_cfl(1.0)
                        .with_callback_interval(config.interval)
                        .run_with_callback(&mut q, 0.0, config.t_end, &mut send)
                };
                let _ = tx.send(SolverMessage::Finished {
                    steps: result.n_steps,
                    wall: started.elapsed().as_secs_f64(),
                    error: result.error,
                });
            });
        })
        .expect("spawn the solver thread");
    rx
}

/// [`spawn`] for a scenario with a 3D model: the stratified channel of
/// `examples/farm_3d.rs` (GLS k-ε, log-layer bottom drag, background plus Smagorinsky
/// viscosity, net-cage drag per layer), from rest.
fn spawn_3d(
    scenario: &Scenario,
    config: SolverConfig,
    save: Option<SnapshotSink>,
) -> Receiver<SolverMessage> {
    let (tx, rx) = channel();
    let three_d = scenario
        .three_d
        .clone()
        .expect("spawn_3d runs a 3D scenario");
    let mesh = scenario.mesh.clone();
    let ops = scenario.ops.clone();
    let geom = scenario.geom.clone();
    let bathymetry = scenario.bathymetry.clone();
    let cages = scenario.cages.clone();
    let forcing = scenario.forcing.clone();
    std::thread::Builder::new()
        .name("dg-rs solver".into())
        .spawn(move || {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(config.threads)
                .thread_name(|i| format!("dg-rs rayon {i}"))
                .build()
                .expect("solver thread pool");
            pool.install(|| {
                let current = match forcing.tide {
                    Tide::BodyForceM2(current) => current,
                    _ => 0.0,
                };
                // The tide through the open boundary drives the barotropic mode, as
                // in 2D; a body force needs none
                let wall = Reflective2D::new();
                let sea = open_boundary(&forcing);
                // The 2D module owns the depth mean's forcing, Coriolis and viscosity;
                // friction and the cages are the 3D model's
                let swe = PhysicsBuilder::swe_2d(
                    mesh.clone(),
                    ops.clone(),
                    geom.clone(),
                    ShallowWater2D::new(G),
                    MultiBoundaryCondition2D::new(&wall).with_open(sea.as_ref()),
                )
                .with_bathymetry(bathymetry.clone())
                .with_source(TidalForce {
                    current,
                    ramp: forcing.ramp,
                })
                .with_source(CoriolisSource2D::f_plane(forcing.coriolis))
                .with_viscosity(HorizontalViscosity2D::constant(three_d.viscosity))
                .build();
                let eos = LinearEOS::default();
                let z0 = three_d.roughness;
                let profile = three_d.profile;
                // The tracers' horizontal limiter bounds their departure from the
                // stratification at rest, which a pycnocline crossing the σ-levels
                // over a sloping bed would otherwise clip. Cubic between its samples:
                // the limiter sees its interpolation error as a departure at rest,
                // and a linear one's drove 4e-5 m/s within 10 min under the fjord's
                // 3 m halocline at 4 cm (TODO F.3)
                let deepest = bathymetry.data.iter().copied().fold(0.0, f64::min);
                let (bottom, top) = (deepest - 1.0, 1.0);
                let limiter = TracerLimiter3DConfig {
                    limiter_type: TracerLimiterType3D::HorizontalKuzmin { relaxation: 1.0 },
                    ..TracerLimiter3DConfig::default()
                }
                .with_reference_profile(TracerReferenceProfile::from_smooth_fn(
                    bottom,
                    top,
                    ((top - bottom) / REFERENCE_SPACING).ceil() as usize + 1,
                    profile,
                ));
                let physics = Hydrostatic3D::new(
                    mesh.clone(),
                    ops.clone(),
                    geom.clone(),
                    three_d.sigma.clone(),
                    bathymetry.clone(),
                    Arc::new(CoriolisSource2D::f_plane(forcing.coriolis)),
                    eos,
                    match three_d.mixing {
                        Mixing::Gls => Box::new(GlsMixing::k_epsilon().with_roughness(0.02, z0))
                            as Box<dyn VerticalMixing + Send + Sync>,
                        Mixing::Constant {
                            viscosity,
                            diffusivity,
                        } => Box::new(ConstantMixing::new(viscosity, diffusivity)),
                    },
                    swe,
                    Forcing3D {
                        surface_stress: [0.0, 0.0],
                        bottom_stress: [0.0, 0.0],
                        surface_heat_flux: 0.0,
                    },
                    G,
                    RHO0,
                )
                .with_bottom_drag(BottomDrag3D::log_layer(z0))
                .with_horizontal_viscosity(three_d.viscosity)
                .with_smagorinsky_viscosity(three_d.smagorinsky)
                .with_tracer_limiter(limiter)
                // Thin surface layers over the deep water would set the step
                .with_implicit_vertical_advection(ImplicitVerticalAdvection::default());
                let mut physics = if config.drag {
                    physics.with_cage_drag(CageDrag2D::new(&mesh, &ops, &cages))
                } else {
                    physics
                };

                // At rest at mean sea level, stratified as the scenario's profile
                let n_levels = three_d.sigma.n_levels();
                let mut state = Solution3D::new(mesh.n_elements, ops.n_nodes, n_levels);
                for (k, (temp, salt)) in state
                    .temp
                    .chunks_exact_mut(n_levels)
                    .zip(state.salt.chunks_exact_mut(n_levels))
                    .enumerate()
                {
                    let depth = -bathymetry.data[k];
                    for ((t, s), &sigma) in temp.iter_mut().zip(salt).zip(three_d.sigma.sigma_rho())
                    {
                        (*t, *s) = profile(sigma * depth);
                    }
                }
                physics.update_density(&mut state);
                // The open faces let internal waves out: T and S relax to the
                // stratification at rest in a band along them (`Nesting3D`'s module
                // docs: without it an open face is unstable to a stratified flow)
                let (width, timescale) = three_d.open_band;
                if width > 0.0 {
                    let band = NestingBand3D {
                        width,
                        profile: SpongeProfile::default(),
                        velocity_timescale: Some(timescale),
                        tracer_timescale: Some(timescale),
                    };
                    match Nesting3D::new(
                        Arc::new(ReferenceColumns::from_state(&state)),
                        &mesh,
                        &ops,
                        n_levels,
                        &[BoundaryTag::Open],
                        &band,
                    ) {
                        Ok(nesting) => physics = physics.with_nesting(nesting),
                        Err(e) => eprintln!("no relaxation band at the open faces: {e}"),
                    }
                }
                // The baroclinic pressure gradient about the profile, exact at rest
                // (Mellor et al. 1998)
                let physics = physics.with_reference_profile(&state, |z| {
                    let (t, s) = profile(z);
                    eos.compute_density(t, s, z)
                });

                let started = Instant::now();
                let mut save = Save::new(save);
                let mut cloud = (config.particles.per_release > 0).then(|| {
                    Cloud3D::new(
                        &mesh,
                        &ops,
                        &three_d.sigma,
                        &bathymetry,
                        &cages,
                        three_d.light,
                        config.particles,
                    )
                });
                let last_sent = std::cell::Cell::new(f64::NEG_INFINITY);
                let mut send = |s: &Solution3D, t: f64| {
                    let particles = match cloud.as_mut() {
                        Some(cloud) => {
                            cloud.advance(s, t);
                            cloud.snapshot(s)
                        }
                        None => ParticleSnapshot::default(),
                    };
                    let snapshot = Snapshot::of_3d(s, t, particles);
                    save.write(&snapshot);
                    last_sent.set(t);
                    let _ = tx.send(SolverMessage::Snapshot(Box::new(snapshot)));
                };
                let result = Simulation3D::new(physics, ModeSplitIntegrator::new())
                    .with_cfl(0.5)
                    .with_callback_interval(config.interval)
                    .run_with_callback(&mut state, 0.0, config.t_end, &mut send);
                // `Simulation3D` calls back at the first step past each interval, so
                // the end of the run may not have been sent
                if result.final_time > last_sent.get() + 1e-6 {
                    send(&state, result.final_time);
                }
                let _ = tx.send(SolverMessage::Finished {
                    steps: result.n_steps,
                    wall: started.elapsed().as_secs_f64(),
                    error: result.error,
                });
            });
        })
        .expect("spawn the solver thread");
    rx
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::particles::Lice;
    use crate::scenario::Mixing;
    use dg_rs::time::ModelClock;

    /// The fjord farm in 3D on its own mesh, P1 on 8 σ-levels.
    fn fjord_3d() -> Scenario {
        let mesh = std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../tests/data/gmsh/fjord_farm_3d.msh"
        ));
        Scenario::fjord_farm_3d(mesh, 1, 8, ModelClock::default()).unwrap()
    }

    /// Run `scenario` for `hours` without cages or particles: the largest layer
    /// speed (m/s) and |η| (m) over its snapshots.
    fn run(scenario: &Scenario, hours: f64) -> (f64, f64) {
        let config = SolverConfig {
            t_end: hours * 3600.0,
            interval: 600.0,
            levels: 0,
            threads: std::thread::available_parallelism().map_or(1, |n| n.get()),
            drag: false,
            particles: ParticleConfig {
                per_release: 0,
                release_every: 60.0,
                kh: 0.0,
                lice: Lice::Passive,
            },
        };
        let (mut speed, mut eta) = (0.0_f64, 0.0_f64);
        for message in spawn(scenario, config, None) {
            match message {
                SolverMessage::Snapshot(s) => {
                    let layers = s.layers.as_ref().unwrap();
                    for (u, v) in layers.u.iter().zip(&layers.v) {
                        speed = speed.max(f64::from(u.hypot(*v)));
                    }
                    for e in &s.eta {
                        eta = eta.max(f64::from(e.abs()));
                    }
                }
                SolverMessage::Finished { error, .. } => {
                    assert!(error.is_none(), "{error:?}");
                    return (speed, eta);
                }
                _ => {}
            }
        }
        panic!("the solver ended without finishing");
    }

    /// The stratified fjord at rest, with its open boundary and band in place but
    /// no tide, stays at rest to round-off for an hour (3.6e-10 m/s): the PGF and
    /// the tracer limiter both take the profile as their reference. Mixing is
    /// constant without diffusion, so the stratification itself stays: with GLS
    /// (or a constant 1e-6 m²/s, GLS's background) the halocline diffuses over the
    /// sloping bed and drives 3.6e-5 m/s within the hour, a flow of the physics.
    /// With the limiter's reference linear between samples 4 cm apart this was
    /// 4e-5 m/s within 10 min (TODO F.3).
    #[test]
    fn the_stratified_fjord_stays_at_rest() {
        let mut scenario = fjord_3d();
        scenario.forcing.tide = Tide::UniformM2(0.0);
        scenario.three_d.as_mut().unwrap().mixing = Mixing::Constant {
            viscosity: 1e-3,
            diffusivity: 0.0,
        };
        let (speed, eta) = run(&scenario, 1.0);
        assert!(speed < 1e-8, "{speed:e} m/s");
        assert!(eta < 1e-10, "{eta:e} m");
    }

    /// The tide comes in through the open boundary and drives the 3D fjord.
    #[test]
    fn the_tide_enters_the_3d_fjord() {
        let mut scenario = fjord_3d();
        scenario.forcing.ramp = 600.0;
        let (speed, eta) = run(&scenario, 0.25);
        assert!(eta > 0.5, "{eta} m");
        assert!(speed > 0.02 && speed < 1.0, "{speed} m/s");
    }
}

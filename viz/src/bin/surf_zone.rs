//! Swell breaking on a plane beach, from the coupled wave–circulation model.
//!
//! The beach of the surf-zone gates in `tests/wave_coupling_test.rs`: a 1:100
//! slope from 4 m of water to a wall at 0.5 m, uniform alongshore, with 8 s swell
//! of H_rms 0.6 m arriving straight on or 20° off the normal. At start-up both
//! cases run to steady state in-process: the spectral wave model (Battjes–Janssen
//! breaking the only source) and the shallow-water circulation it drives through
//! its radiation stress. That takes a few seconds.
//!
//! The view animates what the phase-averaged model knows: at every point across
//! the beach its wave height H_rms, mean direction and wavenumber (from the depth
//! with the setup), and the fraction of breaking waves. The crests are
//! reconstructed from these, so their positions are illustrative while their
//! height, spacing, refraction and where they break are the model's: shoaling
//! waves steepen, breaking ones turn into bores with white water, and the mean
//! level sets down and up as the circulation computed. The white water follows
//! the share of its energy a wave loses to breaking over a period, `2 Q_b/β²`
//! in Battjes–Janssen: about 6 % at the breaker line, half near the wall. In the oblique case the
//! floats drift with the model's longshore current.
//!
//! ```bash
//! cd viz && cargo run --release --bin surf_zone -- [--vz 8] [--case oblique|straight]
//! ```
//!
//! Options: `--vz Z` vertical exaggeration [8]; `--case` the case shown first
//! [oblique]; `--rate R` model seconds per second [1]; `--view surf|whole` the
//! framing at the start [surf]; `--screenshot PATH` save a frame after `--at S`
//! seconds of animation [6] and quit.
//!
//! Keys: 1 straight on, 2 oblique; Space pause; [ and ] slower and faster; F the
//! surf zone close up, O the whole beach. Mouse: left-drag turns, right-drag pans,
//! the wheel zooms.

#[path = "../camera.rs"]
#[allow(dead_code)] // the viewer's named places: the beach has none
mod camera;

use std::f64::consts::{FRAC_PI_2, PI, TAU};
use std::sync::Arc;

use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::NoFrustumCulling;
use bevy::mesh::{Indices, PrimitiveTopology, VertexAttributeValues};
use bevy::prelude::*;
use bevy::render::view::screenshot::{Screenshot, save_to_disk};
use rayon::prelude::*;

use dg_rs::boundary::Reflective2D;
use dg_rs::equations::ShallowWater2D;
use dg_rs::mesh::{Bathymetry2D, BoundaryTag, Mesh2D};
use dg_rs::operators::{DGOperators2D, GeometricFactors2D};
use dg_rs::physics::{PhysicsBuilder, SWEPhysics2DBuilder};
use dg_rs::simulation::Simulation;
use dg_rs::solver::{SWESolution2D, SWEState2D};
use dg_rs::source::{ChezyFriction2D, SourceContext2D, SourceTerm2D, WaveForce2D};
use dg_rs::time::SSPRK3;
use dg_rs::types::ElementIndex;
use dg_rs::waves::{
    SourceTerms, SpectralGrid, WaveModel2D, WaveSolution, WaveWorkspace, breaking_fraction,
    wavenumber,
};

use camera::{CameraPlugin, OrbitCamera, Views};

const G: f64 = 9.81;
const PERIOD: f64 = 8.0;
const H_RMS: f64 = 0.6;
/// The beach: depth 4 m at y = 0 falling to 0.5 m at the wall, y = L (m)
const L: f64 = 350.0;
const H0: f64 = 4.0;
const H1: f64 = 0.5;
/// Breaking parameter of Battjes–Janssen
const GAMMA: f64 = 0.73;
/// Shown: cross-shore from `SEA` (offshore of the model, the swell as it enters)
/// to `LAND`, alongshore over `ALONG` (m)
const SEA: f64 = -250.0;
const LAND: f64 = 440.0;
const ALONG: f64 = 260.0;
/// Grid spacing of the water surface (m)
const SPACING: f64 = 1.0;

fn depth(y: f64) -> f64 {
    H0 + (H1 - H0) * y / L
}

// ---------------------------------------------------------------- the model

/// Linear damping of the cross-shore momentum, to settle the seiches.
struct CrossShoreDamping(f64);

impl SourceTerm2D for CrossShoreDamping {
    fn evaluate(&self, ctx: &SourceContext2D) -> SWEState2D {
        SWEState2D::new(0.0, 0.0, -self.0 * ctx.state.hv)
    }

    fn name(&self) -> &'static str {
        "cross_shore_damping"
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Case {
    Straight,
    Oblique,
}

impl Case {
    fn direction(self) -> f64 {
        match self {
            Case::Straight => FRAC_PI_2,
            Case::Oblique => FRAC_PI_2 - 20f64.to_radians(),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Case::Straight => "straight on",
            Case::Oblique => "20 deg oblique",
        }
    }
}

/// The steady state across the beach at the model's nodes, by distance from the
/// sea: H_rms (m), mean direction (rad, from the alongshore +x), mean level η
/// (m, relative to the seaward end) and the alongshore current (m/s).
struct Steady {
    y: Vec<f64>,
    hrms: Vec<f64>,
    direction: Vec<f64>,
    eta: Vec<f64>,
    current: Vec<f64>,
}

/// Run `case` to steady state (as the surf-zone gates do).
fn run(case: Case) -> Steady {
    let ny = 14;
    let mesh = Mesh2D::channel_periodic_x_with_sides(
        0.0,
        L / ny as f64,
        0.0,
        L,
        1,
        ny,
        [BoundaryTag::Open, BoundaryTag::Wall],
    );
    let ops = Arc::new(DGOperators2D::new(2));
    let geom = Arc::new(GeometricFactors2D::compute(&mesh, &ops));
    let bathymetry = Arc::new(Bathymetry2D::from_function(&mesh, &ops, &geom, |_, y| {
        -depth(y)
    }));
    let mesh = Arc::new(mesh);

    // The swell entering at y = 0
    let f = 1.0 / PERIOD;
    let n_dir = 72;
    let grid = SpectralGrid::new(f, 1.2 * f, 2, n_dir);
    let mut weights: Vec<f64> = grid
        .theta
        .iter()
        .map(|&t| {
            let c = (t - case.direction()).cos();
            match case {
                Case::Oblique => c.max(0.0).powi(20),
                Case::Straight if c > 1.0 - 1e-9 => 1.0,
                Case::Straight => 0.0,
            }
        })
        .collect();
    let total: f64 = weights.iter().sum::<f64>() * grid.d_theta;
    weights.iter_mut().for_each(|w| *w /= total);
    let mut e = vec![0.0; grid.n_components()];
    for (j, w) in weights.iter().enumerate() {
        e[grid.component(0, j)] = H_RMS * H_RMS / 8.0 / grid.d_sigma[0] * w;
    }
    let mut waves = WaveModel2D::new(
        mesh.clone(),
        ops.clone(),
        geom.clone(),
        &bathymetry,
        grid,
        G,
    )
    .with_sources(SourceTerms::none(G).with_breaking(Some((1.0, GAMMA))))
    .with_boundary_spectrum(&e);

    let settle = |waves: &WaveModel2D, n: &mut WaveSolution| {
        let mut ws = WaveWorkspace::default();
        let dt = waves.compute_dt(0.5);
        let steps = (4.0 * L / (2.0 * dt)).ceil() as usize;
        for _ in 0..steps {
            waves.step(n, dt, &mut ws);
        }
    };
    let physics = || -> SWEPhysics2DBuilder<Reflective2D> {
        PhysicsBuilder::swe_2d(
            mesh.clone(),
            ops.clone(),
            geom.clone(),
            ShallowWater2D::new(G),
            Reflective2D::new(),
        )
        .with_bathymetry(bathymetry.clone())
        .with_source(CrossShoreDamping(1e-2))
    };
    let nn = ops.n_nodes;
    let mut q = SWESolution2D::new(mesh.n_elements, nn);
    for k in ElementIndex::iter(mesh.n_elements) {
        for i in 0..nn {
            q.set_state(k, i, SWEState2D::new(-bathymetry.get(k, i), 0.0, 0.0));
        }
    }
    let eta_of = |q: &SWESolution2D| -> Vec<f64> {
        q.h_data()
            .iter()
            .zip(&bathymetry.data)
            .map(|(h, b)| h + b)
            .collect()
    };

    let mut n = waves.zero_state();
    match case {
        // Two-way: the waves see the level the circulation sets
        Case::Straight => {
            for pass in 0..3 {
                settle(&waves, &mut n);
                let force = WaveForce2D::new(&waves, &n);
                let force = if pass == 0 {
                    force.with_ramp(300.0)
                } else {
                    force
                };
                let result = Simulation::new(physics().with_source(force).build(), SSPRK3)
                    .run(&mut q, 0.0, 3000.0);
                assert!(result.success, "{result:?}");
                waves.set_water_level(&eta_of(&q));
            }
            settle(&waves, &mut n);
        }
        Case::Oblique => {
            settle(&waves, &mut n);
            let force = WaveForce2D::new(&waves, &n).with_ramp(300.0);
            let built = physics()
                .with_source(force)
                .with_implicit_friction(ChezyFriction2D::new(2.5e-3))
                .build();
            let result = Simulation::new(built, SSPRK3).run(&mut q, 0.0, 10_000.0);
            assert!(result.success, "{result:?}");
        }
    }

    // Node values by y (the field is uniform alongshore)
    let params = waves.parameters(&n);
    let eta = eta_of(&q);
    let mut rows: Vec<(f64, usize)> = ElementIndex::iter(mesh.n_elements)
        .flat_map(|k| {
            let (mesh, ops) = (&mesh, &ops);
            (0..nn).map(move |i| {
                let y = mesh.reference_to_physical(k, ops.nodes_r[i], ops.nodes_s[i])[1];
                (y, k.as_usize() * nn + i)
            })
        })
        .collect();
    rows.sort_by(|a, b| a.0.total_cmp(&b.0));
    rows.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-9);
    let eta0 = eta[rows[0].1];
    Steady {
        y: rows.iter().map(|r| r.0).collect(),
        hrms: rows.iter().map(|r| (8.0 * params[r.1].m0).sqrt()).collect(),
        direction: rows.iter().map(|r| params[r.1].direction).collect(),
        eta: rows.iter().map(|r| eta[r.1] - eta0).collect(),
        current: rows
            .iter()
            .map(|r| q.hu_data()[r.1] / q.h_data()[r.1])
            .collect(),
    }
}

fn interp(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    if x <= xs[0] {
        return ys[0];
    }
    let n = xs.len() - 1;
    if x >= xs[n] {
        return ys[n];
    }
    let i = xs.partition_point(|&v| v <= x) - 1;
    let t = (x - xs[i]) / (xs[i + 1] - xs[i]);
    ys[i] + t * (ys[i + 1] - ys[i])
}

/// A case on the surface grid's columns (every `SPACING` from `SEA` to the
/// wall): what the crests are drawn from.
struct Profile {
    case: Case,
    /// Alongshore wavenumber, constant by Snell's law (rad/m)
    kx: f64,
    /// Per column: cross-shore phase `∫ k_y dy`, wavenumber, amplitude
    /// (`H_rms/√2`), the fraction of its energy a wave loses to breaking over a
    /// period (Battjes–Janssen: `D T/E = 2 Q_b/β²`, β = H_rms/H_max), mean
    /// level, current
    psi: Vec<f64>,
    k: Vec<f64>,
    amplitude: Vec<f64>,
    breaking: Vec<f64>,
    eta: Vec<f64>,
    current: Vec<f64>,
    /// Summary for the status line
    facts: String,
}

impl Profile {
    fn new(case: Case, s: &Steady) -> Self {
        let sigma = TAU / PERIOD;
        let columns = ((L - SEA) / SPACING).round() as usize + 1;
        let theta0 = s.direction[0];
        let kx = wavenumber(sigma, depth(0.0), G) * theta0.cos();
        let mut p = Profile {
            case,
            kx,
            psi: Vec::with_capacity(columns),
            k: Vec::with_capacity(columns),
            amplitude: Vec::with_capacity(columns),
            breaking: Vec::with_capacity(columns),
            eta: Vec::with_capacity(columns),
            current: Vec::with_capacity(columns),
            facts: String::new(),
        };
        let mut psi = 0.0;
        let mut previous: Option<f64> = None;
        for c in 0..columns {
            let y = SEA + c as f64 * SPACING;
            let yc = y.max(0.0);
            let eta = interp(&s.y, &s.eta, yc);
            let d = depth(yc) + eta;
            let k = wavenumber(sigma, d, G);
            let ky = (k * k - kx * kx).max(0.0).sqrt();
            if let Some(prev) = previous {
                psi += 0.5 * (ky + prev) * SPACING;
            }
            previous = Some(ky);
            let h = interp(&s.y, &s.hrms, yc);
            p.psi.push(psi);
            p.k.push(k);
            p.amplitude.push(h / std::f64::consts::SQRT_2);
            let beta = h / (GAMMA * d);
            p.breaking
                .push(2.0 * breaking_fraction(beta) / (beta * beta).max(1e-12));
            p.eta.push(eta);
            p.current.push(interp(&s.y, &s.current, yc));
        }
        // The breaker line: the largest H_rms
        let (i_break, _) = s
            .hrms
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap();
        let low = s.eta.iter().cloned().fold(f64::INFINITY, f64::min);
        p.facts = match case {
            Case::Straight => format!(
                "Straight on: breakers at {:.0} m ({:.1} m deep), H_rms {:.2} m there\n\
                 set-down {:.1} cm at the breakers, setup {:.1} cm at the wall",
                s.y[i_break],
                depth(s.y[i_break]),
                s.hrms[i_break],
                -100.0 * low,
                100.0 * s.eta.last().unwrap(),
            ),
            Case::Oblique => {
                let fastest = s.current.iter().cloned().fold(0.0, f64::max);
                format!(
                    "20 deg oblique: breakers at {:.0} m; the waves turn from {:.0} to {:.0} deg off the normal\n\
                     longshore current up to {:.2} m/s (floats drift with it)",
                    s.y[i_break],
                    90.0 - s.direction[0].to_degrees(),
                    90.0 - s.direction.last().unwrap().to_degrees(),
                    fastest,
                )
            }
        };
        p
    }

    /// The column at cross-shore `y` (clamped to the grid).
    fn column(&self, y: f64) -> usize {
        (((y - SEA) / SPACING).round().max(0.0) as usize).min(self.psi.len() - 1)
    }
}

/// The shape of a wave at phase φ (crest at 0): a Stokes-like wave with
/// sharper crests (`skew`), blended into a bore with a steep front as the waves
/// break (`bore` 0–1). Zero mean, ±1 crest to trough.
fn shape(phi: f64, skew: f64, bore: f64) -> f64 {
    let regular = (phi.cos() + skew * (2.0 * phi).cos()) / (1.0 + skew);
    if bore <= 0.0 {
        return regular;
    }
    // Going along the waves' travel from a crest: a drop over `w`, then the
    // next wave's back rising slowly to its crest
    let w = 0.6;
    let p = phi.rem_euclid(TAU);
    let saw = if p < w {
        (PI * p / w).cos()
    } else {
        -1.0 + 2.0 * (p - w) / (TAU - w)
    };
    (1.0 - bore) * regular + bore * saw
}

fn hash(a: usize, b: usize) -> f64 {
    let s = ((a as f64) * 12.9898 + (b as f64) * 78.233).sin() * 43758.5453;
    s - s.floor()
}

// ---------------------------------------------------------------- the view

#[derive(Resource)]
struct Beach {
    profiles: [Profile; 2],
    shown: usize,
    t: f64,
    rate: f64,
    paused: bool,
    vz: f32,
    water: Handle<Mesh>,
}

impl Beach {
    fn profile(&self) -> &Profile {
        &self.profiles[self.shown]
    }
}

#[derive(Component)]
struct Float {
    y: f64,
    x: f64,
}

#[derive(Component)]
struct Status;

#[derive(Resource)]
struct Capture {
    path: Option<String>,
    at: f64,
    frames: Option<u32>,
}

struct Args {
    vz: f32,
    case: Case,
    rate: f64,
    screenshot: Option<String>,
    at: f64,
    whole: bool,
}

fn parse() -> Result<Args, String> {
    let mut args = Args {
        vz: 8.0,
        case: Case::Oblique,
        rate: 1.0,
        screenshot: None,
        at: 6.0,
        whole: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let value = it.next().ok_or_else(|| format!("{flag} needs a value"))?;
        let number = |v: &str| v.parse::<f64>().map_err(|e| format!("{flag}: {e}"));
        match flag.as_str() {
            "--vz" => args.vz = number(&value)? as f32,
            "--rate" => args.rate = number(&value)?,
            "--at" => args.at = number(&value)?,
            "--screenshot" => args.screenshot = Some(value),
            "--view" => args.whole = value == "whole",
            "--case" => {
                args.case = match value.as_str() {
                    "straight" => Case::Straight,
                    "oblique" => Case::Oblique,
                    _ => return Err(format!("--case straight|oblique, not {value}")),
                }
            }
            _ => return Err(format!("unknown option {flag}")),
        }
    }
    Ok(args)
}

const KEYS: &str = "1 straight on | 2 oblique | Space pause | [ ] slower, faster\n\
                    F surf zone | O whole beach | drag to turn, right-drag to pan, wheel to zoom";

fn main() {
    let args = match parse() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };
    println!("Running both cases to steady state (waves ↔ circulation)…");
    let start = std::time::Instant::now();
    let (straight, oblique) = std::thread::scope(|s| {
        let a = s.spawn(|| run(Case::Straight));
        let b = s.spawn(|| run(Case::Oblique));
        (a.join().unwrap(), b.join().unwrap())
    });
    println!("done in {:.1?}", start.elapsed());
    let profiles = [
        Profile::new(Case::Straight, &straight),
        Profile::new(Case::Oblique, &oblique),
    ];
    for p in &profiles {
        println!("{}", p.facts);
    }
    let shown = (args.case == Case::Oblique) as usize;

    // Scene: x = cross-shore y (sea at −, beach at +), z = −alongshore, up = Y
    let close = OrbitCamera {
        focus: Vec3::new(250.0, 0.0, -150.0),
        yaw: -1.05,
        pitch: 0.42,
        distance: 210.0,
    };
    let whole = OrbitCamera {
        focus: Vec3::new(120.0, 0.0, -130.0),
        yaw: -0.75,
        pitch: 0.8,
        distance: 620.0,
    };

    let first = if args.whole { whole } else { close };
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "dg-viz · surf zone".into(),
                resolution: (1600, 900).into(),
                ..default()
            }),
            ..default()
        }))
        .insert_resource(ClearColor(Color::srgb(0.62, 0.74, 0.84)))
        .insert_resource(GlobalAmbientLight {
            color: Color::srgb(0.8, 0.88, 1.0),
            brightness: 600.0,
            ..default()
        })
        .insert_resource(Views {
            farm: close,
            domain: whole,
            places: Vec::new(),
        })
        .insert_resource(Capture {
            path: args.screenshot,
            at: args.at,
            frames: None,
        })
        .insert_resource(Beach {
            profiles,
            shown,
            t: 0.0,
            rate: args.rate,
            paused: false,
            vz: args.vz,
            water: Handle::default(),
        })
        .add_plugins(CameraPlugin)
        .add_systems(Startup, spawn)
        .add_systems(Update, (keys, animate, drift, status, capture).chain())
        .add_systems(Startup, move |mut commands: Commands| {
            commands.spawn((
                Camera3d::default(),
                Projection::Perspective(PerspectiveProjection {
                    near: 0.5,
                    far: 20_000.0,
                    ..default()
                }),
                first.transform(),
                first,
                DistanceFog {
                    color: Color::srgb(0.62, 0.74, 0.84),
                    falloff: FogFalloff::Linear {
                        start: 600.0,
                        end: 2200.0,
                    },
                    ..default()
                },
            ));
            // Sun high over the sea, from the side
            commands.spawn((
                DirectionalLight {
                    illuminance: 11_000.0,
                    shadow_maps_enabled: false,
                    ..default()
                },
                Transform::from_xyz(-0.35, 1.0, 0.75).looking_at(Vec3::ZERO, Vec3::Y),
            ));
        })
        .run();
}

fn linear(c: Color) -> [f32; 4] {
    let l = c.to_linear();
    [l.red, l.green, l.blue, l.alpha]
}

fn spawn(
    mut commands: Commands,
    mut beach: ResMut<Beach>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let vz = beach.vz;
    let nz = (ALONG / SPACING).round() as usize + 1;

    // The bed: the slope continues past the wall and out of the water
    let bed_height = |y: f64| -> f64 {
        let y = y.max(0.0);
        let z = -depth(y);
        if z > 0.0 { z.min(1.6) } else { z }
    };
    let sand = Color::srgb(0.84, 0.74, 0.55);
    let wet_sand = Color::srgb(0.55, 0.47, 0.34);
    let deep = Color::srgb(0.30, 0.30, 0.25);
    {
        let step = 2.0;
        let nx = ((LAND - SEA) / step).round() as usize + 1;
        let nzb = (ALONG / step).round() as usize + 1;
        let (mut positions, mut normals, mut colours, mut indices) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for i in 0..nx {
            let y = SEA + i as f64 * step;
            let z = bed_height(y);
            let colour = if z > 0.3 {
                sand
            } else if z > -0.2 {
                wet_sand
            } else {
                let t = ((-z) / H0).min(1.0) as f32;
                wet_sand.mix(&deep, t)
            };
            for j in 0..nzb {
                let n = hash(i, j) as f32 * 0.06 - 0.03;
                let [r, g, b, a] = linear(colour);
                positions.push([y as f32, z as f32 * vz, -(j as f64 * step) as f32]);
                normals.push([0.0, 1.0, 0.0]);
                colours.push([r + n, g + n, b + n, a]);
            }
        }
        for i in 0..nx - 1 {
            for j in 0..nzb - 1 {
                let a = (i * nzb + j) as u32;
                let b = a + nzb as u32;
                indices.extend([a, b, b + 1, a, b + 1, a + 1]);
            }
        }
        let mut mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, colours)
        .with_inserted_indices(Indices::U32(indices));
        mesh.compute_smooth_normals();
        commands.spawn((
            Name::new("Bed"),
            Mesh3d(meshes.add(mesh)),
            MeshMaterial3d(materials.add(StandardMaterial {
                perceptual_roughness: 0.95,
                reflectance: 0.1,
                double_sided: true,
                cull_mode: None,
                ..default()
            })),
        ));
    }

    // The wall where the model ends, 0.5 m deep
    let wall_height = 1.2;
    commands.spawn((
        Name::new("Wall"),
        Mesh3d(meshes.add(Cuboid::new(
            1.2,
            (H1 + wall_height) as f32 * vz,
            ALONG as f32,
        ))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.42, 0.42, 0.4),
            perceptual_roughness: 0.85,
            ..default()
        })),
        Transform::from_xyz(
            L as f32 + 0.6,
            (wall_height - H1) as f32 * 0.5 * vz,
            -(ALONG as f32) * 0.5,
        ),
    ));

    // The water surface: a grid moved every frame
    let nx = beach.profile().psi.len();
    let mut positions = Vec::with_capacity(nx * nz);
    for i in 0..nx {
        for j in 0..nz {
            positions.push([
                (SEA + i as f64 * SPACING) as f32,
                0.0,
                -(j as f64 * SPACING) as f32,
            ]);
        }
    }
    let mut indices = Vec::with_capacity(6 * (nx - 1) * (nz - 1));
    for i in 0..nx - 1 {
        for j in 0..nz - 1 {
            let a = (i * nz + j) as u32;
            let b = a + nz as u32;
            indices.extend([a, b, b + 1, a, b + 1, a + 1]);
        }
    }
    let n = positions.len();
    let mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0f32, 1.0, 0.0]; n])
    .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, vec![[0.1f32, 0.3, 0.4, 0.9]; n])
    .with_inserted_indices(Indices::U32(indices));
    let water = meshes.add(mesh);
    beach.water = water.clone();
    commands.spawn((
        Name::new("Water"),
        Mesh3d(water),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::WHITE,
            alpha_mode: AlphaMode::Blend,
            perceptual_roughness: 0.12,
            reflectance: 0.55,
            double_sided: true,
            cull_mode: None,
            ..default()
        })),
        NoFrustumCulling,
    ));

    // Floats in the surf zone, carried by the longshore current
    let float_mesh = meshes.add(Sphere::new(0.7).mesh().ico(2).unwrap());
    let float_material = materials.add(StandardMaterial {
        base_color: Color::srgb(1.0, 0.45, 0.1),
        perceptual_roughness: 0.5,
        ..default()
    });
    for i in 0..120 {
        let y = 140.0 + 205.0 * hash(i, 7);
        let x = ALONG * hash(i, 13);
        commands.spawn((
            Float { y, x },
            Mesh3d(float_mesh.clone()),
            MeshMaterial3d(float_material.clone()),
            Transform::from_xyz(y as f32, 0.0, -x as f32),
        ));
    }

    let font = |size: f32| TextFont {
        font_size: FontSize::Px(size),
        ..default()
    };
    let shadow = TextShadow {
        offset: Vec2::splat(1.5),
        color: Color::srgba(0.0, 0.0, 0.0, 0.85),
    };
    commands.spawn((
        Status,
        Text::new(""),
        font(16.0),
        shadow,
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(14.0),
            top: Val::Px(12.0),
            ..default()
        },
    ));
    commands.spawn((
        Text::new(KEYS),
        font(13.0),
        TextColor(Color::srgba(1.0, 1.0, 1.0, 0.85)),
        shadow,
        TextLayout {
            justify: Justify::Right,
            ..default()
        },
        Node {
            position_type: PositionType::Absolute,
            right: Val::Px(14.0),
            bottom: Val::Px(12.0),
            ..default()
        },
    ));
}

fn keys(
    keys: Res<ButtonInput<KeyCode>>,
    views: Res<Views>,
    mut beach: ResMut<Beach>,
    mut cameras: Query<&mut OrbitCamera>,
) {
    if let Ok(mut orbit) = cameras.single_mut() {
        if keys.just_pressed(KeyCode::KeyF) {
            *orbit = views.farm;
        }
        if keys.just_pressed(KeyCode::KeyO) {
            *orbit = views.domain;
        }
    }
    if keys.just_pressed(KeyCode::Digit1) {
        beach.shown = 0;
    }
    if keys.just_pressed(KeyCode::Digit2) {
        beach.shown = 1;
    }
    if keys.just_pressed(KeyCode::Space) {
        beach.paused = !beach.paused;
    }
    if keys.just_pressed(KeyCode::BracketLeft) {
        beach.rate = (beach.rate / 2.0).max(1.0 / 16.0);
    }
    if keys.just_pressed(KeyCode::BracketRight) {
        beach.rate = (beach.rate * 2.0).min(16.0);
    }
}

/// Surface elevation (m, about the still level) and its gradient at
/// cross-shore column `c` (`y`), alongshore `x`, time `t`.
fn surface(p: &Profile, c: usize, x: f64, t: f64) -> (f64, f64, f64) {
    let sigma = TAU / PERIOD;
    let phase = p.kx * x + p.psi[c] - sigma * t;
    let a = p.amplitude[c];
    let skew = (0.6 * a * p.k[c]).min(0.35);
    let bore = (p.breaking[c] / 0.3).min(1.0);
    let z = a * shape(phase, skew, bore) + p.eta[c];
    let eps = 1e-3;
    let dz = a * (shape(phase + eps, skew, bore) - shape(phase - eps, skew, bore)) / (2.0 * eps);
    // ∂φ/∂y = k_y, ∂φ/∂x = k_x
    let ky = (p.k[c] * p.k[c] - p.kx * p.kx).max(0.0).sqrt();
    (z, dz * ky, dz * p.kx)
}

fn animate(time: Res<Time>, mut beach: ResMut<Beach>, mut meshes: ResMut<Assets<Mesh>>) {
    if !beach.paused {
        beach.t += time.delta_secs_f64() * beach.rate;
    }
    let (t, vz) = (beach.t, beach.vz as f64);
    let p = beach.profile();
    let Some(mut mesh) = meshes.get_mut(&beach.water) else {
        return;
    };
    let nz = (ALONG / SPACING).round() as usize + 1;
    let deep = linear(Color::srgba(0.03, 0.22, 0.33, 0.95));
    let shallow = linear(Color::srgba(0.10, 0.52, 0.50, 0.78));
    let foam = linear(Color::srgba(0.97, 0.98, 1.0, 1.0));

    // Heights, normals and colours, column by column in parallel
    let nx = p.psi.len();
    let mut heights = vec![0.0f32; nx * nz];
    let mut normals = vec![[0.0f32; 3]; nx * nz];
    let mut colours = vec![[0.0f32; 4]; nx * nz];
    heights
        .par_chunks_mut(nz)
        .zip(normals.par_chunks_mut(nz))
        .zip(colours.par_chunks_mut(nz))
        .enumerate()
        .for_each(|(c, ((h, n), col))| {
            let y = SEA + c as f64 * SPACING;
            let s = (1.0 - depth(y.max(0.0)) / H0).clamp(0.0, 1.0) as f32;
            let base: [f32; 4] = std::array::from_fn(|i| deep[i] + (shallow[i] - deep[i]) * s);
            let sigma = TAU / PERIOD;
            for j in 0..nz {
                let x = j as f64 * SPACING;
                let (z, dzdy, dzdx) = surface(p, c, x, t);
                h[j] = (z * vz) as f32;
                // Scene z is −x: ∂/∂(scene z) = −∂/∂x
                let normal = Vec3::new(-(dzdy * vz) as f32, 1.0, (dzdx * vz) as f32).normalize();
                n[j] = normal.to_array();
                // White water: the roller on the front of a breaking crest and the
                // foam it leaves behind, broken up by a little noise
                // Saturates where a wave loses 30 % of its energy per period
                let q = (p.breaking[c] / 0.3).min(1.0);
                let mut white = 0.0;
                if q > 0.02 {
                    let phase = (p.kx * x + p.psi[c] - sigma * t).rem_euclid(TAU);
                    let roller = if phase < 0.75 {
                        1.0 - phase / 0.75
                    } else {
                        0.0
                    };
                    let behind = TAU - phase;
                    let trail = if behind < 2.4 {
                        (1.0 - behind / 2.4).powi(2)
                    } else {
                        0.0
                    };
                    let noise = hash(c / 2 + (t * 0.5) as usize, j / 2);
                    white = (q * (1.6 * roller + 1.1 * trail * (0.3 + 0.7 * noise))).min(1.0);
                }
                // Brighter on the crests as they steepen
                let lift =
                    (0.25 * (z - p.eta[c]) / p.amplitude[c].max(1e-3)).clamp(-0.2, 0.3) as f32;
                let w = white as f32;
                col[j] = std::array::from_fn(|i| {
                    let lit = if i < 3 {
                        base[i] * (1.0 + lift)
                    } else {
                        base[i]
                    };
                    lit + (foam[i] - lit) * w
                });
            }
        });

    if let Some(VertexAttributeValues::Float32x3(positions)) =
        mesh.attribute_mut(Mesh::ATTRIBUTE_POSITION)
    {
        for (pos, &h) in positions.iter_mut().zip(&heights) {
            pos[1] = h;
        }
    }
    if let Some(VertexAttributeValues::Float32x3(out)) = mesh.attribute_mut(Mesh::ATTRIBUTE_NORMAL)
    {
        out.copy_from_slice(&normals);
    }
    if let Some(VertexAttributeValues::Float32x4(out)) = mesh.attribute_mut(Mesh::ATTRIBUTE_COLOR) {
        out.copy_from_slice(&colours);
    }
}

fn drift(
    time: Res<Time>,
    beach: Res<Beach>,
    mut floats: Query<(&mut Float, &mut Transform, &mut Visibility)>,
) {
    let p = beach.profile();
    let dt = if beach.paused {
        0.0
    } else {
        time.delta_secs_f64() * beach.rate
    };
    let shown = p.case == Case::Oblique;
    for (mut f, mut transform, mut visibility) in &mut floats {
        *visibility = if shown {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
        if !shown {
            continue;
        }
        let c = p.column(f.y);
        f.x = (f.x + p.current[c] * dt).rem_euclid(ALONG);
        let (z, _, _) = surface(p, c, f.x, beach.t);
        *transform =
            Transform::from_xyz(f.y as f32, (z * beach.vz as f64) as f32 + 0.3, -f.x as f32);
    }
}

fn status(beach: Res<Beach>, mut text: Query<&mut Text, With<Status>>) {
    let Ok(mut text) = text.single_mut() else {
        return;
    };
    let p = beach.profile();
    let state = if beach.paused { " | paused" } else { "" };
    let content = format!(
        "Swell breaking on a 1:100 beach: 8 s, H_rms {H_RMS} m offshore, {}\n{}\n\
         t = {:.0} s | {}x real time{state} | heights x{}\n\
         The model ends at a wall in 0.5 m of water (grey)",
        p.case.name(),
        p.facts,
        beach.t,
        beach.rate,
        beach.vz,
    );
    if text.0 != content {
        text.0 = content;
    }
}

fn capture(
    mut commands: Commands,
    mut capture: ResMut<Capture>,
    beach: Res<Beach>,
    mut exit: MessageWriter<AppExit>,
) {
    let Some(path) = capture.path.clone() else {
        return;
    };
    let at = capture.at;
    match capture.frames.as_mut() {
        None if beach.t >= at => capture.frames = Some(0),
        None => {}
        Some(n) => {
            *n += 1;
            if *n == 3 {
                commands
                    .spawn(Screenshot::primary_window())
                    .observe(save_to_disk(path));
            } else if *n == 30 {
                exit.write(AppExit::Success);
            }
        }
    }
}

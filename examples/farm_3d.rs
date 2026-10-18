//! A fish farm in a stratified tidal channel, in 3D and in 2D (TODO F.1).
//!
//! A straight channel, 6 km long (periodic along it) and 600 m wide, 40 m
//! deep, carries a tidal current of ≈ 0.5 m/s: a depth-uniform M2 body force
//! in the 2D module stands in for the tidal surface slope, ramped up over
//! the first hour. A row of two circular cages (50 m across, nets 20 m deep)
//! stands across the flow in the middle of the channel. The water is weakly
//! stratified (2 °C warmer over the top 10 m), mixed by GLS k-ε, against a
//! log-layer bottom drag.
//!
//! Four runs from rest: 3D with and without the cages
//! (`Hydrostatic3D::with_cage_drag`), and the 2D model with and without them
//! (`SWEPhysics2DBuilder::with_cage_drag`, Chézy friction with the 3D
//! drag's `C_d` bound). At the end it reports
//! - the depth-mean current behind the farm and its deficit in 2D and 3D;
//! - the 3D profile behind a cage: the deficit sits in the layers the net
//!   reaches, and part of the flow goes underneath;
//! - the drag force on the cages.
//!
//! `layout` places the cages: `across` (the default, a row across the flow,
//! 120 m apart) or `along` (one behind the other along the channel, 120 m
//! apart, so the tide carries water from each to the other).
//!
//! With `particles=N`, the 3D run with cages also releases N particles of
//! each kind in each cage at `release` hours (default 2, in the running
//! tide), and again every `release_every` hours (default 1) for `releases`
//! batches (default 1), and tracks them online (`ParticleTracker3D`, TODO
//! F.2) between the solver's states every `particle_seconds` (default 60),
//! with a horizontal walk `kh` (m²/s, default 0.1; a list such as
//! `kh=0.01,0.1,1` tracks the larvae in each, for the connectivity's
//! sensitivity, the other kinds in the first; `smag` is Smagorinsky's `K`
//! from each layer's strain with the model's `cs` and `Pr_t` = 1, without
//! the model's numerical background `nu`, and `smag+0.01` adds a constant
//! to it: `particles::diffusivity`) and Visser's vertical
//! walk in the model's GLS diffusivity:
//! - lice larvae, neutrally buoyant, released over the top 5 m, swimming
//!   by `lice` (`SalmonLice`: `ladim`, the default, the operational IMR
//!   parameters; `johnsen`, Johnsen et al. 2014; `passive`): up towards the
//!   light, a clear sky at Mausund (63.87° N, 8.67° E) from `start` (UTC),
//!   and down out of water fresher than their threshold;
//! - faeces, sinking at 3 cm/s, released over the net's depth;
//! - feed pellets, sinking at 10 cm/s, released over the top 2 m.
//!
//! Faeces and feed settle on the bed; the report gives the larvae's depths,
//! spread and swimming, and where the rest landed.
//!
//! Pen-to-pen connectivity of the larvae (`particles::connectivity`): the
//! share of each cage's larvae that come into contact with each cage (inside
//! its footprint, in the top `contact_depth` m, default the net's 20 m; its
//! own cage only once it has been a cage radius beyond it, so the walk
//! stepping out and back in at the edge is no return), per release batch
//! with the tide's phase at release, and over all batches. Exposure per pair is the time in
//! contact: per released larva, and its 10/50/90 % among the larvae in
//! contact, with the age at first contact. `seeds=R` (default 1) tracks the
//! larvae R times with independent random walks (the first with the other
//! kinds, the rest alone) and gives each share's spread across the seeds
//! beside one seed's binomial error. `connectivity=<file.csv>` writes every
//! seed's matrix per batch, for comparing runs (layouts, `kh`, meshes).
//! `runs=cages` runs only the 3D run with cages.
//!
//! `stokes=H_s,T_p,direction` (m, s, degrees counter-clockwise from along the
//! channel; default off) adds the Stokes drift of a uniform sea to the
//! particles (`WithStokesDrift3D`, TODO F.4): a JONSWAP spectrum (γ = 3.3,
//! cos⁴ spreading) on the farm's mesh, each particle drifting at its own depth.
//! The circulation does not feel the waves.
//!
//! `wind` (Pa, default 0) blows along the channel: through the columns'
//! surface stress in 3D, as a body force `τ/ρ₀` in 2D. `waves` (m, default
//! off) adds breaking waves to GLS (`GlsMixing::with_wave_breaking`, Craig &
//! Banner) with that surface roughness, a wave height's (Terray et al.:
//! `z₀ ≈ 0.6 H_s`); `closure` picks the GLS model (`k-epsilon`, the
//! default, `k-omega` or `generic`; with waves k-ω or the generic model,
//! whose wave-affected layer metre layers resolve), or `none`: a constant
//! viscosity (1e-2 m²/s) and no diffusivity of the tracers.
//!
//! Each 3D run reports its mixing, the growth of the reference potential
//! energy (`PotentialEnergy3D`), every half hour with the range of T and
//! the drift of the volume and heat inventories. With `closure=none` all of
//! it is the advection's. For measuring that: `limiter` (`none`, the
//! default, or `kuzmin`: the horizontal Kuzmin tracer limiter), `tvadv`
//! (the tracers' vertical advection: `limited-akima`, the default, `akima`,
//! `tvd` or `upwind`) and `runs=open` (only the 3D run without cages;
//! `runs=cages`, only the one with).
//!
//! ```bash
//! cargo run --release --no-default-features --features parallel,simd \
//!     --example farm_3d -- [hours=3] [order=2] [levels=16] [dx=60] [nu=1] [cs=0.2] \
//!     [layout=across] [particles=0] [release=2] [releases=1] [release_every=1] \
//!     [particle_seconds=60] [kh=0.1] [seeds=1] [contact_depth=20] \
//!     [connectivity=<file.csv>] \
//!     [lice=ladim] [start=2025-06-15T00:00:00Z] [stokes=H_s,T_p,direction] \
//!     [wind=0] [waves=0] [closure=k-epsilon] \
//!     [limiter=none] [tvadv=limited-akima] [runs=all] \
//!     [snapshot=<file.dgsnap>] [snapshot_minutes=10]
//! ```
//!
//! `snapshot=` writes the 3D run with cages to a snapshot file
//! (`io::SnapshotWriter::create_3d`: η, ū, v̄ and u, v, T, S on every level
//! every `snapshot_minutes`, with the mesh, σ-grid, clock, cages and a
//! section through the first cage), and with `particles=` the particles
//! beside it (`<file>.dgpart`, `io::ParticleFileWriter`), which the viewer
//! replays: `cd viz && cargo run --release -- --replay <file.dgsnap>`.

use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::Arc;
use std::time::Instant;

use dg_rs::boundary::Reflective2D;
use dg_rs::equations::ShallowWater2D;
use dg_rs::io::{ParticleFileWriter, ParticleFrame, SnapshotWriter, status_code};
use dg_rs::mesh::data::Bathymetry2D;
use dg_rs::mesh::{Mesh2D, PointLocator2D};
use dg_rs::operators::{DGOperators2D, GeometricFactors2D};
use dg_rs::particles::{
    ClearSkyLight, ConnectivityEnsemble, ConnectivityRecorder, Contact, ContactZone,
    DiffusivityInTime, HorizontalDiffusivity, HorizontalDiffusivityField, Particle3D,
    ParticleStatus, ParticleTracker3D, ParticleVelocity3D, SalmonLice, Solution3DVelocity,
    StokesDrift, WithHorizontalDiffusivity3D, WithStokesDrift3D,
};
use dg_rs::physics::cage_drag::{for_each_caged_node, layer_coefficient};
use dg_rs::physics::{
    BottomDrag3D, ConstantMixing, EquationOfState, Forcing, GlsMixing, Hydrostatic3D, LinearEOS,
    PhysicsBuilder, SWEPhysics2D, VerticalMixing,
};
use dg_rs::simulation::{Simulation, Simulation3D};
use dg_rs::solver::rhs::VerticalAdvection;
use dg_rs::solver::state::Solution3D;
use dg_rs::solver::{
    PotentialEnergy3D, Probe2D, SWESolution2D, SWEState2D, TracerBounds, TracerLimiter3DConfig,
};
use dg_rs::source::{
    CageDrag2D, ChezyFriction2D, CoriolisSource2D, HorizontalViscosity2D, NetCage, SourceContext2D,
    SourceTerm2D,
};
use dg_rs::time::{ModeSplitIntegrator, ModelClock, SSPRK3};
use dg_rs::types::ElementIndex;
use dg_rs::vertical::{SigmaGrid, UniformStretching};
use dg_rs::waves::{SpectralGrid, StokesDriftField, WaveModel2D};

const G: f64 = 9.81;
const RHO0: f64 = 1025.0;
const LX: f64 = 6_000.0;
const LY: f64 = 600.0;
const DEPTH: f64 = 40.0;
const NET_DEPTH: f64 = 20.0;
const RADIUS: f64 = 25.0;
/// Cage centres of each layout: a row across the flow in the middle of the
/// channel, or one behind the other along it
const ACROSS: [[f64; 2]; 2] = [[3_000.0, 240.0], [3_000.0, 360.0]];
const ALONG: [[f64; 2]; 2] = [[2_940.0, 300.0], [3_060.0, 300.0]];
const M2: f64 = 2.0 * PI / 44_714.16;
/// Where the sun is: Mausund, off Frøya (longitude, latitude)
const SITE: [f64; 2] = [8.67, 63.87];
/// Tidal current amplitude without drag (m/s)
const U_TIDE: f64 = 0.5;
const RAMP: f64 = 3_600.0;
/// Bed roughness of the log-layer drag (m) and its lower `C_d` bound
const Z0: f64 = 0.005;
const CD_MIN: f64 = 2.5e-3;

/// A uniform wind stress along the channel as a body force in 2D,
/// `∂(hu)/∂t = τ/ρ₀`.
struct WindForce(f64);

impl SourceTerm2D for WindForce {
    fn evaluate(&self, _ctx: &SourceContext2D) -> SWEState2D {
        SWEState2D::new(0.0, self.0 / RHO0, 0.0)
    }

    fn name(&self) -> &'static str {
        "wind_force"
    }
}

/// The tidal surface slope as a depth-uniform body force along the
/// channel, `∂(hu)/∂t = h·a(t)`, `a = U ω r(t) cos ωt` with a smooth ramp
/// `r` over the first hour.
struct TidalForce;

impl TidalForce {
    fn acceleration(t: f64) -> f64 {
        let ramp = if t < RAMP {
            0.5 * (1.0 - (PI * t / RAMP).cos())
        } else {
            1.0
        };
        U_TIDE * M2 * ramp * (M2 * t).cos()
    }
}

impl SourceTerm2D for TidalForce {
    fn evaluate(&self, ctx: &SourceContext2D) -> SWEState2D {
        SWEState2D::new(0.0, ctx.state.h * Self::acceleration(ctx.time), 0.0)
    }

    fn name(&self) -> &'static str {
        "tidal_force"
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: HashMap<String, String> = std::env::args()
        .skip(1)
        .filter_map(|a| a.split_once('=').map(|(k, v)| (k.into(), v.into())))
        .collect();
    let get = |key: &str, default: f64| args.get(key).map_or(Ok(default), |v| v.parse());
    let hours: f64 = get("hours", 3.0)?;
    let order = get("order", 2.0)? as usize;
    let n_levels = get("levels", 16.0)? as usize;
    let dx: f64 = get("dx", 60.0)?;
    let nu: f64 = get("nu", 1.0)?;
    let cs: f64 = get("cs", 0.2)?;
    let centres: Vec<[f64; 2]> = match args.get("layout").map_or("across", String::as_str) {
        "across" => ACROSS.to_vec(),
        "along" => ALONG.to_vec(),
        other => return Err(format!("layout={other}: across or along").into()),
    };
    let particles_per_kind = get("particles", 0.0)? as usize;
    let release_time = get("release", 2.0)? * 3600.0;
    let release_every = get("release_every", 1.0)? * 3600.0;
    let batches: Vec<f64> = (0..get("releases", 1.0)? as usize)
        .map(|b| release_time + b as f64 * release_every)
        .collect();
    let particle_seconds: f64 = get("particle_seconds", 60.0)?;
    // One or more horizontal diffusivities: the larvae are tracked in each
    // (the other kinds and the snapshot's larvae in the first)
    let kh: Vec<Mixing> = match args.get("kh") {
        Some(v) => v.split(',').map(Mixing::parse).collect::<Result<_, _>>()?,
        None => vec![Mixing::Constant(0.1)],
    };
    let seeds = (get("seeds", 1.0)? as usize).max(1);
    let contact_depth: f64 = get("contact_depth", NET_DEPTH)?;
    let connectivity_path = args.get("connectivity").cloned();
    let clock = ModelClock::parse(
        args.get("start")
            .map_or("2025-06-15T00:00:00Z", String::as_str),
    )?;
    let light = ClearSkyLight::new(clock, SITE[0], SITE[1]);
    let snapshot_path = args.get("snapshot").cloned();
    let snapshot_seconds = get("snapshot_minutes", 10.0)? * 60.0;
    let lice = match args.get("lice").map_or("ladim", String::as_str) {
        "ladim" => Some(SalmonLice::ladim(light)),
        "johnsen" => Some(SalmonLice::johnsen_2014(light)),
        "passive" => None,
        other => return Err(format!("lice={other}: ladim, johnsen or passive").into()),
    };
    // The tracers' numerics, and which runs: for measuring their mixing
    let tracer_limiter = match args.get("limiter").map_or("none", String::as_str) {
        "none" => TracerLimiter3DConfig::none(),
        "kuzmin" => TracerLimiter3DConfig::horizontal_kuzmin(TracerBounds::default(), 1.0),
        other => return Err(format!("limiter={other}: none or kuzmin").into()),
    };
    let tracer_vertical = match args.get("tvadv").map_or("limited-akima", String::as_str) {
        "limited-akima" => VerticalAdvection::LimitedAkima,
        "akima" => VerticalAdvection::Akima,
        "tvd" => VerticalAdvection::Tvd,
        "upwind" => VerticalAdvection::Upwind,
        other => return Err(format!("tvadv={other}: limited-akima, akima, tvd or upwind").into()),
    };
    let runs = args.get("runs").map_or("all", String::as_str);
    if !["all", "open", "cages"].contains(&runs) {
        return Err(format!("runs={runs}: all, open or cages").into());
    }
    let sea: Option<[f64; 3]> = match args.get("stokes") {
        Some(v) => {
            let parts: Vec<f64> = v.split(',').map(str::parse).collect::<Result<_, _>>()?;
            let [hs, tp, direction] = parts[..] else {
                return Err(format!("stokes={v}: H_s,T_p,direction").into());
            };
            Some([hs, tp, direction])
        }
        None => None,
    };
    let wind: f64 = get("wind", 0.0)?;
    let waves: f64 = get("waves", 0.0)?;
    // `none`: no diffusivity of the tracers (a constant viscosity), so that
    // any change of RPE is the advection's
    let closure_name = args.get("closure").map_or("k-epsilon", String::as_str);
    let no_diffusion = closure_name == "none";
    let closure = match closure_name {
        "k-epsilon" | "none" => GlsMixing::k_epsilon(),
        "k-omega" => GlsMixing::k_omega(),
        "generic" => GlsMixing::generic(),
        other => {
            return Err(format!("closure={other}: k-epsilon, k-omega, generic or none").into());
        }
    };
    let mixing = if waves > 0.0 {
        closure
            .with_roughness(waves, Z0)
            .with_wave_breaking(GlsMixing::DEFAULT_WAVE_BREAKING)
    } else {
        closure.with_roughness(GlsMixing::DEFAULT_ROUGHNESS, Z0)
    };
    let t_end = hours * 3600.0;

    let (nx, ny) = ((LX / dx).round() as usize, (LY / dx).round() as usize);
    let mesh = Arc::new(Mesh2D::channel_periodic_x(0.0, LX, 0.0, LY, nx, ny));
    let ops = Arc::new(DGOperators2D::new(order));
    let geom = Arc::new(GeometricFactors2D::compute(&mesh, &ops));
    let bathymetry = Arc::new(Bathymetry2D::constant(mesh.n_elements, ops.n_nodes, -DEPTH));
    let sigma = Arc::new(SigmaGrid::new(n_levels, UniformStretching));
    // The Stokes drift of a uniform sea over the farm, for the particles
    let stokes = sea.map(|[hs, tp, direction]| {
        let model = WaveModel2D::new(
            mesh.clone(),
            ops.clone(),
            geom.clone(),
            &bathymetry,
            SpectralGrid::new(0.04, 1.0, 30, 24),
            G,
        );
        let e = model.grid.jonswap(hs, tp, 3.3, direction.to_radians(), 4.0);
        println!(
            "Sea: H_s {hs} m, T_p {tp} s towards {direction}°; Stokes drift for the particles"
        );
        StokesDriftField::new(&model, &model.uniform_state(&e))
    });
    let cages: Vec<NetCage> = centres
        .iter()
        .map(|&c| NetCage::circular(c, RADIUS, NET_DEPTH, 0.25))
        .collect();
    println!(
        "Channel {:.0} × {:.0} m, {} quads of {dx:.0} m (P{order}), {n_levels} σ-levels, {DEPTH} m deep",
        LX, LY, mesh.n_elements
    );
    println!(
        "GLS {:?}, {}; wind {wind} Pa along the channel",
        mixing.parameters(),
        if waves > 0.0 {
            format!("breaking waves, z₀ = {waves} m")
        } else {
            "log-layer surface".into()
        }
    );
    println!(
        "Cages: {} × R {RADIUS} m, nets {NET_DEPTH} m deep, C_d·a = {:.4} /m",
        cages.len(),
        cages[0].drag_per_length
    );

    // The 2D module: in 3D it owns the depth mean (tidal force, Coriolis,
    // viscosity of ⟨u⟩); friction and cages are the 3D model's
    let swe = |with_friction: bool, with_cages: bool| -> SWEPhysics2D<Reflective2D> {
        let builder = PhysicsBuilder::swe_2d(
            mesh.clone(),
            ops.clone(),
            geom.clone(),
            ShallowWater2D::new(G),
            Reflective2D::new(),
        )
        .with_bathymetry(bathymetry.clone())
        .with_source(TidalForce)
        .with_source(CoriolisSource2D::f_plane(1.2e-4))
        .with_viscosity(HorizontalViscosity2D::constant(nu));
        // The standalone 2D runs (with friction) take the wind as a body
        // force; in 3D it reaches the depth mean through the columns
        let builder = if with_friction && wind != 0.0 {
            builder.with_source(WindForce(wind))
        } else {
            builder
        };
        let builder = if with_friction {
            // The 3D log-layer drag of a well-mixed column: C_d at the
            // bottom-layer centre, bounded below as in BottomDrag3D
            let z_b = 0.5 * DEPTH / n_levels as f64;
            let cd = BottomDrag3D::log_layer(Z0)
                .drag_coefficient(z_b)
                .max(CD_MIN);
            builder.with_implicit_friction(ChezyFriction2D::new(cd))
        } else {
            builder
        };
        if with_cages {
            builder.with_cage_drag(CageDrag2D::new(&mesh, &ops, &cages))
        } else {
            builder
        }
        .build()
    };

    let eos = LinearEOS::default();
    let physics_3d = |with_cages: bool| {
        let closure: Box<dyn VerticalMixing + Send + Sync> = if no_diffusion {
            Box::new(ConstantMixing::new(1e-2, 0.0))
        } else {
            Box::new(mixing.clone())
        };
        let physics = Hydrostatic3D::new(
            mesh.clone(),
            ops.clone(),
            geom.clone(),
            sigma.clone(),
            bathymetry.clone(),
            Arc::new(CoriolisSource2D::f_plane(1.2e-4)),
            eos,
            closure,
            swe(false, false),
            Forcing {
                surface_stress: [wind, 0.0],
                bottom_stress: [0.0, 0.0],
                surface_heat_flux: 0.0,
            },
            G,
            RHO0,
        )
        .with_bottom_drag(BottomDrag3D::log_layer(Z0))
        .with_horizontal_viscosity(nu)
        .with_smagorinsky_viscosity(cs)
        .with_vertical_advection(tracer_vertical)
        .with_tracer_limiter(tracer_limiter.clone());
        if with_cages {
            physics.with_cage_drag(CageDrag2D::new(&mesh, &ops, &cages))
        } else {
            physics
        }
    };

    let run_3d = |with_cages: bool| -> Result<Solution3D, String> {
        let physics = physics_3d(with_cages);
        let mut state = Solution3D::new(mesh.n_elements, ops.n_nodes, n_levels);
        state.salt.fill(eos.s0);
        // 2 °C warmer over the top 10 m, a smooth step
        for column in state.temp.chunks_exact_mut(n_levels) {
            for (l, t) in column.iter_mut().enumerate() {
                let z = DEPTH * sigma.sigma_rho()[l];
                *t = eos.t0 + 1.0 + (0.5 * (z + 10.0)).tanh();
            }
        }
        physics.update_density(&mut state);
        let mut energy = PotentialEnergy3D::new(&ops, &geom, &bathymetry, G, RHO0);
        let reference_0 = energy.compute(&state, &sigma).reference;
        let label = if with_cages { "3D cages" } else { "3D open" };
        let start = Instant::now();
        let mut tracking = (with_cages && particles_per_kind > 0).then(|| {
            FarmParticles::new(
                &mesh,
                &ops,
                particles_per_kind,
                &centres,
                batches.clone(),
                kh[0].constant(),
                lice,
            )
            .with_stokes_drift(stokes.as_ref())
            .with_connectivity(&kh, seeds, contact_depth, &geom, cs)
        });
        let mut sim = Simulation3D::new(physics, ModeSplitIntegrator::new()).with_cfl(0.5);
        // The run with cages to the snapshot file, if asked
        let mut snapshot = match snapshot_path.as_ref().filter(|_| with_cages) {
            Some(path) => {
                let farm = [
                    0.5 * (centres[0][0] + centres[1][0]),
                    0.5 * (centres[0][1] + centres[1][1]),
                ];
                let cage_lines: Vec<String> = cages
                    .iter()
                    .zip(&centres)
                    .map(|(c, [x, y])| {
                        format!("{x},{y},{RADIUS},{NET_DEPTH},{}", c.drag_per_length)
                    })
                    .collect();
                let poi = format!("{},{}", farm[0], farm[1]);
                // Along the flow through the first cage
                let (x0, y0) = (centres[0][0], centres[0][1]);
                let section = format!("{},{y0},{},{y0}", x0 - 600.0, x0 + 600.0);
                let periodic = format!("{LX},inf");
                let mut metadata = vec![
                    (
                        "title",
                        "Farm channel, 3D: M2 0.5 m/s, 2 C thermocline at 10 m",
                    ),
                    ("source", "farm_3d"),
                    ("point_of_interest", poi.as_str()),
                    ("section", section.as_str()),
                    ("periodic", periodic.as_str()),
                ];
                metadata.extend(cage_lines.iter().map(|c| ("cage", c.as_str())));
                let writer = SnapshotWriter::create_3d(
                    path,
                    &mesh,
                    &ops,
                    &bathymetry.data,
                    &sigma,
                    Some(&clock),
                    &metadata,
                )
                .map_err(|e| format!("{path}: {e}"))?;
                // The particles beside it, at the same times
                let particles = match &tracking {
                    Some(_) => {
                        let path = std::path::Path::new(path).with_extension("dgpart");
                        let kinds: Vec<&str> = KINDS.iter().map(|k| k.0).collect();
                        Some(
                            ParticleFileWriter::create(&path, &kinds, &[("source", "farm_3d")])
                                .map_err(|e| format!("{}: {e}", path.display()))?,
                        )
                    }
                    None => None,
                };
                let mut writers = (writer, particles);
                write_frame(&mut writers, &state, 0.0, tracking.as_ref(), &bathymetry)?;
                println!("{label:>9}: snapshots every {snapshot_seconds} s to {path}");
                Some(writers)
            }
            None => None,
        };
        let (mut next_snapshot, mut last_snapshot) = (snapshot_seconds, 0.0);
        let mut snapshot_error = None;
        let interval = if tracking.is_some() {
            particle_seconds
        } else {
            1800.0
        };
        let interval = if snapshot.is_some() {
            interval.min(snapshot_seconds)
        } else {
            interval
        };
        sim = sim.with_callback_interval(interval);
        // The mixing over each half hour, the range of T, and the volume and
        // heat inventories
        let inventory = |s: &Solution3D| {
            let (mut volume, mut heat) = (0.0, 0.0);
            for idx in 0..s.eta.data.len() {
                let area = ops.weights[idx % ops.n_nodes]
                    * geom.jacobian(idx / ops.n_nodes, idx % ops.n_nodes);
                let depth = s.eta.data[idx] - bathymetry.data[idx];
                volume += area * depth;
                for (l, ds) in sigma.d_sigma().iter().enumerate() {
                    heat += area * depth * ds * s.temp[idx * n_levels + l];
                }
            }
            (volume, heat)
        };
        let (volume_0, heat_0) = inventory(&state);
        let (mut density, mut last) = (state.clone(), (0.0, reference_0));
        let result = sim.run_with_callback(&mut state, 0.0, t_end, |s, t| {
            if let Some(tracking) = tracking.as_mut() {
                tracking.advance(s, t, &sigma, &bathymetry);
            }
            // Callbacks land at the first step past each interval: a frame at
            // the first callback past each snapshot time
            if let Some(writers) = snapshot.as_mut()
                && t >= next_snapshot - 1e-6
            {
                if let Err(e) = write_frame(writers, s, t, tracking.as_ref(), &bathymetry) {
                    snapshot_error.get_or_insert(e);
                }
                last_snapshot = t;
                while next_snapshot <= t + 1e-6 {
                    next_snapshot += snapshot_seconds;
                }
            }
            if t - last.0 >= 1800.0 - 1e-6 {
                density.clone_from(s);
                eos.update_density(&mut density);
                let reference = energy.compute(&density, &sigma).reference;
                let mixing = (reference - last.1) / (t - last.0) / (LX * LY);
                let (lo, hi) = s
                    .temp
                    .iter()
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &t| {
                        (lo.min(t), hi.max(t))
                    });
                let (volume, heat) = inventory(s);
                println!(
                    "{label:>9}: {:4.1} h, mixing {mixing:+.3e} W/m², T in [{lo:.6}, {hi:.6}] °C, volume {:+.2e}, heat {:+.2e} (relative)",
                    t / 3600.0,
                    volume / volume_0 - 1.0,
                    heat / heat_0 - 1.0
                );
                last = (t, reference);
            }
        });
        // The end of the run, unless a frame already holds it (the particles
        // moved on to it first)
        if let Some(writers) = snapshot.as_mut()
            && result.final_time > last_snapshot + 1e-6
        {
            if let Some(tracking) = tracking.as_mut() {
                tracking.advance(&state, result.final_time, &sigma, &bathymetry);
            }
            if let Err(e) = write_frame(
                writers,
                &state,
                result.final_time,
                tracking.as_ref(),
                &bathymetry,
            ) {
                snapshot_error.get_or_insert(e);
            }
        }
        if let Some(e) = snapshot_error {
            return Err(format!("{label}: the snapshot file: {e}"));
        }
        if let Some(tracking) = &tracking {
            tracking.report(&state, &sigma, &bathymetry);
            tracking
                .report_connectivity(connectivity_path.as_deref())
                .map_err(|e| format!("{label}: the connectivity file: {e}"))?;
        }
        // Mixing: the growth of the reference potential energy, GLS's and
        // the advection's spurious mixing together
        eos.update_density(&mut state);
        let mixing = (energy.compute(&state, &sigma).reference - reference_0) / t_end / (LX * LY);
        println!(
            "{label:>9}: {:5} steps, dt {:.1}–{:.1} s, {:6.1} s wall, mixing {mixing:.3e} W/m²",
            result.n_steps,
            result.dt_min,
            result.dt_max,
            start.elapsed().as_secs_f64()
        );
        if !result.success {
            return Err(format!("{label}: {:?}", result.error));
        }
        Ok(state)
    };

    let run_2d = |with_cages: bool| -> Result<SWESolution2D, String> {
        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        q.data[0].fill(DEPTH);
        let label = if with_cages { "2D cages" } else { "2D open" };
        let start = Instant::now();
        let result = Simulation::new(swe(true, with_cages), SSPRK3)
            .with_cfl(0.5)
            .run(&mut q, 0.0, t_end);
        println!(
            "{label:>9}: {:5} steps, dt {:.2}–{:.2} s, {:6.1} s wall",
            result.n_steps,
            result.dt_min,
            result.dt_max,
            start.elapsed().as_secs_f64()
        );
        if !result.success {
            return Err(format!("{label}: {:?}", result.error));
        }
        Ok(q)
    };

    println!("\nM2 body force from rest, {hours} h:");
    match runs {
        "open" => {
            run_3d(false)?;
            return Ok(());
        }
        "cages" => {
            run_3d(true)?;
            return Ok(());
        }
        _ => {}
    }
    let caged_3d = run_3d(true)?;
    let open_3d = run_3d(false)?;
    let caged_2d = run_2d(true)?;
    let open_2d = run_2d(false)?;

    // The flow direction at the end: report the lee side
    let locator = PointLocator2D::new(&mesh);
    let probe = |p: [f64; 2]| Probe2D::at(&locator, &ops, p).expect("the point is in the channel");
    let mean_3d = |s: &Solution3D, p: &Probe2D| p.evaluate_field(&s.ubar.data);
    let mean_2d = |q: &SWESolution2D, p: &Probe2D| {
        let sample = p.sample_swe(q, Some(&bathymetry), 1e-3);
        sample.u
    };
    let farm_u = mean_3d(&open_3d, &probe(centres[0]));
    let lee = farm_u.signum();
    println!(
        "\nUndisturbed current at the farm: 3D {:+.3} m/s, 2D {:+.3} m/s (the lee is {} of the farm)",
        farm_u,
        mean_2d(&open_2d, &probe(centres[0])),
        if lee > 0.0 { "east" } else { "west" }
    );

    println!("\nDepth-mean current behind cage 0 (deficit against the run without cages):");
    println!("  behind    3D (m/s)   deficit    2D (m/s)   deficit");
    for distance in [-150.0, 0.0, 50.0, 100.0, 200.0, 400.0, 800.0] {
        let p = probe([centres[0][0] + lee * distance, centres[0][1]]);
        let (u3, u3o) = (mean_3d(&caged_3d, &p), mean_3d(&open_3d, &p));
        let (u2, u2o) = (mean_2d(&caged_2d, &p), mean_2d(&open_2d, &p));
        println!(
            "  {distance:5.0} m  {u3:+9.4}   {:6.2} %   {u2:+9.4}   {:6.2} %",
            100.0 * (1.0 - u3 / u3o),
            100.0 * (1.0 - u2 / u2o)
        );
    }

    let column = |s: &Solution3D, p: &Probe2D| -> Vec<f64> {
        let k = p.element().as_usize();
        (0..n_levels)
            .map(|l| {
                let values: Vec<f64> = (0..ops.n_nodes)
                    .map(|i| s.u[(k * ops.n_nodes + i) * n_levels + l])
                    .collect();
                p.evaluate(&values)
            })
            .collect()
    };
    for distance in [0.0, 100.0] {
        let p = probe([centres[0][0] + lee * distance, centres[0][1]]);
        let (caged, open) = (column(&caged_3d, &p), column(&open_3d, &p));
        println!("\n3D profile {distance:.0} m behind cage 0 (z of the layer centre):");
        println!("      z (m)   cages (m/s)   open (m/s)   deficit");
        for l in (0..n_levels).rev() {
            let z = DEPTH * sigma.sigma_rho()[l];
            let marker = if -z < NET_DEPTH { "  net" } else { "" };
            println!(
                "  {z:9.2}   {:+11.4}   {:+10.4}   {:6.2} %{marker}",
                caged[l],
                open[l],
                100.0 * (1.0 - caged[l] / open[l])
            );
        }
    }

    // Drag force on the cages, F = ρ₀ ∫ Σ_l H_l λ_l u_l dA (3D) and
    // ρ₀ ∫ Λ hu dA (2D), along the channel
    let cage_drag = CageDrag2D::new(&mesh, &ops, &cages);
    let nn = ops.n_nodes;
    let mut force_3d = 0.0;
    for k in 0..mesh.n_elements {
        for_each_caged_node(&cage_drag, k, nn, |i, entries| {
            let idx = k * nn + i;
            let depth = caged_3d.eta.data[idx] + DEPTH;
            let mut drag = 0.0;
            for l in 0..n_levels {
                let (u, v) = (
                    caged_3d.u[idx * n_levels + l],
                    caged_3d.v[idx * n_levels + l],
                );
                let rate = layer_coefficient(entries, &sigma, l, depth) * u.hypot(v);
                drag += sigma.d_sigma()[l] * depth * rate * u;
            }
            force_3d += RHO0 * geom.mass[idx] * drag;
        });
    }
    let mut force_2d = 0.0;
    for k in ElementIndex::iter(mesh.n_elements) {
        for_each_caged_node(&cage_drag, k.as_usize(), nn, |i, entries| {
            let s = caged_2d.get_state(k, i);
            let speed = s.hu.hypot(s.hv) / s.h;
            let rate = CageDrag2D::damping_rate(entries, s.h, speed);
            force_2d += RHO0 * geom.mass[k.as_usize() * nn + i] * rate * s.hu;
        });
    }
    println!(
        "\nDrag on the farm along the channel: 3D {:.1} kN, 2D {:.1} kN",
        force_3d / 1e3,
        force_2d / 1e3
    );
    Ok(())
}

/// What a particle kind is: name, own vertical speed (m/s, positive up),
/// and the depth range it is released over (m below the surface).
const KINDS: [(&str, f64, [f64; 2]); 3] = [
    ("larvae", 0.0, [0.0, 5.0]),
    ("faeces", -0.03, [0.0, NET_DEPTH]),
    ("feed", -0.10, [0.0, 2.0]),
];

/// Index of the larvae in [`KINDS`].
const LARVAE: usize = 0;

/// The particles' horizontal walk: a constant `K` (m²/s), or Smagorinsky's
/// from each layer's strain (the model's `cs`, `Pr_t` = 1) plus a constant.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Mixing {
    Constant(f64),
    Smagorinsky(f64),
}

impl Mixing {
    /// `0.1`, `smag` or `smag+0.01`.
    fn parse(s: &str) -> Result<Self, String> {
        let error = || format!("kh={s}: a diffusivity (m²/s), smag or smag+K");
        match s.strip_prefix("smag") {
            Some("") => Ok(Self::Smagorinsky(0.0)),
            Some(rest) => rest
                .strip_prefix('+')
                .and_then(|k| k.parse().ok())
                .map(Self::Smagorinsky)
                .ok_or_else(error),
            None => s.parse().map(Self::Constant).map_err(|_| error()),
        }
    }

    /// The tracker's constant part (m²/s).
    fn constant(self) -> f64 {
        match self {
            Self::Constant(k) | Self::Smagorinsky(k) => k,
        }
    }

    fn is_smagorinsky(self) -> bool {
        matches!(self, Self::Smagorinsky(_))
    }
}

impl std::fmt::Display for Mixing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let label = match *self {
            Self::Constant(k) => format!("{k}"),
            Self::Smagorinsky(0.0) => "smag".into(),
            Self::Smagorinsky(k) => format!("smag+{k}"),
        };
        f.pad(&label)
    }
}

/// `n` steps of `particles` through `field` from `t`, by `lice` if given.
fn step_kind(
    tracker: &ParticleTracker3D,
    particles: &mut [Particle3D],
    field: &impl ParticleVelocity3D,
    lice: Option<&SalmonLice<ClearSkyLight>>,
    t: f64,
    dt: f64,
) {
    match lice {
        Some(lice) => tracker.step_with(particles, field, lice, t, dt),
        None => tracker.step(particles, field, t, dt),
    }
}

/// A snapshot frame of `state` at `t`, and the particles' frame beside it.
fn write_frame(
    (fields, particles): &mut (SnapshotWriter, Option<ParticleFileWriter>),
    state: &Solution3D,
    t: f64,
    tracking: Option<&FarmParticles>,
    bed: &Bathymetry2D,
) -> Result<(), String> {
    fields
        .write_solution_3d(t, state)
        .map_err(|e| e.to_string())?;
    if let (Some(writer), Some(tracking)) = (particles.as_mut(), tracking) {
        writer
            .write(&tracking.frame(state, t, bed))
            .map_err(|e| format!("particles: {e}"))?;
    }
    Ok(())
}

/// Particles released from the cages and tracked online in the 3D flow.
struct FarmParticles<'a> {
    tracker: ParticleTracker3D<'a>,
    tracker_mesh: &'a Mesh2D,
    ops: &'a DGOperators2D,
    n: usize,
    /// Cage centres
    centres: Vec<[f64; 2]>,
    /// Release time of each batch, and how many have been released
    batches: Vec<f64>,
    released_batches: usize,
    next_id: u64,
    /// The larvae's behaviour (`None`: passive)
    lice: Option<SalmonLice<ClearSkyLight>>,
    /// The waves' Stokes drift, added to the flow
    stokes: Option<&'a StokesDriftField>,
    /// Particles of each kind once released, with the cage and release
    /// time of each
    particles: [Vec<Particle3D>; 3],
    cages: [Vec<usize>; 3],
    born: [Vec<f64>; 3],
    /// The larvae again with other seeds and diffusivities (the
    /// connectivity's spread and sensitivity to `K`)
    replicates: Vec<(ParticleTracker3D<'a>, Vec<Particle3D>)>,
    /// The walks the larvae are tracked with, and the seeds of each
    diffusivities: Vec<Mixing>,
    /// Smagorinsky's diffusivity, if a walk takes it: the builder, `C_s`,
    /// and the field of the previous state
    smagorinsky: Option<(HorizontalDiffusivity<'a>, f64)>,
    smagorinsky_previous: Option<HorizontalDiffusivityField>,
    seeds: usize,
    /// The larvae's contacts with the cages, one recorder per diffusivity
    /// and seed, `[K × seed]` (the first for `particles[LARVAE]`, then one
    /// per replicate)
    contacts: Vec<ConnectivityRecorder>,
    /// The depth-mean current at the farm, and its value at each release
    farm: Probe2D,
    release_current: Vec<f64>,
    previous: Option<(f64, Solution3D)>,
    seconds: f64,
}

impl<'a> FarmParticles<'a> {
    fn new(
        mesh: &'a Mesh2D,
        ops: &'a DGOperators2D,
        n: usize,
        centres: &[[f64; 2]],
        batches: Vec<f64>,
        kh: f64,
        lice: Option<SalmonLice<ClearSkyLight>>,
    ) -> Self {
        let n_cages = centres.len() as f64;
        let farm = [
            centres.iter().map(|c| c[0]).sum::<f64>() / n_cages,
            centres.iter().map(|c| c[1]).sum::<f64>() / n_cages,
        ];
        let farm =
            Probe2D::at(&PointLocator2D::new(mesh), ops, farm).expect("the farm is in the channel");
        Self {
            tracker: ParticleTracker3D::new(mesh, ops)
                .with_horizontal_diffusivity(kh)
                .with_vertical_random_walk()
                .with_bed_settling(),
            tracker_mesh: mesh,
            ops,
            n,
            centres: centres.to_vec(),
            batches,
            released_batches: 0,
            next_id: 0,
            lice,
            stokes: None,
            particles: Default::default(),
            cages: Default::default(),
            born: Default::default(),
            replicates: Vec::new(),
            diffusivities: vec![Mixing::Constant(kh)],
            smagorinsky: None,
            smagorinsky_previous: None,
            seeds: 1,
            contacts: Vec::new(),
            farm,
            release_current: Vec::new(),
            previous: None,
            seconds: 0.0,
        }
    }

    fn with_stokes_drift(mut self, stokes: Option<&'a StokesDriftField>) -> Self {
        self.stokes = stokes;
        self
    }

    /// Record the larvae's contacts with the cages (each footprint over the
    /// top `depth` m) for `seeds` independent random walks in each of the
    /// walks `diffusivities` (the first is the tracker's own; Smagorinsky's
    /// with coefficient `cs` on `geom`). The seeds are the same for each
    /// walk, so the walks of two `K` are correlated and the independent
    /// errors of their difference overestimate it.
    fn with_connectivity(
        mut self,
        diffusivities: &[Mixing],
        seeds: usize,
        depth: f64,
        geom: &'a GeometricFactors2D,
        cs: f64,
    ) -> Self {
        let zones: Vec<ContactZone> = self
            .centres
            .iter()
            .map(|&c| ContactZone::circular(c, RADIUS, [0.0, depth]))
            .collect();
        // A return to its own cage only after a cage radius beyond it: not
        // the walk stepping out and back in at the edge
        let recorder = ConnectivityRecorder::new(zones).with_return_distance(RADIUS);
        self.contacts = vec![recorder; diffusivities.len() * seeds];
        self.replicates = diffusivities
            .iter()
            .flat_map(|&k| (0..seeds as u64).map(move |seed| (k, seed)))
            .skip(1)
            .map(|(k, seed)| {
                let tracker = self
                    .tracker
                    .clone()
                    .with_horizontal_diffusivity(k.constant())
                    .with_seed(seed);
                (tracker, Vec::new())
            })
            .collect();
        self.diffusivities = diffusivities.to_vec();
        self.seeds = seeds;
        if diffusivities.iter().any(|m| m.is_smagorinsky()) {
            let builder = HorizontalDiffusivity::new(self.tracker_mesh, self.ops, geom);
            self.smagorinsky = Some((builder, cs));
        }
        self
    }

    fn released(&self) -> usize {
        self.particles.iter().map(Vec::len).sum()
    }

    /// Every particle released by `t`, kind by kind, at its height in `state`.
    fn frame(&self, state: &Solution3D, t: f64, bed: &Bathymetry2D) -> ParticleFrame {
        let mut frame = ParticleFrame {
            t,
            ..Default::default()
        };
        let mut weights = vec![0.0; self.ops.n_nodes];
        for (kind, particles) in self.particles.iter().enumerate() {
            for (p, &born) in particles.iter().zip(&self.born[kind]) {
                let point = p.point();
                self.ops
                    .interpolation_weights_into(point.r, point.s, &mut weights);
                let (k, n) = (point.element.as_usize(), self.ops.n_nodes);
                let eta = &state.eta.data[k * n..(k + 1) * n];
                let (mut e, mut b) = (0.0, 0.0);
                for (i, (w, z)) in weights.iter().zip(bed.element(point.element)).enumerate() {
                    e += w * eta[i];
                    b += w * z;
                }
                let [x, y] = p.position();
                frame.xy.push([x as f32, y as f32]);
                frame.z.push((e + p.sigma() * (e - b).max(0.0)) as f32);
                frame.status.push(status_code(p.status()));
                frame.kind.push(kind as u8);
                frame.born.push(born as f32);
            }
        }
        frame
    }

    /// The next batch at `t`: `n` particles of each kind in each cage, a
    /// sunflower pattern over the footprint, evenly over the kind's depth
    /// range; the larvae also in each replicate and recorder. `current` is
    /// the depth-mean current at the farm.
    fn release(&mut self, t: f64, current: f64) {
        let batch = self.released_batches;
        let golden_angle = PI * (3.0 - 5.0_f64.sqrt());
        for (c, center) in self.centres.iter().enumerate() {
            for (kind, &(_, speed, [top, bottom])) in KINDS.iter().enumerate() {
                for i in 0..self.n {
                    let r = RADIUS * ((i as f64 + 0.5) / self.n as f64).sqrt();
                    let angle = i as f64 * golden_angle;
                    let p = [center[0] + r * angle.cos(), center[1] + r * angle.sin()];
                    let below = top + (bottom - top) * (i as f64 + 0.5) / self.n as f64;
                    let release = |tracker: &ParticleTracker3D| {
                        tracker
                            .release(self.next_id, p, -below / DEPTH, speed)
                            .expect("cages are in the channel")
                    };
                    if kind == LARVAE {
                        for (tracker, larvae) in &mut self.replicates {
                            larvae.push(release(tracker));
                        }
                        for recorder in &mut self.contacts {
                            recorder.add(c, batch, t);
                        }
                    }
                    self.particles[kind].push(release(&self.tracker));
                    self.cages[kind].push(c);
                    self.born[kind].push(t);
                    self.next_id += 1;
                }
            }
        }
        self.release_current.push(current);
        self.released_batches += 1;
    }

    /// Move the particles from the previous state to `state` at `t`, linear
    /// in time between the two, releasing the batches due on the way.
    fn advance(&mut self, state: &Solution3D, t: f64, sigma: &SigmaGrid, bed: &Bathymetry2D) {
        let start = Instant::now();
        let previous = self.previous.take();
        // Smagorinsky's diffusivity of this state, linear in time from the
        // previous one's
        let mixing = self.smagorinsky.as_ref().map(|(builder, cs)| {
            let viscosity = HorizontalViscosity2D::smagorinsky(*cs);
            builder.smagorinsky_3d(state, sigma, &viscosity, 1.0)
        });
        let mixing_previous = self.smagorinsky_previous.take();
        if let Some((t0, s0)) = &previous {
            let field = Solution3DVelocity::between(*t0, s0, t, state, sigma, bed, 0.05);
            let (u0, u1) = (
                self.farm.evaluate_field(&s0.ubar.data),
                self.farm.evaluate_field(&state.ubar.data),
            );
            let diffusivity = match (&mixing_previous, &mixing) {
                (Some(f0), Some(f1)) => Some(DiffusivityInTime::between(*t0, f0, t, f1)),
                _ => None,
            };
            match self.stokes {
                Some(stokes) => {
                    let field = WithStokesDrift3D::new(field, StokesDrift::steady(stokes));
                    self.track(field, diffusivity, *t0, t, [u0, u1]);
                }
                None => self.track(field, diffusivity, *t0, t, [u0, u1]),
            }
        }
        self.smagorinsky_previous = mixing;
        self.previous = match previous {
            Some((_, mut sp)) => {
                sp.clone_from(state);
                Some((t, sp))
            }
            None => Some((t, state.clone())),
        };
        self.seconds += start.elapsed().as_secs_f64();
    }

    /// Track from `t0` to `t` through `field`, and with the diffusivity
    /// `diffusivity` for the walks that take Smagorinsky's.
    fn track<F: ParticleVelocity3D + Copy>(
        &mut self,
        field: F,
        diffusivity: Option<DiffusivityInTime>,
        t0: f64,
        t: f64,
        current: [f64; 2],
    ) {
        match diffusivity {
            Some(d) => {
                let mixed = WithHorizontalDiffusivity3D::new(field, d);
                self.segments(&field, &mixed, t0, t, current);
            }
            None => self.segments(&field, &field, t0, t, current),
        }
    }

    /// Track from `t0` to `t`, releasing the batches due on the way, with
    /// the depth-mean current at the farm `current` at `t0` and `t`: the
    /// walks with a constant `K` through `plain`, Smagorinsky's through
    /// `mixed`.
    fn segments(
        &mut self,
        plain: &impl ParticleVelocity3D,
        mixed: &impl ParticleVelocity3D,
        t0: f64,
        t: f64,
        current: [f64; 2],
    ) {
        let mut from = t0;
        while from < t {
            while self
                .batches
                .get(self.released_batches)
                .is_some_and(|&b| b <= from + 1e-6)
            {
                let f = (from - t0) / (t - t0);
                self.release(from, current[0] + f * (current[1] - current[0]));
            }
            let to = match self.batches.get(self.released_batches) {
                Some(&b) if b < t => b,
                _ => t,
            };
            if self.released() > 0 {
                self.steps(plain, mixed, from, to);
            }
            from = to;
        }
    }

    /// Steps of at most 10 s from `from` to `to`, through `plain` or, for
    /// the walks that take Smagorinsky's diffusivity, `mixed`; the larvae by
    /// their behaviour, recording their contacts after each.
    fn steps(
        &mut self,
        plain: &impl ParticleVelocity3D,
        mixed: &impl ParticleVelocity3D,
        from: f64,
        to: f64,
    ) {
        let n = ((to - from) / 10.0).ceil().max(1.0) as usize;
        let dt = (to - from) / n as f64;
        let ops = self.ops;
        let mut weights = vec![0.0; ops.n_nodes];
        for s in 0..n {
            let t = from + s as f64 * dt;
            let main = self.diffusivities[0].is_smagorinsky();
            for (kind, particles) in self.particles.iter_mut().enumerate() {
                let lice = self.lice.as_ref().filter(|_| kind == LARVAE);
                if main {
                    step_kind(&self.tracker, particles, mixed, lice, t, dt);
                } else {
                    step_kind(&self.tracker, particles, plain, lice, t, dt);
                }
            }
            // Replicate r is walk (r + 1)/seeds of [walk × seed], seed 0 of
            // the first walk being the main tracker
            for (r, (tracker, larvae)) in self.replicates.iter_mut().enumerate() {
                let lice = self.lice.as_ref();
                if self.diffusivities[(r + 1) / self.seeds].is_smagorinsky() {
                    step_kind(tracker, larvae, mixed, lice, t, dt);
                } else {
                    step_kind(tracker, larvae, plain, lice, t, dt);
                }
            }
            // Contacts at the end of the step, of the larvae in the water
            let larvae = std::iter::once(&self.particles[LARVAE])
                .chain(self.replicates.iter().map(|(_, larvae)| larvae));
            for (recorder, larvae) in self.contacts.iter_mut().zip(larvae) {
                for (i, p) in larvae.iter().enumerate() {
                    if p.status() != ParticleStatus::Active {
                        continue;
                    }
                    let depth = || {
                        let point = p.point();
                        ops.interpolation_weights_into(point.r, point.s, &mut weights);
                        p.depth_below_surface(plain.depth(point.element, &weights, t + dt))
                    };
                    recorder.record(i, t + dt, dt, p.position(), depth, 1.0);
                }
            }
        }
    }

    /// The larvae's pen-to-pen connectivity in each diffusivity, per release
    /// batch (one diffusivity) and over all batches (mean ± spread across
    /// the seeds), then the shares against the diffusivity; every seed's
    /// matrix per batch to `csv`.
    fn report_connectivity(&self, csv: Option<&str>) -> std::io::Result<()> {
        if self.contacts.is_empty() || self.released() == 0 {
            return Ok(());
        }
        let zones = self.contacts[0].zones();
        let n_cages = zones.len();
        let recorders = |k: usize| &self.contacts[k * self.seeds..(k + 1) * self.seeds];
        let ensemble = |k: usize, groups: &dyn Fn(usize) -> bool| {
            ConnectivityEnsemble::new(recorders(k).iter().map(|r| r.matrix(0.0, groups)).collect())
        };
        println!(
            "\nPen-to-pen connectivity of the larvae: contact inside a cage's footprint over the top {} m (its own cage after a radius beyond it), {} seed{}",
            zones[0].depth[1],
            self.seeds,
            if self.seeds == 1 { "" } else { "s" }
        );
        println!(
            "  share in contact: mean ± spread across the seeds (one seed's binomial error); exposure per released larva; among those in contact, exposure 10/50/90 % and age at first contact 50 %"
        );
        let pair_lines = |k: usize, groups: &dyn Fn(usize) -> bool| {
            let ensemble = ensemble(k, groups);
            for i in 0..n_cages {
                for j in 0..n_cages {
                    let mut contacts: Vec<Contact> = recorders(k)
                        .iter()
                        .flat_map(|r| r.contacts(i, j, groups))
                        .collect();
                    let binomial = ensemble.members()[0].binomial_error(i, j);
                    let (exposure, _) = ensemble.mean_exposure(i, j);
                    print!(
                        "    cage {i} → {j}: {:5.1} ± {:4.1} % ({:4.1}), {exposure:6.1} s per larva",
                        100.0 * ensemble.mean(i, j),
                        100.0 * ensemble.spread(i, j),
                        100.0 * binomial
                    );
                    if !contacts.is_empty() {
                        let pct =
                            |v: &[f64], q: f64| v[(q * (v.len() - 1) as f64).round() as usize];
                        contacts.sort_by(|a, b| a.exposure.total_cmp(&b.exposure));
                        let e: Vec<f64> = contacts.iter().map(|c| c.exposure).collect();
                        let mut first: Vec<f64> = contacts.iter().map(|c| c.first).collect();
                        first.sort_by(f64::total_cmp);
                        print!(
                            "; exposure {:.0}/{:.0}/{:.0} s, first contact at {:.0} min",
                            pct(&e, 0.1),
                            pct(&e, 0.5),
                            pct(&e, 0.9),
                            pct(&first, 0.5) / 60.0
                        );
                    }
                    println!();
                }
            }
        };
        let phase = |t: f64| (M2 * t).to_degrees().rem_euclid(360.0);
        let released = &self.batches[..self.released_batches];
        let sweep = self.diffusivities.len() > 1;
        for (k, kh) in self.diffusivities.iter().enumerate() {
            if sweep {
                println!("  K = {kh} (m²/s):");
            } else {
                // Per batch for a single diffusivity only, to keep a sweep short
                for (b, &t) in released.iter().enumerate() {
                    println!(
                        "  batch {b}, released at {:.2} h (M2 forcing phase {:.0}°, depth-mean current at the farm {:+.2} m/s):",
                        t / 3600.0,
                        phase(t),
                        self.release_current[b]
                    );
                    pair_lines(k, &|g| g == b);
                }
            }
            if released.len() > 1 || sweep {
                println!("  all batches:");
                pair_lines(k, &|_| true);
            }
        }
        if sweep {
            // Each share's mean ± standard error of the mean, and its
            // difference from the first diffusivity in standard errors
            println!(
                "  Shares against K over all batches (mean ± standard error of the mean; z against K = {}):",
                self.diffusivities[0]
            );
            let pairs: Vec<(usize, usize)> = (0..n_cages)
                .flat_map(|i| (0..n_cages).map(move |j| (i, j)))
                .collect();
            let header: Vec<String> = pairs
                .iter()
                .map(|(i, j)| format!("{:>24}", format!("{i} → {j}")))
                .collect();
            println!("    K (m²/s){}", header.join(""));
            let reference = ensemble(0, &|_| true);
            for (k, kh) in self.diffusivities.iter().enumerate() {
                let e = ensemble(k, &|_| true);
                let cells: Vec<String> = pairs
                    .iter()
                    .map(|&(i, j)| {
                        let z = if k == 0 {
                            String::new()
                        } else {
                            format!(" z {:+.0}", e.difference(&reference, i, j).z())
                        };
                        let cell = format!(
                            "{:.1} ± {:.1} %{z}",
                            100.0 * e.mean(i, j),
                            100.0 * e.standard_error(i, j)
                        );
                        format!("{cell:>24}")
                    })
                    .collect();
                println!("    {kh:8}{}", cells.join(""));
            }
        }
        if let Some(path) = csv {
            use std::io::Write;
            let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
            writeln!(
                out,
                "kh,seed,batch,release_h,m2_phase_deg,current_farm,source,receiver,released,reached,share,exposure_per_larva_s"
            )?;
            for (index, recorder) in self.contacts.iter().enumerate() {
                let (kh, seed) = (self.diffusivities[index / self.seeds], index % self.seeds);
                for (b, &t) in released.iter().enumerate() {
                    let m = recorder.matrix(0.0, |g| g == b);
                    for i in 0..n_cages {
                        for j in 0..n_cages {
                            writeln!(
                                out,
                                "{kh},{seed},{b},{},{},{},{i},{j},{},{},{},{}",
                                t / 3600.0,
                                phase(t),
                                self.release_current[b],
                                m.released(i),
                                m.reached(i, j),
                                m.share(i, j),
                                m.mean_exposure(i, j)
                            )?;
                        }
                    }
                }
            }
            println!("  every seed's matrix per batch written to {path}");
        }
        Ok(())
    }

    /// Per kind: in the water and on the bed; depths, drift and spread of
    /// those in the water; where the settled ones landed.
    fn report(&self, state: &Solution3D, sigma: &SigmaGrid, bed: &Bathymetry2D) {
        if self.released() == 0 {
            println!("  (no particles released before the end)");
            return;
        }
        println!(
            "\nParticles: {} released in {} batch{} from {:.1} h, {:.2} s tracking ({} seed{} of the larvae)",
            self.released(),
            self.released_batches,
            if self.released_batches == 1 { "" } else { "es" },
            self.batches[0] / 3600.0,
            self.seconds,
            self.seeds,
            if self.seeds == 1 { "" } else { "s" }
        );
        if let Some(lice) = &self.lice {
            let t = self.previous.as_ref().map_or(0.0, |(t, _)| *t);
            println!(
                "  Sun at the end: {:.1}° high, {:.0} µmol photons/m²/s at the surface; larvae swim up where the light exceeds {} (≈ {:.0} m deep)",
                lice.light.solar_height(t),
                lice.light.irradiance(t),
                lice.light_threshold[0],
                (lice.light.irradiance(t) / lice.light_threshold[0]).ln() / lice.attenuation
            );
        }
        let field = Solution3DVelocity::steady(state, sigma, bed, 0.05);
        // The vertical diffusivity the walk sees, 150 m up-current of the farm
        let locator = PointLocator2D::new(self.tracker_mesh);
        let ahead = [self.centres[0][0] - 150.0, self.centres[0][1]];
        if let Some(point) = locator.locate(ahead) {
            let w = self.ops.interpolation_weights(point.r, point.s);
            let depth = field.depth(point.element, &w, 0.0);
            let profile: Vec<String> = [1.0, 3.0, 6.0, 10.0, 15.0, 25.0, 35.0]
                .iter()
                .map(|&z| {
                    let k = field
                        .diffusivity(point.element, &w, -z / depth, 0.0)
                        .map_or(0.0, |(k, _)| k);
                    format!("{z:.0} m {k:.1e}")
                })
                .collect();
            println!("  K (m²/s) up-current of the farm: {}", profile.join(", "));
            // Smagorinsky's horizontal K at the end, up-current and behind
            // the first cage, and over the domain's top layer
            if let Some((builder, cs)) = &self.smagorinsky {
                let viscosity = HorizontalViscosity2D::smagorinsky(*cs);
                let k_h = builder.smagorinsky_3d(state, sigma, &viscosity, 1.0);
                let at = |p: [f64; 2], z: f64| {
                    locator.locate(p).map_or(f64::NAN, |point| {
                        let w = self.ops.interpolation_weights(point.r, point.s);
                        let depth = field.depth(point.element, &w, 0.0);
                        k_h.at(point.element, &w, -z / depth).0
                    })
                };
                let lee = [self.centres[0][0] + 100.0, self.centres[0][1]];
                let n_total = self.tracker_mesh.n_elements * self.ops.n_nodes;
                let mut top = k_h.values()[k_h.values().len() - n_total..].to_vec();
                top.sort_by(f64::total_cmp);
                let pct = |q: f64| top[(q * (top.len() - 1) as f64).round() as usize];
                println!(
                    "  Smagorinsky K_h (m²/s, C_s {cs}, Pr_t 1) at 1/10 m: up-current {:.1e}/{:.1e}, 100 m east of cage 0 {:.1e}/{:.1e}; top layer 50/90/100 % {:.1e}/{:.1e}/{:.1e}",
                    at(ahead, 1.0),
                    at(ahead, 10.0),
                    at(lee, 1.0),
                    at(lee, 10.0),
                    pct(0.5),
                    pct(0.9),
                    pct(1.0)
                );
            }
            if let Some(stokes) = self.stokes {
                let profile: Vec<String> = [0.0, 1.0, 3.0, 5.0, 10.0, 20.0]
                    .iter()
                    .map(|&z| {
                        let [u, v] = stokes.at(point.element, &w, z);
                        format!("{z:.0} m {:.1}", 100.0 * u.hypot(v))
                    })
                    .collect();
                println!("  Stokes drift (cm/s): {}", profile.join(", "));
            }
        }
        // Displacement from the cage, the nearest periodic image along x
        let offset = |p: &Particle3D, c: usize| {
            let [x, y] = p.position();
            let dx = (x - self.centres[c][0] + 0.5 * LX).rem_euclid(LX) - 0.5 * LX;
            [dx, y - self.centres[c][1]]
        };
        for (kind, &(name, ..)) in KINDS.iter().enumerate() {
            let of_kind: Vec<(&Particle3D, usize)> = self.particles[kind]
                .iter()
                .zip(self.cages[kind].iter().copied())
                .collect();
            let settled: Vec<[f64; 2]> = of_kind
                .iter()
                .filter(|(p, _)| p.status() == ParticleStatus::Settled)
                .map(|&(p, c)| offset(p, c))
                .collect();
            let water: Vec<(f64, [f64; 2])> = of_kind
                .iter()
                .filter(|(p, _)| p.status() == ParticleStatus::Active)
                .map(|&(p, c)| {
                    let point = p.point();
                    let w = self.ops.interpolation_weights(point.r, point.s);
                    let depth = field.depth(point.element, &w, 0.0);
                    (p.depth_below_surface(depth), offset(p, c))
                })
                .collect();
            print!(
                "  {name:>6}: {:4} in the water, {:4} on the bed",
                water.len(),
                settled.len()
            );
            if !water.is_empty() {
                let mut depths: Vec<f64> = water.iter().map(|(d, _)| *d).collect();
                depths.sort_by(f64::total_cmp);
                let pct = |q: f64| depths[(q * (depths.len() - 1) as f64).round() as usize];
                let n = water.len() as f64;
                let mean = |d: usize| water.iter().map(|(_, x)| x[d]).sum::<f64>() / n;
                let spread = (water
                    .iter()
                    .map(|(_, x)| (x[0] - mean(0)).powi(2) + (x[1] - mean(1)).powi(2))
                    .sum::<f64>()
                    / n)
                    .sqrt();
                print!(
                    "; depth 10/50/90 % {:.1}/{:.1}/{:.1} m, drift ({:+.0}, {:+.0}) m, spread {spread:.0} m",
                    pct(0.1),
                    pct(0.5),
                    pct(0.9),
                    mean(0),
                    mean(1)
                );
            }
            if kind == LARVAE && self.lice.is_some() && !water.is_empty() {
                let swimming = |up: bool| {
                    of_kind
                        .iter()
                        .filter(|(p, _)| p.status() == ParticleStatus::Active)
                        .filter(|(p, _)| {
                            let w = p.swimming_speed();
                            if up { w > 0.0 } else { w < 0.0 }
                        })
                        .count() as f64
                        / water.len() as f64
                };
                print!(
                    "; swimming up {:.0} %, down {:.0} %",
                    100.0 * swimming(true),
                    100.0 * swimming(false)
                );
            }
            if !settled.is_empty() {
                let mut distance: Vec<f64> = settled.iter().map(|x| x[0].hypot(x[1])).collect();
                distance.sort_by(f64::total_cmp);
                let mean_x = settled.iter().map(|x| x[0]).sum::<f64>() / settled.len() as f64;
                print!(
                    "; landed {mean_x:+.0} m along the channel on average, 50/90 % within {:.0}/{:.0} m of the cage",
                    distance[distance.len() / 2],
                    distance[(9 * distance.len() / 10).min(distance.len() - 1)]
                );
            }
            println!();
        }
    }
}

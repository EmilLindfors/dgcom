//! dg-viz: a 3D view of a dg-rs simulation while it runs.
//!
//! The solver runs on its own thread ([`solver`]) and sends a snapshot of η and the
//! depth-averaged current every `--interval` seconds of model time; the viewer plays
//! them back ([`playback`]) as the water surface over the bed ([`surface`]), coloured by
//! current speed or elevation, with depth contours on the bed ([`contours`]), current
//! arrows ([`arrows`]), the land and sea around a domain on a real coast ([`terrain`]),
//! the farm's net cages riding the surface ([`cages`]) and particles released from
//! the cages, tracked with the flow ([`particles`]). R switches to the photo view
//! ([`photo`]): the sea as a camera would see it, wind waves riding the model's
//! current, the sky and the sun reflected in them, the bed through the water.
//!
//! Scenarios ([`scenario`]): the farm fjord of `examples/local_time_stepping_farm.rs`,
//! Frøya–Smøla–Hitra on the coastline mesh with NorKyst-800 boundary tides, as
//! `examples/froya_real_data.rs` runs it (its data files in `../data/`), or the
//! stratified farm channel of `examples/farm_3d.rs` or the farm fjord in 3D: there
//! the water shows the depth-mean current or any σ-layer's ([`layers`]), a sheet
//! at the strongest stratification of every column shows the pycnocline, halocline or thermocline ([`stratification`]), and lice larvae, faeces
//! and feed are tracked in 3D ([`cloud_3d`]).
//!
//! ```bash
//! cd viz && cargo run --release -- [options]
//! ```
//!
//! Without options (or with `--menu`) the viewer opens its start menu ([`menu`]): the
//! scenarios to run and the saved runs under `../output` to replay, with the common
//! options, and M in the viewer returns to it (`--menu --screenshot PATH` saves the
//! menu's window and ends).
//!
//! Options (defaults in brackets):
//! - `--scenario fjord|fjord3d|froya|channel` [fjord, given any other option]. Frøya reads
//!   `../data/froya_coast.msh` (from `scripts/gmsh_coastline_mesh.py`),
//!   `../data/froya_topobathy.tif` and `../data/froya_boundary_tides.txt`;
//!   the close-up (F) is the Mausund tide gauge, and it has no cages or particles.
//!   Its run is ≈ 30× faster than real time on 12 threads, so play it back at
//!   `--rate 30` or below to keep up with the solver.
//!   The channel runs in 3D, from rest with the tide ramped up over an hour; its
//!   particles start at once, three kinds per cage. `fjord3d` runs the fjord farm in
//!   3D the same way: its M2 through the open boundary, a brackish surface layer and a
//!   summer thermocline, T and S relaxing to them along the open boundary.
//! - `--replay FILE|DIR` play back a run instead of running the solver ([`replay`]),
//!   shown linear in time between its frames; the rate defaults to an hour per
//!   second, and the status shows the run's UTC date when the run has a clock.
//!   - A snapshot file (`froya_real_data snapshot_minutes=N` writes
//!     `<output>/froya.dgsnap`) carries its mesh and bed: no scenario is built, and
//!     the close-up is its first station. Its frames are read as they are shown, so
//!     a run of any length (15 days of 10-minute frames are ≈ 3 GB at Frøya) plays
//!     and seeks without `--memory`, and the trace has every frame.
//!   - A directory of VTU frames, the `froya_NNNN.vtu` of `examples/froya_real_data.rs`
//!     with `mesh=data/froya_coast.msh`, e.g. `--replay ../output/froya_15d_k1o1`. The
//!     scenario (Frøya unless `--scenario` says otherwise) must be built as the run
//!     was: the frames' nodes and bed are checked against it. Frames are read in
//!     parallel on `--threads`; the clock comes from the run's `run.log`.
//! - `--save-snapshot FILE` write the snapshots shown to a snapshot file (with the
//!   mesh, bed, cages and, in 3D, the σ-grid and u, v, T on every level): a live run as
//!   it runs, or the frames of a `--replay` as they are read (a tenth of VTU frames'
//!   size). `--replay FILE` replays it later without the scenario's data.
//! - `--terrain FILE[,FILE...]|none` elevation models (longitude/latitude GeoTIFFs of
//!   bed elevations with land heights, `scripts/kartverket_topobathy.sh`) drawn around
//!   and under the domain ([`terrain`]): the land, and the sea bed beyond the model
//!   under a still sea. The first is the base, the others finer patches over it (the
//!   1 m level around a farm, e.g. `../data/kattholmen_topobathy_2m.tif`). Only for a
//!   domain on a real coast, placed by its projection (Frøya, or a snapshot file with
//!   `projection=` metadata) [`../data/froya_topobathy.tif` if present]. `--terrain-margin F` how far around the mesh, in mesh extents [0.25];
//!   `--origin LAT,LON` the point the mesh's local projection is about, for snapshot
//!   files written before they carried it (`froya_real_data`: its `bbox=` centre,
//!   63.8,8.6 by default)
//! - `--site FILE` a fish farm site (`scripts/fiskeridir_site.sh`, e.g.
//!   `../data/sites/14042.txt`) on the map ([`site`]): its frame, moorings, feed raft
//!   and cages, the cages dragging on the flow of a live run. Repeat it for more sites;
//!   the close-up (F) then frames the first. Needs the domain's projection, as
//!   `--terrain` does
//! - `--pin X,Y` pin the point at mesh coordinates X, Y (km) at the start, as a click
//!   on it does ([`inspect`]: hover any point for its readout, click to pin one, Esc
//!   to let go)
//! - `--place NAME,X,Y` a named place the number keys frame (1 to 9, in the order:
//!   the run's stations, the `--site`s, then these), at mesh coordinates X, Y in km
//!   as the solver's logs print them; repeat it for more
//! - `--gauge FILE` a tide-gauge record (`scripts/kartverket_gauge.sh` format, e.g.
//!   `../data/tide_gauges/mausund_obs.txt`) drawn against the model's surface at the
//!   close-up point in the trace (top right, G; [`trace`]); it needs the run's clock,
//!   so a replay's or a `--start`-dated run's
//! - `--sigma N` σ-levels of a 3D run [16; 12 for `fjord3d`]; `--start TIME` the UTC
//!   instant of model time 0, for the larvae's daylight [2025-06-15T00:00:00Z]; `--lice
//!   ladim|johnsen|passive` the larvae's behaviour (`dg_rs::particles::SalmonLice`)
//!   [ladim]; `--sheet density|salinity|temperature|off` what the stratification
//!   sheet follows at the start ([`stratification`]) [density]; `--layer
//!   mean|surface|bed|N` the current the water shows at the start (N: σ-layer from the
//!   bed, 0) [mean]
//! - `--mesh PATH` Gmsh mesh [`../tests/data/gmsh/fjord_farm.msh`; for `fjord3d`
//!   `fjord_farm_3d.msh`, 40 m quads at the farm; or Frøya's]
//! - `--order N` polynomial order [2; 1 for `fjord3d`]; `--levels N` local time
//!   stepping levels, 0 for global SSP-RK3 (2D only) [8]
//! - `--hours H` model hours to run [25]; `--interval S` model seconds between
//!   snapshots [60]; `--threads N` solver threads [all but two cores]
//! - `--memory MB` snapshots kept, oldest dropped first (a live run, or VTU frames)
//!   [2048]
//! - `--rate R` model seconds shown per second [60; 3600 in a replay]
//! - `--vz Z` vertical exaggeration [3]; `--speed-max V` fix the speed scale (m/s)
//!   instead of letting it follow the flow in round steps
//! - `--colour speed|elevation|hs|tp|dissipation` what the water's colour shows at the
//!   start (C cycles it; the last three are a run's waves: a snapshot file of a
//!   coupled wave run, `froya_real_data wave_coupling=`) [speed]
//! - `--water-alpha A` opacity of the translucent water, 0.05–0.95 [0.4]; lower shows
//!   more of the bed (its depth scale is in the legend), higher more of the colouring
//! - `--ramp S` seconds over which the tide ramps up from rest; 0 switches it on at
//!   once, and the start-up surge then leaves a drag wake at the farm that outlasts
//!   the tidal current there [3600]
//! - `--drag on|off` whether the nets drag on the flow (drawn either way) [on]
//! - `--particles N` particles released in each cage every `--release S` model seconds
//!   [20, 60] (in 3D, of each kind); 0 for none. `--kh K` horizontal diffusivity of
//!   their random walk (m²/s) [0.1]
//! - `--view farm|domain|N` starting framing: the close-up (the farm, or the gauge),
//!   the whole domain, or the Nth named place (its number key) [farm]
//! - `--eye X,Y,HEIGHT,BEARING,TILT` start from a photographer's view instead: standing
//!   over mesh point X, Y (km), HEIGHT m above mean sea level, looking toward the
//!   compass BEARING and TILT degrees down (the photo view's status prints the view's
//!   own; with `--vz 1` the heights are true)
//! - `--minimap on|off` the map of the whole domain in a corner ([`minimap`]) [on]
//! - `--photo on|off` start in the photo view [off]; `--wind SPEED,FROM` its wind, in
//!   m/s at 10 m and the compass degrees it blows from [6,225]; `--fetch KM` the open
//!   water upwind of the view, which sets how developed its sea is [10]
//! - `--show keys|run|view|colour|overlays|photo|help` open the keys' help or a menu
//!   of the menu bar at the start ([`menubar`]), e.g. for a `--screenshot`
//! - `--frame-times N` time N frames once the shown time reaches `--at` (as a
//!   screenshot waits), without vsync, print the frame times' mean, median and 95th
//!   percentile, and end: the cost of a view (the minimap, the photo view) on this
//!   machine's GPU
//! - `--screenshot PATH` save a frame once the shown time reaches `--at S` (model
//!   seconds; default the end of the run) and quit: for checking a change headlessly.
//!
//! The menu bar along the top ([`menubar`]) does by mouse what the keys do, F1 (or H,
//! or ?) lists the keys with what each shows now; both read one table of actions
//! ([`actions`]).

mod actions;
mod arrows;
mod cages;
mod camera;
mod cloud_3d;
mod colormap;
mod contours;
mod field;
mod hud;
mod inspect;
mod layers;
mod menu;
mod menubar;
mod minimap;
mod particles;
mod photo;
mod playback;
mod plot;
mod replay;
mod scenario;
mod site;
mod solver;
mod stratification;
mod surface;
mod terrain;
mod trace;

/// The viewer's font: Fira Mono (SIL Open Font License 1.1, `assets/fonts/OFL.txt`),
/// whole, where Bevy bundles a subset without Norwegian letters.
const FONT: &[u8] = include_bytes!("../assets/fonts/FiraMono-Regular.ttf");

use std::path::PathBuf;
use std::sync::Mutex;

use bevy::camera::visibility::RenderLayers;
use bevy::prelude::*;
use bevy::render::view::screenshot::{Screenshot, save_to_disk};
use dg_rs::mesh::PointLocator2D;
use dg_rs::time::ModelClock;

use arrows::{Arrows, ArrowsPlugin};
use cages::{CageLayout, CagesPlugin};
use camera::{CameraPlugin, OrbitCamera, Views};
use contours::ContoursPlugin;
use field::{Field, Frame, Nodes, Probe, ShownLayer};
use hud::{HudPlugin, RunClock, Title};
use layers::{LayersPlugin, Levels};
use particles::{Lice, ParticleConfig, ParticlesPlugin, Periodic};
use photo::{Photo, PhotoPlugin};
use playback::{Playback, PlaybackPlugin, SolverChannel, SolverState, Source};
use replay::Replay;
use scenario::Scenario;
use site::{SiteDrawing, SitePlugin, Sites};
use solver::SolverConfig;
use stratification::{SheetShows, Stratification, StratificationPlugin};
use surface::{ColourBy, Colouring, SurfacePlugin, WaterOpacity};
use terrain::{Terrain, TerrainPlugin};
use trace::{Trace, TracePlugin};

struct Args {
    scenario: Option<String>,
    replay: Option<PathBuf>,
    save_snapshot: Option<PathBuf>,
    gauge: Option<PathBuf>,
    terrain: Option<String>,
    terrain_margin: f64,
    sites: Vec<PathBuf>,
    /// Named places (`--place NAME,X_KM,Y_KM`), in mesh metres
    places: Vec<(String, [f64; 2])>,
    /// A point pinned at the start (`--pin X_KM,Y_KM`), in mesh metres
    pin: Option<[f64; 2]>,
    origin: Option<String>,
    mesh: Option<PathBuf>,
    /// `--order`, `--sigma`: else the scenario's defaults
    order: Option<usize>,
    levels: usize,
    hours: f64,
    interval: f64,
    threads: usize,
    memory_mb: usize,
    rate: Option<f64>,
    vz: f32,
    speed_max: Option<f32>,
    colour: ColourBy,
    water_alpha: f32,
    drag: bool,
    ramp: Option<f64>,
    particles: ParticleConfig,
    view: String,
    screenshot: Option<String>,
    at: Option<f64>,
    /// `--frame-times N`: time N frames without vsync, print them and end
    frame_times: Option<usize>,
    sigma: Option<usize>,
    start: String,
    sheet: SheetShows,
    layer: String,
    photo: Photo,
    minimap: bool,
    /// `--show`: the help or a menu open at the start
    show: actions::Shown,
    /// `--eye`: mesh point (m), height (m), bearing and tilt (degrees)
    eye: Option<([f64; 2], f32, f32, f32)>,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut args = Self {
            scenario: None,
            replay: None,
            save_snapshot: None,
            gauge: None,
            terrain: None,
            terrain_margin: 0.25,
            sites: Vec::new(),
            places: Vec::new(),
            pin: None,
            origin: None,
            mesh: None,
            order: None,
            levels: 8,
            hours: 25.0,
            interval: 60.0,
            threads: std::thread::available_parallelism()
                .map_or(1, |n| n.get().saturating_sub(2).max(1)),
            memory_mb: 2048,
            rate: None,
            vz: 3.0,
            speed_max: None,
            colour: ColourBy::Speed,
            water_alpha: WaterOpacity::default().0,
            drag: true,
            ramp: None,
            particles: ParticleConfig {
                per_release: 20,
                release_every: 60.0,
                kh: 0.1,
                lice: Lice::Ladim,
            },
            view: "farm".into(),
            screenshot: None,
            at: None,
            frame_times: None,
            sigma: None,
            start: "2025-06-15T00:00:00Z".into(),
            sheet: SheetShows::Density,
            layer: "mean".into(),
            minimap: true,
            show: actions::Shown::default(),
            photo: Photo {
                on: false,
                wind_speed: 6.0,
                wind_from: 225.0,
                fetch: 10e3,
                place: None,
            },
            eye: None,
        };
        let mut it = std::env::args().skip(1);
        while let Some(flag) = it.next() {
            let value = it.next().ok_or_else(|| format!("{flag} needs a value"))?;
            match flag.as_str() {
                "--scenario" => args.scenario = Some(value),
                "--replay" => args.replay = Some(value.into()),
                "--gauge" => args.gauge = Some(value.into()),
                "--terrain" => args.terrain = Some(value),
                "--terrain-margin" => args.terrain_margin = num(&flag, &value)?,
                "--site" => args.sites.push(value.into()),
                "--place" => args.places.push(place(&value)?),
                "--pin" => {
                    let (_, at) = place(&format!("pin,{value}"))
                        .map_err(|_| format!("--pin takes X,Y (km), not {value}"))?;
                    args.pin = Some(at);
                }
                "--origin" => args.origin = Some(value),
                "--save-snapshot" => args.save_snapshot = Some(value.into()),
                "--mesh" => args.mesh = Some(value.into()),
                "--order" => args.order = Some(num(&flag, &value)?),
                "--levels" => args.levels = num(&flag, &value)?,
                "--hours" => args.hours = num(&flag, &value)?,
                "--interval" => args.interval = num(&flag, &value)?,
                "--threads" => args.threads = num(&flag, &value)?,
                "--memory" => args.memory_mb = num(&flag, &value)?,
                "--rate" => args.rate = Some(num(&flag, &value)?),
                "--vz" => args.vz = num(&flag, &value)?,
                "--colour" | "--color" => {
                    args.colour = ColourBy::parse(&value).ok_or_else(|| {
                        format!(
                            "--colour takes speed, elevation, hs, tp or dissipation, not {value}"
                        )
                    })?;
                }
                "--speed-max" => args.speed_max = Some(num(&flag, &value)?),
                "--water-alpha" => {
                    args.water_alpha = num(&flag, &value)?;
                    if !(0.0..=1.0).contains(&args.water_alpha) {
                        return Err(format!("--water-alpha takes 0 to 1, not {value}"));
                    }
                }
                "--ramp" => args.ramp = Some(num(&flag, &value)?),
                "--drag" => {
                    args.drag = match value.as_str() {
                        "on" => true,
                        "off" => false,
                        _ => return Err(format!("--drag takes on or off, not {value}")),
                    }
                }
                "--particles" => args.particles.per_release = num(&flag, &value)?,
                "--release" => args.particles.release_every = num(&flag, &value)?,
                "--kh" => args.particles.kh = num(&flag, &value)?,
                "--lice" => {
                    args.particles.lice = match value.as_str() {
                        "ladim" => Lice::Ladim,
                        "johnsen" => Lice::Johnsen,
                        "passive" => Lice::Passive,
                        _ => {
                            return Err(format!(
                                "--lice takes ladim, johnsen or passive, not {value}"
                            ));
                        }
                    }
                }
                "--sigma" => args.sigma = Some(num(&flag, &value)?),
                "--start" => args.start = value,
                "--sheet" => {
                    args.sheet = match value.as_str() {
                        "density" => SheetShows::Density,
                        "salinity" => SheetShows::Salinity,
                        "temperature" => SheetShows::Temperature,
                        "off" => SheetShows::Off,
                        _ => {
                            return Err(format!(
                                "--sheet takes density, salinity, temperature or off, not {value}"
                            ));
                        }
                    }
                }
                "--layer" => args.layer = value,
                "--minimap" => {
                    args.minimap = match value.as_str() {
                        "on" => true,
                        "off" => false,
                        _ => return Err(format!("--minimap takes on or off, not {value}")),
                    }
                }
                "--photo" => {
                    args.photo.on = match value.as_str() {
                        "on" => true,
                        "off" => false,
                        _ => return Err(format!("--photo takes on or off, not {value}")),
                    }
                }
                "--wind" => {
                    let bad = || format!("--wind takes SPEED,FROM (m/s, degrees), not {value}");
                    let (speed, from) = value.split_once(',').ok_or_else(bad)?;
                    args.photo.wind_speed = speed.trim().parse().map_err(|_| bad())?;
                    args.photo.wind_from = from.trim().parse().map_err(|_| bad())?;
                    if args.photo.wind_speed < 0.0 {
                        return Err(bad());
                    }
                }
                "--show" => {
                    args.show = actions::Shown::parse(&value).ok_or_else(|| {
                        format!(
                            "--show takes keys, run, view, colour, overlays, photo or help,                              not {value}"
                        )
                    })?;
                }
                "--eye" => args.eye = Some(eye(&value)?),
                "--fetch" => args.photo.fetch = 1e3 * num::<f32>(&flag, &value)?,
                "--view" => args.view = value,
                "--screenshot" => args.screenshot = Some(value),
                "--at" => args.at = Some(num(&flag, &value)?),
                "--frame-times" => args.frame_times = Some(num(&flag, &value)?),
                _ => {
                    return Err(format!(
                        "unknown option {flag} (see the docs at the top of viz/src/main.rs)"
                    ));
                }
            }
        }
        Ok(args)
    }
}

fn num<T: std::str::FromStr>(flag: &str, value: &str) -> Result<T, String> {
    value
        .parse()
        .map_err(|_| format!("bad value for {flag}: {value}"))
}

/// `--screenshot`: the frame to save, and how many frames since the shown time reached it.
#[derive(Resource)]
struct Capture {
    path: Option<String>,
    at: f64,
    frames: Option<u32>,
}

/// `X,Y,HEIGHT,BEARING,TILT` (X, Y in km of mesh coordinates) as a photographer's view.
fn eye(value: &str) -> Result<([f64; 2], f32, f32, f32), String> {
    let bad = || format!("--eye takes X,Y,HEIGHT,BEARING,TILT (km, km, m, °, °), not {value}");
    let parts: Vec<f64> = value
        .split(',')
        .map(|s| s.trim().parse().map_err(|_| bad()))
        .collect::<Result<_, _>>()?;
    match parts[..] {
        [x, y, height, bearing, tilt] => Ok((
            [1e3 * x, 1e3 * y],
            height as f32,
            bearing as f32,
            tilt as f32,
        )),
        _ => Err(bad()),
    }
}

/// `NAME,X,Y` (X, Y in km of mesh coordinates) as a named place in metres.
fn place(value: &str) -> Result<(String, [f64; 2]), String> {
    let bad = || format!("--place takes NAME,X,Y (km), not {value}");
    let mut parts = value.rsplitn(3, ',');
    let y: f64 = parts
        .next()
        .and_then(|s| s.trim().parse().ok())
        .ok_or_else(bad)?;
    let x: f64 = parts
        .next()
        .and_then(|s| s.trim().parse().ok())
        .ok_or_else(bad)?;
    let name = parts
        .next()
        .map(str::trim)
        .filter(|n| !n.is_empty())
        .ok_or_else(bad)?;
    Ok((name.to_string(), [1e3 * x, 1e3 * y]))
}

/// Fira Mono in full in place of Bevy's default subset, which has no ø, æ, å.
fn install_font(app: &mut App) {
    app.world_mut()
        .resource_mut::<Assets<Font>>()
        .insert(AssetId::default(), Font::from_bytes(FONT.to_vec()))
        .expect("the default font's handle");
}

fn main() -> AppExit {
    let repo = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/.."));
    let given: Vec<String> = std::env::args().skip(1).collect();
    match given.iter().map(String::as_str).collect::<Vec<_>>()[..] {
        [] | ["--menu"] => return menu::run(repo, None, install_font),
        ["--menu", "--screenshot", path] => {
            return menu::run(repo, Some(path.into()), install_font);
        }
        _ => {}
    }
    let args = match Args::parse() {
        Ok(args) => args,
        Err(e) => {
            eprintln!("{e}");
            return AppExit::from_code(2);
        }
    };
    // A snapshot file holds its own domain; VTU frames are of a Frøya run unless
    // told otherwise
    let snapshot_file = args.replay.as_deref().filter(|p| p.is_file());
    let (mut scenario, replay) = if let Some(path) = snapshot_file {
        match Replay::snapshot(path) {
            Ok((replay, scenario)) => (scenario, Some(replay)),
            Err(e) => {
                eprintln!("cannot replay {}: {e}", path.display());
                return AppExit::from_code(1);
            }
        }
    } else {
        let scenario_name = args.scenario.clone().unwrap_or_else(|| {
            if args.replay.is_some() {
                "froya"
            } else {
                "fjord"
            }
            .into()
        });
        // The 3D fjord at P1 on 12 levels, ≈ 4× cheaper than P2 on 16
        let (order, sigma) = match scenario_name.as_str() {
            "fjord3d" => (args.order.unwrap_or(1), args.sigma.unwrap_or(12)),
            _ => (args.order.unwrap_or(2), args.sigma.unwrap_or(16)),
        };
        let built = match scenario_name.as_str() {
            "fjord" => {
                let mesh = args
                    .mesh
                    .clone()
                    .unwrap_or_else(|| repo.join("tests/data/gmsh/fjord_farm.msh"));
                Scenario::fjord_farm(&mesh, order)
            }
            "froya" => {
                let data = repo.join("data");
                let mesh = args
                    .mesh
                    .clone()
                    .unwrap_or_else(|| data.join("froya_coast.msh"));
                println!(
                    "Building Frøya from {} (the bed projection takes a while)...",
                    data.display()
                );
                Scenario::froya(
                    &mesh,
                    &data.join("froya_topobathy.tif"),
                    &data.join("froya_boundary_tides.txt"),
                    order,
                    args.hours * 3600.0,
                )
            }
            "channel" => match ModelClock::parse(&args.start) {
                Ok(clock) => Ok(Scenario::farm_channel(order, sigma, clock)),
                Err(e) => Err(format!("--start {}: {e}", args.start).into()),
            },
            "fjord3d" => match ModelClock::parse(&args.start) {
                Ok(clock) => {
                    let mesh = args
                        .mesh
                        .clone()
                        .unwrap_or_else(|| repo.join("tests/data/gmsh/fjord_farm_3d.msh"));
                    Scenario::fjord_farm_3d(&mesh, order, sigma, clock)
                }
                Err(e) => Err(format!("--start {}: {e}", args.start).into()),
            },
            other => {
                eprintln!("unknown scenario {other}: fjord, fjord3d, froya or channel");
                return AppExit::from_code(2);
            }
        };
        let mut scenario = match built {
            Ok(s) => s,
            Err(e) => {
                eprintln!("cannot build the {scenario_name} scenario: {e}");
                return AppExit::from_code(1);
            }
        };
        let replay = match args
            .replay
            .as_deref()
            .map(|d| Replay::vtu(d, &mut scenario))
        {
            None => None,
            Some(Ok(replay)) => Some(replay),
            Some(Err(e)) => {
                eprintln!(
                    "cannot replay {}: {e}",
                    args.replay.as_ref().unwrap().display()
                );
                return AppExit::from_code(1);
            }
        };
        (scenario, replay)
    };
    if let Some(origin) = &args.origin {
        match scenario::parse_projection(&format!("local,{origin}")) {
            Some(projection) => scenario.projection = Some(projection),
            None => {
                eprintln!("--origin takes LAT,LON in degrees, not {origin}");
                return AppExit::from_code(2);
            }
        }
    }
    // The gauge trace stays at the scenario's point of interest when a site takes the
    // close-up
    let gauge_point = scenario.farm;
    // The scenario's own framing, for its stations and `--place`s
    let place_framing = scenario.close_up;
    // Fish farm sites: their cages join the scenario's, the close-up frames the first
    let mut sites = Vec::new();
    let mut site_places = Vec::new();
    for path in &args.sites {
        let Some(projection) = scenario.projection else {
            eprintln!(
                "--site {}: the domain has no projection (--origin LAT,LON)",
                path.display()
            );
            return AppExit::from_code(2);
        };
        let site = match dg_rs::io::read_farm_site_file(path) {
            Ok(site) => site,
            Err(e) => {
                eprintln!("cannot read the site {}: {e}", path.display());
                return AppExit::from_code(1);
            }
        };
        let cages = site::site_cages(&site, &projection, &scenario.cages);
        println!(
            "Site {} {}: {} cages added",
            site.number,
            site.name,
            cages.len()
        );
        if sites.is_empty() {
            scenario.farm = site::site_centre(&site, &projection);
            scenario.close_up = scenario::CloseUp {
                distance: 700.0,
                yaw: 0.7,
                pitch: 0.45,
                arrow_spacing: 25.0,
                arrow_radius: 600.0,
            };
        }
        scenario.cages.extend(cages);
        site_places.push((site.name.clone(), site::site_centre(&site, &projection)));
        sites.push(site);
    }
    let mut replay = replay;
    if let Some(replay) = &mut replay {
        println!(
            "Replaying {} frames of {}, {} s apart, t = {} to {} s",
            replay.frames(),
            args.replay.as_ref().unwrap().display(),
            replay.interval,
            replay.t_first,
            replay.t_last
        );
        let title = scenario
            .name
            .split(':')
            .next()
            .unwrap_or_default()
            .to_string();
        if let Some(path) = &args.save_snapshot {
            if let Err(e) = replay.save_to(path, &scenario) {
                eprintln!("cannot write {}: {e}", path.display());
                return AppExit::from_code(1);
            }
            println!("Saving the frames to {}", path.display());
        }
        scenario.name = match replay.clock {
            Some(clock) => format!("{title}: replay from {} UTC", &clock.format(0.0)[..16]),
            None => format!("{title}: replay"),
        };
    }
    // A live run saved as it runs
    let live_save = match (&replay, &args.save_snapshot) {
        (None, Some(path)) => match replay::SnapshotSink::create(
            path,
            &scenario,
            scenario.clock.as_ref(),
            args.particles.per_release > 0,
            false,
        ) {
            Ok(sink) => {
                println!("Saving the run to {}", path.display());
                Some(sink)
            }
            Err(e) => {
                eprintln!("cannot write {}: {e}", path.display());
                return AppExit::from_code(1);
            }
        },
        _ => None,
    };
    if let Some(ramp) = args.ramp {
        scenario.forcing.ramp = ramp;
    }
    if !sites.is_empty() {
        let names: Vec<&str> = sites.iter().map(|s| s.name.as_str()).collect();
        scenario.name = format!("{} + {}", scenario.name, names.join(", "));
    }

    let (lo, hi) = scenario.mesh.vertices.iter().fold(
        ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]),
        |(lo, hi), v| {
            (
                [lo[0].min(v[0]), lo[1].min(v[1])],
                [hi[0].max(v[0]), hi[1].max(v[1])],
            )
        },
    );
    let frame = Frame {
        origin: [0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])],
        vz: args.vz,
    };
    let nodes = Nodes::new(&scenario, &frame);
    let locator = PointLocator2D::new(&scenario.mesh);
    let cages = CageLayout(
        scenario
            .cages
            .iter()
            .map(|cage| {
                let (c0, c1) = cage.footprint.bounding_box();
                let centre = [0.5 * (c0[0] + c1[0]), 0.5 * (c0[1] + c1[1])];
                (cage.clone(), Probe::at(&locator, &scenario, centre))
            })
            .collect(),
    );
    // The land and sea around a domain on a real coast
    let terrain_paths: Option<Vec<PathBuf>> = match args.terrain.as_deref() {
        Some("none") => None,
        Some(list) => Some(list.split(',').map(PathBuf::from).collect()),
        None => Some(vec![repo.join("data/froya_topobathy.tif")]).filter(|p| p[0].is_file()),
    };
    let terrain = match (terrain_paths, &scenario.projection) {
        (Some(paths), Some(projection)) => {
            let started = std::time::Instant::now();
            let built = Terrain::build(
                &paths,
                projection,
                args.terrain_margin,
                &scenario,
                &nodes,
                &locator,
                &frame,
            );
            match built {
                Ok(terrain) => {
                    let grids: Vec<String> = terrain
                        .grids()
                        .iter()
                        .map(|([nx, ny], dx)| format!("{nx} x {ny} at {dx:.1} m"))
                        .collect();
                    println!(
                        "Terrain: {}, built in {:.1} s",
                        grids.join(" + "),
                        started.elapsed().as_secs_f64()
                    );
                    Some(terrain)
                }
                Err(e) => {
                    eprintln!("no terrain: {e}");
                    None
                }
            }
        }
        (Some(_), None) if args.terrain.is_some() => {
            eprintln!("no terrain: the domain has no projection (--origin LAT,LON)");
            None
        }
        _ => None,
    };
    let sites = scenario.projection.map(|projection| {
        Sites(
            sites
                .iter()
                .map(|site| SiteDrawing::new(site, &projection, &scenario, &nodes, &locator))
                .collect(),
        )
    });
    let arrows = Arrows::new(
        &locator,
        &scenario,
        48,
        scenario.close_up.arrow_spacing,
        scenario.close_up.arrow_radius,
    );
    println!(
        "{}: {} elements (P{}), {} nodes; {} + {} current arrows",
        scenario.name,
        scenario.mesh.n_elements,
        scenario.ops.order,
        nodes.len(),
        arrows.count().0,
        arrows.count().1
    );

    let extent = (hi[0] - lo[0]).max(hi[1] - lo[1]) as f32;
    // A place framed as `close_up` frames the point of interest
    let framed = |at: [f64; 2], close_up: &scenario::CloseUp| OrbitCamera {
        focus: frame.world(at, 0.0),
        yaw: close_up.yaw,
        pitch: close_up.pitch,
        distance: close_up.distance,
    };
    // The run's stations and the `--place`s as the scenario frames its point of
    // interest, the sites as the farm close-up
    let places = scenario
        .places
        .iter()
        .map(|(name, at)| (name.clone(), framed(*at, &place_framing)))
        .chain(
            site_places
                .iter()
                .map(|(name, at)| (name.clone(), framed(*at, &scenario.close_up))),
        )
        .chain(
            args.places
                .iter()
                .map(|(name, at)| (name.clone(), framed(*at, &place_framing))),
        )
        .collect();
    let views = Views {
        farm: framed(scenario.farm, &scenario.close_up),
        places,
        domain: OrbitCamera {
            focus: Vec3::ZERO,
            yaw: 0.35,
            pitch: 0.7,
            distance: 0.85 * extent,
        },
    };
    let start = match args.eye {
        Some((at, height, bearing, tilt)) => photo::eye_view(&frame, at, height, bearing, tilt),
        None => match args.view.as_str() {
            "farm" => views.farm,
            "domain" => views.domain,
            n => match n
                .parse::<usize>()
                .ok()
                .and_then(|n| views.places.get(n.wrapping_sub(1)))
            {
                Some((_, view)) => *view,
                None => {
                    eprintln!(
                        "--view takes farm, domain or a place's number (1 to {}), not {n}",
                        views.places.len()
                    );
                    return AppExit::from_code(2);
                }
            },
        },
    };

    // The water column of a 3D run: its levels
    let levels = scenario.three_d.as_ref().map(|three_d| {
        let farm = Probe::at(&locator, &scenario, scenario.farm);
        let bed: Vec<f32> = scenario.bathymetry.data.iter().map(|&b| b as f32).collect();
        Levels {
            sigma: three_d
                .sigma
                .sigma_rho()
                .iter()
                .map(|&s| s as f32)
                .collect(),
            depth: farm.map_or(0.0, |p| -p.eval(&bed)),
        }
    });
    let top = scenario
        .three_d
        .as_ref()
        .map_or(0, |three_d| three_d.sigma.n_levels() - 1);
    let shown = match args.layer.as_str() {
        "mean" => ShownLayer::DepthMean,
        "surface" => ShownLayer::Level(top),
        "bed" => ShownLayer::Level(0),
        n => match n.parse::<usize>() {
            Ok(l) if l <= top => ShownLayer::Level(l),
            _ => {
                eprintln!("--layer takes mean, surface, bed or 0 to {top}, not {n}");
                return AppExit::from_code(2);
            }
        },
    };
    let periodic = Periodic(scenario.periodic.map(|p| p.map(|x| x as f32)));

    // The model times the gauge trace covers, and the run's clock
    let (span, run_clock) = match &replay {
        Some(replay) => ([replay.t_first, replay.t_last], replay.clock),
        None => ([0.0, args.hours * 3600.0], scenario.clock),
    };
    let gauge = match &args.gauge {
        None => None,
        Some(path) => match dg_rs::io::read_tide_gauge_file(path) {
            Ok(file) => {
                let name = file.station.as_ref().map_or_else(
                    || {
                        path.file_stem()
                            .unwrap_or_default()
                            .to_string_lossy()
                            .into_owned()
                    },
                    |s| s.name.clone(),
                );
                let series = &file.time_series;
                let record = series
                    .times()
                    .iter()
                    .copied()
                    .zip(series.values().iter().copied())
                    .collect();
                Some((name, record))
            }
            Err(e) => {
                eprintln!("cannot read the gauge {}: {e}", path.display());
                return AppExit::from_code(1);
            }
        },
    };
    if gauge.is_some() && run_clock.is_none() {
        eprintln!("--gauge needs the run's clock: the gauge is not drawn");
    }
    let mut trace = Trace::new(&scenario, &locator, gauge_point, span, gauge, run_clock);
    // VTU frames are hourly: the run's own record of the point, if it wrote one
    if let (Some(dir), None, Some(clock), Some(projection)) = (
        args.replay.as_deref().filter(|p| p.is_dir()),
        snapshot_file,
        run_clock,
        scenario.projection,
    ) && let Some((name, record)) = replay::station_record(dir, gauge_point, &projection)
    {
        println!(
            "Trace: the run's record at {name}, {} samples",
            record.len()
        );
        trace = trace.with_record(&record, clock);
    }
    let mut inspector = inspect::Inspector::new(
        scenario.mesh.clone(),
        scenario.ops.clone(),
        scenario.projection,
        &scenario.bathymetry.data,
        scenario.three_d.as_ref().map(|three_d| {
            three_d
                .sigma
                .sigma_rho()
                .iter()
                .map(|&s| s as f32)
                .collect()
        }),
    );

    // A snapshot file is read on demand: its frame times, and where to ask for frames
    let mut on_demand = None;
    let (channel, source, t_end, interval, rate) = match replay {
        Some(replay) => {
            let source = Source::Replay {
                name: args
                    .replay
                    .as_ref()
                    .and_then(|d| d.file_name())
                    .map_or(String::new(), |n| n.to_string_lossy().into_owned()),
                frames: replay.frames(),
                t_last: replay.t_last,
            };
            let (t_end, interval) = (replay.t_last, replay.interval);
            let channel = match replay::serve(replay, trace.probe().cloned()) {
                Ok(served) => {
                    on_demand = Some((served.times, served.requests));
                    served.messages
                }
                Err(replay) => replay::spawn(*replay, args.threads),
            };
            (
                channel,
                source,
                t_end,
                interval,
                args.rate.unwrap_or(3600.0),
            )
        }
        None => {
            let t_end = args.hours * 3600.0;
            let channel = solver::spawn(
                &scenario,
                SolverConfig {
                    t_end,
                    interval: args.interval,
                    levels: args.levels,
                    threads: args.threads,
                    drag: args.drag,
                    particles: args.particles,
                },
                live_save,
            );
            let rate = args.rate.unwrap_or(60.0);
            (channel, Source::Solver, t_end, args.interval, rate)
        }
    };
    let mut playback = Playback::new(rate, interval, args.memory_mb << 20);
    if let Some((times, requests)) = on_demand {
        playback = playback.read_on_demand(times, requests);
    }

    let mut app = App::new();
    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            title: "dg-viz".into(),
            resolution: (1600, 900).into(),
            // Timed frames are not held to the display's rate
            present_mode: if args.frame_times.is_some() {
                bevy::window::PresentMode::AutoNoVsync
            } else {
                bevy::window::PresentMode::default()
            },
            ..default()
        }),
        ..default()
    }))
    .insert_resource(ClearColor(Color::srgb(0.05, 0.065, 0.08)))
    .insert_resource(GlobalAmbientLight {
        color: Color::srgb(0.75, 0.85, 1.0),
        brightness: 350.0,
        ..default()
    })
    .insert_resource(frame)
    .insert_resource(nodes)
    .insert_resource(cages)
    .insert_resource(arrows)
    .insert_resource(actions::Actions::new(&views))
    .insert_resource(args.show)
    .insert_resource(views)
    .insert_resource(Field::default())
    .insert_resource(shown)
    .insert_resource(periodic)
    .insert_resource(Colouring::new(args.speed_max, args.colour))
    // The waves' maps read on opaque water, not mixed with the bed's colours
    .insert_resource(if args.colour.is_waves() {
        surface::SurfaceStyle::Opaque
    } else {
        surface::SurfaceStyle::Translucent
    })
    .insert_resource(WaterOpacity(args.water_alpha))
    .insert_resource(Title(scenario.name.clone()))
    .insert_resource(Photo {
        // The real sun stands over the domain's centre
        place: scenario.projection.map(|p| {
            let [x, y] = frame.origin;
            let (lat, lon) = dg_rs::io::CoordinateProjection::xy_to_geo(&p, x, y);
            [lat, lon]
        }),
        ..args.photo.clone()
    })
    .insert_resource(trace)
    .insert_resource({
        if let Some(at) = args.pin {
            inspector.pin(at);
        }
        inspector
    })
    .insert_resource(playback)
    .insert_resource(SolverChannel(Mutex::new(channel)))
    .insert_resource(source)
    .insert_resource(minimap::Minimap::new(
        frame.world(lo, 0.0).xz().min(frame.world(hi, 0.0).xz()),
        frame.world(lo, 0.0).xz().max(frame.world(hi, 0.0).xz()),
        args.minimap,
    ))
    .insert_resource(Capture {
        path: args.screenshot,
        at: args.at.unwrap_or(t_end),
        frames: None,
    })
    .insert_resource(FrameTimes {
        wanted: args.frame_times.unwrap_or(0),
        at: args.at.unwrap_or(t_end),
        samples: Vec::new(),
    })
    .add_plugins((actions::ActionsPlugin, menubar::MenubarPlugin))
    .add_plugins((
        PlaybackPlugin,
        SurfacePlugin,
        ContoursPlugin,
        CagesPlugin,
        ArrowsPlugin,
        ParticlesPlugin,
        CameraPlugin,
        minimap::MinimapPlugin,
        HudPlugin,
        TracePlugin,
        TerrainPlugin,
        SitePlugin,
        inspect::InspectPlugin,
        PhotoPlugin,
    ))
    .add_systems(Startup, move |mut commands: Commands| {
        commands.spawn((
            Camera3d::default(),
            // The HUD's, not the minimap's
            IsDefaultUiCamera,
            RenderLayers::from_layers(&[0, minimap::MAIN_ONLY]),
            Projection::Perspective(PerspectiveProjection {
                near: 1.0,
                far: 400_000.0,
                ..default()
            }),
            start.transform(),
            start,
        ));
        // Sun from the south-west, 35° up.
        commands.spawn((
            DirectionalLight {
                illuminance: 9_000.0,
                shadow_maps_enabled: false,
                ..default()
            },
            Transform::from_xyz(-1.0, 1.0, 1.0).looking_at(Vec3::ZERO, Vec3::Y),
        ));
    })
    .add_systems(Update, field::interpolate)
    .add_systems(Update, capture.after(field::interpolate))
    .add_systems(Last, frame_times)
    .add_systems(Update, (back_to_menu, jump))
    .add_systems(Last, playback::settle);
    if let Some(clock) = run_clock {
        app.insert_resource(RunClock(clock));
    }
    if let Some(terrain) = terrain {
        app.insert_resource(terrain);
    }
    if let Some(sites) = sites {
        app.insert_resource(sites);
    }
    if let Some(levels) = levels {
        app.insert_resource(levels)
            .insert_resource(Stratification::new(args.sheet))
            .add_plugins((LayersPlugin, StratificationPlugin));
    }
    install_font(&mut app);
    app.run()
}

/// M (or Run → Back to the list of runs): back to the start menu; Run → Quit ends. A viewer the menu started ends with
/// [`menu::BACK_TO_MENU`] and the menu shows again; one started from the command
/// line opens a menu and ends.
fn back_to_menu(mut acts: MessageReader<actions::Act>, mut exit: MessageWriter<AppExit>) {
    use actions::{Act, Action};
    let mut back = false;
    for &Act(action) in acts.read() {
        match action {
            Action::Menu => back = true,
            Action::Quit => {
                exit.write(AppExit::Success);
            }
            _ => {}
        }
    }
    if !back {
        return;
    }
    if std::env::var_os(menu::MENU_ENV).is_some() {
        exit.write(AppExit::from_code(menu::BACK_TO_MENU));
        return;
    }
    match std::env::current_exe().and_then(|exe| std::process::Command::new(exe).spawn()) {
        Ok(_) => {
            exit.write(AppExit::Success);
        }
        Err(e) => eprintln!("cannot open the menu: {e}"),
    }
}

/// F, O and 1 to 9 (or the View menu): the close-up, the whole domain, a named place.
fn jump(
    mut acts: MessageReader<actions::Act>,
    views: Res<Views>,
    mut cameras: Query<&mut OrbitCamera>,
) {
    use actions::{Act, Action};
    let Ok(mut orbit) = cameras.single_mut() else {
        return;
    };
    for &Act(action) in acts.read() {
        let to = match action {
            Action::CloseUp => Some(views.farm),
            Action::Overview => Some(views.domain),
            Action::Place(i) => views.places.get(i).map(|(_, view)| *view),
            _ => None,
        };
        if let Some(to) = to {
            *orbit = to;
        }
    }
}

/// `--frame-times`: how many frames to time, from when, and their times so far (s).
#[derive(Resource)]
struct FrameTimes {
    wanted: usize,
    at: f64,
    samples: Vec<f32>,
}

/// Frames to let pass before timing, for the meshes and pipelines to reach the GPU.
const WARM_UP: usize = 60;

/// Times `--frame-times` frames once the shown time reaches `--at` (paused there), then
/// prints their mean, median and 95th percentile (ms) and ends.
fn frame_times(
    time: Res<Time<Real>>,
    field: Res<Field>,
    mut playback: ResMut<Playback>,
    mut times: ResMut<FrameTimes>,
    mut exit: MessageWriter<AppExit>,
) {
    if times.wanted == 0 || field.t.is_none_or(|t| t < times.at - 1.0) {
        return;
    }
    playback.paused = true;
    times.samples.push(time.delta_secs());
    if times.samples.len() < WARM_UP + times.wanted {
        return;
    }
    let mut ms: Vec<f32> = times.samples[WARM_UP..].iter().map(|s| 1e3 * s).collect();
    ms.sort_by(f32::total_cmp);
    let mean = ms.iter().sum::<f32>() / ms.len() as f32;
    println!(
        "frame times over {} frames: mean {mean:.2} ms, median {:.2} ms, p95 {:.2} ms",
        ms.len(),
        ms[ms.len() / 2],
        ms[(ms.len() * 95 / 100).min(ms.len() - 1)]
    );
    times.wanted = 0;
    exit.write(AppExit::Success);
}

fn capture(
    mut commands: Commands,
    mut capture: ResMut<Capture>,
    mut playback: ResMut<Playback>,
    field: Res<Field>,
    mut exit: MessageWriter<AppExit>,
) {
    let Some(path) = capture.path.clone() else {
        return;
    };
    match capture.frames.as_mut() {
        None => {
            let ended = !matches!(playback.solver, SolverState::Running)
                && !playback.growing()
                && playback.newest().is_some_and(|t| playback.t >= t);
            if field.t.is_some_and(|t| t >= capture.at) || (field.t.is_some() && ended) {
                playback.paused = true;
                capture.frames = Some(0);
            }
        }
        Some(n) => {
            *n += 1;
            // A few frames for the meshes to reach the GPU, then some for the file to be written.
            if *n == 10 {
                commands
                    .spawn(Screenshot::primary_window())
                    .observe(save_to_disk(path));
            } else if *n == 40 {
                exit.write(AppExit::Success);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_place_is_named_in_km() {
        assert_eq!(
            place("pinnacle, 10.6,-13.8").unwrap(),
            ("pinnacle".to_string(), [10_600.0, -13_800.0])
        );
        // The name may hold commas; the last two fields are the point
        assert_eq!(place("Hitra, south,1,2").unwrap().0, "Hitra, south");
        assert!(place("10.6,-13.8").is_err());
        assert!(place("pinnacle,x,-13.8").is_err());
    }
}

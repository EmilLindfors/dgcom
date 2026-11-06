//! `--record DIR`: a video's frames, rendered on a fixed clock and saved as
//! `DIR/frame_00000.png`, `frame_00001.png`, …, then the viewer ends.
//!
//! The camera and the model time follow a path file (`--camera FILE`), one key per
//! line, `#` starting a comment:
//!
//! ```text
//! # video_s  model_h  kind   X_km  Y_km  DIST_km|HEIGHT_m  BEARING  PITCH|TILT
//! 0          10.0     orbit  0     0     60                340      40
//! 8          16.0     orbit  4.5   7.0   3                 250      25
//! 12         16.0     eye    3.0   7.7   20                270      4
//! ```
//!
//! - `orbit`: the camera looks at mesh point X, Y (km) from DIST km away, toward the
//!   compass BEARING, PITCH degrees above the horizontal (the orbit camera's view).
//! - `eye`: a photographer standing over mesh point X, Y, HEIGHT m above mean sea
//!   level, looking toward BEARING, TILT degrees down (as `--eye`).
//!
//! The video lasts until the last key, `--fps` frames per second [30]. Between the
//! keys the model time runs linearly, and the view (focus, log distance, heading,
//! pitch) follows a cubic through the keys with Catmull-Rom slopes inside and none at
//! the ends, so a shot starts and stops smoothly; a key repeated at a later time
//! holds the view. Without `--camera`, the view stays where `--view`/`--eye` put it
//! and the model time runs from `--at` at `--rate` model seconds per video second.
//!
//! Each frame waits until the shown field is exactly the frame's model time (an
//! on-demand replay's frames read and interpolated), one more frame for the meshes to
//! reach the GPU, and is saved; only then does the clock (Bevy's [`Time`], so the
//! photo view's waves too) advance by 1/fps. Nothing depends on how fast the machine
//! renders. `--ui off|legend|on` [legend] hides the menus, status and trace (`legend`
//! keeps the colour scale of the water); `--size WxH` [1920x1080] sets the frames'
//! pixels.

use std::path::PathBuf;
use std::time::Duration;

use bevy::prelude::*;
use bevy::render::view::screenshot::{Screenshot, save_to_disk};
use bevy::time::TimeUpdateStrategy;

use crate::camera::OrbitCamera;
use crate::field::{Field, Frame};
use crate::hud::{BedLegend, Legend};
use crate::photo::eye_view;
use crate::playback::Playback;

/// Which overlays a recording keeps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ui {
    Off,
    Legend,
    On,
}

impl std::str::FromStr for Ui {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "off" => Ok(Self::Off),
            "legend" => Ok(Self::Legend),
            "on" => Ok(Self::On),
            _ => Err(format!("--ui takes off, legend or on, not {s}")),
        }
    }
}

/// A key's view as written in the path file (mesh km, metres, degrees).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum KeyView {
    Orbit {
        at: [f64; 2],
        distance_km: f32,
        bearing: f32,
        pitch: f32,
    },
    Eye {
        at: [f64; 2],
        height: f32,
        bearing: f32,
        tilt: f32,
    },
}

/// One line of the path file.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Key {
    /// Video time (s)
    pub video: f64,
    /// Model time (s)
    pub model: f64,
    pub view: KeyView,
}

/// Read a path file's keys, in video-time order.
pub fn parse(text: &str) -> Result<Vec<Key>, String> {
    let mut keys = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let bad = || {
            format!(
                "line {} of the camera path: `{line}` is not \
                 VIDEO_S MODEL_H orbit|eye X_KM Y_KM DIST_KM|HEIGHT_M BEARING PITCH|TILT",
                i + 1
            )
        };
        let words: Vec<&str> = line.split_whitespace().collect();
        if words.len() != 8 {
            return Err(bad());
        }
        let n = |j: usize| words[j].parse::<f64>().map_err(|_| bad());
        let at = [1e3 * n(3)?, 1e3 * n(4)?];
        let (a, b, c) = (n(5)? as f32, n(6)? as f32, n(7)? as f32);
        let view = match words[2] {
            "orbit" => KeyView::Orbit {
                at,
                distance_km: a,
                bearing: b,
                pitch: c,
            },
            "eye" => KeyView::Eye {
                at,
                height: a,
                bearing: b,
                tilt: c,
            },
            _ => return Err(bad()),
        };
        let key = Key {
            video: n(0)?,
            model: 3600.0 * n(1)?,
            view,
        };
        if keys.last().is_some_and(|k: &Key| key.video <= k.video) {
            return Err(format!(
                "line {} of the camera path: video times must increase",
                i + 1
            ));
        }
        keys.push(key);
    }
    if keys.is_empty() {
        return Err("the camera path has no keys".into());
    }
    Ok(keys)
}

/// The keys in the orbit camera's numbers: focus X, focus Z (world m), ln distance,
/// yaw (unwrapped, so that consecutive keys turn the short way) and pitch (rad).
#[derive(Clone, Debug)]
pub struct CameraPath {
    video: Vec<f64>,
    model: Vec<f64>,
    params: Vec<[f32; 5]>,
}

impl CameraPath {
    pub fn new(keys: &[Key], frame: &Frame) -> Self {
        let mut params: Vec<[f32; 5]> = Vec::with_capacity(keys.len());
        for key in keys {
            let orbit = match key.view {
                KeyView::Orbit {
                    at,
                    distance_km,
                    bearing,
                    pitch,
                } => OrbitCamera {
                    focus: frame.world(at, 0.0),
                    yaw: -bearing.to_radians(),
                    pitch: pitch.to_radians(),
                    distance: 1e3 * distance_km,
                },
                KeyView::Eye {
                    at,
                    height,
                    bearing,
                    tilt,
                } => eye_view(frame, at, height, bearing, tilt),
            };
            let mut yaw = orbit.yaw;
            if let Some(last) = params.last() {
                let tau = std::f32::consts::TAU;
                yaw = last[3] + (yaw - last[3] + std::f32::consts::PI).rem_euclid(tau)
                    - std::f32::consts::PI;
            }
            params.push([
                orbit.focus.x,
                orbit.focus.z,
                orbit.distance.max(1.0).ln(),
                yaw,
                orbit.pitch,
            ]);
        }
        Self {
            video: keys.iter().map(|k| k.video).collect(),
            model: keys.iter().map(|k| k.model).collect(),
            params,
        }
    }

    /// Length of the video (s).
    pub fn duration(&self) -> f64 {
        *self.video.last().unwrap()
    }

    /// Model time (s) and view at video time `t`.
    pub fn at(&self, t: f64) -> (f64, OrbitCamera) {
        let n = self.video.len();
        let i = self.video.partition_point(|&v| v <= t).clamp(1, n.max(2) - 1);
        let p = if n == 1 {
            self.params[0]
        } else {
            let (t0, t1) = (self.video[i - 1], self.video[i]);
            let h = t1 - t0;
            let s = ((t - t0) / h).clamp(0.0, 1.0);
            // Catmull-Rom slopes (per second) inside, none at the ends
            let slope = |k: usize| -> [f64; 5] {
                if k == 0 || k == n - 1 {
                    return [0.0; 5];
                }
                let dt = self.video[k + 1] - self.video[k - 1];
                std::array::from_fn(|c| {
                    (self.params[k + 1][c] as f64 - self.params[k - 1][c] as f64) / dt
                })
            };
            let (m0, m1) = (slope(i - 1), slope(i));
            let (h00, h10, h01, h11) = (
                2.0 * s * s * s - 3.0 * s * s + 1.0,
                s * s * s - 2.0 * s * s + s,
                -2.0 * s * s * s + 3.0 * s * s,
                s * s * s - s * s,
            );
            std::array::from_fn(|c| {
                (h00 * self.params[i - 1][c] as f64
                    + h10 * h * m0[c]
                    + h01 * self.params[i][c] as f64
                    + h11 * h * m1[c]) as f32
            })
        };
        let model = if n == 1 {
            self.model[0]
        } else {
            let (t0, t1) = (self.video[i - 1], self.video[i]);
            let s = ((t - t0) / (t1 - t0)).clamp(0.0, 1.0);
            self.model[i - 1] + s * (self.model[i] - self.model[i - 1])
        };
        let orbit = OrbitCamera {
            focus: Vec3::new(p[0], 0.0, p[1]),
            yaw: p[3],
            pitch: p[4].clamp(0.005, 1.56),
            distance: p[2].exp(),
        };
        (model, orbit)
    }
}

/// Where the model time and the view come from.
pub enum Drive {
    Path(CameraPath),
    /// No path: the view stays, the model time runs from `start` at `rate`
    Clock { start: f64, rate: f64, seconds: f64 },
}

enum State {
    /// Frames rendered before the first, for the terrain and meshes to load
    Warmup(u32),
    /// Show the next frame's time and view
    Set,
    /// Until the field is the frame's time
    Wait,
    /// One more frame, then save
    Settle,
    /// Frames for the last files to be written
    Done(u32),
}

#[derive(Resource)]
pub struct Recording {
    dir: PathBuf,
    fps: f64,
    drive: Drive,
    ui: Ui,
    frame: usize,
    frames: usize,
    state: State,
}

impl Recording {
    pub fn new(dir: PathBuf, fps: f64, drive: Drive, ui: Ui) -> Self {
        let seconds = match &drive {
            Drive::Path(path) => path.duration(),
            Drive::Clock { seconds, .. } => *seconds,
        };
        Self {
            dir,
            fps,
            drive,
            ui,
            frame: 0,
            frames: (seconds * fps).floor() as usize + 1,
            state: State::Warmup(0),
        }
    }

    fn target(&self) -> (f64, Option<OrbitCamera>) {
        let t = self.frame as f64 / self.fps;
        match &self.drive {
            Drive::Path(path) => {
                let (model, view) = path.at(t);
                (model, Some(view))
            }
            Drive::Clock { start, rate, .. } => (start + rate * t, None),
        }
    }
}

/// Frames rendered at the first frame's time before it is saved
const WARMUP: u32 = 45;

pub struct RecordPlugin;

impl Plugin for RecordPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (
                show.before(crate::field::interpolate),
                save.after(crate::field::interpolate),
            )
                .run_if(resource_exists::<Recording>),
        )
        .add_systems(PostUpdate, hide_ui.run_if(resource_exists::<Recording>));
    }
}

/// Puts the frame's model time and view on screen.
fn show(
    mut recording: ResMut<Recording>,
    mut playback: ResMut<Playback>,
    mut strategy: ResMut<TimeUpdateStrategy>,
    mut cameras: Query<&mut OrbitCamera>,
) {
    if matches!(recording.state, State::Settle | State::Done(_)) {
        return;
    }
    let (model, view) = recording.target();
    // Within the run's frames, so that the field can reach it
    let model = match playback.span() {
        Some((first, last)) => model.clamp(first, last),
        None => model,
    };
    playback.paused = true;
    if playback.t != model {
        playback.t = model;
    }
    // Frames of an on-demand replay arrive while waiting: interpolate again
    playback.changed = true;
    if let (Some(view), Ok(mut orbit)) = (view, cameras.single_mut())
        && *orbit != view
    {
        *orbit = view;
    }
    if matches!(recording.state, State::Set) {
        *strategy = TimeUpdateStrategy::ManualDuration(Duration::ZERO);
        recording.state = State::Wait;
    }
}

/// Saves the frame once the field is the frame's time, then advances the clock.
fn save(
    mut commands: Commands,
    mut recording: ResMut<Recording>,
    playback: Res<Playback>,
    field: Res<Field>,
    mut strategy: ResMut<TimeUpdateStrategy>,
    mut exit: MessageWriter<AppExit>,
) {
    let ready = playback.exact() && field.t == Some(playback.t);
    match recording.state {
        State::Warmup(n) => {
            *strategy = TimeUpdateStrategy::ManualDuration(Duration::ZERO);
            let n = if ready { n + 1 } else { 0 };
            recording.state = if n >= WARMUP {
                State::Settle
            } else {
                State::Warmup(n)
            };
        }
        State::Set => {}
        State::Wait => {
            if ready {
                recording.state = State::Settle;
            }
        }
        State::Settle => {
            let path = recording.dir.join(format!("frame_{:05}.png", recording.frame));
            commands
                .spawn(Screenshot::primary_window())
                .observe(save_to_disk(path));
            recording.frame += 1;
            if recording.frame.is_multiple_of(30) || recording.frame == recording.frames {
                println!(
                    "recorded {} of {} frames (model t = {:.2} h)",
                    recording.frame,
                    recording.frames,
                    playback.t / 3600.0
                );
            }
            if recording.frame >= recording.frames {
                recording.state = State::Done(0);
            } else {
                *strategy =
                    TimeUpdateStrategy::ManualDuration(Duration::from_secs_f64(1.0 / recording.fps));
                recording.state = State::Set;
            }
        }
        State::Done(n) => {
            if n >= 60 {
                exit.write(AppExit::Success);
            } else {
                recording.state = State::Done(n + 1);
            }
        }
    }
}

/// Hides the overlays a recording leaves out.
#[allow(clippy::type_complexity)]
fn hide_ui(
    recording: Res<Recording>,
    mut roots: Query<(&mut Visibility, Has<Legend>), (With<Node>, Without<ChildOf>)>,
    mut bed: Query<&mut Visibility, (With<BedLegend>, With<ChildOf>)>,
) {
    if recording.ui == Ui::On {
        return;
    }
    for (mut visibility, legend) in &mut roots {
        let keep = legend && recording.ui == Ui::Legend;
        let wanted = if keep {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
        if *visibility != wanted {
            *visibility = wanted;
        }
    }
    for mut visibility in &mut bed {
        if *visibility != Visibility::Hidden {
            *visibility = Visibility::Hidden;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame() -> Frame {
        Frame {
            origin: [0.0, 0.0],
            vz: 1.0,
        }
    }

    #[test]
    fn a_path_file_is_read_in_km_hours_and_degrees() {
        let keys = parse(
            "# a comment\n0 10 orbit 1 2 3 90 30\n\n4.5 12 eye 3.0 7.7 20 270 4 # trailing\n",
        )
        .unwrap();
        assert_eq!(keys.len(), 2);
        assert_eq!(keys[0].model, 36_000.0);
        assert_eq!(
            keys[0].view,
            KeyView::Orbit {
                at: [1000.0, 2000.0],
                distance_km: 3.0,
                bearing: 90.0,
                pitch: 30.0
            }
        );
        assert!(matches!(keys[1].view, KeyView::Eye { height: 20.0, .. }));
        assert!(parse("1 0 orbit 0 0 1 0 30\n0 0 orbit 0 0 1 0 30").is_err());
        assert!(parse("0 0 fly 0 0 1 0 30").is_err());
        assert!(parse("# nothing").is_err());
    }

    #[test]
    fn the_path_passes_through_its_keys_and_starts_and_stops_at_rest() {
        let keys = parse("0 0 orbit 0 0 10 0 40\n5 2 orbit 4 0 2 90 20\n10 4 orbit 4 4 2 90 20")
            .unwrap();
        let path = CameraPath::new(&keys, &frame());
        let (model, at_key) = path.at(5.0);
        assert_eq!(model, 7200.0);
        assert!((at_key.focus.x - 4000.0).abs() < 1e-2);
        assert!((at_key.distance - 2000.0).abs() < 1e-1);
        assert!((at_key.yaw + 90f32.to_radians()).abs() < 1e-5);
        // Model time is linear between keys
        assert_eq!(path.at(2.5).0, 3600.0);
        // Eased: the first and last 1 % of the time move the focus far less than 1 %
        let start = path.at(0.0).1.focus;
        let early = path.at(0.05).1.focus;
        assert!((early - start).length() < 0.001 * 4000.0);
        let end = path.at(10.0).1.focus;
        let late = path.at(9.95).1.focus;
        assert!((end - late).length() < 0.001 * 4000.0);
        // Beyond the ends, the end keys
        assert_eq!(path.at(20.0).0, 14_400.0);
    }

    #[test]
    fn the_heading_turns_the_short_way_round() {
        // From 350° to 10°: through north, not back through south
        let keys = parse("0 0 orbit 0 0 1 350 30\n2 0 orbit 0 0 1 10 30").unwrap();
        let path = CameraPath::new(&keys, &frame());
        let mid = -path.at(1.0).1.yaw.to_degrees();
        assert!((mid.rem_euclid(360.0) - 0.0).abs() < 1e-3 || (mid.rem_euclid(360.0) - 360.0).abs() < 1e-3);
    }

    #[test]
    fn an_eye_key_is_the_photographers_view() {
        let keys = parse("0 0 eye 3 7.7 20 270 4").unwrap();
        let path = CameraPath::new(&keys, &frame());
        let expected = eye_view(&frame(), [3000.0, 7700.0], 20.0, 270.0, 4.0);
        let got = path.at(0.0).1;
        assert!((got.focus - expected.focus).length() < 1e-2);
        assert!((got.distance - expected.distance).abs() < 1e-2);
    }

    /// Bevy checks a system's queries for conflicting access when it first runs: run
    /// the recording's systems once (`hide_ui`'s two `Visibility` queries once panicked
    /// there, B0001).
    #[test]
    fn the_recording_systems_run() {
        let keys = parse("0 0 orbit 0 0 1 0 30\n1 0 orbit 0 0 1 0 30").unwrap();
        let mut world = World::new();
        world.insert_resource(Recording::new(
            PathBuf::from("x"),
            30.0,
            Drive::Path(CameraPath::new(&keys, &frame())),
            Ui::Legend,
        ));
        world.insert_resource(Playback::new(3600.0, 3600.0, 1 << 20));
        world.insert_resource(Field::default());
        world.insert_resource(TimeUpdateStrategy::default());
        world.init_resource::<Messages<AppExit>>();
        let root = world.spawn((Node::default(), Visibility::default())).id();
        let legend = world.spawn((Node::default(), Visibility::default(), Legend)).id();
        let bed = world
            .spawn((Node::default(), Visibility::default(), BedLegend, ChildOf(legend)))
            .id();
        let mut schedule = Schedule::default();
        schedule.add_systems((show, save, hide_ui).chain());
        schedule.run(&mut world);
        assert_eq!(world.get::<Visibility>(root), Some(&Visibility::Hidden));
        assert_eq!(world.get::<Visibility>(legend), Some(&Visibility::Inherited));
        assert_eq!(world.get::<Visibility>(bed), Some(&Visibility::Hidden));
    }

    #[test]
    fn frames_cover_the_video_inclusive() {
        let keys = parse("0 0 orbit 0 0 1 0 30\n2 1 orbit 0 0 1 0 30").unwrap();
        let rec = Recording::new(
            PathBuf::from("x"),
            30.0,
            Drive::Path(CameraPath::new(&keys, &frame())),
            Ui::Off,
        );
        assert_eq!(rec.frames, 61);
    }
}

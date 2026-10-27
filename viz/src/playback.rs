//! The snapshots received so far and the model time the viewer shows.
//!
//! [`Playback`] keeps the solver's snapshots in arrival order, dropping the oldest
//! beyond a memory budget (counted in the bytes held: snapshots grow with their
//! particles), and a clock that runs `rate` seconds of model time per
//! second. The clock never passes the newest snapshot: while the solver is behind, the
//! view waits for it. Between two snapshots the field is interpolated linearly in
//! time ([`crate::field::Field`]). A replay ([`Source::Replay`]) sends the frames of a
//! finished run down the same channel.
//!
//! A snapshot file is read on demand instead ([`OnDemand`], `crate::replay::serve`):
//! the playback knows every frame's time from the start, asks the reading thread for
//! the two frames around the shown time and the next one, and holds only those
//! ([`HELD`]), so a run of any length plays and seeks within a few megabytes. The
//! gauge trace's samples of every frame arrive beside them ([`Playback::series`]). A
//! file a run is still writing grows as new frames appear; the clock stops at the
//! newest, as it waits for a running solver.
//!
//! Keys: Space pauses, `[` and `]` halve and double the rate, ← and → step back and
//! forward by ten snapshots, Home and End jump to the oldest and newest.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use bevy::prelude::*;

use crate::actions::{Act, Action};
use crate::screen::{RunResources, Screen, Viewing};
use crate::solver::{Snapshot, SolverMessage};

/// Where the snapshots come from.
#[derive(Resource)]
pub enum Source {
    /// The solver, running while the viewer draws
    Solver,
    /// The frames of a finished run ([`crate::replay`])
    Replay {
        /// The run's output directory, as named on the command line
        name: String,
        /// Frames in the run
        frames: usize,
        /// Model time of the last frame (s)
        t_last: f64,
    },
}

/// Frames an on-demand replay holds at most: the two around the shown time, the next
/// one, and one being replaced.
pub const HELD: usize = 4;

/// Wall time (s) after its last new frame for which a file read on demand counts as
/// still being written: the playback waits at its newest frame meanwhile.
pub const GROWING_FOR: f64 = 30.0;

/// A snapshot file read on demand: every frame's time, and where to ask for frames.
pub struct OnDemand {
    /// Model time of every frame (s)
    pub times: Vec<f64>,
    /// Frame indices the playback needs and does not hold, newest list first served
    requests: Sender<Vec<usize>>,
    /// The frames last asked for
    asked: Vec<usize>,
    /// Wall time (s since the viewer started) at which the file last grew
    pub grew: Option<f64>,
}

/// How the run on the solver thread ended.
pub enum SolverState {
    Running,
    Finished { steps: usize, wall: f64 },
    Failed(String),
}

#[derive(Resource)]
pub struct Playback {
    pub frames: VecDeque<Snapshot>,
    /// Model time shown (s)
    pub t: f64,
    /// Model seconds per second
    pub rate: f64,
    pub paused: bool,
    pub solver: SolverState,
    /// Model time between snapshots (s)
    pub interval: f64,
    /// Bytes of snapshots kept at most
    budget: usize,
    /// Bytes of the snapshots received from the solver and kept
    kept: usize,
    /// Set when the shown model time or the snapshots around it changed this frame
    pub changed: bool,
    started: Instant,
    /// Wall time (s) at which the newest snapshot arrived
    pub newest_wall: f64,
    /// Set for a snapshot file read on demand
    pub on_demand: Option<OnDemand>,
    /// (model time, η) at the trace's probe of every frame of an on-demand replay, in
    /// time order as they are read
    pub series: Vec<(f64, f32)>,
}

impl Playback {
    pub fn new(rate: f64, interval: f64, budget_bytes: usize) -> Self {
        Self {
            frames: VecDeque::new(),
            t: 0.0,
            rate,
            paused: false,
            solver: SolverState::Running,
            interval,
            budget: budget_bytes,
            kept: 0,
            changed: true,
            started: Instant::now(),
            newest_wall: 0.0,
            on_demand: None,
            series: Vec::new(),
        }
    }

    /// Play the frames at `times` (s) of a file, asking for them on `requests`: every
    /// frame is there from the start, so nothing waits for a solver.
    pub fn read_on_demand(mut self, times: Vec<f64>, requests: Sender<Vec<usize>>) -> Self {
        self.t = times.first().copied().unwrap_or(0.0);
        self.solver = SolverState::Finished {
            steps: times.len(),
            wall: 0.0,
        };
        self.on_demand = Some(OnDemand {
            times,
            requests,
            asked: Vec::new(),
            grew: None,
        });
        self
    }

    /// Keep a snapshot from the solver (or a streamed replay), dropping the oldest
    /// beyond the budget but always keeping two to interpolate between.
    fn keep(&mut self, snapshot: Snapshot) {
        self.kept += snapshot.bytes();
        self.frames.push_back(snapshot);
        while self.frames.len() > 2 && self.kept > self.budget {
            let oldest = self.frames.pop_front().expect("more than two snapshots");
            self.kept -= oldest.bytes();
        }
    }

    /// Hold a frame of an on-demand replay, in time order, dropping the frames
    /// farthest from the shown time beyond [`HELD`].
    fn hold(&mut self, snapshot: Snapshot) {
        let at = self.frames.partition_point(|s| s.t < snapshot.t);
        if self.frames.get(at).is_some_and(|s| s.t == snapshot.t) {
            return;
        }
        self.frames.insert(at, snapshot);
        let shown = self.t;
        let far = |s: Option<&Snapshot>| s.map_or(0.0, |s| (s.t - shown).abs());
        while self.frames.len() > HELD {
            if far(self.frames.front()) >= far(self.frames.back()) {
                self.frames.pop_front();
            } else {
                self.frames.pop_back();
            }
        }
        self.changed = true;
    }

    /// Model time of the first frame there is: held, or in the file read on demand.
    pub fn oldest(&self) -> Option<f64> {
        match &self.on_demand {
            Some(file) => file.times.first().copied(),
            None => self.frames.front().map(|s| s.t),
        }
    }

    /// Model time of the last frame there is.
    pub fn newest(&self) -> Option<f64> {
        match &self.on_demand {
            Some(file) => file.times.last().copied(),
            None => self.frames.back().map(|s| s.t),
        }
    }

    /// Wall time (s) since the viewer started.
    pub fn started_secs(&self) -> f64 {
        self.started.elapsed().as_secs_f64()
    }

    /// Model seconds the solver advances per second of wall time.
    pub fn solver_speed(&self) -> f64 {
        self.newest().unwrap_or(0.0) / self.newest_wall.max(1e-9)
    }

    /// Bytes held in snapshots.
    pub fn bytes(&self) -> usize {
        self.frames.iter().map(Snapshot::bytes).sum()
    }

    /// Whether the clock is held at the newest snapshot waiting for the solver, or
    /// for the run writing a file read on demand.
    pub fn waiting(&self) -> bool {
        !self.paused
            && (matches!(self.solver, SolverState::Running) || self.growing())
            && self.newest().is_none_or(|t| self.t >= t)
    }

    /// Whether a file read on demand grew within the last [`GROWING_FOR`].
    pub fn growing(&self) -> bool {
        self.on_demand
            .as_ref()
            .and_then(|file| file.grew)
            .is_some_and(|wall| self.started_secs() - wall < GROWING_FOR)
    }

    /// The snapshots around `self.t` and the weight of the later one.
    pub fn bracket(&self) -> Option<(&Snapshot, &Snapshot, f32)> {
        let later = self.frames.partition_point(|s| s.t < self.t);
        let later = later.min(self.frames.len().checked_sub(1)?);
        let earlier = later.saturating_sub(1);
        let (a, b) = (&self.frames[earlier], &self.frames[later]);
        let w = if b.t > a.t {
            ((self.t - a.t) / (b.t - a.t)).clamp(0.0, 1.0)
        } else {
            1.0
        };
        Some((a, b, w as f32))
    }

    fn clamp(&mut self) {
        if let (Some(lo), Some(hi)) = (self.oldest(), self.newest()) {
            self.t = self.t.clamp(lo, hi);
        }
    }
}

/// The receiving end of the solver's channel (a `Receiver` is `Send` but not `Sync`),
/// and the live solver's stop flag.
#[derive(Resource)]
pub struct SolverChannel {
    pub messages: Mutex<Receiver<SolverMessage>>,
    /// Set on leaving the run: the solver ends after the step it is on. A replay's
    /// reading threads have none: they end once the channel is gone.
    pub stop: Option<Arc<AtomicBool>>,
}

pub struct PlaybackPlugin;

impl Plugin for PlaybackPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            PreUpdate,
            (receive, keys, advance, request).chain().in_set(Viewing),
        )
        .run_resource::<Playback>()
        .run_resource::<Source>()
        .run_resource::<SolverChannel>()
        .add_systems(OnExit(Screen::Viewer), stop);
    }
}

/// Leaving the run stops its solver.
fn stop(channel: Option<Res<SolverChannel>>) {
    if let Some(stop) = channel.as_ref().and_then(|c| c.stop.as_ref()) {
        stop.store(true, Ordering::Relaxed);
    }
}

fn receive(channel: Res<SolverChannel>, mut playback: ResMut<Playback>) {
    let rx = channel.messages.lock().expect("solver channel");
    loop {
        match rx.try_recv() {
            Ok(SolverMessage::Snapshot(snapshot)) => {
                playback.keep(*snapshot);
                playback.newest_wall = playback.started.elapsed().as_secs_f64();
                playback.clamp();
                playback.changed = true;
            }
            Ok(SolverMessage::Frame(snapshot)) => playback.hold(*snapshot),
            Ok(SolverMessage::Series(samples)) => playback.series.extend(samples),
            Ok(SolverMessage::Grown(times)) => {
                let wall = playback.started.elapsed().as_secs_f64();
                if let Some(file) = playback.on_demand.as_mut() {
                    file.times.extend(times);
                    file.grew = Some(wall);
                    playback.changed = true;
                }
            }
            Ok(SolverMessage::Finished { steps, wall, error }) => {
                playback.solver = match error {
                    None => SolverState::Finished { steps, wall },
                    Some(e) => SolverState::Failed(e),
                };
            }
            Err(TryRecvError::Empty) => break,
            Err(TryRecvError::Disconnected) => {
                if matches!(playback.solver, SolverState::Running) {
                    playback.solver = SolverState::Failed("the solver thread stopped".into());
                }
                break;
            }
        }
    }
}

fn keys(mut acts: MessageReader<Act>, mut playback: ResMut<Playback>) {
    for &Act(action) in acts.read() {
        let step = 10.0 * playback.interval;
        let seek = match action {
            Action::Pause => {
                playback.paused = !playback.paused;
                None
            }
            Action::Slower => {
                playback.rate = (playback.rate / 2.0).max(1.0);
                None
            }
            Action::Faster => {
                playback.rate = (playback.rate * 2.0).min(1e5);
                None
            }
            Action::SeekBack => Some(playback.t - step),
            Action::SeekForward => Some(playback.t + step),
            Action::First => playback.oldest(),
            Action::Last => playback.newest(),
            _ => None,
        };
        if let Some(t) = seek {
            playback.t = t;
            playback.clamp();
            playback.changed = true;
        }
    }
}

fn advance(time: Res<Time<Real>>, mut playback: ResMut<Playback>) {
    if playback.paused {
        return;
    }
    let before = playback.t;
    playback.t += playback.rate * time.delta_secs_f64();
    playback.clamp();
    if playback.t != before {
        playback.changed = true;
    }
}

/// The frames of an on-demand replay that the shown time needs and the playback does
/// not hold: the two around it and the next one. A new list is sent only when it
/// needs a frame the last one did not ask for.
fn request(mut playback: ResMut<Playback>) {
    let t = playback.t;
    let Some(file) = &playback.on_demand else {
        return;
    };
    let held = |i: usize| {
        let ti = file.times[i];
        let at = playback.frames.partition_point(|s| s.t < ti);
        playback.frames.get(at).is_some_and(|s| s.t == ti)
    };
    let missing: Vec<usize> = wanted(&file.times, t)
        .into_iter()
        .filter(|&i| !held(i))
        .collect();
    if missing.iter().any(|i| !file.asked.contains(i)) {
        // The reading thread has stopped if this fails; it reports why itself
        let _ = file.requests.send(missing.clone());
        playback.on_demand.as_mut().unwrap().asked = missing;
    }
}

/// The frames of `times` the model time `t` needs: the two around it and the next.
fn wanted(times: &[f64], t: f64) -> Vec<usize> {
    let Some(last) = times.len().checked_sub(1) else {
        return Vec::new();
    };
    let later = times.partition_point(|&s| s < t).min(last);
    let mut wanted = vec![later.saturating_sub(1), later, (later + 1).min(last)];
    wanted.dedup();
    wanted
}

/// Clears the change flag once every system has seen it.
pub fn settle(mut playback: ResMut<Playback>) {
    playback.changed = false;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc::channel;

    fn frame(t: f64) -> Snapshot {
        Snapshot {
            t,
            eta: vec![t as f32; 4],
            u: vec![0.0; 4],
            v: vec![0.0; 4],
            particles: Default::default(),
            layers: None,
            waves: None,
        }
    }

    #[test]
    fn the_budget_counts_the_bytes_held() {
        // Snapshots of 48 bytes, then larger as particles join them
        let larger = |t: f64, values: usize| {
            let mut s = frame(t);
            for field in [&mut s.eta, &mut s.u, &mut s.v] {
                field.resize(values, 0.0);
            }
            s
        };
        let mut playback = Playback::new(1.0, 60.0, 200);
        for t in [0.0, 60.0, 120.0, 180.0] {
            playback.keep(frame(t));
        }
        assert_eq!(playback.frames.len(), 4);
        // 72 bytes: the two oldest go, where a count by the newest's size kept two
        playback.keep(larger(240.0, 6));
        let kept: Vec<f64> = playback.frames.iter().map(|s| s.t).collect();
        assert_eq!(kept, [120.0, 180.0, 240.0]);
        assert_eq!(playback.bytes(), 168);
        // One snapshot over the whole budget: two are kept to interpolate between
        playback.keep(larger(300.0, 100));
        assert_eq!(playback.frames.len(), 2);
        assert_eq!(playback.bytes(), playback.kept);
    }

    #[test]
    fn the_frames_around_the_shown_time_and_the_next_are_wanted() {
        let times = [0.0, 600.0, 1200.0, 1800.0];
        assert_eq!(wanted(&times, 0.0), [0, 1]);
        assert_eq!(wanted(&times, 300.0), [0, 1, 2]);
        assert_eq!(wanted(&times, 600.0), [0, 1, 2]);
        assert_eq!(wanted(&times, 1500.0), [2, 3]);
        assert_eq!(wanted(&times, 1800.0), [2, 3]);
        assert!(wanted(&[], 0.0).is_empty());
    }

    #[test]
    fn an_on_demand_replay_holds_the_frames_nearest_the_shown_time() {
        let times: Vec<f64> = (0..10).map(|i| 600.0 * i as f64).collect();
        let (requests, _asked) = channel();
        let mut playback = Playback::new(1.0, 600.0, 0).read_on_demand(times, requests);
        assert_eq!(
            (playback.oldest(), playback.newest()),
            (Some(0.0), Some(5400.0))
        );
        assert!(!playback.waiting());
        playback.t = 3000.0;
        for t in [0.0, 600.0, 3000.0, 3600.0, 2400.0, 4200.0, 3600.0] {
            playback.hold(frame(t));
        }
        let held: Vec<f64> = playback.frames.iter().map(|s| s.t).collect();
        assert_eq!(held, [2400.0, 3000.0, 3600.0, 4200.0]);
        let (a, b, w) = playback.bracket().unwrap();
        assert_eq!((a.t, b.t, w), (2400.0, 3000.0, 1.0));
        playback.t = 3300.0;
        let (a, b, w) = playback.bracket().unwrap();
        assert_eq!((a.t, b.t, w), (3000.0, 3600.0, 0.5));
    }
}

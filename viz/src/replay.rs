//! A finished run played back from its output instead of the solver.
//!
//! Two kinds of output replay:
//! - A snapshot file (`dg_rs::io::SnapshotWriter`, `.dgsnap`), as
//!   `examples/froya_real_data.rs snapshot_minutes=N` writes it: f32 η, u, v per node,
//!   with the mesh, the bed, the clock and the stations in its header; a 3D file
//!   (`examples/farm_3d.rs snapshot=`, or a live 3D run saved by the viewer) adds the
//!   σ-grid and u, v, T on every level, which become the snapshots' [`Layers`]. The
//!   scenario is built from the file alone ([`Scenario::from_snapshot`]: mesh, bed,
//!   cages, σ-grid), so the run's domain data and builder are not needed.
//!   The particles, if the run had them, are in a particle file beside it
//!   ([`particle_path`], `dg_rs::io::ParticleFileWriter`), and are attached to the
//!   frames of the same time. A [`SnapshotSink`] writes both from the viewer's own
//!   snapshots (`--save-snapshot`, live or replayed).
//! - A directory of VTU frames (`froya_NNNN.vtu`, `dg_rs::io::write_vtk_swe`): every
//!   element's nodes in the solver's order, with `eta`, `u`, `v`, `bathymetry` and the
//!   model time (`TimeValue`), in ASCII. These carry no mesh connectivity, so the
//!   scenario is built as the run was (Frøya), and [`Replay::vtu`] checks the first
//!   frame's nodes and bed against it. `--save-snapshot` writes the frames to a
//!   snapshot file while they are read, a tenth of their size.
//!
//! A snapshot file is read on demand ([`serve`]): a thread of its own reads the
//! frames the playback asks for (the two around the shown time and the next), and
//! first the gauge trace's η at its probe from every frame, a few bytes each
//! (`SnapshotReader::read_eta_into`). Nothing else is held, so 15 days of 10-minute
//! frames (2.3k, ≈ 3 GB at Frøya) play and seek back as hourly ones do. While idle
//! the thread looks for new complete frames every [`FOLLOW`], so a file a run is
//! still writing plays up to its newest frame and goes on as it grows.
//!
//! Otherwise [`spawn`] reads the frames on a thread of its own (VTU a batch at a time
//! in parallel) and sends them down the solver's channel ([`SolverMessage`]), so the
//! playback treats a replay as a solver that runs ahead of the view: the first frame
//! shows at once, and the playback keeps what its memory budget allows. That is the
//! path for VTU frames, and for a snapshot file saved again (`--save-snapshot`), which
//! needs every frame in order. Between frames the field is linear in time
//! ([`crate::field::Field`]), as between the solver's snapshots.

use std::error::Error;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, TryRecvError, channel};
use std::time::{Duration, Instant};

use dg_rs::io::{
    ParticleFileReader, ParticleFileWriter, ParticleFrame, SnapshotError, SnapshotFrame,
    SnapshotLayout, SnapshotReader, SnapshotWriter,
};
use dg_rs::source::CageFootprint;
use dg_rs::time::ModelClock;
use dg_rs::types::ElementIndex;
use dg_rs::waves::SeaStateFields;
use rayon::prelude::*;

use crate::cloud_3d::KINDS;
use crate::field::Probe;
use crate::particles::ParticleSnapshot;
use crate::scenario::Scenario;
use crate::solver::{H_DRY, Layers, Snapshot, SolverMessage, Waves};

/// How far (m) the frames' node positions and bed may lie from the scenario's: the
/// files carry ten significant digits, ≈ 1e-6 m over a domain tens of km wide.
const TOLERANCE: f64 = 1e-3;

/// Where the frames are.
enum Frames {
    /// VTU files, in time order
    Vtu(Vec<PathBuf>),
    /// A snapshot file, its number of frames when opened, where its frames hold the
    /// layers and the waves, and its particles
    Snapshot(
        Box<SnapshotReader>,
        usize,
        FileFields,
        Option<Box<Particles>>,
    ),
}

/// A replay's particle file, and its kinds as the viewer's ([`KINDS`]; none in 2D).
struct Particles {
    reader: ParticleFileReader,
    kinds: Option<Vec<u8>>,
    /// Where it is, to read it again as a run writes it
    path: PathBuf,
}

impl Particles {
    /// The particle frame at the time of each of the field frames at `times` (s), if
    /// it has one.
    fn matching(&self, times: &[f64]) -> Vec<Option<usize>> {
        let r = &self.reader;
        let mut next = 0;
        times
            .iter()
            .map(|&t| {
                while next < r.n_frames() && r.time(next) < t - 1e-6 {
                    next += 1;
                }
                (next < r.n_frames() && (r.time(next) - t).abs() <= 1e-6).then_some(next)
            })
            .collect()
    }
}

/// The particle file beside the snapshot file `path`.
pub fn particle_path(path: &Path) -> PathBuf {
    path.with_extension("dgpart")
}

/// The frames of a finished run.
pub struct Replay {
    frames: Frames,
    /// Model time of the first and the last frame (s)
    pub t_first: f64,
    pub t_last: f64,
    /// Model time between frames (s)
    pub interval: f64,
    /// The run's clock (UTC of model time 0)
    pub clock: Option<ModelClock>,
    /// The scenario's bed at the nodes, for the dry nodes
    bed: Vec<f64>,
    /// Where the frames are also written as they are read
    save: Option<SnapshotSink>,
}

impl Replay {
    /// The `*.vtu` frames in `dir` (in file-name order, which is time order for
    /// zero-padded numbers), checked against `scenario`. The clock is the run's
    /// ([`folder_clock`]), if it wrote one.
    ///
    /// The nodes must be the scenario's. A different bed is the run's own (e.g.
    /// smoothed for 3D, `froya_real_data slopes3d=on`): the scenario takes it.
    pub fn vtu(dir: &Path, scenario: &mut Scenario) -> Result<Self, Box<dyn Error>> {
        let mut files: Vec<PathBuf> = std::fs::read_dir(dir)?
            .filter_map(|entry| entry.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e == "vtu"))
            .collect();
        files.sort();
        let (Some(first), Some(last)) = (files.first(), files.last()) else {
            return Err(format!("no .vtu frames in {}", dir.display()).into());
        };

        // The nodes and the bed of the first frame against the scenario's
        let text = std::fs::read_to_string(first)?;
        let (mesh, ops) = (&scenario.mesh, &scenario.ops);
        let n = mesh.n_elements * ops.n_nodes;
        let points = floats(array(&text, "<Points>")?)?;
        if points.len() != 3 * n {
            return Err(format!(
                "{} has {} points, but the scenario has {n} nodes ({} elements of {}): \
                 the run used another mesh or order",
                first.display(),
                points.len() / 3,
                mesh.n_elements,
                ops.n_nodes
            )
            .into());
        }
        let mut off = 0.0f64;
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let [x, y] = mesh.reference_to_physical(k, ops.nodes_r[i], ops.nodes_s[i]);
                let p = &points[3 * (k.as_usize() * ops.n_nodes + i)..];
                off = off.max((p[0] - x).hypot(p[1] - y));
            }
        }
        if off > TOLERANCE {
            return Err(format!(
                "the nodes of {} lie up to {off:.3} m from the scenario's: the run used another mesh",
                first.display()
            )
            .into());
        }
        let bed = floats(array(&text, "Name=\"bathymetry\"")?)?;
        let bed_off = bed
            .iter()
            .zip(&scenario.bathymetry.data)
            .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
        if bed.len() != n {
            return Err(format!(
                "{} has {} bed values for {n} nodes",
                first.display(),
                bed.len()
            )
            .into());
        }
        if bed_off > TOLERANCE {
            println!(
                "The run's bed differs from the scenario's by up to {bed_off:.1} m (smoothed \
                 for 3D?): showing the run's"
            );
            let mut bathymetry = (*scenario.bathymetry).clone();
            bathymetry.data.copy_from_slice(&bed);
            bathymetry.compute_gradients(&scenario.ops, &scenario.geom);
            scenario.bathymetry = std::sync::Arc::new(bathymetry);
        }

        let t_first = time_value(&text)?;
        let t_last = time_value(&tail(last)?)?;
        let interval = match files.get(1) {
            Some(second) => time_value(&tail(second)?)? - t_first,
            None => 0.0,
        };
        Ok(Self {
            frames: Frames::Vtu(files),
            t_first,
            t_last,
            interval,
            clock: folder_clock(dir),
            bed: scenario.bathymetry.data.clone(),
            save: None,
        })
    }

    /// The snapshot file at `path`, and the scenario of its header.
    pub fn snapshot(path: &Path) -> Result<(Self, Scenario), Box<dyn Error>> {
        let mut reader = SnapshotReader::open(path)?;
        let n = reader.n_frames()?;
        if n == 0 {
            return Err("the file has no frames yet".into());
        }
        let t_first = reader.time(0)?;
        let t_last = reader.time(n - 1)?;
        // The mean spacing: a run's callbacks need not be evenly spaced
        let interval = if n > 1 {
            (t_last - t_first) / (n - 1) as f64
        } else {
            0.0
        };
        let header = reader.header().clone();
        let clock = header.clock;
        // The layered fields the viewer draws: u, v, T and S on every level
        let layers = match &header.levels {
            Some(levels) => {
                let field = |name: &str| {
                    levels
                        .field(name)
                        .ok_or_else(|| format!("the file's levels have no {name} field"))
                };
                Some(LayerFields {
                    n_levels: levels.sigma.n_levels(),
                    u: field("u")?,
                    v: field("v")?,
                    temp: levels.field("temp"),
                    salt: levels.field("salt"),
                })
            }
            None => None,
        };
        // The sea state, if the run had waves: every field of it, or none
        let waves = SeaStateFields::NAMES
            .iter()
            .map(|name| header.surface_field(name))
            .collect::<Option<Vec<usize>>>()
            .map(|at| std::array::from_fn(|i| at[i]));
        let scenario = Scenario::from_snapshot(header);
        // The particles beside it: in 3D of the viewer's kinds, matched by name
        let particles = match ParticleFileReader::open(particle_path(path)) {
            Ok(reader) => {
                let kinds = scenario.three_d.is_some().then(|| {
                    reader
                        .kinds
                        .iter()
                        .map(|name| {
                            let word = name.split_whitespace().next().unwrap_or_default();
                            KINDS
                                .iter()
                                .position(|k| k.name.contains(word))
                                .unwrap_or(0) as u8
                        })
                        .collect()
                });
                Some(Box::new(Particles {
                    reader,
                    kinds,
                    path: particle_path(path),
                }))
            }
            Err(SnapshotError::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => return Err(format!("{}: {e}", particle_path(path).display()).into()),
        };
        let replay = Self {
            frames: Frames::Snapshot(Box::new(reader), n, FileFields { layers, waves }, particles),
            t_first,
            t_last,
            interval,
            clock,
            bed: scenario.bathymetry.data.clone(),
            save: None,
        };
        Ok((replay, scenario))
    }

    /// Also write the frames to the snapshot file `path` as they are read, with their
    /// particles if the replay has them ([`SnapshotSink`]).
    pub fn save_to(&mut self, path: &Path, scenario: &Scenario) -> Result<(), Box<dyn Error>> {
        let particles = self.has_particles();
        self.save = Some(SnapshotSink::create(
            path,
            scenario,
            self.clock.as_ref(),
            particles,
            self.has_waves(),
        )?);
        Ok(())
    }

    pub fn frames(&self) -> usize {
        match &self.frames {
            Frames::Vtu(files) => files.len(),
            Frames::Snapshot(_, n, _, _) => *n,
        }
    }

    /// Whether the replay has particles.
    pub fn has_particles(&self) -> bool {
        matches!(self.frames, Frames::Snapshot(_, _, _, Some(_)))
    }

    /// Whether the replay has the sea state of a run with waves.
    pub fn has_waves(&self) -> bool {
        matches!(&self.frames, Frames::Snapshot(_, _, f, _) if f.waves.is_some())
    }
}

/// Where a snapshot file's frames hold what the viewer draws besides η, u and v.
#[derive(Clone, Copy)]
struct FileFields {
    layers: Option<LayerFields>,
    /// Indices into the frame's surface fields of the sea state, in the order of
    /// `SeaStateFields::NAMES`
    waves: Option<[usize; 7]>,
}

/// Where a 3D snapshot file's frames hold what the viewer's [`Layers`] draw.
#[derive(Clone, Copy)]
struct LayerFields {
    n_levels: usize,
    /// Indices into the frame's layered fields; no temperature is drawn as 0 °C,
    /// no salinity is none
    u: usize,
    v: usize,
    temp: Option<usize>,
    salt: Option<usize>,
}

/// The layered fields the viewer writes: what [`Layers`] holds.
const VIEWER_FIELDS: [&str; 4] = ["u", "v", "temp", "salt"];

/// Where the viewer's snapshots are saved: a snapshot file and, if the run has
/// particles, a particle file beside it ([`particle_path`]).
pub struct SnapshotSink {
    fields: SnapshotWriter,
    particles: Option<ParticleFileWriter>,
    /// The particle frame, reused
    frame: ParticleFrame,
}

impl SnapshotSink {
    /// A snapshot file at `path` for the scenario's runs: its mesh, bed and (in 3D)
    /// σ-grid, `clock`, and as metadata the title, the point of interest, the cages
    /// (`cage=x,y,radius,net_depth,drag_per_length`), the periods of a periodic mesh (`periodic=x,y`) and
    /// the projection of a mesh on a real coast (`projection=local,lat,lon`), so
    /// that [`Scenario::from_snapshot`] rebuilds what the viewer draws. With
    /// `particles`, a particle file of the viewer's kinds beside it (3D: [`KINDS`];
    /// 2D: one kind). With `waves`, the frames carry the sea state
    /// (`SeaStateFields::NAMES`), which every snapshot written must then have.
    pub fn create(
        path: &Path,
        scenario: &Scenario,
        clock: Option<&ModelClock>,
        particles: bool,
        waves: bool,
    ) -> Result<Self, Box<dyn Error>> {
        let title = scenario
            .name
            .split(':')
            .next()
            .unwrap_or_default()
            .to_string();
        let mut metadata = vec![
            ("title", title),
            ("source", "dg-viz --save-snapshot".to_string()),
            (
                "point_of_interest",
                format!("{:.1},{:.1}", scenario.farm[0], scenario.farm[1]),
            ),
        ];
        for cage in &scenario.cages {
            if let CageFootprint::Circle { center, radius } = cage.footprint {
                metadata.push((
                    "cage",
                    format!(
                        "{},{},{radius},{},{}",
                        center[0], center[1], cage.net_depth, cage.drag_per_length
                    ),
                ));
            }
        }
        if let Some([x, y]) = scenario.periodic {
            metadata.push(("periodic", format!("{x},{y}")));
        }
        if let Some(projection) = &scenario.projection {
            metadata.push(("projection", crate::scenario::format_projection(projection)));
        }
        let metadata: Vec<(&str, &str)> = metadata.iter().map(|(k, v)| (*k, v.as_str())).collect();
        let levels = scenario
            .three_d
            .as_ref()
            .map(|three_d| (three_d.sigma.as_ref(), &VIEWER_FIELDS[..]));
        let surface: &[&str] = if waves { &SeaStateFields::NAMES } else { &[] };
        let fields = SnapshotWriter::create_with(
            path,
            &scenario.mesh,
            &scenario.ops,
            &scenario.bathymetry.data,
            clock,
            &metadata,
            H_DRY as f64,
            SnapshotLayout { surface, levels },
        )?;
        let particles = if particles {
            let kinds: Vec<&str> = match scenario.three_d {
                Some(_) => KINDS.iter().map(|k| k.name).collect(),
                None => vec!["particles"],
            };
            Some(ParticleFileWriter::create(
                particle_path(path),
                &kinds,
                &[("source", "dg-viz --save-snapshot")],
            )?)
        } else {
            None
        };
        Ok(Self {
            fields,
            particles,
            frame: ParticleFrame::default(),
        })
    }

    /// Append the viewer's snapshot `s`, and its particles.
    pub fn write(&mut self, s: &Snapshot) -> Result<(), SnapshotError> {
        let waves = s.waves.as_ref().map(Waves::fields);
        let surface: Vec<&[f32]> = waves.iter().flatten().map(|f| f.as_slice()).collect();
        match &s.layers {
            // A frame without salinity (a file that had none) as uniform 0
            Some(l) if l.salt.is_empty() => {
                let none = vec![0.0; l.u.len()];
                self.fields.write_fields(
                    s.t,
                    &s.eta,
                    &s.u,
                    &s.v,
                    &surface,
                    &[&l.u, &l.v, &l.temp, &none],
                )?
            }
            Some(l) => self.fields.write_fields(
                s.t,
                &s.eta,
                &s.u,
                &s.v,
                &surface,
                &[&l.u, &l.v, &l.temp, &l.salt],
            )?,
            None => self
                .fields
                .write_fields(s.t, &s.eta, &s.u, &s.v, &surface, &[])?,
        }
        if let Some(writer) = self.particles.as_mut() {
            let (p, f) = (&s.particles, &mut self.frame);
            f.t = s.t;
            f.xy.clone_from(&p.xy);
            f.z.clone_from(&p.z);
            f.status.clone_from(&p.state);
            f.born.clone_from(&p.born);
            if p.kind.is_empty() {
                f.kind.clear();
                f.kind.resize(p.len(), 0);
            } else {
                f.kind.clone_from(&p.kind);
            }
            writer.write(f)?;
        }
        Ok(())
    }
}

/// Read the frames on a thread of its own (VTU files `threads` at a time); they
/// arrive on the returned channel in time order, then [`SolverMessage::Finished`]
/// with the number of frames as its steps.
pub fn spawn(replay: Replay, threads: usize) -> Receiver<SolverMessage> {
    let (tx, rx) = channel();
    std::thread::Builder::new()
        .name("dg-viz replay".into())
        .spawn(move || {
            let started = Instant::now();
            let n = replay.frames();
            let error = read_all(replay, threads, |snapshot| {
                tx.send(SolverMessage::Snapshot(Box::new(snapshot))).is_ok()
            })
            .err();
            let _ = tx.send(SolverMessage::Finished {
                steps: n,
                wall: started.elapsed().as_secs_f64(),
                error,
            });
        })
        .expect("spawn the replay thread");
    rx
}

/// Hand every frame to `send` in time order (saving it too if asked), until `send`
/// returns false (the viewer has closed).
fn read_all(
    mut replay: Replay,
    threads: usize,
    mut send: impl FnMut(Snapshot) -> bool,
) -> Result<(), String> {
    let mut deliver = |snapshot: Snapshot, sink: &mut Option<SnapshotSink>| {
        if let Some(sink) = sink {
            sink.write(&snapshot)
                .map_err(|e| format!("saving the snapshot file: {e}"))?;
        }
        Ok::<bool, String>(send(snapshot))
    };
    match replay.frames {
        Frames::Vtu(files) => {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .thread_name(|i| format!("dg-viz replay {i}"))
                .build()
                .map_err(|e| e.to_string())?;
            for batch in files.chunks(threads.max(1)) {
                let frames: Vec<_> = pool.install(|| {
                    batch
                        .par_iter()
                        .map(|path| {
                            vtu_frame(path, &replay.bed)
                                .map_err(|e| format!("{}: {e}", path.display()))
                        })
                        .collect()
                });
                for frame in frames {
                    if !deliver(frame?, &mut replay.save)? {
                        return Ok(());
                    }
                }
            }
        }
        Frames::Snapshot(mut reader, n, fields, mut particles) => {
            let times = reader.times().map_err(|e| e.to_string())?;
            let matching = particles.as_ref().map(|p| p.matching(&times[..n]));
            let mut frame = SnapshotFrame::default();
            for i in 0..n {
                let snapshot = file_snapshot(
                    &mut reader,
                    i,
                    &mut frame,
                    fields,
                    particles
                        .as_deref_mut()
                        .zip(matching.as_ref().map(|m| m[i])),
                    &replay.bed,
                )?;
                if !deliver(snapshot, &mut replay.save)? {
                    return Ok(());
                }
            }
        }
    }
    Ok(())
}

/// Frame `i` of a snapshot file as the viewer's snapshot (read into `frame`), with
/// its layers in 3D, its waves and its particles (`particles`: the file and the
/// particle frame of the same time, if any).
fn file_snapshot(
    reader: &mut SnapshotReader,
    i: usize,
    frame: &mut SnapshotFrame,
    fields: FileFields,
    particles: Option<(&mut Particles, Option<usize>)>,
    bed: &[f64],
) -> Result<Snapshot, String> {
    reader
        .read_frame_into(i, frame)
        .map_err(|e| format!("frame {i}: {e}"))?;
    let mut snapshot = snapshot(frame.t, &frame.eta, &frame.u, &frame.v, bed);
    snapshot.waves = fields.waves.map(|at| {
        let mut waves = Waves::default();
        for (out, &j) in waves.fields_mut().into_iter().zip(&at) {
            out.clone_from(&frame.surface[j]);
        }
        waves
    });
    snapshot.layers = fields.layers.map(|f| Layers {
        n_levels: f.n_levels,
        u: frame.layers[f.u].clone(),
        v: frame.layers[f.v].clone(),
        temp: match f.temp {
            Some(t) => frame.layers[t].clone(),
            None => vec![0.0; frame.layers[f.u].len()],
        },
        salt: f.salt.map_or_else(Vec::new, |s| frame.layers[s].clone()),
    });
    if let Some((p, Some(j))) = particles {
        let f = p
            .reader
            .read_frame(j)
            .map_err(|e| format!("particle frame {j}: {e}"))?;
        snapshot.particles = ParticleSnapshot {
            kind: match &p.kinds {
                Some(map) => f.kind.iter().map(|&k| map[k as usize]).collect(),
                None => Vec::new(),
            },
            xy: f.xy,
            z: f.z,
            state: f.status,
            born: f.born,
        };
    }
    Ok(snapshot)
}

/// A snapshot file served on demand ([`serve`]).
pub struct Served {
    /// The frames asked for ([`SolverMessage::Frame`]), the trace's samples
    /// ([`SolverMessage::Series`]), or why reading failed
    pub messages: Receiver<SolverMessage>,
    /// Where the playback asks for frames: each list replaces the last
    pub requests: Sender<Vec<usize>>,
    /// Model time of every frame (s)
    pub times: Vec<f64>,
}

/// Frames of the trace's series read between two looks for requests.
const SERIES_CHUNK: usize = 256;

/// How often an idle reading thread looks for frames a run has appended.
const FOLLOW: Duration = Duration::from_secs(1);

/// Read a snapshot file on demand, on a thread of its own: the frames the playback
/// asks for, as soon as it does; in between, η at `probe` from every frame, a chunk at
/// a time; when idle, the times of frames appended since ([`SolverMessage::Grown`]).
/// A replay of VTU frames, or one being saved, is handed back for [`spawn`].
pub fn serve(replay: Replay, probe: Option<Probe>) -> Result<Served, Box<Replay>> {
    if replay.save.is_some() || !matches!(replay.frames, Frames::Snapshot(..)) {
        return Err(Box::new(replay));
    }
    let Frames::Snapshot(mut reader, n, fields, mut particles) = replay.frames else {
        unreachable!("checked above")
    };
    let (tx, rx) = channel();
    let (requests, asked) = channel::<Vec<usize>>();
    let times = match reader.times() {
        Ok(times) => times[..n].to_vec(),
        Err(e) => {
            let _ = tx.send(failed(format!("reading the frame times: {e}")));
            return Ok(Served {
                messages: rx,
                requests,
                times: Vec::new(),
            });
        }
    };
    let mut matching = particles.as_ref().map(|p| p.matching(&times));
    let bed = replay.bed;
    let mut series_times = times.clone();
    let mut n = n;
    std::thread::Builder::new()
        .name("dg-viz replay".into())
        .spawn(move || {
            let mut frame = SnapshotFrame::default();
            let mut wanted: Vec<usize> = Vec::new();
            let mut sampled = if probe.is_some() { 0 } else { n };
            let mut values = vec![0.0f32; probe.as_ref().map_or(0, |p| p.nodes().len())];
            loop {
                // The newest list replaces the older ones
                loop {
                    match asked.try_recv() {
                        Ok(list) => wanted = list,
                        Err(TryRecvError::Empty) => break,
                        Err(TryRecvError::Disconnected) => return,
                    }
                }
                if !wanted.is_empty() {
                    let i = wanted.remove(0);
                    let snapshot = file_snapshot(
                        &mut reader,
                        i,
                        &mut frame,
                        fields,
                        particles
                            .as_deref_mut()
                            .zip(matching.as_ref().map(|m| m[i])),
                        &bed,
                    );
                    let message = match snapshot {
                        Ok(s) => SolverMessage::Frame(Box::new(s)),
                        Err(e) => failed(e),
                    };
                    if tx.send(message).is_err() {
                        return;
                    }
                } else if let (Some(probe), true) = (&probe, sampled < n) {
                    let end = (sampled + SERIES_CHUNK).min(n);
                    let mut chunk = Vec::with_capacity(end - sampled);
                    for (i, &t) in series_times.iter().enumerate().take(end).skip(sampled) {
                        if let Err(e) = reader.read_eta_into(i, probe.nodes().start, &mut values) {
                            let _ = tx.send(failed(format!("frame {i}: {e}")));
                            return;
                        }
                        chunk.push((t, probe.eval_element(&values)));
                    }
                    sampled = end;
                    if tx.send(SolverMessage::Series(chunk)).is_err() {
                        return;
                    }
                } else {
                    match asked.recv_timeout(FOLLOW) {
                        Ok(list) => wanted = list,
                        Err(RecvTimeoutError::Disconnected) => return,
                        Err(RecvTimeoutError::Timeout) => {
                            let grown = appended(&mut reader, n).and_then(|new| {
                                if !new.is_empty()
                                    && let Some(p) = particles.as_deref_mut()
                                {
                                    p.reader = ParticleFileReader::open(&p.path)
                                        .map_err(|e| format!("{}: {e}", p.path.display()))?;
                                }
                                Ok(new)
                            });
                            match grown {
                                Ok(new) if new.is_empty() => {}
                                Ok(new) => {
                                    n += new.len();
                                    series_times.extend(&new);
                                    matching =
                                        particles.as_ref().map(|p| p.matching(&series_times));
                                    if tx.send(SolverMessage::Grown(new)).is_err() {
                                        return;
                                    }
                                }
                                Err(e) => {
                                    let _ = tx.send(failed(e));
                                    return;
                                }
                            }
                        }
                    }
                }
            }
        })
        .expect("spawn the replay thread");
    Ok(Served {
        messages: rx,
        requests,
        times,
    })
}

/// The times of the frames after the first `n` of the file, if a run has appended
/// any.
fn appended(reader: &mut SnapshotReader, n: usize) -> Result<Vec<f64>, String> {
    let now = reader.n_frames().map_err(|e| e.to_string())?;
    (n..now)
        .map(|i| reader.time(i).map_err(|e| format!("frame {i}: {e}")))
        .collect()
}

/// The message that ends a replay with `error`.
fn failed(error: String) -> SolverMessage {
    SolverMessage::Finished {
        steps: 0,
        wall: 0.0,
        error: Some(error),
    }
}

/// The viewer's snapshot of η and (u, v), the velocity zeroed where the water is no
/// deeper than [`H_DRY`] over the bed, as the solver's snapshots have it.
fn snapshot<T: Copy + Into<f64>>(t: f64, eta: &[T], u: &[T], v: &[T], bed: &[f64]) -> Snapshot {
    let wet = |i: usize| eta[i].into() - bed[i] > H_DRY as f64;
    let velocity = |c: &[T]| {
        (0..bed.len())
            .map(|i| if wet(i) { c[i].into() as f32 } else { 0.0 })
            .collect()
    };
    Snapshot {
        t,
        eta: eta.iter().map(|&x| x.into() as f32).collect(),
        u: velocity(u),
        v: velocity(v),
        particles: ParticleSnapshot::default(),
        layers: None,
        waves: None,
    }
}

/// One VTU frame as a snapshot.
fn vtu_frame(path: &Path, bed: &[f64]) -> Result<Snapshot, Box<dyn Error>> {
    let text = std::fs::read_to_string(path)?;
    let n = bed.len();
    let field = |name: &str| -> Result<Vec<f64>, Box<dyn Error>> {
        let values = floats(array(&text, &format!("Name=\"{name}\""))?)?;
        if values.len() != n {
            return Err(format!("{name} has {} values, not {n}", values.len()).into());
        }
        Ok(values)
    };
    let (eta, u, v) = (field("eta")?, field("u")?, field("v")?);
    Ok(snapshot(time_value(&text)?, &eta, &u, &v, bed))
}

/// The body of a data array: the text from the first `>` after `key` (an attribute of
/// the array's tag, such as `Name="eta"`, or the element holding it, `<Points>`) to
/// the next `</DataArray>`.
fn array<'a>(text: &'a str, key: &str) -> Result<&'a str, String> {
    let missing = || format!("no {key} data array");
    let after = text.find(key).ok_or_else(missing)? + key.len();
    let body = after + text[after..].find('>').ok_or_else(missing)? + 1;
    let end = body + text[body..].find("</DataArray>").ok_or_else(missing)?;
    Ok(&text[body..end])
}

fn floats(body: &str) -> Result<Vec<f64>, String> {
    body.split_ascii_whitespace()
        .map(|s| s.parse().map_err(|_| format!("bad number {s}")))
        .collect()
}

/// The model time of a frame (its `TimeValue` field).
fn time_value(text: &str) -> Result<f64, String> {
    match floats(array(text, "Name=\"TimeValue\"")?)?.as_slice() {
        [t] => Ok(*t),
        _ => Err("TimeValue is not one number".into()),
    }
}

/// The end of a frame file, which holds its `TimeValue`.
fn tail(path: &Path) -> Result<String, Box<dyn Error>> {
    let mut file = File::open(path)?;
    let len = file.metadata()?.len();
    file.seek(SeekFrom::Start(len.saturating_sub(1024)))?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

/// The run's own record of η at the station within [`STATION_REACH`] of `at` (mesh
/// metres), from the `station_*.txt` files `froya_real_data` writes into `dir`: its
/// name and (Unix time, η). The run samples every step at the nearest submerged node,
/// as the trace does, so this is the trace at the solver's resolution (10 min at
/// Frøya) where hourly VTU frames clip the peaks.
pub fn station_record(
    dir: &Path,
    at: [f64; 2],
    projection: &impl dg_rs::io::CoordinateProjection,
) -> Option<StationRecord> {
    let mut best: Option<(f64, StationRecord)> = None;
    for entry in std::fs::read_dir(dir).ok()?.flatten() {
        let path = entry.path();
        let name = path.file_name()?.to_string_lossy().into_owned();
        if !(name.starts_with("station_") && name.ends_with(".txt")) {
            continue;
        }
        let Ok(file) = dg_rs::io::read_tide_gauge_file(&path) else {
            continue;
        };
        let Some(station) = &file.station else {
            continue;
        };
        let (x, y) = projection.geo_to_xy(station.latitude, station.longitude);
        let distance = (x - at[0]).hypot(y - at[1]);
        if distance > STATION_REACH || best.as_ref().is_some_and(|b| b.0 <= distance) {
            continue;
        }
        let series = &file.time_series;
        let record = series
            .times()
            .iter()
            .copied()
            .zip(series.values().iter().copied())
            .collect();
        best = Some((distance, (station.name.clone(), record)));
    }
    best.map(|(_, found)| found)
}

/// A station's name and its record, (Unix time, η).
pub type StationRecord = (String, Vec<(f64, f64)>);

/// How far (m) a run's station may be from the trace's point to stand for it.
const STATION_REACH: f64 = 500.0;

/// The clock of the `froya_real_data` run in `dir`, from the `Clock: t = 0 at … UTC`
/// line of its `clock.txt` (written since 2026-10-20) or of its `run.log`.
pub fn folder_clock(dir: &Path) -> Option<ModelClock> {
    ["clock.txt", "run.log"].iter().find_map(|name| {
        let text = std::fs::read_to_string(dir.join(name)).ok()?;
        let line = text
            .lines()
            .find_map(|l| l.trim().strip_prefix("Clock: t = 0 at "))?;
        ModelClock::parse(line.trim_end_matches(" UTC")).ok()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `froya_real_data`'s `clock.txt` dates its frames; a `run.log` still does.
    #[test]
    fn a_folders_clock_is_its_clock_file_or_its_log() {
        let dir = std::env::temp_dir().join(format!("dg-viz-clock-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert!(folder_clock(&dir).is_none());
        std::fs::write(
            dir.join("run.log"),
            "Frøya\n  Clock: t = 0 at 2025-06-14 00:00:00 UTC\n",
        )
        .unwrap();
        let from_log = folder_clock(&dir).map(|c| c.format(0.0));
        std::fs::write(
            dir.join("clock.txt"),
            "Clock: t = 0 at 2025-06-15 00:00:00 UTC\n",
        )
        .unwrap();
        let from_file = folder_clock(&dir).map(|c| c.format(0.0));
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(from_log.as_deref().map(|d| &d[..10]), Some("2025-06-14"));
        assert_eq!(from_file.as_deref().map(|d| &d[..10]), Some("2025-06-15"));
    }

    /// The trace of an hourly VTU run takes the run's record of its point: the station
    /// within reach, the nearest of several.
    #[test]
    fn the_runs_record_of_the_trace_point_is_found() {
        use dg_rs::io::{CoordinateProjection, LocalProjection, TideGaugeFile};
        use dg_rs::{TideGaugeStation, TimeSeries};
        let dir = std::env::temp_dir().join(format!("dg-viz-station-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let projection = LocalProjection::new(63.87, 8.66);
        let write = |slug: &str, name: &str, at: [f64; 2], eta: f64| {
            let (lat, lon) = projection.xy_to_geo(at[0], at[1]);
            let series = TimeSeries::new(&[1.0e9, 1.0e9 + 600.0], &[eta, -eta]);
            let file = TideGaugeFile::from_time_series(series)
                .with_station(TideGaugeStation::new(name, lon, lat));
            dg_rs::io::write_tide_gauge_file(&dir.join(format!("station_{slug}.txt")), &file)
                .unwrap();
        };
        write("near", "Near", [120.0, -80.0], 0.5);
        write("nearer", "Nearer", [30.0, 40.0], 0.7);
        write("far", "Far", [2000.0, 0.0], 0.9);
        let found = station_record(&dir, [0.0, 0.0], &projection);
        let none = station_record(&dir, [10_000.0, 0.0], &projection);
        std::fs::remove_dir_all(&dir).unwrap();
        let (name, record) = found.unwrap();
        assert_eq!(name, "Nearer");
        assert_eq!(record.len(), 2);
        assert!(
            (record[0].1 - 0.7).abs() < 1e-6 && (record[1].0 - record[0].0 - 600.0).abs() < 1e-6
        );
        assert!(none.is_none());
    }

    #[test]
    fn a_snapshot_file_is_served_frame_by_frame() {
        use dg_rs::mesh::{Mesh2D, PointLocator2D};
        use dg_rs::operators::DGOperators2D;
        use std::time::Duration;

        // η = t everywhere, in five frames ten minutes apart, then two more while it
        // is served, as a run still writing appends them
        let path = std::env::temp_dir().join(format!("dg-viz-serve-{}.dgsnap", std::process::id()));
        let mesh = Mesh2D::uniform_rectangle(0.0, 100.0, 0.0, 100.0, 2, 2);
        let ops = DGOperators2D::new(1);
        let n = mesh.n_elements * ops.n_nodes;
        let mut writer =
            SnapshotWriter::create(&path, &mesh, &ops, &vec![-5.0; n], None, &[], 1e-3).unwrap();
        let mut times: Vec<f64> = (0..5).map(|i| 600.0 * i as f64).collect();
        let mut write = |t: f64| {
            let zero = vec![0.0; n];
            writer
                .write_fields(t, &vec![t as f32; n], &zero, &zero, &[], &[])
                .unwrap();
        };
        for &t in &times {
            write(t);
        }

        let (replay, scenario) = Replay::snapshot(&path).unwrap();
        let probe = Probe::at(
            &PointLocator2D::new(&scenario.mesh),
            &scenario,
            [30.0, 70.0],
        );
        let Ok(served) = serve(replay, probe) else {
            panic!("a snapshot file is served on demand");
        };
        assert_eq!(served.times, times);
        served.requests.send(vec![3, 1]).unwrap();
        let (mut frames, mut series, mut grown) = (Vec::new(), Vec::new(), Vec::new());
        let receive =
            |frames: &mut Vec<f64>, series: &mut Vec<_>, grown: &mut Vec<f64>| match served
                .messages
                .recv_timeout(Duration::from_secs(10))
                .unwrap()
            {
                SolverMessage::Frame(s) => {
                    assert!(s.eta.iter().all(|&eta| eta == s.t as f32));
                    frames.push(s.t);
                }
                SolverMessage::Series(chunk) => series.extend(chunk),
                SolverMessage::Grown(new) => grown.extend(new),
                SolverMessage::Snapshot(_) => panic!("a whole run sent"),
                SolverMessage::Finished { error, .. } => panic!("finished: {error:?}"),
            };
        while frames.len() < 2 || series.len() < times.len() {
            receive(&mut frames, &mut series, &mut grown);
        }
        // The frames in the order asked
        assert_eq!(frames, [1800.0, 600.0]);
        assert!(grown.is_empty());

        // The run appends two frames: their times arrive, then their samples, and they
        // are served as the others
        for t in [3000.0, 3600.0] {
            write(t);
            times.push(t);
        }
        while grown.len() < 2 || series.len() < times.len() {
            receive(&mut frames, &mut series, &mut grown);
        }
        assert_eq!(grown, [3000.0, 3600.0]);
        served.requests.send(vec![6]).unwrap();
        while frames.len() < 3 {
            receive(&mut frames, &mut series, &mut grown);
        }
        assert_eq!(frames[2], 3600.0);
        for ((t, eta), &expected) in series.iter().zip(&times) {
            assert_eq!(*t, expected);
            assert!(
                (eta - expected as f32).abs() < 1e-3 * (1.0 + expected as f32),
                "{eta}"
            );
        }
        drop(served);
        drop(writer);
        std::fs::remove_file(&path).unwrap();
    }

    /// A coupled wave run's sea state comes in as the snapshots' waves, field by
    /// field, and a replay saved again keeps it.
    #[test]
    fn a_runs_waves_replay_and_save_again() {
        use dg_rs::mesh::Mesh2D;
        use dg_rs::operators::DGOperators2D;
        use std::time::Duration;

        let dir = std::env::temp_dir();
        let path = dir.join(format!("dg-viz-waves-{}.dgsnap", std::process::id()));
        let again = dir.join(format!("dg-viz-waves-again-{}.dgsnap", std::process::id()));
        let mesh = Mesh2D::uniform_rectangle(0.0, 100.0, 0.0, 100.0, 2, 2);
        let ops = DGOperators2D::new(1);
        let n = mesh.n_elements * ops.n_nodes;
        // Field f at node i of the frame at t: its own value
        let value = |t: f64, f: usize, i: usize| t as f32 + 10.0 * f as f32 + 0.01 * i as f32;
        let fields = |t: f64| -> Vec<Vec<f32>> {
            (0..7)
                .map(|f| (0..n).map(|i| value(t, f, i)).collect())
                .collect()
        };
        let layout = SnapshotLayout {
            surface: &SeaStateFields::NAMES,
            levels: None,
        };
        let mut writer = SnapshotWriter::create_with(
            &path,
            &mesh,
            &ops,
            &vec![-5.0; n],
            None,
            &[],
            1e-3,
            layout,
        )
        .unwrap();
        let zero = vec![0.0; n];
        for t in [0.0, 600.0] {
            let f = fields(t);
            let surface: Vec<&[f32]> = f.iter().map(Vec::as_slice).collect();
            writer
                .write_fields(t, &zero, &zero, &zero, &surface, &[])
                .unwrap();
        }
        drop(writer);

        let (mut replay, scenario) = Replay::snapshot(&path).unwrap();
        assert!(replay.has_waves());
        replay.save_to(&again, &scenario).unwrap();
        let messages = spawn(replay, 1);
        let mut seen = 0;
        loop {
            match messages.recv_timeout(Duration::from_secs(10)).unwrap() {
                SolverMessage::Snapshot(s) => {
                    let waves = s.waves.expect("the sea state");
                    for (f, field) in waves.fields().into_iter().enumerate() {
                        assert_eq!(*field, fields(s.t)[f], "field {f} at {} s", s.t);
                    }
                    seen += 1;
                }
                SolverMessage::Finished { error, .. } => {
                    assert!(error.is_none(), "{error:?}");
                    break;
                }
                _ => {}
            }
        }
        assert_eq!(seen, 2);
        let mut saved = SnapshotReader::open(&again).unwrap();
        assert_eq!(saved.header().surface, SeaStateFields::NAMES);
        assert_eq!(saved.read_frame(1).unwrap().surface, fields(600.0));
        // A file without the whole sea state has no waves
        let mut writer =
            SnapshotWriter::create(&path, &mesh, &ops, &vec![-5.0; n], None, &[], 1e-3).unwrap();
        writer
            .write_fields(0.0, &zero, &zero, &zero, &[], &[])
            .unwrap();
        drop(writer);
        assert!(!Replay::snapshot(&path).unwrap().0.has_waves());
        std::fs::remove_file(&path).unwrap();
        std::fs::remove_file(&again).unwrap();
    }

    const FRAME: &str = r#"<VTKFile>
      <Points>
        <DataArray type="Float64" NumberOfComponents="3" format="ascii">
          1.0e0 2.0e0 0.0 3.0e0 4.0e0 0.0
        </DataArray>
      </Points>
      <PointData Scalars="h">
        <DataArray type="Float64" Name="u" format="ascii">
          5.0e-1 -2.5e-1
        </DataArray>
        <DataArray type="Float64" Name="eta" format="ascii">
          1.0e-1 -1.0e0
        </DataArray>
      </PointData>
    <FieldData>
      <DataArray type="Float64" Name="TimeValue" NumberOfTuples="1" format="ascii">
        3.6000000000e3
      </DataArray>
    </FieldData>
    </VTKFile>"#;

    #[test]
    fn reads_the_arrays_of_a_frame() {
        assert_eq!(
            floats(array(FRAME, "<Points>").unwrap()).unwrap(),
            [1.0, 2.0, 0.0, 3.0, 4.0, 0.0]
        );
        // The array named, not the next one; `Name="u"` is not matched inside another name
        assert_eq!(
            floats(array(FRAME, "Name=\"u\"").unwrap()).unwrap(),
            [0.5, -0.25]
        );
        assert_eq!(
            floats(array(FRAME, "Name=\"eta\"").unwrap()).unwrap(),
            [0.1, -1.0]
        );
        assert_eq!(time_value(FRAME).unwrap(), 3600.0);
        assert!(array(FRAME, "Name=\"v\"").is_err());
    }
}

//! A dgcom run's own output as the parent of a cut-out of its mesh.
//!
//! A case run on a patch of its mesh (`[mesh.patch]`, [`crate::case::Patch`])
//! nested in its parent model at the cut sees the parent there, not the
//! state the whole run had: the two part within minutes (52 mm in η at
//! Mausund after 10 minutes, `notes/case-files.md` P8.14). [`RunParent`]
//! gives the patch the whole run's own state instead, from the whole run's
//! snapshot file ([`crate::io::SnapshotReader`]), so a patch replays the
//! whole run inside it, up to what the snapshots' interval and precision
//! (`f32`) and the boundary condition leave out.
//!
//! The patch's elements are a subset of the whole mesh's, so its nodes are
//! the whole mesh's nodes: no spatial interpolation. At a node of a face
//! the cut made (an interior face of the whole mesh), the external state is
//! the dropped neighbour's value at the same point, the trace the whole
//! run's flux saw there; at the nodes of the whole mesh's own open faces
//! and across the band inside the patch's open faces, the node's own value.
//! Between frames the state is linear in time.
//!
//! It is a parent's depth-mean series per node ([`ParentSeries2D`]: no ramp,
//! no datum shift, no transport scaling, linear in time), so the
//! [`ExternalStateProvider`] of the patch's characteristic open boundaries
//! (η, ū, v̄) and the parent of its 2D relaxation band
//! ([`NestingRelaxation2D`]), on the CPU and on a device
//! ([`DeviceExternal2D::Parent`], as a parent model's); and the parent of a
//! 3D run's band ([`ParentColumns3D`]: u, v, T and S on the σ-levels, which
//! are the whole run's as the patch is the same case), which a device
//! evaluates on the host.

use std::path::Path;
use std::sync::Arc;

use thiserror::Error;

use crate::boundary::band::boundary_distances;
use crate::boundary::{
    BCContext2D, ColumnContext3D, DeviceExternal2D, ExternalState, ExternalStateProvider,
    NestingRelaxation2D, PARENT_MOMENT, ParentColumn, ParentColumns3D, ParentSeries2D, Supplied,
    parent_external_state,
};
use crate::io::{SnapshotError, SnapshotReader};
use crate::mesh::{BoundaryTag, Mesh2D};
use crate::operators::DGOperators2D;
use crate::source::SpongeProfile;
use crate::types::ElementIndex;
use crate::vertical::SigmaGrid;

/// Errors reading a run's snapshots as a patch's parent.
#[derive(Debug, Error)]
pub enum RunParentError {
    /// The snapshot file could not be read.
    #[error(transparent)]
    Snapshot(#[from] SnapshotError),
    /// The snapshot's mesh is not the one the patch was cut from.
    #[error("the snapshot's mesh is not the patch's whole mesh: {0}")]
    OtherMesh(String),
    /// The snapshots do not cover the patch's run.
    #[error(
        "the snapshots cover {from:.3}–{to:.3} h, the patch runs {start:.3}–{end:.3} h (the whole \
         run must write its snapshots over the patch's window)"
    )]
    Window {
        /// First and last frame (h)
        from: f64,
        to: f64,
        /// The patch's run (h)
        start: f64,
        end: f64,
    },
    /// The patch has no open face to nest.
    #[error("the patch has no open face to nest in the whole run")]
    NoBoundary,
}

/// The whole run's state at a patch's open faces and band (see the module
/// docs).
#[derive(Clone)]
pub struct RunParent {
    inner: Arc<Inner>,
}

struct Inner {
    /// Frame times (model s), ascending
    times: Vec<f64>,
    /// Slot of each flat node of the patch, `u32::MAX` if not sampled
    slot_of_node: Vec<u32>,
    /// Position of each slot, for a lookup without a node index
    positions: Vec<(f64, f64)>,
    /// Relaxation weight of each slot, 1 at the open faces
    weight: Vec<f64>,
    /// Whether each element has a sampled node
    band_elements: Vec<bool>,
    n_slots: usize,
    /// η, ū, v̄ per `[slot][time]`
    depth_mean: Vec<[f32; 3]>,
    /// σ-levels of a 3D file (0 in 2D)
    n_levels: usize,
    /// u, v, T, S per `[time][slot][level]`, bed up
    layers: Vec<[f32; 4]>,
}

/// Linear time weights between two frames.
#[derive(Clone, Copy)]
struct Frames {
    a: usize,
    b: usize,
    w: f64,
}

impl std::fmt::Debug for RunParent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = &self.inner;
        f.debug_struct("RunParent")
            .field("slots", &inner.n_slots)
            .field("frames", &inner.times.len())
            .field("levels", &inner.n_levels)
            .finish_non_exhaustive()
    }
}

impl RunParent {
    /// Read the frames of the whole run's snapshot `path` that cover the
    /// patch's run `(t_start, t_end)` (model s) at the nodes of `mesh`'s
    /// faces tagged one of `tags` and within `band_width` (m) of them,
    /// weighted across the band by `profile`. `mesh` is the patch and
    /// `whole_elements` its elements in the snapshot's mesh (ascending).
    pub fn new(
        path: &Path,
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        whole_elements: &[usize],
        tags: &[BoundaryTag],
        (band_width, profile): (f64, SpongeProfile),
        (t_start, t_end): (f64, f64),
    ) -> Result<Self, RunParentError> {
        let mut reader = SnapshotReader::open(path)?;
        let header = reader.header();
        let whole = &header.mesh;
        let n_nodes = ops.n_nodes;
        if header.order != ops.order {
            return Err(RunParentError::OtherMesh(format!(
                "order {}, the patch's {}",
                header.order, ops.order
            )));
        }
        for (k, &e) in whole_elements.iter().enumerate() {
            let same = e < whole.n_elements
                && whole.element_vertices(ElementIndex::new(e))
                    == mesh.element_vertices(ElementIndex::new(k));
            if !same {
                return Err(RunParentError::OtherMesh(format!(
                    "the patch's element {k} is not the snapshot's element {e} (the snapshot \
                     has {} elements)",
                    whole.n_elements
                )));
            }
        }
        let n_levels = header.n_levels();
        let fields = match &header.levels {
            Some(levels) => {
                let field = |name: &str| {
                    levels.field(name).ok_or_else(|| {
                        RunParentError::OtherMesh(format!("the 3D snapshot has no field {name}"))
                    })
                };
                Some([field("u")?, field("v")?, field("temp")?, field("salt")?])
            }
            None => None,
        };

        // The sampled nodes: the open faces' and the band's, each with the
        // whole mesh's node it reads
        let distance =
            boundary_distances(mesh, ops, tags, band_width).ok_or(RunParentError::NoBoundary)?;
        let position = |m: &Mesh2D, k: usize, i: usize| {
            m.reference_to_physical(ElementIndex::new(k), ops.nodes_r[i], ops.nodes_s[i])
        };
        let mut reads = vec![usize::MAX; distance.len()];
        for (k, &e) in whole_elements.iter().enumerate() {
            for face in 0..4 {
                let open = mesh
                    .boundary_tag(ElementIndex::new(k), face)
                    .is_some_and(|t| tags.contains(&t));
                if !open {
                    continue;
                }
                // A face the cut made: the dropped neighbour's trace
                let neighbour = whole.neighbor(ElementIndex::new(e), face);
                for &i in &ops.face_nodes[face] {
                    let flat = k * n_nodes + i;
                    if reads[flat] != usize::MAX {
                        continue;
                    }
                    reads[flat] = match neighbour {
                        Some(nb) => {
                            let [x, y] = position(mesh, k, i);
                            let j = ops.face_nodes[nb.face]
                                .iter()
                                .copied()
                                .min_by(|&a, &b| {
                                    let d = |j| {
                                        let [px, py] = position(whole, nb.element, j);
                                        (px - x).hypot(py - y)
                                    };
                                    d(a).total_cmp(&d(b))
                                })
                                .expect("a face has nodes");
                            nb.element * n_nodes + j
                        }
                        None => e * n_nodes + i,
                    };
                }
            }
        }
        let mut slot_of_node = vec![u32::MAX; distance.len()];
        let (mut nodes, mut positions, mut weight) = (Vec::new(), Vec::new(), Vec::new());
        let mut band_elements = vec![false; mesh.n_elements];
        for (flat, &d) in distance.iter().enumerate() {
            let w = if d == 0.0 {
                1.0
            } else if d < band_width {
                profile.evaluate(1.0 - d / band_width)
            } else {
                continue;
            };
            let (k, i) = (flat / n_nodes, flat % n_nodes);
            let read = match reads[flat] {
                usize::MAX => whole_elements[k] * n_nodes + i,
                r => r,
            };
            slot_of_node[flat] = nodes.len() as u32;
            nodes.push(read);
            let [x, y] = position(mesh, k, i);
            positions.push((x, y));
            weight.push(w);
            band_elements[k] = true;
        }

        // The frames over the run, one more on either side
        let all = reader.times()?;
        let first = all
            .partition_point(|&t| t <= t_start + 1e-6)
            .saturating_sub(1);
        let last = all
            .partition_point(|&t| t < t_end - 1e-6)
            .min(all.len() - 1);
        let covered = !all.is_empty()
            && all[first] <= t_start + 1e-6
            && all[last] >= t_end - 1e-6
            && last >= first;
        if !covered {
            return Err(RunParentError::Window {
                from: all.first().copied().unwrap_or(f64::NAN) / 3600.0,
                to: all.last().copied().unwrap_or(f64::NAN) / 3600.0,
                start: t_start / 3600.0,
                end: t_end / 3600.0,
            });
        }
        let (n_slots, n_times) = (nodes.len(), last - first + 1);
        let mut times = Vec::with_capacity(n_times);
        let mut depth_mean = vec![[0.0; 3]; n_slots * n_times];
        let mut layers = Vec::with_capacity(n_times * n_slots * n_levels);
        for (i, f) in (first..=last).enumerate() {
            let frame = reader.read_frame(f)?;
            times.push(frame.t);
            for (slot, &p) in nodes.iter().enumerate() {
                depth_mean[slot * n_times + i] = [frame.eta[p], frame.u[p], frame.v[p]];
            }
            if let Some([u, v, t, s]) = fields {
                for &p in &nodes {
                    for l in 0..n_levels {
                        let at = p * n_levels + l;
                        layers.push([
                            frame.layers[u][at],
                            frame.layers[v][at],
                            frame.layers[t][at],
                            frame.layers[s][at],
                        ]);
                    }
                }
            }
        }
        Ok(Self {
            inner: Arc::new(Inner {
                times,
                slot_of_node,
                positions,
                weight,
                band_elements,
                n_slots,
                depth_mean,
                n_levels,
                layers,
            }),
        })
    }

    /// The frames read: their number and their first and last time (s).
    pub fn frames(&self) -> (usize, f64, f64) {
        let t = &self.inner.times;
        (t.len(), t[0], t[t.len() - 1])
    }

    /// The 2D band's relaxation towards the whole run, at rate
    /// `weight / timescale` (s).
    pub fn relaxation(&self, timescale: f64) -> NestingRelaxation2D<RunParent> {
        NestingRelaxation2D::new(self.clone(), timescale)
    }

    /// The frames at time `t` and the second's weight (held outside them).
    fn frames_at(&self, t: f64) -> Frames {
        let times = &self.inner.times;
        if times.len() == 1 {
            return Frames { a: 0, b: 0, w: 0.0 };
        }
        let b = times.partition_point(|&x| x < t).clamp(1, times.len() - 1);
        let a = b - 1;
        let w = ((t - times[a]) / (times[b] - times[a])).clamp(0.0, 1.0);
        Frames { a, b, w }
    }

    /// The slot of the flat node `node`, if sampled.
    fn slot(&self, node: usize) -> Option<usize> {
        match self.inner.slot_of_node.get(node) {
            Some(&s) if s != u32::MAX => Some(s as usize),
            _ => None,
        }
    }
}

impl ParentSeries2D for RunParent {
    fn n_slots(&self) -> usize {
        self.inner.n_slots
    }

    fn n_times(&self) -> usize {
        self.inner.times.len()
    }

    fn slot_series(&self, slot: usize) -> &[[f32; 3]] {
        let n = self.inner.times.len();
        &self.inner.depth_mean[slot * n..(slot + 1) * n]
    }

    fn slot_of_node(&self) -> &[u32] {
        &self.inner.slot_of_node
    }

    fn band_elements(&self) -> &[bool] {
        &self.inner.band_elements
    }

    /// By nodal index, or else the nearest slot's by position.
    fn slot_of(&self, node_index: Option<usize>, (x, y): (f64, f64)) -> usize {
        if let Some(slot) = node_index.and_then(|i| self.slot(i)) {
            return slot;
        }
        let positions = &self.inner.positions;
        (0..self.inner.n_slots)
            .min_by(|&a, &b| {
                let d = |s: usize| (positions[s].0 - x).hypot(positions[s].1 - y);
                d(a).total_cmp(&d(b))
            })
            .expect("a run parent has slots")
    }

    fn band_weight(&self, slot: usize) -> f64 {
        self.inner.weight[slot]
    }

    /// The whole run's bed is the patch's: no transport scaling.
    fn parent_depth(&self, _slot: usize) -> f64 {
        f64::NAN
    }

    fn has_velocity(&self) -> bool {
        true
    }

    fn max_transport_ratio(&self) -> Option<f64> {
        None
    }

    fn reference_level(&self) -> f64 {
        0.0
    }

    /// The two frames around `t`, linear, no ramp (the whole run's own is
    /// in its frames).
    fn moment(&self, t: f64, moment: &mut [f64]) {
        let Frames { a, b, w } = self.frames_at(t);
        moment[..PARENT_MOMENT].fill(0.0);
        moment[0] = 1.0;
        moment[2] = a as f64;
        if a == b {
            moment[1] = 1.0;
            moment[6] = 1.0;
        } else {
            moment[1] = 2.0;
            moment[3] = b as f64;
            moment[6] = 1.0 - w;
            moment[7] = w;
        }
    }

    fn samples(&self) -> *const () {
        Arc::as_ptr(&self.inner).cast()
    }
}

impl ExternalStateProvider for RunParent {
    fn external_state(&self, ctx: &BCContext2D) -> ExternalState {
        let slot = self.slot_of(ctx.node_index, ctx.position);
        let mut moment = [0.0; PARENT_MOMENT];
        self.moment(ctx.time, &mut moment);
        parent_external_state(self.evaluate(slot, &moment, ctx.bathymetry))
    }

    fn device_external(&self) -> Option<DeviceExternal2D<'_>> {
        Some(DeviceExternal2D::Parent(self))
    }
}

impl ParentColumns3D for RunParent {
    fn supplies(&self) -> Supplied {
        Supplied {
            velocity: true,
            tracers: true,
        }
    }

    fn column(&self, ctx: &ColumnContext3D, sigma: &SigmaGrid, out: ParentColumn<'_>) -> bool {
        let inner = &self.inner;
        let nl = inner.n_levels;
        if nl == 0 || sigma.n_levels() != nl {
            return false;
        }
        let Some(slot) = self.slot(ctx.node) else {
            return false;
        };
        let m = self.frames_at(ctx.time);
        let at = |f: usize, l: usize| inner.layers[(f * inner.n_slots + slot) * nl + l];
        for l in 0..nl {
            let (a, b) = (at(m.a, l), at(m.b, l));
            let lerp = |c: usize| (1.0 - m.w) * f64::from(a[c]) + m.w * f64::from(b[c]);
            out.u[l] = lerp(0);
            out.v[l] = lerp(1);
            out.temp[l] = lerp(2);
            out.salt[l] = lerp(3);
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The moment of a time gives the CPU's linear interpolation between the
    /// frames around it, held outside them: what a device evaluates
    /// (`ParentSeries2D::evaluate`) is the two frames' blend, bit for bit.
    #[test]
    fn the_moment_blends_the_two_frames_around_a_time() {
        let times = vec![0.0, 60.0, 120.0];
        // One slot, its (η, ū, v̄) per frame
        let depth_mean = vec![[0.1, 0.2, -0.3], [0.4, -0.1, 0.25], [-0.2, 0.05, 0.0]];
        let parent = RunParent {
            inner: Arc::new(Inner {
                times: times.clone(),
                slot_of_node: vec![0],
                positions: vec![(0.0, 0.0)],
                weight: vec![1.0],
                band_elements: vec![true],
                n_slots: 1,
                depth_mean: depth_mean.clone(),
                n_levels: 0,
                layers: Vec::new(),
            }),
        };
        let mut moment = [0.0; PARENT_MOMENT];
        for (t, (a, b, w)) in [
            (-30.0, (0, 1, 0.0)),
            (0.0, (0, 1, 0.0)),
            (15.0, (0, 1, 0.25)),
            (60.0, (0, 1, 1.0)),
            (100.0, (1, 2, 40.0 / 60.0)),
            (200.0, (1, 2, 1.0)),
        ] {
            parent.moment(t, &mut moment);
            let (eta, velocity) = parent.evaluate(0, &moment, -10.0);
            let (u, v) = velocity.expect("a velocity");
            let lerp = |c: usize| {
                (1.0 - w) * f64::from(depth_mean[a][c]) + w * f64::from(depth_mean[b][c])
            };
            assert_eq!(
                [eta, u, v].map(f64::to_bits),
                [lerp(0), lerp(1), lerp(2)].map(f64::to_bits),
                "t = {t}"
            );
        }
    }
}

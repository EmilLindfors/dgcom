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
//! It is the [`ExternalStateProvider`] of the patch's characteristic open
//! boundaries (η, ū, v̄), the parent of its 2D relaxation band
//! ([`RunParentRelaxation2D`]), and of a 3D run's band ([`ParentColumns3D`]:
//! u, v, T and S on the σ-levels, which are the whole run's as the patch is
//! the same case). It has no device form: a run on a device refuses it.

use std::path::Path;
use std::sync::Arc;

use thiserror::Error;

use crate::boundary::band::boundary_distances;
use crate::boundary::{
    BCContext2D, ColumnContext3D, ExternalState, ExternalStateProvider, ParentColumn,
    ParentColumns3D, Supplied, relaxation_source,
};
use crate::io::{SnapshotError, SnapshotReader};
use crate::mesh::{BoundaryTag, Mesh2D};
use crate::operators::DGOperators2D;
use crate::solver::SWEState2D;
use crate::source::{ElementSources, SourceContext2D, SourceTerm2D, SpongeProfile};
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
    n_nodes: usize,
    n_slots: usize,
    /// η, ū, v̄ per `[time][slot]`
    depth_mean: Vec<[f32; 3]>,
    /// σ-levels of a 3D file (0 in 2D)
    n_levels: usize,
    /// u, v, T, S per `[time][slot][level]`, bed up
    layers: Vec<[f32; 4]>,
}

/// Linear time weights between two frames.
#[derive(Clone, Copy)]
struct Moment {
    a: usize,
    b: usize,
    w: f64,
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
        let n_slots = nodes.len();
        let mut times = Vec::new();
        let mut depth_mean = Vec::with_capacity((last - first + 1) * n_slots);
        let mut layers = Vec::with_capacity((last - first + 1) * n_slots * n_levels);
        for f in first..=last {
            let frame = reader.read_frame(f)?;
            times.push(frame.t);
            depth_mean.extend(
                nodes
                    .iter()
                    .map(|&p| [frame.eta[p], frame.u[p], frame.v[p]]),
            );
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
                n_nodes,
                n_slots,
                depth_mean,
                n_levels,
                layers,
            }),
        })
    }

    /// The number of sampled nodes (the open faces' and the band's).
    pub fn n_slots(&self) -> usize {
        self.inner.n_slots
    }

    /// The frames read: their number and their first and last time (s).
    pub fn frames(&self) -> (usize, f64, f64) {
        let t = &self.inner.times;
        (t.len(), t[0], t[t.len() - 1])
    }

    /// The 2D band's relaxation towards the whole run, at rate
    /// `weight / timescale` (s).
    pub fn relaxation(&self, timescale: f64) -> RunParentRelaxation2D {
        assert!(timescale > 0.0, "relaxation timescale must be positive");
        RunParentRelaxation2D {
            parent: self.clone(),
            rate: 1.0 / timescale,
            wet_depth: 0.05,
        }
    }

    fn moment(&self, t: f64) -> Moment {
        let times = &self.inner.times;
        if times.len() == 1 {
            return Moment { a: 0, b: 0, w: 0.0 };
        }
        let b = times.partition_point(|&x| x < t).clamp(1, times.len() - 1);
        let a = b - 1;
        let w = ((t - times[a]) / (times[b] - times[a])).clamp(0.0, 1.0);
        Moment { a, b, w }
    }

    /// η, ū, v̄ at slot `slot`.
    fn depth_mean(&self, slot: usize, m: Moment) -> (f64, f64, f64) {
        let n = self.inner.n_slots;
        let [a, b] = [m.a, m.b].map(|f| self.inner.depth_mean[f * n + slot]);
        let lerp = |c: usize| (1.0 - m.w) * f64::from(a[c]) + m.w * f64::from(b[c]);
        (lerp(0), lerp(1), lerp(2))
    }

    /// The slot of the flat node `node`, or else the nearest slot to `p`.
    fn slot_of(&self, node: Option<usize>, (x, y): (f64, f64)) -> Option<usize> {
        let inner = &self.inner;
        match node.and_then(|i| inner.slot_of_node.get(i)) {
            Some(&s) if s != u32::MAX => Some(s as usize),
            Some(_) => None,
            None => (0..inner.n_slots).min_by(|&a, &b| {
                let d = |s: usize| {
                    let (px, py) = inner.positions[s];
                    (px - x).hypot(py - y)
                };
                d(a).total_cmp(&d(b))
            }),
        }
    }
}

impl ExternalStateProvider for RunParent {
    fn external_state(&self, ctx: &BCContext2D) -> ExternalState {
        let Some(slot) = self.slot_of(ctx.node_index, ctx.position) else {
            // A node not on a nested face: the interior as it is
            let q = &ctx.interior_state;
            let (u, v) = if q.h > ctx.h_min {
                (q.hu / q.h, q.hv / q.h)
            } else {
                (0.0, 0.0)
            };
            return ExternalState::new(q.h + ctx.bathymetry, u, v);
        };
        let (eta, u, v) = self.depth_mean(slot, self.moment(ctx.time));
        ExternalState::new(eta, u, v)
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
        let Some(slot) = self.slot_of(Some(ctx.node), ctx.position) else {
            return false;
        };
        let m = self.moment(ctx.time);
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

/// The 2D relaxation towards the whole run across a patch's band
/// ([`RunParent::relaxation`]): `γ (q_run − q)`, `γ = weight / τ`, at the
/// sampled nodes deeper than 5 cm in both ([`relaxation_source`]).
pub struct RunParentRelaxation2D {
    parent: RunParent,
    rate: f64,
    wet_depth: f64,
}

impl RunParentRelaxation2D {
    fn relax(&self, slot: usize, m: Moment, q: &SWEState2D, bed: f64) -> SWEState2D {
        relaxation_source(
            self.parent.inner.weight[slot],
            self.rate,
            self.wet_depth,
            false,
            q,
            bed,
            || {
                let (eta, u, v) = self.parent.depth_mean(slot, m);
                (eta, Some((u, v)))
            },
        )
    }

    fn element_sources(
        &self,
        element: &ElementSources<'_>,
        mut add: impl FnMut(usize, SWEState2D),
    ) {
        let inner = &self.parent.inner;
        let k = element.element.as_usize();
        if !inner.band_elements[k] {
            return;
        }
        let m = self.parent.moment(element.time);
        let base = k * inner.n_nodes;
        for i in 0..element.n_nodes() {
            let slot = inner.slot_of_node[base + i];
            if slot == u32::MAX {
                continue;
            }
            let bed = element
                .bathymetry
                .map_or(0.0, |b| b.get(element.element, i));
            add(i, self.relax(slot as usize, m, &element.state(i), bed));
        }
    }
}

impl SourceTerm2D for RunParentRelaxation2D {
    fn evaluate(&self, ctx: &SourceContext2D) -> SWEState2D {
        match self.parent.slot_of(None, ctx.position) {
            Some(slot) => self.relax(
                slot,
                self.parent.moment(ctx.time),
                &ctx.state,
                ctx.bathymetry,
            ),
            None => SWEState2D::zero(),
        }
    }

    fn add_element(
        &self,
        element: &ElementSources<'_>,
        h: &mut [f64],
        hu: &mut [f64],
        hv: &mut [f64],
    ) {
        self.element_sources(element, |i, s| {
            h[i] += s.h;
            hu[i] += s.hu;
            hv[i] += s.hv;
        });
    }

    fn is_node_local(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "run_parent_relaxation_2d"
    }

    fn changes_mass(&self) -> bool {
        true
    }

    fn add_mass_element(&self, element: &ElementSources<'_>, h: &mut [f64]) {
        self.element_sources(element, |i, s| h[i] += s.h);
    }
}

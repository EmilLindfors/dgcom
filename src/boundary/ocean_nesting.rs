//! One-way nesting in a parent ocean model (NorKyst-800, ROMS).
//!
//! [`OceanModelState`] samples the parent's sea surface height ζ and
//! depth-mean velocity `(ū, v̄)` (an [`OceanModelReader`]) at the open-boundary
//! nodes of a mesh, and optionally at every node of a relaxation band inside
//! them. Everything spatial is precomputed at setup: per node a bilinear
//! stencil over the parent's wet points, the rotation from east/north into
//! the mesh axes and the parent depth. At run time a node costs a time
//! stencil (cubic by default: hourly output of a tide, linear, would miss 3 %
//! of the M2 amplitude) and a dozen multiply-adds.
//!
//! It is the [`ExternalStateProvider`] of a
//! [`CharacteristicOBC`](crate::boundary::CharacteristicOBC): the incoming
//! Riemann invariant comes from the parent, the outgoing one from the
//! interior (the nonlinear Flather condition). The parent carries tides,
//! the coastal current and wind-driven flow alike.
//!
//! # Transport
//!
//! The parent's ū is its transport divided by *its* depth. Where the child's
//! bed differs (the parent's is smoothed at 800 m), imposing ū unchanged
//! imposes the transport `ū·D_child`, not the parent's. With transport
//! scaling (default) the boundary velocity is
//!
//! ```text
//! ū_child = ū_parent · D_parent / D_child,    D = ζ + h   (total depths)
//! ```
//!
//! limited to a factor [`NestingOptions::max_transport_ratio`] either way
//! (a 5 m child node under a 60 m parent cell would otherwise get 12× the
//! velocity). [`OceanModelState::blend_bathymetry`] removes the mismatch at
//! the source: it blends the child bed to the parent's across the relaxation
//! band, so the ratio is 1 at the boundary, except at the child's shores,
//! which the parent's grid does not resolve.
//!
//! # Relaxation band
//!
//! The characteristic condition prescribes only the incoming invariant, and
//! a 1D condition partly reflects oblique waves. A flow-relaxation band
//! (Martinsen & Engedahl 1987; ROMS nudging) additionally pulls the state
//! towards the parent over a width `w` inside the boundary:
//!
//! ```text
//! ∂q/∂t = … + γ(d) (q_parent − q),    γ(d) = profile(1 − d/w) / τ
//! ```
//!
//! with `q = (h, hu, hv)`, `d` the distance to the open boundary and `τ` the
//! timescale at the boundary ([`OceanModelState::relaxation`],
//! [`NestingRelaxation2D`]). It is explicit: keep `τ` well above the time
//! step. Relaxing `h` adds or removes mass, as nesting does through the
//! boundary; nodes that are dry in the child or the parent are left alone.
//!
//! # Time base
//!
//! Simulation time `t` is the parent instant `clock.unix(t)`. A time outside
//! the parent file panics instead of freezing the forcing; check a run up
//! front with [`OceanModelState::check_time_coverage`].
//!
//! # Example
//!
//! ```ignore
//! use dg_rs::boundary::{CharacteristicOBC, NestingOptions, OceanModelState};
//! use dg_rs::io::{LocalProjection, OceanModelReader};
//! use dg_rs::time::ModelClock;
//!
//! let reader = Arc::new(OceanModelReader::from_file("norkyst_subset.nc")?);
//! let clock = ModelClock::parse("2025-06-15T00:00:00Z")?;
//! let options = NestingOptions::default().with_band(3000.0);
//! let parent = OceanModelState::new(reader, &mesh, &ops, &projection, BoundaryTag::Open, clock, &options)?;
//! parent.check_time_coverage(0.0, t_end)?;
//! parent.blend_bathymetry(&mut bathymetry, &ops, &geom, 2.0);
//! let band = parent.relaxation(1800.0);     // a SourceTerm2D
//! let bc = CharacteristicOBC::new(parent);
//! ```

use std::sync::Arc;

use thiserror::Error;

use crate::boundary::band::boundary_distances;
use crate::boundary::{
    BCContext2D, ExternalState, ExternalStateProvider, TidalAtlas, TidalAtlasError, tidal_ramp,
};
use crate::io::{
    CoordinateProjection, OceanModelReader, Stencil, TimeInterpolation, TimeStencil, east_axis,
};
use crate::mesh::{Bathymetry2D, BoundaryTag, Mesh2D};
use crate::operators::{DGOperators2D, GeometricFactors2D};
use crate::solver::SWEState2D;
use crate::source::{ElementSources, SourceContext2D, SourceTerm2D, SpongeProfile};
use crate::time::ModelClock;
use crate::types::ElementIndex;

/// Errors setting up nesting.
#[derive(Debug, Error)]
pub enum NestingError {
    /// The parent has no sea surface height.
    #[error("the parent model has no sea surface height")]
    NoSurface,
    /// No mesh face has the boundary tag.
    #[error("no boundary face is tagged {0:?}")]
    NoBoundary(BoundaryTag),
    /// An open-boundary node the parent cannot serve.
    #[error(
        "open-boundary node at (x, y) = ({x:.0}, {y:.0}) m (lat {lat:.4}, lon {lon:.4}) has no wet \
         parent point within {max_snap:.0} m"
    )]
    NotCovered {
        /// Projected position (m)
        x: f64,
        /// Projected position (m)
        y: f64,
        /// Latitude (°)
        lat: f64,
        /// Longitude (°)
        lon: f64,
        /// Allowed snapping distance (m)
        max_snap: f64,
    },
}

/// Options of [`OceanModelState::new`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NestingOptions {
    /// Width (m) of the relaxation band inside the open boundary (0: none)
    pub band_width: f64,
    /// Shape of the relaxation rate across the band (1 at the boundary)
    pub band_profile: SpongeProfile,
    /// Farthest (m) an open-boundary node in a parent land cell may take the
    /// nearest wet parent point
    pub max_snap: f64,
    /// Interpolation between parent snapshots
    pub time_interpolation: TimeInterpolation,
    /// Scale the parent velocity by `D_parent/D_child` (see the module docs)
    pub transport_scaling: bool,
    /// Largest factor transport scaling may change the velocity by
    pub max_transport_ratio: f64,
    /// Added to the parent's ζ (m), e.g. to move it to the child's datum
    pub reference_level: f64,
    /// Ramp the parent state up from rest over this many seconds (a run
    /// started at rest would otherwise be hit by the parent's full ζ and ū);
    /// the reference level ramps with it
    pub ramp: Option<f64>,
}

impl Default for NestingOptions {
    fn default() -> Self {
        Self {
            band_width: 0.0,
            band_profile: SpongeProfile::Cosine,
            max_snap: 2000.0,
            time_interpolation: TimeInterpolation::Cubic,
            transport_scaling: true,
            max_transport_ratio: 3.0,
            reference_level: 0.0,
            ramp: None,
        }
    }
}

impl NestingOptions {
    /// A relaxation band of `width` metres.
    pub fn with_band(mut self, width: f64) -> Self {
        self.band_width = width;
        self
    }

    /// Transport scaling on or off.
    pub fn with_transport_scaling(mut self, enable: bool) -> Self {
        self.transport_scaling = enable;
        self
    }

    /// Time interpolation between parent snapshots.
    pub fn with_time_interpolation(mut self, method: TimeInterpolation) -> Self {
        self.time_interpolation = method;
        self
    }

    /// Add `level` (m) to the parent's ζ.
    pub fn with_reference_level(mut self, level: f64) -> Self {
        self.reference_level = level;
        self
    }

    /// Ramp the parent's ζ and velocity up from rest over `duration` seconds
    /// (smoothstep, as the tidal ramps).
    pub fn with_ramp_up(mut self, duration: f64) -> Self {
        self.ramp = Some(duration);
        self
    }
}

/// A mesh node the parent is sampled at.
#[derive(Clone, Copy, Debug)]
struct NestedNode {
    position: (f64, f64),
    stencil: Stencil,
    /// East in the mesh plane, `(cos θ, sin θ)`
    east: (f64, f64),
    /// Parent still-water depth (m), NaN if unknown
    parent_depth: f64,
    /// Relaxation weight in [0, 1] (1 at the boundary; 0 outside the band)
    weight: f64,
    /// On an open-boundary face
    on_boundary: bool,
}

#[derive(Clone, Debug)]
struct Inner {
    reader: Arc<OceanModelReader>,
    clock: ModelClock,
    options: NestingOptions,
    /// Slot per flat mesh node, `u32::MAX` if not sampled
    slot_of_node: Vec<u32>,
    nodes: Vec<NestedNode>,
    /// Elements with a node in the relaxation band
    band_elements: Vec<bool>,
    n_nodes: usize,
    snapped: usize,
    /// Parent `(ζ, u, v)` per slot and snapshot, `[slot][time][3]`, the
    /// velocity in the mesh axes: interpolated in space once, so a node
    /// costs only the time stencil at run time
    series: Vec<f32>,
    has_velocity: bool,
}

/// Parent ocean model at the open boundary (and relaxation band) of a mesh;
/// see the module docs. Cheap to clone.
#[derive(Clone, Debug)]
pub struct OceanModelState {
    inner: Arc<Inner>,
}

impl OceanModelState {
    /// Sample `reader` at the nodes of the faces tagged `tag` and, with
    /// `options.band_width > 0`, at every node within that distance of them.
    ///
    /// Mesh coordinates map to longitude/latitude with `projection`,
    /// simulation time to UTC with `clock`. Every open-boundary node must
    /// lie in a parent cell with a wet corner or within `options.max_snap`
    /// of a wet parent point; band nodes the parent does not cover are not
    /// relaxed.
    pub fn new<P: CoordinateProjection>(
        reader: Arc<OceanModelReader>,
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        projection: &P,
        tag: BoundaryTag,
        clock: ModelClock,
        options: &NestingOptions,
    ) -> Result<Self, NestingError> {
        if !reader.has_ssh() {
            return Err(NestingError::NoSurface);
        }
        let n_nodes = ops.n_nodes;
        let position = |flat: usize| {
            let (k, i) = (flat / n_nodes, flat % n_nodes);
            let [x, y] =
                mesh.reference_to_physical(ElementIndex::new(k), ops.nodes_r[i], ops.nodes_s[i]);
            (x, y)
        };

        // Distance to the open boundary of every node within the band
        let distance = boundary_distances(mesh, ops, &[tag], options.band_width)
            .ok_or(NestingError::NoBoundary(tag))?;

        let mut slot_of_node = vec![u32::MAX; distance.len()];
        let mut nodes = Vec::new();
        let mut band_elements = vec![false; mesh.n_elements];
        let mut snapped = 0;
        for (flat, &d) in distance.iter().enumerate() {
            let on_boundary = d == 0.0;
            if !(on_boundary || d < options.band_width) {
                continue;
            }
            let (x, y) = position(flat);
            let (lat, lon) = projection.xy_to_geo(x, y);
            let stencil = match reader.stencil(lon, lat) {
                Some((_, s)) => Some(s),
                None => reader
                    .grid()
                    .nearest(lon, lat, options.max_snap, |k| reader.is_wet(k))
                    .map(|(k, _)| {
                        snapped += on_boundary as usize;
                        Stencil::point(k)
                    }),
            };
            let Some(stencil) = stencil else {
                if on_boundary {
                    return Err(NestingError::NotCovered {
                        x,
                        y,
                        lat,
                        lon,
                        max_snap: options.max_snap,
                    });
                }
                continue;
            };
            let parent_depth = reader.depth().map_or(f64::NAN, |h| {
                stencil
                    .idx
                    .iter()
                    .zip(&stencil.w)
                    .map(|(&k, w)| w * h[k as usize])
                    .sum()
            });
            let weight = if on_boundary {
                1.0
            } else {
                options.band_profile.evaluate(1.0 - d / options.band_width)
            };
            slot_of_node[flat] = nodes.len() as u32;
            if weight > 0.0 {
                band_elements[flat / n_nodes] = true;
            }
            nodes.push(NestedNode {
                position: (x, y),
                stencil,
                east: east_axis(projection, lat, lon),
                parent_depth,
                weight,
                on_boundary,
            });
        }

        // (ζ, u, v) per slot and snapshot, the velocity rotated to the mesh
        let n_times = reader.n_times();
        let has_velocity = reader.has_currents();
        let mut series = Vec::with_capacity(nodes.len() * n_times * 3);
        for node in &nodes {
            let space = node
                .stencil
                .idx
                .iter()
                .map(|&k| k as usize)
                .zip(node.stencil.w);
            for t in 0..n_times {
                let at = TimeStencil {
                    idx: [t, 0, 0, 0],
                    w: [1.0, 0.0, 0.0, 0.0],
                    len: 1,
                };
                let zeta = reader
                    .ssh()
                    .expect("checked")
                    .interpolate(&at, space.clone());
                let (u, v) = match (reader.u(), reader.v()) {
                    (Some(u), Some(v)) => {
                        let e = u.interpolate(&at, space.clone());
                        let n = v.interpolate(&at, space.clone());
                        let (c, s) = node.east;
                        (e * c - n * s, e * s + n * c)
                    }
                    _ => (0.0, 0.0),
                };
                series.extend([zeta as f32, u as f32, v as f32]);
            }
        }

        Ok(Self {
            inner: Arc::new(Inner {
                reader,
                clock,
                options: *options,
                slot_of_node,
                nodes,
                band_elements,
                n_nodes,
                snapped,
                series,
                has_velocity,
            }),
        })
    }

    /// A clock whose `t = 0` is the first snapshot of `reader`.
    pub fn first_snapshot(reader: &OceanModelReader) -> ModelClock {
        ModelClock::new(reader.times()[0])
    }

    /// The model clock.
    pub fn clock(&self) -> &ModelClock {
        &self.inner.clock
    }

    /// The parent fields.
    pub fn reader(&self) -> &OceanModelReader {
        &self.inner.reader
    }

    /// Number of forced open-boundary nodes.
    pub fn n_boundary_nodes(&self) -> usize {
        self.inner.nodes.iter().filter(|n| n.on_boundary).count()
    }

    /// Number of relaxed nodes inside the boundary.
    pub fn n_band_nodes(&self) -> usize {
        self.inner
            .nodes
            .iter()
            .filter(|n| !n.on_boundary && n.weight > 0.0)
            .count()
    }

    /// Open-boundary nodes that took the nearest wet parent point (their
    /// parent cell had no wet corner).
    pub fn n_snapped(&self) -> usize {
        self.inner.snapped
    }

    /// Check that simulation times `[t_start, t_end]` lie within the parent
    /// file, so a run cannot hit the out-of-range panic.
    pub fn check_time_coverage(&self, t_start: f64, t_end: f64) -> Result<(), String> {
        for t in [t_start, t_end] {
            if !self.inner.reader.covers_time(self.inner.clock.unix(t)) {
                return Err(self.coverage_message(t));
            }
        }
        Ok(())
    }

    fn coverage_message(&self, t: f64) -> String {
        let (t0, t1) = self.simulation_time_range();
        format!(
            "OceanModelState: simulation time {t} s is outside the parent file, which covers \
             simulation times [{t0}, {t1}] s (epoch {})",
            self.inner.clock.format(0.0)
        )
    }

    /// Time range of the parent file in Unix seconds.
    pub fn time_range(&self) -> (f64, f64) {
        self.inner
            .reader
            .time_range()
            .expect("at least one snapshot")
    }

    /// Time range of the parent file in simulation time.
    pub fn simulation_time_range(&self) -> (f64, f64) {
        let (t0, t1) = self.time_range();
        (
            self.inner.clock.model_time(t0),
            self.inner.clock.model_time(t1),
        )
    }

    /// Bounding box of the parent grid, `(min_lon, min_lat, max_lon, max_lat)`.
    pub fn spatial_bounds(&self) -> (f64, f64, f64, f64) {
        self.inner.reader.bbox()
    }

    /// Slot of the node of `ctx`: by nodal index, or else the nearest
    /// open-boundary node by position.
    fn slot(&self, node_index: Option<usize>, (x, y): (f64, f64)) -> usize {
        if let Some(&slot) = node_index
            .and_then(|i| self.inner.slot_of_node.get(i))
            .filter(|&&s| s != u32::MAX)
        {
            return slot as usize;
        }
        self.inner
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.on_boundary)
            .min_by(|a, b| {
                let da = (a.1.position.0 - x).hypot(a.1.position.1 - y);
                let db = (b.1.position.0 - x).hypot(b.1.position.1 - y);
                da.total_cmp(&db)
            })
            .map(|(i, _)| i)
            .expect("OceanModelState has boundary nodes")
    }

    /// Snapshot weights and ramp at simulation time `t`.
    fn moment(&self, t: f64) -> Moment {
        let inner = &self.inner;
        let time = inner
            .reader
            .time_stencil(inner.clock.unix(t), inner.options.time_interpolation)
            // Out-of-range times are a configuration error. Falling back (or
            // clamping to the nearest snapshot) would silently freeze the
            // forcing.
            .unwrap_or_else(|| panic!("{}", self.coverage_message(t)));
        Moment {
            time,
            ramp: tidal_ramp(t, inner.options.ramp),
        }
    }

    /// Parent `(η, u, v)` at slot `slot` in the mesh axes, over a child bed
    /// at `bed` (for transport scaling); velocity `None` if the parent has
    /// none.
    fn evaluate(&self, slot: usize, moment: &Moment, bed: f64) -> (f64, Option<(f64, f64)>) {
        let inner = &self.inner;
        let node = &inner.nodes[slot];
        let series = &inner.series[slot * inner.reader.n_times() * 3..];
        let mut sum = [0.0; 3];
        for (t, w) in moment.time.terms() {
            for (q, s) in sum.iter_mut().enumerate() {
                *s += w * series[3 * t + q] as f64;
            }
        }
        let [zeta, u, v] = sum.map(|x| moment.ramp * x);
        // The datum shift ramps with ζ: the run starts at rest at 0
        let eta = zeta + moment.ramp * inner.options.reference_level;
        if !inner.has_velocity {
            return (eta, None);
        }
        let (mut um, mut vm) = (u, v);
        if inner.options.transport_scaling && node.parent_depth.is_finite() {
            let (d_parent, d_child) = (zeta + node.parent_depth, eta - bed);
            if d_parent > 0.0 && d_child > 0.0 {
                let max = inner.options.max_transport_ratio;
                let ratio = (d_parent / d_child).clamp(1.0 / max, max);
                um *= ratio;
                vm *= ratio;
            }
        }
        (eta, Some((um, vm)))
    }

    /// Relaxation band source term with timescale `timescale` (s) at the
    /// boundary; see the module docs. Needs `NestingOptions::band_width > 0`
    /// for anything beyond the boundary nodes themselves.
    pub fn relaxation(&self, timescale: f64) -> NestingRelaxation2D {
        assert!(timescale > 0.0, "relaxation timescale must be positive");
        NestingRelaxation2D {
            parent: self.clone(),
            rate: 1.0 / timescale,
            momentum_only: false,
            wet_depth: 0.05,
        }
    }

    /// Add the tide of `correction` to the parent at every forced and band
    /// node and snapshot: ζ, ū and v̄ of the parent become parent +
    /// correction. With `correction = corrected.difference(&raw)`, where
    /// `raw` is a harmonic atlas of the parent itself and `corrected` the
    /// same atlas with some constituents fixed (e.g. N2 re-inferred from a
    /// gauge, [`TidalAtlas::infer`]), the parent keeps its residual (coastal
    /// current, surge) and gets the corrected tides. Band nodes are
    /// corrected too, so the relaxation does not pull the tide back.
    ///
    /// The correction is interpolated like boundary tides
    /// ([`TidalAtlas::tides_at`]), with the nodal `f`, `u` at the middle of
    /// a run of `duration` on this state's clock; every node must lie within
    /// `coverage_radius` (m) of a point of `correction`.
    pub fn with_tidal_correction<P: CoordinateProjection>(
        self,
        correction: &TidalAtlas,
        projection: &P,
        duration: f64,
        coverage_radius: f64,
    ) -> Result<Self, TidalAtlasError> {
        let inner = &self.inner;
        let positions: Vec<(f64, f64)> = inner.nodes.iter().map(|n| n.position).collect();
        let tides = correction.tides_at(
            &positions,
            projection,
            &inner.clock,
            duration,
            coverage_radius,
        )?;
        let n_times = inner.reader.n_times();
        let mut series = inner.series.clone();
        for (slot, node_series) in series.chunks_exact_mut(3 * n_times).enumerate() {
            for (snapshot, q) in node_series.as_chunks_mut::<3>().0.iter_mut().enumerate() {
                let t = inner.clock.model_time(inner.reader.times()[snapshot]);
                let (eta, u, v) = tides.evaluate(slot, t);
                for (value, delta) in q.iter_mut().zip([eta, u, v]) {
                    *value = (*value as f64 + delta) as f32;
                }
            }
        }
        Ok(Self {
            inner: Arc::new(Inner {
                series,
                ..(**inner).clone()
            }),
        })
    }

    /// Blend the child bed towards the parent's across the relaxation band:
    /// `B ← (1 − w) B + w (−h_parent)` with the band weight `w` (1 on the
    /// boundary), at wet child nodes where the parent depth is known; then
    /// recompute the bed gradients. Coincident nodes of neighbouring
    /// elements get the same value, so a continuous bed stays continuous.
    ///
    /// Shores keep their bed: the nodes of every element with a node
    /// shallower than `shore_depth` (m; take the depth of the lowest water,
    /// below which the shore dries), and those coincident with them. A parent
    /// grid of hundreds of metres has no skerries or shore, and blending its
    /// smooth depth into a shore element deepened a 0.4 m node beside dry land
    /// to 32 m, half-way across a 3 km band at Mausund: the cliff carried
    /// 3.9 m/s and η from −1.0 to +4.2 m within three minutes of the start
    /// (TODO P1.5). 0 blends every wet node.
    ///
    /// Returns the number of nodes changed.
    pub fn blend_bathymetry(
        &self,
        bathymetry: &mut Bathymetry2D,
        ops: &DGOperators2D,
        geom: &GeometricFactors2D,
        shore_depth: f64,
    ) -> usize {
        // Nodes by position to the millimetre: coincident nodes of
        // neighbouring elements share a position
        let at = |(x, y): (f64, f64)| ((x * 1e3).round() as i64, (y * 1e3).round() as i64);
        let n = ops.n_nodes;
        let shore: std::collections::HashSet<(i64, i64)> = self
            .inner
            .slot_of_node
            .iter()
            .enumerate()
            .filter_map(|(flat, &slot)| {
                let node = self.inner.nodes.get(slot as usize)?;
                let k = flat / n;
                bathymetry.data[k * n..(k + 1) * n]
                    .iter()
                    .any(|&b| b > -shore_depth)
                    .then(|| at(node.position))
            })
            .collect();
        let mut changed = 0;
        for (flat, &slot) in self.inner.slot_of_node.iter().enumerate() {
            let Some(node) = self.inner.nodes.get(slot as usize) else {
                continue;
            };
            let b = bathymetry.data[flat];
            if node.weight > 0.0
                && node.parent_depth > 0.0
                && b < 0.0
                && !shore.contains(&at(node.position))
            {
                bathymetry.data[flat] = (1.0 - node.weight) * b - node.weight * node.parent_depth;
                changed += 1;
            }
        }
        bathymetry.compute_gradients(ops, geom);
        changed
    }

    /// Parent over child still-water depth at the open-boundary nodes,
    /// `(min, median, max)`, for a bed `bathymetry`; `None` without parent
    /// depth.
    pub fn depth_ratios(&self, bathymetry: &Bathymetry2D) -> Option<(f64, f64, f64)> {
        let mut ratios: Vec<f64> = self
            .inner
            .slot_of_node
            .iter()
            .enumerate()
            .filter_map(|(flat, &slot)| {
                let node = self.inner.nodes.get(slot as usize)?;
                let b = bathymetry.data[flat];
                (node.on_boundary && node.parent_depth > 0.0 && b < 0.0)
                    .then(|| node.parent_depth / -b)
            })
            .collect();
        ratios.sort_by(f64::total_cmp);
        Some((*ratios.first()?, ratios[ratios.len() / 2], *ratios.last()?))
    }
}

impl ExternalStateProvider for OceanModelState {
    /// Parent `(η, u, v)` in the mesh axes, the velocity transport-scaled.
    fn external_state(&self, ctx: &BCContext2D) -> ExternalState {
        let slot = self.slot(ctx.node_index, ctx.position);
        match self.evaluate(slot, &self.moment(ctx.time), ctx.bathymetry) {
            (eta, Some((u, v))) => ExternalState::new(eta, u, v),
            (eta, None) => ExternalState::elevation(eta),
        }
    }
}

/// Weights of the parent snapshots and the ramp factor at one time.
#[derive(Clone, Copy, Debug)]
struct Moment {
    time: TimeStencil,
    ramp: f64,
}

/// Flow-relaxation band towards the parent state: `γ (q_parent − q)` with
/// `γ = weight(d)/τ`. Built by [`OceanModelState::relaxation`]; see the
/// module docs.
#[derive(Clone, Debug)]
pub struct NestingRelaxation2D {
    parent: OceanModelState,
    rate: f64,
    momentum_only: bool,
    wet_depth: f64,
}

impl NestingRelaxation2D {
    /// Relax only the momentum, keeping the mass (the depth) free.
    pub fn with_momentum_only(mut self, enable: bool) -> Self {
        self.momentum_only = enable;
        self
    }

    /// Depth (m) below which a node, in the child or the parent, is not
    /// relaxed (default 5 cm).
    pub fn with_wet_depth(mut self, depth: f64) -> Self {
        self.wet_depth = depth;
        self
    }

    /// The source at a slot for state `q` over bed `bed`.
    #[inline]
    fn relax(&self, slot: usize, moment: &Moment, q: &SWEState2D, bed: f64) -> SWEState2D {
        let weight = self.parent.inner.nodes[slot].weight;
        if weight <= 0.0 || q.h < self.wet_depth {
            return SWEState2D::zero();
        }
        let (eta, velocity) = self.parent.evaluate(slot, moment, bed);
        let h = eta - bed;
        if h < self.wet_depth {
            return SWEState2D::zero();
        }
        let gamma = weight * self.rate;
        let (u, v) = velocity.unwrap_or((q.hu / q.h, q.hv / q.h));
        let dh = if self.momentum_only { 0.0 } else { h - q.h };
        let depth = if self.momentum_only { q.h } else { h };
        SWEState2D::new(
            gamma * dh,
            gamma * (depth * u - q.hu),
            gamma * (depth * v - q.hv),
        )
    }
}

impl SourceTerm2D for NestingRelaxation2D {
    /// Per-node evaluation by position (slow: a search over the boundary
    /// nodes). The RHS kernels use [`SourceTerm2D::add_element`].
    fn evaluate(&self, ctx: &SourceContext2D) -> SWEState2D {
        let slot = self.parent.slot(None, ctx.position);
        self.relax(
            slot,
            &self.parent.moment(ctx.time),
            &ctx.state,
            ctx.bathymetry,
        )
    }

    fn add_element(
        &self,
        element: &ElementSources<'_>,
        h: &mut [f64],
        hu: &mut [f64],
        hv: &mut [f64],
    ) {
        let inner = &self.parent.inner;
        let k = element.element.as_usize();
        if !inner.band_elements[k] {
            return;
        }
        let moment = self.parent.moment(element.time);
        let base = k * inner.n_nodes;
        for i in 0..element.n_nodes() {
            let slot = inner.slot_of_node[base + i];
            if slot == u32::MAX {
                continue;
            }
            let bed = element
                .bathymetry
                .map_or(0.0, |b| b.get(element.element, i));
            let s = self.relax(slot as usize, &moment, &element.state(i), bed);
            h[i] += s.h;
            hu[i] += s.hu;
            hv[i] += s.hv;
        }
    }

    fn name(&self) -> &'static str {
        "nesting_relaxation_2d"
    }

    fn changes_mass(&self) -> bool {
        !self.momentum_only
    }

    fn add_mass_element(&self, element: &ElementSources<'_>, h: &mut [f64]) {
        let inner = &self.parent.inner;
        let k = element.element.as_usize();
        if self.momentum_only || !inner.band_elements[k] {
            return;
        }
        let moment = self.parent.moment(element.time);
        let base = k * inner.n_nodes;
        for (i, h) in h.iter_mut().enumerate().take(element.n_nodes()) {
            let slot = inner.slot_of_node[base + i];
            if slot == u32::MAX {
                continue;
            }
            let bed = element
                .bathymetry
                .map_or(0.0, |b| b.get(element.element, i));
            *h += self.relax(slot as usize, &moment, &element.state(i), bed).h;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::{FieldSeries, GeoGrid, LocalProjection};

    const T0: f64 = 1_706_594_400.0;
    const DEPTH: f64 = 50.0;

    /// A parent on a regular grid around the origin of `projection`, hourly
    /// ζ = 0.1, 0.3, 0.7 m, uniform (east, north) = (0.2, −0.1) m/s, depth
    /// `DEPTH`.
    fn reader(projection: &LocalProjection) -> Arc<OceanModelReader> {
        let (lat0, lon0) = projection.xy_to_geo(-20_000.0, -20_000.0);
        let (lat1, lon1) = projection.xy_to_geo(20_000.0, 20_000.0);
        let n = 9;
        let lon: Vec<f64> = (0..n)
            .map(|i| lon0 + (lon1 - lon0) * i as f64 / 8.0)
            .collect();
        let lat: Vec<f64> = (0..n)
            .map(|j| lat0 + (lat1 - lat0) * j as f64 / 8.0)
            .collect();
        let grid = GeoGrid::regular(lon, lat).unwrap();
        let m = grid.len();
        let series = |values: &[f32]| {
            FieldSeries::new(
                m,
                values
                    .iter()
                    .flat_map(|&v| std::iter::repeat_n(v, m))
                    .collect(),
            )
        };
        Arc::new(
            OceanModelReader::new(grid, vec![T0, T0 + 3600.0, T0 + 7200.0])
                .unwrap()
                .with_ssh(series(&[0.1, 0.3, 0.7]))
                .with_velocity(series(&[0.2; 3]), series(&[-0.1; 3]), "test")
                .with_depth(vec![DEPTH; m]),
        )
    }

    fn setup(options: &NestingOptions) -> (Mesh2D, DGOperators2D, OceanModelState) {
        let projection = LocalProjection::new(63.5, 8.5);
        let mesh = Mesh2D::uniform_rectangle_with_bc(
            -5000.0,
            5000.0,
            -5000.0,
            5000.0,
            5,
            5,
            BoundaryTag::Open,
        );
        let ops = DGOperators2D::new(2);
        let clock = ModelClock::new(T0);
        let parent = OceanModelState::new(
            reader(&projection),
            &mesh,
            &ops,
            &projection,
            BoundaryTag::Open,
            clock,
            options,
        )
        .unwrap();
        (mesh, ops, parent)
    }

    fn ctx(t: f64, bed: f64) -> BCContext2D {
        let interior = SWEState2D::new(-bed, 0.0, 0.0);
        BCContext2D::new(t, (5000.0, 0.0), interior, bed, (1.0, 0.0), 9.81, 1e-6)
    }

    /// Cubic interpolation through 0.1, 0.3, 0.7 at hourly steps: the
    /// quadratic ζ(τ) = 0.1 + 0.1 τ + 0.1 τ² (τ in hours).
    #[test]
    fn boundary_state_interpolates_the_parent_in_time() {
        let (_, _, parent) = setup(&NestingOptions::default());
        for (t, zeta) in [(0.0, 0.1), (1800.0, 0.175), (5400.0, 0.475), (7200.0, 0.7)] {
            let s = parent.external_state(&ctx(t, -DEPTH));
            assert!((s.eta - zeta).abs() < 1e-6, "t = {t}: {} vs {zeta}", s.eta);
            let (u, v) = s.velocity.unwrap();
            assert!(
                (u - 0.2).abs() < 1e-3 && (v + 0.1).abs() < 1e-3,
                "({u}, {v})"
            );
        }
        let linear = NestingOptions::default().with_time_interpolation(TimeInterpolation::Linear);
        let (_, _, parent) = setup(&linear);
        assert!((parent.external_state(&ctx(1800.0, -DEPTH)).eta - 0.2).abs() < 1e-6);
        assert!(parent.check_time_coverage(0.0, 7200.0).is_ok());
        assert!(parent.check_time_coverage(0.0, 7201.0).is_err());
        assert_eq!(parent.simulation_time_range(), (0.0, 7200.0));
    }

    /// A child half as deep as the parent gets twice the velocity (same
    /// transport), limited by `max_transport_ratio`.
    #[test]
    fn transport_is_conserved_across_a_bed_mismatch() {
        let (_, _, parent) = setup(&NestingOptions::default());
        let (u, _) = parent.external_state(&ctx(0.0, -DEPTH)).velocity.unwrap();
        let (u_half, _) = parent
            .external_state(&ctx(0.0, -0.5 * DEPTH))
            .velocity
            .unwrap();
        // D_parent/D_child = (0.1 + 50)/(0.1 + 25)
        assert!((u_half / u - 50.1 / 25.1).abs() < 1e-9, "{}", u_half / u);
        let (u_shallow, _) = parent.external_state(&ctx(0.0, -2.0)).velocity.unwrap();
        assert!((u_shallow / u - 3.0).abs() < 1e-9);
        let (_, _, plain) = setup(&NestingOptions::default().with_transport_scaling(false));
        let (u_plain, _) = plain.external_state(&ctx(0.0, -2.0)).velocity.unwrap();
        assert!((u_plain - u).abs() < 1e-12);
    }

    /// A tidal correction adds its tide (with the clock's astronomy) to the
    /// parent's ζ, ū and v̄ at every snapshot, band nodes included.
    #[test]
    fn tidal_correction_is_added_to_the_parent() {
        use std::f64::consts::PI;
        let options = NestingOptions::default().with_band(2000.0);
        let (_, _, parent) = setup(&options);
        let projection = LocalProjection::new(63.5, 8.5);
        let (lat, lon) = projection.xy_to_geo(0.0, 0.0);
        let correction = TidalAtlas::parse(&format!(
            "{lon} {lat} 50.0 M2 0.25 40.0 0.05 130.0 0.02 300.0\n"
        ))
        .unwrap();
        let duration = 7200.0;
        let corrected = parent
            .clone()
            .with_tidal_correction(&correction, &projection, duration, 1e5)
            .unwrap();
        let clock = ModelClock::new(T0);
        let n = clock.nodal_correction("M2", 0.5 * duration).unwrap();
        let omega = 2.0 * PI / crate::tides::constituent_period("M2").unwrap();
        let tide = |amp: f64, lag: f64, t: f64| {
            n.f * amp * (omega * t + n.phase_offset_rad() - lag.to_radians()).cos()
        };
        // The mesh is centred on the projection origin: east is x there
        for t in [0.0, 3600.0, 7200.0] {
            let (a, b) = (
                parent.external_state(&ctx(t, -DEPTH)),
                corrected.external_state(&ctx(t, -DEPTH)),
            );
            assert!(
                (b.eta - a.eta - tide(0.25, 40.0, t)).abs() < 1e-6,
                "t = {t}"
            );
            let ((ua, va), (ub, vb)) = (a.velocity.unwrap(), b.velocity.unwrap());
            assert!((ub - ua - tide(0.05, 130.0, t)).abs() < 1e-4, "t = {t}");
            assert!((vb - va - tide(0.02, 300.0, t)).abs() < 1e-4, "t = {t}");
        }
        // Band nodes carry the correction too (their slots follow the
        // boundary ones)
        let n_slots = parent.n_boundary_nodes() + parent.n_band_nodes();
        assert!(n_slots > parent.n_boundary_nodes());
        let n_times = 3;
        let changed = (0..n_slots)
            .filter(|&slot| {
                let (a, b) = (
                    &parent.inner.series[slot * n_times * 3..][..3],
                    &corrected.inner.series[slot * n_times * 3..][..3],
                );
                (b[0] - a[0] - tide(0.25, 40.0, 0.0) as f32).abs() < 1e-5
            })
            .count();
        assert_eq!(changed, n_slots);
        // Not covered: an error, not a silent zero
        assert!(
            parent
                .with_tidal_correction(&correction, &projection, duration, 10.0)
                .is_err()
        );
    }

    /// The ramp starts the parent state from rest.
    #[test]
    fn ramp_starts_from_rest() {
        let (_, _, parent) = setup(&NestingOptions::default().with_ramp_up(3600.0));
        let s = parent.external_state(&ctx(0.0, -DEPTH));
        assert_eq!((s.eta, s.velocity), (0.0, Some((0.0, 0.0))));
        let s = parent.external_state(&ctx(1800.0, -DEPTH));
        assert!((s.eta - 0.5 * 0.175).abs() < 1e-6, "{}", s.eta);
        let s = parent.external_state(&ctx(3600.0, -DEPTH));
        assert!((s.eta - 0.3).abs() < 1e-6);
    }

    /// TODO P1.5 regression: the reference level ramps with ζ. Added after
    /// the ramp, it stood at the boundary from the start (NorKyst at Mausund:
    /// +0.27 m at rest, which flooded in as a bore, the mean level rising
    /// 5 cm in three minutes).
    #[test]
    fn the_reference_level_ramps_with_the_parent() {
        let options = NestingOptions::default()
            .with_ramp_up(3600.0)
            .with_reference_level(0.25);
        let (_, _, parent) = setup(&options);
        let s = parent.external_state(&ctx(0.0, -DEPTH));
        assert_eq!(s.eta, 0.0);
        let s = parent.external_state(&ctx(1800.0, -DEPTH));
        assert!((s.eta - 0.5 * (0.175 + 0.25)).abs() < 1e-6, "{}", s.eta);
        let s = parent.external_state(&ctx(3600.0, -DEPTH));
        assert!((s.eta - (0.3 + 0.25)).abs() < 1e-6, "{}", s.eta);
    }

    #[test]
    #[should_panic(expected = "outside the parent file")]
    fn forcing_past_the_last_snapshot_panics() {
        // P0.20: this used to return the last snapshot forever.
        let (_, _, parent) = setup(&NestingOptions::default());
        parent.external_state(&ctx(7260.0, -DEPTH));
    }

    /// The band covers the nodes within its width; weights fall from 1 at
    /// the boundary to 0 at the inner edge, and a state equal to the parent
    /// is not changed.
    #[test]
    fn relaxation_band_weights_and_fixed_point() {
        let options = NestingOptions::default().with_band(2000.0);
        let (mesh, ops, parent) = setup(&options);
        assert_eq!(parent.n_snapped(), 0);
        assert!(parent.n_band_nodes() > 0);
        let inner = &parent.inner;
        for node in &inner.nodes {
            let (x, y) = node.position;
            let d = (5000.0 - x.abs()).min(5000.0 - y.abs());
            let expected = SpongeProfile::Cosine.evaluate(1.0 - d / 2000.0);
            assert!((node.weight - expected).abs() < 1e-9, "d = {d}");
        }
        // Centre element: no band
        assert!(!inner.band_elements[12]);
        assert!(inner.band_elements[0]);
        let relax = parent.relaxation(600.0);
        let time = parent.moment(0.0);
        let slot = inner.slot_of_node[0] as usize;
        let at_parent = SWEState2D::from_primitives(DEPTH + 0.1, 0.2, -0.1);
        let s = relax.relax(slot, &time, &at_parent, -DEPTH);
        assert!(
            s.h.abs() < 1e-10 && s.hu.abs() < 1e-6 && s.hv.abs() < 1e-6,
            "{s:?}"
        );
        // At rest and 0.1 m low: pulled up at 1/τ on the boundary
        let s = relax.relax(slot, &time, &SWEState2D::new(DEPTH, 0.0, 0.0), -DEPTH);
        assert!((s.h - 0.1 / 600.0).abs() < 1e-10);
        let dry = relax.relax(slot, &time, &SWEState2D::new(0.01, 0.0, 0.0), -DEPTH);
        assert_eq!(dry, SWEState2D::zero());
        let _ = (mesh, ops);
    }

    /// Blending pulls the child bed to the parent's at the boundary and
    /// leaves the interior alone.
    #[test]
    fn bathymetry_is_blended_to_the_parent_across_the_band() {
        let (mesh, ops, parent) = setup(&NestingOptions::default().with_band(2000.0));
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let mut bed = Bathymetry2D::constant(mesh.n_elements, ops.n_nodes, -20.0);
        assert_eq!(parent.depth_ratios(&bed), Some((2.5, 2.5, 2.5)));
        let changed = parent.blend_bathymetry(&mut bed, &ops, &geom, 0.0);
        assert!(changed > 0);
        let (r0, _, r1) = parent.depth_ratios(&bed).unwrap();
        assert!((r0 - 1.0).abs() < 1e-12 && (r1 - 1.0).abs() < 1e-12);
        assert_eq!(bed.get(ElementIndex::new(12), 4), -20.0);
    }

    /// TODO P1.5 regression: shores keep their bed, and the bed stays
    /// continuous. A 1 m deep corner element in a 20 m sea: it and the
    /// elements touching it keep their nodes; the rest of the band blends.
    #[test]
    fn blending_leaves_the_shores_alone() {
        let (mesh, ops, parent) = setup(&NestingOptions::default().with_band(2000.0));
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let shallow = |x: f64, y: f64| x <= -3000.0 + 1e-6 && y <= -3000.0 + 1e-6;
        let bed_at = |x: f64, y: f64| if shallow(x, y) { -1.0 } else { -20.0 };
        let mut bed = Bathymetry2D::from_function(&mesh, &ops, &geom, bed_at);
        let changed = parent.blend_bathymetry(&mut bed, &ops, &geom, 2.0);
        assert!(changed > 0);
        let mut by_position = std::collections::HashMap::new();
        let mut far_corner = None;
        for k in ElementIndex::iter(mesh.n_elements) {
            // Elements with a shallow node
            let shore = (0..ops.n_nodes).any(|i| {
                let [x, y] = mesh.reference_to_physical(k, ops.nodes_r[i], ops.nodes_s[i]);
                shallow(x, y)
            });
            for i in 0..ops.n_nodes {
                let [x, y] = mesh.reference_to_physical(k, ops.nodes_r[i], ops.nodes_s[i]);
                let b = bed.get(k, i);
                if shore {
                    assert_eq!(b, bed_at(x, y), "shore element node ({x}, {y})");
                }
                let key = ((x * 1e3).round() as i64, (y * 1e3).round() as i64);
                let first = *by_position.entry(key).or_insert(b);
                assert_eq!(first, b, "discontinuous at ({x}, {y})");
                if (x - 5000.0).abs() < 1e-6 && (y + 5000.0).abs() < 1e-6 {
                    far_corner = Some(b);
                }
            }
        }
        // The open corner far from the shore is the parent's depth
        assert_eq!(far_corner, Some(-50.0));
    }

    #[test]
    fn uncovered_boundary_is_an_error() {
        let projection = LocalProjection::new(63.5, 8.5);
        let mesh = Mesh2D::uniform_rectangle_with_bc(
            -50_000.0,
            50_000.0,
            -5000.0,
            5000.0,
            10,
            1,
            BoundaryTag::Open,
        );
        let ops = DGOperators2D::new(1);
        let result = OceanModelState::new(
            reader(&projection),
            &mesh,
            &ops,
            &projection,
            BoundaryTag::Open,
            ModelClock::new(T0),
            &NestingOptions::default(),
        );
        assert!(matches!(result, Err(NestingError::NotCovered { .. })));
    }
}

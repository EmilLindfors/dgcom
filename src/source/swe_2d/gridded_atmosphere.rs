//! Wind stress and atmospheric pressure from a weather model's grid.
//!
//! [`GriddedAtmosphere2D`] forces the 2D SWE with the 10 m wind and the
//! sea-level pressure of an [`AtmosphereReader`] (MET Nordic, MEPS,
//! AROME-Arctic, ERA5):
//!
//! ```text
//! S_hu = τ_x/ρ − (h/ρ) ∂p/∂x,    S_hv = τ_y/ρ − (h/ρ) ∂p/∂y,
//! τ = ρ_air C_d(|U₁₀|) |U₁₀| U₁₀
//! ```
//!
//! as [`WindStress2D`](super::WindStress2D) and
//! [`AtmosphericPressure2D`](super::AtmosphericPressure2D) do for analytic
//! fields. Everything spatial is precomputed per mesh node: the bilinear
//! stencil in the weather-model grid, the weights of the *exact* gradient of
//! that bilinear interpolant in mesh coordinates (for ∇p), and the rotation
//! from east/north into the mesh axes. At run time a node costs one time
//! stencil (linear: weather fields change on fronts, where cubic
//! interpolation would overshoot) and a few dozen multiply-adds.
//!
//! The sea settles under a pressure field at the inverse-barometer level
//! `η = −(p − p_ref)/(ρ g)`, `p_ref` = 101 325 Pa. Open boundaries whose
//! external data lack that response (tides, a parent model forced without
//! pressure) must add it, or they pin the boundary to the wrong level and
//! drive a current through it: [`GriddedAtmosphere2D`] implements
//! [`BoundaryLevel`], for
//! [`InverseBarometer`](crate::boundary::InverseBarometer).
//!
//! Both forcings can be ramped up from zero over a spin-up time
//! ([`GriddedAtmosphere2D::with_ramp_up`]); the inverse-barometer level
//! ramps with them.
//!
//! # 3D models
//!
//! In a mode-split 3D model ([`crate::physics::Hydrostatic3D`]) the wind
//! stress is the surface boundary condition of the columns, while the
//! pressure gradient is depth-uniform and stays with the 2D module.
//! [`GriddedAtmosphere2D::split_for_3d`] makes the two parts: the source term
//! without its wind for the 2D module, and a [`GriddedWindStress`] for
//! [`crate::physics::Hydrostatic3D::with_surface_stress`]. Both share the
//! node stencils and the regridded snapshots.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use crate::boundary::{BCContext2D, BoundaryLevel, tidal_ramp};
use crate::io::{
    AtmosphereReader, CoordinateProjection, P_REFERENCE, Stencil, TimeInterpolation, TimeStencil,
    east_axis,
};
use crate::mesh::Mesh2D;
use crate::operators::DGOperators2D;
use crate::physics::SurfaceStress3D;
use crate::solver::SWEState2D;
use crate::solver::core::blocks::for_each_block;
use crate::source::{ElementSources, SourceContext2D, SourceTerm2D};
use crate::time::ModelClock;
use crate::types::ElementIndex;

use super::wind::{DragCoefficient, RHO_AIR, RHO_WATER};

const G: f64 = 9.81;

/// Precomputed sampling of the weather grid at one mesh node.
#[derive(Clone, Copy, Debug)]
struct AtmosphereNode {
    stencil: Stencil,
    /// `(∂w/∂x, ∂w/∂y)` of the stencil's corners
    gradient: [(f64, f64); 4],
    /// East in the mesh plane, `(cos θ, sin θ)`
    east: (f64, f64),
}

impl AtmosphereNode {
    /// Sampling of `grid` at mesh position `(x, y)`, `None` outside it.
    fn at(
        grid: &crate::io::GeoGrid,
        projection: &(dyn CoordinateProjection + Send + Sync),
        (x, y): (f64, f64),
    ) -> Option<Self> {
        let (lat, lon) = projection.xy_to_geo(x, y);
        let p = grid.locate(lon, lat)?;
        Some(Self {
            stencil: Stencil::bilinear(grid, &p, |_| true)?,
            gradient: grid
                .plane_frame(&p, projection)
                .gradient_weights(p.fx, p.fy),
            east: east_axis(projection, lat, lon),
        })
    }
}

struct Inner {
    /// Unique per atmosphere (clones share it): the key of the per-thread
    /// snapshot cache
    id: u64,
    reader: Arc<AtmosphereReader>,
    projection: Arc<dyn CoordinateProjection + Send + Sync>,
    clock: ModelClock,
    nodes: Vec<AtmosphereNode>,
    n_nodes: usize,
    drag: DragCoefficient,
    rho_air: f64,
    rho_water: f64,
    h_min: f64,
    ramp: Option<f64>,
    /// Whether the source term applies the wind stress and the pressure
    /// gradient; the snapshots carry whatever the reader has
    wind: bool,
    pressure: bool,
    /// Recently used snapshots regridded onto the nodes
    cache: RwLock<Vec<(usize, Arc<Snapshot>)>>,
}

/// One weather snapshot on the mesh nodes: `[u, v, ∂p/∂x, ∂p/∂y, p − p_ref]`
/// per node, the wind in the mesh axes (unramped).
type Snapshot = Vec<[f32; 5]>;

/// Snapshots kept regridded: the bracketing pair and the neighbours a
/// multirate step may reach.
const CACHED_SNAPSHOTS: usize = 4;

/// Source of [`Inner::id`]. Not the `Arc`'s address: the thread-local
/// caches outlive an atmosphere (rayon's workers keep running), and a new
/// one allocated at a dropped one's address would read its snapshots.
static NEXT_ID: AtomicU64 = AtomicU64::new(0);

thread_local! {
    /// Per-thread references to regridded snapshots, `(source id, snapshot,
    /// values)`: the RHS kernels look snapshots up once per element on
    /// every thread, and a shared lock (or a shared reference count) there
    /// made the forcing cost 4× the RHS at 24 threads.
    static LOCAL_SNAPSHOTS: std::cell::RefCell<Vec<(u64, usize, Arc<Snapshot>)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Wind stress and pressure-gradient force from gridded weather-model
/// fields; see the module docs. Cheap to clone.
#[derive(Clone)]
pub struct GriddedAtmosphere2D {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for GriddedAtmosphere2D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GriddedAtmosphere2D")
            .field("reader", &self.inner.reader.summary())
            .field("nodes", &self.inner.nodes.len())
            .field("drag", &self.inner.drag)
            .field("ramp", &self.inner.ramp)
            .field("wind", &self.inner.wind)
            .field("pressure", &self.inner.pressure)
            .finish()
    }
}

impl GriddedAtmosphere2D {
    /// Sample `reader` at every node of `mesh`. Mesh coordinates map to
    /// longitude/latitude with `projection`, simulation time to UTC with
    /// `clock`. Every node must lie inside the weather grid.
    ///
    /// Wind stress uses Large & Pond (1981) drag, ρ_air = 1.225 kg/m³,
    /// ρ = 1025 kg/m³; nodes shallower than 1 cm are not forced.
    pub fn new<P: CoordinateProjection + Send + Sync + 'static>(
        reader: Arc<AtmosphereReader>,
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        projection: P,
        clock: ModelClock,
    ) -> Result<Self, String> {
        let projection: Arc<dyn CoordinateProjection + Send + Sync> = Arc::new(projection);
        let n_nodes = ops.n_nodes;
        let mut nodes = Vec::with_capacity(mesh.n_elements * n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..n_nodes {
                let [x, y] = mesh.reference_to_physical(k, ops.nodes_r[i], ops.nodes_s[i]);
                let node = AtmosphereNode::at(reader.grid(), projection.as_ref(), (x, y))
                    .ok_or_else(|| {
                        let (lat, lon) = projection.xy_to_geo(x, y);
                        format!(
                            "GriddedAtmosphere2D: mesh node at (x, y) = ({x:.0}, {y:.0}) m \
                             (lat {lat:.4}, lon {lon:.4}) is outside the weather grid"
                        )
                    })?;
                nodes.push(node);
            }
        }
        let (wind, pressure) = (reader.has_wind(), reader.has_pressure());
        Ok(Self {
            inner: Arc::new(Inner {
                id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
                reader,
                projection,
                clock,
                nodes,
                n_nodes,
                drag: DragCoefficient::LargePond,
                rho_air: RHO_AIR,
                rho_water: RHO_WATER,
                h_min: 0.01,
                ramp: None,
                wind,
                pressure,
                cache: RwLock::new(Vec::new()),
            }),
        })
    }

    /// The same weather sampled at the nodes of another `mesh` in the same
    /// coordinates (e.g. a wave model's), sharing the reader, projection and
    /// clock, with this one's drag, ramp and switches.
    ///
    /// # Errors
    /// If a node of `mesh` lies outside the weather grid.
    pub fn on_mesh(&self, mesh: &Mesh2D, ops: &DGOperators2D) -> Result<Self, String> {
        let inner = &self.inner;
        let mut nodes = Vec::with_capacity(mesh.n_elements * ops.n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let [x, y] = mesh.reference_to_physical(k, ops.nodes_r[i], ops.nodes_s[i]);
                nodes.push(self.node_at((x, y)).ok_or_else(|| {
                    let (lat, lon) = inner.projection.xy_to_geo(x, y);
                    format!(
                        "GriddedAtmosphere2D: mesh node at (x, y) = ({x:.0}, {y:.0}) m \
                         (lat {lat:.4}, lon {lon:.4}) is outside the weather grid"
                    )
                })?);
            }
        }
        Ok(Self {
            inner: Arc::new(Inner {
                id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
                reader: Arc::clone(&inner.reader),
                projection: Arc::clone(&inner.projection),
                clock: inner.clock,
                nodes,
                n_nodes: ops.n_nodes,
                drag: inner.drag,
                rho_air: inner.rho_air,
                rho_water: inner.rho_water,
                h_min: inner.h_min,
                ramp: inner.ramp,
                wind: inner.wind,
                pressure: inner.pressure,
                cache: RwLock::new(Vec::new()),
            }),
        })
    }

    /// The number of nodes sampled (elements × nodes per element).
    pub fn n_points(&self) -> usize {
        self.inner.nodes.len()
    }

    /// The weather model's 10 m wind (m/s, mesh axes) at every node at time
    /// `t`, linear in time between its snapshots, whatever the switches and
    /// not ramped: for a wave model, which starts from a sea of its own
    /// rather than from rest. Zero if the reader has no wind.
    ///
    /// # Panics
    /// If `wind` is not one per node, or `t` lies outside the weather file.
    pub fn wind_into(&self, t: f64, wind: &mut [[f64; 2]]) {
        let inner = &self.inner;
        assert_eq!(wind.len(), inner.nodes.len(), "one wind per node");
        if !inner.reader.has_wind() {
            wind.fill([0.0; 2]);
            return;
        }
        let time = self.time_stencil(t);
        let nn = inner.n_nodes;
        for_each_block(
            inner.nodes.len() / nn,
            [wind],
            || (),
            |_, k, [wind]| {
                self.with_snapshots(&time, |snapshots| {
                    for (i, w) in wind.iter_mut().enumerate() {
                        let [u, v, ..] = Self::cached(snapshots, k * nn + i);
                        *w = [u, v];
                    }
                });
            },
        );
    }

    fn inner_mut(&mut self) -> &mut Inner {
        Arc::get_mut(&mut self.inner).expect("configure GriddedAtmosphere2D before cloning it")
    }

    /// Wind drag coefficient (default Large & Pond).
    pub fn with_drag(mut self, drag: DragCoefficient) -> Self {
        self.inner_mut().drag = drag;
        self
    }

    /// Ramp both forcings (and the inverse-barometer level) up from zero
    /// over `duration` seconds of simulation time.
    pub fn with_ramp_up(mut self, duration: f64) -> Self {
        self.inner_mut().ramp = Some(duration);
        self
    }

    /// Switch the wind stress off (pressure only).
    ///
    /// For a 3D model use [`Self::split_for_3d`], which hands the wind to the
    /// columns instead.
    pub fn without_wind(mut self) -> Self {
        self.inner_mut().wind = false;
        self
    }

    /// Switch the pressure gradient off (wind only).
    pub fn without_pressure(mut self) -> Self {
        self.inner_mut().pressure = false;
        self
    }

    /// Split for a mode-split 3D model ([`crate::physics::Hydrostatic3D`]):
    /// this source term without its wind stress, for the 2D module (the
    /// pressure gradient, if on), and the wind stress on the 3D columns, for
    /// [`crate::physics::Hydrostatic3D::with_surface_stress`]. Both keep this
    /// one's drag, ramp and clock, and share its stencils and snapshots.
    ///
    /// # Panics
    /// If `self` has been cloned (configure it before cloning).
    pub fn split_for_3d(self) -> (Self, GriddedWindStress) {
        let wind = self.inner.wind;
        let atmosphere = self.without_wind();
        let stress = GriddedWindStress {
            atmosphere: atmosphere.clone(),
            wind,
        };
        (atmosphere, stress)
    }

    /// The weather fields.
    pub fn reader(&self) -> &AtmosphereReader {
        &self.inner.reader
    }

    /// Check that simulation times `[t_start, t_end]` lie within the weather
    /// file.
    pub fn check_time_coverage(&self, t_start: f64, t_end: f64) -> Result<(), String> {
        for t in [t_start, t_end] {
            if !self.inner.reader.covers_time(self.inner.clock.unix(t)) {
                return Err(self.coverage_message(t));
            }
        }
        Ok(())
    }

    fn coverage_message(&self, t: f64) -> String {
        let (t0, t1) = self.inner.reader.time_range();
        let clock = &self.inner.clock;
        format!(
            "GriddedAtmosphere2D: simulation time {t} s is outside the weather file, which covers \
             simulation times [{}, {}] s (epoch {})",
            clock.model_time(t0),
            clock.model_time(t1),
            clock.format(0.0)
        )
    }

    fn time_stencil(&self, t: f64) -> TimeStencil {
        TimeStencil::new(
            self.inner.reader.times(),
            self.inner.clock.unix(t),
            TimeInterpolation::Linear,
        )
        // Out-of-range times are a configuration error, not a reason to
        // freeze the forcing
        .unwrap_or_else(|| panic!("{}", self.coverage_message(t)))
    }

    /// Wind (mesh axes) and pressure gradient at a node, ramped; each zero
    /// unless `wind`, `pressure`.
    #[inline]
    fn fields(
        &self,
        node: &AtmosphereNode,
        time: &TimeStencil,
        ramp: f64,
        [wind, pressure]: [bool; 2],
    ) -> ((f64, f64), (f64, f64)) {
        let inner = &self.inner;
        let reader = &inner.reader;
        let space = node.stencil.idx.iter().map(|&k| k as usize);
        let wind = match reader.wind() {
            Some((u, v)) if wind => {
                let w = space.clone().zip(node.stencil.w);
                let (e, n) = (u.interpolate(time, w.clone()), v.interpolate(time, w));
                let (c, s) = node.east;
                (ramp * (e * c - n * s), ramp * (e * s + n * c))
            }
            _ => (0.0, 0.0),
        };
        let gradient = match reader.pressure() {
            Some(p) if pressure => (
                ramp * p.interpolate(time, space.clone().zip(node.gradient.map(|g| g.0))),
                ramp * p.interpolate(time, space.zip(node.gradient.map(|g| g.1))),
            ),
            _ => (0.0, 0.0),
        };
        (wind, gradient)
    }

    /// Momentum source at depth `h` from wind `(u, v)` and pressure gradient.
    #[inline]
    fn source(&self, h: f64, (u, v): (f64, f64), (px, py): (f64, f64)) -> (f64, f64) {
        let inner = &self.inner;
        if h < inner.h_min {
            return (0.0, 0.0);
        }
        let (tau_x, tau_y) = self.wind_stress(u, v);
        (
            (tau_x - h * px) / inner.rho_water,
            (tau_y - h * py) / inner.rho_water,
        )
    }

    /// Wind stress `ρ_air C_d(|U|) |U| U` (N/m²) of the wind `(u, v)`.
    #[inline]
    fn wind_stress(&self, u: f64, v: f64) -> (f64, f64) {
        let inner = &self.inner;
        let speed = u.hypot(v);
        let stress = inner.rho_air * inner.drag.compute(speed) * speed;
        (stress * u, stress * v)
    }

    /// Whether the source term applies the wind and the pressure gradient.
    fn switches(&self) -> [bool; 2] {
        [self.inner.wind, self.inner.pressure]
    }

    /// Node sampling at a mesh position, computed on the fly.
    fn node_at(&self, position: (f64, f64)) -> Option<AtmosphereNode> {
        AtmosphereNode::at(
            self.inner.reader.grid(),
            self.inner.projection.as_ref(),
            position,
        )
    }

    /// Inverse-barometer level (m) at a node.
    fn inverse_barometer(&self, node: &AtmosphereNode, t: f64) -> f64 {
        let inner = &self.inner;
        let Some(p) = inner.reader.pressure().filter(|_| inner.pressure) else {
            return 0.0;
        };
        let time = self.time_stencil(t);
        let space = node
            .stencil
            .idx
            .iter()
            .map(|&k| k as usize)
            .zip(node.stencil.w);
        let ramp = tidal_ramp(t, inner.ramp);
        -ramp * (p.interpolate(&time, space) - P_REFERENCE) / (inner.rho_water * G)
    }
}

impl GriddedAtmosphere2D {
    /// Snapshot `t` on the nodes, from the shared cache or regridded now.
    fn shared_snapshot(&self, t: usize) -> Arc<Snapshot> {
        let inner = &self.inner;
        if let Some((_, s)) = inner.cache.read().unwrap().iter().find(|(k, _)| *k == t) {
            return Arc::clone(s);
        }
        let single = TimeStencil {
            idx: [t, 0, 0, 0],
            w: [1.0, 0.0, 0.0, 0.0],
            len: 1,
        };
        // Everything the reader has, whatever the switches: a split
        // atmosphere's wind stress shares the snapshots
        let pressure = inner.reader.pressure();
        let values: Snapshot = inner
            .nodes
            .iter()
            .map(|node| {
                let ((u, v), (px, py)) = self.fields(node, &single, 1.0, [true, true]);
                let p = pressure.map_or(0.0, |p| {
                    let space = node
                        .stencil
                        .idx
                        .iter()
                        .map(|&k| k as usize)
                        .zip(node.stencil.w);
                    p.interpolate(&single, space) - P_REFERENCE
                });
                [u, v, px, py, p].map(|x| x as f32)
            })
            .collect();
        let values = Arc::new(values);
        let mut cache = inner.cache.write().unwrap();
        if let Some((_, s)) = cache.iter().find(|(k, _)| *k == t) {
            return Arc::clone(s);
        }
        if cache.len() == CACHED_SNAPSHOTS {
            cache.remove(0);
        }
        cache.push((t, Arc::clone(&values)));
        values
    }

    /// Call `f` with the regridded snapshots of `time` and their weights,
    /// from this thread's cache.
    fn with_snapshots<R>(
        &self,
        time: &TimeStencil,
        f: impl FnOnce(&[(&[[f32; 5]], f64)]) -> R,
    ) -> R {
        let id = self.inner.id;
        for (t, _) in time.terms() {
            let hit = LOCAL_SNAPSHOTS.with_borrow(|l| l.iter().any(|e| e.0 == id && e.1 == t));
            if !hit {
                let snapshot = self.shared_snapshot(t);
                LOCAL_SNAPSHOTS.with_borrow_mut(|l| {
                    if l.len() >= 2 * CACHED_SNAPSHOTS {
                        l.remove(0);
                    }
                    l.push((id, t, snapshot));
                });
            }
        }
        LOCAL_SNAPSHOTS.with_borrow(|l| {
            let mut refs: [(&[[f32; 5]], f64); 4] = [(&[], 0.0); 4];
            for (slot, (t, w)) in refs.iter_mut().zip(time.terms()) {
                let entry = l
                    .iter()
                    .find(|e| e.0 == id && e.1 == t)
                    .expect("inserted above");
                *slot = (&entry.2[..], w);
            }
            f(&refs[..time.len])
        })
    }

    /// `[u, v, ∂p/∂x, ∂p/∂y, p − p_ref]` at flat node `node`, unramped.
    #[inline]
    fn cached(snapshots: &[(&[[f32; 5]], f64)], node: usize) -> [f64; 5] {
        let mut sum = [0.0; 5];
        for (s, w) in snapshots {
            for (x, &v) in sum.iter_mut().zip(&s[node]) {
                *x += w * v as f64;
            }
        }
        sum
    }
}

impl SourceTerm2D for GriddedAtmosphere2D {
    /// Per-node evaluation by position, sampling the grid on the fly (slow;
    /// the RHS kernels use [`SourceTerm2D::add_element`]).
    fn evaluate(&self, ctx: &SourceContext2D) -> SWEState2D {
        let Some(node) = self.node_at(ctx.position) else {
            return SWEState2D::zero();
        };
        let ramp = tidal_ramp(ctx.time, self.inner.ramp);
        let time = self.time_stencil(ctx.time);
        let (wind, gradient) = self.fields(&node, &time, ramp, self.switches());
        let (su, sv) = self.source(ctx.state.h, wind, gradient);
        SWEState2D::new(0.0, su, sv)
    }

    fn add_element(
        &self,
        element: &ElementSources<'_>,
        _h: &mut [f64],
        hu: &mut [f64],
        hv: &mut [f64],
    ) {
        let inner = &self.inner;
        if !inner.wind && !inner.pressure {
            return;
        }
        let ramp = tidal_ramp(element.time, inner.ramp);
        let [wind, pressure] = self.switches().map(|on| if on { ramp } else { 0.0 });
        let base = element.element.as_usize() * inner.n_nodes;
        let depths = element.solution.element_h(element.element);
        self.with_snapshots(&self.time_stencil(element.time), |snapshots| {
            for (i, &h) in depths.iter().enumerate() {
                if h < inner.h_min {
                    continue;
                }
                let [u, v, px, py, _] = Self::cached(snapshots, base + i);
                let (su, sv) = self.source(h, (wind * u, wind * v), (pressure * px, pressure * py));
                hu[i] += su;
                hv[i] += sv;
            }
        });
    }

    fn name(&self) -> &'static str {
        "gridded_atmosphere_2d"
    }
}

/// The wind stress of a [`GriddedAtmosphere2D`] on the columns of a 3D model
/// ([`crate::physics::Hydrostatic3D::with_surface_stress`]); made by
/// [`GriddedAtmosphere2D::split_for_3d`].
///
/// `τ = ρ_air C_d(|U₁₀|) |U₁₀| U₁₀` (N/m²) of the 10 m wind interpolated
/// linearly in time between the regridded snapshots, in the mesh axes, with
/// the atmosphere's drag and ramp. Thin and dry columns are the 3D model's to
/// mask.
#[derive(Clone, Debug)]
pub struct GriddedWindStress {
    atmosphere: GriddedAtmosphere2D,
    /// Whether the atmosphere's wind was on before the split
    wind: bool,
}

impl SurfaceStress3D for GriddedWindStress {
    fn surface_stress_into(&self, t: f64, tau_x: &mut [f64], tau_y: &mut [f64]) {
        let atmosphere = &self.atmosphere;
        let inner = &atmosphere.inner;
        assert_eq!(tau_x.len(), inner.nodes.len(), "one stress per node");
        assert_eq!(tau_y.len(), inner.nodes.len(), "one stress per node");
        if !self.wind || !inner.reader.has_wind() {
            tau_x.fill(0.0);
            tau_y.fill(0.0);
            return;
        }
        let ramp = tidal_ramp(t, inner.ramp);
        let time = atmosphere.time_stencil(t);
        let nn = inner.n_nodes;
        for_each_block(
            inner.nodes.len() / nn,
            [tau_x, tau_y],
            || (),
            |_, k, [tau_x, tau_y]| {
                atmosphere.with_snapshots(&time, |snapshots| {
                    for (i, (tx, ty)) in tau_x.iter_mut().zip(tau_y).enumerate() {
                        let [u, v, ..] = GriddedAtmosphere2D::cached(snapshots, k * nn + i);
                        (*tx, *ty) = atmosphere.wind_stress(ramp * u, ramp * v);
                    }
                });
            },
        );
    }
}

impl BoundaryLevel for GriddedAtmosphere2D {
    /// The inverse-barometer level at the boundary node.
    fn level(&self, ctx: &BCContext2D) -> f64 {
        match ctx.node_index.filter(|&i| i < self.inner.nodes.len()) {
            Some(i) => {
                let inner = &self.inner;
                if !inner.reader.has_pressure() || !inner.pressure {
                    return 0.0;
                }
                let p = self.with_snapshots(&self.time_stencil(ctx.time), |snapshots| {
                    Self::cached(snapshots, i)[4]
                });
                -tidal_ramp(ctx.time, inner.ramp) * p / (inner.rho_water * G)
            }
            None => self
                .node_at(ctx.position)
                .map_or(0.0, |n| self.inverse_barometer(&n, ctx.time)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::{FieldSeries, GeoGrid, LocalProjection};
    use crate::mesh::BoundaryTag;
    use crate::solver::SWESolution2D;

    const T0: f64 = 1_750_000_000.0;

    /// A weather grid over ±30 km around the projection origin with a wind
    /// of (east, north) = (10, 0) m/s and pressure rising 1 hPa per 10 km to
    /// the east, then 2 hPa after an hour.
    fn reader(projection: &LocalProjection) -> Arc<AtmosphereReader> {
        let (lat0, lon0) = projection.xy_to_geo(-30_000.0, -30_000.0);
        let (lat1, lon1) = projection.xy_to_geo(30_000.0, 30_000.0);
        let n = 7;
        let lon: Vec<f64> = (0..n)
            .map(|i| lon0 + (lon1 - lon0) * i as f64 / 6.0)
            .collect();
        let lat: Vec<f64> = (0..n)
            .map(|j| lat0 + (lat1 - lat0) * j as f64 / 6.0)
            .collect();
        let grid = GeoGrid::regular(lon.clone(), lat.clone()).unwrap();
        let m = grid.len();
        let pressure: Vec<f64> = [1.0, 2.0]
            .iter()
            .flat_map(|&scale| {
                let lat = lat.clone();
                let lon = lon.clone();
                (0..m).map(move |k| {
                    let (x, _) = projection.geo_to_xy(lat[k / n], lon[k % n]);
                    P_REFERENCE + scale * 0.01 * x
                })
            })
            .collect();
        Arc::new(
            AtmosphereReader::new(grid, vec![T0, T0 + 3600.0])
                .unwrap()
                .with_wind(
                    FieldSeries::new(m, vec![10.0; 2 * m]),
                    FieldSeries::new(m, vec![0.0; 2 * m]),
                )
                .with_pressure(FieldSeries::with_offset(m, &pressure, P_REFERENCE)),
        )
    }

    fn setup() -> (Mesh2D, DGOperators2D, GriddedAtmosphere2D) {
        let projection = LocalProjection::new(63.5, 8.5);
        let mesh = Mesh2D::uniform_rectangle_with_bc(
            -10_000.0,
            10_000.0,
            -10_000.0,
            10_000.0,
            4,
            4,
            BoundaryTag::Open,
        );
        let ops = DGOperators2D::new(2);
        let atmosphere = GriddedAtmosphere2D::new(
            reader(&projection),
            &mesh,
            &ops,
            projection,
            ModelClock::new(T0),
        )
        .unwrap();
        (mesh, ops, atmosphere)
    }

    /// Wind stress of Large & Pond at 10 m/s, and the force of a uniform
    /// pressure gradient, interpolated in time.
    #[test]
    fn element_source_is_wind_stress_and_pressure_gradient() {
        let (mesh, ops, atmosphere) = setup();
        let h = 20.0;
        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        q.h_data_mut().fill(h);
        let element = ElementSources {
            element: ElementIndex::new(5),
            time: 1800.0,
            solution: &q,
            mesh: &mesh,
            ops: &ops,
            bathymetry: None,
            g: G,
            h_min: 1e-6,
        };
        let (mut sh, mut su, mut sv) = (vec![0.0; 9], vec![0.0; 9], vec![0.0; 9]);
        atmosphere.add_element(&element, &mut sh, &mut su, &mut sv);
        let stress = RHO_AIR * 1.2e-3 * 100.0 / RHO_WATER;
        // ∂p/∂x = 1.5 · 0.01 Pa/m at half past
        let expected = stress - h * 0.015 / RHO_WATER;
        for i in 0..9 {
            assert!((su[i] - expected).abs() < 1e-9, "{} vs {expected}", su[i]);
            assert!(sv[i].abs() < 1e-12 && sh[i] == 0.0);
        }
        // Per-node evaluation by position agrees
        let s = atmosphere.evaluate(&element.context(4));
        assert!((s.hu - expected).abs() < 1e-7, "{} vs {expected}", s.hu);
    }

    /// Split for a 3D model: the 2D source keeps the pressure gradient only,
    /// and the columns get the wind stress the unsplit source applied (×ρ),
    /// ramped with it; without wind before the split, none after.
    #[test]
    fn split_for_3d_hands_the_wind_to_the_columns() {
        let (mesh, ops, atmosphere) = setup();
        let n = mesh.n_elements * ops.n_nodes;
        let (pressure_only, wind) = atmosphere.with_ramp_up(7200.0).split_for_3d();
        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        let h = 20.0;
        q.h_data_mut().fill(h);
        let element = ElementSources {
            element: ElementIndex::new(5),
            time: 1800.0,
            solution: &q,
            mesh: &mesh,
            ops: &ops,
            bathymetry: None,
            g: G,
            h_min: 1e-6,
        };
        let (mut sh, mut su, mut sv) = (vec![0.0; 9], vec![0.0; 9], vec![0.0; 9]);
        pressure_only.add_element(&element, &mut sh, &mut su, &mut sv);
        let ramp = tidal_ramp(1800.0, Some(7200.0));
        assert!(ramp > 0.0 && ramp < 1.0);
        let gradient = -ramp * h * 0.015 / RHO_WATER;
        for i in 0..9 {
            assert!((su[i] - gradient).abs() < 1e-9, "{} vs {gradient}", su[i]);
            assert!(sv[i].abs() < 1e-12);
        }

        let (mut tau_x, mut tau_y) = (vec![0.0; n], vec![0.0; n]);
        wind.surface_stress_into(1800.0, &mut tau_x, &mut tau_y);
        let speed = ramp * 10.0;
        let expected = RHO_AIR * 1.2e-3 * speed * speed;
        for (tx, ty) in tau_x.iter().zip(&tau_y) {
            assert!((tx - expected).abs() < 1e-12, "{tx} vs {expected}");
            assert!(ty.abs() < 1e-12);
        }

        let (_, none) = setup().2.without_wind().split_for_3d();
        none.surface_stress_into(1800.0, &mut tau_x, &mut tau_y);
        assert!(tau_x.iter().chain(&tau_y).all(|&t| t == 0.0));
    }

    /// A wind linear in mesh coordinates (east, north), the bilinear stencils
    /// reproduce it.
    fn linear_wind(x: f64, y: f64) -> (f64, f64) {
        (8.0 + 4.0 * x / 30e3, -2.0 + 3.0 * y / 30e3)
    }

    /// A weather grid over ±30 km around the projection origin with
    /// `scale[i]` × [`linear_wind`] at T0 + i hours.
    fn linear_wind_reader(projection: &LocalProjection, scale: [f64; 2]) -> Arc<AtmosphereReader> {
        let (lat0, lon0) = projection.xy_to_geo(-30_000.0, -30_000.0);
        let (lat1, lon1) = projection.xy_to_geo(30_000.0, 30_000.0);
        let n = 7;
        let lon: Vec<f64> = (0..n)
            .map(|i| lon0 + (lon1 - lon0) * i as f64 / 6.0)
            .collect();
        let lat: Vec<f64> = (0..n)
            .map(|j| lat0 + (lat1 - lat0) * j as f64 / 6.0)
            .collect();
        let grid = GeoGrid::regular(lon.clone(), lat.clone()).unwrap();
        let m = grid.len();
        let (mut u, mut v) = (Vec::new(), Vec::new());
        for scale in scale {
            for k in 0..m {
                let (x, y) = projection.geo_to_xy(lat[k / n], lon[k % n]);
                let (e, nn) = linear_wind(x, y);
                u.push((scale * e) as f32);
                v.push((scale * nn) as f32);
            }
        }
        Arc::new(
            AtmosphereReader::new(grid, vec![T0, T0 + 3600.0])
                .unwrap()
                .with_wind(FieldSeries::new(m, u), FieldSeries::new(m, v)),
        )
    }

    /// A wind linear in mesh coordinates, which the bilinear stencils
    /// reproduce: the columns' stress is `ρ_air C_d |U| U` of that wind at
    /// every node, in the 3D model's `[element][node]` order (to the f32
    /// storage of the snapshots).
    #[test]
    fn gridded_wind_stress_varies_from_node_to_node() {
        let projection = LocalProjection::new(63.5, 8.5);
        let reader = linear_wind_reader(&projection, [1.0, 1.0]);
        let mesh = Mesh2D::uniform_rectangle(-20e3, 20e3, -15e3, 15e3, 4, 3);
        let ops = DGOperators2D::new(2);
        let (_, gridded) =
            GriddedAtmosphere2D::new(reader, &mesh, &ops, projection, ModelClock::new(T0))
                .unwrap()
                .split_for_3d();
        let analytic = crate::physics::AnalyticSurfaceStress::new(&mesh, &ops, move |x, y, _| {
            let (u, v) = linear_wind(x, y);
            let speed = u.hypot(v);
            let stress = RHO_AIR * DragCoefficient::LargePond.compute(speed) * speed;
            [stress * u, stress * v]
        });
        let n_nodes = mesh.n_elements * ops.n_nodes;
        let [mut gx, mut gy, mut ax, mut ay] = std::array::from_fn(|_| vec![0.0; n_nodes]);
        gridded.surface_stress_into(1800.0, &mut gx, &mut gy);
        analytic.surface_stress_into(1800.0, &mut ax, &mut ay);
        let (lo, hi) = ax
            .iter()
            .fold((f64::MAX, 0.0_f64), |(lo, hi), &t| (lo.min(t), hi.max(t)));
        assert!(
            hi > 2.0 * lo,
            "the test wind should vary: τ_x in [{lo}, {hi}]"
        );
        for (g, a) in gx.iter().zip(&ax).chain(gy.iter().zip(&ay)) {
            assert!((g - a).abs() < 1e-6 * a.abs() + 1e-12, "{g} vs {a}");
        }
    }

    /// The weather resampled on another mesh (a wave model's: coarser, another
    /// order) gives the 10 m wind at its nodes, linear in time and not
    /// ramped, as an atmosphere built on that mesh does; a mesh reaching
    /// beyond the weather grid is an error.
    #[test]
    fn the_wind_on_another_mesh_is_the_weathers_at_its_nodes() {
        let projection = LocalProjection::new(63.5, 8.5);
        let reader = linear_wind_reader(&projection, [1.0, 2.0]);
        let circulation = Mesh2D::uniform_rectangle(-20e3, 20e3, -15e3, 15e3, 4, 3);
        let atmosphere = GriddedAtmosphere2D::new(
            Arc::clone(&reader),
            &circulation,
            &DGOperators2D::new(2),
            projection,
            ModelClock::new(T0),
        )
        .unwrap()
        .with_ramp_up(7200.0);
        let waves = Mesh2D::uniform_rectangle(-25e3, 25e3, -20e3, 20e3, 3, 2);
        let ops = DGOperators2D::new(1);
        let resampled = atmosphere.on_mesh(&waves, &ops).unwrap();
        let n = waves.n_elements * ops.n_nodes;
        assert_eq!(resampled.n_points(), n);
        let direct =
            GriddedAtmosphere2D::new(reader, &waves, &ops, projection, ModelClock::new(T0))
                .unwrap();
        let (mut wind, mut expected) = (vec![[0.0; 2]; n], vec![[0.0; 2]; n]);
        resampled.wind_into(1800.0, &mut wind);
        direct.wind_into(1800.0, &mut expected);
        assert_eq!(wind, expected);
        for k in ElementIndex::iter(waves.n_elements) {
            for i in 0..ops.n_nodes {
                let [x, y] = waves.reference_to_physical(k, ops.nodes_r[i], ops.nodes_s[i]);
                let (u, v) = linear_wind(x, y);
                let [wu, wv] = wind[k.as_usize() * ops.n_nodes + i];
                assert!(
                    (wu - 1.5 * u).abs() < 1e-5 * u.abs().max(1.0),
                    "{wu} vs {}",
                    1.5 * u
                );
                assert!(
                    (wv - 1.5 * v).abs() < 1e-5 * v.abs().max(1.0),
                    "{wv} vs {}",
                    1.5 * v
                );
            }
        }
        let beyond = Mesh2D::uniform_rectangle(-40e3, 40e3, -10e3, 10e3, 2, 1);
        assert!(atmosphere.on_mesh(&beyond, &ops).is_err());
    }

    /// Every atmosphere reads its own weather, also one allocated where a
    /// dropped one was: the per-thread snapshot caches outlive both. Keyed
    /// by the `Arc`'s address, 6 of 10 reused addresses read the dropped
    /// atmosphere's wind (0.147 Pa from 10 m/s, for 0.037 Pa from 5 m/s).
    #[test]
    fn a_new_atmosphere_does_not_read_a_dropped_ones_snapshots() {
        let projection = LocalProjection::new(63.5, 8.5);
        let mesh = Mesh2D::uniform_rectangle(-10e3, 10e3, -10e3, 10e3, 2, 2);
        let ops = DGOperators2D::new(1);
        let grid = reader(&projection).grid().clone();
        let m = grid.len();
        let make = |speed: f32| {
            let reader = AtmosphereReader::new(grid.clone(), vec![T0, T0 + 3600.0])
                .unwrap()
                .with_wind(
                    FieldSeries::new(m, vec![speed; 2 * m]),
                    FieldSeries::new(m, vec![0.0; 2 * m]),
                );
            GriddedAtmosphere2D::new(
                Arc::new(reader),
                &mesh,
                &ops,
                projection,
                ModelClock::new(T0),
            )
            .unwrap()
        };
        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        q.h_data_mut().fill(10.0);
        // The 2D source on this thread, then the columns' stress on rayon's
        let source_hu = |atmosphere: &GriddedAtmosphere2D| {
            let element = ElementSources {
                element: ElementIndex::new(0),
                time: 0.0,
                solution: &q,
                mesh: &mesh,
                ops: &ops,
                bathymetry: None,
                g: G,
                h_min: 1e-6,
            };
            let (mut sh, mut su, mut sv) = (vec![0.0; 4], vec![0.0; 4], vec![0.0; 4]);
            atmosphere.add_element(&element, &mut sh, &mut su, &mut sv);
            su[0]
        };
        let stress = |speed: f64| RHO_AIR * 1.2e-3 * speed * speed;
        let n = mesh.n_elements * ops.n_nodes;
        let (mut tau_x, mut tau_y) = (vec![0.0; n], vec![0.0; n]);
        for (round, speed) in [10.0, 5.0, 8.0, 3.0]
            .into_iter()
            .cycle()
            .take(40)
            .enumerate()
        {
            let atmosphere = make(speed as f32);
            let hu = source_hu(&atmosphere);
            assert!(
                (hu * RHO_WATER - stress(speed)).abs() < 1e-12,
                "round {round}: 2D source of a {speed} m/s wind is {hu}"
            );
            let (_, columns) = atmosphere.split_for_3d();
            columns.surface_stress_into(0.0, &mut tau_x, &mut tau_y);
            assert!(
                tau_x.iter().all(|t| (t - stress(speed)).abs() < 1e-12),
                "round {round}: column stress of a {speed} m/s wind is {:?}",
                &tau_x[..4]
            );
        }
    }

    /// The inverse-barometer level is −(p − p_ref)/(ρg), and ramps.
    #[test]
    fn inverse_barometer_level() {
        let (_, _, atmosphere) = setup();
        let ctx = |t: f64, x: f64| {
            BCContext2D::new(
                t,
                (x, 0.0),
                SWEState2D::new(10.0, 0.0, 0.0),
                -10.0,
                (1.0, 0.0),
                G,
                1e-6,
            )
        };
        let level = atmosphere.level(&ctx(0.0, 10_000.0));
        assert!((level + 100.0 / (RHO_WATER * G)).abs() < 1e-6, "{level}");
        // By node index: the level at that node's position
        let (mesh, ops, _) = setup();
        let [x, _] =
            mesh.reference_to_physical(ElementIndex::new(3), ops.nodes_r[2], ops.nodes_s[2]);
        let by_index = atmosphere.level(&ctx(0.0, 0.0).with_node_index(3 * 9 + 2));
        assert!(
            (by_index + 0.01 * x / (RHO_WATER * G)).abs() < 1e-9,
            "{by_index} at x = {x}"
        );
        let ramped = atmosphere.with_ramp_up(7200.0);
        assert_eq!(ramped.level(&ctx(0.0, 10_000.0)), 0.0);
    }

    #[test]
    fn mesh_outside_the_grid_is_an_error() {
        let projection = LocalProjection::new(63.5, 8.5);
        let mesh =
            Mesh2D::uniform_rectangle_with_bc(0.0, 50_000.0, 0.0, 1000.0, 5, 1, BoundaryTag::Wall);
        let ops = DGOperators2D::new(1);
        assert!(
            GriddedAtmosphere2D::new(
                reader(&projection),
                &mesh,
                &ops,
                projection,
                ModelClock::new(T0)
            )
            .is_err()
        );
    }
}

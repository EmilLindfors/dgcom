//! Rivers as volume sources.
//!
//! A river adds water: its discharge `Q(t)` (m³/s) enters the element that
//! holds its mouth, spread uniformly over the element,
//!
//! ```text
//!     ∂h/∂t = … + Q(t)/A_k    at every node of element k,
//! ```
//!
//! with `A_k` the element's area. A constant nodal rate is a constant
//! polynomial, whose integral over the element is `Q` exactly: the volume of
//! the domain grows by `∫ Q dt` to round-off. The water enters with no
//! momentum of its own (`hu`, `hv` are untouched, so the river dilutes the
//! local velocity), as ROMS's `LwSrc` point sources do.
//!
//! The mouth is the river's position. A position outside the mesh (river
//! databases put mouths on the coastline, often a little inland of a water-only
//! mesh) snaps to the nearest boundary element within a given distance.
//!
//! # 2D
//!
//! [`RiverSources`] is a [`SourceTerm2D`] for the 2D model
//! (`SWEPhysics2DBuilder::with_source`).
//!
//! # 3D
//!
//! Under mode splitting the rivers belong to the 3D model
//! (`Hydrostatic3D::with_rivers`), not to its 2D module: the splitter adds
//! them to every barotropic stage and averages each river's discharge over
//! the pass with the weights of the barotropic transport `DU_avg2`, so that
//! the free surface moves by
//!
//! ```text
//!     η̄ − ηⁿ = −Δt·∇·DU_avg2 + Δt·Q̄/A_k
//! ```
//!
//! ([`crate::time::BarotropicTransport`]). The 3D layers take the step-mean
//! discharge ([`RiverInflow`]) as a volume source shared out over the levels
//! by the river's [`RiverProfile`], `s_l = w_l·Q̄/A_k` with `Σ_l w_l = 1`:
//!
//! ```text
//!     ∂H_z,l/∂t + ∇·Q_l + Ω_{l+1/2} − Ω_{l−1/2} = s_l,
//!     ∂(H_z C)_l/∂t = … + s_l·C_river.
//! ```
//!
//! Ω then still closes at the surface, a tracer with the river's own
//! concentration stays constant, and the tracer inventory grows by exactly
//! `∫ Q̄ C_river dt` (Shchepetkin & McWilliams 2005, §3; ROMS `LwSrc`
//! with `Qshape`).

use thiserror::Error;

use crate::mesh::PointLocator2D;
use crate::mesh::{Bathymetry2D, Mesh2D};
use crate::operators::GeometricFactors2D;
use crate::solver::SWEState2D;
use crate::source::{ElementSources, SourceContext2D, SourceTerm2D};
use crate::types::ElementIndex;
use crate::vertical::SigmaGrid;

/// A value given at times (s of model time) and interpolated linearly
/// between them; constant before the first and after the last.
#[derive(Clone, Debug, PartialEq)]
pub struct RiverSeries {
    times: Vec<f64>,
    values: Vec<f64>,
}

impl RiverSeries {
    /// A value that does not change.
    pub fn constant(value: f64) -> Self {
        Self {
            times: vec![0.0],
            values: vec![value],
        }
    }

    /// `values[j]` at `times[j]` (strictly increasing, at least one).
    pub fn new(times: Vec<f64>, values: Vec<f64>) -> Result<Self, RiverError> {
        if times.is_empty() || times.len() != values.len() {
            return Err(RiverError::Series(format!(
                "{} times and {} values (need the same number, at least one)",
                times.len(),
                values.len()
            )));
        }
        if times.iter().chain(&values).any(|x| !x.is_finite()) {
            return Err(RiverError::Series("a time or value is not finite".into()));
        }
        if times.windows(2).any(|w| w[1] <= w[0]) {
            return Err(RiverError::Series("times are not increasing".into()));
        }
        Ok(Self { times, values })
    }

    /// The value at time `t`.
    pub fn at(&self, t: f64) -> f64 {
        let (times, values) = (&self.times, &self.values);
        let n = times.len();
        if t <= times[0] {
            return values[0];
        }
        if t >= times[n - 1] {
            return values[n - 1];
        }
        // First time after t
        let j = times.partition_point(|&s| s <= t);
        let w = (t - times[j - 1]) / (times[j] - times[j - 1]);
        values[j - 1] + w * (values[j] - values[j - 1])
    }

    /// The smallest value.
    pub fn min(&self) -> f64 {
        self.values.iter().copied().fold(f64::INFINITY, f64::min)
    }
}

/// How a river's water is shared out over the σ-levels of a 3D model, as
/// weights `w_l ≥ 0` with `Σ_l w_l = 1` (ROMS `Qshape`).
///
/// The profiles in metres ([`Self::TopMetres`], [`Self::TaperedTopMetres`])
/// are laid on each node's still-water column `h = −B`: layer `l` spans the
/// depths `−σ_{l+1/2} h … −σ_{l−1/2} h` below the surface and takes the
/// profile's share of them. A column shallower than the profile takes all
/// of it (the share within the column, normalised). The weights stay fixed
/// in time, so the free surface stretches the profile with the column (by
/// `(h + η)/h`), as it does the σ-profiles; fixed weights keep the layers'
/// volume sources, Ω and the tracer sources the same at every stage.
#[derive(Clone, Debug, PartialEq)]
pub enum RiverProfile {
    /// In proportion to the layer thickness, `w_l = Δσ_l`.
    DepthUniform,
    /// Over the top fraction `f ∈ (0, 1]` of the column, `σ ∈ [−f, 0]`, in
    /// proportion to each layer's overlap with it. In deep water this goes
    /// deep: the top 20 % of a 400 m fjord is 80 m.
    TopFraction(f64),
    /// Over the top `d` metres (`d > 0`), uniformly per metre.
    TopMetres(f64),
    /// Over the top `d` metres (`d > 0`), per metre falling linearly from
    /// the surface to zero at `d`: `2/d (1 − z/d)` at depth `z`. NorKyst-800's
    /// rivers enter so over the top ≈ 3 m (its `river_Vshape`, whatever the
    /// depth; `notes/operational.md`, P5.1).
    TaperedTopMetres(f64),
    /// Given weights per level, bed first; normalised to sum to one.
    Levels(Vec<f64>),
}

impl RiverProfile {
    /// The weights on `sigma` (bed first) at a node of still-water depth
    /// `depth` (m; only the profiles in metres read it), summing to one.
    pub fn weights(&self, sigma: &SigmaGrid, depth: f64) -> Result<Vec<f64>, RiverError> {
        let d_sigma = sigma.d_sigma();
        let raw: Vec<f64> = match self {
            Self::DepthUniform => d_sigma.to_vec(),
            Self::TopFraction(f) => {
                if !(*f > 0.0 && *f <= 1.0) {
                    return Err(RiverError::Profile(format!(
                        "top fraction {f} is not in (0, 1]"
                    )));
                }
                sigma
                    .sigma_w()
                    .windows(2)
                    .map(|w| (w[1].min(0.0) - w[0].max(-f)).max(0.0))
                    .collect()
            }
            Self::TopMetres(d) | Self::TaperedTopMetres(d) => {
                if !(d.is_finite() && *d > 0.0) {
                    return Err(RiverError::Profile(format!("depth {d} m is not positive")));
                }
                // On land or a dry node there is no column to lay the profile
                // on: the water enters as in a column shallower than it
                if depth.is_nan() || depth <= 0.0 {
                    d_sigma.to_vec()
                } else {
                    // The profile's share from the surface down to z
                    let tapered = matches!(self, Self::TaperedTopMetres(_));
                    let share = |z: f64| {
                        let x = (z / d).clamp(0.0, 1.0);
                        if tapered { x * (2.0 - x) } else { x }
                    };
                    sigma
                        .sigma_w()
                        .windows(2)
                        .map(|w| share(-w[0] * depth) - share(-w[1] * depth))
                        .collect()
                }
            }
            Self::Levels(weights) => {
                if weights.len() != d_sigma.len() {
                    return Err(RiverError::Profile(format!(
                        "{} level weights for {} levels",
                        weights.len(),
                        d_sigma.len()
                    )));
                }
                if weights.iter().any(|w| !(w.is_finite() && *w >= 0.0)) {
                    return Err(RiverError::Profile(
                        "level weights must be finite and non-negative".into(),
                    ));
                }
                weights.clone()
            }
        };
        let sum: f64 = raw.iter().sum();
        if !sum.is_finite() || sum <= 0.0 {
            return Err(RiverError::Profile("the weights sum to zero".into()));
        }
        Ok(raw.into_iter().map(|w| w / sum).collect())
    }
}

/// A river mouth.
#[derive(Clone, Debug, PartialEq)]
pub struct River {
    /// Name, for messages.
    pub name: String,
    /// Mouth position in the mesh coordinates (m).
    pub position: [f64; 2],
    /// Discharge `Q` (m³/s, ≥ 0).
    pub discharge: RiverSeries,
    /// Temperature of the river water (°C; 3D only).
    pub temperature: RiverSeries,
    /// Salinity of the river water (psu; 3D only).
    pub salinity: RiverSeries,
    /// Vertical distribution of the inflow (3D only).
    pub profile: RiverProfile,
}

impl River {
    /// A fresh-water river at `position` with a constant discharge, at 5 °C,
    /// entering over the top 3 m, tapered as NorKyst-800's
    /// ([`RiverProfile::TaperedTopMetres`]).
    pub fn new(name: impl Into<String>, position: [f64; 2], discharge: f64) -> Self {
        Self {
            name: name.into(),
            position,
            discharge: RiverSeries::constant(discharge),
            temperature: RiverSeries::constant(5.0),
            salinity: RiverSeries::constant(0.0),
            profile: RiverProfile::TaperedTopMetres(3.0),
        }
    }

    /// With a discharge series (m³/s).
    pub fn with_discharge(mut self, discharge: RiverSeries) -> Self {
        self.discharge = discharge;
        self
    }

    /// With the river water's temperature (°C).
    pub fn with_temperature(mut self, temperature: RiverSeries) -> Self {
        self.temperature = temperature;
        self
    }

    /// With the river water's salinity (psu).
    pub fn with_salinity(mut self, salinity: RiverSeries) -> Self {
        self.salinity = salinity;
        self
    }

    /// With a vertical distribution of the inflow.
    pub fn with_profile(mut self, profile: RiverProfile) -> Self {
        self.profile = profile;
        self
    }
}

/// Errors setting up rivers.
#[derive(Debug, Error)]
pub enum RiverError {
    /// A malformed time series.
    #[error("river series: {0}")]
    Series(String),
    /// A malformed vertical profile.
    #[error("river profile: {0}")]
    Profile(String),
    /// A negative discharge.
    #[error("river {name}: negative discharge {discharge} m³/s")]
    NegativeDischarge {
        /// River name
        name: String,
        /// Smallest discharge (m³/s)
        discharge: f64,
    },
    /// A mouth too far from the mesh.
    #[error(
        "river {name}: the mouth at ({x:.0}, {y:.0}) m is {distance:.0} m from the mesh, \
         more than {max_snap:.0} m"
    )]
    OutsideMesh {
        /// River name
        name: String,
        /// Mouth position (m)
        x: f64,
        /// Mouth position (m)
        y: f64,
        /// Distance to the nearest boundary element (m)
        distance: f64,
        /// Allowed snapping distance (m)
        max_snap: f64,
    },
}

/// Rivers placed in the elements of a mesh: a volume source (see the module
/// docs).
#[derive(Clone, Debug)]
pub struct RiverSources {
    rivers: Vec<River>,
    /// Element of every river.
    elements: Vec<ElementIndex>,
    /// `1/A_k` of every river's element (1/m²).
    inv_area: Vec<f64>,
    /// Distance the mouth was moved onto the mesh (0 inside it).
    snapped: Vec<f64>,
    /// Compressed rows: the rivers of element `k` are
    /// `by_element[offsets[k]..offsets[k + 1]]`.
    offsets: Vec<usize>,
    by_element: Vec<usize>,
    /// Weights of the levels of a 3D model at every node of the river's
    /// element, `[river][node][level]` (empty until [`Self::set_levels`]).
    shape: Vec<f64>,
    n_nodes: usize,
    n_levels: usize,
}

impl RiverSources {
    /// Place `rivers` on `mesh`: each in the element holding its mouth, or,
    /// for a mouth outside the mesh, the nearest boundary element if it is
    /// within `max_snap` (m).
    pub fn new(
        rivers: Vec<River>,
        mesh: &Mesh2D,
        geom: &GeometricFactors2D,
        max_snap: f64,
    ) -> Result<Self, RiverError> {
        let placer = Placer::new(mesh);
        let placed = rivers
            .iter()
            .map(|river| placer.place(river, max_snap))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self::placed(rivers, placed, mesh, geom))
    }

    /// [`Self::new`] for the rivers that can be placed, with the errors of
    /// those that cannot (a mouth beyond `max_snap`, a negative discharge):
    /// for rivers selected by a box around a mesh, some of which flow into
    /// water the mesh does not reach.
    pub fn on_mesh(
        rivers: Vec<River>,
        mesh: &Mesh2D,
        geom: &GeometricFactors2D,
        max_snap: f64,
    ) -> (Self, Vec<RiverError>) {
        let placer = Placer::new(mesh);
        let (mut kept, mut placed, mut errors) = (Vec::new(), Vec::new(), Vec::new());
        for river in rivers {
            match placer.place(&river, max_snap) {
                Ok(p) => {
                    kept.push(river);
                    placed.push(p);
                }
                Err(e) => errors.push(e),
            }
        }
        (Self::placed(kept, placed, mesh, geom), errors)
    }

    /// The rivers in their elements, `placed[i]` = (element, snap
    /// distance).
    fn placed(
        rivers: Vec<River>,
        placed: Vec<(ElementIndex, f64)>,
        mesh: &Mesh2D,
        geom: &GeometricFactors2D,
    ) -> Self {
        let (elements, snapped): (Vec<ElementIndex>, Vec<f64>) = placed.into_iter().unzip();
        let inv_area = elements
            .iter()
            .map(|k| 1.0 / geom.area[k.as_usize()])
            .collect();
        let mut offsets = vec![0; mesh.n_elements + 1];
        for k in &elements {
            offsets[k.as_usize() + 1] += 1;
        }
        for k in 0..mesh.n_elements {
            offsets[k + 1] += offsets[k];
        }
        let mut fill = offsets.clone();
        let mut by_element = vec![0; rivers.len()];
        for (i, k) in elements.iter().enumerate() {
            by_element[fill[k.as_usize()]] = i;
            fill[k.as_usize()] += 1;
        }
        Self {
            rivers,
            elements,
            inv_area,
            snapped,
            offsets,
            by_element,
            shape: Vec::new(),
            n_nodes: 0,
            n_levels: 0,
        }
    }

    /// Number of rivers.
    pub fn len(&self) -> usize {
        self.rivers.len()
    }

    /// Whether there are no rivers.
    pub fn is_empty(&self) -> bool {
        self.rivers.is_empty()
    }

    /// River `i`.
    pub fn river(&self, i: usize) -> &River {
        &self.rivers[i]
    }

    /// The element of river `i`.
    pub fn element(&self, i: usize) -> ElementIndex {
        self.elements[i]
    }

    /// `1/A_k` of river `i`'s element (1/m²).
    pub fn inv_area(&self, i: usize) -> f64 {
        self.inv_area[i]
    }

    /// How far river `i`'s mouth was moved onto the mesh (m; 0 inside it).
    pub fn snap_distance(&self, i: usize) -> f64 {
        self.snapped[i]
    }

    /// The rivers in element `k`.
    pub fn in_element(&self, k: usize) -> &[usize] {
        &self.by_element[self.offsets[k]..self.offsets[k + 1]]
    }

    /// Discharge of river `i` at time `t` (m³/s).
    pub fn discharge(&self, i: usize, t: f64) -> f64 {
        self.rivers[i].discharge.at(t)
    }

    /// Total discharge of all rivers at time `t` (m³/s).
    pub fn total_discharge(&self, t: f64) -> f64 {
        (0..self.len()).map(|i| self.discharge(i, t)).sum()
    }

    /// The rate of the depth `∂h/∂t` (m/s) the rivers add at every node of
    /// element `k` at time `t`.
    pub fn volume_rate(&self, k: usize, t: f64) -> f64 {
        self.in_element(k)
            .iter()
            .map(|&i| self.discharge(i, t) * self.inv_area[i])
            .sum()
    }

    /// Resolve the rivers' vertical profiles on the levels of `sigma` at
    /// every node of their elements, over the still-water depth `−B` of
    /// `bathymetry` (for a 3D model; see [`RiverProfile`]).
    pub fn set_levels(
        &mut self,
        sigma: &SigmaGrid,
        bathymetry: &Bathymetry2D,
    ) -> Result<(), RiverError> {
        let nn = bathymetry.n_nodes;
        let mut shape = Vec::with_capacity(self.len() * nn * sigma.n_levels());
        for (river, k) in self.rivers.iter().zip(&self.elements) {
            for i in 0..nn {
                let depth = -bathymetry.data[k.as_usize() * nn + i];
                shape.extend(river.profile.weights(sigma, depth)?);
            }
        }
        self.shape = shape;
        self.n_nodes = nn;
        self.n_levels = sigma.n_levels();
        Ok(())
    }

    /// Weights of river `i` on the levels (bed first) at node `node` of its
    /// element, after [`Self::set_levels`].
    pub fn level_weights(&self, i: usize, node: usize) -> &[f64] {
        assert!(
            self.n_levels > 0,
            "the rivers' levels are not set (RiverSources::set_levels)"
        );
        let start = (i * self.n_nodes + node) * self.n_levels;
        &self.shape[start..start + self.n_levels]
    }

    /// Number of levels of [`Self::set_levels`] (0 before).
    pub fn n_levels(&self) -> usize {
        self.n_levels
    }
}

/// Finds the element of a river's mouth.
struct Placer<'m> {
    mesh: &'m Mesh2D,
    locator: PointLocator2D<'m>,
    boundary_elements: Vec<ElementIndex>,
}

impl<'m> Placer<'m> {
    fn new(mesh: &'m Mesh2D) -> Self {
        Self {
            mesh,
            locator: PointLocator2D::new(mesh),
            boundary_elements: (0..mesh.n_elements)
                .map(ElementIndex::new)
                .filter(|&k| (0..4).any(|f| mesh.is_boundary_face(k, f)))
                .collect(),
        }
    }

    /// The element holding `river`'s mouth, or the nearest boundary element
    /// within `max_snap`, with the distance the mouth moves.
    fn place(&self, river: &River, max_snap: f64) -> Result<(ElementIndex, f64), RiverError> {
        let low = river.discharge.min();
        if low < 0.0 {
            return Err(RiverError::NegativeDischarge {
                name: river.name.clone(),
                discharge: low,
            });
        }
        let p = river.position;
        if let Some(point) = self.locator.locate(p) {
            return Ok((point.element, 0.0));
        }
        let (distance, k) = self
            .boundary_elements
            .iter()
            .map(|&k| (distance_to_quad(&self.mesh.element_vertices(k), p), k))
            .min_by(|a, b| a.0.total_cmp(&b.0))
            .unwrap_or((f64::INFINITY, ElementIndex::new(0)));
        if !distance.is_finite() || distance > max_snap {
            return Err(RiverError::OutsideMesh {
                name: river.name.clone(),
                x: p[0],
                y: p[1],
                distance,
                max_snap,
            });
        }
        Ok((k, distance))
    }
}

/// Distance from `p` to the quadrilateral with straight edges `vertices`
/// (counter-clockwise), for a point outside it.
fn distance_to_quad(vertices: &[[f64; 2]; 4], p: [f64; 2]) -> f64 {
    (0..4)
        .map(|e| {
            let (a, b) = (vertices[e], vertices[(e + 1) % 4]);
            let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
            let length2 = dx * dx + dy * dy;
            let t = if length2 > 0.0 {
                (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length2).clamp(0.0, 1.0)
            } else {
                0.0
            };
            (p[0] - a[0] - t * dx).hypot(p[1] - a[1] - t * dy)
        })
        .fold(f64::INFINITY, f64::min)
}

impl SourceTerm2D for RiverSources {
    /// Zero: a river is a source of its element as a whole, which a node
    /// alone (it may lie on the faces of several elements) cannot tell. The
    /// RHS kernels call [`Self::add_element`].
    fn evaluate(&self, _ctx: &SourceContext2D) -> SWEState2D {
        SWEState2D::zero()
    }

    fn add_element(
        &self,
        element: &ElementSources<'_>,
        h: &mut [f64],
        _hu: &mut [f64],
        _hv: &mut [f64],
    ) {
        let k = element.element.as_usize();
        if self.in_element(k).is_empty() {
            return;
        }
        let rate = self.volume_rate(k, element.time);
        h.iter_mut().for_each(|h| *h += rate);
    }

    fn name(&self) -> &'static str {
        "rivers"
    }

    fn changes_mass(&self) -> bool {
        true
    }

    fn add_mass_element(&self, element: &ElementSources<'_>, h: &mut [f64]) {
        let k = element.element.as_usize();
        if !self.in_element(k).is_empty() {
            let rate = self.volume_rate(k, element.time);
            h.iter_mut().for_each(|h| *h += rate);
        }
    }
}

/// The rivers of one baroclinic step as the 3D layers see them: each
/// river's discharge averaged over the barotropic pass with the weights of
/// `DU_avg2` (see the module docs).
#[derive(Clone, Copy)]
pub struct RiverInflow<'a> {
    /// The rivers, with their levels set ([`RiverSources::set_levels`]).
    pub rivers: &'a RiverSources,
    /// Step-mean discharge of every river (m³/s).
    pub discharge: &'a [f64],
}

impl RiverInflow<'_> {
    /// The rate of the depth (m/s) the rivers add at every node of element `k`
    /// over the step.
    pub fn volume_rate(&self, k: usize) -> f64 {
        self.rivers
            .in_element(k)
            .iter()
            .map(|&i| self.discharge[i] * self.rivers.inv_area[i])
            .sum()
    }

    /// Add the layer volume sources `s_l = Σ w_l Q̄/A_k` (m/s) at node
    /// `node` of element `k` to `out` (one per level, bed first).
    pub fn add_layer_rates(&self, k: usize, node: usize, out: &mut [f64]) {
        for &i in self.rivers.in_element(k) {
            let rate = self.discharge[i] * self.rivers.inv_area[i];
            for (o, &w) in out.iter_mut().zip(self.rivers.level_weights(i, node)) {
                *o += w * rate;
            }
        }
    }

    /// Add the inventory sources `s_l·C_river` of a tracer to `rhs`
    /// (`[element][node][level]`, `n_nodes` per element), with the river
    /// water's concentration `concentration(river)`.
    pub fn add_tracer_sources(
        &self,
        rhs: &mut [f64],
        n_nodes: usize,
        concentration: impl Fn(&River) -> f64,
    ) {
        let nl = self.rivers.n_levels();
        for i in 0..self.rivers.len() {
            let k = self.rivers.element(i).as_usize();
            let rate = self.discharge[i] * self.rivers.inv_area[i];
            let c = concentration(self.rivers.river(i));
            let columns = rhs[k * n_nodes * nl..(k + 1) * n_nodes * nl].chunks_exact_mut(nl);
            for (node, column) in columns.enumerate() {
                for (r, &w) in column.iter_mut().zip(self.rivers.level_weights(i, node)) {
                    *r += w * rate * c;
                }
            }
        }
    }

    /// Add the inventory sources of a field `field` at the w-points
    /// (`[element][node][w-point]`, `n_levels + 1` per column) that the river
    /// water brings at the column's own value, so that its volume dilutes
    /// nothing: `s_w φ`, with `s_w` the layer sources averaged to the
    /// w-cells (half of the end layers', the mean of the two layers'
    /// between; [`crate::solver::rhs::w_cell_thicknesses`]).
    pub fn add_w_point_sources_at_own_value(&self, rhs: &mut [f64], field: &[f64], n_nodes: usize) {
        let nw = self.rivers.n_levels() + 1;
        for i in 0..self.rivers.len() {
            let k = self.rivers.element(i).as_usize();
            let rate = self.discharge[i] * self.rivers.inv_area[i];
            let columns = k * n_nodes * nw..(k + 1) * n_nodes * nw;
            for (node, (r, phi)) in rhs[columns.clone()]
                .chunks_exact_mut(nw)
                .zip(field[columns].chunks_exact(nw))
                .enumerate()
            {
                let weights = self.rivers.level_weights(i, node);
                for j in 0..nw {
                    let below = if j > 0 { weights[j - 1] } else { 0.0 };
                    let above = weights.get(j).copied().unwrap_or(0.0);
                    r[j] += 0.5 * (below + above) * rate * phi[j];
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::DGOperators2D;
    use crate::solver::SWESolution2D;
    use crate::vertical::UniformStretching;

    fn mesh() -> (Mesh2D, DGOperators2D, GeometricFactors2D) {
        let mesh = Mesh2D::uniform_rectangle(0.0, 400.0, 0.0, 200.0, 4, 2);
        let ops = DGOperators2D::new(2);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        (mesh, ops, geom)
    }

    #[test]
    fn series_interpolates_linearly_and_holds_its_ends() {
        let s = RiverSeries::new(vec![0.0, 10.0, 30.0], vec![1.0, 3.0, 2.0]).unwrap();
        assert_eq!(s.at(-5.0), 1.0);
        assert!((s.at(5.0) - 2.0).abs() < 1e-15);
        assert_eq!(s.at(10.0), 3.0);
        assert!((s.at(20.0) - 2.5).abs() < 1e-15);
        assert_eq!(s.at(100.0), 2.0);
        assert_eq!(s.min(), 1.0);
        assert!(RiverSeries::new(vec![0.0, 0.0], vec![1.0, 2.0]).is_err());
        assert!(RiverSeries::new(vec![0.0], vec![]).is_err());
        assert!(RiverSeries::new(vec![0.0], vec![f64::NAN]).is_err());
    }

    #[test]
    fn profiles_sum_to_one_where_they_should() {
        let sigma = SigmaGrid::new(5, UniformStretching);
        let uniform = RiverProfile::DepthUniform.weights(&sigma, 10.0).unwrap();
        assert!(uniform.iter().all(|w| (w - 0.2).abs() < 1e-15));
        // Top 30 %: the top layer whole, the next one half
        let top = RiverProfile::TopFraction(0.3)
            .weights(&sigma, 10.0)
            .unwrap();
        let expected = [0.0, 0.0, 0.0, 1.0 / 3.0, 2.0 / 3.0];
        for (w, e) in top.iter().zip(expected) {
            assert!((w - e).abs() < 1e-14, "{top:?}");
        }
        let levels = RiverProfile::Levels(vec![0.0, 0.0, 0.0, 1.0, 3.0])
            .weights(&sigma, 10.0)
            .unwrap();
        assert_eq!(levels, vec![0.0, 0.0, 0.0, 0.25, 0.75]);
        assert!(
            RiverProfile::Levels(vec![1.0; 4])
                .weights(&sigma, 10.0)
                .is_err()
        );
        assert!(
            RiverProfile::TopFraction(0.0)
                .weights(&sigma, 10.0)
                .is_err()
        );
        assert!(
            RiverProfile::Levels(vec![0.0; 5])
                .weights(&sigma, 10.0)
                .is_err()
        );
        assert!(RiverProfile::TopMetres(0.0).weights(&sigma, 10.0).is_err());
        assert!(
            RiverProfile::TaperedTopMetres(f64::NAN)
                .weights(&sigma, 10.0)
                .is_err()
        );
    }

    /// The profiles in metres reach the same depth in any column: the
    /// top 3 m of 10 m and of 300 m of water.
    #[test]
    fn profiles_in_metres_keep_their_depth_in_any_column() {
        let close = |w: &[f64], e: &[f64]| {
            assert_eq!(w.len(), e.len());
            for (w, e) in w.iter().zip(e) {
                assert!((w - e).abs() < 1e-13, "{w:?} ≠ {e:?}");
            }
        };
        let sigma = SigmaGrid::new(5, UniformStretching);
        // 10 m: 2 m layers; the top 3 m are the top layer and half the next
        let w = RiverProfile::TopMetres(3.0).weights(&sigma, 10.0).unwrap();
        close(&w, &[0.0, 0.0, 0.0, 1.0 / 3.0, 2.0 / 3.0]);
        // Tapered, 2x − x² of x = z/3: 8/9 above 2 m, 1/9 below
        let w = RiverProfile::TaperedTopMetres(3.0)
            .weights(&sigma, 10.0)
            .unwrap();
        close(&w, &[0.0, 0.0, 0.0, 1.0 / 9.0, 8.0 / 9.0]);
        // 300 m: all of it in the top layer (60 m)
        let w = RiverProfile::TaperedTopMetres(3.0)
            .weights(&sigma, 300.0)
            .unwrap();
        close(&w, &[0.0, 0.0, 0.0, 0.0, 1.0]);
        // A 1 m column takes the profile's share within it (2/3 − 1/9 = 5/9
        // of the tapered), normalised; layer l spans the depths 0.8 − 0.2 l
        // to 1 − 0.2 l
        let w = RiverProfile::TaperedTopMetres(3.0)
            .weights(&sigma, 1.0)
            .unwrap();
        let share = |z: f64| (z / 3.0) * (2.0 - z / 3.0);
        let expected: Vec<f64> = (0..5)
            .map(|l| {
                let top = 0.8 - 0.2 * l as f64;
                (share(top + 0.2) - share(top)) / (5.0 / 9.0)
            })
            .collect();
        close(&w, &expected);
        // Land or a dry node: as the layers' thickness
        let w = RiverProfile::TopMetres(3.0).weights(&sigma, -2.0).unwrap();
        close(&w, &[0.2; 5]);
        // A finer, stretched grid: the share above 3 m is all of it at any depth
        let sigma = SigmaGrid::new(
            40,
            crate::vertical::SongHaidvogelStretching::new(5.0, 0.4, 10.0),
        );
        for depth in [10.0, 61.0, 286.0, 800.0] {
            let w = RiverProfile::TaperedTopMetres(3.0)
                .weights(&sigma, depth)
                .unwrap();
            let sigma_w = sigma.sigma_w();
            let below: f64 = (0..40)
                .filter(|&l| -sigma_w[l + 1] * depth >= 3.0)
                .map(|l| w[l])
                .sum();
            assert!(below == 0.0, "{depth} m: {below} below 3 m");
            assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-14);
        }
    }

    #[test]
    fn rivers_are_placed_in_their_elements_or_snapped_onto_the_mesh() {
        let (mesh, _, geom) = mesh();
        let rivers = vec![
            River::new("inside", [150.0, 50.0], 10.0),
            River::new("same element", [160.0, 60.0], 5.0),
            // 30 m north of the top boundary, above element (3, 1)
            River::new("outside", [350.0, 230.0], 1.0),
        ];
        let sources = RiverSources::new(rivers, &mesh, &geom, 50.0).unwrap();
        let locator = PointLocator2D::new(&mesh);
        let inside = locator.locate([150.0, 50.0]).unwrap().element;
        let corner = locator.locate([350.0, 190.0]).unwrap().element;
        assert_eq!(sources.element(0), inside);
        assert_eq!(sources.element(1), inside);
        assert_eq!(sources.element(2), corner);
        assert!((sources.snap_distance(2) - 30.0).abs() < 1e-9);
        assert_eq!(sources.in_element(inside.as_usize()), &[0, 1]);
        assert_eq!(sources.in_element(corner.as_usize()), &[2]);
        let empty = (0..mesh.n_elements)
            .filter(|&k| k != inside.as_usize() && k != corner.as_usize())
            .all(|k| sources.in_element(k).is_empty());
        assert!(empty);
        // Element area 100 m × 100 m
        assert!((sources.volume_rate(inside.as_usize(), 0.0) - 15.0 / 1e4).abs() < 1e-15);
        assert!((sources.total_discharge(0.0) - 16.0).abs() < 1e-12);

        let far = vec![River::new("far", [350.0, 300.0], 1.0)];
        assert!(matches!(
            RiverSources::new(far, &mesh, &geom, 50.0),
            Err(RiverError::OutsideMesh { .. })
        ));
        let negative = vec![River::new("dry", [10.0, 10.0], -1.0)];
        assert!(matches!(
            RiverSources::new(negative, &mesh, &geom, 50.0),
            Err(RiverError::NegativeDischarge { .. })
        ));

        // The ones that can be placed, and why the others cannot
        let rivers = vec![
            River::new("far", [350.0, 300.0], 1.0),
            River::new("inside", [150.0, 50.0], 10.0),
            River::new("dry", [10.0, 10.0], -1.0),
            River::new("outside", [350.0, 230.0], 1.0),
        ];
        let (sources, errors) = RiverSources::on_mesh(rivers, &mesh, &geom, 50.0);
        assert_eq!(sources.len(), 2);
        assert_eq!(sources.river(0).name, "inside");
        assert_eq!(sources.element(0), inside);
        assert_eq!(sources.river(1).name, "outside");
        assert_eq!(sources.element(1), corner);
        assert_eq!(sources.in_element(corner.as_usize()), &[1]);
        assert!(matches!(
            errors.as_slice(),
            [
                RiverError::OutsideMesh { .. },
                RiverError::NegativeDischarge { .. }
            ]
        ));
    }

    /// The source integrates to the discharge over the element, and touches
    /// nothing else.
    #[test]
    fn the_source_integrates_to_the_discharge() {
        let (mesh, ops, geom) = mesh();
        let discharge = RiverSeries::new(vec![0.0, 100.0], vec![4.0, 8.0]).unwrap();
        let rivers = vec![River::new("r", [250.0, 150.0], 0.0).with_discharge(discharge)];
        let sources = RiverSources::new(rivers, &mesh, &geom, 0.0).unwrap();
        let solution = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        let nn = ops.n_nodes;
        let mut total = 0.0;
        for k in 0..mesh.n_elements {
            let element = ElementSources {
                element: ElementIndex::new(k),
                time: 25.0,
                solution: &solution,
                mesh: &mesh,
                ops: &ops,
                bathymetry: None,
                g: 9.81,
                h_min: 1e-6,
            };
            let (mut h, mut hu, mut hv) = (vec![0.0; nn], vec![0.0; nn], vec![0.0; nn]);
            sources.add_element(&element, &mut h, &mut hu, &mut hv);
            assert!(hu.iter().chain(&hv).all(|&x| x == 0.0));
            total += geom.integrate_element(k, &h);
        }
        assert!((total - 5.0).abs() < 1e-12, "∫ ∂h/∂t = {total}");
    }
}

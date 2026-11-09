//! Rivers as volume sources.
//!
//! A river adds water: its discharge `Q(t)` (m³/s) enters an element under
//! water (every node below the still water, `B < 0`), shared over the
//! element's nodes in proportion to their still-water depth `d = −B`,
//!
//! ```text
//!     ∂h/∂t = … + Q(t)·d_j / ∫_k d dA    at node j of element k.
//! ```
//!
//! Its integral over the element is `Q` exactly (in the quadrature the volume
//! is measured with), so the volume of the domain grows by `∫ Q dt` to
//! round-off. The dilution rate `s/d` is the same at every node,
//! `Q/∫_k d dA`: the river freshens the element's water evenly, and a shallow
//! node (one the low tide may dry) takes little of it. The shares use the
//! still water, not the moving surface, so that the barotropic steps and the
//! 3D layers of a mode-split step add the same water at the same nodes. The
//! water enters with no momentum of its own (`hu`, `hv` are untouched, so the
//! river dilutes the local velocity), as ROMS's `LwSrc` point sources do.
//!
//! The mouth is the river's position. River databases put mouths on the
//! coastline, often a little inland of a mesh's water: a mouth outside the
//! mesh, or in an element on the coast (one with a node on land), moves to
//! the nearest element under water within a given distance, as FVCOM's and
//! SCHISM's sources must sit in the water. A mouth outside the mesh whose
//! nearest boundary is open is left out: it lies outside the domain, and the
//! water that enters through the boundary (a parent model's, which has its
//! own rivers) carries it in; placed inside, its water would enter twice, or
//! be relaxed away in a nesting band. In a coastal element the water
//! would stand on the land nodes at the land's height (TODO P1.6: η of 5 m
//! on the 5 m land of the Frøya mesh, with the rate shared evenly), or crowd
//! into the element's sliver of water and spill up the beach.
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
use crate::solver::{Blocks, SWEState2D};
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

    /// The mean over `[t0, t1]`, exactly (the trapezoid rule between the
    /// series' times, where it is linear); the value at `t0` if `t1 ≤ t0`.
    pub fn mean_over(&self, t0: f64, t1: f64) -> f64 {
        if t1 <= t0 {
            return self.at(t0);
        }
        let inside = self.times.iter().copied().filter(|&t| t > t0 && t < t1);
        let mut points = std::iter::once(t0).chain(inside).chain(std::iter::once(t1));
        let mut last = points.next().map(|t| (t, self.at(t)));
        let mut integral = 0.0;
        for t in points {
            let v = self.at(t);
            if let Some((s, u)) = last {
                integral += 0.5 * (t - s) * (u + v);
            }
            last = Some((t, v));
        }
        integral / (t1 - t0)
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

/// A profile from its text: `uniform` ([`RiverProfile::DepthUniform`]),
/// `fraction:F`, `top:D` or `tapered:D` (D in metres), as case files and
/// `froya_real_data river_profile=` give it.
impl std::str::FromStr for RiverProfile {
    type Err = RiverError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let bad = || {
            RiverError::Profile(format!(
                "{text:?} is not uniform, fraction:F, top:D or tapered:D"
            ))
        };
        if text == "uniform" {
            return Ok(Self::DepthUniform);
        }
        let (kind, value) = text.split_once(':').ok_or_else(bad)?;
        let value: f64 = value.parse().map_err(|_| bad())?;
        let profile = match kind {
            "fraction" => Self::TopFraction(value),
            "top" => Self::TopMetres(value),
            "tapered" => Self::TaperedTopMetres(value),
            _ => return Err(bad()),
        };
        // The values' ranges, as `weights` checks them
        let fits = match profile {
            Self::TopFraction(f) => f > 0.0 && f <= 1.0,
            _ => value.is_finite() && value > 0.0,
        };
        match fits {
            true => Ok(profile),
            false => Err(RiverError::Profile(format!(
                "{text:?}: a fraction in (0, 1], a depth positive"
            ))),
        }
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
    /// Levels set over another bed than the rivers were placed on.
    #[error("rivers: {0}")]
    OtherBed(String),
    /// A mouth too far from the mesh's water.
    #[error(
        "river {name}: the mouth at ({x:.0}, {y:.0}) m is {distance:.0} m from the mesh's \
         water, more than {max_snap:.0} m"
    )]
    FarFromWater {
        /// River name
        name: String,
        /// Mouth position (m)
        x: f64,
        /// Mouth position (m)
        y: f64,
        /// Distance to the nearest element under water (m)
        distance: f64,
        /// Allowed snapping distance (m)
        max_snap: f64,
    },
    /// A mouth beyond an open boundary: outside the domain, where the water
    /// that enters through the boundary (a parent model's) carries it.
    #[error(
        "river {name}: the mouth at ({x:.0}, {y:.0}) m lies {distance:.0} m beyond an open \
         boundary, whose water carries it in"
    )]
    BeyondOpenBoundary {
        /// River name
        name: String,
        /// Mouth position (m)
        x: f64,
        /// Mouth position (m)
        y: f64,
        /// Distance to the open boundary (m)
        distance: f64,
    },
}

/// Rivers placed in the elements of a mesh: a volume source (see the module
/// docs).
#[derive(Clone, Debug)]
pub struct RiverSources {
    rivers: Vec<River>,
    /// Element of every river.
    elements: Vec<ElementIndex>,
    /// Distance the mouth was moved to the mesh's water (0 if it is in it).
    snapped: Vec<f64>,
    /// Compressed rows: the rivers of element `k` are
    /// `by_element[offsets[k]..offsets[k + 1]]`.
    offsets: Vec<usize>,
    by_element: Vec<usize>,
    /// Still-water depth `d = max(−B, 0)` at every node of each river's
    /// element, `[river][node]` (m).
    depth: Vec<f64>,
    /// Share of each river's discharge at every node of its element,
    /// `d_j / ∫_k d dA`, `[river][node]` (1/m²).
    share: Vec<f64>,
    n_nodes: usize,
    /// Weights of the levels of a 3D model at every node of the river's
    /// element, `[river][node][level]` (empty until [`Self::set_levels`]).
    shape: Vec<f64>,
    n_levels: usize,
}

impl RiverSources {
    /// Place `rivers` on `mesh` over its still-water bed `bathymetry` (the
    /// model's own): each in the element holding its mouth if that element
    /// is under water, or else in the nearest element under water if it is
    /// within `max_snap` (m) of the mouth.
    pub fn new(
        rivers: Vec<River>,
        mesh: &Mesh2D,
        geom: &GeometricFactors2D,
        bathymetry: &Bathymetry2D,
        max_snap: f64,
    ) -> Result<Self, RiverError> {
        let placer = Placer::new(mesh, bathymetry);
        let placed = rivers
            .iter()
            .map(|river| placer.place(river, max_snap))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self::placed(rivers, placed, mesh, geom, bathymetry))
    }

    /// [`Self::new`] for the rivers that can be placed, with the errors of
    /// those that cannot (a mouth beyond `max_snap` of the water, a negative
    /// discharge): for rivers selected by a box around a mesh, some of which
    /// flow into water the mesh does not reach.
    pub fn on_mesh(
        rivers: Vec<River>,
        mesh: &Mesh2D,
        geom: &GeometricFactors2D,
        bathymetry: &Bathymetry2D,
        max_snap: f64,
    ) -> (Self, Vec<RiverError>) {
        let placer = Placer::new(mesh, bathymetry);
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
        (Self::placed(kept, placed, mesh, geom, bathymetry), errors)
    }

    /// The rivers in their elements, `placed[i]` = (element, snap
    /// distance), each element under water.
    fn placed(
        rivers: Vec<River>,
        placed: Vec<(ElementIndex, f64)>,
        mesh: &Mesh2D,
        geom: &GeometricFactors2D,
        bathymetry: &Bathymetry2D,
    ) -> Self {
        let nn = bathymetry.n_nodes;
        assert_eq!(
            geom.n_nodes, nn,
            "the bathymetry's nodes are not the geometry's"
        );
        let (elements, snapped): (Vec<ElementIndex>, Vec<f64>) = placed.into_iter().unzip();
        let mut depth = Vec::with_capacity(elements.len() * nn);
        let mut share = Vec::with_capacity(elements.len() * nn);
        for &k in &elements {
            let d: Vec<f64> = bathymetry
                .element(k)
                .iter()
                .map(|&b| (-b).max(0.0))
                .collect();
            // Positive: the placer only takes elements under water
            let volume = geom.integrate_element(k.as_usize(), &d);
            share.extend(d.iter().map(|d| d / volume));
            depth.extend(d);
        }
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
            snapped,
            offsets,
            by_element,
            depth,
            share,
            n_nodes: nn,
            shape: Vec::new(),
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

    /// River `i`'s share of its discharge at every node of its element,
    /// `d_j / ∫_k d dA` (1/m²; zero on land, integrating to one over the
    /// element).
    pub fn share(&self, i: usize) -> &[f64] {
        &self.share[i * self.n_nodes..(i + 1) * self.n_nodes]
    }

    /// The still-water depths of river `i`'s element, shallowest and
    /// deepest node (m, both positive), for logs: a shallow mouth shows.
    pub fn depth_range(&self, i: usize) -> (f64, f64) {
        self.depth[i * self.n_nodes..(i + 1) * self.n_nodes]
            .iter()
            .fold((f64::INFINITY, 0.0_f64), |(lo, hi), &d| {
                (lo.min(d), hi.max(d))
            })
    }

    /// How far river `i`'s mouth was moved to the mesh's water (m; 0 if it
    /// is in it).
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

    /// Add the rate of the depth `∂h/∂t` (m/s) the rivers bring at time `t`
    /// to `h`, the nodes of element `k`.
    pub fn add_volume_rates(&self, k: usize, t: f64, h: &mut [f64]) {
        for &i in self.in_element(k) {
            let q = self.discharge(i, t);
            for (h, s) in h.iter_mut().zip(self.share(i)) {
                *h += q * s;
            }
        }
    }

    /// Resolve the rivers' vertical profiles on the levels of `sigma` at
    /// every node of their elements, over the still-water depth (for a 3D
    /// model; see [`RiverProfile`]). `bathymetry` is the model's, which must
    /// be the one the rivers were placed on (their shares are its depths).
    pub fn set_levels(
        &mut self,
        sigma: &SigmaGrid,
        bathymetry: &Bathymetry2D,
    ) -> Result<(), RiverError> {
        let nn = self.n_nodes;
        if bathymetry.n_nodes != nn {
            return Err(RiverError::OtherBed(format!(
                "{} nodes per element, the rivers were placed on {nn}",
                bathymetry.n_nodes
            )));
        }
        let mut shape = Vec::with_capacity(self.len() * nn * sigma.n_levels());
        for (i, (river, &k)) in self.rivers.iter().zip(&self.elements).enumerate() {
            let depth = &self.depth[i * nn..(i + 1) * nn];
            let same = bathymetry
                .element(k)
                .iter()
                .zip(depth)
                .all(|(&b, &d)| (-b).max(0.0) == d);
            if !same {
                return Err(RiverError::OtherBed(format!(
                    "river {}: the bed of element {} is not the one it was placed on",
                    river.name,
                    k.as_usize()
                )));
            }
            for &d in depth {
                shape.extend(river.profile.weights(sigma, d)?);
            }
        }
        self.shape = shape;
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
    /// Whether each element is under water at rest (every node `B < 0`).
    wet: Vec<bool>,
    /// The elements under water, with their bounding boxes
    /// `[x_min, y_min, x_max, y_max]`.
    water: Vec<(ElementIndex, [f64; 4])>,
    /// The boundary faces, as their end points and whether they are open
    /// (open, tidal or river faces: [`crate::mesh::BoundaryTag::is_open`]).
    boundary: Vec<([f64; 2], [f64; 2], bool)>,
}

impl<'m> Placer<'m> {
    fn new(mesh: &'m Mesh2D, bathymetry: &Bathymetry2D) -> Self {
        let wet: Vec<bool> = (0..mesh.n_elements)
            .map(|k| {
                bathymetry
                    .element(ElementIndex::new(k))
                    .iter()
                    .all(|&b| b < 0.0)
            })
            .collect();
        let water = (0..mesh.n_elements)
            .filter(|&k| wet[k])
            .map(|k| {
                let k = ElementIndex::new(k);
                let corners = mesh.element_vertices(k);
                let bounds = corners.iter().fold(
                    [
                        f64::INFINITY,
                        f64::INFINITY,
                        f64::NEG_INFINITY,
                        f64::NEG_INFINITY,
                    ],
                    |[x0, y0, x1, y1], &[x, y]| [x0.min(x), y0.min(y), x1.max(x), y1.max(y)],
                );
                (k, bounds)
            })
            .collect();
        let mut boundary = Vec::new();
        for k in ElementIndex::iter(mesh.n_elements) {
            let corners = mesh.element_vertices(k);
            for face in (0..4).filter(|&f| mesh.is_boundary_face(k, f)) {
                let open = mesh.boundary_tag(k, face).is_some_and(|tag| tag.is_open());
                boundary.push((corners[face], corners[(face + 1) % 4], open));
            }
        }
        Self {
            mesh,
            locator: PointLocator2D::new(mesh),
            wet,
            water,
            boundary,
        }
    }

    /// The element holding `river`'s mouth if it is under water, or else
    /// the nearest element under water within `max_snap`, with the distance
    /// the mouth moves. A mouth outside the mesh whose nearest boundary is
    /// open lies outside the domain: the water entering there carries it.
    fn place(&self, river: &River, max_snap: f64) -> Result<(ElementIndex, f64), RiverError> {
        let low = river.discharge.min();
        if low < 0.0 {
            return Err(RiverError::NegativeDischarge {
                name: river.name.clone(),
                discharge: low,
            });
        }
        let p = river.position;
        match self.locator.locate(p) {
            Some(point) if self.wet[point.element.as_usize()] => {
                return Ok((point.element, 0.0));
            }
            Some(_) => {}
            None => {
                let (distance, open) = self
                    .boundary
                    .iter()
                    .map(|&(a, b, open)| (distance_to_segment(a, b, p), open))
                    .min_by(|x, y| x.0.total_cmp(&y.0))
                    .unwrap_or((f64::INFINITY, false));
                if open {
                    return Err(RiverError::BeyondOpenBoundary {
                        name: river.name.clone(),
                        x: p[0],
                        y: p[1],
                        distance,
                    });
                }
            }
        }
        // The bounding box is never farther than the element: skip the
        // elements whose box is beyond the nearest so far
        let (mut distance, mut nearest) = (f64::INFINITY, None);
        for &(k, [x0, y0, x1, y1]) in &self.water {
            let dx = (x0 - p[0]).max(p[0] - x1).max(0.0);
            let dy = (y0 - p[1]).max(p[1] - y1).max(0.0);
            if dx.hypot(dy) >= distance {
                continue;
            }
            let d = distance_to_quad(&self.mesh.element_vertices(k), p);
            if d < distance {
                (distance, nearest) = (d, Some(k));
            }
        }
        match nearest {
            Some(k) if distance <= max_snap => Ok((k, distance)),
            _ => Err(RiverError::FarFromWater {
                name: river.name.clone(),
                x: p[0],
                y: p[1],
                distance,
                max_snap,
            }),
        }
    }
}

/// Distance from `p` to the quadrilateral with straight edges `vertices`
/// (counter-clockwise), for a point outside it.
fn distance_to_quad(vertices: &[[f64; 2]; 4], p: [f64; 2]) -> f64 {
    (0..4)
        .map(|e| distance_to_segment(vertices[e], vertices[(e + 1) % 4], p))
        .fold(f64::INFINITY, f64::min)
}

/// Distance from `p` to the segment from `a` to `b`.
fn distance_to_segment(a: [f64; 2], b: [f64; 2], p: [f64; 2]) -> f64 {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let length2 = dx * dx + dy * dy;
    let t = if length2 > 0.0 {
        (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (p[0] - a[0] - t * dx).hypot(p[1] - a[1] - t * dy)
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
        self.add_volume_rates(element.element.as_usize(), element.time, h);
    }

    fn name(&self) -> &'static str {
        "rivers"
    }

    fn changes_mass(&self) -> bool {
        true
    }

    fn add_mass_element(&self, element: &ElementSources<'_>, h: &mut [f64]) {
        self.add_volume_rates(element.element.as_usize(), element.time, h);
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
    /// The rate of the depth (m/s) the rivers add at node `node` of element
    /// `k` over the step.
    pub fn volume_rate(&self, k: usize, node: usize) -> f64 {
        self.rivers
            .in_element(k)
            .iter()
            .map(|&i| self.discharge[i] * self.rivers.share(i)[node])
            .sum()
    }

    /// Add the layer volume sources `s_l = Σ w_l Q̄ d_j/∫_k d dA` (m/s) at
    /// node `node` (`j`) of element `k` to `out` (one per level, bed first).
    pub fn add_layer_rates(&self, k: usize, node: usize, out: &mut [f64]) {
        for &i in self.rivers.in_element(k) {
            let rate = self.discharge[i] * self.rivers.share(i)[node];
            for (o, &w) in out.iter_mut().zip(self.rivers.level_weights(i, node)) {
                *o += w * rate;
            }
        }
    }

    /// Add the inventory sources `s_l·C_river` of a tracer to `rhs`
    /// (`[element][node][level]`, `n_nodes` per element) in `elements`, with
    /// the river water's concentration `concentration(river, element)`.
    pub fn add_tracer_sources(
        &self,
        rhs: &mut [f64],
        n_nodes: usize,
        elements: Blocks,
        concentration: impl Fn(&River, usize) -> f64,
    ) {
        let nl = self.rivers.n_levels();
        for i in 0..self.rivers.len() {
            let k = self.rivers.element(i).as_usize();
            if !elements.contains(k) {
                continue;
            }
            let (q, share) = (self.discharge[i], self.rivers.share(i));
            let c = concentration(self.rivers.river(i), k);
            let columns = rhs[k * n_nodes * nl..(k + 1) * n_nodes * nl].chunks_exact_mut(nl);
            for (node, column) in columns.enumerate() {
                let rate = q * share[node];
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
    /// between; [`crate::solver::rhs::w_cell_thicknesses`]), in `elements`.
    pub fn add_w_point_sources_at_own_value(
        &self,
        rhs: &mut [f64],
        field: &[f64],
        n_nodes: usize,
        elements: Blocks,
    ) {
        let nw = self.rivers.n_levels() + 1;
        for i in 0..self.rivers.len() {
            let k = self.rivers.element(i).as_usize();
            if !elements.contains(k) {
                continue;
            }
            let (q, share) = (self.discharge[i], self.rivers.share(i));
            let columns = k * n_nodes * nw..(k + 1) * n_nodes * nw;
            for (node, (r, phi)) in rhs[columns.clone()]
                .chunks_exact_mut(nw)
                .zip(field[columns].chunks_exact(nw))
                .enumerate()
            {
                let weights = self.rivers.level_weights(i, node);
                let rate = q * share[node];
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

    #[test]
    fn a_profile_reads_from_its_text() {
        let parse = |text: &str| text.parse::<RiverProfile>();
        assert_eq!(parse("uniform").unwrap(), RiverProfile::DepthUniform);
        assert_eq!(
            parse("fraction:0.2").unwrap(),
            RiverProfile::TopFraction(0.2)
        );
        assert_eq!(parse("top:5").unwrap(), RiverProfile::TopMetres(5.0));
        assert_eq!(
            parse("tapered:3").unwrap(),
            RiverProfile::TaperedTopMetres(3.0)
        );
        for bad in [
            "",
            "tapered",
            "tapered:x",
            "bottom:3",
            "fraction:1.5",
            "top:0",
            "top:-2",
        ] {
            assert!(parse(bad).is_err(), "{bad:?} parsed");
        }
    }

    fn mesh() -> (Mesh2D, DGOperators2D, GeometricFactors2D, Bathymetry2D) {
        let mesh = Mesh2D::uniform_rectangle(0.0, 400.0, 0.0, 200.0, 4, 2);
        let ops = DGOperators2D::new(2);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let bed = Bathymetry2D::constant(mesh.n_elements, ops.n_nodes, -10.0);
        (mesh, ops, geom, bed)
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
        let (mesh, _, geom, bed) = mesh();
        let rivers = vec![
            River::new("inside", [150.0, 50.0], 10.0),
            River::new("same element", [160.0, 60.0], 5.0),
            // 30 m north of the top boundary, above element (3, 1)
            River::new("outside", [350.0, 230.0], 1.0),
        ];
        let sources = RiverSources::new(rivers, &mesh, &geom, &bed, 50.0).unwrap();
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
        let mut rate = vec![0.0; sources.share(0).len()];
        sources.add_volume_rates(inside.as_usize(), 0.0, &mut rate);
        assert!(
            rate.iter().all(|r| (r - 15.0 / 1e4).abs() < 1e-15),
            "{rate:?}"
        );
        assert!((sources.total_discharge(0.0) - 16.0).abs() < 1e-12);

        let far = vec![River::new("far", [350.0, 300.0], 1.0)];
        assert!(matches!(
            RiverSources::new(far, &mesh, &geom, &bed, 50.0),
            Err(RiverError::FarFromWater { .. })
        ));
        let negative = vec![River::new("dry", [10.0, 10.0], -1.0)];
        assert!(matches!(
            RiverSources::new(negative, &mesh, &geom, &bed, 50.0),
            Err(RiverError::NegativeDischarge { .. })
        ));

        // The ones that can be placed, and why the others cannot
        let rivers = vec![
            River::new("far", [350.0, 300.0], 1.0),
            River::new("inside", [150.0, 50.0], 10.0),
            River::new("dry", [10.0, 10.0], -1.0),
            River::new("outside", [350.0, 230.0], 1.0),
        ];
        let (sources, errors) = RiverSources::on_mesh(rivers, &mesh, &geom, &bed, 50.0);
        assert_eq!(sources.len(), 2);
        assert_eq!(sources.river(0).name, "inside");
        assert_eq!(sources.element(0), inside);
        assert_eq!(sources.river(1).name, "outside");
        assert_eq!(sources.element(1), corner);
        assert_eq!(sources.in_element(corner.as_usize()), &[1]);
        assert!(matches!(
            errors.as_slice(),
            [
                RiverError::FarFromWater { .. },
                RiverError::NegativeDischarge { .. }
            ]
        ));
    }

    /// The source integrates to the discharge over the element, and touches
    /// nothing else.
    #[test]
    fn the_source_integrates_to_the_discharge() {
        let (mesh, ops, geom, bed) = mesh();
        let discharge = RiverSeries::new(vec![0.0, 100.0], vec![4.0, 8.0]).unwrap();
        let rivers = vec![River::new("r", [250.0, 150.0], 0.0).with_discharge(discharge)];
        let sources = RiverSources::new(rivers, &mesh, &geom, &bed, 0.0).unwrap();
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

    /// A river enters an element under water, never one on the coast
    /// (TODO P1.6: the Frøya rivers ponded on the land's 5 m), shared by
    /// depth; a mouth on the coast, on land or off the mesh moves to the
    /// nearest element under water.
    #[test]
    fn a_rivers_water_enters_the_water_not_the_land() {
        let (mesh, ops, geom, _) = mesh();
        // The coast at x = 250 m, across the elements of 200–300 m; land
        // beyond, up to 15 m high at 400 m; 5–15 m deep in 100–200 m
        let bed = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, _| (x - 250.0) / 10.0);
        let locator = PointLocator2D::new(&mesh);
        let at = |x: f64| locator.locate([x, 50.0]).unwrap().element;
        let rivers = vec![
            River::new("sea", [150.0, 50.0], 4.0),
            // In the coastal element, 20 m from the water
            River::new("coast", [220.0, 50.0], 2.0),
            // On land, 150 m from the water
            River::new("inland", [350.0, 50.0], 1.0),
        ];
        let sources = RiverSources::new(rivers.clone(), &mesh, &geom, &bed, 150.0).unwrap();
        for (i, distance) in [0.0, 20.0, 150.0].into_iter().enumerate() {
            assert_eq!(sources.element(i), at(150.0), "{}", rivers[i].name);
            assert!((sources.snap_distance(i) - distance).abs() < 1e-9);
        }
        let too_far = RiverSources::new(rivers, &mesh, &geom, &bed, 149.0);
        assert!(
            matches!(too_far, Err(RiverError::FarFromWater { distance, .. })
            if (distance - 150.0).abs() < 1e-9)
        );

        // Shared by depth: the same dilution rate at every node, the share
        // integrating to one
        let k = at(150.0);
        let share = sources.share(0);
        assert!((geom.integrate_element(k.as_usize(), share) - 1.0).abs() < 1e-15);
        let dilution: Vec<f64> = bed
            .element(k)
            .iter()
            .zip(share)
            .map(|(&b, &s)| s / -b)
            .collect();
        let (lo, hi) = dilution
            .iter()
            .fold((f64::INFINITY, 0.0_f64), |(lo, hi), &r| {
                (lo.min(r), hi.max(r))
            });
        assert!(hi - lo < 1e-14 * hi, "{dilution:?}");
        // The 15 m deep side takes three times the 5 m side's
        let (deep, shallow) = (share[0], share[ops.n_nodes - 1]);
        assert!(deep > 2.9 * shallow, "{share:?}");

        // All 7 m³/s into the element, nothing on the coast
        let mut rate = vec![0.0; ops.n_nodes];
        sources.add_volume_rates(k.as_usize(), 0.0, &mut rate);
        assert!((geom.integrate_element(k.as_usize(), &rate) - 7.0).abs() < 1e-12);
        let mut coast = vec![0.0; ops.n_nodes];
        sources.add_volume_rates(at(220.0).as_usize(), 0.0, &mut coast);
        assert!(coast.iter().all(|&r| r == 0.0));

        // Off the mesh, 50 m from its (dry) edge but 250 m from its water
        let off = vec![River::new("off", [450.0, 50.0], 1.0)];
        assert!(RiverSources::new(off.clone(), &mesh, &geom, &bed, 249.0).is_err());
        let off = RiverSources::new(off, &mesh, &geom, &bed, 250.0).unwrap();
        assert_eq!(off.element(0), k);

        // The levels are laid over the bed the rivers were placed on
        let sigma = SigmaGrid::new(4, UniformStretching);
        let mut placed = sources.clone();
        placed.set_levels(&sigma, &bed).unwrap();
        let flat = Bathymetry2D::constant(mesh.n_elements, ops.n_nodes, -10.0);
        let mut other = sources;
        assert!(matches!(
            other.set_levels(&sigma, &flat),
            Err(RiverError::OtherBed(_))
        ));
    }

    /// A mouth beyond an open boundary lies outside the domain, where the
    /// water entering through the boundary (a parent model's) carries it:
    /// left out, not snapped in. One beyond a wall still snaps.
    #[test]
    fn a_mouth_beyond_an_open_boundary_is_left_out() {
        use crate::mesh::BoundaryTag::{Open, Wall};
        // Open to the east, walls on the other sides
        let mesh = Mesh2D::uniform_rectangle_with_sides(
            0.0,
            400.0,
            0.0,
            200.0,
            4,
            2,
            [Wall, Open, Wall, Wall],
        );
        let ops = DGOperators2D::new(2);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let bed = Bathymetry2D::constant(mesh.n_elements, ops.n_nodes, -10.0);
        let rivers = vec![
            // 50 m east of the open side
            River::new("east", [450.0, 50.0], 1.0),
            // 30 m north of the wall, 60 m from the open side
            River::new("north", [340.0, 230.0], 2.0),
            River::new("inside", [390.0, 50.0], 3.0),
        ];
        let (sources, errors) = RiverSources::on_mesh(rivers, &mesh, &geom, &bed, 100.0);
        assert_eq!(sources.len(), 2);
        assert_eq!(sources.river(0).name, "north");
        assert!((sources.snap_distance(0) - 30.0).abs() < 1e-9);
        assert_eq!(sources.river(1).name, "inside");
        assert!(matches!(
            errors.as_slice(),
            [RiverError::BeyondOpenBoundary { name, distance, .. }]
                if name == "east" && (distance - 50.0).abs() < 1e-9
        ));
        // However close: the parent carries it
        let near = vec![River::new("near", [400.5, 50.0], 1.0)];
        assert!(matches!(
            RiverSources::new(near, &mesh, &geom, &bed, 1e4),
            Err(RiverError::BeyondOpenBoundary { .. })
        ));
    }

    #[test]
    fn a_series_mean_is_exact_between_its_times() {
        let s = RiverSeries::new(vec![0.0, 10.0, 30.0], vec![1.0, 3.0, 2.0]).unwrap();
        // ∫_0^30 = 20 + 50 = 70
        assert!((s.mean_over(0.0, 30.0) - 70.0 / 30.0).abs() < 1e-15);
        // Held beyond its ends: ∫_-10^40 = 10 + 70 + 20
        assert!((s.mean_over(-10.0, 40.0) - 100.0 / 50.0).abs() < 1e-15);
        // Within one piece: the mean of its ends
        assert!((s.mean_over(2.0, 6.0) - s.at(4.0)).abs() < 1e-15);
        assert_eq!(s.mean_over(5.0, 5.0), s.at(5.0));
    }
}

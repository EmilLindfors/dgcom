//! The case-file schema: plain `serde` structs, strict about unknown keys.
//!
//! Every section and key is documented on its field below, and
//! `cases/tidal_channel/case.toml` is a complete example. Lengths are in
//! metres, times in the unit their key names (`duration_hours`,
//! `interval_minutes`), and each default is stated once, here, in the
//! `default_*` functions.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::CaseError;
use crate::io::LocalProjection;
use crate::mesh::BoundaryTag;
use crate::solver::{LINEAR_CFL_SAFETY, linear_cfl_swe_2d};
use crate::tides::{canonical_name, constituent_period};
use crate::time::ModelClock;
use crate::time::{IntegratorInfo, Multirate, StandardIntegrator};

/// A complete case: what to run, on which domain, forced how, written where.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Case {
    /// Name, length and start of the run.
    pub run: RunSection,
    /// The horizontal mesh and the polynomial order.
    pub mesh: MeshSection,
    /// The bed.
    pub bathymetry: BathymetrySection,
    /// Gravity, rotation, friction, viscosity, wetting and drying.
    #[serde(default)]
    pub physics: PhysicsSection,
    /// The state at time 0.
    #[serde(default)]
    pub initial: InitialSection,
    /// The open boundaries, one `[[boundary]]` per boundary tag; faces with
    /// no entry are walls, and an open tag without one is an error.
    #[serde(default, rename = "boundary", skip_serializing_if = "Vec::is_empty")]
    pub boundaries: Vec<BoundarySection>,
    /// A weather model's wind and pressure (default: none).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub atmosphere: Option<AtmosphereSection>,
    /// Time stepping.
    #[serde(default)]
    pub numerics: NumericsSection,
    /// What is written, and where.
    #[serde(default)]
    pub output: OutputSection,
}

/// `[run]`
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunSection {
    /// Name of the run, used for output file names (default `"case"`).
    #[serde(default = "default_name")]
    pub name: String,
    /// UTC instant of model time 0, e.g. `"2025-06-15T00:00:00Z"`. With it,
    /// tidal phases are Greenwich phase lags and the nodal corrections
    /// apply; without it, phases are relative to time 0.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start: Option<String>,
    /// Length of the run (hours).
    pub duration_hours: f64,
}

/// `[mesh]`: exactly one of `rectangle`, `gmsh` and `geographic`.
///
/// A mesh is in metres. It is placed on the Earth by a local projection
/// (a transverse Mercator about a point, [`crate::io::LocalProjection`]):
/// about `geographic`'s centre, or about `origin` for a rectangle or a Gmsh
/// mesh. A placed mesh can take a bed raster, stations by longitude and
/// latitude, and the beta plane.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeshSection {
    /// Polynomial order of the elements (default 2).
    #[serde(default = "default_order")]
    pub order: usize,
    /// A rectangle of `nx` × `ny` equal quadrilaterals.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rectangle: Option<Rectangle>,
    /// A quadrilateral Gmsh mesh (MSH 4.1 or 2.2), in metres. Boundary
    /// lines take their tag from their physical group's name (`open`,
    /// `tidal`, `river`; walls otherwise): see `docs/gmsh-meshes.md`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gmsh: Option<PathBuf>,
    /// A longitude/latitude box meshed as a rectangle in the projection
    /// about its centre, with open sides.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub geographic: Option<Geographic>,
    /// Where the point x = y = 0 of a rectangle or Gmsh mesh lies on the
    /// Earth (default: not placed).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origin: Option<GeoPoint>,
    /// Drop the elements with no node below mean sea level; their faces to
    /// the kept elements become walls (default true). Land that floods at
    /// high water needs `false`.
    #[serde(default = "default_true")]
    pub drop_land: bool,
}

/// `[mesh.geographic]`
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Geographic {
    /// `[west, south, east, north]` (degrees)
    pub bbox: [f64; 4],
    /// Elements along x (east)
    pub nx: usize,
    /// Elements along y (north)
    pub ny: usize,
}

impl Geographic {
    /// The box's centre.
    pub fn center(&self) -> GeoPoint {
        let [w, s, e, n] = self.bbox;
        GeoPoint {
            lat: 0.5 * (s + n),
            lon: 0.5 * (w + e),
        }
    }
}

/// A point on the Earth: `{ lat = 63.8, lon = 8.6 }` (degrees).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeoPoint {
    /// Latitude (degrees north)
    pub lat: f64,
    /// Longitude (degrees east)
    pub lon: f64,
}

/// `[mesh.rectangle]`
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Rectangle {
    /// `[x0, x1]` (m)
    pub x_m: [f64; 2],
    /// `[y0, y1]` (m)
    pub y_m: [f64; 2],
    /// Elements along x
    pub nx: usize,
    /// Elements along y
    pub ny: usize,
    /// The side at `y = y0` (default `"wall"`).
    #[serde(default)]
    pub south: SideTag,
    /// The side at `x = x1` (default `"wall"`).
    #[serde(default)]
    pub east: SideTag,
    /// The side at `y = y1` (default `"wall"`).
    #[serde(default)]
    pub north: SideTag,
    /// The side at `x = x0` (default `"wall"`).
    #[serde(default)]
    pub west: SideTag,
}

/// A boundary tag in a case file.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SideTag {
    /// A solid wall (no flow through it).
    #[default]
    Wall,
    /// An open boundary.
    Open,
    /// An open boundary for tidal forcing (Gmsh `tidal`).
    Tidal,
    /// An open boundary for a river (Gmsh `river`).
    River,
}

impl SideTag {
    /// Its name in a case file.
    pub fn name(self) -> &'static str {
        match self {
            Self::Wall => "wall",
            Self::Open => "open",
            Self::Tidal => "tidal",
            Self::River => "river",
        }
    }

    /// The mesh's tag.
    pub fn boundary_tag(self) -> BoundaryTag {
        match self {
            Self::Wall => BoundaryTag::Wall,
            Self::Open => BoundaryTag::Open,
            Self::Tidal => BoundaryTag::TidalForcing,
            Self::River => BoundaryTag::River,
        }
    }

    /// The case-file name of a mesh tag, if a case file can name it.
    pub fn from_boundary_tag(tag: BoundaryTag) -> Option<Self> {
        match tag {
            BoundaryTag::Wall => Some(Self::Wall),
            BoundaryTag::Open => Some(Self::Open),
            BoundaryTag::TidalForcing => Some(Self::Tidal),
            BoundaryTag::River => Some(Self::River),
            _ => None,
        }
    }
}

/// `[bathymetry]`: exactly one of `depth_m` and `ramp`. Depths are
/// positive below mean sea level and negative on land, which then wets and
/// dries.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BathymetrySection {
    /// One depth everywhere (m).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub depth_m: Option<f64>,
    /// A depth varying linearly along a line.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ramp: Option<Ramp>,
    /// A GeoTIFF of bed elevations on longitude/latitude (needs a placed
    /// mesh).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raster: Option<Raster>,
    /// Smooth the bed to a slope factor r_x0 = |ΔH|/(H₁ + H₂) of at most
    /// this between neighbouring nodes at least `rx0_min_depth_m` deep,
    /// keeping the volume (default: no smoothing).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub smooth_rx0: Option<f64>,
    /// The shallowest depth `smooth_rx0` acts at (m, default 3).
    #[serde(default = "default_rx0_min_depth")]
    pub rx0_min_depth_m: f64,
    /// Lower the land at the walls of elements that are otherwise water to
    /// the water beside it (`Bathymetry2D::lower_wall_land`; default false).
    /// A simplified coastline leaves a strip of the raster's land in front
    /// of its walls, a fraction of a node spacing wide, that keeps the
    /// element partly dry at every water level. Only elements wet off their
    /// walls below `wall_low_water_m` are lowered.
    #[serde(default)]
    pub lower_wall_land: bool,
    /// The lowest water of `lower_wall_land` (m above mean sea level,
    /// default −2): below the lowest astronomical tide, so that the elements
    /// it lowers stay wet through the tide.
    #[serde(default = "default_wall_low_water")]
    pub wall_low_water_m: f64,
    /// The bed for a 3D model (default: none). Smooths the bed, keeping its
    /// volume, until no element's depth range lets a σ-level cross the
    /// pycnocline between two of its nodes, then raises cliff shores to land
    /// and fills tidal ponds. A 2D run on this bed shows what the 3D bed
    /// does to the tide.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub slopes_3d: Option<Slopes3D>,
}

/// `[bathymetry.slopes_3d]`: the bed a σ-coordinate model needs (TODO
/// P1.3, `notes/3d-validation.md`), after the bed rasters and `smooth_rx0`.
///
/// 1. Within an element, the slope factor r_x0 between two columns at least
///    `thin_depth_m` deep is at most `r_max` wherever the deeper one is below
///    `free_depth_m` (`Bathymetry2D::smooth_element_slopes`). The shores are
///    the wetting and drying's and do not move.
/// 2. With `raise_cliffs`, the shore nodes of elements with a node deeper
///    than `cliff_depth_m`, which a tide would make resolved columns beside
///    the deep ones, are raised to land at `high_water_m`
///    (`Bathymetry2D::raise_cliff_shores`).
/// 3. With `fill_ponds`, the tidal ponds of at most one element's nodes at
///    water levels from `pond_low_water_m` to `high_water_m` (nodes the open
///    water does not reach along the grid lines) are filled to their sills
///    (`Bathymetry2D::fill_tidal_ponds`).
///
/// The defaults are `examples/froya_real_data.rs`'s `slopes3d=on`: a summer
/// pycnocline whose bottom is 19 m deep, and Mausund's lowest (−1.43 m) and
/// highest (under +2 m) astronomical tides. Set the tidal range for another
/// coast.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Slopes3D {
    /// Largest slope factor within an element (default 0.15).
    #[serde(default = "default_slopes_r_max")]
    pub r_max: f64,
    /// Depth above which an element's depth range is free (m, default 19:
    /// the pycnocline's bottom).
    #[serde(default = "default_free_depth")]
    pub free_depth_m: f64,
    /// Columns shallower than this are thin (m, default 0.1, the 3D model's
    /// default): the shores, left alone.
    #[serde(default = "default_thin_depth")]
    pub thin_depth_m: f64,
    /// Raise the shore nodes of cliff elements to land (default true).
    #[serde(default = "default_true")]
    pub raise_cliffs: bool,
    /// An element with a node deeper than this is a cliff (m, default 20).
    #[serde(default = "default_cliff_depth")]
    pub cliff_depth_m: f64,
    /// Fill the tidal ponds (default true).
    #[serde(default = "default_true")]
    pub fill_ponds: bool,
    /// The lowest water at which a pond is looked for (m above mean sea
    /// level, default −1.43).
    #[serde(default = "default_pond_low_water")]
    pub pond_low_water_m: f64,
    /// The highest water (m above mean sea level, default 2): where the
    /// ponds are looked for up to, and the height cliff shores are raised
    /// to.
    #[serde(default = "default_high_water")]
    pub high_water_m: f64,
}

impl Slopes3D {
    /// The library's bound.
    pub fn bound(&self) -> crate::mesh::ElementSlopeBound {
        crate::mesh::ElementSlopeBound {
            r_max: self.r_max,
            free_depth: self.free_depth_m,
        }
    }

    fn validate(&self) -> Result<(), CaseError> {
        let invalid = |m: String| Err(CaseError::Invalid(format!("bathymetry.slopes_3d.{m}")));
        if !(self.r_max > 0.0 && self.r_max < 1.0) {
            return invalid(format!("r_max must lie in (0, 1), not {}", self.r_max));
        }
        if !(non_negative(self.free_depth_m) && self.free_depth_m.is_finite()) {
            return invalid(format!(
                "free_depth_m must be ≥ 0, not {}",
                self.free_depth_m
            ));
        }
        if !(positive(self.thin_depth_m) && self.thin_depth_m.is_finite()) {
            return invalid(format!(
                "thin_depth_m must be positive, not {}",
                self.thin_depth_m
            ));
        }
        if self.raise_cliffs
            && !(self.cliff_depth_m.is_finite() && self.cliff_depth_m > self.thin_depth_m)
        {
            return invalid(format!(
                "cliff_depth_m must be deeper than thin_depth_m ({} m), not {}",
                self.thin_depth_m, self.cliff_depth_m
            ));
        }
        if !self.high_water_m.is_finite() {
            return invalid("high_water_m must be finite".into());
        }
        if self.fill_ponds
            && !(self.pond_low_water_m.is_finite() && self.pond_low_water_m < self.high_water_m)
        {
            return invalid(format!(
                "pond_low_water_m must be below high_water_m ({} m), not {}",
                self.high_water_m, self.pond_low_water_m
            ));
        }
        Ok(())
    }
}

/// `[bathymetry.raster]`: a GeoTIFF of bed elevation (m, negative under
/// water) on a longitude/latitude grid. A raster bed also has its isolated
/// wet nodes (water one node wide, which the mesh does not resolve) raised
/// to the lowest of their neighbours.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Raster {
    /// The GeoTIFF.
    pub file: PathBuf,
    /// What its pixels hold (default `"bathymetry"`).
    #[serde(default)]
    pub kind: RasterKind,
    /// For `"bathymetry"`: a shapefile of land polygons (e.g. GSHHS) for
    /// the land mask, at 4 × 4 cells per pixel (default: land where the
    /// raster is dry or has no data).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coastline: Option<PathBuf>,
    /// The bed of land (m above mean sea level, default 5): the elevation of
    /// a `"bathymetry"` raster's land, the cap of a `"topobathy"` raster's.
    #[serde(default = "default_land_elevation")]
    pub land_elevation_m: f64,
    /// For `"topobathy"`: fill the pixels that are exactly 0 from their
    /// surroundings, where the raster marks missing depths with the water
    /// surface (Kartverket's topobathy model does; default false).
    #[serde(default)]
    pub fill_zero_holes: bool,
    /// How the raster reaches the nodes (default `"projected"`).
    #[serde(default)]
    pub sampling: Sampling,
}

/// What a raster's pixels hold.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RasterKind {
    /// Depths below mean sea level; dry and no-data pixels are land.
    #[default]
    Bathymetry,
    /// A complete elevation model, land heights included (no pixel may
    /// lack data).
    Topobathy,
}

/// How a raster reaches the nodes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Sampling {
    /// The L2 projection onto each element's polynomials, at the raster's
    /// resolution: an element's mean bed is the raster's.
    #[default]
    Projected,
    /// The raster's value at each node.
    Point,
}

/// `[bathymetry.ramp]`: the depth goes linearly from `depth_m[0]` at
/// `from_m` to `depth_m[1]` at `to_m`, along the line between them, and is
/// constant beyond either end.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Ramp {
    /// Start point `[x, y]` (m)
    pub from_m: [f64; 2],
    /// End point `[x, y]` (m)
    pub to_m: [f64; 2],
    /// Depths at the start and the end (m)
    pub depth_m: [f64; 2],
}

impl Ramp {
    /// Depth at `(x, y)` (m).
    pub fn depth(&self, x: f64, y: f64) -> f64 {
        let [x0, y0] = self.from_m;
        let (dx, dy) = (self.to_m[0] - x0, self.to_m[1] - y0);
        let s = (((x - x0) * dx + (y - y0) * dy) / (dx * dx + dy * dy)).clamp(0.0, 1.0);
        self.depth_m[0] + s * (self.depth_m[1] - self.depth_m[0])
    }
}

/// `[physics]`
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhysicsSection {
    /// Gravitational acceleration (m/s², default 9.81).
    #[serde(default = "default_gravity")]
    pub gravity: f64,
    /// Rotation (default none).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coriolis: Option<Coriolis>,
    /// Bed friction (default none), applied point-implicitly.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub friction: Option<Friction>,
    /// Constant horizontal eddy viscosity (m²/s, default 0).
    #[serde(default)]
    pub viscosity_m2_s: f64,
    /// Smagorinsky coefficient of the horizontal viscosity (default 0: off).
    #[serde(default)]
    pub smagorinsky: f64,
    /// Wetting and drying: the positivity-preserving wet/dry scheme
    /// (default true). Off, the depth must stay positive everywhere.
    #[serde(default = "default_true")]
    pub wetting_drying: bool,
}

impl Default for PhysicsSection {
    fn default() -> Self {
        Self {
            gravity: default_gravity(),
            coriolis: None,
            friction: None,
            viscosity_m2_s: 0.0,
            smagorinsky: 0.0,
            wetting_drying: true,
        }
    }
}

/// `coriolis = { f_plane = 1.2e-4 }`, `coriolis = { latitude = 63.5 }` or
/// `coriolis = "beta_plane"`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Coriolis {
    /// A constant Coriolis parameter f (s⁻¹).
    FPlane(f64),
    /// The f-plane at a latitude (degrees north): f = 2Ω sin φ.
    Latitude(f64),
    /// f = f₀ + β y about the latitude of the mesh's projection centre
    /// (needs a placed mesh).
    BetaPlane,
}

/// `friction = { manning = 0.025 }` or `friction = { drag_coefficient = 0.0025 }`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Friction {
    /// Manning's n (s/m^{1/3}).
    Manning(f64),
    /// A constant quadratic drag coefficient C_D (τ_b = ρ C_D |u| u).
    DragCoefficient(f64),
}

/// `[initial]`
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InitialSection {
    /// Still water at this surface elevation (m, default 0).
    #[serde(default)]
    pub eta_m: f64,
}

/// `[[boundary]]`: the condition on the faces of one tag; exactly one of
/// `still`, `tide` and `atlas`. Every open boundary is a characteristic (Flather-type)
/// condition: it lets outgoing waves leave and takes the external state
/// given here for the incoming one.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoundarySection {
    /// The faces it applies to: `"open"`, `"tidal"` or `"river"`.
    pub tag: SideTag,
    /// Still water outside: a radiation condition.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub still: Option<Still>,
    /// A tide, the same along the whole boundary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tide: Option<Tide>,
    /// A tide varying along the boundary, from a tidal atlas (needs a
    /// placed mesh and `run.start`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub atlas: Option<Atlas>,
    /// Nested in a parent ocean model: its level and depth-mean current at
    /// the boundary (needs a placed mesh and `run.start`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nesting: Option<Nesting>,
    /// The external velocity where the external state is a level only
    /// (`still`, `tide`, or an atlas or parent without currents; default
    /// `"at_rest"`).
    #[serde(default)]
    pub velocity: BoundaryVelocity,
    /// With `[atmosphere]`'s pressure, the fraction of the inverse-barometer
    /// level `−(p − p_ref)/(ρg)` added to the external level (0–2; default
    /// 1, the full response that `still`, `tide` and `atlas` lack, and 0 for
    /// `nesting`, a parent that carries its own). Without it the boundary
    /// pins the sea to the wrong level under a low and drives a current
    /// through itself. NorKyst-800's ζ carries ≈ 0.22 of the static
    /// response (`notes/norkyst-issues.md` §6), so a fraction may fit a
    /// nested boundary best.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inverse_barometer: Option<f64>,
}

impl BoundarySection {
    /// The fraction of the inverse-barometer level the boundary adds (see
    /// `inverse_barometer`).
    pub fn inverse_barometer_fraction(&self) -> f64 {
        self.inverse_barometer
            .unwrap_or(if self.nesting.is_some() { 0.0 } else { 1.0 })
    }
}

/// `[atmosphere]`: the 10 m wind and the sea-level pressure of a weather
/// model (MET Nordic, MEPS, AROME-Arctic or ERA5 NetCDF, e.g. from
/// `met_forcing_subset`; [`crate::source::GriddedAtmosphere2D`] on an
/// [`crate::io::AtmosphereReader`]; needs a placed mesh, `run.start` and the
/// `netcdf` feature). The wind stress takes Large & Pond's (1981) drag; the
/// pressure gradient acts on the whole column, and the open boundaries add
/// the inverse-barometer level (`[[boundary]] inverse_barometer`). Both
/// ramp up from zero, with the inverse barometer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AtmosphereSection {
    /// The weather files, joined in time (consecutive forecast runs or
    /// analysis hours). They must cover the mesh and the run.
    pub files: Vec<PathBuf>,
    /// Ramp-up from zero (hours, default 1).
    #[serde(default = "default_ramp_hours")]
    pub ramp_hours: f64,
    /// The wind stress (default true).
    #[serde(default = "default_true")]
    pub wind: bool,
    /// The pressure gradient (default true).
    #[serde(default = "default_true")]
    pub pressure: bool,
}

/// `[boundary.nesting]`: one-way nesting in a parent ocean model
/// ([`crate::boundary::OceanModelState`]). The boundary takes the parent's
/// ζ, ū and v̄ (its tides, coastal current and wind-driven flow), interpolated
/// in space and cubically in time, ramped up from rest. The velocity is
/// scaled to carry the parent's transport over this mesh's bed. A band
/// inside the boundary relaxes the state towards the parent, and the bed is
/// blended to the parent's across it.
///
/// The parent's ζ has no geodetic datum, so its mean comes off: `level_m`
/// if given, else minus the mean level `Z0` of `atlas` (the parent's own
/// harmonic fit over a month), else minus the parent file's own mean over
/// the boundary, which must then be a month long.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Nesting {
    /// The parent's output: NetCDF with ζ, ū and v̄ on a longitude/latitude
    /// grid (`norkyst_nesting_subset`; needs the `netcdf` feature).
    pub file: PathBuf,
    /// Width of the relaxation band inside the boundary (km, default 3; 0
    /// for none, and then no bed blending either).
    #[serde(default = "default_band_km")]
    pub band_km: f64,
    /// Relaxation timescale at the boundary (minutes, default 30). The
    /// relaxation is explicit: keep it well above the time step.
    #[serde(default = "default_band_minutes")]
    pub band_minutes: f64,
    /// Blend the bed to the parent's across the band (default true), so the
    /// two carry the same transport at the boundary.
    #[serde(default = "default_true")]
    pub blend_bed: bool,
    /// The blending leaves the elements with a node shallower than this
    /// alone (m, default 2): the parent's grid has no shores.
    #[serde(default = "default_blend_shore_depth")]
    pub blend_shore_depth_m: f64,
    /// Largest factor the transport scaling may change the parent's
    /// velocity by (default 3; 1 keeps the parent's velocity).
    #[serde(default = "default_transport_ratio")]
    pub transport_scaling: f64,
    /// Largest fraction of the tag's open faces that may become walls where
    /// the parent has no water (default 0.05); more is an error, since a
    /// parent cut for another domain would close the boundary.
    #[serde(default = "default_max_closed")]
    pub max_closed: f64,
    /// The parent is ramped up from rest over this time (hours, default 1).
    #[serde(default = "default_ramp_hours")]
    pub ramp_hours: f64,
    /// Added to the parent's ζ (m; default: from `atlas`, or the parent's
    /// own mean).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub level_m: Option<f64>,
    /// The parent's own tidal atlas (`norkyst_boundary_tides`): its `Z0`
    /// sets the datum, and it is what `correction` or `gauge` corrects.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub atlas: Option<PathBuf>,
    /// Largest distance from a nested node to a point of `atlas` (m, default
    /// 5000, plus the band's width).
    #[serde(default = "default_coverage_m")]
    pub atlas_coverage_m: f64,
    /// Correct the parent's tide (default `"none"`): the parent gets
    /// (corrected atlas − atlas) added, which keeps its residual flow.
    #[serde(default)]
    pub correction: AtlasCorrection,
    /// Correct the parent's tide at a tide gauge instead (default: no
    /// gauge).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gauge: Option<GaugeCorrection>,
}

/// The velocity outside an open boundary whose external state is a level
/// only (`boundary.velocity`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryVelocity {
    /// Water at rest outside. The boundary's level equals the given one only
    /// at an antinode of a standing tide; a wave entering from outside
    /// arrives at half the given amplitude.
    #[default]
    AtRest,
    /// A wave entering from still water: a progressive tide arrives at the
    /// given amplitude, and outgoing waves still leave.
    IncomingWave,
}

/// `still = { eta_m = 0.0 }`
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Still {
    /// Surface elevation outside (m, default 0).
    #[serde(default)]
    pub eta_m: f64,
}

/// `[boundary.tide]`
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tide {
    /// The constituents.
    pub constituents: Vec<Constituent>,
    /// Mean level (m, default 0).
    #[serde(default)]
    pub mean_m: f64,
    /// The tide is ramped up from 0 over this time (hours, default 1).
    #[serde(default = "default_ramp_hours")]
    pub ramp_hours: f64,
}

/// `[boundary.atlas]`: harmonic constants of the level, and of the
/// depth-mean current if it has them, at points along the boundary (the
/// text format of `examples/norkyst_boundary_tides.rs`: lines of
/// `lon lat depth name eta_amp eta_lag [u_amp u_lag v_amp v_lag]`, lags in
/// degrees Greenwich, `Z0` the mean level). Each boundary node takes the
/// constants of the atlas near it, with the nodal corrections of
/// `run.start`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Atlas {
    /// The atlas file.
    pub file: PathBuf,
    /// Largest distance from a boundary node to an atlas point (m, default
    /// 5000). Open faces farther away become walls: water the atlas's
    /// source does not resolve has no tide to force.
    #[serde(default = "default_coverage_m")]
    pub coverage_m: f64,
    /// Largest fraction of the tag's open faces that may become walls for
    /// lack of coverage (default 0.05); more is an error, since an atlas
    /// made for another domain would close the boundary.
    #[serde(default = "default_max_closed")]
    pub max_closed: f64,
    /// Correct the atlas before use (default `"none"`).
    #[serde(default)]
    pub correction: AtlasCorrection,
    /// Correct the atlas at a tide gauge instead (default: no gauge).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gauge: Option<GaugeCorrection>,
    /// Scale the atlas velocity to carry its source's transport over this
    /// mesh's bed, by at most this factor (default: no scaling).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transport_scaling: Option<f64>,
    /// The tide is ramped up from 0 over this time (hours, default 1).
    #[serde(default = "default_ramp_hours")]
    pub ramp_hours: f64,
}

/// `[boundary.atlas.gauge]`: correct the atlas at a tide gauge
/// ([`crate::boundary::TidalAtlas::correct_at_gauge`]). The gauge's whole
/// record is fitted ([`crate::analysis::HarmonicAnalysis::fit_gauge_record`]).
/// `gains` are scaled by the gauge's constant over the atlas source's at
/// the gauge, which keeps the source's spatial structure and fixes it at
/// the gauge. `ratios` are then re-inferred from their neighbours with the
/// gauge's ratios.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GaugeCorrection {
    /// The gauge's record, with its position in the header (the format of
    /// `scripts/kartverket_gauge.sh`).
    pub record: PathBuf,
    /// The atlas's source at the gauge: an atlas from the same model with a
    /// point near the gauge (`norkyst_boundary_tides points=…`).
    pub source: PathBuf,
    /// Largest distance from the gauge to that point (m, default 3000).
    #[serde(default = "default_max_offset_m")]
    pub max_offset_m: f64,
    /// Constituents scaled at the gauge (default K1, O1, S2).
    #[serde(default = "default_gains")]
    pub gains: Vec<String>,
    /// Constituents re-inferred with the gauge's ratios, of P1, K2, N2 and
    /// Q1 (default N2, Q1, P1, K2).
    #[serde(default = "default_ratios")]
    pub ratios: Vec<String>,
}

/// A correction of a tidal atlas before use.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AtlasCorrection {
    /// The atlas as it is.
    #[default]
    None,
    /// K2 and P1 at their equilibrium ratios to S2 and K1, and the minor
    /// semidiurnals 2N2, μ2, ν2, λ2 and L2 by admittance
    /// ([`crate::boundary::TidalAtlas::infer_from_theory`]): no gauge needed.
    /// Needs M2, S2, N2 and K1 in the atlas.
    Theory,
}

/// One constituent: `{ name = "M2", amplitude_m = 1.0, phase_deg = 0.0 }`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Constituent {
    /// Its name (M2, S2, N2, K2, K1, O1, P1, Q1, M4, …).
    pub name: String,
    /// Amplitude (m)
    pub amplitude_m: f64,
    /// Phase lag (degrees, default 0): the Greenwich phase lag with
    /// `run.start`, else the lag relative to time 0.
    #[serde(default)]
    pub phase_deg: f64,
}

/// `[numerics]`
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NumericsSection {
    /// The time stepper (default `"ssp_rk43"`).
    #[serde(default)]
    pub integrator: Integrator,
    /// CFL number of the step (default: 0.9 times the linear stability
    /// limit of the order and the stepper, e.g. 1.36 for order 2 with
    /// SSP-RK(4,3); 1 beyond order 4). The wet/dry positivity bound
    /// shortens the step further where it binds. The resolved case
    /// (`case.resolved.toml`) states the value used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cfl: Option<f64>,
    /// Upper bound of the time step (s, default none).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dt_max_s: Option<f64>,
    /// Local time stepping: each element steps at its own power-of-two
    /// fraction of the step, over up to this many levels (default 0: one
    /// step for all; at most 16).
    #[serde(default)]
    pub local_time_stepping_levels: usize,
}

/// The time stepper.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Integrator {
    /// SSP-RK(4,3): four stages, about twice SSP-RK3's step under the
    /// wet/dry positivity bound.
    #[default]
    SspRk43,
    /// SSP-RK3 (Shu–Osher): three stages.
    SspRk3,
}

impl Integrator {
    /// The crate's stepper.
    pub fn standard(self) -> StandardIntegrator {
        match self {
            Self::SspRk43 => StandardIntegrator::SSPRK43,
            Self::SspRk3 => StandardIntegrator::SSPRK3,
        }
    }
}

impl NumericsSection {
    /// The CFL number: `cfl`, else its default at polynomial order `order`.
    pub fn cfl_at(&self, order: usize) -> f64 {
        self.cfl.unwrap_or_else(|| {
            let scheme = self
                .integrator
                .standard()
                .ssp_scheme()
                .expect("an SSP stepper");
            linear_cfl_swe_2d(order, scheme).map_or(1.0, |limit| LINEAR_CFL_SAFETY * limit)
        })
    }
}

/// `[output]`
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputSection {
    /// Directory for the output, relative to the case file (default
    /// `output/<run.name>`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub directory: Option<PathBuf>,
    /// Time between outputs of the fields (snapshot, VTK, NetCDF; minutes,
    /// default 60).
    #[serde(default = "default_interval_minutes")]
    pub interval_minutes: f64,
    /// Time between station records (minutes, default `interval_minutes`).
    /// One of the two intervals must be a whole multiple of the other. The
    /// run's steps land on both (as ROMS's `NSTA` and `NHIS`), so the
    /// intervals are part of the run, to round-off.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub station_interval_minutes: Option<f64>,
    /// The whole state in the viewer's snapshot format,
    /// `<name>.dgsnap` (default true).
    #[serde(default = "default_true")]
    pub snapshot: bool,
    /// One VTK file per output, `vtk/<name>_NNNN.vtu`, for ParaView
    /// (default false).
    #[serde(default)]
    pub vtk: bool,
    /// CF NetCDF, `<name>.nc` (default false; needs the `netcdf` feature).
    #[serde(default)]
    pub netcdf: bool,
    /// Points where the surface and the current are recorded every
    /// `station_interval_minutes`, into `stations.csv`.
    #[serde(default, rename = "station", skip_serializing_if = "Vec::is_empty")]
    pub stations: Vec<Station>,
}

impl OutputSection {
    /// Time between station records (minutes).
    pub fn station_interval(&self) -> f64 {
        self.station_interval_minutes
            .unwrap_or(self.interval_minutes)
    }
}

impl Default for OutputSection {
    fn default() -> Self {
        Self {
            directory: None,
            interval_minutes: default_interval_minutes(),
            station_interval_minutes: None,
            snapshot: true,
            vtk: false,
            netcdf: false,
            stations: Vec::new(),
        }
    }
}

/// `[[output.station]]`
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Station {
    /// Its name in `stations.csv`.
    pub name: String,
    /// x (m), with `y_m`; or `lon` and `lat` on a placed mesh.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub x_m: Option<f64>,
    /// y (m)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub y_m: Option<f64>,
    /// Longitude (degrees east)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lon: Option<f64>,
    /// Latitude (degrees north)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lat: Option<f64>,
    /// Sample in open water (m, default 3): at the station if every node of
    /// its element is at least this deep below mean sea level, else at the
    /// nearest point of such an element within `max_offset_m`
    /// (`Probe2D::in_open_water`). A tide gauge at the shore is often in an
    /// element that is partly land, whose surface mixes in the land's bed.
    /// 0 samples at the station as it is.
    #[serde(default = "default_open_water_depth")]
    pub open_water_depth_m: f64,
    /// How far a station may be moved into open water (m, default 3000).
    #[serde(default = "default_max_offset")]
    pub max_offset_m: f64,
}

fn default_open_water_depth() -> f64 {
    3.0
}
fn default_max_offset() -> f64 {
    3000.0
}

fn default_name() -> String {
    "case".to_string()
}
fn default_order() -> usize {
    2
}
fn default_gravity() -> f64 {
    9.81
}
fn default_true() -> bool {
    true
}
fn default_band_km() -> f64 {
    3.0
}
fn default_band_minutes() -> f64 {
    30.0
}
fn default_blend_shore_depth() -> f64 {
    2.0
}
fn default_transport_ratio() -> f64 {
    3.0
}
fn default_ramp_hours() -> f64 {
    1.0
}
fn default_max_offset_m() -> f64 {
    3000.0
}
fn default_gains() -> Vec<String> {
    ["K1", "O1", "S2"].map(String::from).to_vec()
}
fn default_ratios() -> Vec<String> {
    ["N2", "Q1", "P1", "K2"].map(String::from).to_vec()
}
fn default_coverage_m() -> f64 {
    5000.0
}
fn default_max_closed() -> f64 {
    0.05
}
fn default_rx0_min_depth() -> f64 {
    3.0
}
fn default_wall_low_water() -> f64 {
    -2.0
}
fn default_slopes_r_max() -> f64 {
    0.15
}
fn default_free_depth() -> f64 {
    19.0
}
fn default_thin_depth() -> f64 {
    0.1 // `Hydrostatic3D::DEFAULT_MIN_COLUMN_DEPTH`
}
fn default_cliff_depth() -> f64 {
    20.0
}
fn default_pond_low_water() -> f64 {
    -1.43
}
fn default_high_water() -> f64 {
    2.0
}
fn default_land_elevation() -> f64 {
    5.0
}
fn default_interval_minutes() -> f64 {
    60.0
}

/// Whether `x` is positive (NaN is not).
fn positive(x: f64) -> bool {
    x > 0.0
}

/// Whether `x` is 0 or more (false for NaN).
fn non_negative(x: f64) -> bool {
    x >= 0.0
}

/// Polynomial orders the case layer accepts.
const ORDERS: std::ops::RangeInclusive<usize> = 1..=8;

impl Nesting {
    /// Check the section's values.
    fn validate(&self) -> Result<(), CaseError> {
        let invalid = |msg: &str| Err(CaseError::Invalid(format!("boundary.nesting.{msg}")));
        if !(non_negative(self.band_km) && self.band_km.is_finite()) {
            return invalid("band_km must be ≥ 0");
        }
        if !positive(self.band_minutes) {
            return invalid("band_minutes must be positive");
        }
        if !non_negative(self.blend_shore_depth_m) {
            return invalid("blend_shore_depth_m must be ≥ 0");
        }
        if !(self.transport_scaling >= 1.0 && self.transport_scaling.is_finite()) {
            return invalid("transport_scaling must be ≥ 1");
        }
        if !(0.0..=1.0).contains(&self.max_closed) {
            return invalid("max_closed must be in [0, 1]");
        }
        if !non_negative(self.ramp_hours) {
            return invalid("ramp_hours must be ≥ 0");
        }
        if self.level_m.is_some_and(|level| !level.is_finite()) {
            return invalid("level_m must be finite");
        }
        if !positive(self.atlas_coverage_m) {
            return invalid("atlas_coverage_m must be positive");
        }
        if self.atlas.is_none()
            && (self.correction != AtlasCorrection::None || self.gauge.is_some())
        {
            return Err(CaseError::Invalid(
                "boundary.nesting: correcting the parent's tide needs its tidal atlas (`atlas`)"
                    .into(),
            ));
        }
        validate_gauge("boundary.nesting", self.correction, self.gauge.as_ref())
    }
}

/// Check a tide correction of `section` (`boundary.atlas` or
/// `boundary.nesting`): `correction` or `gauge`, and the gauge's terms.
fn validate_gauge(
    section: &str,
    correction: AtlasCorrection,
    gauge: Option<&GaugeCorrection>,
) -> Result<(), CaseError> {
    let Some(gauge) = gauge else {
        return Ok(());
    };
    let invalid = |msg: String| Err(CaseError::Invalid(msg));
    if correction != AtlasCorrection::None {
        return invalid(format!(
            "{section}: a gauge correction or `correction`, not both"
        ));
    }
    if !positive(gauge.max_offset_m) {
        return invalid(format!("{section}.gauge.max_offset_m must be positive"));
    }
    for name in &gauge.gains {
        if canonical_name(name).is_none() {
            return invalid(format!(
                "{section}.gauge.gains: unknown constituent {name:?}"
            ));
        }
    }
    for name in &gauge.ratios {
        if !["P1", "K2", "N2", "Q1"].contains(&name.as_str()) {
            return invalid(format!(
                "{section}.gauge.ratios: {name:?} is not P1, K2, N2 or Q1"
            ));
        }
    }
    Ok(())
}

impl Case {
    /// Check what the types cannot: one choice where the schema offers
    /// several, positive sizes and times, known constituents.
    pub fn validate(&self) -> Result<(), CaseError> {
        let invalid = |msg: String| Err(CaseError::Invalid(msg));
        if !positive(self.run.duration_hours) {
            return invalid(format!(
                "run.duration_hours must be positive, not {}",
                self.run.duration_hours
            ));
        }
        self.clock()?;
        if !ORDERS.contains(&self.mesh.order) {
            return invalid(format!(
                "mesh.order must be {}–{}, not {}",
                ORDERS.start(),
                ORDERS.end(),
                self.mesh.order
            ));
        }
        let m = &self.mesh;
        match (&m.rectangle, &m.gmsh, &m.geographic) {
            (Some(r), None, None) => {
                if !(r.x_m[1] > r.x_m[0] && r.y_m[1] > r.y_m[0]) {
                    return invalid(format!(
                        "mesh.rectangle needs x_m[0] < x_m[1] and y_m[0] < y_m[1], not {:?} and {:?}",
                        r.x_m, r.y_m
                    ));
                }
                if r.nx == 0 || r.ny == 0 {
                    return invalid("mesh.rectangle needs nx and ny of at least 1".into());
                }
            }
            (None, Some(_), None) => {}
            (None, None, Some(g)) => {
                let [w, s, e, n] = g.bbox;
                if !(w < e && s < n && (-90.0..=90.0).contains(&s) && (-90.0..=90.0).contains(&n)) {
                    return invalid(format!(
                        "mesh.geographic.bbox must be [west, south, east, north], not {:?}",
                        g.bbox
                    ));
                }
                if g.nx == 0 || g.ny == 0 {
                    return invalid("mesh.geographic needs nx and ny of at least 1".into());
                }
                if m.origin.is_some() {
                    return invalid(
                        "mesh.origin places a rectangle or Gmsh mesh; a geographic mesh is \
                         placed about its box's centre"
                            .into(),
                    );
                }
            }
            _ => {
                return invalid(
                    "[mesh] needs exactly one of `rectangle`, `gmsh` and `geographic`".into(),
                );
            }
        }
        if let Some(o) = m.origin
            && !((-90.0..=90.0).contains(&o.lat) && o.lon.is_finite())
        {
            return invalid(format!("mesh.origin {o:?} is not on the Earth"));
        }
        let placed = self.placed();
        let b = &self.bathymetry;
        match (&b.depth_m, &b.ramp, &b.raster) {
            (Some(d), None, None) if d.is_finite() => {}
            (Some(d), None, None) => return invalid(format!("bathymetry.depth_m is {d}")),
            (None, Some(r), None) => {
                if r.from_m == r.to_m {
                    return invalid("bathymetry.ramp needs from_m and to_m apart".into());
                }
            }
            (None, None, Some(r)) => {
                if !placed {
                    return invalid(
                        "bathymetry.raster needs a placed mesh: mesh.geographic, or mesh.origin"
                            .into(),
                    );
                }
                if !positive(r.land_elevation_m) {
                    return invalid(format!(
                        "bathymetry.raster.land_elevation_m must be positive, not {}",
                        r.land_elevation_m
                    ));
                }
                if r.kind == RasterKind::Topobathy && r.coastline.is_some() {
                    return invalid(
                        "bathymetry.raster.coastline is for a \"bathymetry\" raster; a \
                         \"topobathy\" raster has its own land"
                            .into(),
                    );
                }
                if r.kind == RasterKind::Bathymetry && r.fill_zero_holes {
                    return invalid(
                        "bathymetry.raster.fill_zero_holes is for a \"topobathy\" raster".into(),
                    );
                }
            }
            _ => {
                return invalid(
                    "[bathymetry] needs exactly one of `depth_m`, `ramp` and `raster`".into(),
                );
            }
        }
        if let Some(r) = b.smooth_rx0
            && !positive(r)
        {
            return invalid(format!("bathymetry.smooth_rx0 must be positive, not {r}"));
        }
        if b.rx0_min_depth_m < 0.0 {
            return invalid("bathymetry.rx0_min_depth_m must be ≥ 0".into());
        }
        if !b.wall_low_water_m.is_finite() {
            return invalid("bathymetry.wall_low_water_m must be finite".into());
        }
        if let Some(s) = &b.slopes_3d {
            s.validate()?;
        }
        if let Some(a) = &self.atmosphere {
            if !self.placed() {
                return invalid(
                    "[atmosphere] needs a placed mesh: mesh.geographic, or mesh.origin".into(),
                );
            }
            if self.run.start.is_none() {
                return invalid(
                    "[atmosphere] needs run.start, the weather's instant of time 0".into(),
                );
            }
            if a.files.is_empty() {
                return invalid("atmosphere.files needs at least one file".into());
            }
            if !(non_negative(a.ramp_hours) && a.ramp_hours.is_finite()) {
                return invalid(format!(
                    "atmosphere.ramp_hours must be ≥ 0, not {}",
                    a.ramp_hours
                ));
            }
            if !a.wind && !a.pressure {
                return invalid(
                    "[atmosphere] with neither wind nor pressure forces nothing".into(),
                );
            }
        }
        let p = &self.physics;
        if !positive(p.gravity) {
            return invalid(format!(
                "physics.gravity must be positive, not {}",
                p.gravity
            ));
        }
        if p.viscosity_m2_s < 0.0 || p.smagorinsky < 0.0 {
            return invalid("physics.viscosity_m2_s and physics.smagorinsky must be ≥ 0".into());
        }
        match p.friction {
            Some(Friction::Manning(n)) if !positive(n) => {
                return invalid(format!(
                    "physics.friction.manning must be positive, not {n}"
                ));
            }
            Some(Friction::DragCoefficient(c)) if !positive(c) => {
                return invalid(format!(
                    "physics.friction.drag_coefficient must be positive, not {c}"
                ));
            }
            _ => {}
        }
        match p.coriolis {
            Some(Coriolis::Latitude(lat)) if !(-90.0..=90.0).contains(&lat) => {
                return invalid(format!(
                    "physics.coriolis.latitude must be in [-90, 90], not {lat}"
                ));
            }
            Some(Coriolis::BetaPlane) if !placed => {
                return invalid(
                    "physics.coriolis = \"beta_plane\" needs a placed mesh: mesh.geographic, \
                     or mesh.origin"
                        .into(),
                );
            }
            _ => {}
        }
        let mut tags = Vec::new();
        for b in &self.boundaries {
            if b.tag == SideTag::Wall {
                return invalid(
                    "[[boundary]] is for open boundaries; faces without one are walls already"
                        .into(),
                );
            }
            if tags.contains(&b.tag) {
                return invalid(format!("two [[boundary]] entries for tag {:?}", b.tag));
            }
            tags.push(b.tag);
            match (&b.still, &b.tide, &b.atlas, &b.nesting) {
                (Some(_), None, None, None) => {}
                (None, None, Some(atlas), None) => {
                    if !placed {
                        return invalid(format!(
                            "[[boundary]] {:?}: an atlas needs a placed mesh: mesh.geographic, \
                             or mesh.origin",
                            b.tag.name()
                        ));
                    }
                    if self.run.start.is_none() {
                        return invalid(format!(
                            "[[boundary]] {:?}: an atlas's phases are Greenwich lags, which \
                             need run.start",
                            b.tag.name()
                        ));
                    }
                    if !positive(atlas.coverage_m) {
                        return invalid("boundary.atlas.coverage_m must be positive".into());
                    }
                    if !(0.0..=1.0).contains(&atlas.max_closed) {
                        return invalid("boundary.atlas.max_closed must be in [0, 1]".into());
                    }
                    if let Some(r) = atlas.transport_scaling
                        && !positive(r)
                    {
                        return invalid("boundary.atlas.transport_scaling must be positive".into());
                    }
                    if atlas.ramp_hours < 0.0 {
                        return invalid("boundary.atlas.ramp_hours must be ≥ 0".into());
                    }
                    validate_gauge("boundary.atlas", atlas.correction, atlas.gauge.as_ref())?;
                }
                (None, None, None, Some(nesting)) => {
                    if !placed {
                        return invalid(format!(
                            "[[boundary]] {:?}: nesting needs a placed mesh: mesh.geographic, \
                             or mesh.origin",
                            b.tag.name()
                        ));
                    }
                    if self.run.start.is_none() {
                        return invalid(format!(
                            "[[boundary]] {:?}: nesting needs run.start, the parent's instant \
                             of time 0",
                            b.tag.name()
                        ));
                    }
                    nesting.validate()?;
                }
                (None, Some(tide), None, None) => {
                    if tide.constituents.is_empty() {
                        return invalid("boundary.tide needs at least one constituent".into());
                    }
                    if tide.ramp_hours < 0.0 {
                        return invalid("boundary.tide.ramp_hours must be ≥ 0".into());
                    }
                    for c in &tide.constituents {
                        let known = canonical_name(&c.name).and_then(constituent_period);
                        if known.is_none() {
                            return invalid(format!("unknown tidal constituent {:?}", c.name));
                        }
                    }
                }
                _ => {
                    return invalid(format!(
                        "[[boundary]] {:?} needs exactly one of `still`, `tide`, `atlas` and \
                         `nesting`",
                        b.tag.name()
                    ));
                }
            }
            if let Some(f) = b.inverse_barometer {
                if !(0.0..=2.0).contains(&f) {
                    return invalid(format!(
                        "[[boundary]] {:?}: inverse_barometer is a fraction of the response, \
                         0–2, not {f}",
                        b.tag.name()
                    ));
                }
                if !self.atmosphere.as_ref().is_some_and(|a| a.pressure) {
                    return invalid(format!(
                        "[[boundary]] {:?}: inverse_barometer needs [atmosphere] with its \
                         pressure",
                        b.tag.name()
                    ));
                }
            }
        }
        if let Some(cfl) = self.numerics.cfl
            && !positive(cfl)
        {
            return invalid(format!("numerics.cfl must be positive, not {cfl}"));
        }
        if self.numerics.local_time_stepping_levels > Multirate::<StandardIntegrator>::MAX_LEVELS {
            return invalid(format!(
                "numerics.local_time_stepping_levels must be at most {}, not {}",
                Multirate::<StandardIntegrator>::MAX_LEVELS,
                self.numerics.local_time_stepping_levels
            ));
        }
        if let Some(dt) = self.numerics.dt_max_s
            && !positive(dt)
        {
            return invalid(format!("numerics.dt_max_s must be positive, not {dt}"));
        }
        let o = &self.output;
        if !positive(o.interval_minutes) {
            return invalid(format!(
                "output.interval_minutes must be positive, not {}",
                o.interval_minutes
            ));
        }
        if let Some(s) = o.station_interval_minutes {
            if !(positive(s) && s.is_finite()) {
                return invalid(format!(
                    "output.station_interval_minutes must be positive, not {s}"
                ));
            }
            let (fine, coarse) = (s.min(o.interval_minutes), s.max(o.interval_minutes));
            let ratio = coarse / fine;
            if (ratio - ratio.round()).abs() > 1e-9 * ratio {
                return invalid(format!(
                    "output.interval_minutes ({}) and station_interval_minutes ({s}): one \
                     must be a whole multiple of the other",
                    o.interval_minutes
                ));
            }
        }
        if o.netcdf && !cfg!(feature = "netcdf") {
            return invalid(
                "output.netcdf needs dgcom built with the `netcdf` feature (on by default)".into(),
            );
        }
        let mut names = Vec::new();
        for s in &o.stations {
            if names.contains(&&s.name) {
                return invalid(format!("two stations named {:?}", s.name));
            }
            names.push(&s.name);
            match (s.x_m, s.y_m, s.lon, s.lat) {
                (Some(_), Some(_), None, None) => {}
                (None, None, Some(_), Some(_)) if placed => {}
                (None, None, Some(_), Some(_)) => {
                    return invalid(format!(
                        "station {:?} is given by lon and lat, which needs a placed mesh: \
                         mesh.geographic, or mesh.origin",
                        s.name
                    ));
                }
                _ => {
                    return invalid(format!(
                        "station {:?} needs x_m and y_m, or lon and lat",
                        s.name
                    ));
                }
            }
            for (key, value) in [
                ("open_water_depth_m", s.open_water_depth_m),
                ("max_offset_m", s.max_offset_m),
            ] {
                if !(non_negative(value) && value.is_finite()) {
                    return invalid(format!(
                        "station {:?}: {key} must be ≥ 0, not {value}",
                        s.name
                    ));
                }
            }
        }
        // Last, so that a build without NetCDF still checks the rest
        if self.atmosphere.is_some() && !cfg!(feature = "netcdf") {
            return invalid(
                "[atmosphere] needs dgcom built with the `netcdf` feature (on by default)".into(),
            );
        }
        Ok(())
    }

    /// Whether the mesh is placed on the Earth (`mesh.geographic` or
    /// `mesh.origin`).
    pub fn placed(&self) -> bool {
        self.mesh.geographic.is_some() || self.mesh.origin.is_some()
    }

    /// The centre of the projection that places the mesh (at x = y = 0), if
    /// it is placed.
    pub fn projection_center(&self) -> Option<GeoPoint> {
        match (&self.mesh.geographic, self.mesh.origin) {
            (Some(g), _) => Some(g.center()),
            (None, origin) => origin,
        }
    }

    /// The projection that places the mesh, if it is placed.
    pub fn projection(&self) -> Option<LocalProjection> {
        self.projection_center()
            .map(|c| LocalProjection::new(c.lat, c.lon))
    }

    /// This case with every default stated, as `case.resolved.toml` records
    /// it: the CFL number computed from the order and stepper.
    pub fn resolved(&self) -> Self {
        let mut case = self.clone();
        case.numerics.cfl = Some(self.numerics.cfl_at(self.mesh.order));
        case
    }

    /// The run's clock, from `run.start`.
    pub fn clock(&self) -> Result<Option<ModelClock>, CaseError> {
        self.run
            .start
            .as_deref()
            .map(|s| {
                ModelClock::parse(s)
                    .map_err(|e| CaseError::Invalid(format!("run.start {s:?}: {e}")))
            })
            .transpose()
    }
}

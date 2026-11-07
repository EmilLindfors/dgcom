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
use crate::mesh::BoundaryTag;
use crate::tides::{canonical_name, constituent_period};
use crate::time::ModelClock;

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

/// `[mesh]`: exactly one of `rectangle` and `gmsh`.
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

/// `coriolis = { f_plane = 1.2e-4 }` or `coriolis = { latitude = 63.5 }`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Coriolis {
    /// A constant Coriolis parameter f (s⁻¹).
    FPlane(f64),
    /// The f-plane at a latitude (degrees north): f = 2Ω sin φ.
    Latitude(f64),
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
/// `still` and `tide`. Every open boundary is a characteristic (Flather-type)
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
    /// The external velocity, since `still` and `tide` give only a level
    /// (default `"at_rest"`).
    #[serde(default)]
    pub velocity: BoundaryVelocity,
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
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NumericsSection {
    /// CFL number of the SSP-RK3 step (default 1).
    #[serde(default = "default_cfl")]
    pub cfl: f64,
    /// Upper bound of the time step (s, default none).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dt_max_s: Option<f64>,
}

impl Default for NumericsSection {
    fn default() -> Self {
        Self {
            cfl: default_cfl(),
            dt_max_s: None,
        }
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
    /// Time between outputs (minutes, default 60).
    #[serde(default = "default_interval_minutes")]
    pub interval_minutes: f64,
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
    /// Points where the surface and the current are recorded at every
    /// output, into `stations.csv`.
    #[serde(default, rename = "station", skip_serializing_if = "Vec::is_empty")]
    pub stations: Vec<Station>,
}

impl Default for OutputSection {
    fn default() -> Self {
        Self {
            directory: None,
            interval_minutes: default_interval_minutes(),
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
    /// x (m)
    pub x_m: f64,
    /// y (m)
    pub y_m: f64,
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
fn default_ramp_hours() -> f64 {
    1.0
}
fn default_cfl() -> f64 {
    1.0
}
fn default_interval_minutes() -> f64 {
    60.0
}

/// Whether `x` is positive (NaN is not).
fn positive(x: f64) -> bool {
    x > 0.0
}

/// Polynomial orders the case layer accepts.
const ORDERS: std::ops::RangeInclusive<usize> = 1..=8;

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
        match (&self.mesh.rectangle, &self.mesh.gmsh) {
            (Some(r), None) => {
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
            (None, Some(_)) => {}
            _ => return invalid("[mesh] needs exactly one of `rectangle` and `gmsh`".into()),
        }
        match (&self.bathymetry.depth_m, &self.bathymetry.ramp) {
            (Some(d), None) if d.is_finite() => {}
            (None, Some(r)) => {
                if r.from_m == r.to_m {
                    return invalid("bathymetry.ramp needs from_m and to_m apart".into());
                }
            }
            (Some(d), None) => return invalid(format!("bathymetry.depth_m is {d}")),
            _ => return invalid("[bathymetry] needs exactly one of `depth_m` and `ramp`".into()),
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
        if let Some(Coriolis::Latitude(lat)) = p.coriolis
            && !(-90.0..=90.0).contains(&lat)
        {
            return invalid(format!(
                "physics.coriolis.latitude must be in [-90, 90], not {lat}"
            ));
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
            match (&b.still, &b.tide) {
                (Some(_), None) => {}
                (None, Some(tide)) => {
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
                        "[[boundary]] {:?} needs exactly one of `still` and `tide`",
                        b.tag
                    ));
                }
            }
        }
        if !positive(self.numerics.cfl) {
            return invalid(format!(
                "numerics.cfl must be positive, not {}",
                self.numerics.cfl
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
        }
        Ok(())
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

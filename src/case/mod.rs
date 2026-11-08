//! Case files: a whole run described in TOML (or JSON), so any coastal
//! domain runs without writing Rust (`dgcom run case.toml`, TODO P8).
//!
//! A case names the mesh, the bed, the physics, the open boundaries, the
//! time stepping and the output ([`Case`]; every key is documented in
//! [`schema`], and `cases/tidal_channel/case.toml` is a complete example).
//! The schema is strict: an unknown or misspelt key is an error, not
//! ignored, and each default is defined once, in the schema. The run writes
//! the case it ran, with every default filled in, to
//! `case.resolved.toml` in its output directory.
//!
//! Paths in a case file are relative to the file's directory.
//!
//! What a case can hold so far (P8.1, 2D):
//! - a rectangle, a Gmsh mesh, or a longitude/latitude box, placed on the
//!   Earth or not, with the land dropped;
//! - a uniform, linear or GeoTIFF bed, smoothed to a slope factor or for a
//!   3D model (element slopes, cliff shores, tidal ponds), the land at its
//!   walls lowered;
//! - Coriolis (f-plane or beta plane), Manning or quadratic-drag friction,
//!   viscosity, wetting and drying;
//! - still-water, harmonic-tide and tidal-atlas open boundaries, and
//!   nesting in a parent ocean model (NetCDF, with the `netcdf` feature);
//! - a weather model's wind and pressure, with the inverse-barometer level
//!   at the open boundaries (NetCDF);
//! - snapshot, VTK, NetCDF and station output, the stations scored
//!   against tide gauges;
//! - a 3D run (`[three_d]`): σ-layers under the depth mean, started from
//!   and nested in a parent's columns or a profile;
//! - spectral waves coupled to the 2D run (`[waves]`), on its mesh or their
//!   own, from a JONSWAP sea or a parent wave model's spectra;
//! - a fish farm's net cages dragging on the flow (`[farm]`).
//!
//! Next: the farm's analysis and particles (TODO F.1, F.2), and retiring
//! `froya_real_data`'s moved modes (`notes/case-files.md`).

mod domain;
mod run;
pub mod schema;

use std::path::{Path, PathBuf};

pub use domain::{Domain, ParentWavePoints, StationProbe, WaveDomain};
pub use run::{MeshReport, Progress, RunReport, StationSeries};
pub use schema::*;

/// What can go wrong with a case.
#[derive(Debug, thiserror::Error)]
pub enum CaseError {
    /// The case file could not be read or an output not written.
    #[error("{0}")]
    Io(String),
    /// The case file is not valid TOML or JSON, or not a case.
    #[error("{0}")]
    Parse(String),
    /// The case is well-formed but inconsistent (see [`Case::validate`]).
    #[error("{0}")]
    Invalid(String),
    /// The run failed.
    #[error("{0}")]
    Run(String),
}

/// A case read from a file, with the directory its paths are relative to.
#[derive(Clone, Debug)]
pub struct CaseFile {
    /// The case.
    pub case: Case,
    /// The directory of the case file.
    pub base_dir: PathBuf,
}

impl CaseFile {
    /// Read and validate `path` (`.json` as JSON, anything else as TOML),
    /// after applying `overrides`: `key.path=value` with a TOML value
    /// (`numerics.cfl=0.5`, `run.name="test"`; a bare word is a string).
    pub fn read(path: &Path, overrides: &[String]) -> Result<Self, CaseError> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| CaseError::Io(format!("{}: {e}", path.display())))?;
        let json = path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("json"));
        let case = Case::parse(&text, json, overrides)
            .map_err(|e| CaseError::Parse(format!("{}: {e}", path.display())))?;
        case.validate()?;
        let base_dir = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."))
            .to_path_buf();
        Ok(Self { case, base_dir })
    }

    /// The output directory: `output.directory`, else `output/<run.name>`,
    /// relative to the case file.
    pub fn output_dir(&self) -> PathBuf {
        let dir = self
            .case
            .output
            .directory
            .clone()
            .unwrap_or_else(|| Path::new("output").join(&self.case.run.name));
        self.base_dir.join(dir)
    }

    /// A path from the case file, relative to its directory.
    pub fn resolve(&self, path: &Path) -> PathBuf {
        self.base_dir.join(path)
    }
}

impl Case {
    /// A case from TOML (or JSON) text with `overrides` applied (see
    /// [`CaseFile::read`]); not yet validated.
    pub fn parse(text: &str, json: bool, overrides: &[String]) -> Result<Self, String> {
        if overrides.is_empty() {
            // Straight from the text, so errors carry their line
            return if json {
                serde_json::from_str(text).map_err(|e| e.to_string())
            } else {
                toml::from_str(text).map_err(|e| e.to_string())
            };
        }
        let mut table = if json {
            let value: serde_json::Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
            match toml::Value::try_from(value).map_err(|e| e.to_string())? {
                toml::Value::Table(table) => table,
                _ => return Err("a case file is a table (an object in JSON)".into()),
            }
        } else {
            toml::from_str::<toml::Table>(text).map_err(|e| e.to_string())?
        };
        for o in overrides {
            apply_override(&mut table, o)?;
        }
        toml::Value::Table(table)
            .try_into()
            .map_err(|e: toml::de::Error| e.to_string())
    }

    /// This case as TOML, with every default written out.
    pub fn to_toml(&self) -> String {
        toml::to_string(self).expect("a case serializes to TOML")
    }
}

/// Set `key.path=value` in `table`, creating tables on the way. A number
/// after an array of tables picks its entry: `boundary.0.ramp_hours` is the
/// first `[[boundary]]`'s.
fn apply_override(table: &mut toml::Table, assignment: &str) -> Result<(), String> {
    let (path, value) = assignment
        .split_once('=')
        .ok_or_else(|| format!("override {assignment:?} is not key.path=value"))?;
    let (path, value) = (path.trim(), value.trim());
    // A TOML value, else the text as a string
    let value = toml::from_str::<toml::Table>(&format!("v = {value}"))
        .ok()
        .and_then(|mut t| t.remove("v"))
        .unwrap_or_else(|| toml::Value::String(value.to_string()));
    let keys: Vec<&str> = path.split('.').collect();
    let (last, parents) = keys.split_last().expect("split yields one key at least");
    let mut current = table;
    let mut parents = parents.iter().peekable();
    while let Some(key) = parents.next() {
        let entry = current
            .entry(key.to_string())
            .or_insert_with(|| toml::Value::Table(toml::Table::new()));
        let entry = match entry {
            toml::Value::Array(entries) => {
                let index = parents
                    .next()
                    .and_then(|i| i.parse::<usize>().ok())
                    .ok_or_else(|| {
                        format!("override {path:?}: {key} is a list; pick an entry, {key}.0")
                    })?;
                let n = entries.len();
                entries.get_mut(index).ok_or_else(|| {
                    format!(
                        "override {path:?}: {key} has {n} entries, not {}",
                        index + 1
                    )
                })?
            }
            entry => entry,
        };
        current = entry
            .as_table_mut()
            .ok_or_else(|| format!("override {path:?}: {key} is not a table"))?;
    }
    current.insert(last.to_string(), value);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL: &str = r#"
        [run]
        duration_hours = 1.0

        [mesh.rectangle]
        x_m = [0.0, 1000.0]
        y_m = [0.0, 100.0]
        nx = 4
        ny = 1

        [bathymetry]
        depth_m = 10.0
    "#;

    fn parse(text: &str) -> Result<Case, String> {
        Case::parse(text, false, &[])
    }

    #[test]
    fn a_minimal_case_takes_the_defaults() {
        let case = parse(MINIMAL).unwrap();
        case.validate().unwrap();
        assert_eq!(case.run.name, "case");
        assert_eq!(case.mesh.order, 2);
        assert_eq!(case.physics.gravity, 9.81);
        assert!(case.physics.wetting_drying);
        assert_eq!(case.numerics.cfl, None);
        assert_eq!(case.numerics.integrator, Integrator::SspRk43);
        assert!((case.numerics.cfl_at(2) - 0.9 * 1.51).abs() < 1e-12);
        assert!(case.mesh.drop_land);
        assert_eq!(case.output.interval_minutes, 60.0);
        assert_eq!(case.mesh.rectangle.as_ref().unwrap().west, SideTag::Wall);
    }

    #[test]
    fn an_unknown_key_is_an_error_with_its_line() {
        let text = MINIMAL.replace("nx = 4", "nx = 4\n        nz = 3");
        let err = parse(&text).unwrap_err();
        assert!(err.contains("nz"), "{err}");
        assert!(err.contains("line"), "{err}");
    }

    #[test]
    fn an_unknown_section_is_an_error() {
        let err = parse(&format!("{MINIMAL}\n[phyiscs]\ngravity = 9.8\n")).unwrap_err();
        assert!(err.contains("phyiscs"), "{err}");
    }

    #[test]
    fn the_mesh_needs_exactly_one_source() {
        let text = format!("{MINIMAL}\n[mesh]\ngmsh = \"a.msh\"\n");
        let mut case = parse(&text).unwrap();
        assert!(case.mesh.rectangle.is_some() && case.mesh.gmsh.is_some());
        case.mesh.gmsh = None;
        case.mesh.rectangle = None;
        assert!(matches!(case.validate(), Err(CaseError::Invalid(_))));
    }

    #[test]
    fn an_unknown_constituent_is_an_error() {
        let text = format!(
            "{MINIMAL}\n[[boundary]]\ntag = \"open\"\ntide = {{ constituents = [{{ name = \"X9\", amplitude_m = 1.0 }}] }}\n"
        );
        let err = parse(&text).unwrap().validate().unwrap_err();
        assert!(err.to_string().contains("X9"), "{err}");
    }

    #[test]
    fn a_boundary_needs_exactly_one_external_state() {
        let text = format!("{MINIMAL}\n[[boundary]]\ntag = \"open\"\n");
        assert!(parse(&text).unwrap().validate().is_err());
        let text = format!("{MINIMAL}\n[[boundary]]\ntag = \"open\"\nstill = {{}}\n");
        parse(&text).unwrap().validate().unwrap();
    }

    #[test]
    fn overrides_set_values_and_create_tables() {
        let overrides = [
            "numerics.cfl=0.5".to_string(),
            "run.name=sweep".to_string(),
            "physics.coriolis.latitude = 60".to_string(),
        ];
        let case = Case::parse(MINIMAL, false, &overrides).unwrap();
        assert_eq!(case.numerics.cfl, Some(0.5));
        assert_eq!(case.run.name, "sweep");
        assert_eq!(case.physics.coriolis, Some(Coriolis::Latitude(60.0)));
        let err = Case::parse(MINIMAL, false, &["numerics.cfll=0.5".into()]).unwrap_err();
        assert!(err.contains("cfll"), "{err}");

        // An entry of an array of tables, by its index
        let two = format!(
            "{}\n[[output.station]]\nname = \"a\"\nx_m = 1.0\ny_m = 1.0\n\
             [[output.station]]\nname = \"b\"\nx_m = 2.0\ny_m = 2.0\n",
            MINIMAL.replace("ny = 1", "ny = 1\n        west = \"open\"")
        );
        let case = Case::parse(&two, false, &["output.station.1.x_m=5.0".into()]).unwrap();
        assert_eq!(
            (case.output.stations[0].x_m, case.output.stations[1].x_m),
            (Some(1.0), Some(5.0))
        );
        for (bad, says) in [
            ("output.station.2.x_m=5.0", "has 2 entries"),
            ("output.station.x_m=5.0", "pick an entry"),
        ] {
            let err = Case::parse(&two, false, &[bad.into()]).unwrap_err();
            assert!(err.contains(says), "{bad}: {err}");
        }
    }

    #[test]
    fn the_resolved_case_reads_back_the_same() {
        let text = format!(
            "{MINIMAL}\n[physics]\nfriction = {{ manning = 0.025 }}\n\n[[boundary]]\ntag = \"tidal\"\n\
             tide = {{ constituents = [{{ name = \"M2\", amplitude_m = 1.0, phase_deg = 30.0 }}] }}\n\n\
             [[output.station]]\nname = \"a\"\nx_m = 500.0\ny_m = 50.0\n"
        );
        let case = parse(&text).unwrap();
        let text = case.to_toml();
        assert_eq!(parse(&text).unwrap(), case, "{text}");
        // The resolved case states the CFL number it computed
        let resolved = case.resolved();
        assert_eq!(resolved.numerics.cfl, Some(case.numerics.cfl_at(2)));
        assert_eq!(parse(&resolved.to_toml()).unwrap(), resolved);
    }

    #[test]
    fn json_reads_the_same_schema() {
        let case = parse(MINIMAL).unwrap();
        let json = serde_json::to_string(&case).unwrap();
        assert_eq!(Case::parse(&json, true, &[]).unwrap(), case);
        let with = Case::parse(&json, true, &["numerics.cfl=0.25".into()]).unwrap();
        assert_eq!(with.numerics.cfl, Some(0.25));
    }

    /// What needs the mesh placed on the Earth is refused without it, and
    /// accepted with `mesh.origin`.
    #[test]
    fn rasters_beta_planes_and_lon_lat_stations_need_a_placed_mesh() {
        let raster = MINIMAL.replace(
            "depth_m = 10.0",
            "raster = { file = \"bed.tif\", kind = \"topobathy\" }",
        );
        let beta = format!(
            "{MINIMAL}
[physics]
coriolis = \"beta_plane\"
"
        );
        let station = format!(
            "{MINIMAL}
[[output.station]]
name = \"a\"
lon = 8.5
lat = 63.7
"
        );
        for text in [&raster, &beta, &station] {
            let err = parse(text).unwrap().validate().unwrap_err().to_string();
            assert!(err.contains("placed"), "{err}");
            let placed = text.replace(
                "[bathymetry]",
                "[mesh.origin]
lat = 63.7
lon = 8.5

[bathymetry]",
            );
            parse(&placed).unwrap().validate().unwrap();
        }
    }

    #[test]
    fn a_geographic_mesh_is_placed_about_its_centre() {
        let text = MINIMAL.replace(
            "[mesh.rectangle]
        x_m = [0.0, 1000.0]
        y_m = [0.0, 100.0]",
            "[mesh.geographic]
        bbox = [8.0, 63.6, 9.2, 64.0]",
        );
        let case = parse(&text).unwrap();
        case.validate().unwrap();
        let c = case.projection_center().unwrap();
        assert!((c.lat - 63.8).abs() < 1e-12 && (c.lon - 8.6).abs() < 1e-12);
        let both = text.replace(
            "[bathymetry]",
            "[mesh.origin]
lat = 63.7
lon = 8.5

[bathymetry]",
        );
        assert!(parse(&both).unwrap().validate().is_err());
        let backwards = text.replace("[8.0, 63.6, 9.2, 64.0]", "[9.2, 63.6, 8.0, 64.0]");
        assert!(parse(&backwards).unwrap().validate().is_err());
    }

    #[test]
    fn an_atlas_needs_a_placed_mesh_and_a_start() {
        let boundary = "
[[boundary]]
tag = \"open\"
atlas = { file = \"a.txt\" }
";
        let open = MINIMAL.replace(
            "ny = 1",
            "ny = 1
        west = \"open\"",
        );
        let placed = open.replace(
            "[bathymetry]",
            "[mesh.origin]
lat = 63.7
lon = 8.5

[bathymetry]",
        );
        let started = |t: &str| {
            t.replace(
                "duration_hours = 1.0",
                "duration_hours = 1.0
        start = \"2025-06-15T00:00:00Z\"",
            )
        };
        for (text, needs) in [
            (format!("{}{boundary}", started(&open)), "placed"),
            (format!("{placed}{boundary}"), "run.start"),
        ] {
            let err = parse(&text).unwrap().validate().unwrap_err().to_string();
            assert!(err.contains(needs), "{err}");
        }
        parse(&format!("{}{boundary}", started(&placed)))
            .unwrap()
            .validate()
            .unwrap();
        let both = format!(
            "{}{boundary}still = {{}}
",
            started(&placed)
        );
        assert!(parse(&both).unwrap().validate().is_err());
    }

    #[test]
    fn nesting_needs_a_placed_mesh_a_start_and_an_atlas_to_correct() {
        let open = MINIMAL.replace(
            "ny = 1",
            "ny = 1
        west = \"open\"",
        );
        let placed = open.replace(
            "[bathymetry]",
            "[mesh.origin]
lat = 63.7
lon = 8.5

[bathymetry]",
        );
        let started = |t: &str| {
            t.replace(
                "duration_hours = 1.0",
                "duration_hours = 1.0
        start = \"2025-06-15T00:00:00Z\"",
            )
        };
        let nested = |extra: &str| {
            format!(
                "
[[boundary]]
tag = \"open\"
[boundary.nesting]
file = \"parent.nc\"
{extra}
"
            )
        };
        for (text, needs) in [
            (format!("{}{}", started(&open), nested("")), "placed"),
            (format!("{placed}{}", nested("")), "run.start"),
            (
                format!("{}{}", started(&placed), nested("correction = \"theory\"")),
                "atlas",
            ),
            (
                format!(
                    "{}{}",
                    started(&placed),
                    nested("[boundary.nesting.gauge]\nrecord = \"g.txt\"\nsource = \"s.txt\"")
                ),
                "atlas",
            ),
            (
                format!("{}{}", started(&placed), nested("transport_scaling = 0.5")),
                "transport_scaling",
            ),
            (
                format!("{}{}", started(&placed), nested("band_minutes = 0")),
                "band_minutes",
            ),
            (
                format!("{}{}", started(&placed), nested("band_km = -1")),
                "band_km",
            ),
            (
                format!(
                    "{}{}",
                    started(&placed),
                    nested(
                        "atlas = \"a.txt\"\ncorrection = \"theory\"\n[boundary.nesting.gauge]\nrecord = \"g.txt\"\nsource = \"s.txt\""
                    )
                ),
                "not both",
            ),
        ] {
            let err = parse(&text).unwrap().validate().unwrap_err().to_string();
            assert!(err.contains(needs), "{needs}: {err}");
        }
        let case = parse(&format!(
            "{}{}",
            started(&placed),
            nested("atlas = \"a.txt\"\ncorrection = \"theory\"")
        ))
        .unwrap();
        case.validate().unwrap();
        let n = case.boundaries[0].nesting.as_ref().unwrap();
        assert_eq!(
            (n.band_km, n.band_minutes, n.blend_bed, n.ramp_hours),
            (3.0, 30.0, true, 1.0)
        );
        assert_eq!((n.transport_scaling, n.max_closed), (3.0, 0.05));
        // One external state per boundary
        let both = format!("{}{}", started(&placed), nested(""))
            .replace("[boundary.nesting]", "still = {}\n[boundary.nesting]");
        let err = parse(&both).unwrap().validate().unwrap_err().to_string();
        assert!(err.contains("exactly one"), "{err}");
    }

    #[test]
    fn the_3d_bed_takes_the_examples_defaults_and_checks_its_values() {
        let with = |extra: &str| format!("{MINIMAL}\n[bathymetry.slopes_3d]\n{extra}\n");
        let case = parse(&with("")).unwrap();
        case.validate().unwrap();
        let b = &case.bathymetry;
        assert!(!b.lower_wall_land);
        assert_eq!(b.wall_low_water_m, -2.0);
        let s = b.slopes_3d.as_ref().unwrap();
        assert_eq!(
            (s.r_max, s.free_depth_m, s.thin_depth_m),
            (0.15, 19.0, None)
        );
        assert_eq!(
            s.thin_depth(None),
            0.1,
            "Hydrostatic3D's default without [three_d]"
        );
        assert_eq!(
            s.bound(),
            crate::mesh::ElementSlopeBound::for_pycnocline(19.0)
        );
        assert!(s.raise_cliffs && s.fill_ponds);
        assert_eq!(
            (s.cliff_depth_m, s.pond_low_water_m, s.high_water_m),
            (20.0, -1.43, 2.0)
        );
        for (extra, needs) in [
            ("r_max = 1.0", "r_max"),
            ("r_max = 0.0", "r_max"),
            ("free_depth_m = -1.0", "free_depth_m"),
            ("thin_depth_m = 0.0", "thin_depth_m"),
            ("cliff_depth_m = 0.05", "cliff_depth_m"),
            ("pond_low_water_m = 2.0", "pond_low_water_m"),
            ("high_water_m = nan", "high_water_m"),
        ] {
            let err = parse(&with(extra))
                .unwrap()
                .validate()
                .unwrap_err()
                .to_string();
            assert!(err.contains(needs), "{needs}: {err}");
        }
        // A switched-off step is not checked
        parse(&with("raise_cliffs = false\ncliff_depth_m = 0.05"))
            .unwrap()
            .validate()
            .unwrap();
        let err =
            parse(&MINIMAL.replace("depth_m = 10.0", "depth_m = 10.0\nwall_low_water_m = inf"))
                .unwrap()
                .validate()
                .unwrap_err();
        assert!(err.to_string().contains("wall_low_water_m"), "{err}");
    }

    #[test]
    fn stations_sample_in_open_water_by_default() {
        let with = |extra: &str| {
            format!("{MINIMAL}\n[[output.station]]\nname = \"s\"\nx_m = 500.0\ny_m = 50.0\n{extra}")
        };
        let case = parse(&with("")).unwrap();
        case.validate().unwrap();
        let s = &case.output.stations[0];
        assert_eq!((s.open_water_depth_m, s.max_offset_m), (3.0, 3000.0));
        for extra in ["open_water_depth_m = -1.0", "max_offset_m = inf"] {
            let err = parse(&with(extra)).unwrap().validate().unwrap_err();
            assert!(err.to_string().contains(&extra[..10]), "{err}");
        }
    }

    #[test]
    fn the_atmosphere_needs_a_placed_mesh_and_a_start() {
        let open = MINIMAL.replace("ny = 1", "ny = 1\n        west = \"open\"");
        let placed = open.replace(
            "[bathymetry]",
            "[mesh.origin]\nlat = 63.7\nlon = 8.5\n\n[bathymetry]",
        );
        let started = placed.replace(
            "duration_hours = 1.0",
            "duration_hours = 1.0\n        start = \"2025-06-15T00:00:00Z\"",
        );
        let weather = |extra: &str| format!("\n[atmosphere]\nfiles = [\"met.nc\"]\n{extra}\n");
        let boundary =
            |extra: &str| format!("\n[[boundary]]\ntag = \"open\"\nstill = {{}}\n{extra}\n");
        let check = |text: String| parse(&text).unwrap().validate().map_err(|e| e.to_string());
        for (text, says) in [
            (format!("{open}{}{}", boundary(""), weather("")), "placed"),
            (
                format!("{placed}{}{}", boundary(""), weather("")),
                "run.start",
            ),
            (
                format!("{started}{}{}", boundary(""), weather("ramp_hours = -1.0")),
                "ramp_hours",
            ),
            (
                format!(
                    "{started}{}{}",
                    boundary(""),
                    weather("wind = false\npressure = false")
                ),
                "neither",
            ),
            (
                format!("{started}{}", boundary("inverse_barometer = 0.5")),
                "[atmosphere]",
            ),
            (
                format!(
                    "{started}{}{}",
                    boundary("inverse_barometer = 0.5"),
                    weather("pressure = false")
                ),
                "pressure",
            ),
            (
                format!(
                    "{started}{}{}",
                    boundary("inverse_barometer = 3.0"),
                    weather("")
                ),
                "0–2",
            ),
        ] {
            let err = check(text).unwrap_err();
            assert!(err.contains(says), "{says}: {err}");
        }
        let text = format!("{started}{}{}", boundary(""), weather(""));
        let case = parse(&text).unwrap();
        let a = case.atmosphere.as_ref().unwrap();
        assert_eq!((a.ramp_hours, a.wind, a.pressure), (1.0, true, true));
        // The full response at a still or tidal boundary, none at a nested one
        assert_eq!(case.boundaries[0].inverse_barometer_fraction(), 1.0);
        let mut nested = case.boundaries[0].clone();
        nested.still = None;
        nested.nesting = parse(
            "[run]\nduration_hours = 1.0\n[mesh.rectangle]\nx_m = [0.0, 1.0]\ny_m = [0.0, 1.0]\n\
             nx = 1\nny = 1\n[bathymetry]\ndepth_m = 1.0\n[[boundary]]\ntag = \"open\"\n\
             [boundary.nesting]\nfile = \"p.nc\"\n",
        )
        .unwrap()
        .boundaries[0]
            .nesting
            .clone();
        assert_eq!(nested.inverse_barometer_fraction(), 0.0);
        if cfg!(feature = "netcdf") {
            case.validate().unwrap();
        } else {
            let err = case.validate().unwrap_err().to_string();
            assert!(err.contains("netcdf"), "{err}");
        }
    }

    #[test]
    fn gauge_scores_start_when_the_forcing_is_fully_on() {
        let open = MINIMAL.replace("ny = 1", "ny = 1\n        west = \"open\"");
        let tide = |ramp: f64| {
            format!(
                "{open}\n[[boundary]]\ntag = \"open\"\n[boundary.tide]\nramp_hours = {ramp}\n\
                 constituents = [{{ name = \"M2\", amplitude_m = 1.0 }}]\n\
                 [[output.station]]\nname = \"s\"\nx_m = 500.0\ny_m = 50.0\n\
                 gauge = \"g.txt\"\n"
            )
        };
        // A gauge's record is in UTC
        let err = parse(&tide(2.0)).unwrap().validate().unwrap_err();
        assert!(err.to_string().contains("run.start"), "{err}");
        let started = |text: String| {
            text.replace(
                "duration_hours = 1.0",
                "duration_hours = 1.0\n        start = \"2025-06-15T00:00:00Z\"",
            )
        };
        let case = parse(&started(tide(2.0))).unwrap();
        case.validate().unwrap();
        assert_eq!(case.spinup_hours(), 2.0, "the tide's ramp");
        let set = parse(&started(tide(2.0)).replace(
            "[[output.station]]",
            "[output]\nspinup_hours = 5.0\n[[output.station]]",
        ))
        .unwrap();
        assert_eq!(set.spinup_hours(), 5.0);
        let bad = parse(&started(tide(2.0)).replace(
            "[[output.station]]",
            "[output]\nspinup_hours = -1.0\n[[output.station]]",
        ))
        .unwrap();
        assert!(
            bad.validate()
                .unwrap_err()
                .to_string()
                .contains("spinup_hours")
        );
        // Still water has no ramp
        assert_eq!(parse(MINIMAL).unwrap().spinup_hours(), 0.0);
    }

    #[test]
    fn a_3d_run_takes_the_examples_defaults_and_refuses_what_it_replaces() {
        let profile = "profile = [[0.0, 12.0, 32.0], [-20.0, 8.0, 34.0]]";
        let three_d = |extra: &str| format!("{MINIMAL}\n[three_d]\nlevels = 10\n{extra}\n");
        let case = parse(&three_d(profile)).unwrap();
        case.validate().unwrap();
        let d = case.three_d.as_ref().unwrap();
        assert_eq!(
            (d.theta_s, d.theta_b, d.hc_m, d.thin_depth_m),
            (5.0, 0.4, 10.0, 1.0)
        );
        assert_eq!((d.cfl(), d.dt_max_s, d.barotropic_cfl), (4.0, 60.0, 2.0));
        assert_eq!((d.barotropic_levels, d.levels_lts), (2, 0));
        assert_eq!(
            (d.mixing, d.bottom_roughness_m),
            (Mixing3D::KEpsilon, 0.003)
        );
        assert_eq!(
            parse(&three_d(&format!("{profile}\nscheme = \"ssp_rk3\"")))
                .unwrap()
                .three_d
                .unwrap()
                .cfl(),
            2.0
        );
        // The profile: linear between its points, held beyond
        assert_eq!(d.profile_at(5.0), Some((12.0, 32.0)));
        assert_eq!(d.profile_at(-10.0), Some((10.0, 33.0)));
        assert_eq!(d.profile_at(-50.0), Some((8.0, 34.0)));
        // A 3D step reads its forcing a third of its longest step past the end
        assert_eq!(case.forcing_end_s(), 3600.0 + 20.0);
        // The bed's thin depth is the 3D model's
        let with_slopes = parse(
            &three_d(profile).replace("depth_m = 10.0", "depth_m = 10.0\n[bathymetry.slopes_3d]"),
        )
        .unwrap();
        let slopes = with_slopes.bathymetry.slopes_3d.as_ref().unwrap();
        assert_eq!(slopes.thin_depth(with_slopes.three_d.as_ref()), 1.0);

        for (text, says) in [
            (
                three_d(profile).replace("levels = 10", "levels = 1"),
                "levels",
            ),
            (three_d(""), "profile"),
            (
                three_d("profile = [[-20.0, 8.0, 34.0], [0.0, 12.0, 32.0]]"),
                "top down",
            ),
            (three_d(&format!("{profile}\ntheta_b = 2.0")), "theta_b"),
            (three_d(&format!("{profile}\njerlov = \"IX\"")), "jerlov"),
            (
                three_d(profile).replace(
                    "[bathymetry]",
                    "[physics]\nfriction = { manning = 0.025 }\n[bathymetry]",
                ),
                "friction",
            ),
            (
                format!("{}\n[numerics]\ncfl = 0.5\n", three_d(profile)),
                "[numerics]",
            ),
            (
                format!("{}\n[output]\nvtk = true\n", three_d(profile)),
                "2D only",
            ),
        ] {
            let err = parse(&text).unwrap().validate().unwrap_err().to_string();
            assert!(err.contains(says), "{says}: {err}");
        }
    }

    #[test]
    fn waves_take_the_examples_defaults_and_their_checks() {
        let open = MINIMAL.replace("ny = 1", "ny = 1\n        west = \"open\"");
        let with = |extra: &str| {
            format!(
                "{open}\n[[boundary]]\ntag = \"open\"\nstill = {{}}\n[output]\ninterval_minutes = \
                 10.0\n[waves]\nexchange_minutes = 10.0\n{extra}\n"
            )
        };
        let sea = "[waves.sea]\nhs_m = 1.0\ntp_s = 6.0\nfrom_deg = 270.0";
        let case = parse(&with(sea)).unwrap();
        case.validate().unwrap();
        let w = case.waves.as_ref().unwrap();
        assert_eq!((w.frequencies, w.directions), (25, 36));
        assert!(w.implicit && w.absorbing_land && w.depth_limit);
        assert_eq!(w.force, WaveForce::Dissipation);
        // WAM's integration by default, SWAN's on request
        assert_eq!(
            w.integration(),
            (WaveIntegration::Implicit, Some(WaveLimiter::Rate), 4, 1.0)
        );
        let swan = parse(&with(&format!("sources = \"swan\"\n{sea}"))).unwrap();
        assert_eq!(
            swan.waves.unwrap().integration(),
            (WaveIntegration::Frozen, None, 1, 0.5)
        );
        for (text, says) in [
            (with(""), "exactly one"),
            (
                with(&format!("{sea}\n[waves.spectra]\nfile = \"s.nc\"")),
                "exactly one",
            ),
            (
                with(sea).replace("exchange_minutes = 10.0", "exchange_minutes = 15.0"),
                "whole multiple",
            ),
            (with(&format!("frequencies = 2\n{sea}")), "frequencies"),
            (
                with(&format!(
                    "{sea}\n[waves.mesh]\norigin = {{ lat = 63.0, lon = 8.0 }}\ngmsh = \"w.msh\""
                )),
                "origin",
            ),
            (
                with(&format!("{sea}\n[waves.mesh]\norder = 1")),
                "waves.mesh",
            ),
            (
                with(&format!("{sea}\n[waves.spectra]\nfile = \"s.nc\"")).replace(sea, ""),
                "placed",
            ),
            (
                format!(
                    "{}\n[three_d]\nlevels = 4\nprofile = [[0.0, 10.0, 34.0]]\n",
                    with(sea)
                ),
                "three_d",
            ),
        ] {
            let err = parse(&text).unwrap().validate().unwrap_err().to_string();
            assert!(err.contains(says), "{says}: {err}");
        }
    }

    #[test]
    fn a_farm_takes_a_ring_frames_defaults_and_needs_a_placed_mesh() {
        let farm = |extra: &str| format!("\n[farm]\nsite = \"site.txt\"\n{extra}\n");
        let placed = MINIMAL.replace(
            "[bathymetry]",
            "[mesh.origin]\nlat = 63.7\nlon = 8.5\n\n[bathymetry]",
        );
        let case = parse(&format!("{placed}{}", farm(""))).unwrap();
        case.validate().unwrap();
        let f = case.farm.as_ref().unwrap();
        let ring = crate::io::CageGrid::RING_160M;
        assert_eq!(
            (
                f.cage_spacing_m,
                f.cage_radius_m,
                f.net_depth_m,
                f.solidity,
                f.drag_share
            ),
            (
                ring.spacing,
                ring.radius,
                ring.net_depth,
                ring.solidity,
                1.0
            )
        );
        for (text, says) in [
            (format!("{MINIMAL}{}", farm("")), "placed"),
            (format!("{placed}{}", farm("solidity = 1.0")), "solidity"),
            (
                format!("{placed}{}", farm("drag_share = -0.1")),
                "drag_share",
            ),
            (
                format!("{placed}{}", farm("net_depth_m = 0.0")),
                "net_depth_m",
            ),
        ] {
            let err = parse(&text).unwrap().validate().unwrap_err().to_string();
            assert!(err.contains(says), "{says}: {err}");
        }
    }

    #[test]
    fn local_time_stepping_has_at_most_sixteen_levels() {
        let ok = format!(
            "{MINIMAL}
[numerics]
local_time_stepping_levels = 16
"
        );
        parse(&ok).unwrap().validate().unwrap();
        let err = parse(&ok.replace("= 16", "= 17"))
            .unwrap()
            .validate()
            .unwrap_err();
        assert!(err.to_string().contains("16"), "{err}");
    }

    #[test]
    fn a_ramp_is_linear_between_its_ends_and_constant_beyond() {
        let ramp = Ramp {
            from_m: [0.0, 0.0],
            to_m: [100.0, 0.0],
            depth_m: [20.0, 0.0],
        };
        assert_eq!(ramp.depth(-50.0, 7.0), 20.0);
        assert_eq!(ramp.depth(25.0, 3.0), 15.0);
        assert_eq!(ramp.depth(500.0, 0.0), 0.0);
    }
}

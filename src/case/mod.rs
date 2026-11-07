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
//! What a case can hold so far (P8.1, 2D): a rectangle or a Gmsh mesh, a
//! uniform or linear bed, Coriolis, Manning or quadratic-drag friction,
//! viscosity, wetting and drying, still-water and harmonic-tide open
//! boundaries, and snapshot, VTK, NetCDF and station output. Next: bed
//! rasters, tidal atlases, nesting in a parent model and the atmosphere
//! (`notes/case-files.md`).

mod run;
pub mod schema;

use std::path::{Path, PathBuf};

pub use run::{Domain, Progress, RunReport, StationSeries};
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

/// Set `key.path=value` in `table`, creating tables on the way.
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
    for key in parents {
        let entry = current
            .entry(key.to_string())
            .or_insert_with(|| toml::Value::Table(toml::Table::new()));
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
        assert_eq!(case.numerics.cfl, 1.0);
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
        assert_eq!(case.numerics.cfl, 0.5);
        assert_eq!(case.run.name, "sweep");
        assert_eq!(case.physics.coriolis, Some(Coriolis::Latitude(60.0)));
        let err = Case::parse(MINIMAL, false, &["numerics.cfll=0.5".into()]).unwrap_err();
        assert!(err.contains("cfll"), "{err}");
    }

    #[test]
    fn the_resolved_case_reads_back_the_same() {
        let text = format!(
            "{MINIMAL}\n[physics]\nfriction = {{ manning = 0.025 }}\n\n[[boundary]]\ntag = \"tidal\"\n\
             tide = {{ constituents = [{{ name = \"M2\", amplitude_m = 1.0, phase_deg = 30.0 }}] }}\n\n\
             [[output.station]]\nname = \"a\"\nx_m = 500.0\ny_m = 50.0\n"
        );
        let case = parse(&text).unwrap();
        let resolved = case.to_toml();
        assert_eq!(parse(&resolved).unwrap(), case, "{resolved}");
        assert!(resolved.contains("cfl = 1.0"), "{resolved}");
    }

    #[test]
    fn json_reads_the_same_schema() {
        let case = parse(MINIMAL).unwrap();
        let json = serde_json::to_string(&case).unwrap();
        assert_eq!(Case::parse(&json, true, &[]).unwrap(), case);
        let with = Case::parse(&json, true, &["numerics.cfl=0.25".into()]).unwrap();
        assert_eq!(with.numerics.cfl, 0.25);
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

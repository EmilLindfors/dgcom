//! A case run's restarts (`[restart]`): when they are written, what a case
//! run keeps in them beside the state, and the check of a resumed run's case
//! against the one that wrote the restart.
//!
//! A restart's driver data holds:
//! - `text.case`: the resolved case of the run that wrote it, as
//!   `case.resolved.toml` (with its build line);
//! - `metadata`: `build` ([`crate::provenance::describe`]) and each station's
//!   name (`station.<j>`);
//! - `extra.station.<j>.<field>`: each station's records so far
//!   ([`StationSeries`]'s fields);
//! - `extra.output.counters`: the VTK frames written and the volume at the
//!   start.
//!
//! On resume the case must resolve as the restart's does, key for key, but
//! for [`RESUME_MAY_CHANGE`] and `restart.changes`: comparing the resolved
//! cases, not the files as given, sees a default changed between two builds
//! and takes an option left out as its default (P5.3's gaps in
//! `froya_real_data`'s check of the command line).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use super::StationSeries;
use crate::case::{Case, CaseError, CaseFile};
use crate::io::{DriverData, RestartError};

/// The keys any resume may change, each with the keys under it: when the run
/// stops, what it writes and how often, its time stepping, its restarts.
/// The stations may not (their records continue).
pub(super) const RESUME_MAY_CHANGE: &[&str] = &[
    "run.duration_hours",
    "output.directory",
    "output.interval_minutes",
    "output.station_interval_minutes",
    "output.spinup_hours",
    "output.snapshot",
    "output.vtk",
    "output.netcdf",
    "numerics",
    "three_d.scheme",
    "three_d.cfl",
    "three_d.dt_max_s",
    "three_d.levels_lts",
    "three_d.barotropic_scheme",
    "three_d.barotropic_cfl",
    "three_d.barotropic_levels",
    "restart",
];

/// A restart a run resumes from: its file, its time and the driver's data
/// (the state goes to the run).
pub(super) struct Resumed {
    pub path: PathBuf,
    pub time: f64,
    pub driver: DriverData,
}

impl Resumed {
    /// What the log says of it.
    pub fn note(&self) -> String {
        format!(
            "resumed from {} at {:.2} h, written by {}",
            self.path.display(),
            self.time / 3600.0,
            self.driver.metadata("build").unwrap_or("an unknown build")
        )
    }
}

impl CaseFile {
    /// The restart this run resumes from (`restart.from`), if any.
    pub(super) fn resume_path(&self) -> Option<PathBuf> {
        let from = self.case.restart.as_ref()?.from.as_ref()?;
        Some(self.resolve(from))
    }

    /// Refuse a restart written by a run whose resolved case differs from
    /// this one's beyond [`RESUME_MAY_CHANGE`] and `restart.changes`.
    pub(super) fn check_resumed(&self, path: &Path, driver: &DriverData) -> Result<(), CaseError> {
        let refuse = |why: String| {
            Err(CaseError::Invalid(format!(
                "cannot resume from {}: {why}",
                path.display()
            )))
        };
        let Some(text) = driver.text("case") else {
            return refuse("it was not written by a case run (it holds no case)".into());
        };
        let theirs = match Case::parse(text, false, &[]) {
            Ok(case) => flatten_case(&case.resolved()),
            Err(e) => return refuse(format!("its case does not read in this build: {e}")),
        };
        let ours = flatten_case(&self.case.resolved());
        let changes = self
            .case
            .restart
            .as_ref()
            .map_or(&[][..], |r| r.changes.as_slice());
        let free = |key: &str| {
            RESUME_MAY_CHANGE
                .iter()
                .copied()
                .chain(changes.iter().map(String::as_str))
                .any(|p| key == p || key.strip_prefix(p).is_some_and(|r| r.starts_with('.')))
        };
        let shown = |v: Option<&toml::Value>| v.map_or("(none)".to_string(), |v| v.to_string());
        let differ: Vec<String> = ours
            .keys()
            .chain(theirs.keys())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .filter(|key| !free(key) && ours.get(*key) != theirs.get(*key))
            .map(|key| {
                format!(
                    "{key} = {} (the restart's run: {})",
                    shown(ours.get(key)),
                    shown(theirs.get(key))
                )
            })
            .collect();
        if differ.is_empty() {
            return Ok(());
        }
        refuse(format!(
            "its run's case differs: {}. Give them as it did, or name the keys this run \
             changes on purpose in restart.changes",
            differ.join("; ")
        ))
    }
}

/// `case` as `key.path → value`, tables and arrays of tables by their keys
/// and indices (`boundary.0.tag`), other arrays whole.
fn flatten_case(case: &Case) -> BTreeMap<String, toml::Value> {
    fn walk(prefix: &str, value: &toml::Value, out: &mut BTreeMap<String, toml::Value>) {
        let join = |key: &str| match prefix {
            "" => key.to_string(),
            _ => format!("{prefix}.{key}"),
        };
        match value {
            toml::Value::Table(table) => {
                for (key, v) in table {
                    walk(&join(key), v, out);
                }
            }
            toml::Value::Array(items)
                if !items.is_empty() && items.iter().all(|v| v.is_table()) =>
            {
                for (i, v) in items.iter().enumerate() {
                    walk(&join(&i.to_string()), v, out);
                }
            }
            v => {
                out.insert(prefix.to_string(), v.clone());
            }
        }
    }
    let value = toml::Value::try_from(case).expect("a case serializes to TOML");
    let mut out = BTreeMap::new();
    walk("", &value, &mut out);
    out
}

/// The driver's data of a case run's restart (see the module docs).
pub(super) fn driver_data(
    file: &CaseFile,
    stations: &[StationSeries],
    vtk_frames: usize,
    volume_start: f64,
) -> DriverData {
    let mut driver = DriverData::default();
    let build = crate::provenance::describe();
    driver.set_metadata("build", build.as_str());
    driver.set_text(
        "case",
        format!("# {build}\n{}", file.case.resolved().to_toml()),
    );
    driver.set_extra("output.counters", vec![vtk_frames as f64, volume_start]);
    for (j, s) in stations.iter().enumerate() {
        driver.set_metadata(&format!("station.{j}"), s.name.as_str());
        for (field, values) in station_fields(s) {
            driver.set_extra(&format!("station.{j}.{field}"), values.clone());
        }
    }
    driver
}

/// The VTK frames written and the volume at the start, from a restart's
/// driver data.
pub(super) fn counters(driver: &DriverData) -> Result<(usize, f64), CaseError> {
    match driver.extra("output.counters") {
        Some(&[frames, volume]) => Ok((frames as usize, volume)),
        _ => Err(CaseError::Invalid(
            "the restart has no output.counters".into(),
        )),
    }
}

fn station_fields(s: &StationSeries) -> [(&'static str, &Vec<f64>); 8] {
    [
        ("time_s", &s.time_s),
        ("eta_m", &s.eta_m),
        ("u_m_s", &s.u_m_s),
        ("v_m_s", &s.v_m_s),
        ("depth_m", &s.depth_m),
        ("hs_m", &s.hs_m),
        ("tp_s", &s.tp_s),
        ("wave_from_deg", &s.wave_from_deg),
    ]
}

/// Continue `stations`' records from a restart's driver data, which must
/// have the same stations.
pub(super) fn restore_stations(
    stations: &mut [StationSeries],
    driver: &DriverData,
) -> Result<(), CaseError> {
    let invalid = |m: String| CaseError::Invalid(format!("the restart's stations: {m}"));
    for (j, s) in stations.iter_mut().enumerate() {
        let theirs = driver.metadata(&format!("station.{j}"));
        if theirs != Some(s.name.as_str()) {
            return Err(invalid(format!(
                "station {j} is {}, the restart's {}",
                s.name,
                theirs.unwrap_or("(none)")
            )));
        }
        let record = |field: &str| {
            driver
                .extra(&format!("station.{j}.{field}"))
                .map(<[f64]>::to_vec)
                .ok_or_else(|| invalid(format!("no station.{j}.{field}")))
        };
        s.time_s = record("time_s")?;
        s.eta_m = record("eta_m")?;
        s.u_m_s = record("u_m_s")?;
        s.v_m_s = record("v_m_s")?;
        s.depth_m = record("depth_m")?;
        s.hs_m = record("hs_m")?;
        s.tp_s = record("tp_s")?;
        s.wave_from_deg = record("wave_from_deg")?;
    }
    if driver
        .metadata(&format!("station.{}", stations.len()))
        .is_some()
    {
        return Err(invalid("it has more stations than this run".into()));
    }
    Ok(())
}

/// When and where a run writes its restarts (`[restart]`): at the first
/// output at or after every multiple of the interval, to
/// `<output>/<name>.restart`, or with `keep` to
/// `<output>/<name>_<hours>h.restart` per time.
pub(super) struct RestartSchedule {
    dir: PathBuf,
    name: String,
    keep: bool,
    interval: f64,
    next: f64,
    /// The restarts written: their times and files
    pub written: Vec<(f64, PathBuf)>,
}

impl RestartSchedule {
    /// The schedule of `file`'s run from `t_start` (s); `None` without
    /// `restart.interval_hours`.
    pub fn new(file: &CaseFile, t_start: f64) -> Option<Self> {
        let r = file.case.restart.as_ref()?;
        let interval = 3600.0 * r.interval_hours?;
        Some(Self {
            dir: file.output_dir(),
            name: file.case.run.name.clone(),
            keep: r.keep,
            interval,
            next: ((t_start / interval + 1e-9).floor() + 1.0) * interval,
            written: Vec::new(),
        })
    }

    /// Whether a restart is due at `t`.
    pub fn due(&self, t: f64) -> bool {
        t >= self.next - 1e-6
    }

    fn path(&self, t: f64) -> PathBuf {
        match self.keep {
            true => self
                .dir
                .join(format!("{}_{:06.2}h.restart", self.name, t / 3600.0)),
            false => self.dir.join(format!("{}.restart", self.name)),
        }
    }

    /// Write `write`'s restart for time `t` and move on to the next
    /// multiple.
    pub fn write(
        &mut self,
        t: f64,
        write: impl FnOnce(&Path) -> Result<(), RestartError>,
    ) -> Result<(), CaseError> {
        let path = self.path(t);
        write(&path).map_err(|e| CaseError::Io(format!("{}: {e}", path.display())))?;
        self.written.push((t, path));
        while self.next <= t + 1e-6 {
            self.next += self.interval;
        }
        Ok(())
    }

    /// What the log says of the restarts written.
    pub fn note(&self) -> String {
        match self.written.last() {
            None => format!("restarts every {} h: none written", self.interval / 3600.0),
            Some((t, path)) => format!(
                "restarts every {} h: {} written, the last at {:.2} h to {}",
                self.interval / 3600.0,
                self.written.len(),
                t / 3600.0,
                path.display()
            ),
        }
    }
}

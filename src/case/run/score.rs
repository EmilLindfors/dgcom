//! `dgcom score`: a finished run's gauge scores again, from its station
//! records (`stations.csv`) and its case, without the run (TODO P8.1).
//!
//! A run scores its gauged stations at its end (`gauge_skill.csv`,
//! `gauge_constituents.csv`). What goes into the scores beside the records is
//! the gauges' records, the window (`output.spinup_hours`) and the model's
//! harmonic fit, whose candidates follow from what the boundaries carry
//! ([`CaseFile::tide_constituents`]). A change to any of them, or to the fit
//! itself in a later build, needs only the records, which `stations.csv`
//! holds to the bit: each value is written in the shortest form that reads
//! back as the same `f64`. So the run's own scores come back byte for byte.

use std::fs;
use std::path::Path;

use super::{StationSeries, restart, score_gauges, write_gauge_scores};
use crate::case::{Case, CaseError, CaseFile};

/// The keys of a case that the score reads from the run's case, beside the
/// records: the clock of the records (`run.start`), what the boundaries
/// carry, and the stations, but for each one's `gauge` ([`may_change`]).
const READS: &[&str] = &["run.start", "boundary", "output.station"];

/// What a score may take from the case given instead of the run's: each
/// station's gauge record (the window, `output.spinup_hours`, is not among
/// [`READS`]).
fn may_change(key: &str) -> bool {
    let mut parts = key.split('.');
    matches!(
        (parts.next(), parts.next(), parts.next(), parts.next(), parts.next()),
        (Some("output"), Some("station"), Some(i), Some("gauge"), None)
            if i.parse::<usize>().is_ok()
    )
}

/// A run scored again ([`CaseFile::score`]).
#[derive(Clone, Debug)]
pub struct ScoreReport {
    /// The stations' records, those with a gauge scored against it
    pub stations: Vec<StationSeries>,
    /// What the score read and did, line by line
    pub notes: Vec<String>,
}

impl CaseFile {
    /// Score the run in [`Self::output_dir`] again, as the run did at its
    /// end: read its `stations.csv`, score each station that has a gauge and
    /// write `gauge_skill.csv` and `gauge_constituents.csv` there, in place
    /// of the run's.
    ///
    /// The run's case (its `case.resolved.toml`) must resolve as this one
    /// does in what the score reads of it (`run.start`, the boundaries, the
    /// stations), but for each station's `gauge`. The
    /// window and the gauges are this case's, so a run can be scored from
    /// another spin-up (`--set output.spinup_hours=…`) or against another
    /// record; another run of the case is scored with
    /// `--set output.directory=…`.
    pub fn score(&self) -> Result<ScoreReport, CaseError> {
        let dir = self.output_dir();
        let mut notes = Vec::new();
        self.check_scored_run(&dir, &mut notes)?;
        let clock = self.case.clock()?.ok_or_else(|| {
            CaseError::Invalid("the case has no run.start, so no gauge to score against".into())
        })?;
        let mut gauges = Vec::new();
        for s in &self.case.output.stations {
            gauges.push(self.station_gauge(s)?.map(|(gauge, note)| {
                notes.push(note);
                gauge
            }));
        }
        if gauges.iter().all(Option::is_none) {
            return Err(CaseError::Invalid(
                "no [[output.station]] has a gauge to score against".into(),
            ));
        }
        let names: Vec<&str> = self
            .case
            .output
            .stations
            .iter()
            .map(|s| s.name.as_str())
            .collect();
        let mut stations = read_station_records(&dir.join("stations.csv"), &names)?;
        let carried = self.tide_constituents()?;
        notes.push(format!(
            "the model's record fitted with {}",
            crate::analysis::HarmonicAnalysis::station_candidates(&carried).join(" ")
        ));
        score_gauges(
            &carried,
            gauges.iter().map(Option::as_ref),
            &mut stations,
            &clock,
            3600.0 * self.case.spinup_hours(),
        );
        write_gauge_scores(&dir, &stations)?;
        Ok(ScoreReport { stations, notes })
    }

    /// Refuse to score the run in `dir` if its case differs from this one in
    /// what the score reads of it ([`READS`]); note the build that ran it.
    fn check_scored_run(&self, dir: &Path, notes: &mut Vec<String>) -> Result<(), CaseError> {
        let path = dir.join("case.resolved.toml");
        let text = fs::read_to_string(&path).map_err(|e| {
            CaseError::Io(format!(
                "{}: {e} (a case run's output directory holds the case it ran)",
                path.display()
            ))
        })?;
        if let Some(build) = text.lines().next().and_then(|l| l.strip_prefix("# ")) {
            notes.push(format!("the run in {}: {build}", dir.display()));
        }
        // As this build resolves it; a case from another build that no longer
        // reads is compared as written, in the keys it has (a key new in this
        // build is a default it did not know)
        let (theirs, as_written) = match Case::parse(&text, false, &[]) {
            Ok(case) => (restart::flatten_resolved(&case), false),
            Err(e) => {
                let value: toml::Value = toml::from_str(&text)
                    .map_err(|e| CaseError::Parse(format!("{}: {e}", path.display())))?;
                notes.push(format!(
                    "the run's case does not read in this build ({}); its stations, \
                     boundaries and start compared as written",
                    e.lines().next().unwrap_or_default()
                ));
                (restart::flatten(&value), true)
            }
        };
        let reads = |key: &str| {
            READS.iter().any(|p| restart::covers(p, key))
                && !may_change(key)
                && (!as_written || theirs.contains_key(key))
        };
        let ours = restart::flatten_resolved(&self.case);
        let differ = restart::differences(&ours, &theirs, reads, "the run's");
        if differ.is_empty() {
            return Ok(());
        }
        Err(CaseError::Invalid(format!(
            "cannot score the run in {}: its case differs in what the score reads: {}",
            dir.display(),
            differ.join("; ")
        )))
    }
}

/// The records of the stations `names` in a run's `stations.csv`, in that
/// order: `station,time_s,eta_m,u_m_s,v_m_s,depth_m`, with
/// `hs_m,tp_s,wave_from_deg` after a run with waves, a row per station and
/// time. A name may hold commas: the numbers are the last columns.
fn read_station_records(path: &Path, names: &[&str]) -> Result<Vec<StationSeries>, CaseError> {
    let invalid = |m: String| CaseError::Invalid(format!("{}: {m}", path.display()));
    let text =
        fs::read_to_string(path).map_err(|e| CaseError::Io(format!("{}: {e}", path.display())))?;
    let mut lines = text.lines();
    let header = lines.next().unwrap_or_default();
    let values = match header {
        "station,time_s,eta_m,u_m_s,v_m_s,depth_m" => 5,
        "station,time_s,eta_m,u_m_s,v_m_s,depth_m,hs_m,tp_s,wave_from_deg" => 8,
        _ => {
            return Err(invalid(format!(
                "not a case run's station records: {header:?}"
            )));
        }
    };
    let mut stations: Vec<StationSeries> = names
        .iter()
        .map(|name| StationSeries {
            name: name.to_string(),
            ..Default::default()
        })
        .collect();
    for (n, line) in lines.enumerate().filter(|(_, l)| !l.is_empty()) {
        let row = n + 2;
        let mut columns: Vec<&str> = line.rsplitn(values + 1, ',').collect();
        if columns.len() != values + 1 {
            return Err(invalid(format!("row {row} has {} columns", columns.len())));
        }
        columns.reverse();
        let name = columns[0];
        let s = stations
            .iter_mut()
            .find(|s| s.name == name)
            .ok_or_else(|| invalid(format!("row {row}: {name:?} is not a station of the case")))?;
        let x: Vec<f64> = columns[1..]
            .iter()
            .map(|c| c.parse::<f64>())
            .collect::<Result<_, _>>()
            .map_err(|e| invalid(format!("row {row}: {e}")))?;
        s.time_s.push(x[0]);
        s.eta_m.push(x[1]);
        s.u_m_s.push(x[2]);
        s.v_m_s.push(x[3]);
        s.depth_m.push(x[4]);
        if values == 8 {
            s.hs_m.push(x[5]);
            s.tp_s.push(x[6]);
            s.wave_from_deg.push(x[7]);
        }
    }
    let records = stations[0].time_s.len();
    if records == 0 {
        return Err(invalid("no records".into()));
    }
    if let Some(s) = stations.iter().find(|s| s.time_s.len() != records) {
        return Err(invalid(format!(
            "{} records of {:?}, {records} of {:?}",
            s.time_s.len(),
            s.name,
            stations[0].name
        )));
    }
    Ok(stations)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_station_name_with_commas_reads_with_its_records() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("stations.csv");
        fs::write(
            &path,
            "station,time_s,eta_m,u_m_s,v_m_s,depth_m\n\
             a,b,0,0.1,-0.2,0.3,10\n\
             c,0,1e-7,NaN,inf,5\n\
             a,b,600,0.30000000000000004,0,0,10\n\
             c,600,-1,0,0,5\n",
        )
        .unwrap();
        let s = read_station_records(&path, &["a,b", "c"]).unwrap();
        assert_eq!(s[0].time_s, [0.0, 600.0]);
        assert_eq!(s[0].eta_m, [0.1, 0.1 + 0.2]);
        assert_eq!(s[1].eta_m, [1e-7, -1.0]);
        assert!(s[1].u_m_s[0].is_nan() && s[1].v_m_s[0] == f64::INFINITY);
        assert!(s[0].hs_m.is_empty());

        let err = read_station_records(&path, &["a,b"])
            .unwrap_err()
            .to_string();
        assert!(err.contains("\"c\" is not a station of the case"), "{err}");
        let err = read_station_records(&path, &["a,b", "c", "d"])
            .unwrap_err()
            .to_string();
        assert!(err.contains("0 records of \"d\""), "{err}");
    }

    #[test]
    fn only_a_stations_gauge_may_change() {
        assert!(may_change("output.station.3.gauge"));
        for key in [
            "output.station.3.lon",
            "output.station.gauge",
            "output.station.x.gauge",
            "output.station.0.gauge.x",
            "boundary.0.atlas.gauge.record",
        ] {
            assert!(!may_change(key), "{key}");
        }
    }
}

//! Tide gauge observation file reader.
//!
//! Reads tide gauge water level observations from various file formats:
//! - Simple CSV/text format
//! - Norwegian Kartverket format
//! - Generic time series format
//!
//! # File Formats
//!
//! ## Simple Text Format
//!
//! ```text
//! # Tide gauge observations
//! # station: Bergen
//! # longitude: 5.32
//! # latitude: 60.39
//! # datum: MSL
//! # units: m
//! # columns: time(s) water_level(m)
//! 0.0 0.15
//! 3600.0 0.42
//! 7200.0 0.58
//! ```
//!
//! ## CSV Format
//!
//! ```text
//! time,water_level
//! 0.0,0.15
//! 3600.0,0.42
//! 7200.0,0.58
//! ```
//!
//! ## ISO DateTime Format
//!
//! ```text
//! # station: Bergen
//! # columns: datetime water_level(m)
//! 2024-01-01T00:00:00Z 0.15
//! 2024-01-01T01:00:00Z 0.42
//! 2024-01-01T02:00:00Z 0.58
//! ```

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use thiserror::Error;

use super::datetime::parse_datetime_seconds;
use crate::analysis::{TideGaugeStation, TimeSeries};

/// Error type for tide gauge file operations.
#[derive(Debug, Error)]
pub enum TideGaugeFileError {
    /// IO error reading file
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Parse error in file content
    #[error("Parse error: {0}")]
    ParseError(String),

    /// Missing required metadata
    #[error("Missing metadata: {0}")]
    MissingMetadata(String),

    /// Invalid file format
    #[error("Invalid format: {0}")]
    InvalidFormat(String),
}

/// Parsed tide gauge observation file.
#[derive(Clone, Debug)]
pub struct TideGaugeFile {
    /// Station metadata (if present in file)
    pub station: Option<TideGaugeStation>,
    /// Time series data
    pub time_series: TimeSeries,
    /// Original file path
    pub source_file: Option<String>,
    /// Reference datum (if specified)
    pub datum: Option<String>,
    /// Units (if specified)
    pub units: Option<String>,
}

impl TideGaugeFile {
    /// Create from time series without station metadata.
    pub fn from_time_series(time_series: TimeSeries) -> Self {
        Self {
            station: None,
            time_series,
            source_file: None,
            datum: None,
            units: None,
        }
    }

    /// Set station metadata.
    pub fn with_station(mut self, station: TideGaugeStation) -> Self {
        self.station = Some(station);
        self
    }

    /// Set source file path.
    pub fn with_source(mut self, path: impl Into<String>) -> Self {
        self.source_file = Some(path.into());
        self
    }
}

/// Read a tide gauge observation file.
///
/// Supports multiple formats:
/// - Space/tab-separated text with optional headers
/// - CSV with header row
/// - Comments starting with #
///
/// # Arguments
/// * `path` - Path to the observation file
///
/// # Returns
/// Parsed tide gauge file with time series and optional metadata.
pub fn read_tide_gauge_file(path: &Path) -> Result<TideGaugeFile, TideGaugeFileError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut metadata: HashMap<String, String> = HashMap::new();
    let mut times: Vec<f64> = Vec::new();
    let mut values: Vec<f64> = Vec::new();
    let mut is_csv = false;

    for (line_num, line_result) in reader.lines().enumerate() {
        let line = line_result?;
        let line = line.trim();

        // Skip empty lines
        if line.is_empty() {
            continue;
        }

        // Parse metadata comments
        if line.starts_with('#') {
            let content = line.trim_start_matches('#').trim();
            if let Some((key, value)) = content.split_once(':') {
                metadata.insert(key.trim().to_lowercase(), value.trim().to_string());
            }
            continue;
        }

        // Check for CSV header
        if line_num < 5 && line.contains(',') && line.chars().any(|c| c.is_alphabetic()) {
            is_csv = true;
            continue;
        }

        // Parse data line
        let parts: Vec<&str> = if is_csv || line.contains(',') {
            is_csv = true;
            line.split(',').map(|s| s.trim()).collect()
        } else {
            line.split_whitespace().collect()
        };

        if parts.len() < 2 {
            continue;
        }

        // Parse time (first column)
        let time = parse_time_value(parts[0]).map_err(|e| {
            TideGaugeFileError::ParseError(format!(
                "Line {}: time parse error: {}",
                line_num + 1,
                e
            ))
        })?;

        // Parse water level (second column)
        let value: f64 = parts[1].parse().map_err(|e| {
            TideGaugeFileError::ParseError(format!(
                "Line {}: water level parse error: {}",
                line_num + 1,
                e
            ))
        })?;

        times.push(time);
        values.push(value);
    }

    if times.is_empty() {
        return Err(TideGaugeFileError::InvalidFormat(
            "No data points found".to_string(),
        ));
    }

    // Build time series
    let mut time_series = TimeSeries::new(&times, &values);

    // Extract station metadata if present
    let station = if let Some(name) = metadata.get("station") {
        let lon = metadata
            .get("longitude")
            .or_else(|| metadata.get("lon"))
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0);
        let lat = metadata
            .get("latitude")
            .or_else(|| metadata.get("lat"))
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0);

        let mut station = TideGaugeStation::new(name.clone(), lon, lat);

        if let Some(id) = metadata.get("id").or_else(|| metadata.get("station_id")) {
            station = station.with_external_id(id.clone());
        }

        time_series = time_series.with_name(name.clone());
        if lon != 0.0 && lat != 0.0 {
            // Note: These are geographic coords, not local coords
            // Local coords would need to be set via projection
        }

        Some(station)
    } else {
        None
    };

    Ok(TideGaugeFile {
        station,
        time_series,
        source_file: Some(path.to_string_lossy().to_string()),
        datum: metadata.get("datum").cloned(),
        units: metadata.get("units").cloned(),
    })
}

/// Parse a time value from string.
///
/// Supports:
/// - Numeric seconds: "3600.0"
/// - Hours: "1.0h" or "1.0H"
/// - ISO 8601 / RFC3339 datetime: "2024-01-01T00:00:00Z", converted to Unix
///   seconds by the same exact proleptic-Gregorian parser the NorKyst readers
///   use, so gauge and model series share one absolute time axis
fn parse_time_value(s: &str) -> Result<f64, String> {
    let s = s.trim();

    // Check for hour suffix
    if s.ends_with('h') || s.ends_with('H') {
        let num = s.trim_end_matches(['h', 'H']);
        return match num.parse::<f64>() {
            Ok(h) if h.is_finite() => Ok(h * 3600.0),
            Ok(_) => Err(format!("non-finite time {s:?}")),
            Err(e) => Err(e.to_string()),
        };
    }

    parse_datetime_seconds(s).map_err(|e| format!("Unable to parse time value {s:?}: {e}"))
}

/// Read a list of tide gauges: a CSV file with a header row naming the
/// columns `name`, `lat` and `lon` (degrees north, and east within ±180),
/// in any order, and optionally `code` (the gauge's ID at its provider,
/// kept as [`TideGaugeStation::external_id`]). Other columns are ignored, as
/// are blank lines, lines starting with `#` and a UTF-8 byte-order mark.
/// Fields are not quoted: every row has exactly the header's number of
/// fields, so a comma inside a name is an error rather than a shifted row.
///
/// `scripts/norkyst_tide_gauges.csv` (Kartverket's 17 permanent gauges on
/// the Norwegian mainland) is such a file.
pub fn read_tide_gauge_stations(path: &Path) -> Result<Vec<TideGaugeStation>, TideGaugeFileError> {
    let mut lines = BufReader::new(File::open(path)?)
        .lines()
        .enumerate()
        .filter(|(_, line)| {
            line.as_ref().map_or(true, |l| {
                let l = l.trim_start_matches('\u{feff}').trim();
                !l.is_empty() && !l.starts_with('#')
            })
        });
    let (_, header) = lines
        .next()
        .ok_or_else(|| TideGaugeFileError::InvalidFormat("no header row".into()))?;
    let header = header?;
    let columns: Vec<&str> = header
        .trim_start_matches('\u{feff}')
        .split(',')
        .map(str::trim)
        .collect();
    let column = |name: &str| columns.iter().position(|c| c.eq_ignore_ascii_case(name));
    let required = |name: &str| {
        column(name).ok_or_else(|| {
            TideGaugeFileError::MissingMetadata(format!(
                "a station list needs a {name:?} column (the header is {header:?})"
            ))
        })
    };
    let (name, lat, lon) = (required("name")?, required("lat")?, required("lon")?);
    let code = column("code");

    let mut stations = Vec::new();
    for (index, line) in lines {
        let line = line?;
        let at = |message: String| {
            TideGaugeFileError::ParseError(format!("line {}: {message}", index + 1))
        };
        let fields: Vec<&str> = line.split(',').map(str::trim).collect();
        if fields.len() != columns.len() {
            return Err(at(format!(
                "{} fields, the header has {}",
                fields.len(),
                columns.len()
            )));
        }
        let degrees = |i: usize, what: &str, limit: f64| {
            let text = fields[i];
            let value: f64 = text
                .parse()
                .map_err(|_| at(format!("{what} {text:?} is not a number")))?;
            if value.is_finite() && value.abs() <= limit {
                Ok(value)
            } else {
                Err(at(format!("{what} {text:?} is not within ±{limit}°")))
            }
        };
        if fields[name].is_empty() {
            return Err(at("the name is empty".into()));
        }
        let mut station = TideGaugeStation::new(
            fields[name],
            degrees(lon, "longitude", 180.0)?,
            degrees(lat, "latitude", 90.0)?,
        );
        if let Some(code) = code.map(|i| fields[i]).filter(|c| !c.is_empty()) {
            station = station.with_external_id(code);
        }
        stations.push(station);
    }
    Ok(stations)
}

/// Read multiple tide gauge files from a directory.
///
/// Reads all .txt, .csv, and .dat files in the directory.
pub fn read_tide_gauge_directory(dir: &Path) -> Result<Vec<TideGaugeFile>, TideGaugeFileError> {
    let mut files = Vec::new();

    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_file()
            && let Some(ext) = path.extension()
        {
            let ext = ext.to_string_lossy().to_lowercase();
            if ext == "txt" || ext == "csv" || ext == "dat" {
                match read_tide_gauge_file(&path) {
                    Ok(file) => files.push(file),
                    Err(e) => {
                        eprintln!("Warning: Failed to read {:?}: {}", path, e);
                    }
                }
            }
        }
    }

    Ok(files)
}

/// Convert tide gauge files to a HashMap for validation.
///
/// Maps station name to time series.
pub fn files_to_observation_map(files: &[TideGaugeFile]) -> HashMap<String, TimeSeries> {
    let mut map = HashMap::new();

    for file in files {
        let name = file
            .station
            .as_ref()
            .map(|s| s.name.clone())
            .or_else(|| file.time_series.name.clone())
            .unwrap_or_else(|| {
                file.source_file
                    .as_ref()
                    .map(|p| {
                        Path::new(p)
                            .file_stem()
                            .map(|s| s.to_string_lossy().to_string())
                            .unwrap_or_default()
                    })
                    .unwrap_or_else(|| "unknown".to_string())
            });

        map.insert(name, file.time_series.clone());
    }

    map
}

/// Write tide gauge data to a file.
///
/// Writes in the simple text format with metadata headers.
pub fn write_tide_gauge_file(path: &Path, data: &TideGaugeFile) -> Result<(), TideGaugeFileError> {
    use std::io::Write;

    let mut file = File::create(path)?;

    // Write metadata
    writeln!(file, "# Tide gauge observations")?;

    if let Some(ref station) = data.station {
        writeln!(file, "# station: {}", station.name)?;
        writeln!(file, "# longitude: {:.4}", station.longitude)?;
        writeln!(file, "# latitude: {:.4}", station.latitude)?;
        if let Some(ref id) = station.external_id {
            writeln!(file, "# id: {}", id)?;
        }
    }

    if let Some(ref datum) = data.datum {
        writeln!(file, "# datum: {}", datum)?;
    }
    if let Some(ref units) = data.units {
        writeln!(file, "# units: {}", units)?;
    }

    writeln!(file, "# columns: time(s) water_level(m)")?;

    // Write data
    let series = &data.time_series;
    for (time, value) in series.times().iter().zip(series.values()) {
        writeln!(file, "{time:.1} {value:.6}")?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_parse_numeric_time() {
        assert!((parse_time_value("3600.0").unwrap() - 3600.0).abs() < 1e-10);
        assert!((parse_time_value("0").unwrap() - 0.0).abs() < 1e-10);
        assert!((parse_time_value("1.5h").unwrap() - 5400.0).abs() < 1e-10);
        assert!((parse_time_value("2H").unwrap() - 7200.0).abs() < 1e-10);
    }

    #[test]
    fn test_iso_time_is_exact_unix_seconds() {
        // Regression: the old parser used 365-day years and 30-day months, so
        // 2024-06-01 came out ~13 days away from the NorKyst readers' value.
        let expected = 19_875.0 * 86_400.0; // 2024-06-01 00:00:00 UTC
        let t = parse_time_value("2024-06-01T00:00:00Z").unwrap();
        assert!((t - expected).abs() < 1e-9, "t = {t}");
        assert_eq!(
            t,
            crate::io::datetime::parse_datetime_seconds("2024-06-01 00:00:00 UTC").unwrap()
        );
        // Spacing across a leap day is exact.
        let feb28 = parse_time_value("2024-02-28T12:00:00Z").unwrap();
        let mar01 = parse_time_value("2024-03-01T12:00:00Z").unwrap();
        assert!((mar01 - feb28 - 2.0 * 86_400.0).abs() < 1e-9);
        assert!(parse_time_value("2024-02-30T00:00:00Z").is_err());
        assert!(parse_time_value("NaN").is_err());
    }

    #[test]
    fn test_read_simple_format() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "# station: TestStation").unwrap();
        writeln!(file, "# longitude: 5.32").unwrap();
        writeln!(file, "# latitude: 60.39").unwrap();
        writeln!(file, "# columns: time water_level").unwrap();
        writeln!(file, "0.0 0.15").unwrap();
        writeln!(file, "3600.0 0.42").unwrap();
        writeln!(file, "7200.0 0.58").unwrap();

        let result = read_tide_gauge_file(file.path()).unwrap();

        assert!(result.station.is_some());
        let station = result.station.unwrap();
        assert_eq!(station.name, "TestStation");
        assert!((station.longitude - 5.32).abs() < 1e-10);
        assert!((station.latitude - 60.39).abs() < 1e-10);

        assert_eq!(result.time_series.len(), 3);
        assert!((result.time_series.values()[0] - 0.15).abs() < 1e-10);
        assert!((result.time_series.times()[1] - 3600.0).abs() < 1e-10);
    }

    #[test]
    fn test_read_csv_format() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "time,water_level").unwrap();
        writeln!(file, "0.0,0.15").unwrap();
        writeln!(file, "3600.0,0.42").unwrap();
        writeln!(file, "7200.0,0.58").unwrap();

        let result = read_tide_gauge_file(file.path()).unwrap();

        assert_eq!(result.time_series.len(), 3);
        assert!((result.time_series.values()[2] - 0.58).abs() < 1e-10);
    }

    /// The station list the repository ships (Kartverket's permanent
    /// gauges), which replaced the approximate positions once held in code
    /// (Heimsjø and Bodø ≈ 0.5 km off, Bergen 0.9 km).
    #[test]
    fn the_shipped_station_list_reads() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/norkyst_tide_gauges.csv");
        let stations = read_tide_gauge_stations(&path).unwrap();
        assert_eq!(stations.len(), 17);
        let find = |name: &str| stations.iter().find(|s| s.name == name).unwrap();
        let heimsjo = find("Heimsjøen");
        assert_eq!((heimsjo.longitude, heimsjo.latitude), (9.101504, 63.425224));
        assert_eq!(heimsjo.external_id.as_deref(), Some("HEI"));
        assert_eq!(find("Bodø").external_id.as_deref(), Some("BOO"));
        // South to north along the coast, all on the mainland
        assert!(stations.iter().all(|s| (57.0..72.0).contains(&s.latitude)));
    }

    #[test]
    fn a_station_list_takes_its_columns_by_name() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "# SHOA gauges\nlon, name, datum, lat").unwrap();
        writeln!(file, "-72.9667, Puerto Montt, NRS, -41.4833\n").unwrap();
        writeln!(file, "-73.8333, Ancud, NRS, -41.8667").unwrap();
        let stations = read_tide_gauge_stations(file.path()).unwrap();
        assert_eq!(stations.len(), 2);
        assert_eq!(stations[0].name, "Puerto Montt");
        assert_eq!(
            (stations[1].longitude, stations[1].latitude),
            (-73.8333, -41.8667)
        );
        assert_eq!(stations[0].external_id, None);
    }

    #[test]
    fn a_bad_station_list_names_what_is_wrong() {
        let read = |text: &str| {
            let mut file = NamedTempFile::new().unwrap();
            write!(file, "{text}").unwrap();
            read_tide_gauge_stations(file.path())
                .unwrap_err()
                .to_string()
        };
        assert!(read("").contains("no header"));
        assert!(read("name,latitude,lon\nA,1,2\n").contains("\"lat\""));
        assert!(
            read("name,lat,lon\nA,95.0,2\n").contains("line 2: latitude \"95.0\" is not within")
        );
        assert!(read("name,lat,lon\nA,1,-300\n").contains("longitude \"-300\" is not within ±180"));
        assert!(read("name,lat,lon\nA,abc,2\n").contains("latitude \"abc\" is not a number"));
        assert!(read("name,lat,lon\nA,1,inf\n").contains("longitude \"inf\" is not within"));
        assert!(read("name,lat,lon\nA,1,2\nB,1\n").contains("line 3: 2 fields, the header has 3"));
        // An unquoted comma in a name shifts the row: refused, not misread
        assert!(read("name,lat,lon\nBodø, havn,67.29,14.40\n").contains("4 fields"));
        assert!(read("name,lat,lon\n ,1,2\n").contains("line 2: the name is empty"));
    }

    /// A spreadsheet's CSV export starts with a byte-order mark.
    #[test]
    fn a_station_list_with_a_byte_order_mark_reads() {
        let mut file = NamedTempFile::new().unwrap();
        write!(file, "\u{feff}name,lat,lon\nA,1,2\n").unwrap();
        let stations = read_tide_gauge_stations(file.path()).unwrap();
        assert_eq!(stations[0].name, "A");
    }

    #[test]
    fn test_write_and_read() {
        let station = TideGaugeStation::new("Bergen", 5.32, 60.39);
        let times = vec![0.0, 3600.0, 7200.0];
        let values = vec![0.1, 0.5, 0.3];
        let ts = TimeSeries::new(&times, &values);

        let data = TideGaugeFile::from_time_series(ts).with_station(station);

        let file = NamedTempFile::new().unwrap();
        write_tide_gauge_file(file.path(), &data).unwrap();

        let read_back = read_tide_gauge_file(file.path()).unwrap();

        assert!(read_back.station.is_some());
        assert_eq!(read_back.station.unwrap().name, "Bergen");
        assert_eq!(read_back.time_series.len(), 3);
    }

    #[test]
    fn test_files_to_map() {
        let station = TideGaugeStation::new("Test", 0.0, 0.0);
        let ts = TimeSeries::new(&[0.0, 1.0], &[0.5, 0.6]);
        let file = TideGaugeFile::from_time_series(ts).with_station(station);

        let map = files_to_observation_map(&[file]);

        assert!(map.contains_key("Test"));
        assert_eq!(map.get("Test").unwrap().len(), 2);
    }

    #[test]
    fn test_empty_file_error() {
        let file = NamedTempFile::new().unwrap();
        writeln!(&file, "# Just comments").unwrap();

        let result = read_tide_gauge_file(file.path());
        assert!(result.is_err());
    }
}

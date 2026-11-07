//! NetCDF I/O for oceanographic simulations.
//!
//! This module provides CF-conventions compliant NetCDF output for simulation
//! results. Parent-model and weather-model input is read in `netcdf_read`
//! (`OceanModelReader::from_file`, `AtmosphereReader::from_file`).
//!
//! # CF-Conventions
//!
//! Output files follow CF-1.8 conventions:
//! - Standard coordinate variables (time, lat, lon)
//! - Standard names for variables (sea_surface_height, eastward_sea_water_velocity, etc.)
//! - Time encoded as "seconds since 1970-01-01" (Unix epoch)
//!
//! # Example
//!
//! ```rust,ignore
//! use dgcom::io::{NetCDFWriter, NetCDFWriterConfig};
//!
//! let config = NetCDFWriterConfig::new("output.nc")
//!     .with_title("Froya Simulation")
//!     .with_institution("NTNU");
//!
//! let mut writer = NetCDFWriter::create(config, &mesh)?;
//! writer.write_timestep(0.0, &solution)?;
//! ```

#[cfg(feature = "netcdf")]
use chrono::Utc;
#[cfg(feature = "netcdf")]
use netcdf::create;
use thiserror::Error;

use crate::time::ModelClock;

/// Error type for NetCDF operations.
#[derive(Debug, Error)]
pub enum NetCDFError {
    /// File I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// NetCDF library error
    #[cfg(feature = "netcdf")]
    #[error("NetCDF error: {0}")]
    NetCDF(#[from] netcdf::Error),

    /// Invalid data
    #[error("Invalid data: {0}")]
    InvalidData(String),

    /// Missing variable
    #[error("Missing variable: {0}")]
    MissingVariable(String),

    /// Feature not enabled
    #[error("NetCDF feature not enabled")]
    FeatureDisabled,
}

/// Fill value for missing data (CF-conventions standard).
pub const FILL_VALUE_F64: f64 = 9.96920996838687e+36;
pub const FILL_VALUE_F32: f32 = 9.96921e+36;

/// Check if a value is valid (not a fill value).
#[inline]
pub fn is_valid_f32(v: f32) -> bool {
    v.is_finite() && v.abs() < 1.0e+30
}

/// Check if a value is valid (not a fill value).
#[inline]
pub fn is_valid_f64(v: f64) -> bool {
    v.is_finite() && v.abs() < 1.0e+30
}

// ============================================================================
// NetCDF Writer
// ============================================================================

/// Configuration for NetCDF output.
#[derive(Debug, Clone)]
pub struct NetCDFWriterConfig {
    /// Output file path
    pub path: String,
    /// Title attribute (CF-conventions)
    pub title: Option<String>,
    /// Institution attribute
    pub institution: Option<String>,
    /// Source attribute (model name/version)
    pub source: Option<String>,
    /// History attribute (processing steps)
    pub history: Option<String>,
    /// References attribute
    pub references: Option<String>,
    /// Comment attribute
    pub comment: Option<String>,
    /// Whether to include velocity components (u, v)
    pub include_velocity: bool,
    /// Whether to include tracers (temperature, salinity)
    pub include_tracers: bool,
    /// Compression level (0-9, 0=none)
    pub compression_level: u8,
    /// UTC instant of simulation time 0; the `time` variable is written in
    /// simulation seconds with units `seconds since <epoch>`. Default: the
    /// Unix epoch (simulation time is Unix time).
    pub clock: ModelClock,
}

impl NetCDFWriterConfig {
    /// Create a new configuration with the given output path.
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            title: None,
            institution: None,
            source: Some("dgcom".to_string()),
            history: None,
            references: None,
            comment: None,
            include_velocity: true,
            include_tracers: false,
            compression_level: 4,
            clock: ModelClock::default(),
        }
    }

    /// Set the model clock, so the `time` units name the run's epoch.
    pub fn with_clock(mut self, clock: ModelClock) -> Self {
        self.clock = clock;
        self
    }

    /// Set the title attribute.
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Set the institution attribute.
    pub fn with_institution(mut self, institution: impl Into<String>) -> Self {
        self.institution = Some(institution.into());
        self
    }

    /// Set the source attribute.
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Set the comment attribute.
    pub fn with_comment(mut self, comment: impl Into<String>) -> Self {
        self.comment = Some(comment.into());
        self
    }

    /// Enable/disable velocity output.
    pub fn with_velocity(mut self, include: bool) -> Self {
        self.include_velocity = include;
        self
    }

    /// Enable/disable tracer output.
    pub fn with_tracers(mut self, include: bool) -> Self {
        self.include_tracers = include;
        self
    }

    /// Set compression level (0-9).
    pub fn with_compression(mut self, level: u8) -> Self {
        self.compression_level = level.min(9);
        self
    }
}

/// Mesh information for NetCDF output.
#[derive(Debug, Clone)]
pub struct NetCDFMeshInfo {
    /// Number of nodes
    pub n_nodes: usize,
    /// Node x-coordinates
    pub x: Vec<f64>,
    /// Node y-coordinates
    pub y: Vec<f64>,
    /// Node latitudes (if available)
    pub lat: Option<Vec<f64>>,
    /// Node longitudes (if available)
    pub lon: Option<Vec<f64>>,
}

impl NetCDFMeshInfo {
    /// Create mesh info from coordinates.
    pub fn from_xy(x: Vec<f64>, y: Vec<f64>) -> Self {
        let n_nodes = x.len();
        Self {
            n_nodes,
            x,
            y,
            lat: None,
            lon: None,
        }
    }

    /// Create mesh info from lat/lon coordinates.
    pub fn from_latlon(lat: Vec<f64>, lon: Vec<f64>) -> Self {
        let n_nodes = lat.len();
        Self {
            n_nodes,
            x: lon.clone(), // Use lon as x
            y: lat.clone(), // Use lat as y
            lat: Some(lat),
            lon: Some(lon),
        }
    }

    /// Add lat/lon coordinates to existing mesh info.
    pub fn with_latlon(mut self, lat: Vec<f64>, lon: Vec<f64>) -> Self {
        self.lat = Some(lat);
        self.lon = Some(lon);
        self
    }
}

/// NetCDF writer for simulation output.
#[cfg(feature = "netcdf")]
pub struct NetCDFWriter {
    file: netcdf::FileMut,
    config: NetCDFWriterConfig,
    n_nodes: usize,
    time_index: usize,
}

#[cfg(feature = "netcdf")]
impl NetCDFWriter {
    /// Create a new NetCDF file for writing.
    pub fn create(config: NetCDFWriterConfig, mesh: &NetCDFMeshInfo) -> Result<Self, NetCDFError> {
        let mut file = create(&config.path)?;

        // Add dimensions
        file.add_unlimited_dimension("time")?;
        file.add_dimension("node", mesh.n_nodes)?;

        // Add coordinate variables
        {
            let mut time_var = file.add_variable::<f64>("time", &["time"])?;
            time_var.put_attribute("standard_name", "time")?;
            time_var.put_attribute("long_name", "simulation time")?;
            time_var.put_attribute("units", config.clock.cf_time_units().as_str())?;
            time_var.put_attribute("calendar", "standard")?;
        }

        // Add x coordinate
        {
            let mut x_var = file.add_variable::<f64>("x", &["node"])?;
            x_var.put_attribute("standard_name", "projection_x_coordinate")?;
            x_var.put_attribute("long_name", "x coordinate")?;
            x_var.put_attribute("units", "m")?;
            x_var.put_values(&mesh.x, ..)?;
        }

        // Add y coordinate
        {
            let mut y_var = file.add_variable::<f64>("y", &["node"])?;
            y_var.put_attribute("standard_name", "projection_y_coordinate")?;
            y_var.put_attribute("long_name", "y coordinate")?;
            y_var.put_attribute("units", "m")?;
            y_var.put_values(&mesh.y, ..)?;
        }

        // Add lat/lon if available
        if let Some(ref lat) = mesh.lat {
            let mut lat_var = file.add_variable::<f64>("lat", &["node"])?;
            lat_var.put_attribute("standard_name", "latitude")?;
            lat_var.put_attribute("long_name", "latitude")?;
            lat_var.put_attribute("units", "degrees_north")?;
            lat_var.put_values(lat, ..)?;
        }

        if let Some(ref lon) = mesh.lon {
            let mut lon_var = file.add_variable::<f64>("lon", &["node"])?;
            lon_var.put_attribute("standard_name", "longitude")?;
            lon_var.put_attribute("long_name", "longitude")?;
            lon_var.put_attribute("units", "degrees_east")?;
            lon_var.put_values(lon, ..)?;
        }

        // Add data variables
        {
            let mut eta_var = file.add_variable::<f32>("eta", &["time", "node"])?;
            eta_var.put_attribute("standard_name", "sea_surface_height_above_geoid")?;
            eta_var.put_attribute("long_name", "sea surface elevation")?;
            eta_var.put_attribute("units", "m")?;
            eta_var.put_attribute("_FillValue", FILL_VALUE_F32)?;
        }

        {
            let mut h_var = file.add_variable::<f32>("h", &["time", "node"])?;
            h_var.put_attribute("standard_name", "sea_floor_depth_below_sea_surface")?;
            h_var.put_attribute("long_name", "water depth")?;
            h_var.put_attribute("units", "m")?;
            h_var.put_attribute("_FillValue", FILL_VALUE_F32)?;
        }

        if config.include_velocity {
            {
                let mut u_var = file.add_variable::<f32>("u", &["time", "node"])?;
                u_var.put_attribute("standard_name", "eastward_sea_water_velocity")?;
                u_var.put_attribute("long_name", "eastward velocity")?;
                u_var.put_attribute("units", "m s-1")?;
                u_var.put_attribute("_FillValue", FILL_VALUE_F32)?;
            }

            {
                let mut v_var = file.add_variable::<f32>("v", &["time", "node"])?;
                v_var.put_attribute("standard_name", "northward_sea_water_velocity")?;
                v_var.put_attribute("long_name", "northward velocity")?;
                v_var.put_attribute("units", "m s-1")?;
                v_var.put_attribute("_FillValue", FILL_VALUE_F32)?;
            }

            {
                let mut speed_var = file.add_variable::<f32>("speed", &["time", "node"])?;
                speed_var.put_attribute("standard_name", "sea_water_speed")?;
                speed_var.put_attribute("long_name", "current speed")?;
                speed_var.put_attribute("units", "m s-1")?;
                speed_var.put_attribute("_FillValue", FILL_VALUE_F32)?;
            }
        }

        if config.include_tracers {
            {
                let mut temp_var = file.add_variable::<f32>("temperature", &["time", "node"])?;
                temp_var.put_attribute("standard_name", "sea_water_temperature")?;
                temp_var.put_attribute("long_name", "sea water temperature")?;
                temp_var.put_attribute("units", "degC")?;
                temp_var.put_attribute("_FillValue", FILL_VALUE_F32)?;
            }

            {
                let mut sal_var = file.add_variable::<f32>("salinity", &["time", "node"])?;
                sal_var.put_attribute("standard_name", "sea_water_salinity")?;
                sal_var.put_attribute("long_name", "sea water salinity")?;
                sal_var.put_attribute("units", "1e-3")?; // PSU = g/kg = 1e-3
                sal_var.put_attribute("_FillValue", FILL_VALUE_F32)?;
            }
        }

        // Add global attributes
        file.add_attribute("Conventions", "CF-1.8")?;
        file.add_attribute("featureType", "point")?;

        if let Some(ref title) = config.title {
            file.add_attribute("title", title.as_str())?;
        }
        if let Some(ref institution) = config.institution {
            file.add_attribute("institution", institution.as_str())?;
        }
        if let Some(ref source) = config.source {
            file.add_attribute("source", source.as_str())?;
        }
        if let Some(ref comment) = config.comment {
            file.add_attribute("comment", comment.as_str())?;
        }

        // Add creation timestamp
        let now = Utc::now();
        file.add_attribute(
            "history",
            format!("{}: Created by dgcom", now.format("%Y-%m-%d %H:%M:%S UTC")).as_str(),
        )?;

        Ok(Self {
            file,
            config,
            n_nodes: mesh.n_nodes,
            time_index: 0,
        })
    }

    /// Write a timestep to the file.
    ///
    /// # Arguments
    /// * `time` - Simulation time in seconds
    /// * `h` - Water depth at each node
    /// * `eta` - Surface elevation at each node
    /// * `u` - Eastward velocity (optional)
    /// * `v` - Northward velocity (optional)
    pub fn write_timestep(
        &mut self,
        time: f64,
        h: &[f64],
        eta: &[f64],
        u: Option<&[f64]>,
        v: Option<&[f64]>,
    ) -> Result<(), NetCDFError> {
        let t_idx = self.time_index;
        for (name, field) in [("h", Some(h)), ("eta", Some(eta)), ("u", u), ("v", v)] {
            if let Some(field) = field.filter(|f| f.len() != self.n_nodes) {
                return Err(NetCDFError::InvalidData(format!(
                    "{name} has {} values for {} nodes",
                    field.len(),
                    self.n_nodes
                )));
            }
        }

        // Write time
        {
            let mut time_var = self
                .file
                .variable_mut("time")
                .ok_or_else(|| NetCDFError::MissingVariable("time".to_string()))?;
            time_var.put_value(time, [t_idx])?;
        }

        // Write h
        {
            let h_f32: Vec<f32> = h.iter().map(|&x| x as f32).collect();
            let mut h_var = self
                .file
                .variable_mut("h")
                .ok_or_else(|| NetCDFError::MissingVariable("h".to_string()))?;
            h_var.put_values(&h_f32, (t_idx, ..))?;
        }

        // Write eta
        {
            let eta_f32: Vec<f32> = eta.iter().map(|&x| x as f32).collect();
            let mut eta_var = self
                .file
                .variable_mut("eta")
                .ok_or_else(|| NetCDFError::MissingVariable("eta".to_string()))?;
            eta_var.put_values(&eta_f32, (t_idx, ..))?;
        }

        // Write velocities if provided
        if self.config.include_velocity
            && let (Some(u_data), Some(v_data)) = (u, v)
        {
            let u_f32: Vec<f32> = u_data.iter().map(|&x| x as f32).collect();
            let v_f32: Vec<f32> = v_data.iter().map(|&x| x as f32).collect();
            let speed_f32: Vec<f32> = u_data
                .iter()
                .zip(v_data.iter())
                .map(|(&u, &v)| (u * u + v * v).sqrt() as f32)
                .collect();

            {
                let mut u_var = self
                    .file
                    .variable_mut("u")
                    .ok_or_else(|| NetCDFError::MissingVariable("u".to_string()))?;
                u_var.put_values(&u_f32, (t_idx, ..))?;
            }

            {
                let mut v_var = self
                    .file
                    .variable_mut("v")
                    .ok_or_else(|| NetCDFError::MissingVariable("v".to_string()))?;
                v_var.put_values(&v_f32, (t_idx, ..))?;
            }

            {
                let mut speed_var = self
                    .file
                    .variable_mut("speed")
                    .ok_or_else(|| NetCDFError::MissingVariable("speed".to_string()))?;
                speed_var.put_values(&speed_f32, (t_idx, ..))?;
            }
        }

        self.time_index += 1;
        Ok(())
    }

    /// Write tracer data for a timestep.
    pub fn write_tracers(
        &mut self,
        temperature: Option<&[f64]>,
        salinity: Option<&[f64]>,
    ) -> Result<(), NetCDFError> {
        if !self.config.include_tracers {
            return Ok(());
        }

        let t_idx = self.time_index.saturating_sub(1); // Use previous time index

        if let Some(temp) = temperature {
            let temp_f32: Vec<f32> = temp.iter().map(|&x| x as f32).collect();
            let mut temp_var = self
                .file
                .variable_mut("temperature")
                .ok_or_else(|| NetCDFError::MissingVariable("temperature".to_string()))?;
            temp_var.put_values(&temp_f32, (t_idx, ..))?;
        }

        if let Some(sal) = salinity {
            let sal_f32: Vec<f32> = sal.iter().map(|&x| x as f32).collect();
            let mut sal_var = self
                .file
                .variable_mut("salinity")
                .ok_or_else(|| NetCDFError::MissingVariable("salinity".to_string()))?;
            sal_var.put_values(&sal_f32, (t_idx, ..))?;
        }

        Ok(())
    }

    /// Get the number of timesteps written.
    pub fn n_timesteps(&self) -> usize {
        self.time_index
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fill_value_check() {
        assert!(is_valid_f32(10.0));
        assert!(is_valid_f32(-5.0));
        assert!(!is_valid_f32(f32::NAN));
        assert!(!is_valid_f32(f32::INFINITY));
        assert!(!is_valid_f32(FILL_VALUE_F32));
        assert!(!is_valid_f32(1.0e31));
    }

    #[test]
    fn test_netcdf_config() {
        let config = NetCDFWriterConfig::new("test.nc")
            .with_title("Test Simulation")
            .with_institution("Test University")
            .with_compression(6);

        assert_eq!(config.path, "test.nc");
        assert_eq!(config.title, Some("Test Simulation".to_string()));
        assert_eq!(config.compression_level, 6);
    }

    /// The `time` variable holds simulation seconds, so its units must name
    /// the model epoch (they used to claim the Unix epoch).
    #[test]
    fn writer_time_units_name_the_model_epoch() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("out.nc");
        let clock = ModelClock::parse("2025-06-01T00:00:00Z").unwrap();
        let config = NetCDFWriterConfig::new(path.to_string_lossy()).with_clock(clock);
        let mesh = NetCDFMeshInfo::from_xy(vec![0.0, 1.0], vec![0.0, 0.0]);
        let mut writer = NetCDFWriter::create(config, &mesh).unwrap();
        writer
            .write_timestep(3600.0, &[1.0, 1.0], &[0.0, 0.0], None, None)
            .unwrap();
        drop(writer);

        let file = netcdf::open(&path).unwrap();
        let time = file.variable("time").unwrap();
        let units = super::super::netcdf_read::attr_string(&time, "units").unwrap();
        assert_eq!(units, "seconds since 2025-06-01 00:00:00");
        let (scale, reference) = super::super::datetime::parse_cf_time_units(&units, None).unwrap();
        let t: f64 = time.get_value([0]).unwrap();
        assert_eq!(reference + scale * t, clock.unix(3600.0));
    }

    /// A field of the wrong length is refused before anything is written.
    #[test]
    fn writer_refuses_fields_of_the_wrong_length() {
        let dir = tempfile::tempdir().unwrap();
        let config = NetCDFWriterConfig::new(dir.path().join("out.nc").to_string_lossy());
        let mesh = NetCDFMeshInfo::from_xy(vec![0.0, 1.0], vec![0.0, 0.0]);
        let mut writer = NetCDFWriter::create(config, &mesh).unwrap();
        let (two, three) = ([0.0; 2], [0.0; 3]);
        let error = writer
            .write_timestep(0.0, &two, &two, Some(&three), Some(&two))
            .unwrap_err();
        assert!(
            matches!(&error, NetCDFError::InvalidData(m) if m == "u has 3 values for 2 nodes"),
            "{error}"
        );
        assert!(
            writer
                .write_timestep(0.0, &three, &two, None, None)
                .is_err()
        );
        writer
            .write_timestep(0.0, &two, &two, Some(&two), Some(&two))
            .unwrap();
    }

    #[test]
    fn test_mesh_info() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 1.0, 2.0];
        let mesh = NetCDFMeshInfo::from_xy(x.clone(), y.clone());

        assert_eq!(mesh.n_nodes, 3);
        assert_eq!(mesh.x, x);
        assert_eq!(mesh.y, y);
        assert!(mesh.lat.is_none());
    }
}

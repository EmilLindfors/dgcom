//! A parent wave model's 2D spectra at points (TODO F.4 forcing).
//!
//! MET Norway's MyWave WAM 800 m publishes, next to its gridded fields, the
//! full spectrum at a few dozen points per area, hourly over the forecast
//! (`https://thredds.met.no/thredds/dodsC/fou-hi/mywavewam800m/MyWave_wam800_c2SPC00.nc`
//! for Midt-Norge, the 00 UTC run; `…SPC12.nc` the 12 UTC one):
//!
//! - `SPEC[time][y][x][freq][direction]`: variance density in m²/(Hz·rad)
//!   (MET writes the units as `m**2 s`); `4 √(∫∫ SPEC df dθ)` with θ in
//!   radians is the file's `hs` (checked on 2025-10-08: equal at the first
//!   point, within 4 % at the others with bin widths from the frequency
//!   centres);
//! - `direction`: the direction the waves travel to, in degrees clockwise
//!   from north (`direction_convention`: "propagating towards the North
//!   (Oceanographic convention)"); a `standard_name` with `from` in it
//!   (WAVEWATCH III's `efth`) is turned around;
//! - `freq` (Hz), `time` (CF units), `latitude`/`longitude` per point, and
//!   `hs`, `tp`, `depth`, the wind `ff` (m/s) and `dd` (the direction it
//!   blows to, degrees from north) when present.
//!
//! [`WaveSpectraFile`] reads that layout from a file or an OPeNDAP URL,
//! keeps a subset of its points ([`WaveSpectraFile::select`]), writes it back
//! in the same layout ([`WaveSpectraFile::write`]), and turns it into a
//! [`PointSpectra`] in a model's mesh coordinates and time, and into a
//! [`WindSeries`].

use std::path::Path;

use crate::waves::{PointSpectra, Wind, WindSeries};

use super::netcdf_io::NetCDFError;
use super::netcdf_read::{attr_string, read_time, read_values};

/// Point spectra of a parent wave model (see the module docs).
#[derive(Clone, Debug, PartialEq)]
pub struct WaveSpectraFile {
    /// Point positions (degrees)
    pub longitude: Vec<f64>,
    pub latitude: Vec<f64>,
    /// The parent's depth at each point (m), if given
    pub depth: Option<Vec<f64>>,
    /// Times (Unix seconds), ascending
    pub times: Vec<f64>,
    /// The forecast's reference time (Unix seconds), if given
    pub forecast_reference_time: Option<f64>,
    /// Frequencies (Hz), ascending
    pub frequencies: Vec<f64>,
    /// The directions the waves travel to (degrees clockwise from north)
    pub directions_to: Vec<f64>,
    /// Variance density (m²/(Hz·rad)), `[time][point][frequency][direction]`;
    /// fill values are NaN
    pub density: Vec<f64>,
    /// Significant wave height (m), `[time][point]`, if given
    pub hs: Option<Vec<f64>>,
    /// Peak period (s), `[time][point]`, if given
    pub tp: Option<Vec<f64>>,
    /// The wind speed (m/s) and the direction it blows to (degrees from
    /// north), `[time][point]`, if given
    pub wind: Option<(Vec<f64>, Vec<f64>)>,
}

impl WaveSpectraFile {
    /// Number of points.
    pub fn n_points(&self) -> usize {
        self.longitude.len()
    }

    /// Read a file, or an OPeNDAP URL (netCDF-C with DAP).
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, NetCDFError> {
        let file = netcdf::open(path.as_ref())?;
        let var = |names: &[&str]| names.iter().find_map(|n| file.variable(n));
        let missing = |what: &str| NetCDFError::MissingVariable(what.to_string());
        let spec = var(&["SPEC", "efth"]).ok_or_else(|| missing("SPEC (or efth)"))?;
        let freq_var = var(&["freq", "frequency"]).ok_or_else(|| missing("freq"))?;
        let dir_var = var(&["direction"]).ok_or_else(|| missing("direction"))?;
        let frequencies = read_values(&freq_var, vec![])?;
        let directions = read_values(&dir_var, vec![])?;
        let from = attr_string(&dir_var, "standard_name").is_some_and(|s| s.contains("from"));
        let directions_to = directions
            .iter()
            .map(|d| {
                if from {
                    (d + 180.0).rem_euclid(360.0)
                } else {
                    *d
                }
            })
            .collect();
        let (times, _) = read_time(&file)?;
        let latitude = read_values(
            &var(&["latitude", "lat"]).ok_or_else(|| missing("latitude"))?,
            vec![],
        )?;
        let longitude = read_values(
            &var(&["longitude", "lon"]).ok_or_else(|| missing("longitude"))?,
            vec![],
        )?;
        let (nt, np, nf, nd) = (
            times.len(),
            longitude.len(),
            frequencies.len(),
            directions.len(),
        );
        if latitude.len() != np {
            return Err(NetCDFError::InvalidData(format!(
                "{} latitudes for {np} longitudes",
                latitude.len()
            )));
        }
        let density = read_values(&spec, vec![])?;
        if density.len() != nt * np * nf * nd {
            return Err(NetCDFError::InvalidData(format!(
                "{} values in {}, expected time × points × frequencies × directions = {}",
                density.len(),
                spec.name(),
                nt * np * nf * nd
            )));
        }
        let field = |name: &str, len: usize| -> Result<Option<Vec<f64>>, NetCDFError> {
            match file.variable(name) {
                Some(v) => {
                    let values = read_values(&v, vec![])?;
                    if values.len() == len {
                        Ok(Some(values))
                    } else {
                        Err(NetCDFError::InvalidData(format!(
                            "{name}: {} values, expected {len}",
                            values.len()
                        )))
                    }
                }
                None => Ok(None),
            }
        };
        let forecast_reference_time = match file.variable("forecast_reference_time") {
            Some(v) => {
                let value = read_values(&v, vec![])?.first().copied();
                let units = attr_string(&v, "units");
                // Informative only: units the CF parser does not know (MET's
                // "days since 1970-01-01 00:00:00 +00:00") leave it out
                let units = units.map(|u| u.trim_end_matches(" +00:00").to_string());
                match (value, units) {
                    (Some(x), Some(units)) => super::datetime::parse_cf_time_units(&units, None)
                        .ok()
                        .map(|(seconds, reference)| reference + x * seconds),
                    _ => None,
                }
            }
            None => None,
        };
        let wind = match (field("ff", nt * np)?, field("dd", nt * np)?) {
            (Some(ff), Some(dd)) => Some((ff, dd)),
            _ => None,
        };
        Ok(Self {
            longitude,
            latitude,
            depth: field("depth", np)?,
            times,
            forecast_reference_time,
            frequencies,
            directions_to,
            density,
            hs: field("hs", nt * np)?,
            tp: field("tp", nt * np)?,
            wind,
        })
    }

    /// The points `points` (indices, in this order) alone.
    pub fn select(&self, points: &[usize]) -> Self {
        let (nt, np) = (self.times.len(), self.n_points());
        let nc = self.frequencies.len() * self.directions_to.len();
        let pick = |v: &[f64]| points.iter().map(|&p| v[p]).collect::<Vec<_>>();
        let per_time = |v: &[f64]| {
            (0..nt)
                .flat_map(|t| points.iter().map(move |&p| v[t * np + p]))
                .collect::<Vec<_>>()
        };
        let density = (0..nt)
            .flat_map(|t| {
                points.iter().flat_map(move |&p| {
                    self.density[(t * np + p) * nc..(t * np + p + 1) * nc]
                        .iter()
                        .copied()
                })
            })
            .collect();
        Self {
            longitude: pick(&self.longitude),
            latitude: pick(&self.latitude),
            depth: self.depth.as_deref().map(pick),
            times: self.times.clone(),
            forecast_reference_time: self.forecast_reference_time,
            frequencies: self.frequencies.clone(),
            directions_to: self.directions_to.clone(),
            density,
            hs: self.hs.as_deref().map(per_time),
            tp: self.tp.as_deref().map(per_time),
            wind: self
                .wind
                .as_ref()
                .map(|(ff, dd)| (per_time(ff), per_time(dd))),
        }
    }

    /// Write in MET's layout (`SPEC[time][y = 1][x][freq][direction]`), so
    /// that [`Self::from_file`] reads it back.
    pub fn write(&self, path: impl AsRef<Path>, title: &str) -> Result<(), NetCDFError> {
        let (nt, np, nf, nd) = (
            self.times.len(),
            self.n_points(),
            self.frequencies.len(),
            self.directions_to.len(),
        );
        let mut file = netcdf::create(path.as_ref())?;
        file.add_dimension("time", nt)?;
        file.add_dimension("y", 1)?;
        file.add_dimension("x", np)?;
        file.add_dimension("freq", nf)?;
        file.add_dimension("direction", nd)?;
        file.add_attribute("title", title)?;
        file.add_attribute(
            "direction_convention",
            "A direction of 0 degrees means a wave propagating towards the North \
             (Oceanographic convention)",
        )?;
        file.add_attribute("history", "written by dgcom io::WaveSpectraFile")?;
        let mut v = file.add_variable::<f64>("time", &["time"])?;
        v.put_attribute("units", "seconds since 1970-01-01 00:00:00")?;
        v.put_attribute("calendar", "gregorian")?;
        v.put_attribute("standard_name", "time")?;
        v.put_values(&self.times, ..)?;
        if let Some(reference) = self.forecast_reference_time {
            let mut v = file.add_variable::<f64>("forecast_reference_time", &[])?;
            v.put_attribute("units", "seconds since 1970-01-01 00:00:00")?;
            v.put_attribute("standard_name", "forecast_reference_time")?;
            v.put_value(reference, ..)?;
        }
        let mut v = file.add_variable::<f64>("freq", &["freq"])?;
        v.put_attribute("units", "1/s")?;
        v.put_values(&self.frequencies, ..)?;
        let mut v = file.add_variable::<f64>("direction", &["direction"])?;
        v.put_attribute("units", "degree")?;
        v.put_values(&self.directions_to, ..)?;
        for (name, values, units) in [
            ("latitude", &self.latitude, "degree_north"),
            ("longitude", &self.longitude, "degree_east"),
        ] {
            let mut v = file.add_variable::<f64>(name, &["y", "x"])?;
            v.put_attribute("units", units)?;
            v.put_attribute("standard_name", name)?;
            v.put_values(values, ..)?;
        }
        if let Some(depth) = &self.depth {
            let mut v = file.add_variable::<f64>("depth", &["y", "x"])?;
            v.put_attribute("units", "m")?;
            v.put_values(depth, ..)?;
        }
        const FILL: f64 = -99999.0;
        let filled = |v: &[f64]| -> Vec<f64> {
            v.iter()
                .map(|x| if x.is_finite() { *x } else { FILL })
                .collect()
        };
        let mut v = file.add_variable::<f64>("SPEC", &["time", "y", "x", "freq", "direction"])?;
        v.put_attribute("_FillValue", FILL)?;
        v.put_attribute("long_name", "2-D spectrum of total sea")?;
        v.put_attribute("units", "m**2 s")?;
        v.put_values(&filled(&self.density), ..)?;
        let mut series: Vec<(&str, &Vec<f64>, &str)> = Vec::new();
        if let Some(hs) = &self.hs {
            series.push(("hs", hs, "m"));
        }
        if let Some(tp) = &self.tp {
            series.push(("tp", tp, "s"));
        }
        if let Some((ff, dd)) = &self.wind {
            series.push(("ff", ff, "m s-1"));
            series.push(("dd", dd, "degree"));
        }
        for (name, values, units) in series {
            let mut v = file.add_variable::<f64>(name, &["time", "y", "x"])?;
            v.put_attribute("_FillValue", FILL)?;
            v.put_attribute("units", units)?;
            v.put_values(&filled(values), ..)?;
        }
        Ok(())
    }

    /// The spectra as a [`PointSpectra`]: the points at `position(lon, lat)`
    /// (the model's mesh coordinates), the times at `model_time(unix)`, the
    /// directions turned from clockwise-from-north to counter-clockwise from
    /// the mesh's x, where geographic east lies at the angle `east` (rad; 0
    /// when x is east: `atan2` of [`super::east_axis`] over the domain).
    pub fn point_spectra(
        &self,
        position: impl Fn(f64, f64) -> [f64; 2],
        model_time: impl Fn(f64) -> f64,
        east: f64,
    ) -> PointSpectra {
        PointSpectra {
            positions: self
                .longitude
                .iter()
                .zip(&self.latitude)
                .map(|(&lon, &lat)| position(lon, lat))
                .collect(),
            times: self.times.iter().map(|&t| model_time(t)).collect(),
            frequencies: self.frequencies.clone(),
            directions: self
                .directions_to
                .iter()
                .map(|d| (90.0 - d).to_radians() + east)
                .collect(),
            density: self.density.clone(),
        }
    }

    /// The wind of the points `points` (all if empty), averaged as vectors at
    /// each time, as a uniform [`WindSeries`] at `model_time(unix)` in the mesh
    /// axes (`east` as for [`Self::point_spectra`]); none without `ff` and
    /// `dd`.
    pub fn wind_series(
        &self,
        points: &[usize],
        model_time: impl Fn(f64) -> f64,
        east: f64,
    ) -> Option<WindSeries> {
        let (ff, dd) = self.wind.as_ref()?;
        let np = self.n_points();
        let all: Vec<usize> = (0..np).collect();
        let points = if points.is_empty() { &all[..] } else { points };
        let winds: Vec<Wind> = (0..self.times.len())
            .map(|t| {
                let (mut u, mut v, mut n) = (0.0, 0.0, 0);
                for &p in points {
                    let (speed, to) = (ff[t * np + p], dd[t * np + p]);
                    if speed.is_finite() && to.is_finite() {
                        let theta = (90.0 - to).to_radians() + east;
                        u += speed * theta.cos();
                        v += speed * theta.sin();
                        n += 1;
                    }
                }
                let (u, v) = (u / n.max(1) as f64, v / n.max(1) as f64);
                Wind {
                    u10: u.hypot(v),
                    direction: v.atan2(u),
                }
            })
            .collect();
        Some(WindSeries::new(
            self.times.iter().map(|&t| model_time(t)).collect(),
            &winds,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Written and read back unchanged, a selection keeps its points' data,
    /// and the directions and wind come out in the model's convention.
    #[test]
    fn point_spectra_survive_a_write_and_a_read() {
        let (nt, np, nf, nd) = (3, 4, 5, 8);
        let n = nt * np * nf * nd;
        let file = WaveSpectraFile {
            longitude: vec![8.0, 8.4, 9.0, 9.2],
            latitude: vec![63.75, 63.75, 64.0, 63.9],
            depth: Some(vec![230.0, 87.0, 110.0, 218.0]),
            times: vec![1.7598600e9, 1.7598636e9, 1.7598672e9],
            forecast_reference_time: Some(1.75986e9),
            frequencies: (0..nf).map(|i| 0.05 * 1.1f64.powi(i as i32)).collect(),
            directions_to: (0..nd).map(|j| 22.5 + 45.0 * j as f64).collect(),
            density: (0..n)
                .map(|k| if k == 7 { f64::NAN } else { k as f64 * 1e-3 })
                .collect(),
            hs: Some((0..nt * np).map(|k| 1.0 + k as f64).collect()),
            tp: Some(vec![10.0; nt * np]),
            wind: Some((vec![8.0; nt * np], vec![90.0; nt * np])),
        };
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("spc.nc");
        file.write(&path, "test").unwrap();
        let back = WaveSpectraFile::from_file(&path).unwrap();
        assert_eq!(back.times, file.times);
        assert_eq!(back.forecast_reference_time, file.forecast_reference_time);
        assert_eq!(back.hs, file.hs);
        assert!(back.density[7].is_nan());
        for (k, (a, b)) in back.density.iter().zip(&file.density).enumerate() {
            assert!(k == 7 || a == b);
        }

        let some = file.select(&[2, 0]);
        assert_eq!(some.longitude, vec![9.0, 8.0]);
        let nc = nf * nd;
        for t in 0..nt {
            assert_eq!(
                &some.density[(t * 2) * nc..(t * 2 + 1) * nc],
                &file.density[(t * np + 2) * nc..(t * np + 3) * nc]
            );
            assert_eq!(
                some.hs.as_ref().unwrap()[t * 2 + 1],
                file.hs.as_ref().unwrap()[t * np]
            );
        }

        // Towards 22.5° from north is 67.5° counter-clockwise from east; the
        // wind blowing towards the east is along +x
        let spectra = file.point_spectra(|lon, lat| [lon, lat], |t| t - 1.75986e9, 0.0);
        assert!((spectra.directions[0] - 67.5f64.to_radians()).abs() < 1e-12);
        assert_eq!(spectra.times[1], 3600.0);
        let wind = file
            .wind_series(&[], |t| t - 1.75986e9, 0.0)
            .unwrap()
            .at(1800.0);
        assert!((wind.u10 - 8.0).abs() < 1e-12 && wind.direction.abs() < 1e-12);
        // A mesh whose x is 10° clockwise of east
        let turned = file.point_spectra(|lon, lat| [lon, lat], |t| t, 10f64.to_radians());
        assert!((turned.directions[0] - 77.5f64.to_radians()).abs() < 1e-12);
    }
}

//! River forcing from ROMS river files (NorKyst-800's NVE rivers).
//!
//! A ROMS river file (`frc_rivers.cdl`) holds, per river, its position on
//! the parent's grid as indices (`river_Xposition`, `river_Eposition`), the
//! kind of point source (`river_direction`), a vertical shape on the
//! parent's levels (`river_Vshape`), and time series of the vertically
//! integrated transport (`river_transport`, m³/s) and of the river water's
//! temperature and salinity (`river_temp`, `river_salt`, per level).
//! NorKyst-800 v3 has 1801 rivers from NVE's runoff data, daily since 2011
//! (`romshindcast/norkyst_v3/forcing/river.nc` on MET's THREDDS), as
//! horizontal sources (`LuvSrc`): each enters through a face of a coastal
//! cell, the sign of the transport giving the direction across it.
//!
//! [`RiverForcing`] holds the rivers with their mouths as longitude and
//! latitude and their discharge, temperature and salinity in time.
//! [`RiverForcing::rivers`] turns them into [`River`]s on a model's plane
//! and clock, for [`crate::source::RiverSources`].
//!
//! # Mouths
//!
//! The mouth of a river on a face is the face's midpoint, half-way between
//! the land cell and the sea cell: the parent's coastline. ROMS indices are
//! those of the ρ-points (0 at the grid's first, boundary, point, as in the
//! NetCDF arrays), and a source on face `(ξ, η)` lies on
//!
//! - `river_direction` 0, the u-face between ρ(ξ−1, η) and ρ(ξ, η);
//! - 1, the v-face between ρ(ξ, η−1) and ρ(ξ, η);
//! - 2 (`LwSrc`), at ρ(ξ, η) itself.
//!
//! A child mesh resolves the coast differently, so the mouth is snapped to
//! the mesh by [`crate::source::RiverSources`].
//!
//! # Temperature and salinity
//!
//! The river water's temperature and salinity are the levels' values
//! weighted by the river's `river_Vshape`: the tracer the river carries
//! into the parent, which the child's own vertical profile then shares out
//! ([`crate::source::RiverProfile`]). A river whose `river_flag` turns a
//! tracer off brings the receiving cell's own value in ROMS; that has no
//! equivalent here and is an error.
//!
//! [`RiverForcing::from_file`] (feature `netcdf`) reads a river file with
//! the mouths as `river_lon`, `river_lat` (as `norkyst_river_subset`
//! writes), or [`RiverForcing::from_file_with_grid`] an original ROMS file
//! with its grid.

use super::projection::{CoordinateProjection, GeoBoundingBox};
use crate::source::{River, RiverError, RiverProfile, RiverSeries};
use crate::time::ModelClock;

/// A river's identity and mouth.
#[derive(Clone, Debug, PartialEq)]
pub struct RiverMouth {
    /// The file's identification number (`river`).
    pub id: i64,
    /// Longitude of the mouth (degrees east).
    pub lon: f64,
    /// Latitude of the mouth (degrees north).
    pub lat: f64,
}

/// The ρ-points of a ROMS grid, or of a window of it, to place point
/// sources with [`roms_source_position`]: `lon` and `lat` row-major
/// `[η][ξ]`, `n_eta × n_xi`, the first point at ROMS indices
/// `(xi0, eta0)`.
#[derive(Clone, Copy, Debug)]
pub struct RomsRhoPoints<'a> {
    /// Longitudes, row-major `[η][ξ]`.
    pub lon: &'a [f64],
    /// Latitudes, row-major `[η][ξ]`.
    pub lat: &'a [f64],
    /// Points along ξ.
    pub n_xi: usize,
    /// Points along η.
    pub n_eta: usize,
    /// ROMS ξ-index of the first point.
    pub xi0: usize,
    /// ROMS η-index of the first point.
    pub eta0: usize,
}

impl RomsRhoPoints<'_> {
    /// `(lon, lat)` of ρ(ξ, η), if it is in the window.
    fn at(&self, xi: i64, eta: i64) -> Option<(f64, f64)> {
        let i = usize::try_from(xi).ok()?.checked_sub(self.xi0)?;
        let j = usize::try_from(eta).ok()?.checked_sub(self.eta0)?;
        (i < self.n_xi && j < self.n_eta).then(|| {
            let k = j * self.n_xi + i;
            (self.lon[k], self.lat[k])
        })
    }
}

/// `(lon, lat)` of a ROMS point source at indices `(xi, eta)` of kind
/// `direction` (`river_direction`; see the module docs), or `None` if its
/// cells are not in `points`. An unknown direction or a non-integer index
/// is an error.
pub fn roms_source_position(
    points: &RomsRhoPoints,
    xi: f64,
    eta: f64,
    direction: f64,
) -> Result<Option<(f64, f64)>, String> {
    let index = |v: f64| {
        (v.fract() == 0.0 && v >= 0.0)
            .then_some(v as i64)
            .ok_or_else(|| format!("position {v} is not a grid index"))
    };
    let (i, j) = (index(xi)?, index(eta)?);
    let neighbour = match index(direction) {
        Ok(0) => Some((i - 1, j)),
        Ok(1) => Some((i, j - 1)),
        Ok(2) => None,
        _ => return Err(format!("unknown river_direction {direction}")),
    };
    let Some(cell) = points.at(i, j) else {
        return Ok(None);
    };
    Ok(match neighbour {
        None => Some(cell),
        Some((i, j)) => points
            .at(i, j)
            .map(|(lon, lat)| (0.5 * (cell.0 + lon), 0.5 * (cell.1 + lat))),
    })
}

/// Rivers with their mouths and their discharge, temperature and salinity
/// at the file's times; see the module docs.
#[derive(Clone, Debug)]
pub struct RiverForcing {
    mouths: Vec<RiverMouth>,
    /// Times, Unix seconds, strictly increasing.
    time: Vec<f64>,
    /// Discharge into the sea (m³/s, ≥ 0), `[time][river]`.
    discharge: Vec<f64>,
    /// River-water temperature (°C), `[time][river]`.
    temperature: Vec<f64>,
    /// River-water salinity, `[time][river]`.
    salinity: Vec<f64>,
}

impl RiverForcing {
    /// Rivers `mouths` with `discharge` (m³/s, ≥ 0), `temperature` (°C)
    /// and `salinity`, each `[time][river]`, at `time` (Unix seconds,
    /// strictly increasing).
    pub fn new(
        mouths: Vec<RiverMouth>,
        time: Vec<f64>,
        discharge: Vec<f64>,
        temperature: Vec<f64>,
        salinity: Vec<f64>,
    ) -> Result<Self, String> {
        if time.is_empty() {
            return Err("RiverForcing: no times".into());
        }
        if time.iter().any(|t| !t.is_finite()) || time.windows(2).any(|w| w[1] <= w[0]) {
            return Err("RiverForcing: times must be finite and strictly increasing".into());
        }
        let n = time.len() * mouths.len();
        for (name, values) in [
            ("discharge", &discharge),
            ("temperature", &temperature),
            ("salinity", &salinity),
        ] {
            if values.len() != n {
                return Err(format!(
                    "RiverForcing: {} {name} values for {} times × {} rivers",
                    values.len(),
                    time.len(),
                    mouths.len()
                ));
            }
            if let Some(k) = values.iter().position(|v| !v.is_finite()) {
                let mouth = &mouths[k % mouths.len()];
                return Err(format!(
                    "RiverForcing: river {} has a missing {name} at time index {}",
                    mouth.id,
                    k / mouths.len()
                ));
            }
        }
        if let Some(k) = discharge.iter().position(|&q| q < 0.0) {
            return Err(format!(
                "RiverForcing: river {} has a negative discharge {} m³/s",
                mouths[k % mouths.len()].id,
                discharge[k]
            ));
        }
        Ok(Self {
            mouths,
            time,
            discharge,
            temperature,
            salinity,
        })
    }

    /// Number of rivers.
    pub fn len(&self) -> usize {
        self.mouths.len()
    }

    /// Whether there are no rivers.
    pub fn is_empty(&self) -> bool {
        self.mouths.is_empty()
    }

    /// The rivers' mouths.
    pub fn mouths(&self) -> &[RiverMouth] {
        &self.mouths
    }

    /// The times (Unix seconds).
    pub fn time(&self) -> &[f64] {
        &self.time
    }

    /// First and last time (Unix seconds).
    pub fn time_range(&self) -> (f64, f64) {
        (self.time[0], self.time[self.time.len() - 1])
    }

    /// Discharge of river `i` at time index `n` (m³/s).
    pub fn discharge(&self, n: usize, i: usize) -> f64 {
        self.discharge[n * self.len() + i]
    }

    /// Temperature of river `i`'s water at time index `n` (°C).
    pub fn temperature(&self, n: usize, i: usize) -> f64 {
        self.temperature[n * self.len() + i]
    }

    /// Salinity of river `i`'s water at time index `n`.
    pub fn salinity(&self, n: usize, i: usize) -> f64 {
        self.salinity[n * self.len() + i]
    }

    /// Mean discharge of river `i` over the file (m³/s).
    pub fn mean_discharge(&self, i: usize) -> f64 {
        (0..self.time.len())
            .map(|n| self.discharge(n, i))
            .sum::<f64>()
            / self.time.len() as f64
    }

    /// The rivers `keep` selects, in order.
    pub fn select(&self, mut keep: impl FnMut(&RiverMouth) -> bool) -> Self {
        let kept: Vec<usize> = (0..self.len()).filter(|&i| keep(&self.mouths[i])).collect();
        let pick = |values: &[f64]| -> Vec<f64> {
            (0..self.time.len())
                .flat_map(|n| kept.iter().map(move |&i| values[n * self.len() + i]))
                .collect()
        };
        Self {
            mouths: kept.iter().map(|&i| self.mouths[i].clone()).collect(),
            time: self.time.clone(),
            discharge: pick(&self.discharge),
            temperature: pick(&self.temperature),
            salinity: pick(&self.salinity),
        }
    }

    /// The rivers with their mouth in `bbox`.
    pub fn within(&self, bbox: &GeoBoundingBox) -> Self {
        self.select(|m| bbox.contains(m.lat, m.lon))
    }

    /// The rivers on a model's plane (`projection`) and clock: positions in
    /// metres, the series in model seconds (constant beyond the file's
    /// times), entering over `profile`. Named by their ids (errors say `river <id>`).
    pub fn rivers(
        &self,
        projection: &(impl CoordinateProjection + ?Sized),
        clock: &ModelClock,
        profile: &RiverProfile,
    ) -> Result<Vec<River>, RiverError> {
        let model_time: Vec<f64> = self.time.iter().map(|&t| clock.model_time(t)).collect();
        let series = |values: &[f64], i: usize| {
            RiverSeries::new(
                model_time.clone(),
                (0..self.time.len())
                    .map(|n| values[n * self.len() + i])
                    .collect(),
            )
        };
        self.mouths
            .iter()
            .enumerate()
            .map(|(i, mouth)| {
                let (x, y) = projection.geo_to_xy(mouth.lat, mouth.lon);
                Ok(River::new(mouth.id.to_string(), [x, y], 0.0)
                    .with_discharge(series(&self.discharge, i)?)
                    .with_temperature(series(&self.temperature, i)?)
                    .with_salinity(series(&self.salinity, i)?)
                    .with_profile(profile.clone()))
            })
            .collect()
    }
}

#[cfg(feature = "netcdf")]
mod netcdf_file {
    use std::path::Path;

    use netcdf::Extent;

    use super::{RiverForcing, RiverMouth, RomsRhoPoints, roms_source_position};
    use crate::io::GeoBoundingBox;
    use crate::io::netcdf_io::NetCDFError;
    use crate::io::netcdf_read::{read_time_named, read_values};

    fn slice(start: usize, count: usize) -> Extent {
        Extent::SliceCount {
            start,
            count,
            stride: 1,
        }
    }

    fn invalid(message: String) -> NetCDFError {
        NetCDFError::InvalidData(message)
    }

    /// A variable the file must have.
    fn variable<'f>(
        file: &'f netcdf::File,
        name: &str,
    ) -> Result<netcdf::Variable<'f>, NetCDFError> {
        file.variable(name)
            .ok_or_else(|| NetCDFError::MissingVariable(name.into()))
    }

    /// Values of the `[river]` variable `name`, all rivers.
    fn per_river(file: &netcdf::File, name: &str) -> Result<Vec<f64>, NetCDFError> {
        read_values(&variable(file, name)?, vec![])
    }

    /// The rivers' identification numbers (`river`), else 1, 2, ….
    fn ids(file: &netcdf::File, n: usize) -> Result<Vec<i64>, NetCDFError> {
        match file.variable("river") {
            Some(var) => Ok(read_values(&var, vec![])?
                .into_iter()
                .enumerate()
                .map(|(i, id)| {
                    if id.is_finite() {
                        id.round() as i64
                    } else {
                        i as i64 + 1
                    }
                })
                .collect()),
            None => Ok((1..=n as i64).collect()),
        }
    }

    impl RiverForcing {
        /// All rivers of a river file whose mouths are given as `river_lon`,
        /// `river_lat` (as [`Self::write`] and `norkyst_river_subset` write
        /// them); see the module docs.
        pub fn from_file(path: impl AsRef<Path>) -> Result<Self, NetCDFError> {
            let file = netcdf::open(path.as_ref())?;
            let missing = || {
                NetCDFError::MissingVariable(
                    "river_lon, river_lat (an original ROMS river file is read with its \
                     grid, RiverForcing::from_roms_file)"
                        .into(),
                )
            };
            let lon = file
                .variable("river_lon")
                .ok_or_else(missing)
                .and_then(|v| read_values(&v, vec![]))?;
            let lat = file
                .variable("river_lat")
                .ok_or_else(missing)
                .and_then(|v| read_values(&v, vec![]))?;
            let ids = ids(&file, lon.len())?;
            let rivers: Vec<(usize, RiverMouth)> = (0..lon.len())
                .map(|i| {
                    (
                        i,
                        RiverMouth {
                            id: ids[i],
                            lon: lon[i],
                            lat: lat[i],
                        },
                    )
                })
                .collect();
            read_series(&file, rivers, None)
        }

        /// The rivers of a ROMS river file on the grid whose ρ-points are
        /// `grid` (or the window of it they are in), with their mouths in
        /// `bbox` (all if `None`), at the times covering `window` (Unix
        /// seconds; all if `None`). Only those rivers' and times' values are
        /// read, so a whole-coast file (NorKyst's has 3.3 GB of river
        /// temperatures) can be read, over OPeNDAP too.
        pub fn from_roms_file(
            path: impl AsRef<Path>,
            grid: &RomsRhoPoints,
            bbox: Option<&GeoBoundingBox>,
            window: Option<(f64, f64)>,
        ) -> Result<Self, NetCDFError> {
            let file = netcdf::open(path.as_ref())?;
            let xi = per_river(&file, "river_Xposition")?;
            let eta = per_river(&file, "river_Eposition")?;
            let direction = match file.variable("river_direction") {
                Some(var) => read_values(&var, vec![])?,
                None => vec![2.0; xi.len()],
            };
            let ids = ids(&file, xi.len())?;
            let mut rivers = Vec::new();
            for i in 0..xi.len() {
                let position = roms_source_position(grid, xi[i], eta[i], direction[i])
                    .map_err(|e| invalid(format!("river {}: {e}", ids[i])))?;
                if let Some((lon, lat)) = position
                    && bbox.is_none_or(|b| b.contains(lat, lon))
                {
                    rivers.push((
                        i,
                        RiverMouth {
                            id: ids[i],
                            lon,
                            lat,
                        },
                    ));
                }
            }
            read_series(&file, rivers, window)
        }

        /// Write the rivers as a river file that [`Self::from_file`] reads:
        /// dimensions `river`, `river_time`; `river` (the ids), `river_lon`,
        /// `river_lat`, `river_time` (seconds since 1970), and
        /// `river_transport`, `river_temp`, `river_salt` as `[time][river]`.
        pub fn write(&self, path: impl AsRef<Path>) -> Result<(), NetCDFError> {
            let mut file = netcdf::create(path.as_ref())?;
            file.add_attribute("title", "River forcing (dg-rs RiverForcing)")?;
            file.add_dimension("river", self.len())?;
            file.add_dimension("river_time", self.time.len())?;
            let ids: Vec<f64> = self.mouths.iter().map(|m| m.id as f64).collect();
            let lon: Vec<f64> = self.mouths.iter().map(|m| m.lon).collect();
            let lat: Vec<f64> = self.mouths.iter().map(|m| m.lat).collect();
            for (name, long_name, units, values) in [
                (
                    "river",
                    "river runoff identification number",
                    "nondimensional",
                    &ids,
                ),
                (
                    "river_lon",
                    "longitude of the river mouth",
                    "degrees_east",
                    &lon,
                ),
                (
                    "river_lat",
                    "latitude of the river mouth",
                    "degrees_north",
                    &lat,
                ),
            ] {
                let mut var = file.add_variable::<f64>(name, &["river"])?;
                var.put_attribute("long_name", long_name)?;
                var.put_attribute("units", units)?;
                var.put_values(values, ..)?;
            }
            let mut time = file.add_variable::<f64>("river_time", &["river_time"])?;
            time.put_attribute("long_name", "river runoff time")?;
            time.put_attribute("units", "seconds since 1970-01-01 00:00:00")?;
            time.put_values(&self.time, ..)?;
            for (name, long_name, units, values) in [
                (
                    "river_transport",
                    "river runoff vertically integrated mass transport",
                    "meter3 second-1",
                    &self.discharge,
                ),
                (
                    "river_temp",
                    "river runoff potential temperature",
                    "Celsius",
                    &self.temperature,
                ),
                ("river_salt", "river runoff salinity", "PSU", &self.salinity),
            ] {
                let mut var = file.add_variable::<f64>(name, &["river_time", "river"])?;
                var.put_attribute("long_name", long_name)?;
                var.put_attribute("units", units)?;
                var.put_values(values, ..)?;
            }
            Ok(())
        }
    }

    /// The series of `rivers` (file index, mouth; indices ascending) at the
    /// times covering `window`.
    fn read_series(
        file: &netcdf::File,
        rivers: Vec<(usize, RiverMouth)>,
        window: Option<(f64, f64)>,
    ) -> Result<RiverForcing, NetCDFError> {
        let (all_times, _) = read_time_named(file, &["river_time"])?;
        let (n0, n1) = match window {
            None => (0, all_times.len() - 1),
            Some((t0, t1)) => {
                let (first, last) = (all_times[0], all_times[all_times.len() - 1]);
                if t1 < first || t0 > last || t1 < t0 {
                    return Err(invalid(format!(
                        "the rivers' times ({first:.0}–{last:.0} s, Unix) do not cover \
                         {t0:.0}–{t1:.0} s"
                    )));
                }
                // The last time at or before t0 and the first at or after t1
                let n0 = all_times.partition_point(|&t| t <= t0).saturating_sub(1);
                let n1 = all_times
                    .partition_point(|&t| t < t1)
                    .min(all_times.len() - 1);
                (n0, n1)
            }
        };
        let time = all_times[n0..=n1].to_vec();
        let nt = time.len();
        let Some(r0) = rivers.first().map(|r| r.0) else {
            return RiverForcing::new(Vec::new(), time, Vec::new(), Vec::new(), Vec::new())
                .map_err(invalid);
        };
        let nr = rivers[rivers.len() - 1].0 + 1 - r0;
        let nk = rivers.len();
        let ids: Vec<i64> = rivers.iter().map(|r| r.1.id).collect();

        // The rivers' columns of a [time][river] block of nr rivers from r0
        let pick = |block: &[f64], width: usize| -> Vec<f64> {
            (0..nt)
                .flat_map(|n| rivers.iter().map(move |r| block[n * width + r.0 - r0]))
                .collect()
        };

        // Transport: its sign is the direction across the face
        let transport = read_values(
            &variable(file, "river_transport")?,
            vec![slice(n0, nt), slice(r0, nr)],
        )?;
        let transport = pick(&transport, nr);
        for (k, id) in ids.iter().enumerate() {
            let q = (0..nt).map(|n| transport[n * nk + k]);
            let (lo, hi) = q.fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), q| {
                (a.min(q), b.max(q))
            });
            if lo < 0.0 && hi > 0.0 {
                return Err(invalid(format!(
                    "river {id}: the transport changes sign ({lo} to {hi} m³/s)"
                )));
            }
        }
        let discharge: Vec<f64> = transport.iter().map(|q| q.abs()).collect();

        // Which tracers each river brings
        let flags = match file.variable("river_flag") {
            Some(var) => read_values(&var, vec![slice(r0, nr)])?,
            None => vec![3.0; nr],
        };
        for (r, id) in rivers.iter().map(|r| r.0).zip(&ids) {
            if flags[r - r0] != 3.0 {
                return Err(invalid(format!(
                    "river {id}: river_flag {} turns a tracer off (the river would bring the \
                     sea's own value, which RiverForcing cannot represent)",
                    flags[r - r0]
                )));
            }
        }

        // Temperature and salinity, per level weighted by the river's shape
        let tracer = |name: &str| -> Result<Vec<f64>, NetCDFError> {
            let var = variable(file, name)?;
            match var.dimensions().len() {
                2 => Ok(pick(
                    &read_values(&var, vec![slice(n0, nt), slice(r0, nr)])?,
                    nr,
                )),
                3 => {
                    let n_levels = var.dimensions()[1].len();
                    let shape = read_values(
                        &variable(file, "river_Vshape")?,
                        vec![slice(0, n_levels), slice(r0, nr)],
                    )?;
                    let values =
                        read_values(&var, vec![slice(n0, nt), slice(0, n_levels), slice(r0, nr)])?;
                    let mut mean = vec![0.0; nt * nr];
                    for r in 0..nr {
                        let total: f64 = (0..n_levels).map(|l| shape[l * nr + r]).sum();
                        for n in 0..nt {
                            mean[n * nr + r] = (0..n_levels)
                                .map(|l| shape[l * nr + r] * values[(n * n_levels + l) * nr + r])
                                .sum::<f64>()
                                / total;
                        }
                    }
                    Ok(pick(&mean, nr))
                }
                d => Err(invalid(format!("{name} has {d} dimensions"))),
            }
        };
        let temperature = tracer("river_temp")?;
        let salinity = tracer("river_salt")?;
        RiverForcing::new(
            rivers.into_iter().map(|r| r.1).collect(),
            time,
            discharge,
            temperature,
            salinity,
        )
        .map_err(invalid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::LocalProjection;

    /// A 3 × 4 window of ρ-points at ROMS indices ξ 10–13, η 20–22, with
    /// `lon = ξ/100`, `lat = 60 + η/100`.
    fn window() -> (Vec<f64>, Vec<f64>) {
        let (mut lon, mut lat) = (Vec::new(), Vec::new());
        for eta in 20..23 {
            for xi in 10..14 {
                lon.push(xi as f64 / 100.0);
                lat.push(60.0 + eta as f64 / 100.0);
            }
        }
        (lon, lat)
    }

    #[test]
    fn sources_sit_on_the_faces_roms_puts_them_on() {
        let (lon, lat) = window();
        let points = RomsRhoPoints {
            lon: &lon,
            lat: &lat,
            n_xi: 4,
            n_eta: 3,
            xi0: 10,
            eta0: 20,
        };
        let at = |xi, eta, dir| roms_source_position(&points, xi, eta, dir).unwrap();
        let close = |a: Option<(f64, f64)>, b: (f64, f64)| {
            let a = a.expect("inside the window");
            assert!(
                (a.0 - b.0).abs() < 1e-12 && (a.1 - b.1).abs() < 1e-12,
                "{a:?} ≠ {b:?}"
            );
        };
        // A u-face between ρ(11, 21) and ρ(12, 21)
        close(at(12.0, 21.0, 0.0), (0.115, 60.21));
        // A v-face between ρ(12, 20) and ρ(12, 21)
        close(at(12.0, 21.0, 1.0), (0.12, 60.205));
        // A vertical source at ρ(12, 21)
        close(at(12.0, 21.0, 2.0), (0.12, 60.21));
        // A face whose other cell is outside the window, and a cell outside
        assert_eq!(at(10.0, 21.0, 0.0), None);
        assert_eq!(at(12.0, 20.0, 1.0), None);
        assert_eq!(at(9.0, 21.0, 2.0), None);
        assert!(roms_source_position(&points, 12.5, 21.0, 0.0).is_err());
        assert!(roms_source_position(&points, 12.0, 21.0, 3.0).is_err());
    }

    fn forcing() -> RiverForcing {
        let mouths = vec![
            RiverMouth {
                id: 7,
                lon: 8.5,
                lat: 63.7,
            },
            RiverMouth {
                id: 9,
                lon: 9.5,
                lat: 63.9,
            },
        ];
        // Two days, [time][river]
        RiverForcing::new(
            mouths,
            vec![1.7e9, 1.7e9 + 86_400.0],
            vec![10.0, 100.0, 20.0, 300.0],
            vec![4.0, 5.0, 6.0, 7.0],
            vec![0.0; 4],
        )
        .unwrap()
    }

    #[test]
    fn rivers_carry_their_series_on_the_model_clock() {
        let forcing = forcing();
        assert_eq!(forcing.mean_discharge(1), 200.0);
        // The clock starts 6 h into the file
        let clock = ModelClock::new(1.7e9 + 6.0 * 3600.0);
        let projection = LocalProjection::new(63.7, 8.5);
        let rivers = forcing
            .rivers(&projection, &clock, &RiverProfile::TopFraction(0.1))
            .unwrap();
        assert_eq!(rivers.len(), 2);
        assert_eq!(rivers[0].name, "7");
        assert!(rivers[0].position[0].abs() < 1e-9 && rivers[0].position[1].abs() < 1e-9);
        assert!(rivers[1].position[0] > 40e3 && rivers[1].position[1] > 20e3);
        // A quarter of the way from day 1 to day 2 at t = 0
        assert!((rivers[1].discharge.at(0.0) - 150.0).abs() < 1e-9);
        assert!((rivers[0].temperature.at(18.0 * 3600.0) - 6.0).abs() < 1e-9);
        assert!((rivers[0].discharge.at(-1e6) - 10.0).abs() < 1e-12);
        assert_eq!(rivers[0].profile, RiverProfile::TopFraction(0.1));
    }

    #[test]
    fn selection_keeps_each_rivers_own_series() {
        let forcing = forcing();
        let east = forcing.within(&GeoBoundingBox::new(9.0, 63.0, 10.0, 64.0));
        assert_eq!(east.len(), 1);
        assert_eq!(east.mouths()[0].id, 9);
        assert_eq!(east.discharge(1, 0), 300.0);
        assert_eq!(east.temperature(0, 0), 5.0);
    }

    /// A ROMS river file on [`window`]'s grid, 4 days from day 19000 since
    /// 1970, two levels: river 11 on the u-face (12, 21), 10…40 m³/s along
    /// +ξ, its water at 1 °C at the bed and 9 °C on top, all of it on top;
    /// river 12 on the v-face (12, 21), 5 m³/s along −η, 2 and 4 °C shared
    /// equally; river 13 off the grid. `transport_12` and `flag_12`
    /// override river 12's.
    #[cfg(feature = "netcdf")]
    fn write_roms_file(path: &std::path::Path, transport_12: [f64; 4], flag_12: f64) {
        let mut file = netcdf::create(path).unwrap();
        file.add_dimension("river", 3).unwrap();
        file.add_dimension("s_rho", 2).unwrap();
        file.add_dimension("river_time", 4).unwrap();
        let mut put = |name: &str, dims: &[&str], values: &[f64]| {
            let mut var = file.add_variable::<f64>(name, dims).unwrap();
            if name == "river_time" {
                var.put_attribute("units", "days since 1970-01-01 00:00:00")
                    .unwrap();
            }
            var.put_values(values, ..).unwrap();
        };
        put("river", &["river"], &[11.0, 12.0, 13.0]);
        put("river_Xposition", &["river"], &[12.0, 12.0, 30.0]);
        put("river_Eposition", &["river"], &[21.0, 21.0, 21.0]);
        put("river_direction", &["river"], &[0.0, 1.0, 0.0]);
        put("river_flag", &["river"], &[3.0, flag_12, 3.0]);
        put(
            "river_time",
            &["river_time"],
            &[19000.0, 19001.0, 19002.0, 19003.0],
        );
        // [s_rho][river], bed first
        put(
            "river_Vshape",
            &["s_rho", "river"],
            &[0.0, 0.5, 0.5, 1.0, 0.5, 0.5],
        );
        let transport: Vec<f64> = (0..4)
            .flat_map(|n| [10.0 * (n + 1) as f64, transport_12[n], 1.0])
            .collect();
        put("river_transport", &["river_time", "river"], &transport);
        let temp: Vec<f64> = (0..4)
            .flat_map(|_| [1.0, 2.0, 0.0, 9.0, 4.0, 0.0])
            .collect();
        put("river_temp", &["river_time", "s_rho", "river"], &temp);
        put("river_salt", &["river_time", "s_rho", "river"], &[0.0; 24]);
    }

    #[cfg(feature = "netcdf")]
    #[test]
    fn a_roms_river_file_is_read_on_its_grid_and_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rivers.nc");
        write_roms_file(&path, [-5.0; 4], 3.0);
        let (lon, lat) = window();
        let grid = RomsRhoPoints {
            lon: &lon,
            lat: &lat,
            n_xi: 4,
            n_eta: 3,
            xi0: 10,
            eta0: 20,
        };
        let day = 86_400.0;
        let window = (19001.5 * day, 19002.2 * day);
        let forcing = RiverForcing::from_roms_file(&path, &grid, None, Some(window)).unwrap();
        // River 13 is off the grid; the times are those around the window
        assert_eq!(
            forcing.mouths().iter().map(|m| m.id).collect::<Vec<_>>(),
            [11, 12]
        );
        assert_eq!(
            forcing.time(),
            [19001.0 * day, 19002.0 * day, 19003.0 * day]
        );
        let m = &forcing.mouths()[0];
        assert!((m.lon - 0.115).abs() < 1e-12 && (m.lat - 60.21).abs() < 1e-12);
        let m = &forcing.mouths()[1];
        assert!((m.lon - 0.12).abs() < 1e-12 && (m.lat - 60.205).abs() < 1e-12);
        // The discharge into the sea whatever the face's direction
        assert_eq!(forcing.discharge(0, 0), 20.0);
        assert_eq!(forcing.discharge(2, 1), 5.0);
        // The levels' temperatures weighted by the river's shape
        assert_eq!(forcing.temperature(0, 0), 9.0);
        assert_eq!(forcing.temperature(1, 1), 3.0);

        // A box around river 12 only
        let bbox = GeoBoundingBox::new(0.118, 60.2, 0.13, 60.21);
        let one = RiverForcing::from_roms_file(&path, &grid, Some(&bbox), None).unwrap();
        assert_eq!(one.len(), 1);
        assert_eq!(one.mouths()[0].id, 12);
        assert_eq!(one.time().len(), 4);

        // Written and read back
        let copy = dir.path().join("copy.nc");
        forcing.write(&copy).unwrap();
        let back = RiverForcing::from_file(&copy).unwrap();
        assert_eq!(back.mouths(), forcing.mouths());
        assert_eq!(back.time(), forcing.time());
        for n in 0..3 {
            for i in 0..2 {
                assert_eq!(back.discharge(n, i), forcing.discharge(n, i));
                assert_eq!(back.temperature(n, i), forcing.temperature(n, i));
                assert_eq!(back.salinity(n, i), forcing.salinity(n, i));
            }
        }
        // The original file has no river_lon/river_lat
        assert!(RiverForcing::from_file(&path).is_err());
    }

    #[cfg(feature = "netcdf")]
    #[test]
    fn roms_rivers_that_cannot_be_represented_are_refused() {
        let dir = tempfile::tempdir().unwrap();
        let (lon, lat) = window();
        let grid = RomsRhoPoints {
            lon: &lon,
            lat: &lat,
            n_xi: 4,
            n_eta: 3,
            xi0: 10,
            eta0: 20,
        };
        let read = |transport, flag| {
            let path = dir.path().join(format!("rivers_{flag}.nc"));
            write_roms_file(&path, transport, flag);
            RiverForcing::from_roms_file(&path, &grid, None, None)
                .unwrap_err()
                .to_string()
        };
        // Flowing out of the sea and in again
        assert!(read([-5.0, 5.0, 5.0, 5.0], 3.0).contains("changes sign"));
        // Salinity off: the river would bring the sea's own
        assert!(read([5.0; 4], 1.0).contains("river_flag"));
    }

    #[test]
    fn bad_series_are_refused() {
        let mouth = RiverMouth {
            id: 1,
            lon: 8.0,
            lat: 63.0,
        };
        let make = |q: f64, t: f64| {
            RiverForcing::new(vec![mouth.clone()], vec![0.0], vec![q], vec![t], vec![0.0])
        };
        assert!(make(1.0, 5.0).is_ok());
        assert!(make(-1.0, 5.0).unwrap_err().contains("negative"));
        assert!(
            make(1.0, f64::NAN)
                .unwrap_err()
                .contains("missing temperature")
        );
        assert!(
            RiverForcing::new(
                vec![mouth],
                vec![0.0, 0.0],
                vec![1.0; 2],
                vec![1.0; 2],
                vec![0.0; 2]
            )
            .is_err()
        );
    }
}

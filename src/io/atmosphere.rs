//! Gridded atmospheric forcing: 10 m wind, sea-level pressure and the
//! fields of the surface heat budget.
//!
//! [`AtmosphereReader`] holds the fields of a weather model (MET Nordic
//! analysis, MEPS, AROME-Arctic, ERA5, or the wind NorKyst-800 was forced
//! with) on the model's own grid: the 10 m wind as **east/north**
//! components, and scalar [`AtmosphereField`]s: the air pressure at sea
//! level and, for the bulk heat and freshwater fluxes
//! ([`crate::physics::air_sea`]), the 2 m air temperature and relative
//! humidity, the downwelling shortwave and longwave radiation and the
//! precipitation. Times are Unix seconds.
//!
//! [`AtmosphereReader::from_file`] / [`AtmosphereReader::from_files`]
//! (feature `netcdf`) read CF NetCDF: grid-relative winds (`x_wind_10m`,
//! `y_wind_10m` on a Lambert grid) are rotated to east/north with the local
//! angle of the grid, wind speed and direction (`wind_speed_10m`,
//! `wind_direction_10m`, direction the wind blows *from*) become components,
//! pressure in hPa becomes Pa, and singleton dimensions (MEPS's `height*`)
//! are dropped. Consecutive forecast files are joined, the later file
//! winning where they overlap.
//!
//! `source::GriddedAtmosphere2D` turns the fields into wind stress and the
//! pressure-gradient force on a mesh, and `source::GriddedSurfaceFluxes`
//! into the heat and freshwater fluxes of a 3D model's columns.

use super::field_series::{FieldSeries, TimeInterpolation, TimeStencil};
use super::geo_grid::{GeoGrid, Stencil};

/// Standard sea-level pressure (Pa), the storage offset of pressure fields.
pub const P_REFERENCE: f64 = 101_325.0;

/// A scalar field of an [`AtmosphereReader`], in SI units (temperature in
/// °C).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AtmosphereField {
    /// Air pressure at sea level (Pa)
    Pressure,
    /// 2 m air temperature (°C)
    AirTemperature,
    /// 2 m relative humidity (fraction, 0–1)
    RelativeHumidity,
    /// Downwelling shortwave radiation at the surface (W/m²)
    Shortwave,
    /// Downwelling longwave radiation at the surface (W/m²)
    Longwave,
    /// Precipitation (kg m⁻² s⁻¹)
    Precipitation,
}

impl AtmosphereField {
    /// Every field, in storage order.
    pub const ALL: [Self; 6] = [
        Self::Pressure,
        Self::AirTemperature,
        Self::RelativeHumidity,
        Self::Shortwave,
        Self::Longwave,
        Self::Precipitation,
    ];

    /// The fields the bulk heat flux needs ([`crate::physics::air_sea`]);
    /// precipitation is optional (none: no rain).
    pub const HEAT_BUDGET: [Self; 4] = [
        Self::AirTemperature,
        Self::RelativeHumidity,
        Self::Shortwave,
        Self::Longwave,
    ];

    /// Short name, for summaries and messages.
    pub fn name(self) -> &'static str {
        match self {
            Self::Pressure => "pressure",
            Self::AirTemperature => "air temperature",
            Self::RelativeHumidity => "humidity",
            Self::Shortwave => "shortwave",
            Self::Longwave => "longwave",
            Self::Precipitation => "precipitation",
        }
    }

    fn index(self) -> usize {
        self as usize
    }
}

/// Atmospheric state at one point and time.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AtmosphereState {
    /// 10 m wind (east, north) (m/s)
    pub wind: Option<(f64, f64)>,
    /// The scalar fields, indexed by [`AtmosphereField`] (`None` where the
    /// reader has no such field)
    pub fields: [Option<f64>; 6],
}

impl AtmosphereState {
    /// The value of `field`, if the reader has it.
    pub fn get(&self, field: AtmosphereField) -> Option<f64> {
        self.fields[field.index()]
    }
}

/// Weather-model fields on the model grid; see the module docs.
#[derive(Clone, Debug)]
pub struct AtmosphereReader {
    /// Model grid
    grid: GeoGrid,
    /// Snapshot times, Unix seconds, strictly increasing
    time: Vec<f64>,
    /// 10 m wind, eastward (m/s)
    u10: Option<FieldSeries>,
    /// 10 m wind, northward (m/s)
    v10: Option<FieldSeries>,
    /// The scalar fields, indexed by [`AtmosphereField`]
    fields: [Option<FieldSeries>; 6],
    /// Where the fields came from (variable names, rotation)
    source: String,
}

impl AtmosphereReader {
    /// Fields on `grid` at snapshot `time` (Unix seconds, strictly
    /// increasing); add fields with the `with_*` methods.
    pub fn new(grid: GeoGrid, time: Vec<f64>) -> Result<Self, String> {
        if time.is_empty() {
            return Err("AtmosphereReader: no snapshots".into());
        }
        if time.iter().any(|t| !t.is_finite()) || time.windows(2).any(|w| w[1] <= w[0]) {
            return Err("AtmosphereReader: times must be finite and strictly increasing".into());
        }
        Ok(Self {
            grid,
            time,
            u10: None,
            v10: None,
            fields: Default::default(),
            source: String::new(),
        })
    }

    fn check(&self, name: &str, series: &FieldSeries) {
        assert!(
            series.n_points() == self.grid.len() && series.n_times() == self.time.len(),
            "AtmosphereReader: {name} has {} × {} values, expected {} snapshots × {} points",
            series.n_times(),
            series.n_points(),
            self.time.len(),
            self.grid.len()
        );
    }

    /// Set the 10 m wind (east, north) (m/s).
    pub fn with_wind(mut self, u10: FieldSeries, v10: FieldSeries) -> Self {
        self.check("u10", &u10);
        self.check("v10", &v10);
        self.u10 = Some(u10);
        self.v10 = Some(v10);
        self
    }

    /// Set a scalar field (SI units, temperature in °C).
    pub fn with_field(mut self, field: AtmosphereField, series: FieldSeries) -> Self {
        self.check(field.name(), &series);
        self.fields[field.index()] = Some(series);
        self
    }

    /// Describe where the fields came from.
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = source.into();
        self
    }

    /// Model grid.
    pub fn grid(&self) -> &GeoGrid {
        &self.grid
    }

    /// Snapshot times, Unix seconds, strictly increasing.
    pub fn times(&self) -> &[f64] {
        &self.time
    }

    /// 10 m wind (east, north) (m/s).
    pub fn wind(&self) -> Option<(&FieldSeries, &FieldSeries)> {
        self.u10.as_ref().zip(self.v10.as_ref())
    }

    /// A scalar field, if present.
    pub fn field(&self, field: AtmosphereField) -> Option<&FieldSeries> {
        self.fields[field.index()].as_ref()
    }

    /// Where the fields came from (variable names, rotation).
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Whether the fields include wind.
    pub fn has_wind(&self) -> bool {
        self.u10.is_some() && self.v10.is_some()
    }

    /// Whether the reader has `field`.
    pub fn has(&self, field: AtmosphereField) -> bool {
        self.fields[field.index()].is_some()
    }

    /// Whether the reader has every field of the bulk heat flux
    /// ([`AtmosphereField::HEAT_BUDGET`]).
    pub fn has_heat_budget(&self) -> bool {
        AtmosphereField::HEAT_BUDGET.iter().all(|&f| self.has(f))
    }

    /// First and last snapshot time (Unix seconds).
    pub fn time_range(&self) -> (f64, f64) {
        (self.time[0], self.time[self.time.len() - 1])
    }

    /// Whether `time` (Unix seconds) lies within the snapshots.
    pub fn covers_time(&self, time: f64) -> bool {
        TimeStencil::new(&self.time, time, TimeInterpolation::Linear).is_some()
    }

    /// State at `(lon, lat)` and `time` (Unix seconds), bilinear in space and
    /// linear in time; `None` outside the grid or the file's times.
    pub fn get(&self, lon: f64, lat: f64, time: f64) -> Option<AtmosphereState> {
        let t = TimeStencil::new(&self.time, time, TimeInterpolation::Linear)?;
        let p = self.grid.locate(lon, lat)?;
        let s = Stencil::bilinear(&self.grid, &p, |_| true)?;
        let space = s.idx.iter().map(|&k| k as usize).zip(s.w);
        let field = |f: &Option<FieldSeries>| {
            f.as_ref()
                .map(|f| f.interpolate(&t, space.clone()))
                .filter(|v| v.is_finite())
        };
        Some(AtmosphereState {
            wind: field(&self.u10).zip(field(&self.v10)),
            fields: AtmosphereField::ALL.map(|f| field(&self.fields[f.index()])),
        })
    }

    /// Join consecutive files on the same grid. Where two overlap, the later
    /// one wins (the newer forecast): each file keeps only its snapshots
    /// before the next file's first.
    pub fn concat(parts: Vec<Self>) -> Result<Self, String> {
        let mut parts = parts.into_iter();
        let first = parts.next().ok_or("AtmosphereReader::concat: no parts")?;
        let rest: Vec<Self> = parts.collect();
        if rest.is_empty() {
            return Ok(first);
        }
        let all: Vec<&Self> = std::iter::once(&first).chain(&rest).collect();
        if all.windows(2).any(|w| w[1].time[0] <= w[0].time[0]) {
            return Err("AtmosphereReader::concat: files out of order".into());
        }
        for p in &all[1..] {
            if p.grid.dims() != first.grid.dims()
                || p.grid.lon() != first.grid.lon()
                || p.grid.lat() != first.grid.lat()
            {
                return Err("AtmosphereReader::concat: the files have different grids".into());
            }
            if p.has_wind() != first.has_wind()
                || AtmosphereField::ALL
                    .iter()
                    .any(|&f| p.has(f) != first.has(f))
            {
                return Err("AtmosphereReader::concat: the files have different fields".into());
            }
        }
        // Snapshots kept from each part
        let mut keep: Vec<(usize, usize)> = Vec::new();
        for (n, p) in all.iter().enumerate() {
            let end = all.get(n + 1).map_or(p.time.len(), |next| {
                p.time.partition_point(|&t| t < next.time[0])
            });
            keep.push((n, end));
        }
        let time: Vec<f64> = keep
            .iter()
            .flat_map(|&(n, end)| all[n].time[..end].iter().copied())
            .collect();
        let n_points = first.grid.len();
        let join = |get: &dyn Fn(&Self) -> Option<&FieldSeries>| -> Option<FieldSeries> {
            let offset = get(&first)?.offset();
            let values: Vec<f64> = keep
                .iter()
                .flat_map(|&(n, end)| {
                    let f = get(all[n]).expect("checked");
                    (0..end).flat_map(move |t| (0..n_points).map(move |k| f.get(t, k)))
                })
                .collect();
            Some(FieldSeries::with_offset(n_points, &values, offset))
        };
        let mut joined = Self::new(first.grid.clone(), time)?.with_source(first.source.clone());
        if let (Some(u), Some(v)) = (join(&|p| p.u10.as_ref()), join(&|p| p.v10.as_ref())) {
            joined = joined.with_wind(u, v);
        }
        for field in AtmosphereField::ALL {
            if let Some(series) = join(&|p| p.field(field)) {
                joined = joined.with_field(field, series);
            }
        }
        Ok(joined)
    }

    /// One-line description of the grid, times and fields.
    pub fn summary(&self) -> String {
        let (ny, nx) = self.grid.dims();
        let (x0, y0, x1, y1) = self.grid.bbox();
        let fields: Vec<&str> = std::iter::once(self.has_wind().then_some("wind"))
            .chain(
                AtmosphereField::ALL
                    .iter()
                    .map(|&f| self.has(f).then_some(f.name())),
            )
            .flatten()
            .collect();
        format!(
            "Atmosphere: {ny}x{nx} grid, {} times ({} h), lon [{x0:.2}, {x1:.2}], lat [{y0:.2}, {y1:.2}], {} ({})",
            self.time.len(),
            (self.time_range().1 - self.time_range().0) / 3600.0,
            fields.join(" + "),
            self.source
        )
    }
}

/// Wind (east, north) from speed and the direction it blows *from*
/// (degrees clockwise from north, meteorological convention).
#[inline]
pub fn wind_from_direction(speed: f64, from_deg: f64) -> (f64, f64) {
    let (s, c) = from_deg.to_radians().sin_cos();
    (-speed * s, -speed * c)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn part(times: &[f64], value: f32) -> AtmosphereReader {
        let grid = GeoGrid::regular(vec![5.0, 6.0], vec![60.0, 61.0]).unwrap();
        let n = times.len() * 4;
        AtmosphereReader::new(grid, times.to_vec())
            .unwrap()
            .with_wind(
                FieldSeries::new(4, vec![value; n]),
                FieldSeries::new(4, vec![0.0; n]),
            )
            .with_field(
                AtmosphereField::Pressure,
                FieldSeries::with_offset(4, &vec![P_REFERENCE + value as f64; n], P_REFERENCE),
            )
            .with_field(
                AtmosphereField::Shortwave,
                FieldSeries::new(4, vec![100.0 * value; n]),
            )
    }

    #[test]
    fn later_forecast_wins_the_overlap() {
        let a = part(&[0.0, 1.0, 2.0, 3.0], 1.0);
        let b = part(&[2.0, 3.0, 4.0], 2.0);
        let joined = AtmosphereReader::concat(vec![a, b]).unwrap();
        assert_eq!(joined.time, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        let wind = |t| joined.get(5.5, 60.5, t).unwrap().wind.unwrap().0;
        assert_eq!((wind(1.0), wind(2.0), wind(1.5)), (1.0, 2.0, 1.5));
        let state = joined.get(5.5, 60.5, 4.0).unwrap();
        let p = state.get(AtmosphereField::Pressure).unwrap();
        assert!((p - P_REFERENCE - 2.0).abs() < 1e-9);
        assert_eq!(state.get(AtmosphereField::Shortwave), Some(200.0));
        assert_eq!(state.get(AtmosphereField::Longwave), None);
        // Files with different fields do not join
        let other = part(&[5.0], 1.0).with_field(
            AtmosphereField::Longwave,
            FieldSeries::new(4, vec![300.0; 4]),
        );
        assert!(AtmosphereReader::concat(vec![part(&[4.0], 1.0), other]).is_err());
        assert!(AtmosphereReader::concat(vec![part(&[3.0], 1.0), part(&[1.0], 1.0)]).is_err());
        assert!(AtmosphereReader::concat(vec![part(&[1.0], 1.0), part(&[1.0, 2.0], 1.0)]).is_err());
    }

    #[test]
    fn meteorological_direction() {
        // From the south-west: blows towards the north-east
        let (u, v) = wind_from_direction(10.0, 225.0);
        assert!((u - 10.0 / 2f64.sqrt()).abs() < 1e-12 && (v - u).abs() < 1e-12);
        let (u, v) = wind_from_direction(5.0, 0.0);
        assert!(u.abs() < 1e-12 && (v + 5.0).abs() < 1e-12);
    }
}

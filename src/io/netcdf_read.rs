//! Reading parent ocean-model and weather-model NetCDF files.
//!
//! Shared CF handling for [`OceanModelReader::from_file`] and
//! [`AtmosphereReader::from_file`]:
//!
//! - Numeric variables are read by their declared type and unpacked
//!   (`scale_factor`, `add_offset`); `_FillValue`, `missing_value` and the
//!   netCDF default integer fills become NaN.
//! - Time is decoded from the CF `units`/`calendar` to Unix seconds.
//! - A variable's dimensions are matched to the grid: the last two are the
//!   horizontal ones (ρ-points, or ROMS u-/v-points with one column/row
//!   fewer, averaged to ρ-points), one may be time, one a vertical axis;
//!   dimensions of length 1 (MEPS's `height0`…`height6`) are dropped.

use std::path::Path;

use netcdf::{Extent, Variable};

use super::atmosphere::{AtmosphereReader, P_REFERENCE, wind_from_direction};
use super::field_series::FieldSeries;
use super::geo_grid::GeoGrid;
use super::netcdf_io::NetCDFError;
use super::ocean_model::OceanModelReader;
use super::profile_series::{ProfileLevels, ProfileSeries};
use super::z_levels::{depth_average_z, s_level_weights, weighted_mean};

/// A numeric attribute as `f64`.
pub(crate) fn attr_f64(var: &Variable, name: &str) -> Option<f64> {
    use netcdf::AttributeValue as A;
    match var.attribute_value(name)?.ok()? {
        A::Double(d) => Some(d),
        A::Float(f) => Some(f as f64),
        A::Schar(x) => Some(x as f64),
        A::Uchar(x) => Some(x as f64),
        A::Short(x) => Some(x as f64),
        A::Ushort(x) => Some(x as f64),
        A::Int(x) => Some(x as f64),
        A::Uint(x) => Some(x as f64),
        A::Longlong(x) => Some(x as f64),
        A::Ulonglong(x) => Some(x as f64),
        A::Doubles(d) => d.first().copied(),
        A::Floats(f) => f.first().map(|&f| f as f64),
        _ => None,
    }
}

/// A string attribute.
pub(crate) fn attr_string(var: &Variable, name: &str) -> Option<String> {
    match var.attribute_value(name)?.ok()? {
        netcdf::AttributeValue::Str(s) => Some(s),
        _ => None,
    }
}

/// NetCDF default fill value of an integer type (`NC_FILL_*`), used when a
/// packed variable has no `_FillValue` attribute.
fn default_integer_fill(int_type: netcdf::types::IntType) -> Option<f64> {
    use netcdf::types::IntType;
    match int_type {
        IntType::I8 => Some(-127.0),
        IntType::I16 => Some(-32_767.0),
        IntType::I32 => Some(-2_147_483_647.0),
        IntType::U8 => Some(255.0),
        IntType::U16 => Some(65_535.0),
        IntType::U32 => Some(4_294_967_295.0),
        IntType::I64 | IntType::U64 => None,
    }
}

/// Values of `var` over `extents` (all of it if empty), read by the declared
/// type and CF-unpacked; missing values are NaN.
///
/// Reading by the declared type matters: asking the library for `i16` first
/// makes it convert unpacked floats to integers (0.37 m of SSH became 0).
pub(crate) fn read_values(var: &Variable, extents: Vec<Extent>) -> Result<Vec<f64>, NetCDFError> {
    let packed_integer = match var.vartype() {
        netcdf::types::NcVariableType::Int(int_type) => Some(int_type),
        netcdf::types::NcVariableType::Float(_) => None,
        other => {
            return Err(NetCDFError::InvalidData(format!(
                "variable {}: unsupported type {other:?}",
                var.name()
            )));
        }
    };
    let raw: Vec<f64> = if extents.is_empty() {
        var.get_values(..)?
    } else {
        var.get_values(extents)?
    };
    let scale = attr_f64(var, "scale_factor").unwrap_or(1.0);
    let offset = attr_f64(var, "add_offset").unwrap_or(0.0);
    let fill =
        attr_f64(var, "_FillValue").or_else(|| packed_integer.and_then(default_integer_fill));
    let missing = attr_f64(var, "missing_value");
    Ok(raw
        .into_iter()
        .map(|v| {
            if !v.is_finite() || Some(v) == fill || Some(v) == missing || v.abs() > 1e30 {
                f64::NAN
            } else {
                v * scale + offset
            }
        })
        .collect())
}

/// The first of `names` that the file has.
fn find_var<'f>(file: &'f netcdf::File, names: &[&str]) -> Option<Variable<'f>> {
    names.iter().find_map(|n| file.variable(n))
}

/// Grid from the first matching latitude/longitude variables (1D or 2D).
fn read_grid(
    file: &netcdf::File,
    lat_names: &[&str],
    lon_names: &[&str],
) -> Result<GeoGrid, NetCDFError> {
    let missing = || NetCDFError::MissingVariable(format!("latitude ({})", lat_names.join(", ")));
    let lat_var = find_var(file, lat_names).ok_or_else(missing)?;
    let lon_var = find_var(file, lon_names).ok_or_else(|| {
        NetCDFError::MissingVariable(format!("longitude ({})", lon_names.join(", ")))
    })?;
    let (lat, lon) = (
        read_values(&lat_var, vec![])?,
        read_values(&lon_var, vec![])?,
    );
    let invalid = |e: super::geo_grid::GeoGridError| NetCDFError::InvalidData(e.to_string());
    match lat_var.dimensions() {
        [y, x] => GeoGrid::curvilinear(y.len(), x.len(), lon, lat).map_err(invalid),
        [_] => GeoGrid::regular(lon, lat).map_err(invalid),
        dims => Err(NetCDFError::InvalidData(format!(
            "latitude has {} dimensions",
            dims.len()
        ))),
    }
}

/// Snapshot times as Unix seconds and the name of the time dimension.
///
/// Decoded with the CF `units` (`"<unit> since <date>"`) and `calendar`. A
/// time variable without `units`, with an unsupported calendar, or with
/// non-increasing values is an error: raw values in an unknown unit silently
/// mis-time the forcing. A file without a time variable is one
/// time-invariant snapshot.
pub(crate) fn read_time(file: &netcdf::File) -> Result<(Vec<f64>, Option<String>), NetCDFError> {
    for name in ["time", "ocean_time", "Time", "valid_time"] {
        let Some(var) = file.variable(name) else {
            continue;
        };
        let invalid =
            |msg: String| NetCDFError::InvalidData(format!("time variable {name:?}: {msg}"));
        let units = attr_string(&var, "units")
            .ok_or_else(|| invalid("no `units` attribute".to_string()))?;
        let calendar = attr_string(&var, "calendar");
        let (seconds_per_unit, reference) =
            super::datetime::parse_cf_time_units(&units, calendar.as_deref()).map_err(invalid)?;
        let raw: Vec<f64> = var.get_values(..)?;
        let time: Vec<f64> = raw
            .iter()
            .map(|&v| reference + v * seconds_per_unit)
            .collect();
        if time.iter().any(|t| !t.is_finite()) || time.windows(2).any(|w| w[1] <= w[0]) {
            return Err(invalid(
                "values must be finite and strictly increasing".to_string(),
            ));
        }
        let dim = var.dimensions().first().map(|d| d.name());
        return Ok((time, dim));
    }
    Ok((vec![0.0], None))
}

/// Horizontal position of a variable's points on the grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stagger {
    /// ρ-points: the grid itself
    Rho,
    /// ROMS u-points: between columns (`nx − 1`)
    U,
    /// ROMS v-points: between rows (`ny − 1`)
    V,
}

/// How a variable's dimensions map onto time, vertical and the grid.
#[derive(Clone, Debug)]
struct Shape {
    lens: Vec<usize>,
    time_axis: Option<usize>,
    /// Axis and name of the vertical dimension
    vertical: Option<(usize, String)>,
    /// Axes read at index 0 (length-1 dimensions, ensemble members)
    fixed: Vec<usize>,
    stagger: Stagger,
}

impl Shape {
    /// The shape of `var` on an `ny × nx` grid; `None` if its horizontal
    /// dimensions do not fit the grid.
    fn of(
        var: &Variable,
        time_dim: Option<&str>,
        (ny, nx): (usize, usize),
    ) -> Result<Option<Self>, NetCDFError> {
        let dims = var.dimensions();
        let n = dims.len();
        if n < 2 {
            return Ok(None);
        }
        let lens: Vec<usize> = dims.iter().map(|d| d.len()).collect();
        let stagger = match (lens[n - 2], lens[n - 1]) {
            (a, b) if (a, b) == (ny, nx) => Stagger::Rho,
            (a, b) if (a, b) == (ny, nx - 1) => Stagger::U,
            (a, b) if (a, b) == (ny - 1, nx) => Stagger::V,
            _ => return Ok(None),
        };
        let mut shape = Self {
            lens: lens.clone(),
            time_axis: None,
            vertical: None,
            fixed: Vec::new(),
            stagger,
        };
        for (axis, dim) in dims[..n - 2].iter().enumerate() {
            let name = dim.name();
            if Some(name.as_str()) == time_dim {
                shape.time_axis = Some(axis);
            } else if lens[axis] == 1 || name == "ensemble_member" {
                shape.fixed.push(axis);
            } else if shape.vertical.is_none() {
                shape.vertical = Some((axis, name));
            } else {
                return Err(NetCDFError::InvalidData(format!(
                    "variable {}: unsupported dimension {name}",
                    var.name()
                )));
            }
        }
        Ok(Some(shape))
    }

    /// Size of one horizontal slab on the variable's own points.
    fn plane(&self) -> usize {
        let n = self.lens.len();
        self.lens[n - 2] * self.lens[n - 1]
    }

    /// Number of vertical levels (1 without a vertical axis).
    fn levels(&self) -> usize {
        self.vertical
            .as_ref()
            .map_or(1, |(axis, _)| self.lens[*axis])
    }

    /// Extents selecting snapshot `t` (all levels, whole plane).
    fn at_time(&self, t: usize) -> Vec<Extent> {
        (0..self.lens.len())
            .map(|axis| {
                if Some(axis) == self.time_axis {
                    Extent::Index(t)
                } else if self.fixed.contains(&axis) {
                    Extent::Index(0)
                } else {
                    Extent::from(..)
                }
            })
            .collect()
    }
}

/// Values on u-/v-points averaged to the ρ-points between them; the edge
/// rows or columns take the nearest value.
fn destagger(values: &[f64], stagger: Stagger, (ny, nx): (usize, usize)) -> Vec<f64> {
    match stagger {
        Stagger::Rho => values.to_vec(),
        Stagger::U => {
            let mut out = Vec::with_capacity(ny * nx);
            for j in 0..ny {
                let row = &values[j * (nx - 1)..(j + 1) * (nx - 1)];
                for i in 0..nx {
                    out.push(match i {
                        0 => row[0],
                        i if i == nx - 1 => row[nx - 2],
                        i => mean_or_either(row[i - 1], row[i]),
                    });
                }
            }
            out
        }
        Stagger::V => {
            let mut out = Vec::with_capacity(ny * nx);
            for j in 0..ny {
                for i in 0..nx {
                    let at = |jj: usize| values[jj * nx + i];
                    out.push(match j {
                        0 => at(0),
                        j if j == ny - 1 => at(ny - 2),
                        j => mean_or_either(at(j - 1), at(j)),
                    });
                }
            }
            out
        }
    }
}

/// Mean of two values, or the one that is not NaN (a u-point next to land).
fn mean_or_either(a: f64, b: f64) -> f64 {
    match (a.is_finite(), b.is_finite()) {
        (true, true) => 0.5 * (a + b),
        (true, false) => a,
        (false, true) => b,
        (false, false) => f64::NAN,
    }
}

/// A 2D (time-invariant, single-level) field on the ρ-grid, if the file has
/// one of `names`.
fn read_static(
    file: &netcdf::File,
    names: &[&str],
    time_dim: Option<&str>,
    dims: (usize, usize),
) -> Result<Option<Vec<f64>>, NetCDFError> {
    for name in names {
        let Some(var) = file.variable(name) else {
            continue;
        };
        let Some(shape) = Shape::of(&var, time_dim, dims)? else {
            continue;
        };
        if shape.time_axis.is_some() || shape.vertical.is_some() {
            continue;
        }
        let values = read_values(&var, shape.at_time(0))?;
        return Ok(Some(destagger(&values, shape.stagger, dims)));
    }
    Ok(None)
}

/// Vertical coordinate of 3D fields.
enum Vertical {
    /// Fixed depths (m, positive down), in the file's level order
    Z(Vec<f64>),
    /// ROMS s-levels: per ρ-point layer weights of the depth mean (bottom
    /// first; `None` on land)
    S(Vec<Option<Vec<f64>>>),
}

impl Vertical {
    /// The vertical coordinate named `dim`.
    fn read(
        file: &netcdf::File,
        dim: &str,
        depth: Option<&[f64]>,
        n_points: usize,
    ) -> Result<Self, NetCDFError> {
        let invalid = |msg: String| NetCDFError::InvalidData(format!("vertical axis {dim}: {msg}"));
        if dim.starts_with("s_rho") {
            let get = |name: &str| -> Result<Vec<f64>, NetCDFError> {
                let var = file
                    .variable(name)
                    .ok_or_else(|| invalid(format!("s-levels need `{name}`")))?;
                read_values(&var, vec![])
            };
            let (s_w, cs_w) = (get("s_w")?, get("Cs_w")?);
            let scalar = |name: &str| -> Option<f64> {
                file.variable(name)
                    .and_then(|v| v.get_value::<f64, _>(..).ok())
                    .or_else(|| match file.attribute(name)?.value().ok()? {
                        netcdf::AttributeValue::Double(d) => Some(d),
                        netcdf::AttributeValue::Float(f) => Some(f as f64),
                        netcdf::AttributeValue::Int(i) => Some(i as f64),
                        netcdf::AttributeValue::Short(i) => Some(i as f64),
                        _ => None,
                    })
            };
            let hc = scalar("hc").ok_or_else(|| invalid("s-levels need `hc`".into()))?;
            let vtransform = scalar("Vtransform")
                .ok_or_else(|| invalid("s-levels need `Vtransform`".into()))?
                as i32;
            let depth = depth.ok_or_else(|| invalid("s-levels need the depth `h`".into()))?;
            let weights = (0..n_points)
                .map(|k| s_level_weights(&s_w, &cs_w, hc, depth[k], vtransform))
                .collect();
            if s_level_weights(&s_w, &cs_w, hc, 100.0, vtransform).is_none() {
                return Err(invalid(format!(
                    "unsupported Vtransform {vtransform} or bad s_w/Cs_w"
                )));
            }
            return Ok(Self::S(weights));
        }
        let var = file
            .variable(dim)
            .ok_or_else(|| invalid("no coordinate variable".into()))?;
        let values = read_values(&var, vec![])?;
        let positive = attr_string(&var, "positive").map(|p| p.to_ascii_lowercase());
        let depths: Vec<f64> = match positive.as_deref() {
            Some("down") => values,
            Some("up") => values.iter().map(|z| -z).collect(),
            // No attribute: depths if all ≥ 0, heights if all ≤ 0
            _ if values.iter().all(|&z| z >= 0.0) => values,
            _ if values.iter().all(|&z| z <= 0.0) => values.iter().map(|z| -z).collect(),
            _ => return Err(invalid("no `positive` attribute and mixed signs".into())),
        };
        if depths.iter().any(|d| !d.is_finite()) {
            return Err(invalid("non-finite levels".into()));
        }
        Ok(Self::Z(depths))
    }

    /// Depth mean of the column `column` (one value per level, file order)
    /// at ρ-point `k` of still-water depth `depth` (NaN if unknown).
    fn mean(&self, column: &[Option<f64>], k: usize, depth: f64, order: &[usize]) -> Option<f64> {
        match self {
            Self::S(weights) => weighted_mean(weights[k].as_ref()?, column),
            Self::Z(levels) => {
                let (z, v): (Vec<f64>, Vec<Option<f64>>) =
                    order.iter().map(|&l| (levels[l], column[l])).unzip();
                // Unknown depth: the column ends at its deepest value
                let bed = if depth > 0.0 {
                    depth
                } else {
                    z.iter()
                        .zip(&v)
                        .filter(|(_, v)| v.is_some())
                        .map(|(z, _)| *z)
                        .fold(f64::NAN, f64::max)
                        + 1e-9
                };
                depth_average_z(&z, &v, bed)
            }
        }
    }

    /// Index of the surface level.
    fn surface(&self, n_levels: usize) -> usize {
        match self {
            // ROMS s-coordinates run from the bed up
            Self::S(_) => n_levels - 1,
            Self::Z(levels) => (0..levels.len())
                .min_by(|&a, &b| levels[a].total_cmp(&levels[b]))
                .unwrap_or(0),
        }
    }

    /// Where the levels lie from the surface down, and the file's level of
    /// each: z-levels at their depths; s-levels at the fraction of the column
    /// of their layer's centre, halfway between its interfaces (the weights
    /// of the depth mean; ROMS's `s_rho`/`Cs_r` differ by the curvature of
    /// the stretching).
    fn profile_levels(&self, n_points: usize) -> (ProfileLevels, Vec<usize>) {
        match self {
            Self::Z(levels) => {
                let order = self.order();
                let depths = order.iter().map(|&l| levels[l]).collect();
                (ProfileLevels::Depth(depths), order)
            }
            Self::S(weights) => {
                let n = weights.iter().flatten().next().map_or(0, Vec::len);
                // Bottom first in the file
                let order: Vec<usize> = (0..n).rev().collect();
                let mut fractions = vec![f64::NAN; n_points * n];
                for (k, w) in weights.iter().enumerate() {
                    let Some(w) = w else { continue };
                    let mut above = 0.0;
                    for (j, &l) in order.iter().enumerate() {
                        fractions[k * n + j] = above + 0.5 * w[l];
                        above += w[l];
                    }
                }
                (ProfileLevels::Fraction(fractions), order)
            }
        }
    }

    /// Level order from the surface down (z-levels).
    fn order(&self) -> Vec<usize> {
        match self {
            Self::S(w) => (0..w.iter().flatten().next().map_or(0, Vec::len)).collect(),
            Self::Z(levels) => {
                let mut order: Vec<usize> = (0..levels.len()).collect();
                order.sort_by(|&a, &b| levels[a].total_cmp(&levels[b]));
                order
            }
        }
    }
}

/// What a 3D field is reduced to.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Reduce {
    DepthMean,
    Surface,
}

/// Context for reading fields onto the ρ-grid.
struct FieldReader<'f> {
    file: &'f netcdf::File,
    dims: (usize, usize),
    n_times: usize,
    time_dim: Option<String>,
    depth: Option<Vec<f64>>,
    /// ρ-points outside the model (mask 0)
    land: Option<Vec<bool>>,
}

impl FieldReader<'_> {
    fn n_points(&self) -> usize {
        self.dims.0 * self.dims.1
    }

    fn shape(&self, var: &Variable) -> Result<Option<Shape>, NetCDFError> {
        Shape::of(var, self.time_dim.as_deref(), self.dims)
    }

    /// `var` on the ρ-grid per snapshot, strided `[time][point]`, with 3D
    /// fields reduced by `reduce`.
    fn read(&self, var: &Variable, shape: &Shape, reduce: Reduce) -> Result<Vec<f64>, NetCDFError> {
        let n = self.n_points();
        let vertical = match &shape.vertical {
            Some((_, dim)) => Some(Vertical::read(self.file, dim, self.depth.as_deref(), n)?),
            None => None,
        };
        let order = vertical.as_ref().map(Vertical::order).unwrap_or_default();
        let n_levels = shape.levels();
        let mut out = Vec::with_capacity(self.n_times * n);
        let snapshots = if shape.time_axis.is_some() {
            self.n_times
        } else {
            1
        };
        for t in 0..snapshots {
            let raw = read_values(var, shape.at_time(t))?;
            // [level][point] on ρ-points
            let layers: Vec<Vec<f64>> = raw
                .chunks_exact(shape.plane())
                .map(|layer| destagger(layer, shape.stagger, self.dims))
                .collect();
            let plane: Vec<f64> = match (&vertical, reduce) {
                (None, _) => layers.into_iter().next().unwrap_or_default(),
                (Some(v), Reduce::Surface) => layers[v.surface(n_levels)].clone(),
                (Some(v), Reduce::DepthMean) => {
                    let mut column = vec![None; n_levels];
                    (0..n)
                        .map(|k| {
                            for (l, c) in column.iter_mut().enumerate() {
                                let x = layers[l][k];
                                *c = x.is_finite().then_some(x);
                            }
                            let depth = self.depth.as_ref().map_or(f64::NAN, |d| d[k]);
                            v.mean(&column, k, depth, &order).unwrap_or(f64::NAN)
                        })
                        .collect()
                }
            };
            out.extend(plane);
        }
        // A time-invariant field holds for every snapshot
        if snapshots == 1 && self.n_times > 1 {
            let plane = out.clone();
            for _ in 1..self.n_times {
                out.extend_from_slice(&plane);
            }
        }
        if let Some(land) = &self.land {
            for (k, v) in out.iter_mut().enumerate() {
                if land[k % n] {
                    *v = f64::NAN;
                }
            }
        }
        Ok(out)
    }

    /// The profiles of the 3D field `var` on the ρ-grid per snapshot, levels
    /// from the surface down (see [`ProfileSeries`]).
    fn read_profiles(&self, var: &Variable, shape: &Shape) -> Result<ProfileSeries, NetCDFError> {
        let n = self.n_points();
        let Some((_, dim)) = &shape.vertical else {
            return Err(NetCDFError::InvalidData(format!(
                "{} has no vertical axis",
                var.name()
            )));
        };
        let vertical = Vertical::read(self.file, dim, self.depth.as_deref(), n)?;
        let (levels, order) = vertical.profile_levels(n);
        let snapshots = if shape.time_axis.is_some() {
            self.n_times
        } else {
            1
        };
        let mut data = Vec::with_capacity(self.n_times * n * order.len());
        for t in 0..snapshots {
            let raw = read_values(var, shape.at_time(t))?;
            let layers: Vec<Vec<f64>> = raw
                .chunks_exact(shape.plane())
                .map(|layer| destagger(layer, shape.stagger, self.dims))
                .collect();
            for k in 0..n {
                let land = self.land.as_ref().is_some_and(|land| land[k]);
                data.extend(order.iter().map(|&l| {
                    let x = layers[l][k];
                    if land || !x.is_finite() {
                        f32::NAN
                    } else {
                        x as f32
                    }
                }));
            }
        }
        // A time-invariant field holds for every snapshot
        if snapshots == 1 && self.n_times > 1 {
            let snapshot = data.clone();
            for _ in 1..self.n_times {
                data.extend_from_slice(&snapshot);
            }
        }
        Ok(ProfileSeries::new(n, levels, data))
    }

    /// The first of `names` that fits the grid, as a series.
    fn series(
        &self,
        names: &[&str],
        reduce: Reduce,
    ) -> Result<Option<(String, Vec<f64>)>, NetCDFError> {
        for name in names {
            let Some(var) = self.file.variable(name) else {
                continue;
            };
            let Some(shape) = self.shape(&var)? else {
                continue;
            };
            return Ok(Some((name.to_string(), self.read(&var, &shape, reduce)?)));
        }
        Ok(None)
    }
}

fn to_series(n_points: usize, values: &[f64]) -> FieldSeries {
    FieldSeries::new(n_points, values.iter().map(|&v| v as f32).collect())
}

/// Rotate grid-relative components `(a, b)` in place to east/north, with
/// the angle of the grid's x axis from east per point.
fn rotate_to_east_north(a: &mut [f64], b: &mut [f64], angles: &[f64]) {
    let n = angles.len();
    for (k, (x, y)) in a.iter_mut().zip(b.iter_mut()).enumerate() {
        let (s, c) = angles[k % n].sin_cos();
        (*x, *y) = (*x * c - *y * s, *x * s + *y * c);
    }
}

/// [`rotate_to_east_north`] for profiles: the angle of each point for all
/// its levels.
fn rotate_profiles_to_east_north(u: &mut ProfileSeries, v: &mut ProfileSeries, angles: &[f64]) {
    let (n, nl) = (angles.len(), u.n_levels());
    let mut rotated_u = Vec::with_capacity(u.n_times() * n * nl);
    let mut rotated_v = Vec::with_capacity(rotated_u.capacity());
    for t in 0..u.n_times() {
        for (k, &angle) in angles.iter().enumerate() {
            let (s, c) = angle.sin_cos();
            for (&a, &b) in u.column(t, k).iter().zip(v.column(t, k)) {
                let (a, b) = (a as f64, b as f64);
                rotated_u.push((a * c - b * s) as f32);
                rotated_v.push((a * s + b * c) as f32);
            }
        }
    }
    *u = ProfileSeries::new(n, u.levels().clone(), rotated_u);
    *v = ProfileSeries::new(n, v.levels().clone(), rotated_v);
}

/// Axes of a pair of vector components in a file.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Axes {
    /// Eastward and northward
    EastNorth,
    /// Along the grid's x and y axes
    Grid,
    /// Speed and the direction it comes from (degrees from north)
    SpeedFrom,
}

impl OceanModelReader {
    /// Read parent-model output from a NetCDF file (or OPeNDAP URL).
    ///
    /// - Grid: `lat`/`lon`, `latitude`/`longitude`, `lat_rho`/`lon_rho` or
    ///   `nav_lat`/`nav_lon`, 1D or 2D; `mask_rho` marks land.
    /// - Depth: `h` (or `sea_floor_depth_below_sea_level`).
    /// - SSH: `zeta`, `ssh`, `sea_surface_height` or `sea_surface_elevation`.
    ///   Not `h`, which in ROMS/NorKyst files is the depth.
    /// - Velocity, the first pair present: the barotropic
    ///   `ubar_eastward`/`vbar_northward` or grid-relative `ubar`/`vbar`, else
    ///   the 3D `u_eastward`/`v_northward`, `u`/`v` or
    ///   `eastward_sea_water_velocity`/`northward_sea_water_velocity`
    ///   averaged over the water column: on z-levels by the trapezoid rule
    ///   down to the bed ([`depth_average_z`], the level axis's `positive`
    ///   attribute gives its direction; NorKyst's `depth` is positive down,
    ///   level 0 at the surface), on ROMS s-levels (`s_rho`) weighted by the
    ///   layer thicknesses ([`s_level_weights`], from `s_w`, `Cs_w`, `hc`,
    ///   `Vtransform` and `h`). Grid-relative components on u-/v-points are
    ///   averaged to ρ-points and rotated to east/north by `angle`, or by the
    ///   grid angle from the coordinates.
    /// - Temperature and salinity: the surface level.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, NetCDFError> {
        Self::read_file(path.as_ref(), false)
    }

    /// [`Self::from_file`], and the profiles of the 3D velocity (the first
    /// 3D pair of those listed there), temperature and salinity for 3D
    /// nesting ([`ProfileSeries`]: z-levels at their depths, s-levels at the
    /// fraction of the column of their layer centres). Grid-relative
    /// components are rotated to east/north as the depth means are.
    ///
    /// The profiles are `n_times × n_points × n_levels` values each (a
    /// 3-day hourly 100 × 100 NorKyst subset on 16 levels: 46 MB per field).
    pub fn from_file_with_profiles(path: impl AsRef<Path>) -> Result<Self, NetCDFError> {
        Self::read_file(path.as_ref(), true)
    }

    fn read_file(path: &Path, profiles: bool) -> Result<Self, NetCDFError> {
        let file = netcdf::open(path)?;
        let grid = read_grid(
            &file,
            &["lat", "latitude", "lat_rho", "nav_lat"],
            &["lon", "longitude", "lon_rho", "nav_lon"],
        )?;
        let dims = grid.dims();
        let n_points = grid.len();
        let (time, time_dim) = read_time(&file)?;
        let n_times = time.len();
        let land = read_static(&file, &["mask_rho"], time_dim.as_deref(), dims)?.map(|m| {
            m.iter()
                .map(|&m| m.is_nan() || m <= 0.5)
                .collect::<Vec<bool>>()
        });
        let mut depth = read_static(
            &file,
            &["h", "sea_floor_depth_below_sea_level"],
            time_dim.as_deref(),
            dims,
        )?;
        if let (Some(depth), Some(land)) = (&mut depth, &land) {
            for (d, &l) in depth.iter_mut().zip(land) {
                if l {
                    *d = f64::NAN;
                }
            }
        }
        let fields = FieldReader {
            file: &file,
            dims,
            n_times,
            time_dim,
            depth: depth.clone(),
            land,
        };
        let invalid = NetCDFError::InvalidData;
        let mut reader = Self::new(grid, time).map_err(invalid)?;
        if let Some(depth) = depth {
            reader = reader.with_depth(depth);
        }
        if let Some((_, ssh)) = fields.series(
            &["zeta", "ssh", "sea_surface_height", "sea_surface_elevation"],
            Reduce::Surface,
        )? {
            reader = reader.with_ssh(to_series(n_points, &ssh));
        }

        const PAIRS: [(&str, &str, Axes); 5] = [
            ("ubar_eastward", "vbar_northward", Axes::EastNorth),
            ("ubar", "vbar", Axes::Grid),
            ("u_eastward", "v_northward", Axes::EastNorth),
            ("u", "v", Axes::Grid),
            (
                "eastward_sea_water_velocity",
                "northward_sea_water_velocity",
                Axes::EastNorth,
            ),
        ];
        for (u_name, v_name, axes) in PAIRS {
            let (Some(u_var), Some(v_var)) = (file.variable(u_name), file.variable(v_name)) else {
                continue;
            };
            let (Some(u_shape), Some(v_shape)) = (fields.shape(&u_var)?, fields.shape(&v_var)?)
            else {
                continue;
            };
            let (mut u, mut v) = (
                fields.read(&u_var, &u_shape, Reduce::DepthMean)?,
                fields.read(&v_var, &v_shape, Reduce::DepthMean)?,
            );
            let mut source = format!("{u_name}/{v_name}");
            if let Some((_, dim)) = &u_shape.vertical {
                source += &format!(" averaged over {dim}");
            }
            if axes == Axes::Grid {
                let angles = match read_static(&file, &["angle"], fields.time_dim.as_deref(), dims)?
                {
                    Some(a) => {
                        source += ", rotated by `angle`";
                        a
                    }
                    None => {
                        source += ", rotated by the grid angle";
                        reader.grid().x_axis_angles()
                    }
                };
                rotate_to_east_north(&mut u, &mut v, &angles);
            }
            reader =
                reader.with_velocity(to_series(n_points, &u), to_series(n_points, &v), &source);
            break;
        }

        let mut velocity_profiles = None;
        if profiles {
            for (u_name, v_name, axes) in PAIRS {
                let (Some(u_var), Some(v_var)) = (file.variable(u_name), file.variable(v_name))
                else {
                    continue;
                };
                let (Some(u_shape), Some(v_shape)) = (fields.shape(&u_var)?, fields.shape(&v_var)?)
                else {
                    continue;
                };
                if u_shape.vertical.is_none() || v_shape.vertical.is_none() {
                    continue;
                }
                let (mut u, mut v) = (
                    fields.read_profiles(&u_var, &u_shape)?,
                    fields.read_profiles(&v_var, &v_shape)?,
                );
                if axes == Axes::Grid {
                    let angles =
                        match read_static(&file, &["angle"], fields.time_dim.as_deref(), dims)? {
                            Some(a) => a,
                            None => reader.grid().x_axis_angles(),
                        };
                    rotate_profiles_to_east_north(&mut u, &mut v, &angles);
                }
                velocity_profiles = Some((u, v));
                break;
            }
        }

        let temperature = fields.series(
            &["temperature", "temp", "sea_water_temperature"],
            Reduce::Surface,
        )?;
        let salinity =
            fields.series(&["salinity", "salt", "sea_water_salinity"], Reduce::Surface)?;
        let reader = reader.with_tracers(
            temperature.map(|(_, t)| to_series(n_points, &t)),
            salinity.map(|(_, s)| to_series(n_points, &s)),
        );
        if !profiles {
            return Ok(reader);
        }
        let profile = |names: &[&str]| -> Result<Option<ProfileSeries>, NetCDFError> {
            for name in names {
                let Some(var) = file.variable(name) else {
                    continue;
                };
                match fields.shape(&var)? {
                    Some(shape) if shape.vertical.is_some() => {
                        return fields.read_profiles(&var, &shape).map(Some);
                    }
                    _ => continue,
                }
            }
            Ok(None)
        };
        let temperature = profile(&["temperature", "temp", "sea_water_temperature"])?;
        let salinity = profile(&["salinity", "salt", "sea_water_salinity"])?;
        Ok(reader.with_profiles(velocity_profiles, temperature, salinity))
    }
}

impl AtmosphereReader {
    /// Read 10 m wind and sea-level pressure from a NetCDF file (or OPeNDAP
    /// URL) of a weather model.
    ///
    /// - Grid: `latitude`/`longitude` or `lat`/`lon`, 1D (ERA5, either
    ///   direction) or 2D (MEPS, MET Nordic, AROME-Arctic Lambert grids).
    /// - Wind, the first present of: `x_wind_10m`/`y_wind_10m` (grid-relative,
    ///   rotated to east/north by the local grid angle), `u10`/`v10`,
    ///   `10u`/`10v`, `Uwind_eastward`/`Vwind_northward` (NorKyst's forcing),
    ///   `Uwind`/`Vwind`, or `wind_speed_10m` with `wind_direction_10m`
    ///   (MET Nordic; the direction the wind blows from). A `standard_name`
    ///   of `x_wind` or `eastward_wind` overrides the axes.
    /// - Pressure: `air_pressure_at_sea_level`, `msl`, `prmsl`, `mslp` or
    ///   `Pair`, in Pa, or in hPa/mbar by its `units` (without units, values
    ///   below 2000 are taken as hPa).
    ///
    /// Errors if the file has neither wind nor pressure.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, NetCDFError> {
        let path = path.as_ref();
        let file = netcdf::open(path)?;
        let grid = read_grid(
            &file,
            &["latitude", "lat", "nav_lat"],
            &["longitude", "lon", "nav_lon"],
        )?;
        let dims = grid.dims();
        let n_points = grid.len();
        let (time, time_dim) = read_time(&file)?;
        let fields = FieldReader {
            file: &file,
            dims,
            n_times: time.len(),
            time_dim,
            depth: None,
            land: None,
        };
        let mut reader = Self::new(grid, time).map_err(NetCDFError::InvalidData)?;
        let mut sources = Vec::new();

        const PAIRS: [(&str, &str, Axes); 6] = [
            ("x_wind_10m", "y_wind_10m", Axes::Grid),
            ("u10", "v10", Axes::EastNorth),
            ("10u", "10v", Axes::EastNorth),
            ("Uwind_eastward", "Vwind_northward", Axes::EastNorth),
            ("Uwind", "Vwind", Axes::EastNorth),
            ("wind_speed_10m", "wind_direction_10m", Axes::SpeedFrom),
        ];
        for (a_name, b_name, axes) in PAIRS {
            let (Some(a_var), Some(b_var)) = (file.variable(a_name), file.variable(b_name)) else {
                continue;
            };
            let (Some(a_shape), Some(b_shape)) = (fields.shape(&a_var)?, fields.shape(&b_var)?)
            else {
                continue;
            };
            let axes = match attr_string(&a_var, "standard_name").as_deref() {
                Some("x_wind") => Axes::Grid,
                Some("eastward_wind") => Axes::EastNorth,
                _ => axes,
            };
            for shape in [&a_shape, &b_shape] {
                if shape.vertical.is_some() {
                    return Err(NetCDFError::InvalidData(format!(
                        "{a_name}/{b_name}: unexpected vertical dimension"
                    )));
                }
            }
            let (mut a, mut b) = (
                fields.read(&a_var, &a_shape, Reduce::Surface)?,
                fields.read(&b_var, &b_shape, Reduce::Surface)?,
            );
            match axes {
                Axes::EastNorth => sources.push(format!("wind {a_name}/{b_name}")),
                Axes::Grid => {
                    rotate_to_east_north(&mut a, &mut b, &reader.grid.x_axis_angles());
                    sources.push(format!("wind {a_name}/{b_name} rotated from the grid axes"));
                }
                Axes::SpeedFrom => {
                    for (s, d) in a.iter_mut().zip(b.iter_mut()) {
                        (*s, *d) = wind_from_direction(*s, *d);
                    }
                    sources.push(format!("wind from {a_name} and {b_name}"));
                }
            }
            reader = reader.with_wind(to_series(n_points, &a), to_series(n_points, &b));
            break;
        }

        for name in ["air_pressure_at_sea_level", "msl", "prmsl", "mslp", "Pair"] {
            let Some(var) = file.variable(name) else {
                continue;
            };
            let Some(shape) = fields.shape(&var)? else {
                continue;
            };
            let mut p = fields.read(&var, &shape, Reduce::Surface)?;
            let units = attr_string(&var, "units").map(|u| u.trim().to_ascii_lowercase());
            let hpa = match units.as_deref() {
                Some("pa") => false,
                Some("hpa" | "mbar" | "millibar" | "mb") => true,
                Some(other) => {
                    return Err(NetCDFError::InvalidData(format!(
                        "{name}: unsupported pressure units {other:?}"
                    )));
                }
                None => p.iter().filter(|v| v.is_finite()).all(|&v| v < 2000.0),
            };
            if hpa {
                p.iter_mut().for_each(|v| *v *= 100.0);
            }
            reader = reader.with_pressure(FieldSeries::with_offset(n_points, &p, P_REFERENCE));
            sources.push(format!(
                "pressure {name}{}",
                if hpa { " (hPa)" } else { "" }
            ));
            break;
        }
        if !reader.has_wind() && !reader.has_pressure() {
            return Err(NetCDFError::MissingVariable(format!(
                "wind or sea-level pressure in {}",
                path.display()
            )));
        }
        Ok(reader.with_source(sources.join(", ")))
    }

    /// Read and join consecutive files (e.g. one per forecast run or per
    /// hour of an analysis); see [`AtmosphereReader::concat`].
    pub fn from_files<P: AsRef<Path>>(paths: &[P]) -> Result<Self, NetCDFError> {
        let parts = paths
            .iter()
            .map(Self::from_file)
            .collect::<Result<Vec<_>, _>>()?;
        Self::concat(parts).map_err(NetCDFError::InvalidData)
    }
}

/// Small NetCDF files for reader tests.
#[cfg(test)]
mod test_files {
    use std::path::{Path, PathBuf};

    /// 2024-01-30 06:00:00 UTC in Unix seconds.
    pub const T0: f64 = 1_706_594_400.0;
    /// Cell centre of the 2×2 grid (lat 63.0–63.1, lon 8.0–8.1).
    pub const CENTRE: (f64, f64) = (63.05, 8.05);

    /// How the `zeta` variable is stored.
    #[derive(Clone, Copy)]
    pub enum Zeta {
        /// Unpacked f32 values
        Float,
        /// i16 packed with scale_factor 0.001 and _FillValue −32767 at (0, 0, 0)
        PackedI16,
        /// No `zeta` at all
        Absent,
    }

    /// Write a regular 2×2 grid with hourly snapshots of uniform `ssh[t]` and
    /// uniform velocity `(0.2, −0.1)`, plus bathymetry `h` = 50 m.
    pub fn write(dir: &Path, time_units: Option<&str>, ssh: &[f64], zeta: Zeta) -> PathBuf {
        let path = dir.join("parent.nc");
        let n_t = ssh.len();
        let mut file = netcdf::create(&path).unwrap();
        file.add_dimension("time", n_t).unwrap();
        file.add_dimension("lat", 2).unwrap();
        file.add_dimension("lon", 2).unwrap();

        let mut lat = file.add_variable::<f64>("lat", &["lat"]).unwrap();
        lat.put_values(&[63.0, 63.1], ..).unwrap();
        let mut lon = file.add_variable::<f64>("lon", &["lon"]).unwrap();
        lon.put_values(&[8.0, 8.1], ..).unwrap();

        let mut time = file.add_variable::<f64>("time", &["time"]).unwrap();
        if let Some(units) = time_units {
            time.put_attribute("units", units).unwrap();
        }
        let hours: Vec<f64> = (0..n_t).map(|t| t as f64).collect();
        time.put_values(&hours, ..).unwrap();

        let mut h = file.add_variable::<f32>("h", &["lat", "lon"]).unwrap();
        h.put_values(&[50.0_f32; 4], ..).unwrap();

        let field =
            |value: f64| -> Vec<f64> { (0..n_t).flat_map(|t| [value * ssh[t]; 4]).collect() };
        match zeta {
            Zeta::Float => {
                let mut z = file
                    .add_variable::<f32>("zeta", &["time", "lat", "lon"])
                    .unwrap();
                let values: Vec<f32> = field(1.0).iter().map(|&v| v as f32).collect();
                z.put_values(&values, ..).unwrap();
            }
            Zeta::PackedI16 => {
                let mut z = file
                    .add_variable::<i16>("zeta", &["time", "lat", "lon"])
                    .unwrap();
                z.put_attribute("scale_factor", 0.001_f64).unwrap();
                z.put_attribute("add_offset", 0.0_f64).unwrap();
                z.put_attribute("_FillValue", -32_767_i16).unwrap();
                let mut values: Vec<i16> =
                    field(1000.0).iter().map(|&v| v.round() as i16).collect();
                values[0] = -32_767;
                z.put_values(&values, ..).unwrap();
            }
            Zeta::Absent => {}
        }
        for (name, value) in [("ubar_eastward", 0.2_f32), ("vbar_northward", -0.1_f32)] {
            let mut var = file
                .add_variable::<f32>(name, &["time", "lat", "lon"])
                .unwrap();
            var.put_values(&vec![value; 4 * n_t], ..).unwrap();
        }
        path
    }
}

#[cfg(test)]
mod tests {
    use super::test_files::{self, CENTRE, T0, Zeta};
    use super::*;
    use std::path::PathBuf;

    const HOURS: &str = "hours since 2024-01-30 06:00:00";

    fn read(units: Option<&str>, ssh: &[f64], zeta: Zeta) -> Result<OceanModelReader, NetCDFError> {
        let dir = tempfile::tempdir().unwrap();
        let path = test_files::write(dir.path(), units, ssh, zeta);
        OceanModelReader::from_file(path)
    }

    #[test]
    fn time_is_decoded_from_cf_units() {
        // P0.20: raw values were returned, so "hours since …" files were
        // compared against simulation seconds.
        let reader = read(Some(HOURS), &[0.1, 0.2, 0.3], Zeta::Float).unwrap();
        assert_eq!(reader.times(), [T0, T0 + 3600.0, T0 + 7200.0]);
        assert_eq!(reader.time_range(), Some((T0, T0 + 7200.0)));
    }

    #[test]
    fn time_without_units_is_an_error() {
        assert!(matches!(
            read(None, &[0.1, 0.2], Zeta::Float),
            Err(NetCDFError::InvalidData(_))
        ));
    }

    #[test]
    fn time_interpolation_does_not_clamp() {
        // P0.20: out-of-range times clamped to the first/last snapshot, so
        // the forcing froze silently.
        let reader = read(Some(HOURS), &[0.1, 0.3], Zeta::Float).unwrap();
        let (lat, lon) = CENTRE;
        let mid = reader
            .get_state_interpolated(lon, lat, T0 + 1800.0)
            .unwrap();
        assert!((mid.ssh.unwrap() - 0.2).abs() < 1e-6);
        assert!(reader.get_state_interpolated(lon, lat, T0 - 1.0).is_none());
        assert!(
            reader
                .get_state_interpolated(lon, lat, T0 + 3601.0)
                .is_none()
        );
        assert!(reader.covers_time(T0) && reader.covers_time(T0 + 3600.0));
    }

    #[test]
    fn float_ssh_is_not_truncated() {
        // P0.20: i16 was tried first, and netCDF converts floats to shorts on
        // request, so 0.37 m became 0.
        let reader = read(Some(HOURS), &[0.37], Zeta::Float).unwrap();
        let (lat, lon) = CENTRE;
        let ssh = reader.get_state(lon, lat, 0).unwrap().ssh.unwrap();
        assert!((ssh - 0.37).abs() < 1e-6, "ssh = {ssh}");
        assert_eq!(reader.depth(), Some(&[50.0; 4][..]));
        let (u, v) = reader.get_state(lon, lat, 0).unwrap().velocity.unwrap();
        assert!((u - 0.2).abs() < 1e-6 && (v + 0.1).abs() < 1e-6);
    }

    #[test]
    fn packed_i16_ssh_is_unpacked_with_fill() {
        let reader = read(Some(HOURS), &[0.37, -0.25], Zeta::PackedI16).unwrap();
        let ssh = reader.ssh().unwrap();
        assert!(ssh.get(0, 0).is_nan(), "fill value must become NaN");
        assert!((ssh.get(0, 3) - 0.37).abs() < 1e-6);
        assert!((ssh.get(1, 0) + 0.25).abs() < 1e-6);
        // The filled point is land for the whole record
        assert!(!reader.is_wet(0) && reader.is_wet(1));
    }

    #[test]
    fn bathymetry_h_is_not_read_as_ssh() {
        // P0.20: "h" was an SSH candidate; in ROMS/NorKyst it is the depth.
        let reader = read(Some(HOURS), &[0.1], Zeta::Absent).unwrap();
        assert!(reader.ssh().is_none());
    }

    /// Writes a 2 × 3 regular grid with the given extra variables.
    fn write_file(
        build: impl FnOnce(&mut netcdf::FileMut),
    ) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f.nc");
        let mut file = netcdf::create(&path).unwrap();
        file.add_dimension("time", 1).unwrap();
        file.add_dimension("Y", 2).unwrap();
        file.add_dimension("X", 3).unwrap();
        let mut lat = file.add_variable::<f64>("lat", &["Y", "X"]).unwrap();
        lat.put_values(&[63.0, 63.0, 63.0, 63.1, 63.1, 63.1], ..)
            .unwrap();
        let mut lon = file.add_variable::<f64>("lon", &["Y", "X"]).unwrap();
        lon.put_values(&[8.0, 8.1, 8.2, 8.0, 8.1, 8.2], ..).unwrap();
        let mut time = file.add_variable::<f64>("time", &["time"]).unwrap();
        time.put_attribute("units", "seconds since 1970-01-01")
            .unwrap();
        time.put_values(&[0.0], ..).unwrap();
        build(&mut file);
        drop(file);
        (dir, path)
    }

    /// A NorKyst-like z-level file: `depth` 0, 10, 50 m (`positive = "down"`),
    /// `h` 20 m in the first column and 60 m elsewhere, `u_eastward` 1, 0.5, 0
    /// (fill below the bed), `temperature` 12, 8, 8.
    fn z_level_file() -> (tempfile::TempDir, PathBuf) {
        write_file(|file| {
            file.add_dimension("depth", 3).unwrap();
            let mut depth = file.add_variable::<f64>("depth", &["depth"]).unwrap();
            depth.put_attribute("positive", "down").unwrap();
            depth.put_values(&[0.0, 10.0, 50.0], ..).unwrap();
            let mut h = file.add_variable::<f64>("h", &["Y", "X"]).unwrap();
            // The first column is 20 m deep, the rest 60 m
            h.put_values(&[20.0, 60.0, 60.0, 20.0, 60.0, 60.0], ..)
                .unwrap();
            // u = 1 at the surface, 0.5 at 10 m, 0 at 50 m (NaN below the bed)
            let dims = &["time", "depth", "Y", "X"];
            let mut u = file.add_variable::<f32>("u_eastward", dims).unwrap();
            u.put_attribute("_FillValue", -999.0_f32).unwrap();
            let mut values = vec![1.0_f32; 6];
            values.extend([0.5_f32; 6]);
            values.extend([-999.0, 0.0, 0.0, -999.0, 0.0, 0.0]);
            u.put_values(&values, ..).unwrap();
            let mut v = file.add_variable::<f32>("v_northward", dims).unwrap();
            v.put_values(&[0.0_f32; 18], ..).unwrap();
            let mut t = file
                .add_variable::<f32>("temperature", &["time", "depth", "Y", "X"])
                .unwrap();
            let mut temps = vec![12.0_f32; 6];
            temps.extend([8.0_f32; 12]);
            t.put_values(&temps, ..).unwrap();
        })
    }

    /// NorKyst z-level files: level 0 is the surface (`positive = "down"`),
    /// and the velocity is averaged down to the bed `h`. The old reader took
    /// the last level, the deepest, as the surface (P1.5).
    #[test]
    fn z_level_velocity_is_averaged_to_the_bed() {
        let (_dir, path) = z_level_file();
        let reader = OceanModelReader::from_file(path).unwrap();
        let u = reader.u().unwrap();
        // 20 m column: (0.75·10 + 0.5·10)/20; 60 m column: (0.75·10 + 0.25·40 + 0·10)/60
        assert!((u.get(0, 0) - 0.625).abs() < 1e-6, "{}", u.get(0, 0));
        assert!((u.get(0, 1) - 17.5 / 60.0).abs() < 1e-6, "{}", u.get(0, 1));
        assert!(
            reader
                .velocity_source()
                .unwrap()
                .contains("averaged over depth")
        );
        assert!((reader.temperature().unwrap().get(0, 2) - 12.0).abs() < 1e-6);
    }

    /// A ROMS-like s-level file: two layers (Vtransform 2, `hc` 0, the
    /// bottom layer 80 % of the column), `h` 100 m, `angle` 30°, along-grid
    /// `u` 1.0 at the bottom and 2.0 on top on u-points, `v` 0.
    fn s_level_file(angle: f64) -> (tempfile::TempDir, PathBuf) {
        write_file(|file| {
            file.add_dimension("s_rho", 2).unwrap();
            file.add_dimension("s_w", 3).unwrap();
            file.add_dimension("X_u", 2).unwrap();
            file.add_dimension("Y_v", 1).unwrap();
            for (name, dim, values) in [
                ("s_w", "s_w", [-1.0, -0.5, 0.0]),
                ("Cs_w", "s_w", [-1.0, -0.2, 0.0]),
            ] {
                let mut var = file.add_variable::<f64>(name, &[dim]).unwrap();
                var.put_values(&values, ..).unwrap();
            }
            let mut s_rho = file.add_variable::<f64>("s_rho", &["s_rho"]).unwrap();
            s_rho.put_values(&[-0.75, -0.25], ..).unwrap();
            let mut hc = file.add_variable::<f64>("hc", &[]).unwrap();
            hc.put_value(0.0, ..).unwrap();
            let mut vt = file.add_variable::<i32>("Vtransform", &[]).unwrap();
            vt.put_value(2, ..).unwrap();
            let mut h = file.add_variable::<f64>("h", &["Y", "X"]).unwrap();
            h.put_values(&[100.0; 6], ..).unwrap();
            let mut a = file.add_variable::<f64>("angle", &["Y", "X"]).unwrap();
            a.put_values(&[angle; 6], ..).unwrap();
            // Along-grid u: 1.0 in the bottom layer (80 % of the column), 2.0
            // on top: mean 1.2 m/s; v = 0
            let mut u = file
                .add_variable::<f64>("u", &["time", "s_rho", "Y", "X_u"])
                .unwrap();
            u.put_values(&[1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0], ..)
                .unwrap();
            let mut v = file
                .add_variable::<f64>("v", &["time", "s_rho", "Y_v", "X"])
                .unwrap();
            v.put_values(&[0.0; 6], ..).unwrap();
        })
    }

    /// ROMS s-levels (bottom first) with Vtransform 2, and grid-relative `u`
    /// on u-points rotated by `angle` to east/north.
    #[test]
    fn s_level_grid_relative_velocity() {
        let angle = 30f64.to_radians();
        let (_dir, path) = s_level_file(angle);
        let reader = OceanModelReader::from_file(path).unwrap();
        let (lat, lon) = (63.05, 8.1);
        let (e, n) = reader.get_state(lon, lat, 0).unwrap().velocity.unwrap();
        assert!((e - 1.2 * angle.cos()).abs() < 1e-6, "{e}");
        assert!((n - 1.2 * angle.sin()).abs() < 1e-6, "{n}");
        assert!(reader.velocity_source().unwrap().contains("angle"));
    }

    /// Profiles for 3D nesting keep the z-levels: from the surface down, NaN
    /// below the bed, interpolated in depth; the 2D fields are as without.
    #[test]
    fn z_level_profiles_keep_the_levels() {
        let (_dir, path) = z_level_file();
        let reader = OceanModelReader::from_file_with_profiles(&path).unwrap();
        let (u, _) = reader.velocity_profiles().unwrap();
        assert_eq!(u.levels(), &ProfileLevels::Depth(vec![0.0, 10.0, 50.0]));
        let column = u.column(0, 0);
        assert_eq!(&column[..2], &[1.0, 0.5]);
        assert!(column[2].is_nan());
        assert_eq!(u.value_at(0, 0, 5.0, 20.0), Some(0.75));
        // Below the bed of the 20 m column: its deepest value
        assert_eq!(u.value_at(0, 0, 30.0, 20.0), Some(0.5));
        assert_eq!(u.value_at(0, 1, 30.0, 60.0), Some(0.25));
        let t = reader.temperature_profile().unwrap();
        assert_eq!(t.column(0, 3), &[12.0, 8.0, 8.0]);
        assert!(reader.has_current_profiles());
        // No salinity in the file
        assert!(!reader.has_tracer_profiles());
        let plain = OceanModelReader::from_file(&path).unwrap();
        assert_eq!(plain.u(), reader.u());
        assert!(plain.velocity_profiles().is_none());
    }

    /// Profiles on ROMS s-levels are fractions of the column at the layer
    /// centres, surface first (the top layer is 20 % of the column: centre
    /// at 0.1, the bottom layer's at 0.6), rotated to east/north per level.
    #[test]
    fn s_level_profiles_are_fractions_of_the_column() {
        let angle = 30f64.to_radians();
        let (_dir, path) = s_level_file(angle);
        let reader = OceanModelReader::from_file_with_profiles(path).unwrap();
        let (u, v) = reader.velocity_profiles().unwrap();
        let ProfileLevels::Fraction(fractions) = u.levels() else {
            panic!("s-levels should be fractions");
        };
        assert!((fractions[0] - 0.1).abs() < 1e-12 && (fractions[1] - 0.6).abs() < 1e-12);
        let (e, n) = (u.column(0, 1), v.column(0, 1));
        for (l, along) in [(0, 2.0), (1, 1.0)] {
            assert!((e[l] as f64 - along * angle.cos()).abs() < 1e-6, "{e:?}");
            assert!((n[l] as f64 - along * angle.sin()).abs() < 1e-6, "{n:?}");
        }
    }

    /// A MEPS-like file: grid-relative 10 m wind with a singleton height
    /// dimension on a grid rotated 90°, and pressure in hPa (`units`).
    #[test]
    fn meps_like_grid_relative_wind() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("meps.nc");
        {
            let mut file = netcdf::create(&path).unwrap();
            for (name, len) in [("time", 2), ("height1", 1), ("y", 2), ("x", 2)] {
                file.add_dimension(name, len).unwrap();
            }
            // Grid x axis points north, y axis west
            let mut lat = file.add_variable::<f64>("latitude", &["y", "x"]).unwrap();
            lat.put_values(&[60.0, 60.1, 60.0, 60.1], ..).unwrap();
            let mut lon = file.add_variable::<f64>("longitude", &["y", "x"]).unwrap();
            lon.put_values(&[5.2, 5.2, 5.0, 5.0], ..).unwrap();
            let mut time = file.add_variable::<f64>("time", &["time"]).unwrap();
            time.put_attribute("units", "seconds since 1970-01-01 00:00:00 +00:00")
                .unwrap();
            time.put_values(&[0.0, 3600.0], ..).unwrap();
            let dims = &["time", "height1", "y", "x"];
            let mut xw = file.add_variable::<f32>("x_wind_10m", dims).unwrap();
            xw.put_attribute("standard_name", "x_wind").unwrap();
            xw.put_values(&[5.0_f32; 8], ..).unwrap();
            let mut yw = file.add_variable::<f32>("y_wind_10m", dims).unwrap();
            yw.put_values(&[0.0_f32; 8], ..).unwrap();
            let mut p = file
                .add_variable::<f32>("air_pressure_at_sea_level", dims)
                .unwrap();
            p.put_attribute("units", "hPa").unwrap();
            p.put_values(
                &[
                    1000.0_f32, 1000.0, 1010.0, 1010.0, 990.0, 990.0, 1000.0, 1000.0,
                ],
                ..,
            )
            .unwrap();
        }
        let reader = AtmosphereReader::from_file(&path).unwrap();
        let state = reader.get(5.1, 60.05, 1800.0).unwrap();
        let (u, v) = state.wind.unwrap();
        assert!(u.abs() < 0.05 && (v - 5.0).abs() < 1e-3, "({u}, {v})");
        assert!((state.pressure.unwrap() - 100_000.0).abs() < 1e-3);
        assert!(reader.source.contains("rotated"));
        let joined = AtmosphereReader::from_files(&[&path]).unwrap();
        assert_eq!(joined.time, reader.time);
        assert!(AtmosphereReader::from_files(&[&path, &path]).is_err());
    }

    #[test]
    fn speed_and_direction_become_components() {
        let (_dir, path) = write_file(|file| {
            let dims = &["time", "Y", "X"];
            let mut s = file.add_variable::<f32>("wind_speed_10m", dims).unwrap();
            s.put_values(&[10.0_f32; 6], ..).unwrap();
            let mut d = file
                .add_variable::<f32>("wind_direction_10m", dims)
                .unwrap();
            d.put_values(&[270.0_f32; 6], ..).unwrap();
        });
        let reader = AtmosphereReader::from_file(path).unwrap();
        let (u, v) = reader.get(8.1, 63.05, 0.0).unwrap().wind.unwrap();
        assert!((u - 10.0).abs() < 1e-5 && v.abs() < 1e-5);
        assert!(!reader.has_pressure());
    }
}

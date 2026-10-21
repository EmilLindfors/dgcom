//! OPeNDAP helpers shared by the data-fetching examples (`#[path]`-included).

// Not every example uses every helper
#![allow(dead_code)]

use std::error::Error;
use std::time::Duration;

use netcdf::{AttributeValue, Extent, Variable};

/// Extent of `count` values from `start`.
pub fn slice(start: usize, count: usize) -> Extent {
    Extent::SliceCount {
        start,
        count,
        stride: 1,
    }
}

/// A numeric attribute.
pub fn attribute_f64(var: &Variable, name: &str) -> Option<f64> {
    match var.attribute_value(name)?.ok()? {
        AttributeValue::Float(v) => Some(v as f64),
        AttributeValue::Double(v) => Some(v),
        AttributeValue::Short(v) => Some(v as f64),
        AttributeValue::Int(v) => Some(v as f64),
        _ => None,
    }
}

/// A string attribute.
pub fn attribute_string(var: &Variable, name: &str) -> Option<String> {
    match var.attribute_value(name)?.ok()? {
        AttributeValue::Str(s) => Some(s),
        _ => None,
    }
}

/// Retry a DAP request: THREDDS answers 503 under load.
pub fn retry<T, E: std::fmt::Display>(
    what: &str,
    mut f: impl FnMut() -> Result<T, E>,
) -> Result<T, Box<dyn Error>> {
    for attempt in 1..=5 {
        match f() {
            Ok(v) => return Ok(v),
            Err(e) if attempt < 5 => {
                eprintln!("  {what}: {e}; retrying in {} s", 10 * attempt);
                std::thread::sleep(Duration::from_secs(10 * attempt));
            }
            Err(e) => return Err(format!("{what}: {e}").into()),
        }
    }
    unreachable!()
}

/// A rectangular window `[y0, y0 + ny) × [x0, x0 + nx)` of a 2D grid.
#[derive(Clone, Copy, Debug)]
pub struct Window {
    pub y0: usize,
    pub x0: usize,
    pub ny: usize,
    pub nx: usize,
}

impl Window {
    /// Extents of the window for a `[.., Y, X]` variable, after `leading`.
    pub fn extents(&self, leading: &[Extent]) -> Vec<Extent> {
        let mut e = leading.to_vec();
        e.push(slice(self.y0, self.ny));
        e.push(slice(self.x0, self.nx));
        e
    }

    /// Number of points.
    pub fn len(&self) -> usize {
        self.ny * self.nx
    }
}

/// The window of the 2D grid with coordinates `lon_name`/`lat_name` that
/// covers `[min_lon, max_lon] × [min_lat, max_lat]` plus `margin` degrees,
/// found on a copy of the grid thinned by `stride` (and widened by one
/// stride so no covering cell is missed).
pub fn locate_window(
    file: &netcdf::File,
    lon_name: &str,
    lat_name: &str,
    [min_lon, min_lat, max_lon, max_lat]: [f64; 4],
    margin: f64,
    stride: usize,
) -> Result<Window, Box<dyn Error>> {
    let var = |name: &str| file.variable(name).ok_or(format!("no variable {name}"));
    let (ny_all, nx_all) = {
        let lon = var(lon_name)?;
        let dims = lon.dimensions();
        (dims[0].len(), dims[1].len())
    };
    let coarse = |name: &str| -> Result<Vec<f64>, Box<dyn Error>> {
        let v = var(name)?;
        retry(name, || {
            v.get_values([
                Extent::SliceCount {
                    start: 0,
                    count: ny_all.div_ceil(stride),
                    stride: stride as isize,
                },
                Extent::SliceCount {
                    start: 0,
                    count: nx_all.div_ceil(stride),
                    stride: stride as isize,
                },
            ])
        })
    };
    let (clon, clat) = (coarse(lon_name)?, coarse(lat_name)?);
    let cnx = nx_all.div_ceil(stride);
    let (mut y_range, mut x_range) = ((usize::MAX, 0), (usize::MAX, 0));
    for (k, (&lo, &la)) in clon.iter().zip(&clat).enumerate() {
        if (min_lon - margin..=max_lon + margin).contains(&lo)
            && (min_lat - margin..=max_lat + margin).contains(&la)
        {
            let (y, x) = ((k / cnx) * stride, (k % cnx) * stride);
            y_range = (y_range.0.min(y), y_range.1.max(y));
            x_range = (x_range.0.min(x), x_range.1.max(x));
        }
    }
    if y_range.0 == usize::MAX {
        return Err("the box is outside the grid".into());
    }
    let y0 = y_range.0.saturating_sub(stride);
    let x0 = x_range.0.saturating_sub(stride);
    Ok(Window {
        y0,
        x0,
        ny: (y_range.1 + stride + 1).min(ny_all) - y0,
        nx: (x_range.1 + stride + 1).min(nx_all) - x0,
    })
}

/// Parse `key=value` arguments.
pub fn arguments() -> std::collections::HashMap<String, String> {
    std::env::args()
        .skip(1)
        .filter_map(|a| a.split_once('=').map(|(k, v)| (k.into(), v.into())))
        .collect()
}

/// `min_lon,min_lat,max_lon,max_lat`.
pub fn parse_bbox(text: &str) -> Result<[f64; 4], String> {
    let values: Vec<f64> = text
        .split(',')
        .map(|v| v.trim().parse().map_err(|_| format!("bad bbox value {v}")))
        .collect::<Result<_, _>>()?;
    values
        .try_into()
        .map_err(|_| "bbox=min_lon,min_lat,max_lon,max_lat".to_string())
}

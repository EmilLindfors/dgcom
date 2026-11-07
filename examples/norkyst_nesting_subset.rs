//! NorKyst-800 nesting data for a model domain (OPeNDAP → NetCDF).
//!
//! Cuts the NorKyst-800 window around a lon/lat box out of MET Norway's
//! THREDDS aggregation and writes what a 2D child needs at its open
//! boundary, hourly, as a small CF NetCDF file that
//! `OceanModelReader::from_file` reads directly (`froya_real_data
//! norkyst=<file>`):
//!
//! - `zeta` (sea surface height) and `h` (still-water depth),
//! - `ubar_eastward`, `vbar_northward`: the z-level `u_eastward` and
//!   `v_northward` averaged over the water column
//!   ([`depth_average_z`](dgcom::io::depth_average_z), trapezoid rule down
//!   to `h`). NorKyst's aggregation has no barotropic velocity, and its
//!   levels end at 300 m: deeper columns hold the 300 m value to the bed.
//! - with `wind=1` also `Uwind_eastward`, `Vwind_northward`, the 10 m wind
//!   NorKyst was forced with (`AtmosphereReader` reads it as `met=`; no
//!   pressure).
//! - with `profiles=1` also the 3D fields for nesting a 3D child (P4.2,
//!   `Nesting3D`): `u_eastward`, `v_northward`, `temperature` and `salinity`
//!   on NorKyst's z-levels (`depth`, positive down, from the surface), which
//!   `OceanModelReader::from_file_with_profiles` reads as profiles. A day of a
//!   90 × 90 window on 16 levels is ≈ 50 MB more.
//!
//! The window extends `margin` degrees beyond the box, so that a relaxation
//! band and the bilinear stencils of boundary nodes near the edge stay
//! inside. Land is NaN. One DAP request per day and variable: THREDDS pays
//! per stored chunk, not per cell.
//!
//! ## Run
//!
//! ```bash
//! cargo run --release --example norkyst_nesting_subset -- \
//!     [bbox=8.0,63.6,9.2,64.0] [start=2025-06-15] [hours=72] [margin=0.1] \
//!     [out=data/froya_norkyst.nc] [wind=1] [profiles=1] [url=<OPeNDAP URL>]
//! ```
//!
//! Requires the `netcdf` feature (default) with a DAP-enabled netCDF-C.

#[cfg(feature = "netcdf")]
#[path = "common/dap.rs"]
mod dap;

#[cfg(feature = "netcdf")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    app::run()
}

#[cfg(not(feature = "netcdf"))]
fn main() {
    eprintln!("This example requires the `netcdf` feature.");
}

#[cfg(feature = "netcdf")]
mod app {
    use std::error::Error;
    use std::path::PathBuf;
    use std::time::Instant;

    use dgcom::io::depth_average_z;
    use dgcom::time::ModelClock;
    use netcdf::{Extent, Variable};

    use super::dap::{
        arguments, attribute_f64, attribute_string, locate_window, parse_bbox, retry, slice,
    };

    const URL: &str = "https://thredds.met.no/thredds/dodsC/fou-hi/norkystv3_800m_m00_be";
    /// Coarse stride for locating the domain in the NorKyst grid
    const STRIDE: usize = 8;
    /// Packed-integer fill value of NorKyst's Int16 variables
    const FILL: i16 = -32767;
    /// Snapshots per DAP request
    const CHUNK: usize = 24;

    /// A packed Int16 variable and its unpacking.
    struct Packed<'f> {
        var: Variable<'f>,
        scale: f64,
        offset: f64,
    }

    impl<'f> Packed<'f> {
        fn new(file: &'f netcdf::File, name: &str) -> Result<Self, Box<dyn Error>> {
            let var = file.variable(name).ok_or(format!("no variable {name}"))?;
            Ok(Self {
                scale: attribute_f64(&var, "scale_factor").unwrap_or(1.0),
                offset: attribute_f64(&var, "add_offset").unwrap_or(0.0),
                var,
            })
        }

        fn read(&self, extents: Vec<Extent>) -> Result<Vec<Option<f64>>, Box<dyn Error>> {
            let raw: Vec<i16> = retry(&self.var.name(), || self.var.get_values(extents.clone()))?;
            Ok(raw
                .into_iter()
                .map(|v| (v != FILL).then_some(self.offset + self.scale * v as f64))
                .collect())
        }
    }

    fn to_f32(values: impl Iterator<Item = Option<f64>>) -> Vec<f32> {
        values.map(|v| v.map_or(f32::NAN, |v| v as f32)).collect()
    }

    pub fn run() -> Result<(), Box<dyn Error>> {
        let args = arguments();
        let bbox = parse_bbox(args.get("bbox").map_or("8.0,63.6,9.2,64.0", String::as_str))?;
        let start = args.get("start").map_or("2025-06-15", String::as_str);
        let num = |key: &str, default: f64| -> Result<f64, String> {
            args.get(key).map_or(Ok(default), |v| {
                v.parse().map_err(|_| format!("bad {key}={v}"))
            })
        };
        let hours = num("hours", 72.0)? as usize;
        let margin = num("margin", 0.1)?;
        let wind = num("wind", 0.0)? != 0.0;
        let profiles = num("profiles", 0.0)? != 0.0;
        let out = args
            .get("out")
            .map_or("data/froya_norkyst.nc".into(), PathBuf::from);
        let url = args.get("url").map_or(URL, String::as_str);
        let start_unix = ModelClock::parse(start)?.epoch_unix;
        println!("NorKyst-800 nesting data for {bbox:?} (+{margin}°), {hours} h from {start}");

        let file = retry("open", || netcdf::open(url))?;
        let var = |name: &str| file.variable(name).ok_or(format!("no variable {name}"));

        // Hourly snapshots [t0, t0 + hours]
        let time_var = var("time")?;
        let units = attribute_string(&time_var, "units").unwrap_or_default();
        if !units.starts_with("seconds since 1970-01-01") {
            return Err(format!("unexpected time units {units:?}").into());
        }
        let time: Vec<f64> = retry("time", || time_var.get_values(..))?;
        let t0 = time
            .iter()
            .position(|&t| t >= start_unix)
            .ok_or("start is after the end of the aggregation")?;
        let n = hours + 1;
        if t0 + n > time.len() {
            return Err("the record runs past the end of the aggregation".into());
        }
        let times = &time[t0..t0 + n];
        if times.windows(2).any(|w| (w[1] - w[0] - 3600.0).abs() > 1.0) {
            return Err("the record is not hourly".into());
        }

        let win = locate_window(&file, "lon", "lat", bbox, margin, STRIDE)?;
        println!(
            "  grid window: Y {}+{}, X {}+{}",
            win.y0, win.ny, win.x0, win.nx
        );
        let read2d = |name: &str| -> Result<Vec<f64>, Box<dyn Error>> {
            let v = var(name)?;
            retry(name, || v.get_values(win.extents(&[])))
        };
        let (lon, lat, h) = (read2d("lon")?, read2d("lat")?, read2d("h")?);
        let levels: Vec<f64> = retry("depth", || var("depth").unwrap().get_values(..))?;
        let nz = levels.len();
        let zeta = Packed::new(&file, "zeta")?;
        let (u3, v3) = (
            Packed::new(&file, "u_eastward")?,
            Packed::new(&file, "v_northward")?,
        );
        let tracers = if profiles {
            Some((
                Packed::new(&file, "temperature")?,
                Packed::new(&file, "salinity")?,
            ))
        } else {
            None
        };
        let winds = if wind {
            Some((
                Packed::new(&file, "Uwind_eastward")?,
                Packed::new(&file, "Vwind_northward")?,
            ))
        } else {
            None
        };

        // Output
        if let Some(dir) = out.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let mut nc = netcdf::create(&out)?;
        nc.add_attribute("title", "NorKyst-800 nesting data")?;
        nc.add_attribute(
            "source",
            format!(
                "{url}, window Y {}+{}, X {}+{}",
                win.y0, win.ny, win.x0, win.nx
            ),
        )?;
        nc.add_attribute(
            "comment",
            "ubar/vbar: z-level u_eastward/v_northward averaged over the water column (levels end at 300 m)",
        )?;
        nc.add_dimension("time", n)?;
        nc.add_dimension("Y", win.ny)?;
        nc.add_dimension("X", win.nx)?;
        if profiles {
            nc.add_dimension("depth", nz)?;
            let mut dv = nc.add_variable::<f64>("depth", &["depth"])?;
            dv.put_attribute("units", "m")?;
            dv.put_attribute("positive", "down")?;
            dv.put_attribute("standard_name", "depth")?;
            dv.put_values(&levels, ..)?;
        }
        let mut tv = nc.add_variable::<f64>("time", &["time"])?;
        tv.put_attribute("units", "seconds since 1970-01-01 00:00:00")?;
        tv.put_attribute("calendar", "standard")?;
        tv.put_values(times, ..)?;
        for (name, values, units) in [
            ("lon", &lon, "degrees_east"),
            ("lat", &lat, "degrees_north"),
            ("h", &h, "m"),
        ] {
            let mut v = nc.add_variable::<f64>(name, &["Y", "X"])?;
            v.put_attribute("units", units)?;
            v.put_values(values, ..)?;
        }
        let mut fields = vec![
            ("zeta", "m", "sea_surface_height_above_geoid"),
            (
                "ubar_eastward",
                "m s-1",
                "barotropic_eastward_sea_water_velocity",
            ),
            (
                "vbar_northward",
                "m s-1",
                "barotropic_northward_sea_water_velocity",
            ),
        ];
        if wind {
            fields.push(("Uwind_eastward", "m s-1", "eastward_wind"));
            fields.push(("Vwind_northward", "m s-1", "northward_wind"));
        }
        for (name, units, standard_name) in &fields {
            let mut v = nc.add_variable::<f32>(name, &["time", "Y", "X"])?;
            v.put_attribute("units", *units)?;
            v.put_attribute("standard_name", *standard_name)?;
        }
        const PROFILES: [(&str, &str, &str); 4] = [
            ("u_eastward", "m s-1", "eastward_sea_water_velocity"),
            ("v_northward", "m s-1", "northward_sea_water_velocity"),
            ("temperature", "degC", "sea_water_temperature"),
            ("salinity", "1e-3", "sea_water_salinity"),
        ];
        if profiles {
            for (name, units, standard_name) in PROFILES {
                let mut v = nc.add_variable::<f32>(name, &["time", "depth", "Y", "X"])?;
                v.put_attribute("units", units)?;
                v.put_attribute("standard_name", standard_name)?;
            }
        }

        let plane = win.len();
        let started = Instant::now();
        for c0 in (0..n).step_by(CHUNK) {
            let count = CHUNK.min(n - c0);
            let ts = t0 + c0;
            let zeta_values = zeta.read(win.extents(&[slice(ts, count)]))?;
            let column = |p: &Packed| p.read(win.extents(&[slice(ts, count), slice(0, nz)]));
            let (u, v) = (column(&u3)?, column(&v3)?);
            let mean = |data: &[Option<f64>], hour: usize, k: usize| -> Option<f64> {
                let profile: Vec<Option<f64>> =
                    (0..nz).map(|l| data[(hour * nz + l) * plane + k]).collect();
                depth_average_z(&levels, &profile, h[k])
            };
            let (mut ubar, mut vbar) = (
                Vec::with_capacity(count * plane),
                Vec::with_capacity(count * plane),
            );
            for hour in 0..count {
                for k in 0..plane {
                    let wet = zeta_values[hour * plane + k].is_some();
                    ubar.push(if wet { mean(&u, hour, k) } else { None });
                    vbar.push(if wet { mean(&v, hour, k) } else { None });
                }
            }
            let extents = [slice(c0, count), slice(0, win.ny), slice(0, win.nx)];
            let mut put = |name: &str, values: Vec<f32>| -> Result<(), Box<dyn Error>> {
                nc.variable_mut(name)
                    .ok_or(format!("no output variable {name}"))?
                    .put_values(&values, extents)?;
                Ok(())
            };
            put("zeta", to_f32(zeta_values.into_iter()))?;
            put("ubar_eastward", to_f32(ubar.into_iter()))?;
            put("vbar_northward", to_f32(vbar.into_iter()))?;
            if let Some((uw, vw)) = &winds {
                put(
                    "Uwind_eastward",
                    to_f32(uw.read(win.extents(&[slice(ts, count)]))?.into_iter()),
                )?;
                put(
                    "Vwind_northward",
                    to_f32(vw.read(win.extents(&[slice(ts, count)]))?.into_iter()),
                )?;
            }
            if let Some((temperature, salinity)) = &tracers {
                // [hour][level][Y][X], as read
                let volume = [
                    slice(c0, count),
                    slice(0, nz),
                    slice(0, win.ny),
                    slice(0, win.nx),
                ];
                for (name, values) in [
                    ("u_eastward", u.clone()),
                    ("v_northward", v.clone()),
                    ("temperature", column(temperature)?),
                    ("salinity", column(salinity)?),
                ] {
                    nc.variable_mut(name)
                        .ok_or(format!("no output variable {name}"))?
                        .put_values(&to_f32(values.into_iter()), volume)?;
                }
            }
            println!(
                "  hours {c0:3}–{:3} of {hours}: {:.0} s elapsed",
                c0 + count - 1,
                started.elapsed().as_secs_f64()
            );
        }
        println!(
            "Wrote {} × {} points × {n} hours to {}",
            win.ny,
            win.nx,
            out.display()
        );
        Ok(())
    }
}

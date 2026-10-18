//! Atmospheric forcing for a model domain from MET Nordic (OPeNDAP → NetCDF).
//!
//! MET Norway's MET Nordic analysis (1 km Lambert grid over the Nordic
//! countries, MEPS corrected with observations) is archived as one file per
//! hour. This cuts the window around a lon/lat box out of each hour's file
//! and writes one CF NetCDF file with the 10 m wind (`wind_speed_10m`,
//! `wind_direction_10m`: the direction it blows *from*), the air pressure
//! at sea level, and what the surface heat and freshwater fluxes need: the
//! 2 m air temperature and relative humidity, the downwelling shortwave and
//! longwave radiation and the precipitation. `AtmosphereReader::from_file`
//! reads it directly (`froya_real_data met=<file>`).
//!
//! The analysis stores radiation and precipitation as integrals over the
//! hour ending at the file's time. They are written as fluxes at the
//! snapshot times (W/m², kg m⁻² s⁻¹): the mean of the two hours around each
//! snapshot, so that linear interpolation in time is centred and keeps the
//! daily totals. A run of `hours` therefore reads `hours + 2` files, the
//! hour after the last snapshot included.
//!
//! MEPS or AROME-Arctic forecast files (grid-relative `x_wind_10m`,
//! `y_wind_10m`) can be read by `AtmosphereReader` as they are, e.g. a
//! THREDDS NCSS subset; consecutive runs are joined with
//! `AtmosphereReader::from_files`.
//!
//! ## Run
//!
//! ```bash
//! cargo run --release --example met_forcing_subset -- \
//!     [bbox=8.0,63.6,9.2,64.0] [start=2025-06-15] [hours=72] [margin=0.1] \
//!     [out=data/froya_met.nc]
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

    use dg_rs::time::ModelClock;

    use super::dap::{
        Window, arguments, attribute_string, locate_window, parse_bbox, retry, slice,
    };

    /// Hourly MET Nordic analysis files
    const ARCHIVE: &str = "https://thredds.met.no/thredds/dodsC/metpparchive";
    /// Coarse stride for locating the domain in the 1 km grid
    const STRIDE: usize = 16;
    /// Fields at the analysis time: `(name, units, standard name)`
    const FIELDS: [(&str, &str, &str); 5] = [
        ("wind_speed_10m", "m/s", "wind_speed"),
        ("wind_direction_10m", "degree", "wind_from_direction"),
        (
            "air_pressure_at_sea_level",
            "Pa",
            "air_pressure_at_sea_level",
        ),
        ("air_temperature_2m", "K", "air_temperature"),
        ("relative_humidity_2m", "1", "relative_humidity"),
    ];
    /// Integrals over the hour before the analysis time: `(name, units,
    /// written name, written units, written standard name)`
    const INTEGRALS: [(&str, &str, &str, &str); 3] = [
        (
            "integral_of_surface_downwelling_shortwave_flux_in_air_wrt_time",
            "W s/m^2",
            "surface_downwelling_shortwave_flux_in_air",
            "W m-2",
        ),
        (
            "integral_of_surface_downwelling_longwave_flux_in_air_wrt_time",
            "J/m^2",
            "surface_downwelling_longwave_flux_in_air",
            "W m-2",
        ),
        (
            "precipitation_amount",
            "kg/m^2",
            "precipitation_flux",
            "kg m-2 s-1",
        ),
    ];

    /// The analysis file of the hour starting at Unix time `t`.
    fn url(t: f64) -> String {
        // "YYYY-MM-DD HH:MM:SS"
        let s = ModelClock::new(t).format(0.0);
        let (y, m, d, h) = (&s[0..4], &s[5..7], &s[8..10], &s[11..13]);
        format!("{ARCHIVE}/{y}/{m}/{d}/met_analysis_1_0km_nordic_{y}{m}{d}T{h}Z.nc")
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
        let out = args
            .get("out")
            .map_or("data/froya_met.nc".into(), PathBuf::from);
        let t0 = ModelClock::parse(start)?.epoch_unix;
        if t0 % 3600.0 != 0.0 {
            return Err("start must be on the hour".into());
        }
        let n = hours + 1;
        println!("MET Nordic analysis for {bbox:?} (+{margin}°), {n} hourly files from {start}");

        // Window from the first file; the grid is the same in every file
        let first = retry("open", || netcdf::open(url(t0)))?;
        let win: Window = locate_window(&first, "longitude", "latitude", bbox, margin, STRIDE)?;
        println!(
            "  grid window: y {}+{}, x {}+{}",
            win.y0, win.ny, win.x0, win.nx
        );
        let read2d = |name: &str| -> Result<Vec<f64>, Box<dyn Error>> {
            let v = first.variable(name).ok_or(format!("no variable {name}"))?;
            retry(name, || v.get_values(win.extents(&[])))
        };
        let (lon, lat) = (read2d("longitude")?, read2d("latitude")?);
        let expected = FIELDS
            .iter()
            .map(|&(name, units, _)| (name, units))
            .chain(INTEGRALS.iter().map(|&(name, units, ..)| (name, units)));
        for (name, units) in expected {
            let var = first.variable(name).ok_or(format!("no variable {name}"))?;
            let found = attribute_string(&var, "units").unwrap_or_default();
            if found != units {
                return Err(format!("{name}: units {found:?}, expected {units:?}").into());
            }
        }
        drop(first);

        if let Some(dir) = out.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let mut nc = netcdf::create(&out)?;
        nc.add_attribute("title", "MET Nordic analysis subset")?;
        nc.add_attribute(
            "source",
            format!(
                "{ARCHIVE} met_analysis_1_0km_nordic, window y {}+{}, x {}+{}",
                win.y0, win.ny, win.x0, win.nx
            ),
        )?;
        nc.add_dimension("time", n)?;
        nc.add_dimension("y", win.ny)?;
        nc.add_dimension("x", win.nx)?;
        let mut tv = nc.add_variable::<f64>("time", &["time"])?;
        tv.put_attribute("units", "seconds since 1970-01-01 00:00:00")?;
        let times: Vec<f64> = (0..n).map(|h| t0 + 3600.0 * h as f64).collect();
        tv.put_values(&times, ..)?;
        for (name, values, units) in [
            ("longitude", &lon, "degree_east"),
            ("latitude", &lat, "degree_north"),
        ] {
            let mut v = nc.add_variable::<f64>(name, &["y", "x"])?;
            v.put_attribute("units", units)?;
            v.put_values(values, ..)?;
        }
        for (name, units, standard_name) in FIELDS {
            let mut v = nc.add_variable::<f32>(name, &["time", "y", "x"])?;
            v.put_attribute("units", units)?;
            v.put_attribute("standard_name", standard_name)?;
        }
        for (_, _, name, units) in INTEGRALS {
            let mut v = nc.add_variable::<f32>(name, &["time", "y", "x"])?;
            v.put_attribute("units", units)?;
            v.put_attribute("standard_name", name)?;
            v.put_attribute(
                "cell_methods",
                "time: mean (interval: 2 hours, centred on the snapshot)",
            )?;
        }

        let started = Instant::now();
        let window = win.extents(&[slice(0, 1)]);
        // The previous file's hourly integrals, ending at the snapshot before
        let mut before: Option<Vec<Vec<f32>>> = None;
        // The first snapshot through the hour after the last
        for file_hour in 0..=n {
            let t = t0 + 3600.0 * file_hour as f64;
            let file = retry("open", || netcdf::open(url(t)))?;
            let at = |hour: usize| [slice(hour, 1), slice(0, win.ny), slice(0, win.nx)];
            if file_hour < n {
                for (name, _, _) in FIELDS {
                    let var = file.variable(name).ok_or(format!("no variable {name}"))?;
                    let values: Vec<f32> = retry(name, || var.get_values(window.clone()))?;
                    nc.variable_mut(name)
                        .expect("defined above")
                        .put_values(&values, at(file_hour))?;
                }
            }
            // This file's integrals end at its time, so the hour before it
            // and the hour after the snapshot before it
            let integrals = INTEGRALS
                .iter()
                .map(|&(name, ..)| {
                    let var = file.variable(name).ok_or(format!("no variable {name}"))?;
                    retry(name, || var.get_values::<f32, _>(window.clone()))
                })
                .collect::<Result<Vec<_>, _>>()?;
            if file_hour == 0 {
                // The hour ending at the first snapshot needs the file of
                // the snapshot after it as well
                before = Some(integrals);
                continue;
            }
            let hour = file_hour - 1;
            let previous = before.replace(integrals);
            let (previous, next) = (previous.expect("set above"), before.as_ref().expect("set"));
            for ((_, _, name, _), (a, b)) in INTEGRALS.iter().zip(previous.iter().zip(next)) {
                let mean: Vec<f32> = a.iter().zip(b).map(|(a, b)| (a + b) / 7200.0).collect();
                nc.variable_mut(name)
                    .expect("defined above")
                    .put_values(&mean, at(hour))?;
            }
            if hour % 12 == 0 || hour + 1 == n {
                println!(
                    "  {} ({}/{n}): {:.0} s elapsed",
                    ModelClock::new(t).format(0.0),
                    hour + 1,
                    started.elapsed().as_secs_f64()
                );
            }
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

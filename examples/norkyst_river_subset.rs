//! NorKyst-800's rivers for a model domain (OPeNDAP → NetCDF).
//!
//! Cuts the rivers of a lon/lat box out of NorKyst-800 v3's river forcing
//! (1801 rivers from NVE's runoff data, daily since 2011, on MET Norway's
//! THREDDS) and writes them, with their mouths as longitude and latitude,
//! as a small river file that `RiverForcing::from_file` reads directly
//! (`froya_real_data rivers=<file>`):
//!
//! - the mouths are the midpoints of the coastal faces the rivers enter
//!   NorKyst through (`roms_source_position`), from the grid's ρ-points;
//! - the discharge (m³/s) and the river water's temperature and salinity
//!   (NorKyst's levels weighted by the river's `river_Vshape`) at the days
//!   covering `start` to `start + days`.
//!
//! The box extends `margin` degrees beyond `bbox`, so that rivers whose
//! NorKyst mouth lies a little outside the domain are kept for snapping.
//!
//! ## Run
//!
//! ```bash
//! cargo run --release --example norkyst_river_subset -- \
//!     [bbox=8.0,63.6,9.2,64.0] [start=2025-06-15] [days=3] [margin=0.05] \
//!     [out=data/froya_rivers.nc] [url=<OPeNDAP river file>] [grid=<OPeNDAP grid file>]
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

    use dg_rs::io::{GeoBoundingBox, RiverForcing, RomsRhoPoints};
    use dg_rs::time::ModelClock;

    use super::dap::{arguments, locate_window, parse_bbox, retry};

    const RIVERS: &str =
        "https://thredds.met.no/thredds/dodsC/romshindcast/norkyst_v3/forcing/river.nc";
    const GRID: &str = "https://thredds.met.no/thredds/dodsC/fou-hi/norkystv3_his_files/static_nc_files/norkyst_grd_v31.nc";
    /// Coarse stride for locating the domain in the NorKyst grid
    const STRIDE: usize = 8;
    const DAY: f64 = 86_400.0;

    pub fn run() -> Result<(), Box<dyn Error>> {
        let args = arguments();
        let bbox = parse_bbox(args.get("bbox").map_or("8.0,63.6,9.2,64.0", String::as_str))?;
        let start = args.get("start").map_or("2025-06-15", String::as_str);
        let num = |key: &str, default: f64| -> Result<f64, String> {
            args.get(key).map_or(Ok(default), |v| {
                v.parse().map_err(|_| format!("bad {key}={v}"))
            })
        };
        let days = num("days", 3.0)?;
        let margin = num("margin", 0.05)?;
        let out = args
            .get("out")
            .map_or("data/froya_rivers.nc".into(), PathBuf::from);
        let url = args.get("url").map_or(RIVERS, String::as_str);
        let grid_url = args.get("grid").map_or(GRID, String::as_str);
        let start_unix = ModelClock::parse(start)?.epoch_unix;
        println!("NorKyst-800 rivers in {bbox:?} (+{margin}°), {days} days from {start}");

        // The grid's ρ-points around the box
        let clock = Instant::now();
        let grid = retry("open grid", || netcdf::open(grid_url))?;
        let window = locate_window(&grid, "lon_rho", "lat_rho", bbox, margin, STRIDE)?;
        let read = |name: &str| -> Result<Vec<f64>, Box<dyn Error>> {
            let var = grid.variable(name).ok_or(format!("no variable {name}"))?;
            retry(name, || var.get_values(window.extents(&[])))
        };
        let (lon, lat) = (read("lon_rho")?, read("lat_rho")?);
        println!(
            "  Grid window: η {}–{}, ξ {}–{} ({:.1} s)",
            window.y0,
            window.y0 + window.ny - 1,
            window.x0,
            window.x0 + window.nx - 1,
            clock.elapsed().as_secs_f64()
        );
        let points = RomsRhoPoints {
            lon: &lon,
            lat: &lat,
            n_xi: window.nx,
            n_eta: window.ny,
            xi0: window.x0,
            eta0: window.y0,
        };

        // The rivers in the box, at the days covering the run
        let clock = Instant::now();
        let [min_lon, min_lat, max_lon, max_lat] = bbox;
        let selection = GeoBoundingBox::new(
            min_lon - margin,
            min_lat - margin,
            max_lon + margin,
            max_lat + margin,
        );
        let window = (start_unix, start_unix + days * DAY);
        let rivers = retry("rivers", || {
            RiverForcing::from_roms_file(url, &points, Some(&selection), Some(window))
        })?;
        println!(
            "  {} rivers, {} days ({:.1} s)",
            rivers.len(),
            rivers.time().len(),
            clock.elapsed().as_secs_f64()
        );

        // The largest ones
        let mut order: Vec<usize> = (0..rivers.len()).collect();
        order.sort_by(|&a, &b| {
            rivers
                .mean_discharge(b)
                .total_cmp(&rivers.mean_discharge(a))
        });
        let total: f64 = order.iter().map(|&i| rivers.mean_discharge(i)).sum();
        println!("  Total mean discharge {total:.1} m³/s; the largest:");
        for &i in order.iter().take(10) {
            let m = &rivers.mouths()[i];
            println!(
                "    river {:>5} at {:.4}°E {:.4}°N: {:7.2} m³/s, {:.1} °C, S {:.1}",
                m.id,
                m.lon,
                m.lat,
                rivers.mean_discharge(i),
                rivers.temperature(0, i),
                rivers.salinity(0, i)
            );
        }

        if let Some(dir) = out.parent() {
            std::fs::create_dir_all(dir)?;
        }
        rivers.write(&out)?;
        println!("Wrote {}", out.display());
        Ok(())
    }
}

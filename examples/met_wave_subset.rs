//! Wave spectra for a model domain from MET Norway's MyWave WAM 800 m
//! (OPeNDAP → NetCDF), for the open boundary of the spectral wave model
//! (TODO F.4 forcing; `froya_real_data wave_spectra=<file>`).
//!
//! MET's coastal wave model runs at 800 m in five areas (Finnmark `c0`, Nord-
//! Norge `c1`, Midt-Norge `c2`, Vestlandet `c3`, Skagerrak `c4`), twice a day
//! (00 and 12 UTC), and publishes the full 2D spectrum (36 frequencies from
//! 0.035 Hz, 36 directions) hourly over the 72 h forecast at a few dozen
//! points per area, with H_s, T_p and its wind at them. This keeps the points
//! of the latest run inside a lon/lat box (with a margin) and writes them in
//! MET's own layout, which `io::WaveSpectraFile` reads.
//!
//! Only the latest run of each cycle is on the server: fetch the forecast
//! you want to run while it is there.
//!
//! ## Run
//!
//! ```bash
//! cargo run --release --example met_wave_subset -- \
//!     [bbox=8.0,63.6,9.2,64.0] [margin=0.1] [area=mywavewam800m] [run=00|12] \
//!     [url=<OPeNDAP URL of a SPC file>] [out=data/froya_wave_spectra.nc]
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

    use dgcom::io::WaveSpectraFile;
    use dgcom::time::ModelClock;

    use super::dap::{arguments, parse_bbox};

    const SERVER: &str = "https://thredds.met.no/thredds/dodsC/fou-hi";

    /// The SPC file of each area's latest run (`run` 00 or 12).
    fn latest(area: &str, run: &str) -> Result<String, String> {
        let code = match area {
            "mywavewam800f" => "c0",
            "mywavewam800n" => "c1",
            "mywavewam800m" => "c2",
            "mywavewam800v" => "c3",
            "mywavewam800s" => "c4",
            other => return Err(format!("unknown area {other}")),
        };
        Ok(format!("{SERVER}/{area}/MyWave_wam800_{code}SPC{run}.nc"))
    }

    pub fn run() -> Result<(), Box<dyn Error>> {
        let args = arguments();
        let bbox = parse_bbox(args.get("bbox").map_or("8.0,63.6,9.2,64.0", String::as_str))?;
        let margin: f64 = args.get("margin").map_or(Ok(0.1), |v| v.parse())?;
        let url = match args.get("url") {
            Some(url) => url.clone(),
            None => latest(
                args.get("area").map_or("mywavewam800m", String::as_str),
                args.get("run").map_or("00", String::as_str),
            )?,
        };
        let out = PathBuf::from(
            args.get("out")
                .map_or("data/froya_wave_spectra.nc", String::as_str),
        );

        println!("Reading {url} ...");
        let start = Instant::now();
        let mut attempt = 0;
        let file = loop {
            match WaveSpectraFile::from_file(&url) {
                Ok(file) => break file,
                Err(e) if attempt < 3 => {
                    attempt += 1;
                    eprintln!("  {e}; retrying ({attempt}/3)");
                    std::thread::sleep(std::time::Duration::from_secs(5 * attempt));
                }
                Err(e) => return Err(e.into()),
            }
        };
        println!(
            "  {} points, {} times, {} frequencies ({:.3}–{:.3} Hz) × {} directions, in {:.1} s",
            file.n_points(),
            file.times.len(),
            file.frequencies.len(),
            file.frequencies[0],
            file.frequencies[file.frequencies.len() - 1],
            file.directions_to.len(),
            start.elapsed().as_secs_f64()
        );
        let [lon0, lat0, lon1, lat1] = bbox;
        // Points WAM has on land carry fill values only: leave them out
        let nc = file.frequencies.len() * file.directions_to.len();
        let np_all = file.n_points();
        // (and an `hs` of fill values: their SPEC can hold zeros)
        let has_sea = |p: usize| {
            (0..file.times.len()).any(|t| {
                let hs = file
                    .hs
                    .as_ref()
                    .is_none_or(|hs| hs[t * np_all + p].is_finite());
                hs && file.density[(t * np_all + p) * nc..][..nc]
                    .iter()
                    .any(|x| x.is_finite() && *x > 0.0)
            })
        };
        let inside: Vec<usize> = (0..file.n_points())
            .filter(|&p| {
                let (lon, lat) = (file.longitude[p], file.latitude[p]);
                (lon0 - margin..=lon1 + margin).contains(&lon)
                    && (lat0 - margin..=lat1 + margin).contains(&lat)
            })
            .filter(|&p| {
                let sea = has_sea(p);
                if !sea {
                    println!(
                        "  {:.3}°E {:.3}°N has no data (on land in WAM): left out",
                        file.longitude[p], file.latitude[p]
                    );
                }
                sea
            })
            .collect();
        if inside.is_empty() {
            return Err(format!(
                "no spectral point within {margin}° of {bbox:?}; try another area= or a larger margin="
            )
            .into());
        }
        let subset = file.select(&inside);
        let time = |t: f64| ModelClock::new(t).format(0.0);
        if let Some(reference) = subset.forecast_reference_time {
            println!("  Forecast from {} UTC", time(reference));
        }
        println!(
            "  {} → {} UTC, hourly",
            time(subset.times[0]),
            time(subset.times[subset.times.len() - 1])
        );
        let nt = subset.times.len();
        let np = subset.n_points();
        for p in 0..np {
            let range = |v: &Option<Vec<f64>>| {
                v.as_ref().map(|v| {
                    (0..nt)
                        .map(|t| v[t * np + p])
                        .filter(|x| x.is_finite())
                        .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), x| {
                            (a.min(x), b.max(x))
                        })
                })
            };
            let hs =
                range(&subset.hs).map_or(String::new(), |(a, b)| format!(", H_s {a:.2}–{b:.2} m"));
            let tp =
                range(&subset.tp).map_or(String::new(), |(a, b)| format!(", T_p {a:.1}–{b:.1} s"));
            println!(
                "  point {p}: {:.3}°E {:.3}°N{}{hs}{tp}",
                subset.longitude[p],
                subset.latitude[p],
                subset
                    .depth
                    .as_ref()
                    .filter(|d| d[p].is_finite() && d[p] < 1e30)
                    .map_or(String::new(), |d| format!(", {:.0} m deep", d[p]))
            );
        }
        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent)?;
        }
        subset.write(&out, &format!("MyWave WAM 800 m spectra from {url}"))?;
        println!("Wrote {}", out.display());
        Ok(())
    }
}

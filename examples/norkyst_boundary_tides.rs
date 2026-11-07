//! Boundary tides for a model domain from NorKyst-800 (OPeNDAP).
//!
//! Builds a [`TidalAtlas`] along the sides of a lon/lat box, the open
//! boundary of a domain meshed on that box (e.g. `froya_real_data`):
//!
//! 1. Sample the box perimeter every `spacing_km` and snap each sample to the
//!    nearest wet NorKyst cell (samples on land are dropped).
//! 2. Fetch hourly `zeta` and the z-level `u_eastward`/`v_northward` of those
//!    cells for `days` days from `start` (one window request per day and
//!    variable: THREDDS pays per stored chunk, not per cell).
//! 3. Depth-average the velocities over the water column
//!    ([`depth_average_z`](dgcom::io::depth_average_z)).
//! 4. Fit reference constants `(H, G)` of η, ū and v̄ with
//!    [`HarmonicAnalysis::fit`](dgcom::analysis::HarmonicAnalysis::fit) (V₀ at the first sample, f and u at the
//!    record midpoint), inferring P1 from K1 and K2 from S2: at the
//!    equilibrium ratios, or with `ratios=<series>` at the ratios NorKyst
//!    itself carries, fitted from its ζ at one cell over half a year or more
//!    (`scripts/norkyst_gauge_series.py`), so that the atlas holds NorKyst's
//!    own K2 and P1 (0.8× and 1.1–1.4× the equilibrium ones in 2026), as
//!    `froya_real_data nest_tides=theory` needs.
//! 5. Write the atlas (text format of `dgcom::boundary::TidalAtlas`).
//!
//! `points=lon,lat;lon,lat…` also fits those points (e.g. tide gauges inside
//! the domain, for validation), snapped to the nearest wet cell, into a
//! second atlas `points_out`. `spacing_km=0` skips the perimeter.
//!
//! NorKyst's aggregation has no `ubar`/`vbar`; its z-levels stop at 300 m, so
//! deeper columns extend the 300 m velocity to the bed. Tidal currents on the
//! shelf are close to depth-uniform, so the error is small, but the transports
//! are approximate.
//!
//! ## Run
//!
//! ```bash
//! cargo run --release --example norkyst_boundary_tides -- \
//!     [bbox=8.0,63.6,9.2,64.0] [start=2025-06-01] [days=30] [spacing_km=1.0] \
//!     [out=data/froya_boundary_tides.txt] [url=<OPeNDAP URL>] \
//!     [points=8.665231,63.869331] [points_out=data/froya_station_tides.txt] //!     [ratios=data/norkyst_gauges/2026-04_2026-10/MSU.txt]
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
    use std::collections::HashMap;
    use std::error::Error;
    use std::path::PathBuf;
    use std::time::Instant;

    use dgcom::analysis::{AnalysisError, HarmonicAnalysis, Inference, ReferenceFit};
    use dgcom::boundary::{AtlasConstituent, AtlasPoint, TidalAtlas};
    use dgcom::io::{depth_average_z, read_tide_gauge_file};
    use dgcom::time::ModelClock;
    use netcdf::Variable;

    use super::dap::{attribute_f64, locate_window, retry, slice};

    const URL: &str = "https://thredds.met.no/thredds/dodsC/fou-hi/norkystv3_800m_m00_be";
    /// Constituents resolvable from a month of hourly data
    const NAMES: [&str; 10] = ["M2", "S2", "N2", "K1", "O1", "Q1", "M4", "MS4", "MN4", "M6"];
    const INFERRED: [Inference; 2] = [Inference::P1_FROM_K1, Inference::K2_FROM_S2];
    /// What half a year separates, for `ratios=`
    const HALF_YEAR: [&str; 12] = [
        "M2", "S2", "N2", "K1", "O1", "Q1", "K2", "P1", "M4", "MS4", "MN4", "M6",
    ];
    /// Coarse stride for locating the domain in the NorKyst grid
    const STRIDE: usize = 8;
    /// Farthest a perimeter sample may snap to a wet cell (m)
    const MAX_SNAP: f64 = 1200.0;
    /// Packed-integer fill value of NorKyst's Int16 variables
    const FILL: i16 = -32767;

    struct Options {
        bbox: [f64; 4],
        start: String,
        days: usize,
        spacing_km: f64,
        out: PathBuf,
        url: String,
        points: Vec<(f64, f64)>,
        points_out: PathBuf,
        ratios: Option<PathBuf>,
    }

    impl Options {
        fn parse() -> Result<Self, String> {
            let args: HashMap<String, String> = std::env::args()
                .skip(1)
                .filter_map(|a| a.split_once('=').map(|(k, v)| (k.into(), v.into())))
                .collect();
            let num = |key: &str, default: f64| -> Result<f64, String> {
                args.get(key).map_or(Ok(default), |v| {
                    v.parse().map_err(|_| format!("bad {key}={v}"))
                })
            };
            let bbox: Vec<f64> = args
                .get("bbox")
                .map_or("8.0,63.6,9.2,64.0", String::as_str)
                .split(',')
                .map(|v| v.parse().map_err(|_| format!("bad bbox value {v}")))
                .collect::<Result<_, _>>()?;
            Ok(Self {
                bbox: bbox
                    .try_into()
                    .map_err(|_| "bbox=min_lon,min_lat,max_lon,max_lat".to_string())?,
                start: args.get("start").cloned().unwrap_or("2025-06-01".into()),
                days: num("days", 30.0)? as usize,
                spacing_km: num("spacing_km", 1.0)?,
                out: args
                    .get("out")
                    .map_or("data/froya_boundary_tides.txt".into(), PathBuf::from),
                url: args.get("url").cloned().unwrap_or(URL.into()),
                points: args
                    .get("points")
                    .map_or("", String::as_str)
                    .split(';')
                    .filter(|p| !p.trim().is_empty())
                    .map(|p| {
                        let (lon, lat) = p.split_once(',').ok_or(format!("bad point {p}"))?;
                        let num = |v: &str| v.trim().parse().map_err(|_| format!("bad point {p}"));
                        Ok((num(lon)?, num(lat)?))
                    })
                    .collect::<Result<_, String>>()?,
                points_out: args
                    .get("points_out")
                    .map_or("data/froya_station_tides.txt".into(), PathBuf::from),
                ratios: args.get("ratios").map(PathBuf::from),
            })
        }
    }

    /// Distance (m) between two nearby points, equirectangular.
    fn distance((lon1, lat1): (f64, f64), (lon2, lat2): (f64, f64)) -> f64 {
        let r = 6_371_000.0;
        let x = (lon2 - lon1).to_radians() * (0.5 * (lat1 + lat2)).to_radians().cos();
        let y = (lat2 - lat1).to_radians();
        r * x.hypot(y)
    }

    /// A wet cell on the perimeter and its accumulated series.
    struct Cell {
        j: usize,
        i: usize,
        lon: f64,
        lat: f64,
        depth: f64,
        /// A requested point (`points=`), not a perimeter sample
        station: bool,
        eta: Vec<f64>,
        u: Vec<f64>,
        v: Vec<f64>,
    }

    /// P1/K1 and K2/S2 as a series of the source carries them (`ratios=`),
    /// else the equilibrium ones, and how they were found.
    fn inferences(ratios: Option<&PathBuf>) -> Result<([Inference; 2], String), Box<dyn Error>> {
        let Some(path) = ratios else {
            return Ok((INFERRED, "at the equilibrium ratios".into()));
        };
        let series = read_tide_gauge_file(path)?.time_series;
        let (t, v): (Vec<f64>, Vec<f64>) = series
            .times()
            .iter()
            .zip(series.values())
            .filter(|(_, v)| v.is_finite())
            .map(|(&t, &v)| (t, v))
            .unzip();
        let analysis = HarmonicAnalysis::new(&HALF_YEAR);
        let (Some(&first), Some(&last)) = (t.first(), t.last()) else {
            return Err(format!("ratios={}: no samples", path.display()).into());
        };
        if last - first < analysis.minimum_record_length() {
            return Err(format!(
                "ratios={}: {:.0} days cannot separate K2 from S2 ({:.0} needed)",
                path.display(),
                (last - first) / 86_400.0,
                analysis.minimum_record_length() / 86_400.0
            )
            .into());
        }
        let fit = analysis.fit(&t, &v)?;
        let inferences = INFERRED.map(|i| {
            Inference::from_reference(i.name, i.from, &fit).expect("fitted K2, S2, P1, K1")
        });
        let how = format!(
            "at the ratios of {} ({} to {}): {}",
            path.display(),
            &ModelClock::new(first).format(0.0)[..10],
            &ModelClock::new(last).format(0.0)[..10],
            inferences
                .iter()
                .map(|i| format!(
                    "{}/{} {:.3}, {:+.1}°",
                    i.name, i.from, i.amplitude_ratio, i.lag_offset_deg
                ))
                .collect::<Vec<_>>()
                .join("; ")
        );
        Ok((inferences, how))
    }

    pub fn run() -> Result<(), Box<dyn Error>> {
        let opts = Options::parse()?;
        let (inferred, how) = inferences(opts.ratios.as_ref())?;
        println!("  P1 and K2 inferred {how}");
        let [min_lon, min_lat, max_lon, max_lat] = opts.bbox;
        let start = ModelClock::parse(&opts.start)?.epoch_unix;
        let hours = opts.days * 24;
        println!("NorKyst-800 boundary tides for [{min_lon}, {max_lon}] × [{min_lat}, {max_lat}]");
        println!("  {} from {} ({} days)", opts.url, opts.start, opts.days);

        let file = retry("open", || netcdf::open(&opts.url))?;
        let var = |name: &str| file.variable(name).ok_or(format!("no variable {name}"));

        // Time window (hourly)
        let time: Vec<f64> = retry("time", || var("time").unwrap().get_values(..))?;
        let t0 = time
            .iter()
            .position(|&t| t >= start)
            .ok_or("start is after the end of the aggregation")?;
        if t0 + hours > time.len() {
            return Err("the record runs past the end of the aggregation".into());
        }
        let times = &time[t0..t0 + hours];
        if times.windows(2).any(|w| (w[1] - w[0] - 3600.0).abs() > 1.0) {
            return Err("the record is not hourly".into());
        }

        // Locate the box on a coarse copy of the grid, then read it in full
        let win = locate_window(
            &file,
            "lon",
            "lat",
            [min_lon, min_lat, max_lon, max_lat],
            0.05,
            STRIDE,
        )?;
        println!(
            "  grid window: Y {}+{}, X {}+{}",
            win.y0, win.ny, win.x0, win.nx
        );
        let read2d = |name: &str| -> Result<Vec<f64>, Box<dyn Error>> {
            retry(name, || {
                var(name)
                    .unwrap()
                    .get_values([slice(win.y0, win.ny), slice(win.x0, win.nx)])
            })
        };
        let (lon, lat, h) = (read2d("lon")?, read2d("lat")?, read2d("h")?);
        let zeta_var = var("zeta")?;
        let wet: Vec<bool> = retry("land mask", || {
            zeta_var.get_values::<i16, _>([
                slice(t0, 1),
                slice(win.y0, win.ny),
                slice(win.x0, win.nx),
            ])
        })?
        .iter()
        .map(|&z| z != FILL)
        .collect();

        // Perimeter samples → nearest wet cells, in perimeter order
        let corners = [
            (min_lon, min_lat),
            (max_lon, min_lat),
            (max_lon, max_lat),
            (min_lon, max_lat),
        ];
        let mut cells: Vec<Cell> = Vec::new();
        // Snap `p` to the nearest wet cell within MAX_SNAP and add it once
        let mut add = |p: (f64, f64), station: bool| -> Option<f64> {
            let (d, k) = (0..lon.len())
                .filter(|&k| wet[k])
                .map(|k| (distance(p, (lon[k], lat[k])), k))
                .min_by(|x, y| x.0.total_cmp(&y.0))
                .filter(|&(d, _)| d <= MAX_SNAP)?;
            let (j, i) = (k / win.nx, k % win.nx);
            if !cells
                .iter()
                .any(|c| (c.j, c.i, c.station) == (j, i, station))
            {
                cells.push(Cell {
                    j,
                    i,
                    lon: lon[k],
                    lat: lat[k],
                    depth: h[k],
                    station,
                    eta: Vec::with_capacity(hours),
                    u: Vec::with_capacity(hours),
                    v: Vec::with_capacity(hours),
                });
            }
            Some(d)
        };
        if opts.spacing_km > 0.0 {
            for side in 0..4 {
                let (a, b) = (corners[side], corners[(side + 1) % 4]);
                let n = (distance(a, b) / (1000.0 * opts.spacing_km)).ceil() as usize;
                for s in 0..n {
                    let f = s as f64 / n as f64;
                    add((a.0 + f * (b.0 - a.0), a.1 + f * (b.1 - a.1)), false);
                }
            }
        }
        for &p in &opts.points {
            match add(p, true) {
                Some(d) => println!("  point ({:.4}, {:.4}): wet cell {d:.0} m away", p.0, p.1),
                None => eprintln!(
                    "  point ({:.4}, {:.4}): no wet cell within {MAX_SNAP} m",
                    p.0, p.1
                ),
            }
        }
        let n_perimeter = cells.iter().filter(|c| !c.station).count();
        println!(
            "  {n_perimeter} wet boundary cells, {} points",
            cells.len() - n_perimeter
        );

        // Hourly data, one day per request
        let depth_var = var("depth")?;
        let levels: Vec<f64> = retry("depth", || depth_var.get_values(..))?;
        let nz = levels.len();
        let (u_var, v_var) = (var("u_eastward")?, var("v_northward")?);
        let scale = |v: &Variable| {
            (
                attribute_f64(v, "scale_factor").unwrap_or(1.0),
                attribute_f64(v, "add_offset").unwrap_or(0.0),
            )
        };
        let (zeta_scale, u_scale, v_scale) = (scale(&zeta_var), scale(&u_var), scale(&v_var));
        let started = Instant::now();
        for day in 0..opts.days {
            let ts = t0 + 24 * day;
            let zeta: Vec<i16> = retry("zeta", || {
                zeta_var.get_values([slice(ts, 24), slice(win.y0, win.ny), slice(win.x0, win.nx)])
            })?;
            let read3d = |var: &Variable| {
                retry(var.name().as_str(), || {
                    var.get_values::<i16, _>([
                        slice(ts, 24),
                        slice(0, nz),
                        slice(win.y0, win.ny),
                        slice(win.x0, win.nx),
                    ])
                })
            };
            let (u, v) = (read3d(&u_var)?, read3d(&v_var)?);
            let plane = win.ny * win.nx;
            for cell in &mut cells {
                let at = cell.j * win.nx + cell.i;
                for hour in 0..24 {
                    let z = zeta[hour * plane + at];
                    cell.eta.push(if z == FILL {
                        f64::NAN
                    } else {
                        zeta_scale.1 + zeta_scale.0 * z as f64
                    });
                    let column = |data: &[i16], (s, o): (f64, f64)| -> Vec<Option<f64>> {
                        (0..nz)
                            .map(|l| {
                                let raw = data[(hour * nz + l) * plane + at];
                                (raw != FILL).then_some(o + s * raw as f64)
                            })
                            .collect()
                    };
                    let mean = |column: Vec<Option<f64>>| {
                        depth_average_z(&levels, &column, cell.depth).unwrap_or(f64::NAN)
                    };
                    cell.u.push(mean(column(&u, u_scale)));
                    cell.v.push(mean(column(&v, v_scale)));
                }
            }
            println!(
                "  day {:2}/{}: {:.0} s elapsed",
                day + 1,
                opts.days,
                started.elapsed().as_secs_f64()
            );
        }

        // Harmonic fits: perimeter cells into `atlas`, requested points into
        // `stations`
        let header = vec![
            "dgcom tidal atlas: reference constants (H, Greenwich lag G)".to_string(),
            format!("source: NorKyst-800 {}", opts.url),
            format!(
                "record: {} + {} days hourly; fit {} with {} inferred {how}",
                ModelClock::new(times[0]).format(0.0),
                opts.days,
                NAMES.join(" "),
                inferred.map(|i| format!("{} from {}", i.name, i.from)).join(", ")
            ),
            "velocity: z-level u_eastward/v_northward averaged over the column (levels end at 300 m)".into(),
        ];
        let mut atlas = TidalAtlas {
            points: Vec::new(),
            header: header.clone(),
        };
        let mut stations = TidalAtlas {
            points: Vec::new(),
            header,
        };
        let (mut r2, mut r2_stations) = (Vec::new(), Vec::new());
        let analysis = HarmonicAnalysis::new(&NAMES).with_inferences(&inferred);
        for cell in &cells {
            let fit = |values: &[f64]| -> Result<ReferenceFit, AnalysisError> {
                let (t, x): (Vec<f64>, Vec<f64>) = times
                    .iter()
                    .zip(values)
                    .filter(|(_, v)| v.is_finite())
                    .map(|(&t, &v)| (t, v))
                    .unzip();
                analysis.fit(&t, &x)
            };
            let (eta, u, v) = match (fit(&cell.eta), fit(&cell.u), fit(&cell.v)) {
                (Ok(e), Ok(u), Ok(v)) => (e, u, v),
                (e, u, v) => {
                    let error = [e.err(), u.err(), v.err()].into_iter().flatten().next();
                    eprintln!(
                        "  skipping cell ({:.4}, {:.4}): {}",
                        cell.lon,
                        cell.lat,
                        error.map_or(String::new(), |e| e.to_string())
                    );
                    continue;
                }
            };
            let constituents = eta
                .constants
                .iter()
                .zip(&u.constants)
                .zip(&v.constants)
                .map(|((e, u), v)| AtlasConstituent {
                    name: e.name,
                    eta: (e.amplitude, e.lag_deg),
                    velocity: Some(((u.amplitude, u.lag_deg), (v.amplitude, v.lag_deg))),
                })
                .collect();
            let (target, r2) = if cell.station {
                (&mut stations, &mut r2_stations)
            } else {
                (&mut atlas, &mut r2)
            };
            r2.push(eta.r_squared);
            target.points.push(AtlasPoint {
                lon: cell.lon,
                lat: cell.lat,
                depth: cell.depth,
                mean: Some(eta.mean),
                constituents,
            });
        }

        for (atlas, mut r2, out) in [
            (atlas, r2, &opts.out),
            (stations, r2_stations, &opts.points_out),
        ] {
            if atlas.points.is_empty() {
                continue;
            }
            r2.sort_by(f64::total_cmp);
            let mut atlas = atlas;
            atlas.header.push(format!(
                "eta R^2: min {:.3}, median {:.3} over {} points",
                r2[0],
                r2[r2.len() / 2],
                r2.len()
            ));
            summarize(&atlas);
            if let Some(dir) = out.parent() {
                std::fs::create_dir_all(dir)?;
            }
            atlas.write(out)?;
            println!("Wrote {} points to {}", atlas.points.len(), out.display());
        }
        Ok(())
    }

    /// M2 amplitude and phase per side, for a quick look.
    fn summarize(atlas: &TidalAtlas) {
        println!("\n  lon      lat      depth |  M2 H (m)  G (°) | M2 |U| (m/s)");
        for p in atlas
            .points
            .iter()
            .step_by(atlas.points.len().div_ceil(20).max(1))
        {
            let m2 = p.constituents.iter().find(|c| c.name == "M2").unwrap();
            let speed = m2.velocity.map_or(0.0, |((u, _), (v, _))| u.hypot(v));
            println!(
                "  {:7.4}  {:7.4}  {:5.0} |  {:.3}  {:6.1} | {:.3}",
                p.lon, p.lat, p.depth, m2.eta.0, m2.eta.1, speed
            );
        }
    }
}

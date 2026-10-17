//! A 3D run against NorKyst-800, depth by depth (TODO P4.6): the run's
//! snapshot file (`froya_real_data levels=N tide3d=1 snapshot_minutes=60`)
//! against the parent file it was nested in.
//!
//! ```text
//! cargo run --release --example norkyst_profiles_3d -- <run.dgsnap> <norkyst.nc> \
//!     [points=name:lon,lat;name:xy:x_km,y_km;…] //!     [section=name:lon0,lat0:lon1,lat1:n] [spinup_hours=3] [out=<dir>]
//! ```
//!
//! The snapshot carries the mesh, the bed, the σ-grid, the clock and the
//! mesh's local projection, so nothing of the run's setup is rebuilt. At
//! every frame and each point:
//!
//! - the model's T, S, u, v on every σ-level, interpolated within the
//!   element holding the point (its nodal basis), at the layer centres'
//!   depths below the surface `d_l = −σ_l (η − B)`;
//! - NorKyst's profiles at the same instant, bilinear over the wet corners
//!   of the parent cell, linear in time between its snapshots
//!   (`ProfileSeries::sample`, as `OceanModelColumns` nests them);
//! - both at NorKyst's own z-levels down to the shallower of the two
//!   columns (the beds differ: the NorKyst point is 49 m deep in NorKyst,
//!   23 m in the coastline mesh), linear between levels or layer centres.
//!
//! Printed per point, over the frames after `spinup_hours`: per depth, the
//! bias, RMSE and correlation of T and S, the mean current of each, the RMS
//! of the vector difference, and the M2 tidal ellipse of each where the
//! record resolves it; the depth means; the bottom water of each column;
//! and the daily means of surface and bottom T and S. Then the domain
//! means of the surface T and S (model area-weighted over its wet nodes,
//! NorKyst over its wet points inside the mesh), every three hours.
//!
//! Written to `out` (default: the snapshot's directory): `profiles_<point>.txt`
//! (every frame and depth, both models) and `domain_mean.txt`.
//!
//! `section=` adds `n` points evenly spaced in longitude and latitude from
//! the first end to the second, named `<name>_00` … (`;` between sections);
//! `scripts/norkyst_section_3d.py` draws them as depth sections. With a
//! section and no `points=`, only the section is compared.
//!
//! The default points are the Mausund NorKyst point (the station atlas's
//! point next to the gauge), the Kattholmen farm (site 14042) and a deep
//! mid-domain point in Frohavet.

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
    use std::fmt::Write as _;
    use std::fs;
    use std::path::{Path, PathBuf};

    use dg_rs::analysis::{HarmonicAnalysis, Inference, TidalEllipse};
    use dg_rs::io::{
        CoordinateProjection, LocalProjection, OceanModelReader, ProfileLevels, ProfileSeries,
        SnapshotFrame, SnapshotReader, Stencil, TimeInterpolation, east_axis,
    };
    use dg_rs::mesh::PointLocator2D;
    use dg_rs::operators::DGOperators2D;
    use dg_rs::solver::Probe2D;
    use dg_rs::time::ModelClock;
    use dg_rs::types::ElementIndex;

    /// The Kattholmen point is the centre of the site's NYTEK boundary
    /// (`data/sites/14042.txt`), among the cages; the register's position
    /// (8.677683, 63.867683) is 400 m south of it, outside the boundary.
    const DEFAULT_POINTS: &str = "NorKyst point:8.6752,63.8726;\
                                  Kattholmen:8.67895,63.87098;\
                                  Frohavet:8.95,63.95";
    /// Constituents a current record may resolve (the others inferred)
    const CANDIDATES: [&str; 5] = ["M2", "S2", "K1", "O1", "M4"];
    /// Columns shallower than this (m) are not compared
    const MIN_DEPTH: f64 = 1.0;

    /// A comparison point.
    struct Point {
        name: String,
        lon: f64,
        lat: f64,
        probe: Probe2D,
        /// East in the mesh axes, `(cos, sin)` of its angle from x
        east: (f64, f64),
        parent: Stencil,
        /// NorKyst's still-water depth there (m)
        parent_depth: f64,
    }

    /// Both models' values at one depth and frame.
    #[derive(Clone, Copy)]
    struct Pair {
        model: [f64; 4],
        parent: [f64; 4],
    }

    /// The points of a section `name:lon0,lat0:lon1,lat1:n` as a `points=`
    /// list (`;name_00:lon,lat;name_01:…`).
    fn section_points(section: &str) -> Result<String, Box<dyn Error>> {
        let parts: Vec<&str> = section.split(':').collect();
        let [name, start, end, n] = parts[..] else {
            return Err(format!("section {section:?}: name:lon0,lat0:lon1,lat1:n").into());
        };
        let end_point = |p: &str| -> Result<(f64, f64), Box<dyn Error>> {
            let (lon, lat) = p
                .split_once(',')
                .ok_or(format!("section {section:?}: lon,lat ends"))?;
            Ok((lon.trim().parse()?, lat.trim().parse()?))
        };
        let ((lon0, lat0), (lon1, lat1)) = (end_point(start)?, end_point(end)?);
        let n: usize = n.trim().parse()?;
        if n < 2 {
            return Err(format!("section {section:?}: at least two points").into());
        }
        let mut list = String::new();
        for i in 0..n {
            let f = i as f64 / (n - 1) as f64;
            write!(
                list,
                ";{}_{i:02}:{:.6},{:.6}",
                name.trim(),
                lon0 + f * (lon1 - lon0),
                lat0 + f * (lat1 - lat0)
            )?;
        }
        Ok(list)
    }

    /// One frame at one point.
    struct Sample {
        /// Model time (s)
        t: f64,
        /// Column depths: model, NorKyst (m)
        depth: (f64, f64),
        /// Per comparison depth (NorKyst's levels), `None` below either bed
        pairs: Vec<Option<Pair>>,
        /// Depth means of u, v (east, north): model, NorKyst
        mean: ([f64; 2], [f64; 2]),
        /// T and S of the bottom water: the model's bed layer, NorKyst's
        /// deepest valid level
        bottom: ([f64; 2], [f64; 2]),
    }

    pub fn run() -> Result<(), Box<dyn Error>> {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let usage = "usage: norkyst_profiles_3d <run.dgsnap> <norkyst.nc> \
                     [points=name:lon,lat;…] [section=name:lon0,lat0:lon1,lat1:n]                      [spinup_hours=3] [out=<dir>]";
        let positional: Vec<&String> = args.iter().filter(|a| !a.contains('=')).collect();
        let [snapshot_path, parent_path] = [positional.first(), positional.get(1)]
            .map(|a| a.map(|s| PathBuf::from(s.as_str())).ok_or(usage));
        let (snapshot_path, parent_path) = (snapshot_path?, parent_path?);
        let options: HashMap<&str, &str> = args.iter().filter_map(|a| a.split_once('=')).collect();
        let spinup = 3600.0 * options.get("spinup_hours").map_or(Ok(3.0), |s| s.parse())?;
        let out_dir = options.get("out").map_or_else(
            || {
                snapshot_path
                    .parent()
                    .unwrap_or(Path::new("."))
                    .to_path_buf()
            },
            PathBuf::from,
        );

        let mut snapshot = SnapshotReader::open(&snapshot_path)?;
        let header = snapshot.header().clone();
        let levels = header.levels.clone().ok_or("not a 3D snapshot")?;
        let field = |name: &str| levels.field(name).ok_or(format!("no layered field {name}"));
        let fields = [field("temp")?, field("salt")?, field("u")?, field("v")?];
        let clock = header.clock.ok_or("the snapshot has no clock")?;
        let projection = match header
            .metadata("projection")
            .and_then(|p| p.strip_prefix("local,"))
            .and_then(|p| p.split_once(','))
        {
            Some((lat, lon)) => LocalProjection::new(lat.parse()?, lon.parse()?),
            None => return Err("the snapshot has no local projection".into()),
        };
        let ops = DGOperators2D::new(header.order);
        let (nn, nl) = (ops.n_nodes, header.n_levels());
        let sigma = levels.sigma.sigma_rho().to_vec();
        let mesh = &header.mesh;
        let locator = PointLocator2D::new(mesh);

        println!("Reading {} …", parent_path.display());
        let parent = OceanModelReader::from_file_with_profiles(&parent_path)?;
        let (Some(p_temp), Some(p_salt), Some((p_u, p_v))) = (
            parent.temperature_profile(),
            parent.salinity_profile(),
            parent.velocity_profiles(),
        ) else {
            return Err("the parent has no T, S, u and v profiles".into());
        };
        let series = [p_temp, p_salt, p_u, p_v];
        let ProfileLevels::Depth(depths) = p_temp.levels().clone() else {
            return Err("the parent's profiles are not on z-levels".into());
        };
        let parent_bed = parent.depth().ok_or("the parent has no depth")?;
        let wet = parent.wet_mask();

        // The points
        let sections = options.get("section").copied().unwrap_or("");
        let default_points = if sections.is_empty() {
            DEFAULT_POINTS
        } else {
            ""
        };
        let mut spec = options
            .get("points")
            .copied()
            .unwrap_or(default_points)
            .to_string();
        for section in sections.split(';').filter(|s| !s.trim().is_empty()) {
            spec.push_str(&section_points(section)?);
        }
        let mut points = Vec::new();
        for item in spec.split(';').filter(|s| !s.trim().is_empty()) {
            let (name, place) = item
                .split_once(':')
                .ok_or(format!("point {item:?}: name:lon,lat"))?;
            let (xy, place) = match place.strip_prefix("xy:") {
                Some(rest) => (true, rest),
                None => (false, place),
            };
            let (a, b) = place
                .split_once(',')
                .ok_or(format!("point {item:?}: two coordinates"))?;
            let (a, b): (f64, f64) = (a.trim().parse()?, b.trim().parse()?);
            let (lat, lon) = if xy {
                projection.xy_to_geo(1e3 * a, 1e3 * b)
            } else {
                (b, a)
            };
            let (x, y) = projection.geo_to_xy(lat, lon);
            let Some(probe) = Probe2D::at(&locator, &ops, [x, y]) else {
                println!("  {name}: outside the mesh, left out");
                continue;
            };
            let Some((_, parent_stencil)) = parent.stencil(lon, lat) else {
                println!("  {name}: outside NorKyst's wet cells, left out");
                continue;
            };
            let parent_depth = weighted(&parent_stencil, |k| parent_bed[k]);
            points.push(Point {
                name: name.trim().to_string(),
                lon,
                lat,
                probe,
                east: east_axis(&projection, lat, lon),
                parent: parent_stencil,
                parent_depth,
            });
        }
        if points.is_empty() {
            return Err("no point is inside both the mesh and NorKyst".into());
        }

        // NorKyst's wet points inside the mesh, for the domain means
        let inside: Vec<usize> = (0..parent.grid().len())
            .filter(|&k| wet[k])
            .filter(|&k| {
                let (lon, lat) = parent.grid().position(k);
                let (x, y) = projection.geo_to_xy(lat, lon);
                locator.locate([x, y]).is_some()
            })
            .collect();
        // Area per node (element area over its nodes, GLL-weighted)
        let node_area: Vec<f64> = {
            let w = &ops.weights;
            let total: f64 = w.iter().sum();
            (0..mesh.n_elements)
                .flat_map(|k| {
                    let area = quad_area(&mesh.element_vertices(ElementIndex::new(k)));
                    w.iter().map(move |wi| area * wi / total)
                })
                .collect()
        };

        let n_frames = snapshot.n_frames()?;
        println!(
            "{}: {n_frames} frames, {} elements × {nn} nodes × {nl} levels, P{}; {} NorKyst points \
             inside the mesh; NorKyst levels {:?} m",
            snapshot_path.display(),
            mesh.n_elements,
            header.order,
            inside.len(),
            depths
        );
        let mut samples: Vec<Vec<Sample>> = points.iter().map(|_| Vec::new()).collect();
        let mut domain = String::from(
            "# Domain means of the surface layer: model (area-weighted over nodes ≥ 1 m deep), \
             NorKyst (wet points inside the mesh, shallowest level)\n\
             # columns: unix_time hours T_model S_model T_norkyst S_norkyst\n",
        );
        let mut domain_rows = Vec::new();
        let mut frame = SnapshotFrame::default();
        let mut column = vec![0.0; nl];
        let mut element = vec![0.0; nn];
        let mut parent_out = vec![0.0; depths.len()];
        for i in 0..n_frames {
            snapshot.read_frame_into(i, &mut frame)?;
            let t = frame.t;
            let unix = clock.unix(t);
            let Some(time) = parent.time_stencil(unix, TimeInterpolation::Linear) else {
                continue;
            };
            for (point, record) in points.iter().zip(&mut samples) {
                let k = point.probe.element().as_usize();
                let at = |values: &[f32], element: &mut [f64]| {
                    for (e, j) in element.iter_mut().zip(k * nn..(k + 1) * nn) {
                        *e = f64::from(values[j]);
                    }
                    point.probe.evaluate(element)
                };
                let bed = point
                    .probe
                    .evaluate(&header.bathymetry[k * nn..(k + 1) * nn]);
                let eta = at(&frame.eta, &mut element);
                let model_depth = eta - bed;
                let state = parent.evaluate(&point.parent, &time);
                let parent_depth = point.parent_depth + state.ssh.unwrap_or(0.0);
                if model_depth < MIN_DEPTH {
                    continue;
                }
                // The model's columns at the layer centres, from the surface down
                let layer_depths: Vec<f64> = sigma.iter().rev().map(|s| -s * model_depth).collect();
                let mut model_columns =
                    [vec![0.0; nl], vec![0.0; nl], vec![0.0; nl], vec![0.0; nl]];
                for (f, &index) in fields.iter().enumerate() {
                    let layer = &frame.layers[index];
                    for l in 0..nl {
                        for (e, j) in element.iter_mut().zip(k * nn..(k + 1) * nn) {
                            *e = f64::from(layer[j * nl + l]);
                        }
                        column[l] = point.probe.evaluate(&element);
                    }
                    // Surface first
                    model_columns[f] = column.iter().rev().copied().collect();
                }
                rotate_to_east_north(&mut model_columns, point.east);
                let space = point
                    .parent
                    .idx
                    .iter()
                    .map(|&k| k as usize)
                    .zip(point.parent.w);
                let mut parent_columns = [
                    vec![0.0; depths.len()],
                    vec![0.0; depths.len()],
                    vec![0.0; depths.len()],
                    vec![0.0; depths.len()],
                ];
                for (f, s) in series.iter().enumerate() {
                    if !s.sample(&time, space.clone(), &depths, parent_depth, &mut parent_out) {
                        parent_out.fill(f64::NAN);
                    }
                    parent_columns[f].copy_from_slice(&parent_out);
                }
                let limit = model_depth.min(parent_depth);
                let pairs = depths
                    .iter()
                    .enumerate()
                    .map(|(j, &d)| {
                        (d <= limit).then(|| Pair {
                            model: std::array::from_fn(|f| {
                                interpolate(&layer_depths, &model_columns[f], d)
                            }),
                            parent: std::array::from_fn(|f| parent_columns[f][j]),
                        })
                    })
                    .collect();
                // Depth means: the model's barotropic mode, NorKyst's ubar/vbar
                let (c, s) = point.east;
                let (ubar, vbar) = (at(&frame.u, &mut element), at(&frame.v, &mut element));
                let model_mean = [ubar * c + vbar * s, -ubar * s + vbar * c];
                let parent_mean = state.velocity.map_or([f64::NAN; 2], |(u, v)| [u, v]);
                let parent_bottom = [0, 1].map(|f| {
                    deepest_valid(series[f], &time, &point.parent, &depths, point.parent_depth)
                });
                record.push(Sample {
                    t,
                    depth: (model_depth, parent_depth),
                    pairs,
                    mean: (model_mean, parent_mean),
                    bottom: (
                        [model_columns[0][nl - 1], model_columns[1][nl - 1]],
                        parent_bottom,
                    ),
                });
            }
            // Domain means of the surface
            let (mut area, mut sum_t, mut sum_s) = (0.0, 0.0, 0.0);
            let (temp, salt) = (&frame.layers[fields[0]], &frame.layers[fields[1]]);
            for (j, &a) in node_area.iter().enumerate() {
                if f64::from(frame.eta[j]) - header.bathymetry[j] >= MIN_DEPTH {
                    area += a;
                    sum_t += a * f64::from(temp[j * nl + nl - 1]);
                    sum_s += a * f64::from(salt[j * nl + nl - 1]);
                }
            }
            let parent_surface = |s: &ProfileSeries| {
                let (mut n, mut sum) = (0usize, 0.0);
                for &k in &inside {
                    let value: f64 = time
                        .terms()
                        .map(|(ti, w)| w * f64::from(s.column(ti, k)[0]))
                        .sum();
                    if value.is_finite() {
                        n += 1;
                        sum += value;
                    }
                }
                sum / n.max(1) as f64
            };
            let row = (
                t,
                sum_t / area,
                sum_s / area,
                parent_surface(p_temp),
                parent_surface(p_salt),
            );
            writeln!(
                domain,
                "{unix:.0} {:.3} {:.4} {:.4} {:.4} {:.4}",
                t / 3600.0,
                row.1,
                row.2,
                row.3,
                row.4
            )?;
            domain_rows.push(row);
        }

        fs::create_dir_all(&out_dir)?;
        fs::write(out_dir.join("domain_mean.txt"), &domain)?;
        for (point, record) in points.iter().zip(&samples) {
            let path = out_dir.join(format!("profiles_{}.txt", slug(&point.name)));
            fs::write(&path, profile_file(point, record, &depths, &clock))?;
            report(point, record, &depths, spinup, &clock);
            println!("  → {}", path.display());
        }

        println!("\nDomain means of the surface layer (model | NorKyst):");
        println!(
            "  {:>7} | {:>7} {:>7} | {:>7} {:>7}",
            "t (h)", "T", "S", "T", "S"
        );
        for (t, tm, sm, tp, sp) in &domain_rows {
            if (t / 3600.0).round() as i64 % 3 == 0 {
                println!(
                    "  {:>7.1} | {tm:>7.3} {sm:>7.3} | {tp:>7.3} {sp:>7.3}",
                    t / 3600.0
                );
            }
        }
        println!("  → {}", out_dir.join("domain_mean.txt").display());
        Ok(())
    }

    /// The per-depth, depth-mean, bottom and daily comparison at one point.
    fn report(point: &Point, record: &[Sample], depths: &[f64], spinup: f64, clock: &ModelClock) {
        let after: Vec<&Sample> = record.iter().filter(|s| s.t >= spinup - 1e-6).collect();
        println!(
            "\n{} ({:.5} E, {:.5} N), element {}: {} frames after {:.1} h; column {:.1} m in the \
             model, {:.1} m in NorKyst (means)",
            point.name,
            point.lon,
            point.lat,
            point.probe.element().as_usize(),
            after.len(),
            spinup / 3600.0,
            mean(after.iter().map(|s| s.depth.0)),
            mean(after.iter().map(|s| s.depth.1)),
        );
        if after.len() < 2 {
            println!("  too few frames to compare");
            return;
        }
        let times: Vec<f64> = after.iter().map(|s| clock.unix(s.t)).collect();
        let length = times[times.len() - 1] - times[0];
        let analysis =
            HarmonicAnalysis::resolvable(&CANDIDATES, length, 1.0, &Inference::EQUILIBRIUM);
        let m2 = |u: &[f64], v: &[f64]| -> Option<TidalEllipse> {
            analysis
                .fit_ellipses(&times, u, v)
                .ok()?
                .ellipses
                .into_iter()
                .find(|e| e.name == "M2" && !e.inferred)
        };
        println!(
            "  {:>5} {:>3} | {:>6} {:>6} {:>5} | {:>6} {:>6} {:>5} | {:>15} {:>15} {:>6} | {:>17} {:>17}",
            "depth",
            "n",
            "T bias",
            "RMSE",
            "r",
            "S bias",
            "RMSE",
            "r",
            "mean u,v model",
            "NorKyst",
            "|Δu|",
            "M2 major/inc model",
            "NorKyst"
        );
        for (j, &d) in depths.iter().enumerate() {
            let pairs: Vec<Pair> = after.iter().filter_map(|s| s.pairs[j]).collect();
            if pairs.len() < 2 {
                continue;
            }
            let column = |f: usize, model: bool| -> Vec<f64> {
                pairs
                    .iter()
                    .map(|p| if model { p.model[f] } else { p.parent[f] })
                    .collect()
            };
            let (tm, tp, sm, sp) = (
                column(0, true),
                column(0, false),
                column(1, true),
                column(1, false),
            );
            let (um, vm, up, vp) = (
                column(2, true),
                column(3, true),
                column(2, false),
                column(3, false),
            );
            let vector_rms = (pairs
                .iter()
                .map(|p| (p.model[2] - p.parent[2]).powi(2) + (p.model[3] - p.parent[3]).powi(2))
                .sum::<f64>()
                / pairs.len() as f64)
                .sqrt();
            // The tidal fit only on a complete record at this depth
            let ellipses = if pairs.len() == after.len() {
                (m2(&um, &vm), m2(&up, &vp))
            } else {
                (None, None)
            };
            let ellipse = |e: Option<TidalEllipse>| {
                e.map_or("—".to_string(), |e| {
                    format!("{:.3} {:4.0}°", e.major, e.inclination_deg)
                })
            };
            println!(
                "  {d:>5.0} {:>3} | {:>+6.2} {:>6.2} {:>5.2} | {:>+6.2} {:>6.2} {:>5.2} | {:>+7.3},{:>+7.3} {:>+7.3},{:>+7.3} {:>6.3} | {:>17} {:>17}",
                pairs.len(),
                mean(tm.iter().zip(&tp).map(|(a, b)| a - b)),
                rmse(&tm, &tp),
                correlation(&tm, &tp),
                mean(sm.iter().zip(&sp).map(|(a, b)| a - b)),
                rmse(&sm, &sp),
                correlation(&sm, &sp),
                mean(um.iter().copied()),
                mean(vm.iter().copied()),
                mean(up.iter().copied()),
                mean(vp.iter().copied()),
                vector_rms,
                ellipse(ellipses.0),
                ellipse(ellipses.1),
            );
        }
        // Depth means
        let (um, vm): (Vec<f64>, Vec<f64>) =
            after.iter().map(|s| (s.mean.0[0], s.mean.0[1])).unzip();
        let (up, vp): (Vec<f64>, Vec<f64>) =
            after.iter().map(|s| (s.mean.1[0], s.mean.1[1])).unzip();
        let ellipse = |u: &[f64], v: &[f64]| {
            m2(u, v).map_or("—".to_string(), |e| {
                format!(
                    "{:.3} m/s at {:.0}°, G {:.0}°",
                    e.major, e.inclination_deg, e.lag_deg
                )
            })
        };
        println!(
            "  depth mean: speed {:.3} m/s in the model, {:.3} in NorKyst (rms); M2 {} against {}",
            rms(um.iter().zip(&vm).map(|(u, v)| u.hypot(*v))),
            rms(up.iter().zip(&vp).map(|(u, v)| u.hypot(*v))),
            ellipse(&um, &vm),
            ellipse(&up, &vp),
        );
        // Bottom water and daily means
        let (tb, tbp): (Vec<f64>, Vec<f64>) =
            after.iter().map(|s| (s.bottom.0[0], s.bottom.1[0])).unzip();
        println!(
            "  bottom water: T {:.2}–{:.2} °C in the model's bed layer, {:.2}–{:.2} in NorKyst's \
             deepest level (bias {:+.2})",
            min(&tb),
            max(&tb),
            min(&tbp),
            max(&tbp),
            mean(tb.iter().zip(&tbp).map(|(a, b)| a - b)),
        );
        println!(
            "  daily means | surface T model NorKyst | S model NorKyst | bottom T model NorKyst | S model NorKyst"
        );
        let first_day = (clock.unix(record[0].t) / 86_400.0).floor();
        let mut day = first_day;
        loop {
            let in_day: Vec<&Sample> = record
                .iter()
                .filter(|s| (clock.unix(s.t) / 86_400.0).floor() == day && s.pairs[0].is_some())
                .collect();
            if in_day.is_empty() {
                break;
            }
            let surface = |f: usize, model: bool| {
                mean(in_day.iter().map(|s| {
                    let p = s.pairs[0].expect("surface pair");
                    if model { p.model[f] } else { p.parent[f] }
                }))
            };
            let bottom = |f: usize, model: bool| {
                mean(
                    in_day
                        .iter()
                        .map(|s| if model { s.bottom.0[f] } else { s.bottom.1[f] }),
                )
            };
            println!(
                "  {} ({:>2}) | {:>7.2} {:>7.2} | {:>7.3} {:>7.3} | {:>7.2} {:>7.2} | {:>7.3} {:>7.3}",
                clock
                    .format(clock.model_time(day * 86_400.0))
                    .get(..10)
                    .unwrap_or("?"),
                in_day.len(),
                surface(0, true),
                surface(0, false),
                surface(1, true),
                surface(1, false),
                bottom(0, true),
                bottom(0, false),
                bottom(1, true),
                bottom(1, false),
            );
            day += 1.0;
        }
    }

    /// Every frame and depth of both models at one point.
    fn profile_file(
        point: &Point,
        record: &[Sample],
        depths: &[f64],
        clock: &ModelClock,
    ) -> String {
        let mut out = format!(
            "# {} ({:.5} E, {:.5} N): the 3D run against NorKyst-800 at NorKyst's z-levels\n\
             # u, v east and north (m/s); T (°C); depth below the surface (m)\n\
             # columns: unix_time hours depth_model depth_norkyst depth \
             T_model S_model u_model v_model T_norkyst S_norkyst u_norkyst v_norkyst\n",
            point.name, point.lon, point.lat
        );
        for s in record {
            for (j, &d) in depths.iter().enumerate() {
                if let Some(p) = s.pairs[j] {
                    let _ = writeln!(
                        out,
                        "{:.0} {:.3} {:.2} {:.2} {d:.1} {:.4} {:.4} {:.4} {:.4} {:.4} {:.4} {:.4} {:.4}",
                        clock.unix(s.t),
                        s.t / 3600.0,
                        s.depth.0,
                        s.depth.1,
                        p.model[0],
                        p.model[1],
                        p.model[2],
                        p.model[3],
                        p.parent[0],
                        p.parent[1],
                        p.parent[2],
                        p.parent[3],
                    );
                }
            }
        }
        out
    }

    /// The model's velocity columns (`[T, S, u, v]`) from the mesh axes to
    /// east and north.
    fn rotate_to_east_north(columns: &mut [Vec<f64>; 4], (c, s): (f64, f64)) {
        let [_, _, u, v] = columns;
        for (u, v) in u.iter_mut().zip(v.iter_mut()) {
            let (x, y) = (*u, *v);
            (*u, *v) = (x * c + y * s, -x * s + y * c);
        }
    }

    /// `values` at depth `d`, linear between `at` (increasing) and held
    /// beyond them.
    fn interpolate(at: &[f64], values: &[f64], d: f64) -> f64 {
        let j = at.partition_point(|&a| a < d);
        if j == 0 {
            values[0]
        } else if j == at.len() {
            values[at.len() - 1]
        } else {
            let w = (d - at[j - 1]) / (at[j] - at[j - 1]);
            values[j - 1] + w * (values[j] - values[j - 1])
        }
    }

    /// The parent's value at its deepest valid level within `bed` (m), over
    /// the stencil and in time.
    fn deepest_valid(
        series: &ProfileSeries,
        time: &dg_rs::io::TimeStencil,
        stencil: &Stencil,
        depths: &[f64],
        bed: f64,
    ) -> f64 {
        let deepest = depths.partition_point(|&d| d <= bed).max(1) - 1;
        let space = stencil.idx.iter().map(|&k| k as usize).zip(stencil.w);
        let mut out = [0.0];
        if series.sample(time, space, &depths[deepest..=deepest], bed, &mut out) {
            out[0]
        } else {
            f64::NAN
        }
    }

    fn weighted(stencil: &Stencil, value: impl Fn(usize) -> f64) -> f64 {
        stencil
            .idx
            .iter()
            .zip(stencil.w)
            .filter(|(_, w)| *w > 0.0)
            .map(|(&k, w)| w * value(k as usize))
            .sum()
    }

    /// Area of a quadrilateral (shoelace)
    fn quad_area(v: &[[f64; 2]; 4]) -> f64 {
        0.5 * (0..4)
            .map(|i| {
                let (a, b) = (v[i], v[(i + 1) % 4]);
                a[0] * b[1] - b[0] * a[1]
            })
            .sum::<f64>()
            .abs()
    }

    fn mean(x: impl Iterator<Item = f64>) -> f64 {
        let (n, sum) = x.fold((0usize, 0.0), |(n, s), v| (n + 1, s + v));
        sum / n.max(1) as f64
    }

    fn rms(x: impl Iterator<Item = f64>) -> f64 {
        mean(x.map(|v| v * v)).sqrt()
    }

    fn rmse(a: &[f64], b: &[f64]) -> f64 {
        rms(a.iter().zip(b).map(|(x, y)| x - y))
    }

    fn correlation(a: &[f64], b: &[f64]) -> f64 {
        let (ma, mb) = (mean(a.iter().copied()), mean(b.iter().copied()));
        let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
        for (x, y) in a.iter().zip(b) {
            sab += (x - ma) * (y - mb);
            saa += (x - ma).powi(2);
            sbb += (y - mb).powi(2);
        }
        sab / (saa * sbb).sqrt()
    }

    fn min(x: &[f64]) -> f64 {
        x.iter().copied().fold(f64::INFINITY, f64::min)
    }

    fn max(x: &[f64]) -> f64 {
        x.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    }

    fn slug(name: &str) -> String {
        name.to_lowercase()
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { '_' })
            .collect()
    }
}

//! Measured depths at points, and the field that corrects a raster to them.
//!
//! A bed raster's sea is an interpolation wherever its source had no
//! soundings: Kartverket's 50 m topobathy model is 10–35 m too shallow at
//! every station of Kattholmen's environmental survey, whose nearest chart
//! soundings are 340–460 m away (`notes/inputs-forcing.md` P1.6). Surveys
//! measure the bed at points: a fish farm's MOM-B survey has 10–20+
//! stations under and around its cages. [`crate::io::BedRaster::correct_to_soundings`]
//! moves the raster's sea towards them with the field [`SimpleKriging`]
//! builds here.
//!
//! # The correction
//!
//! At station k the residual is `e_k = B_k − B̃(x_k)`, its measured bed
//! elevation less the raster's. The residuals are taken as samples of a
//! zero-mean random field with the spherical covariance `σ² ρ(d/R)`,
//!
//! ```text
//! ρ(s) = 1 − 3s/2 + s³/2   for s < 1,   0 beyond,
//! ```
//!
//! plus independent noise of variance `τ²` at each station (its position,
//! the tide at the time of sounding, the bed's slope inside a pixel). The
//! simple-kriging estimate, the field's mean given the residuals (Cressie,
//! *Statistics for Spatial Data*, 1993, §3.4.5; Chilès and Delfiner,
//! *Geostatistics*, 2012, §3.3), is
//!
//! ```text
//! ê(x) = Σ_k σ² ρ(|x − x_k|/R) α_k,   (σ² P + τ² I) α = e,   P_kj = ρ(|x_k − x_j|/R).
//! ```
//!
//! It tends to `e_k` at a station as `τ → 0`, shrinks towards 0 where few
//! stations agree under a large noise, and is exactly 0 farther than `R`
//! from every station: there the raster is left as it is. The zero mean is
//! the choice: a survey's bias near its stations says nothing about the bed
//! away from them (farms are placed in deep water). The spherical
//! covariance is the one the farms' residuals show: their semivariance
//! rises nearly linearly with distance and levels off at the sill by about
//! 800 m at Frøya's 654 stations (`notes/inputs-forcing.md`).
//!
//! `σ²` is the stations' mean square residual less `τ²` (0 if that is
//! negative: residuals no larger than the noise correct nothing). Stations
//! farther than `R` apart do not interact, so the system splits into one
//! block per cluster of stations within `R` of each other, each solved by
//! its Cholesky factor.
//!
//! # File format
//!
//! A CSV file with a header row naming its columns; `lon`, `lat` (degrees)
//! and `depth_m` (m below mean sea level, positive under water) are read and
//! the others ignored. Blank lines and lines starting with `#` are skipped.
//! A survey's depths are as sounded, the tide at the time not taken off:
//! that is part of the noise.
//!
//! ```text
//! # Kattholmen, MOM-B 2026-04-21
//! site,report,date,lon,lat,depth_m
//! 14042,25633,2026-04-21,8.67452,63.86931,41.0
//! ```

use std::path::Path;

/// A measured depth at a point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sounding {
    /// Longitude (degrees east)
    pub lon: f64,
    /// Latitude (degrees north)
    pub lat: f64,
    /// Depth below mean sea level (m, positive under water)
    pub depth: f64,
}

/// A soundings file that could not be read.
#[derive(Debug, thiserror::Error)]
pub enum SoundingsError {
    /// The file could not be read.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// A line is malformed.
    #[error("line {line}: {message}")]
    Parse {
        /// 1-based line number
        line: usize,
        /// What is wrong with it
        message: String,
    },
}

/// Read a soundings file (see the [module docs](self)).
pub fn read_soundings_file(path: impl AsRef<Path>) -> Result<Vec<Sounding>, SoundingsError> {
    parse_soundings_str(&std::fs::read_to_string(path)?)
}

/// Parse a soundings file's contents (see the [module docs](self)).
pub fn parse_soundings_str(text: &str) -> Result<Vec<Sounding>, SoundingsError> {
    let mut columns: Option<(usize, [usize; 3])> = None;
    let mut soundings = Vec::new();
    for (n, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let error = |message: String| SoundingsError::Parse {
            line: n + 1,
            message,
        };
        let fields: Vec<&str> = line.split(',').map(str::trim).collect();
        let Some((width, [lon, lat, depth])) = columns else {
            let find = |name: &str| {
                fields
                    .iter()
                    .position(|f| *f == name)
                    .ok_or_else(|| error(format!("the header has no `{name}` column")))
            };
            columns = Some((fields.len(), [find("lon")?, find("lat")?, find("depth_m")?]));
            continue;
        };
        if fields.len() != width {
            return Err(error(format!(
                "{} fields where the header has {width}",
                fields.len()
            )));
        }
        let number = |i: usize, name: &str| -> Result<f64, SoundingsError> {
            fields[i]
                .parse::<f64>()
                .ok()
                .filter(|v| v.is_finite())
                .ok_or_else(|| error(format!("`{name}` is not a number: {:?}", fields[i])))
        };
        soundings.push(Sounding {
            lon: number(lon, "lon")?,
            lat: number(lat, "lat")?,
            depth: number(depth, "depth_m")?,
        });
    }
    if columns.is_none() {
        return Err(SoundingsError::Parse {
            line: 0,
            message: "no header row".into(),
        });
    }
    Ok(soundings)
}

/// What [`crate::io::BedRaster::correct_to_soundings`] did.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SoundingCorrection {
    /// The stations given
    pub stations: usize,
    /// Those outside the raster, left out
    pub off_raster: usize,
    /// Those on the raster's land, left out
    pub on_land: usize,
    /// Those used
    pub used: usize,
    /// At each station used, its depth less the raster's before the
    /// correction (m; positive where the raster was too shallow)
    pub deeper_than_raster: Vec<f64>,
    /// The same after it
    pub deeper_than_corrected: Vec<f64>,
    /// The field's standard deviation `σ` (m)
    pub signal: f64,
    /// Pixels moved
    pub pixels: usize,
    /// Pixels held at the shallowest depth a correction lifts to
    pub held: usize,
    /// The largest deepening of a pixel (m)
    pub deepened_by: f64,
    /// The largest shoaling of a pixel (m)
    pub shoaled_by: f64,
}

impl SoundingCorrection {
    /// The median and quartiles of `values` (NaN if empty).
    pub fn quartiles(values: &[f64]) -> [f64; 3] {
        let mut v = values.to_vec();
        v.sort_by(f64::total_cmp);
        let at = |q: f64| {
            if v.is_empty() {
                return f64::NAN;
            }
            let x = q * (v.len() - 1) as f64;
            let (i, t) = (x.floor() as usize, x.fract());
            v[i] + t * (v[(i + 1).min(v.len() - 1)] - v[i])
        };
        [at(0.25), at(0.5), at(0.75)]
    }
}

/// The spherical correlation `ρ(d/R)`: 1 at 0, falling nearly linearly,
/// 0 from `range` on.
pub fn spherical_correlation(distance: f64, range: f64) -> f64 {
    let s = distance / range;
    if s < 1.0 {
        1.0 - s * (1.5 - 0.5 * s * s)
    } else {
        0.0
    }
}

/// The simple-kriging estimate of a zero-mean field from noisy samples,
/// under the spherical covariance (see the [module docs](self)).
#[derive(Clone, Debug)]
pub struct SimpleKriging {
    points: Vec<[f64; 2]>,
    /// `σ² α_k`: each sample's weight on the correlation around it
    weights: Vec<f64>,
    range: f64,
    signal_variance: f64,
}

impl SimpleKriging {
    /// Fit the field to `values` at `points` (m, on one plane), with the
    /// correlation range `range` (m) and the samples' noise `noise` (its
    /// standard deviation, in the values' unit).
    ///
    /// # Panics
    /// If the lengths differ, `range` or `noise` is not positive, or a
    /// value is not finite.
    pub fn new(points: &[[f64; 2]], values: &[f64], range: f64, noise: f64) -> Self {
        assert_eq!(points.len(), values.len(), "one value per point");
        assert!(
            range.is_finite() && range > 0.0,
            "the range is positive, not {range}"
        );
        assert!(
            noise.is_finite() && noise > 0.0,
            "the noise is positive, not {noise}"
        );
        assert!(values.iter().all(|v| v.is_finite()), "values are finite");
        let n = points.len();
        let noise_variance = noise * noise;
        let mean_square = values.iter().map(|v| v * v).sum::<f64>() / n.max(1) as f64;
        let signal_variance = (mean_square - noise_variance).max(0.0);
        let mut weights = vec![0.0; n];
        if signal_variance > 0.0 {
            let distance = |i: usize, j: usize| {
                (points[i][0] - points[j][0]).hypot(points[i][1] - points[j][1])
            };
            for cluster in clusters(n, |i, j| distance(i, j) < range) {
                let m = cluster.len();
                let mut matrix = vec![0.0; m * m];
                for (a, &i) in cluster.iter().enumerate() {
                    for (b, &j) in cluster.iter().enumerate().take(a + 1) {
                        let c = signal_variance * spherical_correlation(distance(i, j), range);
                        matrix[a * m + b] = c;
                        matrix[b * m + a] = c;
                    }
                    matrix[a * m + a] += noise_variance;
                }
                let mut rhs: Vec<f64> = cluster.iter().map(|&i| values[i]).collect();
                cholesky_solve(&mut matrix, &mut rhs, m);
                for (a, &i) in cluster.iter().enumerate() {
                    weights[i] = signal_variance * rhs[a];
                }
            }
        }
        Self {
            points: points.to_vec(),
            weights,
            range,
            signal_variance,
        }
    }

    /// The estimate at `point`.
    pub fn estimate(&self, point: [f64; 2]) -> f64 {
        self.terms()
            .map(|(p, w)| {
                w * spherical_correlation((point[0] - p[0]).hypot(point[1] - p[1]), self.range)
            })
            .sum()
    }

    /// Each sample's point and weight: the estimate is the sum of
    /// `weight · ρ(|x − point|/R)` over them, so a caller can spread each
    /// over the points within the range of it alone.
    pub fn terms(&self) -> impl Iterator<Item = ([f64; 2], f64)> + '_ {
        self.points
            .iter()
            .copied()
            .zip(self.weights.iter().copied())
    }

    /// The correlation range (m).
    pub fn range(&self) -> f64 {
        self.range
    }

    /// `σ²`, the field's variance: the samples' mean square less the noise's
    /// variance, or 0.
    pub fn signal_variance(&self) -> f64 {
        self.signal_variance
    }
}

/// The connected components of `0..n` under `linked` (each sorted).
fn clusters(n: usize, linked: impl Fn(usize, usize) -> bool) -> Vec<Vec<usize>> {
    let mut parent: Vec<usize> = (0..n).collect();
    fn root(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    for i in 0..n {
        for j in 0..i {
            if linked(i, j) {
                let (a, b) = (root(&mut parent, i), root(&mut parent, j));
                parent[a.max(b)] = a.min(b);
            }
        }
    }
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut slot = vec![usize::MAX; n];
    for i in 0..n {
        let r = root(&mut parent, i);
        if slot[r] == usize::MAX {
            slot[r] = groups.len();
            groups.push(Vec::new());
        }
        groups[slot[r]].push(i);
    }
    groups
}

/// Solve `A x = b` for a symmetric positive definite `m × m` matrix
/// (row-major, overwritten by its Cholesky factor), `b` overwritten by `x`.
fn cholesky_solve(a: &mut [f64], b: &mut [f64], m: usize) {
    for j in 0..m {
        let mut d = a[j * m + j];
        for k in 0..j {
            d -= a[j * m + k] * a[j * m + k];
        }
        assert!(d > 0.0, "the kriging matrix is positive definite");
        let d = d.sqrt();
        a[j * m + j] = d;
        for i in j + 1..m {
            let mut s = a[i * m + j];
            for k in 0..j {
                s -= a[i * m + k] * a[j * m + k];
            }
            a[i * m + j] = s / d;
        }
    }
    for i in 0..m {
        let mut s = b[i];
        for k in 0..i {
            s -= a[i * m + k] * b[k];
        }
        b[i] = s / a[i * m + i];
    }
    for i in (0..m).rev() {
        let mut s = b[i];
        for k in i + 1..m {
            s -= a[k * m + i] * b[k];
        }
        b[i] = s / a[i * m + i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_file_reads_by_its_header() {
        let text = "# a survey\n\nsite,lat,depth_m,lon\n7,63.8,41.5,8.6\n8, 63.9 ,12,8.7\n";
        let s = parse_soundings_str(text).unwrap();
        assert_eq!(
            s,
            vec![
                Sounding {
                    lon: 8.6,
                    lat: 63.8,
                    depth: 41.5
                },
                Sounding {
                    lon: 8.7,
                    lat: 63.9,
                    depth: 12.0
                },
            ]
        );
        let err = |t: &str| parse_soundings_str(t).unwrap_err().to_string();
        assert!(err("lon,lat\n8,63\n").contains("no `depth_m` column"));
        assert!(err("lon,lat,depth_m\n8,63\n").contains("line 2: 2 fields"));
        assert!(err("lon,lat,depth_m\n8,63,deep\n").contains("`depth_m` is not a number"));
        assert!(err("# only\n").contains("no header"));
    }

    #[test]
    fn the_spherical_correlation_falls_to_zero_at_the_range() {
        assert_eq!(spherical_correlation(0.0, 800.0), 1.0);
        assert!((spherical_correlation(400.0, 800.0) - 0.3125).abs() < 1e-15);
        assert_eq!(spherical_correlation(800.0, 800.0), 0.0);
        assert_eq!(spherical_correlation(5000.0, 800.0), 0.0);
    }

    /// With little noise the estimate goes through each sample, and it is 0
    /// beyond the range of every sample.
    #[test]
    fn kriging_honours_the_samples_and_vanishes_beyond_the_range() {
        let points = [[0.0, 0.0], [300.0, 0.0], [0.0, 250.0], [5000.0, 0.0]];
        let values = [-20.0, -12.0, -25.0, 8.0];
        let k = SimpleKriging::new(&points, &values, 800.0, 1e-4);
        for (p, v) in points.iter().zip(values) {
            assert!((k.estimate(*p) - v).abs() < 1e-3, "{}", k.estimate(*p));
        }
        assert_eq!(k.estimate([2500.0, 0.0]), 0.0);
        assert_eq!(k.estimate([0.0, -1100.0]), 0.0);
        // Between the first two, the estimate lies between their values
        let mid = k.estimate([150.0, 0.0]);
        assert!(mid < -12.0 && mid > -25.0, "{mid}");
    }

    /// Two samples at one point that disagree: the estimate there is their
    /// mean, shrunk towards 0 by the noise.
    #[test]
    fn coincident_samples_average_under_noise() {
        let points = [[0.0, 0.0], [0.0, 0.0]];
        let k = SimpleKriging::new(&points, &[10.0, 20.0], 500.0, 2.0);
        // σ² = 250 − 4; the estimate is 2σ²/(2σ² + τ²) · 15
        let s2 = 246.0;
        let expected = 2.0 * s2 / (2.0 * s2 + 4.0) * 15.0;
        assert!((k.estimate([0.0, 0.0]) - expected).abs() < 1e-12);
        assert!((k.signal_variance() - s2).abs() < 1e-12);
    }

    /// A dense survey of one offset gives the offset inside it, between
    /// the stations as at them, and a station off the others' pattern pulls
    /// the field towards it only near itself.
    #[test]
    fn a_dense_survey_of_one_offset_reproduces_it_inside() {
        let mut points = Vec::new();
        for i in 0..7 {
            for j in 0..7 {
                points.push([100.0 * i as f64, 100.0 * j as f64]);
            }
        }
        let mut values = vec![25.0; points.len()];
        let k = SimpleKriging::new(&points, &values, 800.0, 3.0);
        for p in [[300.0, 300.0], [350.0, 350.0], [150.0, 420.0]] {
            let e = k.estimate(p);
            assert!((e - 25.0).abs() < 0.01 * 25.0, "{e} at {p:?}");
        }
        assert_eq!(k.estimate([300.0, 1400.0]), 0.0);
        // One station 10 m off at the centre (index 3·7 + 3)
        values[24] = 35.0;
        let k = SimpleKriging::new(&points, &values, 800.0, 3.0);
        let (at, beside) = (k.estimate([300.0, 300.0]), k.estimate([500.0, 300.0]));
        assert!(at > 30.0 && at < 35.0, "{at}");
        assert!((beside - 25.0).abs() < 0.5, "{beside}");
    }

    /// Residuals smaller than the noise are noise: no correction.
    #[test]
    fn residuals_within_the_noise_correct_nothing() {
        let k = SimpleKriging::new(&[[0.0, 0.0], [50.0, 0.0]], &[1.0, -2.0], 400.0, 3.0);
        assert_eq!(k.signal_variance(), 0.0);
        assert_eq!(k.estimate([0.0, 0.0]), 0.0);
    }

    /// Stations farther than the range apart are solved in separate blocks.
    #[test]
    fn clusters_split_by_the_range() {
        // 0–1–2 and 3–4: the link 2–3 is cut
        let groups = clusters(5, |i, j| i == j + 1 && i != 3);
        assert_eq!(groups, vec![vec![0, 1, 2], vec![3, 4]]);
        let groups = clusters(4, |i, j| i + j == 3);
        assert_eq!(groups, vec![vec![0, 3], vec![1, 2]]);
    }

    #[test]
    fn cholesky_solves_a_spd_system() {
        let mut a = vec![4.0, 2.0, 0.6, 2.0, 5.0, 1.0, 0.6, 1.0, 3.0];
        let x = [1.0, -2.0, 0.5];
        let mut b: Vec<f64> = (0..3)
            .map(|i| (0..3).map(|j| a[i * 3 + j] * x[j]).sum())
            .collect();
        cholesky_solve(&mut a, &mut b, 3);
        for (got, want) in b.iter().zip(x) {
            assert!((got - want).abs() < 1e-14);
        }
    }
}

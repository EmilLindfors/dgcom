//! Spatially varying boundary tides from a tidal atlas.
//!
//! A [`TidalAtlas`] holds harmonic constants at geographic points: for each
//! constituent the amplitude and Greenwich phase lag `G` of the surface
//! elevation and, optionally, of the depth-averaged eastward and northward
//! velocity. They are *reference* constants (nodal modulation removed,
//! phases relative to the equilibrium tide at Greenwich), as in TPXO, FES or a
//! harmonic analysis of NorKyst-800 output
//! (`examples/norkyst_boundary_tides.rs`).
//!
//! [`TidalAtlas::boundary_tides`] projects the atlas once onto the boundary
//! nodes of a mesh, applying the nodal corrections of a
//! [`ModelClock`], and returns [`BoundaryTides`], the
//! [`ExternalStateProvider`] of a [`CharacteristicOBC`](super::CharacteristicOBC):
//!
//! ```text
//! η(x, t) = R(t) Σⱼ fⱼ Hⱼ(x) cos(ωⱼ t + V₀ⱼ + uⱼ − Gⱼ(x))      (same for u, v)
//! ```
//!
//! # Spatial interpolation
//!
//! The complex amplitudes `H e^{−iG}` (not `H` and `G` separately, which would
//! break at the 0/360° wrap) are interpolated with inverse-distance weights
//! (power 2) from the [`NEIGHBOURS`] nearest atlas points. A node farther than
//! the coverage radius from every atlas point is an error: the atlas must
//! cover the open boundary.
//!
//! Velocities are rotated from east/north into the mesh axes using the
//! projection's local grid convergence.
//!
//! # File format
//!
//! Whitespace-separated text, `#` comments:
//!
//! ```text
//! # lon lat depth name eta_amp eta_lag [u_amp u_lag v_amp v_lag]
//! 8.0125 63.6031 212.0 M2 0.8123 285.31 0.0412 318.2 0.0233 12.9
//! 8.0125 63.6031 212.0 Z0 0.0214 0
//! ```
//!
//! Amplitudes in m and m/s, lags in degrees, `depth` the source model's
//! still-water depth (m, `nan` if unknown). `Z0` is the mean level of the
//! source over the analysed record (used only on request, see
//! [`BoundaryTides::with_mean_level`]). Lines of one point must be
//! consecutive and list the same constituents at every point.

use std::f64::consts::PI;
use std::fmt::Write as _;
use std::path::Path;

use thiserror::Error;

use super::BCContext2D;
use super::characteristic::{ExternalState, ExternalStateProvider};
use super::harmonic_tide::tidal_ramp;
use crate::io::CoordinateProjection;
use crate::mesh::{BoundaryTag, Mesh2D};
use crate::operators::DGOperators2D;
use crate::tides::{NodalDrift, canonical_name, constituent_period};
use crate::time::ModelClock;
use crate::types::ElementIndex;

/// Number of nearest atlas points a boundary node interpolates from.
pub const NEIGHBOURS: usize = 4;

/// Errors reading or applying a tidal atlas.
#[derive(Debug, Error)]
pub enum TidalAtlasError {
    /// File I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// Malformed line.
    #[error("tidal atlas line {line}: {message}")]
    Parse {
        /// 1-based line number
        line: usize,
        /// What is wrong
        message: String,
    },
    /// Points with different constituent sets, or no points.
    #[error("tidal atlas: {0}")]
    Inconsistent(String),
    /// A boundary node too far from every atlas point.
    #[error(
        "boundary node at (x, y) = ({x:.0}, {y:.0}) m (lat {lat:.4}, lon {lon:.4}) is {distance:.0} m \
         from the nearest atlas point (coverage radius {radius:.0} m)"
    )]
    NotCovered {
        /// Projected position (m)
        x: f64,
        /// Projected position (m)
        y: f64,
        /// Latitude (°)
        lat: f64,
        /// Longitude (°)
        lon: f64,
        /// Distance to the nearest atlas point (m)
        distance: f64,
        /// Allowed distance (m)
        radius: f64,
    },
}

/// Amplitude and Greenwich phase lag (degrees) of one quantity.
pub type Harmonic = (f64, f64);

/// Reference constants of one constituent at one point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AtlasConstituent {
    /// Constituent name (see [`crate::tides::CONSTITUENT_NAMES`]).
    pub name: &'static str,
    /// Surface elevation (m, °).
    pub eta: Harmonic,
    /// Depth-averaged eastward and northward velocity (m/s, °), if known.
    pub velocity: Option<(Harmonic, Harmonic)>,
}

/// Harmonic constants at one geographic point.
#[derive(Clone, Debug, PartialEq)]
pub struct AtlasPoint {
    /// Longitude (°E).
    pub lon: f64,
    /// Latitude (°N).
    pub lat: f64,
    /// Source-model still-water depth (m), NaN if unknown.
    pub depth: f64,
    /// Mean level `Z0` of the source over the analysed record (m), if given.
    pub mean: Option<f64>,
    /// Constituents, in the atlas order.
    pub constituents: Vec<AtlasConstituent>,
}

/// Harmonic constants at a set of points; see the module docs.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TidalAtlas {
    /// Points, each with the same constituents.
    pub points: Vec<AtlasPoint>,
    /// Free-text provenance (the file's leading comment lines).
    pub header: Vec<String>,
}

impl TidalAtlas {
    /// Read an atlas file.
    pub fn read(path: &Path) -> Result<Self, TidalAtlasError> {
        Self::parse(&std::fs::read_to_string(path)?)
    }

    /// Parse the text format of the module docs.
    pub fn parse(text: &str) -> Result<Self, TidalAtlasError> {
        let mut atlas = Self::default();
        for (i, raw) in text.lines().enumerate() {
            let line = raw.trim();
            if let Some(comment) = line.strip_prefix('#') {
                if atlas.points.is_empty() {
                    atlas.header.push(comment.trim().to_string());
                }
                continue;
            }
            if line.is_empty() {
                continue;
            }
            let err = |message: String| TidalAtlasError::Parse {
                line: i + 1,
                message,
            };
            let fields: Vec<&str> = line.split_whitespace().collect();
            if !matches!(fields.len(), 6 | 10) {
                return Err(err(format!(
                    "expected 6 or 10 columns, found {}",
                    fields.len()
                )));
            }
            let num = |k: usize| -> Result<f64, TidalAtlasError> {
                fields[k]
                    .parse::<f64>()
                    .map_err(|_| err(format!("bad number {:?}", fields[k])))
            };
            let (lon, lat, depth) = (num(0)?, num(1)?, num(2)?);
            let new_point = atlas
                .points
                .last()
                .is_none_or(|p| p.lon != lon || p.lat != lat);
            if new_point {
                atlas.points.push(AtlasPoint {
                    lon,
                    lat,
                    depth,
                    mean: None,
                    constituents: Vec::new(),
                });
            }
            let point = atlas.points.last_mut().expect("pushed above");
            if fields[3].eq_ignore_ascii_case("Z0") {
                point.mean = Some(num(4)?);
                continue;
            }
            let name = canonical_name(fields[3])
                .ok_or_else(|| err(format!("unsupported constituent {}", fields[3])))?;
            let velocity = if fields.len() == 10 {
                Some(((num(6)?, num(7)?), (num(8)?, num(9)?)))
            } else {
                None
            };
            point.constituents.push(AtlasConstituent {
                name,
                eta: (num(4)?, num(5)?),
                velocity,
            });
        }
        atlas.validate()?;
        Ok(atlas)
    }

    /// Check that there are points and that all carry the same constituents,
    /// with velocity everywhere or nowhere.
    pub fn validate(&self) -> Result<(), TidalAtlasError> {
        let first = self
            .points
            .first()
            .ok_or_else(|| TidalAtlasError::Inconsistent("no points".into()))?;
        let signature = |p: &AtlasPoint| {
            p.constituents
                .iter()
                .map(|c| (c.name, c.velocity.is_some()))
                .collect::<Vec<_>>()
        };
        let expected = signature(first);
        if expected.is_empty() {
            return Err(TidalAtlasError::Inconsistent("no constituents".into()));
        }
        if expected.iter().any(|&(_, v)| v != expected[0].1) {
            return Err(TidalAtlasError::Inconsistent(
                "velocity given for some constituents only".into(),
            ));
        }
        for p in &self.points {
            if signature(p) != expected {
                return Err(TidalAtlasError::Inconsistent(format!(
                    "point ({}, {}) has constituents {:?}, expected {:?}",
                    p.lon,
                    p.lat,
                    signature(p),
                    expected
                )));
            }
        }
        Ok(())
    }

    /// Constituent names, in order.
    pub fn names(&self) -> Vec<&'static str> {
        self.points
            .first()
            .map(|p| p.constituents.iter().map(|c| c.name).collect())
            .unwrap_or_default()
    }

    /// Replace (or add) constituent `name` at every point by `from` scaled by
    /// `ratio` and lagged by `lag_offset_deg`, for elevation and velocity
    /// alike: inference, for a constituent the source misrepresents but
    /// whose ratio to a neighbour in frequency is known (e.g. from a long
    /// gauge record). Constituents close in frequency share their spatial
    /// structure, so one ratio serves a small domain. With `from == name`
    /// it is a complex gain on the constituent itself: calibration of one
    /// the source has the right structure for but the wrong amplitude or
    /// phase (e.g. its ratio to a gauge where both are known).
    pub fn infer(
        &mut self,
        name: &str,
        from: &str,
        ratio: f64,
        lag_offset_deg: f64,
    ) -> Result<(), TidalAtlasError> {
        let name = canonical_name(name)
            .ok_or_else(|| TidalAtlasError::Inconsistent(format!("unknown constituent {name}")))?;
        let scaled = |(amp, lag): Harmonic| (ratio * amp, (lag + lag_offset_deg).rem_euclid(360.0));
        for p in &mut self.points {
            let source = p
                .constituents
                .iter()
                .find(|c| c.name.eq_ignore_ascii_case(from))
                .ok_or_else(|| {
                    TidalAtlasError::Inconsistent(format!(
                        "{name} is inferred from {from}, which is missing"
                    ))
                })?;
            let inferred = AtlasConstituent {
                name,
                eta: scaled(source.eta),
                velocity: source.velocity.map(|(u, v)| (scaled(u), scaled(v))),
            };
            match p.constituents.iter_mut().find(|c| c.name == name) {
                Some(c) => *c = inferred,
                None => p.constituents.push(inferred),
            }
        }
        Ok(())
    }

    /// The point nearest to (`lon`, `lat`) and its distance (m, spherical
    /// earth); `None` for an empty atlas.
    pub fn nearest(&self, lon: f64, lat: f64) -> Option<(&AtlasPoint, f64)> {
        const EARTH_RADIUS: f64 = 6_371_000.0;
        let distance = |p: &AtlasPoint| {
            let (la1, la2) = (lat.to_radians(), p.lat.to_radians());
            let dlon = (p.lon - lon).to_radians();
            let c = la1.sin() * la2.sin() + la1.cos() * la2.cos() * dlon.cos();
            EARTH_RADIUS * c.clamp(-1.0, 1.0).acos()
        };
        self.points
            .iter()
            .map(|p| (p, distance(p)))
            .min_by(|a, b| a.1.total_cmp(&b.1))
    }

    /// Whether the atlas has velocities.
    pub fn has_velocity(&self) -> bool {
        self.points
            .first()
            .and_then(|p| p.constituents.first())
            .is_some_and(|c| c.velocity.is_some())
    }

    /// The text format of the module docs.
    pub fn to_text(&self) -> String {
        let mut out = String::new();
        for line in &self.header {
            let _ = writeln!(out, "# {line}");
        }
        let _ = writeln!(
            out,
            "# lon lat depth name eta_amp eta_lag [u_amp u_lag v_amp v_lag]"
        );
        for p in &self.points {
            let head = format!("{:.5} {:.5} {:.1}", p.lon, p.lat, p.depth);
            if let Some(mean) = p.mean {
                let _ = writeln!(out, "{head} Z0 {mean:.5} 0");
            }
            for c in &p.constituents {
                let _ = write!(out, "{head} {} {:.5} {:.2}", c.name, c.eta.0, c.eta.1);
                if let Some(((ua, ug), (va, vg))) = c.velocity {
                    let _ = write!(out, " {ua:.5} {ug:.2} {va:.5} {vg:.2}");
                }
                out.push('\n');
            }
        }
        out
    }

    /// Write the atlas to `path`.
    pub fn write(&self, path: &Path) -> Result<(), TidalAtlasError> {
        std::fs::write(path, self.to_text())?;
        Ok(())
    }

    /// Project the atlas onto the boundary nodes of `mesh` whose faces carry
    /// `tag`, for a run on `clock` (`V₀` at its epoch, the nodal `f`, `u` at
    /// each time, [`ModelClock::nodal_drift`]).
    ///
    /// Every such node must lie within `coverage_radius` (m) of an atlas point.
    pub fn boundary_tides<P: CoordinateProjection>(
        &self,
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        projection: &P,
        tag: BoundaryTag,
        clock: &ModelClock,
        coverage_radius: f64,
    ) -> Result<BoundaryTides, TidalAtlasError> {
        let n_nodes = ops.n_nodes;
        let mut slot_of_node = vec![u32::MAX; mesh.n_elements * n_nodes];
        let mut positions = Vec::new();
        for k in ElementIndex::iter(mesh.n_elements) {
            for face in 0..4 {
                if mesh.neighbor(k, face).is_some() || mesh.boundary_tag(k, face) != Some(tag) {
                    continue;
                }
                for &node in &ops.face_nodes[face] {
                    let flat = k.as_usize() * n_nodes + node;
                    if slot_of_node[flat] != u32::MAX {
                        continue;
                    }
                    let [x, y] =
                        mesh.reference_to_physical(k, ops.nodes_r[node], ops.nodes_s[node]);
                    slot_of_node[flat] = positions.len() as u32;
                    positions.push((x, y));
                }
            }
        }
        let mut tides = self.tides_at(&positions, projection, clock, coverage_radius)?;
        tides.slot_of_node = slot_of_node;
        Ok(tides)
    }

    /// The atlas at the points `positions` (mesh coordinates), in that order
    /// of slots, for a run on `clock`: as
    /// [`boundary_tides`](Self::boundary_tides), but anywhere (e.g. the
    /// nodes of a nesting band, [`OceanModelState::with_tidal_correction`](crate::boundary::OceanModelState::with_tidal_correction)).
    /// A [`BoundaryTides`] built here finds its slot by position only.
    pub fn tides_at<P: CoordinateProjection>(
        &self,
        positions: &[(f64, f64)],
        projection: &P,
        clock: &ModelClock,
        coverage_radius: f64,
    ) -> Result<BoundaryTides, TidalAtlasError> {
        self.validate()?;
        let names = self.names();
        let velocity = self.has_velocity();
        let drifts: Vec<NodalDrift> = names
            .iter()
            .map(|n| clock.nodal_drift(n).unwrap_or(NodalDrift::IDENTITY))
            .collect();
        // The nodal phase's drift u̇ adds to the angular speed
        let omega: Vec<f64> = names
            .iter()
            .zip(&drifts)
            .map(|(n, d)| {
                2.0 * PI / constituent_period(n).expect("canonical name") + d.u_rate_rad()
            })
            .collect();
        let f_rate: Vec<f64> = drifts.iter().map(NodalDrift::relative_f_rate).collect();
        let xy: Vec<(f64, f64)> = self
            .points
            .iter()
            .map(|p| projection.geo_to_xy(p.lat, p.lon))
            .collect();

        let mut coefficients = Vec::with_capacity(positions.len() * names.len() * 6);
        let mut mean = Vec::with_capacity(positions.len());
        let mut source_depth = Vec::with_capacity(positions.len());
        let mut nearest: Vec<(f64, usize)> = Vec::with_capacity(self.points.len());

        for &(x, y) in positions {
            // Inverse-distance weights of the nearest atlas points
            nearest.clear();
            nearest.extend(
                xy.iter()
                    .enumerate()
                    .map(|(i, &(px, py))| ((px - x).hypot(py - y), i)),
            );
            nearest.sort_by(|a, b| a.0.total_cmp(&b.0));
            nearest.truncate(NEIGHBOURS);
            let d_min = nearest[0].0;
            if d_min > coverage_radius {
                let (lat, lon) = projection.xy_to_geo(x, y);
                return Err(TidalAtlasError::NotCovered {
                    x,
                    y,
                    lat,
                    lon,
                    distance: d_min,
                    radius: coverage_radius,
                });
            }
            let weights: Vec<(f64, usize)> = if d_min < 1.0 {
                vec![(1.0, nearest[0].1)]
            } else {
                let w: Vec<(f64, usize)> =
                    nearest.iter().map(|&(d, i)| (1.0 / (d * d), i)).collect();
                let total: f64 = w.iter().map(|p| p.0).sum();
                w.into_iter().map(|(wi, i)| (wi / total, i)).collect()
            };

            // Rotation from east/north to the mesh axes: the direction of
            // local north in the projected plane
            let (lat, lon) = projection.xy_to_geo(x, y);
            let (nx, ny) = projection.geo_to_xy(lat + 1e-4, lon);
            let theta = (nx - x).atan2(ny - y); // angle of north from +y
            let (sin_t, cos_t) = theta.sin_cos();

            mean.push(
                weights
                    .iter()
                    .map(|&(w, i)| w * self.points[i].mean.unwrap_or(0.0))
                    .sum::<f64>(),
            );
            source_depth.push(
                weights
                    .iter()
                    .map(|&(w, i)| w * self.points[i].depth)
                    .sum::<f64>(),
            );

            for (j, drift) in drifts.iter().enumerate() {
                let n = drift.at_epoch;
                // Complex amplitude H e^{−iG}, interpolated
                let interp = |get: &dyn Fn(&AtlasConstituent) -> Harmonic| {
                    weights.iter().fold((0.0, 0.0), |(re, im), &(w, i)| {
                        let (amp, lag) = get(&self.points[i].constituents[j]);
                        let g = lag.to_radians();
                        (re + w * amp * g.cos(), im - w * amp * g.sin())
                    })
                };
                let eta = interp(&|c| c.eta);
                let (u, v) = if velocity {
                    let east = interp(&|c| c.velocity.expect("validated").0);
                    let north = interp(&|c| c.velocity.expect("validated").1);
                    // Mesh components: u = E cos θ + N sin θ, v = −E sin θ + N cos θ
                    (
                        (
                            east.0 * cos_t + north.0 * sin_t,
                            east.1 * cos_t + north.1 * sin_t,
                        ),
                        (
                            -east.0 * sin_t + north.0 * cos_t,
                            -east.1 * sin_t + north.1 * cos_t,
                        ),
                    )
                } else {
                    ((0.0, 0.0), (0.0, 0.0))
                };
                // f₀ Re[A e^{i(ω′t + V₀ + u₀)}] = C cos ω′t + S sin ω′t
                let (sin_p, cos_p) = n.phase_offset_rad().sin_cos();
                for (re, im) in [eta, u, v] {
                    let (re, im) = (re * cos_p - im * sin_p, re * sin_p + im * cos_p);
                    coefficients.push(n.f * re);
                    coefficients.push(-n.f * im);
                }
            }
        }

        Ok(BoundaryTides {
            names,
            omega,
            f_rate,
            velocity,
            slot_of_node: Vec::new(),
            positions: positions.to_vec(),
            coefficients,
            mean,
            source_depth,
            use_mean: false,
            ramp_duration: None,
            max_transport_ratio: None,
        })
    }

    /// `self − other`, constituent by constituent (complex amplitudes
    /// `H e^{−iG}` of η and velocity) at every point: the tide that turns
    /// `other` into `self`, e.g. an atlas corrected with [`infer`](Self::infer)
    /// minus the raw one. The points must be the same; a constituent missing
    /// from one side counts as zero there. The mean level is the difference
    /// of the means where both have one.
    pub fn difference(&self, other: &Self) -> Result<Self, TidalAtlasError> {
        self.validate()?;
        other.validate()?;
        if self.points.len() != other.points.len() || self.has_velocity() != other.has_velocity() {
            return Err(TidalAtlasError::Inconsistent(
                "difference of atlases with different points or velocity".into(),
            ));
        }
        let complex = |(amp, lag): Harmonic| {
            let g = lag.to_radians();
            (amp * g.cos(), -amp * g.sin())
        };
        let harmonic =
            |(re, im): (f64, f64)| (re.hypot(im), (-im).atan2(re).to_degrees().rem_euclid(360.0));
        let minus = |a: Option<Harmonic>, b: Option<Harmonic>| {
            let (a, b) = (a.map_or((0.0, 0.0), complex), b.map_or((0.0, 0.0), complex));
            harmonic((a.0 - b.0, a.1 - b.1))
        };
        let mut names = self.names();
        for name in other.names() {
            if !names.contains(&name) {
                names.push(name);
            }
        }
        let points = self
            .points
            .iter()
            .zip(&other.points)
            .map(|(p, q)| {
                if (p.lon - q.lon).abs() > 1e-9 || (p.lat - q.lat).abs() > 1e-9 {
                    return Err(TidalAtlasError::Inconsistent(format!(
                        "difference of atlases with different points: ({}, {}) and ({}, {})",
                        p.lon, p.lat, q.lon, q.lat
                    )));
                }
                let find = |point: &AtlasPoint, name| {
                    point.constituents.iter().find(|c| c.name == name).copied()
                };
                let constituents = names
                    .iter()
                    .map(|&name| {
                        let (a, b) = (find(p, name), find(q, name));
                        AtlasConstituent {
                            name,
                            eta: minus(a.map(|c| c.eta), b.map(|c| c.eta)),
                            velocity: self.has_velocity().then(|| {
                                let (ua, va) = (
                                    a.and_then(|c| c.velocity).map(|v| v.0),
                                    a.and_then(|c| c.velocity).map(|v| v.1),
                                );
                                let (ub, vb) = (
                                    b.and_then(|c| c.velocity).map(|v| v.0),
                                    b.and_then(|c| c.velocity).map(|v| v.1),
                                );
                                (minus(ua, ub), minus(va, vb))
                            }),
                        }
                    })
                    .collect();
                Ok(AtlasPoint {
                    constituents,
                    mean: p.mean.zip(q.mean).map(|(a, b)| a - b),
                    ..p.clone()
                })
            })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            points,
            header: vec!["difference of two atlases".into()],
        })
    }
}

/// Atlas tides at the boundary nodes of a mesh: the external state of a
/// characteristic OBC. Built by [`TidalAtlas::boundary_tides`].
#[derive(Clone, Debug)]
pub struct BoundaryTides {
    names: Vec<&'static str>,
    /// Per constituent `ω′ = ω + u̇` (rad/s): the angular speed with the
    /// drift of the nodal phase
    omega: Vec<f64>,
    /// Per constituent `ḟ / f₀` (1/s): the drift of the nodal factor
    f_rate: Vec<f64>,
    velocity: bool,
    /// Slot per flat mesh node (`u32::MAX`: not a forced boundary node)
    slot_of_node: Vec<u32>,
    positions: Vec<(f64, f64)>,
    /// Per slot and constituent: `C, S` of η, u, v at the epoch's nodal
    /// terms (`f(t)/f₀ · (C cos ω′t + S sin ω′t)`)
    coefficients: Vec<f64>,
    mean: Vec<f64>,
    source_depth: Vec<f64>,
    use_mean: bool,
    ramp_duration: Option<f64>,
    /// Largest transport-scaling factor, if the velocity is scaled
    max_transport_ratio: Option<f64>,
}

impl BoundaryTides {
    /// Ramp the tide up over `duration` seconds.
    pub fn with_ramp_up(mut self, duration: f64) -> Self {
        self.ramp_duration = Some(duration);
        self
    }

    /// Scale the velocity to carry the source model's transport over the
    /// child's bed: `u · D_source/D_child` with total depths `D = h + η`
    /// (the source's from the atlas depths, [`Self::source_depths`]),
    /// limited to a factor `max_ratio` either way, as
    /// [`OceanModelState`](super::OceanModelState) nesting does.
    ///
    /// The atlas velocity is the source model's depth mean over its own
    /// depth. Where the child's bed differs (a coastline-fitted bed from a
    /// 50 m elevation model against NorKyst's smoothed 800 m one), imposing
    /// it unscaled imposes a different transport. Off by default.
    pub fn with_transport_scaling(mut self, max_ratio: f64) -> Self {
        assert!(max_ratio >= 1.0, "transport ratio limit below 1");
        self.max_transport_ratio = Some(max_ratio);
        self
    }

    /// Add the atlas mean level `Z0` to the elevation (not ramped).
    ///
    /// Off by default: the mean of a regional model over a short record
    /// includes its datum offset and wind setup.
    pub fn with_mean_level(mut self, enable: bool) -> Self {
        self.use_mean = enable;
        self
    }

    /// Constituent names.
    pub fn names(&self) -> &[&'static str] {
        &self.names
    }

    /// Number of forced boundary nodes.
    pub fn n_nodes(&self) -> usize {
        self.positions.len()
    }

    /// Positions (m) of the forced boundary nodes.
    pub fn positions(&self) -> &[(f64, f64)] {
        &self.positions
    }

    /// Source-model still-water depth at the forced boundary nodes (m).
    pub fn source_depths(&self) -> &[f64] {
        &self.source_depth
    }

    /// Whether the tides carry velocities.
    pub fn has_velocity(&self) -> bool {
        self.velocity
    }

    /// Amplitude (m, `f H` at t = 0) and Greenwich-referenced phase (°,
    /// `V₀ + u − G` at t = 0)
    /// of constituent `j`'s elevation at forced node `slot`.
    pub fn elevation_constants(&self, slot: usize, j: usize) -> (f64, f64) {
        let base = (slot * self.names.len() + j) * 6;
        let (c, s) = (self.coefficients[base], self.coefficients[base + 1]);
        (c.hypot(s), (-s).atan2(c).to_degrees())
    }

    /// `(η, u, v)` at forced node `slot` and time `t`, with the nodal terms
    /// of `t`.
    pub fn evaluate(&self, slot: usize, t: f64) -> (f64, f64, f64) {
        let n_c = self.names.len();
        let coefficients = &self.coefficients[slot * n_c * 6..][..n_c * 6];
        let mut sum = [0.0; 3];
        for (j, c) in coefficients.as_chunks::<6>().0.iter().enumerate() {
            let (sin, cos) = (self.omega[j] * t).sin_cos();
            let f = 1.0 + self.f_rate[j] * t;
            for (q, s) in sum.iter_mut().enumerate() {
                *s += f * (c[2 * q] * cos + c[2 * q + 1] * sin);
            }
        }
        let ramp = tidal_ramp(t, self.ramp_duration);
        let mean = if self.use_mean { self.mean[slot] } else { 0.0 };
        (mean + ramp * sum[0], ramp * sum[1], ramp * sum[2])
    }

    /// Slot of the node of `ctx`: by nodal index, or else the nearest forced
    /// node by position.
    fn slot(&self, ctx: &BCContext2D) -> usize {
        if let Some(slot) = ctx
            .node_index
            .and_then(|i| self.slot_of_node.get(i))
            .filter(|&&s| s != u32::MAX)
        {
            return *slot as usize;
        }
        let (x, y) = ctx.position;
        self.positions
            .iter()
            .enumerate()
            .min_by(|a, b| {
                let da = (a.1.0 - x).hypot(a.1.1 - y);
                let db = (b.1.0 - x).hypot(b.1.1 - y);
                da.total_cmp(&db)
            })
            .map(|(i, _)| i)
            .expect("BoundaryTides has no forced nodes")
    }
}

impl ExternalStateProvider for BoundaryTides {
    fn external_state(&self, ctx: &BCContext2D) -> ExternalState {
        let slot = self.slot(ctx);
        let (eta, mut u, mut v) = self.evaluate(slot, ctx.time);
        if !self.velocity {
            return ExternalState::elevation(eta);
        }
        if let Some(max) = self.max_transport_ratio {
            let h_source = self.source_depth[slot];
            let (d_source, d_child) = (h_source + eta, eta - ctx.bathymetry);
            if h_source > 0.0 && d_source > 0.0 && d_child > 0.0 {
                let ratio = (d_source / d_child).clamp(1.0 / max, max);
                u *= ratio;
                v *= ratio;
            }
        }
        ExternalState::new(eta, u, v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::LocalProjection;
    use crate::solver::SWEState2D;

    const ATLAS: &str = "\
# test atlas
# lon lat depth name eta_amp eta_lag u_amp u_lag v_amp v_lag
8.0 63.0 100.0 Z0 0.05 0
8.0 63.0 100.0 M2 1.0 350.0 0.1 80.0 0.2 170.0
8.0 63.0 100.0 S2 0.3 20.0 0.0 0.0 0.0 0.0
8.1 63.0 120.0 Z0 0.05 0
8.1 63.0 120.0 M2 1.0 10.0 0.1 100.0 0.2 190.0
8.1 63.0 120.0 S2 0.3 40.0 0.0 0.0 0.0 0.0
";

    #[test]
    fn parse_and_write_round_trip() {
        let atlas = TidalAtlas::parse(ATLAS).unwrap();
        assert_eq!(atlas.points.len(), 2);
        assert_eq!(atlas.names(), vec!["M2", "S2"]);
        assert!(atlas.has_velocity());
        assert_eq!(atlas.points[1].mean, Some(0.05));
        assert_eq!(atlas.header[0], "test atlas");
        let again = TidalAtlas::parse(&atlas.to_text()).unwrap();
        assert_eq!(again.points, atlas.points);
    }

    /// Inference replaces a constituent (or adds a new one) at every point,
    /// velocities included, and needs its source.
    #[test]
    fn inference_replaces_and_adds_constituents() {
        let mut atlas = TidalAtlas::parse(ATLAS).unwrap();
        atlas.infer("S2", "M2", 0.2, -30.0).unwrap();
        let s2 = atlas.points[0].constituents[1];
        assert_eq!(s2.name, "S2");
        assert!((s2.eta.0 - 0.2).abs() < 1e-12 && (s2.eta.1 - 320.0).abs() < 1e-12);
        let ((ua, ug), _) = s2.velocity.unwrap();
        assert!((ua - 0.02).abs() < 1e-12 && (ug - 50.0).abs() < 1e-12);
        // Wraps past 360°
        assert!((atlas.points[1].constituents[1].eta.1 - 340.0).abs() < 1e-12);

        atlas.infer("n2", "M2", 0.19, 0.0).unwrap();
        assert_eq!(atlas.names(), vec!["M2", "S2", "N2"]);
        atlas.validate().unwrap();
        assert!(atlas.infer("Q1", "O1", 0.19, 0.0).is_err());

        // From itself: a complex gain, in place, velocities included
        atlas.infer("M2", "M2", 1.1, -15.0).unwrap();
        let m2 = atlas.points[1].constituents[0];
        assert_eq!(atlas.names(), vec!["M2", "S2", "N2"]);
        assert!((m2.eta.0 - 1.1).abs() < 1e-12 && (m2.eta.1 - 355.0).abs() < 1e-12);
        let ((ua, ug), (va, vg)) = m2.velocity.unwrap();
        assert!((ua - 0.11).abs() < 1e-12 && (ug - 85.0).abs() < 1e-12);
        assert!((va - 0.22).abs() < 1e-12 && (vg - 175.0).abs() < 1e-12);
    }

    /// Corrected minus raw is the correction alone: zero where nothing
    /// changed, the replaced constituent's complex difference where it did,
    /// and a constituent only one side has, whole (negated for the other).
    #[test]
    fn difference_of_corrected_and_raw_atlas() {
        let raw = TidalAtlas::parse(ATLAS).unwrap();
        let mut corrected = raw.clone();
        corrected.infer("S2", "M2", 0.3, 20.0).unwrap(); // S2: 0.3 at 10°/30°
        corrected.infer("N2", "M2", 0.2, 0.0).unwrap(); // new
        let d = corrected.difference(&raw).unwrap();
        assert_eq!(d.names(), vec!["M2", "S2", "N2"]);
        for (p, (c, r)) in d
            .points
            .iter()
            .zip(corrected.points.iter().zip(&raw.points))
        {
            assert_eq!(p.constituents[0].eta.0, 0.0);
            assert_eq!(p.mean, Some(0.0));
            // S2: 0.3 e^{−i(G+20)} − 0.3 e^{−iG'} with the raw S2 lag G'
            let complex = |(a, g): Harmonic| (a * g.to_radians().cos(), -a * g.to_radians().sin());
            let (a, b) = (
                complex(c.constituents[1].eta),
                complex(r.constituents[1].eta),
            );
            let s2 = complex(p.constituents[1].eta);
            assert!((s2.0 - (a.0 - b.0)).abs() < 1e-12 && (s2.1 - (a.1 - b.1)).abs() < 1e-12);
            let (n2, cn2) = (p.constituents[2], c.constituents[2]);
            assert!((n2.eta.0 - cn2.eta.0).abs() < 1e-12 && (n2.eta.1 - cn2.eta.1).abs() < 1e-9);
        }
        let back = raw.difference(&corrected).unwrap();
        let n2 = back.points[0].constituents[2];
        assert!((n2.eta.0 - 0.2).abs() < 1e-12);
        assert!(
            (n2.eta.1 - (corrected.points[0].constituents[2].eta.1 + 180.0).rem_euclid(360.0))
                .abs()
                < 1e-9
        );
        let mut moved = raw.clone();
        moved.points[1].lon += 0.01;
        assert!(corrected.difference(&moved).is_err());
    }

    /// 0.1° of longitude at 63° N is ≈ 5.05 km.
    #[test]
    fn nearest_point_and_distance() {
        let atlas = TidalAtlas::parse(ATLAS).unwrap();
        let (p, d) = atlas.nearest(8.07, 63.0).unwrap();
        assert_eq!(p.lon, 8.1);
        let expected = 6_371_000.0 * 0.03_f64.to_radians() * 63.0_f64.to_radians().cos();
        assert!((d - expected).abs() < 1.0, "{d} vs {expected}");
        assert!(TidalAtlas::default().nearest(8.0, 63.0).is_none());
    }

    #[test]
    fn inconsistent_constituents_are_rejected() {
        let bad = "8.0 63.0 1 M2 1 0\n8.1 63.0 1 S2 1 0\n";
        assert!(matches!(
            TidalAtlas::parse(bad),
            Err(TidalAtlasError::Inconsistent(_))
        ));
        assert!(matches!(
            TidalAtlas::parse("8.0 63.0 1 XX9 1 0\n"),
            Err(TidalAtlasError::Parse { line: 1, .. })
        ));
    }

    /// Mesh whose south side runs between the two atlas points.
    fn setup() -> (Mesh2D, DGOperators2D, LocalProjection) {
        let projection = LocalProjection::new(63.0, 8.05);
        let (x0, y0) = projection.geo_to_xy(63.0, 8.0);
        let (x1, _) = projection.geo_to_xy(63.0, 8.1);
        let mesh = Mesh2D::uniform_rectangle_with_sides(
            x0,
            x1,
            y0,
            y0 + 2000.0,
            4,
            2,
            [
                BoundaryTag::Open,
                BoundaryTag::Wall,
                BoundaryTag::Wall,
                BoundaryTag::Wall,
            ],
        );
        (mesh, DGOperators2D::new(2), projection)
    }

    /// At an atlas point the tide is the atlas constituent, nodal-corrected;
    /// midway the phase is interpolated across the 0/360° wrap.
    #[test]
    fn boundary_tides_reproduce_and_interpolate_the_atlas() {
        let atlas = TidalAtlas::parse(ATLAS).unwrap();
        let (mesh, ops, projection) = setup();
        let clock = ModelClock::at_datetime(2025, 6, 1, 0, 0);
        let tides = atlas
            .boundary_tides(&mesh, &ops, &projection, BoundaryTag::Open, &clock, 3000.0)
            .unwrap();
        // 4 elements × 3 face nodes (DG nodes are not shared between elements)
        assert_eq!(tides.n_nodes(), 12);
        let m2 = clock.nodal_correction("M2", 0.0).unwrap();

        let (x_west, _) = projection.geo_to_xy(63.0, 8.0);
        let west = tides
            .positions()
            .iter()
            .position(|p| (p.0 - x_west).abs() < 1e-6)
            .unwrap();
        let (amp, phase) = tides.elevation_constants(west, 0);
        assert!((amp - m2.f).abs() < 1e-12);
        let expected = (m2.v0_deg + m2.u_deg - 350.0).rem_euclid(360.0);
        assert!((phase.rem_euclid(360.0) - expected).abs() < 1e-9);

        // Mid-boundary node: equal weights, G = 0° (not 180°), |H| = cos 10°
        let x_mid = 0.5 * (x_west + projection.geo_to_xy(63.0, 8.1).0);
        let mid = tides
            .positions()
            .iter()
            .position(|p| (p.0 - x_mid).abs() < 1e-6)
            .unwrap();
        let (amp, phase) = tides.elevation_constants(mid, 0);
        assert!((amp - m2.f * 10f64.to_radians().cos()).abs() < 1e-3);
        let expected = (m2.v0_deg + m2.u_deg).rem_euclid(360.0);
        assert!((phase.rem_euclid(360.0) - expected).abs() < 0.5);

        // Mean level only on request; ramp scales the tide
        let t = 3600.0;
        let (eta, _, _) = tides.evaluate(mid, t);
        let tides = tides.with_mean_level(true).with_ramp_up(2.0 * t);
        let (eta_ramped, _, _) = tides.evaluate(mid, t);
        assert!((eta_ramped - (0.05 + 0.5 * eta)).abs() < 1e-12);
    }

    /// The nodal terms are those of the time the tide is evaluated at, not
    /// of a run's middle (TODO P1.5: a run resumed to another end was forced
    /// differently): 20 days in, M2 at the atlas point is `f`, `u` at 20 days.
    #[test]
    fn boundary_tides_take_the_nodal_terms_of_the_time() {
        let atlas = TidalAtlas::parse(ATLAS).unwrap();
        let (mesh, ops, projection) = setup();
        let clock = ModelClock::at_datetime(2025, 6, 1, 0, 0);
        let tides = atlas
            .boundary_tides(&mesh, &ops, &projection, BoundaryTag::Open, &clock, 3000.0)
            .unwrap();
        let (x_west, _) = projection.geo_to_xy(63.0, 8.0);
        let west = tides
            .positions()
            .iter()
            .position(|p| (p.0 - x_west).abs() < 1e-6)
            .unwrap();
        let constituent = |name: &str, amp: f64, lag: f64, t: f64| {
            let n = clock.nodal_correction(name, t).unwrap();
            let omega = 2.0 * PI / constituent_period(name).unwrap();
            n.f * amp * (omega * t + n.phase_offset_rad() - lag.to_radians()).cos()
        };
        for t in [0.0, 20.0 * 86_400.0 + 4321.0] {
            let (eta, u, _) = tides.evaluate(west, t);
            let expected = constituent("M2", 1.0, 350.0, t) + constituent("S2", 0.3, 20.0, t);
            // The linear drift is at most 6e-6 m off; held at the epoch, 7e-4 m
            assert!(
                (eta - expected).abs() < 2e-5,
                "η at {t}: {eta} vs {expected}"
            );
            let expected = constituent("M2", 0.1, 80.0, t);
            assert!((u - expected).abs() < 2e-6, "u at {t}: {u} vs {expected}");
        }
    }

    /// With transport scaling the velocity carries the source's transport
    /// over the child's bed, limited by the ratio; without it, or without
    /// source depth, it is the atlas velocity.
    #[test]
    fn transport_scaling_conserves_the_source_transport() {
        let atlas = TidalAtlas::parse(ATLAS).unwrap();
        let (mesh, ops, projection) = setup();
        let clock = ModelClock::default();
        let tides = atlas
            .boundary_tides(&mesh, &ops, &projection, BoundaryTag::Open, &clock, 3000.0)
            .unwrap();
        let (x_west, _) = projection.geo_to_xy(63.0, 8.0);
        let west = tides
            .positions()
            .iter()
            .position(|p| (p.0 - x_west).abs() < 1e-6)
            .unwrap();
        assert_eq!(tides.source_depths()[west], 100.0);
        let t = 5000.0;
        let (eta, u, v) = tides.evaluate(west, t);
        assert!(u.abs() > 1e-3 && v.abs() > 1e-3);
        let velocity = |tides: &BoundaryTides, bed: f64| {
            let ctx = BCContext2D::new(
                t,
                tides.positions()[west],
                SWEState2D::new(eta - bed, 0.0, 0.0),
                bed,
                (-1.0, 0.0),
                9.81,
                1e-6,
            );
            let state = tides.external_state(&ctx);
            assert_eq!(state.eta, eta);
            state.velocity.unwrap()
        };
        // Unscaled: the atlas velocity whatever the bed
        assert_eq!(velocity(&tides, -40.0), (u, v));
        let scaled = tides.clone().with_transport_scaling(3.0);
        // A child half as deep (in total depth) carries the source's
        // transport at twice the speed
        let d_source = 100.0 + eta;
        let (us, vs) = velocity(&scaled, eta - 0.5 * d_source);
        assert!((us - 2.0 * u).abs() < 1e-12 && (vs - 2.0 * v).abs() < 1e-12);
        // Limited to the ratio: a 10 m child under a 100 m source
        let (ul, vl) = velocity(&scaled, -10.0);
        assert!((ul - 3.0 * u).abs() < 1e-12 && (vl - 3.0 * v).abs() < 1e-12);
        // The same depth: unchanged
        assert_eq!(velocity(&scaled, -100.0), (u, v));
    }

    #[test]
    fn uncovered_boundary_is_an_error() {
        let atlas = TidalAtlas::parse(ATLAS).unwrap();
        let (mesh, ops, projection) = setup();
        let clock = ModelClock::default();
        let result =
            atlas.boundary_tides(&mesh, &ops, &projection, BoundaryTag::Open, &clock, 100.0);
        assert!(matches!(result, Err(TidalAtlasError::NotCovered { .. })));
    }

    /// Velocities: with a local projection north is +y, so u = east.
    #[test]
    fn velocity_is_rotated_into_mesh_axes() {
        let atlas = TidalAtlas::parse(ATLAS).unwrap();
        let (mesh, ops, projection) = setup();
        let clock = ModelClock::default();
        let tides = atlas
            .boundary_tides(&mesh, &ops, &projection, BoundaryTag::Open, &clock, 3000.0)
            .unwrap();
        let (x_west, _) = projection.geo_to_xy(63.0, 8.0);
        let west = tides
            .positions()
            .iter()
            .position(|p| (p.0 - x_west).abs() < 1e-6)
            .unwrap();
        let m2 = clock.nodal_correction("M2", 0.0).unwrap();
        let s2 = clock.nodal_correction("S2", 0.0).unwrap();
        let (_, u, v) = tides.evaluate(west, 0.0);
        let phase = m2.v0_deg + m2.u_deg;
        let u_expected = m2.f * 0.1 * (phase - 80.0).to_radians().cos();
        let v_expected = m2.f * 0.2 * (phase - 170.0).to_radians().cos();
        // S2 has zero velocity amplitude
        let _ = s2;
        assert!((u - u_expected).abs() < 1e-6, "{u} vs {u_expected}");
        assert!((v - v_expected).abs() < 1e-6, "{v} vs {v_expected}");
    }
}

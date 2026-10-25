//! Fish farm sites and their cage layout (TODO F.1).
//!
//! A farm site file holds what Fiskeridirektoratet publishes about a site,
//! as `scripts/fiskeridir_site.sh <site number>` fetches it. That is the
//! register entry (Akvakulturregisteret: name, position, capacity) and the
//! installation's certified geometry (NYTEK: the boundary polygon of the
//! mooring frame and the mooring lines):
//!
//! ```text
//! # Coordinates are longitude latitude (WGS84, degrees).
//! site 14042 KATTHOLMEN
//! position 8.677683 63.867683
//! capacity 7800.0 TN
//! boundary 8.6771000 63.8726167
//! boundary 8.6808167 63.8726000
//! boundary 8.6808167 63.8693333
//! boundary 8.6772667 63.8693667
//! mooring 8.6808167 63.8726000 8.6856500 63.8724667 farm
//! cage 8.6780 63.8720 25.5 20
//! ```
//!
//! `boundary` lines are the polygon's vertices in order. `mooring` lines run
//! from the frame (or the feed raft, `raft`) to the anchor. `cage` lines
//! (longitude, latitude, and optionally radius and net depth in m) give the
//! cages where they are known. Neither public source has them.
//!
//! # The cage layout
//!
//! [`FarmSite::cage_layout`] gives the site's cages as [`NetCage`]s in mesh
//! coordinates. These are the `cage` lines if the file has any. Otherwise it
//! lays out a frame mooring: the boundary is the frame (`n_s × n_t` cells of
//! about `spacing` m, with one cage per cell at the cell's centre), and
//! [`CageGrid`] sets the spacing, radius, net depth and solidity. A
//! four-sided boundary is mapped bilinearly, so a skewed frame keeps its
//! cells. Any other polygon gets the rectangle along its longest edge that
//! holds it. Count the cells against the site's real frame before trusting
//! the result: a 180 × 365 m frame of 90 m cells holds 2 × 4 cages.

use std::path::Path;

use thiserror::Error;

use super::projection::CoordinateProjection;
use crate::source::NetCage;

/// Errors reading a farm site file.
#[derive(Debug, Error)]
pub enum FarmSiteError {
    /// The file could not be read.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    /// A line could not be parsed.
    #[error("line {line}: {message}")]
    Parse {
        /// 1-based line number
        line: usize,
        /// What is wrong with it
        message: String,
    },
    /// The file has no `site` line.
    #[error("no `site` line")]
    MissingSite,
}

/// What a mooring line holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MooringKind {
    /// The cage frame
    Farm,
    /// The feed raft (barge)
    Raft,
}

/// A mooring line, frame (or raft) to anchor, longitude and latitude.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mooring {
    /// Where it leaves the frame or raft
    pub from: [f64; 2],
    /// The anchor end
    pub to: [f64; 2],
    /// What it holds
    pub kind: MooringKind,
}

/// A cage given in the file.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SiteCage {
    /// Centre, longitude and latitude
    pub position: [f64; 2],
    /// Radius (m), if given
    pub radius: Option<f64>,
    /// Net depth (m), if given
    pub net_depth: Option<f64>,
}

/// A fish farm site (see the [module docs](self)).
#[derive(Clone, Debug, PartialEq)]
pub struct FarmSite {
    /// Site number in the register (lokalitetsnummer)
    pub number: u32,
    /// Site name
    pub name: String,
    /// Register position, longitude and latitude
    pub position: [f64; 2],
    /// Licensed capacity and its unit (`TN`: tonnes of maximum allowed
    /// biomass)
    pub capacity: Option<(f64, String)>,
    /// The installation's boundary polygon, longitude and latitude
    pub boundary: Vec<[f64; 2]>,
    /// Mooring lines
    pub moorings: Vec<Mooring>,
    /// Cages given in the file
    pub cages: Vec<SiteCage>,
}

/// How [`FarmSite::cage_layout`] lays out cages on a frame, and the cages'
/// own dimensions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CageGrid {
    /// Frame cell size (m)
    pub spacing: f64,
    /// Cage radius (m), unless the file gives one
    pub radius: f64,
    /// Net depth (m), unless the file gives one
    pub net_depth: f64,
    /// Net solidity, for the drag ([`NetCage::circular`])
    pub solidity: f64,
}

impl CageGrid {
    /// The common ring of 160 m circumference (radius 25.5 m) in frame cells
    /// of 90 m, nets 20 m deep of solidity 0.25: Kattholmen's 180 × 365 m
    /// frame holds 2 × 4 of them.
    pub const RING_160M: Self = Self {
        spacing: 90.0,
        radius: 25.5,
        net_depth: 20.0,
        solidity: 0.25,
    };

    /// Cages of `radius` and `net_depth` (m) in frame cells of `spacing` m,
    /// nets of solidity 0.25.
    pub fn new(spacing: f64, radius: f64, net_depth: f64) -> Self {
        assert!(spacing > 0.0 && radius > 0.0 && net_depth > 0.0);
        Self {
            spacing,
            radius,
            net_depth,
            solidity: 0.25,
        }
    }

    /// The same with net solidity `solidity`.
    pub fn with_solidity(mut self, solidity: f64) -> Self {
        self.solidity = solidity;
        self
    }
}

/// The frame of a site in mesh coordinates, divided into cells of about a
/// given size: [`FarmSite::frame_cells`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameCells {
    /// The frame's corners, in order around it
    pub corners: [[f64; 2]; 4],
    /// Cells along the first side (corner 0 to 1) and along the second (1 to 2)
    pub n_s: usize,
    pub n_t: usize,
}

impl FrameCells {
    /// The point at frame coordinates `(s, t)` ∈ [0, 1]²: bilinear in the
    /// corners, `s` along the first side, `t` along the second.
    pub fn point(&self, s: f64, t: f64) -> [f64; 2] {
        let [p0, p1, p2, p3] = self.corners;
        [0, 1].map(|d| {
            (1.0 - s) * (1.0 - t) * p0[d]
                + s * (1.0 - t) * p1[d]
                + s * t * p2[d]
                + (1.0 - s) * t * p3[d]
        })
    }

    /// The cell centres, row by row along the first side.
    pub fn centres(&self) -> Vec<[f64; 2]> {
        (0..self.n_t)
            .flat_map(|j| {
                (0..self.n_s).map(move |i| {
                    let s = (i as f64 + 0.5) / self.n_s as f64;
                    let t = (j as f64 + 0.5) / self.n_t as f64;
                    self.point(s, t)
                })
            })
            .collect()
    }

    /// The frame's lines: its sides and the lines between the cells, each
    /// as its two ends.
    pub fn lines(&self) -> Vec<[[f64; 2]; 2]> {
        let across = (0..=self.n_s).map(|i| {
            let s = i as f64 / self.n_s as f64;
            [self.point(s, 0.0), self.point(s, 1.0)]
        });
        let along = (0..=self.n_t).map(|j| {
            let t = j as f64 / self.n_t as f64;
            [self.point(0.0, t), self.point(1.0, t)]
        });
        across.chain(along).collect()
    }
}

/// Read a farm site file (see the [module docs](self)).
pub fn read_farm_site_file(path: impl AsRef<Path>) -> Result<FarmSite, FarmSiteError> {
    parse_farm_site_str(&std::fs::read_to_string(path)?)
}

/// Parse a farm site file's contents (see the [module docs](self)).
pub fn parse_farm_site_str(text: &str) -> Result<FarmSite, FarmSiteError> {
    let mut site: Option<(u32, String)> = None;
    let mut position = None;
    let mut capacity = None;
    let (mut boundary, mut moorings, mut cages) = (Vec::new(), Vec::new(), Vec::new());
    for (n, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let error = |message: String| FarmSiteError::Parse {
            line: n + 1,
            message,
        };
        let (key, rest) = line.split_once(char::is_whitespace).unwrap_or((line, ""));
        let rest = rest.trim();
        let numbers = |count: std::ops::RangeInclusive<usize>| -> Result<Vec<f64>, FarmSiteError> {
            let values: Vec<f64> = rest
                .split_whitespace()
                .take(*count.end())
                .map(str::parse)
                .collect::<Result<_, _>>()
                .map_err(|e| error(format!("`{key}`: {e}")))?;
            if count.contains(&values.len()) {
                Ok(values)
            } else {
                Err(error(format!("`{key}` needs {count:?} numbers")))
            }
        };
        match key {
            "site" => {
                let (number, name) = rest.split_once(char::is_whitespace).unwrap_or((rest, ""));
                let number = number
                    .parse()
                    .map_err(|e| error(format!("site number: {e}")))?;
                site = Some((number, name.trim().to_string()));
            }
            "position" => {
                let v = numbers(2..=2)?;
                position = Some([v[0], v[1]]);
            }
            "capacity" => {
                let (value, unit) = rest.split_once(char::is_whitespace).unwrap_or((rest, ""));
                let value = value.parse().map_err(|e| error(format!("capacity: {e}")))?;
                capacity = Some((value, unit.trim().to_string()));
            }
            "boundary" => {
                let v = numbers(2..=2)?;
                boundary.push([v[0], v[1]]);
            }
            "mooring" => {
                let parts: Vec<&str> = rest.split_whitespace().collect();
                let kind = match parts.get(4).copied() {
                    Some("farm") | None => MooringKind::Farm,
                    Some("raft") => MooringKind::Raft,
                    Some(other) => return Err(error(format!("mooring kind `{other}`"))),
                };
                let v = numbers(4..=4)?;
                moorings.push(Mooring {
                    from: [v[0], v[1]],
                    to: [v[2], v[3]],
                    kind,
                });
            }
            "cage" => {
                let v = numbers(2..=4)?;
                if v.len() == 3 {
                    return Err(error("`cage` takes a radius only with a net depth".into()));
                }
                cages.push(SiteCage {
                    position: [v[0], v[1]],
                    radius: v.get(2).copied(),
                    net_depth: v.get(3).copied(),
                });
            }
            other => return Err(error(format!("unknown key `{other}`"))),
        }
    }
    let (number, name) = site.ok_or(FarmSiteError::MissingSite)?;
    Ok(FarmSite {
        number,
        name,
        position: position.unwrap_or_else(|| centroid(&boundary)),
        capacity,
        boundary,
        moorings,
        cages,
    })
}

/// Mean of the points (NaN for none).
fn centroid(points: &[[f64; 2]]) -> [f64; 2] {
    let n = points.len() as f64;
    [0, 1].map(|d| points.iter().map(|p| p[d]).sum::<f64>() / n)
}

/// `p` projected to mesh coordinates.
fn project(projection: &impl CoordinateProjection, [lon, lat]: [f64; 2]) -> [f64; 2] {
    let (x, y) = projection.geo_to_xy(lat, lon);
    [x, y]
}

impl FarmSite {
    /// The boundary polygon in mesh coordinates.
    pub fn boundary_xy(&self, projection: &impl CoordinateProjection) -> Vec<[f64; 2]> {
        self.boundary
            .iter()
            .map(|&p| project(projection, p))
            .collect()
    }

    /// The frame in mesh coordinates, in cells of about `spacing` m: the
    /// boundary if it has four vertices, else the rectangle along its
    /// longest edge that holds it (see the [module docs](self)). `None`
    /// without a boundary polygon.
    pub fn frame_cells(
        &self,
        projection: &impl CoordinateProjection,
        spacing: f64,
    ) -> Option<FrameCells> {
        let corners = self.frame(projection)?;
        let [p0, p1, p2, p3] = corners;
        let length = |a: [f64; 2], b: [f64; 2]| (b[0] - a[0]).hypot(b[1] - a[1]);
        let cells = |l: f64| ((l / spacing).round() as usize).max(1);
        Some(FrameCells {
            corners,
            n_s: cells(0.5 * (length(p0, p1) + length(p3, p2))),
            n_t: cells(0.5 * (length(p1, p2) + length(p0, p3))),
        })
    }

    /// The four corners of the frame in mesh coordinates, in order around
    /// it: the boundary if it has four vertices, else the rectangle along
    /// its longest edge that holds it.
    fn frame(&self, projection: &impl CoordinateProjection) -> Option<[[f64; 2]; 4]> {
        let points = self.boundary_xy(projection);
        if points.len() < 3 {
            return None;
        }
        if let [a, b, c, d] = points[..] {
            return Some([a, b, c, d]);
        }
        let n = points.len();
        let (i, _) = (0..n)
            .map(|i| {
                let (a, b) = (points[i], points[(i + 1) % n]);
                (i, (b[0] - a[0]).hypot(b[1] - a[1]))
            })
            .max_by(|x, y| x.1.total_cmp(&y.1))?;
        let (a, b) = (points[i], points[(i + 1) % n]);
        let length = (b[0] - a[0]).hypot(b[1] - a[1]);
        let e = [(b[0] - a[0]) / length, (b[1] - a[1]) / length];
        let normal = [-e[1], e[0]];
        let along = |p: [f64; 2]| (p[0] - a[0]) * e[0] + (p[1] - a[1]) * e[1];
        let across = |p: [f64; 2]| (p[0] - a[0]) * normal[0] + (p[1] - a[1]) * normal[1];
        let range = |f: &dyn Fn([f64; 2]) -> f64| {
            points
                .iter()
                .map(|&p| f(p))
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
                    (lo.min(v), hi.max(v))
                })
        };
        let ((s0, s1), (t0, t1)) = (range(&along), range(&across));
        let corner = |s: f64, t: f64| {
            [
                a[0] + s * e[0] + t * normal[0],
                a[1] + s * e[1] + t * normal[1],
            ]
        };
        Some([
            corner(s0, t0),
            corner(s1, t0),
            corner(s1, t1),
            corner(s0, t1),
        ])
    }

    /// The cage centres in mesh coordinates: the `cage` lines, or one per
    /// frame cell of about `spacing` m (see the [module docs](self)).
    pub fn cage_centres(
        &self,
        projection: &impl CoordinateProjection,
        spacing: f64,
    ) -> Vec<[f64; 2]> {
        if !self.cages.is_empty() {
            return self
                .cages
                .iter()
                .map(|c| project(projection, c.position))
                .collect();
        }
        self.frame_cells(projection, spacing)
            .map_or_else(Vec::new, |frame| frame.centres())
    }

    /// The site's cages as [`NetCage`]s in mesh coordinates (see the
    /// [module docs](self)).
    pub fn cage_layout(
        &self,
        projection: &impl CoordinateProjection,
        grid: &CageGrid,
    ) -> Vec<NetCage> {
        let centres = self.cage_centres(projection, grid.spacing);
        centres
            .into_iter()
            .enumerate()
            .map(|(i, centre)| {
                let given = self.cages.get(i);
                NetCage::circular(
                    centre,
                    given.and_then(|c| c.radius).unwrap_or(grid.radius),
                    given.and_then(|c| c.net_depth).unwrap_or(grid.net_depth),
                    grid.solidity,
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::LocalProjection;
    use crate::source::CageFootprint;

    const KATTHOLMEN: &str = "\
# Fiskeridirektoratet site 14042
site 14042 KATTHOLMEN
position 8.677683 63.867683
capacity 7800.0 TN
boundary 8.6771000 63.8726167
boundary 8.6808167 63.8726000
boundary 8.6808167 63.8693333
boundary 8.6772667 63.8693667
mooring 8.6752500 63.8678000 8.6757000 63.8687667 raft
mooring 8.6808167 63.8726000 8.6856500 63.8724667 farm
";

    fn centre(cage: &NetCage) -> [f64; 2] {
        match cage.footprint {
            CageFootprint::Circle { center, .. } => center,
            _ => panic!("circular cages"),
        }
    }

    #[test]
    fn reads_a_fetched_site() {
        let site = parse_farm_site_str(KATTHOLMEN).unwrap();
        assert_eq!((site.number, site.name.as_str()), (14042, "KATTHOLMEN"));
        assert_eq!(site.capacity, Some((7800.0, "TN".into())));
        assert_eq!(site.boundary.len(), 4);
        assert_eq!(site.moorings[0].kind, MooringKind::Raft);
        assert_eq!(site.moorings[1].to, [8.68565, 63.8724667]);
    }

    /// A 180 × 365 m frame of 90 m cells: 2 × 4 cages at the cells' centres,
    /// about 90 m apart and well inside the frame.
    #[test]
    fn lays_out_cages_on_the_frame() {
        let site = parse_farm_site_str(KATTHOLMEN).unwrap();
        let projection = LocalProjection::new(63.871, 8.679);
        let cages = site.cage_layout(&projection, &CageGrid::new(90.0, 25.0, 20.0));
        assert_eq!(cages.len(), 8);
        let xy: Vec<[f64; 2]> = cages.iter().map(centre).collect();
        let frame = site.boundary_xy(&projection);
        let (x0, x1) = (frame[0][0].max(frame[3][0]), frame[1][0].min(frame[2][0]));
        let (y0, y1) = (frame[2][1].max(frame[3][1]), frame[0][1].min(frame[1][1]));
        for p in &xy {
            assert!(p[0] > x0 + 25.0 && p[0] < x1 - 25.0 && p[1] > y0 + 25.0 && p[1] < y1 - 25.0);
        }
        // Neighbours across (first row) and along (first column)
        let gap = |a: [f64; 2], b: [f64; 2]| (b[0] - a[0]).hypot(b[1] - a[1]);
        assert!(
            (gap(xy[0], xy[1]) - 91.0).abs() < 3.0,
            "{}",
            gap(xy[0], xy[1])
        );
        assert!(
            (gap(xy[0], xy[2]) - 91.0).abs() < 3.0,
            "{}",
            gap(xy[0], xy[2])
        );
        assert!(cages.iter().all(|c| c.net_depth == 20.0));
    }

    /// The frame's lines are its sides and the lines between its cells, and
    /// the cages sit at the cells' centres.
    #[test]
    fn the_frame_lines_bound_the_cells() {
        let site = parse_farm_site_str(KATTHOLMEN).unwrap();
        let projection = LocalProjection::new(63.871, 8.679);
        let frame = site.frame_cells(&projection, 90.0).unwrap();
        assert_eq!((frame.n_s, frame.n_t), (2, 4));
        let lines = frame.lines();
        assert_eq!(lines.len(), (frame.n_s + 1) + (frame.n_t + 1));
        // The sides are the boundary's: corners 0-3 (s = 0) and 0-1 (t = 0)
        let boundary = site.boundary_xy(&projection);
        let close = |a: [f64; 2], b: [f64; 2]| (a[0] - b[0]).hypot(a[1] - b[1]) < 1e-9;
        assert!(close(lines[0][0], boundary[0]) && close(lines[0][1], boundary[3]));
        assert!(close(lines[frame.n_s + 1][0], boundary[0]));
        assert!(close(lines[frame.n_s + 1][1], boundary[1]));
        // Each centre is the mean of its cell's corners on a parallelogram,
        // and the layout's cages are the centres
        let centres = frame.centres();
        let cages = site.cage_layout(&projection, &CageGrid::new(90.0, 25.0, 20.0));
        assert!(
            centres
                .iter()
                .zip(&cages)
                .all(|(&p, c)| close(p, centre(c)))
        );
    }

    /// `cage` lines win over the frame, with their own dimensions; a
    /// polygon of other than four vertices gets its bounding rectangle.
    #[test]
    fn explicit_cages_and_other_polygons() {
        let text = format!("{KATTHOLMEN}cage 8.678 63.872 30 15\ncage 8.679 63.871\n");
        let site = parse_farm_site_str(&text).unwrap();
        let projection = LocalProjection::new(63.871, 8.679);
        let cages = site.cage_layout(&projection, &CageGrid::new(90.0, 25.0, 20.0));
        assert_eq!(cages.len(), 2);
        assert_eq!(cages[0].net_depth, 15.0);
        assert_eq!(cages[1].net_depth, 20.0);
        let (x, y) = projection.geo_to_xy(63.871, 8.679);
        assert!((centre(&cages[1])[0] - x).abs() < 1e-9 && (centre(&cages[1])[1] - y).abs() < 1e-9);
        // A pentagon in a 200 × 100 m box: 2 × 1 cells of 100 m
        let projection = LocalProjection::new(0.0, 0.0);
        let deg = |m: f64| m / 111_319.49;
        let mut pentagon = String::from("site 1 TEST\n");
        for [x, y] in [
            [0.0, 0.0],
            [200.0, 0.0],
            [200.0, 100.0],
            [100.0, 120.0],
            [0.0, 100.0],
        ] {
            pentagon += &format!("boundary {} {}\n", deg(x), deg(y));
        }
        let site = parse_farm_site_str(&pentagon).unwrap();
        let centres = site.cage_centres(&projection, 100.0);
        assert_eq!(centres.len(), 2);
        assert!((centres[0][0] - 50.0).abs() < 0.5 && (centres[0][1] - 60.0).abs() < 0.5);
    }

    #[test]
    fn rejects_bad_lines() {
        assert!(matches!(
            parse_farm_site_str("position 1 2\n"),
            Err(FarmSiteError::MissingSite)
        ));
        for bad in [
            "site x NAME",
            "site 1 A\nboundary 1",
            "site 1 A\ncage 1 2 3",
            "site 1 A\nfoo 1",
        ] {
            assert!(
                matches!(parse_farm_site_str(bad), Err(FarmSiteError::Parse { .. })),
                "{bad}"
            );
        }
    }
}

//! Bed elevation raster: bathymetry and land in one field defined everywhere.
//!
//! A mesh node needs a bed elevation B wherever it lies, on land as well as
//! under water: `WetDry` treats nodes with B above the surface as dry shore.
//! A [`BedRaster`] is one of two things:
//!
//! - **An elevation model** ([`BedRaster::new`]): B on a grid, land heights
//!   included (a bathymetry merged with a foreshore DEM), sampled bilinearly
//!   everywhere.
//! - **A sea bed and a land mask** ([`BedRaster::from_geotiff`]): bathymetry
//!   rasters have no land heights (the Frøya GeoTIFF stores land as 0), and
//!   the land/water line is better known from a coastline than from the
//!   raster's pixels. The sea bed is the bathymetry, with dry and no-data
//!   pixels at 0 (the shore at mean sea level), sampled bilinearly; a land
//!   mask on a grid `refinement` times finer, rasterised from the coastline,
//!   puts B = `land_elevation` on land.
//!
//! The mask matters: baking `land_elevation` into the pixels instead ramps
//! the bed up to it across a whole pixel at every coast, which closes narrow
//! sounds at 1 km resolution (it cut M2 at Mausund to 0.6 of the gauge's).
//!
//! [`crate::mesh::Bathymetry2D::from_raster`] projects a raster onto the
//! element nodes ([`crate::mesh::Bathymetry2D::project`]), which does not
//! alias it where elements are coarser than the pixels.

use super::coastline::CoastlineData;
use super::geotiff::{GeoTiffBathymetry, GeoTiffError};
use super::projection::{CoordinateProjection, GeoBoundingBox, LocalProjection};
use super::soundings::{SimpleKriging, Sounding, SoundingCorrection, spherical_correlation};

/// Bed elevation (m above mean sea level, negative under water) on a
/// north-up longitude/latitude grid, defined everywhere.
///
/// Pixel `(row, col)` covers `[min_lon + col·Δλ, min_lon + (col + 1)·Δλ] ×
/// [max_lat − (row + 1)·Δφ, max_lat − row·Δφ]`, its value belongs to its
/// centre, and row 0 is the northernmost. See the module docs for the two
/// kinds of raster.
#[derive(Debug, Clone)]
pub struct BedRaster {
    /// Elevations, row-major `[row × width + col]`
    elevation: Vec<f64>,
    width: usize,
    height: usize,
    bbox: GeoBoundingBox,
    land: Option<LandMask>,
}

/// The whole pixels of a raster touched by a window.
struct Crop {
    rows: std::ops::Range<usize>,
    cols: std::ops::Range<usize>,
    bbox: GeoBoundingBox,
}

impl Crop {
    /// # Panics
    /// If `window` does not overlap the raster.
    fn new(raster: &GeoTiffBathymetry, window: &GeoBoundingBox) -> Self {
        let full = raster.bbox();
        let (full_width, full_height) = raster.dimensions();
        let dlon = (full.max_lon - full.min_lon) / full_width as f64;
        let dlat = (full.max_lat - full.min_lat) / full_height as f64;
        let col0 = ((window.min_lon - full.min_lon) / dlon).floor().max(0.0) as usize;
        let col1 = (((window.max_lon - full.min_lon) / dlon).ceil() as usize).min(full_width);
        let row0 = ((full.max_lat - window.max_lat) / dlat).floor().max(0.0) as usize;
        let row1 = (((full.max_lat - window.min_lat) / dlat).ceil() as usize).min(full_height);
        assert!(
            col0 < col1 && row0 < row1,
            "window {window:?} does not overlap the raster ({full:?})"
        );
        let bbox = GeoBoundingBox::new(
            full.min_lon + col0 as f64 * dlon,
            full.max_lat - row1 as f64 * dlat,
            full.min_lon + col1 as f64 * dlon,
            full.max_lat - row0 as f64 * dlat,
        );
        Self {
            rows: row0..row1,
            cols: col0..col1,
            bbox,
        }
    }

    fn width(&self) -> usize {
        self.cols.len()
    }

    fn height(&self) -> usize {
        self.rows.len()
    }

    /// `value(row, col)` of every pixel in the crop, row-major (source
    /// raster indices).
    fn values(&self, value: impl Fn(usize, usize) -> f64) -> Vec<f64> {
        self.rows
            .clone()
            .flat_map(|row| self.cols.clone().map(move |col| (row, col)))
            .map(|(row, col)| value(row, col))
            .collect()
    }
}

/// The pixels along one axis whose centres may lie within `reach` of a
/// point `offset` from the raster's first edge (pixels of size `pixel`,
/// `count` of them; all in the same unit).
fn pixel_span(offset: f64, reach: f64, pixel: f64, count: usize) -> std::ops::Range<usize> {
    let first = ((offset - reach) / pixel).floor().max(0.0) as usize;
    let end = (((offset + reach) / pixel).ceil().max(0.0) as usize).min(count);
    first.min(end)..end
}

/// Land cells on a grid `refinement` times finer than the pixels, with the
/// bed elevation given to them.
#[derive(Debug, Clone)]
struct LandMask {
    cells: Vec<bool>,
    refinement: usize,
    elevation: f64,
}

impl BedRaster {
    /// An elevation model: `width` × `height` elevations (row-major, row 0
    /// north) covering `bbox`, land heights included.
    ///
    /// # Panics
    /// If the sizes disagree, the raster is empty or a value is not finite.
    pub fn new(bbox: GeoBoundingBox, width: usize, height: usize, elevation: Vec<f64>) -> Self {
        assert!(width > 0 && height > 0, "empty bed raster");
        assert_eq!(
            elevation.len(),
            width * height,
            "bed raster has {} values for {width} × {height} pixels",
            elevation.len()
        );
        assert!(
            elevation.iter().all(|b| b.is_finite()),
            "bed raster values must be finite"
        );
        assert!(
            bbox.max_lon > bbox.min_lon && bbox.max_lat > bbox.min_lat,
            "bed raster extent must be positive"
        );
        Self {
            elevation,
            width,
            height,
            bbox,
            land: None,
        }
    }

    /// A sea bed and a land mask over `window` (cropped to the whole pixels
    /// of `bathymetry` that it touches).
    ///
    /// The sea bed is the bathymetry where it is below mean sea level, and 0
    /// at dry and no-data pixels. The land mask has `refinement` ×
    /// `refinement` cells per pixel. With a coastline, a cell is land where
    /// its centre is inside a land polygon, or where the bilinear sea bed
    /// there is not below mean sea level (the interior of the raster's land);
    /// water pixels that the coastline puts on land, and dry pixels that it
    /// leaves in the water, follow the coastline, which keeps narrow sounds
    /// open that the pixels close. Without one, a cell is land where its pixel
    /// is dry or no-data. Land cells get the bed elevation `land_elevation`.
    ///
    /// # Panics
    /// If `land_elevation` is not positive, `refinement` is 0, or `window`
    /// does not overlap the bathymetry.
    pub fn from_geotiff(
        bathymetry: &GeoTiffBathymetry,
        coastline: Option<&CoastlineData>,
        land_elevation: f64,
        window: &GeoBoundingBox,
        refinement: usize,
    ) -> Self {
        assert!(
            land_elevation > 0.0,
            "land must lie above mean sea level, got land_elevation = {land_elevation}"
        );
        assert!(refinement > 0, "land mask refinement must be positive");
        let crop = Crop::new(bathymetry, window);
        let sea_bed = crop.values(|row, col| bathymetry.pixel(row, col).unwrap_or(0.0).min(0.0));
        let mut raster = Self::new(crop.bbox, crop.width(), crop.height(), sea_bed);
        let bbox = crop.bbox;

        let (mask_width, mask_height) = (raster.width * refinement, raster.height * refinement);
        let cell_lon = (bbox.max_lon - bbox.min_lon) / mask_width as f64;
        let cell_lat = (bbox.max_lat - bbox.min_lat) / mask_height as f64;
        let cells = (0..mask_height)
            .flat_map(|row| (0..mask_width).map(move |col| (row, col)))
            .map(|(row, col)| {
                let lat = bbox.max_lat - (row as f64 + 0.5) * cell_lat;
                let lon = bbox.min_lon + (col as f64 + 0.5) * cell_lon;
                match coastline {
                    Some(c) => raster.bilinear(lat, lon) >= 0.0 || c.is_land(lat, lon),
                    None => raster.pixel(row / refinement, col / refinement) >= 0.0,
                }
            })
            .collect();
        raster.land = Some(LandMask {
            cells,
            refinement,
            elevation: land_elevation,
        });
        raster
    }

    /// An elevation model read from a GeoTIFF of bed elevations with land
    /// heights (e.g. Kartverket's topobathy model,
    /// `scripts/kartverket_topobathy.sh`), over `window` (cropped to the
    /// whole pixels that it touches).
    ///
    /// # Errors
    /// [`GeoTiffError::MissingData`] if a pixel in the window has no data:
    /// an elevation model must be complete.
    ///
    /// # Panics
    /// If `window` does not overlap the raster.
    pub fn elevation_model(
        elevation: &GeoTiffBathymetry,
        window: &GeoBoundingBox,
    ) -> Result<Self, GeoTiffError> {
        let crop = Crop::new(elevation, window);
        let values = crop.values(|row, col| elevation.pixel(row, col).unwrap_or(f64::NAN));
        let missing = values.iter().filter(|v| v.is_nan()).count();
        if missing > 0 {
            return Err(GeoTiffError::MissingData(format!(
                "{missing} of {} pixels in {window:?} have no data",
                values.len()
            )));
        }
        Ok(Self::new(crop.bbox, crop.width(), crop.height(), values))
    }

    /// Replace the pixels where `is_hole(value)` by the harmonic
    /// interpolation of the others (Laplace's equation with the surrounding
    /// pixels as boundary values, Gauss–Seidel to 1 mm), and return how many
    /// were filled. Holes at the raster edge take no boundary value there; a
    /// hole with no data around it at all keeps its values.
    ///
    /// Kartverket's topobathy model marks missing depths with an exact 0,
    /// the water surface (a 30 × 30 pixel patch in 320 m of water off Frøya,
    /// where the flat 0 piled water up to +1.2 m). Harmonic filling
    /// reproduces linear beds exactly and leaves a 0 among near-zero shore
    /// pixels near 0.
    pub fn fill_holes(&mut self, is_hole: impl Fn(f64) -> bool) -> usize {
        let (w, h) = (self.width, self.height);
        let holes: Vec<usize> = (0..w * h).filter(|&i| is_hole(self.elevation[i])).collect();
        let mut hole = vec![false; w * h];
        for &i in &holes {
            hole[i] = true;
        }
        let neighbours = |i: usize| {
            let (row, col) = (i / w, i % w);
            [
                (row > 0).then(|| i - w),
                (row + 1 < h).then(|| i + w),
                (col > 0).then(|| i - 1),
                (col + 1 < w).then(|| i + 1),
            ]
            .into_iter()
            .flatten()
        };
        // Start from the mean of the known neighbours, spreading inwards
        let mut known: Vec<bool> = hole.iter().map(|&h| !h).collect();
        let mut pending = holes.clone();
        while !pending.is_empty() {
            let mut next = Vec::new();
            let mut updates = Vec::new();
            for &i in &pending {
                let (sum, n) = neighbours(i)
                    .filter(|&j| known[j])
                    .fold((0.0, 0), |(s, n), j| (s + self.elevation[j], n + 1));
                if n > 0 {
                    updates.push((i, sum / n as f64));
                } else {
                    next.push(i);
                }
            }
            if updates.is_empty() {
                break; // no data around these at all
            }
            for (i, value) in updates {
                self.elevation[i] = value;
                known[i] = true;
            }
            pending = next;
        }
        let fillable: Vec<usize> = holes.iter().copied().filter(|&i| known[i]).collect();
        for _ in 0..100_000 {
            let mut change: f64 = 0.0;
            for &i in &fillable {
                let (sum, n) =
                    neighbours(i).fold((0.0, 0), |(s, n), j| (s + self.elevation[j], n + 1));
                let value = sum / n as f64;
                change = change.max((value - self.elevation[i]).abs());
                self.elevation[i] = value;
            }
            if change < 1e-3 {
                break;
            }
        }
        fillable.len()
    }

    /// Cap the bed at `max_elevation`: land that no water level reaches
    /// only makes cliffs in shoreline elements (a wet corner at −7 m beside
    /// +30 m land drew a creeping film in `WetDry` subcells), so keep the
    /// foreshore and flatten the rest. The land elevation of a mask is capped
    /// too.
    pub fn clamp_land(mut self, max_elevation: f64) -> Self {
        for b in &mut self.elevation {
            *b = b.min(max_elevation);
        }
        if let Some(mask) = &mut self.land {
            mask.elevation = mask.elevation.min(max_elevation);
        }
        self
    }

    /// Lower the sea part (the pixels below 0) from its own datum to mean sea
    /// level, `offset` m below it, and return how many pixels moved.
    ///
    /// An elevation model may hold its land and its sea in different
    /// datums: Kartverket's topobathy model has land heights in NN2000 (8 cm
    /// above mean sea level at Mausund) and depths in chart datum (LAT), 1.43
    /// m below mean sea level at Mausund and 1.69 m at Bodø
    /// (`notes/inputs-forcing.md` P1.6: its flat beds equal the chart's
    /// soundings to centimetres). A pixel `b < 0` becomes `b + max(b,
    /// −offset)`: the full offset from `offset` m of chart depth down, and
    /// half of it over the shallower band, so the 0 m line (the raster's
    /// coast) stays where it is and the bed has no step there. The band is
    /// where the raster's sea meets its land, whose datum it does not say.
    ///
    /// # Panics
    /// If `offset` is negative or not finite.
    pub fn lower_sea_to_mean_sea_level(&mut self, offset: f64) -> usize {
        assert!(
            offset.is_finite() && offset >= 0.0,
            "a sea datum below mean sea level is a positive offset, not {offset}"
        );
        let mut moved = 0;
        for b in self.elevation.iter_mut().filter(|b| **b < 0.0) {
            *b += b.max(-offset);
            moved += 1;
        }
        moved
    }

    /// Correct the sea towards measured depths: add the simple-kriging
    /// estimate of the stations' residuals (their measured bed less this
    /// raster's, bilinear) to every sea pixel within `range` m of a station
    /// (`super::soundings`, the method and its reasoning), and report it.
    ///
    /// Stations off the raster or on its land are left out: the raster's
    /// coast is not theirs to move. For the same reason a sea pixel stays in
    /// the sea: a correction lifts it at most to [`Self::SHALLOWEST_SOUNDED`]
    /// below mean sea level, or leaves it where it is if it is shallower
    /// already; the report counts the pixels so held. Distances are on the
    /// raster's central local projection.
    ///
    /// # Panics
    /// If `range` or `noise` (m) is not positive.
    pub fn correct_to_soundings(
        &mut self,
        soundings: &[Sounding],
        range: f64,
        noise: f64,
    ) -> SoundingCorrection {
        let (lat0, lon0) = self.bbox.center();
        let (m_lat, m_lon) = LocalProjection::new(lat0, lon0).scale_factors();
        let xy = |lat: f64, lon: f64| [(lon - lon0) * m_lon, (lat - lat0) * m_lat];
        let bbox = self.bbox;
        let mut report = SoundingCorrection {
            stations: soundings.len(),
            ..Default::default()
        };
        let mut used = Vec::new();
        for s in soundings {
            let inside = (bbox.min_lon..=bbox.max_lon).contains(&s.lon)
                && (bbox.min_lat..=bbox.max_lat).contains(&s.lat);
            if !inside {
                report.off_raster += 1;
            } else if self.is_land(s.lat, s.lon) {
                report.on_land += 1;
            } else {
                used.push(*s);
            }
        }
        let points: Vec<[f64; 2]> = used.iter().map(|s| xy(s.lat, s.lon)).collect();
        let residuals: Vec<f64> = used
            .iter()
            .map(|s| -s.depth - self.bilinear(s.lat, s.lon))
            .collect();
        report.used = used.len();
        report.deeper_than_raster = residuals.iter().map(|e| -e).collect();
        let field = SimpleKriging::new(&points, &residuals, range, noise);
        report.signal = field.signal_variance().sqrt();

        let (w, h) = (self.width, self.height);
        let dlon = (bbox.max_lon - bbox.min_lon) / w as f64;
        let dlat = (bbox.max_lat - bbox.min_lat) / h as f64;
        let mut correction = vec![0.0; w * h];
        for (p, weight) in field.terms().filter(|(_, weight)| *weight != 0.0) {
            let (lon, lat) = (lon0 + p[0] / m_lon, lat0 + p[1] / m_lat);
            let cols = pixel_span(lon - bbox.min_lon, range / m_lon, dlon, w);
            let rows = pixel_span(bbox.max_lat - lat, range / m_lat, dlat, h);
            for row in rows {
                let y = (bbox.max_lat - (row as f64 + 0.5) * dlat - lat0) * m_lat;
                for col in cols.clone() {
                    let x = (bbox.min_lon + (col as f64 + 0.5) * dlon - lon0) * m_lon;
                    let d = (x - p[0]).hypot(y - p[1]);
                    correction[row * w + col] += weight * spherical_correlation(d, range);
                }
            }
        }
        for (b, c) in self.elevation.iter_mut().zip(&correction) {
            if *b >= 0.0 || *c == 0.0 {
                continue;
            }
            let ceiling = b.max(-Self::SHALLOWEST_SOUNDED);
            let moved = (*b + c).min(ceiling);
            if moved < *b + c {
                report.held += 1;
            }
            report.deepened_by = report.deepened_by.max(*b - moved);
            report.shoaled_by = report.shoaled_by.max(moved - *b);
            if moved != *b {
                report.pixels += 1;
            }
            *b = moved;
        }
        report.deeper_than_corrected = used
            .iter()
            .map(|s| s.depth + self.bilinear(s.lat, s.lon))
            .collect();
        report
    }

    /// The shallowest depth (m) [`Self::correct_to_soundings`] lifts a sea
    /// pixel to, so that a correction never moves the coast.
    pub const SHALLOWEST_SOUNDED: f64 = 1.0;

    /// Extent (outer pixel edges).
    pub fn bbox(&self) -> &GeoBoundingBox {
        &self.bbox
    }

    /// Size in pixels (width, height).
    pub fn dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    /// Value of pixel `(row, col)`: the elevation, or the sea bed under a
    /// land mask.
    pub fn pixel(&self, row: usize, col: usize) -> f64 {
        self.elevation[row * self.width + col]
    }

    /// Bed elevation at a longitude/latitude: the land elevation in a land
    /// cell of the mask, else bilinear between pixel centres. Points in the
    /// outer half pixel, or outside the raster, take the value at the
    /// nearest point of the centres' hull (and of the mask).
    pub fn elevation(&self, lat: f64, lon: f64) -> f64 {
        match &self.land {
            Some(mask) if self.is_land_cell(mask, lat, lon) => mask.elevation,
            _ => self.bilinear(lat, lon),
        }
    }

    /// Whether `(lat, lon)` is land: in a land cell of the mask, or, for an
    /// elevation model, not below mean sea level.
    pub fn is_land(&self, lat: f64, lon: f64) -> bool {
        match &self.land {
            Some(mask) => self.is_land_cell(mask, lat, lon),
            None => self.bilinear(lat, lon) >= 0.0,
        }
    }

    fn is_land_cell(&self, mask: &LandMask, lat: f64, lon: f64) -> bool {
        let bbox = &self.bbox;
        let (w, h) = (self.width * mask.refinement, self.height * mask.refinement);
        let col = ((lon - bbox.min_lon) / (bbox.max_lon - bbox.min_lon) * w as f64)
            .clamp(0.0, (w - 1) as f64) as usize;
        let row = ((bbox.max_lat - lat) / (bbox.max_lat - bbox.min_lat) * h as f64)
            .clamp(0.0, (h - 1) as f64) as usize;
        mask.cells[row * w + col]
    }

    /// Bilinear interpolation of the pixel values between pixel centres.
    fn bilinear(&self, lat: f64, lon: f64) -> f64 {
        let bbox = &self.bbox;
        let col = ((lon - bbox.min_lon) / (bbox.max_lon - bbox.min_lon) * self.width as f64 - 0.5)
            .clamp(0.0, (self.width - 1) as f64);
        let row = ((bbox.max_lat - lat) / (bbox.max_lat - bbox.min_lat) * self.height as f64 - 0.5)
            .clamp(0.0, (self.height - 1) as f64);
        let (c0, r0) = (col.floor() as usize, row.floor() as usize);
        let (c1, r1) = ((c0 + 1).min(self.width - 1), (r0 + 1).min(self.height - 1));
        let (s, t) = (col - c0 as f64, row - r0 as f64);
        (1.0 - t) * ((1.0 - s) * self.pixel(r0, c0) + s * self.pixel(r0, c1))
            + t * ((1.0 - s) * self.pixel(r1, c0) + s * self.pixel(r1, c1))
    }

    /// The bed elevation as a function of mesh coordinates (m) under
    /// `projection`, for [`crate::mesh::Bathymetry2D::project`] or
    /// [`crate::mesh::Bathymetry2D::from_function`].
    pub fn sampler<'a, P: CoordinateProjection>(
        &'a self,
        projection: &'a P,
    ) -> impl Fn(f64, f64) -> f64 + 'a {
        move |x, y| {
            let (lat, lon) = projection.xy_to_geo(x, y);
            self.elevation(lat, lon)
        }
    }

    /// The smaller side (m) of a pixel, or of a mask cell if finer, at the
    /// raster's central latitude: the resolution to hand
    /// [`crate::mesh::Bathymetry2D::project`].
    pub fn pixel_size(&self) -> f64 {
        let (lat, lon) = self.bbox.center();
        let (m_per_deg_lat, m_per_deg_lon) = LocalProjection::new(lat, lon).scale_factors();
        let refinement = self.land.as_ref().map_or(1, |mask| mask.refinement) as f64;
        let dlon = (self.bbox.max_lon - self.bbox.min_lon) / (self.width as f64 * refinement);
        let dlat = (self.bbox.max_lat - self.bbox.min_lat) / (self.height as f64 * refinement);
        (dlon * m_per_deg_lon).min(dlat * m_per_deg_lat)
    }

    /// Fraction of the area that is water: mask cells, or pixels below mean
    /// sea level for an elevation model.
    pub fn water_fraction(&self) -> f64 {
        match &self.land {
            Some(mask) => {
                mask.cells.iter().filter(|&&land| !land).count() as f64 / mask.cells.len() as f64
            }
            None => {
                self.elevation.iter().filter(|&&b| b < 0.0).count() as f64
                    / self.elevation.len() as f64
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 4 × 3 elevation model over [8, 8.4] × [63.7, 63.73]: 0.1° × 0.01°
    /// pixels.
    fn raster() -> BedRaster {
        #[rustfmt::skip]
        let elevation = vec![
            -10.0, -20.0, -30.0, -40.0,
            -12.0, -22.0, -32.0, -42.0,
              2.0,   2.0, -34.0, -44.0,
        ];
        BedRaster::new(GeoBoundingBox::new(8.0, 63.7, 8.4, 63.73), 4, 3, elevation)
    }

    #[test]
    fn pixel_centres_return_pixel_values() {
        let r = raster();
        for row in 0..3 {
            for col in 0..4 {
                let lat = 63.73 - (row as f64 + 0.5) * 0.01;
                let lon = 8.0 + (col as f64 + 0.5) * 0.1;
                assert!((r.elevation(lat, lon) - r.pixel(row, col)).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn bilinear_between_centres_and_clamped_outside() {
        let r = raster();
        // Midway between the centres of (0, 0) and (0, 1), and of (1, 0)
        assert!((r.elevation(63.725, 8.1) - (-15.0)).abs() < 1e-9);
        assert!((r.elevation(63.72, 8.05) - (-11.0)).abs() < 1e-9);
        // Centre of the 2 × 2 block (0..2, 0..2): mean of −10, −20, −12, −22
        assert!((r.elevation(63.72, 8.1) - (-16.0)).abs() < 1e-9);
        // Outside: the nearest corner centre
        assert!((r.elevation(64.0, 7.0) - (-10.0)).abs() < 1e-12);
        assert!((r.elevation(63.0, 9.0) - (-44.0)).abs() < 1e-12);
        // An elevation model has land where it is not below sea level
        assert!(r.is_land(63.705, 8.05) && !r.is_land(63.705, 8.35));
    }

    #[test]
    fn sampler_goes_through_the_projection() {
        let r = raster();
        let projection = LocalProjection::new(63.72, 8.1);
        let bed = r.sampler(&projection);
        assert!((bed(0.0, 0.0) - (-16.0)).abs() < 1e-9);
    }

    /// The sea moves down by the datum's offset below its depth, by half of
    /// it in the shallower band (so the coast and the bed's continuity stay),
    /// and the land does not move.
    #[test]
    fn the_sea_part_is_lowered_to_mean_sea_level() {
        #[rustfmt::skip]
        let elevation = vec![
            -40.0, -1.5, -1.43, -0.5,
              0.0,  0.5,   2.0, -1e-9,
        ];
        let mut r = BedRaster::new(GeoBoundingBox::new(8.0, 63.7, 8.4, 63.72), 4, 2, elevation);
        let moved = r.lower_sea_to_mean_sea_level(1.43);
        assert_eq!(moved, 5);
        let expected = [-41.43, -2.93, -2.86, -1.0, 0.0, 0.5, 2.0, -2e-9];
        for (i, e) in expected.iter().enumerate() {
            let got = r.pixel(i / 4, i % 4);
            assert!((got - e).abs() < 1e-12, "pixel {i}: {got} against {e}");
        }
        // No offset, no change
        let before = raster();
        let mut same = raster();
        same.lower_sea_to_mean_sea_level(0.0);
        assert_eq!(same.elevation, before.elevation);
    }

    /// 21 × 21 pixels of 0.001° × 0.0005° (≈ 49 × 56 m) at 63.7° N, all at
    /// `depth` below mean sea level.
    fn flat_sea(depth: f64) -> BedRaster {
        let bbox = GeoBoundingBox::new(8.0, 63.7, 8.021, 63.7105);
        BedRaster::new(bbox, 21, 21, vec![-depth; 21 * 21])
    }

    /// The position of pixel `(row, col)`'s centre in [`flat_sea`].
    fn station(row: usize, col: usize, depth: f64) -> Sounding {
        Sounding {
            lon: 8.0 + (col as f64 + 0.5) * 0.001,
            lat: 63.7105 - (row as f64 + 0.5) * 0.0005,
            depth,
        }
    }

    /// A station 20 m deeper than the raster deepens its pixel to it, the
    /// sea around it less with distance, and nothing beyond the range or on
    /// land.
    #[test]
    fn soundings_correct_the_sea_within_their_range_only() {
        let mut r = flat_sea(10.0);
        r.elevation[10 * 21 + 12] = 2.0; // land, 99 m east of the station
        let report = r.correct_to_soundings(&[station(10, 10, 30.0)], 300.0, 0.01);
        assert_eq!((report.stations, report.used), (1, 1));
        assert!((report.deeper_than_raster[0] - 20.0).abs() < 1e-12);
        assert!(report.deeper_than_corrected[0].abs() < 1e-3);
        assert!((r.pixel(10, 10) + 30.0).abs() < 1e-3, "{}", r.pixel(10, 10));
        // 99 m west: deeper, by less; 198 m: less again; 346 m: untouched
        let (near, far) = (r.pixel(10, 8), r.pixel(10, 6));
        assert!(near < far && far < -10.0, "{near} {far}");
        assert_eq!(r.pixel(10, 3), -10.0);
        assert_eq!(r.pixel(4, 10), -10.0); // 334 m north
        assert_eq!(r.pixel(10, 12), 2.0);
        assert!((report.deepened_by - 20.0).abs() < 1e-3);
        assert_eq!((report.shoaled_by, report.held), (0.0, 0));
        // The pixels within 300 m: π · 300² / (49.4 × 55.7 m²) ≈ 103
        assert!(
            report.pixels > 90 && report.pixels < 115,
            "{}",
            report.pixels
        );
    }

    /// A station shallower than the raster lifts the sea no higher than the
    /// shallowest sounded depth, and a pixel shallower than that stays; the
    /// stations off the raster and on its land are left out.
    #[test]
    fn a_correction_keeps_the_sea_in_the_sea() {
        let mut r = flat_sea(3.0);
        r.elevation[10 * 21 + 11] = -0.5;
        r.elevation[0] = 2.0;
        let off = Sounding {
            lon: 7.9,
            ..station(10, 10, 50.0)
        };
        let soundings = [station(10, 10, 0.2), off, station(0, 0, 40.0)];
        let report = r.correct_to_soundings(&soundings, 300.0, 0.01);
        assert_eq!(
            (
                report.stations,
                report.off_raster,
                report.on_land,
                report.used
            ),
            (3, 1, 1, 1)
        );
        assert_eq!(r.pixel(10, 10), -BedRaster::SHALLOWEST_SOUNDED);
        assert_eq!(r.pixel(10, 11), -0.5);
        assert_eq!(r.pixel(0, 0), 2.0);
        assert!(report.held >= 2);
        assert!((report.shoaled_by - 2.0).abs() < 1e-12);
        // Farther out the lift is less than 2 m and not held
        let lifted = r.pixel(10, 7);
        assert!(lifted > -3.0 && lifted < -1.0, "{lifted}");
    }

    #[test]
    fn pixel_size_is_the_shorter_side_in_metres() {
        // 0.01° of latitude ≈ 1.1 km, 0.1° of longitude at 63.7° ≈ 4.9 km
        let size = raster().pixel_size();
        assert!((size - 1113.0).abs() < 5.0, "{size}");
    }

    #[test]
    fn water_fraction_counts_pixels_below_sea_level() {
        assert!((raster().water_fraction() - 10.0 / 12.0).abs() < 1e-15);
    }

    /// 6 × 4 pixels of 0.1° × 0.05° over [8, 8.6] × [63.6, 63.8] with depths
    /// −(10·row + col + 1), except a dry pixel (0) at (1, 1), a land height
    /// (3) at (1, 2) and no-data at (2, 3); and a square island around the
    /// centre of pixel (2, 4) (8.45° E, 63.675° N), 0.06° × 0.03°.
    fn bathymetry_and_island() -> (GeoTiffBathymetry, CoastlineData) {
        let mut values: Vec<f32> = (0..4)
            .flat_map(|row| (0..6).map(move |col| -((10 * row + col + 1) as f32)))
            .collect();
        values[6 + 1] = 0.0;
        values[6 + 2] = 3.0;
        values[2 * 6 + 3] = -9999.0;
        let bbox = GeoBoundingBox::new(8.0, 63.6, 8.6, 63.8);
        let tiff = GeoTiffBathymetry::from_pixels(bbox, 6, 4, &values, -9999.0);
        let island = vec![(8.42, 63.66), (8.48, 63.66), (8.48, 63.69), (8.42, 63.69)];
        (tiff, CoastlineData::from_polygons(&[island], &bbox))
    }

    #[test]
    fn from_geotiff_crops_to_the_window_and_keeps_the_sea_bed() {
        let (tiff, coast) = bathymetry_and_island();
        // A window inside pixels 1..5 × 1..4: cropped to those whole pixels
        let window = GeoBoundingBox::new(8.12, 63.61, 8.47, 63.74);
        let raster = BedRaster::from_geotiff(&tiff, Some(&coast), 5.0, &window, 4);
        assert_eq!(raster.dimensions(), (4, 3));
        let b = raster.bbox();
        assert!((b.min_lon - 8.1).abs() < 1e-12 && (b.max_lon - 8.5).abs() < 1e-12);
        assert!((b.min_lat - 63.6).abs() < 1e-12 && (b.max_lat - 63.75).abs() < 1e-12);
        // Sea bed: dry, land and no-data pixels at 0; the island's pixel keeps
        // its depth (the mask makes it land)
        #[rustfmt::skip]
        let expected = [
            0.0, 0.0, -14.0, -15.0,
            -22.0, -23.0, 0.0, -25.0,
            -32.0, -33.0, -34.0, -35.0,
        ];
        for row in 0..3 {
            for col in 0..4 {
                assert_eq!(
                    raster.pixel(row, col),
                    expected[row * 4 + col],
                    "({row}, {col})"
                );
            }
        }
    }

    #[test]
    fn land_mask_follows_the_coastline_at_its_own_resolution() {
        let (tiff, coast) = bathymetry_and_island();
        let window = GeoBoundingBox::new(8.0, 63.6, 8.6, 63.8);
        let raster = BedRaster::from_geotiff(&tiff, Some(&coast), 5.0, &window, 4);
        // Inside the island, a quarter pixel from its edge, and just outside
        assert_eq!(raster.elevation(63.675, 8.45), 5.0);
        assert_eq!(raster.elevation(63.68, 8.47), 5.0);
        assert!(!raster.is_land(63.675, 8.495));
        let beside = raster.elevation(63.675, 8.495);
        assert!(beside < -20.0, "{beside}");
        // The coastline decides: the raster's dry pixels (1, 1) and (1, 2)
        // are shallow water, the sea bed going to 0 at their centres, not a
        // ramp up to the land elevation
        assert!(!raster.is_land(63.725, 8.15));
        assert!(raster.elevation(63.725, 8.15).abs() < 1e-9);
        let shore = raster.elevation(63.725, 8.28);
        assert!((-14.0..0.0).contains(&shore), "{shore}");
        // Mask cells of 0.025° × 0.0125° (≈ 1.24 × 1.39 km): the resolution
        assert!(
            (raster.pixel_size() - 1236.0).abs() < 10.0,
            "{}",
            raster.pixel_size()
        );
        // 24 pixels × 16 cells: the island covers 4 to 9 cells
        let water = raster.water_fraction();
        assert!(
            (1.0 - 9.0 / 384.0..=1.0 - 4.0 / 384.0).contains(&water),
            "{water}"
        );

        // Without the coastline the island is water, and the dry, land and
        // no-data pixels are land
        let raster = BedRaster::from_geotiff(&tiff, None, 5.0, &window, 4);
        assert!(!raster.is_land(63.675, 8.45));
        assert!((raster.elevation(63.675, 8.45) - (-25.0)).abs() < 1e-9);
        assert_eq!(raster.elevation(63.725, 8.15), 5.0);
        assert_eq!(raster.elevation(63.725, 8.29), 5.0);
        assert_eq!(raster.elevation(63.675, 8.35), 5.0);
        assert!((raster.water_fraction() - 21.0 / 24.0).abs() < 1e-12);
    }

    #[test]
    fn elevation_model_keeps_land_heights_and_rejects_gaps() {
        let (tiff, _) = bathymetry_and_island();
        // Pixels 3..6 × 0..2 have data (depths), pixel (1, 2) a land height
        let window = GeoBoundingBox::new(8.21, 63.71, 8.59, 63.79);
        let dem = BedRaster::elevation_model(&tiff, &window).unwrap();
        assert_eq!(dem.dimensions(), (4, 2));
        assert_eq!(dem.pixel(1, 0), 3.0);
        assert_eq!(dem.pixel(0, 3), -6.0);
        assert!(dem.is_land(63.725, 8.25));
        // Real heights, not a land elevation: bilinear through the land pixel
        let mid = dem.elevation(63.725, 8.3);
        assert!((mid - 0.5 * (3.0 - 14.0)).abs() < 1e-9, "{mid}");
        // Capped land keeps the foreshore below the cap
        let capped = dem.clone().clamp_land(1.0);
        assert_eq!(capped.pixel(1, 0), 1.0);
        assert_eq!(capped.pixel(0, 3), -6.0);
        // The no-data pixel (2, 3) makes a larger window fail
        let window = GeoBoundingBox::new(8.0, 63.6, 8.6, 63.8);
        assert!(matches!(
            BedRaster::elevation_model(&tiff, &window),
            Err(GeoTiffError::MissingData(_))
        ));
    }

    #[test]
    fn fill_holes_interpolates_harmonically() {
        // A linear bed with a 3 × 3 hole of zeros inside and one at the edge
        let (w, h) = (9, 7);
        let bed = |row: usize, col: usize| -50.0 + 2.0 * col as f64 - 3.0 * row as f64;
        let mut values: Vec<f64> = (0..h)
            .flat_map(|r| (0..w).map(move |c| bed(r, c)))
            .collect();
        for r in 2..5 {
            for c in 3..6 {
                values[r * w + c] = 0.0;
            }
        }
        values[w - 1] = 0.0;
        let mut raster = BedRaster::new(GeoBoundingBox::new(8.0, 63.0, 8.9, 63.7), w, h, values);
        assert_eq!(raster.fill_holes(|b| b == 0.0), 10);
        // Linear inside; the corner hole, with no boundary value beyond the
        // raster, takes the mean of its two neighbours
        assert!((raster.pixel(0, w - 1) - 0.5 * (bed(0, w - 2) + bed(1, w - 1))).abs() < 0.01);
        for r in 0..h {
            for c in (0..w).filter(|&c| (r, c) != (0, w - 1)) {
                let error = (raster.pixel(r, c) - bed(r, c)).abs();
                assert!(
                    error < 0.01,
                    "({r}, {c}): {} vs {}",
                    raster.pixel(r, c),
                    bed(r, c)
                );
            }
        }
        // A raster of holes only is left alone
        let mut empty = BedRaster::new(
            GeoBoundingBox::new(8.0, 63.0, 8.2, 63.2),
            2,
            2,
            vec![0.0; 4],
        );
        assert_eq!(empty.fill_holes(|b| b == 0.0), 0);
    }

    #[test]
    #[should_panic(expected = "12 values")]
    fn wrong_size_panics() {
        BedRaster::new(GeoBoundingBox::new(0.0, 0.0, 1.0, 1.0), 3, 3, vec![0.0; 12]);
    }
}

//! GeoTIFF bathymetry reader.
//!
//! Loads bathymetry data from GeoTIFF files and provides depth interpolation.
//! Uses pure Rust `tiff` crate - no system dependencies required.
//!
//! # Example
//!
//! ```ignore
//! use std::path::Path;
//! use dg::io::GeoTiffBathymetry;
//!
//! let bathy = GeoTiffBathymetry::load(Path::new("data/bathymetry.tif"))?;
//!
//! // Get depth at a geographic coordinate
//! if let Some(depth) = bathy.get_depth(63.8, 8.9) {
//!     println!("Depth: {} m", depth);
//! }
//! ```

use std::fmt;
use std::fs::File;
use std::path::Path;

use thiserror::Error;
use tiff::decoder::{ChunkType, Decoder, DecodingResult};
use tiff::tags::Tag;

use super::projection::GeoBoundingBox;

/// Error type for GeoTIFF operations.
#[derive(Debug, Error)]
pub enum GeoTiffError {
    /// File I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// TIFF decoding error
    #[error("TIFF error: {0}")]
    Tiff(String),

    /// Missing or invalid geotransform tags
    #[error("Missing geotransform: {0}")]
    MissingGeotransform(String),

    /// Unsupported data type
    #[error("Unsupported data type: {0}")]
    UnsupportedDataType(String),

    /// The raster is not in geographic (longitude/latitude) coordinates, or
    /// is rotated
    #[error("Unsupported georeferencing: {0}")]
    UnsupportedGeoreferencing(String),

    /// Pixels without data where a complete raster is required
    #[error("Missing data: {0}")]
    MissingData(String),
}

/// GeoTIFF tags (OGC GeoTIFF 1.1, 19-008r4) and GDAL's no-data tag.
///
/// The `tiff` crate decodes these numbers (33550, 33922, 34264, 34735,
/// 42113) to named variants, so `Tag::Unknown(n)` never matches them: that is
/// how the loader used to miss every georeferencing tag and fall back to the
/// bbox hint.
const MODEL_PIXEL_SCALE: Tag = Tag::ModelPixelScaleTag;
const MODEL_TIEPOINT: Tag = Tag::ModelTiepointTag;
const MODEL_TRANSFORMATION: Tag = Tag::ModelTransformationTag;
const GEO_KEY_DIRECTORY: Tag = Tag::GeoKeyDirectoryTag;
const GDAL_NODATA: Tag = Tag::GdalNodata;

/// GeoKeys: GTModelTypeGeoKey (2 = geographic) and GTRasterTypeGeoKey
/// (1 = PixelIsArea, 2 = PixelIsPoint).
const MODEL_TYPE_KEY: u16 = 1024;
const RASTER_TYPE_KEY: u16 = 1025;
const MODEL_TYPE_GEOGRAPHIC: u16 = 2;
const RASTER_PIXEL_IS_POINT: u16 = 2;

/// Value of `key` in a GeoKeyDirectory (`[version, revision, minor, n,
/// (key, location, count, value) × n]`), for keys stored inline.
fn geo_key(directory: &[u16], key: u16) -> Option<u16> {
    directory
        .get(4..)?
        .as_chunks::<4>()
        .0
        .iter()
        .find(|entry| entry[0] == key && entry[1] == 0)
        .map(|entry| entry[3])
}

/// Extent (outer pixel edges) of a `width` × `height` north-up raster in
/// longitude/latitude, from ModelTransformation, or else ModelPixelScale plus
/// ModelTiepoint. `None` if the raster has neither.
///
/// `pixel_is_point`: the tie point refers to a pixel centre instead of its
/// upper-left corner (GTRasterTypeGeoKey = PixelIsPoint).
fn raster_extent(
    width: usize,
    height: usize,
    pixel_scale: Option<&[f64]>,
    tiepoint: Option<&[f64]>,
    transformation: Option<&[f64]>,
    pixel_is_point: bool,
) -> Result<Option<GeoBoundingBox>, GeoTiffError> {
    // Longitude/latitude of raster (col, row) = (0, 0) and the pixel size
    // (positive; rows go south)
    let (lon0, lat0, dlon, dlat) = if let Some(m) = transformation {
        // Row-major 4×4: lon = m0·col + m1·row + m3, lat = m4·col + m5·row + m7
        if m.len() < 16 {
            return Err(GeoTiffError::MissingGeotransform(format!(
                "ModelTransformation has {} values, expected 16",
                m.len()
            )));
        }
        if m[1] != 0.0 || m[4] != 0.0 || m[0] <= 0.0 || m[5] >= 0.0 {
            return Err(GeoTiffError::UnsupportedGeoreferencing(format!(
                "only north-up, unrotated rasters are supported (ModelTransformation {:?})",
                &m[..8]
            )));
        }
        (m[3], m[7], m[0], -m[5])
    } else if let (Some(scale), Some(tie)) = (pixel_scale, tiepoint) {
        // Tie point [I, J, K, X, Y, Z]: raster (I, J) lies at (X, Y)
        if tie.len() < 6 || scale.len() < 2 {
            return Err(GeoTiffError::MissingGeotransform(
                "ModelTiepoint/ModelPixelScale too short".to_string(),
            ));
        }
        (
            tie[3] - tie[0] * scale[0],
            tie[4] + tie[1] * scale[1],
            scale[0],
            scale[1],
        )
    } else {
        return Ok(None);
    };

    let (min_lon, max_lat) = if pixel_is_point {
        (lon0 - 0.5 * dlon, lat0 + 0.5 * dlat)
    } else {
        (lon0, lat0)
    };
    Ok(Some(GeoBoundingBox::new(
        min_lon,
        max_lat - height as f64 * dlat,
        min_lon + width as f64 * dlon,
        max_lat,
    )))
}

impl From<tiff::TiffError> for GeoTiffError {
    fn from(e: tiff::TiffError) -> Self {
        GeoTiffError::Tiff(e.to_string())
    }
}

/// GeoTIFF-based bathymetry provider.
///
/// Loads depth data from a GeoTIFF file and provides interpolation methods.
/// Depth values are stored as negative (below sea level) following oceanographic convention.
pub struct GeoTiffBathymetry {
    /// Depth data (rows x cols), stored as f32 for memory efficiency
    depths: Vec<Vec<f32>>,
    /// Geographic bounding box
    bbox: GeoBoundingBox,
    /// Width in pixels
    width: usize,
    /// Height in pixels
    height: usize,
    /// No data value
    nodata: f32,
}

/// The pixel values as `f32`.
fn to_f32(result: DecodingResult) -> Vec<f32> {
    match result {
        DecodingResult::U8(data) => data.into_iter().map(|v| v as f32).collect(),
        DecodingResult::U16(data) => data.into_iter().map(|v| v as f32).collect(),
        DecodingResult::U32(data) => data.into_iter().map(|v| v as f32).collect(),
        DecodingResult::U64(data) => data.into_iter().map(|v| v as f32).collect(),
        DecodingResult::F32(data) => data,
        DecodingResult::F64(data) => data.into_iter().map(|v| v as f32).collect(),
        DecodingResult::I8(data) => data.into_iter().map(|v| v as f32).collect(),
        DecodingResult::I16(data) => data.into_iter().map(|v| v as f32).collect(),
        DecodingResult::I32(data) => data.into_iter().map(|v| v as f32).collect(),
        DecodingResult::I64(data) => data.into_iter().map(|v| v as f32).collect(),
    }
}

/// The image's `width` × `height` pixels, row-major from the top.
///
/// A chunk (tile or strip) with no bytes is sparse, as GDAL writes a block
/// that is all no-data or all zero: it reads as `sparse` (the no-data value,
/// else 0), as GDAL reads it. The `tiff` crate (0.9) does not know sparse
/// chunks and decodes garbage from offset 0 for them: Kartverket's 1 m level
/// writes its flat-sea tiles sparse (64 of 144 tiles around the Kattholmen
/// farm), which read as values up to 10³⁵. Files without sparse chunks are
/// decoded whole.
fn read_pixels(
    decoder: &mut Decoder<File>,
    width: u32,
    height: u32,
    sparse: f32,
) -> Result<Vec<f32>, GeoTiffError> {
    let counts_tag = match decoder.get_chunk_type() {
        ChunkType::Tile => Tag::TileByteCounts,
        ChunkType::Strip => Tag::StripByteCounts,
    };
    let counts = decoder.get_tag_u64_vec(counts_tag)?;
    if counts.iter().all(|&count| count > 0) {
        return Ok(to_f32(decoder.read_image()?));
    }
    let (width, height) = (width as usize, height as usize);
    let mut pixels = vec![sparse; width * height];
    let (chunk_width, chunk_height) = decoder.chunk_dimensions();
    let across = width.div_ceil(chunk_width as usize);
    for (index, &count) in counts.iter().enumerate() {
        if count == 0 {
            continue;
        }
        let (data_width, data_height) = decoder.chunk_data_dimensions(index as u32);
        let data = to_f32(decoder.read_chunk(index as u32)?);
        let col0 = (index % across) * chunk_width as usize;
        let row0 = (index / across) * chunk_height as usize;
        let data_width = data_width as usize;
        for r in 0..(data_height as usize).min(height - row0) {
            let start = (row0 + r) * width + col0;
            pixels[start..start + data_width]
                .copy_from_slice(&data[r * data_width..(r + 1) * data_width]);
        }
    }
    Ok(pixels)
}

impl GeoTiffBathymetry {
    /// Load bathymetry from a GeoTIFF file.
    ///
    /// The raster must be north-up in geographic coordinates (e.g. EPSG:4326),
    /// georeferenced by ModelTransformation (tag 34264) or by ModelPixelScale
    /// (33550) plus ModelTiepoint (33922). Projected rasters (UTM, …) are
    /// rejected. GDAL's no-data tag (42113) is honoured.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, GeoTiffError> {
        Self::load_with_bbox(path, None)
    }

    /// Load bathymetry, with `bbox_hint` as the extent of a plain TIFF that
    /// has no georeferencing tags. The hint is ignored when the file is
    /// georeferenced (see [`Self::load`]).
    pub fn load_with_bbox<P: AsRef<Path>>(
        path: P,
        bbox_hint: Option<GeoBoundingBox>,
    ) -> Result<Self, GeoTiffError> {
        let file = File::open(&path)?;
        let mut decoder = Decoder::new(file)?;

        let (width, height) = decoder.dimensions()?;

        let geo_keys = decoder.get_tag_u16_vec(GEO_KEY_DIRECTORY).ok();
        let key = |key| geo_keys.as_deref().and_then(|keys| geo_key(keys, key));
        if let Some(model_type) = key(MODEL_TYPE_KEY)
            && model_type != MODEL_TYPE_GEOGRAPHIC
        {
            return Err(GeoTiffError::UnsupportedGeoreferencing(format!(
                "GTModelTypeGeoKey = {model_type}: only geographic (longitude/latitude) \
                 rasters are supported; reproject to EPSG:4326 first"
            )));
        }
        let pixel_is_point = key(RASTER_TYPE_KEY) == Some(RASTER_PIXEL_IS_POINT);

        let pixel_scale = decoder.get_tag_f64_vec(MODEL_PIXEL_SCALE).ok();
        let tiepoint = decoder.get_tag_f64_vec(MODEL_TIEPOINT).ok();
        let transformation = decoder.get_tag_f64_vec(MODEL_TRANSFORMATION).ok();
        let bbox = match raster_extent(
            width as usize,
            height as usize,
            pixel_scale.as_deref(),
            tiepoint.as_deref(),
            transformation.as_deref(),
            pixel_is_point,
        )? {
            Some(bbox) => bbox,
            None => bbox_hint.ok_or_else(|| {
                GeoTiffError::MissingGeotransform(
                    "No GeoTIFF geotransform found and no bbox hint provided".to_string(),
                )
            })?,
        };
        let nodata_tag = decoder
            .get_tag_ascii_string(GDAL_NODATA)
            .ok()
            .and_then(|s| s.trim_matches(char::from(0)).trim().parse::<f32>().ok());
        let nodata = nodata_tag.unwrap_or(-9999.0);

        let depths_flat = read_pixels(&mut decoder, width, height, nodata_tag.unwrap_or(0.0))?;

        // Convert flat array to 2D array
        let mut depths = Vec::with_capacity(height as usize);
        for row in 0..height {
            let row_start = (row * width) as usize;
            let row_end = row_start + width as usize;
            depths.push(depths_flat[row_start..row_end].to_vec());
        }

        Ok(Self {
            depths,
            bbox,
            width: width as usize,
            height: height as usize,
            nodata,
        })
    }

    /// Bathymetry from pixel values in memory: `width` × `height` values,
    /// row-major with row 0 the northernmost, covering `bbox` (outer pixel
    /// edges), with `nodata` marking missing pixels.
    ///
    /// # Panics
    /// If the sizes disagree or the raster is empty.
    pub fn from_pixels(
        bbox: GeoBoundingBox,
        width: usize,
        height: usize,
        values: &[f32],
        nodata: f32,
    ) -> Self {
        assert!(width > 0 && height > 0, "empty raster");
        assert_eq!(values.len(), width * height, "raster size mismatch");
        Self {
            depths: values.chunks_exact(width).map(<[f32]>::to_vec).collect(),
            bbox,
            width,
            height,
            nodata,
        }
    }

    /// Value of pixel `(row, col)` (row 0 north), or `None` for no-data and
    /// non-finite values. Unlike the depth accessors it does not treat
    /// positive values (land heights) as missing.
    pub fn pixel(&self, row: usize, col: usize) -> Option<f64> {
        let value = self.depths[row][col];
        (value.is_finite() && (value - self.nodata).abs() > 0.01).then_some(value as f64)
    }

    /// Set the no-data value.
    pub fn set_nodata(&mut self, nodata: f32) {
        self.nodata = nodata;
    }

    /// Get the bounding box of this bathymetry data.
    pub fn bbox(&self) -> &GeoBoundingBox {
        &self.bbox
    }

    /// Get the dimensions (width, height) in pixels.
    pub fn dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    /// Convert lat/lon to pixel coordinates.
    fn latlon_to_pixel(&self, lat: f64, lon: f64) -> Option<(usize, usize)> {
        if !self.bbox.contains(lat, lon) {
            return None;
        }

        let col = ((lon - self.bbox.min_lon) / (self.bbox.max_lon - self.bbox.min_lon)
            * self.width as f64) as usize;
        let row = ((self.bbox.max_lat - lat) / (self.bbox.max_lat - self.bbox.min_lat)
            * self.height as f64) as usize;

        if row >= self.height || col >= self.width {
            return None;
        }

        Some((row, col))
    }

    /// Convert lat/lon to fractional pixel coordinates for interpolation.
    ///
    /// A pixel's value belongs to its centre, so row/column `i` sits `i + ½`
    /// pixels from the edge; points in the outer half pixels clamp to the
    /// outermost centres.
    fn latlon_to_pixel_frac(&self, lat: f64, lon: f64) -> Option<(f64, f64)> {
        if !self.bbox.contains(lat, lon) {
            return None;
        }

        let col_frac = (lon - self.bbox.min_lon) / (self.bbox.max_lon - self.bbox.min_lon)
            * self.width as f64
            - 0.5;
        let row_frac = (self.bbox.max_lat - lat) / (self.bbox.max_lat - self.bbox.min_lat)
            * self.height as f64
            - 0.5;

        Some((
            row_frac.clamp(0.0, (self.height - 1) as f64),
            col_frac.clamp(0.0, (self.width - 1) as f64),
        ))
    }

    /// Check if a depth value is valid (not nodata, not NaN, not land).
    fn is_valid_depth(&self, depth: f32) -> bool {
        !depth.is_nan()
            && !depth.is_infinite()
            && (depth - self.nodata).abs() > 0.01
            && depth <= 0.0 // Positive values indicate land elevation
    }

    /// Get depth at geographic coordinates using nearest neighbor.
    ///
    /// Returns None if the point is outside the data bounds, on land, or nodata.
    /// Depth is returned as f64 for DG precision requirements.
    pub fn get_depth(&self, lat: f64, lon: f64) -> Option<f64> {
        let (row, col) = self.latlon_to_pixel(lat, lon)?;
        let depth = self.depths[row][col];

        if self.is_valid_depth(depth) {
            Some(depth as f64)
        } else {
            None
        }
    }

    /// Get depth at geographic coordinates using bilinear interpolation.
    ///
    /// Provides smoother depth values for better gradient computation.
    /// Returns None if any of the interpolation points are invalid.
    pub fn get_depth_bilinear(&self, lat: f64, lon: f64) -> Option<f64> {
        let (row_frac, col_frac) = self.latlon_to_pixel_frac(lat, lon)?;

        let row0 = row_frac.floor() as usize;
        let col0 = col_frac.floor() as usize;
        let row1 = (row0 + 1).min(self.height - 1);
        let col1 = (col0 + 1).min(self.width - 1);

        // Get four corner depths
        let d00 = self.depths[row0][col0];
        let d01 = self.depths[row0][col1];
        let d10 = self.depths[row1][col0];
        let d11 = self.depths[row1][col1];

        // Check all corners are valid
        if !self.is_valid_depth(d00)
            || !self.is_valid_depth(d01)
            || !self.is_valid_depth(d10)
            || !self.is_valid_depth(d11)
        {
            // Fall back to nearest neighbor
            return self.get_depth(lat, lon);
        }

        // Bilinear interpolation
        let t = row_frac - row0 as f64;
        let s = col_frac - col0 as f64;

        let depth = (1.0 - t) * (1.0 - s) * d00 as f64
            + (1.0 - t) * s * d01 as f64
            + t * (1.0 - s) * d10 as f64
            + t * s * d11 as f64;

        Some(depth)
    }

    /// Check if a point is in water (valid depth data exists).
    pub fn is_water(&self, lat: f64, lon: f64) -> bool {
        self.get_depth(lat, lon).is_some()
    }

    /// Get the depth range in the data.
    pub fn depth_range(&self) -> (f64, f64) {
        let mut min_depth = f64::INFINITY;
        let mut max_depth = f64::NEG_INFINITY;

        for row in &self.depths {
            for &depth in row {
                if self.is_valid_depth(depth) {
                    let d = depth as f64;
                    min_depth = min_depth.min(d);
                    max_depth = max_depth.max(d);
                }
            }
        }

        (min_depth, max_depth)
    }

    /// Get statistics about the bathymetry data.
    pub fn statistics(&self) -> BathymetryStatistics {
        let mut valid_count = 0usize;
        let mut nodata_count = 0usize;
        let mut land_count = 0usize;
        let mut sum = 0.0f64;
        let mut min_depth = f64::INFINITY;
        let mut max_depth = f64::NEG_INFINITY;

        for row in &self.depths {
            for &depth in row {
                if (depth - self.nodata).abs() < 0.01 || depth.is_nan() || depth.is_infinite() {
                    nodata_count += 1;
                } else if depth > 0.0 {
                    land_count += 1;
                } else {
                    valid_count += 1;
                    let d = depth as f64;
                    sum += d;
                    min_depth = min_depth.min(d);
                    max_depth = max_depth.max(d);
                }
            }
        }

        let mean = if valid_count > 0 {
            sum / valid_count as f64
        } else {
            0.0
        };

        BathymetryStatistics {
            width: self.width,
            height: self.height,
            valid_count,
            nodata_count,
            land_count,
            min_depth: if min_depth.is_infinite() {
                0.0
            } else {
                min_depth
            },
            max_depth: if max_depth.is_infinite() {
                0.0
            } else {
                max_depth
            },
            mean_depth: mean,
            bbox: self.bbox,
        }
    }
}

/// Statistics about a bathymetry dataset.
#[derive(Debug, Clone)]
pub struct BathymetryStatistics {
    /// Width in pixels
    pub width: usize,
    /// Height in pixels
    pub height: usize,
    /// Number of valid water cells
    pub valid_count: usize,
    /// Number of nodata cells
    pub nodata_count: usize,
    /// Number of land cells (positive elevation)
    pub land_count: usize,
    /// Minimum depth (most negative, deepest)
    pub min_depth: f64,
    /// Maximum depth (least negative, shallowest water)
    pub max_depth: f64,
    /// Mean depth
    pub mean_depth: f64,
    /// Geographic bounding box
    pub bbox: GeoBoundingBox,
}

impl fmt::Display for BathymetryStatistics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Bathymetry Statistics:")?;
        writeln!(f, "  Dimensions: {}x{} pixels", self.width, self.height)?;
        writeln!(f, "  Total cells: {}", self.width * self.height)?;
        writeln!(f, "  Valid water cells: {}", self.valid_count)?;
        writeln!(f, "  Land cells: {}", self.land_count)?;
        writeln!(f, "  NoData cells: {}", self.nodata_count)?;
        writeln!(
            f,
            "  Depth range: {:.1} to {:.1} m",
            self.min_depth, self.max_depth
        )?;
        writeln!(f, "  Mean depth: {:.1} m", self.mean_depth)?;
        writeln!(
            f,
            "  Bounding box: lon [{:.4}, {:.4}], lat [{:.4}, {:.4}]",
            self.bbox.min_lon, self.bbox.max_lon, self.bbox.min_lat, self.bbox.max_lat
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_geo_bbox_contains() {
        let bbox = GeoBoundingBox::new(8.0, 63.5, 9.5, 64.0);
        assert!(bbox.contains(63.75, 8.75));
        assert!(!bbox.contains(65.0, 8.75));
    }

    /// Write a 4 × 3 (width × height) Float32 GeoTIFF with pixel value
    /// −(10·row + col + 1), the given georeferencing tags and GeoKeys.
    fn write_geotiff(name: &str, tags: &[(Tag, &[f64])], geo_keys: &[u16]) -> std::path::PathBuf {
        use tiff::encoder::{TiffEncoder, colortype::Gray32Float};
        let path = std::env::temp_dir().join(format!("dgcom_geotiff_{name}.tif"));
        let data: Vec<f32> = (0..3)
            .flat_map(|row| (0..4).map(move |col| -((10 * row + col + 1) as f32)))
            .collect();
        let mut encoder = TiffEncoder::new(File::create(&path).unwrap()).unwrap();
        let mut image = encoder.new_image::<Gray32Float>(4, 3).unwrap();
        for &(tag, value) in tags {
            image.encoder().write_tag(tag, value).unwrap();
        }
        image
            .encoder()
            .write_tag(GEO_KEY_DIRECTORY, geo_keys)
            .unwrap();
        image.write_data(&data).unwrap();
        path
    }

    /// Geographic (EPSG:4326), PixelIsArea
    const GEOGRAPHIC: [u16; 16] = [1, 1, 0, 3, 1024, 0, 1, 2, 1025, 0, 1, 1, 2048, 0, 1, 4326];

    /// Write a 40 × 16 Float32 GeoTIFF in three 16 × 16 tiles (the last
    /// partial) with pixel value row·100 + col + 1, and the middle tile
    /// sparse (no bytes, offset 0), as GDAL and Kartverket's service write an
    /// all-zero tile; with a GDAL no-data tag if given. Written by hand: the
    /// `tiff` crate's encoder writes neither tiles nor sparse chunks.
    fn write_sparse_tiled(name: &str, nodata: Option<&str>) -> std::path::PathBuf {
        const TILE: u32 = 16;
        let path = std::env::temp_dir().join(format!("dgcom_geotiff_{name}.tif"));
        let tile = |t: u32| -> Vec<u8> {
            (0..TILE)
                .flat_map(|r| (0..TILE).map(move |c| (r, t * TILE + c)))
                .flat_map(|(r, c)| ((r * 100 + c + 1) as f32).to_le_bytes())
                .collect()
        };
        // Header, then tiles 0 and 2, then the arrays the IFD points to
        let mut bytes: Vec<u8> = vec![b'I', b'I', 42, 0, 0, 0, 0, 0];
        let tile_offsets = [8u32, 0, 8 + 4 * TILE * TILE];
        let tile_counts = [4 * TILE * TILE, 0, 4 * TILE * TILE];
        bytes.extend(tile(0));
        bytes.extend(tile(2));
        let extra = |bytes: &mut Vec<u8>, data: Vec<u8>| -> u32 {
            let at = bytes.len() as u32;
            bytes.extend(data);
            at
        };
        let le32 = |v: &[u32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let le64 = |v: &[f64]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let offsets_at = extra(&mut bytes, le32(&tile_offsets));
        let counts_at = extra(&mut bytes, le32(&tile_counts));
        let scale_at = extra(&mut bytes, le64(&[0.01, 0.01, 0.0]));
        let tie_at = extra(&mut bytes, le64(&[0.0, 0.0, 0.0, 8.0, 64.0, 0.0]));
        let keys: Vec<u8> = GEOGRAPHIC.iter().flat_map(|k| k.to_le_bytes()).collect();
        let keys_at = extra(&mut bytes, keys);
        let nodata = nodata.map(|s| format!("{s}\0"));
        let nodata_at = nodata
            .as_ref()
            .map(|s| extra(&mut bytes, s.as_bytes().to_vec()));
        // (tag, type, count, value or offset): SHORT 3, LONG 4, ASCII 2, DOUBLE 12
        let mut entries: Vec<(u16, u16, u32, u32)> = vec![
            (256, 4, 1, 40),
            (257, 4, 1, TILE),
            (258, 3, 1, 32),
            (259, 3, 1, 1),
            (262, 3, 1, 1),
            (277, 3, 1, 1),
            (322, 4, 1, TILE),
            (323, 4, 1, TILE),
            (324, 4, 3, offsets_at),
            (325, 4, 3, counts_at),
            (339, 3, 1, 3),
            (33550, 12, 3, scale_at),
            (33922, 12, 6, tie_at),
            (34735, 3, 16, keys_at),
        ];
        if let (Some(s), Some(at)) = (&nodata, nodata_at) {
            entries.push((42113, 2, s.len() as u32, at));
        }
        if bytes.len() % 2 == 1 {
            bytes.push(0);
        }
        let ifd_at = bytes.len() as u32;
        bytes[4..8].copy_from_slice(&ifd_at.to_le_bytes());
        bytes.extend((entries.len() as u16).to_le_bytes());
        for (tag, kind, count, value) in entries {
            bytes.extend(tag.to_le_bytes());
            bytes.extend(kind.to_le_bytes());
            bytes.extend(count.to_le_bytes());
            bytes.extend(value.to_le_bytes());
        }
        bytes.extend(0u32.to_le_bytes());
        std::fs::write(&path, bytes).unwrap();
        path
    }

    /// Regression: a sparse tile (no bytes) read as garbage from offset 0
    /// (the `tiff` crate does not know them), up to 10³⁵ in Kartverket's
    /// 1 m level, whose flat-sea tiles are sparse. It reads as 0, or as
    /// no-data with a no-data tag; the other tiles, the partial one too, are
    /// in place.
    #[test]
    fn sparse_tiles_read_as_zero_or_no_data() {
        let bathy = GeoTiffBathymetry::load(write_sparse_tiled("sparse", None)).unwrap();
        assert_eq!(bathy.dimensions(), (40, 16));
        for row in 0..16 {
            for col in 0..40 {
                let expected = if (16..32).contains(&col) {
                    0.0
                } else {
                    (row * 100 + col + 1) as f64
                };
                assert_eq!(bathy.pixel(row, col), Some(expected), "({row}, {col})");
            }
        }
        let bathy =
            GeoTiffBathymetry::load(write_sparse_tiled("sparse_nodata", Some("-9999"))).unwrap();
        assert_eq!(bathy.pixel(3, 20), None);
        assert_eq!(bathy.pixel(3, 35), Some(336.0));
    }

    /// Origin (7°E, 64°N), 0.5° × 0.25° pixels: extent 7–9°E, 63.25–64°N
    const TRANSFORMATION: [f64; 16] = [
        0.5, 0.0, 0.0, 7.0, 0.0, -0.25, 0.0, 64.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];

    /// Centre of pixel (row, col): (lat, lon)
    fn centre(row: f64, col: f64) -> (f64, f64) {
        (64.0 - 0.25 * (row + 0.5), 7.0 + 0.5 * (col + 0.5))
    }

    #[test]
    fn test_model_transformation_georeferencing() {
        // Regression: ModelTransformation (tag 34264) was ignored and the
        // extent silently taken from the bbox hint, which misplaced the
        // Frøya bathymetry by up to ~20 km
        let path = write_geotiff(
            "transformation",
            &[(MODEL_TRANSFORMATION, &TRANSFORMATION)],
            &GEOGRAPHIC,
        );
        let wrong_hint = GeoBoundingBox::new(7.5, 63.3, 10.0, 64.2);
        let bathy = GeoTiffBathymetry::load_with_bbox(&path, Some(wrong_hint)).unwrap();
        let bbox = bathy.bbox();
        assert_eq!(
            (bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat),
            (7.0, 63.25, 9.0, 64.0)
        );

        // Nearest and bilinear sampling return a pixel's value at its centre
        for row in 0..3 {
            for col in 0..4 {
                let (lat, lon) = centre(row as f64, col as f64);
                let value = -((10 * row + col + 1) as f64);
                assert_eq!(bathy.get_depth(lat, lon), Some(value));
                let bilinear = bathy.get_depth_bilinear(lat, lon).unwrap();
                assert!(
                    (bilinear - value).abs() < 1e-12,
                    "({row}, {col}): {bilinear}"
                );
            }
        }
        // Halfway between the centres of (1, 1) and (1, 2): their mean
        let (lat, lon) = centre(1.0, 1.5);
        let mid = bathy.get_depth_bilinear(lat, lon).unwrap();
        assert!((mid - (-12.5)).abs() < 1e-12, "{mid}");
    }

    #[test]
    fn test_tiepoint_georeferencing_matches_transformation() {
        let scale: [f64; 3] = [0.5, 0.25, 0.0];
        // Tie raster (1, 2) to its corner in model space
        let tiepoint: [f64; 6] = [1.0, 2.0, 0.0, 7.5, 63.5, 0.0];
        let path = write_geotiff(
            "tiepoint",
            &[(MODEL_PIXEL_SCALE, &scale), (MODEL_TIEPOINT, &tiepoint)],
            &GEOGRAPHIC,
        );
        let bathy = GeoTiffBathymetry::load(&path).unwrap();
        let bbox = bathy.bbox();
        assert_eq!(
            (bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat),
            (7.0, 63.25, 9.0, 64.0)
        );
    }

    #[test]
    fn test_projected_raster_is_rejected() {
        // GTModelTypeGeoKey = 1 (projected, e.g. UTM 33N): coordinates are in
        // metres, and reading them as degrees gives nonsense
        let mut projected = GEOGRAPHIC;
        projected[7] = 1;
        let path = write_geotiff(
            "projected",
            &[(MODEL_TRANSFORMATION, &TRANSFORMATION)],
            &projected,
        );
        assert!(matches!(
            GeoTiffBathymetry::load(&path),
            Err(GeoTiffError::UnsupportedGeoreferencing(_))
        ));
    }
}

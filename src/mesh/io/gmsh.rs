//! Gmsh mesh file I/O.
//!
//! Reads Gmsh MSH 4.1 (ASCII and binary, the Gmsh default since 4.0) and
//! MSH 2.2 (ASCII); writes MSH 2.2 (ASCII). Coastline-fitted meshes are made
//! in Gmsh, so this is how real domains enter the model (TODO P1.3).
//!
//! ## What the solver takes
//!
//! A 2D mesh of first-order quadrilaterals (Gmsh type 3). Any other element
//! type is an error, not silently dropped: triangles from a quad-dominant mesh
//! (recombine *and* subdivide in Gmsh for an all-quad mesh:
//! `Mesh.RecombineAll = 1; Mesh.SubdivisionAlgorithm = 1;`), high-order
//! elements (`Mesh.ElementOrder = 1`) and 3D elements. Points (type 15) are
//! ignored, and 2-node lines (type 1) carry boundary tags.
//!
//! ## Nodes and elements
//!
//! Node tags may be sparse and in any order. The mesh keeps the nodes that the
//! quadrilaterals use, in file order, and the quadrilaterals in file order.
//! Clockwise quadrilaterals (a surface whose normal points down) are reordered
//! counter-clockwise, as `Mesh2D` requires. A quadrilateral that is degenerate
//! or non-convex has a bilinear map with a non-positive Jacobian and is an
//! error, as are edges shared by more than two elements and neighbours that
//! overlap (traverse their shared edge in the same direction).
//!
//! ## Boundary tags
//!
//! A boundary edge takes its tag from the physical group of the line element
//! on it (in MSH 4.1 the physical groups of the line's curve entity):
//!
//! - a named group by its name, case-insensitive, with spaces and dashes read
//!   as `_`: `wall`, `walls`, `land`, `coast`, `coastline`, `shore`,
//!   `shoreline`, `closed` → [`BoundaryTag::Wall`]; `open`, `ocean`, `sea`,
//!   `open_boundary` → [`BoundaryTag::Open`]; `tidal`, `tide`, `tides`,
//!   `tidal_forcing` → [`BoundaryTag::TidalForcing`]; `river`, `rivers` →
//!   [`BoundaryTag::River`]; any other name → [`BoundaryTag::Custom`] of the
//!   group's number;
//! - an unnamed group by its number: 1 wall, 2 open, 3 tidal forcing, 4 river,
//!   5 Dirichlet, 6 Neumann, other n → `Custom(n)`.
//!
//! Boundary edges without a line element are walls. An edge whose groups map
//! to different tags is an error. Lines on interior edges (embedded curves)
//! are ignored; lines that are not an edge of the mesh are an error.
//!
//! Only elements in physical groups are saved once a model has any
//! (`Mesh.SaveAll = 0`), so the domain needs a physical surface too.
//!
//! Not supported (error): MSH 4.0, binary MSH 2, periodic meshes
//! (`$Periodic`), partitioned meshes. Other sections (`$NodeData`,
//! `$Comments`, ...) are skipped.
//!
//! ## Example
//! ```no_run
//! use dgcom::mesh::read_gmsh_mesh;
//! use std::path::Path;
//!
//! let mesh = read_gmsh_mesh(Path::new("mesh.msh")).expect("Failed to read mesh");
//! ```

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::str::FromStr;

use thiserror::Error;

use crate::mesh::core::{Mesh2D, QuadMeshError};
use crate::mesh::data::BoundaryTag;

/// Error type for Gmsh I/O operations.
#[derive(Debug, Error)]
pub enum GmshError {
    /// File could not be read or written.
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// Malformed file: bad number, truncated section, missing end marker.
    #[error("Parse error: {0}")]
    ParseError(String),

    /// Unsupported MSH format version or file type.
    #[error("Unsupported Gmsh version: {0}")]
    UnsupportedVersion(String),

    /// Element types the solver cannot use: `(Gmsh element type, count)`.
    #[error("{}", describe_unsupported(.0))]
    UnsupportedElements(Vec<(i32, usize)>),

    /// A file feature the reader does not support (periodic, partitioned).
    #[error("Unsupported Gmsh feature: {0}")]
    Unsupported(String),

    /// Missing required section.
    #[error("Missing section: {0}")]
    MissingSection(String),

    /// The file parses but is not a valid quadrilateral mesh.
    #[error("Invalid mesh: {0}")]
    InvalidMesh(String),
}

type Result<T> = std::result::Result<T, GmshError>;

/// Read a Gmsh MSH file (4.1 ASCII or binary, 2.2 ASCII).
///
/// See the module docs for what is accepted and how boundary tags are set.
pub fn read_gmsh_mesh(path: &Path) -> Result<Mesh2D> {
    let bytes = std::fs::read(path)?;
    parse_gmsh_mesh(&bytes)
}

/// Parse the contents of a Gmsh MSH file (4.1 ASCII or binary, 2.2 ASCII).
pub fn parse_gmsh_mesh(bytes: &[u8]) -> Result<Mesh2D> {
    build_mesh(parse_msh(bytes)?)
}

// =============================================================================
// Element types
// =============================================================================

const GMSH_LINE: i32 = 1;
const GMSH_QUAD: i32 = 3;
const GMSH_POINT: i32 = 15;

/// Node count and name of a Gmsh element type (the fixed-order types).
fn element_type_info(element_type: i32) -> Option<(usize, &'static str)> {
    Some(match element_type {
        1 => (2, "2-node line"),
        2 => (3, "3-node triangle"),
        3 => (4, "4-node quadrangle"),
        4 => (4, "4-node tetrahedron"),
        5 => (8, "8-node hexahedron"),
        6 => (6, "6-node prism"),
        7 => (5, "5-node pyramid"),
        8 => (3, "3-node line"),
        9 => (6, "6-node triangle"),
        10 => (9, "9-node quadrangle"),
        11 => (10, "10-node tetrahedron"),
        12 => (27, "27-node hexahedron"),
        13 => (18, "18-node prism"),
        14 => (14, "14-node pyramid"),
        15 => (1, "1-node point"),
        16 => (8, "8-node quadrangle"),
        17 => (20, "20-node hexahedron"),
        18 => (15, "15-node prism"),
        19 => (13, "13-node pyramid"),
        20 => (9, "9-node triangle"),
        21 => (10, "10-node triangle"),
        22 => (12, "12-node triangle"),
        23 => (15, "15-node triangle"),
        24 => (15, "15-node incomplete triangle"),
        25 => (21, "21-node triangle"),
        26 => (4, "4-node line"),
        27 => (5, "5-node line"),
        28 => (6, "6-node line"),
        29 => (20, "20-node tetrahedron"),
        30 => (35, "35-node tetrahedron"),
        31 => (56, "56-node tetrahedron"),
        36 => (16, "16-node quadrangle"),
        37 => (25, "25-node quadrangle"),
        38 => (36, "36-node quadrangle"),
        _ => return None,
    })
}

fn describe_unsupported(counts: &[(i32, usize)]) -> String {
    let list: Vec<String> = counts
        .iter()
        .map(|&(t, n)| {
            let name = element_type_info(t).map_or("unknown element", |(_, name)| name);
            format!("{n} × {name} (type {t})")
        })
        .collect();
    format!(
        "unsupported Gmsh elements: {}. The solver takes 2D meshes of first-order \
         quadrilaterals: recombine and subdivide in Gmsh (Mesh.RecombineAll = 1; \
         Mesh.SubdivisionAlgorithm = 1), keep Mesh.ElementOrder = 1 and mesh in 2D",
        list.join(", ")
    )
}

// =============================================================================
// Parsed file contents
// =============================================================================

/// Where a line element's physical groups come from.
#[derive(Clone, Copy, Debug)]
enum LineGroups {
    /// MSH 4.1: the physical groups of this curve entity.
    Curve(i32),
    /// MSH 2.2: this physical group (0 = none).
    Physical(i32),
}

/// The file contents the mesh is built from, in Gmsh tags.
#[derive(Default)]
struct RawMesh {
    /// (node tag, [x, y]) in file order
    nodes: Vec<(u64, [f64; 2])>,
    /// (element tag, node tags)
    quads: Vec<(u64, [u64; 4])>,
    /// (element tag, node tags, groups)
    lines: Vec<(u64, [u64; 2], LineGroups)>,
    /// (dimension, physical tag) → name
    physical_names: HashMap<(i32, i32), String>,
    /// curve entity tag → physical tags (MSH 4.1 `$Entities`)
    curve_groups: HashMap<i32, Vec<i32>>,
    /// element type → count, for the types the solver cannot use
    unsupported: BTreeMap<i32, usize>,
}

impl RawMesh {
    /// Keep an element; `nodes` has the node count of its type.
    fn add_element(&mut self, element_type: i32, tag: u64, nodes: &[u64], groups: LineGroups) {
        match element_type {
            GMSH_QUAD => self
                .quads
                .push((tag, [nodes[0], nodes[1], nodes[2], nodes[3]])),
            GMSH_LINE => self.lines.push((tag, [nodes[0], nodes[1]], groups)),
            GMSH_POINT => {}
            _ => *self.unsupported.entry(element_type).or_default() += 1,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Version {
    V2,
    V41,
}

// =============================================================================
// Reader: ASCII tokens and binary values over the file bytes
// =============================================================================

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
    /// Binary `$Entities`/`$Nodes`/`$Elements` (MSH 4.1 file-type 1)
    binary: bool,
    big_endian: bool,
    /// Bytes of a `size_t` in binary sections (the header's data-size)
    size_bytes: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self {
            buf,
            pos: 0,
            binary: false,
            big_endian: false,
            size_bytes: 8,
        }
    }

    /// A parse error at the current position.
    fn error(&self, msg: impl std::fmt::Display) -> GmshError {
        let line = 1 + self.buf[..self.pos].iter().filter(|&&b| b == b'\n').count();
        GmshError::ParseError(format!("{msg} (line {line}, byte {})", self.pos))
    }

    fn skip_ws(&mut self) {
        while self.pos < self.buf.len() && self.buf[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    fn at_end(&mut self) -> bool {
        self.skip_ws();
        self.pos >= self.buf.len()
    }

    /// The next whitespace-delimited ASCII token.
    fn token(&mut self, what: &str) -> Result<&'a str> {
        self.skip_ws();
        let start = self.pos;
        while self.pos < self.buf.len() && !self.buf[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
        if start == self.pos {
            return Err(self.error(format!("unexpected end of file, expected {what}")));
        }
        std::str::from_utf8(&self.buf[start..self.pos])
            .map_err(|_| self.error(format!("invalid {what}: not ASCII")))
    }

    fn parse<T: FromStr>(&mut self, what: &str) -> Result<T> {
        let token = self.token(what)?;
        token
            .parse()
            .map_err(|_| self.error(format!("invalid {what}: {token:?}")))
    }

    /// The rest of the current line, without the line break.
    fn line(&mut self) -> Result<&'a str> {
        let start = self.pos;
        while self.pos < self.buf.len() && self.buf[self.pos] != b'\n' {
            self.pos += 1;
        }
        let end = self.pos;
        if self.pos < self.buf.len() {
            self.pos += 1;
        }
        std::str::from_utf8(&self.buf[start..end])
            .map(|s| s.trim_end_matches('\r'))
            .map_err(|_| self.error("invalid text: not UTF-8"))
    }

    fn expect(&mut self, expected: &str) -> Result<()> {
        let token = self.token(expected)?;
        if token == expected {
            Ok(())
        } else {
            Err(self.error(format!("expected {expected}, found {token:?}")))
        }
    }

    /// Skip a section whose contents are not read, up to and including its
    /// end marker.
    fn skip_section(&mut self, name: &str) -> Result<()> {
        let end = format!("$End{name}");
        match self.buf[self.pos..]
            .windows(end.len())
            .position(|w| w == end.as_bytes())
        {
            Some(offset) => {
                self.pos += offset + end.len();
                Ok(())
            }
            None => Err(self.error(format!("missing {end}"))),
        }
    }

    fn bytes<const N: usize>(&mut self, what: &str) -> Result<[u8; N]> {
        let bytes = self
            .buf
            .get(self.pos..self.pos + N)
            .ok_or_else(|| self.error(format!("unexpected end of file, expected {what}")))?;
        self.pos += N;
        let mut out = [0; N];
        out.copy_from_slice(bytes);
        if self.big_endian {
            out.reverse();
        }
        Ok(out)
    }

    /// An `int`: a binary 4-byte integer or an ASCII token.
    fn int(&mut self, what: &str) -> Result<i32> {
        if self.binary {
            Ok(i32::from_le_bytes(self.bytes(what)?))
        } else {
            self.parse(what)
        }
    }

    /// A `size_t`: a binary integer of the header's data-size or an ASCII token.
    fn size(&mut self, what: &str) -> Result<u64> {
        if !self.binary {
            return self.parse(what);
        }
        if self.size_bytes == 4 {
            Ok(u64::from(u32::from_le_bytes(self.bytes(what)?)))
        } else {
            Ok(u64::from_le_bytes(self.bytes(what)?))
        }
    }

    /// A count, bounded by what the rest of the file can hold (each item
    /// takes at least `min_bytes`), so a corrupt count cannot trigger a huge
    /// allocation.
    fn count(&mut self, what: &str, min_bytes: usize) -> Result<usize> {
        let n = self.size(what)?;
        let room = (self.buf.len() - self.pos) / min_bytes.max(1) + 1;
        match usize::try_from(n) {
            Ok(n) if n <= room => Ok(n),
            _ => Err(self.error(format!("{what} {n} exceeds the file size"))),
        }
    }

    fn double(&mut self, what: &str) -> Result<f64> {
        if self.binary {
            Ok(f64::from_le_bytes(self.bytes(what)?))
        } else {
            self.parse(what)
        }
    }
}

// =============================================================================
// Sections
// =============================================================================

fn parse_msh(buf: &[u8]) -> Result<RawMesh> {
    let mut r = Reader::new(buf);
    let mut raw = RawMesh::default();
    let mut version = None;

    while !r.at_end() {
        let header = r.line()?.trim();
        let Some(name) = header.strip_prefix('$') else {
            return Err(r.error(format!("expected a section header, found {header:?}")));
        };
        let name = name.to_string();
        let needs_format = || {
            version.ok_or_else(|| GmshError::MissingSection(format!("MeshFormat (before ${name})")))
        };
        match name.as_str() {
            "MeshFormat" => version = Some(parse_mesh_format(&mut r)?),
            "PhysicalNames" => parse_physical_names(&mut r, &mut raw)?,
            "Entities" => match needs_format()? {
                Version::V41 => parse_entities(&mut r, &mut raw)?,
                Version::V2 => return Err(r.error("$Entities in an MSH 2 file")),
            },
            "Nodes" => match needs_format()? {
                Version::V41 => parse_nodes_v41(&mut r, &mut raw)?,
                Version::V2 => parse_nodes_v2(&mut r, &mut raw)?,
            },
            "Elements" => match needs_format()? {
                Version::V41 => parse_elements_v41(&mut r, &mut raw)?,
                Version::V2 => parse_elements_v2(&mut r, &mut raw)?,
            },
            "Periodic" => {
                return Err(GmshError::Unsupported(
                    "periodic meshes ($Periodic)".to_string(),
                ));
            }
            "PartitionedEntities" => {
                return Err(GmshError::Unsupported(
                    "partitioned meshes ($PartitionedEntities): save the mesh unpartitioned"
                        .to_string(),
                ));
            }
            _ => {
                r.skip_section(&name)?;
                continue;
            }
        }
        r.expect(&format!("$End{name}"))?;
    }

    if version.is_none() {
        return Err(GmshError::MissingSection("MeshFormat".to_string()));
    }
    Ok(raw)
}

/// `$MeshFormat`: version, file-type (0 ASCII, 1 binary), data-size, and in
/// binary files the integer 1 for the byte order.
fn parse_mesh_format(r: &mut Reader) -> Result<Version> {
    let version = r.token("format version")?;
    let file_type: i32 = r.parse("file type")?;
    let data_size: usize = r.parse("data size")?;
    let version = match version {
        "4.1" => Version::V41,
        v if v.starts_with("2.") => Version::V2,
        v if v.starts_with('4') => {
            return Err(GmshError::UnsupportedVersion(format!(
                "{v}: save as MSH 4.1 (Mesh.MshFileVersion = 4.1) or 2.2"
            )));
        }
        v => return Err(GmshError::UnsupportedVersion(v.to_string())),
    };
    match (file_type, version) {
        (0, _) => {}
        (1, Version::V41) => {
            if data_size != 4 && data_size != 8 {
                return Err(r.error(format!("unsupported data size {data_size}")));
            }
            r.line()?; // the rest of the header line
            r.binary = true;
            r.size_bytes = data_size;
            r.big_endian = match r.bytes::<4>("byte-order mark")? {
                [1, 0, 0, 0] => false,
                [0, 0, 0, 1] => true,
                other => return Err(r.error(format!("invalid byte-order mark {other:?}"))),
            };
        }
        (1, Version::V2) => {
            return Err(GmshError::UnsupportedVersion(
                "binary MSH 2: save as ASCII or as MSH 4.1".to_string(),
            ));
        }
        (t, _) => return Err(r.error(format!("invalid file type {t}"))),
    }
    Ok(version)
}

/// `$PhysicalNames` (always ASCII): `dimension tag "name"` per line.
fn parse_physical_names(r: &mut Reader, raw: &mut RawMesh) -> Result<()> {
    let binary = std::mem::replace(&mut r.binary, false);
    let n: usize = r.parse("physical name count")?;
    for _ in 0..n {
        let dim: i32 = r.parse("physical group dimension")?;
        let tag: i32 = r.parse("physical group tag")?;
        let name = r.line()?.trim();
        let name = name
            .strip_prefix('"')
            .and_then(|s| s.strip_suffix('"'))
            .unwrap_or(name);
        raw.physical_names.insert((dim, tag), name.to_string());
    }
    r.binary = binary;
    Ok(())
}

/// Physical tags of an entity: `numPhysicalTags(size_t) physicalTag(int)...`
fn parse_entity_groups(r: &mut Reader) -> Result<Vec<i32>> {
    let n = r.count("physical tag count", 1)?;
    (0..n).map(|_| r.int("physical tag")).collect()
}

/// `$Entities` (MSH 4.1): keeps the physical groups of the curves.
fn parse_entities(r: &mut Reader, raw: &mut RawMesh) -> Result<()> {
    let mut counts = [0; 4];
    for count in &mut counts {
        *count = r.count("entity count", 1)?;
    }
    for (dim, &n) in counts.iter().enumerate() {
        for _ in 0..n {
            let tag = r.int("entity tag")?;
            // A point has its coordinates, the others their bounding box
            let n_coords = if dim == 0 { 3 } else { 6 };
            for _ in 0..n_coords {
                r.double("entity coordinate")?;
            }
            let groups = parse_entity_groups(r)?;
            if dim > 0 {
                let n_bounding = r.count("bounding entity count", 1)?;
                for _ in 0..n_bounding {
                    r.int("bounding entity tag")?;
                }
            }
            if dim == 1 {
                raw.curve_groups.insert(tag, groups);
            }
        }
    }
    Ok(())
}

fn push_node(r: &Reader, raw: &mut RawMesh, tag: u64, x: f64, y: f64) -> Result<()> {
    if !(x.is_finite() && y.is_finite()) {
        return Err(r.error(format!("node {tag} has non-finite coordinates ({x}, {y})")));
    }
    raw.nodes.push((tag, [x, y]));
    Ok(())
}

/// `$Nodes` (MSH 4.1): entity blocks of tags, then coordinates (plus the
/// parametric coordinates of parametric blocks).
fn parse_nodes_v41(r: &mut Reader, raw: &mut RawMesh) -> Result<()> {
    let n_blocks = r.count("node block count", 4)?;
    let n_nodes = r.count("node count", 4)?;
    r.size("minimum node tag")?;
    r.size("maximum node tag")?;
    raw.nodes.reserve(n_nodes);
    let start = raw.nodes.len();

    let mut tags = Vec::new();
    for _ in 0..n_blocks {
        let dim = r.int("entity dimension")?;
        r.int("entity tag")?;
        let parametric = r.int("parametric flag")?;
        let n = r.count("nodes in block", 4)?;
        let n_param = match (parametric, dim) {
            (0, _) => 0,
            (1, 0..=3) => dim as usize,
            _ => return Err(r.error(format!("invalid node block ({dim}, {parametric})"))),
        };
        tags.clear();
        for _ in 0..n {
            tags.push(r.size("node tag")?);
        }
        for &tag in &tags {
            let x = r.double("node x")?;
            let y = r.double("node y")?;
            for _ in 0..1 + n_param {
                r.double("node coordinate")?;
            }
            push_node(r, raw, tag, x, y)?;
        }
    }

    let read = raw.nodes.len() - start;
    if read != n_nodes {
        return Err(r.error(format!("$Nodes declares {n_nodes} nodes, found {read}")));
    }
    Ok(())
}

/// `$Nodes` (MSH 2): `tag x y z` per line.
fn parse_nodes_v2(r: &mut Reader, raw: &mut RawMesh) -> Result<()> {
    let n = r.count("node count", 4)?;
    raw.nodes.reserve(n);
    for _ in 0..n {
        let tag = r.parse("node tag")?;
        let x = r.parse("node x")?;
        let y = r.parse("node y")?;
        r.parse::<f64>("node z")?;
        push_node(r, raw, tag, x, y)?;
    }
    Ok(())
}

/// `$Elements` (MSH 4.1): entity blocks of one element type each.
fn parse_elements_v41(r: &mut Reader, raw: &mut RawMesh) -> Result<()> {
    let n_blocks = r.count("element block count", 4)?;
    let n_elements = r.count("element count", 4)?;
    r.size("minimum element tag")?;
    r.size("maximum element tag")?;

    let mut read = 0;
    let mut nodes = Vec::new();
    for _ in 0..n_blocks {
        r.int("entity dimension")?;
        let entity = r.int("entity tag")?;
        let element_type = r.int("element type")?;
        let n = r.count("elements in block", 4)?;
        let Some((n_nodes, _)) = element_type_info(element_type) else {
            return Err(GmshError::UnsupportedElements(vec![(element_type, n)]));
        };
        for _ in 0..n {
            let tag = r.size("element tag")?;
            nodes.clear();
            for _ in 0..n_nodes {
                nodes.push(r.size("element node tag")?);
            }
            raw.add_element(element_type, tag, &nodes, LineGroups::Curve(entity));
        }
        read += n;
    }

    if read != n_elements {
        return Err(r.error(format!(
            "$Elements declares {n_elements} elements, found {read}"
        )));
    }
    Ok(())
}

/// `$Elements` (MSH 2): `tag type n_tags tags... nodes...` per line; the
/// first tag is the physical group.
fn parse_elements_v2(r: &mut Reader, raw: &mut RawMesh) -> Result<()> {
    let n = r.count("element count", 4)?;
    let mut nodes = Vec::new();
    for _ in 0..n {
        let tag = r.parse("element tag")?;
        let element_type: i32 = r.parse("element type")?;
        let n_tags: usize = r.parse("element tag count")?;
        let mut physical = 0;
        for i in 0..n_tags {
            let t: i32 = r.parse("element tag")?;
            if i == 0 {
                physical = t;
            }
        }
        let Some((n_nodes, _)) = element_type_info(element_type) else {
            return Err(GmshError::UnsupportedElements(vec![(element_type, 1)]));
        };
        nodes.clear();
        for _ in 0..n_nodes {
            nodes.push(r.parse("element node tag")?);
        }
        raw.add_element(element_type, tag, &nodes, LineGroups::Physical(physical));
    }
    Ok(())
}

// =============================================================================
// Boundary tags
// =============================================================================

/// The boundary tag of a physical group: by name if it has one, else by
/// number (module docs).
fn group_boundary_tag(tag: i32, name: Option<&str>) -> Result<BoundaryTag> {
    let number = u32::try_from(tag)
        .map_err(|_| GmshError::InvalidMesh(format!("negative physical group tag {tag}")))?;
    let Some(name) = name else {
        return Ok(physical_tag_to_boundary_tag(number));
    };
    let name = name.to_ascii_lowercase().replace([' ', '-'], "_");
    Ok(match name.as_str() {
        "wall" | "walls" | "land" | "coast" | "coastline" | "shore" | "shoreline" | "closed" => {
            BoundaryTag::Wall
        }
        "open" | "ocean" | "sea" | "open_boundary" => BoundaryTag::Open,
        "tidal" | "tide" | "tides" | "tidal_forcing" => BoundaryTag::TidalForcing,
        "river" | "rivers" => BoundaryTag::River,
        _ => BoundaryTag::Custom(number),
    })
}

/// Convert an unnamed Gmsh physical tag to a BoundaryTag.
fn physical_tag_to_boundary_tag(tag: u32) -> BoundaryTag {
    match tag {
        1 => BoundaryTag::Wall,
        2 => BoundaryTag::Open,
        3 => BoundaryTag::TidalForcing,
        4 => BoundaryTag::River,
        5 => BoundaryTag::Dirichlet,
        6 => BoundaryTag::Neumann,
        _ => BoundaryTag::Custom(tag),
    }
}

/// Convert a BoundaryTag to a Gmsh physical tag.
fn boundary_tag_to_physical(tag: &BoundaryTag) -> u32 {
    match tag {
        BoundaryTag::Wall => 1,
        BoundaryTag::Open => 2,
        BoundaryTag::TidalForcing => 3,
        BoundaryTag::River => 4,
        BoundaryTag::Periodic(_) => 0, // Periodic not supported in Gmsh export
        BoundaryTag::Dirichlet => 5,
        BoundaryTag::Neumann => 6,
        BoundaryTag::Custom(t) => *t,
    }
}

/// The boundary tag of a line element, `None` without a physical group.
fn line_boundary_tag(raw: &RawMesh, groups: LineGroups) -> Result<Option<BoundaryTag>> {
    let physical: &[i32] = match groups {
        LineGroups::Curve(entity) => raw.curve_groups.get(&entity).map_or(&[], Vec::as_slice),
        LineGroups::Physical(0) => &[],
        LineGroups::Physical(ref p) => std::slice::from_ref(p),
    };
    let mut tag = None;
    for &p in physical {
        let name = raw.physical_names.get(&(1, p)).map(String::as_str);
        let t = group_boundary_tag(p, name)?;
        match tag {
            Some(existing) if existing != t => {
                return Err(GmshError::InvalidMesh(format!(
                    "a curve is in physical groups that map to different boundary tags \
                     ({existing:?} and {t:?})"
                )));
            }
            _ => tag = Some(t),
        }
    }
    Ok(tag)
}

// =============================================================================
// Mesh construction
// =============================================================================

/// z-component of (b - a) × (c - b): positive at a convex, counter-clockwise
/// corner b.
fn corner_cross(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> (f64, f64) {
    let (e0, e1) = ([b[0] - a[0], b[1] - a[1]], [c[0] - b[0], c[1] - b[1]]);
    let cross = e0[0] * e1[1] - e0[1] * e1[0];
    let scale = e0[0].hypot(e0[1]) * e1[0].hypot(e1[1]);
    (cross, scale)
}

/// Build a `Mesh2D` from the file contents: renumber nodes, orient and check
/// the quadrilaterals, build the edges and apply the boundary tags.
fn build_mesh(raw: RawMesh) -> Result<Mesh2D> {
    if !raw.unsupported.is_empty() {
        return Err(GmshError::UnsupportedElements(
            raw.unsupported.iter().map(|(&t, &n)| (t, n)).collect(),
        ));
    }
    if raw.nodes.is_empty() {
        return Err(GmshError::MissingSection("Nodes".to_string()));
    }
    if raw.quads.is_empty() {
        return Err(GmshError::InvalidMesh(
            "no quadrilateral elements. Once a model has physical groups, Gmsh saves only \
             the elements in them: add a physical surface for the domain (or set \
             Mesh.SaveAll = 1)"
                .to_string(),
        ));
    }

    // Node tag → index in raw.nodes
    let mut node_index: HashMap<u64, usize> = HashMap::with_capacity(raw.nodes.len());
    for (i, &(tag, _)) in raw.nodes.iter().enumerate() {
        if node_index.insert(tag, i).is_some() {
            return Err(GmshError::InvalidMesh(format!(
                "node {tag} is defined twice"
            )));
        }
    }
    let lookup = |element: u64, tag: u64| {
        node_index.get(&tag).copied().ok_or_else(|| {
            GmshError::InvalidMesh(format!("element {element} uses undefined node {tag}"))
        })
    };

    // Vertices: the nodes the quadrilaterals use, in file order
    // (usize::MAX for the others)
    let mut used = vec![false; raw.nodes.len()];
    for &(element, tags) in &raw.quads {
        for tag in tags {
            used[lookup(element, tag)?] = true;
        }
    }
    let mut vertices = Vec::new();
    let vertex_of_node: Vec<usize> = raw
        .nodes
        .iter()
        .zip(&used)
        .map(|(&(_, xy), &used)| {
            if !used {
                return usize::MAX;
            }
            vertices.push(xy);
            vertices.len() - 1
        })
        .collect();

    // Elements, counter-clockwise, with a positive Jacobian
    let mut elements = Vec::with_capacity(raw.quads.len());
    for &(element, tags) in &raw.quads {
        let mut quad = [0; 4];
        for (v, tag) in quad.iter_mut().zip(tags) {
            *v = vertex_of_node[lookup(element, tag)?];
        }
        let p = quad.map(|v| vertices[v]);
        let twice_area: f64 = (0..4)
            .map(|i| p[i][0] * p[(i + 1) % 4][1] - p[(i + 1) % 4][0] * p[i][1])
            .sum();
        if twice_area < 0.0 {
            quad = [quad[0], quad[3], quad[2], quad[1]];
        }
        let p = quad.map(|v| vertices[v]);
        // det J of the bilinear map is affine in r and in s, so positive at
        // the corners means positive everywhere
        for i in 0..4 {
            let (cross, scale) = corner_cross(p[(i + 3) % 4], p[i], p[(i + 1) % 4]);
            if cross.is_nan() || cross <= 1e-12 * scale {
                return Err(GmshError::InvalidMesh(format!(
                    "element {element} is degenerate or non-convex at node {} ({}, {}): \
                     its bilinear map has a non-positive Jacobian",
                    tags[if twice_area < 0.0 { (4 - i) % 4 } else { i }],
                    p[i][0],
                    p[i][1]
                )));
            }
        }
        elements.push(quad);
    }
    let element_tag = |k: usize| raw.quads[k].0;

    // Edges, checked: at most two faces per edge, traversed in opposite
    // directions. Boundary edges are walls unless a line element tags them.
    let coordinates = vertices.clone();
    let segment = |(a, b): (usize, usize)| {
        let ([x0, y0], [x1, y1]) = (coordinates[a], coordinates[b]);
        format!("({x0}, {y0})–({x1}, {y1})")
    };
    let mut mesh = Mesh2D::from_quads(vertices, elements, &[], |_| BoundaryTag::Wall).map_err(
        |error| {
            GmshError::InvalidMesh(match error {
                QuadMeshError::NonManifoldEdge { vertices, faces } => format!(
                    "the edge {} is shared by more than two elements (elements {}, {} and {})",
                    segment(vertices),
                    element_tag(faces[0].element),
                    element_tag(faces[1].element),
                    element_tag(faces[2].element)
                ),
                QuadMeshError::Overlap { vertices, faces } => format!(
                    "elements {} and {} overlap: they traverse their shared edge {} in the same direction",
                    element_tag(faces[0].element),
                    element_tag(faces[1].element),
                    segment(vertices)
                ),
                other => other.to_string(),
            })
        },
    )?;

    // Boundary tags from the line elements
    let edge_index: HashMap<(usize, usize), usize> = mesh
        .edges
        .iter()
        .enumerate()
        .map(|(e, edge)| (edge.vertices, e))
        .collect();
    let mut line_tags: Vec<Option<BoundaryTag>> = vec![None; mesh.n_edges];
    for &(element, tags, groups) in &raw.lines {
        let [a, b] = tags.map(|tag| {
            node_index
                .get(&tag)
                .map(|&i| vertex_of_node[i])
                .filter(|&v| v != usize::MAX)
        });
        let edge = match (a, b) {
            (Some(a), Some(b)) => edge_index.get(&(a.min(b), a.max(b))).copied(),
            _ => None,
        };
        let Some(e) = edge else {
            return Err(GmshError::InvalidMesh(format!(
                "line element {element} (nodes {} and {}) is not an edge of the quadrilateral mesh",
                tags[0], tags[1]
            )));
        };
        if mesh.edges[e].is_interior() {
            continue;
        }
        let Some(tag) = line_boundary_tag(&raw, groups)? else {
            continue;
        };
        match line_tags[e] {
            Some(existing) if existing != tag => {
                return Err(GmshError::InvalidMesh(format!(
                    "the boundary edge {} has two boundary tags, {existing:?} and {tag:?}",
                    segment(mesh.edges[e].vertices)
                )));
            }
            _ => line_tags[e] = Some(tag),
        }
    }
    for (edge, tag) in mesh.edges.iter_mut().zip(line_tags) {
        if tag.is_some() {
            edge.boundary_tag = tag;
        }
    }
    Ok(mesh)
}

// =============================================================================
// Writer
// =============================================================================

/// Write a Mesh2D to Gmsh MSH format 2.2.
///
/// Boundary edges are written as lines in the physical group of their tag
/// (1 wall, 2 open, 3 tidal forcing, 4 river, 5 Dirichlet, 6 Neumann,
/// `Custom(n)` → n), so the tags survive a round trip.
///
/// # Arguments
/// * `mesh` - The mesh to write
/// * `path` - Output file path
pub fn write_gmsh_mesh(mesh: &Mesh2D, path: &Path) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // Write header
    writeln!(writer, "$MeshFormat")?;
    writeln!(writer, "2.2 0 8")?;
    writeln!(writer, "$EndMeshFormat")?;

    // Write nodes
    writeln!(writer, "$Nodes")?;
    writeln!(writer, "{}", mesh.vertices.len())?;
    for (i, &[x, y]) in mesh.vertices.iter().enumerate() {
        writeln!(writer, "{} {} {} 0", i + 1, x, y)?;
    }
    writeln!(writer, "$EndNodes")?;

    // Count boundary edges
    let boundary_edges: Vec<_> = mesh.edges.iter().filter(|e| e.is_boundary()).collect();
    let n_total = boundary_edges.len() + mesh.n_elements;

    writeln!(writer, "$Elements")?;
    writeln!(writer, "{}", n_total)?;

    // Write boundary edges first
    let mut elem_id = 1;
    for edge in &boundary_edges {
        let (n0, n1) = edge.vertices;
        let physical_tag = edge
            .boundary_tag
            .as_ref()
            .map(boundary_tag_to_physical)
            .unwrap_or(1);
        // Format: elem_id type n_tags physical_tag geometrical_tag node1 node2
        writeln!(
            writer,
            "{} 1 2 {} {} {} {}",
            elem_id,
            physical_tag,
            physical_tag,
            n0 + 1,
            n1 + 1
        )?;
        elem_id += 1;
    }

    // Write quadrilaterals
    for elem in &mesh.elements {
        // Format: elem_id type n_tags physical_tag geometrical_tag node1 node2 node3 node4
        writeln!(
            writer,
            "{} 3 2 0 0 {} {} {} {}",
            elem_id,
            elem[0] + 1,
            elem[1] + 1,
            elem[2] + 1,
            elem[3] + 1
        )?;
        elem_id += 1;
    }

    writeln!(writer, "$EndElements")?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::core::ElementFace;
    use tempfile::NamedTempFile;

    fn parse(text: &str) -> Result<Mesh2D> {
        parse_gmsh_mesh(text.as_bytes())
    }

    fn count_tag(mesh: &Mesh2D, tag: BoundaryTag) -> usize {
        mesh.edges
            .iter()
            .filter(|e| e.boundary_tag == Some(tag))
            .count()
    }

    fn signed_area(mesh: &Mesh2D, k: usize) -> f64 {
        let p = mesh.elements[k].map(|v| mesh.vertices[v]);
        0.5 * (0..4)
            .map(|i| p[i][0] * p[(i + 1) % 4][1] - p[(i + 1) % 4][0] * p[i][1])
            .sum::<f64>()
    }

    /// Two unit squares side by side, MSH 4.1 ASCII, with sparse node tags,
    /// named groups and the right element clockwise.
    const TWO_QUADS_V41: &str = r#"$MeshFormat
4.1 0 8
$EndMeshFormat
$PhysicalNames
3
1 7 "Coast"
1 8 "open boundary"
2 9 "water"
$EndPhysicalNames
$Entities
0 3 1 0
1 0 0 0 2 0 0 1 7 0
2 2 0 0 2 1 0 1 8 0
3 0 1 0 2 1 0 0 0
1 0 0 0 2 1 0 1 9 0
$EndEntities
$Nodes
2 6 10 600
2 1 0 4
10
20
400
600
0 0 0
1 0 0
1 1 0
0 1 0
2 1 1 2
30
500
2 0 0 0.5 0.5
2 1 0 0.5 1.0
$EndNodes
$Elements
3 5 1 5
2 1 3 2
1 10 20 400 600
2 20 400 500 30
1 1 1 2
3 10 20
4 20 30
1 2 1 1
5 30 500
$EndElements
"#;

    #[test]
    fn test_read_v41_ascii() {
        let mesh = parse(TWO_QUADS_V41).unwrap();
        assert_eq!(mesh.n_vertices, 6);
        assert_eq!(mesh.n_elements, 2);
        assert_eq!(mesh.n_edges, 7);
        assert_eq!(mesh.n_boundary_edges, 6);
        // Vertices in file order, parametric coordinates skipped
        assert_eq!(mesh.vertices[4], [2.0, 0.0]);
        assert_eq!(mesh.vertices[5], [2.0, 1.0]);
        // The clockwise element is reordered counter-clockwise
        assert_eq!(mesh.elements[0], [0, 1, 2, 3]);
        assert_eq!(mesh.elements[1], [1, 4, 5, 2]);
        for k in 0..2 {
            assert!((signed_area(&mesh, k) - 1.0).abs() < 1e-14);
        }
        // Named groups: "Coast" → wall, "open boundary" → open; the rest
        // default to walls
        assert_eq!(count_tag(&mesh, BoundaryTag::Open), 1);
        assert_eq!(count_tag(&mesh, BoundaryTag::Wall), 5);
        let open = mesh
            .edges
            .iter()
            .find(|e| e.boundary_tag == Some(BoundaryTag::Open))
            .unwrap();
        assert_eq!(open.vertices, (4, 5));
        // Neighbours traverse the shared edge in opposite directions
        let interior = mesh.edges.iter().find(|e| e.is_interior()).unwrap();
        let (l, r) = (interior.left, interior.right.unwrap());
        let face = |f: ElementFace| {
            let quad = mesh.elements[f.element];
            (quad[f.face], quad[(f.face + 1) % 4])
        };
        let (a, b) = face(l);
        assert_eq!(face(r), (b, a));
    }

    /// The same mesh as binary MSH 4.1.
    fn two_quads_v41_binary(big_endian: bool) -> Vec<u8> {
        let mut b: Vec<u8> = b"$MeshFormat\n4.1 1 8\n".to_vec();
        let int = |b: &mut Vec<u8>, v: i32| {
            b.extend(if big_endian {
                v.to_be_bytes()
            } else {
                v.to_le_bytes()
            })
        };
        let size = |b: &mut Vec<u8>, v: u64| {
            b.extend(if big_endian {
                v.to_be_bytes()
            } else {
                v.to_le_bytes()
            })
        };
        let double = |b: &mut Vec<u8>, v: f64| {
            b.extend(if big_endian {
                v.to_be_bytes()
            } else {
                v.to_le_bytes()
            })
        };
        int(&mut b, 1);
        b.extend(b"\n$EndMeshFormat\n$PhysicalNames\n1\n1 2 \"tide\"\n$EndPhysicalNames\n");
        b.extend(b"$Entities\n");
        for n in [0, 1, 1, 0] {
            size(&mut b, n);
        }
        // Curve 5 in group 2, bounding points 1 and -2
        int(&mut b, 5);
        for v in [2.0, 0.0, 0.0, 2.0, 1.0, 0.0] {
            double(&mut b, v);
        }
        size(&mut b, 1);
        int(&mut b, 2);
        size(&mut b, 2);
        int(&mut b, 1);
        int(&mut b, -2);
        // Surface 1, no group, bounded by curve 5
        int(&mut b, 1);
        for v in [0.0, 0.0, 0.0, 2.0, 1.0, 0.0] {
            double(&mut b, v);
        }
        size(&mut b, 0);
        size(&mut b, 1);
        int(&mut b, 5);
        b.extend(b"\n$EndEntities\n$Nodes\n");
        for v in [1, 6, 3, 71] {
            size(&mut b, v);
        }
        for v in [2, 1, 0] {
            int(&mut b, v);
        }
        size(&mut b, 6);
        for tag in [3, 71, 5, 4, 50, 51] {
            size(&mut b, tag);
        }
        for [x, y] in [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [2.0, 0.0],
            [2.0, 1.0],
        ] {
            for v in [x, y, 0.0] {
                double(&mut b, v);
            }
        }
        b.extend(b"\n$EndNodes\n$Elements\n");
        for v in [2, 3, 1, 3] {
            size(&mut b, v);
        }
        for v in [2, 1, GMSH_QUAD] {
            int(&mut b, v);
        }
        size(&mut b, 2);
        for v in [1, 3, 71, 5, 4, 2, 71, 50, 51, 5] {
            size(&mut b, v);
        }
        for v in [1, 5, GMSH_LINE] {
            int(&mut b, v);
        }
        size(&mut b, 1);
        for v in [3, 50, 51] {
            size(&mut b, v);
        }
        b.extend(b"\n$EndElements\n");
        b
    }

    #[test]
    fn test_read_v41_binary_both_byte_orders() {
        for big_endian in [false, true] {
            let mesh = parse_gmsh_mesh(&two_quads_v41_binary(big_endian)).unwrap();
            assert_eq!(mesh.n_vertices, 6);
            assert_eq!(mesh.n_elements, 2);
            assert_eq!(mesh.n_boundary_edges, 6);
            assert_eq!(mesh.elements[1], [1, 4, 5, 2]);
            assert_eq!(count_tag(&mesh, BoundaryTag::TidalForcing), 1);
            assert_eq!(count_tag(&mesh, BoundaryTag::Wall), 5);
        }
    }

    /// Cutting the file anywhere is an error, never a panic.
    #[test]
    fn test_truncated_files_are_errors() {
        let binary = two_quads_v41_binary(false);
        for len in 0..binary.len() {
            // A cut in the final "$EndElements\n" after its "$EndElements"
            // still reads the whole mesh
            if len >= binary.len() - 1 {
                continue;
            }
            assert!(
                parse_gmsh_mesh(&binary[..len]).is_err(),
                "binary cut at {len}"
            );
        }
        let ascii = TWO_QUADS_V41.as_bytes();
        for len in 0..ascii.len() - 1 {
            // Cutting inside the last number still parses, as a different
            // number: skip the element lines
            let _ = parse_gmsh_mesh(&ascii[..len]);
        }
    }

    #[test]
    fn test_triangles_are_an_error() {
        let text = TWO_QUADS_V41.replace(
            "1 1 1 2\n3 10 20\n4 20 30\n",
            "2 1 2 2\n3 10 20 400\n4 10 400 600\n",
        );
        match parse(&text) {
            Err(GmshError::UnsupportedElements(counts)) => assert_eq!(counts, vec![(2, 2)]),
            other => panic!(
                "expected UnsupportedElements, got {:?}",
                other.map(|m| m.n_elements)
            ),
        }
        let message = parse(&text).err().unwrap().to_string();
        assert!(
            message.contains("2 × 3-node triangle (type 2)"),
            "{message}"
        );
        assert!(message.contains("Mesh.SubdivisionAlgorithm"), "{message}");
    }

    #[test]
    fn test_unsupported_versions() {
        for (header, expected) in [
            ("4 0 8", "MSH 4.1"),
            ("4.0 0 8", "MSH 4.1"),
            ("3.0 0 8", "3.0"),
        ] {
            let text = TWO_QUADS_V41.replace("4.1 0 8", header);
            match parse(&text) {
                Err(GmshError::UnsupportedVersion(v)) => assert!(v.contains(expected), "{v}"),
                other => panic!(
                    "{header}: expected UnsupportedVersion, got {:?}",
                    other.err()
                ),
            }
        }
        let binary_v2 = "$MeshFormat\n2.2 1 8\n";
        assert!(matches!(
            parse_gmsh_mesh(binary_v2.as_bytes()),
            Err(GmshError::UnsupportedVersion(_))
        ));
    }

    #[test]
    fn test_malformed_numbers_report_the_line() {
        let text = TWO_QUADS_V41.replace("1 1 0\n0 1 0", "1 1 0\n0 x 0");
        let message = parse(&text).err().unwrap().to_string();
        assert!(message.contains("invalid node y: \"x\""), "{message}");
        assert!(message.contains("line 27"), "{message}");
    }

    #[test]
    fn test_undefined_node_is_an_error() {
        let text = TWO_QUADS_V41.replace("1 10 20 400 600", "1 10 20 400 601");
        let message = parse(&text).err().unwrap().to_string();
        assert!(
            message.contains("element 1 uses undefined node 601"),
            "{message}"
        );
    }

    #[test]
    fn test_duplicate_node_is_an_error() {
        let text = TWO_QUADS_V41.replace("30\n500\n", "30\n10\n");
        assert!(matches!(parse(&text), Err(GmshError::InvalidMesh(_))));
    }

    #[test]
    fn test_non_convex_and_degenerate_elements_are_errors() {
        // A dart: node 400 moved inside the element
        let dart = TWO_QUADS_V41.replace("1 1 0\n0 1 0", "0.3 0.3 0\n0 1 0");
        let message = parse(&dart).err().unwrap().to_string();
        assert!(
            message.contains("element 1 is degenerate or non-convex at node 400"),
            "{message}"
        );
        // A repeated node
        let degenerate = TWO_QUADS_V41.replace("1 10 20 400 600", "1 10 20 20 600");
        assert!(matches!(parse(&degenerate), Err(GmshError::InvalidMesh(_))));
    }

    #[test]
    fn test_non_manifold_and_overlapping_elements_are_errors() {
        // A third element on the shared edge 20–400
        let text = TWO_QUADS_V41
            .replace("2 6 10 600", "3 8 10 800")
            .replace(
                "$EndNodes",
                "2 1 0 2\n700\n800\n1.5 0 0\n1.5 1 0\n$EndNodes",
            )
            .replace("3 5 1 5\n2 1 3 2\n", "3 6 1 6\n2 1 3 3\n")
            .replace("2 20 400 500 30\n", "2 20 400 500 30\n6 400 20 700 800\n");
        let message = parse(&text).err().unwrap().to_string();
        assert!(
            message.contains("shared by more than two elements"),
            "{message}"
        );
        // Two elements on the same side of an edge: the right element moved
        // onto the left one
        let text = TWO_QUADS_V41.replace(
            "2 0 0 0.5 0.5\n2 1 0 0.5 1.0",
            "0.5 0 0 0.5 0.5\n0.5 1 0 0.5 1.0",
        );
        let message = parse(&text).err().unwrap().to_string();
        assert!(message.contains("overlap"), "{message}");
    }

    #[test]
    fn test_conflicting_boundary_tags_are_an_error() {
        // The coast curve also in the "open boundary" group
        let text = TWO_QUADS_V41.replace("1 0 0 0 2 0 0 1 7 0", "1 0 0 0 2 0 0 2 7 8 0");
        let message = parse(&text).err().unwrap().to_string();
        assert!(message.contains("different boundary tags"), "{message}");
    }

    #[test]
    fn test_line_off_the_mesh_is_an_error() {
        let text = TWO_QUADS_V41.replace("3 10 20\n", "3 10 400\n");
        let message = parse(&text).err().unwrap().to_string();
        assert!(message.contains("line element 3"), "{message}");
    }

    #[test]
    fn test_no_quads_is_an_error_with_advice() {
        let text = TWO_QUADS_V41.replace(
            "3 5 1 5\n2 1 3 2\n1 10 20 400 600\n2 20 400 500 30\n",
            "2 3 3 5\n",
        );
        let message = parse(&text).err().unwrap().to_string();
        assert!(message.contains("physical surface"), "{message}");
    }

    #[test]
    fn test_skips_unknown_sections_and_rejects_periodic() {
        let text = format!(
            "{TWO_QUADS_V41}$Comments\nanything $Nodes\n$EndComments\n\
             $NodeData\n1\n\"eta\"\n$EndNodeData\n"
        );
        assert_eq!(parse(&text).unwrap().n_elements, 2);
        let text = format!("{TWO_QUADS_V41}$Periodic\n0\n$EndPeriodic\n");
        assert!(matches!(parse(&text), Err(GmshError::Unsupported(_))));
        let text = TWO_QUADS_V41.replace("$EndNodes", "$EndNode");
        assert!(matches!(parse(&text), Err(GmshError::ParseError(_))));
    }

    #[test]
    fn test_crlf_line_endings() {
        let mesh = parse(&TWO_QUADS_V41.replace('\n', "\r\n")).unwrap();
        assert_eq!(mesh.n_elements, 2);
        assert_eq!(count_tag(&mesh, BoundaryTag::Open), 1);
    }

    #[test]
    fn test_read_simple_mesh() {
        let mesh = parse(
            r#"$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
4
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 1.0 1.0 0.0
4 0.0 1.0 0.0
$EndNodes
$Elements
1
1 3 2 0 0 1 2 3 4
$EndElements"#,
        )
        .unwrap();
        assert_eq!(mesh.vertices.len(), 4);
        assert_eq!(mesh.n_elements, 1);
        assert_eq!(mesh.n_edges, 4);
        assert_eq!(mesh.n_boundary_edges, 4);
    }

    #[test]
    fn test_read_mesh_with_boundary() {
        let mesh = parse(
            r#"$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
4
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 1.0 1.0 0.0
4 0.0 1.0 0.0
$EndNodes
$Elements
5
1 1 2 1 1 1 2
2 1 2 2 2 2 3
3 1 2 1 1 3 4
4 1 2 2 2 4 1
5 3 2 0 0 1 2 3 4
$EndElements"#,
        )
        .unwrap();
        assert_eq!(mesh.vertices.len(), 4);
        assert_eq!(mesh.n_elements, 1);
        assert_eq!(mesh.n_boundary_edges, 4);
        assert_eq!(count_tag(&mesh, BoundaryTag::Wall), 2);
        assert_eq!(count_tag(&mesh, BoundaryTag::Open), 2);
    }

    /// MSH 2 with sparse node tags, a named group, a point element and an
    /// unused node.
    #[test]
    fn test_read_v2_sparse_tags_and_names() {
        let mesh = parse(
            r#"$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
1
1 17 "river"
$EndPhysicalNames
$Nodes
5
100 0.0 0.0 0.0
7 1.0 0.0 0.0
999 5.0 5.0 0.0
42 1.0 1.0 0.0
3 0.0 1.0 0.0
$EndNodes
$Elements
3
1 15 2 0 1 999
2 1 2 17 4 7 42
3 3 2 0 0 100 7 42 3
$EndElements"#,
        )
        .unwrap();
        assert_eq!(
            mesh.vertices,
            vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        );
        assert_eq!(count_tag(&mesh, BoundaryTag::River), 1);
        assert_eq!(count_tag(&mesh, BoundaryTag::Wall), 3);
    }

    #[test]
    fn test_roundtrip() {
        let mut mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 3, 2);
        let tags = [
            BoundaryTag::Open,
            BoundaryTag::TidalForcing,
            BoundaryTag::River,
            BoundaryTag::Dirichlet,
            BoundaryTag::Neumann,
            BoundaryTag::Custom(42),
        ];
        for (i, edge) in mesh
            .edges
            .iter_mut()
            .filter(|e| e.is_boundary())
            .enumerate()
        {
            edge.boundary_tag = Some(tags[i % tags.len()]);
        }

        let file = NamedTempFile::new().unwrap();
        write_gmsh_mesh(&mesh, file.path()).unwrap();
        let mesh2 = read_gmsh_mesh(file.path()).unwrap();

        assert_eq!(mesh.vertices, mesh2.vertices);
        assert_eq!(mesh.elements, mesh2.elements);
        assert_eq!(mesh.n_edges, mesh2.n_edges);
        let boundary_tags = |m: &Mesh2D| {
            let mut t: Vec<_> = m
                .edges
                .iter()
                .filter(|e| e.is_boundary())
                .map(|e| (e.vertices, e.boundary_tag))
                .collect();
            t.sort_by_key(|&(v, _)| v);
            t
        };
        assert_eq!(boundary_tags(&mesh), boundary_tags(&mesh2));
    }

    #[test]
    fn test_physical_tag_conversion() {
        assert_eq!(physical_tag_to_boundary_tag(1), BoundaryTag::Wall);
        assert_eq!(physical_tag_to_boundary_tag(2), BoundaryTag::Open);
        assert_eq!(physical_tag_to_boundary_tag(3), BoundaryTag::TidalForcing);
        assert_eq!(physical_tag_to_boundary_tag(4), BoundaryTag::River);
        assert_eq!(physical_tag_to_boundary_tag(99), BoundaryTag::Custom(99));

        assert_eq!(boundary_tag_to_physical(&BoundaryTag::Wall), 1);
        assert_eq!(boundary_tag_to_physical(&BoundaryTag::Open), 2);

        assert_eq!(
            group_boundary_tag(2, Some("Land")).unwrap(),
            BoundaryTag::Wall
        );
        assert_eq!(
            group_boundary_tag(1, Some("tidal-forcing")).unwrap(),
            BoundaryTag::TidalForcing
        );
        assert_eq!(
            group_boundary_tag(1, Some("north")).unwrap(),
            BoundaryTag::Custom(1)
        );
        assert!(group_boundary_tag(-1, None).is_err());
    }

    #[test]
    fn test_error_missing_nodes() {
        let result = parse(
            r#"$MeshFormat
2.2 0 8
$EndMeshFormat
$Elements
1
1 3 2 0 0 1 2 3 4
$EndElements"#,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_two_element_mesh() {
        let mesh = parse(
            r#"$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
6
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 2.0 0.0 0.0
4 0.0 1.0 0.0
5 1.0 1.0 0.0
6 2.0 1.0 0.0
$EndNodes
$Elements
2
1 3 2 0 0 1 2 5 4
2 3 2 0 0 2 3 6 5
$EndElements"#,
        )
        .unwrap();
        assert_eq!(mesh.n_elements, 2);
        assert_eq!(mesh.vertices.len(), 6);
        // 4 + 2 boundary edges on the long sides, one interior edge
        assert_eq!(mesh.n_edges, 7);
        assert_eq!(mesh.n_boundary_edges, 6);
        let interior_edges: Vec<_> = mesh.edges.iter().filter(|e| e.is_interior()).collect();
        assert_eq!(interior_edges.len(), 1);
    }
}

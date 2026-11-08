//! VTK output for DG solutions.
//!
//! Provides VTU (XML UnstructuredGrid) output for visualization in ParaView
//! and other VTK-compatible tools.
//!
//! # High-Order Visualization
//!
//! DG elements with polynomial order p have (p+1)² nodes per element.
//! To preserve solution detail, we decompose each element into p² sub-quads,
//! where each GLL node becomes a VTK point.
//!
//! # Example
//!
//! ```ignore
//! use dgcom::io::write_vtk_swe;
//!
//! write_vtk_swe("output.vtu", &mesh, &ops, &swe_solution, Some(&bathymetry), time, h_min)?;
//! ```

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use thiserror::Error;

use crate::mesh::{Bathymetry2D, Mesh2D};
use crate::operators::DGOperators2D;
use crate::solver::{SWESolution2D, SWEState2D};
use crate::types::{Depth, ElementIndex};

/// Error type for VTK operations.
#[derive(Debug, Error)]
pub enum VtkError {
    /// I/O error during file operations.
    #[error("VTK I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Invalid mesh configuration.
    #[error("Invalid mesh: {0}")]
    InvalidMesh(String),
}

/// Internal mesh representation for VTK output.
struct VtkMesh {
    /// Physical coordinates of all points [x, y].
    points: Vec<[f64; 2]>,
    /// Sub-cell connectivity (4 point indices per quad).
    cells: Vec<[usize; 4]>,
    /// Original DG element index for each sub-cell.
    element_ids: Vec<usize>,
}

/// Build VTK mesh from DG mesh with sub-cell decomposition.
fn build_vtk_mesh(mesh: &Mesh2D, ops: &DGOperators2D) -> VtkMesh {
    let n_elements = mesh.n_elements;
    let n_nodes = ops.n_nodes;
    let n_1d = ops.n_1d;
    let n_subcells_per_elem = (n_1d - 1) * (n_1d - 1);

    let mut points = Vec::with_capacity(n_elements * n_nodes);
    let mut cells = Vec::with_capacity(n_elements * n_subcells_per_elem);
    let mut element_ids = Vec::with_capacity(n_elements * n_subcells_per_elem);

    for k in ElementIndex::iter(n_elements) {
        let ki = k.as_usize();
        let base_point = ki * n_nodes;

        // Compute physical coordinates for all nodes in this element
        for i in 0..n_nodes {
            let r = ops.nodes_r[i];
            let s = ops.nodes_s[i];
            let [x, y] = mesh.reference_to_physical(k, r, s);
            points.push([x, y]);
        }

        // Create sub-cells from adjacent nodes
        // Node layout for n_1d = 4:
        // 12--13--14--15
        // |   |   |   |
        // 8---9--10--11
        // |   |   |   |
        // 4---5---6---7
        // |   |   |   |
        // 0---1---2---3
        for j in 0..(n_1d - 1) {
            for i in 0..(n_1d - 1) {
                // Sub-cell (i, j) corners (counter-clockwise for VTK)
                let v0 = base_point + j * n_1d + i;
                let v1 = base_point + j * n_1d + i + 1;
                let v2 = base_point + (j + 1) * n_1d + i + 1;
                let v3 = base_point + (j + 1) * n_1d + i;
                cells.push([v0, v1, v2, v3]);
                element_ids.push(ki);
            }
        }
    }

    VtkMesh {
        points,
        cells,
        element_ids,
    }
}

/// VTK XML writer helper.
struct VtkWriter<W: Write> {
    writer: BufWriter<W>,
    indent: usize,
}

impl<W: Write> VtkWriter<W> {
    fn new(writer: W) -> Self {
        Self {
            writer: BufWriter::new(writer),
            indent: 0,
        }
    }

    fn write_indent(&mut self) -> std::io::Result<()> {
        for _ in 0..self.indent {
            write!(self.writer, "  ")?;
        }
        Ok(())
    }

    fn write_header(&mut self) -> std::io::Result<()> {
        writeln!(self.writer, "<?xml version=\"1.0\"?>")?;
        writeln!(
            self.writer,
            "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">"
        )?;
        self.indent += 1;
        Ok(())
    }

    fn write_footer(&mut self) -> std::io::Result<()> {
        self.indent -= 1;
        writeln!(self.writer, "</VTKFile>")?;
        self.writer.flush()?;
        Ok(())
    }

    fn start_element(&mut self, name: &str, attrs: &[(&str, &str)]) -> std::io::Result<()> {
        self.write_indent()?;
        write!(self.writer, "<{}", name)?;
        for (key, value) in attrs {
            write!(self.writer, " {}=\"{}\"", key, value)?;
        }
        writeln!(self.writer, ">")?;
        self.indent += 1;
        Ok(())
    }

    fn end_element(&mut self, name: &str) -> std::io::Result<()> {
        self.indent -= 1;
        self.write_indent()?;
        writeln!(self.writer, "</{}>", name)?;
        Ok(())
    }

    fn write_data_array_f64(
        &mut self,
        name: &str,
        data: &[f64],
        components: usize,
    ) -> std::io::Result<()> {
        self.write_indent()?;
        if components > 1 {
            writeln!(
                self.writer,
                "<DataArray type=\"Float64\" Name=\"{}\" NumberOfComponents=\"{}\" format=\"ascii\">",
                name, components
            )?;
        } else {
            writeln!(
                self.writer,
                "<DataArray type=\"Float64\" Name=\"{}\" format=\"ascii\">",
                name
            )?;
        }

        self.indent += 1;
        self.write_indent()?;
        for (i, &v) in data.iter().enumerate() {
            write!(self.writer, "{:.10e}", v)?;
            if i < data.len() - 1 {
                write!(self.writer, " ")?;
            }
            // Line break every 6 values for readability
            if (i + 1) % 6 == 0 && i < data.len() - 1 {
                writeln!(self.writer)?;
                self.write_indent()?;
            }
        }
        writeln!(self.writer)?;
        self.indent -= 1;

        self.write_indent()?;
        writeln!(self.writer, "</DataArray>")?;
        Ok(())
    }

    fn write_data_array_i32(&mut self, name: &str, data: &[i32]) -> std::io::Result<()> {
        self.write_indent()?;
        writeln!(
            self.writer,
            "<DataArray type=\"Int32\" Name=\"{}\" format=\"ascii\">",
            name
        )?;

        self.indent += 1;
        self.write_indent()?;
        for (i, &v) in data.iter().enumerate() {
            write!(self.writer, "{}", v)?;
            if i < data.len() - 1 {
                write!(self.writer, " ")?;
            }
            if (i + 1) % 20 == 0 && i < data.len() - 1 {
                writeln!(self.writer)?;
                self.write_indent()?;
            }
        }
        writeln!(self.writer)?;
        self.indent -= 1;

        self.write_indent()?;
        writeln!(self.writer, "</DataArray>")?;
        Ok(())
    }

    fn write_data_array_u8(&mut self, name: &str, data: &[u8]) -> std::io::Result<()> {
        self.write_indent()?;
        writeln!(
            self.writer,
            "<DataArray type=\"UInt8\" Name=\"{}\" format=\"ascii\">",
            name
        )?;

        self.indent += 1;
        self.write_indent()?;
        for (i, &v) in data.iter().enumerate() {
            write!(self.writer, "{}", v)?;
            if i < data.len() - 1 {
                write!(self.writer, " ")?;
            }
            if (i + 1) % 20 == 0 && i < data.len() - 1 {
                writeln!(self.writer)?;
                self.write_indent()?;
            }
        }
        writeln!(self.writer)?;
        self.indent -= 1;

        self.write_indent()?;
        writeln!(self.writer, "</DataArray>")?;
        Ok(())
    }

    fn write_points(&mut self, points: &[[f64; 2]]) -> std::io::Result<()> {
        self.start_element("Points", &[])?;

        self.write_indent()?;
        writeln!(
            self.writer,
            "<DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">"
        )?;

        self.indent += 1;
        self.write_indent()?;
        for (i, &[x, y]) in points.iter().enumerate() {
            write!(self.writer, "{:.10e} {:.10e} 0.0", x, y)?;
            if i < points.len() - 1 {
                write!(self.writer, " ")?;
            }
            if (i + 1) % 2 == 0 && i < points.len() - 1 {
                writeln!(self.writer)?;
                self.write_indent()?;
            }
        }
        writeln!(self.writer)?;
        self.indent -= 1;

        self.write_indent()?;
        writeln!(self.writer, "</DataArray>")?;

        self.end_element("Points")?;
        Ok(())
    }

    fn write_cells(&mut self, cells: &[[usize; 4]]) -> std::io::Result<()> {
        self.start_element("Cells", &[])?;

        // Connectivity
        let connectivity: Vec<i32> = cells
            .iter()
            .flat_map(|c| c.iter().map(|&v| v as i32))
            .collect();
        self.write_data_array_i32("connectivity", &connectivity)?;

        // Offsets (cumulative vertex count)
        let offsets: Vec<i32> = (1..=cells.len()).map(|i| (i * 4) as i32).collect();
        self.write_data_array_i32("offsets", &offsets)?;

        // Types (VTK_QUAD = 9)
        let types: Vec<u8> = vec![9; cells.len()];
        self.write_data_array_u8("types", &types)?;

        self.end_element("Cells")?;
        Ok(())
    }

    fn write_field_data(&mut self, name: &str, value: f64) -> std::io::Result<()> {
        self.start_element("FieldData", &[])?;
        self.write_indent()?;
        writeln!(
            self.writer,
            "<DataArray type=\"Float64\" Name=\"{}\" NumberOfTuples=\"1\" format=\"ascii\">",
            name
        )?;
        self.indent += 1;
        self.write_indent()?;
        writeln!(self.writer, "{:.10e}", value)?;
        self.indent -= 1;
        self.write_indent()?;
        writeln!(self.writer, "</DataArray>")?;
        self.end_element("FieldData")?;
        Ok(())
    }
}

/// Write fields on the mesh to a VTU file: `node_fields` at the DG nodes
/// (`[element × node]`, as a solution's variables), `element_fields` one value
/// per element, and the element's index as `element_id`, with `time` as the
/// file's `TimeValue`.
///
/// Each element is drawn as its nodes' sub-quads (the module's
/// documentation), so a node field shows its polynomial's detail.
pub fn write_vtk_fields(
    path: impl AsRef<Path>,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    node_fields: &[(&str, &[f64])],
    element_fields: &[(&str, &[f64])],
    time: Option<f64>,
) -> Result<(), VtkError> {
    let n_values = mesh.n_elements * ops.n_nodes;
    for (name, values) in node_fields {
        if values.len() != n_values {
            return Err(VtkError::InvalidMesh(format!(
                "{name}: {} values for {n_values} nodes",
                values.len()
            )));
        }
    }
    for (name, values) in element_fields {
        if values.len() != mesh.n_elements {
            return Err(VtkError::InvalidMesh(format!(
                "{name}: {} values for {} elements",
                values.len(),
                mesh.n_elements
            )));
        }
    }
    let vtk_mesh = build_vtk_mesh(mesh, ops);
    let file = File::create(path)?;
    let mut writer = VtkWriter::new(file);

    writer.write_header()?;
    writer.start_element("UnstructuredGrid", &[])?;
    writer.start_element(
        "Piece",
        &[
            ("NumberOfPoints", &vtk_mesh.points.len().to_string()),
            ("NumberOfCells", &vtk_mesh.cells.len().to_string()),
        ],
    )?;
    writer.write_points(&vtk_mesh.points)?;
    writer.write_cells(&vtk_mesh.cells)?;

    // The node fields: the VTK points are the nodes, in the same order
    let scalars = node_fields.first().map_or("", |(name, _)| *name);
    writer.start_element("PointData", &[("Scalars", scalars)])?;
    for (name, values) in node_fields {
        writer.write_data_array_f64(name, values, 1)?;
    }
    writer.end_element("PointData")?;

    // The element fields on each of the element's sub-quads
    writer.start_element("CellData", &[("Scalars", "element_id")])?;
    let element_ids: Vec<i32> = vtk_mesh.element_ids.iter().map(|&id| id as i32).collect();
    writer.write_data_array_i32("element_id", &element_ids)?;
    let mut per_cell = Vec::with_capacity(vtk_mesh.element_ids.len());
    for (name, values) in element_fields {
        per_cell.clear();
        per_cell.extend(vtk_mesh.element_ids.iter().map(|&k| values[k]));
        writer.write_data_array_f64(name, &per_cell, 1)?;
    }
    writer.end_element("CellData")?;

    writer.end_element("Piece")?;
    // Time as field data (outside Piece, inside UnstructuredGrid)
    if let Some(time) = time {
        writer.write_field_data("TimeValue", time)?;
    }
    writer.end_element("UnstructuredGrid")?;
    writer.write_footer()?;
    Ok(())
}

/// Write SWE solution to VTK file.
///
/// Outputs water depth, velocity components, velocity magnitude, and optionally
/// surface elevation (if bathymetry provided).
pub fn write_vtk_swe(
    path: impl AsRef<Path>,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    solution: &SWESolution2D,
    bathymetry: Option<&Bathymetry2D>,
    time: f64,
    h_min: f64,
) -> Result<(), VtkError> {
    let h_min = Depth::new(h_min);
    let n_points = mesh.n_elements * ops.n_nodes;
    let mut h_data = Vec::with_capacity(n_points);
    let mut u_data = Vec::with_capacity(n_points);
    let mut v_data = Vec::with_capacity(n_points);
    let mut speed_data = Vec::with_capacity(n_points);
    let mut eta_data = Vec::new();
    let mut bathy_data = Vec::new();

    for k in ElementIndex::iter(mesh.n_elements) {
        for i in 0..ops.n_nodes {
            let h = solution.get_var(k, i, 0);
            let hu = solution.get_var(k, i, 1);
            let hv = solution.get_var(k, i, 2);

            let state = SWEState2D { h, hu, hv };
            let (u, v) = state.velocity(h_min);
            h_data.push(h);
            u_data.push(u);
            v_data.push(v);
            speed_data.push(state.velocity_magnitude(h_min));

            if let Some(bathy) = bathymetry {
                let b = bathy.data[k.as_usize() * ops.n_nodes + i];
                bathy_data.push(b);
                eta_data.push(h + b); // Surface elevation = depth + bathymetry
            }
        }
    }

    let mut fields: Vec<(&str, &[f64])> = vec![
        ("h", &h_data),
        ("u", &u_data),
        ("v", &v_data),
        ("velocity_magnitude", &speed_data),
    ];
    if bathymetry.is_some() {
        fields.push(("eta", &eta_data));
        fields.push(("bathymetry", &bathy_data));
    }
    write_vtk_fields(path, mesh, ops, &fields, &[], Some(time))
}

/// Write VTK file with automatic frame numbering.
///
/// Creates filename like `base_0001.vtu` for frame 1.
/// Returns the full path of the created file.
pub fn write_vtk_series(
    base_path: impl AsRef<Path>,
    frame: usize,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    swe: &SWESolution2D,
    bathymetry: Option<&Bathymetry2D>,
    time: f64,
    h_min: f64,
) -> Result<PathBuf, VtkError> {
    let base = base_path.as_ref();
    let stem = base.file_stem().unwrap_or_default().to_string_lossy();
    let parent = base.parent().unwrap_or(Path::new("."));

    let filename = format!("{}_{:04}.vtu", stem, frame);
    let path = parent.join(filename);

    write_vtk_swe(&path, mesh, ops, swe, bathymetry, time, h_min)?;

    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::DGOperators2D;

    #[test]
    fn test_build_vtk_mesh_subcells() {
        // Create a simple 2x2 mesh with P1 (2x2 nodes per element)
        let mesh = Mesh2D::uniform_rectangle(0.0, 2.0, 0.0, 2.0, 2, 2);
        let ops = DGOperators2D::new(1); // P1 = 2x2 nodes

        let vtk_mesh = build_vtk_mesh(&mesh, &ops);

        // 4 elements, each with 4 nodes = 16 points
        assert_eq!(vtk_mesh.points.len(), 16);

        // P1: (2-1)² = 1 subcell per element, 4 elements = 4 subcells
        assert_eq!(vtk_mesh.cells.len(), 4);
        assert_eq!(vtk_mesh.element_ids.len(), 4);
    }

    #[test]
    fn test_build_vtk_mesh_p3() {
        // P3 element has 4x4 = 16 nodes, creates 3x3 = 9 subcells
        let mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 1, 1);
        let ops = DGOperators2D::new(3); // P3 = 4x4 nodes

        let vtk_mesh = build_vtk_mesh(&mesh, &ops);

        // 1 element with 16 nodes
        assert_eq!(vtk_mesh.points.len(), 16);

        // 9 subcells
        assert_eq!(vtk_mesh.cells.len(), 9);
    }

    /// Element fields go on every sub-quad of their element, and fields of
    /// the wrong length are refused.
    #[test]
    fn element_fields_cover_their_elements_sub_quads() {
        let mesh = Mesh2D::uniform_rectangle(0.0, 2.0, 0.0, 1.0, 2, 1);
        let ops = DGOperators2D::new(2); // 4 sub-quads per element
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("fields.vtu");
        let bed = vec![-1.0; 2 * ops.n_nodes];
        write_vtk_fields(
            &path,
            &mesh,
            &ops,
            &[("bed", &bed)],
            &[("dt", &[3.0, 7.0])],
            None,
        )
        .unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let dt = text.split("Name=\"dt\"").nth(1).unwrap();
        let values: Vec<f64> = dt
            .split("</DataArray>")
            .next()
            .unwrap()
            .split_once('>')
            .unwrap()
            .1
            .split_whitespace()
            .map(|v| v.parse().unwrap())
            .collect();
        assert_eq!(values, [3.0, 3.0, 3.0, 3.0, 7.0, 7.0, 7.0, 7.0]);
        assert!(!text.contains("TimeValue"));
        let short = write_vtk_fields(&path, &mesh, &ops, &[], &[("dt", &[3.0])], None);
        assert!(matches!(short, Err(VtkError::InvalidMesh(_))));
    }

    #[test]
    fn test_vtk_mesh_coordinates() {
        // Single element [0,1] x [0,1]
        let mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 1, 1);
        let ops = DGOperators2D::new(1); // P1

        let vtk_mesh = build_vtk_mesh(&mesh, &ops);

        // Check corners are present (approximately)
        let has_origin = vtk_mesh
            .points
            .iter()
            .any(|&[x, y]| x.abs() < 1e-10 && y.abs() < 1e-10);
        let has_corner = vtk_mesh
            .points
            .iter()
            .any(|&[x, y]| (x - 1.0).abs() < 1e-10 && (y - 1.0).abs() < 1e-10);

        assert!(has_origin, "Should have point at origin");
        assert!(has_corner, "Should have point at (1,1)");
    }

    #[test]
    fn test_write_vtk_swe_creates_file() {
        use std::fs;
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let path = dir.path().join("test.vtu");

        let mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 2, 2);
        let ops = DGOperators2D::new(2);
        let solution = SWESolution2D::new(mesh.n_elements, ops.n_nodes);

        write_vtk_swe(&path, &mesh, &ops, &solution, None, 0.0, 1e-6).unwrap();

        assert!(path.exists());
        let content = fs::read_to_string(&path).unwrap();
        assert!(content.contains("VTKFile"));
        assert!(content.contains("UnstructuredGrid"));
        assert!(content.contains("DataArray"));
    }

    #[test]
    fn test_write_vtk_series_naming() {
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let base = dir.path().join("output");

        let mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 1, 1);
        let ops = DGOperators2D::new(1);
        let solution = SWESolution2D::new(mesh.n_elements, ops.n_nodes);

        let path = write_vtk_series(&base, 42, &mesh, &ops, &solution, None, 0.0, 1e-6).unwrap();

        assert!(path.to_string_lossy().contains("output_0042.vtu"));
        assert!(path.exists());
    }
}

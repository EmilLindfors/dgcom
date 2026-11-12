//! 2D mesh representation for quadrilateral elements.
//!
//! The mesh stores:
//! - Vertex coordinates
//! - Element-vertex connectivity (counter-clockwise ordering)
//! - Edge-based connectivity for inter-element flux computation
//! - Boundary edge identification
//!
//! Face convention (counter-clockwise around element):
//! - Face 0 (bottom): from vertex 0 to vertex 1
//! - Face 1 (right):  from vertex 1 to vertex 2
//! - Face 2 (top):    from vertex 2 to vertex 3
//! - Face 3 (left):   from vertex 3 to vertex 0

use crate::mesh::data::BoundaryTag;
use crate::mesh::traits::{
    FaceConnection, Mesh2DGeometry, MeshGeometry, MeshGeometryExt, MeshTopology, Neighbor,
};
use crate::types::ElementIndex;

/// Reference to an element and one of its faces.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ElementFace {
    /// Element index
    pub element: usize,
    /// Face index (0-3 for quads)
    pub face: usize,
}

impl ElementFace {
    pub fn new(element: usize, face: usize) -> Self {
        Self { element, face }
    }
}

/// Information about an edge in the mesh.
#[derive(Clone, Debug)]
pub struct Edge {
    /// Vertex indices (v0, v1) with v0 < v1 for consistent ordering
    pub vertices: (usize, usize),
    /// Left element-face (always present)
    pub left: ElementFace,
    /// Right element-face (None for boundary edges)
    pub right: Option<ElementFace>,
    /// Boundary tag (only for boundary edges)
    pub boundary_tag: Option<BoundaryTag>,
}

/// Position of the cell `(x, y)` of a `2^bits × 2^bits` grid along the
/// Hilbert curve through it: cells next to each other on the curve share a
/// side, and cells near each other in the plane are mostly near on the curve.
fn hilbert_index(mut x: u32, mut y: u32, bits: u32) -> u64 {
    let mut d = 0;
    let mut s = 1u32 << (bits - 1);
    while s > 0 {
        let (rx, ry) = (x & s != 0, y & s != 0);
        d += u64::from(s) * u64::from(s) * ((3 * u64::from(rx)) ^ u64::from(ry));
        // The quadrant's own curve, rotated to start where this one enters it
        x &= s - 1;
        y &= s - 1;
        if !ry {
            if rx {
                x = s - 1 - x;
                y = s - 1 - y;
            }
            std::mem::swap(&mut x, &mut y);
        }
        s >>= 1;
    }
    d
}

impl Edge {
    /// Check if this is a boundary edge.
    pub fn is_boundary(&self) -> bool {
        self.right.is_none()
    }

    /// Check if this is an interior edge.
    pub fn is_interior(&self) -> bool {
        self.right.is_some()
    }
}

/// Why [`Mesh2D::from_quads`] rejected a mesh.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuadMeshError {
    /// An element refers to a vertex that does not exist.
    VertexOutOfRange {
        /// The element
        element: usize,
        /// Its vertex index
        vertex: usize,
    },
    /// More than two faces share the edge between these vertices.
    NonManifoldEdge {
        /// The edge's vertices
        vertices: (usize, usize),
        /// Three of the faces on it
        faces: [ElementFace; 3],
    },
    /// Two faces traverse their shared edge in the same direction: the
    /// elements overlap, or one of them is clockwise.
    Overlap {
        /// The edge's vertices
        vertices: (usize, usize),
        /// The two faces
        faces: [ElementFace; 2],
    },
    /// A periodic face does not exist, is in two pairs, or shares its
    /// vertices with another face.
    Periodic {
        /// The face
        face: ElementFace,
    },
}

impl std::fmt::Display for QuadMeshError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::VertexOutOfRange { element, vertex } => {
                write!(
                    f,
                    "element {element} uses vertex {vertex}, which does not exist"
                )
            }
            Self::NonManifoldEdge { vertices, faces } => write!(
                f,
                "the edge between vertices {} and {} is shared by more than two elements \
                 ({}, {} and {})",
                vertices.0, vertices.1, faces[0].element, faces[1].element, faces[2].element
            ),
            Self::Overlap { vertices, faces } => write!(
                f,
                "elements {} and {} traverse their shared edge (vertices {} and {}) in the \
                 same direction: they overlap, or one is clockwise",
                faces[0].element, faces[1].element, vertices.0, vertices.1
            ),
            Self::Periodic { face } => write!(
                f,
                "periodic face {} of element {} does not exist, is paired twice, or is \
                 shared with another element",
                face.face, face.element
            ),
        }
    }
}

impl std::error::Error for QuadMeshError {}

/// Vertices and counter-clockwise elements of an `nx` × `ny` grid of
/// [x0, x1] × [y0, y1], row by row from the bottom left.
fn structured_quads(
    x0: f64,
    x1: f64,
    y0: f64,
    y1: f64,
    nx: usize,
    ny: usize,
) -> (Vec<[f64; 2]>, Vec<[usize; 4]>) {
    assert!(
        nx > 0 && ny > 0,
        "Need at least one element in each direction"
    );
    assert!(x1 > x0 && y1 > y0, "Invalid domain bounds");
    let dx = (x1 - x0) / nx as f64;
    let dy = (y1 - y0) / ny as f64;
    let vertices = (0..=ny)
        .flat_map(|j| (0..=nx).map(move |i| [x0 + i as f64 * dx, y0 + j as f64 * dy]))
        .collect();
    let elements = (0..ny)
        .flat_map(|j| {
            (0..nx).map(move |i| {
                let v0 = j * (nx + 1) + i; // bottom-left
                [v0, v0 + 1, v0 + nx + 2, v0 + nx + 1]
            })
        })
        .collect();
    (vertices, elements)
}

/// The right face of the last element of row `j` of an `nx`-wide grid and
/// the left face of its first element.
fn periodic_x_pair(nx: usize, j: usize) -> (ElementFace, ElementFace) {
    (
        ElementFace::new(j * nx + nx - 1, 1),
        ElementFace::new(j * nx, 3),
    )
}

/// 2D mesh of quadrilateral elements.
#[derive(Clone)]
pub struct Mesh2D {
    /// Vertex coordinates: vertices[i] = [x, y]
    pub vertices: Vec<[f64; 2]>,

    /// Element-vertex connectivity: elements[k] = [v0, v1, v2, v3]
    /// Vertices are in counter-clockwise order:
    /// - v0: bottom-left  (r=-1, s=-1)
    /// - v1: bottom-right (r=+1, s=-1)
    /// - v2: top-right    (r=+1, s=+1)
    /// - v3: top-left     (r=-1, s=+1)
    pub elements: Vec<[usize; 4]>,

    /// Edge list with connectivity information
    pub edges: Vec<Edge>,

    /// Element-to-edge mapping: element_edges[k][f] = edge index for face f of element k
    pub element_edges: Vec<[usize; 4]>,

    /// Number of elements
    pub n_elements: usize,

    /// Number of edges
    pub n_edges: usize,

    /// Number of boundary edges
    pub n_boundary_edges: usize,

    /// Number of vertices
    pub n_vertices: usize,

    /// Vertex-to-element connectivity: vertex_to_elements[v] = list of element indices
    /// containing vertex v. Used by vertex-based slope limiters (e.g., Kuzmin).
    ///
    /// For structured quad meshes:
    /// - Interior vertices have 4 elements
    /// - Edge boundary vertices have 2 elements
    /// - Corner boundary vertices have 1 element
    pub vertex_to_elements: Vec<Vec<usize>>,
}

impl Mesh2D {
    /// Create a uniform rectangular mesh of [x0, x1] × [y0, y1].
    ///
    /// # Arguments
    /// * `x0`, `x1` - x-coordinate bounds
    /// * `y0`, `y1` - y-coordinate bounds
    /// * `nx` - number of elements in x-direction
    /// * `ny` - number of elements in y-direction
    pub fn uniform_rectangle(x0: f64, x1: f64, y0: f64, y1: f64, nx: usize, ny: usize) -> Self {
        Self::uniform_rectangle_with_bc(x0, x1, y0, y1, nx, ny, BoundaryTag::Wall)
    }

    /// Create a uniform rectangular mesh with a specific boundary tag on all boundaries.
    pub fn uniform_rectangle_with_bc(
        x0: f64,
        x1: f64,
        y0: f64,
        y1: f64,
        nx: usize,
        ny: usize,
        bc_tag: BoundaryTag,
    ) -> Self {
        Self::uniform_rectangle_with_sides(x0, x1, y0, y1, nx, ny, [bc_tag; 4])
    }

    /// Create a uniform rectangular mesh with different boundary tags on each side.
    ///
    /// # Arguments
    /// * `x0`, `x1` - x-coordinate bounds
    /// * `y0`, `y1` - y-coordinate bounds
    /// * `nx` - number of elements in x-direction
    /// * `ny` - number of elements in y-direction
    /// * `bc_tags` - boundary tags for [south, east, north, west] sides
    pub fn uniform_rectangle_with_sides(
        x0: f64,
        x1: f64,
        y0: f64,
        y1: f64,
        nx: usize,
        ny: usize,
        bc_tags: [BoundaryTag; 4],
    ) -> Self {
        let (vertices, elements) = structured_quads(x0, x1, y0, y1, nx, ny);
        // A boundary face of the grid lies on the side of its face number
        Self::from_quads(vertices, elements, &[], |face| bc_tags[face.face])
            .expect("a structured grid is a valid quad mesh")
    }

    /// Create a mesh that is periodic in the x-direction (channel flow), with
    /// walls at y0 and y1.
    pub fn channel_periodic_x(x0: f64, x1: f64, y0: f64, y1: f64, nx: usize, ny: usize) -> Self {
        Self::channel_periodic_x_with_sides(
            x0,
            x1,
            y0,
            y1,
            nx,
            ny,
            [BoundaryTag::Wall, BoundaryTag::Wall],
        )
    }

    /// A mesh periodic in x with the boundary tags `[south, north]` at y0 and
    /// y1: e.g. an alongshore-uniform beach, open to the sea at y0.
    pub fn channel_periodic_x_with_sides(
        x0: f64,
        x1: f64,
        y0: f64,
        y1: f64,
        nx: usize,
        ny: usize,
        [south, north]: [BoundaryTag; 2],
    ) -> Self {
        let (vertices, elements) = structured_quads(x0, x1, y0, y1, nx, ny);
        let periodic: Vec<_> = (0..ny).map(|j| periodic_x_pair(nx, j)).collect();
        // Only the bottom faces (0) of the first row and the top faces (2) of
        // the last are boundaries
        Self::from_quads(vertices, elements, &periodic, |face| {
            if face.face == 0 { south } else { north }
        })
        .expect("a structured grid is a valid quad mesh")
    }

    /// Create a fully periodic mesh (periodic in both x and y).
    pub fn uniform_periodic(x0: f64, x1: f64, y0: f64, y1: f64, nx: usize, ny: usize) -> Self {
        let (vertices, elements) = structured_quads(x0, x1, y0, y1, nx, ny);
        // The top face of the top row meets the bottom face of the bottom row
        let periodic: Vec<_> = (0..ny)
            .map(|j| periodic_x_pair(nx, j))
            .chain((0..nx).map(|i| {
                (
                    ElementFace::new((ny - 1) * nx + i, 2),
                    ElementFace::new(i, 0),
                )
            }))
            .collect();
        Self::from_quads(vertices, elements, &periodic, |_| BoundaryTag::Wall)
            .expect("a structured grid is a valid quad mesh")
    }

    /// Mesh of the quadrilaterals `elements` (vertex indices into
    /// `vertices`, counter-clockwise: face f runs from vertex f to vertex
    /// f + 1), with its edge connectivity.
    ///
    /// - Faces with the same two vertices are the two sides of one interior
    ///   edge. They must traverse it in opposite directions, as they do when
    ///   both elements are counter-clockwise: the kernels pair the face nodes
    ///   of neighbours in reverse order.
    /// - `periodic` pairs faces that are identified although their vertices
    ///   differ; each pair is one interior edge. Neither face may share its
    ///   vertices with another face, and their nodes must match in reverse
    ///   order (e.g. the right face of the last element in a row and the left
    ///   face of the first).
    /// - Every other face is a boundary edge, tagged `boundary_tag(face)`.
    ///
    /// Edges are numbered in the order their first face appears (by element,
    /// then face). The left side of an edge is its first face, or the first
    /// face of its periodic pair, and the edge's vertices are those of its
    /// left side.
    pub fn from_quads(
        vertices: Vec<[f64; 2]>,
        elements: Vec<[usize; 4]>,
        periodic: &[(ElementFace, ElementFace)],
        mut boundary_tag: impl FnMut(ElementFace) -> BoundaryTag,
    ) -> Result<Self, QuadMeshError> {
        use std::collections::HashMap;
        use std::collections::hash_map::Entry;

        let n_elements = elements.len();
        let n_vertices = vertices.len();
        for (k, quad) in elements.iter().enumerate() {
            if let Some(&vertex) = quad.iter().find(|&&v| v >= n_vertices) {
                return Err(QuadMeshError::VertexOutOfRange { element: k, vertex });
            }
        }
        let face_vertices = |face: ElementFace| {
            let quad = elements[face.element];
            (quad[face.face], quad[(face.face + 1) % 4])
        };
        let key = |(a, b): (usize, usize)| (a.min(b), a.max(b));

        // Periodic faces → their pair
        let mut pair_of: HashMap<(usize, usize), usize> =
            HashMap::with_capacity(2 * periodic.len());
        for (p, &(a, b)) in periodic.iter().enumerate() {
            for face in [a, b] {
                if face.element >= n_elements
                    || face.face >= 4
                    || pair_of.insert((face.element, face.face), p).is_some()
                {
                    return Err(QuadMeshError::Periodic { face });
                }
            }
        }

        let mut edges: Vec<Edge> = Vec::with_capacity(2 * n_elements + 64);
        let mut element_edges = vec![[usize::MAX; 4]; n_elements];
        // Vertex pair → edge of ordinary faces; periodic pair → edge
        let mut edge_of_key: HashMap<(usize, usize), usize> =
            HashMap::with_capacity(2 * n_elements);
        let mut edge_of_pair = vec![usize::MAX; periodic.len()];
        for (k, faces) in element_edges.iter_mut().enumerate() {
            for (f, slot) in faces.iter_mut().enumerate() {
                let face = ElementFace::new(k, f);
                if let Some(&p) = pair_of.get(&(k, f)) {
                    if edge_of_pair[p] == usize::MAX {
                        let (left, right) = periodic[p];
                        edge_of_pair[p] = edges.len();
                        edges.push(Edge {
                            vertices: key(face_vertices(left)),
                            left,
                            right: Some(right),
                            boundary_tag: None,
                        });
                    }
                    *slot = edge_of_pair[p];
                    continue;
                }
                let (a, b) = face_vertices(face);
                match edge_of_key.entry(key((a, b))) {
                    Entry::Vacant(entry) => {
                        entry.insert(edges.len());
                        *slot = edges.len();
                        edges.push(Edge {
                            vertices: key((a, b)),
                            left: face,
                            right: None,
                            boundary_tag: None,
                        });
                    }
                    Entry::Occupied(entry) => {
                        let e = *entry.get();
                        let edge = &mut edges[e];
                        if let Some(right) = edge.right {
                            return Err(QuadMeshError::NonManifoldEdge {
                                vertices: edge.vertices,
                                faces: [edge.left, right, face],
                            });
                        }
                        if face_vertices(edge.left) != (b, a) {
                            return Err(QuadMeshError::Overlap {
                                vertices: edge.vertices,
                                faces: [edge.left, face],
                            });
                        }
                        edge.right = Some(face);
                        *slot = e;
                    }
                }
            }
        }
        // A periodic face must not also be the side of an ordinary edge
        for &(a, b) in periodic {
            for face in [a, b] {
                if edge_of_key.contains_key(&key(face_vertices(face))) {
                    return Err(QuadMeshError::Periodic { face });
                }
            }
        }
        for edge in edges.iter_mut().filter(|e| e.right.is_none()) {
            edge.boundary_tag = Some(boundary_tag(edge.left));
        }

        Ok(Self {
            vertex_to_elements: Self::build_vertex_to_elements(&elements, n_vertices),
            vertices,
            elements,
            n_edges: edges.len(),
            n_boundary_edges: edges.iter().filter(|e| e.is_boundary()).count(),
            edges,
            element_edges,
            n_elements,
            n_vertices,
        })
    }

    /// Get the vertices of an element.
    pub fn element_vertices(&self, k: ElementIndex) -> [[f64; 2]; 4] {
        let [v0, v1, v2, v3] = self.elements[k.as_usize()];
        [
            self.vertices[v0],
            self.vertices[v1],
            self.vertices[v2],
            self.vertices[v3],
        ]
    }

    /// Map reference coordinates (r, s) in [-1, 1]² to physical coordinates (x, y).
    ///
    /// Uses bilinear interpolation for quadrilateral elements:
    /// ```text
    /// x(r, s) = (1-r)(1-s)/4 * x0 + (1+r)(1-s)/4 * x1
    ///         + (1+r)(1+s)/4 * x2 + (1-r)(1+s)/4 * x3
    /// ```
    pub fn reference_to_physical(&self, k: ElementIndex, r: f64, s: f64) -> [f64; 2] {
        Self::bilinear(&self.element_vertices(k), r, s)
    }

    /// [`Self::reference_to_physical`] in the element with corners `verts`
    /// ([`Self::element_vertices`]), for many points of one element.
    #[inline]
    pub fn bilinear(verts: &[[f64; 2]; 4], r: f64, s: f64) -> [f64; 2] {
        let [x0, y0] = verts[0];
        let [x1, y1] = verts[1];
        let [x2, y2] = verts[2];
        let [x3, y3] = verts[3];

        // Bilinear shape functions
        let n0 = (1.0 - r) * (1.0 - s) / 4.0;
        let n1 = (1.0 + r) * (1.0 - s) / 4.0;
        let n2 = (1.0 + r) * (1.0 + s) / 4.0;
        let n3 = (1.0 - r) * (1.0 + s) / 4.0;

        let x = n0 * x0 + n1 * x1 + n2 * x2 + n3 * x3;
        let y = n0 * y0 + n1 * y1 + n2 * y2 + n3 * y3;

        [x, y]
    }

    /// Get the edge index for a given element face.
    pub fn edge_for_face(&self, element: ElementIndex, face: usize) -> usize {
        self.element_edges[element.as_usize()][face]
    }

    /// Get the neighbor element across a face, if it exists.
    pub fn neighbor(&self, element: ElementIndex, face: usize) -> Option<ElementFace> {
        let edge_idx = self.element_edges[element.as_usize()][face];
        let edge = &self.edges[edge_idx];

        if edge.left.element == element.as_usize() && edge.left.face == face {
            edge.right
        } else {
            Some(edge.left)
        }
    }

    /// Check if a face is on the boundary.
    pub fn is_boundary_face(&self, element: ElementIndex, face: usize) -> bool {
        let edge_idx = self.element_edges[element.as_usize()][face];
        self.edges[edge_idx].is_boundary()
    }

    /// Get the boundary tag for a face, if it's a boundary face.
    pub fn boundary_tag(&self, element: ElementIndex, face: usize) -> Option<BoundaryTag> {
        let edge_idx = self.element_edges[element.as_usize()][face];
        self.edges[edge_idx].boundary_tag
    }

    /// Get minimum element diameter (for CFL computation).
    pub fn h_min(&self) -> f64 {
        let mut h_min = f64::INFINITY;
        for k in ElementIndex::iter(self.n_elements) {
            let verts = self.element_vertices(k);
            // Approximate diameter as minimum edge length
            for i in 0..4 {
                let [x0, y0] = verts[i];
                let [x1, y1] = verts[(i + 1) % 4];
                let len = ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt();
                h_min = h_min.min(len);
            }
        }
        h_min
    }

    /// Get maximum element diameter.
    pub fn h_max(&self) -> f64 {
        let mut h_max: f64 = 0.0;
        for k in ElementIndex::iter(self.n_elements) {
            let verts = self.element_vertices(k);
            // Use diagonal as maximum dimension
            let [x0, y0] = verts[0];
            let [x2, y2] = verts[2];
            let diag = ((x2 - x0).powi(2) + (y2 - y0).powi(2)).sqrt();
            h_max = h_max.max(diag);
        }
        h_max
    }

    /// Get the diameter of a specific element (diagonal length).
    pub fn element_diameter(&self, k: ElementIndex) -> f64 {
        let verts = self.element_vertices(k);
        let [x0, y0] = verts[0];
        let [x2, y2] = verts[2];
        ((x2 - x0).powi(2) + (y2 - y0).powi(2)).sqrt()
    }

    /// Get all elements sharing a given vertex.
    ///
    /// Used by vertex-based slope limiters (e.g., Kuzmin) to compute
    /// local bounds from the vertex patch.
    #[inline]
    pub fn elements_at_vertex(&self, vertex: usize) -> &[usize] {
        &self.vertex_to_elements[vertex]
    }

    /// Get the global vertex indices for an element.
    #[inline]
    pub fn element_vertex_indices(&self, k: ElementIndex) -> [usize; 4] {
        self.elements[k.as_usize()]
    }

    /// The submesh of the elements for which `keep` is true, e.g. the water
    /// elements of a rectangular grid over a coastline (TODO P2.4).
    ///
    /// Kept elements keep their vertices (renumbered), orientation and face
    /// order; boundary faces keep their tags. A face whose neighbour is dropped
    /// becomes a boundary face tagged `new_boundary`.
    ///
    /// Returns the submesh and, for each of its elements, the index of that
    /// element in `self`.
    pub fn retain_elements(
        &self,
        keep: impl Fn(ElementIndex) -> bool,
        new_boundary: BoundaryTag,
    ) -> (Mesh2D, Vec<usize>) {
        let kept: Vec<usize> = (0..self.n_elements)
            .filter(|&k| keep(ElementIndex::new(k)))
            .collect();
        (self.select_elements(&kept, new_boundary), kept)
    }

    /// The mesh with its elements numbered along a Hilbert curve through
    /// their centroids, and, for each of its elements, the index of that
    /// element in `self`.
    ///
    /// A mesh generator's numbering follows its own algorithm: Gmsh's
    /// recombined quadrilaterals have one face neighbour next to them and
    /// the others anywhere (on the 95,000-element Chiloé coastline mesh 44 %
    /// of the face neighbours were more than 1000 elements apart, 39 % more
    /// than 8000). Every face flux then reads a neighbour's state, bed and
    /// geometry from memory that is in no cache. Along the curve 92 % are
    /// within 64 elements and 99 % within 1024, and the same run takes 1.26×
    /// less time (`notes/performance.md`, P2.8). The solution changes by
    /// round-off only (sums in another order).
    pub fn in_hilbert_order(&self) -> (Mesh2D, Vec<usize>) {
        const BITS: u32 = 16;
        let centroid = |k: usize| {
            let v = self.element_vertices(ElementIndex::new(k));
            [0, 1].map(|d| 0.25 * (v[0][d] + v[1][d] + v[2][d] + v[3][d]))
        };
        let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
        for k in 0..self.n_elements {
            let c = centroid(k);
            for d in 0..2 {
                lo[d] = lo[d].min(c[d]);
                hi[d] = hi[d].max(c[d]);
            }
        }
        // One scale for both directions: the curve's cells are squares
        let extent = (hi[0] - lo[0]).max(hi[1] - lo[1]);
        let scale = if extent > 0.0 {
            ((1u32 << BITS) - 1) as f64 / extent
        } else {
            0.0
        };
        let mut order: Vec<(u64, usize)> = (0..self.n_elements)
            .map(|k| {
                let c = centroid(k);
                let [x, y] = [0, 1].map(|d| ((c[d] - lo[d]) * scale) as u32);
                (hilbert_index(x, y, BITS), k)
            })
            .collect();
        // Ties (and equal keys) keep the old order
        order.sort();
        let order: Vec<usize> = order.into_iter().map(|(_, k)| k).collect();
        (self.select_elements(&order, BoundaryTag::Wall), order)
    }

    /// The mesh of the elements `kept` of `self` (distinct), in that order:
    /// [`Self::retain_elements`] for a given list, which may also reorder.
    fn select_elements(&self, kept: &[usize], new_boundary: BoundaryTag) -> Mesh2D {
        let mut new_element = vec![None; self.n_elements];
        for (new, &old) in kept.iter().enumerate() {
            new_element[old] = Some(new);
        }

        let mut new_vertex = vec![None; self.n_vertices];
        let mut vertices = Vec::new();
        let elements: Vec<[usize; 4]> = kept
            .iter()
            .map(|&old| {
                self.elements[old].map(|v| {
                    *new_vertex[v].get_or_insert_with(|| {
                        vertices.push(self.vertices[v]);
                        vertices.len() - 1
                    })
                })
            })
            .collect();

        // Interior edges between kept elements whose faces do not share
        // vertices (periodic ones) stay interior
        let map_face =
            |ef: ElementFace| new_element[ef.element].map(|k| ElementFace::new(k, ef.face));
        let face_key = |ef: ElementFace| {
            let quad = self.elements[ef.element];
            let (a, b) = (quad[ef.face], quad[(ef.face + 1) % 4]);
            (a.min(b), a.max(b))
        };
        let periodic: Vec<_> = self
            .edges
            .iter()
            .filter_map(|edge| {
                let right = edge.right?;
                let pair = (map_face(edge.left)?, map_face(right)?);
                (face_key(edge.left) != face_key(right)).then_some(pair)
            })
            .collect();
        Mesh2D::from_quads(vertices, elements, &periodic, |face| {
            // A boundary face of the submesh keeps its tag, or faced a
            // dropped element
            let old = ElementFace::new(kept[face.element], face.face);
            let edge = &self.edges[self.element_edges[old.element][old.face]];
            edge.boundary_tag.unwrap_or(new_boundary)
        })
        .expect("a submesh of a valid mesh is valid")
    }

    /// Build vertex-to-element connectivity from element-vertex connectivity.
    pub(crate) fn build_vertex_to_elements(
        elements: &[[usize; 4]],
        n_vertices: usize,
    ) -> Vec<Vec<usize>> {
        let mut v2e = vec![Vec::with_capacity(4); n_vertices];
        for (elem_idx, elem) in elements.iter().enumerate() {
            for &vertex in elem {
                v2e[vertex].push(elem_idx);
            }
        }
        v2e
    }

    /// Compute face normal and surface Jacobian for an element face.
    ///
    /// This computes the geometry on-the-fly. For performance-critical code,
    /// use `GeometricFactors2D` which pre-computes these values.
    fn compute_face_geometry(&self, element: ElementIndex, face: usize) -> ([f64; 2], f64) {
        let verts = self.element_vertices(element);
        let (v_start, v_end) = match face {
            0 => (verts[0], verts[1]), // bottom: v0 -> v1
            1 => (verts[1], verts[2]), // right:  v1 -> v2
            2 => (verts[2], verts[3]), // top:    v2 -> v3
            3 => (verts[3], verts[0]), // left:   v3 -> v0
            _ => panic!("Invalid face index {face} for 2D quad (expected 0-3)"),
        };

        let dx = v_end[0] - v_start[0];
        let dy = v_end[1] - v_start[1];
        let edge_len = (dx * dx + dy * dy).sqrt();
        let surface_j = edge_len / 2.0; // Reference edge has length 2

        // Outward normal: rotate edge tangent 90 degrees clockwise
        // (for CCW vertex ordering, this points outward)
        let nx = dy / edge_len;
        let ny = -dx / edge_len;

        ([nx, ny], surface_j)
    }
}

// =============================================================================
// Trait Implementations
// =============================================================================

impl MeshTopology for Mesh2D {
    type Coord = [f64; 2];
    type RefCoord = [f64; 2];
    type BoundaryTag = BoundaryTag;

    const FACES_PER_ELEMENT: usize = 4;

    #[inline]
    fn n_elements(&self) -> usize {
        self.n_elements
    }

    #[inline]
    fn n_faces(&self) -> usize {
        self.n_edges
    }

    #[inline]
    fn n_boundary_faces(&self) -> usize {
        self.n_boundary_edges
    }

    fn face_connection(
        &self,
        element: usize,
        local_face: usize,
    ) -> FaceConnection<Self::BoundaryTag> {
        let edge_idx = self.element_edges[element][local_face];
        let edge = &self.edges[edge_idx];

        // Determine if we're the "left" or "right" side of this edge
        let is_left = edge.left.element == element && edge.left.face == local_face;

        if is_left {
            match edge.right {
                Some(ef) => FaceConnection::Interior(Neighbor {
                    element: ef.element,
                    face: ef.face,
                }),
                None => FaceConnection::Boundary(edge.boundary_tag.unwrap_or_default()),
            }
        } else {
            // We must be the right side, so left is our neighbor
            FaceConnection::Interior(Neighbor {
                element: edge.left.element,
                face: edge.left.face,
            })
        }
    }
}

impl MeshGeometry for Mesh2D {
    fn reference_to_physical(&self, element: usize, ref_coord: [f64; 2]) -> [f64; 2] {
        Mesh2D::reference_to_physical(self, ElementIndex::new(element), ref_coord[0], ref_coord[1])
    }

    fn physical_to_reference(&self, element: usize, coord: [f64; 2]) -> [f64; 2] {
        // For bilinear quads, we need Newton iteration to invert the mapping.
        // For simplicity, use a Newton-Raphson iteration.
        let tol = 1e-12;
        let max_iter = 20;

        let mut r = 0.0;
        let mut s = 0.0;

        let verts = self.element_vertices(ElementIndex::new(element));
        let [x0, y0] = verts[0];
        let [x1, y1] = verts[1];
        let [x2, y2] = verts[2];
        let [x3, y3] = verts[3];

        for _ in 0..max_iter {
            // Evaluate mapping at current (r, s)
            let nr = (1.0 - r) / 2.0;
            let pr = (1.0 + r) / 2.0;
            let ns = (1.0 - s) / 2.0;
            let ps = (1.0 + s) / 2.0;

            let x_curr = nr * ns * x0 + pr * ns * x1 + pr * ps * x2 + nr * ps * x3;
            let y_curr = nr * ns * y0 + pr * ns * y1 + pr * ps * y2 + nr * ps * y3;

            let fx = x_curr - coord[0];
            let fy = y_curr - coord[1];

            if fx * fx + fy * fy < tol * tol {
                break;
            }

            // Compute Jacobian
            let dx_dr = -0.5 * ns * x0 + 0.5 * ns * x1 + 0.5 * ps * x2 - 0.5 * ps * x3;
            let dx_ds = -0.5 * nr * x0 - 0.5 * pr * x1 + 0.5 * pr * x2 + 0.5 * nr * x3;
            let dy_dr = -0.5 * ns * y0 + 0.5 * ns * y1 + 0.5 * ps * y2 - 0.5 * ps * y3;
            let dy_ds = -0.5 * nr * y0 - 0.5 * pr * y1 + 0.5 * pr * y2 + 0.5 * nr * y3;

            let det = dx_dr * dy_ds - dx_ds * dy_dr;
            let dr = (dy_ds * fx - dx_ds * fy) / det;
            let ds = (-dy_dr * fx + dx_dr * fy) / det;

            r -= dr;
            s -= ds;
        }

        [r, s]
    }

    fn jacobian_det(&self, element: usize) -> f64 {
        // For bilinear quads, the Jacobian is not constant.
        // We evaluate at the element center (r=0, s=0) as a representative value.
        let verts = self.element_vertices(ElementIndex::new(element));
        let [x0, y0] = verts[0];
        let [x1, y1] = verts[1];
        let [x2, y2] = verts[2];
        let [x3, y3] = verts[3];

        // At (r=0, s=0):
        // dx/dr = 0.25 * (-x0 + x1 + x2 - x3)
        // dx/ds = 0.25 * (-x0 - x1 + x2 + x3)
        // dy/dr = 0.25 * (-y0 + y1 + y2 - y3)
        // dy/ds = 0.25 * (-y0 - y1 + y2 + y3)
        let dx_dr = 0.25 * (-x0 + x1 + x2 - x3);
        let dx_ds = 0.25 * (-x0 - x1 + x2 + x3);
        let dy_dr = 0.25 * (-y0 + y1 + y2 - y3);
        let dy_ds = 0.25 * (-y0 - y1 + y2 + y3);

        (dx_dr * dy_ds - dx_ds * dy_dr).abs()
    }

    #[inline]
    fn h_min(&self) -> f64 {
        Mesh2D::h_min(self)
    }

    #[inline]
    fn h_max(&self) -> f64 {
        Mesh2D::h_max(self)
    }

    #[inline]
    fn element_diameter(&self, element: usize) -> f64 {
        Mesh2D::element_diameter(self, ElementIndex::new(element))
    }
}

impl MeshGeometryExt for Mesh2D {
    type Normal = [f64; 2];

    fn face_normal(&self, element: usize, local_face: usize) -> [f64; 2] {
        let (normal, _) = self.compute_face_geometry(ElementIndex::new(element), local_face);
        normal
    }

    fn surface_jacobian(&self, element: usize, local_face: usize) -> f64 {
        let (_, surface_j) = self.compute_face_geometry(ElementIndex::new(element), local_face);
        surface_j
    }
}

impl Mesh2DGeometry for Mesh2D {
    fn face_normals_array(&self, element: usize) -> [[f64; 2]; 4] {
        let mut normals = [[0.0; 2]; 4];
        let k = ElementIndex::new(element);
        for f in 0..4 {
            let (n, _) = self.compute_face_geometry(k, f);
            normals[f] = n;
        }
        normals
    }

    fn surface_jacobians_array(&self, element: usize) -> [f64; 4] {
        let mut sj = [0.0; 4];
        let k = ElementIndex::new(element);
        for f in 0..4 {
            let (_, s) = self.compute_face_geometry(k, f);
            sj[f] = s;
        }
        sj
    }

    fn element_vertices(&self, element: usize) -> [[f64; 2]; 4] {
        Mesh2D::element_vertices(self, ElementIndex::new(element))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ElementIndex;

    fn k(idx: usize) -> ElementIndex {
        ElementIndex::new(idx)
    }

    /// Every interior face points back at itself; every boundary face has a
    /// tag; element_edges and edges agree.
    fn assert_consistent(mesh: &Mesh2D) {
        for ki in 0..mesh.n_elements {
            for face in 0..4 {
                let edge = &mesh.edges[mesh.edge_for_face(k(ki), face)];
                let me = ElementFace::new(ki, face);
                assert!(edge.left == me || edge.right == Some(me));
                match mesh.neighbor(k(ki), face) {
                    Some(nb) => {
                        let back = mesh.neighbor(k(nb.element), nb.face);
                        assert_eq!(back, Some(me), "element {ki} face {face}");
                    }
                    None => assert!(mesh.boundary_tag(k(ki), face).is_some()),
                }
            }
            for v in mesh.elements[ki] {
                assert!(mesh.vertex_to_elements[v].contains(&ki));
            }
        }
        assert_eq!(
            mesh.n_boundary_edges,
            mesh.edges.iter().filter(|e| e.is_boundary()).count()
        );
    }

    /// Two unit squares side by side, counter-clockwise.
    fn two_squares() -> (Vec<[f64; 2]>, Vec<[usize; 4]>) {
        let vertices = vec![
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
        ];
        (vertices, vec![[0, 1, 4, 3], [1, 2, 5, 4]])
    }

    #[test]
    fn test_from_quads_connects_and_tags() {
        let (vertices, elements) = two_squares();
        let mesh = Mesh2D::from_quads(vertices, elements, &[], |face| {
            if face.element == 0 && face.face == 3 {
                BoundaryTag::Open
            } else {
                BoundaryTag::Wall
            }
        })
        .unwrap();
        assert_eq!((mesh.n_edges, mesh.n_boundary_edges), (7, 6));
        assert_consistent(&mesh);
        // The shared edge: left is the first face to appear
        assert_eq!(
            mesh.neighbor(k(0), 1),
            Some(ElementFace::new(1, 3)),
            "right face of 0 meets the left face of 1"
        );
        let shared = &mesh.edges[mesh.edge_for_face(k(0), 1)];
        assert_eq!(
            (shared.left, shared.right),
            (ElementFace::new(0, 1), Some(ElementFace::new(1, 3)))
        );
        assert_eq!(mesh.boundary_tag(k(0), 3), Some(BoundaryTag::Open));
        assert_eq!(mesh.boundary_tag(k(1), 1), Some(BoundaryTag::Wall));
    }

    #[test]
    fn test_from_quads_rejects_bad_topology() {
        let wall = |_| BoundaryTag::Wall;
        // Second element clockwise: both traverse the shared edge upwards
        let (vertices, _) = two_squares();
        let error = Mesh2D::from_quads(
            vertices.clone(),
            vec![[0, 1, 4, 3], [1, 4, 5, 2]],
            &[],
            wall,
        );
        assert!(
            matches!(
                error,
                Err(QuadMeshError::Overlap {
                    vertices: (1, 4),
                    ..
                })
            ),
            "{:?}",
            error.as_ref().err()
        );
        // A third element on the edge 1–4
        let mut more = vertices.clone();
        more.extend([[1.5, 0.5], [1.5, 0.7]]);
        let error = Mesh2D::from_quads(
            more,
            vec![[0, 1, 4, 3], [1, 2, 5, 4], [4, 1, 6, 7]],
            &[],
            wall,
        );
        assert!(
            matches!(
                error,
                Err(QuadMeshError::NonManifoldEdge {
                    vertices: (1, 4),
                    ..
                })
            ),
            "{:?}",
            error.as_ref().err()
        );
        let error = Mesh2D::from_quads(vertices.clone(), vec![[0, 1, 4, 9]], &[], wall);
        assert_eq!(
            error.err(),
            Some(QuadMeshError::VertexOutOfRange {
                element: 0,
                vertex: 9
            })
        );
        // A periodic face that is an interior face, and one paired twice
        let (a, b) = (ElementFace::new(0, 1), ElementFace::new(1, 1));
        let error = Mesh2D::from_quads(vertices.clone(), two_squares().1, &[(a, b)], wall);
        assert_eq!(error.err(), Some(QuadMeshError::Periodic { face: a }));
        let (c, d) = (ElementFace::new(0, 3), ElementFace::new(1, 1));
        let error = Mesh2D::from_quads(vertices, two_squares().1, &[(c, d), (d, c)], wall);
        assert!(
            matches!(error, Err(QuadMeshError::Periodic { .. })),
            "{:?}",
            error.as_ref().err()
        );
    }

    /// A single element periodic in both directions is its own neighbour
    /// across both pairs of faces.
    #[test]
    fn test_single_periodic_element() {
        let mesh = Mesh2D::uniform_periodic(0.0, 1.0, 0.0, 1.0, 1, 1);
        assert_eq!((mesh.n_edges, mesh.n_boundary_edges), (2, 0));
        assert_consistent(&mesh);
        assert_eq!(mesh.neighbor(k(0), 1), Some(ElementFace::new(0, 3)));
        assert_eq!(mesh.neighbor(k(0), 2), Some(ElementFace::new(0, 0)));
    }

    /// A submesh of a periodic mesh keeps the periodic edges between kept
    /// elements, and a periodic face whose partner is dropped becomes a
    /// boundary. (With the edge's vertices taken from the dropped side, this
    /// used to panic.)
    #[test]
    fn test_retain_elements_of_a_periodic_mesh() {
        let mesh = Mesh2D::uniform_periodic(0.0, 3.0, 0.0, 2.0, 3, 2);
        // Drop the top row's last element (k = 5)
        let (sub, old) = mesh.retain_elements(|k| k.as_usize() != 5, BoundaryTag::Open);
        assert_eq!(old, [0, 1, 2, 3, 4]);
        assert_consistent(&sub);
        // Bottom row: still periodic in x
        assert_eq!(sub.neighbor(k(2), 1), Some(ElementFace::new(0, 3)));
        // Top row: element 3's left face faced the dropped element 5
        assert_eq!(sub.neighbor(k(3), 3), None);
        assert_eq!(sub.boundary_tag(k(3), 3), Some(BoundaryTag::Open));
        // Periodic in y between kept elements: top of 3 meets bottom of 0
        assert_eq!(sub.neighbor(k(3), 2), Some(ElementFace::new(0, 0)));
        // ... and the bottom of 2 faced the dropped top-row element 5
        assert_eq!(sub.boundary_tag(k(2), 0), Some(BoundaryTag::Open));
    }

    /// The Hilbert order is the same mesh (elements, neighbours, boundary
    /// tags) renumbered, and on a 2ⁿ × 2ⁿ grid every element shares a face
    /// with the next one (row-major order has neighbours a row apart).
    #[test]
    fn hilbert_order_is_the_same_mesh_with_neighbours_next_to_each_other() {
        // The curve visits every cell once, moving one cell at a time
        let mut cells: Vec<_> = (0..8u32)
            .flat_map(|y| (0..8u32).map(move |x| (hilbert_index(x, y, 3), x, y)))
            .collect();
        cells.sort();
        for (i, pair) in cells.windows(2).enumerate() {
            assert_eq!(pair[0].0, i as u64);
            assert_eq!(
                pair[0].1.abs_diff(pair[1].1) + pair[0].2.abs_diff(pair[1].2),
                1
            );
        }

        let mut mesh = Mesh2D::uniform_rectangle(0.0, 16.0, 0.0, 16.0, 16, 16);
        for edge in mesh.edges.iter_mut().filter(|e| e.is_boundary()) {
            let x = 0.5 * (mesh.vertices[edge.vertices.0][0] + mesh.vertices[edge.vertices.1][0]);
            if x == 0.0 {
                edge.boundary_tag = Some(BoundaryTag::Open);
            }
        }
        let (sorted, old) = mesh.in_hilbert_order();
        assert_eq!(sorted.n_elements, mesh.n_elements);
        assert_eq!(sorted.n_vertices, mesh.n_vertices);
        assert_eq!(sorted.n_edges, mesh.n_edges);
        assert_consistent(&sorted);
        let mut seen = old.clone();
        seen.sort_unstable();
        assert!(seen.iter().copied().eq(0..mesh.n_elements), "a permutation");
        for (new, &o) in old.iter().enumerate() {
            assert_eq!(sorted.element_vertices(k(new)), mesh.element_vertices(k(o)));
            for face in 0..4 {
                match mesh.neighbor(k(o), face) {
                    Some(nb) => {
                        let sorted_nb = sorted.neighbor(k(new), face).unwrap();
                        assert_eq!(
                            (old[sorted_nb.element], sorted_nb.face),
                            (nb.element, nb.face)
                        );
                    }
                    None => assert_eq!(
                        sorted.boundary_tag(k(new), face),
                        mesh.boundary_tag(k(o), face)
                    ),
                }
            }
            if new + 1 < sorted.n_elements {
                let next = (0..4).filter_map(|face| sorted.neighbor(k(new), face));
                assert!(
                    next.map(|nb| nb.element).any(|e| e == new + 1),
                    "element {new}"
                );
            }
        }
    }

    #[test]
    fn test_retain_elements() {
        // 4 × 3 grid, open on the west side; drop the top-right 2 × 2 block
        // to leave an L shape
        let mut mesh = Mesh2D::uniform_rectangle(0.0, 4.0, 0.0, 3.0, 4, 3);
        for edge in mesh.edges.iter_mut().filter(|e| e.is_boundary()) {
            let x = 0.5 * (mesh.vertices[edge.vertices.0][0] + mesh.vertices[edge.vertices.1][0]);
            if x == 0.0 {
                edge.boundary_tag = Some(BoundaryTag::Open);
            }
        }
        let dropped = |k: ElementIndex| {
            let [x, y] = mesh.reference_to_physical(k, 0.0, 0.0);
            x > 2.0 && y > 1.0
        };
        let (sub, old) = mesh.retain_elements(|k| !dropped(k), BoundaryTag::Wall);

        assert_eq!(sub.n_elements, 8);
        assert_eq!(old.len(), 8);
        assert!(old.iter().all(|&o| !dropped(k(o))));
        // No kept element uses the vertices with x ∈ {3, 4}, y ∈ {2, 3}
        assert_eq!(sub.n_vertices, mesh.n_vertices - 4);
        // Perimeter of the L: 4 + 1 + 2 + 2 + 2 + 3 unit edges
        assert_eq!(sub.n_boundary_edges, 14);
        assert_consistent(&sub);

        for (new, &o) in old.iter().enumerate() {
            assert_eq!(sub.element_vertices(k(new)), mesh.element_vertices(k(o)));
            for face in 0..4 {
                let tag = sub.boundary_tag(k(new), face);
                match mesh.neighbor(k(o), face) {
                    // Faces to dropped elements become walls
                    Some(nb) if dropped(k(nb.element)) => assert_eq!(tag, Some(BoundaryTag::Wall)),
                    // Interior faces stay interior, with the same neighbour
                    Some(nb) => {
                        let sub_nb = sub.neighbor(k(new), face).unwrap();
                        assert_eq!((old[sub_nb.element], sub_nb.face), (nb.element, nb.face));
                    }
                    // Original boundary faces keep their tags (open in the west)
                    None => assert_eq!(tag, mesh.boundary_tag(k(o), face)),
                }
            }
        }

        // Keeping everything reproduces the mesh
        let (all, _) = mesh.retain_elements(|_| true, BoundaryTag::Wall);
        assert_eq!(all.n_edges, mesh.n_edges);
        assert_eq!(all.element_edges, mesh.element_edges);
        assert_consistent(&all);
    }

    #[test]
    fn test_uniform_rectangle_dimensions() {
        let mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 3, 2);

        assert_eq!(mesh.n_elements, 6); // 3 × 2
        assert_eq!(mesh.n_vertices, 12); // 4 × 3
        assert_eq!(mesh.elements.len(), 6);
        assert_eq!(mesh.vertices.len(), 12);
    }

    #[test]
    fn test_uniform_rectangle_vertices() {
        let mesh = Mesh2D::uniform_rectangle(0.0, 2.0, 0.0, 1.0, 2, 1);

        // Should have 6 vertices in a 3×2 grid
        assert_eq!(mesh.n_vertices, 6);

        // Check corner vertices
        assert!((mesh.vertices[0][0] - 0.0).abs() < 1e-14);
        assert!((mesh.vertices[0][1] - 0.0).abs() < 1e-14);
        assert!((mesh.vertices[2][0] - 2.0).abs() < 1e-14);
        assert!((mesh.vertices[2][1] - 0.0).abs() < 1e-14);
        assert!((mesh.vertices[5][0] - 2.0).abs() < 1e-14);
        assert!((mesh.vertices[5][1] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_element_vertex_ordering() {
        let mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 2, 2);

        // Check that each element has counter-clockwise vertices
        for ki in 0..mesh.n_elements {
            let verts = mesh.element_vertices(k(ki));

            // Compute signed area (should be positive for CCW)
            let mut area = 0.0;
            for i in 0..4 {
                let [x0, y0] = verts[i];
                let [x1, y1] = verts[(i + 1) % 4];
                area += (x1 - x0) * (y1 + y0);
            }
            assert!(area < 0.0, "Element {} should have CCW vertices", ki);
        }
    }

    #[test]
    fn test_edge_count() {
        let mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 3, 2);

        // For a 3×2 grid:
        // Horizontal edges: 3 × 3 = 9
        // Vertical edges: 4 × 2 = 8
        // Total: 17
        assert_eq!(mesh.n_edges, 17);
    }

    #[test]
    fn test_boundary_edges() {
        let mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 3, 2);

        // Boundary edges: 3 + 3 + 2 + 2 = 10
        assert_eq!(mesh.n_boundary_edges, 10);

        // Verify boundary detection
        let boundary_count = mesh.edges.iter().filter(|e| e.is_boundary()).count();
        assert_eq!(boundary_count, 10);
    }

    #[test]
    fn test_neighbor_connectivity() {
        let mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 2, 2);

        // Element 0 (bottom-left) should have:
        // - No neighbor on bottom (face 0)
        // - Element 1 on right (face 1)
        // - Element 2 on top (face 2)
        // - No neighbor on left (face 3)
        assert!(mesh.is_boundary_face(k(0), 0));
        assert!(!mesh.is_boundary_face(k(0), 1));
        assert!(!mesh.is_boundary_face(k(0), 2));
        assert!(mesh.is_boundary_face(k(0), 3));

        let right_neighbor = mesh.neighbor(k(0), 1).unwrap();
        assert_eq!(right_neighbor.element, 1);

        let top_neighbor = mesh.neighbor(k(0), 2).unwrap();
        assert_eq!(top_neighbor.element, 2);
    }

    #[test]
    fn test_periodic_mesh_no_boundaries() {
        let mesh = Mesh2D::uniform_periodic(0.0, 1.0, 0.0, 1.0, 3, 2);

        assert_eq!(mesh.n_boundary_edges, 0);

        // All faces should have neighbors
        for ki in 0..mesh.n_elements {
            for face in 0..4 {
                assert!(
                    !mesh.is_boundary_face(k(ki), face),
                    "Element {} face {} should not be boundary",
                    ki,
                    face
                );
                assert!(
                    mesh.neighbor(k(ki), face).is_some(),
                    "Element {} face {} should have neighbor",
                    ki,
                    face
                );
            }
        }
    }

    #[test]
    fn test_channel_mesh_periodicity() {
        let mesh = Mesh2D::channel_periodic_x(0.0, 1.0, 0.0, 1.0, 3, 2);

        // Top and bottom are walls, left and right are periodic
        // Wall edges: 3 + 3 = 6
        assert_eq!(mesh.n_boundary_edges, 6);

        // Check that left-right faces are connected periodically
        // Element 0's left face should connect to element 2 (rightmost in row)
        let left_neighbor = mesh.neighbor(k(0), 3).unwrap();
        assert_eq!(left_neighbor.element, 2);

        // Element 2's right face should connect to element 0
        let right_neighbor = mesh.neighbor(k(2), 1).unwrap();
        assert_eq!(right_neighbor.element, 0);
    }

    #[test]
    fn a_channel_takes_its_south_and_north_tags() {
        let mesh = Mesh2D::channel_periodic_x_with_sides(
            0.0,
            1.0,
            0.0,
            1.0,
            3,
            2,
            [BoundaryTag::Open, BoundaryTag::Wall],
        );
        assert_eq!(mesh.n_boundary_edges, 6);
        for i in 0..3 {
            assert_eq!(mesh.boundary_tag(k(i), 0), Some(BoundaryTag::Open));
            assert_eq!(mesh.boundary_tag(k(3 + i), 2), Some(BoundaryTag::Wall));
            assert_eq!(mesh.boundary_tag(k(i), 2), None);
        }
        assert_eq!(mesh.neighbor(k(3), 3).unwrap().element, 5);
    }

    #[test]
    fn test_h_min_h_max() {
        let mesh = Mesh2D::uniform_rectangle(0.0, 2.0, 0.0, 1.0, 2, 1);

        // Elements are 1.0 × 1.0 squares
        assert!((mesh.h_min() - 1.0).abs() < 1e-14);
        assert!((mesh.h_max() - 2.0_f64.sqrt()).abs() < 1e-14); // diagonal
    }

    #[test]
    fn test_reference_to_physical() {
        let mesh = Mesh2D::uniform_rectangle(0.0, 2.0, 0.0, 1.0, 2, 1);

        // Element 0 spans [0, 1] × [0, 1]
        // Reference corner (-1, -1) should map to (0, 0)
        let [x, y] = mesh.reference_to_physical(k(0), -1.0, -1.0);
        assert!((x - 0.0).abs() < 1e-14);
        assert!((y - 0.0).abs() < 1e-14);

        // Reference corner (1, 1) should map to (1, 1)
        let [x, y] = mesh.reference_to_physical(k(0), 1.0, 1.0);
        assert!((x - 1.0).abs() < 1e-14);
        assert!((y - 1.0).abs() < 1e-14);

        // Reference center (0, 0) should map to (0.5, 0.5)
        let [x, y] = mesh.reference_to_physical(k(0), 0.0, 0.0);
        assert!((x - 0.5).abs() < 1e-14);
        assert!((y - 0.5).abs() < 1e-14);

        // Element 1 spans [1, 2] × [0, 1]
        // Reference center (0, 0) should map to (1.5, 0.5)
        let [x, y] = mesh.reference_to_physical(k(1), 0.0, 0.0);
        assert!((x - 1.5).abs() < 1e-14);
        assert!((y - 0.5).abs() < 1e-14);
    }

    #[test]
    fn test_element_edges_mapping() {
        let mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 2, 2);

        // Each element should have 4 edge indices
        for k in 0..mesh.n_elements {
            for face in 0..4 {
                let edge_idx = mesh.element_edges[k][face];
                assert!(edge_idx < mesh.n_edges);
            }
        }
    }

    #[test]
    fn test_vertex_to_elements_structured() {
        // 3x3 mesh of elements
        let mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 3, 3);

        // 4x4 = 16 vertices
        assert_eq!(mesh.vertex_to_elements.len(), 16);

        // Corner vertices should have 1 element
        // Vertex 0 is bottom-left corner
        assert_eq!(mesh.elements_at_vertex(0).len(), 1);
        // Vertex 3 is bottom-right corner
        assert_eq!(mesh.elements_at_vertex(3).len(), 1);
        // Vertex 12 is top-left corner
        assert_eq!(mesh.elements_at_vertex(12).len(), 1);
        // Vertex 15 is top-right corner
        assert_eq!(mesh.elements_at_vertex(15).len(), 1);

        // Edge vertices (not corners) should have 2 elements
        // Vertex 1 is on bottom edge (between corners 0 and 3)
        assert_eq!(mesh.elements_at_vertex(1).len(), 2);
        // Vertex 4 is on left edge (between corners 0 and 12)
        assert_eq!(mesh.elements_at_vertex(4).len(), 2);

        // Interior vertices should have 4 elements
        // Vertex 5 is interior (second row, second column)
        assert_eq!(mesh.elements_at_vertex(5).len(), 4);
        // Vertex 6 is interior
        assert_eq!(mesh.elements_at_vertex(6).len(), 4);
        // Vertex 9 is interior
        assert_eq!(mesh.elements_at_vertex(9).len(), 4);
        // Vertex 10 is interior
        assert_eq!(mesh.elements_at_vertex(10).len(), 4);
    }

    #[test]
    fn test_vertex_to_elements_periodic() {
        // Periodic mesh: all vertices effectively interior
        let mesh = Mesh2D::uniform_periodic(0.0, 1.0, 0.0, 1.0, 3, 3);

        // In a periodic mesh, each vertex should have 4 elements
        // (because periodicity wraps around)
        for v in 0..mesh.n_vertices {
            let patch_size = mesh.elements_at_vertex(v).len();
            // Due to periodic wrapping, some vertices appear at edges of the
            // physical grid but are connected to elements via periodicity.
            // For a structured periodic mesh, each vertex should have 4 elements.
            assert!(
                (1..=4).contains(&patch_size),
                "Vertex {} has {} elements, expected 1-4",
                v,
                patch_size
            );
        }
    }

    #[test]
    fn test_vertex_to_elements_consistency() {
        let mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 2, 2);

        // Check that each element-vertex pair is consistent
        for (k, elem) in mesh.elements.iter().enumerate() {
            for &v in elem {
                let patch = mesh.elements_at_vertex(v);
                assert!(
                    patch.contains(&k),
                    "Element {} contains vertex {} but vertex_to_elements[{}] = {:?}",
                    k,
                    v,
                    v,
                    patch
                );
            }
        }

        // Check reverse: each vertex-element pair should have the vertex in the element
        for (v, elems) in mesh.vertex_to_elements.iter().enumerate() {
            for &k in elems {
                assert!(
                    mesh.elements[k].contains(&v),
                    "vertex_to_elements[{}] contains {} but element {} = {:?}",
                    v,
                    k,
                    k,
                    mesh.elements[k]
                );
            }
        }
    }

    #[test]
    fn test_element_vertex_indices() {
        let mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 2, 2);

        for ki in 0..mesh.n_elements {
            let indices = mesh.element_vertex_indices(k(ki));
            assert_eq!(indices, mesh.elements[ki]);
        }
    }

    // =========================================================================
    // Trait Implementation Tests
    // =========================================================================

    use crate::mesh::traits::{MeshCFL, MeshGeometry, MeshGeometryExt, MeshIter, MeshTopology};

    #[test]
    fn test_mesh2d_topology_trait() {
        let mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 2, 2);

        assert_eq!(MeshTopology::n_elements(&mesh), 4);
        assert_eq!(mesh.n_faces(), mesh.n_edges);
        assert_eq!(mesh.n_boundary_faces(), mesh.n_boundary_edges);

        // Test face connection for corner element (should have 2 boundary faces)
        let mut boundary_count = 0;
        for f in 0..4 {
            if mesh.face_connection(0, f).is_boundary() {
                boundary_count += 1;
            }
        }
        assert_eq!(
            boundary_count, 2,
            "Corner element should have 2 boundary faces"
        );

        // Test interior connectivity
        // Element 0 (bottom-left) face 1 (right) should connect to element 1
        let conn = mesh.face_connection(0, 1);
        assert!(conn.is_interior());
        let neighbor = conn.neighbor().unwrap();
        assert_eq!(neighbor.element, 1);
        assert_eq!(neighbor.face, 3); // Element 1's left face
    }

    #[test]
    fn test_mesh2d_geometry_trait() {
        let mesh = Mesh2D::uniform_rectangle(0.0, 2.0, 0.0, 2.0, 2, 2);

        // Test reference to physical mapping
        // Element 0 is [0,1] x [0,1], center at (0.5, 0.5)
        let [x, y] = MeshGeometry::reference_to_physical(&mesh, 0, [0.0, 0.0]);
        assert!((x - 0.5).abs() < 1e-12);
        assert!((y - 0.5).abs() < 1e-12);

        // Test roundtrip
        let phys = [0.75, 0.25];
        let ref_coord = MeshGeometry::physical_to_reference(&mesh, 0, phys);
        let phys_back = MeshGeometry::reference_to_physical(&mesh, 0, ref_coord);
        assert!((phys[0] - phys_back[0]).abs() < 1e-10);
        assert!((phys[1] - phys_back[1]).abs() < 1e-10);

        // Test h_min/h_max
        assert!((MeshGeometry::h_min(&mesh) - 1.0).abs() < 1e-12);
        assert!(MeshGeometry::h_max(&mesh) > 1.0); // Diagonal is longer

        // Test Jacobian (for 1x1 element, J = 0.25)
        let j = mesh.jacobian_det(0);
        assert!((j - 0.25).abs() < 1e-12);
    }

    #[test]
    fn test_mesh2d_geometry_ext_trait() {
        let mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 1, 1);

        // Check normals point outward and are unit vectors
        for f in 0..4 {
            let [nx, ny] = MeshGeometryExt::face_normal(&mesh, 0, f);
            let len = (nx * nx + ny * ny).sqrt();
            assert!((len - 1.0).abs() < 1e-14, "Normal should be unit vector");
        }

        // Check specific normals for unit square
        let n0 = MeshGeometryExt::face_normal(&mesh, 0, 0); // bottom
        assert!(n0[0].abs() < 1e-14);
        assert!((n0[1] - (-1.0)).abs() < 1e-14); // pointing down

        let n1 = MeshGeometryExt::face_normal(&mesh, 0, 1); // right
        assert!((n1[0] - 1.0).abs() < 1e-14); // pointing right
        assert!(n1[1].abs() < 1e-14);

        let n2 = MeshGeometryExt::face_normal(&mesh, 0, 2); // top
        assert!(n2[0].abs() < 1e-14);
        assert!((n2[1] - 1.0).abs() < 1e-14); // pointing up

        let n3 = MeshGeometryExt::face_normal(&mesh, 0, 3); // left
        assert!((n3[0] - (-1.0)).abs() < 1e-14); // pointing left
        assert!(n3[1].abs() < 1e-14);

        // Check surface Jacobian (for unit square, edge=1, ref_edge=2, so sJ=0.5)
        for f in 0..4 {
            let sj = MeshGeometryExt::surface_jacobian(&mesh, 0, f);
            assert!((sj - 0.5).abs() < 1e-14);
        }
    }

    #[test]
    fn test_mesh2d_cfl_trait() {
        let mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 10, 10);

        let dt = mesh.compute_dt(1.0, 2, 0.5);
        // dt = CFL * h_min / (wave_speed * (2*order + 1))
        let expected = 0.5 * mesh.h_min() / 5.0;
        assert!((dt - expected).abs() < 1e-14);
    }

    #[test]
    fn test_mesh2d_iter_trait() {
        let mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 3, 2);

        let elements: Vec<usize> = mesh.elements().collect();
        assert_eq!(elements.len(), 6);
        assert_eq!(elements, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_mesh2d_geometry_trait_impl() {
        use crate::mesh::traits::Mesh2DGeometry;

        let mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 1, 1);

        let normals = Mesh2DGeometry::face_normals_array(&mesh, 0);
        assert_eq!(normals.len(), 4);

        let sjs = Mesh2DGeometry::surface_jacobians_array(&mesh, 0);
        for sj in sjs {
            assert!((sj - 0.5).abs() < 1e-14);
        }

        let verts = Mesh2DGeometry::element_vertices(&mesh, 0);
        assert_eq!(verts[0], [0.0, 0.0]);
        assert_eq!(verts[1], [1.0, 0.0]);
        assert_eq!(verts[2], [1.0, 1.0]);
        assert_eq!(verts[3], [0.0, 1.0]);
    }
}

//! Bathymetry (bottom topography) storage for 2D shallow water simulations.
//!
//! Bathymetry B(x, y) represents the bed elevation above some reference datum.
//! The water surface elevation is η = h + B, where h is the water depth.
//!
//! For well-balanced schemes, we need both B and its gradients ∂B/∂x and ∂B/∂y.
//!
//! Nodal values come from a function: sampled at the nodes
//! ([`Bathymetry2D::from_function`]), or L2-projected onto each element's
//! polynomial space and limited to the data range ([`Bathymetry2D::project`]),
//! the right choice for raster data finer than the node spacing (e.g.
//! [`crate::io::BedRaster`]).

use faer::Mat;
use faer::linalg::solvers::Solve;

use crate::mesh::{BoundaryTag, Mesh2D};
use crate::operators::{DGOperators2D, GeometricFactors2D};
use crate::polynomial::{gauss_legendre_nodes_weights, legendre, mode_degrees};
use crate::types::ElementIndex;

/// Physical position and Jacobian determinant of the bilinear map of an
/// element with vertices `v` (counter-clockwise) at reference `(r, s)`.
fn bilinear_map(v: &[[f64; 2]; 4], r: f64, s: f64) -> ([f64; 2], f64) {
    let shape = [
        (1.0 - r) * (1.0 - s),
        (1.0 + r) * (1.0 - s),
        (1.0 + r) * (1.0 + s),
        (1.0 - r) * (1.0 + s),
    ];
    let d_r = [-(1.0 - s), 1.0 - s, 1.0 + s, -(1.0 + s)];
    let d_s = [-(1.0 - r), -(1.0 + r), 1.0 + r, 1.0 - r];
    let [mut x, mut y, mut x_r, mut y_r, mut x_s, mut y_s] = [0.0; 6];
    for (c, &[vx, vy]) in v.iter().enumerate() {
        x += shape[c] * vx;
        y += shape[c] * vy;
        x_r += d_r[c] * vx;
        y_r += d_r[c] * vy;
        x_s += d_s[c] * vx;
        y_s += d_s[c] * vy;
    }
    ([0.25 * x, 0.25 * y], 0.0625 * (x_r * y_s - x_s * y_r))
}

/// The representative of every node's set of coincident nodes (on shared
/// faces and vertices), `[k·n_nodes + i]`, from the mesh connectivity:
/// neighbours list their shared face nodes in reverse order, and corner
/// nodes of elements that touch at a vertex only are matched through the
/// vertex.
fn coincident_nodes(mesh: &Mesh2D, ops: &DGOperators2D) -> Vec<usize> {
    let n = ops.n_nodes;
    let n_face = ops.n_face_nodes;
    let mut parent: Vec<usize> = (0..mesh.n_elements * n).collect();
    fn root(parent: &mut [usize], mut a: usize) -> usize {
        while parent[a] != a {
            parent[a] = parent[parent[a]];
            a = parent[a];
        }
        a
    }
    let mut union = |a: usize, b: usize| {
        let (ra, rb) = (root(&mut parent, a), root(&mut parent, b));
        if ra != rb {
            parent[ra.max(rb)] = ra.min(rb);
        }
    };
    for edge in &mesh.edges {
        let Some(right) = edge.right else { continue };
        let left = edge.left;
        for fi in 0..n_face {
            union(
                left.element * n + ops.face_nodes[left.face][fi],
                right.element * n + ops.face_nodes[right.face][n_face - 1 - fi],
            );
        }
    }
    let corner = |r: f64, s: f64| {
        (0..n)
            .find(|&i| ops.nodes_r[i] == r && ops.nodes_s[i] == s)
            .expect("GLL nodes include the corners")
    };
    let corners = [
        corner(-1.0, -1.0),
        corner(1.0, -1.0),
        corner(1.0, 1.0),
        corner(-1.0, 1.0),
    ];
    let mut vertex_node = vec![None; mesh.n_vertices];
    for (k, vertices) in mesh.elements.iter().enumerate() {
        for (c, &v) in vertices.iter().enumerate() {
            let node = k * n + corners[c];
            match vertex_node[v] {
                Some(first) => union(first, node),
                None => vertex_node[v] = Some(node),
            }
        }
    }
    (0..parent.len()).map(|a| root(&mut parent, a)).collect()
}

/// The nodes of a continuous field: every set of coincident nodes is one
/// global node, and neighbours along the elements' grid lines are joined.
struct NodeGraph {
    /// Global node of every `k·n_nodes + i`
    global: Vec<usize>,
    /// Mass `Σ w J` of every global node (over its coincident nodes)
    area: Vec<f64>,
    /// Neighbouring global nodes along the grid lines, each pair once
    pairs: Vec<[usize; 2]>,
}

impl NodeGraph {
    fn new(mesh: &Mesh2D, ops: &DGOperators2D, geom: &GeometricFactors2D) -> Self {
        let group = coincident_nodes(mesh, ops);
        let mut index = vec![usize::MAX; group.len()];
        let mut global = Vec::with_capacity(group.len());
        let mut area = Vec::new();
        for (node, &g) in group.iter().enumerate() {
            if index[g] == usize::MAX {
                index[g] = area.len();
                area.push(0.0);
            }
            global.push(index[g]);
            area[index[g]] += geom.mass[node];
        }
        let (n, n_1d) = (ops.n_nodes, ops.n_1d);
        let mut pairs = Vec::with_capacity(2 * group.len());
        for k in 0..mesh.n_elements {
            for i in 0..n {
                let (a, b) = (i % n_1d, i / n_1d);
                let right = (a + 1 < n_1d).then_some(i + 1);
                let up = (b + 1 < n_1d).then_some(i + n_1d);
                for j in [right, up].into_iter().flatten() {
                    let (p, q) = (global[k * n + i], global[k * n + j]);
                    if p != q {
                        pairs.push([p.min(q), p.max(q)]);
                    }
                }
            }
        }
        // Face lines are shared by two elements
        pairs.sort_unstable();
        pairs.dedup();
        Self {
            global,
            area,
            pairs,
        }
    }

    /// Still-water depth −B of every global node (the mass-weighted mean
    /// over its coincident nodes).
    fn depths(&self, bed: &[f64], geom: &GeometricFactors2D) -> Vec<f64> {
        let mut depth = vec![0.0; self.area.len()];
        for (node, &g) in self.global.iter().enumerate() {
            depth[g] -= geom.mass[node] * bed[node];
        }
        for (d, a) in depth.iter_mut().zip(&self.area) {
            *d /= a;
        }
        depth
    }

    /// Every pair of distinct global nodes of one element, each pair once.
    fn element_pairs(&self, n_elements: usize, n_nodes: usize) -> Vec<[usize; 2]> {
        let mut pairs = Vec::with_capacity(n_elements * n_nodes * (n_nodes - 1) / 2);
        for k in 0..n_elements {
            let nodes = &self.global[k * n_nodes..(k + 1) * n_nodes];
            for (i, &p) in nodes.iter().enumerate() {
                for &q in &nodes[i + 1..] {
                    if p != q {
                        pairs.push([p.min(q), p.max(q)]);
                    }
                }
            }
        }
        pairs.sort_unstable();
        pairs.dedup();
        pairs
    }

    /// Every element's excess over `bound` (m), from the global nodes'
    /// `depth`: its deepest node beside its shallowest, over the nodes at
    /// least `min_depth` deep (zero with fewer than two of them).
    fn element_excess<'a>(
        &'a self,
        n_nodes: usize,
        depth: &'a [f64],
        bound: ElementSlopeBound,
        min_depth: f64,
    ) -> impl Iterator<Item = f64> + 'a {
        self.global.chunks(n_nodes).map(move |nodes| {
            let (shallow, deep) = nodes
                .iter()
                .map(|&g| depth[g])
                .filter(|&d| d >= min_depth)
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(s, d), h| {
                    (s.min(h), d.max(h))
                });
            if shallow.is_finite() {
                bound.excess(shallow, deep)
            } else {
                0.0
            }
        })
    }

    /// A node of every global node (`k·n_nodes + i`).
    fn representatives(&self) -> Vec<usize> {
        let mut node = vec![usize::MAX; self.area.len()];
        for (i, &g) in self.global.iter().enumerate().rev() {
            node[g] = i;
        }
        node
    }
}

/// Slope factor r_x0 = |h₁ − h₂| / (h₁ + h₂) of two neighbouring depths.
fn rx0(h1: f64, h2: f64) -> f64 {
    (h1 - h2).abs() / (h1 + h2)
}

/// The largest slope factor of a bed and where it is, from
/// [`Bathymetry2D::max_rx0`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rx0 {
    /// r_x0 = |h₁ − h₂| / (h₁ + h₂)
    pub value: f64,
    /// The pair's nodes (`k·n_nodes + i`), the shallower first
    pub nodes: [usize; 2],
}

/// What [`Bathymetry2D::smooth_rx0`] did.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rx0Smoothing {
    /// Largest slope factor before and after
    pub before: f64,
    pub after: f64,
    /// Global nodes (sets of coincident nodes) whose bed changed
    pub changed: usize,
    /// Largest change of the bed (m)
    pub max_change: f64,
    /// Sweeps over the constrained pairs
    pub sweeps: usize,
}

/// What [`Bathymetry2D::raise_cliff_shores`] did.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CliffShores {
    /// Cliff elements with a shore node raised
    pub elements: usize,
    /// Global nodes (sets of coincident nodes) raised to land
    pub nodes: usize,
    /// Largest rise of the bed (m)
    pub max_raise: f64,
    /// Area wet at mean sea level that became land, `Σ w J` (m²)
    pub area: f64,
}

/// What [`Bathymetry2D::fill_tidal_ponds`] did.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TidalPonds {
    /// Ponds filled (connected sets of pond nodes)
    pub ponds: usize,
    /// Global nodes (sets of coincident nodes) raised to their sill
    pub nodes: usize,
    /// Largest rise of the bed (m)
    pub max_raise: f64,
    /// Area of the raised nodes, `Σ w J` (m²)
    pub area: f64,
    /// Volume the beds rose by, `Σ w J ΔB` (m³)
    pub volume: f64,
    /// Ponds of more than `max_nodes` nodes, left alone
    pub left: usize,
}

/// What [`Bathymetry2D::lower_wall_land`] did.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WallLandLowering {
    /// Elements whose wall nodes were lowered
    pub elements: usize,
    /// Global nodes (sets of coincident nodes) whose bed changed
    pub changed: usize,
    /// Largest change of the bed (m)
    pub max_change: f64,
    /// Still water added below `level`, `Σ w J ΔB` over the changed nodes (m³)
    pub volume: f64,
}

/// How much the still-water depth may change within one element for the 3D
/// model's stratified rest state to be stable
/// ([`Bathymetry2D::smooth_element_slopes`]).
///
/// Within an element the deepest node may be at most `max(free_depth,
/// R·h_s)` deep beside its shallowest node `h_s`, with
/// `R = (1 + r_max)/(1 − r_max)`: a slope factor r_x0 ≤ `r_max` between
/// any two nodes of the element, except that water shallower than
/// `free_depth` is never constrained.
///
/// # Why
///
/// The σ-pairs pressure gradient and the split-form advection keep a
/// stratified fluid at rest stable while no σ-level crosses the pycnocline
/// between two nodes of one element (TODO P1.3; the analysis is in
/// `docs/stratified-rest-over-steep-beds.md`). Calibrated on x–z slices
/// (1 km P2 elements, 20 levels, the bed stepping inside one element) under a
/// 2 kg/m³ pycnocline between 11 and 19 m: steps from 60, 100 and 150 m
/// hold at a depth ratio of 1.5 (r_x0 = 0.2) and grow at 1.67–2.3
/// (e-folding 27 h down to 3 h); from 30 m, 45 m holds and 55 m grows
/// (10 h). In shallow water the depth ratio does not matter: steps from
/// 0.5–5 m of water to 21–40 m hold, to 60 m grow (2 h). The slices allowed a
/// free depth of 1.5 times the pycnocline's bottom, but in 2D shallow–deep
/// pairs below it grew (the Frøya coastline mesh: 2.6 m beside 28.5 m inside
/// one element, e-folding 43 min), so [`Self::for_pycnocline`] takes the
/// pycnocline's bottom itself. Steps to land need no bound: a thin
/// column exchanges no baroclinic transport, and its element moves the
/// layers on the 2D wet/dry kernel's subcells (land beside 300 m holds at
/// 1e-10 m/s for a day once the wet pairs are smoothed; see
/// [`Bathymetry2D::smooth_element_slopes`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ElementSlopeBound {
    /// Largest slope factor r_x0 between two nodes of one element whose
    /// deeper node is deeper than `free_depth` (ROMS practice is 0.2 between
    /// neighbours; 0.15 within an element, see [`Self::for_pycnocline`]).
    pub r_max: f64,
    /// Depth (m, positive) above which the element's depth range is free.
    pub free_depth: f64,
}

impl ElementSlopeBound {
    /// The calibrated bound for a pycnocline whose bottom is `bottom` m
    /// deep: r_x0 ≤ 0.15 and a free depth of `bottom`.
    ///
    /// The x–z slices hold at r_x0 0.2, but on the Frøya coastline mesh an
    /// element at that bound (33.6 m nodes and one 50.4 m corner) grew with
    /// an e-folding of 36 min; at 0.15 a patch around it holds round-off for
    /// 12 h. With a free depth of 1.5 × `bottom` (28.5 m), shallow–deep pairs
    /// below it grew (e-folding 43 min). At `bottom` the full coastline mesh
    /// at rest holds 8.4e-9 m/s for 11.7 h with constant mixing, with the
    /// 3D model's vertical reference (`Hydrostatic3D::with_vertical_reference`),
    /// and levels off at 2.8e-3 m/s with GLS (TODO P1.3). The 2D tide at
    /// Mausund on that bed: centred RMSE 2.8 cm against the gauge's
    /// prediction, 4.0 cm against the observations.
    pub fn for_pycnocline(bottom: f64) -> Self {
        Self {
            r_max: 0.15,
            free_depth: bottom,
        }
    }

    /// Depth ratio `R = (1 + r_max)/(1 − r_max)` of the slope factor bound.
    fn ratio(&self) -> f64 {
        (1.0 + self.r_max) / (1.0 - self.r_max)
    }

    /// The deepest depth allowed in an element whose shallowest node is
    /// `shallow` m deep (zero or less on land).
    pub fn deepest_beside(&self, shallow: f64) -> f64 {
        self.free_depth.max(self.ratio() * shallow)
    }

    /// The excess (m) up to which an element counts as within the bound:
    /// round-off of [`Bathymetry2D::smooth_element_slopes`], which smooths
    /// to within it.
    pub fn tolerance(&self) -> f64 {
        1e-6 * self.free_depth
    }

    /// How much deeper than allowed `deep` is beside `shallow` (m, zero
    /// when within the bound).
    pub fn excess(&self, shallow: f64, deep: f64) -> f64 {
        (deep - self.deepest_beside(shallow)).max(0.0)
    }
}

/// What [`Bathymetry2D::smooth_element_slopes`] did.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ElementSlopeSmoothing {
    /// Elements over the bound before and after
    pub elements_before: usize,
    pub elements_after: usize,
    /// Largest excess depth before and after (m)
    pub excess_before: f64,
    pub excess_after: f64,
    /// Global nodes (sets of coincident nodes) whose bed changed
    pub changed: usize,
    /// Largest change of the bed (m)
    pub max_change: f64,
    /// Sweeps over the element pairs
    pub sweeps: usize,
}

/// A 1D quadrature rule on [−1, 1] with the orthonormal Legendre values of
/// every degree ≤ N at its points (`legendre[point × (N + 1) + degree]`), for
/// [`Bathymetry2D::project`].
#[derive(Clone)]
struct ProjectionRule {
    points: Vec<f64>,
    weights: Vec<f64>,
    legendre: Vec<f64>,
}

impl ProjectionRule {
    /// Call `visit(x, y, w·J, φ)` at every point of the tensor rule `self`
    /// (in r) × `s_rule` (in s) on the element with vertices `verts`, where
    /// `φ[m]` is the orthonormal mode m (numbered by [`mode_degrees`]) there.
    fn for_each_point(
        &self,
        s_rule: &Self,
        verts: &[[f64; 2]; 4],
        phi: &mut [f64],
        mut visit: impl FnMut(f64, f64, f64, &[f64]),
    ) {
        let n_1d = self.legendre.len() / self.points.len();
        for (b, &s) in s_rule.points.iter().enumerate() {
            let l_s = &s_rule.legendre[b * n_1d..][..n_1d];
            for (a, &r) in self.points.iter().enumerate() {
                let l_r = &self.legendre[a * n_1d..][..n_1d];
                for (m, value) in phi.iter_mut().enumerate() {
                    let (i, j) = mode_degrees(m, n_1d);
                    *value = l_r[i] * l_s[j];
                }
                let ([x, y], jac) = bilinear_map(verts, r, s);
                visit(x, y, self.weights[a] * s_rule.weights[b] * jac, phi);
            }
        }
    }
}

/// Bathymetry data for 2D shallow water simulations.
///
/// Stores nodal bathymetry values and their spatial gradients.
/// Layout: `data[k * n_nodes + i]` for element k, node i.
#[derive(Clone)]
pub struct Bathymetry2D {
    /// Nodal bathymetry values B(x, y)
    pub data: Vec<f64>,
    /// Pre-computed x-gradient ∂B/∂x at each node
    pub gradient_x: Vec<f64>,
    /// Pre-computed y-gradient ∂B/∂y at each node
    pub gradient_y: Vec<f64>,
    /// Number of elements
    pub n_elements: usize,
    /// Number of nodes per element
    pub n_nodes: usize,
}

impl Bathymetry2D {
    /// Create flat bathymetry (B = 0 everywhere).
    pub fn flat(n_elements: usize, n_nodes: usize) -> Self {
        Self {
            data: vec![0.0; n_elements * n_nodes],
            gradient_x: vec![0.0; n_elements * n_nodes],
            gradient_y: vec![0.0; n_elements * n_nodes],
            n_elements,
            n_nodes,
        }
    }

    /// Create constant bathymetry (B = constant everywhere).
    pub fn constant(n_elements: usize, n_nodes: usize, value: f64) -> Self {
        Self {
            data: vec![value; n_elements * n_nodes],
            gradient_x: vec![0.0; n_elements * n_nodes], // Zero gradient for constant
            gradient_y: vec![0.0; n_elements * n_nodes],
            n_elements,
            n_nodes,
        }
    }

    /// Initialize bathymetry from a function B(x, y).
    ///
    /// Evaluates B(x, y) at each physical node location and computes gradients.
    pub fn from_function<F>(
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        geom: &GeometricFactors2D,
        f: F,
    ) -> Self
    where
        F: Fn(f64, f64) -> f64,
    {
        let n_elements = mesh.n_elements;
        let n_nodes = ops.n_nodes;
        let mut bathy = Self::flat(n_elements, n_nodes);

        // Set bathymetry values at each node
        for k in ElementIndex::iter(n_elements) {
            for i in 0..n_nodes {
                let r = ops.nodes_r[i];
                let s = ops.nodes_s[i];
                let [x, y] = mesh.reference_to_physical(k, r, s);
                bathy.set(k, i, f(x, y));
            }
        }

        // Compute gradients
        bathy.compute_gradients(ops, geom);

        bathy
    }

    /// L2 projection of the bed elevation `f(x, y)` onto each element's
    /// polynomial space, limited to the range of `f` over the element.
    ///
    /// Point sampling ([`Self::from_function`]) aliases data that varies on
    /// scales below the node spacing, such as a bathymetry raster whose pixels
    /// are finer than a coarse element. The projection instead finds, per
    /// element k, the B ∈ Q_N minimising ∫_k (B − f)² dx:
    ///
    /// ```text
    /// Σ_n (∫_k φ_m φ_n J) c_n = ∫_k φ_m f J,   B = V c
    /// ```
    ///
    /// with the orthonormal Legendre modes φ_m. The right side is integrated
    /// by composite Gauss–Legendre quadrature on sub-cells no longer than
    /// `resolution / 2` (N + 2 points per sub-cell and direction, at most
    /// [`Self::MAX_PROJECTION_SUBCELLS`] sub-cells per direction), where
    /// `resolution` is the length (m) on which `f` varies, the pixel size for
    /// a raster.
    ///
    /// Properties:
    /// - Exact for `f` in Q_N (e.g. linear fields on any quadrilateral).
    /// - Volume-preserving: `Σ_i w_i J_i B_i = ∫_k f` to quadrature accuracy
    ///   (for N ≥ 2, or on parallelograms), so the still-water volume is the
    ///   data's.
    /// - Bounded: an unresolved step (a coastline cliff) would overshoot like
    ///   any L2 projection, so the nodal values are scaled towards the element
    ///   mean until they lie within the minimum and maximum of `f` over the
    ///   quadrature points and nodes (Zhang & Shu 2010). The scaling keeps the
    ///   mean, and changes smooth data only at the level of the projection
    ///   error, so the projection still converges at N + 1.
    /// - Continuous: the element projections are independent, so a vertex on
    ///   an unresolved coastline could get −18 m in the water element and
    ///   +5 m in its land neighbours, leaving a lone deep node in a pocket
    ///   (it jetted at 8 m/s in the Frøya run). Coincident nodes are
    ///   therefore averaged ([`Self::make_continuous`]), which keeps the
    ///   bounds, the total volume and the exactness for Q_N.
    pub fn project<F>(
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        geom: &GeometricFactors2D,
        f: F,
        resolution: f64,
    ) -> Self
    where
        F: Fn(f64, f64) -> f64,
    {
        assert!(
            resolution > 0.0 && resolution.is_finite(),
            "projection resolution must be positive, got {resolution}"
        );
        let n_1d = ops.n_1d;
        let n = ops.n_nodes;
        let (gauss_x, gauss_w) = gauss_legendre_nodes_weights(ops.order + 2);
        // Orthonormal Legendre values of every degree at `points`:
        // `[point × degree]`
        let legendre_table = |points: &[f64]| -> Vec<f64> {
            points
                .iter()
                .flat_map(|&x| {
                    (0..n_1d).map(move |i| ((2 * i + 1) as f64 / 2.0).sqrt() * legendre(i, x))
                })
                .collect()
        };
        // Composite rule on `m` equal sub-intervals of [−1, 1], with the
        // Legendre table at its points
        let composite = |m: usize| -> ProjectionRule {
            let half = 1.0 / m as f64;
            let (points, weights): (Vec<f64>, Vec<f64>) = (0..m)
                .flat_map(|c| {
                    let centre = -1.0 + (2 * c + 1) as f64 * half;
                    gauss_x
                        .iter()
                        .zip(&gauss_w)
                        .map(move |(&x, &w)| (centre + half * x, half * w))
                })
                .unzip();
            let legendre = legendre_table(&points);
            ProjectionRule {
                points,
                weights,
                legendre,
            }
        };
        // The mass matrix ∫ φ_m φ_n J: a polynomial of degree 2N + 1 per
        // direction (J is bilinear), exact with the N + 2 point rule
        let mass_rule = composite(1);
        let mut rules: Vec<Option<ProjectionRule>> = vec![None; Self::MAX_PROJECTION_SUBCELLS + 1];

        let mut bathy = Self::flat(mesh.n_elements, n);
        let mut mass = Mat::<f64>::zeros(n, n);
        let mut rhs = Mat::<f64>::zeros(n, 1);
        let mut phi = vec![0.0; n];
        for k in ElementIndex::iter(mesh.n_elements) {
            let verts = mesh.element_vertices(k);
            let dist = |a: usize, b: usize| {
                let [xa, ya] = verts[a];
                let [xb, yb] = verts[b];
                ((xb - xa).powi(2) + (yb - ya).powi(2)).sqrt()
            };
            let subcells = |length: f64| {
                ((2.0 * length / resolution).ceil() as usize)
                    .clamp(1, Self::MAX_PROJECTION_SUBCELLS)
            };
            let m_r = subcells(dist(0, 1).max(dist(3, 2)));
            let m_s = subcells(dist(0, 3).max(dist(1, 2)));
            for m in [m_r, m_s] {
                if rules[m].is_none() {
                    rules[m] = Some(composite(m));
                }
            }
            let (rule_r, rule_s) = (rules[m_r].as_ref().unwrap(), rules[m_s].as_ref().unwrap());

            mass.fill(0.0);
            mass_rule.for_each_point(&mass_rule, &verts, &mut phi, |_, _, w, phi| {
                for row in 0..n {
                    for col in 0..n {
                        mass[(row, col)] += w * phi[row] * phi[col];
                    }
                }
            });

            rhs.fill(0.0);
            let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
            rule_r.for_each_point(rule_s, &verts, &mut phi, |x, y, w, phi| {
                let value = f(x, y);
                lo = lo.min(value);
                hi = hi.max(value);
                for row in 0..n {
                    rhs[(row, 0)] += w * value * phi[row];
                }
            });
            let modal = mass.full_piv_lu().solve(&rhs);

            let ki = k.as_usize();
            let nodal = &mut bathy.data[ki * n..][..n];
            for (i, b) in nodal.iter_mut().enumerate() {
                *b = (0..n)
                    .map(|m| ops.vandermonde.v[(i, m)] * modal[(m, 0)])
                    .sum();
                let [x, y] = mesh.reference_to_physical(k, ops.nodes_r[i], ops.nodes_s[i]);
                let value = f(x, y);
                lo = lo.min(value);
                hi = hi.max(value);
            }

            // Scale towards the mean into [lo, hi]
            let mean = geom.element_mean(ki, nodal);
            let (b_min, b_max) = nodal
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| {
                    (a.min(v), b.max(v))
                });
            let mut theta: f64 = 1.0;
            if b_max > hi {
                theta = theta.min((hi - mean).max(0.0) / (b_max - mean));
            }
            if b_min < lo {
                theta = theta.min((mean - lo).max(0.0) / (mean - b_min));
            }
            if theta < 1.0 {
                for b in nodal.iter_mut() {
                    *b = mean + theta * (*b - mean);
                }
            }
        }

        bathy.make_continuous(mesh, ops, geom);
        bathy
    }

    /// Make B continuous across elements: every set of coincident nodes
    /// (on shared faces and vertices) gets the mean of its values weighted by
    /// the nodes' mass w·J, as in the direct stiffness summation of spectral
    /// elements; then recompute the gradients.
    ///
    /// The total `Σ w J B` is unchanged and every new value is a convex
    /// combination of old ones, so bounds hold; a continuous B is unchanged.
    /// Nodes are matched by the mesh connectivity (neighbours list their
    /// shared face nodes in reverse order), not by position.
    pub fn make_continuous(
        &mut self,
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        geom: &GeometricFactors2D,
    ) {
        let group = coincident_nodes(mesh, ops);
        let mut mass = vec![0.0; self.data.len()];
        let mut moment = vec![0.0; self.data.len()];
        for node in 0..self.data.len() {
            mass[group[node]] += geom.mass[node];
            moment[group[node]] += geom.mass[node] * self.data[node];
        }
        for node in 0..self.data.len() {
            self.data[node] = moment[group[node]] / mass[group[node]];
        }
        self.compute_gradients(ops, geom);
    }

    /// Raise isolated wet nodes to dry shore, and return how many (sets of
    /// coincident nodes) were raised.
    ///
    /// A node below `level` whose neighbours along the element's grid lines,
    /// in every element that shares it, are all at or above `level` is water
    /// the mesh cannot resolve: a narrow sound or a lake one node wide. In
    /// `WetDry` such a lone wet node against dry nodes carries η jumps of up
    /// to 1 m between the elements that share it and spurious currents of
    /// 1–3 m/s through the tide (Frøya at 1 km). It is raised to the lowest
    /// of its neighbours, repeatedly until none is left, and the gradients
    /// are recomputed.
    pub fn raise_isolated_wet_nodes(
        &mut self,
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        geom: &GeometricFactors2D,
        level: f64,
    ) -> usize {
        let n = self.n_nodes;
        let n_1d = ops.n_1d;
        let group = coincident_nodes(mesh, ops);
        let mut members: Vec<Vec<usize>> = vec![Vec::new(); self.data.len()];
        for (node, &g) in group.iter().enumerate() {
            members[g].push(node);
        }
        // Neighbours along the grid lines of a node within its element
        let grid_neighbours = |node: usize| {
            let (k, i) = (node / n, node % n);
            let (a, b) = (i % n_1d, i / n_1d);
            [
                (a > 0).then(|| i - 1),
                (a + 1 < n_1d).then(|| i + 1),
                (b > 0).then(|| i - n_1d),
                (b + 1 < n_1d).then(|| i + n_1d),
            ]
            .into_iter()
            .flatten()
            .map(move |j| k * n + j)
        };
        let mut raised = 0;
        loop {
            let mut changed = false;
            for nodes in &members {
                if nodes.is_empty() || nodes.iter().all(|&m| self.data[m] >= level) {
                    continue;
                }
                let lowest = nodes
                    .iter()
                    .flat_map(|&m| grid_neighbours(m))
                    .map(|j| self.data[j])
                    .fold(f64::INFINITY, f64::min);
                if lowest >= level && lowest.is_finite() {
                    for &m in nodes {
                        self.data[m] = lowest;
                    }
                    raised += 1;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        if raised > 0 {
            self.compute_gradients(ops, geom);
        }
        raised
    }

    /// Lower the land at the walls of elements that are otherwise water: in
    /// every element with a [`BoundaryTag::Wall`] face whose nodes off its
    /// walls are all below `level`, a wall node at or above `level` takes
    /// the bed of its inward neighbour across the wall (or, where that is a
    /// wall node too, the element's shallowest node off its walls).
    /// Coincident nodes move together, to the shallowest such value over the
    /// elements that lower them; the gradients are recomputed.
    ///
    /// A coastline-fitted mesh puts its walls on the coast, but the coast it
    /// fits is simplified to the element size, so the bed at the walls is
    /// the elevation model's on either side of the real coastline: on the
    /// Frøya mesh (`data/froya_coast.msh`, ≈ 100 m faces) two thirds of the
    /// wall nodes are land, by 2 m in the median. An element of water whose
    /// wall nodes stand on such land is partly dry at every water level
    /// (1016 of its 12,101 elements at mean sea level): it carries the
    /// `WetDry` subcells, the steep dry–wet pair that sets the shoreline
    /// time step, and no baroclinic pressure in 3D, for a strip of land a
    /// fraction of a node spacing wide that the mesh puts outside its coast
    /// anyway. Lowered, the wall stands in the water in front of it.
    ///
    /// Take `level` below the lowest water: an element that dries off its
    /// walls at low tide gains nothing, and lowering its wall only widens a
    /// flat (at 0 m on the Mausund mesh, +3–4 m wall nodes of a sound
    /// 0.2–3 m deep went to −0.5 m, and its current rose from 2.4 to
    /// 3.0 m/s). Elements with nodes off their walls at or above `level` are
    /// shores the mesh resolves and are left alone: a node shared with one
    /// stays unless that element is below `level` everywhere else, so a
    /// lowered corner never stands as a lone deep node in a shore. The
    /// lowered nodes only join water their element already connects, so no
    /// sound opens.
    pub fn lower_wall_land(
        &mut self,
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        geom: &GeometricFactors2D,
        level: f64,
    ) -> WallLandLowering {
        let (n, n_1d) = (self.n_nodes, ops.n_1d);
        let mut on_wall = vec![false; self.data.len()];
        let mut wall_faces: Vec<Vec<usize>> = vec![Vec::new(); mesh.n_elements];
        for k in ElementIndex::iter(mesh.n_elements) {
            for face in 0..4 {
                if mesh.boundary_tag(k, face) == Some(BoundaryTag::Wall) {
                    wall_faces[k.as_usize()].push(face);
                    for &i in &ops.face_nodes[face] {
                        on_wall[k.as_usize() * n + i] = true;
                    }
                }
            }
        }
        // One step into the element from a node of each face
        let inward = [n_1d as isize, -1, -(n_1d as isize), 1];
        let group = coincident_nodes(mesh, ops);
        let mut target = vec![f64::NEG_INFINITY; self.data.len()];
        let mut qualifies = vec![false; mesh.n_elements];
        for (k, faces) in wall_faces.iter().enumerate() {
            let bed = &self.data[k * n..][..n];
            let shallowest = (0..n)
                .filter(|&i| !on_wall[k * n + i])
                .map(|i| bed[i])
                .fold(f64::NEG_INFINITY, f64::max);
            let land = faces
                .iter()
                .any(|&f| ops.face_nodes[f].iter().any(|&i| bed[i] >= level));
            if !(land && shallowest.is_finite() && shallowest < level) {
                continue;
            }
            qualifies[k] = true;
            for &f in faces {
                for &i in &ops.face_nodes[f] {
                    if bed[i] < level {
                        continue;
                    }
                    let j = i.checked_add_signed(inward[f]).expect("inward node");
                    let value = if on_wall[k * n + j] {
                        shallowest
                    } else {
                        bed[j]
                    };
                    let g = group[k * n + i];
                    target[g] = target[g].max(value);
                }
            }
        }
        // A node shared with a shore the mesh resolves stays
        let mut accepted = vec![true; self.data.len()];
        for node in 0..self.data.len() {
            let (g, k) = (group[node], node / n);
            if target[g].is_finite() && !qualifies[k] {
                let dry = (0..n).any(|i| group[k * n + i] != g && self.data[k * n + i] >= level);
                if dry {
                    accepted[g] = false;
                }
            }
        }
        let mut report = WallLandLowering::default();
        let mut changed = vec![false; self.data.len()];
        let mut lowered = vec![false; mesh.n_elements];
        for node in 0..self.data.len() {
            let g = group[node];
            let (old, new) = (self.data[node], target[g]);
            if new.is_finite() && accepted[g] && old > new {
                report.max_change = report.max_change.max(old - new);
                report.volume += geom.mass[node] * ((-new).max(0.0) - (-old).max(0.0));
                self.data[node] = new;
                lowered[node / n] |= qualifies[node / n];
                if !changed[g] {
                    changed[g] = true;
                    report.changed += 1;
                }
            }
        }
        report.elements = lowered.iter().filter(|&&l| l).count();
        if report.changed > 0 {
            self.compute_gradients(ops, geom);
        }
        report
    }

    /// The largest slope factor r_x0 = |h₁ − h₂| / (h₁ + h₂) between
    /// neighbouring nodes along the elements' grid lines, over the pairs
    /// whose still-water depths h = −B are both at least `min_depth`, or
    /// `None` if there is no such pair.
    ///
    /// r_x0 (Beckmann & Haidvogel 1993) is the relative depth change between
    /// neighbours. It bounds the hydrostatic inconsistency of the σ-coordinate
    /// pressure gradient (ROMS models are smoothed to r_x0 ≲ 0.2), and in 2D
    /// it finds shoals and pits a node wide, over which the depth-averaged
    /// velocity hu/h of a smooth transport spikes. Coincident nodes count
    /// once, with their mass-weighted mean depth.
    pub fn max_rx0(
        &self,
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        geom: &GeometricFactors2D,
        min_depth: f64,
    ) -> Option<Rx0> {
        let graph = NodeGraph::new(mesh, ops, geom);
        let depth = graph.depths(&self.data, geom);
        let node = graph.representatives();
        graph
            .pairs
            .iter()
            .filter(|&&[a, b]| depth[a] >= min_depth && depth[b] >= min_depth)
            .map(|&[a, b]| {
                let [s, d] = if depth[a] <= depth[b] { [a, b] } else { [b, a] };
                Rx0 {
                    value: rx0(depth[a], depth[b]),
                    nodes: [node[s], node[d]],
                }
            })
            .max_by(|x, y| x.value.total_cmp(&y.value))
    }

    /// Largest number of sweeps of [`Self::smooth_rx0`].
    pub const MAX_RX0_SWEEPS: usize = 10_000;

    /// Smooth the bed to a slope factor r_x0 ≤ `r_max` between neighbouring
    /// nodes whose depths h = −B are both at least `min_depth`, keeping the
    /// volume `Σ w J B`, and recompute the gradients.
    ///
    /// This is the volume-preserving "PlusMinus" smoothing of Sikirić,
    /// Janeković & Kuzmić (2009, Ocean Modelling 29): every pair over the
    /// bound, deeper h_d and shallower h_s with node masses A_d and A_s, is
    /// brought to r_x0 = `r_max` by moving the volume
    ///
    /// ```text
    /// V = (h_d (1 − r) − h_s (1 + r)) / ((1 − r)/A_d + (1 + r)/A_s)
    /// ```
    ///
    /// from the shallow node's water column to the deep one's (h_d −= V/A_d,
    /// h_s += V/A_s), sweeping over the pairs until the largest r_x0 is
    /// within 1e-6 of `r_max` (at most [`Self::MAX_RX0_SWEEPS`] sweeps; the
    /// result reports what is left). Only local extremes and steep steps
    /// change, by as little as the bound needs.
    ///
    /// Nodes shallower than `min_depth`, land and the shore, are left alone,
    /// and no node crosses it (the shallower node of a pair only deepens, the
    /// deeper one stays deeper), so the coastline and the dry area do not
    /// move. Coincident nodes move together, so a continuous bed stays
    /// continuous; a discontinuous one keeps its jumps.
    pub fn smooth_rx0(
        &mut self,
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        geom: &GeometricFactors2D,
        r_max: f64,
        min_depth: f64,
    ) -> Rx0Smoothing {
        assert!(
            r_max > 0.0 && r_max < 1.0,
            "r_max must lie in (0, 1), got {r_max}"
        );
        assert!(
            min_depth > 0.0,
            "min_depth must be positive, got {min_depth}"
        );
        let graph = NodeGraph::new(mesh, ops, geom);
        let original = graph.depths(&self.data, geom);
        let mut depth = original.clone();
        let pairs: Vec<[usize; 2]> = graph
            .pairs
            .iter()
            .copied()
            .filter(|&[a, b]| depth[a] >= min_depth && depth[b] >= min_depth)
            .collect();
        let largest = |depth: &[f64]| {
            pairs
                .iter()
                .map(|&[a, b]| rx0(depth[a], depth[b]))
                .fold(0.0, f64::max)
        };
        let before = largest(&depth);
        let area = &graph.area;
        let tolerance = r_max * (1.0 + 1e-6);
        let mut sweeps = 0;
        while sweeps < Self::MAX_RX0_SWEEPS && largest(&depth) > tolerance {
            for &[a, b] in &pairs {
                let (d, s) = if depth[a] >= depth[b] { (a, b) } else { (b, a) };
                let excess = depth[d] * (1.0 - r_max) - depth[s] * (1.0 + r_max);
                if excess > 0.0 {
                    let volume = excess / ((1.0 - r_max) / area[d] + (1.0 + r_max) / area[s]);
                    depth[d] -= volume / area[d];
                    depth[s] += volume / area[s];
                }
            }
            sweeps += 1;
        }

        let mut changed = 0;
        let mut max_change: f64 = 0.0;
        for (&new, &old) in depth.iter().zip(&original) {
            if new != old {
                changed += 1;
                max_change = max_change.max((new - old).abs());
            }
        }
        if changed > 0 {
            for (b, &g) in self.data.iter_mut().zip(&graph.global) {
                *b -= depth[g] - original[g];
            }
            self.compute_gradients(ops, geom);
        }
        Rx0Smoothing {
            before,
            after: largest(&depth),
            changed,
            max_change,
            sweeps,
        }
    }

    /// How much deeper than `bound` allows each element's deepest node is
    /// beside its shallowest (m, zero within the bound), over the nodes at
    /// least `min_depth` deep, from the still-water depths h = −B (coincident
    /// nodes count with their mass-weighted mean, as
    /// [`Self::smooth_element_slopes`] sees them).
    pub fn element_slope_excess(
        &self,
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        geom: &GeometricFactors2D,
        bound: ElementSlopeBound,
        min_depth: f64,
    ) -> Vec<f64> {
        let graph = NodeGraph::new(mesh, ops, geom);
        let depth = graph.depths(&self.data, geom);
        graph
            .element_excess(ops.n_nodes, &depth, bound, min_depth)
            .collect()
    }

    /// Smooth the bed until every element is within `bound`
    /// ([`ElementSlopeBound`]: the deepest node at most `max(free_depth,
    /// R·h_s)` beside the shallowest `h_s`) over its nodes at least
    /// `min_depth` deep, keeping the volume `Σ w J B`, and recompute the
    /// gradients.
    ///
    /// For the 3D model: a σ-level that crosses the pycnocline between two
    /// nodes of one element makes the stratified rest state unstable (TODO
    /// P1.3). Every pair of nodes of an element over the bound, deeper `h_d`
    /// and shallower `h_s` with node masses `A_d`, `A_s`, moves the least
    /// volume from the shallow node's water column to the deep one's that
    /// brings it within the bound (as [`Self::smooth_rx0`]'s "PlusMinus"):
    /// the smaller of `(h_d − free_depth) A_d` (the deep node at the free
    /// depth) and `(h_d − R h_s)/(1/A_d + R/A_s)` (the pair at the ratio).
    /// Sweeps run until the largest excess is within 1e-6 of the free depth
    /// (at most [`Self::MAX_RX0_SWEEPS`]).
    ///
    /// Pairs with a node shallower than `min_depth` are left alone, and those
    /// nodes never move, so the coastline stays. For the 3D model take its
    /// thin-column depth (`Hydrostatic3D::min_column_depth`): a thin column
    /// exchanges no baroclinic transport, and an element with one moves its
    /// layers on the 2D wet/dry kernel's subcells, which keep a stratified
    /// shore at rest without bounding its slope. Coincident nodes move
    /// together; no node crosses `min_depth` (the shallower node of a pair
    /// only deepens, the deeper one stays below the free depth).
    ///
    /// # Panics
    /// If `bound.r_max` is not in (0, 1), or `min_depth` is not positive and
    /// below `bound.free_depth`.
    pub fn smooth_element_slopes(
        &mut self,
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        geom: &GeometricFactors2D,
        bound: ElementSlopeBound,
        min_depth: f64,
    ) -> ElementSlopeSmoothing {
        assert!(
            bound.r_max > 0.0 && bound.r_max < 1.0,
            "r_max must lie in (0, 1), got {}",
            bound.r_max
        );
        assert!(
            min_depth > 0.0 && min_depth < bound.free_depth,
            "min_depth must lie in (0, free_depth = {}), got {min_depth}",
            bound.free_depth
        );
        let graph = NodeGraph::new(mesh, ops, geom);
        let original = graph.depths(&self.data, geom);
        let mut depth = original.clone();
        let pairs: Vec<[usize; 2]> = graph
            .element_pairs(mesh.n_elements, ops.n_nodes)
            .into_iter()
            .filter(|&[a, b]| depth[a] >= min_depth && depth[b] >= min_depth)
            .collect();
        let ordered = |depth: &[f64], [a, b]: [usize; 2]| {
            if depth[a] >= depth[b] { (a, b) } else { (b, a) }
        };
        let tolerance = bound.tolerance();
        // Elements over the bound (beyond round-off) and the largest excess
        let survey = |depth: &[f64]| {
            graph
                .element_excess(ops.n_nodes, depth, bound, min_depth)
                .fold((0, 0.0_f64), |(n, largest), e| {
                    (n + usize::from(e > tolerance), largest.max(e))
                })
        };
        let (elements_before, excess_before) = survey(&depth);
        let area = &graph.area;
        let ratio = bound.ratio();
        let mut sweeps = 0;
        while sweeps < Self::MAX_RX0_SWEEPS && survey(&depth).1 > tolerance {
            for &pair in &pairs {
                let (d, s) = ordered(&depth, pair);
                if bound.excess(depth[s], depth[d]) <= tolerance {
                    continue;
                }
                let to_free_depth = (depth[d] - bound.free_depth) * area[d];
                let to_ratio = (depth[d] - ratio * depth[s]) / (1.0 / area[d] + ratio / area[s]);
                let volume = to_free_depth.min(to_ratio);
                depth[d] -= volume / area[d];
                depth[s] += volume / area[s];
            }
            sweeps += 1;
        }
        let (elements_after, excess_after) = survey(&depth);

        let mut changed = 0;
        let mut max_change: f64 = 0.0;
        for (&new, &old) in depth.iter().zip(&original) {
            if new != old {
                changed += 1;
                max_change = max_change.max((new - old).abs());
            }
        }
        if changed > 0 {
            for (b, &g) in self.data.iter_mut().zip(&graph.global) {
                *b -= depth[g] - original[g];
            }
            self.compute_gradients(ops, geom);
        }
        ElementSlopeSmoothing {
            elements_before,
            elements_after,
            excess_before,
            excess_after,
            changed,
            max_change,
            sweeps,
        }
    }

    /// Raise the shore nodes of cliff elements to land at `land`, and
    /// recompute the gradients.
    ///
    /// A cliff element has a node deeper than `deep` (m). Its shore nodes are
    /// those [`Self::smooth_element_slopes`] leaves alone, shallower than
    /// `min_depth` at mean sea level, that a tide up to `high_water` (m above
    /// MSL) would make a resolved column at least `min_depth` deep. Such a
    /// node is a thin column at low water and a resolved one at high water,
    /// beside the deep nodes and outside any slope bound. On the Frøya
    /// coastline mesh, a corner 0.74 m deep among 61–82 m (element 5179)
    /// held the 3D step below 0.4 s for hours of each high water, with the
    /// deep nodes' surface layers at 24–30 m/s against a depth mean of
    /// 0.03 m/s (TODO P1.3). Land beside deep water is the stable case
    /// (an element with a dry node moves its layers on the 2D wet/dry
    /// kernel's subcells and has no baroclinic exchange); the x–z slices
    /// let 0.5–5 m of water beside 21–40 m hold and beside 60 m grow
    /// ([`ElementSlopeBound`]). ROMS-type models mask such cells as land.
    ///
    /// Shore nodes too shallow to be resolved below `high_water` stay, and
    /// so does every node of an element no deeper than `deep`. The coastline
    /// moves by at most one node at cliffs. Coincident nodes move together.
    ///
    /// # Panics
    /// If `min_depth` is not positive, `deep` is not above it, or `land` is
    /// not high enough to stay a thin column at `high_water`.
    pub fn raise_cliff_shores(
        &mut self,
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        geom: &GeometricFactors2D,
        deep: f64,
        min_depth: f64,
        high_water: f64,
        land: f64,
    ) -> CliffShores {
        assert!(
            min_depth > 0.0 && deep > min_depth,
            "need 0 < min_depth < deep, got {min_depth} and {deep}"
        );
        assert!(
            high_water - land < min_depth,
            "land at {land} m is a resolved column at high water {high_water} m"
        );
        let graph = NodeGraph::new(mesh, ops, geom);
        let depth = graph.depths(&self.data, geom);
        let shore = |g: usize| depth[g] < min_depth && depth[g] + high_water >= min_depth;
        let mut raise = vec![false; depth.len()];
        let mut report = CliffShores::default();
        for nodes in graph.global.chunks(ops.n_nodes) {
            let deepest = nodes
                .iter()
                .map(|&g| depth[g])
                .fold(f64::NEG_INFINITY, f64::max);
            if deepest <= deep || !nodes.iter().any(|&g| shore(g)) {
                continue;
            }
            report.elements += 1;
            for &g in nodes.iter().filter(|&&g| shore(g)) {
                raise[g] = true;
            }
        }
        for g in (0..raise.len()).filter(|&g| raise[g]) {
            report.nodes += 1;
            report.max_raise = report.max_raise.max(land + depth[g]);
            if depth[g] > 0.0 {
                report.area += graph.area[g];
            }
        }
        if report.nodes > 0 {
            for (b, &g) in self.data.iter_mut().zip(&graph.global) {
                if raise[g] {
                    *b = land;
                }
            }
            self.compute_gradients(ops, geom);
        }
        report
    }

    /// Fill the tidal ponds of at most `max_nodes` nodes to their sills:
    /// every node that the open water does not reach along the elements'
    /// grid lines (through wet nodes, coincident nodes together) at some
    /// water level between `low_water` and `high_water` takes the bed of the
    /// lowest sill between it and the open water, and the gradients are
    /// recomputed. The open water is the nodes on the mesh's boundary faces
    /// that are not walls (on a mesh with none, its deepest node).
    ///
    /// The sills come from a priority flood (Barnes, Lehman & Mulla 2014):
    /// a node's spill level is the lowest water level at which a path of
    /// wet nodes joins it to the open water, the smallest over the paths of
    /// the highest bed along each. A node is cut off at the levels between
    /// its bed and its spill level, so it is a pond within the tide where
    /// its bed is below its spill level, its spill level is at or above
    /// `low_water` (below, it is open water at every level) and its bed is
    /// below `high_water` (above, the water never reaches it).
    ///
    /// Many such nodes are single P2 nodes of 20–65 m, sampled from a
    /// channel of the elevation model that the mesh does not resolve. Below
    /// its sill a pond moves no water, but the wet/dry kernel let one hold a
    /// current: on the Frøya coastline bed (3D smoothing, 1 m thin depth),
    /// 58 ponds at −0.6 m, a 25 m single-node pond held a 3.7 m/s jet
    /// through a 5 h ebb in 3D, and a 2.2 m pond held 1.2 m/s and set the 3D
    /// step after the wall at dry banks stopped the first
    /// (`notes/2d-numerics.md`). Filled, a pond is a flat at its sill: dry
    /// below it, and joined to the open water above it. Ponds cut off only
    /// above the lowest water count too: filled below −1.43 m alone, 1 m
    /// nodes beside the filled ones were cut off at −0.6 m and held
    /// 0.7–1.4 m/s.
    ///
    /// A larger pond is a basin the mesh resolves behind an inlet it does
    /// not (a 35 m lagoon of 52 nodes at Frøya), still in the runs: it is
    /// left alone (`TidalPonds::left`); one element's nodes
    /// (`ops.n_nodes`) is a natural `max_nodes`.
    ///
    /// Take `low_water` as the lowest water level the runs see, not lower: a
    /// node whose sill lies below the lowest water is open water all the
    /// time, and filling it would turn a resolved channel into a flat.
    pub fn fill_tidal_ponds(
        &mut self,
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        geom: &GeometricFactors2D,
        low_water: f64,
        high_water: f64,
        max_nodes: usize,
    ) -> TidalPonds {
        let graph = NodeGraph::new(mesh, ops, geom);
        let n_global = graph.area.len();
        let bed: Vec<f64> = graph.depths(&self.data, geom).iter().map(|d| -d).collect();
        let mut neighbours = vec![Vec::new(); n_global];
        for &[p, q] in &graph.pairs {
            neighbours[p].push(q);
            neighbours[q].push(p);
        }
        // The open water: the nodes of the open boundary faces
        let mut open = Vec::new();
        for k in 0..mesh.n_elements {
            for face in 0..ops.face_nodes.len() {
                let tag = mesh.boundary_tag(ElementIndex::new(k), face);
                if tag.is_some_and(|tag| tag != BoundaryTag::Wall) {
                    open.extend(
                        ops.face_nodes[face]
                            .iter()
                            .map(|&i| graph.global[k * ops.n_nodes + i]),
                    );
                }
            }
        }
        if open.is_empty() {
            open.extend((0..n_global).min_by(|&a, &b| bed[a].total_cmp(&bed[b])));
        }
        // Priority flood: the spill level of every node
        #[derive(PartialEq)]
        struct Level(f64, usize);
        impl Eq for Level {}
        impl PartialOrd for Level {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for Level {
            // Reversed: the heap pops the lowest level first
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                other.0.total_cmp(&self.0).then(other.1.cmp(&self.1))
            }
        }
        let mut spill = vec![f64::INFINITY; n_global];
        let mut heap = std::collections::BinaryHeap::new();
        for g in open {
            if bed[g] < spill[g] {
                spill[g] = bed[g];
                heap.push(Level(bed[g], g));
            }
        }
        while let Some(Level(level, g)) = heap.pop() {
            if level > spill[g] {
                continue;
            }
            for &n in &neighbours[g] {
                let reach = level.max(bed[n]);
                if reach < spill[n] {
                    spill[n] = reach;
                    heap.push(Level(reach, n));
                }
            }
        }
        let pond = |g: usize| bed[g] < spill[g] && spill[g] >= low_water && bed[g] < high_water;
        let mut report = TidalPonds::default();
        // Ponds: connected sets of pond nodes
        let mut parent: Vec<usize> = (0..n_global).collect();
        fn root(parent: &mut [usize], mut a: usize) -> usize {
            while parent[a] != a {
                parent[a] = parent[parent[a]];
                a = parent[a];
            }
            a
        }
        for &[p, q] in &graph.pairs {
            if pond(p) && pond(q) {
                let (rp, rq) = (root(&mut parent, p), root(&mut parent, q));
                parent[rp.max(rq)] = rp.min(rq);
            }
        }
        let mut size = vec![0usize; n_global];
        for g in (0..n_global).filter(|&g| pond(g)) {
            size[root(&mut parent, g)] += 1;
        }
        let mut fill = vec![false; n_global];
        for g in (0..n_global).filter(|&g| pond(g)) {
            let r = root(&mut parent, g);
            if size[r] > max_nodes {
                if r == g {
                    report.left += 1;
                }
                continue;
            }
            if r == g {
                report.ponds += 1;
            }
            fill[g] = true;
            report.nodes += 1;
            report.max_raise = report.max_raise.max(spill[g] - bed[g]);
            report.area += graph.area[g];
            report.volume += graph.area[g] * (spill[g] - bed[g]);
        }
        if report.nodes > 0 {
            for (b, &g) in self.data.iter_mut().zip(&graph.global) {
                if fill[g] {
                    *b = spill[g];
                }
            }
            self.compute_gradients(ops, geom);
        }
        report
    }

    /// Bed from a merged elevation raster (bathymetry and land,
    /// [`crate::io::BedRaster`]), projected onto the nodes
    /// ([`Self::project`], at the raster's pixel size) through the map
    /// `projection` from mesh coordinates to longitude/latitude.
    ///
    /// Nodes on land get the raster's land elevation, so `WetDry` treats them
    /// as dry shore; nodes beyond the raster take its edge values.
    pub fn from_raster<P: crate::io::CoordinateProjection>(
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        geom: &GeometricFactors2D,
        raster: &crate::io::BedRaster,
        projection: &P,
    ) -> Self {
        Self::project(
            mesh,
            ops,
            geom,
            raster.sampler(projection),
            raster.pixel_size(),
        )
    }

    /// Largest number of projection sub-cells per element and direction
    /// ([`Self::project`]).
    pub const MAX_PROJECTION_SUBCELLS: usize = 64;

    /// The bathymetry of the elements `kept` (old element indices, as
    /// returned by [`Mesh2D::retain_elements`]), in that order.
    ///
    /// The elements keep their geometry, so the gradients are copied too.
    pub fn select_elements(&self, kept: &[usize]) -> Self {
        let n = self.n_nodes;
        let pick = |field: &[f64]| -> Vec<f64> {
            kept.iter()
                .flat_map(|&k| field[k * n..][..n].iter().copied())
                .collect()
        };
        Self {
            data: pick(&self.data),
            gradient_x: pick(&self.gradient_x),
            gradient_y: pick(&self.gradient_y),
            n_elements: kept.len(),
            n_nodes: n,
        }
    }

    /// Get bathymetry at node i in element k.
    #[inline]
    pub fn get(&self, k: ElementIndex, i: usize) -> f64 {
        self.data[k.as_usize() * self.n_nodes + i]
    }

    /// Set bathymetry at node i in element k.
    #[inline]
    pub fn set(&mut self, k: ElementIndex, i: usize, value: f64) {
        self.data[k.as_usize() * self.n_nodes + i] = value;
    }

    /// Get bathymetry gradient (∂B/∂x, ∂B/∂y) at node i in element k.
    #[inline]
    pub fn get_gradient(&self, k: ElementIndex, i: usize) -> (f64, f64) {
        let idx = k.as_usize() * self.n_nodes + i;
        (self.gradient_x[idx], self.gradient_y[idx])
    }

    /// Compute positive water-column depth from free surface elevation.
    ///
    /// Bathymetry is stored as bed elevation B, so depth is eta - B.
    #[inline]
    pub fn water_depth(&self, k: ElementIndex, i: usize, eta: f64) -> f64 {
        eta - self.get(k, i)
    }

    /// Compute the gradient of positive water-column depth.
    ///
    /// Since depth D = eta - B, grad(D) = grad(eta) - grad(B).
    #[inline]
    pub fn water_depth_gradient(
        &self,
        k: ElementIndex,
        i: usize,
        d_eta_dx: f64,
        d_eta_dy: f64,
    ) -> (f64, f64) {
        let (db_dx, db_dy) = self.get_gradient(k, i);
        (d_eta_dx - db_dx, d_eta_dy - db_dy)
    }

    /// Get all bathymetry values for element k.
    pub fn element(&self, k: ElementIndex) -> &[f64] {
        let start = k.as_usize() * self.n_nodes;
        &self.data[start..start + self.n_nodes]
    }

    /// Get all x-gradient values for element k.
    pub fn element_gradient_x(&self, k: usize) -> &[f64] {
        let start = k * self.n_nodes;
        &self.gradient_x[start..start + self.n_nodes]
    }

    /// Get all y-gradient values for element k.
    pub fn element_gradient_y(&self, k: usize) -> &[f64] {
        let start = k * self.n_nodes;
        &self.gradient_y[start..start + self.n_nodes]
    }

    /// Compute gradients ∂B/∂x and ∂B/∂y using the differentiation matrices.
    ///
    /// Process:
    /// 1. Compute ∂B/∂r and ∂B/∂s using Dr and Ds matrices
    /// 2. Transform to physical derivatives using geometric factors:
    ///    ∂B/∂x = rx * ∂B/∂r + sx * ∂B/∂s
    ///    ∂B/∂y = ry * ∂B/∂r + sy * ∂B/∂s
    pub fn compute_gradients(&mut self, ops: &DGOperators2D, geom: &GeometricFactors2D) {
        let n = self.n_nodes;

        for k in ElementIndex::iter(self.n_elements) {
            let ki = k.as_usize();
            let b_k = self.element(k);

            // Compute ∂B/∂r = Dr * B_k
            let mut db_dr = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    db_dr[i] += ops.dr[(i, j)] * b_k[j];
                }
            }

            // Compute ∂B/∂s = Ds * B_k
            let mut db_ds = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    db_ds[i] += ops.ds[(i, j)] * b_k[j];
                }
            }

            // Transform to physical derivatives and store
            let start = ki * n;
            for i in 0..n {
                let (db_dx, db_dy) = geom.transform_derivatives(ki, i, db_dr[i], db_ds[i]);
                self.gradient_x[start + i] = db_dx;
                self.gradient_y[start + i] = db_dy;
            }
        }
    }

    /// Get bathymetry at a specific face of element k.
    ///
    /// Returns values at all face nodes in the order defined by `ops.face_nodes[face]`.
    pub fn face_values(&self, k: ElementIndex, ops: &DGOperators2D, face: usize) -> Vec<f64> {
        ops.face_nodes[face]
            .iter()
            .map(|&i| self.get(k, i))
            .collect()
    }

    /// Get maximum bathymetry value in the domain.
    pub fn max(&self) -> f64 {
        self.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    }

    /// Get minimum bathymetry value in the domain.
    pub fn min(&self) -> f64 {
        self.data.iter().cloned().fold(f64::INFINITY, f64::min)
    }

    /// Get maximum absolute gradient magnitude in the domain.
    pub fn max_gradient_magnitude(&self) -> f64 {
        self.gradient_x
            .iter()
            .zip(self.gradient_y.iter())
            .map(|(&gx, &gy)| (gx * gx + gy * gy).sqrt())
            .fold(0.0, f64::max)
    }

    /// Get the gradient magnitude at a specific node.
    #[inline]
    pub fn gradient_magnitude(&self, k: ElementIndex, i: usize) -> f64 {
        let (gx, gy) = self.get_gradient(k, i);
        (gx * gx + gy * gy).sqrt()
    }

    /// Project bathymetry to linear (planar) within each element.
    ///
    /// For each element, sets the nodal values to the planar function
    ///   B_linear(x, y) = B_mean + (∂B/∂x)_mean * (x - x_c) + (∂B/∂y)_mean * (y - y_c)
    ///
    /// where (x_c, y_c) is the mean of the nodal coordinates. The fit is done
    /// element by element, so the result generally jumps across faces.
    ///
    /// # Well-balancing
    ///
    /// Planar B is **not** well-balanced at every order with
    /// `SWEFormulation2D::Standard`. The collocated volume term differentiates
    /// the interpolant of ½gh², which balances the `BathymetrySource2D` term
    /// g·h·∂B only if ½gh² is resolved by the element polynomials (deg B ≤ p/2):
    /// - p = 1: ½gh² is quadratic but interpolated linearly, leaving a nodal
    ///   residual g(h̄ − hᵢ)∂B (≈ 0.07 m/s² on steep coastal slopes). Not balanced.
    /// - p ≥ 2: balanced to round-off, provided `BathymetrySource2D` is in
    ///   `source_terms` and `with_well_balanced(true)` handles the face jumps.
    ///
    /// `SWEFormulation2D::EntropyStable` is well-balanced for any nodal B at every
    /// order, so this projection (and its loss of sub-element detail) is not
    /// needed there.
    ///
    /// # Arguments
    /// * `mesh` - The 2D mesh for element geometry
    /// * `ops` - DG operators for node coordinates
    /// * `geom` - Geometric factors for gradient recomputation
    pub fn linearize(&mut self, mesh: &Mesh2D, ops: &DGOperators2D, geom: &GeometricFactors2D) {
        let n = self.n_nodes;

        for k in ElementIndex::iter(self.n_elements) {
            let ki = k.as_usize();
            let start = ki * n;

            // Compute element centroid and mean values
            let mut x_sum = 0.0;
            let mut y_sum = 0.0;
            let mut b_sum = 0.0;
            let mut gx_sum = 0.0;
            let mut gy_sum = 0.0;

            for i in 0..n {
                let r = ops.nodes_r[i];
                let s = ops.nodes_s[i];
                let [x, y] = mesh.reference_to_physical(k, r, s);
                x_sum += x;
                y_sum += y;
                b_sum += self.data[start + i];
                gx_sum += self.gradient_x[start + i];
                gy_sum += self.gradient_y[start + i];
            }

            let n_f = n as f64;
            let x_c = x_sum / n_f;
            let y_c = y_sum / n_f;
            let b_mean = b_sum / n_f;
            let gx_mean = gx_sum / n_f;
            let gy_mean = gy_sum / n_f;

            // Set nodal values to linear function
            for i in 0..n {
                let r = ops.nodes_r[i];
                let s = ops.nodes_s[i];
                let [x, y] = mesh.reference_to_physical(k, r, s);

                // B_linear = B_mean + gx_mean * (x - x_c) + gy_mean * (y - y_c)
                self.data[start + i] = b_mean + gx_mean * (x - x_c) + gy_mean * (y - y_c);
            }
        }

        // Recompute gradients (they should now be constant within each element)
        self.compute_gradients(ops, geom);
    }

    /// Set bathymetry to cell-average (constant) within each element.
    ///
    /// This is the simplest approach for well-balanced schemes:
    /// - Zero gradients within elements → no volume source term
    /// - All bathymetry effects come from interface reconstruction
    /// - Requires `well_balanced=true` in RHS config for proper treatment
    ///
    /// Note: This loses sub-element bathymetry detail but ensures
    /// lake-at-rest preservation. Use with hydrostatic reconstruction.
    pub fn to_cell_average(&mut self) {
        let n = self.n_nodes;

        for k in 0..self.n_elements {
            let start = k * n;

            // Compute element mean
            let mut b_sum = 0.0;
            for i in 0..n {
                b_sum += self.data[start + i];
            }
            let b_mean = b_sum / n as f64;

            // Set all nodes to mean value
            for i in 0..n {
                self.data[start + i] = b_mean;
            }

            // Zero gradients
            for i in 0..n {
                self.gradient_x[start + i] = 0.0;
                self.gradient_y[start + i] = 0.0;
            }
        }
    }
}

/// Common 2D bathymetry profiles for testing.
pub mod profiles {
    /// Gaussian bump centered at (x_c, y_c) with height A and width σ.
    ///
    /// B(x, y) = A * exp(-((x - x_c)² + (y - y_c)²) / (2σ²))
    pub fn gaussian_bump(x: f64, y: f64, x_c: f64, y_c: f64, amplitude: f64, sigma: f64) -> f64 {
        let dx = x - x_c;
        let dy = y - y_c;
        let r2 = dx * dx + dy * dy;
        amplitude * (-r2 / (2.0 * sigma * sigma)).exp()
    }

    /// Linear slope in the x-direction.
    ///
    /// B(x, y) = slope_x * x + offset
    pub fn linear_slope_x(x: f64, _y: f64, slope_x: f64, offset: f64) -> f64 {
        slope_x * x + offset
    }

    /// Linear slope in both directions.
    ///
    /// B(x, y) = slope_x * x + slope_y * y + offset
    pub fn linear_slope(x: f64, y: f64, slope_x: f64, slope_y: f64, offset: f64) -> f64 {
        slope_x * x + slope_y * y + offset
    }

    /// Fjord sill profile: a ridge across the y-direction.
    ///
    /// B(x, y) = B_base + A * exp(-((x - x_sill)² / (2σ²)))
    ///
    /// This creates a sill (underwater ridge) perpendicular to the x-axis.
    pub fn sill(x: f64, _y: f64, x_sill: f64, amplitude: f64, sigma: f64, b_base: f64) -> f64 {
        let dx = x - x_sill;
        b_base + amplitude * (-dx * dx / (2.0 * sigma * sigma)).exp()
    }

    /// Channel profile: deeper in the center, shallower at edges.
    ///
    /// B(x, y) = B_center + A * (1 - cos(π * (y - y_min) / (y_max - y_min)))
    ///
    /// where B_center is the center depth and A is the wall height.
    pub fn channel(y: f64, y_min: f64, y_max: f64, b_center: f64, wall_height: f64) -> f64 {
        let t = (y - y_min) / (y_max - y_min);
        b_center + wall_height * (1.0 - (std::f64::consts::PI * t).cos()) / 2.0
    }

    /// Parabolic bowl for wetting/drying tests.
    ///
    /// B(x, y) = h_0 * ((x - x_c)² + (y - y_c)²) / a²
    pub fn parabolic_bowl(x: f64, y: f64, x_center: f64, y_center: f64, a: f64, h_0: f64) -> f64 {
        let dx = x - x_center;
        let dy = y - y_center;
        h_0 * (dx * dx + dy * dy) / (a * a)
    }

    /// Step function in x-direction (discontinuous bathymetry).
    ///
    /// B(x, y) = B_left if x < x_step, B_right otherwise
    pub fn step_x(x: f64, _y: f64, x_step: f64, b_left: f64, b_right: f64) -> f64 {
        if x < x_step { b_left } else { b_right }
    }

    /// Narrow strait: elevated bathymetry except in a narrow passage.
    ///
    /// Creates a topographic barrier with a gap.
    /// B = B_barrier everywhere except where |y - y_center| < gap_width/2
    /// In the gap: B = B_channel
    pub fn strait(
        x: f64,
        y: f64,
        x_barrier: f64,
        barrier_width: f64,
        y_center: f64,
        gap_width: f64,
        b_channel: f64,
        b_barrier: f64,
    ) -> f64 {
        // Check if within the barrier x-range
        if (x - x_barrier).abs() > barrier_width / 2.0 {
            return b_channel; // Outside barrier
        }

        // Within barrier x-range: check if in gap
        if (y - y_center).abs() < gap_width / 2.0 {
            b_channel // In the strait gap
        } else {
            b_barrier // On the barrier
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn k(idx: usize) -> ElementIndex {
        ElementIndex::new(idx)
    }

    const TOL: f64 = 1e-10;

    fn make_mesh_and_ops() -> (Mesh2D, DGOperators2D, GeometricFactors2D) {
        let mesh = Mesh2D::uniform_rectangle(0.0, 10.0, 0.0, 10.0, 4, 4);
        let ops = DGOperators2D::new(3);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        (mesh, ops, geom)
    }

    #[test]
    fn test_flat_bathymetry() {
        let bathy = Bathymetry2D::flat(4, 16);

        assert_eq!(bathy.data.len(), 64);
        assert_eq!(bathy.gradient_x.len(), 64);
        assert_eq!(bathy.gradient_y.len(), 64);

        for &b in &bathy.data {
            assert!(b.abs() < TOL);
        }
        for &g in &bathy.gradient_x {
            assert!(g.abs() < TOL);
        }
        for &g in &bathy.gradient_y {
            assert!(g.abs() < TOL);
        }
    }

    #[test]
    fn test_constant_bathymetry() {
        let bathy = Bathymetry2D::constant(4, 16, 5.0);

        for &b in &bathy.data {
            assert!((b - 5.0).abs() < TOL);
        }
        for &g in &bathy.gradient_x {
            assert!(g.abs() < TOL); // Gradient of constant is zero
        }
        for &g in &bathy.gradient_y {
            assert!(g.abs() < TOL);
        }
    }

    #[test]
    fn test_from_function_constant() {
        let (mesh, ops, geom) = make_mesh_and_ops();
        let bathy = Bathymetry2D::from_function(&mesh, &ops, &geom, |_, _| 3.0);

        for &b in &bathy.data {
            assert!((b - 3.0).abs() < TOL);
        }
        for &g in &bathy.gradient_x {
            assert!(g.abs() < 1e-8);
        }
        for &g in &bathy.gradient_y {
            assert!(g.abs() < 1e-8);
        }
    }

    #[test]
    fn test_from_function_linear_x() {
        let (mesh, ops, geom) = make_mesh_and_ops();

        // Linear bathymetry: B(x, y) = 2*x
        let bathy = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, _y| 2.0 * x);

        // Check values
        for ki in 0..mesh.n_elements {
            for i in 0..ops.n_nodes {
                let r = ops.nodes_r[i];
                let s = ops.nodes_s[i];
                let [x, _y] = mesh.reference_to_physical(k(ki), r, s);
                let b = bathy.get(k(ki), i);
                assert!(
                    (b - 2.0 * x).abs() < TOL,
                    "B({}, _) = {}, expected {}",
                    x,
                    b,
                    2.0 * x
                );
            }
        }

        // Gradient should be (2, 0) everywhere
        for ki in 0..mesh.n_elements {
            for i in 0..ops.n_nodes {
                let (gx, gy) = bathy.get_gradient(k(ki), i);
                assert!((gx - 2.0).abs() < 1e-8, "∂B/∂x = {}, expected 2.0", gx);
                assert!(gy.abs() < 1e-8, "∂B/∂y = {}, expected 0.0", gy);
            }
        }
    }

    #[test]
    fn test_from_function_linear_y() {
        let (mesh, ops, geom) = make_mesh_and_ops();

        // Linear bathymetry: B(x, y) = 3*y
        let bathy = Bathymetry2D::from_function(&mesh, &ops, &geom, |_x, y| 3.0 * y);

        // Gradient should be (0, 3) everywhere
        for ki in 0..mesh.n_elements {
            for i in 0..ops.n_nodes {
                let (gx, gy) = bathy.get_gradient(k(ki), i);
                assert!(gx.abs() < 1e-8, "∂B/∂x = {}, expected 0.0", gx);
                assert!((gy - 3.0).abs() < 1e-8, "∂B/∂y = {}, expected 3.0", gy);
            }
        }
    }

    #[test]
    fn test_from_function_linear_xy() {
        let (mesh, ops, geom) = make_mesh_and_ops();

        // Linear bathymetry: B(x, y) = 2*x + 3*y + 1
        let bathy = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, y| 2.0 * x + 3.0 * y + 1.0);

        // Gradient should be (2, 3) everywhere
        for ki in 0..mesh.n_elements {
            for i in 0..ops.n_nodes {
                let (gx, gy) = bathy.get_gradient(k(ki), i);
                assert!((gx - 2.0).abs() < 1e-8, "∂B/∂x = {}, expected 2.0", gx);
                assert!((gy - 3.0).abs() < 1e-8, "∂B/∂y = {}, expected 3.0", gy);
            }
        }
    }

    #[test]
    fn test_from_function_quadratic() {
        let (mesh, ops, geom) = make_mesh_and_ops();

        // Quadratic bathymetry: B(x, y) = x² + y²
        // Gradients: ∂B/∂x = 2x, ∂B/∂y = 2y
        let bathy = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, y| x * x + y * y);

        for ki in 0..mesh.n_elements {
            for i in 0..ops.n_nodes {
                let r = ops.nodes_r[i];
                let s = ops.nodes_s[i];
                let [x, y] = mesh.reference_to_physical(k(ki), r, s);
                let (gx, gy) = bathy.get_gradient(k(ki), i);

                assert!(
                    (gx - 2.0 * x).abs() < 1e-6,
                    "∂B/∂x at ({}, {}) is {}, expected {}",
                    x,
                    y,
                    gx,
                    2.0 * x
                );
                assert!(
                    (gy - 2.0 * y).abs() < 1e-6,
                    "∂B/∂y at ({}, {}) is {}, expected {}",
                    x,
                    y,
                    gy,
                    2.0 * y
                );
            }
        }
    }

    #[test]
    fn test_element_access() {
        let (mesh, ops, geom) = make_mesh_and_ops();
        let bathy = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, y| x + y);

        let elem = bathy.element(k(2));
        assert_eq!(elem.len(), ops.n_nodes);

        let grad_x = bathy.element_gradient_x(2);
        assert_eq!(grad_x.len(), ops.n_nodes);

        let grad_y = bathy.element_gradient_y(2);
        assert_eq!(grad_y.len(), ops.n_nodes);
    }

    #[test]
    fn test_min_max() {
        let (mesh, ops, geom) = make_mesh_and_ops();
        let bathy = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, y| x + y);

        let min = bathy.min();
        let max = bathy.max();

        // Domain is [0, 10] × [0, 10], so min ≈ 0, max ≈ 20
        assert!(min < 0.5);
        assert!(max > 19.5);
    }

    #[test]
    fn test_gradient_magnitude() {
        let (mesh, ops, geom) = make_mesh_and_ops();

        // B(x, y) = 3*x + 4*y, gradient = (3, 4), magnitude = 5
        let bathy = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, y| 3.0 * x + 4.0 * y);

        let max_mag = bathy.max_gradient_magnitude();
        assert!((max_mag - 5.0).abs() < 1e-8);

        for ki in 0..mesh.n_elements {
            for i in 0..ops.n_nodes {
                let mag = bathy.gradient_magnitude(k(ki), i);
                assert!((mag - 5.0).abs() < 1e-8);
            }
        }
    }

    #[test]
    fn test_gaussian_bump_profile() {
        let b = profiles::gaussian_bump(5.0, 5.0, 5.0, 5.0, 1.0, 1.0);
        assert!((b - 1.0).abs() < TOL); // At center, B = amplitude

        let b_off = profiles::gaussian_bump(0.0, 0.0, 5.0, 5.0, 1.0, 1.0);
        assert!(b_off < 1e-5); // Far from center, B ≈ 0
    }

    #[test]
    fn test_sill_profile() {
        // Sill at x = 5 with amplitude 1, sigma 1
        let b_center = profiles::sill(5.0, 0.0, 5.0, 1.0, 1.0, 0.0);
        assert!((b_center - 1.0).abs() < TOL); // At sill center

        let b_far = profiles::sill(0.0, 0.0, 5.0, 1.0, 1.0, 0.0);
        assert!(b_far < 1e-5); // Far from sill
    }

    #[test]
    fn test_strait_profile() {
        // Barrier at x = 5, width 2, gap at y = 5, gap width 1
        // Channel depth 0, barrier height 10

        // In the gap (should be channel depth)
        let b_gap = profiles::strait(5.0, 5.0, 5.0, 2.0, 5.0, 1.0, 0.0, 10.0);
        assert!((b_gap - 0.0).abs() < TOL);

        // On the barrier (outside gap)
        let b_barrier = profiles::strait(5.0, 7.0, 5.0, 2.0, 5.0, 1.0, 0.0, 10.0);
        assert!((b_barrier - 10.0).abs() < TOL);

        // Outside barrier x-range
        let b_outside = profiles::strait(0.0, 5.0, 5.0, 2.0, 5.0, 1.0, 0.0, 10.0);
        assert!((b_outside - 0.0).abs() < TOL);
    }

    #[test]
    fn test_parabolic_bowl_profile() {
        let h_0 = 1.0;
        let a = 5.0;

        let b_center = profiles::parabolic_bowl(5.0, 5.0, 5.0, 5.0, a, h_0);
        assert!(b_center.abs() < TOL); // At center, B = 0

        // At (10, 5): dx = 5, dy = 0, B = 1.0 * 25 / 25 = 1.0
        let b_edge = profiles::parabolic_bowl(10.0, 5.0, 5.0, 5.0, a, h_0);
        assert!((b_edge - h_0).abs() < TOL);
    }

    /// `n` × `n` mesh of [0, 1]² with interior vertices moved, so that
    /// the elements are general (non-parallelogram) quadrilaterals.
    fn distorted_mesh(n: usize) -> Mesh2D {
        let mut mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, n, n);
        let tau = 2.0 * std::f64::consts::PI;
        let a = 0.3 / n as f64;
        for v in &mut mesh.vertices {
            let [x, y] = *v;
            let bump = (tau * x).sin() * (tau * y).sin();
            *v = [x + a * bump, y - 0.7 * a * bump];
        }
        mesh
    }

    fn setup(mesh: Mesh2D, order: usize) -> (Mesh2D, DGOperators2D, GeometricFactors2D) {
        let ops = DGOperators2D::new(order);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        (mesh, ops, geom)
    }

    /// Largest |B − f| over the nodes.
    fn nodal_error(
        bathy: &Bathymetry2D,
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        f: impl Fn(f64, f64) -> f64,
    ) -> f64 {
        let mut error: f64 = 0.0;
        for e in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let [x, y] = mesh.reference_to_physical(e, ops.nodes_r[i], ops.nodes_s[i]);
                error = error.max((bathy.get(e, i) - f(x, y)).abs());
            }
        }
        error
    }

    #[test]
    fn test_projection_is_exact_for_polynomials_of_the_element_degree() {
        // x^a y^b with a + b ≤ N lies in Q_N of a bilinear element
        for order in 1..=4 {
            let (mesh, ops, geom) = setup(distorted_mesh(3), order);
            let f = |x: f64, y: f64| {
                -20.0 + 3.0 * x - 2.0 * y
                    + (0..=order)
                        .map(|a| x.powi(a as i32) * y.powi((order - a) as i32))
                        .sum::<f64>()
            };
            let bathy = Bathymetry2D::project(&mesh, &ops, &geom, f, 0.2);
            let error = nodal_error(&bathy, &mesh, &ops, f);
            assert!(error < 1e-11, "N = {order}: {error:e}");
        }
    }

    #[test]
    fn test_projection_keeps_volume_and_bounds_at_a_cliff() {
        // A coastline cliff from −30 m to +5 m at x = 0.5, inside the second
        // column of 1/3-wide elements and on a sub-cell edge (1/60-wide
        // sub-cells), so the quadrature of the step is exact
        let (lo, hi, x_c) = (-30.0, 5.0, 0.5);
        let f = |x: f64, _y: f64| if x < x_c { lo } else { hi };
        for order in 1..=4 {
            let (mesh, ops, geom) =
                setup(Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 3, 3), order);
            let bathy = Bathymetry2D::project(&mesh, &ops, &geom, f, 0.034);
            // The total volume is the data's (element volumes move between
            // neighbours when coincident nodes are averaged)
            let volume: f64 = ElementIndex::iter(mesh.n_elements)
                .map(|e| geom.integrate_element(e.as_usize(), bathy.element(e)))
                .sum();
            let exact = lo * x_c + hi * (1.0 - x_c);
            assert!(
                (volume - exact).abs() < 1e-12,
                "N = {order}: {volume} vs {exact}"
            );
            assert!(
                bathy
                    .data
                    .iter()
                    .all(|&b| (lo - 1e-9..=hi + 1e-9).contains(&b)),
                "N = {order}: [{}, {}]",
                bathy.min(),
                bathy.max()
            );
            // The step elements keep a shoreline: dry and wet nodes
            let step = ElementIndex::new(1);
            let (b_min, b_max) = bathy
                .element(step)
                .iter()
                .fold((hi, lo), |(a, b), &v| (a.min(v), b.max(v)));
            assert!(
                b_min < -10.0 && b_max > 0.0,
                "N = {order}: [{b_min}, {b_max}]"
            );
            // Continuous across elements; away from the step column (off its
            // faces, which are averaged with it) the data
            assert_continuous(&bathy, &mesh, &ops, order);
            for e in ElementIndex::iter(mesh.n_elements) {
                for i in 0..ops.n_nodes {
                    let [x, y] = mesh.reference_to_physical(e, ops.nodes_r[i], ops.nodes_s[i]);
                    if !(1.0 / 3.0 - 1e-9..=2.0 / 3.0 + 1e-9).contains(&x) {
                        let error = (bathy.get(e, i) - f(x, y)).abs();
                        assert!(error < 1e-11, "N = {order}, ({x}, {y}): {error:e}");
                    }
                }
            }
        }
    }

    /// Coincident nodes (same position to 1e-9) carry the same B.
    fn assert_continuous(bathy: &Bathymetry2D, mesh: &Mesh2D, ops: &DGOperators2D, order: usize) {
        let mut seen: std::collections::HashMap<(i64, i64), f64> = Default::default();
        for e in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let [x, y] = mesh.reference_to_physical(e, ops.nodes_r[i], ops.nodes_s[i]);
                let key = ((x * 1e9).round() as i64, (y * 1e9).round() as i64);
                let b = bathy.get(e, i);
                let first = *seen.entry(key).or_insert(b);
                assert!(
                    (b - first).abs() < 1e-12,
                    "N = {order}, ({x}, {y}): {b} vs {first}"
                );
            }
        }
    }

    #[test]
    fn test_projection_does_not_alias_unresolved_data() {
        // Ripples of wavelength 0.13 on elements of 1/2 (P2 node spacing
        // 1/4): point sampling returns their amplitude, the projection
        // nearly their mean
        let (mesh, ops, geom) = setup(distorted_mesh(2), 2);
        let ripple = |x: f64, y: f64| {
            let k = 2.0 * std::f64::consts::PI / 0.13;
            -50.0 + 10.0 * (k * x + 0.3).sin() * (k * y + 0.7).sin()
        };
        let sampled = Bathymetry2D::from_function(&mesh, &ops, &geom, ripple);
        let projected = Bathymetry2D::project(&mesh, &ops, &geom, ripple, 0.13 / 4.0);
        let spread =
            |b: &Bathymetry2D| b.data.iter().map(|&v| (v + 50.0).abs()).fold(0.0, f64::max);
        assert!(spread(&sampled) > 5.0, "sampled: {}", spread(&sampled));
        assert!(
            spread(&projected) < 1.0,
            "projected: {}",
            spread(&projected)
        );
    }

    #[test]
    fn test_projection_converges_at_n_plus_one() {
        let f = |x: f64, y: f64| -40.0 + 10.0 * (3.0 * x + 1.0).sin() * (2.0 * y - 0.5).cos();
        for order in 2..=3 {
            let errors: Vec<f64> = [4, 8, 16]
                .iter()
                .map(|&n| {
                    let (mesh, ops, geom) = setup(distorted_mesh(n), order);
                    let bathy = Bathymetry2D::project(&mesh, &ops, &geom, f, 0.25 / n as f64);
                    nodal_error(&bathy, &mesh, &ops, f)
                })
                .collect();
            for pair in errors.windows(2) {
                let rate = (pair[0] / pair[1]).log2();
                assert!(
                    rate > order as f64 + 0.7,
                    "N = {order}: rate {rate:.2} ({errors:?})"
                );
            }
        }
    }

    #[test]
    fn test_raise_isolated_wet_nodes() {
        // Dry land at +2 m with a lone −10 m node at an interior vertex, a
        // wet pair along a grid line, and open water along the west column
        let (mesh, ops, geom) = setup(Mesh2D::uniform_rectangle(0.0, 4.0, 0.0, 4.0, 4, 4), 2);
        let lone = [2.0, 2.0];
        let pair = [[3.0, 1.0], [3.0, 1.5]];
        let bed = |x: f64, y: f64| {
            let at = |p: [f64; 2]| (x - p[0]).abs() < 1e-9 && (y - p[1]).abs() < 1e-9;
            if at(lone) {
                -10.0
            } else if pair.iter().any(|&p| at(p)) {
                -5.0
            } else if x < 1.0 {
                -20.0
            } else {
                2.0
            }
        };
        let mut bathy = Bathymetry2D::from_function(&mesh, &ops, &geom, bed);
        assert_eq!(bathy.raise_isolated_wet_nodes(&mesh, &ops, &geom, 0.0), 1);
        for e in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let [x, y] = mesh.reference_to_physical(e, ops.nodes_r[i], ops.nodes_s[i]);
                let is_lone = (x - lone[0]).abs() < 1e-9 && (y - lone[1]).abs() < 1e-9;
                // The lone node is raised to its lowest neighbour
                let expected = if is_lone { 2.0 } else { bed(x, y) };
                assert_eq!(bathy.get(e, i), expected, "({x}, {y})");
            }
        }
        // Nothing left to raise
        assert_eq!(bathy.raise_isolated_wet_nodes(&mesh, &ops, &geom, 0.0), 0);
    }

    #[test]
    fn test_lower_wall_land() {
        // 3 × 2 unit squares, P2, walls on the west, north and east, open to
        // the south. Water deepening eastward, land (+3 m) along the north
        // wall and the open south side, and a dry node inside the north-east
        // element: a shore the mesh resolves
        let mut mesh =
            Mesh2D::uniform_rectangle_with_bc(0.0, 3.0, 0.0, 2.0, 3, 2, BoundaryTag::Wall);
        for e in 0..mesh.edges.len() {
            let edge = &mesh.edges[e];
            if edge.right.is_none() {
                let (k, face) = (ElementIndex::new(edge.left.element), edge.left.face);
                let (r, s) = [(0.0, -1.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)][face];
                let [_, y] = mesh.reference_to_physical(k, r, s);
                if y < 1e-9 {
                    mesh.edges[e].boundary_tag = Some(BoundaryTag::Open);
                }
            }
        }
        let (mesh, ops, geom) = setup(mesh, 2);
        let at = |x: f64, y: f64, p: [f64; 2]| (x - p[0]).abs() < 1e-9 && (y - p[1]).abs() < 1e-9;
        let water = |x: f64, y: f64| -10.0 + y - x;
        let bed = |x: f64, y: f64| {
            if y > 2.0 - 1e-9 || y < 1e-9 || at(x, y, [2.5, 1.5]) {
                3.0
            } else {
                water(x, y)
            }
        };
        let mut bathy = Bathymetry2D::from_function(&mesh, &ops, &geom, bed);

        // Below a lowest water of −12 m the north row is a flat: it stays
        let mut flat = bathy.clone();
        assert_eq!(flat.lower_wall_land(&mesh, &ops, &geom, -12.0).changed, 0);
        assert_eq!(flat.data, bathy.data);

        let report = bathy.lower_wall_land(&mesh, &ops, &geom, 0.0);

        // North-west: the corner's inward neighbours are wall nodes, so it
        // takes the element's shallowest node off the walls, (0.5, 1.5);
        // the other north nodes take the node below them; (1, 2) and (2, 2)
        // take the shallower of their two elements' (the same node). The
        // north-east element is a shore and keeps its land, and its corner
        // (2, 2) with it, though the north-middle element would lower it. The
        // open side is not a wall
        let expected = |x: f64, y: f64| {
            if y > 2.0 - 1e-9 && x < 2.0 - 1e-9 {
                if x < 1e-9 {
                    water(0.5, 1.5)
                } else {
                    water(x, 1.5)
                }
            } else {
                bed(x, y)
            }
        };
        for e in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let [x, y] = mesh.reference_to_physical(e, ops.nodes_r[i], ops.nodes_s[i]);
                assert_eq!(bathy.get(e, i), expected(x, y), "({x}, {y})");
            }
        }
        assert_eq!(report.elements, 2);
        assert_eq!(report.changed, 4);
        assert_eq!(report.max_change, 3.0 - water(1.5, 1.5));
        let reference = Bathymetry2D::from_function(&mesh, &ops, &geom, expected);
        let added: f64 = (0..bathy.data.len())
            .map(|m| {
                geom.mass[m] * (reference.data[m].min(0.0) - bed_at(&mesh, &ops, m, bed).min(0.0))
            })
            .sum::<f64>()
            .abs();
        assert!(
            (report.volume - added).abs() < 1e-12 * added,
            "{} {added}",
            report.volume
        );
        assert_eq!(bathy.gradient_x, reference.gradient_x);
        assert_eq!(bathy.gradient_y, reference.gradient_y);

        // Nothing left: the north-east element is a resolved shore
        assert_eq!(bathy.lower_wall_land(&mesh, &ops, &geom, 0.0).changed, 0);
    }

    /// `f` at node `m` (`k·n_nodes + i`).
    fn bed_at(mesh: &Mesh2D, ops: &DGOperators2D, m: usize, f: impl Fn(f64, f64) -> f64) -> f64 {
        let (k, i) = (ElementIndex::new(m / ops.n_nodes), m % ops.n_nodes);
        let [x, y] = mesh.reference_to_physical(k, ops.nodes_r[i], ops.nodes_s[i]);
        f(x, y)
    }

    #[test]
    fn test_select_elements_follows_retain_elements() {
        let (mesh, ops, geom) = setup(distorted_mesh(3), 2);
        let bed = |x: f64, y: f64| x - 0.5 + 0.1 * y;
        let bathy = Bathymetry2D::project(&mesh, &ops, &geom, bed, 0.1);
        let wet = |e: ElementIndex| bathy.element(e).iter().any(|&b| b < 0.0);
        let (water, kept) = mesh.retain_elements(wet, crate::mesh::BoundaryTag::Wall);
        let selected = bathy.select_elements(&kept);
        assert!(water.n_elements < mesh.n_elements);
        let water_geom = GeometricFactors2D::compute(&water, &ops);
        let direct = Bathymetry2D::project(&water, &ops, &water_geom, bed, 0.1);
        for (field, reference) in [
            (&selected.data, &direct.data),
            (&selected.gradient_x, &direct.gradient_x),
            (&selected.gradient_y, &direct.gradient_y),
        ] {
            let diff = field
                .iter()
                .zip(reference)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            assert!(diff < 1e-12, "{diff:e}");
        }
    }

    #[test]
    fn test_max_rx0_of_a_slope() {
        // P1 nodes at x = 0, 0.5, 1: depths 10, 15, 20 m, so r_x0 is 0.2
        // between the first two and 1/7 between the last two
        let (mesh, ops, geom) = setup(Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 2, 2), 1);
        let bathy = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, _| -10.0 - 10.0 * x);
        let position = |node: usize| {
            let (k, i) = (ElementIndex::new(node / ops.n_nodes), node % ops.n_nodes);
            mesh.reference_to_physical(k, ops.nodes_r[i], ops.nodes_s[i])
        };
        let steepest = bathy.max_rx0(&mesh, &ops, &geom, 1.0).unwrap();
        assert!((steepest.value - 0.2).abs() < 1e-14, "{steepest:?}");
        assert!(position(steepest.nodes[0])[0].abs() < 1e-12);
        assert!((position(steepest.nodes[1])[0] - 0.5).abs() < 1e-12);
        // Pairs with a node shallower than `min_depth` do not count
        let deep = bathy.max_rx0(&mesh, &ops, &geom, 12.0).unwrap();
        assert!((deep.value - 1.0 / 7.0).abs() < 1e-14, "{deep:?}");
        assert_eq!(bathy.max_rx0(&mesh, &ops, &geom, 25.0), None);
    }

    /// Distorted mesh with a sharp shoal rising from 40 m to 6 m in open
    /// water (centred on a vertex the distortion leaves in place), and a 1 m
    /// deep shore along x < 0.2.
    fn shoal_bed(order: usize) -> (Mesh2D, DGOperators2D, GeometricFactors2D, Bathymetry2D) {
        let (mesh, ops, geom) = setup(distorted_mesh(4), order);
        let bed = |x: f64, y: f64| {
            if x < 0.2 {
                -1.0
            } else {
                -40.0 + 34.0 * (-((x - 0.5).powi(2) + (y - 0.5).powi(2)) / 0.01).exp()
            }
        };
        let bathy = Bathymetry2D::from_function(&mesh, &ops, &geom, bed);
        (mesh, ops, geom, bathy)
    }

    fn volume(bathy: &Bathymetry2D, geom: &GeometricFactors2D) -> f64 {
        bathy.data.iter().zip(&geom.mass).map(|(b, m)| b * m).sum()
    }

    #[test]
    fn test_element_slope_bound() {
        let bound = ElementSlopeBound::for_pycnocline(19.0);
        assert_eq!(bound.free_depth, 19.0);
        // Shallow water and land: free down to the pycnocline's bottom
        for shallow in [-3.0, 0.0, 5.0, 14.0] {
            assert_eq!(bound.deepest_beside(shallow), 19.0);
        }
        // Deeper: r_x0 = 0.15, a depth ratio of 1.15/0.85
        let ratio = 1.15 / 0.85;
        assert!((bound.deepest_beside(30.0) - 30.0 * ratio).abs() < 1e-12);
        assert!((bound.excess(30.0, 50.0) - (50.0 - 30.0 * ratio)).abs() < 1e-12);
        assert_eq!(bound.excess(30.0, 40.0), 0.0);
        assert_eq!(bound.excess(0.0, 18.0), 0.0);
        assert!((bound.excess(0.0, 40.0) - 21.0).abs() < 1e-12);
    }

    #[test]
    fn test_smooth_element_slopes_bounds_every_element_and_keeps_the_volume_and_the_shore() {
        let bound = ElementSlopeBound::for_pycnocline(19.0);
        let min_depth = 3.0;
        for order in 1..=3 {
            let (mesh, ops, geom, mut bathy) = shoal_bed(order);
            let original = bathy.clone();
            let excess = bathy.element_slope_excess(&mesh, &ops, &geom, bound, min_depth);
            let report = bathy.smooth_element_slopes(&mesh, &ops, &geom, bound, min_depth);
            let tolerance = 1e-6 * bound.free_depth;
            assert_eq!(
                report.elements_before,
                excess.iter().filter(|&&e| e > tolerance).count()
            );
            assert_eq!(
                report.excess_before,
                excess.iter().copied().fold(0.0, f64::max)
            );
            // The shoal (6 m among 40 m); the shore (1 m beside 40 m) is
            // left to the wetting and drying
            assert!(report.elements_before >= 4, "N = {order}: {report:?}");
            assert!(report.excess_before > 5.0, "N = {order}: {report:?}");
            let after = bathy.element_slope_excess(&mesh, &ops, &geom, bound, min_depth);
            let largest = after.iter().copied().fold(0.0, f64::max);
            assert!(largest <= tolerance, "N = {order}: {largest}");
            assert!((largest - report.excess_after).abs() < 1e-12);
            assert_eq!(report.elements_after, 0, "N = {order}: {report:?}");
            // The shore never moves, and no node crosses `min_depth`
            for (node, (&b, &b0)) in bathy.data.iter().zip(&original.data).enumerate() {
                if b0 > -min_depth {
                    assert_eq!(b, b0, "N = {order}: node {node}");
                } else {
                    assert!(-b >= min_depth, "N = {order}: node {node} {b0} → {b}");
                }
            }
            // The volume is kept, the bed stays continuous
            let (v0, v1) = (volume(&original, &geom), volume(&bathy, &geom));
            assert!(
                (v1 - v0).abs() < 1e-13 * v0.abs(),
                "N = {order}: {v0} vs {v1}"
            );
            assert_continuous(&bathy, &mesh, &ops, order);
            let mut fresh = bathy.clone();
            fresh.compute_gradients(&ops, &geom);
            assert_eq!(fresh.gradient_x, bathy.gradient_x);
            // A second pass has nothing to do
            let again = bathy.smooth_element_slopes(&mesh, &ops, &geom, bound, min_depth);
            assert_eq!(again.elements_before, 0, "N = {order}: {again:?}");
            assert_eq!(again.changed, 0, "N = {order}: {again:?}");
        }
    }

    #[test]
    fn test_raise_cliff_shores_makes_land_of_resolvable_shore_nodes_beside_deep_water() {
        // 4 × 4 P2 elements, node spacing 1/8. The shore strip x ≤ 1/8 is
        // 0.7 m deep beside 80 m for y < 1/2 and beside 15 m above, with a
        // node at +1.5 m (never resolved below +2 m) and a 5 m one (the
        // slope smoothing's) in the corner y ≤ 1/8.
        let (mesh, ops, geom) = setup(Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 4, 4), 2);
        let bed = |x: f64, y: f64| match (x < 0.2, y < 0.2) {
            (true, true) if x < 0.1 => 1.5,
            (true, true) => -5.0,
            (true, false) => -0.7,
            (false, _) if y < 0.5 => -80.0,
            (false, _) => -15.0,
        };
        let mut bathy = Bathymetry2D::from_function(&mesh, &ops, &geom, bed);
        let original = bathy.clone();
        let (deep, min_depth, high_water) = (40.0, 1.0, 2.0);
        let report =
            bathy.raise_cliff_shores(&mesh, &ops, &geom, deep, min_depth, high_water, high_water);
        // The shore nodes of the two 80 m elements of the strip: x ∈ {0, 1/8},
        // y ∈ {1/4, 3/8, 1/2}
        let raised = |x: f64, y: f64| x < 0.2 && (0.2..0.55).contains(&y);
        let mut area = 0.0;
        for e in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let [x, y] = mesh.reference_to_physical(e, ops.nodes_r[i], ops.nodes_s[i]);
                let expected = if raised(x, y) { high_water } else { bed(x, y) };
                assert_eq!(bathy.get(e, i), expected, "({x}, {y})");
                if raised(x, y) {
                    area += geom.mass[e.as_usize() * ops.n_nodes + i];
                }
            }
        }
        assert_eq!(report.elements, 2, "{report:?}");
        assert_eq!(report.nodes, 6, "{report:?}");
        assert!((report.max_raise - 2.7).abs() < 1e-12, "{report:?}");
        assert!(
            (report.area - area).abs() < 1e-12 * area,
            "{report:?} vs {area}"
        );
        assert_continuous(&bathy, &mesh, &ops, 2);
        let mut fresh = bathy.clone();
        fresh.compute_gradients(&ops, &geom);
        assert_eq!(fresh.gradient_x, bathy.gradient_x);
        assert_ne!(bathy.gradient_x, original.gradient_x);
        // A second pass has nothing to do; a deeper threshold leaves it all
        let again =
            bathy.raise_cliff_shores(&mesh, &ops, &geom, deep, min_depth, high_water, high_water);
        assert_eq!(again, CliffShores::default());
        let mut untouched = original.clone();
        let none = untouched.raise_cliff_shores(&mesh, &ops, &geom, 80.5, 1.0, 2.0, 2.0);
        assert_eq!(none, CliffShores::default());
        assert_eq!(untouched.data, original.data);
    }

    #[test]
    fn test_fill_tidal_ponds_raises_a_pond_to_its_lowest_sill() {
        // 4 × 2 unit squares, P2 (node spacing 1/2), open to the south. Sea
        // (−10 m) for y ≤ 1/2, a ridge (+3 m) along y = 1, and behind it
        // basin A (−20 m, x ≤ 3/2) and basin B (−15 m, x ≥ 5/2), parted by
        // land at x = 2. A joins the sea over sills at −0.5 m (x = 1) and
        // +0.4 m (x = 0), B over one at −2 m (x = 3).
        let mesh = Mesh2D::uniform_rectangle_with_sides(
            0.0,
            4.0,
            0.0,
            2.0,
            4,
            2,
            [
                BoundaryTag::Open,
                BoundaryTag::Wall,
                BoundaryTag::Wall,
                BoundaryTag::Wall,
            ],
        );
        let (mesh, ops, geom) = setup(mesh, 2);
        let near = |a: f64, b: f64| (a - b).abs() < 1e-9;
        let in_a = |x: f64, y: f64| y > 1.2 && x < 1.7;
        let bed = move |x: f64, y: f64| match () {
            _ if y < 0.7 => -10.0,
            _ if near(y, 1.0) && near(x, 0.0) => 0.4,
            _ if near(y, 1.0) && near(x, 1.0) => -0.5,
            _ if near(y, 1.0) && near(x, 3.0) => -2.0,
            _ if near(y, 1.0) || near(x, 2.0) => 3.0,
            _ if in_a(x, y) => -20.0,
            _ => -15.0,
        };
        let mut bathy = Bathymetry2D::from_function(&mesh, &ops, &geom, bed);
        let original = bathy.clone();
        let report = bathy.fill_tidal_ponds(&mesh, &ops, &geom, -1.0, 2.0, 9);
        // A's eight nodes take the lower sill; nothing else moves
        let mut area = 0.0;
        for e in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let [x, y] = mesh.reference_to_physical(e, ops.nodes_r[i], ops.nodes_s[i]);
                let expected = if in_a(x, y) { -0.5 } else { bed(x, y) };
                assert_eq!(bathy.get(e, i), expected, "({x}, {y})");
                if in_a(x, y) {
                    area += geom.mass[e.as_usize() * ops.n_nodes + i];
                }
            }
        }
        assert_eq!(report.ponds, 1, "{report:?}");
        assert_eq!(report.nodes, 8, "{report:?}");
        assert!((report.max_raise - 19.5).abs() < 1e-12, "{report:?}");
        assert!((report.area - area).abs() < 1e-12 * area, "{report:?}");
        assert!(
            (report.volume - 19.5 * area).abs() < 1e-12 * area,
            "{report:?}"
        );
        assert_continuous(&bathy, &mesh, &ops, 2);
        let mut fresh = bathy.clone();
        fresh.compute_gradients(&ops, &geom);
        assert_eq!(fresh.gradient_x, bathy.gradient_x);
        // A second pass has nothing to do
        let again = bathy.fill_tidal_ponds(&mesh, &ops, &geom, -1.0, 2.0, 9);
        assert_eq!(again, TidalPonds::default());
        // Below B's sill both basins are ponds, also with the lowest water
        // below their beds (they are cut off on the way down); above A's
        // sill neither, and neither with the highest water below their beds
        for low_water in [-3.0, -30.0] {
            let mut low = original.clone();
            let report = low.fill_tidal_ponds(&mesh, &ops, &geom, low_water, 2.0, 9);
            assert_eq!((report.ponds, report.nodes), (2, 16), "{report:?}");
        }
        let mut never = original.clone();
        assert_eq!(
            never.fill_tidal_ponds(&mesh, &ops, &geom, -30.0, -25.0, 9),
            TidalPonds::default()
        );
        let mut high = original.clone();
        assert_eq!(
            high.fill_tidal_ponds(&mesh, &ops, &geom, 0.0, 2.0, 9),
            TidalPonds::default()
        );
        assert_eq!(high.data, original.data);
        // A pond of more nodes than `max_nodes` is left alone
        let mut capped = original.clone();
        let report = capped.fill_tidal_ponds(&mesh, &ops, &geom, -1.0, 2.0, 7);
        assert_eq!(
            report,
            TidalPonds {
                left: 1,
                ..TidalPonds::default()
            }
        );
        assert_eq!(capped.data, original.data);
    }

    #[test]
    fn test_smooth_element_slopes_leaves_a_bed_within_the_bound_alone() {
        // 20 → 28 m over 4 elements: within the depth ratio everywhere
        let bound = ElementSlopeBound::for_pycnocline(19.0);
        let (mesh, ops, geom) = setup(distorted_mesh(4), 2);
        let mut bathy = Bathymetry2D::from_function(&mesh, &ops, &geom, |x, _| -20.0 - 8.0 * x);
        let original = bathy.clone();
        let report = bathy.smooth_element_slopes(&mesh, &ops, &geom, bound, 3.0);
        assert_eq!(report.elements_before, 0);
        assert_eq!(report.changed, 0);
        assert_eq!(bathy.data, original.data);
    }

    #[test]
    fn test_smooth_rx0_bounds_a_shoal_and_keeps_the_volume_and_the_shore() {
        let (r_max, min_depth) = (0.2, 3.0);
        for order in 1..=3 {
            let (mesh, ops, geom, mut bathy) = shoal_bed(order);
            let original = bathy.clone();
            let before = bathy.max_rx0(&mesh, &ops, &geom, min_depth).unwrap();
            let report = bathy.smooth_rx0(&mesh, &ops, &geom, r_max, min_depth);
            assert_eq!(report.before, before.value);
            assert!(report.before > 0.5, "N = {order}: {report:?}");
            assert!(
                report.after <= r_max * (1.0 + 1e-6),
                "N = {order}: {report:?}"
            );
            let after = bathy.max_rx0(&mesh, &ops, &geom, min_depth).unwrap();
            assert!((after.value - report.after).abs() < 1e-12);
            // The volume is kept, the bed stays continuous
            let (v0, v1) = (volume(&original, &geom), volume(&bathy, &geom));
            assert!(
                (v1 - v0).abs() < 1e-13 * v0.abs(),
                "N = {order}: {v0} vs {v1}"
            );
            assert_continuous(&bathy, &mesh, &ops, order);
            // The shoal is lowered, the shore untouched, and every node
            // that changed stays deeper than `min_depth`
            let mut change: f64 = 0.0;
            for (node, (&b, &b0)) in bathy.data.iter().zip(&original.data).enumerate() {
                change = change.max((b - b0).abs());
                if b != b0 {
                    assert!(
                        -b >= min_depth && -b0 >= min_depth,
                        "N = {order}: {b0} → {b}"
                    );
                }
                if b0 > -min_depth {
                    assert_eq!(b, b0, "N = {order}: node {node}");
                }
            }
            assert!((change - report.max_change).abs() < 1e-12);
            let top = original
                .data
                .iter()
                .copied()
                .fold(
                    f64::NEG_INFINITY,
                    |a, b| {
                        if b < -min_depth { a.max(b) } else { a }
                    },
                );
            assert!(top > -6.5, "N = {order}: top of the shoal {top}");
            let new_top = bathy
                .data
                .iter()
                .zip(&original.data)
                .filter(|&(_, &b0)| b0 < -min_depth)
                .map(|(&b, _)| b)
                .fold(f64::NEG_INFINITY, f64::max);
            assert!(new_top < top - 2.0, "N = {order}: {top} → {new_top}");
            // Gradients follow the new bed
            let mut fresh = bathy.clone();
            fresh.compute_gradients(&ops, &geom);
            assert_eq!(fresh.gradient_x, bathy.gradient_x);
            // A second pass has nothing to do
            let again = bathy.smooth_rx0(&mesh, &ops, &geom, r_max, min_depth);
            assert!(again.changed == 0 || again.max_change < 1e-6, "{again:?}");
        }
    }

    #[test]
    fn test_smooth_rx0_leaves_a_bed_within_the_bound_alone() {
        let (mesh, ops, geom, bathy) = shoal_bed(2);
        let bound = bathy.max_rx0(&mesh, &ops, &geom, 3.0).unwrap().value;
        let mut smoothed = bathy.clone();
        let report = smoothed.smooth_rx0(&mesh, &ops, &geom, bound, 3.0);
        assert_eq!((report.changed, report.sweeps), (0, 0));
        assert_eq!(smoothed.data, bathy.data);
    }
}

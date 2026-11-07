//! Real Gmsh meshes (TODO P1.3): the reader on files Gmsh wrote, and the 2D
//! solver on the unstructured, coastline-fitted quadrilaterals they hold.
//!
//! `tests/data/gmsh/` holds a small bay with a curved southern coastline and
//! an island, meshed all-quad by Gmsh 4.15 (`scripts/gmsh_fixtures.py`), as
//! MSH 4.1 ASCII, MSH 4.1 binary and MSH 2.2, plus a quad-dominant version
//! with triangles left in. Physical groups: `coast` (tag 11, the coastline and
//! the island), `open` (12, west and north), `outflow` (13, east).
//!
//! Checked here:
//! - the three formats load to the same mesh, the tags land on the right
//!   edges, and the triangles of the quad-dominant mesh are an error;
//! - lake at rest over a rough bed and across a shoreline, with walls on the
//!   coast and open boundaries elsewhere, is exact on these elements;
//! - a closed basin conserves mass through a run;
//! - raising the sea level outside fills the bay through the open boundaries
//!   (per-tag boundary dispatch from Gmsh physical groups).

use std::f64::consts::PI;
use std::path::PathBuf;

use dgcom::boundary::{CharacteristicOBC, MultiBoundaryCondition2D, Reflective2D, StillWater};
use dgcom::mesh::{Bathymetry2D, GmshError, read_gmsh_mesh};
use dgcom::time::{SSPRK3, TimeIntegrator};
use dgcom::types::ElementIndex;
use dgcom::{
    BoundaryTag, DGOperators2D, GeometricFactors2D, Mesh2D, SWE2DRhsConfig, SWEFormulation2D,
    SWESolution2D, SWEState2D, ShallowWater2D, compute_dt_swe_2d, compute_rhs_swe_2d,
};

const G: f64 = 9.81;
/// Bay extent and geometry (scripts/gmsh_fixtures.py)
const LX: f64 = 2000.0;
const LY: f64 = 1200.0;
const COAST_AMP: f64 = 150.0;
const ISLAND: (f64, f64, f64, f64) = (1200.0, 650.0, 220.0, 150.0);
const OUTFLOW: BoundaryTag = BoundaryTag::Custom(13);

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/gmsh")
        .join(name)
}

fn bay() -> Mesh2D {
    read_gmsh_mesh(&fixture("bay_quads_v41.msh")).expect("bay mesh")
}

fn edge_length(mesh: &Mesh2D, (a, b): (usize, usize)) -> f64 {
    let ([x0, y0], [x1, y1]) = (mesh.vertices[a], mesh.vertices[b]);
    (x1 - x0).hypot(y1 - y0)
}

fn on_island([x, y]: [f64; 2]) -> bool {
    let (cx, cy, a, b) = ISLAND;
    (((x - cx) / a).powi(2) + ((y - cy) / b).powi(2) - 1.0).abs() < 1e-6
}

#[test]
fn msh41_ascii_binary_and_msh22_load_to_the_same_mesh() {
    let ascii = bay();
    for name in ["bay_quads_v41_binary.msh", "bay_quads_v22.msh"] {
        let other = read_gmsh_mesh(&fixture(name)).unwrap();
        // Gmsh writes ASCII coordinates with 16 significant digits, binary
        // ones exactly
        assert_eq!(ascii.n_vertices, other.n_vertices, "{name}");
        for (a, b) in ascii.vertices.iter().zip(&other.vertices) {
            assert!(
                (a[0] - b[0]).abs() + (a[1] - b[1]).abs() < 1e-12 * LX,
                "{name}"
            );
        }
        assert_eq!(ascii.elements, other.elements, "{name}");
        assert_eq!(ascii.element_edges, other.element_edges, "{name}");
        for (e, f) in ascii.edges.iter().zip(&other.edges) {
            assert_eq!(
                (e.vertices, e.left, e.right, e.boundary_tag),
                (f.vertices, f.left, f.right, f.boundary_tag),
                "{name}"
            );
        }
    }
}

#[test]
fn bay_mesh_geometry_and_boundary_tags() {
    let mesh = bay();
    assert_eq!(mesh.n_elements, 192);

    // Unstructured: interior vertices that are not shared by four elements
    let interior: Vec<bool> = {
        let mut on_boundary = vec![false; mesh.n_vertices];
        for e in mesh.edges.iter().filter(|e| e.is_boundary()) {
            on_boundary[e.vertices.0] = true;
            on_boundary[e.vertices.1] = true;
        }
        on_boundary.iter().map(|b| !b).collect()
    };
    let valences: Vec<usize> = (0..mesh.n_vertices)
        .filter(|&v| interior[v])
        .map(|v| mesh.elements_at_vertex(v).len())
        .collect();
    assert!(valences.iter().any(|&n| n != 4), "{valences:?}");

    // General quadrilaterals, all with a positive Jacobian at every node
    let ops = DGOperators2D::new(3);
    let geom = GeometricFactors2D::compute(&mesh, &ops);
    assert!(!geom.is_affine());
    // The bay: rectangle − area under the coastline − island. The mesh
    // inscribes the curves (the island is convex, the coastline mostly so),
    // so it is within a percent.
    let area: f64 = geom.mass.iter().sum();
    let exact = LX * LY - 0.5 * COAST_AMP * LX - PI * ISLAND.2 * ISLAND.3;
    assert!((area - exact).abs() < 0.01 * exact, "{area} vs {exact}");

    // Every boundary edge is in a physical group, on the right curve
    let mut lengths = [0.0; 3];
    for edge in mesh.edges.iter().filter(|e| e.is_boundary()) {
        let (a, b) = (
            mesh.vertices[edge.vertices.0],
            mesh.vertices[edge.vertices.1],
        );
        let length = edge_length(&mesh, edge.vertices);
        match edge.boundary_tag {
            Some(BoundaryTag::Wall) => {
                let island = on_island(a) && on_island(b);
                let coast = a[1] <= COAST_AMP + 1.0 && b[1] <= COAST_AMP + 1.0;
                assert!(island || coast, "wall edge {a:?}–{b:?}");
                lengths[0] += length;
            }
            Some(BoundaryTag::Open) => {
                let west = a[0] == 0.0 && b[0] == 0.0;
                let north = a[1] == LY && b[1] == LY;
                assert!(west || north, "open edge {a:?}–{b:?}");
                lengths[1] += length;
            }
            Some(OUTFLOW) => {
                assert!(a[0] == LX && b[0] == LX, "outflow edge {a:?}–{b:?}");
                lengths[2] += length;
            }
            other => panic!("boundary edge {a:?}–{b:?} tagged {other:?}"),
        }
    }
    // Straight sides from y = 75 m (the coastline's end points) to LY
    let side = LY - 0.5 * COAST_AMP;
    assert!((lengths[1] - (LX + side)).abs() < 1e-9 * LX, "{lengths:?}");
    assert!((lengths[2] - side).abs() < 1e-9 * LX, "{lengths:?}");
    // Coastline (≥ LX) plus the island perimeter (Ramanujan), inscribed
    let (a, b) = (ISLAND.2, ISLAND.3);
    let h = ((a - b) / (a + b)).powi(2);
    let perimeter = PI * (a + b) * (1.0 + 3.0 * h / (10.0 + (4.0 - 3.0 * h).sqrt()));
    assert!(lengths[0] > LX + 0.97 * perimeter && lengths[0] < 1.1 * LX + perimeter);
}

#[test]
fn quad_dominant_mesh_is_rejected_with_advice() {
    match read_gmsh_mesh(&fixture("bay_quad_dominant_v41.msh")) {
        Err(error @ GmshError::UnsupportedElements(_)) => {
            let GmshError::UnsupportedElements(ref counts) = error else {
                unreachable!()
            };
            assert_eq!(counts, &vec![(2, 34)]);
            let message = error.to_string();
            assert!(message.contains("34 × 3-node triangle"), "{message}");
            assert!(
                message.contains("Mesh.SubdivisionAlgorithm = 1"),
                "{message}"
            );
        }
        other => panic!("expected UnsupportedElements, got {:?}", other.err()),
    }
}

/// The bay at order N with a nodal bed.
struct Bay {
    mesh: Mesh2D,
    ops: DGOperators2D,
    geom: GeometricFactors2D,
    bed: Bathymetry2D,
}

impl Bay {
    fn new(order: usize, bed: impl Fn(ElementIndex, f64, f64) -> f64) -> Self {
        let mesh = bay();
        let ops = DGOperators2D::new(order);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let mut bathymetry = Bathymetry2D::flat(mesh.n_elements, ops.n_nodes);
        let mut s = Self {
            mesh,
            ops,
            geom,
            bed: Bathymetry2D::flat(0, 0),
        };
        for (k, i) in s.nodes() {
            let [x, y] = s.xy(k, i);
            bathymetry.set(k, i, bed(k, x, y));
        }
        bathymetry.compute_gradients(&s.ops, &s.geom);
        s.bed = bathymetry;
        s
    }

    fn nodes(&self) -> impl Iterator<Item = (ElementIndex, usize)> + '_ {
        ElementIndex::iter(self.mesh.n_elements)
            .flat_map(|k| (0..self.ops.n_nodes).map(move |i| (k, i)))
    }

    fn xy(&self, k: ElementIndex, i: usize) -> [f64; 2] {
        self.mesh
            .reference_to_physical(k, self.ops.nodes_r[i], self.ops.nodes_s[i])
    }

    /// Water at level η wherever the bed is below it (dry above)
    fn at_level(&self, eta: impl Fn(f64, f64) -> f64) -> SWESolution2D {
        let mut q = SWESolution2D::new(self.mesh.n_elements, self.ops.n_nodes);
        for (k, i) in self.nodes() {
            let [x, y] = self.xy(k, i);
            let h = (eta(x, y) - self.bed.get(k, i)).max(0.0);
            q.set_state(k, i, SWEState2D::new(h, 0.0, 0.0));
        }
        q
    }

    fn mass(&self, q: &SWESolution2D) -> f64 {
        self.nodes()
            .map(|(k, i)| {
                self.geom.mass[self.geom.node_index(k.as_usize(), i)] * q.get_state(k, i).h
            })
            .sum()
    }

    fn mean_eta(&self, q: &SWESolution2D) -> f64 {
        let area: f64 = self.geom.mass.iter().sum();
        self.nodes()
            .map(|(k, i)| {
                let m = self.geom.mass[self.geom.node_index(k.as_usize(), i)];
                m * (q.get_state(k, i).h + self.bed.get(k, i))
            })
            .sum::<f64>()
            / area
    }
}

/// Walls on the coast, open boundaries (still water at `eta`) elsewhere.
fn run_config<'a>(
    equation: &'a ShallowWater2D,
    bc: &'a MultiBoundaryCondition2D<'a>,
    bay: &'a Bay,
    formulation: SWEFormulation2D,
    h_dry: f64,
) -> SWE2DRhsConfig<'a, MultiBoundaryCondition2D<'a>> {
    SWE2DRhsConfig::new(equation, bc)
        .with_coriolis(false)
        .with_formulation(formulation)
        .with_dry_threshold(h_dry)
        .with_bathymetry(&bay.bed)
}

/// Lake at rest over a rough bed (jumps at every face) and across a
/// shoreline, walls on the coast and still water at the lake level outside
/// the open boundaries: the RHS vanishes on the Gmsh elements.
#[test]
fn lake_at_rest_on_the_gmsh_bay() {
    let equation = ShallowWater2D::new(G);
    let wall = Reflective2D::new();
    let eta0 = 0.3;
    let outside = CharacteristicOBC::new(StillWater::at(eta0));
    let bc = MultiBoundaryCondition2D::new(&wall)
        .with_open(&outside)
        .with_custom(13, &outside);

    for order in 1..=3 {
        // Fully wet: 15–35 m with a jump per element
        let rough = Bay::new(order, |k, x, y| {
            let jump = ((7 * k.as_usize()) % 5) as f64 - 2.0;
            -25.0 + 8.0 * (2.0 * PI * x / LX).sin() * (PI * y / LY).cos() + 0.8 * jump
        });
        let q = rough.at_level(|_, _| eta0);
        let scale = G * 35.0 * 10.0 / 100.0;
        for formulation in [
            SWEFormulation2D::EntropyConservative,
            SWEFormulation2D::EntropyStable,
            SWEFormulation2D::WetDry,
        ] {
            let config = run_config(&equation, &bc, &rough, formulation, 1e-3);
            let rhs = compute_rhs_swe_2d(&q, &rough.mesh, &rough.ops, &rough.geom, &config, 0.0);
            println!(
                "P{order} {formulation:?}: max |rhs| = {:.2e}",
                rhs.max_abs()
            );
            assert!(
                rhs.max_abs() < 1e-11 * scale,
                "P{order} {formulation:?}: {:.3e}",
                rhs.max_abs()
            );
        }

        // Shoreline: the bed rises above the surface towards the coastline
        // and around the island
        let shore = Bay::new(order, |_, x, y| {
            let (cx, cy, a, b) = ISLAND;
            let r = (((x - cx) / a).powi(2) + ((y - cy) / b).powi(2)).sqrt();
            let coast = 2.0 - 0.02 * (y - 0.5 * COAST_AMP);
            let island = 2.0 - 12.0 * (r - 1.0);
            coast.max(island).max(-12.0)
        });
        let q = shore.at_level(|_, _| eta0);
        let n_dry = shore
            .nodes()
            .filter(|&(k, i)| q.get_state(k, i).h == 0.0)
            .count();
        assert!(n_dry > 0, "P{order}: the test needs dry nodes");
        let config = run_config(&equation, &bc, &shore, SWEFormulation2D::WetDry, 1e-3);
        let rhs = compute_rhs_swe_2d(&q, &shore.mesh, &shore.ops, &shore.geom, &config, 0.0);
        let scale = G * 12.0 * 12.0 / 100.0;
        println!(
            "P{order} shoreline ({n_dry} dry nodes): max |rhs| = {:.2e}",
            rhs.max_abs()
        );
        assert!(
            rhs.max_abs() < 1e-11 * scale,
            "P{order} shoreline: {:.3e}",
            rhs.max_abs()
        );
    }
}

/// Advance `q` over `t_end` with SSP-RK3 at a fixed dt from the CFL of `q`.
fn run(
    bay: &Bay,
    config: &SWE2DRhsConfig<'_, MultiBoundaryCondition2D<'_>>,
    q: &mut SWESolution2D,
    t_end: f64,
) {
    let equation = ShallowWater2D::new(G);
    let dt = compute_dt_swe_2d(q, &bay.mesh, &bay.geom, &equation, bay.ops.order, 0.4);
    let n_steps = (t_end / dt).ceil() as usize;
    let dt = t_end / n_steps as f64;
    let mut t = 0.0;
    for _ in 0..n_steps {
        SSPRK3.step(q, dt, t, |s, stage_t| {
            compute_rhs_swe_2d(s, &bay.mesh, &bay.ops, &bay.geom, config, stage_t)
        });
        t += dt;
    }
}

/// A hump released in the bay closed on all sides (walls on every tag) keeps
/// its mass to round-off.
#[test]
fn closed_bay_conserves_mass() {
    let equation = ShallowWater2D::new(G);
    let wall = Reflective2D::new();
    let bc = MultiBoundaryCondition2D::new(&wall);
    let bay = Bay::new(1, |_, x, _| -20.0 + 5.0 * x / LX);
    let mut q =
        bay.at_level(|x, y| 0.2 * (-((x - 500.0).powi(2) + (y - 700.0).powi(2)) / 1e5).exp());
    let mass0 = bay.mass(&q);
    for formulation in [SWEFormulation2D::Standard, SWEFormulation2D::WetDry] {
        let config = run_config(&equation, &bc, &bay, formulation, 1e-3);
        run(&bay, &config, &mut q, 100.0);
        let drift = (bay.mass(&q) - mass0).abs() / mass0;
        println!("{formulation:?}: relative mass drift {drift:.2e}");
        assert!(q.max_abs().is_finite(), "{formulation:?}: blew up");
        // Round-off accumulated over the steps
        assert!(
            drift < 1e-12,
            "{formulation:?}: relative mass drift {drift:.2e}"
        );
    }
}

/// Raising the sea outside by 10 cm fills the bay through the open and
/// outflow boundaries while the coast stays closed: the physical groups of
/// the Gmsh file dispatch the boundary conditions.
#[test]
fn raised_sea_fills_the_bay_through_the_open_boundaries() {
    let equation = ShallowWater2D::new(G);
    let wall = Reflective2D::new();
    let rise = 0.1;
    let outside = CharacteristicOBC::new(StillWater::at(rise));
    let bay = Bay::new(1, |_, _, _| -20.0);

    // Coast closed, the rest open: the bay fills to the outside level
    let bc = MultiBoundaryCondition2D::new(&wall)
        .with_open(&outside)
        .with_custom(13, &outside);
    let config = run_config(&equation, &bc, &bay, SWEFormulation2D::EntropyStable, 0.0);
    let mut q = bay.at_level(|_, _| 0.0);
    // A few crossings of the bay at √(g H) ≈ 14 m/s
    run(&bay, &config, &mut q, 500.0);
    let mean = bay.mean_eta(&q);
    println!("open: mean η {mean:.4} m after 500 s");
    assert!(q.max_abs().is_finite());
    assert!((mean - rise).abs() < 0.05 * rise, "mean η {mean:.4} m");

    // Only the outflow side open: it fills too, slower; with every tag
    // closed nothing enters
    let east_only = MultiBoundaryCondition2D::new(&wall).with_custom(13, &outside);
    let config = run_config(
        &equation,
        &east_only,
        &bay,
        SWEFormulation2D::EntropyStable,
        0.0,
    );
    let mut q = bay.at_level(|_, _| 0.0);
    run(&bay, &config, &mut q, 60.0);
    println!("outflow only: mean η {:.4} m after 60 s", bay.mean_eta(&q));
    assert!(bay.mean_eta(&q) > 0.01 * rise, "{}", bay.mean_eta(&q));

    let closed = MultiBoundaryCondition2D::new(&wall);
    let config = run_config(
        &equation,
        &closed,
        &bay,
        SWEFormulation2D::EntropyStable,
        0.0,
    );
    let mut q = bay.at_level(|_, _| 0.0);
    run(&bay, &config, &mut q, 60.0);
    assert!(bay.mean_eta(&q).abs() < 1e-12, "{}", bay.mean_eta(&q));
}

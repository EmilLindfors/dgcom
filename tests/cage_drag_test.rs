//! Gates for the drag of fish-farm net cages (`CageDrag2D`, TODO F.1, 2D).
//!
//! - The per-node footprint weights integrate to the cage area wherever the
//!   cage sits in the mesh, also on distorted quadrilaterals.
//! - Magnitude: a body force balanced by a cage over the whole domain gives
//!   the analytic speed u² = 2 G h / (C_d a min(d, h)).
//! - A porous band across a channel: the momentum-flux and pressure loss
//!   across the band equals the integrated drag minus the forcing, and the
//!   domain-integrated drag equals the domain-integrated forcing.
//! - Mass is conserved and a lake at rest over a rough bed with cages is
//!   exact.
//!
//! All runs use the production path: `Simulation` + `SSPRK3` + `SWEPhysics2D`.

use std::f64::consts::PI;
use std::sync::Arc;

use dgcom::boundary::Reflective2D;
use dgcom::equations::ShallowWater2D;
use dgcom::mesh::{Bathymetry2D, Mesh2D};
use dgcom::operators::{DGOperators2D, GeometricFactors2D};
use dgcom::physics::{PhysicsBuilder, SWEPhysics2DBuilder};
use dgcom::simulation::Simulation;
use dgcom::solver::{SWESolution2D, SWEState2D, WetDryConfig};
use dgcom::source::{
    CageDrag2D, CageFootprint, NetCage, SourceContext2D, SourceTerm2D, loland_drag_coefficient,
};
use dgcom::time::SSPRK3;
use dgcom::types::{Depth, ElementIndex};

const G: f64 = 9.81;

struct Setup {
    mesh: Arc<Mesh2D>,
    ops: Arc<DGOperators2D>,
    geom: Arc<GeometricFactors2D>,
}

impl Setup {
    fn new(mesh: Mesh2D, order: usize) -> Self {
        let ops = Arc::new(DGOperators2D::new(order));
        let geom = Arc::new(GeometricFactors2D::compute(&mesh, &ops));
        Self {
            mesh: Arc::new(mesh),
            ops,
            geom,
        }
    }

    fn builder(&self) -> SWEPhysics2DBuilder<Reflective2D> {
        PhysicsBuilder::swe_2d(
            self.mesh.clone(),
            self.ops.clone(),
            self.geom.clone(),
            ShallowWater2D::new(G),
            Reflective2D::new(),
        )
    }

    fn cages(&self, cages: &[NetCage]) -> CageDrag2D {
        CageDrag2D::new(&self.mesh, &self.ops, cages)
    }

    fn node_xy(&self, k: ElementIndex, i: usize) -> (f64, f64) {
        let [x, y] = self
            .mesh
            .reference_to_physical(k, self.ops.nodes_r[i], self.ops.nodes_s[i]);
        (x, y)
    }

    fn nodes(&self) -> impl Iterator<Item = (ElementIndex, usize)> + '_ {
        ElementIndex::iter(self.mesh.n_elements)
            .flat_map(|k| (0..self.ops.n_nodes).map(move |i| (k, i)))
    }

    fn fill(&self, f: impl Fn(f64, f64) -> SWEState2D) -> SWESolution2D {
        let mut q = SWESolution2D::new(self.mesh.n_elements, self.ops.n_nodes);
        for (k, i) in self.nodes() {
            let (x, y) = self.node_xy(k, i);
            q.set_state(k, i, f(x, y));
        }
        q
    }

    /// Quadrature weight w·J of global node `node`
    fn weight(&self, node: usize) -> f64 {
        let (k, i) = (node / self.ops.n_nodes, node % self.ops.n_nodes);
        self.ops.weights[i] * self.geom.jacobian(k, i)
    }

    fn mass(&self, q: &SWESolution2D) -> f64 {
        q.h_data()
            .iter()
            .enumerate()
            .map(|(j, &h)| self.weight(j) * h)
            .sum()
    }

    /// Σ wJ φ over the cage nodes, per unit ½C_d·a: the weighted footprint
    /// area
    fn weighted_area(&self, drag: &CageDrag2D, drag_per_length: f64) -> f64 {
        drag.nodes()
            .iter()
            .map(|n| self.weight(n.node) * n.coefficient / (0.5 * drag_per_length))
            .sum()
    }

    /// x-momentum removed by the cages per unit time, ∫ ½C_d a min(d, h) |u| u dA
    fn integrated_drag(&self, drag: &CageDrag2D, q: &SWESolution2D) -> f64 {
        drag.nodes()
            .iter()
            .map(|n| {
                let (h, hu, hv) = (q.h_data()[n.node], q.hu_data()[n.node], q.hv_data()[n.node]);
                let speed = (hu * hu + hv * hv).sqrt() / h;
                self.weight(n.node) * n.coefficient * n.net_depth.min(h) * speed * hu / h
            })
            .sum()
    }
}

/// Uniform body force (0, h·G_x, 0): a mean surface slope driving the flow
/// through a periodic channel.
struct BodyForce(f64);

impl SourceTerm2D for BodyForce {
    fn evaluate(&self, ctx: &SourceContext2D) -> SWEState2D {
        SWEState2D::new(0.0, ctx.state.h * self.0, 0.0)
    }

    fn name(&self) -> &'static str {
        "body_force"
    }
}

fn distorted_mesh(l: f64, n: usize) -> Mesh2D {
    let mut mesh = Mesh2D::uniform_periodic(0.0, l, 0.0, l, n, n);
    let tau = 2.0 * PI / l;
    let a = 0.04 * l;
    for v in &mut mesh.vertices {
        let [x, y] = *v;
        let (sx, sy) = ((tau * x).sin(), (tau * y).sin());
        *v = [
            x + a * sx * sy + 0.5 * a * (tau * y).sin(),
            y - 0.7 * a * sx * sy + 0.4 * a * (tau * x).sin(),
        ];
    }
    mesh
}

// ---------------------------------------------------------------------------
// Footprint weighting
// ---------------------------------------------------------------------------

#[test]
fn footprint_weights_integrate_to_the_cage_area() {
    let radius = 25.0;
    let area = PI * radius * radius;
    let square = CageFootprint::Polygon(vec![
        [310.0, 120.0],
        [350.0, 150.0],
        [320.0, 190.0],
        [280.0, 160.0],
    ]);
    for order in 1..=4 {
        // 40 m elements: the cage spans a few elements, at several offsets
        // against the element edges
        let setup = Setup::new(
            Mesh2D::uniform_periodic(0.0, 400.0, 0.0, 400.0, 10, 10),
            order,
        );
        for center in [[200.0, 200.0], [213.7, 187.1], [180.0, 240.0]] {
            let cage = NetCage::circular(center, radius, 20.0, 0.25);
            let drag = setup.cages(std::slice::from_ref(&cage));
            let weighted = setup.weighted_area(&drag, cage.drag_per_length);
            assert!(
                (weighted / area - 1.0).abs() < 5e-5,
                "P{order} at {center:?}: {weighted} vs {area}"
            );
            assert!(drag.nodes().iter().all(|n| n.coefficient > 0.0
                && n.coefficient <= 0.5 * cage.drag_per_length * (1.0 + 1e-12)));
        }
        let cage = NetCage::new(square.clone(), 15.0, 0.02);
        let weighted = setup.weighted_area(&setup.cages(std::slice::from_ref(&cage)), 0.02);
        assert!(
            (weighted / square.area() - 1.0).abs() < 5e-5,
            "P{order} square: {weighted} vs {}",
            square.area()
        );
    }
}

#[test]
fn footprint_weights_on_distorted_quadrilaterals() {
    // The weights are normalised by wJ, so the quadrature gives the footprint
    // area on any quadrilateral, not only on parallelograms
    let setup_area = |order| {
        let setup = Setup::new(distorted_mesh(400.0, 10), order);
        let cage = NetCage::circular([190.0, 215.0], 25.0, 20.0, 0.25);
        let drag = setup.cages(std::slice::from_ref(&cage));
        setup.weighted_area(&drag, cage.drag_per_length) / (PI * 625.0) - 1.0
    };
    for order in 1..=4 {
        let error = setup_area(order);
        assert!(error.abs() < 5e-5, "P{order}: relative error {error:.2e}");
    }
}

// ---------------------------------------------------------------------------
// Magnitude
// ---------------------------------------------------------------------------

#[test]
fn body_force_balanced_by_cage_drag_gives_the_analytic_speed() {
    // A cage over the whole domain: uniform flow with ∂(hu)/∂t = hG −
    // ½C_d a min(d, h)|u|u, steady at u² = 2 G h / (C_d a min(d, h))
    let (h, forcing) = (30.0, 1e-4);
    let setup = Setup::new(
        Mesh2D::uniform_periodic(0.0, 10_000.0, 0.0, 10_000.0, 2, 2),
        2,
    );
    let everywhere = CageFootprint::Polygon(vec![
        [-1.0, -1.0],
        [10_001.0, -1.0],
        [10_001.0, 10_001.0],
        [-1.0, 10_001.0],
    ]);
    for net_depth in [15.0, 45.0] {
        let cage = NetCage::new(everywhere.clone(), net_depth, 0.021);
        let drag = setup.cages(std::slice::from_ref(&cage));
        let physics = setup
            .builder()
            .with_source(BodyForce(forcing))
            .with_cage_drag(drag)
            .build();
        let mut q = setup.fill(|_, _| SWEState2D::new(h, 0.0, 0.0));
        let result = Simulation::new(physics, SSPRK3).run(&mut q, 0.0, 40_000.0);
        assert!(result.success, "{result:?}");

        let exact = (2.0 * forcing * h / (0.021 * net_depth.min(h))).sqrt();
        for (&hq, &hu) in q.h_data().iter().zip(q.hu_data()) {
            let u = hu / hq;
            assert!(
                (u / exact - 1.0).abs() < 1e-6,
                "d = {net_depth}: u = {u}, exact {exact}"
            );
        }
        assert!(q.hv_data().iter().all(|&m| m.abs() < 1e-12));
    }
}

#[test]
fn decay_of_uniform_flow_through_a_cage_converges_to_exact() {
    // No forcing: d|u|/dt = −k|u|² with k = ½C_d a min(d, h)/h, so
    // u = u0 / (1 + k u0 t). The point-implicit drag is first order.
    let (h, u0, net_depth, cd_a, t_end) = (10.0_f64, 1.0, 5.0, 0.4, 20.0);
    let k = 0.5 * cd_a * net_depth / h;
    let exact = u0 / (1.0 + k * u0 * t_end);
    let setup = Setup::new(Mesh2D::uniform_periodic(0.0, 1000.0, 0.0, 1000.0, 2, 2), 1);
    let everywhere = CageFootprint::Polygon(vec![
        [-1.0, -1.0],
        [1001.0, -1.0],
        [1001.0, 1001.0],
        [-1.0, 1001.0],
    ]);
    let error = |dt: f64| {
        let drag = setup.cages(&[NetCage::new(everywhere.clone(), net_depth, cd_a)]);
        let physics = setup.builder().with_cage_drag(drag).build();
        let mut q = setup.fill(|_, _| SWEState2D::from_primitives(h, u0, 0.0));
        Simulation::new(physics, SSPRK3)
            .with_dt_max(dt)
            .run(&mut q, 0.0, t_end);
        (q.hu_data()[0] / h - exact).abs()
    };
    let (coarse, fine) = (error(0.2), error(0.1));
    let rate = (coarse / fine).log2();
    assert!(
        fine < 2e-3 * exact && rate > 0.9,
        "errors {coarse:.2e}, {fine:.2e}, rate {rate:.2}"
    );
}

// ---------------------------------------------------------------------------
// Porous band across a channel
// ---------------------------------------------------------------------------

#[test]
fn drag_of_a_porous_band_equals_the_momentum_flux_and_pressure_loss() {
    // Periodic channel driven by a body force G, with a porous band over
    // 4 km ≤ x ≤ 6 km on element edges. At steady state (1D, q = hu constant):
    //   [h u² + ½ g h²]_4km − [h u² + ½ g h²]_6km = ∫_band (drag − h G) dx
    // and over the whole domain ∫ h G dA = ∫ drag dA.
    let (l, width, h0, forcing) = (10_000.0, 1000.0, 30.0, 1e-4);
    let (x_in, x_out) = (4000.0, 6000.0);
    let order = 2;
    let setup = Setup::new(Mesh2D::uniform_periodic(0.0, l, 0.0, width, 20, 2), order);
    let band = CageFootprint::Polygon(vec![
        [x_in, -1.0],
        [x_out, -1.0],
        [x_out, width + 1.0],
        [x_in, width + 1.0],
    ]);
    let drag = setup.cages(&[NetCage::new(band, 15.0, 0.021)]);
    let physics = setup
        .builder()
        .with_source(BodyForce(forcing))
        .with_cage_drag(drag.clone())
        .build();
    let mut q = setup.fill(|_, _| SWEState2D::new(h0, 0.0, 0.0));
    let mass0 = setup.mass(&q);
    let result = Simulation::new(physics, SSPRK3).run(&mut q, 0.0, 60_000.0);
    assert!(result.success, "{result:?}");
    let mass_error = setup.mass(&q) / mass0 - 1.0;
    // Round-off accumulated over ≈ 15k steps
    assert!(mass_error.abs() < 1e-11, "mass error {mass_error:.2e}");

    // Global balance
    let forcing_total: f64 = q
        .h_data()
        .iter()
        .enumerate()
        .map(|(j, &h)| setup.weight(j) * h * forcing)
        .sum();
    let drag_total = setup.integrated_drag(&drag, &q);
    assert!(
        (drag_total / forcing_total - 1.0).abs() < 1e-4,
        "drag {drag_total:.6e} vs forcing {forcing_total:.6e}"
    );

    // Momentum budget of the band: momentum flux + pressure at its faces,
    // from the mean of the two traces at the face nodes
    let flux_at = |x_face: f64| {
        let (mut sum, mut count) = (0.0, 0);
        for (k, i) in setup.nodes() {
            let (x, _) = setup.node_xy(k, i);
            if (x - x_face).abs() < 1e-6 {
                let s = q.get_state(k, i);
                sum += s.hu * s.hu / s.h + 0.5 * G * s.h * s.h;
                count += 1;
            }
        }
        sum / count as f64
    };
    let band_source: f64 = setup
        .nodes()
        // Nodes of the elements inside the band (by their centre)
        .filter(|&(k, _)| (x_in..x_out).contains(&setup.mesh.reference_to_physical(k, 0.0, 0.0)[0]))
        .map(|(k, i)| {
            let j = k.as_usize() * setup.ops.n_nodes + i;
            -setup.weight(j) * q.h_data()[j] * forcing
        })
        .sum::<f64>()
        + drag_total;
    let loss = flux_at(x_in) - flux_at(x_out);
    let expected = band_source / width;
    assert!(
        loss > 0.0 && (loss / expected - 1.0).abs() < 1e-3,
        "flux loss {loss:.6e} vs (drag − forcing) {expected:.6e}"
    );

    // The drag holds the current well below the free acceleration and raises
    // the surface upstream of the band
    let eta = |x_face: f64| {
        let (mut sum, mut n) = (0.0, 0);
        for (k, i) in setup.nodes() {
            if (setup.node_xy(k, i).0 - x_face).abs() < 1e-6 {
                sum += q.get_state(k, i).h;
                n += 1;
            }
        }
        sum / n as f64
    };
    assert!(eta(x_in) > eta(x_out), "{} {}", eta(x_in), eta(x_out));
    let speed = q.hu_data()[0] / q.h_data()[0];
    let expected_speed = (5.0 * 2.0 * forcing * h0 / (0.021 * 15.0)).sqrt();
    assert!(
        (speed / expected_speed - 1.0).abs() < 0.02,
        "u = {speed}, drag balance {expected_speed}"
    );
}

// ---------------------------------------------------------------------------
// Conservation and lake at rest
// ---------------------------------------------------------------------------

#[test]
fn lake_at_rest_over_a_rough_bed_with_cages_is_exact() {
    let setup = Setup::new(distorted_mesh(400.0, 8), 3);
    let bed = |x: f64, y: f64| -20.0 + 6.0 * (0.03 * x).sin() * (0.02 * y).cos();
    let bathymetry = Bathymetry2D::from_function(&setup.mesh, &setup.ops, &setup.geom, bed);
    let cages = [
        NetCage::circular([150.0, 180.0], 25.0, 15.0, 0.3),
        NetCage::circular([180.0, 200.0], 25.0, 30.0, 0.3),
    ];
    let physics = setup
        .builder()
        .with_bathymetry(Arc::new(bathymetry))
        .with_wet_dry(WetDryConfig::new(Depth::new(1e-3), G))
        .with_cage_drag(setup.cages(&cages))
        .build();
    let mut q = setup.fill(|x, y| SWEState2D::new(-bed(x, y), 0.0, 0.0));
    let q0 = q.clone();
    let result = Simulation::new(physics, SSPRK3).run(&mut q, 0.0, 200.0);
    assert!(result.success, "{result:?}");
    for j in 0..q.h_data().len() {
        assert!((q.h_data()[j] - q0.h_data()[j]).abs() < 1e-11);
        assert!(q.hu_data()[j].abs() < 1e-10 && q.hv_data()[j].abs() < 1e-10);
    }
}

#[test]
fn cages_conserve_mass_and_only_remove_momentum() {
    // A current through two overlapping cages on a distorted mesh
    let setup = Setup::new(distorted_mesh(400.0, 8), 2);
    let cages = [
        NetCage::circular([200.0, 200.0], 25.0, 10.0, 0.3),
        NetCage::new(
            CageFootprint::Polygon(vec![
                [190.0, 190.0],
                [240.0, 190.0],
                [240.0, 230.0],
                [190.0, 230.0],
            ]),
            20.0,
            0.05,
        ),
    ];
    let initial = setup.fill(|x, _| {
        SWEState2D::from_primitives(20.0 + 0.1 * (2.0 * PI * x / 400.0).sin(), 0.5, 0.1)
    });
    let run = |with_cages: bool| {
        let mut builder = setup.builder();
        if with_cages {
            builder = builder.with_cage_drag(setup.cages(&cages));
        }
        let mut q = initial.clone();
        let result = Simulation::new(builder.build(), SSPRK3).run(&mut q, 0.0, 300.0);
        assert!(result.success, "{result:?}");
        q
    };
    let (free, caged) = (run(false), run(true));
    let mass0 = setup.mass(&initial);
    assert!((setup.mass(&caged) / mass0 - 1.0).abs() < 1e-13);
    let momentum = |q: &SWESolution2D| -> f64 {
        q.hu_data()
            .iter()
            .enumerate()
            .map(|(j, &m)| setup.weight(j) * m)
            .sum()
    };
    // Periodic and without other sources, only the cages change the momentum
    let lost = momentum(&free) - momentum(&caged);
    assert!(
        lost > 0.01 * momentum(&free),
        "lost {lost} of {}",
        momentum(&free)
    );
}

#[test]
fn circular_cage_default_coefficients() {
    // Løland at normal incidence for S_n = 0.25 and a = 4/(πR)
    let cage = NetCage::circular([0.0, 0.0], 25.0, 20.0, 0.25);
    let expected = loland_drag_coefficient(0.25, 0.0) * 4.0 / (PI * 25.0);
    assert!((cage.drag_per_length - expected).abs() < 1e-15);
}

#[cfg(feature = "parallel")]
#[test]
fn parallel_cage_drag_matches_serial() {
    use dgcom::solver::{
        ImplicitDamping2D, apply_implicit_damping_2d, apply_implicit_damping_2d_parallel,
    };
    // > 4096 nodes so that the cage entries straddle parallel chunks
    let setup = Setup::new(
        Mesh2D::uniform_periodic(0.0, 1000.0, 0.0, 1000.0, 30, 30),
        2,
    );
    // One cage centred on the element holding the first node of the second
    // chunk (4096), plus a few elsewhere
    let chunk_element = ElementIndex::new(4096 / setup.ops.n_nodes);
    let [xc, yc] = setup.mesh.reference_to_physical(chunk_element, 0.0, 0.0);
    let mut cages = vec![NetCage::circular([xc, yc], 30.0, 15.0, 0.3)];
    cages.extend(
        (0..6).map(|c| NetCage::circular([150.0 * c as f64 + 60.0, 800.0], 30.0, 15.0, 0.3)),
    );
    let drag = setup.cages(&cages);
    let nodes: Vec<_> = drag.nodes().iter().map(|n| n.node).collect();
    assert!(nodes.iter().any(|&n| n < 4096) && nodes.iter().any(|&n| n >= 4096));
    assert!(
        nodes.contains(&4095) || nodes.contains(&4096),
        "cage does not reach the chunk edge"
    );
    let from = setup
        .fill(|x, y| SWEState2D::from_primitives(20.0 + 0.001 * x, 0.5, 0.2 * (y / 100.0).sin()));
    let damping = ImplicitDamping2D {
        friction: None,
        cages: Some(&drag),
        wet_dry: None,
        h_min: Depth::new(1e-6),
    };
    let (mut serial, mut parallel) = (from.clone(), from.clone());
    apply_implicit_damping_2d(&mut serial, &from, 5.0, &damping);
    apply_implicit_damping_2d_parallel(&mut parallel, &from, 5.0, &damping);
    assert_eq!(serial.data, parallel.data);
    assert_ne!(serial.data, from.data);
}

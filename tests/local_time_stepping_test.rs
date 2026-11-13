//! Gates for local time stepping (`MultirateSSPRK3`, `MultirateSSPRK43`, TODO
//! P2.5, P2.1). Every multirate gate runs on both base schemes.
//!
//! - One level is the base scheme bit for bit (SSP-RK3, SSP-RK(4,3)), with
//!   wetting/drying, implicit friction and a time-dependent source.
//! - The element-subset RHS gives the selected elements exactly the full RHS.
//! - Mass is conserved to rounding across level interfaces, and a lake at
//!   rest over a rough bed with dry land stays at rest.
//! - Stage times: a time-dependent body force on a uniform state is
//!   integrated to the accuracy of each element's own substeps.
//! - Temporal convergence to the global SSP-RK3 solution at (at least)
//!   second order.
//! - A wave running up a beach across several levels keeps h ≥ 0 without a
//!   single negative-mean clip, and conserves mass; SSP-RK(4,3) at twice
//!   SSP-RK3's positivity-capped step. The film of the first flooded element
//!   does not stretch the coarse step (the run to 500 s).
//! - Line-local elements: a stage value reads exactly the nodes the stage
//!   rates are spread by (one node of the input perturbed at a time, bit for
//!   bit), and a run on node-by-node rates is the run on whole-element rates
//!   bit for bit, with fewer evaluations.
//! - With horizontal viscosity (a two-hop stencil): one level is still
//!   SSP-RK3 bit for bit; several levels conserve mass and momentum to
//!   rounding and converge to the global solution.
//!
//! All runs use the production path: `Simulation` + `SWEPhysics2D`.

use std::sync::Arc;

use dgcom::boundary::Reflective2D;
use dgcom::equations::ShallowWater2D;
use dgcom::mesh::{Bathymetry2D, Mesh2D};
use dgcom::operators::{DGOperators2D, GeometricFactors2D};
use dgcom::physics::{PhysicsBuilder, PhysicsModule, SWEPhysics2D, SWEPhysics2DBuilder};
use dgcom::simulation::Simulation;
use dgcom::solver::{
    KuzminParameter2D, SWEFormulation2D, SWESolution2D, SWEState2D, StandardLimiter2D, WetDryConfig,
};
use dgcom::source::{HorizontalViscosity2D, ManningFriction2D, SourceContext2D, SourceTerm2D};
use dgcom::time::{
    ElementStage, IntegratorInfo, Multirate, MultirateSSPRK3, MultirateStepper, SSPRK3, SspScheme,
    StandardIntegrator,
};
use dgcom::types::{Depth, ElementIndex};

const G: f64 = 9.81;

/// The base schemes of local time stepping.
const BASES: [StandardIntegrator; 2] = [StandardIntegrator::SSPRK3, StandardIntegrator::SSPRK43];

fn multirate(base: StandardIntegrator, max_levels: usize) -> Multirate<StandardIntegrator> {
    Multirate::with_base(base, max_levels)
}

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

    fn bathymetry(&self, bed: impl Fn(f64, f64) -> f64) -> Arc<Bathymetry2D> {
        Arc::new(Bathymetry2D::from_function(
            &self.mesh, &self.ops, &self.geom, bed,
        ))
    }

    fn node_xy(&self, k: ElementIndex, i: usize) -> (f64, f64) {
        let [x, y] = self
            .mesh
            .reference_to_physical(k, self.ops.nodes_r[i], self.ops.nodes_s[i]);
        (x, y)
    }

    fn fill(&self, f: impl Fn(f64, f64) -> SWEState2D) -> SWESolution2D {
        let mut q = SWESolution2D::new(self.mesh.n_elements, self.ops.n_nodes);
        for k in ElementIndex::iter(self.mesh.n_elements) {
            for i in 0..self.ops.n_nodes {
                let (x, y) = self.node_xy(k, i);
                q.set_state(k, i, f(x, y));
            }
        }
        q
    }

    /// Quadrature weight w·J of global node `node`
    fn weight(&self, node: usize) -> f64 {
        let (k, i) = (node / self.ops.n_nodes, node % self.ops.n_nodes);
        self.ops.weights[i] * self.geom.jacobian(k, i)
    }

    fn integral(&self, values: &[f64]) -> f64 {
        values
            .iter()
            .enumerate()
            .map(|(j, &v)| self.weight(j) * v)
            .sum()
    }

    fn mass(&self, q: &SWESolution2D) -> f64 {
        self.integral(q.h_data())
    }

    /// L2 norm of the difference of all three fields
    fn l2_difference(&self, a: &SWESolution2D, b: &SWESolution2D) -> f64 {
        let mut sum = 0.0;
        for var in 0..3 {
            let d: Vec<f64> = a.data[var]
                .iter()
                .zip(&b.data[var])
                .map(|(x, y)| (x - y) * (x - y))
                .collect();
            sum += self.integral(&d);
        }
        sum.sqrt()
    }
}

/// `nx × ny` elements over `[0, lx] × [0, ly]`, the columns growing
/// geometrically by `ratio` from left to right.
fn graded_mesh(lx: f64, ly: f64, nx: usize, ny: usize, ratio: f64) -> Mesh2D {
    let mut mesh = Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, ly, nx, ny);
    let n = nx as f64;
    for v in &mut mesh.vertices {
        v[0] = lx * (ratio.powf(n * v[0]) - 1.0) / (ratio.powf(n) - 1.0);
    }
    mesh
}

/// Body force (0, 0, h·A·cos ωt): uniform in space, periodic in time.
struct OscillatingForce {
    amplitude: f64,
    omega: f64,
}

impl SourceTerm2D for OscillatingForce {
    fn evaluate(&self, ctx: &SourceContext2D) -> SWEState2D {
        let force = self.amplitude * (self.omega * ctx.time).cos();
        SWEState2D::new(0.0, 0.0, ctx.state.h * force)
    }

    fn is_node_local(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "oscillating_force"
    }
}

fn gaussian(x: f64, y: f64, (x0, y0): (f64, f64), width: f64) -> f64 {
    (-((x - x0).powi(2) + (y - y0).powi(2)) / (width * width)).exp()
}

/// Sea surface `eta` over `bed`, at rest, dry (h = 0) where the bed is above.
fn at_rest(
    bed: impl Fn(f64, f64) -> f64,
    eta: impl Fn(f64, f64) -> f64,
) -> impl Fn(f64, f64) -> SWEState2D {
    move |x, y| SWEState2D::new((eta(x, y) - bed(x, y)).max(0.0), 0.0, 0.0)
}

fn max_speed(q: &SWESolution2D) -> f64 {
    q.h_data()
        .iter()
        .zip(q.hu_data().iter().zip(q.hv_data()))
        .filter(|&(&h, _)| h > 1e-3)
        .map(|(&h, (&hu, &hv))| (hu * hu + hv * hv).sqrt() / h)
        .fold(0.0, f64::max)
}

// =============================================================================
// One level is SSP-RK3
// =============================================================================

/// A shoreline run with every stage component: WetDry split form, the
/// positivity limiter and wet/dry correction, implicit Manning friction and a
/// time-dependent source.
fn shoreline_physics(setup: &Setup) -> SWEPhysics2D<Reflective2D> {
    shoreline_builder(setup).build()
}

fn shoreline_builder(setup: &Setup) -> SWEPhysics2DBuilder<Reflective2D> {
    let bed = |x: f64, y: f64| -2.0 + 3.0 * x / 1000.0 + 0.3 * (y / 150.0).sin();
    setup
        .builder()
        .with_bathymetry(setup.bathymetry(bed))
        .with_limiter(StandardLimiter2D::Positivity(WetDryConfig::DEFAULT_H_DRY))
        .with_wet_dry(WetDryConfig::default())
        .with_implicit_friction(ManningFriction2D::new(G, 0.03))
        .with_source(OscillatingForce {
            amplitude: 1e-3,
            omega: 2.0 * std::f64::consts::PI / 600.0,
        })
}

fn shoreline_state(setup: &Setup) -> SWESolution2D {
    let bed = |x: f64, y: f64| -2.0 + 3.0 * x / 1000.0 + 0.3 * (y / 150.0).sin();
    setup.fill(at_rest(bed, |x, y| {
        0.2 * gaussian(x, y, (300.0, 250.0), 120.0)
    }))
}

#[test]
fn one_level_is_ssp_rk3_bit_for_bit() {
    one_level_matches_global(StandardIntegrator::SSPRK3, None);
}

#[test]
fn one_level_with_viscosity_is_ssp_rk3_bit_for_bit() {
    one_level_matches_global(
        StandardIntegrator::SSPRK3,
        Some(HorizontalViscosity2D::constant(5.0)),
    );
}

#[test]
fn one_level_is_ssp_rk43_bit_for_bit() {
    one_level_matches_global(StandardIntegrator::SSPRK43, None);
    one_level_matches_global(
        StandardIntegrator::SSPRK43,
        Some(HorizontalViscosity2D::constant(5.0)),
    );
}

/// One level of the multirate stepper, and `Simulation`'s fused stages, give
/// the whole-state steps of `base` bit for bit.
fn one_level_matches_global(base: StandardIntegrator, viscosity: Option<HorizontalViscosity2D>) {
    let setup = Setup::new(Mesh2D::uniform_rectangle(0.0, 1000.0, 0.0, 500.0, 8, 4), 2);
    let q0 = shoreline_state(&setup);
    let physics = || {
        let builder = shoreline_builder(&setup);
        match &viscosity {
            Some(v) => builder.with_viscosity(*v),
            None => builder,
        }
        .build()
    };
    let dt = 0.2 * physics().compute_dt(&q0, 1.0);
    let run = |sim_q: &mut SWESolution2D, levels: Option<usize>| {
        let physics = physics();
        let result = match levels {
            // The whole-state stages: `Simulation` otherwise runs the base
            // scheme through the fused ones too
            None => Simulation::new(physics, base)
                .with_fused_stages(false)
                .with_dt_max(dt)
                .run(sim_q, 3.0, 3.0 + 40.0 * dt),
            Some(l) => Simulation::new(physics, multirate(base, l))
                .with_dt_max(dt)
                .run(sim_q, 3.0, 3.0 + 40.0 * dt),
        };
        assert!(result.success);
        result
    };

    let mut global = q0.clone();
    run(&mut global, None);
    // Zero levels, and levels allowed but not usable (dt_max below every
    // element's time step)
    for levels in [0, 6] {
        let mut local = q0.clone();
        let result = run(&mut local, Some(levels));
        let stats = result.local_time_stepping.expect("multirate stats");
        assert_eq!(stats.finest_level, 0);
        assert_eq!(stats.speedup(), 1.0);
        for var in 0..3 {
            assert!(
                global.data[var] == local.data[var],
                "{}, levels = {levels}: variable {var} differs from the base",
                base.name()
            );
        }
    }
    // `Simulation` runs the base scheme through the fused stages by default
    let mut fused = q0.clone();
    let result =
        Simulation::new(physics(), base)
            .with_dt_max(dt)
            .run(&mut fused, 3.0, 3.0 + 40.0 * dt);
    assert!(result.success && result.local_time_stepping.is_none());
    for var in 0..3 {
        assert!(
            global.data[var] == fused.data[var],
            "fused {}: variable {var} differs from the whole-state stages",
            base.name()
        );
    }
    assert!(
        global.h_data() != q0.h_data(),
        "the run must actually do something"
    );
}

/// The element-subset RHS (the kernel of local time stepping) writes exactly
/// the full RHS into the selected rows and leaves the others alone,
/// including dry, shoreline and boundary elements.
#[test]
fn subset_rhs_matches_the_full_rhs() {
    use dgcom::physics::PhysicsModule;
    use dgcom::solver::{SWE2DRhsConfig, compute_rhs_swe_2d_into, compute_rhs_swe_2d_where_into};

    let setup = Setup::new(Mesh2D::uniform_rectangle(0.0, 1000.0, 0.0, 500.0, 8, 4), 2);
    let mut q = shoreline_state(&setup);
    // Some motion
    let [h, hu, _] = &mut q.data;
    for (hu, &h) in hu.iter_mut().zip(h.iter()) {
        *hu = 0.3 * h;
    }
    let t = 17.0;

    // The shoreline configuration: WetDry split form with a dry region, wall
    // boundaries and a time-dependent source
    let equation = ShallowWater2D::new(G);
    let wall = Reflective2D::new();
    let bed = setup.bathymetry(|x, y| -2.0 + 3.0 * x / 1000.0 + 0.3 * (y / 150.0).sin());
    let force = OscillatingForce {
        amplitude: 1e-3,
        omega: 0.01,
    };
    let config = SWE2DRhsConfig::new(&equation, &wall)
        .with_formulation(SWEFormulation2D::WetDry)
        .with_bathymetry(&bed)
        .with_dry_threshold(WetDryConfig::DEFAULT_H_DRY)
        .with_source_terms(&force);
    let (mesh, ops, geom) = (&setup.mesh, &setup.ops, &setup.geom);
    let mut full = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
    compute_rhs_swe_2d_into(&q, mesh, ops, geom, &config, t, &mut full);

    let sentinel = 12345.0;
    let mut out = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
    for data in &mut out.data {
        data.fill(sentinel);
    }
    let selected = |k: usize| k % 3 != 1;
    compute_rhs_swe_2d_where_into(
        &q,
        mesh,
        ops,
        geom,
        &config,
        &|k| selected(k).then_some(t),
        &mut out,
    );
    for k in ElementIndex::iter(mesh.n_elements) {
        for var in 0..3 {
            let got = out.element_var(k, var);
            if selected(k.as_usize()) {
                assert_eq!(
                    got,
                    full.element_var(k, var),
                    "element {k:?}, variable {var}"
                );
            } else {
                assert!(got.iter().all(|&v| v == sentinel));
            }
        }
    }

    // Element time steps are bounded by the global one
    let physics = shoreline_physics(&setup);
    let local = physics.local_time_stepping().expect("SWE supports LTS");
    let mut element_dt = vec![0.0; setup.mesh.n_elements];
    local.element_dt(&q, 0.5, SspScheme::Rk3, &mut element_dt);
    let global_dt = physics.compute_dt(&q, 0.5);
    let min_dt = element_dt.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(min_dt <= global_dt, "{min_dt} > {global_dt}");
    // The global step of the wet/dry run is the smallest element step
    assert_eq!(
        min_dt,
        physics.compute_dt_ssp(&q, 0.5, Some(SspScheme::Rk3))
    );
    assert!(
        min_dt > 0.5 * global_dt,
        "neighbour traces should matter little here"
    );
}

// =============================================================================
// Several levels
// =============================================================================

/// 16 columns growing by 1.25 (28× from the first to the last) over a bed
/// from 20 to 110 m: levels from both size and depth.
fn graded_setup() -> Setup {
    Setup::new(graded_mesh(10_000.0, 2_000.0, 16, 3, 1.25), 2)
}

fn graded_bed(x: f64, y: f64) -> f64 {
    -(20.0 + 90.0 * x / 10_000.0) - 5.0 * (2.0 * std::f64::consts::PI * y / 2_000.0).cos()
}

#[test]
fn several_levels_conserve_mass() {
    let setup = graded_setup();
    for base in BASES {
        let physics = setup
            .builder()
            .with_bathymetry(setup.bathymetry(graded_bed))
            .build();
        let mut q = setup.fill(at_rest(graded_bed, |x, y| {
            0.5 * gaussian(x, y, (2_000.0, 1_000.0), 800.0)
        }));
        let mass0 = setup.mass(&q);

        let sim = Simulation::new(physics, multirate(base, 8)).with_cfl(0.5);
        let result = sim.run(&mut q, 0.0, 400.0);
        assert!(result.success);
        let stats = result.local_time_stepping.unwrap();
        println!(
            "{}: levels up to {}, {:.2}x less RHS work, {} coarse steps",
            base.name(),
            stats.finest_level,
            stats.speedup(),
            stats.steps
        );
        // Levels 0–2. (Before the builder applied the bed slope by default,
        // the bed pushed nothing: the "lake" ran downhill in h, and the faster
        // flow made a third level.)
        assert!(stats.finest_level >= 2, "{stats:?}");
        assert!(stats.speedup() > 1.3, "{stats:?}");

        // Rounding only: global SSP-RK3 drifts 4.6e-14 on this run
        let drift = (setup.mass(&q) - mass0).abs() / mass0;
        assert!(
            drift < 1e-13,
            "{}: relative mass drift {drift:e}",
            base.name()
        );
        assert!(q.h_data().iter().all(|h| h.is_finite()));
        // The wave has moved through the level interfaces
        assert!(max_speed(&q) > 1e-3);
    }
}

#[test]
fn lake_at_rest_with_dry_land_across_levels() {
    let setup = graded_setup();
    // A rough bed with an island reaching above the surface
    let bed = |x: f64, y: f64| {
        graded_bed(x, y)
            + 8.0 * (x / 700.0).sin() * (y / 400.0).cos()
            + 140.0 * gaussian(x, y, (6_000.0, 1_000.0), 900.0)
    };
    let q0 = setup.fill(at_rest(bed, |_, _| 0.0));
    assert!(q0.h_data().contains(&0.0), "the island must be dry");
    for base in BASES {
        let physics = setup
            .builder()
            .with_bathymetry(setup.bathymetry(bed))
            .with_limiter(StandardLimiter2D::Positivity(WetDryConfig::DEFAULT_H_DRY))
            .with_wet_dry(WetDryConfig::default())
            .with_implicit_friction(ManningFriction2D::new(G, 0.025))
            .build();
        let mut q = q0.clone();
        let sim = Simulation::new(physics, multirate(base, 8)).with_cfl(1.0);
        let result = sim.run(&mut q, 0.0, 600.0);
        assert!(result.success);
        let stats = result.local_time_stepping.unwrap();
        assert!(stats.finest_level >= 2, "{stats:?}");
        assert_eq!(sim.physics().negative_depth_clips(), 0);

        let speed = max_speed(&q);
        let dh = q
            .h_data()
            .iter()
            .zip(q0.h_data())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(speed < 1e-10, "{}: spurious speed {speed:e}", base.name());
        assert!(dh < 1e-10, "{}: spurious depth change {dh:e}", base.name());
    }
}

/// Stage times: a uniform body force h·A·cos(ωt) along y on a state at rest,
/// over a mesh graded along x (levels 0–3 side by side). The flux along x
/// of y-momentum is hu·v = 0 and the entropy-conservative flux adds no
/// dissipation, so every element integrates the ODE on its own substeps.
/// SSP-RK3's weights at c = (0, 1, ½) and SSP-RK(4,3)'s at c = (0, ½, 1, ½)
/// are both Simpson's rule, so hv follows h·A·sin(ωt)/ω to O((ωΔt)⁴), unless
/// an element evaluates the force at the wrong stage times (an O(Δt) error).
#[test]
fn stage_times_follow_each_element() {
    let mut mesh = Mesh2D::uniform_periodic(0.0, 1.0, 0.0, 2_000.0, 16, 2);
    for v in &mut mesh.vertices {
        v[0] = 10_000.0 * (1.25_f64.powf(16.0 * v[0]) - 1.0) / (1.25_f64.powf(16.0) - 1.0);
    }
    let setup = Setup::new(mesh, 2);
    let (amplitude, omega) = (1e-3, 2.0 * std::f64::consts::PI / 300.0);
    let depth = 50.0;
    for base in BASES {
        let physics = setup
            .builder()
            .with_formulation(SWEFormulation2D::EntropyConservative)
            .with_source(OscillatingForce { amplitude, omega })
            .build();
        let mut q = setup.fill(|_, _| SWEState2D::new(depth, 0.0, 0.0));

        let sim = Simulation::new(physics, multirate(base, 8)).with_cfl(0.5);
        let t_end = 250.0;
        let result = sim.run(&mut q, 0.0, t_end);
        let stats = result.local_time_stepping.unwrap();
        assert!(stats.finest_level >= 3, "{stats:?}");

        let exact = depth * amplitude * (omega * t_end).sin() / omega;
        let error = q
            .hv_data()
            .iter()
            .map(|hv| (hv - exact).abs())
            .fold(0.0, f64::max);
        // Coarsest steps ≈ 5 s: (ωΔt)⁴/180 ≈ 6e-7 per unit time
        assert!(
            error < 1e-6 * exact.abs(),
            "{}: hv error {error:e} against {exact:e}",
            base.name()
        );
        let still = |values: &[f64], reference: f64| {
            values
                .iter()
                .all(|&v| (v - reference).abs() < 1e-11 * depth)
        };
        assert!(still(q.hu_data(), 0.0), "no flow along x");
        assert!(still(q.h_data(), depth), "no change of depth");
    }
}
/// The multirate solution converges to the (spatially identical) global
/// SSP-RK3 solution at second order in the time step at least.
#[test]
fn converges_to_the_global_solution() {
    let setup = graded_setup();
    let physics = || {
        setup
            .builder()
            .with_bathymetry(setup.bathymetry(graded_bed))
            .build()
    };
    let q0 = setup.fill(at_rest(graded_bed, |x, y| {
        0.2 * gaussian(x, y, (3_000.0, 1_000.0), 1_500.0)
    }));
    let t_end = 200.0;

    let mut reference = q0.clone();
    let result = Simulation::new(physics(), SSPRK3)
        .with_cfl(0.02)
        .run(&mut reference, 0.0, t_end);
    assert!(result.success);

    for base in BASES {
        let errors: Vec<f64> = [0.8, 0.4, 0.2]
            .iter()
            .map(|&cfl| {
                let mut q = q0.clone();
                let result = Simulation::new(physics(), multirate(base, 8))
                    .with_cfl(cfl)
                    .run(&mut q, 0.0, t_end);
                assert!(result.local_time_stepping.unwrap().finest_level >= 2);
                setup.l2_difference(&q, &reference)
            })
            .collect();
        let orders: Vec<f64> = errors.windows(2).map(|e| (e[0] / e[1]).log2()).collect();
        println!("{}: errors {errors:?}, orders {orders:?}", base.name());
        for order in orders {
            assert!(
                order > 1.8,
                "{}: temporal order {order:.2} (errors {errors:?})",
                base.name()
            );
        }
    }
}

/// A wave runs up a beach; the offshore water sets the finest level. The
/// positivity guarantee holds per element at its own time step: h ≥ 0, no
/// negative-mean clip, mass conserved. SSP-RK(4,3) runs at twice SSP-RK3's
/// capped CFL (its SSP coefficient), with local and with global steps.
#[test]
fn runup_keeps_positivity_across_levels() {
    let setup = Setup::new(graded_mesh(3_000.0, 600.0, 20, 3, 1.12), 2);
    // Deep water at the (large-element) right, a beach on the left
    let bed = |x: f64, y: f64| -60.0 * (x / 3_000.0).powi(2) + 2.0 - 0.5 * (y / 150.0).sin();
    let physics = || {
        setup
            .builder()
            .with_bathymetry(setup.bathymetry(bed))
            .with_limiter(StandardLimiter2D::Positivity(WetDryConfig::DEFAULT_H_DRY))
            .with_wet_dry(WetDryConfig::new(
                Depth::new(WetDryConfig::DEFAULT_H_DRY),
                G,
            ))
            .with_implicit_friction(ManningFriction2D::new(G, 0.02))
            .build()
    };
    let q0 = setup.fill(at_rest(bed, |x, y| {
        1.5 * gaussian(x, y, (2_000.0, 300.0), 300.0)
    }));
    let mass0 = setup.mass(&q0);

    let check = |name: &str, q: &SWESolution2D, min_h: f64, clips: usize| {
        assert_eq!(clips, 0, "{name}: negative-mean clips");
        assert!(min_h >= 0.0, "{name}: min h {min_h:e}");
        let drift = (setup.mass(q) - mass0).abs() / mass0;
        assert!(drift < 1e-13, "{name}: relative mass drift {drift:e}");
    };
    let min_depth = |q: &SWESolution2D| q.h_data().iter().copied().fold(f64::INFINITY, f64::min);

    let mut coarse_steps = Vec::new();
    for base in BASES {
        // To 300 s, and to 500 s: once the first film is on the beach (at
        // 104 s) its own time step, minutes long, must not become the coarse
        // step. It did: the run to 300 s got by on a last step cut to less
        // than half its levels' length, and the run to 500 s blew up.
        for t_end in [300.0, 500.0] {
            // A linear CFL above both positivity bounds, so they set the steps
            let sim = Simulation::new(physics(), multirate(base, 8)).with_cfl(2.0);
            let mut q = q0.clone();
            let mut min_h = f64::INFINITY;
            let result = sim.run_with_callback(&mut q, 0.0, t_end, |q, _| {
                min_h = min_h.min(min_depth(q));
            });
            assert!(result.success, "{} to {t_end} s", base.name());
            let stats = result.local_time_stepping.unwrap();
            println!(
                "runup to {t_end} s, {}: {stats:?}, {:.2}x",
                base.name(),
                stats.speedup()
            );
            assert!(stats.finest_level >= 2, "{stats:?}");
            assert!(result.dt_max < 30.0, "a coarse step of {} s", result.dt_max);
            // A wave of 1.5 m on a beach: no element marked line-local (1 m
            // of water over its shallowest node) runs shallow within a
            // coarse step
            assert!(stats.line_local_elements > 0, "{stats:?}");
            assert_eq!(stats.line_local_breaks, 0, "{stats:?}");
            check(base.name(), &q, min_h, sim.physics().negative_depth_clips());
            if t_end == 300.0 {
                coarse_steps.push(stats.steps);
            }
        }
    }
    // The doubled positivity bound halves the steps (up to the levels'
    // rounding)
    let ratio = coarse_steps[0] as f64 / coarse_steps[1] as f64;
    assert!(ratio > 1.6, "coarse steps {coarse_steps:?}");

    // Global SSP-RK(4,3) at its capped CFL, 2·positivity_cfl_swe_2d(2)
    let sim = Simulation::new(physics(), StandardIntegrator::SSPRK43).with_cfl(1.0);
    let mut q = q0.clone();
    let mut min_h = f64::INFINITY;
    let result = sim.run_with_callback(&mut q, 0.0, 300.0, |q, _| {
        min_h = min_h.min(min_depth(q));
    });
    assert!(result.success);
    let expected_dt = sim
        .physics()
        .compute_dt_ssp(&q0, 1.0, Some(SspScheme::Rk43));
    assert!(
        (result.dt_max / expected_dt - 1.0).abs() < 0.05,
        "dt {} against {expected_dt}",
        result.dt_max
    );
    check(
        "global ssp-rk43",
        &q,
        min_h,
        sim.physics().negative_depth_clips(),
    );
}

#[test]
#[should_panic(expected = "not element-local")]
fn kuzmin_limiter_is_rejected() {
    let setup = Setup::new(Mesh2D::uniform_rectangle(0.0, 1.0, 0.0, 1.0, 2, 2), 1);
    let physics = setup
        .builder()
        .with_limiter(StandardLimiter2D::Kuzmin(KuzminParameter2D::strict()))
        .build();
    let mut q = setup.fill(|_, _| SWEState2D::new(1.0, 0.0, 0.0));
    Simulation::new(physics, MultirateSSPRK3::new(2)).run(&mut q, 0.0, 0.01);
}

/// The farm-refined fjord (`scripts/gmsh_farm_mesh.py`: 20 m quads at the
/// farm, ≈ 450 m far away): levels 0–6 from the element sizes, with a
/// clear saving of RHS work over global steps at the finest time step. Mass
/// is conserved.
#[test]
fn farm_refinement_saves_rhs_work() {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/gmsh/fjord_farm.msh");
    let setup = Setup::new(dgcom::mesh::read_gmsh_mesh(&path).expect("farm mesh"), 1);
    let bed = |x: f64, y: f64| -(30.0 + 120.0 * (1.0 - x / 12_000.0)) - 10.0 * (y / 2_000.0).sin();
    let physics = setup
        .builder()
        .with_bathymetry(setup.bathymetry(bed))
        .build();
    let mut q = setup.fill(at_rest(bed, |x, y| {
        0.3 * gaussian(x, y, (6_000.0, 3_000.0), 400.0)
    }));
    let mass0 = setup.mass(&q);

    let sim = Simulation::new(physics, MultirateSSPRK3::new(8)).with_cfl(0.8);
    let result = sim.run(&mut q, 0.0, 10.0);
    assert!(result.success);
    let stats = result.local_time_stepping.unwrap();
    println!("farm: {stats:?}, {:.2}x", stats.speedup());
    assert!(stats.finest_level >= 5, "{stats:?}");
    assert!(stats.speedup() > 2.0, "{stats:?}");
    let drift = (setup.mass(&q) - mass0).abs() / mass0;
    assert!(drift < 1e-13, "relative mass drift {drift:e}");
}

// =============================================================================
// Horizontal viscosity
// =============================================================================

/// ∫hu and ∫hv
fn momentum(setup: &Setup, q: &SWESolution2D) -> [f64; 2] {
    [setup.integral(q.hu_data()), setup.integral(q.hv_data())]
}

/// A periodic 10 × 2 km mesh, 20 × 8 elements with the interior vertices
/// moved pseudo-randomly by up to 0.4 of the spacing, over a flat bed 40 m
/// deep, with a surface bump and a sheared current: irregular levels.
///
/// Irregular on purpose: a neighbour's gradient at the shared face reads
/// beyond that neighbour only at the face's corner nodes (the GLL lift of a
/// face touches only its own nodes), i.e. from the elements diagonal to the
/// evaluating one. On a graded tensor-product mesh the diagonal element
/// always shares a level with one of the common face neighbours, so an RHS
/// reused with a one-hop stencil is never stale there.
fn periodic_viscous_setup() -> (Setup, SWESolution2D) {
    let (lx, ly, nx, ny) = (10_000.0, 2_000.0, 20, 8);
    let mut mesh = Mesh2D::uniform_periodic(0.0, lx, 0.0, ly, nx, ny);
    let (dx, dy) = (lx / nx as f64, ly / ny as f64);
    let noise = |i: f64| ((i * 12.9898).sin() * 43_758.545_3).fract();
    for (i, v) in mesh.vertices.iter_mut().enumerate() {
        let interior = v[0] > 1.0 && v[0] < lx - 1.0 && v[1] > 1.0 && v[1] < ly - 1.0;
        if interior {
            v[0] += 0.4 * dx * noise(i as f64);
            v[1] += 0.4 * dy * noise(i as f64 + 0.5);
        }
    }
    let setup = Setup::new(mesh, 2);
    let q = setup.fill(|x, y| {
        let h = 40.0 + 0.3 * gaussian(x, y, (1_500.0, 1_000.0), 400.0);
        // Sheared across the level interfaces (along x) and along them
        let (kx, ky) = (
            2.0 * std::f64::consts::PI / 2_500.0,
            2.0 * std::f64::consts::PI / 2_000.0,
        );
        let u = 0.1 + 0.2 * (ky * y).sin() + 0.1 * (kx * x).cos();
        let v = 0.2 * (kx * x).sin();
        SWEState2D::new(h, h * u, h * v)
    });
    (setup, q)
}

/// Across levels with BR1 viscosity, every face flux (the viscous one reads
/// the neighbours' gradients, two face hops of the state) is the same from
/// both sides at every stage: mass and momentum are conserved to rounding.
/// An RHS reused past a change two hops away (the stepper following a
/// one-hop stencil) breaks it at the level interfaces.
#[test]
fn viscosity_across_levels_conserves_mass_and_momentum() {
    let (setup, mut q) = periodic_viscous_setup();
    let physics = setup
        .builder()
        .with_viscosity(HorizontalViscosity2D::constant(50.0))
        .build();
    let (mass0, momentum0) = (setup.mass(&q), momentum(&setup, &q));
    let q0 = q.clone();

    let sim = Simulation::new(physics, MultirateSSPRK3::new(8)).with_cfl(0.8);
    let result = sim.run(&mut q, 0.0, 300.0);
    assert!(result.success);
    let stats = result.local_time_stepping.unwrap();
    println!("viscous: {stats:?}, {:.2}x", stats.speedup());
    assert!(stats.finest_level >= 3, "{stats:?}");

    let drift = (setup.mass(&q) - mass0).abs() / mass0;
    println!("mass drift {drift:e}");
    assert!(drift < 1e-13, "relative mass drift {drift:e}");
    // Scale: ∫h|u|, as ∫hv starts near zero
    let speeds: Vec<f64> = q0
        .hu_data()
        .iter()
        .zip(q0.hv_data())
        .map(|(hu, hv)| hu.hypot(*hv))
        .collect();
    let scale = setup.integral(&speeds);
    for (component, (m1, m0)) in momentum(&setup, &q).iter().zip(momentum0).enumerate() {
        let drift = (m1 - m0).abs() / scale;
        println!("momentum {component} drift {drift:e}");
        assert!(
            drift < 1e-13,
            "momentum {component}: relative drift {drift:e}"
        );
    }
    assert!(max_speed(&q) > 0.1, "the current must survive");
}

/// With viscosity the multirate solution converges to the global SSP-RK3
/// one at second order in the time step at least (as without).
#[test]
fn viscous_run_converges_to_the_global_solution() {
    let setup = graded_setup();
    let physics = |nu: f64| {
        let builder = setup
            .builder()
            .with_bathymetry(setup.bathymetry(graded_bed));
        if nu > 0.0 {
            builder.with_viscosity(HorizontalViscosity2D::constant(nu))
        } else {
            builder
        }
        .build()
    };
    let nu = 200.0;
    let q0 = setup.fill(|x, y| {
        let mut state = at_rest(graded_bed, |x, y| {
            0.2 * gaussian(x, y, (3_000.0, 1_000.0), 1_500.0)
        })(x, y);
        // A sheared current for the viscosity to act on
        let u = 0.3 * (2.0 * std::f64::consts::PI * y / 2_000.0).sin() * (x / 10_000.0);
        state.hu = state.h * u;
        state
    });
    let t_end = 200.0;

    let mut reference = q0.clone();
    let result =
        Simulation::new(physics(nu), SSPRK3)
            .with_cfl(0.02)
            .run(&mut reference, 0.0, t_end);
    assert!(result.success);
    let mut inviscid = q0.clone();
    Simulation::new(physics(0.0), SSPRK3)
        .with_cfl(0.1)
        .run(&mut inviscid, 0.0, t_end);
    let viscous_effect = setup.l2_difference(&reference, &inviscid);

    let errors: Vec<f64> = [0.8, 0.4, 0.2]
        .iter()
        .map(|&cfl| {
            let mut q = q0.clone();
            let result = Simulation::new(physics(nu), MultirateSSPRK3::new(8))
                .with_cfl(cfl)
                .run(&mut q, 0.0, t_end);
            assert!(result.local_time_stepping.unwrap().finest_level >= 2);
            setup.l2_difference(&q, &reference)
        })
        .collect();
    let orders: Vec<f64> = errors.windows(2).map(|e| (e[0] / e[1]).log2()).collect();
    println!("viscous effect {viscous_effect:e}, errors {errors:?}, orders {orders:?}");
    assert!(
        errors[0] < 0.1 * viscous_effect,
        "the viscous term must matter more than the time error"
    );
    for order in orders {
        assert!(order > 1.8, "temporal order {order:.2} (errors {errors:?})");
    }
}

/// The mode splitter finishes each barotropic stage element by element in
/// its fused pass (`BarotropicPhysics::finish_stage_element`): the implicit
/// damping and the post-processing of every element alone are the
/// whole-state operations bit for bit, and count the same clips.
#[test]
fn finishing_a_stage_by_element_matches_the_whole_state() {
    use dgcom::time::BarotropicPhysics;

    let setup = Setup::new(Mesh2D::uniform_rectangle(0.0, 1000.0, 0.0, 500.0, 8, 4), 2);
    let physics = shoreline_physics(&setup);
    assert!(physics.finishes_by_element());
    let mut from = shoreline_state(&setup);
    let [h, hu, hv] = &mut from.data;
    for ((hu, hv), &h) in hu.iter_mut().zip(hv.iter_mut()).zip(h.iter()) {
        *hu = 0.4 * h;
        *hv = -0.1 * h;
    }
    // A stage value that drains the shore below its bed in places
    let mut stage = from.clone();
    stage.data[0].iter_mut().for_each(|h| *h -= 0.02);
    let dt = 0.7;

    let mut whole = stage.clone();
    physics.implicit_damping(&mut whole, &from, dt);
    physics.post_process(&mut whole);
    let whole_clips = physics.negative_depth_clips();

    let mut by_element = stage.clone();
    let nn = setup.ops.n_nodes;
    let [h, hu, hv] = &mut by_element.data;
    let mut clips = 0;
    for k in 0..setup.mesh.n_elements {
        let rows = k * nn..(k + 1) * nn;
        let from = [0, 1, 2].map(|var| &from.data[var][rows.clone()]);
        clips += physics.finish_stage_element(
            k,
            [
                &mut h[rows.clone()],
                &mut hu[rows.clone()],
                &mut hv[rows.clone()],
            ],
            from,
            dt,
        );
    }
    physics.record_clips(clips);

    let bits = |s: &[f64]| s.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    for var in 0..3 {
        assert_ne!(
            bits(&whole.data[var]),
            bits(&stage.data[var]),
            "test regime: var {var} unchanged"
        );
        assert_eq!(
            bits(&whole.data[var]),
            bits(&by_element.data[var]),
            "var {var}"
        );
    }
    assert_eq!(physics.negative_depth_clips(), 2 * whole_clips, "clips");
}

// =============================================================================
// Line-local elements: stage rates node by node
// =============================================================================

/// A beach under a graded mesh with every stage component of a tidal run:
/// the `WetDry` split form with its wet/dry correction, implicit Manning
/// friction, Coriolis on a β-plane and a time-dependent force. 80 m of
/// water under the small elements on the left, a beach under the large ones
/// on the right: element time steps from 1 to 30 s.
fn line_local_setup() -> (Setup, SWEPhysics2D<Reflective2D>, SWESolution2D) {
    let setup = Setup::new(graded_mesh(6_000.0, 1_500.0, 12, 3, 1.3), 2);
    let bed = |x: f64, y: f64| -80.0 + 83.0 * (x / 6_000.0).powi(2) + 0.5 * (y / 400.0).sin();
    let physics = setup
        .builder()
        .with_bathymetry(setup.bathymetry(bed))
        .with_wet_dry(WetDryConfig::default())
        .with_implicit_friction(ManningFriction2D::new(G, 0.025))
        .with_source(dgcom::source::CoriolisSource2D::beta_plane(
            1.2e-4, 1.5e-11, 0.0,
        ))
        .with_source(OscillatingForce {
            amplitude: 1e-3,
            omega: 2.0 * std::f64::consts::PI / 600.0,
        })
        .build();
    let mut q = setup.fill(at_rest(bed, |x, y| {
        0.5 * gaussian(x, y, (2_000.0, 700.0), 800.0)
    }));
    // Some motion, different at every node
    let [h, hu, hv] = &mut q.data;
    for (i, ((hu, hv), &h)) in hu.iter_mut().zip(hv.iter_mut()).zip(h.iter()).enumerate() {
        *hu = h * 0.3 * (0.37 * i as f64).sin();
        *hv = h * 0.2 * (0.23 * i as f64).cos();
    }
    (setup, physics, q)
}

/// What `LocalTimeStepping::line_local` promises is what the stage kernel
/// does: perturb one node of a stage's input and compare every stage value
/// bit for bit. A line-local element changes only on the GLL lines through
/// the node, or at the node coinciding with it across a face; any other
/// element only if the node is its own, or on the shared face of a neighbour
/// or one in from it. The stage rates of local time stepping are spread by
/// exactly these sets (`spread_over_lines`): a kernel that read more would
/// leave stage values a substep behind.
#[test]
fn a_stage_value_reads_what_line_local_says() {
    let (setup, physics, q) = line_local_setup();
    let (mesh, ops) = (&setup.mesh, &setup.ops);
    let (n_elements, nn, n1) = (mesh.n_elements, ops.n_nodes, ops.n_1d);
    let local = physics.local_time_stepping().expect("SWE supports LTS");
    let mut line_local = vec![false; n_elements];
    let lines = local
        .line_local(&q, &mut line_local)
        .expect("the wet/dry split form has line-local elements");
    assert_eq!((lines.n_1d, lines.face_nodes), (n1, &ops.face_nodes));
    let n_line_local = line_local.iter().filter(|&&l| l).count();
    assert!(
        n_line_local >= n_elements / 3 && n_line_local < n_elements,
        "{n_line_local} of {n_elements} line-local: both kinds are needed"
    );

    // A stage with every term: ¾ base + ¼ (input + Δt F), then the damping
    // and the wet/dry correction
    let elements: Vec<u32> = (0..n_elements as u32).collect();
    let plan = |_: usize| ElementStage {
        time: 40.0,
        a: 0.75,
        b: 0.25,
        c: 0.25 * 0.2,
        accumulate: None,
    };
    let stage = |input: &SWESolution2D| {
        let mut out = SWESolution2D::new(n_elements, nn);
        local.stage_where(&q, input, &elements, &plan, &mut out, None);
        out
    };
    // The input differs from the base, to tell their weights apart
    let mut input = q.clone();
    for data in &mut input.data {
        for (i, v) in data.iter_mut().enumerate() {
            *v *= 1.0 + 1e-3 * (0.11 * i as f64).sin();
        }
    }
    let reference = stage(&input);

    // The node one in from `node` of `face`
    let inner = |face: usize, node: usize| match face {
        0 => node + n1,
        1 => node - 1,
        2 => node - n1,
        _ => node + 1,
    };
    let (mut changed_own, mut changed_across) = (0, 0);
    for j in 0..n_elements {
        for m in 0..nn {
            let mut perturbed = input.clone();
            for data in &mut perturbed.data {
                data[j * nn + m] = data[j * nn + m] * 1.01 + 1e-3;
            }
            let out = stage(&perturbed);
            for k in 0..n_elements {
                // The nodes of k that may read node m of j
                let mut may = vec![false; nn];
                if k == j {
                    for n in 0..nn {
                        may[n] = !line_local[k] || n / n1 == m / n1 || n % n1 == m % n1;
                    }
                }
                for face in 0..4 {
                    let Some(nb) = mesh.neighbor(ElementIndex::new(k), face) else {
                        continue;
                    };
                    if nb.element != j {
                        continue;
                    }
                    for (fi, &node) in ops.face_nodes[face].iter().enumerate() {
                        let across = ops.face_nodes[nb.face][n1 - 1 - fi];
                        if line_local[k] {
                            may[node] |= across == m;
                        } else if across == m || inner(nb.face, across) == m {
                            may.fill(true);
                        }
                    }
                }
                for n in 0..nn {
                    let i = k * nn + n;
                    let differs = (0..3)
                        .any(|var| out.data[var][i].to_bits() != reference.data[var][i].to_bits());
                    assert!(
                        !differs || may[n],
                        "node {n} of element {k} (line-local: {}) read node {m} of element {j}",
                        line_local[k]
                    );
                    if differs && k == j {
                        changed_own += 1;
                    } else if differs {
                        changed_across += 1;
                    }
                }
            }
        }
    }
    // The test sees the dependence it rules on
    assert!(changed_own > 4 * n_elements * nn && changed_across > n_elements * nn);
}

/// The node-by-node stage rates leave out only evaluations whose result
/// would have been the same: a run on them is the run on whole-element
/// rates, here bit for bit, with fewer evaluations and the same mass, and no
/// line-local element left the form within a coarse step.
///
/// (An element's substep result is the mean of its last stage's values over
/// the substep. The whole-element rates take it over more values, equal in
/// twos or fours where the node-by-node rates take one: sums of up to four
/// equal terms are exact, so three levels as here give the same bits; from
/// eight on the two may differ in the last place.)
#[test]
fn line_local_rates_are_the_whole_element_rates_bit_for_bit() {
    let (setup, physics, q0) = line_local_setup();
    let mesh = &setup.mesh;
    let local = physics.local_time_stepping().expect("SWE supports LTS");
    let mass0 = setup.mass(&q0);
    for scheme in [SspScheme::Rk3, SspScheme::Rk43] {
        let run = |line_rates: bool| {
            let mut stepper = MultirateStepper::new(8, scheme).with_line_local_rates(line_rates);
            let (mut q, mut t) = (q0.clone(), 0.0);
            for _ in 0..12 {
                let dt = stepper.assign_levels(local, mesh, &q, 1.0, None, None);
                stepper.step(local, &mut q, t, dt);
                t += dt;
            }
            (q, t, stepper.stats())
        };
        let (whole, t_whole, whole_stats) = run(false);
        let (lines, t_lines, line_stats) = run(true);
        println!("{scheme:?}: whole {whole_stats:?}\n  lines {line_stats:?}");
        assert_eq!(whole_stats.finest_level, 2, "{whole_stats:?}");
        assert_eq!(whole_stats.line_local_elements, 0);
        assert!(line_stats.line_local_elements > 0);
        assert_eq!(line_stats.line_local_breaks, 0);
        // Fewer evaluations for the same levels
        assert_eq!(line_stats.level_evaluations, whole_stats.level_evaluations);
        assert!(
            (line_stats.element_evaluations as f64) < 0.97 * whole_stats.element_evaluations as f64,
            "{} against {}",
            line_stats.element_evaluations,
            whole_stats.element_evaluations
        );

        // The same run
        assert_eq!(t_lines, t_whole);
        assert!(setup.l2_difference(&whole, &q0) > 0.0);
        for var in 0..3 {
            assert!(
                lines.data[var]
                    .iter()
                    .zip(&whole.data[var])
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "{scheme:?}, variable {var}: {:e}",
                setup.l2_difference(&lines, &whole)
            );
        }
        for (name, q) in [("whole", &whole), ("lines", &lines)] {
            let drift = (setup.mass(q) - mass0).abs() / mass0;
            assert!(drift < 1e-13, "{scheme:?}, {name}: mass drift {drift:e}");
        }
        assert_eq!(physics.negative_depth_clips(), 0);
    }
}

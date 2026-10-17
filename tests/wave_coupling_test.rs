//! Gates of the coupling from the spectral wave model to the circulation (TODO
//! F.4): the radiation-stress force of `WaveForce2D` and the wave-enhanced bed
//! friction of `WaveCurrentFriction2D` on the 2D shallow-water model, through
//! the production path (`Simulation` + `SSPRK3` + `SWEPhysics2D`).

use std::sync::Arc;

use dg_rs::boundary::Reflective2D;
use dg_rs::equations::ShallowWater2D;
use dg_rs::io::{AtmosphereReader, CoordinateProjection, FieldSeries, GeoGrid, LocalProjection};
use dg_rs::mesh::{Bathymetry2D, BoundaryTag, Mesh2D};
use dg_rs::operators::{DGOperators2D, GeometricFactors2D};
use dg_rs::physics::{PhysicsBuilder, SWEPhysics2DBuilder};
use dg_rs::simulation::Simulation;
use dg_rs::solver::{SWESolution2D, SWEState2D};
use dg_rs::source::{
    BottomFriction2D, ChezyFriction2D, GriddedAtmosphere2D, SourceContext2D, SourceTerm2D,
    WaveCurrentFriction2D, WaveForce2D,
};
use dg_rs::time::{ModelClock, SSPRK3};
use dg_rs::types::ElementIndex;
use dg_rs::waves::{
    CoupledWaves2D, SourceTerms, SpectralGrid, WaveCoupling2D, WaveForceForm, WaveModel2D,
    WaveSolution, WaveWorkspace, group_velocity, wavenumber,
};

const G: f64 = 9.81;

/// Linear damping of the momentum, `−r (hu, hv)`, to settle the basin.
struct LinearDamping(f64);

impl SourceTerm2D for LinearDamping {
    fn evaluate(&self, ctx: &SourceContext2D) -> SWEState2D {
        SWEState2D::new(0.0, -self.0 * ctx.state.hu, -self.0 * ctx.state.hv)
    }

    fn name(&self) -> &'static str {
        "linear_damping"
    }
}

/// Swell of 10 s and 1 m normal to the shore over a closed basin shoaling from
/// 20 to 3 m (no breaking): seaward of the breakers the mean level sets down as
/// `η = −E k / sinh 2kh` (Longuet-Higgins & Stewart 1962; E = H²/8, the local
/// variance), the balance of the radiation-stress gradient and the pressure
/// gradient, `g h ∂η/∂x = −g ∂S_xx/∂x`. The wave model shoals the swell, its
/// force drives the basin to rest, and the set-down follows the formula to
/// 0.27 % of its 3.3 cm range across the basin. A current of 2e-5 m/s remains,
/// held by the damping against the part of the force the discrete pressure
/// gradient does not balance: discretisation error, 3e-6 m/s at half the
/// element size.
#[test]
fn shoaling_waves_set_the_water_down() {
    let (range, worst, speed) = shoaling_set_down(None);
    assert!(range > 0.01, "a set-down of {range} m");
    assert!(speed < 5e-5, "not at rest: {speed} m/s");
    assert!(worst < 0.01 * range, "η off the formula by {worst:e} m");
}

/// The same basin with the waves on a mesh of their own (TODO F.4 cost), coupled
/// by `WaveCoupling2D`: the radiation stress interpolated onto the
/// circulation's 50 m P2 nodes and differentiated there. The wave meshes have
/// 7 and 14 elements of 286 and 143 m, whose faces fall inside the
/// circulation's elements.
///
/// The wave state is exact shoaling at its own nodes (to 1e-12); what departs
/// is its polynomial between them, which the force differentiates. The
/// set-down is off the formula by 5.2 / 2.2 % of its range at 7 / 14 P1
/// elements (0.88 % at 28; second order as the elements shrink, the error at
/// the 3 m end where S curves most), and by 1.0 % at 7 P2 elements (0.40 % at
/// 14, then the same-mesh 0.27 %). P2 on the coarser mesh is the better use
/// of the nodes: 21 nodes along the basin against 28 for P1 at 14 elements.
#[test]
fn shoaling_waves_on_a_coarse_mesh_of_their_own_set_the_water_down() {
    let departure = |nx: usize, order: usize| {
        let (range, worst, speed) = shoaling_set_down(Some((nx, order)));
        assert!(range > 0.01, "a set-down of {range} m");
        assert!(speed < 1e-4, "not at rest: {speed} m/s");
        worst / range
    };
    let p1 = [departure(7, 1), departure(14, 1)];
    let p2 = departure(7, 2);
    assert!(p1[0] < 0.06 && p1[1] < 0.025, "P1: {p1:?}");
    assert!(p1[1] < 0.5 * p1[0], "P1 does not converge: {p1:?}");
    assert!(p2 < 0.012, "P2: {p2}");
}

/// The shoaling basin at rest under the waves' force, with the waves on the
/// circulation's mesh (`None`) or on `Some((nx, order))`, `nx` elements of
/// their own: the set-down's range across the basin (m), the largest
/// departure from the formula (m), and the largest current left (m/s).
fn shoaling_set_down(wave_mesh: Option<(usize, usize)>) -> (f64, f64, f64) {
    const L: f64 = 2000.0;
    let depth = |x: f64| 20.0 - 17.0 * x / L;
    let sides = [
        BoundaryTag::Wall,
        BoundaryTag::Open,
        BoundaryTag::Wall,
        BoundaryTag::Open,
    ];
    let mesh = Mesh2D::uniform_rectangle_with_sides(0.0, L, 0.0, 100.0, 40, 1, sides);
    let ops = Arc::new(DGOperators2D::new(2));
    let geom = Arc::new(GeometricFactors2D::compute(&mesh, &ops));
    let bathymetry = Arc::new(Bathymetry2D::from_function(&mesh, &ops, &geom, |x, _| {
        -depth(x)
    }));
    let mesh = Arc::new(mesh);

    // The steady swell: one component, 10 s along +x, H = 1 m offshore
    let grid = SpectralGrid::new(0.1, 0.12, 2, 36);
    let mut e = vec![0.0; grid.n_components()];
    let c = grid.component(0, 0);
    e[c] = 1.0 / 8.0 / (grid.d_sigma[0] * grid.d_theta);
    let waves = match wave_mesh {
        None => WaveModel2D::new(
            mesh.clone(),
            ops.clone(),
            geom.clone(),
            &bathymetry,
            grid,
            G,
        ),
        Some((nx, order)) => {
            let wave_mesh = Mesh2D::uniform_rectangle_with_sides(0.0, L, 0.0, 100.0, nx, 1, sides);
            let wave_ops = Arc::new(DGOperators2D::new(order));
            let wave_geom = Arc::new(GeometricFactors2D::compute(&wave_mesh, &wave_ops));
            let wave_bed =
                Bathymetry2D::from_function(&wave_mesh, &wave_ops, &wave_geom, |x, _| -depth(x));
            WaveModel2D::new(Arc::new(wave_mesh), wave_ops, wave_geom, &wave_bed, grid, G)
        }
    }
    .with_boundary_spectrum(&e);
    let mut n = waves.zero_state();
    let mut ws = WaveWorkspace::default();
    let dt = waves.compute_dt(0.5);
    let steps = (3.0 * L / (3.0 * dt)).ceil() as usize; // c_g ≥ 3 m/s
    for s in 0..steps {
        waves.step(&mut n, s as f64 * dt, dt, &mut ws);
    }

    // The swell shoals with E c_g constant, exactly at the waves' nodes
    let sigma = waves.grid.sigma[0];
    let cg = |h: f64| group_velocity(sigma, wavenumber(sigma, h, G), h);
    let shoaled = |x: f64| cg(depth(0.0)) / cg(depth(x)) / 8.0;
    for (p, params) in waves.parameters(&n).iter().enumerate() {
        let (k, i) = (
            ElementIndex::new(p / waves.ops.n_nodes),
            p % waves.ops.n_nodes,
        );
        let [x, _] =
            waves
                .mesh
                .reference_to_physical(k, waves.ops.nodes_r[i], waves.ops.nodes_s[i]);
        assert!((params.m0 / shoaled(x) - 1.0).abs() < 1e-12, "x = {x}");
    }

    // The basin at rest under the waves' force
    let coupling = WaveCoupling2D::new(&waves, mesh.clone(), ops.clone(), geom.clone());
    let force = match wave_mesh {
        None => WaveForce2D::new(&waves, &n),
        Some(_) => coupling.force(&waves, &n),
    }
    .with_ramp(1200.0);
    let physics = PhysicsBuilder::swe_2d(
        mesh.clone(),
        ops.clone(),
        geom.clone(),
        ShallowWater2D::new(G),
        Reflective2D::new(),
    )
    .with_bathymetry(bathymetry.clone())
    .with_source(force)
    .with_source(LinearDamping(2e-3))
    .build();
    let nn = ops.n_nodes;
    let mut q = SWESolution2D::new(mesh.n_elements, nn);
    for k in ElementIndex::iter(mesh.n_elements) {
        for i in 0..nn {
            q.set_state(k, i, SWEState2D::new(-bathymetry.get(k, i), 0.0, 0.0));
        }
    }
    let result = Simulation::new(physics, SSPRK3).run(&mut q, 0.0, 8000.0);
    assert!(result.success, "{result:?}");

    // η against the formula, both relative to the offshore end
    let mut samples: Vec<(f64, f64, f64)> = Vec::new(); // (x, η, formula)
    for k in ElementIndex::iter(mesh.n_elements) {
        for i in 0..nn {
            let p = k.as_usize() * nn + i;
            let [x, _] = mesh.reference_to_physical(k, ops.nodes_r[i], ops.nodes_s[i]);
            let h = depth(x);
            let eta = q.h_data()[p] + bathymetry.get(k, i);
            let kw = wavenumber(waves.grid.sigma[0], h, G);
            let formula = -shoaled(x) * kw / (2.0 * kw * h).sinh();
            samples.push((x, eta, formula));
        }
    }
    samples.sort_by(|a, b| a.0.total_cmp(&b.0));
    let (_, eta0, formula0) = samples[0];
    let range = samples
        .iter()
        .map(|s| s.2 - formula0)
        .fold(0.0f64, |a, b| a.max(b.abs()));
    let worst = samples
        .iter()
        .map(|s| ((s.1 - eta0) - (s.2 - formula0)).abs())
        .fold(0.0f64, f64::max);
    let speed = q
        .hu_data()
        .iter()
        .zip(q.h_data())
        .map(|(hu, h)| (hu / h).abs())
        .fold(0.0f64, f64::max);
    println!(
        "waves on {}: set-down across the basin {:.2} cm; largest departure {:.2e} m ({:.2} %); |u| ≤ {speed:.1e} m/s",
        wave_mesh.map_or("the same mesh".to_string(), |(nx, order)| format!(
            "{nx} P{order} elements"
        )),
        100.0 * range,
        worst,
        100.0 * worst / range
    );

    (range, worst, speed)
}

/// A uniform body force `G` per unit mass along x.
struct BodyForce(f64);

impl SourceTerm2D for BodyForce {
    fn evaluate(&self, ctx: &SourceContext2D) -> SWEState2D {
        SWEState2D::new(0.0, ctx.state.h * self.0, 0.0)
    }

    fn name(&self) -> &'static str {
        "body_force"
    }
}

/// A current driven along a periodic channel 10 m deep by a body force, under
/// a uniform sea whose bed stress is that of a 12.5 s wave of 1 m (from the
/// wave model, on a 2 mm roughness). Steady at
/// `G h = C_d u² [1 + 1.2 (τ_w/(C_d u² + τ_w))^3.2]`: the waves slow the
/// current by enhancing its friction, and the model holds the root of that
/// balance to 1e-7 (the point-implicit friction, applied through
/// `with_implicit_friction`): 0.63 m/s without the waves, 0.46 m/s under
/// their 6 Pa.
#[test]
fn waves_slow_a_current_by_soulsbys_enhanced_friction() {
    let (h, forcing, cd, z0) = (10.0, 1e-4, 2.5e-3, 0.002);
    let mesh = Mesh2D::uniform_periodic(0.0, 2000.0, 0.0, 2000.0, 2, 2);
    let ops = Arc::new(DGOperators2D::new(2));
    let geom = Arc::new(GeometricFactors2D::compute(&mesh, &ops));
    let bathymetry = Arc::new(Bathymetry2D::constant(mesh.n_elements, ops.n_nodes, -h));
    let mesh = Arc::new(mesh);
    // The waves' bed stress: one component, 1 m, along y
    let grid = SpectralGrid::new(0.05, 0.3, 20, 12);
    let (i, j) = (5, 3);
    let mut e = vec![0.0; grid.n_components()];
    e[grid.component(i, j)] = 1.0 / 8.0 / (grid.d_sigma[i] * grid.d_theta);
    let waves = WaveModel2D::new(
        mesh.clone(),
        ops.clone(),
        geom.clone(),
        &bathymetry,
        grid,
        G,
    );
    let tau_w = waves.bed_wave_stress(&waves.uniform_state(&e), z0);
    let tw = tau_w[0];
    assert!(tau_w.iter().all(|&t| (t - tw).abs() < 1e-15 * tw));

    let speed = |tau_w: f64| {
        let law =
            WaveCurrentFriction2D::new(ChezyFriction2D::new(cd), vec![tau_w; waves.n_points()]);
        let physics = PhysicsBuilder::swe_2d(
            mesh.clone(),
            ops.clone(),
            geom.clone(),
            ShallowWater2D::new(G),
            Reflective2D::new(),
        )
        .with_bathymetry(bathymetry.clone())
        .with_source(BodyForce(forcing))
        .with_implicit_friction(law)
        .build();
        let mut q = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                q.set_state(k, i, SWEState2D::new(h, 0.0, 0.0));
            }
        }
        let result = Simulation::new(physics, SSPRK3).run(&mut q, 0.0, 60_000.0);
        assert!(result.success, "{result:?}");
        q.hu_data()[0] / q.h_data()[0]
    };
    // The root of G h = C_d u² E(C_d u², τ_w), by bisection
    let exact = |tau_w: f64| {
        let residual = |u: f64| {
            let tau_c = cd * u * u;
            tau_c * WaveCurrentFriction2D::<ChezyFriction2D>::enhancement(tau_c, tau_w)
                - forcing * h
        };
        let (mut lo, mut hi) = (0.0, 1.0);
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if residual(mid) > 0.0 {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        0.5 * (lo + hi)
    };
    let (calm, rough) = (speed(0.0), speed(tw));
    println!(
        "τ_w {:.2} Pa: u {calm:.4} m/s without waves, {rough:.4} m/s under them (exact {:.4}, {:.4})",
        1025.0 * tw,
        exact(0.0),
        exact(tw)
    );
    assert!((calm / exact(0.0) - 1.0).abs() < 1e-7, "calm: {calm}");
    assert!(
        (rough / exact(tw) - 1.0).abs() < 1e-9,
        "under waves: {rough}"
    );
    assert!(rough < 0.75 * calm, "the waves barely slowed the current");
}

/// Linear damping of the cross-shore (y) momentum only, `−r hv`, to settle the
/// beach's seiches without touching the alongshore balance.
struct CrossShoreDamping(f64);

impl SourceTerm2D for CrossShoreDamping {
    fn evaluate(&self, ctx: &SourceContext2D) -> SWEState2D {
        SWEState2D::new(0.0, 0.0, -self.0 * ctx.state.hv)
    }

    fn name(&self) -> &'static str {
        "cross_shore_damping"
    }
}

/// A plane beach, uniform alongshore (x, periodic, one element wide): the depth
/// falls linearly from 4 m at the open sea (y = 0) to 0.5 m at a wall at
/// y = 350 m, a little offshore of where the shoreline would be. P2, 25 m
/// elements.
struct Beach {
    mesh: Arc<Mesh2D>,
    ops: Arc<DGOperators2D>,
    geom: Arc<GeometricFactors2D>,
    bathymetry: Arc<Bathymetry2D>,
}

impl Beach {
    const L: f64 = 350.0;
    const H0: f64 = 4.0;
    const H1: f64 = 0.5;
    const NY: usize = 14;

    fn depth(y: f64) -> f64 {
        Self::H0 + (Self::H1 - Self::H0) * y / Self::L
    }

    fn new() -> Self {
        Self::with(Self::NY, 2)
    }

    /// The beach on `ny` elements of order `order` across it.
    fn with(ny: usize, order: usize) -> Self {
        let width = Self::L / ny as f64;
        let mesh = Mesh2D::channel_periodic_x_with_sides(
            0.0,
            width,
            0.0,
            Self::L,
            1,
            ny,
            [BoundaryTag::Open, BoundaryTag::Wall],
        );
        let ops = Arc::new(DGOperators2D::new(order));
        let geom = Arc::new(GeometricFactors2D::compute(&mesh, &ops));
        let bathymetry = Arc::new(Bathymetry2D::from_function(&mesh, &ops, &geom, |_, y| {
            -Self::depth(y)
        }));
        Self {
            mesh: Arc::new(mesh),
            ops,
            geom,
            bathymetry,
        }
    }

    /// Swell of 8 s and H_rms = 0.6 m offshore travelling to `direction` (rad
    /// from +x; π/2 is straight at the beach) over `n_dir` directions, all in
    /// one bin, or cos^`m` spread. Battjes–Janssen breaking (α = 1, γ = 0.73)
    /// is the only source.
    fn waves(&self, direction: f64, m: Option<i32>, n_dir: usize) -> WaveModel2D {
        let f = 1.0 / 8.0;
        let grid = SpectralGrid::new(f, 1.2 * f, 2, n_dir);
        let mut weights: Vec<f64> = grid
            .theta
            .iter()
            .map(|&t| {
                let c = (t - direction).cos();
                match m {
                    Some(m) => c.max(0.0).powi(m),
                    None if c > 1.0 - 1e-9 => 1.0,
                    None => 0.0,
                }
            })
            .collect();
        let total: f64 = weights.iter().sum::<f64>() * grid.d_theta;
        weights.iter_mut().for_each(|w| *w /= total);
        let mut e = vec![0.0; grid.n_components()];
        for (j, w) in weights.iter().enumerate() {
            e[grid.component(0, j)] = 0.6f64.powi(2) / 8.0 / grid.d_sigma[0] * w;
        }
        WaveModel2D::new(
            self.mesh.clone(),
            self.ops.clone(),
            self.geom.clone(),
            &self.bathymetry,
            grid,
            G,
        )
        .with_sources(SourceTerms::none(G).with_breaking(Some((1.0, 0.73))))
        .with_boundary_spectrum(&e)
    }

    /// Run `waves` to steady from `n` (four crossings at 2 m/s).
    fn settle(waves: &WaveModel2D, n: &mut WaveSolution) {
        let mut ws = WaveWorkspace::default();
        let dt = waves.compute_dt(0.5);
        let steps = (4.0 * Self::L / (2.0 * dt)).ceil() as usize;
        for s in 0..steps {
            waves.step(n, s as f64 * dt, dt, &mut ws);
        }
    }

    fn physics(&self) -> SWEPhysics2DBuilder<Reflective2D> {
        PhysicsBuilder::swe_2d(
            self.mesh.clone(),
            self.ops.clone(),
            self.geom.clone(),
            ShallowWater2D::new(G),
            Reflective2D::new(),
        )
        .with_bathymetry(self.bathymetry.clone())
        .with_source(CrossShoreDamping(1e-2))
    }

    fn still_water(&self) -> SWESolution2D {
        let nn = self.ops.n_nodes;
        let mut q = SWESolution2D::new(self.mesh.n_elements, nn);
        for k in ElementIndex::iter(self.mesh.n_elements) {
            for i in 0..nn {
                q.set_state(k, i, SWEState2D::new(-self.bathymetry.get(k, i), 0.0, 0.0));
            }
        }
        q
    }

    fn eta(&self, q: &SWESolution2D) -> Vec<f64> {
        q.h_data()
            .iter()
            .zip(&self.bathymetry.data)
            .map(|(h, b)| h + b)
            .collect()
    }

    /// The level at the wall above the level at the open sea (m).
    fn setup(&self, q: &SWESolution2D) -> f64 {
        let eta = self.eta(q);
        let balance = self.balance(&vec![[0.0; 2]; eta.len()], q);
        eta[balance.last().unwrap().0] - eta[balance[0].0]
    }

    /// y of every node.
    fn ys(&self) -> Vec<f64> {
        ElementIndex::iter(self.mesh.n_elements)
            .flat_map(|k| {
                (0..self.ops.n_nodes).map(move |i| {
                    self.mesh
                        .reference_to_physical(k, self.ops.nodes_r[i], self.ops.nodes_s[i])[1]
                })
            })
            .collect()
    }

    /// The nodes of the column r = −1 from the sea to the wall, and at each
    /// one `∫₀^y F_y/(g D) dy`: the level the momentum balance
    /// `g D ∂η/∂y = F_y` gives for the force `force` on the depths of `q`
    /// (integrated exactly for the quadratic through each element's three
    /// nodes).
    fn balance(&self, force: &[[f64; 2]], q: &SWESolution2D) -> Vec<(usize, f64)> {
        let (ops, nn) = (&self.ops, self.ops.n_nodes);
        let mut column: Vec<usize> = (0..nn)
            .filter(|&i| (ops.nodes_r[i] + 1.0).abs() < 1e-12)
            .collect();
        column.sort_by(|&a, &b| ops.nodes_s[a].total_cmp(&ops.nodes_s[b]));
        let ys = self.ys();
        let mut out = vec![(column[0], 0.0)];
        let mut integral = 0.0;
        for k in 0..self.mesh.n_elements {
            let p: Vec<usize> = column.iter().map(|&i| k * nn + i).collect();
            let f: Vec<f64> = p
                .iter()
                .map(|&p| force[p][1] / (G * q.h_data()[p]))
                .collect();
            let dy = ys[p[2]] - ys[p[0]];
            let mid = integral + dy / 24.0 * (5.0 * f[0] + 8.0 * f[1] - f[2]);
            integral += dy / 6.0 * (f[0] + 4.0 * f[1] + f[2]);
            out.push((p[1], mid));
            out.push((p[2], integral));
        }
        out
    }
}

/// Swell of 8 s and H_rms = 0.6 m runs straight at a plane beach (1:100, 4 m
/// to a wall at 0.5 m) and breaks (Battjes–Janssen). The radiation stress
/// sets the water down to the breaker line (the largest H_rms) and up across
/// the surf zone: 0.69 cm down, 5.6 cm up at the wall. Two-way coupled: the
/// waves see the circulation's level (`set_water_level`), which deepens the
/// surf zone and lowers the setup at the wall by 3.5 % against waves on still
/// water; the third pass changes it by 0.15 %.
///
/// The level follows the cross-shore momentum balance `g D ∂η/∂y = F_y` of the
/// force on the circulation's own depths to 0.04 % of its range. The setup
/// slope over the inner surf zone is 0.039 of the bed's, below Bowen et al.'s
/// (1968) `3κ²/8/(1 + 3κ²/8)` = 0.062 for its mean κ = H_rms/D = 0.42:
/// Bowen's waves are saturated (κ constant), while Battjes–Janssen's κ rises
/// from 0.34 to 0.48 towards the wall, so S falls more slowly than `(3/16)κ²D²`
/// with κ fixed.
#[test]
fn breaking_waves_set_the_water_up_in_the_surf_zone() {
    let beach = Beach::new();
    let mut waves = beach.waves(std::f64::consts::FRAC_PI_2, None, 36);
    let mut n = waves.zero_state();
    let mut q = beach.still_water();
    let ys = beach.ys();
    let mut walls = Vec::new();
    for pass in 0..3 {
        Beach::settle(&waves, &mut n);
        let force = WaveForce2D::new(&waves, &n);
        let f = force.force().to_vec();
        // Ramp the first pass up from rest; later ones start from the last level
        let force = if pass == 0 {
            force.with_ramp(300.0)
        } else {
            force
        };
        let physics = beach.physics().with_source(force).build();
        let result = Simulation::new(physics, SSPRK3).run(&mut q, 0.0, 3000.0);
        assert!(result.success, "{result:?}");
        let eta = beach.eta(&q);

        let balance = beach.balance(&f, &q);
        let eta0 = eta[balance[0].0];
        let rise: Vec<f64> = balance.iter().map(|&(p, _)| eta[p] - eta0).collect();
        let (low, high) = rise
            .iter()
            .fold((0.0f64, 0.0f64), |(lo, hi), &r| (lo.min(r), hi.max(r)));
        let worst = balance
            .iter()
            .zip(&rise)
            .map(|(&(_, level), r)| (r - level).abs())
            .fold(0.0f64, f64::max);
        let speed = q
            .hv_data()
            .iter()
            .zip(q.h_data())
            .map(|(hv, h)| (hv / h).abs())
            .fold(0.0f64, f64::max);

        // The breaker line (the largest H_rms) and the inner surf zone
        let params = waves.parameters(&n);
        let &(breaker, _) = balance
            .iter()
            .max_by(|a, b| params[a.0].m0.total_cmp(&params[b.0].m0))
            .unwrap();
        let &(lowest, _) = balance
            .iter()
            .min_by(|a, b| eta[a.0].total_cmp(&eta[b.0]))
            .unwrap();
        let wall = balance.last().unwrap().0;
        let kappa: Vec<f64> = balance
            .iter()
            .filter(|&&(p, _)| ys[p] > ys[breaker])
            .map(|&(p, _)| (8.0 * params[p].m0).sqrt() / q.h_data()[p])
            .collect();
        let mean_kappa = kappa.iter().sum::<f64>() / kappa.len() as f64;
        let slope =
            (eta[wall] - eta[breaker]) / (Beach::depth(ys[breaker]) - Beach::depth(ys[wall]));
        let bowen = 3.0 * mean_kappa.powi(2) / 8.0;
        println!(
            "pass {pass}: set-down {:.2} cm at y = {:.0} m (breaker line {:.0} m), setup {:.2} cm \
             at the wall; balance off by {worst:.1e} m ({:.3} %), |v| ≤ {speed:.1e} m/s; \
             surf slope {slope:.3} of the bed's, Bowen {:.3} for κ = {mean_kappa:.2} \
             ({:.2}–{:.2})",
            100.0 * low,
            ys[lowest],
            ys[breaker],
            100.0 * (eta[wall] - eta0),
            100.0 * worst / (high - low),
            bowen / (1.0 + bowen),
            kappa.iter().cloned().fold(f64::INFINITY, f64::min),
            kappa.iter().cloned().fold(0.0, f64::max),
        );
        assert!(
            worst < 2e-3 * (high - low),
            "off the balance by {worst:e} m"
        );
        assert!(speed < 1e-3, "not at rest: {speed} m/s");
        assert!(
            (ys[lowest] - ys[breaker]).abs() <= Beach::L / Beach::NY as f64,
            "the set-down is lowest at {} m, the breaker line is at {} m",
            ys[lowest],
            ys[breaker]
        );
        assert!(low < -5e-3 && eta[wall] - eta0 > 5e-2, "{low}, {high}");
        walls.push(eta[wall] - eta0);
        waves.set_water_level(&eta);
    }
    let change = (walls[2] / walls[1] - 1.0).abs();
    assert!(change < 5e-3, "the coupling has not converged: {walls:?}");
}

/// The surf zone with the waves on a mesh of their own, two-way coupled by
/// `WaveCoupling2D` (TODO F.4 cost): the force interpolated onto the
/// circulation's 25 m P2 elements, and the circulation's level back onto the
/// waves' (`update_waves`), for two passes. On 7 P2 elements of 50 m the setup
/// at the wall is 5.793 then 5.597 cm, against 5.806 and 5.604 cm with the
/// waves on the circulation's own mesh (0.2 % and 0.1 % apart).
#[test]
fn breaking_waves_on_a_coarse_mesh_of_their_own_set_the_water_up() {
    let beach = Beach::new();
    let same = setups_by_passes(&beach, &beach, 2);
    let coarse = setups_by_passes(&beach, &Beach::with(7, 2), 2);
    println!(
        "setup at the wall: waves on the same mesh {:.3} → {:.3} cm, on 7 P2 elements {:.3} → {:.3} cm",
        100.0 * same[0],
        100.0 * same[1],
        100.0 * coarse[0],
        100.0 * coarse[1]
    );
    assert!(same[1] > 0.05, "{same:?}");
    for pass in 0..2 {
        assert!(
            (coarse[pass] / same[pass] - 1.0).abs() < 0.01,
            "pass {pass}: {coarse:?} against {same:?}"
        );
    }
    assert!(
        coarse[1] < coarse[0],
        "the coupling lowers the setup: {coarse:?}"
    );
}

/// The setup at the wall of `beach` after each of `passes` passes of the
/// waves on `wave_beach` (settled on the last level) and the circulation
/// (settled under their force), coupled by `WaveCoupling2D`.
fn setups_by_passes(beach: &Beach, wave_beach: &Beach, passes: usize) -> Vec<f64> {
    let mut waves = wave_beach.waves(std::f64::consts::FRAC_PI_2, None, 36);
    let coupling = WaveCoupling2D::new(
        &waves,
        beach.mesh.clone(),
        beach.ops.clone(),
        beach.geom.clone(),
    );
    let mut n = waves.zero_state();
    let mut q = beach.still_water();
    let mut setups = Vec::new();
    for pass in 0..passes {
        Beach::settle(&waves, &mut n);
        let force = coupling.force(&waves, &n);
        let force = if pass == 0 {
            force.with_ramp(300.0)
        } else {
            force
        };
        let physics = beach.physics().with_source(force).build();
        let result = Simulation::new(physics, SSPRK3).run(&mut q, 0.0, 3000.0);
        assert!(result.success, "{result:?}");
        setups.push(beach.setup(&q));
        coupling.update_waves(&mut waves, &q, &beach.bathymetry);
    }
    setups
}

/// The dissipation's force is what the sinks remove over the step the
/// waves took in the interval (`span / steps`, shorter than `compute_dt`
/// unless that divides the span), not over `compute_dt`: a regression test
/// for the exchange, which passed `compute_dt` (TODO F.4).
#[test]
fn the_dissipation_force_is_over_the_wave_step_taken() {
    let beach = Beach::new();
    let wave_beach = Beach::with(7, 2);
    let model = wave_beach.waves(std::f64::consts::FRAC_PI_2, None, 36);
    let mut n = model.zero_state();
    Beach::settle(&model, &mut n);
    let physics = beach.physics().build();
    let mut waves =
        CoupledWaves2D::new(model, n, &physics, 0.0).with_force_form(WaveForceForm::Dissipation);
    let interval = 30.0;
    let mut sim = Simulation::new(physics, SSPRK3);
    let mut q = beach.still_water();
    let result = sim.run_with_exchange(
        &mut q,
        0.0,
        interval,
        interval,
        |physics, q, t, t_next| waves.exchange(physics, q, t, t_next),
        |_, _| {},
    );
    assert!(result.success, "{result:?}");
    let stats = waves.stats();
    assert_eq!(stats.exchanges, 1);
    let taken = interval / stats.wave_steps as f64;
    let largest = waves.model().compute_dt(waves.cfl());
    assert!(
        taken < 0.99 * largest,
        "step {taken} s, compute_dt {largest} s"
    );
    let over = |dt: f64| {
        waves
            .coupling()
            .dissipation_force(waves.model(), waves.state(), dt)
    };
    let force = waves.force().unwrap();
    assert_eq!(force, over(taken).force());
    let peak = force.iter().map(|f| f[0].hypot(f[1])).fold(0.0, f64::max);
    assert!(peak > 0.0, "no breaking force");
    assert_ne!(force, over(largest).force());
}

/// The surf zone with the waves and the circulation running together
/// (`CoupledWaves2D` through `Simulation::run_with_exchange`, exchanging every
/// 30 s), the waves on 7 P2 elements of 50 m and the circulation on 14 of
/// 25 m, from still water and no waves: the swell comes in from the open
/// sea, the force ramps up over 300 s and is linear in time across each
/// interval. After half an hour the setup at the wall is 5.6051 cm (5.6076
/// cm at 15 minutes, 5.6051 cm still after an hour), the fixed point of the
/// coupling by passes (waves to steady on the level, circulation to steady
/// under the force, repeated: 5.793, 5.597, 5.6049, 5.6045 cm) to 1.0e-4 of
/// it, about as far as the passes have converged; the beach is at rest
/// (2.6e-4 m/s).
///
/// The circulation's Chézy friction becomes the wave-enhanced one at the
/// first exchange; at rest it changes nothing.
#[test]
fn breaking_waves_coupled_in_time_set_the_water_up() {
    let beach = Beach::new();
    let wave_beach = Beach::with(7, 2);
    let passes = setups_by_passes(&beach, &wave_beach, 4);

    let model = wave_beach.waves(std::f64::consts::FRAC_PI_2, None, 36);
    let n = model.zero_state();
    let cd = 2.5e-3;
    let physics = beach
        .physics()
        .with_implicit_friction(ChezyFriction2D::new(cd))
        .build();
    let mut waves = CoupledWaves2D::new(model, n, &physics, 0.0).with_ramp(300.0);
    let (interval, t_end) = (30.0, 1800.0);
    let mut sim = Simulation::new(physics, SSPRK3).with_callback_interval(interval);
    let mut q = beach.still_water();
    let mut setups = Vec::new();
    let result = sim.run_with_exchange(
        &mut q,
        0.0,
        t_end,
        interval,
        |physics, q, t, t_next| waves.exchange(physics, q, t, t_next),
        |q, _| setups.push(beach.setup(q)),
    );
    assert!(result.success, "{result:?}");
    let stats = waves.stats();
    assert_eq!(stats.exchanges, 60);
    assert_eq!(waves.time(), t_end);

    let setup = beach.setup(&q);
    let speed = q
        .hv_data()
        .iter()
        .zip(q.h_data())
        .map(|(hv, h)| (hv / h).abs())
        .fold(0.0f64, f64::max);
    // Half way, for how settled it is
    let half = setups[setups.len() / 2];
    println!(
        "setup at the wall coupled in time {:.4} cm (half way {:.4} cm), by passes \
         {:?} cm; |v| ≤ {speed:.1e} m/s; {} wave steps in {} exchanges",
        100.0 * setup,
        100.0 * half,
        passes.iter().map(|s| 100.0 * s).collect::<Vec<_>>(),
        stats.wave_steps,
        stats.exchanges
    );
    let fixed_point = passes[passes.len() - 1];
    assert!(
        (passes[passes.len() - 2] / fixed_point - 1.0).abs() < 1e-3,
        "the passes have not converged: {passes:?}"
    );
    assert!(
        (setup / fixed_point - 1.0).abs() < 3e-4,
        "coupled in time {setup} m, by passes {fixed_point} m"
    );
    assert!(speed < 1e-3, "not at rest: {speed} m/s");

    // The friction is the enhanced Chézy, per node, and stronger under waves
    let friction = sim.physics().friction.as_ref().expect("the friction");
    let n_total = beach.mesh.n_elements * beach.ops.n_nodes;
    assert_eq!(friction.n_total_nodes(), Some(n_total));
    let enhanced = (0..n_total)
        .filter(|&p| friction.damping_rate(p, 2.0, 0.2) > 1.01 * cd * 0.2 / 2.0)
        .count();
    assert!(
        enhanced > n_total / 2,
        "{enhanced} of {n_total} nodes enhanced"
    );
}

/// A weather model's wind on the coupled waves: sampled on the waves' own
/// mesh ([`GriddedAtmosphere2D::on_mesh`] of the circulation's), it reaches
/// every wave node at every wave step, not ramped (the circulation's forcing
/// ramps up over a day here), and the wind sea grows where it blows: the
/// alongshore wind rises from 2 m/s at the open sea to 20 m/s at the wall, and
/// after two minutes the energy travelling with it is ≥ 10× larger at the wall
/// than offshore.
#[test]
fn a_weather_models_wind_reaches_the_coupled_waves_per_node() {
    const T0: f64 = 1_750_000_000.0;
    let projection = LocalProjection::new(63.8, 8.7);
    // Alongshore (east), rising across the beach (north)
    let wind = |y: f64| 2.0 + 18.0 * y / Beach::L;
    let n = 9;
    let (lat0, lon0) = projection.xy_to_geo(-2e3, -2e3);
    let (lat1, lon1) = projection.xy_to_geo(2e3, 2e3);
    let lon: Vec<f64> = (0..n)
        .map(|i| lon0 + (lon1 - lon0) * i as f64 / (n - 1) as f64)
        .collect();
    let lat: Vec<f64> = (0..n)
        .map(|j| lat0 + (lat1 - lat0) * j as f64 / (n - 1) as f64)
        .collect();
    let grid = GeoGrid::regular(lon.clone(), lat.clone()).unwrap();
    let m = grid.len();
    let east: Vec<f32> = (0..2 * m)
        .map(|k| wind(projection.geo_to_xy(lat[k % m / n], lon[k % n]).1) as f32)
        .collect();
    let reader = AtmosphereReader::new(grid, vec![T0, T0 + 3600.0])
        .unwrap()
        .with_wind(
            FieldSeries::new(m, east),
            FieldSeries::new(m, vec![0.0; 2 * m]),
        );

    let beach = Beach::new();
    let wave_beach = Beach::with(7, 2);
    let physics = beach.physics().build();
    let atmosphere = GriddedAtmosphere2D::new(
        Arc::new(reader),
        &beach.mesh,
        &beach.ops,
        projection,
        ModelClock::new(T0),
    )
    .unwrap()
    .with_ramp_up(86_400.0);
    let on_waves = atmosphere
        .on_mesh(&wave_beach.mesh, &wave_beach.ops)
        .unwrap();
    // Calm, no swell: the wind's sea alone
    let mut model = wave_beach.waves(std::f64::consts::FRAC_PI_2, None, 36);
    model.sources = SourceTerms::swan_defaults(G).with_breaking(None);
    model.set_boundary_spectra(&vec![
        0.0;
        model.open_boundary_points().len()
            * model.grid.n_components()
    ]);
    let n_waves = model.zero_state();
    let mut waves = CoupledWaves2D::new(model, n_waves, &physics, 0.0)
        .with_ramp(300.0)
        .with_gridded_wind(on_waves);
    let mut q = beach.still_water();
    let mut sim = Simulation::new(physics, SSPRK3);
    let interval = 60.0;
    let result = sim.run_with_exchange(
        &mut q,
        0.0,
        2.0 * interval,
        interval,
        |physics, q, t, t_next| waves.exchange(physics, q, t, t_next),
        |_, _| {},
    );
    assert!(result.success, "{result:?}");

    let (model, state) = (waves.model(), waves.state());
    let ys = wave_beach.ys();
    let mut e = vec![0.0; model.grid.n_components()];
    let mut along = |p: usize| {
        model.energy_spectrum_into(state, p, &mut e);
        (0..model.grid.n_freq())
            .map(|i| e[model.grid.component(i, 0)])
            .sum::<f64>()
    };
    let (sea, wall) = (
        (0..ys.len())
            .min_by(|&a, &b| ys[a].total_cmp(&ys[b]))
            .unwrap(),
        (0..ys.len())
            .max_by(|&a, &b| ys[a].total_cmp(&ys[b]))
            .unwrap(),
    );
    for (p, &y) in ys.iter().enumerate() {
        let w = model.wind_at(p);
        assert!(
            (w.u10 - wind(y)).abs() < 1e-4 * wind(y) && w.direction.abs() < 1e-4,
            "node {p} at y = {y:.1} m: {w:?} against {} m/s along x",
            wind(y)
        );
    }
    let (at_sea, at_wall) = (along(sea), along(wall));
    println!("energy along the wind: {at_sea:.3e} at the open sea, {at_wall:.3e} at the wall");
    assert!(
        at_wall > 10.0 * at_sea && at_sea > 0.0,
        "{at_sea} {at_wall}"
    );
}

/// The longshore force by Longuet-Higgins (1970) per node, from the waves'
/// dissipation alone: with Snell's `k cos θ` (θ from the shore) constant along
/// each ray, the alongshore radiation stress changes only by what the waves
/// lose, `−∂S_xy/∂y = Σ (k cos θ/σ) D` (`D` the variance each component loses
/// per second, its rate `−B` times `E`), so `F_x = g Σ (k cos θ/σ) D`.
fn longuet_higgins_force(waves: &WaveModel2D, n: &WaveSolution) -> Vec<f64> {
    let grid = &waves.grid;
    let nf = grid.n_freq();
    let mut e = vec![0.0; grid.n_components()];
    let (mut a, mut b) = (e.clone(), e.clone());
    let mut k = vec![0.0; nf];
    (0..waves.n_points())
        .map(|p| {
            waves.energy_spectrum_into(n, p, &mut e);
            (0..nf).for_each(|i| k[i] = waves.wavenumber(i, p));
            let depth = waves.depth()[p];
            waves
                .sources
                .rates(grid, &e, &k, depth, Default::default(), &mut a, &mut b);
            let mut force = 0.0;
            for (i, &ki) in k.iter().enumerate() {
                for (j, cos) in grid.cos_theta.iter().enumerate() {
                    let c = grid.component(i, j);
                    let lost = -b[c] * e[c] * grid.d_sigma[i] * grid.d_theta;
                    force += ki * cos / grid.sigma[i] * lost;
                }
            }
            G * force
        })
        .collect()
}

/// The same swell, cos²⁰-spread about 20° off the normal (72 directions),
/// drives a current along the beach. With no lateral mixing each line across
/// the beach is its own balance, `C_d |u| u = F_x`, and Longuet-Higgins's
/// (1970) force from the dissipation (`longuet_higgins_force`) gives
/// `u = √(F_x/C_d)`: 0.60 m/s at most (the model's 0.61), two thirds of the
/// way across the surf zone. Where that force is over a tenth of its largest
/// the model's current follows it to 3.8 % (2.2 % at twice the resolution).
/// The radiation-stress force (in flux form) is 4–5 % of the largest force
/// above the dissipation's across the inner surf zone, 6.8 % at most: the
/// resolution of the wave field, which halves at 12.5 m elements. (The
/// element-local force without the face terms reached 9.5 % at element ends.)
///
/// Seaward of the breakers there is no dissipation and the force should
/// vanish: what remains is the discrete refraction (the direction bins), which
/// leaves the flux of alongshore momentum not quite constant, and a current of
/// −0.5 cm/s; it falls about fivefold for each halving of the bins.
#[test]
fn oblique_breaking_waves_drive_a_longshore_current() {
    let beach = Beach::new();
    let direction = std::f64::consts::FRAC_PI_2 - 20f64.to_radians();
    let waves = beach.waves(direction, Some(20), 72);
    let mut n = waves.zero_state();
    Beach::settle(&waves, &mut n);
    let force = WaveForce2D::new(&waves, &n);
    let lh = longuet_higgins_force(&waves, &n);
    let cd = 2.5e-3;
    let physics = beach
        .physics()
        .with_source(force.clone().with_ramp(300.0))
        .with_implicit_friction(ChezyFriction2D::new(cd))
        .build();
    let mut q = beach.still_water();
    let result = Simulation::new(physics, SSPRK3).run(&mut q, 0.0, 10_000.0);
    assert!(result.success, "{result:?}");

    let largest = lh.iter().cloned().fold(0.0, f64::max);
    let expected = |p: usize| (lh[p].max(0.0) / cd).sqrt();
    let (mut surf, mut offshore, mut fastest) = (0.0f64, 0.0f64, 0.0f64);
    for (p, &f) in lh.iter().enumerate() {
        let u = q.hu_data()[p] / q.h_data()[p];
        fastest = fastest.max(u);
        if f > 0.1 * largest {
            surf = surf.max((u / expected(p) - 1.0).abs());
        } else if f < 1e-3 * largest {
            offshore = offshore.max(u.abs());
        }
    }
    let excess = force
        .force()
        .iter()
        .zip(&lh)
        .map(|(f, lh)| (f[0] - lh).abs())
        .fold(0.0f64, f64::max)
        / largest;
    println!(
        "longshore current {fastest:.3} m/s (Longuet-Higgins {:.3}); off by ≤ {:.2} % in the \
         surf zone; force off the dissipation's by ≤ {:.1} % of its largest; |u| ≤ {offshore:.4} m/s offshore",
        (largest / cd).sqrt(),
        100.0 * surf,
        100.0 * excess
    );
    assert!(surf < 0.05, "the current is {surf} off Longuet-Higgins's");
    assert!(excess < 0.08, "the force is {excess} off the dissipation's");
    assert!(
        offshore < 0.02 * fastest,
        "an offshore current of {offshore} m/s"
    );
    assert!((fastest / (largest / cd).sqrt() - 1.0).abs() < 0.05);
}

/// The force of the waves' dissipation (Dingemans et al. 1987,
/// `WaveForce2D::from_dissipation`) in place of `−∇·S`, on the beach.
///
/// - The oblique swell of `oblique_breaking_waves_drive_a_longshore_current`:
///   per node its alongshore part is Longuet-Higgins's force
///   (`longuet_higgins_force`, computed independently) to round-off, so in
///   the surf zone the current follows `√(F_x/C_d)` to 0.58 % (`−∇·S`:
///   3.8 %). Seaward of the breakers there is no dissipation and no force;
///   the current there is 1.3 mm/s at most, carried across the breaker line
///   by the discretisation, where `−∇·S`'s discrete refraction drove 5 mm/s.
/// - Swell straight at the beach (one pass on still water each): the
///   cross-shore part pushes shoreward wherever the waves break, and sets
///   the water up at the wall by 5.73 cm, within 1.4 % of `−∇·S`'s 5.81 cm.
///   But there is no set-down seaward of the breakers (`−∇·S`: −0.68 cm),
///   where the waves shoal without losing energy.
#[test]
fn the_dissipation_force_drives_the_longshore_current_and_no_other() {
    let beach = Beach::new();
    let direction = std::f64::consts::FRAC_PI_2 - 20f64.to_radians();
    let waves = beach.waves(direction, Some(20), 72);
    let mut n = waves.zero_state();
    Beach::settle(&waves, &mut n);
    let force = WaveForce2D::from_dissipation(&waves, &n, 0.0);
    let lh = longuet_higgins_force(&waves, &n);
    let largest = lh.iter().cloned().fold(0.0, f64::max);
    let off = force
        .force()
        .iter()
        .zip(&lh)
        .map(|(f, lh)| (f[0] - lh).abs())
        .fold(0.0f64, f64::max);
    assert!(off <= 1e-12 * largest, "{off} off Longuet-Higgins's force");
    let cd = 2.5e-3;
    let physics = beach
        .physics()
        .with_source(force.clone().with_ramp(300.0))
        .with_implicit_friction(ChezyFriction2D::new(cd))
        .build();
    let mut q = beach.still_water();
    let result = Simulation::new(physics, SSPRK3).run(&mut q, 0.0, 10_000.0);
    assert!(result.success, "{result:?}");
    let expected = |p: usize| (lh[p].max(0.0) / cd).sqrt();
    let (mut surf, mut offshore, mut fastest) = (0.0f64, 0.0f64, 0.0f64);
    for (p, &f) in lh.iter().enumerate() {
        let u = q.hu_data()[p] / q.h_data()[p];
        fastest = fastest.max(u);
        if f > 0.1 * largest {
            surf = surf.max((u / expected(p) - 1.0).abs());
        } else if f < 1e-3 * largest {
            offshore = offshore.max(u.abs());
        }
    }

    // Straight at the beach: the setup and set-down of either force
    let waves = beach.waves(std::f64::consts::FRAC_PI_2, None, 36);
    let mut n = waves.zero_state();
    Beach::settle(&waves, &mut n);
    let levels = |force: WaveForce2D| {
        let physics = beach.physics().with_source(force.with_ramp(300.0)).build();
        let mut q = beach.still_water();
        let result = Simulation::new(physics, SSPRK3).run(&mut q, 0.0, 3000.0);
        assert!(result.success, "{result:?}");
        let eta = beach.eta(&q);
        let low = eta.iter().cloned().fold(f64::INFINITY, f64::min);
        (
            beach.setup(&q),
            low - eta[beach.balance(&vec![[0.0; 2]; eta.len()], &q)[0].0],
        )
    };
    let (setup_s, down_s) = levels(WaveForce2D::new(&waves, &n));
    let (setup_d, down_d) = levels(WaveForce2D::from_dissipation(&waves, &n, 0.0));
    println!(
        "dissipation force: longshore current {fastest:.3} m/s (Longuet-Higgins {:.3}), off by \
         ≤ {:.2} % in the surf zone, |u| ≤ {offshore:.2e} m/s offshore; straight in, setup at the \
         wall {:.2} cm (−∇·S: {:.2}), set-down {:.2} cm (−∇·S: {:.2})",
        (largest / cd).sqrt(),
        100.0 * surf,
        100.0 * setup_d,
        100.0 * setup_s,
        100.0 * down_d,
        100.0 * down_s,
    );
    assert!(surf < 0.05, "the current is {surf} off Longuet-Higgins's");
    assert!((fastest / (largest / cd).sqrt() - 1.0).abs() < 0.05);
    assert!(
        offshore < 3e-3 * fastest,
        "an offshore current of {offshore} m/s"
    );
    assert!(
        (setup_d / setup_s - 1.0).abs() < 0.03,
        "setup {setup_d} against {setup_s}"
    );
    assert!(
        down_s < -5e-3 && down_d > -1e-4,
        "set-down {down_d} against {down_s}"
    );
}

/// The oblique swell of `oblique_breaking_waves_drive_a_longshore_current` on
/// the 3D model (`Hydrostatic3D`, 10 uniform σ-levels, mode splitting, a
/// constant eddy viscosity ν = 0.01 m²/s and quadratic bottom drag
/// C_d = 2.5e-3 on the bottom layer). `WaveForce2D` on the barotropic module
/// reaches the columns as a depth-uniform acceleration `F/D` (the splitting
/// spreads the depth mean over the levels). Each column then holds the exact
/// parabola of a uniform force under a uniform viscosity: the stress
/// `ν ∂u/∂z` carries the force of the water above down to the bed, where the
/// drag takes it, so the bottom layer moves at `√(F/C_d)` (the 2D balance) and
/// `u(ζ) = u_b + (F/νD)((Dζ − ζ²/2) − (D ζ_b − ζ_b²/2))` above it (ζ the height
/// above the bed). The model's layers are that profile at their centres in
/// the discrete steady state too (the midpoint rule is exact for the linear
/// stress). Where the force is over a tenth of its largest, after 3 h from
/// rest, they follow it to 0.22 % of the fastest current (0.67 m/s) at the
/// nodes inside elements, and to 3.5 % on the faces between them across the
/// beach, where the numerical flux couples the two sides' different forces
/// (the 2D model's face nodes are as far off its balance). Where the force is
/// largest the surface runs at 0.66 m/s over a 0.61 m/s bottom layer.
///
/// The radiation stress is still depth-uniform here: no undertow, and no
/// vertical structure from where the waves break (TODO F.4: the vortex force,
/// or a surface-intensified breaking force with the Stokes return flow).
#[test]
fn a_longshore_current_in_3d_has_the_viscous_profile_of_a_uniform_force() {
    use dg_rs::physics::vertical_mixing::{ConstantMixing, Forcing};
    use dg_rs::physics::{BottomDrag3D, Hydrostatic3D, LinearEOS};
    use dg_rs::simulation::Simulation3D;
    use dg_rs::solver::state::Solution3D;
    use dg_rs::source::CoriolisSource2D;
    use dg_rs::time::ModeSplitIntegrator;
    use dg_rs::vertical::{SigmaGrid, UniformStretching};

    let beach = Beach::new();
    let direction = std::f64::consts::FRAC_PI_2 - 20f64.to_radians();
    let waves = beach.waves(direction, Some(20), 72);
    let mut n = waves.zero_state();
    Beach::settle(&waves, &mut n);
    let force = WaveForce2D::new(&waves, &n);
    let (nu, cd, levels, rho0) = (1e-2, 2.5e-3, 10, 1025.0);

    let swe = beach
        .physics()
        .with_source(force.clone().with_ramp(300.0))
        .build();
    let eos = LinearEOS::default();
    let sigma = Arc::new(SigmaGrid::new(levels, UniformStretching));
    let physics = Hydrostatic3D::new(
        beach.mesh.clone(),
        beach.ops.clone(),
        beach.geom.clone(),
        sigma.clone(),
        beach.bathymetry.clone(),
        Arc::new(CoriolisSource2D::f_plane(0.0)),
        eos,
        ConstantMixing::new(nu, nu),
        swe,
        Forcing {
            surface_stress: [0.0, 0.0],
            bottom_stress: [0.0, 0.0],
            surface_buoyancy_flux: 0.0,
        },
        G,
        rho0,
    )
    .with_bottom_drag(BottomDrag3D::quadratic(cd));
    let mut state = Solution3D::new(beach.mesh.n_elements, beach.ops.n_nodes, levels);
    state.temp.fill(eos.t0);
    state.salt.fill(eos.s0);
    physics.update_density(&mut state);
    let mut sim = Simulation3D::new(physics, ModeSplitIntegrator::new())
        .with_cfl(10.0)
        .with_dt_max(20.0);
    let result = sim.run(&mut state, 0.0, 10_800.0);
    assert!(result.success, "3D run failed: {:?}", result.error);

    // Each column against its parabola, where the force is over a tenth of
    // its largest
    let largest = force.force().iter().map(|f| f[0]).fold(0.0, f64::max);
    let nn = beach.ops.n_nodes;
    // Inside elements, and on the faces between them across the beach
    let (mut inside, mut on_faces) = (0.0f64, 0.0f64);
    let (mut fastest, mut strongest) = (0.0f64, (0.0, 0.0, 0.0));
    for (p, f) in force.force().iter().enumerate() {
        let column = &state.u[p * levels..(p + 1) * levels];
        fastest = fastest.max(column[levels - 1]);
        if f[0] < 0.1 * largest {
            continue;
        }
        let d = state.eta.data[p] - beach.bathymetry.data[p];
        let shape = |z: f64| d * z - 0.5 * z * z;
        let zb = (1.0 + sigma.sigma_rho()[0]) * d;
        let ub = (f[0] / cd).sqrt();
        for (l, &u) in column.iter().enumerate() {
            let z = (1.0 + sigma.sigma_rho()[l]) * d;
            let exact = ub + f[0] / (nu * d) * (shape(z) - shape(zb));
            let error = (u - exact).abs();
            if beach.ops.nodes_s[p % nn].abs() > 1.0 - 1e-12 {
                on_faces = on_faces.max(error);
            } else {
                inside = inside.max(error);
            }
        }
        if f[0] > strongest.0 {
            strongest = (f[0], column[0], column[levels - 1]);
        }
    }
    println!(
        "longshore current in 3D: off the viscous parabola by ≤ {:.2} % of the fastest \
         ({fastest:.3} m/s) inside elements, {:.2} % on their faces; where the force is largest \
         {:.3} m/s at the bed layer, {:.3} m/s at the surface",
        100.0 * inside / fastest,
        100.0 * on_faces / fastest,
        strongest.1,
        strongest.2
    );
    assert!(fastest > 0.5, "the current is {fastest} m/s");
    assert!(inside < 5e-3 * fastest, "off the parabola by {inside} m/s");
    assert!(
        on_faces < 0.05 * fastest,
        "off the parabola by {on_faces} m/s"
    );
}

/// What the depth limit (`SourceTerms::with_depth_limit`) takes from a
/// resolved surf zone, where breaking alone should do the work. The energy
/// it removes is not in `dissipation()`, so wherever it binds, the
/// dissipation's force (and with it the setup and the longshore current)
/// misses that share. On the plane beach (4 → 0.5 m, 25 m P2 elements), for
/// the oblique swell of the longshore gate and the swell straight in, it
/// compares the breaking dissipation over the beach, and the force, with and
/// without the limit.
#[test]
fn the_depth_limit_leaves_a_resolved_surf_zone_to_breaking() {
    let beach = Beach::new();
    let oblique = std::f64::consts::FRAC_PI_2 - 20f64.to_radians();
    let straight = std::f64::consts::FRAC_PI_2;
    for (name, direction, m, n_dir) in [
        ("oblique", oblique, Some(20), 72),
        ("straight", straight, None, 36),
    ] {
        let settled = |limit: bool| {
            let mut waves = beach.waves(direction, m, n_dir);
            waves.sources = waves.sources.clone().with_depth_limit(limit);
            let mut n = waves.zero_state();
            Beach::settle(&waves, &mut n);
            (waves, n)
        };
        let (free, n_free) = settled(false);
        let (limited, n_limited) = settled(true);
        // The breaking dissipation over the beach (m⁴/s per metre of width
        // times the width)
        let breaking = |waves: &WaveModel2D, n: &WaveSolution| {
            let d = waves.dissipation(n, 0.0);
            let nn = waves.ops.n_nodes;
            let mut total = 0.0;
            for k in 0..waves.mesh.n_elements {
                for a in 0..nn {
                    total +=
                        waves.ops.weights[a] * waves.geom.jacobian(k, a) * d[k * nn + a].breaking;
                }
            }
            total
        };
        // The largest H_rms/(γ d) without the limit
        let ratio = free
            .parameters(&n_free)
            .iter()
            .zip(free.depth())
            .map(|(p, &d)| p.hs / 2f64.sqrt() / (0.73 * d))
            .fold(0.0f64, f64::max);
        let removed = 1.0 - breaking(&limited, &n_limited) / breaking(&free, &n_free);
        let (f_free, f_limited) = (
            WaveForce2D::from_dissipation(&free, &n_free, 0.0),
            WaveForce2D::from_dissipation(&limited, &n_limited, 0.0),
        );
        let largest = f_free
            .force()
            .iter()
            .map(|f| f[0].hypot(f[1]))
            .fold(0.0f64, f64::max);
        let apart = f_free
            .force()
            .iter()
            .zip(f_limited.force())
            .map(|(a, b)| (a[0] - b[0]).hypot(a[1] - b[1]))
            .fold(0.0f64, f64::max);
        println!(
            "{name}: H_rms/(γd) ≤ {ratio:.3} without the limit; it takes {:.2} % of the breaking \
             dissipation, and moves the dissipation's force by ≤ {:.2} % of its largest",
            100.0 * removed,
            100.0 * apart / largest,
        );
        // Breaking holds H_rms to 0.68 γd at most (both swells), so the
        // limit never binds and the force is the same to the bit
        assert!(ratio < 0.8, "H_rms/(γd) up to {ratio}");
        assert_eq!(removed, 0.0);
        assert_eq!(apart, 0.0);
    }
}

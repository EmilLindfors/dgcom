//! Gate tests for the 3D model on general (non-parallelogram) quadrilaterals,
//! TODO P1.3 (the 3D half).
//!
//! Coastline-fitted meshes are made of general quadrilaterals. The 3D
//! horizontal kernels take the metric at every node, in the 2D module's
//! metric form (`MetricForm`: averaged under the split forms, conservative
//! under `Standard`): the layer continuity and the tracer and momentum
//! advection, the baroclinic pressure gradient as their adjoint with each
//! face node's normal, the tracer limiters' means with the nodal masses
//! `w_i J_i`, and the time step with each node's length. On distorted meshes
//! (a smooth field plus a checkerboard jitter, so that every element is a
//! general quadrilateral with an O(h) bilinear term) they must still:
//!
//! - keep a stratified fluid at rest at rest (the pressure gradient is exact
//!   at rest for a linear `ρ(z)`, over a seamount, in the whole mode-split
//!   model too, with the Kuzmin limiter);
//! - keep uniform tracers uniform under a tide over a sloping bed (tracer
//!   constancy: the layers carry exactly the water the free surface moved);
//! - conserve the tracer inventories and the volume, and on a flat periodic
//!   bed the momentum (the pressure gradient moves momentum only across
//!   faces);
//! - converge: the pressure gradient's element means at about N + 1 (the
//!   averaged metric is one order lower pointwise), the tracer advection
//!   at N + 1.

use std::f64::consts::PI;
use std::sync::Arc;

use dg_rs::boundary::Reflective2D;
use dg_rs::equations::ShallowWater2D;
use dg_rs::mesh::{Bathymetry2D, Mesh2D};
use dg_rs::operators::{DGOperators2D, GeometricFactors2D};
use dg_rs::physics::{ConstantMixing, Forcing, Hydrostatic3D, LinearEOS, PhysicsBuilder};
use dg_rs::solver::rhs::{MetricForm, PressureGradientForm, compute_pressure_gradient};
use dg_rs::solver::state::Solution3D;
use dg_rs::solver::{DGSolution2D, SWEFormulation2D, TracerLimiter3DConfig, TracerLimiterType3D};
use dg_rs::source::CoriolisSource2D;
use dg_rs::time::ModeSplitIntegrator;
use dg_rs::types::ElementIndex;
use dg_rs::vertical::{SigmaGrid, SongHaidvogelStretching, UniformStretching};

const G: f64 = 9.81;
const RHO0: f64 = 1025.0;

type Physics = Hydrostatic3D<LinearEOS, ConstantMixing, Reflective2D>;

/// A structured `nx × ny` mesh of `[0, lx] × [0, ly]` with its vertices
/// moved by a smooth field and a checkerboard jitter of a tenth of an
/// element, so that every element is a general quadrilateral. Periodic: the
/// field is periodic and moves every vertex (`nx`, `ny` even, so the jitter
/// is periodic too). Closed: the field vanishes on the boundary and only
/// interior vertices jitter, so the basin stays a rectangle.
fn distorted_mesh(lx: f64, ly: f64, nx: usize, ny: usize, periodic: bool) -> Mesh2D {
    let mut mesh = if periodic {
        assert!(
            nx.is_multiple_of(2) && ny.is_multiple_of(2),
            "periodic jitter needs even counts"
        );
        Mesh2D::uniform_periodic(0.0, lx, 0.0, ly, nx, ny)
    } else {
        Mesh2D::uniform_rectangle(0.0, lx, 0.0, ly, nx, ny)
    };
    let (hx, hy) = (lx / nx as f64, ly / ny as f64);
    for v in &mut mesh.vertices {
        let [x, y] = *v;
        let (i, j) = ((x / hx).round() as i64, (y / hy).round() as i64);
        let boundary = i == 0 || j == 0 || i == nx as i64 || j == ny as i64;
        let (dx, dy) = if periodic {
            let (sx, sy) = ((2.0 * PI * x / lx).sin(), (2.0 * PI * y / ly).sin());
            (
                0.04 * lx * (sx * sy + 0.5 * sy),
                0.04 * ly * (-0.7 * sx * sy + 0.4 * sx),
            )
        } else {
            let bump = (PI * x / lx).sin() * (PI * y / ly).sin();
            (
                0.06 * lx * bump * (2.0 * y / ly - 1.0),
                0.04 * ly * bump * (1.0 - 2.0 * x / lx),
            )
        };
        let jitter = if periodic || !boundary {
            if (i + j) % 2 == 0 { 0.1 } else { -0.1 }
        } else {
            0.0
        };
        *v = [x + dx + jitter * hx, y + dy + 0.5 * jitter * hy];
    }
    mesh
}

/// A 3D configuration on a distorted mesh.
struct Case {
    mesh: Arc<Mesh2D>,
    ops: Arc<DGOperators2D>,
    geom: Arc<GeometricFactors2D>,
    bathymetry: Arc<Bathymetry2D>,
    sigma: SigmaGrid,
}

impl Case {
    fn new(mesh: Mesh2D, order: usize, sigma: SigmaGrid, bed: impl Fn(f64, f64) -> f64) -> Self {
        let ops = DGOperators2D::new(order);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let non_affine = (0..mesh.n_elements)
            .filter(|&k| !geom.element_is_affine(k))
            .count();
        assert_eq!(
            non_affine, mesh.n_elements,
            "every element of the test mesh must be a general quadrilateral"
        );
        let bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, bed);
        Self {
            mesh: Arc::new(mesh),
            ops: Arc::new(ops),
            geom: Arc::new(geom),
            bathymetry: Arc::new(bathymetry),
            sigma,
        }
    }

    fn n_columns(&self) -> usize {
        self.mesh.n_elements * self.ops.n_nodes
    }

    /// Position of column `idx` (`element · n_nodes + node`).
    fn xy(&self, idx: usize) -> [f64; 2] {
        let (k, i) = (idx / self.ops.n_nodes, idx % self.ops.n_nodes);
        self.mesh.reference_to_physical(
            ElementIndex::new(k),
            self.ops.nodes_r[i],
            self.ops.nodes_s[i],
        )
    }

    /// The model with `eos`, `mixing`, Coriolis `f`, the `EntropyStable` 2D
    /// module and no surface or bottom stress.
    fn physics(&self, eos: LinearEOS, mixing: ConstantMixing, f: f64) -> Physics {
        let swe = PhysicsBuilder::swe_2d(
            self.mesh.clone(),
            self.ops.clone(),
            self.geom.clone(),
            ShallowWater2D::new(G),
            Reflective2D::default(),
        )
        .with_bathymetry(self.bathymetry.clone())
        .with_formulation(SWEFormulation2D::EntropyStable)
        .with_source(CoriolisSource2D::f_plane(f))
        .build();
        Hydrostatic3D::new(
            self.mesh.clone(),
            self.ops.clone(),
            self.geom.clone(),
            Arc::new(self.sigma.clone()),
            self.bathymetry.clone(),
            Arc::new(CoriolisSource2D::f_plane(f)),
            eos,
            mixing,
            swe,
            Forcing {
                surface_stress: [0.0, 0.0],
                bottom_stress: [0.0, 0.0],
                surface_heat_flux: 0.0,
            },
            G,
            RHO0,
        )
    }

    /// The state with `η = eta(x, y)` and `(u, v, T, S) = fields(x, y, z, σ)`
    /// in every layer, its density up to date.
    fn state(
        &self,
        physics: &Physics,
        eta: impl Fn(f64, f64) -> f64,
        fields: impl Fn(f64, f64, f64, f64) -> [f64; 4],
    ) -> Solution3D {
        let nl = self.sigma.n_levels();
        let mut state = Solution3D::new(self.mesh.n_elements, self.ops.n_nodes, nl);
        for idx in 0..self.n_columns() {
            let [x, y] = self.xy(idx);
            let e = eta(x, y);
            state.eta.data[idx] = e;
            let depth = e - self.bathymetry.data[idx];
            for (l, &s) in self.sigma.sigma_rho().iter().enumerate() {
                let n = idx * nl + l;
                [state.u[n], state.v[n], state.temp[n], state.salt[n]] =
                    fields(x, y, e + s * depth, s);
                state.ubar.data[idx] += self.sigma.d_sigma()[l] * state.u[n];
                state.vbar.data[idx] += self.sigma.d_sigma()[l] * state.v[n];
            }
        }
        physics.update_density(&mut state);
        state
    }

    /// `∫ Σ_l H_z φ_l dA` of a layer field.
    fn inventory(&self, state: &Solution3D, field: &[f64]) -> f64 {
        let nl = self.sigma.n_levels();
        let mut column = DGSolution2D::new(self.mesh.n_elements, self.ops.n_nodes);
        for (idx, c) in column.data.iter_mut().enumerate() {
            let depth = state.eta.data[idx] - self.bathymetry.data[idx];
            *c = (0..nl)
                .map(|l| depth * self.sigma.d_sigma()[l] * field[idx * nl + l])
                .sum();
        }
        column.integrate(&self.ops, &self.geom)
    }

    /// The pressure gradient force of `state` (x, y; `[column][level]`).
    fn pgf(
        &self,
        state: &Solution3D,
        rho_ref: f64,
        (form, metric): (PressureGradientForm, MetricForm),
    ) -> [Vec<f64>; 2] {
        let n = state.rho.len();
        let (mut fx, mut fy) = (vec![f64::NAN; n], vec![f64::NAN; n]);
        compute_pressure_gradient(
            state,
            &self.mesh,
            &self.bathymetry,
            &self.sigma,
            &self.ops,
            &self.geom,
            G,
            RHO0,
            rho_ref,
            0.0,
            form,
            metric,
            &mut fx,
            &mut fy,
        );
        [fx, fy]
    }

    /// A state with density `rho(x, y, z)` (only `η` and `ρ` are read by the
    /// pressure gradient).
    fn density_state(
        &self,
        eta: impl Fn(f64, f64) -> f64,
        rho: impl Fn(f64, f64, f64) -> f64,
    ) -> Solution3D {
        let nl = self.sigma.n_levels();
        let mut state = Solution3D::new(self.mesh.n_elements, self.ops.n_nodes, nl);
        for idx in 0..self.n_columns() {
            let [x, y] = self.xy(idx);
            let e = eta(x, y);
            state.eta.data[idx] = e;
            let depth = e - self.bathymetry.data[idx];
            for (l, &s) in self.sigma.sigma_rho().iter().enumerate() {
                state.rho[idx * nl + l] = rho(x, y, e + s * depth);
            }
        }
        state
    }
}

/// `steps` mode-split steps of `dt` on `state`, asserting that every step
/// stays finite and calling `check` after each.
fn run(
    physics: &Physics,
    state: &mut Solution3D,
    dt: f64,
    steps: usize,
    mut check: impl FnMut(&Solution3D, &Physics),
) {
    let mut integrator = ModeSplitIntegrator::new();
    for n in 0..steps {
        physics.update_density(state);
        integrator.step(state, physics, dt, n as f64 * dt);
        physics.post_process(state);
        assert!(
            max_or_nan(
                state
                    .u
                    .iter()
                    .chain(&state.v)
                    .chain(&state.temp)
                    .map(|x| x.abs())
            )
            .is_finite(),
            "step {n}: the run blew up"
        );
        check(state, physics);
    }
}

/// The largest value, or NaN if any is NaN (`f64::max` drops NaN, which
/// would let a blown-up run pass).
fn max_or_nan(values: impl IntoIterator<Item = f64>) -> f64 {
    values
        .into_iter()
        .fold(0.0, |m, x| if x.is_nan() || x > m { x } else { m })
}

fn largest_force([fx, fy]: &[Vec<f64>; 2]) -> f64 {
    max_or_nan(fx.iter().zip(fy).map(|(x, y)| x.hypot(*y)))
}

/// Both pressure gradient forms, each with the volume form of either 2D
/// module (`Standard`: each node's chain rule; the split forms: the pair's
/// averaged metric).
const FORMS: [(PressureGradientForm, MetricForm); 4] = [
    (PressureGradientForm::SigmaPairs, MetricForm::Averaged),
    (PressureGradientForm::SigmaPairs, MetricForm::Conservative),
    (PressureGradientForm::ConstantDepth, MetricForm::Averaged),
    (
        PressureGradientForm::ConstantDepth,
        MetricForm::Conservative,
    ),
];

/// A seamount `400 m · (1 − 0.9 e^{−r²/(4 km)²})` in the middle of a doubly
/// periodic 24 km square.
fn seamount_bed(x: f64, y: f64) -> f64 {
    let r2 = (x - 12e3).powi(2) + (y - 12e3).powi(2);
    -400.0 * (1.0 - 0.9 * (-r2 / 4e3_f64.powi(2)).exp())
}

/// Constant `N² = 1e-4 s⁻²`.
fn linear_density(z: f64) -> f64 {
    RHO0 - RHO0 * 1e-4 / G * z
}

/// A linear stratification at rest over a seamount is balanced to
/// round-off at every order, on uniform and stretched levels, in both
/// forms, for the full and the baroclinic-only pressure: each pair's
/// pressure difference vanishes whatever the element's shape.
#[test]
fn constant_n2_at_rest_is_balanced_on_distorted_elements() {
    for order in 1..=4 {
        for sigma in [
            SigmaGrid::new(12, UniformStretching),
            SigmaGrid::new(10, SongHaidvogelStretching::new(5.0, 0.4, 20.0)),
        ] {
            let case = Case::new(
                distorted_mesh(24e3, 24e3, 6, 6, true),
                order,
                sigma,
                seamount_bed,
            );
            let state = case.density_state(|_, _| 0.0, |_, _, z| linear_density(z));
            for rho_ref in [RHO0, 0.0] {
                for form in FORMS {
                    // Measured ≤ 7.8e-15 m/s²
                    let force = largest_force(&case.pgf(&state, rho_ref, form));
                    assert!(
                        force < 1e-13,
                        "P{order}, rho_ref {rho_ref}, {form:?}: spurious |F| = {force:.3e} m/s²"
                    );
                }
            }
        }
    }
}

/// A tilted free surface over constant density gives the full force `−g∇η`
/// and no baroclinic force, and a horizontally linear density over a flat bed
/// gives `−(g/ρ₀)(η − z)∇ρ`, exactly, in a closed distorted basin (the
/// bilinear map is in the element space, so linear fields are differentiated
/// exactly by each node's chain rule).
#[test]
fn linear_pressure_fields_are_exact_on_distorted_elements() {
    let case = Case::new(
        distorted_mesh(4000.0, 3000.0, 6, 4, false),
        2,
        SigmaGrid::new(6, UniformStretching),
        |_, _| -50.0,
    );
    let (sx, sy) = (1e-4, -0.6e-4);
    let tilted = case.density_state(|x, y| sx * x + sy * y, |_, _, _| RHO0);
    for form in FORMS {
        let [fx, fy] = case.pgf(&tilted, 0.0, form);
        let error = max_or_nan(
            fx.iter()
                .zip(&fy)
                .map(|(x, y)| (x + G * sx).hypot(y + G * sy)),
        );
        // Measured 9.7e-16 m/s² against 1e-3
        assert!(
            error < 1e-12,
            "{form:?}: full PGF off −g∇η by {error:.3e} m/s²"
        );
        let baroclinic = largest_force(&case.pgf(&tilted, RHO0, form));
        assert!(
            baroclinic < 1e-13,
            "{form:?}: baroclinic-only PGF {baroclinic:.3e}"
        );
    }

    let (a, b) = (2e-4, -1e-4);
    let fronted = case.density_state(|_, _| 0.0, |x, y, _| RHO0 + a * x + b * y);
    let nl = case.sigma.n_levels();
    for form in FORMS {
        let [fx, fy] = case.pgf(&fronted, RHO0, form);
        let mut error = 0.0_f64;
        for idx in 0..case.n_columns() {
            let depth = -case.bathymetry.data[idx];
            for (l, &s) in case.sigma.sigma_rho().iter().enumerate() {
                let scale = -G / RHO0 * (0.0 - s * depth);
                let n = idx * nl + l;
                error = max_or_nan([error, (fx[n] - scale * a).hypot(fy[n] - scale * b)]);
            }
        }
        // Measured 9.7e-16 m/s² against 1e-4
        assert!(
            error < 1e-13,
            "{form:?}: off −(g/ρ₀)(η − z)∇ρ by {error:.3e} m/s²"
        );
    }
}

/// Over a flat bed with a flat surface the pressure gradient only moves
/// momentum between elements: summed over a periodic domain, `∫ Σ_l H_z F`
/// vanishes for any density field, here a rough one (discontinuous between
/// nodes and levels). Summation by parts with the exact metric identities of
/// the bilinear map leaves only the face values, and the central face
/// pressure is single-valued.
#[test]
fn the_pressure_gradient_moves_momentum_only_across_faces_on_distorted_elements() {
    for order in [1, 3] {
        let case = Case::new(
            distorted_mesh(4000.0, 4000.0, 4, 4, true),
            order,
            SigmaGrid::new(5, UniformStretching),
            |_, _| -30.0,
        );
        let mut state = case.density_state(|_, _| 0.0, |_, _, _| RHO0);
        for (n, rho) in state.rho.iter_mut().enumerate() {
            // A hash of the index: rough on every scale
            let h = ((n as u64).wrapping_mul(2654435761) % 1000) as f64 / 1000.0;
            *rho = RHO0 + 3.0 * h;
        }
        let nl = case.sigma.n_levels();
        for form in FORMS {
            let force = case.pgf(&state, RHO0, form);
            let (mut total, mut scale) = ([0.0; 2], 0.0);
            for k in 0..case.mesh.n_elements {
                for i in 0..case.ops.n_nodes {
                    let idx = k * case.ops.n_nodes + i;
                    let mass = case.geom.node_mass(k, i) * 30.0;
                    for (l, &ds) in case.sigma.d_sigma().iter().enumerate() {
                        for c in 0..2 {
                            let f = mass * ds * force[c][idx * nl + l];
                            total[c] += f;
                            scale += f.abs();
                        }
                    }
                }
            }
            let imbalance = total[0].hypot(total[1]) / scale;
            // Measured ≤ 5.3e-17
            assert!(
                imbalance < 1e-13,
                "P{order}, {form:?}: net force {imbalance:.3e} of the total"
            );
        }
    }
}

/// A smooth baroclinic field over a varying bed,
/// `ρ′ = a sin(kx) sin(ky) e^{z/H}`, `∂p/∂x|_z = g a k H cos(kx) sin(ky)
/// (1 − e^{z/H})`, under joint refinement of the distorted mesh and the
/// levels (4 → 32 elements across, 10 → 80 levels). Measured orders of the
/// largest error over the four meshes:
///
/// | | P2 | P3 |
/// |---|---|---|
/// | σ-pairs, chain rule (`Conservative`) | 1.65, 1.91, 1.92 | 2.86, 2.98, 2.84 |
/// | σ-pairs, averaged metric | 1.48, 1.68, 1.08 | 2.38, 2.10, 2.11 |
/// | constant depth, chain rule | 1.64, 1.71, 1.70 | 2.37, 1.27, 0.86 |
/// | σ-pairs, element means, either metric | 2.15, 3.22, 2.86 | 2.23, 2.83, 2.92 |
///
/// Each node's chain rule converges at the order of the nodal derivative,
/// `N`, as on rectangles. The averaged metric (the adjoint of the split
/// forms' advection) loses one order pointwise on these meshes, which are
/// not asymptotically parallelograms (the jitter is a fixed fraction of the
/// element): `D(J∇r·p)` aliases the O(1) relative variation of the metric
/// over the element. That error has zero mean on every element: the
/// element means of the two forms are the same, and converge at about
/// `N + 1` (P2: 3.9e-7 → 7.6e-10 m/s²). The constant-depth form at P3 is
/// limited by its first-order one-sided pairs near the bed (as on
/// rectangles), and is not checked there.
#[test]
fn the_pressure_gradient_converges_on_distorted_elements() {
    let (a, h_scale, length) = (0.5, 40.0, 8000.0);
    let k = 2.0 * PI / length;
    let density =
        |x: f64, y: f64, z: f64| RHO0 + a * (k * x).sin() * (k * y).sin() * (z / h_scale).exp();
    let bed = |x: f64, y: f64| -100.0 - 20.0 * (k * x).cos() * (k * y).cos();
    let rates =
        |errors: &[f64]| -> Vec<f64> { errors.windows(2).map(|p| (p[0] / p[1]).log2()).collect() };
    for order in [2, 3] {
        for (form, metric) in FORMS {
            if form == PressureGradientForm::ConstantDepth && order == 3 {
                continue;
            }
            // (largest error, largest error of an element-layer mean)
            let errors: Vec<(f64, f64)> = [(4, 10), (8, 20), (16, 40), (32, 80)]
                .iter()
                .map(|&(n, levels)| {
                    let case = Case::new(
                        distorted_mesh(length, length, n, n, true),
                        order,
                        SigmaGrid::new(levels, UniformStretching),
                        bed,
                    );
                    let state = case.density_state(|_, _| 0.0, density);
                    let [fx, fy] = case.pgf(&state, RHO0, (form, metric));
                    let (mut largest, mut largest_mean) = (0.0_f64, 0.0_f64);
                    for el in 0..case.mesh.n_elements {
                        for (l, &s) in case.sigma.sigma_rho().iter().enumerate() {
                            let (mut mean, mut mass) = ([0.0; 2], 0.0);
                            for i in 0..case.ops.n_nodes {
                                let idx = el * case.ops.n_nodes + i;
                                let [x, y] = case.xy(idx);
                                let depth = -case.bathymetry.data[idx];
                                let profile = -G / RHO0
                                    * a
                                    * k
                                    * h_scale
                                    * (1.0 - (s * depth / h_scale).exp());
                                let n = idx * levels + l;
                                let error = [
                                    fx[n] - profile * (k * x).cos() * (k * y).sin(),
                                    fy[n] - profile * (k * x).sin() * (k * y).cos(),
                                ];
                                largest = max_or_nan([largest, error[0].hypot(error[1])]);
                                let m = case.geom.node_mass(el, i);
                                mean = [mean[0] + m * error[0], mean[1] + m * error[1]];
                                mass += m;
                            }
                            largest_mean =
                                max_or_nan([largest_mean, mean[0].hypot(mean[1]) / mass]);
                        }
                    }
                    (largest, largest_mean)
                })
                .collect();
            let (largest, means): (Vec<f64>, Vec<f64>) = errors.into_iter().unzip();
            let (orders, mean_orders) = (rates(&largest), rates(&means));
            println!(
                "P{order} {form:?}, {metric:?}: largest {largest:?}, orders {orders:.2?}; \
                 element means {means:?}, orders {mean_orders:.2?}"
            );
            // The chain rule at N, the averaged metric one lower
            let n = order as f64;
            let expected = match metric {
                MetricForm::Conservative => n - 0.4,
                MetricForm::Averaged => n - 1.2,
            };
            for &rate in &orders {
                assert!(
                    rate > expected,
                    "P{order} {form:?}, {metric:?}: order {rate:.2} (errors {largest:?})"
                );
            }
            // The resolved force, the element means, at least at N
            if form == PressureGradientForm::SigmaPairs {
                for &rate in &mean_orders[1..] {
                    assert!(
                        rate > n - 0.3,
                        "P{order} {metric:?}: element-mean order {rate:.2} (errors {means:?})"
                    );
                }
            }
        }
    }
}

/// A closed 1 km × 500 m basin (8 × 4 distorted P2 elements) over a bed
/// sloping from 12 to 4 m with a cross-basin ripple, a 0.5 m seiche
/// `η = 0.5 cos(πx/L)`, a sheared flow `u = 0.1(σ + ½)`, three levels, the
/// T/S-dependent EOS, vertical mixing and the horizontal Kuzmin limiter.
fn tidal_basin() -> Case {
    let length = 1000.0;
    Case::new(
        distorted_mesh(length, 500.0, 8, 4, false),
        2,
        SigmaGrid::new(3, UniformStretching),
        move |x, y| -12.0 + 8.0 * x / length + 0.5 * (PI * y / 500.0).cos(),
    )
}

/// The model of [`tidal_basin`], with the Kuzmin limiter.
fn tidal_physics(case: &Case) -> Physics {
    case.physics(LinearEOS::default(), ConstantMixing::new(1e-3, 1e-4), 0.0)
        .with_tracer_limiter(TracerLimiter3DConfig {
            limiter_type: TracerLimiterType3D::HorizontalKuzmin { relaxation: 1.0 },
            ..TracerLimiter3DConfig::default()
        })
}

/// One seiche period of [`tidal_basin`] in 40 steps, with the tracers from
/// `tracers(x, σ)`, calling `check` after every step.
fn run_tide(
    case: &Case,
    physics: &Physics,
    tracers: impl Fn(f64, f64) -> (f64, f64),
    check: impl FnMut(&Solution3D, &Physics),
) -> Solution3D {
    let mut state = case.state(
        physics,
        |x, _| 0.5 * (PI * x / 1000.0).cos(),
        |x, _, _, s| {
            let (t, salt) = tracers(x, s);
            [0.1 * (s + 0.5), 0.0, t, salt]
        },
    );
    let period = 2.0 * 1000.0 / (G * 8.0).sqrt();
    run(physics, &mut state, period / 40.0, 40, check);
    state
}

/// Tracer constancy: uniform T and S stay uniform under the tide on general
/// quadrilaterals, with the limiter on, and `Ω` closes at the surface (the
/// layer transports are the barotropic pass's on every element shape).
#[test]
fn uniform_tracers_stay_uniform_under_a_tide_on_distorted_elements() {
    let case = tidal_basin();
    let physics = tidal_physics(&case);
    let eos = LinearEOS::default();
    let (t0, s0) = (eos.t0 + 2.3, eos.s0 - 0.9);
    let (mut drift, mut residual, mut omega, mut speed) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    run_tide(
        &case,
        &physics,
        |_, _| (t0, s0),
        |s, p| {
            drift = max_or_nan(
                s.temp
                    .iter()
                    .map(|t| (t - t0).abs())
                    .chain(s.salt.iter().map(|x| (x - s0).abs()))
                    .chain([drift]),
            );
            residual = residual.max(p.last_surface_residual());
            omega = max_or_nan(s.w.iter().map(|w| w.abs()).chain([omega]));
            speed = max_or_nan(
                s.u.iter()
                    .zip(&s.v)
                    .map(|(u, v)| u.hypot(*v))
                    .chain([speed]),
            );
        },
    );
    assert!(speed > 0.1, "test regime: no tide ({speed:.3e} m/s)");
    assert!(omega > 1e-5, "test regime: Ω {omega:.2e}");
    // Measured 7.0e-13, Ω surface residual 3.5e-18 against Ω of 7.3e-3 m/s
    // (the P4.2 gate on rectangles: 9.3e-13)
    assert!(
        drift < 1e-11 * eos.s0,
        "uniform tracers drifted by {drift:.3e} under the tide"
    );
    assert!(
        residual < 1e-10 * omega,
        "Ω surface residual {residual:.2e} (Ω scale {omega:.2e})"
    );
}

/// Conservation: with gradients in T and S (and the baroclinic flow they
/// drive), vertical diffusion and the Kuzmin limiter, the tracer inventories
/// `∫ Σ H_z C dA` and the volume `∫ D dA` keep their values to round-off.
#[test]
fn inventories_and_volume_are_conserved_on_distorted_elements() {
    let case = tidal_basin();
    let physics = tidal_physics(&case);
    let tracers = |x: f64, s: f64| (10.0 + 3.0 * x / 1000.0 - 2.0 * s, 33.0 + x / 1000.0 + s);
    let initial = case.state(
        &physics,
        |x, _| 0.5 * (PI * x / 1000.0).cos(),
        |x, _, _, s| {
            let (t, salt) = tracers(x, s);
            [0.1 * (s + 0.5), 0.0, t, salt]
        },
    );
    let ones = vec![1.0; initial.temp.len()];
    let (t0, s0, v0) = (
        case.inventory(&initial, &initial.temp),
        case.inventory(&initial, &initial.salt),
        case.inventory(&initial, &ones),
    );
    let (mut tracer_error, mut volume_error) = (0.0_f64, 0.0_f64);
    run_tide(&case, &physics, tracers, |s, _| {
        tracer_error = max_or_nan([
            tracer_error,
            (case.inventory(s, &s.temp) - t0).abs() / t0,
            (case.inventory(s, &s.salt) - s0).abs() / s0,
        ]);
        volume_error = max_or_nan([volume_error, (case.inventory(s, &ones) - v0).abs() / v0]);
    });
    // Measured 2.9e-15, volume 1.3e-14 (the P4.2 gate on rectangles: 3.1e-15)
    assert!(
        tracer_error < 1e-12,
        "tracer inventory drifted by {tracer_error:.3e} (relative)"
    );
    assert!(
        volume_error < 1e-13,
        "volume drifted by {volume_error:.3e} (relative)"
    );
}

/// Momentum: over a flat bed in a doubly periodic domain without rotation,
/// a sheared flow with a free-surface bump and horizontal viscosity keep the
/// total momentum `∫ Σ_l H_z u_l dA` to round-off: the 2D pass conserves it,
/// and the 3D tendencies `G` hands it (the advection of the shear in split
/// form with the averaged metric, the viscosity) only move it between
/// elements. (A horizontal density gradient under the tilted σ-levels adds
/// the pressure gradient's truncation error, which conserves momentum only
/// as the scheme converges: 8e-12 of the transport here on rectangles,
/// 5e-10 on the distorted mesh. On level σ-surfaces it is exact, see
/// `the_pressure_gradient_moves_momentum_only_across_faces_on_distorted_elements`.)
#[test]
fn momentum_is_conserved_over_a_flat_periodic_bed_on_distorted_elements() {
    let (length, depth) = (1000.0, 20.0);
    let case = Case::new(
        distorted_mesh(length, length, 4, 4, true),
        2,
        SigmaGrid::new(4, UniformStretching),
        move |_, _| -depth,
    );
    let physics = case
        .physics(LinearEOS::default(), ConstantMixing::new(1e-3, 0.0), 0.0)
        .with_horizontal_viscosity(1.0);
    let k = 2.0 * PI / length;
    let mut state = case.state(
        &physics,
        |x, y| 0.05 * (k * x).sin() * (k * y).cos(),
        |x, y, _, s| {
            [
                0.05 + 0.1 * (k * y).sin() * (s + 0.5),
                -0.03 + 0.08 * (k * x).cos() * s,
                10.0,
                35.0,
            ]
        },
    );
    let momentum = |s: &Solution3D| [case.inventory(s, &s.u), case.inventory(s, &s.v)];
    let initial = momentum(&state);
    // The transport scale: 0.1 m/s through the domain
    let scale = 0.1 * depth * length * length;
    let mut error = 0.0_f64;
    let first_u = state.u.clone();
    run(&physics, &mut state, 5.0, 30, |s, _| {
        let [mx, my] = momentum(s);
        error = max_or_nan([
            error,
            (mx - initial[0]).abs() / scale,
            (my - initial[1]).abs() / scale,
        ]);
    });
    let changed = max_or_nan(state.u.iter().zip(&first_u).map(|(a, b)| (a - b).abs()));
    assert!(
        changed > 1e-3,
        "test regime: the flow changed by {changed:.3e} m/s"
    );
    // Measured 5.1e-14
    assert!(
        error < 1e-12,
        "momentum drifted by {error:.3e} of the transport scale"
    );
}

/// A stratified fluid at rest over a steep seamount on distorted elements
/// stays at rest in the whole mode-split model, with the σ-pairs pressure
/// gradient, the split-form advection and the Kuzmin limiter: the pair is
/// consistent in energy on any element shape (the averaged-metric pressure
/// gradient is the adjoint of the averaged-metric split form), and constant
/// `N²` is exact at rest, so only round-off moves. Measured 3.6e-11 m/s in
/// 36 h (7.9e-11 on rectangles). With the layer continuity in the
/// collocated form under the `EntropyStable` 2D module the splitter's nodal
/// identity failed in every element, and the flow reached 0.9 m/s.
#[test]
fn a_stratified_seamount_at_rest_stays_at_rest_on_distorted_elements() {
    let case = Case::new(
        distorted_mesh(24e3, 24e3, 6, 6, true),
        2,
        SigmaGrid::new(10, SongHaidvogelStretching::new(5.0, 0.4, 20.0)),
        seamount_bed,
    );
    let physics = case
        .physics(LinearEOS::default(), ConstantMixing::new(1e-4, 0.0), 1.2e-4)
        .with_tracer_limiter(TracerLimiter3DConfig {
            limiter_type: TracerLimiterType3D::HorizontalKuzmin { relaxation: 1.0 },
            ..TracerLimiter3DConfig::default()
        });
    let eos = LinearEOS::default();
    // 3 kg/m³ from the surface to 400 m: N² ≈ 7e-5 s⁻²
    let rho = |z: f64| RHO0 - 3.0 * (1.0 + z / 400.0);
    let mut state = case.state(
        &physics,
        |_, _| 0.0,
        |_, _, z, _| {
            [
                0.0,
                0.0,
                eos.t0 + (1.0 - rho(z) / eos.rho0) / eos.alpha,
                eos.s0,
            ]
        },
    );
    run(&physics, &mut state, 300.0, 432, |_, _| ());
    let speed = max_or_nan(state.u.iter().zip(&state.v).map(|(u, v)| u.hypot(*v)));
    println!("seamount at rest on distorted elements: {speed:.3e} m/s after 36 h");
    assert!(speed < 1e-9, "the seamount spun up {speed:.3e} m/s in 36 h");
}

/// A smooth tracer carried by a uniform current over a flat bed through a
/// doubly periodic distorted mesh, against the exact translate: N + 1 at P2
/// (the density does not depend on it, and a uniform flow stays uniform on
/// any element shape). The step follows the element size, so SSP-RK3's error
/// falls at third order with it.
#[test]
fn tracer_advection_converges_on_distorted_elements() {
    let (length, depth, (u, v)) = (1000.0, 10.0, (0.5, 0.25));
    let k = 2.0 * PI / length;
    let tracer = |x: f64, y: f64| 10.0 + (k * x).sin() * (k * y).sin();
    let duration = 400.0;
    let errors: Vec<f64> = [4, 8, 16]
        .iter()
        .map(|&n| {
            let case = Case::new(
                distorted_mesh(length, length, n, n, true),
                2,
                SigmaGrid::new(2, UniformStretching),
                move |_, _| -depth,
            );
            let passive = LinearEOS {
                alpha: 0.0,
                beta: 0.0,
                ..LinearEOS::default()
            };
            let physics = case.physics(passive, ConstantMixing::new(0.0, 0.0), 0.0);
            let mut state = case.state(
                &physics,
                |_, _| 0.0,
                |x, y, _, _| [u, v, tracer(x, y), 35.0],
            );
            let steps = 50 * n / 4;
            run(
                &physics,
                &mut state,
                duration / steps as f64,
                steps,
                |_, _| (),
            );
            let nl = case.sigma.n_levels();
            let mut column = DGSolution2D::new(case.mesh.n_elements, case.ops.n_nodes);
            for (idx, c) in column.data.iter_mut().enumerate() {
                let [x, y] = case.xy(idx);
                let exact = tracer(x - u * duration, y - v * duration);
                *c = (0..nl)
                    .map(|l| case.sigma.d_sigma()[l] * (state.temp[idx * nl + l] - exact).powi(2))
                    .sum();
            }
            (column.integrate(&case.ops, &case.geom) / (length * length)).sqrt()
        })
        .collect();
    let orders: Vec<f64> = errors.windows(2).map(|p| (p[0] / p[1]).log2()).collect();
    println!("tracer advection, P2: L2 errors {errors:?}, orders {orders:.2?}");
    for &rate in &orders {
        assert!(rate > 2.5, "order {rate:.2} (errors {errors:?})");
    }
}

/// The baroclinic step of an elongated element follows its narrow side, as
/// the 2D step does: on 10 km × 1 km P1 rectangles at rest the length is
/// `1/(1/Δx + 1/Δy)` = 909 m (it was `√J` = 1.58 km, √10/2 times the
/// limit of the narrow side). Squares are unchanged (`√J`).
#[test]
fn the_3d_step_follows_the_narrow_side_of_elongated_elements() {
    for (lx, ly, expected_length) in [
        (40e3, 4e3, 1.0 / (1.0 / 10e3 + 1.0 / 1e3)),
        (40e3, 40e3, 5e3),
    ] {
        let mesh = Mesh2D::uniform_periodic(0.0, lx, 0.0, ly, 4, 4);
        let ops = Arc::new(DGOperators2D::new(1));
        let geom = Arc::new(GeometricFactors2D::compute(&mesh, &ops));
        let bathymetry = Bathymetry2D::constant(mesh.n_elements, ops.n_nodes, -100.0);
        let case = Case {
            mesh: Arc::new(mesh),
            ops,
            geom,
            bathymetry: Arc::new(bathymetry),
            sigma: SigmaGrid::new(5, UniformStretching),
        };
        let physics = case.physics(LinearEOS::default(), ConstantMixing::new(0.0, 0.0), 0.0);
        let eos = LinearEOS::default();
        let state = case.state(
            &physics,
            |_, _| 0.0,
            |_, _, _, _| [0.0, 0.0, eos.t0, eos.s0],
        );
        let cfl = 1.0;
        // (N+1)² = 4, the speed the floor of unstratified water at rest
        let expected = cfl * expected_length / Physics::MIN_INTERNAL_WAVE_SPEED / 4.0;
        let dt = physics.compute_dt(&state, cfl);
        assert!(
            (dt / expected - 1.0).abs() < 1e-12,
            "{lx} × {ly} m domain: {dt} s against {expected} s"
        );
    }
}

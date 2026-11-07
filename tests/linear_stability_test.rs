//! Linear stability limit of the 2D SWE DGSEM under SSP-RK3 and SSP-RK(4,3)
//! (TODO P2.1).
//!
//! The Jacobian of the RHS (central differences) around an equilibrium on a
//! periodic mesh, its eigenvalues λ, and the largest CFL (in the units of
//! `compute_dt_swe_2d`) for which every `Δt·λ` lies in the scheme's
//! stability region `|R(z)| ≤ 1`.
//!
//! On a uniform mesh with a uniform state the operator is translation
//! invariant, and its spectrum on the infinite mesh is the union over Bloch
//! wavenumbers θ of the eigenvalues of the element symbol
//! `A(θ) = Σ_e J_{e,0} e^{−iθ·p_e}` (`p_e` the offset of element e from
//! element 0): one small matrix per θ instead of the whole Jacobian. The gate
//! uses it; distorted meshes and beds, which only raise the limit, need the
//! whole Jacobian (`print_linear_cfl_limits`, ignored: slow in debug builds).
//!
//! Elements with a dry node replace the volume term by the `WetDry` subcell
//! finite volumes, whose limit is lower and is measured separately, with
//! every element on the subcells (`subcell_*`).

use std::f64::consts::PI;
use std::sync::Arc;

use dgcom::boundary::Reflective2D;
use dgcom::equations::ShallowWater2D;
use dgcom::mesh::{Bathymetry2D, Mesh2D};
use dgcom::operators::{DGOperators2D, GeometricFactors2D};
use dgcom::physics::{PhysicsBuilder, PhysicsModule, SWEPhysics2D};
use dgcom::solver::{
    SWEFormulation2D, SWESolution2D, SWEState2D, StandardLimiter2D, WetDryConfig,
    linear_cfl_subcells_swe_2d, linear_cfl_swe_2d, positivity_cfl_swe_2d,
};
use dgcom::time::SspScheme;
use dgcom::types::{Depth, ElementIndex};
use faer::Mat;

const G: f64 = 9.81;
const DEPTH: f64 = 10.0;

type C64 = faer::c64;

/// Stability polynomial of SSP-RK3.
fn r_rk3(z: C64) -> C64 {
    let one = C64::new(1.0, 0.0);
    one + z + z * z / 2.0 + z * z * z / 6.0
}

/// Stability polynomial of SSP-RK(4,3):
/// ⅔(1 + z/2) + ⅓(1 + z/2)⁴ = 1 + z + z²/2 + z³/6 + z⁴/48.
fn r_rk43(z: C64) -> C64 {
    let one = C64::new(1.0, 0.0);
    one + z + z * z / 2.0 + z * z * z / 6.0 + z * z * z * z / 48.0
}

/// A scheme's stability polynomial R(z).
type StabilityPolynomial = fn(C64) -> C64;

const SCHEMES: [(SspScheme, StabilityPolynomial); 2] =
    [(SspScheme::Rk3, r_rk3), (SspScheme::Rk43, r_rk43)];

#[derive(Clone, Copy, Debug)]
struct Case {
    order: usize,
    formulation: SWEFormulation2D,
    /// Element width in x and y
    size: (f64, f64),
    /// Flow speed as a fraction of the celerity, diagonal
    froude: f64,
    /// Vertices moved by a smooth periodic field (general quadrilaterals)
    distorted: bool,
    /// Relative amplitude of a sinusoidal bed (lake at rest over it)
    bed_amplitude: f64,
    /// Every element on the `WetDry` subcells (a dry threshold above the
    /// depth), as elements with a dry node are
    subcells: bool,
}

impl Case {
    fn new(order: usize, formulation: SWEFormulation2D, size: (f64, f64), froude: f64) -> Self {
        Self {
            order,
            formulation,
            size,
            froude,
            distorted: false,
            bed_amplitude: 0.0,
            subcells: false,
        }
    }
}

/// A case on an `n × n` periodic mesh, with its equilibrium state.
struct Linearised {
    physics: SWEPhysics2D<Reflective2D>,
    mesh: Arc<Mesh2D>,
    q0: SWESolution2D,
    /// The time step at CFL 1
    dt1: f64,
}

impl Linearised {
    fn new(case: &Case, n: usize) -> Self {
        let (lx, ly) = (n as f64 * case.size.0, n as f64 * case.size.1);
        let mut mesh = Mesh2D::uniform_periodic(0.0, lx, 0.0, ly, n, n);
        if case.distorted {
            let tau = (2.0 * PI / lx, 2.0 * PI / ly);
            let a = (0.16 * case.size.0, 0.16 * case.size.1);
            for v in &mut mesh.vertices {
                let [x, y] = *v;
                let (sx, sy) = ((tau.0 * x).sin(), (tau.1 * y).sin());
                *v = [
                    x + a.0 * sx * sy + 0.5 * a.0 * (tau.1 * y).sin(),
                    y - 0.7 * a.1 * sx * sy + 0.4 * a.1 * (tau.0 * x).sin(),
                ];
            }
        }
        let ops = Arc::new(DGOperators2D::new(case.order));
        let geom = Arc::new(GeometricFactors2D::compute(&mesh, &ops));
        let mesh = Arc::new(mesh);
        let amplitude = case.bed_amplitude * DEPTH;
        let bed = move |x: f64, y: f64| {
            -DEPTH + amplitude * (2.0 * PI * x / lx).sin() * (2.0 * PI * y / ly).cos()
        };
        let bathymetry = Arc::new(Bathymetry2D::from_function(&mesh, &ops, &geom, bed));
        let physics = PhysicsBuilder::swe_2d(
            mesh.clone(),
            ops.clone(),
            geom.clone(),
            ShallowWater2D::new(G),
            Reflective2D::new(),
        )
        .with_bathymetry(bathymetry.clone())
        .with_formulation(case.formulation)
        .with_limiter(StandardLimiter2D::Positivity(WetDryConfig::DEFAULT_H_DRY))
        .with_wet_dry(WetDryConfig::new(
            Depth::new(if case.subcells {
                2.0 * DEPTH
            } else {
                WetDryConfig::DEFAULT_H_DRY
            }),
            G,
        ))
        .build();

        let speed = case.froude * (G * DEPTH).sqrt() / 2.0_f64.sqrt();
        let mut q0 = SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                let h = -bathymetry.get(k, i);
                q0.set_state(k, i, SWEState2D::new(h, h * speed, h * speed));
            }
        }
        let dt1 = physics.compute_dt(&q0, 1.0);
        Self {
            physics,
            mesh,
            q0,
            dt1,
        }
    }

    fn n_nodes(&self) -> usize {
        self.q0.n_nodes
    }

    /// Column of the Jacobian for variable `var` at node `i` of element `k`
    /// (central differences), indexed `var · (K·n) + k·n + node`.
    fn column(&self, var: usize, k: usize, i: usize) -> Vec<f64> {
        let idx = k * self.n_nodes() + i;
        let eps = 1e-6 * DEPTH;
        let (mut plus, mut minus) = (self.q0.clone(), self.q0.clone());
        plus.data[var][idx] += eps;
        minus.data[var][idx] -= eps;
        let (mut r_plus, mut r_minus) = (self.q0.clone(), self.q0.clone());
        self.physics.compute_rhs_into(&plus, 0.0, &mut r_plus);
        self.physics.compute_rhs_into(&minus, 0.0, &mut r_minus);
        r_plus
            .data
            .iter()
            .flatten()
            .zip(r_minus.data.iter().flatten())
            .map(|(a, b)| (a - b) / (2.0 * eps))
            .collect()
    }

    /// Eigenvalues of the whole Jacobian.
    fn spectrum(&self) -> Vec<C64> {
        let n = self.n_nodes();
        let per_var = self.mesh.n_elements * n;
        let mut jac = Mat::<f64>::zeros(3 * per_var, 3 * per_var);
        for j in 0..3 * per_var {
            let (var, idx) = (j / per_var, j % per_var);
            for (i, value) in self.column(var, idx / n, idx % n).into_iter().enumerate() {
                jac[(i, j)] = value;
            }
        }
        jac.eigenvalues().expect("eigenvalues")
    }

    /// Eigenvalues of the element symbols `A(θ)` for θ = 2π(a, b)/m (uniform
    /// mesh and state; `n ≥ 3`, so that element 0's neighbours are distinct).
    ///
    /// The Jacobian is real, so `A(−θ)` is the conjugate of `A(θ)` and has the
    /// conjugate eigenvalues, which the stability regions treat alike: a runs
    /// over 0..=m/2 only. At rest the equilibrium is also mirror symmetric in
    /// x and y, and b runs over 0..=m/2 too.
    fn bloch_spectrum(&self, case: &Case, m: usize) -> Vec<C64> {
        let n = self.n_nodes();
        let per_var = self.mesh.n_elements * n;
        let size = 3 * n;
        // Offsets of the elements from element 0, wrapped to the nearest
        // periodic image
        let n_side = (self.mesh.n_elements as f64).sqrt().round() as i64;
        let centre = |k: usize| {
            self.mesh
                .reference_to_physical(ElementIndex::new(k), 0.0, 0.0)
        };
        let c0 = centre(0);
        let offsets: Vec<(i64, i64)> = (0..self.mesh.n_elements)
            .map(|k| {
                let c = centre(k);
                let wrap = |d: f64, h: f64| {
                    let p = (d / h).round() as i64;
                    (p + n_side / 2).rem_euclid(n_side) - n_side / 2
                };
                (
                    wrap(c[0] - c0[0], case.size.0),
                    wrap(c[1] - c0[1], case.size.1),
                )
            })
            .collect();
        // J_{e,0} for every element e: columns of element 0's unknowns
        let columns: Vec<Vec<f64>> = (0..size).map(|j| self.column(j / n, 0, j % n)).collect();

        let mut eigenvalues = Vec::new();
        let b_range = if case.froude == 0.0 { m / 2 + 1 } else { m };
        for a in 0..=m / 2 {
            for b in 0..b_range {
                let theta = (
                    2.0 * PI * a as f64 / m as f64,
                    2.0 * PI * b as f64 / m as f64,
                );
                let mut symbol = Mat::<C64>::zeros(size, size);
                for (k, &(px, py)) in offsets.iter().enumerate() {
                    let phase = -(theta.0 * px as f64 + theta.1 * py as f64);
                    let factor = C64::new(phase.cos(), phase.sin());
                    for (j, column) in columns.iter().enumerate() {
                        for i in 0..size {
                            let value = column[(i / n) * per_var + k * n + i % n];
                            if value != 0.0 {
                                symbol[(i, j)] += factor * value;
                            }
                        }
                    }
                }
                eigenvalues.extend(symbol.eigenvalues().expect("eigenvalues"));
            }
        }
        eigenvalues
    }
}

/// Largest CFL with every `CFL·dt1·λ` in `|R| ≤ 1` (bisection).
fn cfl_limit(eigenvalues: &[C64], dt1: f64, r: StabilityPolynomial) -> f64 {
    let stable = |cfl: f64| {
        eigenvalues
            .iter()
            .all(|&lambda| r(lambda * (cfl * dt1)).norm() <= 1.0 + 1e-9)
    };
    let (mut lo, mut hi) = (0.0, 8.0);
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if stable(mid) { lo = mid } else { hi = mid }
    }
    lo
}

/// Bloch wavenumbers per direction in the gate: θ = 2πa/8 includes the
/// modes of 2 × 2, 4 × 4 and 8 × 8 periodic meshes.
const THETAS: usize = 8;

/// The tabulated limits (`linear_cfl_swe_2d`) lie below the measured ones:
/// the relaxed positivity bound (`PositivityBound`) steps up to 0.9 of them.
///
/// Measured (Bloch) at rest on squares, 3:1 and 10:1 rectangles, which set
/// the limits: SSP-RK3's falls with the aspect ratio towards the 1D one
/// (N = 4: 0.625, 0.583, 0.568 at 1:1, 3:1, 10:1), SSP-RK(4,3)'s is set by
/// the squares. A diagonal flow at Froude 0.5, distorted quadrilaterals and
/// a varying bed raise both (`print_linear_cfl_limits`): the time step's
/// per-node metric is conservative there. `WetDry` and `Standard` agree at
/// rest (checked at N = 2).
#[test]
fn linear_cfl_limits_are_inside_the_stability_regions() {
    let cases = [(1.0, 1.0), (3.0, 1.0), (10.0, 1.0)]
        .into_iter()
        .flat_map(|size| (1..=4).map(move |order| (SWEFormulation2D::WetDry, order, size)))
        .chain([(SWEFormulation2D::Standard, 2, (1.0, 1.0))]);
    for (formulation, order, size) in cases {
        let case = Case::new(order, formulation, size, 0.0);
        let linearised = Linearised::new(&case, 3);
        let eigenvalues = linearised.bloch_spectrum(&case, THETAS);
        for (scheme, r) in SCHEMES {
            let measured = cfl_limit(&eigenvalues, linearised.dt1, r);
            let table = linear_cfl_swe_2d(order, scheme).unwrap();
            println!("{case:?}: {scheme:?} {measured:.3} (table {table})");
            assert!(
                table <= measured,
                "{case:?}: {scheme:?} limit {measured:.3} below the table's {table}"
            );
            // Some room above the positivity bound (times the SSP
            // coefficient), which kept every wet/dry run inside
            let positivity = scheme.ssp_coefficient() * positivity_cfl_swe_2d(order);
            assert!(table > 1.4 * positivity, "{scheme:?} N = {order}");
        }
    }
}

/// The Bloch spectrum at the wavenumbers of a periodic mesh is that mesh's
/// spectrum, up to the conjugates the symmetries drop.
#[test]
fn bloch_spectrum_matches_the_whole_jacobian() {
    for (order, froude) in [(1, 0.5), (2, 0.0)] {
        let case = Case::new(order, SWEFormulation2D::WetDry, (2.0, 1.0), froude);
        let linearised = Linearised::new(&case, 4);
        let whole = linearised.spectrum();
        let bloch = linearised.bloch_spectrum(&case, 4);
        for (_, r) in SCHEMES {
            let (a, b) = (
                cfl_limit(&whole, linearised.dt1, r),
                cfl_limit(&bloch, linearised.dt1, r),
            );
            assert!((a - b).abs() < 1e-9 * a, "N = {order}: {a} against {b}");
        }
        // Every whole-mesh eigenvalue is a Bloch one or its conjugate
        let scale = whole.iter().map(|l| l.norm()).fold(0.0, f64::max);
        for l in &whole {
            let nearest = bloch
                .iter()
                .map(|m| (l - m).norm().min((l - m.conj()).norm()))
                .fold(f64::INFINITY, f64::min);
            assert!(
                nearest < 1e-6 * scale,
                "N = {order}: {l} has no Bloch match"
            );
        }
    }
}

/// The limits over whole 2 × 2 to 4 × 4 periodic meshes, including distorted
/// quadrilaterals and a bed (`--ignored`, in release).
///
/// Measured 2026-09-29 (minimum over the meshes, SSP-RK3 / SSP-RK(4,3)):
/// - squares at rest: 1.722 / 2.378, 1.064 / 1.514, 0.785 / 1.156, 0.625 / 0.937
///   for N = 1–4; 3:1: 1.672 / 2.386, 1.058 / 1.599, 0.750 / 1.256, 0.583 /
///   1.043; 10:1: 1.627 / 2.242, 1.048 / 1.602, 0.735 / 1.290, 0.568 / 1.098;
/// - distorted squares (3 × 3, 4 × 4): 2.110 / 2.958, 1.304 / 1.878, 0.939 /
///   1.430, 0.734 / 1.154; with a diagonal flow at Froude 0.5 higher still;
/// - distorted 3:1 over a bed (±60 % of the depth): 2.030 / 3.023, 1.120 /
///   2.009, 0.782 / 1.583, 0.603 / 1.236.
#[test]
#[ignore = "measurement: slow in debug builds"]
fn print_linear_cfl_limits() {
    let cases = [
        ((1.0, 1.0), 0.0, false, 0.0),
        ((1.0, 1.0), 0.5, false, 0.0),
        ((3.0, 1.0), 0.0, false, 0.0),
        ((3.0, 1.0), 0.5, false, 0.0),
        ((10.0, 1.0), 0.0, false, 0.0),
        ((1.0, 1.0), 0.0, true, 0.0),
        ((1.0, 1.0), 0.5, true, 0.0),
        ((3.0, 1.0), 0.0, true, 0.6),
    ];
    for (size, froude, distorted, bed_amplitude) in cases {
        for order in 1..=4 {
            let case = Case {
                distorted,
                bed_amplitude,
                ..Case::new(order, SWEFormulation2D::WetDry, size, froude)
            };
            let limits: Vec<_> = [2, 3, 4]
                .map(|n| {
                    let linearised = Linearised::new(&case, n);
                    let eigenvalues = linearised.spectrum();
                    SCHEMES.map(|(_, r)| cfl_limit(&eigenvalues, linearised.dt1, r))
                })
                .to_vec();
            println!(
                "N={order} size {size:?} Fr {froude} distorted {distorted} bed {bed_amplitude}: {limits:.3?}"
            );
        }
    }
}

/// Whether a random perturbation of the equilibrium (1e-3 of the depth)
/// stays bounded (below 100× its size) over `steps` steps of `scheme` at
/// `cfl`, without post-processing. The subcell reconstruction's limiter is
/// positively homogeneous, so the outcome does not depend on the amplitude.
fn perturbation_stays_bounded(
    linearised: &Linearised,
    scheme: SspScheme,
    cfl: f64,
    steps: usize,
) -> bool {
    let mut state = 7_u64;
    let mut rand = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) as f64 / u64::MAX as f64) - 0.5
    };
    let q0 = &linearised.q0;
    let mut q = q0.clone();
    for var in 0..3 {
        for v in &mut q.data[var] {
            *v += 1e-3 * DEPTH * rand();
        }
    }
    let size = |q: &SWESolution2D| {
        q.data
            .iter()
            .flatten()
            .zip(q0.data.iter().flatten())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max)
    };
    let before = size(&q);
    let dt = cfl * linearised.dt1;
    let physics = &linearised.physics;
    let mut rhs = q.clone();
    // q ← a·u0 + b·(q + c·dt·L(q))
    let mut stage = |q: &mut SWESolution2D, u0: &SWESolution2D, a: f64, b: f64, c: f64| {
        physics.compute_rhs_into(q, 0.0, &mut rhs);
        for ((x, r), y) in q
            .data
            .iter_mut()
            .flatten()
            .zip(rhs.data.iter().flatten())
            .zip(u0.data.iter().flatten())
        {
            *x = a * y + b * (*x + c * dt * r);
        }
    };
    for _ in 0..steps {
        let u0 = q.clone();
        match scheme {
            SspScheme::Rk3 => {
                stage(&mut q, &u0, 0.0, 1.0, 1.0);
                stage(&mut q, &u0, 0.75, 0.25, 1.0);
                stage(&mut q, &u0, 1.0 / 3.0, 2.0 / 3.0, 1.0);
            }
            SspScheme::Rk43 => {
                stage(&mut q, &u0, 0.0, 1.0, 0.5);
                stage(&mut q, &u0, 0.0, 1.0, 0.5);
                stage(&mut q, &u0, 2.0 / 3.0, 1.0 / 3.0, 0.5);
                stage(&mut q, &u0, 0.0, 1.0, 0.5);
            }
        }
        let grown = size(&q);
        if grown.is_nan() || grown >= 100.0 * before {
            return false;
        }
    }
    true
}

/// The subcells' limit, taken as the smaller of two measurements with every
/// element on the subcells: the linear limit (the Bloch spectrum at rest:
/// a one-node perturbation of a uniform state leaves every limited slope
/// zero, so this is the first-order scheme), and the largest CFL at which
/// the limited second-order reconstruction keeps a random perturbation
/// bounded.
fn subcell_case(order: usize, size: (f64, f64), froude: f64) -> Case {
    Case {
        subcells: true,
        ..Case::new(order, SWEFormulation2D::WetDry, size, froude)
    }
}

/// The tabulated subcell limits (`linear_cfl_subcells_swe_2d`) lie below
/// the linear limits of the first-order subcells on squares, 3:1 and 10:1
/// rectangles, and below the DGSEM ones: `PositivityBound` caps elements
/// with a dry node at 0.9 of them.
#[test]
fn subcell_limits_are_inside_the_stability_regions() {
    for size in [(1.0, 1.0), (3.0, 1.0), (10.0, 1.0)] {
        for order in 1..=4 {
            let case = subcell_case(order, size, 0.0);
            let linearised = Linearised::new(&case, 3);
            let eigenvalues = linearised.bloch_spectrum(&case, THETAS);
            for (scheme, r) in SCHEMES {
                let measured = cfl_limit(&eigenvalues, linearised.dt1, r);
                let table = linear_cfl_subcells_swe_2d(order, scheme).unwrap();
                println!("{case:?}: {scheme:?} {measured:.3} (table {table})");
                assert!(
                    table <= measured,
                    "{case:?}: {scheme:?} limit {measured:.3} below the table's {table}"
                );
                assert!(table < linear_cfl_swe_2d(order, scheme).unwrap());
                // At least the plain positivity bound (times the SSP
                // coefficient), which every wet/dry run has stepped at
                let positivity = scheme.ssp_coefficient() * positivity_cfl_swe_2d(order);
                assert!(table >= positivity, "{scheme:?} N = {order}");
            }
        }
    }
}

/// With the limited reconstruction a random perturbation stays bounded at
/// the tabulated limit, and grows at the first-order linear limit: at N = 2
/// under SSP-RK(4,3) on elongated elements the reconstruction, not the
/// first-order scheme, sets the limit (1.13 against 1.30 on 10:1).
#[test]
fn subcell_reconstruction_is_stable_at_the_tabulated_limit() {
    let case = subcell_case(2, (10.0, 1.0), 0.0);
    let perturbed = Linearised::new(&case, 6);
    for (scheme, _) in SCHEMES {
        let table = linear_cfl_subcells_swe_2d(2, scheme).unwrap();
        assert!(
            perturbation_stays_bounded(&perturbed, scheme, table, 1000),
            "{scheme:?} at {table}"
        );
    }
    assert!(!perturbation_stays_bounded(
        &perturbed,
        SspScheme::Rk43,
        1.25,
        1000
    ));
}

/// Both measurements over square to 30:1 rectangles, at rest and with a
/// diagonal flow at Froude 0.5 (`--ignored`, in release).
///
/// Measured 2026-09-29 (linear / nonlinear, SSP-RK3 then SSP-RK(4,3)), for
/// N = 1–4:
/// - squares: 0.942 / 1.005, 1.764 / 1.761; 0.685 / 0.670, 1.267 / 1.244;
///   0.520 / 0.519, 0.994 / 0.993; 0.407 / 0.407, 0.787 / 0.789;
/// - 10:1: 0.942 / 1.000, 1.555 / 1.554; 0.660 / 0.662, 1.296 / 1.134; 0.478 /
///   0.485, 0.980 / 0.990; 0.370 / 0.373, 0.759 / 0.766;
/// - 30:1: 0.942 / 0.996, 1.525 / 1.520; 0.658 / 0.659, 1.282 / 1.111; 0.476 /
///   0.480, 0.976 / 0.985; 0.369 / 0.373, 0.756 / 0.764;
/// - 3:1 in between, and a flow raises every limit (squares at Froude 0.5:
///   1.02–1.78 at N = 1 to 0.51–1.05 at N = 4).
#[test]
#[ignore = "measurement"]
fn print_subcell_limits() {
    let cases = [
        ((1.0, 1.0), 0.0),
        ((3.0, 1.0), 0.0),
        ((10.0, 1.0), 0.0),
        ((30.0, 1.0), 0.0),
        ((1.0, 1.0), 0.5),
        ((10.0, 1.0), 0.5),
    ];
    for (size, froude) in cases {
        for order in 1..=4 {
            let case = subcell_case(order, size, froude);
            let linearised = Linearised::new(&case, 3);
            let eigenvalues = linearised.bloch_spectrum(&case, THETAS);
            let linear = SCHEMES.map(|(_, r)| cfl_limit(&eigenvalues, linearised.dt1, r));
            let perturbed = Linearised::new(&case, 6);
            let nonlinear = SCHEMES.map(|(scheme, _)| {
                let (mut lo, mut hi) = (0.1, 4.0);
                for _ in 0..12 {
                    let mid = 0.5 * (lo + hi);
                    if perturbation_stays_bounded(&perturbed, scheme, mid, 1000) {
                        lo = mid
                    } else {
                        hi = mid
                    }
                }
                lo
            });
            println!(
                "N={order} size {size:?} Fr {froude}: linear {linear:.3?}, nonlinear {nonlinear:.3?}"
            );
        }
    }
}

//! Gate tests for the characteristic open boundary (`CharacteristicOBC`).
//!
//! The boundary state takes the outgoing Riemann invariant from the interior
//! and the incoming one from the external data, and the boundary flux is its
//! physical flux, so every Riemann solver and every spatial formulation must
//! give the same boundary behaviour. For the linearised problem the condition
//! is Flather's `u_n = u_n,ext + sqrt(g/H)(η − η_ext)`.
//!
//! The history: the Flather-type BCs used to build the ghost normal velocity
//! as `u_n,ext + sqrt(g/H)(η_int − η_ext)` and let the Riemann solver apply
//! the relation again, reflecting outgoing waves with coefficient −1/3
//! (P0.12). The ghost = external state fix was exact only for Roe.
//!
//! Checked here, for Roe, HLL, Rusanov and the split forms:
//!
//! 1. an outgoing pulse leaves with < 1 % reflection, and nothing grows after;
//! 2. a progressive wave forced through the boundary is delivered at the
//!    prescribed amplitude, from full external data, from elevation-only data
//!    as an incoming wave, and from a parent time series; elevation-only data
//!    at rest delivers half;
//! 3. a plane wave at 45° to the boundary normal reflects with the linear
//!    coefficient (1 − cos θ)/(1 + cos θ) ≈ 0.17 of a 1D characteristic OBC;
//! 4. the boundary flux does not depend on the Riemann solver;
//! 5. lake at rest over varying bathymetry is exact with open boundaries.

use std::f64::consts::PI;

use dgcom::boundary::{
    CharacteristicOBC, ElevationOnly, ExternalState, MultiBoundaryCondition2D, ParentTimeSeries,
    Reflective2D, SWEBoundaryCondition2D, StillWater,
};
use dgcom::flux::SWEFluxType2D;
use dgcom::io::{BoundaryTimeSeries, TimeSeriesRecord};
use dgcom::mesh::{Bathymetry2D, BoundaryTag};
use dgcom::time::{SSPRK3, TimeIntegrator};
use dgcom::types::ElementIndex;
use dgcom::{
    DGOperators2D, GeometricFactors2D, Mesh2D, SWE2DRhsConfig, SWEFormulation2D, SWESolution2D,
    SWEState2D, ShallowWater2D, compute_dt_swe_2d, compute_rhs_swe_2d,
};

const G: f64 = 9.81;
/// Still-water depth (m); bed elevation is B = −H0.
const H0: f64 = 10.0;
/// Channel length (m).
const LENGTH: f64 = 500.0;
/// Number of elements along the channel.
const NX: usize = 25;
/// Element size (m); the channel is one square element wide.
const DX: f64 = LENGTH / NX as f64;
/// Polynomial order.
const ORDER: usize = 3;
const CFL: f64 = 0.5;

fn celerity() -> f64 {
    (G * H0).sqrt()
}

/// A spatial discretization: collocated DG with a Riemann solver, or a split form.
#[derive(Clone, Copy, Debug)]
enum Scheme {
    Flux(SWEFluxType2D),
    Split(SWEFormulation2D),
}

const SCHEMES: [Scheme; 5] = [
    Scheme::Flux(SWEFluxType2D::Roe),
    Scheme::Flux(SWEFluxType2D::HLL),
    Scheme::Flux(SWEFluxType2D::Rusanov),
    Scheme::Split(SWEFormulation2D::EntropyStable),
    Scheme::Split(SWEFormulation2D::WetDry),
];

/// A mesh with flat bed B = −H0 and the solver pieces.
struct Domain {
    mesh: Mesh2D,
    ops: DGOperators2D,
    geom: GeometricFactors2D,
    equation: ShallowWater2D,
    bathymetry: Bathymetry2D,
}

impl Domain {
    fn new(mesh: Mesh2D) -> Self {
        let ops = DGOperators2D::new(ORDER);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let bathymetry = Bathymetry2D::constant(mesh.n_elements, ops.n_nodes, -H0);
        Self {
            mesh,
            ops,
            geom,
            equation: ShallowWater2D::new(G),
            bathymetry,
        }
    }

    /// Quasi-1D channel `[0, LENGTH] × [0, DX]` with walls on the long sides
    /// and the given tags on the west (x = 0) and east (x = LENGTH) ends.
    fn channel(west: BoundaryTag, east: BoundaryTag) -> Self {
        // Side order: [south, east, north, west]
        Self::new(Mesh2D::uniform_rectangle_with_sides(
            0.0,
            LENGTH,
            0.0,
            DX,
            NX,
            1,
            [BoundaryTag::Wall, east, BoundaryTag::Wall, west],
        ))
    }

    fn nodes(&self) -> impl Iterator<Item = (ElementIndex, usize)> + '_ {
        ElementIndex::iter(self.mesh.n_elements)
            .flat_map(|k| (0..self.ops.n_nodes).map(move |i| (k, i)))
    }

    fn xy(&self, k: ElementIndex, i: usize) -> [f64; 2] {
        let (r, s) = (self.ops.nodes_r[i], self.ops.nodes_s[i]);
        self.mesh.reference_to_physical(k, r, s)
    }

    /// State with surface elevation η and velocity (u, v) given per point.
    fn state(&self, f: impl Fn(f64, f64) -> (f64, f64, f64)) -> SWESolution2D {
        let mut q = SWESolution2D::new(self.mesh.n_elements, self.ops.n_nodes);
        for (k, i) in self.nodes() {
            let [x, y] = self.xy(k, i);
            let (eta, u, v) = f(x, y);
            q.set_state(k, i, SWEState2D::from_primitives(H0 + eta, u, v));
        }
        q
    }

    fn config<'a, BC: SWEBoundaryCondition2D>(
        &'a self,
        bc: &'a BC,
        scheme: Scheme,
    ) -> SWE2DRhsConfig<'a, BC> {
        let config = SWE2DRhsConfig::new(&self.equation, bc)
            .with_coriolis(false)
            .with_bathymetry(&self.bathymetry);
        match scheme {
            Scheme::Flux(flux) => config.with_flux_type(flux),
            Scheme::Split(formulation) => config.with_formulation(formulation),
        }
    }

    /// Advance `q` from t = 0 to `t_end` with SSP-RK3, calling `observe`
    /// after every step.
    fn run<BC: SWEBoundaryCondition2D>(
        &self,
        bc: &BC,
        scheme: Scheme,
        q: &mut SWESolution2D,
        t_end: f64,
        mut observe: impl FnMut(&SWESolution2D, f64),
    ) {
        let config = self.config(bc, scheme);
        // The linear-regime wave speed is ~constant, so a single dt suffices.
        let dt_cfl = compute_dt_swe_2d(q, &self.mesh, &self.geom, &self.equation, ORDER, CFL);
        let n_steps = (t_end / dt_cfl).ceil() as usize;
        let dt = t_end / n_steps as f64;

        let mut t = 0.0;
        for _ in 0..n_steps {
            SSPRK3.step(q, dt, t, |s, stage_t| {
                compute_rhs_swe_2d(s, &self.mesh, &self.ops, &self.geom, &config, stage_t)
            });
            t += dt;
            observe(q, t);
        }
    }

    /// Largest |η| over the nodes (NaN counts as unbounded).
    fn max_abs_eta(&self, q: &SWESolution2D) -> f64 {
        self.nodes()
            .map(|(k, i)| (q.get_state(k, i).h - H0).abs())
            .fold(
                0.0,
                |m, e| if e.is_nan() { f64::INFINITY } else { m.max(e) },
            )
    }
}

fn with_walls(open: &dyn SWEBoundaryCondition2D) -> MultiBoundaryCondition2D<'_> {
    static WALL: Reflective2D = Reflective2D { h_min: 1e-6 };
    MultiBoundaryCondition2D::new(&WALL).with_open(open)
}

// ---------------------------------------------------------------------------
// Outgoing pulse: reflection
// ---------------------------------------------------------------------------

/// Pulse amplitude (m). a/H = 1e-3 keeps the problem in the linear regime.
const PULSE_AMP: f64 = 0.01;
const PULSE_X0: f64 = 250.0;
const PULSE_WIDTH: f64 = 30.0;
/// By this time the pulse (and its tail) has left through the east end.
const PULSE_EXIT_TIME: f64 = 50.0;
/// Three channel crossings after the pulse has left.
const QUIET_T_END: f64 = 200.0;
/// Maximum allowed reflected amplitude (fraction of the incident pulse).
const MAX_REFLECTION: f64 = 0.01;

/// Launch a right-going pulse from mid-channel (both ends open, still water
/// outside) and return the largest |η| left in the channel, relative to the
/// pulse, at `PULSE_EXIT_TIME` and over `[PULSE_EXIT_TIME, QUIET_T_END]`.
fn outgoing_pulse(bc: &dyn SWEBoundaryCondition2D, scheme: Scheme) -> (f64, f64) {
    let ch = Domain::channel(BoundaryTag::Open, BoundaryTag::Open);
    // Exact right-going simple wave: the left-going invariant
    // u − 2 sqrt(g h) is uniform, so no left-going signal is launched.
    let eta0 = |x: f64| PULSE_AMP * (-((x - PULSE_X0) / PULSE_WIDTH).powi(2)).exp();
    let mut q = ch.state(|x, _| {
        let eta = eta0(x);
        (eta, 2.0 * ((G * (H0 + eta)).sqrt() - celerity()), 0.0)
    });

    let bc = with_walls(bc);
    let (mut at_exit, mut after) = (f64::NAN, 0.0_f64);
    ch.run(&bc, scheme, &mut q, QUIET_T_END, |q, t| {
        if t >= PULSE_EXIT_TIME {
            let eta = ch.max_abs_eta(q);
            if at_exit.is_nan() {
                at_exit = eta;
            }
            after = after.max(eta);
        }
    });
    (at_exit / PULSE_AMP, after / PULSE_AMP)
}

#[test]
fn outgoing_pulse_leaves_for_every_scheme() {
    let bc = CharacteristicOBC::still_water();
    for scheme in SCHEMES {
        let (reflection, after) = outgoing_pulse(&bc, scheme);
        println!(
            "{scheme:?}: reflected {:.4} %, max |η| after exit {:.2e} × pulse",
            100.0 * reflection,
            after
        );
        assert!(
            reflection < MAX_REFLECTION && after < MAX_REFLECTION,
            "{scheme:?}: reflected {reflection:.3e}, max after exit {after:.3e} × pulse \
             (limit {MAX_REFLECTION})"
        );
    }
}

// ---------------------------------------------------------------------------
// Forced progressive wave: delivered amplitude
// ---------------------------------------------------------------------------

/// Prescribed wave amplitude (m).
const WAVE_AMP: f64 = 0.01;
/// Wavelength (m): 6.25 elements (25 nodes) per wavelength at order 3.
const WAVELENGTH: f64 = 125.0;

fn wave_period() -> f64 {
    WAVELENGTH / celerity()
}

/// Eastward progressive wave at the west end (x = 0): η = A sin(ωt) for
/// t ≥ 0, at rest before; returns (η, u).
fn eastward_wave_at_west(t: f64) -> (f64, f64) {
    let omega = 2.0 * PI / wave_period();
    let eta = if t > 0.0 {
        WAVE_AMP * (omega * t).sin()
    } else {
        0.0
    };
    (eta, (G / H0).sqrt() * eta)
}

/// Force an eastward progressive wave through the west end (tag
/// `TidalForcing`, boundary condition `forcing`) of a channel whose east end
/// radiates to still water, and return the (min, max) over the channel
/// interior of the wave amplitude relative to the prescribed one, over the
/// last period. Reflections at either end would superpose a standing
/// component whose envelope varies along the channel.
fn delivered_amplitude(forcing: &dyn SWEBoundaryCondition2D, scheme: Scheme) -> (f64, f64) {
    let ch = Domain::channel(BoundaryTag::TidalForcing, BoundaryTag::Open);
    let mut q = ch.state(|_, _| (0.0, 0.0, 0.0));

    let period = wave_period();
    let t_end = 2.0 * LENGTH / celerity() + 3.0 * period;
    let t_measure = t_end - period;

    // Interior away from the boundaries (50 m ≤ x ≤ 450 m)
    let samples: Vec<_> = ch
        .nodes()
        .filter(|&(k, i)| (50.0..=450.0).contains(&ch.xy(k, i)[0]))
        .collect();
    let mut amplitude = vec![0.0_f64; samples.len()];

    static WALL: Reflective2D = Reflective2D { h_min: 1e-6 };
    let radiating = CharacteristicOBC::still_water();
    let bc = MultiBoundaryCondition2D::new(&WALL)
        .with_tidal(forcing)
        .with_open(&radiating);
    ch.run(&bc, scheme, &mut q, t_end, |q, t| {
        if t >= t_measure {
            for (a, &(k, i)) in amplitude.iter_mut().zip(&samples) {
                *a = a.max((q.get_state(k, i).h - H0).abs());
            }
        }
    });

    let min = amplitude.iter().copied().fold(f64::INFINITY, f64::min);
    let max = amplitude.iter().copied().fold(0.0, f64::max);
    (min / WAVE_AMP, max / WAVE_AMP)
}

fn assert_amplitude(name: &str, scheme: Scheme, (min, max): (f64, f64), expected: f64) {
    println!("{name}, {scheme:?}: delivered amplitude in [{min:.3}, {max:.3}] × prescribed");
    assert!(
        min >= expected - 0.05 && max <= expected + 0.05,
        "{name}, {scheme:?}: delivered amplitude ranges over [{min:.3}, {max:.3}] of the \
         prescribed amplitude (expected {expected} ± 0.05)"
    );
}

#[test]
fn full_external_state_delivers_the_wave_for_every_scheme() {
    // u_n along the outward normal, which points in −x at the west end
    let forcing = CharacteristicOBC::new(|_, _, t| {
        let (eta, u) = eastward_wave_at_west(t);
        ExternalState::new(eta, u, 0.0)
    });
    for scheme in SCHEMES {
        assert_amplitude("(η, u)", scheme, delivered_amplitude(&forcing, scheme), 1.0);
    }
}

#[test]
fn elevation_only_forcing_as_incoming_wave_delivers_the_wave() {
    let forcing =
        CharacteristicOBC::new(|_, _, t| ExternalState::elevation(eastward_wave_at_west(t).0))
            .with_incoming_wave();
    for scheme in [SCHEMES[0], SCHEMES[2], SCHEMES[4]] {
        assert_amplitude(
            "η, incoming",
            scheme,
            delivered_amplitude(&forcing, scheme),
            1.0,
        );
    }
}

/// With the external velocity taken at rest, a progressive wave arrives at
/// half amplitude (the boundary is then an antinode of the external state).
#[test]
fn elevation_only_forcing_at_rest_delivers_half() {
    let forcing =
        CharacteristicOBC::new(|_, _, t| ExternalState::elevation(eastward_wave_at_west(t).0))
            .with_elevation_only(ElevationOnly::AtRest);
    assert_amplitude(
        "η, at rest",
        SCHEMES[0],
        delivered_amplitude(&forcing, SCHEMES[0]),
        0.5,
    );
}

#[test]
fn parent_time_series_delivers_the_wave() {
    let t_end = 2.0 * LENGTH / celerity() + 4.0 * wave_period();
    let records = (0..=(t_end / 0.1).ceil() as usize)
        .map(|n| {
            let t = n as f64 * 0.1;
            let (eta, u) = eastward_wave_at_west(t);
            TimeSeriesRecord::from_primitives(t, H0 + eta, u, 0.0)
        })
        .collect();
    let parent = ParentTimeSeries::new(BoundaryTimeSeries::from_records(records).unwrap(), H0);
    let forcing = CharacteristicOBC::new(parent);
    assert_amplitude(
        "parent series",
        SCHEMES[1],
        delivered_amplitude(&forcing, SCHEMES[1]),
        1.0,
    );
}

// ---------------------------------------------------------------------------
// Oblique incidence
// ---------------------------------------------------------------------------

/// Reflection coefficient of a plane wave at angle `theta` from the normal
/// of the north boundary (still water outside), measured from the standing
/// pattern of the steady state.
///
/// The domain is periodic in x with one x-wavelength across and one
/// y-wavelength high. The south boundary forces the incident plane wave
/// (full external state); whatever it does to the returning wave, the ratio
/// of the southward to the northward wave in the interior is the north
/// boundary's reflection coefficient |R| = (E_max − E_min)/(E_max + E_min),
/// with E(y) the local amplitude.
fn oblique_reflection(theta: f64) -> f64 {
    let lx = 200.0;
    let kx = 2.0 * PI / lx;
    let k = kx / theta.sin();
    let ky = k * theta.cos();
    let ly = 2.0 * PI / ky;
    let omega = celerity() * k;
    let (n_elem_x, n_elem_y) = (10, (10.0 * ly / lx).round() as usize);

    let mut mesh = Mesh2D::channel_periodic_x(0.0, lx, 0.0, ly, n_elem_x, n_elem_y);
    for edge in mesh.edges.iter_mut().filter(|e| e.is_boundary()) {
        let y = mesh.vertices[edge.vertices.0][1];
        edge.boundary_tag = Some(if y < 0.5 * ly {
            BoundaryTag::TidalForcing
        } else {
            BoundaryTag::Open
        });
    }
    let domain = Domain::new(mesh);

    // Incident wave A cos(kx x + ky y − ωt), ramped in over two periods
    let period = 2.0 * PI / omega;
    let incident = move |x: f64, y: f64, t: f64| {
        let ramp = (t / (2.0 * period)).clamp(0.0, 1.0);
        let eta = ramp * WAVE_AMP * (kx * x + ky * y - omega * t).cos();
        let speed = (G / H0).sqrt() * eta;
        ExternalState::new(eta, speed * theta.sin(), speed * theta.cos())
    };
    let forcing = CharacteristicOBC::new(incident);
    let open = CharacteristicOBC::still_water();
    static WALL: Reflective2D = Reflective2D { h_min: 1e-6 };
    let bc = MultiBoundaryCondition2D::new(&WALL)
        .with_tidal(&forcing)
        .with_open(&open);

    let t_end = 12.0 * period;
    let t_measure = t_end - period;
    let mut envelope = vec![0.0_f64; domain.mesh.n_elements * domain.ops.n_nodes];
    let mut q = domain.state(|_, _| (0.0, 0.0, 0.0));
    let n_nodes = domain.ops.n_nodes;
    domain.run(&bc, SCHEMES[0], &mut q, t_end, |q, t| {
        if t >= t_measure {
            for (k, i) in domain.nodes() {
                let e = &mut envelope[k.as_usize() * n_nodes + i];
                *e = e.max((q.get_state(k, i).h - H0).abs());
            }
        }
    });
    let e_max = envelope.iter().copied().fold(0.0, f64::max);
    let e_min = envelope.iter().copied().fold(f64::INFINITY, f64::min);
    (e_max - e_min) / (e_max + e_min)
}

#[test]
fn oblique_plane_wave_reflects_as_a_1d_characteristic_condition() {
    let theta = PI / 4.0;
    let expected = (1.0 - theta.cos()) / (1.0 + theta.cos());
    let measured = oblique_reflection(theta);
    println!("45°: |R| = {measured:.4} (theory {expected:.4})");
    assert!(
        (measured - expected).abs() < 0.02,
        "reflection at 45°: {measured:.4}, theory {expected:.4}"
    );
}

// ---------------------------------------------------------------------------
// Flux independence and lake at rest
// ---------------------------------------------------------------------------

/// One element, all faces open: the RHS consists of the volume term and the
/// boundary flux alone, and must be the same for every Riemann solver.
#[test]
fn boundary_flux_does_not_depend_on_the_riemann_solver() {
    let domain = Domain::new(Mesh2D::uniform_rectangle_with_bc(
        0.0,
        100.0,
        0.0,
        80.0,
        1,
        1,
        BoundaryTag::Open,
    ));
    let q = domain.state(|x, y| {
        (
            0.3 * (x / 40.0).sin() * (y / 30.0).cos(),
            0.4 + 0.2 * (y / 25.0).sin(),
            -0.3 + 0.1 * (x / 35.0).cos(),
        )
    });
    let obc = CharacteristicOBC::new(|x: f64, y: f64, t: f64| {
        ExternalState::new(0.2 * (x / 50.0 + t).cos(), 0.1 * (y / 20.0).sin(), -0.2)
    });

    let rhs = |flux| {
        let config = domain.config(&obc, Scheme::Flux(flux));
        compute_rhs_swe_2d(&q, &domain.mesh, &domain.ops, &domain.geom, &config, 0.7)
    };
    let roe = rhs(SWEFluxType2D::Roe);
    for flux in [SWEFluxType2D::HLL, SWEFluxType2D::Rusanov] {
        let other = rhs(flux);
        let diff = (0..3)
            .flat_map(|v| roe.data[v].iter().zip(&other.data[v]))
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(
            diff < 1e-11 * roe.max_abs(),
            "{flux:?} differs from Roe by {diff:.3e} (|rhs| {:.3e})",
            roe.max_abs()
        );
    }
}

/// Lake at rest over a rough bed with every side open to still water at the
/// same level: the RHS vanishes for every scheme.
#[test]
fn lake_at_rest_is_exact_with_open_boundaries() {
    let mut domain = Domain::new(Mesh2D::uniform_rectangle_with_bc(
        0.0,
        400.0,
        0.0,
        300.0,
        4,
        3,
        BoundaryTag::Open,
    ));
    let (mesh, ops, geom) = (&domain.mesh, &domain.ops, &domain.geom);
    let mut bathymetry = Bathymetry2D::flat(mesh.n_elements, ops.n_nodes);
    for k in ElementIndex::iter(mesh.n_elements) {
        for i in 0..ops.n_nodes {
            let [x, y] = mesh.reference_to_physical(k, ops.nodes_r[i], ops.nodes_s[i]);
            // Smooth relief plus a step per element (face-discontinuous bed)
            let step = 0.4 * (k.as_usize() % 3) as f64;
            let b = -H0 + 3.0 * (x / 60.0).sin() * (y / 45.0).cos() + step;
            bathymetry.set(k, i, b);
        }
    }
    bathymetry.compute_gradients(ops, geom);
    domain.bathymetry = bathymetry;

    let eta0 = 0.25;
    let mut q = SWESolution2D::new(domain.mesh.n_elements, domain.ops.n_nodes);
    for (k, i) in domain.nodes() {
        let h = eta0 - domain.bathymetry.get(k, i);
        q.set_state(k, i, SWEState2D::new(h, 0.0, 0.0));
    }
    let obc = CharacteristicOBC::new(StillWater::at(eta0));
    for scheme in [
        Scheme::Split(SWEFormulation2D::EntropyStable),
        Scheme::Split(SWEFormulation2D::WetDry),
    ] {
        let config = domain.config(&obc, scheme);
        let rhs = compute_rhs_swe_2d(&q, &domain.mesh, &domain.ops, &domain.geom, &config, 0.0);
        println!("{scheme:?}: max |rhs| = {:.2e}", rhs.max_abs());
        assert!(rhs.max_abs() < 1e-10, "{scheme:?}: {:.3e}", rhs.max_abs());
    }
}

/// A uniform atmospheric pressure gradient over a basin open on all sides:
/// the sea at its inverse-barometer level is at rest (the pressure force
/// balances the surface slope) only if the open boundaries carry that level
/// too; still water at mean sea level outside drives flow through them.
#[test]
fn inverse_barometer_at_open_boundaries_keeps_the_sea_at_rest() {
    use dgcom::boundary::InverseBarometer;
    use dgcom::source::AtmosphericPressure2D;

    let domain = Domain::new(Mesh2D::uniform_rectangle_with_bc(
        0.0,
        400.0,
        0.0,
        300.0,
        4,
        3,
        BoundaryTag::Open,
    ));
    let pressure = AtmosphericPressure2D::uniform_gradient(2e-2, -1e-2);
    let q = domain.state(|x, y| (pressure.inverse_barometer(x, y, 0.0), 0.0, 0.0));

    let rhs_with = |bc: &dyn SWEBoundaryCondition2D| {
        let bc = with_walls(bc);
        let config = domain
            .config(&bc, Scheme::Split(SWEFormulation2D::EntropyStable))
            .with_source_terms(&pressure);
        compute_rhs_swe_2d(&q, &domain.mesh, &domain.ops, &domain.geom, &config, 0.0).max_abs()
    };
    let with_ib = rhs_with(&CharacteristicOBC::new(InverseBarometer::new(
        StillWater::default(),
        |x, y, t| pressure.inverse_barometer(x, y, t),
    )));
    let without = rhs_with(&CharacteristicOBC::still_water());
    println!("max |rhs|: {with_ib:.2e} with the IB level outside, {without:.2e} without");
    assert!(with_ib < 1e-12, "{with_ib:.3e}");
    assert!(without > 1e-6, "{without:.3e}");
}

//! A deep fjord behind a sill, opening into a wider basin, with a river at
//! its head: the experiment of `examples/sill_fjord_3d.rs` and of its gate
//! in `tests/sill_fjord_test.rs` (`#[path]`-included by both). See the
//! example's docs for the theory it is compared with.

// Not every user reads every field
#![allow(dead_code)]

use std::sync::Arc;

use dg_rs::boundary::{
    CharacteristicOBC, MultiBoundaryCondition2D, Nesting3D, NestingBand3D, ReferenceColumns,
    Reflective2D, StillWater,
};
use dg_rs::equations::ShallowWater2D;
use dg_rs::mesh::{Bathymetry2D, BoundaryTag, Mesh2D};
use dg_rs::operators::{DGOperators2D, GeometricFactors2D};
use dg_rs::physics::{
    BottomDrag3D, Forcing, GlsMixing, Hydrostatic3D, ImplicitVerticalAdvection, LinearEOS,
    PhysicsBuilder,
};
use dg_rs::solver::state::Solution3D;
use dg_rs::solver::{TracerLimiter3DConfig, TracerLimiterType3D};
use dg_rs::source::{CoriolisSource2D, River, RiverProfile, RiverSeries, RiverSources};
use dg_rs::time::{ModeSplitIntegrator, SspScheme};
use dg_rs::types::ElementIndex;
use dg_rs::vertical::{SigmaGrid, SongHaidvogelStretching};

pub const G: f64 = 9.81;
pub const RHO0: f64 = 1025.0;
/// The sea's salinity, and the fjord's at the start
pub const S_SEA: f64 = 33.0;

/// The experiment's parameters (lengths in m).
#[derive(Clone, Debug)]
pub struct Config {
    /// River discharge (m³/s)
    pub qf: f64,
    /// Fjord width B
    pub width: f64,
    pub length: f64,
    pub depth: f64,
    /// Depth over the sill's crest, 500 m inside the mouth
    pub sill: f64,
    /// Side of the square outer basin
    pub basin: f64,
    /// Element size
    pub dx: f64,
    pub order: usize,
    pub levels: usize,
    /// Song–Haidvogel surface stretching and critical depth (θ_b 0.4)
    pub theta_s: f64,
    pub hc: f64,
    pub hours: f64,
    /// The last hours, over which the sections are averaged
    pub average: f64,
    /// Coriolis parameter (1/s)
    pub f: f64,
    /// Courant number of the SSP-RK(4,3) 3D stages, and the longest step (s)
    pub cfl: f64,
    pub dt_max: f64,
    /// Width of the band relaxing the tracers at the open sides
    pub band: f64,
    /// The brackish layer ends at the first level at least `s₂ − delta`
    pub delta: f64,
    /// Hours between progress reports (0: none)
    pub report: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            qf: 500.0,
            width: 500.0,
            length: 16e3,
            depth: 100.0,
            sill: 30.0,
            basin: 6e3,
            dx: 250.0,
            order: 2,
            levels: 30,
            theta_s: 6.0,
            hc: 5.0,
            hours: 48.0,
            average: 12.0,
            f: 1.2e-4,
            cfl: 4.0,
            dt_max: 120.0,
            band: 1500.0,
            delta: 1.0,
            report: 2.0,
        }
    }
}

impl Config {
    /// The position of the sill's crest (x, m).
    pub fn crest(&self) -> f64 {
        self.length - 500.0
    }

    /// The bed elevation at `x` (m, negative): a Gaussian ridge across the
    /// mouth, `sill` m deep at the crest.
    pub fn bed(&self, x: f64) -> f64 {
        -self.depth + (self.depth - self.sill) * (-((x - self.crest()) / 600.0).powi(2)).exp()
    }

    pub fn sigma(&self) -> SigmaGrid {
        SigmaGrid::new(
            self.levels,
            SongHaidvogelStretching::new(self.theta_s, 0.4, self.hc),
        )
    }

    /// `βg` of the linear equation of state (m/s² per psu).
    pub fn beta_g(&self) -> f64 {
        LinearEOS::default().beta * G
    }

    /// Stigebrandt's speed of the brackish layer at the mouth,
    /// `(Q_f βg s₂/B)^⅓` (m/s).
    pub fn mouth_speed(&self, s2: f64) -> f64 {
        (self.qf * self.beta_g() * s2 / self.width).cbrt()
    }

    /// Stigebrandt's thickness of the brackish layer at the mouth with the
    /// mixing factor `p`, `(P² Q_f²/(βg s₂ B²))^⅓` (m).
    pub fn mouth_thickness(&self, s2: f64, p: f64) -> f64 {
        (p * p * self.qf * self.qf / (self.beta_g() * s2 * self.width * self.width)).cbrt()
    }
}

/// The nodes of a cross-section `x = const` of the fjord, with their
/// quadrature weights across it (m): a line of GLL nodes, half weighted on
/// each side where the line is an element face.
pub struct Section {
    pub x: f64,
    nodes: Vec<(usize, f64)>,
}

/// Time sums over a section, per node and level: `H_z u` and `H_z u S`
/// (the transport per unit width and its salt), `H_z`, `H_z S` and the
/// layer's depth below the surface.
#[derive(Clone)]
struct SectionSums {
    q: Vec<f64>,
    qs: Vec<f64>,
    hz: Vec<f64>,
    hzs: Vec<f64>,
    z: Vec<f64>,
    samples: usize,
}

/// A section's exchange split by the sign of the transport of each node
/// and level: Knudsen's in- and outflow.
#[derive(Clone, Copy, Debug)]
pub struct Exchange {
    /// Outflow and inflow (m³/s, both positive)
    pub q_out: f64,
    pub q_in: f64,
    /// Their flux-weighted salinities
    pub s_out: f64,
    pub s_in: f64,
}

impl Exchange {
    /// Knudsen's closure `Q_out/(P Q_f)`, `P = s_in/(s_in − s_out)`: 1 in a
    /// steady state.
    pub fn knudsen(&self, qf: f64) -> f64 {
        self.q_out * (self.s_in - self.s_out) / (self.s_in * qf)
    }
}

/// The brackish layer of a mean profile as one layer of uniform speed: the
/// levels from the surface down to the halocline's foot (the first one at
/// least `s₂ − δ`), with the kinematic equivalents of a continuous profile,
/// `h₁ = (∫u dz)²/∫u² dz` and `u₁ = ∫u² dz/∫u dz` (they keep its transport
/// and momentum flux), and the transport-weighted salinity `s₁`.
#[derive(Clone, Copy, Debug)]
pub struct Brackish {
    pub h1: f64,
    pub u1: f64,
    pub s1: f64,
}

impl Brackish {
    pub fn of(profile: &[Level], s2: f64, delta: f64) -> Self {
        let (mut q, mut q2, mut qs) = (0.0, 0.0, 0.0);
        for level in profile.iter().take_while(|l| l.salt < s2 - delta) {
            q += level.u * level.hz;
            q2 += level.u * level.u * level.hz;
            qs += level.u * level.salt * level.hz;
        }
        Self {
            h1: q * q / q2,
            u1: q2 / q,
            s1: qs / q,
        }
    }

    /// The mixing factor `P = s₂/(s₂ − s₁)`.
    pub fn mixing_factor(&self, s2: f64) -> f64 {
        s2 / (s2 - self.s1)
    }

    /// `F₁² = u₁²/(g′h₁)`, `g′ = βg(s₂ − s₁)`.
    pub fn froude2(&self, s2: f64, beta_g: f64) -> f64 {
        self.u1 * self.u1 / (beta_g * (s2 - self.s1) * self.h1)
    }
}

/// One level of a mean profile across a section: the mean depth of its
/// centre and its mean thickness (m), the width-mean speed (m/s) and
/// salinity.
#[derive(Clone, Copy, Debug)]
pub struct Level {
    pub depth: f64,
    pub hz: f64,
    pub u: f64,
    pub salt: f64,
}

impl SectionSums {
    fn new(n: usize) -> Self {
        Self {
            q: vec![0.0; n],
            qs: vec![0.0; n],
            hz: vec![0.0; n],
            hzs: vec![0.0; n],
            z: vec![0.0; n],
            samples: 0,
        }
    }

    /// The mean profile across the section, surface first.
    fn profile(&self, section: &Section, nl: usize, width: f64) -> Vec<Level> {
        let n = self.samples.max(1) as f64;
        (0..nl)
            .rev()
            .map(|l| {
                let (mut z, mut q, mut hz, mut hzs) = (0.0, 0.0, 0.0, 0.0);
                for (j, &(_, w)) in section.nodes.iter().enumerate() {
                    let c = j * nl + l;
                    z += w * self.z[c];
                    q += w * self.q[c];
                    hz += w * self.hz[c];
                    hzs += w * self.hzs[c];
                }
                Level {
                    depth: z / (n * width),
                    hz: hz / (n * width),
                    u: q / hz,
                    salt: hzs / hz,
                }
            })
            .collect()
    }

    /// The exchange of the mean: each node-level cell is outflow or inflow
    /// by the sign of its mean transport.
    fn exchange(&self, section: &Section, nl: usize) -> Exchange {
        let n = self.samples.max(1) as f64;
        let (mut q1, mut q2, mut f1, mut f2) = (0.0, 0.0, 0.0, 0.0);
        for (j, &(_, w)) in section.nodes.iter().enumerate() {
            for l in 0..nl {
                let c = j * nl + l;
                let (q, qs) = (w * self.q[c] / n, w * self.qs[c] / n);
                if q > 0.0 {
                    q1 += q;
                    f1 += qs;
                } else {
                    q2 -= q;
                    f2 -= qs;
                }
            }
        }
        Exchange {
            q_out: q1,
            q_in: q2,
            s_out: f1 / q1,
            s_in: f2 / q2,
        }
    }
}

/// The mouth at a progress report.
pub struct Progress {
    pub t: f64,
    pub dt: f64,
    pub exchange: Exchange,
    pub brackish: Brackish,
    pub salt_min: f64,
}

/// A section's mean over the averaging window.
pub struct SectionMean {
    pub x: f64,
    pub bed: f64,
    pub exchange: Exchange,
    /// Surface first
    pub profile: Vec<Level>,
}

impl SectionMean {
    /// The brackish layer above `S = s₂ − delta`, `s₂` the inflow's.
    pub fn brackish(&self, delta: f64) -> Brackish {
        Brackish::of(&self.profile, self.exchange.s_in, delta)
    }
}

/// What a run gives: the sections' means (the mouth last) and the run's
/// salinity range and cost.
pub struct Outcome {
    pub n_elements: usize,
    pub sections: Vec<SectionMean>,
    pub salt_range: (f64, f64),
    /// Where and when S was lowest: time (s), (x, y) (m), the still-water
    /// depth (m), the level (bed first)
    pub salt_min_at: (f64, [f64; 2], f64, usize),
    pub steps: usize,
    /// Whether every report and the end were finite
    pub finite: bool,
}

impl Outcome {
    pub fn mouth(&self) -> &SectionMean {
        self.sections.last().expect("sections")
    }
}

/// Run the experiment, calling `progress` every `config.report` hours.
pub fn run(config: &Config, mut progress: impl FnMut(&Progress)) -> Outcome {
    let c = config;
    // A rectangle over the fjord and the basin; the fjord is the strip
    // |y| < B/2 landward of x = length
    let (nx, ny) = (
        ((c.length + c.basin) / c.dx).round() as usize,
        (c.basin / c.dx).round() as usize,
    );
    let full = Mesh2D::uniform_rectangle_with_sides(
        0.0,
        c.length + c.basin,
        -0.5 * c.basin,
        0.5 * c.basin,
        nx,
        ny,
        [
            BoundaryTag::Open,
            BoundaryTag::Open,
            BoundaryTag::Open,
            BoundaryTag::Wall,
        ],
    );
    let (mesh, _) = full.retain_elements(
        |k| {
            let v = full.element_vertices(k);
            let (cx, cy) = (
                0.25 * v.iter().map(|p| p[0]).sum::<f64>(),
                0.25 * v.iter().map(|p| p[1]).sum::<f64>(),
            );
            cx > c.length || cy.abs() < 0.5 * c.width
        },
        BoundaryTag::Wall,
    );
    let mesh = Arc::new(mesh);
    let ops = Arc::new(DGOperators2D::new(c.order));
    let geom = Arc::new(GeometricFactors2D::compute(&mesh, &ops));
    let bathymetry = Arc::new(Bathymetry2D::from_function(&mesh, &ops, &geom, |x, _| {
        c.bed(x)
    }));
    let sigma = c.sigma();
    let (nn, nl) = (ops.n_nodes, sigma.n_levels());

    let wall = Reflective2D::new();
    let open = CharacteristicOBC::new(StillWater::default());
    let swe = PhysicsBuilder::swe_2d(
        mesh.clone(),
        ops.clone(),
        geom.clone(),
        ShallowWater2D::new(G),
        MultiBoundaryCondition2D::new(&wall).with_open(&open),
    )
    .with_bathymetry(bathymetry.clone())
    .with_source(CoriolisSource2D::f_plane(c.f))
    .build();

    let eos = LinearEOS::default();
    let river = River::new("head", [0.5 * c.dx, 1.0], c.qf)
        .with_temperature(RiverSeries::constant(eos.t0))
        .with_salinity(RiverSeries::constant(0.0))
        .with_profile(RiverProfile::TaperedTopMetres(3.0));
    let rivers = RiverSources::new(vec![river], &mesh, &geom, 0.0).expect("the head");

    // The sea at rest, uniform: the fjord starts full of it
    let mut state = Solution3D::new(mesh.n_elements, nn, nl);
    state.temp.fill(eos.t0);
    state.salt.fill(S_SEA);
    let sea = Arc::new(ReferenceColumns::from_state(&state));
    let nesting = Nesting3D::new(
        sea,
        &mesh,
        &ops,
        nl,
        &[BoundaryTag::Open],
        &NestingBand3D {
            width: c.band,
            tracer_timescale: Some(1800.0),
            ..NestingBand3D::default()
        },
    )
    .expect("open faces");

    let physics = Hydrostatic3D::new(
        mesh.clone(),
        ops.clone(),
        geom.clone(),
        Arc::new(sigma.clone()),
        bathymetry.clone(),
        Arc::new(CoriolisSource2D::f_plane(c.f)),
        eos,
        GlsMixing::k_epsilon(),
        swe,
        Forcing {
            surface_stress: [0.0, 0.0],
            bottom_stress: [0.0, 0.0],
            surface_heat_flux: 0.0,
        },
        G,
        RHO0,
    )
    .with_rivers(rivers)
    .with_nesting(nesting)
    .with_bottom_drag(BottomDrag3D::log_layer(0.003))
    .with_smagorinsky_viscosity(0.1)
    .with_implicit_vertical_advection(ImplicitVerticalAdvection::default())
    .with_tracer_limiter(TracerLimiter3DConfig {
        limiter_type: TracerLimiterType3D::HorizontalKuzmin { relaxation: 1.0 },
        ..TracerLimiter3DConfig::default()
    });
    physics.update_density(&mut state);

    // The sections: node lines across the fjord
    let position = |idx: usize| {
        let (k, i) = (idx / nn, idx % nn);
        mesh.reference_to_physical(ElementIndex::new(k), ops.nodes_r[i], ops.nodes_s[i])
    };
    let section = |x: f64| {
        let mut nodes = Vec::new();
        for k in 0..mesh.n_elements {
            let v = mesh.element_vertices(ElementIndex::new(k));
            let (x0, x1) = (v[0][0].min(v[2][0]), v[0][0].max(v[2][0]));
            let (y0, y1) = (v[0][1].min(v[2][1]), v[0][1].max(v[2][1]));
            if x < x0 - 1e-6 || x > x1 + 1e-6 || y0.abs().max(y1.abs()) > 0.5 * c.width + 1e-6 {
                continue;
            }
            // Half on each side of an element face
            let share = if (x - x0).abs() < 1e-6 || (x - x1).abs() < 1e-6 {
                0.5
            } else {
                1.0
            };
            for i in 0..nn {
                if (position(k * nn + i)[0] - x).abs() < 1e-6 {
                    let j = i / ops.n_1d;
                    nodes.push((k * nn + i, share * ops.weights_1d[j] * 0.5 * (y1 - y0)));
                }
            }
        }
        assert!(!nodes.is_empty(), "no node line at x = {x}");
        let across: f64 = nodes.iter().map(|n| n.1).sum();
        assert!(
            (across - c.width).abs() < 1e-6 * c.width,
            "x = {x}: the section's weights add up to {across} m, not {}",
            c.width
        );
        Section { x, nodes }
    };
    let node_spacing = c.dx / c.order as f64;
    let snap = |x: f64| (x / node_spacing).round() * node_spacing;
    // A quarter, half and three quarters of the way to the sill's flank, the
    // flank (1 km before the crest), the crest and the mouth
    let flank = c.crest() - 1000.0;
    let sections: Vec<Section> = [0.25, 0.5, 0.75]
        .iter()
        .map(|&s| snap(s * flank))
        .chain([flank, c.crest(), c.length])
        .map(section)
        .collect();
    let sample = |sums: &mut SectionSums, section: &Section, s: &Solution3D| {
        for (j, &(idx, _)) in section.nodes.iter().enumerate() {
            let h = s.eta.data[idx] - bathymetry.data[idx];
            for (l, &ds) in sigma.d_sigma().iter().enumerate() {
                let c = j * nl + l;
                let hz = h * ds;
                let q = hz * s.u[idx * nl + l];
                sums.q[c] += q;
                sums.qs[c] += q * s.salt[idx * nl + l];
                sums.hz[c] += hz;
                sums.hzs[c] += hz * s.salt[idx * nl + l];
                sums.z[c] += -sigma.sigma_rho()[l] * h;
            }
        }
        sums.samples += 1;
    };

    let scheme = SspScheme::Rk43;
    let mut integrator = ModeSplitIntegrator::new()
        .with_barotropic_scheme(SspScheme::Rk43)
        .with_baroclinic_scheme(scheme);
    let t_end = c.hours * 3600.0;
    let t_average = t_end - c.average * 3600.0;
    let mut sums: Vec<SectionSums> = sections
        .iter()
        .map(|s| SectionSums::new(s.nodes.len() * nl))
        .collect();
    let mouth = sections.len() - 1;
    let report = if c.report > 0.0 {
        c.report * 3600.0
    } else {
        f64::INFINITY
    };
    let (mut t, mut steps, mut next_report) = (0.0, 0usize, report);
    let mut salt_range = (f64::INFINITY, f64::NEG_INFINITY);
    let mut salt_min_at = (0.0, [0.0; 2], 0.0, 0);
    let mut finite = true;
    while t < t_end - 1e-9 {
        let dt = physics
            .compute_dt_ssp(&state, c.cfl, scheme.ssp_coefficient())
            .min(c.dt_max)
            .min(next_report - t)
            .min(t_end - t);
        physics.update_density(&mut state);
        integrator.step(&mut state, &physics, dt, t);
        physics.post_process(&mut state);
        t += dt;
        steps += 1;
        if t > t_average {
            for (sum, section) in sums.iter_mut().zip(&sections) {
                sample(sum, section, &state);
            }
        }
        for (i, &s) in state.salt.iter().enumerate() {
            if s < salt_range.0 {
                let idx = i / nl;
                salt_min_at = (t, position(idx), -bathymetry.data[idx], i % nl);
            }
            salt_range = (salt_range.0.min(s), salt_range.1.max(s));
        }
        if !(salt_range.0.is_finite() && salt_range.1.is_finite()) {
            finite = false;
            break;
        }
        if t >= next_report - 1e-9 {
            let mut now = SectionSums::new(sections[mouth].nodes.len() * nl);
            sample(&mut now, &sections[mouth], &state);
            let exchange = now.exchange(&sections[mouth], nl);
            let brackish = Brackish::of(
                &now.profile(&sections[mouth], nl, c.width),
                exchange.s_in,
                c.delta,
            );
            progress(&Progress {
                t,
                dt,
                exchange,
                brackish,
                salt_min: state.salt.iter().copied().fold(f64::INFINITY, f64::min),
            });
            next_report += report;
        }
    }
    Outcome {
        n_elements: mesh.n_elements,
        sections: sections
            .iter()
            .zip(&sums)
            .map(|(section, sum)| SectionMean {
                x: section.x,
                bed: c.bed(section.x),
                exchange: sum.exchange(section, nl),
                profile: sum.profile(section, nl, c.width),
            })
            .collect(),
        salt_range,
        salt_min_at,
        steps,
        finite,
    }
}

//! A sunlit tidal flat in 3D: the experiment of `examples/tidal_flat_3d.rs`
//! and of its gate in `tests/tidal_flat_test.rs` (`#[path]`-included by
//! both). See the example's docs.

// Not every user reads every field
#![allow(dead_code)]

use std::sync::Arc;

use dgcom::boundary::Reflective2D;
use dgcom::equations::ShallowWater2D;
use dgcom::mesh::{Bathymetry2D, Mesh2D};
use dgcom::operators::{DGOperators2D, GeometricFactors2D};
use dgcom::physics::{
    BottomDrag3D, Forcing, GlsMixing, Hydrostatic3D, ImplicitVerticalAdvection, JerlovWater,
    LinearEOS, PhysicsBuilder, SeaSurface, SurfaceFluxFields, SurfaceFluxes3D,
};
use dgcom::simulation::Simulation3D;
use dgcom::solver::state::Solution3D;
use dgcom::solver::{
    SWEState2D, StandardLimiter2D, TracerLimiter3DConfig, TracerLimiterType3D,
    TracerReferenceProfile, WetDryConfig,
};
use dgcom::source::{CoriolisSource2D, SourceContext2D, SourceTerm2D};
use dgcom::time::{ModeSplitIntegrator, SspScheme};
use dgcom::types::ElementIndex;
use dgcom::vertical::{SigmaGrid, SongHaidvogelStretching};

pub const G: f64 = 9.81;
pub const RHO0: f64 = 1025.0;

/// The experiment's parameters (lengths in m, times in s unless named).
#[derive(Clone, Debug)]
pub struct Config {
    pub length: f64,
    pub width: f64,
    pub nx: usize,
    pub ny: usize,
    pub order: usize,
    pub levels: usize,
    /// Columns shallower than this are thin
    /// (`Hydrostatic3D::with_min_column_depth`)
    pub thin_depth: f64,
    /// Period of the rocking force (the tide)
    pub tide_period: f64,
    /// Its amplitude: the level at the shore end swings by about this
    pub tide: f64,
    /// The weather's period (a day), and the hours to run
    pub day: f64,
    pub hours: f64,
    /// Whether the weather's heat and freshwater fluxes act
    pub heat: bool,
    /// The shortwave at noon (W/m²)
    pub shortwave: f64,
    /// The non-solar loss at 10 °C (W/m²) and what each degree above adds
    /// (W m⁻² K⁻¹)
    pub cooling: f64,
    pub feedback: f64,
    /// Amplitude of E − P (kg m⁻² s⁻¹): evaporation by day, rain by night
    pub freshwater: f64,
    pub jerlov: JerlovWater,
    /// Uniform T and S instead of the stratification
    pub uniform: Option<(f64, f64)>,
    /// GLS closure: `k-epsilon` or `generic`
    pub generic_mixing: bool,
    pub limiter: bool,
    pub scheme: SspScheme,
    pub cfl: f64,
    pub dt_max: f64,
    /// Local time stepping levels of the 3D stages
    pub levels_lts: usize,
    pub implicit_vertical: bool,
    /// Depth (m) from which a column counts as resolved water in the
    /// report's third class
    pub deep: f64,
    /// Whether every step compares each thin column's values with those
    /// around it before the step (`Outcome::local_excess`)
    pub local_check: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            length: 2400.0,
            width: 480.0,
            nx: 10,
            ny: 2,
            order: 2,
            levels: 20,
            thin_depth: 1.0,
            tide_period: 3.0 * 3600.0,
            tide: 1.6,
            day: 6.0 * 3600.0,
            hours: 12.0,
            heat: true,
            shortwave: 800.0,
            cooling: 150.0,
            feedback: 30.0,
            freshwater: 5e-5,
            jerlov: JerlovWater::III,
            uniform: None,
            generic_mixing: false,
            limiter: true,
            scheme: SspScheme::Rk43,
            cfl: 4.0,
            dt_max: 60.0,
            levels_lts: 0,
            implicit_vertical: true,
            deep: 3.0,
            local_check: false,
        }
    }
}

impl Config {
    /// The bed elevation (m): a basin rising from −6 m to −0.9 m over the
    /// first 40 % of the length, a flat rising to −0.1 m at 80 %, and a
    /// shore rising to +2.3 m, tilted by ±0.15 m across.
    pub fn bed(&self, x: f64, y: f64) -> f64 {
        let s = x / self.length;
        let along = if s < 0.4 {
            -6.0 + 5.1 * s / 0.4
        } else if s < 0.8 {
            -0.9 + 0.8 * (s - 0.4) / 0.4
        } else {
            -0.1 + 2.4 * (s - 0.8) / 0.2
        };
        along + 0.15 * (2.0 * y / self.width - 1.0)
    }

    /// The initial temperature and salinity at height `z` (m): 11 °C and 33
    /// at the surface, 8 °C and 34 at −6 m.
    pub fn profile(&self, z: f64) -> (f64, f64) {
        match self.uniform {
            Some(uniform) => uniform,
            None => {
                let z = z.clamp(-6.0, 0.0);
                (11.0 + 0.5 * z, 33.0 - z / 6.0)
            }
        }
    }

    pub fn sigma(&self) -> SigmaGrid {
        SigmaGrid::new(self.levels, SongHaidvogelStretching::new(5.0, 0.4, 10.0))
    }

    /// The weather at time `t`: the shortwave, the non-solar flux of a
    /// surface at `sst`, and E − P.
    pub fn weather(&self, t: f64, sst: f64) -> (f64, f64, f64) {
        if !self.heat {
            return (0.0, 0.0, 0.0);
        }
        // Noon at the first low water, midnight at the second
        let phase = std::f64::consts::TAU * (t - 0.25 * self.tide_period) / self.day;
        let sun = phase.sin();
        (
            self.shortwave * sun.max(0.0),
            -self.cooling - self.feedback * (sst - 10.0),
            self.freshwater * sun,
        )
    }
}

/// The force that rocks the basin: `g h s(t)` along x, the slope of an
/// equilibrium tide `s(t) = (2a/L) sin(2πt/T)`.
struct RockingForce {
    slope: f64,
    period: f64,
}

impl SourceTerm2D for RockingForce {
    fn evaluate(&self, ctx: &SourceContext2D) -> SWEState2D {
        let slope = self.slope * (std::f64::consts::TAU * ctx.time / self.period).sin();
        SWEState2D::new(0.0, ctx.g * ctx.state.h * slope, 0.0)
    }

    fn name(&self) -> &'static str {
        "rocking_force"
    }
}

/// [`Config::weather`] on every column.
struct Weather(Config);

impl SurfaceFluxes3D for Weather {
    fn surface_fluxes_into(&self, t: f64, surface: SeaSurface<'_>, fluxes: SurfaceFluxFields<'_>) {
        for (((q_ns, q_sw), f), &sst) in fluxes
            .nonsolar
            .iter_mut()
            .zip(fluxes.shortwave.iter_mut())
            .zip(fluxes.freshwater.iter_mut())
            .zip(surface.temperature)
        {
            (*q_sw, *q_ns, *f) = self.0.weather(t, sst);
        }
    }
}

pub type Physics = Hydrostatic3D<LinearEOS, GlsMixing, Reflective2D>;

/// The flat's model and its state at rest at mean sea level.
pub fn build(c: &Config) -> (Physics, Solution3D) {
    let mesh = Arc::new(Mesh2D::uniform_rectangle(
        0.0, c.length, 0.0, c.width, c.nx, c.ny,
    ));
    let ops = Arc::new(DGOperators2D::new(c.order));
    let geom = Arc::new(GeometricFactors2D::compute(&mesh, &ops));
    let bathymetry = Arc::new(Bathymetry2D::from_function(&mesh, &ops, &geom, |x, y| {
        c.bed(x, y)
    }));
    let f = 1.3e-4;
    // The 2D module of a case's 3D run (`dgcom::case`)
    let swe = PhysicsBuilder::swe_2d(
        mesh.clone(),
        ops.clone(),
        geom.clone(),
        ShallowWater2D::new(G),
        Reflective2D::default(),
    )
    .with_bathymetry(bathymetry.clone())
    .with_limiter(StandardLimiter2D::Positivity(WetDryConfig::DEFAULT_H_DRY))
    .with_wet_dry(WetDryConfig::default())
    .with_source(CoriolisSource2D::f_plane(f))
    .with_source(RockingForce {
        slope: 2.0 * c.tide / c.length,
        period: c.tide_period,
    })
    .build();

    let sigma = Arc::new(c.sigma());
    let (nn, nl) = (ops.n_nodes, sigma.n_levels());
    let mut state = Solution3D::new(mesh.n_elements, nn, nl);
    for idx in 0..mesh.n_elements * nn {
        let b = bathymetry.data[idx];
        let eta = b.max(0.0);
        state.eta.data[idx] = eta;
        for (l, &s) in sigma.sigma_rho().iter().enumerate() {
            let (t, salt) = c.profile(eta + s * (eta - b));
            state.temp[idx * nl + l] = t;
            state.salt[idx * nl + l] = salt;
        }
    }
    let reference = Arc::new(TracerReferenceProfile::from_fn(-6.0, 0.0, 13, |z| {
        c.profile(z)
    }));
    let limiter = if c.limiter {
        TracerLimiter3DConfig {
            limiter_type: TracerLimiterType3D::HorizontalKuzmin { relaxation: 1.0 },
            ..TracerLimiter3DConfig::default()
        }
        .with_reference_profile(reference.clone())
    } else {
        TracerLimiter3DConfig::default()
    };
    let mixing = if c.generic_mixing {
        GlsMixing::generic()
    } else {
        GlsMixing::k_epsilon()
    };
    let mut physics = Hydrostatic3D::new(
        mesh.clone(),
        ops.clone(),
        geom.clone(),
        sigma,
        bathymetry,
        Arc::new(CoriolisSource2D::f_plane(f)),
        LinearEOS::default(),
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
    .with_bottom_drag(BottomDrag3D::log_layer(0.003))
    .with_min_column_depth(c.thin_depth)
    .with_smagorinsky_viscosity(0.1)
    .with_tracer_limiter(limiter);
    if c.heat {
        physics = physics
            .with_surface_fluxes(Weather(c.clone()))
            .with_shortwave_absorption(c.jerlov);
    }
    if c.implicit_vertical {
        physics = physics.with_implicit_vertical_advection(ImplicitVerticalAdvection::default());
    }
    physics.update_density(&mut state);
    let physics = physics.with_reference_profile(reference);
    (physics, state)
}

/// The lowest and highest value seen, and where the highest departure
/// from `[low, high]` was.
#[derive(Clone, Copy, Debug)]
pub struct Range {
    pub low: f64,
    pub high: f64,
}

impl Default for Range {
    fn default() -> Self {
        Self {
            low: f64::INFINITY,
            high: f64::NEG_INFINITY,
        }
    }
}

impl Range {
    pub fn add(&mut self, x: f64) {
        self.low = self.low.min(x);
        self.high = self.high.max(x);
    }

    pub fn merge(&mut self, other: Range) {
        self.low = self.low.min(other.low);
        self.high = self.high.max(other.high);
    }

    pub fn is_empty(&self) -> bool {
        self.low > self.high
    }

    /// How far `self` reaches outside `other` (0 if inside).
    pub fn outside(&self, other: Range) -> f64 {
        if self.is_empty() {
            return 0.0;
        }
        (other.low - self.low).max(self.high - other.high).max(0.0)
    }
}

/// The kinds of column the ranges are kept for.
pub const DRY: usize = 0;
pub const THIN: usize = 1;
pub const SHALLOW: usize = 2;
pub const DEEP: usize = 3;
pub const KINDS: [&str; 4] = ["dry", "thin", "shallow", "deep"];

/// T and S ranges by kind of column (`[kind][T, S]`): dry (no water, under
/// a micrometre), thin (under the thin depth), shallow (to `Config::deep`)
/// and deep.
#[derive(Clone, Copy, Debug, Default)]
pub struct Ranges(pub [[Range; 2]; 4]);

impl Ranges {
    pub fn of(c: &Config, physics: &Physics, s: &Solution3D) -> Self {
        let nl = s.n_levels;
        let mut ranges = Self::default();
        for idx in 0..s.eta.data.len() {
            let depth = s.eta.data[idx] - physics.bathymetry.data[idx];
            let kind = if depth < 1e-6 {
                DRY
            } else if depth < c.thin_depth {
                THIN
            } else if depth < c.deep {
                SHALLOW
            } else {
                DEEP
            };
            for l in 0..nl {
                ranges.0[kind][0].add(s.temp[idx * nl + l]);
                ranges.0[kind][1].add(s.salt[idx * nl + l]);
            }
        }
        ranges
    }

    pub fn merge(&mut self, other: &Ranges) {
        for (mine, theirs) in self.0.iter_mut().zip(&other.0) {
            for (a, b) in mine.iter_mut().zip(theirs) {
                a.merge(*b);
            }
        }
    }
}

/// `∫ Σ_l H_z C dA` of a tracer (dry columns hold none), and the water's
/// volume.
pub fn inventory(physics: &Physics, s: &Solution3D, tracer: &[f64]) -> f64 {
    let (nn, nl) = (s.n_nodes, s.n_levels);
    let d_sigma = physics.sigma.d_sigma();
    let mut sum = 0.0;
    for k in 0..s.n_elements {
        for i in 0..nn {
            let idx = k * nn + i;
            let depth = (s.eta.data[idx] - physics.bathymetry.data[idx]).max(0.0);
            let column: f64 = (0..nl)
                .map(|l| depth * d_sigma[l] * tracer[idx * nl + l])
                .sum();
            sum += physics.geom.node_mass(k, i) * column;
        }
    }
    sum
}

pub fn volume(physics: &Physics, s: &Solution3D) -> f64 {
    let nn = s.n_nodes;
    let mut sum = 0.0;
    for k in 0..s.n_elements {
        for i in 0..nn {
            let idx = k * nn + i;
            let depth = (s.eta.data[idx] - physics.bathymetry.data[idx]).max(0.0);
            sum += physics.geom.node_mass(k, i) * depth;
        }
    }
    sum
}

/// A report of a run, every `report` seconds: the time, the step, the
/// ranges over the interval, the level at the shore end's wet edge.
pub struct Progress<'a> {
    pub t: f64,
    pub steps: usize,
    pub dt: (f64, f64),
    pub ranges: Ranges,
    pub state: &'a Solution3D,
    pub physics: &'a Physics,
}

/// What a run gives.
pub struct Outcome {
    pub steps: usize,
    pub dt: (f64, f64),
    pub wall: f64,
    /// The ranges over the whole run
    pub ranges: Ranges,
    /// The initial ranges of T and S
    pub initial: [Range; 2],
    /// Relative drift of the volume and of the T and S inventories (the
    /// last two meaningful without fluxes)
    pub drift: [f64; 3],
    pub clips: usize,
    pub error: Option<String>,
    /// Columns seen dry, thin with water, and crossing the thin depth
    pub seen: [usize; 3],
    /// How far T and S of a thin or dry column were outside the range the
    /// other columns have had up to then, the largest over the steps (with
    /// the weather, a thin column's own fluxes are outside it too)
    pub thin_excess: [f64; 2],
    /// ... and where the largest of T was: time (s), node, level, depth (m)
    pub thin_excess_at: (f64, usize, usize, f64),
    /// How far T of a thin column was, after a step, outside the range of T
    /// before the step within four elements of it (the reach of four
    /// stages; on every level, since the columns that are not thin exchange
    /// between theirs), the largest over the steps, and how many values were
    /// outside by more than 1e-9 °C; with `Config::local_check`
    pub local_excess: (f64, usize),
    /// The largest change of T at a node that was dry before and after a
    /// step (a dry node keeps its value)
    pub dry_change: f64,
}

/// The elements within `rings` elements of each element (sharing a vertex,
/// ring by ring), itself included.
fn neighbourhoods(mesh: &Mesh2D, rings: usize) -> Vec<Vec<usize>> {
    (0..mesh.n_elements)
        .map(|k| {
            let mut near = vec![k];
            let mut from = 0;
            for _ in 0..rings {
                let until = near.len();
                for j in from..until {
                    for vertex in mesh.element_vertex_indices(ElementIndex::new(near[j])) {
                        for &e in mesh.elements_at_vertex(vertex) {
                            if !near.contains(&e) {
                                near.push(e);
                            }
                        }
                    }
                }
                from = until;
            }
            near
        })
        .collect()
}

/// Run the experiment, calling `progress` every `report` seconds (none if
/// not positive) and `each` after every step.
pub fn run(
    c: &Config,
    report: f64,
    mut progress: impl FnMut(&Progress),
    mut each: impl FnMut(&Solution3D, f64, &Physics),
) -> Outcome {
    let (physics, mut state) = build(c);
    let mut initial = [Range::default(); 2];
    for (&t, &s) in state.temp.iter().zip(&state.salt) {
        initial[0].add(t);
        initial[1].add(s);
    }
    let start = [
        volume(&physics, &state),
        inventory(&physics, &state, &state.temp),
        inventory(&physics, &state, &state.salt),
    ];
    let integrator = ModeSplitIntegrator::new()
        .with_barotropic_cfl(2.0)
        .with_barotropic_scheme(SspScheme::Rk43)
        .with_baroclinic_scheme(c.scheme)
        .with_barotropic_levels(2)
        .with_baroclinic_levels(c.levels_lts);
    let mut sim = Simulation3D::new(physics, integrator)
        .with_cfl(c.cfl)
        .with_dt_max(c.dt_max);
    if report > 0.0 {
        sim = sim.with_landing_interval(report);
    }
    let t_end = 3600.0 * c.hours;
    let mut ranges = Ranges::default();
    // The range of the columns that are not thin, up to now, from the
    // start's range of every column (the land's values too)
    let mut resolved = initial;
    let mut thin_excess = [0.0_f64; 2];
    let mut thin_excess_at = (0.0, 0, 0, 0.0);
    let near = neighbourhoods(&sim.physics().mesh, 4);
    let mut previous: Option<Solution3D> = None;
    let mut local_excess = (0.0_f64, 0usize);
    let mut dry_change = 0.0_f64;
    let mut interval = Ranges::default();
    let mut drift = [0.0_f64; 3];
    let mut seen = [0usize; 3];
    let mut was_thin: Vec<bool> = Vec::new();
    let (mut steps, mut last, mut dt_range) = (0usize, 0.0, (f64::INFINITY, 0.0_f64));
    let mut interval_dt = (f64::INFINITY, 0.0_f64);
    let mut next = report;
    let result = sim.run_with_physics_callback(&mut state, 0.0, t_end, |s, t, physics| {
        let now = Ranges::of(c, physics, s);
        ranges.merge(&now);
        interval.merge(&now);
        for (j, r) in resolved.iter_mut().enumerate() {
            r.merge(now.0[SHALLOW][j]);
            r.merge(now.0[DEEP][j]);
        }
        let nl = s.n_levels;
        for idx in 0..s.eta.data.len() {
            let depth = s.eta.data[idx] - physics.bathymetry.data[idx];
            if depth >= c.thin_depth {
                continue;
            }
            for l in 0..nl {
                for (j, field) in [&s.temp, &s.salt].into_iter().enumerate() {
                    let x = field[idx * nl + l];
                    let outside = (resolved[j].low - x).max(x - resolved[j].high);
                    if outside > thin_excess[j] {
                        thin_excess[j] = outside;
                        if j == 0 {
                            thin_excess_at = (t, idx, l, depth);
                        }
                    }
                }
            }
        }
        if let Some(before) = &previous {
            let nn = s.n_nodes;
            let depth_of =
                |state: &Solution3D, idx: usize| state.eta.data[idx] - physics.bathymetry.data[idx];
            for idx in 0..s.eta.data.len() {
                let (was, is) = (depth_of(before, idx), depth_of(s, idx));
                if was < 1e-6 && is < 1e-6 {
                    for l in 0..nl {
                        let change = (s.temp[idx * nl + l] - before.temp[idx * nl + l]).abs();
                        dry_change = dry_change.max(change);
                    }
                }
                if !c.local_check || is >= c.thin_depth || is < 1e-6 {
                    continue;
                }
                // Every level: the columns that are not thin exchange
                // between theirs
                let mut local = Range::default();
                for &e in &near[idx / nn] {
                    for &x in &before.temp[e * nn * nl..(e + 1) * nn * nl] {
                        local.add(x);
                    }
                }
                for l in 0..nl {
                    let x = s.temp[idx * nl + l];
                    let outside = (local.low - x).max(x - local.high);
                    local_excess.0 = local_excess.0.max(outside);
                    local_excess.1 += usize::from(outside > 1e-9);
                }
            }
        }
        previous = Some(s.clone());
        let thin: Vec<bool> = (0..s.eta.data.len())
            .map(|idx| s.eta.data[idx] - physics.bathymetry.data[idx] < c.thin_depth)
            .collect();
        for (idx, &is_thin) in thin.iter().enumerate() {
            let depth = s.eta.data[idx] - physics.bathymetry.data[idx];
            seen[0] += usize::from(depth < 1e-6);
            seen[1] += usize::from(is_thin && depth >= 1e-6);
            seen[2] += usize::from(was_thin.get(idx).is_some_and(|&w| w != is_thin));
        }
        was_thin = thin;
        let end = [
            volume(physics, s),
            inventory(physics, s, &s.temp),
            inventory(physics, s, &s.salt),
        ];
        for ((d, a), b) in drift.iter_mut().zip(end).zip(start) {
            *d = d.max((a - b).abs() / b.abs());
        }
        if t > 0.0 {
            steps += 1;
            let dt = t - last;
            dt_range = (dt_range.0.min(dt), dt_range.1.max(dt));
            interval_dt = (interval_dt.0.min(dt), interval_dt.1.max(dt));
            last = t;
            each(s, t, physics);
        }
        if report > 0.0 && t >= next - 1e-6 {
            progress(&Progress {
                t,
                steps,
                dt: interval_dt,
                ranges: interval,
                state: s,
                physics,
            });
            interval = Ranges::default();
            interval_dt = (f64::INFINITY, 0.0);
            next += report;
        }
    });
    Outcome {
        steps,
        dt: dt_range,
        wall: result.wall_time,
        ranges,
        initial,
        drift,
        clips: sim.physics().swe_physics.negative_depth_clips(),
        error: (!result.success).then(|| result.error.unwrap_or_default()),
        seen,
        thin_excess,
        thin_excess_at,
        local_excess,
        dry_change,
    }
}

/// The position of node `idx` (`[element][node]`).
pub fn position(physics: &Physics, idx: usize) -> [f64; 2] {
    let nn = physics.ops.n_nodes;
    let (k, i) = (idx / nn, idx % nn);
    physics.mesh.reference_to_physical(
        ElementIndex::new(k),
        physics.ops.nodes_r[i],
        physics.ops.nodes_s[i],
    )
}

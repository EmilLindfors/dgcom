//! Cost profile of the 3D mode-split step (TODO P4.5: parallel,
//! non-allocating 3D kernels).
//!
//! A stratified fjord-like basin (bed 20 → 200 m across a sill, a tanh
//! pycnocline, Coriolis, wind, quadratic bottom drag, Smagorinsky viscosity
//! of the shear, Pacanowski–Philander mixing or GLS k-ε (`mixing=gls`), the
//! horizontal Kuzmin tracer limiter) stepped with [`ModeSplitIntegrator`];
//! with GLS, `turbulence=local` keeps `k` and `ψ` in their columns instead
//! of advecting them. Every [`ModeSplitPhysics`]
//! call is timed through a wrapper; the barotropic pass and the splitter's
//! own work are the rest of the step. `mform=conservative` advects the
//! momentum in conservative instead of split form.
//!
//! ```text
//! cargo run --release --no-default-features --features parallel,simd --example profile_3d -- \
//!     nx=64 ny=16 order=2 levels=20 steps=20
//! ```

use std::cell::RefCell;
use std::sync::Arc;
use std::time::{Duration, Instant};

use dg_rs::boundary::Reflective2D;
use dg_rs::equations::ShallowWater2D;
use dg_rs::mesh::Mesh2D;
use dg_rs::mesh::data::Bathymetry2D;
use dg_rs::operators::{DGOperators2D, GeometricFactors2D};
use dg_rs::physics::{
    BottomDrag3D, Forcing, GlsMixing, Hydrostatic3D, LinearEOS, PacanowskiPhilanderMixing,
    PhysicsBuilder, VerticalMixing,
};
use dg_rs::solver::rhs::{BarotropicFlux, MomentumAdvectionForm, VerticalAdvection};
use dg_rs::solver::state::Solution3D;
use dg_rs::solver::{SWEFormulation2D, SWESolution2D, TracerLimiter3DConfig, TracerLimiterType3D};
use dg_rs::source::CoriolisSource2D;
use dg_rs::time::{ModeSplitIntegrator, ModeSplitPhysics, StepDrag};
use dg_rs::vertical::{SigmaGrid, SongHaidvogelStretching};

const G: f64 = 9.81;
const RHO0: f64 = 1025.0;

/// Wall-clock time of each [`ModeSplitPhysics`] call of the wrapped model.
struct Timed<'a, P> {
    inner: &'a P,
    times: RefCell<Vec<(&'static str, Duration, usize)>>,
}

impl<'a, P> Timed<'a, P> {
    fn new(inner: &'a P) -> Self {
        Self {
            inner,
            times: RefCell::new(Vec::new()),
        }
    }

    fn time<R>(&self, name: &'static str, f: impl FnOnce() -> R) -> R {
        let start = Instant::now();
        let r = f();
        let elapsed = start.elapsed();
        let mut times = self.times.borrow_mut();
        match times.iter_mut().find(|(n, _, _)| *n == name) {
            Some((_, t, count)) => {
                *t += elapsed;
                *count += 1;
            }
            None => times.push((name, elapsed, 1)),
        }
        r
    }
}

impl<P: ModeSplitPhysics> ModeSplitPhysics for Timed<'_, P> {
    type Barotropic = P::Barotropic;

    fn barotropic(&self) -> &Self::Barotropic {
        self.inner.barotropic()
    }

    fn sigma(&self) -> &SigmaGrid {
        self.inner.sigma()
    }

    fn bathymetry(&self) -> &Bathymetry2D {
        self.inner.bathymetry()
    }

    fn momentum_rhs_into(&self, state: &Solution3D, t: f64, out: &mut Solution3D) {
        self.time("momentum_rhs", || {
            self.inner.momentum_rhs_into(state, t, out)
        })
    }

    fn transport_rhs_into(
        &self,
        state: &Solution3D,
        t: f64,
        barotropic: BarotropicFlux,
        out: &mut Solution3D,
    ) {
        self.time("transport_rhs", || {
            self.inner.transport_rhs_into(state, t, barotropic, out)
        })
    }

    fn slow_forcing_into(
        &self,
        state: &Solution3D,
        rhs: &Solution3D,
        t: f64,
        g: &mut SWESolution2D,
    ) {
        self.time("slow_forcing", || {
            self.inner.slow_forcing_into(state, rhs, t, g)
        })
    }

    fn bottom_drag_into(&self, state: &Solution3D, t: f64, rate: &mut [f64]) -> bool {
        self.time("bottom_drag", || {
            self.inner.bottom_drag_into(state, t, rate)
        })
    }

    fn layer_drag_into(&self, state: &Solution3D, t: f64, rate: &mut [f64]) -> bool {
        self.time("layer_drag", || self.inner.layer_drag_into(state, t, rate))
    }

    fn vertical_implicit(&self, state: &mut Solution3D, t: f64, dt: f64, drag: StepDrag<'_>) {
        self.time("vertical_implicit", || {
            self.inner.vertical_implicit(state, t, dt, drag)
        })
    }

    fn post_stage(&self, state: &mut Solution3D) {
        self.time("post_stage", || self.inner.post_stage(state))
    }

    fn end_density(&self, state: &mut Solution3D) {
        self.time("end_density", || self.inner.end_density(state))
    }

    fn mask_stage(&self, state: &mut Solution3D, stash: &mut dg_rs::time::StageStash) {
        self.inner.mask_stage(state, stash)
    }

    fn unmask_stage(&self, state: &mut Solution3D, stash: &dg_rs::time::StageStash) {
        self.inner.unmask_stage(state, stash)
    }
}

fn arg<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::args()
        .skip(1)
        .find_map(|a| {
            a.strip_prefix(name)
                .and_then(|v| v.strip_prefix('='))
                .and_then(|v| v.parse().ok())
        })
        .unwrap_or(default)
}

fn main() {
    let nx: usize = arg("nx", 64);
    let ny: usize = arg("ny", 16);
    let order: usize = arg("order", 2);
    let levels: usize = arg("levels", 20);
    let steps: usize = arg("steps", 20);
    let dt: f64 = arg("dt", 60.0);
    let cs: f64 = arg("cs", 0.5);
    let mixing: Box<dyn VerticalMixing> = match arg("mixing", "pp".to_string()).as_str() {
        "pp" => Box::new(PacanowskiPhilanderMixing::new(1e-2, 1e-5, 1e-5, G, RHO0)),
        "gls" => Box::new(GlsMixing::k_epsilon().with_roughness(0.02, 0.005)),
        other => panic!("mixing={other}: expected pp or gls"),
    };

    let (length, width) = (40e3, 10e3);
    let mesh = Arc::new(Mesh2D::uniform_rectangle(0.0, length, 0.0, width, nx, ny));
    let ops = Arc::new(DGOperators2D::new(order));
    let geom = Arc::new(GeometricFactors2D::compute(&mesh, &ops));
    // A 200 m basin behind a 20 m sill at x = 10 km, shoaling across the width
    let bathymetry = Arc::new(Bathymetry2D::from_function(&mesh, &ops, &geom, |x, y| {
        let sill = 1.0 - 0.9 * (-((x - 10e3) / 2e3).powi(2)).exp();
        -(20.0 + 180.0 * sill * (1.0 - 0.5 * (y / width - 0.5).abs()))
    }));
    let sigma = SigmaGrid::new(levels, SongHaidvogelStretching::new(5.0, 0.4, 20.0));
    let f = 1.2e-4;
    let swe = PhysicsBuilder::swe_2d(
        mesh.clone(),
        ops.clone(),
        geom.clone(),
        ShallowWater2D::new(G),
        Reflective2D::default(),
    )
    .with_bathymetry(bathymetry.clone())
    .with_formulation(SWEFormulation2D::EntropyStable)
    .with_source(CoriolisSource2D::f_plane(f))
    .build();
    let forcing = Forcing {
        surface_stress: [0.1, 0.05],
        bottom_stress: [0.0, 0.0],
        surface_heat_flux: 0.0,
    };
    let physics = Hydrostatic3D::new(
        mesh.clone(),
        ops.clone(),
        geom.clone(),
        Arc::new(sigma.clone()),
        bathymetry.clone(),
        Arc::new(CoriolisSource2D::f_plane(f)),
        LinearEOS::default(),
        mixing,
        swe,
        forcing,
        G,
        RHO0,
    )
    .with_bottom_drag(BottomDrag3D::log_layer(0.005))
    .with_turbulence_advection(match arg("turbulence", "advected".to_string()).as_str() {
        "advected" => Some(VerticalAdvection::LimitedAkima),
        "local" => None,
        other => panic!("turbulence={other}: expected advected or local"),
    })
    .with_smagorinsky_viscosity(cs)
    .with_momentum_advection_form(match arg("mform", "split".to_string()).as_str() {
        "split" => MomentumAdvectionForm::Split,
        "conservative" => MomentumAdvectionForm::Conservative,
        other => panic!("mform={other}: expected split or conservative"),
    })
    .with_tracer_limiter(TracerLimiter3DConfig {
        limiter_type: TracerLimiterType3D::HorizontalKuzmin { relaxation: 1.0 },
        ..TracerLimiter3DConfig::default()
    });

    let (nn, nl) = (ops.n_nodes, sigma.n_levels());
    let mut state = Solution3D::new(mesh.n_elements, nn, nl);
    let eos = LinearEOS::default();
    for idx in 0..mesh.n_elements * nn {
        let depth = -bathymetry.data[idx];
        for (l, &s) in sigma.sigma_rho().iter().enumerate() {
            // 8 °C across a pycnocline at 15 m
            state.temp[idx * nl + l] = eos.t0 + 4.0 * ((s * depth + 15.0) / 5.0).tanh();
            state.salt[idx * nl + l] = eos.s0;
        }
    }
    physics.update_density(&mut state);
    // The vertical tracer advection about the initial stratification, as
    // `Simulation3D` does
    let physics = physics.with_vertical_reference(&state);

    println!(
        "profile_3d: {} elements × {nn} nodes × {nl} levels = {} 3D points, P{order}, dt {dt} s, \
         {} threads",
        mesh.n_elements,
        mesh.n_elements * nn * nl,
        rayon_threads(),
    );

    let timed = Timed::new(&physics);
    let mut integrator = ModeSplitIntegrator::new();
    let (mut step_time, mut density_time, mut post_time, mut dt_time) = (
        Duration::ZERO,
        Duration::ZERO,
        Duration::ZERO,
        Duration::ZERO,
    );
    // One warm-up step allocates the buffers
    for n in 0..=steps {
        if n == 1 {
            timed.times.borrow_mut().clear();
            (step_time, density_time, post_time, dt_time) = (
                Duration::ZERO,
                Duration::ZERO,
                Duration::ZERO,
                Duration::ZERO,
            );
        }
        let start = Instant::now();
        let _ = physics.compute_dt(&state, 0.5);
        dt_time += start.elapsed();
        let start = Instant::now();
        physics.update_density(&mut state);
        density_time += start.elapsed();
        let start = Instant::now();
        integrator.step(&mut state, &timed, dt, n as f64 * dt);
        step_time += start.elapsed();
        let start = Instant::now();
        physics.post_process(&mut state);
        post_time += start.elapsed();
    }

    let max_speed = state
        .u
        .iter()
        .zip(&state.v)
        .map(|(u, v)| u.hypot(*v))
        .fold(0.0_f64, f64::max);
    assert!(max_speed.is_finite(), "the run blew up");
    // Bit pattern of the end state: the kernels must not change it with the
    // thread count (or when only made faster)
    let checksum = [
        &state.u,
        &state.v,
        &state.w,
        &state.temp,
        &state.salt,
        &state.eta.data,
        &state.ubar.data,
        &state.vbar.data,
    ]
    .iter()
    .flat_map(|field| field.iter())
    .fold(0xcbf29ce484222325_u64, |h, x| {
        (h ^ x.to_bits()).wrapping_mul(0x100000001b3)
    });
    println!("state checksum {checksum:016x}");
    let per_step = |d: Duration| d.as_secs_f64() * 1e3 / steps as f64;
    let total = step_time + density_time + post_time + dt_time;
    println!(
        "{steps} steps, {} barotropic substeps each, max |u| {max_speed:.3} m/s",
        integrator.last_substeps()
    );
    println!("{:<28}{:>10}{:>8}{:>8}", "ms per step", "ms", "%", "calls");
    let row = |name: &str, d: Duration, calls: usize| {
        println!(
            "{name:<28}{:>10.2}{:>7.1}%{:>8}",
            per_step(d),
            100.0 * d.as_secs_f64() / total.as_secs_f64(),
            calls / steps
        );
    };
    let times = timed.times.borrow();
    let inside: Duration = times.iter().map(|(_, d, _)| *d).sum();
    for &(name, d, calls) in times.iter() {
        row(name, d, calls);
    }
    row("pass + splitter", step_time - inside, steps);
    row("compute_dt", dt_time, steps);
    row("update_density", density_time, steps);
    row("post_process", post_time, steps);
    row("total", total, steps);
}

fn rayon_threads() -> usize {
    #[cfg(feature = "parallel")]
    {
        rayon::current_num_threads()
    }
    #[cfg(not(feature = "parallel"))]
    {
        1
    }
}

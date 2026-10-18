//! Stratified seamount at rest (Beckmann & Haidvogel 1993), TODO P4.6.
//!
//! A Gaussian seamount `H(r) = H₀(1 − A e^{−r²/L²})` in a doubly periodic
//! ocean on an f-plane, horizontally uniform stratification, everything at
//! rest. The exact solution is rest; every current the model makes comes
//! from the pressure-gradient error over the steep flanks (and, once the
//! flow moves density, its adjustment). The measure is the largest speed
//! over days, as in Beckmann & Haidvogel (1993) and Shchepetkin &
//! McWilliams (2003, their Fig. 10).
//!
//! No horizontal or vertical tracer diffusion (a real boundary flow on a
//! slope would follow from vertical diffusion: Phillips 1970, Wunsch 1970);
//! a small vertical viscosity.
//!
//! What it found (2026-10-01; see `dg_rs::solver::rhs::baroclinic`): with
//! the constant-depth PGF and conservative tracer advection a stratified
//! fluid at rest over the slope was unstable, round-off growing tenfold
//! every 6 h even for constant N², at any time step and against any
//! viscosity tried. The σ-pairs PGF with split-form tracer advection, the
//! pair consistent in energy, holds constant N² at round-off; its σ-form
//! error at rest is taken back by a balanced reference state.
//!
//! ```text
//! cargo run --release --no-default-features --features parallel,simd --example seamount_3d -- \
//!     nx=16 order=2 levels=20 days=10 profile=tanh
//! ```
//!
//! Arguments: `nx` elements across, `order`, `levels` (Song–Haidvogel θs 5,
//! θb 0.4), `days`, `dt` (s, 0 for the 3D CFL bound at `cfl`), `profile`
//! (`tanh`: a 3 kg/m³ pycnocline at 30 m, 10 m thick; `exp`: Shchepetkin &
//! McWilliams's `Δρ e^{z/d}` with `d` a fifth of the deep depth; `linear`:
//! constant N², balanced to round-off), `amp` (A), `depth` (H₀, m),
//! `width` (L, m), `domain` (m), `f` (s⁻¹), `uniform=1` for uniform levels,
//! `pgf` (`sigma`, the default σ-pairs, or `depth`, constant depth),
//! `balanced=1` to balance the initial stratification
//! (`Hydrostatic3D::with_balanced_reference`), `nu`/`kappa` vertical
//! viscosity and diffusivity, `nuh` or `cs` horizontal viscosity (constant
//! or Smagorinsky), `vadv`/`madv` the tracers' and the momentum's vertical
//! advection (`vref=0`: without the initial stratification as the vertical
//! reference, `Hydrostatic3D::with_vertical_reference`, the default as in
//! `Simulation3D`), `mform` the momentum's horizontal advection (`split`, the
//! default, or `conservative`), `kuzmin=1` the horizontal Kuzmin tracer limiter (`kuzmin=2`
//! with the initial stratification as its reference profile; `relax` its
//! bounds' relaxation, default 1), `report` the interval of the printed
//! lines (h).

use std::sync::Arc;
use std::time::Instant;

use dg_rs::boundary::Reflective2D;
use dg_rs::equations::ShallowWater2D;
use dg_rs::mesh::Mesh2DBuilder;
use dg_rs::mesh::data::Bathymetry2D;
use dg_rs::operators::{DGOperators2D, GeometricFactors2D};
use dg_rs::physics::{ConstantMixing, Forcing, Hydrostatic3D, LinearEOS, PhysicsBuilder};
use dg_rs::solver::rhs::{MomentumAdvectionForm, PressureGradientForm, VerticalAdvection};
use dg_rs::solver::state::Solution3D;
use dg_rs::solver::{
    SWEFormulation2D, TracerLimiter3DConfig, TracerLimiterType3D, TracerReferenceProfile,
};
use dg_rs::source::CoriolisSource2D;
use dg_rs::time::ModeSplitIntegrator;
use dg_rs::types::ElementIndex;
use dg_rs::vertical::{SigmaGrid, SongHaidvogelStretching, UniformStretching};

const G: f64 = 9.81;
const RHO0: f64 = 1025.0;

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
    let nx: usize = arg("nx", 16);
    let order: usize = arg("order", 2);
    let levels: usize = arg("levels", 20);
    let days: f64 = arg("days", 10.0);
    let dt_fixed: f64 = arg("dt", 0.0);
    let cfl: f64 = arg("cfl", 1.0);
    let profile: String = arg("profile", "tanh".to_string());
    let amp: f64 = arg("amp", 0.9);
    let depth: f64 = arg("depth", 400.0);
    let width: f64 = arg("width", 8e3);
    let domain: f64 = arg("domain", 64e3);
    let f: f64 = arg("f", 1.2e-4);
    let uniform: usize = arg("uniform", 0);
    let viscosity: f64 = arg("nu", 1e-4);
    let nu_h: f64 = arg("nuh", 0.0);
    let kappa: f64 = arg("kappa", 0.0);
    let kuzmin: usize = arg("kuzmin", 0);
    let cs: f64 = arg("cs", 0.0);
    let scheme = |name: &str| match name {
        "centred" => VerticalAdvection::Centred,
        "upwind" => VerticalAdvection::Upwind,
        "akima" => VerticalAdvection::Akima,
        "tvd" => VerticalAdvection::Tvd,
        "limited" => VerticalAdvection::LimitedAkima,
        other => panic!("vertical advection {other}: centred, upwind, akima, tvd or limited"),
    };
    let tracer_vadv = scheme(&arg("vadv", "limited".to_string()));
    let momentum_vadv = scheme(&arg("madv", "centred".to_string()));
    let pgf = match arg("pgf", "sigma".to_string()).as_str() {
        "sigma" => PressureGradientForm::SigmaPairs,
        "depth" => PressureGradientForm::ConstantDepth,
        other => panic!("pgf={other}: expected sigma or depth"),
    };
    let balanced: usize = arg("balanced", 0);
    let momentum_form = match arg("mform", "split".to_string()).as_str() {
        "split" => MomentumAdvectionForm::Split,
        "conservative" => MomentumAdvectionForm::Conservative,
        other => panic!("mform={other}: expected split or conservative"),
    };

    let mesh = Arc::new(
        Mesh2DBuilder::new(-domain / 2.0, domain / 2.0, -domain / 2.0, domain / 2.0)
            .with_resolution(nx, nx)
            .fully_periodic()
            .build(),
    );
    let ops = Arc::new(DGOperators2D::new(order));
    let geom = Arc::new(GeometricFactors2D::compute(&mesh, &ops));
    let bathymetry = Arc::new(Bathymetry2D::from_function(&mesh, &ops, &geom, |x, y| {
        -depth * (1.0 - amp * (-(x * x + y * y) / (width * width)).exp())
    }));
    let sigma = if uniform != 0 {
        SigmaGrid::new(levels, UniformStretching)
    } else {
        SigmaGrid::new(levels, SongHaidvogelStretching::new(5.0, 0.4, 20.0))
    };
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
    let no_stress = Forcing {
        surface_stress: [0.0, 0.0],
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
        ConstantMixing::new(viscosity, kappa),
        swe,
        no_stress,
        G,
        RHO0,
    );
    let physics = physics
        .with_vertical_advection(tracer_vadv)
        .with_momentum_vertical_advection(momentum_vadv)
        .with_momentum_advection_form(momentum_form)
        .with_pressure_gradient(pgf);
    let mut physics = if cs > 0.0 {
        physics.with_smagorinsky_viscosity(cs)
    } else if nu_h > 0.0 {
        physics.with_horizontal_viscosity(nu_h)
    } else {
        physics
    };

    // Density as a function of z, through T alone (S at the reference)
    let eos = LinearEOS::default();
    let rho_of_z: Box<dyn Fn(f64) -> f64> = match profile.as_str() {
        "tanh" => Box::new(|z: f64| RHO0 - 1.5 + 1.5 * (-(z + 30.0) / 10.0).tanh()),
        "exp" => {
            let scale = depth / 5.0;
            Box::new(move |z: f64| RHO0 - 3.0 * (z / scale).exp())
        }
        "linear" => Box::new(move |z: f64| RHO0 - 3.0 * (1.0 + z / depth)),
        other => panic!("profile={other}: expected tanh, exp or linear"),
    };
    let temperature = |z: f64| eos.t0 + (1.0 - rho_of_z(z) / eos.rho0) / eos.alpha;
    if kuzmin != 0 {
        let limiter = TracerLimiter3DConfig {
            limiter_type: TracerLimiterType3D::HorizontalKuzmin {
                relaxation: arg("relax", 1.0),
            },
            ..TracerLimiter3DConfig::default()
        };
        physics.set_tracer_limiter(if kuzmin == 2 {
            // The initial stratification, every 5 cm (linear between samples:
            // ≈ 2e-5 °C off a 10 m pycnocline; every 0.5 m, 2e-3)
            let samples = (20.0 * depth) as usize + 1;
            limiter.with_reference_profile(TracerReferenceProfile::from_fn(
                -depth,
                0.0,
                samples,
                |z| (temperature(z), eos.s0),
            ))
        } else {
            limiter
        });
    }

    let (nn, nl) = (ops.n_nodes, sigma.n_levels());
    let mut state = Solution3D::new(mesh.n_elements, nn, nl);
    for idx in 0..mesh.n_elements * nn {
        let h = -bathymetry.data[idx];
        for (l, &s) in sigma.sigma_rho().iter().enumerate() {
            state.temp[idx * nl + l] = temperature(s * h);
            state.salt[idx * nl + l] = eos.s0;
        }
    }
    physics.update_density(&mut state);
    if balanced != 0 {
        physics.set_balanced_reference(&state);
    }
    if arg("vref", 1usize) != 0 {
        physics.set_vertical_reference(&state);
    }

    // Steepness: the largest r_x0 between neighbouring nodes of an element
    // and Haney's r_x1 on the σ-levels
    let (mut rx0, mut rx1) = (0.0_f64, 0.0_f64);
    let n1 = order + 1;
    for k in 0..mesh.n_elements {
        for j in 0..n1 {
            for i in 0..n1 {
                let a = k * nn + j * n1 + i;
                for b in [
                    (i + 1 < n1).then(|| k * nn + j * n1 + i + 1),
                    (j + 1 < n1).then(|| k * nn + (j + 1) * n1 + i),
                ]
                .into_iter()
                .flatten()
                {
                    let (ha, hb) = (-bathymetry.data[a], -bathymetry.data[b]);
                    rx0 = rx0.max((ha - hb).abs() / (ha + hb));
                    let z = sigma.sigma_w();
                    for l in 1..z.len() {
                        // |Δz_top + Δz_bottom| / |z_top − z_bottom| across the pair
                        let num = (z[l] + z[l - 1]) * (ha - hb);
                        let den = (z[l] - z[l - 1]) * (ha + hb);
                        rx1 = rx1.max((num / den).abs());
                    }
                }
            }
        }
    }
    println!(
        "seamount_3d: {nx}×{nx} P{order} ({:.2} km node spacing), {nl} {} levels, \
         H₀ {depth} m, A {amp}, L {:.1} km, f {f:.1e}, profile {profile}, ν {viscosity:.0e},          {pgf:?}, {momentum_form:?} momentum{}",
        domain / nx as f64 / order as f64 / 1e3,
        if uniform != 0 { "uniform" } else { "stretched" },
        width / 1e3,
        if balanced != 0 { ", balanced" } else { "" },
    );
    println!("  r_x0 {rx0:.3}, r_x1 {rx1:.2} (node to node)");

    let mut integrator = ModeSplitIntegrator::new();
    let t_end = days * 86400.0;
    let report = 3600.0 * arg("report", 6.0);
    let mut next_report = report;
    let (mut t, mut steps) = (0.0, 0usize);
    let start = Instant::now();
    let speed_stats = |s: &Solution3D| {
        let (mut max, mut ke, mut at) = (0.0_f64, 0.0, 0);
        for (i, (u, v)) in s.u.iter().zip(&s.v).enumerate() {
            let speed = u.hypot(*v);
            if speed.is_nan() || speed > max {
                max = speed;
                at = i;
            }
            ke += speed * speed;
        }
        let bar = s
            .ubar
            .data
            .iter()
            .zip(&s.vbar.data)
            .map(|(u, v)| u.hypot(*v))
            .fold(0.0_f64, f64::max);
        (max, (ke / s.u.len() as f64).sqrt(), bar, at)
    };
    let positions: Vec<[f64; 2]> = ElementIndex::iter(mesh.n_elements)
        .flat_map(|k| {
            let (mesh, ops) = (&mesh, &ops);
            (0..nn).map(move |i| mesh.reference_to_physical(k, ops.nodes_r[i], ops.nodes_s[i]))
        })
        .collect();
    println!(
        "{:>7} {:>11} {:>11} {:>11} {:>7} {:>7} {:>7} {:>7} {:>7}",
        "hours", "max|u| m/s", "rms|u| m/s", "max|ū| m/s", "at z/H", "x km", "y km", "H m", "dt s"
    );
    while t < t_end - 1e-9 {
        let step_dt = if dt_fixed > 0.0 {
            dt_fixed
        } else {
            physics.compute_dt(&state, cfl)
        };
        let dt = step_dt.min(next_report - t);
        physics.update_density(&mut state);
        integrator.step(&mut state, &physics, dt, t);
        physics.post_process(&mut state);
        t += dt;
        steps += 1;
        if t >= next_report - 1e-9 {
            let (max, rms, bar, at) = speed_stats(&state);
            let [x, y] = positions[at / nl];
            println!(
                "{:>7.1} {:>11.3e} {:>11.3e} {:>11.3e} {:>7.2} {:>7.2} {:>7.2} {:>7.1} {:>7.1}",
                t / 3600.0,
                max,
                rms,
                bar,
                sigma.sigma_rho()[at % nl],
                x / 1e3,
                y / 1e3,
                -bathymetry.data[at / nl],
                step_dt
            );
            if !max.is_finite() {
                break;
            }
            next_report += report;
        }
    }
    println!(
        "{steps} steps, {} barotropic substeps, {:.1} s wall",
        integrator.last_substeps(),
        start.elapsed().as_secs_f64()
    );
}

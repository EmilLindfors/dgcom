//! Gate tests for general (non-parallelogram) quadrilateral elements, TODO P1.3.
//!
//! Coastline-fitted meshes are made of general quadrilaterals, whose bilinear
//! map has a Jacobian that varies over the element. The 2D kernels take the
//! metric at every node (`GeometricFactors2D`) in conservative form, so on a
//! distorted periodic mesh they must still:
//!
//! - preserve a uniform flow (free-stream preservation, from the discrete
//!   metric identities);
//! - conserve mass exactly, and momentum on a flat bed;
//! - keep a lake at rest over a rough, face-discontinuous bed (split forms),
//!   also across a shoreline (`WetDry`, with and without subcells);
//! - conserve entropy (`EntropyConservative`) or dissipate it (`EntropyStable`).
//!
//! Convergence at N+1 on distorted meshes is checked in `convergence_test.rs`.

use std::f64::consts::PI;

use dgcom::boundary::Reflective2D;
use dgcom::equations::ShallowWater2D;
use dgcom::flux::{SWENodeState2D, swe_entropy_variables_2d};
use dgcom::mesh::{Bathymetry2D, Mesh2D};
use dgcom::operators::{DGOperators2D, GeometricFactors2D};
use dgcom::solver::{
    SWE2DRhsConfig, SWEFormulation2D, SWESolution2D, SWEState2D, compute_rhs_swe_2d,
};
use dgcom::source::HorizontalViscosity2D;
use dgcom::types::ElementIndex;

const G: f64 = 9.81;
const L: f64 = 1000.0;

/// A periodic `n × n` mesh of [0, L]² whose vertices are moved by a smooth
/// periodic field: every element is a general quadrilateral (the bilinear
/// term is of the order of 10 % of the element size).
fn distorted_periodic_mesh(n: usize) -> Mesh2D {
    let mut mesh = Mesh2D::uniform_periodic(0.0, L, 0.0, L, n, n);
    let tau = 2.0 * PI / L;
    let a = 0.04 * L;
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

struct Setup {
    mesh: Mesh2D,
    ops: DGOperators2D,
    geom: GeometricFactors2D,
}

impl Setup {
    fn new(n: usize, order: usize) -> Self {
        let mesh = distorted_periodic_mesh(n);
        let ops = DGOperators2D::new(order);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        assert!(!geom.is_affine(), "the test mesh must be non-affine");
        Self { mesh, ops, geom }
    }

    fn nodes(&self) -> impl Iterator<Item = (ElementIndex, usize)> + '_ {
        ElementIndex::iter(self.mesh.n_elements)
            .flat_map(|k| (0..self.ops.n_nodes).map(move |i| (k, i)))
    }

    fn xy(&self, k: ElementIndex, i: usize) -> (f64, f64) {
        let [x, y] = self
            .mesh
            .reference_to_physical(k, self.ops.nodes_r[i], self.ops.nodes_s[i]);
        (x, y)
    }

    fn state(&self, f: impl Fn(f64, f64) -> SWEState2D) -> SWESolution2D {
        let mut q = SWESolution2D::new(self.mesh.n_elements, self.ops.n_nodes);
        for (k, i) in self.nodes() {
            let (x, y) = self.xy(k, i);
            q.set_state(k, i, f(x, y));
        }
        q
    }

    /// Nodal bed from a smooth field plus a per-element offset: discontinuous
    /// across every face.
    fn rough_bed(&self, f: impl Fn(f64, f64) -> f64) -> Bathymetry2D {
        let mut bed = Bathymetry2D::flat(self.mesh.n_elements, self.ops.n_nodes);
        for (k, i) in self.nodes() {
            let (x, y) = self.xy(k, i);
            let jump = ((7 * k.as_usize()) % 5) as f64 - 2.0;
            bed.set(k, i, f(x, y) + 0.8 * jump);
        }
        bed.compute_gradients(&self.ops, &self.geom);
        bed
    }

    /// Σ_k Σ_i w_i J_i f(k, i)
    fn integrate(&self, f: impl Fn(ElementIndex, usize) -> f64) -> f64 {
        self.nodes()
            .map(|(k, i)| self.geom.mass[self.geom.node_index(k.as_usize(), i)] * f(k, i))
            .sum()
    }

    fn rhs(
        &self,
        q: &SWESolution2D,
        formulation: SWEFormulation2D,
        subcell_depth: f64,
        bed: Option<&Bathymetry2D>,
    ) -> SWESolution2D {
        let equation = ShallowWater2D::new(G);
        let bc = Reflective2D::new(); // never called on a periodic mesh
        let mut config = SWE2DRhsConfig::new(&equation, &bc)
            .with_coriolis(false)
            .with_formulation(formulation)
            .with_subcell_depth(subcell_depth);
        if let Some(bed) = bed {
            config = config.with_bathymetry(bed);
        }
        compute_rhs_swe_2d(q, &self.mesh, &self.ops, &self.geom, &config, 0.0)
    }
}

/// The formulations under test, with the subcell depth that selects them:
/// `WetDry` with a subcell depth above every depth runs the subcell finite volumes
/// in every element.
fn formulations(depth_scale: f64) -> Vec<(&'static str, SWEFormulation2D, f64)> {
    vec![
        ("Standard", SWEFormulation2D::Standard, 0.0),
        (
            "EntropyConservative",
            SWEFormulation2D::EntropyConservative,
            0.0,
        ),
        ("EntropyStable", SWEFormulation2D::EntropyStable, 0.0),
        ("WetDry", SWEFormulation2D::WetDry, 1e-3),
        (
            "WetDry subcells",
            SWEFormulation2D::WetDry,
            10.0 * depth_scale,
        ),
    ]
}

fn max_abs(rhs: &SWESolution2D) -> f64 {
    rhs.data
        .iter()
        .flat_map(|v| v.iter())
        .fold(0.0_f64, |m, x| m.max(x.abs()))
}

/// A uniform flow over a flat bed is steady on a distorted mesh: the metric
/// identities make the conservative volume term and the subcell interface
/// metrics sum to zero.
#[test]
fn uniform_flow_is_preserved_on_distorted_elements() {
    let (h, u, v) = (10.0, 0.8, -0.5);
    for order in 1..=4 {
        let s = Setup::new(5, order);
        let q = s.state(|_, _| SWEState2D::from_primitives(h, u, v));
        // Scale of the individual terms: the momentum flux divergence per
        // element, g h² / Δx
        let scale = G * h * h / (L / 5.0);
        for (name, formulation, subcell_depth) in formulations(h) {
            let rhs = s.rhs(&q, formulation, subcell_depth, None);
            let err = max_abs(&rhs);
            assert!(
                err < 1e-12 * scale,
                "P{order} {name}: uniform flow drifts at {err:.3e} (scale {scale:.3e})"
            );
        }
    }
}

/// Lake at rest over a rough bed that jumps at every face, fully wet: the
/// split forms balance the pressure and the bed on distorted elements
/// (Wintermeyer et al. 2017, curvilinear form; the subcells by their
/// interface metrics).
#[test]
fn lake_at_rest_over_rough_bed_on_distorted_elements() {
    for order in 1..=4 {
        let s = Setup::new(5, order);
        let bed =
            s.rough_bed(|x, y| -40.0 + 15.0 * (2.0 * PI * x / L).sin() * (2.0 * PI * y / L).cos());
        let q = s.state(|_, _| SWEState2D::new(0.0, 0.0, 0.0));
        let mut q = q;
        for (k, i) in s.nodes() {
            q.set_state(k, i, SWEState2D::new(-bed.get(k, i), 0.0, 0.0));
        }
        let scale = G * 60.0 * 20.0 / (L / 5.0);
        for (name, formulation, subcell_depth) in formulations(60.0).into_iter().skip(1) {
            let rhs = s.rhs(&q, formulation, subcell_depth, Some(&bed));
            let err = max_abs(&rhs);
            assert!(
                err < 1e-11 * scale,
                "P{order} {name}: lake at rest drifts at {err:.3e} (scale {scale:.3e})"
            );
        }
    }
}

/// Lake at rest across shorelines: dry islands (bed above the surface) and
/// partially dry elements on a distorted mesh, `WetDry`.
#[test]
fn shoreline_lake_at_rest_on_distorted_elements() {
    for order in 1..=4 {
        let s = Setup::new(6, order);
        let bed =
            s.rough_bed(|x, y| -1.0 + 4.0 * (2.0 * PI * x / L).sin() * (2.0 * PI * y / L).sin());
        let mut q = SWESolution2D::new(s.mesh.n_elements, s.ops.n_nodes);
        let mut n_dry = 0;
        for (k, i) in s.nodes() {
            let h = (-bed.get(k, i)).max(0.0);
            n_dry += usize::from(h == 0.0);
            q.set_state(k, i, SWEState2D::new(h, 0.0, 0.0));
        }
        assert!(n_dry > 0, "the test needs dry nodes");
        let rhs = s.rhs(&q, SWEFormulation2D::WetDry, 1e-3, Some(&bed));
        let err = max_abs(&rhs);
        let scale = G * 5.0 * 5.0 / (L / 6.0);
        assert!(
            err < 1e-11 * scale,
            "P{order}: shoreline lake at rest drifts at {err:.3e} (scale {scale:.3e})"
        );
    }
}

/// Smooth, non-uniform flow over a rough bed: the element-integrated depth
/// tendency telescopes to the face fluxes, so the total mass rate is zero to
/// round-off; on a flat bed the momentum rate is zero too.
#[test]
fn mass_and_momentum_are_conserved_on_distorted_elements() {
    let tau = 2.0 * PI / L;
    for order in 1..=4 {
        let s = Setup::new(5, order);
        let bed = s.rough_bed(|x, y| -40.0 + 10.0 * (tau * x).cos() * (tau * y).sin());
        let flow = |x: f64, y: f64| {
            SWEState2D::from_primitives(
                30.0 + 3.0 * (tau * x).sin() * (tau * y).cos(),
                0.6 * (tau * y).sin(),
                -0.4 * (tau * x).cos(),
            )
        };
        let q = s.state(flow);
        let q_bed = {
            let mut q = q.clone();
            for (k, i) in s.nodes() {
                let (x, y) = s.xy(k, i);
                let state = flow(x, y);
                let h = state.h - 40.0 - bed.get(k, i);
                q.set_state(
                    k,
                    i,
                    SWEState2D::from_primitives(h.max(1.0), state.hu / state.h, state.hv / state.h),
                );
            }
            q
        };

        for (name, formulation, subcell_depth) in formulations(40.0) {
            // Mass, with the bed (split forms) or without (Standard)
            let bed_arg = (formulation != SWEFormulation2D::Standard).then_some(&bed);
            let state = if bed_arg.is_some() { &q_bed } else { &q };
            let rhs = s.rhs(state, formulation, subcell_depth, bed_arg);
            let rate = s.integrate(|k, i| rhs.get_state(k, i).h);
            let scale = s.integrate(|k, i| rhs.get_state(k, i).h.abs());
            assert!(
                rate.abs() < 1e-12 * scale,
                "P{order} {name}: mass rate {rate:.3e} of {scale:.3e}"
            );

            // Momentum on a flat bed
            let rhs = s.rhs(&q, formulation, subcell_depth, None);
            for (c, label) in [(1, "x"), (2, "y")] {
                let rate = s.integrate(|k, i| rhs.get_var(k, i, c));
                let scale = s.integrate(|k, i| rhs.get_var(k, i, c).abs());
                assert!(
                    rate.abs() < 1e-12 * scale,
                    "P{order} {name}: {label}-momentum rate {rate:.3e} of {scale:.3e}"
                );
            }
        }
    }
}

/// `EntropyConservative` conserves the total energy on distorted elements
/// (to round-off), `EntropyStable` dissipates it.
#[test]
fn entropy_is_conserved_or_dissipated_on_distorted_elements() {
    let tau = 2.0 * PI / L;
    for order in 1..=4 {
        let s = Setup::new(5, order);
        let bed = s.rough_bed(|x, y| -40.0 + 10.0 * (tau * x).cos() * (tau * y).sin());
        let mut q = SWESolution2D::new(s.mesh.n_elements, s.ops.n_nodes);
        for (k, i) in s.nodes() {
            let (x, y) = s.xy(k, i);
            let eta = 0.5 * (tau * x).sin() * (tau * y).cos();
            q.set_state(
                k,
                i,
                SWEState2D::from_primitives(
                    eta - bed.get(k, i),
                    0.5 * (tau * y).sin(),
                    -0.3 * (tau * x).cos(),
                ),
            );
        }
        let entropy_rate = |rhs: &SWESolution2D| {
            let w_dot_rhs = |k: ElementIndex, i: usize| {
                let node = SWENodeState2D::new(&q.get_state(k, i), bed.get(k, i), 1e-6);
                let w = swe_entropy_variables_2d(&node, G);
                let r = rhs.get_state(k, i);
                w.h * r.h + w.hu * r.hu + w.hv * r.hv
            };
            (
                s.integrate(w_dot_rhs),
                s.integrate(|k, i| w_dot_rhs(k, i).abs()),
            )
        };

        let rhs = s.rhs(&q, SWEFormulation2D::EntropyConservative, 0.0, Some(&bed));
        let (rate, scale) = entropy_rate(&rhs);
        assert!(
            rate.abs() < 1e-11 * scale,
            "P{order} EntropyConservative: entropy rate {rate:.3e} of {scale:.3e}"
        );

        let rhs = s.rhs(&q, SWEFormulation2D::EntropyStable, 0.0, Some(&bed));
        let (rate, scale) = entropy_rate(&rhs);
        assert!(
            rate < 1e-11 * scale,
            "P{order} EntropyStable: entropy rate {rate:.3e} of {scale:.3e} should be ≤ 0"
        );
    }
}

/// Horizontal (Smagorinsky) viscosity on distorted elements conserves
/// momentum: the BR1 diffusion is in conservative form.
#[test]
fn viscosity_conserves_momentum_on_distorted_elements() {
    let tau = 2.0 * PI / L;
    let equation = ShallowWater2D::new(G);
    let bc = Reflective2D::new();
    let viscosity = HorizontalViscosity2D::smagorinsky(0.2);
    for order in 1..=4 {
        let s = Setup::new(5, order);
        let q = s.state(|x, y| {
            SWEState2D::from_primitives(
                30.0,
                0.6 * (tau * y).sin() * (tau * x).cos(),
                -0.4 * (tau * x).cos(),
            )
        });
        let with = SWE2DRhsConfig::new(&equation, &bc)
            .with_coriolis(false)
            .with_viscosity(&viscosity);
        let without = SWE2DRhsConfig::new(&equation, &bc).with_coriolis(false);
        let rhs_with = compute_rhs_swe_2d(&q, &s.mesh, &s.ops, &s.geom, &with, 0.0);
        let rhs_without = compute_rhs_swe_2d(&q, &s.mesh, &s.ops, &s.geom, &without, 0.0);
        for c in [1, 2] {
            let visc = |k: ElementIndex, i: usize| {
                rhs_with.get_var(k, i, c) - rhs_without.get_var(k, i, c)
            };
            let rate = s.integrate(visc);
            let scale = s.integrate(|k, i| visc(k, i).abs());
            assert!(scale > 0.0, "P{order}: the viscous term should act");
            assert!(
                rate.abs() < 1e-11 * scale,
                "P{order}: viscous momentum rate {rate:.3e} of {scale:.3e}"
            );
        }
    }
}

//! The force of the waves on the depth-averaged flow: the divergence of the
//! radiation stress (Longuet-Higgins & Stewart 1964), TODO F.4 coupling.
//!
//! Phase-averaged waves carry momentum; where they shoal, refract or break, the
//! change of their momentum flux, the radiation stress `S`, acts on the mean flow:
//!
//! ```text
//! ∂(hu)/∂t = … − (1/ρ) ∂S_xj/∂x_j,   ∂(hv)/∂t = … − (1/ρ) ∂S_yj/∂x_j
//! ```
//!
//! It sets the water down seaward of the breakers and up in the surf zone, and
//! drives longshore currents under oblique waves. [`WaveForce2D`] takes `S/(ρg)`
//! (m²) per node, as [`WaveModel2D::radiation_stress`] gives it on the same mesh
//! and order, and keeps `−g ∇·(S/ρg)` per node: the strong-form DG divergence of
//! each row of `S` (Hesthaven & Warburton 2008, §6.2), the contravariant volume
//! term plus the lifted jumps to the central flux `S* = ½(S⁻ + S⁺)` on every
//! interior face (`S* = S⁻` on the boundary). On GLL nodes summation by parts
//! makes it a flux form: the force's total over the domain is exactly
//! `−g ∮ S·n` over the boundary, with the momentum the waves give one element
//! taken from its neighbour. Dry nodes (`h ≤ h_min`) get no force.
//!
//! [`WaveForce2D::from_dissipation`] is the alternative of Dingemans et al.
//! (1987), SWAN's `DISSIPATION` force: `F = g Σ (k/σ) e_θ D`, the momentum
//! the waves lose where their sinks take energy. `−∇·S` is that plus a part
//! that is nearly a gradient, from shoaling and refraction without loss,
//! which the mean level balances. The dissipation's force keeps the currents
//! and the setup where the waves break, but has no set-down seaward of the
//! breakers. On a plane beach (`tests/wave_coupling_test.rs`) the longshore
//! current follows Longuet-Higgins's to 0.6 % against `−∇·S`'s 3.8 %, the
//! setup at the shore is within 1.4 % of `−∇·S`'s, and the set-down is gone
//! (−0.68 cm). It also drops the large, nearly cancelling `−∇·S` where a
//! coarse wave field shoals and refracts unresolved against a steep coast.
//!
//! The vortex-force form of McWilliams et al. (2004), which separates the
//! conservative part into a Bernoulli head and needs the Stokes drift in the
//! advection, is an alternative for 3D (TODO F.4).
//!
//! [`WaveCurrentFriction2D`] raises the bed friction of the current under
//! waves: the wave boundary layer's turbulence enhances the mean stress,
//! Soulsby's (1995) fit to the combined wave–current models,
//!
//! ```text
//! τ_m = τ_c [1 + 1.2 (τ_w/(τ_c + τ_w))^3.2],
//! ```
//!
//! with `τ_c` the stress of the current alone (any [`BottomFriction2D`] law) and
//! `τ_w` the waves' ([`WaveModel2D::bed_wave_stress`]). Up to 2.2× the current's
//! where the waves dominate; unchanged without them.

use crate::boundary::tidal_ramp;
use crate::mesh::Mesh2D;
use crate::operators::{DGOperators2D, GeometricFactors2D};
use crate::solver::SWEState2D;
use crate::source::{BottomFriction2D, ElementSources, SourceContext2D, SourceTerm2D};
use crate::types::ElementIndex;
use crate::waves::{WaveModel2D, WaveSolution};

/// The force of waves on the 2D flow, from the radiation stress or the
/// dissipation (see the module docs).
#[derive(Clone, Debug)]
pub struct WaveForce2D {
    n_nodes: usize,
    /// The force per node (m²/s², force per unit area over ρ): `−g ∇·(S/ρg)`
    /// or the dissipation's
    force: Vec<[f64; 2]>,
    /// Node positions, for [`SourceTerm2D::evaluate`]
    positions: Vec<[f64; 2]>,
    ramp: Option<f64>,
    h_min: f64,
    /// Linear in time from `force` toward another force ([`Self::between`])
    toward: Option<Toward>,
}

/// The second end of a force linear in time: `force` at `t1`, the first end's
/// at `t0`.
#[derive(Clone, Debug)]
struct Toward {
    t0: f64,
    t1: f64,
    force: Vec<[f64; 2]>,
}

impl WaveForce2D {
    /// The force of the wave state `n` of `model`, on the model's mesh and order.
    pub fn new(model: &WaveModel2D, n: &WaveSolution) -> Self {
        Self::from_radiation_stress(
            &model.mesh,
            &model.ops,
            &model.geom,
            &model.radiation_stress(n),
            model.g(),
        )
    }

    /// The force of the radiation stress `stress` (`[S_xx, S_xy, S_yy]/(ρg)`, m²,
    /// per node, element by element on `mesh`) under gravity `g`.
    pub fn from_radiation_stress(
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        geom: &GeometricFactors2D,
        stress: &[[f64; 3]],
        g: f64,
    ) -> Self {
        assert_eq!(
            stress.len(),
            mesh.n_elements * ops.n_nodes,
            "a nodal field element by element"
        );
        let force = stress_divergence(mesh, ops, geom, stress)
            .into_iter()
            .map(|[x, y]| [-g * x, -g * y])
            .collect();
        Self::from_force(mesh, ops, force)
    }

    /// The force of the dissipation of the wave state `n` of `model`, on the
    /// model's mesh and order: `F = g Σ (k/σ) e_θ D` per node (Dingemans et
    /// al. 1987; [`crate::waves::Dissipation`]) in place of `−∇·S`: the
    /// wave-driven currents and the setup where the waves break, without the
    /// set-down where they shoal (see the module docs). `D` is what the sinks
    /// remove over a wave step `dt` (s), or at the instant for `dt = 0`
    /// ([`WaveModel2D::dissipation`]).
    pub fn from_dissipation(model: &WaveModel2D, n: &WaveSolution, dt: f64) -> Self {
        let force = model.dissipation(n, dt).iter().map(|d| d.force).collect();
        Self::from_force(&model.mesh, &model.ops, force)
    }

    /// The force `force` per node (m²/s², force per unit area over ρ;
    /// element by element on `mesh` at the order of `ops`), as it is.
    pub fn from_force(mesh: &Mesh2D, ops: &DGOperators2D, force: Vec<[f64; 2]>) -> Self {
        assert_eq!(
            force.len(),
            mesh.n_elements * ops.n_nodes,
            "a nodal field element by element"
        );
        let positions = ElementIndex::iter(mesh.n_elements)
            .flat_map(|k| {
                (0..ops.n_nodes)
                    .map(move |i| mesh.reference_to_physical(k, ops.nodes_r[i], ops.nodes_s[i]))
            })
            .collect();
        Self {
            n_nodes: ops.n_nodes,
            force,
            positions,
            ramp: None,
            h_min: 1e-6,
            toward: None,
        }
    }

    /// The force linear in time from `f0` at `t0` to `f1` at `t1 > t0`, and
    /// constant beyond them, as a coupling gives the circulation between two
    /// exchanges with the waves ([`crate::waves::CoupledWaves2D`]): no jump at
    /// an exchange to ring the basin. The ramp and `h_min` are `f0`'s.
    pub fn between(t0: f64, f0: WaveForce2D, t1: f64, f1: &WaveForce2D) -> Self {
        assert!(t1 > t0, "wave forces must be in time order: {t0} → {t1}");
        assert_eq!(
            f0.force.len(),
            f1.force.len(),
            "wave forces on different meshes"
        );
        Self {
            toward: Some(Toward {
                t0,
                t1,
                force: f1.force.clone(),
            }),
            ..f0
        }
    }

    /// The weights of the force's two ends at time `t`, with the ramp.
    fn weights(&self, t: f64) -> (f64, f64) {
        let ramp = tidal_ramp(t, self.ramp);
        match &self.toward {
            None => (ramp, 0.0),
            Some(toward) => {
                let a = ((t - toward.t0) / (toward.t1 - toward.t0)).clamp(0.0, 1.0);
                (ramp * (1.0 - a), ramp * a)
            }
        }
    }

    /// The force at node `p` with the weights of [`Self::weights`].
    #[inline]
    fn at(&self, p: usize, (w0, w1): (f64, f64)) -> [f64; 2] {
        let [fx, fy] = self.force[p];
        match &self.toward {
            None => [w0 * fx, w0 * fy],
            Some(toward) => {
                let [gx, gy] = toward.force[p];
                [w0 * fx + w1 * gx, w0 * fy + w1 * gy]
            }
        }
    }

    /// Ramp the force up smoothly from 0 at t = 0 to full at `seconds` (the
    /// tidal ramp's `3τ² − 2τ³`), so that a run from rest does not ring.
    pub fn with_ramp(mut self, seconds: f64) -> Self {
        self.ramp = Some(seconds);
        self
    }

    /// No force where the water is shallower than `h_min` (m; default 1e-6).
    pub fn with_h_min(mut self, h_min: f64) -> Self {
        self.h_min = h_min;
        self
    }

    /// The force per node (m²/s²: `−g ∇·(S/ρg)`, or the dissipation's),
    /// unramped; of a force [`Self::between`] two, the first.
    pub fn force(&self) -> &[[f64; 2]] {
        &self.force
    }
}

impl SourceTerm2D for WaveForce2D {
    /// Per-node evaluation by position, a search over the nodes (slow; the RHS
    /// kernels use [`SourceTerm2D::add_element`]). Zero away from a node.
    fn evaluate(&self, ctx: &SourceContext2D) -> SWEState2D {
        if ctx.state.h <= self.h_min {
            return SWEState2D::zero();
        }
        let (x, y) = ctx.position;
        let scale = 1e-9 * (1.0 + x.abs() + y.abs());
        let Some(p) = self
            .positions
            .iter()
            .position(|q| (q[0] - x).abs() <= scale && (q[1] - y).abs() <= scale)
        else {
            return SWEState2D::zero();
        };
        let [fx, fy] = self.at(p, self.weights(ctx.time));
        SWEState2D::new(0.0, fx, fy)
    }

    fn add_element(
        &self,
        element: &ElementSources<'_>,
        _h: &mut [f64],
        hu: &mut [f64],
        hv: &mut [f64],
    ) {
        let weights = self.weights(element.time);
        let base = element.element.as_usize() * self.n_nodes;
        let depths = element.solution.element_h(element.element);
        for (i, &h) in depths.iter().enumerate() {
            if h > self.h_min {
                let [fx, fy] = self.at(base + i, weights);
                hu[i] += fx;
                hv[i] += fy;
            }
        }
    }

    fn is_node_local(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "wave_force_2d"
    }
}

/// The divergence `[∂S_xx/∂x + ∂S_xy/∂y, ∂S_xy/∂x + ∂S_yy/∂y]` of the symmetric
/// nodal tensor `stress` (`[S_xx, S_xy, S_yy]`) in strong DG form: per element
/// `J⁻¹ (D_r F̃_r + D_s F̃_s)` of the contravariant fluxes of each row, plus the
/// lifted `(S* − S⁻)·n` with the central `S* = ½(S⁻ + S⁺)` on interior faces.
fn stress_divergence(
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    stress: &[[f64; 3]],
) -> Vec<[f64; 2]> {
    let (nn, nfn) = (ops.n_nodes, ops.n_face_nodes);
    let mut out = vec![[0.0; 2]; stress.len()];
    let (mut fr, mut fs) = (vec![[0.0; 2]; nn], vec![[0.0; 2]; nn]);
    for k in 0..mesh.n_elements {
        let base = k * nn;
        // The rows (S_xx, S_xy) and (S_xy, S_yy) on the contravariant vectors
        for a in 0..nn {
            let [sxx, sxy, syy] = stress[base + a];
            let ((jrx, jry), (jsx, jsy)) = geom.contravariant(k, a);
            fr[a] = [jrx * sxx + jry * sxy, jrx * sxy + jry * syy];
            fs[a] = [jsx * sxx + jsy * sxy, jsx * sxy + jsy * syy];
        }
        for a in 0..nn {
            let (dr, ds) = (
                &ops.dr_row_major[a * nn..(a + 1) * nn],
                &ops.ds_row_major[a * nn..(a + 1) * nn],
            );
            let mut d = [0.0; 2];
            for b in 0..nn {
                for (row, d) in d.iter_mut().enumerate() {
                    *d += dr[b] * fr[b][row] + ds[b] * fs[b][row];
                }
            }
            let j_inv = geom.jacobian_inv(k, a);
            out[base + a] = [d[0] * j_inv, d[1] * j_inv];
        }
        let element = ElementIndex::new(k);
        for face in 0..4 {
            let Some(neighbour) = mesh.neighbor(element, face) else {
                continue;
            };
            let lift = &ops.lift_row_major[face];
            for fi in 0..nfn {
                let inside = stress[base + ops.face_nodes[face][fi]];
                let q = neighbour.element * nn + ops.face_nodes[neighbour.face][nfn - 1 - fi];
                let outside = stress[q];
                let (nx, ny) = geom.normal(k, face, fi);
                let half: [f64; 3] = std::array::from_fn(|c| 0.5 * (outside[c] - inside[c]));
                let jump = [half[0] * nx + half[1] * ny, half[1] * nx + half[2] * ny];
                let sj = geom.surface_jacobian(k, face, fi);
                for b in 0..nn {
                    let l = lift[b * nfn + fi];
                    if l != 0.0 {
                        let w = l * sj * geom.jacobian_inv(k, b);
                        out[base + b][0] += w * jump[0];
                        out[base + b][1] += w * jump[1];
                    }
                }
            }
        }
    }
    out
}

/// Bottom friction of a current enhanced by waves (see the module docs):
/// `inner`'s damping rate times `1 + 1.2 (τ_w/(τ_c + τ_w))^3.2`.
#[derive(Clone, Debug)]
pub struct WaveCurrentFriction2D<F> {
    inner: F,
    /// The waves' bed stress per ρ (m²/s²) per node
    wave_stress: Vec<f64>,
}

impl<F: BottomFriction2D> WaveCurrentFriction2D<F> {
    /// `inner` under waves whose bed stress per ρ is `wave_stress` (m²/s², per
    /// node, element by element: [`WaveModel2D::bed_wave_stress`]).
    pub fn new(inner: F, wave_stress: Vec<f64>) -> Self {
        assert!(
            wave_stress.iter().all(|&t| t >= 0.0),
            "wave stresses must be non-negative"
        );
        if let Some(n) = inner.n_total_nodes() {
            assert_eq!(n, wave_stress.len(), "one wave stress per node of the law");
        }
        Self { inner, wave_stress }
    }

    /// Replace the waves' stress, as a coupling does every interval.
    pub fn set_wave_stress(&mut self, wave_stress: &[f64]) {
        assert_eq!(wave_stress.len(), self.wave_stress.len());
        self.wave_stress.copy_from_slice(wave_stress);
    }

    /// Soulsby's enhancement `1 + 1.2 (τ_w/(τ_c + τ_w))^3.2` of the current's
    /// stress `tau_c` under the waves' `tau_w` (any common units).
    pub fn enhancement(tau_c: f64, tau_w: f64) -> f64 {
        if tau_w <= 0.0 {
            return 1.0;
        }
        1.0 + 1.2 * (tau_w / (tau_c + tau_w)).powf(3.2)
    }
}

impl<F: BottomFriction2D> BottomFriction2D for WaveCurrentFriction2D<F> {
    /// The inner rate `Λ_c`, times the enhancement with `τ_c/ρ = Λ_c h |u|`.
    fn damping_rate(&self, node: usize, h: f64, speed: f64) -> f64 {
        let rate = self.inner.damping_rate(node, h, speed);
        rate * Self::enhancement(rate * h * speed, self.wave_stress[node])
    }

    fn n_total_nodes(&self) -> Option<usize> {
        Some(self.wave_stress.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A radiation stress quadratic in x and y (exact at P2 on any
    /// parallelogram) gives its exact divergence at every node, and the force
    /// acts only on wet nodes, ramped.
    #[test]
    fn the_force_is_the_exact_divergence_of_a_polynomial_stress() {
        let mut mesh = Mesh2D::uniform_rectangle(0.0, 300.0, 0.0, 200.0, 3, 2);
        // Shear the mesh: the derivatives go through the metric terms
        for v in mesh.vertices.iter_mut() {
            v[0] += 0.3 * v[1];
        }
        let ops = DGOperators2D::new(2);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let g = 9.81;
        let stress_at = |x: f64, y: f64| {
            [
                0.2 + 1e-3 * x - 2e-6 * x * x + 1e-6 * x * y,
                0.05 - 3e-4 * y + 1e-6 * x * y,
                0.1 + 2e-4 * x + 4e-6 * y * y,
            ]
        };
        // −g (∂S_xx/∂x + ∂S_xy/∂y), −g (∂S_xy/∂x + ∂S_yy/∂y)
        let exact = |x: f64, y: f64| {
            [
                -g * ((1e-3 - 4e-6 * x + 1e-6 * y) + (-3e-4 + 1e-6 * x)),
                -g * (1e-6 * y + 8e-6 * y),
            ]
        };
        let xy: Vec<[f64; 2]> = ElementIndex::iter(mesh.n_elements)
            .flat_map(|k| {
                let (mesh, ops) = (&mesh, &ops);
                (0..ops.n_nodes)
                    .map(move |i| mesh.reference_to_physical(k, ops.nodes_r[i], ops.nodes_s[i]))
            })
            .collect();
        let stress: Vec<[f64; 3]> = xy.iter().map(|p| stress_at(p[0], p[1])).collect();
        let force =
            WaveForce2D::from_radiation_stress(&mesh, &ops, &geom, &stress, g).with_ramp(100.0);
        for (p, f) in force.force().iter().enumerate() {
            let e = exact(xy[p][0], xy[p][1]);
            assert!(
                (f[0] - e[0]).abs() < 1e-14 && (f[1] - e[1]).abs() < 1e-14,
                "node {p}: {f:?} against {e:?}"
            );
        }
        // Half way up the ramp, and nothing on a dry node
        let mut q = crate::solver::SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        let k = ElementIndex::new(1);
        for i in 0..ops.n_nodes {
            let h = if i == 0 { 0.0 } else { 5.0 };
            q.set_state(k, i, SWEState2D::new(h, 0.0, 0.0));
        }
        let element = ElementSources {
            element: k,
            time: 50.0,
            solution: &q,
            mesh: &mesh,
            ops: &ops,
            bathymetry: None,
            g,
            h_min: 1e-6,
        };
        let n = ops.n_nodes;
        let (mut h, mut hu, mut hv) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        force.add_element(&element, &mut h, &mut hu, &mut hv);
        assert_eq!((hu[0], hv[0]), (0.0, 0.0));
        for i in 1..n {
            let f = force.force()[n + i];
            assert!((hu[i] - 0.5 * f[0]).abs() < 1e-15 && (hv[i] - 0.5 * f[1]).abs() < 1e-15);
        }
        assert!(h.iter().all(|&x| x == 0.0));
    }

    /// Between two exchanges the force is linear in time from the first wave
    /// state's to the second's, ramped, and constant beyond them.
    #[test]
    fn a_force_between_two_exchanges_is_linear_in_time() {
        let mesh = Mesh2D::uniform_rectangle(0.0, 300.0, 0.0, 200.0, 3, 2);
        let ops = DGOperators2D::new(2);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let g = 9.81;
        let n_total = mesh.n_elements * ops.n_nodes;
        let xs: Vec<f64> = ElementIndex::iter(mesh.n_elements)
            .flat_map(|k| {
                let (mesh, ops) = (&mesh, &ops);
                (0..ops.n_nodes)
                    .map(move |i| mesh.reference_to_physical(k, ops.nodes_r[i], ops.nodes_s[i])[0])
            })
            .collect();
        // S_xx linear in x: uniform forces −g·a
        let uniform = |a: f64| {
            let stress: Vec<[f64; 3]> = xs.iter().map(|x| [a * x, 0.0, 0.0]).collect();
            WaveForce2D::from_radiation_stress(&mesh, &ops, &geom, &stress, g)
        };
        let force = WaveForce2D::between(
            600.0,
            uniform(1e-4).with_ramp(1200.0),
            1200.0,
            &uniform(3e-4),
        );
        let mut q = crate::solver::SWESolution2D::new(mesh.n_elements, ops.n_nodes);
        for k in ElementIndex::iter(mesh.n_elements) {
            for i in 0..ops.n_nodes {
                q.set_state(k, i, SWEState2D::new(5.0, 0.0, 0.0));
            }
        }
        let n = ops.n_nodes;
        let k = ElementIndex::new(4);
        // The first state's before 600 s, the second's after 1200 s
        for (t, a) in [
            (0.0, 1e-4),
            (300.0, 1e-4),
            (600.0, 1e-4),
            (900.0, 2e-4),
            (1200.0, 3e-4),
            (5000.0, 3e-4),
        ] {
            let expected = tidal_ramp(t, Some(1200.0)) * a;
            let element = ElementSources {
                element: k,
                time: t,
                solution: &q,
                mesh: &mesh,
                ops: &ops,
                bathymetry: None,
                g,
                h_min: 1e-6,
            };
            let (mut h, mut hu, mut hv) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
            force.add_element(&element, &mut h, &mut hu, &mut hv);
            for i in 0..n {
                assert!(
                    (hu[i] + g * expected).abs() < 1e-13 && hv[i].abs() < 1e-13,
                    "t = {t}: {} against {}",
                    hu[i],
                    -g * expected
                );
            }
        }
        assert_eq!(force.force().len(), n_total);
    }

    /// The force is in flux form: for a stress that jumps between every pair
    /// of elements, its total `Σ w J F` is `−g ∮ S·n` over the boundary to
    /// round-off, on a sheared mesh with walls and on a periodic one (no
    /// boundary: no net force). Dropping the face terms (the element-local
    /// derivative) leaves a net force of the order of the jumps.
    #[test]
    fn the_force_conserves_momentum_across_jumps() {
        let g = 9.81;
        let ops = DGOperators2D::new(3);
        let mut sheared = Mesh2D::uniform_rectangle(0.0, 300.0, 0.0, 200.0, 4, 3);
        for v in sheared.vertices.iter_mut() {
            v[0] += 0.3 * v[1] + 1e-4 * v[1] * v[1];
        }
        let periodic = Mesh2D::uniform_periodic(0.0, 300.0, 0.0, 200.0, 4, 3);
        for mesh in [sheared, periodic] {
            let geom = GeometricFactors2D::compute(&mesh, &ops);
            let nn = ops.n_nodes;
            // Smooth within elements, discontinuous across them
            let stress: Vec<[f64; 3]> = (0..mesh.n_elements * nn)
                .map(|p| {
                    let (k, i) = (p / nn, p % nn);
                    let [x, y] = mesh.reference_to_physical(
                        ElementIndex::new(k),
                        ops.nodes_r[i],
                        ops.nodes_s[i],
                    );
                    let e = k as f64;
                    [
                        0.3 + 0.05 * (e * 1.7).sin() + 1e-3 * x,
                        0.1 * (e * 0.9).cos() + 2e-4 * y,
                        0.2 + 0.04 * (e * 2.3).sin() + 1e-6 * x * y,
                    ]
                })
                .collect();
            let force = WaveForce2D::from_radiation_stress(&mesh, &ops, &geom, &stress, g);
            let (mut total, mut size) = ([0.0; 2], 0.0);
            for k in 0..mesh.n_elements {
                for a in 0..nn {
                    let w = ops.weights[a] * geom.jacobian(k, a);
                    let f = force.force()[k * nn + a];
                    total[0] += w * f[0];
                    total[1] += w * f[1];
                    size += w * (f[0].abs() + f[1].abs());
                }
            }
            let mut flux = [0.0; 2];
            for k in 0..mesh.n_elements {
                for face in 0..4 {
                    if mesh.neighbor(ElementIndex::new(k), face).is_some() {
                        continue;
                    }
                    for fi in 0..ops.n_face_nodes {
                        let [sxx, sxy, syy] = stress[k * nn + ops.face_nodes[face][fi]];
                        let (nx, ny) = geom.normal(k, face, fi);
                        let w = ops.weights_1d[fi] * geom.surface_jacobian(k, face, fi);
                        flux[0] -= g * w * (sxx * nx + sxy * ny);
                        flux[1] -= g * w * (sxy * nx + syy * ny);
                    }
                }
            }
            // Against the element-local derivative alone
            let mut local = [0.0; 2];
            let mut gradient = vec![[0.0; 2]; stress.len()];
            for (c, rows) in [(0, [0, 1]), (1, [1, 2])] {
                for (d, &row) in rows.iter().enumerate() {
                    let field: Vec<f64> = stress.iter().map(|s| s[row]).collect();
                    crate::waves::model::nodal_gradient(&ops, &geom, &field, &mut gradient);
                    for k in 0..mesh.n_elements {
                        for a in 0..nn {
                            local[c] -=
                                g * ops.weights[a] * geom.jacobian(k, a) * gradient[k * nn + a][d];
                        }
                    }
                }
            }
            // Round-off of the sum of the force over the domain
            let scale = size;
            for c in 0..2 {
                assert!(
                    (total[c] - flux[c]).abs() < 1e-12 * scale,
                    "component {c}: {} against the boundary flux {}",
                    total[c],
                    flux[c]
                );
                assert!(
                    (local[c] - flux[c]).abs() > 1e-3 * scale,
                    "the element-local force conserves too ({} against {})",
                    local[c],
                    flux[c]
                );
            }
        }
    }

    /// Soulsby's enhancement: none without waves, 1 + 1.2·2^(−3.2) when the
    /// stresses are equal, 2.2 where the waves dominate; the friction law
    /// scales its inner rate by it, node by node.
    #[test]
    fn waves_enhance_the_bed_friction_as_soulsby_fits() {
        use crate::source::ChezyFriction2D;
        type Law = WaveCurrentFriction2D<ChezyFriction2D>;
        assert_eq!(Law::enhancement(0.3, 0.0), 1.0);
        assert!((Law::enhancement(0.3, 0.3) - (1.0 + 1.2 * 0.5f64.powf(3.2))).abs() < 1e-15);
        assert!((Law::enhancement(1e-9, 1.0) - 2.2).abs() < 1e-8);
        let (cd, h, speed) = (2.5e-3, 10.0, 0.4);
        let law = Law::new(ChezyFriction2D::new(cd), vec![0.0, 4e-4, 10.0]);
        let plain = cd * speed / h;
        let tau_c = cd * speed * speed;
        assert_eq!(law.damping_rate(0, h, speed), plain);
        let expected = plain * Law::enhancement(tau_c, 4e-4);
        assert!((law.damping_rate(1, h, speed) - expected).abs() < 1e-18);
        assert!((law.damping_rate(2, h, speed) / plain - 2.2).abs() < 1e-3);
        assert_eq!(law.n_total_nodes(), Some(3));
    }
}

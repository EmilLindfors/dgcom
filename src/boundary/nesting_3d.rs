//! 3D nesting: a parent model's velocity and tracer profiles at open
//! boundaries and in a relaxation band.
//!
//! The 2D module nests the depth mean (`η`, `ū`) through its open-boundary
//! condition and [`crate::boundary::NestingRelaxation2D`]. [`Nesting3D`] adds
//! what depends on the vertical structure, at the nodes of the open faces
//! tagged for nesting and within a band around them (ROMS: `u3d`/`v3d`,
//! `temp`/`salt` boundary conditions with `M3nudg`/`Tnudg` nudging zones):
//!
//! - **Boundary values.** Outside those faces the 3D kernels see the
//!   parent's columns ([`crate::solver::rhs::Exterior3D`]): the layer volume
//!   fluxes take the central average of the interior's and the parent's
//!   `H_z u` (corrected to the 2D boundary flux, as everywhere), and water
//!   flowing in brings the parent's velocity, temperature and salinity.
//!   Water flowing out takes the interior's.
//! - **Relaxation.** Within the band (weight 1 on the boundary, falling to 0
//!   at `width` with the band's [`SpongeProfile`]), the shear relaxes to the
//!   parent's with the rate `weight/τ_uv`,
//!
//!   ```text
//!       ∂u_l/∂t += γ_uv [(u_p,l − ⟨u_p⟩) − (u_l − ⟨u⟩)],
//!   ```
//!
//!   with `⟨·⟩` the depth mean, so the depth mean stays the 2D module's (and
//!   the relaxation adds nothing to the slow forcing), and the tracers relax
//!   to the parent's with `weight/τ_TS`, `∂(H_z C)/∂t += H_z γ_TS (C_p − C)`.
//!   The relaxation is explicit: keep `τ` well above the baroclinic step.
//!
//! The parent is any [`ParentColumns3D`]: it returns its profiles at the
//! child's layer centres for a node, a time and the child's column. A parent
//! model's output is [`crate::boundary::OceanModelColumns`]; without one,
//! [`ReferenceColumns`] relaxes to a fixed state (ROMS's nudging to
//! climatology), e.g. the initial stratification at rest.
//!
//! # Stratified open boundaries need the relaxation
//!
//! Without nesting, the 3D kernels extrapolate the velocity and the tracers
//! at open faces and see no pressure outside them. For stratified flow that
//! is unstable: an internal wave reaching such a boundary starts an exchange
//! flow that displaces the boundary column's isopycnals without bound (a
//! mode-1 pulse of 0.24 °C in a 50 m channel, `N = 0.02 s⁻¹`: 6 °C and
//! 0.26 m/s at the boundary within 8 h, TODO P4.2). Boundary values alone
//! (no band) are stable but reflect the wave completely; a band relaxing to
//! the state at rest lets it out, reflecting 18–32 % of its amplitude for
//! bands of 2–6 km and time scales of 30 min to 2 h at `c₁` ≈ 0.32 m/s.

use std::sync::{Arc, Mutex, MutexGuard};

use crate::boundary::NestingError;
use crate::boundary::band::boundary_distances;
use crate::mesh::{Bathymetry2D, BoundaryTag, Mesh2D};
use crate::operators::DGOperators2D;
use crate::solver::Blocks;
use crate::solver::rhs::{Exterior3D, ExteriorField};
use crate::solver::state::Solution3D;
use crate::source::SpongeProfile;
use crate::time::mode_split::StageTime;
use crate::types::ElementIndex;
use crate::vertical::SigmaGrid;

/// The child column a parent profile is wanted for.
#[derive(Clone, Copy, Debug)]
pub struct ColumnContext3D {
    /// Simulation time (s).
    pub time: f64,
    /// The node, `[element][node]`.
    pub node: usize,
    /// Mesh coordinates of the node (m).
    pub position: (f64, f64),
    /// Bed elevation `B` and free surface `η` of the child (m): the layer
    /// centres are at `z_l = η + σ_l (η − B)`.
    pub bed: f64,
    pub eta: f64,
}

/// A parent's profiles at the child's layer centres, one value per layer.
/// Only the fields the parent supplies ([`ParentColumns3D::supplies`]) are
/// read.
pub struct ParentColumn<'a> {
    /// Velocity in mesh axes (m/s).
    pub u: &'a mut [f64],
    pub v: &'a mut [f64],
    /// Temperature (°C) and salinity.
    pub temp: &'a mut [f64],
    pub salt: &'a mut [f64],
}

/// What a parent supplies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Supplied {
    pub velocity: bool,
    pub tracers: bool,
}

/// A parent model's water columns (see the module docs).
pub trait ParentColumns3D: Send + Sync {
    /// The fields [`Self::column`] writes.
    fn supplies(&self) -> Supplied;

    /// Write the parent's profiles at the layer centres of `sigma` in the
    /// child column `ctx` to `out`, and return `true`; `false` where the
    /// parent does not cover the column (the node then keeps its own values:
    /// extrapolation at the boundary, no relaxation).
    fn column(&self, ctx: &ColumnContext3D, sigma: &SigmaGrid, out: ParentColumn<'_>) -> bool;
}

/// A fixed reference state as the parent: the columns of a [`Solution3D`]
/// (on the child's own grid), the same at every time. For open boundaries
/// without a parent model: relax to the stratification at rest (see the
/// module docs).
pub struct ReferenceColumns {
    n_levels: usize,
    u: Vec<f64>,
    v: Vec<f64>,
    temp: Vec<f64>,
    salt: Vec<f64>,
}

impl ReferenceColumns {
    /// The velocity, temperature and salinity of `state`.
    pub fn from_state(state: &Solution3D) -> Self {
        Self {
            n_levels: state.n_levels,
            u: state.u.clone(),
            v: state.v.clone(),
            temp: state.temp.clone(),
            salt: state.salt.clone(),
        }
    }
}

impl ParentColumns3D for ReferenceColumns {
    fn supplies(&self) -> Supplied {
        Supplied {
            velocity: true,
            tracers: true,
        }
    }

    fn column(&self, ctx: &ColumnContext3D, sigma: &SigmaGrid, out: ParentColumn<'_>) -> bool {
        let nl = self.n_levels;
        assert_eq!(
            sigma.n_levels(),
            nl,
            "ReferenceColumns: layers of the reference state"
        );
        let column = ctx.node * nl..(ctx.node + 1) * nl;
        out.u.copy_from_slice(&self.u[column.clone()]);
        out.v.copy_from_slice(&self.v[column.clone()]);
        out.temp.copy_from_slice(&self.temp[column.clone()]);
        out.salt.copy_from_slice(&self.salt[column]);
        true
    }
}

/// The relaxation band of a [`Nesting3D`].
#[derive(Clone, Copy, Debug)]
pub struct NestingBand3D {
    /// Width of the band (m); 0 for boundary values only.
    pub width: f64,
    /// Shape of the weight across the band.
    pub profile: SpongeProfile,
    /// Relaxation time scale of the shear on the boundary (s), if relaxed.
    pub velocity_timescale: Option<f64>,
    /// Relaxation time scale of the tracers on the boundary (s), if relaxed.
    pub tracer_timescale: Option<f64>,
}

impl Default for NestingBand3D {
    /// Boundary values only, no relaxation.
    fn default() -> Self {
        Self {
            width: 0.0,
            profile: SpongeProfile::Cosine,
            velocity_timescale: None,
            tracer_timescale: None,
        }
    }
}

/// A parent's profiles at the open faces of some tags and in a band around
/// them (see the module docs).
pub struct Nesting3D {
    parent: Arc<dyn ParentColumns3D>,
    supplied: Supplied,
    tags: Vec<BoundaryTag>,
    n_levels: usize,
    /// `[element][node]` of every slot.
    nodes: Vec<usize>,
    positions: Vec<(f64, f64)>,
    /// Slot of every node, `u32::MAX` for none.
    slot_of_node: Vec<u32>,
    /// Relaxation rates of every slot (1/s).
    velocity_rate: Vec<f64>,
    tracer_rate: Vec<f64>,
    columns: Mutex<Columns>,
}

/// The parent's columns at every slot for one child state.
struct Columns {
    /// Per band slot, the time and child `η` its column was evaluated for.
    time: Vec<f64>,
    eta: Vec<f64>,
    /// Per element, whether the last [`Nesting3D::columns`] was for it.
    listed: Vec<bool>,
    /// Nodes per element.
    n_nodes: usize,
    u: Vec<f64>,
    v: Vec<f64>,
    temp: Vec<f64>,
    salt: Vec<f64>,
}

impl Nesting3D {
    /// Nest `parent` at the open faces tagged `tags` of `mesh` (with `ops`'s
    /// nodes and `n_levels` layers), relaxing within `band`.
    pub fn new(
        parent: Arc<dyn ParentColumns3D>,
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        n_levels: usize,
        tags: &[BoundaryTag],
        band: &NestingBand3D,
    ) -> Result<Self, NestingError> {
        let tag = *tags.first().expect("at least one boundary tag to nest");
        let distance =
            boundary_distances(mesh, ops, tags, band.width).ok_or(NestingError::NoBoundary(tag))?;
        let rate = |timescale: Option<f64>, weight: f64| {
            timescale.map_or(0.0, |tau| {
                assert!(
                    tau > 0.0,
                    "relaxation time scale must be positive, got {tau}"
                );
                weight / tau
            })
        };
        let n_nodes = ops.n_nodes;
        let mut slot_of_node = vec![u32::MAX; distance.len()];
        let (mut nodes, mut positions) = (Vec::new(), Vec::new());
        let (mut velocity_rate, mut tracer_rate) = (Vec::new(), Vec::new());
        for (flat, &d) in distance.iter().enumerate() {
            let weight = if d == 0.0 {
                1.0
            } else if d < band.width {
                band.profile.evaluate(1.0 - d / band.width)
            } else {
                continue;
            };
            let (k, i) = (flat / n_nodes, flat % n_nodes);
            let [x, y] =
                mesh.reference_to_physical(ElementIndex::new(k), ops.nodes_r[i], ops.nodes_s[i]);
            slot_of_node[flat] = nodes.len() as u32;
            nodes.push(flat);
            positions.push((x, y));
            velocity_rate.push(rate(band.velocity_timescale, weight));
            tracer_rate.push(rate(band.tracer_timescale, weight));
        }
        let n_values = nodes.len() * n_levels;
        let columns = Columns {
            time: vec![f64::NAN; nodes.len()],
            eta: vec![f64::NAN; nodes.len()],
            listed: Vec::new(),
            n_nodes: 1,
            u: vec![0.0; n_values],
            v: vec![0.0; n_values],
            temp: vec![0.0; n_values],
            salt: vec![0.0; n_values],
        };
        Ok(Self {
            supplied: parent.supplies(),
            parent,
            tags: tags.to_vec(),
            n_levels,
            nodes,
            positions,
            slot_of_node,
            velocity_rate,
            tracer_rate,
            columns: Mutex::new(columns),
        })
    }

    /// The nested open-boundary tags.
    pub fn tags(&self) -> &[BoundaryTag] {
        &self.tags
    }

    /// Number of nodes on the nested boundary or in its band.
    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// The parent's columns for `state` at the band nodes of `elements`,
    /// each at its element's stage time in `time`; a column is evaluated
    /// unless its last evaluation was for the same time and free surface.
    /// The relaxations then act on those elements only. A node the parent
    /// does not cover keeps the child's own column, its velocity at rest
    /// where the column is thinner than `min_column_depth` (as every 3D
    /// kernel sees a thin column; then it does not matter whether `state`
    /// holds the thin columns' velocities or has them masked).
    pub fn columns(
        &self,
        state: &Solution3D,
        bathymetry: &Bathymetry2D,
        sigma: &SigmaGrid,
        time: StageTime,
        elements: Blocks,
        min_column_depth: f64,
    ) -> NestingColumns<'_> {
        let mut columns = self.columns.lock().expect("Failed to lock nesting columns");
        {
            let nl = self.n_levels;
            let nn = state.n_nodes;
            let Columns {
                time: times,
                eta,
                u,
                v,
                temp,
                salt,
                listed,
                n_nodes,
            } = &mut *columns;
            *n_nodes = nn;
            listed.clear();
            listed.resize(state.n_elements, elements.is_all());
            if !elements.is_all() {
                elements.iter().for_each(|k| listed[k] = true);
            }
            let listed: &[bool] = listed;
            // Each band node's column on its own (a parent evaluation is a
            // few µs, ≈ 10⁴ band nodes on a coastline mesh)
            let column_of = |(slot, ((t_slot, eta), (u, (v, (temp, salt))))): BandColumn<'_>| {
                let flat = self.nodes[slot];
                if !listed[flat / nn] {
                    return;
                }
                let t = time.at(flat / nn);
                if *t_slot == t && *eta == state.eta.data[flat] {
                    return;
                }
                *t_slot = t;
                *eta = state.eta.data[flat];
                let covered = self.parent.column(
                    &ColumnContext3D {
                        time: t,
                        node: flat,
                        position: self.positions[slot],
                        bed: bathymetry.data[flat],
                        eta: *eta,
                    },
                    sigma,
                    ParentColumn {
                        u: &mut *u,
                        v: &mut *v,
                        temp: &mut *temp,
                        salt: &mut *salt,
                    },
                );
                if !covered {
                    let column = flat * nl..(flat + 1) * nl;
                    if *eta - bathymetry.data[flat] < min_column_depth {
                        u.fill(0.0);
                        v.fill(0.0);
                    } else {
                        u.copy_from_slice(&state.u[column.clone()]);
                        v.copy_from_slice(&state.v[column.clone()]);
                    }
                    temp.copy_from_slice(&state.temp[column.clone()]);
                    salt.copy_from_slice(&state.salt[column]);
                }
            };
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                times
                    .par_iter_mut()
                    .zip(eta.par_iter_mut())
                    .zip(
                        u.par_chunks_exact_mut(nl).zip(
                            v.par_chunks_exact_mut(nl).zip(
                                temp.par_chunks_exact_mut(nl)
                                    .zip(salt.par_chunks_exact_mut(nl)),
                            ),
                        ),
                    )
                    .enumerate()
                    .for_each(column_of);
            }
            #[cfg(not(feature = "parallel"))]
            times
                .iter_mut()
                .zip(eta.iter_mut())
                .zip(
                    u.chunks_exact_mut(nl).zip(
                        v.chunks_exact_mut(nl)
                            .zip(temp.chunks_exact_mut(nl).zip(salt.chunks_exact_mut(nl))),
                    ),
                )
                .enumerate()
                .for_each(column_of);
        }
        NestingColumns {
            nesting: self,
            columns,
        }
    }
}

/// A band node's slot and its cached `η` and `u`, `v`, `T`, `S` columns, as
/// [`Nesting3D::columns`] fills them.
type BandColumn<'a> = (
    usize,
    (
        (&'a mut f64, &'a mut f64),
        (
            &'a mut [f64],
            (&'a mut [f64], (&'a mut [f64], &'a mut [f64])),
        ),
    ),
);

/// The parent's columns of a [`Nesting3D`] for one child state.
pub struct NestingColumns<'a> {
    nesting: &'a Nesting3D,
    columns: MutexGuard<'a, Columns>,
}

impl NestingColumns<'_> {
    /// Whether node `flat`'s element was listed in [`Nesting3D::columns`].
    fn listed(&self, flat: usize) -> bool {
        self.columns.listed[flat / self.columns.n_nodes]
    }

    fn field<'b>(&'b self, values: &'b [f64]) -> ExteriorField<'b> {
        ExteriorField {
            tags: &self.nesting.tags,
            slot_of_node: &self.nesting.slot_of_node,
            n_levels: self.nesting.n_levels,
            values,
        }
    }

    /// The exterior values of the 3D kernels at the nested open faces.
    pub fn exterior(&self) -> Exterior3D<'_> {
        let supplied = self.nesting.supplied;
        let c = &self.columns;
        Exterior3D {
            velocity: supplied
                .velocity
                .then(|| [self.field(&c.u), self.field(&c.v)]),
            temp: supplied.tracers.then(|| self.field(&c.temp)),
            salt: supplied.tracers.then(|| self.field(&c.salt)),
        }
    }

    /// Add the relaxation of the shear to the velocity tendency `rhs`
    /// (`∂u/∂t`, layout of [`Solution3D`]) of `state`, except in the columns
    /// `skip` rejects (thin columns). Its depth mean is zero.
    pub fn relax_shear(
        &self,
        state: &Solution3D,
        sigma: &SigmaGrid,
        rhs: &mut Solution3D,
        skip: impl Fn(usize) -> bool,
    ) {
        if !self.nesting.supplied.velocity {
            return;
        }
        let nl = self.nesting.n_levels;
        for (slot, (&flat, &rate)) in self
            .nesting
            .nodes
            .iter()
            .zip(&self.nesting.velocity_rate)
            .enumerate()
        {
            if rate == 0.0 || !self.listed(flat) || skip(flat) {
                continue;
            }
            let parent = slot * nl..(slot + 1) * nl;
            let column = flat * nl..(flat + 1) * nl;
            for (parent, field, out) in [
                (
                    &self.columns.u[parent.clone()],
                    &state.u[column.clone()],
                    &mut rhs.u,
                ),
                (
                    &self.columns.v[parent],
                    &state.v[column.clone()],
                    &mut rhs.v,
                ),
            ] {
                let offset = sigma.depth_average(parent) - sigma.depth_average(field);
                for ((r, &p), &x) in out[column.clone()].iter_mut().zip(parent).zip(field) {
                    *r += rate * (p - x - offset);
                }
            }
        }
    }

    /// Add the relaxation of the tracers to their inventory tendencies
    /// `rhs.temp`, `rhs.salt` (`∂(H_z C)/∂t`) for `state` (concentrations),
    /// except in the columns `skip` rejects.
    pub fn relax_tracers(
        &self,
        state: &Solution3D,
        bathymetry: &Bathymetry2D,
        sigma: &SigmaGrid,
        rhs: &mut Solution3D,
        skip: impl Fn(usize) -> bool,
    ) {
        if !self.nesting.supplied.tracers {
            return;
        }
        let nl = self.nesting.n_levels;
        for (slot, (&flat, &rate)) in self
            .nesting
            .nodes
            .iter()
            .zip(&self.nesting.tracer_rate)
            .enumerate()
        {
            if rate == 0.0 || !self.listed(flat) || skip(flat) {
                continue;
            }
            let depth = state.eta.data[flat] - bathymetry.data[flat];
            let parent = slot * nl..(slot + 1) * nl;
            let column = flat * nl..(flat + 1) * nl;
            for (parent, field, out) in [
                (
                    &self.columns.temp[parent.clone()],
                    &state.temp[column.clone()],
                    &mut rhs.temp,
                ),
                (
                    &self.columns.salt[parent],
                    &state.salt[column.clone()],
                    &mut rhs.salt,
                ),
            ] {
                for (((r, &p), &c), &ds) in out[column.clone()]
                    .iter_mut()
                    .zip(parent)
                    .zip(field)
                    .zip(sigma.d_sigma())
                {
                    *r += depth * ds * rate * (p - c);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A parent with a linear shear and stratification, the same everywhere.
    struct Profile;

    impl ParentColumns3D for Profile {
        fn supplies(&self) -> Supplied {
            Supplied {
                velocity: true,
                tracers: true,
            }
        }

        fn column(&self, ctx: &ColumnContext3D, sigma: &SigmaGrid, out: ParentColumn<'_>) -> bool {
            for (l, &s) in sigma.sigma_rho().iter().enumerate() {
                out.u[l] = 0.2 + 0.1 * s + 1e-3 * ctx.time;
                out.v[l] = -0.1 * s + ctx.eta;
                out.temp[l] = 10.0 + 4.0 * s;
                out.salt[l] = 34.0 - s;
            }
            // Not covered beyond x = 500 m
            ctx.position.0 < 500.0
        }
    }

    fn setup(band: NestingBand3D) -> (Mesh2D, DGOperators2D, Nesting3D) {
        let mesh = Mesh2D::uniform_rectangle_with_sides(
            0.0,
            4000.0,
            0.0,
            1000.0,
            4,
            1,
            [
                BoundaryTag::Wall,
                BoundaryTag::Wall,
                BoundaryTag::Wall,
                BoundaryTag::Open,
            ],
        );
        let ops = DGOperators2D::new(2);
        let nesting = Nesting3D::new(
            Arc::new(Profile),
            &mesh,
            &ops,
            4,
            &[BoundaryTag::Open],
            &band,
        )
        .unwrap();
        (mesh, ops, nesting)
    }

    /// The band covers the boundary nodes and those within its width, the
    /// relaxation of the shear has no depth mean, the tracers relax as
    /// inventories, the exterior values apply only to the nested tag, and
    /// nodes the parent does not cover keep their own values.
    #[test]
    fn relaxation_keeps_the_depth_mean_and_weights_the_band() {
        let band = NestingBand3D {
            width: 1500.0,
            velocity_timescale: Some(3600.0),
            tracer_timescale: Some(7200.0),
            ..NestingBand3D::default()
        };
        let (mesh, ops, nesting) = setup(band);
        let sigma = SigmaGrid::uniform(4);
        let bathymetry = Bathymetry2D::constant(mesh.n_elements, ops.n_nodes, -20.0);
        let mut state = Solution3D::new(mesh.n_elements, ops.n_nodes, 4);
        state.temp.fill(12.0);
        state.salt.fill(33.0);
        state.eta.fill(0.5);
        let columns = nesting.columns(
            &state,
            &bathymetry,
            &sigma,
            StageTime::Uniform(100.0),
            Blocks::All(state.n_elements),
            0.0,
        );
        let mut rhs = Solution3D::new(mesh.n_elements, ops.n_nodes, 4);
        columns.relax_shear(&state, &sigma, &mut rhs, |_| false);
        columns.relax_tracers(&state, &bathymetry, &sigma, &mut rhs, |_| false);

        let mut relaxed = 0;
        for k in 0..mesh.n_elements {
            for i in 0..ops.n_nodes {
                let flat = k * ops.n_nodes + i;
                let [x, _] = mesh.reference_to_physical(
                    ElementIndex::new(k),
                    ops.nodes_r[i],
                    ops.nodes_s[i],
                );
                let column = &rhs.u[flat * 4..(flat + 1) * 4];
                assert!(sigma.depth_average(column).abs() < 1e-15);
                // Relaxed within the band where the parent covers the column
                let relaxed_here = x < 500.0;
                assert_eq!(column[0] != 0.0, relaxed_here, "x = {x}");
                if x == 0.0 {
                    // Weight 1: the shear's rate is (u_p − ⟨u_p⟩)/τ, the
                    // tracer's H_z (C_p − C)/τ
                    let s = sigma.sigma_rho()[0];
                    assert!((column[0] - 0.1 * (s + 0.5) / 3600.0).abs() < 1e-15);
                    let expected = 20.5 * 0.25 * (10.0 + 4.0 * s - 12.0) / 7200.0;
                    assert!((rhs.temp[flat * 4] - expected).abs() < 1e-14);
                    relaxed += 1;
                }
            }
        }
        assert_eq!(relaxed, ops.n_face_nodes);

        let exterior = columns.exterior();
        let [u, _] = exterior.velocity.unwrap();
        assert_eq!(
            u.at(BoundaryTag::Open, 0, 3),
            Some(0.2 + 0.1 * sigma.sigma_rho()[3] + 0.1)
        );
        assert_eq!(u.at(BoundaryTag::River, 0, 3), None);
        assert_eq!(
            u.at(BoundaryTag::Open, mesh.n_elements * ops.n_nodes - 1, 0),
            None
        );
    }

    /// The columns are evaluated again for a new time or a new free surface,
    /// not otherwise.
    #[test]
    fn columns_follow_the_time_and_the_free_surface() {
        let (mesh, ops, nesting) = setup(NestingBand3D::default());
        let sigma = SigmaGrid::uniform(4);
        let bathymetry = Bathymetry2D::constant(mesh.n_elements, ops.n_nodes, -20.0);
        let mut state = Solution3D::new(mesh.n_elements, ops.n_nodes, 4);
        let at = |state: &Solution3D, t: f64| {
            let columns = nesting.columns(
                state,
                &bathymetry,
                &sigma,
                StageTime::Uniform(t),
                Blocks::All(state.n_elements),
                0.0,
            );
            let [u, v] = columns.exterior().velocity.unwrap();
            (
                u.at(BoundaryTag::Open, 0, 0).unwrap(),
                v.at(BoundaryTag::Open, 0, 0).unwrap(),
            )
        };
        let (u0, v0) = at(&state, 0.0);
        let (u1, v1) = at(&state, 10.0);
        assert!((u1 - u0 - 1e-2).abs() < 1e-15 && v1 == v0);
        state.eta.fill(0.1);
        // Same time, other η: evaluated again
        let (u2, v2) = at(&state, 10.0);
        assert!(u2 == u1 && (v2 - v1 - 0.1).abs() < 1e-15);
        assert_eq!(nesting.n_nodes(), ops.n_face_nodes);
    }
}

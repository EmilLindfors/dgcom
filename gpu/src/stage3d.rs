//! The 3D stage-values gate (P2.7 Phase 4.3a, `notes/gpu.md`): what the mode
//! splitter does to a stage's inventories to make the values the next
//! stage's RHS reads, with `Hydrostatic3D`'s hooks there but the tracer
//! limiter ([`StageHooks3D`]), on the GPU against dgcom on the CPU composed in
//! `ModeSplitIntegrator::step`'s order, value for value:
//!
//! - the step's start: the merged cells' groups to their means
//!   (`MergedCells::start`) and their density, the first stage's values
//!   masked (`StageStash::mask`), the inventories (`to_inventory`), the
//!   flushed elements' values passed through (`MergedCells::pass_through`);
//! - a stage from another buffer's values (the slots of local time
//!   stepping: `from_inventory_after` with the last values and their
//!   stash's velocities, the groups, the flushed elements), the density,
//!   T and S back to inventories (`inventory_of`), the dry nodes'
//!   inventories (`dry_inventories`), the mask;
//! - a stage in place (global steps), the step's last (`end_density`);
//! - the step's end: the mask undone (`restore_into`), the values of its
//!   inventories (`from_inventory` at the `end`);
//! - the thin columns' momentum tendencies (`zero_thin_momentum`);
//!
//! on walled beaches whose stages are long enough that thin cells merge
//! with their neighbours and films are flushed (`MergedCells::find` on the
//! host, its groups uploaded). Mutants it must catch, and timed at Chiloé's
//! size (96,000 P2 elements × 20 levels).

use std::sync::Arc;
use std::time::Instant;

use cuda_core::DeviceBuffer;
use dgcom::physics::{EquationOfState, LinearEOS, zero_thin_momentum};
use dgcom::solver::Blocks;
use dgcom::solver::rhs::transport_3d::{dry_inventories, from_inventory_after, inventory_of};
use dgcom::solver::rhs::{
    MergedCells, MergedTables, StepWater, from_inventory, to_inventory, w_cell_thicknesses,
};
use dgcom::solver::state::Solution3D;
use dgcom::time::StageStash;
use dgcom::vertical::depth_average;

use crate::hooks::{Merged3D, N_FIELDS, StageHooks3D, StageValues3D, State3D};
use crate::swe2d::{Gpu, NN, ORDER};
use crate::three_d::Statics3D;
use crate::transport3d::{DT, Fixture, LEVELS, Shape, agreement};

/// What the gate compares, by name, at every point of the pipeline.
type Snapshots = Vec<(String, Vec<f64>)>;

/// The inventory fields' names, in the splitter's order.
const NAMES: [&str; N_FIELDS] = ["u", "v", "T", "S", "k", "psi"];

fn field(s: &Solution3D, which: usize) -> &Vec<f64> {
    match which {
        0 => &s.u,
        1 => &s.v,
        2 => &s.temp,
        3 => &s.salt,
        4 => &s.tke,
        _ => &s.gls,
    }
}

fn field_mut(s: &mut Solution3D, which: usize) -> &mut Vec<f64> {
    match which {
        0 => &mut s.u,
        1 => &mut s.v,
        2 => &mut s.temp,
        3 => &mut s.salt,
        4 => &mut s.tke,
        _ => &mut s.gls,
    }
}

/// `s`'s inventory fields `which` (with the density and `η`, `ū`, `v̄` if
/// `all`) into `out` as `at`'s.
fn take(out: &mut Snapshots, at: &str, s: &Solution3D, which: std::ops::Range<usize>, all: bool) {
    for w in which {
        out.push((format!("{at} {}", NAMES[w]), field(s, w).clone()));
    }
    if all {
        out.push((format!("{at} rho"), s.rho.clone()));
        out.push((format!("{at} eta"), s.eta.data.clone()));
        out.push((format!("{at} ubar"), s.ubar.data.clone()));
        out.push((format!("{at} vbar"), s.vbar.data.clone()));
    }
}

/// The step's start of `fx`: its state, the thin columns carrying their
/// depth means (made up there, the layers' means elsewhere), and the density
/// of an earlier step's end (not of its T and S).
fn start_state(fx: &Fixture) -> Solution3D {
    let mut state = fx.state.clone();
    let (nl, d_sigma) = (LEVELS, fx.sigma.d_sigma());
    for idx in 0..state.eta.data.len() {
        let column = idx * nl..(idx + 1) * nl;
        let depth = state.eta.data[idx] - fx.bathymetry.data[idx];
        if depth < fx.min_column_depth {
            let x = idx as f64;
            let (ub, vb) = (0.3 * (0.37 * x).cos(), 0.2 * (0.53 * x).sin());
            state.ubar.data[idx] = ub;
            state.vbar.data[idx] = vb;
            state.u[column.clone()].fill(ub);
            state.v[column].fill(vb);
        } else {
            state.ubar.data[idx] = depth_average(&state.u[column.clone()], d_sigma);
            state.vbar.data[idx] = depth_average(&state.v[column], d_sigma);
        }
    }
    for (j, r) in state.rho.iter_mut().enumerate() {
        *r += 1e-3 * (0.29 * j as f64).sin();
    }
    state
}

/// A stage's inventories from the ones before (`inv`), as a stage's
/// combination moves them: every value by up to 1 %, `η` by a quarter to
/// three quarters of the step's rise, `ū`, `v̄` by up to 1 cm/s (`seed`
/// varies the pattern).
fn combine(inv: &mut Solution3D, eta_rate: &[f64], seed: f64) {
    for which in 0..N_FIELDS {
        for (t, x) in field_mut(inv, which).iter_mut().enumerate() {
            *x *= 1.0 + 0.01 * (0.731 * t as f64 + seed + which as f64).sin();
        }
    }
    for (idx, e) in inv.eta.data.iter_mut().enumerate() {
        *e += 0.5 * DT * eta_rate[idx] * (1.0 + 0.5 * (seed + 0.17 * idx as f64).sin());
    }
    for (idx, (u, v)) in inv
        .ubar
        .data
        .iter_mut()
        .zip(inv.vbar.data.iter_mut())
        .enumerate()
    {
        *u += 0.01 * (seed + 0.41 * idx as f64).sin();
        *v += 0.01 * (seed + 0.23 * idx as f64).cos();
    }
}

/// A momentum tendency of every layer of `like`'s shape.
fn tendencies(like: &Solution3D) -> Solution3D {
    let mut rhs = like.clone();
    for (t, (u, v)) in rhs.u.iter_mut().zip(rhs.v.iter_mut()).enumerate() {
        *u = 1e-3 * (0.13 * t as f64).sin();
        *v = 1e-3 * (0.17 * t as f64).cos();
    }
    rhs
}

/// The CPU's side: dgcom's functions in the splitter's order.
struct Cpu<'a> {
    fx: &'a Fixture,
    merged: &'a MergedCells,
    eos: LinearEOS,
    d_sigma_w: Vec<f64>,
}

impl Cpu<'_> {
    fn all(&self) -> Blocks<'static> {
        Blocks::All(self.fx.n_elements())
    }

    fn cells_of(&self, which: usize) -> &[f64] {
        if which < 4 {
            self.fx.sigma.d_sigma()
        } else {
            &self.d_sigma_w
        }
    }

    fn merged_of(&self, which: usize) -> Option<&MergedCells> {
        (which >= 2).then_some(self.merged)
    }

    fn thin(&self) -> impl Fn(usize, f64) -> bool + Sync + Send + '_ {
        let (bed, thin) = (&self.fx.bathymetry.data, self.fx.min_column_depth);
        move |idx, eta| eta - bed[idx] < thin
    }

    /// The splitter's `values_of` of `inv` into `out` (from `from`'s last
    /// values and stash, or in place with `stash`, `out`'s own).
    fn values_of(
        &self,
        inv: &Solution3D,
        out: &mut Solution3D,
        stash: &StageStash,
        from: Option<(&Solution3D, &StageStash)>,
    ) {
        let (fx, all) = (self.fx, self.all());
        match from {
            None => {
                stash.restore_into(out, all);
                self.copy_barotropic(inv, out);
                self.to_values(inv, out, false);
            }
            Some((from, from_stash)) => {
                for which in 0..N_FIELDS {
                    let q = field(inv, which);
                    if q.is_empty() {
                        continue;
                    }
                    let last = field(from, which);
                    let len = last.len() / fx.n_elements();
                    from_inventory_after(
                        q,
                        &inv.eta.data,
                        self.cells_of(which),
                        &fx.bathymetry,
                        &fx.geom,
                        self.merged_of(which),
                        field_mut(out, which),
                        all,
                        false,
                        |k, out_k| {
                            out_k.copy_from_slice(&last[k * len..(k + 1) * len]);
                            from_stash.restore_velocity(which, k, NN, out_k);
                        },
                    );
                }
                self.copy_barotropic(inv, out);
            }
        }
    }

    /// The splitter's `to_values`.
    fn to_values(&self, inv: &Solution3D, out: &mut Solution3D, end: bool) {
        let fx = self.fx;
        for which in 0..N_FIELDS {
            let q = field(inv, which);
            if !q.is_empty() {
                from_inventory(
                    q,
                    &inv.eta.data,
                    self.cells_of(which),
                    &fx.bathymetry,
                    &fx.geom,
                    self.merged_of(which),
                    field_mut(out, which),
                    self.all(),
                    end,
                );
            }
        }
    }

    fn copy_barotropic(&self, from: &Solution3D, to: &mut Solution3D) {
        to.eta.data.copy_from_slice(&from.eta.data);
        to.ubar.data.copy_from_slice(&from.ubar.data);
        to.vbar.data.copy_from_slice(&from.vbar.data);
    }

    /// The splitter's `stage_values` with `Hydrostatic3D`'s hooks (no
    /// tracer limiter: `post_stage` and `end_density` are the density).
    fn stage_values(
        &self,
        inv: &mut Solution3D,
        out: &mut Solution3D,
        stash: &mut StageStash,
        from: Option<(&Solution3D, &StageStash)>,
    ) {
        let (fx, all) = (self.fx, self.all());
        self.values_of(inv, out, stash, from);
        self.eos.update_density_on(out, all);
        let d_sigma = fx.sigma.d_sigma();
        for (q, c) in [(&mut inv.temp, &out.temp), (&mut inv.salt, &out.salt)] {
            let merged = Some(self.merged);
            inventory_of(q, c, &inv.eta.data, d_sigma, &fx.bathymetry, merged, all);
        }
        let Solution3D {
            eta,
            u,
            v,
            tke,
            gls,
            ..
        } = inv;
        for (which, q, c) in [
            (0, u, &out.u),
            (1, v, &out.v),
            (4, tke, &out.tke),
            (5, gls, &out.gls),
        ] {
            if !q.is_empty() {
                let (cells, merged) = (self.cells_of(which), self.merged_of(which));
                dry_inventories(q, c, &eta.data, cells, &fx.bathymetry, merged, all);
            }
        }
        stash.forget(all);
        stash.mask(out, all, self.thin());
    }

    /// The whole pipeline from the step's start `start`.
    fn run(&self, start: &Solution3D) -> Snapshots {
        let (fx, all) = (self.fx, self.all());
        let mut out = Snapshots::new();
        let mut state = start.clone();
        // The step's start: the groups to their means, their density
        if !self.merged.elements().is_empty() {
            let Solution3D {
                eta,
                temp,
                salt,
                tke,
                gls,
                ..
            } = &mut state;
            for (which, f) in [(2, temp), (3, salt), (4, tke), (5, gls)] {
                if !f.is_empty() {
                    let cells = self.cells_of(which);
                    self.merged
                        .start(f, cells, &eta.data, &fx.bathymetry, &fx.geom);
                }
            }
            let list = Blocks::list(fx.n_elements(), self.merged.elements());
            self.eos.update_density_on(&mut state, list);
        }
        take(&mut out, "start", &state, 2..N_FIELDS, true);
        // The first stage's values, masked; the inventories; the flushed
        // elements' values passed through
        let mut v0 = state.clone();
        let mut stash0 = StageStash::default();
        stash0.forget(all);
        stash0.mask(&mut v0, all, self.thin());
        for which in 0..N_FIELDS {
            let f = field_mut(&mut state, which);
            if !f.is_empty() {
                to_inventory(
                    f,
                    &start.eta.data,
                    self.cells_of(which),
                    &fx.bathymetry,
                    all,
                );
            }
        }
        for which in 2..N_FIELDS {
            let q = field(&state, which);
            if !q.is_empty() {
                self.merged.pass_through(
                    q,
                    &state.eta.data,
                    self.cells_of(which),
                    &fx.bathymetry,
                    &fx.geom,
                    field_mut(&mut v0, which),
                    all,
                    false,
                );
            }
        }
        take(&mut out, "first values", &v0, 0..N_FIELDS, true);
        out.push(("first values masked".into(), masked(&stash0, fx)));
        take(&mut out, "inventories", &state, 0..N_FIELDS, false);
        // A stage from another buffer's values
        let mut inv = state;
        combine(&mut inv, &fx.eta_rate, 1.0);
        let mut v1 = start.clone();
        let mut stash1 = StageStash::default();
        self.stage_values(&mut inv, &mut v1, &mut stash1, Some((&v0, &stash0)));
        take(&mut out, "stage values", &v1, 0..N_FIELDS, true);
        out.push(("stage values masked".into(), masked(&stash1, fx)));
        take(&mut out, "stage inventories", &inv, 0..N_FIELDS, false);
        // A stage in place, the step's last
        combine(&mut inv, &fx.eta_rate, 2.0);
        let mut stash = std::mem::take(&mut stash1);
        self.stage_values(&mut inv, &mut v1, &mut stash, None);
        take(&mut out, "in place values", &v1, 0..N_FIELDS, true);
        out.push(("in place values masked".into(), masked(&stash, fx)));
        take(&mut out, "in place inventories", &inv, 0..N_FIELDS, false);
        // The step's end: unmasked, the values of its inventories
        stash.restore_into(&mut v1, all);
        stash.forget(all);
        combine(&mut inv, &fx.eta_rate, 3.0);
        self.to_values(&inv, &mut v1, true);
        take(&mut out, "end values", &v1, 0..N_FIELDS, false);
        // The thin columns' momentum tendencies
        let mut rhs = tendencies(&inv);
        let bed = &fx.bathymetry.data;
        zero_thin_momentum(&mut rhs, &inv.eta.data, bed, fx.min_column_depth, all);
        take(&mut out, "zero thin", &rhs, 0..2, false);
        out
    }
}

/// `stash`'s masked columns as numbers.
fn masked(stash: &StageStash, fx: &Fixture) -> Vec<f64> {
    (0..fx.n_elements() * NN)
        .map(|idx| f64::from(u8::from(stash.is_masked(idx))))
        .collect()
}

/// The device's mutations of the pipeline (none: the gate's own run).
#[derive(Clone, Copy, Default)]
struct Mutations {
    /// The density of every element at the step's start
    density_everywhere: bool,
    /// The first stage's last values without its stash's velocities
    no_stash: bool,
}

/// The device's side.
struct Device<'a> {
    gpu: &'a Arc<Gpu>,
    fx: &'a Fixture,
    statics: &'a Statics3D,
    hooks: &'a StageHooks3D,
}

impl Device<'_> {
    fn download(
        &self,
        s: &State3D,
        like: &Solution3D,
    ) -> Result<Solution3D, Box<dyn std::error::Error>> {
        let mut host = like.clone();
        s.download_into(self.gpu, &mut host)?;
        Ok(host)
    }

    fn masked(&self, v: &StageValues3D) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        let flags = v.stash.masked.to_host_vec(&self.gpu.stream)?;
        Ok(flags.into_iter().map(f64::from).collect())
    }

    /// The whole pipeline from `start` with the groups `merged`.
    fn run(
        &self,
        merged: &Merged3D,
        start: &Solution3D,
        mutations: Mutations,
    ) -> Result<Snapshots, Box<dyn std::error::Error>> {
        let (gpu, s, hooks, fx) = (self.gpu, self.statics, self.hooks, self.fx);
        let mut out = Snapshots::new();
        let mut state = State3D::upload(gpu, start)?;
        hooks.start(s, merged, &mut state)?;
        if mutations.density_everywhere {
            hooks.density(s, &mut state, None)?;
        }
        take(
            &mut out,
            "start",
            &self.download(&state, start)?,
            2..N_FIELDS,
            true,
        );
        let mut v0 = hooks.values_like(s, start)?;
        v0.values.copy_from(gpu, &state)?;
        hooks.mask(s, &mut v0)?;
        hooks.to_inventories(s, &mut state)?;
        for which in 2..N_FIELDS {
            let q = state.fields()[which];
            let targets = v0.values.fields_mut();
            hooks.pass_through(s, merged, which, q, &state.eta, &mut *targets[which], false)?;
        }
        take(
            &mut out,
            "first values",
            &self.download(&v0.values, start)?,
            0..N_FIELDS,
            true,
        );
        out.push(("first values masked".into(), self.masked(&v0)?));
        let host = self.download(&state, start)?;
        take(&mut out, "inventories", &host, 0..N_FIELDS, false);
        // A stage from another buffer's values
        let mut inv_host = host;
        combine(&mut inv_host, &fx.eta_rate, 1.0);
        let mut inv = State3D::upload(gpu, &inv_host)?;
        let mut v1 = hooks.values_like(s, start)?;
        if mutations.no_stash {
            v0.stash.forget(gpu)?;
        }
        hooks.stage_values(s, merged, &mut inv, &mut v1, Some(&v0), false)?;
        take(
            &mut out,
            "stage values",
            &self.download(&v1.values, start)?,
            0..N_FIELDS,
            true,
        );
        out.push(("stage values masked".into(), self.masked(&v1)?));
        let mut inv_host = self.download(&inv, start)?;
        take(&mut out, "stage inventories", &inv_host, 0..N_FIELDS, false);
        // A stage in place, the step's last
        combine(&mut inv_host, &fx.eta_rate, 2.0);
        let mut inv = State3D::upload(gpu, &inv_host)?;
        hooks.stage_values(s, merged, &mut inv, &mut v1, None, true)?;
        take(
            &mut out,
            "in place values",
            &self.download(&v1.values, start)?,
            0..N_FIELDS,
            true,
        );
        out.push(("in place values masked".into(), self.masked(&v1)?));
        let mut inv_host = self.download(&inv, start)?;
        take(
            &mut out,
            "in place inventories",
            &inv_host,
            0..N_FIELDS,
            false,
        );
        // The step's end
        hooks.restore(s, &mut v1)?;
        v1.stash.forget(gpu)?;
        combine(&mut inv_host, &fx.eta_rate, 3.0);
        let inv = State3D::upload(gpu, &inv_host)?;
        hooks.to_values(s, merged, &inv, &mut v1, None, true)?;
        take(
            &mut out,
            "end values",
            &self.download(&v1.values, start)?,
            0..N_FIELDS,
            false,
        );
        // The thin columns' momentum tendencies
        let rhs = tendencies(&inv_host);
        let (mut rhs_u, mut rhs_v) = (
            DeviceBuffer::from_host(&gpu.stream, &rhs.u)?,
            DeviceBuffer::from_host(&gpu.stream, &rhs.v)?,
        );
        hooks.zero_thin_momentum(s, &inv.eta, &mut rhs_u, &mut rhs_v)?;
        out.push(("zero thin u".into(), rhs_u.to_host_vec(&gpu.stream)?));
        out.push(("zero thin v".into(), rhs_v.to_host_vec(&gpu.stream)?));
        Ok(out)
    }
}

/// `device`'s snapshots against `cpu`'s, a line per stage of the pipeline
/// (and per value that differs); whether all are the same to the bit.
fn report(cpu: &Snapshots, device: &Snapshots) -> bool {
    assert_eq!(cpu.len(), device.len(), "the snapshots");
    let mut same = true;
    let mut group: Option<(String, usize, usize, usize)> = None;
    let flush = |g: &Option<(String, usize, usize, usize)>| {
        if let Some((name, fields, identical, len)) = g {
            println!("    {name:<22} {fields} fields, {identical} of {len} bit-identical");
        }
    };
    for ((name, c), (device_name, g)) in cpu.iter().zip(device) {
        assert_eq!(name, device_name, "the snapshots' order");
        let a = agreement(c, g);
        let stage = name
            .rsplit_once(' ')
            .map_or(name.as_str(), |(stage, _)| stage);
        match &mut group {
            Some((current, fields, identical, len)) if current == stage => {
                *fields += 1;
                *identical += a.identical;
                *len += a.len;
            }
            _ => {
                flush(&group);
                group = Some((stage.to_string(), 1, a.identical, a.len));
            }
        }
        if a.identical < a.len {
            same = false;
            let at = (0..a.len)
                .find(|&n| c[n].to_bits() != g[n].to_bits())
                .expect("a difference");
            println!(
                "    {name:<22} {} of {} differ, by up to {:.1e} of the field's largest; first \
                 at {at}: CPU {:e}, GPU {:e}",
                a.len - a.identical,
                a.len,
                a.worst_rel,
                c[at],
                g[at]
            );
        }
    }
    flush(&group);
    same
}

/// How many of `cpu`'s values `device`'s differ from, and where most; prints
/// the mutant's line. Whether the gate catches it.
fn report_mutant(what: &str, cpu: &Snapshots, device: &Snapshots) -> bool {
    let (mut total, mut worst) = (0, (0, ""));
    for ((name, c), (_, g)) in cpu.iter().zip(device) {
        let a = agreement(c, g);
        let differ = a.len - a.identical;
        total += differ;
        if differ > worst.0 {
            worst = (differ, name.as_str());
        }
    }
    let caught = total > 0;
    if caught {
        println!(
            "    mutant ({what}): {total} values differ, most in {} ({}), caught",
            worst.1, worst.0
        );
    } else {
        println!("    mutant ({what}): NOT CAUGHT");
    }
    caught
}

/// The groups `tables` with what `edit` changes, uploaded.
fn mutated(
    gpu: &Gpu,
    tables: MergedTables,
    edit: impl FnOnce(&mut MergedTables, &mut Vec<bool>),
) -> Result<Merged3D, Box<dyn std::error::Error>> {
    let mut flushed = tables.flushed.to_vec();
    let mut edited = tables;
    edit(&mut edited, &mut flushed);
    Merged3D::upload(
        gpu,
        MergedTables {
            flushed: &flushed,
            ..edited
        },
    )
}

/// The step's water of `fx`: its pass, and a column source no flux carried.
fn find_groups(fx: &Fixture, start: &Solution3D) -> MergedCells {
    let ne = fx.n_elements();
    let source: Vec<f64> = (0..ne * NN)
        .map(|idx| 2e-5 * (0.61 * idx as f64).sin())
        .collect();
    let mut merged = MergedCells::new(ne, NN);
    merged.find(&StepWater {
        mesh: &fx.mesh,
        ops: &fx.ops,
        geom: &fx.geom,
        bathymetry: &fx.bathymetry,
        eta: &start.eta.data,
        eta_rate: &fx.eta_rate,
        dt: DT,
        euler_dt: &fx.euler_dt,
        face: &fx.bar_face,
        subcell: &fx.subcell,
        source: &source,
        thin_depth: fx.min_column_depth,
    });
    merged
}

/// One fixture on both, and the mutants. Whether all are the same to the
/// bit (and every mutant caught).
fn gate(gpu: &Arc<Gpu>, label: &str, shape: Shape) -> Result<bool, Box<dyn std::error::Error>> {
    let fx = Fixture::new(shape);
    let ne = fx.n_elements();
    let start = start_state(&fx);
    let merged = find_groups(&fx, &start);
    let tables = merged.tables();
    let joined = tables.joined.iter().filter(|&&j| j).count();
    let groups = Merged3D::upload(gpu, tables)?;
    println!(
        "{label}: {ne} P{ORDER} elements × {LEVELS} levels, forward-Euler steps of {} s: {} \
         elements with merged cells ({joined} cells), {} flushed ({} inflows, {} waves)",
        shape.euler_dt,
        tables.elements.len(),
        tables.flushed_elements.len(),
        tables.inflows.len(),
        groups.n_waves()
    );
    let mut d_sigma_w = vec![0.0; LEVELS + 1];
    w_cell_thicknesses(fx.sigma.d_sigma(), &mut d_sigma_w);
    let cpu = Cpu {
        fx: &fx,
        merged: &merged,
        eos: LinearEOS::default(),
        d_sigma_w,
    }
    .run(&start);
    let statics = fx.statics(gpu)?;
    let hooks = StageHooks3D::new(gpu, LinearEOS::default(), fx.min_column_depth)?;
    let device = Device {
        gpu,
        fx: &fx,
        statics: &statics,
        hooks: &hooks,
    };
    let mut same = report(&cpu, &device.run(&groups, &start, Mutations::default())?);
    // Every part of the pipeline must act: the groups, the flushed elements,
    // the mask
    let mut needed = tables.elements.is_empty() || tables.flushed_elements.is_empty();
    needed |= !cpu
        .iter()
        .any(|(name, v)| name == "first values masked" && v.contains(&1.0));
    if needed {
        println!("    the fixture lacks merged cells, flushed elements or thin columns");
        same = false;
    }

    // The mutants: the gate must catch each
    let none = Merged3D::none(gpu, &statics)?;
    let mut reversed = Merged3D::upload(gpu, tables)?;
    reversed.reverse_waves(gpu)?;
    let runs = [
        ("no merged cells", none, Mutations::default()),
        (
            "the flushed elements' waves in reverse",
            reversed,
            Mutations::default(),
        ),
        (
            "the flushed elements not passed through",
            mutated(gpu, tables, |t, _| {
                t.flushed_elements = &[];
                t.inflows = &[];
            })?,
            Mutations::default(),
        ),
        (
            "the flushed elements as groups, their inventories from their values",
            mutated(gpu, tables, |_, flushed| flushed.fill(false))?,
            Mutations::default(),
        ),
        (
            "the last values without the stash's velocities",
            Merged3D::upload(gpu, tables)?,
            Mutations {
                no_stash: true,
                ..Mutations::default()
            },
        ),
        (
            "the step's start's density on every element",
            Merged3D::upload(gpu, tables)?,
            Mutations {
                density_everywhere: true,
                ..Mutations::default()
            },
        ),
    ];
    for (what, groups, mutations) in runs {
        same &= report_mutant(what, &cpu, &device.run(&groups, &start, mutations)?);
    }
    println!(
        "  {}",
        if same {
            "bit for bit"
        } else {
            "NOT bit for bit"
        }
    );
    Ok(same)
}

/// Time a stage's values on the CPU and the GPU (`stage_values` from another
/// buffer's values, the groups of `shape`'s step), per element and level.
fn time(
    gpu: &Arc<Gpu>,
    label: &str,
    shape: Shape,
    iterations: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let fx = Fixture::new(shape);
    let ne = fx.n_elements();
    let points = (ne * LEVELS) as f64;
    let start = start_state(&fx);
    let merged = find_groups(&fx, &start);
    let mut d_sigma_w = vec![0.0; LEVELS + 1];
    w_cell_thicknesses(fx.sigma.d_sigma(), &mut d_sigma_w);
    let cpu = Cpu {
        fx: &fx,
        merged: &merged,
        eos: LinearEOS::default(),
        d_sigma_w,
    };
    let all = Blocks::All(ne);
    // The first stage's values and the inventories, as the step's start
    // makes them
    let mut v0 = start.clone();
    let mut stash0 = StageStash::default();
    stash0.mask(&mut v0, all, cpu.thin());
    let mut inv = start.clone();
    for which in 0..N_FIELDS {
        let f = field_mut(&mut inv, which);
        to_inventory(f, &start.eta.data, cpu.cells_of(which), &fx.bathymetry, all);
    }
    combine(&mut inv, &fx.eta_rate, 1.0);
    let inv_start = inv.clone();
    let mut v1 = start.clone();
    let mut stash1 = StageStash::default();
    let mut cpu_time = 0.0;
    for it in 0..iterations + 1 {
        let t0 = Instant::now();
        cpu.stage_values(&mut inv, &mut v1, &mut stash1, Some((&v0, &stash0)));
        if it >= 1 {
            cpu_time += t0.elapsed().as_secs_f64();
        }
    }
    drop(v1);
    let statics = fx.statics(gpu)?;
    let hooks = StageHooks3D::new(gpu, LinearEOS::default(), fx.min_column_depth)?;
    let groups = Merged3D::upload(gpu, merged.tables())?;
    println!(
        "timing, {label}: {ne} P{ORDER} elements × {LEVELS} levels ({:.1} M points), {} elements \
         with merged cells, {} flushed ({} waves), {iterations} stages",
        (ne * NN * LEVELS) as f64 / 1e6,
        merged.elements().len(),
        merged.tables().flushed_elements.len(),
        groups.n_waves()
    );
    let mut d_v0 = hooks.values_like(&statics, &start)?;
    hooks.mask(&statics, &mut d_v0)?;
    let mut d_inv = State3D::upload(gpu, &inv_start)?;
    let mut d_v1 = hooks.values_like(&statics, &start)?;
    let gpu_iterations = 5 * iterations;
    gpu.stream.synchronize()?;
    let mut gpu_time = 0.0;
    for it in 0..gpu_iterations + 2 {
        let t0 = Instant::now();
        hooks.stage_values(&statics, &groups, &mut d_inv, &mut d_v1, Some(&d_v0), false)?;
        gpu.stream.synchronize()?;
        if it >= 2 {
            gpu_time += t0.elapsed().as_secs_f64();
        }
    }
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let (c, g) = (
        cpu_time * 1e9 / (iterations as f64 * points),
        gpu_time * 1e9 / (gpu_iterations as f64 * points),
    );
    println!(
        "  a stage's values, per element and level: CPU ({threads} threads) {c:.2} ns | GPU \
         {g:.2} ns ({:.1}×); per stage {:.1} ms | {:.2} ms",
        c / g,
        c * points / 1e6,
        g * points / 1e6
    );
    Ok(())
}

pub fn run(gpu: &Arc<Gpu>) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "== The 3D stage values: conversions, merged cells, the mask, the density (Phase 4.3a) =="
    );
    let long = Shape {
        euler_dt: 40.0,
        ..Shape::BEACH
    };
    let cases = [
        ("beach, parallelograms", long),
        (
            "beach, jittered, stretched",
            Shape {
                jitter: 0.2,
                stretched: true,
                ..long
            },
        ),
    ];
    let mut failed = Vec::new();
    for (label, shape) in cases {
        if !gate(gpu, label, shape)? {
            failed.push(label);
        }
    }
    time(
        gpu,
        "beach",
        Shape {
            nx: 320,
            ny: 300,
            jitter: 0.15,
            stretched: true,
            ..long
        },
        10,
    )?;
    if !failed.is_empty() {
        println!(
            "(not bit for bit: {}; expected only in a build with multiply–add contraction)",
            failed.join("; ")
        );
    }
    Ok(())
}

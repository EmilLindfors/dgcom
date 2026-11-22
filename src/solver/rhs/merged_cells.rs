//! Thin cells that lose more than their water in a stage, carried together
//! with their neighbours through the step.
//!
//! # Why
//!
//! In a subcell element every node's layers are finite volumes over the
//! node's GLL subcell (see [`crate::solver::rhs::transport_3d`]). A thin
//! column gives and takes its tracers at the donor's value, so one
//! forward-Euler step of length `Δt_E` leaves the cell of node `i` on
//! level `l` with
//!
//! ```text
//!     V_i' C_i' = (V_i − Δt_E O_i) C_i + Δt_E Σ_j I_ij C_j,
//! ```
//!
//! `O_i` its outflow, `I_ij` its inflow from the cell `j` and
//! `V_i' = V_i − Δt_E O_i + Δt_E Σ_j I_ij` its new volume: the mean of its
//! own value and of what flowed in, weighted by their volumes, as long as
//! `Δt_E O_i ≤ V_i`. The stages of an SSP scheme are such steps from the
//! stage values, combined with weights that are not negative (Shu & Osher
//! 1988; Gottlieb, Shu & Tadmor 2001), so they keep it. This is Zhang &
//! Shu's (2010) argument for the mean of a cell under a forward-Euler
//! bound, with the subcell as the cell.
//!
//! The 3D step does not bound a thin column's outflow (films are the 2D
//! module's, whose substeps do): where the water passes through a film
//! cell faster than the step, `Δt_E O_i > V_i`, the cell's own value gets a
//! negative weight. Such a cell is the small cell of a cut-cell mesh, and
//! the cure here is that literature's: merge it with neighbours until the
//! merged cell holds what leaves it (cell merging; in state form, Berger &
//! Giuliani 2021). A group `A` of cells of one element with one value `C_A`
//! is one cell: the fluxes between its members cancel in its inventory,
//! and
//!
//! ```text
//!     V_A' C_A' = (V_A − Δt_E O_A) C_A + Δt_E Σ I C,
//! ```
//!
//! with `O_A` what leaves the group. Its members are set to their common
//! mean before the step's first stage ([`MergedCells::start`]), so that
//! every stage's values (which the neighbours read on the element's faces)
//! are the group's, and take the group's inventory over its volume after
//! every stage ([`crate::solver::rhs::from_inventory`]). Both are means
//! weighted by volumes, and keep the group's inventory.
//!
//! # The groups
//!
//! [`MergedCells::find`] takes them once per step from what the barotropic
//! pass moved: the mean fluxes through the subcell interfaces and the
//! element faces, constant over the step, and the water each cell holds at
//! least (its depth at the step's start or end, whichever is less: the
//! stage depths lie between). A thin cell with inflow and
//! `Δt_E O_i > V_i` starts a group, which takes in the neighbour it gives
//! the most water (or, with none, the one that gives it the most, or the
//! deepest) until `Δt_E O_A ≤ V_A`.
//!
//! # Flushed elements
//!
//! A group that holds every cell of its element and still loses more than
//! its water is flushed: water crosses the whole element within a stage (a
//! film), or the element was dry at the step's start and passes on, in the
//! mean fluxes of the step, water that reaches it during the step (a
//! wetting front that runs along an element's edge). No value of its own
//! can be what leaves it: more leaves than it holds. So what leaves it is,
//! as far as it goes, what enters it. With `I` and `O` the water entering
//! and leaving through its faces, `C_in` the mean of its neighbours' values
//! on the faces the water enters through, weighted by their fluxes, and
//! `C_A` its inventory over its volume, its nodes show (to its own
//! transport and to its neighbours)
//!
//! ```text
//!     X = [min(I, O) C_in + (O − I)⁺ C_A] / O,
//! ```
//!
//! at every stage ([`MergedCells::pass_through`]): the water that passes
//! through keeps its value, and only what the element loses on balance is
//! its own. A forward-Euler step of any length `τ` then leaves its
//! inventory
//!
//! ```text
//!     V' C_A' = [V − τ (O − I)⁺] C_A + τ (I − O)⁺ C_in:
//! ```
//!
//! filling, it is the mean of its own and of what stayed; draining, it
//! keeps its value. Its inventory is never set from its nodes' values
//! within the step, so it is kept exactly, and at the step's end the
//! element takes `C_A`. The run counts the merged and the flushed cells
//! ([`MergedCells::counts`]).

use crate::mesh::Mesh2D;
use crate::mesh::data::Bathymetry2D;
use crate::operators::{DGOperators2D, GeometricFactors2D};
use crate::solver::core::blocks::Blocks;
use crate::solver::rhs::subcells::{line_node, subcell_interfaces, subcell_slot};
use crate::solver::rhs::transport_3d::layer_volume_of;
use crate::types::ElementIndex;

/// What the barotropic pass of a step moved, as [`MergedCells::find`]
/// reads it.
pub struct StepWater<'a> {
    pub mesh: &'a Mesh2D,
    pub ops: &'a DGOperators2D,
    pub geom: &'a GeometricFactors2D,
    pub bathymetry: &'a Bathymetry2D,
    /// `η` at the step's start, `[element][node]`
    pub eta: &'a [f64],
    /// `∂η/∂t` over the step (m/s), `[element][node]`
    pub eta_rate: &'a [f64],
    /// The step (s)
    pub dt: f64,
    /// Every element's longest forward-Euler step in the stages (s)
    pub euler_dt: &'a [f64],
    /// The mean mass flux out of every element face node (m²/s),
    /// `(k · 4 + face) · n_face_nodes + fi`
    pub face: &'a [f64],
    /// The mean mass flux through every subcell interface, `k · n_sub +
    /// slot`; NaN where the pass did not move the element on its subcells
    /// throughout
    pub subcell: &'a [f64],
    /// The water of every column that no flux carried (m/s, `[element][node]`):
    /// `∂η/∂t` plus the fluxes' divergence
    pub source: &'a [f64],
    /// Columns shallower than this are thin (m)
    pub thin_depth: f64,
}

/// The groups of cells of a step (see the module docs).
pub struct MergedCells {
    n_nodes: usize,
    /// `[element][node]`: the first node of the node's group (its own:
    /// the node is on its own)
    group: Vec<u8>,
    /// `[element][node]`: in a group of two or more
    joined: Vec<bool>,
    /// `[element]`: the element is flushed
    flushed: Vec<bool>,
    /// The elements with a group
    elements: Vec<u32>,
    /// `[element]`: whether the element has a group
    grouped: Vec<bool>,
    /// The flushed elements, in order, with the water entering and leaving
    /// each through its faces (m³/s per unit of the reference element's
    /// area)
    flushed_elements: Vec<(u32, f64, f64)>,
    /// What flows into them from other elements: the element, the
    /// neighbour's node the water comes from (`[element][node]`) and its
    /// rate; element by element
    inflows: Vec<(u32, u32, f64)>,
    /// Cells merged and flushed, summed over the steps
    counts: [usize; 2],
}

/// Most nodes of an element the groups are found for (order 8).
const MAX_NODES: usize = 81;

impl MergedCells {
    /// No groups, on `n_elements` elements of `n_nodes` nodes.
    pub fn new(n_elements: usize, n_nodes: usize) -> Self {
        assert!(n_nodes <= u8::MAX as usize, "{n_nodes} nodes per element");
        Self {
            n_nodes,
            group: (0..n_elements * n_nodes)
                .map(|idx| (idx % n_nodes) as u8)
                .collect(),
            joined: vec![false; n_elements * n_nodes],
            flushed: vec![false; n_elements],
            elements: Vec::new(),
            grouped: vec![false; n_elements],
            flushed_elements: Vec::new(),
            inflows: Vec::new(),
            counts: [0; 2],
        }
    }

    /// The cells merged with others and the cells of flushed elements,
    /// summed over the steps so far (a cell counts once per step).
    pub fn counts(&self) -> [usize; 2] {
        self.counts
    }

    /// Whether element `k` has a group.
    #[inline]
    pub fn in_element(&self, k: usize) -> bool {
        self.grouped[k]
    }

    /// Whether element `k` is flushed: its inventory is not its nodes'
    /// values times their volumes within the step.
    #[inline]
    pub fn is_flushed(&self, k: usize) -> bool {
        self.flushed[k]
    }

    /// The elements with a group.
    pub fn elements(&self) -> &[u32] {
        &self.elements
    }

    fn clear(&mut self) {
        let nn = self.n_nodes;
        for &k in &self.elements {
            let k = k as usize;
            for i in 0..nn {
                self.group[k * nn + i] = i as u8;
                self.joined[k * nn + i] = false;
            }
            self.flushed[k] = false;
            self.grouped[k] = false;
        }
        self.elements.clear();
        self.flushed_elements.clear();
        self.inflows.clear();
    }

    /// Find the step's groups (see the module docs), replacing the last
    /// step's.
    pub fn find(&mut self, water: &StepWater) {
        self.clear();
        let StepWater {
            mesh,
            ops,
            geom,
            bathymetry,
            ..
        } = *water;
        let (nn, nfn, n1) = (ops.n_nodes, ops.n_face_nodes, ops.n_1d);
        assert_eq!(nn, self.n_nodes, "nodes per element");
        if nn > MAX_NODES || water.thin_depth <= 0.0 {
            return;
        }
        let n_sub = subcell_interfaces(n1);
        let w = &ops.weights_1d;
        // Per cell: the water it holds at least (m³ per unit of the
        // reference element's area), what leaves it and what enters it
        // through the faces and as a source (m³/s), and its group
        let mut least = [0.0_f64; MAX_NODES];
        let mut out = [0.0_f64; MAX_NODES];
        let mut inflow = [0.0_f64; MAX_NODES];
        let mut thin = [false; MAX_NODES];
        let mut group = [0usize; MAX_NODES];
        // The interfaces: (node a, node b, volume rate from a to b)
        let mut links: Vec<(usize, usize, f64)> = Vec::with_capacity(2 * n1 * (n1 - 1));
        for k in 0..self.grouped.len() {
            let fluxes = &water.subcell[k * n_sub..(k + 1) * n_sub];
            if !fluxes.iter().all(|f| f.is_finite()) {
                continue;
            }
            let bed = bathymetry.element(ElementIndex::new(k));
            let mut any_thin = false;
            for i in 0..nn {
                let idx = k * nn + i;
                let start = water.eta[idx] - bed[i];
                let end = start + water.dt * water.eta_rate[idx];
                thin[i] = start.min(end) < water.thin_depth;
                any_thin |= thin[i];
                let mass = geom.node_mass(k, i);
                least[i] = mass * start.min(end).max(0.0);
                let source = mass * water.source[idx];
                out[i] = (-source).max(0.0);
                inflow[i] = source.max(0.0);
                group[i] = i;
            }
            if !any_thin {
                continue;
            }
            // Through the faces: the element's, for a flushed one
            let (mut entering, mut leaving) = (0.0, 0.0);
            for f in 0..4 {
                for (fi, &node) in ops.face_nodes[f].iter().enumerate() {
                    let rate = face_rate(ops, geom, water.face, k, f, fi, node);
                    out[node] += rate.max(0.0);
                    inflow[node] += (-rate).max(0.0);
                    leaving += rate.max(0.0);
                    entering += (-rate).max(0.0);
                }
            }
            links.clear();
            for dir in 0..2 {
                for line in 0..n1 {
                    for a in 0..n1 - 1 {
                        let flux = fluxes[subcell_slot(n1, dir, line, a)];
                        // Per unit of the reference element's area: the
                        // line's weight across it
                        links.push((
                            line_node(n1, dir, line, a),
                            line_node(n1, dir, line, a + 1),
                            flux * w[line],
                        ));
                    }
                }
            }
            let euler = water.euler_dt[k];
            // A thin cell that water passes through faster than the stage
            let drained = |i: usize| {
                let (mut o, mut inn) = (out[i], inflow[i]);
                for &(a, b, rate) in &links {
                    if a == i {
                        o += rate.max(0.0);
                        inn += (-rate).max(0.0);
                    } else if b == i {
                        o += (-rate).max(0.0);
                        inn += rate.max(0.0);
                    }
                }
                thin[i] && inn > 0.0 && euler * o > least[i]
            };
            let (mut any, mut flushed) = (false, false);
            for seed in 0..nn {
                if !drained(seed) {
                    continue;
                }
                any = true;
                loop {
                    let root = group[seed];
                    // What the group holds, loses and takes
                    let (mut holds, mut loses, mut takes) = (0.0, 0.0, 0.0);
                    for i in (0..nn).filter(|&i| group[i] == root) {
                        holds += least[i];
                        loses += out[i];
                        takes += inflow[i];
                    }
                    // The neighbour it gives the most, the one that gives
                    // it the most, the deepest beside it
                    let mut partner: Option<(u8, f64, usize)> = None;
                    let mut offer = |rank: u8, size: f64, node: usize| {
                        if partner.is_none_or(|(r, s, _)| (rank, size) > (r, s)) {
                            partner = Some((rank, size, node));
                        }
                    };
                    for &(a, b, rate) in &links {
                        let (inside_a, inside_b) = (group[a] == root, group[b] == root);
                        if inside_a == inside_b {
                            continue;
                        }
                        // From the group to its neighbour
                        let (given, other) = if inside_a { (rate, b) } else { (-rate, a) };
                        loses += given.max(0.0);
                        takes += (-given).max(0.0);
                        if given > 0.0 {
                            offer(2, given, other);
                        } else if given < 0.0 {
                            offer(1, -given, other);
                        } else {
                            offer(0, least[other], other);
                        }
                    }
                    if takes <= 0.0 || euler * loses <= holds {
                        break;
                    }
                    let Some((_, _, other)) = partner else {
                        // Every cell of the element, and still not enough
                        flushed = true;
                        break;
                    };
                    let (from, to) = (group[other], root.min(group[other]));
                    for g in group[..nn].iter_mut() {
                        if *g == root || *g == from {
                            *g = to;
                        }
                    }
                }
                if flushed {
                    break;
                }
            }
            if !any {
                continue;
            }
            if flushed {
                // One group of every cell, and where its water comes from
                group[..nn].fill(0);
                self.flushed[k] = true;
                self.flushed_elements.push((k as u32, entering, leaving));
                for f in 0..4 {
                    let Some(neighbour) = mesh.neighbor(ElementIndex::new(k), f) else {
                        continue;
                    };
                    for (fi, &node) in ops.face_nodes[f].iter().enumerate() {
                        let rate = -face_rate(ops, geom, water.face, k, f, fi, node);
                        if rate > 0.0 {
                            let from = neighbour.element * nn
                                + ops.face_nodes[neighbour.face][nfn - 1 - fi];
                            self.inflows.push((k as u32, from as u32, rate));
                        }
                    }
                }
            }
            let mut grouped = false;
            for i in 0..nn {
                let size = group[..nn].iter().filter(|&&g| g == group[i]).count();
                let joined = size > 1;
                self.group[k * nn + i] = group[i] as u8;
                self.joined[k * nn + i] = joined;
                self.counts[usize::from(flushed)] += usize::from(joined);
                grouped |= joined;
            }
            if grouped {
                self.grouped[k] = true;
                self.elements.push(k as u32);
            }
        }
    }

    /// The state a step starts from, in which a group is one cell: set the
    /// members of every group to their mean of `field` (a concentration
    /// over cells of σ-thickness `cells` in a column,
    /// `[element][node][cell]`), weighted by their volumes under `eta`. A
    /// group without water is left, and so is a flushed element (its
    /// nodes' values at the first stage are [`Self::pass_through`]'s).
    pub fn start(
        &self,
        field: &mut [f64],
        cells: &[f64],
        eta: &[f64],
        bathymetry: &Bathymetry2D,
        geom: &GeometricFactors2D,
    ) {
        let (nn, nc) = (self.n_nodes, cells.len());
        for &k in &self.elements {
            let k = k as usize;
            if self.flushed[k] {
                continue;
            }
            let bed = bathymetry.element(ElementIndex::new(k));
            let field_k = &mut field[k * nn * nc..(k + 1) * nn * nc];
            self.for_each_group(k, |members| {
                for (c, &dc) in cells.iter().enumerate() {
                    let (mut inventory, mut volume) = (0.0, 0.0);
                    for &i in members {
                        let weight =
                            geom.node_mass(k, i) * layer_volume_of(eta[k * nn + i] - bed[i], dc);
                        inventory += weight * field_k[i * nc + c];
                        volume += weight;
                    }
                    if volume > 0.0 {
                        for &i in members {
                            field_k[i * nc + c] = inventory / volume;
                        }
                    }
                }
            });
        }
    }

    /// Call `f` with the nodes of every group of two or more of element
    /// `k`.
    fn for_each_group(&self, k: usize, mut f: impl FnMut(&[usize])) {
        let nn = self.n_nodes;
        let group = &self.group[k * nn..(k + 1) * nn];
        let mut members = [0usize; MAX_NODES];
        for root in 0..nn {
            if group[root] as usize != root || !self.joined[k * nn + root] {
                continue;
            }
            let mut n = 0;
            for i in (0..nn).filter(|&i| group[i] as usize == root) {
                members[n] = i;
                n += 1;
            }
            f(&members[..n]);
        }
    }

    /// Whether node `i` of element `k` is carried with others: its value is
    /// its group's, not its own inventory over its volume.
    #[inline]
    pub(crate) fn carries(&self, k: usize, i: usize) -> bool {
        self.joined[k * self.n_nodes + i]
    }

    /// Element `k`'s groups' values from the inventories `q_k` into
    /// `out_k` (`[node][cell]`), which holds the last values: every member
    /// the group's inventory over its volume under `eta_k` (a group without
    /// water keeps the last values). Not a flushed element's
    /// ([`Self::pass_through`]).
    pub(crate) fn convert_element(
        &self,
        k: usize,
        q_k: &[f64],
        eta_k: &[f64],
        bed: &[f64],
        cells: &[f64],
        geom: &GeometricFactors2D,
        out_k: &mut [f64],
    ) {
        if self.flushed[k] {
            return;
        }
        let nc = cells.len();
        self.for_each_group(k, |members| {
            for (c, &dc) in cells.iter().enumerate() {
                let (mut inventory, mut volume) = (0.0, 0.0);
                for &i in members {
                    let mass = geom.node_mass(k, i);
                    inventory += mass * q_k[i * nc + c];
                    volume += mass * layer_volume_of(eta_k[i] - bed[i], dc);
                }
                if volume > 0.0 {
                    for &i in members {
                        out_k[i * nc + c] = inventory / volume;
                    }
                }
            }
        });
    }

    /// The values of the flushed elements of `elements` into `out`
    /// (`[element][node][cell]`, which holds the other elements' values
    /// and, in these, the last ones), from their inventories `q` under
    /// `eta`: at a step's end (`end`) each element's inventory over its
    /// volume, before that the value that passes its inflow through (see
    /// the module docs), from its neighbours' values in `out`. An element
    /// without water takes what flows in, or keeps the last values.
    #[allow(clippy::too_many_arguments)]
    pub fn pass_through(
        &self,
        q: &[f64],
        eta: &[f64],
        cells: &[f64],
        bathymetry: &Bathymetry2D,
        geom: &GeometricFactors2D,
        out: &mut [f64],
        elements: Blocks,
        end: bool,
    ) {
        let (nn, nc) = (self.n_nodes, cells.len());
        let mut inflows = &self.inflows[..];
        for &(k, entering, leaving) in &self.flushed_elements {
            let (inflow, rest) = inflows.split_at(inflows.iter().take_while(|i| i.0 == k).count());
            inflows = rest;
            let k = k as usize;
            if !elements.contains(k) {
                continue;
            }
            let bed = bathymetry.element(ElementIndex::new(k));
            // From other elements: the open faces' inflow has no value here
            let from_elements: f64 = inflow.iter().map(|i| i.2).sum();
            // The share of what leaves that is what enters
            let passing = if leaving > 0.0 && from_elements > 0.0 {
                (entering.min(leaving) / leaving).min(1.0)
            } else {
                0.0
            };
            for (c, &dc) in cells.iter().enumerate() {
                let (mut inventory, mut volume) = (0.0, 0.0);
                for i in 0..nn {
                    let mass = geom.node_mass(k, i);
                    inventory += mass * q[(k * nn + i) * nc + c];
                    volume += mass * layer_volume_of(eta[k * nn + i] - bed[i], dc);
                }
                let own = (volume > 0.0).then(|| inventory / volume);
                let value = if end || passing == 0.0 {
                    own
                } else {
                    let entering: f64 = inflow
                        .iter()
                        .map(|&(_, from, rate)| rate * out[from as usize * nc + c])
                        .sum();
                    let entering = entering / from_elements;
                    Some(match own {
                        Some(own) => passing * entering + (1.0 - passing) * own,
                        None => entering,
                    })
                };
                if let Some(value) = value {
                    for i in 0..nn {
                        out[(k * nn + i) * nc + c] = value;
                    }
                }
            }
        }
    }
}

/// The volume rate out of element `k` at face node `fi` of `face` (which is
/// the element's node `node`), per unit of the reference element's area,
/// from the mass fluxes `fluxes` out of every face node.
#[inline]
fn face_rate(
    ops: &DGOperators2D,
    geom: &GeometricFactors2D,
    fluxes: &[f64],
    k: usize,
    face: usize,
    fi: usize,
    node: usize,
) -> f64 {
    geom.node_mass(k, node)
        * geom.lift_scale(k, face, fi, node)
        * ops.lift[face][(node, fi)]
        * fluxes[(k * 4 + face) * ops.n_face_nodes + fi]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two P2 elements 60 m across side by side, the right one's right
    /// column of nodes a film (`film` m deep, the rest of both `deep` m),
    /// still, and the pass's fluxes: `through` m²/s from the left element
    /// through the right one's two interfaces of every line and out of its
    /// right face (`out` times as much).
    struct Flume {
        mesh: Mesh2D,
        ops: DGOperators2D,
        geom: GeometricFactors2D,
        bathymetry: Bathymetry2D,
        eta: Vec<f64>,
        face: Vec<f64>,
        subcell: Vec<f64>,
    }

    /// The flume's right element.
    const RIGHT: usize = 1;

    impl Flume {
        fn new(film: f64, deep: f64, through: f64) -> Self {
            Self::with_outflow(film, deep, through, 1.0)
        }

        fn with_outflow(film: f64, deep: f64, through: f64, out: f64) -> Self {
            let mesh = Mesh2D::uniform_rectangle(0.0, 120.0, 0.0, 60.0, 2, 1);
            let ops = DGOperators2D::new(2);
            let geom = GeometricFactors2D::compute(&mesh, &ops);
            let bathymetry = Bathymetry2D::from_function(&mesh, &ops, &geom, |_, _| -1.0);
            let (nn, n1, nfn) = (ops.n_nodes, ops.n_1d, ops.n_face_nodes);
            let x_of = |k: usize, i: usize| {
                mesh.reference_to_physical(ElementIndex::new(k), ops.nodes_r[i], ops.nodes_s[i])[0]
            };
            let eta = (0..2 * nn)
                .map(|idx| {
                    -1.0 + if x_of(idx / nn, idx % nn) > 119.0 {
                        film
                    } else {
                        deep
                    }
                })
                .collect();
            // Rightward flow through both elements: out where the normal
            // points right
            let mut face = vec![0.0; 2 * 4 * nfn];
            for k in 0..2 {
                for f in 0..4 {
                    for fi in 0..nfn {
                        let (nx, _) = geom.normal(k, f, fi);
                        let last = k == RIGHT && nx > 0.5;
                        if nx.abs() > 0.5 {
                            face[(k * 4 + f) * nfn + fi] =
                                nx * through * if last { out } else { 1.0 };
                        }
                    }
                }
            }
            // The interfaces across x carry the same flow: the contravariant
            // flux J ∇r · (hu, 0) = (Δy/2) hu
            let n_sub = subcell_interfaces(n1);
            let mut subcell = vec![0.0; 2 * n_sub];
            let x_dir = (0..2)
                .find(|&dir| {
                    let (a, b) = (line_node(n1, dir, 0, 0), line_node(n1, dir, 0, 1));
                    (ops.nodes_r[a] - ops.nodes_r[b]).abs() > 0.5
                })
                .expect("a direction along r");
            let sign = {
                let (a, b) = (line_node(n1, x_dir, 0, 0), line_node(n1, x_dir, 0, 1));
                (ops.nodes_r[b] - ops.nodes_r[a]).signum()
            };
            for k in 0..2 {
                for line in 0..n1 {
                    for a in 0..n1 - 1 {
                        subcell[k * n_sub + subcell_slot(n1, x_dir, line, a)] =
                            sign * 30.0 * through;
                    }
                }
            }
            Self {
                mesh,
                ops,
                geom,
                bathymetry,
                eta,
                face,
                subcell,
            }
        }

        fn merged(&self, euler: f64) -> MergedCells {
            let nn = self.ops.n_nodes;
            let mut merged = MergedCells::new(2, nn);
            merged.find(&StepWater {
                mesh: &self.mesh,
                ops: &self.ops,
                geom: &self.geom,
                bathymetry: &self.bathymetry,
                eta: &self.eta,
                eta_rate: &vec![0.0; 2 * nn],
                dt: 2.0 * euler,
                euler_dt: &[euler; 2],
                face: &self.face,
                subcell: &self.subcell,
                source: &vec![0.0; 2 * nn],
                thin_depth: 1.0,
            });
            merged
        }

        fn x(&self, idx: usize) -> f64 {
            let nn = self.ops.n_nodes;
            self.mesh.reference_to_physical(
                ElementIndex::new(idx / nn),
                self.ops.nodes_r[idx % nn],
                self.ops.nodes_s[idx % nn],
            )[0]
        }
    }

    /// A film cell 10 m across with 1 cm of water and 0.01 m²/s passing
    /// through it loses its water in 10 s: on its own at a 5 s stage; at
    /// 20 s merged with the cell that gives it its water, whose 2 m hold
    /// the flow (it gives none to a cell of its element). With 1 cm
    /// everywhere at 200 s the right element is one group and still not
    /// enough: flushed.
    #[test]
    fn a_film_cell_is_merged_with_its_neighbour() {
        let flume = Flume::new(0.01, 2.0, 0.01);
        let nn = flume.ops.n_nodes;
        let film: Vec<usize> = (RIGHT * nn..2 * nn)
            .filter(|&i| flume.x(i) > 119.0)
            .collect();
        assert_eq!(film.len(), 3);
        let merged = flume.merged(5.0);
        assert_eq!(merged.counts(), [0, 0]);
        assert!(!merged.in_element(RIGHT));

        let merged = flume.merged(20.0);
        assert_eq!(merged.counts(), [2 * film.len(), 0]);
        assert!(merged.in_element(RIGHT) && !merged.in_element(0));
        for &idx in &film {
            let i = idx % nn;
            assert!(merged.carries(RIGHT, i), "film node {i}");
            let root = merged.group[idx];
            let partners: Vec<usize> = (0..nn)
                .filter(|&j| j != i && merged.group[RIGHT * nn + j] == root)
                .collect();
            assert_eq!(partners.len(), 1, "one partner");
            let j = partners[0];
            assert!(
                (flume.x(RIGHT * nn + j) - 90.0).abs() < 1.0
                    && flume.ops.nodes_s[j] == flume.ops.nodes_s[i],
                "node {i} merged with {j}, not the cell that feeds it"
            );
        }

        let flat = Flume::new(0.01, 0.01, 0.01);
        let merged = flat.merged(200.0);
        assert_eq!(merged.counts(), [0, 2 * nn]);
        assert!(merged.is_flushed(RIGHT) && merged.is_flushed(0));
        // The right element's water comes from the left one's right face
        let from: Vec<usize> = merged
            .inflows
            .iter()
            .filter(|inflow| inflow.0 as usize == RIGHT)
            .map(|inflow| inflow.1 as usize)
            .collect();
        assert_eq!(from.len(), 3);
        for idx in from {
            assert!(idx < nn && (flume.x(idx) - 60.0).abs() < 1e-9, "node {idx}");
        }
    }

    /// The start of a step and the conversion of a group's inventories keep
    /// the group's inventory, give its members one value between theirs,
    /// and leave the others.
    #[test]
    fn a_group_takes_its_mean_and_keeps_its_inventory() {
        let flume = Flume::new(0.01, 2.0, 0.01);
        let (nn, nc) = (flume.ops.n_nodes, 3);
        let merged = flume.merged(20.0);
        let cells = [0.2, 0.3, 0.5];
        let bed = flume.bathymetry.element(ElementIndex::new(RIGHT));
        let eta = &flume.eta[RIGHT * nn..];
        let volume = |i: usize, c: usize| {
            flume.geom.node_mass(RIGHT, i) * layer_volume_of(eta[i] - bed[i], cells[c])
        };
        let values: Vec<f64> = (0..2 * nn * nc).map(|j| 5.0 + (j % 7) as f64).collect();
        let right = |field: &[f64]| field[RIGHT * nn * nc..].to_vec();
        let inventory = |field: &[f64], c: usize| -> f64 {
            (0..nn).map(|i| volume(i, c) * field[i * nc + c]).sum()
        };
        let mut mixed = values.clone();
        merged.start(
            &mut mixed,
            &cells,
            &flume.eta,
            &flume.bathymetry,
            &flume.geom,
        );
        assert_eq!(
            mixed[..nn * nc],
            values[..nn * nc],
            "the left element has no group"
        );
        let (values, mixed) = (right(&values), right(&mixed));
        // The same from inventories
        let q: Vec<f64> = (0..nn * nc)
            .map(|j| values[j] * layer_volume_of(eta[j / nc] - bed[j / nc], cells[j % nc]))
            .collect();
        let mut converted = values.clone();
        merged.convert_element(RIGHT, &q, eta, bed, &cells, &flume.geom, &mut converted);
        for c in 0..nc {
            let (before, after) = (inventory(&values, c), inventory(&mixed, c));
            assert!((before - after).abs() < 1e-13 * before.abs());
            for i in 0..nn {
                let (was, is) = (values[i * nc + c], mixed[i * nc + c]);
                assert!((is - converted[i * nc + c]).abs() < 1e-12);
                if !merged.carries(RIGHT, i) {
                    assert_eq!(was, is, "node {i} is on its own");
                    continue;
                }
                let root = merged.group[RIGHT * nn + i];
                let group: Vec<usize> = (0..nn)
                    .filter(|&j| merged.group[RIGHT * nn + j] == root)
                    .collect();
                let (lo, hi) = group
                    .iter()
                    .map(|&j| values[j * nc + c])
                    .fold((f64::MAX, f64::MIN), |(a, b), x| (a.min(x), b.max(x)));
                assert!(
                    is >= lo - 1e-12 && is <= hi + 1e-12,
                    "{is} outside {lo}..{hi}"
                );
                for &j in &group {
                    assert_eq!(mixed[j * nc + c], is, "one value in the group");
                }
            }
        }
    }

    /// A flushed element's nodes show what passes through it: all of what
    /// flows in where no more leaves than enters (and where it holds no
    /// water), a mix with its own water where more leaves; at a step's end
    /// its inventory over its volume. One forward-Euler step with those
    /// values on its outflow leaves its inventory's mean between its own
    /// and what flowed in, over a stage longer than its water lasts.
    #[test]
    fn a_flushed_element_passes_its_inflow_through() {
        let (nc, cells) = (2, [0.4, 0.6]);
        // (film depth, outflow over inflow)
        for (film, out) in [(0.0, 0.5), (0.01, 1.0), (0.01, 0.5), (0.01, 1.5)] {
            let mut flume = Flume::with_outflow(film, film, 0.01, out);
            let nn = flume.ops.n_nodes;
            // Water in the left element
            flume.eta[..nn].fill(1.0);
            let euler = 200.0;
            let merged = flume.merged(euler);
            assert!(merged.is_flushed(RIGHT) && !merged.is_flushed(0));
            // 8 and 9 on the left (by level), 20 and 21 in the right
            let base = |j: usize| if j / nc < nn { 8.0 } else { 20.0 } + (j % nc) as f64;
            let field: Vec<f64> = (0..2 * nn * nc).map(base).collect();
            let bed = flume.bathymetry.element(ElementIndex::new(RIGHT));
            let volume: Vec<f64> = (0..nn * nc)
                .map(|j| {
                    flume.geom.node_mass(RIGHT, j / nc)
                        * layer_volume_of(
                            flume.eta[RIGHT * nn + j / nc] - bed[j / nc],
                            cells[j % nc],
                        )
                })
                .collect();
            let mut q = field.clone();
            for (j, q) in q.iter_mut().enumerate() {
                let depth = flume.eta[j / nc] - flume.bathymetry.data[j / nc];
                *q *= layer_volume_of(depth, cells[j % nc]);
            }
            let both = Blocks::All(2);
            let convert = |end: bool| {
                let mut out = field.clone();
                let (eta, bathymetry, geom) = (&flume.eta, &flume.bathymetry, &flume.geom);
                merged.pass_through(&q, eta, &cells, bathymetry, geom, &mut out, both, end);
                out
            };
            let (shown, ended) = (convert(false), convert(true));
            assert_eq!(
                shown[..nn * nc],
                field[..nn * nc],
                "the left element is not flushed"
            );
            let passing = if film == 0.0 {
                1.0
            } else {
                (1.0 / out).min(1.0)
            };
            for c in 0..nc {
                let (entering, own) = (8.0 + c as f64, 20.0 + c as f64);
                let expected = passing * entering + (1.0 - passing) * own;
                for i in 0..nn {
                    let j = (RIGHT * nn + i) * nc + c;
                    assert!(
                        (shown[j] - expected).abs() < 1e-12,
                        "film {film}, out {out}"
                    );
                    // At the end: its own, or with no water the last value
                    assert!((ended[j] - own).abs() < 1e-12);
                }
                // One forward-Euler step of the element's inventory: what
                // enters at the left's value, what leaves at the shown one
                let (held, rate): (f64, f64) = (
                    (0..nn).map(|i| volume[i * nc + c]).sum(),
                    merged.flushed_elements[0].1 * cells[c],
                );
                let after = held + euler * rate * (1.0 - out);
                if after <= 0.0 {
                    continue;
                }
                let mean = (held * own + euler * rate * (entering - out * expected)) / after;
                assert!(
                    mean >= entering - 1e-9 && mean <= own + 1e-9,
                    "film {film}, out {out}: {mean} outside {entering}..{own}"
                );
                assert!(euler * rate * out > held, "test regime: flushed");
            }
        }
    }
}

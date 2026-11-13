//! 2D Source terms for shallow water equations.
//!
//! Source terms represent forces not part of the hyperbolic flux:
//! - Coriolis: S = (0, f·hv, -f·hu)
//! - Tidal potential: S = (0, -gh ∂Φ/∂x, -gh ∂Φ/∂y)
//! - Bottom friction (Manning): S = (0, -gn²|u|u/h^{1/3}, -gn²|u|v/h^{1/3})
//! - Sponge layer damping: S = γ(q_ref - q)
//!
//! Source terms are evaluated at each quadrature node and added to the RHS.

use std::sync::Arc;

use crate::mesh::{Bathymetry2D, Mesh2D};
use crate::operators::DGOperators2D;
use crate::solver::{SWESolution2D, SWEState2D};
use crate::types::ElementIndex;

/// Context for 2D source term evaluation.
///
/// Provides all information needed to evaluate a source term at a single node.
#[derive(Clone, Copy, Debug)]
pub struct SourceContext2D {
    /// Current simulation time
    pub time: f64,
    /// Physical position (x, y)
    pub position: (f64, f64),
    /// Current state (h, hu, hv)
    pub state: SWEState2D,
    /// Bathymetry (bottom elevation) at this point
    pub bathymetry: f64,
    /// Bathymetry gradients (∂B/∂x, ∂B/∂y)
    pub bathymetry_gradient: (f64, f64),
    /// Gravitational acceleration
    pub g: f64,
    /// Minimum depth threshold for wet/dry
    pub h_min: f64,
}

impl SourceContext2D {
    /// Create a new source context.
    pub fn new(
        time: f64,
        position: (f64, f64),
        state: SWEState2D,
        bathymetry: f64,
        bathymetry_gradient: (f64, f64),
        g: f64,
        h_min: f64,
    ) -> Self {
        Self {
            time,
            position,
            state,
            bathymetry,
            bathymetry_gradient,
            g,
            h_min,
        }
    }

    /// Get velocity (u, v) with desingularization for dry cells.
    pub fn velocity(&self) -> (f64, f64) {
        if self.state.h < self.h_min {
            (0.0, 0.0)
        } else {
            (self.state.hu / self.state.h, self.state.hv / self.state.h)
        }
    }

    /// Get water surface elevation (η = h + B).
    pub fn surface_elevation(&self) -> f64 {
        self.state.h + self.bathymetry
    }
}

/// One element's worth of source-term inputs, for
/// [`SourceTerm2D::add_element`]: the RHS kernel calls each source once per
/// element instead of once per node, and a source reads only what it needs
/// (the node position costs a bilinear map, the per-node context a copy of
/// every field).
#[derive(Clone, Copy)]
pub struct ElementSources<'a> {
    /// The element
    pub element: ElementIndex,
    /// Current simulation time
    pub time: f64,
    /// The solution the RHS is evaluated at
    pub solution: &'a SWESolution2D,
    /// Mesh (for node positions)
    pub mesh: &'a Mesh2D,
    /// Reference-element operators (for node positions)
    pub ops: &'a DGOperators2D,
    /// Nodal bathymetry, if any
    pub bathymetry: Option<&'a Bathymetry2D>,
    /// Gravitational acceleration
    pub g: f64,
    /// Minimum depth threshold for wet/dry
    pub h_min: f64,
}

impl ElementSources<'_> {
    /// Number of nodes of the element.
    #[inline]
    pub fn n_nodes(&self) -> usize {
        self.ops.n_nodes
    }

    /// State at node `i`.
    #[inline]
    pub fn state(&self, i: usize) -> SWEState2D {
        self.solution.get_state(self.element, i)
    }

    /// Physical position of node `i`.
    #[inline]
    pub fn position(&self, i: usize) -> (f64, f64) {
        let [x, y] =
            self.mesh
                .reference_to_physical(self.element, self.ops.nodes_r[i], self.ops.nodes_s[i]);
        (x, y)
    }

    /// The full per-node context of [`SourceTerm2D::evaluate`] at node `i`.
    pub fn context(&self, i: usize) -> SourceContext2D {
        let (bathymetry, gradient) = self.bathymetry.map_or((0.0, (0.0, 0.0)), |b| {
            (b.get(self.element, i), b.get_gradient(self.element, i))
        });
        SourceContext2D::new(
            self.time,
            self.position(i),
            self.state(i),
            bathymetry,
            gradient,
            self.g,
            self.h_min,
        )
    }
}

/// Trait for 2D source terms in shallow water equations.
///
/// Source terms modify the RHS of the equations:
/// dq/dt = -∇·F + S(q, x, y, t)
///
/// Implementations must be thread-safe (`Send + Sync`) for parallel computation.
pub trait SourceTerm2D: Send + Sync {
    /// Evaluate the source term contribution at a single node.
    ///
    /// # Arguments
    /// * `ctx` - Context containing state, position, time, and bathymetry
    ///
    /// # Returns
    /// Source contribution as SWEState2D (S_h, S_hu, S_hv)
    fn evaluate(&self, ctx: &SourceContext2D) -> SWEState2D;

    /// Add this source at every node of one element to the element's RHS
    /// (`h`, `hu`, `hv`, one value per node). This is what the 2D SWE RHS
    /// calls.
    ///
    /// The default builds the full [`SourceContext2D`] per node and calls
    /// [`Self::evaluate`]. Sources that need less (Coriolis on an f-plane
    /// needs only the momentum) override it with a plain loop, which saves
    /// the node position, the context and a virtual call per node.
    fn add_element(
        &self,
        element: &ElementSources<'_>,
        h: &mut [f64],
        hu: &mut [f64],
        hv: &mut [f64],
    ) {
        for i in 0..element.n_nodes() {
            let s = self.evaluate(&element.context(i));
            h[i] += s.h;
            hu[i] += s.hu;
            hv[i] += s.hv;
        }
    }

    /// Name of this source term for debugging and logging.
    fn name(&self) -> &'static str;

    /// Whether this source term requires special treatment (e.g., implicit).
    ///
    /// Stiff source terms (e.g., strong friction, large sponge damping)
    /// may require implicit or semi-implicit time integration.
    fn is_stiff(&self) -> bool {
        false
    }

    /// Whether the source at a node reads, of the state, that node only
    /// (and whatever is not the state: time, position, bed, forcing). The
    /// default [`Self::add_element`] does, node by node; an override must
    /// keep to it to say so. Local time stepping then re-evaluates an
    /// element only when the nodes its kernel reads have changed
    /// ([`crate::time::LocalTimeStepping::line_local`]); `false` (the
    /// default) keeps every element of the run on whole-element stage rates.
    fn is_node_local(&self) -> bool {
        false
    }

    /// Whether this source term discretizes the bed-slope force `−g h ∇B`.
    ///
    /// The split-form formulations (`SWEFormulation2D::EntropyStable` and
    /// `EntropyConservative`) build the bed slope into the DG operator itself, so
    /// the 2D SWE RHS refuses source terms that would add it a second time.
    fn includes_bathymetry_slope(&self) -> bool {
        false
    }

    /// Whether this source changes the water's volume (a nonzero `S_h`):
    /// a relaxation of the level (`NestingRelaxation2D`, `SpongeLayer2D`
    /// unless momentum-only) or rivers. No face flux carries that water, so
    /// a 3D model over the 2D module has to give it to its layers as a
    /// volume source (`crate::time::BarotropicPhysics::mass_sources_into`).
    fn changes_mass(&self) -> bool {
        false
    }

    /// Add `S_h` alone at every node of one element to `h` (nothing where
    /// [`Self::changes_mass`] is false). The default evaluates the whole
    /// source per node.
    fn add_mass_element(&self, element: &ElementSources<'_>, h: &mut [f64]) {
        if !self.changes_mass() {
            return;
        }
        for (i, h) in h.iter_mut().enumerate().take(element.n_nodes()) {
            *h += self.evaluate(&element.context(i)).h;
        }
    }
}

/// Combine multiple 2D source terms into one.
///
/// The combined source evaluates all constituent sources and sums their contributions.
///
/// # Example
/// ```ignore
/// let coriolis = CoriolisSource2D::f_plane(1.2e-4);
/// let friction = ManningFriction2D::new(0.03, 9.81);
/// let combined = CombinedSource2D::new(vec![&coriolis, &friction]);
/// ```
pub struct CombinedSource2D<'a> {
    sources: Vec<&'a dyn SourceTerm2D>,
}

impl<'a> CombinedSource2D<'a> {
    /// Create a new combined source from a list of source terms.
    pub fn new(sources: Vec<&'a dyn SourceTerm2D>) -> Self {
        Self { sources }
    }

    /// Add a source term to the combination.
    pub fn add(&mut self, source: &'a dyn SourceTerm2D) {
        self.sources.push(source);
    }

    /// Number of source terms in the combination.
    pub fn len(&self) -> usize {
        self.sources.len()
    }

    /// Whether the combination is empty.
    pub fn is_empty(&self) -> bool {
        self.sources.is_empty()
    }
}

impl<'a> SourceTerm2D for CombinedSource2D<'a> {
    fn evaluate(&self, ctx: &SourceContext2D) -> SWEState2D {
        let mut total = SWEState2D::zero();
        for source in &self.sources {
            total = total + source.evaluate(ctx);
        }
        total
    }

    fn add_element(
        &self,
        element: &ElementSources<'_>,
        h: &mut [f64],
        hu: &mut [f64],
        hv: &mut [f64],
    ) {
        for source in &self.sources {
            source.add_element(element, h, hu, hv);
        }
    }

    fn is_node_local(&self) -> bool {
        self.sources.iter().all(|s| s.is_node_local())
    }

    fn name(&self) -> &'static str {
        "combined_2d"
    }

    fn is_stiff(&self) -> bool {
        self.sources.iter().any(|s| s.is_stiff())
    }

    fn includes_bathymetry_slope(&self) -> bool {
        self.sources.iter().any(|s| s.includes_bathymetry_slope())
    }

    fn changes_mass(&self) -> bool {
        self.sources.iter().any(|s| s.changes_mass())
    }

    fn add_mass_element(&self, element: &ElementSources<'_>, h: &mut [f64]) {
        for source in &self.sources {
            source.add_mass_element(element, h);
        }
    }
}

/// Owned counterpart of [`CombinedSource2D`]: sums source terms held by
/// `Arc`, so the combination can be stored in `'static` contexts such as
/// `SWEPhysics2D` (`SWEPhysics2DBuilder::with_source` builds one).
#[derive(Clone, Default)]
pub struct SourceTerms2D {
    sources: Vec<Arc<dyn SourceTerm2D>>,
}

impl SourceTerms2D {
    /// Combine `sources`.
    pub fn new(sources: Vec<Arc<dyn SourceTerm2D>>) -> Self {
        Self { sources }
    }

    /// Add a source term to the combination.
    pub fn push(&mut self, source: Arc<dyn SourceTerm2D>) {
        self.sources.push(source);
    }

    /// Number of source terms in the combination.
    pub fn len(&self) -> usize {
        self.sources.len()
    }

    /// Whether the combination is empty.
    pub fn is_empty(&self) -> bool {
        self.sources.is_empty()
    }
}

impl SourceTerm2D for SourceTerms2D {
    fn evaluate(&self, ctx: &SourceContext2D) -> SWEState2D {
        self.sources
            .iter()
            .fold(SWEState2D::zero(), |total, s| total + s.evaluate(ctx))
    }

    fn add_element(
        &self,
        element: &ElementSources<'_>,
        h: &mut [f64],
        hu: &mut [f64],
        hv: &mut [f64],
    ) {
        for source in &self.sources {
            source.add_element(element, h, hu, hv);
        }
    }

    fn is_node_local(&self) -> bool {
        self.sources.iter().all(|s| s.is_node_local())
    }

    fn name(&self) -> &'static str {
        "source_terms_2d"
    }

    fn is_stiff(&self) -> bool {
        self.sources.iter().any(|s| s.is_stiff())
    }

    fn includes_bathymetry_slope(&self) -> bool {
        self.sources.iter().any(|s| s.includes_bathymetry_slope())
    }

    fn changes_mass(&self) -> bool {
        self.sources.iter().any(|s| s.changes_mass())
    }

    fn add_mass_element(&self, element: &ElementSources<'_>, h: &mut [f64]) {
        for source in &self.sources {
            source.add_mass_element(element, h);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct ZeroSource2D;

    impl SourceTerm2D for ZeroSource2D {
        fn evaluate(&self, _ctx: &SourceContext2D) -> SWEState2D {
            SWEState2D::zero()
        }

        fn name(&self) -> &'static str {
            "zero_2d"
        }
    }

    struct ConstantSource2D {
        value: SWEState2D,
    }

    impl SourceTerm2D for ConstantSource2D {
        fn evaluate(&self, _ctx: &SourceContext2D) -> SWEState2D {
            self.value
        }

        fn name(&self) -> &'static str {
            "constant_2d"
        }
    }

    fn make_test_context() -> SourceContext2D {
        SourceContext2D::new(
            0.0,
            (0.0, 0.0),
            SWEState2D::new(1.0, 0.5, 0.3),
            0.0,
            (0.0, 0.0),
            9.81,
            1e-6,
        )
    }

    #[test]
    fn test_zero_source_2d() {
        let source = ZeroSource2D;
        let ctx = make_test_context();
        let result = source.evaluate(&ctx);

        assert!(result.h.abs() < 1e-14);
        assert!(result.hu.abs() < 1e-14);
        assert!(result.hv.abs() < 1e-14);
    }

    #[test]
    fn test_constant_source_2d() {
        let source = ConstantSource2D {
            value: SWEState2D::new(1.0, 2.0, 3.0),
        };
        let ctx = make_test_context();
        let result = source.evaluate(&ctx);

        assert!((result.h - 1.0).abs() < 1e-14);
        assert!((result.hu - 2.0).abs() < 1e-14);
        assert!((result.hv - 3.0).abs() < 1e-14);
    }

    #[test]
    fn test_combined_source_2d() {
        let s1 = ConstantSource2D {
            value: SWEState2D::new(1.0, 2.0, 3.0),
        };
        let s2 = ConstantSource2D {
            value: SWEState2D::new(0.5, 1.0, 1.5),
        };

        let combined = CombinedSource2D::new(vec![&s1, &s2]);
        let ctx = make_test_context();
        let result = combined.evaluate(&ctx);

        assert!((result.h - 1.5).abs() < 1e-14);
        assert!((result.hu - 3.0).abs() < 1e-14);
        assert!((result.hv - 4.5).abs() < 1e-14);
    }

    #[test]
    fn test_combined_source_empty() {
        let combined: CombinedSource2D = CombinedSource2D::new(vec![]);
        let ctx = make_test_context();
        let result = combined.evaluate(&ctx);

        assert!(result.h.abs() < 1e-14);
        assert!(result.hu.abs() < 1e-14);
        assert!(result.hv.abs() < 1e-14);
        assert!(combined.is_empty());
    }

    #[test]
    fn test_source_context_velocity() {
        let ctx = SourceContext2D::new(
            0.0,
            (0.0, 0.0),
            SWEState2D::new(2.0, 4.0, 6.0), // h=2, hu=4, hv=6 => u=2, v=3
            0.0,
            (0.0, 0.0),
            9.81,
            1e-6,
        );

        let (u, v) = ctx.velocity();
        assert!((u - 2.0).abs() < 1e-14);
        assert!((v - 3.0).abs() < 1e-14);
    }

    #[test]
    fn test_source_context_dry_velocity() {
        let ctx = SourceContext2D::new(
            0.0,
            (0.0, 0.0),
            SWEState2D::new(1e-10, 0.1, 0.2), // Very shallow - should return zero velocity
            0.0,
            (0.0, 0.0),
            9.81,
            1e-6,
        );

        let (u, v) = ctx.velocity();
        assert!(u.abs() < 1e-14);
        assert!(v.abs() < 1e-14);
    }

    #[test]
    fn test_source_context_surface_elevation() {
        let ctx = SourceContext2D::new(
            0.0,
            (0.0, 0.0),
            SWEState2D::new(5.0, 0.0, 0.0),
            -10.0, // Bottom at -10m
            (0.0, 0.0),
            9.81,
            1e-6,
        );

        let eta = ctx.surface_elevation();
        assert!((eta - (-5.0)).abs() < 1e-14); // Surface at h + B = 5 + (-10) = -5
    }

    #[test]
    fn test_combined_is_stiff() {
        struct StiffSource;
        impl SourceTerm2D for StiffSource {
            fn evaluate(&self, _: &SourceContext2D) -> SWEState2D {
                SWEState2D::zero()
            }
            fn name(&self) -> &'static str {
                "stiff"
            }
            fn is_stiff(&self) -> bool {
                true
            }
        }

        let normal = ZeroSource2D;
        let stiff = StiffSource;

        let combined_normal = CombinedSource2D::new(vec![&normal]);
        assert!(!combined_normal.is_stiff());

        let combined_stiff = CombinedSource2D::new(vec![&normal, &stiff]);
        assert!(combined_stiff.is_stiff());
    }
}

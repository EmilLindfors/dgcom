//! Multi-boundary condition dispatcher for 2D shallow water equations.
//!
//! This module provides a way to apply different boundary conditions to
//! different parts of the domain boundary based on their `BoundaryTag`.
//!
//! # Example
//!
//! ```ignore
//! use dg::boundary::{MultiBoundaryCondition2D, Reflective2D, Flather2D, Discharge2D};
//! use dg::mesh::BoundaryTag;
//!
//! // Create individual BCs
//! let wall = Reflective2D::new();
//! let tidal = Flather2D::new(|x, y, t| 0.5 * (omega * t).sin(), 10.0, (0.0, 0.0), 1e-6);
//! let river = Discharge2D::new(|_x, _y, _t| 100.0);  // 100 m²/s
//!
//! // Create multi-BC dispatcher with wall as default
//! let multi_bc = MultiBoundaryCondition2D::new(&wall)
//!     .with_tidal(&tidal)
//!     .with_river(&river);
//!
//! // Use in RHS config
//! let config = SWE2DRhsConfig::new(&equation, &multi_bc);
//! ```

use crate::boundary::{BCContext2D, BoundaryState, DeviceBoundary2D, SWEBoundaryCondition2D};
use crate::mesh::BoundaryTag;
use crate::solver::SWEState2D;

/// Multi-boundary condition dispatcher.
///
/// Routes boundary condition evaluation to different implementations
/// based on the `BoundaryTag` in the context.
///
/// # Design
///
/// The dispatcher holds references to different BC implementations for
/// each boundary type. When `ghost_state` is called, it checks the
/// `boundary_tag` in the context and dispatches to the appropriate BC.
///
/// If no tag is present or no specific BC is registered for that tag,
/// the default BC is used. It holds references only, so a copy is cheap (the
/// 3D model's barotropic module takes its boundary condition by value).
#[derive(Clone)]
pub struct MultiBoundaryCondition2D<'a> {
    /// Default BC for untagged or unknown boundaries
    default_bc: &'a dyn SWEBoundaryCondition2D,
    /// BC for Wall boundaries
    wall_bc: Option<&'a dyn SWEBoundaryCondition2D>,
    /// BC for Open boundaries
    open_bc: Option<&'a dyn SWEBoundaryCondition2D>,
    /// BC for TidalForcing boundaries
    tidal_bc: Option<&'a dyn SWEBoundaryCondition2D>,
    /// BC for River boundaries
    river_bc: Option<&'a dyn SWEBoundaryCondition2D>,
    /// BCs for Custom boundaries (indexed by custom ID)
    custom_bcs: Vec<(u32, &'a dyn SWEBoundaryCondition2D)>,
}

impl<'a> MultiBoundaryCondition2D<'a> {
    /// Create a new multi-BC dispatcher with the given default BC.
    ///
    /// The default BC is used for boundaries without a tag or for
    /// tag types that don't have a specific BC registered.
    pub fn new(default_bc: &'a dyn SWEBoundaryCondition2D) -> Self {
        Self {
            default_bc,
            wall_bc: None,
            open_bc: None,
            tidal_bc: None,
            river_bc: None,
            custom_bcs: Vec::new(),
        }
    }

    /// Set the BC for Wall boundaries.
    pub fn with_wall(mut self, bc: &'a dyn SWEBoundaryCondition2D) -> Self {
        self.wall_bc = Some(bc);
        self
    }

    /// Set the BC for Open boundaries.
    pub fn with_open(mut self, bc: &'a dyn SWEBoundaryCondition2D) -> Self {
        self.open_bc = Some(bc);
        self
    }

    /// Set the BC for TidalForcing boundaries.
    pub fn with_tidal(mut self, bc: &'a dyn SWEBoundaryCondition2D) -> Self {
        self.tidal_bc = Some(bc);
        self
    }

    /// Set the BC for River boundaries.
    pub fn with_river(mut self, bc: &'a dyn SWEBoundaryCondition2D) -> Self {
        self.river_bc = Some(bc);
        self
    }

    /// Set a BC for a Custom boundary with a specific ID.
    pub fn with_custom(mut self, id: u32, bc: &'a dyn SWEBoundaryCondition2D) -> Self {
        // Remove any existing BC for this ID
        self.custom_bcs
            .retain(|(existing_id, _)| *existing_id != id);
        self.custom_bcs.push((id, bc));
        self
    }

    /// Get the BC for a given boundary tag.
    fn bc_for_tag(&self, tag: &BoundaryTag) -> &dyn SWEBoundaryCondition2D {
        match tag {
            BoundaryTag::Wall => self.wall_bc.unwrap_or(self.default_bc),
            BoundaryTag::Open => self.open_bc.unwrap_or(self.default_bc),
            BoundaryTag::TidalForcing => self.tidal_bc.unwrap_or(self.default_bc),
            BoundaryTag::River => self.river_bc.unwrap_or(self.default_bc),
            BoundaryTag::Custom(id) => self
                .custom_bcs
                .iter()
                .find(|(i, _)| i == id)
                .map(|(_, bc)| *bc)
                .unwrap_or(self.default_bc),
            // Periodic boundaries shouldn't reach BC evaluation, use default
            BoundaryTag::Periodic(_) => self.default_bc,
            // Dirichlet and Neumann use default (or could be extended later)
            BoundaryTag::Dirichlet => self.default_bc,
            BoundaryTag::Neumann => self.default_bc,
        }
    }

    /// The BC for the boundary tag of `ctx` (the default without a tag).
    fn bc_for_context(&self, ctx: &BCContext2D) -> &dyn SWEBoundaryCondition2D {
        match &ctx.boundary_tag {
            Some(tag) => self.bc_for_tag(tag),
            None => self.default_bc,
        }
    }
}

impl<'a> SWEBoundaryCondition2D for MultiBoundaryCondition2D<'a> {
    fn ghost_state(&self, ctx: &BCContext2D) -> SWEState2D {
        self.bc_for_context(ctx).ghost_state(ctx)
    }

    fn boundary_state(&self, ctx: &BCContext2D) -> BoundaryState {
        self.bc_for_context(ctx).boundary_state(ctx)
    }

    fn name(&self) -> &'static str {
        "multi_bc_2d"
    }

    fn is_wall(&self, tag: Option<BoundaryTag>) -> Option<bool> {
        match &tag {
            Some(t) => self.bc_for_tag(t).is_wall(tag),
            None => self.default_bc.is_wall(tag),
        }
    }

    fn device_form(&self, tag: Option<BoundaryTag>) -> Option<DeviceBoundary2D<'_>> {
        match &tag {
            Some(t) => self.bc_for_tag(t).device_form(tag),
            None => self.default_bc.device_form(tag),
        }
    }

    fn allows_inflow(&self) -> bool {
        // Conservative: allow if any component allows
        true
    }

    fn allows_outflow(&self) -> bool {
        // Conservative: allow if any component allows
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::Reflective2D;

    struct MockBC {
        name: &'static str,
        h_value: f64,
    }

    impl SWEBoundaryCondition2D for MockBC {
        fn ghost_state(&self, _ctx: &BCContext2D) -> SWEState2D {
            SWEState2D::new(self.h_value, 0.0, 0.0)
        }

        fn name(&self) -> &'static str {
            self.name
        }
    }

    fn make_context_with_tag(tag: BoundaryTag) -> BCContext2D {
        BCContext2D::with_tag(
            0.0,
            (0.0, 0.0),
            SWEState2D::new(10.0, 0.0, 0.0),
            0.0,
            (1.0, 0.0),
            9.81,
            1e-6,
            tag,
        )
    }

    fn make_context_no_tag() -> BCContext2D {
        BCContext2D::new(
            0.0,
            (0.0, 0.0),
            SWEState2D::new(10.0, 0.0, 0.0),
            0.0,
            (1.0, 0.0),
            9.81,
            1e-6,
        )
    }

    #[test]
    fn test_default_bc_used_when_no_tag() {
        let default = MockBC {
            name: "default",
            h_value: 1.0,
        };
        let multi = MultiBoundaryCondition2D::new(&default);

        let ctx = make_context_no_tag();
        let ghost = multi.ghost_state(&ctx);

        assert!((ghost.h - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_wall_bc_dispatch() {
        let default = MockBC {
            name: "default",
            h_value: 1.0,
        };
        let wall = MockBC {
            name: "wall",
            h_value: 2.0,
        };
        let multi = MultiBoundaryCondition2D::new(&default).with_wall(&wall);

        let ctx = make_context_with_tag(BoundaryTag::Wall);
        let ghost = multi.ghost_state(&ctx);

        assert!((ghost.h - 2.0).abs() < 1e-14);
    }

    #[test]
    fn test_tidal_bc_dispatch() {
        let default = MockBC {
            name: "default",
            h_value: 1.0,
        };
        let tidal = MockBC {
            name: "tidal",
            h_value: 3.0,
        };
        let multi = MultiBoundaryCondition2D::new(&default).with_tidal(&tidal);

        let ctx = make_context_with_tag(BoundaryTag::TidalForcing);
        let ghost = multi.ghost_state(&ctx);

        assert!((ghost.h - 3.0).abs() < 1e-14);
    }

    #[test]
    fn test_river_bc_dispatch() {
        let default = MockBC {
            name: "default",
            h_value: 1.0,
        };
        let river = MockBC {
            name: "river",
            h_value: 4.0,
        };
        let multi = MultiBoundaryCondition2D::new(&default).with_river(&river);

        let ctx = make_context_with_tag(BoundaryTag::River);
        let ghost = multi.ghost_state(&ctx);

        assert!((ghost.h - 4.0).abs() < 1e-14);
    }

    #[test]
    fn test_open_bc_dispatch() {
        let default = MockBC {
            name: "default",
            h_value: 1.0,
        };
        let open = MockBC {
            name: "open",
            h_value: 5.0,
        };
        let multi = MultiBoundaryCondition2D::new(&default).with_open(&open);

        let ctx = make_context_with_tag(BoundaryTag::Open);
        let ghost = multi.ghost_state(&ctx);

        assert!((ghost.h - 5.0).abs() < 1e-14);
    }

    #[test]
    fn test_custom_bc_dispatch() {
        let default = MockBC {
            name: "default",
            h_value: 1.0,
        };
        let custom1 = MockBC {
            name: "custom1",
            h_value: 10.0,
        };
        let custom2 = MockBC {
            name: "custom2",
            h_value: 20.0,
        };
        let multi = MultiBoundaryCondition2D::new(&default)
            .with_custom(1, &custom1)
            .with_custom(2, &custom2);

        let ctx1 = make_context_with_tag(BoundaryTag::Custom(1));
        let ghost1 = multi.ghost_state(&ctx1);
        assert!((ghost1.h - 10.0).abs() < 1e-14);

        let ctx2 = make_context_with_tag(BoundaryTag::Custom(2));
        let ghost2 = multi.ghost_state(&ctx2);
        assert!((ghost2.h - 20.0).abs() < 1e-14);

        // Unknown custom ID falls back to default
        let ctx3 = make_context_with_tag(BoundaryTag::Custom(99));
        let ghost3 = multi.ghost_state(&ctx3);
        assert!((ghost3.h - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_fallback_to_default() {
        let default = MockBC {
            name: "default",
            h_value: 1.0,
        };
        let wall = MockBC {
            name: "wall",
            h_value: 2.0,
        };
        // Only set wall BC
        let multi = MultiBoundaryCondition2D::new(&default).with_wall(&wall);

        // River tag should fall back to default since no river BC set
        let ctx = make_context_with_tag(BoundaryTag::River);
        let ghost = multi.ghost_state(&ctx);

        assert!((ghost.h - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_chained_builder() {
        let default = MockBC {
            name: "default",
            h_value: 1.0,
        };
        let wall = MockBC {
            name: "wall",
            h_value: 2.0,
        };
        let tidal = MockBC {
            name: "tidal",
            h_value: 3.0,
        };
        let river = MockBC {
            name: "river",
            h_value: 4.0,
        };
        let open = MockBC {
            name: "open",
            h_value: 5.0,
        };

        let multi = MultiBoundaryCondition2D::new(&default)
            .with_wall(&wall)
            .with_tidal(&tidal)
            .with_river(&river)
            .with_open(&open);

        // Test all tags route correctly
        assert!(
            (multi
                .ghost_state(&make_context_with_tag(BoundaryTag::Wall))
                .h
                - 2.0)
                .abs()
                < 1e-14
        );
        assert!(
            (multi
                .ghost_state(&make_context_with_tag(BoundaryTag::TidalForcing))
                .h
                - 3.0)
                .abs()
                < 1e-14
        );
        assert!(
            (multi
                .ghost_state(&make_context_with_tag(BoundaryTag::River))
                .h
                - 4.0)
                .abs()
                < 1e-14
        );
        assert!(
            (multi
                .ghost_state(&make_context_with_tag(BoundaryTag::Open))
                .h
                - 5.0)
                .abs()
                < 1e-14
        );
    }

    #[test]
    fn test_with_reflective_default() {
        let reflective = Reflective2D::new();
        let multi = MultiBoundaryCondition2D::new(&reflective);

        // With no tag and reflective default, should mirror velocity
        let ctx = BCContext2D::new(
            0.0,
            (0.0, 0.0),
            SWEState2D::new(10.0, 20.0, 5.0), // h=10, hu=20, hv=5
            0.0,
            (1.0, 0.0), // Normal in +x
            9.81,
            1e-6,
        );

        let ghost = multi.ghost_state(&ctx);

        // Reflective BC should reverse normal velocity (u)
        // u_interior = 2.0, v_interior = 0.5
        // u_ghost = u - 2*u*nx*nx = 2 - 2*2*1*1 = -2
        // v_ghost = v - 2*u*nx*ny = 0.5 - 0 = 0.5
        assert!((ghost.h - 10.0).abs() < 1e-14);
        assert!((ghost.hu / ghost.h + 2.0).abs() < 1e-10); // u_ghost = -2
        assert!((ghost.hv / ghost.h - 0.5).abs() < 1e-10); // v_ghost = 0.5
    }

    #[test]
    fn test_name() {
        let reflective = Reflective2D::new();
        let multi = MultiBoundaryCondition2D::new(&reflective);
        assert_eq!(multi.name(), "multi_bc_2d");
    }
}

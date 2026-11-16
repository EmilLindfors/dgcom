//! The 2D physics as a device runs it (P2.7, `notes/gpu.md`).
//!
//! A device cannot call a trait object, so a device backend (`gpu/`) runs a
//! closed set of forms, as data: each boundary condition, source term and
//! friction law that has a device form says what it is
//! ([`SWEBoundaryCondition2D::device_form`], [`SourceTerm2D::device_form`],
//! [`BottomFriction2D::device_form`]), and [`SWEPhysics2D::device_form`]
//! gathers them, or refuses the device naming what has none. The pointwise
//! physics the forms stand for is dgcom's own, which the device calls.
//!
//! [`SourceTerm2D::device_form`]: crate::source::SourceTerm2D::device_form
//! [`BottomFriction2D::device_form`]: crate::source::BottomFriction2D::device_form

use std::sync::Arc;

use super::SWEPhysics2D;
use crate::boundary::{DeviceBoundary2D, SWEBoundaryCondition2D};
use crate::equations::ShallowWater2D;
use crate::mesh::{Bathymetry2D, BoundaryTag, Mesh2D};
use crate::operators::{DGOperators2D, GeometricFactors2D};
use crate::solver::{SWEFormulation2D, StandardLimiter2D, WetDryConfig};
use crate::source::{DeviceFriction2D, DeviceSource2D};

/// [`SWEPhysics2D`] as a device runs it: the mesh and operators, the bed, the
/// equation, the split form, the wet/dry treatment, the friction law, the
/// source and the boundary condition of each boundary tag of the mesh.
pub struct SWEDeviceForm2D<'a> {
    /// The mesh
    pub mesh: &'a Arc<Mesh2D>,
    /// DG operators
    pub ops: &'a Arc<DGOperators2D>,
    /// Geometric factors
    pub geom: &'a Arc<GeometricFactors2D>,
    /// The bed (flat at 0 without one)
    pub bathymetry: Option<&'a Bathymetry2D>,
    /// Gravity and the velocity threshold `h_min`
    pub equation: &'a ShallowWater2D,
    /// `EntropyStable` or `WetDry`
    pub formulation: SWEFormulation2D,
    /// `WetDry`'s subcell depth and dry-node depth
    /// (`SWE2DRhsConfig::subcells`)
    pub subcells: Option<(f64, f64)>,
    /// The wet/dry correction after every stage, the thin-layer relaxation
    /// and the positivity-bounded step (the positivity limiter of a wet/dry
    /// run is the redundant one, at or below its `h_dry`)
    pub wet_dry: Option<WetDryConfig>,
    /// Bottom friction, point-implicit in every stage
    pub friction: Option<DeviceFriction2D>,
    /// The source term
    pub source: Option<DeviceSource2D>,
    /// The boundary condition of each boundary tag of the mesh's boundary
    /// faces
    pub boundaries: Vec<(Option<BoundaryTag>, DeviceBoundary2D<'a>)>,
    /// Local time stepping: the depth below which a node takes its element
    /// out of the line-local form (`LocalTimeStepping::line_local`), `None`
    /// if no element is line-local
    pub line_local_depth: Option<f64>,
    /// The `h_dry` of the positivity bound on the element steps and the
    /// coarse step (`LocalTimeStepping::coarse_step_bound`), `None` without
    /// wetting and drying
    pub positivity_h_dry: Option<f64>,
}

impl<'a> SWEDeviceForm2D<'a> {
    /// The condition of the boundary faces tagged `tag`.
    pub fn boundary(&self, tag: Option<BoundaryTag>) -> Option<DeviceBoundary2D<'a>> {
        self.boundaries
            .iter()
            .find_map(|&(t, form)| (t == tag).then_some(form))
    }
}

impl<BC: SWEBoundaryCondition2D> SWEPhysics2D<BC> {
    /// This module as a device runs it, or what it has that a device does
    /// not run: a formulation other than the split forms `EntropyStable`
    /// and `WetDry`, a limiter beyond the wet/dry run's redundant
    /// positivity pass, horizontal viscosity, net cages, and any friction
    /// law, source term or boundary condition without a device form.
    pub fn device_form(&self) -> Result<SWEDeviceForm2D<'_>, String> {
        let config = self.rhs_config();
        if !matches!(
            self.formulation,
            SWEFormulation2D::EntropyStable | SWEFormulation2D::WetDry
        ) {
            return Err(format!(
                "the {:?} formulation (EntropyStable and WetDry run on a device)",
                self.formulation
            ));
        }
        if !(matches!(self.limiter, StandardLimiter2D::None) || self.positivity_pass_is_redundant())
        {
            return Err(format!("the limiter {:?}", self.limiter));
        }
        if self.viscosity.is_some() {
            return Err("horizontal viscosity".into());
        }
        if self.cages.is_some() {
            return Err("net-cage drag".into());
        }
        let friction = match &self.friction {
            Some(law) => Some(
                law.device_form()
                    .ok_or("a bottom-friction law without a device form (Manning has one)")?,
            ),
            None => None,
        };
        let source = match &self.source {
            Some(source) => Some(
                source
                    .device_form()
                    .ok_or_else(|| format!("the source term {}", source.name()))?,
            ),
            None => None,
        };
        let mut boundaries: Vec<(Option<BoundaryTag>, DeviceBoundary2D<'_>)> = Vec::new();
        for edge in self.mesh.edges.iter().filter(|e| e.right.is_none()) {
            let tag = edge.boundary_tag;
            if boundaries.iter().any(|&(t, _)| t == tag) {
                continue;
            }
            let form = self.bc.device_form(tag).ok_or_else(|| {
                format!(
                    "the boundary condition of the faces tagged {tag:?} ({})",
                    self.bc.name()
                )
            })?;
            boundaries.push((tag, form));
        }
        Ok(SWEDeviceForm2D {
            mesh: &self.mesh,
            ops: &self.ops,
            geom: &self.geom,
            bathymetry: self.bathymetry.as_deref(),
            equation: &self.equation,
            formulation: self.formulation,
            subcells: config.subcells(),
            wet_dry: self.wet_dry.clone(),
            friction,
            source,
            boundaries,
            line_local_depth: self.line_local_depth(),
            positivity_h_dry: self.positivity_bound(None).map(|bound| bound.h_dry),
        })
    }
}

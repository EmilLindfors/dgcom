//! `dgcom mesh`: a case's mesh and bed, written to inspect before a run.
//!
//! The mesh goes to Gmsh (`<name>.msh`, with the boundary tags as physical
//! groups, so Gmsh shows which faces are open) and to VTK (`<name>.vtu`) with
//! the bed and the depth at rest at the nodes and, per element, its diameter
//! and the time step it allows at rest: the elements that limit the run's step
//! are the ones to look at. With local time stepping the VTK also has each
//! element's level. A separate wave mesh is written beside it
//! (`<name>_waves.*`).

use std::fs;
use std::path::{Path, PathBuf};

use super::super::{CaseError, CaseFile, Domain};
use super::{Reflective2D, boundary_conditions, still_water};
use crate::io::write_vtk_fields;
use crate::mesh::{Bathymetry2D, Mesh2D, write_gmsh_mesh};
use crate::operators::DGOperators2D;
use crate::physics::PhysicsModule;
use crate::time::{IntegratorInfo, MultirateStepper};
use crate::types::ElementIndex;

/// What [`CaseFile::write_mesh`] wrote and found.
#[derive(Clone, Debug, Default)]
pub struct MeshReport {
    /// The files written
    pub files: Vec<PathBuf>,
    /// The 2D run's element time steps at rest (s), the smallest and the
    /// largest finite one; `None` for a 3D case (whose columns' steps are
    /// not the 2D module's) or a dry mesh
    pub element_dt_s: Option<(f64, f64)>,
    /// The step the 2D run would take at rest (s): the smallest element's,
    /// or with local time stepping the coarse step, at most `dt_max_s`
    pub step_s: Option<f64>,
    /// With local time stepping, the number of elements at each level
    /// (level ℓ takes 2^ℓ steps per coarse step)
    pub elements_per_level: Vec<usize>,
    /// With local time stepping, the element evaluations per second of
    /// model time at rest ([`MultirateStepper::planned_work`]): every
    /// element at its own step, at its level's step, and as scheduled
    pub lts_work: Option<(f64, f64, f64)>,
}

impl CaseFile {
    /// Write `domain`'s mesh and bed (built by [`Self::build_domain`]) to
    /// the `mesh` directory of the case's output directory.
    pub fn write_mesh(&self, domain: &Domain) -> Result<MeshReport, CaseError> {
        let case = &self.case;
        let dir = self.output_dir().join("mesh");
        let io = |what: &Path, e: &dyn std::fmt::Display| {
            CaseError::Io(format!("{}: {e}", what.display()))
        };
        fs::create_dir_all(&dir).map_err(|e| io(&dir, &e))?;
        let name = &case.run.name;
        let mut report = MeshReport::default();

        let q = still_water(case, domain);
        let mut element_fields: Vec<(&str, Vec<f64>)> = vec![(
            "diameter_m",
            ElementIndex::iter(domain.mesh.n_elements)
                .map(|k| domain.mesh.element_diameter(k))
                .collect(),
        )];
        if case.three_d.is_none() {
            // The 2D run's element steps at rest, as its first step takes them
            let clock = case.clock()?;
            let opens = self.open_boundaries(domain, clock.as_ref())?;
            let wall = Reflective2D::new();
            let physics = self.physics(domain, boundary_conditions(&wall, &opens));
            let local = physics
                .local_time_stepping()
                .expect("the 2D physics steps locally");
            let numerics = &case.numerics;
            let cfl = numerics.cfl_at(case.mesh.order);
            let scheme = numerics
                .integrator
                .standard()
                .ssp_scheme()
                .expect("the case's integrators are SSP");
            let mut dt = vec![0.0; domain.mesh.n_elements];
            local.element_dt(&q, cfl, scheme, &mut dt);
            let finite = dt.iter().copied().filter(|d| d.is_finite());
            let (lo, hi) = finite.fold((f64::INFINITY, 0.0_f64), |(lo, hi), d| {
                (lo.min(d), hi.max(d))
            });
            if lo.is_finite() {
                report.element_dt_s = Some((lo, hi));
                report.step_s = Some(numerics.dt_max_s.map_or(lo, |cap| lo.min(cap)));
            }
            let levels = numerics.local_time_stepping_levels;
            if levels > 0 {
                let mut stepper = MultirateStepper::new(levels, scheme);
                let coarse =
                    stepper.assign_levels(local, &domain.mesh, &q, cfl, numerics.dt_max_s, None);
                if coarse.is_finite() {
                    report.step_s = Some(coarse);
                    report.lts_work = Some(stepper.planned_work(coarse));
                }
                let level = stepper.levels();
                let finest = level.iter().copied().max().unwrap_or(0) as usize;
                report.elements_per_level = vec![0; finest + 1];
                for &l in level {
                    report.elements_per_level[l as usize] += 1;
                }
                element_fields.push(("lts_level", level.iter().map(|&l| l as f64).collect()));
            }
            // −1 where nothing limits the step (a dry element): VTK readers
            // take no infinity
            let dt = dt
                .iter()
                .map(|&d| if d.is_finite() { d } else { -1.0 })
                .collect();
            element_fields.push(("dt_s", dt));
        }

        let fields: Vec<(&str, &[f64])> = element_fields
            .iter()
            .map(|(n, v)| (*n, v.as_slice()))
            .collect();
        write_files(
            &dir,
            name,
            &domain.mesh,
            &domain.ops,
            &domain.bathymetry,
            &[("depth_at_rest_m", &q.data[0])],
            &fields,
            &mut report.files,
        )?;
        if let (Some(spec), Some(waves)) = (&case.waves, &domain.waves)
            && spec.mesh.is_some()
        {
            let diameter: Vec<f64> = ElementIndex::iter(waves.mesh.n_elements)
                .map(|k| waves.mesh.element_diameter(k))
                .collect();
            write_files(
                &dir,
                &format!("{name}_waves"),
                &waves.mesh,
                &waves.ops,
                &waves.bathymetry,
                &[],
                &[("diameter_m", &diameter)],
                &mut report.files,
            )?;
        }
        Ok(report)
    }
}

/// `<stem>.msh` and `<stem>.vtu` in `dir`, the VTK with the bed and
/// `node_fields` at the nodes and `element_fields`.
#[allow(clippy::too_many_arguments)]
fn write_files(
    dir: &Path,
    stem: &str,
    mesh: &Mesh2D,
    ops: &DGOperators2D,
    bed: &Bathymetry2D,
    node_fields: &[(&str, &[f64])],
    element_fields: &[(&str, &[f64])],
    files: &mut Vec<PathBuf>,
) -> Result<(), CaseError> {
    let io =
        |what: &Path, e: &dyn std::fmt::Display| CaseError::Io(format!("{}: {e}", what.display()));
    let msh = dir.join(format!("{stem}.msh"));
    write_gmsh_mesh(mesh, &msh).map_err(|e| io(&msh, &e))?;
    files.push(msh);
    let vtu = dir.join(format!("{stem}.vtu"));
    let mut nodes: Vec<(&str, &[f64])> = vec![("bed_m", &bed.data)];
    nodes.extend_from_slice(node_fields);
    write_vtk_fields(&vtu, mesh, ops, &nodes, element_fields, None).map_err(|e| io(&vtu, &e))?;
    files.push(vtu);
    Ok(())
}

//! Building and running a case: the 2D production path (`Simulation` +
//! `SSPRK3` + `SWEPhysics2D`) with the wet/dry defaults of the Frøya runs:
//! the positivity limiter, the `WetDry` scheme and point-implicit bed
//! friction.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use super::schema::{BoundaryVelocity, Coriolis, Friction, SideTag};
use super::{CaseError, CaseFile};
use crate::boundary::{
    CharacteristicOBC, ExternalStateProvider, HarmonicTide, MultiBoundaryCondition2D, Reflective2D,
    SWEBoundaryCondition2D, StillWater, TidalConstituent,
};
use crate::equations::ShallowWater2D;
use crate::io::{SnapshotWriter, write_vtk_swe};
use crate::mesh::{Bathymetry2D, Mesh2D, PointLocator2D, read_gmsh_mesh};
use crate::operators::{DGOperators2D, GeometricFactors2D};
use crate::physics::{PhysicsBuilder, SWEPhysics2D};
use crate::simulation::Simulation;
use crate::solver::{
    Probe2D, SWESolution2D, SWEState2D, StandardLimiter2D, WetDryConfig, total_mass_2d,
};
use crate::source::{
    ChezyFriction2D, CoriolisSource2D, HorizontalViscosity2D, ManningFriction2D, coriolis_parameter,
};
use crate::tides::{canonical_name, constituent_period};
use crate::time::SSPRK3;
use crate::types::ElementIndex;

/// Depth below which the velocity is reported as 0 (m).
const H_VELOCITY: f64 = WetDryConfig::DEFAULT_H_DRY;

/// The discretized domain of a case.
pub struct Domain {
    /// The mesh
    pub mesh: Arc<Mesh2D>,
    /// The reference element's operators
    pub ops: Arc<DGOperators2D>,
    /// The geometric factors
    pub geom: Arc<GeometricFactors2D>,
    /// The bed elevation B (negative under water)
    pub bathymetry: Arc<Bathymetry2D>,
    /// The boundary tags the mesh uses, walls left out
    pub open_tags: Vec<SideTag>,
}

impl Domain {
    /// Positions of the nodes, element-major.
    pub fn node_positions(&self) -> Vec<[f64; 2]> {
        ElementIndex::iter(self.mesh.n_elements)
            .flat_map(|k| {
                (0..self.ops.n_nodes).map(move |i| {
                    self.mesh
                        .reference_to_physical(k, self.ops.nodes_r[i], self.ops.nodes_s[i])
                })
            })
            .collect()
    }
}

/// Where a run is, at each output.
#[derive(Clone, Copy, Debug)]
pub struct Progress {
    /// Model time (s)
    pub time_s: f64,
    /// End of the run (s)
    pub end_s: f64,
    /// Lowest and highest surface elevation over the wet nodes (m)
    pub eta_range: (f64, f64),
    /// Largest depth-averaged speed (m/s)
    pub max_speed: f64,
}

/// What a station recorded.
#[derive(Clone, Debug, Default)]
pub struct StationSeries {
    /// The station's name
    pub name: String,
    /// Output times (s)
    pub time_s: Vec<f64>,
    /// Surface elevation (m)
    pub eta_m: Vec<f64>,
    /// Depth-averaged velocity along x (m/s)
    pub u_m_s: Vec<f64>,
    /// Depth-averaged velocity along y (m/s)
    pub v_m_s: Vec<f64>,
}

/// The outcome of a run.
#[derive(Clone, Debug)]
pub struct RunReport {
    /// Model time reached (s)
    pub final_time_s: f64,
    /// Time steps taken
    pub n_steps: usize,
    /// Smallest and largest time step (s)
    pub dt_range_s: (f64, f64),
    /// Wall-clock time (s)
    pub wall_time_s: f64,
    /// Water volume at the start and the end (m³)
    pub volume_m3: (f64, f64),
    /// The stations' records
    pub stations: Vec<StationSeries>,
}

impl CaseFile {
    /// The mesh, operators and bed of the case, with its boundary tags
    /// checked against the `[[boundary]]` entries.
    pub fn build_domain(&self) -> Result<Domain, CaseError> {
        let case = &self.case;
        let mesh = if let Some(r) = &case.mesh.rectangle {
            Mesh2D::uniform_rectangle_with_sides(
                r.x_m[0],
                r.x_m[1],
                r.y_m[0],
                r.y_m[1],
                r.nx,
                r.ny,
                [r.south, r.east, r.north, r.west].map(SideTag::boundary_tag),
            )
        } else {
            let path = self.resolve(case.mesh.gmsh.as_ref().expect("validated"));
            read_gmsh_mesh(&path)
                .map_err(|e| CaseError::Io(format!("mesh.gmsh {}: {e}", path.display())))?
        };
        let ops = DGOperators2D::new(case.mesh.order);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let bed = &case.bathymetry;
        let bathymetry = match (bed.depth_m, &bed.ramp) {
            (Some(depth), _) => Bathymetry2D::from_function(&mesh, &ops, &geom, |_, _| -depth),
            (None, Some(ramp)) => {
                Bathymetry2D::from_function(&mesh, &ops, &geom, |x, y| -ramp.depth(x, y))
            }
            (None, None) => unreachable!("validated"),
        };

        // The tags on the mesh's boundary faces
        let mut open_tags = Vec::new();
        for k in ElementIndex::iter(mesh.n_elements) {
            for face in 0..4 {
                let Some(tag) = mesh.boundary_tag(k, face) else {
                    continue;
                };
                let side = SideTag::from_boundary_tag(tag).ok_or_else(|| {
                    CaseError::Invalid(format!(
                        "the mesh has boundary faces tagged {tag:?}; a case names wall, open, \
                         tidal and river faces only"
                    ))
                })?;
                if side != SideTag::Wall && !open_tags.contains(&side) {
                    open_tags.push(side);
                }
            }
        }
        for tag in &open_tags {
            if !case.boundaries.iter().any(|b| b.tag == *tag) {
                return Err(CaseError::Invalid(format!(
                    "the mesh has {tag:?} boundary faces but the case no [[boundary]] with \
                     tag = {:?}",
                    tag_name(*tag)
                )));
            }
        }
        for b in &case.boundaries {
            if !open_tags.contains(&b.tag) {
                return Err(CaseError::Invalid(format!(
                    "[[boundary]] tag = {:?}: the mesh has no faces with that tag",
                    tag_name(b.tag)
                )));
            }
        }
        Ok(Domain {
            mesh: Arc::new(mesh),
            ops: Arc::new(ops),
            geom: Arc::new(geom),
            bathymetry: Arc::new(bathymetry),
            open_tags,
        })
    }

    /// Run the case, writing its output and reporting `progress` at every
    /// output time.
    pub fn run(&self, progress: &mut dyn FnMut(&Progress)) -> Result<RunReport, CaseError> {
        let case = &self.case;
        let domain = self.build_domain()?;
        let clock = case.clock()?;
        let t_end = 3600.0 * case.run.duration_hours;
        let interval = 60.0 * case.output.interval_minutes;

        // Open boundaries: one characteristic condition per tag
        let opens: Vec<(SideTag, Box<dyn SWEBoundaryCondition2D>)> = case
            .boundaries
            .iter()
            .map(|b| {
                let bc: Box<dyn SWEBoundaryCondition2D> = match (&b.still, &b.tide) {
                    (Some(still), _) => Box::new(with_velocity(
                        CharacteristicOBC::new(StillWater::at(still.eta_m)),
                        b.velocity,
                    )),
                    (None, Some(tide)) => {
                        let constituents = tide
                            .constituents
                            .iter()
                            .map(|c| {
                                let name = canonical_name(&c.name).expect("validated");
                                let period = constituent_period(name).expect("validated");
                                // η = A cos(ωt − G)
                                TidalConstituent::new(
                                    name,
                                    c.amplitude_m,
                                    period,
                                    -c.phase_deg.to_radians(),
                                )
                            })
                            .collect();
                        let mut harmonic = HarmonicTide::new(constituents)
                            .with_mean_elevation(tide.mean_m)
                            .with_ramp_up(3600.0 * tide.ramp_hours);
                        if let Some(clock) = &clock {
                            harmonic = harmonic.with_nodal_corrections(clock);
                        }
                        Box::new(with_velocity(CharacteristicOBC::new(harmonic), b.velocity))
                    }
                    (None, None) => unreachable!("validated"),
                };
                (b.tag, bc)
            })
            .collect();
        let wall = Reflective2D::new();
        let mut bc = MultiBoundaryCondition2D::new(&wall);
        for (tag, open) in &opens {
            bc = match tag {
                SideTag::Open => bc.with_open(open.as_ref()),
                SideTag::Tidal => bc.with_tidal(open.as_ref()),
                SideTag::River => bc.with_river(open.as_ref()),
                SideTag::Wall => unreachable!("validated"),
            };
        }
        let physics = self.physics(&domain, bc);

        // Still water at the initial level
        let eta0 = case.initial.eta_m;
        let mut q = SWESolution2D::new(domain.mesh.n_elements, domain.ops.n_nodes);
        for k in ElementIndex::iter(domain.mesh.n_elements) {
            for i in 0..domain.ops.n_nodes {
                let h = (eta0 - domain.bathymetry.get(k, i)).max(0.0);
                q.set_state(k, i, SWEState2D::new(h, 0.0, 0.0));
            }
        }

        let mut output = Output::create(self, &domain, clock.as_ref())?;
        let volume = |q: &SWESolution2D| volume(&domain, q);
        let volume_start = volume(&q);

        let stop = Arc::new(AtomicBool::new(false));
        let mut sim = Simulation::new(physics, SSPRK3)
            .with_cfl(case.numerics.cfl)
            .with_callback_interval(interval)
            .with_stop(stop.clone());
        if let Some(dt) = case.numerics.dt_max_s {
            sim = sim.with_dt_max(dt);
        }
        let mut failure = None;
        let result = sim.run_with_callback(&mut q, 0.0, t_end, |q, t| {
            if failure.is_some() {
                return;
            }
            match output.write(t, q) {
                Ok(()) => progress(&summarize(&domain, q, t, t_end)),
                Err(e) => {
                    failure = Some(e);
                    stop.store(true, Ordering::Relaxed);
                }
            }
        });
        if let Some(e) = failure {
            return Err(e);
        }
        if !result.success {
            return Err(CaseError::Run(format!(
                "the run stopped at {:.1} h: {}",
                result.final_time / 3600.0,
                result.error.as_deref().unwrap_or("unknown error")
            )));
        }
        // The last output, unless the run ended on an output time
        if output.last_time < result.final_time - 1e-6 * interval {
            output.write(result.final_time, &q)?;
        }
        Ok(RunReport {
            final_time_s: result.final_time,
            n_steps: result.n_steps,
            dt_range_s: (result.dt_min, result.dt_max),
            wall_time_s: result.wall_time,
            volume_m3: (volume_start, volume(&q)),
            stations: output.finish(),
        })
    }

    /// The 2D physics of the case on `domain` with boundary conditions `bc`.
    fn physics<BC: SWEBoundaryCondition2D>(&self, domain: &Domain, bc: BC) -> SWEPhysics2D<BC> {
        let p = &self.case.physics;
        let mut builder = PhysicsBuilder::swe_2d(
            domain.mesh.clone(),
            domain.ops.clone(),
            domain.geom.clone(),
            ShallowWater2D::new(p.gravity),
            bc,
        )
        .with_bathymetry(domain.bathymetry.clone());
        if p.wetting_drying {
            builder = builder
                .with_limiter(StandardLimiter2D::Positivity(WetDryConfig::DEFAULT_H_DRY))
                .with_wet_dry(WetDryConfig::default());
        }
        match p.coriolis {
            Some(Coriolis::FPlane(f)) => {
                builder = builder.with_source(CoriolisSource2D::f_plane(f))
            }
            Some(Coriolis::Latitude(lat)) => {
                builder = builder.with_source(CoriolisSource2D::f_plane(coriolis_parameter(lat)));
            }
            None => {}
        }
        match p.friction {
            Some(Friction::Manning(n)) => {
                builder = builder.with_implicit_friction(ManningFriction2D::new(p.gravity, n));
            }
            Some(Friction::DragCoefficient(c_d)) => {
                builder = builder.with_implicit_friction(ChezyFriction2D::new(c_d));
            }
            None => {}
        }
        if p.viscosity_m2_s > 0.0 || p.smagorinsky > 0.0 {
            builder = builder.with_viscosity(
                HorizontalViscosity2D::smagorinsky(p.smagorinsky).with_background(p.viscosity_m2_s),
            );
        }
        builder.build()
    }
}

/// `obc` with the external velocity `velocity` for level-only data.
fn with_velocity<P: ExternalStateProvider>(
    obc: CharacteristicOBC<P>,
    velocity: BoundaryVelocity,
) -> CharacteristicOBC<P> {
    match velocity {
        BoundaryVelocity::AtRest => obc,
        BoundaryVelocity::IncomingWave => obc.with_incoming_wave(),
    }
}

fn tag_name(tag: SideTag) -> &'static str {
    match tag {
        SideTag::Wall => "wall",
        SideTag::Open => "open",
        SideTag::Tidal => "tidal",
        SideTag::River => "river",
    }
}

/// Water volume ∫ h dA (m³).
fn volume(domain: &Domain, q: &SWESolution2D) -> f64 {
    total_mass_2d(q, &domain.ops, &domain.geom)
}

fn summarize(domain: &Domain, q: &SWESolution2D, t: f64, t_end: f64) -> Progress {
    let (mut lo, mut hi, mut speed) = (f64::INFINITY, f64::NEG_INFINITY, 0.0_f64);
    for k in ElementIndex::iter(domain.mesh.n_elements) {
        for i in 0..domain.ops.n_nodes {
            let s = q.get_state(k, i);
            if s.h > H_VELOCITY {
                let eta = s.h + domain.bathymetry.get(k, i);
                lo = lo.min(eta);
                hi = hi.max(eta);
                speed = speed.max(s.hu.hypot(s.hv) / s.h);
            }
        }
    }
    Progress {
        time_s: t,
        end_s: t_end,
        eta_range: (lo, hi),
        max_speed: speed,
    }
}

/// The run's writers.
struct Output<'a> {
    domain: &'a Domain,
    name: String,
    dir: Option<std::path::PathBuf>,
    snapshot: Option<SnapshotWriter>,
    vtk_frames: Option<usize>,
    #[cfg(feature = "netcdf")]
    netcdf: Option<crate::io::NetCDFWriter>,
    probes: Vec<Probe2D>,
    stations: Vec<StationSeries>,
    csv: Option<BufWriter<File>>,
    last_time: f64,
}

impl<'a> Output<'a> {
    fn create(
        file: &CaseFile,
        domain: &'a Domain,
        clock: Option<&crate::time::ModelClock>,
    ) -> Result<Self, CaseError> {
        let case = &file.case;
        let io = |what: &str, e: &dyn std::fmt::Display| CaseError::Io(format!("{what}: {e}"));
        let locator = PointLocator2D::new(&domain.mesh);
        let probes = case
            .output
            .stations
            .iter()
            .map(|s| {
                Probe2D::at(&locator, &domain.ops, [s.x_m, s.y_m]).ok_or_else(|| {
                    CaseError::Invalid(format!(
                        "station {:?} at ({}, {}) is outside the mesh",
                        s.name, s.x_m, s.y_m
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let stations = case
            .output
            .stations
            .iter()
            .map(|s| StationSeries {
                name: s.name.clone(),
                ..Default::default()
            })
            .collect();
        let dir = file.output_dir();
        fs::create_dir_all(&dir).map_err(|e| io(&dir.display().to_string(), &e))?;
        let resolved = dir.join("case.resolved.toml");
        fs::write(&resolved, case.to_toml())
            .map_err(|e| io(&resolved.display().to_string(), &e))?;
        let name = case.run.name.clone();
        let snapshot = if case.output.snapshot {
            let path = dir.join(format!("{name}.dgsnap"));
            Some(
                SnapshotWriter::create(
                    &path,
                    &domain.mesh,
                    &domain.ops,
                    &domain.bathymetry.data,
                    clock,
                    &[("source", "dgcom"), ("case", &name)],
                    H_VELOCITY,
                )
                .map_err(|e| io(&path.display().to_string(), &e))?,
            )
        } else {
            None
        };
        let vtk_frames = if case.output.vtk {
            let vtk = dir.join("vtk");
            fs::create_dir_all(&vtk).map_err(|e| io(&vtk.display().to_string(), &e))?;
            Some(0)
        } else {
            None
        };
        #[cfg(feature = "netcdf")]
        let netcdf = if case.output.netcdf {
            use crate::io::{NetCDFMeshInfo, NetCDFWriter, NetCDFWriterConfig};
            let path = dir.join(format!("{name}.nc"));
            let (x, y) = domain
                .node_positions()
                .into_iter()
                .map(|[x, y]| (x, y))
                .unzip();
            let mut config = NetCDFWriterConfig::new(path.display().to_string())
                .with_title(name.clone())
                .with_source("dgcom")
                .with_velocity(true);
            if let Some(clock) = clock {
                config = config.with_clock(*clock);
            }
            Some(
                NetCDFWriter::create(config, &NetCDFMeshInfo::from_xy(x, y))
                    .map_err(|e| io(&path.display().to_string(), &e))?,
            )
        } else {
            None
        };
        let csv = if probes.is_empty() {
            None
        } else {
            let path = dir.join("stations.csv");
            let mut csv = BufWriter::new(
                File::create(&path).map_err(|e| io(&path.display().to_string(), &e))?,
            );
            writeln!(csv, "station,time_s,eta_m,u_m_s,v_m_s,depth_m")
                .map_err(|e| io("stations.csv", &e))?;
            Some(csv)
        };
        Ok(Self {
            domain,
            name,
            dir: Some(dir),
            snapshot,
            vtk_frames,
            #[cfg(feature = "netcdf")]
            netcdf,
            probes,
            stations,
            csv,
            last_time: f64::NEG_INFINITY,
        })
    }

    fn write(&mut self, t: f64, q: &SWESolution2D) -> Result<(), CaseError> {
        let io = |what: &str, e: &dyn std::fmt::Display| CaseError::Io(format!("{what}: {e}"));
        let domain = self.domain;
        for (probe, series) in self.probes.iter().zip(&mut self.stations) {
            let s = probe.sample_swe(q, Some(&domain.bathymetry), H_VELOCITY);
            series.time_s.push(t);
            series.eta_m.push(s.eta);
            series.u_m_s.push(s.u);
            series.v_m_s.push(s.v);
            if let Some(csv) = &mut self.csv {
                writeln!(csv, "{},{t},{},{},{},{}", series.name, s.eta, s.u, s.v, s.h)
                    .map_err(|e| io("stations.csv", &e))?;
            }
        }
        if let Some(csv) = &mut self.csv {
            csv.flush().map_err(|e| io("stations.csv", &e))?;
        }
        if let Some(snapshot) = &mut self.snapshot {
            snapshot
                .write_state(t, q, &[])
                .map_err(|e| io("snapshot", &e))?;
        }
        if let (Some(frame), Some(dir)) = (&mut self.vtk_frames, &self.dir) {
            let path = dir
                .join("vtk")
                .join(format!("{}_{:04}.vtu", self.name, frame));
            write_vtk_swe(
                &path,
                &domain.mesh,
                &domain.ops,
                q,
                Some(&domain.bathymetry),
                t,
                H_VELOCITY,
            )
            .map_err(|e| io(&path.display().to_string(), &e))?;
            *frame += 1;
        }
        #[cfg(feature = "netcdf")]
        if let Some(netcdf) = &mut self.netcdf {
            let n = q.data[0].len();
            let (mut h, mut eta, mut u, mut v) = (
                Vec::with_capacity(n),
                Vec::with_capacity(n),
                Vec::with_capacity(n),
                Vec::with_capacity(n),
            );
            for k in ElementIndex::iter(domain.mesh.n_elements) {
                for i in 0..domain.ops.n_nodes {
                    let s = q.get_state(k, i);
                    let wet = s.h > H_VELOCITY;
                    h.push(s.h);
                    eta.push(s.h + domain.bathymetry.get(k, i));
                    u.push(if wet { s.hu / s.h } else { 0.0 });
                    v.push(if wet { s.hv / s.h } else { 0.0 });
                }
            }
            netcdf
                .write_timestep(t, &h, &eta, Some(&u), Some(&v))
                .map_err(|e| io("NetCDF", &e))?;
        }
        self.last_time = t;
        Ok(())
    }

    /// The stations' records; the writers flush every frame as it is written.
    fn finish(self) -> Vec<StationSeries> {
        self.stations
    }
}

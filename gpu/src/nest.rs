//! The nesting gate (P2.7 Phase 2): a run nested in a parent ocean model,
//! its boundary from the parent's series per node (`OceanModelState`) and a
//! relaxation band towards it (`NestingRelaxation2D`), on the CPU and on the
//! GPU, with global steps and local time stepping, compared bit for bit.
//!
//! The parent is made up and held in memory (no NetCDF needed): a tide
//! entering a channel and a coastal current, hourly on a 800 m grid, its
//! depth varying so that the transport scaling acts. The channel is
//! `cases/tidal_channel`'s (40 × 4 km, its bed from 30 m down to 2 m above
//! mean sea level, drying flats), at 500 m, placed at 63.5°N 8°E, its west
//! side open; the physics as the case runner builds a nested case (the
//! wet/dry stage, Coriolis, Manning, the bed blended to the parent's across
//! the band, the band's relaxation after the Coriolis force).

use std::error::Error;
use std::f64::consts::PI;
use std::sync::Arc;
use std::time::Instant;

use dgcom::boundary::{
    CharacteristicOBC, MultiBoundaryCondition2D, NestingOptions, OceanModelState, ParentDatum,
    ParentSeries2D, Reflective2D,
};
use dgcom::equations::ShallowWater2D;
use dgcom::io::{
    CoordinateProjection, FieldSeries, GeoGrid, LocalProjection, OceanModelReader,
    TimeInterpolation,
};
use dgcom::mesh::{Bathymetry2D, BoundaryTag, Mesh2D};
use dgcom::operators::{DGOperators2D, GeometricFactors2D};
use dgcom::physics::{PhysicsBuilder, SWEPhysics2D};
use dgcom::simulation::Simulation;
use dgcom::solver::{SWESolution2D, SWEState2D, StandardLimiter2D, WetDryConfig};
use dgcom::source::{CoriolisSource2D, ManningFriction2D, coriolis_parameter};
use dgcom::time::{ModelClock, Multirate, SSPRK43};
use dgcom::types::ElementIndex;

use crate::gate::{self, Bar};
use crate::phase0::{agreement, cpu_kernel, holds};
use crate::swe2d::{DeviceState, Gpu, GpuSWE2D, NN};

const G: f64 = 9.81;
/// The parent's first snapshot (Unix s), the run's time 0.
const T0: f64 = 1_750_000_000.0;
/// The run (hours).
const HOURS: f64 = 8.0;
/// The CFL number of both runs.
const CFL: f64 = 0.5;

/// The made-up parent: ζ, the current (east, north) and the depth over a
/// grid around the channel's mouth, hourly over the run.
fn parent_reader(projection: &LocalProjection) -> Arc<OceanModelReader> {
    let xs: Vec<f64> = (0..21).map(|i| -6000.0 + 800.0 * i as f64).collect();
    let ys: Vec<f64> = (0..16).map(|j| -4000.0 + 800.0 * j as f64).collect();
    let lon: Vec<f64> = xs.iter().map(|&x| projection.xy_to_geo(x, 0.0).1).collect();
    let lat: Vec<f64> = ys.iter().map(|&y| projection.xy_to_geo(0.0, y).0).collect();
    let grid = GeoGrid::regular(lon, lat).expect("a regular grid");
    let m = grid.len();
    let times: Vec<f64> = (0..=HOURS as usize + 1)
        .map(|k| T0 + 3600.0 * k as f64)
        .collect();
    // An M2-like wave entering eastward, a quarter-diurnal and a mean
    let omega = 2.0 * PI / 44_714.0;
    let kappa = omega / (G * 30.0).sqrt();
    let (mut zeta, mut east, mut north) = (Vec::new(), Vec::new(), Vec::new());
    for &t in &times {
        let t = t - T0;
        for &y in &ys {
            for &x in &xs {
                let phase = omega * t - kappa * x - 0.15 * y / 4000.0;
                zeta.push((0.9 * phase.cos() + 0.2 * (2.0 * phase + 0.4).cos() + 0.04) as f32);
                east.push((0.32 * (phase + 0.1).cos() + 0.03) as f32);
                north.push((0.05 * phase.sin() - 0.02 * x / 6000.0) as f32);
            }
        }
    }
    let depth: Vec<f64> = (0..ys.len())
        .flat_map(|_| xs.iter().map(|&x| 36.0 + 0.0006 * x))
        .collect();
    Arc::new(
        OceanModelReader::new(grid, times)
            .expect("a parent")
            .with_ssh(FieldSeries::new(m, zeta))
            .with_velocity(
                FieldSeries::new(m, east),
                FieldSeries::new(m, north),
                "made up",
            )
            .with_depth(depth),
    )
}

/// One nested configuration.
pub(crate) struct Nest {
    pub(crate) label: &'static str,
    pub(crate) options: NestingOptions,
    /// The band's timescale at the boundary (s), and whether it relaxes the
    /// momentum alone
    pub(crate) timescale: f64,
    pub(crate) momentum_only: bool,
}

impl Nest {
    /// The first of the gate's nests: a cubic, transport-scaled parent whose
    /// band relaxes the level too.
    pub(crate) fn cubic() -> Self {
        Self {
            label: "nested, cubic, transport-scaled",
            options: NestingOptions::default()
                .with_band(3000.0)
                .with_datum(ParentDatum::Shift(-0.04))
                .with_ramp_up(3600.0),
            timescale: 1800.0,
            momentum_only: false,
        }
    }
}

/// The channel nested as `nest` says: its mesh, operators, bed (blended)
/// and parent.
pub(crate) struct Domain {
    pub(crate) mesh: Arc<Mesh2D>,
    pub(crate) ops: Arc<DGOperators2D>,
    pub(crate) geom: Arc<GeometricFactors2D>,
    pub(crate) bed: Arc<Bathymetry2D>,
    pub(crate) parent: OceanModelState,
}

impl Domain {
    pub(crate) fn new(nest: &Nest) -> Result<Self, Box<dyn Error>> {
        let projection = LocalProjection::new(63.5, 8.0);
        let mesh = Mesh2D::uniform_rectangle_with_sides(
            0.0,
            40_000.0,
            0.0,
            4000.0,
            80,
            8,
            // South, east, north, west
            [
                BoundaryTag::Wall,
                BoundaryTag::Wall,
                BoundaryTag::Wall,
                BoundaryTag::Open,
            ],
        );
        let ops = DGOperators2D::new(2);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        let parent = OceanModelState::new(
            parent_reader(&projection),
            &mesh,
            &ops,
            &projection,
            BoundaryTag::Open,
            ModelClock::new(T0),
            &nest.options,
        )?;
        let mut bed =
            Bathymetry2D::from_function(&mesh, &ops, &geom, |x, _| -(30.0 - 32.0 * x / 40_000.0));
        // As the case runner: the bed blended to the parent's, shores kept
        parent.blend_bathymetry(&mut bed, &ops, &geom, 2.0);
        Ok(Self {
            mesh: Arc::new(mesh),
            ops: Arc::new(ops),
            geom: Arc::new(geom),
            bed: Arc::new(bed),
            parent,
        })
    }

    /// Still water at mean sea level.
    fn still_water(&self) -> SWESolution2D {
        let mut q = SWESolution2D::new(self.mesh.n_elements, self.ops.n_nodes);
        for k in ElementIndex::iter(self.mesh.n_elements) {
            for i in 0..self.ops.n_nodes {
                let h = (0.0 - self.bed.get(k, i)).max(0.0);
                q.set_state(k, i, SWEState2D::new(h, 0.0, 0.0));
            }
        }
        q
    }

    /// The physics as the case runner builds a nested case, on `bc`.
    pub(crate) fn physics<'a>(
        &self,
        nest: &Nest,
        bc: MultiBoundaryCondition2D<'a>,
    ) -> SWEPhysics2D<MultiBoundaryCondition2D<'a>> {
        PhysicsBuilder::swe_2d(
            self.mesh.clone(),
            self.ops.clone(),
            self.geom.clone(),
            ShallowWater2D::new(G),
            bc,
        )
        .with_bathymetry(self.bed.clone())
        .with_limiter(StandardLimiter2D::Positivity(WetDryConfig::DEFAULT_H_DRY))
        .with_wet_dry(WetDryConfig::default())
        .with_source(CoriolisSource2D::f_plane(coriolis_parameter(63.5)))
        .with_source(
            self.parent
                .relaxation(nest.timescale)
                .with_momentum_only(nest.momentum_only),
        )
        .with_implicit_friction(ManningFriction2D::new(G, 0.025))
        .build()
    }
}

/// `nest` on the CPU and the GPU with `levels` of local time stepping (0:
/// global steps): the steps and the final state, compared.
fn gate(gpu: &Arc<Gpu>, nest: &Nest, levels: usize) -> Result<bool, Box<dyn Error>> {
    let domain = Domain::new(nest)?;
    let wall = Reflective2D::new();
    let open = CharacteristicOBC::new(domain.parent.clone());
    let bc = || MultiBoundaryCondition2D::new(&wall).with_open(&open);
    let q0 = domain.still_water();

    let t_end = 3600.0 * HOURS;
    let mut cpu = q0.clone();
    let start = Instant::now();
    let cpu_result = if levels > 0 {
        let integrator = Multirate::with_base(SSPRK43, levels);
        Simulation::new(domain.physics(nest, bc()), integrator)
            .with_cfl(CFL)
            .run(&mut cpu, 0.0, t_end)
    } else {
        Simulation::new(domain.physics(nest, bc()), SSPRK43)
            .with_cfl(CFL)
            .run(&mut cpu, 0.0, t_end)
    };
    let cpu_seconds = start.elapsed().as_secs_f64();
    assert!(cpu_result.success, "CPU run: {cpu_result:?}");

    let twin = domain.physics(nest, bc());
    let form = twin.device_form()?;
    let physics = GpuSWE2D::from_form(gpu, &form)?;
    let mut state = DeviceState::upload(gpu, &q0);
    let start = Instant::now();
    let gpu_result = if levels > 0 {
        let integrator = Multirate::with_base(SSPRK43, levels);
        Simulation::new(physics, integrator)
            .with_cfl(CFL)
            .run(&mut state, 0.0, t_end)
    } else {
        Simulation::new(physics, SSPRK43)
            .with_cfl(CFL)
            .run(&mut state, 0.0, t_end)
    };
    gpu.stream.synchronize()?;
    let gpu_seconds = start.elapsed().as_secs_f64();
    assert!(gpu_result.success, "GPU run: {gpu_result:?}");

    let n_el = domain.mesh.n_elements;
    let band = domain.parent.band_elements().iter().filter(|&&b| b).count();
    let (identical, worst_ulps, worst_rel) = agreement(&cpu, &state.download());
    let steps = if levels > 0 { "coarse steps" } else { "steps" };
    println!(
        "{}, {}: {n_el} P2 elements ({band} in the band), {} {steps} on the CPU ({cpu_seconds:.1} s) \
         and {} on the GPU ({gpu_seconds:.1} s) to {HOURS} h; after them {identical} of {} nodes \
         bit-identical to the {} CPU kernel, largest difference {worst_ulps} ulp, {worst_rel:.1e} \
         of the field's largest",
        nest.label,
        if levels > 0 {
            format!("{levels} levels")
        } else {
            "global steps".into()
        },
        cpu_result.n_steps,
        gpu_result.n_steps,
        n_el * NN,
        cpu_kernel()
    );
    let same = cpu_result.n_steps == gpu_result.n_steps && holds(identical, n_el * NN, worst_rel);
    println!("  {}", gate::verdict(nest.label, same));
    Ok(same)
}

pub fn run_gates(gpu: &Arc<Gpu>) -> Result<(), Box<dyn Error>> {
    println!("== Phase 2: a run nested in a parent model ==");
    // Whole runs of 8 h: contraction moves these by up to 4e-11
    gate::start("nest", Bar::within(1e-8));
    let nests = [
        Nest::cubic(),
        Nest {
            label: "nested, linear, unscaled, momentum relaxed",
            options: NestingOptions::default()
                .with_band(3000.0)
                .with_datum(ParentDatum::Shift(0.0))
                .with_time_interpolation(TimeInterpolation::Linear)
                .with_transport_scaling(false)
                .with_ramp_up(1800.0),
            timescale: 900.0,
            momentum_only: true,
        },
    ];
    let mut all = true;
    for nest in &nests {
        for levels in [0, 3] {
            all &= gate(gpu, nest, levels)?;
        }
    }
    if all {
        Ok(())
    } else {
        Err("a nested run on the GPU differs from the CPU's".into())
    }
}

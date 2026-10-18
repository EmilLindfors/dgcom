//! Gate tests for nesting in a parent ocean model and for gridded
//! atmospheric forcing (TODO P1.5, P1.6).
//!
//! The parent and the weather model are given on regular longitude/latitude
//! grids around a mesh that is rotated 30° against east (a projection with
//! a rotated frame), so every run also checks the rotation of east/north
//! vectors into the mesh axes.
//!
//! 1. A progressive long wave supplied by the parent at both ends of a
//!    channel is reproduced inside it; with hourly-like parent output the
//!    cubic time interpolation keeps the error at the percent level, where
//!    linear interpolation does not.
//! 2. Over a child bed half as deep as the parent's, transport scaling
//!    delivers the parent's transport (`ū·D_parent/D_child`); without it the
//!    child carries half.
//! 3. A parent at rest keeps a lake at rest over a rough bed exactly, with a
//!    relaxation band and the bed blended to the parent's.
//! 4. A pressure field from a weather grid holds the sea at its
//!    inverse-barometer level, in a closed basin and through open boundaries
//!    that add the inverse-barometer level (without it, water flows).
//! 5. Wind from a weather grid sets up the surface slope τ/(ρ g H).

use std::f64::consts::PI;
use std::sync::Arc;

use dg_rs::boundary::{
    CharacteristicOBC, InverseBarometer, MultiBoundaryCondition2D, NestingOptions, OceanModelState,
    Reflective2D, SWEBoundaryCondition2D, StillWater,
};
use dg_rs::equations::ShallowWater2D;
use dg_rs::io::{
    AtmosphereField, AtmosphereReader, CoordinateProjection, FieldSeries, GeoGrid, LocalProjection,
    OceanModelReader, P_REFERENCE, TimeInterpolation,
};
use dg_rs::mesh::{Bathymetry2D, BoundaryTag, Mesh2D};
use dg_rs::operators::{DGOperators2D, GeometricFactors2D};
use dg_rs::physics::{PhysicsBuilder, SWEPhysics2DBuilder};
use dg_rs::simulation::Simulation;
use dg_rs::solver::{SWEFormulation2D, SWESolution2D, SWEState2D};
use dg_rs::source::{GriddedAtmosphere2D, RHO_AIR, RHO_WATER};
use dg_rs::time::{ModelClock, SSPRK3};
use dg_rs::types::ElementIndex;

const G: f64 = 9.81;
/// 2025-06-15 00:00 UTC
const T0: f64 = 1_749_945_600.0;
/// Angle of the mesh x axis from east
const ROTATION_DEG: f64 = 30.0;

/// A local tangent plane whose x axis points `ROTATION_DEG` north of east.
#[derive(Clone, Copy)]
struct Rotated {
    local: LocalProjection,
    sin: f64,
    cos: f64,
}

impl Rotated {
    fn new() -> Self {
        let (sin, cos) = ROTATION_DEG.to_radians().sin_cos();
        Self {
            local: LocalProjection::new(63.5, 8.5),
            sin,
            cos,
        }
    }

    /// East/north components of a vector along the mesh axes.
    fn east_north(&self, (x, y): (f64, f64)) -> (f64, f64) {
        (x * self.cos - y * self.sin, x * self.sin + y * self.cos)
    }
}

impl CoordinateProjection for Rotated {
    fn geo_to_xy(&self, lat: f64, lon: f64) -> (f64, f64) {
        let (e, n) = self.local.geo_to_xy(lat, lon);
        (e * self.cos + n * self.sin, -e * self.sin + n * self.cos)
    }

    fn xy_to_geo(&self, x: f64, y: f64) -> (f64, f64) {
        let (e, n) = self.east_north((x, y));
        self.local.xy_to_geo(e, n)
    }
}

/// A regular lon/lat grid covering the mesh-plane box `[x0, x1] × [y0, y1]`
/// (plus a margin) with points about `spacing` metres apart.
fn grid_around(
    projection: &Rotated,
    (x0, x1, y0, y1): (f64, f64, f64, f64),
    spacing: f64,
) -> GeoGrid {
    let margin = 2.0 * spacing;
    let corners = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)].map(|(x, y)| projection.xy_to_geo(x, y));
    let lat_min = corners.iter().map(|c| c.0).fold(f64::INFINITY, f64::min);
    let lat_max = corners
        .iter()
        .map(|c| c.0)
        .fold(f64::NEG_INFINITY, f64::max);
    let lon_min = corners.iter().map(|c| c.1).fold(f64::INFINITY, f64::min);
    let lon_max = corners
        .iter()
        .map(|c| c.1)
        .fold(f64::NEG_INFINITY, f64::max);
    let (dlat, dlon) = (
        spacing / 111_000.0,
        spacing / (111_000.0 * 63.5_f64.to_radians().cos()),
    );
    let (mlat, mlon) = (
        margin / 111_000.0,
        margin / (111_000.0 * 63.5_f64.to_radians().cos()),
    );
    let axis = |lo: f64, hi: f64, step: f64| -> Vec<f64> {
        let n = ((hi - lo) / step).ceil() as usize + 1;
        (0..n)
            .map(|i| lo + (hi - lo) * i as f64 / (n - 1) as f64)
            .collect()
    };
    GeoGrid::regular(
        axis(lon_min - mlon, lon_max + mlon, dlon),
        axis(lat_min - mlat, lat_max + mlat, dlat),
    )
    .unwrap()
}

/// Mesh-plane position of every grid point.
fn grid_xy(grid: &GeoGrid, projection: &Rotated) -> Vec<(f64, f64)> {
    (0..grid.len())
        .map(|k| {
            let (lon, lat) = grid.position(k);
            projection.geo_to_xy(lat, lon)
        })
        .collect()
}

/// A parent with `ζ(x, y, t)` and mesh-axis velocity `u(x, y, t)` given in
/// mesh coordinates, sampled at `times` (seconds after `T0`), depth `depth`.
fn parent(
    projection: &Rotated,
    bounds: (f64, f64, f64, f64),
    times: &[f64],
    depth: f64,
    field: impl Fn(f64, f64, f64) -> (f64, (f64, f64)),
) -> Arc<OceanModelReader> {
    let grid = grid_around(projection, bounds, 500.0);
    let xy = grid_xy(&grid, projection);
    let (mut zeta, mut east, mut north) = (Vec::new(), Vec::new(), Vec::new());
    for &t in times {
        for &(x, y) in &xy {
            let (z, u) = field(x, y, t);
            let (e, n) = projection.east_north(u);
            zeta.push(z as f32);
            east.push(e as f32);
            north.push(n as f32);
        }
    }
    let m = grid.len();
    Arc::new(
        OceanModelReader::new(grid, times.iter().map(|t| T0 + t).collect())
            .unwrap()
            .with_ssh(FieldSeries::new(m, zeta))
            .with_velocity(
                FieldSeries::new(m, east),
                FieldSeries::new(m, north),
                "analytic",
            )
            .with_depth(vec![depth; m]),
    )
}

struct Domain {
    mesh: Arc<Mesh2D>,
    ops: Arc<DGOperators2D>,
    geom: Arc<GeometricFactors2D>,
    bounds: (f64, f64, f64, f64),
}

impl Domain {
    fn new(
        bounds: (f64, f64, f64, f64),
        nx: usize,
        ny: usize,
        order: usize,
        sides: [BoundaryTag; 4],
    ) -> Self {
        let (x0, x1, y0, y1) = bounds;
        let mesh = Mesh2D::uniform_rectangle_with_sides(x0, x1, y0, y1, nx, ny, sides);
        let ops = DGOperators2D::new(order);
        let geom = GeometricFactors2D::compute(&mesh, &ops);
        Self {
            mesh: Arc::new(mesh),
            ops: Arc::new(ops),
            geom: Arc::new(geom),
            bounds,
        }
    }

    /// Channel along x with walls on the long sides and open ends.
    fn channel(length: f64, width: f64, nx: usize, ny: usize, order: usize) -> Self {
        // Side order: [south, east, north, west]
        let sides = [
            BoundaryTag::Wall,
            BoundaryTag::Open,
            BoundaryTag::Wall,
            BoundaryTag::Open,
        ];
        Self::new((0.0, length, 0.0, width), nx, ny, order, sides)
    }

    fn builder<BC: SWEBoundaryCondition2D>(
        &self,
        bc: BC,
        bathymetry: Bathymetry2D,
    ) -> SWEPhysics2DBuilder<BC> {
        PhysicsBuilder::swe_2d(
            self.mesh.clone(),
            self.ops.clone(),
            self.geom.clone(),
            ShallowWater2D::new(G),
            bc,
        )
        .with_bathymetry(Arc::new(bathymetry))
        .with_formulation(SWEFormulation2D::EntropyStable)
    }

    fn nodes(&self) -> impl Iterator<Item = (ElementIndex, usize, (f64, f64))> + '_ {
        ElementIndex::iter(self.mesh.n_elements).flat_map(move |k| {
            (0..self.ops.n_nodes).map(move |i| {
                let [x, y] =
                    self.mesh
                        .reference_to_physical(k, self.ops.nodes_r[i], self.ops.nodes_s[i]);
                (k, i, (x, y))
            })
        })
    }

    /// State with elevation η and velocity (u, v) from `f(x, y)` over `bed`.
    fn fill(&self, bed: &Bathymetry2D, f: impl Fn(f64, f64) -> (f64, f64, f64)) -> SWESolution2D {
        let mut q = SWESolution2D::new(self.mesh.n_elements, self.ops.n_nodes);
        for (k, i, (x, y)) in self.nodes() {
            let (eta, u, v) = f(x, y);
            q.set_state(k, i, SWEState2D::from_primitives(eta - bed.get(k, i), u, v));
        }
        q
    }

    fn nested(&self, reader: Arc<OceanModelReader>, options: &NestingOptions) -> OceanModelState {
        OceanModelState::new(
            reader,
            &self.mesh,
            &self.ops,
            &Rotated::new(),
            BoundaryTag::Open,
            ModelClock::new(T0),
            options,
        )
        .unwrap()
    }
}

// ---------------------------------------------------------------------------
// 1. Progressive wave
// ---------------------------------------------------------------------------

/// Largest surface error, relative to the amplitude, after two periods of a
/// progressive wave nested at both ends, with parent output every T/8.
fn progressive_wave_error(interpolation: TimeInterpolation) -> f64 {
    let (length, depth, amplitude, period) = (20_000.0, 20.0, 0.05, 7200.0);
    let domain = Domain::channel(length, 2000.0, 20, 2, 2);
    let c = (G * depth).sqrt();
    let (k, omega) = (2.0 * PI / (c * period), 2.0 * PI / period);
    let wave = move |x: f64, t: f64| amplitude * (k * x - omega * t).cos();
    let t_end = 2.0 * period;
    let times: Vec<f64> = (0..=16).map(|i| i as f64 * period / 8.0).collect();
    let reader = parent(&Rotated::new(), domain.bounds, &times, depth, |x, _, t| {
        let eta = wave(x, t);
        (eta, ((G / depth).sqrt() * eta, 0.0))
    });
    let nesting = domain.nested(
        reader,
        &NestingOptions::default().with_time_interpolation(interpolation),
    );
    nesting.check_time_coverage(0.0, t_end).unwrap();

    let open = CharacteristicOBC::new(nesting);
    let wall = Reflective2D::new();
    let bc = MultiBoundaryCondition2D::new(&wall).with_open(&open);
    let bed = Bathymetry2D::constant(domain.mesh.n_elements, domain.ops.n_nodes, -depth);
    let physics = domain.builder(bc, bed.clone()).build();
    let mut q = domain.fill(&bed, |x, _| {
        let eta = wave(x, 0.0);
        (eta, (G / depth).sqrt() * eta, 0.0)
    });
    Simulation::new(physics, SSPRK3)
        .with_cfl(0.5)
        .run(&mut q, 0.0, t_end);

    domain
        .nodes()
        .map(|(k, i, (x, _))| (q.get_state(k, i).h - depth - wave(x, t_end)).abs())
        .fold(0.0, f64::max)
        / amplitude
}

#[test]
fn nested_progressive_wave_is_reproduced() {
    let cubic = progressive_wave_error(TimeInterpolation::Cubic);
    let linear = progressive_wave_error(TimeInterpolation::Linear);
    println!("progressive wave: max error / amplitude = {cubic:.4} (cubic), {linear:.4} (linear)");
    assert!(cubic < 0.03, "cubic: {cubic}");
    assert!(linear > 2.0 * cubic, "linear {linear} vs cubic {cubic}");
}

// ---------------------------------------------------------------------------
// 2. Transport over a bed mismatch
// ---------------------------------------------------------------------------

/// Mean velocity along the channel after spin-up from rest, nested in a
/// uniform parent flow of 0.5 m/s over 20 m, on a 10 m deep child.
fn channel_velocity(transport_scaling: bool) -> f64 {
    let (parent_depth, child_depth, u_parent) = (20.0, 10.0, 0.5);
    let domain = Domain::channel(10_000.0, 1000.0, 10, 1, 2);
    let reader = parent(
        &Rotated::new(),
        domain.bounds,
        &[0.0, 4.0 * 3600.0],
        parent_depth,
        |_, _, _| (0.0, (u_parent, 0.0)),
    );
    let options = NestingOptions::default().with_transport_scaling(transport_scaling);
    let nesting = domain.nested(reader, &options);
    let open = CharacteristicOBC::new(nesting);
    let wall = Reflective2D::new();
    let bc = MultiBoundaryCondition2D::new(&wall).with_open(&open);
    let bed = Bathymetry2D::constant(domain.mesh.n_elements, domain.ops.n_nodes, -child_depth);
    let physics = domain.builder(bc, bed.clone()).build();
    let mut q = domain.fill(&bed, |_, _| (0.0, 0.0, 0.0));
    Simulation::new(physics, SSPRK3)
        .with_cfl(0.5)
        .run(&mut q, 0.0, 3.0 * 3600.0);

    let (sum, n) = domain
        .nodes()
        .map(|(k, i, _)| {
            let s = q.get_state(k, i);
            s.hu / s.h
        })
        .fold((0.0, 0), |(s, n), u| (s + u, n + 1));
    sum / n as f64
}

#[test]
fn transport_scaling_delivers_the_parent_transport() {
    let scaled = channel_velocity(true);
    let plain = channel_velocity(false);
    println!("child velocity: {scaled:.4} m/s (scaled), {plain:.4} m/s (plain)");
    // Parent transport 0.5 · 20 = 10 m²/s over 10 m
    assert!((scaled - 1.0).abs() < 0.01, "scaled: {scaled}");
    assert!((plain - 0.5).abs() < 0.01, "plain: {plain}");
}

// ---------------------------------------------------------------------------
// 3. Lake at rest with a relaxation band and a blended bed
// ---------------------------------------------------------------------------

#[test]
fn parent_at_rest_keeps_a_lake_at_rest_with_band_and_blended_bed() {
    let depth = 20.0;
    let domain = Domain::new(
        (-5000.0, 5000.0, -5000.0, 5000.0),
        6,
        6,
        3,
        [BoundaryTag::Open; 4],
    );
    let reader = parent(
        &Rotated::new(),
        domain.bounds,
        &[0.0, 7200.0],
        depth,
        |_, _, _| (0.0, (0.0, 0.0)),
    );
    let nesting = domain.nested(reader, &NestingOptions::default().with_band(2500.0));
    assert!(nesting.n_band_nodes() > 0 && nesting.n_boundary_nodes() > 0);

    let mut bed = Bathymetry2D::from_function(&domain.mesh, &domain.ops, &domain.geom, |x, y| {
        -12.0 - 8.0 * (x / 1300.0).sin() * (y / 1700.0).cos()
    });
    let changed = nesting.blend_bathymetry(&mut bed, &domain.ops, &domain.geom, 0.0);
    assert!(changed > 0);
    let (r0, _, r1) = nesting.depth_ratios(&bed).unwrap();
    assert!(
        (r0 - 1.0).abs() < 1e-12 && (r1 - 1.0).abs() < 1e-12,
        "{r0} {r1}"
    );

    let band = nesting.relaxation(600.0);
    let open = CharacteristicOBC::new(nesting);
    let wall = Reflective2D::new();
    let bc = MultiBoundaryCondition2D::new(&wall).with_open(&open);
    let physics = domain.builder(bc, bed.clone()).with_source(band).build();
    let mut q = domain.fill(&bed, |_, _| (0.0, 0.0, 0.0));
    Simulation::new(physics, SSPRK3)
        .with_cfl(0.5)
        .run(&mut q, 0.0, 3600.0);

    let (mut eta_max, mut u_max) = (0.0_f64, 0.0_f64);
    for (k, i, _) in domain.nodes() {
        let s = q.get_state(k, i);
        eta_max = eta_max.max((s.h + bed.get(k, i)).abs());
        u_max = u_max.max(s.hu.hypot(s.hv) / s.h);
    }
    assert!(
        eta_max < 1e-11 && u_max < 1e-11,
        "η {eta_max:e}, |u| {u_max:e}"
    );
}

// ---------------------------------------------------------------------------
// 4–5. Gridded atmosphere
// ---------------------------------------------------------------------------

/// Weather fields on a grid around `bounds`: uniform wind `wind` along the
/// mesh axes (m/s, ramped by the source) and sea-level pressure
/// `p_ref + ∇p·(x, y)` (Pa, mesh coordinates).
fn weather(
    projection: &Rotated,
    bounds: (f64, f64, f64, f64),
    wind: (f64, f64),
    gradient: (f64, f64),
) -> Arc<AtmosphereReader> {
    let grid = grid_around(projection, bounds, 2500.0);
    let xy = grid_xy(&grid, projection);
    let m = grid.len();
    let times = [0.0, 12.0 * 3600.0];
    let (e, n) = projection.east_north(wind);
    let pressure: Vec<f64> = times
        .iter()
        .flat_map(|_| {
            xy.iter()
                .map(|&(x, y)| P_REFERENCE + gradient.0 * x + gradient.1 * y)
        })
        .collect();
    Arc::new(
        AtmosphereReader::new(grid, times.iter().map(|t| T0 + t).collect())
            .unwrap()
            .with_wind(
                FieldSeries::new(m, vec![e as f32; 2 * m]),
                FieldSeries::new(m, vec![n as f32; 2 * m]),
            )
            .with_field(
                AtmosphereField::Pressure,
                FieldSeries::with_offset(m, &pressure, P_REFERENCE),
            ),
    )
}

/// Largest speed after an hour in a basin started at the inverse-barometer
/// level of a pressure field with gradient (1.0, −0.5) Pa/km, closed or open.
fn inverse_barometer_speed(open: bool, add_level: bool) -> f64 {
    let depth = 10.0;
    let gradient = (1e-3, -5e-4);
    let tag = if open {
        BoundaryTag::Open
    } else {
        BoundaryTag::Wall
    };
    let domain = Domain::new(
        (-10_000.0, 10_000.0, -10_000.0, 10_000.0),
        5,
        5,
        2,
        [tag; 4],
    );
    let reader = weather(&Rotated::new(), domain.bounds, (0.0, 0.0), gradient);
    let atmosphere = GriddedAtmosphere2D::new(
        reader,
        &domain.mesh,
        &domain.ops,
        Rotated::new(),
        ModelClock::new(T0),
    )
    .unwrap()
    .without_wind();
    let wall = Reflective2D::new();
    let open_bc: Box<dyn SWEBoundaryCondition2D> = if add_level {
        Box::new(CharacteristicOBC::new(InverseBarometer::new(
            StillWater::default(),
            atmosphere.clone(),
        )))
    } else {
        Box::new(CharacteristicOBC::still_water())
    };
    let bc = MultiBoundaryCondition2D::new(&wall).with_open(open_bc.as_ref());
    let bed = Bathymetry2D::constant(domain.mesh.n_elements, domain.ops.n_nodes, -depth);
    let physics = domain
        .builder(bc, bed.clone())
        .with_source(atmosphere)
        .build();
    let ib = |x: f64, y: f64| -(gradient.0 * x + gradient.1 * y) / (RHO_WATER * G);
    let mut q = domain.fill(&bed, |x, y| (ib(x, y), 0.0, 0.0));
    Simulation::new(physics, SSPRK3)
        .with_cfl(0.5)
        .run(&mut q, 0.0, 3600.0);
    domain
        .nodes()
        .map(|(k, i, _)| {
            let s = q.get_state(k, i);
            s.hu.hypot(s.hv) / s.h
        })
        .fold(0.0, f64::max)
}

#[test]
fn pressure_field_holds_the_inverse_barometer_level() {
    let closed = inverse_barometer_speed(false, false);
    let open = inverse_barometer_speed(true, true);
    let open_without_level = inverse_barometer_speed(true, false);
    println!(
        "inverse barometer: |u| max {closed:e} (closed), {open:e} (open), {open_without_level:e} (open, no level)"
    );
    // Balanced to the f32 storage of the pressure (~1e-6 Pa on ±15 Pa)
    assert!(closed < 1e-8, "closed: {closed:e}");
    assert!(open < 1e-8, "open: {open:e}");
    assert!(
        open_without_level > 1e-3,
        "without the level: {open_without_level:e}"
    );
}

#[test]
fn wind_from_a_weather_grid_sets_up_the_surface() {
    let (depth, length, speed) = (10.0, 10_000.0, 10.0);
    let domain = Domain::new((0.0, length, 0.0, 1000.0), 10, 1, 2, [BoundaryTag::Wall; 4]);
    let reader = weather(&Rotated::new(), domain.bounds, (speed, 0.0), (0.0, 0.0));
    let atmosphere = GriddedAtmosphere2D::new(
        reader,
        &domain.mesh,
        &domain.ops,
        Rotated::new(),
        ModelClock::new(T0),
    )
    .unwrap()
    .with_ramp_up(3.0 * 3600.0);
    let bed = Bathymetry2D::constant(domain.mesh.n_elements, domain.ops.n_nodes, -depth);
    let physics = domain
        .builder(Reflective2D::new(), bed.clone())
        .with_source(atmosphere)
        .build();
    let mut q = domain.fill(&bed, |_, _| (0.0, 0.0, 0.0));
    Simulation::new(physics, SSPRK3)
        .with_cfl(0.5)
        .run(&mut q, 0.0, 6.0 * 3600.0);

    // Least-squares slope of η along the channel
    let points: Vec<(f64, f64)> = domain
        .nodes()
        .map(|(k, i, (x, _))| (x, q.get_state(k, i).h - depth))
        .collect();
    let n = points.len() as f64;
    let (mx, me) = points
        .iter()
        .fold((0.0, 0.0), |(a, b), p| (a + p.0 / n, b + p.1 / n));
    let (sxy, sxx) = points.iter().fold((0.0, 0.0), |(a, b), p| {
        (a + (p.0 - mx) * (p.1 - me), b + (p.0 - mx).powi(2))
    });
    let slope = sxy / sxx;
    // Large & Pond at 10 m/s: C_d = 1.2e-3
    let expected = RHO_AIR * 1.2e-3 * speed * speed / (RHO_WATER * G * depth);
    println!("wind set-up slope {slope:e}, expected {expected:e}");
    assert!(
        (slope / expected - 1.0).abs() < 0.1,
        "{slope:e} vs {expected:e}"
    );
}

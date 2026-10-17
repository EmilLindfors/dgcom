//! A parent ocean model's 3D fields as the parent of a
//! [`crate::boundary::Nesting3D`].
//!
//! [`OceanModelColumns`] reads the profiles of an [`OceanModelReader`]
//! (`OceanModelReader::from_file_with_profiles` with the `netcdf` feature:
//! NorKyst-800 z-levels, ROMS s-levels) at a child node:
//!
//! - horizontally over the wet corners of the parent cell holding the node
//!   (or the nearest wet parent point within `max_snap`), as the 2D nesting
//!   does; the stencil of each node is computed on first use;
//! - in time with the snapshots around the time (linear or cubic);
//! - vertically at the depths of the child's layer centres,
//!   `d_l = −σ_l (η − B)` below the surface: each parent column is
//!   interpolated linearly between its valid levels, held constant beyond
//!   them ([`crate::io::ProfileSeries::sample`]); with a [`DeepReference`],
//!   tracers below the parent's bed relax to a horizontally uniform profile;
//! - the velocity is rotated from east/north to the mesh axes.
//!
//! The depth mean is the 2D nesting's ([`crate::boundary::OceanModelState`]);
//! the 3D nesting only uses the shear and the tracers, so the parent's
//! velocity profile is not rescaled to the child's depth.

use std::sync::{Arc, OnceLock};

use crate::boundary::{ColumnContext3D, ParentColumn, ParentColumns3D, Supplied};
use crate::io::{
    CoordinateProjection, OceanModelReader, ProfileSeries, Stencil, TimeInterpolation, TimeStencil,
    east_axis,
};
use crate::mesh::Mesh2D;
use crate::operators::DGOperators2D;
use crate::time::ModelClock;
use crate::vertical::SigmaGrid;

/// The tracers below a parent's bed ([`OceanModelColumns::with_deep_reference`]).
///
/// A parent grid of hundreds of metres has a smoother, shallower bed than a
/// child that resolves deep holes and fjord basins, and its profiles end at
/// its bed. Held below it, each child column takes its parent cell's bottom
/// water down its whole hole, so neighbouring holes carry different water
/// down their walls: a horizontal density difference the parent does not
/// have, over the height of the cliff. NorKyst at Mausund is 10–59 m deep
/// over a 90–140 m hole; its bottom water, 0.6 °C apart between neighbouring
/// cells, drove a bed jet of 0.4 m/s within five minutes of a run at rest
/// (TODO P1.3). Below the parent's bed `h`, the tracers are instead
///
/// ```text
/// C(d) = C_ref(d) + (C_parent(d) − C_ref(h)) e^{−(d − h)/decay},
/// ```
///
/// the reference profile plus the parent's anomaly at its bed, decaying with
/// the depth below it.
#[derive(Clone)]
pub struct DeepReference {
    /// Temperature (°C) and salinity at depth `d` below the surface (m,
    /// positive down), the same everywhere
    pub profile: Arc<dyn Fn(f64) -> (f64, f64) + Send + Sync>,
    /// e-folding depth of the parent's anomaly below its bed (m)
    pub decay: f64,
}

/// Most child layers [`OceanModelColumns`] samples (a stack buffer of the
/// layer depths).
pub const MAX_LEVELS: usize = 128;

/// Options of [`OceanModelColumns`].
#[derive(Clone, Copy, Debug)]
pub struct OceanColumnsOptions {
    /// Interpolation between snapshots.
    pub time_interpolation: TimeInterpolation,
    /// Nodes outside the parent's wet cells take the nearest wet parent point
    /// within this distance (m); farther ones are not covered.
    pub max_snap: f64,
}

impl Default for OceanColumnsOptions {
    /// Linear in time (the tracers and the shear are not tidal signals, and a
    /// cubic overshoots at fronts); snapping within 2 km, as the 2D nesting.
    fn default() -> Self {
        Self {
            time_interpolation: TimeInterpolation::Linear,
            max_snap: 2000.0,
        }
    }
}

/// A parent ocean model's profiles at child nodes (see the module docs).
pub struct OceanModelColumns<P> {
    reader: Arc<OceanModelReader>,
    projection: P,
    clock: ModelClock,
    options: OceanColumnsOptions,
    /// The tracers below the parent's bed, if not held
    deep: Option<DeepReference>,
    /// Horizontal stencil and east axis of every mesh node (`[element][node]`),
    /// `None` if the parent does not cover it; computed on first use.
    nodes: Vec<OnceLock<Option<NodeStencil>>>,
}

/// Where a child node samples the parent.
struct NodeStencil {
    stencil: Stencil,
    /// East in the mesh axes, `(cos, sin)` of its angle from x.
    east: (f64, f64),
}

impl<P: CoordinateProjection> OceanModelColumns<P> {
    /// The profiles of `reader` at the nodes of `mesh` (with `ops`'s nodes),
    /// whose coordinates map to longitude/latitude with `projection`;
    /// simulation time maps to UTC with `clock`.
    ///
    /// # Panics
    /// If the reader has neither velocity nor tracer profiles (read them
    /// with `OceanModelReader::from_file_with_profiles`).
    pub fn new(
        reader: Arc<OceanModelReader>,
        mesh: &Mesh2D,
        ops: &DGOperators2D,
        projection: P,
        clock: ModelClock,
        options: OceanColumnsOptions,
    ) -> Self {
        assert!(
            reader.has_current_profiles() || reader.has_tracer_profiles(),
            "OceanModelColumns: the parent has no profiles (read it with \
             OceanModelReader::from_file_with_profiles)"
        );
        Self {
            reader,
            projection,
            clock,
            options,
            deep: None,
            nodes: (0..mesh.n_elements * ops.n_nodes)
                .map(|_| OnceLock::new())
                .collect(),
        }
    }

    /// Below the parent's bed (its `depth` field, interpolated as the
    /// profiles are), the tracers relax to `deep` (see [`DeepReference`]).
    /// Without the parent's depth they are held as before.
    ///
    /// # Panics
    /// If `deep.decay` is not positive.
    pub fn with_deep_reference(mut self, deep: DeepReference) -> Self {
        assert!(deep.decay > 0.0, "DeepReference: decay must be positive");
        self.deep = Some(deep);
        self
    }

    /// The stencil and east axis of `node` at `position`.
    fn node(&self, node: usize, position: (f64, f64)) -> Option<&NodeStencil> {
        self.nodes[node]
            .get_or_init(|| {
                let (lat, lon) = self.projection.xy_to_geo(position.0, position.1);
                let stencil = match self.reader.stencil(lon, lat) {
                    Some((_, s)) => s,
                    None => {
                        let wet = |k: usize| self.reader.is_wet(k);
                        let (k, _) =
                            self.reader
                                .grid()
                                .nearest(lon, lat, self.options.max_snap, wet)?;
                        Stencil::point(k)
                    }
                };
                Some(NodeStencil {
                    stencil,
                    east: east_axis(&self.projection, lat, lon),
                })
            })
            .as_ref()
    }

    /// Snapshot weights at simulation time `t`.
    fn moment(&self, t: f64) -> TimeStencil {
        let unix = self.clock.unix(t);
        self.reader
            .time_stencil(unix, self.options.time_interpolation)
            .unwrap_or_else(|| {
                let (first, last) = self.reader.time_range().expect("at least one snapshot");
                panic!(
                    "3D nesting: {} is outside the parent's snapshots ({} to {})",
                    self.clock.format(t),
                    self.clock.format(self.clock.model_time(first)),
                    self.clock.format(self.clock.model_time(last)),
                )
            })
    }
}

impl<P: CoordinateProjection + Send + Sync> ParentColumns3D for OceanModelColumns<P> {
    fn supplies(&self) -> Supplied {
        Supplied {
            velocity: self.reader.has_current_profiles(),
            tracers: self.reader.has_tracer_profiles(),
        }
    }

    fn column(&self, ctx: &ColumnContext3D, sigma: &SigmaGrid, out: ParentColumn<'_>) -> bool {
        let Some(NodeStencil {
            stencil,
            east: (c, s),
        }) = self.node(ctx.node, ctx.position)
        else {
            return false;
        };
        let n_levels = sigma.n_levels();
        assert!(
            n_levels <= MAX_LEVELS,
            "OceanModelColumns: at most {MAX_LEVELS} layers, got {n_levels}"
        );
        let column_depth = (ctx.eta - ctx.bed).max(0.0);
        let mut depths = [0.0; MAX_LEVELS];
        for (d, &s) in depths.iter_mut().zip(sigma.sigma_rho()) {
            *d = -s * column_depth;
        }
        let depths = &depths[..n_levels];
        let time = self.moment(ctx.time);
        let space = stencil.idx.iter().map(|&k| k as usize).zip(stencil.w);
        let sample = |series: Option<&ProfileSeries>, out: &mut [f64]| {
            series.is_some_and(|p| p.sample(&time, space.clone(), depths, column_depth, out))
        };
        let supplied = self.supplies();
        if supplied.velocity {
            let profiles = self.reader.velocity_profiles();
            if !(sample(profiles.map(|p| p.0), out.u) && sample(profiles.map(|p| p.1), out.v)) {
                return false;
            }
            for (u, v) in out.u.iter_mut().zip(out.v.iter_mut()) {
                let (east, north) = (*u, *v);
                (*u, *v) = (east * c - north * s, east * s + north * c);
            }
        }
        if supplied.tracers
            && !(sample(self.reader.temperature_profile(), out.temp)
                && sample(self.reader.salinity_profile(), out.salt))
        {
            return false;
        }
        // Below the parent's bed: the reference plus the decaying anomaly
        if supplied.tracers
            && let (Some(deep), Some(depth)) = (&self.deep, self.reader.depth())
        {
            let h: f64 = stencil
                .idx
                .iter()
                .zip(stencil.w)
                .map(|(&k, w)| w * depth[k as usize])
                .sum();
            if h.is_finite() {
                let (t_h, s_h) = (deep.profile)(h);
                for (l, &d) in depths.iter().enumerate() {
                    if d > h {
                        let decay = (-(d - h) / deep.decay).exp();
                        let (t_d, s_d) = (deep.profile)(d);
                        out.temp[l] = t_d + decay * (out.temp[l] - t_h);
                        out.salt[l] = s_d + decay * (out.salt[l] - s_h);
                    }
                }
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::{GeoGrid, LocalProjection, ProfileLevels};
    use crate::types::ElementIndex;

    const T0: f64 = 1_706_594_400.0;

    /// A parent on a regular grid around the origin of `projection`, two
    /// snapshots an hour apart, on z-levels 0, 5, 20 m: `u = 0.3 − 0.01 d`
    /// east and `0.1` north, `T = 12 − 0.2 d (+1 in the second snapshot)`,
    /// `S = 34`, the same everywhere; the grid's east half is land.
    fn reader(projection: &LocalProjection) -> Arc<OceanModelReader> {
        let (lat0, lon0) = projection.xy_to_geo(-10_000.0, -10_000.0);
        let (lat1, lon1) = projection.xy_to_geo(10_000.0, 10_000.0);
        let n = 5;
        let axis = |a: f64, b: f64| (0..n).map(|i| a + (b - a) * i as f64 / 4.0).collect();
        let grid = GeoGrid::regular(axis(lon0, lon1), axis(lat0, lat1)).unwrap();
        let m = grid.len();
        let levels = [0.0, 5.0, 20.0];
        let land = |k: usize| k % n > 2;
        let profiles = |f: &dyn Fn(usize, f64) -> f64| {
            let data = (0..2)
                .flat_map(|t| (0..m).map(move |k| (t, k)))
                .flat_map(|(t, k)| levels.map(|d| if land(k) { f32::NAN } else { f(t, d) as f32 }))
                .collect();
            ProfileSeries::new(m, ProfileLevels::Depth(levels.to_vec()), data)
        };
        let ssh = crate::io::FieldSeries::new(
            m,
            (0..2 * m)
                .map(|i| if land(i % m) { f32::NAN } else { 0.0 })
                .collect(),
        );
        Arc::new(
            OceanModelReader::new(grid, vec![T0, T0 + 3600.0])
                .unwrap()
                .with_ssh(ssh)
                .with_profiles(
                    Some((profiles(&|_, d| 0.3 - 0.01 * d), profiles(&|_, _| 0.1))),
                    Some(profiles(&|t, d| 12.0 - 0.2 * d + t as f64)),
                    Some(profiles(&|_, _| 34.0)),
                ),
        )
    }

    /// Profiles linear in depth come out exactly at the child's layer
    /// centres, interpolated in time, rotated to the mesh axes (east is x
    /// here); nodes over land farther than `max_snap` are not covered.
    #[test]
    fn columns_are_sampled_at_the_child_layer_depths() {
        let projection = LocalProjection::new(63.5, 8.5);
        let mesh = Mesh2D::uniform_rectangle(-2000.0, 2000.0, -2000.0, 2000.0, 2, 2);
        let ops = DGOperators2D::new(1);
        let options = OceanColumnsOptions {
            max_snap: 500.0,
            ..OceanColumnsOptions::default()
        };
        let parent = OceanModelColumns::new(
            reader(&projection),
            &mesh,
            &ops,
            projection,
            ModelClock::new(T0),
            options,
        );
        assert_eq!(
            parent.supplies(),
            Supplied {
                velocity: true,
                tracers: true
            }
        );
        let sigma = SigmaGrid::uniform(4);
        let [x, y] =
            mesh.reference_to_physical(ElementIndex::new(0), ops.nodes_r[0], ops.nodes_s[0]);
        let ctx = ColumnContext3D {
            time: 1800.0,
            node: 0,
            position: (x, y),
            bed: -12.0,
            eta: 0.4,
        };
        let mut values = [[0.0; 4]; 4];
        let [u, v, temp, salt] = &mut values;
        let out = ParentColumn { u, v, temp, salt };
        assert!(parent.column(&ctx, &sigma, out));
        for (l, &s) in sigma.sigma_rho().iter().enumerate() {
            let d = -s * 12.4;
            assert!(
                (values[0][l] - (0.3 - 0.01 * d)).abs() < 1e-6,
                "{:?}",
                values[0]
            );
            assert!((values[1][l] - 0.1).abs() < 1e-6);
            assert!(
                (values[2][l] - (12.5 - 0.2 * d)).abs() < 1e-5,
                "{:?}",
                values[2]
            );
            assert!((values[3][l] - 34.0).abs() < 1e-5);
        }

        // Far over land: not covered
        let land = ColumnContext3D {
            position: (9000.0, 0.0),
            node: 1,
            ..ctx
        };
        let [u, v, temp, salt] = &mut values;
        assert!(!parent.column(&land, &sigma, ParentColumn { u, v, temp, salt }));
    }

    /// TODO P1.3 regression: below the parent's bed (5 m here), the tracers
    /// are the deep reference plus the parent's anomaly at its bed, decaying
    /// with the depth below it; above it they are the parent's.
    #[test]
    fn below_the_parent_bed_the_tracers_decay_to_the_reference() {
        let projection = LocalProjection::new(63.5, 8.5);
        let mesh = Mesh2D::uniform_rectangle(-2000.0, 2000.0, -2000.0, 2000.0, 2, 2);
        let ops = DGOperators2D::new(1);
        let shallow = Arc::try_unwrap(reader(&projection)).expect("one owner");
        let m = shallow.grid().len();
        let reference = |d: f64| (10.0 - 0.1 * d, 35.0);
        let parent = OceanModelColumns::new(
            Arc::new(shallow.with_depth(vec![5.0; m])),
            &mesh,
            &ops,
            projection,
            ModelClock::new(T0),
            OceanColumnsOptions::default(),
        )
        .with_deep_reference(DeepReference {
            profile: Arc::new(reference),
            decay: 2.0,
        });
        let sigma = SigmaGrid::uniform(4);
        let [x, y] =
            mesh.reference_to_physical(ElementIndex::new(0), ops.nodes_r[0], ops.nodes_s[0]);
        let ctx = ColumnContext3D {
            time: 1800.0,
            node: 0,
            position: (x, y),
            bed: -12.0,
            eta: 0.4,
        };
        let mut values = [[0.0; 4]; 4];
        let [u, v, temp, salt] = &mut values;
        assert!(parent.column(&ctx, &sigma, ParentColumn { u, v, temp, salt }));
        let (t_h, s_h) = reference(5.0);
        for (l, &s) in sigma.sigma_rho().iter().enumerate() {
            let d = -s * 12.4;
            let (t_parent, s_parent) = (12.5 - 0.2 * d, 34.0);
            let (t, s) = if d > 5.0 {
                let decay = (-(d - 5.0) / 2.0).exp();
                let (t_d, s_d) = reference(d);
                (
                    t_d + decay * (t_parent - t_h),
                    s_d + decay * (s_parent - s_h),
                )
            } else {
                (t_parent, s_parent)
            };
            assert!((values[2][l] - t).abs() < 1e-5, "{l}: {:?}", values[2]);
            assert!((values[3][l] - s).abs() < 1e-5, "{l}: {:?}", values[3]);
        }
        // The velocity is the parent's
        assert!((values[1][3] - 0.1).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "outside the parent's snapshots")]
    fn times_outside_the_parent_panic() {
        let projection = LocalProjection::new(63.5, 8.5);
        let mesh = Mesh2D::uniform_rectangle(-2000.0, 2000.0, -2000.0, 2000.0, 2, 2);
        let ops = DGOperators2D::new(1);
        let parent = OceanModelColumns::new(
            reader(&projection),
            &mesh,
            &ops,
            projection,
            ModelClock::new(T0),
            OceanColumnsOptions::default(),
        );
        let sigma = SigmaGrid::uniform(2);
        let mut values = [[0.0; 2]; 4];
        let [u, v, temp, salt] = &mut values;
        let ctx = ColumnContext3D {
            time: 7200.0,
            node: 0,
            position: (0.0, 0.0),
            bed: -10.0,
            eta: 0.0,
        };
        parent.column(&ctx, &sigma, ParentColumn { u, v, temp, salt });
    }
}

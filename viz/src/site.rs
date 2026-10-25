//! Fish farm sites on the map (`--site FILE`): a Fiskeridirektoratet site file
//! (`scripts/fiskeridir_site.sh`, read by `dg_rs::io::farm_site`) placed by the
//! scenario's projection.
//!
//! The site's cages ([`site_cages`], `FarmSite::cage_layout`: the file's `cage` lines,
//! or one per frame cell) join the scenario's, so [`crate::cages`] draws them and a
//! live run drags on them (`--drag`). This module draws the rest of the installation:
//! the frame as floating pipes along its sides and between its cells, the mooring
//! lines from the frame (or the feed raft) down to their anchors on the bed, a buoy
//! where each leaves the frame, and the feed raft. The frame rides η at the site's
//! centre, as each cage rides its own.
//!
//! The frame's cells and the cages' size are guesses from the certified frame (see
//! `io::farm_site`): check them against an aerial photo before reading anything into
//! the drag.

use bevy::prelude::*;
use dg_rs::io::{CageGrid, FarmSite, LocalProjection, MooringKind};
use dg_rs::mesh::PointLocator2D;
use dg_rs::source::NetCage;

use crate::field::{Field, Frame, Nodes, Probe};
use crate::scenario::Scenario;

/// Frame cells (m), cage radius (m) and net depth (m) when the file gives no cages:
/// Kattholmen's 180 × 365 m frame holds 2 × 4 cages of 160 m circumference.
pub const CAGE_GRID: CageGrid = CageGrid::RING_160M;

/// Radius (m) of the frame's pipes and of the mooring lines as drawn: thicker than
/// real, to be seen from a few hundred metres.
const FRAME_PIPE: f32 = 0.6;
const MOORING_LINE: f32 = 0.25;
const BUOY: f32 = 1.5;
/// The feed raft: length, beam and height above the water (m).
const RAFT: [f32; 3] = [24.0, 10.0, 3.0];
/// Depth (m) of an anchor that lies outside the mesh, where the bed is unknown.
const ANCHOR_DEPTH: f32 = 30.0;

/// The site's cages in mesh coordinates, but those the scenario already has (a
/// replay of a run saved with the site).
pub fn site_cages(
    site: &FarmSite,
    projection: &LocalProjection,
    existing: &[NetCage],
) -> Vec<NetCage> {
    let centre = |cage: &NetCage| {
        let (lo, hi) = cage.footprint.bounding_box();
        [0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])]
    };
    site.cage_layout(projection, &CAGE_GRID)
        .into_iter()
        .filter(|cage| {
            let c = centre(cage);
            !existing.iter().any(|e| {
                let d = centre(e);
                (c[0] - d[0]).hypot(c[1] - d[1]) < 1.0
            })
        })
        .collect()
}

/// The centre (m, mesh coordinates) of the site's frame, or its register position.
pub fn site_centre(site: &FarmSite, projection: &LocalProjection) -> [f64; 2] {
    let points = site.boundary_xy(projection);
    if points.is_empty() {
        let (x, y) = dg_rs::io::CoordinateProjection::geo_to_xy(
            projection,
            site.position[1],
            site.position[0],
        );
        return [x, y];
    }
    let n = points.len() as f64;
    points
        .iter()
        .fold([0.0; 2], |a, p| [a[0] + p[0] / n, a[1] + p[1] / n])
}

/// One site as drawn, in mesh coordinates.
pub struct SiteDrawing {
    /// The frame's pipes
    frame: Vec<[[f64; 2]; 2]>,
    /// Mooring lines: from the frame or raft, to the anchor, the anchor's height (m)
    moorings: Vec<([f64; 2], [f64; 2], f32, MooringKind)>,
    /// The feed raft's centre and the direction of its mooring line
    raft: Option<([f64; 2], [f64; 2])>,
    /// η at the site's centre
    probe: Option<Probe>,
    centre: [f64; 2],
}

impl SiteDrawing {
    pub fn new(
        site: &FarmSite,
        projection: &LocalProjection,
        scenario: &Scenario,
        nodes: &Nodes,
        locator: &PointLocator2D,
    ) -> Self {
        let xy = |[lon, lat]: [f64; 2]| {
            let (x, y) = dg_rs::io::CoordinateProjection::geo_to_xy(projection, lat, lon);
            [x, y]
        };
        let bed = |p: [f64; 2]| {
            Probe::at(locator, scenario, p).map_or(-ANCHOR_DEPTH, |probe| probe.eval(&nodes.bed))
        };
        let frame = site
            .frame_cells(projection, CAGE_GRID.spacing)
            .map_or_else(Vec::new, |cells| cells.lines());
        let moorings: Vec<_> = site
            .moorings
            .iter()
            .map(|m| (xy(m.from), xy(m.to), bed(xy(m.to)), m.kind))
            .collect();
        let raft = moorings
            .iter()
            .find(|m| m.3 == MooringKind::Raft)
            .map(|&(from, to, _, _)| (from, [to[0] - from[0], to[1] - from[1]]));
        let centre = site_centre(site, projection);
        Self {
            frame,
            moorings,
            raft,
            probe: Probe::at(locator, scenario, centre),
            centre,
        }
    }
}

/// The sites to draw.
#[derive(Resource)]
pub struct Sites(pub Vec<SiteDrawing>);

#[derive(Component)]
struct SiteRoot(Option<Probe>);

pub struct SitePlugin;

impl Plugin for SitePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn)
            .add_systems(Update, ride.after(crate::field::interpolate));
    }
}

/// A cylinder of `radius` from `a` to `b` (world).
fn pipe(a: Vec3, b: Vec3, radius: f32) -> Transform {
    let d = b - a;
    Transform::from_translation(0.5 * (a + b))
        .with_rotation(Quat::from_rotation_arc(Vec3::Y, d.normalize_or(Vec3::Y)))
        .with_scale(Vec3::new(radius, d.length(), radius))
}

fn spawn(
    mut commands: Commands,
    sites: Option<Res<Sites>>,
    frame: Res<Frame>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let Some(sites) = sites else { return };
    let cylinder = meshes.add(Cylinder::new(1.0, 1.0));
    let buoy = meshes.add(Sphere::new(BUOY));
    let material = |materials: &mut Assets<StandardMaterial>, c: [f32; 3], rough: f32| {
        materials.add(StandardMaterial {
            base_color: Color::srgb(c[0], c[1], c[2]),
            perceptual_roughness: rough,
            ..default()
        })
    };
    let steel = material(&mut materials, [0.04, 0.04, 0.045], 0.4);
    let rope = material(&mut materials, [0.85, 0.72, 0.25], 0.8);
    let orange = material(&mut materials, [1.0, 0.42, 0.08], 0.5);
    let hull = material(&mut materials, [0.82, 0.83, 0.8], 0.6);

    for site in &sites.0 {
        // Children are placed at the site's centre at rest: the root moves with η
        let origin = frame.world(site.centre, 0.0);
        let local = |p: [f64; 2], z: f32| frame.world(p, z) - origin;
        let mut root = commands.spawn((
            Name::new("Farm site"),
            SiteRoot(site.probe.clone()),
            Transform::from_translation(origin),
            Visibility::default(),
        ));
        for &[a, b] in &site.frame {
            root.with_child((
                Mesh3d(cylinder.clone()),
                MeshMaterial3d(steel.clone()),
                pipe(local(a, 0.0), local(b, 0.0), FRAME_PIPE),
            ));
        }
        for &(from, to, anchor, _) in &site.moorings {
            let top = local(from, 0.0);
            root.with_child((
                Mesh3d(cylinder.clone()),
                MeshMaterial3d(rope.clone()),
                pipe(top, local(to, anchor), MOORING_LINE),
            ));
            root.with_child((
                Mesh3d(buoy.clone()),
                MeshMaterial3d(orange.clone()),
                Transform::from_translation(top),
            ));
        }
        if let Some((at, along)) = site.raft {
            // On the water, broadside to its mooring line
            let line = Vec3::new(along[0] as f32, 0.0, -along[1] as f32).normalize_or(Vec3::Z);
            let [length, beam, height] = RAFT;
            root.with_child((
                Mesh3d(meshes.add(Cuboid::new(length, height * frame.vz, beam))),
                MeshMaterial3d(hull.clone()),
                Transform::from_translation(local(at, 0.5 * height))
                    .with_rotation(Quat::from_rotation_arc(Vec3::X, line.cross(Vec3::Y))),
            ));
        }
    }
}

/// Moves each site's frame to the surface at its centre.
fn ride(field: Res<Field>, frame: Res<Frame>, mut roots: Query<(&SiteRoot, &mut Transform)>) {
    if field.t.is_none() || !field.is_changed() {
        return;
    }
    for (root, mut transform) in &mut roots {
        if let Some(probe) = &root.0 {
            transform.translation.y = probe.eval(&field.eta) * frame.vz;
        }
    }
}

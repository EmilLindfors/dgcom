//! Depth contours of the bed, so the bathymetry reads through the water.
//!
//! Lines where the bed crosses round depths, found by marching triangles over the same
//! triangles the bed is drawn with (each element's p² sub-quads between its GLL nodes,
//! split along the same diagonal), so they follow the drawn bed. The interval is a
//! round step giving about ten lines over the depth range; every fifth line is
//! brighter.
//!
//! Lines are drawn as flat ribbons a few pixels wide: 1-pixel lines vanish under the
//! translucent water at high resolution. The width follows the camera's distance, and
//! the ribbons are rebuilt when it changes by a fifth. B (key) toggles them; the photo
//! view ([`crate::photo`]) hides them.

use bevy::asset::RenderAssetUsages;
use bevy::camera::visibility::NoFrustumCulling;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;

use crate::actions::{Act, Action};
use crate::camera::OrbitCamera;
use crate::field::{Frame, Nodes};
use crate::photo::Photo;
use crate::screen::{RUN, RunResources, Screen, Viewing};
use crate::surface::nice_ceil;

pub struct ContoursPlugin;

impl Plugin for ContoursPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(OnEnter(Screen::Viewer), spawn)
            .add_systems(Update, (toggle, rebuild).in_set(Viewing))
            .run_resource::<Contours>()
            .run_resource::<Segments>();
    }
}

/// The contour interval (m) and whether the contours are shown, for the legend.
#[derive(Resource, Clone, Copy, Debug)]
pub struct Contours {
    pub interval: f32,
    pub on: bool,
}

/// The contour segments in the world, and the ribbon width they are drawn at.
#[derive(Resource)]
struct Segments {
    /// World (X, Z) ends and height (world units) of every segment, and whether it
    /// is on an index (bright) contour
    lines: Vec<([Vec2; 2], f32, bool)>,
    mesh: Handle<Mesh>,
    width: f32,
}

#[derive(Component)]
struct ContourLines;

/// Contours per bright (index) contour.
const INDEX_EVERY: i64 = 5;
/// Ribbon width per unit of camera distance: about 3 pixels on a 1080-line screen.
const WIDTH_PER_DISTANCE: f32 = 0.0025;
const REGULAR: [f32; 4] = [0.55, 0.62, 0.68, 1.0];
const INDEX: [f32; 4] = [0.95, 0.97, 1.0, 1.0];

/// Line segments where the nodal bed crosses multiples of `interval` (m), with the
/// contour number (level / interval) of each; the triangles are those of
/// `surface::Sheet::surface`.
fn contour_segments(nodes: &Nodes, interval: f32) -> Vec<([Vec2; 2], i64)> {
    let n1 = nodes.n_1d;
    let mut segments = Vec::new();
    for k in 0..nodes.n_elements {
        let base = k * nodes.n_nodes;
        for j in 0..n1 - 1 {
            for i in 0..n1 - 1 {
                let v0 = base + j * n1 + i;
                let (v1, v2, v3) = (v0 + 1, v0 + n1 + 1, v0 + n1);
                for tri in [[v0, v1, v2], [v0, v2, v3]] {
                    let b = tri.map(|g| nodes.bed[g]);
                    let lo = (b[0].min(b[1]).min(b[2]) / interval).ceil() as i64;
                    let hi = (b[0].max(b[1]).max(b[2]) / interval).floor() as i64;
                    for level in lo..=hi {
                        let z = level as f32 * interval;
                        // Nodes at or above the level count as above: every crossing
                        // edge then has one end strictly below, so a triangle is cut
                        // by two edges or none
                        let mut ends = [Vec2::ZERO; 2];
                        let mut found = 0;
                        for (a, c) in [(0, 1), (1, 2), (2, 0)] {
                            if (b[a] >= z) != (b[c] >= z) {
                                let t = (z - b[a]) / (b[c] - b[a]);
                                ends[found] = nodes.xz[tri[a]].lerp(nodes.xz[tri[c]], t);
                                found += 1;
                            }
                        }
                        if found == 2 {
                            segments.push((ends, level));
                        }
                    }
                }
            }
        }
    }
    segments
}

/// Flat ribbons `width` wide along the segments, lifted clear of the bed on slopes
/// up to about 60° (a ribbon lies on the contour, its edges beside it).
fn ribbons(lines: &[([Vec2; 2], f32, bool)], width: f32) -> Mesh {
    let mut positions = Vec::with_capacity(4 * lines.len());
    let mut colours = Vec::with_capacity(4 * lines.len());
    let mut indices = Vec::with_capacity(6 * lines.len());
    let half = 0.5 * width;
    for &([p0, p1], y, index) in lines {
        let along = (p1 - p0).normalize_or_zero();
        let side = Vec2::new(-along.y, along.x) * half;
        let y = y + width;
        let base = positions.len() as u32;
        for p in [p0 - side, p0 + side, p1 + side, p1 - side] {
            positions.push([p.x, y, p.y]);
            colours.push(if index { INDEX } else { REGULAR });
        }
        indices.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    let normals = vec![[0.0, 1.0, 0.0]; positions.len()];
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, colours)
    .with_inserted_indices(Indices::U32(indices))
}

fn spawn(
    mut commands: Commands,
    nodes: Res<Nodes>,
    frame: Res<Frame>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let (deepest, shallowest) = nodes
        .bed
        .iter()
        .fold((0.0_f32, f32::NEG_INFINITY), |(lo, hi), &b| {
            (lo.min(b), hi.max(b))
        });
    let range = shallowest.min(0.0) - deepest;
    let interval = nice_ceil(range / 10.0).max(1.0);
    commands.insert_resource(Contours { interval, on: true });

    let lines: Vec<_> = contour_segments(&nodes, interval)
        .into_iter()
        .map(|(ends, level)| {
            (
                ends,
                level as f32 * interval * frame.vz,
                level % INDEX_EVERY == 0,
            )
        })
        .collect();
    // Rebuilt at the camera's width once the camera exists
    let width = 1.0;
    let mesh = meshes.add(ribbons(&lines, width));
    commands.spawn((
        RUN,
        Name::new("Bed contours"),
        ContourLines,
        Mesh3d(mesh.clone()),
        MeshMaterial3d(materials.add(StandardMaterial {
            unlit: true,
            double_sided: true,
            cull_mode: None,
            ..default()
        })),
        NoFrustumCulling,
    ));
    commands.insert_resource(Segments { lines, mesh, width });
}

/// Rebuilds the ribbons when the camera's distance changes their width by a fifth.
fn rebuild(
    camera: Query<&OrbitCamera, Changed<OrbitCamera>>,
    segments: Option<ResMut<Segments>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    let (Some(mut segments), Ok(camera)) = (segments, camera.single()) else {
        return;
    };
    let width = WIDTH_PER_DISTANCE * camera.distance;
    let ratio = width / segments.width;
    if (0.8..1.25).contains(&ratio) {
        return;
    }
    segments.width = width;
    let mesh = ribbons(&segments.lines, width);
    if let Some(mut handle) = meshes.get_mut(&segments.mesh) {
        *handle = mesh;
    }
}

fn toggle(
    mut acts: MessageReader<Act>,
    photo: Res<Photo>,
    contours: Option<ResMut<Contours>>,
    mut lines: Query<&mut Visibility, With<ContourLines>>,
) {
    let Some(mut contours) = contours else { return };
    let toggles = acts.read().filter(|a| a.0 == Action::Contours).count();
    if toggles % 2 == 1 {
        contours.on = !contours.on;
    } else if !photo.is_changed() {
        return;
    }
    for mut visibility in &mut lines {
        *visibility = if contours.on && !photo.on {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
    }
}

//! A map of the whole domain in the bottom-left corner of the view (V hides it): the
//! scene seen straight down by a second, orthographic camera over the mesh's bounds,
//! north up, with the main view's footprint on the sea and its focus drawn on it. The
//! cursor over the map gets the inspector's readout there ([`crate::inspect`]); a
//! click moves the main view's focus to the point, keeping its angle and distance.
//! The photo view has no map.
//!
//! The map renders the scene a second time, but into a few hundred pixels: the
//! vertices are the main view's, the fragments a small fraction of them.

use bevy::camera::visibility::RenderLayers;
use bevy::camera::{ScalingMode, Viewport};
use bevy::prelude::*;
use bevy::window::PrimaryWindow;

use crate::actions::{Act, Action};
use crate::camera::{Inset, OrbitCamera, OverUi};
use crate::photo::Photo;
use crate::screen::{RUN, RunResources, Screen, Viewing};

/// The map's longer side (logical px), at most this share of the window's height.
const SIZE: f32 = 300.0;
const SHARE: f32 = 0.3;
/// From the window's left edge, and above the colour scales at the bottom (logical px)
const LEFT: f32 = 14.0;
const BOTTOM: f32 = 175.0;
/// The render layer of what only the map shows (the footprint)
const LAYER: usize = 1;
/// The render layer of what only the main view shows (the stratification sheet, which
/// would hide the water's colouring on the map, and the current arrows); the main
/// camera sees 0 and this
pub const MAIN_ONLY: usize = 2;
/// The camera's height above mean sea level (m): above any land or exaggerated tide
const HEIGHT: f32 = 50_000.0;

/// The map: whether it is shown, the domain it frames and where it is on the window.
#[derive(Resource)]
pub struct Minimap {
    pub shown: bool,
    /// Half the framed world extent in X (east) and Z (south)
    half: Vec2,
    /// Its rectangle on the window (logical px, top-left origin), while shown
    rect: Option<Rect>,
}

impl Minimap {
    /// Framing the world box `lo..hi` (X, Z) about the origin, which is the mesh's
    /// bounding box's centre ([`crate::field::Frame`]), with a little room.
    pub fn new(lo: Vec2, hi: Vec2, shown: bool) -> Self {
        Self {
            shown,
            half: 0.52 * (hi - lo).max(Vec2::splat(1.0)),
            rect: None,
        }
    }

    /// The world point on the sea under the window position `cursor` (logical px),
    /// if it is on the map.
    pub fn world_at(&self, cursor: Vec2) -> Option<Vec3> {
        let rect = self.rect.filter(|r| r.contains(cursor))?;
        let u = (cursor - rect.min) / rect.size();
        // North (−Z) up, east (+X) right
        Some(Vec3::new(
            (2.0 * u.x - 1.0) * self.half.x,
            0.0,
            (2.0 * u.y - 1.0) * self.half.y,
        ))
    }

    /// Its rectangle on a window of `size` (logical px): the domain's aspect, the
    /// longer side [`SIZE`] or less.
    fn layout(&self, size: Vec2) -> Option<Rect> {
        let long = SIZE.min(SHARE * size.y);
        let wh = if self.half.x >= self.half.y {
            Vec2::new(long, long * self.half.y / self.half.x)
        } else {
            Vec2::new(long * self.half.x / self.half.y, long)
        };
        let min = Vec2::new(LEFT, size.y - BOTTOM - wh.y);
        (min.y >= 0.0 && min.x + wh.x <= size.x && wh.min_element() >= 16.0)
            .then(|| Rect::from_corners(min, min + wh))
    }
}

#[derive(Component)]
struct MapCamera;

/// The footprint's lines, on the map's layer only.
#[derive(Default, Reflect, GizmoConfigGroup)]
struct Footprint;

pub struct MinimapPlugin;

impl Plugin for MinimapPlugin {
    fn build(&self, app: &mut App) {
        app.init_gizmo_group::<Footprint>()
            .init_run_resource::<Inset>()
            .init_run_resource::<OverUi>()
            .run_resource::<Minimap>()
            .add_systems(OnEnter(Screen::Viewer), spawn)
            .add_systems(
                Update,
                (keys, place, footprint, click).chain().in_set(Viewing),
            );
    }
}

fn spawn(mut commands: Commands, map: Res<Minimap>, mut gizmos: ResMut<GizmoConfigStore>) {
    commands.spawn((
        RUN,
        MapCamera,
        Camera3d::default(),
        Camera {
            order: 1,
            is_active: false,
            clear_color: ClearColorConfig::Custom(Color::srgb(0.035, 0.04, 0.05)),
            ..default()
        },
        Projection::from(OrthographicProjection {
            near: 1.0,
            far: 2.0 * HEIGHT,
            scaling_mode: ScalingMode::Fixed {
                width: 2.0 * map.half.x,
                height: 2.0 * map.half.y,
            },
            ..OrthographicProjection::default_3d()
        }),
        Transform::from_xyz(0.0, HEIGHT, 0.0).looking_at(Vec3::ZERO, Vec3::NEG_Z),
        RenderLayers::from_layers(&[0, LAYER]),
    ));
    let (config, _) = gizmos.config_mut::<Footprint>();
    config.render_layers = RenderLayers::layer(LAYER);
    // Over the water and the land
    config.depth_bias = -1.0;
    config.line.width = 2.0;
    // The current arrows (the default gizmos) are too fine for the map
    gizmos
        .config_mut::<DefaultGizmoConfigGroup>()
        .0
        .render_layers = RenderLayers::layer(MAIN_ONLY);
}

fn keys(mut acts: MessageReader<Act>, mut map: ResMut<Minimap>) {
    for _ in acts.read().filter(|a| a.0 == Action::Minimap) {
        map.shown = !map.shown;
    }
}

/// The map's viewport follows the window, and is off in the photo view.
fn place(
    mut map: ResMut<Minimap>,
    mut inset: ResMut<Inset>,
    photo: Res<Photo>,
    shown: Option<Res<crate::actions::Shown>>,
    window: Query<&Window, With<PrimaryWindow>>,
    mut camera: Query<&mut Camera, With<MapCamera>>,
) {
    let (Ok(window), Ok(mut camera)) = (window.single(), camera.single_mut()) else {
        return;
    };
    // The map is a camera of its own, drawn over the menus and the help: off while
    // they are open
    let covered = shown.is_some_and(|s| s.help || s.menu.is_some());
    let rect = (map.shown && !photo.on && !covered)
        .then(|| map.layout(window.size()))
        .flatten();
    if map.rect != rect {
        map.rect = rect;
        inset.0 = rect;
    }
    camera.is_active = rect.is_some();
    if let Some(rect) = rect {
        let scale = window.scale_factor();
        let viewport = Viewport {
            physical_position: (rect.min * scale).as_uvec2(),
            physical_size: (rect.size() * scale).as_uvec2().max(UVec2::ONE),
            ..default()
        };
        let same = camera.viewport.as_ref().is_some_and(|v| {
            (v.physical_position, v.physical_size)
                == (viewport.physical_position, viewport.physical_size)
        });
        if !same {
            camera.viewport = Some(viewport);
        }
    }
}

/// The main view's footprint on the sea, clipped to the map: its corners' rays cut
/// with mean sea level (those at or above the horizon carried far out), and its focus.
fn footprint(
    map: Res<Minimap>,
    window: Query<&Window, With<PrimaryWindow>>,
    main: Query<(&Camera, &GlobalTransform, &OrbitCamera)>,
    mut gizmos: Gizmos<Footprint>,
) {
    let (Some(_), Ok(window), Ok((camera, transform, orbit))) =
        (map.rect, window.single(), main.single())
    else {
        return;
    };
    let far = 100.0 * map.half.max_element();
    let [w, h] = window.size().to_array();
    let corners = [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]].map(|c| {
        let ray = camera.viewport_to_world(transform, Vec2::from(c)).ok()?;
        let o = Vec2::new(ray.origin.x, ray.origin.z);
        let flat = Vec2::new(ray.direction.x, ray.direction.z).normalize_or_zero();
        Some(
            match ray.intersect_plane(Vec3::ZERO, InfinitePlane3d::new(Vec3::Y)) {
                Some(t) => {
                    let p = ray.get_point(t);
                    let p = Vec2::new(p.x, p.z);
                    if (p - o).length() > far {
                        o + far * flat
                    } else {
                        p
                    }
                }
                None => o + far * flat,
            },
        )
    });
    let colour = Color::srgb(1.0, 0.85, 0.2);
    if let [Some(a), Some(b), Some(c), Some(d)] = corners {
        let outline = clip(&[a, b, c, d], map.half);
        if outline.len() >= 2 {
            let at = |p: &Vec2| Vec3::new(p.x, 0.0, p.y);
            gizmos.linestrip(outline.iter().chain(&outline[..1]).map(at), colour);
        }
    }
    let r = 0.02 * map.half.max_element();
    let f = Vec3::new(orbit.focus.x, 0.0, orbit.focus.z);
    gizmos.line(f - r * Vec3::X, f + r * Vec3::X, colour);
    gizmos.line(f - r * Vec3::Z, f + r * Vec3::Z, colour);
}

/// The polygon `points` (X, Z) clipped to the box `|x| <= half.x`, `|z| <= half.y`
/// (Sutherland–Hodgman).
fn clip(points: &[Vec2], half: Vec2) -> Vec<Vec2> {
    let mut out = points.to_vec();
    // Each side as (axis, sign): inside where sign * p[axis] <= half[axis]
    for (axis, sign) in [(0, 1.0), (0, -1.0), (1, 1.0), (1, -1.0)] {
        let inside = |p: Vec2| sign * p[axis] <= half[axis];
        let input = std::mem::take(&mut out);
        for (i, &p) in input.iter().enumerate() {
            let q = input[(i + 1) % input.len()];
            if inside(p) {
                out.push(p);
            }
            if inside(p) != inside(q) {
                let t = (sign * half[axis] - p[axis]) / (q[axis] - p[axis]);
                out.push(p + t * (q - p));
            }
        }
        if out.is_empty() {
            break;
        }
    }
    out
}

/// A click on the map moves the main view's focus there.
fn click(
    mouse: Res<ButtonInput<MouseButton>>,
    map: Res<Minimap>,
    window: Query<&Window, With<PrimaryWindow>>,
    mut main: Query<&mut OrbitCamera>,
    mut press: Local<Option<Vec2>>,
    over_ui: Res<OverUi>,
) {
    let (Ok(window), Ok(mut orbit)) = (window.single(), main.single_mut()) else {
        return;
    };
    let cursor = window.cursor_position();
    if mouse.just_pressed(MouseButton::Left) {
        *press = cursor.filter(|&c| !over_ui.0 && map.world_at(c).is_some());
    }
    if mouse.just_released(MouseButton::Left)
        && let (Some(_), Some(up)) = (press.take(), cursor)
        && let Some(at) = map.world_at(up)
    {
        orbit.focus = at;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_footprint_is_clipped_to_the_map() {
        let half = Vec2::new(30.0, 20.0);
        // A view's trapezoid reaching far beyond the map's top
        let cut = clip(
            &[
                Vec2::new(-5.0, 0.0),
                Vec2::new(5.0, 0.0),
                Vec2::new(500.0, -1000.0),
                Vec2::new(-500.0, -1000.0),
            ],
            half,
        );
        assert!(
            cut.iter()
                .all(|p| p.x.abs() <= 30.0 + 1e-4 && p.y.abs() <= 20.0 + 1e-4)
        );
        assert!(cut.iter().any(|p| (p.y + 20.0).abs() < 1e-4));
        assert!(cut.contains(&Vec2::new(-5.0, 0.0)));
        // Inside: unchanged; outside: nothing
        let inside = [Vec2::ZERO, Vec2::X, Vec2::Y];
        assert_eq!(clip(&inside, half), inside.to_vec());
        assert!(
            clip(
                &[
                    Vec2::splat(100.0),
                    Vec2::new(101.0, 100.0),
                    Vec2::splat(101.0)
                ],
                half
            )
            .is_empty()
        );
    }

    #[test]
    fn the_map_keeps_the_domains_aspect_with_north_up() {
        // A domain 60 km east-west by 40 km north-south, about the origin
        let mut map = Minimap::new(Vec2::new(-30e3, -20e3), Vec2::new(30e3, 20e3), true);
        let window = Vec2::new(2000.0, 1125.0);
        let rect = map.layout(window).unwrap();
        assert!((rect.width() / rect.height() - 1.5).abs() < 1e-4);
        assert!((rect.width() - SIZE).abs() < 1e-4);
        assert_eq!(rect.max.y, window.y - BOTTOM);
        map.rect = Some(rect);
        // The map's top-left corner is the framed box's north-west corner
        let nw = map.world_at(rect.min + Vec2::splat(1e-3)).unwrap();
        assert!(nw.x < -0.99 * map.half.x && nw.z < -0.99 * map.half.y);
        // Its centre the origin, and off the map nothing
        let centre = map.world_at(rect.center()).unwrap();
        assert!(centre.length() < 1e-2);
        assert!(map.world_at(rect.max + Vec2::ONE).is_none());
        // A small window: smaller, or none
        assert!(map.layout(Vec2::new(400.0, 200.0)).is_none());
    }
}

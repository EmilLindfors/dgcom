//! An orbit camera: left-drag turns about the focus, right- or middle-drag (or
//! Shift+left-drag) slides the focus over the water, the wheel zooms. [`Views`] are
//! the framings to jump to (in the viewer F the farm, O the whole domain, 1 to 9 the
//! scenario's named places; the jumps are the viewer's actions, not this module's,
//! which `surf_zone` shares).

use bevy::input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll, MouseScrollUnit};
use bevy::prelude::*;
use bevy::window::PrimaryWindow;

#[derive(Component, Clone, Copy, Debug)]
pub struct OrbitCamera {
    pub focus: Vec3,
    /// Angle about +Y from +Z (rad)
    pub yaw: f32,
    /// Angle above the horizontal (rad)
    pub pitch: f32,
    pub distance: f32,
}

impl OrbitCamera {
    pub fn transform(&self) -> Transform {
        let dir = Vec3::new(
            self.pitch.cos() * self.yaw.sin(),
            self.pitch.sin(),
            self.pitch.cos() * self.yaw.cos(),
        );
        Transform::from_translation(self.focus + self.distance * dir)
            .looking_at(self.focus, Vec3::Y)
    }
}

/// The framings F, O and the number keys return to.
#[derive(Resource, Clone, Debug)]
pub struct Views {
    pub farm: OrbitCamera,
    pub domain: OrbitCamera,
    /// Named places, framed as the farm is: keys 1 to 9
    pub places: Vec<(String, OrbitCamera)>,
}

/// A rectangle of the window (logical px) over which drags do not move the view: an
/// inset such as the minimap, which takes its own clicks.
#[derive(Resource, Default)]
pub struct Inset(pub Option<Rect>);

/// Whether the cursor is over the menus or the help ([`crate::menubar`]), which take
/// its clicks and the wheel.
#[derive(Resource, Default)]
pub struct OverUi(pub bool);

pub struct CameraPlugin;

impl Plugin for CameraPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, (input, place).chain());
    }
}

#[allow(clippy::too_many_arguments)] // a Bevy system's parameters are its queries
fn input(
    mouse: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    motion: Res<AccumulatedMouseMotion>,
    scroll: Res<AccumulatedMouseScroll>,
    inset: Option<Res<Inset>>,
    over_ui: Option<Res<OverUi>>,
    window: Query<&Window, With<PrimaryWindow>>,
    mut cameras: Query<&mut OrbitCamera>,
    mut on_inset: Local<bool>,
) {
    let Ok(mut orbit) = cameras.single_mut() else {
        return;
    };
    let over_ui = over_ui.is_some_and(|o| o.0);
    // A drag that starts on the inset or the menus does not move the view
    if mouse.any_just_pressed([MouseButton::Left, MouseButton::Right, MouseButton::Middle]) {
        let cursor = window.single().ok().and_then(Window::cursor_position);
        let rect = inset.as_ref().and_then(|inset| inset.0);
        *on_inset = over_ui || matches!((cursor, rect), (Some(c), Some(r)) if r.contains(c));
    }
    let d = if *on_inset { Vec2::ZERO } else { motion.delta };
    let shift = keys.any_pressed([KeyCode::ShiftLeft, KeyCode::ShiftRight]);
    let pan = mouse.pressed(MouseButton::Right)
        || mouse.pressed(MouseButton::Middle)
        || (shift && mouse.pressed(MouseButton::Left));
    if pan && d != Vec2::ZERO {
        // Along the ground: the view's right and its forward flattened.
        let (sin, cos) = orbit.yaw.sin_cos();
        let right = Vec3::new(cos, 0.0, -sin);
        let forward = Vec3::new(-sin, 0.0, -cos);
        let k = 0.0012 * orbit.distance;
        orbit.focus += (-right * d.x + forward * d.y) * k;
    } else if mouse.pressed(MouseButton::Left) && d != Vec2::ZERO {
        orbit.yaw -= 0.005 * d.x;
        orbit.pitch = (orbit.pitch + 0.005 * d.y).clamp(0.02, 1.55);
    }
    let notches = match scroll.unit {
        MouseScrollUnit::Line => scroll.delta.y,
        MouseScrollUnit::Pixel => scroll.delta.y / 40.0,
    };
    if notches != 0.0 && !over_ui {
        orbit.distance = (orbit.distance * (-0.12 * notches).exp()).clamp(5.0, 200_000.0);
    }
}

fn place(mut cameras: Query<(&OrbitCamera, &mut Transform), Changed<OrbitCamera>>) {
    for (orbit, mut transform) in &mut cameras {
        *transform = orbit.transform();
    }
}

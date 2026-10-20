//! What the water does at a point on the map: hover for a readout at the cursor, click
//! to pin a point.
//!
//! The cursor's ray is cut with mean sea level (the water's height is exaggerated
//! `vz` times, but a tide of a metre moves the point by a few metres at most), and the
//! point located in the mesh. The readout is the element polynomial's value there at
//! the shown time ([`Probe`], linear in time between the snapshots): the position (with
//! latitude and longitude on a real coast), the depth at rest and now, the surface, the
//! current (depth mean, and in 3D at the surface and the bed) with the bearing it flows
//! towards, and in 3D the temperature and salinity at the surface and the bed (from
//! the element's wet nodes: [`Probe::eval_level_wet`]) and the
//! depths of the strongest density and salinity steps
//! ([`crate::stratification`]); in a run with waves, the sea state: H_s, T_p, the
//! direction the waves come from and their spread, and the power whitecapping and
//! breaking take from them.
//!
//! A click (a press and release of the left button that did not drag the view) pins
//! the point: a marker stands in the water there, and a panel under the gauge trace
//! keeps its readout, in 3D with the profiles of temperature, salinity and current
//! speed over the depth, following the playback. Esc lets go.

use bevy::asset::RenderAssetUsages;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy::text::FontSize;
use bevy::window::PrimaryWindow;
use dg_rs::io::{CoordinateProjection, LocalProjection};
use dg_rs::mesh::{Mesh2D, PointLocator2D};
use dg_rs::operators::DGOperators2D;
use std::sync::Arc;

use crate::actions::{Act, Action};
use crate::camera::{OrbitCamera, OverUi};
use crate::field::{Frame, Probe};
use crate::minimap::Minimap;
use crate::playback::Playback;
use crate::plot::Canvas;
use crate::solver::H_DRY;
use crate::stratification::{self, SheetShows, column_strongest};

/// The profile plot in pixels (drawn at half this size on screen).
const WIDTH: usize = 820;
const HEIGHT: usize = 480;
/// A click moves the cursor less than this (px) between press and release.
const CLICK: f32 = 5.0;

const TEMPERATURE: [u8; 3] = [255, 140, 60];
const SALINITY: [u8; 3] = [90, 200, 255];
const SPEED: [u8; 3] = [240, 240, 240];
const PYCNOCLINE: [u8; 3] = [255, 225, 80];

/// Where the inspector looks, and what it holds.
#[derive(Resource)]
pub struct Inspector {
    /// Over a mesh that lives as long as the viewer (see [`Self::new`])
    locator: PointLocator2D<'static>,
    ops: Arc<DGOperators2D>,
    projection: Option<LocalProjection>,
    /// Bed elevation at every node (m)
    bed: Vec<f32>,
    /// σ of the layer centres, bed up, in a 3D run
    sigma: Option<Vec<f32>>,
    /// The point under the cursor, and the pinned one
    hover: Option<Spot>,
    pinned: Option<Spot>,
    /// Where the left button went down, to tell a click from a drag
    press: Option<Vec2>,
}

impl Inspector {
    /// The inspector of the mesh `mesh`: it keeps a reference to the mesh for the
    /// rest of the program (the viewer's mesh never goes away), so its locator can
    /// live in a resource.
    pub fn new(
        mesh: Arc<Mesh2D>,
        ops: Arc<DGOperators2D>,
        projection: Option<LocalProjection>,
        bed: &[f64],
        sigma: Option<Vec<f32>>,
    ) -> Self {
        let mesh: &'static Mesh2D = Box::leak(Box::new(mesh));
        Self {
            locator: PointLocator2D::new(mesh),
            ops,
            projection,
            bed: bed.iter().map(|&b| b as f32).collect(),
            sigma,
            hover: None,
            pinned: None,
            press: None,
        }
    }
}

/// A point of the map: where, and its element's probe (none outside the mesh).
#[derive(Clone)]
struct Spot {
    at: [f64; 2],
    probe: Option<Probe>,
}

/// The water column at a point, at the shown time.
struct Column {
    /// Layer centres' depth below the surface (m), bed up
    depth: Vec<f32>,
    temp: Vec<f32>,
    salt: Vec<f32>,
    speed: Vec<f32>,
    /// u, v of the bed and the surface layer
    bed_current: [f32; 2],
    surface_current: [f32; 2],
    /// Depth (m) and strength of the strongest density (N², s⁻²) and salinity (per m)
    /// steps
    pycnocline: Option<(f32, f32)>,
    halocline: Option<(f32, f32)>,
}

/// The readout of a point.
struct Readout {
    /// Bed depth below mean sea level (m)
    bed: f32,
    eta: f32,
    /// Depth-mean current
    current: [f32; 2],
    column: Option<Column>,
    sea: Option<Sea>,
}

/// The sea state at a point.
struct Sea {
    /// H_s (m), T_p (s)
    hs: f32,
    tp: f32,
    /// Mean direction of travel, mesh x and y (not normalised)
    toward: [f32; 2],
    /// Directional spread (rad)
    spread: f32,
    /// Power whitecapping and breaking take (W/m²)
    whitecapping: f32,
    breaking: f32,
}

impl Inspector {
    /// Pin the point at mesh coordinates `at` (m), as a click on it does.
    pub fn pin(&mut self, at: [f64; 2]) {
        self.pinned = Some(Spot {
            at,
            probe: Probe::with_operators(&self.locator, &self.ops, at),
        });
    }

    pub fn is_pinned(&self) -> bool {
        self.pinned.is_some()
    }

    /// The point's readout at the playback's time (`None` outside the mesh or before a
    /// snapshot).
    fn readout(&self, spot: &Spot, playback: &Playback) -> Option<Readout> {
        let probe = spot.probe.as_ref()?;
        let (a, b, w) = playback.bracket()?;
        let lerp = |x: f32, y: f32| x + w * (y - x);
        let both = |fa: &[f32], fb: &[f32]| lerp(probe.eval(fa), probe.eval(fb));
        let bed = probe.eval(&self.bed);
        let eta = both(&a.eta, &b.eta);
        let current = [both(&a.u, &b.u), both(&a.v, &b.v)];
        let column = match (&self.sigma, &a.layers, &b.layers) {
            (Some(sigma), Some(la), Some(lb)) if eta - bed > H_DRY => {
                let nl = la.n_levels;
                let level = |fa: &[f32], fb: &[f32], l: usize| {
                    lerp(probe.eval_level(fa, nl, l), probe.eval_level(fb, nl, l))
                };
                // The tracers from the element's wet nodes only
                let tracer = |fa: &[f32], fb: &[f32], l: usize| {
                    let wet =
                        |field, eta| probe.eval_level_wet(field, nl, l, [eta, &self.bed], H_DRY);
                    lerp(wet(fa, &a.eta), wet(fb, &b.eta))
                };
                let has_salt = !la.salt.is_empty() && !lb.salt.is_empty();
                let total = eta - bed;
                let z: Vec<f32> = sigma.iter().map(|&s| eta + s * total).collect();
                let temp: Vec<f32> = (0..nl).map(|l| tracer(&la.temp, &lb.temp, l)).collect();
                let salt: Vec<f32> = if has_salt {
                    (0..nl).map(|l| tracer(&la.salt, &lb.salt, l)).collect()
                } else {
                    Vec::new()
                };
                let uv: Vec<[f32; 2]> = (0..nl)
                    .map(|l| [level(&la.u, &lb.u, l), level(&la.v, &lb.v, l)])
                    .collect();
                // As the sheet: none in a column too shallow to be stratified
                let step = |shows| {
                    (has_salt && total >= stratification::MIN_DEPTH)
                        .then(|| column_strongest(shows, &z, &temp, &salt))
                        .flatten()
                        .map(|(height, strength)| (eta - height, strength))
                };
                Some(Column {
                    depth: z.iter().map(|z| eta - z).collect(),
                    pycnocline: step(SheetShows::Density),
                    halocline: step(SheetShows::Salinity),
                    speed: uv.iter().map(|[u, v]| u.hypot(*v)).collect(),
                    bed_current: uv[0],
                    surface_current: uv[nl - 1],
                    temp,
                    salt,
                })
            }
            _ => None,
        };
        let sea = match (&a.waves, &b.waves) {
            (Some(wa), Some(wb)) => Some(Sea {
                hs: both(&wa.hs, &wb.hs),
                tp: both(&wa.tp, &wb.tp),
                toward: [both(&wa.dir_x, &wb.dir_x), both(&wa.dir_y, &wb.dir_y)],
                spread: both(&wa.spread, &wb.spread),
                whitecapping: both(&wa.whitecapping, &wb.whitecapping).max(0.0),
                breaking: both(&wa.breaking, &wb.breaking).max(0.0),
            }),
            _ => None,
        };
        Some(Readout {
            bed: -bed,
            eta,
            current,
            column,
            sea,
        })
    }

    /// The point's place: mesh km, and latitude and longitude on a real coast.
    fn place(&self, [x, y]: [f64; 2]) -> String {
        let km = format!("{:.2}, {:.2} km", x / 1e3, y / 1e3);
        match &self.projection {
            Some(p) => {
                let (lat, lon) = p.xy_to_geo(x, y);
                format!("{km} ({:.4}° N, {:.4}° E)", lat, lon)
            }
            None => km,
        }
    }
}

/// Speed and the compass bearing it flows towards.
fn current([u, v]: [f32; 2]) -> String {
    let speed = u.hypot(v);
    if speed < 5e-4 {
        return "0.00 m/s".into();
    }
    let bearing = u.atan2(v).to_degrees().rem_euclid(360.0);
    format!("{speed:.2} m/s towards {bearing:03.0}°")
}

/// The sea state as a line: H_s and T_p, the compass bearing the waves come from (as
/// wave directions are given) and their spread; with `full`, a second with the power
/// they lose.
fn sea(s: &Sea, full: bool) -> String {
    if s.hs < 0.005 {
        return "\nwaves: none".into();
    }
    let [x, y] = s.toward;
    let from = (x.atan2(y).to_degrees() + 180.0).rem_euclid(360.0);
    let mut line = format!(
        "\nwaves H_s {:.2} m, T_p {:.1} s, from {from:03.0}° ({}), spread {:.0}°",
        s.hs,
        s.tp,
        crate::photo::compass(from),
        s.spread.to_degrees()
    );
    if full {
        // About three figures, none below a milliwatt
        let power = |x: f32| match x {
            x if x >= 100.0 => format!("{x:.0}"),
            x if x >= 1.0 => format!("{x:.1}"),
            x if x >= 1e-3 => format!("{x:.3}"),
            _ => "0".into(),
        };
        line += &format!(
            "\n  losing {} W/m² to whitecapping, {} to breaking",
            power(s.whitecapping),
            power(s.breaking)
        );
    }
    line
}

/// The readout as lines: `full` adds the currents at the surface and the bed, and
/// the power the waves lose.
fn describe(r: &Readout, full: bool) -> String {
    let water = r.eta + r.bed;
    let mut s = if water <= H_DRY {
        format!("dry: bed {:+.1} m above mean sea level", -r.bed)
    } else {
        format!(
            "{:.1} m deep at rest, {water:.1} m now; surface {:+.2} m",
            r.bed, r.eta
        )
    };
    if water > H_DRY {
        s += &format!("\ncurrent (depth mean) {}", current(r.current));
        if let Some(w) = &r.sea {
            s += &sea(w, full);
        }
    }
    if let Some(c) = &r.column {
        if full {
            s += &format!(
                "\n  at the surface {}\n  at the bed {}",
                current(c.surface_current),
                current(c.bed_current)
            );
        }
        let (top, bottom) = (c.temp.len() - 1, 0);
        s += &format!(
            "\nT {:.2} °C at the surface, {:.2} °C at the bed",
            c.temp[top], c.temp[bottom]
        );
        if !c.salt.is_empty() {
            s += &format!(
                "\nS {:.2} at the surface, {:.2} at the bed",
                c.salt[top], c.salt[bottom]
            );
        }
        let step = |name: &str, step: Option<(f32, f32)>, unit: &str| match step {
            Some((depth, strength)) => {
                format!("\n{name} {depth:.1} m deep ({strength:.1e} {unit})")
            }
            None => format!("\n{name}: none (mixed)"),
        };
        if !c.salt.is_empty() {
            s += &step("strongest density step (N²)", c.pycnocline, "s⁻²");
            s += &step("strongest salinity step", c.halocline, "/m");
        }
    }
    s
}

#[derive(Component)]
struct Tooltip;
#[derive(Component)]
struct TooltipText;
#[derive(Component)]
struct PinPanel;
#[derive(Component)]
enum PinLabel {
    Title,
    Body,
    Profile,
}
#[derive(Component)]
struct PinMarker;
#[derive(Component)]
struct ProfileNode;
#[derive(Resource)]
struct ProfileImage(Handle<Image>);

pub struct InspectPlugin;

impl Plugin for InspectPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn).add_systems(
            Update,
            (
                pick,
                (tooltip, pinned, marker).after(crate::field::interpolate),
            )
                .chain(),
        );
    }
}

fn font(size: f32) -> TextFont {
    TextFont {
        font_size: FontSize::Px(size),
        ..default()
    }
}

fn spawn(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let shadow = TextShadow {
        offset: Vec2::splat(1.5),
        color: Color::srgba(0.0, 0.0, 0.0, 0.85),
    };
    commands
        .spawn((
            Tooltip,
            Node {
                position_type: PositionType::Absolute,
                padding: UiRect::axes(Val::Px(8.0), Val::Px(5.0)),
                ..default()
            },
            BackgroundColor(Color::srgba(0.02, 0.03, 0.05, 0.8)),
            Visibility::Hidden,
            GlobalZIndex(10),
        ))
        .with_children(|tip| {
            tip.spawn((TooltipText, Text::new(""), font(13.0)));
        });

    let image = images.add(Image::new_fill(
        Extent3d {
            width: WIDTH as u32,
            height: HEIGHT as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0; 4],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    ));
    commands.insert_resource(ProfileImage(image.clone()));
    commands
        .spawn((
            PinPanel,
            Node {
                position_type: PositionType::Absolute,
                right: Val::Px(14.0),
                top: Val::Px(crate::menubar::BAR_HEIGHT + 230.0),
                width: Val::Px((WIDTH / 2) as f32),
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(4.0),
                padding: UiRect::all(Val::Px(8.0)),
                ..default()
            },
            BackgroundColor(Color::srgba(0.02, 0.03, 0.05, 0.8)),
            Visibility::Hidden,
        ))
        .with_children(|panel| {
            panel.spawn((PinLabel::Title, Text::new(""), font(13.0), shadow));
            panel.spawn((PinLabel::Body, Text::new(""), font(12.0), shadow));
            panel.spawn((
                ProfileNode,
                ImageNode::new(image),
                Node {
                    width: Val::Px((WIDTH / 2) as f32),
                    height: Val::Px((HEIGHT / 2) as f32),
                    ..default()
                },
            ));
            panel.spawn((PinLabel::Profile, Text::new(""), font(11.0), shadow));
        });

    commands.spawn((
        PinMarker,
        Mesh3d(meshes.add(Cuboid::new(1.0, 1.0, 1.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(1.0, 0.35, 0.1),
            unlit: true,
            ..default()
        })),
        Transform::default(),
        Visibility::Hidden,
    ));
}

/// The point under the cursor, and clicks that pin it.
#[allow(clippy::too_many_arguments)] // a Bevy system's parameters are its queries
fn pick(
    mouse: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    mut acts: MessageReader<Act>,
    over_ui: Res<OverUi>,
    window: Query<&Window, With<PrimaryWindow>>,
    camera: Query<(&Camera, &GlobalTransform), With<OrbitCamera>>,
    frame: Res<Frame>,
    minimap: Res<Minimap>,
    mut inspector: ResMut<Inspector>,
) {
    if acts.read().any(|a| a.0 == Action::Unpin) {
        inspector.pinned = None;
    }
    let (Ok(window), Ok((camera, transform))) = (window.single(), camera.single()) else {
        return;
    };
    let cursor = window.cursor_position();
    // The point on the minimap, or the ray under the cursor cut with mean sea level
    let on_map = cursor.and_then(|c| minimap.world_at(c));
    let hover = on_map
        .or_else(|| {
            let ray = camera.viewport_to_world(transform, cursor?).ok()?;
            let distance = ray.intersect_plane(Vec3::ZERO, InfinitePlane3d::new(Vec3::Y))?;
            Some(ray.get_point(distance))
        })
        .map(|p| [p.x as f64 + frame.origin[0], -p.z as f64 + frame.origin[1]])
        .map(|at| Spot {
            at,
            probe: Probe::with_operators(&inspector.locator, &inspector.ops, at),
        });
    // A click on the minimap moves the view instead ([`crate::minimap`])
    if mouse.just_pressed(MouseButton::Left) {
        inspector.press = cursor.filter(|_| on_map.is_none() && !over_ui.0);
    }
    if mouse.just_released(MouseButton::Left) {
        let shift = keys.any_pressed([KeyCode::ShiftLeft, KeyCode::ShiftRight]);
        if let (Some(down), Some(up)) = (inspector.press.take(), cursor)
            && down.distance(up) < CLICK
            && !shift
            && let Some(spot) = hover.as_ref().filter(|s| s.probe.is_some())
        {
            inspector.pinned = Some(spot.clone());
        }
    }
    inspector.hover = hover;
}

fn tooltip(
    inspector: Res<Inspector>,
    playback: Res<Playback>,
    mouse: Res<ButtonInput<MouseButton>>,
    window: Query<&Window, With<PrimaryWindow>>,
    mut tip: Query<(&mut Node, &mut Visibility), With<Tooltip>>,
    mut text: Query<&mut Text, With<TooltipText>>,
) {
    let (Ok((mut node, mut visible)), Ok(mut text), Ok(window)) =
        (tip.single_mut(), text.single_mut(), window.single())
    else {
        return;
    };
    // Not while the view is dragged
    let dragging = mouse.any_pressed([MouseButton::Left, MouseButton::Right, MouseButton::Middle]);
    let (Some(spot), Some(cursor), false) = (&inspector.hover, window.cursor_position(), dragging)
    else {
        visible.set_if_neq(Visibility::Hidden);
        return;
    };
    let s = match inspector.readout(spot, &playback) {
        Some(r) => format!("{}\n{}", inspector.place(spot.at), describe(&r, false)),
        None if spot.probe.is_none() => format!("{}\noutside the model", inspector.place(spot.at)),
        None => return,
    };
    if text.0 != s {
        text.0 = s;
    }
    // Beside the cursor, kept on the screen
    let size = window.size();
    let (x, y) = (cursor.x + 18.0, cursor.y + 18.0);
    node.left = Val::Px(x.min(size.x - 360.0).max(0.0));
    node.top = Val::Px(y.min(size.y - 170.0).max(0.0));
    visible.set_if_neq(Visibility::Inherited);
}

#[allow(clippy::too_many_arguments)] // a Bevy system's parameters are its queries
fn pinned(
    inspector: Res<Inspector>,
    playback: Res<Playback>,
    plot: Res<ProfileImage>,
    mut images: ResMut<Assets<Image>>,
    mut panel: Query<&mut Visibility, With<PinPanel>>,
    mut profile: Query<&mut Node, With<ProfileNode>>,
    mut labels: Query<(&PinLabel, &mut Text)>,
) {
    let Ok(mut visible) = panel.single_mut() else {
        return;
    };
    let Some(spot) = &inspector.pinned else {
        visible.set_if_neq(Visibility::Hidden);
        return;
    };
    if !(playback.changed || inspector.is_changed()) {
        return;
    }
    let Some(readout) = inspector.readout(spot, &playback) else {
        return;
    };
    visible.set_if_neq(Visibility::Inherited);
    let column = readout.column.as_ref();
    if let Ok(mut node) = profile.single_mut() {
        node.display = if column.is_some() {
            Display::Flex
        } else {
            Display::None
        };
    }
    let ranges = column.map(|c| draw_profile(c, &plot.0, &mut images));
    for (label, mut text) in &mut labels {
        let s = match label {
            PinLabel::Title => format!("pinned at {} (Esc)", inspector.place(spot.at)),
            PinLabel::Body => describe(&readout, true),
            PinLabel::Profile => ranges.clone().unwrap_or_default(),
        };
        if text.0 != s {
            text.0 = s;
        }
    }
}

/// One profile of the plot: its values (bed up), colour, the least range it is
/// drawn over, name and unit.
type Series<'a> = (&'a [f32], [u8; 3], f32, &'static str, &'static str);

/// The column's profiles over its depth into `image`: three panels side by side
/// (temperature, salinity, speed), the surface at the top, a dashed line at the
/// strongest density step. Returns their ranges, for the legend under them.
fn draw_profile(column: &Column, image: &Handle<Image>, images: &mut Assets<Image>) -> String {
    let mut canvas = Canvas::new(WIDTH, HEIGHT);
    let deepest = column.depth.first().copied().unwrap_or(1.0).max(1.0);
    let series: Vec<Series> = [
        (&column.temp[..], TEMPERATURE, 0.1, "T", " °C"),
        (&column.salt[..], SALINITY, 0.02, "S", ""),
        (&column.speed[..], SPEED, 0.02, "speed", " m/s"),
    ]
    .into_iter()
    .filter(|(v, ..)| !v.is_empty())
    .collect();
    let panel = WIDTH / series.len().max(1);
    let margin = 12.0;
    let y_of = |d: f32| margin + (d / deepest) * (HEIGHT as f32 - 2.0 * margin);
    let range = |v: &[f32], floor: f32| {
        let (lo, hi) = v
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &x| {
                (a.min(x), b.max(x))
            });
        let pad = (0.08 * (hi - lo)).max(0.5 * floor);
        [lo - pad, hi + pad]
    };
    let mut legend = Vec::new();
    for (p, (values, colour, floor, name, unit)) in series.into_iter().enumerate() {
        let [lo, hi] = range(values, floor);
        let x0 = (p * panel) as f32 + margin;
        let width = panel as f32 - 2.0 * margin;
        let x_of = |v: f32| x0 + (v - lo) / (hi - lo) * width;
        // The panel's frame: its left edge, and the surface
        for y in 0..HEIGHT {
            canvas.stamp(x0 as usize, y, [120, 120, 120], 0.5);
        }
        let points: Vec<(f32, f32)> = values
            .iter()
            .zip(&column.depth)
            .map(|(&v, &d)| (x_of(v), y_of(d)))
            .collect();
        for pair in points.windows(2) {
            canvas.segment(pair[0], pair[1], colour, 3.0);
        }
        for &(x, y) in &points {
            canvas.dot(x, y, colour, 5.0);
        }
        legend.push(format!("{name} {lo:.2}–{hi:.2}{unit}"));
    }
    if let Some((depth, _)) = column.pycnocline {
        let y = y_of(depth).round() as usize;
        canvas.dashed_row(y, [0, WIDTH], PYCNOCLINE, 0.9);
    }
    if let Some(mut image) = images.get_mut(image) {
        image.data = Some(canvas.rgba);
    }
    format!(
        "{}\nsurface at the top, {deepest:.1} m at the bottom; dashed: strongest density step",
        legend.join("   ")
    )
}

/// The pinned point's marker: a post from the bed to above the surface, as thick as
/// a few pixels at the camera's distance.
fn marker(
    inspector: Res<Inspector>,
    playback: Res<Playback>,
    frame: Res<Frame>,
    camera: Query<&OrbitCamera>,
    mut post: Query<(&mut Transform, &mut Visibility), With<PinMarker>>,
) {
    let (Ok((mut transform, mut visible)), Ok(orbit)) = (post.single_mut(), camera.single()) else {
        return;
    };
    let Some((spot, readout)) = inspector
        .pinned
        .as_ref()
        .and_then(|s| Some((s, inspector.readout(s, &playback)?)))
    else {
        visible.set_if_neq(Visibility::Hidden);
        return;
    };
    let bottom = frame.world(spot.at, -readout.bed);
    let top = frame.world(spot.at, readout.eta.max(-readout.bed)) + Vec3::Y * 0.03 * orbit.distance;
    let thickness = 0.002 * orbit.distance;
    *transform = Transform::from_translation(0.5 * (bottom + top)).with_scale(Vec3::new(
        thickness,
        (top.y - bottom.y).max(thickness),
        thickness,
    ));
    visible.set_if_neq(Visibility::Inherited);
}

//! What is shown and how far the solver has got (top left, under the menu bar), and
//! the colour scales of the water and of the bed's depth (bottom left). In a 3D run
//! the status also names the layer the water shows and the particles of each kind.
//! The photo view ([`crate::photo`]) has no water colour scale; the status names its
//! wind. The keys are in the menus and the help popup ([`crate::menubar`]); the
//! status's hints take their names from the same table ([`Actions`]).

use bevy::prelude::*;
use bevy::text::FontSize;
use bevy::ui::{BackgroundGradient, ColorStop, LinearGradient};
use dg_rs::time::ModelClock;

use crate::actions::{Action, Actions};
use crate::camera::OrbitCamera;
use crate::cloud_3d::KINDS;
use crate::colormap::{SEABED, srgb_at};
use crate::contours::Contours;
use crate::field::{Frame, ShownLayer};
use crate::layers::Levels;
use crate::menubar::BAR_HEIGHT;
use crate::particles::{ACTIVE, DEAD, EXITED, Particles, SETTLED, STRANDED};
use crate::photo::Photo;
use crate::playback::{Playback, SolverState, Source};
use crate::stratification::{SheetShows, Stratification};
use crate::surface::{BedScale, ColourBy, Colouring, SurfaceStyle, WaterOpacity};

/// The scenario's one-line description, heading the status.
#[derive(Resource)]
pub struct Title(pub String);

/// The UTC of model time 0, when the run has a date: the status then shows the date
/// next to the model time.
#[derive(Resource)]
pub struct RunClock(pub ModelClock);

#[derive(Component)]
struct Status;

#[derive(Component)]
struct Bar;

/// The labels under the colour bar: low, middle, high.
#[derive(Component)]
struct Tick(usize);

#[derive(Component)]
struct ScaleTitle;

/// The bed's depth scale: its title, its bar and its labels (shallow, middle, deep).
#[derive(Component)]
struct BedTitle;

#[derive(Component)]
struct BedBar;

#[derive(Component)]
struct BedTick(usize);

pub struct HudPlugin;

impl Plugin for HudPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn)
            .add_systems(Update, (status, scale, bed_scale, photo_scale));
    }
}

fn shadow() -> TextShadow {
    TextShadow {
        offset: Vec2::splat(1.5),
        color: Color::srgba(0.0, 0.0, 0.0, 0.85),
    }
}

fn font(size: f32) -> TextFont {
    TextFont {
        font_size: FontSize::Px(size),
        ..default()
    }
}

fn spawn(mut commands: Commands) {
    commands.spawn((
        Status,
        Text::new(""),
        font(15.0),
        shadow(),
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(14.0),
            top: Val::Px(BAR_HEIGHT + 10.0),
            ..default()
        },
    ));
    commands
        .spawn(Node {
            position_type: PositionType::Absolute,
            left: Val::Px(14.0),
            bottom: Val::Px(14.0),
            width: Val::Px(260.0),
            flex_direction: FlexDirection::Column,
            row_gap: Val::Px(4.0),
            ..default()
        })
        .with_children(|scale| {
            scale.spawn((BedTitle, Text::new("bed depth (m)"), font(13.0), shadow()));
            scale.spawn((
                BedBar,
                Node {
                    width: Val::Percent(100.0),
                    height: Val::Px(12.0),
                    ..default()
                },
                BackgroundGradient::default(),
            ));
            scale
                .spawn(Node {
                    justify_content: JustifyContent::SpaceBetween,
                    margin: UiRect::bottom(Val::Px(8.0)),
                    ..default()
                })
                .with_children(|ticks| {
                    for i in 0..3 {
                        ticks.spawn((BedTick(i), Text::new(""), font(12.0), shadow()));
                    }
                });
            scale.spawn((ScaleTitle, Text::new(""), font(13.0), shadow()));
            scale.spawn((
                Bar,
                Node {
                    width: Val::Percent(100.0),
                    height: Val::Px(12.0),
                    ..default()
                },
                BackgroundGradient::default(),
            ));
            scale
                .spawn(Node {
                    justify_content: JustifyContent::SpaceBetween,
                    ..default()
                })
                .with_children(|ticks| {
                    for i in 0..3 {
                        ticks.spawn((Tick(i), Text::new(""), font(12.0), shadow()));
                    }
                });
        });
}

/// `3 h 07 min`, from seconds.
fn clock(t: f64) -> String {
    let minutes = (t / 60.0).floor() as u64;
    format!("{} h {:02} min", minutes / 60, minutes % 60)
}

#[allow(clippy::too_many_arguments)] // a Bevy system's parameters are its queries
fn status(
    playback: Res<Playback>,
    title: Res<Title>,
    source: Res<Source>,
    run_clock: Option<Res<RunClock>>,
    (style, opacity): (Res<SurfaceStyle>, Res<WaterOpacity>),
    particles: Res<Particles>,
    shown: Res<ShownLayer>,
    levels: Option<Res<Levels>>,
    sheet: Option<Res<Stratification>>,
    photo: Res<Photo>,
    frame: Res<Frame>,
    sun: Option<Res<crate::photo::Sun>>,
    sea: Option<Res<crate::photo::ShownSea>>,
    camera: Query<&OrbitCamera>,
    actions: Res<Actions>,
    mut text: Query<&mut Text, With<Status>>,
) {
    let key = |action| actions.key_name(action);
    let Ok(mut text) = text.single_mut() else {
        return;
    };
    let mut s = format!("{}\n", title.0);
    let replay = matches!(*source, Source::Replay { .. });
    match playback.newest() {
        None if replay => s += "reading the frames...\n",
        None => s += "starting the solver...\n",
        Some(_) => {
            let state = if playback.paused {
                "  paused"
            } else if playback.waiting() && replay {
                "  waiting for the frames"
            } else if playback.waiting() {
                "  waiting for the solver"
            } else {
                ""
            };
            // The date to the minute, `YYYY-MM-DD HH:MM`
            let date = run_clock.map_or(String::new(), |c| {
                format!(" ({} UTC)", &c.0.format(playback.t)[..16])
            });
            s += &format!(
                "t = {}{date}   ×{:.0}{state}\n",
                clock(playback.t),
                playback.rate
            );
        }
    }
    let newest = playback.newest().unwrap_or(0.0);
    s += &match (&*source, &playback.solver) {
        (Source::Solver, SolverState::Running) => format!(
            "solver at {}, {:.0}x real time\n",
            clock(newest),
            playback.solver_speed()
        ),
        (Source::Solver, SolverState::Finished { steps, wall }) => format!(
            "solver done: {} in {steps} steps, {wall:.0} s\n",
            clock(newest)
        ),
        (Source::Solver, SolverState::Failed(e)) => format!("solver failed: {e}\n"),
        (Source::Replay { name, t_last, .. }, SolverState::Running) => format!(
            "replay of {name}: read to {} of {}\n",
            clock(newest),
            clock(*t_last)
        ),
        (Source::Replay { name, .. }, SolverState::Finished { .. })
            if let Some(file) = &playback.on_demand =>
        {
            // A file a run is still writing
            let growing = match file.grew {
                Some(wall) if playback.growing() => {
                    let ago = playback.started_secs() - wall;
                    format!(", growing (last frame {ago:.0} s ago)")
                }
                _ => String::new(),
            };
            format!(
                "replay of {name}: {} frames {} apart{growing}, read as shown,\n\
                 linear in time between them\n",
                file.times.len(),
                clock(playback.interval)
            )
        }
        (Source::Replay { name, frames, .. }, SolverState::Finished { wall, .. }) => format!(
            "replay of {name}: {frames} frames {} apart (read in {wall:.0} s),\n\
             linear in time between them\n",
            clock(playback.interval)
        ),
        (Source::Replay { .. }, SolverState::Failed(e)) => format!("replay failed: {e}\n"),
    };
    if let Some(file) = &playback.on_demand {
        s += &format!(
            "holding {} of {} frames ({:.0} MB)",
            playback.frames.len(),
            file.times.len(),
            playback.bytes() as f64 / 1e6
        );
    } else if let (Some(lo), Some(hi)) = (playback.oldest(), playback.newest()) {
        s += &format!(
            "kept {} to {} ({} snapshots, {:.0} MB)",
            clock(lo),
            clock(hi),
            playback.frames.len(),
            playback.bytes() as f64 / 1e6
        );
    }
    if let Some(levels) = &levels {
        let n = levels.sigma.len();
        s += &match *shown {
            ShownLayer::DepthMean => format!(
                "\ncurrent: depth mean ({} {})",
                key(Action::LayerUp),
                key(Action::LayerDown)
            ),
            ShownLayer::Level(l) => format!(
                "\ncurrent: layer {} of {n} from the surface, {:.1} m deep ({} {})",
                n - l,
                -levels.sigma[l] * levels.depth,
                key(Action::LayerUp),
                key(Action::LayerDown)
            ),
        };
    }
    if let Some(sheet) = &sheet {
        let (name, unit) = sheet.shows.describe();
        s += &match (sheet.shows, sheet.threshold(), sheet.depth_scale()) {
            (SheetShows::Off, ..) => format!("\nsheet: off ({})", key(Action::Sheet)),
            (_, Some(threshold), Some([shallow, deep])) => format!(
                "\nsheet: {name} at its strongest, {shallow:.0} (yellow) to {deep:.0} m deep \
                 (blue);\n  cut out below {threshold:.1e} {unit}; {} columns ({})",
                sheet.shown_columns,
                key(Action::Sheet)
            ),
            _ => format!("\nsheet: {name} ({})", key(Action::Sheet)),
        };
    }
    let counts = particles.counts;
    if counts.iter().sum::<usize>() > 0 {
        if particles.kinds.is_empty() {
            s += &format!(
                "\nparticles: {} in the water, {} stranded, {} out\n\
                 colour: age, yellow new to dark blue {}; grey stranded",
                counts[ACTIVE as usize],
                counts[STRANDED as usize],
                counts[EXITED as usize],
                clock(particles.age_scale as f64)
            );
        } else {
            let in_water: Vec<String> = KINDS
                .iter()
                .zip(&particles.kinds)
                .map(|(kind, n)| format!("{n} {}", kind.name))
                .collect();
            s += &format!(
                "\nparticles in the water: {}\n{} on the bed (dark), {} dead (grey), {} out\n\
                 colour: pink lice larvae, brown faeces, yellow feed",
                in_water.join(", "),
                counts[SETTLED as usize],
                counts[DEAD as usize],
                counts[EXITED as usize]
            );
        }
        if !particles.on {
            s += &format!(" (hidden, {})", key(Action::Particles));
        }
    }
    s += &match (photo.on, *style) {
        (true, _) if photo.wind_speed > 0.0 => format!(
            "\nphoto view ({}): wind {:.0} m/s from {} ({}, {}), fetch {:.0} km",
            key(Action::Photo),
            photo.wind_speed,
            photo.wind_name(),
            key(Action::Wind),
            key(Action::WindFrom),
            photo.fetch / 1e3
        ),
        (true, _) => format!(
            "\nphoto view ({}): calm ({})",
            key(Action::Photo),
            key(Action::Wind)
        ),
        (false, SurfaceStyle::Translucent) => format!(
            "\nwater {:.0} % opaque ({} {})",
            100.0 * opacity.0,
            key(Action::Clearer),
            key(Action::Denser)
        ),
        (false, SurfaceStyle::Opaque) => {
            format!("\nwater opaque ({})", key(Action::WaterStyle))
        }
        (false, SurfaceStyle::Hidden) => {
            format!("\nwater hidden ({})", key(Action::WaterStyle))
        }
    };
    if photo.on
        && let Some(sun) = &sun
    {
        let place = if sun.real {
            "at the shown time"
        } else {
            "the scene's light"
        };
        s += &format!(
            "\nsun {:.0}° {} the horizon in the {} ({place})",
            sun.elevation.abs(),
            if sun.elevation >= 0.0 {
                "above"
            } else {
                "below"
            },
            crate::photo::compass(sun.azimuth)
        );
    }
    if photo.on
        && let Some(sea) = sea.as_ref().and_then(|s| s.0)
    {
        // World (cos, sin) is mesh (cos, −sin): the bearing it travels toward, then
        // where it comes from
        let toward = sea.toward.cos().atan2(-sea.toward.sin()).to_degrees();
        let from = (toward as f32 + 180.0).rem_euclid(360.0);
        s += &format!(
            "\nthe model's sea here: T_p {:.1} s from the {}, spread {:.0}°; \
             H_s per pixel, the wind's waves shorter",
            sea.tp,
            crate::photo::compass(from),
            sea.spread.to_degrees()
        );
    }
    // The view as `--eye` gives it, to come back to it
    if photo.on
        && let Ok(camera) = camera.single()
    {
        let (at, height, bearing, tilt) = crate::photo::eye_of(camera, &frame);
        s += &format!(
            "\n--eye {:.3},{:.3},{height:.0},{bearing:.0},{tilt:.1}",
            at[0] / 1e3,
            at[1] / 1e3
        );
    }
    if text.0 != s {
        text.0 = s;
    }
}

fn scale(
    colouring: Res<Colouring>,
    mut bar: Query<&mut BackgroundGradient, With<Bar>>,
    mut ticks: Query<(&Tick, &mut Text), Without<ScaleTitle>>,
    mut title: Query<&mut Text, With<ScaleTitle>>,
    mut shown: Local<Option<(ColourBy, [String; 3])>>,
) {
    let (heading, labels) = colouring.legend();
    let key = (colouring.by, labels);
    if shown.as_ref() == Some(&key) {
        return;
    }
    let labels = &key.1;
    let stops = colouring.stops();
    if let Ok(mut bar) = bar.single_mut() {
        let colours = (0..=16)
            .map(|i| ColorStop::from(Color::from(srgb_at(stops, i as f32 / 16.0))))
            .collect();
        *bar = LinearGradient::to_right(colours).into();
    }
    if let Ok(mut title) = title.single_mut() {
        title.0 = heading.into();
    }
    for (tick, mut text) in &mut ticks {
        text.0.clone_from(&labels[tick.0]);
    }
    *shown = Some(key);
}

/// The bed's depth scale, once the bed exists (it is fixed), and its contour interval.
fn bed_scale(
    bed: Option<Res<BedScale>>,
    contours: Option<Res<Contours>>,
    mut bar: Query<&mut BackgroundGradient, With<BedBar>>,
    mut ticks: Query<(&BedTick, &mut Text), Without<BedTitle>>,
    mut title: Query<&mut Text, With<BedTitle>>,
    mut done: Local<bool>,
) {
    if let (Some(contours), Ok(mut title)) = (contours, title.single_mut()) {
        let heading = if contours.on {
            format!("bed depth (m), contour {} m", contours.interval)
        } else {
            "bed depth (m)".to_string()
        };
        if title.0 != heading {
            title.0 = heading;
        }
    }
    let Some(bed) = bed else { return };
    if *done {
        return;
    }
    *done = true;
    if let Ok(mut bar) = bar.single_mut() {
        let colours = (0..=16)
            .map(|i| ColorStop::from(Color::from(srgb_at(SEABED, i as f32 / 16.0))))
            .collect();
        *bar = LinearGradient::to_right(colours).into();
    }
    for (tick, mut text) in &mut ticks {
        let depth = bed.shallow + (bed.deep - bed.shallow) * tick.0 as f32 / 2.0;
        text.0 = format!("{depth:.0}");
    }
}

/// The water's colour scale: its title, bar and labels.
type WaterScale = Or<(With<Bar>, With<Tick>, With<ScaleTitle>)>;

/// The water's colour scale has nothing to say in the photo view.
fn photo_scale(photo: Res<Photo>, mut parts: Query<&mut Visibility, WaterScale>) {
    if !photo.is_changed() {
        return;
    }
    for mut visibility in &mut parts {
        *visibility = if photo.on {
            Visibility::Hidden
        } else {
            Visibility::Inherited
        };
    }
}

//! A run being built: the screen between the list of runs and the viewer.
//!
//! The menu (or Run → Restart) asks for a run with [`Launch`]; entering
//! [`Screen::Loading`] builds it ([`Run::build`]) on a thread of its own, since a
//! scenario on a real coast takes a while (Frøya's bed projection, the terrain),
//! while the window shows what is being built and for how long. A built run is
//! started ([`Run::start`]) and shown; one that fails returns to the menu with the
//! error. Esc returns at once (the build then ends unseen).

use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::mpsc::{Receiver, TryRecvError};
use std::time::Instant;

use bevy::prelude::*;
use bevy::text::FontSize;

use crate::menu::{self, Menu};
use crate::screen::Screen;
use crate::{Failed, Run};

/// The dg-rs checkout: data, meshes and `output/` are found under it.
#[derive(Resource)]
pub struct Repo(pub PathBuf);

/// The run to build on entering [`Screen::Loading`]: its options, and what to call
/// it meanwhile.
#[derive(Resource)]
pub struct Launch {
    pub args: Vec<String>,
    pub title: String,
}

/// A run being built: its title, since when, and where the result arrives.
#[derive(Resource)]
struct Pending {
    title: String,
    started: Instant,
    result: Mutex<Receiver<Result<Run, Failed>>>,
}

#[derive(Component)]
struct Elapsed;

pub struct LoadingPlugin;

impl Plugin for LoadingPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(OnEnter(Screen::Loading), begin)
            .add_systems(
                Update,
                (back, elapsed, arrived)
                    .chain()
                    .run_if(in_state(Screen::Loading)),
            )
            .add_systems(OnExit(Screen::Loading), |mut commands: Commands| {
                commands.remove_resource::<Pending>();
            });
    }
}

fn font(size: f32) -> TextFont {
    TextFont {
        font_size: FontSize::Px(size),
        ..default()
    }
}

/// Start building the [`Launch`]ed run, and show that it is being built.
fn begin(mut commands: Commands, launch: Option<Res<Launch>>, repo: Res<Repo>) {
    let Some(launch) = launch else {
        return;
    };
    commands.remove_resource::<Launch>();
    let (tx, rx) = std::sync::mpsc::channel();
    let (args, repo) = (launch.args.clone(), repo.0.clone());
    std::thread::Builder::new()
        .name("dg-viz build".into())
        .spawn(move || {
            let _ = tx.send(Run::build(args, &repo));
        })
        .expect("spawn the build thread");
    commands.insert_resource(Pending {
        title: launch.title.clone(),
        started: Instant::now(),
        result: Mutex::new(rx),
    });

    let scope = DespawnOnExit(Screen::Loading);
    commands.spawn((scope.clone(), Camera2d));
    commands
        .spawn((
            scope,
            Node {
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                flex_direction: FlexDirection::Column,
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                row_gap: Val::Px(10.0),
                ..default()
            },
        ))
        .with_children(|root| {
            root.spawn((
                Text::new(format!("Building {}", launch.title)),
                font(24.0),
                TextColor(menu::ACCENT),
            ));
            root.spawn((Elapsed, Text::new(""), font(15.0), TextColor(menu::DIM)));
            root.spawn((
                Text::new("Esc back to the list of runs"),
                font(12.0),
                TextColor(menu::FAINT),
            ));
        });
}

fn elapsed(pending: Option<Res<Pending>>, mut texts: Query<&mut Text, With<Elapsed>>) {
    let Some(pending) = pending else { return };
    let seconds = pending.started.elapsed().as_secs();
    for mut text in &mut texts {
        let now = format!("{seconds} s");
        if text.0 != now {
            text.0 = now;
        }
    }
}

/// Esc: back to the list; the build finishes unseen.
fn back(
    keys: Res<ButtonInput<KeyCode>>,
    pending: Option<Res<Pending>>,
    mut menu: ResMut<Menu>,
    mut next: ResMut<NextState<Screen>>,
) {
    if keys.just_pressed(KeyCode::Escape) {
        if let Some(pending) = pending {
            menu.say(format!("Stopped waiting for {}.", pending.title), false);
        }
        next.set(Screen::Menu);
    }
}

/// A built run is started and shown; a failure goes back to the list with its error.
fn arrived(world: &mut World) {
    let Some(pending) = world.get_resource::<Pending>() else {
        // Nothing launched: back to the list
        world.resource_mut::<NextState<Screen>>().set(Screen::Menu);
        return;
    };
    let title = pending.title.clone();
    let received = pending.result.lock().expect("build channel").try_recv();
    let failed = match received {
        Err(TryRecvError::Empty) => return,
        Err(TryRecvError::Disconnected) => "the build stopped (see the console)".to_string(),
        Ok(Err(failed)) => failed.message,
        Ok(Ok(run)) => match run.start(world) {
            Ok(()) => {
                println!(
                    "Built {title} in {:.1} s",
                    world.resource::<Pending>().started.elapsed().as_secs_f64()
                );
                world
                    .resource_mut::<NextState<Screen>>()
                    .set(Screen::Viewer);
                return;
            }
            Err(failed) => failed.message,
        },
    };
    eprintln!("{title}: {failed}");
    world
        .resource_mut::<Menu>()
        .say(format!("{title} could not start: {failed}"), true);
    world.resource_mut::<NextState<Screen>>().set(Screen::Menu);
}

//! The window's screens: the list of runs ([`crate::menu`]), a run being built, and
//! the viewer of a run.
//!
//! One window shows them in turn, as the states of [`Screen`]. A run's resources are
//! inserted as it is entered (`crate::Run::insert`), its plugins spawn their
//! entities `OnEnter(Screen::Viewer)` and mark them [`RUN`], and leaving the viewer
//! despawns them, removes every resource the run and its plugins inserted
//! ([`RunResources`]) and stops its solver. The viewer's per-frame systems run in
//! [`Viewing`], only while a run is shown.

use bevy::prelude::*;

#[derive(States, Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum Screen {
    /// The list of runs to start or replay
    #[default]
    Menu,
    /// A run's scenario, terrain and solver being built, off the main thread
    Loading,
    /// A run, shown
    Viewer,
}

/// The viewer's systems, in every schedule they use: they run while a run is shown.
#[derive(SystemSet, Clone, PartialEq, Eq, Hash, Debug)]
pub struct Viewing;

/// The mark of an entity of a run: despawned (with its children) on leaving it.
pub const RUN: DespawnOnExit<Screen> = DespawnOnExit(Screen::Viewer);

pub struct ScreenPlugin;

impl Plugin for ScreenPlugin {
    fn build(&self, app: &mut App) {
        let viewing = || Viewing.run_if(in_state(Screen::Viewer));
        app.configure_sets(PreUpdate, viewing())
            .configure_sets(Update, viewing())
            .configure_sets(PostUpdate, viewing())
            .configure_sets(Last, viewing());
    }
}

/// Resources that live as long as a run: removed on leaving it, so that the next run
/// neither sees them (one inserted only for some runs, such as the terrain) nor keeps
/// their assets alive in the menu.
pub trait RunResources {
    /// `R`, inserted while the run is entered, is removed on leaving it.
    fn run_resource<R: Resource>(&mut self) -> &mut Self;
    /// `R` starts each run at its default and is removed on leaving it.
    fn init_run_resource<R: Resource + Default>(&mut self) -> &mut Self;
}

impl RunResources for App {
    fn run_resource<R: Resource>(&mut self) -> &mut Self {
        self.add_systems(OnExit(Screen::Viewer), |mut commands: Commands| {
            commands.remove_resource::<R>();
        })
    }

    fn init_run_resource<R: Resource + Default>(&mut self) -> &mut Self {
        self.add_systems(OnEnter(Screen::Viewer), |mut commands: Commands| {
            commands.init_resource::<R>();
        })
        .run_resource::<R>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::state::app::StatesPlugin;

    #[derive(Resource, Default)]
    struct Per;

    #[derive(Resource)]
    struct Inserted;

    #[derive(Resource, Default)]
    struct Ticks(u32);

    #[derive(Component)]
    struct Thing;

    fn tick(mut ticks: ResMut<Ticks>) {
        ticks.0 += 1;
    }

    /// Entering a run makes its resources, leaving it removes them and its
    /// entities, and its systems run only meanwhile.
    #[test]
    fn a_run_leaves_nothing_behind() {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, StatesPlugin, ScreenPlugin))
            .init_state::<Screen>()
            .init_resource::<Ticks>()
            .init_run_resource::<Per>()
            .run_resource::<Inserted>()
            .add_systems(OnEnter(Screen::Viewer), |mut commands: Commands| {
                commands.insert_resource(Inserted);
                commands.spawn((Thing, RUN)).with_child(Thing);
            })
            .add_systems(Update, tick.in_set(Viewing));
        let things = |app: &mut App| {
            app.world_mut()
                .query_filtered::<(), With<Thing>>()
                .iter(app.world())
                .count()
        };

        app.update();
        assert_eq!(app.world().resource::<Ticks>().0, 0, "not in the menu");
        assert!(!app.world().contains_resource::<Per>());

        for run in 1..=2 {
            app.world_mut()
                .resource_mut::<NextState<Screen>>()
                .set(Screen::Viewer);
            app.update();
            app.update();
            assert!(app.world().contains_resource::<Per>());
            assert!(app.world().contains_resource::<Inserted>());
            assert_eq!(things(&mut app), 2);
            assert_eq!(app.world().resource::<Ticks>().0, 2 * run);

            app.world_mut()
                .resource_mut::<NextState<Screen>>()
                .set(Screen::Menu);
            app.update();
            assert!(!app.world().contains_resource::<Per>());
            assert!(!app.world().contains_resource::<Inserted>());
            assert_eq!(things(&mut app), 0, "the run's entities and children");
            assert_eq!(app.world().resource::<Ticks>().0, 2 * run);
        }
    }
}

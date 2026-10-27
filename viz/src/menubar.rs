//! The menu bar along the top of the window (Run, View, Colour, Overlays, Photo,
//! Help) and the help popup (F1, H or ?), both read from the table of actions
//! ([`crate::actions`]).
//!
//! A click on a menu's title opens it; a click on an entry acts as its key does and
//! closes the menu, except for the entries one steps through (the rate, the opacity,
//! the layer, the wind), which keep it open. Each entry shows its key and its state:
//! a box for a switch, a dot for the colouring shown, the value of a setting. Only
//! what the run has is listed (3D layers in a 3D run, the waves' colourings in a run
//! with waves, the wind in the photo view). Esc or a click elsewhere closes a menu.
//!
//! The popup lists the keys by the same groups, with each one's state, and the
//! mouse's controls. Over the bar, an open menu or the popup the mouse does not move
//! the view or pin a point ([`OverUi`]).

use bevy::prelude::*;
use bevy::text::{FontSize, LineBreak};
use bevy::ui::{RelativeCursorPosition, UiSystems};

use crate::actions::{Act, Action, Actions, Context, Menu, Shown, State};
use crate::camera::OverUi;
use crate::screen::{RUN, RunResources, Screen, Viewing};

/// The bar's height (logical px): the HUD's panels start below it.
pub const BAR_HEIGHT: f32 = 28.0;

const BAR: Color = Color::srgba(0.06, 0.075, 0.09, 0.92);
const PANEL: Color = Color::srgb(0.075, 0.095, 0.115);
const HOVERED: Color = Color::srgba(1.0, 1.0, 1.0, 0.08);
const OPEN: Color = Color::srgba(0.25, 0.6, 0.75, 0.35);
const ACCENT: Color = Color::srgb(0.45, 0.8, 0.9);
const DIM: Color = Color::srgba(1.0, 1.0, 1.0, 0.55);

/// The mouse's controls, in the popup's View group.
const MOUSE: [(&str, &str); 5] = [
    ("drag", "Turn about the focus"),
    ("right-drag", "Slide the focus (or Shift+drag)"),
    ("wheel", "Zoom"),
    ("hover", "Read out the point"),
    ("click", "Pin the point, its profiles"),
];

/// A menu's title on the bar.
#[derive(Component)]
struct Title(Menu);

/// A menu's list of entries, under its title.
#[derive(Component)]
struct Dropdown(Menu);

/// An entry of an open menu.
#[derive(Component)]
struct Item(Action);

/// The help popup, and the panel inside it that holds the groups.
#[derive(Component)]
struct Popup;

#[derive(Component)]
struct PopupBody;

/// Nodes over which the mouse belongs to the menus.
#[derive(Component)]
struct Blocks;

/// What the open menu and the popup were last filled with: they are rebuilt only when
/// that changes. A run's own, as its menus are.
#[derive(Resource, Default)]
struct Filled {
    menu: Option<(Menu, Vec<Row>)>,
    popup: Option<Vec<Group>>,
}

pub struct MenubarPlugin;

impl Plugin for MenubarPlugin {
    fn build(&self, app: &mut App) {
        app.init_run_resource::<OverUi>()
            .init_run_resource::<Filled>()
            .add_systems(OnEnter(Screen::Viewer), spawn)
            .add_systems(PreUpdate, over_ui.after(UiSystems::Focus).in_set(Viewing))
            .add_systems(
                Update,
                (help, clicks, show_menus, fill_menu, fill_popup, highlight)
                    .chain()
                    .in_set(Viewing),
            );
    }
}

fn font(size: f32) -> TextFont {
    TextFont {
        font_size: FontSize::Px(size),
        ..default()
    }
}

fn spawn(mut commands: Commands) {
    commands
        .spawn((
            RUN,
            Blocks,
            RelativeCursorPosition::default(),
            GlobalZIndex(10),
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(0.0),
                top: Val::Px(0.0),
                width: Val::Percent(100.0),
                height: Val::Px(BAR_HEIGHT),
                padding: UiRect::horizontal(Val::Px(6.0)),
                align_items: AlignItems::Center,
                ..default()
            },
            BackgroundColor(BAR),
        ))
        .with_children(|bar| {
            for menu in Menu::ALL {
                bar.spawn((
                    Title(menu),
                    RelativeCursorPosition::default(),
                    Node {
                        height: Val::Percent(100.0),
                        padding: UiRect::horizontal(Val::Px(11.0)),
                        align_items: AlignItems::Center,
                        ..default()
                    },
                    BackgroundColor(Color::NONE),
                ))
                .with_children(|title| {
                    title.spawn((Text::new(menu.title()), font(14.0)));
                    title.spawn((
                        Dropdown(menu),
                        Blocks,
                        RelativeCursorPosition::default(),
                        Visibility::Hidden,
                        Node {
                            position_type: PositionType::Absolute,
                            left: Val::Px(0.0),
                            top: Val::Px(BAR_HEIGHT),
                            min_width: Val::Px(420.0),
                            flex_direction: FlexDirection::Column,
                            padding: UiRect::vertical(Val::Px(4.0)),
                            ..default()
                        },
                        BackgroundColor(PANEL),
                    ));
                });
            }
            bar.spawn(Node {
                flex_grow: 1.0,
                ..default()
            });
            bar.spawn((
                Text::new("F1 keys   M back to the runs"),
                font(13.0),
                TextColor(DIM),
            ));
        });
    // The popup: a panel in the middle of the window
    commands
        .spawn((
            RUN,
            Popup,
            Visibility::Hidden,
            GlobalZIndex(20),
            Node {
                position_type: PositionType::Absolute,
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                ..default()
            },
        ))
        .with_children(|popup| {
            popup.spawn((
                PopupBody,
                Blocks,
                RelativeCursorPosition::default(),
                Node {
                    max_width: Val::Percent(92.0),
                    flex_direction: FlexDirection::Row,
                    flex_wrap: FlexWrap::Wrap,
                    column_gap: Val::Px(36.0),
                    row_gap: Val::Px(18.0),
                    padding: UiRect::all(Val::Px(20.0)),
                    ..default()
                },
                BackgroundColor(PANEL),
            ));
        });
}

/// Whether the mouse is over the bar, an open menu or the popup.
fn over_ui(
    nodes: Query<(&RelativeCursorPosition, &InheritedVisibility), With<Blocks>>,
    mut over: ResMut<OverUi>,
) {
    // Hidden nodes keep the cursor's last position: only shown ones count
    let now = nodes
        .iter()
        .any(|(cursor, visible)| visible.get() && cursor.cursor_over());
    if over.0 != now {
        over.0 = now;
    }
}

/// F1, H, ? and the Help menu's entry open and close the popup, and close a menu;
/// Esc closes both.
fn help(mut acts: MessageReader<Act>, mut shown: ResMut<Shown>) {
    for &Act(action) in acts.read() {
        match action {
            Action::Help => {
                shown.help = !shown.help;
                shown.menu = None;
            }
            Action::Dismiss => *shown = Shown::default(),
            _ => {}
        }
    }
}

/// Clicks on the titles open and close the menus, clicks on entries act; moving
/// over another title while a menu is open opens that one.
fn clicks(
    mouse: Res<ButtonInput<MouseButton>>,
    titles: Query<(&Title, &RelativeCursorPosition)>,
    items: Query<(&Item, &RelativeCursorPosition, &InheritedVisibility)>,
    actions: Res<Actions>,
    mut shown: ResMut<Shown>,
    mut acts: MessageWriter<Act>,
) {
    let title = titles
        .iter()
        .find(|(_, cursor)| cursor.cursor_over())
        .map(|(t, _)| t.0);
    if let (Some(open), Some(over)) = (shown.menu, title)
        && open != over
    {
        shown.menu = Some(over);
    }
    if !mouse.just_pressed(MouseButton::Left) {
        return;
    }
    if let Some(menu) = title {
        shown.menu = if shown.menu == Some(menu) {
            None
        } else {
            Some(menu)
        };
        return;
    }
    let item = items
        .iter()
        .find(|(_, cursor, visible)| visible.get() && cursor.cursor_over())
        .map(|(item, ..)| item.0);
    if let Some(action) = item {
        acts.write(Act(action));
        if !actions.entry(action).is_some_and(|e| e.repeats) {
            shown.menu = None;
        }
    } else if shown.menu.is_some() {
        shown.menu = None;
    }
}

/// The open menu's list and the popup shown, the others hidden.
fn show_menus(
    shown: Res<Shown>,
    mut dropdowns: Query<(&Dropdown, &mut Visibility), Without<Popup>>,
    mut popup: Query<&mut Visibility, With<Popup>>,
) {
    if !shown.is_changed() {
        return;
    }
    let visible = |on: bool| {
        if on {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        }
    };
    for (dropdown, mut visibility) in &mut dropdowns {
        visibility.set_if_neq(visible(shown.menu == Some(dropdown.0)));
    }
    for mut visibility in &mut popup {
        visibility.set_if_neq(visible(shown.help));
    }
}

/// An entry as a menu shows it: its action, mark, label, state and key.
#[derive(Clone, PartialEq)]
struct Row {
    action: Action,
    mark: &'static str,
    label: String,
    state: String,
    key: &'static str,
}

/// The entries of `menu` the run has now, with their states.
fn rows(actions: &Actions, context: &Context, menu: Menu) -> Vec<Row> {
    actions
        .0
        .iter()
        .filter(|e| e.menu == menu && context.applies(e.action))
        .map(|e| {
            let state = context.state(e.action);
            Row {
                action: e.action,
                mark: state.mark(),
                label: e.label.to_string(),
                state: match &state {
                    State::Value(v) => v.clone(),
                    _ => String::new(),
                },
                key: e.key.map_or("", |k| k.name),
            }
        })
        .collect()
}

/// The open menu's entries, rebuilt when what it lists or their states change.
fn fill_menu(
    mut commands: Commands,
    shown: Res<Shown>,
    actions: Res<Actions>,
    context: Context,
    dropdowns: Query<(Entity, &Dropdown)>,
    mut filled: ResMut<Filled>,
) {
    let Some(menu) = shown.menu else {
        filled.menu = None;
        return;
    };
    let rows = rows(&actions, &context, menu);
    if filled
        .menu
        .as_ref()
        .is_some_and(|(m, r)| *m == menu && *r == rows)
    {
        return;
    }
    let Some((list, _)) = dropdowns.iter().find(|(_, d)| d.0 == menu) else {
        return;
    };
    commands.entity(list).despawn_children();
    commands.entity(list).with_children(|list| {
        for row in &rows {
            list.spawn((
                Item(row.action),
                RelativeCursorPosition::default(),
                Node {
                    padding: UiRect::axes(Val::Px(10.0), Val::Px(4.0)),
                    column_gap: Val::Px(8.0),
                    align_items: AlignItems::Center,
                    ..default()
                },
                BackgroundColor(Color::NONE),
            ))
            .with_children(|item| {
                item.spawn((
                    Text::new(row.mark),
                    font(13.0),
                    TextColor(ACCENT),
                    Node {
                        width: Val::Px(14.0),
                        ..default()
                    },
                ));
                item.spawn((
                    Text::new(row.label.clone()),
                    font(14.0),
                    no_wrap(),
                    Node {
                        flex_grow: 1.0,
                        ..default()
                    },
                ));
                item.spawn((
                    Text::new(row.state.clone()),
                    no_wrap(),
                    font(13.0),
                    TextColor(DIM),
                    Node {
                        margin: UiRect::left(Val::Px(16.0)),
                        ..default()
                    },
                ));
                item.spawn((
                    Text::new(row.key),
                    font(13.0),
                    TextColor(ACCENT),
                    TextLayout {
                        justify: Justify::Right,
                        ..default()
                    },
                    Node {
                        min_width: Val::Px(64.0),
                        justify_content: JustifyContent::FlexEnd,
                        ..default()
                    },
                ));
            });
        }
    });
    filled.menu = Some((menu, rows));
}

/// A group of the popup: its heading and its lines (key, label, state).
type Group = (&'static str, Vec<(String, String, String)>);

/// The popup's groups: the keyed entries the run has now, by menu.
fn groups(actions: &Actions, context: &Context) -> Vec<Group> {
    Menu::ALL
        .iter()
        .map(|&menu| {
            let mut lines: Vec<_> = actions
                .0
                .iter()
                .filter(|e| e.menu == menu && context.applies(e.action))
                .filter_map(|e| {
                    let key = e.key?;
                    let state = context.state(e.action).words().into_owned();
                    Some((key.name.to_string(), e.label.to_string(), state))
                })
                .collect();
            if menu == Menu::View {
                lines.extend(
                    MOUSE
                        .iter()
                        .map(|(k, l)| (k.to_string(), l.to_string(), String::new())),
                );
            }
            let heading = match menu {
                Menu::Run => "Run and time",
                Menu::View => "View",
                Menu::Colour => "Colour and water (C cycles the colourings)",
                Menu::Overlays => "Overlays",
                Menu::Photo => "Photo view",
                Menu::Help => "Help (Esc closes)",
            };
            (heading, lines)
        })
        .filter(|(_, lines)| !lines.is_empty())
        .collect()
}

/// The popup's groups, rebuilt while it is shown when they change.
fn fill_popup(
    mut commands: Commands,
    shown: Res<Shown>,
    actions: Res<Actions>,
    context: Context,
    body: Query<Entity, With<PopupBody>>,
    mut filled: ResMut<Filled>,
) {
    if !shown.help {
        return;
    }
    let groups = groups(&actions, &context);
    if filled.popup.as_ref() == Some(&groups) {
        return;
    }
    let Ok(body) = body.single() else { return };
    commands.entity(body).despawn_children();
    commands.entity(body).with_children(|body| {
        for (heading, lines) in &groups {
            body.spawn(Node {
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(3.0),
                min_width: Val::Px(330.0),
                ..default()
            })
            .with_children(|group| {
                group.spawn((
                    Text::new(*heading),
                    font(15.0),
                    TextColor(ACCENT),
                    Node {
                        margin: UiRect::bottom(Val::Px(4.0)),
                        ..default()
                    },
                ));
                for (key, label, state) in lines {
                    group
                        .spawn(Node {
                            column_gap: Val::Px(10.0),
                            ..default()
                        })
                        .with_children(|line| {
                            line.spawn((
                                Text::new(key.clone()),
                                font(13.0),
                                TextColor(ACCENT),
                                Node {
                                    min_width: Val::Px(84.0),
                                    ..default()
                                },
                            ));
                            line.spawn((
                                Text::new(label.clone()),
                                font(13.0),
                                Node {
                                    flex_grow: 1.0,
                                    ..default()
                                },
                            ));
                            line.spawn((Text::new(state.clone()), font(13.0), TextColor(DIM)));
                        });
                }
            });
        }
    });
    filled.popup = Some(groups);
}

/// The open menu's title and the entry under the mouse lit.
fn highlight(
    shown: Res<Shown>,
    mut titles: Query<(&Title, &RelativeCursorPosition, &mut BackgroundColor), Without<Item>>,
    mut items: Query<(&RelativeCursorPosition, &mut BackgroundColor), With<Item>>,
) {
    for (title, cursor, mut background) in &mut titles {
        let colour = if shown.menu == Some(title.0) {
            OPEN
        } else if cursor.cursor_over() {
            HOVERED
        } else {
            Color::NONE
        };
        background.set_if_neq(BackgroundColor(colour));
    }
    for (cursor, mut background) in &mut items {
        let colour = if cursor.cursor_over() {
            HOVERED
        } else {
            Color::NONE
        };
        background.set_if_neq(BackgroundColor(colour));
    }
}

/// One line, however long: wrapping text makes the menu taller than its rows.
fn no_wrap() -> TextLayout {
    TextLayout {
        linebreak: LineBreak::NoWrap,
        ..default()
    }
}

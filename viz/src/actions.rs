//! The viewer's actions, in one table: each with its key, its label, the menu and
//! group it belongs to, when the run has it and its current state. The keys
//! ([`keys`]), the help popup and the menu bar ([`crate::menubar`]) all read the
//! table and send the same [`Act`] messages, and the plugins act on those, so a
//! key, its entry in the menu and its line in the help cannot drift apart.
//!
//! The keys are sent in `PreUpdate`, so the plugins' `Update` systems see them in the
//! same frame. A key does nothing where its action does not apply (W outside the
//! photo view, `,` and `.` in a 2D run).

use std::borrow::Cow;

use bevy::ecs::system::SystemParam;
use bevy::input::InputSystems;
use bevy::input::keyboard::Key as LogicalKey;
use bevy::prelude::*;

use crate::arrows::Arrows;
use crate::camera::Views;
use crate::contours::Contours;
use crate::field::{Field, ShownLayer};
use crate::inspect::Inspector;
use crate::layers::Levels;
use crate::minimap::Minimap;
use crate::particles::Particles;
use crate::photo::Photo;
use crate::playback::Playback;
use crate::screen::{RunResources, Viewing};
use crate::stratification::{SheetShows, Stratification};
use crate::surface::{ColourBy, Colouring, SurfaceStyle, WaterOpacity};
use crate::terrain::Terrain;
use crate::trace::Trace;

/// Something the viewer does on a key, a menu entry or both.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Action {
    Pause,
    Slower,
    Faster,
    SeekBack,
    SeekForward,
    First,
    Last,
    CloseUp,
    Overview,
    /// The named place `i` (from 0; key `i + 1`)
    Place(usize),
    Minimap,
    Unpin,
    NextColour,
    Colour(ColourBy),
    WaterStyle,
    Clearer,
    Denser,
    LayerUp,
    LayerDown,
    Arrows,
    Contours,
    Land,
    Particles,
    Sheet,
    Trace,
    Photo,
    Wind,
    WindFrom,
    Help,
    Menu,
    /// Build the run anew from its options
    Restart,
    Quit,
    /// Close the help and the open menu (Esc while one is shown; not in the table)
    Dismiss,
}

/// An action to take this frame, from a key or a menu entry.
#[derive(Message, Clone, Copy, Debug, PartialEq, Eq)]
pub struct Act(pub Action);

/// The menus of the menu bar, in its order; the help popup groups its keys the same way.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Menu {
    Run,
    View,
    Colour,
    Overlays,
    Photo,
    Help,
}

impl Menu {
    pub const ALL: [Self; 6] = [
        Self::Run,
        Self::View,
        Self::Colour,
        Self::Overlays,
        Self::Photo,
        Self::Help,
    ];

    pub fn title(self) -> &'static str {
        match self {
            Self::Run => "Run",
            Self::View => "View",
            Self::Colour => "Colour",
            Self::Overlays => "Overlays",
            Self::Photo => "Photo",
            Self::Help => "Help",
        }
    }
}

/// A key: any of `codes`, with Shift held or not.
#[derive(Clone, Copy, Debug)]
pub struct Key {
    pub codes: &'static [KeyCode],
    pub shift: bool,
    /// As the help and the menus show it
    pub name: &'static str,
}

const fn key(code: &'static [KeyCode], name: &'static str) -> Option<Key> {
    Some(Key {
        codes: code,
        shift: false,
        name,
    })
}

/// One action's entry.
#[derive(Clone, Debug)]
pub struct Entry {
    pub action: Action,
    pub menu: Menu,
    pub key: Option<Key>,
    pub label: Cow<'static, str>,
    /// Whether choosing it from a menu leaves the menu open, to choose it again
    /// (the steps of the rate, the opacity, the layer, the wind)
    pub repeats: bool,
}

/// What an entry shows beside its label.
#[derive(Clone, Debug, PartialEq)]
pub enum State {
    None,
    /// A switch, on or off
    On(bool),
    /// One of several choices, chosen or not
    Chosen(bool),
    /// A value, such as the rate or the layer
    Value(String),
}

impl State {
    /// The mark before the label in a menu: a box for a switch, a dot for a choice.
    pub fn mark(&self) -> &'static str {
        match self {
            Self::On(true) => "■",
            Self::On(false) => "□",
            Self::Chosen(true) => "●",
            Self::Chosen(false) => "○",
            Self::None | Self::Value(_) => " ",
        }
    }

    /// The state in words, for the help popup.
    pub fn words(&self) -> Cow<'static, str> {
        match self {
            Self::None => "".into(),
            Self::On(true) => "on".into(),
            Self::On(false) => "off".into(),
            Self::Chosen(true) => "shown".into(),
            Self::Chosen(false) => "".into(),
            Self::Value(v) => v.clone().into(),
        }
    }
}

/// The number keys, 1 to 9, for the named places, and their names.
const DIGITS: [KeyCode; 9] = [
    KeyCode::Digit1,
    KeyCode::Digit2,
    KeyCode::Digit3,
    KeyCode::Digit4,
    KeyCode::Digit5,
    KeyCode::Digit6,
    KeyCode::Digit7,
    KeyCode::Digit8,
    KeyCode::Digit9,
];
const DIGIT_NAMES: [&str; 9] = ["1", "2", "3", "4", "5", "6", "7", "8", "9"];

/// The colourings, in the order C steps through them.
pub const COLOURINGS: [ColourBy; 5] = [
    ColourBy::Speed,
    ColourBy::Elevation,
    ColourBy::WaveHeight,
    ColourBy::WavePeriod,
    ColourBy::WaveDissipation,
];

/// The viewer's actions (see the module docs), with the run's named places.
#[derive(Resource)]
pub struct Actions(pub Vec<Entry>);

impl Actions {
    /// The table, with a place entry per named place (at most nine: the digits).
    pub fn new(views: &Views) -> Self {
        use Action as A;
        use KeyCode as K;
        let entry = |action, menu, key, label: &'static str, repeats| Entry {
            action,
            menu,
            key,
            label: label.into(),
            repeats,
        };
        let mut table = vec![
            entry(
                A::Menu,
                Menu::Run,
                key(&[K::KeyM], "M"),
                "Back to the list of runs",
                false,
            ),
            entry(
                A::Pause,
                Menu::Run,
                key(&[K::Space], "Space"),
                "Pause",
                false,
            ),
            entry(
                A::Slower,
                Menu::Run,
                key(&[K::BracketLeft], "["),
                "Slower (half the rate)",
                true,
            ),
            entry(
                A::Faster,
                Menu::Run,
                key(&[K::BracketRight], "]"),
                "Faster (twice the rate)",
                true,
            ),
            entry(
                A::SeekBack,
                Menu::Run,
                key(&[K::ArrowLeft], "←"),
                "Back 10 frames",
                true,
            ),
            entry(
                A::SeekForward,
                Menu::Run,
                key(&[K::ArrowRight], "→"),
                "Forward 10 frames",
                true,
            ),
            entry(
                A::First,
                Menu::Run,
                key(&[K::Home], "Home"),
                "To the first frame held",
                false,
            ),
            entry(
                A::Last,
                Menu::Run,
                key(&[K::End], "End"),
                "To the newest frame",
                false,
            ),
            entry(
                A::Restart,
                Menu::Run,
                None,
                "Restart (build the run anew)",
                false,
            ),
            entry(A::Quit, Menu::Run, None, "Quit", false),
            entry(
                A::CloseUp,
                Menu::View,
                key(&[K::KeyF], "F"),
                "Close-up",
                false,
            ),
            entry(
                A::Overview,
                Menu::View,
                key(&[K::KeyO], "O"),
                "Overview",
                false,
            ),
        ];
        for (i, (name, _)) in views.places.iter().take(DIGITS.len()).enumerate() {
            table.push(Entry {
                action: A::Place(i),
                menu: Menu::View,
                key: Some(Key {
                    codes: std::slice::from_ref(&DIGITS[i]),
                    shift: false,
                    name: DIGIT_NAMES[i],
                }),
                label: name.clone().into(),
                repeats: false,
            });
        }
        table.extend([
            entry(
                A::Minimap,
                Menu::View,
                key(&[K::KeyV], "V"),
                "Map of the domain",
                false,
            ),
            entry(
                A::Unpin,
                Menu::View,
                key(&[K::Escape], "Esc"),
                "Let go of the pinned point",
                false,
            ),
            entry(
                A::NextColour,
                Menu::Colour,
                key(&[K::KeyC], "C"),
                "Next colouring",
                true,
            ),
        ]);
        for by in COLOURINGS {
            table.push(Entry {
                action: A::Colour(by),
                menu: Menu::Colour,
                key: None,
                label: colouring_name(by).into(),
                repeats: false,
            });
        }
        table.extend([
            entry(
                A::WaterStyle,
                Menu::Colour,
                key(&[K::KeyT], "T"),
                "Water: translucent, opaque, hidden",
                true,
            ),
            entry(
                A::Clearer,
                Menu::Colour,
                key(&[K::Minus, K::NumpadSubtract], "-"),
                "Clearer water",
                true,
            ),
            entry(
                A::Denser,
                Menu::Colour,
                key(&[K::Equal, K::NumpadAdd], "="),
                "Denser water",
                true,
            ),
            entry(
                A::LayerUp,
                Menu::Colour,
                key(&[K::Comma], ","),
                "Layer up (to the depth mean)",
                true,
            ),
            entry(
                A::LayerDown,
                Menu::Colour,
                key(&[K::Period], "."),
                "Layer down (to the bed)",
                true,
            ),
            entry(
                A::Arrows,
                Menu::Overlays,
                key(&[K::KeyA], "A"),
                "Arrows (the waves' under a wave colouring)",
                false,
            ),
            entry(
                A::Contours,
                Menu::Overlays,
                key(&[K::KeyB], "B"),
                "Bed contours",
                false,
            ),
            entry(
                A::Land,
                Menu::Overlays,
                key(&[K::KeyL], "L"),
                "Land and the sea around",
                false,
            ),
            entry(
                A::Particles,
                Menu::Overlays,
                key(&[K::KeyP], "P"),
                "Particles",
                false,
            ),
            entry(
                A::Sheet,
                Menu::Overlays,
                key(&[K::KeyN], "N"),
                "Stratification sheet",
                true,
            ),
            entry(
                A::Trace,
                Menu::Overlays,
                key(&[K::KeyG], "G"),
                "Gauge trace",
                false,
            ),
            entry(
                A::Photo,
                Menu::Photo,
                key(&[K::KeyR], "R"),
                "Photo view",
                false,
            ),
            entry(
                A::Wind,
                Menu::Photo,
                key(&[K::KeyW], "W"),
                "Wind speed",
                true,
            ),
            Entry {
                key: Some(Key {
                    codes: &[K::KeyW],
                    shift: true,
                    name: "Shift+W",
                }),
                ..entry(A::WindFrom, Menu::Photo, None, "Wind direction", true)
            },
            entry(
                A::Help,
                Menu::Help,
                key(&[K::F1, K::KeyH], "F1 / H"),
                "Keys and what they show",
                false,
            ),
        ]);
        Self(table)
    }

    pub fn entry(&self, action: Action) -> Option<&Entry> {
        self.0.iter().find(|e| e.action == action)
    }

    /// The name of `action`'s key, for the status lines' hints ("" if it has none).
    pub fn key_name(&self, action: Action) -> &'static str {
        self.entry(action)
            .and_then(|e| e.key)
            .map_or("", |k| k.name)
    }
}

/// A colouring's name in the menus.
pub fn colouring_name(by: ColourBy) -> &'static str {
    match by {
        ColourBy::Speed => "Current speed",
        ColourBy::Elevation => "Surface elevation η",
        ColourBy::WaveHeight => "Wave height H_s",
        ColourBy::WavePeriod => "Peak period T_p",
        ColourBy::WaveDissipation => "Wave energy lost (whitecapping, breaking)",
    }
}

/// The help popup and the open menu: Esc closes them before it lets go of a pin.
#[derive(Resource, Default, Debug, PartialEq, Eq)]
pub struct Shown {
    pub help: bool,
    pub menu: Option<Menu>,
}

impl Shown {
    /// `--show`: `keys` for the help, else a menu by its title.
    pub fn parse(name: &str) -> Option<Self> {
        if name == "keys" {
            return Some(Self {
                help: true,
                menu: None,
            });
        }
        let menu = Menu::ALL
            .into_iter()
            .find(|m| m.title().eq_ignore_ascii_case(name))?;
        Some(Self {
            help: false,
            menu: Some(menu),
        })
    }
}

/// What the entries' states and whether they apply are read from.
#[derive(SystemParam)]
pub struct Context<'w> {
    playback: Res<'w, Playback>,
    colouring: Res<'w, Colouring>,
    style: Res<'w, SurfaceStyle>,
    opacity: Res<'w, WaterOpacity>,
    field: Res<'w, Field>,
    shown_layer: Res<'w, ShownLayer>,
    levels: Option<Res<'w, Levels>>,
    sheet: Option<Res<'w, Stratification>>,
    arrows: Option<Res<'w, Arrows>>,
    contours: Option<Res<'w, Contours>>,
    terrain: Option<Res<'w, Terrain>>,
    particles: Option<Res<'w, Particles>>,
    trace: Option<Res<'w, Trace>>,
    minimap: Option<Res<'w, Minimap>>,
    photo: Res<'w, Photo>,
    inspector: Option<Res<'w, Inspector>>,
    shown: Res<'w, Shown>,
}

impl Context<'_> {
    /// Whether the run has `action` now.
    pub fn applies(&self, action: Action) -> bool {
        match action {
            Action::Colour(by) => !by.is_waves() || self.field.waves.is_some(),
            Action::LayerUp | Action::LayerDown => self.levels.is_some(),
            Action::Sheet => self.sheet.is_some(),
            Action::Contours => self.contours.is_some(),
            Action::Land => self.terrain.is_some(),
            Action::Particles => self
                .particles
                .as_ref()
                .is_some_and(|p| p.counts.iter().any(|&n| n > 0)),
            Action::Minimap => self.minimap.is_some() && !self.photo.on,
            Action::Unpin => self.inspector.as_ref().is_some_and(|i| i.is_pinned()),
            Action::Wind | Action::WindFrom => self.photo.on,
            _ => true,
        }
    }

    pub fn state(&self, action: Action) -> State {
        let on = State::On;
        match action {
            Action::Pause => on(self.playback.paused),
            Action::Slower | Action::Faster => State::Value(format!("×{:.0}", self.playback.rate)),
            Action::Minimap => on(self.minimap.as_ref().is_some_and(|m| m.shown)),
            Action::NextColour => State::Value(colouring_name(self.colouring.by).into()),
            Action::Colour(by) => State::Chosen(self.colouring.by == by),
            Action::WaterStyle => State::Value(
                match *self.style {
                    SurfaceStyle::Translucent => "translucent",
                    SurfaceStyle::Opaque => "opaque",
                    SurfaceStyle::Hidden => "hidden",
                }
                .into(),
            ),
            Action::Clearer | Action::Denser => {
                State::Value(format!("{:.0} % opaque", 100.0 * self.opacity.0))
            }
            Action::LayerUp | Action::LayerDown => match (*self.shown_layer, &self.levels) {
                (ShownLayer::DepthMean, _) => State::Value("depth mean".into()),
                (ShownLayer::Level(l), Some(levels)) => {
                    let n = levels.sigma.len();
                    State::Value(format!("layer {} of {n} from the surface", n - l))
                }
                (ShownLayer::Level(l), None) => State::Value(format!("layer {l}")),
            },
            Action::Arrows => on(self.arrows.as_ref().is_some_and(|a| a.on)),
            Action::Contours => on(self.contours.as_ref().is_some_and(|c| c.on)),
            Action::Land => on(self.terrain.as_ref().is_some_and(|t| t.on)),
            Action::Particles => on(self.particles.as_ref().is_some_and(|p| p.on)),
            Action::Sheet => match &self.sheet {
                Some(sheet) if sheet.shows == SheetShows::Off => State::Value("off".into()),
                Some(sheet) => State::Value(sheet.shows.describe().0.into()),
                None => State::None,
            },
            Action::Trace => on(self.trace.as_ref().is_some_and(|t| t.on)),
            Action::Photo => on(self.photo.on),
            Action::Wind => State::Value(match self.photo.wind_speed {
                0.0 => "calm".into(),
                speed => format!("{speed:.0} m/s"),
            }),
            Action::WindFrom => State::Value(format!("from {}", self.photo.wind_name())),
            Action::Help => on(self.shown.help),
            _ => State::None,
        }
    }
}

pub struct ActionsPlugin;

impl Plugin for ActionsPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<Act>()
            .run_resource::<Shown>()
            .run_resource::<Actions>()
            .add_systems(PreUpdate, keys.after(InputSystems).in_set(Viewing));
    }
}

/// The keys pressed this frame as actions, those the run has now. Esc closes the
/// help or a menu first; `?` opens the help as F1 does.
fn keys(
    codes: Res<ButtonInput<KeyCode>>,
    logical: Res<ButtonInput<LogicalKey>>,
    actions: Res<Actions>,
    context: Context,
    mut acts: MessageWriter<Act>,
) {
    if codes.just_pressed(KeyCode::Escape) && (context.shown.help || context.shown.menu.is_some()) {
        acts.write(Act(Action::Dismiss));
        return;
    }
    // `?` is a shifted key on most layouts (Shift and the + key, the - key's place,
    // on a Norwegian one): it opens the help and nothing else
    if logical.just_pressed(LogicalKey::Character("?".into())) {
        acts.write(Act(Action::Help));
        return;
    }
    let shift = codes.any_pressed([KeyCode::ShiftLeft, KeyCode::ShiftRight]);
    for code in codes.get_just_pressed() {
        if let Some(action) = action_of(&actions, *code, shift)
            && context.applies(action)
        {
            acts.write(Act(action));
        }
    }
}

/// The action of `code`, Shift held or not: a Shift entry of the key if Shift is held
/// and it has one, else its plain entry (which Shift does not change).
fn action_of(actions: &Actions, code: KeyCode, shift: bool) -> Option<Action> {
    let mut plain = None;
    for entry in &actions.0 {
        let Some(key) = entry.key.filter(|k| k.codes.contains(&code)) else {
            continue;
        };
        match (key.shift, shift) {
            (true, true) => return Some(entry.action),
            (false, _) => plain = plain.or(Some(entry.action)),
            (true, false) => {}
        }
    }
    plain
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::OrbitCamera;

    fn views(places: usize) -> Views {
        let view = OrbitCamera {
            focus: Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.5,
            distance: 100.0,
        };
        Views {
            farm: view,
            domain: view,
            places: (0..places).map(|i| (format!("place {i}"), view)).collect(),
        }
    }

    #[test]
    fn every_key_names_one_action() {
        let actions = Actions::new(&views(12));
        let mut seen = std::collections::HashMap::new();
        for entry in &actions.0 {
            let Some(key) = entry.key else { continue };
            for code in key.codes {
                if let Some(other) = seen.insert((*code, key.shift), entry.action) {
                    panic!("{code:?} is both {other:?} and {:?}", entry.action);
                }
            }
        }
        // Nine places at most, the digits
        let places = actions
            .0
            .iter()
            .filter(|e| matches!(e.action, Action::Place(_)))
            .count();
        assert_eq!(places, 9);
    }

    #[test]
    fn shift_picks_the_shifted_entry_and_leaves_the_others() {
        let actions = Actions::new(&views(2));
        assert_eq!(
            action_of(&actions, KeyCode::KeyW, false),
            Some(Action::Wind)
        );
        assert_eq!(
            action_of(&actions, KeyCode::KeyW, true),
            Some(Action::WindFrom)
        );
        // Shift does not change a key without a Shift entry
        assert_eq!(
            action_of(&actions, KeyCode::KeyC, true),
            Some(Action::NextColour)
        );
        assert_eq!(
            action_of(&actions, KeyCode::Digit2, false),
            Some(Action::Place(1))
        );
        assert_eq!(action_of(&actions, KeyCode::Digit3, false), None);
        assert_eq!(
            action_of(&actions, KeyCode::NumpadAdd, false),
            Some(Action::Denser)
        );
    }

    #[test]
    fn every_menu_has_entries() {
        let actions = Actions::new(&views(0));
        for menu in Menu::ALL {
            assert!(
                actions.0.iter().any(|e| e.menu == menu),
                "{} is empty",
                menu.title()
            );
        }
    }

    /// The actions `keys` sends for the keys `pressed`, with `photo` on or off and
    /// the help shown or not.
    fn sent(pressed: &[KeyCode], photo: bool, help: bool) -> Vec<Action> {
        use bevy::ecs::system::RunSystemOnce;
        let mut world = World::new();
        let mut codes = ButtonInput::<KeyCode>::default();
        for &code in pressed {
            codes.press(code);
        }
        world.insert_resource(codes);
        world.insert_resource(ButtonInput::<LogicalKey>::default());
        world.insert_resource(Actions::new(&views(1)));
        world.insert_resource(Playback::new(60.0, 60.0, 1 << 20));
        world.insert_resource(Colouring::new(None, ColourBy::Speed));
        world.insert_resource(SurfaceStyle::Translucent);
        world.insert_resource(WaterOpacity::default());
        world.insert_resource(Field::default());
        world.insert_resource(ShownLayer::default());
        world.insert_resource(Photo {
            on: photo,
            wind_speed: 6.0,
            wind_from: 225.0,
            fetch: 10e3,
            place: None,
        });
        world.insert_resource(Shown { help, menu: None });
        world.init_resource::<Messages<Act>>();
        world.run_system_once(keys).unwrap();
        let messages = world.resource::<Messages<Act>>();
        messages.get_cursor().read(messages).map(|a| a.0).collect()
    }

    #[test]
    fn a_key_acts_only_where_the_run_has_its_action() {
        assert_eq!(sent(&[KeyCode::Space], false, false), [Action::Pause]);
        // W is the photo view's wind; `,` a 3D run's layer; Esc lets go of a pin
        assert!(sent(&[KeyCode::KeyW], false, false).is_empty());
        assert_eq!(sent(&[KeyCode::KeyW], true, false), [Action::Wind]);
        assert_eq!(
            sent(&[KeyCode::ShiftLeft, KeyCode::KeyW], true, false),
            [Action::WindFrom]
        );
        assert!(sent(&[KeyCode::Comma], false, false).is_empty());
        assert!(sent(&[KeyCode::Escape], false, false).is_empty());
        // Esc closes the help first, and does nothing else
        assert_eq!(sent(&[KeyCode::Escape], false, true), [Action::Dismiss]);
        assert_eq!(sent(&[KeyCode::F1], false, false), [Action::Help]);
    }

    #[test]
    fn show_opens_the_help_or_a_menu_by_its_title() {
        assert!(Shown::parse("keys").unwrap().help);
        assert_eq!(Shown::parse("Overlays").unwrap().menu, Some(Menu::Overlays));
        assert_eq!(Shown::parse("colour").unwrap().menu, Some(Menu::Colour));
        assert!(Shown::parse("nonsense").is_none());
    }
}

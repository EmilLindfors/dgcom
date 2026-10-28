//! The start menu: pick a model run or a saved run, and the viewer shows it.
//!
//! `dg-viz` without arguments (or with `--menu`) opens on this screen instead of a
//! scenario. It lists the scenarios the viewer can run live (the fjord farm, Frøya,
//! the 3D farm channel, the fjord farm in 3D) and the saved runs found under
//! `../output` (snapshot files, `*.dgsnap`, and folders of VTU frames,
//! `froya_NNNN.vtu`), newest first, with the
//! common options: model hours, the starting view, the photo view, the Mausund
//! gauge in the trace and saving the run as a snapshot file. A file or folder
//! dropped on the window is replayed directly, here or in the viewer.
//!
//! Starting the selection hands its options ([`Menu::command`], also printed as a
//! `cargo run` line to repeat it) to [`crate::loading`], which builds the run in the
//! same window; M in the viewer stops the run and comes back here, and a run that
//! cannot start comes back with its error ([`Menu::say`]).

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use bevy::input::mouse::{MouseScrollUnit, MouseWheel};
use bevy::prelude::*;
use bevy::render::view::screenshot::{Screenshot, save_to_disk};
use bevy::text::FontSize;
use bevy::ui::RelativeCursorPosition;
use bevy::window::{FileDragAndDrop, PrimaryWindow};
use bevy::winit::{UpdateMode, WinitSettings};
use dg_rs::io::SnapshotReader;
use dg_rs::time::ModelClock;

use crate::loading::Launch;
use crate::screen::Screen;

/// Height (logical pixels) of the window outside the saved runs' rows, and of a
/// row: the rows that fit are shown, and the list scrolls.
const FIXED_HEIGHT: f32 = 480.0;
const ROW_HEIGHT: f32 = 29.0;

/// Model hours a live run can be given.
const HOURS: [f64; 6] = [3.0, 6.0, 12.0, 25.0, 48.0, 72.0];

/// The tide gauge drawn in the trace of Frøya runs.
const GAUGE: &str = "data/tide_gauges/mausund_obs.txt";

const BACKGROUND: Color = Color::srgb(0.05, 0.065, 0.08);
const PANEL: Color = Color::srgb(0.075, 0.095, 0.115);
const SELECTED: Color = Color::srgba(0.25, 0.6, 0.75, 0.35);
const HOVERED: Color = Color::srgba(1.0, 1.0, 1.0, 0.06);
pub const ACCENT: Color = Color::srgb(0.45, 0.8, 0.9);
pub const DIM: Color = Color::srgba(1.0, 1.0, 1.0, 0.55);
pub const FAINT: Color = Color::srgba(1.0, 1.0, 1.0, 0.3);
const ERROR: Color = Color::srgb(1.0, 0.5, 0.45);

pub struct MenuPlugin;

impl Plugin for MenuPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(OnEnter(Screen::Menu), enter)
            .add_systems(
                Update,
                (keys, mouse, fit, draw, highlight)
                    .chain()
                    .run_if(in_state(Screen::Menu)),
            )
            .add_systems(Update, dropped.run_if(not(in_state(Screen::Loading))))
            // The list and a run being built wake four times a second without
            // input; a run is drawn every frame
            .add_systems(OnEnter(Screen::Loading), react)
            .add_systems(
                OnEnter(Screen::Viewer),
                |mut winit: ResMut<WinitSettings>| *winit = WinitSettings::default(),
            );
    }
}

fn react(mut winit: ResMut<WinitSettings>) {
    *winit = WinitSettings {
        focused_mode: UpdateMode::reactive(Duration::from_millis(250)),
        unfocused_mode: UpdateMode::reactive_low_power(Duration::from_millis(250)),
    };
}

/// The list again, looked for anew (a run may have saved one).
fn enter(
    mut commands: Commands,
    mut menu: ResMut<Menu>,
    winit: ResMut<WinitSettings>,
    mut clear: ResMut<ClearColor>,
) {
    commands.spawn((DespawnOnExit(Screen::Menu), Camera2d));
    clear.0 = BACKGROUND;
    react(winit);
    menu.rescan();
}

/// `--menu --screenshot PATH`: save the menu once drawn, and end (a headless check).
pub fn screenshot(app: &mut App, path: PathBuf) {
    app.add_systems(
        Update,
        move |mut frames: Local<u32>,
              mut winit: ResMut<WinitSettings>,
              mut commands: Commands,
              mut exit: MessageWriter<AppExit>| {
            *frames += 1;
            if *frames == 1 {
                *winit = WinitSettings::game();
            } else if *frames == 10 {
                commands
                    .spawn(Screenshot::primary_window())
                    .observe(save_to_disk(path.clone()));
            } else if *frames == 40 {
                exit.write(AppExit::Success);
            }
        },
    );
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    /// A scenario the viewer runs (`--scenario`)
    Live(&'static str),
    /// A snapshot file (`--replay FILE`)
    Snapshot,
    /// A folder of VTU frames (`--replay DIR`)
    Frames,
}

#[derive(Clone)]
struct Entry {
    kind: Kind,
    title: String,
    about: String,
    path: Option<PathBuf>,
    modified: Option<SystemTime>,
    /// What a closer look found (a snapshot's header), once selected
    details: Option<String>,
    /// Why it cannot run, if it cannot
    missing: Option<String>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Choice {
    Hours,
    View,
    Photo,
    Gauge,
    Save,
}

impl Choice {
    const ALL: [Choice; 5] = [
        Choice::Hours,
        Choice::View,
        Choice::Photo,
        Choice::Gauge,
        Choice::Save,
    ];

    fn key(self) -> (KeyCode, &'static str) {
        match self {
            Choice::Hours => (KeyCode::KeyH, "H"),
            Choice::View => (KeyCode::KeyV, "V"),
            Choice::Photo => (KeyCode::KeyP, "P"),
            Choice::Gauge => (KeyCode::KeyG, "G"),
            Choice::Save => (KeyCode::KeyS, "S"),
        }
    }
}

struct Options {
    /// Index into [`HOURS`]
    hours: usize,
    overview: bool,
    photo: bool,
    gauge: bool,
    save: bool,
}

/// What a row of the window does when clicked.
#[derive(Component, Clone, Copy, PartialEq, Eq, Debug)]
enum Target {
    Entry(usize),
    Choice(Choice),
    Launch,
}

#[derive(Component)]
struct MenuRoot;

#[derive(Resource)]
pub struct Menu {
    repo: PathBuf,
    /// The live scenarios, then the saved runs, newest first
    entries: Vec<Entry>,
    n_live: usize,
    selected: usize,
    /// First saved run shown, and how many fit
    scroll: usize,
    visible: usize,
    options: Options,
    /// Lines under the lists, and whether each is an error
    status: Vec<(String, bool)>,
    dirty: bool,
    last_click: Option<(usize, Instant)>,
}

impl Menu {
    pub fn new(repo: PathBuf) -> Self {
        let has = |p: &str| repo.join(p).is_file();
        let froya_missing = [
            "data/froya_coast.msh",
            "data/froya_topobathy.tif",
            "data/froya_boundary_tides.txt",
        ]
        .into_iter()
        .find(|p| !has(p))
        .map(|p| format!("needs {p}"));
        let live = |scenario, title: &str, about: &str, missing: Option<String>| Entry {
            kind: Kind::Live(scenario),
            title: title.into(),
            about: about.into(),
            path: None,
            modified: None,
            details: None,
            missing,
        };
        let entries = vec![
            live(
                "fjord",
                "Fjord farm",
                "2D tide through a fish farm in a fjord: the cages drag on the flow, \
                 particles leave them",
                (!has("tests/data/gmsh/fjord_farm.msh"))
                    .then(|| "needs tests/data/gmsh/fjord_farm.msh".into()),
            ),
            live(
                "froya",
                "Frøya–Smøla–Hitra",
                "2D tide on the coastline mesh, NorKyst-800 boundary tides from \
                 2025-06-15; the bed takes a while to build",
                froya_missing,
            ),
            live(
                "channel",
                "Farm channel in 3D",
                "A stratified channel with σ-layers around two cages, lice larvae, \
                 faeces and feed",
                None,
            ),
            live(
                "fjord3d",
                "Fjord farm in 3D",
                "The fjord's tide over a brackish layer and a thermocline: σ-layers, \
                 larvae diving out of the fresh water",
                (!has("tests/data/gmsh/fjord_farm_3d.msh"))
                    .then(|| "needs tests/data/gmsh/fjord_farm_3d.msh".into()),
            ),
        ];
        Self {
            n_live: entries.len(),
            entries,
            repo,
            selected: 0,
            scroll: 0,
            visible: 10,
            options: Options {
                hours: 3,
                overview: false,
                photo: false,
                gauge: true,
                save: false,
            },
            status: Vec::new(),
            dirty: true,
            last_click: None,
        }
    }

    /// Look again for saved runs under `../output`, keeping the selection.
    fn rescan(&mut self) {
        let chosen = self.entries[self.selected].path.clone();
        self.entries.truncate(self.n_live);
        let mut saved = Vec::new();
        scan(&self.repo.join("output"), 3, &mut saved);
        saved.sort_by_key(|e| std::cmp::Reverse(e.modified));
        self.entries.extend(saved);
        self.selected = chosen
            .and_then(|p| {
                self.entries
                    .iter()
                    .position(|e| e.path.as_ref() == Some(&p))
            })
            .unwrap_or(self.selected.min(self.entries.len() - 1));
        self.scroll = self.scroll.min(self.n_saved().saturating_sub(self.visible));
        self.reveal();
        self.dirty = true;
    }

    /// A line under the lists, replacing the last: how a run went, or why it could
    /// not start.
    pub fn say(&mut self, line: String, error: bool) {
        self.status = vec![(line, error)];
        self.dirty = true;
    }

    fn n_saved(&self) -> usize {
        self.entries.len() - self.n_live
    }

    fn select(&mut self, i: usize) {
        self.selected = i.min(self.entries.len() - 1);
        self.reveal();
        self.dirty = true;
    }

    /// Scroll the saved runs so that the selected one shows.
    fn reveal(&mut self) {
        if let Some(j) = self.selected.checked_sub(self.n_live) {
            if j < self.scroll {
                self.scroll = j;
            } else if j >= self.scroll + self.visible {
                self.scroll = j + 1 - self.visible;
            }
        }
        let entry = &mut self.entries[self.selected];
        if entry.details.is_none() {
            entry.details = Some(details(entry));
        }
    }

    fn entry(&self) -> &Entry {
        &self.entries[self.selected]
    }

    fn gauge(&self) -> Option<PathBuf> {
        let path = self.repo.join(GAUGE);
        let frøya = !matches!(
            self.entry().kind,
            Kind::Live("fjord" | "channel" | "fjord3d")
        );
        (frøya && path.is_file()).then_some(path)
    }

    /// Where saving would write, or why it cannot.
    fn save_path(&self) -> Result<PathBuf, &'static str> {
        let entry = self.entry();
        match entry.kind {
            Kind::Live(scenario) => {
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map_or(0.0, |d| d.as_secs_f64());
                let stamp: String = ModelClock::new(now)
                    .format(0.0)
                    .chars()
                    .filter_map(|c| match c {
                        ' ' => Some('-'),
                        ':' | '-' => None,
                        c => Some(c),
                    })
                    .collect();
                Ok(self
                    .repo
                    .join("output/viz")
                    .join(format!("{scenario}-{stamp}.dgsnap")))
            }
            Kind::Snapshot => Err("a snapshot file already"),
            Kind::Frames => {
                let path = entry.path.as_ref().unwrap().join("froya.dgsnap");
                if path.exists() {
                    Err("froya.dgsnap is there already")
                } else {
                    Ok(path)
                }
            }
        }
    }

    /// The option's value as shown, and whether it applies to the selection.
    fn choice(&self, choice: Choice) -> (String, bool) {
        let o = &self.options;
        let on_off = |b: bool| if b { "on" } else { "off" }.to_string();
        let live = matches!(self.entry().kind, Kind::Live(_));
        match choice {
            Choice::Hours => (format!("{} h", HOURS[o.hours]), live),
            Choice::View => (
                if o.overview { "overview" } else { "close-up" }.into(),
                true,
            ),
            Choice::Photo => (on_off(o.photo), true),
            Choice::Gauge => match self.gauge() {
                Some(_) => (on_off(o.gauge), true),
                None => ("—".into(), false),
            },
            Choice::Save => match self.save_path() {
                Ok(_) => (on_off(o.save), true),
                Err(why) => (why.into(), false),
            },
        }
    }

    fn cycle(&mut self, choice: Choice) {
        if !self.choice(choice).1 {
            return;
        }
        let o = &mut self.options;
        match choice {
            Choice::Hours => o.hours = (o.hours + 1) % HOURS.len(),
            Choice::View => o.overview = !o.overview,
            Choice::Photo => o.photo = !o.photo,
            Choice::Gauge => o.gauge = !o.gauge,
            Choice::Save => o.save = !o.save,
        }
        self.dirty = true;
    }

    /// The viewer's options for the selection.
    fn command(&self) -> Vec<String> {
        let entry = self.entry();
        let o = &self.options;
        let mut args: Vec<String> = match entry.kind {
            Kind::Live(scenario) => vec![
                "--scenario".into(),
                scenario.into(),
                "--hours".into(),
                HOURS[o.hours].to_string(),
            ],
            Kind::Snapshot | Kind::Frames => vec![
                "--replay".into(),
                entry.path.as_ref().unwrap().display().to_string(),
            ],
        };
        args.extend([
            "--view".into(),
            if o.overview { "domain" } else { "farm" }.into(),
        ]);
        if o.photo {
            args.extend(["--photo".into(), "on".into()]);
        }
        if let (true, Some(gauge)) = (o.gauge, self.gauge()) {
            args.extend(["--gauge".into(), gauge.display().to_string()]);
        }
        if let (true, Ok(path)) = (o.save, self.save_path()) {
            args.extend(["--save-snapshot".into(), path.display().to_string()]);
        }
        args
    }
}

/// The saved runs under `dir`, `depth` folders down.
fn scan(dir: &Path, depth: usize, out: &mut Vec<Entry>) {
    let Ok(read) = std::fs::read_dir(dir) else {
        return;
    };
    let mut frames = Vec::new();
    for item in read.flatten() {
        let path = item.path();
        let Ok(meta) = item.metadata() else { continue };
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        if meta.is_dir() {
            if depth > 1 {
                scan(&path, depth - 1, out);
            }
        } else if path.extension().is_some_and(|e| e == "dgsnap") {
            out.push(Entry {
                kind: Kind::Snapshot,
                title: shown_path(&path),
                about: format!("snapshot file, {}", size(meta.len())),
                modified: meta.modified().ok(),
                path: Some(path),
                details: None,
                missing: None,
            });
        } else if name.starts_with("froya_") && name.ends_with(".vtu") {
            frames.push((meta.len(), meta.modified().ok()));
        }
    }
    if !frames.is_empty() {
        let bytes: u64 = frames.iter().map(|f| f.0).sum();
        let clock = crate::replay::folder_clock(dir).is_some();
        out.push(Entry {
            kind: Kind::Frames,
            title: format!("{}/", shown_path(dir)),
            about: format!(
                "{} VTU frames, {}{}",
                frames.len(),
                size(bytes),
                if clock { "" } else { ", no clock (no date)" }
            ),
            modified: frames.iter().filter_map(|f| f.1).max(),
            path: Some(dir.to_path_buf()),
            details: None,
            missing: None,
        });
    }
}

/// `path` from the `output` folder on.
fn shown_path(path: &Path) -> String {
    let parts: Vec<String> = path
        .components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect();
    match parts.iter().rposition(|p| p == "output") {
        Some(i) => parts[i + 1..].join("/"),
        None => path.display().to_string(),
    }
}

fn size(bytes: u64) -> String {
    match bytes {
        b if b >= 1 << 30 => format!("{:.1} GB", b as f64 / (1u64 << 30) as f64),
        b if b >= 1 << 20 => format!("{:.0} MB", b as f64 / (1u64 << 20) as f64),
        b => format!("{:.0} kB", b as f64 / 1024.0),
    }
}

/// `YYYY-MM-DD HH:MM` (UTC) of a file time.
fn when(time: SystemTime) -> String {
    let unix = time
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64());
    ModelClock::new(unix).format(0.0)[..16].to_string()
}

/// A closer look at an entry: a snapshot's title, span and domain, a folder's clock.
fn details(entry: &Entry) -> String {
    let path = entry.path.as_deref();
    match (entry.kind, path) {
        (Kind::Snapshot, Some(path)) => {
            snapshot_details(path).unwrap_or_else(|e| format!("cannot read it: {e}"))
        }
        (Kind::Frames, Some(dir)) => {
            let clock = crate::replay::folder_clock(dir).map_or(
                "No clock.txt or run.log with the run's clock: no date, and no gauge in the \
                 trace. Write a clock.txt with the run's `Clock: t = 0 at …` line."
                    .to_string(),
                |c| format!("Starts {} UTC.", c.format(0.0)),
            );
            format!(
                "Frames of `froya_real_data mesh=data/froya_coast.msh`, replayed on the \
                 Frøya scenario built as the run was. {clock} Saving writes froya.dgsnap \
                 next to them, a tenth of their size."
            )
        }
        _ => entry.about.clone(),
    }
}

fn snapshot_details(path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let mut reader = SnapshotReader::open(path)?;
    let (title, clock, order, elements, levels) = {
        let h = reader.header();
        (
            h.metadata("title").unwrap_or("untitled").to_string(),
            h.clock,
            h.order,
            h.mesh.n_elements,
            h.n_levels(),
        )
    };
    let frames = reader.n_frames()?;
    let span = if frames > 0 {
        let (first, last) = (reader.time(0)?, reader.time(frames - 1)?);
        let every = if frames > 1 {
            format!(
                ", every {:.0} min",
                (last - first) / (frames - 1) as f64 / 60.0
            )
        } else {
            String::new()
        };
        let start = clock.map_or(String::new(), |c| {
            format!(" from {} UTC", &c.format(first)[..16])
        });
        format!(
            "{frames} frames over {:.1} h{every}{start}",
            (last - first) / 3600.0
        )
    } else {
        "no frames yet".into()
    };
    let dims = if levels > 0 {
        format!("3D, {levels} levels")
    } else {
        "2D".into()
    };
    Ok(format!(
        "{title}\n{span}.\n{elements} elements (P{order}), {dims}."
    ))
}

fn font(size: f32) -> TextFont {
    TextFont {
        font_size: FontSize::Px(size),
        ..default()
    }
}

/// Build the selection and show it ([`crate::loading`]).
fn launch(menu: &mut Menu, commands: &mut Commands, next: &mut NextState<Screen>) {
    if let Some(missing) = &menu.entry().missing {
        let line = format!("{}: {missing}", menu.entry().title);
        menu.say(line, true);
        return;
    }
    let args = menu.command();
    let quoted: Vec<String> = args
        .iter()
        .map(|a| {
            if a.contains(' ') {
                format!("\"{a}\"")
            } else {
                a.clone()
            }
        })
        .collect();
    println!("dg-viz: cargo run --release -- {}", quoted.join(" "));
    let title = menu.entry().title.clone();
    menu.say(format!("Started {title}."), false);
    commands.insert_resource(Launch { args, title });
    next.set(Screen::Loading);
}

fn keys(
    keys: Res<ButtonInput<KeyCode>>,
    mut menu: ResMut<Menu>,
    mut commands: Commands,
    mut next: ResMut<NextState<Screen>>,
    mut exit: MessageWriter<AppExit>,
) {
    let n = menu.entries.len();
    let s = menu.selected;
    if keys.just_pressed(KeyCode::ArrowDown) {
        menu.select((s + 1) % n);
    } else if keys.just_pressed(KeyCode::ArrowUp) {
        menu.select((s + n - 1) % n);
    } else if keys.just_pressed(KeyCode::PageDown) {
        let page = menu.visible;
        menu.select(s + page);
    } else if keys.just_pressed(KeyCode::PageUp) {
        let page = menu.visible;
        menu.select(s.saturating_sub(page));
    } else if keys.just_pressed(KeyCode::Home) {
        menu.select(0);
    } else if keys.just_pressed(KeyCode::End) {
        menu.select(n - 1);
    } else if keys.just_pressed(KeyCode::Enter) || keys.just_pressed(KeyCode::NumpadEnter) {
        launch(&mut menu, &mut commands, &mut next);
    } else if keys.just_pressed(KeyCode::F5) {
        menu.rescan();
        menu.status = vec![(format!("{} saved runs found.", menu.n_saved()), false)];
    } else if keys.just_pressed(KeyCode::Escape) {
        exit.write(AppExit::Success);
    }
    for choice in Choice::ALL {
        if keys.just_pressed(choice.key().0) {
            menu.cycle(choice);
        }
    }
}

fn mouse(
    buttons: Res<ButtonInput<MouseButton>>,
    mut wheel: MessageReader<MouseWheel>,
    rows: Query<(&Target, &RelativeCursorPosition)>,
    mut menu: ResMut<Menu>,
    mut commands: Commands,
    mut next: ResMut<NextState<Screen>>,
) {
    let lines: f32 = wheel
        .read()
        .map(|w| match w.unit {
            MouseScrollUnit::Line => w.y,
            MouseScrollUnit::Pixel => w.y / 40.0,
        })
        .sum();
    if lines.abs() >= 0.5 {
        let max = menu.n_saved().saturating_sub(menu.visible);
        let to = (menu.scroll as f32 - lines.round()).clamp(0.0, max as f32) as usize;
        if to != menu.scroll {
            menu.scroll = to;
            menu.dirty = true;
        }
    }
    if !buttons.just_pressed(MouseButton::Left) {
        return;
    }
    let Some(target) = rows
        .iter()
        .find(|(_, cursor)| cursor.cursor_over())
        .map(|(t, _)| *t)
    else {
        return;
    };
    match target {
        Target::Entry(i) => {
            let double = menu
                .last_click
                .is_some_and(|(j, at)| j == i && at.elapsed() < Duration::from_millis(450));
            menu.select(i);
            menu.last_click = Some((i, Instant::now()));
            if double {
                menu.last_click = None;
                launch(&mut menu, &mut commands, &mut next);
            }
        }
        Target::Choice(choice) => menu.cycle(choice),
        Target::Launch => launch(&mut menu, &mut commands, &mut next),
    }
}

/// A snapshot file, or a folder of VTU frames (or one frame of it), dropped on the
/// window is replayed, with the menu's options; in the viewer, in place of its run.
fn dropped(
    mut drops: MessageReader<FileDragAndDrop>,
    mut menu: ResMut<Menu>,
    mut commands: Commands,
    mut next: ResMut<NextState<Screen>>,
) {
    for drop in drops.read() {
        let FileDragAndDrop::DroppedFile { path_buf, .. } = drop else {
            continue;
        };
        let path = if path_buf.extension().is_some_and(|e| e == "vtu") {
            path_buf.parent().map(Path::to_path_buf).unwrap_or_default()
        } else {
            path_buf.clone()
        };
        let mut found = Vec::new();
        if path.is_dir() {
            scan(&path, 1, &mut found);
            found.retain(|e| e.kind == Kind::Frames);
        } else if path.extension().is_some_and(|e| e == "dgsnap") {
            scan(path.parent().unwrap_or(Path::new(".")), 1, &mut found);
            found.retain(|e| e.path.as_ref() == Some(&path));
        }
        let Some(entry) = found.pop() else {
            menu.status = vec![(
                format!(
                    "{}: not a snapshot file (.dgsnap) or a folder of VTU frames",
                    path_buf.display()
                ),
                true,
            )];
            menu.dirty = true;
            continue;
        };
        let i = match menu.entries.iter().position(|e| e.path == entry.path) {
            Some(i) => i,
            None => {
                let at = menu.n_live;
                menu.entries.insert(at, entry);
                at
            }
        };
        menu.select(i);
        launch(&mut menu, &mut commands, &mut next);
    }
}

/// Rebuild the window's contents after a change.
/// As many saved runs as the window's height has room for.
fn fit(window: Single<&Window, With<PrimaryWindow>>, mut menu: ResMut<Menu>) {
    let rows = ((window.resolution.height() - FIXED_HEIGHT) / ROW_HEIGHT).floor();
    let visible = (rows.max(3.0) as usize).min(60);
    if visible != menu.visible {
        menu.visible = visible;
        let max = menu.n_saved().saturating_sub(visible);
        menu.scroll = menu.scroll.min(max);
        menu.reveal();
        menu.dirty = true;
    }
}

fn draw(mut commands: Commands, mut menu: ResMut<Menu>, roots: Query<Entity, With<MenuRoot>>) {
    if !menu.dirty {
        return;
    }
    menu.dirty = false;
    for root in &roots {
        commands.entity(root).despawn();
    }
    let menu = &*menu;
    commands
        .spawn((
            MenuRoot,
            DespawnOnExit(Screen::Menu),
            Node {
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                flex_direction: FlexDirection::Column,
                padding: UiRect::axes(Val::Px(32.0), Val::Px(24.0)),
                row_gap: Val::Px(14.0),
                ..default()
            },
        ))
        .with_children(|root| {
            root.spawn(Node {
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(2.0),
                ..default()
            })
            .with_children(|head| {
                head.spawn((Text::new("dg-viz"), font(28.0), TextColor(ACCENT)));
                head.spawn((
                    Text::new(format!(
                        "Run a model or replay a saved run. Saved runs are looked for in {}.",
                        menu.repo.join("output").display()
                    )),
                    font(13.0),
                    TextColor(DIM),
                ));
            });
            root.spawn(Node {
                flex_grow: 1.0,
                column_gap: Val::Px(24.0),
                ..default()
            })
            .with_children(|body| {
                body.spawn(panel(64.0))
                    .with_children(|list| entries(list, menu));
                body.spawn(panel(36.0))
                    .with_children(|side| choices(side, menu));
            });
            root.spawn(Node {
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(2.0),
                min_height: Val::Px(40.0),
                ..default()
            })
            .with_children(|status| {
                for (line, error) in &menu.status {
                    status.spawn((
                        Text::new(line.clone()),
                        font(13.0),
                        TextColor(if *error { ERROR } else { Color::WHITE }),
                    ));
                }
            });
            root.spawn((
                Text::new(
                    "↑ ↓ select   Enter or double-click start   H V P G S options   \
                     F5 look again   Esc quit   drop a .dgsnap file or a folder of VTU \
                     frames here to replay it",
                ),
                font(12.0),
                TextColor(FAINT),
            ));
        });
}

fn panel(width: f32) -> impl Bundle {
    (
        Node {
            width: Val::Percent(width),
            flex_direction: FlexDirection::Column,
            row_gap: Val::Px(4.0),
            padding: UiRect::all(Val::Px(16.0)),
            ..default()
        },
        BackgroundColor(PANEL),
    )
}

fn heading(parent: &mut ChildSpawnerCommands, text: String) {
    parent.spawn((
        Text::new(text),
        font(15.0),
        TextColor(ACCENT),
        Node {
            margin: UiRect::vertical(Val::Px(6.0)),
            ..default()
        },
    ));
}

/// A clickable row: a title and a line under it, or after it (`inline`, the saved
/// runs, to fit more of them).
fn row(
    parent: &mut ChildSpawnerCommands,
    target: Target,
    title: String,
    about: String,
    dim: bool,
    inline: bool,
) {
    parent
        .spawn((
            target,
            RelativeCursorPosition::default(),
            BackgroundColor(Color::NONE),
            Node {
                flex_direction: if inline {
                    FlexDirection::Row
                } else {
                    FlexDirection::Column
                },
                align_items: if inline {
                    AlignItems::Baseline
                } else {
                    AlignItems::Start
                },
                column_gap: Val::Px(16.0),
                padding: UiRect::axes(Val::Px(10.0), Val::Px(if inline { 4.0 } else { 5.0 })),
                ..default()
            },
        ))
        .with_children(|r| {
            r.spawn((
                Text::new(title),
                font(15.0),
                TextColor(if dim { DIM } else { Color::WHITE }),
            ));
            r.spawn((Text::new(about), font(12.0), TextColor(DIM)));
        });
}

fn entries(list: &mut ChildSpawnerCommands, menu: &Menu) {
    heading(list, "Run the model".into());
    for (i, e) in menu.entries[..menu.n_live].iter().enumerate() {
        let about = match &e.missing {
            Some(missing) => format!("{} ({missing})", e.about),
            None => e.about.clone(),
        };
        row(
            list,
            Target::Entry(i),
            e.title.clone(),
            about,
            e.missing.is_some(),
            false,
        );
    }
    let n_saved = menu.n_saved();
    heading(
        list,
        match n_saved {
            0 => "Replay a saved run: none found in output/ yet".into(),
            n => format!("Replay a saved run ({n}, newest first)"),
        },
    );
    if menu.scroll > 0 {
        list.spawn((
            Text::new(format!("  ↑ {} more", menu.scroll)),
            font(12.0),
            TextColor(FAINT),
        ));
    }
    let shown = menu.n_live + menu.scroll
        ..(menu.n_live + menu.scroll + menu.visible).min(menu.entries.len());
    for i in shown.clone() {
        let e = &menu.entries[i];
        let about = match e.modified {
            Some(t) => format!("{}   {}", when(t), e.about),
            None => e.about.clone(),
        };
        row(list, Target::Entry(i), e.title.clone(), about, false, true);
    }
    let below = menu.entries.len() - shown.end;
    if below > 0 {
        list.spawn((
            Text::new(format!("  ↓ {below} more")),
            font(12.0),
            TextColor(FAINT),
        ));
    }
}

fn choices(side: &mut ChildSpawnerCommands, menu: &Menu) {
    let entry = menu.entry();
    heading(side, entry.title.clone());
    side.spawn((
        Text::new(entry.details.clone().unwrap_or_else(|| entry.about.clone())),
        font(13.0),
        TextColor(DIM),
        Node {
            margin: UiRect::bottom(Val::Px(10.0)),
            ..default()
        },
    ));
    heading(side, "Options".into());
    for choice in Choice::ALL {
        let (value, applies) = menu.choice(choice);
        let name = match choice {
            Choice::Hours => "Model hours",
            Choice::View => "Start in",
            Choice::Photo => "Photo view",
            Choice::Gauge => "Mausund gauge in the trace",
            Choice::Save => "Save as a snapshot file",
        };
        side.spawn((
            Target::Choice(choice),
            RelativeCursorPosition::default(),
            BackgroundColor(Color::NONE),
            Node {
                justify_content: JustifyContent::SpaceBetween,
                padding: UiRect::axes(Val::Px(10.0), Val::Px(4.0)),
                ..default()
            },
        ))
        .with_children(|r| {
            let colour = if applies { Color::WHITE } else { FAINT };
            r.spawn((
                Text::new(format!("{}  {name}", choice.key().1)),
                font(14.0),
                TextColor(colour),
            ));
            r.spawn((
                Text::new(value),
                font(14.0),
                TextColor(if applies { ACCENT } else { FAINT }),
            ));
        });
    }
    side.spawn(Node {
        flex_grow: 1.0,
        ..default()
    });
    let ready = entry.missing.is_none();
    side.spawn((
        Target::Launch,
        RelativeCursorPosition::default(),
        BackgroundColor(if ready { SELECTED } else { HOVERED }),
        Node {
            justify_content: JustifyContent::Center,
            padding: UiRect::all(Val::Px(12.0)),
            ..default()
        },
    ))
    .with_children(|b| {
        b.spawn((
            Text::new(match entry.kind {
                Kind::Live(_) => "Run  (Enter)",
                _ => "Replay  (Enter)",
            }),
            font(17.0),
            TextColor(if ready { Color::WHITE } else { FAINT }),
        ));
    });
}

/// The selected row stands out, the one under the cursor a little.
fn highlight(
    menu: Res<Menu>,
    mut rows: Query<(&Target, &RelativeCursorPosition, &mut BackgroundColor)>,
) {
    for (target, cursor, mut background) in &mut rows {
        let colour = match target {
            Target::Entry(i) if *i == menu.selected => SELECTED,
            Target::Launch => SELECTED,
            _ if cursor.cursor_over() => HOVERED,
            _ => Color::NONE,
        };
        if background.0 != colour {
            background.0 = colour;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repo() -> PathBuf {
        PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/.."))
    }

    /// A live scenario becomes `--scenario` with its hours, a saved run `--replay`
    /// with its path, and the options follow the choices that apply.
    #[test]
    fn the_command_follows_the_selection_and_the_options() {
        let mut menu = Menu::new(repo());
        menu.select(2);
        assert_eq!(
            menu.command(),
            ["--scenario", "channel", "--hours", "25", "--view", "farm"]
        );
        menu.cycle(Choice::Hours);
        menu.cycle(Choice::View);
        menu.cycle(Choice::Photo);
        // No gauge for the channel: it is not at Frøya
        menu.cycle(Choice::Gauge);
        let command = menu.command();
        assert_eq!(
            &command[..8],
            [
                "--scenario",
                "channel",
                "--hours",
                "48",
                "--view",
                "domain",
                "--photo",
                "on"
            ]
        );
        assert_eq!(command.len(), 8);

        // The fjord in 3D, after the channel; no gauge there either
        menu.select(3);
        assert_eq!(&menu.command()[..2], ["--scenario", "fjord3d"]);
        assert!(menu.gauge().is_none());

        menu.entries.push(Entry {
            kind: Kind::Snapshot,
            title: "x/froya.dgsnap".into(),
            about: String::new(),
            path: Some(PathBuf::from("x/froya.dgsnap")),
            modified: None,
            details: Some(String::new()),
            missing: None,
        });
        menu.select(menu.entries.len() - 1);
        assert_eq!(&menu.command()[..2], ["--replay", "x/froya.dgsnap"]);
        assert!(!menu.choice(Choice::Hours).1, "a replay has its own hours");
        assert!(!menu.choice(Choice::Save).1, "a snapshot is saved already");
        menu.options.save = true;
        assert!(!menu.command().contains(&"--save-snapshot".to_string()));
    }

    /// A saved live run goes to `output/viz`, named by scenario and time.
    #[test]
    fn a_live_run_is_saved_under_output_viz() {
        let mut menu = Menu::new(repo());
        menu.select(0);
        let path = menu.save_path().unwrap();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        assert!(path.parent().unwrap().ends_with("output/viz"), "{path:?}");
        assert!(
            name.starts_with("fjord-") && name.ends_with(".dgsnap"),
            "{name}"
        );
        // fjord-YYYYMMDD-HHMMSS.dgsnap
        assert_eq!(name.len(), "fjord-".len() + 15 + ".dgsnap".len(), "{name}");
    }

    #[test]
    fn paths_are_shown_from_the_output_folder() {
        assert_eq!(
            shown_path(Path::new("/a/roms-rs/output/froya_15d/froya.dgsnap")),
            "froya_15d/froya.dgsnap"
        );
        assert_eq!(size(3 << 30), "3.0 GB");
        assert_eq!(size(18 << 20), "18 MB");
    }

    /// Saved runs are found in nested folders: snapshot files and folders of
    /// frames, the folders' `run.log` noted.
    #[test]
    fn saved_runs_are_found() {
        let root = std::env::temp_dir().join(format!("dg-viz-menu-{}", std::process::id()));
        let frames = root.join("output/run_a");
        let nested = root.join("output/b/c");
        std::fs::create_dir_all(&frames).unwrap();
        std::fs::create_dir_all(&nested).unwrap();
        for i in 0..3 {
            std::fs::write(frames.join(format!("froya_{i:04}.vtu")), [0u8; 100]).unwrap();
        }
        std::fs::write(
            frames.join("run.log"),
            "  Clock: t = 0 at 2025-06-15 00:00:00 UTC\n",
        )
        .unwrap();
        std::fs::write(nested.join("froya.dgsnap"), [0u8; 10]).unwrap();
        std::fs::write(nested.join("notes.txt"), "x").unwrap();
        let mut found = Vec::new();
        scan(&root.join("output"), 3, &mut found);
        std::fs::remove_dir_all(&root).unwrap();
        found.sort_by(|a, b| a.title.cmp(&b.title));
        assert_eq!(found.len(), 2);
        assert_eq!(
            (found[0].kind, found[0].title.as_str()),
            (Kind::Snapshot, "b/c/froya.dgsnap")
        );
        assert_eq!(
            (found[1].kind, found[1].title.as_str()),
            (Kind::Frames, "run_a/")
        );
        assert_eq!(found[1].about, "3 VTU frames, 0 kB");
    }
}

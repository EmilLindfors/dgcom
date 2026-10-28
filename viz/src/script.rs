//! `--menu --script FILE`: keys pressed and screenshots taken at set times, for
//! checking the screens and the way between them without anyone at the keyboard
//! (on a locked desktop, too: the keys go to the app, not the window system).
//!
//! One step per line, run in order; `#` starts a comment:
//! - `wait S`: let S seconds of wall time pass
//! - `key NAME`: press and release a key: a letter or digit (`m`, `1`), `Enter`,
//!   `Escape`, `Space`, `ArrowUp`, `ArrowDown`, `ArrowLeft`, `ArrowRight`, `Home`,
//!   `End`, `F1`
//! - `shot PATH`: save the window to PATH, and wait a second for it (the image is
//!   read back some frames later, and should not show the next step)
//! - `say TEXT`: print TEXT with the time since the start (to line up a log)
//! - `quit`: end the app

use std::collections::VecDeque;
use std::path::Path;
use std::time::{Duration, Instant};

use bevy::input::ButtonState;
use bevy::input::keyboard::{Key, KeyboardInput};
use bevy::prelude::*;
use bevy::render::view::screenshot::{Screenshot, save_to_disk};
use bevy::window::PrimaryWindow;

#[derive(Clone, Debug, PartialEq)]
enum Step {
    Wait(Duration),
    Key(KeyCode, Key),
    Shot(String),
    Say(String),
    Quit,
}

#[derive(Resource)]
struct Script {
    steps: VecDeque<Step>,
    started: Instant,
    /// Until when the current wait lasts
    until: Option<Instant>,
    /// A key pressed last frame, to release
    held: Option<(KeyCode, Key)>,
}

/// Read the script at `path` and run it in `app`.
pub fn run(app: &mut App, path: &Path) -> Result<(), String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let steps = parse(&text)?;
    app.insert_resource(Script {
        steps,
        started: Instant::now(),
        until: None,
        held: None,
    })
    .add_systems(First, step);
    Ok(())
}

fn parse(text: &str) -> Result<VecDeque<Step>, String> {
    text.lines()
        .enumerate()
        .map(|(i, line)| (i + 1, line.split('#').next().unwrap_or("").trim()))
        .filter(|(_, line)| !line.is_empty())
        .map(|(n, line)| {
            let (word, rest) = line.split_once(' ').unwrap_or((line, ""));
            let rest = rest.trim();
            match word {
                "wait" => rest
                    .parse::<f64>()
                    .ok()
                    .filter(|s| *s >= 0.0)
                    .map(|s| Step::Wait(Duration::from_secs_f64(s))),
                "key" => key(rest).map(|(code, key)| Step::Key(code, key)),
                "shot" if !rest.is_empty() => Some(Step::Shot(rest.to_string())),
                "say" => Some(Step::Say(rest.to_string())),
                "quit" => Some(Step::Quit),
                _ => None,
            }
            .ok_or_else(|| format!("line {n} of the script: cannot read `{line}`"))
        })
        .collect()
}

/// A key by name: its physical code and the logical key it gives on a US layout.
fn key(name: &str) -> Option<(KeyCode, Key)> {
    let named = |code, key| Some((code, key));
    match name {
        "Enter" => named(KeyCode::Enter, Key::Enter),
        "Escape" => named(KeyCode::Escape, Key::Escape),
        "Space" => named(KeyCode::Space, Key::Space),
        "ArrowUp" => named(KeyCode::ArrowUp, Key::ArrowUp),
        "ArrowDown" => named(KeyCode::ArrowDown, Key::ArrowDown),
        "ArrowLeft" => named(KeyCode::ArrowLeft, Key::ArrowLeft),
        "ArrowRight" => named(KeyCode::ArrowRight, Key::ArrowRight),
        "F1" => named(KeyCode::F1, Key::F1),
        "Home" => named(KeyCode::Home, Key::Home),
        "End" => named(KeyCode::End, Key::End),
        _ => {
            let mut chars = name.chars();
            let (Some(c), None) = (chars.next(), chars.next()) else {
                return None;
            };
            const LETTERS: [KeyCode; 26] = [
                KeyCode::KeyA,
                KeyCode::KeyB,
                KeyCode::KeyC,
                KeyCode::KeyD,
                KeyCode::KeyE,
                KeyCode::KeyF,
                KeyCode::KeyG,
                KeyCode::KeyH,
                KeyCode::KeyI,
                KeyCode::KeyJ,
                KeyCode::KeyK,
                KeyCode::KeyL,
                KeyCode::KeyM,
                KeyCode::KeyN,
                KeyCode::KeyO,
                KeyCode::KeyP,
                KeyCode::KeyQ,
                KeyCode::KeyR,
                KeyCode::KeyS,
                KeyCode::KeyT,
                KeyCode::KeyU,
                KeyCode::KeyV,
                KeyCode::KeyW,
                KeyCode::KeyX,
                KeyCode::KeyY,
                KeyCode::KeyZ,
            ];
            const DIGITS: [KeyCode; 10] = [
                KeyCode::Digit0,
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
            let code = match c {
                'a'..='z' => LETTERS[c as usize - 'a' as usize],
                '0'..='9' => DIGITS[c as usize - '0' as usize],
                _ => return None,
            };
            Some((code, Key::Character(c.to_string().into())))
        }
    }
}

/// Take the script's steps that are due: a key is held for one frame.
fn step(
    mut script: ResMut<Script>,
    window: Option<Single<Entity, With<PrimaryWindow>>>,
    mut input: MessageWriter<KeyboardInput>,
    mut commands: Commands,
    mut exit: MessageWriter<AppExit>,
) {
    let Some(window) = window.map(|w| *w) else {
        return;
    };
    let message = |(key_code, logical_key): (KeyCode, Key), state| KeyboardInput {
        key_code,
        logical_key,
        state,
        text: None,
        repeat: false,
        window,
    };
    if let Some(held) = script.held.take() {
        input.write(message(held, ButtonState::Released));
        return;
    }
    if script.until.is_some_and(|until| Instant::now() < until) {
        return;
    }
    script.until = None;
    while let Some(step) = script.steps.pop_front() {
        match step {
            Step::Wait(d) => {
                script.until = Some(Instant::now() + d);
                return;
            }
            Step::Key(code, key) => {
                input.write(message((code, key.clone()), ButtonState::Pressed));
                script.held = Some((code, key));
                return;
            }
            Step::Shot(path) => {
                commands
                    .spawn(Screenshot::primary_window())
                    .observe(save_to_disk(path));
                script.until = Some(Instant::now() + Duration::from_secs(1));
                return;
            }
            Step::Say(text) => {
                println!("[{:.1} s] {text}", script.started.elapsed().as_secs_f64());
            }
            Step::Quit => {
                exit.write(AppExit::Success);
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_script_is_read_step_by_step() {
        let steps =
            parse("wait 2.5  # settle\n\nkey m\nkey Enter\nshot a b.png\nsay hi there\nquit\n")
                .unwrap();
        assert_eq!(
            Vec::from(steps),
            [
                Step::Wait(Duration::from_secs_f64(2.5)),
                Step::Key(KeyCode::KeyM, Key::Character("m".into())),
                Step::Key(KeyCode::Enter, Key::Enter),
                Step::Shot("a b.png".into()),
                Step::Say("hi there".into()),
                Step::Quit,
            ]
        );
        assert!(parse("key Ctrl").unwrap_err().contains("line 1"));
        assert!(parse("wait\n").is_err());
        assert!(parse("\nwalk 3").unwrap_err().contains("line 2"));
    }
}

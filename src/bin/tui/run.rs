use crossterm::event::{Event, KeyEventKind};
use ratatui::prelude::Backend;
use std::io;
use std::time::Duration;

use super::app::App;
use super::keymap::{self, Action};

pub fn run<B: Backend>(
    terminal: &mut ratatui::Terminal<B>,
    app: &mut App,
    poll_duration: Duration,
) -> io::Result<()> {
    loop {
        if app.needs_redraw {
            terminal.draw(|f| super::ui::render(f, app))?;
            app.needs_redraw = false;
        }

        if crossterm::event::poll(poll_duration)? {
            let ev = crossterm::event::read()?;
            match ev {
                Event::Key(key) => {
                    if key.kind != KeyEventKind::Press {
                        continue;
                    }
                    if handle_event(app, key) {
                        return Ok(());
                    }
                    app.needs_redraw = true;
                }
                Event::Resize(_, _) => {
                    app.needs_redraw = true;
                }
                _ => {}
            }
        }
    }
}

pub fn handle_event(app: &mut App, key: crossterm::event::KeyEvent) -> bool {
    if app.mode == super::app::AppMode::Search {
        let action = keymap::from_key_event(key, &app.keymap);
        match action {
            Action::Cancel => {
                app.set_mode_normal();
                app.search_clear();
            }
            Action::Open => {
                app.set_mode_normal();
            }
            Action::None => {
                if let crossterm::event::KeyCode::Char(c) = key.code {
                    if c.is_ascii() && !c.is_control() {
                        app.search_push_char(c);
                    }
                }
                if key.code == crossterm::event::KeyCode::Backspace {
                    app.search_pop_char();
                }
            }
            _ => {}
        }
        return false;
    }

    if app.mode != super::app::AppMode::Normal {
        let action = keymap::from_key_event(key, &app.keymap);
        if action == Action::Cancel {
            app.set_mode_normal();
        }
        return false;
    }

    let action = keymap::from_key_event(key, &app.keymap);
    match action {
        Action::Quit => return true,
        Action::FocusNext => app.focus_next(),
        Action::FocusPrev => app.focus_prev(),
        Action::FocusSample => app.focus_sample(),
        Action::FocusTag => app.focus_tag(),
        Action::FocusExperiment => app.focus_experiment(),
        Action::FocusBrowser => app.focus_browser(),
        Action::MoveDown => app.list_down(),
        Action::MoveUp => app.list_up(),
        Action::MoveFirst => app.list_first(),
        Action::MoveLast => app.list_last(),
        Action::Search => app.set_mode_search(),
        Action::Cancel => {}
        Action::Rename => app.set_mode_rename(),
        Action::Retag => app.set_mode_retag(),
        Action::Open => {
            if matches!(
                app.focus,
                super::app::Focus::SampleList
                    | super::app::Focus::TagList
                    | super::app::Focus::ExperimentList
            ) {
                app.toggle_filter();
            }
        }
        Action::None => {}
    }
    false
}

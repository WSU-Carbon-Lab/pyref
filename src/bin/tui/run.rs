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
    terminal.clear()?;
    terminal.draw(|f| super::ui::render(f, app))?;
    loop {
        if app.try_complete_loading() {
            app.needs_redraw = true;
        }
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
    use crossterm::event::KeyCode;

    if app.mode == super::app::AppMode::Search {
        match key.code {
            KeyCode::Esc => {
                app.set_mode_normal();
                app.search_clear();
            }
            KeyCode::Enter => {
                app.set_mode_normal();
            }
            KeyCode::Backspace => app.search_pop_char(),
            KeyCode::Char(c) if c.is_ascii() && !c.is_control() => app.search_push_char(c),
            _ => {}
        }
        return false;
    }

    if app.mode == super::app::AppMode::ChangeDir {
        let action = keymap::from_key_event(key, &app.keymap);
        if action == Action::AcceptPath {
            app.apply_path();
            return false;
        }
        if action == Action::MoveDown {
            app.dir_browser_move_down();
            return false;
        }
        if action == Action::MoveUp {
            app.dir_browser_move_up();
            return false;
        }
        if action == Action::MoveFirst {
            app.dir_browser_move_first();
            return false;
        }
        if action == Action::MoveLast {
            app.dir_browser_move_last();
            return false;
        }
        match key.code {
            KeyCode::Esc => {
                app.set_mode_normal();
                app.path_clear();
            }
            KeyCode::Enter => app.open_selected_dir(),
            KeyCode::Tab => app.path_autocomplete(),
            KeyCode::Backspace => app.path_pop_char(),
            KeyCode::Char('j') => app.dir_browser_move_down(),
            KeyCode::Char('k') => app.dir_browser_move_up(),
            KeyCode::Down => app.dir_browser_move_down(),
            KeyCode::Up => app.dir_browser_move_up(),
            KeyCode::Char(c) if c.is_ascii() && !c.is_control() => app.path_push_char(c),
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
        Action::ChangeDir => app.set_mode_change_dir(),
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
        Action::AcceptPath => {}
        Action::None => {}
    }
    false
}

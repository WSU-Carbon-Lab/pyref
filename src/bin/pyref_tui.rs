#![cfg(feature = "tui")]

mod tui;

use crossterm::event::{self, Event, KeyEventKind};
use ratatui::backend::CrosstermBackend;
use ratatui::prelude::{Backend, Terminal};
use std::io;
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tui::terminal_guard::install_panic_hook();
    let config = tui::TuiConfig::load();
    let current_root = config
        .last_root
        .clone()
        .unwrap_or_else(|| "/path/to/experiments".to_string());

    let mut terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;
    tui::terminal_guard::setup_terminal()?;

    let poll_duration = config
        .poll_interval_ms
        .and_then(|ms| if ms == 0 { None } else { Some(ms) })
        .map(Duration::from_millis)
        .unwrap_or(Duration::from_secs(30));

    let mut app = tui::App::new(
        current_root.clone(),
        config.layout.clone(),
        config.keymap.clone(),
        config.keybind_bar_lines,
        config.theme.clone(),
    );
    if let Some(s) = config.last_sample.as_ref() {
        if let Some(i) = app.samples.iter().position(|x| x == s) {
            app.sample_state.select(Some(i));
        }
    }
    if let Some(t) = config.last_tag.as_ref() {
        if let Some(i) = app.tags.iter().position(|x| x == t) {
            app.tag_state.select(Some(i));
        }
    }
    if let Some(e) = config.last_experiment.as_ref() {
        if let Ok(n) = e.parse::<u32>() {
            if let Some(i) = app.experiments.iter().position(|(x, _)| *x == n) {
                app.experiment_state.select(Some(i));
            }
        }
    }
    app.refresh_filtered();

    let mut config_save = config;

    let run = run_app(&mut terminal, &mut app, poll_duration);
    tui::terminal_guard::restore_terminal()?;

    if let Err(e) = run {
        eprintln!("{:?}", e);
    }

    config_save.set_last_root(&app.current_root);
    let exp_str = app.selected_experiment().map(|n| n.to_string());
    config_save.set_last_selection(
        app.selected_sample().as_deref(),
        app.selected_tag().as_deref(),
        exp_str.as_deref(),
    );
    let _ = config_save.save();

    Ok(())
}

fn run_app<B: Backend>(
    terminal: &mut Terminal<B>,
    app: &mut tui::App,
    poll_duration: Duration,
) -> io::Result<()> {
    loop {
        if app.needs_redraw {
            terminal.draw(|f| tui::render(f, app))?;
            app.needs_redraw = false;
        }

        if event::poll(poll_duration)? {
            let ev = event::read()?;
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

fn handle_event(app: &mut tui::App, key: crossterm::event::KeyEvent) -> bool {
    if app.mode == tui::AppMode::Search {
        let action = tui::keymap::from_key_event(key, &app.keymap);
        match action {
            tui::keymap::Action::Cancel => {
                app.set_mode_normal();
                app.search_clear();
            }
            tui::keymap::Action::Open => {
                app.set_mode_normal();
            }
            tui::keymap::Action::None => {
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

    if app.mode != tui::AppMode::Normal {
        let action = tui::keymap::from_key_event(key, &app.keymap);
        if action == tui::keymap::Action::Cancel {
            app.set_mode_normal();
        }
        return false;
    }

    let action = tui::keymap::from_key_event(key, &app.keymap);
    match action {
        tui::keymap::Action::Quit => return true,
        tui::keymap::Action::FocusNext => app.focus_next(),
        tui::keymap::Action::FocusPrev => app.focus_prev(),
        tui::keymap::Action::FocusSample => app.focus_sample(),
        tui::keymap::Action::FocusTag => app.focus_tag(),
        tui::keymap::Action::FocusExperiment => app.focus_experiment(),
        tui::keymap::Action::FocusBrowser => app.focus_browser(),
        tui::keymap::Action::MoveDown => app.list_down(),
        tui::keymap::Action::MoveUp => app.list_up(),
        tui::keymap::Action::MoveFirst => app.list_first(),
        tui::keymap::Action::MoveLast => app.list_last(),
        tui::keymap::Action::Search => app.set_mode_search(),
        tui::keymap::Action::Cancel => {}
        tui::keymap::Action::Rename => app.set_mode_rename(),
        tui::keymap::Action::Retag => app.set_mode_retag(),
        tui::keymap::Action::Open => {
            if matches!(
                app.focus,
                tui::Focus::SampleList | tui::Focus::TagList | tui::Focus::ExperimentList
            ) {
                app.toggle_filter();
            }
        }
        tui::keymap::Action::None => {}
    }
    false
}

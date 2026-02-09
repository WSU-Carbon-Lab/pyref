#![cfg(feature = "tui")]

mod tui;

use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::backend::CrosstermBackend;
use ratatui::prelude::{Backend, Terminal};
use std::io;
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = tui::TuiConfig::load();
    let current_root = config
        .last_root
        .clone()
        .unwrap_or_else(|| "/path/to/experiments".to_string());

    let mut terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;
    crossterm::terminal::enable_raw_mode()?;
    crossterm::execute!(io::stdout(), crossterm::terminal::EnterAlternateScreen)?;

    let mut app = tui::App::new(current_root.clone());
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

    let run = run_app(&mut terminal, &mut app);
    crossterm::terminal::disable_raw_mode()?;
    crossterm::execute!(io::stdout(), crossterm::terminal::LeaveAlternateScreen)?;
    terminal.show_cursor()?;

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

fn run_app<B: Backend>(terminal: &mut Terminal<B>, app: &mut tui::App) -> io::Result<()> {
    loop {
        terminal.draw(|f| tui::render(f, app))?;

        if event::poll(Duration::from_millis(100))? {
            let ev = event::read()?;
            if let Event::Key(key) = ev {
                if key.kind != KeyEventKind::Press {
                    continue;
                }
                if handle_key(app, key.code, key.modifiers) {
                    return Ok(());
                }
            }
        }
    }
}

fn handle_key(app: &mut tui::App, code: KeyCode, modifiers: KeyModifiers) -> bool {
    if app.mode != tui::AppMode::Normal {
        if code == KeyCode::Esc {
            app.set_mode_normal();
        }
        return false;
    }

    match code {
        KeyCode::Char('q') | KeyCode::Esc if modifiers.contains(KeyModifiers::CONTROL) => return true,
        KeyCode::Char('q') if modifiers.is_empty() => return true,
        KeyCode::Tab => {
            if modifiers.contains(KeyModifiers::SHIFT) {
                app.focus_prev();
            } else {
                app.focus_next();
            }
        }
        KeyCode::Down | KeyCode::Char('j') => app.list_down(),
        KeyCode::Up | KeyCode::Char('k') => app.list_up(),
        KeyCode::Char('r') => app.set_mode_rename(),
        KeyCode::Char('t') => app.set_mode_retag(),
        _ => {}
    }
    false
}

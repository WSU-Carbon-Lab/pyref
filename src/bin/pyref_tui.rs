#![cfg(feature = "tui")]

mod tui;

use ratatui::backend::CrosstermBackend;
use ratatui::prelude::Terminal;
use std::io;
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    tui::terminal_guard::install_panic_hook();

    let config = tui::TuiConfig::load_or_default();
    let current_root = config.last_root.clone().unwrap_or_else(String::new);

    let poll_duration = config
        .poll_interval_ms
        .and_then(|ms| if ms == 0 { None } else { Some(ms) })
        .map(Duration::from_millis)
        .unwrap_or(Duration::from_secs(30));

    let mut terminal = Terminal::new(CrosstermBackend::new(io::stdout()))
        .map_err(tui::TuiError::terminal_setup)?;
    tui::terminal_guard::setup_terminal().map_err(tui::TuiError::terminal_setup)?;

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
    app.set_mode_change_dir();

    let mut config_save = config;

    let run_result = tui::run(&mut terminal, &mut app, poll_duration);

    if let Err(e) = tui::terminal_guard::restore_terminal() {
        eprintln!("{}", tui::TuiError::terminal_restore(e).report());
    }

    if let Err(e) = run_result {
        let err = tui::TuiError::io("run_loop", e.to_string(), e);
        eprintln!("{}", err.report());
    }

    config_save.set_last_root(&app.current_root);
    let exp_str = app.focused_experiment().map(|n| n.to_string());
    config_save.set_last_selection(
        app.focused_sample().as_deref(),
        app.focused_tag().as_deref(),
        exp_str.as_deref(),
    );
    if let Err(e) = config_save.save() {
        eprintln!("{}", e.report());
    }

    Ok(())
}

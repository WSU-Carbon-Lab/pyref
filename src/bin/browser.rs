#![cfg(feature = "tui")]

mod tui;

use log::{LevelFilter, Log, Metadata, Record};
use ratatui::backend::CrosstermBackend;
use ratatui::prelude::Terminal;
use std::env;
use std::io;
use std::path::Path;
use std::sync::mpsc;
use std::time::Duration;

struct NoopLogger;

impl Log for NoopLogger {
    fn enabled(&self, _: &Metadata) -> bool {
        false
    }
    fn log(&self, _: &Record) {}
    fn flush(&self) {}
}

fn resolve_root(config: &tui::TuiConfig) -> Result<String, String> {
    let raw = env::args()
        .nth(1)
        .or_else(|| config.last_root.clone())
        .unwrap_or_else(|| "/path/to/experiments".to_string());
    let path = Path::new(&raw);
    if !path.exists() {
        return Err(format!(
            "Path does not exist: {}. Check the path or create the directory.",
            path.display()
        ));
    }
    if !path.is_dir() {
        return Err(format!(
            "Path is not a directory: {}. Point pyref-tui at a beamtime directory.",
            path.display()
        ));
    }
    let canonical = path
        .canonicalize()
        .map_err(|e| format!("Failed to resolve path {}: {}", path.display(), e))?;
    Ok(canonical.to_string_lossy().into_owned())
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    env::set_var("POLARS_VERBOSE", "0");
    static NOOP: NoopLogger = NoopLogger;
    let _ = log::set_logger(&NOOP).map(|()| log::set_max_level(LevelFilter::Off));

    tui::terminal_guard::install_panic_hook();

    let mut config = tui::TuiConfig::load_or_default();

    #[cfg(feature = "catalog")]
    let initial_root = if env::args().nth(1).is_some() {
        Some(match resolve_root(&config) {
            Ok(s) => s,
            Err(msg) => {
                eprintln!("{}", msg);
                std::process::exit(1);
            }
        })
    } else {
        None
    };

    #[cfg(not(feature = "catalog"))]
    let initial_root = Some(match resolve_root(&config) {
        Ok(s) => s,
        Err(msg) => {
            eprintln!("{}", msg);
            std::process::exit(1);
        }
    });

    let poll_duration = config
        .poll_interval_ms
        .and_then(|ms| if ms == 0 { None } else { Some(ms) })
        .map(Duration::from_millis)
        .unwrap_or(Duration::from_secs(30));

    let (preview_tx, preview_rx) = mpsc::channel();

    #[cfg(target_os = "macos")]
    {
        tui::preview::run_preview_window(preview_rx);
        let mut terminal = Terminal::new(CrosstermBackend::new(io::stdout()))
            .map_err(tui::TuiError::terminal_setup)?;
        tui::terminal_guard::setup_terminal().map_err(tui::TuiError::terminal_setup)?;

        let mut app = match initial_root {
            Some(ref current_root) => {
                let mut app = tui::App::new(
                    current_root.clone(),
                    config.layout.clone(),
                    config.keymap.clone(),
                    config.keybind_bar_lines,
                    config.theme.clone(),
                );
                app.set_preview_tx(preview_tx.clone());
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
                if let Some(e) = config.last_scan.as_ref() {
                    if let Ok(n) = e.parse::<u32>() {
                        if let Some(i) = app.scans.iter().position(|(x, _)| *x == n) {
                            app.scan_list_state.select(Some(i));
                        }
                    }
                }
                app.refresh_filtered();
                app
            }
            None => {
                #[cfg(feature = "catalog")]
                {
                    let mut app = tui::App::new_for_launcher(
                        config.layout.clone(),
                        config.keymap.clone(),
                        config.keybind_bar_lines,
                        config.theme.clone(),
                    );
                    app.set_preview_tx(preview_tx.clone());
                    app
                }
                #[cfg(not(feature = "catalog"))]
                unreachable!()
            }
        };

        let run_result = tui::run(&mut terminal, &mut app, poll_duration);

        if let Err(e) = tui::terminal_guard::restore_terminal() {
            eprintln!("{}", tui::TuiError::terminal_restore(e).report());
        }

        if let Err(e) = run_result {
            let err = tui::TuiError::io("run_loop", e.to_string(), e);
            eprintln!("{}", err.report());
        }

        if app.screen == tui::Screen::Beamtime && !app.current_root.is_empty() {
            config.set_last_root(&app.current_root);
            let scan_str = app.focused_scan().map(|n| n.to_string());
            config.set_last_selection(
                app.focused_sample().as_deref(),
                app.focused_tag().as_deref(),
                scan_str.as_deref(),
            );
            let mut sel_samples: Vec<String> = app.selected_samples.iter().cloned().collect();
            sel_samples.sort();
            let mut sel_tags: Vec<String> = app.selected_tags.iter().cloned().collect();
            sel_tags.sort();
            let mut sel_scans: Vec<u32> = app.selected_scans.iter().cloned().collect();
            sel_scans.sort();
            config.set_selection_export(&sel_samples, &sel_tags, &sel_scans);
        }
        if let Err(e) = config.save() {
            eprintln!("{}", e.report());
        }
    }

    #[cfg(not(target_os = "macos"))]
    {
        std::thread::spawn(move || {
            let mut terminal = match Terminal::new(CrosstermBackend::new(io::stdout())) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("{}", tui::TuiError::terminal_setup(e).report());
                    std::process::exit(1);
                }
            };
            if let Err(e) = tui::terminal_guard::setup_terminal() {
                eprintln!("{}", tui::TuiError::terminal_setup(e).report());
                std::process::exit(1);
            }

            let mut app = match initial_root {
                Some(ref current_root) => {
                    let mut app = tui::App::new(
                        current_root.clone(),
                        config.layout.clone(),
                        config.keymap.clone(),
                        config.keybind_bar_lines,
                        config.theme.clone(),
                    );
                    app.set_preview_tx(preview_tx.clone());
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
                    if let Some(e) = config.last_scan.as_ref() {
                        if let Ok(n) = e.parse::<u32>() {
                            if let Some(i) = app.scans.iter().position(|(x, _)| *x == n) {
                                app.scan_list_state.select(Some(i));
                            }
                        }
                    }
                    app.refresh_filtered();
                    app
                }
                None => {
                    #[cfg(feature = "catalog")]
                    {
                        let mut app = tui::App::new_for_launcher(
                            config.layout.clone(),
                            config.keymap.clone(),
                            config.keybind_bar_lines,
                            config.theme.clone(),
                        );
                        app.set_preview_tx(preview_tx.clone());
                        app
                    }
                    #[cfg(not(feature = "catalog"))]
                    unreachable!()
                }
            };

            let run_result = tui::run(&mut terminal, &mut app, poll_duration);

            if let Err(e) = tui::terminal_guard::restore_terminal() {
                eprintln!("{}", tui::TuiError::terminal_restore(e).report());
            }

            if let Err(e) = run_result {
                let err = tui::TuiError::io("run_loop", e.to_string(), e);
                eprintln!("{}", err.report());
            }

            if app.screen == tui::Screen::Beamtime && !app.current_root.is_empty() {
                config.set_last_root(&app.current_root);
                let scan_str = app.focused_scan().map(|n| n.to_string());
                config.set_last_selection(
                    app.focused_sample().as_deref(),
                    app.focused_tag().as_deref(),
                    scan_str.as_deref(),
                );
                let mut sel_samples: Vec<String> = app.selected_samples.iter().cloned().collect();
                sel_samples.sort();
                let mut sel_tags: Vec<String> = app.selected_tags.iter().cloned().collect();
                sel_tags.sort();
                let mut sel_scans: Vec<u32> = app.selected_scans.iter().cloned().collect();
                sel_scans.sort();
                config.set_selection_export(&sel_samples, &sel_tags, &sel_scans);
            }
            if let Err(e) = config.save() {
                eprintln!("{}", e.report());
            }

            std::process::exit(0);
        });

        tui::preview::run_preview_window(preview_rx);
        std::process::exit(0);
    }

    Ok(())
}

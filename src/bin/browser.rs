#![cfg(feature = "tui")]

mod tui;

use log::{LevelFilter, Log, Metadata, Record};
use ratatui::backend::CrosstermBackend;
use ratatui::prelude::Terminal;
use std::env;
use std::io;
use std::path::Path;
use std::path::PathBuf;
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

fn resolve_existing_dir(path: &Path) -> Result<String, String> {
    if !path.exists() {
        return Err(format!(
            "Path does not exist: {}. Check the path or create the directory.",
            path.display()
        ));
    }
    if !path.is_dir() {
        return Err(format!("Path is not a directory: {}.", path.display()));
    }
    let canonical = path
        .canonicalize()
        .map_err(|e| format!("Failed to resolve path {}: {}", path.display(), e))?;
    Ok(canonical.to_string_lossy().into_owned())
}

#[cfg(feature = "catalog")]
fn pick_startup_root(config: &tui::TuiConfig) -> Result<Option<String>, String> {
    if let Some(arg) = env::args().nth(1) {
        return Ok(Some(resolve_existing_dir(Path::new(&arg))?));
    }
    if let Ok(v) = env::var("PYREF_DATA_ROOT") {
        let p = Path::new(v.trim());
        if p.is_dir() {
            return Ok(Some(resolve_existing_dir(p)?));
        }
    }
    if let Some(ref dr) = config.data_root {
        let p = Path::new(dr);
        if p.is_dir() {
            return Ok(Some(resolve_existing_dir(p)?));
        }
    }
    #[cfg(target_os = "macos")]
    {
        let p = Path::new("/Volumes/DATA");
        if p.is_dir() {
            return Ok(Some(resolve_existing_dir(p)?));
        }
    }
    if let Some(ref lr) = config.last_root {
        let p = Path::new(lr);
        if p.is_dir() {
            return Ok(Some(resolve_existing_dir(p)?));
        }
    }
    Ok(None)
}

#[cfg(not(feature = "catalog"))]
fn resolve_root_fallback(config: &tui::TuiConfig) -> Result<String, String> {
    let raw = env::args()
        .nth(1)
        .or_else(|| config.last_root.clone())
        .unwrap_or_else(|| "/path/to/experiments".to_string());
    resolve_existing_dir(Path::new(&raw))
}

/// Detect if a path should be opened in Explorer mode.
/// Explorer mode: path contains only subdirectories (no .fits files directly inside)
/// Beamtime mode: path contains .fits files directly inside
fn should_open_as_explorer(path: &Path) -> bool {
    use std::fs;

    match fs::read_dir(path) {
        Ok(entries) => {
            // If all entries are directories, use Explorer mode
            entries
                .filter_map(|e| e.ok())
                .all(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
        }
        Err(_) => false,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    env::set_var("POLARS_VERBOSE", "0");
    static NOOP: NoopLogger = NoopLogger;
    let _ = log::set_logger(&NOOP).map(|()| log::set_max_level(LevelFilter::Off));

    tui::terminal_guard::install_panic_hook();

    let mut config = tui::TuiConfig::load_or_default();

    #[cfg(feature = "catalog")]
    let initial_root = match pick_startup_root(&config) {
        Ok(o) => o,
        Err(msg) => {
            eprintln!("{}", msg);
            std::process::exit(1);
        }
    };

    #[cfg(not(feature = "catalog"))]
    let initial_root = Some(match resolve_root_fallback(&config) {
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

    let (preview_tx, preview_rx) = mpsc::channel::<(PathBuf, Option<f64>)>();

    #[cfg(feature = "catalog")]
    let (beamspot_tx, beamspot_rx) = mpsc::channel::<(PathBuf, i64, i64, Option<f64>)>();

    #[cfg(feature = "catalog")]
    let (preview_cmd_tx, preview_cmd_rx) = mpsc::channel::<tui::preview::PreviewCommand>();

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
                let path = Path::new(current_root);
                if should_open_as_explorer(path) {
                    // Open in Explorer mode
                    tui::App::new_for_explorer(
                        path.to_path_buf(),
                        config.layout.clone(),
                        config.keymap.clone(),
                        config.keybind_bar_lines,
                        config.theme.clone(),
                    )
                } else {
                    // Open in Beamtime mode (original behavior)
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

        #[cfg(feature = "catalog")]
        app.set_beamspot_rx(beamspot_rx);

        #[cfg(feature = "catalog")]
        app.set_preview_cmd_rx(preview_cmd_rx);

        let run_result = tui::run(&mut terminal, &mut app, poll_duration);

        if let Err(e) = tui::terminal_guard::restore_terminal() {
            eprintln!("{}", tui::TuiError::terminal_restore(e).report());
        }

        if let Err(e) = run_result {
            let err = tui::TuiError::io("run_loop", e.to_string(), e);
            eprintln!("{}", err.report());
        }

        if app.current_screen() == tui::Screen::Explorer {
            if let Some(ex) = app.explorer_state() {
                let root = ex.data_root.to_string_lossy().into_owned();
                config.set_last_root(&root);
                config.data_root = Some(root);
            }
        }
        if app.current_screen() == tui::Screen::Beamtime && !app.current_root().is_empty() {
            config.set_last_root(app.current_root());
            let scan_str = app.focused_scan().map(|n| n.to_string());
            config.set_last_selection(
                app.focused_sample().as_deref(),
                app.focused_tag().as_deref(),
                scan_str.as_deref(),
            );
            let mut sel_samples: Vec<String> = app.selected_samples().iter().cloned().collect();
            sel_samples.sort();
            let mut sel_tags: Vec<String> = app.selected_tags().iter().cloned().collect();
            sel_tags.sort();
            let mut sel_scans: Vec<u32> = app.selected_scans().iter().cloned().collect();
            sel_scans.sort();
            config.set_selection_export(&sel_samples, &sel_tags, &sel_scans);
        }
        if let Err(e) = config.save() {
            eprintln!("{}", e.report());
        }

        std::process::exit(0);
    });

    #[cfg(feature = "catalog")]
    tui::preview::run_preview_window_on_first_path(
        preview_rx,
        Some(beamspot_tx),
        Some(preview_cmd_tx),
    );
    #[cfg(not(feature = "catalog"))]
    tui::preview::run_preview_window_on_first_path(preview_rx, None, None);
    std::process::exit(0);
}

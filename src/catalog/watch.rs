#![cfg(feature = "watch")]

use crate::catalog::ingest_beamtime;
use crate::catalog::CatalogError;
use crate::catalog::IngestSelection;
use notify_debouncer_mini::{new_debouncer, DebounceEventResult};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

pub const DEFAULT_DEBOUNCE_MS: u64 = 1500;
const DEFAULT_HEADER_KEYS: &[&str] = &[
    "DATE",
    "Beamline Energy",
    "Sample Theta",
    "CCD Theta",
    "Higher Order Suppressor",
    "EPU Polarization",
];

pub struct WatchHandle {
    stop_tx: Option<mpsc::Sender<()>>,
}

impl WatchHandle {
    pub fn stop(mut self) {
        let _ = self.stop_tx.take().map(|tx| tx.send(()));
    }
}

impl Drop for WatchHandle {
    fn drop(&mut self) {
        let _ = self.stop_tx.take().map(|tx| tx.send(()));
    }
}

pub fn run_catalog_watcher(
    beamtime_dir: &Path,
    header_items: &[String],
    debounce_ms: u64,
    on_ingest_start: Option<Box<dyn Fn() + Send>>,
    on_ingest_end: Option<Box<dyn Fn() + Send>>,
) -> Result<WatchHandle, crate::catalog::CatalogError> {
    let beamtime_dir = beamtime_dir.to_path_buf();
    let header_items = header_items.to_vec();
    let (stop_tx, stop_rx) = mpsc::channel();
    let debounce = if debounce_ms == 0 {
        DEFAULT_DEBOUNCE_MS
    } else {
        debounce_ms.max(100)
    };
    let on_start = on_ingest_start;
    let on_end = on_ingest_end;
    thread::spawn(move || {
        let (event_tx, event_rx) = mpsc::channel();
        let mut debouncer = match new_debouncer(
            Duration::from_millis(debounce),
            move |res: DebounceEventResult| {
                if let Ok(events) = res {
                    let has_fits = events.iter().any(|e| {
                        e.path
                            .extension()
                            .map(|e| e.eq_ignore_ascii_case("fits"))
                            .unwrap_or(false)
                    });
                    if has_fits {
                        let _ = event_tx.send(());
                    }
                }
            },
        ) {
            Ok(d) => d,
            Err(_) => return,
        };
        if debouncer
            .watcher()
            .watch(
                &beamtime_dir,
                notify_debouncer_mini::notify::RecursiveMode::Recursive,
            )
            .is_err()
        {
            return;
        }
        let keys = if header_items.is_empty() {
            DEFAULT_HEADER_KEYS
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
        } else {
            header_items
        };
        while stop_rx.try_recv().is_err() {
            match event_rx.recv_timeout(Duration::from_millis(500)) {
                Ok(()) => {
                    if let Some(ref f) = on_start {
                        f();
                    }
                    let _ = ingest_beamtime(
                        &beamtime_dir,
                        &keys,
                        true,
                        None,
                        crate::catalog::IngestSelection::default(),
                    );
                    if let Some(ref f) = on_end {
                        f();
                    }
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }
    });
    Ok(WatchHandle {
        stop_tx: Some(stop_tx),
    })
}

pub fn run_catalog_watcher_blocking(
    beamtime_dir: &Path,
    header_items: &[String],
    debounce_ms: u64,
    subpaths: &[PathBuf],
    cancel: Arc<AtomicBool>,
) -> Result<(), CatalogError> {
    if !beamtime_dir.is_dir() {
        return Err(CatalogError::Validation(format!(
            "beamtime_dir is not a directory: {}",
            beamtime_dir.display()
        )));
    }
    if cancel.load(Ordering::Relaxed) {
        return Ok(());
    }

    let beamtime_dir = beamtime_dir.canonicalize().map_err(CatalogError::Io)?;
    let watch_roots: Vec<PathBuf> = if subpaths.is_empty() {
        vec![beamtime_dir.clone()]
    } else {
        let mut v = Vec::with_capacity(subpaths.len());
        for rel in subpaths {
            let p = if rel.is_absolute() {
                rel.clone()
            } else {
                beamtime_dir.join(rel)
            };
            let p = p.canonicalize().map_err(CatalogError::Io)?;
            if !p.starts_with(&beamtime_dir) {
                return Err(CatalogError::Validation(format!(
                    "watch subpath escapes beamtime root: {}",
                    p.display()
                )));
            }
            if !p.is_dir() {
                return Err(CatalogError::Validation(format!(
                    "watch path is not a directory: {}",
                    p.display()
                )));
            }
            v.push(p);
        }
        v
    };

    let debounce = if debounce_ms == 0 {
        DEFAULT_DEBOUNCE_MS
    } else {
        debounce_ms.max(100)
    };

    let keys: Vec<String> = if header_items.is_empty() {
        DEFAULT_HEADER_KEYS
            .iter()
            .map(|s| (*s).to_string())
            .collect()
    } else {
        header_items.to_vec()
    };

    let (event_tx, event_rx) = mpsc::channel();
    let mut debouncer = new_debouncer(
        Duration::from_millis(debounce),
        move |res: DebounceEventResult| {
            if let Ok(events) = res {
                let has_fits = events.iter().any(|e| {
                    e.path
                        .extension()
                        .map(|e| e.eq_ignore_ascii_case("fits"))
                        .unwrap_or(false)
                });
                if has_fits {
                    let _ = event_tx.send(());
                }
            }
        },
    )
    .map_err(|e| CatalogError::Validation(format!("file watcher debouncer: {e}")))?;

    for root in &watch_roots {
        debouncer
            .watcher()
            .watch(
                root,
                notify_debouncer_mini::notify::RecursiveMode::Recursive,
            )
            .map_err(|e| CatalogError::Validation(format!("watch {}: {e}", root.display())))?;
    }

    while !cancel.load(Ordering::Relaxed) {
        match event_rx.recv_timeout(Duration::from_millis(200)) {
            Ok(()) => {
                let _ =
                    ingest_beamtime(&beamtime_dir, &keys, true, None, IngestSelection::default());
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    Ok(())
}

#![cfg(feature = "watch")]

use crate::catalog::ingest_beamtime;
use notify_debouncer_mini::{new_debouncer, DebounceEventResult};
use std::path::Path;
use std::sync::mpsc;
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
) -> Result<WatchHandle, crate::catalog::CatalogError> {
    let beamtime_dir = beamtime_dir.to_path_buf();
    let header_items = header_items.to_vec();
    let (stop_tx, stop_rx) = mpsc::channel();
    let debounce = if debounce_ms == 0 {
        DEFAULT_DEBOUNCE_MS
    } else {
        debounce_ms.max(100)
    };
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
            .watch(&beamtime_dir, notify_debouncer_mini::notify::RecursiveMode::Recursive)
            .is_err()
        {
            return;
        }
        let keys = if header_items.is_empty() {
            DEFAULT_HEADER_KEYS.iter().map(|s| s.to_string()).collect::<Vec<_>>()
        } else {
            header_items
        };
        while stop_rx.try_recv().is_err() {
            match event_rx.recv_timeout(Duration::from_millis(500)) {
                Ok(()) => {
                    let _ = ingest_beamtime(&beamtime_dir, &keys, true);
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

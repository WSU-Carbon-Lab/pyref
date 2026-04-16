//! End-to-end ingest test against a synthetic beamtime.
//!
//! Guarantees that a freshly built beamtime of 10 scans × 10 frames ingests
//! cleanly, populates the expected row counts in ``files``/``frames``/``scans``,
//! and emits interleaved ``catalog_row`` / ``file_complete`` progress events so
//! streaming hosts (TUI, Python tqdm) see incremental progress rather than a
//! single end-of-run burst.

#![cfg(all(feature = "catalog", feature = "parallel_ingest"))]

mod common;

use std::sync::{Arc, Mutex};
use std::time::Instant;

use diesel::dsl::count_star;
use diesel::prelude::*;
use pyref::catalog::{
    ingest_beamtime, ingest_beamtime_with_progress_sink, open_catalog_db, IngestParallelism,
    IngestProgress, IngestProgressSink,
};
use pyref::schema::{files, frames, scans};

use common::{build_synthetic_beamtime, SyntheticLayout};

fn header_items() -> Vec<String> {
    [
        "DATE",
        "Beamline Energy",
        "Sample Theta",
        "CCD Theta",
        "Higher Order Suppressor",
        "EPU Polarization",
    ]
    .iter()
    .map(|s| (*s).to_string())
    .collect()
}

static ENV_LOCK: Mutex<()> = Mutex::new(());

struct EnvGuard {
    keys: Vec<&'static str>,
    previous: Vec<Option<String>>,
}

impl EnvGuard {
    fn set(pairs: &[(&'static str, String)]) -> Self {
        let mut keys = Vec::with_capacity(pairs.len());
        let mut previous = Vec::with_capacity(pairs.len());
        for (k, v) in pairs {
            keys.push(*k);
            previous.push(std::env::var(k).ok());
            std::env::set_var(k, v);
        }
        Self { keys, previous }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        for (k, prev) in self.keys.iter().zip(self.previous.iter()) {
            match prev {
                Some(v) => std::env::set_var(k, v),
                None => std::env::remove_var(k),
            }
        }
    }
}

#[test]
fn synthetic_beamtime_ingests_with_expected_row_counts() {
    const SCANS: usize = 10;
    const FRAMES: usize = 10;
    const EXPECTED_FILES: i64 = (SCANS * FRAMES) as i64;

    let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let tmp = tempfile::tempdir().expect("tempdir");
    let layout = SyntheticLayout::uniform(SCANS, FRAMES, 16, 16);
    let beamtime = build_synthetic_beamtime(layout, tmp.path()).expect("build synthetic beamtime");

    let catalog_db = tmp.path().join("catalog.db");
    let cache_root = tmp.path().join("cache");
    let _env = EnvGuard::set(&[
        ("PYREF_CATALOG_DB", catalog_db.display().to_string()),
        ("PYREF_CACHE_ROOT", cache_root.display().to_string()),
    ]);

    let items = header_items();
    let returned = ingest_beamtime(&beamtime.root, &items, false, None).expect("ingest_beamtime");
    assert_eq!(
        returned, catalog_db,
        "ingest should return the PYREF_CATALOG_DB path"
    );
    assert!(returned.exists(), "catalog.db should be created on disk");

    let mut conn = open_catalog_db(&returned).expect("open catalog");
    let files_count: i64 = files::table
        .select(count_star())
        .first(&mut conn)
        .expect("count files");
    let frames_count: i64 = frames::table
        .select(count_star())
        .first(&mut conn)
        .expect("count frames");
    let scans_count: i64 = scans::table
        .select(count_star())
        .first(&mut conn)
        .expect("count scans");

    assert_eq!(files_count, EXPECTED_FILES, "files row count");
    assert_eq!(frames_count, EXPECTED_FILES, "frames row count");
    assert_eq!(scans_count, SCANS as i64, "scans row count");
}

#[test]
fn synthetic_beamtime_streams_progress_incrementally() {
    const SCANS: usize = 4;
    const FRAMES: usize = 4;
    const EXPECTED_FILES: usize = SCANS * FRAMES;

    let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let tmp = tempfile::tempdir().expect("tempdir");
    let layout = SyntheticLayout::uniform(SCANS, FRAMES, 16, 16);
    let beamtime = build_synthetic_beamtime(layout, tmp.path()).expect("build synthetic beamtime");

    let catalog_db = tmp.path().join("catalog.db");
    let cache_root = tmp.path().join("cache");
    let _env = EnvGuard::set(&[
        ("PYREF_CATALOG_DB", catalog_db.display().to_string()),
        ("PYREF_CACHE_ROOT", cache_root.display().to_string()),
    ]);

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum Kind {
        CatalogRow,
        FileComplete,
    }
    let events: Arc<Mutex<Vec<(Kind, Instant)>>> = Arc::new(Mutex::new(Vec::new()));
    let sink_events = Arc::clone(&events);

    let progress = IngestProgressSink::from_callback(move |ev| match ev {
        IngestProgress::CatalogRow { .. } => {
            let mut g = sink_events.lock().unwrap_or_else(|e| e.into_inner());
            g.push((Kind::CatalogRow, Instant::now()));
        }
        IngestProgress::FileComplete { .. } => {
            let mut g = sink_events.lock().unwrap_or_else(|e| e.into_inner());
            g.push((Kind::FileComplete, Instant::now()));
        }
        _ => {}
    });

    let items = header_items();
    ingest_beamtime_with_progress_sink(
        &beamtime.root,
        &items,
        false,
        Some(progress),
        IngestParallelism::default(),
    )
    .expect("ingest_beamtime_with_progress_sink");

    let events = events.lock().unwrap_or_else(|e| e.into_inner());
    let catalog_events: Vec<&(Kind, Instant)> = events
        .iter()
        .filter(|(k, _)| *k == Kind::CatalogRow)
        .collect();
    let file_events: Vec<&(Kind, Instant)> = events
        .iter()
        .filter(|(k, _)| *k == Kind::FileComplete)
        .collect();

    assert_eq!(
        catalog_events.len(),
        EXPECTED_FILES,
        "one catalog_row event per file"
    );
    assert_eq!(
        file_events.len(),
        EXPECTED_FILES,
        "one file_complete event per file"
    );

    let first_catalog_idx = events
        .iter()
        .position(|(k, _)| *k == Kind::CatalogRow)
        .expect("at least one catalog_row event");
    let first_file_idx = events
        .iter()
        .position(|(k, _)| *k == Kind::FileComplete)
        .expect("at least one file_complete event");
    assert!(
        first_catalog_idx < first_file_idx,
        "catalog_row events should precede the first file_complete event \
         (got catalog_row at {first_catalog_idx} and file_complete at {first_file_idx})",
    );
}

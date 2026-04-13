//! Structured ingest progress for UIs (TUI, Python tqdm/rich).
//!
//! [`BeamtimeIngestLayout`] is derived from discovered FITS paths and filename stems only (no FITS
//! I/O). [`IngestProgressSink`] forwards [`IngestProgress::Layout`] and per-file completion events
//! to an optional callback while preserving legacy ``(current, total)`` channel updates during the
//! catalog insert loop for the TUI.

use std::collections::BTreeMap;
use std::path::Path;
use std::path::PathBuf;
use std::sync::mpsc;

use crate::io::parse_fits_stem;

/// Scan identifier and how many FITS files fall under that stem-derived scan in the beamtime.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScanFileCount {
    pub scan_number: i32,
    pub file_count: usize,
}

/// Summary used to size global and per-scan progress bars before ingest runs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BeamtimeIngestLayout {
    pub total_files: usize,
    pub scans: Vec<ScanFileCount>,
}

/// One progress event during beamtime ingest.
#[derive(Clone, Debug)]
pub enum IngestProgress {
    /// Emitted once after discovery; ``scans`` are ordered by ``scan_number`` (unparseable stems
    /// contribute to scan ``0``).
    Layout {
        total_files: u32,
        scans: Vec<(i32, u32)>,
    },
    /// Optional coarse phase label for hosts that refresh banners (``headers``, ``catalog``,
    /// ``zarr``).
    Phase { name: String },
    /// Emitted after a frame's pixels are written to zarr; use for nested per-scan bars and a
    /// global ``fully processed`` counter.
    FileComplete {
        scan_number: i32,
        scan_done: u32,
        scan_total: u32,
        global_done: u32,
        global_total: u32,
    },
}

/// Forwards progress to an optional legacy channel and/or a callback.
pub struct IngestProgressSink {
    pub legacy: Option<mpsc::Sender<(u32, u32)>>,
    pub on_event: Option<Box<dyn Fn(IngestProgress) + Send>>,
}

impl IngestProgressSink {
    pub fn from_channel(tx: mpsc::Sender<(u32, u32)>) -> Self {
        Self {
            legacy: Some(tx),
            on_event: None,
        }
    }

    pub fn from_callback<F: Fn(IngestProgress) + Send + 'static>(f: F) -> Self {
        Self {
            legacy: None,
            on_event: Some(Box::new(f)),
        }
    }

    pub fn channel_and_callback<F: Fn(IngestProgress) + Send + 'static>(
        tx: mpsc::Sender<(u32, u32)>,
        f: F,
    ) -> Self {
        Self {
            legacy: Some(tx),
            on_event: Some(Box::new(f)),
        }
    }

    pub fn emit(&self, ev: IngestProgress) {
        if let IngestProgress::Layout { total_files, .. } = &ev {
            if let Some(tx) = &self.legacy {
                let _ = tx.send((0, *total_files));
            }
        }
        if let Some(cb) = &self.on_event {
            cb(ev);
        }
    }

    /// Legacy ``(current, total)`` updates during the catalog insert loop (TUI progress bar).
    pub fn legacy_catalog_row(&self, current: u32, total: u32) {
        if let Some(tx) = &self.legacy {
            let _ = tx.send((current, total));
        }
    }
}

fn scan_number_for_path(path: &Path) -> i32 {
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    parse_fits_stem(stem)
        .map(|p| p.scan_number as i32)
        .unwrap_or(0)
}

/// Groups FITS paths by stem-derived scan number; keys iterate in ascending order.
pub(crate) fn partition_paths_by_scan(paths: &[PathBuf]) -> Vec<(i32, Vec<PathBuf>)> {
    let mut m: BTreeMap<i32, Vec<PathBuf>> = BTreeMap::new();
    for p in paths {
        let sn = scan_number_for_path(p);
        m.entry(sn).or_default().push(p.clone());
    }
    m.into_iter().collect()
}

/// Builds layout and scan groups from a path list (same partitioning ingest uses for parallel scans).
pub(crate) fn layout_and_groups_from_paths(paths: &[PathBuf]) -> (BeamtimeIngestLayout, Vec<(i32, Vec<PathBuf>)>) {
    let groups = partition_paths_by_scan(paths);
    let scans: Vec<ScanFileCount> = groups
        .iter()
        .map(|(sn, v)| ScanFileCount {
            scan_number: *sn,
            file_count: v.len(),
        })
        .collect();
    let layout = BeamtimeIngestLayout {
        total_files: paths.len(),
        scans,
    };
    (layout, groups)
}

#![cfg(feature = "parallel_ingest")]

use crate::catalog::ingest::{
    ensure_bt_beamtime, prune_bt_scan_points, prune_missing_files, upsert_bt_batch,
    upsert_bt_batch_rows, upsert_files_batch, resolve_ingest_target,
};
use crate::io::BtIngestRow;
use crate::catalog::open_or_create_db;
use crate::catalog::Result;
use crate::catalog::CatalogError;
use crate::io::options::ReadFitsOptions;
use crate::loader::{read_fits_metadata_batch, read_multiple_fits_headers_only_rows};
use polars::prelude::DataFrame;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;

const BATCH_SIZE: usize = 500;
const CHANNEL_BOUND: usize = 2;

enum PipelineMsg {
    Batch(DataFrame),
    Rows(Vec<BtIngestRow>),
    Error(String),
    Done,
}

pub fn ingest_beamtime_pipelined(
    beamtime_dir: &std::path::Path,
    header_items: &[String],
    incremental: bool,
    progress_tx: Option<mpsc::Sender<(u32, u32)>>,
) -> Result<std::path::PathBuf> {
    let db_path = super::resolve_catalog_path(beamtime_dir);
    let conn = open_or_create_db(beamtime_dir)?;
    let beamtime_id = ensure_bt_beamtime(&conn, beamtime_dir)?;
    let discovered = super::discover_fits_paths_parallel(beamtime_dir)?;
    let path_to_mtime: HashMap<String, i64> = discovered
        .iter()
        .map(|(p, m)| (p.to_string_lossy().to_string(), *m))
        .collect();
    let paths: Vec<PathBuf> = discovered.into_iter().map(|(p, _)| p).collect();

    let use_normalized_only = super::is_new_catalog_layout(beamtime_dir);
    let to_ingest: Vec<PathBuf> = if incremental {
        if use_normalized_only {
            let existing: HashMap<String, i64> = conn
                .prepare(
                    "SELECT source_path, source_mtime FROM bt_scan_points WHERE source_path IS NOT NULL",
                )?
                .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1).unwrap_or(0))))?
                .filter_map(|r| r.ok())
                .collect();
            paths
                .into_iter()
                .filter(|p| {
                    let key = p.to_string_lossy().to_string();
                    let mtime = path_to_mtime.get(&key).copied().unwrap_or(0);
                    existing
                        .get(&key)
                        .is_none_or(|&stored| stored == 0 || mtime > stored)
                })
                .collect()
        } else {
            let existing: HashMap<String, i64> = conn
                .prepare("SELECT path, mtime FROM files")?
                .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))?
                .filter_map(|r| r.ok())
                .collect();
            paths
                .into_iter()
                .filter(|p| {
                    let key = p.to_string_lossy().to_string();
                    let mtime = path_to_mtime.get(&key).copied().unwrap_or(0);
                    existing.get(&key).is_none_or(|&stored| mtime > stored)
                })
                .collect()
        }
    } else {
        paths
    };

    let total = to_ingest.len() as u32;
    if total == 0 {
        let path_list: Vec<&str> = path_to_mtime.keys().map(|s| s.as_str()).collect();
        if !use_normalized_only {
            prune_missing_files(&conn, &path_list)?;
        }
        prune_bt_scan_points(&conn, &path_list)?;
        return Ok(db_path);
    }

    let opts = ReadFitsOptions {
        header_items: header_items.to_vec(),
        batch_size: BATCH_SIZE,
        ..ReadFitsOptions::default()
    };

    let paths_arc = Arc::new(to_ingest);
    let (tx, rx) = mpsc::sync_channel::<PipelineMsg>(CHANNEL_BOUND);

    let paths_arc_reader = Arc::clone(&paths_arc);
    let opts_reader = opts.clone();
    let header_items_owned = header_items.to_vec();
    let use_rows_only = use_normalized_only;
    let reader = thread::spawn(move || {
        let len = paths_arc_reader.len();
        let mut start = 0usize;
        while start < len {
            let end = (start + BATCH_SIZE).min(len);
            let chunk_vec = paths_arc_reader[start..end].to_vec();
            let chunk_len = chunk_vec.len();
            if use_rows_only {
                match read_multiple_fits_headers_only_rows(chunk_vec, &header_items_owned) {
                    Ok(rows) => {
                        if tx.send(PipelineMsg::Rows(rows)).is_err() {
                            return;
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(PipelineMsg::Error(e.to_string()));
                        return;
                    }
                }
            } else {
                match read_fits_metadata_batch(chunk_vec, &opts_reader) {
                    Ok(df) => {
                        if tx.send(PipelineMsg::Batch(df)).is_err() {
                            return;
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(PipelineMsg::Error(e.to_string()));
                        return;
                    }
                }
            }
            start += chunk_len;
        }
        let _ = tx.send(PipelineMsg::Done);
    });

    let mut processed: u32 = 0;
    loop {
        match rx.recv() {
            Ok(PipelineMsg::Rows(rows)) => {
                if !rows.is_empty() {
                    upsert_bt_batch_rows(&conn, beamtime_id, &rows, &path_to_mtime)?;
                    processed = (processed as usize + rows.len()).min(paths_arc.len()) as u32;
                    if let Some(ref ptx) = progress_tx {
                        let _ = ptx.send((processed, total));
                    }
                }
            }
            Ok(PipelineMsg::Batch(df)) => {
                let n = df.height();
                if n == 0 {
                    continue;
                }
                if !use_normalized_only {
                    upsert_files_batch(&conn, &df, &path_to_mtime)?;
                }
                upsert_bt_batch(&conn, beamtime_id, &df, &path_to_mtime)?;
                processed = (processed as usize + n).min(paths_arc.len()) as u32;
                if let Some(ref ptx) = progress_tx {
                    let _ = ptx.send((processed, total));
                }
            }
            Ok(PipelineMsg::Error(e)) => {
                let _ = reader.join();
                return Err(CatalogError::Validation(format!(
                    "read_fits_metadata_batch: {}",
                    e
                )));
            }
            Ok(PipelineMsg::Done) => break,
            Err(_) => {
                let _ = reader.join();
                return Err(CatalogError::Validation(
                    "ingest pipeline reader disconnected".into(),
                ));
            }
        }
    }
    let _ = reader.join();

    let path_list: Vec<&str> = path_to_mtime.keys().map(|s| s.as_str()).collect();
    if !use_normalized_only {
        prune_missing_files(&conn, &path_list)?;
    }
    prune_bt_scan_points(&conn, &path_list)?;

    Ok(db_path)
}

pub fn ingest_beamtime_pipelined_with_context(
    beamtime_dir: &std::path::Path,
    data_root: Option<&std::path::Path>,
    experimentalist: Option<&str>,
    header_items: &[String],
    incremental: bool,
    progress_tx: Option<mpsc::Sender<(u32, u32)>>,
    cancel: Option<Arc<std::sync::atomic::AtomicBool>>,
) -> Result<std::path::PathBuf> {
    let db_path = resolve_ingest_target(beamtime_dir, data_root);
    let conn = open_or_create_db(beamtime_dir)?;
    let beamtime_id = ensure_bt_beamtime(&conn, beamtime_dir)?;

    use crate::catalog::ingest::upsert_beamtime_record as upsert_record;
    upsert_record(&conn, beamtime_dir, data_root, experimentalist)?;

    let discovered = super::discover_fits_paths_parallel(beamtime_dir)?;
    let path_to_mtime: HashMap<String, i64> = discovered
        .iter()
        .map(|(p, m)| (p.to_string_lossy().to_string(), *m))
        .collect();
    let paths: Vec<PathBuf> = discovered.into_iter().map(|(p, _)| p).collect();

    let use_normalized_only = super::is_new_catalog_layout(beamtime_dir);
    let to_ingest: Vec<PathBuf> = if incremental {
        if use_normalized_only {
            let existing: HashMap<String, i64> = conn
                .prepare(
                    "SELECT source_path, source_mtime FROM bt_scan_points WHERE source_path IS NOT NULL",
                )?
                .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1).unwrap_or(0))))?
                .filter_map(|r| r.ok())
                .collect();
            paths
                .into_iter()
                .filter(|p| {
                    let key = p.to_string_lossy().to_string();
                    let mtime = path_to_mtime.get(&key).copied().unwrap_or(0);
                    existing
                        .get(&key)
                        .is_none_or(|&stored| stored == 0 || mtime > stored)
                })
                .collect()
        } else {
            let existing: HashMap<String, i64> = conn
                .prepare("SELECT path, mtime FROM files")?
                .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))?
                .filter_map(|r| r.ok())
                .collect();
            paths
                .into_iter()
                .filter(|p| {
                    let key = p.to_string_lossy().to_string();
                    let mtime = path_to_mtime.get(&key).copied().unwrap_or(0);
                    existing.get(&key).is_none_or(|&stored| mtime > stored)
                })
                .collect()
        }
    } else {
        paths
    };

    let total = to_ingest.len() as u32;
    if total == 0 {
        let path_list: Vec<&str> = path_to_mtime.keys().map(|s| s.as_str()).collect();
        if !use_normalized_only {
            prune_missing_files(&conn, &path_list)?;
        }
        prune_bt_scan_points(&conn, &path_list)?;
        return Ok(db_path);
    }

    let opts = ReadFitsOptions {
        header_items: header_items.to_vec(),
        batch_size: BATCH_SIZE,
        ..ReadFitsOptions::default()
    };

    let paths_arc = Arc::new(to_ingest);
    let (tx, rx) = mpsc::sync_channel::<PipelineMsg>(CHANNEL_BOUND);

    let paths_arc_reader = Arc::clone(&paths_arc);
    let opts_reader = opts.clone();
    let header_items_owned = header_items.to_vec();
    let use_rows_only = use_normalized_only;
    let cancel_clone = cancel.clone();
    let reader = thread::spawn(move || {
        let len = paths_arc_reader.len();
        let mut start = 0usize;
        while start < len {
            if cancel_clone.as_ref().map(|c| c.load(std::sync::atomic::Ordering::Relaxed)).unwrap_or(false) {
                break;
            }
            let end = (start + BATCH_SIZE).min(len);
            let chunk_vec = paths_arc_reader[start..end].to_vec();
            let chunk_len = chunk_vec.len();
            if use_rows_only {
                match read_multiple_fits_headers_only_rows(chunk_vec, &header_items_owned) {
                    Ok(rows) => {
                        if tx.send(PipelineMsg::Rows(rows)).is_err() {
                            return;
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(PipelineMsg::Error(e.to_string()));
                        return;
                    }
                }
            } else {
                match read_fits_metadata_batch(chunk_vec, &opts_reader) {
                    Ok(df) => {
                        if tx.send(PipelineMsg::Batch(df)).is_err() {
                            return;
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(PipelineMsg::Error(e.to_string()));
                        return;
                    }
                }
            }
            start += chunk_len;
        }
        let _ = tx.send(PipelineMsg::Done);
    });

    let mut processed: u32 = 0;
    loop {
        // Check cancel token before processing each batch
        if let Some(ref cancel) = cancel {
            if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                break;
            }
        }

        match rx.recv() {
            Ok(PipelineMsg::Rows(rows)) => {
                if !rows.is_empty() {
                    upsert_bt_batch_rows(&conn, beamtime_id, &rows, &path_to_mtime)?;
                    processed = (processed as usize + rows.len()).min(paths_arc.len()) as u32;
                    if let Some(ref ptx) = progress_tx {
                        let _ = ptx.send((processed, total));
                    }
                }
            }
            Ok(PipelineMsg::Batch(df)) => {
                let n = df.height();
                if n == 0 {
                    continue;
                }
                if !use_normalized_only {
                    upsert_files_batch(&conn, &df, &path_to_mtime)?;
                }
                upsert_bt_batch(&conn, beamtime_id, &df, &path_to_mtime)?;
                processed = (processed as usize + n).min(paths_arc.len()) as u32;
                if let Some(ref ptx) = progress_tx {
                    let _ = ptx.send((processed, total));
                }
            }
            Ok(PipelineMsg::Error(e)) => {
                let _ = reader.join();
                return Err(CatalogError::Validation(format!(
                    "read_fits_metadata_batch: {}",
                    e
                )));
            }
            Ok(PipelineMsg::Done) => break,
            Err(_) => {
                let _ = reader.join();
                return Err(CatalogError::Validation(
                    "ingest pipeline reader disconnected".into(),
                ));
            }
        }
    }
    let _ = reader.join();

    let path_list: Vec<&str> = path_to_mtime.keys().map(|s| s.as_str()).collect();
    if !use_normalized_only {
        prune_missing_files(&conn, &path_list)?;
    }
    prune_bt_scan_points(&conn, &path_list)?;

    Ok(db_path)
}

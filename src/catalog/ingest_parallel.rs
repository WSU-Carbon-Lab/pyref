#![cfg(feature = "parallel_ingest")]

use crate::catalog::ingest::{
    ensure_bt_beamtime, prune_bt_scan_points, prune_fits_files, resolve_ingest_target,
    update_beamtime_catalog_layout, upsert_beamtime_record, upsert_bt_batch_rows,
};
use crate::catalog::open_or_create_db_at;
use crate::catalog::discover_paths_for_catalog_ingest;
use crate::catalog::CatalogError;
use crate::catalog::Result;
use crate::io::BtIngestRow;
use crate::loader::read_multiple_fits_headers_only_rows;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;

const BATCH_SIZE: usize = 500;
const CHANNEL_BOUND: usize = 2;

enum PipelineMsg {
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
    ingest_beamtime_pipelined_with_context(
        beamtime_dir,
        None,
        None,
        header_items,
        incremental,
        progress_tx,
        None,
    )
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
    let conn = open_or_create_db_at(&db_path)?;
    let beamtime_id = ensure_bt_beamtime(&conn, beamtime_dir)?;
    upsert_beamtime_record(&conn, beamtime_dir, data_root, experimentalist)?;
    let (discovered, layout) = discover_paths_for_catalog_ingest(beamtime_dir)?;
    update_beamtime_catalog_layout(&conn, beamtime_id, layout)?;
    let path_to_mtime: HashMap<String, i64> = discovered
        .iter()
        .map(|(p, m)| (p.to_string_lossy().to_string(), *m))
        .collect();
    let paths: Vec<PathBuf> = discovered.into_iter().map(|(p, _)| p).collect();
    let to_ingest: Vec<PathBuf> = if incremental {
        let existing: HashMap<String, i64> = conn
            .prepare(
                "SELECT path, source_mtime FROM fits_files WHERE beamtime_id = ?1",
            )?
            .query_map(rusqlite::params![beamtime_id], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?))
            })?
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
        paths
    };
    let total = to_ingest.len() as u32;
    if total == 0 {
        let path_list: Vec<&str> = path_to_mtime.keys().map(|s| s.as_str()).collect();
        prune_fits_files(&conn, beamtime_id, &path_list)?;
        prune_bt_scan_points(&conn, &path_list)?;
        super::profile_persist::recompute_reflectivity_profiles_for_beamtime(&conn, beamtime_id)?;
        return Ok(db_path);
    }
    let paths_arc = Arc::new(to_ingest);
    let (tx, rx) = mpsc::sync_channel::<PipelineMsg>(CHANNEL_BOUND);
    let paths_arc_reader = Arc::clone(&paths_arc);
    let header_items_owned = header_items.to_vec();
    let cancel_clone = cancel.clone();
    let reader = thread::spawn(move || {
        let len = paths_arc_reader.len();
        let mut start = 0usize;
        while start < len {
            if cancel_clone
                .as_ref()
                .map(|c| c.load(std::sync::atomic::Ordering::Relaxed))
                .unwrap_or(false)
            {
                break;
            }
            let end = (start + BATCH_SIZE).min(len);
            let chunk_vec = paths_arc_reader[start..end].to_vec();
            let chunk_len = chunk_vec.len();
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
            start += chunk_len;
        }
        let _ = tx.send(PipelineMsg::Done);
    });
    let mut processed: u32 = 0;
    loop {
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
            Ok(PipelineMsg::Error(e)) => {
                let _ = reader.join();
                return Err(CatalogError::Validation(format!(
                    "read_multiple_fits_headers_only_rows: {}",
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
    prune_fits_files(&conn, beamtime_id, &path_list)?;
    prune_bt_scan_points(&conn, &path_list)?;
    super::profile_persist::recompute_reflectivity_profiles_for_beamtime(&conn, beamtime_id)?;
    Ok(db_path)
}

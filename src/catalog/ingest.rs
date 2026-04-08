#![cfg(feature = "catalog")]

use crate::catalog::{discover_paths_for_catalog_ingest, CatalogError, Result};
use crate::io::parse_fits_stem;
use crate::io::BtIngestRow;
#[cfg(not(feature = "parallel_ingest"))]
use crate::catalog::open_or_create_db;
use rusqlite::Error as RusqliteError;

pub const DEFAULT_INGEST_HEADER_ITEMS: &[&str] = &[
    "DATE",
    "Beamline Energy",
    "Sample Theta",
    "CCD Theta",
    "Higher Order Suppressor",
    "EPU Polarization",
    "EXPOSURE",
    "Sample Name",
    "Scan ID",
];
#[cfg(not(feature = "parallel_ingest"))]
use crate::io::options::ReadFitsOptions;
#[cfg(not(feature = "parallel_ingest"))]
use crate::loader::read_fits_metadata_batch;
use polars::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc;

#[cfg(not(feature = "parallel_ingest"))]
const BATCH_SIZE: usize = 500;

fn resolve_sample(conn: &rusqlite::Connection, name: &str) -> Result<String> {
    let name = name.trim();
    if name.is_empty() {
        return Ok(String::new());
    }
    let first = name.split('_').next().unwrap_or(name);
    let existing: Option<String> = match conn.query_row(
        "SELECT name FROM samples WHERE LOWER(TRIM(name)) = LOWER(?1) OR LOWER(TRIM(name)) = LOWER(?2)",
        rusqlite::params![name, first],
        |r| r.get(0),
    ) {
        Ok(v) => Some(v),
        Err(RusqliteError::QueryReturnedNoRows) => None,
        Err(e) => return Err(e.into()),
    };
    if let Some(n) = existing {
        return Ok(n);
    }
    conn.execute(
        "INSERT INTO samples (name, chemical_formula, created_at, updated_at) VALUES (?1, NULL, strftime('%s','now'), strftime('%s','now'))",
        rusqlite::params![name],
    )?;
    Ok(name.to_string())
}

fn resolve_tag(conn: &rusqlite::Connection, name: &str) -> Result<Option<String>> {
    let name = name.trim();
    if name.is_empty() {
        return Ok(None);
    }
    if crate::io::is_polarization_tag(name) {
        return Ok(None);
    }
    let slug = name.to_lowercase().replace(' ', "_");
    let existing: Option<String> = match conn.query_row(
        "SELECT name FROM tags WHERE LOWER(TRIM(name)) = LOWER(?1)",
        rusqlite::params![name],
        |r| r.get(0),
    ) {
        Ok(v) => Some(v),
        Err(RusqliteError::QueryReturnedNoRows) => None,
        Err(e) => return Err(e.into()),
    };
    if let Some(n) = existing {
        return Ok(Some(n));
    }
    conn.execute("INSERT INTO tags (name, slug) VALUES (?1, ?2)", rusqlite::params![name, slug])?;
    Ok(Some(name.to_string()))
}

fn ensure_sample_tag(
    conn: &rusqlite::Connection,
    sample_name: &str,
    tag_name: Option<&str>,
) -> Result<()> {
    let tag_name = match tag_name {
        Some(t) if !t.trim().is_empty() => t.trim(),
        _ => return Ok(()),
    };
    let sample_id: i64 = match conn.query_row(
        "SELECT id FROM samples WHERE LOWER(TRIM(name)) = LOWER(?1)",
        rusqlite::params![sample_name],
        |r| r.get(0),
    ) {
        Ok(v) => v,
        Err(RusqliteError::QueryReturnedNoRows) => return Ok(()),
        Err(e) => return Err(e.into()),
    };
    let tag_id: i64 = match conn.query_row(
        "SELECT id FROM tags WHERE LOWER(TRIM(name)) = LOWER(?1)",
        rusqlite::params![tag_name],
        |r| r.get(0),
    ) {
        Ok(v) => v,
        Err(RusqliteError::QueryReturnedNoRows) => return Ok(()),
        Err(e) => return Err(e.into()),
    };
    conn.execute(
        "INSERT OR IGNORE INTO sample_tags (sample_id, tag_id) VALUES (?1, ?2)",
        rusqlite::params![sample_id, tag_id],
    )?;
    Ok(())
}

pub fn ingest_beamtime(
    beamtime_dir: &Path,
    header_items: &[String],
    incremental: bool,
    progress_tx: Option<mpsc::Sender<(u32, u32)>>,
) -> Result<PathBuf> {
    ingest_beamtime_with_context(beamtime_dir, None, None, header_items, incremental, progress_tx, None)
}

pub fn ingest_beamtime_with_context(
    beamtime_dir: &Path,
    data_root: Option<&Path>,
    experimentalist: Option<&str>,
    header_items: &[String],
    incremental: bool,
    progress_tx: Option<mpsc::Sender<(u32, u32)>>,
    cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
) -> Result<PathBuf> {
    #[cfg(feature = "parallel_ingest")]
    {
        return super::ingest_parallel::ingest_beamtime_pipelined_with_context(
            beamtime_dir,
            data_root,
            experimentalist,
            header_items,
            incremental,
            progress_tx,
            cancel,
        );
    }
    #[cfg(not(feature = "parallel_ingest"))]
    {
        ingest_beamtime_sequential(beamtime_dir, data_root, experimentalist, header_items, incremental, progress_tx, cancel)
    }
}

#[cfg(not(feature = "parallel_ingest"))]
fn ingest_beamtime_sequential(
    beamtime_dir: &Path,
    data_root: Option<&Path>,
    experimentalist: Option<&str>,
    header_items: &[String],
    incremental: bool,
    progress_tx: Option<mpsc::Sender<(u32, u32)>>,
    cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
) -> Result<PathBuf> {
    let db_path = resolve_ingest_target(beamtime_dir, data_root);
    let conn = open_or_create_db(beamtime_dir)?;
    let beamtime_id = ensure_bt_beamtime(&conn, beamtime_dir)?;
    upsert_beamtime_record(&conn, beamtime_dir, data_root, experimentalist)?;
    let discovered = super::discover_fits_paths(beamtime_dir)?;
    let path_to_mtime: HashMap<String, i64> = discovered
        .iter()
        .map(|(p, m)| (p.to_string_lossy().to_string(), *m))
        .collect();
    let paths: Vec<PathBuf> = discovered.into_iter().map(|(p, _)| p).collect();

    let use_normalized_only = super::is_new_catalog_layout(beamtime_dir);
    let to_ingest: Vec<PathBuf> = if incremental {
        if use_normalized_only {
            let existing: HashMap<String, i64> = conn
                .prepare("SELECT source_path, source_mtime FROM bt_scan_points WHERE source_path IS NOT NULL")?
                .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1).unwrap_or(0))))?
                .filter_map(|r| r.ok())
                .collect();
            paths
                .into_iter()
                .filter(|p| {
                    let key = p.to_string_lossy().to_string();
                    let mtime = path_to_mtime.get(&key).copied().unwrap_or(0);
                    existing.get(&key).is_none_or(|&stored| stored == 0 || mtime > stored)
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
    let mut processed: u32 = 0;
    let opts = ReadFitsOptions {
        header_items: header_items.to_vec(),
        batch_size: BATCH_SIZE,
        ..ReadFitsOptions::default()
    };
    for chunk in to_ingest.chunks(BATCH_SIZE) {
        if cancel.as_ref().map(|c| c.load(std::sync::atomic::Ordering::Relaxed)).unwrap_or(false) {
            break;
        }
        let chunk_vec: Vec<PathBuf> = chunk.to_vec();
        let df = read_fits_metadata_batch(chunk_vec, &opts)?;
        if !use_normalized_only {
            upsert_files_batch(&conn, &df, &path_to_mtime)?;
        }
        upsert_bt_batch(&conn, beamtime_id, &df, &path_to_mtime)?;
        processed = (processed as usize + chunk.len()).min(to_ingest.len()) as u32;
        if let Some(ref tx) = progress_tx {
            let _ = tx.send((processed, total));
        }
    }

    let path_list: Vec<&str> = path_to_mtime.keys().map(|s| s.as_str()).collect();
    if !use_normalized_only {
        prune_missing_files(&conn, &path_list)?;
    }
    prune_bt_scan_points(&conn, &path_list)?;
    super::profile_persist::recompute_reflectivity_profiles_for_beamtime(&conn, beamtime_id)?;

    Ok(db_path)
}

pub(crate) fn upsert_files_batch(
    conn: &rusqlite::Connection,
    df: &DataFrame,
    path_to_mtime: &HashMap<String, i64>,
) -> Result<()> {
    let n = df.height();
    if n == 0 {
        return Ok(());
    }
    let file_path_col = df
        .column("file_path")
        .map_err(|e| CatalogError::Validation(e.to_string()))?
        .str()
        .map_err(|e| CatalogError::Validation(e.to_string()))?;
    let get_str = |name: &str| -> Result<Vec<Option<String>>> {
        match df.column(name) {
            Ok(c) => Ok(c
                .str()
                .map_err(|e| CatalogError::Validation(e.to_string()))?
                .iter()
                .map(|s| s.map(|v| v.to_string()))
                .collect()),
            _ => Ok(std::iter::repeat_n(None, n).collect()),
        }
    };
    let get_i64 = |name: &str| -> Result<Vec<Option<i64>>> {
        match df.column(name) {
            Ok(c) => Ok(c
                .i64()
                .map_err(|e| CatalogError::Validation(e.to_string()))?
                .iter()
                .collect()),
            _ => Ok(std::iter::repeat_n(None, n).collect()),
        }
    };
    let get_f64 = |name: &str| -> Result<Vec<Option<f64>>> {
        match df.column(name) {
            Ok(c) => Ok(c
                .f64()
                .map_err(|e| CatalogError::Validation(e.to_string()))?
                .iter()
                .collect()),
            _ => Ok(std::iter::repeat_n(None, n).collect()),
        }
    };

    let data_offset = get_i64("data_offset")?;
    let naxis1 = get_i64("naxis1")?;
    let naxis2 = get_i64("naxis2")?;
    let bitpix = get_i64("bitpix")?;
    let bzero = get_i64("bzero")?;
    let data_size = get_i64("data_size")?;
    let file_name = get_str("file_name")?;
    let sample_name = get_str("sample_name")?;
    let tag = get_str("tag")?;
    let scan_number = get_i64("scan_number")?;
    let frame_number = get_i64("frame_number")?;
    let date = get_str("DATE")?;
    let beamline_energy = get_f64("Beamline Energy")?;
    let sample_theta = get_f64("Sample Theta")?;
    let ccd_theta = get_f64("CCD Theta")?;
    let hos = get_f64("Higher Order Suppressor")?;
    let epu = get_f64("EPU Polarization")?;
    let exposure = get_f64("EXPOSURE")?;
    let sample_name_h = get_str("Sample Name")?;
    let scan_id = get_f64("Scan ID")?;
    let lambda = get_f64("Lambda")?;
    let q = get_f64("Q")?;

    let mut stmt = conn.prepare_cached(
        r#"
        INSERT OR REPLACE INTO files (
            path, mtime, file_path, data_offset, naxis1, naxis2, bitpix, bzero, data_size,
            file_name, sample_name, tag, scan_number, frame_number,
            "DATE", "Beamline Energy", "Sample Theta", "CCD Theta", "Higher Order Suppressor",
            "EPU Polarization", EXPOSURE, "Sample Name", "Scan ID", Lambda, Q
        ) VALUES (
            ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14,
            ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?24, ?25
        )"#,
    )?;

    let mut canonical_cache: HashMap<(String, Option<String>), (String, Option<String>)> =
        HashMap::new();
    for i in 0..n {
        let sn = sample_name
            .get(i)
            .cloned()
            .flatten()
            .unwrap_or_default();
        let t = tag.get(i).cloned().flatten();
        let key = (sn.clone(), t.clone());
        let (canonical_sn, canonical_t) = if let Some(cached) = canonical_cache.get(&key) {
            cached.clone()
        } else {
            let cs = resolve_sample(conn, &sn)?;
            let ct = t
                .as_ref()
                .map(|x| x.as_str())
                .and_then(|x| resolve_tag(conn, x).ok().flatten());
            ensure_sample_tag(conn, &cs, ct.as_deref())?;
            canonical_cache.insert(key, (cs.clone(), ct.clone()));
            (cs, ct)
        };

        let path_str = file_path_col.get(i).unwrap_or("").to_string();
        let mtime = path_to_mtime.get(&path_str).copied().unwrap_or(0);
        stmt.execute(rusqlite::params![
            path_str,
            mtime,
            path_str,
            data_offset.get(i).copied().flatten().unwrap_or(0),
            naxis1.get(i).copied().flatten().unwrap_or(0),
            naxis2.get(i).copied().flatten().unwrap_or(0),
            bitpix.get(i).copied().flatten().unwrap_or(0),
            bzero.get(i).copied().flatten().unwrap_or(0),
            data_size.get(i).copied().flatten().unwrap_or(0),
            file_name.get(i).cloned().unwrap_or_default(),
            canonical_sn,
            canonical_t,
            scan_number.get(i).copied().flatten().unwrap_or(0),
            frame_number.get(i).copied().flatten().unwrap_or(0),
            date.get(i).clone(),
            beamline_energy.get(i).copied().flatten(),
            sample_theta.get(i).copied().flatten(),
            ccd_theta.get(i).copied().flatten(),
            hos.get(i).copied().flatten(),
            epu.get(i).copied().flatten(),
            exposure.get(i).copied().flatten(),
            sample_name_h.get(i).clone(),
            scan_id.get(i).copied().flatten(),
            lambda.get(i).copied().flatten(),
            q.get(i).copied().flatten(),
        ])?;
    }
    Ok(())
}

pub(crate) fn prune_missing_files(conn: &rusqlite::Connection, known_paths: &[&str]) -> Result<()> {
    if known_paths.is_empty() {
        conn.execute("DELETE FROM files", [])?;
        return Ok(());
    }
    let placeholders = known_paths
        .iter()
        .enumerate()
        .map(|(i, _)| format!("?{}", i + 1))
        .collect::<Vec<_>>()
        .join(",");
    let sql = format!("DELETE FROM files WHERE path NOT IN ({})", placeholders);
    conn.execute(&sql, rusqlite::params_from_iter(known_paths.iter()))?;
    Ok(())
}

pub fn resolve_ingest_target(beamtime_dir: &Path, data_root: Option<&Path>) -> PathBuf {
    match data_root {
        Some(root) => root.join(".pyref").join("catalog.db"),
        None => super::resolve_catalog_path(beamtime_dir),
    }
}

pub(crate) fn upsert_beamtime_record(
    conn: &rusqlite::Connection,
    beamtime_dir: &Path,
    data_root: Option<&Path>,
    experimentalist: Option<&str>,
) -> Result<()> {
    let path_str = beamtime_dir.to_string_lossy().to_string();
    let data_root_str = data_root.and_then(|p| p.to_str()).unwrap_or("");
    let expt_str = experimentalist.unwrap_or("");
    conn.execute(
        "INSERT INTO bt_beamtimes (beamtime_path, data_root, experimentalist, last_indexed_at)
         VALUES (?1, ?2, ?3, strftime('%s','now'))
         ON CONFLICT(beamtime_path) DO UPDATE SET
             data_root = excluded.data_root,
             experimentalist = excluded.experimentalist,
             last_indexed_at = excluded.last_indexed_at",
        rusqlite::params![path_str, data_root_str, expt_str],
    )?;
    Ok(())
}

pub(crate) fn ensure_bt_beamtime(conn: &rusqlite::Connection, beamtime_dir: &Path) -> Result<i64> {
    let path_str = beamtime_dir.to_string_lossy().to_string();
    conn.execute(
        "INSERT OR IGNORE INTO bt_beamtimes (beamtime_path) VALUES (?1)",
        rusqlite::params![path_str],
    )?;
    let id: i64 = conn.query_row(
        "SELECT id FROM bt_beamtimes WHERE beamtime_path = ?1",
        rusqlite::params![path_str],
        |r| r.get(0),
    )?;
    Ok(id)
}

fn ensure_bt_sample(
    conn: &rusqlite::Connection,
    beamtime_id: i64,
    name: &str,
    tag: Option<&str>,
) -> Result<i64> {
    conn.execute(
        "INSERT OR IGNORE INTO bt_samples (beamtime_id, name, tag) VALUES (?1, ?2, ?3)",
        rusqlite::params![beamtime_id, name, tag],
    )?;
    let id: i64 = conn.query_row(
        "SELECT id FROM bt_samples WHERE beamtime_id = ?1 AND name = ?2 AND (tag IS ?3 OR (tag IS NULL AND ?3 IS NULL))",
        rusqlite::params![beamtime_id, name, tag],
        |r| r.get(0),
    )?;
    Ok(id)
}

fn ensure_bt_scan(
    conn: &rusqlite::Connection,
    beamtime_id: i64,
    sample_id: i64,
    scan_number: i64,
) -> Result<String> {
    let uid = format!("s_{}_{}", beamtime_id, scan_number);
    conn.execute(
        "INSERT OR REPLACE INTO bt_scans (uid, beamtime_id, sample_id) VALUES (?1, ?2, ?3)",
        rusqlite::params![uid, beamtime_id, sample_id],
    )?;
    Ok(uid)
}

fn ensure_bt_stream(
    conn: &rusqlite::Connection,
    beamtime_id: i64,
    scan_number: i64,
) -> Result<String> {
    let scan_uid = format!("s_{}_{}", beamtime_id, scan_number);
    let stream_uid = format!("st_{}_{}", beamtime_id, scan_number);
    conn.execute(
        "INSERT OR IGNORE INTO bt_streams (uid, scan_uid, name) VALUES (?1, ?2, 'primary')",
        rusqlite::params![stream_uid, scan_uid],
    )?;
    Ok(stream_uid)
}

pub(crate) fn upsert_bt_batch(
    conn: &rusqlite::Connection,
    beamtime_id: i64,
    df: &DataFrame,
    path_to_mtime: &HashMap<String, i64>,
) -> Result<()> {
    let n = df.height();
    if n == 0 {
        return Ok(());
    }
    let file_path_col = df
        .column("file_path")
        .map_err(|e| CatalogError::Validation(e.to_string()))?
        .str()
        .map_err(|e| CatalogError::Validation(e.to_string()))?;
    let get_str = |name: &str| -> Result<Vec<Option<String>>> {
        match df.column(name) {
            Ok(c) => Ok(c
                .str()
                .map_err(|e| CatalogError::Validation(e.to_string()))?
                .iter()
                .map(|s| s.map(|v| v.to_string()))
                .collect()),
            _ => Ok(std::iter::repeat_n(None, n).collect()),
        }
    };
    let get_i64 = |name: &str| -> Result<Vec<Option<i64>>> {
        match df.column(name) {
            Ok(c) => Ok(c
                .i64()
                .map_err(|e| CatalogError::Validation(e.to_string()))?
                .iter()
                .collect()),
            _ => Ok(std::iter::repeat_n(None, n).collect()),
        }
    };
    let get_f64 = |name: &str| -> Result<Vec<Option<f64>>> {
        match df.column(name) {
            Ok(c) => Ok(c
                .f64()
                .map_err(|e| CatalogError::Validation(e.to_string()))?
                .iter()
                .collect()),
            _ => Ok(std::iter::repeat_n(None, n).collect()),
        }
    };

    let data_offset = get_i64("data_offset")?;
    let naxis1 = get_i64("naxis1")?;
    let naxis2 = get_i64("naxis2")?;
    let bitpix = get_i64("bitpix")?;
    let bzero = get_i64("bzero")?;
    let _file_name = get_str("file_name")?;
    let sample_name = get_str("sample_name")?;
    let tag = get_str("tag")?;
    let scan_number = get_i64("scan_number")?;
    let frame_number = get_i64("frame_number")?;
    let beamline_energy = get_f64("Beamline Energy")?;
    let sample_theta = get_f64("Sample Theta")?;
    let ccd_theta = get_f64("CCD Theta")?;
    let epu = get_f64("EPU Polarization")?;
    let exposure = get_f64("EXPOSURE")?;

    let mut stmt = conn.prepare_cached(
        r#"
        INSERT OR REPLACE INTO bt_scan_points (
            uid, stream_uid, scan_uid, sample_id, seq_index, time,
            exposure, beamline_energy, epu_polarization, sample_theta, ccd_theta,
            source_path, source_data_offset, source_naxis1, source_naxis2,
            source_bitpix, source_bzero, source_mtime, beam_row, beam_col, beam_sigma
        ) VALUES (
            ?1, ?2, ?3, ?4, ?5, 0,
            ?6, ?7, ?8, ?9, ?10,
            ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20
        )"#,
    )?;

    let mut canonical_cache: HashMap<(String, Option<String>), i64> = HashMap::new();
    for i in 0..n {
        let sn = sample_name
            .get(i)
            .cloned()
            .flatten()
            .unwrap_or_default();
        let t = tag.get(i).cloned().flatten();
        let key = (sn.clone(), t.clone());
        let sample_id = if let Some(&sid) = canonical_cache.get(&key) {
            sid
        } else {
            let cs = resolve_sample(conn, &sn)?;
            let ct = t
                .as_ref()
                .map(|x| x.as_str())
                .and_then(|x| resolve_tag(conn, x).ok().flatten());
            ensure_sample_tag(conn, &cs, ct.as_deref())?;
            let sid = ensure_bt_sample(conn, beamtime_id, &cs, ct.as_deref())?;
            canonical_cache.insert(key, sid);
            sid
        };

        let scan_no = scan_number.get(i).copied().flatten().unwrap_or(0);
        let frame_no = frame_number.get(i).copied().flatten().unwrap_or(0);
        let scan_uid = format!("s_{}_{}", beamtime_id, scan_no);
        let stream_uid = format!("st_{}_{}", beamtime_id, scan_no);
        let point_uid = format!("sp_{}_{}_{}", beamtime_id, scan_no, frame_no);

        let _ = ensure_bt_scan(conn, beamtime_id, sample_id, scan_no)?;
        let _ = ensure_bt_stream(conn, beamtime_id, scan_no)?;

        let path_str = file_path_col.get(i).unwrap_or("").to_string();
        let mtime = path_to_mtime.get(&path_str).copied().unwrap_or(0);

        stmt.execute(rusqlite::params![
            point_uid,
            stream_uid,
            scan_uid,
            sample_id,
            frame_no,
            exposure.get(i).copied().flatten(),
            beamline_energy.get(i).copied().flatten(),
            epu.get(i).copied().flatten(),
            sample_theta.get(i).copied().flatten(),
            ccd_theta.get(i).copied().flatten(),
            path_str,
            data_offset.get(i).copied().flatten().unwrap_or(0),
            naxis1.get(i).copied().flatten().unwrap_or(0),
            naxis2.get(i).copied().flatten().unwrap_or(0),
            bitpix.get(i).copied().flatten().unwrap_or(0),
            bzero.get(i).copied().flatten().unwrap_or(0),
            mtime,
            Option::<i64>::None,
            Option::<i64>::None,
            Option::<f64>::None,
        ])?;
    }
    Ok(())
}

pub(crate) fn upsert_bt_batch_rows(
    conn: &rusqlite::Connection,
    beamtime_id: i64,
    rows: &[BtIngestRow],
    path_to_mtime: &HashMap<String, i64>,
) -> Result<()> {
    if rows.is_empty() {
        return Ok(());
    }
    let mut stmt = conn.prepare_cached(
        r#"
        INSERT OR REPLACE INTO bt_scan_points (
            uid, stream_uid, scan_uid, sample_id, seq_index, time,
            exposure, beamline_energy, epu_polarization, sample_theta, ccd_theta,
            source_path, source_data_offset, source_naxis1, source_naxis2,
            source_bitpix, source_bzero, source_mtime, beam_row, beam_col, beam_sigma
        ) VALUES (
            ?1, ?2, ?3, ?4, ?5, 0,
            ?6, ?7, ?8, ?9, ?10,
            ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20
        )"#,
    )?;
    let mut canonical_cache: HashMap<(String, Option<String>), i64> = HashMap::new();
    for row in rows {
        let sn = row.sample_name.as_str();
        let t = row.tag.clone();
        let key = (sn.to_string(), t.clone());
        let sample_id = if let Some(&sid) = canonical_cache.get(&key) {
            sid
        } else {
            let cs = resolve_sample(conn, sn)?;
            let ct = t
                .as_ref()
                .map(|x| x.as_str())
                .and_then(|x| resolve_tag(conn, x).ok().flatten());
            ensure_sample_tag(conn, &cs, ct.as_deref())?;
            let sid = ensure_bt_sample(conn, beamtime_id, &cs, ct.as_deref())?;
            canonical_cache.insert(key, sid);
            sid
        };
        let scan_no = row.scan_number;
        let frame_no = row.frame_number;
        let scan_uid = format!("s_{}_{}", beamtime_id, scan_no);
        let stream_uid = format!("st_{}_{}", beamtime_id, scan_no);
        let point_uid = format!("sp_{}_{}_{}", beamtime_id, scan_no, frame_no);
        let _ = ensure_bt_scan(conn, beamtime_id, sample_id, scan_no)?;
        let _ = ensure_bt_stream(conn, beamtime_id, scan_no)?;
        let path_str = row.file_path.clone();
        let mtime = path_to_mtime.get(&path_str).copied().unwrap_or(0);
        stmt.execute(rusqlite::params![
            point_uid,
            stream_uid,
            scan_uid,
            sample_id,
            frame_no,
            row.exposure,
            row.beamline_energy,
            row.epu_polarization,
            row.sample_theta,
            row.ccd_theta,
            path_str,
            row.data_offset,
            row.naxis1,
            row.naxis2,
            row.bitpix,
            row.bzero,
            mtime,
            Option::<i64>::None,
            Option::<i64>::None,
            Option::<f64>::None,
        ])?;
    }
    Ok(())
}

pub(crate) fn prune_bt_scan_points(conn: &rusqlite::Connection, known_source_paths: &[&str]) -> Result<()> {
    if known_source_paths.is_empty() {
        conn.execute("DELETE FROM bt_scan_points", [])?;
        return Ok(());
    }
    let placeholders = known_source_paths
        .iter()
        .enumerate()
        .map(|(i, _)| format!("?{}", i + 1))
        .collect::<Vec<_>>()
        .join(",");
    let sql = format!(
        "DELETE FROM bt_scan_points WHERE source_path IS NOT NULL AND source_path NOT IN ({})",
        placeholders
    );
    conn.execute(&sql, rusqlite::params_from_iter(known_source_paths.iter()))?;
    Ok(())
}

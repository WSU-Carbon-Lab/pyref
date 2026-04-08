#![cfg(feature = "catalog")]

use crate::catalog::layout::BeamtimeLayout;
use crate::catalog::{CatalogError, Result, FILE_FLAG_PARSE_FAILURE};
#[cfg(not(feature = "parallel_ingest"))]
use crate::catalog::{discover_paths_for_catalog_ingest, open_or_create_db_at};
use crate::io::parse_fits_stem;
use crate::io::BtIngestRow;
#[cfg(not(feature = "parallel_ingest"))]
use polars::prelude::*;
#[cfg(not(feature = "parallel_ingest"))]
use crate::io::options::ReadFitsOptions;
#[cfg(not(feature = "parallel_ingest"))]
use crate::loader::read_fits_metadata_batch;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc;

#[cfg(not(feature = "parallel_ingest"))]
const BATCH_SIZE: usize = 500;

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

fn layout_label(layout: BeamtimeLayout) -> &'static str {
    match layout {
        BeamtimeLayout::Nested => "nested",
        BeamtimeLayout::Flat => "flat",
    }
}

pub(crate) fn update_beamtime_catalog_layout(
    conn: &rusqlite::Connection,
    beamtime_id: i64,
    layout: BeamtimeLayout,
) -> Result<()> {
    conn.execute(
        "UPDATE bt_beamtimes SET catalog_layout = ?1 WHERE id = ?2",
        rusqlite::params![layout_label(layout), beamtime_id],
    )?;
    Ok(())
}

fn ensure_catalog_tag_id(conn: &rusqlite::Connection, name: &str) -> Result<i64> {
    let name = name.trim();
    if name.is_empty() {
        return Err(CatalogError::Validation("empty tag name".into()));
    }
    let slug = name.to_lowercase().replace(' ', "_");
    conn.execute(
        "INSERT OR IGNORE INTO catalog_tags (name, slug) VALUES (?1, ?2)",
        rusqlite::params![name, slug],
    )?;
    let id: i64 = conn.query_row(
        "SELECT id FROM catalog_tags WHERE name = ?1",
        rusqlite::params![name],
        |r| r.get(0),
    )?;
    Ok(id)
}

fn set_fits_file_tags(
    conn: &rusqlite::Connection,
    file_id: i64,
    tag_names: &[String],
) -> Result<()> {
    conn.execute(
        "DELETE FROM fits_file_tags WHERE file_id = ?1",
        rusqlite::params![file_id],
    )?;
    for t in tag_names {
        let t = t.trim();
        if t.is_empty() || crate::io::is_polarization_tag(t) {
            continue;
        }
        let tid = ensure_catalog_tag_id(conn, t)?;
        conn.execute(
            "INSERT OR IGNORE INTO fits_file_tags (file_id, tag_id) VALUES (?1, ?2)",
            rusqlite::params![file_id, tid],
        )?;
    }
    Ok(())
}

fn ensure_bt_sample(conn: &rusqlite::Connection, beamtime_id: i64, name: &str) -> Result<i64> {
    let name = name.trim();
    let name_for_row = if name.is_empty() { "<unparsed>" } else { name };
    conn.execute(
        "INSERT OR IGNORE INTO bt_samples (beamtime_id, name) VALUES (?1, ?2)",
        rusqlite::params![beamtime_id, name_for_row],
    )?;
    let id: i64 = conn.query_row(
        "SELECT id FROM bt_samples WHERE beamtime_id = ?1 AND name = ?2",
        rusqlite::params![beamtime_id, name_for_row],
        |r| r.get(0),
    )?;
    Ok(id)
}

fn upsert_fits_file_row(
    conn: &rusqlite::Connection,
    beamtime_id: i64,
    sample_id: Option<i64>,
    path_str: &str,
    file_stem: &str,
    scan_number: i64,
    frame_number: i64,
    parse_ok: bool,
    source_mtime: i64,
) -> Result<i64> {
    let file_flags = if parse_ok {
        0i64
    } else {
        FILE_FLAG_PARSE_FAILURE
    };
    let parse_ok_i = if parse_ok { 1i64 } else { 0i64 };
    conn.execute(
        r#"INSERT INTO fits_files (beamtime_id, sample_id, path, file_name, scan_number, frame_number, parse_ok, file_flags, source_mtime)
           VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
           ON CONFLICT(path) DO UPDATE SET
             beamtime_id = excluded.beamtime_id,
             sample_id = excluded.sample_id,
             file_name = excluded.file_name,
             scan_number = excluded.scan_number,
             frame_number = excluded.frame_number,
             parse_ok = excluded.parse_ok,
             file_flags = excluded.file_flags,
             source_mtime = excluded.source_mtime"#,
        rusqlite::params![
            beamtime_id,
            sample_id,
            path_str,
            file_stem,
            scan_number,
            frame_number,
            parse_ok_i,
            file_flags,
            source_mtime
        ],
    )?;
    let id: i64 = conn.query_row(
        "SELECT id FROM fits_files WHERE path = ?1",
        rusqlite::params![path_str],
        |r| r.get(0),
    )?;
    Ok(id)
}

fn stem_parse_fields(file_stem: &str) -> (String, Vec<String>, i64, i64, bool) {
    match parse_fits_stem(file_stem) {
        Some(p) => (p.sample_name, p.tags, p.scan_number, p.frame_number, true),
        None => (String::new(), Vec::new(), 0, 0, false),
    }
}

pub(crate) fn prune_fits_files(
    conn: &rusqlite::Connection,
    beamtime_id: i64,
    known_paths: &[&str],
) -> Result<()> {
    if known_paths.is_empty() {
        conn.execute(
            "DELETE FROM fits_files WHERE beamtime_id = ?1",
            rusqlite::params![beamtime_id],
        )?;
        return Ok(());
    }
    let placeholders = known_paths
        .iter()
        .enumerate()
        .map(|(i, _)| format!("?{}", i + 2))
        .collect::<Vec<_>>()
        .join(",");
    let sql = format!(
        "DELETE FROM fits_files WHERE beamtime_id = ?1 AND path NOT IN ({})",
        placeholders
    );
    let mut v: Vec<Box<dyn rusqlite::ToSql>> = vec![Box::new(beamtime_id)];
    for p in known_paths {
        v.push(Box::new(*p));
    }
    let refs: Vec<&dyn rusqlite::ToSql> = v.iter().map(|b| b.as_ref()).collect();
    conn.execute(&sql, rusqlite::params_from_iter(refs))?;
    Ok(())
}

pub fn ingest_beamtime(
    beamtime_dir: &Path,
    header_items: &[String],
    incremental: bool,
    progress_tx: Option<mpsc::Sender<(u32, u32)>>,
) -> Result<PathBuf> {
    ingest_beamtime_with_context(
        beamtime_dir,
        None,
        None,
        header_items,
        incremental,
        progress_tx,
        None,
    )
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
        ingest_beamtime_sequential(
            beamtime_dir,
            data_root,
            experimentalist,
            header_items,
            incremental,
            progress_tx,
            cancel,
        )
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
    let mut processed: u32 = 0;
    let opts = ReadFitsOptions {
        header_items: header_items.to_vec(),
        batch_size: BATCH_SIZE,
        ..ReadFitsOptions::default()
    };
    for chunk in to_ingest.chunks(BATCH_SIZE) {
        if cancel
            .as_ref()
            .map(|c| c.load(std::sync::atomic::Ordering::Relaxed))
            .unwrap_or(false)
        {
            break;
        }
        let chunk_vec: Vec<PathBuf> = chunk.to_vec();
        let df = read_fits_metadata_batch(chunk_vec, &opts)?;
        upsert_fits_and_bt_batch(&conn, beamtime_id, &df, &path_to_mtime)?;
        processed = (processed as usize + chunk.len()).min(to_ingest.len()) as u32;
        if let Some(ref tx) = progress_tx {
            let _ = tx.send((processed, total));
        }
    }
    let path_list: Vec<&str> = path_to_mtime.keys().map(|s| s.as_str()).collect();
    prune_fits_files(&conn, beamtime_id, &path_list)?;
    prune_bt_scan_points(&conn, &path_list)?;
    super::profile_persist::recompute_reflectivity_profiles_for_beamtime(&conn, beamtime_id)?;
    Ok(db_path)
}

#[cfg(not(feature = "parallel_ingest"))]
pub(crate) fn upsert_fits_and_bt_batch(
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
    let file_stem_col = get_str("file_name")?;
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
            source_bitpix, source_bzero, source_mtime, beam_row, beam_col, beam_sigma,
            fits_file_id
        ) VALUES (
            ?1, ?2, ?3, ?4, ?5, 0,
            ?6, ?7, ?8, ?9, ?10,
            ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21
        )"#,
    )?;
    for i in 0..n {
        let path_str = file_path_col.get(i).unwrap_or("").to_string();
        let mtime = path_to_mtime.get(&path_str).copied().unwrap_or(0);
        let stem = file_stem_col
            .get(i)
            .cloned()
            .flatten()
            .unwrap_or_default();
        let (sample_name, tags, scan_number, frame_number, parse_ok) = stem_parse_fields(&stem);
        let sample_id_row = ensure_bt_sample(conn, beamtime_id, &sample_name)?;
        let fits_sample_id = if parse_ok {
            Some(sample_id_row)
        } else {
            None
        };
        let file_id = upsert_fits_file_row(
            conn,
            beamtime_id,
            fits_sample_id,
            &path_str,
            &stem,
            scan_number,
            frame_number,
            parse_ok,
            mtime,
        )?;
        set_fits_file_tags(conn, file_id, &tags)?;
        let scan_no = scan_number;
        let frame_no = frame_number;
        let scan_uid = format!("s_{}_{}", beamtime_id, scan_no);
        let stream_uid = format!("st_{}_{}", beamtime_id, scan_no);
        let point_uid = format!("sp_{}_{}_{}", beamtime_id, scan_no, frame_no);
        let _ = ensure_bt_scan(conn, beamtime_id, sample_id_row, scan_no)?;
        let _ = ensure_bt_stream(conn, beamtime_id, scan_no)?;
        stmt.execute(rusqlite::params![
            point_uid,
            stream_uid,
            scan_uid,
            sample_id_row,
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
            file_id,
        ])?;
    }
    Ok(())
}

pub fn resolve_ingest_target(beamtime_dir: &Path, data_root: Option<&Path>) -> PathBuf {
    match data_root {
        Some(root) => super::data_root_catalog_path(root),
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
            source_bitpix, source_bzero, source_mtime, beam_row, beam_col, beam_sigma,
            fits_file_id
        ) VALUES (
            ?1, ?2, ?3, ?4, ?5, 0,
            ?6, ?7, ?8, ?9, ?10,
            ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21
        )"#,
    )?;
    for row in rows {
        let (sample_name, tags, scan_number, frame_number, parse_ok) =
            stem_parse_fields(&row.file_name);
        let sample_id_row = ensure_bt_sample(conn, beamtime_id, &sample_name)?;
        let fits_sample_id = if parse_ok {
            Some(sample_id_row)
        } else {
            None
        };
        let path_str = row.file_path.as_str();
        let mtime = path_to_mtime.get(path_str).copied().unwrap_or(0);
        let file_id = upsert_fits_file_row(
            conn,
            beamtime_id,
            fits_sample_id,
            path_str,
            row.file_name.as_str(),
            scan_number,
            frame_number,
            parse_ok,
            mtime,
        )?;
        set_fits_file_tags(conn, file_id, &tags)?;
        let scan_uid = format!("s_{}_{}", beamtime_id, scan_number);
        let stream_uid = format!("st_{}_{}", beamtime_id, scan_number);
        let point_uid = format!("sp_{}_{}_{}", beamtime_id, scan_number, frame_number);
        let _ = ensure_bt_scan(conn, beamtime_id, sample_id_row, scan_number)?;
        let _ = ensure_bt_stream(conn, beamtime_id, scan_number)?;
        stmt.execute(rusqlite::params![
            point_uid,
            stream_uid,
            scan_uid,
            sample_id_row,
            row.frame_number,
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
            file_id,
        ])?;
    }
    Ok(())
}

pub(crate) fn prune_bt_scan_points(
    conn: &rusqlite::Connection,
    known_source_paths: &[&str],
) -> Result<()> {
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

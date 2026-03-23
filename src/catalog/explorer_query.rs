#![cfg(feature = "catalog")]

use crate::catalog::{open_catalog_db, Result};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct ExptMeta {
    pub name: String,
    pub beamtime_count: u32,
    pub fits_count: u32,
    pub last_indexed: Option<i64>,
}

#[derive(Debug, Clone)]
pub struct BeamtimeMeta {
    pub path: std::path::PathBuf,
    pub fits_count: Option<u32>,
    pub last_indexed: Option<i64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DbCatalogStatus {
    Indexed,
    Stale,
    NotIndexed,
}

/// Returns experimentalists that have at least one beamtime indexed under data_root.
pub fn list_experimentalists(db_path: &Path, data_root: &Path) -> Result<Vec<ExptMeta>> {
    if !db_path.exists() {
        return Ok(vec![]);
    }
    let conn = open_catalog_db(db_path)?;
    let data_root_str = data_root.to_string_lossy().to_string();
    let mut stmt = conn.prepare(
        "SELECT experimentalist,
                COUNT(*) as beamtime_count,
                COALESCE(
                    (SELECT COUNT(*) FROM bt_scan_points WHERE scan_uid IN
                     (SELECT uid FROM bt_scans WHERE beamtime_id IN
                      (SELECT id FROM bt_beamtimes WHERE data_root = ?1 AND experimentalist = bt_beamtimes.experimentalist))),
                    0
                ) as fits_count,
                MAX(last_indexed_at) as last_indexed
         FROM bt_beamtimes
         WHERE data_root = ?1 AND experimentalist != ''
         GROUP BY experimentalist
         ORDER BY experimentalist"
    )?;
    let rows = stmt.query_map(rusqlite::params![&data_root_str], |row| {
        Ok(ExptMeta {
            name: row.get(0)?,
            beamtime_count: row.get::<_, i64>(1)? as u32,
            fits_count: row.get::<_, i64>(2)? as u32,
            last_indexed: row.get(3)?,
        })
    })?;
    rows.collect::<std::result::Result<Vec<_>, _>>().map_err(|e| e.into())
}

/// Returns beamtimes for one experimentalist under data_root.
pub fn list_beamtimes_for_expt(
    db_path: &Path,
    data_root: &Path,
    experimentalist: &str,
) -> Result<Vec<BeamtimeMeta>> {
    if !db_path.exists() {
        return Ok(vec![]);
    }
    let conn = open_catalog_db(db_path)?;
    let data_root_str = data_root.to_string_lossy().to_string();
    let mut stmt = conn.prepare(
        "SELECT beamtime_path, last_indexed_at,
                COALESCE((SELECT COUNT(*) FROM bt_scan_points WHERE scan_uid IN
                 (SELECT uid FROM bt_scans WHERE beamtime_id = bt_beamtimes.id)), 0) as fits_count
         FROM bt_beamtimes
         WHERE data_root = ?1 AND experimentalist = ?2
         ORDER BY beamtime_path"
    )?;
    let rows = stmt.query_map(rusqlite::params![&data_root_str, experimentalist], |row| {
        Ok(BeamtimeMeta {
            path: std::path::PathBuf::from(row.get::<_, String>(0)?),
            fits_count: Some(row.get::<_, i64>(2)? as u32),
            last_indexed: row.get(1)?,
        })
    })?;
    rows.collect::<std::result::Result<Vec<_>, _>>().map_err(|e| e.into())
}

/// Returns catalog status for a beamtime path.
/// Staleness: compares dir mtime with last_indexed_at.
pub fn catalog_status_for_path(db_path: &Path, beamtime_path: &Path) -> DbCatalogStatus {
    if !db_path.exists() {
        return DbCatalogStatus::NotIndexed;
    }
    let conn = match open_catalog_db(db_path) {
        Ok(c) => c,
        Err(_) => return DbCatalogStatus::NotIndexed,
    };
    let path_str = beamtime_path.to_string_lossy().to_string();
    let last_indexed_at: Option<i64> = conn
        .query_row(
            "SELECT last_indexed_at FROM bt_beamtimes WHERE beamtime_path = ?1",
            rusqlite::params![&path_str],
            |r| r.get(0),
        )
        .ok();

    match last_indexed_at {
        None => DbCatalogStatus::NotIndexed,
        Some(indexed_timestamp) => {
            let dir_mtime = get_dir_max_file_mtime(beamtime_path).unwrap_or(0);
            if dir_mtime > indexed_timestamp {
                DbCatalogStatus::Stale
            } else {
                DbCatalogStatus::Indexed
            }
        }
    }
}

fn get_dir_max_file_mtime(dir: &Path) -> std::io::Result<i64> {
    let mut max_mtime: i64 = 0;
    for entry in walkdir::WalkDir::new(dir)
        .follow_links(false)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        if path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("fits"))
            != Some(true)
        {
            continue;
        }
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        if super::is_skippable_stem(stem) {
            continue;
        }
        if let Ok(metadata) = std::fs::metadata(path) {
            if let Ok(modified) = metadata.modified() {
                if let Ok(duration) = modified.duration_since(std::time::UNIX_EPOCH) {
                    let mtime = duration.as_secs() as i64;
                    if mtime > max_mtime {
                        max_mtime = mtime;
                    }
                }
            }
        }
    }
    Ok(max_mtime)
}

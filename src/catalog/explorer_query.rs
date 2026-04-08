use std::path::Path;

use crate::catalog::{open_catalog_db, paths, Result};
use crate::schema::beamtimes;
use diesel::prelude::*;
use diesel::OptionalExtension;

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

pub fn list_experimentalists(_db_path: &Path, _data_root: &Path) -> Result<Vec<ExptMeta>> {
    Ok(Vec::new())
}

pub fn list_beamtimes_for_expt(
    _db_path: &Path,
    _data_root: &Path,
    _experimentalist: &str,
) -> Result<Vec<BeamtimeMeta>> {
    Ok(Vec::new())
}

pub fn catalog_status_for_path(db_path: &Path, beamtime_path: &Path) -> DbCatalogStatus {
    if !db_path.exists() {
        return DbCatalogStatus::NotIndexed;
    }
    let mut conn = match open_catalog_db(db_path) {
        Ok(c) => c,
        Err(_) => return DbCatalogStatus::NotIndexed,
    };
    let nas_uri = match paths::file_uri_for_path(beamtime_path) {
        Ok(u) => u,
        Err(_) => return DbCatalogStatus::NotIndexed,
    };
    let nested = match beamtimes::table
        .filter(beamtimes::nas_uri.eq(&nas_uri))
        .select(beamtimes::last_indexed_at)
        .first::<Option<i32>>(&mut conn)
        .optional()
    {
        Ok(n) => n,
        Err(_) => return DbCatalogStatus::NotIndexed,
    };
    let Some(Some(ts)) = nested else {
        return DbCatalogStatus::NotIndexed;
    };
    let indexed_timestamp = ts as i64;
    let dir_mtime = get_dir_max_file_mtime(beamtime_path).unwrap_or(0);
    if dir_mtime > indexed_timestamp {
        DbCatalogStatus::Stale
    } else {
        DbCatalogStatus::Indexed
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

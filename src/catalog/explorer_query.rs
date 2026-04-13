//! Catalog paths and explorer-facing queries over the main SQLite catalog.
//!
//! [`catalog_status_for_path`] compares on-disk FITS mtimes against catalog rows.
//! [`list_experimentalists`] and [`list_beamtimes_for_expt`] are deliberate stubs until the
//! explorer lists experimentalists and beamtimes from the beamtime index (or equivalent
//! aggregates). Each stub keeps a compile-time-greppable `todo!` invocation behind `if false` so
//! the integration point stays obvious without panicking at runtime; `std::hint::black_box`
//! references parameters so the current empty results stay warning-free.

use std::path::Path;

use crate::catalog::{open_catalog_db, paths, Result};
use crate::schema::beamtimes;
use diesel::prelude::*;
use diesel::OptionalExtension;

/// Expands to `todo!` with a stable `pyref catalog explorer` prefix for stub catalog explorer APIs.
macro_rules! catalog_explorer_todo {
    ($fn:ident, $detail:literal) => {
        todo!(concat!(
            "pyref catalog explorer — ",
            stringify!($fn),
            ": ",
            $detail
        ))
    };
}

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

/// Lists experimentalist directories under `data_root` with catalog-derived counts.
///
/// Parameters are accepted for the eventual query surface: `db_path` selects the main catalog
/// SQLite file; `data_root` is the NAS or local root that contains experimentalist subfolders.
/// Today this returns an empty `Vec` without touching the database for listing; callers should
/// treat that as "no indexed experimentalists" until the stub is replaced.
#[allow(unreachable_code)]
pub fn list_experimentalists(db_path: &Path, data_root: &Path) -> Result<Vec<ExptMeta>> {
    if false {
        catalog_explorer_todo!(
            list_experimentalists,
            "return one ExptMeta per experimentalist under data_root with beamtime_count, fits_count, last_indexed from beamtime_index or catalog aggregates"
        );
    }
    std::hint::black_box((db_path, data_root));
    Ok(Vec::new())
}

/// Lists beamtime directories for one `experimentalist` under `data_root` with optional catalog metadata.
///
/// Parameters: `db_path` is the main catalog database; `data_root` is the explorer root; `experimentalist`
/// names the immediate child directory under `data_root` whose beamtime subfolders should be listed.
/// Today this returns an empty `Vec` until joined queries against the catalog or beamtime index exist.
#[allow(unreachable_code)]
pub fn list_beamtimes_for_expt(
    db_path: &Path,
    data_root: &Path,
    experimentalist: &str,
) -> Result<Vec<BeamtimeMeta>> {
    if false {
        catalog_explorer_todo!(
            list_beamtimes_for_expt,
            "return BeamtimeMeta rows for the experimentalist subtree (path, fits_count, last_indexed) joined against catalog or beamtime_index"
        );
    }
    std::hint::black_box((db_path, data_root, experimentalist));
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

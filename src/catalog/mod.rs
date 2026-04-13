#![cfg(feature = "catalog")]

mod beamtime_index;
pub mod db;
mod ingest;
mod ingest_progress;
mod layout;
mod models;
mod parallelism;
pub mod paths;
mod query;
mod zarr_write;
mod explorer_query;
mod reflectivity_profile;
mod beamspot_qc;

#[cfg(feature = "watch")]
mod watch;

pub use beamtime_index::{
    ensure_beamtime_index_dir, list_beamtimes, open_beamtime_index_db, register_beamtime,
};

pub use ingest::{
    beamtime_ingest_layout, ingest_beamtime, ingest_beamtime_parallel,
    ingest_beamtime_with_context, ingest_beamtime_with_progress_sink,
    DEFAULT_INGEST_HEADER_ITEMS,
};
pub use ingest_progress::{
    BeamtimeIngestLayout, IngestProgress, IngestProgressSink, ScanFileCount,
};
#[cfg(feature = "parallel_ingest")]
pub use ingest::{ingest_beamtime_pipelined, ingest_beamtime_pipelined_with_context};
pub use parallelism::IngestParallelism;
pub use layout::{detect_beamtime_layout, discover_fits_for_layout, BeamtimeLayout};
pub use query::{
    catalog_file_count, get_overrides, get_scan_point_uid_by_source_path, list_beamtime_entries,
    list_beamtime_entries_v2, list_beamtimes_from_catalog, query_files, query_scan_points,
    rename_file_in_catalog, scan_from_catalog, scan_from_catalog_for_beamtime, set_override,
    update_beamspot, update_beamspot_scan_point, BeamtimeEntries, CatalogFilter, FileRow,
};
pub use explorer_query::{
    list_experimentalists, list_beamtimes_for_expt, catalog_status_for_path, BeamtimeMeta,
    DbCatalogStatus, ExptMeta,
};
pub use beamspot_qc::{beamspot_status, domain_for_row, fit_beamspot_linear, BeamspotLinearFit};
pub use reflectivity_profile::{
    classify_scan_type, segment_reflectivity_profiles, ProfileSegment, ReflectivityScanType,
};

#[cfg(feature = "watch")]
pub use watch::{run_catalog_watcher, WatchHandle, DEFAULT_DEBOUNCE_MS};

use diesel::sqlite::SqliteConnection;
use std::path::{Path, PathBuf};
use thiserror::Error;

pub const CATALOG_DB_NAME: &str = ".pyref_catalog.db";

pub const NEW_CATALOG_DB_NAME: &str = "catalog.db";

#[derive(Error, Debug)]
pub enum CatalogError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("diesel: {0}")]
    Diesel(#[from] diesel::result::Error),
    #[error("diesel connection: {0}")]
    DieselConnection(#[from] diesel::ConnectionError),
    #[error("migrations: {0}")]
    Migrations(String),
    #[error("{0}")]
    Validation(String),
    #[error("not found: {path_or_id} (operation: {operation})")]
    NotFound {
        path_or_id: String,
        operation: String,
    },
    #[error("constraint violation: {message}")]
    ConstraintViolation { message: String },
    #[error("fits read failed: {0}")]
    FitsReadFailed(#[from] crate::errors::FitsError),
}

impl CatalogError {
    pub fn retryable(&self) -> bool {
        match self {
            CatalogError::FitsReadFailed(e) => e.retryable == crate::errors::Retryable::Temporary,
            CatalogError::Io(_) => true,
            CatalogError::Diesel(_) | CatalogError::DieselConnection(_) => true,
            CatalogError::Migrations(_)
            | CatalogError::Validation(_)
            | CatalogError::NotFound { .. }
            | CatalogError::ConstraintViolation { .. } => false,
        }
    }
}

pub type Result<T> = std::result::Result<T, CatalogError>;

pub const FILE_FLAG_PARSE_FAILURE: i64 = 1;

pub fn catalog_path_new(beamtime_dir: &Path) -> PathBuf {
    paths::resolve_catalog_path(beamtime_dir)
}

pub fn data_root_catalog_path(data_root: &Path) -> PathBuf {
    paths::data_root_catalog_path(data_root)
}

pub fn resolve_catalog_path(beamtime_dir: &Path) -> PathBuf {
    paths::resolve_catalog_path(beamtime_dir)
}

pub(crate) fn discover_paths_for_catalog_ingest(
    beamtime_dir: &Path,
) -> Result<(Vec<(PathBuf, i64)>, layout::BeamtimeLayout)> {
    let layout_kind = layout::detect_beamtime_layout(beamtime_dir)?;
    let paths = layout::discover_fits_for_layout(beamtime_dir, layout_kind)?;
    Ok((paths, layout_kind))
}

pub fn is_skippable_stem(stem: &str) -> bool {
    stem.is_empty() || stem.starts_with('_')
}

pub fn discover_fits_paths(beamtime_dir: &Path) -> Result<Vec<(PathBuf, i64)>> {
    if !beamtime_dir.is_dir() {
        return Err(CatalogError::Validation(format!(
            "not a directory: {}",
            beamtime_dir.display()
        )));
    }
    let layout = layout::detect_beamtime_layout(beamtime_dir)?;
    layout::discover_fits_for_layout(beamtime_dir, layout)
}

#[cfg(feature = "parallel_ingest")]
pub fn discover_fits_paths_parallel(beamtime_dir: &Path) -> Result<Vec<(PathBuf, i64)>> {
    discover_fits_paths(beamtime_dir)
}

pub fn open_catalog_db(db_path: &Path) -> Result<SqliteConnection> {
    db::establish_connection(db_path)
}

pub fn open_or_create_db_at(catalog_db_path: &Path) -> Result<SqliteConnection> {
    db::establish_connection(catalog_db_path)
}

pub fn open_or_create_db(beamtime_dir: &Path) -> Result<SqliteConnection> {
    if !beamtime_dir.is_dir() {
        return Err(CatalogError::Validation(format!(
            "beamtime_dir is not a directory: {}",
            beamtime_dir.display()
        )));
    }
    let db_path = resolve_catalog_path(beamtime_dir);
    open_or_create_db_at(&db_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use diesel::prelude::*;
    use tempfile::TempDir;

    use crate::schema::beamtimes;

    #[test]
    fn test_establish_connection_creates_beamtimes_table() {
        let tmp = TempDir::new().unwrap();
        let db = tmp.path().join("catalog.db");
        let mut conn = db::establish_connection(&db).unwrap();
        let _: Vec<i32> = beamtimes::table
            .select(beamtimes::id)
            .limit(0)
            .load(&mut conn)
            .unwrap();
    }

    #[test]
    fn test_open_or_create_db_rejects_file() {
        let tmp = TempDir::new().unwrap();
        let file_path = tmp.path().join("notadir");
        std::fs::write(&file_path, "").unwrap();
        let r = open_or_create_db(&file_path);
        assert!(r.is_err());
    }

    #[test]
    fn test_discover_fits_paths_empty_flat_ccd() {
        let tmp = TempDir::new().unwrap();
        std::fs::create_dir_all(tmp.path().join("CCD")).unwrap();
        let paths = discover_fits_paths(tmp.path()).unwrap();
        assert!(paths.is_empty());
    }

    #[test]
    fn test_is_skippable_stem() {
        assert!(is_skippable_stem(""));
        assert!(is_skippable_stem("_skip"));
        assert!(!is_skippable_stem("sample"));
    }
}

#![cfg(feature = "catalog")]

mod ingest;
mod query;

#[cfg(feature = "watch")]
mod watch;

use rusqlite::Connection;
use std::path::{Path, PathBuf};
use thiserror::Error;
use walkdir::WalkDir;

pub const CATALOG_DB_NAME: &str = ".pyref_catalog.db";

pub use ingest::ingest_beamtime;
pub use query::{
    get_overrides, list_beamtime_entries, query_files, scan_from_catalog, set_override,
    BeamtimeEntries, CatalogFilter, FileRow,
};

#[cfg(feature = "watch")]
pub use watch::{run_catalog_watcher, WatchHandle, DEFAULT_DEBOUNCE_MS};

#[derive(Error, Debug)]
pub enum CatalogError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("sqlite: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("{0}")]
    Validation(String),
}

pub type Result<T> = std::result::Result<T, CatalogError>;

fn catalog_path(beamtime_dir: &Path) -> PathBuf {
    beamtime_dir.join(CATALOG_DB_NAME)
}

const FILES_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS files (
    path TEXT PRIMARY KEY,
    mtime INTEGER NOT NULL,
    file_path TEXT NOT NULL,
    data_offset INTEGER NOT NULL,
    naxis1 INTEGER NOT NULL,
    naxis2 INTEGER NOT NULL,
    bitpix INTEGER NOT NULL,
    bzero INTEGER NOT NULL,
    data_size INTEGER NOT NULL,
    file_name TEXT NOT NULL,
    sample_name TEXT NOT NULL,
    tag TEXT,
    experiment_number INTEGER NOT NULL,
    frame_number INTEGER NOT NULL,
    "DATE" TEXT,
    "Beamline Energy" REAL,
    "Sample Theta" REAL,
    "CCD Theta" REAL,
    "Higher Order Suppressor" REAL,
    "EPU Polarization" REAL,
    EXPOSURE REAL,
    "Sample Name" TEXT,
    "Scan ID" REAL,
    Lambda REAL,
    Q REAL
)"#;

const FILES_INDEX_MTIME: &str = "CREATE INDEX IF NOT EXISTS idx_files_mtime ON files(mtime)";
const FILES_INDEX_SAMPLE: &str = "CREATE INDEX IF NOT EXISTS idx_files_sample_name ON files(sample_name)";
const FILES_INDEX_TAG: &str = "CREATE INDEX IF NOT EXISTS idx_files_tag ON files(tag)";
const FILES_INDEX_EXP: &str = "CREATE INDEX IF NOT EXISTS idx_files_experiment_number ON files(experiment_number)";

const OVERRIDES_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS overrides (
    path TEXT PRIMARY KEY REFERENCES files(path) ON DELETE CASCADE,
    sample_name TEXT,
    tag TEXT,
    notes TEXT
)"#;

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
    let mut out: Vec<(PathBuf, i64)> = Vec::new();
    for entry in WalkDir::new(beamtime_dir)
        .follow_links(false)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        if path.extension().and_then(|e| e.to_str()) != Some("fits") {
            continue;
        }
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        if is_skippable_stem(stem) {
            continue;
        }
        let path_buf = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        let mtime = std::fs::metadata(path)
            .ok()
            .and_then(|m| m.modified().ok())
            .map(|t| {
                t.duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs() as i64
            })
            .unwrap_or(0);
        out.push((path_buf, mtime));
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

pub fn open_or_create_db(beamtime_dir: &Path) -> Result<Connection> {
    let db_path = catalog_path(beamtime_dir);
    if !beamtime_dir.is_dir() {
        return Err(CatalogError::Validation(format!(
            "beamtime_dir is not a directory: {}",
            beamtime_dir.display()
        )));
    }
    let conn = Connection::open(&db_path)?;
    conn.execute_batch(FILES_TABLE)?;
    conn.execute_batch(FILES_INDEX_MTIME)?;
    conn.execute_batch(FILES_INDEX_SAMPLE)?;
    conn.execute_batch(FILES_INDEX_TAG)?;
    conn.execute_batch(FILES_INDEX_EXP)?;
    conn.execute_batch(OVERRIDES_TABLE)?;
    Ok(conn)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_open_or_create_db_creates_schema() {
        let tmp = TempDir::new().unwrap();
        let conn = open_or_create_db(tmp.path()).unwrap();
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name IN ('files','overrides')",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(count, 2);
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
    fn test_discover_fits_paths_empty_dir() {
        let tmp = TempDir::new().unwrap();
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

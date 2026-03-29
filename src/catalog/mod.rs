mod beamtime_index;
mod ingest;
#[cfg(feature = "parallel_ingest")]
mod ingest_parallel;
mod query;
mod explorer_query;
mod reflectivity_profile;

#[cfg(feature = "zarr")]
mod materialize;

#[cfg(feature = "watch")]
mod watch;

pub use beamtime_index::{
    ensure_beamtime_index_dir, list_beamtimes, open_beamtime_index_db, register_beamtime,
};

use rusqlite::Connection;
use std::path::{Path, PathBuf};
use thiserror::Error;
use walkdir::WalkDir;

pub const CATALOG_DB_NAME: &str = ".pyref_catalog.db";

pub use ingest::{ingest_beamtime, ingest_beamtime_with_context, DEFAULT_INGEST_HEADER_ITEMS};
#[cfg(feature = "parallel_ingest")]
pub use ingest_parallel::ingest_beamtime_pipelined;
pub use query::{
    catalog_file_count, get_overrides, get_scan_point_uid_by_source_path, list_beamtime_entries,
    list_beamtime_entries_v2, list_beamtimes_from_catalog, query_files, query_scan_points,
    rename_file_in_catalog, scan_from_catalog, set_override, update_beamspot, update_beamspot_scan_point,
    BeamtimeEntries, CatalogFilter, FileRow,
};
pub use explorer_query::{
    list_experimentalists, list_beamtimes_for_expt, catalog_status_for_path,
    ExptMeta, BeamtimeMeta, DbCatalogStatus,
};
pub use reflectivity_profile::{classify_scan_type, ReflectivityScanType};

#[cfg(feature = "zarr")]
pub use materialize::materialize_beamtime;

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
            CatalogError::Sqlite(_) => true,
            CatalogError::Validation(_)
            | CatalogError::NotFound { .. }
            | CatalogError::ConstraintViolation { .. } => false,
        }
    }
}

pub type Result<T> = std::result::Result<T, CatalogError>;

fn catalog_path(beamtime_dir: &Path) -> PathBuf {
    beamtime_dir.join(CATALOG_DB_NAME)
}

const PYREF_CATALOG_DIR: &str = ".pyref";
const NEW_CATALOG_DB_NAME: &str = "catalog.db";

pub fn catalog_path_new(beamtime_dir: &Path) -> PathBuf {
    let parent = beamtime_dir.parent();
    if let Some(p) = parent {
        if !p.as_os_str().is_empty() && p != Path::new("/") {
            return p.join(PYREF_CATALOG_DIR).join(NEW_CATALOG_DB_NAME);
        }
    }
    beamtime_dir.join(PYREF_CATALOG_DIR).join(NEW_CATALOG_DB_NAME)
}

pub fn data_root_catalog_path(data_root: &Path) -> PathBuf {
    data_root.join(PYREF_CATALOG_DIR).join(NEW_CATALOG_DB_NAME)
}

pub fn resolve_catalog_path(beamtime_dir: &Path) -> PathBuf {
    let new_path = catalog_path_new(beamtime_dir);
    let legacy_path = catalog_path(beamtime_dir);
    if new_path.exists() {
        new_path
    } else if legacy_path.exists() {
        legacy_path
    } else {
        new_path
    }
}

pub fn is_new_catalog_layout(beamtime_dir: &Path) -> bool {
    catalog_path_new(beamtime_dir) != catalog_path(beamtime_dir)
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
    scan_number INTEGER NOT NULL,
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
    Q REAL,
    beam_row INTEGER,
    beam_col INTEGER,
    beam_sigma REAL
)"#;

const FILES_INDEX_MTIME: &str = "CREATE INDEX IF NOT EXISTS idx_files_mtime ON files(mtime)";
const FILES_INDEX_SAMPLE: &str =
    "CREATE INDEX IF NOT EXISTS idx_files_sample_name ON files(sample_name)";
const FILES_INDEX_TAG: &str = "CREATE INDEX IF NOT EXISTS idx_files_tag ON files(tag)";
const FILES_INDEX_SCAN: &str =
    "CREATE INDEX IF NOT EXISTS idx_files_scan_number ON files(scan_number)";

const SAMPLES_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    chemical_formula TEXT,
    created_at INTEGER,
    updated_at INTEGER
)"#;

const TAGS_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    slug TEXT
)"#;

const SAMPLE_TAGS_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS sample_tags (
    sample_id INTEGER NOT NULL REFERENCES samples(id) ON DELETE CASCADE,
    tag_id INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
    PRIMARY KEY (sample_id, tag_id)
)"#;

const OVERRIDES_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS overrides (
    path TEXT PRIMARY KEY REFERENCES files(path) ON DELETE CASCADE,
    sample_name TEXT,
    tag TEXT,
    notes TEXT
)"#;

const BT_BEAMTIMES_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS bt_beamtimes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    beamtime_path TEXT NOT NULL UNIQUE,
    user TEXT NOT NULL DEFAULT '',
    year INTEGER NOT NULL DEFAULT 0,
    month INTEGER NOT NULL DEFAULT 1,
    esaf_number TEXT NOT NULL DEFAULT '',
    label TEXT,
    experimentalist TEXT NOT NULL DEFAULT '',
    data_root TEXT NOT NULL DEFAULT '',
    last_indexed_at INTEGER
)"#;

const BT_MOTOR_NAMES_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS bt_motor_names (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE
)"#;

const BT_AI_CHANNEL_NAMES_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS bt_ai_channel_names (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE
)"#;

const BT_SAMPLES_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS bt_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    beamtime_id INTEGER NOT NULL REFERENCES bt_beamtimes(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    tag TEXT,
    version INTEGER,
    chemical_formula TEXT,
    serial TEXT,
    beamline_pos TEXT,
    extra TEXT,
    UNIQUE (beamtime_id, name, tag, version)
)"#;

const BT_SCANS_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS bt_scans (
    uid TEXT PRIMARY KEY,
    beamtime_id INTEGER REFERENCES bt_beamtimes(id) ON DELETE SET NULL,
    sample_id INTEGER REFERENCES bt_samples(id) ON DELETE SET NULL,
    plan_name TEXT NOT NULL DEFAULT 'reflectivity',
    time_start REAL NOT NULL DEFAULT 0,
    time_stop REAL,
    exit_status TEXT,
    num_points INTEGER DEFAULT 0,
    operator TEXT,
    metadata TEXT
)"#;

const BT_STREAMS_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS bt_streams (
    uid TEXT PRIMARY KEY,
    scan_uid TEXT NOT NULL REFERENCES bt_scans(uid) ON DELETE CASCADE,
    name TEXT NOT NULL DEFAULT 'primary',
    time_created REAL NOT NULL DEFAULT 0,
    UNIQUE (scan_uid, name)
)"#;

const BT_SCAN_POINTS_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS bt_scan_points (
    uid TEXT PRIMARY KEY,
    stream_uid TEXT NOT NULL REFERENCES bt_streams(uid) ON DELETE CASCADE,
    scan_uid TEXT NOT NULL REFERENCES bt_scans(uid) ON DELETE CASCADE,
    sample_id INTEGER REFERENCES bt_samples(id) ON DELETE SET NULL,
    seq_index INTEGER NOT NULL,
    time REAL NOT NULL DEFAULT 0,
    exposure REAL,
    beamline_energy REAL,
    epu_polarization REAL,
    sample_theta REAL,
    sample_x REAL,
    sample_y REAL,
    sample_z REAL,
    ccd_theta REAL,
    source_path TEXT,
    source_data_offset INTEGER,
    source_naxis1 INTEGER,
    source_naxis2 INTEGER,
    source_bitpix INTEGER,
    source_bzero INTEGER,
    source_mtime INTEGER,
    beam_row INTEGER,
    beam_col INTEGER,
    beam_sigma REAL,
    UNIQUE (stream_uid, seq_index)
)"#;

const BT_SCAN_POINT_MOTOR_POSITIONS_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS bt_scan_point_motor_positions (
    scan_point_uid TEXT NOT NULL REFERENCES bt_scan_points(uid) ON DELETE CASCADE,
    motor_id INTEGER NOT NULL REFERENCES bt_motor_names(id) ON DELETE CASCADE,
    position REAL NOT NULL,
    PRIMARY KEY (scan_point_uid, motor_id)
)"#;

const BT_SCAN_POINT_AI_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS bt_scan_point_ai (
    scan_point_uid TEXT NOT NULL REFERENCES bt_scan_points(uid) ON DELETE CASCADE,
    channel_id INTEGER NOT NULL REFERENCES bt_ai_channel_names(id) ON DELETE CASCADE,
    nominal REAL NOT NULL,
    err REAL,
    PRIMARY KEY (scan_point_uid, channel_id)
)"#;

const BT_SCAN_POINT_AI_READINGS_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS bt_scan_point_ai_readings (
    scan_point_uid TEXT NOT NULL REFERENCES bt_scan_points(uid) ON DELETE CASCADE,
    channel_id INTEGER NOT NULL REFERENCES bt_ai_channel_names(id) ON DELETE CASCADE,
    reading_index INTEGER NOT NULL,
    value REAL NOT NULL,
    PRIMARY KEY (scan_point_uid, channel_id, reading_index)
)"#;

const BT_IMAGE_REFS_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS bt_image_refs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_point_uid TEXT NOT NULL REFERENCES bt_scan_points(uid) ON DELETE CASCADE,
    field_name TEXT NOT NULL DEFAULT 'detector_image',
    zarr_group TEXT NOT NULL,
    index_in_stack INTEGER NOT NULL,
    shape_x INTEGER NOT NULL,
    shape_y INTEGER NOT NULL,
    dtype TEXT NOT NULL,
    compression_codec TEXT
)"#;

const BT_INDEX_BEAMTIMES_PATH: &str =
    "CREATE INDEX IF NOT EXISTS idx_bt_beamtimes_path ON bt_beamtimes(beamtime_path)";
const BT_INDEX_SCANS_BEAMTIME: &str =
    "CREATE INDEX IF NOT EXISTS idx_bt_scans_beamtime ON bt_scans(beamtime_id)";
const BT_INDEX_SCAN_POINTS_SCAN: &str =
    "CREATE INDEX IF NOT EXISTS idx_bt_scan_points_scan ON bt_scan_points(scan_uid)";
const BT_INDEX_SCAN_POINTS_SOURCE: &str =
    "CREATE INDEX IF NOT EXISTS idx_bt_scan_points_source ON bt_scan_points(source_path)";
const BT_INDEX_IMAGE_REFS_POINT: &str =
    "CREATE INDEX IF NOT EXISTS idx_bt_image_refs_scan_point ON bt_image_refs(scan_point_uid)";

const BT_FILE_OVERRIDES_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS bt_file_overrides (
    source_path TEXT PRIMARY KEY,
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
        if path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("fits"))
            != Some(true)
        {
            continue;
        }
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        if is_skippable_stem(stem) {
            continue;
        }
        let path_buf = path.canonicalize()?;
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

#[cfg(feature = "parallel_ingest")]
fn collect_fits_under(root: &Path) -> Result<Vec<(PathBuf, i64)>> {
    let mut out: Vec<(PathBuf, i64)> = Vec::new();
    for entry in WalkDir::new(root)
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
        if is_skippable_stem(stem) {
            continue;
        }
        let path_buf = path.canonicalize()?;
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
    Ok(out)
}

#[cfg(feature = "parallel_ingest")]
pub fn discover_fits_paths_parallel(beamtime_dir: &Path) -> Result<Vec<(PathBuf, i64)>> {
    if !beamtime_dir.is_dir() {
        return Err(CatalogError::Validation(format!(
            "not a directory: {}",
            beamtime_dir.display()
        )));
    }
    let subdirs: Vec<PathBuf> = std::fs::read_dir(beamtime_dir)
        .map_err(CatalogError::Io)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .map(|e| e.path())
        .collect();
    if subdirs.len() <= 1 {
        return discover_fits_paths(beamtime_dir);
    }
    use rayon::prelude::*;
    let parts: Result<Vec<Vec<(PathBuf, i64)>>> = subdirs
        .par_iter()
        .map(|d| collect_fits_under(d.as_path()))
        .collect();
    let mut out = parts?;
    let root_only: Vec<(PathBuf, i64)> = std::fs::read_dir(beamtime_dir)
        .map_err(CatalogError::Io)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_file())
        .filter_map(|e| {
            let path = e.path();
            if path.extension()?.to_str()?.eq_ignore_ascii_case("fits") != true {
                return None;
            }
            let stem = path.file_stem()?.to_str()?;
            if is_skippable_stem(stem) {
                return None;
            }
            let path_buf = path.canonicalize().ok()?;
            let mtime = std::fs::metadata(&path)
                .ok()?
                .modified()
                .ok()
                .map(|t| {
                    t.duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs() as i64
                })
                .unwrap_or(0);
            Some((path_buf, mtime))
        })
        .collect();
    out.push(root_only);
    let mut flat: Vec<(PathBuf, i64)> = out.into_iter().flatten().collect();
    flat.sort_by(|a, b| a.0.cmp(&b.0));
    flat.dedup_by(|a, b| a.0 == b.0);
    Ok(flat)
}

fn migrate_add_beamspot_columns(conn: &Connection) -> Result<()> {
    let has_beam_row: bool = conn.query_row(
        "SELECT COUNT(1) FROM pragma_table_info('files') WHERE name = 'beam_row'",
        [],
        |r| r.get(0),
    )?;
    if !has_beam_row {
        conn.execute("ALTER TABLE files ADD COLUMN beam_row INTEGER", [])?;
        conn.execute("ALTER TABLE files ADD COLUMN beam_col INTEGER", [])?;
    }
    Ok(())
}

fn migrate_add_beam_sigma_column(conn: &Connection) -> Result<()> {
    let has_beam_sigma: bool = conn.query_row(
        "SELECT COUNT(1) FROM pragma_table_info('files') WHERE name = 'beam_sigma'",
        [],
        |r| r.get(0),
    )?;
    if !has_beam_sigma {
        conn.execute("ALTER TABLE files ADD COLUMN beam_sigma REAL", [])?;
    }
    Ok(())
}

fn migrate_add_experimentalist_and_data_root(conn: &Connection) -> Result<()> {
    let has_experimentalist: bool = conn.query_row(
        "SELECT COUNT(1) FROM pragma_table_info('bt_beamtimes') WHERE name = 'experimentalist'",
        [],
        |r| r.get(0),
    )?;
    let has_data_root: bool = conn.query_row(
        "SELECT COUNT(1) FROM pragma_table_info('bt_beamtimes') WHERE name = 'data_root'",
        [],
        |r| r.get(0),
    )?;
    if !has_experimentalist {
        let _ = conn.execute(
            "ALTER TABLE bt_beamtimes ADD COLUMN experimentalist TEXT NOT NULL DEFAULT ''",
            [],
        );
    }
    if !has_data_root {
        let _ = conn.execute(
            "ALTER TABLE bt_beamtimes ADD COLUMN data_root TEXT NOT NULL DEFAULT ''",
            [],
        );
    }
    let _ = conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_bt_experimentalist ON bt_beamtimes(data_root, experimentalist)",
        [],
    );
    Ok(())
}

fn migrate_add_last_indexed_at_column(conn: &Connection) -> Result<()> {
    let table_exists: i64 = conn.query_row(
        "SELECT COUNT(1) FROM sqlite_master WHERE type='table' AND name='bt_beamtimes'",
        [],
        |r| r.get(0),
    )?;
    if table_exists == 0 {
        return Ok(());
    }
    let has_col: i64 = conn.query_row(
        "SELECT COUNT(1) FROM pragma_table_info('bt_beamtimes') WHERE name = 'last_indexed_at'",
        [],
        |r| r.get(0),
    )?;
    if has_col == 0 {
        conn.execute(
            "ALTER TABLE bt_beamtimes ADD COLUMN last_indexed_at INTEGER",
            [],
        )?;
    }
    Ok(())
}

pub fn open_catalog_db(db_path: &Path) -> Result<Connection> {
    let conn = Connection::open(db_path)?;
    conn.execute_batch(FILES_TABLE)?;
    conn.execute_batch(SAMPLES_TABLE)?;
    conn.execute_batch(TAGS_TABLE)?;
    conn.execute_batch(SAMPLE_TAGS_TABLE)?;
    conn.execute_batch(OVERRIDES_TABLE)?;
    migrate_experiment_number_to_scan_number(&conn)?;
    migrate_add_beamspot_columns(&conn)?;
    migrate_add_beam_sigma_column(&conn)?;
    conn.execute_batch(FILES_INDEX_MTIME)?;
    conn.execute_batch(FILES_INDEX_SAMPLE)?;
    conn.execute_batch(FILES_INDEX_TAG)?;
    conn.execute_batch(FILES_INDEX_SCAN)?;
    conn.execute_batch(BT_BEAMTIMES_TABLE)?;
    conn.execute_batch(BT_MOTOR_NAMES_TABLE)?;
    conn.execute_batch(BT_AI_CHANNEL_NAMES_TABLE)?;
    conn.execute_batch(BT_SAMPLES_TABLE)?;
    conn.execute_batch(BT_SCANS_TABLE)?;
    conn.execute_batch(BT_STREAMS_TABLE)?;
    conn.execute_batch(BT_SCAN_POINTS_TABLE)?;
    conn.execute_batch(BT_SCAN_POINT_MOTOR_POSITIONS_TABLE)?;
    conn.execute_batch(BT_SCAN_POINT_AI_TABLE)?;
    conn.execute_batch(BT_SCAN_POINT_AI_READINGS_TABLE)?;
    conn.execute_batch(BT_IMAGE_REFS_TABLE)?;
    conn.execute_batch(BT_INDEX_BEAMTIMES_PATH)?;
    conn.execute_batch(BT_INDEX_SCANS_BEAMTIME)?;
    conn.execute_batch(BT_INDEX_SCAN_POINTS_SCAN)?;
    conn.execute_batch(BT_INDEX_SCAN_POINTS_SOURCE)?;
    conn.execute_batch(BT_INDEX_IMAGE_REFS_POINT)?;
    conn.execute_batch(BT_FILE_OVERRIDES_TABLE)?;
    migrate_add_experimentalist_and_data_root(&conn)?;
    migrate_add_last_indexed_at_column(&conn)?;
    Ok(conn)
}

pub fn open_or_create_db(beamtime_dir: &Path) -> Result<Connection> {
    if !beamtime_dir.is_dir() {
        return Err(CatalogError::Validation(format!(
            "beamtime_dir is not a directory: {}",
            beamtime_dir.display()
        )));
    }
    let db_path = resolve_catalog_path(beamtime_dir);
    if let Some(parent) = db_path.parent() {
        if !parent.as_os_str().is_empty() && !db_path.exists() {
            std::fs::create_dir_all(parent).map_err(CatalogError::Io)?;
        }
    }
    open_catalog_db(&db_path)
}

fn migrate_experiment_number_to_scan_number(conn: &Connection) -> Result<()> {
    let has_old: bool = conn.query_row(
        "SELECT COUNT(1) FROM pragma_table_info('files') WHERE name = 'experiment_number'",
        [],
        |r| r.get(0),
    )?;
    if has_old {
        conn.execute(
            "ALTER TABLE files RENAME COLUMN experiment_number TO scan_number",
            [],
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;
    use tempfile::TempDir;

    #[test]
    fn test_open_or_create_db_creates_schema() {
        let tmp = TempDir::new().unwrap();
        let conn = open_or_create_db(tmp.path()).unwrap();
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name IN ('files','overrides','samples','tags','sample_tags')",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(count, 5);
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

    #[test]
    fn test_open_or_create_db_has_beam_sigma_column() {
        let tmp = TempDir::new().unwrap();
        let conn = open_or_create_db(tmp.path()).unwrap();
        let has: i64 = conn
            .query_row(
                "SELECT COUNT(1) FROM pragma_table_info('files') WHERE name = 'beam_sigma'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(has, 1);
    }

    #[test]
    fn test_migrate_restores_last_indexed_at_on_bt_beamtimes() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("catalog.db");
        {
            let _ = open_catalog_db(&db_path).unwrap();
        }
        {
            let conn = Connection::open(&db_path).unwrap();
            let _ = conn.execute(
                "ALTER TABLE bt_beamtimes DROP COLUMN last_indexed_at",
                [],
            );
        }
        let conn = open_catalog_db(&db_path).unwrap();
        let has: i64 = conn
            .query_row(
                "SELECT COUNT(1) FROM pragma_table_info('bt_beamtimes') WHERE name = 'last_indexed_at'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(has, 1);
    }
}

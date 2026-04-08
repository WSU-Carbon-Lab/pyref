mod beamtime_index;
mod ingest;
mod layout;
#[cfg(feature = "parallel_ingest")]
mod ingest_parallel;
mod query;
mod explorer_query;
mod reflectivity_profile;
mod beamspot_qc;
mod profile_persist;

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
pub const CATALOG_DB_NAME: &str = ".pyref_catalog.db";

pub use ingest::{
    ingest_beamtime, ingest_beamtime_with_context, DEFAULT_INGEST_HEADER_ITEMS,
};
pub use layout::{detect_beamtime_layout, discover_fits_for_layout, BeamtimeLayout};
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
pub use beamspot_qc::{beamspot_status, domain_for_row, fit_beamspot_linear, BeamspotLinearFit};
pub use reflectivity_profile::{
    classify_scan_type, segment_reflectivity_profiles, ProfileSegment, ReflectivityScanType,
};

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

/// Bit in ``fits_files.file_flags`` when the filename stem did not satisfy the AGENTS parse contract.
pub const FILE_FLAG_PARSE_FAILURE: i64 = 1;

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

/// Resolves the catalog database path for a beamtime directory (AGENTS layout: `parent/.pyref/catalog.db`).
pub fn resolve_catalog_path(beamtime_dir: &Path) -> PathBuf {
    catalog_path_new(beamtime_dir)
}

const FITS_FILES_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS fits_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    beamtime_id INTEGER NOT NULL REFERENCES bt_beamtimes(id) ON DELETE CASCADE,
    sample_id INTEGER REFERENCES bt_samples(id) ON DELETE SET NULL,
    path TEXT NOT NULL UNIQUE,
    file_name TEXT NOT NULL,
    scan_number INTEGER NOT NULL,
    frame_number INTEGER NOT NULL,
    parse_ok INTEGER NOT NULL DEFAULT 0,
    file_flags INTEGER NOT NULL DEFAULT 0,
    source_mtime INTEGER NOT NULL DEFAULT 0
)"#;

const FITS_FILES_INDEX_BEAMTIME: &str =
    "CREATE INDEX IF NOT EXISTS idx_fits_files_beamtime ON fits_files(beamtime_id)";
const FITS_FILES_INDEX_SCAN: &str =
    "CREATE INDEX IF NOT EXISTS idx_fits_files_scan ON fits_files(scan_number)";

const CATALOG_TAGS_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS catalog_tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    slug TEXT NOT NULL
)"#;

const FITS_FILE_TAGS_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS fits_file_tags (
    file_id INTEGER NOT NULL REFERENCES fits_files(id) ON DELETE CASCADE,
    tag_id INTEGER NOT NULL REFERENCES catalog_tags(id) ON DELETE CASCADE,
    PRIMARY KEY (file_id, tag_id)
)"#;

const CATALOG_BEAM_FINDING_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS catalog_beam_finding (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_point_uid TEXT NOT NULL UNIQUE REFERENCES bt_scan_points(uid) ON DELETE CASCADE,
    edge_trim_applied INTEGER,
    bg_row_params TEXT,
    bg_col_params TEXT,
    gaussian_sigma REAL,
    centroid_row REAL,
    centroid_col REAL,
    roi_intensity REAL,
    fit_sigma REAL,
    bg_intensity REAL,
    bg_intensity_err REAL,
    detection_ok INTEGER,
    quality_flags INTEGER NOT NULL DEFAULT 0
)"#;

const CATALOG_STITCH_CORRECTION_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS catalog_stitch_correction (
    scan_uid TEXT NOT NULL REFERENCES bt_scans(uid) ON DELETE CASCADE,
    stitch_index INTEGER NOT NULL,
    fano_factor REAL,
    overlap_scale REAL,
    i0_value REAL,
    i0_source TEXT,
    external_i0_scan_uid TEXT,
    frame_class TEXT,
    PRIMARY KEY (scan_uid, stitch_index)
)"#;

const CATALOG_REFLECTIVITY_FRAME_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS catalog_reflectivity_frame (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    beam_finding_id INTEGER NOT NULL REFERENCES catalog_beam_finding(id) ON DELETE CASCADE,
    theta REAL,
    energy REAL,
    q REAL,
    intensity REAL,
    intensity_err REAL,
    frame_class TEXT,
    quality_flags INTEGER NOT NULL DEFAULT 0
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
    sample_x REAL,
    sample_y REAL,
    chemical_formula TEXT,
    serial TEXT,
    beamline_pos TEXT,
    extra TEXT,
    UNIQUE (beamtime_id, name)
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
    fits_file_id INTEGER REFERENCES fits_files(id) ON DELETE SET NULL,
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

const BT_REFLECTIVITY_PROFILES_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS bt_reflectivity_profiles (
    scan_uid TEXT NOT NULL REFERENCES bt_scans(uid) ON DELETE CASCADE,
    profile_index INTEGER NOT NULL,
    scan_type TEXT NOT NULL,
    seq_index_first INTEGER NOT NULL,
    seq_index_last INTEGER NOT NULL,
    e_min REAL,
    e_max REAL,
    t_min REAL,
    t_max REAL,
    PRIMARY KEY (scan_uid, profile_index)
)"#;

const BT_INDEX_REFLECTIVITY_PROFILES_SCAN: &str =
    "CREATE INDEX IF NOT EXISTS idx_bt_reflectivity_profiles_scan ON bt_reflectivity_profiles(scan_uid)";

const BT_FITS_DISCOVERY_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS bt_fits_discovery (
    beamtime_id INTEGER NOT NULL REFERENCES bt_beamtimes(id) ON DELETE CASCADE,
    source_path TEXT NOT NULL,
    source_mtime INTEGER NOT NULL,
    file_stem TEXT NOT NULL,
    stem_sample_name TEXT NOT NULL,
    stem_tag TEXT,
    scan_number INTEGER NOT NULL,
    frame_number INTEGER NOT NULL,
    stem_parse_ok INTEGER NOT NULL,
    resolved_sample_name TEXT NOT NULL,
    resolved_tag TEXT,
    PRIMARY KEY (beamtime_id, source_path)
)"#;

const BT_INDEX_FITS_DISCOVERY_BEAMTIME: &str =
    "CREATE INDEX IF NOT EXISTS idx_bt_fits_discovery_beamtime ON bt_fits_discovery(beamtime_id)";

fn migrate_bt_fits_discovery(conn: &Connection) -> Result<()> {
    conn.execute_batch(BT_FITS_DISCOVERY_TABLE)?;
    conn.execute_batch(BT_INDEX_FITS_DISCOVERY_BEAMTIME)?;
    Ok(())
}

fn migrate_drop_legacy_denormalized_catalog(conn: &Connection) -> Result<()> {
    let legacy: i64 = conn.query_row(
        "SELECT COUNT(1) FROM sqlite_master WHERE type='table' AND name IN ('files','overrides','sample_tags','samples','tags')",
        [],
        |r| r.get(0),
    )?;
    if legacy == 0 {
        return Ok(());
    }
    conn.execute_batch(
        r#"
        DROP TABLE IF EXISTS overrides;
        DROP TABLE IF EXISTS sample_tags;
        DROP TABLE IF EXISTS files;
        DROP TABLE IF EXISTS tags;
        DROP TABLE IF EXISTS samples;
    "#,
    )?;
    Ok(())
}

fn migrate_bt_samples_schema_agents(conn: &Connection) -> Result<()> {
    let has_table: i64 = conn.query_row(
        "SELECT COUNT(1) FROM sqlite_master WHERE type='table' AND name='bt_samples'",
        [],
        |r| r.get(0),
    )?;
    if has_table == 0 {
        return Ok(());
    }
    let has_tag: i64 = conn.query_row(
        "SELECT COUNT(1) FROM pragma_table_info('bt_samples') WHERE name='tag'",
        [],
        |r| r.get(0),
    )?;
    if has_tag == 0 {
        return Ok(());
    }
    conn.execute_batch(
        r#"
        PRAGMA foreign_keys=OFF;
        DROP TABLE IF EXISTS catalog_reflectivity_frame;
        DROP TABLE IF EXISTS catalog_stitch_correction;
        DROP TABLE IF EXISTS catalog_beam_finding;
        DROP TABLE IF EXISTS bt_image_refs;
        DROP TABLE IF EXISTS fits_file_tags;
        DROP TABLE IF EXISTS fits_files;
        DROP TABLE IF EXISTS bt_scan_points;
        DROP TABLE IF EXISTS bt_streams;
        DROP TABLE IF EXISTS bt_scans;
        DROP TABLE IF EXISTS bt_reflectivity_profiles;
        DROP TABLE IF EXISTS bt_fits_discovery;
        DROP TABLE IF EXISTS bt_samples;
        PRAGMA foreign_keys=ON;
    "#,
    )?;
    Ok(())
}

fn migrate_bt_beamtimes_catalog_fields(conn: &Connection) -> Result<()> {
    let has_layout: i64 = conn.query_row(
        "SELECT COUNT(1) FROM pragma_table_info('bt_beamtimes') WHERE name='catalog_layout'",
        [],
        |r| r.get(0),
    )?;
    if has_layout == 0 {
        conn.execute(
            "ALTER TABLE bt_beamtimes ADD COLUMN catalog_layout TEXT",
            [],
        )?;
    }
    let has_date: i64 = conn.query_row(
        "SELECT COUNT(1) FROM pragma_table_info('bt_beamtimes') WHERE name='beamtime_date'",
        [],
        |r| r.get(0),
    )?;
    if has_date == 0 {
        conn.execute(
            "ALTER TABLE bt_beamtimes ADD COLUMN beamtime_date TEXT",
            [],
        )?;
    }
    Ok(())
}

fn migrate_bt_scan_points_fits_file_id(conn: &Connection) -> Result<()> {
    let has_table: i64 = conn.query_row(
        "SELECT COUNT(1) FROM sqlite_master WHERE type='table' AND name='bt_scan_points'",
        [],
        |r| r.get(0),
    )?;
    if has_table == 0 {
        return Ok(());
    }
    let has_col: i64 = conn.query_row(
        "SELECT COUNT(1) FROM pragma_table_info('bt_scan_points') WHERE name='fits_file_id'",
        [],
        |r| r.get(0),
    )?;
    if has_col == 0 {
        conn.execute(
            "ALTER TABLE bt_scan_points ADD COLUMN fits_file_id INTEGER REFERENCES fits_files(id) ON DELETE SET NULL",
            [],
        )?;
    }
    Ok(())
}

/// Discover FITS paths for ingest using AGENTS layout rules under ``beamtime_dir``.
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

/// Discovers FITS under ``beamtime_dir`` using AGENTS layout rules only (no silent full-tree fallback).
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

fn migrate_reflectivity_profiles_schema(conn: &Connection) -> Result<()> {
    conn.execute_batch(BT_REFLECTIVITY_PROFILES_TABLE)?;
    conn.execute_batch(BT_INDEX_REFLECTIVITY_PROFILES_SCAN)?;
    let has_col: i64 = conn.query_row(
        "SELECT COUNT(1) FROM pragma_table_info('bt_scan_points') WHERE name = 'reflectivity_profile_index'",
        [],
        |r| r.get(0),
    )?;
    if has_col == 0 {
        conn.execute(
            "ALTER TABLE bt_scan_points ADD COLUMN reflectivity_profile_index INTEGER",
            [],
        )?;
    }
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
    migrate_drop_legacy_denormalized_catalog(&conn)?;
    migrate_bt_samples_schema_agents(&conn)?;
    conn.execute_batch(BT_BEAMTIMES_TABLE)?;
    migrate_bt_beamtimes_catalog_fields(&conn)?;
    migrate_add_experimentalist_and_data_root(&conn)?;
    migrate_add_last_indexed_at_column(&conn)?;
    conn.execute_batch(BT_MOTOR_NAMES_TABLE)?;
    conn.execute_batch(BT_AI_CHANNEL_NAMES_TABLE)?;
    conn.execute_batch(BT_SAMPLES_TABLE)?;
    conn.execute_batch(CATALOG_TAGS_TABLE)?;
    conn.execute_batch(FITS_FILES_TABLE)?;
    conn.execute_batch(FITS_FILES_INDEX_BEAMTIME)?;
    conn.execute_batch(FITS_FILES_INDEX_SCAN)?;
    conn.execute_batch(FITS_FILE_TAGS_TABLE)?;
    conn.execute_batch(BT_SCANS_TABLE)?;
    conn.execute_batch(BT_STREAMS_TABLE)?;
    conn.execute_batch(BT_SCAN_POINTS_TABLE)?;
    migrate_bt_scan_points_fits_file_id(&conn)?;
    conn.execute_batch(BT_SCAN_POINT_MOTOR_POSITIONS_TABLE)?;
    conn.execute_batch(BT_SCAN_POINT_AI_TABLE)?;
    conn.execute_batch(BT_SCAN_POINT_AI_READINGS_TABLE)?;
    conn.execute_batch(CATALOG_BEAM_FINDING_TABLE)?;
    conn.execute_batch(CATALOG_STITCH_CORRECTION_TABLE)?;
    conn.execute_batch(CATALOG_REFLECTIVITY_FRAME_TABLE)?;
    conn.execute_batch(BT_IMAGE_REFS_TABLE)?;
    conn.execute_batch(BT_INDEX_BEAMTIMES_PATH)?;
    conn.execute_batch(BT_INDEX_SCANS_BEAMTIME)?;
    conn.execute_batch(BT_INDEX_SCAN_POINTS_SCAN)?;
    conn.execute_batch(BT_INDEX_SCAN_POINTS_SOURCE)?;
    conn.execute_batch(BT_INDEX_IMAGE_REFS_POINT)?;
    conn.execute_batch(BT_FILE_OVERRIDES_TABLE)?;
    migrate_reflectivity_profiles_schema(&conn)?;
    migrate_bt_fits_discovery(&conn)?;
    Ok(conn)
}

/// Opens or creates the catalog at ``catalog_db_path`` (creates parent directories when missing).
pub fn open_or_create_db_at(catalog_db_path: &Path) -> Result<Connection> {
    if let Some(parent) = catalog_db_path.parent() {
        if !parent.as_os_str().is_empty() && !catalog_db_path.exists() {
            std::fs::create_dir_all(parent).map_err(CatalogError::Io)?;
        }
    }
    open_catalog_db(catalog_db_path)
}

pub fn open_or_create_db(beamtime_dir: &Path) -> Result<Connection> {
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
    use rusqlite::Connection;
    use tempfile::TempDir;

    #[test]
    fn test_open_or_create_db_creates_schema() {
        let tmp = TempDir::new().unwrap();
        std::fs::create_dir_all(tmp.path().join("CCD")).unwrap();
        let conn = open_or_create_db(tmp.path()).unwrap();
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name IN ('fits_files','catalog_tags','bt_beamtimes','bt_samples')",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(count, 4);
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

    #[test]
    fn test_open_or_create_db_has_fits_files_parse_ok() {
        let tmp = TempDir::new().unwrap();
        std::fs::create_dir_all(tmp.path().join("CCD")).unwrap();
        let conn = open_or_create_db(tmp.path()).unwrap();
        let has: i64 = conn
            .query_row(
                "SELECT COUNT(1) FROM pragma_table_info('fits_files') WHERE name = 'parse_ok'",
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

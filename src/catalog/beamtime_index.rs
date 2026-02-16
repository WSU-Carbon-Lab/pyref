#![cfg(feature = "catalog")]

use rusqlite::Connection;
use std::path::{Path, PathBuf};

use crate::catalog::{CatalogError, Result};

const BEAMTIME_INDEX_DB_NAME: &str = "beamtime_index.sqlite3";
const PYREF_DIR: &str = ".pyref";

const BEAMTIMES_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS beamtimes (
    path TEXT PRIMARY KEY,
    last_indexed_at INTEGER NOT NULL,
    file_count INTEGER
)"#;

fn beamtime_index_dir() -> PathBuf {
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home).join(PYREF_DIR);
    }
    PathBuf::from(".").join(PYREF_DIR)
}

pub fn ensure_beamtime_index_dir() -> Result<()> {
    let dir = beamtime_index_dir();
    std::fs::create_dir_all(&dir).map_err(CatalogError::from)
}

pub fn open_beamtime_index_db() -> Result<Connection> {
    ensure_beamtime_index_dir()?;
    let path = beamtime_index_dir().join(BEAMTIME_INDEX_DB_NAME);
    let conn = Connection::open(&path)?;
    conn.execute_batch(BEAMTIMES_TABLE)?;
    Ok(conn)
}

pub fn list_beamtimes() -> Result<Vec<(PathBuf, i64)>> {
    let conn = open_beamtime_index_db()?;
    let mut stmt =
        conn.prepare("SELECT path, last_indexed_at FROM beamtimes ORDER BY last_indexed_at DESC")?;
    let rows = stmt.query_map([], |r| {
        Ok((PathBuf::from(r.get::<_, String>(0)?), r.get::<_, i64>(1)?))
    })?;
    let out: Vec<_> = rows.filter_map(|r| r.ok()).collect();
    Ok(out)
}

pub fn register_beamtime(path: &Path, file_count: Option<u32>) -> Result<()> {
    let canonical = path.canonicalize().map_err(|e| {
        CatalogError::Validation(format!(
            "path cannot be canonicalized: {}: {}",
            path.display(),
            e
        ))
    })?;
    let path_str = canonical.to_string_lossy().into_owned();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let conn = open_beamtime_index_db()?;
    conn.execute(
        "INSERT OR REPLACE INTO beamtimes (path, last_indexed_at, file_count) VALUES (?1, ?2, ?3)",
        rusqlite::params![path_str, now, file_count.map(i64::from)],
    )?;
    Ok(())
}

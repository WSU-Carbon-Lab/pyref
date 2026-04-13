use diesel::connection::SimpleConnection;
use diesel::deserialize::{self, QueryableByName};
use diesel::sql_types::{BigInt, Nullable, Text};
use diesel::sqlite::{Sqlite, SqliteConnection};
use diesel::{sql_query, Connection, RunQueryDsl};
use std::path::{Path, PathBuf};

use crate::catalog::{CatalogError, Result};

const BEAMTIME_INDEX_DB_NAME: &str = "beamtime_index.sqlite3";
const PYREF_DIR: &str = ".pyref";

const BEAMTIMES_TABLE: &str = r#"
CREATE TABLE IF NOT EXISTS beamtimes (
    path TEXT PRIMARY KEY,
    last_indexed_at INTEGER NOT NULL,
    file_count INTEGER
);"#;

fn beamtime_index_dir() -> PathBuf {
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home).join(PYREF_DIR);
    }
    PathBuf::from(".").join(PYREF_DIR)
}

pub fn ensure_beamtime_index_dir() -> Result<()> {
    let dir = beamtime_index_dir();
    std::fs::create_dir_all(&dir).map_err(CatalogError::Io)
}

pub fn open_beamtime_index_db() -> Result<SqliteConnection> {
    ensure_beamtime_index_dir()?;
    let path = beamtime_index_dir().join(BEAMTIME_INDEX_DB_NAME);
    let s = path
        .to_str()
        .ok_or_else(|| CatalogError::Validation("beamtime index path is not valid UTF-8".into()))?;
    let mut conn = SqliteConnection::establish(s).map_err(CatalogError::DieselConnection)?;
    conn.batch_execute(BEAMTIMES_TABLE)
        .map_err(CatalogError::Diesel)?;
    Ok(conn)
}

struct BeamtimeIdxEntry {
    path: String,
    last_indexed_at: i64,
}

impl QueryableByName<Sqlite> for BeamtimeIdxEntry {
    fn build<'a>(row: &impl diesel::row::NamedRow<'a, Sqlite>) -> deserialize::Result<Self> {
        Ok(Self {
            path: diesel::row::NamedRow::get::<Text, String>(row, "path")?,
            last_indexed_at: diesel::row::NamedRow::get::<BigInt, i64>(row, "last_indexed_at")?,
        })
    }
}

pub fn list_beamtimes() -> Result<Vec<(PathBuf, i64)>> {
    let mut conn = open_beamtime_index_db()?;
    let rows: Vec<BeamtimeIdxEntry> =
        sql_query("SELECT path, last_indexed_at FROM beamtimes ORDER BY last_indexed_at DESC")
            .load(&mut conn)
            .map_err(CatalogError::Diesel)?;
    Ok(rows
        .into_iter()
        .map(|r| (PathBuf::from(r.path), r.last_indexed_at))
        .collect())
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
    let mut conn = open_beamtime_index_db()?;
    let fc: Option<i64> = file_count.map(|x| x as i64);
    sql_query(
        "INSERT OR REPLACE INTO beamtimes (path, last_indexed_at, file_count) VALUES (?1, ?2, ?3)",
    )
    .bind::<Text, _>(&path_str)
    .bind::<BigInt, _>(now)
    .bind::<Nullable<BigInt>, _>(fc)
    .execute(&mut conn)
    .map_err(CatalogError::Diesel)?;
    Ok(())
}

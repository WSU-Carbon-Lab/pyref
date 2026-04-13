//! Diesel SQLite connection: foreign keys, embedded migrations.

use diesel::connection::SimpleConnection;
use diesel::sqlite::SqliteConnection;
use diesel::Connection;
use diesel_migrations::{embed_migrations, EmbeddedMigrations, MigrationHarness};
use std::path::Path;

use super::{CatalogError, Result};

pub const MIGRATIONS: EmbeddedMigrations = embed_migrations!("migrations");

/// Opens a SQLite connection to ``database_url`` (path to the ``.db`` file), enables foreign keys,
/// and runs pending Diesel migrations. Creates parent directories when the file is new.
pub fn establish_connection(database_url: &Path) -> Result<SqliteConnection> {
    if let Some(parent) = database_url.parent() {
        if !parent.as_os_str().is_empty() && !database_url.exists() {
            std::fs::create_dir_all(parent).map_err(CatalogError::Io)?;
        }
    }
    let path_str = database_url.to_str().ok_or_else(|| {
        CatalogError::Validation("catalog database path is not valid UTF-8".into())
    })?;
    let mut conn = SqliteConnection::establish(path_str).map_err(CatalogError::DieselConnection)?;
    conn.batch_execute("PRAGMA foreign_keys = ON;")
        .map_err(CatalogError::Diesel)?;
    conn.run_pending_migrations(MIGRATIONS)
        .map_err(|e| CatalogError::Migrations(format!("{e:?}")))?;
    Ok(conn)
}

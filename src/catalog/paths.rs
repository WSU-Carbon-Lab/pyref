//! Catalog and zarr paths.
//!
//! Default ``catalog.db`` and zarr cache live under ``~/.config/pyref`` on macOS (``XDG_CONFIG_HOME``
//! is not used for the default tree, so Application Support cannot hijack the location). On Linux
//! and Windows, ``$XDG_CONFIG_HOME/pyref`` is used when ``XDG_CONFIG_HOME`` is set; otherwise
//! ``~/.config/pyref`` (e.g. ``C:\\Users\\name\\.config\\pyref`` on Windows).
//!
//! Overrides (optional, for ``set-catalog`` / ``set-cache`` style workflows):
//!
//! - ``PYREF_CATALOG_DB``: absolute path to ``catalog.db``. Parent directories are created when
//!   missing. When set, ``default_catalog_db_path`` ignores other defaults.
//! - ``PYREF_HOME``: directory used as the catalog parent when ``PYREF_CATALOG_DB`` is unset
//!   (typically tests; catalog path is ``<PYREF_HOME>/catalog.db``).
//! - ``PYREF_CACHE_ROOT``: directory under which each beamtime gets ``<sha256>/beamtime.zarr``.
//!   When unset, zarr uses ``<pyref_config_dir>/cache/<sha256>/beamtime.zarr``.

use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};

use super::{CatalogError, Result};

const ENV_CATALOG_DB: &str = "PYREF_CATALOG_DB";
const ENV_CACHE_ROOT: &str = "PYREF_CACHE_ROOT";
#[cfg(not(target_os = "macos"))]
const ENV_XDG_CONFIG_HOME: &str = "XDG_CONFIG_HOME";

/// User config and local cache root: same tree as ``default_catalog_db_path`` parent.
/// On macOS this is always ``~/.config/pyref`` unless ``PYREF_HOME`` is set. Elsewhere,
/// ``$XDG_CONFIG_HOME/pyref`` when ``XDG_CONFIG_HOME`` is set, otherwise ``~/.config/pyref``,
/// or ``PYREF_HOME`` in tests.
pub fn pyref_config_dir() -> Result<PathBuf> {
    if let Ok(h) = std::env::var("PYREF_HOME") {
        let p = PathBuf::from(h);
        if !p.exists() {
            fs::create_dir_all(&p).map_err(CatalogError::Io)?;
        }
        return Ok(p);
    }
    #[cfg(target_os = "macos")]
    {
        let base = directories::BaseDirs::new()
            .ok_or_else(|| CatalogError::Validation("could not resolve home directory".into()))?;
        let d = base.home_dir().join(".config").join("pyref");
        if !d.exists() {
            fs::create_dir_all(&d).map_err(CatalogError::Io)?;
        }
        Ok(d)
    }
    #[cfg(not(target_os = "macos"))]
    {
        if let Ok(xdg) = std::env::var(ENV_XDG_CONFIG_HOME) {
            if !xdg.is_empty() {
                let d = PathBuf::from(xdg).join("pyref");
                if !d.exists() {
                    fs::create_dir_all(&d).map_err(CatalogError::Io)?;
                }
                return Ok(d);
            }
        }
        let base = directories::BaseDirs::new()
            .ok_or_else(|| CatalogError::Validation("could not resolve home directory".into()))?;
        let d = base.home_dir().join(".config").join("pyref");
        if !d.exists() {
            fs::create_dir_all(&d).map_err(CatalogError::Io)?;
        }
        Ok(d)
    }
}

/// Alias for [`pyref_config_dir`]. Catalog, zarr cache, CLI config, and daemons share this root.
pub fn pyref_data_dir() -> Result<PathBuf> {
    pyref_config_dir()
}

fn default_catalog_parent_dir() -> Result<PathBuf> {
    pyref_config_dir()
}

/// Absolute path to the global catalog database.
///
/// Honors ``PYREF_CATALOG_DB`` when set. Otherwise ``PYREF_HOME/catalog.db`` when ``PYREF_HOME`` is
/// set, or ``<default_catalog_parent_dir>/catalog.db`` (``~/.config/pyref/catalog.db`` when
/// ``XDG_CONFIG_HOME`` is unset).
pub fn default_catalog_db_path() -> Result<PathBuf> {
    if let Ok(p) = std::env::var(ENV_CATALOG_DB) {
        let path = PathBuf::from(p);
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).map_err(CatalogError::Io)?;
            }
        }
        return Ok(path);
    }
    Ok(default_catalog_parent_dir()?.join("catalog.db"))
}

/// Stable SHA-256 hex digest of the canonical beamtime root path (for cache directory names).
pub fn beamtime_sha256_hex(beamtime_dir: &Path) -> Result<String> {
    let canon = beamtime_dir.canonicalize().map_err(CatalogError::Io)?;
    let s = canon.to_string_lossy();
    let d = Sha256::digest(s.as_bytes());
    Ok(d.iter().fold(String::with_capacity(64), |mut acc, b| {
        use std::fmt::Write;
        let _ = write!(acc, "{b:02x}");
        acc
    }))
}

/// Local zarr store path for one beamtime.
///
/// Default: ``<pyref_config_dir>/cache/<sha256>/beamtime.zarr``.
/// With ``PYREF_CACHE_ROOT``: ``<PYREF_CACHE_ROOT>/<sha256>/beamtime.zarr``.
pub fn beamtime_zarr_path(beamtime_dir: &Path) -> Result<PathBuf> {
    let hash = beamtime_sha256_hex(beamtime_dir)?;
    let dir = if let Ok(root) = std::env::var(ENV_CACHE_ROOT) {
        PathBuf::from(root).join(&hash)
    } else {
        pyref_config_dir()?.join("cache").join(&hash)
    };
    fs::create_dir_all(&dir).map_err(CatalogError::Io)?;
    Ok(dir.join("beamtime.zarr"))
}

/// Logical URI for a locally mounted beamtime directory (provenance / re-ingestion key).
pub fn file_uri_for_path(path: &Path) -> Result<String> {
    let canon = path.canonicalize().map_err(CatalogError::Io)?;
    let s = canon.to_string_lossy();
    Ok(format!("file://{s}"))
}

/// Logical URI for a beamtime path with offline-friendly fallback.
///
/// Uses canonicalized absolute path when possible. If canonicalization fails and
/// the input is already absolute (for example, NAS mount unavailable), falls back
/// to the literal absolute path string.
pub fn file_uri_for_path_relaxed(path: &Path) -> Result<String> {
    match path.canonicalize() {
        Ok(canon) => Ok(format!("file://{}", canon.to_string_lossy())),
        Err(err) => {
            if path.is_absolute() {
                Ok(format!("file://{}", path.to_string_lossy()))
            } else {
                Err(CatalogError::Io(err))
            }
        }
    }
}

/// Resolves the catalog database path. The beamtime argument is accepted for API compatibility;
/// the catalog is global and the path does not depend on it.
pub fn resolve_catalog_path(_beamtime_dir: &Path) -> PathBuf {
    default_catalog_db_path()
        .expect("catalog path: set PYREF_HOME or ensure a writable user data directory")
}

/// Legacy helper: returns the global catalog path (single-catalog layout).
pub fn data_root_catalog_path(_data_root: &Path) -> PathBuf {
    resolve_catalog_path(_data_root)
}

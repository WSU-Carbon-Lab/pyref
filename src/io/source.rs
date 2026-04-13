//! FITS source and resolution: where to read from (file, paths, dir, catalog)
//! and whether to prefer catalog or disk when both exist.

use std::path::{Path, PathBuf};

use crate::errors::FitsError;

#[cfg(feature = "catalog")]
use crate::catalog::{discover_fits_paths, resolve_catalog_path as catalog_resolve_path};

/// When the source could be satisfied from catalog or disk (e.g. beamtime dir with `.pyref_catalog.db`),
/// this selects which to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResolvePreference {
    /// Use catalog if present, otherwise read from disk.
    #[default]
    PreferCatalog,
    /// Use disk; ignore catalog (e.g. after adding new files).
    PreferDisk,
    /// Use catalog only; fail if no catalog.
    FromCatalog,
    /// Read from disk only; do not use catalog.
    FromDisk,
}

/// Resolved outcome of source resolution: either read from SQLite catalog or from FITS on disk.
#[derive(Debug, Clone)]
pub enum ResolvedSource {
    /// Read metadata from this catalog DB path.
    FromCatalog { db_path: PathBuf },
    /// Read from these FITS file paths.
    FromDisk { paths: Vec<PathBuf> },
}

/// Identifies where FITS metadata should be read from.
#[derive(Debug, Clone)]
pub enum FitsSource {
    /// Single FITS file.
    File(PathBuf),
    /// Explicit list of FITS paths.
    Paths(Vec<PathBuf>),
    /// Directory: discover FITS files (and optionally use catalog if present).
    Dir(PathBuf),
    /// Path to ``catalog.db`` (typically ``parent/.pyref/catalog.db``) or a beamtime directory.
    Catalog(PathBuf),
}

impl FitsSource {
    /// Resolves this source with the given preference to either a catalog path or a list of disk paths.
    pub fn resolve(self, preference: ResolvePreference) -> Result<ResolvedSource, FitsError> {
        match self {
            FitsSource::File(p) => {
                if p.extension().and_then(|e| e.to_str()) != Some("fits") {
                    return Err(FitsError::validation("path is not a .fits file")
                        .with_context("path", p.display().to_string()));
                }
                Ok(ResolvedSource::FromDisk { paths: vec![p] })
            }
            FitsSource::Paths(paths) => {
                let mut out = Vec::with_capacity(paths.len());
                for p in paths {
                    if p.extension().and_then(|e| e.to_str()) == Some("fits") {
                        out.push(p);
                    }
                }
                Ok(ResolvedSource::FromDisk { paths: out })
            }
            FitsSource::Dir(dir) => resolve_dir(dir, preference),
            FitsSource::Catalog(path) => resolve_catalog_path(path, preference),
        }
    }
}

#[cfg(feature = "catalog")]
fn resolve_dir(dir: PathBuf, preference: ResolvePreference) -> Result<ResolvedSource, FitsError> {
    let catalog_db = catalog_resolve_path(&dir);
    let use_catalog = match preference {
        ResolvePreference::FromDisk => false,
        ResolvePreference::PreferDisk => false,
        ResolvePreference::FromCatalog => catalog_db.is_file(),
        ResolvePreference::PreferCatalog => catalog_db.is_file(),
    };
    if use_catalog && catalog_db.is_file() {
        return Ok(ResolvedSource::FromCatalog {
            db_path: catalog_db,
        });
    }
    if preference == ResolvePreference::FromCatalog {
        return Err(FitsError::not_found(format!(
            "catalog required but not found: {}",
            catalog_db.display()
        ))
        .with_context("operation", "resolve_dir")
        .with_context("path", dir.display().to_string()));
    }
    let discovered = discover_fits_paths(&dir).map_err(|e| {
        FitsError::validation(e.to_string())
            .with_context("operation", "resolve_dir")
            .with_context("path", dir.display().to_string())
    })?;
    let paths = discovered.into_iter().map(|(p, _)| p).collect();
    Ok(ResolvedSource::FromDisk { paths })
}

#[cfg(not(feature = "catalog"))]
fn resolve_dir(dir: PathBuf, preference: ResolvePreference) -> Result<ResolvedSource, FitsError> {
    if preference == ResolvePreference::FromCatalog
        || preference == ResolvePreference::PreferCatalog
    {
        return Err(
            FitsError::unsupported("catalog not available (catalog feature disabled)")
                .with_context("operation", "resolve_dir")
                .with_context("path", dir.display().to_string()),
        );
    }
    let paths = discover_fits_in_dir(&dir)?;
    Ok(ResolvedSource::FromDisk { paths })
}

#[cfg(feature = "catalog")]
fn resolve_catalog_path(
    path: PathBuf,
    _preference: ResolvePreference,
) -> Result<ResolvedSource, FitsError> {
    let db_path = if path.is_file() {
        if path.file_name().and_then(|n| n.to_str()) != Some("catalog.db") {
            return Err(FitsError::validation(
                "catalog source must be catalog.db or a beamtime directory",
            )
            .with_context("path", path.display().to_string()));
        }
        path
    } else if path.is_dir() {
        catalog_resolve_path(&path)
    } else {
        return Err(
            FitsError::not_found(format!("path does not exist: {}", path.display()))
                .with_context("operation", "resolve_catalog"),
        );
    };
    if !db_path.is_file() {
        return Err(
            FitsError::not_found(format!("catalog not found: {}", db_path.display()))
                .with_context("operation", "resolve_catalog"),
        );
    }
    Ok(ResolvedSource::FromCatalog { db_path })
}

#[cfg(not(feature = "catalog"))]
fn resolve_catalog_path(
    path: PathBuf,
    preference: ResolvePreference,
) -> Result<ResolvedSource, FitsError> {
    let _ = preference;
    let _ = path;
    Err(
        FitsError::unsupported("catalog not available (catalog feature disabled)")
            .with_context("operation", "resolve_catalog"),
    )
}

#[cfg(not(feature = "catalog"))]
fn discover_fits_in_dir(dir: &Path) -> Result<Vec<PathBuf>, FitsError> {
    if !dir.is_dir() {
        return Err(
            FitsError::not_found(format!("not a directory: {}", dir.display()))
                .with_context("operation", "discover_fits_in_dir"),
        );
    }
    let mut out = Vec::new();
    let entries = std::fs::read_dir(dir).map_err(|e| {
        FitsError::io("read_dir", e)
            .with_context("operation", "discover_fits_in_dir")
            .with_context("path", dir.display().to_string())
    })?;
    for entry in entries {
        let entry = entry.map_err(|e| FitsError::io("read_dir entry", e))?;
        let p = entry.path();
        if p.is_file() && p.extension().and_then(|e| e.to_str()) == Some("fits") {
            let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("");
            if stem.is_empty() || stem.starts_with('_') {
                continue;
            }
            out.push(p);
        }
    }
    out.sort();
    Ok(out)
}

impl From<PathBuf> for FitsSource {
    fn from(p: PathBuf) -> Self {
        if p.is_dir() {
            FitsSource::Dir(p)
        } else {
            FitsSource::File(p)
        }
    }
}

impl From<Vec<PathBuf>> for FitsSource {
    fn from(paths: Vec<PathBuf>) -> Self {
        FitsSource::Paths(paths)
    }
}

impl From<&Path> for FitsSource {
    fn from(p: &Path) -> Self {
        FitsSource::from(p.to_path_buf())
    }
}

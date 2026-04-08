//! Beamtime directory layout detection and FITS discovery per AGENTS.md.
//!
//! Detects nested (date / CCD Scan N / instrument) vs flat (beamtime / CCD or Axis Photonique).
//! Returns a structured error when neither layout matches.

use crate::catalog::{is_skippable_stem, CatalogError, Result};
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};

/// Recognized beamtime directory layout from AGENTS.md.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BeamtimeLayout {
    /// Date-grouped scan folders with `CCD Scan <n>` and `CCD` or `Axis Photonique` subdirs.
    Nested,
    /// Single `CCD` or `Axis Photonique` under beamtime root holding FITS from multiple scans.
    Flat,
}

fn is_fits_file(path: &Path) -> bool {
    path.is_file()
        && path
            .extension()
            .and_then(OsStr::to_str)
            .map(|e| e.eq_ignore_ascii_case("fits"))
            == Some(true)
}

fn instrument_subdir_name() -> &'static [&'static str] {
    &["CCD", "Axis Photonique"]
}

/// Inspects ``beamtime_root`` and returns which AGENTS layout applies.
///
/// Raises [`CatalogError::Validation`] when neither nested nor flat criteria match.
pub fn detect_beamtime_layout(beamtime_root: &Path) -> Result<BeamtimeLayout> {
    if !beamtime_root.is_dir() {
        return Err(CatalogError::Validation(format!(
            "not a directory: {}",
            beamtime_root.display()
        )));
    }

    if detect_nested_layout(beamtime_root) {
        return Ok(BeamtimeLayout::Nested);
    }
    if detect_flat_layout(beamtime_root) {
        return Ok(BeamtimeLayout::Flat);
    }

    Err(CatalogError::Validation(format!(
        "unrecognized beamtime directory layout (expected nested date/CCD Scan structure or flat CCD/Axis Photonique): {}",
        beamtime_root.display()
    )))
}

fn detect_nested_layout(root: &Path) -> bool {
    let Ok(entries) = fs::read_dir(root) else {
        return false;
    };
    for e in entries.flatten() {
        let date_dir = e.path();
        if !date_dir.is_dir() {
            continue;
        }
        let Ok(scan_entries) = fs::read_dir(&date_dir) else {
            continue;
        };
        for se in scan_entries.flatten() {
            let scan_dir = se.path();
            if !scan_dir.is_dir() {
                continue;
            }
            let name = scan_dir.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if !scan_dir_name_looks_like_ccd_scan(name) {
                continue;
            }
            for inst in instrument_subdir_name() {
                let p = scan_dir.join(inst);
                if p.is_dir() {
                    return true;
                }
            }
        }
    }
    false
}

fn scan_dir_name_looks_like_ccd_scan(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.contains("ccd scan") || lower.starts_with("ccd scan")
}

fn detect_flat_layout(root: &Path) -> bool {
    for inst in instrument_subdir_name() {
        let inst_dir = root.join(inst);
        if !inst_dir.is_dir() {
            continue;
        }
        if inst_dir.is_dir() {
            return true;
        }
    }
    false
}

/// Discovers FITS paths and mtimes using layout-specific rules.
pub fn discover_fits_for_layout(
    beamtime_root: &Path,
    layout: BeamtimeLayout,
) -> Result<Vec<(PathBuf, i64)>> {
    match layout {
        BeamtimeLayout::Nested => discover_nested_fits(beamtime_root),
        BeamtimeLayout::Flat => discover_flat_fits(beamtime_root),
    }
}

fn file_mtime(path: &Path) -> i64 {
    fs::metadata(path)
        .ok()
        .and_then(|m| m.modified().ok())
        .map(|t| {
            t.duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as i64
        })
        .unwrap_or(0)
}

fn push_canonical_fits(out: &mut Vec<(PathBuf, i64)>, path: &Path) -> Result<()> {
    if !is_fits_file(path) {
        return Ok(());
    }
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    if is_skippable_stem(stem) {
        return Ok(());
    }
    let path_buf = path.canonicalize()?;
    out.push((path_buf, file_mtime(path)));
    Ok(())
}

fn discover_nested_fits(root: &Path) -> Result<Vec<(PathBuf, i64)>> {
    let mut out: Vec<(PathBuf, i64)> = Vec::new();
    let Ok(date_entries) = fs::read_dir(root) else {
        return Ok(out);
    };
    for de in date_entries.flatten() {
        let date_dir = de.path();
        if !date_dir.is_dir() {
            continue;
        }
        let Ok(scan_entries) = fs::read_dir(&date_dir) else {
            continue;
        };
        for se in scan_entries.flatten() {
            let scan_dir = se.path();
            if !scan_dir.is_dir() {
                continue;
            }
            let name = scan_dir.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if !scan_dir_name_looks_like_ccd_scan(name) {
                continue;
            }
            for inst in instrument_subdir_name() {
                let inst_dir = scan_dir.join(inst);
                if !inst_dir.is_dir() {
                    continue;
                }
                let Ok(fits_entries) = fs::read_dir(&inst_dir) else {
                    continue;
                };
                for fe in fits_entries.flatten() {
                    let p = fe.path();
                    push_canonical_fits(&mut out, &p)?;
                }
            }
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out.dedup_by(|a, b| a.0 == b.0);
    Ok(out)
}

fn discover_flat_fits(root: &Path) -> Result<Vec<(PathBuf, i64)>> {
    let mut out: Vec<(PathBuf, i64)> = Vec::new();
    for inst in instrument_subdir_name() {
        let inst_dir = root.join(inst);
        if !inst_dir.is_dir() {
            continue;
        }
        let Ok(entries) = fs::read_dir(&inst_dir) else {
            continue;
        };
        for e in entries.flatten() {
            let p = e.path();
            push_canonical_fits(&mut out, &p)?;
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out.dedup_by(|a, b| a.0 == b.0);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn detect_flat_layout_ccd_multi_scan() {
        let tmp = TempDir::new().unwrap();
        let ccd = tmp.path().join("CCD");
        fs::create_dir_all(&ccd).unwrap();
        fs::write(ccd.join("a_00001-00001.fits"), b"X").unwrap();
        fs::write(ccd.join("b_00002-00001.fits"), b"X").unwrap();
        assert_eq!(
            detect_beamtime_layout(tmp.path()).unwrap(),
            BeamtimeLayout::Flat
        );
    }

    #[test]
    fn detect_nested_layout_minimal() {
        let tmp = TempDir::new().unwrap();
        let scan = tmp.path().join("20240101").join("CCD Scan 1").join("CCD");
        fs::create_dir_all(&scan).unwrap();
        fs::write(scan.join("s_00001-00001.fits"), b"X").unwrap();
        assert_eq!(
            detect_beamtime_layout(tmp.path()).unwrap(),
            BeamtimeLayout::Nested
        );
    }

    #[test]
    fn detect_unknown_errors() {
        let tmp = TempDir::new().unwrap();
        fs::create_dir_all(tmp.path().join("empty")).unwrap();
        assert!(detect_beamtime_layout(tmp.path()).is_err());
    }
}

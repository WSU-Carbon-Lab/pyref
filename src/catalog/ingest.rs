#![cfg(feature = "catalog")]

use crate::catalog::{open_or_create_db, CatalogError, Result};
use crate::io::add_calculated_domains;
use crate::loader::read_multiple_fits_headers_only;
use polars::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

const BATCH_SIZE: usize = 500;

pub fn ingest_beamtime(
    beamtime_dir: &Path,
    header_items: &[String],
    incremental: bool,
) -> Result<PathBuf> {
    let db_path = beamtime_dir.join(super::CATALOG_DB_NAME);
    let conn = open_or_create_db(beamtime_dir)?;
    let discovered = super::discover_fits_paths(beamtime_dir)?;
    let path_to_mtime: HashMap<String, i64> = discovered
        .iter()
        .map(|(p, m)| (p.to_string_lossy().to_string(), *m))
        .collect();
    let paths: Vec<PathBuf> = discovered.into_iter().map(|(p, _)| p).collect();

    let to_ingest: Vec<PathBuf> = if incremental {
        let existing: HashMap<String, i64> = conn
            .prepare("SELECT path, mtime FROM files")?
            .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))?
            .filter_map(|r| r.ok())
            .collect();
        paths
            .into_iter()
            .filter(|p| {
                let key = p.to_string_lossy().to_string();
                let mtime = path_to_mtime.get(&key).copied().unwrap_or(0);
                existing.get(&key).map_or(true, |&stored| mtime > stored)
            })
            .collect()
    } else {
        paths
    };

    for chunk in to_ingest.chunks(BATCH_SIZE) {
        let chunk_vec: Vec<PathBuf> = chunk.to_vec();
        let df = read_multiple_fits_headers_only(chunk_vec, header_items)
            .map_err(|e| CatalogError::Validation(e.to_string()))?;
        let with_domains = add_calculated_domains(df.lazy());
        upsert_files_batch(&conn, &with_domains, &path_to_mtime)?;
    }

    let path_list: Vec<&str> = path_to_mtime.keys().map(|s| s.as_str()).collect();
    prune_missing_files(&conn, &path_list)?;

    Ok(db_path)
}

fn upsert_files_batch(
    conn: &rusqlite::Connection,
    df: &DataFrame,
    path_to_mtime: &HashMap<String, i64>,
) -> Result<()> {
    let n = df.height();
    if n == 0 {
        return Ok(());
    }
    let file_path_col = df.column("file_path").map_err(|e| CatalogError::Validation(e.to_string()))?.str().map_err(|e| CatalogError::Validation(e.to_string()))?;
    let get_str = |name: &str| -> Result<Vec<Option<String>>> {
        match df.column(name) {
            Ok(c) => Ok(c.str().map_err(|e| CatalogError::Validation(e.to_string()))?.iter().map(|s| s.map(|v| v.to_string())).collect()),
            _ => Ok(std::iter::repeat(None).take(n).collect()),
        }
    };
    let get_i64 = |name: &str| -> Result<Vec<Option<i64>>> {
        match df.column(name) {
            Ok(c) => Ok(c.i64().map_err(|e| CatalogError::Validation(e.to_string()))?.iter().collect()),
            _ => Ok(std::iter::repeat(None).take(n).collect()),
        }
    };
    let get_f64 = |name: &str| -> Result<Vec<Option<f64>>> {
        match df.column(name) {
            Ok(c) => Ok(c.f64().map_err(|e| CatalogError::Validation(e.to_string()))?.iter().collect()),
            _ => Ok(std::iter::repeat(None).take(n).collect()),
        }
    };

    let data_offset = get_i64("data_offset")?;
    let naxis1 = get_i64("naxis1")?;
    let naxis2 = get_i64("naxis2")?;
    let bitpix = get_i64("bitpix")?;
    let bzero = get_i64("bzero")?;
    let data_size = get_i64("data_size")?;
    let file_name = get_str("file_name")?;
    let sample_name = get_str("sample_name")?;
    let tag = get_str("tag")?;
    let experiment_number = get_i64("experiment_number")?;
    let frame_number = get_i64("frame_number")?;
    let date = get_str("DATE")?;
    let beamline_energy = get_f64("Beamline Energy")?;
    let sample_theta = get_f64("Sample Theta")?;
    let ccd_theta = get_f64("CCD Theta")?;
    let hos = get_f64("Higher Order Suppressor")?;
    let epu = get_f64("EPU Polarization")?;
    let exposure = get_f64("EXPOSURE")?;
    let sample_name_h = get_str("Sample Name")?;
    let scan_id = get_f64("Scan ID")?;
    let lambda = get_f64("Lambda")?;
    let q = get_f64("Q")?;

    let mut stmt = conn.prepare_cached(
        r#"
        INSERT OR REPLACE INTO files (
            path, mtime, file_path, data_offset, naxis1, naxis2, bitpix, bzero, data_size,
            file_name, sample_name, tag, experiment_number, frame_number,
            "DATE", "Beamline Energy", "Sample Theta", "CCD Theta", "Higher Order Suppressor",
            "EPU Polarization", EXPOSURE, "Sample Name", "Scan ID", Lambda, Q
        ) VALUES (
            ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14,
            ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?24, ?25
        )"#,
    )?;

    for i in 0..n {
        let path_str = file_path_col.get(i).unwrap_or("").to_string();
        let mtime = path_to_mtime.get(&path_str).copied().unwrap_or(0);
        stmt.execute(rusqlite::params![
            path_str,
            mtime,
            path_str,
            data_offset.get(i).copied().flatten().unwrap_or(0),
            naxis1.get(i).copied().flatten().unwrap_or(0),
            naxis2.get(i).copied().flatten().unwrap_or(0),
            bitpix.get(i).copied().flatten().unwrap_or(0),
            bzero.get(i).copied().flatten().unwrap_or(0),
            data_size.get(i).copied().flatten().unwrap_or(0),
            file_name.get(i).cloned().unwrap_or_default(),
            sample_name.get(i).cloned().unwrap_or_default(),
            tag.get(i).clone(),
            experiment_number.get(i).copied().flatten().unwrap_or(0),
            frame_number.get(i).copied().flatten().unwrap_or(0),
            date.get(i).clone(),
            beamline_energy.get(i).copied().flatten(),
            sample_theta.get(i).copied().flatten(),
            ccd_theta.get(i).copied().flatten(),
            hos.get(i).copied().flatten(),
            epu.get(i).copied().flatten(),
            exposure.get(i).copied().flatten(),
            sample_name_h.get(i).clone(),
            scan_id.get(i).copied().flatten(),
            lambda.get(i).copied().flatten(),
            q.get(i).copied().flatten(),
        ])?;
    }
    Ok(())
}

fn prune_missing_files(conn: &rusqlite::Connection, known_paths: &[&str]) -> Result<()> {
    if known_paths.is_empty() {
        conn.execute("DELETE FROM files", [])?;
        return Ok(());
    }
    let placeholders = known_paths
        .iter()
        .enumerate()
        .map(|(i, _)| format!("?{}", i + 1))
        .collect::<Vec<_>>()
        .join(",");
    let sql = format!("DELETE FROM files WHERE path NOT IN ({})", placeholders);
    conn.execute(&sql, rusqlite::params_from_iter(known_paths.iter()))?;
    Ok(())
}

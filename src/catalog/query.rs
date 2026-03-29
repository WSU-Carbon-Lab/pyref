#![cfg(feature = "catalog")]

use crate::catalog::{open_catalog_db, CatalogError, Result};
use polars::prelude::*;
use std::path::Path;

#[derive(Default, Debug, Clone)]
pub struct CatalogFilter {
    pub sample_name: Option<String>,
    pub tag: Option<String>,
    pub scan_numbers: Option<Vec<i64>>,
    pub energy_min: Option<f64>,
    pub energy_max: Option<f64>,
}

pub struct BeamtimeEntries {
    pub samples: Vec<String>,
    pub tags: Vec<String>,
    pub scans: Vec<(i64, String)>,
}

#[derive(Debug, Clone)]
pub struct FileRow {
    pub file_path: String,
    pub sample_name: String,
    pub tag: Option<String>,
    pub scan_number: i64,
    pub frame_number: i64,
    pub beamline_energy: Option<f64>,
    pub sample_theta: Option<f64>,
    pub epu_polarization: Option<f64>,
    pub q: Option<f64>,
    pub date_iso: Option<String>,
    pub beam_row: Option<i64>,
    pub beam_col: Option<i64>,
    pub beam_sigma: Option<f64>,
    pub scan_point_uid: Option<String>,
}

fn beamtime_id_from_path(conn: &rusqlite::Connection, beamtime_dir: &Path) -> Result<Option<i64>> {
    let path_str = beamtime_dir.to_string_lossy().to_string();
    let id: Option<i64> = match conn.query_row(
        "SELECT id FROM bt_beamtimes WHERE beamtime_path = ?1",
        rusqlite::params![path_str],
        |r| r.get(0),
    ) {
        Ok(x) => Some(x),
        Err(rusqlite::Error::QueryReturnedNoRows) => None,
        Err(e) => return Err(e.into()),
    };
    Ok(id)
}

pub fn catalog_file_count(db_path: &Path, beamtime_dir: Option<&Path>) -> Result<u32> {
    let conn = open_catalog_db(db_path)?;
    let count: i64 = if let Some(dir) = beamtime_dir {
        if super::is_new_catalog_layout(dir) {
            let path_str = dir.to_string_lossy().to_string();
            conn.query_row(
                "SELECT COUNT(*) FROM bt_scan_points sp
                 JOIN bt_scans sc ON sp.scan_uid = sc.uid
                 JOIN bt_beamtimes b ON sc.beamtime_id = b.id
                 WHERE b.beamtime_path = ?1 AND sp.source_path IS NOT NULL",
                rusqlite::params![path_str],
                |r| r.get(0),
            )
            .unwrap_or(0)
        } else {
            conn.query_row("SELECT COUNT(*) FROM files", [], |r| r.get(0)).unwrap_or(0)
        }
    } else {
        conn.query_row("SELECT COUNT(*) FROM files", [], |r| r.get(0)).unwrap_or(0)
    };
    Ok(count as u32)
}

pub fn list_beamtimes_from_catalog(db_path: &Path) -> Result<Vec<(std::path::PathBuf, i64)>> {
    use std::path::PathBuf;
    let conn = open_catalog_db(db_path)?;
    let mut stmt = conn.prepare(
        "SELECT beamtime_path, id FROM bt_beamtimes ORDER BY id DESC",
    )?;
    let rows = stmt.query_map([], |r| {
        Ok((PathBuf::from(r.get::<_, String>(0)?), r.get::<_, i64>(1)?))
    })?;
    rows.map(|r| r.map_err(CatalogError::from)).collect()
}

pub fn list_beamtime_entries(db_path: &Path) -> Result<BeamtimeEntries> {
    let conn = open_catalog_db(db_path)?;
    let mut samples: Vec<String> = conn
        .prepare(
            r#"SELECT DISTINCT COALESCE(o.sample_name, f.sample_name) FROM files f
               LEFT JOIN overrides o ON f.path = o.path
               WHERE COALESCE(o.sample_name, f.sample_name) != '' ORDER BY 1"#,
        )?
        .query_map([], |r| r.get::<_, String>(0))?
        .filter_map(|r| r.ok())
        .collect();
    let mut tags: Vec<String> = conn
        .prepare(
            r#"SELECT DISTINCT COALESCE(o.tag, f.tag) FROM files f
               LEFT JOIN overrides o ON f.path = o.path
               WHERE COALESCE(o.tag, f.tag) IS NOT NULL AND COALESCE(o.tag, f.tag) != ''
               ORDER BY COALESCE(o.tag, f.tag)"#,
        )?
        .query_map([], |r| r.get::<_, String>(0))?
        .filter_map(|r| r.ok())
        .collect();
    let scans: Vec<(i64, String)> = conn
        .prepare(
            r#"SELECT DISTINCT f.scan_number FROM files f
               WHERE f.scan_number != 0 ORDER BY f.scan_number"#,
        )?
        .query_map([], |r| {
            let n: i64 = r.get(0)?;
            Ok((n, format!("Scan {}", n)))
        })?
        .filter_map(|r| r.ok())
        .collect();
    samples.sort();
    tags.sort();
    Ok(BeamtimeEntries {
        samples,
        tags,
        scans,
    })
}

fn scan_number_from_uid(scan_uid: &str) -> i64 {
    scan_uid
        .rsplit_once('_')
        .and_then(|(_, tail)| tail.parse::<i64>().ok())
        .unwrap_or(0)
}

pub fn list_beamtime_entries_v2(
    db_path: &Path,
    beamtime_dir: &Path,
) -> Result<BeamtimeEntries> {
    let conn = open_catalog_db(db_path)?;
    let beamtime_id = match beamtime_id_from_path(&conn, beamtime_dir)? {
        Some(id) => id,
        None => {
            return Ok(BeamtimeEntries {
                samples: Vec::new(),
                tags: Vec::new(),
                scans: Vec::new(),
            })
        }
    };
    let mut samples: Vec<String> = conn
        .prepare(
            "SELECT DISTINCT name FROM bt_samples WHERE beamtime_id = ?1 AND name != '' ORDER BY 1",
        )?
        .query_map(rusqlite::params![beamtime_id], |r| r.get::<_, String>(0))?
        .filter_map(|r| r.ok())
        .collect();
    let mut tags: Vec<String> = conn
        .prepare(
            "SELECT DISTINCT tag FROM bt_samples WHERE beamtime_id = ?1 AND tag IS NOT NULL AND tag != '' ORDER BY tag",
        )?
        .query_map(rusqlite::params![beamtime_id], |r| r.get::<_, String>(0))?
        .filter_map(|r| r.ok())
        .collect();
    let scans: Vec<(i64, String)> = conn
        .prepare("SELECT uid FROM bt_scans WHERE beamtime_id = ?1 ORDER BY uid")?
        .query_map(rusqlite::params![beamtime_id], |r| {
            let uid: String = r.get(0)?;
            let n = scan_number_from_uid(&uid);
            Ok((n, format!("Scan {}", n)))
        })?
        .filter_map(|r| r.ok())
        .collect();
    samples.sort();
    tags.sort();
    Ok(BeamtimeEntries {
        samples,
        tags,
        scans,
    })
}

pub fn query_files(db_path: &Path, filter: Option<&CatalogFilter>) -> Result<Vec<FileRow>> {
    let (where_clause, params) = filter
        .map(build_where_and_params)
        .unwrap_or_else(|| (String::new(), Vec::new()));
    let conn = open_catalog_db(db_path)?;
    let sql = format!(
        r#"SELECT f.file_path,
            COALESCE(o.sample_name, f.sample_name),
            COALESCE(o.tag, f.tag),
            f.scan_number, f.frame_number,
            f."Beamline Energy", f."Sample Theta", f."EPU Polarization", f.Q,
            f."DATE", f.beam_row, f.beam_col, f.beam_sigma
            FROM files f LEFT JOIN overrides o ON f.path = o.path{}
            ORDER BY f.scan_number, f.frame_number"#,
        where_clause
    );
    let mut stmt = conn.prepare(&sql)?;
    let map_row = |row: &rusqlite::Row| {
        Ok(FileRow {
            file_path: row.get(0)?,
            sample_name: row.get(1)?,
            tag: row.get(2)?,
            scan_number: row.get(3)?,
            frame_number: row.get(4)?,
            beamline_energy: row.get(5)?,
            sample_theta: row.get(6)?,
            epu_polarization: row.get(7)?,
            q: row.get(8)?,
            date_iso: row.get(9)?,
            beam_row: row.get(10)?,
            beam_col: row.get(11)?,
            beam_sigma: row.get(12)?,
            scan_point_uid: None,
        })
    };
    let rows = if params.is_empty() {
        stmt.query_map([], map_row)?
    } else {
        let pr: Vec<&dyn rusqlite::ToSql> = params.iter().map(|b| b.as_ref()).collect();
        stmt.query_map(rusqlite::params_from_iter(pr), map_row)?
    };
    rows.map(|r| r.map_err(CatalogError::Sqlite)).collect()
}

const SCAN_POINTS_QUERY: &str = r#"
SELECT
    sp.uid AS scan_point_uid,
    sp.source_path, sp.source_data_offset, sp.source_naxis1, sp.source_naxis2,
    sp.source_bitpix, sp.source_bzero, sp.seq_index,
    sp.beamline_energy, sp.sample_theta, sp.epu_polarization, sp.ccd_theta, sp.exposure,
    sp.beam_row, sp.beam_col, sp.beam_sigma,
    COALESCE(o.sample_name, s.name) AS sample_name, COALESCE(o.tag, s.tag) AS tag,
    sc.uid AS scan_uid
FROM bt_scan_points sp
JOIN bt_samples s ON sp.sample_id = s.id
JOIN bt_scans sc ON sp.scan_uid = sc.uid
LEFT JOIN bt_file_overrides o ON o.source_path = sp.source_path
WHERE sc.beamtime_id = ?1
"#;

fn build_scan_points_where_and_params(
    filter: Option<&CatalogFilter>,
    beamtime_id: i64,
    param_start: usize,
) -> (String, Vec<Box<dyn rusqlite::ToSql + '_>>) {
    let Some(filter) = filter else {
        return (String::new(), Vec::new());
    };
    let mut conditions: Vec<String> = Vec::new();
    let mut params: Vec<Box<dyn rusqlite::ToSql + '_>> = Vec::new();
    let mut idx = param_start;
    if let Some(ref s) = filter.sample_name {
        idx += 1;
        conditions.push(format!("COALESCE(o.sample_name, s.name) = ?{}", idx));
        params.push(Box::new(s.clone()));
    }
    if let Some(ref t) = filter.tag {
        idx += 1;
        conditions.push(format!("COALESCE(o.tag, s.tag) = ?{}", idx));
        params.push(Box::new(t.clone()));
    }
    if let Some(ref scan_nos) = filter.scan_numbers {
        if !scan_nos.is_empty() {
            let placeholders: Vec<String> = (0..scan_nos.len())
                .map(|_| {
                    idx += 1;
                    format!("?{}", idx)
                })
                .collect();
            conditions.push(format!("sc.uid IN ({})", placeholders.join(",")));
            for n in scan_nos {
                params.push(Box::new(format!("s_{}_{}", beamtime_id, n)));
            }
        }
    }
    if let Some(em) = filter.energy_min {
        idx += 1;
        conditions.push(format!("sp.beamline_energy >= ?{}", idx));
        params.push(Box::new(em));
    }
    if let Some(em) = filter.energy_max {
        idx += 1;
        conditions.push(format!("sp.beamline_energy <= ?{}", idx));
        params.push(Box::new(em));
    }
    let where_clause = if conditions.is_empty() {
        String::new()
    } else {
        format!(" AND {}", conditions.join(" AND "))
    };
    (where_clause, params)
}

pub fn query_scan_points(
    db_path: &Path,
    beamtime_dir: &Path,
    filter: Option<&CatalogFilter>,
) -> Result<Vec<FileRow>> {
    let conn = open_catalog_db(db_path)?;
    let beamtime_id = match beamtime_id_from_path(&conn, beamtime_dir)? {
        Some(id) => id,
        None => return Ok(Vec::new()),
    };
    let (extra_where, mut params) = build_scan_points_where_and_params(filter, beamtime_id, 1);
    let order_clause = " ORDER BY sc.uid, sp.seq_index";
    let sql = format!("{}{}{}", SCAN_POINTS_QUERY, extra_where, order_clause);
    let mut stmt = conn.prepare(&sql)?;
    let mut query_params: Vec<Box<dyn rusqlite::ToSql + '_>> = vec![Box::new(beamtime_id)];
    query_params.append(&mut params);
    let map_row = |row: &rusqlite::Row| {
        let scan_uid: String = row.get(18)?;
        let scan_no = scan_number_from_uid(&scan_uid);
        Ok(FileRow {
            file_path: row.get::<_, Option<String>>(1)?.unwrap_or_default(),
            sample_name: row.get(16)?,
            tag: row.get(17)?,
            scan_number: scan_no,
            frame_number: row.get(7)?,
            beamline_energy: row.get(8)?,
            sample_theta: row.get(9)?,
            epu_polarization: row.get(10)?,
            q: None,
            date_iso: None,
            beam_row: row.get(13)?,
            beam_col: row.get(14)?,
            beam_sigma: row.get(15)?,
            scan_point_uid: Some(row.get(0)?),
        })
    };
    let param_refs: Vec<&dyn rusqlite::ToSql> = query_params.iter().map(|b| b.as_ref()).collect();
    let rows = stmt.query_map(rusqlite::params_from_iter(param_refs), map_row)?;
    rows.map(|r| r.map_err(CatalogError::Sqlite)).collect()
}

pub fn update_beamspot(
    db_path: &Path,
    file_path: &str,
    beam_row: i64,
    beam_col: i64,
    beam_sigma: Option<f64>,
) -> Result<()> {
    let conn = open_catalog_db(db_path)?;
    match beam_sigma {
        Some(sigma) => {
            conn.execute(
                "UPDATE files SET beam_row = ?1, beam_col = ?2, beam_sigma = ?3 WHERE path = ?4",
                rusqlite::params![beam_row, beam_col, sigma, file_path],
            )?;
        }
        None => {
            conn.execute(
                "UPDATE files SET beam_row = ?1, beam_col = ?2 WHERE path = ?3",
                rusqlite::params![beam_row, beam_col, file_path],
            )?;
        }
    }
    Ok(())
}

pub fn update_beamspot_scan_point(
    db_path: &Path,
    scan_point_uid: &str,
    beam_row: i64,
    beam_col: i64,
    beam_sigma: Option<f64>,
) -> Result<()> {
    let conn = open_catalog_db(db_path)?;
    match beam_sigma {
        Some(sigma) => {
            conn.execute(
                "UPDATE bt_scan_points SET beam_row = ?1, beam_col = ?2, beam_sigma = ?3 WHERE uid = ?4",
                rusqlite::params![beam_row, beam_col, sigma, scan_point_uid],
            )?;
        }
        None => {
            conn.execute(
                "UPDATE bt_scan_points SET beam_row = ?1, beam_col = ?2 WHERE uid = ?3",
                rusqlite::params![beam_row, beam_col, scan_point_uid],
            )?;
        }
    }
    Ok(())
}

pub fn get_scan_point_uid_by_source_path(
    db_path: &Path,
    source_path: &str,
) -> Result<Option<String>> {
    let conn = open_catalog_db(db_path)?;
    let uid: Option<String> = match conn.query_row(
        "SELECT uid FROM bt_scan_points WHERE source_path = ?1 LIMIT 1",
        rusqlite::params![source_path],
        |r| r.get(0),
    ) {
        Ok(u) => Some(u),
        Err(rusqlite::Error::QueryReturnedNoRows) => None,
        Err(e) => return Err(e.into()),
    };
    Ok(uid)
}

const RESOLVED_QUERY: &str = r#"
SELECT
    f.file_path, f.data_offset, f.naxis1, f.naxis2, f.bitpix, f.bzero, f.data_size,
    f.file_name,
    COALESCE(o.sample_name, f.sample_name) AS sample_name,
    COALESCE(o.tag, f.tag) AS tag,
    f.scan_number, f.frame_number,
    f."DATE", f."Beamline Energy", f."Sample Theta", f."CCD Theta",
    f."Higher Order Suppressor", f."EPU Polarization", f.EXPOSURE,
    f."Sample Name", f."Scan ID", f.Lambda, f.Q,
    f.beam_row, f.beam_col, f.beam_sigma
FROM files f
LEFT JOIN overrides o ON f.path = o.path
"#;

fn build_where_and_params(filter: &CatalogFilter) -> (String, Vec<Box<dyn rusqlite::ToSql + '_>>) {
    let mut conditions: Vec<String> = Vec::new();
    let mut params: Vec<Box<dyn rusqlite::ToSql + '_>> = Vec::new();
    if let Some(ref s) = filter.sample_name {
        conditions.push("COALESCE(o.sample_name, f.sample_name) = ?1".to_string());
        params.push(Box::new(s.clone()));
    }
    if let Some(ref t) = filter.tag {
        let idx = params.len() + 1;
        conditions.push(format!("COALESCE(o.tag, f.tag) = ?{}", idx));
        params.push(Box::new(t.clone()));
    }
    if let Some(ref scan_nos) = filter.scan_numbers {
        if !scan_nos.is_empty() {
            let placeholders: Vec<String> = (0..scan_nos.len())
                .map(|i| format!("?{}", params.len() + 1 + i))
                .collect();
            conditions.push(format!("f.scan_number IN ({})", placeholders.join(",")));
            for s in scan_nos {
                params.push(Box::new(*s));
            }
        }
    }
    if let Some(em) = filter.energy_min {
        let idx = params.len() + 1;
        conditions.push(format!(r#"f."Beamline Energy" >= ?{}"#, idx));
        params.push(Box::new(em));
    }
    if let Some(em) = filter.energy_max {
        let idx = params.len() + 1;
        conditions.push(format!(r#"f."Beamline Energy" <= ?{}"#, idx));
        params.push(Box::new(em));
    }
    let where_clause = if conditions.is_empty() {
        String::new()
    } else {
        format!(" WHERE {}", conditions.join(" AND "))
    };
    (where_clause, params)
}

const BT_SCAN_POINTS_DF_QUERY: &str = r#"
SELECT
    sp.source_path, sp.source_data_offset, sp.source_naxis1, sp.source_naxis2,
    sp.source_bitpix, sp.source_bzero,
    (CASE WHEN sp.source_naxis1 IS NOT NULL AND sp.source_naxis2 IS NOT NULL AND sp.source_bitpix IS NOT NULL
      THEN sp.source_naxis1 * sp.source_naxis2 * max(1, abs(sp.source_bitpix) / 8) ELSE 0 END) AS data_size,
    COALESCE(o.sample_name, s.name) AS sample_name, COALESCE(o.tag, s.tag) AS tag, sc.uid AS scan_uid, sp.seq_index,
    sp.beamline_energy, sp.sample_theta, sp.ccd_theta, sp.epu_polarization, sp.exposure,
    sp.beam_row, sp.beam_col, sp.beam_sigma
FROM bt_scan_points sp
JOIN bt_samples s ON sp.sample_id = s.id
JOIN bt_scans sc ON sp.scan_uid = sc.uid
LEFT JOIN bt_file_overrides o ON o.source_path = sp.source_path
"#;

fn scan_from_catalog_bt(
    conn: &rusqlite::Connection,
    filter: Option<&CatalogFilter>,
) -> Result<DataFrame> {
    let (where_clause, params) = build_bt_df_where_and_params(filter);
    let order_clause = " ORDER BY sc.uid, sp.seq_index";
    let sql = format!("{}{}{}", BT_SCAN_POINTS_DF_QUERY, where_clause, order_clause);
    let mut stmt = conn.prepare(&sql)?;
    let row_from_query = |row: &rusqlite::Row| {
        Ok((
            row.get::<_, Option<String>>(0)?,
            row.get::<_, Option<i64>>(1)?,
            row.get::<_, Option<i64>>(2)?,
            row.get::<_, Option<i64>>(3)?,
            row.get::<_, Option<i64>>(4)?,
            row.get::<_, Option<i64>>(5)?,
            row.get::<_, i64>(6)?,
            row.get::<_, String>(7)?,
            row.get::<_, Option<String>>(8)?,
            row.get::<_, String>(9)?,
            row.get::<_, i64>(10)?,
            row.get::<_, Option<f64>>(11)?,
            row.get::<_, Option<f64>>(12)?,
            row.get::<_, Option<f64>>(13)?,
            row.get::<_, Option<f64>>(14)?,
            row.get::<_, Option<f64>>(15)?,
            row.get::<_, Option<i64>>(16)?,
            row.get::<_, Option<i64>>(17)?,
            row.get::<_, Option<f64>>(18)?,
        ))
    };
    let rows = if params.is_empty() {
        stmt.query_map([], row_from_query)?
    } else {
        let param_refs: Vec<&dyn rusqlite::ToSql> = params.iter().map(|b| b.as_ref()).collect();
        stmt.query_map(rusqlite::params_from_iter(param_refs), row_from_query)?
    };

    let mut file_path = Vec::new();
    let mut data_offset = Vec::new();
    let mut naxis1 = Vec::new();
    let mut naxis2 = Vec::new();
    let mut bitpix = Vec::new();
    let mut bzero = Vec::new();
    let mut data_size = Vec::new();
    let mut file_name = Vec::new();
    let mut sample_name = Vec::new();
    let mut tag = Vec::new();
    let mut scan_number = Vec::new();
    let mut frame_number = Vec::new();
    let mut date: Vec<Option<String>> = Vec::new();
    let mut beamline_energy = Vec::new();
    let mut sample_theta = Vec::new();
    let mut ccd_theta = Vec::new();
    let mut hos: Vec<Option<f64>> = Vec::new();
    let mut epu = Vec::new();
    let mut exposure = Vec::new();
    let mut sample_name_h: Vec<Option<String>> = Vec::new();
    let mut scan_id = Vec::new();
    let mut lambda: Vec<Option<f64>> = Vec::new();
    let mut q: Vec<Option<f64>> = Vec::new();
    let mut beam_row = Vec::new();
    let mut beam_col = Vec::new();
    let mut beam_sigma = Vec::new();

    for row in rows {
        let r = row.map_err(CatalogError::Sqlite)?;
        let path_str = r.0.unwrap_or_default();
        let scan_no = scan_number_from_uid(&r.9);
        let fname = std::path::Path::new(&path_str)
            .file_name()
            .and_then(|o| o.to_str())
            .unwrap_or("")
            .to_string();
        file_path.push(path_str.clone());
        data_offset.push(r.1.unwrap_or(0));
        naxis1.push(r.2.unwrap_or(0));
        naxis2.push(r.3.unwrap_or(0));
        bitpix.push(r.4.unwrap_or(0));
        bzero.push(r.5.unwrap_or(0));
        data_size.push(r.6);
        file_name.push(fname);
        sample_name.push(r.7.clone());
        tag.push(r.8);
        scan_number.push(scan_no);
        frame_number.push(r.10);
        date.push(None);
        beamline_energy.push(r.11);
        sample_theta.push(r.12);
        ccd_theta.push(r.13);
        hos.push(None);
        epu.push(r.14);
        exposure.push(r.15);
        sample_name_h.push(Some(r.7));
        scan_id.push(Some(scan_no as f64));
        lambda.push(None);
        q.push(None);
        beam_row.push(r.16);
        beam_col.push(r.17);
        beam_sigma.push(r.18);
    }

    let series = vec![
        Series::new("file_path".into(), file_path),
        Series::new("data_offset".into(), data_offset),
        Series::new("naxis1".into(), naxis1),
        Series::new("naxis2".into(), naxis2),
        Series::new("bitpix".into(), bitpix),
        Series::new("bzero".into(), bzero),
        Series::new("data_size".into(), data_size),
        Series::new("file_name".into(), file_name),
        Series::new("sample_name".into(), sample_name),
        Series::new("tag".into(), tag),
        Series::new("scan_number".into(), scan_number),
        Series::new("frame_number".into(), frame_number),
        Series::new("DATE".into(), date),
        Series::new("Beamline Energy".into(), beamline_energy),
        Series::new("Sample Theta".into(), sample_theta),
        Series::new("CCD Theta".into(), ccd_theta),
        Series::new("Higher Order Suppressor".into(), hos),
        Series::new("EPU Polarization".into(), epu),
        Series::new("EXPOSURE".into(), exposure),
        Series::new("Sample Name".into(), sample_name_h),
        Series::new("Scan ID".into(), scan_id),
        Series::new("Lambda".into(), lambda),
        Series::new("Q".into(), q),
        Series::new("beam_row".into(), beam_row),
        Series::new("beam_col".into(), beam_col),
        Series::new("beam_sigma".into(), beam_sigma),
    ];
    let columns: Vec<Column> = series.into_iter().map(|s| s.into()).collect();
    DataFrame::new(columns).map_err(|e| CatalogError::Validation(e.to_string()))
}

fn build_bt_df_where_and_params(
    filter: Option<&CatalogFilter>,
) -> (String, Vec<Box<dyn rusqlite::ToSql + '_>>) {
    let Some(filter) = filter else {
        return (String::new(), Vec::new());
    };
    let mut conditions: Vec<String> = Vec::new();
    let mut params: Vec<Box<dyn rusqlite::ToSql + '_>> = Vec::new();
    let mut idx = 0;
    if let Some(ref s) = filter.sample_name {
        idx += 1;
        conditions.push(format!("COALESCE(o.sample_name, s.name) = ?{}", idx));
        params.push(Box::new(s.clone()));
    }
    if let Some(ref t) = filter.tag {
        idx += 1;
        conditions.push(format!("COALESCE(o.tag, s.tag) = ?{}", idx));
        params.push(Box::new(t.clone()));
    }
    if let Some(em) = filter.energy_min {
        idx += 1;
        conditions.push(format!("sp.beamline_energy >= ?{}", idx));
        params.push(Box::new(em));
    }
    if let Some(em) = filter.energy_max {
        idx += 1;
        conditions.push(format!("sp.beamline_energy <= ?{}", idx));
        params.push(Box::new(em));
    }
    if let Some(ref scan_nos) = filter.scan_numbers {
        if !scan_nos.is_empty() {
            let or_parts: Vec<String> = scan_nos
                .iter()
                .map(|_| {
                    idx += 1;
                    format!("sc.uid LIKE '%' || ?{}", idx)
                })
                .collect();
            conditions.push(format!("({})", or_parts.join(" OR ")));
            for n in scan_nos {
                params.push(Box::new(format!("_{}", n)));
            }
        }
    }
    let where_clause = if conditions.is_empty() {
        String::new()
    } else {
        format!(" WHERE {}", conditions.join(" AND "))
    };
    (where_clause, params)
}

pub fn scan_from_catalog(db_path: &Path, filter: Option<&CatalogFilter>) -> Result<DataFrame> {
    let conn = open_catalog_db(db_path)?;
    let use_bt: bool = conn.query_row(
        "SELECT COUNT(*) FROM bt_scan_points",
        [],
        |r| r.get::<_, i64>(0),
    )? > 0;
    if use_bt {
        return scan_from_catalog_bt(&conn, filter);
    }
    let (where_clause, params) = filter
        .map(build_where_and_params)
        .unwrap_or_else(|| (String::new(), Vec::new()));
    let order_clause = " ORDER BY f.scan_number, f.frame_number";
    let sql = format!("{}{}{}", RESOLVED_QUERY, where_clause, order_clause);

    let mut stmt = conn.prepare(&sql)?;
    let row_from_query = |row: &rusqlite::Row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, i64>(1)?,
            row.get::<_, i64>(2)?,
            row.get::<_, i64>(3)?,
            row.get::<_, i64>(4)?,
            row.get::<_, i64>(5)?,
            row.get::<_, i64>(6)?,
            row.get::<_, String>(7)?,
            row.get::<_, String>(8)?,
            row.get::<_, Option<String>>(9)?,
            row.get::<_, i64>(10)?,
            row.get::<_, i64>(11)?,
            row.get::<_, Option<String>>(12)?,
            row.get::<_, Option<f64>>(13)?,
            row.get::<_, Option<f64>>(14)?,
            row.get::<_, Option<f64>>(15)?,
            row.get::<_, Option<f64>>(16)?,
            row.get::<_, Option<f64>>(17)?,
            row.get::<_, Option<f64>>(18)?,
            row.get::<_, Option<f64>>(19)?,
            row.get::<_, Option<String>>(20)?,
            row.get::<_, Option<f64>>(21)?,
            row.get::<_, Option<f64>>(22)?,
            row.get::<_, Option<i64>>(23)?,
            row.get::<_, Option<i64>>(24)?,
            row.get::<_, Option<f64>>(25)?,
        ))
    };
    let rows = if params.is_empty() {
        stmt.query_map([], row_from_query)?
    } else {
        let param_refs: Vec<&dyn rusqlite::ToSql> = params.iter().map(|b| b.as_ref()).collect();
        stmt.query_map(rusqlite::params_from_iter(param_refs), row_from_query)?
    };

    let mut file_path = Vec::new();
    let mut data_offset = Vec::new();
    let mut naxis1 = Vec::new();
    let mut naxis2 = Vec::new();
    let mut bitpix = Vec::new();
    let mut bzero = Vec::new();
    let mut data_size = Vec::new();
    let mut file_name = Vec::new();
    let mut sample_name = Vec::new();
    let mut tag = Vec::new();
    let mut scan_number = Vec::new();
    let mut frame_number = Vec::new();
    let mut date = Vec::new();
    let mut beamline_energy = Vec::new();
    let mut sample_theta = Vec::new();
    let mut ccd_theta = Vec::new();
    let mut hos = Vec::new();
    let mut epu = Vec::new();
    let mut exposure = Vec::new();
    let mut sample_name_h = Vec::new();
    let mut scan_id = Vec::new();
    let mut lambda = Vec::new();
    let mut q = Vec::new();
    let mut beam_row = Vec::new();
    let mut beam_col = Vec::new();
    let mut beam_sigma = Vec::new();

    for row in rows {
        let r = row.map_err(CatalogError::Sqlite)?;
        file_path.push(r.0);
        data_offset.push(r.1);
        naxis1.push(r.2);
        naxis2.push(r.3);
        bitpix.push(r.4);
        bzero.push(r.5);
        data_size.push(r.6);
        file_name.push(r.7);
        sample_name.push(r.8);
        tag.push(r.9);
        scan_number.push(r.10);
        frame_number.push(r.11);
        date.push(r.12);
        beamline_energy.push(r.13);
        sample_theta.push(r.14);
        ccd_theta.push(r.15);
        hos.push(r.16);
        epu.push(r.17);
        exposure.push(r.18);
        sample_name_h.push(r.19);
        scan_id.push(r.20);
        lambda.push(r.21);
        q.push(r.22);
        beam_row.push(r.23);
        beam_col.push(r.24);
        beam_sigma.push(r.25);
    }

    let series = vec![
        Series::new("file_path".into(), file_path),
        Series::new("data_offset".into(), data_offset),
        Series::new("naxis1".into(), naxis1),
        Series::new("naxis2".into(), naxis2),
        Series::new("bitpix".into(), bitpix),
        Series::new("bzero".into(), bzero),
        Series::new("data_size".into(), data_size),
        Series::new("file_name".into(), file_name),
        Series::new("sample_name".into(), sample_name),
        Series::new("tag".into(), tag),
        Series::new("scan_number".into(), scan_number),
        Series::new("frame_number".into(), frame_number),
        Series::new("DATE".into(), date),
        Series::new("Beamline Energy".into(), beamline_energy),
        Series::new("Sample Theta".into(), sample_theta),
        Series::new("CCD Theta".into(), ccd_theta),
        Series::new("Higher Order Suppressor".into(), hos),
        Series::new("EPU Polarization".into(), epu),
        Series::new("EXPOSURE".into(), exposure),
        Series::new("Sample Name".into(), sample_name_h),
        Series::new("Scan ID".into(), scan_id),
        Series::new("Lambda".into(), lambda),
        Series::new("Q".into(), q),
        Series::new("beam_row".into(), beam_row),
        Series::new("beam_col".into(), beam_col),
        Series::new("beam_sigma".into(), beam_sigma),
    ];
    let columns: Vec<Column> = series.into_iter().map(|s| s.into()).collect();
    let df = DataFrame::new(columns).map_err(|e| CatalogError::Validation(e.to_string()))?;
    Ok(df)
}

type OverrideRowSql = (String, Option<String>, Option<String>, Option<String>);

fn map_override_four_cols(row: &rusqlite::Row) -> rusqlite::Result<OverrideRowSql> {
    Ok((
        row.get(0)?,
        row.get(1)?,
        row.get(2)?,
        row.get(3)?,
    ))
}

/// Loads override rows into a Polars frame with columns `path`, `sample_name`, `tag`, `notes`.
///
/// Concatenates the legacy `overrides` table (`path` is `files.path`) and `bt_file_overrides`
/// (`path` is `source_path` matching `bt_scan_points`). When `path` is `Some`, returns rows from
/// both tables that match that exact string.
pub fn get_overrides(db_path: &Path, path: Option<&str>) -> Result<DataFrame> {
    let conn = open_catalog_db(db_path)?;
    let mut paths = Vec::new();
    let mut sample_names = Vec::new();
    let mut tags = Vec::new();
    let mut notes = Vec::new();
    match path {
        Some(p) => {
            let mut stmt = conn.prepare(
                "SELECT path, sample_name, tag, notes FROM overrides WHERE path = ?1",
            )?;
            for row in stmt.query_map([p], map_override_four_cols)? {
                let r = row.map_err(CatalogError::Sqlite)?;
                paths.push(r.0);
                sample_names.push(r.1);
                tags.push(r.2);
                notes.push(r.3);
            }
            let mut stmt = conn.prepare(
                "SELECT source_path, sample_name, tag, notes FROM bt_file_overrides WHERE source_path = ?1",
            )?;
            for row in stmt.query_map([p], map_override_four_cols)? {
                let r = row.map_err(CatalogError::Sqlite)?;
                paths.push(r.0);
                sample_names.push(r.1);
                tags.push(r.2);
                notes.push(r.3);
            }
        }
        None => {
            let mut stmt =
                conn.prepare("SELECT path, sample_name, tag, notes FROM overrides")?;
            for row in stmt.query_map([], map_override_four_cols)? {
                let r = row.map_err(CatalogError::Sqlite)?;
                paths.push(r.0);
                sample_names.push(r.1);
                tags.push(r.2);
                notes.push(r.3);
            }
            let mut stmt = conn.prepare(
                "SELECT source_path, sample_name, tag, notes FROM bt_file_overrides",
            )?;
            for row in stmt.query_map([], map_override_four_cols)? {
                let r = row.map_err(CatalogError::Sqlite)?;
                paths.push(r.0);
                sample_names.push(r.1);
                tags.push(r.2);
                notes.push(r.3);
            }
        }
    }
    let df = DataFrame::new(vec![
        Series::new("path".into(), paths).into(),
        Series::new("sample_name".into(), sample_names).into(),
        Series::new("tag".into(), tags).into(),
        Series::new("notes".into(), notes).into(),
    ])
    .map_err(|e| CatalogError::Validation(e.to_string()))?;
    Ok(df)
}

#[cfg(test)]
fn scan_from_catalog_columns() -> Vec<&'static str> {
    vec![
        "file_path",
        "data_offset",
        "naxis1",
        "naxis2",
        "bitpix",
        "bzero",
        "data_size",
        "file_name",
        "sample_name",
        "tag",
        "scan_number",
        "frame_number",
        "DATE",
        "Beamline Energy",
        "Sample Theta",
        "CCD Theta",
        "Higher Order Suppressor",
        "EPU Polarization",
        "EXPOSURE",
        "Sample Name",
        "Scan ID",
        "Lambda",
        "Q",
        "beam_row",
        "beam_col",
        "beam_sigma",
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_scan_from_catalog_empty_db() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join(".pyref_catalog.db");
        crate::catalog::open_or_create_db(tmp.path()).unwrap();
        let df = scan_from_catalog(&db_path, None).unwrap();
        assert_eq!(df.height(), 0);
        for col in scan_from_catalog_columns() {
            assert!(df.column(col).is_ok(), "missing column {}", col);
        }
    }

    #[test]
    fn test_get_overrides_empty() {
        let tmp = TempDir::new().unwrap();
        crate::catalog::open_or_create_db(tmp.path()).unwrap();
        let db_path = tmp.path().join(".pyref_catalog.db");
        let df = get_overrides(&db_path, None).unwrap();
        assert_eq!(df.height(), 0);
    }

    #[test]
    fn test_list_beamtime_entries_empty() {
        let tmp = TempDir::new().unwrap();
        crate::catalog::open_or_create_db(tmp.path()).unwrap();
        let db_path = tmp.path().join(".pyref_catalog.db");
        let e = list_beamtime_entries(&db_path).unwrap();
        assert!(e.samples.is_empty());
        assert!(e.tags.is_empty());
        assert!(e.scans.is_empty());
    }

    #[test]
    fn test_set_override_bt_source_path_without_files_row() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("catalog.db");
        {
            let conn = crate::catalog::open_catalog_db(&db_path).unwrap();
            conn.execute(
                "INSERT INTO bt_beamtimes (beamtime_path) VALUES (?1)",
                ["/tmp/beam"],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO bt_samples (beamtime_id, name, tag) VALUES (1, 'orig', 't1')",
                [],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO bt_scans (uid, beamtime_id, sample_id) VALUES ('s_1_1', 1, 1)",
                [],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO bt_streams (uid, scan_uid) VALUES ('st_1_1', 's_1_1')",
                [],
            )
            .unwrap();
            conn.execute(
                r#"INSERT INTO bt_scan_points (
                    uid, stream_uid, scan_uid, sample_id, seq_index,
                    source_path, source_data_offset, source_naxis1, source_naxis2, source_bitpix, source_bzero
                ) VALUES (
                    'sp_1_1_0', 'st_1_1', 's_1_1', 1, 0,
                    '/data/example.fits', 0, 2, 2, -32, 0
                )"#,
                [],
            )
            .unwrap();
        }
        set_override(
            &db_path,
            "/data/example.fits",
            Some("corrected"),
            Some("t2"),
            None,
        )
        .unwrap();
        let df = scan_from_catalog(&db_path, None).unwrap();
        assert_eq!(df.height(), 1);
        let sn = df.column("sample_name").unwrap().str().unwrap().get(0);
        assert_eq!(sn, Some("corrected"));
        let tg = df.column("tag").unwrap().str().unwrap().get(0);
        assert_eq!(tg, Some("t2"));
        let all_ov = get_overrides(&db_path, None).unwrap();
        assert_eq!(all_ov.height(), 1);
        let p = all_ov.column("path").unwrap().str().unwrap().get(0);
        assert_eq!(p, Some("/data/example.fits"));
    }
}

/// Upserts `sample_name`, `tag`, and `notes` for a catalog file path.
///
/// When `path` exists in `files`, writes `overrides`. When it exists only as `bt_scan_points.source_path`,
/// writes `bt_file_overrides`. Returns [`CatalogError::Validation`] when `path` matches neither.
pub fn set_override(
    db_path: &Path,
    path: &str,
    sample_name: Option<&str>,
    tag: Option<&str>,
    notes: Option<&str>,
) -> Result<()> {
    let conn = open_catalog_db(db_path)?;
    let in_files: i64 =
        conn.query_row("SELECT COUNT(1) FROM files WHERE path = ?1", [path], |r| {
            r.get(0)
        })?;
    if in_files > 0 {
        conn.execute(
            "INSERT OR REPLACE INTO overrides (path, sample_name, tag, notes) VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![path, sample_name, tag, notes],
        )?;
        return Ok(());
    }
    let in_bt: i64 = conn.query_row(
        "SELECT COUNT(1) FROM bt_scan_points WHERE source_path = ?1",
        [path],
        |r| r.get(0),
    )?;
    if in_bt > 0 {
        conn.execute(
            "INSERT OR REPLACE INTO bt_file_overrides (source_path, sample_name, tag, notes) VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![path, sample_name, tag, notes],
        )?;
        return Ok(());
    }
    Err(CatalogError::Validation(format!(
        "path not in catalog (not in files.path and not in bt_scan_points.source_path): {}",
        path
    )))
}

pub fn rename_file_in_catalog(
    db_path: &Path,
    old_path: &str,
    new_path: &str,
    new_file_name: &str,
    new_sample_name: &str,
    new_tag: Option<&str>,
) -> Result<()> {
    let conn = open_catalog_db(db_path)?;
    let exists: i64 = conn.query_row(
        "SELECT COUNT(1) FROM files WHERE path = ?1",
        [old_path],
        |r| r.get(0),
    )?;
    if exists == 0 {
        return Err(CatalogError::Validation(format!(
            "path not in files table: {}",
            old_path
        )));
    }
    conn.execute(
        "UPDATE files SET path = ?1, file_path = ?1, file_name = ?2, sample_name = ?3, tag = ?4 WHERE path = ?5",
        rusqlite::params![new_path, new_file_name, new_sample_name, new_tag, old_path],
    )?;
    conn.execute("DELETE FROM overrides WHERE path = ?1", [old_path])?;
    Ok(())
}

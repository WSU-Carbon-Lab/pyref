#![cfg(feature = "catalog")]

use crate::catalog::{CatalogError, Result};
use polars::prelude::*;
use rusqlite::Connection;
use std::path::Path;

#[derive(Default, Debug, Clone)]
pub struct CatalogFilter {
    pub sample_name: Option<String>,
    pub tag: Option<String>,
    pub experiment_numbers: Option<Vec<i64>>,
    pub energy_min: Option<f64>,
    pub energy_max: Option<f64>,
}

pub struct BeamtimeEntries {
    pub samples: Vec<String>,
    pub tags: Vec<String>,
    pub experiments: Vec<(i64, String)>,
}

pub struct FileRow {
    pub file_path: String,
    pub sample_name: String,
    pub tag: Option<String>,
    pub experiment_number: i64,
    pub frame_number: i64,
    pub beamline_energy: Option<f64>,
    pub sample_theta: Option<f64>,
    pub q: Option<f64>,
}

pub fn list_beamtime_entries(db_path: &Path) -> Result<BeamtimeEntries> {
    let conn = Connection::open(db_path)?;
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
    let experiments: Vec<(i64, String)> = conn
        .prepare(
            r#"SELECT DISTINCT f.experiment_number FROM files f
               WHERE f.experiment_number != 0 ORDER BY f.experiment_number"#,
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
        experiments,
    })
}

pub fn query_files(
    db_path: &Path,
    filter: Option<&CatalogFilter>,
) -> Result<Vec<FileRow>> {
    let (where_clause, params) = filter
        .map(build_where_and_params)
        .unwrap_or_else(|| (String::new(), Vec::new()));
    let conn = Connection::open(db_path)?;
    let sql = format!(
        r#"SELECT f.file_path,
            COALESCE(o.sample_name, f.sample_name),
            COALESCE(o.tag, f.tag),
            f.experiment_number, f.frame_number,
            f."Beamline Energy", f."Sample Theta", f.Q
            FROM files f LEFT JOIN overrides o ON f.path = o.path{}
            ORDER BY f.experiment_number, f.frame_number"#,
        where_clause
    );
    let mut stmt = conn.prepare(&sql)?;
    let map_row = |row: &rusqlite::Row| {
        Ok(FileRow {
            file_path: row.get(0)?,
            sample_name: row.get(1)?,
            tag: row.get(2)?,
            experiment_number: row.get(3)?,
            frame_number: row.get(4)?,
            beamline_energy: row.get(5)?,
            sample_theta: row.get(6)?,
            q: row.get(7)?,
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

const RESOLVED_QUERY: &str = r#"
SELECT
    f.file_path, f.data_offset, f.naxis1, f.naxis2, f.bitpix, f.bzero, f.data_size,
    f.file_name,
    COALESCE(o.sample_name, f.sample_name) AS sample_name,
    COALESCE(o.tag, f.tag) AS tag,
    f.experiment_number, f.frame_number,
    f."DATE", f."Beamline Energy", f."Sample Theta", f."CCD Theta",
    f."Higher Order Suppressor", f."EPU Polarization", f.EXPOSURE,
    f."Sample Name", f."Scan ID", f.Lambda, f.Q
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
    if let Some(ref exp) = filter.experiment_numbers {
        if !exp.is_empty() {
            let placeholders: Vec<String> = (0..exp.len()).map(|i| format!("?{}", params.len() + 1 + i)).collect();
            conditions.push(format!("f.experiment_number IN ({})", placeholders.join(",")));
            for e in exp {
                params.push(Box::new(*e));
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

pub fn scan_from_catalog(
    db_path: &Path,
    filter: Option<&CatalogFilter>,
) -> Result<DataFrame> {
    let conn = Connection::open(db_path)?;
    let (where_clause, params) = filter
        .map(|f| build_where_and_params(f))
        .unwrap_or_else(|| (String::new(), Vec::new()));
    let order_clause = " ORDER BY f.experiment_number, f.frame_number";
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
    let mut experiment_number = Vec::new();
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
        experiment_number.push(r.10);
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
        Series::new("experiment_number".into(), experiment_number),
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
    ];
    let columns: Vec<Column> = series.into_iter().map(|s| s.into()).collect();
    let df = DataFrame::new(columns).map_err(|e| CatalogError::Validation(e.to_string()))?;
    Ok(df)
}

pub fn get_overrides(db_path: &Path, path: Option<&str>) -> Result<DataFrame> {
    let conn = Connection::open(db_path)?;
    let (sql, params): (String, Vec<Box<dyn rusqlite::ToSql + '_>>) = match path {
        Some(p) => (
            "SELECT path, sample_name, tag, notes FROM overrides WHERE path = ?1".to_string(),
            vec![Box::new(p.to_string())],
        ),
        None => (
            "SELECT path, sample_name, tag, notes FROM overrides".to_string(),
            Vec::new(),
        ),
    };
    let mut stmt = conn.prepare(&sql)?;
    let map_row = |row: &rusqlite::Row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, Option<String>>(1)?,
            row.get::<_, Option<String>>(2)?,
            row.get::<_, Option<String>>(3)?,
        ))
    };
    let rows = if params.is_empty() {
        stmt.query_map([], map_row)?
    } else {
        let pr: Vec<&dyn rusqlite::ToSql> = params.iter().map(|b| b.as_ref()).collect();
        stmt.query_map(rusqlite::params_from_iter(pr), map_row)?
    };
    let mut paths = Vec::new();
    let mut sample_names = Vec::new();
    let mut tags = Vec::new();
    let mut notes = Vec::new();
    for row in rows {
        let r = row.map_err(CatalogError::Sqlite)?;
        paths.push(r.0);
        sample_names.push(r.1);
        tags.push(r.2);
        notes.push(r.3);
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
        "file_path", "data_offset", "naxis1", "naxis2", "bitpix", "bzero", "data_size",
        "file_name", "sample_name", "tag", "experiment_number", "frame_number",
        "DATE", "Beamline Energy", "Sample Theta", "CCD Theta", "Higher Order Suppressor",
        "EPU Polarization", "EXPOSURE", "Sample Name", "Scan ID", "Lambda", "Q",
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
        assert!(e.experiments.is_empty());
    }
}

pub fn set_override(
    db_path: &Path,
    path: &str,
    sample_name: Option<&str>,
    tag: Option<&str>,
    notes: Option<&str>,
) -> Result<()> {
    let conn = Connection::open(db_path)?;
    let exists: i64 = conn.query_row(
        "SELECT COUNT(1) FROM files WHERE path = ?1",
        [path],
        |r| r.get(0),
    )?;
    if exists == 0 {
        return Err(CatalogError::Validation(format!(
            "path not in files table: {}",
            path
        )));
    }
    conn.execute(
        "INSERT OR REPLACE INTO overrides (path, sample_name, tag, notes) VALUES (?1, ?2, ?3, ?4)",
        rusqlite::params![path, sample_name, tag, notes],
    )?;
    Ok(())
}

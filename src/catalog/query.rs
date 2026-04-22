use diesel::deserialize::{self, QueryableByName};
use diesel::dsl::count_star;
use diesel::prelude::*;
use diesel::sql_types::{BigInt, Double, Integer, Nullable, Text};
use diesel::sqlite::{Sqlite, SqliteConnection};
use diesel::{sql_query, OptionalExtension, RunQueryDsl};
use polars::prelude::*;
use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::catalog::{db, paths, CatalogError, Result};
use crate::loader::read_fits_headers_only_row;
use crate::schema::{
    beam_finding, beamtimes, file_overrides, file_tags, files, frames, header_cards, header_values,
    samples, scans, tags,
};

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

fn beamtime_id_for_dir(conn: &mut SqliteConnection, beamtime_dir: &Path) -> Result<Option<i32>> {
    let uri = match paths::file_uri_for_path_relaxed(beamtime_dir) {
        Ok(u) => u,
        Err(CatalogError::Io(_)) => return Ok(None),
        Err(e) => return Err(e),
    };
    beamtimes::table
        .filter(beamtimes::nas_uri.eq(uri))
        .select(beamtimes::id)
        .first(conn)
        .optional()
        .map_err(CatalogError::Diesel)
}

#[derive(Debug, Clone, Copy, Default)]
pub struct HeaderSyncReport {
    pub files_checked: u32,
    pub files_updated: u32,
    pub header_rows_inserted: u32,
}

fn normalize_header_display_name(card_name: &str) -> String {
    card_name
        .split_whitespace()
        .map(|part| part.to_ascii_lowercase().replace("sample", "sam"))
        .collect::<Vec<String>>()
        .join("_")
}

pub fn catalog_file_count(db_path: &Path, beamtime_dir: Option<&Path>) -> Result<u32> {
    let mut conn = db::establish_connection(db_path)?;
    let n: i64 = if let Some(dir) = beamtime_dir {
        let uri = paths::file_uri_for_path_relaxed(dir)?;
        files::table
            .inner_join(beamtimes::table.on(files::beamtime_id.eq(beamtimes::id)))
            .filter(beamtimes::nas_uri.eq(uri))
            .select(count_star())
            .first(&mut conn)
            .map_err(CatalogError::Diesel)?
    } else {
        files::table
            .select(count_star())
            .first(&mut conn)
            .map_err(CatalogError::Diesel)?
    };
    Ok(n as u32)
}

pub fn list_beamtimes_from_catalog(db_path: &Path) -> Result<Vec<(std::path::PathBuf, i64)>> {
    let mut conn = db::establish_connection(db_path)?;
    let rows: Vec<(String, i32)> = beamtimes::table
        .select((beamtimes::nas_uri, beamtimes::id))
        .order(beamtimes::id.desc())
        .load(&mut conn)
        .map_err(CatalogError::Diesel)?;
    Ok(rows
        .into_iter()
        .map(|(uri, id)| {
            let p = uri
                .strip_prefix("file://")
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|| std::path::PathBuf::from(&uri));
            (p, id as i64)
        })
        .collect())
}

pub fn list_beamtime_entries(db_path: &Path) -> Result<BeamtimeEntries> {
    let mut conn = db::establish_connection(db_path)?;
    let mut samples: Vec<String> = samples::table
        .select(samples::name)
        .distinct()
        .filter(samples::name.ne(""))
        .load(&mut conn)
        .map_err(CatalogError::Diesel)?;
    let mut tag_rows: Vec<String> = tags::table
        .inner_join(file_tags::table.on(file_tags::tag_id.eq(tags::id)))
        .select(tags::slug)
        .distinct()
        .load(&mut conn)
        .map_err(CatalogError::Diesel)?;
    let scan_rows: Vec<i32> = files::table
        .select(files::scan_number)
        .distinct()
        .filter(files::scan_number.ne(0))
        .load(&mut conn)
        .map_err(CatalogError::Diesel)?;
    let mut scans: Vec<(i64, String)> = scan_rows
        .into_iter()
        .map(|n| (n as i64, format!("Scan {}", n)))
        .collect();
    samples.sort();
    tag_rows.sort();
    scans.sort_by_key(|a| a.0);
    Ok(BeamtimeEntries {
        samples,
        tags: tag_rows,
        scans,
    })
}

pub fn list_beamtime_entries_v2(db_path: &Path, beamtime_dir: &Path) -> Result<BeamtimeEntries> {
    let mut conn = db::establish_connection(db_path)?;
    let Some(bid) = beamtime_id_for_dir(&mut conn, beamtime_dir)? else {
        return Ok(BeamtimeEntries {
            samples: Vec::new(),
            tags: Vec::new(),
            scans: Vec::new(),
        });
    };
    let mut samples: Vec<String> = samples::table
        .filter(samples::beamtime_id.eq(bid))
        .select(samples::name)
        .distinct()
        .filter(samples::name.ne(""))
        .load(&mut conn)
        .map_err(CatalogError::Diesel)?;
    let mut tag_rows: Vec<String> = tags::table
        .inner_join(file_tags::table.on(file_tags::tag_id.eq(tags::id)))
        .inner_join(files::table.on(files::id.eq(file_tags::file_id)))
        .filter(files::beamtime_id.eq(bid))
        .select(tags::slug)
        .distinct()
        .load(&mut conn)
        .map_err(CatalogError::Diesel)?;
    let scan_rows: Vec<i32> = scans::table
        .filter(scans::beamtime_id.eq(bid))
        .select(scans::scan_number)
        .order(scans::scan_number.asc())
        .load(&mut conn)
        .map_err(CatalogError::Diesel)?;
    let scans: Vec<(i64, String)> = scan_rows
        .into_iter()
        .map(|n| (n as i64, format!("Scan {}", n)))
        .collect();
    samples.sort();
    tag_rows.sort();
    Ok(BeamtimeEntries {
        samples,
        tags: tag_rows,
        scans,
    })
}

struct CatalogFileSql {
    nas_uri: String,
    sample_name: String,
    tag: Option<String>,
    scan_number: i32,
    frame_number: i32,
    beamline_energy: Option<f64>,
    sample_theta: Option<f64>,
    epu_polarization: Option<f64>,
    acquired_at: Option<String>,
    centroid_row: Option<f64>,
    centroid_col: Option<f64>,
    fit_std: Option<f64>,
    frame_id: i32,
}

impl QueryableByName<Sqlite> for CatalogFileSql {
    fn build<'a>(row: &impl diesel::row::NamedRow<'a, Sqlite>) -> deserialize::Result<Self> {
        Ok(Self {
            nas_uri: diesel::row::NamedRow::get::<Text, String>(row, "nas_uri")?,
            sample_name: diesel::row::NamedRow::get::<Text, String>(row, "sample_name")?,
            tag: diesel::row::NamedRow::get::<Nullable<Text>, Option<String>>(row, "tag")?,
            scan_number: diesel::row::NamedRow::get::<Integer, i32>(row, "scan_number")?,
            frame_number: diesel::row::NamedRow::get::<Integer, i32>(row, "frame_number")?,
            beamline_energy: diesel::row::NamedRow::get::<Nullable<Double>, Option<f64>>(
                row,
                "beamline_energy",
            )?,
            sample_theta: diesel::row::NamedRow::get::<Nullable<Double>, Option<f64>>(
                row,
                "sample_theta",
            )?,
            epu_polarization: diesel::row::NamedRow::get::<Nullable<Double>, Option<f64>>(
                row,
                "epu_polarization",
            )?,
            acquired_at: diesel::row::NamedRow::get::<Nullable<Text>, Option<String>>(
                row,
                "acquired_at",
            )?,
            centroid_row: diesel::row::NamedRow::get::<Nullable<Double>, Option<f64>>(
                row,
                "centroid_row",
            )?,
            centroid_col: diesel::row::NamedRow::get::<Nullable<Double>, Option<f64>>(
                row,
                "centroid_col",
            )?,
            fit_std: diesel::row::NamedRow::get::<Nullable<Double>, Option<f64>>(row, "fit_std")?,
            frame_id: diesel::row::NamedRow::get::<Integer, i32>(row, "frame_id")?,
        })
    }
}

fn build_files_sql(beamtime_id_sql: Option<i32>, filter: Option<&CatalogFilter>) -> String {
    let mut sql = String::from(
        r#"SELECT f.nas_uri AS nas_uri,
  COALESCE(o.sample_name, s.name) AS sample_name,
  COALESCE(o.tag, (SELECT t.slug FROM file_tags ft JOIN tags t ON ft.tag_id = t.id WHERE ft.file_id = f.id ORDER BY t.slug LIMIT 1)) AS tag,
  sc.scan_number,
  fr.frame_number,
  (SELECT hv.value
   FROM header_values hv
   INNER JOIN header_cards hc ON hv.header_card_id = hc.id
   WHERE hv.frame_id = fr.id AND hc.name = 'Beamline Energy'
   LIMIT 1) AS beamline_energy,
  (SELECT hv.value
   FROM header_values hv
   INNER JOIN header_cards hc ON hv.header_card_id = hc.id
   WHERE hv.frame_id = fr.id AND hc.name = 'Sample Theta'
   LIMIT 1) AS sample_theta,
  (SELECT hv.value
   FROM header_values hv
   INNER JOIN header_cards hc ON hv.header_card_id = hc.id
   WHERE hv.frame_id = fr.id AND hc.name = 'EPU Polarization'
   LIMIT 1) AS epu_polarization,
  fr.acquired_at,
  bf.centroid_row,
  bf.centroid_col,
  bf.fit_std,
  fr.id AS frame_id
FROM frames fr
INNER JOIN files f ON fr.file_id = f.id
INNER JOIN scans sc ON fr.scan_id = sc.id
INNER JOIN samples s ON f.sample_id = s.id
LEFT JOIN file_overrides o ON o.source_path = f.nas_uri
LEFT JOIN beam_finding bf ON bf.frame_id = fr.id
WHERE 1=1"#,
    );
    if let Some(bid) = beamtime_id_sql {
        sql.push_str(&format!(" AND f.beamtime_id = {bid}"));
    }
    if let Some(f) = filter {
        if let Some(s) = &f.sample_name {
            sql.push_str(&format!(
                " AND (COALESCE(o.sample_name, s.name) = '{}')",
                s.replace('\'', "''")
            ));
        }
        if let Some(t) = &f.tag {
            sql.push_str(&format!(
                " AND (COALESCE(o.tag, (SELECT t2.slug FROM file_tags ft2 JOIN tags t2 ON ft2.tag_id = t2.id WHERE ft2.file_id = f.id ORDER BY t2.slug LIMIT 1)) = '{}')",
                t.replace('\'', "''")
            ));
        }
        if let Some(ref sns) = f.scan_numbers {
            if !sns.is_empty() {
                let ns: Vec<String> = sns.iter().map(|n| n.to_string()).collect();
                sql.push_str(&format!(" AND sc.scan_number IN ({})", ns.join(",")));
            }
        }
        if let Some(em) = f.energy_min {
            sql.push_str(&format!(
                " AND (SELECT hv.value FROM header_values hv \
INNER JOIN header_cards hc ON hv.header_card_id = hc.id \
WHERE hv.frame_id = fr.id AND hc.name = 'Beamline Energy' LIMIT 1) >= {em}"
            ));
        }
        if let Some(em) = f.energy_max {
            sql.push_str(&format!(
                " AND (SELECT hv.value FROM header_values hv \
INNER JOIN header_cards hc ON hv.header_card_id = hc.id \
WHERE hv.frame_id = fr.id AND hc.name = 'Beamline Energy' LIMIT 1) <= {em}"
            ));
        }
    }
    sql.push_str(" ORDER BY sc.scan_number, fr.frame_number");
    sql
}

pub fn query_files(db_path: &Path, filter: Option<&CatalogFilter>) -> Result<Vec<FileRow>> {
    let mut conn = db::establish_connection(db_path)?;
    let sql = build_files_sql(None, filter);
    let rows: Vec<CatalogFileSql> = sql_query(&sql)
        .load(&mut conn)
        .map_err(CatalogError::Diesel)?;
    Ok(rows
        .into_iter()
        .map(|r| FileRow {
            file_path: r.nas_uri,
            sample_name: r.sample_name,
            tag: r.tag,
            scan_number: r.scan_number as i64,
            frame_number: r.frame_number as i64,
            beamline_energy: r.beamline_energy,
            sample_theta: r.sample_theta,
            epu_polarization: r.epu_polarization,
            q: None,
            date_iso: r.acquired_at,
            beam_row: r.centroid_row.map(|x| x as i64),
            beam_col: r.centroid_col.map(|x| x as i64),
            beam_sigma: r.fit_std,
            scan_point_uid: Some(format!("frame_{}", r.frame_id)),
        })
        .collect())
}

pub fn query_scan_points(
    db_path: &Path,
    beamtime_dir: &Path,
    filter: Option<&CatalogFilter>,
) -> Result<Vec<FileRow>> {
    let mut conn = db::establish_connection(db_path)?;
    let bid = match beamtime_id_for_dir(&mut conn, beamtime_dir)? {
        Some(id) => id,
        None => return Ok(Vec::new()),
    };
    let sql = build_files_sql(Some(bid), filter);
    let rows: Vec<CatalogFileSql> = sql_query(&sql)
        .load(&mut conn)
        .map_err(CatalogError::Diesel)?;
    Ok(rows
        .into_iter()
        .map(|r| FileRow {
            file_path: r.nas_uri,
            sample_name: r.sample_name,
            tag: r.tag,
            scan_number: r.scan_number as i64,
            frame_number: r.frame_number as i64,
            beamline_energy: r.beamline_energy,
            sample_theta: r.sample_theta,
            epu_polarization: r.epu_polarization,
            q: None,
            date_iso: r.acquired_at,
            beam_row: r.centroid_row.map(|x| x as i64),
            beam_col: r.centroid_col.map(|x| x as i64),
            beam_sigma: r.fit_std,
            scan_point_uid: Some(format!("frame_{}", r.frame_id)),
        })
        .collect())
}

struct CatalogScanSql {
    file_path: String,
    data_offset: i64,
    naxis1: i32,
    naxis2: i32,
    bitpix: i32,
    bzero: i64,
    data_size: i64,
    file_name: String,
    sample_name: String,
    tag: Option<String>,
    scan_number: i32,
    frame_number: i32,
    zarr_group_key: i32,
    zarr_frame_index: i32,
    zarr_bucket_frame_index: Option<i32>,
    zarr_path: String,
    date_iso: Option<String>,
    beamline_energy: Option<f64>,
    sample_theta: Option<f64>,
    ccd_theta: Option<f64>,
    hos: Option<f64>,
    epu: Option<f64>,
    exposure: Option<f64>,
    ai3_izero: Option<f64>,
    sample_name_h: String,
    scan_id: f64,
    beam_row: Option<f64>,
    beam_col: Option<f64>,
    beam_sigma: Option<f64>,
    reflectivity_profile_index: Option<i32>,
    reflectivity_scan_type: Option<String>,
    catalog_scan_type: Option<String>,
}

impl QueryableByName<Sqlite> for CatalogScanSql {
    fn build<'a>(row: &impl diesel::row::NamedRow<'a, Sqlite>) -> deserialize::Result<Self> {
        Ok(Self {
            file_path: diesel::row::NamedRow::get::<Text, String>(row, "file_path")?,
            data_offset: diesel::row::NamedRow::get::<BigInt, i64>(row, "data_offset")?,
            naxis1: diesel::row::NamedRow::get::<Integer, i32>(row, "naxis1")?,
            naxis2: diesel::row::NamedRow::get::<Integer, i32>(row, "naxis2")?,
            bitpix: diesel::row::NamedRow::get::<Integer, i32>(row, "bitpix")?,
            bzero: diesel::row::NamedRow::get::<BigInt, i64>(row, "bzero")?,
            data_size: diesel::row::NamedRow::get::<BigInt, i64>(row, "data_size")?,
            file_name: diesel::row::NamedRow::get::<Text, String>(row, "file_name")?,
            sample_name: diesel::row::NamedRow::get::<Text, String>(row, "sample_name")?,
            tag: diesel::row::NamedRow::get::<Nullable<Text>, Option<String>>(row, "tag")?,
            scan_number: diesel::row::NamedRow::get::<Integer, i32>(row, "scan_number")?,
            frame_number: diesel::row::NamedRow::get::<Integer, i32>(row, "frame_number")?,
            zarr_group_key: diesel::row::NamedRow::get::<Integer, i32>(row, "zarr_group_key")?,
            zarr_frame_index: diesel::row::NamedRow::get::<Integer, i32>(row, "zarr_frame_index")?,
            zarr_bucket_frame_index: diesel::row::NamedRow::get::<Nullable<Integer>, Option<i32>>(
                row,
                "zarr_bucket_frame_index",
            )?,
            zarr_path: diesel::row::NamedRow::get::<Text, String>(row, "zarr_path")?,
            date_iso: diesel::row::NamedRow::get::<Nullable<Text>, Option<String>>(
                row, "date_iso",
            )?,
            beamline_energy: diesel::row::NamedRow::get::<Nullable<Double>, Option<f64>>(
                row,
                "beamline_energy",
            )?,
            sample_theta: diesel::row::NamedRow::get::<Nullable<Double>, Option<f64>>(
                row,
                "sample_theta",
            )?,
            ccd_theta: diesel::row::NamedRow::get::<Nullable<Double>, Option<f64>>(
                row,
                "ccd_theta",
            )?,
            hos: diesel::row::NamedRow::get::<Nullable<Double>, Option<f64>>(row, "hos")?,
            epu: diesel::row::NamedRow::get::<Nullable<Double>, Option<f64>>(row, "epu")?,
            exposure: diesel::row::NamedRow::get::<Nullable<Double>, Option<f64>>(row, "exposure")?,
            ai3_izero: diesel::row::NamedRow::get::<Nullable<Double>, Option<f64>>(
                row,
                "ai3_izero",
            )?,
            sample_name_h: diesel::row::NamedRow::get::<Text, String>(row, "sample_name_h")?,
            scan_id: diesel::row::NamedRow::get::<Double, f64>(row, "scan_id")?,
            beam_row: diesel::row::NamedRow::get::<Nullable<Double>, Option<f64>>(row, "beam_row")?,
            beam_col: diesel::row::NamedRow::get::<Nullable<Double>, Option<f64>>(row, "beam_col")?,
            beam_sigma: diesel::row::NamedRow::get::<Nullable<Double>, Option<f64>>(
                row,
                "beam_sigma",
            )?,
            reflectivity_profile_index: diesel::row::NamedRow::get::<Nullable<Integer>, Option<i32>>(
                row,
                "reflectivity_profile_index",
            )?,
            reflectivity_scan_type: diesel::row::NamedRow::get::<Nullable<Text>, Option<String>>(
                row,
                "reflectivity_scan_type",
            )?,
            catalog_scan_type: diesel::row::NamedRow::get::<Nullable<Text>, Option<String>>(
                row,
                "catalog_scan_type",
            )?,
        })
    }
}

fn build_scan_df_sql(beamtime_id: Option<i32>, filter: Option<&CatalogFilter>) -> String {
    let mut sql = String::from(
        r#"SELECT
  f.nas_uri AS file_path,
  f.data_offset,
  f.naxis1,
  f.naxis2,
  f.bitpix,
  f.bzero,
  CAST(ABS(f.bitpix) / 8 * f.naxis1 * f.naxis2 AS INTEGER) AS data_size,
  f.filename AS file_name,
  COALESCE(o.sample_name, s.name) AS sample_name,
  COALESCE(o.tag, (SELECT t.slug FROM file_tags ft JOIN tags t ON ft.tag_id = t.id WHERE ft.file_id = f.id ORDER BY t.slug LIMIT 1)) AS tag,
  sc.scan_number,
  fr.frame_number,
  fr.zarr_group_key,
  fr.zarr_frame_index,
  fr.zarr_bucket_frame_index,
  bt.zarr_path,
  fr.acquired_at AS date_iso,
  (SELECT hv.value
   FROM header_values hv
   INNER JOIN header_cards hc ON hv.header_card_id = hc.id
   WHERE hv.frame_id = fr.id AND hc.name = 'Beamline Energy'
   LIMIT 1) AS beamline_energy,
  (SELECT hv.value
   FROM header_values hv
   INNER JOIN header_cards hc ON hv.header_card_id = hc.id
   WHERE hv.frame_id = fr.id AND hc.name = 'Sample Theta'
   LIMIT 1) AS sample_theta,
  (SELECT hv.value
   FROM header_values hv
   INNER JOIN header_cards hc ON hv.header_card_id = hc.id
   WHERE hv.frame_id = fr.id AND hc.name = 'CCD Theta'
   LIMIT 1) AS ccd_theta,
  CAST(NULL AS REAL) AS hos,
  (SELECT hv.value
   FROM header_values hv
   INNER JOIN header_cards hc ON hv.header_card_id = hc.id
   WHERE hv.frame_id = fr.id AND hc.name = 'EPU Polarization'
   LIMIT 1) AS epu,
  (SELECT hv.value
   FROM header_values hv
   INNER JOIN header_cards hc ON hv.header_card_id = hc.id
   WHERE hv.frame_id = fr.id AND hc.name = 'EXPOSURE'
   LIMIT 1) AS exposure,
  (SELECT hv.value
   FROM header_values hv
   INNER JOIN header_cards hc ON hv.header_card_id = hc.id
   WHERE hv.frame_id = fr.id AND hc.name = 'AI 3 Izero'
   LIMIT 1) AS ai3_izero,
  COALESCE(o.sample_name, s.name) AS sample_name_h,
  CAST(sc.scan_number AS REAL) AS scan_id,
  CAST(NULL AS REAL) AS lambda,
  CAST(NULL AS REAL) AS q,
  bf.centroid_row AS beam_row,
  bf.centroid_col AS beam_col,
  bf.fit_std AS beam_sigma,
  CAST(NULL AS INTEGER) AS reflectivity_profile_index,
  CAST(NULL AS TEXT) AS reflectivity_scan_type,
  sc.scan_type AS catalog_scan_type
FROM frames fr
INNER JOIN files f ON fr.file_id = f.id
INNER JOIN scans sc ON fr.scan_id = sc.id
INNER JOIN samples s ON f.sample_id = s.id
INNER JOIN beamtimes bt ON f.beamtime_id = bt.id
LEFT JOIN file_overrides o ON o.source_path = f.nas_uri
LEFT JOIN beam_finding bf ON bf.frame_id = fr.id
WHERE 1=1"#,
    );
    if let Some(bid) = beamtime_id {
        sql.push_str(&format!(" AND f.beamtime_id = {bid}"));
    }
    if let Some(f) = filter {
        if let Some(s) = &f.sample_name {
            sql.push_str(&format!(
                " AND (COALESCE(o.sample_name, s.name) = '{}')",
                s.replace('\'', "''")
            ));
        }
        if let Some(t) = &f.tag {
            sql.push_str(&format!(
                " AND (COALESCE(o.tag, (SELECT t2.slug FROM file_tags ft2 JOIN tags t2 ON ft2.tag_id = t2.id WHERE ft2.file_id = f.id ORDER BY t2.slug LIMIT 1)) = '{}')",
                t.replace('\'', "''")
            ));
        }
        if let Some(ref sns) = f.scan_numbers {
            if !sns.is_empty() {
                let ns: Vec<String> = sns.iter().map(|n| n.to_string()).collect();
                sql.push_str(&format!(" AND sc.scan_number IN ({})", ns.join(",")));
            }
        }
        if let Some(em) = f.energy_min {
            sql.push_str(&format!(
                " AND (SELECT hv.value FROM header_values hv \
INNER JOIN header_cards hc ON hv.header_card_id = hc.id \
WHERE hv.frame_id = fr.id AND hc.name = 'Beamline Energy' LIMIT 1) >= {em}"
            ));
        }
        if let Some(em) = f.energy_max {
            sql.push_str(&format!(
                " AND (SELECT hv.value FROM header_values hv \
INNER JOIN header_cards hc ON hv.header_card_id = hc.id \
WHERE hv.frame_id = fr.id AND hc.name = 'Beamline Energy' LIMIT 1) <= {em}"
            ));
        }
    }
    sql.push_str(" ORDER BY sc.scan_number, fr.frame_number");
    sql
}

fn catalog_scan_sql_rows_to_dataframe(rows: Vec<CatalogScanSql>) -> Result<DataFrame> {
    const HC_EV_ANGSTROM: f64 = 12_398.419_843_320_025;

    let mut file_path = Vec::new();
    let mut data_offset = Vec::new();
    let mut naxis1 = Vec::new();
    let mut naxis2 = Vec::new();
    let mut bitpix = Vec::new();
    let mut bzero = Vec::new();
    let mut data_size = Vec::new();
    let mut file_name = Vec::new();
    let mut sample_name = Vec::new();
    let mut tag: Vec<Option<String>> = Vec::new();
    let mut scan_number = Vec::new();
    let mut frame_number = Vec::new();
    let mut zarr_group_key = Vec::new();
    let mut zarr_frame_index = Vec::new();
    let mut zarr_bucket_frame_index: Vec<Option<i64>> = Vec::new();
    let mut zarr_path = Vec::new();
    let mut date: Vec<Option<String>> = Vec::new();
    let mut beamline_energy = Vec::new();
    let mut sample_theta = Vec::new();
    let mut ccd_theta = Vec::new();
    let mut hos: Vec<Option<f64>> = Vec::new();
    let mut epu = Vec::new();
    let mut exposure = Vec::new();
    let mut ai3_izero = Vec::new();
    let mut sample_name_h: Vec<Option<String>> = Vec::new();
    let mut scan_id = Vec::new();
    let mut lambda: Vec<Option<f64>> = Vec::new();
    let mut q: Vec<Option<f64>> = Vec::new();
    let mut beam_row: Vec<Option<i64>> = Vec::new();
    let mut beam_col: Vec<Option<i64>> = Vec::new();
    let mut beam_sigma = Vec::new();
    let mut reflectivity_profile_index: Vec<Option<i64>> = Vec::new();
    let mut reflectivity_scan_type: Vec<Option<String>> = Vec::new();
    let mut catalog_scan_type: Vec<Option<String>> = Vec::new();
    for r in rows {
        file_path.push(r.file_path);
        data_offset.push(r.data_offset);
        naxis1.push(r.naxis1 as i64);
        naxis2.push(r.naxis2 as i64);
        bitpix.push(r.bitpix as i64);
        bzero.push(r.bzero);
        data_size.push(r.data_size);
        file_name.push(r.file_name);
        sample_name.push(r.sample_name.clone());
        tag.push(r.tag);
        scan_number.push(r.scan_number as i64);
        frame_number.push(r.frame_number as i64);
        zarr_group_key.push(r.zarr_group_key as i64);
        zarr_frame_index.push(r.zarr_frame_index as i64);
        zarr_bucket_frame_index.push(r.zarr_bucket_frame_index.map(|x| x as i64));
        zarr_path.push(r.zarr_path);
        date.push(r.date_iso);
        beamline_energy.push(r.beamline_energy);
        sample_theta.push(r.sample_theta);
        ccd_theta.push(r.ccd_theta);
        hos.push(r.hos);
        epu.push(r.epu);
        exposure.push(r.exposure);
        ai3_izero.push(r.ai3_izero);
        sample_name_h.push(Some(r.sample_name_h));
        scan_id.push(Some(r.scan_id));
        let lambda_value = r
            .beamline_energy
            .and_then(|energy| (energy > 0.0).then_some(HC_EV_ANGSTROM / energy));
        let q_value = match (lambda_value, r.sample_theta) {
            (Some(lam), Some(theta)) => Some(crate::io::q(lam, theta)),
            _ => None,
        };
        lambda.push(lambda_value);
        q.push(q_value);
        beam_row.push(r.beam_row.map(|x| x as i64));
        beam_col.push(r.beam_col.map(|x| x as i64));
        beam_sigma.push(r.beam_sigma);
        reflectivity_profile_index.push(r.reflectivity_profile_index.map(|x| x as i64));
        reflectivity_scan_type.push(r.reflectivity_scan_type);
        catalog_scan_type.push(r.catalog_scan_type);
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
        Series::new("zarr_group_key".into(), zarr_group_key),
        Series::new("zarr_frame_index".into(), zarr_frame_index),
        Series::new("zarr_bucket_frame_index".into(), zarr_bucket_frame_index),
        Series::new("zarr_path".into(), zarr_path),
        Series::new("DATE".into(), date),
        Series::new("Beamline Energy".into(), beamline_energy),
        Series::new("Sample Theta".into(), sample_theta),
        Series::new("CCD Theta".into(), ccd_theta),
        Series::new("Higher Order Suppressor".into(), hos),
        Series::new("EPU Polarization".into(), epu),
        Series::new("EXPOSURE".into(), exposure),
        Series::new("AI 3 Izero".into(), ai3_izero),
        Series::new("Sample Name".into(), sample_name_h),
        Series::new("Scan ID".into(), scan_id),
        Series::new("Lambda".into(), lambda),
        Series::new("Q".into(), q),
        Series::new("beam_row".into(), beam_row),
        Series::new("beam_col".into(), beam_col),
        Series::new("beam_sigma".into(), beam_sigma),
        Series::new(
            "reflectivity_profile_index".into(),
            reflectivity_profile_index,
        ),
        Series::new("reflectivity_scan_type".into(), reflectivity_scan_type),
        Series::new("catalog_scan_type".into(), catalog_scan_type),
    ];
    let columns: Vec<polars::prelude::Column> = series.into_iter().map(|s| s.into()).collect();
    DataFrame::new(columns).map_err(|e| CatalogError::Validation(e.to_string()))
}

pub fn scan_from_catalog(db_path: &Path, filter: Option<&CatalogFilter>) -> Result<DataFrame> {
    let mut conn = db::establish_connection(db_path)?;
    let sql = build_scan_df_sql(None, filter);
    let rows: Vec<CatalogScanSql> = sql_query(&sql)
        .load(&mut conn)
        .map_err(CatalogError::Diesel)?;
    catalog_scan_sql_rows_to_dataframe(rows)
}

/// Frame-level scan rows for one beamtime root directory (matched via [`paths::file_uri_for_path`]).
///
/// When the beamtime is not present in `beamtimes`, returns an empty [`DataFrame`] with the same
/// schema as [`scan_from_catalog`].
pub fn scan_from_catalog_for_beamtime(
    db_path: &Path,
    beamtime_dir: &Path,
    filter: Option<&CatalogFilter>,
) -> Result<DataFrame> {
    let mut conn = db::establish_connection(db_path)?;
    let bid = match beamtime_id_for_dir(&mut conn, beamtime_dir)? {
        Some(id) => id,
        None => return catalog_scan_sql_rows_to_dataframe(Vec::new()),
    };
    let sql = build_scan_df_sql(Some(bid), filter);
    let rows: Vec<CatalogScanSql> = sql_query(&sql)
        .load(&mut conn)
        .map_err(CatalogError::Diesel)?;
    catalog_scan_sql_rows_to_dataframe(rows)
}

struct HeaderValueViewSql {
    frame_id: i32,
    scan_number: i32,
    file_path: String,
    frame_number: i32,
    zarr_group_key: i32,
    zarr_frame_index: i32,
    zarr_bucket_frame_index: Option<i32>,
    header_name: String,
    header_display_name: String,
    header_value: f64,
}

impl QueryableByName<Sqlite> for HeaderValueViewSql {
    fn build<'a>(row: &impl diesel::row::NamedRow<'a, Sqlite>) -> deserialize::Result<Self> {
        Ok(Self {
            frame_id: diesel::row::NamedRow::get::<Integer, i32>(row, "frame_id")?,
            scan_number: diesel::row::NamedRow::get::<Integer, i32>(row, "scan_number")?,
            file_path: diesel::row::NamedRow::get::<Text, String>(row, "file_path")?,
            frame_number: diesel::row::NamedRow::get::<Integer, i32>(row, "frame_number")?,
            zarr_group_key: diesel::row::NamedRow::get::<Integer, i32>(row, "zarr_group_key")?,
            zarr_frame_index: diesel::row::NamedRow::get::<Integer, i32>(row, "zarr_frame_index")?,
            zarr_bucket_frame_index: diesel::row::NamedRow::get::<Nullable<Integer>, Option<i32>>(
                row,
                "zarr_bucket_frame_index",
            )?,
            header_name: diesel::row::NamedRow::get::<Text, String>(row, "header_name")?,
            header_display_name: diesel::row::NamedRow::get::<Text, String>(
                row,
                "header_display_name",
            )?,
            header_value: diesel::row::NamedRow::get::<Double, f64>(row, "header_value")?,
        })
    }
}

fn build_header_values_view_sql(beamtime_id: Option<i32>) -> String {
    let mut sql = String::from(
        r#"SELECT
  fr.id AS frame_id,
  sc.scan_number,
  f.nas_uri AS file_path,
  fr.frame_number,
  fr.zarr_group_key,
  fr.zarr_frame_index,
  fr.zarr_bucket_frame_index,
  hc.name AS header_name,
  hc.display_name AS header_display_name,
  hv.value AS header_value
FROM header_values hv
INNER JOIN frames fr ON hv.frame_id = fr.id
INNER JOIN files f ON fr.file_id = f.id
INNER JOIN scans sc ON fr.scan_id = sc.id
INNER JOIN header_cards hc ON hv.header_card_id = hc.id
WHERE 1=1"#,
    );
    if let Some(bid) = beamtime_id {
        sql.push_str(&format!(" AND f.beamtime_id = {bid}"));
    }
    sql.push_str(" ORDER BY sc.scan_number, fr.frame_number, hc.name");
    sql
}

/// Returns a merged frame/header view from `frames`, `header_cards`, and `header_values`.
///
/// When `beamtime_dir` is provided, rows are scoped to that single beamtime root.
pub fn header_values_view(db_path: &Path, beamtime_dir: Option<&Path>) -> Result<DataFrame> {
    let mut conn = db::establish_connection(db_path)?;
    let beamtime_id = match beamtime_dir {
        Some(dir) => match beamtime_id_for_dir(&mut conn, dir)? {
            Some(id) => Some(id),
            None => None,
        },
        None => None,
    };
    if beamtime_dir.is_some() && beamtime_id.is_none() {
        return DataFrame::new(vec![
            Series::new("frame_id".into(), Vec::<i64>::new()).into(),
            Series::new("scan_number".into(), Vec::<i64>::new()).into(),
            Series::new("file_path".into(), Vec::<String>::new()).into(),
            Series::new("frame_number".into(), Vec::<i64>::new()).into(),
            Series::new("zarr_group_key".into(), Vec::<i64>::new()).into(),
            Series::new("zarr_frame_index".into(), Vec::<i64>::new()).into(),
            Series::new("zarr_bucket_frame_index".into(), Vec::<Option<i64>>::new()).into(),
            Series::new("header_name".into(), Vec::<String>::new()).into(),
            Series::new("header_display_name".into(), Vec::<String>::new()).into(),
            Series::new("header_value".into(), Vec::<f64>::new()).into(),
        ])
        .map_err(|e| CatalogError::Validation(e.to_string()));
    }
    let sql = build_header_values_view_sql(beamtime_id);
    let rows: Vec<HeaderValueViewSql> = sql_query(&sql)
        .load(&mut conn)
        .map_err(CatalogError::Diesel)?;
    let mut frame_id = Vec::with_capacity(rows.len());
    let mut scan_number = Vec::with_capacity(rows.len());
    let mut file_path = Vec::with_capacity(rows.len());
    let mut frame_number = Vec::with_capacity(rows.len());
    let mut zarr_group_key = Vec::with_capacity(rows.len());
    let mut zarr_frame_index = Vec::with_capacity(rows.len());
    let mut zarr_bucket_frame_index: Vec<Option<i64>> = Vec::with_capacity(rows.len());
    let mut header_name = Vec::with_capacity(rows.len());
    let mut header_display_name = Vec::with_capacity(rows.len());
    let mut header_value = Vec::with_capacity(rows.len());
    for row in rows {
        frame_id.push(row.frame_id as i64);
        scan_number.push(row.scan_number as i64);
        file_path.push(row.file_path);
        frame_number.push(row.frame_number as i64);
        zarr_group_key.push(row.zarr_group_key as i64);
        zarr_frame_index.push(row.zarr_frame_index as i64);
        zarr_bucket_frame_index.push(row.zarr_bucket_frame_index.map(i64::from));
        header_name.push(row.header_name);
        header_display_name.push(row.header_display_name);
        header_value.push(row.header_value);
    }
    DataFrame::new(vec![
        Series::new("frame_id".into(), frame_id).into(),
        Series::new("scan_number".into(), scan_number).into(),
        Series::new("file_path".into(), file_path).into(),
        Series::new("frame_number".into(), frame_number).into(),
        Series::new("zarr_group_key".into(), zarr_group_key).into(),
        Series::new("zarr_frame_index".into(), zarr_frame_index).into(),
        Series::new("zarr_bucket_frame_index".into(), zarr_bucket_frame_index).into(),
        Series::new("header_name".into(), header_name).into(),
        Series::new("header_display_name".into(), header_display_name).into(),
        Series::new("header_value".into(), header_value).into(),
    ])
    .map_err(|e| CatalogError::Validation(e.to_string()))
}

pub fn sync_missing_headers_for_beamtime(
    db_path: &Path,
    beamtime_dir: &Path,
) -> Result<HeaderSyncReport> {
    use super::ingest::DEFAULT_INGEST_HEADER_ITEMS;
    use std::path::PathBuf;

    let mut conn = db::establish_connection(db_path)?;
    let Some(bid) = beamtime_id_for_dir(&mut conn, beamtime_dir)? else {
        return Ok(HeaderSyncReport::default());
    };
    let frame_rows: Vec<(i32, String)> = frames::table
        .inner_join(files::table.on(frames::file_id.eq(files::id)))
        .filter(files::beamtime_id.eq(bid))
        .select((frames::id, files::nas_uri))
        .load(&mut conn)
        .map_err(CatalogError::Diesel)?;

    let mut card_cache: HashMap<String, i32> = header_cards::table
        .select((header_cards::name, header_cards::id))
        .load(&mut conn)
        .map_err(CatalogError::Diesel)?
        .into_iter()
        .collect();
    let header_items: Vec<String> = DEFAULT_INGEST_HEADER_ITEMS
        .iter()
        .map(|s| (*s).to_string())
        .collect();
    let mut report = HeaderSyncReport::default();

    conn.transaction::<(), CatalogError, _>(|conn| {
        for (frame_id, file_uri) in &frame_rows {
            report.files_checked += 1;
            let Some(path_str) = file_uri.strip_prefix("file://") else {
                continue;
            };
            let row = read_fits_headers_only_row(PathBuf::from(path_str), &header_items)?;
            let existing: Vec<i32> = header_values::table
                .filter(header_values::frame_id.eq(*frame_id))
                .select(header_values::header_card_id)
                .load(conn)?;
            let mut existing_set: HashSet<i32> = existing.into_iter().collect();
            let mut inserted_for_file: u32 = 0;

            for (card_name, value) in row.non_promoted_header_values {
                let card_id = if let Some(id) = card_cache.get(card_name.as_str()) {
                    *id
                } else {
                    let existing_id: Option<i32> = header_cards::table
                        .filter(header_cards::name.eq(card_name.as_str()))
                        .select(header_cards::id)
                        .first(conn)
                        .optional()?;
                    let id = match existing_id {
                        Some(id) => id,
                        None => diesel::insert_into(header_cards::table)
                            .values((
                                header_cards::name.eq(card_name.as_str()),
                                header_cards::display_name
                                    .eq(normalize_header_display_name(card_name.as_str())),
                            ))
                            .returning(header_cards::id)
                            .get_result(conn)?,
                    };
                    card_cache.insert(card_name.clone(), id);
                    id
                };

                if existing_set.insert(card_id) {
                    diesel::insert_into(header_values::table)
                        .values((
                            header_values::frame_id.eq(*frame_id),
                            header_values::header_card_id.eq(card_id),
                            header_values::value.eq(value),
                        ))
                        .execute(conn)?;
                    inserted_for_file += 1;
                }
            }
            if inserted_for_file > 0 {
                report.files_updated += 1;
                report.header_rows_inserted += inserted_for_file;
            }
        }
        Ok(())
    })?;
    Ok(report)
}

pub fn update_beamspot(
    db_path: &Path,
    file_path: &str,
    beam_row: i64,
    beam_col: i64,
    beam_sigma: Option<f64>,
) -> Result<()> {
    let mut conn = db::establish_connection(db_path)?;
    let fid: i32 = frames::table
        .inner_join(files::table.on(frames::file_id.eq(files::id)))
        .filter(files::nas_uri.eq(file_path))
        .select(frames::id)
        .first(&mut conn)
        .map_err(CatalogError::Diesel)?;
    let row_bf: Option<i32> = beam_finding::table
        .filter(beam_finding::frame_id.eq(fid))
        .select(beam_finding::id)
        .first(&mut conn)
        .optional()
        .map_err(CatalogError::Diesel)?;
    let br = Some(beam_row as f64);
    let bc = Some(beam_col as f64);
    if row_bf.is_some() {
        let q = diesel::update(beam_finding::table.filter(beam_finding::frame_id.eq(fid)));
        if let Some(sigma) = beam_sigma {
            q.set((
                beam_finding::centroid_row.eq(br),
                beam_finding::centroid_col.eq(bc),
                beam_finding::fit_std.eq(Some(sigma)),
            ))
            .execute(&mut conn)
            .map_err(CatalogError::Diesel)?;
        } else {
            q.set((
                beam_finding::centroid_row.eq(br),
                beam_finding::centroid_col.eq(bc),
            ))
            .execute(&mut conn)
            .map_err(CatalogError::Diesel)?;
        }
    } else {
        diesel::insert_into(beam_finding::table)
            .values((
                beam_finding::frame_id.eq(fid),
                beam_finding::edge_removal_applied.eq(0i16),
                beam_finding::centroid_row.eq(br),
                beam_finding::centroid_col.eq(bc),
                beam_finding::fit_std.eq(beam_sigma),
                beam_finding::detection_flag.eq("ok"),
            ))
            .execute(&mut conn)
            .map_err(CatalogError::Diesel)?;
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
    let fid = scan_point_uid
        .strip_prefix("frame_")
        .and_then(|s| s.parse::<i32>().ok())
        .ok_or_else(|| {
            CatalogError::Validation(format!("invalid scan_point_uid: {scan_point_uid}"))
        })?;
    let mut conn = db::establish_connection(db_path)?;
    let row_bf: Option<i32> = beam_finding::table
        .filter(beam_finding::frame_id.eq(fid))
        .select(beam_finding::id)
        .first(&mut conn)
        .optional()
        .map_err(CatalogError::Diesel)?;
    let br = Some(beam_row as f64);
    let bc = Some(beam_col as f64);
    if row_bf.is_some() {
        let q = diesel::update(beam_finding::table.filter(beam_finding::frame_id.eq(fid)));
        if let Some(sigma) = beam_sigma {
            q.set((
                beam_finding::centroid_row.eq(br),
                beam_finding::centroid_col.eq(bc),
                beam_finding::fit_std.eq(Some(sigma)),
            ))
            .execute(&mut conn)
            .map_err(CatalogError::Diesel)?;
        } else {
            q.set((
                beam_finding::centroid_row.eq(br),
                beam_finding::centroid_col.eq(bc),
            ))
            .execute(&mut conn)
            .map_err(CatalogError::Diesel)?;
        }
    } else {
        diesel::insert_into(beam_finding::table)
            .values((
                beam_finding::frame_id.eq(fid),
                beam_finding::edge_removal_applied.eq(0i16),
                beam_finding::centroid_row.eq(br),
                beam_finding::centroid_col.eq(bc),
                beam_finding::fit_std.eq(beam_sigma),
                beam_finding::detection_flag.eq("ok"),
            ))
            .execute(&mut conn)
            .map_err(CatalogError::Diesel)?;
    }
    Ok(())
}

pub fn get_scan_point_uid_by_source_path(
    db_path: &Path,
    source_path: &str,
) -> Result<Option<String>> {
    let mut conn = db::establish_connection(db_path)?;
    let fid: Option<i32> = frames::table
        .inner_join(files::table.on(frames::file_id.eq(files::id)))
        .filter(files::nas_uri.eq(source_path))
        .select(frames::id)
        .first(&mut conn)
        .optional()
        .map_err(CatalogError::Diesel)?;
    Ok(fid.map(|id| format!("frame_{id}")))
}

pub fn get_overrides(db_path: &Path, path: Option<&str>) -> Result<DataFrame> {
    let mut conn = db::establish_connection(db_path)?;
    let mut paths_v = Vec::new();
    let mut sample_names = Vec::new();
    let mut tags = Vec::new();
    let mut notes = Vec::new();
    type OverrideRow = (String, Option<String>, Option<String>, Option<String>);
    let rows: Vec<OverrideRow> = if let Some(p) = path {
        file_overrides::table
            .filter(file_overrides::source_path.eq(p))
            .select((
                file_overrides::source_path,
                file_overrides::sample_name,
                file_overrides::tag,
                file_overrides::notes,
            ))
            .load(&mut conn)
            .map_err(CatalogError::Diesel)?
    } else {
        file_overrides::table
            .select((
                file_overrides::source_path,
                file_overrides::sample_name,
                file_overrides::tag,
                file_overrides::notes,
            ))
            .load(&mut conn)
            .map_err(CatalogError::Diesel)?
    };
    for r in rows {
        paths_v.push(r.0);
        sample_names.push(r.1);
        tags.push(r.2);
        notes.push(r.3);
    }
    DataFrame::new(vec![
        Series::new("path".into(), paths_v).into(),
        Series::new("sample_name".into(), sample_names).into(),
        Series::new("tag".into(), tags).into(),
        Series::new("notes".into(), notes).into(),
    ])
    .map_err(|e| CatalogError::Validation(e.to_string()))
}

pub fn set_override(
    db_path: &Path,
    path: &str,
    sample_name: Option<&str>,
    tag: Option<&str>,
    notes: Option<&str>,
) -> Result<()> {
    let mut conn = db::establish_connection(db_path)?;
    let n: i64 = files::table
        .filter(files::nas_uri.eq(path))
        .select(count_star())
        .first(&mut conn)
        .map_err(CatalogError::Diesel)?;
    if n == 0 {
        return Err(CatalogError::Validation(format!(
            "path not in catalog (no files.nas_uri match): {path}"
        )));
    }
    diesel::delete(file_overrides::table.filter(file_overrides::source_path.eq(path)))
        .execute(&mut conn)
        .map_err(CatalogError::Diesel)?;
    diesel::insert_into(file_overrides::table)
        .values((
            file_overrides::source_path.eq(path),
            file_overrides::sample_name.eq(sample_name),
            file_overrides::tag.eq(tag),
            file_overrides::notes.eq(notes),
        ))
        .execute(&mut conn)
        .map_err(CatalogError::Diesel)?;
    Ok(())
}

pub fn set_scan_type_for_beamtime_scan(
    db_path: &Path,
    beamtime_dir: &Path,
    scan_number: i64,
    scan_type: &str,
) -> Result<()> {
    if scan_type != "fixed_energy" && scan_type != "fixed_angle" {
        return Err(CatalogError::Validation(format!(
            "scan_type must be 'fixed_energy' or 'fixed_angle', got: {scan_type}"
        )));
    }
    let mut conn = db::establish_connection(db_path)?;
    let bid = match beamtime_id_for_dir(&mut conn, beamtime_dir)? {
        Some(id) => id,
        None => {
            return Err(CatalogError::Validation(format!(
                "beamtime not found in catalog: {}",
                beamtime_dir.display()
            )))
        }
    };
    let updated = diesel::update(
        scans::table
            .filter(scans::beamtime_id.eq(bid))
            .filter(scans::scan_number.eq(scan_number as i32)),
    )
    .set(scans::scan_type.eq(scan_type))
    .execute(&mut conn)
    .map_err(CatalogError::Diesel)?;
    if updated == 0 {
        return Err(CatalogError::Validation(format!(
            "scan not found for beamtime={} scan_number={scan_number}",
            beamtime_dir.display()
        )));
    }
    Ok(())
}

pub fn rename_file_in_catalog(
    db_path: &Path,
    old_path: &str,
    new_path: &str,
    new_file_name: &str,
    new_sample_name: &str,
    new_tag: Option<&str>,
) -> Result<()> {
    let mut conn = db::establish_connection(db_path)?;
    let n: i64 = files::table
        .filter(files::nas_uri.eq(old_path))
        .select(count_star())
        .first(&mut conn)
        .map_err(CatalogError::Diesel)?;
    if n == 0 {
        return Err(CatalogError::Validation(format!(
            "path not in files.nas_uri: {old_path}"
        )));
    }
    diesel::update(files::table.filter(files::nas_uri.eq(old_path)))
        .set((
            files::nas_uri.eq(new_path),
            files::filename.eq(new_file_name),
        ))
        .execute(&mut conn)
        .map_err(CatalogError::Diesel)?;
    diesel::update(file_overrides::table.filter(file_overrides::source_path.eq(old_path)))
        .set(file_overrides::source_path.eq(new_path))
        .execute(&mut conn)
        .map_err(CatalogError::Diesel)?;
    if !new_sample_name.is_empty() || new_tag.is_some() {
        let sn = (!new_sample_name.is_empty()).then_some(new_sample_name);
        diesel::delete(file_overrides::table.filter(file_overrides::source_path.eq(new_path)))
            .execute(&mut conn)
            .map_err(CatalogError::Diesel)?;
        diesel::insert_into(file_overrides::table)
            .values((
                file_overrides::source_path.eq(new_path),
                file_overrides::sample_name.eq(sn),
                file_overrides::tag.eq(new_tag),
                file_overrides::notes.eq(None::<&str>),
            ))
            .execute(&mut conn)
            .map_err(CatalogError::Diesel)?;
    }
    Ok(())
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
        "zarr_group_key",
        "zarr_frame_index",
        "zarr_bucket_frame_index",
        "zarr_path",
        "DATE",
        "Beamline Energy",
        "Sample Theta",
        "CCD Theta",
        "Higher Order Suppressor",
        "EPU Polarization",
        "EXPOSURE",
        "AI 3 Izero",
        "Sample Name",
        "Scan ID",
        "Lambda",
        "Q",
        "beam_row",
        "beam_col",
        "beam_sigma",
        "reflectivity_profile_index",
        "reflectivity_scan_type",
        "catalog_scan_type",
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::beamtimes;
    use diesel::RunQueryDsl;
    use tempfile::TempDir;

    #[test]
    fn test_scan_from_catalog_empty_db() {
        let tmp = TempDir::new().unwrap();
        let db = tmp.path().join("catalog.db");
        db::establish_connection(&db).unwrap();
        let df = scan_from_catalog(&db, None).unwrap();
        assert_eq!(df.height(), 0);
        for col in scan_from_catalog_columns() {
            assert!(df.column(col).is_ok(), "missing column {}", col);
        }
    }

    #[test]
    fn test_scan_from_catalog_for_beamtime_unknown_root() {
        let tmp = TempDir::new().unwrap();
        let db = tmp.path().join("catalog.db");
        db::establish_connection(&db).unwrap();
        let missing = tmp.path().join("no_such_beamtime");
        let df = scan_from_catalog_for_beamtime(&db, &missing, None).unwrap();
        assert_eq!(df.height(), 0);
        for col in scan_from_catalog_columns() {
            assert!(df.column(col).is_ok(), "missing column {}", col);
        }
    }

    #[test]
    fn test_get_overrides_empty() {
        let tmp = TempDir::new().unwrap();
        let db = tmp.path().join("catalog.db");
        db::establish_connection(&db).unwrap();
        let df = get_overrides(&db, None).unwrap();
        assert_eq!(df.height(), 0);
    }

    #[test]
    fn test_list_beamtime_entries_empty() {
        let tmp = TempDir::new().unwrap();
        let db = tmp.path().join("catalog.db");
        db::establish_connection(&db).unwrap();
        let e = list_beamtime_entries(&db).unwrap();
        assert!(e.samples.is_empty());
        assert!(e.tags.is_empty());
        assert!(e.scans.is_empty());
    }

    #[test]
    fn test_set_override_with_files_row() {
        let tmp = TempDir::new().unwrap();
        let db = tmp.path().join("catalog.db");
        let mut conn = db::establish_connection(&db).unwrap();
        diesel::insert_into(beamtimes::table)
            .values((
                beamtimes::nas_uri.eq("file:///tmp/beam"),
                beamtimes::zarr_path.eq("/tmp/z.zarr"),
                beamtimes::date.eq("2024-01-01"),
                beamtimes::last_indexed_at.eq(Some(0)),
            ))
            .execute(&mut conn)
            .unwrap();
        diesel::insert_into(samples::table)
            .values((
                samples::beamtime_id.eq(1),
                samples::name.eq("orig"),
                samples::representative_x.eq(0.0),
                samples::representative_y.eq(0.0),
                samples::representative_z.eq(0.0),
            ))
            .execute(&mut conn)
            .unwrap();
        diesel::insert_into(scans::table)
            .values((
                scans::beamtime_id.eq(1),
                scans::sample_id.eq(1),
                scans::scan_number.eq(1),
                scans::scan_type.eq("fixed_energy"),
                scans::started_at.eq(None::<String>),
                scans::ended_at.eq(None::<String>),
            ))
            .execute(&mut conn)
            .unwrap();
        diesel::insert_into(files::table)
            .values((
                files::beamtime_id.eq(1),
                files::sample_id.eq(1),
                files::scan_number.eq(1),
                files::frame_number.eq(0),
                files::nas_uri.eq("/data/example.fits"),
                files::filename.eq("example.fits"),
                files::parse_flag.eq(None::<String>),
                files::data_offset.eq(0_i64),
                files::naxis1.eq(0),
                files::naxis2.eq(0),
                files::bitpix.eq(16),
                files::bzero.eq(0_i64),
            ))
            .execute(&mut conn)
            .unwrap();
        diesel::insert_into(frames::table)
            .values((
                frames::scan_id.eq(1),
                frames::file_id.eq(1),
                frames::frame_number.eq(0),
                frames::zarr_group_key.eq(1),
                frames::zarr_frame_index.eq(0),
                frames::zarr_bucket_frame_index.eq(None::<i32>),
                frames::acquired_at.eq(None::<String>),
                frames::quality_flag.eq(None::<String>),
            ))
            .execute(&mut conn)
            .unwrap();
        drop(conn);
        set_override(
            &db,
            "/data/example.fits",
            Some("corrected"),
            Some("t2"),
            None,
        )
        .unwrap();
        let df = scan_from_catalog(&db, None).unwrap();
        assert_eq!(df.height(), 1);
        let sn = df.column("sample_name").unwrap().str().unwrap().get(0);
        assert_eq!(sn, Some("corrected"));
        let tg = df.column("tag").unwrap().str().unwrap().get(0);
        assert_eq!(tg, Some("t2"));
        let all_ov = get_overrides(&db, None).unwrap();
        assert_eq!(all_ov.height(), 1);
        let p = all_ov.column("path").unwrap().str().unwrap().get(0);
        assert_eq!(p, Some("/data/example.fits"));
    }
}

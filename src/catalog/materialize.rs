#![cfg(feature = "zarr")]

use crate::catalog::{open_or_create_db, resolve_catalog_path, CatalogError, Result};
use crate::io::image_mmap::materialize_image;
use crate::io::zarr_store::{
    create_detector_array, detector_zarr_group, open_store, write_detector_frame, zarr_root,
};
use crate::io::ImageInfo;
use ndarray::Array2;
use std::path::Path;
use std::sync::mpsc;

struct ScanPointSource {
    uid: String,
    source_path: String,
    source_data_offset: i64,
    source_naxis1: i64,
    source_naxis2: i64,
    source_bitpix: i64,
    source_bzero: i64,
    seq_index: i64,
}

fn get_scan_uids_for_beamtime(conn: &rusqlite::Connection, beamtime_id: i64) -> Result<Vec<String>> {
    let mut stmt = conn.prepare("SELECT uid FROM bt_scans WHERE beamtime_id = ?1 ORDER BY uid")?;
    let uids: Vec<String> = stmt
        .query_map(rusqlite::params![beamtime_id], |r| r.get(0))?
        .filter_map(|r| r.ok())
        .collect();
    Ok(uids)
}

fn get_scan_points_with_source(
    conn: &rusqlite::Connection,
    scan_uid: &str,
) -> Result<Vec<ScanPointSource>> {
    let mut stmt = conn.prepare(
        "SELECT uid, source_path, source_data_offset, source_naxis1, source_naxis2, source_bitpix, source_bzero, seq_index
         FROM bt_scan_points WHERE scan_uid = ?1 AND source_path IS NOT NULL ORDER BY seq_index",
    )?;
    let rows = stmt.query_map(rusqlite::params![scan_uid], |r| {
        Ok(ScanPointSource {
            uid: r.get(0)?,
            source_path: r.get(1)?,
            source_data_offset: r.get(2)?,
            source_naxis1: r.get(3)?,
            source_naxis2: r.get(4)?,
            source_bitpix: r.get(5)?,
            source_bzero: r.get(6)?,
            seq_index: r.get(7)?,
        })
    })?;
    rows.map(|r| r.map_err(CatalogError::Sqlite)).collect()
}

fn insert_image_ref(
    conn: &rusqlite::Connection,
    scan_point_uid: &str,
    zarr_group: &str,
    index_in_stack: i64,
    shape_x: i64,
    shape_y: i64,
    dtype: &str,
) -> Result<()> {
    conn.execute(
        "INSERT OR REPLACE INTO bt_image_refs (scan_point_uid, field_name, zarr_group, index_in_stack, shape_x, shape_y, dtype)
         VALUES (?1, 'detector_image', ?2, ?3, ?4, ?5, ?6)",
        rusqlite::params![
            scan_point_uid,
            zarr_group,
            index_in_stack,
            shape_x,
            shape_y,
            dtype
        ],
    )?;
    Ok(())
}

pub fn materialize_beamtime(
    beamtime_dir: &Path,
    scan_uids_filter: Option<&[String]>,
    progress_tx: Option<mpsc::Sender<(String, u32, u32)>>,
) -> Result<()> {
    let _ = resolve_catalog_path(beamtime_dir);
    let conn = open_or_create_db(beamtime_dir)?;
    let beamtime_id: i64 = conn.query_row(
        "SELECT id FROM bt_beamtimes WHERE beamtime_path = ?1",
        rusqlite::params![beamtime_dir.to_string_lossy().to_string()],
        |r| r.get(0),
    )?;
    let scan_uids: Vec<String> = match scan_uids_filter {
        Some(list) => list.to_vec(),
        None => get_scan_uids_for_beamtime(&conn, beamtime_id)?,
    };
    let zarr_root_path = zarr_root(beamtime_dir);
    let store = open_store(&zarr_root_path)?;
    for scan_uid in &scan_uids {
        let points = get_scan_points_with_source(&conn, scan_uid)?;
        if points.is_empty() {
            continue;
        }
        let n_frames = points.len();
        let first = &points[0];
        let info = ImageInfo {
            path: std::path::PathBuf::from(&first.source_path),
            data_offset: first.source_data_offset as u64,
            naxis1: first.source_naxis1 as usize,
            naxis2: first.source_naxis2 as usize,
            bitpix: first.source_bitpix as i32,
            bzero: first.source_bzero,
        };
        let (_, processed) = materialize_image(info.path.as_path(), &info)?;
        let height = processed.nrows();
        let width = processed.ncols();
        let array = create_detector_array(&store, scan_uid, n_frames, height, width)?;
        let zarr_group = detector_zarr_group(scan_uid);
        for (idx, pt) in points.iter().enumerate() {
            let info_pt = ImageInfo {
                path: std::path::PathBuf::from(&pt.source_path),
                data_offset: pt.source_data_offset as u64,
                naxis1: pt.source_naxis1 as usize,
                naxis2: pt.source_naxis2 as usize,
                bitpix: pt.source_bitpix as i32,
                bzero: pt.source_bzero,
            };
            let (_, proc) = materialize_image(
                std::path::Path::new(&pt.source_path),
                &info_pt,
            )?;
            let clamped: Array2<i32> = Array2::from_shape_fn(proc.dim(), |(r, c)| {
                let v = proc[[r, c]];
                v.clamp(i32::MIN as i64, i32::MAX as i64) as i32
            });
            write_detector_frame(&array, idx, &clamped)?;
            insert_image_ref(
                &conn,
                &pt.uid,
                &zarr_group,
                pt.seq_index,
                width as i64,
                height as i64,
                "int32",
            )?;
            if let Some(ref tx) = progress_tx {
                let _ = tx.send((scan_uid.clone(), idx as u32 + 1, n_frames as u32));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::materialize_beamtime;
    use crate::catalog::Result;
    use std::path::Path;
    use std::sync::mpsc::Sender;

    #[test]
    fn test_materialize_symbol_is_feature_enabled() {
        let function_ptr: fn(&Path, Option<&[String]>, Option<Sender<(String, u32, u32)>>) -> Result<()>
            = materialize_beamtime;
        let _ = function_ptr;
    }
}

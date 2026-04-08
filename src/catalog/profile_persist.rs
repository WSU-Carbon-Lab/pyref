//! Writes reflectivity profile segmentation into SQLite after ingest so metadata stays tied to the
//! indexed beamtime (`bt_reflectivity_profiles` + `bt_scan_points.reflectivity_profile_index`).

use rusqlite::Connection;

use crate::catalog::reflectivity_profile::{
    classify_scan_type, segment_reflectivity_profiles, ReflectivityScanType,
};
use crate::catalog::Result;

fn scan_type_to_sql(st: ReflectivityScanType) -> &'static str {
    match st {
        ReflectivityScanType::FixedEnergy => "fixed_energy",
        ReflectivityScanType::FixedAngle => "fixed_angle",
        ReflectivityScanType::SinglePoint => "single_point",
    }
}

/// Recomputes profile segments for every scan in the beamtime and updates `bt_scan_points`.
///
/// Call after ingest commits so `seq_index` and energy/theta columns match the on-disk index.
/// Safe to call on empty beamtimes (no-op).
pub(crate) fn recompute_reflectivity_profiles_for_beamtime(
    conn: &Connection,
    beamtime_id: i64,
) -> Result<()> {
    let mut stmt = conn.prepare("SELECT uid FROM bt_scans WHERE beamtime_id = ?1 ORDER BY uid")?;
    let scan_uids: Vec<String> = stmt
        .query_map([beamtime_id], |r| r.get(0))?
        .filter_map(std::result::Result::ok)
        .collect();
    for uid in scan_uids {
        recompute_reflectivity_profiles_for_scan(conn, &uid)?;
    }
    Ok(())
}

fn recompute_reflectivity_profiles_for_scan(conn: &Connection, scan_uid: &str) -> Result<()> {
    let mut stmt = conn.prepare(
        "SELECT uid, seq_index, beamline_energy, sample_theta FROM bt_scan_points WHERE scan_uid = ?1 ORDER BY seq_index ASC",
    )?;
    let rows: Vec<(String, i64, Option<f64>, Option<f64>)> = stmt
        .query_map([scan_uid], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))?
        .filter_map(std::result::Result::ok)
        .collect();
    conn.execute(
        "UPDATE bt_scan_points SET reflectivity_profile_index = NULL WHERE scan_uid = ?1",
        [scan_uid],
    )?;
    conn.execute(
        "DELETE FROM bt_reflectivity_profiles WHERE scan_uid = ?1",
        [scan_uid],
    )?;
    if rows.is_empty() {
        return Ok(());
    }
    let pairs: Vec<(Option<f64>, Option<f64>)> =
        rows.iter().map(|(_, _, e, t)| (*e, *t)).collect();
    let segs = segment_reflectivity_profiles(&pairs);
    let mut insert_prof = conn.prepare(
        r#"INSERT INTO bt_reflectivity_profiles (
            scan_uid, profile_index, scan_type, seq_index_first, seq_index_last,
            e_min, e_max, t_min, t_max
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)"#,
    )?;
    let mut upd = conn.prepare(
        "UPDATE bt_scan_points SET reflectivity_profile_index = ?1 WHERE uid = ?2",
    )?;
    for (pi, seg) in segs.iter().enumerate() {
        let slice_pairs = &pairs[seg.start..seg.end];
        let (_st, emin, emax, tmin, tmax) = classify_scan_type(slice_pairs);
        let seq_first = rows[seg.start].1;
        let seq_last = rows[seg.end - 1].1;
        insert_prof.execute(rusqlite::params![
            scan_uid,
            pi as i64,
            scan_type_to_sql(seg.scan_type),
            seq_first,
            seq_last,
            emin,
            emax,
            tmin,
            tmax,
        ])?;
        for r in rows.iter().take(seg.end).skip(seg.start) {
            upd.execute(rusqlite::params![pi as i64, &r.0])?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::recompute_reflectivity_profiles_for_beamtime;
    use tempfile::TempDir;

    #[test]
    fn recompute_sets_profile_index_and_summary_row() {
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
                "INSERT INTO bt_samples (beamtime_id, name) VALUES (1, 's')",
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
            for (uid, seq, e, th) in [
                ("sp_1_1_0", 0, 280.0_f64, 0.0_f64),
                ("sp_1_1_1", 1, 280.0_f64, 1.0_f64),
                ("sp_1_1_2", 2, 280.0_f64, 2.0_f64),
            ] {
                conn.execute(
                    r#"INSERT INTO bt_scan_points (
                        uid, stream_uid, scan_uid, sample_id, seq_index,
                        beamline_energy, sample_theta,
                        source_path, source_data_offset, source_naxis1, source_naxis2, source_bitpix, source_bzero
                    ) VALUES (?1, 'st_1_1', 's_1_1', 1, ?2, ?3, ?4, '', 0, 0, 0, 0, 0)"#,
                    rusqlite::params![uid, seq, e, th],
                )
                .unwrap();
            }
            recompute_reflectivity_profiles_for_beamtime(&conn, 1).unwrap();
            let n: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM bt_reflectivity_profiles WHERE scan_uid = 's_1_1'",
                    [],
                    |r| r.get(0),
                )
                .unwrap();
            assert_eq!(n, 1);
            let idx: i64 = conn
                .query_row(
                    "SELECT reflectivity_profile_index FROM bt_scan_points WHERE uid = 'sp_1_1_1'",
                    [],
                    |r| r.get(0),
                )
                .unwrap();
            assert_eq!(idx, 0);
        }
    }

    #[test]
    fn recompute_izero_then_ramp_two_profiles() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("catalog.db");
        let conn = crate::catalog::open_catalog_db(&db_path).unwrap();
        conn.execute(
            "INSERT INTO bt_beamtimes (beamtime_path) VALUES (?1)",
            ["/tmp/beam"],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO bt_samples (beamtime_id, name) VALUES (1, 's')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO bt_scans (uid, beamtime_id, sample_id) VALUES ('s_1_2', 1, 1)",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO bt_streams (uid, scan_uid) VALUES ('st_1_2', 's_1_2')",
            [],
        )
        .unwrap();
        let mut seq = 0_i64;
        for _ in 0..4 {
            let uid = format!("sp_1_2_{}", seq);
            conn.execute(
                r#"INSERT INTO bt_scan_points (
                    uid, stream_uid, scan_uid, sample_id, seq_index,
                    beamline_energy, sample_theta,
                    source_path, source_data_offset, source_naxis1, source_naxis2, source_bitpix, source_bzero
                ) VALUES (?1, 'st_1_2', 's_1_2', 1, ?2, 284.0, 0.0, '', 0, 0, 0, 0, 0)"#,
                rusqlite::params![uid, seq],
            )
            .unwrap();
            seq += 1;
        }
        for i in 1..6 {
            let uid = format!("sp_1_2_{}", seq);
            conn.execute(
                r#"INSERT INTO bt_scan_points (
                    uid, stream_uid, scan_uid, sample_id, seq_index,
                    beamline_energy, sample_theta,
                    source_path, source_data_offset, source_naxis1, source_naxis2, source_bitpix, source_bzero
                ) VALUES (?1, 'st_1_2', 's_1_2', 1, ?2, 284.0, ?3, '', 0, 0, 0, 0, 0)"#,
                rusqlite::params![uid, seq, i as f64 * 0.5],
            )
            .unwrap();
            seq += 1;
        }
        recompute_reflectivity_profiles_for_beamtime(&conn, 1).unwrap();
        let n: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM bt_reflectivity_profiles WHERE scan_uid = 's_1_2'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n, 2);
    }
}

#![cfg(feature = "catalog")]

use std::collections::HashMap;
use std::io::Cursor;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use diesel::prelude::*;
use pyref::catalog::db;
use pyref::catalog::paths;
use pyref::catalog::zarr_write::{
    prepare_shape_scan_bucket_arrays, scan_raw_array_path, shape_bucket_key, ShapeScanBucketSpec,
    ZARR_U16_ZSTD_LEVEL,
};
use pyref::schema::{beamtimes, files, frames};
use zarrs::array::{Array, ArraySubset};
use zarrs::storage::{ReadableWritableListableStorage, ReadableWritableListableStorageTraits};

#[derive(Debug, Clone)]
struct CliOptions {
    db_path: Option<PathBuf>,
    beamtime_id: Option<i32>,
    dry_run: bool,
    keep_legacy: bool,
}

#[derive(Debug, Queryable)]
struct BeamtimeRow {
    id: i32,
    zarr_path: String,
}

#[derive(Debug, Queryable)]
struct FrameRow {
    id: i32,
    zarr_group_key: i32,
    zarr_frame_index: i32,
}

#[derive(Debug, Clone)]
struct FramePlan {
    frame_id: i32,
    group_key: i32,
    frame_index: i32,
    shape_bucket: String,
    bucket_frame_index: i32,
    height: usize,
    width: usize,
}

const ETA_SAMPLE_FRAMES: usize = 8;

fn parse_args() -> Result<CliOptions, String> {
    let mut args = std::env::args().skip(1);
    let mut out = CliOptions {
        db_path: None,
        beamtime_id: None,
        dry_run: false,
        keep_legacy: false,
    };
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--db" => {
                let Some(v) = args.next() else {
                    return Err("--db requires a path".into());
                };
                out.db_path = Some(PathBuf::from(v));
            }
            "--beamtime-id" => {
                let Some(v) = args.next() else {
                    return Err("--beamtime-id requires an integer value".into());
                };
                let parsed = v
                    .parse::<i32>()
                    .map_err(|_| format!("invalid --beamtime-id value: {v}"))?;
                out.beamtime_id = Some(parsed);
            }
            "--dry-run" => out.dry_run = true,
            "--keep-legacy" => out.keep_legacy = true,
            "--help" | "-h" => {
                return Err(
                    "Usage: migrate-zarr-3d [--db <catalog.db>] [--beamtime-id <id>] [--dry-run] [--keep-legacy]"
                        .into(),
                );
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    Ok(out)
}

fn shuffle_encode_zarr_bytes(decoded: &[u8], elementsize: usize) -> Vec<u8> {
    let mut encoded_value = decoded.to_vec();
    let count = encoded_value.len() / elementsize;
    for i in 0..count {
        let offset = i * elementsize;
        for byte_index in 0..elementsize {
            let j = byte_index * count + i;
            encoded_value[j] = decoded[offset + byte_index];
        }
    }
    encoded_value
}

fn shuffle_zstd_compressed_len_u16(pixels: &[u16]) -> Result<usize, String> {
    let mut le = Vec::with_capacity(pixels.len() * 2);
    for &v in pixels {
        le.extend_from_slice(&v.to_le_bytes());
    }
    let shuffled = shuffle_encode_zarr_bytes(&le, 2);
    zstd::encode_all(Cursor::new(shuffled), ZARR_U16_ZSTD_LEVEL)
        .map(|v| v.len())
        .map_err(|e| e.to_string())
}

fn load_legacy_frame(
    store: &ReadableWritableListableStorage,
    group_key: i32,
    frame_index: i32,
) -> Result<
    Option<(
        Array<dyn ReadableWritableListableStorageTraits>,
        usize,
        usize,
    )>,
    String,
> {
    let path = format!("/{group_key}/{frame_index:05}/raw");
    let Ok(array) = Array::open(store.clone(), &path) else {
        return Ok(None);
    };
    let shape = array.shape().to_vec();
    if shape.len() != 2 {
        return Ok(None);
    }
    Ok(Some((array, shape[0] as usize, shape[1] as usize)))
}

fn sample_frame_indices(frame_count: usize, max_samples: usize) -> Vec<usize> {
    if frame_count == 0 {
        return Vec::new();
    }
    let k = max_samples.min(frame_count).max(1);
    if k == 1 {
        return vec![0];
    }
    (0..k).map(|i| i * (frame_count - 1) / (k - 1)).collect()
}

fn format_duration_hms(secs: f64) -> String {
    if secs < 60.0 {
        return format!("{secs:.0}s");
    }
    if secs < 3600.0 {
        let m = (secs / 60.0).floor() as u64;
        let s = secs - (m as f64) * 60.0;
        return format!("{m}m {s:.0}s");
    }
    let h = (secs / 3600.0).floor() as u64;
    let rem = secs - (h as f64) * 3600.0;
    let m = (rem / 60.0).floor() as u64;
    format!("{h}h {m}m")
}

fn retrieve_legacy_pixels_i32(
    array: &Array<dyn ReadableWritableListableStorageTraits>,
    height: usize,
    width: usize,
) -> Result<Vec<i32>, String> {
    let subset_region = ArraySubset::new_with_ranges(&[0..height as u64, 0..width as u64]);
    let subset: Vec<i32> = array
        .retrieve_array_subset::<Vec<i32>>(&subset_region)
        .map_err(|e| e.to_string())?;
    Ok(subset)
}

fn migrate_beamtime(
    conn: &mut SqliteConnection,
    beamtime: &BeamtimeRow,
    options: &CliOptions,
) -> Result<(), String> {
    let zarr_root = PathBuf::from(&beamtime.zarr_path);
    if !zarr_root.exists() {
        println!(
            "beamtime {}: zarr store missing at {}, skipping",
            beamtime.id,
            zarr_root.display()
        );
        return Ok(());
    }
    let store: ReadableWritableListableStorage =
        Arc::new(zarrs::filesystem::FilesystemStore::new(&zarr_root).map_err(|e| e.to_string())?);
    let frame_rows: Vec<FrameRow> = frames::table
        .inner_join(files::table.on(files::id.eq(frames::file_id)))
        .filter(files::beamtime_id.eq(beamtime.id))
        .select((frames::id, frames::zarr_group_key, frames::zarr_frame_index))
        .order((
            frames::zarr_group_key.asc(),
            frames::zarr_frame_index.asc(),
            frames::id.asc(),
        ))
        .load(conn)
        .map_err(|e| e.to_string())?;
    if frame_rows.is_empty() {
        return Ok(());
    }

    let mut specs_by_combo: HashMap<(String, i32), ShapeScanBucketSpec> = HashMap::new();
    let mut plans: Vec<FramePlan> = Vec::new();
    let mut next_idx: HashMap<(String, i32), i32> = HashMap::new();
    for row in &frame_rows {
        let Some((_array, h, w)) =
            load_legacy_frame(&store, row.zarr_group_key, row.zarr_frame_index)?
        else {
            continue;
        };
        let bucket = shape_bucket_key(h, w);
        let gk = row.zarr_group_key;
        let combo = (bucket.clone(), gk);
        let idx = next_idx.entry(combo.clone()).or_insert(0);
        plans.push(FramePlan {
            frame_id: row.id,
            group_key: gk,
            frame_index: row.zarr_frame_index,
            shape_bucket: bucket.clone(),
            bucket_frame_index: *idx,
            height: h,
            width: w,
        });
        *idx += 1;
        specs_by_combo
            .entry(combo)
            .and_modify(|s| s.frames += 1)
            .or_insert(ShapeScanBucketSpec {
                shape_bucket: bucket,
                scan_number: gk,
                height: h,
                width: w,
                frames: 1,
            });
    }
    if plans.is_empty() {
        println!("beamtime {}: no legacy frames found", beamtime.id);
        return Ok(());
    }
    let mut specs: Vec<ShapeScanBucketSpec> = specs_by_combo.into_values().collect();
    specs.sort_by(|a, b| {
        (a.shape_bucket.as_str(), a.scan_number).cmp(&(b.shape_bucket.as_str(), b.scan_number))
    });
    if !options.dry_run {
        prepare_shape_scan_bucket_arrays(&store, &specs).map_err(|e| e.to_string())?;
    }

    let mut migrated_frames = 0usize;
    if options.dry_run {
        migrated_frames = plans.len();
        let total_pixels: u128 = plans.iter().map(|p| (p.height * p.width) as u128).sum();
        let gib = 1024.0_f64.powi(3);
        let legacy_read_gib = (total_pixels as f64 * 4.0) / gib;
        let new_write_gib = (total_pixels as f64 * 2.0) / gib;
        let mpix = total_pixels as f64 / 1_000_000.0;
        let sample_idxs = sample_frame_indices(plans.len(), ETA_SAMPLE_FRAMES);
        let mut sample_time_secs = 0.0_f64;
        let mut sample_pixels: u128 = 0;
        let mut sample_ok: usize = 0;
        let mut sample_raw_u16_bytes: u128 = 0;
        let mut sample_compressed_bytes: u128 = 0;
        for &idx in &sample_idxs {
            let plan = &plans[idx];
            let Some((legacy_array, _, _)) =
                load_legacy_frame(&store, plan.group_key, plan.frame_index)?
            else {
                continue;
            };
            let t0 = Instant::now();
            let legacy = retrieve_legacy_pixels_i32(&legacy_array, plan.height, plan.width)?;
            let converted: Vec<u16> = legacy.into_iter().map(|v| (v as i16) as u16).collect();
            sample_time_secs += t0.elapsed().as_secs_f64();
            sample_pixels += (plan.height * plan.width) as u128;
            sample_ok += 1;
            let raw_b = converted.len() * 2;
            sample_raw_u16_bytes += raw_b as u128;
            sample_compressed_bytes += shuffle_zstd_compressed_len_u16(&converted)? as u128;
        }
        let eta_note = if sample_pixels > 0 && sample_time_secs > 0.0 {
            let scaled = sample_time_secs * (total_pixels as f64 / sample_pixels as f64);
            let wall = format_duration_hms(scaled);
            format!(
                "estimated wall time ~{wall} (decode+u16 remap from {sample_ok} sample frames; omits DB updates, Zarr chunk headers, legacy removal)",
            )
        } else {
            "estimated wall time unavailable (samples missing or zero timing)".to_string()
        };
        let compress_note = if sample_raw_u16_bytes > 0 && sample_compressed_bytes > 0 {
            let ratio = sample_compressed_bytes as f64 / sample_raw_u16_bytes as f64;
            let est_gib = (new_write_gib * 1024.0_f64.powi(3) * ratio) / 1024.0_f64.powi(3);
            format!(
                "shuffle+u16 Zstd level {ZARR_U16_ZSTD_LEVEL} on {sample_ok} sample frames: ~{:.2}% of raw uint16 (~{:.2} GiB vs ~{:.2} GiB); on-disk Zarr may differ slightly",
                ratio * 100.0,
                est_gib,
                new_write_gib
            )
        } else {
            "compression sample unavailable".to_string()
        };
        println!(
            "beamtime {}: dry-run OK, {} frames -> {} shape x scan arrays, {:.2} Mpix total",
            beamtime.id,
            migrated_frames,
            specs.len(),
            mpix
        );
        println!(
            "  resources: ~{legacy_read_gib:.2} GiB legacy i32 read, ~{new_write_gib:.2} GiB uint16 payload (uncompressed)",
        );
        println!("  {compress_note}");
        println!("  {eta_note}");
        return Ok(());
    }
    for plan in &plans {
        let Some((legacy_array, _, _)) =
            load_legacy_frame(&store, plan.group_key, plan.frame_index)?
        else {
            continue;
        };
        let legacy = retrieve_legacy_pixels_i32(&legacy_array, plan.height, plan.width)?;
        let converted: Vec<u16> = legacy.into_iter().map(|v| (v as i16) as u16).collect();
        let bucket_path = scan_raw_array_path(&plan.shape_bucket, plan.group_key);
        let bucket_array = Array::open(store.clone(), &bucket_path).map_err(|e| e.to_string())?;
        let subset = ArraySubset::new_with_start_shape(
            vec![plan.bucket_frame_index as u64, 0_u64, 0_u64],
            vec![1_u64, plan.height as u64, plan.width as u64],
        )
        .map_err(|e| e.to_string())?;
        bucket_array
            .store_array_subset(&subset, converted)
            .map_err(|e| e.to_string())?;
        diesel::update(frames::table.filter(frames::id.eq(plan.frame_id)))
            .set((
                frames::zarr_shape_bucket.eq(Some(plan.shape_bucket.as_str())),
                frames::zarr_bucket_frame_index.eq(Some(plan.bucket_frame_index)),
            ))
            .execute(conn)
            .map_err(|e| e.to_string())?;
        if !options.keep_legacy {
            let raw_dir = zarr_root
                .join(plan.group_key.to_string())
                .join(format!("{:05}", plan.frame_index))
                .join("raw");
            if raw_dir.exists() {
                let _ = std::fs::remove_dir_all(&raw_dir);
            }
        }
        migrated_frames += 1;
    }
    println!(
        "beamtime {}: migrated {} frames into {} per-scan shape bucket arrays",
        beamtime.id,
        migrated_frames,
        specs.len(),
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = parse_args().map_err(std::io::Error::other)?;
    let db_path = if let Some(path) = options.db_path.clone() {
        path
    } else {
        paths::default_catalog_db_path()?
    };
    let mut conn = db::establish_connection(&db_path)?;
    let mut beamtimes_q = beamtimes::table
        .select((beamtimes::id, beamtimes::zarr_path))
        .into_boxed();
    if let Some(beamtime_id) = options.beamtime_id {
        beamtimes_q = beamtimes_q.filter(beamtimes::id.eq(beamtime_id));
    }
    let rows: Vec<BeamtimeRow> = beamtimes_q.load(&mut conn).map_err(std::io::Error::other)?;
    if rows.is_empty() {
        println!("no beamtimes selected");
        return Ok(());
    }
    for beamtime in &rows {
        migrate_beamtime(&mut conn, beamtime, &options).map_err(std::io::Error::other)?;
    }
    Ok(())
}

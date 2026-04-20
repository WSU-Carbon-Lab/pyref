//! Beamtime ingest: Diesel catalog rows and zarr arrays in one pass.
//!
//! Ingest runs in three phases: headers (parallel FITS header parsing),
//! catalog (batched SQLite transactions of scans/files/tags/frames), and
//! zarr (raw pixel write). On mid-ingest failure, batches that already
//! committed remain in the catalog and partial zarr writes remain on disk.
//! Re-invoking `ingest_beamtime*` on the same beamtime directory deletes
//! and re-inserts the `beamtimes` row, cascading away stale scan/file
//! rows before inserting fresh data.
//!
//! SQLite allows one writer at a time. Ingest uses a single [`diesel::SqliteConnection`] for all
//! catalog mutations in this process. After `fork` or when spawning a subprocess, open a new
//! connection in the child; do not share a connection across process boundaries.
//!
//! FITS header reads and pixel reads for zarr use a [`rayon::ThreadPool`] sized by
//! [`crate::catalog::IngestParallelism`] (after [`IngestParallelism::from_options_or_env`]). The
//! headers phase parallelizes **per file** (a flat [`rayon::prelude::ParallelIterator`] over every
//! discovered FITS path) so worker utilization stays even when scan sizes are skewed. Diesel
//! transactions and [`super::zarr_write::write_frame_raw`] run on the calling thread in global row
//! order so catalog rows and zarr datasets stay aligned.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use diesel::prelude::*;
use diesel::OptionalExtension;
use ndarray::Array2;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

use crate::io::raw_pixels::read_bitpix16_be_bytes;
use crate::io::BtIngestRow;
use crate::loader::read_fits_headers_only_row;
use crate::schema::{beamtimes, file_tags, files, frames, samples, scans, tags};

use super::discover_paths_for_catalog_ingest;
use super::ingest_progress::{
    layout_and_groups_from_paths, BeamtimeIngestLayout, IngestPhase, IngestProgress,
    IngestProgressSink,
};
use super::parallelism::IngestParallelism;
use super::zarr_write::{
    open_zarr_store, prepare_shape_scan_bucket_arrays, shape_bucket_key,
    write_scan_shape_bucket_frame_raw, ShapeScanBucketSpec,
};
use super::{db, paths, CatalogError, Result};

/// Optional subset of scans to ingest (disk order is ascending scan number; explicit
/// [`scan_numbers`](Self::scan_numbers) preserves caller order).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IngestSelection {
    pub max_scans: Option<u32>,
    pub scan_numbers: Option<Vec<i32>>,
}

fn filter_paths_by_selection(
    paths_only: Vec<PathBuf>,
    selection: &IngestSelection,
) -> Result<(Vec<PathBuf>, BeamtimeIngestLayout)> {
    let (_, scan_groups) = layout_and_groups_from_paths(&paths_only);
    if selection.max_scans.is_some() && selection.scan_numbers.is_some() {
        return Err(CatalogError::Validation(
            "ingest selection: use only one of max_scans or scan_numbers".into(),
        ));
    }
    let selected_groups: Vec<(i32, Vec<PathBuf>)> = if let Some(ref wanted) = selection.scan_numbers
    {
        if wanted.is_empty() {
            return Err(CatalogError::Validation(
                "scan_numbers must not be empty when set".into(),
            ));
        }
        let map: HashMap<i32, Vec<PathBuf>> = scan_groups.iter().cloned().collect();
        let mut out = Vec::with_capacity(wanted.len());
        for &sn in wanted {
            match map.get(&sn) {
                Some(paths) => out.push((sn, paths.clone())),
                None => {
                    return Err(CatalogError::Validation(format!(
                        "scan_number {sn} not found under beamtime"
                    )));
                }
            }
        }
        out
    } else if let Some(n) = selection.max_scans {
        if n == 0 {
            return Err(CatalogError::Validation(
                "max_scans must be at least 1 when set".into(),
            ));
        }
        scan_groups.into_iter().take(n as usize).collect()
    } else {
        scan_groups
    };

    let filtered_paths: Vec<PathBuf> = selected_groups
        .iter()
        .flat_map(|(_, ps)| ps.iter().cloned())
        .collect();

    if filtered_paths.is_empty() {
        return Err(CatalogError::Validation(
            "ingest selection produced no FITS paths".into(),
        ));
    }

    let (layout, _) = layout_and_groups_from_paths(&filtered_paths);
    Ok((filtered_paths, layout))
}

/// Shared per-ingest state passed through the three phase functions.
///
/// Borrows caller-owned `progress`, `cancel`, and `pool`. Owns the per-scan
/// total map so phases can emit progress events without rebuilding it.
struct IngestContext<'a> {
    beamtime_id: i32,
    progress: Option<&'a IngestProgressSink>,
    cancel: Option<&'a AtomicBool>,
    pool: &'a rayon::ThreadPool,
    scan_total_map: HashMap<i32, u32>,
}

impl IngestContext<'_> {
    fn is_cancelled(&self) -> bool {
        self.cancel.is_some_and(|c| c.load(Ordering::Relaxed))
    }
}

const MAX_BATCH_ROWS: usize = 1000;

fn plan_catalog_batches(rows: &[BtIngestRow]) -> Vec<(usize, usize)> {
    let mut batches: Vec<(usize, usize)> = Vec::new();
    if rows.is_empty() {
        return batches;
    }

    let mut scan_segments: Vec<(usize, usize)> = Vec::new();
    let mut seg_start = 0_usize;
    for i in 1..rows.len() {
        if rows[i].scan_number != rows[seg_start].scan_number {
            scan_segments.push((seg_start, i));
            seg_start = i;
        }
    }
    scan_segments.push((seg_start, rows.len()));

    let mut batch_start = scan_segments[0].0;
    let mut batch_rows: usize = 0;
    for (s, e) in scan_segments {
        let scan_rows = e - s;
        if batch_rows > 0 && batch_rows + scan_rows > MAX_BATCH_ROWS {
            batches.push((batch_start, s));
            batch_start = s;
            batch_rows = 0;
        }
        batch_rows += scan_rows;
    }
    batches.push((batch_start, rows.len()));
    batches
}

pub const DEFAULT_INGEST_HEADER_ITEMS: &[&str] = &[
    "DATE",
    "Beamline Energy",
    "Sample Theta",
    "CCD Theta",
    "Higher Order Suppressor",
    "EPU Polarization",
    "EXPOSURE",
    "Sample Name",
    "Scan ID",
    "Sample X",
    "Sample Y",
    "Sample Z",
    "RINGCRNT",
    "AI 3 Izero",
    "Beam Current",
];

#[derive(Debug, Clone, PartialEq, Eq)]
struct FrameZarrAssignment {
    shape_bucket: String,
    scan_number: i32,
    bucket_frame_index: i32,
}

fn build_frame_zarr_plan(
    rows: &[BtIngestRow],
) -> (Vec<FrameZarrAssignment>, Vec<ShapeScanBucketSpec>) {
    let mut combo_counts: HashMap<(String, i32), usize> = HashMap::new();
    let mut combo_dims: HashMap<(String, i32), (usize, usize)> = HashMap::new();
    for row in rows {
        let height = row.naxis2 as usize;
        let width = row.naxis1 as usize;
        let bucket = shape_bucket_key(height, width);
        let scan_number = row.scan_number as i32;
        let key = (bucket.clone(), scan_number);
        *combo_counts.entry(key.clone()).or_insert(0) += 1;
        combo_dims.entry(key).or_insert((height, width));
    }
    let mut combos: Vec<(String, i32)> = combo_counts.keys().cloned().collect();
    combos.sort_by(|a, b| (a.0.as_str(), a.1).cmp(&(b.0.as_str(), b.1)));
    let specs: Vec<ShapeScanBucketSpec> = combos
        .iter()
        .map(|(bucket, scan_number)| {
            let key = &(bucket.clone(), *scan_number);
            let (height, width) = combo_dims[key];
            ShapeScanBucketSpec {
                shape_bucket: bucket.clone(),
                scan_number: *scan_number,
                height,
                width,
                frames: combo_counts[key],
            }
        })
        .collect();
    let mut next_idx: HashMap<(String, i32), usize> = HashMap::new();
    let mut assignments: Vec<FrameZarrAssignment> = Vec::with_capacity(rows.len());
    for row in rows {
        let bucket = shape_bucket_key(row.naxis2 as usize, row.naxis1 as usize);
        let scan_number = row.scan_number as i32;
        let key = (bucket.clone(), scan_number);
        let idx = next_idx.entry(key).or_insert(0);
        assignments.push(FrameZarrAssignment {
            shape_bucket: bucket,
            scan_number,
            bucket_frame_index: *idx as i32,
        });
        *idx += 1;
    }
    (assignments, specs)
}

fn read_image_u16(row: &BtIngestRow) -> Result<Array2<u16>> {
    if row.bitpix != 16 {
        return Err(CatalogError::Validation(format!(
            "unsupported BITPIX {} for zarr (expected 16): {}",
            row.bitpix, row.file_path
        )));
    }
    let path = Path::new(&row.file_path);
    let n = (row.naxis1 * row.naxis2) as usize;
    let buf = read_bitpix16_be_bytes(path, row.data_offset as u64, n * 2)
        .map_err(super::flatten_fits_error)?;
    let out: Vec<u16> = buf
        .chunks_exact(2)
        .map(|c| u16::from_be_bytes([c[0], c[1]]))
        .collect();
    Array2::from_shape_vec((row.naxis2 as usize, row.naxis1 as usize), out)
        .map_err(|e| CatalogError::Validation(e.to_string()))
}

/// Returns scan and file counts from discovered FITS paths using filename stems only (no FITS I/O).
pub fn beamtime_ingest_layout(beamtime_dir: &Path) -> Result<BeamtimeIngestLayout> {
    let (discovered, _) = discover_paths_for_catalog_ingest(beamtime_dir)?;
    let paths: Vec<PathBuf> = discovered.into_iter().map(|(p, _)| p).collect();
    let (layout, _) = layout_and_groups_from_paths(&paths);
    Ok(layout)
}

fn beamtime_date_label(beamtime_dir: &Path) -> String {
    beamtime_dir
        .file_name()
        .and_then(|n| n.to_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| "unknown".into())
}

/// Emits `Phase(Headers)` then reads every FITS header in parallel on
/// `ctx.pool` (flat per-file iteration), returning rows sorted by
/// `(scan_number, frame_number, file_path)`.
fn run_headers_phase(
    ctx: &IngestContext<'_>,
    paths: &[PathBuf],
    header_items: &[String],
) -> Result<Vec<BtIngestRow>> {
    if let Some(sink) = ctx.progress {
        sink.emit(IngestProgress::Phase {
            phase: IngestPhase::Headers,
        });
    }
    let mut rows: Vec<BtIngestRow> = ctx
        .pool
        .install(|| {
            paths
                .par_iter()
                .map(|p| read_fits_headers_only_row(p.clone(), header_items))
                .collect::<std::result::Result<Vec<_>, _>>()
        })
        .map_err(CatalogError::FitsReadFailed)?;
    rows.sort_by(|a, b| {
        (a.scan_number, a.frame_number, a.file_path.as_str()).cmp(&(
            b.scan_number,
            b.frame_number,
            b.file_path.as_str(),
        ))
    });
    Ok(rows)
}

/// Picks the first-seen sample name for each scan number (empty names
/// normalized to `"_"`), mirroring the sample chosen as each scan's representative.
fn build_scan_first_sample(rows: &[BtIngestRow]) -> HashMap<i32, String> {
    let mut scan_first_sample: HashMap<i32, String> = HashMap::new();
    for r in rows {
        let sn = r.scan_number as i32;
        let sk = if r.sample_name.trim().is_empty() {
            "_".to_string()
        } else {
            r.sample_name.clone()
        };
        scan_first_sample.entry(sn).or_insert(sk);
    }
    scan_first_sample
}

/// Inserts one row per unique sample name under one transaction and returns
/// the populated `name -> sample_id` cache.
fn insert_samples(
    conn: &mut diesel::SqliteConnection,
    beamtime_id: i32,
    rows: &[BtIngestRow],
    cancel: Option<&AtomicBool>,
) -> Result<HashMap<String, i32>> {
    let unique_samples: HashSet<String> = rows
        .iter()
        .map(|r| {
            let n = r.sample_name.trim();
            if n.is_empty() {
                "_".to_string()
            } else {
                n.to_string()
            }
        })
        .collect();

    let mut sample_cache: HashMap<String, i32> = HashMap::new();
    conn.transaction::<(), diesel::result::Error, _>(|conn| {
        for name in &unique_samples {
            if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
                return Err(diesel::result::Error::RollbackTransaction);
            }
            let sid: i32 = diesel::insert_into(samples::table)
                .values((
                    samples::beamtime_id.eq(beamtime_id),
                    samples::name.eq(name.as_str()),
                    samples::representative_x.eq(0.0_f64),
                    samples::representative_y.eq(0.0_f64),
                    samples::representative_z.eq(0.0_f64),
                ))
                .returning(samples::id)
                .get_result(conn)?;
            sample_cache.insert(name.clone(), sid);
        }
        Ok(())
    })
    .map_err(CatalogError::Diesel)?;
    Ok(sample_cache)
}

/// Runs ONE catalog batch transaction: inserts scans new to this batch, then
/// per-row inserts files, tags, file_tags, and frames. Emits `CatalogRow`
/// progress events per row.
#[allow(clippy::too_many_arguments)]
fn insert_catalog_batch(
    conn: &mut diesel::SqliteConnection,
    ctx: &IngestContext<'_>,
    rows: &[BtIngestRow],
    zarr_assignments: &[FrameZarrAssignment],
    start: usize,
    end: usize,
    sample_cache: &HashMap<String, i32>,
    scan_cache: &mut HashMap<i32, i32>,
    scan_first_sample: &HashMap<i32, String>,
    catalog_scan_done: &mut HashMap<i32, u32>,
    global_total: u32,
) -> Result<()> {
    conn.transaction::<(), diesel::result::Error, _>(|conn| {
        let mut batch_scans_seen: HashSet<i32> = HashSet::new();
        for row in &rows[start..end] {
            let sn = row.scan_number as i32;
            if !batch_scans_seen.insert(sn) {
                continue;
            }
            if scan_cache.contains_key(&sn) {
                continue;
            }
            let sk = scan_first_sample
                .get(&sn)
                .cloned()
                .unwrap_or_else(|| "_".to_string());
            let rep_sample = *sample_cache
                .get(&sk)
                .ok_or(diesel::result::Error::NotFound)?;
            let scid: i32 = diesel::insert_into(scans::table)
                .values((
                    scans::beamtime_id.eq(ctx.beamtime_id),
                    scans::sample_id.eq(rep_sample),
                    scans::scan_number.eq(sn),
                    scans::scan_type.eq("fixed_energy"),
                    scans::started_at.eq(None::<String>),
                    scans::ended_at.eq(None::<String>),
                ))
                .returning(scans::id)
                .get_result(conn)?;
            scan_cache.insert(sn, scid);
        }

        for (local_idx, row) in rows[start..end].iter().enumerate() {
            if ctx.is_cancelled() {
                return Err(diesel::result::Error::RollbackTransaction);
            }
            let row_index = start + local_idx;
            let zarr_assignment = &zarr_assignments[row_index];
            let sample_key = if row.sample_name.trim().is_empty() {
                "_".to_string()
            } else {
                row.sample_name.clone()
            };
            let sample_id = *sample_cache
                .get(&sample_key)
                .ok_or(diesel::result::Error::NotFound)?;

            let scan_no = row.scan_number as i32;
            let scan_id = *scan_cache
                .get(&scan_no)
                .ok_or(diesel::result::Error::NotFound)?;

            let parse_flag = if row.scan_number == 0 || row.frame_number == 0 {
                Some("parse_failure".to_string())
            } else {
                None
            };

            let file_id: i32 = diesel::insert_into(files::table)
                .values((
                    files::beamtime_id.eq(ctx.beamtime_id),
                    files::sample_id.eq(sample_id),
                    files::scan_number.eq(scan_no),
                    files::frame_number.eq(row.frame_number as i32),
                    files::nas_uri.eq(row.file_path.as_str()),
                    files::filename.eq(Path::new(&row.file_path)
                        .file_name()
                        .and_then(|s| s.to_str())
                        .unwrap_or("")),
                    files::parse_flag.eq(parse_flag.as_deref()),
                    files::data_offset.eq(row.data_offset),
                    files::naxis1.eq(row.naxis1 as i32),
                    files::naxis2.eq(row.naxis2 as i32),
                    files::bitpix.eq(row.bitpix as i32),
                    files::bzero.eq(row.bzero),
                ))
                .returning(files::id)
                .get_result(conn)?;

            if let Some(tag_slug) = row.tag.as_ref().filter(|t| !t.is_empty()) {
                let tid: i32 = match tags::table
                    .filter(tags::slug.eq(tag_slug.as_str()))
                    .select(tags::id)
                    .first(conn)
                    .optional()?
                {
                    Some(id) => id,
                    None => diesel::insert_into(tags::table)
                        .values(tags::slug.eq(tag_slug.as_str()))
                        .returning(tags::id)
                        .get_result(conn)?,
                };
                let ft_exists: Option<i32> = file_tags::table
                    .filter(file_tags::file_id.eq(file_id))
                    .filter(file_tags::tag_id.eq(tid))
                    .select(file_tags::id)
                    .first(conn)
                    .optional()?;
                if ft_exists.is_none() {
                    diesel::insert_into(file_tags::table)
                        .values((file_tags::file_id.eq(file_id), file_tags::tag_id.eq(tid)))
                        .execute(conn)?;
                }
            }

            let sx = row.sample_x.unwrap_or(0.0);
            let sy = row.sample_y.unwrap_or(0.0);
            let sz = row.sample_z.unwrap_or(0.0);
            let st = row.sample_theta.unwrap_or(0.0);
            let ccd = row.ccd_theta.unwrap_or(0.0);
            let epu = row.epu_polarization.unwrap_or(0.0);
            let exp = row.exposure.unwrap_or(0.0);
            let be = row.beamline_energy.unwrap_or(0.0);
            let ring = row.ring_current.unwrap_or(0.0);
            let ai3 = row.ai3_izero.unwrap_or(0.0);
            let bcm = row.beam_current.unwrap_or(0.0);

            diesel::insert_into(frames::table)
                .values((
                    frames::scan_id.eq(scan_id),
                    frames::file_id.eq(file_id),
                    frames::frame_number.eq(row.frame_number as i32),
                    frames::zarr_group_key.eq(scan_no),
                    frames::zarr_frame_index.eq(row.frame_number as i32),
                    frames::zarr_shape_bucket.eq(Some(zarr_assignment.shape_bucket.as_str())),
                    frames::zarr_bucket_frame_index.eq(Some(zarr_assignment.bucket_frame_index)),
                    frames::acquired_at.eq(row.date_iso.clone()),
                    frames::sample_x.eq(sx),
                    frames::sample_y.eq(sy),
                    frames::sample_z.eq(sz),
                    frames::sample_theta.eq(st),
                    frames::ccd_theta.eq(ccd),
                    frames::beamline_energy.eq(be),
                    frames::epu_polarization.eq(epu),
                    frames::exposure.eq(exp),
                    frames::ring_current.eq(ring),
                    frames::ai3_izero.eq(ai3),
                    frames::beam_current.eq(bcm),
                    frames::quality_flag.eq(None::<String>),
                ))
                .execute(conn)?;

            if let Some(sink) = ctx.progress {
                let sn = row.scan_number as i32;
                let e = catalog_scan_done.entry(sn).or_insert(0);
                *e += 1;
                let sd = *e;
                let st = ctx.scan_total_map.get(&sn).copied().unwrap_or(0);
                let global_idx = (start + local_idx + 1) as u32;
                sink.emit(IngestProgress::CatalogRow {
                    scan_number: sn,
                    scan_done: sd,
                    scan_total: st,
                    global_done: global_idx,
                    global_total,
                });
            }
        }
        Ok(())
    })
    .map_err(CatalogError::Diesel)
}

/// Emits `Phase(Catalog)` then `CatalogRow` per row. Runs batched SQLite
/// transactions via `plan_catalog_batches`, owning the sample and scan
/// caches locally across the batches.
fn run_catalog_phase(
    ctx: &IngestContext<'_>,
    conn: &mut diesel::SqliteConnection,
    rows: &[BtIngestRow],
    zarr_assignments: &[FrameZarrAssignment],
) -> Result<()> {
    if let Some(sink) = ctx.progress {
        sink.emit(IngestProgress::Phase {
            phase: IngestPhase::Catalog,
        });
    }

    let sample_cache = insert_samples(conn, ctx.beamtime_id, rows, ctx.cancel)?;
    let scan_first_sample = build_scan_first_sample(rows);

    let batches = plan_catalog_batches(rows);
    let global_total = rows.len() as u32;
    let mut scan_cache: HashMap<i32, i32> = HashMap::new();
    let mut catalog_scan_done: HashMap<i32, u32> = HashMap::new();

    for (start, end) in batches {
        insert_catalog_batch(
            conn,
            ctx,
            rows,
            zarr_assignments,
            start,
            end,
            &sample_cache,
            &mut scan_cache,
            &scan_first_sample,
            &mut catalog_scan_done,
            global_total,
        )?;
    }

    Ok(())
}

/// Emits `Phase(Zarr)` then `FileComplete` per row. Uses a bounded crossbeam
/// channel between a single reader pool task (running `ctx.pool` in parallel)
/// and the calling-thread writer so the zstore sees writes in completion order.
fn run_zarr_phase(
    ctx: &IngestContext<'_>,
    rows: &[BtIngestRow],
    zarr_assignments: &[FrameZarrAssignment],
    bucket_specs: &[ShapeScanBucketSpec],
    zstore: &zarrs::storage::ReadableWritableListableStorage,
) -> Result<()> {
    if let Some(sink) = ctx.progress {
        sink.emit(IngestProgress::Phase {
            phase: IngestPhase::Zarr,
        });
    }

    let n_workers = ctx.pool.current_num_threads();
    let channel_cap = (n_workers.saturating_mul(2)).max(4);
    let (tx, rx) = crossbeam_channel::bounded::<Result<(usize, Array2<u16>)>>(channel_cap);
    prepare_shape_scan_bucket_arrays(zstore, bucket_specs)
        .map_err(|e| CatalogError::Validation(e.to_string()))?;

    let stop = AtomicBool::new(false);
    let mut scan_done: HashMap<i32, u32> = HashMap::new();
    let mut global_done: u32 = 0;
    let global_total = rows.len() as u32;

    std::thread::scope(|s| -> Result<()> {
        let rows_ref: &[BtIngestRow] = rows;
        let pool_ref: &rayon::ThreadPool = ctx.pool;
        let cancel_ref = ctx.cancel;
        let stop_ref = &stop;

        let reader_handle = s.spawn(move || {
            pool_ref.install(move || {
                rows_ref
                    .par_iter()
                    .enumerate()
                    .for_each_with(tx, |tx_c, (row_i, row)| {
                        if stop_ref.load(Ordering::Relaxed) {
                            return;
                        }
                        if cancel_ref.is_some_and(|c| c.load(Ordering::Relaxed)) {
                            return;
                        }
                        let item = read_image_u16(row).map(|img| (row_i, img));
                        if tx_c.send(item).is_err() {
                            stop_ref.store(true, Ordering::Relaxed);
                        }
                    });
            });
        });

        let mut write_err: Option<CatalogError> = None;
        for item in rx.iter() {
            if write_err.is_some() {
                continue;
            }
            let step: Result<()> = (|| {
                let (row_i, img) = item?;
                let row = &rows[row_i];
                let zarr_assignment = &zarr_assignments[row_i];
                write_scan_shape_bucket_frame_raw(
                    zstore,
                    &zarr_assignment.shape_bucket,
                    zarr_assignment.scan_number,
                    zarr_assignment.bucket_frame_index as usize,
                    &img,
                )
                .map_err(|e| CatalogError::Validation(e.to_string()))?;
                drop(img);
                let sn = row.scan_number as i32;
                let entry = scan_done.entry(sn).or_insert(0);
                *entry += 1;
                let sd = *entry;
                let st = ctx.scan_total_map.get(&sn).copied().unwrap_or(0);
                global_done += 1;
                if let Some(sink) = ctx.progress {
                    sink.emit(IngestProgress::FileComplete {
                        scan_number: sn,
                        scan_done: sd,
                        scan_total: st,
                        global_done,
                        global_total,
                    });
                }
                Ok(())
            })();
            if let Err(e) = step {
                stop.store(true, Ordering::Relaxed);
                write_err = Some(e);
            }
        }

        reader_handle
            .join()
            .map_err(|_| CatalogError::Validation("ingest zarr reader thread panicked".into()))?;

        match write_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    })
}

/// Ingests a beamtime directory into the global catalog and local zarr store.
pub fn ingest_beamtime(
    beamtime_dir: &Path,
    header_items: &[String],
    incremental: bool,
    progress_tx: Option<std::sync::mpsc::Sender<(u32, u32)>>,
    selection: IngestSelection,
) -> Result<PathBuf> {
    let progress = progress_tx.map(IngestProgressSink::from_channel);
    ingest_beamtime_inner(
        beamtime_dir,
        header_items,
        incremental,
        progress,
        IngestParallelism::default(),
        selection,
        None,
    )
}

/// Ingest with explicit parallelism (worker threads or resource fraction).
pub fn ingest_beamtime_parallel(
    beamtime_dir: &Path,
    header_items: &[String],
    incremental: bool,
    progress_tx: Option<std::sync::mpsc::Sender<(u32, u32)>>,
    parallelism: IngestParallelism,
    selection: IngestSelection,
) -> Result<PathBuf> {
    let progress = progress_tx.map(IngestProgressSink::from_channel);
    ingest_beamtime_inner(
        beamtime_dir,
        header_items,
        incremental,
        progress,
        parallelism,
        selection,
        None,
    )
}

/// Ingest with structured progress (layout, phases, per-file completion after zarr) and optional
/// legacy channel behavior via [`IngestProgressSink::from_channel`].
pub fn ingest_beamtime_with_progress_sink(
    beamtime_dir: &Path,
    header_items: &[String],
    incremental: bool,
    progress: Option<IngestProgressSink>,
    parallelism: IngestParallelism,
    selection: IngestSelection,
) -> Result<PathBuf> {
    ingest_beamtime_inner(
        beamtime_dir,
        header_items,
        incremental,
        progress,
        parallelism,
        selection,
        None,
    )
}

/// Ingest with optional data-root / experimentalist context (accepted for API compatibility).
#[allow(clippy::too_many_arguments)]
pub fn ingest_beamtime_with_context(
    beamtime_dir: &Path,
    _data_root: Option<&Path>,
    _experimentalist: Option<&str>,
    header_items: &[String],
    incremental: bool,
    progress_tx: Option<std::sync::mpsc::Sender<(u32, u32)>>,
    parallelism: IngestParallelism,
    selection: IngestSelection,
    cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
) -> Result<PathBuf> {
    if cancel
        .as_ref()
        .map(|c| c.load(std::sync::atomic::Ordering::Relaxed))
        .unwrap_or(false)
    {
        return Err(CatalogError::Validation("ingest cancelled".into()));
    }
    let progress = progress_tx.map(IngestProgressSink::from_channel);
    ingest_beamtime_inner(
        beamtime_dir,
        header_items,
        incremental,
        progress,
        parallelism,
        selection,
        cancel,
    )
}

#[cfg(feature = "parallel_ingest")]
#[allow(clippy::too_many_arguments)]
pub fn ingest_beamtime_pipelined_with_context(
    beamtime_dir: &Path,
    data_root: Option<&Path>,
    experimentalist: Option<&str>,
    header_items: &[String],
    incremental: bool,
    progress_tx: Option<std::sync::mpsc::Sender<(u32, u32)>>,
    parallelism: IngestParallelism,
    selection: IngestSelection,
    cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
) -> Result<PathBuf> {
    ingest_beamtime_with_context(
        beamtime_dir,
        data_root,
        experimentalist,
        header_items,
        incremental,
        progress_tx,
        parallelism,
        selection,
        cancel,
    )
}

#[cfg(feature = "parallel_ingest")]
pub fn ingest_beamtime_pipelined(
    beamtime_dir: &Path,
    header_items: &[String],
    incremental: bool,
    progress_tx: Option<std::sync::mpsc::Sender<(u32, u32)>>,
) -> Result<PathBuf> {
    ingest_beamtime(
        beamtime_dir,
        header_items,
        incremental,
        progress_tx,
        IngestSelection::default(),
    )
}

fn ingest_beamtime_inner(
    beamtime_dir: &Path,
    header_items: &[String],
    incremental: bool,
    progress: Option<IngestProgressSink>,
    parallelism: IngestParallelism,
    selection: IngestSelection,
    cancel: Option<std::sync::Arc<AtomicBool>>,
) -> Result<PathBuf> {
    let parallelism = IngestParallelism::from_options_or_env(
        parallelism.worker_threads,
        parallelism.resource_fraction,
    );
    if !beamtime_dir.is_dir() {
        return Err(CatalogError::Validation(format!(
            "beamtime_dir is not a directory: {}",
            beamtime_dir.display()
        )));
    }

    let db_path = paths::default_catalog_db_path()?;
    let nas_uri = paths::file_uri_for_path(beamtime_dir)?;
    let zarr_path = paths::beamtime_zarr_path(beamtime_dir)?;
    let date_label = beamtime_date_label(beamtime_dir);
    let (discovered, _layout) = discover_paths_for_catalog_ingest(beamtime_dir)?;

    let mut conn = db::establish_connection(&db_path)?;
    let now_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i32)
        .unwrap_or(0);

    conn.transaction::<(), diesel::result::Error, _>(|conn| {
        diesel::delete(beamtimes::table.filter(beamtimes::nas_uri.eq(&nas_uri))).execute(conn)?;
        diesel::insert_into(beamtimes::table)
            .values((
                beamtimes::nas_uri.eq(&nas_uri),
                beamtimes::zarr_path.eq(zarr_path.to_string_lossy().as_ref()),
                beamtimes::date.eq(&date_label),
                beamtimes::last_indexed_at.eq(Some(now_secs)),
            ))
            .execute(conn)?;
        Ok(())
    })
    .map_err(CatalogError::Diesel)?;

    let beamtime_id: i32 = beamtimes::table
        .filter(beamtimes::nas_uri.eq(&nas_uri))
        .select(beamtimes::id)
        .first(&mut conn)
        .map_err(CatalogError::Diesel)?;

    if discovered.is_empty() {
        if let Some(ref sink) = progress {
            sink.emit(IngestProgress::Layout {
                total_files: 0,
                scans: vec![],
            });
        }
        return Ok(db_path);
    }

    let paths_only: Vec<PathBuf> = discovered.iter().map(|(p, _)| p.clone()).collect();
    let (paths_only, layout_summary) = filter_paths_by_selection(paths_only, &selection)?;
    if let Some(ref sink) = progress {
        sink.emit(IngestProgress::Layout {
            total_files: layout_summary.total_files as u32,
            scans: layout_summary
                .scans
                .iter()
                .map(|s| (s.scan_number, s.file_count as u32))
                .collect(),
        });
    }

    let n_workers = parallelism.resolve_worker_count()?;
    let pool = ThreadPoolBuilder::new()
        .num_threads(n_workers)
        .build()
        .map_err(|e| CatalogError::Validation(format!("rayon thread pool: {e}")))?;

    let scan_total_map: HashMap<i32, u32> = layout_summary
        .scans
        .iter()
        .map(|s| (s.scan_number, s.file_count as u32))
        .collect();

    let cancel_borrow: Option<&AtomicBool> = cancel.as_ref().map(|c| c.as_ref());

    let ctx = IngestContext {
        beamtime_id,
        progress: progress.as_ref(),
        cancel: cancel_borrow,
        pool: &pool,
        scan_total_map,
    };

    let rows = run_headers_phase(&ctx, &paths_only, header_items)?;
    let (zarr_assignments, bucket_specs) = build_frame_zarr_plan(&rows);

    let zstore =
        open_zarr_store(&zarr_path).map_err(|e| CatalogError::Validation(e.to_string()))?;

    run_catalog_phase(&ctx, &mut conn, &rows, &zarr_assignments)?;
    run_zarr_phase(&ctx, &rows, &zarr_assignments, &bucket_specs, &zstore)?;

    let _ = incremental;
    Ok(db_path)
}

#[cfg(all(test, feature = "catalog"))]
mod tests {
    use super::*;
    use crate::loader::read_fits_headers_only_row;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn fixture_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("minimal.fits")
    }

    #[test]
    fn read_image_u16_bulk_read_decodes_minimal_fits() {
        let path = fixture_path();
        if !path.exists() {
            panic!("required fixture missing: {}", path.display());
        }
        let header_items: Vec<String> = Vec::new();
        let row = read_fits_headers_only_row(path, &header_items)
            .expect("minimal.fits fixture should parse into BtIngestRow");
        let img = read_image_u16(&row).expect("read_image_u16 failed on minimal.fits");
        let rows = row.naxis2 as usize;
        let cols = row.naxis1 as usize;
        assert_eq!(img.shape(), [rows, cols]);
        assert_eq!(img.len(), rows * cols);
        assert_eq!(rows, 2, "fixture minimal.fits is a 2x2 image");
        assert_eq!(cols, 2, "fixture minimal.fits is a 2x2 image");
        assert_eq!(
            img[[0, 0]],
            0_u16,
            "first pixel (raw big-endian i16 -> i32)"
        );
        assert_eq!(img[[0, 1]], 1_u16, "row-major second pixel");
        assert_eq!(img[[1, 0]], 2_u16, "second row first pixel");
        assert_eq!(img[[1, 1]], 3_u16, "last pixel");
    }

    #[test]
    fn read_image_u16_rejects_non_bitpix_16() {
        let path = fixture_path();
        if !path.exists() {
            panic!("required fixture missing: {}", path.display());
        }
        let header_items: Vec<String> = Vec::new();
        let mut row = read_fits_headers_only_row(path, &header_items)
            .expect("minimal.fits fixture should parse into BtIngestRow");
        row.bitpix = 8;
        match read_image_u16(&row) {
            Err(CatalogError::Validation(msg)) => {
                assert!(
                    msg.contains("unsupported BITPIX"),
                    "unexpected message: {msg}"
                );
            }
            other => panic!("expected CatalogError::Validation, got {other:?}"),
        }
    }

    struct EnvGuard {
        key: &'static str,
        prev: Option<std::ffi::OsString>,
    }

    impl EnvGuard {
        fn set(key: &'static str, value: &std::path::Path) -> Self {
            let prev = std::env::var_os(key);
            std::env::set_var(key, value);
            Self { key, prev }
        }

        fn unset(key: &'static str) -> Self {
            let prev = std::env::var_os(key);
            std::env::remove_var(key);
            Self { key, prev }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match self.prev.take() {
                Some(v) => std::env::set_var(self.key, v),
                None => std::env::remove_var(self.key),
            }
        }
    }

    fn count_rows(db_path: &std::path::Path) -> Result<(i64, i64, i64)> {
        let mut conn = db::establish_connection(db_path)?;
        let s: i64 = samples::table
            .count()
            .get_result(&mut conn)
            .map_err(CatalogError::Diesel)?;
        let sc: i64 = scans::table
            .count()
            .get_result(&mut conn)
            .map_err(CatalogError::Diesel)?;
        let f: i64 = files::table
            .count()
            .get_result(&mut conn)
            .map_err(CatalogError::Diesel)?;
        Ok((s, sc, f))
    }

    fn make_row(scan_number: i64, frame_number: i64) -> BtIngestRow {
        BtIngestRow {
            file_path: format!("/tmp/{scan_number}_{frame_number}.fits"),
            data_offset: 0,
            naxis1: 1,
            naxis2: 1,
            bitpix: 16,
            bzero: 0,
            file_name: format!("{scan_number}_{frame_number}.fits"),
            sample_name: "s".into(),
            tag: None,
            scan_number,
            frame_number,
            beamline_energy: None,
            sample_theta: None,
            ccd_theta: None,
            epu_polarization: None,
            exposure: None,
            sample_x: None,
            sample_y: None,
            sample_z: None,
            ring_current: None,
            ai3_izero: None,
            beam_current: None,
            date_iso: None,
        }
    }

    fn rows_for_scans(sizes: &[(i64, usize)]) -> Vec<BtIngestRow> {
        let mut out = Vec::new();
        for &(sn, count) in sizes {
            for f in 0..count {
                out.push(make_row(sn, f as i64));
            }
        }
        out
    }

    #[test]
    fn plan_catalog_batches_groups_small_scans() {
        let rows = rows_for_scans(&[(1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (6, 3000)]);
        let batches = plan_catalog_batches(&rows);
        assert_eq!(
            batches,
            vec![(0, 25), (25, 3025)],
            "small scans must coalesce then seal when the next large scan would overflow MAX"
        );
        const _: () = assert!(MAX_BATCH_ROWS < 3000);
    }

    #[test]
    fn plan_catalog_batches_isolates_oversized_scan() {
        let rows = rows_for_scans(&[(1, 3000)]);
        let batches = plan_catalog_batches(&rows);
        assert_eq!(
            batches,
            vec![(0, 3000)],
            "single scan > MAX gets its own overflow batch"
        );
    }

    #[test]
    fn plan_catalog_batches_empty_rows_empty_plan() {
        let rows: Vec<BtIngestRow> = Vec::new();
        assert!(plan_catalog_batches(&rows).is_empty());
    }

    #[test]
    fn ingest_produces_expected_row_counts_for_minimal_fixture() {
        let fixture = fixture_path();
        if !fixture.exists() {
            panic!("required fixture missing: {}", fixture.display());
        }
        let tmp = tempfile::tempdir().expect("create tempdir for PYREF_HOME");
        let beamtime_dir = tmp.path().join("2024-02-02");
        let ccd_dir = beamtime_dir.join("CCD");
        std::fs::create_dir_all(&ccd_dir).expect("create CCD dir");
        std::fs::copy(&fixture, ccd_dir.join("minimal.fits")).expect("copy fixture");

        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let _home = EnvGuard::set("PYREF_HOME", tmp.path());
        let _db = EnvGuard::unset("PYREF_CATALOG_DB");
        let _cache = EnvGuard::unset("PYREF_CACHE_ROOT");

        let header_items: Vec<String> = DEFAULT_INGEST_HEADER_ITEMS
            .iter()
            .map(|s| (*s).to_string())
            .collect();

        let db_path = ingest_beamtime(
            &beamtime_dir,
            &header_items,
            false,
            None,
            IngestSelection::default(),
        )
        .expect("batched ingest should succeed");
        let counts = count_rows(&db_path).expect("count rows after ingest");
        let mut conn = db::establish_connection(&db_path).expect("open catalog db");
        let frame_count: i64 = frames::table
            .count()
            .get_result(&mut conn)
            .expect("count frames");

        assert_eq!(
            counts,
            (1, 1, 1),
            "batched ingest must produce exactly 1 sample/scan/file, got {counts:?}"
        );
        assert_eq!(
            frame_count, 1,
            "batched ingest must produce exactly 1 frame"
        );
    }

    #[test]
    fn ingest_is_idempotent_across_reingests() {
        let fixture = fixture_path();
        if !fixture.exists() {
            panic!("required fixture missing: {}", fixture.display());
        }
        let tmp = tempfile::tempdir().expect("create tempdir for PYREF_HOME");
        let beamtime_dir = tmp.path().join("2024-01-01");
        let ccd_dir = beamtime_dir.join("CCD");
        std::fs::create_dir_all(&ccd_dir).expect("create CCD dir");
        std::fs::copy(&fixture, ccd_dir.join("minimal.fits")).expect("copy fixture");

        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let _home = EnvGuard::set("PYREF_HOME", tmp.path());
        let _db = EnvGuard::unset("PYREF_CATALOG_DB");
        let _cache = EnvGuard::unset("PYREF_CACHE_ROOT");

        let header_items: Vec<String> = DEFAULT_INGEST_HEADER_ITEMS
            .iter()
            .map(|s| (*s).to_string())
            .collect();

        let db_path1 = ingest_beamtime(
            &beamtime_dir,
            &header_items,
            false,
            None,
            IngestSelection::default(),
        )
        .expect("first ingest run should succeed");
        let counts_1 = count_rows(&db_path1).expect("count rows after run 1");

        let db_path2 = ingest_beamtime(
            &beamtime_dir,
            &header_items,
            false,
            None,
            IngestSelection::default(),
        )
        .expect("second ingest run should succeed");
        let counts_2 = count_rows(&db_path2).expect("count rows after run 2");

        assert_eq!(db_path1, db_path2, "catalog path should be deterministic");
        assert_eq!(
            counts_1,
            (1, 1, 1),
            "first run should produce exactly 1 sample/scan/file, got {counts_1:?}"
        );
        assert_eq!(
            counts_1, counts_2,
            "reingest must be exactly idempotent (run1={counts_1:?}, run2={counts_2:?})"
        );
    }

    #[test]
    fn zarr_phase_streams_and_writes_raw_for_each_frame() {
        let fixture = fixture_path();
        if !fixture.exists() {
            panic!("required fixture missing: {}", fixture.display());
        }
        let tmp = tempfile::tempdir().expect("create tempdir for PYREF_HOME");
        let beamtime_dir = tmp.path().join("2024-03-03");
        let ccd_dir = beamtime_dir.join("CCD");
        std::fs::create_dir_all(&ccd_dir).expect("create CCD dir");
        std::fs::copy(&fixture, ccd_dir.join("minimal.fits")).expect("copy fixture");

        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let _home = EnvGuard::set("PYREF_HOME", tmp.path());
        let _db = EnvGuard::unset("PYREF_CATALOG_DB");
        let _cache = EnvGuard::unset("PYREF_CACHE_ROOT");

        let header_items: Vec<String> = DEFAULT_INGEST_HEADER_ITEMS
            .iter()
            .map(|s| (*s).to_string())
            .collect();

        let db_path = ingest_beamtime(
            &beamtime_dir,
            &header_items,
            false,
            None,
            IngestSelection::default(),
        )
        .expect("streaming zarr ingest should succeed");

        let zarr_root =
            paths::beamtime_zarr_path(&beamtime_dir).expect("resolve beamtime zarr path");
        assert!(
            zarr_root.is_dir(),
            "expected zarr root dir at {}",
            zarr_root.display()
        );

        let mut conn = db::establish_connection(&db_path).expect("open catalog db");
        let shape_bucket: Option<String> = frames::table
            .select(frames::zarr_shape_bucket)
            .first(&mut conn)
            .expect("select shape bucket for only cataloged frame");
        let shape_bucket = shape_bucket.expect("zarr shape bucket should be set");
        let scan_no: i32 = files::table
            .select(files::scan_number)
            .first(&mut conn)
            .expect("select scan of only cataloged file");
        let raw_path = zarr_root
            .join("images")
            .join("by_shape")
            .join(&shape_bucket)
            .join("scans")
            .join(scan_no.to_string())
            .join("raw");
        assert!(
            raw_path.is_dir(),
            "streaming zarr writer must create raw array dir at {}",
            raw_path.display()
        );

        let payload_found = walkdir::WalkDir::new(&raw_path).into_iter().any(|entry| {
            entry
                .as_ref()
                .map(|e| e.file_type().is_file())
                .unwrap_or(false)
        });
        assert!(
            payload_found,
            "streaming zarr writer must emit at least one file under {}",
            raw_path.display()
        );
    }
}

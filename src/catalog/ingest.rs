//! Beamtime ingest: Diesel catalog rows and zarr arrays in one pass.
//!
//! SQLite allows one writer at a time. Ingest uses a single [`diesel::SqliteConnection`] for all
//! catalog mutations in this process. After `fork` or when spawning a subprocess, open a new
//! connection in the child; do not share a connection across process boundaries.
//!
//! FITS header reads and pixel reads for zarr use a [`rayon::ThreadPool`] sized by
//! [`crate::catalog::IngestParallelism`] (after [`IngestParallelism::from_options_or_env`]). Work is
//! parallelized **per scan** (each worker owns one scan's files sequentially); nested
//! [`rayon::prelude::ParallelIterator`] runs on that pool via [`rayon::ThreadPool::install`]. Diesel
//! transactions and [`super::zarr_write::write_frame_raw`] run on the calling thread in global row
//! order so catalog rows and zarr datasets stay aligned.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use diesel::prelude::*;
use diesel::OptionalExtension;
use ndarray::Array2;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

use crate::io::BtIngestRow;
use crate::loader::read_multiple_fits_headers_only_rows;
use crate::schema::{beamtimes, file_tags, files, frames, samples, scans, tags};

use super::discover_paths_for_catalog_ingest;
use super::ingest_progress::{
    layout_and_groups_from_paths, BeamtimeIngestLayout, IngestProgress, IngestProgressSink,
};
use super::layout::BeamtimeLayout;
use super::parallelism::IngestParallelism;
use super::zarr_write::{open_zarr_store, write_frame_raw};
use super::{db, paths, CatalogError, Result};

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

fn layout_label(layout: BeamtimeLayout) -> &'static str {
    match layout {
        BeamtimeLayout::Nested => "nested",
        BeamtimeLayout::Flat => "flat",
    }
}

fn read_image_i32(row: &BtIngestRow) -> Result<Array2<i32>> {
    if row.bitpix != 16 {
        return Err(CatalogError::Validation(format!(
            "unsupported BITPIX {} for zarr (expected 16): {}",
            row.bitpix, row.file_path
        )));
    }
    let path = Path::new(&row.file_path);
    let mut f = File::open(path).map_err(CatalogError::Io)?;
    f.seek(SeekFrom::Start(row.data_offset as u64))
        .map_err(CatalogError::Io)?;
    let n = (row.naxis1 * row.naxis2) as usize;
    let mut out = Vec::with_capacity(n);
    let mut b = [0u8; 2];
    for _ in 0..n {
        f.read_exact(&mut b).map_err(CatalogError::Io)?;
        out.push(i16::from_be_bytes(b) as i32);
    }
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

/// Ingests a beamtime directory into the global catalog and local zarr store.
pub fn ingest_beamtime(
    beamtime_dir: &Path,
    header_items: &[String],
    incremental: bool,
    progress_tx: Option<std::sync::mpsc::Sender<(u32, u32)>>,
) -> Result<PathBuf> {
    let progress = progress_tx.map(IngestProgressSink::from_channel);
    ingest_beamtime_inner(
        beamtime_dir,
        header_items,
        incremental,
        progress,
        IngestParallelism::default(),
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
) -> Result<PathBuf> {
    let progress = progress_tx.map(IngestProgressSink::from_channel);
    ingest_beamtime_inner(
        beamtime_dir,
        header_items,
        incremental,
        progress,
        parallelism,
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
) -> Result<PathBuf> {
    ingest_beamtime_inner(
        beamtime_dir,
        header_items,
        incremental,
        progress,
        parallelism,
        None,
    )
}

/// Ingest with optional data-root / experimentalist context (accepted for API compatibility).
pub fn ingest_beamtime_with_context(
    beamtime_dir: &Path,
    _data_root: Option<&Path>,
    _experimentalist: Option<&str>,
    header_items: &[String],
    incremental: bool,
    progress_tx: Option<std::sync::mpsc::Sender<(u32, u32)>>,
    parallelism: IngestParallelism,
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
        cancel,
    )
}

#[cfg(feature = "parallel_ingest")]
pub fn ingest_beamtime_pipelined_with_context(
    beamtime_dir: &Path,
    data_root: Option<&Path>,
    experimentalist: Option<&str>,
    header_items: &[String],
    incremental: bool,
    progress_tx: Option<std::sync::mpsc::Sender<(u32, u32)>>,
    parallelism: IngestParallelism,
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
    ingest_beamtime(beamtime_dir, header_items, incremental, progress_tx)
}

fn ingest_beamtime_inner(
    beamtime_dir: &Path,
    header_items: &[String],
    incremental: bool,
    progress: Option<IngestProgressSink>,
    parallelism: IngestParallelism,
    cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
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
    let (discovered, layout) = discover_paths_for_catalog_ingest(beamtime_dir)?;
    let _ = layout_label(layout);

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
    let (layout_summary, scan_groups) = layout_and_groups_from_paths(&paths_only);
    if let Some(ref sink) = progress {
        sink.emit(IngestProgress::Layout {
            total_files: layout_summary.total_files as u32,
            scans: layout_summary
                .scans
                .iter()
                .map(|s| (s.scan_number, s.file_count as u32))
                .collect(),
        });
        sink.emit(IngestProgress::Phase {
            name: "headers".into(),
        });
    }

    let n_workers = parallelism.resolve_worker_count()?;
    let pool = ThreadPoolBuilder::new()
        .num_threads(n_workers)
        .build()
        .map_err(|e| CatalogError::Validation(format!("rayon thread pool: {e}")))?;
    let header_groups: Vec<Vec<BtIngestRow>> = pool
        .install(|| {
            scan_groups
                .par_iter()
                .map(|(_sn, paths)| {
                    read_multiple_fits_headers_only_rows(paths.clone(), header_items)
                })
                .collect::<std::result::Result<Vec<_>, _>>()
        })
        .map_err(CatalogError::FitsReadFailed)?;
    let mut rows: Vec<BtIngestRow> = header_groups.into_iter().flatten().collect();
    rows.sort_by(|a, b| {
        (a.scan_number, a.frame_number, a.file_path.as_str()).cmp(&(
            b.scan_number,
            b.frame_number,
            b.file_path.as_str(),
        ))
    });

    if let Some(ref sink) = progress {
        sink.emit(IngestProgress::Phase {
            name: "catalog".into(),
        });
    }

    let zstore =
        open_zarr_store(&zarr_path).map_err(|e| CatalogError::Validation(e.to_string()))?;

    let mut sample_cache: HashMap<String, i32> = HashMap::new();
    let mut scan_cache: HashMap<i32, i32> = HashMap::new();

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

    conn.transaction::<(), diesel::result::Error, _>(|conn| {
        for name in &unique_samples {
            diesel::insert_into(samples::table)
                .values((
                    samples::beamtime_id.eq(beamtime_id),
                    samples::name.eq(name.as_str()),
                    samples::representative_x.eq(0.0_f64),
                    samples::representative_y.eq(0.0_f64),
                    samples::representative_z.eq(0.0_f64),
                ))
                .execute(conn)?;
            let sid: i32 = samples::table
                .filter(samples::beamtime_id.eq(beamtime_id))
                .filter(samples::name.eq(name.as_str()))
                .select(samples::id)
                .first(conn)?;
            sample_cache.insert(name.clone(), sid);
        }

        let mut scan_first_sample: HashMap<i32, String> = HashMap::new();
        for r in &rows {
            let sn = r.scan_number as i32;
            let sk = if r.sample_name.trim().is_empty() {
                "_".to_string()
            } else {
                r.sample_name.clone()
            };
            scan_first_sample.entry(sn).or_insert(sk);
        }
        let unique_scans: HashSet<i32> = rows.iter().map(|r| r.scan_number as i32).collect();
        for sn in &unique_scans {
            let sk = scan_first_sample
                .get(sn)
                .cloned()
                .unwrap_or_else(|| "_".to_string());
            let rep_sample = *sample_cache
                .get(&sk)
                .ok_or_else(|| diesel::result::Error::NotFound)?;
            diesel::insert_into(scans::table)
                .values((
                    scans::beamtime_id.eq(beamtime_id),
                    scans::sample_id.eq(rep_sample),
                    scans::scan_number.eq(*sn),
                    scans::scan_type.eq("fixed_energy"),
                    scans::started_at.eq(None::<String>),
                    scans::ended_at.eq(None::<String>),
                ))
                .execute(conn)?;
            let scid: i32 = scans::table
                .filter(scans::beamtime_id.eq(beamtime_id))
                .filter(scans::scan_number.eq(*sn))
                .select(scans::id)
                .first(conn)?;
            scan_cache.insert(*sn, scid);
        }

        for (idx, row) in rows.iter().enumerate() {
            if cancel
                .as_ref()
                .map(|c| c.load(std::sync::atomic::Ordering::Relaxed))
                .unwrap_or(false)
            {
                return Err(diesel::result::Error::RollbackTransaction);
            }
            let sample_key = if row.sample_name.trim().is_empty() {
                "_".to_string()
            } else {
                row.sample_name.clone()
            };
            let sample_id = *sample_cache
                .get(&sample_key)
                .ok_or_else(|| diesel::result::Error::NotFound)?;

            let scan_no = row.scan_number as i32;
            let scan_id = *scan_cache
                .get(&scan_no)
                .ok_or_else(|| diesel::result::Error::NotFound)?;

            let parse_flag = if row.scan_number == 0 || row.frame_number == 0 {
                Some("parse_failure".to_string())
            } else {
                None
            };

            diesel::insert_into(files::table)
                .values((
                    files::beamtime_id.eq(beamtime_id),
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
                .execute(conn)?;

            let file_id: i32 = files::table
                .filter(files::beamtime_id.eq(beamtime_id))
                .filter(files::nas_uri.eq(row.file_path.as_str()))
                .select(files::id)
                .first(conn)?;

            if let Some(tag_slug) = row.tag.as_ref().filter(|t| !t.is_empty()) {
                let tid: i32 = match tags::table
                    .filter(tags::slug.eq(tag_slug.as_str()))
                    .select(tags::id)
                    .first(conn)
                    .optional()?
                {
                    Some(id) => id,
                    None => {
                        diesel::insert_into(tags::table)
                            .values(tags::slug.eq(tag_slug.as_str()))
                            .execute(conn)?;
                        tags::table
                            .filter(tags::slug.eq(tag_slug.as_str()))
                            .select(tags::id)
                            .first(conn)?
                    }
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

            if let Some(ref sink) = progress {
                sink.legacy_catalog_row((idx + 1) as u32, rows.len() as u32);
            }
        }
        Ok(())
    })
    .map_err(CatalogError::Diesel)?;

    if let Some(ref sink) = progress {
        sink.emit(IngestProgress::Phase {
            name: "zarr".into(),
        });
    }

    let scan_total_map: HashMap<i32, u32> = layout_summary
        .scans
        .iter()
        .map(|s| (s.scan_number, s.file_count as u32))
        .collect();
    let mut by_scan: BTreeMap<i32, Vec<usize>> = BTreeMap::new();
    for (i, r) in rows.iter().enumerate() {
        by_scan.entry(r.scan_number as i32).or_default().push(i);
    }
    let scan_order: Vec<i32> = by_scan.keys().copied().collect();

    let read_chunks: Vec<Vec<(usize, Array2<i32>)>> = pool.install(|| {
        scan_order
            .par_iter()
            .map(|&sn| -> Result<Vec<(usize, Array2<i32>)>> {
                let idxs: Vec<usize> = by_scan.get(&sn).cloned().unwrap_or_default();
                idxs.into_iter()
                    .map(|row_i| read_image_i32(&rows[row_i]).map(|img| (row_i, img)))
                    .collect::<std::result::Result<Vec<_>, _>>()
            })
            .collect::<std::result::Result<Vec<_>, _>>()
    })?;
    let mut flat: Vec<(usize, Array2<i32>)> = read_chunks.into_iter().flatten().collect();
    flat.sort_by_key(|(i, _)| *i);

    let mut scan_done: HashMap<i32, u32> = HashMap::new();
    let mut global_done: u32 = 0;
    let global_total = rows.len() as u32;
    for (i, img) in flat {
        let row = &rows[i];
        write_frame_raw(&zstore, row.scan_number, row.frame_number, &img)
            .map_err(|e| CatalogError::Validation(e.to_string()))?;
        let sn = row.scan_number as i32;
        let e = scan_done.entry(sn).or_insert(0);
        *e += 1;
        let sd = *e;
        let st = scan_total_map.get(&sn).copied().unwrap_or(0);
        global_done += 1;
        if let Some(ref sink) = progress {
            sink.emit(IngestProgress::FileComplete {
                scan_number: sn,
                scan_done: sd,
                scan_total: st,
                global_done,
                global_total,
            });
        }
    }

    let _ = incremental;
    Ok(db_path)
}

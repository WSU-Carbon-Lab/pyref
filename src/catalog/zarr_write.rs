//! Write detector arrays into the beamtime zarr store.
//!
//! Layout: legacy ``/{scan}/{frame}/raw`` (2D int32) may still exist for older
//! archives. New ingest writes **per-scan** 3D stacks at
//! ``/images/by_shape/<HxW>/scans/<scan_number>/raw`` as ``uint16`` with
//! shuffle + Zstd so each scan compresses independently.

use ndarray::Array2;
use std::path::Path;
use std::sync::Arc;
use zarrs::array::data_type;
use zarrs::array::Array;
use zarrs::array::ArrayBuilder;
use zarrs::array::ArraySubset;
use zarrs::group::GroupBuilder;
use zarrs::storage::{ReadableWritableListableStorage, ReadableWritableListableStorageTraits};

use crate::errors::FitsError;

const IMAGES_ROOT: &str = "/images";
const BY_SHAPE_ROOT: &str = "/images/by_shape";
const RAW_DATASET_NAME: &str = "raw";
const SCANS_SEGMENT: &str = "scans";
const DEFAULT_SHARD_FRAMES: u64 = 64;

pub const ZARR_U16_ZSTD_LEVEL: i32 = 9;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShapeScanBucketSpec {
    pub shape_bucket: String,
    pub scan_number: i32,
    pub height: usize,
    pub width: usize,
    pub frames: usize,
}

fn ensure_group(store: &ReadableWritableListableStorage, path: &str) -> Result<(), FitsError> {
    let group = GroupBuilder::new()
        .build(store.clone(), path)
        .map_err(|e| FitsError::validation(e.to_string()))?;
    group
        .store_metadata()
        .map_err(|e| FitsError::validation(e.to_string()))
}

pub fn scan_raw_array_path(shape_bucket: &str, scan_number: i32) -> String {
    format!("{BY_SHAPE_ROOT}/{shape_bucket}/{SCANS_SEGMENT}/{scan_number}/{RAW_DATASET_NAME}")
}

pub fn open_zarr_store(zarr_root: &Path) -> Result<ReadableWritableListableStorage, FitsError> {
    std::fs::create_dir_all(zarr_root).map_err(|e| FitsError::io("create zarr root", e))?;
    let store: ReadableWritableListableStorage = Arc::new(
        zarrs::filesystem::FilesystemStore::new(zarr_root)
            .map_err(|e| FitsError::validation(e.to_string()))?,
    );
    Ok(store)
}

pub fn shape_bucket_key(height: usize, width: usize) -> String {
    format!("{height}x{width}")
}

fn open_or_create_scan_bucket_array(
    store: &ReadableWritableListableStorage,
    spec: &ShapeScanBucketSpec,
) -> Result<Array<dyn ReadableWritableListableStorageTraits>, FitsError> {
    let path = scan_raw_array_path(&spec.shape_bucket, spec.scan_number);
    if let Ok(array) = Array::open(store.clone(), &path) {
        return Ok(array);
    }
    let n = spec.frames as u64;
    let shard_frames = n.max(1).min(DEFAULT_SHARD_FRAMES);
    let shape = vec![n, spec.height as u64, spec.width as u64];
    let chunk_shape = vec![shard_frames, spec.height as u64, spec.width as u64];
    let mut builder = ArrayBuilder::new(shape, chunk_shape, data_type::uint16(), 0u16);
    builder.bytes_to_bytes_codecs(vec![
        Arc::new(zarrs::array::codec::ShuffleCodec::new(std::mem::size_of::<
            u16,
        >())),
        Arc::new(zarrs::array::codec::ZstdCodec::new(
            ZARR_U16_ZSTD_LEVEL,
            false,
        )),
    ]);
    builder.subchunk_shape(vec![1_u64, spec.height as u64, spec.width as u64]);
    let array = builder
        .build(store.clone(), &path)
        .map_err(|e| FitsError::validation(e.to_string()))?;
    array
        .store_metadata()
        .map_err(|e| FitsError::validation(e.to_string()))?;
    Ok(array)
}

pub fn prepare_shape_scan_bucket_arrays(
    store: &ReadableWritableListableStorage,
    specs: &[ShapeScanBucketSpec],
) -> Result<(), FitsError> {
    ensure_group(store, "/")?;
    ensure_group(store, IMAGES_ROOT)?;
    ensure_group(store, BY_SHAPE_ROOT)?;
    for spec in specs {
        ensure_group(store, &format!("{BY_SHAPE_ROOT}/{}", spec.shape_bucket))?;
        ensure_group(
            store,
            &format!("{BY_SHAPE_ROOT}/{}/{SCANS_SEGMENT}", spec.shape_bucket),
        )?;
        ensure_group(
            store,
            &format!(
                "{BY_SHAPE_ROOT}/{}/{SCANS_SEGMENT}/{}",
                spec.shape_bucket, spec.scan_number
            ),
        )?;
        let _ = open_or_create_scan_bucket_array(store, spec)?;
    }
    Ok(())
}

pub fn write_scan_shape_bucket_frame_raw(
    store: &ReadableWritableListableStorage,
    shape_bucket: &str,
    scan_number: i32,
    bucket_frame_index: usize,
    data: &Array2<u16>,
) -> Result<(), FitsError> {
    let path = scan_raw_array_path(shape_bucket, scan_number);
    let array =
        Array::open(store.clone(), &path).map_err(|e| FitsError::validation(e.to_string()))?;
    let subset = ArraySubset::new_with_start_shape(
        vec![bucket_frame_index as u64, 0_u64, 0_u64],
        vec![1_u64, data.nrows() as u64, data.ncols() as u64],
    )
    .map_err(|e| FitsError::validation(e.to_string()))?;
    let flat: Vec<u16> = data.iter().copied().collect();
    array
        .store_array_subset(&subset, flat)
        .map_err(|e| FitsError::validation(e.to_string()))
}

#[allow(dead_code)]
pub fn write_frame_raw(
    store: &ReadableWritableListableStorage,
    scan_number: i64,
    frame_number: i64,
    data: &Array2<i32>,
) -> Result<(), FitsError> {
    let base = format!("/{scan_number}/{frame_number:05}");
    ensure_group(store, "/")?;
    ensure_group(store, &format!("/{scan_number}"))?;
    ensure_group(store, &base)?;
    let path = format!("{base}/raw");
    let h = data.nrows();
    let w = data.ncols();
    let shape: Vec<u64> = vec![h as u64, w as u64];
    let chunk_shape: Vec<u64> = vec![h as u64, w as u64];
    let flat: Vec<i32> = data.iter().copied().collect();
    let array = ArrayBuilder::new(shape, chunk_shape, data_type::int32(), 0i32)
        .build(store.clone(), &path)
        .map_err(|e| FitsError::validation(e.to_string()))?;
    array
        .store_metadata()
        .map_err(|e| FitsError::validation(e.to_string()))?;
    array
        .store_chunk(&[0u64, 0u64], flat)
        .map_err(|e| FitsError::validation(e.to_string()))?;
    Ok(())
}

#[cfg(all(test, feature = "catalog"))]
mod tests {
    use super::*;
    use ndarray::Array2;
    use tempfile::TempDir;
    use walkdir::WalkDir;

    #[test]
    fn write_frame_raw_creates_only_raw_group() {
        let tmp = TempDir::new().expect("create tempdir");
        let store = open_zarr_store(tmp.path()).expect("open zarr store");
        let scan_number: i64 = 42;
        let frame_number: i64 = 7;
        let data: Array2<i32> = Array2::from_shape_fn((4, 5), |(r, c)| (r * 10 + c) as i32);

        write_frame_raw(&store, scan_number, frame_number, &data).expect("write frame");

        let raw_path = tmp
            .path()
            .join(scan_number.to_string())
            .join(format!("{frame_number:05}"))
            .join("raw");
        assert!(raw_path.is_dir(), "expected raw array dir at {raw_path:?}");

        let proc_path = tmp
            .path()
            .join(scan_number.to_string())
            .join(format!("{frame_number:05}"))
            .join("processed");
        assert!(
            !proc_path.exists(),
            "expected no processed array at {proc_path:?}"
        );
    }

    fn dir_size_bytes(path: &Path) -> u64 {
        WalkDir::new(path)
            .into_iter()
            .filter_map(std::result::Result::ok)
            .filter(|e| e.file_type().is_file())
            .filter_map(|e| e.metadata().ok())
            .map(|m| m.len())
            .sum()
    }

    #[test]
    fn bucketed_uint16_layout_uses_less_disk_than_legacy_per_frame_int32() {
        let old_tmp = TempDir::new().expect("create old tempdir");
        let new_tmp = TempDir::new().expect("create new tempdir");
        let old_store = open_zarr_store(old_tmp.path()).expect("open old zarr store");
        let new_store = open_zarr_store(new_tmp.path()).expect("open new zarr store");
        let h = 64_usize;
        let w = 64_usize;
        let n = 16_usize;
        let old_frame = Array2::from_elem((h, w), 512_i32);
        for i in 0..n {
            write_frame_raw(&old_store, 1, i as i64, &old_frame).expect("write legacy frame");
        }
        let key = shape_bucket_key(h, w);
        let specs = vec![ShapeScanBucketSpec {
            shape_bucket: key.clone(),
            scan_number: 1,
            height: h,
            width: w,
            frames: n,
        }];
        prepare_shape_scan_bucket_arrays(&new_store, &specs).expect("prepare bucket arrays");
        let new_frame = Array2::from_elem((h, w), 512_u16);
        for i in 0..n {
            write_scan_shape_bucket_frame_raw(&new_store, &key, 1, i, &new_frame)
                .expect("write bucket frame");
        }
        let old_bytes = dir_size_bytes(old_tmp.path());
        let new_bytes = dir_size_bytes(new_tmp.path());
        assert!(
            new_bytes < old_bytes,
            "expected bucketed layout to be smaller (new={new_bytes}, old={old_bytes})"
        );
    }
}

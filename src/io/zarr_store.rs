use ndarray::Array2;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use zarrs::array::data_type;
use zarrs::array::{Array, ArrayBuilder, ArraySubset};
use zarrs::group::GroupBuilder;
use zarrs::storage::{ReadableWritableListableStorage, ReadableWritableListableStorageTraits};

use crate::errors::FitsError;

const PYREF_CATALOG_DIR: &str = ".pyref";
const ZARR_SUBDIR: &str = "zarr";
const DETECTOR_ARRAY_NAME: &str = "detector";

fn catalog_parent(beamtime_dir: &Path) -> PathBuf {
    let parent = beamtime_dir.parent();
    if let Some(p) = parent {
        if !p.as_os_str().is_empty() && p != Path::new("/") {
            return p.join(PYREF_CATALOG_DIR);
        }
    }
    beamtime_dir.join(PYREF_CATALOG_DIR)
}

fn beamtime_key(beamtime_dir: &Path) -> String {
    beamtime_dir
        .file_name()
        .map(|s| s.to_string_lossy())
        .unwrap_or_else(|| beamtime_dir.as_os_str().to_string_lossy())
        .replace(std::path::MAIN_SEPARATOR, "_")
}

pub fn zarr_root(beamtime_dir: &Path) -> PathBuf {
    catalog_parent(beamtime_dir)
        .join(ZARR_SUBDIR)
        .join(beamtime_key(beamtime_dir))
}

pub fn detector_zarr_group(scan_uid: &str) -> String {
    format!("scans/{}/{}", scan_uid, DETECTOR_ARRAY_NAME)
}

fn ensure_group(store: &ReadableWritableListableStorage, path: &str) -> Result<(), FitsError> {
    let group = GroupBuilder::new()
        .build(store.clone(), path)
        .map_err(|e| FitsError::validation(e.to_string()))?;
    group
        .store_metadata()
        .map_err(|e| FitsError::validation(e.to_string()))?;
    Ok(())
}

fn ensure_scans_groups(
    store: &ReadableWritableListableStorage,
    scan_uid: &str,
) -> Result<(), FitsError> {
    ensure_group(store, "/")?;
    ensure_group(store, "/scans")?;
    let scan_group = format!("/scans/{}", scan_uid);
    ensure_group(store, &scan_group)?;
    Ok(())
}

pub fn create_detector_array(
    store: &ReadableWritableListableStorage,
    scan_uid: &str,
    n_frames: usize,
    height: usize,
    width: usize,
) -> Result<Array<dyn ReadableWritableListableStorageTraits>, FitsError> {
    ensure_scans_groups(store, scan_uid)?;
    let path = format!("/scans/{}/{}", scan_uid, DETECTOR_ARRAY_NAME);
    let shape: Vec<u64> = vec![n_frames as u64, height as u64, width as u64];
    let chunk_shape: Vec<u64> = vec![1, height as u64, width as u64];
    let array = ArrayBuilder::new(shape, chunk_shape, data_type::int32(), 0i32)
        .build(store.clone(), &path)
        .map_err(|e| FitsError::validation(e.to_string()))?;
    array
        .store_metadata()
        .map_err(|e| FitsError::validation(e.to_string()))?;
    Ok(array)
}

pub fn write_detector_frame(
    array: &Array<dyn ReadableWritableListableStorageTraits>,
    frame_index: usize,
    data: &Array2<i32>,
) -> Result<(), FitsError> {
    let flat: Vec<i32> = data.iter().copied().collect();
    let ci = [frame_index as u64, 0u64, 0u64];
    array
        .store_chunk(&ci, flat)
        .map_err(|e| FitsError::validation(e.to_string()))?;
    Ok(())
}

pub fn open_detector_array(
    store: &ReadableWritableListableStorage,
    scan_uid: &str,
) -> Result<Array<dyn ReadableWritableListableStorageTraits>, FitsError> {
    let path = format!("/scans/{}/{}", scan_uid, DETECTOR_ARRAY_NAME);
    let array =
        Array::open(store.clone(), &path).map_err(|e| FitsError::validation(e.to_string()))?;
    Ok(array)
}

pub fn read_detector_frame(
    array: &Array<dyn ReadableWritableListableStorageTraits>,
    frame_index: usize,
) -> Result<Array2<i32>, FitsError> {
    let shape = array.shape();
    if shape.len() != 3 {
        return Err(FitsError::validation("detector array must be 3D"));
    }
    let height = shape[1] as usize;
    let width = shape[2] as usize;
    let subset_region = ArraySubset::new_with_ranges(&[
        (frame_index as u64)..(frame_index as u64 + 1),
        0..(height as u64),
        0..(width as u64),
    ]);
    let flat: Vec<i32> = array
        .retrieve_array_subset::<Vec<i32>>(&subset_region)
        .map_err(|e| FitsError::validation(e.to_string()))?;
    Array2::from_shape_vec((height, width), flat).map_err(|e| FitsError::validation(e.to_string()))
}

pub fn open_store(zarr_root: &Path) -> Result<ReadableWritableListableStorage, FitsError> {
    std::fs::create_dir_all(zarr_root).map_err(|e| FitsError::io("create zarr root", e))?;
    let store: ReadableWritableListableStorage = Arc::new(
        zarrs::filesystem::FilesystemStore::new(zarr_root)
            .map_err(|e| FitsError::validation(e.to_string()))?,
    );
    Ok(store)
}

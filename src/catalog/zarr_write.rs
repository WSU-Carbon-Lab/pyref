//! Write per-frame detector arrays into the beamtime zarr store (layout ``/{scan}/{frame}/raw``).

use ndarray::Array2;
use std::path::Path;
use std::sync::Arc;
use zarrs::array::data_type;
use zarrs::array::ArrayBuilder;
use zarrs::group::GroupBuilder;
use zarrs::storage::ReadableWritableListableStorage;

use crate::errors::FitsError;

fn ensure_group(store: &ReadableWritableListableStorage, path: &str) -> Result<(), FitsError> {
    let group = GroupBuilder::new()
        .build(store.clone(), path)
        .map_err(|e| FitsError::validation(e.to_string()))?;
    group
        .store_metadata()
        .map_err(|e| FitsError::validation(e.to_string()))
}

/// Opens or creates the filesystem zarr root at ``zarr_root`` (the ``beamtime.zarr`` directory).
pub fn open_zarr_store(zarr_root: &Path) -> Result<ReadableWritableListableStorage, FitsError> {
    std::fs::create_dir_all(zarr_root).map_err(|e| FitsError::io("create zarr root", e))?;
    let store: ReadableWritableListableStorage = Arc::new(
        zarrs::filesystem::FilesystemStore::new(zarr_root)
            .map_err(|e| FitsError::validation(e.to_string()))?,
    );
    Ok(store)
}

/// Creates a 2D ``raw`` int32 array at ``/{scan_number}/{frame_number}/raw`` and writes ``data``.
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
    let proc_path = format!("{base}/processed");
    let proc = ArrayBuilder::new(
        vec![h as u64, w as u64],
        vec![h as u64, w as u64],
        data_type::int32(),
        0i32,
    )
    .build(store.clone(), &proc_path)
    .map_err(|e| FitsError::validation(e.to_string()))?;
    proc.store_metadata()
        .map_err(|e| FitsError::validation(e.to_string()))?;
    proc.store_chunk(&[0u64, 0u64], data.iter().copied().collect::<Vec<i32>>())
        .map_err(|e| FitsError::validation(e.to_string()))?;
    Ok(())
}

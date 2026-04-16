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
    Ok(())
}

#[cfg(all(test, feature = "catalog"))]
mod tests {
    use super::*;
    use ndarray::Array2;
    use tempfile::TempDir;

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
}

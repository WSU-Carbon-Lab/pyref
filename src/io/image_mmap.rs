use std::path::{Path, PathBuf};
#[cfg(feature = "catalog")]
use std::sync::Arc;

use ndarray::Array2;
use polars::prelude::*;
#[cfg(feature = "catalog")]
use zarrs::array::{Array, ArraySubset};
#[cfg(feature = "catalog")]
use zarrs::storage::ReadableWritableListableStorage;

use super::blur::{gaussian_blur_f32_copy, i64_to_f32_array};
use super::raw_pixels::read_bitpix16_be_bytes;
use super::{
    subtract_background_edges, subtract_background_row_strips, subtract_dark_cold_side,
    trim_image_interior, ImageInfo, TRIM_COLS, TRIM_ROWS,
};
use crate::errors::FitsError;
use crate::fits::HduList;

type ImagePair = (Array2<i64>, Array2<i64>);

#[cfg(feature = "catalog")]
fn try_load_image_pixels_from_zarr(info: &ImageInfo) -> Result<Option<Array2<i64>>, FitsError> {
    let zarr_path = match &info.zarr_path {
        Some(path) => path,
        None => return Ok(None),
    };
    let store: ReadableWritableListableStorage = Arc::new(
        zarrs::filesystem::FilesystemStore::new(zarr_path)
            .map_err(|e| FitsError::validation(e.to_string()))?,
    );
    if let (Some(bucket), Some(bucket_idx), Some(scan_no)) = (
        &info.zarr_shape_bucket,
        info.zarr_bucket_frame_index,
        info.zarr_group_key,
    ) {
        let path = format!("/images/by_shape/{bucket}/scans/{scan_no}/raw");
        if let Ok(array) = Array::open(store.clone(), &path) {
            let subset_region = ArraySubset::new_with_ranges(&[
                (bucket_idx as u64)..(bucket_idx as u64 + 1),
                0..(info.naxis2 as u64),
                0..(info.naxis1 as u64),
            ]);
            let subset: Vec<u16> = array
                .retrieve_array_subset::<Vec<u16>>(&subset_region)
                .map_err(|e| FitsError::validation(e.to_string()))?;
            let flat: Vec<i64> = subset
                .iter()
                .map(|raw| (*raw as i16 as i64) + info.bzero)
                .collect();
            let arr = Array2::from_shape_vec((info.naxis2, info.naxis1), flat)
                .map_err(|e| FitsError::validation(e.to_string()))?;
            return Ok(Some(arr));
        }
    }
    if let (Some(group_key), Some(frame_index)) = (info.zarr_group_key, info.zarr_frame_index) {
        let path = format!("/{group_key}/{frame_index:05}/raw");
        if let Ok(array) = Array::open(store, &path) {
            let subset_region =
                ArraySubset::new_with_ranges(&[0..(info.naxis2 as u64), 0..(info.naxis1 as u64)]);
            let subset: Vec<i32> = array
                .retrieve_array_subset::<Vec<i32>>(&subset_region)
                .map_err(|e| FitsError::validation(e.to_string()))?;
            let flat: Vec<i64> = subset.iter().map(|value| *value as i64).collect();
            let arr = Array2::from_shape_vec((info.naxis2, info.naxis1), flat)
                .map_err(|e| FitsError::validation(e.to_string()))?;
            return Ok(Some(arr));
        }
    }
    Ok(None)
}

pub fn load_image_pixels(path: &Path, info: &ImageInfo) -> Result<Array2<i64>, FitsError> {
    #[cfg(feature = "catalog")]
    if let Ok(Some(data)) = try_load_image_pixels_from_zarr(info) {
        return Ok(data);
    }
    if info.bitpix != 16 {
        return Err(FitsError::unsupported(
            "Only BITPIX=16 image HDUs supported",
        ));
    }
    let nelem = info.naxis1 * info.naxis2;
    let nbytes = nelem * 2;
    let bytes = read_bitpix16_be_bytes(path, info.data_offset, nbytes)?;
    let raw: Vec<i64> = bytes
        .chunks_exact(2)
        .map(|c| i16::from_be_bytes([c[0], c[1]]) as i64 + info.bzero)
        .collect();
    Array2::from_shape_vec((info.naxis2, info.naxis1), raw)
        .map_err(|e| FitsError::validation(e.to_string()))
}

pub fn materialize_image(path: &Path, info: &ImageInfo) -> Result<ImagePair, FitsError> {
    let data = load_image_pixels(path, info)?;
    let trimmed_raw = trim_image_interior(&data, TRIM_ROWS, TRIM_COLS);
    let row_corrected = subtract_background_row_strips(&trimmed_raw);
    let trimmed_processed = subtract_dark_cold_side(&row_corrected);
    Ok((trimmed_raw, trimmed_processed))
}

pub fn get_image_for_row(df: &DataFrame, row_index: usize) -> Result<ImagePair, FitsError> {
    let info = ImageInfo::from_dataframe_row(df, row_index)?;
    materialize_image(info.path.as_path(), &info)
}

pub fn materialize_image_from_path(path: &Path) -> Result<ImagePair, FitsError> {
    let path_str = path
        .to_str()
        .ok_or_else(|| FitsError::validation("Invalid UTF-8 in path"))?;
    let hdul = HduList::from_file_headers_only(path_str).map_err(FitsError::from)?;
    let image_header = hdul.image_header.as_ref().ok_or_else(|| {
        FitsError::validation("No image HDU found").with_context("path", path_str)
    })?;
    let info = ImageInfo::from_header(PathBuf::from(path), image_header);
    materialize_image(path, &info)
}

pub fn materialize_image_unprocessed(
    df: &DataFrame,
    row_index: usize,
) -> Result<Array2<i64>, FitsError> {
    let info = ImageInfo::from_dataframe_row(df, row_index)?;
    load_image_pixels(info.path.as_path(), &info)
}

pub fn materialize_image_filtered(
    path: &Path,
    info: &ImageInfo,
    sigma: f64,
) -> Result<Array2<f32>, FitsError> {
    let (_raw, subtracted) = materialize_image(path, info)?;
    let f32_arr = i64_to_f32_array(&subtracted);
    let (h, w) = (f32_arr.nrows(), f32_arr.ncols());
    let slice = f32_arr
        .as_slice()
        .ok_or_else(|| FitsError::validation("image layout not contiguous"))?;
    let blurred = gaussian_blur_f32_copy(slice, w as u32, h as u32, sigma).map_err(|e| {
        FitsError::validation(e.to_string())
            .with_context("operation", "gaussian_blur")
            .with_context("path", path.display().to_string())
    })?;
    Array2::from_shape_vec((h, w), blurred).map_err(|e| FitsError::validation(e.to_string()))
}

pub fn materialize_image_corrected(
    path: &Path,
    info: &ImageInfo,
    bg_rows: usize,
    bg_cols: usize,
) -> Result<Array2<i64>, FitsError> {
    let data = load_image_pixels(path, info)?;
    Ok(subtract_background_edges(&data, bg_rows, bg_cols))
}

pub fn materialize_image_filtered_edges(
    path: &Path,
    info: &ImageInfo,
    sigma: f64,
    bg_rows: usize,
    bg_cols: usize,
) -> Result<Array2<f32>, FitsError> {
    let corrected = materialize_image_corrected(path, info, bg_rows, bg_cols)?;
    let f32_arr = i64_to_f32_array(&corrected);
    let (h, w) = (f32_arr.nrows(), f32_arr.ncols());
    let slice = f32_arr
        .as_slice()
        .ok_or_else(|| FitsError::validation("image layout not contiguous"))?;
    let blurred = gaussian_blur_f32_copy(slice, w as u32, h as u32, sigma).map_err(|e| {
        FitsError::validation(e.to_string())
            .with_context("operation", "gaussian_blur")
            .with_context("path", path.display().to_string())
    })?;
    Array2::from_shape_vec((h, w), blurred).map_err(|e| FitsError::validation(e.to_string()))
}

use std::fs::File;
use std::path::Path;

use memmap2::MmapOptions;
use ndarray::{Array2, ArrayBase, Dim, OwnedRepr};
use polars::prelude::*;

use crate::errors::FitsError;
use super::blur::{gaussian_blur_f32_copy, i64_to_f32_array};
use super::{subtract_background, subtract_background_edges, ImageInfo};

fn load_image_pixels(path: &Path, info: &ImageInfo) -> Result<Array2<i64>, FitsError> {
    if info.bitpix != 16 {
        return Err(FitsError::unsupported("Only BITPIX=16 image HDUs supported"));
    }
    let nelem = info.naxis1 * info.naxis2;
    let nbytes = nelem * 2;
    let file = File::open(path).map_err(|e| FitsError::io("open", e))?;
    let mmap = unsafe {
        MmapOptions::new()
            .offset(info.data_offset)
            .len(nbytes)
            .map(&file)
            .map_err(|e| FitsError::io("mmap", e))?
    };
    let mut raw = Vec::with_capacity(nelem);
    for chunk in mmap.chunks_exact(2) {
        let v = i16::from_be_bytes([chunk[0], chunk[1]]) as i64 + info.bzero;
        raw.push(v);
    }
    drop(mmap);
    drop(file);
    Array2::from_shape_vec((info.naxis2, info.naxis1), raw)
        .map_err(|e| FitsError::validation(e.to_string()))
}

pub fn materialize_image(
    path: &Path,
    info: &ImageInfo,
) -> Result<(ArrayBase<OwnedRepr<i64>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<i64>, Dim<[usize; 2]>>), FitsError> {
    let data = load_image_pixels(path, info)?;
    let subtracted = subtract_background(&data.clone().into_dyn());
    let subtracted_2d = subtracted
        .into_shape_with_order((info.naxis2, info.naxis1))
        .map_err(|e| FitsError::validation(e.to_string()))?;
    Ok((data, subtracted_2d))
}

pub fn get_image_for_row(
    df: &DataFrame,
    row_index: usize,
) -> Result<(ArrayBase<OwnedRepr<i64>, Dim<[usize; 2]>>, ArrayBase<OwnedRepr<i64>, Dim<[usize; 2]>>), FitsError> {
    let info = ImageInfo::from_dataframe_row(df, row_index)?;
    materialize_image(info.path.as_path(), &info)
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
    Array2::from_shape_vec((h, w), blurred)
        .map_err(|e| FitsError::validation(e.to_string()))
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
    Array2::from_shape_vec((h, w), blurred)
        .map_err(|e| FitsError::validation(e.to_string()))
}

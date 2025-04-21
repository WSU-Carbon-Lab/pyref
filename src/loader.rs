use astrors_fork::fits;
use astrors_fork::io::hdulist::HDU;

use polars::{lazy::prelude::*, prelude::*};
use rayon::prelude::*;
use std::fs;
use std::path::PathBuf;

use crate::enums::{ExperimentType, HeaderValue};
use crate::errors::FitsLoaderError;
use crate::io::{add_calculated_domains, process_file_name, process_image, process_metadata};

/// Reads a single FITS file and converts it to a Polars DataFrame.
///
/// # Arguments
///
/// * `file_path` - Path to the FITS file to read
/// * `header_items` - List of header values to extract
///
/// # Returns
///
/// A `Result` containing either the DataFrame or a `FitsLoaderError`.
pub fn read_fits(
    file_path: std::path::PathBuf,
    header_items: &Vec<HeaderValue>,
) -> Result<DataFrame, FitsLoaderError> {
    if file_path.extension().and_then(|ext| ext.to_str()) != Some("fits") {
        return Err(FitsLoaderError::NoData);
    }

    let hdul = fits::fromfile(
        file_path
            .to_str()
            .ok_or_else(|| FitsLoaderError::InvalidFileName("Invalid path".into()))?,
    )?;

    let meta = match hdul.hdus.get(0) {
        Some(HDU::Primary(hdu)) => process_metadata(hdu, header_items)?,
        _ => return Err(FitsLoaderError::NoData),
    };

    let img_data = match hdul.hdus.get(2) {
        Some(HDU::Image(hdu)) => process_image(hdu)?,
        _ => return Err(FitsLoaderError::NoData),
    };

    let names = process_file_name(file_path);

    let mut columns = meta;
    columns.extend(img_data);
    columns.extend(names);

    DataFrame::new(columns).map_err(FitsLoaderError::PolarsError)
}

/// Reads all FITS files in a directory and combines them into a single DataFrame.
///
/// # Arguments
///
/// * `dir` - Path to the directory containing FITS files
/// * `header_items` - List of header values to extract
///
/// # Returns
///
/// A `Result` containing either the combined DataFrame or a `FitsLoaderError`.
pub fn read_experiment(
    dir: &str,
    header_items: &Vec<HeaderValue>,
) -> Result<DataFrame, FitsLoaderError> {
    let dir_path = std::path::PathBuf::from(dir);

    if !dir_path.exists() {
        return Err(FitsLoaderError::NoData);
    }

    let entries: Vec<_> = fs::read_dir(dir)
        .map_err(FitsLoaderError::IoError)?
        .par_bridge()
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().extension().and_then(|ext| ext.to_str()) == Some("fits"))
        .collect();

    let dataframes: Result<Vec<DataFrame>, FitsLoaderError> = entries
        .par_iter()
        .map(|entry| read_fits(entry.path(), &header_items))
        .collect();

    let combined_df = dataframes?
        .into_par_iter()
        .reduce_with(|acc, df| acc.vstack(&df).unwrap_or(DataFrame::empty()))
        .ok_or(FitsLoaderError::NoData)?;

    Ok(add_calculated_domains(combined_df.lazy()))
}

/// Reads multiple specific FITS files and combines them into a single DataFrame.
///
/// # Arguments
///
/// * `file_paths` - Vector of paths to the FITS files to read
/// * `header_items` - List of header values to extract
///
/// # Returns
///
/// A `Result` containing either the combined DataFrame or a `FitsLoaderError`.
pub fn read_multiple_fits(
    file_paths: Vec<PathBuf>,
    header_items: &Vec<HeaderValue>,
) -> Result<DataFrame, FitsLoaderError> {
    if file_paths.is_empty() {
        return Err(FitsLoaderError::NoData);
    }

    let dataframes: Result<Vec<DataFrame>, FitsLoaderError> = file_paths
        .par_iter()
        .map(|path| read_fits(path.clone(), header_items))
        .collect();

    let combined_df = dataframes?
        .into_par_iter()
        .reduce_with(|acc, df| acc.vstack(&df).unwrap_or(DataFrame::empty()))
        .ok_or(FitsLoaderError::NoData)?;

    Ok(add_calculated_domains(combined_df.lazy()))
}

/// Reads FITS files matching a pattern and combines them into a single DataFrame.
///
/// # Arguments
///
/// * `dir` - Directory containing FITS files
/// * `pattern` - Glob pattern to match files (e.g., "Y6_refl_*.fits")
/// * `experiment_type` - Type of experiment
///
/// # Returns
///
/// A `Result` containing either the combined DataFrame or a `FitsLoaderError`.
pub fn read_experiment_pattern(
    dir: &str,
    pattern: &str,
    experiment_type: ExperimentType,
) -> Result<DataFrame, FitsLoaderError> {
    let dir_path = std::path::PathBuf::from(dir);

    if !dir_path.exists() {
        return Err(FitsLoaderError::NoData);
    }

    let header_items = experiment_type.get_keys();

    let entries: Vec<_> = fs::read_dir(dir)
        .map_err(FitsLoaderError::IoError)?
        .par_bridge()
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            let path = entry.path();
            path.extension().and_then(|ext| ext.to_str()) == Some("fits")
                && match path.file_name().and_then(|name| name.to_str()) {
                    Some(name) => glob_match::glob_match(pattern, name),
                    None => false,
                }
        })
        .map(|entry| entry.path())
        .collect();

    read_multiple_fits(entries, &header_items)
}

// Utility test function
pub fn _load() {
    let test_path = "C:/Users/hduva/.projects/pyref-ccd/testing/Y6_refl_ 001096 Images/Y6_refl_ 001096 CCD 000.fits";
    let hdus = ExperimentType::Xrr.get_keys();
    let data = read_fits(test_path.into(), &hdus).unwrap();
    println!("{:?}", data);
}

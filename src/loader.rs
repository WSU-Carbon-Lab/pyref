use astrors_fork::fits;
use astrors_fork::io::hdulist::HDU;

use polars::{lazy::prelude::*, prelude::*}; // Add the import statement for PolarsError
use rayon::prelude::*;
use std::fs;

use crate::enums::HeaderValue;
use crate::errors::FitsLoaderError;
use crate::io::{add_calculated_domains, process_file_name, process_image, process_metadata};
// Enum representing different types of experiments.

// Polars Helper Function

// workhorse functions for loading and processing CCD data.
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

pub fn _load() {
    // let test_path = "/home/hduva/projects/pyref/test/stack/";

    // let data = read_experiment(test_path.into(), &ExperimentType::Xrr.get_keys()).unwrap();
    // println!("{:?}", data);
}

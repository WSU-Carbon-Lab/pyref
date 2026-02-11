use polars::{lazy::prelude::*, prelude::*};
use rayon::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};

use crate::errors::{FitsLoaderError, LoaderError};
use crate::fits::{Hdu, HduList};
use crate::io::{
    add_calculated_domains, parse_fits_stem, process_file_name, process_image, process_image_header,
    process_metadata,
};
use std::collections::HashSet;

#[derive(Debug, Clone, Default)]
pub struct StemCatalog {
    pub samples: Vec<String>,
    pub experiment_count: u32,
    pub fits_count: u32,
}

pub fn catalog_from_stems(paths: &[PathBuf]) -> StemCatalog {
    let fits_count = paths.len() as u32;
    if paths.is_empty() {
        return StemCatalog {
            samples: Vec::new(),
            experiment_count: 0,
            fits_count: 0,
        };
    }
    let (samples_set, experiment_set): (HashSet<String>, HashSet<i64>) = paths
        .par_iter()
        .filter_map(|p| {
            let stem = p.file_stem().and_then(|s| s.to_str())?;
            parse_fits_stem(stem).map(|parsed| {
                let samples: HashSet<String> = if parsed.sample_name.is_empty() {
                    HashSet::new()
                } else {
                    [parsed.sample_name].into_iter().collect()
                };
                let experiments: HashSet<i64> = if parsed.experiment_number > 0 {
                    [parsed.experiment_number].into_iter().collect()
                } else {
                    HashSet::new()
                };
                (samples, experiments)
            })
        })
        .reduce(
            || (HashSet::new(), HashSet::new()),
            |(mut a_s, mut a_e), (b_s, b_e)| {
                a_s.extend(b_s);
                a_e.extend(b_e);
                (a_s, a_e)
            },
        );
    let mut samples: Vec<String> = samples_set.into_iter().collect();
    samples.sort();
    StemCatalog {
        samples,
        experiment_count: experiment_set.len() as u32,
        fits_count,
    }
}

pub fn list_fits_in_dir(dir: &Path) -> Result<Vec<PathBuf>, LoaderError> {
    let entries = fs::read_dir(dir).map_err(|e| LoaderError::list_fits_io(dir.display(), e))?;
    let paths: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .filter(|e| {
            let p = e.path();
            p.extension().and_then(|ext| ext.to_str()) == Some("fits") && p.is_file()
        })
        .map(|e| e.path())
        .collect();
    Ok(paths)
}

pub fn read_fits_metadata_sampled(
    paths: &[PathBuf],
    header_items: &[String],
    max_files: usize,
) -> Result<DataFrame, LoaderError> {
    if paths.is_empty() {
        return Err(LoaderError {
            kind: crate::errors::LoaderErrorKind::ValidationFailed,
            retryable: false,
            message: "no paths provided".into(),
            context: vec![("operation".into(), "read_fits_metadata_sampled".into())],
            source: None,
        });
    }
    let header_vec: Vec<String> = header_items.to_vec();
    let sampled: Vec<PathBuf> = paths.iter().take(max_files).cloned().collect();
    let results: Vec<Result<DataFrame, FitsLoaderError>> = sampled
        .par_iter()
        .map(|path| read_fits_metadata(path.clone(), &header_vec))
        .collect();
    let successful_dfs: Vec<DataFrame> = results.into_iter().filter_map(|r| r.ok()).collect();
    if successful_dfs.is_empty() {
        return Err(LoaderError {
            kind: crate::errors::LoaderErrorKind::Permanent,
            retryable: false,
            message: "none of the sampled files could be read for metadata".into(),
            context: vec![
                ("operation".into(), "read_fits_metadata_sampled".into()),
                ("path_count".into(), sampled.len().to_string()),
            ],
            source: None,
        });
    }
    let combined = successful_dfs
        .into_par_iter()
        .reduce_with(|acc, df| match acc.vstack(&df) {
            Ok(c) => c,
            Err(_) => acc,
        })
        .ok_or_else(|| LoaderError {
            kind: crate::errors::LoaderErrorKind::Permanent,
            retryable: false,
            message: "failed to combine metadata".into(),
            context: vec![("operation".into(), "read_fits_metadata_sampled".into())],
            source: None,
        })?;
    let lz = combined
        .lazy()
        .sort(["experiment_number", "frame_number"], Default::default());
    Ok(add_calculated_domains(lz))
}

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
    header_items: &Vec<String>,
) -> Result<DataFrame, FitsLoaderError> {
    if file_path.extension().and_then(|ext| ext.to_str()) != Some("fits") {
        return Err(FitsLoaderError::NoData);
    }

    // Safely get path as string
    let path_str = file_path
        .to_str()
        .ok_or_else(|| FitsLoaderError::InvalidFileName("Invalid UTF-8 in path".into()))?;

    let result = (|| {
        let hdul = HduList::from_file(path_str)?;
        let meta = match hdul.hdus.get(0) {
            Some(Hdu::Primary(hdu)) => process_metadata(hdu, header_items)?,
            _ => return Err(FitsLoaderError::NoData),
        };
        let img_data = match hdul.hdus.get(2) {
            Some(Hdu::Image(hdu)) => process_image(hdu)?,
            _ => match hdul.hdus.get(1) {
                Some(Hdu::Image(hdu)) => process_image(hdu)?,
                _ => return Err(FitsLoaderError::NoData),
            },
        };

        // Extract file name information
        let names = process_file_name(file_path.clone());

        // Combine all columns
        let mut columns = meta;
        columns.extend(img_data);
        columns.extend(names);

        // Create DataFrame
        DataFrame::new(columns).map_err(FitsLoaderError::PolarsError)
    })();

    // Add file path to error context if an error occurred
    result.map_err(|e| {
        FitsLoaderError::FitsError(format!("Error processing file '{}': {}", path_str, e))
    })
}

pub fn read_fits_metadata(
    file_path: std::path::PathBuf,
    header_items: &Vec<String>,
) -> Result<DataFrame, FitsLoaderError> {
    if file_path.extension().and_then(|ext| ext.to_str()) != Some("fits") {
        return Err(FitsLoaderError::NoData);
    }
    let path_str = file_path
        .to_str()
        .ok_or_else(|| FitsLoaderError::InvalidFileName("Invalid UTF-8 in path".into()))?;
    let result = (|| {
        let hdul = HduList::from_file_metadata_only(path_str)?;
        let meta = match hdul.hdus.get(0) {
            Some(Hdu::Primary(hdu)) => process_metadata(hdu, header_items)?,
            _ => return Err(FitsLoaderError::NoData),
        };
        let naxis_cols = match hdul.hdus.get(2) {
            Some(Hdu::ImageHeader(h)) => process_image_header(h),
            _ => match hdul.hdus.get(1) {
                Some(Hdu::ImageHeader(h)) => process_image_header(h),
                _ => vec![
                    Column::new("NAXIS1".into(), vec![0i64]),
                    Column::new("NAXIS2".into(), vec![0i64]),
                ],
            },
        };
        let names = process_file_name(file_path.clone());
        let mut columns = meta;
        columns.extend(naxis_cols);
        columns.extend(names);
        DataFrame::new(columns).map_err(FitsLoaderError::PolarsError)
    })();
    result.map_err(|e| {
        FitsLoaderError::FitsError(format!(
            "Error reading metadata for file '{}': {}",
            path_str, e
        ))
    })
}

/// Helper function to combine DataFrames with schema alignment
fn combine_dataframes_with_alignment(
    acc: DataFrame,
    df: DataFrame,
) -> Result<DataFrame, FitsLoaderError> {
    // Try simple vstack first
    match acc.vstack(&df) {
        Ok(combined) => Ok(combined),
        Err(_) => {
            // If vstack fails, align the schemas and try again
            let acc_cols = acc.get_column_names();
            let df_cols = df.get_column_names();

            // Find missing columns in each DataFrame
            let missing_in_acc: Vec<_> = df_cols.iter().filter(|c| !acc_cols.contains(c)).collect();
            let missing_in_df: Vec<_> = acc_cols.iter().filter(|c| !df_cols.contains(c)).collect();

            // Add missing columns to each DataFrame with null values
            let mut acc_aligned = acc.clone();
            let mut df_aligned = df.clone();

            for col in missing_in_acc {
                // Convert to PlSmallStr
                let col_name: PlSmallStr = (*col).clone().into();
                let null_series = Series::new_null(col_name, acc.height());
                let _ = acc_aligned.with_column(null_series).unwrap();
            }

            for col in missing_in_df {
                // Convert to PlSmallStr
                let col_name: PlSmallStr = (*col).clone().into();
                let null_series = Series::new_null(col_name, df.height());
                let _ = df_aligned.with_column(null_series).unwrap();
            }

            // Try again with aligned schemas
            acc_aligned
                .vstack(&df_aligned)
                .map_err(|e| FitsLoaderError::PolarsError(e))
        }
    }
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
    header_items: &Vec<String>,
) -> Result<DataFrame, FitsLoaderError> {
    let dir_path = std::path::PathBuf::from(dir);

    if !dir_path.exists() {
        return Err(FitsLoaderError::FitsError(format!(
            "Directory not found: {}",
            dir
        )));
    }

    // Find all FITS files in the directory
    let entries: Vec<_> = fs::read_dir(dir)
        .map_err(|e| FitsLoaderError::IoError(e))?
        .par_bridge()
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().extension().and_then(|ext| ext.to_str()) == Some("fits"))
        .collect();

    if entries.is_empty() {
        return Err(FitsLoaderError::FitsError(format!(
            "No FITS files found in directory: {}",
            dir
        )));
    }

    // Process each file in parallel, collect results
    let results: Vec<Result<DataFrame, FitsLoaderError>> = entries
        .par_iter()
        .map(|entry| read_fits(entry.path(), &header_items))
        .collect();

    // Filter out errors and keep only successful DataFrames
    let successful_dfs: Vec<DataFrame> = results
        .into_iter()
        .filter_map(|result| result.ok())
        .collect();

    // If no files were successfully processed, return an error
    if successful_dfs.is_empty() {
        return Err(FitsLoaderError::FitsError(
            "None of the files in the directory could be processed successfully".into(),
        ));
    }

    // Combine all successful DataFrames
    let combined_df = successful_dfs
        .into_par_iter()
        .reduce_with(|acc, df| {
            let acc_clone = acc.clone();
            combine_dataframes_with_alignment(acc, df).unwrap_or(acc_clone)
        })
        .ok_or(FitsLoaderError::NoData)?;

    // If there is a column for energy, theta add the q column
    Ok(add_calculated_domains(combined_df.lazy()))
}

pub fn read_multiple_fits_metadata(
    file_paths: Vec<PathBuf>,
    header_items: &Vec<String>,
) -> Result<DataFrame, FitsLoaderError> {
    if file_paths.is_empty() {
        return Err(FitsLoaderError::FitsError("No files provided".into()));
    }
    for path in &file_paths {
        if !path.exists() {
            return Err(FitsLoaderError::FitsError(format!(
                "File not found: {}",
                path.display()
            )));
        }
    }
    let results: Vec<Result<DataFrame, FitsLoaderError>> = file_paths
        .par_iter()
        .map(|path| read_fits_metadata(path.clone(), header_items))
        .collect();
    let successful_dfs: Vec<DataFrame> = results
        .into_iter()
        .filter_map(|r| r.ok())
        .collect();
    if successful_dfs.is_empty() {
        return Err(FitsLoaderError::FitsError(
            "None of the provided files could be read for metadata.".into(),
        ));
    }
    let combined = successful_dfs
        .into_par_iter()
        .reduce_with(|acc, df| match acc.vstack(&df) {
            Ok(c) => c,
            Err(_) => acc,
        })
        .ok_or(FitsLoaderError::NoData)?;
    let lz = combined
        .lazy()
        .sort(["experiment_number", "frame_number"], Default::default());
    Ok(add_calculated_domains(lz))
}

pub fn read_experiment_metadata(
    dir: &str,
    header_items: &Vec<String>,
) -> Result<DataFrame, FitsLoaderError> {
    let dir_path = std::path::PathBuf::from(dir);
    if !dir_path.exists() {
        return Err(FitsLoaderError::FitsError(format!(
            "Directory not found: {}",
            dir
        )));
    }
    let entries: Vec<PathBuf> = fs::read_dir(dir)
        .map_err(FitsLoaderError::IoError)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().and_then(|ext| ext.to_str()) == Some("fits"))
        .map(|e| e.path())
        .collect();
    if entries.is_empty() {
        return Err(FitsLoaderError::FitsError(format!(
            "No FITS files found in directory: {}",
            dir
        )));
    }
    read_multiple_fits_metadata(entries, header_items)
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
    header_items: &Vec<String>,
) -> Result<DataFrame, FitsLoaderError> {
    if file_paths.is_empty() {
        return Err(FitsLoaderError::FitsError("No files provided".into()));
    }

    // Check that all files exist
    for path in &file_paths {
        if !path.exists() {
            return Err(FitsLoaderError::FitsError(format!(
                "File not found: {}",
                path.display()
            )));
        }
    }

    // Process each file in parallel, collect results
    let results: Vec<Result<DataFrame, FitsLoaderError>> = file_paths
        .par_iter()
        .map(|path| read_fits(path.clone(), header_items))
        .collect();

    // Filter out errors and keep only successful DataFrames
    let successful_dfs: Vec<DataFrame> = results
        .into_iter()
        .filter_map(|result| result.ok())
        .collect();

    // If no files were successfully processed, return an error
    if successful_dfs.is_empty() {
        return Err(FitsLoaderError::FitsError(
            "None of the provided files could be processed successfully".into(),
        ));
    }

    // Combine all successful DataFrames
    let combined_df = successful_dfs
        .into_par_iter()
        .reduce_with(|acc, df| {
            let acc_clone = acc.clone();
            combine_dataframes_with_alignment(acc, df).unwrap_or(acc_clone)
        })
        .ok_or(FitsLoaderError::NoData)?;

    Ok(add_calculated_domains(combined_df.lazy()))
}

/// Reads FITS files matching a pattern and combines them into a single DataFrame.
///
/// # Arguments
///
/// * `dir` - Directory containing FITS files
/// * `pattern` - Glob pattern to match files (e.g., "Y6_refl_*.fits")
/// * `header_items` - List of header values to extract
///
/// # Returns
///
/// A `Result` containing either the combined DataFrame or a `FitsLoaderError`.
pub fn read_experiment_pattern(
    dir: &str,
    pattern: &str,
    header_items: &Vec<String>,
) -> Result<DataFrame, FitsLoaderError> {
    let dir_path = std::path::PathBuf::from(dir);

    if !dir_path.exists() {
        return Err(FitsLoaderError::FitsError(format!(
            "Directory not found: {}",
            dir
        )));
    }

    // Clone the header items to avoid borrowing issues
    let header_items = header_items
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();

    // Find all matching FITS files
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

    if entries.is_empty() {
        return Err(FitsLoaderError::FitsError(format!(
            "No FITS files matching pattern '{}' found in directory: {}",
            pattern, dir
        )));
    }

    read_multiple_fits(entries, &header_items)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_from_stems_aggregates_and_skips_invalid() {
        let paths: Vec<PathBuf> = [
            "/x/ZnPc_rt 81041-00001.fits",
            "/x/ZnPc_rt 81041-00002.fits",
            "/x/monlayerjune 81041-00007.fits",
            "/x/invalid_stem.fits",
        ]
        .iter()
        .map(|s| PathBuf::from(s))
        .collect();
        let catalog = catalog_from_stems(&paths);
        assert_eq!(catalog.fits_count, 4);
        assert!(catalog.samples.contains(&"ZnPc".to_string()));
        assert!(catalog.samples.contains(&"monlayerjune".to_string()));
        assert_eq!(catalog.experiment_count, 1);
    }
}

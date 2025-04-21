pub mod enums;
pub mod errors;
pub mod io;
pub mod loader;
pub mod documentation {
    // No code changes needed here
}

// Re-export key types and functions for easier access
pub use enums::{ExperimentType, HeaderValue};
pub use errors::FitsLoaderError;
pub use loader::{read_experiment, read_experiment_pattern, read_fits, read_multiple_fits};

/// Reads a FITS file and converts it to a Polars DataFrame.
///
/// # Arguments
///
/// * `file_path` - Path to the FITS file to read
/// * `experiment_type` - Type of experiment (e.g., `ExperimentType::Xrr`)
///
/// # Returns
///
/// A `Result` containing either the DataFrame or a `FitsLoaderError`.
///
/// # Example
///
/// ```
/// use pyref_core::{read_fits_file, ExperimentType};
/// use std::path::Path;
///
/// let df = read_fits_file("path/to/file.fits", ExperimentType::Xrr);
/// ```
pub fn read_fits_file(
    file_path: impl AsRef<std::path::Path>,
    experiment_type: ExperimentType,
) -> Result<polars::prelude::DataFrame, FitsLoaderError> {
    let path = file_path.as_ref().to_path_buf();
    let header_items = experiment_type.get_keys();
    loader::read_fits(path, &header_items)
}

/// Reads all FITS files in a directory and combines them into a single DataFrame.
///
/// # Arguments
///
/// * `dir_path` - Path to the directory containing FITS files
/// * `experiment_type` - Type of experiment (e.g., `ExperimentType::Xrr`)
///
/// # Returns
///
/// A `Result` containing either the combined DataFrame or a `FitsLoaderError`.
///
/// # Example
///
/// ```
/// use pyref_core::{read_experiment_dir, ExperimentType};
///
/// let df = read_experiment_dir("path/to/directory", ExperimentType::Xrr);
/// ```
pub fn read_experiment_dir(
    dir_path: impl AsRef<str>,
    experiment_type: ExperimentType,
) -> Result<polars::prelude::DataFrame, FitsLoaderError> {
    let header_items = experiment_type.get_keys();
    loader::read_experiment(dir_path.as_ref(), &header_items)
}

/// Reads multiple FITS files and combines them into a single DataFrame.
///
/// # Arguments
///
/// * `file_paths` - Collection of paths to FITS files
/// * `experiment_type` - Type of experiment (e.g., `ExperimentType::Xrr`)
///
/// # Returns
///
/// A `Result` containing either the combined DataFrame or a `FitsLoaderError`.
///
/// # Example
///
/// ```
/// use pyref_core::{read_multiple_fits_files, ExperimentType};
/// use std::path::PathBuf;
///
/// let paths = vec![
///     PathBuf::from("path/to/file1.fits"),
///     PathBuf::from("path/to/file2.fits"),
/// ];
/// let df = read_multiple_fits_files(paths, ExperimentType::Xrr);
/// ```
pub fn read_multiple_fits_files(
    file_paths: Vec<std::path::PathBuf>,
    experiment_type: ExperimentType,
) -> Result<polars::prelude::DataFrame, FitsLoaderError> {
    let header_items = experiment_type.get_keys();
    loader::read_multiple_fits(file_paths, &header_items)
}

/// Reads FITS files matching a pattern in a directory and combines them into a single DataFrame.
///
/// # Arguments
///
/// * `dir_path` - Path to the directory containing FITS files
/// * `pattern` - Glob pattern to match files (e.g., "Y6_refl_*.fits")
/// * `experiment_type` - Type of experiment (e.g., `ExperimentType::Xrr`)
///
/// # Returns
///
/// A `Result` containing either the combined DataFrame or a `FitsLoaderError`.
///
/// # Example
///
/// ```
/// use pyref_core::{read_fits_with_pattern, ExperimentType};
///
/// let df = read_fits_with_pattern("path/to/directory", "Y6_refl_*.fits", ExperimentType::Xrr);
/// ```
pub fn read_fits_with_pattern(
    dir_path: impl AsRef<str>,
    pattern: impl AsRef<str>,
    experiment_type: ExperimentType,
) -> Result<polars::prelude::DataFrame, FitsLoaderError> {
    loader::read_experiment_pattern(dir_path.as_ref(), pattern.as_ref(), experiment_type)
}

/// Reads a FITS file and converts it to a Polars DataFrame.
///
/// # Arguments
/// * `file_path` - Path to the FITS file to read
/// * `header_items` - List of header values to extract
///
/// # Returns
///
/// A `Result` containing either the DataFrame or a `FitsLoaderError`.
///
/// # Example
///
/// ```
/// use pyref_core::{read_fits, loader::ExperimentType};
/// use std::path::Path;
///
/// // Using experiment type
/// let df = read_fits("path/to/file.fits", ExperimentType::Xrr);
///
/// // Using raw header keys
/// let df = read_fits("path/to/file.fits", &["LAMBDA", "THETA", "DATA"]);
/// ```
/// Documentation for read_multiple_fits, read_experiment, and read_experiment_pattern
/// functions is available in the loader module where they are defined.
pub mod errors;
pub mod io;
pub mod loader;
pub mod documentation {
    // No code changes needed here
}

// Re-export key types and functions for easier access
pub use errors::FitsLoaderError;
pub use loader::{read_experiment, read_experiment_pattern, read_fits, read_multiple_fits};

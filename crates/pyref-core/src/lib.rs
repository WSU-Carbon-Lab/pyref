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
/// ```no_run
/// use pyref_core::read_fits;
/// use std::path::PathBuf;
///
/// // Read a FITS file with specific header keys
/// let header_keys = vec!["DATE".to_string(), "Beamline Energy".to_string(), "EXPOSURE".to_string()];
/// let df = read_fits(PathBuf::from("path/to/file.fits"), &header_keys).unwrap();
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

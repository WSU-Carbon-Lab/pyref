use polars::error::PolarsError;
use std::io::Error as IoError;
use thiserror::Error;

/// Custom error type for FITS processing.
#[derive(Error, Debug)]
pub enum FitsLoaderError {
    #[error("FITS processing error: {0}")]
    FitsError(String),

    #[error("Polars error: {0}")]
    PolarsError(#[from] PolarsError),

    #[error("IO error: {0}")]
    IoError(#[from] IoError),

    #[error("Invalid experiment type: {0}")]
    InvalidExperimentType(String),

    #[error("Invalid file name format: {0}")]
    InvalidFileName(String),

    #[error("Missing required header key: {0}")]
    MissingHeaderKey(String),

    #[error("Unsupported image data type")]
    UnsupportedImageData,

    #[error("No data found")]
    NoData,

    #[error("Other error: {0}")]
    Other(String),
}

use polars::error::PolarsError;
use std::error::Error;
use std::fmt;
use std::io::Error as IoError;
use thiserror::Error;

use crate::fits::FitsReadError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoaderErrorKind {
    NotFound,
    PermissionDenied,
    ValidationFailed,
    Temporary,
    Permanent,
}

impl fmt::Display for LoaderErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoaderErrorKind::NotFound => write!(f, "not found"),
            LoaderErrorKind::PermissionDenied => write!(f, "permission denied"),
            LoaderErrorKind::ValidationFailed => write!(f, "validation failed"),
            LoaderErrorKind::Temporary => write!(f, "temporary error"),
            LoaderErrorKind::Permanent => write!(f, "permanent error"),
        }
    }
}

#[derive(Debug)]
pub struct LoaderError {
    pub kind: LoaderErrorKind,
    pub retryable: bool,
    pub message: String,
    pub context: Vec<(String, String)>,
    pub source: Option<Box<dyn Error + Send + Sync>>,
}

impl LoaderError {
    pub fn with_context(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.context.push((key.into(), value.into()));
        self
    }

    pub fn list_fits_io(path: impl fmt::Display, source: std::io::Error) -> Self {
        let kind = match source.kind() {
            std::io::ErrorKind::NotFound => LoaderErrorKind::NotFound,
            std::io::ErrorKind::PermissionDenied => LoaderErrorKind::PermissionDenied,
            _ => LoaderErrorKind::Temporary,
        };
        let retryable = kind == LoaderErrorKind::Temporary;
        Self {
            kind,
            retryable,
            message: source.to_string(),
            context: vec![
                ("operation".into(), "list_fits_in_dir".into()),
                ("path".into(), path.to_string()),
            ],
            source: Some(Box::new(source)),
        }
    }
}

impl fmt::Display for LoaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.kind, self.message)
    }
}

impl Error for LoaderError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.source.as_ref().map(|b| b.as_ref() as &(dyn Error + 'static))
    }
}

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

impl From<FitsReadError> for FitsLoaderError {
    fn from(e: FitsReadError) -> Self {
        FitsLoaderError::FitsError(e.to_string())
    }
}

use std::io;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum FitsReadError {
    #[error("FITS read error: {0}")]
    Io(#[from] io::Error),

    #[error("FITS parse error: {0}")]
    Parse(String),

    #[error("Unsupported FITS feature: {0}")]
    Unsupported(String),
}

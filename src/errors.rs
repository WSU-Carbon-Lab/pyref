use polars::error::PolarsError;
use std::error::Error;
use std::fmt;
use std::io::Error as IoError;

use crate::fits::FitsReadError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FitsErrorKind {
    NotFound,
    ValidationFailed,
    Unsupported,
    Io,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Retryable {
    Permanent,
    Temporary,
}

#[derive(Debug)]
pub struct FitsError {
    pub kind: FitsErrorKind,
    pub retryable: Retryable,
    pub message: String,
    pub context: Vec<(String, String)>,
    pub source: Option<Box<dyn Error + Send + Sync>>,
}

impl FitsError {
    pub fn with_context(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.context.push((key.into(), value.into()));
        self
    }

    pub fn not_found(message: impl Into<String>) -> Self {
        Self {
            kind: FitsErrorKind::NotFound,
            retryable: Retryable::Permanent,
            message: message.into(),
            context: Vec::new(),
            source: None,
        }
    }

    pub fn validation(message: impl Into<String>) -> Self {
        Self {
            kind: FitsErrorKind::ValidationFailed,
            retryable: Retryable::Permanent,
            message: message.into(),
            context: Vec::new(),
            source: None,
        }
    }

    pub fn unsupported(message: impl Into<String>) -> Self {
        Self {
            kind: FitsErrorKind::Unsupported,
            retryable: Retryable::Permanent,
            message: message.into(),
            context: Vec::new(),
            source: None,
        }
    }

    pub fn io(message: impl Into<String>, source: IoError) -> Self {
        Self {
            kind: FitsErrorKind::Io,
            retryable: Retryable::Temporary,
            message: message.into(),
            context: Vec::new(),
            source: Some(Box::new(source)),
        }
    }

    pub fn polars(message: impl Into<String>, source: PolarsError) -> Self {
        Self {
            kind: FitsErrorKind::ValidationFailed,
            retryable: Retryable::Permanent,
            message: message.into(),
            context: Vec::new(),
            source: Some(Box::new(source)),
        }
    }
}

impl fmt::Display for FitsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[FitsError] kind={:?} retryable={:?}", self.kind, self.retryable)?;
        for (k, v) in &self.context {
            write!(f, " {}={}", k, v)?;
        }
        write!(f, " message={}", self.message)?;
        if let Some(src) = &self.source {
            write!(f, " source={}", src)?;
        }
        Ok(())
    }
}

impl Error for FitsError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.source.as_ref().map(|b| b.as_ref() as &(dyn Error + 'static))
    }
}

impl From<IoError> for FitsError {
    fn from(e: IoError) -> Self {
        Self::io("IO error", e)
    }
}

impl From<PolarsError> for FitsError {
    fn from(e: PolarsError) -> Self {
        Self::polars("Polars error", e)
    }
}

impl From<FitsReadError> for FitsError {
    fn from(e: FitsReadError) -> Self {
        let (kind, retryable) = match &e {
            FitsReadError::Io(_) => (FitsErrorKind::Io, Retryable::Temporary),
            FitsReadError::Parse(_) => (FitsErrorKind::ValidationFailed, Retryable::Permanent),
            FitsReadError::Unsupported(_) => (FitsErrorKind::Unsupported, Retryable::Permanent),
        };
        Self {
            kind,
            retryable,
            message: e.to_string(),
            context: Vec::new(),
            source: Some(Box::new(e)),
        }
    }
}

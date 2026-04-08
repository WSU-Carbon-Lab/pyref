//! Worker thread count for parallel FITS reads during ingest (not SQLite writers).
//!
//! Environment (read when both struct fields are unset before resolution):
//!
//! - ``PYREF_INGEST_WORKER_THREADS``: positive integer, clamped to available parallelism.
//! - ``PYREF_INGEST_RESOURCE_FRACTION``: fraction in ``(0.0, 1.0]`` of available parallelism.

use super::{CatalogError, Result};

const ENV_WORKER_THREADS: &str = "PYREF_INGEST_WORKER_THREADS";
const ENV_RESOURCE_FRACTION: &str = "PYREF_INGEST_RESOURCE_FRACTION";

/// Selects how many parallel workers ingest uses for FITS I/O (not SQLite writers).
///
/// Set at most one of ``worker_threads`` or ``resource_fraction``. If both are ``None``,
/// uses ``min(8, max(1, available_parallelism))``.
#[derive(Debug, Clone, Default)]
pub struct IngestParallelism {
    pub worker_threads: Option<usize>,
    pub resource_fraction: Option<f64>,
}

impl IngestParallelism {
    pub fn from_env_optional() -> Self {
        let worker_threads = std::env::var(ENV_WORKER_THREADS)
            .ok()
            .and_then(|s| s.parse().ok());
        let resource_fraction = std::env::var(ENV_RESOURCE_FRACTION)
            .ok()
            .and_then(|s| s.parse().ok());
        Self {
            worker_threads,
            resource_fraction,
        }
    }

    pub fn from_options_or_env(
        worker_threads: Option<usize>,
        resource_fraction: Option<f64>,
    ) -> Self {
        if worker_threads.is_some() || resource_fraction.is_some() {
            Self {
                worker_threads,
                resource_fraction,
            }
        } else {
            Self::from_env_optional()
        }
    }

    /// Returns the resolved worker count after validation.
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError::Validation`] if both ``worker_threads`` and ``resource_fraction``
    /// are set, or if ``resource_fraction`` is not in ``(0.0, 1.0]``.
    pub fn resolve_worker_count(&self) -> Result<usize> {
        if self.worker_threads.is_some() && self.resource_fraction.is_some() {
            return Err(CatalogError::Validation(
                "set only one of worker_threads or resource_fraction".into(),
            ));
        }
        let avail = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        if let Some(n) = self.worker_threads {
            return Ok(n.max(1).min(avail.max(1)));
        }
        if let Some(f) = self.resource_fraction {
            if !(f.is_finite()) || f <= 0.0 || f > 1.0 {
                return Err(CatalogError::Validation(
                    "resource_fraction must be in (0.0, 1.0]".into(),
                ));
            }
            let n = ((avail as f64) * f).floor() as usize;
            return Ok(n.max(1));
        }
        Ok(avail.max(1).min(8))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_both_worker_threads_and_fraction() {
        let p = IngestParallelism {
            worker_threads: Some(2),
            resource_fraction: Some(0.5),
        };
        assert!(p.resolve_worker_count().is_err());
    }

    #[test]
    fn fraction_one_yields_at_least_one_worker() {
        let p = IngestParallelism {
            worker_threads: None,
            resource_fraction: Some(1.0),
        };
        let n = p.resolve_worker_count().unwrap();
        assert!(n >= 1);
    }

    #[test]
    fn rejects_invalid_fraction() {
        let p = IngestParallelism {
            worker_threads: None,
            resource_fraction: Some(0.0),
        };
        assert!(p.resolve_worker_count().is_err());
    }

    #[test]
    fn from_options_or_env_keeps_explicit_threads() {
        let p = IngestParallelism::from_options_or_env(Some(3), None);
        assert_eq!(p.worker_threads, Some(3));
        assert!(p.resource_fraction.is_none());
    }
}

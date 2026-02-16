use std::fmt;
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TuiErrorKind {
    ConfigLoad,
    ConfigSave,
    TerminalSetup,
    TerminalRestore,
    Io,
    IngestFailed,
    CatalogUnavailable,
    WatcherFailed,
}

impl fmt::Display for TuiErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TuiErrorKind::ConfigLoad => write!(f, "config load failed"),
            TuiErrorKind::ConfigSave => write!(f, "config save failed"),
            TuiErrorKind::TerminalSetup => write!(f, "terminal setup failed"),
            TuiErrorKind::TerminalRestore => write!(f, "terminal restore failed"),
            TuiErrorKind::Io => write!(f, "io error"),
            TuiErrorKind::IngestFailed => write!(f, "ingest failed"),
            TuiErrorKind::CatalogUnavailable => write!(f, "catalog unavailable"),
            TuiErrorKind::WatcherFailed => write!(f, "watcher failed"),
        }
    }
}

#[derive(Debug)]
pub struct TuiError {
    pub kind: TuiErrorKind,
    pub retryable: bool,
    pub message: String,
    pub context: Vec<(String, String)>,
    pub source: Option<Box<dyn std::error::Error + Send + Sync>>,
    pub suggestion: Option<String>,
}

impl TuiError {
    pub fn config_load(path: impl AsRef<Path>, source: std::io::Error) -> Self {
        Self {
            kind: TuiErrorKind::ConfigLoad,
            retryable: true,
            message: source.to_string(),
            context: vec![
                ("operation".into(), "load_config".into()),
                ("path".into(), path.as_ref().display().to_string()),
            ],
            source: Some(Box::new(source)),
            suggestion: None,
        }
    }

    pub fn config_parse(path: impl AsRef<Path>, message: String) -> Self {
        Self {
            kind: TuiErrorKind::ConfigLoad,
            retryable: false,
            message,
            context: vec![
                ("operation".into(), "parse_config".into()),
                ("path".into(), path.as_ref().display().to_string()),
            ],
            source: None,
            suggestion: None,
        }
    }

    pub fn config_save(path: impl AsRef<Path>, source: std::io::Error) -> Self {
        let retryable = source.kind() == std::io::ErrorKind::PermissionDenied
            || source.kind() == std::io::ErrorKind::Interrupted;
        Self {
            kind: TuiErrorKind::ConfigSave,
            retryable,
            message: source.to_string(),
            context: vec![
                ("operation".into(), "save_config".into()),
                ("path".into(), path.as_ref().display().to_string()),
            ],
            source: Some(Box::new(source)),
            suggestion: None,
        }
    }

    pub fn config_serialize(path: impl AsRef<Path>, message: String) -> Self {
        Self {
            kind: TuiErrorKind::ConfigSave,
            retryable: false,
            message,
            context: vec![
                ("operation".into(), "serialize_config".into()),
                ("path".into(), path.as_ref().display().to_string()),
            ],
            source: None,
            suggestion: None,
        }
    }

    pub fn terminal_setup(source: std::io::Error) -> Self {
        Self {
            kind: TuiErrorKind::TerminalSetup,
            retryable: true,
            message: source.to_string(),
            context: vec![("operation".into(), "setup_terminal".into())],
            source: Some(Box::new(source)),
            suggestion: None,
        }
    }

    pub fn terminal_restore(source: std::io::Error) -> Self {
        Self {
            kind: TuiErrorKind::TerminalRestore,
            retryable: true,
            message: source.to_string(),
            context: vec![("operation".into(), "restore_terminal".into())],
            source: Some(Box::new(source)),
            suggestion: None,
        }
    }

    pub fn io(operation: &'static str, message: String, source: std::io::Error) -> Self {
        Self {
            kind: TuiErrorKind::Io,
            retryable: source.kind() == std::io::ErrorKind::Interrupted,
            message,
            context: vec![("operation".into(), operation.to_string())],
            source: Some(Box::new(source)),
            suggestion: None,
        }
    }

    pub fn ingest_failed(
        path: impl AsRef<Path>,
        message: String,
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    ) -> Self {
        Self {
            kind: TuiErrorKind::IngestFailed,
            retryable: true,
            message,
            context: vec![
                ("operation".into(), "ingest".into()),
                ("path".into(), path.as_ref().display().to_string()),
            ],
            source,
            suggestion: Some("Press i to re-ingest".into()),
        }
    }

    pub fn catalog_unavailable(
        path: impl AsRef<Path>,
        operation: &'static str,
        message: String,
    ) -> Self {
        Self {
            kind: TuiErrorKind::CatalogUnavailable,
            retryable: true,
            message,
            context: vec![
                ("operation".into(), operation.into()),
                ("path".into(), path.as_ref().display().to_string()),
            ],
            source: None,
            suggestion: Some("Open a directory with catalog or re-ingest".into()),
        }
    }

    pub fn watcher_failed(
        path: impl AsRef<Path>,
        message: String,
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    ) -> Self {
        Self {
            kind: TuiErrorKind::WatcherFailed,
            retryable: true,
            message,
            context: vec![
                ("operation".into(), "watcher".into()),
                ("path".into(), path.as_ref().display().to_string()),
            ],
            source,
            suggestion: Some("Re-ingest (i) or continue without watch".into()),
        }
    }

    #[allow(dead_code)]
    pub fn kind(&self) -> &TuiErrorKind {
        &self.kind
    }

    #[allow(dead_code)]
    pub fn retryable(&self) -> bool {
        self.retryable
    }

    #[allow(dead_code)]
    pub fn context(&self) -> &[(String, String)] {
        &self.context
    }

    pub fn report(&self) -> String {
        let mut out = format!("{}: {}", self.kind, self.message);
        for (k, v) in &self.context {
            out.push_str(&format!(" [{}={}]", k, v));
        }
        if self.retryable {
            out.push_str(" (retryable)");
        }
        if let Some(ref s) = self.source {
            out.push_str(&format!(" (cause: {})", s));
        }
        out
    }
}

impl fmt::Display for TuiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.kind, self.message)?;
        for (k, v) in &self.context {
            write!(f, " [{}={}]", k, v)?;
        }
        if let Some(ref s) = self.source {
            write!(f, " (cause: {})", s)?;
        }
        Ok(())
    }
}

impl std::error::Error for TuiError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source
            .as_ref()
            .map(|b| b.as_ref() as &(dyn std::error::Error + 'static))
    }
}

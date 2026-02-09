use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

use super::error::TuiError;

const RECENT_ROOTS_CAP: usize = 20;

fn default_config_path() -> PathBuf {
    if let Ok(p) = std::env::var("PYREF_TUI_CONFIG") {
        return PathBuf::from(p);
    }
    if let Ok(home) = std::env::var("HOME") {
        let p = PathBuf::from(home).join(".config/pyref/tui.toml");
        return p;
    }
    PathBuf::from(".pyref/tui_metadata.toml")
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuiConfig {
    #[serde(default)]
    pub last_root: Option<String>,
    #[serde(default)]
    pub recent_roots: Vec<String>,
    #[serde(default)]
    pub last_sample: Option<String>,
    #[serde(default)]
    pub last_tag: Option<String>,
    #[serde(default)]
    pub last_experiment: Option<String>,
    #[serde(default = "default_keymap")]
    pub keymap: String,
    #[serde(default)]
    pub poll_interval_ms: Option<u64>,
    #[serde(default = "default_theme")]
    pub theme: String,
    #[serde(default)]
    pub color_scheme: Option<toml::Value>,
    #[serde(default = "default_layout")]
    pub layout: String,
    #[serde(default = "default_keybind_bar_lines")]
    pub keybind_bar_lines: u8,
}

fn default_keymap() -> String {
    "vi".to_string()
}
fn default_theme() -> String {
    "16".to_string()
}
fn default_layout() -> String {
    "balanced".to_string()
}
fn default_keybind_bar_lines() -> u8 {
    2
}

impl Default for TuiConfig {
    fn default() -> Self {
        Self {
            last_root: None,
            recent_roots: Vec::new(),
            last_sample: None,
            last_tag: None,
            last_experiment: None,
            keymap: default_keymap(),
            poll_interval_ms: None,
            theme: default_theme(),
            color_scheme: None,
            layout: default_layout(),
            keybind_bar_lines: default_keybind_bar_lines(),
        }
    }
}

impl TuiConfig {
    pub fn path() -> PathBuf {
        default_config_path()
    }

    pub fn load() -> Result<Self, TuiError> {
        Self::load_from_path(&Self::path())
    }

    pub fn load_from_path(path: &Path) -> Result<Self, TuiError> {
        let s = fs::read_to_string(path).map_err(|e| TuiError::config_load(path, e))?;
        toml::from_str(&s).map_err(|e| TuiError::config_parse(path, e.to_string()))
    }

    pub fn load_or_default() -> Self {
        Self::load().unwrap_or_else(|_| Self::default())
    }

    pub fn save(&self) -> Result<(), TuiError> {
        self.save_to_path(&Self::path())
    }

    pub fn save_to_path(&self, path: &Path) -> Result<(), TuiError> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| TuiError::config_save(path, e))?;
        }
        let s = toml::to_string_pretty(self).map_err(|e| TuiError::config_serialize(path, e.to_string()))?;
        fs::write(path, s).map_err(|e| TuiError::config_save(path, e))
    }

    pub fn set_last_root(&mut self, root: &str) {
        self.last_root = Some(root.to_string());
        self.recent_roots.retain(|r| r != root);
        self.recent_roots.insert(0, root.to_string());
        if self.recent_roots.len() > RECENT_ROOTS_CAP {
            self.recent_roots.truncate(RECENT_ROOTS_CAP);
        }
    }

    pub fn set_last_selection(&mut self, sample: Option<&str>, tag: Option<&str>, experiment: Option<&str>) {
        self.last_sample = sample.map(String::from);
        self.last_tag = tag.map(String::from);
        self.last_experiment = experiment.map(String::from);
    }
}

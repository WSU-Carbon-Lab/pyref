use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use walkdir::WalkDir;

use super::EntryKind;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NasLayout {
    pub default_pattern: Option<String>,
    pub default_depth: u8,
    pub experimentalists: HashMap<String, ExptPolicy>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ExptPolicy {
    Auto,
    Custom { pattern: Option<String>, depth: u8 },
    Ignored,
}

impl NasLayout {
    pub fn default() -> Self {
        NasLayout {
            default_pattern: None,
            default_depth: 2,
            experimentalists: HashMap::new(),
        }
    }

    pub fn load(data_root: &Path) -> Self {
        let layout_path = data_root.join(".pyref").join("layout.toml");
        if layout_path.exists() {
            if let Ok(content) = std::fs::read_to_string(&layout_path) {
                if let Ok(layout) = toml::from_str::<NasLayout>(&content) {
                    return layout;
                }
            }
        }
        Self::default()
    }

    pub fn save(&self, data_root: &Path) -> Result<(), String> {
        let pyref_dir = data_root.join(".pyref");
        std::fs::create_dir_all(&pyref_dir)
            .map_err(|e| format!("Failed to create .pyref directory: {}", e))?;

        let layout_path = pyref_dir.join("layout.toml");
        let content = toml::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialize layout: {}", e))?;

        std::fs::write(&layout_path, content)
            .map_err(|e| format!("Failed to write layout.toml: {}", e))?;

        Ok(())
    }

    pub fn policy_for(&self, experimentalist: &str) -> &ExptPolicy {
        self.experimentalists
            .get(experimentalist)
            .unwrap_or(&ExptPolicy::Auto)
    }

    pub fn set_policy(&mut self, experimentalist: String, policy: ExptPolicy) {
        self.experimentalists.insert(experimentalist, policy);
    }
}

/// Built-in patterns for beamtime detection.
fn get_builtin_patterns() -> Vec<Regex> {
    [
        r"^20\d{2}(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)$",
        r"^20\d{2}-\d{2}(-\d{2})?$",
        r"^20\d{6}$",
        r"(?i)beamtime",
    ]
    .iter()
    .filter_map(|pat| Regex::new(pat).ok())
    .collect()
}

/// Detect if a folder name matches a beamtime pattern.
pub fn matches_beamtime_pattern(name: &str) -> bool {
    get_builtin_patterns().iter().any(|p| p.is_match(name))
}

/// Scan top 2 levels and auto-detect experimentalist policies.
pub fn detect_policy(data_root: &Path) -> NasLayout {
    let mut layout = NasLayout::default();

    // Read depth-1 entries (experimentalists)
    if let Ok(depth1_entries) = std::fs::read_dir(data_root) {
        for entry in depth1_entries.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }

            let name = entry.file_name();
            let name_str = name.to_string_lossy().to_string();

            // Skip dotted entries
            if name_str.starts_with('.') {
                continue;
            }

            // Count depth-2 entries that match beamtime pattern
            let mut total_depth2 = 0;
            let mut matching_depth2 = 0;

            if let Ok(depth2_entries) = std::fs::read_dir(&path) {
                for entry2 in depth2_entries.flatten() {
                    let path2 = entry2.path();
                    if !path2.is_dir() {
                        continue;
                    }

                    let name2 = entry2.file_name();
                    let name2_str = name2.to_string_lossy().to_string();

                    // Skip dotted entries
                    if name2_str.starts_with('.') {
                        continue;
                    }

                    total_depth2 += 1;
                    if matches_beamtime_pattern(&name2_str) {
                        matching_depth2 += 1;
                    }
                }
            }

            // Auto-detect: if ≥50% of depth-2 folders match AND at least 2 exist
            let policy = if total_depth2 >= 2 && matching_depth2 * 2 >= total_depth2 {
                ExptPolicy::Auto
            } else if total_depth2 > 0 {
                ExptPolicy::Custom {
                    pattern: None,
                    depth: 2,
                }
            } else {
                ExptPolicy::Custom {
                    pattern: None,
                    depth: 2,
                }
            };

            layout.set_policy(name_str, policy);
        }
    }

    layout
}

/// Classify an entry based on its path and layout policy.
pub fn classify_entry(path: &Path, data_root: &Path, layout: &NasLayout) -> EntryKind {
    // Data root itself
    if path == data_root {
        return EntryKind::DataRoot;
    }

    // Calculate depth relative to data_root
    let depth = path
        .strip_prefix(data_root)
        .ok()
        .map(|p| p.components().count())
        .unwrap_or(0);

    match depth {
        1 => {
            // Depth 1: experimentalist
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if !name.starts_with('.') {
                EntryKind::Experimentalist
            } else {
                EntryKind::Directory
            }
        }
        2.. => {
            // Depth 2+: check parent experimentalist policy
            if let Some(parent) = path.parent() {
                let parent_name = parent
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("");

                let policy = layout.policy_for(parent_name);
                match policy {
                    ExptPolicy::Auto => {
                        // For Auto policy, check if matches beamtime pattern
                        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                        if matches_beamtime_pattern(name) {
                            EntryKind::Beamtime
                        } else {
                            EntryKind::Directory
                        }
                    }
                    ExptPolicy::Custom {
                        pattern: Some(pat),
                        ..
                    } => {
                        // Custom pattern provided
                        if let Ok(regex) = Regex::new(pat) {
                            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                            if regex.is_match(name) {
                                EntryKind::Beamtime
                            } else {
                                EntryKind::Directory
                            }
                        } else {
                            EntryKind::Directory
                        }
                    }
                    ExptPolicy::Custom { pattern: None, .. } | ExptPolicy::Ignored => {
                        // Unresolved or ignored: treat as generic directory
                        EntryKind::Directory
                    }
                }
            } else {
                EntryKind::Directory
            }
        }
        _ => EntryKind::Directory,
    }
}

/// Check if an experimentalist needs resolution.
pub fn needs_resolution(experimentalist: &str, layout: &NasLayout) -> bool {
    matches!(
        layout.policy_for(experimentalist),
        ExptPolicy::Custom {
            pattern: None,
            ..
        }
    )
}

pub fn scan_experimentalist_subtree(
    expt_path: &Path,
    data_root: &Path,
    layout: &NasLayout,
) -> (u32, u32) {
    let mut beamtime_dirs = 0u32;
    if let Ok(rd) = std::fs::read_dir(expt_path) {
        for e in rd.flatten() {
            if !e.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                continue;
            }
            let p = e.path();
            if classify_entry(&p, data_root, layout) == EntryKind::Beamtime {
                beamtime_dirs = beamtime_dirs.saturating_add(1);
            }
        }
    }
    let mut fits = 0u32;
    for w in WalkDir::new(expt_path)
        .follow_links(false)
        .into_iter()
        .filter_map(|x| x.ok())
    {
        let p = w.path();
        if !p.is_file() {
            continue;
        }
        if p.extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("fits"))
            != Some(true)
        {
            continue;
        }
        let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        if pyref::catalog::is_skippable_stem(stem) {
            continue;
        }
        fits = fits.saturating_add(1);
    }
    (beamtime_dirs, fits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_classify_entry_data_root() {
        let layout = NasLayout::default();
        let data_root = std::path::Path::new("/data");
        assert_eq!(
            classify_entry(data_root, data_root, &layout),
            EntryKind::DataRoot
        );
    }

    #[test]
    fn test_classify_entry_experimentalist() {
        let layout = NasLayout::default();
        let data_root = std::path::Path::new("/data");
        let experimentalist = std::path::Path::new("/data/Smith");
        assert_eq!(
            classify_entry(experimentalist, data_root, &layout),
            EntryKind::Experimentalist
        );
    }

    #[test]
    fn test_classify_entry_beamtime_auto() {
        let mut layout = NasLayout::default();
        layout.set_policy("Smith".to_string(), ExptPolicy::Auto);

        let data_root = std::path::Path::new("/data");
        let beamtime = std::path::Path::new("/data/Smith/2024Jan");
        assert_eq!(
            classify_entry(beamtime, data_root, &layout),
            EntryKind::Beamtime
        );
    }

    #[test]
    fn test_classify_entry_directory() {
        let mut layout = NasLayout::default();
        layout.set_policy("Smith".to_string(), ExptPolicy::Auto);

        let data_root = std::path::Path::new("/data");
        let dir = std::path::Path::new("/data/Smith/scratch");
        assert_eq!(
            classify_entry(dir, data_root, &layout),
            EntryKind::Directory
        );
    }

    #[test]
    fn test_detect_policy_auto() {
        let temp_dir = TempDir::new().unwrap();
        let data_root = temp_dir.path();

        // Create experimentalist
        let expt_dir = data_root.join("Smith");
        fs::create_dir(&expt_dir).unwrap();

        // Create beamtimes
        fs::create_dir(expt_dir.join("2024Jan")).unwrap();
        fs::create_dir(expt_dir.join("2024Feb")).unwrap();

        let layout = detect_policy(data_root);
        match layout.policy_for("Smith") {
            ExptPolicy::Auto => {
                // Expected
            }
            _ => panic!("Expected Auto policy for Smith"),
        }
    }

    #[test]
    fn test_detect_policy_custom() {
        let temp_dir = TempDir::new().unwrap();
        let data_root = temp_dir.path();

        // Create experimentalist
        let expt_dir = data_root.join("Smith");
        fs::create_dir(&expt_dir).unwrap();

        // Create generic directories
        fs::create_dir(expt_dir.join("project1")).unwrap();
        fs::create_dir(expt_dir.join("project2")).unwrap();

        let layout = detect_policy(data_root);
        match layout.policy_for("Smith") {
            ExptPolicy::Custom {
                pattern: None,
                depth: 2,
            } => {
                // Expected
            }
            _ => panic!("Expected Custom policy for Smith"),
        }
    }
}

use std::path::Path;
use std::sync::OnceLock;

static MONTH_REGEX: OnceLock<regex::Regex> = OnceLock::new();

const ALS_DATA_SUFFIX: &str = "BL1101/Data";
const INDEXABLE_EXPERIMENTS: [&str; 2] = ["NEXAFS", "XRR"];
const EXCLUDED_SEGMENT: &str = "Liquid";

fn month_regex() -> &'static regex::Regex {
    MONTH_REGEX.get_or_init(|| {
        regex::Regex::new(r"^20\d{2}(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)$")
            .expect("month regex")
    })
}

pub fn is_under_als_data(path: &Path) -> bool {
    let path_str = path.to_string_lossy();
    let normalized = path_str.replace('\\', "/");
    normalized.contains(ALS_DATA_SUFFIX)
}

pub fn has_month_segment(path: &Path) -> bool {
    path.components().any(|c| {
        c.as_os_str()
            .to_str()
            .map_or(false, |s| month_regex().is_match(s))
    })
}

pub fn is_indexable_experiment_dir(path: &Path) -> bool {
    path.components()
        .last()
        .and_then(|c| c.as_os_str().to_str())
        .map_or(false, |name| INDEXABLE_EXPERIMENTS.contains(&name))
}

pub fn path_contains_excluded(path: &Path) -> bool {
    path.components().any(|c| {
        c.as_os_str()
            .to_str()
            .map_or(false, |s| s == EXCLUDED_SEGMENT)
    })
}

pub fn is_indexable_als_path(path: &Path) -> bool {
    is_under_als_data(path)
        && has_month_segment(path)
        && is_indexable_experiment_dir(path)
        && !path_contains_excluded(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indexable_nexafs_under_als_data() {
        let p = Path::new("/mnt/ALS/BL1101/Data/2025Feb/NEXAFS");
        assert!(is_indexable_als_path(p));
    }

    #[test]
    fn indexable_xrr_under_als_data() {
        let p = Path::new("/mnt/ALS/BL1101/Data/2024Jan/XRR");
        assert!(is_indexable_als_path(p));
    }

    #[test]
    fn liquid_excluded() {
        let p = Path::new("/mnt/ALS/BL1101/Data/2025Feb/Liquid");
        assert!(!is_indexable_als_path(p));
    }

    #[test]
    fn other_experiment_excluded() {
        let p = Path::new("/mnt/ALS/BL1101/Data/2025Feb/Other");
        assert!(!is_indexable_als_path(p));
    }

    #[test]
    fn not_under_als_data() {
        let p = Path::new("/other/2025Feb/NEXAFS");
        assert!(!is_indexable_als_path(p));
    }

    #[test]
    fn no_month_segment() {
        let p = Path::new("/mnt/ALS/BL1101/Data/NEXAFS");
        assert!(!is_indexable_als_path(p));
    }

    #[test]
    fn liquid_in_path_excluded() {
        let p = Path::new("/mnt/ALS/BL1101/Data/2025Feb/Liquid/sub");
        assert!(!is_indexable_als_path(p));
    }
}

pub mod heuristic;
pub mod modal;

use ratatui::widgets::{ListState, ScrollbarState};
use std::path::PathBuf;
use std::time::SystemTime;

use self::heuristic::NasLayout;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntryKind {
    DataRoot,
    Experimentalist,
    Beamtime,
    Directory,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CatalogStatus {
    Indexed,
    Stale,
    NotIndexed,
    NotApplicable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExptResolution {
    Resolved,
    NeedsResolution,
    Ignored,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExplorerSort {
    Name,
    Modified,
    Size,
    FileCount,
}

#[derive(Debug, Clone)]
pub struct DirEntry {
    pub path: PathBuf,
    pub name: String,
    pub kind: EntryKind,
    pub modified: Option<SystemTime>,
    pub size_bytes: Option<u64>,
    pub beamtime_count: Option<u32>,
    pub fits_count: Option<u32>,
    pub catalog_status: CatalogStatus,
    pub expt_resolution: Option<ExptResolution>,
}

pub struct ExplorerState {
    pub data_root: PathBuf,
    pub current_dir: PathBuf,
    pub entries: Vec<DirEntry>,
    pub list_state: ListState,
    pub scroll_state: ScrollbarState,
    pub layout_policy: NasLayout,
    pub loading: bool,
    pub status: Option<String>,
    pub sort_col: ExplorerSort,
    pub sort_asc: bool,
    pub filter_query: Option<String>,
}

impl ExplorerState {
    /// Create a new explorer state for the given data root.
    /// Reads the directory synchronously.
    pub fn new(data_root: PathBuf) -> Self {
        let mut state = ExplorerState {
            data_root: data_root.clone(),
            current_dir: data_root.clone(),
            entries: Vec::new(),
            list_state: ListState::default(),
            scroll_state: ScrollbarState::new(0),
            layout_policy: heuristic::detect_policy(&data_root),
            loading: false,
            status: None,
            sort_col: ExplorerSort::Name,
            sort_asc: true,
            filter_query: None,
        };
        state.load_dir(data_root);
        state
    }

    /// Load directory entries from the given path.
    pub fn load_dir(&mut self, path: PathBuf) {
        self.entries.clear();
        self.current_dir = path.clone();

        match std::fs::read_dir(&path) {
            Ok(entries) => {
                for entry in entries.flatten() {
                    if let Ok(metadata) = entry.metadata() {
                        let path = entry.path();
                        let name = entry.file_name();
                        let name_str = name.to_string_lossy().to_string();

                        // Skip .pyref directory
                        if name_str == ".pyref" {
                            continue;
                        }

                        let kind = heuristic::classify_entry(&path, &self.data_root, &self.layout_policy);

                        // Skip entries starting with dot (hidden files/dirs)
                        if name_str.starts_with('.') {
                            continue;
                        }

                        let expt_resolution = if kind == EntryKind::Experimentalist {
                            if heuristic::needs_resolution(&name_str, &self.layout_policy) {
                                Some(ExptResolution::NeedsResolution)
                            } else {
                                Some(ExptResolution::Resolved)
                            }
                        } else {
                            None
                        };

                        self.entries.push(DirEntry {
                            path,
                            name: name_str,
                            kind,
                            modified: metadata.modified().ok(),
                            size_bytes: None,
                            beamtime_count: None,
                            fits_count: None,
                            catalog_status: match kind {
                                EntryKind::Experimentalist | EntryKind::Beamtime => {
                                    CatalogStatus::NotIndexed
                                }
                                _ => CatalogStatus::NotApplicable,
                            },
                            expt_resolution,
                        });
                    }
                }
            }
            Err(_) => {
                self.status = Some("Failed to read directory".to_string());
            }
        }

        self.apply_sort();
        self.list_state.select(Some(0));
        self.scroll_state = ScrollbarState::new(self.entries.len());
    }

    /// Navigate into the selected directory.
    pub fn navigate_into(&mut self, path: PathBuf) {
        self.load_dir(path);
    }

    /// Navigate up one directory (stop at data_root).
    pub fn navigate_up(&mut self) {
        if self.current_dir != self.data_root {
            if let Some(parent) = self.current_dir.parent() {
                self.load_dir(parent.to_path_buf());
            }
        }
    }

    /// Get the selected entry.
    pub fn selected_entry(&self) -> Option<&DirEntry> {
        self.list_state
            .selected()
            .and_then(|i| self.entries.get(i))
    }

    /// Move selection down.
    pub fn move_down(&mut self) {
        if self.entries.is_empty() {
            return;
        }
        match self.list_state.selected() {
            Some(i) if i < self.entries.len() - 1 => {
                self.list_state.select(Some(i + 1));
            }
            None => {
                self.list_state.select(Some(0));
            }
            _ => {}
        }
        self.update_scroll_state();
    }

    /// Move selection up.
    pub fn move_up(&mut self) {
        match self.list_state.selected() {
            Some(i) if i > 0 => {
                self.list_state.select(Some(i - 1));
            }
            Some(0) => {
                if !self.entries.is_empty() {
                    self.list_state.select(Some(self.entries.len() - 1));
                }
            }
            None if !self.entries.is_empty() => {
                self.list_state.select(Some(0));
            }
            _ => {}
        }
        self.update_scroll_state();
    }

    fn update_scroll_state(&mut self) {
        if let Some(selected) = self.list_state.selected() {
            self.scroll_state = self.scroll_state.position(selected);
        }
    }

    pub fn apply_sort(&mut self) {
        self.entries.sort_by(|a, b| {
            let cmp = match self.sort_col {
                ExplorerSort::Name => a.name.cmp(&b.name),
                ExplorerSort::Modified => {
                    match (a.modified, b.modified) {
                        (Some(t_a), Some(t_b)) => t_a.cmp(&t_b),
                        (Some(_), None) => std::cmp::Ordering::Greater,
                        (None, Some(_)) => std::cmp::Ordering::Less,
                        (None, None) => std::cmp::Ordering::Equal,
                    }
                }
                ExplorerSort::Size => {
                    match (a.size_bytes, b.size_bytes) {
                        (Some(sz_a), Some(sz_b)) => sz_a.cmp(&sz_b),
                        (Some(_), None) => std::cmp::Ordering::Greater,
                        (None, Some(_)) => std::cmp::Ordering::Less,
                        (None, None) => std::cmp::Ordering::Equal,
                    }
                }
                ExplorerSort::FileCount => {
                    match (a.fits_count, b.fits_count) {
                        (Some(cnt_a), Some(cnt_b)) => cnt_a.cmp(&cnt_b),
                        (Some(_), None) => std::cmp::Ordering::Greater,
                        (None, Some(_)) => std::cmp::Ordering::Less,
                        (None, None) => std::cmp::Ordering::Equal,
                    }
                }
            };

            if self.sort_asc {
                cmp
            } else {
                cmp.reverse()
            }
        });
    }

    pub fn toggle_sort(&mut self, col: ExplorerSort) {
        if self.sort_col == col {
            self.sort_asc = !self.sort_asc;
        } else {
            self.sort_col = col;
            self.sort_asc = true;
        }
        self.apply_sort();
    }
}

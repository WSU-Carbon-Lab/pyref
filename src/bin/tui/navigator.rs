use super::catalog_handle::CatalogHandle;
use super::app::{
    AppMode, BeamtimeBodyRects, Focus, GroupedProfileRow, LoadingState, OpenDirFocus,
    WatcherEvent, LauncherState,
};
use std::cmp;
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::Instant;
use ratatui::layout::Rect;
use ratatui::widgets::{ListState, ScrollbarState, TableState};

#[cfg(feature = "watch")]
use pyref::catalog::WatchHandle;

/// All beamtime screen state. Moved out of App in Phase 1.
pub struct BeamtimeState {
    pub all_groups: Vec<GroupedProfileRow>,
    pub filtered_groups: Vec<GroupedProfileRow>,
    pub samples: Vec<String>,
    pub tags: Vec<String>,
    pub scans: Vec<(u32, String)>,
    pub selected_samples: HashSet<String>,
    pub selected_tags: HashSet<String>,
    pub selected_scans: HashSet<u32>,
    pub sample_state: ListState,
    pub tag_state: ListState,
    pub scan_list_state: ListState,
    pub table_state: TableState,
    pub sample_scroll_state: ScrollbarState,
    pub tag_scroll_state: ScrollbarState,
    pub scan_scroll_state: ScrollbarState,
    pub table_scroll_state: ScrollbarState,
    pub expanded_files_scroll_state: ScrollbarState,
    pub table_sort_column: Option<usize>,
    pub table_sort_ordering: Option<cmp::Ordering>,
    pub expanded_table_row: Option<usize>,
    pub expanded_selected_file_index: Option<usize>,
    pub expanded_files_scroll_offset: usize,
    pub expanded_files_sort_column: Option<usize>,
    pub expanded_files_sort_ordering: Option<cmp::Ordering>,
    pub last_expanded_files_visible: usize,
    pub focus: Focus,
    pub mode: AppMode,
    pub current_root: String,
    pub search_query: String,
    pub has_catalog: bool,
    pub loading_state: LoadingState,
    pub spinner_frame: u64,
    pub ingest_rx: Option<mpsc::Receiver<Result<(), String>>>,
    #[cfg(feature = "catalog")]
    pub ingest_progress: Option<(u32, u32)>,
    #[cfg(feature = "catalog")]
    pub ingest_progress_rx: Option<mpsc::Receiver<(u32, u32)>>,
    #[cfg(feature = "catalog")]
    pub pending_ingest_path: Option<PathBuf>,
    pub back_stack: Vec<String>,
    pub forward_stack: Vec<String>,
    pub rename_retag_buffer: String,
    pub status_message: Option<(String, bool)>,
    pub status_message_set_at: Option<Instant>,
    #[cfg(feature = "watch")]
    pub catalog_watcher: Option<WatchHandle>,
    #[cfg(feature = "watch")]
    pub watcher_events_rx: Option<mpsc::Receiver<WatcherEvent>>,
    pub last_body_rects: Option<(Rect, BeamtimeBodyRects)>,
    pub scrollbar_drag: Option<super::app::ScrollbarDragTarget>,
    pub app_error: Option<super::error::TuiError>,
    pub app_warning: Option<super::error::TuiError>,
    #[allow(dead_code)]
    pub keybind_bar_lines: u8,

    // Context fields for Phase 4
    pub data_root: Option<PathBuf>,
    pub experimentalist: Option<String>,
    pub beamtime_path: PathBuf,
    pub cancel_flag: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
}

/// Phase 1 transitional launcher screen state — removed in Phase 4
#[cfg(feature = "catalog")]
#[derive(Debug)]
pub struct LauncherScreenState {
    pub launcher_state: Option<LauncherState>,
    pub open_dir_active: bool,
    pub open_dir_path: String,
    pub open_dir_entries: Vec<PathBuf>,
    pub open_dir_list_state: ListState,
    pub open_dir_focus: OpenDirFocus,
}

/// Screen state enum.
pub enum ScreenState {
    Beamtime(BeamtimeState),
    #[cfg(feature = "catalog")]
    Launcher(LauncherScreenState),
    #[cfg(feature = "tui")]
    Explorer(super::explorer::ExplorerState),
    #[cfg(feature = "tui")]
    ConfigModal(super::explorer::modal::ModalState),
}

/// Navigation stack and catalog handle.
pub struct Navigator {
    pub stack: Vec<ScreenState>,
    pub forward: Vec<ScreenState>,
    pub catalog: CatalogHandle,
}

impl BeamtimeState {
    pub fn set_status(&mut self, msg: String, is_error: bool) {
        self.status_message = Some((msg, is_error));
        self.status_message_set_at = Some(Instant::now());
    }

    pub fn set_app_error(&mut self, e: super::error::TuiError) {
        self.app_error = Some(e);
    }

    pub fn clear_app_error(&mut self) {
        self.app_error = None;
    }
}

impl Navigator {
    pub fn new(beamtime_state: BeamtimeState, catalog: CatalogHandle) -> Self {
        Navigator {
            stack: vec![ScreenState::Beamtime(beamtime_state)],
            forward: vec![],
            catalog,
        }
    }
}

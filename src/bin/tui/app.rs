use chrono::NaiveDateTime;
use fuzzy_matcher::skim::SkimMatcherV2;
use fuzzy_matcher::FuzzyMatcher;
use ratatui::layout::Rect;
use ratatui::widgets::{ListState, TableState};
use std::cmp;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::thread;
use std::time::Instant;

#[cfg(feature = "catalog")]
use pyref::build_fits_stem;
#[cfg(feature = "catalog")]
use pyref::catalog::{
    catalog_file_count, list_beamtime_entries, list_beamtimes, query_files, register_beamtime,
    rename_file_in_catalog, FileRow, CATALOG_DB_NAME, DEFAULT_INGEST_HEADER_ITEMS,
};
#[cfg(feature = "catalog")]
use super::scan_type::{classify_scan_type, ReflectivityScanType};
#[cfg(feature = "watch")]
use pyref::catalog::{run_catalog_watcher, DEFAULT_DEBOUNCE_MS};

#[derive(Debug, Clone)]
pub struct ProfileRow {
    pub sample: String,
    pub tag: String,
    pub energy_str: String,
    pub pol: String,
    pub q_range_str: String,
    pub data_points: u32,
    pub quality_placeholder: String,
    pub experiment_number: u32,
    pub file_path: Option<String>,
}

#[derive(Debug, Clone)]
pub struct GroupedProfileRow {
    pub sample: String,
    pub tag: String,
    pub energy_str: String,
    pub energy_min: Option<f64>,
    pub energy_max: Option<f64>,
    pub scan_type: ReflectivityScanType,
    pub pol_str: String,
    pub theta_min: Option<f64>,
    pub theta_max: Option<f64>,
    pub file_rows: Vec<FileRow>,
    pub experiment_numbers: Vec<u32>,
    pub scan_duration_hm: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Focus {
    Nav,
    SampleList,
    TagList,
    ExperimentList,
    Table,
    SearchBar,
}

const FOCUS_ORDER: [Focus; 6] = [
    Focus::Nav,
    Focus::SampleList,
    Focus::TagList,
    Focus::ExperimentList,
    Focus::Table,
    Focus::SearchBar,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppMode {
    Normal,
    RenameSample,
    EditTag,
    Search,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadingState {
    Idle,
    IndexingDirectory,
    CatalogUpdating,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Screen {
    Launcher,
    Beamtime,
}

#[cfg(feature = "catalog")]
#[derive(Debug)]
pub struct LauncherState {
    pub beamtimes: Vec<(PathBuf, i64)>,
    pub list_state: ListState,
}

#[cfg(feature = "catalog")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenDirFocus {
    PathInput,
    List,
}

#[derive(Debug, Clone)]
pub struct BeamtimeBodyRects {
    pub sample_list: Rect,
    pub tag_list: Rect,
    pub experiment_list: Rect,
    pub table: Rect,
}

pub struct App {
    pub screen: Screen,
    #[cfg(feature = "catalog")]
    pub launcher_state: Option<LauncherState>,
    #[cfg(feature = "catalog")]
    pub open_dir_active: bool,
    #[cfg(feature = "catalog")]
    pub open_dir_path: String,
    #[cfg(feature = "catalog")]
    pub open_dir_entries: Vec<PathBuf>,
    #[cfg(feature = "catalog")]
    pub open_dir_list_state: ListState,
    #[cfg(feature = "catalog")]
    pub open_dir_focus: OpenDirFocus,
    pub all_groups: Vec<GroupedProfileRow>,
    pub filtered_groups: Vec<GroupedProfileRow>,
    pub samples: Vec<String>,
    pub tags: Vec<String>,
    pub experiments: Vec<(u32, String)>,
    pub selected_samples: HashSet<String>,
    pub selected_tags: HashSet<String>,
    pub selected_experiments: HashSet<u32>,
    pub sample_state: ListState,
    pub tag_state: ListState,
    pub experiment_state: ListState,
    pub table_state: TableState,
    pub expanded_table_row: Option<usize>,
    pub expanded_files_scroll_offset: usize,
    pub last_expanded_files_visible: usize,
    pub focus: Focus,
    pub mode: AppMode,
    pub current_root: String,
    pub search_query: String,
    pub needs_redraw: bool,
    pub layout: String,
    pub keymap: String,
    pub keybind_bar_lines: u8,
    pub theme: String,
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
    pub catalog_watcher: Option<pyref::catalog::WatchHandle>,
    #[cfg(feature = "watch")]
    pub watcher_events_rx: Option<mpsc::Receiver<WatcherEvent>>,
    pub last_body_rects: Option<(Rect, BeamtimeBodyRects)>,
    pub app_error: Option<super::error::TuiError>,
    pub app_warning: Option<super::error::TuiError>,
}

#[cfg(feature = "watch")]
#[derive(Debug)]
pub enum WatcherEvent {
    IngestStarted,
    IngestEnded,
}

#[cfg(feature = "catalog")]
fn parse_fits_date(s: &str) -> Option<NaiveDateTime> {
    let s = s.trim();
    NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S").ok().or_else(|| {
        NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S%.3f").ok()
    })
}

#[cfg(feature = "catalog")]
fn format_scan_duration_hm<'a>(date_strings: impl Iterator<Item = &'a str>) -> String {
    let mut min_ts: Option<NaiveDateTime> = None;
    let mut max_ts: Option<NaiveDateTime> = None;
    for s in date_strings {
        if let Some(dt) = parse_fits_date(s) {
            min_ts = Some(match min_ts {
                None => dt,
                Some(m) => cmp::min(m, dt),
            });
            max_ts = Some(match max_ts {
                None => dt,
                Some(m) => cmp::max(m, dt),
            });
        }
    }
    match (min_ts, max_ts) {
        (Some(min_t), Some(max_t)) if max_t >= min_t => {
            let d = max_t.signed_duration_since(min_t);
            let total_mins = d.num_minutes();
            let hours = total_mins / 60;
            let mins = total_mins % 60;
            format!("{}h {:02}m", hours, mins)
        }
        _ => "-".to_string(),
    }
}

const E_TOL_EV: f64 = 0.5;

fn energy_range_str(energy_min: Option<f64>, energy_max: Option<f64>) -> String {
    match (energy_min, energy_max) {
        (Some(min), Some(max)) => {
            if (max - min).abs() < E_TOL_EV {
                format!("{:.1}", min)
            } else {
                format!("{:.1}-{:.1}", min, max)
            }
        }
        _ => "-".to_string(),
    }
}

const E_ROUND_EV: f64 = 0.1;

fn round_energy_ev(e: f64) -> i64 {
    (e / E_ROUND_EV).round() as i64
}

#[cfg(feature = "catalog")]
fn build_groups_from_files(rows: Vec<FileRow>) -> Vec<GroupedProfileRow> {
    let mut key_to_rows: HashMap<(String, String, String, i64), Vec<FileRow>> = HashMap::new();
    for r in rows {
        let sample = r.sample_name.clone();
        let tag = r.tag.clone().unwrap_or_default();
        let pol_str = r.epu_polarization.map(|p| {
            let rounded = p.round() as i32;
            if rounded == 100 {
                "S".to_string()
            } else if rounded == 190 {
                "P".to_string()
            } else {
                format!("{:.2}", p)
            }
        }).unwrap_or_else(|| "-".to_string());
        let key = (sample.clone(), tag.clone(), pol_str.clone(), r.experiment_number);
        key_to_rows.entry(key).or_default().push(r);
    }
    let mut groups: Vec<GroupedProfileRow> = key_to_rows
        .into_iter()
        .flat_map(|((sample, tag, pol_str, experiment_number), file_rows)| {
            let energy_theta_pairs: Vec<(Option<f64>, Option<f64>)> = file_rows
                .iter()
                .map(|r| (r.beamline_energy, r.sample_theta))
                .collect();
            let (scan_type, energy_min, energy_max, theta_min, theta_max) =
                classify_scan_type(&energy_theta_pairs);
            let experiment_numbers = vec![experiment_number as u32];
            if scan_type == ReflectivityScanType::FixedEnergy {
                let mut by_energy: HashMap<i64, Vec<FileRow>> = HashMap::new();
                for r in file_rows {
                    let key = r.beamline_energy.map(round_energy_ev).unwrap_or(0);
                    by_energy.entry(key).or_default().push(r);
                }
                by_energy
                    .into_iter()
                    .map(|(e_bin, sub_rows)| {
                        let thetas: Vec<f64> =
                            sub_rows.iter().filter_map(|r| r.sample_theta).collect();
                        let (theta_min, theta_max) = if thetas.is_empty() {
                            (None, None)
                        } else {
                            let min_t = thetas.iter().cloned().fold(f64::INFINITY, f64::min);
                            let max_t =
                                thetas.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                            (Some(min_t), Some(max_t))
                        };
                        let e_val = e_bin as f64 * E_ROUND_EV;
                        let energy_min = Some(e_val);
                        let energy_max = Some(e_val);
                        let energy_str = format!("{:.1}", e_val);
                        let scan_duration_hm =
                            format_scan_duration_hm(sub_rows.iter().filter_map(|r| r.date_iso.as_deref()));
                        GroupedProfileRow {
                            sample: sample.clone(),
                            tag: tag.clone(),
                            energy_str: energy_str.clone(),
                            energy_min,
                            energy_max,
                            scan_type: ReflectivityScanType::FixedEnergy,
                            pol_str: pol_str.clone(),
                            theta_min,
                            theta_max,
                            file_rows: sub_rows,
                            experiment_numbers: experiment_numbers.clone(),
                            scan_duration_hm,
                        }
                    })
                    .collect()
            } else {
                let energy_str = energy_range_str(energy_min, energy_max);
                let scan_duration_hm =
                    format_scan_duration_hm(file_rows.iter().filter_map(|r| r.date_iso.as_deref()));
                vec![GroupedProfileRow {
                    sample,
                    tag,
                    energy_str,
                    energy_min,
                    energy_max,
                    scan_type,
                    pol_str,
                    theta_min,
                    theta_max,
                    file_rows,
                    experiment_numbers,
                    scan_duration_hm,
                }]
            }
        })
        .collect();
    groups.sort_by(|a, b| {
        a.sample
            .cmp(&b.sample)
            .then(a.tag.cmp(&b.tag))
            .then(a.experiment_numbers.first().cmp(&b.experiment_numbers.first()))
            .then(a.energy_min.partial_cmp(&b.energy_min).unwrap_or(cmp::Ordering::Equal))
            .then(a.pol_str.cmp(&b.pol_str))
    });
    groups
}

fn catalog_to_profiles(current_root: &str) -> Option<(Vec<GroupedProfileRow>, Vec<String>, Vec<String>, Vec<(u32, String)>)> {
    #[cfg(feature = "catalog")]
    {
        let db_path = Path::new(current_root).join(CATALOG_DB_NAME);
        if !db_path.exists() {
            return None;
        }
        let entries = list_beamtime_entries(&db_path).ok()?;
        let rows = query_files(&db_path, None).ok()?;
        let groups = build_groups_from_files(rows);
        let experiments: Vec<(u32, String)> = entries
            .experiments
            .into_iter()
            .map(|(n, s)| (n as u32, s))
            .collect();
        return Some((groups, entries.samples, entries.tags, experiments));
    }
    #[cfg(not(feature = "catalog"))]
    None
}

impl App {
    pub fn new(current_root: String, layout: String, keymap: String, keybind_bar_lines: u8, theme: String) -> Self {
        let (all_groups, samples, tags, experiments, has_catalog) =
            catalog_to_profiles(&current_root).map(|(a, s, t, e)| (a, s, t, e, true)).unwrap_or_else(|| {
                (
                    vec![],
                    vec![],
                    vec![],
                    vec![],
                    false,
                )
            });
        let mut sample_state = ListState::default();
        let mut tag_state = ListState::default();
        let mut experiment_state = ListState::default();
        let mut table_state = TableState::default();
        if !samples.is_empty() {
            sample_state.select(Some(0));
        }
        if !tags.is_empty() {
            tag_state.select(Some(0));
        }
        if !experiments.is_empty() {
            experiment_state.select(Some(0));
        }
        let selected_samples = HashSet::new();
        let selected_tags = HashSet::new();
        let selected_experiments = HashSet::new();
        let all_groups_clone = all_groups.clone();
        let filtered = Self::filter_groups(
            &all_groups,
            &selected_samples,
            &selected_tags,
            &selected_experiments,
            "",
        );
        if !filtered.is_empty() {
            table_state.select(Some(0));
        }
        #[cfg(feature = "watch")]
        let catalog_watcher = Path::new(&current_root)
            .join(CATALOG_DB_NAME)
            .exists()
            .then(|| run_catalog_watcher(Path::new(&current_root), &[], DEFAULT_DEBOUNCE_MS, None, None))
            .and_then(Result::ok);
        App {
            screen: Screen::Beamtime,
            #[cfg(feature = "catalog")]
            launcher_state: None,
            #[cfg(feature = "catalog")]
            open_dir_active: false,
            #[cfg(feature = "catalog")]
            open_dir_path: String::new(),
            #[cfg(feature = "catalog")]
            open_dir_entries: vec![],
            #[cfg(feature = "catalog")]
            open_dir_list_state: ListState::default(),
            #[cfg(feature = "catalog")]
            open_dir_focus: OpenDirFocus::PathInput,
            all_groups: all_groups_clone,
            filtered_groups: filtered,
            samples,
            tags,
            experiments,
            selected_samples,
            selected_tags,
            selected_experiments,
            sample_state,
            tag_state,
            experiment_state,
            table_state,
            expanded_table_row: None,
            expanded_files_scroll_offset: 0,
            last_expanded_files_visible: 5,
            focus: Focus::SampleList,
            mode: AppMode::Normal,
            current_root,
            search_query: String::new(),
            needs_redraw: true,
            layout,
            keymap,
            keybind_bar_lines,
            theme,
            has_catalog,
            loading_state: LoadingState::Idle,
            spinner_frame: 0,
            ingest_rx: None,
            #[cfg(feature = "catalog")]
            ingest_progress: None,
            #[cfg(feature = "catalog")]
            ingest_progress_rx: None,
            #[cfg(feature = "catalog")]
            pending_ingest_path: None,
            back_stack: vec![],
            forward_stack: vec![],
            rename_retag_buffer: String::new(),
            status_message: None,
            status_message_set_at: None,
            #[cfg(feature = "watch")]
            catalog_watcher: catalog_watcher,
            #[cfg(feature = "watch")]
            watcher_events_rx: None,
            last_body_rects: None,
            app_error: None,
            app_warning: None,
        }
    }

    #[cfg(feature = "catalog")]
    pub fn new_for_launcher(layout: String, keymap: String, keybind_bar_lines: u8, theme: String) -> Self {
        let beamtimes = list_beamtimes().unwrap_or_default();
        let mut list_state = ListState::default();
        if !beamtimes.is_empty() {
            list_state.select(Some(0));
        }
        let launcher_state = LauncherState { beamtimes, list_state };
        App {
            screen: Screen::Launcher,
            launcher_state: Some(launcher_state),
            open_dir_active: false,
            open_dir_path: String::new(),
            open_dir_entries: vec![],
            open_dir_list_state: ListState::default(),
            open_dir_focus: OpenDirFocus::PathInput,
            all_groups: vec![],
            filtered_groups: vec![],
            samples: vec![],
            tags: vec![],
            experiments: vec![],
            selected_samples: HashSet::new(),
            selected_tags: HashSet::new(),
            selected_experiments: HashSet::new(),
            sample_state: ListState::default(),
            tag_state: ListState::default(),
            experiment_state: ListState::default(),
            table_state: TableState::default(),
            expanded_table_row: None,
            expanded_files_scroll_offset: 0,
            last_expanded_files_visible: 5,
            focus: Focus::SampleList,
            mode: AppMode::Normal,
            current_root: String::new(),
            search_query: String::new(),
            needs_redraw: true,
            layout,
            keymap,
            keybind_bar_lines,
            theme,
            has_catalog: false,
            loading_state: LoadingState::Idle,
            spinner_frame: 0,
            ingest_rx: None,
            ingest_progress: None,
            ingest_progress_rx: None,
            pending_ingest_path: None,
            back_stack: vec![],
            forward_stack: vec![],
            rename_retag_buffer: String::new(),
            status_message: None,
            status_message_set_at: None,
            #[cfg(feature = "watch")]
            catalog_watcher: None,
            #[cfg(feature = "watch")]
            watcher_events_rx: None,
            last_body_rects: None,
            app_error: None,
            app_warning: None,
        }
    }

    pub fn set_app_error(&mut self, e: super::error::TuiError) {
        self.app_error = Some(e);
        self.needs_redraw = true;
    }

    pub fn clear_app_error(&mut self) {
        self.app_error = None;
        self.needs_redraw = true;
    }

    pub fn set_app_warning(&mut self, e: super::error::TuiError) {
        self.app_warning = Some(e);
        self.needs_redraw = true;
    }

    pub fn clear_app_warning(&mut self) {
        self.app_warning = None;
        self.needs_redraw = true;
    }

    pub fn set_status(&mut self, msg: String, is_error: bool) {
        self.status_message = Some((msg, is_error));
        self.status_message_set_at = Some(Instant::now());
    }

    pub fn clear_status_if_stale(&mut self) {
        const STATUS_TTL_SECS: u64 = 3;
        if let (Some(_), Some(at)) = (self.status_message.as_ref(), self.status_message_set_at) {
            if at.elapsed().as_secs() >= STATUS_TTL_SECS {
                self.status_message = None;
                self.status_message_set_at = None;
                self.needs_redraw = true;
            }
        }
    }

    #[cfg(feature = "catalog")]
    pub fn launcher_list_down(&mut self) {
        if let Some(ref mut s) = self.launcher_state {
            let n = s.beamtimes.len();
            if n == 0 {
                return;
            }
            let next = s
                .list_state
                .selected()
                .map(|i| (i + 1).min(n.saturating_sub(1)))
                .unwrap_or(0);
            s.list_state.select(Some(next));
        }
    }

    #[cfg(feature = "catalog")]
    pub fn launcher_list_up(&mut self) {
        if let Some(ref mut s) = self.launcher_state {
            let n = s.beamtimes.len();
            if n == 0 {
                return;
            }
            let next = s
                .list_state
                .selected()
                .map(|i| i.saturating_sub(1))
                .unwrap_or(0);
            s.list_state.select(Some(next));
        }
    }

    #[cfg(feature = "catalog")]
    pub fn go_to_launcher(&mut self) {
        let beamtimes = list_beamtimes().unwrap_or_default();
        let mut list_state = ListState::default();
        if !beamtimes.is_empty() {
            list_state.select(Some(0));
        }
        self.launcher_state = Some(LauncherState { beamtimes, list_state });
        self.screen = Screen::Launcher;
        #[cfg(feature = "watch")]
        {
            self.catalog_watcher = None;
            self.watcher_events_rx = None;
        }
        self.needs_redraw = true;
    }

    #[cfg(feature = "catalog")]
    pub fn launcher_open_selected(&mut self) {
        if let Some(ref mut state) = self.launcher_state {
            if let Some(i) = state.list_state.selected() {
                if let Some((path, _)) = state.beamtimes.get(i) {
                    let root = path.to_string_lossy().into_owned();
                    if self.set_root(root) {
                        self.screen = Screen::Beamtime;
                        self.launcher_state = None;
                    }
                }
            }
        }
    }

    #[cfg(feature = "catalog")]
    pub fn launcher_open_directory(&mut self) {
        self.open_dir_active = true;
        self.open_dir_path = std::env::var("HOME").unwrap_or_else(|_| "/".to_string());
        self.open_dir_focus = OpenDirFocus::List;
        self.refresh_open_dir_entries();
    }

    #[cfg(feature = "catalog")]
    pub fn refresh_open_dir_entries(&mut self) {
        self.open_dir_entries.clear();
        let path = Path::new(&self.open_dir_path);
        if path.exists() && path.is_dir() {
            if let Ok(rd) = std::fs::read_dir(path) {
                let mut dirs: Vec<PathBuf> = rd
                    .filter_map(|e| e.ok())
                    .filter(|e| e.path().is_dir())
                    .map(|e| e.path())
                    .collect();
                dirs.sort_by(|a, b| a.file_name().cmp(&b.file_name()));
                self.open_dir_entries = dirs;
            }
        }
        self.open_dir_list_state = ListState::default();
        if !self.open_dir_entries.is_empty() || path.parent().is_some() {
            self.open_dir_list_state.select(Some(0));
        }
        self.needs_redraw = true;
    }

    #[cfg(feature = "catalog")]
    pub fn open_dir_cancel(&mut self) {
        self.open_dir_active = false;
        self.needs_redraw = true;
    }

    #[cfg(feature = "catalog")]
    pub fn open_dir_list_enter(&mut self) {
        let n = 1 + self.open_dir_entries.len();
        if n == 0 {
            return;
        }
        let i = self.open_dir_list_state.selected().unwrap_or(0);
        if i == 0 {
            if let Some(parent) = Path::new(&self.open_dir_path).parent() {
                self.open_dir_path = parent.to_string_lossy().into_owned();
                self.refresh_open_dir_entries();
            }
        } else if let Some(p) = self.open_dir_entries.get(i - 1) {
            let name = p
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("");
            if name.is_empty() {
                return;
            }
            let sep = if self.open_dir_path.ends_with('/') { "" } else { "/" };
            self.open_dir_path = format!("{}{}{}", self.open_dir_path, sep, name);
            self.refresh_open_dir_entries();
        }
    }

    #[cfg(feature = "catalog")]
    pub fn open_dir_list_down(&mut self) {
        let n = 1 + self.open_dir_entries.len();
        if n == 0 {
            return;
        }
        let next = self
            .open_dir_list_state
            .selected()
            .map(|i| (i + 1).min(n.saturating_sub(1)))
            .unwrap_or(0);
        self.open_dir_list_state.select(Some(next));
        self.needs_redraw = true;
    }

    #[cfg(feature = "catalog")]
    pub fn open_dir_list_up(&mut self) {
        let n = 1 + self.open_dir_entries.len();
        if n == 0 {
            return;
        }
        let next = self
            .open_dir_list_state
            .selected()
            .map(|i| i.saturating_sub(1))
            .unwrap_or(0);
        self.open_dir_list_state.select(Some(next));
        self.needs_redraw = true;
    }

    #[cfg(feature = "catalog")]
    pub fn open_dir_focus_toggle(&mut self) {
        self.open_dir_focus = match self.open_dir_focus {
            OpenDirFocus::PathInput => {
                self.refresh_open_dir_entries();
                OpenDirFocus::List
            }
            OpenDirFocus::List => OpenDirFocus::PathInput,
        };
        self.needs_redraw = true;
    }

    #[cfg(feature = "catalog")]
    pub fn open_dir_path_push(&mut self, c: char) {
        self.open_dir_path.push(c);
        self.needs_redraw = true;
    }

    #[cfg(feature = "catalog")]
    pub fn open_dir_path_pop(&mut self) {
        self.open_dir_path.pop();
        self.needs_redraw = true;
    }

    #[cfg(feature = "catalog")]
    pub fn open_dir_autocomplete(&mut self) {
        let path = Path::new(&self.open_dir_path);
        let (parent, prefix) = if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            let parent = path.parent().unwrap_or(Path::new("/"));
            (parent, name.to_string())
        } else {
            return;
        };
        if !parent.exists() || !parent.is_dir() {
            return;
        }
        let Ok(rd) = std::fs::read_dir(parent) else {
            return;
        };
        let mut matches: Vec<PathBuf> = rd
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_dir())
            .filter(|e| {
                e.path()
                    .file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with(&prefix))
                    .unwrap_or(false)
            })
            .map(|e| e.path())
            .collect();
        matches.sort_by(|a, b| a.file_name().cmp(&b.file_name()));
        if let Some(first) = matches.first() {
            let name = first
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("");
            let parent_str = parent.to_string_lossy();
            let sep = if parent_str.ends_with('/') { "" } else { "/" };
            self.open_dir_path = format!("{}{}{}", parent_str, sep, name);
            self.refresh_open_dir_entries();
            self.needs_redraw = true;
        }
    }

    #[cfg(feature = "catalog")]
    pub fn open_dir_confirm(&mut self) -> bool {
        let path = Path::new(&self.open_dir_path);
        if !path.exists() || !path.is_dir() {
            self.set_status("Path does not exist or is not a directory.".to_string(), true);
            return false;
        }
        let canonical = match path.canonicalize() {
            Ok(p) => p,
            Err(_) => {
                self.set_status("Failed to canonicalize path.".to_string(), true);
                return false;
            }
        };
        self.open_dir_active = false;
        let path_for_thread = canonical.clone();
        self.pending_ingest_path = Some(canonical);
        let header_items: Vec<String> = DEFAULT_INGEST_HEADER_ITEMS
            .iter()
            .map(|s| (*s).to_string())
            .collect();
        let (tx, rx) = mpsc::channel();
        let (progress_tx, progress_rx) = mpsc::channel();
        thread::spawn(move || {
            let result = pyref::catalog::ingest_beamtime(
                path_for_thread.as_path(),
                &header_items,
                false,
                Some(progress_tx),
            )
            .map(|_| ())
            .map_err(|e| e.to_string());
            let _ = tx.send(result);
        });
        self.loading_state = LoadingState::IndexingDirectory;
        self.ingest_rx = Some(rx);
        self.ingest_progress = None;
        self.ingest_progress_rx = Some(progress_rx);
        self.needs_redraw = true;
        true
    }

    #[cfg(feature = "catalog")]
    pub fn start_indexing_at(&mut self, path: &Path) {
        if self.loading_state != LoadingState::Idle {
            return;
        }
        if !path.exists() || !path.is_dir() {
            return;
        }
        let path_buf = path.to_path_buf();
        let path_for_thread = path_buf.clone();
        let (tx, rx) = mpsc::channel();
        let (progress_tx, progress_rx) = mpsc::channel();
        let header_items: Vec<String> = DEFAULT_INGEST_HEADER_ITEMS
            .iter()
            .map(|s| (*s).to_string())
            .collect();
        thread::spawn(move || {
            let result = pyref::catalog::ingest_beamtime(
                path_for_thread.as_path(),
                &header_items,
                false,
                Some(progress_tx),
            )
            .map(|_| ())
            .map_err(|e| e.to_string());
            let _ = tx.send(result);
        });
        self.pending_ingest_path = Some(path_buf);
        self.loading_state = LoadingState::IndexingDirectory;
        self.ingest_rx = Some(rx);
        self.ingest_progress = None;
        self.ingest_progress_rx = Some(progress_rx);
    }

    pub fn set_root(&mut self, new_root: String) -> bool {
        let path = Path::new(&new_root);
        if !path.exists() || !path.is_dir() {
            return false;
        }
        let canonical = match path.canonicalize() {
            Ok(p) => p,
            Err(_) => return false,
        };
        let new_root_str = canonical.to_string_lossy().into_owned();
        #[cfg(feature = "watch")]
        {
            self.catalog_watcher = None;
            self.watcher_events_rx = None;
        }
        self.current_root = new_root_str.clone();
        if let Some((all_groups, samples, tags, experiments)) = catalog_to_profiles(&self.current_root) {
            self.all_groups = all_groups.clone();
            self.samples = samples.clone();
            self.tags = tags.clone();
            self.experiments = experiments.clone();
            self.has_catalog = true;
            self.sample_state = ListState::default();
            self.tag_state = ListState::default();
            self.experiment_state = ListState::default();
            self.table_state = TableState::default();
            if !self.samples.is_empty() {
                self.sample_state.select(Some(0));
            }
            if !self.tags.is_empty() {
                self.tag_state.select(Some(0));
            }
            if !self.experiments.is_empty() {
                self.experiment_state.select(Some(0));
            }
            self.refresh_filtered();
            #[cfg(feature = "watch")]
            if Path::new(&self.current_root).join(CATALOG_DB_NAME).exists() {
                let (tx, rx) = mpsc::channel();
                let tx_end = tx.clone();
                let on_start = Box::new(move || {
                    let _ = tx.send(WatcherEvent::IngestStarted);
                });
                let on_end = Box::new(move || {
                    let _ = tx_end.send(WatcherEvent::IngestEnded);
                });
                self.catalog_watcher = run_catalog_watcher(
                    Path::new(&self.current_root),
                    &[],
                    DEFAULT_DEBOUNCE_MS,
                    Some(on_start),
                    Some(on_end),
                )
                .ok();
                self.watcher_events_rx = Some(rx);
            }
        } else {
            self.all_groups = vec![];
            self.filtered_groups = vec![];
            self.samples = vec![];
            self.tags = vec![];
            self.experiments = vec![];
            self.has_catalog = false;
            self.sample_state = ListState::default();
            self.tag_state = ListState::default();
            self.experiment_state = ListState::default();
            self.table_state = TableState::default();
        }
        self.needs_redraw = true;
        true
    }

    pub fn nav_up(&mut self) {
        let path = Path::new(&self.current_root);
        if let Some(parent) = path.parent() {
            let parent_str = parent.to_string_lossy().into_owned();
            if parent_str != self.current_root {
                self.back_stack.push(self.current_root.clone());
                self.forward_stack.clear();
                if !self.set_root(parent_str) {
                    self.set_status("Path not found or not a directory.".to_string(), true);
                }
            }
        }
    }

    pub fn nav_back(&mut self) {
        if let Some(prev) = self.back_stack.pop() {
            self.forward_stack.push(self.current_root.clone());
            if !self.set_root(prev) {
                self.set_status("Path not found or not a directory.".to_string(), true);
            }
        }
    }

    pub fn nav_fwd(&mut self) {
        if let Some(next) = self.forward_stack.pop() {
            self.back_stack.push(self.current_root.clone());
            if !self.set_root(next) {
                self.set_status("Path not found or not a directory.".to_string(), true);
            }
        }
    }

    fn filter_groups(
        groups: &[GroupedProfileRow],
        samples: &HashSet<String>,
        tags: &HashSet<String>,
        experiments: &HashSet<u32>,
        search: &str,
    ) -> Vec<GroupedProfileRow> {
        let search_lower = search.to_lowercase();
        let matcher = SkimMatcherV2::default();
        groups
            .iter()
            .filter(|r| {
                let sample_ok = samples.is_empty() || samples.contains(&r.sample);
                let tag_ok = tags.is_empty() || tags.contains(&r.tag);
                let exp_ok = experiments.is_empty()
                    || r.experiment_numbers.iter().any(|e| experiments.contains(e));
                let search_ok = if search_lower.is_empty() {
                    true
                } else {
                    let theta_str = match (r.theta_min, r.theta_max) {
                        (Some(mn), Some(mx)) => format!("{:.2} {:.2}", mn, mx),
                        _ => "-".to_string(),
                    };
                    let e_range_str = match (r.energy_min, r.energy_max) {
                        (Some(mn), Some(mx)) => format!("{:.1} {:.1}", mn, mx),
                        (Some(e), None) | (None, Some(e)) => format!("{:.1}", e),
                        _ => "-".to_string(),
                    };
                    let haystack = format!(
                        "{} {} {} {} {} {}",
                        r.sample,
                        r.tag,
                        r.energy_str,
                        e_range_str,
                        r.pol_str,
                        theta_str
                    )
                    .to_lowercase();
                    matcher.fuzzy_match(&haystack, &search_lower).is_some()
                };
                sample_ok && tag_ok && exp_ok && search_ok
            })
            .cloned()
            .collect()
    }

    pub fn focused_sample(&self) -> Option<String> {
        self.sample_state
            .selected()
            .and_then(|i| self.samples.get(i).cloned())
    }

    pub fn focused_tag(&self) -> Option<String> {
        self.tag_state
            .selected()
            .and_then(|i| self.tags.get(i).cloned())
    }

    pub fn focused_experiment(&self) -> Option<u32> {
        self.experiment_state
            .selected()
            .and_then(|i| self.experiments.get(i).map(|(n, _)| *n))
    }

    pub fn refresh_filtered(&mut self) {
        self.filtered_groups = Self::filter_groups(
            &self.all_groups,
            &self.selected_samples,
            &self.selected_tags,
            &self.selected_experiments,
            &self.search_query,
        );
        self.clamp_selections();
        self.table_state.select(None);
        self.expanded_table_row = None;
        if !self.filtered_groups.is_empty() {
            self.table_state.select(Some(0));
        }
        self.needs_redraw = true;
    }

    pub fn reload_from_catalog(&mut self) {
        #[cfg(feature = "catalog")]
        {
            let db_path = Path::new(&self.current_root).join(CATALOG_DB_NAME);
            if !db_path.exists() {
                return;
            }
            let entries = match list_beamtime_entries(&db_path) {
                Ok(e) => e,
                Err(e) => {
                    self.set_app_error(super::error::TuiError::catalog_unavailable(
                        &self.current_root,
                        "list_beamtime_entries",
                        e.to_string(),
                    ));
                    return;
                }
            };
            let rows = match query_files(&db_path, None) {
                Ok(r) => r,
                Err(e) => {
                    self.set_app_error(super::error::TuiError::catalog_unavailable(
                        &self.current_root,
                        "query_files",
                        e.to_string(),
                    ));
                    return;
                }
            };
            let all_groups = build_groups_from_files(rows);
            let experiments: Vec<(u32, String)> = entries
                .experiments
                .into_iter()
                .map(|(n, s)| (n as u32, s))
                .collect();
            self.app_error = None;
            self.all_groups = all_groups.clone();
            self.samples = entries.samples.clone();
            self.tags = entries.tags.clone();
            self.experiments = experiments.clone();
            self.has_catalog = true;
            self.sample_state = ListState::default();
            self.tag_state = ListState::default();
            self.experiment_state = ListState::default();
            self.table_state = TableState::default();
            if !self.samples.is_empty() {
                self.sample_state.select(Some(0));
            }
            if !self.tags.is_empty() {
                self.tag_state.select(Some(0));
            }
            if !self.experiments.is_empty() {
                self.experiment_state.select(Some(0));
            }
            self.refresh_filtered();
            #[cfg(feature = "watch")]
            {
                self.catalog_watcher = None;
                self.watcher_events_rx = None;
                if db_path.exists() {
                    let (tx, rx) = mpsc::channel();
                    let tx_end = tx.clone();
                    let on_start = Box::new(move || {
                        let _ = tx.send(WatcherEvent::IngestStarted);
                    });
                    let on_end = Box::new(move || {
                        let _ = tx_end.send(WatcherEvent::IngestEnded);
                    });
                    match run_catalog_watcher(
                        Path::new(&self.current_root),
                        &[],
                        DEFAULT_DEBOUNCE_MS,
                        Some(on_start),
                        Some(on_end),
                    ) {
                        Ok(handle) => {
                            self.catalog_watcher = Some(handle);
                            self.watcher_events_rx = Some(rx);
                        }
                        Err(e) => {
                            self.set_app_warning(super::error::TuiError::watcher_failed(
                                &self.current_root,
                                e.to_string(),
                                Some(Box::new(e)),
                            ));
                        }
                    }
                }
            }
        }
    }

    #[cfg(feature = "catalog")]
    pub fn start_indexing(&mut self) {
        if self.loading_state != LoadingState::Idle {
            return;
        }
        let root = Path::new(&self.current_root);
        if !root.exists() || !root.is_dir() {
            return;
        }
        let root_str = self.current_root.clone();
        let header_items: Vec<String> = DEFAULT_INGEST_HEADER_ITEMS
            .iter()
            .map(|s| (*s).to_string())
            .collect();
        let (tx, rx) = mpsc::channel();
        let (progress_tx, progress_rx) = mpsc::channel();
        thread::spawn(move || {
            let result = pyref::catalog::ingest_beamtime(
                Path::new(&root_str),
                &header_items,
                false,
                Some(progress_tx),
            )
            .map(|_| ())
            .map_err(|e| e.to_string());
            let _ = tx.send(result);
        });
        self.loading_state = LoadingState::IndexingDirectory;
        self.ingest_rx = Some(rx);
        self.ingest_progress = None;
        self.ingest_progress_rx = Some(progress_rx);
    }

    #[cfg(not(feature = "catalog"))]
    pub fn start_indexing(&mut self) {}

    #[cfg(feature = "catalog")]
    pub fn try_recv_ingest(&mut self) {
        if let Some(ref progress_rx) = self.ingest_progress_rx {
            while let Ok((cur, tot)) = progress_rx.try_recv() {
                self.ingest_progress = Some((cur, tot));
            }
        }
        if let Some(ref rx) = self.ingest_rx {
            if let Ok(result) = rx.try_recv() {
                self.ingest_rx = None;
                self.ingest_progress_rx = None;
                self.ingest_progress = None;
                self.loading_state = LoadingState::Idle;
                match result {
                    Ok(()) => {
                        self.app_error = None;
                        if let Some(path) = self.pending_ingest_path.take() {
                            let db_path = path.join(CATALOG_DB_NAME);
                            let file_count = catalog_file_count(&db_path).ok();
                            let _ = register_beamtime(&path, file_count);
                            if self.screen == Screen::Launcher {
                                let beamtimes = list_beamtimes().unwrap_or_default();
                                let mut list_state = ListState::default();
                                if !beamtimes.is_empty() {
                                    list_state.select(Some(0));
                                }
                                self.launcher_state = Some(LauncherState { beamtimes, list_state });
                            }
                            let root_str = path.to_string_lossy().into_owned();
                            if self.set_root(root_str) {
                                self.screen = Screen::Beamtime;
                                self.launcher_state = None;
                            }
                            self.set_status("Catalog indexed.".to_string(), false);
                        } else {
                            self.reload_from_catalog();
                            let db_path = Path::new(&self.current_root).join(CATALOG_DB_NAME);
                            let file_count = catalog_file_count(&db_path).ok();
                            let _ = register_beamtime(Path::new(&self.current_root), file_count);
                            self.set_status("Catalog indexed.".to_string(), false);
                        }
                    }
                    Err(e) => {
                        let path = self.pending_ingest_path.take().unwrap_or_else(|| PathBuf::from(&self.current_root));
                        self.set_app_error(super::error::TuiError::index_failed(path, e, None));
                    }
                }
                self.needs_redraw = true;
            }
        }
    }

    #[cfg(not(feature = "catalog"))]
    pub fn try_recv_ingest(&mut self) {}

    #[cfg(feature = "watch")]
    pub fn try_recv_watcher(&mut self) {
        let mut events = vec![];
        if let Some(ref rx) = self.watcher_events_rx {
            while let Ok(ev) = rx.try_recv() {
                events.push(ev);
            }
        }
        for ev in events {
            match ev {
                WatcherEvent::IngestStarted => {
                    self.loading_state = LoadingState::CatalogUpdating;
                }
                WatcherEvent::IngestEnded => {
                    self.loading_state = LoadingState::Idle;
                    self.reload_from_catalog();
                }
            }
            self.needs_redraw = true;
        }
    }

    #[cfg(not(feature = "watch"))]
    pub fn try_recv_watcher(&mut self) {}

    fn clamp_selections(&mut self) {
        let n = self.samples.len();
        if let Some(i) = self.sample_state.selected() {
            if i >= n && n > 0 {
                self.sample_state.select(Some(n - 1));
            } else if n == 0 {
                self.sample_state.select(None);
            }
        }
        let n = self.tags.len();
        if let Some(i) = self.tag_state.selected() {
            if i >= n && n > 0 {
                self.tag_state.select(Some(n - 1));
            } else if n == 0 {
                self.tag_state.select(None);
            }
        }
        let n = self.experiments.len();
        if let Some(i) = self.experiment_state.selected() {
            if i >= n && n > 0 {
                self.experiment_state.select(Some(n - 1));
            } else if n == 0 {
                self.experiment_state.select(None);
            }
        }
        let n = self.filtered_groups.len();
        if let Some(i) = self.table_state.selected() {
            if i >= n && n > 0 {
                self.table_state.select(Some(n - 1));
            } else if n == 0 {
                self.table_state.select(None);
            }
        }
    }

    pub fn focus_next(&mut self) {
        let idx = FOCUS_ORDER.iter().position(|&f| f == self.focus).unwrap_or(0);
        self.focus = FOCUS_ORDER[(idx + 1) % FOCUS_ORDER.len()];
    }

    pub fn focus_prev(&mut self) {
        let idx = FOCUS_ORDER.iter().position(|&f| f == self.focus).unwrap_or(0);
        self.focus = FOCUS_ORDER[(idx + FOCUS_ORDER.len() - 1) % FOCUS_ORDER.len()];
    }

    pub fn focus_sample(&mut self) {
        self.focus = Focus::SampleList;
    }

    pub fn focus_tag(&mut self) {
        self.focus = Focus::TagList;
    }

    pub fn focus_experiment(&mut self) {
        self.focus = Focus::ExperimentList;
    }

    pub fn focus_browser(&mut self) {
        self.focus = Focus::Table;
    }

    pub fn list_down(&mut self) {
        match self.focus {
            Focus::SampleList => {
                let len = self.samples.len();
                if len > 0 {
                    let i = self.sample_state.selected().unwrap_or(0);
                    self.sample_state.select(Some((i + 1) % len));
                }
            }
            Focus::TagList => {
                let len = self.tags.len();
                if len > 0 {
                    let i = self.tag_state.selected().unwrap_or(0);
                    self.tag_state.select(Some((i + 1) % len));
                }
            }
            Focus::ExperimentList => {
                let len = self.experiments.len();
                if len > 0 {
                    let i = self.experiment_state.selected().unwrap_or(0);
                    self.experiment_state.select(Some((i + 1) % len));
                }
            }
            Focus::Table => {
                if let Some(idx) = self.expanded_table_row {
                    if idx < self.filtered_groups.len() {
                        let file_count = self.filtered_groups[idx].file_rows.len();
                        let max_offset = file_count.saturating_sub(self.last_expanded_files_visible.max(1));
                        self.expanded_files_scroll_offset =
                            cmp::min(self.expanded_files_scroll_offset + 1, max_offset);
                        self.needs_redraw = true;
                        return;
                    }
                }
                let len = self.filtered_groups.len();
                if len > 0 {
                    let i = self.table_state.selected().unwrap_or(0);
                    self.table_state.select(Some(cmp::min(i + 1, len - 1)));
                }
            }
            _ => {}
        }
    }

    pub fn list_up(&mut self) {
        match self.focus {
            Focus::SampleList => {
                let len = self.samples.len();
                if len > 0 {
                    let i = self.sample_state.selected().unwrap_or(0);
                    self.sample_state.select(Some(if i == 0 { len - 1 } else { i - 1 }));
                }
            }
            Focus::TagList => {
                let len = self.tags.len();
                if len > 0 {
                    let i = self.tag_state.selected().unwrap_or(0);
                    self.tag_state.select(Some(if i == 0 { len - 1 } else { i - 1 }));
                }
            }
            Focus::ExperimentList => {
                let len = self.experiments.len();
                if len > 0 {
                    let i = self.experiment_state.selected().unwrap_or(0);
                    self.experiment_state.select(Some(if i == 0 { len - 1 } else { i - 1 }));
                }
            }
            Focus::Table => {
                if self.expanded_table_row.is_some() {
                    self.expanded_files_scroll_offset = self.expanded_files_scroll_offset.saturating_sub(1);
                    self.needs_redraw = true;
                    return;
                }
                let len = self.filtered_groups.len();
                if len > 0 {
                    let i = self.table_state.selected().unwrap_or(0);
                    self.table_state.select(Some(if i == 0 { 0 } else { i - 1 }));
                }
            }
            _ => {}
        }
    }

    pub fn toggle_filter(&mut self) {
        match self.focus {
            Focus::SampleList => {
                if let Some(i) = self.sample_state.selected() {
                    if let Some(s) = self.samples.get(i) {
                        if self.selected_samples.contains(s) {
                            self.selected_samples.remove(s);
                        } else {
                            self.selected_samples.insert(s.clone());
                        }
                        self.refresh_filtered();
                    }
                }
            }
            Focus::TagList => {
                if let Some(i) = self.tag_state.selected() {
                    if let Some(t) = self.tags.get(i) {
                        if self.selected_tags.contains(t) {
                            self.selected_tags.remove(t);
                        } else {
                            self.selected_tags.insert(t.clone());
                        }
                        self.refresh_filtered();
                    }
                }
            }
            Focus::ExperimentList => {
                if let Some(i) = self.experiment_state.selected() {
                    if let Some((n, _)) = self.experiments.get(i) {
                        if self.selected_experiments.contains(n) {
                            self.selected_experiments.remove(n);
                        } else {
                            self.selected_experiments.insert(*n);
                        }
                        self.refresh_filtered();
                    }
                }
            }
            _ => {}
        }
    }

    pub fn set_mode_rename(&mut self) {
        self.rename_retag_buffer = self
            .table_state
            .selected()
            .and_then(|i| self.filtered_groups.get(i))
            .map(|r| r.sample.clone())
            .unwrap_or_default();
        self.mode = AppMode::RenameSample;
    }

    pub fn set_mode_retag(&mut self) {
        self.rename_retag_buffer = self
            .table_state
            .selected()
            .and_then(|i| self.filtered_groups.get(i))
            .map(|r| r.tag.clone())
            .unwrap_or_default();
        self.mode = AppMode::EditTag;
    }

    pub fn set_mode_normal(&mut self) {
        self.mode = AppMode::Normal;
        self.rename_retag_buffer.clear();
    }

    pub fn rename_retag_push_char(&mut self, c: char) {
        if c.is_ascii() && !c.is_control() {
            self.rename_retag_buffer.push(c);
        }
    }

    pub fn rename_retag_pop_char(&mut self) {
        self.rename_retag_buffer.pop();
    }

    pub fn apply_rename_retag(&mut self) {
        let row = self
            .table_state
            .selected()
            .and_then(|i| self.filtered_groups.get(i));
        let Some(row) = row else { return };
        if !self.has_catalog {
            return;
        }
        let db_path = Path::new(&self.current_root).join(CATALOG_DB_NAME);
        if !db_path.exists() {
            return;
        }
        let sample = self.rename_retag_buffer.trim();
        let (sample_name, tag) = match self.mode {
            AppMode::RenameSample => (Some(sample), None),
            AppMode::EditTag => (None, Some(sample)),
            _ => return,
        };
        let (fallback_sample, fallback_tag) = (row.sample.as_str(), row.tag.as_str());
        let new_sample = sample_name.or(Some(fallback_sample)).unwrap_or("");
        let new_tag = tag.or(Some(fallback_tag));
        if new_sample.is_empty() {
            self.set_status("Sample name cannot be empty.".to_string(), true);
            self.set_mode_normal();
            return;
        }
        let mut renamed = 0u32;
        for file_row in &row.file_rows {
            let old_path = Path::new(&file_row.file_path);
            let parent = match old_path.parent() {
                Some(p) => p,
                None => continue,
            };
            let new_stem = build_fits_stem(
                new_sample,
                new_tag,
                file_row.experiment_number,
                file_row.frame_number,
            );
            let new_path = parent.join(format!("{}.fits", new_stem));
            let old_path_str = file_row.file_path.as_str();
            if old_path == new_path.as_path() {
                continue;
            }
            if new_path.exists() {
                self.set_status(
                    format!("Target already exists: {}", new_path.display()),
                    true,
                );
                self.set_mode_normal();
                return;
            }
            if let Err(e) = fs::rename(old_path_str, &new_path) {
                self.set_status(
                    format!("Rename failed: {}", e),
                    true,
                );
                self.set_mode_normal();
                return;
            }
            let new_path_str = match new_path.to_str() {
                Some(s) => s,
                None => {
                    self.set_status("Invalid UTF-8 in new path.".to_string(), true);
                    self.set_mode_normal();
                    return;
                }
            };
            if let Err(e) = rename_file_in_catalog(
                &db_path,
                old_path_str,
                new_path_str,
                &new_stem,
                new_sample,
                new_tag,
            ) {
                self.set_status(format!("Catalog update failed: {}", e), true);
                self.set_mode_normal();
                return;
            }
            renamed += 1;
        }
        self.reload_from_catalog();
        self.set_status(
            format!("Renamed {} file(s).", renamed),
            false,
        );
        self.set_mode_normal();
    }

    pub fn set_mode_search(&mut self) {
        self.mode = AppMode::Search;
        self.needs_redraw = true;
    }

    pub fn search_push_char(&mut self, c: char) {
        self.search_query.push(c);
        self.refresh_filtered();
    }

    pub fn search_pop_char(&mut self) {
        if self.search_query.pop().is_some() {
            self.refresh_filtered();
        }
    }

    pub fn search_clear(&mut self) {
        self.search_query.clear();
        self.refresh_filtered();
    }

    pub fn list_first(&mut self) {
        match self.focus {
            Focus::SampleList if !self.samples.is_empty() => {
                self.sample_state.select(Some(0));
            }
            Focus::TagList if !self.tags.is_empty() => {
                self.tag_state.select(Some(0));
            }
            Focus::ExperimentList if !self.experiments.is_empty() => {
                self.experiment_state.select(Some(0));
            }
            Focus::Table if !self.filtered_groups.is_empty() => {
                self.table_state.select(Some(0));
            }
            _ => {}
        }
        self.needs_redraw = true;
    }

    pub fn list_last(&mut self) {
        match self.focus {
            Focus::SampleList if !self.samples.is_empty() => {
                let n = self.samples.len() - 1;
                self.sample_state.select(Some(n));
            }
            Focus::TagList if !self.tags.is_empty() => {
                let n = self.tags.len() - 1;
                self.tag_state.select(Some(n));
            }
            Focus::ExperimentList if !self.experiments.is_empty() => {
                let n = self.experiments.len() - 1;
                self.experiment_state.select(Some(n));
            }
            Focus::Table if !self.filtered_groups.is_empty() => {
                let n = self.filtered_groups.len() - 1;
                self.table_state.select(Some(n));
            }
            _ => {}
        }
        self.needs_redraw = true;
    }
}

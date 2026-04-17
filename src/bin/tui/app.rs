use chrono::NaiveDateTime;
use fuzzy_matcher::skim::SkimMatcherV2;
use fuzzy_matcher::FuzzyMatcher;
use ratatui::layout::Rect;
use ratatui::widgets::{ListState, ScrollbarState, TableState};
use std::cmp;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::ops::{Deref, DerefMut};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use super::catalog_handle::CatalogHandle;

#[cfg(feature = "catalog")]
use super::scan_type::{classify_scan_type, ReflectivityScanType};
#[cfg(feature = "catalog")]
use pyref::build_fits_stem;
#[cfg(feature = "catalog")]
use pyref::catalog::{
    catalog_file_count, get_scan_point_uid_by_source_path, list_beamtime_entries,
    list_beamtime_entries_v2, list_beamtimes, query_files, query_scan_points, register_beamtime,
    rename_file_in_catalog, resolve_catalog_path, update_beamspot, update_beamspot_scan_point,
    FileRow, DEFAULT_INGEST_HEADER_ITEMS,
};
#[cfg(feature = "watch")]
use pyref::catalog::{run_catalog_watcher, DEFAULT_DEBOUNCE_MS};

type BeamspotUpdate = (PathBuf, i64, i64, Option<f64>);

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ProfileRow {
    pub sample: String,
    pub tag: String,
    pub energy_str: String,
    pub pol: String,
    pub q_range_str: String,
    pub data_points: u32,
    pub quality_placeholder: String,
    pub scan_number: u32,
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
    pub scan_numbers: Vec<u32>,
    pub scan_duration_hm: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Focus {
    Nav,
    SampleList,
    TagList,
    ScanList,
    Table,
    SearchBar,
}

const FOCUS_ORDER: [Focus; 6] = [
    Focus::Nav,
    Focus::SampleList,
    Focus::TagList,
    Focus::ScanList,
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
    IngestingDirectory,
    CatalogUpdating,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Screen {
    Launcher,
    Beamtime,
    Explorer,
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
    pub scan_list: Rect,
    pub table: Rect,
    pub sample_scrollbar: Option<Rect>,
    pub tag_scrollbar: Option<Rect>,
    pub scan_scrollbar: Option<Rect>,
    pub table_scrollbar: Option<Rect>,
    pub expanded_files: Option<Rect>,
    pub expanded_files_scrollbar: Option<Rect>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScrollbarDragTarget {
    SampleList,
    TagList,
    ScanList,
    Table,
    ExpandedFiles,
}

#[cfg(feature = "catalog")]
pub struct ExplorerFsUpdate {
    pub path: PathBuf,
    pub beamtime_count: u32,
    pub fits_count: u32,
}

pub struct App {
    pub navigator: super::navigator::Navigator,
    pub preview_tx: Option<mpsc::Sender<(PathBuf, Option<f64>)>>,
    #[cfg(feature = "catalog")]
    pub beamspot_rx: Option<mpsc::Receiver<BeamspotUpdate>>,
    #[cfg(feature = "catalog")]
    pub preview_cmd_rx: Option<mpsc::Receiver<super::preview::PreviewCommand>>,
    #[cfg(feature = "catalog")]
    explorer_fs_rx: Option<mpsc::Receiver<ExplorerFsUpdate>>,
    pub layout: String,
    pub keymap: String,
    pub theme: String,
    pub needs_redraw: bool,
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
    NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S")
        .ok()
        .or_else(|| NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S%.3f").ok())
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

fn beamspot_stats(beamspots: &[(i64, i64)]) -> (f64, f64, f64, f64) {
    let n = beamspots.len() as f64;
    if n == 0.0 {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let (row_sum, col_sum): (i64, i64) = beamspots
        .iter()
        .fold((0i64, 0i64), |(r, c), &(ri, ci)| (r + ri, c + ci));
    let row_mean = row_sum as f64 / n;
    let col_mean = col_sum as f64 / n;
    if n < 2.0 {
        return (row_mean, 0.0, col_mean, 0.0);
    }
    let (row_var, col_var) = beamspots
        .iter()
        .fold((0.0f64, 0.0f64), |(rv, cv), &(ri, ci)| {
            let dr = ri as f64 - row_mean;
            let dc = ci as f64 - col_mean;
            (rv + dr * dr, cv + dc * dc)
        });
    let row_std = (row_var / (n - 1.0)).sqrt();
    let col_std = (col_var / (n - 1.0)).sqrt();
    (row_mean, row_std, col_mean, col_std)
}

#[cfg(feature = "catalog")]
fn build_groups_from_files(rows: Vec<FileRow>) -> Vec<GroupedProfileRow> {
    let mut key_to_rows: HashMap<(String, String, String, i64), Vec<FileRow>> = HashMap::new();
    for r in rows {
        let sample = r.sample_name.clone();
        let tag = r.tag.clone().unwrap_or_default();
        let pol_str = r
            .epu_polarization
            .map(|p| {
                let rounded = p.round() as i32;
                if rounded == 100 {
                    "S".to_string()
                } else if rounded == 190 {
                    "P".to_string()
                } else {
                    format!("{:.2}", p)
                }
            })
            .unwrap_or_else(|| "-".to_string());
        let key = (sample.clone(), tag.clone(), pol_str.clone(), r.scan_number);
        key_to_rows.entry(key).or_default().push(r);
    }
    let mut groups: Vec<GroupedProfileRow> = key_to_rows
        .into_iter()
        .flat_map(|((sample, tag, pol_str, scan_number), file_rows)| {
            let energy_theta_pairs: Vec<(Option<f64>, Option<f64>)> = file_rows
                .iter()
                .map(|r| (r.beamline_energy, r.sample_theta))
                .collect();
            let (scan_type, energy_min, energy_max, theta_min, theta_max) =
                classify_scan_type(&energy_theta_pairs);
            let scan_numbers = vec![scan_number as u32];
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
                            let max_t = thetas.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                            (Some(min_t), Some(max_t))
                        };
                        let e_val = e_bin as f64 * E_ROUND_EV;
                        let energy_min = Some(e_val);
                        let energy_max = Some(e_val);
                        let energy_str = format!("{:.1}", e_val);
                        let scan_duration_hm = format_scan_duration_hm(
                            sub_rows.iter().filter_map(|r| r.date_iso.as_deref()),
                        );
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
                            scan_numbers: scan_numbers.clone(),
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
                    scan_numbers,
                    scan_duration_hm,
                }]
            }
        })
        .collect();
    groups.sort_by(|a, b| {
        a.sample
            .cmp(&b.sample)
            .then_with(|| a.tag.cmp(&b.tag))
            .then(a.scan_numbers.first().cmp(&b.scan_numbers.first()))
            .then(
                a.energy_min
                    .partial_cmp(&b.energy_min)
                    .unwrap_or(cmp::Ordering::Equal),
            )
            .then(a.pol_str.cmp(&b.pol_str))
    });
    groups
}

#[cfg(feature = "catalog")]
fn beamspot_mean_std(rows: &[FileRow]) -> (f64, f64, f64, f64) {
    let vals: Vec<(f64, f64)> = rows
        .iter()
        .filter_map(|r| match (r.beam_row, r.beam_col) {
            (Some(a), Some(b)) => Some((a as f64, b as f64)),
            _ => None,
        })
        .collect();
    if vals.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let n = vals.len() as f64;
    let (row_sum, col_sum) = vals
        .iter()
        .fold((0.0, 0.0), |(sr, sc), (r, c)| (sr + r, sc + c));
    let row_mean = row_sum / n;
    let col_mean = col_sum / n;
    let (row_var, col_var) = vals.iter().fold((0.0, 0.0), |(vr, vc), (r, c)| {
        (vr + (r - row_mean).powi(2), vc + (c - col_mean).powi(2))
    });
    let row_std = (row_var / n).max(0.0).sqrt();
    let col_std = (col_var / n).max(0.0).sqrt();
    (row_mean, row_std, col_mean, col_std)
}

fn catalog_to_profiles(
    current_root: &str,
) -> Option<(
    Vec<GroupedProfileRow>,
    Vec<String>,
    Vec<String>,
    Vec<(u32, String)>,
)> {
    #[cfg(feature = "catalog")]
    {
        let db_path = resolve_catalog_path(Path::new(current_root));
        if !db_path.exists() {
            return None;
        }
        let beamtime_path = Path::new(current_root);
        let v2_entries = list_beamtime_entries_v2(&db_path, beamtime_path).ok();
        let v2_rows = query_scan_points(&db_path, beamtime_path, None).ok();
        let use_v2 = v2_entries
            .as_ref()
            .zip(v2_rows.as_ref())
            .map(|(e, r)| {
                !e.samples.is_empty() || !e.tags.is_empty() || !e.scans.is_empty() || !r.is_empty()
            })
            .unwrap_or(false);
        if use_v2 {
            let entries = v2_entries.unwrap();
            let rows = v2_rows.unwrap();
            let groups = build_groups_from_files(rows);
            let scans: Vec<(u32, String)> = entries
                .scans
                .into_iter()
                .map(|(n, s)| (n as u32, s))
                .collect();
            return Some((groups, entries.samples, entries.tags, scans));
        }
        let entries = list_beamtime_entries(&db_path).ok()?;
        let rows = query_files(&db_path, None).ok()?;
        let groups = build_groups_from_files(rows);
        let scans: Vec<(u32, String)> = entries
            .scans
            .into_iter()
            .map(|(n, s)| (n as u32, s))
            .collect();
        Some((groups, entries.samples, entries.tags, scans))
    }
    #[cfg(not(feature = "catalog"))]
    None
}

impl App {
    fn beamtime_state(&self) -> &super::navigator::BeamtimeState {
        for state in self.navigator.stack.iter().rev() {
            if let super::navigator::ScreenState::Beamtime(s) = state {
                return s;
            }
        }
        panic!("no beamtime state on stack");
    }

    fn beamtime_state_mut(&mut self) -> &mut super::navigator::BeamtimeState {
        for state in self.navigator.stack.iter_mut().rev() {
            if let super::navigator::ScreenState::Beamtime(s) = state {
                return s;
            }
        }
        panic!("no beamtime state on stack");
    }

    // Pass-through getters for backward compatibility
    pub fn all_groups(&self) -> &[GroupedProfileRow] {
        &self.beamtime_state().all_groups
    }
    pub fn all_groups_mut(&mut self) -> &mut Vec<GroupedProfileRow> {
        &mut self.beamtime_state_mut().all_groups
    }
    pub fn filtered_groups(&self) -> &[GroupedProfileRow] {
        &self.beamtime_state().filtered_groups
    }
    pub fn filtered_groups_mut(&mut self) -> &mut Vec<GroupedProfileRow> {
        &mut self.beamtime_state_mut().filtered_groups
    }
    pub fn samples(&self) -> &[String] {
        &self.beamtime_state().samples
    }
    pub fn samples_mut(&mut self) -> &mut Vec<String> {
        &mut self.beamtime_state_mut().samples
    }
    pub fn tags(&self) -> &[String] {
        &self.beamtime_state().tags
    }
    pub fn tags_mut(&mut self) -> &mut Vec<String> {
        &mut self.beamtime_state_mut().tags
    }
    pub fn scans(&self) -> &[(u32, String)] {
        &self.beamtime_state().scans
    }
    pub fn scans_mut(&mut self) -> &mut Vec<(u32, String)> {
        &mut self.beamtime_state_mut().scans
    }
    pub fn selected_samples(&self) -> &HashSet<String> {
        &self.beamtime_state().selected_samples
    }
    pub fn selected_samples_mut(&mut self) -> &mut HashSet<String> {
        &mut self.beamtime_state_mut().selected_samples
    }
    pub fn selected_tags(&self) -> &HashSet<String> {
        &self.beamtime_state().selected_tags
    }
    pub fn selected_tags_mut(&mut self) -> &mut HashSet<String> {
        &mut self.beamtime_state_mut().selected_tags
    }
    pub fn selected_scans(&self) -> &HashSet<u32> {
        &self.beamtime_state().selected_scans
    }
    pub fn selected_scans_mut(&mut self) -> &mut HashSet<u32> {
        &mut self.beamtime_state_mut().selected_scans
    }
    pub fn sample_state(&self) -> &ListState {
        &self.beamtime_state().sample_state
    }
    pub fn sample_state_mut(&mut self) -> &mut ListState {
        &mut self.beamtime_state_mut().sample_state
    }
    pub fn tag_state(&self) -> &ListState {
        &self.beamtime_state().tag_state
    }
    pub fn tag_state_mut(&mut self) -> &mut ListState {
        &mut self.beamtime_state_mut().tag_state
    }
    pub fn scan_list_state(&self) -> &ListState {
        &self.beamtime_state().scan_list_state
    }
    pub fn scan_list_state_mut(&mut self) -> &mut ListState {
        &mut self.beamtime_state_mut().scan_list_state
    }
    pub fn table_state(&self) -> &TableState {
        &self.beamtime_state().table_state
    }
    pub fn table_state_mut(&mut self) -> &mut TableState {
        &mut self.beamtime_state_mut().table_state
    }
    pub fn sample_scroll_state(&self) -> &ScrollbarState {
        &self.beamtime_state().sample_scroll_state
    }
    pub fn sample_scroll_state_mut(&mut self) -> &mut ScrollbarState {
        &mut self.beamtime_state_mut().sample_scroll_state
    }
    pub fn tag_scroll_state(&self) -> &ScrollbarState {
        &self.beamtime_state().tag_scroll_state
    }
    pub fn tag_scroll_state_mut(&mut self) -> &mut ScrollbarState {
        &mut self.beamtime_state_mut().tag_scroll_state
    }
    pub fn scan_scroll_state(&self) -> &ScrollbarState {
        &self.beamtime_state().scan_scroll_state
    }
    pub fn scan_scroll_state_mut(&mut self) -> &mut ScrollbarState {
        &mut self.beamtime_state_mut().scan_scroll_state
    }
    pub fn table_scroll_state(&self) -> &ScrollbarState {
        &self.beamtime_state().table_scroll_state
    }
    pub fn table_scroll_state_mut(&mut self) -> &mut ScrollbarState {
        &mut self.beamtime_state_mut().table_scroll_state
    }
    pub fn expanded_files_scroll_state(&self) -> &ScrollbarState {
        &self.beamtime_state().expanded_files_scroll_state
    }
    pub fn expanded_files_scroll_state_mut(&mut self) -> &mut ScrollbarState {
        &mut self.beamtime_state_mut().expanded_files_scroll_state
    }
    pub fn table_sort_column(&self) -> Option<usize> {
        self.beamtime_state().table_sort_column
    }
    pub fn set_table_sort_column(&mut self, col: Option<usize>) {
        self.beamtime_state_mut().table_sort_column = col;
    }
    pub fn table_sort_ordering(&self) -> Option<cmp::Ordering> {
        self.beamtime_state().table_sort_ordering
    }
    pub fn set_table_sort_ordering(&mut self, ord: Option<cmp::Ordering>) {
        self.beamtime_state_mut().table_sort_ordering = ord;
    }
    pub fn expanded_table_row(&self) -> Option<usize> {
        self.beamtime_state().expanded_table_row
    }
    pub fn set_expanded_table_row(&mut self, row: Option<usize>) {
        self.beamtime_state_mut().expanded_table_row = row;
    }
    pub fn expanded_selected_file_index(&self) -> Option<usize> {
        self.beamtime_state().expanded_selected_file_index
    }
    pub fn set_expanded_selected_file_index(&mut self, idx: Option<usize>) {
        self.beamtime_state_mut().expanded_selected_file_index = idx;
    }
    pub fn expanded_files_scroll_offset(&self) -> usize {
        self.beamtime_state().expanded_files_scroll_offset
    }
    pub fn set_expanded_files_scroll_offset(&mut self, offset: usize) {
        self.beamtime_state_mut().expanded_files_scroll_offset = offset;
    }
    pub fn expanded_files_sort_column(&self) -> Option<usize> {
        self.beamtime_state().expanded_files_sort_column
    }
    pub fn set_expanded_files_sort_column(&mut self, col: Option<usize>) {
        self.beamtime_state_mut().expanded_files_sort_column = col;
    }
    pub fn expanded_files_sort_ordering(&self) -> Option<cmp::Ordering> {
        self.beamtime_state().expanded_files_sort_ordering
    }
    pub fn set_expanded_files_sort_ordering(&mut self, ord: Option<cmp::Ordering>) {
        self.beamtime_state_mut().expanded_files_sort_ordering = ord;
    }
    pub fn last_expanded_files_visible(&self) -> usize {
        self.beamtime_state().last_expanded_files_visible
    }
    pub fn set_last_expanded_files_visible(&mut self, visible: usize) {
        self.beamtime_state_mut().last_expanded_files_visible = visible;
    }
    pub fn focus(&self) -> Focus {
        self.beamtime_state().focus
    }
    pub fn set_focus(&mut self, focus: Focus) {
        self.beamtime_state_mut().focus = focus;
    }
    pub fn mode(&self) -> AppMode {
        self.beamtime_state().mode
    }
    pub fn set_mode(&mut self, mode: AppMode) {
        self.beamtime_state_mut().mode = mode;
    }
    pub fn current_root(&self) -> &str {
        &self.beamtime_state().current_root
    }
    pub fn current_root_mut(&mut self) -> &mut String {
        &mut self.beamtime_state_mut().current_root
    }
    pub fn search_query(&self) -> &str {
        &self.beamtime_state().search_query
    }
    pub fn search_query_mut(&mut self) -> &mut String {
        &mut self.beamtime_state_mut().search_query
    }
    pub fn has_catalog(&self) -> bool {
        self.beamtime_state().has_catalog
    }
    pub fn set_has_catalog(&mut self, has: bool) {
        self.beamtime_state_mut().has_catalog = has;
    }
    pub fn loading_state(&self) -> LoadingState {
        self.beamtime_state().loading_state
    }
    pub fn set_loading_state(&mut self, state: LoadingState) {
        self.beamtime_state_mut().loading_state = state;
    }
    pub fn spinner_frame(&self) -> u64 {
        self.beamtime_state().spinner_frame
    }
    pub fn set_spinner_frame(&mut self, frame: u64) {
        self.beamtime_state_mut().spinner_frame = frame;
    }
    pub fn ingest_rx(&self) -> &Option<mpsc::Receiver<Result<(), String>>> {
        &self.beamtime_state().ingest_rx
    }
    pub fn ingest_rx_mut(&mut self) -> &mut Option<mpsc::Receiver<Result<(), String>>> {
        &mut self.beamtime_state_mut().ingest_rx
    }
    #[cfg(feature = "catalog")]
    pub fn ingest_progress(&self) -> Option<(u32, u32)> {
        self.beamtime_state().ingest_progress
    }
    #[cfg(feature = "catalog")]
    pub fn set_ingest_progress(&mut self, progress: Option<(u32, u32)>) {
        self.beamtime_state_mut().ingest_progress = progress;
    }
    #[cfg(feature = "catalog")]
    pub fn ingest_progress_rx(&self) -> &Option<mpsc::Receiver<(u32, u32)>> {
        &self.beamtime_state().ingest_progress_rx
    }
    #[cfg(feature = "catalog")]
    pub fn ingest_progress_rx_mut(&mut self) -> &mut Option<mpsc::Receiver<(u32, u32)>> {
        &mut self.beamtime_state_mut().ingest_progress_rx
    }
    #[cfg(feature = "catalog")]
    pub fn pending_ingest_path(&self) -> &Option<PathBuf> {
        &self.beamtime_state().pending_ingest_path
    }
    #[cfg(feature = "catalog")]
    pub fn pending_ingest_path_mut(&mut self) -> &mut Option<PathBuf> {
        &mut self.beamtime_state_mut().pending_ingest_path
    }
    pub fn back_stack(&self) -> &[String] {
        &self.beamtime_state().back_stack
    }
    pub fn back_stack_mut(&mut self) -> &mut Vec<String> {
        &mut self.beamtime_state_mut().back_stack
    }
    pub fn forward_stack(&self) -> &[String] {
        &self.beamtime_state().forward_stack
    }
    pub fn forward_stack_mut(&mut self) -> &mut Vec<String> {
        &mut self.beamtime_state_mut().forward_stack
    }
    pub fn rename_retag_buffer(&self) -> &str {
        &self.beamtime_state().rename_retag_buffer
    }
    pub fn rename_retag_buffer_mut(&mut self) -> &mut String {
        &mut self.beamtime_state_mut().rename_retag_buffer
    }
    pub fn status_message(&self) -> &Option<(String, bool)> {
        &self.beamtime_state().status_message
    }
    pub fn status_message_mut(&mut self) -> &mut Option<(String, bool)> {
        &mut self.beamtime_state_mut().status_message
    }
    pub fn status_message_set_at(&self) -> &Option<Instant> {
        &self.beamtime_state().status_message_set_at
    }
    pub fn status_message_set_at_mut(&mut self) -> &mut Option<Instant> {
        &mut self.beamtime_state_mut().status_message_set_at
    }
    #[cfg(feature = "watch")]
    pub fn catalog_watcher(&self) -> &Option<pyref::catalog::WatchHandle> {
        &self.beamtime_state().catalog_watcher
    }
    #[cfg(feature = "watch")]
    pub fn catalog_watcher_mut(&mut self) -> &mut Option<pyref::catalog::WatchHandle> {
        &mut self.beamtime_state_mut().catalog_watcher
    }
    #[cfg(feature = "watch")]
    pub fn watcher_events_rx(&self) -> &Option<mpsc::Receiver<WatcherEvent>> {
        &self.beamtime_state().watcher_events_rx
    }
    #[cfg(feature = "watch")]
    pub fn watcher_events_rx_mut(&mut self) -> &mut Option<mpsc::Receiver<WatcherEvent>> {
        &mut self.beamtime_state_mut().watcher_events_rx
    }
    pub fn last_body_rects(&self) -> &Option<(Rect, BeamtimeBodyRects)> {
        &self.beamtime_state().last_body_rects
    }
    pub fn last_body_rects_mut(&mut self) -> &mut Option<(Rect, BeamtimeBodyRects)> {
        &mut self.beamtime_state_mut().last_body_rects
    }
    pub fn scrollbar_drag(&self) -> &Option<ScrollbarDragTarget> {
        &self.beamtime_state().scrollbar_drag
    }
    pub fn scrollbar_drag_mut(&mut self) -> &mut Option<ScrollbarDragTarget> {
        &mut self.beamtime_state_mut().scrollbar_drag
    }
    pub fn app_error(&self) -> &Option<super::error::TuiError> {
        &self.beamtime_state().app_error
    }
    pub fn app_error_mut(&mut self) -> &mut Option<super::error::TuiError> {
        &mut self.beamtime_state_mut().app_error
    }
    pub fn app_warning(&self) -> &Option<super::error::TuiError> {
        &self.beamtime_state().app_warning
    }
    pub fn app_warning_mut(&mut self) -> &mut Option<super::error::TuiError> {
        &mut self.beamtime_state_mut().app_warning
    }

    pub fn current_screen(&self) -> Screen {
        match self.navigator.stack.last() {
            #[cfg(feature = "catalog")]
            Some(super::navigator::ScreenState::Launcher(_)) => Screen::Launcher,
            #[cfg(feature = "tui")]
            Some(super::navigator::ScreenState::Explorer(_)) => Screen::Explorer,
            #[cfg(feature = "tui")]
            Some(super::navigator::ScreenState::ConfigModal(_)) => Screen::Explorer,
            _ => Screen::Beamtime,
        }
    }

    #[cfg(feature = "catalog")]
    fn launcher_screen_state_mut(&mut self) -> &mut super::navigator::LauncherScreenState {
        match self.navigator.stack.last_mut() {
            Some(super::navigator::ScreenState::Launcher(s)) => s,
            _ => panic!("expected launcher state on stack"),
        }
    }

    #[cfg(feature = "catalog")]
    pub fn open_dir_active(&self) -> bool {
        match self.navigator.stack.last() {
            Some(super::navigator::ScreenState::Launcher(s)) => s.open_dir_active,
            _ => false,
        }
    }

    #[cfg(feature = "catalog")]
    pub fn launcher_state(&self) -> Option<&LauncherState> {
        match self.navigator.stack.last() {
            Some(super::navigator::ScreenState::Launcher(s)) => s.launcher_state.as_ref(),
            _ => None,
        }
    }

    #[cfg(feature = "catalog")]
    pub fn launcher_state_mut(&mut self) -> Option<&mut LauncherState> {
        match self.navigator.stack.last_mut() {
            Some(super::navigator::ScreenState::Launcher(s)) => s.launcher_state.as_mut(),
            _ => None,
        }
    }

    #[cfg(feature = "catalog")]
    pub fn open_dir_path(&self) -> &str {
        match self.navigator.stack.last() {
            Some(super::navigator::ScreenState::Launcher(s)) => &s.open_dir_path,
            _ => "",
        }
    }

    #[cfg(feature = "catalog")]
    pub fn open_dir_entries(&self) -> &[PathBuf] {
        match self.navigator.stack.last() {
            Some(super::navigator::ScreenState::Launcher(s)) => &s.open_dir_entries,
            _ => &[],
        }
    }

    #[cfg(feature = "catalog")]
    pub fn open_dir_list_state(&self) -> &ListState {
        match self.navigator.stack.last() {
            Some(super::navigator::ScreenState::Launcher(s)) => &s.open_dir_list_state,
            _ => panic!("no launcher state"),
        }
    }

    #[cfg(feature = "catalog")]
    pub fn open_dir_list_state_mut(&mut self) -> &mut ListState {
        match self.navigator.stack.last_mut() {
            Some(super::navigator::ScreenState::Launcher(s)) => &mut s.open_dir_list_state,
            _ => panic!("no launcher state"),
        }
    }

    #[cfg(feature = "catalog")]
    pub fn open_dir_focus(&self) -> OpenDirFocus {
        match self.navigator.stack.last() {
            Some(super::navigator::ScreenState::Launcher(s)) => s.open_dir_focus,
            _ => OpenDirFocus::PathInput,
        }
    }

    #[cfg(feature = "catalog")]
    pub fn set_open_dir_focus(&mut self, focus: OpenDirFocus) {
        if let Some(super::navigator::ScreenState::Launcher(s)) = self.navigator.stack.last_mut() {
            s.open_dir_focus = focus;
        }
    }

    pub fn new(
        current_root: String,
        layout: String,
        keymap: String,
        keybind_bar_lines: u8,
        theme: String,
    ) -> Self {
        let (all_groups, samples, tags, scans, has_catalog) = catalog_to_profiles(&current_root)
            .map(|(a, s, t, e)| (a, s, t, e, true))
            .unwrap_or_else(|| (vec![], vec![], vec![], vec![], false));
        let mut sample_state = ListState::default();
        let mut tag_state = ListState::default();
        let mut scan_list_state = ListState::default();
        let mut table_state = TableState::default();
        if !samples.is_empty() {
            sample_state.select(Some(0));
        }
        if !tags.is_empty() {
            tag_state.select(Some(0));
        }
        if !scans.is_empty() {
            scan_list_state.select(Some(0));
        }
        let selected_samples = HashSet::new();
        let selected_tags = HashSet::new();
        let selected_scans = HashSet::new();
        let all_groups_clone = all_groups.clone();
        let filtered = Self::filter_groups(
            &all_groups,
            &selected_samples,
            &selected_tags,
            &selected_scans,
            "",
        );
        if !filtered.is_empty() {
            table_state.select(Some(0));
        }
        #[cfg(feature = "watch")]
        let catalog_watcher = resolve_catalog_path(Path::new(&current_root))
            .exists()
            .then(|| {
                run_catalog_watcher(
                    Path::new(&current_root),
                    &[],
                    DEFAULT_DEBOUNCE_MS,
                    None,
                    None,
                )
            })
            .and_then(Result::ok);

        let beamtime_state = super::navigator::BeamtimeState {
            all_groups: all_groups_clone,
            filtered_groups: filtered,
            samples,
            tags,
            scans,
            selected_samples,
            selected_tags,
            selected_scans,
            sample_state,
            tag_state,
            scan_list_state,
            table_state,
            sample_scroll_state: ScrollbarState::default(),
            tag_scroll_state: ScrollbarState::default(),
            scan_scroll_state: ScrollbarState::default(),
            table_scroll_state: ScrollbarState::default(),
            expanded_files_scroll_state: ScrollbarState::default(),
            table_sort_column: None,
            table_sort_ordering: None,
            expanded_table_row: None,
            expanded_selected_file_index: None,
            expanded_files_scroll_offset: 0,
            expanded_files_sort_column: Some(1),
            expanded_files_sort_ordering: Some(cmp::Ordering::Less),
            last_expanded_files_visible: 5,
            focus: Focus::SampleList,
            mode: AppMode::Normal,
            current_root,
            search_query: String::new(),
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
            catalog_watcher,
            #[cfg(feature = "watch")]
            watcher_events_rx: None,
            last_body_rects: None,
            scrollbar_drag: None,
            app_error: None,
            app_warning: None,
            keybind_bar_lines,
            data_root: None,
            experimentalist: None,
            beamtime_path: PathBuf::new(),
            cancel_flag: None,
        };

        App {
            navigator: super::navigator::Navigator::new(
                beamtime_state,
                super::catalog_handle::CatalogHandle {
                    db_path: PathBuf::new(),
                },
            ),
            preview_tx: None,
            #[cfg(feature = "catalog")]
            beamspot_rx: None,
            #[cfg(feature = "catalog")]
            preview_cmd_rx: None,
            #[cfg(feature = "catalog")]
            explorer_fs_rx: None,
            layout,
            keymap,
            theme,
            needs_redraw: true,
        }
    }

    #[cfg(feature = "catalog")]
    pub fn new_for_launcher(
        layout: String,
        keymap: String,
        keybind_bar_lines: u8,
        theme: String,
    ) -> Self {
        let beamtimes = list_beamtimes().unwrap_or_default();
        let mut list_state = ListState::default();
        if !beamtimes.is_empty() {
            list_state.select(Some(0));
        }
        let launcher_state = LauncherState {
            beamtimes,
            list_state,
        };

        let beamtime_state = super::navigator::BeamtimeState {
            all_groups: vec![],
            filtered_groups: vec![],
            samples: vec![],
            tags: vec![],
            scans: vec![],
            selected_samples: HashSet::new(),
            selected_tags: HashSet::new(),
            selected_scans: HashSet::new(),
            sample_state: ListState::default(),
            tag_state: ListState::default(),
            scan_list_state: ListState::default(),
            table_state: TableState::default(),
            sample_scroll_state: ScrollbarState::default(),
            tag_scroll_state: ScrollbarState::default(),
            scan_scroll_state: ScrollbarState::default(),
            table_scroll_state: ScrollbarState::default(),
            expanded_files_scroll_state: ScrollbarState::default(),
            table_sort_column: None,
            table_sort_ordering: None,
            expanded_table_row: None,
            expanded_selected_file_index: None,
            expanded_files_scroll_offset: 0,
            expanded_files_sort_column: Some(1),
            expanded_files_sort_ordering: Some(cmp::Ordering::Less),
            last_expanded_files_visible: 5,
            focus: Focus::SampleList,
            mode: AppMode::Normal,
            current_root: String::new(),
            search_query: String::new(),
            has_catalog: false,
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
            catalog_watcher: None,
            #[cfg(feature = "watch")]
            watcher_events_rx: None,
            last_body_rects: None,
            scrollbar_drag: None,
            app_error: None,
            app_warning: None,
            keybind_bar_lines,
            data_root: None,
            experimentalist: None,
            beamtime_path: PathBuf::new(),
            cancel_flag: None,
        };

        let mut navigator = super::navigator::Navigator::new(
            beamtime_state,
            super::catalog_handle::CatalogHandle {
                db_path: PathBuf::new(),
            },
        );
        navigator
            .stack
            .push(super::navigator::ScreenState::Launcher(
                super::navigator::LauncherScreenState {
                    launcher_state: Some(launcher_state),
                    open_dir_active: false,
                    open_dir_path: String::new(),
                    open_dir_entries: vec![],
                    open_dir_list_state: ListState::default(),
                    open_dir_focus: OpenDirFocus::PathInput,
                },
            ));

        App {
            navigator,
            preview_tx: None,
            #[cfg(feature = "catalog")]
            beamspot_rx: None,
            #[cfg(feature = "catalog")]
            preview_cmd_rx: None,
            #[cfg(feature = "catalog")]
            explorer_fs_rx: None,
            layout,
            keymap,
            theme,
            needs_redraw: true,
        }
    }

    pub fn new_for_explorer(
        data_root: PathBuf,
        layout: String,
        keymap: String,
        keybind_bar_lines: u8,
        theme: String,
    ) -> Self {
        let catalog_db = pyref::catalog::data_root_catalog_path(&data_root);
        let explorer_state = super::explorer::ExplorerState::new(data_root);

        let beamtime_state = super::navigator::BeamtimeState {
            all_groups: vec![],
            filtered_groups: vec![],
            samples: vec![],
            tags: vec![],
            scans: vec![],
            selected_samples: HashSet::new(),
            selected_tags: HashSet::new(),
            selected_scans: HashSet::new(),
            sample_state: ListState::default(),
            tag_state: ListState::default(),
            scan_list_state: ListState::default(),
            table_state: TableState::default(),
            sample_scroll_state: ScrollbarState::default(),
            tag_scroll_state: ScrollbarState::default(),
            scan_scroll_state: ScrollbarState::default(),
            table_scroll_state: ScrollbarState::default(),
            expanded_files_scroll_state: ScrollbarState::default(),
            table_sort_column: None,
            table_sort_ordering: None,
            expanded_table_row: None,
            expanded_selected_file_index: None,
            expanded_files_scroll_offset: 0,
            expanded_files_sort_column: Some(1),
            expanded_files_sort_ordering: Some(cmp::Ordering::Less),
            last_expanded_files_visible: 5,
            focus: Focus::SampleList,
            mode: AppMode::Normal,
            current_root: String::new(),
            search_query: String::new(),
            has_catalog: false,
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
            catalog_watcher: None,
            #[cfg(feature = "watch")]
            watcher_events_rx: None,
            last_body_rects: None,
            scrollbar_drag: None,
            app_error: None,
            app_warning: None,
            keybind_bar_lines,
            data_root: None,
            experimentalist: None,
            beamtime_path: PathBuf::new(),
            cancel_flag: None,
        };

        let mut navigator = super::navigator::Navigator::new(
            beamtime_state,
            super::catalog_handle::CatalogHandle::new(catalog_db),
        );
        navigator
            .stack
            .push(super::navigator::ScreenState::Explorer(explorer_state));

        let mut app = App {
            navigator,
            preview_tx: None,
            #[cfg(feature = "catalog")]
            beamspot_rx: None,
            #[cfg(feature = "catalog")]
            preview_cmd_rx: None,
            #[cfg(feature = "catalog")]
            explorer_fs_rx: None,
            layout,
            keymap,
            theme,
            needs_redraw: true,
        };
        #[cfg(feature = "catalog")]
        {
            app.sync_explorer_catalog_columns();
            app.restart_explorer_fs_scan();
        }
        app
    }

    pub fn set_app_error(&mut self, e: super::error::TuiError) {
        *self.app_error_mut() = Some(e);
        self.needs_redraw = true;
    }

    pub fn clear_app_error(&mut self) {
        *self.app_error_mut() = None;
        self.needs_redraw = true;
    }

    pub fn set_app_warning(&mut self, e: super::error::TuiError) {
        *self.app_warning_mut() = Some(e);
        self.needs_redraw = true;
    }

    pub fn clear_app_warning(&mut self) {
        *self.app_warning_mut() = None;
        self.needs_redraw = true;
    }

    pub fn set_status(&mut self, msg: String, is_error: bool) {
        *self.status_message_mut() = Some((msg, is_error));
        *self.status_message_set_at_mut() = Some(Instant::now());
    }

    pub fn clear_status_if_stale(&mut self) {
        const STATUS_TTL_SECS: u64 = 3;
        if let (Some(_), Some(at)) = (
            self.status_message().as_ref(),
            *self.status_message_set_at(),
        ) {
            if at.elapsed().as_secs() >= STATUS_TTL_SECS {
                *self.status_message_mut() = None;
                *self.status_message_set_at_mut() = None;
                self.needs_redraw = true;
            }
        }
    }

    #[cfg(feature = "catalog")]
    #[cfg(feature = "catalog")]
    pub fn launcher_list_down(&mut self) {
        let s = self.launcher_screen_state_mut();
        if let Some(ref mut state) = s.launcher_state {
            let n = state.beamtimes.len();
            if n == 0 {
                return;
            }
            let next = state
                .list_state
                .selected()
                .map(|i| (i + 1).min(n.saturating_sub(1)))
                .unwrap_or(0);
            state.list_state.select(Some(next));
        }
    }

    #[cfg(feature = "catalog")]
    pub fn launcher_list_up(&mut self) {
        let s = self.launcher_screen_state_mut();
        if let Some(ref mut state) = s.launcher_state {
            let n = state.beamtimes.len();
            if n == 0 {
                return;
            }
            let next = state
                .list_state
                .selected()
                .map(|i| i.saturating_sub(1))
                .unwrap_or(0);
            state.list_state.select(Some(next));
        }
    }

    #[cfg(feature = "catalog")]
    pub fn go_to_launcher(&mut self) {
        let beamtimes = list_beamtimes().unwrap_or_default();
        let mut list_state = ListState::default();
        if !beamtimes.is_empty() {
            list_state.select(Some(0));
        }
        self.navigator
            .stack
            .push(super::navigator::ScreenState::Launcher(
                super::navigator::LauncherScreenState {
                    launcher_state: Some(LauncherState {
                        beamtimes,
                        list_state,
                    }),
                    open_dir_active: false,
                    open_dir_path: String::new(),
                    open_dir_entries: vec![],
                    open_dir_list_state: ListState::default(),
                    open_dir_focus: OpenDirFocus::PathInput,
                },
            ));
        #[cfg(feature = "watch")]
        {
            *self.catalog_watcher_mut() = None;
            *self.watcher_events_rx_mut() = None;
        }
        self.needs_redraw = true;
    }

    #[cfg(feature = "catalog")]
    pub fn launcher_open_selected(&mut self) {
        let s = self.launcher_screen_state_mut();
        if let Some(ref mut state) = s.launcher_state {
            if let Some(i) = state.list_state.selected() {
                if let Some((path, _)) = state.beamtimes.get(i) {
                    let root = path.to_string_lossy().into_owned();
                    if self.set_root(root) {
                        self.navigator.stack.pop();
                    }
                }
            }
        }
    }

    #[cfg(feature = "catalog")]
    pub fn launcher_open_directory(&mut self) {
        let s = self.launcher_screen_state_mut();
        s.open_dir_active = true;
        s.open_dir_path = std::env::var("HOME").unwrap_or_else(|_| "/".to_string());
        s.open_dir_focus = OpenDirFocus::List;
        self.refresh_open_dir_entries();
    }

    #[cfg(feature = "catalog")]
    pub fn refresh_open_dir_entries(&mut self) {
        let s = self.launcher_screen_state_mut();
        s.open_dir_entries.clear();
        let path = Path::new(&s.open_dir_path);
        if path.exists() && path.is_dir() {
            if let Ok(rd) = std::fs::read_dir(path) {
                let mut dirs: Vec<PathBuf> = rd
                    .filter_map(|e| e.ok())
                    .filter(|e| e.path().is_dir())
                    .map(|e| e.path())
                    .collect();
                dirs.sort_by(|a, b| a.file_name().cmp(&b.file_name()));
                s.open_dir_entries = dirs;
            }
        }
        s.open_dir_list_state = ListState::default();
        if !s.open_dir_entries.is_empty() || path.parent().is_some() {
            s.open_dir_list_state.select(Some(0));
        }
        self.needs_redraw = true;
    }

    #[cfg(feature = "catalog")]
    pub fn open_dir_cancel(&mut self) {
        let s = self.launcher_screen_state_mut();
        s.open_dir_active = false;
        self.needs_redraw = true;
    }

    #[cfg(feature = "catalog")]
    pub fn open_dir_list_enter(&mut self) {
        let s = self.launcher_screen_state_mut();
        let n = 1 + s.open_dir_entries.len();
        if n == 0 {
            return;
        }
        let i = s.open_dir_list_state.selected().unwrap_or(0);
        if i == 0 {
            if let Some(parent) = Path::new(&s.open_dir_path).parent() {
                s.open_dir_path = parent.to_string_lossy().into_owned();
                self.refresh_open_dir_entries();
            }
        } else if let Some(p) = s.open_dir_entries.get(i - 1) {
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if name.is_empty() {
                return;
            }
            let sep = if s.open_dir_path.ends_with('/') {
                ""
            } else {
                "/"
            };
            s.open_dir_path = format!("{}{}{}", s.open_dir_path, sep, name);
            self.refresh_open_dir_entries();
        }
    }

    #[cfg(feature = "catalog")]
    pub fn open_dir_list_down(&mut self) {
        let s = self.launcher_screen_state_mut();
        let n = 1 + s.open_dir_entries.len();
        if n == 0 {
            return;
        }
        let next = s
            .open_dir_list_state
            .selected()
            .map(|i| (i + 1).min(n.saturating_sub(1)))
            .unwrap_or(0);
        s.open_dir_list_state.select(Some(next));
        self.needs_redraw = true;
    }

    #[cfg(feature = "catalog")]
    pub fn open_dir_list_up(&mut self) {
        let s = self.launcher_screen_state_mut();
        let n = 1 + s.open_dir_entries.len();
        if n == 0 {
            return;
        }
        let next = s
            .open_dir_list_state
            .selected()
            .map(|i| i.saturating_sub(1))
            .unwrap_or(0);
        s.open_dir_list_state.select(Some(next));
        self.needs_redraw = true;
    }

    #[cfg(feature = "catalog")]
    pub fn open_dir_focus_toggle(&mut self) {
        let focus = self.open_dir_focus();
        let new_focus = match focus {
            OpenDirFocus::PathInput => {
                self.refresh_open_dir_entries();
                OpenDirFocus::List
            }
            OpenDirFocus::List => OpenDirFocus::PathInput,
        };
        self.set_open_dir_focus(new_focus);
        self.needs_redraw = true;
    }

    #[cfg(feature = "catalog")]
    pub fn open_dir_path_push(&mut self, c: char) {
        let s = self.launcher_screen_state_mut();
        s.open_dir_path.push(c);
        self.needs_redraw = true;
    }

    #[cfg(feature = "catalog")]
    pub fn open_dir_path_pop(&mut self) {
        let s = self.launcher_screen_state_mut();
        s.open_dir_path.pop();
        self.needs_redraw = true;
    }

    #[cfg(feature = "catalog")]
    pub fn open_dir_autocomplete(&mut self) {
        let open_dir_path = self.open_dir_path().to_string();
        let path = Path::new(&open_dir_path);
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
            let name = first.file_name().and_then(|n| n.to_str()).unwrap_or("");
            let parent_str = parent.to_string_lossy();
            let sep = if parent_str.ends_with('/') { "" } else { "/" };
            let s = self.launcher_screen_state_mut();
            s.open_dir_path = format!("{}{}{}", parent_str, sep, name);
            self.refresh_open_dir_entries();
            self.needs_redraw = true;
        }
    }

    #[cfg(feature = "catalog")]
    pub fn open_dir_confirm(&mut self) -> bool {
        let s = self.launcher_screen_state_mut();
        let path = Path::new(&s.open_dir_path);
        if !path.exists() || !path.is_dir() {
            self.set_status(
                "Path does not exist or is not a directory.".to_string(),
                true,
            );
            return false;
        }
        let canonical = match path.canonicalize() {
            Ok(p) => p,
            Err(_) => {
                self.set_status("Failed to canonicalize path.".to_string(), true);
                return false;
            }
        };
        s.open_dir_active = false;
        let path_for_thread = canonical.clone();
        *self.pending_ingest_path_mut() = Some(canonical);
        let header_items: Vec<String> = DEFAULT_INGEST_HEADER_ITEMS
            .iter()
            .map(|s| (*s).to_string())
            .collect();
        let cfg = super::config::TuiConfig::load_or_default();
        let parallelism = pyref::catalog::IngestParallelism::from_options_or_env(
            cfg.ingest_worker_threads,
            cfg.ingest_resource_fraction,
        );
        let (tx, rx) = mpsc::channel();
        let (progress_tx, progress_rx) = mpsc::channel();
        thread::spawn(move || {
            let result = pyref::catalog::ingest_beamtime_parallel(
                path_for_thread.as_path(),
                &header_items,
                false,
                Some(progress_tx),
                parallelism,
            )
            .map(|_| ())
            .map_err(|e| e.to_string());
            let _ = tx.send(result);
        });
        self.set_loading_state(LoadingState::IngestingDirectory);
        *self.ingest_rx_mut() = Some(rx);
        self.set_ingest_progress(None);
        *self.ingest_progress_rx_mut() = Some(progress_rx);
        self.needs_redraw = true;
        true
    }

    #[cfg(feature = "catalog")]
    #[allow(dead_code)]
    pub fn start_ingest_at(&mut self, path: &Path) {
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
        let cfg = super::config::TuiConfig::load_or_default();
        let parallelism = pyref::catalog::IngestParallelism::from_options_or_env(
            cfg.ingest_worker_threads,
            cfg.ingest_resource_fraction,
        );
        thread::spawn(move || {
            let result = pyref::catalog::ingest_beamtime_parallel(
                path_for_thread.as_path(),
                &header_items,
                false,
                Some(progress_tx),
                parallelism,
            )
            .map(|_| ())
            .map_err(|e| e.to_string());
            let _ = tx.send(result);
        });
        self.pending_ingest_path = Some(path_buf);
        self.loading_state = LoadingState::IngestingDirectory;
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
        if let Some((all_groups, samples, tags, scans)) = catalog_to_profiles(&self.current_root) {
            self.all_groups = all_groups.clone();
            self.samples = samples.clone();
            self.tags = tags.clone();
            self.scans = scans.clone();
            self.has_catalog = true;
            self.sample_state = ListState::default();
            self.tag_state = ListState::default();
            self.scan_list_state = ListState::default();
            self.table_state = TableState::default();
            self.sample_scroll_state = ScrollbarState::default();
            self.tag_scroll_state = ScrollbarState::default();
            self.scan_scroll_state = ScrollbarState::default();
            self.table_scroll_state = ScrollbarState::default();
            self.expanded_files_scroll_state = ScrollbarState::default();
            self.scrollbar_drag = None;
            if !self.samples.is_empty() {
                self.sample_state.select(Some(0));
            }
            if !self.tags.is_empty() {
                self.tag_state.select(Some(0));
            }
            if !self.scans.is_empty() {
                self.scan_list_state.select(Some(0));
            }
            self.refresh_filtered();
            #[cfg(feature = "watch")]
            if resolve_catalog_path(Path::new(&self.current_root)).exists() {
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
            self.scans = vec![];
            self.has_catalog = false;
            self.sample_state = ListState::default();
            self.tag_state = ListState::default();
            self.scan_list_state = ListState::default();
            self.table_state = TableState::default();
            self.sample_scroll_state = ScrollbarState::default();
            self.tag_scroll_state = ScrollbarState::default();
            self.scan_scroll_state = ScrollbarState::default();
            self.table_scroll_state = ScrollbarState::default();
            self.expanded_files_scroll_state = ScrollbarState::default();
            self.scrollbar_drag = None;
        }
        self.needs_redraw = true;
        true
    }

    pub fn nav_up(&mut self) {
        let current_root = self.current_root.clone();
        let path = Path::new(&current_root);
        if let Some(parent) = path.parent() {
            let parent_str = parent.to_string_lossy().into_owned();
            if parent_str != current_root {
                self.back_stack.push(current_root);
                self.forward_stack.clear();
                if !self.set_root(parent_str) {
                    self.set_status("Path not found or not a directory.".to_string(), true);
                }
            }
        }
    }

    pub fn nav_back(&mut self) {
        if let Some(prev) = self.back_stack.pop() {
            let current_root = self.current_root.clone();
            self.forward_stack.push(current_root);
            if !self.set_root(prev) {
                self.set_status("Path not found or not a directory.".to_string(), true);
            }
        }
    }

    pub fn nav_fwd(&mut self) {
        if let Some(next) = self.forward_stack.pop() {
            let current_root = self.current_root.clone();
            self.back_stack.push(current_root);
            if !self.set_root(next) {
                self.set_status("Path not found or not a directory.".to_string(), true);
            }
        }
    }

    fn filter_groups(
        groups: &[GroupedProfileRow],
        samples: &HashSet<String>,
        tags: &HashSet<String>,
        scans: &HashSet<u32>,
        search: &str,
    ) -> Vec<GroupedProfileRow> {
        let search_lower = search.to_lowercase();
        let matcher = SkimMatcherV2::default();
        groups
            .iter()
            .filter(|r| {
                let sample_ok = samples.is_empty() || samples.contains(&r.sample);
                let tag_ok = tags.is_empty() || tags.contains(&r.tag);
                let exp_ok = scans.is_empty() || r.scan_numbers.iter().any(|e| scans.contains(e));
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
                        r.sample, r.tag, r.energy_str, e_range_str, r.pol_str, theta_str
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

    pub fn focused_scan(&self) -> Option<u32> {
        self.scan_list_state
            .selected()
            .and_then(|i| self.scans.get(i).map(|(n, _)| *n))
    }

    pub fn compute_display_order(&self) -> Vec<usize> {
        let n = self.filtered_groups.len();
        let mut order: Vec<usize> = (0..n).collect();
        if let (Some(col), Some(ord)) = (self.table_sort_column, self.table_sort_ordering) {
            order.sort_by(|&a, &b| {
                let ra = &self.filtered_groups[a];
                let rb = &self.filtered_groups[b];
                Self::compare_group_column(col, ra, rb).then_with(|| a.cmp(&b))
            });
            if ord == cmp::Ordering::Greater {
                order.reverse();
            }
        }
        order
    }

    fn compare_group_column(
        col: usize,
        a: &GroupedProfileRow,
        b: &GroupedProfileRow,
    ) -> cmp::Ordering {
        use super::scan_type::ReflectivityScanType;
        let cmp_opt_f64 =
            |x: Option<f64>, y: Option<f64>| x.partial_cmp(&y).unwrap_or(cmp::Ordering::Equal);
        let cmp_scan_type = |s: ReflectivityScanType, t: ReflectivityScanType| {
            let ord = |r: ReflectivityScanType| match r {
                ReflectivityScanType::FixedEnergy => 0,
                ReflectivityScanType::FixedAngle => 1,
                ReflectivityScanType::SinglePoint => 2,
            };
            ord(s).cmp(&ord(t))
        };
        match col {
            0 => a.sample.cmp(&b.sample),
            1 => a.tag.cmp(&b.tag),
            2 => a.pol_str.cmp(&b.pol_str),
            3 => cmp_scan_type(a.scan_type, b.scan_type),
            4 => cmp_opt_f64(a.energy_min, b.energy_min),
            5 => cmp_opt_f64(a.energy_max, b.energy_max),
            6 => cmp_opt_f64(a.theta_min, b.theta_min),
            7 => cmp_opt_f64(a.theta_max, b.theta_max),
            8 => a.file_rows.len().cmp(&b.file_rows.len()),
            9 => a.scan_duration_hm.cmp(&b.scan_duration_hm),
            _ => cmp::Ordering::Equal,
        }
    }

    pub fn cycle_table_sort(&mut self, data_column: usize) {
        if data_column >= 10 {
            return;
        }
        let old_order = self.compute_display_order();
        let selected_orig = self
            .table_state
            .selected()
            .and_then(|d| old_order.get(d).copied());
        let expanded_orig = self
            .expanded_table_row
            .and_then(|d| old_order.get(d).copied());
        let (new_col, new_ord) = match (self.table_sort_column, self.table_sort_ordering) {
            (Some(c), Some(o)) if c == data_column => {
                if o == cmp::Ordering::Less {
                    (Some(data_column), Some(cmp::Ordering::Greater))
                } else {
                    (None, None)
                }
            }
            _ => (Some(data_column), Some(cmp::Ordering::Less)),
        };
        self.table_sort_column = new_col;
        self.table_sort_ordering = new_ord;
        let new_order = self.compute_display_order();
        if let Some(orig) = selected_orig {
            if let Some(pos) = new_order.iter().position(|&i| i == orig) {
                self.table_state.select(Some(pos));
            }
        }
        if let Some(orig) = expanded_orig {
            if let Some(pos) = new_order.iter().position(|&i| i == orig) {
                self.expanded_table_row = Some(pos);
            }
        }
        self.needs_redraw = true;
    }

    pub fn group_at_display_index(&self, display_index: usize) -> Option<&GroupedProfileRow> {
        let order = self.compute_display_order();
        let orig = order.get(display_index).copied()?;
        self.filtered_groups.get(orig)
    }

    #[cfg(feature = "catalog")]
    pub fn expanded_files_display_order(&self, group: &GroupedProfileRow) -> Vec<usize> {
        use super::beamspot;
        let sort_col = self.expanded_files_sort_column.unwrap_or(1);
        let sort_ord = self
            .expanded_files_sort_ordering
            .unwrap_or(cmp::Ordering::Less);
        let files = &group.file_rows;
        let (row_mean, row_std, col_mean, col_std) = beamspot_mean_std(files);
        let fit = beamspot::fit_beamspot_linear(files, group.scan_type);
        let mut indices: Vec<usize> = (0..files.len()).collect();
        indices.sort_by(|&a, &b| {
            let ra = &files[a];
            let rb = &files[b];
            let ord = match sort_col {
                0 => ra.scan_number.cmp(&rb.scan_number),
                1 => ra.frame_number.cmp(&rb.frame_number),
                2 => ra
                    .epu_polarization
                    .partial_cmp(&rb.epu_polarization)
                    .unwrap_or(cmp::Ordering::Equal),
                3 => ra
                    .beamline_energy
                    .partial_cmp(&rb.beamline_energy)
                    .unwrap_or(cmp::Ordering::Equal),
                4 => ra
                    .sample_theta
                    .partial_cmp(&rb.sample_theta)
                    .unwrap_or(cmp::Ordering::Equal),
                5 => ra
                    .beam_row
                    .cmp(&rb.beam_row)
                    .then_with(|| ra.beam_col.cmp(&rb.beam_col)),
                6 => ra
                    .beam_sigma
                    .partial_cmp(&rb.beam_sigma)
                    .unwrap_or(cmp::Ordering::Equal),
                7 => beamspot::beamspot_status(
                    ra.beam_row,
                    ra.beam_col,
                    beamspot::domain_for_row(ra, group.scan_type),
                    fit.as_ref(),
                    row_mean,
                    row_std,
                    col_mean,
                    col_std,
                )
                .1
                .cmp(
                    &beamspot::beamspot_status(
                        rb.beam_row,
                        rb.beam_col,
                        beamspot::domain_for_row(rb, group.scan_type),
                        fit.as_ref(),
                        row_mean,
                        row_std,
                        col_mean,
                        col_std,
                    )
                    .1,
                ),
                _ => cmp::Ordering::Equal,
            };
            if sort_ord == cmp::Ordering::Greater {
                ord.reverse()
            } else {
                ord
            }
        });
        indices
    }

    #[cfg(feature = "catalog")]
    pub fn cycle_expanded_files_sort(&mut self, column: usize) {
        if column >= 8 {
            return;
        }
        let (new_col, new_ord) = match (
            self.expanded_files_sort_column,
            self.expanded_files_sort_ordering,
        ) {
            (Some(c), Some(o)) if c == column => {
                if o == cmp::Ordering::Less {
                    (Some(column), Some(cmp::Ordering::Greater))
                } else {
                    (None, None)
                }
            }
            _ => (Some(column), Some(cmp::Ordering::Less)),
        };
        self.expanded_files_sort_column = new_col;
        self.expanded_files_sort_ordering = new_ord;
        self.expanded_files_scroll_offset = 0;
        self.needs_redraw = true;
    }

    #[cfg(feature = "catalog")]
    fn problem_display_indices(group: &GroupedProfileRow, display_order: &[usize]) -> Vec<usize> {
        use super::beamspot;
        let files = &group.file_rows;
        let (row_mean, row_std, col_mean, col_std) = beamspot_mean_std(files);
        let fit = beamspot::fit_beamspot_linear(files, group.scan_type);
        (0..display_order.len())
            .filter(|&display_ix| {
                let real_ix = display_order[display_ix];
                let row = match files.get(real_ix) {
                    Some(r) => r,
                    None => return false,
                };
                let (_, key) = beamspot::beamspot_status(
                    row.beam_row,
                    row.beam_col,
                    beamspot::domain_for_row(row, group.scan_type),
                    fit.as_ref(),
                    row_mean,
                    row_std,
                    col_mean,
                    col_std,
                );
                key == 1 || key == 2
            })
            .collect()
    }

    #[cfg(feature = "catalog")]
    pub fn go_to_next_problem(&mut self) {
        let idx = match self.expanded_table_row {
            Some(i) => i,
            None => return,
        };
        let group = match self.group_at_display_index(idx) {
            Some(g) => g,
            None => return,
        };
        let order = self.expanded_files_display_order(group);
        let problems = Self::problem_display_indices(group, &order);
        if problems.is_empty() {
            self.set_status("No err/warning frames in this profile.".to_string(), false);
            return;
        }
        let cur = self.expanded_selected_file_index.unwrap_or(0);
        let next_ix = problems.iter().find(|&&p| p > cur).copied();
        let target = match next_ix {
            Some(ix) => ix,
            None => problems[0],
        };
        self.expanded_selected_file_index = Some(target);
        let visible = self.last_expanded_files_visible.max(1);
        let file_count = order.len();
        let max_offset = file_count.saturating_sub(visible);
        if target >= self.expanded_files_scroll_offset + visible {
            self.expanded_files_scroll_offset =
                cmp::min((target + 1).saturating_sub(visible), max_offset);
        } else if target < self.expanded_files_scroll_offset {
            self.expanded_files_scroll_offset = target;
        }
        self.needs_redraw = true;
        if self.preview_tx.is_some() {
            self.send_preview_path_if_selected();
        }
    }

    #[cfg(feature = "catalog")]
    pub fn go_to_prev_problem(&mut self) {
        let idx = match self.expanded_table_row {
            Some(i) => i,
            None => return,
        };
        let group = match self.group_at_display_index(idx) {
            Some(g) => g,
            None => return,
        };
        let order = self.expanded_files_display_order(group);
        let problems = Self::problem_display_indices(group, &order);
        if problems.is_empty() {
            self.set_status("No err/warning frames in this profile.".to_string(), false);
            return;
        }
        let cur = self.expanded_selected_file_index.unwrap_or(0);
        let prev_ix = problems.iter().rev().find(|&&p| p < cur).copied();
        let target = match prev_ix {
            Some(ix) => ix,
            None => *problems.last().unwrap(),
        };
        self.expanded_selected_file_index = Some(target);
        if target < self.expanded_files_scroll_offset {
            self.expanded_files_scroll_offset = target;
        }
        self.needs_redraw = true;
        if self.preview_tx.is_some() {
            self.send_preview_path_if_selected();
        }
    }

    #[cfg(feature = "catalog")]
    fn last_ok_display_index(
        group: &GroupedProfileRow,
        display_order: &[usize],
        current_ix: usize,
    ) -> Option<usize> {
        use super::beamspot;
        let files = &group.file_rows;
        let (row_mean, row_std, col_mean, col_std) = beamspot_mean_std(files);
        let fit = beamspot::fit_beamspot_linear(files, group.scan_type);
        for display_ix in (0..current_ix).rev() {
            let real_ix = display_order[display_ix];
            let row = files.get(real_ix)?;
            let (_, key) = beamspot::beamspot_status(
                row.beam_row,
                row.beam_col,
                beamspot::domain_for_row(row, group.scan_type),
                fit.as_ref(),
                row_mean,
                row_std,
                col_mean,
                col_std,
            );
            if key == 0 {
                return Some(display_ix);
            }
        }
        None
    }

    #[cfg(feature = "catalog")]
    fn next_ok_display_index(
        group: &GroupedProfileRow,
        display_order: &[usize],
        current_ix: usize,
    ) -> Option<usize> {
        use super::beamspot;
        let files = &group.file_rows;
        let (row_mean, row_std, col_mean, col_std) = beamspot_mean_std(files);
        let fit = beamspot::fit_beamspot_linear(files, group.scan_type);
        for display_ix in (current_ix + 1)..display_order.len() {
            let real_ix = display_order[display_ix];
            let row = files.get(real_ix)?;
            let (_, key) = beamspot::beamspot_status(
                row.beam_row,
                row.beam_col,
                beamspot::domain_for_row(row, group.scan_type),
                fit.as_ref(),
                row_mean,
                row_std,
                col_mean,
                col_std,
            );
            if key == 0 {
                return Some(display_ix);
            }
        }
        None
    }

    #[cfg(feature = "catalog")]
    pub fn use_last_ok_beamspot(&mut self) {
        let idx = match self.expanded_table_row {
            Some(i) => i,
            None => return,
        };
        let group = match self.group_at_display_index(idx) {
            Some(g) => g,
            None => return,
        };
        let order = self.expanded_files_display_order(group);
        let cur = self.expanded_selected_file_index.unwrap_or(0);
        let ok_ix = match Self::last_ok_display_index(group, &order, cur) {
            Some(ix) => ix,
            None => {
                self.set_status(
                    "No previous OK beamspot in this profile.".to_string(),
                    false,
                );
                return;
            }
        };
        let real_ix = order[ok_ix];
        let src_row = match group.file_rows.get(real_ix) {
            Some(r) => r,
            None => return,
        };
        let (beam_row, beam_col) = match (src_row.beam_row, src_row.beam_col) {
            (Some(r), Some(c)) => (r, c),
            _ => return,
        };
        let beam_sigma = src_row.beam_sigma;
        let current_real_ix = order.get(cur).copied().unwrap_or(0);
        let current_row = match group.file_rows.get(current_real_ix) {
            Some(r) => r,
            None => return,
        };
        let db_path = resolve_catalog_path(Path::new(&self.current_root));
        if !db_path.exists() {
            self.set_status("No catalog in current root.".to_string(), true);
            return;
        }
        let file_path = &current_row.file_path;
        if let Err(e) = update_beamspot(&db_path, file_path, beam_row, beam_col, beam_sigma) {
            self.set_status(
                format!("Failed to store beamspot from last OK: {}", e),
                true,
            );
            return;
        }
        if let Some(ref uid) = current_row.scan_point_uid {
            let _ = update_beamspot_scan_point(&db_path, uid, beam_row, beam_col, beam_sigma);
        }
        let was_expanded = self.expanded_table_row.is_some();
        let restore_selection = self.expanded_table_row.or(self.table_state.selected());
        self.reload_from_catalog_impl(was_expanded, restore_selection);
        self.set_status("Beamspot copied from previous OK frame.".to_string(), false);
    }

    #[cfg(feature = "catalog")]
    pub fn use_next_ok_beamspot(&mut self) {
        let idx = match self.expanded_table_row {
            Some(i) => i,
            None => return,
        };
        let group = match self.group_at_display_index(idx) {
            Some(g) => g,
            None => return,
        };
        let order = self.expanded_files_display_order(group);
        let cur = self.expanded_selected_file_index.unwrap_or(0);
        let ok_ix = match Self::next_ok_display_index(group, &order, cur) {
            Some(ix) => ix,
            None => {
                self.set_status("No next OK beamspot in this profile.".to_string(), false);
                return;
            }
        };
        let real_ix = order[ok_ix];
        let src_row = match group.file_rows.get(real_ix) {
            Some(r) => r,
            None => return,
        };
        let (beam_row, beam_col) = match (src_row.beam_row, src_row.beam_col) {
            (Some(r), Some(c)) => (r, c),
            _ => return,
        };
        let beam_sigma = src_row.beam_sigma;
        let current_real_ix = order.get(cur).copied().unwrap_or(0);
        let current_row = match group.file_rows.get(current_real_ix) {
            Some(r) => r,
            None => return,
        };
        let db_path = resolve_catalog_path(Path::new(&self.current_root));
        if !db_path.exists() {
            self.set_status("No catalog in current root.".to_string(), true);
            return;
        }
        let file_path = &current_row.file_path;
        if let Err(e) = update_beamspot(&db_path, file_path, beam_row, beam_col, beam_sigma) {
            self.set_status(
                format!("Failed to store beamspot from next OK: {}", e),
                true,
            );
            return;
        }
        if let Some(ref uid) = current_row.scan_point_uid {
            let _ = update_beamspot_scan_point(&db_path, uid, beam_row, beam_col, beam_sigma);
        }
        let was_expanded = self.expanded_table_row.is_some();
        let restore_selection = self.expanded_table_row.or(self.table_state.selected());
        self.reload_from_catalog_impl(was_expanded, restore_selection);
        self.set_status("Beamspot copied from next OK frame.".to_string(), false);
    }

    pub fn set_preview_tx(&mut self, tx: mpsc::Sender<(PathBuf, Option<f64>)>) {
        self.preview_tx = Some(tx);
    }

    #[cfg(feature = "catalog")]
    pub fn set_beamspot_rx(&mut self, rx: mpsc::Receiver<BeamspotUpdate>) {
        self.beamspot_rx = Some(rx);
    }

    #[cfg(feature = "catalog")]
    pub fn set_preview_cmd_rx(&mut self, rx: mpsc::Receiver<super::preview::PreviewCommand>) {
        self.preview_cmd_rx = Some(rx);
    }

    #[cfg(feature = "catalog")]
    pub fn try_recv_preview_cmd(&mut self) {
        let mut commands = Vec::new();
        if let Some(rx) = self.preview_cmd_rx.as_mut() {
            while let Ok(cmd) = rx.try_recv() {
                commands.push(cmd);
            }
        }
        for cmd in commands {
            match cmd {
                super::preview::PreviewCommand::GoToNextProblem => self.go_to_next_problem(),
                super::preview::PreviewCommand::GoToPrevProblem => self.go_to_prev_problem(),
            }
        }
    }

    #[cfg(feature = "catalog")]
    pub fn try_recv_beamspot_updates(&mut self) {
        let mut updates = Vec::new();
        if let Some(rx) = self.beamspot_rx.as_mut() {
            while let Ok(x) = rx.try_recv() {
                updates.push(x);
            }
        }
        for (path, beam_row, beam_col, beam_sigma) in updates {
            self.apply_beamspot_from_preview(path, beam_row, beam_col, beam_sigma);
        }
    }

    #[cfg(feature = "catalog")]
    fn apply_beamspot_from_preview(
        &mut self,
        path: PathBuf,
        beam_row: i64,
        beam_col: i64,
        beam_sigma: Option<f64>,
    ) {
        let db_path = resolve_catalog_path(Path::new(&self.current_root));
        if !db_path.exists() {
            return;
        }
        let file_path = path.to_string_lossy().into_owned();
        if let Err(e) = update_beamspot(&db_path, &file_path, beam_row, beam_col, beam_sigma) {
            self.set_status(
                format!("Failed to store beamspot from preview: {}", e),
                true,
            );
            return;
        }
        if let Ok(Some(uid)) = get_scan_point_uid_by_source_path(&db_path, &file_path) {
            let _ = update_beamspot_scan_point(&db_path, &uid, beam_row, beam_col, beam_sigma);
        }
        let was_expanded = self.expanded_table_row.is_some();
        let restore_selection = self.expanded_table_row.or(self.table_state.selected());
        self.reload_from_catalog_impl(was_expanded, restore_selection);
        self.set_status(
            "Beamspot from preview stored; table updated.".to_string(),
            false,
        );
    }

    pub fn send_preview_path_if_selected(&mut self) {
        if let (Some(tx), Some(idx), Some(display_ix)) = (
            self.preview_tx.as_ref(),
            self.expanded_table_row,
            self.expanded_selected_file_index,
        ) {
            if let Some(g) = self.group_at_display_index(idx) {
                let order = self.expanded_files_display_order(g);
                if let Some(&real_ix) = order.get(display_ix) {
                    if let Some(row) = g.file_rows.get(real_ix) {
                        let frame1 = g.file_rows.iter().min_by_key(|r| r.frame_number);
                        let profile_sigma = frame1.and_then(|r| r.beam_sigma);
                        let _ = tx.send((PathBuf::from(&row.file_path), profile_sigma));
                    }
                }
            }
        }
    }

    pub fn materialize_profile_beamspots(&mut self) {
        let idx = self
            .expanded_table_row
            .or_else(|| self.table_state.selected());
        let Some(idx) = idx else {
            self.set_status(
                "Select a profile and press m to materialize beamspots.".to_string(),
                false,
            );
            return;
        };
        let was_expanded = self.expanded_table_row.is_some();
        let file_paths: Vec<String> = match self.group_at_display_index(idx) {
            Some(g) => g.file_rows.iter().map(|r| r.file_path.clone()).collect(),
            None => return,
        };
        let db_path = resolve_catalog_path(Path::new(&self.current_root));
        if !db_path.exists() {
            self.set_status("No catalog in current root.".to_string(), true);
            return;
        }
        let mut ok = 0usize;
        let mut err_count = 0usize;
        let mut beamspots: Vec<(i64, i64)> = Vec::new();
        for file_path in &file_paths {
            let path = Path::new(file_path);
            match pyref::io::image_mmap::materialize_image_from_path(path) {
                Ok((_, subtracted)) => {
                    let (r, c) = pyref::beamfinding::locate_beam_simple(&subtracted);
                    let subtracted_f64: ndarray::Array2<f64> = subtracted.mapv(|x| x as f64);
                    let beam_sigma = pyref::gaussian_fit::fit_2d_gaussian(&subtracted_f64, None)
                        .map(|f| (f.sigma_row + f.sigma_col) / 2.0)
                        .map(|s| s.clamp(0.5, 20.0));
                    if let Err(e) =
                        update_beamspot(&db_path, file_path, r as i64, c as i64, beam_sigma)
                    {
                        self.set_status(
                            format!("Failed to store beamspot for {}: {}", file_path, e),
                            true,
                        );
                        return;
                    }
                    if let Ok(Some(uid)) = get_scan_point_uid_by_source_path(&db_path, file_path) {
                        let _ = update_beamspot_scan_point(
                            &db_path, &uid, r as i64, c as i64, beam_sigma,
                        );
                    }
                    beamspots.push((r as i64, c as i64));
                    ok += 1;
                }
                Err(e) => {
                    err_count += 1;
                    self.set_status(format!("Failed to load {}: {}", file_path, e), true);
                }
            }
        }
        self.reload_from_catalog_impl(was_expanded, if was_expanded { None } else { Some(idx) });
        let msg = if beamspots.is_empty() {
            if err_count == 0 {
                format!("Materialized {} file(s); beamspots stored.", ok)
            } else {
                format!("Materialized {} file(s), {} failed.", ok, err_count)
            }
        } else {
            let (row_mean, row_std, col_mean, col_std) = beamspot_stats(&beamspots);
            const HIGH_VARIANCE_THRESHOLD: f64 = 5.0;
            let lost = row_std >= HIGH_VARIANCE_THRESHOLD || col_std >= HIGH_VARIANCE_THRESHOLD;
            let stats = format!(
                "row mean={:.1} std={:.1}  col mean={:.1} std={:.1}",
                row_mean, row_std, col_mean, col_std
            );
            let fit_str = self
                .group_at_display_index(idx)
                .and_then(|g| {
                    super::beamspot::fit_beamspot_linear(&g.file_rows, g.scan_type)
                        .map(|f| (g.scan_type, f))
                })
                .map(|(scan_type, f)| {
                    let domain = match scan_type {
                        super::scan_type::ReflectivityScanType::FixedEnergy => "theta",
                        super::scan_type::ReflectivityScanType::FixedAngle => "E",
                        super::scan_type::ReflectivityScanType::SinglePoint => "?",
                    };
                    format!(
                        "  Fit({}): row={:.3}*x+{:.1} col={:.3}*x+{:.1}  res_std row={:.2} col={:.2}",
                        domain,
                        f.row_slope,
                        f.row_intercept,
                        f.col_slope,
                        f.col_intercept,
                        f.row_residual_std,
                        f.col_residual_std
                    )
                });
            let base = format!(
                "Materialized {} file(s); beamspots stored.  {}{}",
                ok,
                stats,
                fit_str.as_deref().unwrap_or("")
            );
            if lost {
                format!("{}  (high variance: beam may be lost)", base)
            } else {
                base
            }
        };
        self.set_status(msg, err_count > 0);
    }

    pub fn refresh_filtered(&mut self) {
        self.filtered_groups = Self::filter_groups(
            &self.all_groups,
            &self.selected_samples,
            &self.selected_tags,
            &self.selected_scans,
            &self.search_query,
        );
        self.clamp_selections();
        self.table_state.select(None);
        self.expanded_table_row = None;
        self.expanded_selected_file_index = None;
        if !self.filtered_groups.is_empty() {
            self.table_state.select(Some(0));
        }
        self.needs_redraw = true;
    }

    pub fn reload_from_catalog(&mut self, preserve_expansion: bool) {
        self.reload_from_catalog_impl(preserve_expansion, None);
    }

    fn reload_from_catalog_impl(
        &mut self,
        preserve_expansion: bool,
        restore_selection: Option<usize>,
    ) {
        #[cfg(feature = "catalog")]
        {
            let saved_expanded_row = preserve_expansion
                .then_some(self.expanded_table_row)
                .flatten();
            let saved_expanded_file_ix = if preserve_expansion {
                self.expanded_selected_file_index
            } else {
                None
            };
            let saved_expanded_scroll = if preserve_expansion {
                self.expanded_files_scroll_offset
            } else {
                0
            };
            let db_path = resolve_catalog_path(Path::new(&self.current_root));
            if !db_path.exists() {
                return;
            }
            let beamtime_path = Path::new(&self.current_root);
            let (entries, rows) = match (
                list_beamtime_entries_v2(&db_path, beamtime_path),
                query_scan_points(&db_path, beamtime_path, None),
            ) {
                (Ok(e), Ok(r)) if !r.is_empty() => (e, r),
                _ => {
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
                    (entries, rows)
                }
            };
            let all_groups = build_groups_from_files(rows);
            let scans: Vec<(u32, String)> = entries
                .scans
                .into_iter()
                .map(|(n, s)| (n as u32, s))
                .collect();
            self.app_error = None;
            self.all_groups = all_groups.clone();
            self.samples = entries.samples.clone();
            self.tags = entries.tags.clone();
            self.scans = scans.clone();
            self.has_catalog = true;
            self.sample_state = ListState::default();
            self.tag_state = ListState::default();
            self.scan_list_state = ListState::default();
            self.table_state = TableState::default();
            self.sample_scroll_state = ScrollbarState::default();
            self.tag_scroll_state = ScrollbarState::default();
            self.scan_scroll_state = ScrollbarState::default();
            self.table_scroll_state = ScrollbarState::default();
            self.expanded_files_scroll_state = ScrollbarState::default();
            self.scrollbar_drag = None;
            if !self.samples.is_empty() {
                self.sample_state.select(Some(0));
            }
            if !self.tags.is_empty() {
                self.tag_state.select(Some(0));
            }
            if !self.scans.is_empty() {
                self.scan_list_state.select(Some(0));
            }
            self.refresh_filtered();
            if let Some(idx) = saved_expanded_row {
                if idx < self.filtered_groups.len() {
                    self.expanded_table_row = Some(idx);
                    self.table_state.select(Some(idx));
                    let n = self.filtered_groups[idx].file_rows.len();
                    self.expanded_selected_file_index = if n == 0 {
                        None
                    } else {
                        Some(
                            saved_expanded_file_ix
                                .map(|i| i.min(n.saturating_sub(1)))
                                .unwrap_or(0),
                        )
                    };
                    self.expanded_files_scroll_offset = saved_expanded_scroll;
                }
            } else if let Some(idx) = restore_selection {
                if idx < self.filtered_groups.len() {
                    self.table_state.select(Some(idx));
                }
            }
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
    pub fn start_ingest(&mut self) {
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
        let cfg = super::config::TuiConfig::load_or_default();
        let parallelism = pyref::catalog::IngestParallelism::from_options_or_env(
            cfg.ingest_worker_threads,
            cfg.ingest_resource_fraction,
        );
        let (tx, rx) = mpsc::channel();
        let (progress_tx, progress_rx) = mpsc::channel();
        thread::spawn(move || {
            let result = pyref::catalog::ingest_beamtime_parallel(
                Path::new(&root_str),
                &header_items,
                false,
                Some(progress_tx),
                parallelism,
            )
            .map(|_| ())
            .map_err(|e| e.to_string());
            let _ = tx.send(result);
        });
        self.loading_state = LoadingState::IngestingDirectory;
        self.ingest_rx = Some(rx);
        self.ingest_progress = None;
        self.ingest_progress_rx = Some(progress_rx);
    }

    #[cfg(not(feature = "catalog"))]
    pub fn start_ingest(&mut self) {}

    #[cfg(feature = "catalog")]
    pub fn try_recv_ingest(&mut self) {
        // Handle ingest for top BeamtimeState on stack (including ingest-in-progress from Explorer)
        // Extract what we need from the mutable state first
        let (ingest_result, bt_path_cloned, data_root_cloned) = {
            if let Some(super::navigator::ScreenState::Beamtime(bt_state)) =
                self.navigator.stack.last_mut()
            {
                let mut progress_updates = Vec::new();
                if let Some(ref progress_rx) = bt_state.ingest_progress_rx {
                    while let Ok((cur, tot)) = progress_rx.try_recv() {
                        progress_updates.push((cur, tot));
                    }
                }
                for (cur, tot) in progress_updates {
                    bt_state.ingest_progress = Some((cur, tot));
                }

                let ingest_result = if let Some(ref rx) = bt_state.ingest_rx {
                    rx.try_recv().ok()
                } else {
                    None
                };

                let (result, bt_path, data_root) = if let Some(result) = ingest_result {
                    bt_state.ingest_rx = None;
                    bt_state.ingest_progress_rx = None;
                    bt_state.ingest_progress = None;
                    bt_state.loading_state = LoadingState::Idle;
                    (
                        Some(result),
                        bt_state.beamtime_path.clone(),
                        bt_state.data_root.clone(),
                    )
                } else {
                    (None, PathBuf::new(), None)
                };

                (result, bt_path, data_root)
            } else {
                (None, PathBuf::new(), None)
            }
        };

        // Now handle the result without holding the mutable borrow
        if let Some(result) = ingest_result {
            match result {
                Ok(()) => {
                    // Register beamtime
                    let db_path = resolve_catalog_path(&bt_path_cloned);
                    let file_count = catalog_file_count(&db_path, Some(&bt_path_cloned)).ok();
                    let _ = register_beamtime(&bt_path_cloned, file_count);

                    // Reload from DATA catalog and apply it
                    if let Some((groups, samples, tags, scans)) = self.load_beamtime_from_catalog(
                        &bt_path_cloned,
                        data_root_cloned.as_ref(),
                        &self.navigator.catalog,
                    ) {
                        if let Some(super::navigator::ScreenState::Beamtime(bt_state)) =
                            self.navigator.stack.last_mut()
                        {
                            bt_state.all_groups = groups;
                            bt_state.samples = samples;
                            bt_state.tags = tags;
                            bt_state.scans = scans;
                            bt_state.has_catalog = true;
                            bt_state
                                .sample_state
                                .select(if !bt_state.samples.is_empty() {
                                    Some(0)
                                } else {
                                    None
                                });
                            bt_state.tag_state.select(if !bt_state.tags.is_empty() {
                                Some(0)
                            } else {
                                None
                            });
                            bt_state
                                .scan_list_state
                                .select(if !bt_state.scans.is_empty() {
                                    Some(0)
                                } else {
                                    None
                                });
                            // Inline refresh to avoid self borrow conflict
                            bt_state.filtered_groups = Self::filter_groups(
                                &bt_state.all_groups,
                                &bt_state.selected_samples,
                                &bt_state.selected_tags,
                                &bt_state.selected_scans,
                                &bt_state.search_query,
                            );
                            if !bt_state.filtered_groups.is_empty() {
                                bt_state.table_state.select(Some(0));
                            }
                            bt_state.set_status("Catalog indexed.".to_string(), false);
                        }
                    }

                    self.update_explorer_beamtime_status(&bt_path_cloned);
                    self.sync_explorer_catalog_columns();
                }
                Err(e) => {
                    if let Some(super::navigator::ScreenState::Beamtime(bt_state)) =
                        self.navigator.stack.last_mut()
                    {
                        let path = bt_state
                            .pending_ingest_path
                            .take()
                            .unwrap_or_else(|| bt_state.beamtime_path.clone());
                        bt_state
                            .set_app_error(super::error::TuiError::ingest_failed(path, e, None));
                    }
                }
            }
            self.needs_redraw = true;
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
                    self.reload_from_catalog(false);
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
        let n = self.scans.len();
        if let Some(i) = self.scan_list_state.selected() {
            if i >= n && n > 0 {
                self.scan_list_state.select(Some(n - 1));
            } else if n == 0 {
                self.scan_list_state.select(None);
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
        let idx = FOCUS_ORDER
            .iter()
            .position(|&f| f == self.focus)
            .unwrap_or(0);
        self.focus = FOCUS_ORDER[(idx + 1) % FOCUS_ORDER.len()];
    }

    pub fn focus_prev(&mut self) {
        let idx = FOCUS_ORDER
            .iter()
            .position(|&f| f == self.focus)
            .unwrap_or(0);
        self.focus = FOCUS_ORDER[(idx + FOCUS_ORDER.len() - 1) % FOCUS_ORDER.len()];
    }

    pub fn focus_sample(&mut self) {
        self.focus = Focus::SampleList;
    }

    pub fn focus_tag(&mut self) {
        self.focus = Focus::TagList;
    }

    pub fn focus_scan(&mut self) {
        self.focus = Focus::ScanList;
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
                    self.sample_state.select(Some(cmp::min(i + 1, len - 1)));
                }
            }
            Focus::TagList => {
                let len = self.tags.len();
                if len > 0 {
                    let i = self.tag_state.selected().unwrap_or(0);
                    self.tag_state.select(Some(cmp::min(i + 1, len - 1)));
                }
            }
            Focus::ScanList => {
                let len = self.scans.len();
                if len > 0 {
                    let i = self.scan_list_state.selected().unwrap_or(0);
                    self.scan_list_state.select(Some(cmp::min(i + 1, len - 1)));
                }
            }
            Focus::Table => {
                if let Some(idx) = self.expanded_table_row {
                    if let Some(g) = self.group_at_display_index(idx) {
                        let file_count = g.file_rows.len();
                        if file_count > 0 {
                            let cur = self.expanded_selected_file_index.unwrap_or(0);
                            let next = cmp::min(cur + 1, file_count - 1);
                            self.expanded_selected_file_index = Some(next);
                            let visible = self.last_expanded_files_visible.max(1);
                            let max_offset = file_count.saturating_sub(visible);
                            if next >= self.expanded_files_scroll_offset + visible {
                                self.expanded_files_scroll_offset =
                                    cmp::min((next + 1).saturating_sub(visible), max_offset);
                            }
                            self.needs_redraw = true;
                            return;
                        }
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
                    self.sample_state
                        .select(Some(if i == 0 { 0 } else { i - 1 }));
                }
            }
            Focus::TagList => {
                let len = self.tags.len();
                if len > 0 {
                    let i = self.tag_state.selected().unwrap_or(0);
                    self.tag_state.select(Some(if i == 0 { 0 } else { i - 1 }));
                }
            }
            Focus::ScanList => {
                let len = self.scans.len();
                if len > 0 {
                    let i = self.scan_list_state.selected().unwrap_or(0);
                    self.scan_list_state
                        .select(Some(if i == 0 { 0 } else { i - 1 }));
                }
            }
            Focus::Table => {
                if let Some(idx) = self.expanded_table_row {
                    if let Some(g) = self.group_at_display_index(idx) {
                        let file_count = g.file_rows.len();
                        if file_count > 0 {
                            let cur = self.expanded_selected_file_index.unwrap_or(0);
                            let prev = cur.saturating_sub(1);
                            self.expanded_selected_file_index = Some(prev);
                            if prev < self.expanded_files_scroll_offset {
                                self.expanded_files_scroll_offset = prev;
                            }
                            self.needs_redraw = true;
                            return;
                        }
                    }
                }
                let len = self.filtered_groups.len();
                if len > 0 {
                    let i = self.table_state.selected().unwrap_or(0);
                    self.table_state
                        .select(Some(if i == 0 { 0 } else { i - 1 }));
                }
            }
            _ => {}
        }
    }

    pub fn set_sample_list_offset(&mut self, pos: usize) {
        let len = self.samples.len();
        let o = len.saturating_sub(1).min(pos);
        *self.sample_state.offset_mut() = o;
        if let Some(sel) = self.sample_state.selected() {
            if sel < o {
                self.sample_state.select(Some(o));
            } else if len > 0 && sel >= len {
                self.sample_state.select(Some(len - 1));
            }
        }
        self.needs_redraw = true;
    }

    pub fn set_tag_list_offset(&mut self, pos: usize) {
        let len = self.tags.len();
        let o = len.saturating_sub(1).min(pos);
        *self.tag_state.offset_mut() = o;
        if let Some(sel) = self.tag_state.selected() {
            if sel < o {
                self.tag_state.select(Some(o));
            } else if len > 0 && sel >= len {
                self.tag_state.select(Some(len - 1));
            }
        }
        self.needs_redraw = true;
    }

    pub fn set_scan_list_offset(&mut self, pos: usize) {
        let len = self.scans.len();
        let o = len.saturating_sub(1).min(pos);
        *self.scan_list_state.offset_mut() = o;
        if let Some(sel) = self.scan_list_state.selected() {
            if sel < o {
                self.scan_list_state.select(Some(o));
            } else if len > 0 && sel >= len {
                self.scan_list_state.select(Some(len - 1));
            }
        }
        self.needs_redraw = true;
    }

    pub fn set_table_offset(&mut self, pos: usize) {
        let len = self.filtered_groups.len();
        let o = len.saturating_sub(1).min(pos);
        *self.table_state.offset_mut() = o;
        if let Some(sel) = self.table_state.selected() {
            if sel >= len && len > 0 {
                self.table_state.select(Some(len - 1));
            }
        }
        self.needs_redraw = true;
    }

    pub fn set_expanded_files_offset(&mut self, pos: usize) {
        let max_offset = self
            .expanded_table_row
            .and_then(|i| self.group_at_display_index(i))
            .map(|g| {
                g.file_rows
                    .len()
                    .saturating_sub(self.last_expanded_files_visible.max(1))
            })
            .unwrap_or(0);
        self.expanded_files_scroll_offset = pos.min(max_offset);
        self.needs_redraw = true;
    }

    pub fn toggle_filter(&mut self) {
        match self.focus {
            Focus::SampleList => {
                if let Some(i) = self.sample_state.selected() {
                    let sample = self.samples.get(i).cloned();
                    if let Some(s) = sample {
                        if self.selected_samples.contains(&s) {
                            self.selected_samples.remove(&s);
                        } else {
                            self.selected_samples.insert(s);
                        }
                        self.refresh_filtered();
                    }
                }
            }
            Focus::TagList => {
                if let Some(i) = self.tag_state.selected() {
                    let tag = self.tags.get(i).cloned();
                    if let Some(t) = tag {
                        if self.selected_tags.contains(&t) {
                            self.selected_tags.remove(&t);
                        } else {
                            self.selected_tags.insert(t);
                        }
                        self.refresh_filtered();
                    }
                }
            }
            Focus::ScanList => {
                if let Some(i) = self.scan_list_state.selected() {
                    let scan = self.scans.get(i).map(|(n, _)| *n);
                    if let Some(n) = scan {
                        if self.selected_scans.contains(&n) {
                            self.selected_scans.remove(&n);
                        } else {
                            self.selected_scans.insert(n);
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
            .and_then(|i| self.group_at_display_index(i))
            .map(|r| r.sample.clone())
            .unwrap_or_default();
        self.mode = AppMode::RenameSample;
    }

    pub fn set_mode_retag(&mut self) {
        self.rename_retag_buffer = self
            .table_state
            .selected()
            .and_then(|i| self.group_at_display_index(i))
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
            .and_then(|i| self.group_at_display_index(i));
        let Some(row) = row else { return };
        if !self.has_catalog {
            return;
        }
        let db_path = resolve_catalog_path(Path::new(&self.current_root));
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
                file_row.scan_number,
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
                self.set_status(format!("Rename failed: {}", e), true);
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
        self.reload_from_catalog(false);
        self.set_status(format!("Renamed {} file(s).", renamed), false);
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
            Focus::ScanList if !self.scans.is_empty() => {
                self.scan_list_state.select(Some(0));
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
            Focus::ScanList if !self.scans.is_empty() => {
                let n = self.scans.len() - 1;
                self.scan_list_state.select(Some(n));
            }
            Focus::Table if !self.filtered_groups.is_empty() => {
                let n = self.filtered_groups.len() - 1;
                self.table_state.select(Some(n));
            }
            _ => {}
        }
        self.needs_redraw = true;
    }

    // ---- Explorer helper methods ----

    pub fn explorer_state(&self) -> Option<&super::explorer::ExplorerState> {
        match self.navigator.stack.last() {
            Some(super::navigator::ScreenState::Explorer(state)) => Some(state),
            _ => None,
        }
    }

    pub fn explorer_state_mut(&mut self) -> Option<&mut super::explorer::ExplorerState> {
        match self.navigator.stack.last_mut() {
            Some(super::navigator::ScreenState::Explorer(state)) => Some(state),
            _ => None,
        }
    }

    fn explorer_layer_mut(&mut self) -> Option<&mut super::explorer::ExplorerState> {
        self.navigator.stack.iter_mut().find_map(|s| {
            if let super::navigator::ScreenState::Explorer(e) = s {
                Some(e)
            } else {
                None
            }
        })
    }

    pub fn modal_state(&self) -> Option<&super::explorer::modal::ModalState> {
        match self.navigator.stack.last() {
            Some(super::navigator::ScreenState::ConfigModal(state)) => Some(state),
            _ => None,
        }
    }

    pub fn modal_state_mut(&mut self) -> Option<&mut super::explorer::modal::ModalState> {
        match self.navigator.stack.last_mut() {
            Some(super::navigator::ScreenState::ConfigModal(state)) => Some(state),
            _ => None,
        }
    }

    pub fn explorer_list_down(&mut self) {
        if let Some(state) = self.explorer_state_mut() {
            state.move_down();
            self.needs_redraw = true;
        }
    }

    pub fn explorer_list_up(&mut self) {
        if let Some(state) = self.explorer_state_mut() {
            state.move_up();
            self.needs_redraw = true;
        }
    }

    pub fn explorer_enter(&mut self) {
        if let Some(state) = self.explorer_state_mut() {
            if let Some(entry) = state.selected_entry() {
                // Check if this is a Beamtime entry that needs indexing
                if entry.kind == super::explorer::EntryKind::Beamtime
                    && (entry.catalog_status == super::explorer::CatalogStatus::NotIndexed
                        || entry.catalog_status == super::explorer::CatalogStatus::Stale)
                {
                    let bt_path = entry.path.clone();
                    let data_root = state.data_root.clone();
                    // Get experimentalist name from parent directory name
                    let experimentalist = bt_path
                        .parent()
                        .and_then(|p| p.file_name())
                        .and_then(|n| n.to_str())
                        .map(|s| s.to_string());

                    self.start_beamtime_ingest(bt_path, Some(data_root), experimentalist);
                } else {
                    let path = entry.path.clone();
                    state.navigate_into(path);
                }
            }
            self.needs_redraw = true;
        }
        #[cfg(feature = "catalog")]
        {
            if self.explorer_state().is_some() {
                self.explorer_after_nav();
            }
        }
    }

    pub fn explorer_back(&mut self) {
        if let Some(state) = self.explorer_state_mut() {
            state.navigate_up();
            self.needs_redraw = true;
        }
        #[cfg(feature = "catalog")]
        self.explorer_after_nav();
    }

    pub fn explorer_resolve(&mut self) {
        if let Some(state) = self.explorer_state() {
            if let Some(entry) = state.selected_entry() {
                if entry.kind == super::explorer::EntryKind::Experimentalist {
                    let expt_name = entry.name.clone();
                    let data_root = state.data_root.clone();
                    let policy = state.layout_policy.policy_for(&entry.name).clone();

                    let modal =
                        super::explorer::modal::ModalState::new(&expt_name, data_root, &policy);
                    self.navigator
                        .stack
                        .push(super::navigator::ScreenState::ConfigModal(modal));
                    self.needs_redraw = true;
                }
            }
        }
    }

    /// Phase 4: Start auto-ingest of a beamtime directory.
    /// Spawns background ingest thread and pushes BeamtimeState with ingest in progress.
    #[cfg(feature = "catalog")]
    fn start_beamtime_ingest(
        &mut self,
        bt_path: PathBuf,
        data_root: Option<PathBuf>,
        experimentalist: Option<String>,
    ) {
        let cancel_flag = Arc::new(AtomicBool::new(false));
        let (progress_tx, progress_rx) = mpsc::channel::<(u32, u32)>();
        let (done_tx, done_rx) = mpsc::channel::<Result<(), String>>();

        let bt_dir = bt_path.clone();
        let data_root_clone = data_root.clone();
        let experimentalist_clone = experimentalist.clone();
        let cancel_clone = Arc::clone(&cancel_flag);
        let default_headers: Vec<String> = pyref::catalog::DEFAULT_INGEST_HEADER_ITEMS
            .iter()
            .map(|s| s.to_string())
            .collect();
        let cfg = super::config::TuiConfig::load_or_default();
        let parallelism = pyref::catalog::IngestParallelism::from_options_or_env(
            cfg.ingest_worker_threads,
            cfg.ingest_resource_fraction,
        );

        thread::spawn(move || {
            let result = pyref::catalog::ingest_beamtime_with_context(
                &bt_dir,
                data_root_clone.as_deref(),
                experimentalist_clone.as_deref(),
                &default_headers,
                true, // incremental
                Some(progress_tx),
                parallelism,
                pyref::catalog::IngestSelection::default(),
                Some(cancel_clone),
            );
            let _ = done_tx.send(result.map(|_| ()).map_err(|e| e.to_string()));
        });

        // Create BeamtimeState for the ingest-in-progress screen
        let mut bt_state = super::navigator::BeamtimeState {
            all_groups: Vec::new(),
            filtered_groups: Vec::new(),
            samples: Vec::new(),
            tags: Vec::new(),
            scans: Vec::new(),
            selected_samples: HashSet::new(),
            selected_tags: HashSet::new(),
            selected_scans: HashSet::new(),
            sample_state: ListState::default(),
            tag_state: ListState::default(),
            scan_list_state: ListState::default(),
            table_state: TableState::default(),
            sample_scroll_state: ScrollbarState::new(0),
            tag_scroll_state: ScrollbarState::new(0),
            scan_scroll_state: ScrollbarState::new(0),
            table_scroll_state: ScrollbarState::new(0),
            expanded_files_scroll_state: ScrollbarState::new(0),
            table_sort_column: None,
            table_sort_ordering: None,
            expanded_table_row: None,
            expanded_selected_file_index: None,
            expanded_files_scroll_offset: 0,
            expanded_files_sort_column: None,
            expanded_files_sort_ordering: None,
            last_expanded_files_visible: 5,
            focus: Focus::Nav,
            mode: AppMode::Normal,
            current_root: bt_path.to_string_lossy().into_owned(),
            search_query: String::new(),
            has_catalog: false,
            loading_state: LoadingState::IngestingDirectory,
            spinner_frame: 0,
            ingest_rx: Some(done_rx),
            ingest_progress: None,
            ingest_progress_rx: Some(progress_rx),
            pending_ingest_path: Some(bt_path.clone()),
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
            scrollbar_drag: None,
            app_error: None,
            app_warning: None,
            keybind_bar_lines: 0,
            data_root: data_root.clone(),
            experimentalist: experimentalist.clone(),
            beamtime_path: bt_path,
            cancel_flag: Some(cancel_flag),
        };

        // If data_root is present, try to load from DATA catalog immediately
        if let Some(ref root) = data_root {
            if let Some((groups, samples, tags, scans)) = self.load_beamtime_from_catalog(
                &bt_state.beamtime_path,
                Some(root),
                &self.navigator.catalog,
            ) {
                bt_state.all_groups = groups;
                bt_state.samples = samples;
                bt_state.tags = tags;
                bt_state.scans = scans;
                bt_state.has_catalog = true;
                bt_state
                    .sample_state
                    .select(if !bt_state.samples.is_empty() {
                        Some(0)
                    } else {
                        None
                    });
                bt_state.tag_state.select(if !bt_state.tags.is_empty() {
                    Some(0)
                } else {
                    None
                });
                bt_state
                    .scan_list_state
                    .select(if !bt_state.scans.is_empty() {
                        Some(0)
                    } else {
                        None
                    });
                self.refresh_filtered_beamtime(&mut bt_state);
            }
        }

        self.navigator
            .stack
            .push(super::navigator::ScreenState::Beamtime(bt_state));
        self.needs_redraw = true;
    }

    /// Load beamtime data from DATA-level catalog using CatalogHandle.
    #[cfg(feature = "catalog")]
    fn load_beamtime_from_catalog(
        &self,
        beamtime_path: &PathBuf,
        data_root: Option<&PathBuf>,
        catalog: &CatalogHandle,
    ) -> Option<(
        Vec<GroupedProfileRow>,
        Vec<String>,
        Vec<String>,
        Vec<(u32, String)>,
    )> {
        if data_root.is_none() {
            return None;
        }
        let entries = catalog.list_beamtime_entries_v2(beamtime_path)?;
        let rows = catalog.query_scan_points(beamtime_path, None);
        let groups = build_groups_from_files(rows);
        let scans: Vec<(u32, String)> = entries
            .scans
            .into_iter()
            .map(|(n, s)| (n as u32, s))
            .collect();
        Some((groups, entries.samples, entries.tags, scans))
    }

    /// Helper to refresh filtered groups for a BeamtimeState (used during ingest).
    fn refresh_filtered_beamtime(&self, state: &mut super::navigator::BeamtimeState) {
        state.filtered_groups = Self::filter_groups(
            &state.all_groups,
            &state.selected_samples,
            &state.selected_tags,
            &state.selected_scans,
            &state.search_query,
        );
        if !state.filtered_groups.is_empty() {
            state.table_state.select(Some(0));
        }
    }

    #[cfg(feature = "catalog")]
    pub fn sync_explorer_catalog_columns(&mut self) {
        let db_path = self.navigator.catalog.db_path.clone();
        let handle = CatalogHandle::new(db_path);
        let Some(ex) = self.explorer_layer_mut() else {
            return;
        };
        let data_root = ex.data_root.clone();
        let current = ex.current_dir.clone();
        if current == data_root {
            let metas = handle.list_experimentalists(&data_root);
            let map: HashMap<String, pyref::catalog::ExptMeta> =
                metas.into_iter().map(|m| (m.name.clone(), m)).collect();
            for e in &mut ex.entries {
                if e.kind != super::explorer::EntryKind::Experimentalist {
                    continue;
                }
                if let Some(meta) = map.get(&e.name) {
                    e.beamtime_count = Some(meta.beamtime_count);
                    let db_fits = meta.fits_count;
                    e.fits_count = Some(e.fits_count.unwrap_or(0).max(db_fits));
                    e.catalog_status = if meta.beamtime_count > 0 || db_fits > 0 {
                        super::explorer::CatalogStatus::Indexed
                    } else {
                        super::explorer::CatalogStatus::NotIndexed
                    };
                }
            }
        } else if current
            .parent()
            .map(|p| p == data_root.as_path())
            .unwrap_or(false)
        {
            let Some(expt_name) = current.file_name().and_then(|n| n.to_str()) else {
                ex.apply_sort();
                self.needs_redraw = true;
                return;
            };
            let metas = handle.list_beamtimes(&data_root, expt_name);
            let map: HashMap<PathBuf, pyref::catalog::BeamtimeMeta> =
                metas.into_iter().map(|m| (m.path.clone(), m)).collect();
            for e in &mut ex.entries {
                if e.kind != super::explorer::EntryKind::Beamtime {
                    continue;
                }
                if let Some(meta) = map.get(&e.path) {
                    e.fits_count = meta.fits_count.or(e.fits_count);
                }
                let st = handle.catalog_status(&e.path);
                e.catalog_status = match st {
                    pyref::catalog::DbCatalogStatus::Indexed => {
                        super::explorer::CatalogStatus::Indexed
                    }
                    pyref::catalog::DbCatalogStatus::Stale => super::explorer::CatalogStatus::Stale,
                    pyref::catalog::DbCatalogStatus::NotIndexed => {
                        super::explorer::CatalogStatus::NotIndexed
                    }
                };
            }
        }
        ex.apply_sort();
        self.needs_redraw = true;
    }

    #[cfg(feature = "catalog")]
    fn restart_explorer_fs_scan(&mut self) {
        let snapshot = self
            .navigator
            .stack
            .iter()
            .find_map(|s| {
                if let super::navigator::ScreenState::Explorer(ex) = s {
                    Some(ex)
                } else {
                    None
                }
            })
            .map(|ex| {
                let at_root = ex.current_dir == ex.data_root;
                let paths: Vec<PathBuf> = ex
                    .entries
                    .iter()
                    .filter(|e| e.kind == super::explorer::EntryKind::Experimentalist)
                    .map(|e| e.path.clone())
                    .collect();
                (
                    at_root,
                    ex.data_root.clone(),
                    ex.layout_policy.clone(),
                    paths,
                )
            });
        let Some((at_root, data_root, layout, paths)) = snapshot else {
            self.explorer_fs_rx = None;
            return;
        };
        if !at_root || paths.is_empty() {
            self.explorer_fs_rx = None;
            return;
        }
        let (tx, rx) = mpsc::channel();
        self.explorer_fs_rx = Some(rx);
        thread::spawn(move || {
            for path in paths {
                let (bt, fits) = super::explorer::heuristic::scan_experimentalist_subtree(
                    &path, &data_root, &layout,
                );
                let _ = tx.send(ExplorerFsUpdate {
                    path,
                    beamtime_count: bt,
                    fits_count: fits,
                });
            }
        });
    }

    #[cfg(feature = "catalog")]
    pub fn try_recv_explorer_fs_updates(&mut self) {
        let Some(rx) = self.explorer_fs_rx.as_ref() else {
            return;
        };
        let mut batch = Vec::new();
        while let Ok(u) = rx.try_recv() {
            batch.push(u);
        }
        if batch.is_empty() {
            return;
        }
        let Some(ex) = self.explorer_layer_mut() else {
            return;
        };
        if ex.current_dir != ex.data_root {
            return;
        }
        for u in batch {
            if let Some(e) = ex.entries.iter_mut().find(|e| e.path == u.path) {
                e.beamtime_count = Some(u.beamtime_count);
                let prev = e.fits_count.unwrap_or(0);
                e.fits_count = Some(prev.max(u.fits_count));
            }
        }
        ex.apply_sort();
        self.needs_redraw = true;
    }

    #[cfg(feature = "catalog")]
    pub fn explorer_after_nav(&mut self) {
        self.sync_explorer_catalog_columns();
        self.restart_explorer_fs_scan();
    }

    /// Update Explorer status when beamtime ingest completes (direct stack mutation).
    fn update_explorer_beamtime_status(&mut self, bt_path: &PathBuf) {
        // Find the Explorer state in the stack and update it
        for state in self.navigator.stack.iter_mut() {
            if let super::navigator::ScreenState::Explorer(ref mut explorer) = state {
                for entry in explorer.entries.iter_mut() {
                    if entry.path == *bt_path {
                        entry.catalog_status = super::explorer::CatalogStatus::Indexed;
                    }
                }
                break;
            }
        }
    }
}

impl Deref for App {
    type Target = super::navigator::BeamtimeState;

    fn deref(&self) -> &Self::Target {
        self.beamtime_state()
    }
}

impl DerefMut for App {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.beamtime_state_mut()
    }
}

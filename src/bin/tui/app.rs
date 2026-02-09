use fuzzy_matcher::skim::SkimMatcherV2;
use fuzzy_matcher::FuzzyMatcher;
use ratatui::widgets::{ListState, TableState};
use std::cmp;
use std::collections::HashSet;
use std::fs;
use std::path::Path;

use pyref::errors::FitsLoaderError;
use pyref::loader::read_experiment_metadata;

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
}

fn common_prefix_slice(names: &[String]) -> String {
    if names.is_empty() {
        return String::new();
    }
    let first = names[0].as_str();
    let mut len = first.len();
    for name in names.iter().skip(1) {
        let n = name.len();
        len = cmp::min(len, n);
        for (i, (a, b)) in first.bytes().zip(name.bytes()).enumerate() {
            if i >= len {
                break;
            }
            if (a as char).to_ascii_lowercase() != (b as char).to_ascii_lowercase() {
                len = i;
                break;
            }
        }
    }
    first[..len].to_string()
}

fn load_catalog_from_path(
    dir: &str,
) -> Result<(Vec<ProfileRow>, Vec<String>, Vec<String>, Vec<(u32, String)>), FitsLoaderError> {
    let header_items: Vec<String> = vec![];
    let df = read_experiment_metadata(dir, &header_items)?;
    let n = df.height();
    let sample_name_series = df.column("sample_name").map_err(|_| FitsLoaderError::NoData)?;
    let sample_names = sample_name_series.str().map_err(|_| FitsLoaderError::NoData)?;
    let tag_series = df.column("tag").map_err(|_| FitsLoaderError::NoData)?;
    let tags_str = tag_series.str().map_err(|_| FitsLoaderError::NoData)?;
    let exp_series = df.column("experiment_number").map_err(|_| FitsLoaderError::NoData)?;
    let exp_nums = exp_series.i64().map_err(|_| FitsLoaderError::NoData)?;
    let mut profiles = Vec::with_capacity(n);
    let mut samples_set: HashSet<String> = HashSet::new();
    let mut tags_set: HashSet<String> = HashSet::new();
    let mut experiments_set: HashSet<(u32, String)> = HashSet::new();
    for i in 0..n {
        let sample = sample_names.get(i).unwrap_or("").to_string();
        let tag = tags_str.get(i).unwrap_or("-").to_string();
        let exp = exp_nums.get(i).unwrap_or(0);
        let exp_u = exp.max(0) as u32;
        samples_set.insert(sample.clone());
        if tag != "-" {
            tags_set.insert(tag.clone());
        }
        experiments_set.insert((exp_u, format!("CCD Scan {}", exp_u)));
        profiles.push(ProfileRow {
            sample: sample.clone(),
            tag: if tag == "-" { tag } else { tag.clone() },
            energy_str: "-".to_string(),
            pol: "-".to_string(),
            q_range_str: "-".to_string(),
            data_points: 0,
            quality_placeholder: "-".to_string(),
            experiment_number: exp_u,
        });
    }
    let mut samples: Vec<String> = samples_set.into_iter().collect();
    samples.sort();
    let mut tags: Vec<String> = tags_set.into_iter().collect();
    tags.sort();
    let mut experiments: Vec<(u32, String)> = experiments_set.into_iter().collect();
    experiments.sort_by(|a, b| a.0.cmp(&b.0));
    Ok((profiles, samples, tags, experiments))
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
    ChangeDir,
}

pub struct App {
    pub all_profiles: Vec<ProfileRow>,
    pub filtered_profiles: Vec<ProfileRow>,
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
    pub focus: Focus,
    pub mode: AppMode,
    pub current_root: String,
    pub search_query: String,
    pub path_input: String,
    pub needs_redraw: bool,
    pub layout: String,
    pub keymap: String,
    #[allow(dead_code)]
    pub keybind_bar_lines: u8,
    pub theme: String,
}

impl App {
    pub fn new(current_root: String, layout: String, keymap: String, keybind_bar_lines: u8, theme: String) -> Self {
        let (all_profiles, samples, tags, experiments) =
            match load_catalog_from_path(&current_root) {
                Ok(t) => t,
                Err(_) => (vec![], vec![], vec![], vec![]),
            };
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
        let filtered = Self::filter_profiles(
            &all_profiles,
            &selected_samples,
            &selected_tags,
            &selected_experiments,
            "",
        );
        if !filtered.is_empty() {
            table_state.select(Some(0));
        }
        App {
            all_profiles: all_profiles.clone(),
            filtered_profiles: filtered,
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
            focus: Focus::SampleList,
            mode: AppMode::Normal,
            current_root,
            search_query: String::new(),
            path_input: String::new(),
            needs_redraw: true,
            layout,
            keymap,
            keybind_bar_lines,
            theme,
        }
    }

    fn filter_profiles(
        profiles: &[ProfileRow],
        samples: &HashSet<String>,
        tags: &HashSet<String>,
        experiments: &HashSet<u32>,
        search: &str,
    ) -> Vec<ProfileRow> {
        let search_lower = search.to_lowercase();
        let matcher = SkimMatcherV2::default();
        profiles
            .iter()
            .filter(|r| {
                let sample_ok = samples.is_empty() || samples.contains(&r.sample);
                let tag_ok = tags.is_empty() || tags.contains(&r.tag);
                let exp_ok = experiments.is_empty() || experiments.contains(&r.experiment_number);
                let search_ok = if search_lower.is_empty() {
                    true
                } else {
                    let haystack = format!(
                        "{} {} {} {} {}",
                        r.sample,
                        r.tag,
                        r.energy_str,
                        r.pol,
                        r.q_range_str
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
        self.filtered_profiles = Self::filter_profiles(
            &self.all_profiles,
            &self.selected_samples,
            &self.selected_tags,
            &self.selected_experiments,
            &self.search_query,
        );
        self.clamp_selections();
        self.table_state.select(None);
        if !self.filtered_profiles.is_empty() {
            self.table_state.select(Some(0));
        }
        self.needs_redraw = true;
    }

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
        let n = self.filtered_profiles.len();
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
                let len = self.filtered_profiles.len();
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
                let len = self.filtered_profiles.len();
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
        self.mode = AppMode::RenameSample;
    }

    pub fn set_mode_retag(&mut self) {
        self.mode = AppMode::EditTag;
    }

    pub fn set_mode_normal(&mut self) {
        self.mode = AppMode::Normal;
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

    pub fn set_mode_change_dir(&mut self) {
        self.mode = AppMode::ChangeDir;
        self.path_input = self.current_root.clone();
        self.needs_redraw = true;
    }

    pub fn path_push_char(&mut self, c: char) {
        if c.is_ascii() && !c.is_control() {
            self.path_input.push(c);
        }
    }

    pub fn path_pop_char(&mut self) {
        self.path_input.pop();
    }

    pub fn path_clear(&mut self) {
        self.path_input.clear();
    }

    pub fn path_autocomplete(&mut self) {
        let raw = self.path_input.trim();
        let (parent, prefix) = if raw.is_empty() {
            (Path::new("."), "")
        } else {
            let p = Path::new(raw);
            let parent = p.parent().unwrap_or(Path::new("."));
            let prefix = p
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("");
            (parent, prefix)
        };
        let Ok(entries) = fs::read_dir(parent) else {
            return;
        };
        let prefix_lower = prefix.to_lowercase();
        let mut names: Vec<String> = entries
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let name = e.file_name().to_string_lossy().into_owned();
                if prefix_lower.is_empty() || name.to_lowercase().starts_with(&prefix_lower) {
                    Some(name)
                } else {
                    None
                }
            })
            .collect();
        names.sort();
        if names.is_empty() {
            return;
        }
        let new_tail = if names.len() == 1 {
            let entry = &names[0];
            let full = parent.join(entry);
            if full.is_dir() {
                format!("{}/", entry)
            } else {
                entry.clone()
            }
        } else {
            let common = common_prefix_slice(&names);
            if common.len() > prefix.len() {
                let full = parent.join(&common);
                if full.is_dir() {
                    format!("{}/", common)
                } else {
                    common
                }
            } else {
                return;
            }
        };
        let parent_str = parent.to_string_lossy();
        let has_trailing = raw.ends_with('/');
        self.path_input = if parent_str == "." || parent_str.is_empty() {
            new_tail
        } else {
            format!("{}/{}", parent_str.trim_end_matches('/'), new_tail.trim_end_matches('/'))
        };
        if has_trailing && !self.path_input.ends_with('/') && Path::new(&self.path_input).is_dir() {
            self.path_input.push('/');
        }
        self.needs_redraw = true;
    }

    pub fn apply_path(&mut self) {
        let raw = self.path_input.trim();
        if raw.is_empty() {
            self.set_mode_normal();
            return;
        }
        let path = Path::new(raw);
        let canonical = path.canonicalize().ok().filter(|p| p.is_dir());
        let new_root = match canonical {
            Some(p) => p.to_string_lossy().into_owned(),
            None => {
                if path.is_dir() {
                    path.to_string_lossy().into_owned()
                } else {
                    self.set_mode_normal();
                    return;
                }
            }
        };
        self.current_root = new_root.clone();
        let (all_profiles, samples, tags, experiments) =
            match load_catalog_from_path(&new_root) {
                Ok(t) => t,
                Err(_) => (vec![], vec![], vec![], vec![]),
            };
        self.all_profiles = all_profiles.clone();
        self.samples = samples.clone();
        self.tags = tags.clone();
        self.experiments = experiments.clone();
        self.selected_samples.clear();
        self.selected_tags.clear();
        self.selected_experiments.clear();
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
        self.filtered_profiles = Self::filter_profiles(
            &self.all_profiles,
            &self.selected_samples,
            &self.selected_tags,
            &self.selected_experiments,
            &self.search_query,
        );
        if !self.filtered_profiles.is_empty() {
            self.table_state.select(Some(0));
        }
        self.path_input.clear();
        self.set_mode_normal();
        self.needs_redraw = true;
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
            Focus::Table if !self.filtered_profiles.is_empty() => {
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
            Focus::Table if !self.filtered_profiles.is_empty() => {
                let n = self.filtered_profiles.len() - 1;
                self.table_state.select(Some(n));
            }
            _ => {}
        }
        self.needs_redraw = true;
    }
}

use fuzzy_matcher::skim::SkimMatcherV2;
use fuzzy_matcher::FuzzyMatcher;
use ratatui::widgets::{ListState, TableState};
use std::cmp;
use std::collections::HashSet;

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

fn mock_profile_rows() -> Vec<ProfileRow> {
    vec![
        ProfileRow {
            sample: "znpc".to_string(),
            tag: "rt".to_string(),
            energy_str: "250 eV fixed".to_string(),
            pol: "s".to_string(),
            q_range_str: "0-60 deg".to_string(),
            data_points: 120,
            quality_placeholder: "-".to_string(),
            experiment_number: 11111,
        },
        ProfileRow {
            sample: "znpc".to_string(),
            tag: "rt".to_string(),
            energy_str: "283.7 eV fixed".to_string(),
            pol: "s".to_string(),
            q_range_str: "0-40 deg".to_string(),
            data_points: 80,
            quality_placeholder: "-".to_string(),
            experiment_number: 11111,
        },
        ProfileRow {
            sample: "znpc".to_string(),
            tag: "rt".to_string(),
            energy_str: "284.2 eV fixed".to_string(),
            pol: "s".to_string(),
            q_range_str: "0-45 deg".to_string(),
            data_points: 90,
            quality_placeholder: "-".to_string(),
            experiment_number: 11111,
        },
        ProfileRow {
            sample: "znpc".to_string(),
            tag: "rt".to_string(),
            energy_str: "250 eV - 320 eV".to_string(),
            pol: "s".to_string(),
            q_range_str: "1 deg".to_string(),
            data_points: 140,
            quality_placeholder: "-".to_string(),
            experiment_number: 11111,
        },
        ProfileRow {
            sample: "znpc".to_string(),
            tag: "rt".to_string(),
            energy_str: "250 eV - 320 eV".to_string(),
            pol: "s".to_string(),
            q_range_str: "2 deg".to_string(),
            data_points: 140,
            quality_placeholder: "-".to_string(),
            experiment_number: 11111,
        },
        ProfileRow {
            sample: "znpc".to_string(),
            tag: "rt".to_string(),
            energy_str: "250 eV - 320 eV".to_string(),
            pol: "s".to_string(),
            q_range_str: "5 deg".to_string(),
            data_points: 140,
            quality_placeholder: "-".to_string(),
            experiment_number: 11111,
        },
        ProfileRow {
            sample: "ps_pmma".to_string(),
            tag: "rt".to_string(),
            energy_str: "285 eV fixed".to_string(),
            pol: "p".to_string(),
            q_range_str: "0-30 deg".to_string(),
            data_points: 60,
            quality_placeholder: "-".to_string(),
            experiment_number: 81041,
        },
        ProfileRow {
            sample: "ps_pmma".to_string(),
            tag: "vacuum".to_string(),
            energy_str: "300 eV fixed".to_string(),
            pol: "s".to_string(),
            q_range_str: "0-25 deg".to_string(),
            data_points: 50,
            quality_placeholder: "-".to_string(),
            experiment_number: 81042,
        },
    ]
}

fn mock_samples() -> Vec<String> {
    vec!["znpc".to_string(), "ps_pmma".to_string()]
}

fn mock_tags() -> Vec<String> {
    vec!["rt".to_string(), "vacuum".to_string()]
}

fn mock_experiments() -> Vec<(u32, String)> {
    vec![
        (11111, "CCD Scan 11111".to_string()),
        (81041, "CCD Scan 81041".to_string()),
        (81042, "CCD Scan 81042".to_string()),
    ]
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
    pub needs_redraw: bool,
    pub layout: String,
    pub keymap: String,
    pub keybind_bar_lines: u8,
    pub theme: String,
}

impl App {
    pub fn new(current_root: String, layout: String, keymap: String, keybind_bar_lines: u8, theme: String) -> Self {
        let all_profiles = mock_profile_rows();
        let samples = mock_samples();
        let tags = mock_tags();
        let experiments = mock_experiments();
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

use ratatui::widgets::{ListState, TableState};
use std::cmp;

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
}

const FOCUS_ORDER: [Focus; 5] = [
    Focus::Nav,
    Focus::SampleList,
    Focus::TagList,
    Focus::ExperimentList,
    Focus::Table,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppMode {
    Normal,
    RenameSample,
    EditTag,
}

pub struct App {
    pub all_profiles: Vec<ProfileRow>,
    pub filtered_profiles: Vec<ProfileRow>,
    pub samples: Vec<String>,
    pub tags: Vec<String>,
    pub experiments: Vec<(u32, String)>,
    pub sample_state: ListState,
    pub tag_state: ListState,
    pub experiment_state: ListState,
    pub table_state: TableState,
    pub focus: Focus,
    pub mode: AppMode,
    pub current_root: String,
}

impl App {
    pub fn new(current_root: String) -> Self {
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
        let filtered = Self::filter_profiles(&all_profiles, None, None, None);
        if !filtered.is_empty() {
            table_state.select(Some(0));
        }
        App {
            all_profiles: all_profiles.clone(),
            filtered_profiles: filtered,
            samples,
            tags,
            experiments,
            sample_state,
            tag_state,
            experiment_state,
            table_state,
            focus: Focus::SampleList,
            mode: AppMode::Normal,
            current_root,
        }
    }

    fn filter_profiles(
        profiles: &[ProfileRow],
        sample: Option<&str>,
        tag: Option<&str>,
        experiment: Option<u32>,
    ) -> Vec<ProfileRow> {
        profiles
            .iter()
            .filter(|r| {
                sample.map_or(true, |s| r.sample == s)
                    && tag.map_or(true, |t| r.tag == t)
                    && experiment.map_or(true, |e| r.experiment_number == e)
            })
            .cloned()
            .collect()
    }

    pub fn selected_sample(&self) -> Option<String> {
        self.sample_state.selected().map(|i| self.samples.get(i).cloned()).flatten()
    }

    pub fn selected_tag(&self) -> Option<String> {
        self.tag_state.selected().map(|i| self.tags.get(i).cloned()).flatten()
    }

    pub fn selected_experiment(&self) -> Option<u32> {
        self.experiment_state.selected().map(|i| self.experiments.get(i).map(|(n, _)| *n)).flatten()
    }

    pub fn refresh_filtered(&mut self) {
        let s = self.selected_sample();
        let t = self.selected_tag();
        let e = self.selected_experiment();
        self.filtered_profiles = Self::filter_profiles(
            &self.all_profiles,
            s.as_deref(),
            t.as_deref(),
            e,
        );
        self.table_state.select(None);
        if !self.filtered_profiles.is_empty() {
            self.table_state.select(Some(0));
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

    pub fn list_down(&mut self) {
        match self.focus {
            Focus::SampleList => {
                let len = self.samples.len();
                if len > 0 {
                    let i = self.sample_state.selected().unwrap_or(0);
                    self.sample_state.select(Some((i + 1) % len));
                    self.refresh_filtered();
                }
            }
            Focus::TagList => {
                let len = self.tags.len();
                if len > 0 {
                    let i = self.tag_state.selected().unwrap_or(0);
                    self.tag_state.select(Some((i + 1) % len));
                    self.refresh_filtered();
                }
            }
            Focus::ExperimentList => {
                let len = self.experiments.len();
                if len > 0 {
                    let i = self.experiment_state.selected().unwrap_or(0);
                    self.experiment_state.select(Some((i + 1) % len));
                    self.refresh_filtered();
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
                    self.refresh_filtered();
                }
            }
            Focus::TagList => {
                let len = self.tags.len();
                if len > 0 {
                    let i = self.tag_state.selected().unwrap_or(0);
                    self.tag_state.select(Some(if i == 0 { len - 1 } else { i - 1 }));
                    self.refresh_filtered();
                }
            }
            Focus::ExperimentList => {
                let len = self.experiments.len();
                if len > 0 {
                    let i = self.experiment_state.selected().unwrap_or(0);
                    self.experiment_state.select(Some(if i == 0 { len - 1 } else { i - 1 }));
                    self.refresh_filtered();
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

    pub fn set_mode_rename(&mut self) {
        self.mode = AppMode::RenameSample;
    }

    pub fn set_mode_retag(&mut self) {
        self.mode = AppMode::EditTag;
    }

    pub fn set_mode_normal(&mut self) {
        self.mode = AppMode::Normal;
    }
}

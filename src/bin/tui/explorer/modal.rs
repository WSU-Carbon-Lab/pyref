use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    text::{Line, Span},
    widgets::{Block, Paragraph},
    Frame,
};
use regex::Regex;
use std::path::PathBuf;

use super::heuristic::ExptPolicy;
use super::EntryKind;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModalFocus {
    Depth,
    Pattern,
    Choice,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModalChoice {
    CustomPattern,
    MarkIgnored,
    AcceptDefault,
}

pub struct ModalState {
    pub target_experimentalist: String,
    pub depth_input: String,
    pub pattern_input: String,
    pub focus: ModalFocus,
    pub pattern_valid: bool,
    pub last_valid_pattern: Option<String>,
    pub choice: ModalChoice,
    pub preview_entries: Vec<(String, EntryKind)>,
    pub data_root: PathBuf,
}

impl ModalState {
    pub fn new(experimentalist: &str, data_root: PathBuf, existing: &ExptPolicy) -> Self {
        let (depth_input, pattern_input) = match existing {
            ExptPolicy::Auto => ("2".to_string(), "".to_string()),
            ExptPolicy::Custom {
                pattern: Some(pat),
                depth,
            } => (depth.to_string(), pat.clone()),
            ExptPolicy::Custom {
                pattern: None,
                depth,
            } => (depth.to_string(), "".to_string()),
            ExptPolicy::Ignored => ("2".to_string(), "".to_string()),
        };

        let mut state = ModalState {
            target_experimentalist: experimentalist.to_string(),
            depth_input,
            pattern_input,
            focus: ModalFocus::Depth,
            pattern_valid: true,
            last_valid_pattern: None,
            choice: ModalChoice::CustomPattern,
            preview_entries: Vec::new(),
            data_root,
        };

        state.update_preview();
        state
    }

    pub fn update_preview(&mut self) {
        self.preview_entries.clear();

        // Validate pattern if non-empty
        if !self.pattern_input.is_empty() {
            match Regex::new(&self.pattern_input) {
                Ok(_) => {
                    self.pattern_valid = true;
                    self.last_valid_pattern = Some(self.pattern_input.clone());
                }
                Err(_) => {
                    self.pattern_valid = false;
                }
            }
        } else {
            self.pattern_valid = true;
            self.last_valid_pattern = None;
        }

        // Load depth-2 entries from experimentalist directory
        let expt_path = self.data_root.join(&self.target_experimentalist);
        if let Ok(entries) = std::fs::read_dir(&expt_path) {
            for entry in entries.flatten() {
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

                let kind = if self.pattern_valid && !self.pattern_input.is_empty() {
                    if let Ok(regex) = Regex::new(&self.pattern_input) {
                        if regex.is_match(&name_str) {
                            EntryKind::Beamtime
                        } else {
                            EntryKind::Directory
                        }
                    } else {
                        EntryKind::Directory
                    }
                } else if self.pattern_input.is_empty() {
                    // No pattern: use default beamtime patterns
                    if super::heuristic::matches_beamtime_pattern(&name_str) {
                        EntryKind::Beamtime
                    } else {
                        EntryKind::Directory
                    }
                } else {
                    EntryKind::Directory
                };

                self.preview_entries.push((name_str, kind));
            }
        }

        // Sort by name
        self.preview_entries.sort_by(|a, b| a.0.cmp(&b.0));
    }

    pub fn on_char(&mut self, c: char) {
        if c.is_ascii() && !c.is_control() {
            match self.focus {
                ModalFocus::Depth => {
                    if c.is_ascii_digit() {
                        self.depth_input.push(c);
                    }
                }
                ModalFocus::Pattern => {
                    self.pattern_input.push(c);
                    self.update_preview();
                }
                ModalFocus::Choice => {}
            }
        }
    }

    pub fn on_backspace(&mut self) {
        match self.focus {
            ModalFocus::Depth => {
                self.depth_input.pop();
            }
            ModalFocus::Pattern => {
                self.pattern_input.pop();
                self.update_preview();
            }
            ModalFocus::Choice => {}
        }
    }

    pub fn on_tab(&mut self) {
        self.focus = match self.focus {
            ModalFocus::Depth => ModalFocus::Pattern,
            ModalFocus::Pattern => ModalFocus::Choice,
            ModalFocus::Choice => ModalFocus::Depth,
        };
    }

    pub fn on_number(&mut self, num: u8) {
        match num {
            1 => self.choice = ModalChoice::CustomPattern,
            2 => self.choice = ModalChoice::MarkIgnored,
            3 => self.choice = ModalChoice::AcceptDefault,
            _ => {}
        }
    }

    pub fn confirm(&self) -> ExptPolicy {
        match self.choice {
            ModalChoice::CustomPattern => {
                let depth = self.depth_input.parse::<u8>().unwrap_or(2);
                ExptPolicy::Custom {
                    pattern: self.last_valid_pattern.clone(),
                    depth,
                }
            }
            ModalChoice::MarkIgnored => ExptPolicy::Ignored,
            ModalChoice::AcceptDefault => ExptPolicy::Auto,
        }
    }
}

/// Render the modal dialog.
pub fn render_modal(f: &mut Frame, area: Rect, state: &ModalState, _theme: &str) {
    // Center the modal (60% width, 70% height)
    let modal_width = (area.width * 60 / 100).max(40);
    let modal_height = (area.height * 70 / 100).max(15);

    let modal_x = (area.width.saturating_sub(modal_width)) / 2;
    let modal_y = (area.height.saturating_sub(modal_height)) / 2;

    let modal_rect = Rect {
        x: area.x + modal_x,
        y: area.y + modal_y,
        width: modal_width,
        height: modal_height,
    };

    // Draw outer block
    let outer_block =
        Block::bordered().title(format!(" Configure: {} ", state.target_experimentalist));
    f.render_widget(outer_block, modal_rect);

    let inner = Rect {
        x: modal_rect.x + 1,
        y: modal_rect.y + 1,
        width: modal_rect.width.saturating_sub(2),
        height: modal_rect.height.saturating_sub(2),
    };

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Fill(1),
            Constraint::Length(4),
        ])
        .split(inner);

    // Depth input
    let depth_style = if state.focus == ModalFocus::Depth {
        ">> "
    } else {
        "   "
    };
    let depth_line = Line::from(format!(
        "{}Depth (default 2): {}",
        depth_style, state.depth_input
    ));
    f.render_widget(Paragraph::new(depth_line), chunks[0]);

    // Pattern input
    let pattern_style = if state.focus == ModalFocus::Pattern {
        ">> "
    } else {
        "   "
    };
    let pattern_color_text = if state.pattern_valid {
        state.pattern_input.clone()
    } else {
        state.pattern_input.clone()
    };
    let pattern_line = Line::from(vec![
        Span::raw(format!("{}Pattern (regex): ", pattern_style)),
        if state.pattern_valid {
            Span::raw(pattern_color_text)
        } else {
            Span::styled(
                pattern_color_text,
                ratatui::style::Style::new().fg(ratatui::style::Color::Red),
            )
        },
    ]);
    f.render_widget(Paragraph::new(pattern_line), chunks[1]);

    // Preview pane
    let mut preview_lines = Vec::new();
    preview_lines.push(Line::from(Span::raw("Preview (depth-2 folders):")));
    for (name, kind) in &state.preview_entries {
        let kind_str = match kind {
            EntryKind::Beamtime => "[Beamtime ✓]",
            EntryKind::Directory => "[Directory]",
            _ => "[?]",
        };
        preview_lines.push(Line::from(format!("  {} {}", name, kind_str)));
    }

    let preview = Paragraph::new(preview_lines).block(Block::bordered().title("Preview"));
    f.render_widget(preview, chunks[2]);

    // Choice buttons
    let choice_text = vec![
        Line::from(vec![
            if state.choice == ModalChoice::CustomPattern {
                Span::styled(
                    "[1] Custom pattern",
                    ratatui::style::Style::new().fg(ratatui::style::Color::Cyan),
                )
            } else {
                Span::raw("[1] Custom pattern")
            },
            Span::raw("  "),
            if state.choice == ModalChoice::MarkIgnored {
                Span::styled(
                    "[2] Mark ignored",
                    ratatui::style::Style::new().fg(ratatui::style::Color::Cyan),
                )
            } else {
                Span::raw("[2] Mark ignored")
            },
            Span::raw("  "),
            if state.choice == ModalChoice::AcceptDefault {
                Span::styled(
                    "[3] Accept default",
                    ratatui::style::Style::new().fg(ratatui::style::Color::Cyan),
                )
            } else {
                Span::raw("[3] Accept default")
            },
        ]),
        Line::from(Span::raw("Tab: next field  Enter: confirm  Esc: cancel")),
    ];

    let choice_pane = Paragraph::new(choice_text);
    f.render_widget(choice_pane, chunks[3]);
}

use ratatui::style::{Color, Modifier, Style};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThemeMode {
    Bw,
    C16,
    C256,
    Truecolor,
}

impl ThemeMode {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "bw" | "mono" => ThemeMode::Bw,
            "256" => ThemeMode::C256,
            "truecolor" | "24bit" | "rgb" => ThemeMode::Truecolor,
            _ => ThemeMode::C16,
        }
    }
}

pub fn focus_border_style(mode: ThemeMode) -> Style {
    match mode {
        ThemeMode::Bw => Style::new().add_modifier(Modifier::BOLD | Modifier::REVERSED),
        ThemeMode::C16 | ThemeMode::C256 | ThemeMode::Truecolor => Style::new().fg(Color::Cyan),
    }
}

pub fn highlight_style(mode: ThemeMode) -> Style {
    match mode {
        ThemeMode::Bw => Style::new().add_modifier(Modifier::REVERSED),
        ThemeMode::C16 | ThemeMode::C256 | ThemeMode::Truecolor => {
            Style::new().fg(Color::Black).bg(Color::Cyan)
        }
    }
}

pub fn header_style(mode: ThemeMode) -> Style {
    match mode {
        ThemeMode::Bw => Style::new().add_modifier(Modifier::BOLD),
        ThemeMode::C16 | ThemeMode::C256 | ThemeMode::Truecolor => {
            Style::new().fg(Color::Yellow).add_modifier(Modifier::BOLD)
        }
    }
}

pub fn row_highlight_style(mode: ThemeMode) -> Style {
    match mode {
        ThemeMode::Bw => Style::new().add_modifier(Modifier::REVERSED),
        ThemeMode::C16 | ThemeMode::C256 | ThemeMode::Truecolor => {
            Style::new().bg(Color::DarkGray).add_modifier(Modifier::BOLD)
        }
    }
}

pub fn keybind_bar_style(mode: ThemeMode) -> Style {
    match mode {
        ThemeMode::Bw => Style::new().add_modifier(Modifier::BOLD),
        ThemeMode::C16 | ThemeMode::C256 | ThemeMode::Truecolor => {
            Style::new().fg(Color::Black).bg(Color::Cyan)
        }
    }
}

pub fn search_prompt_style(mode: ThemeMode) -> Style {
    match mode {
        ThemeMode::Bw => Style::new().add_modifier(Modifier::UNDERLINED),
        ThemeMode::C16 | ThemeMode::C256 | ThemeMode::Truecolor => Style::new().fg(Color::Green),
    }
}

pub fn empty_message_style(mode: ThemeMode) -> Style {
    match mode {
        ThemeMode::Bw => Style::new().add_modifier(Modifier::ITALIC),
        ThemeMode::C16 | ThemeMode::C256 | ThemeMode::Truecolor => Style::new().fg(Color::DarkGray),
    }
}

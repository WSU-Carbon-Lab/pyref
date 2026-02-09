use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

pub const CMD_SYMBOL: &str = "\u{2318}";
#[allow(dead_code)]
pub const CTRL_SYMBOL: &str = "\u{2303}";
pub const ARROW_UP: &str = "\u{2191}";
pub const ARROW_DOWN: &str = "\u{2193}";
pub const DOUBLE_UP: &str = "\u{21C8}";
pub const DOUBLE_DOWN: &str = "\u{21CA}";

pub const TABLE_TITLE_LEFT: &str = " Reflectivity profiles [b] ";
pub const TABLE_TITLE_RIGHT: &str = " r Rename  R Retag ";

pub fn table_title_padded(width: u16) -> String {
    let w = width as usize;
    let left_len = TABLE_TITLE_LEFT.len();
    let right_len = TABLE_TITLE_RIGHT.len();
    if left_len + right_len <= w {
        let pad = w - left_len - right_len;
        format!("{}{}{}", TABLE_TITLE_LEFT, " ".repeat(pad), TABLE_TITLE_RIGHT)
    } else {
        TABLE_TITLE_LEFT.to_string()
    }
}

pub fn bottom_bar_line() -> String {
    format!(
        "---- j{} k{}  gg{} G{} --------",
        ARROW_DOWN,
        ARROW_UP,
        DOUBLE_UP,
        DOUBLE_DOWN
    )
}

pub fn search_prompt_display() -> String {
    format!(" {}K Search. ", CMD_SYMBOL)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Quit,
    FocusNext,
    FocusPrev,
    FocusSample,
    FocusTag,
    FocusExperiment,
    FocusBrowser,
    MoveDown,
    MoveUp,
    MoveFirst,
    MoveLast,
    Search,
    Cancel,
    Rename,
    Retag,
    Open,
    None,
}

pub fn from_key_event(key: KeyEvent, keymap: &str) -> Action {
    let code = key.code;
    let mods = key.modifiers;
    let ctrl = mods.contains(KeyModifiers::CONTROL);
    let shift = mods.contains(KeyModifiers::SHIFT);
    let alt = mods.contains(KeyModifiers::ALT);
    let super_ = mods.contains(KeyModifiers::SUPER);

    if keymap == "emacs" {
        if (ctrl || super_) && !shift && !alt && code == KeyCode::Char('k') {
            return Action::Search;
        }
        if ctrl && !shift && !alt {
            match code {
                KeyCode::Char('c') => return Action::Quit,
                KeyCode::Char('g') => return Action::Cancel,
                KeyCode::Char('n') => return Action::MoveDown,
                KeyCode::Char('p') => return Action::MoveUp,
                KeyCode::Char('s') => return Action::Search,
                KeyCode::Char('k') => return Action::Search,
                _ => {}
            }
        }
        if code == KeyCode::Tab {
            return if shift { Action::FocusPrev } else { Action::FocusNext };
        }
        if code == KeyCode::Down {
            return Action::MoveDown;
        }
        if code == KeyCode::Up {
            return Action::MoveUp;
        }
        if code == KeyCode::Enter {
            return Action::Open;
        }
        if code == KeyCode::Esc {
            return Action::Cancel;
        }
        if !ctrl && !alt && !super_ {
            match code {
                KeyCode::Char('s') if !shift => return Action::FocusSample,
                KeyCode::Char('t') => return if shift { Action::Retag } else { Action::FocusTag },
                KeyCode::Char('e') => return Action::FocusExperiment,
                KeyCode::Char('b') => return Action::FocusBrowser,
                KeyCode::Char('r') => return Action::Rename,
                _ => {}
            }
        }
        return Action::None;
    }

    if keymap == "vi" {
        if (ctrl || super_) && !shift && !alt && code == KeyCode::Char('k') {
            return Action::Search;
        }
        if code == KeyCode::Char('q') && mods.is_empty() {
            return Action::Quit;
        }
        if code == KeyCode::Esc || (code == KeyCode::Char('q') && ctrl) {
            return Action::Cancel;
        }
        if code == KeyCode::Tab {
            return if shift { Action::FocusPrev } else { Action::FocusNext };
        }
        if code == KeyCode::Char('j') && mods.is_empty() || code == KeyCode::Down {
            return Action::MoveDown;
        }
        if code == KeyCode::Char('k') && mods.is_empty() || code == KeyCode::Up {
            return Action::MoveUp;
        }
        if code == KeyCode::Char('g') && mods.is_empty() && !shift {
            return Action::MoveFirst;
        }
        if code == KeyCode::Char('G') || (code == KeyCode::Char('g') && shift) {
            return Action::MoveLast;
        }
        if code == KeyCode::Char('/') && mods.is_empty() {
            return Action::Search;
        }
        if code == KeyCode::Enter {
            return Action::Open;
        }
        if mods.is_empty() {
            match code {
                KeyCode::Char('s') => return Action::FocusSample,
                KeyCode::Char('t') => return Action::FocusTag,
                KeyCode::Char('e') => return Action::FocusExperiment,
                KeyCode::Char('b') => return Action::FocusBrowser,
                KeyCode::Char('r') => return Action::Rename,
                _ => {}
            }
        }
        if code == KeyCode::Char('R') && shift {
            return Action::Retag;
        }
    }

    Action::None
}

pub fn search_bar_hint(keymap: &str) -> String {
    if keymap == "emacs" {
        format!("{}K Search", CMD_SYMBOL)
    } else {
        format!("{}K or / Search", CMD_SYMBOL)
    }
}

#[allow(dead_code)]
pub fn keybind_bar_lines_vi() -> [(String, String); 12] {
    [
        ("s".to_string(), "Sample".to_string()),
        ("t".to_string(), "Tag".to_string()),
        ("e".to_string(), "Experiment".to_string()),
        ("b".to_string(), "Browser".to_string()),
        ("j/k".to_string(), "Down/Up".to_string()),
        ("gg/G".to_string(), "Top/End".to_string()),
        ("Tab".to_string(), "Focus".to_string()),
        (format!("{}K", CMD_SYMBOL), "Search".to_string()),
        ("r".to_string(), "Rename".to_string()),
        ("R".to_string(), "Retag".to_string()),
        ("Enter".to_string(), "Open".to_string()),
        ("q".to_string(), "Quit".to_string()),
    ]
}

#[allow(dead_code)]
pub fn keybind_bar_lines_emacs() -> [(String, String); 12] {
    [
        ("s".to_string(), "Sample".to_string()),
        ("t".to_string(), "Tag".to_string()),
        ("e".to_string(), "Experiment".to_string()),
        ("b".to_string(), "Browser".to_string()),
        ("^N/^P".to_string(), "Down/Up".to_string()),
        ("Tab".to_string(), "Focus".to_string()),
        (format!("{}K", CMD_SYMBOL), "Search".to_string()),
        ("r".to_string(), "Rename".to_string()),
        ("Enter".to_string(), "Open".to_string()),
        ("^G".to_string(), "Cancel".to_string()),
        (format!("{}X{}C", CTRL_SYMBOL, CTRL_SYMBOL), "Quit".to_string()),
        ("".to_string(), "".to_string()),
    ]
}

pub fn search_line_hotkeys(keymap: &str) -> String {
    let pairs = if keymap == "emacs" {
        [
            ("s", "Sample"),
            ("t", "Tag"),
            ("e", "Experiment"),
            ("b", "Browser"),
        ]
    } else {
        [
            ("s", "Sample"),
            ("t", "Tag"),
            ("e", "Experiment"),
            ("b", "Browser"),
        ]
    };
    pairs
        .iter()
        .map(|(k, d)| format!("{} {}", k, d))
        .collect::<Vec<_>>()
        .join("  ")
}

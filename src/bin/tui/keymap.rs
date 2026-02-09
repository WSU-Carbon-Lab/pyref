use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Quit,
    FocusNext,
    FocusPrev,
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

    if keymap == "emacs" {
        if ctrl && !shift && !alt {
            match code {
                KeyCode::Char('c') => return Action::Quit,
                KeyCode::Char('g') => return Action::Cancel,
                KeyCode::Char('n') => return Action::MoveDown,
                KeyCode::Char('p') => return Action::MoveUp,
                KeyCode::Char('s') => return Action::Search,
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
        if !ctrl && !alt {
            match code {
                KeyCode::Char('r') => return Action::Rename,
                KeyCode::Char('t') => return Action::Retag,
                _ => {}
            }
        }
        return Action::None;
    }

    if keymap == "vi" {
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
        if code == KeyCode::Char('r') && mods.is_empty() {
            return Action::Rename;
        }
        if code == KeyCode::Char('t') && mods.is_empty() {
            return Action::Retag;
        }
    }

    Action::None
}

pub fn keybind_bar_lines_vi() -> [(String, String); 10] {
    [
        ("j".to_string(), "Down".to_string()),
        ("k".to_string(), "Up".to_string()),
        ("gg".to_string(), "Top".to_string()),
        ("G".to_string(), "End".to_string()),
        ("Tab".to_string(), "Focus".to_string()),
        ("/".to_string(), "Search".to_string()),
        ("r".to_string(), "Rename".to_string()),
        ("t".to_string(), "Retag".to_string()),
        ("Enter".to_string(), "Open".to_string()),
        ("q".to_string(), "Quit".to_string()),
    ]
}

pub fn keybind_bar_lines_emacs() -> [(String, String); 10] {
    [
        ("^N".to_string(), "Down".to_string()),
        ("^P".to_string(), "Up".to_string()),
        ("Tab".to_string(), "Focus".to_string()),
        ("^S".to_string(), "Search".to_string()),
        ("r".to_string(), "Rename".to_string()),
        ("t".to_string(), "Retag".to_string()),
        ("Enter".to_string(), "Open".to_string()),
        ("^G".to_string(), "Cancel".to_string()),
        ("^X^C".to_string(), "Quit".to_string()),
        ("".to_string(), "".to_string()),
    ]
}

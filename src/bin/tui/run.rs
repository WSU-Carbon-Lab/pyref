use crossterm::event::{Event, KeyEventKind, MouseButton, MouseEventKind};
use ratatui::layout::Rect;
use ratatui::prelude::Backend;
use std::io;
use std::time::Duration;

use super::app::{App, OpenDirFocus};
use super::keymap::{self, Action};

#[cfg(feature = "catalog")]
fn handle_open_dir_event(app: &mut App, key: crossterm::event::KeyEvent) -> bool {
    let confirm_mod = key.modifiers.contains(crossterm::event::KeyModifiers::CONTROL)
        || key.modifiers.contains(crossterm::event::KeyModifiers::SUPER);
    let index_this_dir = key.code == crossterm::event::KeyCode::Enter && confirm_mod;
    if index_this_dir {
        app.open_dir_confirm();
        return false;
    }
    if key.code == crossterm::event::KeyCode::Char('i') && key.modifiers.is_empty() {
        app.open_dir_confirm();
        return false;
    }
    if key.code == crossterm::event::KeyCode::Char('e') && key.modifiers.is_empty()
        && app.open_dir_focus == OpenDirFocus::List
    {
        app.open_dir_focus = OpenDirFocus::PathInput;
        app.needs_redraw = true;
        return false;
    }
    let action = keymap::from_key_event(key, &app.keymap);
    match action {
        Action::Cancel => {
            if app.open_dir_focus == OpenDirFocus::PathInput {
                app.open_dir_focus = OpenDirFocus::List;
                app.needs_redraw = true;
            } else {
                app.open_dir_cancel();
            }
            return false;
        }
        Action::Open => {
            if app.open_dir_focus == OpenDirFocus::PathInput {
                app.refresh_open_dir_entries();
                app.open_dir_focus = OpenDirFocus::List;
            } else {
                app.open_dir_list_enter();
            }
            return false;
        }
        Action::MoveDown => {
            app.open_dir_list_down();
            return false;
        }
        Action::MoveUp => {
            app.open_dir_list_up();
            return false;
        }
        _ => {}
    }
    if key.code == crossterm::event::KeyCode::Tab {
        if app.open_dir_focus == OpenDirFocus::PathInput {
            app.open_dir_autocomplete();
        } else {
            app.open_dir_focus_toggle();
        }
        return false;
    }
    if action == Action::FocusNext {
        app.open_dir_focus_toggle();
        return false;
    }
    if app.open_dir_focus == OpenDirFocus::PathInput {
        if key.code == crossterm::event::KeyCode::Backspace {
            app.open_dir_path_pop();
        } else if let crossterm::event::KeyCode::Char(c) = key.code {
            if c.is_ascii() && !c.is_control() {
                app.open_dir_path_push(c);
            }
        }
    }
    false
}

pub fn run<B: Backend>(
    terminal: &mut ratatui::Terminal<B>,
    app: &mut App,
    poll_duration: Duration,
) -> io::Result<()> {
    loop {
        app.try_recv_ingest();
        app.try_recv_watcher();
        app.clear_status_if_stale();
        if app.loading_state != super::app::LoadingState::Idle {
            app.needs_redraw = true;
            app.spinner_frame = app.spinner_frame.saturating_add(1);
        }
        if app.needs_redraw {
            terminal.draw(|f| super::ui::render(f, app))?;
            app.needs_redraw = false;
        }

        let timeout = if app.loading_state != super::app::LoadingState::Idle {
            Duration::from_millis(100)
        } else {
            poll_duration
        };
        if crossterm::event::poll(timeout)? {
            let ev = crossterm::event::read()?;
            match ev {
                Event::Key(key) => {
                    if key.kind != KeyEventKind::Press {
                        continue;
                    }
                    if handle_event(app, key) {
                        return Ok(());
                    }
                    app.needs_redraw = true;
                }
                Event::Resize(_, _) => {
                    app.needs_redraw = true;
                }
                Event::Mouse(mouse_event) => {
                    if let Ok(sz) = terminal.size() {
                        if handle_mouse(app, mouse_event, sz.width, sz.height) {
                            app.needs_redraw = true;
                        }
                    }
                }
                _ => {}
            }
        }
    }
}

fn handle_mouse(
    app: &mut App,
    mouse_event: crossterm::event::MouseEvent,
    _width: u16,
    _height: u16,
) -> bool {
    if app.screen != super::app::Screen::Beamtime
        || app.mode != super::app::AppMode::Normal
        || app.open_dir_active
    {
        return false;
    }
    let Some((_, rects)) = app.last_body_rects.as_ref() else {
        return false;
    };
    let col = mouse_event.column;
    let row = mouse_event.row;
    if let MouseEventKind::Down(MouseButton::Left) = mouse_event.kind {
        if rect_contains(rects.sample_list, col, row) {
            let idx = list_index_at(rects.sample_list, row, app.samples.len());
            app.focus = super::app::Focus::SampleList;
            app.sample_state.select(Some(idx));
            if let Some(name) = app.samples.get(idx).cloned() {
                if app.selected_samples.contains(&name) {
                    app.selected_samples.remove(&name);
                } else {
                    app.selected_samples.insert(name);
                }
                app.refresh_filtered();
            }
            return true;
        }
        if rect_contains(rects.tag_list, col, row) {
            let idx = list_index_at(rects.tag_list, row, app.tags.len());
            app.focus = super::app::Focus::TagList;
            app.tag_state.select(Some(idx));
            if let Some(name) = app.tags.get(idx).cloned() {
                if app.selected_tags.contains(&name) {
                    app.selected_tags.remove(&name);
                } else {
                    app.selected_tags.insert(name);
                }
                app.refresh_filtered();
            }
            return true;
        }
        if rect_contains(rects.experiment_list, col, row) {
            let idx = list_index_at(rects.experiment_list, row, app.experiments.len());
            app.focus = super::app::Focus::ExperimentList;
            app.experiment_state.select(Some(idx));
            if let Some(&(num, _)) = app.experiments.get(idx) {
                if app.selected_experiments.contains(&num) {
                    app.selected_experiments.remove(&num);
                } else {
                    app.selected_experiments.insert(num);
                }
                app.refresh_filtered();
            }
            return true;
        }
        if rect_contains(rects.table, col, row) {
            let idx = table_row_at(rects.table, row, app.filtered_groups.len());
            app.focus = super::app::Focus::Table;
            app.table_state.select(Some(idx));
            app.expanded_table_row = if app.expanded_table_row == Some(idx) {
                None
            } else {
                Some(idx)
            };
            return true;
        }
    }
    false
}

fn rect_contains(r: Rect, col: u16, row: u16) -> bool {
    col >= r.x && col < r.x + r.width && row >= r.y && row < r.y + r.height
}

fn list_index_at(rect: Rect, row: u16, len: usize) -> usize {
    let content_y = rect.y + 1;
    if row < content_y || len == 0 {
        return 0;
    }
    let idx = (row - content_y) as usize;
    idx.min(len.saturating_sub(1))
}

fn table_row_at(rect: Rect, row: u16, len: usize) -> usize {
    let data_y = rect.y + 2;
    if row < data_y || len == 0 {
        return 0;
    }
    let idx = (row - data_y) as usize;
    idx.min(len.saturating_sub(1))
}

pub fn handle_event(app: &mut App, key: crossterm::event::KeyEvent) -> bool {
    if app.screen == super::app::Screen::Launcher {
        #[cfg(feature = "catalog")]
        if app.open_dir_active {
            return handle_open_dir_event(app, key);
        }
        let action = keymap::from_key_event(key, &app.keymap);
        let quit = match action {
            Action::Quit => true,
            Action::MoveDown => {
                app.launcher_list_down();
                false
            }
            Action::MoveUp => {
                app.launcher_list_up();
                false
            }
            Action::Open => {
                app.launcher_open_selected();
                false
            }
            _ => {
                if key.code == crossterm::event::KeyCode::Char('o') {
                    app.launcher_open_directory();
                }
                false
            }
        };
        return quit;
    }

    if app.screen == super::app::Screen::Beamtime && app.mode == super::app::AppMode::Normal {
        let action = keymap::from_key_event(key, &app.keymap);
        if action == Action::Cancel {
            if app.app_error.is_some() {
                app.clear_app_error();
                return false;
            }
            if app.app_warning.is_some() {
                app.clear_app_warning();
                return false;
            }
        }
        if let Some(ref e) = app.app_error {
            #[cfg(feature = "catalog")]
            if action == Action::IndexDirectory
                && matches!(e.kind, super::error::TuiErrorKind::IndexFailed)
            {
                app.clear_app_error();
                app.start_indexing();
                return false;
            }
        }
    }

    #[cfg(feature = "catalog")]
    if app.screen == super::app::Screen::Beamtime
        && app.mode == super::app::AppMode::Normal
        && keymap::from_key_event(key, &app.keymap) == Action::Cancel
    {
        app.go_to_launcher();
        return false;
    }

    if app.mode == super::app::AppMode::Search {
        let action = keymap::from_key_event(key, &app.keymap);
        match action {
            Action::Cancel => {
                app.set_mode_normal();
                app.search_clear();
            }
            Action::Open => {
                app.set_mode_normal();
            }
            Action::None => {
                if let crossterm::event::KeyCode::Char(c) = key.code {
                    if c.is_ascii() && !c.is_control() {
                        app.search_push_char(c);
                    }
                }
                if key.code == crossterm::event::KeyCode::Backspace {
                    app.search_pop_char();
                }
            }
            _ => {}
        }
        return false;
    }

    if matches!(
        app.mode,
        super::app::AppMode::RenameSample | super::app::AppMode::EditTag
    ) {
        use crossterm::event::KeyCode;
        match key.code {
            KeyCode::Esc => {
                app.set_mode_normal();
            }
            KeyCode::Enter => {
                app.apply_rename_retag();
            }
            KeyCode::Backspace => {
                app.rename_retag_pop_char();
            }
            KeyCode::Char(c) => {
                if c.is_ascii() && !c.is_control() {
                    app.rename_retag_push_char(c);
                }
            }
            _ => {}
        }
        return false;
    }

    if app.mode != super::app::AppMode::Normal {
        let action = keymap::from_key_event(key, &app.keymap);
        if action == Action::Cancel {
            app.set_mode_normal();
        }
        return false;
    }

    let action = keymap::from_key_event(key, &app.keymap);
    match action {
        Action::Quit => return true,
        Action::FocusNext => app.focus_next(),
        Action::FocusPrev => app.focus_prev(),
        Action::FocusSample => app.focus_sample(),
        Action::FocusTag => app.focus_tag(),
        Action::FocusExperiment => app.focus_experiment(),
        Action::FocusBrowser => app.focus_browser(),
        Action::MoveDown => app.list_down(),
        Action::MoveUp => app.list_up(),
        Action::MoveFirst => app.list_first(),
        Action::MoveLast => app.list_last(),
        Action::Search => app.set_mode_search(),
        Action::Cancel => {}
        Action::Rename => app.set_mode_rename(),
        Action::Retag => app.set_mode_retag(),
        Action::IndexDirectory => app.start_indexing(),
        Action::NavUp => app.nav_up(),
        Action::NavBack => app.nav_back(),
        Action::NavFwd => app.nav_fwd(),
        Action::Open => {
            if matches!(
                app.focus,
                super::app::Focus::SampleList
                    | super::app::Focus::TagList
                    | super::app::Focus::ExperimentList
            ) {
                app.toggle_filter();
            } else if app.focus == super::app::Focus::Table {
                if let Some(i) = app.table_state.selected() {
                    app.expanded_table_row = if app.expanded_table_row == Some(i) {
                        None
                    } else {
                        app.expanded_files_scroll_offset = 0;
                        Some(i)
                    };
                }
            }
        }
        Action::None => {}
    }
    false
}

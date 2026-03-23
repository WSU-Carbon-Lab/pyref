use crossterm::event::{Event, KeyEventKind, MouseButton, MouseEventKind};
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::prelude::Backend;
use std::io;
use std::time::Duration;

use super::app::{App, OpenDirFocus, ScrollbarDragTarget};
use super::keymap::{self, Action};

fn handle_explorer_event(app: &mut App, key: crossterm::event::KeyEvent) -> bool {
    use crossterm::event::KeyCode;

    match key.code {
        KeyCode::Up | KeyCode::Char('k') => {
            app.explorer_list_up();
            return false;
        }
        KeyCode::Down | KeyCode::Char('j') => {
            app.explorer_list_down();
            return false;
        }
        KeyCode::Right | KeyCode::Enter | KeyCode::Char('l') => {
            app.explorer_enter();
            return false;
        }
        KeyCode::Left | KeyCode::Char('h') => {
            app.explorer_back();
            return false;
        }
        KeyCode::Char('r') => {
            app.explorer_resolve();
            return false;
        }
        KeyCode::Char('x') => {
            // Placeholder for Phase 4: toggle ignore
            return false;
        }
        KeyCode::Char('/') => {
            // Placeholder for Phase 4: start filter input
            return false;
        }
        KeyCode::Char('q') | KeyCode::Esc => {
            return true; // Quit
        }
        _ => {}
    }

    false
}

fn handle_modal_event(app: &mut App, key: crossterm::event::KeyEvent) -> bool {
    use crossterm::event::KeyCode;

    match key.code {
        KeyCode::Esc => {
            // Pop the modal
            app.navigator.stack.pop();
            app.needs_redraw = true;
            return false;
        }
        KeyCode::Tab => {
            if let Some(modal) = app.modal_state_mut() {
                modal.on_tab();
                app.needs_redraw = true;
            }
            return false;
        }
        KeyCode::Backspace => {
            if let Some(modal) = app.modal_state_mut() {
                modal.on_backspace();
                app.needs_redraw = true;
            }
            return false;
        }
        KeyCode::Char(c) => {
            if c.is_ascii_digit() {
                if let Some(num) = c.to_digit(10) {
                    if let Some(modal) = app.modal_state_mut() {
                        modal.on_number(num as u8);
                        app.needs_redraw = true;
                    }
                    return false;
                }
            } else if c.is_ascii() && !c.is_control() {
                if let Some(modal) = app.modal_state_mut() {
                    modal.on_char(c);
                    app.needs_redraw = true;
                }
                return false;
            }
        }
        KeyCode::Enter => {
            if let Some(modal) = app.modal_state() {
                let policy = modal.confirm();
                let expt_name = modal.target_experimentalist.clone();
                let data_root = modal.data_root.clone();

                // Pop modal
                app.navigator.stack.pop();

                // Update layout policy
                if let Some(explorer) = app.explorer_state_mut() {
                    explorer.layout_policy.set_policy(expt_name, policy);
                    let _ = explorer.layout_policy.save(&data_root);
                    // Reload current directory to update expt_resolution flags
                    let current_dir = explorer.current_dir.clone();
                    explorer.load_dir(current_dir);
                }

                app.needs_redraw = true;
            }
            return false;
        }
        _ => {}
    }

    false
}

#[cfg(feature = "catalog")]
fn handle_open_dir_event(app: &mut App, key: crossterm::event::KeyEvent) -> bool {
    let confirm_mod = key
        .modifiers
        .contains(crossterm::event::KeyModifiers::CONTROL)
        || key
            .modifiers
            .contains(crossterm::event::KeyModifiers::SUPER);
    let index_this_dir = key.code == crossterm::event::KeyCode::Enter && confirm_mod;
    if index_this_dir {
        app.open_dir_confirm();
        return false;
    }
    if key.code == crossterm::event::KeyCode::Char('i') && key.modifiers.is_empty() {
        app.open_dir_confirm();
        return false;
    }
    if key.code == crossterm::event::KeyCode::Char('e')
        && key.modifiers.is_empty()
        && app.open_dir_focus() == OpenDirFocus::List
    {
        app.set_open_dir_focus(OpenDirFocus::PathInput);
        app.needs_redraw = true;
        return false;
    }
    let action = keymap::from_key_event(key, &app.keymap);
    match action {
        Action::Cancel => {
            if app.open_dir_focus() == OpenDirFocus::PathInput {
                app.set_open_dir_focus(OpenDirFocus::List);
                app.needs_redraw = true;
            } else {
                app.open_dir_cancel();
            }
            return false;
        }
        Action::Open => {
            if app.open_dir_focus() == OpenDirFocus::PathInput {
                app.refresh_open_dir_entries();
                app.set_open_dir_focus(OpenDirFocus::List);
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
        if app.open_dir_focus() == OpenDirFocus::PathInput {
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
    if app.open_dir_focus() == OpenDirFocus::PathInput {
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
        #[cfg(feature = "catalog")]
        app.try_recv_beamspot_updates();
        #[cfg(feature = "catalog")]
        app.try_recv_preview_cmd();
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

fn scroll_position_from_row(sb_rect: Rect, row: u16, content_len: usize, viewport: usize) -> usize {
    if sb_rect.height == 0 || content_len <= viewport {
        return 0;
    }
    let max_pos = content_len.saturating_sub(viewport);
    let row_offset = row.saturating_sub(sb_rect.y) as usize;
    let h = sb_rect.height as usize;
    let pos = if h > 0 {
        (row_offset * content_len) / h
    } else {
        0
    };
    pos.min(max_pos)
}

fn handle_mouse(
    app: &mut App,
    mouse_event: crossterm::event::MouseEvent,
    _width: u16,
    _height: u16,
) -> bool {
    if app.current_screen() != super::app::Screen::Beamtime
        || app.mode() != super::app::AppMode::Normal
        || app.open_dir_active()
    {
        return false;
    }
    let has_body_rects = app.last_body_rects.is_some();
    if !has_body_rects {
        return false;
    }
    let col = mouse_event.column;
    let row = mouse_event.row;

    if let MouseEventKind::Up(MouseButton::Left) = mouse_event.kind {
        if app.scrollbar_drag.take().is_some() {
            return true;
        }
    }

    let rects = match &app.last_body_rects {
        Some((_, r)) => r.clone(),
        None => return false,
    };

    let drag_target_opt = app.scrollbar_drag;
    if let Some(drag_target) = drag_target_opt {
        if let MouseEventKind::Drag(MouseButton::Left) = mouse_event.kind {
            let (sb_rect, content_len, viewport) = {
                match drag_target {
                    ScrollbarDragTarget::SampleList => {
                        let r = match rects.sample_scrollbar {
                            Some(rect) => rect,
                            None => return false,
                        };
                        let viewport = (rects.sample_list.height.saturating_sub(2)).max(1) as usize;
                        (r, app.samples.len(), viewport)
                    }
                    ScrollbarDragTarget::TagList => {
                        let r = match rects.tag_scrollbar {
                            Some(rect) => rect,
                            None => return false,
                        };
                        let viewport = (rects.tag_list.height.saturating_sub(2)).max(1) as usize;
                        (r, app.tags.len(), viewport)
                    }
                    ScrollbarDragTarget::ScanList => {
                        let r = match rects.scan_scrollbar {
                            Some(rect) => rect,
                            None => return false,
                        };
                        let viewport = (rects.scan_list.height.saturating_sub(2)).max(1) as usize;
                        (r, app.scans.len(), viewport)
                    }
                    ScrollbarDragTarget::Table => {
                        let r = match rects.table_scrollbar {
                            Some(rect) => rect,
                            None => return false,
                        };
                        let viewport = (rects.table.height.saturating_sub(1)).max(1) as usize;
                        (r, app.filtered_groups.len(), viewport)
                    }
                    ScrollbarDragTarget::ExpandedFiles => {
                        let r = match rects.expanded_files_scrollbar {
                            Some(rect) => rect,
                            None => return false,
                        };
                        let file_count = app
                            .expanded_table_row
                            .and_then(|i| app.group_at_display_index(i))
                            .map(|g| g.file_rows.len())
                            .unwrap_or(0);
                        (r, file_count, app.last_expanded_files_visible.max(1))
                    }
                }
            };
            let pos = scroll_position_from_row(sb_rect, row, content_len, viewport);
            match drag_target {
                ScrollbarDragTarget::SampleList => app.set_sample_list_offset(pos),
                ScrollbarDragTarget::TagList => app.set_tag_list_offset(pos),
                ScrollbarDragTarget::ScanList => app.set_scan_list_offset(pos),
                ScrollbarDragTarget::Table => app.set_table_offset(pos),
                ScrollbarDragTarget::ExpandedFiles => app.set_expanded_files_offset(pos),
            }
            return true;
        }
    }

    let is_scroll = matches!(
        mouse_event.kind,
        MouseEventKind::ScrollDown | MouseEventKind::ScrollUp
    );
    if is_scroll {
        let down = matches!(mouse_event.kind, MouseEventKind::ScrollDown);
        if rect_contains(rects.sample_list, col, row) {
            app.focus = super::app::Focus::SampleList;
            if down {
                app.list_down();
            } else {
                app.list_up();
            }
            return true;
        }
        if rect_contains(rects.tag_list, col, row) {
            app.focus = super::app::Focus::TagList;
            if down {
                app.list_down();
            } else {
                app.list_up();
            }
            return true;
        }
        if rect_contains(rects.scan_list, col, row) {
            app.focus = super::app::Focus::ScanList;
            if down {
                app.list_down();
            } else {
                app.list_up();
            }
            return true;
        }
        if rect_contains(rects.table, col, row) {
            app.focus = super::app::Focus::Table;
            if down {
                app.list_down();
            } else {
                app.list_up();
            }
            return true;
        }
        if let Some(ef) = rects.expanded_files {
            if rect_contains(ef, col, row) {
                app.focus = super::app::Focus::Table;
                if down {
                    app.list_down();
                } else {
                    app.list_up();
                }
                return true;
            }
        }
    }

    if let MouseEventKind::Down(MouseButton::Left) = mouse_event.kind {
        if let Some(sb) = rects.sample_scrollbar {
            if rect_contains(sb, col, row) {
                app.scrollbar_drag = Some(ScrollbarDragTarget::SampleList);
                return true;
            }
        }
        if let Some(sb) = rects.tag_scrollbar {
            if rect_contains(sb, col, row) {
                app.scrollbar_drag = Some(ScrollbarDragTarget::TagList);
                return true;
            }
        }
        if let Some(sb) = rects.scan_scrollbar {
            if rect_contains(sb, col, row) {
                app.scrollbar_drag = Some(ScrollbarDragTarget::ScanList);
                return true;
            }
        }
        if let Some(sb) = rects.table_scrollbar {
            if rect_contains(sb, col, row) {
                app.scrollbar_drag = Some(ScrollbarDragTarget::Table);
                return true;
            }
        }
        if let Some(sb) = rects.expanded_files_scrollbar {
            if rect_contains(sb, col, row) {
                app.scrollbar_drag = Some(ScrollbarDragTarget::ExpandedFiles);
                return true;
            }
        }

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
        if rect_contains(rects.scan_list, col, row) {
            let idx = list_index_at(rects.scan_list, row, app.scans.len());
            app.focus = super::app::Focus::ScanList;
            app.scan_list_state.select(Some(idx));
            if let Some(&(num, _)) = app.scans.get(idx) {
                if app.selected_scans.contains(&num) {
                    app.selected_scans.remove(&num);
                } else {
                    app.selected_scans.insert(num);
                }
                app.refresh_filtered();
            }
            return true;
        }
        if let Some(ef) = rects.expanded_files {
            if rect_contains(ef, col, row) {
                #[cfg(feature = "catalog")]
                if row == ef.y + 1 {
                    if let Some(layout_col) = expanded_file_header_column_at(ef, col, app) {
                        app.cycle_expanded_files_sort(layout_col);
                        return true;
                    }
                }
                if let Some(idx) = app.expanded_table_row {
                    if let Some(g) = app.group_at_display_index(idx) {
                        let file_count = g.file_rows.len();
                        let file_ix = expanded_file_index_at(
                            ef,
                            row,
                            app.expanded_files_scroll_offset,
                            file_count,
                        );
                        app.focus = super::app::Focus::Table;
                        app.expanded_selected_file_index = Some(file_ix);
                        return true;
                    }
                }
            }
        }
        if rect_contains(rects.table, col, row) {
            if row == rects.table.y {
                if let Some(layout_col) = table_header_column_at(rects.table, col) {
                    if (1..=10).contains(&layout_col) {
                        app.cycle_table_sort(layout_col - 1);
                        return true;
                    }
                }
            }
            let idx = table_row_at(rects.table, row, app.filtered_groups.len());
            app.focus = super::app::Focus::Table;
            app.table_state.select(Some(idx));
            app.expanded_table_row = if app.expanded_table_row == Some(idx) {
                app.expanded_selected_file_index = None;
                None
            } else {
                app.expanded_selected_file_index = Some(0);
                app.expanded_files_scroll_offset = 0;
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

fn expanded_file_index_at(
    rect: Rect,
    row: u16,
    scroll_offset: usize,
    file_count: usize,
) -> usize {
    let data_y = rect.y + 3;
    if row < data_y || file_count == 0 {
        return 0;
    }
    let rel = (row - data_y) as usize;
    let idx = scroll_offset + rel;
    idx.min(file_count.saturating_sub(1))
}

const TABLE_WIDTHS: [Constraint; 11] = [
    Constraint::Length(2),
    Constraint::Percentage(14),
    Constraint::Min(3),
    Constraint::Min(3),
    Constraint::Min(3),
    Constraint::Percentage(8),
    Constraint::Percentage(8),
    Constraint::Percentage(8),
    Constraint::Percentage(8),
    Constraint::Length(6),
    Constraint::Percentage(10),
];

const EXPANDED_FILES_TABLE_COLUMN_SPACING: u16 = 1;

const EXPANDED_FILES_TABLE_LAYOUT: [Constraint; 15] = [
    Constraint::Length(7),
    Constraint::Length(EXPANDED_FILES_TABLE_COLUMN_SPACING),
    Constraint::Length(7),
    Constraint::Length(EXPANDED_FILES_TABLE_COLUMN_SPACING),
    Constraint::Length(5),
    Constraint::Length(EXPANDED_FILES_TABLE_COLUMN_SPACING),
    Constraint::Length(8),
    Constraint::Length(EXPANDED_FILES_TABLE_COLUMN_SPACING),
    Constraint::Length(8),
    Constraint::Length(EXPANDED_FILES_TABLE_COLUMN_SPACING),
    Constraint::Length(10),
    Constraint::Length(EXPANDED_FILES_TABLE_COLUMN_SPACING),
    Constraint::Length(7),
    Constraint::Length(EXPANDED_FILES_TABLE_COLUMN_SPACING),
    Constraint::Length(8),
];

fn expanded_file_header_column_at(rect: Rect, col: u16, app: &App) -> Option<usize> {
    let inner_x = rect.x + 1;
    let inner_width = rect.width.saturating_sub(2);
    let need_scrollbar = app
        .expanded_table_row
        .and_then(|i| app.group_at_display_index(i))
        .map(|g| g.file_rows.len() > app.last_expanded_files_visible)
        .unwrap_or(false);
    let table_width = if need_scrollbar && inner_width > 1 {
        inner_width - 1
    } else {
        inner_width
    };
    if col < inner_x || col >= inner_x + table_width {
        return None;
    }
    let inner = Rect {
        x: inner_x,
        y: rect.y,
        width: table_width,
        height: 1,
    };
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints(EXPANDED_FILES_TABLE_LAYOUT)
        .split(inner);
    for (i, r) in chunks.iter().enumerate() {
        if col >= r.x && col < r.x + r.width {
            return Some(i / 2);
        }
    }
    None
}

fn table_header_column_at(rect: Rect, col: u16) -> Option<usize> {
    if col < rect.x || col >= rect.x + rect.width {
        return None;
    }
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints(TABLE_WIDTHS)
        .split(rect);
    for (i, r) in chunks.iter().enumerate() {
        if col >= r.x && col < r.x + r.width {
            return Some(i);
        }
    }
    None
}

pub fn handle_event(app: &mut App, key: crossterm::event::KeyEvent) -> bool {
    // Handle ConfigModal first (highest priority)
    if app.modal_state().is_some() {
        return handle_modal_event(app, key);
    }

    if app.current_screen() == super::app::Screen::Launcher {
        #[cfg(feature = "catalog")]
        if app.open_dir_active() {
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

    if app.current_screen() == super::app::Screen::Explorer {
        return handle_explorer_event(app, key);
    }

    if app.current_screen() == super::app::Screen::Beamtime && app.mode() == super::app::AppMode::Normal {
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
            if action == Action::IngestDirectory
                && matches!(e.kind, super::error::TuiErrorKind::IngestFailed)
            {
                app.clear_app_error();
                app.start_ingest();
                return false;
            }
        }
    }

    #[cfg(feature = "catalog")]
    if app.current_screen() == super::app::Screen::Beamtime
        && app.mode() == super::app::AppMode::Normal
        && keymap::from_key_event(key, &app.keymap) == Action::Cancel
    {
        app.go_to_launcher();
        return false;
    }

    if app.mode() == super::app::AppMode::Search {
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
        Action::FocusScan => app.focus_scan(),
        Action::FocusBrowser => app.focus_browser(),
        Action::MoveDown => app.list_down(),
        Action::MoveUp => app.list_up(),
        Action::MoveFirst => app.list_first(),
        Action::MoveLast => app.list_last(),
        Action::Search => app.set_mode_search(),
        Action::Cancel => {}
        Action::Rename => app.set_mode_rename(),
        Action::Retag => app.set_mode_retag(),
        Action::IngestDirectory => app.start_ingest(),
        Action::NavUp => app.nav_up(),
        Action::NavBack => app.nav_back(),
        Action::NavFwd => app.nav_fwd(),
        Action::PreviewImage => app.send_preview_path_if_selected(),
        Action::BeamPosition => app.materialize_profile_beamspots(),
        #[cfg(feature = "catalog")]
        Action::NextProblem => app.go_to_next_problem(),
        #[cfg(feature = "catalog")]
        Action::PrevProblem => app.go_to_prev_problem(),
        #[cfg(feature = "catalog")]
        Action::UseLastOkBeamspot => app.use_last_ok_beamspot(),
        #[cfg(feature = "catalog")]
        Action::UseNextOkBeamspot => app.use_next_ok_beamspot(),
        #[cfg(not(feature = "catalog"))]
        Action::NextProblem
        | Action::PrevProblem
        | Action::UseLastOkBeamspot
        | Action::UseNextOkBeamspot => {}
        Action::Open => {
            if matches!(
                app.focus,
                super::app::Focus::SampleList
                    | super::app::Focus::TagList
                    | super::app::Focus::ScanList
            ) {
                app.toggle_filter();
            } else if app.focus == super::app::Focus::Table {
                if let Some(i) = app.table_state.selected() {
                    app.expanded_table_row = if app.expanded_table_row == Some(i) {
                        app.expanded_selected_file_index = None;
                        None
                    } else {
                        app.expanded_selected_file_index = Some(0);
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

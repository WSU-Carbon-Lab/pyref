use ratatui::layout::Alignment;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Cell, List, ListItem, Padding, Paragraph, Row, Table};
use ratatui::Frame;

use super::app::{App, AppMode, Focus, GroupedProfileRow, LoadingState, OpenDirFocus, Screen};
use super::keymap::{bottom_bar_line, search_prompt_display, BROWSE_SHORTCUTS, BROWSE_TITLE};
use super::theme::ThemeMode;
use super::ReflectivityScanType;

const NAV_PATH_TRUNCATE: usize = 60;
const CIRCLE_EMPTY: &str = "\u{25CB}";
const CIRCLE_FILLED: &str = "\u{25CF}";

fn truncate_path(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    format!(
        "...{}",
        s.chars()
            .rev()
            .take(max.saturating_sub(3))
            .collect::<String>()
            .chars()
            .rev()
            .collect::<String>()
    )
}

fn theme_mode(app: &App) -> ThemeMode {
    ThemeMode::from_str(&app.theme)
}

fn layout_constraints(app: &App) -> [Constraint; 2] {
    match app.layout.as_str() {
        "sidebar" => [Constraint::Percentage(35), Constraint::Percentage(65)],
        "table" => [Constraint::Percentage(15), Constraint::Percentage(85)],
        _ => [Constraint::Percentage(22), Constraint::Percentage(78)],
    }
}

const MIN_LIST_HEIGHT: u16 = 3;
const SIDEBAR_GAPS: u16 = 2;

fn left_sidebar_list_heights(
    left_area_height: u16,
    n_sample: usize,
    n_tag: usize,
    n_scan: usize,
) -> (u16, u16, u16) {
    let available = left_area_height.saturating_sub(SIDEBAR_GAPS);
    let p_s = (n_sample as u16).saturating_add(2).max(MIN_LIST_HEIGHT);
    let p_t = (n_tag as u16).saturating_add(2).max(MIN_LIST_HEIGHT);
    let p_sc = (n_scan as u16).saturating_add(2).max(MIN_LIST_HEIGHT);
    if p_s.saturating_add(p_t).saturating_add(p_sc) <= available {
        (p_s, p_t, p_sc)
    } else {
        let base = available / 3;
        let third = available.saturating_sub(2 * base);
        (
            base.max(MIN_LIST_HEIGHT),
            base.max(MIN_LIST_HEIGHT),
            third.max(MIN_LIST_HEIGHT),
        )
    }
}

pub fn beamtime_body_rects(body_area: Rect, app: &App) -> Option<super::app::BeamtimeBodyRects> {
    if !app.has_catalog {
        return None;
    }
    let [left_pct, _] = layout_constraints(app);
    let body_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([left_pct, Constraint::Length(1), Constraint::Min(0)])
        .split(body_area);
    let left_area = body_chunks[0];
    let right_area = body_chunks[2];
    let (s_h, t_h, sc_h) = left_sidebar_list_heights(
        left_area.height,
        app.samples.len(),
        app.tags.len(),
        app.scans.len(),
    );
    let left_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(s_h),
            Constraint::Length(1),
            Constraint::Length(t_h),
            Constraint::Length(1),
            Constraint::Length(sc_h),
        ])
        .split(left_area);
    let right_block = Block::bordered().title(BROWSE_TITLE);
    let right_inner = right_block.inner(right_area);
    let right_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Fill(1),
            Constraint::Length(3),
        ])
        .split(right_inner);
    let table_area = right_chunks[1];
    Some(super::app::BeamtimeBodyRects {
        sample_list: left_chunks[0],
        tag_list: left_chunks[2],
        scan_list: left_chunks[4],
        table: table_area,
    })
}

const LAUNCHER_HINTS: &str = " Enter open  o add directory  q quit ";

pub fn render(frame: &mut Frame, app: &mut App) {
    let area = frame.area();
    let theme = theme_mode(app);

    let outer_block = Block::bordered().padding(Padding::new(1, 1, 1, 1));
    let inner = outer_block.inner(area);

    let bottom_height = if app.app_error.is_some() || app.app_warning.is_some() {
        2
    } else {
        1
    };
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Fill(1),
            Constraint::Length(bottom_height),
        ])
        .split(inner);

    let nav_area = chunks[0];
    let body_area = chunks[1];
    let bottom_area = chunks[2];

    frame.render_widget(outer_block, area);
    if app.screen == Screen::Launcher {
        #[cfg(feature = "catalog")]
        render_launcher_nav(frame, nav_area, app, theme);
        render_launcher_body(frame, app, body_area, theme);
        render_launcher_bottom(frame, bottom_area, theme);
        #[cfg(feature = "catalog")]
        if app.open_dir_active {
            render_open_dir_modal(frame, app, inner, theme);
        }
    } else {
        render_nav(frame, app, nav_area, theme);
        render_body(frame, app, body_area, theme);
        render_bottom_bar(frame, app, bottom_area, theme);
    }
}

#[cfg(feature = "catalog")]
fn render_launcher_nav(frame: &mut Frame, area: Rect, app: &App, theme: ThemeMode) {
    let line = if app.loading_state != LoadingState::Idle {
        let idx = (app.spinner_frame as usize) % SPINNER_FRAMES.chars().count();
        let c = SPINNER_FRAMES.chars().nth(idx).unwrap_or(' ');
        let msg: String = match app.loading_state {
            LoadingState::Idle => String::new(),
            LoadingState::IngestingDirectory => {
                if let Some((cur, tot)) = app.ingest_progress {
                    if tot > 0 {
                        format!(" Ingesting {}/{}...", cur, tot)
                    } else {
                        " Ingesting...".to_string()
                    }
                } else {
                    " Ingesting...".to_string()
                }
            }
            LoadingState::CatalogUpdating => " Updating catalog...".to_string(),
        };
        Line::from(vec![
            Span::styled(c.to_string(), super::theme::spinner_style(theme)),
            Span::raw(msg),
        ])
    } else {
        Line::from(ratatui::text::Span::styled(
            "  Indexed beamtimes (most recent first)",
            super::theme::header_style(theme),
        ))
    };
    frame.render_widget(Paragraph::new(line), area);
}

#[cfg(feature = "catalog")]
fn render_launcher_body(frame: &mut Frame, app: &mut App, area: Rect, theme: ThemeMode) {
    let state = match &mut app.launcher_state {
        Some(s) => s,
        None => return,
    };
    let items: Vec<ListItem> = state
        .beamtimes
        .iter()
        .map(|(path, ts)| {
            let date_str = chrono::DateTime::from_timestamp(*ts, 0)
                .map(|dt| dt.format("%Y-%m-%d").to_string())
                .unwrap_or_else(|| "?".to_string());
            ListItem::new(format!("{}  ({})", path.display(), date_str))
        })
        .collect();
    let list = List::new(items)
        .block(
            Block::bordered()
                .title(" Indexed beamtimes ")
                .border_style(ratatui::style::Style::default()),
        )
        .highlight_style(list_style(true, theme))
        .highlight_symbol("  ");
    frame.render_stateful_widget(list, area, &mut state.list_state);
}

#[cfg(feature = "catalog")]
fn render_launcher_bottom(frame: &mut Frame, area: Rect, theme: ThemeMode) {
    let style = super::theme::keybind_bar_style(theme);
    let line = Line::from(ratatui::text::Span::styled(LAUNCHER_HINTS, style));
    let para = Paragraph::new(line).alignment(Alignment::Center);
    frame.render_widget(para, area);
}

#[cfg(feature = "catalog")]
const OPEN_DIR_TITLE: &str = " j/k move  Enter open  e edit path  i ingest  Esc cancel ";

#[cfg(feature = "catalog")]
const OPEN_DIR_TITLE_EDIT: &str = " [EDIT] type path  Enter apply  Esc back ";

#[cfg(feature = "catalog")]
fn render_open_dir_modal(frame: &mut Frame, app: &mut App, area: Rect, theme: ThemeMode) {
    let w = area.width.min(70).max(40);
    let h = area.height.min(25).max(10);
    let x = area.x + area.width.saturating_sub(w) / 2;
    let y = area.y + area.height.saturating_sub(h) / 2;
    let modal = Rect::new(x, y, w, h);
    let border_style = ratatui::style::Style::default();
    let title = if app.open_dir_focus == OpenDirFocus::PathInput {
        OPEN_DIR_TITLE_EDIT
    } else {
        OPEN_DIR_TITLE
    };
    let block = Block::bordered().title(title).border_style(border_style);
    let inner = block.inner(modal);
    frame.render_widget(block, modal);
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Min(3)])
        .split(inner);
    let path_style = if app.open_dir_focus == OpenDirFocus::PathInput {
        super::theme::focus_border_style(theme)
    } else {
        ratatui::style::Style::default()
    };
    let path_spans: Vec<Span> = if app.open_dir_focus == OpenDirFocus::PathInput {
        vec![
            Span::styled(app.open_dir_path.as_str(), path_style),
            Span::styled("_", super::theme::spinner_style(theme)),
        ]
    } else {
        vec![Span::styled(app.open_dir_path.as_str(), path_style)]
    };
    let path_line = Line::from(path_spans);
    let path_para = Paragraph::new(path_line);
    frame.render_widget(path_para, chunks[0]);
    let mut items: Vec<ListItem> = vec![ListItem::new("..")];
    for p in &app.open_dir_entries {
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("?");
        items.push(ListItem::new(name));
    }
    let list_border = if app.open_dir_focus == OpenDirFocus::List {
        super::theme::focus_border_style(theme)
    } else {
        ratatui::style::Style::default()
    };
    let list = List::new(items)
        .block(Block::bordered().border_style(list_border))
        .highlight_style(list_style(true, theme))
        .highlight_symbol("  ");
    frame.render_stateful_widget(list, chunks[1], &mut app.open_dir_list_state);
}

const SPINNER_FRAMES: &str = "\u{2801}\u{2801}\u{2809}\u{2819}\u{281a}\u{2812}\u{2802}\u{2802}\u{2812}\u{2832}\u{2834}\u{2824}\u{2804}\u{2804}\u{2824}\u{2820}\u{2820}\u{2824}\u{2826}\u{2816}\u{2812}\u{2810}\u{2810}\u{2812}\u{2813}\u{280b}\u{2809}\u{2808}\u{2808}";

fn render_nav(frame: &mut Frame, app: &App, area: Rect, theme: ThemeMode) {
    let path_display = truncate_path(&app.current_root, NAV_PATH_TRUNCATE);
    let filter_hint = if !app.search_query.is_empty() && app.mode != AppMode::Search {
        format!("  filter: {}", app.search_query)
    } else {
        String::new()
    };
    let (loading_spinner, loading_msg) = if app.loading_state != LoadingState::Idle {
        let idx = (app.spinner_frame as usize) % SPINNER_FRAMES.chars().count();
        let c = SPINNER_FRAMES.chars().nth(idx).unwrap_or(' ');
        let msg: String = match app.loading_state {
            LoadingState::Idle => String::new(),
            LoadingState::IngestingDirectory => {
                if let Some((cur, tot)) = app.ingest_progress {
                    if tot > 0 {
                        format!(" Ingesting {}/{}...", cur, tot)
                    } else {
                        " Ingesting...".to_string()
                    }
                } else {
                    " Ingesting...".to_string()
                }
            }
            LoadingState::CatalogUpdating => " Updating catalog...".to_string(),
        };
        (Some(c), msg)
    } else {
        (None, String::new())
    };
    let mut spans: Vec<Span> = vec![
        Span::raw("  "),
        Span::raw(path_display),
        Span::raw(&filter_hint),
    ];
    if let Some(c) = loading_spinner {
        spans.push(Span::styled(
            c.to_string(),
            super::theme::spinner_style(theme),
        ));
        spans.push(Span::raw(loading_msg));
    }
    let (label_style, input_style, hint_style) = (
        super::theme::rename_retag_label_style(theme),
        super::theme::rename_retag_input_style(theme),
        super::theme::rename_retag_hint_style(theme),
    );
    match app.mode {
        AppMode::RenameSample => {
            spans.push(Span::raw("  "));
            spans.push(Span::styled(" Rename sample ", label_style));
            spans.push(Span::raw(" "));
            spans.push(Span::styled(app.rename_retag_buffer.as_str(), input_style));
            if app.rename_retag_buffer.is_empty() {
                spans.push(Span::styled("_", input_style));
            }
            spans.push(Span::styled("  Enter apply  Esc cancel", hint_style));
        }
        AppMode::EditTag => {
            spans.push(Span::raw("  "));
            spans.push(Span::styled(" Retag ", label_style));
            spans.push(Span::raw(" "));
            spans.push(Span::styled(app.rename_retag_buffer.as_str(), input_style));
            if app.rename_retag_buffer.is_empty() {
                spans.push(Span::styled("_", input_style));
            }
            spans.push(Span::styled("  Enter apply  Esc cancel", hint_style));
        }
        _ => {}
    }
    let line = Line::from(spans);
    let para = Paragraph::new(line);
    frame.render_widget(para, area);
}

fn render_bottom_bar(frame: &mut Frame, app: &App, area: Rect, theme: ThemeMode) {
    let err_style = super::theme::status_error_style(theme);
    let warn_style = super::theme::status_warning_style(theme);
    let keybind_style = super::theme::keybind_bar_style(theme);

    if area.height >= 2 {
        if let Some(ref e) = app.app_error {
            let msg_para = Paragraph::new(Line::from(ratatui::text::Span::styled(
                e.message.as_str(),
                err_style,
            )))
            .alignment(Alignment::Center);
            let hint = e
                .suggestion
                .as_deref()
                .unwrap_or("Escape to dismiss.")
                .trim();
            let hint_para =
                Paragraph::new(Line::from(ratatui::text::Span::styled(hint, err_style)))
                    .alignment(Alignment::Center);
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Length(1), Constraint::Length(1)])
                .split(area);
            frame.render_widget(msg_para, chunks[0]);
            frame.render_widget(hint_para, chunks[1]);
            return;
        }
        if let Some(ref e) = app.app_warning {
            let msg_para = Paragraph::new(Line::from(ratatui::text::Span::styled(
                e.message.as_str(),
                warn_style,
            )))
            .alignment(Alignment::Center);
            let hint = e
                .suggestion
                .as_deref()
                .unwrap_or("Escape to dismiss.")
                .trim();
            let hint_para =
                Paragraph::new(Line::from(ratatui::text::Span::styled(hint, warn_style)))
                    .alignment(Alignment::Center);
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Length(1), Constraint::Length(1)])
                .split(area);
            frame.render_widget(msg_para, chunks[0]);
            frame.render_widget(hint_para, chunks[1]);
            return;
        }
    }

    let (content, style) = if let Some(ref e) = app.app_error {
        let hint = e.suggestion.as_deref().unwrap_or("").trim();
        let content = if hint.is_empty() {
            e.message.clone()
        } else {
            format!("{}  {}", e.message, hint)
        };
        (content, err_style)
    } else if let Some(ref e) = app.app_warning {
        let hint = e.suggestion.as_deref().unwrap_or("").trim();
        let content = if hint.is_empty() {
            e.message.clone()
        } else {
            format!("{}  {}", e.message, hint)
        };
        (content, warn_style)
    } else if let Some((ref msg, is_error)) = app.status_message {
        let style = if is_error {
            err_style
        } else {
            super::theme::status_ok_style(theme)
        };
        (msg.clone(), style)
    } else {
        (bottom_bar_line(), keybind_style)
    };
    let line = Line::from(ratatui::text::Span::styled(content, style));
    let para = Paragraph::new(line).alignment(Alignment::Center);
    frame.render_widget(para, area);
}

fn render_browse_shortcuts(frame: &mut Frame, area: Rect, theme: ThemeMode) {
    let style = super::theme::empty_message_style(theme);
    let line = Line::from(ratatui::text::Span::styled(BROWSE_SHORTCUTS, style));
    let para = Paragraph::new(line).alignment(Alignment::Right);
    frame.render_widget(para, area);
}

fn render_search_box(frame: &mut Frame, app: &App, area: Rect, theme: ThemeMode) {
    let border_style = if app.focus == Focus::SearchBar {
        super::theme::focus_border_style(theme)
    } else {
        ratatui::style::Style::default()
    };
    let block = Block::bordered().border_style(border_style);
    let inner = block.inner(area);

    let prompt_style = super::theme::search_prompt_style(theme);
    let search_line = if app.mode == AppMode::Search {
        Line::from(vec![
            ratatui::text::Span::styled("/ ", prompt_style),
            ratatui::text::Span::raw(app.search_query.as_str()),
        ])
    } else {
        Line::from(ratatui::text::Span::styled(
            search_prompt_display(),
            prompt_style,
        ))
    };
    let para = Paragraph::new(search_line).alignment(Alignment::Center);
    frame.render_widget(para, inner);
    frame.render_widget(block, area);
}

fn render_body(frame: &mut Frame, app: &mut App, area: Rect, theme: ThemeMode) {
    if !app.has_catalog {
        app.last_body_rects = None;
        let empty_style = super::theme::empty_message_style(theme);
        let line1 = ratatui::text::Line::from(ratatui::text::Span::styled(
            "No catalog in this directory.",
            empty_style,
        ));
        let line2 = ratatui::text::Line::from(ratatui::text::Span::styled(
            "Press [i] to ingest directory, or from Python: pyref.io.ingest_beamtime(path)",
            empty_style,
        ));
        let path_line = ratatui::text::Line::from(ratatui::text::Span::styled(
            format!("Path: {}", app.current_root),
            empty_style,
        ));
        let para = Paragraph::new(vec![line1, line2, path_line]).alignment(Alignment::Center);
        frame.render_widget(para, area);
        return;
    }
    let [left_pct, _right_pct] = layout_constraints(app);
    let body_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([left_pct, Constraint::Length(1), Constraint::Min(0)])
        .split(area);
    let left_area = body_chunks[0];
    let right_area = body_chunks[2];

    let (s_h, t_h, sc_h) = left_sidebar_list_heights(
        left_area.height,
        app.samples.len(),
        app.tags.len(),
        app.scans.len(),
    );
    let left_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(s_h),
            Constraint::Length(1),
            Constraint::Length(t_h),
            Constraint::Length(1),
            Constraint::Length(sc_h),
        ])
        .split(left_area);
    render_sample_list(frame, app, left_chunks[0], theme);
    render_tag_list(frame, app, left_chunks[2], theme);
    render_scan_list(frame, app, left_chunks[4], theme);

    let right_border_style = if app.focus == Focus::Table || app.focus == Focus::SearchBar {
        super::theme::focus_border_style(theme)
    } else {
        ratatui::style::Style::default()
    };
    let right_block = Block::bordered()
        .title(BROWSE_TITLE)
        .border_style(right_border_style);
    let right_inner = right_block.inner(right_area);
    let right_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Fill(1),
            Constraint::Length(3),
        ])
        .split(right_inner);
    let shortcuts_area = right_chunks[0];
    let table_and_files_area = right_chunks[1];
    let search_area = right_chunks[2];

    let (table_area, files_area) = if let Some(i) = app.expanded_table_row {
        if let Some(g) = app.group_at_display_index(i) {
            let row_count = g.file_rows.len();
            let list_h = (row_count as u16).min(10).saturating_add(3);
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(3), Constraint::Length(list_h)])
                .split(table_and_files_area);
            (chunks[0], Some(chunks[1]))
        } else {
            (table_and_files_area, None)
        }
    } else {
        (table_and_files_area, None)
    };

    frame.render_widget(right_block, right_area);
    render_browse_shortcuts(frame, shortcuts_area, theme);
    render_table(frame, app, table_area, theme);
    if let Some(area) = files_area {
        if app.expanded_table_row.is_some() {
            render_expanded_file_list(frame, area, theme, app);
        }
    }
    render_search_box(frame, app, search_area, theme);
    let rects = super::app::BeamtimeBodyRects {
        sample_list: left_chunks[0],
        tag_list: left_chunks[2],
        scan_list: left_chunks[4],
        table: table_area,
    };
    app.last_body_rects = Some((area, rects));
}

fn list_style(active: bool, theme: ThemeMode) -> ratatui::style::Style {
    if active {
        super::theme::highlight_style(theme)
    } else {
        ratatui::style::Style::default()
    }
}

fn render_sample_list(frame: &mut Frame, app: &mut App, area: Rect, theme: ThemeMode) {
    let items: Vec<ListItem> = app
        .samples
        .iter()
        .map(|s| {
            let circle = if app.selected_samples.contains(s) {
                CIRCLE_FILLED
            } else {
                CIRCLE_EMPTY
            };
            ListItem::new(format!("{} {}", circle, s))
        })
        .collect();
    let border_style = if app.focus == Focus::SampleList {
        super::theme::focus_border_style(theme)
    } else {
        ratatui::style::Style::default()
    };
    let list = List::new(items)
        .block(
            Block::bordered()
                .title(" Sample [s] ")
                .border_style(border_style),
        )
        .highlight_style(list_style(true, theme))
        .highlight_symbol("  ");
    frame.render_stateful_widget(list, area, &mut app.sample_state);
}

fn render_tag_list(frame: &mut Frame, app: &mut App, area: Rect, theme: ThemeMode) {
    let items: Vec<ListItem> = app
        .tags
        .iter()
        .map(|s| {
            let circle = if app.selected_tags.contains(s) {
                CIRCLE_FILLED
            } else {
                CIRCLE_EMPTY
            };
            ListItem::new(format!("{} {}", circle, s))
        })
        .collect();
    let border_style = if app.focus == Focus::TagList {
        super::theme::focus_border_style(theme)
    } else {
        ratatui::style::Style::default()
    };
    let list = List::new(items)
        .block(
            Block::bordered()
                .title(" Tag [t] ")
                .border_style(border_style),
        )
        .highlight_style(list_style(true, theme))
        .highlight_symbol("  ");
    frame.render_stateful_widget(list, area, &mut app.tag_state);
}

fn render_scan_list(frame: &mut Frame, app: &mut App, area: Rect, theme: ThemeMode) {
    let items: Vec<ListItem> = app
        .scans
        .iter()
        .map(|(n, label)| {
            let circle = if app.selected_scans.contains(n) {
                CIRCLE_FILLED
            } else {
                CIRCLE_EMPTY
            };
            ListItem::new(format!("{} {}", circle, label))
        })
        .collect();
    let border_style = if app.focus == Focus::ScanList {
        super::theme::focus_border_style(theme)
    } else {
        ratatui::style::Style::default()
    };
    let list = List::new(items)
        .block(
            Block::bordered()
                .title(" Experiment [e] ")
                .border_style(border_style),
        )
        .highlight_style(list_style(true, theme))
        .highlight_symbol("  ");
    frame.render_stateful_widget(list, area, &mut app.scan_list_state);
}

#[cfg(feature = "catalog")]
fn pol_str_from_epu(epu: Option<f64>) -> String {
    epu.map(|p| {
        let rounded = p.round() as i32;
        if rounded == 100 {
            "S".to_string()
        } else if rounded == 190 {
            "P".to_string()
        } else {
            format!("{:.2}", p)
        }
    })
    .unwrap_or_else(|| "-".to_string())
}

#[cfg(feature = "catalog")]
fn render_expanded_file_list(
    frame: &mut Frame,
    area: Rect,
    theme: ThemeMode,
    app: &mut super::app::App,
) {
    let Some(i) = app.expanded_table_row else {
        return;
    };
    let file_count = app
        .group_at_display_index(i)
        .map(|g| g.file_rows.len())
        .unwrap_or(0);
    let border_style = super::theme::header_style(theme);
    let block = Block::bordered()
        .title(" FITS files for reflectivity profile (j/k scroll, Enter collapse) ")
        .border_style(border_style);
    let inner = block.inner(area);
    let visible_rows = inner.height.saturating_sub(2) as usize;
    app.last_expanded_files_visible = visible_rows.max(1);
    let max_offset = file_count.saturating_sub(app.last_expanded_files_visible);
    if app.expanded_files_scroll_offset > max_offset {
        app.expanded_files_scroll_offset = max_offset;
    }
    let file_rows = match app.group_at_display_index(i) {
        Some(g) => &g.file_rows,
        None => return,
    };
    let offset = app.expanded_files_scroll_offset;
    let start = offset.min(file_rows.len());
    let slice = file_rows.get(start..).unwrap_or(&[]);
    let take = slice.len().min(visible_rows);
    let visible_slice = slice.get(..take).unwrap_or(&[]);
    frame.render_widget(block, area);
    let header_style = super::theme::header_style(theme);
    let header = Row::new(vec![
        Cell::from("Scan"),
        Cell::from("Frame"),
        Cell::from("pol"),
        Cell::from("E (eV)"),
        Cell::from("\u{03B8} (\u{00B0})"),
    ])
    .style(header_style)
    .bottom_margin(1);
    let rows: Vec<Row> = visible_slice
        .iter()
        .map(|r| {
            let scan = r.scan_number.to_string();
            let frame = r.frame_number.to_string();
            let pol = pol_str_from_epu(r.epu_polarization);
            let energy = r
                .beamline_energy
                .map(|e| format!("{:.1}", e))
                .unwrap_or_else(|| "-".to_string());
            let theta = r
                .sample_theta
                .map(|t| format!("{:.2}", t))
                .unwrap_or_else(|| "-".to_string());
            Row::new(vec![
                Cell::from(scan),
                Cell::from(frame),
                Cell::from(pol),
                Cell::from(energy),
                Cell::from(theta),
            ])
        })
        .collect();
    let widths = [
        Constraint::Length(6),
        Constraint::Length(6),
        Constraint::Length(4),
        Constraint::Length(8),
        Constraint::Length(8),
    ];
    let table = Table::new(rows, widths).header(header).column_spacing(1);
    frame.render_widget(table, inner);
}

fn scan_type_label(st: ReflectivityScanType) -> &'static str {
    match st {
        ReflectivityScanType::FixedEnergy => "\u{03B8}",
        ReflectivityScanType::FixedAngle => "E",
        ReflectivityScanType::SinglePoint => "-",
    }
}

fn grouped_row_to_cells(r: &GroupedProfileRow) -> Vec<Cell<'_>> {
    let e_min_str = r
        .energy_min
        .map(|v| format!("{:.1}", v))
        .unwrap_or_else(|| "-".to_string());
    let e_max_str = r
        .energy_max
        .map(|v| format!("{:.1}", v))
        .unwrap_or_else(|| "-".to_string());
    let theta_min_str = r
        .theta_min
        .map(|v| format!("{:.2}", v))
        .unwrap_or_else(|| "-".to_string());
    let theta_max_str = r
        .theta_max
        .map(|v| format!("{:.2}", v))
        .unwrap_or_else(|| "-".to_string());
    let frames_str = r.file_rows.len().to_string();
    vec![
        Cell::from(r.sample.as_str()),
        Cell::from(r.tag.as_str()),
        Cell::from(r.pol_str.as_str()),
        Cell::from(scan_type_label(r.scan_type)),
        Cell::from(e_min_str),
        Cell::from(e_max_str),
        Cell::from(theta_min_str),
        Cell::from(theta_max_str),
        Cell::from(frames_str),
        Cell::from(r.scan_duration_hm.as_str()),
    ]
}

fn table_header_cell(
    label: &str,
    sort_col: Option<usize>,
    sort_ord: Option<std::cmp::Ordering>,
    col_index: usize,
) -> Cell<'static> {
    use std::cmp::Ordering;
    let suffix = if sort_col == Some(col_index) {
        match sort_ord {
            Some(Ordering::Less) => " \u{25B2}",
            Some(Ordering::Greater) => " \u{25BC}",
            Some(Ordering::Equal) | None => " \u{25A0}",
        }
    } else {
        " \u{25A0}"
    };
    Cell::from(format!("{}{}", label, suffix))
}

fn render_table(frame: &mut Frame, app: &mut App, area: Rect, theme: ThemeMode) {
    let header_style = super::theme::header_style(theme);
    let (sort_col, sort_ord) = (app.table_sort_column, app.table_sort_ordering);
    let header = Row::new(vec![
        Cell::from(" "),
        table_header_cell("Sample", sort_col, sort_ord, 0),
        table_header_cell("Tag", sort_col, sort_ord, 1),
        table_header_cell("Pol", sort_col, sort_ord, 2),
        table_header_cell("Type", sort_col, sort_ord, 3),
        table_header_cell("Emin", sort_col, sort_ord, 4),
        table_header_cell("Emax", sort_col, sort_ord, 5),
        table_header_cell("\u{03B8}min", sort_col, sort_ord, 6),
        table_header_cell("\u{03B8}max", sort_col, sort_ord, 7),
        table_header_cell("len", sort_col, sort_ord, 8),
        table_header_cell("\u{0394}T", sort_col, sort_ord, 9),
    ])
    .style(header_style)
    .bottom_margin(1);

    let order = app.compute_display_order();
    let expanded = app.expanded_table_row;
    let rows: Vec<Row> = order
        .iter()
        .enumerate()
        .map(|(disp_idx, &orig_idx)| {
            let r = &app.filtered_groups[orig_idx];
            let caret = if expanded == Some(disp_idx) { "v" } else { ">" };
            let mut cells = vec![Cell::from(caret)];
            cells.extend(grouped_row_to_cells(r));
            Row::new(cells)
        })
        .collect();

    let widths = [
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

    if rows.is_empty() {
        let empty_style = super::theme::empty_message_style(theme);
        let msg = Paragraph::new(Line::from(ratatui::text::Span::styled(
            "No reflectivity profiles match (sample/tag/scan/search)",
            empty_style,
        )));
        frame.render_widget(msg, area);
        return;
    }

    let table = Table::new(rows, widths)
        .header(header)
        .column_spacing(1)
        .row_highlight_style(super::theme::row_highlight_style(theme))
        .highlight_symbol("  ");

    frame.render_stateful_widget(table, area, &mut app.table_state);
}

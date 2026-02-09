use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::text::Line;
use ratatui::widgets::{Block, Cell, List, ListItem, Paragraph, Row, Table};
use ratatui::Frame;
use ratatui::layout::Alignment;

use super::app::{App, AppMode, Focus, ProfileRow};
use super::keymap::{keybind_bar_lines_emacs, keybind_bar_lines_vi, search_bar_hint, search_line_hotkeys};
use super::theme::ThemeMode;

const NAV_PATH_TRUNCATE: usize = 60;

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

pub fn render(frame: &mut Frame, app: &mut App) {
    let area = frame.area();
    let theme = theme_mode(app);

    let mut vertical = vec![Constraint::Length(1), Constraint::Length(1)];
    vertical.push(Constraint::Fill(1));
    for _ in 0..app.keybind_bar_lines {
        vertical.push(Constraint::Length(1));
    }

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(vertical)
        .split(area);

    let nav_area = chunks[0];
    let search_area = chunks[1];
    let body_area = chunks[2];
    let keybind_start = chunks.len().saturating_sub(app.keybind_bar_lines as usize);
    let keybind_areas: Vec<Rect> = (keybind_start..chunks.len()).map(|j| chunks[j]).collect();

    render_nav(frame, app, nav_area, theme);
    render_search_bar(frame, app, search_area, theme);
    render_body(frame, app, body_area, theme);
    for (idx, rect) in keybind_areas.into_iter().enumerate() {
        render_keybind_bar(frame, app, rect, theme, idx);
    }
}

fn render_nav(frame: &mut Frame, app: &App, area: Rect, theme: ThemeMode) {
    let path_display = truncate_path(&app.current_root, NAV_PATH_TRUNCATE);
    let filter_hint = if !app.search_query.is_empty() && app.mode != AppMode::Search {
        format!("  filter: {}", app.search_query)
    } else {
        String::new()
    };
    let line = Line::from(format!("  {}  [up] [back] [fwd]{}", path_display, filter_hint));
    let border_style = if app.focus == Focus::Nav {
        super::theme::focus_border_style(theme)
    } else {
        ratatui::style::Style::default()
    };
    let para = Paragraph::new(line).block(Block::bordered().border_style(border_style));
    frame.render_widget(para, area);
}

fn render_search_bar(frame: &mut Frame, app: &App, area: Rect, theme: ThemeMode) {
    let prompt_style = super::theme::search_prompt_style(theme);
    let style = super::theme::keybind_bar_style(theme);
    let search_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Min(12), Constraint::Fill(1)])
        .split(area);
    let search_slot = search_chunks[0];
    let hotkeys_slot = search_chunks[1];

    let search_line = if app.mode == AppMode::Search {
        Line::from(vec![
            ratatui::text::Span::styled("/ ", prompt_style),
            ratatui::text::Span::raw(app.search_query.as_str()),
        ])
    } else {
        Line::from(ratatui::text::Span::styled(
            search_bar_hint(&app.keymap),
            prompt_style,
        ))
    };
    frame.render_widget(Paragraph::new(search_line), search_slot);

    let hotkeys = search_line_hotkeys(&app.keymap);
    let hotkeys_para = Paragraph::new(Line::from(ratatui::text::Span::styled(hotkeys, style)))
        .alignment(Alignment::Center);
    frame.render_widget(hotkeys_para, hotkeys_slot);
}

fn render_keybind_bar(
    frame: &mut Frame,
    app: &App,
    area: Rect,
    theme: ThemeMode,
    line_index: usize,
) {
    let style = super::theme::keybind_bar_style(theme);
    let pairs = if app.keymap == "emacs" {
        keybind_bar_lines_emacs()
    } else {
        keybind_bar_lines_vi()
    };
    let (keys, descs): (Vec<_>, Vec<_>) = pairs
        .iter()
        .filter(|(k, _)| !k.is_empty())
        .map(|(k, d)| (k.as_str(), d.as_str()))
        .unzip();
    let line_len = keys.len();
    let half = (line_len + 1) / 2;
    let (first_half_k, second_half_k) = keys.split_at(half);
    let (first_half_d, second_half_d) = descs.split_at(half);
    let parts = if line_index == 0 {
        first_half_k
            .iter()
            .zip(first_half_d.iter())
            .map(|(k, d)| format!("{} {}", k, d))
            .collect::<Vec<_>>()
            .join("  ")
    } else {
        second_half_k
            .iter()
            .zip(second_half_d.iter())
            .map(|(k, d)| format!("{} {}", k, d))
            .collect::<Vec<_>>()
            .join("  ")
    };
    let line = Line::from(ratatui::text::Span::styled(parts, style));
    let para = Paragraph::new(line).alignment(Alignment::Center);
    frame.render_widget(para, area);
}

fn render_body(frame: &mut Frame, app: &mut App, area: Rect, theme: ThemeMode) {
    let constraints = layout_constraints(app);
    let body_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints(constraints)
        .split(area);
    let left_final = body_chunks[0];
    let table_final = body_chunks[1];

    let left_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(4),
            Constraint::Length(4),
            Constraint::Min(3),
        ])
        .split(left_final);
    let sample_area = left_chunks[0];
    let tag_area = left_chunks[1];
    let experiment_area = left_chunks[2];

    render_sample_list(frame, app, sample_area, theme);
    render_tag_list(frame, app, tag_area, theme);
    render_experiment_list(frame, app, experiment_area, theme);
    render_table(frame, app, table_final, theme);
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
        .map(|s| ListItem::new(s.as_str()))
        .collect();
    let border_style = if app.focus == Focus::SampleList {
        super::theme::focus_border_style(theme)
    } else {
        ratatui::style::Style::default()
    };
    let list = List::new(items)
        .block(Block::bordered().title(" Sample [s] ").border_style(border_style))
        .highlight_style(list_style(true, theme))
        .highlight_symbol("> ");
    frame.render_stateful_widget(list, area, &mut app.sample_state);
}

fn render_tag_list(frame: &mut Frame, app: &mut App, area: Rect, theme: ThemeMode) {
    let items: Vec<ListItem> = app.tags.iter().map(|s| ListItem::new(s.as_str())).collect();
    let border_style = if app.focus == Focus::TagList {
        super::theme::focus_border_style(theme)
    } else {
        ratatui::style::Style::default()
    };
    let list = List::new(items)
        .block(Block::bordered().title(" Tag [t] ").border_style(border_style))
        .highlight_style(list_style(true, theme))
        .highlight_symbol("> ");
    frame.render_stateful_widget(list, area, &mut app.tag_state);
}

fn render_experiment_list(frame: &mut Frame, app: &mut App, area: Rect, theme: ThemeMode) {
    let items: Vec<ListItem> = app
        .experiments
        .iter()
        .map(|(_, label)| ListItem::new(label.as_str()))
        .collect();
    let border_style = if app.focus == Focus::ExperimentList {
        super::theme::focus_border_style(theme)
    } else {
        ratatui::style::Style::default()
    };
    let list = List::new(items)
        .block(Block::bordered().title(" Experiment [e] ").border_style(border_style))
        .highlight_style(list_style(true, theme))
        .highlight_symbol("> ");
    frame.render_stateful_widget(list, area, &mut app.experiment_state);
}

fn profile_row_to_cells(r: &ProfileRow) -> Vec<Cell<'_>> {
    vec![
        Cell::from(r.sample.as_str()),
        Cell::from(r.tag.as_str()),
        Cell::from(r.energy_str.as_str()),
        Cell::from(r.pol.as_str()),
        Cell::from(r.q_range_str.as_str()),
        Cell::from(r.data_points.to_string()),
        Cell::from(r.quality_placeholder.as_str()),
    ]
}

fn render_table(frame: &mut Frame, app: &mut App, area: Rect, theme: ThemeMode) {
    let header_style = super::theme::header_style(theme);
    let header = Row::new(vec![
        Cell::from("Sample"),
        Cell::from("tag"),
        Cell::from("energy"),
        Cell::from("pol"),
        Cell::from("q-range"),
        Cell::from("Data points"),
        Cell::from("Quality"),
    ])
    .style(header_style)
    .bottom_margin(1);

    let rows: Vec<Row> = app
        .filtered_profiles
        .iter()
        .map(|r| Row::new(profile_row_to_cells(r)))
        .collect();

    let widths = [
        Constraint::Percentage(12),
        Constraint::Percentage(10),
        Constraint::Percentage(18),
        Constraint::Percentage(6),
        Constraint::Percentage(14),
        Constraint::Percentage(12),
        Constraint::Percentage(10),
    ];

    let border_style = if app.focus == Focus::Table {
        super::theme::focus_border_style(theme)
    } else {
        ratatui::style::Style::default()
    };

    if rows.is_empty() {
        let empty_style = super::theme::empty_message_style(theme);
        let msg = Paragraph::new(Line::from(ratatui::text::Span::styled(
            "No profiles match (sample/tag/experiment/search)",
            empty_style,
        )))
        .block(
            Block::bordered()
                .title(" Reflectivity profiles [b] ")
                .border_style(border_style),
        );
        frame.render_widget(msg, area);
        return;
    }

    let table = Table::new(rows, widths)
        .header(header)
        .block(
            Block::bordered()
                .title(" Reflectivity profiles [b] ")
                .border_style(border_style),
        )
        .column_spacing(1)
        .row_highlight_style(super::theme::row_highlight_style(theme))
        .highlight_symbol(">> ");

    frame.render_stateful_widget(table, area, &mut app.table_state);
}

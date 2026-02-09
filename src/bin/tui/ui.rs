use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::text::Line;
use ratatui::widgets::{Block, Cell, List, ListItem, Padding, Paragraph, Row, Table};
use ratatui::Frame;
use ratatui::layout::Alignment;

use super::app::{App, AppMode, Focus, ProfileRow};
use super::keymap::{bottom_bar_line, search_prompt_display, BROWSE_SHORTCUTS, BROWSE_TITLE};
use super::theme::ThemeMode;

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

pub fn render(frame: &mut Frame, app: &mut App) {
    let area = frame.area();
    let theme = theme_mode(app);

    let outer_block = Block::bordered().padding(Padding::new(1, 1, 1, 1));
    let inner = outer_block.inner(area);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Fill(1),
            Constraint::Length(1),
        ])
        .split(inner);

    let nav_area = chunks[0];
    let body_area = chunks[1];
    let bottom_area = chunks[2];

    frame.render_widget(outer_block, area);
    render_nav(frame, app, nav_area, theme);
    render_body(frame, app, body_area, theme);
    render_bottom_bar(frame, app, bottom_area, theme);
}

fn render_nav(frame: &mut Frame, app: &App, area: Rect, _theme: ThemeMode) {
    let path_display = truncate_path(&app.current_root, NAV_PATH_TRUNCATE);
    let filter_hint = if !app.search_query.is_empty() && app.mode != AppMode::Search {
        format!("  filter: {}", app.search_query)
    } else {
        String::new()
    };
    let line = Line::from(format!("  {}  [up] [back] [fwd]{}", path_display, filter_hint));
    let para = Paragraph::new(line);
    frame.render_widget(para, area);
}

fn render_bottom_bar(frame: &mut Frame, _app: &App, area: Rect, theme: ThemeMode) {
    let style = super::theme::keybind_bar_style(theme);
    let content = bottom_bar_line();
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
    let [left_pct, _right_pct] = layout_constraints(app);
    let body_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([left_pct, Constraint::Length(1), Constraint::Min(0)])
        .split(area);
    let left_area = body_chunks[0];
    let right_area = body_chunks[2];

    let left_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(4),
            Constraint::Length(1),
            Constraint::Length(4),
            Constraint::Length(1),
            Constraint::Min(3),
        ])
        .split(left_area);
    render_sample_list(frame, app, left_chunks[0], theme);
    render_tag_list(frame, app, left_chunks[2], theme);
    render_experiment_list(frame, app, left_chunks[4], theme);

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
    let table_area = right_chunks[1];
    let search_area = right_chunks[2];

    frame.render_widget(right_block, right_area);
    render_browse_shortcuts(frame, shortcuts_area, theme);
    render_table(frame, app, table_area, theme);
    render_search_box(frame, app, search_area, theme);
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
        .block(Block::bordered().title(" Sample [s] ").border_style(border_style))
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
        .block(Block::bordered().title(" Tag [t] ").border_style(border_style))
        .highlight_style(list_style(true, theme))
        .highlight_symbol("  ");
    frame.render_stateful_widget(list, area, &mut app.tag_state);
}

fn render_experiment_list(frame: &mut Frame, app: &mut App, area: Rect, theme: ThemeMode) {
    let items: Vec<ListItem> = app
        .experiments
        .iter()
        .map(|(n, label)| {
            let circle = if app.selected_experiments.contains(n) {
                CIRCLE_FILLED
            } else {
                CIRCLE_EMPTY
            };
            ListItem::new(format!("{} {}", circle, label))
        })
        .collect();
    let border_style = if app.focus == Focus::ExperimentList {
        super::theme::focus_border_style(theme)
    } else {
        ratatui::style::Style::default()
    };
    let list = List::new(items)
        .block(Block::bordered().title(" Experiment [e] ").border_style(border_style))
        .highlight_style(list_style(true, theme))
        .highlight_symbol("  ");
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

    if rows.is_empty() {
        let empty_style = super::theme::empty_message_style(theme);
        let msg = Paragraph::new(Line::from(ratatui::text::Span::styled(
            "No profiles match (sample/tag/experiment/search)",
            empty_style,
        )));
        frame.render_widget(msg, area);
        return;
    }

    let table = Table::new(rows, widths)
        .header(header)
        .column_spacing(1)
        .row_highlight_style(super::theme::row_highlight_style(theme))
        .highlight_symbol(">> ");

    frame.render_stateful_widget(table, area, &mut app.table_state);
}

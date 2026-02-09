use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::Line;
use ratatui::widgets::{Block, Cell, List, ListItem, Paragraph, Row, Table};
use ratatui::Frame;

use super::app::{App, Focus, ProfileRow};

const NAV_PATH_TRUNCATE: usize = 60;

fn truncate_path(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    format!("...{}", s.chars().rev().take(max.saturating_sub(3)).collect::<String>().chars().rev().collect::<String>())
}

pub fn render(frame: &mut Frame, app: &mut App) {
    let area = frame.area();
    let [nav_area, body_area] = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Fill(1)])
        .areas(area);

    render_nav(frame, app, nav_area);
    let [left_area, table_area] = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(22), Constraint::Percentage(78)])
        .areas(body_area);

    let [sample_area, tag_area, experiment_area] = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(8),
            Constraint::Length(8),
            Constraint::Min(5),
        ])
        .areas(left_area);

    render_sample_list(frame, app, sample_area);
    render_tag_list(frame, app, tag_area);
    render_experiment_list(frame, app, experiment_area);
    render_table(frame, app, table_area);
}

fn render_nav(frame: &mut Frame, app: &App, area: Rect) {
    let path_display = truncate_path(&app.current_root, NAV_PATH_TRUNCATE);
    let line = Line::from(format!("  {}  [up] [back] [fwd]", path_display));
    let para = Paragraph::new(line).block(
        Block::bordered()
            .border_style(if app.focus == Focus::Nav {
                Style::new().fg(Color::Cyan)
            } else {
                Style::default()
            }),
    );
    frame.render_widget(para, area);
}

fn list_style(active: bool) -> Style {
    if active {
        Style::new().fg(Color::Black).bg(Color::Cyan)
    } else {
        Style::default()
    }
}

fn render_sample_list(frame: &mut Frame, app: &mut App, area: Rect) {
    let items: Vec<ListItem> = app
        .samples
        .iter()
        .map(|s| ListItem::new(s.as_str()))
        .collect();
    let list = List::new(items)
        .block(
            Block::bordered()
                .title(" Sample ")
                .border_style(if app.focus == Focus::SampleList {
                    Style::new().fg(Color::Cyan)
                } else {
                    Style::default()
                }),
        )
        .highlight_style(list_style(true))
        .highlight_symbol("> ");
    frame.render_stateful_widget(list, area, &mut app.sample_state);
}

fn render_tag_list(frame: &mut Frame, app: &mut App, area: Rect) {
    let items: Vec<ListItem> = app.tags.iter().map(|s| ListItem::new(s.as_str())).collect();
    let list = List::new(items)
        .block(
            Block::bordered()
                .title(" Tag ")
                .border_style(if app.focus == Focus::TagList {
                    Style::new().fg(Color::Cyan)
                } else {
                    Style::default()
                }),
        )
        .highlight_style(list_style(true))
        .highlight_symbol("> ");
    frame.render_stateful_widget(list, area, &mut app.tag_state);
}

fn render_experiment_list(frame: &mut Frame, app: &mut App, area: Rect) {
    let items: Vec<ListItem> = app
        .experiments
        .iter()
        .map(|(_, label)| ListItem::new(label.as_str()))
        .collect();
    let list = List::new(items)
        .block(
            Block::bordered()
                .title(" Experiment ")
                .border_style(if app.focus == Focus::ExperimentList {
                    Style::new().fg(Color::Cyan)
                } else {
                    Style::default()
                }),
        )
        .highlight_style(list_style(true))
        .highlight_symbol("> ");
    frame.render_stateful_widget(list, area, &mut app.experiment_state);
}

fn profile_row_to_cells(r: &ProfileRow) -> Vec<Cell> {
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

fn render_table(frame: &mut Frame, app: &mut App, area: Rect) {
    let header = Row::new(vec![
        Cell::from("Sample"),
        Cell::from("tag"),
        Cell::from("energy"),
        Cell::from("pol"),
        Cell::from("q-range"),
        Cell::from("Data points"),
        Cell::from("Quality"),
    ])
    .style(Style::new().fg(Color::Yellow).add_modifier(Modifier::BOLD))
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

    let table = Table::new(rows, widths)
        .header(header)
        .block(
            Block::bordered()
                .title(" Reflectivity profiles ")
                .border_style(if app.focus == Focus::Table {
                    Style::new().fg(Color::Cyan)
                } else {
                    Style::default()
                }),
        )
        .column_spacing(1)
        .row_highlight_style(Style::new().bg(Color::DarkGray).add_modifier(Modifier::BOLD))
        .highlight_symbol(">> ");

    frame.render_stateful_widget(table, area, &mut app.table_state);
}

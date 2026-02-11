use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::text::Line;
use ratatui::widgets::{Block, Cell, List, ListItem, Padding, Paragraph, Row, Table};
use ratatui::Frame;
use ratatui::layout::Alignment;

use super::app::{App, AppMode, DirEntry, Focus, ProfileRow};
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
    let line = if app.mode == AppMode::ChangeDir {
        let path_display = truncate_path(&app.path_input, NAV_PATH_TRUNCATE);
        let mode_label = if app.path_input_active {
            "[TYPE PATH]"
        } else {
            "[BROWSE]"
        };
        Line::from(format!("  {} Path: {}  [d] close browser", mode_label, path_display))
    } else if app.current_root.is_empty() {
        Line::from("  No directory selected  [d] open file browser")
    } else {
        let path_display = truncate_path(&app.current_root, NAV_PATH_TRUNCATE);
        let filter_hint = if !app.search_query.is_empty() && app.mode != AppMode::Search {
            format!("  filter: {}", app.search_query)
        } else {
            String::new()
        };
        Line::from(format!("  {}  [d] file browser{}", path_display, filter_hint))
    };
    let para = Paragraph::new(line);
    frame.render_widget(para, area);
}

fn render_bottom_bar(frame: &mut Frame, app: &App, area: Rect, theme: ThemeMode) {
    let style = super::theme::keybind_bar_style(theme);
    let content: String = if app.mode == AppMode::ChangeDir {
        if app.path_input_active {
            " Type path  Enter apply  Esc cancel ".to_string()
        } else {
            " j/k move  Tab open  Enter set root  / type path  Esc close ".to_string()
        }
    } else {
        bottom_bar_line()
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

fn render_dir_index_panel(frame: &mut Frame, app: &App, area: Rect, theme: ThemeMode) {
    let index_loading = app.dir_index_loading.is_some();
    let block = Block::bordered().title(" Directory index ");
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let muted = super::theme::empty_message_style(theme);
    if index_loading {
        frame.render_widget(
            Paragraph::new("  Indexing...").style(muted),
            inner,
        );
        return;
    }
    let Some(idx) = &app.current_dir_index else {
        frame.render_widget(Paragraph::new("  No index").style(muted), inner);
        return;
    };
    let samples_str = if idx.samples.is_empty() {
        "-".to_string()
    } else if idx.samples.len() <= 3 {
        idx.samples.join(", ")
    } else {
        format!("{} ({} total)", idx.samples[..3].join(", "), idx.samples.len())
    };
    let energy_str = if idx.energies.is_empty() {
        "-".to_string()
    } else if idx.energies.len() == 1 {
        format!("{:.1} eV", idx.energies[0])
    } else {
        let min = idx.energies.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = idx.energies.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        format!("{:.0}-{:.0} eV", min, max)
    };
    let lines = vec![
        Line::from({
            let t: String = samples_str.chars().take(24).collect();
            let suffix = if samples_str.chars().count() > 24 { ".." } else { "" };
            format!("  Samples: {}{}", t, suffix)
        }),
        Line::from(format!("  Experiments: {}", idx.experiment_count)),
        Line::from(format!("  Energies: {}", energy_str)),
        Line::from(format!("  FITS: {} files", idx.fits_count)),
    ];
    frame.render_widget(Paragraph::new(lines).style(muted), inner);
}

const DIR_COL_NAME: usize = 22;
const DIR_COL_MODIFIED: usize = 12;
const DIR_COL_FITS: usize = 5;
const DIR_COL_ENERGY: usize = 10;
const DIR_COL_POL: usize = 4;
const DIR_COL_SAMPLE: usize = 12;
const DIR_COL_TAG: usize = 8;

fn format_dir_entry(entry: &DirEntry) -> String {
    let kind = if entry.is_dir { "  / " } else { "    " };
    let name_trunc = if entry.name.len() > DIR_COL_NAME {
        format!("{}..", &entry.name[..(DIR_COL_NAME - 2)])
    } else {
        entry.name.clone()
    };
    let modified_str = entry
        .modified
        .as_deref()
        .unwrap_or("-")
        .chars()
        .take(DIR_COL_MODIFIED)
        .collect::<String>();
    let fits_str = entry
        .fits_subdir_count
        .map(|n| n.to_string())
        .unwrap_or_else(|| "-".to_string());
    let energy_str = entry.energy.as_deref().unwrap_or("-").chars().take(DIR_COL_ENERGY).collect::<String>();
    let pol_str = entry.pol.as_deref().unwrap_or("-").chars().take(DIR_COL_POL).collect::<String>();
    let sample_str = entry.sample_name.as_deref().unwrap_or("-").chars().take(DIR_COL_SAMPLE).collect::<String>();
    let tag_str = entry.experiment_tag.as_deref().unwrap_or("-").chars().take(DIR_COL_TAG).collect::<String>();
    format!(
        "{}{:<n$}  {:<m$}  {:>f$}  {:<e$}  {:<p$}  {:<s$}  {:<t$}",
        kind,
        name_trunc,
        modified_str,
        fits_str,
        energy_str,
        pol_str,
        sample_str,
        tag_str,
        n = DIR_COL_NAME,
        m = DIR_COL_MODIFIED,
        f = DIR_COL_FITS,
        e = DIR_COL_ENERGY,
        p = DIR_COL_POL,
        s = DIR_COL_SAMPLE,
        t = DIR_COL_TAG
    )
}

fn render_body(frame: &mut Frame, app: &mut App, area: Rect, theme: ThemeMode) {
    if app.mode == AppMode::ChangeDir {
        let path_display = truncate_path(&app.path_input, NAV_PATH_TRUNCATE);
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(1), Constraint::Min(4)])
            .split(area);
        let path_area = chunks[0];
        let body_row = chunks[1];
        let path_line = Line::from(format!("  Path: {}", path_display));
        frame.render_widget(Paragraph::new(path_line), path_area);
        let (list_area, index_area) = {
            let horz = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Min(40), Constraint::Length(28)])
                .split(body_row);
            (horz[0], horz[1])
        };
        let loading = app.dir_browser_loading.is_some();
        let block = Block::bordered().title(" File browser  j/k move  Tab enter dir  Enter set root  Ctrl+Tab path complete ");
        let inner = block.inner(list_area);
        frame.render_widget(block, list_area);
        let inner_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(1), Constraint::Min(2)])
            .split(inner);
        let header_style = super::theme::header_style(theme);
        let header_line = Line::from(ratatui::text::Span::styled(
            format!(
                "  {:<n$}  {:<m$}  {:>f$}  {:<e$}  {:<p$}  {:<s$}  {:<t$}",
                "Name",
                "Mod",
                "Fits",
                "Energy",
                "Pol",
                "Sample",
                "Tag",
                n = DIR_COL_NAME,
                m = DIR_COL_MODIFIED,
                f = DIR_COL_FITS,
                e = DIR_COL_ENERGY,
                p = DIR_COL_POL,
                s = DIR_COL_SAMPLE,
                t = DIR_COL_TAG
            ),
            header_style,
        ));
        frame.render_widget(Paragraph::new(header_line), inner_chunks[0]);
        if loading {
            frame.render_widget(
                Paragraph::new("  Loading...").style(super::theme::empty_message_style(theme)),
                inner_chunks[1],
            );
        } else {
            let items: Vec<ListItem> = app
                .dir_browser_entries
                .iter()
                .map(|e| ListItem::new(format_dir_entry(e)))
                .collect();
            let list = List::new(items)
                .highlight_style(list_style(true, theme))
                .highlight_symbol(">> ");
            frame.render_stateful_widget(list, inner_chunks[1], &mut app.dir_browser_state);
        }
        render_dir_index_panel(frame, app, index_area, theme);
        return;
    }

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

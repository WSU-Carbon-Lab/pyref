use std::io::{self, Write};

fn restore_terminal_impl() -> io::Result<()> {
    let mut out = io::stdout();
    crossterm::terminal::disable_raw_mode()?;
    crossterm::execute!(out, crossterm::terminal::LeaveAlternateScreen)?;
    crossterm::execute!(out, crossterm::cursor::Show)?;
    if let Ok((_, h)) = crossterm::terminal::size() {
        let row = h.saturating_sub(1);
        crossterm::execute!(out, crossterm::cursor::MoveTo(0, row))?;
    }
    out.flush()?;
    Ok(())
}

pub fn setup_terminal() -> io::Result<()> {
    crossterm::terminal::enable_raw_mode()?;
    crossterm::execute!(io::stdout(), crossterm::terminal::EnterAlternateScreen)?;
    Ok(())
}

pub fn restore_terminal() -> io::Result<()> {
    restore_terminal_impl()
}

pub fn install_panic_hook() {
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = restore_terminal_impl();
        default_hook(info);
    }));
}

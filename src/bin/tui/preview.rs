#![cfg(feature = "tui")]

use std::path::PathBuf;
use std::sync::mpsc;

#[cfg(not(target_os = "macos"))]
use eframe::egui;

#[cfg(target_os = "macos")]
pub fn run_preview_window(rx: mpsc::Receiver<PathBuf>) {
    std::thread::spawn(move || {
        while let Ok(path) = rx.recv() {
            if let Err(e) = stream_fits_to_preview_png(&path) {
                eprintln!("Preview: {}", e);
            }
        }
    });
}

#[cfg(target_os = "macos")]
fn stream_fits_to_preview_png(path: &std::path::Path) -> Result<(), String> {
    use plotters::prelude::*;

    let (_, subtracted) = pyref::io::image_mmap::materialize_image_from_path(path)
        .map_err(|e| e.to_string())?;
    let rgba = pyref::colormap::array2_i64_to_rgba_rainbow(&subtracted, None)
        .ok_or("Colormap failed (non-contiguous)")?;
    let height = subtracted.nrows();
    let width = subtracted.ncols();
    if width == 0 || height == 0 || rgba.len() < width * height * 4 {
        return Err("Invalid image dimensions".to_string());
    }

    const MAX_SIDE: u32 = 512;
    let (out_w, out_h) = {
        let w = width as u32;
        let h = height as u32;
        if w <= MAX_SIDE && h <= MAX_SIDE {
            (w, h)
        } else if w >= h {
            (MAX_SIDE, (h * MAX_SIDE / w).max(1))
        } else {
            ((w * MAX_SIDE / h).max(1), MAX_SIDE)
        }
    };

    let temp = std::env::temp_dir().join("pyref_preview.png");
    let root = BitMapBackend::new(temp.as_path(), (out_w, out_h)).into_drawing_area();
    root.fill(&WHITE).map_err(|e| format!("{:?}", e))?;

    let mut chart = ChartBuilder::on(&root)
        .margin(0)
        .set_all_label_area_size(0)
        .build_cartesian_2d(0i32..(out_w as i32), 0i32..(out_h as i32))
        .map_err(|e| format!("{:?}", e))?;

    let rgba_ref = &rgba;
    chart
        .draw_series(
            (0..(out_h as usize))
                .flat_map(|oy| {
                    (0..(out_w as usize)).map(move |ox| {
                        let sx = (ox * width) / out_w as usize;
                        let sy = (oy * height) / out_h as usize;
                        let i = (sy * width + sx) * 4;
                        let r = rgba_ref.get(i).copied().unwrap_or(0);
                        let g = rgba_ref.get(i + 1).copied().unwrap_or(0);
                        let b = rgba_ref.get(i + 2).copied().unwrap_or(0);
                        (ox as i32, (out_h as i32 - 1 - oy as i32), RGBColor(r, g, b))
                    })
                })
                .map(|(x, y, color)| Rectangle::new([(x, y), (x + 1, y + 1)], color.filled())),
        )
        .map_err(|e| format!("{:?}", e))?;

    root.present().map_err(|e| format!("{:?}", e))?;
    std::process::Command::new("open").arg(&temp).status().map_err(|e| e.to_string())?;
    Ok(())
}

#[cfg(not(target_os = "macos"))]
pub fn run_preview_window(rx: mpsc::Receiver<PathBuf>) {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_position([1200.0, 100.0])
            .with_inner_size([600.0, 500.0]),
        ..Default::default()
    };
    let _ = eframe::run_native(
        "FITS preview",
        native_options,
        Box::new(move |cc| Ok(Box::new(PreviewApp::new(cc, rx)))),
    );
}

#[cfg(not(target_os = "macos"))]
struct PreviewApp {
    rx: mpsc::Receiver<PathBuf>,
    rgba: Option<Vec<u8>>,
    width: usize,
    height: usize,
    texture: Option<egui::TextureHandle>,
    error: Option<String>,
}

#[cfg(not(target_os = "macos"))]
impl PreviewApp {
    fn new(_cc: &eframe::CreationContext<'_>, rx: mpsc::Receiver<PathBuf>) -> Self {
        Self {
            rx,
            rgba: None,
            width: 0,
            height: 0,
            texture: None,
            error: None,
        }
    }

    fn load_path(&mut self, path: &PathBuf) {
        self.error = None;
        self.texture = None;
        match pyref::io::image_mmap::materialize_image_from_path(path.as_path()) {
            Ok((_raw, subtracted)) => {
                match pyref::colormap::array2_i64_to_rgba_rainbow(&subtracted, None) {
                    Some(rgba) => {
                        self.width = subtracted.ncols();
                        self.height = subtracted.nrows();
                        self.rgba = Some(rgba);
                    }
                    None => {
                        self.error = Some("Colormap failed (non-contiguous)".to_string());
                    }
                }
            }
            Err(e) => {
                self.error = Some(e.to_string());
            }
        }
    }

    fn update_texture(&mut self, ctx: &egui::Context) {
        if let Some(ref rgba) = self.rgba {
            if self.width > 0 && self.height > 0 && rgba.len() >= self.width * self.height * 4 {
                let size = [self.width, self.height];
                let image = egui::ColorImage::from_rgba_unmultiplied(size, rgba);
                self.texture = Some(ctx.load_texture(
                    "fits_preview",
                    image,
                    egui::TextureOptions::default(),
                ));
            }
        }
    }
}

#[cfg(not(target_os = "macos"))]
impl eframe::App for PreviewApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        while let Ok(path) = self.rx.try_recv() {
            self.load_path(&path);
        }
        if self.rgba.is_some() && self.texture.is_none() {
            self.update_texture(ctx);
        }
        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(ref err) = self.error {
                ui.colored_label(egui::Color32::RED, err.as_str());
            }
            if let Some(ref tex) = self.texture {
                ui.image(tex);
            } else if self.rgba.is_none() && self.error.is_none() {
                ui.label("Expand a profile and select a FITS file (j/k or click).");
            }
        });
    }
}

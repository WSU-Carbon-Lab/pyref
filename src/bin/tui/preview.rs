#![cfg(feature = "tui")]

#[cfg(not(target_os = "macos"))]
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::mpsc;

use ndarray::Array2;

const DEFAULT_SIGMA: f64 = 2.0;
const SIGMA_CLAMP_MIN: f64 = 0.5;
const SIGMA_CLAMP_MAX: f64 = 20.0;
const CROP_SIGMA_MULT: f64 = 4.0;

#[cfg(not(target_os = "macos"))]
use eframe::egui;

#[cfg(target_os = "macos")]
pub fn run_preview_window(rx: mpsc::Receiver<PathBuf>) {
    std::thread::spawn(move || {
        while let Ok(path) = rx.recv() {
            if let Err(e) = stream_fits_three_panel_png(&path) {
                eprintln!("Preview: {}", e);
            }
        }
    });
}

#[cfg(target_os = "macos")]
fn stream_fits_three_panel_png(path: &std::path::Path) -> Result<(), String> {
    use plotters::prelude::*;
    use plotters::series::LineSeries;

    let (_, subtracted) = pyref::io::image_mmap::materialize_image_from_path(path)
        .map_err(|e| e.to_string())?;
    let height = subtracted.nrows();
    let width = subtracted.ncols();
    if width == 0 || height == 0 {
        return Err("Invalid image dimensions".to_string());
    }
    let subtracted_f64: Array2<f64> = subtracted.mapv(|x| x as f64);
    let fit = pyref::gaussian_fit::fit_2d_gaussian(&subtracted_f64, None);
    let sigma = fit
        .map(|f| {
            let s = (f.sigma_row + f.sigma_col) / 2.0;
            s.clamp(SIGMA_CLAMP_MIN, SIGMA_CLAMP_MAX)
        })
        .unwrap_or(DEFAULT_SIGMA);
    let filtered = pyref::io::blur::blur_array2_i64(&subtracted, sigma)
        .map_err(|e| e.to_string())?;
    let (crop_img, crop_w, crop_h, _fit) = crop_4sigma(&filtered, fit, height, width);
    let left_rgba = pyref::colormap::array2_i64_to_rgba_rainbow(&subtracted, None)
        .ok_or("Colormap failed")?;
    let mid_rgba = pyref::colormap::array2_f32_to_rgba_rainbow(&filtered, None)
        .ok_or("Colormap failed")?;
    let right_rgba = pyref::colormap::array2_f32_to_rgba_rainbow(&crop_img, None)
        .ok_or("Colormap failed")?;

    const PANEL_MAX: u32 = 320;
    let scale = |w: usize, h: usize| {
        let (w, h) = (w as u32, h as u32);
        if w <= PANEL_MAX && h <= PANEL_MAX {
            (w, h)
        } else if w >= h {
            (PANEL_MAX, (h * PANEL_MAX / w).max(1))
        } else {
            ((w * PANEL_MAX / h).max(1), PANEL_MAX)
        }
    };
    let (lw, lh) = scale(width, height);
    let (mw, mh) = scale(width, height);
    let (rw, rh) = scale(crop_w, crop_h);
    let total_w = lw + mw + rw + 2;
    let total_h = lh.max(mh).max(rh);

    let temp = std::env::temp_dir().join("pyref_preview.png");
    let root = BitMapBackend::new(temp.as_path(), (total_w, total_h)).into_drawing_area();
    root.fill(&WHITE).map_err(|e| format!("{:?}", e))?;

    let draw_panel = |root: &DrawingArea<_, _>, rgba: &[u8], w: usize, h: usize, out_w: u32, out_h: u32| -> Result<(), String> {
        let mut chart = ChartBuilder::on(root)
            .margin(0)
            .set_all_label_area_size(0)
            .build_cartesian_2d(0i32..(out_w as i32), 0i32..(out_h as i32))
            .map_err(|e| format!("{:?}", e))?;
        let rgba_ref = rgba;
        chart
            .draw_series(
                (0..(out_h as usize))
                    .flat_map(|oy| {
                        (0..(out_w as usize)).map(move |ox| {
                            let sx = (ox * w) / out_w as usize;
                            let sy = (oy * h) / out_h as usize;
                            let i = (sy * w + sx) * 4;
                            let r = rgba_ref.get(i).copied().unwrap_or(0);
                            let g = rgba_ref.get(i + 1).copied().unwrap_or(0);
                            let b = rgba_ref.get(i + 2).copied().unwrap_or(0);
                            (ox as i32, (out_h as i32 - 1 - oy as i32), RGBColor(r, g, b))
                        })
                    })
                    .map(|(x, y, color)| Rectangle::new([(x, y), (x + 1, y + 1)], color.filled())),
            )
            .map_err(|e| format!("{:?}", e))?;
        Ok(())
    };

    let (area_left, rest1) = root.split_horizontally((lw + 1) as i32);
    draw_panel(&area_left, &left_rgba, width, height, lw, lh)?;
    let (area_mid, rest2) = rest1.split_horizontally((mw + 1) as i32);
    draw_panel(&area_mid, &mid_rgba, width, height, mw, mh)?;
    draw_panel(&rest2, &right_rgba, crop_w, crop_h, rw, rh)?;

    if let Some(ref f) = fit {
        let cx = rw as f64 / 2.0;
        let cy = (rh as f64 - 1.0) / 2.0;
        let scale_x = rw as f64 / crop_w as f64;
        let scale_y = rh as f64 / crop_h as f64;
        let mut chart = ChartBuilder::on(&rest2)
            .margin(0)
            .set_all_label_area_size(0)
            .build_cartesian_2d(0.0..(rw as f64), 0.0..(rh as f64))
            .map_err(|e| format!("{:?}", e))?;
        for k in [1.0, 2.0, 3.0, 4.0] {
            let pts = ellipse_contour_points(
                cx,
                cy,
                k * f.sigma_col * scale_x,
                -k * f.sigma_row * scale_y,
                64,
            );
            chart
                .draw_series(LineSeries::new(
                    pts.into_iter(),
                    plotters::style::ShapeStyle::from(&plotters::style::RGBColor(255, 255, 255))
                        .stroke_width(1),
                ))
                .map_err(|e| format!("{:?}", e))?;
        }
    }

    root.present().map_err(|e| format!("{:?}", e))?;
    std::process::Command::new("open").arg(&temp).status().map_err(|e| e.to_string())?;
    Ok(())
}

fn ellipse_contour_points(
    center_col: f64,
    center_row: f64,
    semi_col: f64,
    semi_row: f64,
    n: usize,
) -> Vec<(f64, f64)> {
    (0..=n)
        .map(|i| {
            let t = (i as f64 / n as f64) * 2.0 * std::f64::consts::PI;
            (
                center_col + semi_col * t.cos(),
                center_row + semi_row * t.sin(),
            )
        })
        .collect()
}

fn crop_4sigma(
    filtered: &Array2<f32>,
    fit: Option<pyref::gaussian_fit::Gaussian2DFit>,
    height: usize,
    width: usize,
) -> (Array2<f32>, usize, usize, Option<pyref::gaussian_fit::Gaussian2DFit>) {
    let (cr, cc, half_r, half_c) = if let Some(f) = fit {
        let hr = (CROP_SIGMA_MULT * f.sigma_row).ceil() as i32;
        let hc = (CROP_SIGMA_MULT * f.sigma_col).ceil() as i32;
        let hr = hr.max(2).min((height / 2) as i32);
        let hc = hc.max(2).min((width / 2) as i32);
        (
            f.center_row.round() as i32,
            f.center_col.round() as i32,
            hr,
            hc,
        )
    } else {
        let cr = (height / 2) as i32;
        let cc = (width / 2) as i32;
        let h = 32i32;
        (cr, cc, h, h)
    };
    let r0 = (cr - half_r).max(0).min((height - 1) as i32) as usize;
    let r1 = (cr + half_r + 1).max(0).min(height as i32) as usize;
    let c0 = (cc - half_c).max(0).min((width - 1) as i32) as usize;
    let c1 = (cc + half_c + 1).max(0).min(width as i32) as usize;
    if r0 >= r1 || c0 >= c1 {
        let fallback = Array2::from_shape_vec((1, 1), vec![0.0f32]).unwrap();
        return (fallback, 1, 1, fit);
    }
    let crop = filtered.slice(ndarray::s![r0..r1, c0..c1]).to_owned();
    let (crop_h, crop_w) = (crop.nrows(), crop.ncols());
    (crop, crop_w, crop_h, fit)
}

#[cfg(not(target_os = "macos"))]
pub fn run_preview_window(rx: mpsc::Receiver<PathBuf>) {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_position([1200.0, 100.0])
            .with_inner_size([900.0, 500.0]),
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
    current_path: Option<PathBuf>,
    roi_per_path: HashMap<PathBuf, (usize, usize, usize, usize)>,
    left_rgba: Option<Vec<u8>>,
    left_w: usize,
    left_h: usize,
    mid_rgba: Option<Vec<u8>>,
    mid_w: usize,
    mid_h: usize,
    right_rgba: Option<Vec<u8>>,
    right_w: usize,
    right_h: usize,
    fit: Option<pyref::gaussian_fit::Gaussian2DFit>,
    tex_left: Option<egui::TextureHandle>,
    tex_mid: Option<egui::TextureHandle>,
    tex_right: Option<egui::TextureHandle>,
    error: Option<String>,
    marquee_start: Option<egui::Pos2>,
    marquee_current: Option<egui::Pos2>,
    marquee_rect: Option<egui::Rect>,
    marquee_img_w: usize,
    marquee_img_h: usize,
    marquee_on_left: bool,
}

#[cfg(not(target_os = "macos"))]
impl PreviewApp {
    fn new(_cc: &eframe::CreationContext<'_>, rx: mpsc::Receiver<PathBuf>) -> Self {
        Self {
            rx,
            current_path: None,
            roi_per_path: HashMap::new(),
            left_rgba: None,
            left_w: 0,
            left_h: 0,
            mid_rgba: None,
            mid_w: 0,
            mid_h: 0,
            right_rgba: None,
            right_w: 0,
            right_h: 0,
            fit: None,
            tex_left: None,
            tex_mid: None,
            tex_right: None,
            error: None,
            marquee_start: None,
            marquee_current: None,
            marquee_rect: None,
            marquee_img_w: 0,
            marquee_img_h: 0,
            marquee_on_left: true,
        }
    }

    fn screen_to_image(
        pos: egui::Pos2,
        rect: egui::Rect,
        img_w: usize,
        img_h: usize,
    ) -> (usize, usize) {
        let x = (pos.x - rect.min.x) / rect.width();
        let y = (pos.y - rect.min.y) / rect.height();
        let col = (x * img_w as f32).round().clamp(0.0, (img_w - 1) as f32) as usize;
        let row = (y * img_h as f32).round().clamp(0.0, (img_h - 1) as f32) as usize;
        (row, col)
    }

    fn apply_marquee_roi(&mut self) {
        let path = match &self.current_path {
            Some(p) => p.clone(),
            None => return,
        };
        let (start, current, rect, w, h) = match (
            self.marquee_start,
            self.marquee_current,
            self.marquee_rect,
            self.marquee_img_w,
            self.marquee_img_h,
        ) {
            (Some(s), Some(c), Some(r), w, h) if w > 0 && h > 0 => (s, c, r, w, h),
            _ => return,
        };
        let (r0, c0) = Self::screen_to_image(start, rect, w, h);
        let (r1, c1) = Self::screen_to_image(current, rect, w, h);
        let r_min = r0.min(r1);
        let r_max = r0.max(r1);
        let c_min = c0.min(c1);
        let c_max = c0.max(c1);
        let r1_excl = (r_max + 1).min(h);
        let c1_excl = (c_max + 1).min(w);
        if r_min >= r1_excl || c_min >= c1_excl {
            self.marquee_start = None;
            self.marquee_current = None;
            self.marquee_rect = None;
            return;
        }
        self.roi_per_path
            .insert(path, (r_min, r1_excl, c_min, c1_excl));
        self.load_path(&path);
        self.marquee_start = None;
        self.marquee_current = None;
        self.marquee_rect = None;
    }

    fn load_path(&mut self, path: &PathBuf) {
        self.error = None;
        self.tex_left = None;
        self.tex_mid = None;
        self.tex_right = None;
        self.current_path = Some(path.clone());
        match pyref::io::image_mmap::materialize_image_from_path(path.as_path()) {
            Ok((_raw, subtracted)) => {
                let height = subtracted.nrows();
                let width = subtracted.ncols();
                let roi = self.roi_per_path.get(path).copied();
                let subtracted_f64: Array2<f64> = subtracted.mapv(|x| x as f64);
                let fit = pyref::gaussian_fit::fit_2d_gaussian(&subtracted_f64, roi);
                let sigma = fit
                    .map(|f| {
                        let s = (f.sigma_row + f.sigma_col) / 2.0;
                        s.clamp(SIGMA_CLAMP_MIN, SIGMA_CLAMP_MAX)
                    })
                    .unwrap_or(DEFAULT_SIGMA);
                match pyref::io::blur::blur_array2_i64(&subtracted, sigma) {
                    Ok(filtered) => {
                        self.left_rgba =
                            pyref::colormap::array2_i64_to_rgba_rainbow(&subtracted, None);
                        self.left_w = width;
                        self.left_h = height;
                        self.mid_rgba =
                            pyref::colormap::array2_f32_to_rgba_rainbow(&filtered, None);
                        self.mid_w = width;
                        self.mid_h = height;
                        let (crop, cw, ch, _) = crop_4sigma(&filtered, fit, height, width);
                        self.right_rgba =
                            pyref::colormap::array2_f32_to_rgba_rainbow(&crop, None);
                        self.right_w = cw;
                        self.right_h = ch;
                        self.fit = fit;
                    }
                    Err(e) => {
                        self.error = Some(format!("Blur failed: {}", e));
                    }
                }
            }
            Err(e) => {
                self.error = Some(e.to_string());
            }
        }
    }

    fn update_textures(&mut self, ctx: &egui::Context) {
        if let Some(ref rgba) = self.left_rgba {
            if self.left_w > 0 && self.left_h > 0 && rgba.len() >= self.left_w * self.left_h * 4 {
                let image =
                    egui::ColorImage::from_rgba_unmultiplied([self.left_w, self.left_h], rgba);
                self.tex_left = Some(ctx.load_texture(
                    "preview_left",
                    image,
                    egui::TextureOptions::default(),
                ));
            }
        }
        if let Some(ref rgba) = self.mid_rgba {
            if self.mid_w > 0 && self.mid_h > 0 && rgba.len() >= self.mid_w * self.mid_h * 4 {
                let image =
                    egui::ColorImage::from_rgba_unmultiplied([self.mid_w, self.mid_h], rgba);
                self.tex_mid = Some(ctx.load_texture(
                    "preview_mid",
                    image,
                    egui::TextureOptions::default(),
                ));
            }
        }
        if let Some(ref rgba) = self.right_rgba {
            if self.right_w > 0 && self.right_h > 0 && rgba.len() >= self.right_w * self.right_h * 4
            {
                let image =
                    egui::ColorImage::from_rgba_unmultiplied([self.right_w, self.right_h], rgba);
                self.tex_right = Some(ctx.load_texture(
                    "preview_right",
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
        if (self.left_rgba.is_some() && self.tex_left.is_none())
            || (self.mid_rgba.is_some() && self.tex_mid.is_none())
            || (self.right_rgba.is_some() && self.tex_right.is_none())
        {
            self.update_textures(ctx);
        }
        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(ref err) = self.error {
                ui.colored_label(egui::Color32::RED, err.as_str());
            }
            if self.left_rgba.is_none() && self.error.is_none() {
                ui.label("Expand a profile and select a FITS file (j/k or click).");
                return;
            }
            ui.horizontal(|ui| {
                ui.label("Raw");
                ui.add_space(ui.available_width() / 3.0 - 30.0);
                ui.label("Filtered");
                ui.add_space(ui.available_width() / 3.0 - 30.0);
                ui.label("Crop (4 sigma)");
            });
            ui.horizontal(|ui| {
                let panel_w = (ui.available_width() / 3.0).floor().max(1.0);
                if let Some(ref tex) = self.tex_left {
                    let size = [
                        panel_w,
                        panel_w * (self.left_h as f32 / self.left_w as f32).min(2.0),
                    ];
                    let response = ui.add(
                        egui::Image::new(tex.id(), size)
                            .sense(egui::Sense::drag()),
                    );
                    let rect = response.rect;
                    if response.drag_started() {
                        self.marquee_start = Some(rect.min);
                        self.marquee_current = Some(rect.min);
                        self.marquee_rect = Some(rect);
                        self.marquee_img_w = self.left_w;
                        self.marquee_img_h = self.left_h;
                        self.marquee_on_left = true;
                    }
                    if response.dragged() && self.marquee_on_left {
                        if let Some(pos) = ui.input(|i| i.pointer.pos()) {
                            self.marquee_current = Some(pos);
                        }
                    }
                    if response.drag_released() && self.marquee_on_left {
                        self.apply_marquee_roi();
                    }
                    if let (Some(a), Some(b)) = (self.marquee_start, self.marquee_current) {
                        if self.marquee_on_left {
                            let r = egui::Rect::from_min_max(a, b);
                            ui.painter().rect_stroke(
                                r.intersect(rect),
                                0.0,
                                egui::Stroke::new(2.0, egui::Color32::WHITE),
                            );
                        }
                    }
                }
                if let Some(ref tex) = self.tex_mid {
                    let size = [
                        panel_w,
                        panel_w * (self.mid_h as f32 / self.mid_w as f32).min(2.0),
                    ];
                    let response = ui.add(
                        egui::Image::new(tex.id(), size)
                            .sense(egui::Sense::drag()),
                    );
                    let rect = response.rect;
                    if response.drag_started() {
                        self.marquee_start = Some(rect.min);
                        self.marquee_current = Some(rect.min);
                        self.marquee_rect = Some(rect);
                        self.marquee_img_w = self.mid_w;
                        self.marquee_img_h = self.mid_h;
                        self.marquee_on_left = false;
                    }
                    if response.dragged() && !self.marquee_on_left {
                        if let Some(pos) = ui.input(|i| i.pointer.pos()) {
                            self.marquee_current = Some(pos);
                        }
                    }
                    if response.drag_released() && !self.marquee_on_left {
                        self.apply_marquee_roi();
                    }
                    if let (Some(a), Some(b)) = (self.marquee_start, self.marquee_current) {
                        if !self.marquee_on_left {
                            let r = egui::Rect::from_min_max(a, b);
                            ui.painter().rect_stroke(
                                r.intersect(rect),
                                0.0,
                                egui::Stroke::new(2.0, egui::Color32::WHITE),
                            );
                        }
                    }
                }
                if let Some(ref tex) = self.tex_right {
                    let size = [
                        panel_w,
                        panel_w * (self.right_h as f32 / self.right_w as f32).min(2.0),
                    ];
                    let response = ui.image(tex.id(), size);
                    if let Some(ref f) = self.fit {
                        if self.right_w > 0 && self.right_h > 0 {
                            let rect = response.rect;
                        let cx = self.right_w as f64 / 2.0;
                        let cy = self.right_h as f64 / 2.0;
                        let n = 64;
                        for k in [1.0, 2.0, 3.0, 4.0] {
                            let pts = ellipse_contour_points(
                                cx,
                                cy,
                                k * f.sigma_col,
                                k * f.sigma_row,
                                n,
                            );
                            let screen_pts: Vec<egui::Pos2> = pts
                                .iter()
                                .map(|&(px, py)| {
                                    egui::Pos2::new(
                                        rect.min.x
                                            + (px / self.right_w as f64) as f32 * rect.width(),
                                        rect.min.y
                                            + (py / self.right_h as f64) as f32 * rect.height(),
                                    )
                                })
                                .collect();
                            if screen_pts.len() >= 2 {
                                ui.painter().add(egui::Shape::line(
                                    screen_pts,
                                    egui::Stroke::new(1.0, egui::Color32::WHITE),
                                ));
                            }
                        }
                        }
                    }
                }
            });
        });
    }
}

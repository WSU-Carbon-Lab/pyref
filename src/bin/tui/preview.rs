#![cfg(feature = "tui")]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc;

use eframe::egui;
use ndarray::Array2;

type BeamspotMessage = (PathBuf, i64, i64, Option<f64>);
type PreviewRequest = (PathBuf, Option<(usize, usize, usize, usize)>, Option<f64>);

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PreviewCommand {
    GoToNextProblem,
    GoToPrevProblem,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum InteractionMode {
    Marquee,
    Zoom,
}

const DEFAULT_SIGMA: f64 = 2.0;
const SIGMA_CLAMP_MIN: f64 = 0.5;
const SIGMA_CLAMP_MAX: f64 = 20.0;
const CROP_SIGMA_MULT: f64 = 4.0;
const MIN_THIRD_COLUMN_WIDTH: f32 = 120.0;

fn extract_rgba_slice(
    rgba: &[u8],
    full_w: usize,
    full_h: usize,
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
) -> Vec<u8> {
    let r1 = r1.min(full_h);
    let c1 = c1.min(full_w);
    let r0 = r0.min(r1);
    let c0 = c0.min(c1);
    let h = r1.saturating_sub(r0);
    let w = c1.saturating_sub(c0);
    let mut out = Vec::with_capacity(h * w * 4);
    for row in r0..r1 {
        for col in c0..c1 {
            let i = (row * full_w + col) * 4;
            if i + 4 <= rgba.len() {
                out.extend_from_slice(&rgba[i..i + 4]);
            }
        }
    }
    out
}

pub enum LoadResult {
    Ok(PreviewData),
    Err(String),
}

pub struct PreviewData {
    pub path: PathBuf,
    pub left_rgba: Vec<u8>,
    pub left_w: usize,
    pub left_h: usize,
    pub mid_rgba: Vec<u8>,
    pub mid_w: usize,
    pub mid_h: usize,
    pub right_rgba: Vec<u8>,
    pub right_w: usize,
    pub right_h: usize,
    pub crop_col_sum: Vec<f32>,
    pub crop_row_sum: Vec<f32>,
    pub fit: Option<pyref::gaussian_fit::Gaussian2DFit>,
}

fn load_preview_data(
    path: &Path,
    roi: Option<(usize, usize, usize, usize)>,
    profile_sigma: Option<f64>,
) -> Result<PreviewData, String> {
    let (_, subtracted) = pyref::io::image_mmap::materialize_image_from_path(path)
        .map_err(|e| e.to_string())?;
    let height = subtracted.nrows();
    let width = subtracted.ncols();
    let subtracted_f64: Array2<f64> = subtracted.mapv(|x| x as f64);
    let fit = pyref::gaussian_fit::fit_2d_gaussian(&subtracted_f64, roi);
    let sigma = profile_sigma
        .map(|s| s.clamp(SIGMA_CLAMP_MIN, SIGMA_CLAMP_MAX))
        .or_else(|| {
            fit.map(|f| {
                let s = (f.sigma_row + f.sigma_col) / 2.0;
                s.clamp(SIGMA_CLAMP_MIN, SIGMA_CLAMP_MAX)
            })
        })
        .unwrap_or(DEFAULT_SIGMA);
    let filtered = pyref::io::blur::blur_array2_i64(&subtracted, sigma)
        .map_err(|e| format!("Blur failed: {}", e))?;
    let left_max = subtracted.iter().copied().max().unwrap_or(1).max(1);
    let mid_max = filtered
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max)
        .max(1.0);
    let left_rgba = pyref::colormap::array2_i64_to_rgba(&subtracted, Some((0, left_max)), true)
        .ok_or("Colormap failed")?;
    let mid_rgba =
        pyref::colormap::array2_f32_to_rgba(&filtered, Some((0.0, mid_max)), true)
            .ok_or("Colormap failed")?;
    let (crop, cw, ch, _) = crop_4sigma(&filtered, fit, height, width);
    let right_max = crop
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max)
        .max(1.0);
    let right_rgba =
        pyref::colormap::array2_f32_to_rgba(&crop, Some((0.0, right_max)), true)
            .ok_or("Colormap failed")?;
    let crop_col_sum = (0..cw).map(|j| (0..ch).map(|i| crop[[i, j]]).sum()).collect();
    let crop_row_sum = (0..ch).map(|i| (0..cw).map(|j| crop[[i, j]]).sum()).collect();
    Ok(PreviewData {
        path: path.to_path_buf(),
        left_rgba,
        left_w: width,
        left_h: height,
        mid_rgba,
        mid_w: width,
        mid_h: height,
        right_rgba,
        right_w: cw,
        right_h: ch,
        crop_col_sum,
        crop_row_sum,
        fit,
    })
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
        let half = hr.max(hc).max(2).min((height / 2).min(width / 2) as i32);
        (
            f.center_row.round() as i32,
            f.center_col.round() as i32,
            half,
            half,
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
        let fb_half = 16i32;
        let fr0 = ((height as i32 / 2) - fb_half).max(0) as usize;
        let fr1 = ((height as i32 / 2) + fb_half).min(height as i32) as usize;
        let fc0 = ((width as i32 / 2) - fb_half).max(0) as usize;
        let fc1 = ((width as i32 / 2) + fb_half).min(width as i32) as usize;
        if fr0 < fr1 && fc0 < fc1 {
            let crop = filtered.slice(ndarray::s![fr0..fr1, fc0..fc1]).to_owned();
            let (ch, cw) = (crop.nrows(), crop.ncols());
            return (crop, cw, ch, fit);
        }
        let fallback = Array2::from_shape_vec((2, 2), vec![0.0f32; 4])
            .expect("fallback 2x2 shape is valid");
        return (fallback, 2, 2, fit);
    }
    let crop = filtered.slice(ndarray::s![r0..r1, c0..c1]).to_owned();
    let (crop_h, crop_w) = (crop.nrows(), crop.ncols());
    (crop, crop_w, crop_h, fit)
}

#[allow(dead_code)]
pub fn run_preview_window(rx: mpsc::Receiver<(PathBuf, Option<f64>)>) {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_position([1200.0, 100.0])
            .with_inner_size([900.0, 500.0]),
        ..Default::default()
    };
    let _ = eframe::run_native(
        "FITS preview",
        native_options,
        Box::new(move |cc| Ok(Box::new(PreviewApp::new(cc, rx, None, None)))),
    );
}

pub fn run_preview_window_on_first_path(
    rx: mpsc::Receiver<(PathBuf, Option<f64>)>,
    beamspot_tx: Option<mpsc::Sender<BeamspotMessage>>,
    cmd_tx: Option<mpsc::Sender<PreviewCommand>>,
) {
    let (first_path, first_sigma) = match rx.recv() {
        Ok(p) => p,
        Err(_) => return,
    };
    let (forward_tx, forward_rx) = mpsc::channel();
    let _ = forward_tx.send((first_path, first_sigma));
    std::thread::spawn(move || {
        while let Ok(msg) = rx.recv() {
            let _ = forward_tx.send(msg);
        }
    });
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_position([1200.0, 100.0])
            .with_inner_size([900.0, 500.0]),
        ..Default::default()
    };
    let _ = eframe::run_native(
        "FITS preview",
        native_options,
        Box::new(move |cc| {
            Ok(Box::new(PreviewApp::new(cc, forward_rx, beamspot_tx, cmd_tx)))
        }),
    );
}

struct PreviewApp {
    rx: mpsc::Receiver<(PathBuf, Option<f64>)>,
    path_tx: mpsc::Sender<PreviewRequest>,
    result_rx: mpsc::Receiver<LoadResult>,
    loading: bool,
    last_requested_path: Option<PathBuf>,
    current_path: Option<PathBuf>,
    profile_sigma: Option<f64>,
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
    crop_col_sum: Vec<f32>,
    crop_row_sum: Vec<f32>,
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
    beamspot_tx: Option<mpsc::Sender<BeamspotMessage>>,
    send_beamspot_on_next_apply: bool,
    zoom_per_path: HashMap<PathBuf, (usize, usize, usize, usize)>,
    interaction_mode: InteractionMode,
    cmd_tx: Option<mpsc::Sender<PreviewCommand>>,
}

impl PreviewApp {
    fn new(
        _cc: &eframe::CreationContext<'_>,
        rx: mpsc::Receiver<(PathBuf, Option<f64>)>,
        beamspot_tx: Option<mpsc::Sender<BeamspotMessage>>,
        cmd_tx: Option<mpsc::Sender<PreviewCommand>>,
    ) -> Self {
        let (path_tx, path_rx): (mpsc::Sender<PreviewRequest>, mpsc::Receiver<PreviewRequest>) =
            mpsc::channel();
        let (result_tx, result_rx) = mpsc::channel();
        std::thread::spawn(move || {
            while let Ok((path, roi, profile_sigma)) = path_rx.recv() {
                let result = match load_preview_data(&path, roi, profile_sigma) {
                    Ok(data) => LoadResult::Ok(data),
                    Err(e) => LoadResult::Err(e),
                };
                let _ = result_tx.send(result);
            }
        });
        Self {
            rx,
            path_tx,
            result_rx,
            loading: false,
            last_requested_path: None,
            current_path: None,
            profile_sigma: None,
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
            crop_col_sum: Vec::new(),
            crop_row_sum: Vec::new(),
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
            beamspot_tx,
            send_beamspot_on_next_apply: false,
            zoom_per_path: HashMap::new(),
            interaction_mode: InteractionMode::Marquee,
            cmd_tx,
        }
    }

    fn screen_to_image(
        pos: egui::Pos2,
        rect: egui::Rect,
        img_w: usize,
        img_h: usize,
        visible: Option<(usize, usize, usize, usize)>,
    ) -> (usize, usize) {
        let x = (pos.x - rect.min.x) / rect.width();
        let y = (pos.y - rect.min.y) / rect.height();
        let (row, col) = match visible {
            Some((r0, r1, c0, c1)) => {
                let vis_h = r1.saturating_sub(r0).max(1);
                let vis_w = c1.saturating_sub(c0).max(1);
                let local_r = (y * vis_h as f32).round().clamp(0.0, (vis_h - 1) as f32) as usize;
                let local_c = (x * vis_w as f32).round().clamp(0.0, (vis_w - 1) as f32) as usize;
                (r0 + local_r, c0 + local_c)
            }
            None => {
                let col = (x * img_w as f32).round().clamp(0.0, (img_w - 1) as f32) as usize;
                let row = (y * img_h as f32).round().clamp(0.0, (img_h - 1) as f32) as usize;
                (row, col)
            }
        };
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
        let visible = self.zoom_per_path.get(&path).copied();
        let (r0, c0) = Self::screen_to_image(start, rect, w, h, visible);
        let (r1, c1) = Self::screen_to_image(current, rect, w, h, visible);
        let r_min = r0.min(r1);
        let r_max = r0.max(r1);
        let c_min = c0.min(c1);
        let c_max = c0.max(c1);
        let (full_w, full_h) = (self.left_w, self.left_h);
        let r1_excl = (r_max + 1).min(full_h);
        let c1_excl = (c_max + 1).min(full_w);
        if r_min >= r1_excl || c_min >= c1_excl {
            self.marquee_start = None;
            self.marquee_current = None;
            self.marquee_rect = None;
            return;
        }
        let path_ref = path.clone();
        self.roi_per_path
            .insert(path, (r_min, r1_excl, c_min, c1_excl));
        let roi = (r_min, r1_excl, c_min, c1_excl);
        let _ = self
            .path_tx
            .send((path_ref.clone(), Some(roi), self.profile_sigma));
        self.loading = true;
        self.last_requested_path = Some(path_ref.clone());
        self.send_beamspot_on_next_apply = true;
        self.clear_image_state();
        self.marquee_start = None;
        self.marquee_current = None;
        self.marquee_rect = None;
    }

    fn clear_image_state(&mut self) {
        self.error = None;
        self.tex_left = None;
        self.tex_mid = None;
        self.tex_right = None;
        self.left_rgba = None;
        self.left_w = 0;
        self.left_h = 0;
        self.mid_rgba = None;
        self.mid_w = 0;
        self.mid_h = 0;
        self.right_rgba = None;
        self.right_w = 0;
        self.right_h = 0;
        self.crop_col_sum.clear();
        self.crop_row_sum.clear();
        self.fit = None;
    }

    fn apply_preview_data(&mut self, data: PreviewData) {
        self.current_path = Some(data.path.clone());
        self.left_rgba = Some(data.left_rgba);
        self.left_w = data.left_w;
        self.left_h = data.left_h;
        self.mid_rgba = Some(data.mid_rgba);
        self.mid_w = data.mid_w;
        self.mid_h = data.mid_h;
        self.right_rgba = Some(data.right_rgba);
        self.right_w = data.right_w;
        self.right_h = data.right_h;
        self.crop_col_sum = data.crop_col_sum;
        self.crop_row_sum = data.crop_row_sum;
        self.fit = data.fit;
        self.error = None;
        self.tex_left = None;
        self.tex_mid = None;
        self.tex_right = None;
    }

    fn load_path(&mut self, path: &PathBuf) {
        let _ = self.path_tx.send((
            path.clone(),
            self.roi_per_path.get(path).copied(),
            self.profile_sigma,
        ));
        self.loading = true;
        self.last_requested_path = Some(path.clone());
        self.clear_image_state();
        self.marquee_start = None;
        self.marquee_current = None;
        self.marquee_rect = None;
        self.current_path = Some(path.clone());
    }

    fn update_textures(&mut self, ctx: &egui::Context) {
        let zoom = self
            .current_path
            .as_ref()
            .and_then(|p| self.zoom_per_path.get(p).copied());
        if let Some(ref rgba) = self.left_rgba {
            if self.left_w > 0 && self.left_h > 0 && rgba.len() >= self.left_w * self.left_h * 4 {
                let (tex_w, tex_h, data) = match zoom {
                    Some((r0, r1, c0, c1)) => {
                        let w = c1.saturating_sub(c0).max(1);
                        let h = r1.saturating_sub(r0).max(1);
                        let slice =
                            extract_rgba_slice(rgba, self.left_w, self.left_h, r0, r1, c0, c1);
                        (w, h, slice)
                    }
                    None => (self.left_w, self.left_h, rgba.clone()),
                };
                if !data.is_empty() && data.len() >= tex_w * tex_h * 4 {
                    let image = egui::ColorImage::from_rgba_unmultiplied([tex_w, tex_h], &data);
                    self.tex_left = Some(ctx.load_texture(
                        "preview_left",
                        image,
                        egui::TextureOptions::default(),
                    ));
                }
            }
        }
        if let Some(ref rgba) = self.mid_rgba {
            if self.mid_w > 0 && self.mid_h > 0 && rgba.len() >= self.mid_w * self.mid_h * 4 {
                let (tex_w, tex_h, data) = match zoom {
                    Some((r0, r1, c0, c1)) => {
                        let w = c1.saturating_sub(c0).max(1);
                        let h = r1.saturating_sub(r0).max(1);
                        let slice =
                            extract_rgba_slice(rgba, self.mid_w, self.mid_h, r0, r1, c0, c1);
                        (w, h, slice)
                    }
                    None => (self.mid_w, self.mid_h, rgba.clone()),
                };
                if !data.is_empty() && data.len() >= tex_w * tex_h * 4 {
                    let image = egui::ColorImage::from_rgba_unmultiplied([tex_w, tex_h], &data);
                    self.tex_mid = Some(ctx.load_texture(
                        "preview_mid",
                        image,
                        egui::TextureOptions::default(),
                    ));
                }
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

impl eframe::App for PreviewApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        while let Ok((path, sigma)) = self.rx.try_recv() {
            self.profile_sigma = sigma;
            self.load_path(&path);
        }
        while let Ok(result) = self.result_rx.try_recv() {
            self.loading = false;
            match result {
                LoadResult::Ok(data) => {
                    if self.last_requested_path.as_ref() == Some(&data.path) {
                        if self.send_beamspot_on_next_apply {
                            if let (Some(f), Some(tx)) =
                                (data.fit.as_ref(), self.beamspot_tx.as_ref())
                            {
                                let sigma = ((f.sigma_row + f.sigma_col) / 2.0)
                                    .clamp(SIGMA_CLAMP_MIN, SIGMA_CLAMP_MAX);
                                let _ = tx.send((
                                    data.path.clone(),
                                    f.center_row.round() as i64,
                                    f.center_col.round() as i64,
                                    Some(sigma),
                                ));
                                self.send_beamspot_on_next_apply = false;
                            }
                        }
                        self.apply_preview_data(data);
                    }
                }
                LoadResult::Err(e) => {
                    self.error = Some(e);
                }
            }
        }
        if (self.left_rgba.is_some() && self.tex_left.is_none())
            || (self.mid_rgba.is_some() && self.tex_mid.is_none())
            || (self.right_rgba.is_some() && self.tex_right.is_none())
        {
            self.update_textures(ctx);
        }
        egui::CentralPanel::default().show(ctx, |ui| {
            if ui.input(|i| i.key_pressed(egui::Key::Z)) {
                self.interaction_mode = InteractionMode::Zoom;
            }
            if ui.input(|i| i.key_pressed(egui::Key::M)) {
                self.interaction_mode = InteractionMode::Marquee;
            }
            if let Some(ref path) = self.current_path {
                if ui.input(|i| i.key_pressed(egui::Key::R) || i.key_pressed(egui::Key::Escape)) {
                    if self.zoom_per_path.remove(path).is_some() {
                        self.tex_left = None;
                        self.tex_mid = None;
                        ctx.request_repaint();
                    }
                }
            }
            if let Some(ref err) = self.error {
                ui.colored_label(egui::Color32::RED, err.as_str());
            }
            if self.loading && self.left_rgba.is_none() {
                ui.label("Loading...");
                return;
            }
            if self.left_rgba.is_none() && self.error.is_none() {
                ui.label("Expand a profile and select a FITS file (j/k or click).");
                return;
            }
            let panel_w = (ui.available_width() / 3.0).floor().max(1.0);
            let label_height = 24.0;
            ui.horizontal(|ui| {
                ui.allocate_ui_with_layout(
                    egui::vec2(panel_w, label_height),
                    egui::Layout::top_down(egui::Align::Min).with_cross_align(egui::Align::Center),
                    |ui| {
                        ui.label("Raw");
                    },
                );
                ui.allocate_ui_with_layout(
                    egui::vec2(panel_w, label_height),
                    egui::Layout::top_down(egui::Align::Min).with_cross_align(egui::Align::Center),
                    |ui| {
                        ui.label("Filtered");
                    },
                );
                ui.allocate_ui_with_layout(
                    egui::vec2(panel_w, label_height),
                    egui::Layout::top_down(egui::Align::Min).with_cross_align(egui::Align::Center),
                    |ui| {
                        ui.label("Crop (4 sigma)");
                    },
                );
            });
            ui.horizontal(|ui| {
                let marquee_active = matches!(self.interaction_mode, InteractionMode::Marquee);
                let zoom_active = matches!(self.interaction_mode, InteractionMode::Zoom);
                let marquee_btn = egui::Button::new("Marquee (M)").fill(if marquee_active {
                    ui.visuals().selection.bg_fill
                } else {
                    egui::Color32::TRANSPARENT
                });
                if ui.add(marquee_btn).clicked() {
                    self.interaction_mode = InteractionMode::Marquee;
                }
                let zoom_btn = egui::Button::new("Zoom (Z)").fill(if zoom_active {
                    ui.visuals().selection.bg_fill
                } else {
                    egui::Color32::TRANSPARENT
                });
                if ui.add(zoom_btn).clicked() {
                    self.interaction_mode = InteractionMode::Zoom;
                }
                let zoomed = self
                    .current_path
                    .as_ref()
                    .is_some_and(|p| self.zoom_per_path.contains_key(p));
                if zoomed {
                    if ui.button("Reset zoom (R/Esc)").clicked() {
                        if let Some(ref path) = self.current_path {
                            self.zoom_per_path.remove(path);
                            self.tex_left = None;
                            self.tex_mid = None;
                            ctx.request_repaint();
                        }
                    }
                }
                if let Some(ref tx) = self.cmd_tx {
                    if ui.button("Prev (P)").clicked() {
                        let _ = tx.send(PreviewCommand::GoToPrevProblem);
                    }
                    if ui.button("Next (N)").clicked() {
                        let _ = tx.send(PreviewCommand::GoToNextProblem);
                    }
                }
            });
            ui.horizontal(|ui| {
                if let Some(ref tex) = self.tex_left {
                    let size = egui::vec2(
                        panel_w,
                        panel_w * (self.left_h as f32 / self.left_w as f32).min(2.0),
                    );
                    let response = ui.add(
                        egui::Image::from_texture((tex.id(), size))
                            .sense(egui::Sense::drag()),
                    );
                    let rect = response.rect;
                    if response.drag_started() {
                        let pointer_pos = response
                            .interact_pointer_pos()
                            .or_else(|| ui.input(|i| i.pointer.latest_pos()));
                        if let Some(pos) = pointer_pos {
                            self.marquee_start = Some(pos);
                            self.marquee_current = Some(pos);
                            self.marquee_rect = Some(rect);
                            let (img_w, img_h) = self
                                .current_path
                                .as_ref()
                                .and_then(|p| self.zoom_per_path.get(p))
                                .map(|(r0, r1, c0, c1)| {
                                    (
                                        c1.saturating_sub(*c0).max(1),
                                        r1.saturating_sub(*r0).max(1),
                                    )
                                })
                                .unwrap_or((self.left_w, self.left_h));
                            self.marquee_img_w = img_w;
                            self.marquee_img_h = img_h;
                            self.marquee_on_left = true;
                        }
                    }
                    if response.dragged() && self.marquee_on_left {
                        if let Some(pos) = ui.input(|i| i.pointer.latest_pos()) {
                            self.marquee_current = Some(pos);
                        }
                    }
                    if response.drag_stopped() && self.marquee_on_left {
                        match self.interaction_mode {
                            InteractionMode::Marquee => self.apply_marquee_roi(),
                            InteractionMode::Zoom => {
                                if let (Some(path), Some(start), Some(current), Some(rect)) = (
                                    self.current_path.clone(),
                                    self.marquee_start,
                                    self.marquee_current,
                                    self.marquee_rect,
                                ) {
                                    if self.marquee_img_w > 0 && self.marquee_img_h > 0 {
                                        let visible =
                                            self.zoom_per_path.get(&path).copied();
                                        let (r0, c0) = Self::screen_to_image(
                                            start,
                                            rect,
                                            self.marquee_img_w,
                                            self.marquee_img_h,
                                            visible,
                                        );
                                        let (r1, c1) = Self::screen_to_image(
                                            current,
                                            rect,
                                            self.marquee_img_w,
                                            self.marquee_img_h,
                                            visible,
                                        );
                                        let r_min = r0.min(r1);
                                        let r_max = r0.max(r1);
                                        let c_min = c0.min(c1);
                                        let c_max = c0.max(c1);
                                        let r1_excl = (r_max + 1).min(self.left_h);
                                        let c1_excl = (c_max + 1).min(self.left_w);
                                        if r_min < r1_excl && c_min < c1_excl {
                                            self.zoom_per_path
                                                .insert(path, (r_min, r1_excl, c_min, c1_excl));
                                            self.tex_left = None;
                                            self.tex_mid = None;
                                            ctx.request_repaint();
                                        }
                                    }
                                }
                                self.marquee_start = None;
                                self.marquee_current = None;
                                self.marquee_rect = None;
                            }
                        }
                    }
                    if let (Some(a), Some(b)) = (self.marquee_start, self.marquee_current) {
                        if self.marquee_on_left {
                            let r = egui::Rect::from_min_max(a, b);
                            if r.width() > 0.0 && r.height() > 0.0 {
                                ui.painter().rect_stroke(
                                    r.intersect(rect),
                                    0.0,
                                    egui::Stroke::new(2.0, egui::Color32::WHITE),
                                );
                            }
                        }
                    }
                    if let Some(ref f) = self.fit {
                        let zoom = self
                            .current_path
                            .as_ref()
                            .and_then(|p| self.zoom_per_path.get(p));
                        let (cx, cy, draw_crop) = match zoom {
                            Some(&(r0, r1, c0, c1)) => {
                                let zw = (c1.saturating_sub(c0)) as f32;
                                let zh = (r1.saturating_sub(r0)) as f32;
                                if zw > 0.0 && zh > 0.0 {
                                    let fc = f.center_col as f32;
                                    let fr = f.center_row as f32;
                                    let in_view = fc >= c0 as f32
                                        && fc < c1 as f32
                                        && fr >= r0 as f32
                                        && fr < r1 as f32;
                                    if in_view {
                                        let cx = rect.min.x
                                            + (fc - c0 as f32) / zw * rect.width();
                                        let cy = rect.min.y
                                            + (fr - r0 as f32) / zh * rect.height();
                                        let half = (CROP_SIGMA_MULT * f.sigma_row.max(f.sigma_col))
                                            .ceil()
                                            .max(2.0)
                                            .min((self.left_h / 2).min(self.left_w / 2) as f64)
                                            as f32;
                                        let cr0 = (fr - half).max(r0 as f32).min(r1 as f32 - 1.0);
                                        let cr1 = (fr + half).max(r0 as f32).min(r1 as f32 - 1.0);
                                        let cc0 = (fc - half).max(c0 as f32).min(c1 as f32 - 1.0);
                                        let cc1 = (fc + half).max(c0 as f32).min(c1 as f32 - 1.0);
                                        let x0 =
                                            rect.min.x + (cc0 - c0 as f32) / zw * rect.width();
                                        let x1 =
                                            rect.min.x + (cc1 - c0 as f32) / zw * rect.width();
                                        let y0 =
                                            rect.min.y + (cr0 - r0 as f32) / zh * rect.height();
                                        let y1 =
                                            rect.min.y + (cr1 - r0 as f32) / zh * rect.height();
                                        (cx, cy, Some((x0, y0, x1, y1)))
                                    } else {
                                        (0.0, 0.0, None)
                                    }
                                } else {
                                    (0.0, 0.0, None)
                                }
                            }
                            None => {
                                let cx = rect.min.x
                                    + (f.center_col as f32 / self.left_w as f32) * rect.width();
                                let cy = rect.min.y
                                    + (f.center_row as f32 / self.left_h as f32) * rect.height();
                                let half = (CROP_SIGMA_MULT * f.sigma_row.max(f.sigma_col))
                                    .ceil()
                                    .max(2.0)
                                    .min((self.left_h / 2).min(self.left_w / 2) as f64) as f32;
                                let c0 = (f.center_col as f32 - half)
                                    .max(0.0)
                                    .min((self.left_w - 1) as f32);
                                let c1 = (f.center_col as f32 + half)
                                    .max(0.0)
                                    .min((self.left_w - 1) as f32);
                                let r0 = (f.center_row as f32 - half)
                                    .max(0.0)
                                    .min((self.left_h - 1) as f32);
                                let r1 = (f.center_row as f32 + half)
                                    .max(0.0)
                                    .min((self.left_h - 1) as f32);
                                let x0 = rect.min.x + (c0 / self.left_w as f32) * rect.width();
                                let x1 = rect.min.x + (c1 / self.left_w as f32) * rect.width();
                                let y0 = rect.min.y + (r0 / self.left_h as f32) * rect.height();
                                let y1 = rect.min.y + (r1 / self.left_h as f32) * rect.height();
                                (cx, cy, Some((x0, y0, x1, y1)))
                            }
                        };
                        if rect.width() > 0.0 && rect.height() > 0.0
                            && (zoom.is_none()
                                || (cx >= rect.min.x && cx <= rect.max.x
                                    && cy >= rect.min.y && cy <= rect.max.y))
                        {
                            let stroke = egui::Stroke::new(1.0, egui::Color32::WHITE);
                            ui.painter().line_segment(
                                [
                                    egui::Pos2::new(rect.min.x, cy),
                                    egui::Pos2::new(rect.max.x, cy),
                                ],
                                stroke,
                            );
                            ui.painter().line_segment(
                                [
                                    egui::Pos2::new(cx, rect.min.y),
                                    egui::Pos2::new(cx, rect.max.y),
                                ],
                                stroke,
                            );
                        }
                        if let Some((x0, y0, x1, y1)) = draw_crop {
                            ui.painter().rect_stroke(
                                egui::Rect::from_min_max(
                                    egui::Pos2::new(x0, y0),
                                    egui::Pos2::new(x1, y1),
                                ),
                                0.0,
                                egui::Stroke::new(1.0, egui::Color32::WHITE),
                            );
                        }
                    }
                }
                if let Some(ref tex) = self.tex_mid {
                    let size = egui::vec2(
                        panel_w,
                        panel_w * (self.mid_h as f32 / self.mid_w as f32).min(2.0),
                    );
                    let response = ui.add(
                        egui::Image::from_texture((tex.id(), size))
                            .sense(egui::Sense::drag()),
                    );
                    let rect = response.rect;
                    if response.drag_started() {
                        let pointer_pos = response
                            .interact_pointer_pos()
                            .or_else(|| ui.input(|i| i.pointer.latest_pos()));
                        if let Some(pos) = pointer_pos {
                            self.marquee_start = Some(pos);
                            self.marquee_current = Some(pos);
                            self.marquee_rect = Some(rect);
                            let (img_w, img_h) = self
                                .current_path
                                .as_ref()
                                .and_then(|p| self.zoom_per_path.get(p))
                                .map(|(r0, r1, c0, c1)| {
                                    (
                                        c1.saturating_sub(*c0).max(1),
                                        r1.saturating_sub(*r0).max(1),
                                    )
                                })
                                .unwrap_or((self.mid_w, self.mid_h));
                            self.marquee_img_w = img_w;
                            self.marquee_img_h = img_h;
                            self.marquee_on_left = false;
                        }
                    }
                    if response.dragged() && !self.marquee_on_left {
                        if let Some(pos) = ui.input(|i| i.pointer.latest_pos()) {
                            self.marquee_current = Some(pos);
                        }
                    }
                    if response.drag_stopped() && !self.marquee_on_left {
                        match self.interaction_mode {
                            InteractionMode::Marquee => self.apply_marquee_roi(),
                            InteractionMode::Zoom => {
                                if let (Some(path), Some(start), Some(current), Some(rect)) = (
                                    self.current_path.clone(),
                                    self.marquee_start,
                                    self.marquee_current,
                                    self.marquee_rect,
                                ) {
                                    if self.marquee_img_w > 0 && self.marquee_img_h > 0 {
                                        let visible =
                                            self.zoom_per_path.get(&path).copied();
                                        let (r0, c0) = Self::screen_to_image(
                                            start,
                                            rect,
                                            self.marquee_img_w,
                                            self.marquee_img_h,
                                            visible,
                                        );
                                        let (r1, c1) = Self::screen_to_image(
                                            current,
                                            rect,
                                            self.marquee_img_w,
                                            self.marquee_img_h,
                                            visible,
                                        );
                                        let r_min = r0.min(r1);
                                        let r_max = r0.max(r1);
                                        let c_min = c0.min(c1);
                                        let c_max = c0.max(c1);
                                        let r1_excl = (r_max + 1).min(self.mid_h);
                                        let c1_excl = (c_max + 1).min(self.mid_w);
                                        if r_min < r1_excl && c_min < c1_excl {
                                            self.zoom_per_path
                                                .insert(path, (r_min, r1_excl, c_min, c1_excl));
                                            self.tex_left = None;
                                            self.tex_mid = None;
                                            ctx.request_repaint();
                                        }
                                    }
                                }
                                self.marquee_start = None;
                                self.marquee_current = None;
                                self.marquee_rect = None;
                            }
                        }
                    }
                    if let (Some(a), Some(b)) = (self.marquee_start, self.marquee_current) {
                        if !self.marquee_on_left {
                            let r = egui::Rect::from_min_max(a, b);
                            if r.width() > 0.0 && r.height() > 0.0 {
                                ui.painter().rect_stroke(
                                    r.intersect(rect),
                                    0.0,
                                    egui::Stroke::new(2.0, egui::Color32::WHITE),
                                );
                            }
                        }
                    }
                    if let Some(ref f) = self.fit {
                        let zoom = self
                            .current_path
                            .as_ref()
                            .and_then(|p| self.zoom_per_path.get(p));
                        let (cx, cy, draw_crop) = match zoom {
                            Some(&(r0, r1, c0, c1)) => {
                                let zw = (c1.saturating_sub(c0)) as f32;
                                let zh = (r1.saturating_sub(r0)) as f32;
                                if zw > 0.0 && zh > 0.0 {
                                    let fc = f.center_col as f32;
                                    let fr = f.center_row as f32;
                                    let in_view = fc >= c0 as f32
                                        && fc < c1 as f32
                                        && fr >= r0 as f32
                                        && fr < r1 as f32;
                                    if in_view {
                                        let cx = rect.min.x
                                            + (fc - c0 as f32) / zw * rect.width();
                                        let cy = rect.min.y
                                            + (fr - r0 as f32) / zh * rect.height();
                                        let half = (CROP_SIGMA_MULT * f.sigma_row.max(f.sigma_col))
                                            .ceil()
                                            .max(2.0)
                                            .min((self.mid_h / 2).min(self.mid_w / 2) as f64)
                                            as f32;
                                        let cr0 = (fr - half).max(r0 as f32).min(r1 as f32 - 1.0);
                                        let cr1 = (fr + half).max(r0 as f32).min(r1 as f32 - 1.0);
                                        let cc0 = (fc - half).max(c0 as f32).min(c1 as f32 - 1.0);
                                        let cc1 = (fc + half).max(c0 as f32).min(c1 as f32 - 1.0);
                                        let x0 =
                                            rect.min.x + (cc0 - c0 as f32) / zw * rect.width();
                                        let x1 =
                                            rect.min.x + (cc1 - c0 as f32) / zw * rect.width();
                                        let y0 =
                                            rect.min.y + (cr0 - r0 as f32) / zh * rect.height();
                                        let y1 =
                                            rect.min.y + (cr1 - r0 as f32) / zh * rect.height();
                                        (cx, cy, Some((x0, y0, x1, y1)))
                                    } else {
                                        (0.0, 0.0, None)
                                    }
                                } else {
                                    (0.0, 0.0, None)
                                }
                            }
                            None => {
                                let cx = rect.min.x
                                    + (f.center_col as f32 / self.mid_w as f32) * rect.width();
                                let cy = rect.min.y
                                    + (f.center_row as f32 / self.mid_h as f32) * rect.height();
                                let half = (CROP_SIGMA_MULT * f.sigma_row.max(f.sigma_col))
                                    .ceil()
                                    .max(2.0)
                                    .min((self.mid_h / 2).min(self.mid_w / 2) as f64) as f32;
                                let c0 = (f.center_col as f32 - half)
                                    .max(0.0)
                                    .min((self.mid_w - 1) as f32);
                                let c1 = (f.center_col as f32 + half)
                                    .max(0.0)
                                    .min((self.mid_w - 1) as f32);
                                let r0 = (f.center_row as f32 - half)
                                    .max(0.0)
                                    .min((self.mid_h - 1) as f32);
                                let r1 = (f.center_row as f32 + half)
                                    .max(0.0)
                                    .min((self.mid_h - 1) as f32);
                                let x0 = rect.min.x + (c0 / self.mid_w as f32) * rect.width();
                                let x1 = rect.min.x + (c1 / self.mid_w as f32) * rect.width();
                                let y0 = rect.min.y + (r0 / self.mid_h as f32) * rect.height();
                                let y1 = rect.min.y + (r1 / self.mid_h as f32) * rect.height();
                                (cx, cy, Some((x0, y0, x1, y1)))
                            }
                        };
                        if rect.width() > 0.0 && rect.height() > 0.0
                            && (zoom.is_none()
                                || (cx >= rect.min.x && cx <= rect.max.x
                                    && cy >= rect.min.y && cy <= rect.max.y))
                        {
                            let stroke = egui::Stroke::new(1.0, egui::Color32::WHITE);
                            ui.painter().line_segment(
                                [
                                    egui::Pos2::new(rect.min.x, cy),
                                    egui::Pos2::new(rect.max.x, cy),
                                ],
                                stroke,
                            );
                            ui.painter().line_segment(
                                [
                                    egui::Pos2::new(cx, rect.min.y),
                                    egui::Pos2::new(cx, rect.max.y),
                                ],
                                stroke,
                            );
                        }
                        if let Some((x0, y0, x1, y1)) = draw_crop {
                            ui.painter().rect_stroke(
                                egui::Rect::from_min_max(
                                    egui::Pos2::new(x0, y0),
                                    egui::Pos2::new(x1, y1),
                                ),
                                0.0,
                                egui::Stroke::new(1.0, egui::Color32::WHITE),
                            );
                        }
                    }
                }
                if let Some(ref tex) = self.tex_right {
                    let panel_w = (ui.available_width() / 3.0)
                        .floor()
                        .max(MIN_THIRD_COLUMN_WIDTH);
                    let trace_height = 48.0f32;
                    let trace_width = 48.0f32;
                    let crop_size = (panel_w - trace_width - ui.spacing().item_spacing.x).max(1.0);

                    let draw_col_trace =
                        |ui: &mut egui::Ui, data: &[f32], w: f32, h: f32| {
                            let (rect, _) =
                                ui.allocate_exact_size(egui::vec2(w, h), egui::Sense::hover());
                            let min_v =
                                data.iter().copied().fold(f32::INFINITY, f32::min);
                            let max_v =
                                data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                            let range = (max_v - min_v).max(1e-6);
                            let n = data.len();
                            let mut pts: Vec<egui::Pos2> = (0..n)
                                .map(|j| {
                                    let x = rect.min.x
                                        + (j as f32 / (n - 1).max(1) as f32)
                                            * rect.width();
                                    let t = (data[j] - min_v) / range;
                                    let y = rect.max.y - t * rect.height();
                                    egui::Pos2::new(x, y)
                                })
                                .collect();
                            if pts.len() == 1 {
                                pts.push(egui::Pos2::new(pts[0].x + 1.0, pts[0].y));
                            }
                            if pts.len() >= 2 {
                                ui.painter().add(egui::Shape::line(
                                    pts,
                                    egui::Stroke::new(
                                        1.5,
                                        egui::Color32::from_rgb(0, 200, 200),
                                    ),
                                ));
                            }
                        };

                    let draw_row_trace =
                        |ui: &mut egui::Ui, data: &[f32], w: f32, h: f32| {
                            let (rect, _) =
                                ui.allocate_exact_size(egui::vec2(w, h), egui::Sense::hover());
                            let min_v =
                                data.iter().copied().fold(f32::INFINITY, f32::min);
                            let max_v =
                                data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                            let range = (max_v - min_v).max(1e-6);
                            let n = data.len();
                            let mut pts: Vec<egui::Pos2> = (0..n)
                                .map(|i| {
                                    let y = rect.min.y
                                        + (i as f32 / (n - 1).max(1) as f32)
                                            * rect.height();
                                    let t = (data[i] - min_v) / range;
                                    let x = rect.min.x + t * rect.width();
                                    egui::Pos2::new(x, y)
                                })
                                .collect();
                            if pts.len() == 1 {
                                pts.push(egui::Pos2::new(pts[0].x, pts[0].y + 1.0));
                            }
                            if pts.len() >= 2 {
                                ui.painter().add(egui::Shape::line(
                                    pts,
                                    egui::Stroke::new(
                                        1.5,
                                        egui::Color32::from_rgb(0, 200, 200),
                                    ),
                                ));
                            }
                        };

                    let col_sum = self.crop_col_sum.clone();
                    let row_sum = self.crop_row_sum.clone();
                    ui.vertical(|ui| {
                        if !col_sum.is_empty() {
                            draw_col_trace(ui, &col_sum, crop_size, trace_height);
                        }
                        ui.horizontal(|ui| {
                            let size = egui::vec2(crop_size, crop_size);
                            let response = ui.image((tex.id(), size));
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
                                                        + (px / self.right_w as f64)
                                                            as f32
                                                        * rect.width(),
                                                    rect.min.y
                                                        + (py / self.right_h as f64)
                                                            as f32
                                                        * rect.height(),
                                                )
                                            })
                                            .collect();
                                        if screen_pts.len() >= 2 {
                                            ui.painter().add(egui::Shape::line(
                                                screen_pts,
                                                egui::Stroke::new(
                                                    1.0,
                                                    egui::Color32::WHITE,
                                                ),
                                            ));
                                        }
                                    }
                                }
                            }
                            if !row_sum.is_empty() {
                                draw_row_trace(ui, &row_sum, trace_width, crop_size);
                            }
                        });
                    });
                }
            });
        });
    }
}

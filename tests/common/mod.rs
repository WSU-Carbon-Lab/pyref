//! Local-CI-safe synthetic beamtime generator for integration tests.
//!
//! Produces a minimal flat-layout beamtime root (``<root>/CCD/*.fits``) containing
//! N ingestible BITPIX=16 FITS files. The header schema matches what
//! [`pyref::loader::read_fits_headers_only_row`] actually consumes: a primary HDU
//! with `SIMPLE`/`BITPIX`/`NAXIS=0`/`DATE`/keyed floats and one IMAGE extension
//! HDU with `NAXIS=2`, `NAXIS1`, `NAXIS2`, and `BZERO=32768` followed by
//! big-endian `i16` pixel data. Pixel values encode `scan_index` and
//! `frame_index` so round-trip reads can assert positional fidelity.

#![allow(dead_code)]

use std::error::Error;
use std::fs::{create_dir_all, File};
use std::io::Write;
use std::path::{Path, PathBuf};

const FITS_BLOCK_SIZE: usize = 2880;
const CARD_SIZE: usize = 80;
const BZERO_UNSIGNED_I16: i32 = 32_768;

/// Shape of the synthetic beamtime to generate.
#[derive(Debug, Clone)]
pub struct SyntheticLayout {
    /// Number of scans when `per_scan_frames` is `None`.
    pub scans: usize,
    /// Frames per scan when `per_scan_frames` is `None`.
    pub frames_per_scan: usize,
    /// Pixel width (NAXIS1).
    pub width: u32,
    /// Pixel height (NAXIS2).
    pub height: u32,
    /// Optional uneven scan sizes; overrides `scans` and `frames_per_scan` when set.
    pub per_scan_frames: Option<Vec<usize>>,
}

impl SyntheticLayout {
    /// Helper for uniform layouts.
    pub fn uniform(scans: usize, frames_per_scan: usize, width: u32, height: u32) -> Self {
        Self {
            scans,
            frames_per_scan,
            width,
            height,
            per_scan_frames: None,
        }
    }
}

/// Handle to a generated synthetic beamtime root.
#[derive(Debug, Clone)]
pub struct SyntheticBeamtime {
    /// Path to the beamtime root directory (pass this to `ingest_beamtime`).
    pub root: PathBuf,
}

/// Fixed sample name for every synthetic frame.
pub const SYNTHETIC_SAMPLE_NAME: &str = "synth";

/// Encodes a pixel value from scan/frame/row/column indices so callers can
/// assert round-trips without storing a separate reference image.
pub fn synthetic_pixel_value(scan_idx: usize, frame_idx: usize, row: usize, col: usize) -> i16 {
    let base = (scan_idx as i64) * 37
        + (frame_idx as i64) * 7
        + (row as i64) * 3
        + (col as i64);
    (base.rem_euclid(1_024)) as i16
}

/// Writes the synthetic FITS tree under `tmp_dir/beamtime/CCD/` and returns the
/// beamtime root (the one you'd hand to `ingest_beamtime`).
pub fn build_synthetic_beamtime(
    layout: SyntheticLayout,
    tmp_dir: &Path,
) -> Result<SyntheticBeamtime, Box<dyn Error>> {
    if layout.width == 0 || layout.height == 0 {
        return Err("synthetic layout width/height must be >0".into());
    }

    let beamtime_root = tmp_dir.join("beamtime");
    let ccd_dir = beamtime_root.join("CCD");
    create_dir_all(&ccd_dir)?;

    let per_scan: Vec<usize> = match &layout.per_scan_frames {
        Some(v) => v.clone(),
        None => vec![layout.frames_per_scan; layout.scans],
    };

    for (scan_idx, frame_count) in per_scan.iter().enumerate() {
        let scan_number = (scan_idx + 1) as u32;
        for frame_idx in 0..*frame_count {
            let frame_number = (frame_idx + 1) as u32;
            let stem = format!(
                "{sample}-{scan:05}-{frame:05}",
                sample = SYNTHETIC_SAMPLE_NAME,
                scan = scan_number,
                frame = frame_number,
            );
            let path = ccd_dir.join(format!("{stem}.fits"));
            write_synthetic_fits(&path, &layout, scan_idx, frame_idx)?;
        }
    }

    Ok(SyntheticBeamtime {
        root: beamtime_root,
    })
}

fn write_synthetic_fits(
    path: &Path,
    layout: &SyntheticLayout,
    scan_idx: usize,
    frame_idx: usize,
) -> Result<(), Box<dyn Error>> {
    let mut file = File::create(path)?;

    let primary = build_primary_header_block(scan_idx, frame_idx);
    file.write_all(&primary)?;

    let image_header = build_image_header_block(layout.width, layout.height);
    file.write_all(&image_header)?;

    let pixel_block = build_pixel_data_block(layout.width, layout.height, scan_idx, frame_idx);
    file.write_all(&pixel_block)?;

    file.sync_all()?;
    Ok(())
}

fn build_primary_header_block(scan_idx: usize, frame_idx: usize) -> Vec<u8> {
    let cards: Vec<String> = vec![
        card_fixed_int("SIMPLE", 1),
        card_fixed_int("BITPIX", 16),
        card_fixed_int("NAXIS", 0),
        card_quoted_string("DATE", "2024-02-02T00:00:00"),
        card_float("Beamline Energy", 250.0 + scan_idx as f64),
        card_float("Sample Theta", 1.0 + (frame_idx as f64) * 0.01),
        card_float("CCD Theta", 2.0),
        card_float("EPU Polarization", 1.0),
        card_float("Higher Order Suppressor", 0.0),
        end_card(),
    ];
    pad_cards_to_block(cards)
}

fn build_image_header_block(width: u32, height: u32) -> Vec<u8> {
    let cards: Vec<String> = vec![
        card_quoted_string("XTENSION", "IMAGE   "),
        card_fixed_int("BITPIX", 16),
        card_fixed_int("NAXIS", 2),
        card_fixed_int("NAXIS1", width as i64),
        card_fixed_int("NAXIS2", height as i64),
        card_fixed_int("BZERO", BZERO_UNSIGNED_I16 as i64),
        end_card(),
    ];
    pad_cards_to_block(cards)
}

fn build_pixel_data_block(width: u32, height: u32, scan_idx: usize, frame_idx: usize) -> Vec<u8> {
    let nbytes = (width as usize) * (height as usize) * 2;
    let mut buf = Vec::with_capacity(nbytes);
    for row in 0..(height as usize) {
        for col in 0..(width as usize) {
            let v = synthetic_pixel_value(scan_idx, frame_idx, row, col);
            buf.extend_from_slice(&v.to_be_bytes());
        }
    }
    pad_to_block_boundary(&mut buf);
    buf
}

fn card_fixed_int(keyword: &str, value: i64) -> String {
    if keyword.len() <= 8 {
        format!("{key:<8}= {val:>20}", key = keyword, val = value)
    } else {
        format!("{key}= {val}", key = keyword, val = value)
    }
}

fn card_float(keyword: &str, value: f64) -> String {
    let formatted = format!("{value:+.16E}");
    if keyword.len() <= 8 {
        format!("{key:<8}= {val:>20}", key = keyword, val = formatted)
    } else {
        format!("{key}= {val}", key = keyword, val = formatted)
    }
}

fn card_quoted_string(keyword: &str, value: &str) -> String {
    let padded_value = if value.len() < 8 {
        format!("{value:<8}")
    } else {
        value.to_string()
    };
    if keyword.len() <= 8 {
        format!("{key:<8}= '{val}'", key = keyword, val = padded_value)
    } else {
        format!("{key}= '{val}'", key = keyword, val = padded_value)
    }
}

fn end_card() -> String {
    "END".to_string()
}

fn pad_cards_to_block(cards: Vec<String>) -> Vec<u8> {
    let mut block: Vec<u8> = Vec::with_capacity(FITS_BLOCK_SIZE);
    for card in cards {
        let mut s = card;
        assert!(
            s.len() <= CARD_SIZE,
            "FITS card exceeds 80 bytes: {} ({} bytes)",
            s,
            s.len()
        );
        while s.len() < CARD_SIZE {
            s.push(' ');
        }
        block.extend_from_slice(s.as_bytes());
    }
    pad_to_block_boundary(&mut block);
    block
}

fn pad_to_block_boundary(buf: &mut Vec<u8>) {
    let rem = buf.len() % FITS_BLOCK_SIZE;
    if rem == 0 {
        if buf.is_empty() {
            buf.resize(FITS_BLOCK_SIZE, b' ');
        }
        return;
    }
    let pad = FITS_BLOCK_SIZE - rem;
    buf.resize(buf.len() + pad, b' ');
}

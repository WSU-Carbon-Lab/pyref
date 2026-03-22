use ndarray::Array2;
use std::sync::OnceLock;

const LUT_LEN: usize = 256;

static RAINBOW_LUT: OnceLock<[[u8; 4]; LUT_LEN]> = OnceLock::new();

fn hsv_to_rgb(h: f64, s: f64, v: f64) -> (u8, u8, u8) {
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;
    let (r, g, b) = if h < 60.0 {
        (c, x, 0.0)
    } else if h < 120.0 {
        (x, c, 0.0)
    } else if h < 180.0 {
        (0.0, c, x)
    } else if h < 240.0 {
        (0.0, x, c)
    } else if h < 300.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };
    (
        ((r + m) * 255.0).clamp(0.0, 255.0) as u8,
        ((g + m) * 255.0).clamp(0.0, 255.0) as u8,
        ((b + m) * 255.0).clamp(0.0, 255.0) as u8,
    )
}

fn build_rainbow_lut() -> [[u8; 4]; LUT_LEN] {
    let mut lut = [[0u8; 4]; LUT_LEN];
    for (i, entry) in lut.iter_mut().enumerate() {
        let h = (i as f64 / LUT_LEN as f64) * 360.0;
        let (r, g, b) = hsv_to_rgb(h, 1.0, 1.0);
        *entry = [r, g, b, 255];
    }
    lut
}

fn rainbow_lut() -> &'static [[u8; 4]; LUT_LEN] {
    RAINBOW_LUT.get_or_init(build_rainbow_lut)
}

pub fn scalar_to_rgba(
    data: &[f64],
    min_max: Option<(f64, f64)>,
    reversed: bool,
) -> Vec<u8> {
    let (min, max) = match min_max {
        Some((lo, hi)) => (lo, hi),
        None => {
            let min = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            (min, max)
        }
    };
    let span = (max - min).max(1e-9);
    let scale = (LUT_LEN - 1) as f64 / span;
    let lut = rainbow_lut();
    let mut out = Vec::with_capacity(data.len() * 4);
    for &v in data {
        let mut idx = ((v - min) * scale).clamp(0.0, (LUT_LEN - 1) as f64) as usize;
        if reversed {
            idx = (LUT_LEN - 1) - idx;
        }
        let rgba = lut[idx];
        out.extend_from_slice(&rgba);
    }
    out
}

pub fn scalar_to_rgba_rainbow(
    data: &[f64],
    min_max: Option<(f64, f64)>,
) -> Vec<u8> {
    scalar_to_rgba(data, min_max, false)
}

pub fn array2_to_rgba_rainbow(
    data: &Array2<f64>,
    min_max: Option<(f64, f64)>,
) -> Option<Vec<u8>> {
    let slice = data.as_slice()?;
    Some(scalar_to_rgba_rainbow(slice, min_max))
}

pub fn array2_i64_to_rgba(
    data: &Array2<i64>,
    min_max: Option<(i64, i64)>,
    reversed: bool,
) -> Option<Vec<u8>> {
    let slice = data.as_slice()?;
    let (min, max) = match min_max {
        Some((lo, hi)) => (lo as f64, hi as f64),
        None => {
            let min = *slice.iter().min()? as f64;
            let max = *slice.iter().max()? as f64;
            (min, max)
        }
    };
    let f64_data: Vec<f64> = slice.iter().map(|&x| x as f64).collect();
    Some(scalar_to_rgba(&f64_data, Some((min, max)), reversed))
}

pub fn array2_i64_to_rgba_rainbow(
    data: &Array2<i64>,
    min_max: Option<(i64, i64)>,
) -> Option<Vec<u8>> {
    array2_i64_to_rgba(data, min_max, false)
}

pub fn array2_f32_to_rgba(
    data: &Array2<f32>,
    min_max: Option<(f32, f32)>,
    reversed: bool,
) -> Option<Vec<u8>> {
    let slice = data.as_slice()?;
    let (min, max) = match min_max {
        Some((lo, hi)) => (lo as f64, hi as f64),
        None => {
            let min = slice.iter().fold(f64::INFINITY, |a, &b| a.min(b as f64));
            let max = slice.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b as f64));
            (min, max)
        }
    };
    let f64_data: Vec<f64> = slice.iter().map(|&x| x as f64).collect();
    Some(scalar_to_rgba(&f64_data, Some((min, max)), reversed))
}

pub fn array2_f32_to_rgba_rainbow(
    data: &Array2<f32>,
    min_max: Option<(f32, f32)>,
) -> Option<Vec<u8>> {
    array2_f32_to_rgba(data, min_max, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_input_gives_constant_rgb() {
        let data = [1.0; 10];
        let out = scalar_to_rgba_rainbow(&data, Some((0.0, 2.0)));
        assert_eq!(out.len(), 40);
        let r = out[0];
        let g = out[1];
        let b = out[2];
        for chunk in out.chunks_exact(4) {
            assert_eq!(chunk[0], r);
            assert_eq!(chunk[1], g);
            assert_eq!(chunk[2], b);
            assert_eq!(chunk[3], 255);
        }
    }

    #[test]
    fn min_max_span_gives_full_lut_range() {
        let data: Vec<f64> = (0..LUT_LEN).map(|i| i as f64).collect();
        let out = scalar_to_rgba_rainbow(&data, Some((0.0, (LUT_LEN - 1) as f64)));
        assert_eq!(out.len(), LUT_LEN * 4);
        let first = [out[0], out[1], out[2]];
        let last = [
            out[out.len() - 4],
            out[out.len() - 3],
            out[out.len() - 2],
        ];
        assert_ne!(first, last);
    }

    #[test]
    fn reversed_inverts_lut_index() {
        let data = vec![0.0f64, 255.0f64];
        let fwd = scalar_to_rgba(&data, Some((0.0, 255.0)), false);
        let rev = scalar_to_rgba(&data, Some((0.0, 255.0)), true);
        let fwd_low = [fwd[0], fwd[1], fwd[2]];
        let fwd_high = [fwd[4], fwd[5], fwd[6]];
        let rev_low = [rev[0], rev[1], rev[2]];
        let rev_high = [rev[4], rev[5], rev[6]];
        assert_eq!(fwd_low, rev_high);
        assert_eq!(fwd_high, rev_low);
    }
}

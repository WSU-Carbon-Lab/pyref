#![cfg(feature = "catalog")]

use pyref::catalog::FileRow;
use super::scan_type::ReflectivityScanType;

#[derive(Debug, Clone)]
pub struct BeamspotLinearFit {
    pub row_slope: f64,
    pub row_intercept: f64,
    pub col_slope: f64,
    pub col_intercept: f64,
    pub row_residual_std: f64,
    pub col_residual_std: f64,
}

pub fn domain_for_row(row: &FileRow, scan_type: ReflectivityScanType) -> Option<f64> {
    domain_value(row, scan_type)
}

fn domain_value(row: &FileRow, scan_type: ReflectivityScanType) -> Option<f64> {
    match scan_type {
        ReflectivityScanType::FixedEnergy => row.sample_theta,
        ReflectivityScanType::FixedAngle => row.beamline_energy,
        ReflectivityScanType::SinglePoint => None,
    }
}

fn least_squares_linear(x: &[f64], y: &[f64]) -> Option<(f64, f64)> {
    let n = x.len() as f64;
    if n < 2.0 || x.len() != y.len() {
        return None;
    }
    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_xx: f64 = x.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
    let sum_xy: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-12 {
        return None;
    }
    let slope = (n * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y - slope * sum_x) / n;
    Some((slope, intercept))
}

fn residual_std(x: &[f64], y: &[f64], slope: f64, intercept: f64) -> f64 {
    if x.len() < 2 || x.len() != y.len() {
        return 0.0;
    }
    let n = x.len() as f64;
    let var: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| {
            let pred = slope * xi + intercept;
            let r = yi - pred;
            r * r
        })
        .sum();
    ((var / (n - 1.0)).max(0.0)).sqrt()
}

pub fn fit_beamspot_linear(
    rows: &[FileRow],
    scan_type: ReflectivityScanType,
) -> Option<BeamspotLinearFit> {
    if matches!(scan_type, ReflectivityScanType::SinglePoint) {
        return None;
    }
    let points: Vec<(f64, f64, f64)> = rows
        .iter()
        .filter_map(|r| {
            let domain = domain_value(r, scan_type)?;
            let row = r.beam_row? as f64;
            let col = r.beam_col? as f64;
            Some((domain, row, col))
        })
        .collect();
    if points.len() < 2 {
        return None;
    }
    let x: Vec<f64> = points.iter().map(|p| p.0).collect();
    let row_vals: Vec<f64> = points.iter().map(|p| p.1).collect();
    let col_vals: Vec<f64> = points.iter().map(|p| p.2).collect();
    let (row_slope, row_intercept) = least_squares_linear(&x, &row_vals)?;
    let (col_slope, col_intercept) = least_squares_linear(&x, &col_vals)?;
    let row_residual_std = residual_std(&x, &row_vals, row_slope, row_intercept);
    let col_residual_std = residual_std(&x, &col_vals, col_slope, col_intercept);
    let row_residual_std = if row_residual_std > 1e-9 {
        row_residual_std
    } else {
        1.0
    };
    let col_residual_std = if col_residual_std > 1e-9 {
        col_residual_std
    } else {
        1.0
    };
    Some(BeamspotLinearFit {
        row_slope,
        row_intercept,
        col_slope,
        col_intercept,
        row_residual_std,
        col_residual_std,
    })
}

pub fn beamspot_status(
    row: Option<i64>,
    col: Option<i64>,
    domain: Option<f64>,
    fit: Option<&BeamspotLinearFit>,
    row_mean: f64,
    row_std: f64,
    col_mean: f64,
    col_std: f64,
) -> (&'static str, u8) {
    let (r, c) = match (row, col) {
        (Some(a), Some(b)) => (a as f64, b as f64),
        _ => return ("-", 3),
    };
    let max_z = if let (Some(f), Some(d)) = (fit, domain) {
        let pred_row = f.row_slope * d + f.row_intercept;
        let pred_col = f.col_slope * d + f.col_intercept;
        let z_row = ((r - pred_row) / f.row_residual_std).abs();
        let z_col = ((c - pred_col) / f.col_residual_std).abs();
        z_row.max(z_col)
    } else {
        let z_row = if row_std > 1e-9 {
            ((r - row_mean) / row_std).abs()
        } else {
            0.0
        };
        let z_col = if col_std > 1e-9 {
            ((c - col_mean) / col_std).abs()
        } else {
            0.0
        };
        z_row.max(z_col)
    };
    let (status, key) = if max_z < 2.0 {
        ("ok", 0)
    } else if max_z < 4.0 {
        ("warning", 1)
    } else {
        ("err", 2)
    };
    (status, key)
}

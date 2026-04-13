//! Beamspot position drift versus acquisition domain for catalog file rows.
//!
//! For [`ReflectivityScanType::FixedEnergy`] (theta scan), the independent domain is
//! [`FileRow::sample_theta`]. For [`ReflectivityScanType::FixedAngle`] (energy scan), the domain is
//! [`FileRow::beamline_energy`]. Linear fits and z-scores follow the project spec for TUI parity.

use crate::catalog::query::FileRow;
use crate::catalog::reflectivity_profile::ReflectivityScanType;

/// Ordinary least-squares fit of beam row and column vs domain, with residual scales for z-scoring.
#[derive(Debug, Clone)]
pub struct BeamspotLinearFit {
    pub row_slope: f64,
    pub row_intercept: f64,
    pub col_slope: f64,
    pub col_intercept: f64,
    pub row_residual_std: f64,
    pub col_residual_std: f64,
}

/// Returns the independent variable for beamspot drift for this row and scan type.
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

/// Fits row and column pixel centers vs domain for rows that have beamspot and domain values.
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

/// Labels beamspot deviation: `"ok"` / `"warning"` / `"err"` vs fit or vs batch mean, plus sort key.
#[allow(clippy::too_many_arguments)]
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

#[cfg(test)]
mod tests {
    use super::*;

    fn row(
        energy: Option<f64>,
        theta: Option<f64>,
        beam_row: Option<i64>,
        beam_col: Option<i64>,
    ) -> FileRow {
        FileRow {
            file_path: String::new(),
            sample_name: String::new(),
            tag: None,
            scan_number: 0,
            frame_number: 0,
            beamline_energy: energy,
            sample_theta: theta,
            epu_polarization: None,
            q: None,
            date_iso: None,
            beam_row,
            beam_col,
            beam_sigma: None,
            scan_point_uid: None,
        }
    }

    #[test]
    fn fixed_energy_fit_invariant_to_energy_jitter() {
        let rows = vec![
            row(Some(100.0), Some(0.0), Some(10), Some(20)),
            row(Some(500.0), Some(1.0), Some(12), Some(22)),
            row(Some(200.0), Some(2.0), Some(14), Some(24)),
        ];
        let rows_alt = vec![
            row(Some(999.0), Some(0.0), Some(10), Some(20)),
            row(Some(1.0), Some(1.0), Some(12), Some(22)),
            row(Some(50.0), Some(2.0), Some(14), Some(24)),
        ];
        let a = fit_beamspot_linear(&rows, ReflectivityScanType::FixedEnergy).unwrap();
        let b = fit_beamspot_linear(&rows_alt, ReflectivityScanType::FixedEnergy).unwrap();
        assert!((a.row_slope - b.row_slope).abs() < 1e-9);
        assert!((a.row_intercept - b.row_intercept).abs() < 1e-9);
        assert!((a.col_slope - b.col_slope).abs() < 1e-9);
        assert!((a.col_intercept - b.col_intercept).abs() < 1e-9);
    }

    #[test]
    fn fixed_angle_fit_invariant_to_theta_jitter() {
        let rows = vec![
            row(Some(280.0), Some(0.0), Some(5), Some(8)),
            row(Some(281.0), Some(99.0), Some(7), Some(10)),
            row(Some(282.0), Some(-3.0), Some(9), Some(12)),
        ];
        let rows_alt = vec![
            row(Some(280.0), Some(10.0), Some(5), Some(8)),
            row(Some(281.0), Some(10.0), Some(7), Some(10)),
            row(Some(282.0), Some(10.0), Some(9), Some(12)),
        ];
        let a = fit_beamspot_linear(&rows, ReflectivityScanType::FixedAngle).unwrap();
        let b = fit_beamspot_linear(&rows_alt, ReflectivityScanType::FixedAngle).unwrap();
        assert!((a.row_slope - b.row_slope).abs() < 1e-9);
        assert!((a.row_intercept - b.row_intercept).abs() < 1e-9);
    }
}

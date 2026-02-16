#![cfg(feature = "catalog")]

use std::collections::HashSet;

const E_TOL_EV: f64 = 0.5;
const THETA_TOL_DEG: f64 = 0.1;
const IZERO_FRACTION_THRESHOLD: f64 = 0.25;
const IZERO_MIN_POINTS: usize = 2;
const E_ROUND_EV: f64 = 0.1;
const THETA_ROUND_DEG: f64 = 0.01;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReflectivityScanType {
    FixedEnergy,
    FixedAngle,
    SinglePoint,
}

fn round_energy(e: f64) -> i64 {
    (e / E_ROUND_EV).round() as i64
}

fn round_theta(t: f64) -> i64 {
    (t / THETA_ROUND_DEG).round() as i64
}

pub fn classify_scan_type(
    energy_theta_pairs: &[(Option<f64>, Option<f64>)],
) -> (
    ReflectivityScanType,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
) {
    let energies: Vec<f64> = energy_theta_pairs.iter().filter_map(|(e, _)| *e).collect();
    let thetas: Vec<f64> = energy_theta_pairs.iter().filter_map(|(_, t)| *t).collect();
    if energies.is_empty() || thetas.is_empty() {
        let e_min = energies.iter().cloned().fold(f64::INFINITY, f64::min);
        let e_max = energies.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let t_min = thetas.iter().cloned().fold(f64::INFINITY, f64::min);
        let t_max = thetas.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        return (
            ReflectivityScanType::SinglePoint,
            if energies.is_empty() {
                None
            } else {
                Some(if e_min <= e_max { e_min } else { e_max })
            },
            if energies.is_empty() {
                None
            } else {
                Some(if e_max >= e_min { e_max } else { e_min })
            },
            if thetas.is_empty() {
                None
            } else {
                Some(if t_min <= t_max { t_min } else { t_max })
            },
            if thetas.is_empty() {
                None
            } else {
                Some(if t_max >= t_min { t_max } else { t_min })
            },
        );
    }
    let e_min = energies.iter().cloned().fold(f64::INFINITY, f64::min);
    let e_max = energies.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let t_min = thetas.iter().cloned().fold(f64::INFINITY, f64::min);
    let t_max = thetas.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range_e = e_max - e_min;
    let range_theta = t_max - t_min;
    let n_e: usize = energies
        .iter()
        .map(|&e| round_energy(e))
        .collect::<HashSet<_>>()
        .len();
    let n_theta: usize = thetas
        .iter()
        .map(|&t| round_theta(t))
        .collect::<HashSet<_>>()
        .len();
    let n_total = energy_theta_pairs
        .iter()
        .filter(|(e, t)| e.is_some() && t.is_some())
        .count();
    let n_theta_near_zero = energy_theta_pairs
        .iter()
        .filter(|(_, t)| t.map(|v| v <= THETA_TOL_DEG).unwrap_or(false))
        .count();
    let energy_varies = range_e > E_TOL_EV;
    let theta_varies = range_theta > THETA_TOL_DEG;
    let scan_type = if energy_varies && !theta_varies {
        ReflectivityScanType::FixedAngle
    } else if theta_varies && !energy_varies {
        ReflectivityScanType::FixedEnergy
    } else if energy_varies && theta_varies {
        if n_total > 0
            && n_theta_near_zero >= IZERO_MIN_POINTS
            && (n_theta_near_zero as f64 / n_total as f64) >= IZERO_FRACTION_THRESHOLD
        {
            ReflectivityScanType::FixedEnergy
        } else if n_e >= n_theta {
            ReflectivityScanType::FixedAngle
        } else {
            ReflectivityScanType::FixedEnergy
        }
    } else {
        ReflectivityScanType::SinglePoint
    };
    (
        scan_type,
        Some(e_min),
        Some(e_max),
        Some(t_min),
        Some(t_max),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pair(energy: Option<f64>, theta: Option<f64>) -> (Option<f64>, Option<f64>) {
        (energy, theta)
    }

    #[test]
    fn single_energy_many_thetas_fixed_energy() {
        let rows: Vec<_> = (0..10)
            .map(|i| pair(Some(284.0), Some(0.5 * i as f64)))
            .collect();
        let (st, e_min, e_max, t_min, t_max) = classify_scan_type(&rows);
        assert_eq!(st, ReflectivityScanType::FixedEnergy);
        assert!(e_min.unwrap() <= 284.0 && e_max.unwrap() >= 284.0);
        assert!(t_min.unwrap() <= 0.5 && t_max.unwrap() >= 4.5);
    }

    #[test]
    fn many_energies_single_theta_fixed_angle() {
        let rows: Vec<_> = (0..10)
            .map(|i| pair(Some(250.0 + i as f64), Some(10.0)))
            .collect();
        let (st, _, _, _, _) = classify_scan_type(&rows);
        assert_eq!(st, ReflectivityScanType::FixedAngle);
    }

    #[test]
    fn many_energies_few_thetas_no_izero_fixed_angle() {
        let mut rows = Vec::new();
        for e in [250.0, 251.0, 252.0, 253.0] {
            for &t in &[1.0, 2.0, 4.0, 8.0] {
                rows.push(pair(Some(e), Some(t)));
            }
        }
        let (st, _, _, _, _) = classify_scan_type(&rows);
        assert_eq!(st, ReflectivityScanType::FixedAngle);
    }

    #[test]
    fn one_energy_many_thetas_izero_fixed_energy() {
        let mut rows = Vec::new();
        for _ in 0..20 {
            rows.push(pair(Some(284.0), Some(0.0)));
        }
        for i in 1..5 {
            rows.push(pair(Some(284.0), Some(i as f64 * 2.0)));
        }
        let (st, _, _, _, _) = classify_scan_type(&rows);
        assert_eq!(st, ReflectivityScanType::FixedEnergy);
    }

    #[test]
    fn empty_single_point() {
        let (st, e_min, e_max, t_min, t_max) = classify_scan_type(&[]);
        assert_eq!(st, ReflectivityScanType::SinglePoint);
        assert!(e_min.is_none());
        assert!(e_max.is_none());
        assert!(t_min.is_none());
        assert!(t_max.is_none());
    }
}

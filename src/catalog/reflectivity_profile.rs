//! Reflectivity scan profile classification from beamline energy and sample theta samples.
//!
//! Classifies a sequence of (energy, theta) observations into fixed-energy (theta scan),
//! fixed-angle (energy scan), or degenerate single-point behavior using tolerances and
//! heuristics shared with the TUI. [`segment_reflectivity_profiles`] splits ordered points when
//! an instrument scan contains multiple profiles (e.g. I0 plateau then theta ramp, or successive
//! energy blocks). Callers use the scan type to choose the independent domain for beamspot drift
//! checks (theta vs energy per project spec).

use std::collections::HashSet;

const E_TOL_EV: f64 = 0.5;
const THETA_TOL_DEG: f64 = 0.1;
const IZERO_FRACTION_THRESHOLD: f64 = 0.25;
const IZERO_MIN_POINTS: usize = 2;
const E_ROUND_EV: f64 = 0.1;
const THETA_ROUND_DEG: f64 = 0.01;

/// Kind of reflectivity acquisition implied by energy and theta variation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReflectivityScanType {
    /// Approximately fixed beamline energy; theta sweeps (theta scan).
    FixedEnergy,
    /// Approximately fixed sample theta; energy sweeps (energy scan).
    FixedAngle,
    /// Insufficient variation or missing data to classify as a sweep.
    SinglePoint,
}

fn round_energy(e: f64) -> i64 {
    (e / E_ROUND_EV).round() as i64
}

fn round_theta(t: f64) -> i64 {
    (t / THETA_ROUND_DEG).round() as i64
}

/// Classify reflectivity scan type from paired beamline energy and sample theta values.
///
/// Each pair is one frame or scan point. Missing components are ignored for range and
/// distinct-count statistics; if either axis has no finite samples, the result is
/// [`ReflectivityScanType::SinglePoint`] with `None` extrema on empty axes.
///
/// Returns `(scan_type, e_min, e_max, t_min, t_max)` using finite samples only; extrema
/// are `None` when the corresponding axis has no data.
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

/// One contiguous sub-sequence of frames within an instrument scan, ordered by acquisition.
///
/// `start` is inclusive; `end` is exclusive, indexing into the same slice passed to
/// [`segment_reflectivity_profiles`]. `scan_type` is the classification of that slice alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProfileSegment {
    pub start: usize,
    pub end: usize,
    pub scan_type: ReflectivityScanType,
}

fn energy_step_boundaries(pairs: &[(Option<f64>, Option<f64>)]) -> Vec<usize> {
    let n = pairs.len();
    let mut out = Vec::new();
    for i in 1..n {
        if let (Some(e0), Some(e1)) = (pairs[i - 1].0, pairs[i].0) {
            if (e1 - e0).abs() > E_TOL_EV {
                out.push(i);
            }
        }
    }
    out
}

/// First index after an initial alignment run of near-zero sample theta when the remainder is a theta scan.
///
/// Requires at least `IZERO_MIN_POINTS` finite samples with `|theta|` below the alignment tolerance
/// before the first finite theta above that tolerance. Skips leading rows with missing theta without
/// breaking the prefix. If the suffix does not classify as [`ReflectivityScanType::FixedEnergy`],
/// returns `None` so energy-only or ambiguous scans are not split on noise.
fn izero_then_theta_ramp_boundary(pairs: &[(Option<f64>, Option<f64>)]) -> Option<usize> {
    let mut i = 0usize;
    let mut prefix_near_zero = 0usize;
    while i < pairs.len() {
        match pairs[i].1 {
            Some(tv) if tv.abs() <= THETA_TOL_DEG => {
                prefix_near_zero += 1;
                i += 1;
            }
            None => {
                i += 1;
            }
            Some(_) => break,
        }
    }
    if prefix_near_zero < IZERO_MIN_POINTS || i >= pairs.len() {
        return None;
    }
    let (st, _, _, _, _) = classify_scan_type(&pairs[i..]);
    if matches!(st, ReflectivityScanType::FixedEnergy) {
        Some(i)
    } else {
        None
    }
}

/// Partition ordered `(energy, theta)` points into contiguous profiles for reflectivity grouping.
///
/// Boundaries are inserted when:
///
/// 1. **Energy step:** consecutive finite energies differ by more than the catalog energy
///    tolerance (same scale as classification).
/// 2. **I0 then theta ramp:** an initial stretch of at least `IZERO_MIN_POINTS` samples with
///    near-zero theta (classification alignment tolerance) followed by a suffix classified as a
///    fixed-energy theta scan.
///
/// Each segment is classified with [`classify_scan_type`] on its slice alone. Empty input yields an
/// empty vector.
pub fn segment_reflectivity_profiles(
    energy_theta_pairs: &[(Option<f64>, Option<f64>)],
) -> Vec<ProfileSegment> {
    let n = energy_theta_pairs.len();
    if n == 0 {
        return Vec::new();
    }
    let mut starts: Vec<usize> = vec![0];
    starts.extend(energy_step_boundaries(energy_theta_pairs));
    if let Some(k) = izero_then_theta_ramp_boundary(energy_theta_pairs) {
        starts.push(k);
    }
    starts.sort_unstable();
    starts.dedup();
    starts.retain(|&s| s < n);
    let mut segments = Vec::new();
    for (j, &start) in starts.iter().enumerate() {
        let end = if j + 1 < starts.len() {
            starts[j + 1]
        } else {
            n
        };
        if start >= end {
            continue;
        }
        let slice = &energy_theta_pairs[start..end];
        let (scan_type, _, _, _, _) = classify_scan_type(slice);
        segments.push(ProfileSegment {
            start,
            end,
            scan_type,
        });
    }
    segments
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

    #[test]
    fn segment_izero_then_theta_ramp_two_profiles() {
        let mut rows = Vec::new();
        for _ in 0..4 {
            rows.push(pair(Some(284.0), Some(0.0)));
        }
        for i in 1..6 {
            rows.push(pair(Some(284.0), Some(i as f64 * 0.5)));
        }
        let segs = segment_reflectivity_profiles(&rows);
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].start, 0);
        assert_eq!(segs[0].end, 4);
        assert_eq!(segs[0].scan_type, ReflectivityScanType::SinglePoint);
        assert_eq!(segs[1].start, 4);
        assert_eq!(segs[1].end, rows.len());
        assert_eq!(segs[1].scan_type, ReflectivityScanType::FixedEnergy);
    }

    #[test]
    fn segment_two_energy_blocks_two_fixed_angle() {
        let mut rows = Vec::new();
        for e in [280.0, 280.3, 280.6] {
            rows.push(pair(Some(e), Some(10.0)));
        }
        for e in [295.0, 295.3, 295.6] {
            rows.push(pair(Some(e), Some(10.0)));
        }
        let segs = segment_reflectivity_profiles(&rows);
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].scan_type, ReflectivityScanType::FixedAngle);
        assert_eq!(segs[1].scan_type, ReflectivityScanType::FixedAngle);
        assert_eq!(segs[0].start, 0);
        assert_eq!(segs[0].end, 3);
        assert_eq!(segs[1].start, 3);
        assert_eq!(segs[1].end, 6);
    }

    #[test]
    fn segment_single_block_matches_whole_classify() {
        let rows: Vec<_> = (0..8)
            .map(|i| pair(Some(284.0), Some(0.3 * i as f64)))
            .collect();
        let segs = segment_reflectivity_profiles(&rows);
        assert_eq!(segs.len(), 1);
        let (whole, _, _, _, _) = classify_scan_type(&rows);
        assert_eq!(segs[0].scan_type, whole);
        assert_eq!(segs[0].start, 0);
        assert_eq!(segs[0].end, rows.len());
    }
}

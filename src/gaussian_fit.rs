use ndarray::Array2;

/// 2D Gaussian fit result: center (row, col), sigmas, amplitude, and constant baseline.
#[derive(Debug, Clone, Copy)]
pub struct Gaussian2DFit {
    pub center_row: f64,
    pub center_col: f64,
    pub sigma_row: f64,
    pub sigma_col: f64,
    pub amplitude: f64,
    pub baseline: f64,
}

fn gaussian_2d(r: f64, c: f64, mu_r: f64, mu_c: f64, sr: f64, sc: f64, a: f64, b: f64) -> f64 {
    if sr <= 0.0 || sc <= 0.0 {
        return b;
    }
    let dr = (r - mu_r) / sr;
    let dc = (c - mu_c) / sc;
    a * (-0.5 * (dr * dr + dc * dc)).exp() + b
}

const MAX_ITER: usize = 50;
const LAMBDA_INIT: f64 = 1e-3;
const LAMBDA_UP: f64 = 10.0;
const LAMBDA_DOWN: f64 = 0.1;
const TOL: f64 = 1e-8;

pub fn fit_2d_gaussian(
    image: &Array2<f64>,
    roi: Option<(usize, usize, usize, usize)>,
) -> Option<Gaussian2DFit> {
    let (r0, r1, c0, c1) = match roi {
        Some((r0, r1, c0, c1)) => {
            let r1 = r1.min(image.nrows());
            let c1 = c1.min(image.ncols());
            if r0 >= r1 || c0 >= c1 {
                return None;
            }
            (r0, r1, c0, c1)
        }
        None => (0, image.nrows(), 0, image.ncols()),
    };
    let nr = r1 - r0;
    let nc = c1 - c0;
    if nr < 3 || nc < 3 {
        return None;
    }
    let mut max_val = f64::NEG_INFINITY;
    let mut max_r = r0;
    let mut max_c = c0;
    for r in r0..r1 {
        for c in c0..c1 {
            let v = image[[r, c]];
            if v > max_val {
                max_val = v;
                max_r = r;
                max_c = c;
            }
        }
    }
    let min_val = (r0..r1)
        .flat_map(|r| (c0..c1).map(move |c| image[[r, c]]))
        .fold(f64::INFINITY, f64::min);
    let mut mu_r = max_r as f64;
    let mut mu_c = max_c as f64;
    let mut sr = 2.0;
    let mut sc = 2.0;
    let mut a = (max_val - min_val).max(1.0);
    let mut b = min_val;
    let mut lambda = LAMBDA_INIT;
    let mut prev_ss = f64::INFINITY;
    for _ in 0..MAX_ITER {
        let mut jtj = [[0.0f64; 6]; 6];
        let mut jtr = [0.0f64; 6];
        let mut ss = 0.0;
        for ri in r0..r1 {
            for ci in c0..c1 {
                let r = ri as f64;
                let c = ci as f64;
                let y = image[[ri, ci]];
                let f = gaussian_2d(r, c, mu_r, mu_c, sr, sc, a, b);
                let res = y - f;
                ss += res * res;
                let dr = (r - mu_r) / sr;
                let dc = (c - mu_c) / sc;
                let exp_part = (-0.5 * (dr * dr + dc * dc)).exp();
                let d_mu_r = a * exp_part * dr / sr;
                let d_mu_c = a * exp_part * dc / sc;
                let d_sr = a * exp_part * dr * dr / sr;
                let d_sc = a * exp_part * dc * dc / sc;
                let d_a = exp_part;
                let d_b = 1.0;
                let j = [d_mu_r, d_mu_c, d_sr, d_sc, d_a, d_b];
                for i in 0..6 {
                    jtr[i] += j[i] * res;
                    for jj in 0..6 {
                        jtj[i][jj] += j[i] * j[jj];
                    }
                }
            }
        }
        for i in 0..6 {
            jtj[i][i] += lambda;
        }
        let dp = solve_6x6(&jtj, &jtr)?;
        let new_mu_r = mu_r + dp[0];
        let new_mu_c = mu_c + dp[1];
        let new_sr = (sr + dp[2]).max(0.3);
        let new_sc = (sc + dp[3]).max(0.3);
        let new_a = (a + dp[4]).max(0.1);
        let new_b = b + dp[5];
        if ss < prev_ss {
            lambda *= LAMBDA_DOWN;
            prev_ss = ss;
            mu_r = new_mu_r;
            mu_c = new_mu_c;
            sr = new_sr;
            sc = new_sc;
            a = new_a;
            b = new_b;
            if dp.iter().map(|x| x.abs()).fold(0.0f64, f64::max) < TOL {
                break;
            }
        } else {
            lambda *= LAMBDA_UP;
            if lambda > 1e10 {
                break;
            }
        }
    }
    Some(Gaussian2DFit {
        center_row: mu_r,
        center_col: mu_c,
        sigma_row: sr,
        sigma_col: sc,
        amplitude: a,
        baseline: b,
    })
}

fn solve_6x6(a: &[[f64; 6]; 6], b: &[f64; 6]) -> Option<[f64; 6]> {
    let mut m = [[0.0f64; 7]; 6];
    for i in 0..6 {
        for j in 0..6 {
            m[i][j] = a[i][j];
        }
        m[i][6] = b[i];
    }
    for col in 0..6 {
        let mut pivot = col;
        for row in (col + 1)..6 {
            if m[row][col].abs() > m[pivot][col].abs() {
                pivot = row;
            }
        }
        m.swap(col, pivot);
        let div = m[col][col];
        if div.abs() < 1e-15 {
            return None;
        }
        for j in 0..7 {
            m[col][j] /= div;
        }
        for i in 0..6 {
            if i != col {
                let factor = m[i][col];
                for j in 0..7 {
                    m[i][j] -= factor * m[col][j];
                }
            }
        }
    }
    Some([m[0][6], m[1][6], m[2][6], m[3][6], m[4][6], m[5][6]])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_gaussian(
        nr: usize,
        nc: usize,
        mu_r: f64,
        mu_c: f64,
        sr: f64,
        sc: f64,
    ) -> Array2<f64> {
        let mut arr = Array2::zeros((nr, nc));
        for r in 0..nr {
            for c in 0..nc {
                arr[[r, c]] = gaussian_2d(r as f64, c as f64, mu_r, mu_c, sr, sc, 100.0, 0.0);
            }
        }
        arr
    }

    #[test]
    fn fit_synthetic_full_image() {
        let arr = synthetic_gaussian(32, 32, 15.2, 14.7, 2.5, 3.0);
        let fit = fit_2d_gaussian(&arr, None).unwrap();
        assert!((fit.center_row - 15.2).abs() < 0.2);
        assert!((fit.center_col - 14.7).abs() < 0.2);
        assert!((fit.sigma_row - 2.5).abs() < 0.3);
        assert!((fit.sigma_col - 3.0).abs() < 0.3);
    }

    #[test]
    fn fit_synthetic_with_roi() {
        let arr = synthetic_gaussian(64, 64, 20.0, 30.0, 2.0, 2.5);
        let roi = (10, 35, 22, 42);
        let fit = fit_2d_gaussian(&arr, Some(roi)).unwrap();
        assert!((fit.center_row - 20.0).abs() < 0.3);
        assert!((fit.center_col - 30.0).abs() < 0.3);
    }
}

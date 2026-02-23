use ndarray::Array2;

use crate::io::subtract_background;

pub fn locate_beam_simple(image: &Array2<i64>) -> (usize, usize) {
    let dyn_img = image.clone().into_dyn();
    let subtracted = subtract_background(&dyn_img);
    let view_rows = subtracted.shape()[0];
    let view_cols = subtracted.shape()[1];
    let mut max_val = i64::MIN;
    let mut max_linear = 0usize;
    for (idx, &v) in subtracted.iter().enumerate() {
        if v > max_val {
            max_val = v;
            max_linear = idx;
        }
    }
    let max_row = max_linear / view_cols;
    let max_col = max_linear % view_cols;
    let offset_row = if image.nrows() > view_rows { 5 } else { 0 };
    let offset_col = if image.ncols() > view_cols { 5 } else { 0 };
    let row = if view_rows > 0 {
        (max_row + offset_row).min(image.nrows().saturating_sub(1))
    } else {
        0
    };
    let col = if view_cols > 0 {
        (max_col + offset_col).min(image.ncols().saturating_sub(1))
    } else {
        0
    };
    (row, col)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn locate_beam_simple_known_argmax() {
        let mut arr = Array2::zeros((20, 30));
        arr[[10, 15]] = 1000;
        let (row, col) = locate_beam_simple(&arr);
        assert_eq!(row, 10);
        assert_eq!(col, 15);
    }
}

use ndarray::Array2;
use std::f32;

const I64_CLAMP_MAX: i64 = 16_777_216;
const I64_CLAMP_MIN: i64 = -16_777_216;

pub fn i64_to_f32_array(
    arr: &Array2<i64>,
) -> Array2<f32> {
    let (h, w) = (arr.nrows(), arr.ncols());
    let mut out = Array2::zeros((h, w));
    for (a, b) in arr.iter().zip(out.iter_mut()) {
        let clamped = (*a).clamp(I64_CLAMP_MIN, I64_CLAMP_MAX);
        *b = clamped as f32;
    }
    out
}

pub fn gaussian_blur_f32_copy(
    src: &[f32],
    width: u32,
    height: u32,
    sigma: f64,
) -> Result<Vec<f32>, libblur::BlurError> {
    use libblur::{
        BlurImage, BlurImageMut, EdgeMode, EdgeMode2D, FastBlurChannels, GaussianBlurParams,
        IeeeBinaryConvolutionMode, ThreadingPolicy,
    };
    if sigma <= 0.0 {
        return Err(libblur::BlurError::NegativeOrZeroSigma);
    }
    if (sigma - 0.8).abs() < f64::EPSILON {
        return Err(libblur::BlurError::InvalidArguments);
    }
    let mut dst = src.to_vec();
    let src_img = BlurImage::borrow(src, width, height, FastBlurChannels::Plane);
    let mut dst_img = BlurImageMut::borrow(dst.as_mut_slice(), width, height, FastBlurChannels::Plane);
    let params = GaussianBlurParams::new_from_sigma(sigma);
    let edge = EdgeMode2D::new(EdgeMode::Clamp);
    libblur::gaussian_blur_f32(
        &src_img,
        &mut dst_img,
        params,
        edge,
        ThreadingPolicy::Adaptive,
        IeeeBinaryConvolutionMode::Normal,
    )?;
    Ok(dst)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn i64_to_f32_preserves_shape_and_clamps() {
        let arr = Array2::from_shape_vec((2, 3), vec![0_i64, 1, 2, 3, 4, 5]).unwrap();
        let out = i64_to_f32_array(&arr);
        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(out[[0, 0]], 0.0);
        assert_eq!(out[[1, 2]], 5.0);
        let big = Array2::from_shape_vec((1, 2), vec![I64_CLAMP_MAX + 1, I64_CLAMP_MIN - 1]).unwrap();
        let out_big = i64_to_f32_array(&big);
        assert_eq!(out_big[[0, 0]], I64_CLAMP_MAX as f32);
        assert_eq!(out_big[[0, 1]], I64_CLAMP_MIN as f32);
    }

    #[test]
    fn gaussian_blur_f32_shape_and_smoothing() {
        let w = 16u32;
        let h = 16u32;
        let mut data: Vec<f32> = vec![0.0; (w * h) as usize];
        data[8 * w as usize + 8] = 100.0;
        let blurred = gaussian_blur_f32_copy(&data, w, h, 1.5).unwrap();
        assert_eq!(blurred.len(), (w * h) as usize);
        let peak = blurred[8 * w as usize + 8];
        assert!(peak < 100.0 && peak > 0.0);
        let sum: f32 = blurred.iter().sum();
        assert!((sum - 100.0).abs() < 0.01);
    }
}

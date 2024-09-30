use ndarray::{s, Array, Array2};
use ndarray_ndimage::*;
use polars::prelude::*;
pub mod loader;

pub fn img_to_series(name: &str, array: Array2<u32>) -> Series {
    let mut s = Series::new_empty(name, &DataType::List(Box::new(DataType::UInt32)));
    let flat = array.iter().copied().collect::<Vec<_>>();
    let mut chunked_builder = ListPrimitiveChunkedBuilder::<UInt32Type>::new(
        "",
        array.shape().iter().product::<usize>(),
        array.shape().iter().product::<usize>(),
        DataType::UInt32,
    );
    chunked_builder.append_slice(flat.as_slice());
    let new_series = chunked_builder.finish().into_series();
    let _ = s.extend(&new_series);
    s
}

pub fn get_image(df: &DataFrame, i: &usize, img: &str, shape: usize) -> Array2<u32> {
    let data = match df[img].get(*i).unwrap_or(AnyValue::Null) {
        AnyValue::List(s) => s,
        _ => panic!("Expected list type"),
    };
    to_array(data, shape)
}

fn to_array(data: Series, shape: usize) -> Array2<u32> {
    let dim = (shape, shape);
    let listed = data
        .iter()
        .map(|x| match x {
            AnyValue::UInt32(x) => x,
            _ => panic!("Expected u32 type"),
        })
        .collect::<Vec<_>>();
    Array::from_shape_vec(dim, listed).unwrap_or(Array2::zeros((0, 0)))
}

pub fn max_idx(array: &Array2<f64>) -> Result<(usize, usize), &str> {
    let mut max = 0.0;
    let mut idx = (0, 0);
    for (i, x) in array.iter().enumerate() {
        if x > &max {
            max = *x;
            idx = (i / array.shape()[1], i % array.shape()[1]);
        }
    }
    // if the index is at the border, panic
    if idx.0 == 0 || idx.0 == array.shape()[0] - 1 || idx.1 == 0 || idx.1 == array.shape()[1] - 1 {
        return Err("Max value is at the border, beam intensity less than max value");
    }
    Ok(idx)
}

pub fn detect_beam(
    img: Array2<u32>,
    box_size: usize,
) -> (Array2<u32>, Result<(usize, usize), String>) {
    let imf = img.mapv(|x| x as f64);
    let imf_gf = gaussian_filter(&imf, 5.0, 0, BorderMode::Nearest, 5);
    let (x, y) = match max_idx(&imf_gf) {
        Ok(idx) => idx,
        Err(_) => sobel_detect(&imf).unwrap(),
    };
    let beam = img
        .slice(s![x - box_size..=x + box_size, y - box_size..=y + box_size])
        .to_owned();
    (beam, Ok((x, y)))
}

pub fn sobel_detect(imf: &Array2<f64>) -> Result<(usize, usize), String> {
    let imf_sobel = sobel(&imf, ndarray::Axis(0), BorderMode::Nearest);
    max_idx(&imf_sobel).map_err(|e| e.to_string())
}

fn main() {
    let test_path = "C:/Users/hduva/.projects/pyref/pyref_ccd/test/test.fits";
    let df = loader::read_fits(test_path).unwrap();
    println!("{:?}", df);
}

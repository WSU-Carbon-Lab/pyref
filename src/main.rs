use ndarray::{Array, Array2};
use ndarray_ndimage::*;
use polars::prelude::*;
use rand::Rng;

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

// Poostprocess steps
pub fn beam_spot(array: &Array2<f64>) -> Array2<f64> {
    // apply gaussian filter
    let mode = BorderMode::Nearest;
    let mut blurred = gaussian_filter(array, 5.0, 0, mode, 4);
    let idx = max_idx(&blurred);
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
        panic!("Max index is at the border");
    }
    Ok(idx)
}

fn main() {
    let mut rng = rand::thread_rng();
    let array: Array2<u32> = Array2::from_shape_fn((500, 500), |_| rng.gen());
    let series = img_to_series("image", array);
    let df = DataFrame::new(vec![series]).unwrap();
    println!("{:?}", df);
    let data = get_image(&df, &0, "image", 500);
    println!("{:?}", data);
}

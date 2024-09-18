use ndarray::Array2;
use polars::prelude::*;
use polars_arrow::{array::Array, datatypes::ArrowDataType};
use polars_core::utils::arrow::array::FixedSizeListArray;

fn main() {
    let img = Array2::<i32>::zeros((10, 10));
    let size = img.shape()[1];
    // store each row in a Series with dtype array[u32, 4]
    for i in 0..size {
        let arrow_array =
        let inner = FixedSizeListArray::new(ArrowDataType::Int32, Box::new(converted), None);
    }
    // store the array in a Series with dtype array[u32, 4]
    let s1 = Series::new_empty(
        "tets",
        &DataType::Array(
            Box::new(DataType::Array(Box::new(DataType::Int32), size)),
            size,
        ),
    );
    println!("{:?}", s1);
}

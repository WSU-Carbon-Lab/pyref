// Rust module to load all fits files in a directory into memory and construct a
// polars dataframe from them.

use polars::prelude::*;
use pyo3::{prelude::*, types::PyDict};
extern crate fitrs;
use fitrs::{Fits, FitsData, FitsDataArray};



#[pyfunction]
fn read_fits(f: str) -> PyResult<PyDict> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn pyref(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    Ok(())
}

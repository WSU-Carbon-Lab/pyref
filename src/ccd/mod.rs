use polars_core::frame::DataFrame;
use pyo3::prelude::*;
use pyref_core::ccd::{read_fits, CcdFits};

// Python bindings for the CCD library
#[pyclass]
#[repr(transparent)]
pub struct PyCcdFits {
    pub ccd: CcdFits,
}

impl From<CcdFits> for PyCcdFits {
    fn from(ccd: CcdFits) -> Self {
        PyCcdFits { ccd }
    }
}

impl PyCcdFits {
    pub(crate) fn new(ccd: CcdFits) -> Self {
        PyCcdFits { ccd }
    }
}

#[pyclass]
#[repr(transparent)]
#[derive(Clone)]
pub struct PyDataFrame {
    pub df: DataFrame,
}

impl From<DataFrame> for PyDataFrame {
    fn from(df: DataFrame) -> Self {
        PyDataFrame { df }
    }
}

impl PyDataFrame {
    pub(crate) fn new(df: DataFrame) -> Self {
        PyDataFrame { df }
    }
}

#[pyfunction]
pub fn py_read_fits(path: &str) -> PyResult<PyDataFrame> {
    let df_result = read_fits(path);
    let df = match df_result {
        Ok(data_frame) => data_frame,
        Err(error) => {
            // Handle the error here, e.g. print or log the error message
            panic!("Failed to read FITS file: {}", error);
        }
    };
    Ok(PyDataFrame { df })
}

use crate::ccd::PyDataFrame;
use pyo3::prelude::*;
use pyref_core::ccd::{read_experiment, ExperimentLoader, ExperimentType};

// Python bindings for the CCD library
#[pyclass]
#[repr(transparent)]
pub struct PyExperimentLoader {
    pub exp: ExperimentLoader,
}

impl From<ExperimentLoader> for PyExperimentLoader {
    fn from(exp: ExperimentLoader) -> Self {
        PyExperimentLoader { exp }
    }
}

impl PyExperimentLoader {
    pub(crate) fn new(exp: ExperimentLoader) -> Self {
        PyExperimentLoader { exp }
    }
}

#[pyfunction]
pub fn py_read_experiment(path: &str, exp_type: &str) -> PyResult<PyDataFrame> {
    let exp = match exp_type {
        "Xrr" => ExperimentType::Xrr,
        "Xrs" => ExperimentType::Xrs,
        "Other" => ExperimentType::Other,
        _ => panic!("Invalid experiment type"),
    };
    let df_result = read_experiment(path, exp);
    let df = match df_result {
        Ok(data_frame) => data_frame,
        Err(error) => {
            // Handle the error here, e.g. print or log the error message
            panic!("Failed to read FITS file: {}", error);
        }
    };
    Ok(PyDataFrame { df })
}

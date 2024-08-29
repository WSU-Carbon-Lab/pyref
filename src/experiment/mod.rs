use pyo3::prelude::*;
use pyref_core::ccd::ExperimentLoader;

// Python bindings for the CCD library
#[pyclass]
#[repr(transparent)]
struct PyExperimentLoader {
    pub data: ExperimentLoader,
}

impl From<ExperimentLoader> for PyExperimentLoader {
    fn from(data: ExperimentLoader) -> Self {
        PyCcdFits { data }
    }
}

impl PyExperimentLoader {
    pub fn new(file_path: &str) -> PyResult<Self> {
        Ok(ExperimentLoader::new(file_path)?.into())
    }
}

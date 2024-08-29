use pyo3::prelude::*;
use pyref_core::ccd::CcdFits;

// Python bindings for the CCD library
#[pyclass]
#[repr(transparent)]
struct PyCcdFits {
    pub data: CcdFits,
}

impl From<CcdFits> for PyCcdFits {
    fn from(data: CcdFits) -> Self {
        PyCcdFits { data }
    }
}

impl PyCcdFits {
    pub fn new(file_path: &str) -> PyResult<Self> {
        Ok(CcdFits::new(file_path)?.into())
    }
}

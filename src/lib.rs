pub mod ccd;
pub mod experiment;

use crate::ccd::PyCcdFits;
use crate::experiment::PyExperimentLoader;
use pyo3::prelude::*;

#[pymodule]
fn rs_pyref(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyCcdFits>()?;
    m.add_class::<PyExperimentLoader>()?;
    Ok(())
}

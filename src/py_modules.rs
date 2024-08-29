use pyo3::prelude::*;

pub(crate) static PYREF: Lazy<PyObject> =
    Lazy::new(|| Python::with_gil(|py| PyModule::import_bound(py, "pyref").unwrap().to_object(py)));

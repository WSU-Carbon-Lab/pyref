use astrors::io::header::card::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use pyref_core::loader::*;

/// This converts pyref_core structures to Python objects.
///
/// Each struct gets wrapped in a python binding class that
/// starts with the Py prefix.
///

#[pyclass]
pub struct PyCard {
    pub keyword: String,
    pub value: f64,
    pub comment: String,
}

#[pymethods]
impl PyCard {
    #[new]
    pub fn new(key: &str, value: f64, comment: &str) -> PyCard {
        PyCard {
            keyword: key.to_string(),
            value: value,
            comment: comment.to_string(),
        }
    }
}

impl PyCard {
    pub fn from_card(card: Card) -> PyCard {
        let key = card.keyword.clone();
        let value = card.value.as_float().unwrap();
        let comment = card.comment.unwrap().clone();
        PyCard {
            keyword: key,
            value: value,
            comment: comment,
        }
    }
}

#[pyclass]
pub struct PyExType {
    pub exp: ExperimentType,
}

#[pymethods]
impl PyExType {
    #[new]
    pub fn from_str(exp_type: &str) -> PyResult<PyExType> {
        let exp = ExperimentType::from_str(exp_type).unwrap();
        Ok(PyExType { exp })
    }

    pub fn get_keys(&self) -> Vec<&str> {
        self.exp.get_keys()
    }
}

#[pyclass]
pub struct PyFitsLoader {
    pub loader: FitsLoader,
}

#[pymethods]
impl PyFitsLoader {
    #[new]
    pub fn new(path: &str) -> PyResult<PyFitsLoader> {
        let loader = FitsLoader::new(path).unwrap();
        Ok(PyFitsLoader { loader })
    }
    pub fn get_card(&self, card_name: &str) -> PyResult<PyCard> {
        let card = self.loader.get_card(card_name).unwrap();
        Ok(PyCard::from_card(card))
    }
    pub fn get_value(&self, card_name: &str) -> PyResult<f64> {
        let card = self.loader.get_card(card_name).unwrap();
        let value = card.value.as_float().unwrap();
        Ok(value)
    }
    pub fn get_all_cards(&self) -> Vec<PyCard> {
        let cards = self.loader.get_all_cards();
        cards
            .iter()
            .map(|card| PyCard::from_card(card.clone()))
            .collect()
    }
    pub fn to_polars(&self, keys: Vec<String>) -> PyResult<PyDataFrame> {
        let keys: Vec<&str> = keys.iter().map(|s| s.as_str()).collect();
        let polars = self.loader.to_polars(&keys).unwrap();
        Ok(PyDataFrame(polars))
    }
}

#[pyclass]
pub struct PyExperimentLoader {
    loader: ExperimentLoader,
}

#[pymethods]
impl PyExperimentLoader {
    #[new]
    pub fn new(dir: &str, exp_type: &str) -> PyResult<PyExperimentLoader> {
        let exp_type = ExperimentType::from_str(exp_type).unwrap();
        let loader = ExperimentLoader::new(dir, exp_type).unwrap();
        Ok(PyExperimentLoader { loader })
    }
    pub fn to_polars(&self) -> PyResult<PyDataFrame> {
        let polars = self.loader.to_polars().unwrap();
        Ok(PyDataFrame(polars))
    }
}

#[pyfunction]
pub fn py_read_fits(path: &str) -> PyResult<PyDataFrame> {
    let df = read_fits(path).unwrap();
    Ok(PyDataFrame(df))
}

#[pyfunction]
pub fn py_read_experiment(dir: &str, exp_type: &str) -> PyDataFrame {
    let df = read_experiment(dir, exp_type).unwrap();
    PyDataFrame(df)
}

#[pymodule]
pub fn pyref_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyCard>()?;
    m.add_class::<PyExType>()?;
    m.add_class::<PyFitsLoader>()?;
    m.add_class::<PyExperimentLoader>()?;
    m.add_function(wrap_pyfunction!(py_read_fits, m)?)?;
    m.add_function(wrap_pyfunction!(py_read_experiment, m)?)?;
    Ok(())
}

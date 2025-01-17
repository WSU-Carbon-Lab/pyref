use astrors_fork::io::header::card::*;
// use numpy::PyArray2;
use polars::{prelude::*, series::amortized_iter::*};
use polars_core::{export::num::Pow, utils::align_chunks_binary};
use pyo3::prelude::*;
use pyo3_polars::{derive::polars_expr, PolarsAllocator, PyDataFrame};
use pyref_core::loader::*;

#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();

/// This converts pyref_core structures to Python objects.
///
/// Each struct gets wrapped in a python binding class that
/// starts with the Py prefix.
///

// ==================== Other Structs ====================
#[pyclass(module = "pyref", get_all)]
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

#[pyclass(module = "pyref")]
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

    pub fn get_keys(&self) -> Vec<String> {
        self.exp
            .get_keys()
            .iter()
            .map(|s| s.name().to_string())
            .collect()
    }
}

#[pyclass(module = "pyref")]
pub struct PyHduType {
    pub hdu: HeaderValue,
}

#[pymethods]
impl PyHduType {
    #[new]
    pub fn from_str(hdu: &str) -> PyResult<PyHduType> {
        let hdu = match hdu.to_lowercase().as_str() {
            "sample theta" => HeaderValue::SampleTheta,
            "beamline energy" => HeaderValue::BeamlineEnergy,
            "beam polarization" => HeaderValue::BeamCurrent,
            "epu polarization" => HeaderValue::EPUPolarization,
            "horizontal exit slit size" => HeaderValue::HorizontalExitSlitSize,
            "higher order suppressor" => HeaderValue::HigherOrderSuppressor,
            "exposure" => HeaderValue::Exposure,
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Invalid HDU type",
            ))?,
        };
        Ok(PyHduType { hdu })
    }
}

#[pyclass(module = "pyref")]
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
    pub fn get_card(&self, hdu: usize, card_name: &str) -> PyResult<PyCard> {
        let card = self.loader.get_card(hdu, card_name).unwrap();
        Ok(PyCard::from_card(card))
    }
    pub fn get_value(&self, hdu: usize, card_name: &str) -> PyResult<f64> {
        let card = self.loader.get_card(hdu, card_name).unwrap();
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
        let keys = keys
            .iter()
            .map(|s| match s.as_str() {
                "sample theta" => HeaderValue::SampleTheta,
                "beamline energy" => HeaderValue::BeamlineEnergy,
                "beam current" => HeaderValue::BeamCurrent,
                "epu polarization" => HeaderValue::EPUPolarization,
                "horizontal exit slit size" => HeaderValue::HorizontalExitSlitSize,
                "higher order suppressor" => HeaderValue::HigherOrderSuppressor,
                "exposure" => HeaderValue::Exposure,
                _ => panic!("Invalid HDU type"),
            })
            .collect();
        let polars = self.loader.to_polars(&keys).unwrap();
        Ok(PyDataFrame(polars))
    }
}

#[pyclass(module = "pyref")]
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
// ==================== FUNCTIONS ====================

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

#[pyfunction]
pub fn py_simple_update(df: PyDataFrame, dir: &str) -> PyDataFrame {
    let mut df = df.0;
    let _ = simple_update(&mut df, dir);
    PyDataFrame(df)
}

// #[pyfunction]
// pub fn py_get_image(vec: Vec<u16>, shape: (usize, usize)) -> PyResult<Py<PyAny>> {
//     pyo3::Python::with_gil(|py| {
//         let array = get_image(&vec, shape)
//             .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
//         Ok(PyArray2::from_array_bound(py, &array).into_py(py))
//     })
// }
// ==================== UTILS =====================
fn binary_amortized_elementwise<'a, T, K, F>(
    lhs: &'a ListChunked,
    rhs: &'a ListChunked,
    mut f: F,
) -> ChunkedArray<T>
where
    T: PolarsDataType,
    T::Array: ArrayFromIter<Option<K>>,
    F: FnMut(&AmortSeries, &AmortSeries) -> Option<K> + Copy,
{
    {
        let (lhs, rhs) = align_chunks_binary(lhs, rhs);
        lhs.amortized_iter()
            .zip(rhs.amortized_iter())
            .map(|(lhs, rhs)| match (lhs, rhs) {
                (Some(lhs), Some(rhs)) => f(&lhs, &rhs),
                _ => None,
            })
            .collect_ca(PlSmallStr::EMPTY)
    }
}

#[polars_expr(output_type=Float64)]
fn weighted_mean(inputs: &[Series]) -> PolarsResult<Series> {
    let values = inputs[0].list()?;
    let weights = &inputs[1].list()?;

    let values = values.cast(&DataType::List(Box::new(DataType::Float64)))?;
    let weights = weights.cast(&DataType::List(Box::new(DataType::Float64)))?;

    let values_ca = values.list()?.clone();
    let weights_ca = weights.list()?.clone();
    let out: Float64Chunked = binary_amortized_elementwise(
        &values_ca,
        &weights_ca,
        |values_inner: &AmortSeries, weights_inner: &AmortSeries| -> Option<f64> {
            let values_inner = values_inner.as_ref().f64().unwrap();
            let weights_inner = weights_inner.as_ref().f64().unwrap();
            if values_inner.len() == 0 {
                // Mirror Polars, and return None for empty mean.
                return None;
            }
            let mut numerator: f64 = 0.;
            let mut denominator: f64 = 0.;
            values_inner
                .iter()
                .zip(weights_inner.iter())
                .for_each(|(v, w)| {
                    if let (Some(v), Some(w)) = (v, w) {
                        let w = 1. / w.pow(2.);
                        numerator += v as f64 * w;
                        denominator += w;
                    }
                });
            Some(numerator / denominator)
        },
    );
    Ok(out.into_series())
}

#[polars_expr(output_type=Float64)]
fn weighted_std(inputs: &[Series]) -> PolarsResult<Series> {
    let values = inputs[0].list()?;
    let weights = &inputs[1].list()?;

    let values = values.cast(&DataType::List(Box::new(DataType::Float64)))?;
    let weights = weights.cast(&DataType::List(Box::new(DataType::Float64)))?;

    let values_ca = values.list()?.clone();
    let weights_ca = weights.list()?.clone();

    let out: Float64Chunked = binary_amortized_elementwise(
        &values_ca,
        &weights_ca,
        |values_inner: &AmortSeries, weights_inner: &AmortSeries| -> Option<f64> {
            let values_inner = values_inner.as_ref().f64().unwrap();
            let weights_inner = weights_inner.as_ref().f64().unwrap();
            if values_inner.len() == 0 {
                // Mirror Polars, and return None for empty mean.
                return None;
            }
            let mut denominator: f64 = 0.;
            values_inner
                .iter()
                .zip(weights_inner.iter())
                .for_each(|(v, w)| {
                    if let (Some(_), Some(w)) = (v, w) {
                        let w = 1. / w.pow(2.);
                        denominator += w;
                    }
                });
            Some((1. / denominator).sqrt())
        },
    );
    Ok(out.into_series())
}

#[polars_expr(output_type=Float64)]
fn err_prop_mult(inputs: &[Series]) -> PolarsResult<Series> {
    let lhs: &Float64Chunked = inputs[0].f64()?;
    let lhs_err: &Float64Chunked = inputs[1].f64()?;
    let rhs: &Float64Chunked = inputs[2].f64()?;
    let rhs_err: &Float64Chunked = inputs[3].f64()?;

    let out: Float64Chunked = lhs
        .into_iter()
        .zip(lhs_err.into_iter())
        .zip(rhs.into_iter())
        .zip(rhs_err.into_iter())
        .map(|(((lhs, lhs_err), rhs), rhs_err)| {
            if let (Some(lhs), Some(lhs_err), Some(rhs), Some(rhs_err)) =
                (lhs, lhs_err, rhs, rhs_err)
            {
                Some(((lhs * rhs) * ((lhs_err / lhs).powi(2) + (rhs_err / rhs).powi(2))).sqrt())
            } else {
                None
            }
        })
        .collect();
    Ok(out.into_series())
}

#[polars_expr(output_type=Float64)]
fn err_prop_div(inputs: &[Series]) -> PolarsResult<Series> {
    let lhs: &Float64Chunked = inputs[0].f64()?;
    let lhs_err: &Float64Chunked = inputs[1].f64()?;
    let rhs: &Float64Chunked = inputs[2].f64()?;
    let rhs_err: &Float64Chunked = inputs[3].f64()?;

    let out: Float64Chunked = lhs
        .into_iter()
        .zip(lhs_err.into_iter())
        .zip(rhs.into_iter())
        .zip(rhs_err.into_iter())
        .map(|(((lhs, lhs_err), rhs), rhs_err)| {
            if let (Some(lhs), Some(lhs_err), Some(rhs), Some(rhs_err)) =
                (lhs, lhs_err, rhs, rhs_err)
            {
                Some(((lhs / rhs) * ((lhs_err / lhs).powi(2) + (rhs_err / rhs).powi(2))).sqrt())
            } else {
                None
            }
        })
        .collect();
    Ok(out.into_series())
}

// ==================== MODULE ====================

#[pymodule]
#[pyo3(name = "pyref")]
pub fn pyref(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCard>()?;
    m.add_class::<PyExType>()?;
    m.add_class::<PyFitsLoader>()?;
    m.add_class::<PyExperimentLoader>()?;
    m.add_function(wrap_pyfunction!(py_read_fits, m)?)?;
    m.add_function(wrap_pyfunction!(py_read_experiment, m)?)?;
    // m.add_function(wrap_pyfunction!(py_get_image, m)?)?;
    m.add_function(wrap_pyfunction!(py_simple_update, m)?)?;
    Ok(())
}

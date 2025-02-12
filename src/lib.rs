use polars::{prelude::*, series::amortized_iter::*};
use polars_core::{export::num::Pow, utils::align_chunks_binary};
use pyo3::prelude::*;
use pyo3_polars::{derive::polars_expr, PolarsAllocator, PyDataFrame};
use pyref_core::{
    enums::ExperimentType,
    enums::HeaderValue,
    loader::{read_experiment, read_fits},
};

#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();

// ==================== Modeling XRR ====================

/// This converts pyref_core structures to Python objects.
///
/// Each struct gets wrapped in a python binding class that
/// starts with the Py prefix.
///

// ==================== Other Structs ====================
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
            .map(|s| s.hdu().to_string())
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
            "ccd theta" => HeaderValue::CCDTheta,
            "beamline energy" => HeaderValue::BeamlineEnergy,
            "beam current" => HeaderValue::BeamCurrent,
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

fn get_headers(hdu_strs: Vec<String>) -> Vec<HeaderValue> {
    hdu_strs
        .iter()
        .map(|s| match s.to_lowercase().as_str() {
            "sample theta" => HeaderValue::SampleTheta,
            "ccd theta" => HeaderValue::CCDTheta,
            "beamline energy" => HeaderValue::BeamlineEnergy,
            "beam current" => HeaderValue::BeamCurrent,
            "epu polarization" => HeaderValue::EPUPolarization,
            "horizontal exit slit size" => HeaderValue::HorizontalExitSlitSize,
            "higher order suppressor" => HeaderValue::HigherOrderSuppressor,
            "exposure" => HeaderValue::Exposure,
            _ => panic!("Invalid HDU type"),
        })
        .collect()
}

// ==================== FUNCTIONS ====================

#[pyfunction(text_signature = "(path: str, header_items: List[str])")]
pub fn py_read_fits(path: &str, header_items: Vec<String>) -> PyResult<PyDataFrame> {
    let hdus = get_headers(header_items);
    let df = read_fits(path.into(), &hdus).unwrap();
    Ok(PyDataFrame(df))
}

#[pyfunction(text_signature = "(dir: str, exp_type: str)")]
pub fn py_read_experiment(dir: &str, exp_type: &str) -> PyDataFrame {
    let exp_type = ExperimentType::from_str(exp_type).unwrap().get_keys();
    match read_experiment(dir, &exp_type) {
        Ok(df) => PyDataFrame(df),
        Err(e) => panic!("Failed to load LazyFrame into python: {}", e),
    }
}

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
// ==================== Utils =====================
pub fn err_prop_div(lhs: Expr, rhs: Expr, lhs_err: Expr, rhs_err: Expr) -> Expr {
    ((lhs.clone() / rhs.clone()) * ((lhs_err / lhs.clone()).pow(2) + (rhs_err / rhs).pow(2)))
        .sqrt()
        .into()
}

pub fn err_prop_mult(lhs: Expr, rhs: Expr, lhs_err: Expr, rhs_err: Expr) -> Expr {
    ((lhs.clone() * rhs.clone()) * ((lhs_err / lhs.clone()).pow(2) + (rhs_err / rhs).pow(2)))
        .sqrt()
        .into()
}

pub fn weighted_mean(values: Expr, weights: Expr) -> Expr {
    let values = values.cast(DataType::Float64);
    let weights = weights.cast(DataType::Float64);
    let numerator = values.clone() * weights.clone();
    let denominator = weights.clone();
    numerator.sum() / denominator.sum()
}

pub fn weighted_std(weights: Expr) -> Expr {
    let weights = weights.cast(DataType::Float64);
    let denominator = weights.clone();
    (lit(1.0) / denominator.sum()).sqrt()
}

// ==================== MODULE ====================

#[pymodule]
#[pyo3(name = "pyref")]
pub fn pyref(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyExType>()?;
    m.add_function(wrap_pyfunction!(py_read_fits, m)?)?;
    m.add_function(wrap_pyfunction!(py_read_experiment, m)?)?;
    Ok(())
}

use astrors_fork::io::header::card::*;
// use numpy::PyArray2;
use polars::{prelude::*, series::amortized_iter::*};
use polars_core::{export::num::Pow, utils::align_chunks_binary};
use pyo3::prelude::*;
use pyo3_polars::{derive::polars_expr, PolarsAllocator, PyDataFrame, PyLazyFrame};
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

fn get_headers(hdu_strs: Vec<String>) -> Vec<HeaderValue> {
    hdu_strs
        .iter()
        .map(|s| match s.to_lowercase().as_str() {
            "sample theta" => HeaderValue::SampleTheta,
            "beamline energy" => HeaderValue::BeamlineEnergy,
            "beam polarization" => HeaderValue::BeamCurrent,
            "epu polarization" => HeaderValue::EPUPolarization,
            "horizontal exit slit size" => HeaderValue::HorizontalExitSlitSize,
            "higher order suppressor" => HeaderValue::HigherOrderSuppressor,
            "exposure" => HeaderValue::Exposure,
            _ => panic!("Invalid HDU type"),
        })
        .collect()
}

// ==================== FUNCTIONS ====================

#[pyfunction]
pub fn py_read_fits(path: &str, header_items: Vec<String>) -> PyResult<PyDataFrame> {
    let hdus = get_headers(header_items);
    let df = read_fits(path.into(), &hdus).unwrap();
    Ok(PyDataFrame(df))
}

#[pyfunction]
pub fn py_read_experiment(dir: &str, exp_type: &str) -> PyLazyFrame {
    match read_experiment(dir, exp_type) {
        Ok(df) => PyLazyFrame(df),
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

// // ==================== SITCHING ==================
// pub fn get_izero_df(stitch: DataFrame) -> DataFrame {
//     let i0 = stitch
//         .clone()
//         .lazy()
//         .filter(col("Sample Theta [deg]").eq(lit(0.0)))
//         .collect()
//         .unwrap();
//     let i0_val = i0.column("I [arb. un.]").unwrap().agg_mean().unwrap();
//     let i0_err = i0.column("I [arb. un.]").unwrap().std(1).unwrap();
//     stitch
//         .lazy()
//         .with_column(lit(i0_val).alias("I₀ [arb. un.]"))
//         .with_column(lit(i0_err).alias("δI₀ [arb. un.]"))
//         .collect()
//         .unwrap()
// }

// pub fn get_reletive_izero(current_stitch: DataFrame, prior_stitch: DataFrame) -> DataFrame {
//     let energy: f64 = current_stitch
//         .column("Beamline Energy [eV]")
//         .unwrap()
//         .get(0)
//         .unwrap()
//         .extract::<f64>()
//         .unwrap();

//     // rename the current stitch so that I [arb. un.] = I[current]
//     let current_stitch = current_stitch
//         .clone()
//         .lazy()
//         .rename(["I [arb. un.]"], ["I[current]"])
//         .collect()
//         .unwrap();
//     let prior_stitch = prior_stitch
//         .clone()
//         .lazy()
//         .rename(["I [arb. un.]"], ["I[prior]"])
//         .collect()
//         .unwrap();

//     prior_stitch
//         .tail(Some(10))
//         .lazy()
//         .join(
//             current_stitch.clone().lazy(),
//             [col("Sample Theta [deg]")],
//             [col("Sample Theta [deg]")],
//             JoinArgs::default(),
//         )
//         .group_by(["Sample Theta [deg]"])
//         .agg(vec![
//             col("I[current]"),
//             col("δI[current]"),
//             col("I[prior]"),
//             col("δI[prior]"),
//             col("I₀ [arb. un.]"),
//             col("δI₀ [arb. un.]"),
//         ])
//         .with_columns(vec![
//             weighted_mean(col("I[current]"), col("δI[current]")).alias("I[current]"),
//             weighted_mean(col("I[prior]"), col("δI[prior]")).alias("I[prior]"),
//             weighted_mean(col("I₀ [arb. un.]"), col("δI₀ [arb. un.]")).alias("I₀ [arb. un.]"),
//             weighted_std(col("I[current]")).alias("δI[current]"),
//             weighted_std(col("I[prior]")).alias("δI[prior]"),
//             weighted_std(col("I₀ [arb. un.]")).alias("δI₀ [arb. un.]"),
//         ])
//         .with_columns(vec![
//             (col("I[current]") / col("I[prior]")).alias("k"),
//             err_prop_mult(
//                 col("I[current]"),
//                 col("I[prior]"),
//                 col("δI[current]"),
//                 col("δI[prior]"),
//             )
//             .alias("δk"),
//         ])
//         .with_columns(vec![
//             (col("I₀ [arb. un.]") * col("k")).alias("I₀ʳ [arb. un.]"),
//             err_prop_mult(
//                 col("I₀ [arb. un.]"),
//                 col("k"),
//                 col("δI₀ [arb. un.]"),
//                 col("δk"),
//             )
//             .alias("δI₀ʳ [arb. un.]"),
//             lit(true).alias("dummy"),
//         ])
//         .group_by(["dummy"])
//         .agg(vec![col("I₀ʳ [arb. un.]"), col("δI₀ʳ [arb. un.]")])
//         .with_columns(vec![
//             weighted_mean(col("I₀ʳ [arb. un.]"), col("δI₀ʳ [arb. un.]")).alias("I₀ [arb. un.]"),
//             weighted_std(col("I₀ʳ [arb. un.]")).alias("δI₀ [arb. un.]"),
//             lit(energy).alias("Beamline Energy [eV]"),
//         ])
//         .collect()
//         .unwrap()
// }

// #[pyfunction]
// pub fn py_get_izero_df(stitch: PyDataFrame) -> PyDataFrame {
//     let df = get_izero_df(stitch.0);
//     PyDataFrame(df)
// }

// #[pyfunction]
// pub fn py_get_reletive_izero(
//     current_stitch: PyDataFrame,
//     prior_stitch: PyDataFrame,
// ) -> PyDataFrame {
//     let df = get_reletive_izero(current_stitch.0, prior_stitch.0);
//     PyDataFrame(df)
// }

// ==================== MODULE ====================

#[pymodule]
#[pyo3(name = "pyref")]
pub fn pyref(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCard>()?;
    m.add_class::<PyExType>()?;
    m.add_function(wrap_pyfunction!(py_read_fits, m)?)?;
    m.add_function(wrap_pyfunction!(py_read_experiment, m)?)?;
    // m.add_function(wrap_pyfunction!(py_get_image, m)?)?;
    // m.add_function(wrap_pyfunction!(py_simple_update, m)?)?;
    // m.add_function(wrap_pyfunction!(py_get_izero_df, m)?)?;
    // m.add_function(wrap_pyfunction!(py_get_reletive_izero, m)?)?;
    Ok(())
}

use polars::{prelude::*, series::amortized_iter::*};
use polars_core::{export::num::Pow, utils::align_chunks_binary};
use pyo3::prelude::*;
use pyo3_polars::{derive::polars_expr, PolarsAllocator, PyDataFrame};
use pyref_core::loader::{read_experiment, read_experiment_pattern, read_fits, read_multiple_fits};

#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();

/// Read a FITS file into a DataFrame.
///
/// Parameters
/// ----------
/// path : str
///     Path to the FITS file.
/// header_items : list of str
///     List of FITS header keywords to extract.
///
/// Returns
/// -------
/// polars.DataFrame
///     DataFrame containing data from the FITS file.
///
/// Raises
/// ------
/// RuntimeError
///     If there's an error reading the FITS file.
#[pyfunction]
#[pyo3(name = "py_read_fits", signature = (path, header_items, /), text_signature = "(path, header_items, /)")]
pub fn py_read_fits(path: &str, header_items: Vec<String>) -> PyResult<PyDataFrame> {
    match read_fits(path.into(), &header_items) {
        Ok(df) => Ok(PyDataFrame(df)),
        Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            e.to_string(),
        )),
    }
}

/// Read multiple FITS files into a single DataFrame.
///
/// Parameters
/// ----------
/// file_paths : list of str
///     List of paths to FITS files to read.
/// header_items : list of str
///     List of FITS header keywords to extract.
///
/// Returns
/// -------
/// polars.DataFrame
///     DataFrame containing data from all specified FITS files.
///
/// Raises
/// ------
/// RuntimeError
///     If there's an error reading the FITS files.
///
/// Examples
/// --------
/// >>> import pyref
/// >>> files = ["file1.fits", "file2.fits"]
/// >>> headers = ["EXPTIME", "DATE-OBS"]
/// >>> df = pyref.py_read_multiple_fits(files, headers)
#[pyfunction]
#[pyo3(name = "py_read_multiple_fits")]
#[pyo3(signature = (file_paths, header_items, /), text_signature = "(file_paths, header_items, /)")]
pub fn py_read_multiple_fits(
    file_paths: Vec<String>,
    header_items: Vec<String>,
) -> PyResult<PyDataFrame> {
    // Convert Vec<String> to Vec<PathBuf>
    let fits_files: Vec<_> = file_paths.iter().map(|path| path.into()).collect();
    match read_multiple_fits(fits_files, &header_items) {
        Ok(df) => Ok(PyDataFrame(df)),
        Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            e.to_string(),
        )),
    }
}
#[pyfunction]
#[pyo3(name = "py_read_experiment_pattern")]
#[pyo3(signature = (dir, pattern, header_items, /), text_signature = "(dir, pattern, header_items, /)")]
pub fn py_read_experiment_pattern(
    dir: &str,
    pattern: &str,
    header_items: Vec<String>,
) -> PyResult<PyDataFrame> {
    match read_experiment_pattern(dir, pattern, &header_items) {
        Ok(df) => Ok(PyDataFrame(df)),
        Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            e.to_string(),
        )),
    }
}
#[pyfunction]
#[pyo3(name = "py_read_experiment")]
#[pyo3(signature = (dir, header_items, /), text_signature = "(dir, header_items, /)")]
pub fn py_read_experiment(dir: &str, header_items: Vec<String>) -> PyResult<PyDataFrame> {
    match read_experiment(dir, &header_items) {
        Ok(df) => Ok(PyDataFrame(df)),
        Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            e.to_string(),
        )),
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
    m.add_function(wrap_pyfunction!(py_read_fits, m)?)?;
    m.add_function(wrap_pyfunction!(py_read_experiment, m)?)?;
    m.add_function(wrap_pyfunction!(py_read_experiment_pattern, m)?)?;
    m.add_function(wrap_pyfunction!(py_read_multiple_fits, m)?)?;
    Ok(())
}

#[cfg(feature = "extension-module")]
use polars::prelude::*;

pub mod errors;
pub mod fits;
pub mod io;
pub mod loader;

pub use errors::FitsError;
pub use loader::{
    read_experiment_headers_only, read_fits_headers_only, read_multiple_fits_headers_only,
};
pub use io::{image_mmap, ImageInfo};

#[cfg(feature = "extension-module")]
mod extension {
    use numpy::PyArray2;
    use polars::prelude::*;
    use polars::series::amortized_iter::*;
    use polars_core::{export::num::Pow, utils::align_chunks_binary};
    use pyo3::prelude::*;
    use pyo3_polars::{derive::polars_expr, PolarsAllocator, PyDataFrame};

    use crate::io::image_mmap::{
        get_image_for_row, materialize_image_corrected, materialize_image_filtered,
        materialize_image_filtered_edges, materialize_image_unprocessed,
    };
    use crate::{
        read_experiment_headers_only, read_fits_headers_only, read_multiple_fits_headers_only,
    };

    #[global_allocator]
    static ALLOC: PolarsAllocator = PolarsAllocator::new();

    #[pyfunction]
    #[pyo3(name = "py_read_fits_headers_only")]
    #[pyo3(signature = (path, header_items, /), text_signature = "(path, header_items, /)")]
    pub fn py_read_fits_headers_only(path: &str, header_items: Vec<String>) -> PyResult<PyDataFrame> {
        match read_fits_headers_only(path.into(), &header_items) {
            Ok(df) => Ok(PyDataFrame(df)),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                e.to_string(),
            )),
        }
    }

    #[pyfunction]
    #[pyo3(name = "py_read_experiment_headers_only")]
    #[pyo3(signature = (dir, header_items, /), text_signature = "(dir, header_items, /)")]
    pub fn py_read_experiment_headers_only(
        dir: &str,
        header_items: Vec<String>,
    ) -> PyResult<PyDataFrame> {
        match read_experiment_headers_only(dir, &header_items) {
            Ok(df) => Ok(PyDataFrame(df)),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                e.to_string(),
            )),
        }
    }

    #[pyfunction]
    #[pyo3(name = "py_read_multiple_fits_headers_only")]
    #[pyo3(signature = (file_paths, header_items, /), text_signature = "(file_paths, header_items, /)")]
    pub fn py_read_multiple_fits_headers_only(
        file_paths: Vec<String>,
        header_items: Vec<String>,
    ) -> PyResult<PyDataFrame> {
        let paths: Vec<_> = file_paths.iter().map(|p| std::path::PathBuf::from(p)).collect();
        match read_multiple_fits_headers_only(paths, &header_items) {
            Ok(df) => Ok(PyDataFrame(df)),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                e.to_string(),
            )),
        }
    }

    #[pyfunction]
    #[pyo3(name = "py_get_image")]
    #[pyo3(signature = (df, row_index, /), text_signature = "(df, row_index, /)")]
    pub fn py_get_image(
        py: Python<'_>,
        df: PyDataFrame,
        row_index: usize,
    ) -> PyResult<Bound<'_, PyArray2<i64>>> {
        match materialize_image_unprocessed(&df.0, row_index) {
            Ok(arr) => Ok(PyArray2::from_owned_array_bound(py, arr)),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                e.to_string(),
            )),
        }
    }

    #[pyfunction]
    #[pyo3(name = "py_get_image_for_row")]
    #[pyo3(signature = (df, row_index, /), text_signature = "(df, row_index, /)")]
    pub fn py_get_image_for_row(
        py: Python<'_>,
        df: PyDataFrame,
        row_index: usize,
    ) -> PyResult<(Bound<'_, PyArray2<i64>>, Bound<'_, PyArray2<i64>>)> {
        match get_image_for_row(&df.0, row_index) {
            Ok((raw, subtracted)) => {
                let py_raw = PyArray2::from_owned_array_bound(py, raw);
                let py_sub = PyArray2::from_owned_array_bound(py, subtracted);
                Ok((py_raw, py_sub))
            }
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                e.to_string(),
            )),
        }
    }

    #[pyfunction]
    #[pyo3(name = "py_materialize_image_filtered")]
    #[pyo3(signature = (df, row_index, sigma, /), text_signature = "(df, row_index, sigma, /)")]
    pub fn py_materialize_image_filtered(
        py: Python<'_>,
        df: PyDataFrame,
        row_index: usize,
        sigma: f64,
    ) -> PyResult<Bound<'_, PyArray2<f32>>> {
        let info = crate::io::ImageInfo::from_dataframe_row(&df.0, row_index)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        match materialize_image_filtered(info.path.as_path(), &info, sigma) {
            Ok(arr) => Ok(PyArray2::from_owned_array_bound(py, arr)),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                e.to_string(),
            )),
        }
    }

    #[pyfunction]
    #[pyo3(name = "py_get_image_corrected")]
    #[pyo3(signature = (df, row_index, bg_rows=10, bg_cols=10), text_signature = "(df, row_index, bg_rows=10, bg_cols=10)")]
    pub fn py_get_image_corrected(
        py: Python<'_>,
        df: PyDataFrame,
        row_index: usize,
        bg_rows: usize,
        bg_cols: usize,
    ) -> PyResult<Bound<'_, PyArray2<i64>>> {
        let info = crate::io::ImageInfo::from_dataframe_row(&df.0, row_index)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        match materialize_image_corrected(info.path.as_path(), &info, bg_rows, bg_cols) {
            Ok(arr) => Ok(PyArray2::from_owned_array_bound(py, arr)),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                e.to_string(),
            )),
        }
    }

    #[pyfunction]
    #[pyo3(name = "py_materialize_image_filtered_edges")]
    #[pyo3(signature = (df, row_index, sigma, bg_rows=10, bg_cols=10), text_signature = "(df, row_index, sigma, bg_rows=10, bg_cols=10)")]
    pub fn py_materialize_image_filtered_edges(
        py: Python<'_>,
        df: PyDataFrame,
        row_index: usize,
        sigma: f64,
        bg_rows: usize,
        bg_cols: usize,
    ) -> PyResult<Bound<'_, PyArray2<f32>>> {
        let info = crate::io::ImageInfo::from_dataframe_row(&df.0, row_index)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        match materialize_image_filtered_edges(
            info.path.as_path(),
            &info,
            sigma,
            bg_rows,
            bg_cols,
        ) {
            Ok(arr) => Ok(PyArray2::from_owned_array_bound(py, arr)),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                e.to_string(),
            )),
        }
    }

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
        let (lhs, rhs) = align_chunks_binary(lhs, rhs);
        lhs.amortized_iter()
            .zip(rhs.amortized_iter())
            .map(|(lhs, rhs)| match (lhs, rhs) {
                (Some(lhs), Some(rhs)) => f(&lhs, &rhs),
                _ => None,
            })
            .collect_ca(PlSmallStr::EMPTY)
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

    #[pymodule]
    #[pyo3(name = "pyref")]
    pub fn pyref(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(pyo3::wrap_pyfunction!(py_read_fits_headers_only, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(py_read_experiment_headers_only, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(py_read_multiple_fits_headers_only, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(py_get_image, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(py_get_image_for_row, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(py_materialize_image_filtered, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(py_get_image_corrected, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(py_materialize_image_filtered_edges, m)?)?;
        Ok(())
    }
}

#[cfg(feature = "extension-module")]
pub use extension::*;

#[cfg(feature = "extension-module")]
pub fn err_prop_div(lhs: Expr, rhs: Expr, lhs_err: Expr, rhs_err: Expr) -> Expr {
    ((lhs.clone() / rhs.clone()) * ((lhs_err / lhs.clone()).pow(2) + (rhs_err / rhs).pow(2)))
        .sqrt()
        .into()
}

#[cfg(feature = "extension-module")]
pub fn err_prop_mult(lhs: Expr, rhs: Expr, lhs_err: Expr, rhs_err: Expr) -> Expr {
    ((lhs.clone() * rhs.clone()) * ((lhs_err / lhs.clone()).pow(2) + (rhs_err / rhs).pow(2)))
        .sqrt()
        .into()
}

#[cfg(feature = "extension-module")]
pub fn weighted_mean(values: Expr, weights: Expr) -> Expr {
    let values = values.cast(DataType::Float64);
    let weights = weights.cast(DataType::Float64);
    let numerator = values.clone() * weights.clone();
    let denominator = weights.clone();
    numerator.sum() / denominator.sum()
}

#[cfg(feature = "extension-module")]
pub fn weighted_std(weights: Expr) -> Expr {
    let weights = weights.cast(DataType::Float64);
    let denominator = weights.clone();
    (lit(1.0) / denominator.sum()).sqrt()
}

#[cfg(not(feature = "extension-module"))]
#[allow(dead_code)]
fn _lib_placeholder_for_tui() {}

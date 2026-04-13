#[cfg(feature = "extension-module")]
use polars::prelude::*;

pub mod beamfinding;
pub mod colormap;
pub mod errors;
pub mod fits;
pub mod gaussian_fit;
pub mod io;
pub mod loader;
pub mod path_policy;

#[cfg(feature = "catalog")]
pub mod catalog;

#[cfg(feature = "catalog")]
pub mod schema;

pub use errors::FitsError;
pub use io::options::{ReadFitsOptions, ScanFitsOptions};
pub use io::schema::FitsMetadataSchema;
pub use io::source::{FitsSource, ResolvePreference, ResolvedSource};
pub use io::{build_fits_stem, image_mmap, ImageInfo};
pub use loader::{
    catalog_from_stems, list_fits_in_dir, read_experiment_headers_only, read_fits,
    read_fits_headers_only, read_fits_metadata_batch, read_multiple_fits_headers_only, scan_fits,
    StemCatalog,
};
pub use path_policy::is_indexable_als_path;

#[cfg(feature = "extension-module")]
#[allow(clippy::useless_conversion)]
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

    #[cfg(feature = "catalog")]
    use crate::catalog::{
        beamtime_ingest_layout, classify_scan_type, get_overrides,
        ingest_beamtime_with_progress_sink, list_beamtime_entries_v2, list_beamtimes_from_catalog,
        paths, scan_from_catalog, scan_from_catalog_for_beamtime, set_override, CatalogFilter,
        IngestParallelism, IngestProgress, IngestProgressSink, ReflectivityScanType,
    };

    #[global_allocator]
    static ALLOC: PolarsAllocator = PolarsAllocator::new();

    #[pyfunction]
    #[pyo3(name = "py_read_fits_headers_only")]
    #[pyo3(signature = (path, header_items, /), text_signature = "(path, header_items, /)")]
    pub fn py_read_fits_headers_only(
        path: &str,
        header_items: Vec<String>,
    ) -> PyResult<PyDataFrame> {
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
        let paths: Vec<_> = file_paths.iter().map(std::path::PathBuf::from).collect();
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
    #[allow(clippy::type_complexity)]
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
        match materialize_image_filtered_edges(info.path.as_path(), &info, sigma, bg_rows, bg_cols)
        {
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
                let values_inner = values_inner.as_ref().f64().ok()?;
                let weights_inner = weights_inner.as_ref().f64().ok()?;
                if values_inner.is_empty() {
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
                            numerator += v * w;
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
                let values_inner = values_inner.as_ref().f64().ok()?;
                let weights_inner = weights_inner.as_ref().f64().ok()?;
                if values_inner.is_empty() {
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

    #[cfg(feature = "catalog")]
    #[pyfunction]
    #[pyo3(name = "py_default_catalog_db_path")]
    pub fn py_default_catalog_db_path() -> PyResult<String> {
        match paths::default_catalog_db_path() {
            Ok(p) => Ok(p.to_string_lossy().to_string()),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                e.to_string(),
            )),
        }
    }

    #[cfg(feature = "catalog")]
    fn ingest_progress_to_pydict<'py>(
        py: Python<'py>,
        ev: &IngestProgress,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        use pyo3::types::{PyDict, PyList};
        let d = PyDict::new_bound(py);
        match ev {
            IngestProgress::Layout { total_files, scans } => {
                d.set_item("event", "layout")?;
                d.set_item("total_files", *total_files)?;
                let list = PyList::empty_bound(py);
                for (sn, c) in scans {
                    let m = PyDict::new_bound(py);
                    m.set_item("scan_number", *sn)?;
                    m.set_item("files", *c)?;
                    list.append(m)?;
                }
                d.set_item("scans", list)?;
            }
            IngestProgress::Phase { name } => {
                d.set_item("event", "phase")?;
                d.set_item("phase", name.as_str())?;
            }
            IngestProgress::FileComplete {
                scan_number,
                scan_done,
                scan_total,
                global_done,
                global_total,
            } => {
                d.set_item("event", "file_complete")?;
                d.set_item("scan_number", *scan_number)?;
                d.set_item("scan_done", *scan_done)?;
                d.set_item("scan_total", *scan_total)?;
                d.set_item("global_done", *global_done)?;
                d.set_item("global_total", *global_total)?;
            }
        }
        Ok(d)
    }

    #[cfg(feature = "catalog")]
    #[pyfunction]
    #[pyo3(name = "py_beamtime_ingest_layout")]
    #[pyo3(signature = (beamtime_path), text_signature = "(beamtime_path)")]
    pub fn py_beamtime_ingest_layout<'py>(
        py: Python<'py>,
        beamtime_path: &str,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        use pyo3::types::{PyDict, PyList};
        let layout = beamtime_ingest_layout(std::path::Path::new(beamtime_path))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let d = PyDict::new_bound(py);
        d.set_item("total_files", layout.total_files)?;
        let list = PyList::empty_bound(py);
        for s in &layout.scans {
            let m = PyDict::new_bound(py);
            m.set_item("scan_number", s.scan_number)?;
            m.set_item("files", s.file_count)?;
            list.append(m)?;
        }
        d.set_item("scans", list)?;
        Ok(d)
    }

    #[cfg(feature = "catalog")]
    #[pyfunction]
    #[pyo3(name = "py_ingest_beamtime")]
    #[pyo3(
        signature = (beamtime_path, header_items, incremental=true, worker_threads=None, resource_fraction=None, progress_callback=None),
        text_signature = "(beamtime_path, header_items, incremental=True, worker_threads=None, resource_fraction=None, progress_callback=None)"
    )]
    pub fn py_ingest_beamtime(
        py: Python<'_>,
        beamtime_path: &str,
        header_items: Vec<String>,
        incremental: bool,
        worker_threads: Option<usize>,
        resource_fraction: Option<f64>,
        progress_callback: Option<pyo3::Py<pyo3::PyAny>>,
    ) -> PyResult<String> {
        let path = std::path::Path::new(beamtime_path);
        let parallelism = IngestParallelism {
            worker_threads,
            resource_fraction,
        };
        let progress = progress_callback.map(|cb| {
            IngestProgressSink::from_callback(move |ev| {
                Python::with_gil(|py| {
                    let Ok(d) = ingest_progress_to_pydict(py, &ev) else {
                        return;
                    };
                    let _ = cb.call1(py, (d,));
                });
            })
        });
        let result = py.allow_threads(|| {
            ingest_beamtime_with_progress_sink(
                path,
                &header_items,
                incremental,
                progress,
                parallelism,
            )
        });
        match result {
            Ok(p) => Ok(p.to_string_lossy().to_string()),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                e.to_string(),
            )),
        }
    }

    #[cfg(feature = "catalog")]
    #[pyfunction]
    #[pyo3(name = "py_scan_from_catalog")]
    #[pyo3(signature = (db_path, filter=None), text_signature = "(db_path, filter=None)")]
    pub fn py_scan_from_catalog(
        db_path: &str,
        filter: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        let path = std::path::Path::new(db_path);
        let cat_filter = filter.and_then(|f| dict_to_catalog_filter(f).ok());
        match scan_from_catalog(path, cat_filter.as_ref()) {
            Ok(df) => Ok(PyDataFrame(df)),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                e.to_string(),
            )),
        }
    }

    #[cfg(feature = "catalog")]
    #[pyfunction]
    #[pyo3(name = "py_scan_from_catalog_for_beamtime")]
    #[pyo3(
        signature = (db_path, beamtime_path, filter=None),
        text_signature = "(db_path, beamtime_path, filter=None)"
    )]
    pub fn py_scan_from_catalog_for_beamtime(
        db_path: &str,
        beamtime_path: &str,
        filter: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        let db = std::path::Path::new(db_path);
        let beam = std::path::Path::new(beamtime_path);
        let cat_filter = filter.and_then(|f| dict_to_catalog_filter(f).ok());
        match scan_from_catalog_for_beamtime(db, beam, cat_filter.as_ref()) {
            Ok(df) => Ok(PyDataFrame(df)),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                e.to_string(),
            )),
        }
    }

    #[cfg(feature = "catalog")]
    #[pyfunction]
    #[pyo3(name = "py_beamtime_entries")]
    #[pyo3(
        signature = (db_path, beamtime_path),
        text_signature = "(db_path, beamtime_path)"
    )]
    pub fn py_beamtime_entries<'py>(
        py: Python<'py>,
        db_path: &str,
        beamtime_path: &str,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let db = std::path::Path::new(db_path);
        let beam = std::path::Path::new(beamtime_path);
        let e = list_beamtime_entries_v2(db, beam)
            .map_err(|err| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(err.to_string()))?;
        let d = pyo3::types::PyDict::new_bound(py);
        d.set_item("samples", e.samples)?;
        d.set_item("tags", e.tags)?;
        d.set_item("scans", e.scans)?;
        Ok(d)
    }

    #[cfg(feature = "catalog")]
    #[pyfunction]
    #[pyo3(name = "py_list_beamtimes")]
    #[pyo3(signature = (db_path), text_signature = "(db_path)")]
    pub fn py_list_beamtimes(db_path: &str) -> PyResult<Vec<(String, i64)>> {
        let db = std::path::Path::new(db_path);
        let rows = list_beamtimes_from_catalog(db)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(rows
            .into_iter()
            .map(|(p, id)| (p.to_string_lossy().into_owned(), id))
            .collect())
    }

    #[cfg(feature = "catalog")]
    fn dict_to_catalog_filter(d: &Bound<'_, PyAny>) -> PyResult<CatalogFilter> {
        let dict = d.downcast::<pyo3::types::PyDict>()?;
        let mut f = CatalogFilter::default();
        if let Some(v) = dict.get_item("sample_name")? {
            if !v.is_none() {
                f.sample_name = Some(v.extract::<String>()?);
            }
        }
        if let Some(v) = dict.get_item("tag")? {
            if !v.is_none() {
                f.tag = Some(v.extract::<String>()?);
            }
        }
        if let Some(v) = dict.get_item("scan_numbers")? {
            if !v.is_none() {
                f.scan_numbers = Some(v.extract::<Vec<i64>>()?);
            }
        }
        if let Some(v) = dict.get_item("energy_min")? {
            if !v.is_none() {
                f.energy_min = Some(v.extract::<f64>()?);
            }
        }
        if let Some(v) = dict.get_item("energy_max")? {
            if !v.is_none() {
                f.energy_max = Some(v.extract::<f64>()?);
            }
        }
        Ok(f)
    }

    #[cfg(feature = "catalog")]
    #[pyfunction]
    #[pyo3(name = "py_get_overrides")]
    #[pyo3(signature = (db_path, path=None), text_signature = "(db_path, path=None)")]
    pub fn py_get_overrides(db_path: &str, path: Option<String>) -> PyResult<PyDataFrame> {
        let db = std::path::Path::new(db_path);
        let path_ref = path.as_deref();
        match get_overrides(db, path_ref) {
            Ok(df) => Ok(PyDataFrame(df)),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                e.to_string(),
            )),
        }
    }

    #[cfg(feature = "catalog")]
    #[pyfunction]
    #[pyo3(name = "py_set_override")]
    #[pyo3(signature = (db_path, path, sample_name=None, tag=None, notes=None), text_signature = "(db_path, path, sample_name=None, tag=None, notes=None)")]
    pub fn py_set_override(
        db_path: &str,
        path: &str,
        sample_name: Option<&str>,
        tag: Option<&str>,
        notes: Option<&str>,
    ) -> PyResult<()> {
        let db = std::path::Path::new(db_path);
        match set_override(db, path, sample_name, tag, notes) {
            Ok(()) => Ok(()),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                e.to_string(),
            )),
        }
    }

    #[cfg(feature = "catalog")]
    fn reflectivity_scan_type_id(st: ReflectivityScanType) -> &'static str {
        match st {
            ReflectivityScanType::FixedEnergy => "fixed_energy",
            ReflectivityScanType::FixedAngle => "fixed_angle",
            ReflectivityScanType::SinglePoint => "single_point",
        }
    }

    #[cfg(feature = "catalog")]
    #[pyfunction]
    #[pyo3(name = "py_classify_scan_type")]
    #[pyo3(
        signature = (pairs),
        text_signature = "(pairs)"
    )]
    /// Classify scan type from a list of ``(beamline_energy_eV, sample_theta_deg)`` pairs.
    pub fn py_classify_scan_type(
        pairs: Vec<(Option<f64>, Option<f64>)>,
    ) -> PyResult<(String, Option<f64>, Option<f64>, Option<f64>, Option<f64>)> {
        let (st, e_min, e_max, t_min, t_max) = classify_scan_type(&pairs);
        Ok((
            reflectivity_scan_type_id(st).to_string(),
            e_min,
            e_max,
            t_min,
            t_max,
        ))
    }

    #[pymodule]
    #[pyo3(name = "pyref")]
    pub fn pyref(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(pyo3::wrap_pyfunction!(py_read_fits_headers_only, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(py_read_experiment_headers_only, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(
            py_read_multiple_fits_headers_only,
            m
        )?)?;
        m.add_function(pyo3::wrap_pyfunction!(py_get_image, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(py_get_image_for_row, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(py_materialize_image_filtered, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(py_get_image_corrected, m)?)?;
        m.add_function(pyo3::wrap_pyfunction!(
            py_materialize_image_filtered_edges,
            m
        )?)?;
        #[cfg(feature = "catalog")]
        {
            m.add_function(pyo3::wrap_pyfunction!(py_default_catalog_db_path, m)?)?;
            m.add_function(pyo3::wrap_pyfunction!(py_beamtime_ingest_layout, m)?)?;
            m.add_function(pyo3::wrap_pyfunction!(py_ingest_beamtime, m)?)?;
            m.add_function(pyo3::wrap_pyfunction!(py_scan_from_catalog, m)?)?;
            m.add_function(pyo3::wrap_pyfunction!(
                py_scan_from_catalog_for_beamtime,
                m
            )?)?;
            m.add_function(pyo3::wrap_pyfunction!(py_beamtime_entries, m)?)?;
            m.add_function(pyo3::wrap_pyfunction!(py_list_beamtimes, m)?)?;
            m.add_function(pyo3::wrap_pyfunction!(py_get_overrides, m)?)?;
            m.add_function(pyo3::wrap_pyfunction!(py_set_override, m)?)?;
            m.add_function(pyo3::wrap_pyfunction!(py_classify_scan_type, m)?)?;
        }
        Ok(())
    }
}

#[cfg(feature = "extension-module")]
pub use extension::*;

#[cfg(feature = "extension-module")]
pub fn err_prop_div(lhs: Expr, rhs: Expr, lhs_err: Expr, rhs_err: Expr) -> Expr {
    ((lhs.clone() / rhs.clone()) * ((lhs_err / lhs.clone()).pow(2) + (rhs_err / rhs).pow(2))).sqrt()
}

#[cfg(feature = "extension-module")]
pub fn err_prop_mult(lhs: Expr, rhs: Expr, lhs_err: Expr, rhs_err: Expr) -> Expr {
    ((lhs.clone() * rhs.clone()) * ((lhs_err / lhs.clone()).pow(2) + (rhs_err / rhs).pow(2))).sqrt()
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

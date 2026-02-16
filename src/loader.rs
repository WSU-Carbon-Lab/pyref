//! FITS metadata loading: single-file and batch header reads, and high-level read_fits/scan_fits API.
//! Batch path is used by catalog ingest and by read_fits when source resolves to disk.

use polars::{lazy::prelude::*, prelude::*};
use rayon::prelude::*;
use std::fs;
use std::path::PathBuf;

use crate::errors::FitsError;
use crate::fits::HduList;
use crate::io::options::{ReadFitsOptions, ScanFitsOptions};
use crate::io::source::{FitsSource, ResolvedSource};
use crate::io::{add_calculated_domains, build_headers_only_columns};

#[cfg(feature = "catalog")]
use crate::catalog::scan_from_catalog;

pub fn read_fits_headers_only(
    file_path: PathBuf,
    header_items: &[String],
) -> Result<DataFrame, FitsError> {
    if file_path.extension().and_then(|ext| ext.to_str()) != Some("fits") {
        return Err(FitsError::validation("No data found"));
    }
    let path_str = file_path
        .to_str()
        .ok_or_else(|| FitsError::validation("Invalid UTF-8 in path"))?;
    let hdul = HduList::from_file_headers_only(path_str).map_err(|e| {
        FitsError::from(e)
            .with_context("operation", "read_fits_headers_only")
            .with_context("path", path_str)
    })?;
    let image_header = hdul.image_header.as_ref().ok_or_else(|| {
        FitsError::validation("No image HDU found").with_context("path", path_str)
    })?;
    let columns =
        build_headers_only_columns(&hdul.primary, image_header, file_path.clone(), header_items)
            .map_err(|e| {
                e.with_context("operation", "read_fits_headers_only")
                    .with_context("path", path_str)
            })?;
    DataFrame::new(columns).map_err(FitsError::from)
}

fn combine_dataframes_with_alignment(
    acc: DataFrame,
    df: DataFrame,
) -> Result<DataFrame, FitsError> {
    match acc.vstack(&df) {
        Ok(combined) => Ok(combined),
        Err(_e) => {
            let acc_cols = acc.get_column_names();
            let df_cols = df.get_column_names();
            let missing_in_acc: Vec<_> = df_cols.iter().filter(|c| !acc_cols.contains(c)).collect();
            let missing_in_df: Vec<_> = acc_cols.iter().filter(|c| !df_cols.contains(c)).collect();
            let missing_acc_debug = format!("{:?}", missing_in_acc);
            let missing_df_debug = format!("{:?}", missing_in_df);

            let mut acc_aligned = acc.clone();
            let mut df_aligned = df.clone();

            for col in &missing_in_acc {
                let col_name: PlSmallStr = (**col).clone();
                let null_series = Series::new_null(col_name, acc.height());
                acc_aligned.with_column(null_series).map_err(|e| {
                    FitsError::polars("Schema alignment: add column to accumulator", e)
                        .with_context("missing_in_acc", missing_acc_debug.clone())
                })?;
            }

            for col in &missing_in_df {
                let col_name: PlSmallStr = (**col).clone();
                let null_series = Series::new_null(col_name, df.height());
                df_aligned.with_column(null_series).map_err(|e| {
                    FitsError::polars("Schema alignment: add column to chunk", e)
                        .with_context("missing_in_df", missing_df_debug.clone())
                })?;
            }

            acc_aligned.vstack(&df_aligned).map_err(|pol| {
                FitsError::polars("Schema mismatch on vstack", pol)
                    .with_context("missing_in_acc", missing_acc_debug.clone())
                    .with_context("missing_in_df", missing_df_debug.clone())
            })
        }
    }
}

pub fn read_experiment_headers_only(
    dir: &str,
    header_items: &[String],
) -> Result<DataFrame, FitsError> {
    let dir_path = PathBuf::from(dir);
    if !dir_path.exists() {
        return Err(
            FitsError::not_found(format!("Directory not found: {}", dir))
                .with_context("operation", "read_experiment_headers_only")
                .with_context("path", dir),
        );
    }
    let entries: Vec<_> = fs::read_dir(dir)
        .map_err(|e| {
            FitsError::io("read_dir", e)
                .with_context("operation", "read_experiment_headers_only")
                .with_context("path", dir)
        })?
        .par_bridge()
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().extension().and_then(|ext| ext.to_str()) == Some("fits"))
        .collect();
    if entries.is_empty() {
        return Err(
            FitsError::validation(format!("No FITS files found in directory: {}", dir))
                .with_context("operation", "read_experiment_headers_only")
                .with_context("path", dir),
        );
    }
    let results: Vec<Result<DataFrame, FitsError>> = entries
        .par_iter()
        .map(|entry| read_fits_headers_only(entry.path(), header_items))
        .collect();
    let successful_dfs: Vec<DataFrame> = results.into_iter().filter_map(|r| r.ok()).collect();
    if successful_dfs.is_empty() {
        return Err(FitsError::validation(
            "None of the files in the directory could be processed successfully",
        )
        .with_context("operation", "read_experiment_headers_only")
        .with_context("path", dir));
    }
    let mut iter = successful_dfs.into_iter();
    let mut combined = iter.next().ok_or_else(|| {
        FitsError::validation("no successful FITS reads after filter")
            .with_context("operation", "read_experiment_headers_only")
            .with_context("path", dir)
    })?;
    for df in iter {
        if combined.vstack_mut(&df).is_err() {
            combined = combine_dataframes_with_alignment(combined, df)?;
        }
    }
    combined.as_single_chunk();
    Ok(add_calculated_domains(combined.lazy()))
}

/// Reads FITS metadata from a list of paths in batches, with optional calculated domains.
/// Single code path used by ingest and by `read_fits` when source resolves to disk.
pub fn read_fits_metadata_batch(
    paths: Vec<PathBuf>,
    options: &ReadFitsOptions,
) -> Result<DataFrame, FitsError> {
    if paths.is_empty() {
        return Err(FitsError::validation("No files provided")
            .with_context("operation", "read_fits_metadata_batch"));
    }
    let batch_size = options.batch_size.max(1);
    let mut chunks: Vec<DataFrame> = Vec::new();
    for chunk_paths in paths.chunks(batch_size) {
        let df = read_multiple_fits_headers_only(chunk_paths.to_vec(), &options.header_items)?;
        chunks.push(df);
    }
    if chunks.len() == 1 {
        return chunks.into_iter().next().ok_or_else(|| {
            FitsError::validation("empty batch chunk")
                .with_context("operation", "read_fits_metadata_batch")
        });
    }
    let mut iter = chunks.into_iter();
    let mut combined = iter.next().ok_or_else(|| {
        FitsError::validation("empty chunks").with_context("operation", "read_fits_metadata_batch")
    })?;
    for df in iter {
        if combined.vstack_mut(&df).is_err() {
            combined = combine_dataframes_with_alignment(combined, df)?;
        }
    }
    combined.as_single_chunk();
    Ok(combined)
}

/// Eager read of FITS metadata: resolves source (catalog or disk), returns one DataFrame.
/// Use `options.resolve_preference` to force catalog or disk when both exist.
pub fn read_fits<S, O>(source: S, options: O) -> Result<DataFrame, FitsError>
where
    S: Into<FitsSource>,
    O: Into<ReadFitsOptions>,
{
    let source = source.into();
    let options: ReadFitsOptions = options.into();
    let resolved = source.resolve(options.resolve_preference)?;
    match resolved {
        ResolvedSource::FromCatalog { db_path } => {
            #[cfg(feature = "catalog")]
            {
                let filter = options.catalog_filter.as_ref();
                scan_from_catalog(&db_path, filter).map_err(|e| {
                    FitsError::validation(e.to_string())
                        .with_context("operation", "read_fits")
                        .with_context("path", db_path.display().to_string())
                })
            }
            #[cfg(not(feature = "catalog"))]
            {
                let _ = db_path;
                Err(FitsError::unsupported(
                    "catalog not available (catalog feature disabled)",
                ))
            }
        }
        ResolvedSource::FromDisk { paths } => {
            if paths.is_empty() {
                return Err(FitsError::validation("No FITS paths to read")
                    .with_context("operation", "read_fits"));
            }
            read_fits_metadata_batch(paths, &options)
        }
    }
}

/// Lazy scan of FITS metadata: returns a LazyFrame from catalog (fast) or from disk batches.
/// When source is a dir with `.pyref_catalog.db`, use catalog if preference allows.
pub fn scan_fits<S, O>(source: S, options: O) -> Result<LazyFrame, FitsError>
where
    S: Into<FitsSource>,
    O: Into<ScanFitsOptions>,
{
    let source = source.into();
    let options: ScanFitsOptions = options.into();
    let resolved = source.resolve(options.resolve_preference)?;
    match resolved {
        ResolvedSource::FromCatalog { db_path } => {
            #[cfg(feature = "catalog")]
            {
                let filter = options.catalog_filter.as_ref();
                let df = scan_from_catalog(&db_path, filter).map_err(|e| {
                    FitsError::validation(e.to_string())
                        .with_context("operation", "scan_fits")
                        .with_context("path", db_path.display().to_string())
                })?;
                Ok(df.lazy())
            }
            #[cfg(not(feature = "catalog"))]
            {
                let _ = db_path;
                Err(FitsError::unsupported(
                    "catalog not available (catalog feature disabled)",
                ))
            }
        }
        ResolvedSource::FromDisk { paths } => {
            if paths.is_empty() {
                return Err(FitsError::validation("No FITS paths to scan")
                    .with_context("operation", "scan_fits"));
            }
            let read_opts = ReadFitsOptions {
                header_items: options.header_items,
                header_only: options.header_only,
                add_calculated_domains: options.add_calculated_domains,
                schema: options.schema,
                batch_size: options.batch_size,
                resolve_preference: options.resolve_preference,
                #[cfg(feature = "catalog")]
                catalog_filter: options.catalog_filter,
            };
            read_fits_metadata_batch(paths, &read_opts).map(|df| df.lazy())
        }
    }
}

pub fn read_multiple_fits_headers_only(
    file_paths: Vec<PathBuf>,
    header_items: &[String],
) -> Result<DataFrame, FitsError> {
    if file_paths.is_empty() {
        return Err(FitsError::validation("No files provided")
            .with_context("operation", "read_multiple_fits_headers_only"));
    }
    for path in &file_paths {
        if !path.exists() {
            return Err(
                FitsError::not_found(format!("File not found: {}", path.display()))
                    .with_context("operation", "read_multiple_fits_headers_only")
                    .with_context("path", path.display().to_string()),
            );
        }
    }
    let results: Vec<Result<DataFrame, FitsError>> = file_paths
        .par_iter()
        .map(|path| read_fits_headers_only(path.clone(), header_items))
        .collect();
    let successful_dfs: Vec<DataFrame> = results.into_iter().filter_map(|r| r.ok()).collect();
    if successful_dfs.is_empty() {
        return Err(FitsError::validation(
            "None of the provided files could be processed successfully",
        )
        .with_context("operation", "read_multiple_fits_headers_only"));
    }
    let mut iter = successful_dfs.into_iter();
    let mut combined = iter.next().ok_or_else(|| {
        FitsError::validation("no successful FITS reads")
            .with_context("operation", "read_multiple_fits_headers_only")
    })?;
    for df in iter {
        if combined.vstack_mut(&df).is_err() {
            combined = combine_dataframes_with_alignment(combined, df)?;
        }
    }
    combined.as_single_chunk();
    Ok(add_calculated_domains(combined.lazy()))
}

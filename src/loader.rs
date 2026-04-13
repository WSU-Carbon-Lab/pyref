//! FITS metadata loading: single-file and batch header reads, and high-level read_fits/scan_fits API.
//! Batch path is used by catalog ingest and by read_fits when source resolves to disk.

use polars::{lazy::prelude::*, prelude::*};
use rayon::prelude::*;
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use crate::errors::FitsError;
use crate::fits::HduList;
use crate::io::options::{ReadFitsOptions, ScanFitsOptions};
use crate::io::source::{FitsSource, ResolvedSource};
use crate::io::{
    add_calculated_domains, build_bt_ingest_row, build_headers_only_columns, parse_fits_stem,
    BtIngestRow,
};

#[cfg(feature = "catalog")]
use crate::catalog::scan_from_catalog;

#[derive(Debug, Clone, Default)]
pub struct StemCatalog {
    pub samples: Vec<String>,
    pub experiment_count: u32,
    pub fits_count: u32,
}

pub fn list_fits_in_dir(dir: &Path) -> Result<Vec<PathBuf>, FitsError> {
    let entries = fs::read_dir(dir).map_err(|e| {
        FitsError::io("read_dir", e)
            .with_context("operation", "list_fits_in_dir")
            .with_context("path", dir.display().to_string())
    })?;
    let paths: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .filter(|e| {
            let p = e.path();
            p.extension().and_then(|ext| ext.to_str()) == Some("fits") && p.is_file()
        })
        .map(|e| e.path())
        .collect();
    Ok(paths)
}

pub fn catalog_from_stems(paths: &[PathBuf]) -> StemCatalog {
    let fits_count = paths.len() as u32;
    if paths.is_empty() {
        return StemCatalog {
            samples: Vec::new(),
            experiment_count: 0,
            fits_count: 0,
        };
    }
    let (samples_set, experiment_set): (HashSet<String>, HashSet<i64>) = paths
        .par_iter()
        .filter_map(|p| {
            let stem = p.file_stem().and_then(|s| s.to_str())?;
            parse_fits_stem(stem).map(|parsed| {
                let samples: HashSet<String> = if parsed.sample_name.is_empty() {
                    HashSet::new()
                } else {
                    [parsed.sample_name].into_iter().collect()
                };
                let experiments: HashSet<i64> = if parsed.scan_number > 0 {
                    [parsed.scan_number].into_iter().collect()
                } else {
                    HashSet::new()
                };
                (samples, experiments)
            })
        })
        .reduce(
            || (HashSet::new(), HashSet::new()),
            |(mut a_s, mut a_e), (b_s, b_e)| {
                a_s.extend(b_s);
                a_e.extend(b_e);
                (a_s, a_e)
            },
        );
    let mut samples: Vec<String> = samples_set.into_iter().collect();
    samples.sort();
    StemCatalog {
        samples,
        experiment_count: experiment_set.len() as u32,
        fits_count,
    }
}

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

pub fn read_fits_headers_only_row(
    file_path: PathBuf,
    header_items: &[String],
) -> Result<BtIngestRow, FitsError> {
    if file_path.extension().and_then(|ext| ext.to_str()) != Some("fits") {
        return Err(FitsError::validation("No data found"));
    }
    let path_display = file_path.display().to_string();
    let hdul = HduList::from_file_headers_only(&path_display).map_err(|e| {
        FitsError::from(e)
            .with_context("operation", "read_fits_headers_only_row")
            .with_context("path", path_display.clone())
    })?;
    let image_header = hdul.image_header.as_ref().ok_or_else(|| {
        FitsError::validation("No image HDU found").with_context("path", path_display.clone())
    })?;
    build_bt_ingest_row(&hdul.primary, image_header, file_path, header_items).map_err(|e| {
        e.with_context("operation", "read_fits_headers_only_row")
            .with_context("path", path_display)
    })
}

pub fn read_multiple_fits_headers_only_rows(
    file_paths: Vec<PathBuf>,
    header_items: &[String],
) -> Result<Vec<BtIngestRow>, FitsError> {
    if file_paths.is_empty() {
        return Err(FitsError::validation("No files provided")
            .with_context("operation", "read_multiple_fits_headers_only_rows"));
    }
    for path in &file_paths {
        if !path.exists() {
            return Err(
                FitsError::not_found(format!("File not found: {}", path.display()))
                    .with_context("operation", "read_multiple_fits_headers_only_rows")
                    .with_context("path", path.display().to_string()),
            );
        }
    }
    file_paths
        .par_iter()
        .map(|path| read_fits_headers_only_row(path.clone(), header_items))
        .collect()
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
    let mut combined: Option<DataFrame> = None;
    for (i, res) in results.into_iter().enumerate() {
        let df = res.map_err(|e| {
            e.with_context(
                "path",
                file_paths
                    .get(i)
                    .map(|p| p.display().to_string())
                    .unwrap_or_default(),
            )
        })?;
        combined = Some(match combined.take() {
            Some(mut acc) => {
                if acc.vstack_mut(&df).is_err() {
                    combine_dataframes_with_alignment(acc, df)?
                } else {
                    acc
                }
            }
            None => df,
        });
    }
    let mut combined = combined.ok_or_else(|| {
        FitsError::validation("no FITS files in batch")
            .with_context("operation", "read_multiple_fits_headers_only")
    })?;
    combined.as_single_chunk();
    Ok(add_calculated_domains(combined.lazy()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_from_stems_aggregates_samples_and_experiments() {
        let paths: Vec<PathBuf> = [
            "monlayerjune 81041-00001.fits",
            "monlayerjune 81041-00002.fits",
            "znpc 81042-00001.fits",
        ]
        .iter()
        .map(PathBuf::from)
        .collect();
        let cat = catalog_from_stems(&paths);
        assert_eq!(cat.fits_count, 3);
        assert_eq!(cat.samples.len(), 2);
        assert!(cat.samples.contains(&"monlayerjune".to_string()));
        assert!(cat.samples.contains(&"znpc".to_string()));
        assert_eq!(cat.experiment_count, 2);
    }

    #[test]
    fn catalog_from_stems_skips_invalid_stems() {
        let paths: Vec<PathBuf> = [
            "valid_sample 81041-00001.fits",
            "invalid.fits",
            "short1-00001.fits",
        ]
        .iter()
        .map(PathBuf::from)
        .collect();
        let cat = catalog_from_stems(&paths);
        assert_eq!(cat.fits_count, 3);
        assert!(cat.samples.len() <= 2);
    }

    #[test]
    fn catalog_from_stems_empty() {
        let paths: Vec<PathBuf> = vec![];
        let cat = catalog_from_stems(&paths);
        assert_eq!(cat.fits_count, 0);
        assert!(cat.samples.is_empty());
        assert_eq!(cat.experiment_count, 0);
    }

    #[test]
    fn list_fits_in_dir_nonexistent_returns_error() {
        let res = list_fits_in_dir(Path::new("/nonexistent_path_xyz_123"));
        assert!(res.is_err());
        let err = res.unwrap_err();
        assert!(!err.context.is_empty());
        assert!(err.context.iter().any(|(k, _)| k == "operation"));
    }

    #[test]
    fn list_fits_in_dir_temp_dir() {
        let dir = tempfile::tempdir().expect("tempdir");
        let d = dir.path();
        std::fs::write(d.join("a.fits"), b"").expect("write a.fits");
        std::fs::write(d.join("b.fits"), b"").expect("write b.fits");
        std::fs::write(d.join("c.txt"), b"").expect("write c.txt");
        let paths = list_fits_in_dir(d).expect("list_fits_in_dir");
        assert_eq!(paths.len(), 2);
        let names: Vec<String> = paths
            .iter()
            .filter_map(|p| p.file_name().and_then(|n| n.to_str().map(String::from)))
            .collect();
        assert!(names.contains(&"a.fits".to_string()));
        assert!(names.contains(&"b.fits".to_string()));
    }
}

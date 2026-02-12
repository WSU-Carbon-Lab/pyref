use polars::{lazy::prelude::*, prelude::*};
use rayon::prelude::*;
use std::fs;
use std::path::PathBuf;

use crate::errors::FitsError;
use crate::fits::HduList;
use crate::io::{add_calculated_domains, build_headers_only_columns};

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
    let hdul = HduList::from_file_headers_only(path_str)
        .map_err(|e| FitsError::from(e).with_context("operation", "read_fits_headers_only").with_context("path", path_str))?;
    let image_header = hdul
        .image_header
        .as_ref()
        .ok_or_else(|| FitsError::validation("No image HDU found").with_context("path", path_str))?;
    let columns = build_headers_only_columns(
        &hdul.primary,
        image_header,
        file_path.clone(),
        header_items,
    )
    .map_err(|e| e.with_context("operation", "read_fits_headers_only").with_context("path", path_str))?;
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
                let col_name: PlSmallStr = (**col).clone().into();
                let null_series = Series::new_null(col_name, acc.height());
                let _ = acc_aligned.with_column(null_series).unwrap();
            }

            for col in &missing_in_df {
                let col_name: PlSmallStr = (**col).clone().into();
                let null_series = Series::new_null(col_name, df.height());
                let _ = df_aligned.with_column(null_series).unwrap();
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
    let successful_dfs: Vec<DataFrame> = results
        .into_iter()
        .filter_map(|r| r.ok())
        .collect();
    if successful_dfs.is_empty() {
        return Err(
            FitsError::validation(
                "None of the files in the directory could be processed successfully",
            )
            .with_context("operation", "read_experiment_headers_only")
            .with_context("path", dir),
        );
    }
    let mut iter = successful_dfs.into_iter();
    let mut combined = iter.next().expect("non-empty");
    for df in iter {
        if combined.vstack_mut(&df).is_err() {
            combined = combine_dataframes_with_alignment(combined, df)?;
        }
    }
    combined.as_single_chunk();
    Ok(add_calculated_domains(combined.lazy()))
}

pub fn read_multiple_fits_headers_only(
    file_paths: Vec<PathBuf>,
    header_items: &[String],
) -> Result<DataFrame, FitsError> {
    if file_paths.is_empty() {
        return Err(
            FitsError::validation("No files provided")
                .with_context("operation", "read_multiple_fits_headers_only"),
        );
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
    let successful_dfs: Vec<DataFrame> = results
        .into_iter()
        .filter_map(|r| r.ok())
        .collect();
    if successful_dfs.is_empty() {
        return Err(
            FitsError::validation(
                "None of the provided files could be processed successfully",
            )
            .with_context("operation", "read_multiple_fits_headers_only"),
        );
    }
    let mut iter = successful_dfs.into_iter();
    let mut combined = iter.next().expect("non-empty");
    for df in iter {
        if combined.vstack_mut(&df).is_err() {
            combined = combine_dataframes_with_alignment(combined, df)?;
        }
    }
    combined.as_single_chunk();
    Ok(add_calculated_domains(combined.lazy()))
}

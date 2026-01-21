use astrors_fork::io::hdus::{
    image::{imagehdu::ImageHDU, ImageData},
    primaryhdu::PrimaryHDU,
};
use ndarray::{ArrayBase, Axis, Dim, IxDynImpl, OwnedRepr};
use polars::prelude::*;
use std::ops::Mul;

use crate::errors::FitsLoaderError;

pub fn q(lam: f64, theta: f64) -> f64 {
    let theta = theta;
    match 4.0 * std::f64::consts::PI * theta.to_radians().sin() / lam {
        q if q < 0.0 => 0.0,
        q => q,
    }
}

pub fn col_from_array(
    name: PlSmallStr,
    array: ArrayBase<OwnedRepr<i64>, Dim<IxDynImpl>>,
) -> Result<Column, PolarsError> {
    let rows = array.len_of(Axis(0));
    let cols = array.len_of(Axis(1));

    // Create a C-contiguous version of the array without transposition to
    // preserve native FITS orientation.
    let c_contiguous_array = array.as_standard_layout().into_owned();
    let flat_data = c_contiguous_array.as_slice().unwrap();

    let mut list_builder = ListPrimitiveChunkedBuilder::<Int64Type>::new(
        name.clone(),
        rows,
        rows * cols,
        DataType::Int64,
    );

    for row_chunk in flat_data.chunks_exact(cols) {
        list_builder.append_slice(row_chunk);
    }

    let series_of_lists = list_builder.finish().into_series();

    // Implode the series of lists into a single row containing a list of lists.
    let series_of_list_of_lists = series_of_lists.implode()?;

    // Finally, cast to a 2D Array type with explicit dimensions.
    // This preserves the 2D structure of the data.
    let array_series = series_of_list_of_lists.cast(&DataType::Array(
        Box::new(DataType::Array(Box::new(DataType::Int64), cols)),
        rows,
    ))?;

    Ok(array_series.into_column())
}
// ================== CCD Raw Data Processing ============
pub fn add_calculated_domains(lzf: LazyFrame) -> DataFrame {
    let h = physical_constants::PLANCK_CONSTANT_IN_EV_PER_HZ;
    let c = physical_constants::SPEED_OF_LIGHT_IN_VACUUM * 1e10;

    // Collect schema once to check for column existence
    let schema = lzf.clone().collect_schema().unwrap_or_default();
    let has_column = |name: &str| schema.iter().any(|(col_name, _)| col_name == name);

    // Start with basic sorting that won't fail even if columns don't exist
    let mut lz = lzf;

    // Apply optional sorting only if the columns exist
    if has_column("DATE") {
        lz = lz.sort(["DATE"], Default::default());
    }

    if has_column("file_name") {
        lz = lz.sort(["file_name"], Default::default());
    }

    // Conditionally apply column transformations only if the columns exist
    if has_column("EXPOSURE") {
        lz = lz.with_column(col("EXPOSURE").round(3).alias("EXPOSURE"));
    }

    if has_column("Higher Order Suppressor") {
        lz = lz.with_column(
            col("Higher Order Suppressor")
                .round(2)
                .alias("Higher Order Suppressor"),
        );
    }

    if has_column("Horizontal Exit Slit Size") {
        lz = lz.with_column(
            col("Horizontal Exit Slit Size")
                .round(1)
                .alias("Horizontal Exit Slit Size"),
        );
    }

    if has_column("Beamline Energy") {
        lz = lz.with_column(col("Beamline Energy").round(1).alias("Beamline Energy"));

        // Only calculate Lambda if Beamline Energy exists
        lz = lz.with_column(
            col("Beamline Energy")
                .pow(-1)
                .mul(lit(h * c))
                .alias("Lambda"),
        );
    }

    // Add Q column if required columns exist
    lz = lz.with_column(
        when(
            col("Sample Theta")
                .is_not_null()
                .and(col("Lambda").is_not_null()),
        )
        .then(as_struct(vec![col("Sample Theta"), col("Lambda")]).map(
            move |s| {
                let struc = s.struct_()?;
                let th_series = struc.field_by_name("Sample Theta")?;
                let theta = th_series.f64()?;
                let lam_series = struc.field_by_name("Lambda")?;
                let lam = lam_series.f64()?;

                let out: Float64Chunked = theta
                    .into_iter()
                    .zip(lam.iter())
                    .map(|(theta, lam)| match (theta, lam) {
                        (Some(theta), Some(lam)) => Some(q(lam, theta)),
                        _ => None,
                    })
                    .collect();

                Ok(Some(out.into_column()))
            },
            GetOutput::from_type(DataType::Float64),
        ))
        .otherwise(lit(NULL))
        .alias("Q"),
    );

    // Collect the final DataFrame only once at the end
    lz.collect().unwrap_or_else(|_| DataFrame::empty())
}

/// Reads a single FITS file and converts it to a Polars DataFrame.
///
/// # Arguments
///
/// * `file_path` - Path to the FITS file to read
/// * `header_items` - List of header values to extract
///
/// # Returns
///
/// A `Result` containing either the DataFrame or a `FitsLoaderError`.
pub fn process_image(img: &ImageHDU) -> Result<Vec<Column>, FitsLoaderError> {
    let bzero = img
        .header
        .get_card("BZERO")
        .ok_or_else(|| FitsLoaderError::MissingHeaderKey("BZERO".into()))?
        .value
        .as_int()
        .ok_or_else(|| FitsLoaderError::FitsError("BZERO not an integer".into()))?;

    match &img.data {
        ImageData::I16(image) => {
            let data = image.map(|&x| i64::from(x as i64 + bzero));
            // Implement row-by-row background subtraction
            let subtracted = subtract_background(&data);
            // Locate the index tuple with the maximum value
            // Find the coordinates of the maximum value in the 2D array
            let max_coords = {
                let mut max_coords = (0, 0);
                let mut max_val = i64::MIN;

                for (idx, &val) in subtracted.indexed_iter() {
                    if val > max_val {
                        max_val = val;
                        max_coords = (idx[0], idx[1]);
                    }
                }

                max_coords
            };

            // Check if the beam is too close to any edge (top, bottom, left, right)
            let msg = if max_coords.0 < 20
                || max_coords.0 > (subtracted.len_of(Axis(0)) - 20)
                || max_coords.1 < 20
                || max_coords.1 > (subtracted.len_of(Axis(1)) - 20)
            {
                "Simple Detection Error: Beam is too close to the edge"
            } else {
                ""
            };
            // Calculate a simple reflectivity result from the subtracted data
            let (db_sum, scaled_bg) = { simple_reflectivity(&subtracted, max_coords) };

            Ok(vec![
                col_from_array("RAW".into(), data.clone()).unwrap(),
                col_from_array("SUBTRACTED".into(), subtracted.clone()).unwrap(),
                Column::new("Simple Spot X".into(), vec![max_coords.0 as u64]),
                Column::new("Simple Spot Y".into(), vec![max_coords.1 as u64]),
                Column::new(
                    "Simple Reflectivity".into(),
                    vec![(db_sum - scaled_bg) as f64],
                ),
                Series::new("status".into(), vec![msg.to_string()]).into_column(),
            ])
        }
        _ => Err(FitsLoaderError::UnsupportedImageData),
    }
}

fn simple_reflectivity(
    subtracted: &ArrayBase<OwnedRepr<i64>, Dim<IxDynImpl>>,
    max_index: (usize, usize),
) -> (i64, i64) {
    // Convert max_index to 2D coordinates
    let beam_y = max_index.0;
    // Row
    let beam_x = max_index.1;
    // Column

    // Define ROI size
    let roi = 5;
    // Region of interest size

    // Define ROI boundaries
    let roi_start_y = beam_y.saturating_sub(roi);
    let roi_end_y = (beam_y + roi + 1).min(subtracted.len_of(Axis(0)));
    let roi_start_x = beam_x.saturating_sub(roi);
    let roi_end_x = (beam_x + roi + 1).min(subtracted.len_of(Axis(1)));

    // Initialize sums and counts
    let mut db_sum = 0i64;
    let mut db_count = 0i64;
    let mut bg_sum = 0i64;
    let mut bg_count = 0i64;

    // Iterate over all rows and columns
    for y in 0..subtracted.len_of(Axis(0)) {
        for x in 0..subtracted.len_of(Axis(1)) {
            let value = subtracted[[y, x]];
            if value == 0 {
                continue;
            }

            if (roi_start_y <= y && y < roi_end_y) && (roi_start_x <= x && x < roi_end_x) {
                db_sum += value;
                db_count += 1;
            } else {
                bg_sum += value;
                bg_count += 1;
            }
        }
    }

    // Handle edge cases
    if bg_count == 0 || db_sum == 0 {
        (0, 0)
    } else {
        // Scale background sum based on ratio of counts
        let scaled_bg = (bg_sum * db_count) / bg_count;
        (db_sum, scaled_bg)
    }
}

fn subtract_background(
    data: &ArrayBase<OwnedRepr<i64>, Dim<IxDynImpl>>,
) -> ArrayBase<OwnedRepr<i64>, Dim<IxDynImpl>> {
    // Get a view of the data with 5 pixels sliced from each side
    let view = data.slice(ndarray::s![5..-5, 5..-5]);
    let rows = view.len_of(Axis(0));
    let cols = view.len_of(Axis(1));

    // Extract the left and right columns (first and last 20 columns)
    let left = view.slice(ndarray::s![.., ..20]);
    let right = view.slice(ndarray::s![.., (cols - 20)..]);

    // Calculate the sum of left and right regions
    let left_sum: i64 = left.iter().copied().sum();
    let right_sum: i64 = right.iter().copied().sum();

    // Create background array to store row means
    let mut background = ndarray::Array1::zeros(rows);

    // Determine which side to use for background
    if left_sum < right_sum {
        // Use right side as background
        for (i, row) in right.axis_iter(Axis(0)).enumerate() {
            background[i] = row.iter().copied().sum::<i64>() / row.len() as i64;
        }
    } else {
        // Use left side as background
        for (i, row) in left.axis_iter(Axis(0)).enumerate() {
            background[i] = row.iter().copied().sum::<i64>() / row.len() as i64;
        }
    }

    // Create a new owned array from the view
    let mut result = view.to_owned();

    // Subtract background from each row
    for (i, mut row) in result.axis_iter_mut(Axis(0)).enumerate() {
        let bg = background[i];
        for val in row.iter_mut() {
            *val -= bg;
        }
    }
    result.into_dyn()
}

pub fn process_metadata(
    hdu: &PrimaryHDU,
    keys: &Vec<String>,
) -> Result<Vec<Column>, FitsLoaderError> {
    if keys.is_empty() {
        // If no specific keys are requested, return all header values
        Ok(hdu
            .header
            .iter()
            .filter(|card| !card.keyword.as_str().to_lowercase().contains("comment"))
            .map(|card| {
                let name = card.keyword.as_str();
                let value = card.value.as_float().unwrap_or(0.0);
                Column::new(name.into(), &[value])
            })
            .collect())
    } else {
        // Process each requested header key
        let mut columns = Vec::new();

        for key in keys {
            // Special handling for Beamline Energy
            if key == "Beamline Energy" {
                // First try to get "Beamline Energy"
                if let Some(card) = hdu.header.get_card(key) {
                    if let Some(val) = card.value.as_float() {
                        columns.push(Column::new(key.into(), &[val]));
                        continue;
                    }
                }

                // Then fall back to "Beamline Energy Goal" if "Beamline Energy" is not present
                if let Some(card) = hdu.header.get_card("Beamline Energy Goal") {
                    if let Some(val) = card.value.as_float() {
                        columns.push(Column::new(key.into(), &[val]));
                        continue;
                    }
                }

                // If neither value is available, use a default
                columns.push(Column::new(key.into(), &[0.0]));
                continue;
            }

            // Special handling for Date header (it's a string value, not a float)
            if key == "DATE" {
                if let Some(card) = hdu.header.get_card(key) {
                    let val = card.value.to_string();
                    columns.push(Column::new(key.into(), &[val]));
                    continue;
                }
                // If DATE is not present, use a default empty string
                columns.push(Column::new(key.into(), &["".to_string()]));
                continue;
            }

            // For other headers, don't fail if they're missing
            let val = match hdu.header.get_card(key) {
                Some(card) => card.value.as_float().unwrap_or(1.0),
                None => 0.0, // Default value for missing headers
            };

            // Use the snake_case name from the enum variant
            columns.push(Column::new(key.into(), &[val]));
        }

        Ok(columns)
    }
}

pub fn process_file_name(path: std::path::PathBuf) -> Vec<Column> {
    // Extract just the file name without extension
    let file_name = path.file_stem().unwrap().to_str().unwrap_or("");

    // Just return the file name directly, without extracting frame numbers or scan IDs
    vec![Column::new("file_name".into(), vec![file_name])]
}

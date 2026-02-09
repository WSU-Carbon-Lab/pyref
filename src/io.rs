use ndarray::{ArrayBase, Axis, Dim, IxDynImpl, OwnedRepr};
use polars::prelude::*;
use regex::Regex;
use std::ops::Mul;

use crate::errors::FitsLoaderError;
use crate::fits::{ImageHdu, ImageHduHeader, PrimaryHdu};

#[derive(Debug, Clone)]
pub struct ParsedFitsStem {
    pub file_stem: String,
    pub sample_name: String,
    pub tag: Option<String>,
    pub experiment_number: i64,
    pub frame_number: i64,
}

pub fn parse_fits_stem(stem: &str) -> Option<ParsedFitsStem> {
    let stem = stem.trim();
    let re = Regex::new(r"^(.+?)[\s\-_]?(\d{5})-(\d{5})$").ok()?;
    let cap = re.captures(stem)?;
    let base = cap.get(1)?.as_str().trim();
    let exp_str = cap.get(2)?.as_str();
    let frame_str = cap.get(3)?.as_str();
    let experiment_number: i64 = exp_str.parse().ok()?;
    let frame_number: i64 = frame_str.parse().ok()?;
    let (sample_name, tag) = if base.contains('_') {
        let parts: Vec<&str> = base.split('_').collect();
        let (last, rest) = parts.split_last()?;
        (rest.join("_"), Some((*last).to_string()))
    } else {
        (base.to_string(), None)
    };
    Some(ParsedFitsStem {
        file_stem: stem.to_string(),
        sample_name,
        tag,
        experiment_number,
        frame_number,
    })
}

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

    if has_column("Sample Theta") && has_column("Lambda") {
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
    }

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
pub fn process_image(img: &ImageHdu) -> Result<Vec<Column>, FitsLoaderError> {
    let bzero = img
        .header
        .get_card("BZERO")
        .ok_or_else(|| FitsLoaderError::MissingHeaderKey("BZERO".into()))?
        .value
        .as_int()
        .ok_or_else(|| FitsLoaderError::FitsError("BZERO not an integer".into()))?;
    let data = img.data.map(|&x| x as i64 + bzero).into_dyn();
    let subtracted = subtract_background(&data);
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
    let msg = if max_coords.0 < 20
        || max_coords.0 > (subtracted.len_of(Axis(0)) - 20)
        || max_coords.1 < 20
        || max_coords.1 > (subtracted.len_of(Axis(1)) - 20)
    {
        "Simple Detection Error: Beam is too close to the edge"
    } else {
        ""
    };
    let (db_sum, scaled_bg) = simple_reflectivity(&subtracted, max_coords);
    Ok(vec![
        col_from_array("RAW".into(), data.clone())?,
        col_from_array("SUBTRACTED".into(), subtracted.clone())?,
        Column::new("Simple Spot X".into(), vec![max_coords.0 as u64]),
        Column::new("Simple Spot Y".into(), vec![max_coords.1 as u64]),
        Column::new(
            "Simple Reflectivity".into(),
            vec![(db_sum - scaled_bg) as f64],
        ),
        Series::new("status".into(), vec![msg.to_string()]).into_column(),
    ])
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
    let rows = data.len_of(Axis(0));
    let cols = data.len_of(Axis(1));
    if rows < 11 || cols < 41 {
        return data.to_owned();
    }
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

pub fn process_image_header(img: &ImageHduHeader) -> Vec<Column> {
    vec![
        Column::new("NAXIS1".into(), vec![img.naxis1 as i64]),
        Column::new("NAXIS2".into(), vec![img.naxis2 as i64]),
    ]
}

pub fn process_metadata(
    hdu: &PrimaryHdu,
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
    let file_name = path.file_stem().unwrap().to_str().unwrap_or("");
    let mut columns = vec![Column::new("file_name".into(), vec![file_name.to_string()])];
    match parse_fits_stem(file_name) {
        Some(p) => {
            columns.push(Column::new("sample_name".into(), vec![p.sample_name]));
            let tag_series = Series::from_iter(std::iter::once(p.tag.as_deref()))
                .with_name("tag".into())
                .into_column();
            columns.push(tag_series);
            columns.push(Column::new("experiment_number".into(), vec![p.experiment_number]));
            columns.push(Column::new("frame_number".into(), vec![p.frame_number]));
        }
        None => {
            columns.push(Column::new("sample_name".into(), vec!["".to_string()]));
            columns.push(
                Series::from_iter(std::iter::once(Option::<&str>::None))
                    .with_name("tag".into())
                    .into_column(),
            );
            columns.push(Column::new("experiment_number".into(), vec![0i64]));
            columns.push(Column::new("frame_number".into(), vec![0i64]));
        }
    }
    columns
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_fits_stem_znpc_variants() {
        let cases = [
            ("ZnPc_rt81041-00001", "ZnPc", Some("rt"), 81041, 1),
            ("ZnPc_rt_81041-00001", "ZnPc", Some("rt"), 81041, 1),
            ("ZnPc_rt 81041-00001", "ZnPc", Some("rt"), 81041, 1),
            ("ZnPc_rt-81041-00001", "ZnPc", Some("rt"), 81041, 1),
        ];
        for (stem, sample, tag, exp_num, frame_num) in cases {
            let p = parse_fits_stem(stem).expect(stem);
            assert_eq!(p.sample_name, sample, "stem: {}", stem);
            assert_eq!(p.tag.as_deref(), tag, "stem: {}", stem);
            assert_eq!(p.experiment_number, exp_num, "stem: {}", stem);
            assert_eq!(p.frame_number, frame_num, "stem: {}", stem);
        }
    }

    #[test]
    fn test_parse_fits_stem_ps_pmma() {
        let p = parse_fits_stem("ps_pmma_rt 81041-00001").expect("should parse");
        assert_eq!(p.sample_name, "ps_pmma");
        assert_eq!(p.tag.as_deref(), Some("rt"));
        assert_eq!(p.experiment_number, 81041);
        assert_eq!(p.frame_number, 1);
    }

    #[test]
    fn test_parse_fits_stem_monlayerjune() {
        let p = parse_fits_stem("monlayerjune 81041-00007").expect("should parse");
        assert_eq!(p.sample_name, "monlayerjune");
        assert_eq!(p.tag, None);
        assert_eq!(p.experiment_number, 81041);
        assert_eq!(p.frame_number, 7);
        let p2 = parse_fits_stem("monlayerjune 81041-00001").expect("should parse");
        assert_eq!(p2.sample_name, "monlayerjune");
        assert_eq!(p2.tag, None);
        assert_eq!(p2.frame_number, 1);
    }

    #[test]
    fn test_parse_fits_stem_invalid() {
        assert!(parse_fits_stem("notavalidstem").is_none());
        assert!(parse_fits_stem("short1-00001").is_none());
        assert!(parse_fits_stem("sample12345-678").is_none());
    }
}

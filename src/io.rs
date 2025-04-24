use astrors_fork::io::hdus::{
    image::{imagehdu::ImageHDU, ImageData},
    primaryhdu::PrimaryHDU,
};
use ndarray::{ArrayBase, Axis, Dim, IxDynImpl, OwnedRepr};
use polars::prelude::*;
use std::ops::Mul;

use crate::enums::HeaderValue;
use crate::errors::FitsLoaderError;

// Find the theta offset between theta and the ccd theta or 2theta
pub fn theta_offset(theta: f64, ccd_theta: f64) -> f64 {
    // 2theta = -ccd_theta / 2 - theta rounded to 3 decimal places
    let ccd_theta = ccd_theta / 2.0;
    ccd_theta - theta
}

pub fn q(lam: f64, theta: f64, angle_offset: f64) -> f64 {
    let theta = theta - angle_offset;
    match 4.0 * std::f64::consts::PI * theta.to_radians().sin() / lam {
        q if q < 0.0 => 0.0,
        q => q,
    }
}

pub fn col_from_array(
    name: PlSmallStr,
    array: ArrayBase<OwnedRepr<i64>, Dim<IxDynImpl>>,
) -> Result<Column, PolarsError> {
    let size0 = array.len_of(Axis(0));
    let size1 = array.len_of(Axis(1));

    let mut s = Column::new_empty(
        name,
        &DataType::List(Box::new(DataType::List(Box::new(DataType::Int64)))),
    );

    let mut chunked_builder = ListPrimitiveChunkedBuilder::<Int64Type>::new(
        PlSmallStr::EMPTY,
        array.len_of(Axis(0)),
        array.len_of(Axis(1)) * array.len_of(Axis(0)),
        DataType::Int64,
    );
    for row in array.axis_iter(Axis(0)) {
        match row.as_slice() {
            Some(row) => chunked_builder.append_slice(row),
            None => chunked_builder.append_slice(&row.to_owned().into_raw_vec()),
        }
    }
    let new_series = chunked_builder
        .finish()
        .into_series()
        .implode()
        .unwrap()
        .into_column();
    let _ = s.extend(&new_series);
    let s = s.cast(&DataType::Array(
        Box::new(DataType::Array(Box::new(DataType::Int32), size1)),
        size0,
    ));
    s
}
// ================== CCD Raw Data Processing ============
pub fn add_calculated_domains(lzf: LazyFrame) -> DataFrame {
    let h = physical_constants::PLANCK_CONSTANT_IN_EV_PER_HZ;
    let c = physical_constants::SPEED_OF_LIGHT_IN_VACUUM * 1e10;

    // Calculate lambda and q values in angstrom
    let lz = lzf
        // Sort by date first (primary sorting key)
        .sort(["date"], Default::default())
        // File name as fallback sorting key
        .sort(["file_name"], Default::default())
        .with_columns(&[
            col("exposure").round(3).alias("exposure"),
            col("higher_order_suppressor")
                .round(2)
                .alias("higher_order_suppressor"),
            col("horizontal_exit_slit_size")
                .round(1)
                .alias("horizontal_exit_slit_size"),
            col("beamline_energy").round(1).alias("beamline_energy"),
        ])
        .with_column(
            col("beamline_energy")
                .pow(-1)
                .mul(lit(h * c))
                .alias("lambda"),
        );

    let angle_offset = lz
        .clone()
        .filter(col("sample_theta").eq(0.0))
        .last()
        .select(&[col("ccd_theta"), col("sample_theta")])
        .with_column(
            as_struct(vec![col("sample_theta"), col("ccd_theta")])
                .map(
                    move |s| {
                        let struc = s.struct_()?;
                        let th_series = struc.field_by_name("sample_theta")?;
                        let theta = th_series.f64()?;
                        let ccd_th_series = struc.field_by_name("ccd_theta")?;
                        let ccd_theta = ccd_th_series.f64()?;
                        let out: Float64Chunked = theta
                            .into_iter()
                            .zip(ccd_theta.iter())
                            .map(|(theta, ccd_theta)| match (theta, ccd_theta) {
                                (Some(theta), Some(ccd_theta)) => {
                                    Some(theta_offset(theta, ccd_theta))
                                }
                                _ => Some(0.0),
                            })
                            .collect();
                        Ok(Some(out.into_column()))
                    },
                    GetOutput::from_type(DataType::Float64),
                )
                .alias("theta_offset"),
        )
        .select(&[col("theta_offset")])
        .collect()
        .unwrap()
        .get(0)
        .unwrap()[0]
        .try_extract::<f64>()
        .unwrap_or(0.0); // Add default value if extraction fails

    // get the row cor
    let lz = lz
        .with_column(lit(angle_offset).alias("theta_offset"))
        .with_column(
            as_struct(vec![col("sample_theta"), col("lambda")])
                .map(
                    move |s| {
                        let struc = s.struct_()?;
                        let th_series = struc.field_by_name("sample_theta")?;
                        let theta = th_series.f64()?;
                        let lam_series = struc.field_by_name("lambda")?;
                        let lam = lam_series.f64()?;

                        let out: Float64Chunked = theta
                            .into_iter()
                            .zip(lam.iter())
                            .map(|(theta, lam)| match (theta, lam) {
                                (Some(theta), Some(lam)) => Some(q(lam, theta, angle_offset)),
                                _ => None,
                            })
                            .collect();

                        Ok(Some(out.into_column()))
                    },
                    GetOutput::from_type(DataType::Float64),
                )
                .alias("q"),
        );
    lz.collect().unwrap_or_else(|_| DataFrame::empty())
}

//
// ================== CCD Data Loader ==================
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
            Ok(vec![col_from_array("raw".into(), data.clone()).unwrap()])
        }
        _ => Err(FitsLoaderError::UnsupportedImageData),
    }
}

pub fn process_metadata(
    hdu: &PrimaryHDU,
    keys: &Vec<HeaderValue>,
) -> Result<Vec<Column>, FitsLoaderError> {
    if keys.is_empty() {
        // If no specific keys are requested, return all header values
        Ok(hdu
            .header
            .iter()
            .map(|card| {
                let name = card.keyword.as_str();
                let value = card.value.as_float().unwrap_or(0.0);
                // Convert to snake_case without units
                // let clean_name = to_snake_case(name);
                Column::new(name.into(), &[value])
            })
            .collect())
    } else {
        // Process each requested header key
        let mut columns = Vec::new();

        for key in keys {
            // Special handling for Beamline Energy
            if key.hdu() == "Beamline Energy" {
                // First try to get "Beamline Energy"
                if let Some(card) = hdu.header.get_card(key.hdu()) {
                    if let Some(val) = card.value.as_float() {
                        columns.push(Column::new(key.name().into(), &[val]));
                        continue;
                    }
                }

                // Then fall  backto "Beamline Energy Goal" if "Beamline Energy" is not present
                if let Some(card) = hdu.header.get_card("Beamline Energy Goal") {
                    if let Some(val) = card.value.as_float() {
                        columns.push(Column::new(key.name().into(), &[val]));
                        continue;
                    }
                }

                // If neither value is available, use a default
                columns.push(Column::new(key.name().into(), &[0.0]));
                continue;
            }

            // Special handling for Date header (it's a string value, not a float)
            if key.hdu() == "DATE" {
                if let Some(card) = hdu.header.get_card(key.hdu()) {
                    let val = card.value.to_string();
                    columns.push(Column::new(key.name().into(), &[val]));
                    continue;
                }
                // If DATE is not present, use a default empty string
                columns.push(Column::new(key.name().into(), &["".to_string()]));
                continue;
            }

            // For other headers, don't fail if they're missing
            let val = match hdu.header.get_card(key.hdu()) {
                Some(card) => card.value.as_float().unwrap_or(1.0),
                None => 0.0, // Default value for missing headers
            };

            // Use the snake_case name from the enum
            columns.push(Column::new(key.name().into(), &[val]));
        }

        Ok(columns)
    }
}

/// Convert a header name to snake_case without units
// fn to_snake_case(name: &str) -> String {
//     // First, remove any units in square brackets
//     let name_without_units = name.split(" [").next().unwrap_or(name);

//     // Then convert to snake_case
//     let mut result = String::new();
//     let mut previous_was_uppercase = false;

//     for (i, c) in name_without_units.chars().enumerate() {
//         if c.is_uppercase() {
//             if i > 0 && !previous_was_uppercase {
//                 result.push('_');
//             }
//             result.push(c.to_lowercase().next().unwrap());
//             previous_was_uppercase = true;
//         } else if c.is_lowercase() {
//             result.push(c);
//             previous_was_uppercase = false;
//         } else if c.is_whitespace() {
//             result.push('_');
//             previous_was_uppercase = false;
//         } else if c.is_alphanumeric() {
//             result.push(c);
//             previous_was_uppercase = false;
//         }
//     }

//     result.to_lowercase()
// }

pub fn process_file_name(path: std::path::PathBuf) -> Vec<Column> {
    // Extract just the file name without extension
    let file_name = path
        .file_stem()
        .unwrap()
        .to_str()
        .unwrap_or("")
        .split(" ")
        .next()
        .unwrap_or("");

    // Just return the file name directly, without extracting frame numbers or scan IDs
    vec![Column::new("file_name".into(), vec![file_name])]
}

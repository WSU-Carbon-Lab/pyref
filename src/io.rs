use astrors_fork::io::hdus::{
    image::{imagehdu::ImageHDU, ImageData},
    primaryhdu::PrimaryHDU,
};
use ndarray::{ArrayBase, Axis, Dim, IxDynImpl, OwnedRepr};
use polars::export::chrono::NaiveDateTime;
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
        .sort(["Sample Name"], Default::default())
        .sort(["Scan ID"], Default::default())
        .sort(["Frame Number"], Default::default())
        .with_columns(&[
            col("EXPOSURE [s]").round(3).alias("EXPOSURE [s]"),
            col("Higher Order Suppressor [mm]")
                .round(2)
                .alias("Higher Order Suppressor [mm]"),
            col("Horizontal Exit Slit Size [um]")
                .round(1)
                .alias("Horizontal Exit Slit Size [um]"),
            col("Beamline Energy [eV]")
                .round(1)
                .alias("Beamline Energy [eV]"),
        ])
        .with_column(
            col("Beamline Energy [eV]")
                .pow(-1)
                .mul(lit(h * c))
                .alias("Lambda [Å]"),
        );

    let angle_offset = lz
        .clone()
        .filter(col("Sample Theta [deg]").eq(0.0))
        .last()
        .select(&[col("CCD Theta [deg]"), col("Sample Theta [deg]")])
        .with_column(
            as_struct(vec![col("Sample Theta [deg]"), col("CCD Theta [deg]")])
                .map(
                    move |s| {
                        let struc = s.struct_()?;
                        let th_series = struc.field_by_name("Sample Theta [deg]")?;
                        let theta = th_series.f64()?;
                        let ccd_th_series = struc.field_by_name("CCD Theta [deg]")?;
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
                .alias("Theta Offset [deg]"),
        )
        .select(&[col("Theta Offset [deg]")])
        .collect()
        .unwrap()
        .get(0)
        .unwrap()[0]
        .try_extract::<f64>()
        .unwrap();
    // get the row cor
    let lz = lz
        .with_column(lit(angle_offset).alias("Theta Offset [deg]"))
        .with_column(
            as_struct(vec![col("Sample Theta [deg]"), col("Lambda [Å]")])
                .map(
                    move |s| {
                        let struc = s.struct_()?;
                        let th_series = struc.field_by_name("Sample Theta [deg]")?;
                        let theta = th_series.f64()?;
                        let lam_series = struc.field_by_name("Lambda [Å]")?;
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
                .alias("Q [Å⁻¹]"),
        );
    lz.collect().unwrap()
}

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
            Ok(vec![col_from_array("Raw".into(), data.clone()).unwrap()])
        }
        _ => Err(FitsLoaderError::UnsupportedImageData),
    }
}

pub fn process_metadata(
    hdu: &PrimaryHDU,
    keys: &Vec<HeaderValue>,
) -> Result<Vec<Column>, FitsLoaderError> {
    if keys.is_empty() {
        Ok(hdu
            .header
            .iter()
            .map(|card| {
                let name = card.keyword.as_str();
                let value = card.value.as_float().unwrap_or(0.0);
                Column::new(name.into(), &[value])
            })
            .collect())
    } else {
        keys.iter()
            .map(|key| {
                let val = hdu
                    .header
                    .get_card(key.hdu())
                    .ok_or_else(|| FitsLoaderError::MissingHeaderKey(key.hdu().to_string()))?
                    .value
                    .as_float()
                    .ok_or_else(|| {
                        FitsLoaderError::Other(format!("Value for {} is not a float", key.hdu()))
                    })?;
                Ok(Column::new(key.name().into(), &[val]))
            })
            .collect()
    }
}

pub fn process_file_name(path: std::path::PathBuf) -> Vec<Column> {
    let ts = path
        .metadata()
        .unwrap()
        .modified()
        .unwrap()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let ts = NaiveDateTime::from_timestamp(ts as i64, 0);

    let file_name = path.file_stem().unwrap().to_str().unwrap_or("");

    let filename_segments = file_name.split('-');

    let frame = filename_segments
        .clone()
        .last()
        .unwrap()
        .parse::<u32>()
        .unwrap_or(0);

    let remaining = filename_segments
        .clone()
        .take(filename_segments.clone().count() - 1)
        .collect::<String>();

    if remaining.is_empty() {
        return vec![
            Column::new("Scan ID".into(), vec![0]),
            Column::new("Sample Name".into(), vec![""]),
            Column::new("Frame Number".into(), vec![0]),
            Column::new("Timestamp".into(), vec![ts]),
        ];
    }

    let scan_id = remaining
        .chars()
        .rev()
        .take(5)
        .collect::<String>()
        .chars()
        .rev()
        .collect::<String>();

    let sample_name = remaining
        .chars()
        .take(remaining.len() - 5)
        .collect::<String>()
        .trim_end_matches(|c| c == '-' || c == '_')
        .to_string();

    vec![
        Column::new("Scan ID".into(), vec![scan_id]),
        Column::new("Sample Name".into(), vec![sample_name]),
        Column::new("Frame Number".into(), vec![frame]),
        Column::new("Timestamp".into(), vec![ts]),
    ]
}

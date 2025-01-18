/// This module provides functionality for working with FITS files using the `astrors::fits` crate.
///
/// # Examples
///
/// ```
/// use astrors::fits;
///
/// // Load a FITS file
/// let fits_file = fits::load("path/to/file.fits");
///
/// // Access the header information
/// let header = fits_file.header();
///
/// // Access the data
/// let data = fits_file.data();
/// ```
///
/// For more information, see the [README](README.md).
use astrors_fork::fits;
use astrors_fork::io;
use astrors_fork::io::hdulist::HDU;
use astrors_fork::io::hdus::image::imagehdu::ImageHDU;
use astrors_fork::io::hdus::primaryhdu::PrimaryHDU;
use polars::{lazy::prelude::*, prelude::*};
use rayon::prelude::*;
use std::fs;
use std::ops::Mul;

// #[global_allocator]
// static GLOBAL: Jemalloc = Jemalloc;

// Enum representing different types of experiments.
pub enum ExperimentType {
    Xrr,
    Xrs,
    Other,
}

impl ExperimentType {
    pub fn from_str(exp_type: &str) -> Result<Self, &str> {
        match exp_type.to_lowercase().as_str() {
            "xrr" => Ok(ExperimentType::Xrr),
            "xrs" => Ok(ExperimentType::Xrs),
            "other" => Ok(ExperimentType::Other),
            _ => Err("Invalid experiment type"),
        }
    }

    pub fn get_keys(&self) -> Vec<HeaderValue> {
        match self {
            ExperimentType::Xrr => vec![
                HeaderValue::SampleTheta,
                HeaderValue::CCDTheta,
                HeaderValue::BeamlineEnergy,
                HeaderValue::BeamCurrent,
                HeaderValue::EPUPolarization,
                HeaderValue::HorizontalExitSlitSize,
                HeaderValue::HigherOrderSuppressor,
                HeaderValue::Exposure,
            ],
            ExperimentType::Xrs => vec![HeaderValue::BeamlineEnergy],
            ExperimentType::Other => vec![],
        }
    }

    pub fn names(&self) -> Vec<&str> {
        match self {
            ExperimentType::Xrr => vec![
                "Sample Theta",
                "CCD Theta",
                "Beamline Energy",
                "Beam Current",
                "EPU Polarization",
                "Horizontal Exit Slit Size",
                "Higher Order Suppressor",
                "EXPOSURE",
            ],
            ExperimentType::Xrs => vec!["Beamline Energy"],
            ExperimentType::Other => vec![],
        }
    }
}

pub enum HeaderValue {
    SampleTheta,
    CCDTheta,
    BeamlineEnergy,
    EPUPolarization,
    BeamCurrent,
    HorizontalExitSlitSize,
    HigherOrderSuppressor,
    Exposure,
}

impl HeaderValue {
    pub fn unit(&self) -> &str {
        match self {
            HeaderValue::SampleTheta => "[deg]",
            HeaderValue::CCDTheta => "[deg]",
            HeaderValue::BeamlineEnergy => "[eV]",
            HeaderValue::BeamCurrent => "[mA]",
            HeaderValue::EPUPolarization => "[deg]",
            HeaderValue::HorizontalExitSlitSize => "[um]",
            HeaderValue::HigherOrderSuppressor => "[mm]",
            HeaderValue::Exposure => "[s]",
        }
    }
    pub fn hdu(&self) -> &str {
        match self {
            HeaderValue::SampleTheta => "Sample Theta",
            HeaderValue::CCDTheta => "CCD Theta",
            HeaderValue::BeamlineEnergy => "Beamline Energy",
            HeaderValue::BeamCurrent => "Beam Current",
            HeaderValue::EPUPolarization => "EPU Polarization",
            HeaderValue::HorizontalExitSlitSize => "Horizontal Exit Slit Size",
            HeaderValue::HigherOrderSuppressor => "Higher Order Suppressor",
            HeaderValue::Exposure => "EXPOSURE",
        }
    }
    pub fn name(&self) -> &str {
        // match and return the string "hdu" + "unit"
        match self {
            HeaderValue::SampleTheta => "Sample Theta [deg]",
            HeaderValue::CCDTheta => "CCD Theta [deg]",
            HeaderValue::BeamlineEnergy => "Beamline Energy [eV]",
            HeaderValue::BeamCurrent => "Beam Current [mA]",
            HeaderValue::EPUPolarization => "EPU Polarization [deg]",
            HeaderValue::HorizontalExitSlitSize => "Horizontal Exit Slit Size [um]",
            HeaderValue::HigherOrderSuppressor => "Higher Order Suppressor [mm]",
            HeaderValue::Exposure => "EXPOSURE [s]",
        }
    }
    pub fn round(&self, value: f64) -> f64 {
        match self {
            HeaderValue::Exposure => (value * 1000.0).round() / 1000.0,
            HeaderValue::HigherOrderSuppressor => (value * 100.0).round() / 100.0,
            HeaderValue::HorizontalExitSlitSize => (value * 10.0).round() / 10.0,
            _ => value,
        }
    }
}

// Function facilitate storing the image data as a single element in a Polars DataFrame.
fn vec_i64(name: &str, img: Vec<i64>) -> Column {
    let new_series = [img.iter().collect::<Series>()];
    Column::new(name.into(), new_series)
}

fn vec_u32(name: &str, img: Vec<u32>) -> Column {
    let new_series = [img.iter().collect::<Series>()];
    Column::new(name.into(), new_series)
}

// Post process dataframe
pub fn post_process(lzf: LazyFrame) -> LazyFrame {
    let h = physical_constants::PLANCK_CONSTANT_IN_EV_PER_HZ;
    let c = physical_constants::SPEED_OF_LIGHT_IN_VACUUM * 1e10;
    // Calculate lambda and q values in angstrom
    let lz = lzf
        .sort(["Sample Name"], Default::default())
        .sort(["Scan ID"], Default::default())
        .sort(["Frame Number"], Default::default())
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
    lz
}

pub fn process_image(img: &ImageHDU) -> Result<Vec<Column>, PolarsError> {
    let bzero = img
        .header
        .get_card("BZERO")
        .unwrap()
        .value
        .as_int()
        .unwrap();

    match &img.data {
        io::hdus::image::ImageData::I16(image) => {
            let flat_data: Vec<i64> = image.iter().map(|&x| i64::from(x as i64 + bzero)).collect();
            let shape = image.dim();
            Ok(vec![
                vec_i64("Raw", flat_data),
                vec_u32("Raw Shape", vec![shape[0] as u32, shape[1] as u32]),
            ])
        }
        _ => Err(PolarsError::NoData("Unsupported image data type".into())),
    }
}

pub fn process_metadata(hdu: &PrimaryHDU, keys: &Vec<HeaderValue>) -> Vec<Column> {
    if keys.is_empty() {
        hdu.header
            .iter()
            .map(|card| {
                let name = card.keyword.as_str();
                let value = card.value.as_float().unwrap_or(0.0);
                Column::new(name.into(), vec![value])
            })
            .collect::<Vec<_>>()
    } else {
        keys.iter()
            .filter_map(|key| {
                let val = hdu.header.get_card(key.hdu()).unwrap();
                Some(Column::new(
                    key.name().into(),
                    vec![key.round(val.value.as_float().unwrap_or(0.0))],
                ))
            })
            .collect::<Vec<_>>()
    }
}

pub fn process_file_name(path: &str) -> Vec<Column> {
    let file_name = path.rsplit('/').next().unwrap_or("Unknown");
    let name_spit = file_name.split('-');

    let frame = name_spit
        .clone()
        .last()
        .and_then(|scan_id| scan_id.split('.').next())
        .and_then(|scan_id| scan_id.trim_start_matches('0').parse::<i32>().ok())
        .unwrap_or(0);

    let remaining = name_spit
        .clone()
        .next()
        .and_then(|name| name.starts_with("Captured Image").then(|| &name[14..]))
        .unwrap_or("");

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
        .collect::<String>();

    vec![
        Column::new("Frame Number".into(), vec![frame]),
        Column::new("Sample Name".into(), vec![sample_name]),
        Column::new("Scan ID".into(), vec![scan_id]),
    ]
}

// workhorse functions for loading and processing CCD data.
pub fn read_fits(
    file_path: &str,
    header_items: Vec<HeaderValue>,
) -> Result<DataFrame, PolarsError> {
    let hdul = fits::fromfile(file_path)?;

    let meta = match hdul.hdus.get(0).clone().unwrap() {
        HDU::Primary(hdu) => process_metadata(hdu.clone(), &header_items),
        _ => return Err(PolarsError::NoData("Primary HDU not found".into())),
    };

    let img_data = match hdul.hdus.get(2).clone().unwrap() {
        HDU::Image(hdu) => process_image(hdu)?,
        _ => return Err(PolarsError::NoData("Image HDU not found".into())),
    };

    let names = process_file_name(file_path);

    let mut s_vec = meta;
    s_vec.extend(img_data);
    s_vec.extend(names);
    Ok(DataFrame::new(s_vec).unwrap())
}

pub fn read_experiment(dir: &str, exp_type: &str) -> LazyFrame {
    let exp = ExperimentType::from_str(exp_type).unwrap();
    let header_items = exp.names(); // Clone the header_items vector
                                    // iterate over all files in the directory
    let combined = DataFrame::empty(); // Remove the mut keyword

    let _ = fs::read_dir(dir)
        .unwrap()
        .into_iter()
        .par_bridge()
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().extension().and_then(|ext| ext.to_str()) == Some("fits"))
        .map(|entry| {
            combined.vstack(
                &read_fits(
                    entry
                        .path()
                        .to_str()
                        .expect("Failed to convert path to string"),
                    header_items, // Clone the header_items vector
                )
                .unwrap(),
            )
        });

    post_process(combined.lazy())
}

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

pub fn load() {
    let test_path = "/home/hduva/projects/pyref-ccd/test/";

    let data = read_experiment(test_path, "xrr").unwrap();
    println!("{:?}", data);
}

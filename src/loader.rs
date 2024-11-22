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
use astrors_fork::io::hdulist::*;
use astrors_fork::io::header::*;
use numpy::ndarray::Array2;
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
                HeaderValue::BeamlineEnergy,
                HeaderValue::EPUPolarization,
                HeaderValue::HorizontalExitSlitSize,
                HeaderValue::HigherOrderSuppressor,
                HeaderValue::Exposure,
            ],
            ExperimentType::Xrs => vec![HeaderValue::BeamlineEnergy],
            ExperimentType::Other => vec![],
        }
    }
}

pub enum HeaderValue {
    SampleTheta,
    BeamlineEnergy,
    EPUPolarization,
    HorizontalExitSlitSize,
    HigherOrderSuppressor,
    Exposure,
}

impl HeaderValue {
    pub fn unit(&self) -> &str {
        match self {
            HeaderValue::SampleTheta => "[deg]",
            HeaderValue::BeamlineEnergy => "[eV]",
            HeaderValue::EPUPolarization => "[deg",
            HeaderValue::HorizontalExitSlitSize => "[um]",
            HeaderValue::HigherOrderSuppressor => "[mm]",
            HeaderValue::Exposure => "[s]",
        }
    }
    pub fn hdu(&self) -> &str {
        match self {
            HeaderValue::SampleTheta => "Sample Theta",
            HeaderValue::BeamlineEnergy => "Beamline Energy",
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
            HeaderValue::BeamlineEnergy => "Beamline Energy [eV]",
            HeaderValue::EPUPolarization => "EPU Polarization [deg]",
            HeaderValue::HorizontalExitSlitSize => "Horizontal Exit Slit Size [um]",
            HeaderValue::HigherOrderSuppressor => "Higher Order Suppressor [mm]",
            HeaderValue::Exposure => "EXPOSURE [s]",
        }
    }
}

// Struct representing a CCD FITS file.
pub struct FitsLoader {
    pub path: String,
    pub hdul: HDUList,
}

/// FitsLoader struct for loading and accessing FITS file data.
///
/// The `FitsLoader` struct provides methods for loading and accessing data from a FITS file.
/// It supports retrieving individual card values, all card values, image data, and converting
/// the data to a Polars DataFrame.
///
/// # Example
///
/// ```
/// extern crate pyref_core;
/// use pyref_core::loader::FitsLoader;
///
/// let fits_loader = FitsLoader::new("/path/to/file.fits").unwrap();
///
/// // Get a specific card value
/// let card_value = fits_loader.get_value("CARD_NAME");
///
/// // Get all card values
/// let all_cards = fits_loader.get_all_cards();
///
/// // Get image data
/// let image_data = fits_loader.get_image();
///
/// // Convert data to Polars DataFrame
/// let keys = ["KEY1", "KEY2"];
/// let polars_df = fits_loader.to_polars(&keys);
/// ```
/// A struct representing a FITS loader.

impl FitsLoader {
    /// Creates a new `FitsLoader` instance.
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the FITS file.
    ///
    /// # Returns
    ///
    /// A `Result` containing the `FitsLoader` instance if successful, or a boxed `dyn std::error::Error` if an error occurred.
    pub fn new(path: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let hdul = fits::fromfile(path)?;
        Ok(FitsLoader {
            path: path.to_string(),
            hdul,
        })
    }

    /// Retrieves a specific card from the FITS file.
    ///
    /// # Arguments
    ///
    /// * `card_name` - The name of the card to retrieve.
    ///
    /// # Returns
    ///
    /// An `Option` containing the requested `card::Card` if found, or `None` if not found.
    pub fn get_card(&self, card_name: &str) -> Option<card::Card> {
        match &self.hdul.hdus[0] {
            io::hdulist::HDU::Primary(hdu) => hdu.header.get_card(card_name).cloned(),
            _ => None,
        }
    }

    /// Retrieves the value of a specific card from the FITS file.
    ///
    /// # Arguments
    ///
    /// * `card_name` - The name of the card to retrieve the value from.
    ///
    /// # Returns
    ///
    /// An `Option` containing the value of the requested card as a `f64` if found, or `None` if not found.
    pub fn get_value(&self, card_name: &str) -> Option<f64> {
        if card_name == "EXPOSURE"
            || card_name == "Sample Theta"
            || "Higher Order Suppressor" == card_name
        {
            return match &self.hdul.hdus[0] {
                io::hdulist::HDU::Primary(hdu) => hdu
                    .header
                    .get_card(card_name)
                    .map(|c| (c.value.as_float().unwrap() * 1000.0).round() / 1000.0),
                _ => None,
            };
        }
        match &self.hdul.hdus[0] {
            io::hdulist::HDU::Primary(hdu) => hdu
                .header
                .get_card(card_name)
                .map(|c| (c.value.as_float().unwrap() * 10.0).round() / 10.0),
            _ => None,
        }
    }

    /// Retrieves all cards from the FITS file.
    ///
    /// # Returns
    ///
    /// A `Vec` containing all the cards as `card::Card` instances.
    pub fn get_all_cards(&self) -> Vec<card::Card> {
        match &self.hdul.hdus[0] {
            io::hdulist::HDU::Primary(hdu) => {
                hdu.header.iter().cloned().collect::<Vec<card::Card>>()
            }
            _ => vec![],
        }
    }

    /// Retrieves the image data from the FITS file.
    ///
    /// # Arguments
    ///
    /// * `data` - The image data to retrieve.
    ///
    /// # Returns
    ///
    /// A `Result` containing the image data as a `Array2<u32>` if successful, or a boxed `dyn std::error::Error` if an error occurred.
    fn get_data(
        &self,
        data: &io::hdus::image::ImageData,
    ) -> Result<(Vec<u32>, Vec<u32>), Box<dyn std::error::Error + Send + Sync>> {
        let (flat_data, shape) = match data {
            io::hdus::image::ImageData::I16(image) => {
                let flat_data = image.iter().map(|&x| u32::from(x as u16)).collect();
                let shape = image.dim();
                (flat_data, shape)
            }
            _ => return Err("Unsupported image data type".into()),
        };
        Ok((flat_data, vec![shape[0] as u32, shape[1] as u32]))
    }

    /// Retrieves the image data from the FITS file as an `Array2<u32>`.
    ///
    /// # Returns
    ///
    /// A `Result` containing the image data as a `Array2<u32>` if successful, or a boxed `dyn std::error::Error` if an error occurred.
    pub fn get_image(
        &self,
    ) -> Result<(Vec<u32>, Vec<u32>), Box<dyn std::error::Error + Send + Sync>> {
        match &self.hdul.hdus[2] {
            io::hdulist::HDU::Image(i_hdu) => self.get_data(&i_hdu.data),
            _ => Err("Image HDU not found".into()),
        }
    }

    /// Converts the FITS file data to a `polars::prelude::DataFrame`.
    ///
    /// # Arguments
    ///
    /// * `keys` - The keys of the cards to include in the DataFrame. If empty, all cards will be included.
    ///
    /// # Returns
    ///
    /// A `Result` containing the converted `DataFrame` if successful, or a boxed `dyn std::error::Error` if an error occurred.
    pub fn to_polars(
        &self,
        keys: &Vec<HeaderValue>,
    ) -> Result<DataFrame, Box<dyn std::error::Error + Send + Sync>> {
        let mut s_vec = if keys.is_empty() {
            // When keys are empty, use all cards.
            self.get_all_cards()
                .iter()
                .map(|card| {
                    let name = card.keyword.as_str();
                    let value = card.value.as_float().unwrap_or(0.0);
                    Series::new(name.into(), vec![value])
                })
                .collect::<Vec<_>>()
        } else {
            // Use specified keys
            keys.iter()
                .filter_map(|key| {
                    self.get_value(key.hdu())
                        .map(|value| Series::new(PlSmallStr::from_str(key.name()), vec![value]))
                })
                .collect::<Vec<_>>()
        };
        // Add the image data
        let (image, size) = match self.get_image() {
            Ok(data) => data,
            Err(e) => return Err(e),
        };
        let scan_id = self
            .path
            .split("/")
            .last()
            .unwrap()
            .split("-")
            .last()
            .unwrap();
        let scan_id = scan_id.trim_start_matches('0');
        let scan_id = scan_id.parse::<i32>().unwrap();

        s_vec.push(Series::new("Scan ID".into(), vec![scan_id]));
        s_vec.push(vec_series("Raw", image));
        s_vec.push(vec_series("Raw Shape", size));
        DataFrame::new(s_vec).map_err(From::from)
    }
}
// Function facilitate storing the image data as a single element in a Polars DataFrame.
pub fn vec_series(name: &str, img: Vec<u32>) -> Series {
    let new_series = [img.iter().collect::<Series>()];
    Series::new(name.into(), new_series)
}

pub struct ExperimentLoader {
    pub dir: String,
    pub ccd_files: Vec<FitsLoader>,
    pub experiment_type: ExperimentType,
}

/// FitsLoader struct for loading and accessing FITS file data.
///
/// The `FitsLoader` struct provides methods for loading and accessing data from a FITS file.
/// It supports retrieving individual card values, all card values, image data, and converting
/// the data to a Polars DataFrame.
///
/// # Example
///
/// ```
/// extern crate pyref_core;
/// use pyref_core::loader::{ExperimentLoader, ExperimentType};
///
/// let exp = ExperimentType::from_str(exp_type)?;
/// let fits_loader = ExperimentLoader::new("/path/to/file.fits", exp).unwrap();
///
/// // Mostly this is used to convert the data to a Polars DataFrame
/// let df = fits_loader.to_polars()?;
/// ```

impl ExperimentLoader {
    // Create a new ExperimentLoader instance and load all Fits file in the directory.
    pub fn new(
        dir: &str,
        experiment_type: ExperimentType,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let ccd_files: Vec<_> = fs::read_dir(dir)?
            .filter_map(Result::ok)
            .filter(|entry| entry.path().extension().and_then(|ext| ext.to_str()) == Some("fits"))
            .collect();

        let ccd_files = ccd_files
            .par_iter() // Parallel iterator using Rayon
            .map(|entry| FitsLoader::new(entry.path().to_str().unwrap()))
            .collect::<Result<Vec<_>, Box<dyn std::error::Error + Send + Sync>>>();
        let ccd_files = match ccd_files {
            Ok(ccd_files) => ccd_files,
            Err(e) => return Err(e),
        };

        Ok(ExperimentLoader {
            dir: dir.to_string(),
            ccd_files,
            experiment_type,
        })
    }
    // Package all loaded FITS files into a single Polars DataFrame.
    pub fn to_polars(&self) -> Result<DataFrame, Box<dyn std::error::Error>> {
        let keys = self.experiment_type.get_keys();

        let dfs = self
            .ccd_files
            .par_iter()
            .map(|ccd| ccd.to_polars(&keys))
            .collect::<Result<Vec<_>, _>>();
        let mut dfs = match dfs {
            Ok(dfs) => dfs,
            Err(e) => return Err(e),
        };
        let mut df = dfs.pop().ok_or("No data found")?;
        for mut d in dfs {
            df.vstack_mut(&mut d)?;
        }
        Ok(post_process(df))
    }
}

// Post process dataframe
pub fn post_process(df: DataFrame) -> DataFrame {
    let h = physical_constants::PLANCK_CONSTANT_IN_EV_PER_HZ;
    let c = physical_constants::SPEED_OF_LIGHT_IN_VACUUM * 1e10;
    // Calculate lambda and q values in angstrom
    let lz = df
        .clone()
        .lazy()
        .sort(["Beamline Energy [eV]", "Scan ID"], Default::default())
        .with_column(
            col("Beamline Energy [eV]")
                .pow(-1)
                .mul(lit(h * c))
                .alias("Lambda [Å]"),
        )
        .with_column(
            as_struct(vec![col("Sample Theta [deg]"), col("Lambda [Å]")])
                .map(
                    |s| {
                        let struc = s.struct_()?;
                        let th_series = struc.field_by_name("Sample Theta [deg]")?;
                        let theta = th_series.f64()?;
                        let lam_series = struc.field_by_name("Lambda [Å]")?;
                        let lam = lam_series.f64()?;

                        let out: Float64Chunked = theta
                            .into_iter()
                            .zip(lam.iter())
                            .map(|(theta, lam)| match (theta, lam) {
                                (Some(theta), Some(lam)) => Some(q(lam, theta)),
                                _ => None,
                            })
                            .collect();

                        Ok(Some(out.into_series()))
                    },
                    GetOutput::from_type(DataType::Float64),
                )
                .alias("Q [Å⁻¹]"),
        );
    lz.collect().unwrap()
}

// function to unpack an image wile iterating rhough a polars dataframe.
pub fn get_image(image_data: &[u32], shape: (usize, usize)) -> Result<Array2<u32>, PolarsError> {
    let image_array = Array2::from_shape_vec(shape, image_data.to_vec())
        .map_err(|_| PolarsError::ComputeError("Invalid image data".into()))?;
    Ok(image_array)
}

// workhorse functions for loading and processing CCD data.
pub fn read_fits(file_path: &str) -> Result<DataFrame, Box<dyn std::error::Error>> {
    let loader = match FitsLoader::new(file_path) {
        Ok(loader) => loader,
        Err(e) => return Err(e),
    };
    let df = match loader.to_polars(&vec![]) {
        Ok(df) => df,
        Err(e) => return Err(e),
    };
    Ok(df)
}

pub fn read_experiment(dir: &str, exp_type: &str) -> Result<DataFrame, Box<dyn std::error::Error>> {
    let exp = ExperimentType::from_str(exp_type)?;
    let df = ExperimentLoader::new(dir, exp)?.to_polars()?;
    Ok(df)
}

pub fn simple_update(df: &mut DataFrame, dir: &str) -> Result<(), Box<dyn std::error::Error>> {
    let ccd_files: Vec<_> = fs::read_dir(dir)?
        .filter_map(Result::ok)
        .filter(|entry| entry.path().extension().and_then(|ext| ext.to_str()) == Some("fits"))
        .collect();
    let not_loaded = ccd_files.len() as isize - df.height() as isize;
    if not_loaded == 0 {
        return Ok(());
    } else if not_loaded < 0 {
        return Err("Files out of sync with loaded data, Restart".into());
    }
    let ccd_files = ccd_files[..not_loaded as usize]
        .par_iter() // Parallel iterator using Rayon
        .map(|entry| FitsLoader::new(entry.path().to_str().unwrap()))
        .collect::<Result<Vec<_>, Box<dyn std::error::Error + Send + Sync>>>();
    let ccd_files = match ccd_files {
        Ok(ccd_files) => ccd_files,
        Err(e) => return Err(e),
    };
    let mut new_df = ExperimentLoader {
        dir: dir.to_string(),
        ccd_files,
        experiment_type: ExperimentType::Xrr,
    }
    .to_polars()?;
    df.vstack_mut(&mut new_df)?;
    Ok(())
}

fn q(lam: f64, theta: f64) -> f64 {
    4.0 * std::f64::consts::PI * theta.to_radians().sin() / lam
}

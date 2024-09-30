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
use astrors::fits;
use astrors::io;
use astrors::io::hdulist::*;
use astrors::io::header::*;
use ndarray::{Array2, ArrayD, Axis, Ix2};
use physical_constants;
use polars::prelude::*;
use std::fs;

// Enum representing different types of experiments.
pub enum ExperimentType {
    Xrr,
    Xrs,
    Other,
}

impl ExperimentType {
    pub fn from_str(exp_type: &str) -> Result<Self, &str> {
        match exp_type {
            "Xrr" => Ok(ExperimentType::Xrr),
            "Xrs" => Ok(ExperimentType::Xrs),
            "Other" => Ok(ExperimentType::Other),
            _ => Err("Invalid experiment type"),
        }
    }

    pub fn get_keys(&self) -> Vec<&str> {
        match self {
            ExperimentType::Xrr => vec![
                "Sample Theta",
                "Beamline Energy",
                "EPU Polarization",
                "Horizontal Exit Slit Size",
                "Higher Order Suppressor",
                "EXPOSURE",
            ],
            ExperimentType::Xrs => vec!["Energy"],
            ExperimentType::Other => vec![],
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
    pub fn new(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
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
        if card_name == "Q" {
            let theta = self.get_value("Sample Theta");
            let en = self.get_value("Beamline Energy");
            // calculate the q value from the sample theta and beamline energy
            let lambda = 1e10
                * physical_constants::MOLAR_PLANCK_CONSTANT
                * physical_constants::SPEED_OF_LIGHT_IN_VACUUM
                / en.unwrap();
            return Some(4.0 * std::f64::consts::PI * (theta.unwrap().to_radians().sin() / lambda));
        }
        match &self.hdul.hdus[0] {
            io::hdulist::HDU::Primary(hdu) => hdu
                .header
                .get_card(card_name)
                .map(|c| c.value.as_float().unwrap()),
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
    ) -> Result<Array2<u32>, Box<dyn std::error::Error>> {
        let array_data = match data {
            io::hdus::image::ImageData::I16(image) => ArrayD::from_shape_vec(
                image.raw_dim(),
                image.iter().map(|&x| u32::from(x as u16)).collect(),
            )?,
            _ => return Err("Unsupported image data type".into()),
        };
        Ok(self.ensure_2d(array_data))
    }

    /// Ensures the data is two-dimensional.
    ///
    /// # Arguments
    ///
    /// * `data` - The data to ensure is two-dimensional.
    ///
    /// # Returns
    ///
    /// The data as a `Array2<T>`.
    ///
    /// # Panics
    ///
    /// Panics if the data is not two-dimensional.
    fn ensure_2d<T>(&self, data: ArrayD<T>) -> Array2<T>
    where
        T: Clone + Default,
    {
        data.into_dimensionality::<Ix2>()
            .unwrap_or_else(|_| panic!("Expected 2D data but got different dimensions"))
    }

    /// Retrieves the image data from the FITS file as an `Array2<u32>`.
    ///
    /// # Returns
    ///
    /// A `Result` containing the image data as a `Array2<u32>` if successful, or a boxed `dyn std::error::Error` if an error occurred.
    pub fn get_image(&self) -> Result<Array2<u32>, Box<dyn std::error::Error>> {
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
    pub fn to_polars(&self, keys: &[&str]) -> Result<DataFrame, Box<dyn std::error::Error>> {
        let mut s_vec = if keys.is_empty() {
            // When keys are empty, use all cards.
            self.get_all_cards()
                .iter()
                .map(|card| {
                    let name = card.keyword.clone();
                    let value = card.value.as_float().unwrap_or(0.0);
                    Series::new(&name, vec![value])
                })
                .collect::<Vec<_>>()
        } else {
            // Use specified keys
            keys.iter()
                .filter_map(|key| {
                    self.get_value(key)
                        .map(|value| Series::new(key, vec![value]))
                })
                .collect::<Vec<_>>()
        };
        // Add the image data
        let image = self.get_image()?;
        s_vec.push(image_series("Image", image));
        s_vec.push(Series::new("Q", vec![self.get_value("Q").unwrap()]));
        DataFrame::new(s_vec).map_err(From::from)
    }
}
// Function facilitate storing the image data as a single element in a Polars DataFrame.
pub fn image_series(name: &str, array: Array2<u32>) -> Series {
    let mut s = Series::new_empty(
        name,
        &DataType::List(Box::new(DataType::List(Box::new(DataType::UInt32)))),
    );

    let mut chunked_builder = ListPrimitiveChunkedBuilder::<UInt32Type>::new(
        "",
        array.len_of(Axis(0)),
        array.len_of(Axis(1)) * array.len_of(Axis(0)),
        DataType::UInt32,
    );
    for row in array.axis_iter(Axis(0)) {
        chunked_builder.append_slice(row.as_slice().unwrap_or(&row.to_vec()));
    }
    let new_series = chunked_builder
        .finish()
        .into_series()
        .implode()
        .unwrap()
        .into_series();
    let _ = s.extend(&new_series);
    s
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
        let ccd_files = fs::read_dir(dir)?
            .filter_map(Result::ok)
            .filter(|entry| entry.path().extension().and_then(|ext| ext.to_str()) == Some("fits"))
            .map(|entry| FitsLoader::new(entry.path().to_str().unwrap()))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(ExperimentLoader {
            dir: dir.to_string(),
            ccd_files,
            experiment_type,
        })
    }
    // Package all loaded FITS files into a single Polars DataFrame.
    pub fn to_polars(&self) -> Result<DataFrame, Box<dyn std::error::Error>> {
        let keys = self.experiment_type.get_keys();

        let mut dfs = self
            .ccd_files
            .iter()
            .map(|ccd| ccd.to_polars(&keys))
            .collect::<Result<Vec<_>, _>>()?;

        let mut df = dfs.pop().ok_or("No data found")?;
        for mut d in dfs {
            df.vstack_mut(&mut d)?;
        }
        Ok(df)
    }
}

// workhorse functions for loading and processing CCD data.
pub fn read_fits(file_path: &str) -> Result<DataFrame, Box<dyn std::error::Error>> {
    let df = FitsLoader::new(file_path)?.to_polars(&[])?;
    Ok(df)
}

pub fn read_experiment(dir: &str, exp_type: &str) -> Result<DataFrame, Box<dyn std::error::Error>> {
    let exp = ExperimentType::from_str(exp_type)?;
    let df = ExperimentLoader::new(dir, exp)?.to_polars()?;
    Ok(df)
}

pub fn img_to_series(name: &str, array: Array2<u32>) -> Series {
    let mut s = Series::new_empty(name, &DataType::List(Box::new(DataType::UInt32)));
    let flat = array.iter().copied().collect::<Vec<_>>();
    let mut chunked_builder = ListPrimitiveChunkedBuilder::<UInt32Type>::new(
        "",
        array.shape().iter().product::<usize>(),
        array.shape().iter().product::<usize>(),
        DataType::UInt32,
    );
    chunked_builder.append_slice(flat.as_slice());
    let new_series = chunked_builder.finish().into_series();
    let _ = s.extend(&new_series);
    s
}

pub fn get_image(df: &DataFrame, i: &usize, img: &str, shape: usize) -> Array2<u32> {
    let data = match df[img].get(*i).unwrap_or(AnyValue::Null) {
        AnyValue::List(s) => s,
        _ => panic!("Expected list type"),
    };
    to_array(data, shape)
}

fn to_array(data: Series, shape: usize) -> Array2<u32> {
    let dim = (shape, shape);
    let listed = data
        .iter()
        .map(|x| match x {
            AnyValue::UInt32(x) => x,
            _ => panic!("Expected u32 type"),
        })
        .collect::<Vec<_>>();
    Array2::from_shape_vec(dim, listed).unwrap_or(Array2::zeros((0, 0)))
}

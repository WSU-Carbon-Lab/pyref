use astrors::fits;
use astrors::io;
use astrors::io::hdulist::*;
use astrors::io::header::*;
use ndarray::{Array2, ArrayD, Axis, Ix2};
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

impl FitsLoader {
    pub fn new(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let hdul = fits::fromfile(path)?;
        Ok(FitsLoader {
            path: path.to_string(),
            hdul,
        })
    }
    // Get single card values
    pub fn get_card(&self, card_name: &str) -> Option<card::CardValue> {
        match &self.hdul.hdus[0] {
            io::hdulist::HDU::Primary(hdu) => {
                hdu.header.get_card(card_name).map(|c| c.value.clone())
            }
            _ => None,
        }
    }
    // Get all card values
    pub fn get_all_cards(&self) -> Vec<card::Card> {
        match &self.hdul.hdus[0] {
            io::hdulist::HDU::Primary(hdu) => {
                hdu.header.iter().cloned().collect::<Vec<card::Card>>()
            }
            _ => vec![],
        }
    }
    // Get image data
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
    fn ensure_2d<T>(&self, data: ArrayD<T>) -> Array2<T>
    where
        T: Clone + Default,
    {
        data.into_dimensionality::<Ix2>()
            .unwrap_or_else(|_| panic!("Expected 2D data but got different dimensions"))
    }

    /// Retrieves the image data from the FITS file as an Array2<u32>.
    pub fn get_image(&self) -> Result<Array2<u32>, Box<dyn std::error::Error>> {
        match &self.hdul.hdus[2] {
            io::hdulist::HDU::Image(i_hdu) => self.get_data(&i_hdu.data),
            _ => Err("Image HDU not found".into()),
        }
    }

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
                    self.get_card(key)
                        .map(|card| Series::new(key, vec![card.as_float().unwrap_or(0.0)]))
                })
                .collect::<Vec<_>>()
        };
        // Add the image data
        let image = self.get_image()?;
        s_vec.push(image_series("Image", image));

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

// Structure representing an experiment.
pub struct ExperimentLoader {
    pub dir: String,
    pub ccd_files: Vec<FitsLoader>,
    pub experiment_type: ExperimentType,
}

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

use astrors::fits;
use astrors::io;
use astrors::io::hdulist::*;
use astrors::io::header::*;
use ndarray::{Array2, ArrayD, Axis, Ix2};
use polars::prelude::*;

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
        match row.as_slice() {
            Some(row) => chunked_builder.append_slice(row),
            None => chunked_builder.append_slice(&row.to_vec()),
        }
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

pub fn extend_image(series: &mut Series, name: &str, array: Array2<u32>) {
    let s = image_series(name, array);
    let _ = series.extend(&s);
}

/// Struct representing a CCD FITS file.
pub struct CcdFits {
    pub path: String,
    pub hdul: HDUList,
}

impl CcdFits {
    /// Creates a new CcdFits instance from the given path.
    /// Returns a Result to handle potential errors gracefully.
    pub fn new(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let hdul = fits::fromfile(path)?;
        Ok(CcdFits {
            path: path.to_string(),
            hdul,
        })
    }

    /// Retrieves the card value for the given card name.
    /// Returns an Option to handle the absence of a card.
    pub fn get_card(&self, card_name: &str) -> Option<card::CardValue> {
        match &self.hdul.hdus[0] {
            io::hdulist::HDU::Primary(hdu) => {
                hdu.header.get_card(card_name).map(|c| c.value.clone())
            }
            _ => None,
        }
    }

    pub fn get_all_cards(&self) -> Vec<card::Card> {
        match &self.hdul.hdus[0] {
            io::hdulist::HDU::Primary(hdu) => hdu
                .header
                .iter()
                .map(|c| c.clone())
                .collect::<Vec<card::Card>>(),
            _ => vec![],
        }
    }

    /// Determines the type of image data in the FITS file.

    /// Converts ImageData to Array2<u64>.
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
        match data.into_dimensionality::<Ix2>() {
            Ok(array) => array,
            Err(_) => panic!("Expected 2D data but got different dimensions"),
        }
    }

    /// Retrieves the image data from the FITS file as an Array2<u64>.
    pub fn get_image(&self) -> Result<Array2<u32>, Box<dyn std::error::Error>> {
        match &self.hdul.hdus[2] {
            io::hdulist::HDU::Image(i_hdu) => self.get_data(&i_hdu.data),
            _ => Err("Image HDU not found".into()),
        }
    }

    pub fn to_polars_keys(&self, keys: Vec<&str>) -> Result<DataFrame, Box<dyn std::error::Error>> {
        let mut s_vec = vec![];
        keys.iter().for_each(|key| {
            if let Some(card) = self.get_card(key) {
                s_vec.push(Series::new(key, vec![card.as_float()]));
            }
        });
        // extract the image now
        let image = match self.get_image() {
            Ok(image) => image,
            Err(e) => return Err(e),
        };
        let s = image_series("image", image);
        s_vec.push(s);
        let df = DataFrame::new(s_vec)?;
        Ok(df) // Add return statement to return the DataFrame
    }

    pub fn to_polars(&self) -> Result<DataFrame, Box<dyn std::error::Error>> {
        let cards = self.get_all_cards();
        let s_vec = cards
            .iter()
            .map(|card| {
                let name = &card.keyword;
                let value = card.value.as_float();
                match value {
                    Some(value) => Series::new(name, vec![value]),
                    None => Series::new(name, vec![0.0]),
                }
            })
            .collect::<Vec<Series>>();
        // extract the image now
        let image = match self.get_image() {
            Ok(image) => image,
            Err(e) => return Err(e),
        };
        let s = image_series("image", image);
        let mut s_vec = s_vec;
        s_vec.push(s);
        let df = DataFrame::new(s_vec)?;
        Ok(df) // Add return statement to return the DataFrame
    }
}

pub struct ExperimentLoader {
    pub dir: String,
    pub ccdl: Vec<CcdFits>,
}

impl ExperimentLoader {
    pub fn new(dir: &str) -> Result<Self, Box<dyn std::error::Error>> {
        // Get all the FITS files in the directory
        let fits_files = std::fs::read_dir(dir)?
            .filter_map(|entry| {
                let path = entry.unwrap().path();
                if path.extension()?.to_str()? == "fits" {
                    Some(path.to_str()?.to_string())
                } else {
                    None
                }
            })
            .collect::<Vec<String>>();
        // Load the FITS files
        let ccdl = fits_files
            .iter()
            .map(|fits_file| CcdFits::new(fits_file)?)
            .collect::<Result<Vec<CcdFits>, Box<dyn std::error::Error>>()?
    }

    pub fn to_polars(&self) -> Result<DataFrame, Box<dyn std::error::Error>> {
        let mut dfs = vec![];
        for fits in &self.fits {
            match fits.to_polars() {
                Ok(df) => dfs.push(df),
                Err(e) => eprintln!("Error converting FITS to Polars: {}", e),
            }
        }
        let mut df = dfs.pop().unwrap();
        for mut other_df in dfs {
            df.vstack_mut(&mut other_df)?;
        }
        Ok(df)
    }
}

fn main() {
    let fits_file = CcdFits::new("C:/Users/hduva/.projects/pyref/pyref-ccd/test/test2.fits");

    match fits_file {
        Ok(fits) => match fits.to_polars() {
            Ok(df) => {
                println!("{}", df);
                println!("{:?}", df.select(["Sample Theta", "image"]));
            }
            Err(e) => eprintln!("Error extracting image data: {}", e),
        },
        Err(e) => eprintln!("Error loading FITS file: {}", e),
    }
}

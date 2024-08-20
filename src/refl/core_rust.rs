// /*
// Core Rust functions for indexing the fits files from a data directory and loading their
// contents in a dataframe.
// */
// use astrors::{fits, io::hdus::primaryhdu::PrimaryHDU};
// use ndarray::Array2;
// use polars::prelude::*;
// use pyo3::prelude::*;
// use std::{collections::HashMap, fs::File};

// pub enum MotorPositions {
//     Energy(f64) = "Beamline Energy",
//     SampleTheta(f64) = "Sample Theta",
//     Current(f64) = "Beam Current",
//     HoS(f64) = "Higher Order Suppressor",
//     Pol(Polarization) = "EPU Polarization",
//     Exposure(f64) = "EXPOSURE",
// }

// pub enum Polarization {
//     S(f64) = 100,
//     P(f64) = 190,
// }
// #[pyclass]
// pub struct ImageData {
//     pub image: Array2<f64>,
//     pub direct_beam: (f64, f64),
// }

// #[pyclass]
// pub struct HeaderData {
//     pub energy: MotorPositions::Energy,
//     pub sample_theta: MotorPositions::SampleTheta,
//     pub current: MotorPositions::Current,
//     pub ho_s: MotorPositions::HoS,
//     pub pol: MotorPositions::Pol,
//     pub exposure: MotorPositions::Exposure,
//     pub image_data: ImageData,
// }

// impl HeaderData {
//     pub fn new(
//         energy: f64,
//         sample_theta: f64,
//         current: f64,
//         ho_s: f64,
//         pol: f64,
//         exposure: f64,
//         image: Array2<f64>,
//         beamspot: (u64, u64),
//     ) -> Self {
//         HeaderData {
//             energy: MotorPositions::Energy(energy),
//             sample_theta: MotorPositions::SampleTheta(sample_theta),
//             current: MotorPositions::Current(current),
//             ho_s: MotorPositions::HoS(ho_s),
//             pol: MotorPositions::Pol(pol),
//             exposure: MotorPositions::Exposure(exposure),
//             image_data: ImageData {
//                 image: Array2::zeros((0, 0)),
//                 direct_beam: (0, 0),
//             },
//         }
//     }

//     pub fn from_file(fits_file: &str) -> Self {
//         let mut file = File::open(fits_file)?;
//         let mut hdulist = fits::fromfile(&mut file)?;
//         let
//     }

//     pub fn from_hdu(hdu: &fits::HDU) -> Self {
//         // map the HDU type to the elements of the HeaderData struct
//         header = match hdu {
//             PrimaryHDU => {
//                 let header = hdu.header();
//                 let data = hdu.data();
//                 let energy = header.get("Beamline Energy").unwrap();
//                 let sample_theta = header.get("Sample Theta").unwrap();
//                 let current = header.get("Beam Current").unwrap();
//                 let ho_s = header.get("Higher Order Suppressor").unwrap();
//                 let pol = header.get("EPU Polarization").unwrap();
//                 let exposure = header.get("EXPOSURE").unwrap();
//                 let image = data.get("image").unwrap();
//                 let beamspot = data.get("beamspot").unwrap();
//                 HeaderData::new(energy, sample_theta, current, ho_s, pol, exposure, image, beamspot)
//             }
//         }
//     }
// }

// pub fn load_fits_files(data_dir: &str) -> DataFrame {
//     // Get the list of fits files in the data
// }

// pub fn add_file(data_frame: &DataFrame, fits_file: &File) -> DataFrame {
//     // Append the header of the fits file to the dataframe
// }

// pub fn header_data(fits_file: &File) -> HashMap {
//     // Get the header data from the fits file
// }

// pub fn data(fits_file: &File) -> HashMap {
//     // Get the data from the fits file
// }

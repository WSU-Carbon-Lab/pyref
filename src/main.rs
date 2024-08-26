use astrors::{
    fits,
    io::{self, hdulist::HDUList, header::card::CardValue},
};
use core::panic;
use ndarray::{Array2, ArrayD, Ix2};
use polars::prelude::*;
use std::vec;

pub struct CcdFits {
    pub path: String,
    pub hdul: HDUList,
}

impl CcdFits {
    pub fn new(path: &str) -> Self {
        let hdul = fits::fromfile(&path).unwrap();
        CcdFits {
            path: path.to_string(),
            hdul,
        }
    }

    pub fn get_card(&self, card_name: &str) -> CardValue {
        let p_header = &self.hdul.hdus[0];
        let header = match p_header {
            io::hdulist::HDU::Primary(hdu) => &hdu.header[card_name].value,
            _ => panic!("Primary HDU not found!"),
        };
        header.clone()
    }

    fn ensure_2d(&self, data: ArrayD<u16>) -> Array2<u16> {
        // convert ArrayD to Array2
        let img = match data.into_dimensionality::<Ix2>() {
            Ok(img) => img,
            Err(_) => panic!("Failed to convert ArrayD to Array2!"),
        };
        img
    }

    fn get_data(&self, data: &io::hdus::image::ImageData) -> Array2<u16> {
        match data {
            io::hdus::image::ImageData::U8(image) => {
                let image_data: ArrayD<u16> = image.map(|&x| x as u16);
                self.ensure_2d(image_data)
            }
            io::hdus::image::ImageData::I16(image) => {
                let image_data: ArrayD<u16> = image.map(|&x| x as u16);
                self.ensure_2d(image_data)
            }
            io::hdus::image::ImageData::I32(image) => {
                let image_data: ArrayD<u16> = image.map(|&x| x as u16);
                self.ensure_2d(image_data)
            }
            io::hdus::image::ImageData::F32(image) => {
                let image_data: ArrayD<u16> = image.map(|&x| x as u16);
                self.ensure_2d(image_data)
            }
            io::hdus::image::ImageData::F64(image) => {
                let image_data: ArrayD<u16> = image.map(|&x| x as u16);
                self.ensure_2d(image_data)
            }
            _ => panic!("Image data is not supported!"),
        }
    }

    pub fn get_image(&self) -> Array2<u16> {
        let i_hdu = &self.hdul.hdus[2];
        // Match the i_hdu with the data
        let img = match i_hdu {
            io::hdulist::HDU::Image(i_hdu) => i_hdu,
            _ => panic!("Image HDU not found!"),
        };
        let image_data = self.get_data(&img.data);
        image_data
    }

    pub fn keys_to_polars(&self, keys: Vec<&str>) -> Series {
        let mut cards = vec![];
        for key in &keys {
            let val = match self.get_card(key).as_float() {
                Some(val) => val,
                None => panic!("Invalid card value!"),
            };
            let s = Series::new(key, &vec![val]);
            cards.push(s);
        }
        let image = self.get_image();

        let mut fields: Vec<Field> = keys
            .iter()
            .map(|&key| Field::new(key, DataType::Float64))
            .collect();
        fields.push(Field::new(
            "Image",
            DataType::List(Box::new(DataType::List(Box::new(DataType::UInt16)))),
        ));
        let schema = Schema::from_iter(fields);
        let mut _df = DataFrame::empty_with_schema(&schema);
        println!("{:?}", _df);

        // convert the image into a list of lists
        let mut out_builder = ListPrimitiveChunkedBuilder::<UInt16Type>::new(
            "Image",
            image.nrows(),
            image.nrows() * image.nrows(),
            DataType::List(Box::new(DataType::UInt16)),
        );

        for row in image.axis_iter(ndarray::Axis(0)) {
            match row.as_slice() {
                Some(row) => out_builder.append_slice(row),
                None => panic!("Failed to convert row to slice!"),
            }
        }

        out_builder.finish().into_series()
    }
}

fn main() {
    let ccd_fits = CcdFits::new("C:\\Users\\hduva\\.projects\\pyref-ccd\\test\\test.fits");
    let keys = vec!["EXPOSURE"];
    let df = ccd_fits.keys_to_polars(keys);
    println!("{:?}", df);
}

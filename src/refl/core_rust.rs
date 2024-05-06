/*
Core Rust functions for indexing the fits files from a data directory and loading their
contents in a dataframe.
*/

use polars::prelude::*;
use astrors::io;
use std::fs::File;

// Function that indexes all fits files in a directory and returns a dataframe with their
// header data. And one with thier image_data.

pub fn read_file(file: &str) -> DataFrame {
    let file = File::open(file).unwrap();
    let mut header = io::Header::new();
    let mut image_data = io::ImageHDU::new();

    header.read_from_file(&mut file).unwrap();
    image_data.read_from_file(&mut file, &header).unwrap();

}

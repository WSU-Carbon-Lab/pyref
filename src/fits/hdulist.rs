use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

const FITS_BLOCK_SIZE: usize = 2880;

use crate::fits::error::FitsReadError;
use crate::fits::header::Header;
use crate::fits::image::{ImageHdu, ImageHduHeader};
use crate::fits::primary::PrimaryHdu;

fn has_more_data<R: Read + Seek>(reader: &mut R) -> Result<bool, FitsReadError> {
    let pos = reader.stream_position()?;
    let mut b = [0u8; 1];
    let n = reader.read(&mut b)?;
    reader.seek(SeekFrom::Start(pos))?;
    Ok(n > 0)
}

fn extension_data_size(header: &Header) -> Result<usize, FitsReadError> {
    let naxis = header
        .get_card("NAXIS")
        .and_then(|c| c.value.as_int())
        .unwrap_or(0) as usize;
    let mut nelem = 1usize;
    for i in 1..=naxis {
        let key = format!("NAXIS{i}");
        if let Some(c) = header.get_card(&key) {
            if let Some(v) = c.value.as_int() {
                nelem *= v as usize;
            }
        }
    }
    let bitpix = header
        .get_card("BITPIX")
        .and_then(|c| c.value.as_int())
        .unwrap_or(16) as i32;
    let nbytes = nelem
        * match bitpix {
            8 => 1,
            16 => 2,
            32 | -32 => 4,
            _ => 8,
        };
    Ok(nbytes)
}

#[derive(Debug)]
pub enum Hdu {
    Primary(PrimaryHdu),
    Image(ImageHdu),
    ImageHeader(ImageHduHeader),
}

#[derive(Debug, Default)]
pub struct HduList {
    pub hdus: Vec<Hdu>,
}

impl HduList {
    pub fn from_file(path: &str) -> Result<Self, FitsReadError> {
        let mut f = File::open(path)?;
        let mut hdus = Vec::new();
        let mut is_primary = true;
        loop {
            if is_primary {
                let primary = PrimaryHdu::read_from_file(&mut f)?;
                hdus.push(Hdu::Primary(primary));
                is_primary = false;
            } else {
                let pos = f.stream_position()?;
                let header = Header::read_from_file(&mut f)?;
                let xtension = header
                    .get_card("XTENSION")
                    .map(|c| c.value.to_string())
                    .unwrap_or_default();
                let xtension = xtension.trim().to_uppercase();
                let xtension = xtension.trim_matches(|c: char| c.is_whitespace());
                if xtension == "IMAGE" {
                    f.seek(SeekFrom::Start(pos))?;
                    let image = ImageHdu::read_from_file(&mut f)?;
                    hdus.push(Hdu::Image(image));
                } else {
                    let nbytes = extension_data_size(&header)?;
                    f.seek(SeekFrom::Current(nbytes as i64))?;
                    let rem = nbytes % FITS_BLOCK_SIZE;
                    if rem != 0 {
                        f.seek(SeekFrom::Current((FITS_BLOCK_SIZE - rem) as i64))?;
                    }
                }
            }
            if !has_more_data(&mut f)? {
                break;
            }
        }
        Ok(HduList { hdus })
    }

    pub fn from_file_metadata_only(path: &str) -> Result<Self, FitsReadError> {
        let mut f = File::open(path)?;
        let mut hdus = Vec::new();
        let mut is_primary = true;
        loop {
            if is_primary {
                let primary = PrimaryHdu::read_from_file(&mut f)?;
                hdus.push(Hdu::Primary(primary));
                is_primary = false;
            } else {
                let pos = f.stream_position()?;
                let header = Header::read_from_file(&mut f)?;
                let xtension = header
                    .get_card("XTENSION")
                    .map(|c| c.value.to_string())
                    .unwrap_or_default();
                let xtension = xtension.trim().to_uppercase();
                let xtension = xtension.trim_matches(|c: char| c.is_whitespace());
                if xtension == "IMAGE" {
                    f.seek(SeekFrom::Start(pos))?;
                    let img_header = ImageHduHeader::read_from_file(&mut f)?;
                    hdus.push(Hdu::ImageHeader(img_header));
                } else {
                    let nbytes = extension_data_size(&header)?;
                    f.seek(SeekFrom::Current(nbytes as i64))?;
                    let rem = nbytes % FITS_BLOCK_SIZE;
                    if rem != 0 {
                        f.seek(SeekFrom::Current((FITS_BLOCK_SIZE - rem) as i64))?;
                    }
                }
            }
            if !has_more_data(&mut f)? {
                break;
            }
        }
        Ok(HduList { hdus })
    }
}

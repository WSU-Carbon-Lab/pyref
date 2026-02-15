use std::io::{Read, Seek, SeekFrom};

use crate::fits::error::FitsReadError;
use crate::fits::header::Header;
use crate::fits::utils::{nbytes_from_bitpix, FITS_BLOCK_SIZE};

fn image_size(header: &Header) -> Result<usize, FitsReadError> {
    let naxis = header
        .get_card("NAXIS")
        .and_then(|c| c.value.as_int())
        .unwrap_or(0) as usize;
    if naxis == 0 {
        return Ok(0);
    }
    let mut nelem = 1usize;
    for i in 1..=naxis {
        let key = format!("NAXIS{i}");
        let v = header
            .get_card(&key)
            .and_then(|c| c.value.as_int())
            .ok_or_else(|| FitsReadError::Parse(format!("Missing {}", key)))? as usize;
        nelem *= v;
    }
    let bitpix = header
        .get_card("BITPIX")
        .and_then(|c| c.value.as_int())
        .unwrap_or(16) as i32;
    Ok(nelem * nbytes_from_bitpix(bitpix))
}

#[derive(Debug, Clone)]
pub struct PrimaryHdu {
    pub header: Header,
}

impl PrimaryHdu {
    pub fn read_from_file<R: Read + Seek>(reader: &mut R) -> Result<Self, FitsReadError> {
        let header = Header::read_from_file(reader)?;
        let size = image_size(&header)?;
        if size > 0 {
            reader.seek(SeekFrom::Current(size as i64))?;
            let remainder = size % FITS_BLOCK_SIZE;
            if remainder != 0 {
                let pad = FITS_BLOCK_SIZE - remainder;
                reader.seek(SeekFrom::Current(pad as i64))?;
            }
        }
        Ok(PrimaryHdu { header })
    }
}

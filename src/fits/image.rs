use std::io::{Read, Seek, SeekFrom};

use ndarray::{Array2, ArrayBase, Dim, OwnedRepr};

use crate::fits::error::FitsReadError;
use crate::fits::header::Header;

const FITS_BLOCK_SIZE: usize = 2880;

fn nbytes_from_bitpix(bitpix: i32) -> usize {
    match bitpix {
        8 => 1,
        16 => 2,
        32 => 4,
        -32 => 4,
        -64 => 8,
        _ => 2,
    }
}

fn image_shape(header: &Header) -> Result<(usize, usize), FitsReadError> {
    let naxis = header
        .get_card("NAXIS")
        .and_then(|c| c.value.as_int())
        .unwrap_or(0) as usize;
    if naxis < 2 {
        return Err(FitsReadError::Parse("Image HDU must have NAXIS >= 2".into()));
    }
    let naxis1 = header
        .get_card("NAXIS1")
        .and_then(|c| c.value.as_int())
        .ok_or_else(|| FitsReadError::Parse("Missing NAXIS1".into()))? as usize;
    let naxis2 = header
        .get_card("NAXIS2")
        .and_then(|c| c.value.as_int())
        .ok_or_else(|| FitsReadError::Parse("Missing NAXIS2".into()))? as usize;
    Ok((naxis1, naxis2))
}

#[derive(Debug, Clone)]
pub struct ImageHdu {
    pub header: Header,
    pub data: ArrayBase<OwnedRepr<i16>, Dim<[usize; 2]>>,
}

impl ImageHdu {
    pub fn read_from_file<R: Read + Seek>(reader: &mut R) -> Result<Self, FitsReadError> {
        let header = Header::read_from_file(reader)?;
        let bitpix = header
            .get_card("BITPIX")
            .and_then(|c| c.value.as_int())
            .unwrap_or(16) as i32;
        if bitpix != 16 {
            return Err(FitsReadError::Unsupported(
                "Only BITPIX=16 image HDUs supported".into(),
            ));
        }
        let (naxis1, naxis2) = image_shape(&header)?;
        let nelem = naxis1 * naxis2;
        let nbytes = nelem * nbytes_from_bitpix(bitpix);
        let mut buf = vec![0u8; nbytes];
        reader.read_exact(&mut buf)?;
        let mut vec_i16 = vec![0i16; nelem];
        for (i, chunk) in buf.chunks_exact(2).enumerate() {
            vec_i16[i] = i16::from_be_bytes([chunk[0], chunk[1]]);
        }
        let data = Array2::from_shape_vec((naxis2, naxis1), vec_i16)
            .map_err(|e| FitsReadError::Parse(e.to_string()))?;
        let remainder = nbytes % FITS_BLOCK_SIZE;
        if remainder != 0 {
            let pad = FITS_BLOCK_SIZE - remainder;
            reader.seek(SeekFrom::Current(pad as i64))?;
        }
        Ok(ImageHdu { header, data })
    }
}

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
pub struct ImageHduHeader {
    pub header: Header,
    pub naxis1: usize,
    pub naxis2: usize,
}

impl ImageHduHeader {
    pub fn read_from_file<R: Read + Seek>(reader: &mut R) -> Result<Self, FitsReadError> {
        let header = Header::read_from_file(reader)?;
        let (naxis1, naxis2) = image_shape(&header)?;
        let bitpix = header
            .get_card("BITPIX")
            .and_then(|c| c.value.as_int())
            .unwrap_or(16) as i32;
        let nelem = naxis1 * naxis2;
        let nbytes = nelem * nbytes_from_bitpix(bitpix);
        reader.seek(SeekFrom::Current(nbytes as i64))?;
        let remainder = nbytes % FITS_BLOCK_SIZE;
        if remainder != 0 {
            let pad = FITS_BLOCK_SIZE - remainder;
            reader.seek(SeekFrom::Current(pad as i64))?;
        }
        Ok(ImageHduHeader {
            header,
            naxis1,
            naxis2,
        })
    }
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

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    fn minimal_fits_with_image(naxis1: u32, naxis2: u32) -> Vec<u8> {
        let card = |s: &str| {
            let mut buf = [b' '; 80];
            let b = s.as_bytes();
            let n = b.len().min(80);
            buf[..n].copy_from_slice(&b[..n]);
            buf.to_vec()
        };
        let mut out = Vec::new();
        out.extend(card("SIMPLE  =                    T"));
        out.extend(card("BITPIX  =                    8"));
        out.extend(card("NAXIS   =                    0"));
        out.extend(card("END"));
        while out.len() % 2880 != 0 {
            out.push(b' ');
        }
        out.extend(card("XTENSION= 'IMAGE   '"));
        out.extend(card("BITPIX  =                   16"));
        out.extend(card("NAXIS   =                    2"));
        out.extend(card(&format!("NAXIS1  = {:>20}", naxis1)));
        out.extend(card(&format!("NAXIS2  = {:>20}", naxis2)));
        out.extend(card("PCOUNT  =                    0"));
        out.extend(card("GCOUNT  =                    1"));
        out.extend(card("END"));
        while out.len() % 2880 != 0 {
            out.push(b' ');
        }
        let nelem = (naxis1 as usize) * (naxis2 as usize) * 2;
        out.resize(out.len() + nelem, 0);
        while out.len() % 2880 != 0 {
            out.push(b' ');
        }
        out
    }

    #[test]
    fn image_hdu_header_read_from_file_seeks_past_data() {
        let naxis1 = 10u32;
        let naxis2 = 20u32;
        let fits = minimal_fits_with_image(naxis1, naxis2);
        let mut cursor = Cursor::new(&fits);
        cursor.seek(SeekFrom::Start(2880)).unwrap();
        let header_only = ImageHduHeader::read_from_file(&mut cursor).unwrap();
        assert_eq!(header_only.naxis1, 10);
        assert_eq!(header_only.naxis2, 20);
        let pos_after = cursor.stream_position().unwrap();
        assert_eq!(pos_after, 2880 + 2880 + 10 * 20 * 2 + 2480);
    }
}

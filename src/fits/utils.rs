pub const FITS_BLOCK_SIZE: usize = 2880;

pub fn nbytes_from_bitpix(bitpix: i32) -> usize {
    match bitpix {
        8 => 1,
        16 => 2,
        32 | -32 => 4,
        -64 => 8,
        _ => 2,
    }
}

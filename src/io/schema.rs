//! Canonical FITS metadata schema: one row per FITS file.
//!
//! Used by `scan_from_catalog`, batch read output, and `scan_fits` io_source
//! so that catalog and disk reads produce the same column set.

use polars::prelude::*;

/// Column names for the canonical FITS metadata row (one file = one row).
/// Order matches catalog table and scan_from_catalog output.
pub const FITS_METADATA_COLUMNS: &[&str] = &[
    "file_path",
    "data_offset",
    "naxis1",
    "naxis2",
    "bitpix",
    "bzero",
    "data_size",
    "file_name",
    "sample_name",
    "tag",
    "scan_number",
    "frame_number",
    "DATE",
    "Beamline Energy",
    "Sample Theta",
    "CCD Theta",
    "Higher Order Suppressor",
    "EPU Polarization",
    "EXPOSURE",
    "Sample Name",
    "Scan ID",
    "Lambda",
    "Q",
];

/// Describes the canonical FITS metadata schema used by catalog and batch read.
#[derive(Debug, Clone)]
pub struct FitsMetadataSchema;

impl FitsMetadataSchema {
    /// Returns the ordered list of column names for one FITS file row.
    pub fn columns() -> &'static [&'static str] {
        FITS_METADATA_COLUMNS
    }

    /// Builds a Polars Schema for the canonical columns with default dtypes.
    pub fn schema() -> Schema {
        let fields: Vec<(_, DataType)> = FITS_METADATA_COLUMNS
            .iter()
            .map(|name| {
                let dtype = match *name {
                    "file_path" | "file_name" | "sample_name" | "DATE" | "Sample Name" => {
                        DataType::String
                    }
                    "tag" => DataType::String,
                    "data_offset" | "naxis1" | "naxis2" | "bitpix" | "bzero" | "data_size"
                    | "scan_number" | "frame_number" => DataType::Int64,
                    "Beamline Energy"
                    | "Sample Theta"
                    | "CCD Theta"
                    | "Higher Order Suppressor"
                    | "EPU Polarization"
                    | "EXPOSURE"
                    | "Scan ID"
                    | "Lambda"
                    | "Q" => DataType::Float64,
                    _ => DataType::String,
                };
                ((*name).into(), dtype)
            })
            .collect();
        Schema::from_iter(fields)
    }
}

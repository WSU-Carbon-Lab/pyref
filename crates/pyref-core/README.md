# Rust Backend for accessing CCD data in FITS files

This crate is a light wrapper over the [astrors-fork](https://github.com/HarlanHeilman/astrors)
crate. It provides a simple interface to access CCD data in FITS files and convert them to Polars DataFrames.

## Features

- Read single FITS files into Polars DataFrames
- Process entire directories of FITS files
- Filter files using glob patterns
- Process specific selections of FITS files
- Automatic calculation of important derived values (Q, Lambda, etc.)
- Support for different experiment types (XRR, XRS)
- Automatic inclusion of DATE header for chronological sorting
- Simplified file name handling

## Usage

### Basic Usage

```rust
use pyref_core::*;
use std::path::Path;

fn main() {
    // Read a single FITS file
    let path = "path/to/fits/file.fits";
    let df = read_fits_file(path, ExperimentType::Xrr).unwrap();
    println!("{:?}", df);

    // Load all FITS files in a directory
    let dir_path = "path/to/directory";
    let all_df = read_experiment_dir(dir_path, ExperimentType::Xrr).unwrap();
    println!("{:?}", all_df);
}
```

### Advanced Usage

```rust
use pyref_core::*;
use std::path::PathBuf;

fn main() {
    // Read files matching a pattern
    let pattern_df = read_fits_with_pattern(
        "path/to/directory",
        "Y6_refl_*.fits",
        ExperimentType::Xrr
    ).unwrap();

    // Read specific files
    let file_paths = vec![
        PathBuf::from("path/to/file1.fits"),
        PathBuf::from("path/to/file2.fits"),
        PathBuf::from("path/to/file3.fits"),
    ];

    let selected_df = read_multiple_fits_files(file_paths, ExperimentType::Xrr).unwrap();

    // Work with the DataFrames
    let combined = selected_df.vstack(&pattern_df).unwrap();
    println!("Combined shape: {:?}", combined.shape());
}
```

## Supported Experiment Types

- `ExperimentType::Xrr` - X-ray Reflectivity
- `ExperimentType::Xrs` - X-ray Spectroscopy
- `ExperimentType::Other` - Generic FITS files

## Header Values

Each experiment type automatically extracts the relevant header values, plus standard headers:

```rust
// XRR headers
ExperimentType::Xrr => vec![
    "Sample Theta [deg]",
    "CCD Theta [deg]",
    "Beamline Energy [eV]",
    "Beam Current [mA]",
    "EPU Polarization [deg]",
    "Horizontal Exit Slit Size [um]",
    "Higher Order Suppressor [mm]",
    "EXPOSURE [s]",
]

// Standard headers always included for all experiment types
"DATE" // Date/time information from the FITS header
```

## File Name Handling

The library now extracts only the base file name from the path without parsing frame numbers or scan IDs. This simplifies file handling and makes it more robust.

## Calculated Values

The library automatically calculates:

- Wavelength (Lambda) in Angstroms
- Momentum transfer (Q) in inverse Angstroms
- Theta offset for calibration

## Sorting

Output DataFrames are automatically sorted by:

1. Date (from the DATE header)
2. File name (as fallback sorting key)

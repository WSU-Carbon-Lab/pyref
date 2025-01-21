# Rust Backend for accessing CCD data in FITS files

This crate is a light wrapper over the [astrors](https://github.com/Schwarzam/astrors)
crate. It provides a simple interface to access CCD data in FITS files.

## Usage

```rust
use pyref_ccd::*;
use std::path::Path;

fn main() {
    let path = "path/to/fits/file.fits";
    let df = read_fits(path).unwrap();
    println!("{:?}", df);

    // Or to load all data in a directory

    let path = "path/to/directory";
    let all_df = read_experiment(path, ExperimentType::Xrr).unwrap();
    println!("{:?}", all_df);
}
```

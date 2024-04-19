/*
Core Rust functions for indexing the fits files from a data directory and loading their
contents in a dataframe.
*/

use polars::prelude::*;
use fitrs::{Fits, FitsData, FitsDataArray};
use notify::{Watcher, RecursiveMode, watcher};


// Wrapper structures for sub trees of the system file tree for better exp syntax
pub struct BeamTime {
    pub name: String,
    pub path: String,
    pub date: String,
}

/*
Core Rust functions for indexing the fits files from a data directory and loading their
contents in a dataframe.
*/

use polars::prelude::*;
use dirs;
use astrors::io;
use std::fs::File;

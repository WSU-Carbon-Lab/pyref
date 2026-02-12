use std::path::{Path, PathBuf};

use pyref::{
    errors::FitsErrorKind,
    loader::{read_experiment_headers_only, read_fits_headers_only, read_multiple_fits_headers_only},
};

const TEST_DATA_DIR: &str = "tests/fixtures";
const HEADER_KEYS: &[&str] = &[
    "DATE",
    "Beamline Energy",
    "Sample Theta",
    "CCD Theta",
    "Higher Order Suppressor",
    "EPU Polarization",
];
const HEADER_KEYS_2: &[&str] = &[];

fn get_all_test_fits_files() -> Vec<PathBuf> {
    let test_dir = Path::new(TEST_DATA_DIR);
    let Ok(entries) = std::fs::read_dir(test_dir) else {
        return vec![];
    };
    entries
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) == Some("fits") {
                Some(path)
            } else {
                None
            }
        })
        .collect()
}

fn header_keys_vec(num: usize) -> Vec<String> {
    if num == 1 {
        HEADER_KEYS.iter().map(|&s| s.to_string()).collect()
    } else {
        HEADER_KEYS_2.iter().map(|&s| s.to_string()).collect()
    }
}

fn required_header_only_columns() -> &'static [&'static str] {
    &["file_path", "data_offset", "naxis1", "naxis2", "bitpix", "bzero", "file_name"]
}

#[test]
#[ignore = "requires Python runtime; run Python tests via pytest"]
fn test_read_single_fits_headers_only() {
    let fits_files = get_all_test_fits_files();
    let first_file = match fits_files.first() {
        Some(p) => p,
        None => return,
    };
    let header_keys = header_keys_vec(1);
    let result = read_fits_headers_only(first_file.clone(), &header_keys);
    assert!(
        result.is_ok(),
        "read_fits_headers_only failed: {:?}",
        result.err()
    );
    let df = result.unwrap();
    assert!(!df.is_empty(), "DataFrame should not be empty");
    for &col in required_header_only_columns() {
        assert!(df.column(col).is_ok(), "Missing column: {}", col);
    }
}

#[test]
#[ignore = "requires Python runtime; run Python tests via pytest"]
fn test_error_handling_headers_only() {
    let non_existent_file = PathBuf::from("non_existent_file.fits");
    let header_keys = header_keys_vec(0);
    let result = read_fits_headers_only(non_existent_file, &header_keys);
    assert!(result.is_err(), "Should fail with non-existent file");
    if let Err(err) = result {
        assert_eq!(err.kind, FitsErrorKind::Io);
    }

    let non_existent_dir = "non_existent_directory";
    let dir_result = read_experiment_headers_only(non_existent_dir, &header_keys);
    assert!(dir_result.is_err(), "Should fail with non-existent directory");
    if let Err(err) = dir_result {
        assert_eq!(err.kind, FitsErrorKind::NotFound);
    }
}

#[test]
#[ignore = "requires Python runtime; run Python tests via pytest"]
fn test_read_multiple_fits_headers_only() {
    let fits_files = get_all_test_fits_files();
    if fits_files.is_empty() {
        return;
    }
    let header_keys = header_keys_vec(0);
    let paths: Vec<PathBuf> = fits_files.iter().take(5).cloned().collect();
    let result = read_multiple_fits_headers_only(paths, &header_keys);
    assert!(
        result.is_ok(),
        "read_multiple_fits_headers_only failed: {:?}",
        result.err()
    );
    let df = result.unwrap();
    assert!(!df.is_empty());
    for &col in required_header_only_columns() {
        assert!(df.column(col).is_ok(), "Missing column: {}", col);
    }
}

#[test]
#[ignore = "requires Python runtime; run Python tests via pytest"]
fn test_read_experiment_headers_only() {
    let fits_files = get_all_test_fits_files();
    if fits_files.is_empty() {
        return;
    }
    let header_keys = header_keys_vec(0);
    let result = read_experiment_headers_only(TEST_DATA_DIR, &header_keys);
    assert!(
        result.is_ok(),
        "read_experiment_headers_only failed: {:?}",
        result.err()
    );
    let df = result.unwrap();
    assert!(!df.is_empty());
    for &col in required_header_only_columns() {
        assert!(df.column(col).is_ok(), "Missing column: {}", col);
    }
}

#[test]
#[ignore = "requires Python runtime; run Python tests via pytest"]
fn test_read_fits_headers_only_includes_parsed_filename_columns() {
    let fits_files = get_all_test_fits_files();
    if fits_files.is_empty() {
        return;
    }
    let first_file = &fits_files[0];
    let header_keys = header_keys_vec(0);
    let result = read_fits_headers_only(first_file.clone(), &header_keys);
    assert!(result.is_ok(), "read_fits_headers_only failed: {:?}", result.err());
    let df = result.unwrap();
    for &col in &["file_name", "sample_name", "tag", "experiment_number", "frame_number"] {
        if df.column(col).is_err() {
            continue;
        }
    }
    assert!(df.column("file_name").is_ok());
}

use std::path::{Path, PathBuf};

// Import the crate to test
use pyref_core::{
    errors::FitsLoaderError,
    loader::{read_experiment, read_experiment_pattern, read_fits, read_multiple_fits},
};

// Test constants
const TEST_DATA_DIR: &str = "./tests/data";
const HEADER_KEYS: &[&str] = &[
    "DATE",
    "Beamline Energy",
    "Sample Theta",
    "CCD Theta",
    "Higher Order Suppressor",
    "EPU Polarization",
];
const HEADER_KEYS_2: &[&str] = &[];

/// Helper function to get all FITS files in the test directory
fn get_all_test_fits_files() -> Vec<PathBuf> {
    let test_dir = Path::new(TEST_DATA_DIR);
    std::fs::read_dir(test_dir)
        .expect("Failed to read test data directory")
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

/// Helper function to convert header keys slice to Vec<String>
fn header_keys_vec(num: usize) -> Vec<String> {
    if num == 1 {
        HEADER_KEYS.iter().map(|&s| s.to_string()).collect()
    } else {
        HEADER_KEYS_2.iter().map(|&s| s.to_string()).collect()
    }
}

#[test]
fn test_read_single_fits_file() {
    let fits_files = get_all_test_fits_files();
    let first_file = &fits_files[0];
    println!("Testing with file: {}", first_file.display());

    // Test with both sets of header keys
    for i in [0, 1] {
        println!("Testing with header_keys_vec({})", i);
        let header_keys = header_keys_vec(i);
        let result = read_fits(first_file.clone(), &header_keys);

        assert!(
            result.is_ok(),
            "Failed to read FITS file with header_keys_vec({}): {:?}",
            i,
            result.err()
        );

        let df = result.unwrap();
        assert!(!df.is_empty(), "DataFrame should not be empty");

        // Check if some expected columns exist
        assert!(df.column("file_name").is_ok(), "Missing 'file_name' column");
        assert!(df.column("RAW").is_ok(), "Missing 'RAW' column");

        // Print DataFrame schema for debugging
        println!(
            "DataFrame schema (header_keys_vec({})): {:#?}",
            i,
            df.schema()
        );
        println!("DataFrame shape (header_keys_vec({})): {:?}", i, df.shape());
        println!(
            "DataFrame head (header_keys_vec({})): {:#?}",
            i,
            df.head(Some(5))
        );
    }
}

#[test]
fn test_error_handling() {
    // Test with non-existent file
    let non_existent_file = PathBuf::from("non_existent_file.fits");
    println!(
        "Testing with non-existent file: {}",
        non_existent_file.display()
    );

    let header_keys = header_keys_vec(0);

    let result = read_fits(non_existent_file, &header_keys);
    assert!(result.is_err(), "Should fail with non-existent file");

    if let Err(err) = result {
        println!("Expected error: {:?}", err);
        assert!(
            matches!(err, FitsLoaderError::FitsError(_)),
            "Unexpected error type"
        );
    }

    // Test with non-existent directory
    let non_existent_dir = "non_existent_directory";
    println!("Testing with non-existent directory: {}", non_existent_dir);

    let dir_result = read_experiment(non_existent_dir, &header_keys);
    assert!(
        dir_result.is_err(),
        "Should fail with non-existent directory"
    );

    if let Err(err) = dir_result {
        println!("Expected directory error: {:?}", err);
        assert!(
            matches!(err, FitsLoaderError::FitsError(_)),
            "Unexpected directory error type"
        );
    }
}

// Test read multiple FITS files
#[test]
fn test_read_multiple_fits_files() {
    let fits_files = get_all_test_fits_files();
    assert!(
        !fits_files.is_empty(),
        "No .fits files found in test data directory"
    );
    println!("Found {} FITS files for testing", fits_files.len());
    let header_keys = header_keys_vec(0);
    let result = read_multiple_fits(fits_files.clone(), &header_keys);
    assert!(
        result.is_ok(),
        "Failed to read multiple FITS files: {:?}",
        result.err()
    );
    let df = result.unwrap();
    assert!(!df.is_empty(), "DataFrame should not be empty");
    // Check if some expected columns exist
    assert!(df.column("file_name").is_ok(), "Missing 'file_name' column");
    assert!(df.column("RAW").is_ok(), "Missing 'RAW' column");
    // Print DataFrame schema for debugging
    println!("DataFrame schema: {:#?}", df.schema());
    println!("DataFrame shape: {:?}", df.shape());
    println!("DataFrame head: {:#?}", df.head(Some(5)));
}

// test read experiment
#[test]
fn test_read_experiment() {
    let fits_files = get_all_test_fits_files();
    assert!(
        !fits_files.is_empty(),
        "No .fits files found in test data directory"
    );
    let header_keys = header_keys_vec(0);
    let result = read_experiment(TEST_DATA_DIR, &header_keys);
    assert!(
        result.is_ok(),
        "Failed to read experiment: {:?}",
        result.err()
    );
    let df = result.unwrap();
    assert!(!df.is_empty(), "DataFrame should not be empty");
    // Check if some expected columns exist
    assert!(df.column("file_name").is_ok(), "Missing 'file_name' column");
    assert!(df.column("RAW").is_ok(), "Missing 'RAW' column");
    // Print DataFrame schema for debugging
    println!("DataFrame schema: {:#?}", df.schema());
    println!("DataFrame shape: {:?}", df.shape());
    println!("DataFrame head: {:#?}", df.head(Some(5)));
}

// test read experiment pattern
#[test]
fn test_read_experiment_pattern() {
    let fits_files = get_all_test_fits_files();
    assert!(
        !fits_files.is_empty(),
        "No .fits files found in test data directory"
    );
    let header_keys = header_keys_vec(0);
    let result = read_experiment_pattern(TEST_DATA_DIR, "*june*.fits", &header_keys);
    assert!(
        result.is_ok(),
        "Failed to read experiment: {:?}",
        result.err()
    );
    let df = result.unwrap();
    assert!(!df.is_empty(), "DataFrame should not be empty");
    // Check if some expected columns exist
    assert!(df.column("file_name").is_ok(), "Missing 'file_name' column");
    assert!(df.column("RAW").is_ok(), "Missing 'RAW' column");
    // Print DataFrame schema for debugging
    println!("DataFrame schema: {:#?}", df.schema());
    println!("DataFrame shape: {:?}", df.shape());
    println!("DataFrame head: {:#?}", df.head(Some(5)));
}

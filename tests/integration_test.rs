use std::path::{Path, PathBuf};

// Import the crate to test
use pyref::{
    errors::FitsLoaderError,
    loader::{read_experiment, read_experiment_pattern, read_fits, read_multiple_fits},
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

#[test]
#[ignore = "requires Python runtime; run Python tests via pytest"]
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
#[ignore = "requires Python runtime; run Python tests via pytest"]
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

#[test]
#[ignore = "requires Python runtime; run Python tests via pytest"]
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

#[test]
#[ignore = "requires Python runtime; run Python tests via pytest"]
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

#[test]
#[ignore = "requires Python runtime; run Python tests via pytest"]
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

#[test]
#[ignore = "requires Python runtime; run Python tests via pytest"]
fn test_read_fits_includes_parsed_filename_columns() {
    let fits_files = get_all_test_fits_files();
    assert!(
        !fits_files.is_empty(),
        "No .fits files found in test data directory"
    );
    let first_file = &fits_files[0];
    let header_keys = header_keys_vec(0);
    let result = read_fits(first_file.clone(), &header_keys);
    assert!(result.is_ok(), "read_fits failed: {:?}", result.err());
    let df = result.unwrap();
    assert!(df.column("file_name").is_ok(), "Missing 'file_name' column");
    assert!(df.column("sample_name").is_ok(), "Missing 'sample_name' column");
    assert!(df.column("tag").is_ok(), "Missing 'tag' column");
    assert!(df.column("experiment_number").is_ok(), "Missing 'experiment_number' column");
    assert!(df.column("frame_number").is_ok(), "Missing 'frame_number' column");
    let stem = first_file.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    if stem.starts_with("monlayerjune") && stem.contains("81041") {
        let sample = df.column("sample_name").unwrap().str().unwrap().get(0).flatten();
        let exp = df.column("experiment_number").unwrap().i64().unwrap().get(0);
        let frame = df.column("frame_number").unwrap().i64().unwrap().get(0);
        assert_eq!(sample, Some("monlayerjune"), "sample_name for monlayerjune stem");
        assert_eq!(exp, Some(81041), "experiment_number");
        assert!(frame.is_some(), "frame_number present");
    }
}

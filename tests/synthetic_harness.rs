//! Self-tests for the synthetic-beamtime helper.
//!
//! These exercise the FITS writer independently of the catalog ingest so a
//! broken generator shows up here rather than as mysterious ingest failures.

mod common;

use std::path::PathBuf;

use pyref::fits::HduList;
use pyref::io::image_mmap::load_image_pixels;
use pyref::io::parse_fits_stem;
use pyref::io::ImageInfo;
use pyref::loader::read_fits_headers_only_row;

use common::{build_synthetic_beamtime, synthetic_pixel_value, SyntheticLayout, SYNTHETIC_SAMPLE_NAME};

fn header_items() -> Vec<String> {
    [
        "DATE",
        "Beamline Energy",
        "Sample Theta",
        "CCD Theta",
        "Higher Order Suppressor",
        "EPU Polarization",
    ]
    .iter()
    .map(|s| (*s).to_string())
    .collect()
}

fn collect_fits_paths(ccd_dir: &std::path::Path) -> Vec<PathBuf> {
    let mut v: Vec<_> = std::fs::read_dir(ccd_dir)
        .expect("read CCD dir")
        .filter_map(|e| e.ok().map(|x| x.path()))
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("fits"))
        .collect();
    v.sort();
    v
}

#[test]
fn synthetic_files_parse_through_headers_only_row() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let layout = SyntheticLayout::uniform(2, 3, 16, 16);
    let beamtime = build_synthetic_beamtime(layout, tmp.path()).expect("build");
    let fits_paths = collect_fits_paths(&beamtime.root.join("CCD"));
    assert_eq!(fits_paths.len(), 6, "expected 2*3 synthetic FITS files");

    let items = header_items();
    for path in &fits_paths {
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .expect("UTF-8 stem");
        let parsed =
            parse_fits_stem(stem).unwrap_or_else(|| panic!("parse_fits_stem failed for {stem}"));
        assert_eq!(parsed.sample_name, SYNTHETIC_SAMPLE_NAME);
        assert!(parsed.scan_number >= 1);
        assert!(parsed.frame_number >= 1);

        let row = read_fits_headers_only_row(path.clone(), &items)
            .unwrap_or_else(|e| panic!("read_fits_headers_only_row failed for {stem}: {e}"));
        assert_eq!(row.bitpix, 16_i64);
        assert_eq!(row.naxis1, 16_i64);
        assert_eq!(row.naxis2, 16_i64);
        assert_eq!(row.bzero, 32_768_i64);
        assert!(row.data_offset > 0);
        assert_eq!(row.file_path, path.display().to_string());
    }
}

#[test]
fn synthetic_pixels_round_trip_via_image_mmap() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let layout = SyntheticLayout::uniform(1, 1, 8, 8);
    let beamtime = build_synthetic_beamtime(layout.clone(), tmp.path()).expect("build");
    let fits = collect_fits_paths(&beamtime.root.join("CCD"));
    let path = fits.first().expect("one fits file");

    let hdul = HduList::from_file_headers_only(&path.display().to_string()).expect("headers");
    let image_header = hdul.image_header.as_ref().expect("image HDU");
    let info = ImageInfo::from_header(path.clone(), image_header);
    assert_eq!(info.bitpix, 16);
    assert_eq!(info.naxis1, layout.width as usize);
    assert_eq!(info.naxis2, layout.height as usize);
    assert_eq!(info.bzero, 32_768);

    let data = load_image_pixels(path, &info).expect("load_image_pixels");
    assert_eq!(data.shape(), &[layout.height as usize, layout.width as usize]);

    for row in 0..(layout.height as usize) {
        for col in 0..(layout.width as usize) {
            let expected = synthetic_pixel_value(0, 0, row, col) as i64 + info.bzero;
            let got = data[[row, col]];
            assert_eq!(
                got, expected,
                "pixel round-trip mismatch at ({row},{col}): got {got}, expected {expected}",
            );
        }
    }
}

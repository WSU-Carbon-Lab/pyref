use pyref_core::loader::*;

fn main() {
    let test_dr = "/home/hduva/projects/pyref-ccd/test";
    let df = read_experiment(&test_dr, "xrr").unwrap();
    println!(
        "{:?}",
        df.select([
            "Sample Theta [deg]",
            "EXPOSURE [s]",
            "Higher Order Suppressor [mm]"
        ])
    );
}

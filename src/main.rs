pub mod loader;

fn main() {
    let df = loader::read_experiment("/home/hduva/projects/pyref-ccd/test/", "xrr");
    println!("{:?}", df);
}

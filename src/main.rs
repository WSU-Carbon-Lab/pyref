use astrors::fits;

fn main() {
    let path = "~/.projects/pyref/src/test.fits";
    let hdul = fits::HDUList::fromfile(path).unwrap();
    println!("{:?}", hdul.hdus[0]);
}

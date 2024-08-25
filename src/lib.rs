use astrors::{fits, io};
use core::panic;
use ndarray::ArrayD;

pub struct Measurement {
    pub value: f64,
    pub unit: String,
}

impl Measurement {
    pub fn new(value: f64, unit: &str) -> Self {
        Measurement {
            value,
            unit: unit.to_string(),
        }
    }

    pub fn to_string(&self) -> String {
        format!("{:>10.3} [{}]", self.value, self.unit)
    }
}

pub enum Pol {
    S,
    P,
}

impl Pol {
    pub fn to_string(&self) -> String {
        match self {
            Pol::S => "S".to_string(),
            Pol::P => "P".to_string(),
        }
    }

    pub fn from_hdu(pol: f64) -> Self {
        match pol {
            100.0 => Pol::S,
            190.0 => Pol::P,
            _ => panic!("Invalid polarization!"),
        }
    }
}

pub struct CcdData {
    pub energy: Measurement,
    pub pol: Pol,
    pub current: Measurement,
    pub hos: Measurement,
    pub exposure: Measurement,
    pub image: ArrayD<u16>,
}

impl CcdData {
    pub fn new(
        energy: f64,
        pol: Pol,
        current: f64,
        hos: f64,
        exposure: f64,
        image: ArrayD<u16>,
    ) -> Self {
        CcdData {
            energy: Measurement::new(energy, "eV"),
            pol,
            current: Measurement::new(current, "nA"),
            hos: Measurement::new(hos, "mm"),
            exposure: Measurement::new(exposure, "s"),
            image,
        }
    }

    pub fn read_fits(path: &str) -> Self {
        let hdul = fits::fromfile(&path).unwrap();
        let p_hdu = &hdul.hdus[0];
        if let io::hdulist::HDU::Primary(hdu) = p_hdu {
            let header = &hdu.header;
            let energy = header["Beamline Energy"].value.as_float().unwrap();
            let epu_pol = header["EPU Polarization"].value.as_float().unwrap();
            let pol = Pol::from_hdu(epu_pol);
            let current = header["Beam Current"].value.as_float().unwrap();
            let hos = header["Higher Order Suppressor"].value.as_float().unwrap();
            let exposure = header["EXPOSURE"].value.as_float().unwrap();
            let image = CcdData::get_image(&hdul);
            CcdData::new(energy, pol, current, hos, exposure, image)
        } else {
            panic!("Fits file does not conform to the standard!");
        }
    }

    fn get_image(hdul: &io::hdulist::HDUList) -> ArrayD<u16> {
        let i_hdu = &hdul.hdus[2];
        // Match the i_hdu with the data
        let img = match i_hdu {
            io::hdulist::HDU::Image(i_hdu) => i_hdu,
            _ => panic!("Image HDU not found!"),
        };
        let image_data = match &img.data {
            io::hdus::image::ImageData::U8(image) => ArrayD::from(image.map(|&x| x as u16)),
            io::hdus::image::ImageData::I16(image) => ArrayD::from(image.map(|&x| x as u16)),
            io::hdus::image::ImageData::I32(image) => ArrayD::from(image.map(|&x| x as u16)),
            io::hdus::image::ImageData::F32(image) => ArrayD::from(image.map(|&x| x as u16)),
            io::hdus::image::ImageData::F64(image) => ArrayD::from(image.map(|&x| x as u16)),
            _ => panic!("Image data is not supported!"),
        };
        image_data
    }

    pub fn pretty_print(&self) {
        println!("  ┏━━━━━━━━━━━ CCD Data ━━━━━━━━━━┓");
        println!("  ┃ Energy:       {:>12} ┃", self.energy.to_string());
        println!("  ┃ Polarization: {:>12}    ┃", self.pol.to_string());
        println!("  ┃ Current:      {:>12} ┃", self.current.to_string());
        println!("  ┃ HOS:          {:>12} ┃", self.hos.to_string());
        println!("  ┃ Exposure:     {:>12}  ┃", self.exposure.to_string());
        println!("  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛");
        println!("");
        println!("{}", self.image);
    }
}

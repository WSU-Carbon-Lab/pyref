/// Represents different types of experiments.
pub enum ExperimentType {
    Xrr,
    Xrs,
    All,
}

impl ExperimentType {
    /// Creates an `ExperimentType` from a string.
    pub fn from_str(exp_type: &str) -> Result<Self, crate::errors::FitsLoaderError> {
        match exp_type.to_lowercase().as_str() {
            "xrr" => Ok(ExperimentType::Xrr),
            "xrs" => Ok(ExperimentType::Xrs),
            "other" => Ok(ExperimentType::All),
            _ => Err(crate::errors::FitsLoaderError::InvalidExperimentType(
                exp_type.to_string(),
            )),
        }
    }

    /// Retrieves the relevant header keys for the experiment type.
    pub fn get_keys(&self) -> Vec<HeaderValue> {
        let mut keys = match self {
            ExperimentType::Xrr => vec![
                HeaderValue::SampleTheta,
                HeaderValue::CCDTheta,
                HeaderValue::BeamlineEnergy,
                HeaderValue::BeamCurrent,
                HeaderValue::EPUPolarization,
                HeaderValue::HorizontalExitSlitSize,
                HeaderValue::HigherOrderSuppressor,
                HeaderValue::Exposure,
            ],
            ExperimentType::Xrs => vec![HeaderValue::BeamlineEnergy],
            ExperimentType::All => vec![],
        };

        // Always include the Date header for all experiment types
        keys.push(HeaderValue::Date);
        keys
    }

    /// Retrieves the header names for display purposes.
    pub fn names(&self) -> Vec<&str> {
        match self {
            ExperimentType::Xrr => vec![
                "Sample Theta",
                "CCD Theta",
                "Beamline Energy",
                "Beam Current",
                "EPU Polarization",
                "Horizontal Exit Slit Size",
                "Higher Order Suppressor",
                "EXPOSURE",
            ],
            ExperimentType::Xrs => vec!["Beamline Energy"],
            ExperimentType::All => vec![],
        }
    }
}

/// Represents different header values.
pub enum HeaderValue {
    SampleTheta,
    CCDTheta,
    BeamlineEnergy,
    EPUPolarization,
    BeamCurrent,
    HorizontalExitSlitSize,
    HigherOrderSuppressor,
    Exposure,
    Date,
}

impl HeaderValue {
    /// Returns the unit associated with the header value.
    pub fn unit(&self) -> &str {
        match self {
            HeaderValue::SampleTheta => "[deg]",
            HeaderValue::CCDTheta => "[deg]",
            HeaderValue::BeamlineEnergy => "[eV]",
            HeaderValue::BeamCurrent => "[mA]",
            HeaderValue::EPUPolarization => "[deg]",
            HeaderValue::HorizontalExitSlitSize => "[um]",
            HeaderValue::HigherOrderSuppressor => "[mm]",
            HeaderValue::Exposure => "[s]",
            HeaderValue::Date => "",
        }
    }

    /// Returns the HDU key associated with the header value.
    pub fn hdu(&self) -> &str {
        match self {
            HeaderValue::SampleTheta => "Sample Theta",
            HeaderValue::CCDTheta => "CCD Theta",
            HeaderValue::BeamlineEnergy => "Beamline Energy",
            HeaderValue::BeamCurrent => "Beam Current",
            HeaderValue::EPUPolarization => "EPU Polarization",
            HeaderValue::HorizontalExitSlitSize => "Horizontal Exit Slit Size",
            HeaderValue::HigherOrderSuppressor => "Higher Order Suppressor",
            HeaderValue::Exposure => "EXPOSURE",
            HeaderValue::Date => "DATE",
        }
    }

    /// Returns the full name with units for display.
    pub fn name(&self) -> &str {
        match self {
            HeaderValue::SampleTheta => "Sample Theta [deg]",
            HeaderValue::CCDTheta => "CCD Theta [deg]",
            HeaderValue::BeamlineEnergy => "Beamline Energy [eV]",
            HeaderValue::BeamCurrent => "Beam Current [mA]",
            HeaderValue::EPUPolarization => "EPU Polarization [deg]",
            HeaderValue::HorizontalExitSlitSize => "Horizontal Exit Slit Size [um]",
            HeaderValue::HigherOrderSuppressor => "Higher Order Suppressor [mm]",
            HeaderValue::Exposure => "EXPOSURE [s]",
            HeaderValue::Date => "DATE",
        }
    }

    /// Returns the snake_case name without units.
    pub fn snake_case_name(&self) -> &str {
        match self {
            HeaderValue::SampleTheta => "sample_theta",
            HeaderValue::CCDTheta => "ccd_theta",
            HeaderValue::BeamlineEnergy => "beamline_energy",
            HeaderValue::BeamCurrent => "beam_current",
            HeaderValue::EPUPolarization => "epu_polarization",
            HeaderValue::HorizontalExitSlitSize => "horizontal_exit_slit_size",
            HeaderValue::HigherOrderSuppressor => "higher_order_suppressor",
            HeaderValue::Exposure => "exposure",
            HeaderValue::Date => "date",
        }
    }
}

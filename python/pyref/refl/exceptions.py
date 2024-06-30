class ScanError(Exception):
    """Raised when a there is an error reading a CCD scan directory."""


class FitsReadError(Exception):
    """Raised when there is an error reading a FITS file."""


class AppConfigError(Exception):
    """Raised when there is an error reading the config."""

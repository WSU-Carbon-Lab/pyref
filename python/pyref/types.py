"""Types for the Python reference implementation."""

from __future__ import annotations

from enum import Enum


class HeaderValue(Enum):
    """Enumeration of possible header values."""

    SAMPLE_THETA = 1
    BEAMLINE_ENERGY = 2
    EPU_POLARIZATION = 3
    HORIZONTAL_EXIT_SLIT_SIZE = 4
    HIGHER_ORDER_SUPPRESSOR = 5
    EXPOSURE = 6

    def unit(self) -> str:
        """Return the unit of the header value."""
        match self:
            case HeaderValue.SAMPLE_THETA:
                return "[deg]"
            case HeaderValue.BEAMLINE_ENERGY:
                return "[eV]"
            case HeaderValue.EPU_POLARIZATION:
                return "[deg]"
            case HeaderValue.HORIZONTAL_EXIT_SLIT_SIZE:
                return "[um]"
            case HeaderValue.HIGHER_ORDER_SUPPRESSOR:
                return "mm"
            case HeaderValue.EXPOSURE:
                return "s"

    def hdu(self) -> str:
        """Return the HDU name of the header value."""
        match self:
            case HeaderValue.SAMPLE_THETA:
                return "Sample Theta"
            case HeaderValue.BEAMLINE_ENERGY:
                return "Beamline Energy"
            case HeaderValue.EPU_POLARIZATION:
                return "EPU Polarization"
            case HeaderValue.HORIZONTAL_EXIT_SLIT_SIZE:
                return "Horizontal Exit Slit Size"
            case HeaderValue.HIGHER_ORDER_SUPPRESSOR:
                return "Higher Order Suppressor"
            case HeaderValue.EXPOSURE:
                return "EXPOSURE"

    def name(self) -> str:
        """Return the name of the header value with its unit."""
        return f"{self.hdu()} {self.unit()}"


type Motor = list[HeaderValue]

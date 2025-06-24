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
        if self == HeaderValue.SAMPLE_THETA:
            return "[deg]"
        elif self == HeaderValue.BEAMLINE_ENERGY:
            return "[eV]"
        elif self == HeaderValue.EPU_POLARIZATION:
            return "[deg]"
        elif self == HeaderValue.HORIZONTAL_EXIT_SLIT_SIZE:
            return "[um]"
        elif self == HeaderValue.HIGHER_ORDER_SUPPRESSOR:
            return "mm"
        elif self == HeaderValue.EXPOSURE:
            return "s"
        return ""

    def hdu(self) -> str:
        """Return the HDU name of the header value."""
        if self == HeaderValue.SAMPLE_THETA:
            return "Sample Theta"
        elif self == HeaderValue.BEAMLINE_ENERGY:
            return "Beamline Energy"
        elif self == HeaderValue.EPU_POLARIZATION:
            return "EPU Polarization"
        elif self == HeaderValue.HORIZONTAL_EXIT_SLIT_SIZE:
            return "Horizontal Exit Slit Size"
        elif self == HeaderValue.HIGHER_ORDER_SUPPRESSOR:
            return "Higher Order Suppressor"
        elif self == HeaderValue.EXPOSURE:
            return "EXPOSURE"
        return ""

    def display_name(self) -> str:
        """Return the name of the header value with its unit."""
        return f"{self.hdu()} {self.unit()}"


type Motor = list[HeaderValue]

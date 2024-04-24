from os import environ as env
from pathlib import Path
from typing import ClassVar
from warnings import warn

import numpy as np
import polars as pl

from pyref.core.exceptions import AppConfigError


def _parse_bool(value: bool | str) -> bool:
    return (
        value if isinstance(value, bool) else value.lower() in ("yes", "true", "t", "1")
    )


class Value[T]:
    """Class for representing a value with uncertainty."""

    def __init__(
        self,
        value: T,
        unit: str = "",
        std: T = 0.0,
    ):
        self.value = value
        self.std = std
        self.unit = unit

    def __repr__(self):
        return f"{self.value} ± {self.std} [{self.unit}]"

    def __str__(self):
        return self.__repr__()

    def __add__(self, other: T):
        """Add a number or value to the current value."""
        if isinstance(other, Value):
            assert self.unit == other.unit, "Units must be the same to add values"
            warn("Adding a scalar to a value is not recommended", stacklevel=2)
            return Value(
                self.value + other.value,
                self.unit,
                self.std + other.std,
            )
        else:
            warn("Adding a scalar to a value is not recommended", stacklevel=2)
            return Value(
                self.value + other,
                self.unit,
                self.std,
            )

    def __sub__(self, other: T):
        """Subtract a number or value from the current value."""
        if isinstance(other, Value):
            assert self.unit == other.unit, "Units must be the same to subtract values"
            return Value(
                self.value - other.value,
                self.unit,
                self.std + other.std,
            )
        else:
            warn("Subtracting a scalar from a value is not recommended", stacklevel=2)
            return Value(
                self.value - other,
                self.unit,
                self.std,
            )

    def __mul__(self, other: T):
        """Multiply a number or value by the current value."""
        if isinstance(other, Value):
            return Value(
                self.value * other.value,
                f"{self.unit}/{other.unit}",
                self.std + other.std,
            )
        else:
            warn("Multiplying a scalar with a value is not recommended", stacklevel=2)
            std_scale = (self.std / self.value) ** 2 + (other / self.value) ** 2
            return Value(
                self.value * other,
                self.unit,
                self.value * np.sqrt(std_scale),
            )

    def __truediv__(self, other: T):
        """Divide the current value by a number or value."""
        if isinstance(other, Value):
            return Value(
                self.value / other.value,
                f"{self.unit}/{other.unit}",
                self.std + other.std,
            )
        else:
            warn("Dividing a value by a scalar is not recommended", stacklevel=2)
            std_scale = (self.std / self.value) ** 2 + (other / self.value) ** 2
            return Value(
                self.value / other,
                self.unit,
                self.value * np.sqrt(std_scale),
            )

    def __pow__(self, other: T):
        """Raise the current value to a number or value."""
        if isinstance(other, Value):
            assert (
                self.unit == other.unit
            ), "Units must be the same to raise values to a power"
            return Value(
                self.value**other.value,
                self.unit,
                self.std * other.value * self.value ** (other.value - 1)
                + other.std * np.log(self.value) * self.value**other.value,
            )
        else:
            warn("Raising a value to a scalar is not recommended", stacklevel=2)
            return Value(
                self.value**other,
                self.unit,
                self.std * other * self.value ** (other - 1),
            )

    def __eq__(self, other: T):
        """Check if the current value is equal to a number or value."""
        if isinstance(other, Value):
            assert self.unit == other.unit, "Units must be the same to compare values"
            return self.value == other.value
        else:
            return self.value == other

    def __ne__(self, other: T):
        """Check if the current value is not equal to a number or value."""
        if isinstance(other, Value):
            assert self.unit == other.unit, "Units must be the same to compare values"
            return self.value != other.value
        else:
            return self.value != other

    def __lt__(self, other: T):
        """Check if the current value is less than a number or value."""
        if isinstance(other, Value):
            assert self.unit == other.unit, "Units must be the same to compare values"
            return self.value < other.value
        else:
            return self.value < other

    def __le__(self, other: T):
        """Check if the current value is less than or equal to a number or value."""
        if isinstance(other, Value):
            assert self.unit == other.unit, "Units must be the same to compare values"
            return self.value <= other.value
        else:
            return self.value <= other

    def __gt__(self, other: T):
        """Check if the current value is greater than a number or value."""
        if isinstance(other, Value):
            assert self.unit == other.unit, "Units must be the same to compare values"
            return self.value > other.value
        else:
            return self.value > other

    def __ge__(self, other: T):
        """Check if the current value is greater than or equal to a number or value."""
        if isinstance(other, Value):
            assert self.unit == other.unit, "Units must be the same to compare values"
            return self.value >= other.value
        else:
            return self.value >= other

    def __abs__(self):
        """Return the absolute value of the current value."""
        return Value(
            abs(self.value),
            self.unit,
            self.std,
        )

    def __neg__(self):
        """Return the negation of the current value."""
        return Value(
            -self.value,
            self.unit,
            self.std,
        )

    def __pos__(self):
        """Return the current value."""
        return self

    def __round__(self, n: int | None = None):
        """Round the current value to n decimal places."""
        if n is None:
            # Rounds uncert to 1 sig fig and value to the same decimal place
            uncert = int(np.floor(np.log10(self.std)))
            value = int(np.round(self.value, -uncert))
            return Value(
                value,
                self.unit,
                round(self.std, -uncert),
            )
        else:
            return Value(
                round(self.value, n),
                self.unit,
                round(self.std, n),
            )


class AppConfig:
    """Class for managing the configuration of the PRSOXR package."""

    SOL: Value[float] = Value(299792458, "m/s")
    PLANK_JOULE: Value[float] = Value(6.6267015e-34, "J s")
    ECHARGE: Value[float] = Value(1.60217662e-19, "C")
    PLANK: Value[float] = PLANK_JOULE / ECHARGE
    METER_TO_ANGSTROM: Value = Value(1e10, "Å/m")

    # Fits file values
    FITS_HEADER: ClassVar[dict[str, str]] = {
        "Beamline Energy": "ENERGY",
        "Beam Current": "CURRENT",
        "Sample Theta": "THETA",
        "Higher Order Suppressor": "HOS",
        "EPU Polarization": "POL",
        "EXPOSURE": "EXPOSURE",
        "Horizontal Exit Slit Size": "HES",
        "Date": "DATE",
    }
    FITS_SCHEMA: ClassVar[dict[str, type]] = {
        "ENERGY": pl.Float64,
        "CURRENT": pl.Float64,
        "THETA": pl.Float64,
        "HOS": pl.Float64,
        "POL": pl.Categorical,
        "EXPOSURE": pl.Float64,
        "HES": pl.Float64,
        "DATE": pl.Datetime,
        "SAMPLE": pl.String,
        "SCAN_ID": pl.UInt16,
        "STITCH": pl.UInt8,
    }
    DATA_SHCHEMA: ClassVar[dict[str, type]] = {
        "SAMPLE": pl.String,
        "SCAN_ID": pl.UInt16,
        "STITCH": pl.UInt8,
    }
    VARIABLE_MOTORS: ClassVar[list[str]] = ["HOS", "HES", "EXPOSURE"]

    # PRSOXR API values
    SAVE_BACKEND: str = "parquet"

    # TEST values
    DB: ClassVar[Path] = Path(
        "C:/Users/hduva/Washington State University (email.wsu.edu)/Carbon Lab Research Group - Documents/Harlan Heilman/.refl/.db/"
    )
    DATA_DIR: ClassVar[Path] = DB / "test"

    """
    Map environment variables to class attributes
     * If the environment variable is not set, the default value is used from above
     * If the environment variable is set, the value is converted to the type of the
     default value
    """

    def __init__(self) -> None:
        # Add all class attributes to the environment
        for field in self.__annotations__:
            if not field.isupper():
                continue

            default_value = getattr(self, field)

            # Raise AppConfigError if the enviroment variable is missing
            if default_value is None and env.get(field) is None:
                error_message = f"Environment variable {field} is not set"
                raise AppConfigError(error_message)

            # Save the configuration values to the environment
            env[field] = str(default_value) if default_value is not None else ""

    def __repr__(self) -> str:
        head = "PRSOXR environment variables:\n"
        body = "\n".join(
            f"  {field:<20}  {env.get(field):>30}"
            for field in self.__annotations__
            if field.isupper()
        )
        return head + body

    def __getattr__(self, name: str) -> str | None:
        return env.get(name)

    def __setattr__(self, name: str, value: str) -> None:
        if name.isupper():
            error_message = f"Cannot set attribute {name} of AppConfig"
            raise AttributeError(error_message)
        env.setdefault(name, value)

    def __dir__(self) -> list[str]:
        return [field for field in self.__annotations__ if not field.isupper()]

    def __getitem__(self, name: str) -> str:
        return getattr(self, name)


if __name__ == "__main__":
    config = AppConfig()
    print(config)

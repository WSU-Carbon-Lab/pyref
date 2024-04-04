r"""
Module Containing Type Definitions.

This module contains type definitions for use in the pyref package.
"""

from pathlib import Path
from warnings import warn

import numpy as np

type DataDirectory = str | Path


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
        return f"{self.value} ± {self.std} {self.unit}"

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

from os import environ as env
from typing import ClassVar

from pyref.core.exceptions import AppConfigError
from pyref.core.types import Value


def _parse_bool(value: bool | str) -> bool:
    return (
        value if isinstance(value, bool) else value.lower() in ("yes", "true", "t", "1")
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
    }
    VARIABLE_MOTORS: ClassVar[list[str]] = ["HOS", "HES", "EXPOSURE"]

    # PRSOXR API values
    SAVE_BACKEND: str = "parquet"

    """
    Map environment variables to class attributes
     * If the environment variable is not set, the default value is used from above
     * If the environment variable is set, the value is converted to the type of the default value
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

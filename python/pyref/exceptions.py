"""Exceptions for the pyref package."""


class UnknownPolarizationError(Exception):
    """Exception raised for unknown polarization values."""

    def __init__(self, message="Unknown polarization"):
        self.message = message
        super().__init__(self.message)


class OverlapError(Exception):
    """Exception raised when the overlap between two datasets is too small."""

    def __init__(
        self,
        message="No overlap found between datasets",
        selected=None,
        prior=None,
        current=None,
    ):
        self.message = message
        if prior is not None and current is not None:
            self.message += f"\nPrior: {prior.tail(10)}\nCurrent: {current.head(10)}"
            self.message += f"\nSelected: {selected}"
        super().__init__(self.message)

"""Log Prior Constraint for the fitting of reflectometry data."""

import numpy as np


class LogpExtra:
    """Log Prior Constraint for the fitting of reflectometry data."""

    def __init__(self, objective):
        self.objective = objective

    def __call__(self, model, data):
        """Apply custom log-prior constraint."""
        for pars in self.objective.parameters:
            thick_pars = sort_pars(pars.flattened(), "thick")
            rough_pars = sort_pars(pars.flattened(), "rough")
            for i in range(len(rough_pars)):
                if rough_pars[i].vary or thick_pars[i].vary:
                    interface_limit = np.sqrt(2 * np.pi) * rough_pars[i].value / 2
                    if float(thick_pars[i].value - interface_limit) < 0:
                        return -np.inf
        return 0


def sort_pars(pars, str_check, vary=None, str_not=" "):
    """Sort parameters based on the string in the name."""
    return [
        par
        for par in pars
        if str_check in par.name
        and str_not not in par.name
        and (vary is None or par.vary == vary)
    ]

"""Log Prior Constraint for the fitting of reflectometry data."""

from typing import Literal

from numpy import inf, pi, sqrt

type Parameter = Literal["thick_rough", "birefringence", "delta"]


class LogpExtra:
    """Log Prior Constraint for the fitting of reflectometry data."""

    def __init__(self, objective):
        # we'll store the parameters and objective in this object
        # this will be necessary for pickling in the future
        self.objective = objective  ##Full list of parameters

    def __call__(self, model, data):
        """Apply custom log-prior constraint."""
        ##First constraint condition ---
        ##Load Parameters of interest for each objective
        for pars in self.objective.parameters:
            thick_pars = sort_pars(pars.flattened(), "thick")
            rough_pars = sort_pars(pars.flattened(), "rough")
            ##Check that the roughness is not out of control
            for i in range(len(rough_pars)):  ##Sort through the # of layers
                if (
                    rough_pars[i].vary or thick_pars[i].vary
                ):  # Only constrain parameters that vary
                    interface_limit = (
                        sqrt(2 * pi) * rough_pars[i].value / 2
                    )  # (rough_pars[i].value/(np.sqrt(2*np.pi)))**2
                    if float(thick_pars[i].value - interface_limit) < 0:
                        return -inf
        return 0  ##If all the layers are within the constraint return 0


def sort_pars(pars, str_check, vary=None, str_not=" "):
    """Sort parameters based on the string in the name."""
    temp = []
    num = len(pars)
    for i in range(num):
        if str_check in pars[i].name and str_not not in pars[i].name:
            if vary is True:
                if pars[i].vary is True:
                    temp.append(pars[i])
            elif vary is False:
                if pars[i].vary is False:
                    temp.append(pars[i])
            else:
                temp.append(pars[i])
    return temp

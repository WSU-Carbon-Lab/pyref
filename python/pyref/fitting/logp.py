from typing import Literal

from numpy import inf, pi, sqrt

type Parameter = Literal["thick_rough", "birefringence", "delta"]


class LogpExtra:
    def __init__(
        self,
        objective,
        surface_label: str = "surf",
        constraints: list[Parameter] | None = None,
    ):
        # we'll store the parameters and objective in this object
        # this will be necessary for pickling in the future
        if constraints is None:
            constraints = ["thick_rough", "birefringence", "delta"]
        self.objective = objective  ##Full list of parameters
        self.pars = self.objective.parameters.flattened()
        self.terms = [
            # value, not
            ("thick", " "),
            ("rough", " "),
            ("bire", " "),
            ("diso", " "),
            ("_zz", str(surface_label)),
        ]
        self.constraints = constraints

    def __call__(self, model, data):
        ##First constraint condition ---
        ##Load Parameters of interest for each objective

        self.sorted_pars = {
            par: sort_pars(self.pars, par, str_not=not_in)
            for (par, not_in) in self.terms
        }
        if "thick_rough" in self.constraints:
            self.thick_rough_constraint()
        if "birefringence" in self.constraints:
            self.birefringence_constraint()
        # if "delta" in self.constraints:
        #     self.delta_constraint()

        return 0

    def thick_rough_constraint(self):
        thick_pars = self.sorted_pars["thick"]
        rough_pars = self.sorted_pars["rough"]
        for i in range(len(rough_pars)):  ##Sort through the # of layers
            if (
                rough_pars[i].vary or thick_pars[i].vary
            ):  # Only constrain parameters that vary
                interface_limit = (
                    sqrt(2 * pi) * rough_pars[i].value / 2
                )  # (rough_pars[i].value/(np.sqrt(2*np.pi)))**2
                if (
                    float(thick_pars[i].value - interface_limit) < 0
                ):  # If the interface width is above the corresponding thickness, set logp to -inf
                    return -inf

    def birefringence_constraint(self):
        bire_pars = self.sorted_pars["bire"]
        delta_val = self.sorted_pars["diso"]
        self.delta_val = delta_val
        for bire in bire_pars:
            if delta_val[0] > 0:
                if (
                    float(bire - 3 * delta_val[0]) > 0
                    or float(bire + 3 * delta_val[0] / 2) < 0
                ):
                    return -inf
            if delta_val[0] < 0:
                if (
                    float(bire - 3 * delta_val[0]) < 0
                    or float(bire + 3 * delta_val[0] / 2) > 0
                ):
                    return -inf
            if bire_pars[0].value > 0:
                return -inf


class LogpExtra_rough:
    def __init__(self, objective):
        # we'll store the parameters and objective in this object
        # this will be necessary for pickling in the future
        self.objective = objective  ##Full list of parameters

    def __call__(self, model, data):
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
                    if (
                        float(thick_pars[i].value - interface_limit) < 0
                    ):  # If the interface width is above the corresponding thickness, set logp to -inf
                        return -inf

        return 0  ##If all the layers are within the constraint return 0

    ##Function to sort through ALL parameters in an objective and return based on name keyword
    ##Returns a list of parameters for further use


def sort_pars(pars, str_check, vary=None, str_not=" "):
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

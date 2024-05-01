from typing import Literal

type Parameter = Literal["thick_rough", "birefringence", "delta"]


class LogpExtra(object):
    def __init__(
        self,
        objective,
        surface_label: str = "surf",
        constraints: list[Parameter] = ["thick_rough", "birefringence", "delta"],
    ):
        # we'll store the parameters and objective in this object
        # this will be necessary for pickling in the future
        self.objective = objective  ##Full list of parameters
        self.pars = self.objective.parameters.flattened()
        self.terms = [
            # value, not
            ("thick", " ")("rough", " ")("bire", " ")("diso", " ")("_zz", "surf")
        ]

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
        if "delta" in self.constraints:
            self.delta_constraint()

        return 0

    def thick_rough_constraint(self):
        thick_pars = self.sorted_pars["thick"]
        rough_pars = self.sorted_pars["rough"]
        for i in range(len(rough_pars)):  ##Sort through the # of layers
            if (
                rough_pars[i].vary or thick_pars[i].vary
            ):  # Only constrain parameters that vary
                interface_limit = (
                    np.sqrt(2 * np.pi) * rough_pars[i].value / 2
                )  # (rough_pars[i].value/(np.sqrt(2*np.pi)))**2
                if (
                    float(thick_pars[i].value - interface_limit) < 0
                ):  # If the interface width is above the corresponding thickness, set logp to -inf
                    return -np.inf

    def birefringence_constraint(self):
        bire_pars = self.sorted_pars["bire"]
        delta_val = self.sorted_pars["diso"]
        for bire in bire_pars:
            if bire.vary and delta_val[0] > 0:
                if (
                    float(bire - 3 * delta_val[0]) > 0
                    or float(bire + 3 * delta_val[0] / 2) < 0
                ):
                    return -np.inf
            if bire.vary and delta_val[0] < 0:
                if (
                    float(bire - 3 * delta_val[0]) < 0
                    or float(bire + 3 * delta_val[0] / 2) > 0
                ):
                    return -np.inf

    def delta_constraint(self):
        delta_val = self.sorted_pars["diso"]
        zz_val = self.sorted_pars["_zz"]
        # Remove the surface values
        for i in range(len(delta_val)):
            if delta_val[i].vary and "surf" not in delta_val[i].name:
                if delta_val[i].value < zz_val.value:
                    return -np.inf


class LogpExtra_rough(object):
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
                        np.sqrt(2 * np.pi) * rough_pars[i].value / 2
                    )  # (rough_pars[i].value/(np.sqrt(2*np.pi)))**2
                    if (
                        float(thick_pars[i].value - interface_limit) < 0
                    ):  # If the interface width is above the corresponding thickness, set logp to -inf
                        return -np.inf

        return 0  ##If all the layers are within the constraint return 0

    ##Function to sort through ALL parameters in an objective and return based on name keyword
    ##Returns a list of parameters for further use


def sort_pars(pars, str_check, vary=None, str_not=" "):
    temp = []
    num = len(pars)
    for i in range(num):
        if str_check in pars[i].name and str_not not in pars[i].name:
            if vary == True:
                if pars[i].vary == True:
                    temp.append(pars[i])
            elif vary == False:
                if pars[i].vary == False:
                    temp.append(pars[i])
            else:
                temp.append(pars[i])
    return temp

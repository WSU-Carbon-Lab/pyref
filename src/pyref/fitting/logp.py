import numpy as np


class LogpExtra(object):
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
            bire_pars = sort_pars(pars.flattened(), "bire")
            delta_val = sort_pars(pars.flattened(), "dt", vary=True)
            zz_pars = sort_pars(pars.flattened(), "zz")
            xx_pars = sort_pars(pars.flattened(), "xx")
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
            # Check to see if the Birefringence is physical based on current trace
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
            # check if xx is less than delta_val or zz is greater than delta_val
            for i in range(len(zz_pars)):
                if zz_pars[i].vary or xx_pars[i].vary:
                    if (
                        zz_pars[i].value > delta_val[0]
                        or xx_pars[i].value < delta_val[0]
                    ):
                        return -np.inf

        return 0  ##If all the layers are within the constraint return 0


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


class LogpExtra_rough(object):
    def __init__(self, objective):
        # we'll store the parameters and objective in this object
        # this will be necessary for pickling in the future
        self.objective = objective  ##Full list of parameters

    def __call__(self, model, data):
        ##First constraint condition ---
        ##Load Parameters of interest for each objective
        pars = self.objective.parameters.flattened()
        params_to_check = [
            ("thick", " "),
            ("rough", " "),
            ("_zz", "surf"),
            ("diso", "surf"),
        ]
        sorted_pars = {
            param: sort_pars(pars, param, not_check=notin)
            for (param, notin) in params_to_check
        }
        thick_pars = sorted_pars["thick"]
        rough_pars = sorted_pars["rough"]
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

        delta_zz = sorted_pars["_zz"]
        diso = sorted_pars["diso"]
        for dz, di in zip(delta_zz, diso):
            if dz.vary or di.vary:
                if dz.value > di.value:
                    return -np.inf

        return 0  ##If all the layers are within the constraint return 0

    ##Function to sort through ALL parameters in an objective and return based on name keyword
    ##Returns a list of parameters for further use


def sort_pars(pars, str_check, vary=None, not_check=" "):
    temp = []
    num = len(pars)
    for i in range(num):
        if str_check in pars[i].name and not_check not in pars[i].name:
            if vary == True:
                if pars[i].vary == True:
                    temp.append(pars[i])
            elif vary == False:
                if pars[i].vary == False:
                    temp.append(pars[i])
            else:
                temp.append(pars[i])
    return temp

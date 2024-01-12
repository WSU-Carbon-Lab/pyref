import json

from pypxr.structure import *
from refnx.analysis import CurveFitter, Interval, Objective, Transform
from refnx.dataset import ReflectDataset

lowest_params = [
    "name",
    "_value",
    "_bounds",
    "_deps",
    "units",
]


def serialize(params):
    """
    Serialize a Parameters object to a dictionary.

    Parameters
    ----------
    params : Parameters
        The Parameters object to be serialized.

    Returns
    -------
    dict
        A dictionary containing the serialized Parameters object.
    """
    if isinstance(params, Parameter):
        params_dict = {}
        for param in lowest_params:
            if param == "_bounds":
                params_dict[param] = (str(params.bounds.lb), str(params.bounds.ub)),
            else:
                params_dict[param] = getattr(params, param)
        return params_dict
    
    elif isinstance(params, Parameters):
        params_dict = {}
        for param in params:
            params_dict[param.name] = serialize(param)
        return params_dict
    
    elif isinstance(params, dict):
        params_dict = {}
        for key, value in params.items():
            params_dict[key] = value
        return params_dict



def save_params(refnx_object, filename):
    """
    Save the parameters of a refnx object to a file.

    Parameters
    ----------
    refnx_object : refnx object
        The object whose parameters are to be saved.
    filename : str
        The name of the file to save the parameters to.
    """
    params = refnx_object.parameters
    params_dict = serialize(params)
    with open(filename, 'w') as f:
        json.dump(params_dict, f, indent=4)

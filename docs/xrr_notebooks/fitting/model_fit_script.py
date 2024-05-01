from pathlib import Path
from sys import path as syspath

syspath.append(str(Path().home() / "pyref" / "src"))

import pickle as pkl
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyref as rf
import seaborn as sns
from pypxr.reflectivity import *
from pypxr.structure import *
from pyref.style import science
from refnx._lib.emcee.moves.de import DEMove
from refnx.analysis import CurveFitter, GlobalObjective, Objective, Transform
from refnx.dataset import ReflectDataset

# Import and setup two "Long" data sets

db = rf.db()

znpc_oc = db.get_oc("C32H16N8Zn")
znpc_mono = {
    "283.7": {
        "s": db.get_refl("ZnPc_283.7_100.0 (CCD Scan 82865).parquet", "ZnPc"),
        "p": db.get_refl("ZnPc_283.7_190.0 (CCD Scan 82869).parquet", "ZnPc"),
    },
    "284.1": {
        "s": db.get_refl("ZnPc_284.0_100.0 (CCD Scan 82865).parquet", "ZnPc"),
        "p": db.get_refl("ZnPc_284.0_190.0 (CCD Scan 82869).parquet", "ZnPc"),
    },
}

data = []
en: list[float] = [float(k) for k in znpc_mono.keys() if k != "250"]

print("Data Loaded from Database")


def mask(s, p):
    s_pol = s[(s["Q"] != 0.0757)][s["Q"] != 0.0740]
    p_pol = p[(p["Q"] != 0.0757)][p["Q"] != 0.0740]
    return s_pol.iloc[10:], p_pol.iloc[10:]


for i, e in enumerate(znpc_mono.keys()):
    if e == "250":
        pass
    cutoff = 7 if i <= 3 else 8
    s = znpc_mono[f"{e}"]["s"].iloc[cutoff:]
    p = znpc_mono[f"{e}"]["p"].iloc[cutoff:]
    s, p = mask(znpc_mono[f"{e}"]["s"], znpc_mono[f"{e}"]["p"])
    data.append(rf.to_refnx_dataset(s, pol="sp", second_pol=p))

print("Data Masked")

from typing import Literal

from numpy import array, ndarray

type _bound = tuple[float, float]
type complex_bound = dict[Literal["r", "i"], _bound]
type bound = dict[Literal["xx", "zz"], complex_bound]

# Collect structure parameters
znpc_mono_struct_file = db.get_struct("ZnPc_RoomTemp")
znpc_oc = db.get_oc("C32H16N8Zn")

si_thick = znpc_mono_struct_file["Si"]["thickness"]
si_rough = znpc_mono_struct_file["Si"]["roughness"]
si_density = znpc_mono_struct_file["Si"]["density"]

sio2_thick = znpc_mono_struct_file["SiO2"]["thickness"]
sio2_rough = znpc_mono_struct_file["SiO2"]["roughness"]
sio2_density = znpc_mono_struct_file["SiO2"]["density"]

c_amor_thick = znpc_mono_struct_file["C"]["thickness"]
c_amor_rough = znpc_mono_struct_file["C"]["roughness"]
c_amor_density = znpc_mono_struct_file["C"]["density"]

znpc_thick = znpc_mono_struct_file["C32H16N8Zn"]["thickness"]
znpc_rough = znpc_mono_struct_file["C32H16N8Zn"]["roughness"]
znpc_density = znpc_mono_struct_file["C32H16N8Zn"]["density"]


def n_limits(e, density) -> tuple[list[bound], list[ndarray]]:
    bounds: list[complex_bound] = []
    ns = znpc_oc(e, density=density)
    if not isinstance(ns, list):
        ns = [ns]
    for i in e:
        xx_r = (znpc_oc.zz(i), znpc_oc.xx(i))
        xx_i = (znpc_oc.ixx(i), znpc_oc.izz(i))

        zz_r = xx_r
        zz_i = xx_i

        xx_bound: complex_bound = {"r": xx_r, "i": xx_i}
        zz_bound: complex_bound = {"r": zz_r, "i": zz_i}
        bounds.append({"xx": xx_bound, "zz": zz_bound})

    return bounds, ns


bounds, _ = n_limits(en, znpc_density)
ns = [
    np.asarray([-0.000784 + 0.000627j, -0.000144 + 0.000387j]),
    np.asarray([-0.001457 + 0.001307j, -0.000317 + 0.000387j]),
]

print("Generating Structure...")

vac = [PXR_MaterialSLD("", 0, e)(0, 0) for e in en]

znpc_surf_thick = 10
oxide = [
    PXR_MaterialSLD("C", c_amor_density, e, name="Oxide")(znpc_surf_thick, znpc_rough)
    for e in en
]
surf = [
    PXR_SLD(n, symmetry="uni", name="Surface")(znpc_surf_thick, znpc_rough) for n in ns
]
bulk = [
    PXR_SLD(n, symmetry="uni", name="Bulk")(
        znpc_thick - znpc_surf_thick, znpc_rough / 2
    )
    for n in ns
]
patern = [
    PXR_SLD(n, symmetry="uni", name="Patern")(znpc_surf_thick, znpc_rough) for n in ns
]
c_ordered = [
    PXR_SLD(n, symmetry="uni", name="C_Ordered")(c_amor_thick, c_amor_rough) for n in ns
]
c_amor = [
    PXR_MaterialSLD("C", c_amor_density, e, name="C_Amorphous")(
        c_amor_thick, c_amor_rough
    )
    for e in en
]
si = [PXR_MaterialSLD("Si", 2.33, e, name="Si")(si_thick, si_rough) for e in en]
sio2 = [
    PXR_MaterialSLD("SiO2", sio2_density, e, name="SiO2")(sio2_thick, sio2_rough)
    for e in en
]


for slab in si:
    slab.thick.setp(vary=False)
    slab.rough.setp(vary=False)

for slab in sio2:
    slab.thick.setp(vary=None, constraint=sio2_thick)
    slab.rough.setp(vary=None, constraint=sio2_rough)
    slab.sld.density.setp(vary=False, bounds=slab.sld.density.value * array([0.8, 1.2]))

for slab in c_amor:
    slab.thick.setp(vary=True, bounds=(0, znpc_surf_thick))
    slab.rough.setp(vary=True, bounds=(0, znpc_rough))
    slab.sld.density.setp(vary=True, bounds=slab.sld.density.value * array([0.8, 1.2]))

for slab in oxide:
    slab.thick.setp(vary=True, bounds=(0, znpc_surf_thick))
    slab.rough.setp(vary=True, bounds=(0, znpc_rough))
    slab.sld.density.setp(vary=True, bounds=slab.sld.density.value * array([0.5, 1.2]))


# Applied for the znpc layers
def sld_constraint(
    slab: PXR_Slab,
    bounds: bound | None = None,
    xx_bounds: _bound = (-0.005, 0.005),
    zz_bounds: _bound = (-0.005, 0.005),
    thick_bounds: _bound = None,
):
    if bounds is None:
        bounds = {
            "xx": {"r": xx_bounds, "i": xx_bounds},
            "zz": {"r": zz_bounds, "i": zz_bounds},
        }
    slab.sld.xx.setp(vary=True, bounds=bounds["xx"]["r"])
    slab.sld.zz.setp(vary=True, bounds=bounds["zz"]["r"])
    slab.sld.ixx.setp(vary=True, bounds=bounds["xx"]["i"])
    slab.sld.izz.setp(vary=True, bounds=bounds["zz"]["i"])
    if thick_bounds is not None:
        slab.thick.setp(vary=True, bounds=thick_bounds)
    else:
        slab.thick.setp(vary=True, bounds=(0, znpc_thick))
    slab.rough.setp(vary=True, bounds=(0, znpc_rough))


_ = [sld_constraint(slab, bounds=bound) for (slab, bound) in zip(bulk, bounds)]
_ = [sld_constraint(slab, bounds=bound) for (slab, bound) in zip(surf, bounds)]
_ = [sld_constraint(slab, bounds=bound) for (slab, bound) in zip(c_ordered, bounds)]

for i, slab in enumerate(bulk):
    slab.thick.setp(vary=None, constraint=znpc_thick - surf[i].thick.value)


substrate = [sio2[i] | si[i] for i in range(len(en))]

"""
1. Surface - Ordered Bulk - Patern - Carbon - SiO2 - Si
2. Surface - Ordered Bulk - Carbon (With Ordering) - SiO2 - Si
3. Surface - Ordered Bulk - Carbon - SiO2 - Si
"""
from copy import deepcopy
from tkinter import font

struc_1 = [
    vac[i] | surf[i] | bulk[i] | patern[i] | c_amor[i] | substrate[i]
    for i in range(len(en))
]
struc_2 = [
    vac[i] | surf[i] | bulk[i] | c_ordered[i] | substrate[i] for i in range(len(en))
]
struc_3 = [
    vac[i] | surf[i] | bulk[i] | c_amor[i] | substrate[i] for i in range(len(en))
]

strucs = [struc_1, struc_2, struc_3]

# Apply multi energy thickness and roughness constraints

print("Applying multi-energy constraints...")


def multi_energy_constraints(struc: list[PXR_Structure]):
    variable_slabs = struc[0].data
    const_slabs = struc[1].data

    for vary, const in zip(variable_slabs, const_slabs):
        if const.thick.vary:
            if const.name == "Bulk":
                vary.thick.setp(
                    vary=None,
                    constraint=znpc_thick
                    - np.sum([slab.thick.value for slab in variable_slabs]),
                )
            const.thick.setp(vary=None, constraint=vary.thick.value)
            const.rough.setp(vary=None, constraint=vary.rough.value)
            if const.name == "SiO2":
                const.sld.density.setp(vary=None, constraint=vary.sld.density.value)
            elif "C_Amorphous" in const.name:
                const.sld.density.setp(vary=None, constraint=vary.sld.density.value)


_ = [multi_energy_constraints(struc) for struc in strucs]

from pyref.fitting.logp import LogpExtra_rough as LogpExtra

global_objs = []
for i, struc in enumerate(strucs):
    objs = []
    for i, e in enumerate(en):
        model = PXR_ReflectModel(struc[i], pol="sp", energy=e, name=f"Struc{i}-{e}eV")
        model.scale_s.setp(vary=True, bounds=(0.6, 1.2))
        model.scale_p.setp(vary=True, bounds=(0.6, 1.2))
        model.theta_offset_s.setp(vary=True, bounds=(-1, 1))
        model.theta_offset_p.setp(vary=True, bounds=(-1, 1))

        obj_p = Objective(model, data[i], transform=Transform("logY"))
        lpe = LogpExtra(obj_p)
        obj_p.logp_extra = lpe
        obj_p.plot(resid=True)
        objs.append(obj_p)
    global_obj = GlobalObjective(objs)
    global_objs.append(global_obj)

import pickle as pkl


class Fitter:
    # Suppresses the warning
    from warnings import simplefilter

    simplefilter("ignore")

    def __init__(
        self, obj: Objective | GlobalObjective, en, walkers_per_param=10, burn_in=0
    ):
        self.obj = obj
        self.move = [
            (DEMove(sigma=1e-7), 0.90),
            (DEMove(sigma=1e-7, gamma0=1), 0.1),
        ]
        if obj.__class__.__name__ == "GlobalObjective":
            self.n_params = sum(
                [
                    len(o.data.data[0]) - len(o.varying_parameters())
                    for o in obj.objectives
                ]
            )
        else:
            self.n_params = len(self.obj.data.data[0]) - len(
                self.obj.varying_parameters()
            )
        self._n_walkers = walkers_per_param * len(self.obj.varying_parameters())
        self.burn_in = burn_in
        self.fitter = CurveFitter(obj, nwalkers=self._n_walkers, moves=self.move)
        self.en = en

    @property
    def n_walkers(self):
        return self._n_walkers

    @n_walkers.setter
    def n_walkers(self, value):
        self._n_walkers = value
        self.fitter = CurveFitter(self.obj, nwalkers=self._n_walkers, moves=self.move)

    def red_chisqr(self):
        try:
            return self.obj.chisqr() / self.n_params
        except:
            return np.nan

    def fit(
        self,
        steps_per_param=20,
        thin=1,
        seed=1,
        init: Literal["jitter", "prior"] = "jitter",
        show_output=False,
    ):
        steps = steps_per_param * self.n_params
        burn = int(steps * self.burn_in)

        self.fitter.initialise(init, random_state=seed)
        self.chain = self.fitter.sample(
            steps,
            random_state=seed,
            nthin=thin,
            skip_initial_state_check=True,
            nburn=burn,
        )
        print(f"Reduced χ2 = {self.red_chisqr()}")
        if show_output:
            self.show_output()
            self.export(f"{self.en}.pkl")

    def show_output(self):
        print(self.obj.varying_parameters())
        fig, ax = plt.subplots()
        lp = self.fitter.logpost
        ax.plot(-lp)
        if self.obj.__class__.__name__ == "GlobalObjective":
            for o in self.obj.objectives:
                o.plot(resid=True)
        else:
            self.obj.plot(resid=True)
        self.export(f"{self.en}.pkl")

    def export(self, filename: str):
        with open(filename, "wb") as f:
            pkl.dump(self, f)

    def delete_obj(self):
        del self


print("Fitting...")
for i, obj in enumerate(global_objs):
    fitter = Fitter(obj, en)
    chi_sqr = fitter.red_chisqr()
    chi_sqrs = [chi_sqr]
    depth = 0
    spp = 5
    print(f"Starting fit for structure {i+1}...")
    while chi_sqr > 1.2:
        fitter.fit(init="jitter", steps_per_param=spp, show_output=False)
        chi_sqr = fitter.red_chisqr()
        chi_sqrs.append(chi_sqr)
        depth += 1
        spp += 5
        if depth > 3:
            fitter.fit(init="jitter", steps_per_param=1, show_output=True)
            break
        if np.abs(chi_sqrs[-1] - chi_sqrs[-2]) <= 0.1:
            fitter.fit(init="jitter", steps_per_param=1, show_output=True)
            break
        cache = fitter.red_chisqr()
    else:
        fitter.fit(init="jitter", steps_per_param=1, show_output=True)

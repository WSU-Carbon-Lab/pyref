"""

Input file for running MCMC fits with refnx structure
Updated 06/09/2021 for better scripting

"""
# Basic import items --
import os.path
# For saving MCMC CurveFitters and Objectives
import pickle
import sys

import numpy as np

# Import PyPXR through GitHub
sys.path.append("C:/Users/hduva/CarbonLab/P-RSoXR/src/pypxr") # Exact path for laptop
from mcmc_analysis.export_results import export_mcmc_summary
# Additional support functions
from mcmc_analysis.utilities import (LogpExtra_rough, build_tensor,
                                     compile_data_hdf5, load_prsoxr_hdf5)
from reflectivity import *
from refnx._lib.emcee.moves.de import *  # Differential Evolution MCMC move - typically used in these fits
from refnx.analysis import (CurveFitter, GlobalObjective,  # For fitting
                            Objective, Transform)
# Required modules from Refnx
from refnx.dataset import ReflectDataset  # Object used to define data
from structure import *

print('Initialization Complete')

"""
**********************
User inputs begin here
**********************

Info on sample to load:


mypath : str
    Path to the hdf5 file to be loaded.
myfile : str
    Name of the hdf5 file to load
myhdf5 : dictionary
    HDF5 file imported through load_prsoxr_hdf5
   
data_f : str
    Name of the file to be loaded. Should be an HDF5 file that contains all energies/polarizations for the sample
load_f : dict.
    Composite dictionary of all data loaded from HDF5 file. Separated by keys w/ energy and polarization.
"""

mypath = 'C:/Users/tjf2/Documents/NRC_Projects/HDF5 Storage/'
myfile = 'MF147A.hdf5'

# Select energies and polarizations to be imported
en_list = [284.4, 285.8]
pol_list = [100, 190]
# Apply energy corrections if needed:
energy_offset = 0
en_array = np.array(en_list)
en_array += energy_offset

myhdf5 = load_prsoxr_hdf5(myfile, mypath)
mydata, mymask = compile_data_hdf5(myhdf5, en_list=en_list, pol_list=pol_list, concat=True)
"""
Space for masking
"""




"""
Build ReflectDataSets and apply masks
"""
dzip = zip(mydata, mymask)
reflect_datasets = []
for data, mask in dzip:
    reflect_datasets.append(ReflectDataset(data.T, mask=mask))

"""
Information on Output:
"""
name_save = 'MF147A_res'
path_save = 'C:/Users/tjf2/Documents/NRC_Projects/Reflectivity/PSoXR_MixedPhase_Modeling/PureMat_Reflectivity/' + name_save


print('Data successfully loaded')

"""
------------
#### Generate anisotropic SLD objects for slab assignments

    'SLD' = PXR_SLD(np.ndarray(3,3), name)
          = PXR_MaterialSLD(chemical formula, density, energy, name)
          = PXR_NexafsSLD(Nexafs file, energy, name)

     Isotropic, Uniaxial, and Biaxial (not verified) are supported
"""

"""
Initial conditions for the model: ENERGY DEPENDENT PARAMETERS
    Each element in the list is the same parameter for a separate energy
"""
dsa_delta = [-0.003285347, 0.000833589]
dsa_beta = [0.001766276, 0.001690298]

bulk_bire = [-0.000154791, -0.000337602]
bulk_dichro = [0.000154791, 0.000337602]

surface_bire = [-0.00005, -0.00003]
surface_dichro = [0.00005, 0.00003] 

substrate_bire = [-0.0001, -0.0003]
substrate_dichro = [0.0001, 0.0003] 

# Construct full tensor using utilities function
dsa_bulk = build_tensor(zip(dsa_delta, dsa_beta, bulk_bire, bulk_dichro))
dsa_surface = build_tensor(zip(dsa_delta, dsa_beta, surface_bire, surface_dichro))
dsa_substrate = build_tensor(zip(dsa_delta, dsa_beta, substrate_bire, substrate_dichro))

ratio_bire_dichro = [1.06828, -1.0421] # From Angle dependent NEXAFS
ratio_bire_en = [1, -0.867908] # From Angle dependent NEXAFS


"""
Initial conditions for the model: ENERGY INDEPENDENT PARAMETERS
    Each element of the list is a structural parameter that is held between energies
"""
#Parameters from alternative measurements
surface_rough = 5 #[A]
total_thick = 700 #[A]

sio2_t = 15 #[A]
sio2_r = 5 #[A]
si_t = 0 #[A] (Can be used as placeholder for total thickness if it should vary)
si_r = 1.5 #[A]

l_t = [10, 680, 10] #Layer thickness [A]
l_r = [2, 5, surface_rough] #Layer roughness [A]

"""
Create 'slab' objects that represent a layer in the model

'slab' = 'SLD'(thickness, roughness)

    --Substrate thickness can be any value (not used in calculation) 
    --Roughness corresponds to the top of the layer, I.E. The side closest to the superstrate
    --Superstrate is not assigned through a 'slab' ~~See structure generation
"""
"""
Build slab conditions from tensor objects made previously

"""
# Superstrate
vac = PXR_MaterialSLD('', density=1, name="vac")
vac_slab = vac(0,0)

# Substrate Lists
sio2_slabs = []
si_slabs = []
# Substrate Layers
for i, en in enumerate(en_array):
    sio2_slabs.append(
        PXR_MaterialSLD('SiO2', density = 2.4, energy = en, name='SiO2')(
            sio2_t, sio2_r
        )     
    )
    si_slabs.append(
        PXR_MaterialSLD('Si', density=2.33, energy = en,name='Si')(
            si_t, si_r
        )
    )
        
# Sample Layers
dsa_bulk_slabs = []
dsa_surface_slabs = []
dsa_substrate_slabs = []

for i, en in enumerate(en_array):
    dsa_bulk_slabs.append(
        PXR_SLD(dsa_bulk[i], name=('dsaph_bulk_en'+str(i)))(
            l_t[1], l_r[1]
        )
    )
    dsa_surface_slabs.append(
        PXR_SLD(dsa_surface[i], name=('dsaph_surface_en'+str(i)))(
            l_t[2], l_r[2]
        )
    )
    dsa_substrate_slabs.append(
        PXR_SLD(dsa_substrate[i], name=('dsaph_substrate_en'+str(i)))(
            l_t[0], l_r[0]
        )
    )
    
"""
Generate the structure model
    'structure' = superstrate (sld) | top layer (slab) | ... | bottom layer (slab) | substrate (slab)
"""
structures = []
for i in range(len(en_array)):
    structures.append(
        vac | dsa_surface_slabs[i] | dsa_bulk_slabs[i] | dsa_substrate_slabs[i] | sio2_slabs[i] | si_slabs[i]
    )
    structures[i].name = 'dsa_3layer'+str(en_array[i])


"""
Space for opening parameters and setting constraints
"""
# Substrates
for i, slab in enumerate(sio2_slabs):
    # Conditions
    vary_exp = True if i == 0 else None
    thick_constraint_exp = None if i == 0 else sio2_slabs[0].thick
    rough_constraint_exp = None if i == 0 else sio2_slabs[0].rough

    # Variables
    slab.thick.setp(vary=vary_exp, bounds=(1, 25), constraint=thick_constraint_exp)
    slab.rough.setp(vary=vary_exp, bounds=(1, 25), constraint=rough_constraint_exp)
    slab.sld.density.setp(vary=False, bounds=(1, 25))
    
for slab in si_slabs:
    slab.thick.setp(vary=False)
    slab.rough.setp(vary=False)
    slab.sld.density.setp(vary=False)
  
    
# Material 1 Bulk
for i, slab in enumerate(dsa_bulk_slabs):
    # Conditions
    vary_exp = True if i == 0 else None
    thick_constraint_exp = total_thick - dsa_surface_slabs[0].thick - dsa_substrate_slabs[0].thick
    rough_constraint_exp = None if i == 0 else dsa_bulk_slabs[0].rough
    birefringence_constraint_exp = None if i == 0 else dsa_bulk_slabs[0].sld.birefringence*ratio_bire_en[i]

    # Variables
    slab.thick.setp(vary=None, bounds=(400,800), constraint=thick_constraint_exp)
    slab.rough.setp(vary=vary_exp, bounds=(0,30), constraint=rough_constraint_exp)
    
    # Isotropic Optical Constants
    slab.sld.delta.setp(vary=True, bounds=(-0.01, 0.01))
    slab.sld.beta.setp(vary=True, bounds=(0.0000001, 0.01))
    
    # Anisotropic Optical Constants
    slab.sld.birefringence.setp(vary=vary_exp, bounds=(-0.01, 0.01), constraint=birefringence_constraint_exp)
    slab.sld.dichroism.setp(vary=None, constraint=slab.sld.birefringence*ratio_bire_dichro[i])
    
    # Tensor components (uniaxial constraint in effect)
    slab.sld.xx.setp(vary=None, constraint=(slab.sld.delta + (1/3)*slab.sld.birefringence))
    slab.sld.zz.setp(vary=None, constraint=(slab.sld.delta - (2/3)*slab.sld.birefringence))
    slab.sld.ixx.setp(vary=None, constraint=(slab.sld.beta + (1/3)*slab.sld.dichroism))
    slab.sld.izz.setp(vary=None, constraint=(slab.sld.beta - (2/3)*slab.sld.dichroism))
    
# dsa_ph surface
for i, slab in enumerate(dsa_surface_slabs):
    # energy dependent conditions
    vary_exp = True if i == 0 else None
    thick_constraint_exp = None if i == 0 else dsa_surface_slabs[0].thick
    rough_constraint_exp = None if i == 0 else dsa_surface_slabs[0].rough
    birefringence_constraint_exp = None if i == 0 else dsa_surface_slabs[0].sld.birefringence*ratio_bire_en[i]
    
    # Variables
    slab.thick.setp(vary=vary_exp, bounds=(surface_rough,30), constraint=thick_constraint_exp)
    slab.rough.setp(vary=vary_exp, bounds=(0,30), constraint=rough_constraint_exp)
    
    # Isotropic Optical Constants
    slab.sld.delta.setp(vary=None, constraint=dsa_bulk_slabs[i].sld.delta)
    slab.sld.beta.setp(vary=None, constraint=dsa_bulk_slabs[i].sld.beta)
    
    # Anisotropic Optical Constants
    slab.sld.birefringence.setp(vary=vary_exp, bounds=(-0.01, 0.01), constraint=birefringence_constraint_exp)
    slab.sld.dichroism.setp(vary=None, constraint=slab.sld.birefringence*ratio_bire_dichro[i])
    
    # Tensor components (uniaxial constraint in effect)
    slab.sld.xx.setp(vary=None, constraint=(slab.sld.delta + (1/3)*slab.sld.birefringence))
    slab.sld.zz.setp(vary=None, constraint=(slab.sld.delta - (2/3)*slab.sld.birefringence))
    slab.sld.ixx.setp(vary=None, constraint=(slab.sld.beta + (1/3)*slab.sld.dichroism))
    slab.sld.izz.setp(vary=None, constraint=(slab.sld.beta - (2/3)*slab.sld.dichroism))
    
# dsa_ph Substrate
for i, slab in enumerate(dsa_substrate_slabs):
    # energy dependent conditions
    

    vary_exp = True if i == 0 else None
    thick_constraint_exp = None if i == 0 else dsa_substrate_slabs[0].thick
    rough_constraint_exp = None if i == 0 else dsa_substrate_slabs[0].rough
    birefringence_constraint_exp = None if i == 0 else dsa_substrate_slabs[0].sld.birefringence*ratio_bire_en[i]
    
    # Variables
    slab.thick.setp(vary=vary_exp, bounds=(1,30), constraint=thick_constraint_exp)
    slab.rough.setp(vary=vary_exp, bounds=(0,30), constraint=rough_constraint_exp)
    
    # Isotropic Optical Constants
    slab.sld.delta.setp(vary=None, constraint=dsa_bulk_slabs[i].sld.delta)
    slab.sld.beta.setp(vary=None, constraint=dsa_bulk_slabs[i].sld.beta)
    
    # Anisotropic Optical Constants
    slab.sld.birefringence.setp(vary=vary_exp, bounds=(-0.01, 0.01), constraint=birefringence_constraint_exp)
    slab.sld.dichroism.setp(vary=None, constraint=slab.sld.birefringence*ratio_bire_dichro[i])
    
    # Tensor components (uniaxial constraint in effect)
    slab.sld.xx.setp(vary=None, constraint=(slab.sld.delta + (1/3)*slab.sld.birefringence))
    slab.sld.zz.setp(vary=None, constraint=(slab.sld.delta - (2/3)*slab.sld.birefringence))
    slab.sld.ixx.setp(vary=None, constraint=(slab.sld.beta + (1/3)*slab.sld.dichroism))
    slab.sld.izz.setp(vary=None, constraint=(slab.sld.beta - (2/3)*slab.sld.dichroism))

"""
Other inputs for scattering model
"""
# q-smearing
dq = 0

# Use multiplicative scale parameter
scale_vary = False
scale_lb = 0.6
scale_ub = 1.2

# Use additive background parameter
bkg_vary = False
bkg_lb = 1e-9
bkg_ub = 9e-6


"""
Settings for MCMC fit routine
"""
# Specify random state if you want to repeat a minimization
# Set to None if you want a "random" sampling
random_state = 1124

# Specify the number of markov chains to run
nwalkers = 50

# Number of generations (samples) to run the MCMC
nsamples = 200

# Type of 'move' to use for proposal generation (see https://emcee.readthedocs.io/en/stable/user/moves/#moves-user for options)
# Use none for default, you will need to include alternate moves in the initialization (See line 25)
move = [(DEMove(sigma=1e-7), 0.95), (DEMove(sigma=1e-7, gamma0=1), 0.05)]

# Name of the file you want to save the MCMC chain array as
chain_name = 'SaveChain.txt'

# Name to save the fitter as pkl file
save_fitter = 'fitter.pkl'

"""
***********************
***END OF USER INPUT***
***********************
"""

# Create the Model / Objective
models = []

for i, en in enumerate(en_array):
    models.append(PXR_ReflectModel(structures[i], scale=1, bkg=0, dq=dq, energy=en, pol='sp', name=('en_'+str(en))))
    models[i].scale.setp(vary=scale_vary, bounds=(scale_lb, scale_ub))
    models[i].bkg.setp(vary=bkg_vary, bounds=(bkg_lb, bkg_ub))

obj = []
for i, en in enumerate(en_array):
    obj_temp = Objective(models[i], reflect_datasets[i], transform=Transform('logY'), name='obj_'+str(en))
    lpe = LogpExtra_rough(obj_temp)
    obj_temp.logp_extra = lpe
    
    obj.append(obj_temp)
    
objective = GlobalObjective(obj)

#Create the curvefitter
fitter = CurveFitter(objective, nwalkers=nwalkers, moves=move)
fitter.initialise(pos='jitter')
print('Begin Fitting ' + myfile)

chain = fitter.sample(nsamples, f=(os.path.join(path_save, chain_name)), random_state=random_state)
print('Fitting has finished')

# Save outputs
with open(os.path.join(path_save, save_fitter), 'wb+') as f:
    pickle.dump(fitter, f)

export_mcmc_summary(path_save, name_save, fitter)

print('Output has been generated')























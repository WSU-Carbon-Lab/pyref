## v0.8.1 (2025-07-01)

### Fix

- hopefully this does it
- **fitting**: update ll calculation
- **imports**: circular imports
- imports
- **api**: imports into api module
- **api**: primordial support for api extension
- **rust**: update pyref-core to latest
- fixed sampler kwargs update
- added direct passthough of sampler keywords
- **fitting**: fixed hidden function
- rollback to last stable sampler
- fixind default thread pool allocator
- **fitting**: removing more backend junk
- **fitting**: removed backend checkpointing for more stable api

### Refactor

- start of refactor into api extension
- **io**: fits file and directory io
- **loader**: segregated IO operations from loader base class. added tjf's loader class

## v0.8.0 (2025-04-07)

### Feat

- **fitting**: added improved fitting routines

### Fix

- **formatting**: updated ruff version
- **formatting**: fixed formatting
- **fitting**: extrodinary component fixed
- **fitting**: updated rotations to include the correct tensor rotation calculation
- added default curve fitting walker number
- added self refference to fix curve fitter instantiation
- **fitting**: moved curve fitter object from refnx to pyref
- fixed build

## v0.7.4 (2025-03-04)

### Fix

- **fitting**: interpolate error fix
- **fitting**: anisotropy fix
- **fitting**: upated model anisotropy to be correctly evaluated
- **plotting**: updated plot to fix anisotropy plot
- moved from typechecking block
- **fitting**: updated how offsets are handled
- **fitting**: finally?
- **fitting**: again .. lets see
- **fitting**: fixed anisotropy len ... maybe...lets see
- **fitting**: update to qcomon
- **fitting**: fixed qcomon calculation
- **fitting**: fixed issue with array size mixmatch inanisotropy

## v0.7.3 (2025-03-03)

### Fix

- **fitting**: added energy offset to parameters
- **fitting**: model construction for s or p pol is now allowed

## v0.7.2 (2025-03-02)

### Fix

- **fitting**: update range of errors
- **fitting**: rollback changes to rotation for now. Larger issue than previously thought
- **fitting**: updates rotation matrix to apply 3x3 rotations instead of 2x2

## v0.7.1 (2025-02-24)

### Fix

- **fitting**: added passthough for error updates in model. added custom scale of the anisotropic ratio to xrr

## v0.7.0 (2025-02-21)

### Feat

- added anisotropic ratio as an extra condition on the log likelyhood

## v0.6.3 (2025-02-17)

### Fix

- updated NexafsSLD to UniTensorSLD
- fixed constraints to avoid corruption
- proper fit constraints
- typo
- typo
- constraints properly applied at instantiation
- initial state check change
- remove legacy skip_initial_state_check

## v0.6.2 (2025-02-13)

### Fix

- **fitting**: added df - refnx.ReflectDataset converter

## v0.6.1 (2025-02-13)

### Fix

- **fitting**: added back fitter classes

## v0.6.0 (2025-02-13)

### Feat

- added prototype fitting

### Fix

- **fitting**: rename to remove PXR_ subscript
- **fitting**: add NexafsSLD to __all__

## v0.5.1 (2025-02-11)

### Fix

- ci fix

## v0.5.0 (2025-02-11)

### Fix

- workflow fix to remove cargo need

## v0.4.3 (2025-02-11)

### Fix

- rust compilation

## v0.4.2 (2025-02-11)

### Fix

- remove x-ray tag

## v0.4.1 (2025-02-11)

### Fix

- added refnx[all] deps. updated workflow

## v0.4.0 (2025-02-11)

### Feat

- added fitting and xrr model

## v0.3.0 (2025-02-10)

### Feat

- rename success

## v0.2.1 (2025-01-29)

### Fix

- update io functions at py03 handshake

## v0.2.0 (2025-01-28)

### Fix

- Update to use pyref-core @ latest

## v0.1.3 (2025-01-28)

### Feat

- Rollback to last stable
- Rollback to last stable (#25)
- Rollback to last stable

### Fix

- Bumped rust backend version and fixed new bugs. Discovered bug with unwrapping LazyFrame

## v0.1.5 (2024-04-03)

## v0.1.0 (2023-06-03)

import scipy.constants as const


class XRayData:
    @classmethod
    def energy_to_wavelength(cls, energy):
        return const.h * const.c / energy

    @classmethod
    def wavelength_to_energy(cls, lam):
        return const.h * const.c / lam

    @classmethod
    def energy_to_wavenumber(cls, en):
        return en / const.h

    @classmethod
    def wavenumber_to_energy(cls, k):
        return k * const.h

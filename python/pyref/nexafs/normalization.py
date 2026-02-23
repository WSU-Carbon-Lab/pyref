"""
NEXAFS normalization fit schemes: bare atom, constant, Si/O background, polynomial.

Each scheme fits scale (and optional background terms) in the pre+post edge region
and returns full model curve, background-only curve, and background-subtracted
intensity for widget plotting.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np
from scipy.optimize import curve_fit


class NormalizationScheme(str, Enum):
    """Normalization method for NEXAFS absorption."""

    NONE = "none"
    BARE_ATOM = "bare_atom"
    BARE_ATOM_CONST = "bare_atom_const"
    BARE_ATOM_SI = "bare_atom_si"
    BARE_ATOM_SI_O = "bare_atom_si_o"
    BARE_ATOM_POLY = "bare_atom_poly"


def _fit_region_mask(
    energy: np.ndarray,
    pre_edge: tuple[float | None, float | None],
    post_edge: tuple[float | None, float | None],
) -> np.ndarray:
    pre_lo, pre_hi = pre_edge
    post_lo, post_hi = post_edge
    pre_ok = np.ones(energy.size, dtype=bool)
    if pre_lo is not None:
        pre_ok &= energy >= pre_lo
    if pre_hi is not None:
        pre_ok &= energy <= pre_hi
    post_ok = np.ones(energy.size, dtype=bool)
    if post_lo is not None:
        post_ok &= energy >= post_lo
    if post_hi is not None:
        post_ok &= energy <= post_hi
    return pre_ok | post_ok


def fit_bare_atom(
    energy: np.ndarray,
    intensity: np.ndarray,
    fit_mask: np.ndarray,
    mu_chemical: np.ndarray,
) -> dict[str, Any]:
    """
    Fit scale so that scale * mu(E) matches intensity in fit region.

    Model: scale * mu(E). Background = 0.
    """
    e_fit = energy[fit_mask]
    y_fit = intensity[fit_mask]
    mu_fit = mu_chemical[fit_mask]
    scale = np.nanmean(y_fit / mu_fit)
    if scale <= 0 or not np.isfinite(scale):
        scale = 1.0
    full_curve = scale * mu_chemical
    background_curve = np.zeros_like(energy)
    scaled_mu = scale * mu_chemical
    return {
        "params": {"scale": scale},
        "full_curve": full_curve,
        "background_curve": background_curve,
        "scaled_mu": scaled_mu,
    }


def fit_bare_atom_const(
    energy: np.ndarray,
    intensity: np.ndarray,
    fit_mask: np.ndarray,
    mu_chemical: np.ndarray,
) -> dict[str, Any]:
    """
    Fit scale and constant background: scale * (mu(E) + bkg).
    """

    def model(e: np.ndarray, scale: float, bkg: float) -> np.ndarray:
        mc = np.interp(e, energy, mu_chemical)
        return scale * (mc + bkg)

    e_fit = energy[fit_mask]
    y_fit = intensity[fit_mask]
    mu_fit = mu_chemical[fit_mask]
    p0_scale = float(np.nanmean(y_fit / mu_fit))
    if not np.isfinite(p0_scale) or p0_scale <= 0:
        p0_scale = 1.0
    p0_scale = max(p0_scale, 1e-5)
    p0 = (p0_scale, 0.0)
    bounds = ((1e-6, -np.inf), (np.inf, np.inf))
    popt, _ = curve_fit(model, e_fit, y_fit, p0=p0, bounds=bounds)
    scale, bkg = float(popt[0]), float(popt[1])
    full_curve = model(energy, scale, bkg)
    background_curve = np.full_like(energy, scale * bkg)
    scaled_mu = scale * mu_chemical
    return {
        "params": {"scale": scale, "bkg": bkg},
        "full_curve": full_curve,
        "background_curve": background_curve,
        "scaled_mu": scaled_mu,
    }


def fit_bare_atom_si(
    energy: np.ndarray,
    intensity: np.ndarray,
    fit_mask: np.ndarray,
    mu_chemical: np.ndarray,
    mu_si: np.ndarray,
) -> dict[str, Any]:
    """
    Fit scale and Si composition: scale * (mu(E) + si_comp * mu_si(E)).
    """

    def model(e: np.ndarray, scale: float, si_comp: float) -> np.ndarray:
        mc = np.interp(e, energy, mu_chemical)
        ms = np.interp(e, energy, mu_si)
        return scale * (mc + si_comp * ms)

    e_fit = energy[fit_mask]
    y_fit = intensity[fit_mask]
    mu_fit = mu_chemical[fit_mask]
    p0_scale = float(np.nanmean(y_fit / mu_fit))
    if not np.isfinite(p0_scale) or p0_scale <= 0:
        p0_scale = 1.0
    p0_scale = max(p0_scale, 1e-5)
    p0 = (p0_scale, 0.0)
    bounds = ((1e-6, 0.0), (np.inf, np.inf))
    popt, _ = curve_fit(model, e_fit, y_fit, p0=p0, bounds=bounds)
    scale, si_comp = float(popt[0]), float(popt[1])
    full_curve = model(energy, scale, si_comp)
    background_curve = scale * si_comp * mu_si
    scaled_mu = scale * mu_chemical
    return {
        "params": {"scale": scale, "si_comp": si_comp},
        "full_curve": full_curve,
        "background_curve": background_curve,
        "scaled_mu": scaled_mu,
    }


def fit_bare_atom_si_o(
    energy: np.ndarray,
    intensity: np.ndarray,
    fit_mask: np.ndarray,
    mu_chemical: np.ndarray,
    mu_si: np.ndarray,
    mu_o: np.ndarray,
) -> dict[str, Any]:
    """
    Fit scale, Si and O composition: scale * (mu + si_comp*mu_si + o_comp*mu_o).
    """

    def model(
        e: np.ndarray,
        scale: float,
        si_comp: float,
        o_comp: float,
    ) -> np.ndarray:
        mc = np.interp(e, energy, mu_chemical)
        ms = np.interp(e, energy, mu_si)
        mo = np.interp(e, energy, mu_o)
        return scale * (mc + si_comp * ms + o_comp * mo)

    e_fit = energy[fit_mask]
    y_fit = intensity[fit_mask]
    p0_scale = float(np.nanmean(y_fit / mu_chemical[fit_mask]))
    if not np.isfinite(p0_scale) or p0_scale <= 0:
        p0_scale = 1.0
    p0_scale = max(p0_scale, 1e-5)
    p0 = (p0_scale, 0.0, 0.0)
    bounds = ((1e-6, 0.0, 0.0), (np.inf, np.inf, np.inf))
    popt, _ = curve_fit(model, e_fit, y_fit, p0=p0, bounds=bounds)
    scale, si_comp, o_comp = float(popt[0]), float(popt[1]), float(popt[2])
    full_curve = model(energy, scale, si_comp, o_comp)
    background_curve = scale * (si_comp * mu_si + o_comp * mu_o)
    scaled_mu = scale * mu_chemical
    return {
        "params": {"scale": scale, "si_comp": si_comp, "o_comp": o_comp},
        "full_curve": full_curve,
        "background_curve": background_curve,
        "scaled_mu": scaled_mu,
    }


def fit_bare_atom_poly(
    energy: np.ndarray,
    intensity: np.ndarray,
    fit_mask: np.ndarray,
    mu_chemical: np.ndarray,
) -> dict[str, Any]:
    """
    Fit scale and cubic polynomial background: scale * (mu(E) + a + b*E + c*E^2 + d*E^3).
    """

    def model(
        e: np.ndarray,
        scale: float,
        a: float,
        b: float,
        c: float,
        d: float,
    ) -> np.ndarray:
        mc = np.interp(e, energy, mu_chemical)
        return scale * (mc + a + b * e + c * e**2 + d * e**3)

    e_fit = energy[fit_mask]
    y_fit = intensity[fit_mask]
    p0_scale = float(np.nanmean(y_fit / mu_chemical[fit_mask]))
    if not np.isfinite(p0_scale) or p0_scale <= 0:
        p0_scale = 1.0
    p0_scale = max(p0_scale, 1e-5)
    p0 = (p0_scale, 0.0, 0.0, 0.0, 0.0)
    bounds = (
        (1e-6, -np.inf, -np.inf, -np.inf, -np.inf),
        (np.inf, np.inf, np.inf, np.inf, np.inf),
    )
    popt, _ = curve_fit(model, e_fit, y_fit, p0=p0, bounds=bounds)
    scale, a, b, c, d = (
        float(popt[0]),
        float(popt[1]),
        float(popt[2]),
        float(popt[3]),
        float(popt[4]),
    )
    full_curve = model(energy, scale, a, b, c, d)
    poly = a + b * energy + c * energy**2 + d * energy**3
    background_curve = scale * poly
    scaled_mu = scale * mu_chemical
    return {
        "params": {"scale": scale, "a": a, "b": b, "c": c, "d": d},
        "full_curve": full_curve,
        "background_curve": background_curve,
        "scaled_mu": scaled_mu,
    }


def fit_normalization(
    scheme: NormalizationScheme,
    energy: np.ndarray,
    intensity: np.ndarray,
    pre_edge: tuple[float | None, float | None],
    post_edge: tuple[float | None, float | None],
    mu_chemical: np.ndarray,
    mu_si: np.ndarray | None = None,
    mu_o: np.ndarray | None = None,
) -> dict[str, Any] | None:
    """
    Run the selected normalization fit and return full_curve, background_curve, scaled_mu, params.

    Parameters
    ----------
    scheme : NormalizationScheme
        One of BARE_ATOM, BARE_ATOM_CONST, BARE_ATOM_SI, BARE_ATOM_SI_O, BARE_ATOM_POLY.
    energy : np.ndarray
        Energy (eV) for all points.
    intensity : np.ndarray
        PD Corrected or Norm Abs intensity.
    pre_edge : tuple
        (e_lo, e_hi) for pre-edge; None for open-ended.
    post_edge : tuple
        (e_lo, e_hi) for post-edge.
    mu_chemical : np.ndarray
        Bare-atom mu(E) for main formula.
    mu_si, mu_o : np.ndarray, optional
        Required for BARE_ATOM_SI and BARE_ATOM_SI_O.

    Returns
    -------
    dict with params, full_curve, background_curve, scaled_mu; or None if scheme is NONE.
    """
    if scheme == NormalizationScheme.NONE:
        return None
    fit_mask = _fit_region_mask(energy, pre_edge, post_edge)
    if np.sum(fit_mask) < 2:
        return None
    if scheme == NormalizationScheme.BARE_ATOM:
        return fit_bare_atom(energy, intensity, fit_mask, mu_chemical)
    if scheme == NormalizationScheme.BARE_ATOM_CONST:
        return fit_bare_atom_const(
            energy, intensity, fit_mask, mu_chemical
        )
    if scheme == NormalizationScheme.BARE_ATOM_SI:
        if mu_si is None:
            raise ValueError("mu_si required for BARE_ATOM_SI")
        return fit_bare_atom_si(
            energy, intensity, fit_mask, mu_chemical, mu_si
        )
    if scheme == NormalizationScheme.BARE_ATOM_SI_O:
        if mu_si is None or mu_o is None:
            raise ValueError("mu_si and mu_o required for BARE_ATOM_SI_O")
        return fit_bare_atom_si_o(
            energy, intensity, fit_mask, mu_chemical, mu_si, mu_o
        )
    if scheme == NormalizationScheme.BARE_ATOM_POLY:
        return fit_bare_atom_poly(
            energy, intensity, fit_mask, mu_chemical
        )
    return None

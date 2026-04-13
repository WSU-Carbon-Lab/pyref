"""
NEXAFS database IO: load sample, nexafs, and izero data; compute bare_atom and absorbance columns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from periodictable import xsf
from pint import UnitRegistry

if TYPE_CHECKING:
    from sqlite3 import Connection

_UREG = UnitRegistry()
_NEXAFS_COLS = [
    "scan_id",
    "beamline_energy",
    "sample_theta",
    "tey_signal",
    "photodiode",
    "ai_3_izero",
    "izero_before_ai_3_izero",
    "izero_before_photodiode",
    "izero_after_ai_3_izero",
    "izero_after_photodiode",
]


def _calculate_mu_row(
    row: pd.Series,
    units: str = "g/cm^2",
    formula: str | None = None,
) -> float:
    used_formula = formula if formula is not None else row["chemical_formula"]
    energy = float(row["beamline_energy"])
    n = xsf.index_of_refraction(used_formula, energy=(energy * 1e-3), density=1.0)
    beta = -n.imag
    hc = 12.3984193
    wavelength_A = hc / (energy * 1e-3)
    wavelength_cm = wavelength_A * 1e-8
    mu_over_rho = float(4 * np.pi * beta / wavelength_cm)
    mu_qty = mu_over_rho * _UREG("g/cm^2")
    mu_converted = mu_qty.to(units)
    return float(np.asarray(mu_converted.magnitude))


def _get_samples(
    conn: Connection,
    sample_name: str,
    tag: str | None = None,
    version: int | None = None,
) -> pd.DataFrame:
    query = "SELECT * FROM sample WHERE name = ?"
    params: list[Any] = [sample_name]
    if version is not None:
        query += " AND version = ?"
        params.append(version)
    if tag is not None:
        query += " AND tag = ?"
        params.append(tag)
    return pd.read_sql_query(query, conn, params=params)


def load_nexafs(
    conn: Connection,
    sample_name: str,
    tag: str | None = None,
    version: int | None = None,
    substrate: str = "Si",
    cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load NEXAFS data from the database for one sample and compute absorbance columns.

    Queries sample, nexafs, and izero tables; merges on beamline_energy; adds
    chemical_formula, bare_atom, bare_atom_substrate, pd_response_0/1, raw_abs,
    absorbance_0, and absorbance_1. The result is ready for df.nexafs normalization.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open connection to the NEXAFS database.
    sample_name : str
        Sample name (sample.name).
    tag : str, optional
        Filter by sample.tag.
    version : int, optional
        Filter by sample.version.
    substrate : str, optional
        Chemical formula for substrate bare-atom cross section (default "Si").

    Returns
    -------
    pd.DataFrame
        DataFrame with beamline_energy, sample_theta, tey_signal, photodiode,
        ai_3_izero, izero_before_*, izero_after_*, chemical_formula, bare_atom,
        bare_atom_substrate, pd_response_0, pd_response_1, raw_abs,
        absorbance_0, absorbance_1.
    """
    if cols is None:
        cols = _NEXAFS_COLS
    sample = _get_samples(conn, sample_name, tag=tag, version=version)
    if sample.empty:
        msg = f"No sample found for name={sample_name!r}, tag={tag!r}, version={version!r}"
        raise ValueError(msg)
    sample_id = int(sample["id"].iloc[0])
    sample_formula = str(sample["chemical_formula"].iloc[0])

    nexafs_df = pd.read_sql_query(
        "SELECT * FROM nexafs WHERE sample_id = ?",
        conn,
        params=[sample_id],
    )
    if nexafs_df.empty:
        msg = f"No nexafs rows for sample_id={sample_id}"
        raise ValueError(msg)

    def _scan_id_or_none(val: Any) -> int | None:
        if val is None:
            return None
        if isinstance(val, float) and np.isnan(val):
            return None
        try:
            return int(val)
        except (TypeError, ValueError):
            return None

    izero_before_id = _scan_id_or_none(nexafs_df.iloc[0]["izero_before_scan_id"])
    izero_after_id = _scan_id_or_none(nexafs_df.iloc[0]["izero_after_scan_id"])

    energy_col = nexafs_df[["beamline_energy"]].copy()
    if izero_before_id is not None:
        izero_before = pd.read_sql_query(
            "SELECT * FROM izero WHERE scan_id = ?",
            conn,
            params=[izero_before_id],
        )
    else:
        izero_before = energy_col.copy()
        izero_before["ai_3_izero"] = np.nan
        izero_before["photodiode"] = np.nan

    if izero_after_id is not None:
        izero_after = pd.read_sql_query(
            "SELECT * FROM izero WHERE scan_id = ?",
            conn,
            params=[izero_after_id],
        )
    else:
        izero_after = energy_col.copy()
        izero_after["ai_3_izero"] = np.nan
        izero_after["photodiode"] = np.nan

    merged = pd.merge(
        nexafs_df,
        izero_before.add_prefix("izero_before_"),
        left_on="beamline_energy",
        right_on="izero_before_beamline_energy",
        how="left",
        suffixes=("", "_before"),
    )
    merged = pd.merge(
        merged,
        izero_after.add_prefix("izero_after_"),
        left_on="beamline_energy",
        right_on="izero_after_beamline_energy",
        how="left",
        suffixes=("", "_after"),
    )[cols]

    merged["chemical_formula"] = sample_formula
    merged["bare_atom"] = merged.apply(
        _calculate_mu_row,
        axis=1,
        units="g/cm^2",
    )
    merged["bare_atom_substrate"] = merged.apply(
        _calculate_mu_row,
        axis=1,
        units="g/cm^2",
        formula=substrate,
    )

    merged["pd_response_0"] = (
        merged["izero_before_ai_3_izero"] / merged["izero_before_photodiode"]
    )
    merged["pd_response_1"] = (
        merged["izero_after_ai_3_izero"] / merged["izero_after_photodiode"]
    )
    merged["raw_abs"] = merged["tey_signal"] / merged["ai_3_izero"]
    merged["absorbance_0"] = merged["pd_response_0"] / merged["raw_abs"]
    merged["absorbance_1"] = merged["pd_response_1"] / merged["raw_abs"]

    return cast("pd.DataFrame", merged)

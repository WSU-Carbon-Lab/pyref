"""
Convenience loader for NEXAFS data: single function to load from path to concatenated DataFrame.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pyref.nexafs.directory import NexafsDirectory


def load_nexafs(
    path: str | Path,
    sample_name: str | None = None,
    formula: str = "C8H8",
    pre_edge: tuple[float | None, float | None] = (None, 280.0),
    post_edge: tuple[float | None, float | None] = (360.0, None),
) -> pd.DataFrame:
    """
    Load NEXAFS data from a directory into a single DataFrame.

    Discovers izero and sample files, computes TEY absorption and baseline
    normalization. No background fit is applied; use the widget or
    normalization module for that.

    Parameters
    ----------
    path : str | Path
        Directory containing NEXAFS scan files (izero*.txt, sample_angle_exp.txt).
    sample_name : str | None
        If set, load only this sample; otherwise load all samples.
    formula : str
        Chemical formula for bare-atom normalization, e.g. "C8H8".
    pre_edge : tuple
        Pre-edge region (e_lo, e_hi); None for open-ended.
    post_edge : tuple
        Post-edge region (e_lo, e_hi).

    Returns
    -------
    pd.DataFrame
        Concatenated rows with Energy, PD Corrected, Norm Abs, Bare Atom Step,
        Mass Abs., Sample, Angle, Experiment.
    """
    directory = NexafsDirectory(path)
    samples = [sample_name] if sample_name else directory.list_samples()
    if not samples:
        return pd.DataFrame()
    records: list[pd.DataFrame] = []
    for name in samples:
        dfs = directory.get_sample_dfs(
            name,
            formula=formula,
            pre_edge=pre_edge,
            post_edge=post_edge,
        )
        records.extend(dfs)
    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)

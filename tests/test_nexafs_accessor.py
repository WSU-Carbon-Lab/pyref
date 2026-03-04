"""Tests for df.nexafs DataFrame accessor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyref  # noqa: F401 - registers df.nexafs accessor
import pytest


@pytest.fixture
def single_scan_df() -> pd.DataFrame:
    energy = np.linspace(260, 360, 50)
    mu = 2000 + 100 * (energy - 280)
    mu_sub = 40000 - 50 * (energy - 280)
    abs0 = 1e-8 * (mu + 1.2 * mu_sub) + 0.0002
    abs1 = 1.2e-8 * (mu + 0.9 * mu_sub) + 0.00018
    return pd.DataFrame({
        "beamline_energy": energy,
        "bare_atom": mu,
        "bare_atom_substrate": mu_sub,
        "absorbance_0": abs0,
        "absorbance_1": abs1,
    })


@pytest.fixture
def multi_angle_df(single_scan_df: pd.DataFrame) -> pd.DataFrame:
    dfs = []
    for angle in [20.0, 40.0, 60.0]:
        df = single_scan_df.copy()
        df["sample_theta"] = angle
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def test_nexafs_accessor_registered() -> None:
    df = pd.DataFrame({
        "beamline_energy": [270],
        "bare_atom": [1.0],
        "bare_atom_substrate": [1.0],
        "absorbance_0": [0.1],
    })
    assert hasattr(df, "nexafs")
    assert df.nexafs._obj is df


def test_nexafs_schema_validation() -> None:
    df = pd.DataFrame({"a": [1], "b": [2]})
    with pytest.raises(ValueError, match="Missing"):
        df.nexafs.normalization_region()


def test_normalization_region_mask(single_scan_df: pd.DataFrame) -> None:
    single_scan_df.nexafs.set_regions(
        pre_edge=(270, 281),
        post_edge=(335, 350),
    )
    norm = single_scan_df.nexafs.normalization_region()
    energy = norm["beamline_energy"].to_numpy()
    in_pre = (energy >= 270) & (energy <= 281)
    in_post = (energy >= 335) & (energy <= 350)
    assert np.all(in_pre | in_post)
    assert len(norm) < len(single_scan_df)


def test_normalize_adds_columns_and_stores_params(single_scan_df: pd.DataFrame) -> None:
    single_scan_df.nexafs.set_regions(pre_edge=(270, 281), post_edge=(335, 350))
    single_scan_df.nexafs.normalize()
    assert "mass_absorption_0" in single_scan_df.columns
    assert "mass_absorption_1" in single_scan_df.columns
    assert single_scan_df.nexafs._fit_params is not None
    assert "absorbance_0" in single_scan_df.nexafs._fit_params
    assert "popt" in single_scan_df.nexafs._fit_params["absorbance_0"]


def test_plot_regions_smoke(single_scan_df: pd.DataFrame) -> None:
    single_scan_df.nexafs.set_regions(pre_edge=(270, 281), post_edge=(335, 350))
    ax = single_scan_df.nexafs.plot_regions()
    assert ax is not None


def test_plot_normalization_smoke(single_scan_df: pd.DataFrame) -> None:
    single_scan_df.nexafs.set_regions(pre_edge=(270, 281), post_edge=(335, 350))
    single_scan_df.nexafs.normalize()
    ax = single_scan_df.nexafs.plot_normalization(show_fit=True)
    assert ax is not None


def test_plot_by_group_smoke(multi_angle_df: pd.DataFrame) -> None:
    multi_angle_df.nexafs.set_regions(pre_edge=(270, 281), post_edge=(335, 350))
    ax = multi_angle_df.nexafs.plot(
        x="beamline_energy",
        y="absorbance_0",
        by="sample_theta",
        colorbar="Angle (deg)",
    )
    assert ax is not None


def test_normalize_by_group(multi_angle_df: pd.DataFrame) -> None:
    from pyref.nexafs import normalize_by_group

    result = normalize_by_group(
        multi_angle_df,
        group_column="sample_theta",
        pre_edge=(270, 281),
        post_edge=(335, 350),
    )
    assert "mass_absorption_0" in result.columns
    assert "mass_absorption_1" in result.columns
    assert len(result) == len(multi_angle_df)

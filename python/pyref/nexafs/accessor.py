"""
Pandas DataFrame accessor for NEXAFS normalization: pre/post edge regions, fit, correction, and plotting.
"""

from __future__ import annotations

from typing import Any, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.pyplot import colormaps
from scipy.optimize import curve_fit


_REQUIRED_FIXED_COLUMNS = ("bare_atom", "bare_atom_substrate")
_DEFAULT_PRE_EDGE = (270.0, 281.0)
_DEFAULT_POST_EDGE = (335.0, 350.0)
_DEFAULT_ABSORBANCE_COLS = ["absorbance_0", "absorbance_1"]

NormalizationMode = Literal["full", "bare_atom", "step"]


def _fit_region_mask(
    energy: np.ndarray,
    pre_edge: tuple[float, float],
    post_edge: tuple[float, float],
) -> np.ndarray:
    pre_lo, pre_hi = pre_edge
    post_lo, post_hi = post_edge
    pre_ok = (energy >= pre_lo) & (energy <= pre_hi)
    post_ok = (energy >= post_lo) & (energy <= post_hi)
    return pre_ok | post_ok


def _mu_step(energy: np.ndarray, jump_energy: float) -> np.ndarray:
    out = np.zeros_like(energy, dtype=np.float64)
    out[energy > jump_energy] = 1.0
    return out


def _model_factory_full(mu: np.ndarray, mu_substrate: np.ndarray):
    def model(
        _energy: np.ndarray,
        izero: float,
        composition: float,
        shift: float,
    ) -> np.ndarray:
        out = izero * (mu + composition * mu_substrate) + shift
        return np.asarray(out, dtype=np.float64)

    return model


def _model_factory_bare_atom(mu: np.ndarray):
    def model(_energy: np.ndarray, izero: float, shift: float) -> np.ndarray:
        out = izero * mu + shift
        return np.asarray(out, dtype=np.float64)

    return model


def _model_factory_step(jump_energy: float):
    def model(_energy: np.ndarray, izero: float, shift: float) -> np.ndarray:
        mu_01 = _mu_step(_energy, jump_energy)
        out = izero * mu_01 + shift
        return np.asarray(out, dtype=np.float64)

    return model


def _correct_row_full(
    absorb: np.ndarray,
    bare_atom_substrate: np.ndarray,
    izero: float,
    composition: float,
    shift: float,
) -> np.ndarray:
    return (absorb - shift) / izero - composition * bare_atom_substrate


def _correct_row_bare_or_step(absorb: np.ndarray, izero: float, shift: float) -> np.ndarray:
    return (absorb - shift) / izero


@pd.api.extensions.register_dataframe_accessor("nexafs")
class NexafsAccessor:
    """DataFrame accessor for NEXAFS normalization and plotting (single scan / one angle)."""

    def __init__(self, pandas_obj: pd.DataFrame) -> None:
        self._obj = pandas_obj
        self._pre_edge: tuple[float, float] = _DEFAULT_PRE_EDGE
        self._post_edge: tuple[float, float] = _DEFAULT_POST_EDGE
        self._energy_col: str = "beamline_energy"
        self._absorbance_cols: list[str] = list(_DEFAULT_ABSORBANCE_COLS)
        self._normalization_mode: NormalizationMode = "full"
        self._jump_energy: float | None = None
        self._fit_params: dict[str, dict[str, Any]] | None = None

    def _validate_schema(self) -> None:
        missing = [c for c in (_REQUIRED_FIXED_COLUMNS + (self._energy_col,)) if c not in self._obj.columns]
        if missing:
            raise ValueError(
                f"DataFrame must have columns {list(_REQUIRED_FIXED_COLUMNS)} and '{self._energy_col}'. Missing: {missing}"
            )
        for col in self._absorbance_cols:
            if col not in self._obj.columns:
                raise ValueError(
                    f"Absorbance column '{col}' not in DataFrame. "
                    f"Available: {list(self._obj.columns)}"
                )

    def set_regions(
        self,
        pre_edge: tuple[float, float] | None = None,
        post_edge: tuple[float, float] | None = None,
        energy_col: str | None = None,
        absorbance_cols: list[str] | None = None,
        normalization_mode: NormalizationMode | None = None,
        jump_energy: float | None = None,
    ) -> NexafsAccessor:
        """
        Set pre/post edge regions and optional column names.

        Parameters
        ----------
        pre_edge : tuple of float, optional
            (lo, hi) energy range for pre-edge normalization region.
        post_edge : tuple of float, optional
            (lo, hi) energy range for post-edge normalization region.
        energy_col : str, optional
            Column name for energy (default "beamline_energy").
        absorbance_cols : list of str, optional
            Column names for absorbance signals (default ["absorbance_0", "absorbance_1"]).
        normalization_mode : "full" | "bare_atom" | "step", optional
            "full": izero * (mu + composition * mu_substrate) + shift.
            "bare_atom": izero * mu + shift (no substrate).
            "step": izero * mu_01(energy) + shift with mu_01 = 0 pre-edge, 1 post-edge.
        jump_energy : float, optional
            For "step" mode only. Energy where step goes 0 -> 1. If None, uses
            (pre_edge[1] + post_edge[0]) / 2.

        Returns
        -------
        NexafsAccessor
            Self for method chaining.
        """
        if pre_edge is not None:
            self._pre_edge = pre_edge
        if post_edge is not None:
            self._post_edge = post_edge
        if energy_col is not None:
            self._energy_col = energy_col
        if absorbance_cols is not None:
            self._absorbance_cols = list(absorbance_cols)
        if normalization_mode is not None:
            self._normalization_mode = normalization_mode
        if jump_energy is not None:
            self._jump_energy = jump_energy
        return self

    def normalization_region(self) -> pd.DataFrame:
        """
        Rows where energy falls in pre-edge or post-edge region.

        Returns
        -------
        pd.DataFrame
            Subset of the DataFrame (view).
        """
        self._validate_schema()
        energy = self._obj[self._energy_col].to_numpy()
        mask = _fit_region_mask(energy, self._pre_edge, self._post_edge)
        return self._obj.loc[mask]

    def normalize(self) -> NexafsAccessor:
        """
        Fit izero, composition, shift per absorbance column in the normalization region,
        then add mass_absorption_* columns in-place. Stores fit params for plot_normalization.

        Returns
        -------
        NexafsAccessor
            Self for method chaining.
        """
        self._validate_schema()
        norm = self.normalization_region()
        if len(norm) == 0:
            raise ValueError(
                "Normalization region is empty. Check pre_edge and post_edge."
            )
        energy = np.asarray(norm[self._energy_col], dtype=np.float64)
        mu = np.asarray(norm["bare_atom"], dtype=np.float64)
        mu_sub = np.asarray(norm["bare_atom_substrate"], dtype=np.float64)
        mode = self._normalization_mode
        jump = self._jump_energy
        if mode == "step" and jump is None:
            jump = (self._pre_edge[1] + self._post_edge[0]) / 2.0

        self._fit_params = {}
        for col in self._absorbance_cols:
            y = np.asarray(norm[col], dtype=np.float64)
            mass_col = col.replace("absorbance", "mass_absorption", 1)
            if not np.any(np.isfinite(y)):
                self._obj[mass_col] = np.nan
                continue
            absorb_arr = np.asarray(self._obj[col], dtype=np.float64)
            mu_sub_arr = np.asarray(self._obj["bare_atom_substrate"], dtype=np.float64)

            if mode == "full":
                model = _model_factory_full(mu, mu_sub)
                p0_scale = float(np.nanmean(y / mu))
                if not np.isfinite(p0_scale) or p0_scale <= 0:
                    p0_scale = 1.0
                p0 = (p0_scale, 0.0, 0.0)
                popt, pcov = curve_fit(model, energy, y, p0=p0)
                self._obj[mass_col] = _correct_row_full(
                    absorb_arr, mu_sub_arr, popt[0], popt[1], popt[2]
                )
                self._fit_params[col] = {
                    "mode": "full",
                    "popt": popt,
                    "pcov": pcov,
                    "izero": popt[0],
                    "composition": popt[1],
                    "shift": popt[2],
                }
            elif mode == "bare_atom":
                model = _model_factory_bare_atom(mu)
                p0_scale = float(np.nanmean(y / mu))
                if not np.isfinite(p0_scale) or p0_scale <= 0:
                    p0_scale = 1.0
                p0 = (p0_scale, 0.0)
                popt, pcov = curve_fit(model, energy, y, p0=p0)
                self._obj[mass_col] = _correct_row_bare_or_step(absorb_arr, popt[0], popt[1])
                self._fit_params[col] = {
                    "mode": "bare_atom",
                    "popt": popt,
                    "pcov": pcov,
                    "izero": popt[0],
                    "composition": None,
                    "shift": popt[1],
                }
            else:
                assert mode == "step" and jump is not None
                model = _model_factory_step(jump)
                p0_scale = float(np.nanmean(y))
                if not np.isfinite(p0_scale) or p0_scale <= 0:
                    p0_scale = 1.0
                p0 = (p0_scale, 0.0)
                popt, pcov = curve_fit(model, energy, y, p0=p0)
                self._obj[mass_col] = _correct_row_bare_or_step(absorb_arr, popt[0], popt[1])
                self._fit_params[col] = {
                    "mode": "step",
                    "jump_energy": jump,
                    "popt": popt,
                    "pcov": pcov,
                    "izero": popt[0],
                    "composition": None,
                    "shift": popt[1],
                }
        return self

    def plot_regions(
        self,
        ax: Axes | None = None,
        pre_color: str = "blue",
        post_color: str = "green",
        alpha: float = 0.2,
        **kwargs: Any,
    ) -> Axes:
        """
        Plot full scan (energy vs absorbance) with pre- and post-edge regions shaded.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on; created if None.
        pre_color, post_color : str, optional
            Colors for pre- and post-edge axvspan.
        alpha : float, optional
            Alpha for shaded regions.
        **kwargs
            Passed to DataFrame.plot.

        Returns
        -------
        Axes
        """
        self._validate_schema()
        if ax is None:
            _fig, ax = plt.subplots()
            ax = cast(Axes, ax)  # type: ignore[assignment]
        self._obj.plot(
            x=self._energy_col,
            y=self._absorbance_cols,
            ax=ax,
            **kwargs,
        )
        ax.axvspan(
            self._pre_edge[0],
            self._pre_edge[1],
            color=pre_color,
            alpha=alpha,
        )
        ax.axvspan(
            self._post_edge[0],
            self._post_edge[1],
            color=post_color,
            alpha=alpha,
        )
        return cast(Axes, ax)  # type: ignore[return-value]

    def plot_normalization(
        self,
        show_fit: bool = True,
        ax: Axes | None = None,
        **kwargs: Any,
    ) -> Axes:
        """
        Plot normalization-region points; overlay fit curves if show_fit and normalize() was run.

        Parameters
        ----------
        show_fit : bool, optional
            If True and fit params exist, overlay fitted curves.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on; created if None.
        **kwargs
            Passed to DataFrame.plot for the points.

        Returns
        -------
        Axes
        """
        self._validate_schema()
        norm = self.normalization_region()
        if ax is None:
            _fig, ax = plt.subplots()
            ax = cast(Axes, ax)  # type: ignore[assignment]
        style = kwargs.pop("style", [".", "."])
        norm.plot(
            x=self._energy_col,
            y=self._absorbance_cols,
            ax=ax,
            style=style[: len(self._absorbance_cols)] if isinstance(style, (list, tuple)) else style,
            **kwargs,
        )
        if show_fit and self._fit_params:
            energy = norm[self._energy_col].to_numpy()
            mu = norm["bare_atom"].to_numpy()
            mu_sub = norm["bare_atom_substrate"].to_numpy()
            for col in self._absorbance_cols:
                params = self._fit_params.get(col)
                if params is None:
                    continue
                fit_mode = params.get("mode", "full")
                if fit_mode == "full":
                    model = _model_factory_full(mu, mu_sub)
                elif fit_mode == "bare_atom":
                    model = _model_factory_bare_atom(mu)
                else:
                    model = _model_factory_step(params["jump_energy"])
                ax.plot(
                    energy,
                    model(energy, *params["popt"]),
                    label=f"fit {col}",
                )
            ax.legend()
        return cast(Axes, ax)  # type: ignore[return-value]

    def plot(
        self,
        x: str,
        y: str | list[str],
        by: str,
        colorbar: str = "",
        cmap: str | None = "viridis",
        ba_color: str | None = None,
        ax: Axes | None = None,
        title: str | None = None,
        **kwargs: Any,
    ) -> Axes:
        """
        Multi-line plot: one line per group (by), colored by that variable, with optional colorbar.

        Parameters
        ----------
        x : str
            Column for x-axis (e.g. "beamline_energy").
        y : str or list of str
            Column(s) for y-axis (e.g. "mass_absorption_0"). If list, first is used for coloring by group.
        by : str
            Column to group and color by (e.g. "sample_theta").
        colorbar : str, optional
            Colorbar label (e.g. "Angle (deg)"). If empty, uses by column name.
        ba_color : str, optional
            Line color of the bare atom mass absorption curve.
        cmap : str, optional
            Color map for the colorbar.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on; created if None.
        title : str, optional
            Axes title.
        **kwargs
            Passed to ax.plot.

        Returns
        -------
        Axes
        """
        df = self._obj
        if by not in df.columns:
            raise ValueError(f"Column '{by}' not in DataFrame.")
        y_cols = [y] if isinstance(y, str) else list(y)
        for c in [x] + y_cols:
            if c not in df.columns:
                raise ValueError(f"Column '{c}' not in DataFrame.")
        if ax is None:
            _fig, ax = plt.subplots()
            ax = cast(Axes, ax)  # type: ignore[assignment]
        groups = df.groupby(by, sort=True)
        keys = list(groups.groups.keys())
        if len(keys) == 0:
            return ax
        cmap = colormaps[cmap] if cmap is not None else None  # pyright: ignore[reportAssignmentType]
        all_numeric = all(isinstance(k, (int, float)) for k in keys)
        if all_numeric:
            vmin_val = float(df[by].min())
            vmax_val = float(df[by].max())
            if vmin_val == vmax_val:
                norm_obj = Normalize(vmin=vmin_val, vmax=vmax_val + 1.0)
            else:
                norm_obj = Normalize(vmin=vmin_val, vmax=vmax_val)
        else:
            norm_obj = Normalize(vmin=0, vmax=max(len(keys) - 1, 1))
        for i,key in enumerate(keys):
            grp = groups.get_group(key)
            if len(keys) > 1:
                if all_numeric:
                    scalar: float = float(key) if isinstance(key, (int, float)) else 0.0
                else:
                    scalar = float(keys.index(key))
                color = cmap(norm_obj(scalar)) # type: ignore[reportCallIssue]
            else:
                color = "C0"
            for y_col in y_cols:
                ax.plot(
                    np.asarray(grp[x]),
                    np.asarray(grp[y_col]),
                    color=color,
                    **kwargs,
                )
            if ba_color is not None and i == 0:
                ax.plot(
                    np.asarray(grp[x]),
                    np.asarray(grp["bare_atom"]),
                    color=ba_color,
                    **kwargs,
                )
        if len(keys) > 1:
            sm = cm.ScalarMappable(cmap=cmap, norm=norm_obj)
            sm.set_array([])
            plt.colorbar(sm, ax=ax, label=colorbar or by)
        if title is not None:
            ax.set_title(title)
        ax.set_xlabel(x)
        ax.set_ylim(0, None)
        if len(y_cols) == 1:
            ax.set_ylabel(y_cols[0])
        return cast(Axes, ax)  # type: ignore[return-value]


def normalize_by_group(
    df: pd.DataFrame,
    group_column: str = "sample_theta",
    pre_edge: tuple[float, float] = _DEFAULT_PRE_EDGE,
    post_edge: tuple[float, float] = _DEFAULT_POST_EDGE,
    regions: dict[Any, tuple[tuple[float, float], tuple[float, float]]] | None = None,
    absorbance_cols: list[str] | None = None,
    normalization_mode: NormalizationMode = "full",
    jump_energy: float | None = None,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Run normalize() on each group and return the concatenated DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Full NEXAFS DataFrame with a group column (e.g. multiple angles).
    group_column : str, optional
        Column defining groups (e.g. "sample_theta").
    pre_edge, post_edge : tuple of float, optional
        Default (pre, post) used when regions is None or does not specify a group.
    regions : dict or callable, optional
        If dict: group_key -> (pre_edge, post_edge). If callable: group_key -> (pre_edge, post_edge).
        Otherwise same pre_edge, post_edge for all groups.
    absorbance_cols : list of str, optional
        Absorbance columns; default from accessor.
    normalization_mode : "full" | "bare_atom" | "step", optional
        Same as set_regions(normalization_mode=...).
    jump_energy : float, optional
        For "step" mode only; same as set_regions(jump_energy=...).
    inplace : bool, optional
        If True, modify df in place and return it. Otherwise return a copy with new columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with mass_absorption_* columns filled per group.
    """
    cols = absorbance_cols or _DEFAULT_ABSORBANCE_COLS

    def resolve_regions(key: Any) -> tuple[tuple[float, float], tuple[float, float]]:
        if regions is None:
            return pre_edge, post_edge
        if callable(regions):
            return regions(key)
        return regions.get(key, (pre_edge, post_edge))

    def set_and_normalize(sub: pd.DataFrame, pre: tuple[float, float], post: tuple[float, float]) -> None:
        sub.nexafs.set_regions(
            pre_edge=pre,
            post_edge=post,
            absorbance_cols=cols,
            normalization_mode=normalization_mode,
            jump_energy=jump_energy,
        )
        sub.nexafs.normalize()

    if inplace:
        for key, grp in df.groupby(group_column, group_keys=False):
            pre, post = resolve_regions(key)
            sub = grp.copy()
            set_and_normalize(sub, pre, post)
            for col in sub.columns:
                if col.startswith("mass_absorption_"):
                    df.loc[grp.index, col] = sub[col].values
        return df
    parts: list[pd.DataFrame] = []
    for key, grp in df.groupby(group_column, group_keys=False):
        pre, post = resolve_regions(key)
        sub = grp.copy()
        set_and_normalize(sub, pre, post)
        parts.append(sub)
    return pd.concat(parts, axis=0, ignore_index=False)

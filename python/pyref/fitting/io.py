"""Functions to convert dataframes to refnx objects."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pyref.fitting.model import XrayReflectDataset

if TYPE_CHECKING:
    import polars as pl


def to_reflect_dataset(
    df: pl.DataFrame, *, overwrite_err: bool = True
) -> XrayReflectDataset:
    """Convert a pandas dataframe to a ReflectDataset object."""
    if not overwrite_err:
        e = "overwrite_err=False is not implemented yet."
        raise NotImplementedError(e)
    Q = df["Q"].to_numpy()
    R = df["r"].to_numpy()
    # Calculate initial dR
    dR = 0.15 * R + 0.3e-6 * Q
    # Ensure dR doesn't exceed 90% of R to keep R-dR positive
    dR = np.minimum(dR, 0.9 * R)
    ds = XrayReflectDataset(data=(Q, R, dR))
    return ds


class XrayReflectDataset(ReflectDataset):
    """Overload of the ReflectDataset class from refnx."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initialize_polarizations()

    def _initialize_polarizations(self):
        """Initialize s and p polarization datasets and calculate anisotropy."""
        diff = np.diff(self.x)
        # locate where diff is less than 0 and find that index
        idx = np.where(diff < 0)[0] + 1

        if len(idx) > 0:
            self.s = (
                ReflectDataset(
                    (self.x[: idx[0]], self.y[: idx[0]], self.y_err[: idx[0]])
                )
                if self.y_err is not None
                else ReflectDataset((self.x[: idx[0]], self.y[: idx[0]]))
            )
            self.p = (
                ReflectDataset(
                    (self.x[idx[0] :], self.y[idx[0] :], self.y_err[idx[0] :])
                )
                if self.y_err is not None
                else ReflectDataset((self.x[idx[0] :], self.y[idx[0] :]))
            )
        else:
            self.s = ReflectDataset((self.x, self.y, self.y_err))
            self.p = ReflectDataset((self.x, self.y, self.y_err))

        # Calculate average spacing in each dataset to determine tolerance
        s_spacing = float(np.mean(np.diff(self.s.x))) if len(self.s.x) > 1 else 0.0
        p_spacing = float(np.mean(np.diff(self.p.x))) if len(self.p.x) > 1 else 0.0
        # Use half the average spacing as tolerance
        tolerance = 0.5 * max(s_spacing, p_spacing)

        # Merge and sort all q points from both datasets
        all_q = np.sort(np.unique(np.concatenate([self.s.x, self.p.x])))

        # Find q values where both datasets have points (within tolerance)
        q_common = []
        for q in all_q:
            min_diff_s = np.min(np.abs(self.s.x - q))
            min_diff_p = np.min(np.abs(self.p.x - q))
            if min_diff_s <= tolerance and min_diff_p <= tolerance:
                q_common.append(q)

        # Convert to numpy array
        q_common = np.array(q_common)

        # If we have common q points, interpolate both datasets
        if len(q_common) > 0:
            r_s_interp = np.interp(q_common, self.s.x, self.s.y)
            r_p_interp = np.interp(q_common, self.p.x, self.p.y)
        else:
            # No common points, create empty arrays
            q_common = np.array([])
            r_s_interp = np.array([])
            r_p_interp = np.array([])

        _anisotropy = (r_p_interp - r_s_interp) / (r_p_interp + r_s_interp)
        self.anisotropy = ReflectDataset((q_common, _anisotropy))

    @classmethod
    def from_dataframe(cls, spol, ppol, *, overwrite_err) -> XrayReflectDataset:
        """Create an XrayReflectDataset from s and p polarization dataframes."""
        s_x = spol["Q"].to_numpy()
        s_y = spol["r"].to_numpy()
        s_y_err = spol["dR"].to_numpy()
        p_x = ppol["Q"].to_numpy()
        p_y = ppol["r"].to_numpy()
        p_y_err = ppol["dR"].to_numpy()
        return cls.from_arrays(
            x_s=s_x,
            y_s=y_s,
            y_err_s=y_err_s,
            x_p=p_x,
            y_p=y_p,
            y_err_p=y_err_p,
        )

    def plot(self, ax=None, ax_anisotropy=None, **kwargs):  # type: ignore
        """Plot the reflectivity and anisotropy data."""
        if ax is None:
            fig, axs = plt.subplots(
                nrows=2,
                sharex=True,
                figsize=(8, 6),
                gridspec_kw={"height_ratios": [3, 1]},
            )
            ax = axs[0]
            if ax_anisotropy is None:
                ax_anisotropy = axs[1]

        elif ax_anisotropy is None:
            # If only ax was provided but not ax_anisotropy
            fig = ax.figure
            gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0)
            ax_anisotropy = fig.add_subplot(gs[1], sharex=ax)

        if self.s.y[3] != self.p.y[3]:
            # Plot s and p separately
            ax.errorbar(
                self.s.x,
                self.s.y,
                self.s.y_err,
                label=f"{self.name} s-pol",
                marker="o",
                color="C0",
                ms=3,
                lw=0,
                elinewidth=1,
                capsize=1,
                ecolor="k",
            )
            ax.errorbar(
                self.p.x,
                self.p.y,
                self.p.y_err,
                label=f"{self.name} p-pol",
                marker="o",
                color="C1",
                ms=3,
                lw=0,
                elinewidth=1,
                capsize=1,
                ecolor="k",
            )
        else:
            # Plot together if same x values
            ax.errorbar(
                self.x,
                self.y,
                self.y_err,
                label=self.name,
                marker="o",
                color="C0",
                ms=3,
                lw=0,
                elinewidth=1,
                capsize=1,
                ecolor="k",
            )

        ax_anisotropy.plot(
            self.anisotropy.x,
            self.anisotropy.y,
            label=f"{self.name} anisotropy" if self.name else "anisotropy",
            marker="o",
            markersize=3,
            lw=0,
            color="C2",
        )
        ax_anisotropy.axhline(
            0,
            color=plt.rcParams["axes.edgecolor"],
            ls="-",
            lw=plt.rcParams["axes.linewidth"],
        )

        ax.set_yscale("log")
        ax_anisotropy.set_xlabel(r"$q (\AA^{-1})$")
        ax.set_ylabel(r"$R$")
        plt.legend()
        return ax, ax_anisotropy

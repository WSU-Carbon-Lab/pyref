"""
Directory-based NEXAFS IO: file discovery, izero handling, TEY processing, baseline normalization.

Provides NexafsDirectory for loading NEXAFS scan files and get_sample_dfs returning
DataFrames with Energy, PD Corrected, Bare Atom Step, Sample, Angle, Experiment.
No background fit is applied at this layer; see normalization.py for fit schemes.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pint
from periodictable import xsf

_UREG = pint.UnitRegistry()


def _parse_edge_region(
    df: pd.DataFrame,
    region: tuple[float | None, float | None],
    edge_type: str,
) -> pd.DataFrame:
    energy = df["Energy"]
    if edge_type == "pre":
        if region[0] is not None and region[1] is not None:
            mask = (energy >= region[0]) & (energy <= region[1])
        elif region[0] is None and region[1] is not None:
            mask = energy < region[1]
        elif region[0] is not None and region[1] is None:
            mask = energy >= region[0]
        else:
            raise ValueError("At least one endpoint for pre_edge must be specified.")
    elif edge_type == "post":
        if region[0] is not None and region[1] is not None:
            mask = (energy >= region[0]) & (energy <= region[1])
        elif region[0] is not None and region[1] is None:
            mask = energy > region[0]
        elif region[0] is None and region[1] is not None:
            mask = energy <= region[1]
        else:
            raise ValueError("At least one endpoint for post_edge must be specified.")
    else:
        raise ValueError("edge_type must be 'pre' or 'post'")
    return df.loc[mask]


def quality_level_from_rms(
    rms_pct_pre: float,
    rms_pct_post: float,
    *,
    warning_threshold: float = 2.0,
    error_threshold: float = 5.0,
) -> Literal["ok", "warning", "error"]:
    max_rms = max(
        rms_pct_pre if not np.isnan(rms_pct_pre) else 0.0,
        rms_pct_post if not np.isnan(rms_pct_post) else 0.0,
    )
    if max_rms >= error_threshold:
        return "error"
    if max_rms >= warning_threshold:
        return "warning"
    return "ok"


QUALITY_SYMBOL_OK = "ok"
QUALITY_SYMBOL_WARNING = "\u26a0"
QUALITY_SYMBOL_ERROR = "\u2715"


def quality_display_symbol(quality: Literal["ok", "warning", "error"]) -> str:
    if quality == "warning":
        return QUALITY_SYMBOL_WARNING
    if quality == "error":
        return QUALITY_SYMBOL_ERROR
    return QUALITY_SYMBOL_OK


class NexafsDirectory:
    """
    Directory containing NEXAFS scan files (izero and sample_angle_exp naming).

    Discovers izero and sample .txt files, reads tab-separated tables with
    "Time of Day" header, computes I0 and TEY-normalized absorption, and
    provides baseline normalization (pre/post edge) yielding PD Corrected,
    Norm Abs, Bare Atom Step, Mass Abs. columns. Mu(E) from periodictable.xsf.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Path {self.path} does not exist")
        self.izero_store: dict[pd.Timestamp, dict] = {}
        self._selected_izero_timestamp: pd.Timestamp | None = None
        self._last_sample_context: dict | None = None
        self._scan_and_process_izero()
        self.pre_edge: tuple[float | None, float | None] = (None, 280.0)
        self.post_edge: tuple[float | None, float | None] = (360.0, None)

    @staticmethod
    def read_nexafs(file_path: Path) -> pd.DataFrame:
        """
        Read a NEXAFS tab-separated table; header line contains "Time of Day".

        Parameters
        ----------
        file_path : Path
            Path to .txt file.

        Returns
        -------
        pd.DataFrame
            Raw table with Beamline Energy, Photodiode, AI 3 Izero, etc.
        """
        with open(file_path) as f:
            lines = f.readlines()
        header_idx = next(
            (i for i, line in enumerate(lines) if "Time of Day" in line),
            None,
        )
        if header_idx is None:
            raise ValueError(
                f"Could not find 'Time of Day' header in {file_path}"
            )
        table_text = "".join(lines[header_idx:])
        df = pd.read_csv(StringIO(table_text), sep=r"\t", engine="python")
        df["Beamline Energy"] = df["Beamline Energy"].round(1)
        return df

    @staticmethod
    def compute_izero(izer_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute I0 and energy from izero scan columns.

        Parameters
        ----------
        izer_df : pd.DataFrame
            Must have Beamline Energy, Photodiode, AI 3 Izero.

        Returns
        -------
        pd.DataFrame
            Energy, Photodiode, AI 3 Izero, I0, and Timestamp if present.
        """
        out = pd.DataFrame()
        out["Energy"] = izer_df["Beamline Energy"]
        out["Photodiode"] = izer_df["Photodiode"]
        out["AI 3 Izero"] = izer_df["AI 3 Izero"]
        out["I0"] = izer_df["AI 3 Izero"] / izer_df["Photodiode"]
        if "Timestamp" in izer_df.columns:
            out["Timestamp"] = izer_df["Timestamp"]
        return out

    def _scan_and_process_izero(self) -> None:
        izero_files = list(self.path.glob("izero*"))
        new_store: dict[pd.Timestamp, dict] = {}
        for file in izero_files:
            raw_df = self.read_nexafs(file)
            processed = self.compute_izero(raw_df)
            if "Timestamp" in processed.columns:
                timestamp = pd.to_datetime(processed["Timestamp"].iloc[0])
            else:
                timestamp = pd.to_datetime(file.stat().st_mtime, unit="s")
            energy = processed["Energy"].astype(float)
            epu = None
            if "EPU Polarization" in raw_df.columns:
                epu_val = raw_df["EPU Polarization"].iloc[0]
                if pd.notna(epu_val):
                    epu = float(epu_val)
            new_store[timestamp] = {
                "df": processed,
                "energy_min": float(energy.min()),
                "energy_max": float(energy.max()),
                "epu_polarization": epu,
            }
        self.izero_store = new_store

    def _select_izero_nearest(
        self,
        sample_timestamp: pd.Timestamp,
        energy_range: tuple[float, float] | None = None,
        epu_polarization: float | None = None,
    ) -> pd.Timestamp | None:
        if not self.izero_store:
            return None
        candidates: list[tuple[float, pd.Timestamp]] = []
        sample_min, sample_max = energy_range or (0.0, np.inf)
        for ts, entry in self.izero_store.items():
            e_min, e_max = entry["energy_min"], entry["energy_max"]
            if energy_range and (sample_max < e_min or sample_min > e_max):
                continue
            epu = entry.get("epu_polarization")
            if epu_polarization is not None and epu is not None:
                if not np.isclose(epu_polarization, epu, rtol=1e-5, atol=1e-5):
                    continue
            dt = abs((ts - sample_timestamp).total_seconds())
            candidates.append((dt, ts))
        if not candidates:
            candidates = [
                (abs((ts - sample_timestamp).total_seconds()), ts)
                for ts in self.izero_store
            ]
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1] if candidates else None

    @property
    def izero(self) -> pd.DataFrame | None:
        """Current izero DataFrame (selected or nearest to last sample context)."""
        self._scan_and_process_izero()
        if not self.izero_store:
            return None
        if self._selected_izero_timestamp is not None:
            entry = self.izero_store.get(self._selected_izero_timestamp)
            return entry["df"] if entry else None
        if self._last_sample_context:
            ts = self._select_izero_nearest(
                sample_timestamp=self._last_sample_context["timestamp"],
                energy_range=self._last_sample_context.get("energy_range"),
                epu_polarization=self._last_sample_context.get(
                    "epu_polarization"
                ),
            )
            if ts is not None:
                return self.izero_store[ts]["df"]
        latest_ts = max(self.izero_store)
        return self.izero_store[latest_ts]["df"]

    def set_izero_by_timestamp(self, timestamp: pd.Timestamp | None) -> None:
        """Set the izero scan to use by timestamp; None clears selection."""
        self._scan_and_process_izero()
        if timestamp is None:
            self._selected_izero_timestamp = None
        else:
            if timestamp not in self.izero_store:
                raise ValueError(
                    f"Timestamp {timestamp} not found in available izero scans."
                )
            self._selected_izero_timestamp = timestamp

    def set_izero_nearest(
        self,
        sample_timestamp: pd.Timestamp,
        energy_range: tuple[float, float] | None = None,
        epu_polarization: float | None = None,
    ) -> pd.Timestamp | None:
        """Select izero nearest to sample timestamp (and optional energy/EPU)."""
        self._scan_and_process_izero()
        ts = self._select_izero_nearest(
            sample_timestamp=sample_timestamp,
            energy_range=energy_range,
            epu_polarization=epu_polarization,
        )
        if ts is not None:
            self._selected_izero_timestamp = ts
        return ts

    def list_samples(self) -> list[str]:
        """List sample names inferred from non-izero .txt stems (sample_angle_exp)."""
        sample_names: set[str] = set()
        for file in self.path.glob("*.txt"):
            if file.name.startswith("izero"):
                continue
            parts = file.stem.split("_")
            if len(parts) >= 3:
                sample = "_".join(parts[:-2])
                sample_names.add(sample)
        return sorted(sample_names)

    def get_sample_file_info(
        self, sample_name: str
    ) -> list[dict[str, float | int | Path]]:
        """
        Return list of dicts with file, angle, experiment for the sample.

        Expects stem format sample_angle_exp (e.g. name_55deg_0).
        """
        files: list[dict[str, float | int | Path]] = []
        for file in self.path.glob("*.txt"):
            if file.name.startswith("izero"):
                continue
            parts = file.stem.split("_")
            if "_".join(parts[:-2]) != sample_name:
                continue
            try:
                angle_str = parts[-2].strip("deg")
                angle = float(angle_str)
                experiment_hash = int(parts[-1])
                files.append({
                    "file": file,
                    "angle": angle,
                    "experiment": experiment_hash,
                })
            except (ValueError, IndexError):
                continue
        return files

    def process_tey(
        self, df: pd.DataFrame, izero_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        TEY absorption: Intensity = AI3 Izero / TEY signal, PD Corrected = I0 / Intensity.
        """
        processed = pd.DataFrame()
        processed["Energy"] = df["Beamline Energy"]
        processed["Intensity"] = df["AI 3 Izero"] / df["TEY signal"]
        izero_df = izero_df.copy()
        izero_df["Energy"] = izero_df["Energy"].astype(float)
        processed = processed.merge(izero_df, on="Energy", how="left")
        processed["PD Corrected"] = processed["I0"] / processed["Intensity"]
        return processed

    @staticmethod
    def mu(
        energy: np.ndarray,
        formula: str,
        units: str = "g/cm^2",
    ) -> np.ndarray:
        """
        Mass attenuation coefficient mu(E) for formula at given energies (eV).

        Parameters
        ----------
        energy : np.ndarray
            Energy in eV.
        formula : str
            Chemical formula or element, e.g. 'C8H8', 'Si', 'O'.
        units : str
            Output units, default "g/cm^2".

        Returns
        -------
        np.ndarray
            mu in requested units.
        """
        energy = np.asarray(energy, dtype=float)
        energy_kev = energy * 1e-3
        n = xsf.index_of_refraction(formula, energy=energy_kev, density=1.0)
        beta = -n.imag
        hc = 12.3984193
        wavelength_A = hc / energy_kev
        wavelength_cm = wavelength_A * 1e-8
        mu_over_rho = 4 * np.pi * beta / wavelength_cm
        mu_qty = mu_over_rho * _UREG("g/cm^2")
        mu_converted = mu_qty.to(units)
        return np.asarray(mu_converted.magnitude)

    @staticmethod
    def build_mu_arrays(
        energy: np.ndarray,
        formula: str,
        units: str = "g/cm^2",
    ) -> dict[str, np.ndarray]:
        """
        Precompute mu arrays for formula, Si, and O at given energies (eV).

        Returns
        -------
        dict[str, np.ndarray]
            Keys "chemical", "Si", "O"; values mu arrays.
        """
        return {
            "chemical": NexafsDirectory.mu(energy, formula, units),
            "Si": NexafsDirectory.mu(energy, "Si", units),
            "O": NexafsDirectory.mu(energy, "O", units),
        }

    def _parse_edge_region(
        self,
        df: pd.DataFrame,
        region: tuple[float | None, float | None],
        edge_type: str,
    ) -> pd.DataFrame:
        return _parse_edge_region(df, region, edge_type)

    def normalize_tey(
        self,
        df: pd.DataFrame,
        formula: str,
        pre_edge: tuple[float | None, float | None] | None = None,
        post_edge: tuple[float | None, float | None] | None = None,
    ) -> pd.DataFrame:
        """
        Baseline subtract in pre-edge, scale to bare-atom step; add Norm Abs, Bare Atom Step, Mass Abs.
        """
        df = df.copy()
        df["Bare Atom"] = self.mu(df["Energy"].values, formula)
        pre = pre_edge if pre_edge is not None else self.pre_edge
        post = post_edge if post_edge is not None else self.post_edge
        df_pre = self._parse_edge_region(df, pre, edge_type="pre")
        xpre = df_pre["Energy"].to_numpy()
        ypre = df_pre["PD Corrected"].to_numpy()
        barepre = df_pre["Bare Atom"].to_numpy()
        if len(xpre) < 2:
            raise ValueError("Pre-edge region too small for line fit")
        coef, intercept = np.polyfit(xpre, ypre, 1)
        coef_bare, intercept_bare = np.polyfit(xpre, barepre, 1)
        baseline = coef * df["Energy"] + intercept
        baseline_bare = coef_bare * df["Energy"] + intercept_bare
        df["Norm Abs"] = df["PD Corrected"] - baseline
        df["Bare Atom Step"] = df["Bare Atom"] - baseline_bare
        df_post = self._parse_edge_region(df, post, edge_type="post")
        if len(df_post) < 2:
            df["Mass Abs."] = df["Norm Abs"]
            return df
        post_normabs = df_post["Norm Abs"].median()
        pre_normabs = df_pre["Norm Abs"].median()
        edge_jump_normabs = post_normabs - pre_normabs
        post_bare = df_post["Bare Atom Step"].median()
        pre_bare = df_pre["Bare Atom Step"].median()
        edge_jump_bare = post_bare - pre_bare
        scale = (
            edge_jump_bare / edge_jump_normabs
            if not np.isclose(edge_jump_normabs, 0.0)
            else 1.0
        )
        df["Mass Abs."] = df["Norm Abs"] * scale
        return df

    def normalize_sample_angles(
        self,
        dfs: list[pd.DataFrame],
        formula: str,
        pre_edge: tuple[float | None, float | None] | None = None,
        post_edge: tuple[float | None, float | None] | None = None,
    ) -> list[pd.DataFrame]:
        """
        Normalize multiple angle DataFrames consistently (shared pre/post, edge jump scale).
        """
        for df in dfs:
            df["Bare Atom"] = self.mu(df["Energy"].values, formula)
        pre = pre_edge if pre_edge is not None else self.pre_edge
        post = post_edge if post_edge is not None else self.post_edge
        post_region_len = [
            len(self._parse_edge_region(df, post, edge_type="post")) for df in dfs
        ]
        min_post_points = 5
        complete_mask = np.array([n >= min_post_points for n in post_region_len])
        complete_dfs = [df for df, c in zip(dfs, complete_mask) if c]
        if len(complete_dfs) == 0:
            all_pre = pd.concat(
                [
                    self._parse_edge_region(df, pre, edge_type="pre").assign(
                        idx=i
                    )
                    for i, df in enumerate(dfs)
                ]
            )
            xpre = all_pre["Energy"].to_numpy()
            ypre = all_pre["PD Corrected"].to_numpy()
            barepre = all_pre["Bare Atom"].to_numpy()
            if len(xpre) >= 2:
                coef, intercept = np.polyfit(xpre, ypre, 1)
                coef_bare, intercept_bare = np.polyfit(xpre, barepre, 1)
                for df in dfs:
                    baseline = coef * df["Energy"] + intercept
                    baseline_bare = coef_bare * df["Energy"] + intercept_bare
                    df["Norm Abs"] = df["PD Corrected"] - baseline
                    df["Bare Atom Step"] = df["Bare Atom"] - baseline_bare
            for df in dfs:
                df["Mass Abs."] = df.get("Norm Abs", df["PD Corrected"].copy())
            return dfs
        all_pre = pd.concat(
            [
                self._parse_edge_region(df, pre, edge_type="pre").assign(idx=i)
                for i, df in enumerate(complete_dfs)
            ]
        )
        xpre = all_pre["Energy"].to_numpy()
        ypre = all_pre["PD Corrected"].to_numpy()
        barepre = all_pre["Bare Atom"].to_numpy()
        if len(xpre) < 2:
            raise ValueError("Pre-edge region too small for line fit")
        coef, intercept = np.polyfit(xpre, ypre, 1)
        coef_bare, intercept_bare = np.polyfit(xpre, barepre, 1)
        for df in dfs:
            baseline = coef * df["Energy"] + intercept
            baseline_bare = coef_bare * df["Energy"] + intercept_bare
            df["Norm Abs"] = df["PD Corrected"] - baseline
            df["Bare Atom Step"] = df["Bare Atom"] - baseline_bare
        pre_norm = pd.concat(
            [
                self._parse_edge_region(df, pre, edge_type="pre")[
                    ["Norm Abs", "Bare Atom Step"]
                ]
                for df in complete_dfs
            ]
        )
        post_norm = pd.concat(
            [
                self._parse_edge_region(df, post, edge_type="post")[
                    ["Norm Abs", "Bare Atom Step"]
                ]
                for df in complete_dfs
            ]
        )
        pre_median_normabs = pre_norm["Norm Abs"].median()
        post_median_normabs = post_norm["Norm Abs"].median()
        edge_jump_normabs = post_median_normabs - pre_median_normabs
        pre_median_bare = pre_norm["Bare Atom Step"].median()
        post_median_bare = post_norm["Bare Atom Step"].median()
        edge_jump_bare = post_median_bare - pre_median_bare
        scale = (
            edge_jump_bare / edge_jump_normabs
            if not np.isclose(edge_jump_normabs, 0.0)
            else 1.0
        )
        e0 = pre[1] if pre[1] is not None else pre[0]
        post_first = self._parse_edge_region(
            complete_dfs[0], post, edge_type="post"
        )
        k = 0.0
        if len(post_first) >= 2:
            e_post = post_first["Energy"].to_numpy()
            slope_bare = np.polyfit(
                e_post, post_first["Bare Atom Step"].to_numpy(), 1
            )[0]
            slope_massabs = np.polyfit(
                e_post,
                (post_first["Norm Abs"].to_numpy() * scale),
                1,
            )[0]
            k = slope_bare - slope_massabs
        for i, df in enumerate(dfs):
            if complete_mask[i]:
                df["Mass Abs."] = df["Norm Abs"] * scale
                if k != 0.0 and e0 is not None:
                    mask = df["Energy"] > e0
                    df.loc[mask, "Mass Abs."] = (
                        df.loc[mask, "Mass Abs."]
                        + (df.loc[mask, "Energy"] - e0) * k
                    )
            else:
                df["Mass Abs."] = df["Norm Abs"].copy()
        return dfs

    def get_sample_dfs(
        self,
        sample_name: str,
        formula: str = "C8H8",
        pre_edge: tuple[float | None, float | None] | None = None,
        post_edge: tuple[float | None, float | None] | None = None,
    ) -> list[pd.DataFrame]:
        """
        Load and normalize all angle files for a sample.

        Returns list of DataFrames with Energy, PD Corrected, Norm Abs,
        Bare Atom, Bare Atom Step, Mass Abs., Sample, Angle, Experiment.
        """
        files = self.get_sample_file_info(sample_name)
        if files:
            first_df = self.read_nexafs(files[0]["file"])
            energy = first_df["Beamline Energy"].astype(float)
            epu = None
            if "EPU Polarization" in first_df.columns:
                epu_val = first_df["EPU Polarization"].iloc[0]
                if pd.notna(epu_val):
                    epu = float(epu_val)
            if "Timestamp" in first_df.columns:
                sample_ts = pd.to_datetime(first_df["Timestamp"].iloc[0])
            else:
                sample_ts = pd.to_datetime(
                    files[0]["file"].stat().st_mtime, unit="s"
                )
            self._last_sample_context = {
                "timestamp": sample_ts,
                "energy_range": (float(energy.min()), float(energy.max())),
                "epu_polarization": epu,
            }
        izero_df = self.izero
        if izero_df is None:
            raise RuntimeError("No izero scan available.")
        df_list: list[pd.DataFrame] = []
        for info in files:
            df = self.read_nexafs(info["file"])
            processed = self.process_tey(df, izero_df)
            processed["Sample"] = sample_name
            processed["Angle"] = info["angle"]
            processed["Experiment"] = info["experiment"]
            df_list.append(processed)
        return self.normalize_sample_angles(
            df_list, formula, pre_edge, post_edge
        )

    def normalization_quality(
        self,
        dfs: list[pd.DataFrame],
        pre_edge: tuple[float | None, float | None] | None = None,
        post_edge: tuple[float | None, float | None] | None = None,
        ref_col: str = "Bare Atom Step",
        ycol: str = "Mass Abs.",
        scale_ref_percent: float | None = None,
    ) -> dict:
        pre = pre_edge if pre_edge is not None else self.pre_edge
        post = post_edge if post_edge is not None else self.post_edge
        if isinstance(pre, (int, float)):
            pre = (None, float(pre))
        pre = tuple(pre)
        if isinstance(post, (int, float)):
            post = (float(post), None)
        post = tuple(post)
        all_resid_pre: list[pd.Series] = []
        all_resid_post: list[pd.Series] = []
        for df in dfs:
            r = df[ycol] - df[ref_col]
            pre_df = self._parse_edge_region(df, pre, edge_type="pre")
            post_df = self._parse_edge_region(df, post, edge_type="post")
            if len(pre_df):
                all_resid_pre.append(r.loc[pre_df.index])
            if len(post_df):
                all_resid_post.append(r.loc[post_df.index])
        concat_pre = (
            pd.concat(all_resid_pre, ignore_index=True) if all_resid_pre else pd.Series(dtype=float)
        )
        concat_post = (
            pd.concat(all_resid_post, ignore_index=True)
            if all_resid_post
            else pd.Series(dtype=float)
        )
        rms_pre = (
            float(np.sqrt(np.mean(concat_pre**2))) if len(concat_pre) else np.nan
        )
        rms_post = (
            float(np.sqrt(np.mean(concat_post**2))) if len(concat_post) else np.nan
        )
        if scale_ref_percent is None and len(dfs) > 0:
            post_ref = pd.concat(
                [
                    self._parse_edge_region(df, post, edge_type="post")[ref_col]
                    for df in dfs
                ],
                ignore_index=True,
            )
            scale_ref_percent = (
                float(np.median(np.abs(post_ref))) if len(post_ref) else 1.0
            )
        rms_pct_pre = (
            100.0 * rms_pre / scale_ref_percent if scale_ref_percent else np.nan
        )
        rms_pct_post = (
            100.0 * rms_post / scale_ref_percent if scale_ref_percent else np.nan
        )
        return {
            "rms_pre": rms_pre,
            "rms_post": rms_post,
            "rms_pct_pre": rms_pct_pre,
            "rms_pct_post": rms_pct_post,
            "residuals_pre": all_resid_pre,
            "residuals_post": all_resid_post,
            "scale_ref": scale_ref_percent,
        }

    def build_preview_summary(
        self,
        formula: str = "C8H8",
        pre_edge: tuple[float | None, float | None] | None = None,
        post_edge: tuple[float | None, float | None] | None = None,
        warning_threshold: float = 2.0,
        error_threshold: float = 5.0,
    ) -> pd.DataFrame:
        rows: list[dict] = []
        for sample_name in self.list_samples():
            dfs = self.get_sample_dfs(
                sample_name, formula=formula, pre_edge=pre_edge, post_edge=post_edge
            )
            pol = None
            if self._last_sample_context is not None:
                pol = self._last_sample_context.get("epu_polarization")
            for df in dfs:
                q = self.normalization_quality(
                    [df], pre_edge=pre_edge, post_edge=post_edge
                )
                quality = quality_level_from_rms(
                    q["rms_pct_pre"],
                    q["rms_pct_post"],
                    warning_threshold=warning_threshold,
                    error_threshold=error_threshold,
                )
                angle = float(df["Angle"].iloc[0]) if "Angle" in df.columns else np.nan
                exp = (
                    int(df["Experiment"].iloc[0])
                    if "Experiment" in df.columns
                    else np.nan
                )
                energy = df["Energy"].astype(float)
                ev_lo = float(energy.min())
                ev_hi = float(energy.max())
                rows.append(
                    {
                        "Sample": sample_name,
                        "Scan_Frame": exp,
                        "theta": angle,
                        "ev_lo": ev_lo,
                        "ev_hi": ev_hi,
                        "pol": pol if pol is not None else np.nan,
                        "quality": quality,
                    }
                )
        return pd.DataFrame(rows)

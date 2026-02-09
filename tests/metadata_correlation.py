from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from astropy.io import fits


def load_diff_summary(path: Path) -> pd.DataFrame:
    """Load previously computed diff summary."""
    df = pd.read_csv(path)
    df["stem"] = df["file"].str.replace(".npz", "", regex=False)
    return df


def read_headers(fits_path: Path, keys: list[str]) -> dict[str, Any]:
    """Extract selected header values."""
    with fits.open(fits_path) as hdul:
        primary = cast(fits.PrimaryHDU, hdul[0])
        hdr = primary.header
        return {key: hdr.get(key) for key in keys}


def collect_metadata(base: Path, stems: list[str], keys: list[str]) -> pd.DataFrame:
    """Collect header metadata for target stems."""
    records = []
    for stem in stems:
        path = base / f"{stem}.fits"
        meta = read_headers(path, keys)
        meta["stem"] = stem
        records.append(meta)
    return pd.DataFrame(records)


def summarize_correlations(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """Return correlation matrix for selected numeric columns."""
    numeric = df[numeric_cols].astype(float)
    corr_matrix = np.corrcoef(numeric.to_numpy(), rowvar=False)
    idx = pd.Index(numeric_cols, dtype="object")
    return pd.DataFrame(corr_matrix, index=idx, columns=idx)


def write_markdown(table: pd.DataFrame, corr: pd.DataFrame, output: Path) -> None:
    """Write markdown report."""
    lines = [
        "# Metadata correlation",
        "",
        "## Per-file table",
        "",
        _frame_to_markdown(table),
        "",
        "## Correlations",
        "",
        _frame_to_markdown(corr),
        "",
    ]
    output.write_text("\n".join(lines))


def _frame_to_markdown(df: pd.DataFrame) -> str:
    """Render DataFrame as markdown."""
    headers = list(df.columns)
    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    rows = [
        "| " + " | ".join(_format_value(val) for val in df.iloc[i]) + " |"
        for i in range(len(df))
    ]
    return "\n".join([header, separator, *rows])


def _format_value(value: Any) -> str:
    """Format scalar for markdown export."""
    if isinstance(value, float):
        return f"{value:.3g}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def main() -> None:
    """Combine diff stats with FITS metadata."""
    from pyref import get_data_path

    suffix = os.getenv("ANALYSIS_SUFFIX", "")
    data_dir = get_data_path()
    summary = load_diff_summary(data_dir / f"npz_diff_summary{suffix}.csv")
    keys = [
        "EXPOSURE",
        "Camera ROI X",
        "Camera ROI Y",
        "Camera ROI Width",
        "Camera ROI Height",
        "Camera ROI X Bin",
        "Camera ROI Y Bin",
        "Beamline Energy",
        "Sample Theta",
        "CCD Theta",
    ]
    meta = collect_metadata(data_dir, summary["stem"].tolist(), keys)
    merged = summary.merge(meta, on="stem")
    numeric_cols = [
        "diff_max",
        "diff_mean",
        "diff_std",
        "diff_p99",
        "EXPOSURE",
        "Camera ROI X",
        "Camera ROI Y",
        "Camera ROI Width",
        "Camera ROI Height",
        "Camera ROI X Bin",
        "Camera ROI Y Bin",
        "Beamline Energy",
        "Sample Theta",
        "CCD Theta",
    ]
    corr = summarize_correlations(merged, numeric_cols)
    merged.to_csv(data_dir / f"metadata_correlation{suffix}.csv", index=False)
    write_markdown(merged, corr, data_dir / f"metadata_correlation{suffix}.md")
    print(f"merged table -> {data_dir / f'metadata_correlation{suffix}.csv'}")
    print(f"markdown -> {data_dir / f'metadata_correlation{suffix}.md'}")


if __name__ == "__main__":
    main()

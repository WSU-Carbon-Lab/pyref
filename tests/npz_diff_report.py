from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ArrayStats:
    """Aggregate statistics for a numeric array.

    Parameters
    ----------
    shape : tuple[int, ...]
        Array shape.
    dtype : str
        Array dtype as a string.
    min : float
        Minimum value.
    max : float
        Maximum value.
    mean : float
        Mean value.
    std : float
        Standard deviation.
    median : float
        Median value.
    p95 : float
        95th percentile.
    p99 : float
        99th percentile.
    """

    shape: tuple[int, ...]
    dtype: str
    min: float
    max: float
    mean: float
    std: float
    median: float
    p95: float
    p99: float


@dataclass(frozen=True)
class DiffHotspot:
    """Largest-difference pixel information.

    Parameters
    ----------
    y : int
        Row index of hotspot.
    x : int
        Column index of hotspot.
    value : float
        Diff value at hotspot.
    """

    y: int
    x: int
    value: float


def compute_stats(array: np.ndarray) -> ArrayStats:
    """Return descriptive statistics for an array."""
    arr = np.asarray(array)
    return ArrayStats(
        shape=arr.shape,
        dtype=str(arr.dtype),
        min=float(arr.min()),
        max=float(arr.max()),
        mean=float(arr.mean()),
        std=float(arr.std()),
        median=float(np.median(arr)),
        p95=float(np.percentile(arr, 95)),
        p99=float(np.percentile(arr, 99)),
    )


def top_hotspot(array: np.ndarray) -> DiffHotspot:
    """Return the coordinates and value of the largest element."""
    arr = np.asarray(array)
    flat_idx = int(np.argmax(arr))
    y, x = np.unravel_index(flat_idx, arr.shape)
    return DiffHotspot(y=int(y), x=int(x), value=float(arr[y, x]))


def summarize_file(path: Path) -> dict[str, object]:
    """Summarize a single NPZ diff bundle."""
    data = np.load(path)
    pyref_img = np.asarray(data["pyref_img"])
    astropy_img = np.asarray(data["astropy_img"])
    diff = np.asarray(data["diff"])
    pyref_stats = compute_stats(pyref_img)
    astropy_stats = compute_stats(astropy_img)
    diff_stats = compute_stats(diff)
    hotspot = top_hotspot(diff)
    return {
        "file": path.name,
        "pyref_shape": "x".join(str(d) for d in pyref_stats.shape),
        "astropy_shape": "x".join(str(d) for d in astropy_stats.shape),
        "pyref_dtype": pyref_stats.dtype,
        "astropy_dtype": astropy_stats.dtype,
        "pyref_min": pyref_stats.min,
        "pyref_max": pyref_stats.max,
        "pyref_mean": pyref_stats.mean,
        "pyref_std": pyref_stats.std,
        "astropy_min": astropy_stats.min,
        "astropy_max": astropy_stats.max,
        "astropy_mean": astropy_stats.mean,
        "astropy_std": astropy_stats.std,
        "diff_min": diff_stats.min,
        "diff_max": diff_stats.max,
        "diff_mean": diff_stats.mean,
        "diff_std": diff_stats.std,
        "diff_median": diff_stats.median,
        "diff_p95": diff_stats.p95,
        "diff_p99": diff_stats.p99,
        "diff_nonzero": int(np.count_nonzero(diff)),
        "hotspot_y": hotspot.y,
        "hotspot_x": hotspot.x,
        "hotspot_value": hotspot.value,
    }


def write_markdown(records: Iterable[dict[str, object]], output: Path) -> None:
    """Write a human-readable summary table."""
    df = pd.DataFrame(records)
    lines = ["# NPZ diff summary", "", _dataframe_to_markdown(df), ""]
    output.write_text("\n".join(lines))


def write_csv(records: Iterable[dict[str, object]], output: Path) -> None:
    """Write a CSV summary."""
    df = pd.DataFrame(records)
    df.to_csv(output, index=False)


def _format_value(value: object) -> str:
    """Format scalar for markdown export."""
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame as markdown without optional dependencies."""
    headers = list(df.columns)
    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    rows = [
        "| " + " | ".join(_format_value(val) for val in df.iloc[i]) + " |"
        for i in range(len(df))
    ]
    return "\n".join([header, separator, *rows])


def main() -> None:
    """Generate metrics for all NPZ diff bundles."""
    suffix = os.getenv("ANALYSIS_SUFFIX", "")
    base = Path(__file__).resolve().parent
    npz_dir = base / "data" / f"npz{suffix}"
    output_dir = base / "data"
    records: list[dict[str, object]] = []
    for path in sorted(npz_dir.glob("*.npz")):
        records.append(summarize_file(path))
    if not records:
        raise SystemExit("no NPZ files found")
    write_csv(records, output_dir / f"npz_diff_summary{suffix}.csv")
    write_markdown(records, output_dir / f"npz_diff_summary{suffix}.md")
    print(f"Processed {len(records)} NPZ files")
    print(f"CSV: {output_dir / f'npz_diff_summary{suffix}.csv'}")
    print(f"Markdown: {output_dir / f'npz_diff_summary{suffix}.md'}")


if __name__ == "__main__":
    main()

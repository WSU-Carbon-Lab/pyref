"""Shared helpers for ingest profiling and benchmarking.

Exposes a phase-aware progress collector around :func:`pyref.io.readers.ingest_beamtime`
so :mod:`pyref.cli.bench` and other callers render the same markdown table instead of
diverging their timing logic.
"""

from __future__ import annotations

import os
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping


def format_seconds(seconds: float) -> str:
    """Return a compact fixed-decimal seconds string for table rendering.

    Parameters
    ----------
    seconds : float
        Non-negative wall-time value.

    Returns
    -------
    str
        Three-decimal precision under 10 s, two under 100 s, one beyond.
    """
    if seconds >= 100.0:
        return f"{seconds:.1f}"
    if seconds >= 10.0:
        return f"{seconds:.2f}"
    return f"{seconds:.3f}"


@dataclass
class IngestProfile:
    """Accumulated progress-event statistics from a single ingest run.

    Attributes
    ----------
    wall_seconds : float
        Total wall time from just before ``ingest_beamtime`` to just after.
    phases : dict[str, float]
        Cumulative seconds attributed to each ``phase`` event (``startup`` captures
        the initial slice before the first phase event is emitted).
    layout_files : int
        ``total_files`` as reported by the single ``layout`` event, or 0 if none.
    counts : dict[str, int]
        Per-event-kind counts for ``layout``, ``catalog_row``, ``file_complete``.
    first_catalog_row_seconds : float | None
        Offset from ingest start to the first ``catalog_row`` event.
    first_file_complete_seconds : float | None
        Offset from ingest start to the first ``file_complete`` event.
    """

    wall_seconds: float = 0.0
    phases: dict[str, float] = field(default_factory=dict)
    layout_files: int = 0
    counts: dict[str, int] = field(
        default_factory=lambda: {"layout": 0, "catalog_row": 0, "file_complete": 0}
    )
    first_catalog_row_seconds: float | None = None
    first_file_complete_seconds: float | None = None

    @property
    def startup_seconds(self) -> float:
        """Seconds before the first ``phase`` event (catalog open/discovery/layout)."""
        return self.phases.get("startup", 0.0)

    @property
    def headers_seconds(self) -> float:
        """Seconds spent in the parallel ``headers`` phase."""
        return self.phases.get("headers", 0.0)

    @property
    def catalog_seconds(self) -> float:
        """Seconds spent in the ``catalog`` SQLite-writer phase."""
        return self.phases.get("catalog", 0.0)

    @property
    def zarr_seconds(self) -> float:
        """Seconds spent in the ``zarr`` pixel-write phase."""
        return self.phases.get("zarr", 0.0)

    @property
    def unattributed_seconds(self) -> float:
        """Wall time not accounted for by ``startup+headers+catalog+zarr``."""
        accounted = (
            self.startup_seconds
            + self.headers_seconds
            + self.catalog_seconds
            + self.zarr_seconds
        )
        return max(0.0, self.wall_seconds - accounted)


class _PhaseCollector:
    """Internal accumulator that turns streaming events into an :class:`IngestProfile`.

    The collector is stateful and not thread-safe by design: pyref's Rust
    callback is invoked from a single ingest thread under the GIL.
    """

    def __init__(self) -> None:
        self._t0 = time.perf_counter()
        self._phase_start = self._t0
        self._current_phase = "startup"
        self._profile = IngestProfile()

    def on_event(self, event: Mapping[str, Any]) -> None:
        """Handle one ``progress_callback`` dict emitted by ``py_ingest_beamtime``."""
        ev = event.get("event")
        now = time.perf_counter()
        if ev == "phase":
            next_phase = str(event.get("phase", ""))
            self._profile.phases[self._current_phase] = (
                self._profile.phases.get(self._current_phase, 0.0)
                + (now - self._phase_start)
            )
            self._current_phase = next_phase
            self._phase_start = now
            return
        if ev == "layout":
            self._profile.layout_files = int(event.get("total_files", 0))
            self._profile.counts["layout"] += 1
            return
        if ev == "catalog_row":
            if self._profile.first_catalog_row_seconds is None:
                self._profile.first_catalog_row_seconds = now - self._t0
            self._profile.counts["catalog_row"] += 1
            return
        if ev == "file_complete":
            if self._profile.first_file_complete_seconds is None:
                self._profile.first_file_complete_seconds = now - self._t0
            self._profile.counts["file_complete"] += 1

    def finalize(self) -> IngestProfile:
        """Flush the trailing phase slice and return the accumulated profile."""
        now = time.perf_counter()
        self._profile.phases[self._current_phase] = (
            self._profile.phases.get(self._current_phase, 0.0)
            + (now - self._phase_start)
        )
        self._profile.wall_seconds = now - self._t0
        return self._profile


def run_ingest_with_profile(
    beamtime: Path,
    header_items: Iterable[str] | None = None,
    *,
    incremental: bool = False,
) -> tuple[Path, IngestProfile]:
    """Invoke :func:`pyref.io.readers.ingest_beamtime` and return timing profile.

    Parameters
    ----------
    beamtime : pathlib.Path
        Beamtime root directory (ALS layout; contains ``CCD/`` in the flat layout).
    header_items : iterable of str, optional
        FITS header keys to extract; when ``None``, pyref selects its defaults.
    incremental : bool, optional
        Forwarded to :func:`pyref.io.readers.ingest_beamtime`.

    Returns
    -------
    tuple of (pathlib.Path, IngestProfile)
        The returned catalog path and the accumulated profile.
    """
    from pyref.io.readers import ingest_beamtime

    collector = _PhaseCollector()
    keys = list(header_items) if header_items is not None else None
    db = ingest_beamtime(
        beamtime,
        keys,
        incremental=incremental,
        progress_callback=collector.on_event,
    )
    return Path(db), collector.finalize()


@contextmanager
def isolated_catalog_env(
    *,
    enabled: bool = True,
    prefix: str = "pyref-bench-",
) -> Iterator[Path | None]:
    """Temporarily redirect PYREF_CATALOG_DB and PYREF_CACHE_ROOT into a tempdir.

    Parameters
    ----------
    enabled : bool, optional
        When False, yield ``None`` without touching the environment.
    prefix : str, optional
        Prefix used for the backing ``tempfile.TemporaryDirectory``.

    Yields
    ------
    pathlib.Path or None
        Path to the temp directory when enabled, otherwise ``None``.
    """
    if not enabled:
        yield None
        return

    previous = {
        key: os.environ.get(key)
        for key in ("PYREF_CATALOG_DB", "PYREF_CACHE_ROOT")
    }
    with tempfile.TemporaryDirectory(prefix=prefix) as td:
        tdir = Path(td)
        os.environ["PYREF_CATALOG_DB"] = str((tdir / "catalog.db").resolve())
        os.environ["PYREF_CACHE_ROOT"] = str((tdir / "cache").resolve())
        try:
            yield tdir
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def render_markdown_table(profile: IngestProfile) -> str:
    """Return a markdown table + summary footer describing ``profile`` timings.

    Parameters
    ----------
    profile : IngestProfile
        Populated profile from :func:`run_ingest_with_profile`.

    Returns
    -------
    str
        Multi-line string ending with a trailing newline.
    """
    wall = profile.wall_seconds
    rows = [
        ("Before `headers` (DB open, discovery, layout)", profile.startup_seconds),
        ("Phase `headers` (parallel FITS header reads)", profile.headers_seconds),
        (
            "Phase `catalog` (SQLite transaction + `catalog_row` events)",
            profile.catalog_seconds,
        ),
        (
            "Phase `zarr` (FITS pixels read + zarr write + `file_complete`)",
            profile.zarr_seconds,
        ),
        ("Unattributed (measurement gap)", profile.unattributed_seconds),
    ]
    lines: list[str] = []
    lines.append("| Segment | Seconds | Share of wall |")
    lines.append("|---------|--------:|--------------:|")
    for label, sec in rows:
        share = (sec / wall * 100.0) if wall > 0 else 0.0
        lines.append(f"| {label} | {format_seconds(sec)} | {share:.1f}% |")
    lines.append(
        f"| **Wall time total** | **{format_seconds(wall)}** | **100%** |"
    )
    return "\n".join(lines) + "\n"

"""
Beamtime-scoped views of the global SQLite catalog.

Resolves one beamtime root directory to catalog rows via the same ``file://`` URI key
used at ingest time. Complements :mod:`pyref.io.readers` (global ingest and
unscoped :func:`~pyref.io.readers.scan_experiment`).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

from pyref.io.catalog_path import resolve_catalog_path
from pyref.io.experiment_names import parse_fits_stem
from pyref.io.readers import beamtime_ingest_layout, ingest_beamtime

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from pyref.io.readers import FilePath


def _rich_console_for_progress():
    from rich.console import Console

    try:
        from IPython import get_ipython

        if get_ipython() is not None:
            return Console(force_jupyter=True)
    except ImportError:
        pass
    return Console()


def _ingest_beamtime_rich(
    beamtime_path: Path,
    header_items: list[str] | None,
    *,
    worker_threads: int | None,
    resource_fraction: float | None,
) -> Path:
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    layout = beamtime_ingest_layout(beamtime_path)
    scans = layout.get("scans") or []
    total_files = int(layout["total_files"])
    console = _rich_console_for_progress()
    columns = (
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )
    with Progress(
        *columns,
        console=console,
        transient=True,
        redirect_stdout=False,
        redirect_stderr=False,
    ) as progress:
        scan_tasks: dict[int, int] = {}
        for s in scans:
            sn = int(s["scan_number"])
            scan_tasks[sn] = progress.add_task(
                f"[green]scan {sn}[/green]",
                total=int(s["files"]),
            )
        main_id = progress.add_task("[bold cyan]ingest[/bold cyan]", total=total_files)

        def on_progress(d: Mapping[str, Any]) -> None:
            ev = d.get("event")
            if ev == "phase":
                phase = str(d.get("phase", ""))
                progress.update(
                    main_id,
                    description=(f"[bold cyan]ingest[/bold cyan] [dim]({phase})[/dim]"),
                )
                return
            if ev != "file_complete":
                return
            progress.update(main_id, advance=1)
            sn = int(d["scan_number"])
            tid = scan_tasks.get(sn)
            if tid is not None:
                progress.update(tid, advance=1)

        return ingest_beamtime(
            beamtime_path,
            header_items,
            incremental=True,
            worker_threads=worker_threads,
            resource_fraction=resource_fraction,
            progress_callback=on_progress,
        )


def _files_parse_failures_sqlite(catalog_path: Path) -> pl.DataFrame:
    q = """
    SELECT filename, scan_number, parse_flag
    FROM files
    WHERE parse_flag IS NOT NULL
    """
    with sqlite3.connect(str(catalog_path)) as conn:
        cur = conn.execute(q)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    if not rows:
        return pl.DataFrame(
            schema={
                "filename": pl.String,
                "scan_number": pl.Int64,
                "parse_flag": pl.String,
            }
        )
    return pl.DataFrame({cols[i]: [r[i] for r in rows] for i in range(len(cols))})


@dataclass(frozen=True)
class BeamtimeEntriesView:
    """
    High-level index for one beamtime: distinct samples, tag slugs, and scan numbers.

    Attributes
    ----------
    samples : list of str
        Distinct sample names in the beamtime.
    tags : list of str
        Distinct tag slugs attached to files in the beamtime.
    scans : list of tuple[int, str]
        Scan numbers with a short display label (e.g. ``(42, "Scan 42")``).
    """

    samples: list[str]
    tags: list[str]
    scans: list[tuple[int, str]]

    @classmethod
    def from_py_dict(cls, d: dict[str, Any]) -> BeamtimeEntriesView:
        """Build from the dict returned by ``py_beamtime_entries``."""
        scans_raw = d.get("scans") or []
        scans = [(int(a), str(b)) for a, b in scans_raw]
        return cls(
            samples=list(d.get("samples") or []),
            tags=list(d.get("tags") or []),
            scans=scans,
        )


@dataclass
class BeamtimeCatalogView:
    """
    Catalog path, beamtime root, index summary, and frame rows for one beamtime.

    Attributes
    ----------
    catalog_path : pathlib.Path
        SQLite catalog file used for queries.
    beamtime_path : pathlib.Path
        Resolved beamtime root directory.
    entries : BeamtimeEntriesView
        Samples, tags, and scans for this beamtime only.
    frames : polars.DataFrame
        Same schema as an unscoped catalog scan; rows are limited to this beamtime.
    """

    catalog_path: Path
    beamtime_path: Path
    entries: BeamtimeEntriesView
    frames: pl.DataFrame


def list_beamtimes(catalog_path: FilePath | None = None) -> pl.DataFrame:
    """
    List beamtime roots recorded in the catalog (most recently ingested last).

    Parameters
    ----------
    catalog_path : str or pathlib.Path, optional
        Path to ``catalog.db``. When omitted, uses :func:`resolve_catalog_path`.

    Returns
    -------
    polars.DataFrame
        Columns ``beamtime_path`` and ``beamtime_id``.
    """
    from pyref.pyref import py_list_beamtimes

    if catalog_path is not None:
        db = Path(catalog_path).resolve()
    else:
        db = resolve_catalog_path()
    rows = py_list_beamtimes(str(db))
    paths = [a for a, _ in rows]
    ids = [b for _, b in rows]
    return pl.DataFrame({"beamtime_path": paths, "beamtime_id": ids})


def scan_from_catalog_for_beamtime(
    beamtime_path: FilePath,
    catalog_path: FilePath | None = None,
    *,
    sample_name: str | None = None,
    tag: str | None = None,
    scan_numbers: list[int] | None = None,
    energy_min: float | None = None,
    energy_max: float | None = None,
) -> pl.DataFrame:
    """
    Load frame-level catalog metadata for one beamtime directory.

    The beamtime must match a row in ``beamtimes`` via canonical ``file://`` URI, as
    produced at ingest. If the beamtime is unknown, returns an empty DataFrame with
    the usual catalog columns.

    Parameters
    ----------
    beamtime_path : str or pathlib.Path
        Beamtime root directory (e.g. the date folder containing instrument data).
    catalog_path : str or pathlib.Path, optional
        Path to ``catalog.db``; default from :func:`resolve_catalog_path`.
    sample_name : str, optional
        Restrict to this catalog sample name.
    tag : str, optional
        Restrict to this tag slug.
    scan_numbers : list of int, optional
        Restrict to these scan numbers.
    energy_min : float, optional
        Minimum beamline energy (eV) on frames.
    energy_max : float, optional
        Maximum beamline energy (eV) on frames.

    Returns
    -------
    polars.DataFrame
        Same columns as :func:`pyref.pyref.py_scan_from_catalog` for the global case.
    """
    from pyref.pyref import py_scan_from_catalog_for_beamtime

    if catalog_path is not None:
        db = Path(catalog_path).resolve()
    else:
        db = resolve_catalog_path()
    filt: dict[str, Any] = {}
    if sample_name is not None:
        filt["sample_name"] = sample_name
    if tag is not None:
        filt["tag"] = tag
    if scan_numbers is not None:
        filt["scan_numbers"] = scan_numbers
    if energy_min is not None:
        filt["energy_min"] = energy_min
    if energy_max is not None:
        filt["energy_max"] = energy_max
    return py_scan_from_catalog_for_beamtime(
        str(db),
        str(Path(beamtime_path).resolve()),
        filt if filt else None,
    )


def beamtime_entries(
    beamtime_path: FilePath,
    catalog_path: FilePath | None = None,
) -> BeamtimeEntriesView:
    """
    Return distinct samples, tags, and scans for one beamtime root.

    Parameters
    ----------
    beamtime_path : str or pathlib.Path
        Beamtime root directory.
    catalog_path : str or pathlib.Path, optional
        Path to ``catalog.db``; default from :func:`resolve_catalog_path`.

    Returns
    -------
    BeamtimeEntriesView
        Empty lists when the beamtime is not present in the catalog.
    """
    from pyref.pyref import py_beamtime_entries

    if catalog_path is not None:
        db = Path(catalog_path).resolve()
    else:
        db = resolve_catalog_path()
    raw = py_beamtime_entries(str(db), str(Path(beamtime_path).resolve()))
    return BeamtimeEntriesView.from_py_dict(dict(raw))


def read_beamtime(
    beamtime_path: FilePath,
    *,
    catalog_path: FilePath | None = None,
    ingest: bool = False,
    header_items: list[str] | None = None,
    worker_threads: int | None = None,
    resource_fraction: float | None = None,
    show_progress: bool = True,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> BeamtimeCatalogView:
    """
    Optionally ingest a beamtime, then load its catalog index and frame metadata.

    When ``ingest`` is True, runs :func:`pyref.io.readers.ingest_beamtime`, which writes
    to the process-wide catalog (``PYREF_CATALOG_DB`` or default user data path). The
    ``catalog_path`` argument only affects subsequent queries; ensure it matches the
    catalog ingest targets if you set it explicitly.

    Parameters
    ----------
    beamtime_path : str or pathlib.Path
        Beamtime root directory.
    catalog_path : str or pathlib.Path, optional
        Path to ``catalog.db`` for queries; default from :func:`resolve_catalog_path`.
    ingest : bool, optional
        When True, ingest before querying. Default False.
    header_items : list of str, optional
        Passed to :func:`pyref.io.readers.ingest_beamtime` when ``ingest`` is True.
    worker_threads : int, optional
        Parallel ingest worker count; mutually exclusive with ``resource_fraction``.
    resource_fraction : float, optional
        Fraction of available parallelism for ingest readers.
    show_progress : bool, optional
        When ``ingest`` is True and ``progress_callback`` is omitted, show a Rich
        :class:`rich.progress.Progress` display with one task per scan and a main
        ``ingest`` task (``transient=True`` so the display clears when ingest
        finishes). The main task description updates during ``headers`` / ``catalog`` /
        ``zarr`` phases so progress is visible before per-file ``file_complete`` events.
        Ignored when ``ingest`` is False or when ``progress_callback`` is given.
    progress_callback : callable, optional
        When ``ingest`` is True, passed to :func:`pyref.io.readers.ingest_beamtime` and
        replaces the default Rich progress display (``show_progress`` is ignored).

    Returns
    -------
    BeamtimeCatalogView
        ``entries`` and ``frames`` scoped to ``beamtime_path``.
    """
    bt = Path(beamtime_path).resolve()
    if catalog_path is not None:
        cat = Path(catalog_path).resolve()
    else:
        cat = resolve_catalog_path()
    if ingest:
        if progress_callback is not None:
            ingest_beamtime(
                bt,
                header_items,
                incremental=True,
                worker_threads=worker_threads,
                resource_fraction=resource_fraction,
                progress_callback=progress_callback,
            )
        elif show_progress:
            _ingest_beamtime_rich(
                bt,
                header_items,
                worker_threads=worker_threads,
                resource_fraction=resource_fraction,
            )
        else:
            ingest_beamtime(
                bt,
                header_items,
                incremental=True,
                worker_threads=worker_threads,
                resource_fraction=resource_fraction,
            )
    entries = beamtime_entries(bt, cat)
    frames = scan_from_catalog_for_beamtime(bt, cat)
    return BeamtimeCatalogView(
        catalog_path=cat,
        beamtime_path=bt,
        entries=entries,
        frames=frames,
    )


def _stem_from_catalog_file_name(name: str) -> str:
    return Path(name).stem if name.lower().endswith(".fits") else name


def naming_qc_from_frames(frames: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """
    Heuristic filename-vs-catalog checks on a frame-level catalog DataFrame.

    Compares :func:`~pyref.io.experiment_names.parse_fits_stem` on ``file_name`` to
    catalog ``sample_name``, and flags scan numbers that map to multiple sample names.
    This is not the full ``mislabeled_sample`` pipeline from beam reduction.

    Parameters
    ----------
    frames : polars.DataFrame
        Output of :func:`scan_from_catalog_for_beamtime` or compatible columns
        (``file_name``, ``sample_name``, ``scan_number``).

    Returns
    -------
    dict[str, polars.DataFrame]
        Keys ``parse_failures``, ``sample_mismatches``, ``multi_sample_scans``,
        ``db_parse_failures`` (empty; use :func:`naming_qc_with_db_parse_flags` for DB).
    """
    if frames.height == 0:
        empty = pl.DataFrame()
        return {
            "parse_failures": empty,
            "sample_mismatches": empty,
            "multi_sample_scans": empty,
            "db_parse_failures": empty,
        }

    def parsed_sample(stem: str) -> str | None:
        p = parse_fits_stem(stem)
        return p.sample_name if p is not None else None

    stems = frames["file_name"].map_elements(
        lambda s: _stem_from_catalog_file_name(str(s)),
        return_dtype=pl.String,
    )
    parsed = stems.map_elements(
        lambda s: parsed_sample(str(s)),
        return_dtype=pl.String,
    )
    enriched = frames.with_columns(
        file_stem=stems,
        parsed_sample_name=parsed,
    )
    parse_failures = enriched.filter(pl.col("parsed_sample_name").is_null())
    sample_mismatches = enriched.filter(
        pl.col("parsed_sample_name").is_not_null()
        & (pl.col("parsed_sample_name") != pl.col("sample_name"))
    )
    multi = (
        enriched.group_by("scan_number")
        .agg(pl.col("sample_name").n_unique().alias("n_samples"))
        .filter(pl.col("n_samples") > 1)
    )
    return {
        "parse_failures": parse_failures,
        "sample_mismatches": sample_mismatches,
        "multi_sample_scans": multi,
        "db_parse_failures": pl.DataFrame(),
    }


def naming_qc_with_db_parse_flags(
    frames: pl.DataFrame,
    catalog_path: FilePath,
) -> dict[str, pl.DataFrame]:
    """
    Like :func:`naming_qc_from_frames`, plus rows from ``files.parse_flag`` in SQLite.

    Parameters
    ----------
    frames : polars.DataFrame
        Frame-level catalog metadata.
    catalog_path : str or pathlib.Path
        Path to the same ``catalog.db`` used for ``frames``.

    Returns
    -------
    dict[str, polars.DataFrame]
        Same as :func:`naming_qc_from_frames`, with ``db_parse_failures`` populated
        from ``SELECT filename, scan_number, parse_flag FROM files WHERE parse_flag
        IS NOT NULL``.
    """
    out = naming_qc_from_frames(frames)
    cat = Path(catalog_path).resolve()
    out["db_parse_failures"] = _files_parse_failures_sqlite(cat)
    return out


__all__ = [
    "BeamtimeCatalogView",
    "BeamtimeEntriesView",
    "beamtime_entries",
    "list_beamtimes",
    "naming_qc_from_frames",
    "naming_qc_with_db_parse_flags",
    "read_beamtime",
    "scan_from_catalog_for_beamtime",
]

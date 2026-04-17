"""
Daemon lifecycle for ``pyref watch``: PID files, JSON index, logs, subprocess worker.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from pyref.cli.resolve import daemon_key

INDEX_NAME = "index.json"


@dataclass
class DaemonRecord:
    """One row in the daemon index file."""

    pid: int
    beamtime: str
    subpaths: list[str]
    started_at: str
    memory_mb: int
    cpu: int | None
    log_file: str


def _daemons_dir() -> Path:
    from pyref.pyref import py_pyref_data_dir

    d = Path(py_pyref_data_dir()).resolve() / "daemons"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _index_path() -> Path:
    return _daemons_dir() / INDEX_NAME


def _load_index() -> dict[str, Any]:
    p = _index_path()
    if not p.is_file():
        return {"daemons": {}}
    return json.loads(p.read_text(encoding="utf-8"))


def _save_index(data: dict[str, Any]) -> None:
    p = _index_path()
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(p)


def _update_index_record(key: str, rec: DaemonRecord | None) -> None:
    data = _load_index()
    dm = data.setdefault("daemons", {})
    if rec is None:
        dm.pop(key, None)
    else:
        dm[key] = asdict(rec)
    _save_index(data)


def list_records() -> dict[str, DaemonRecord]:
    """
    Return all daemon records keyed by beamtime hash.

    Returns
    -------
    dict
        Mapping of digest key to :class:`DaemonRecord`.
    """
    raw = _load_index().get("daemons") or {}
    out: dict[str, DaemonRecord] = {}
    for k, v in raw.items():
        out[k] = DaemonRecord(**v)
    return out


def start_watch_daemon_subprocess(
    beamtime: Path,
    header_keys: list[str] | None,
    debounce_ms: int,
    subpaths: list[str],
    memory_mb: int,
    cpu: int | None,
    workers: int,
    log_file: Path | None,
) -> int:
    """
    Spawn a detached worker process that runs the Rust file watcher.

    Parameters
    ----------
    beamtime : pathlib.Path
        Beamtime root directory.
    header_keys : list of str, optional
        FITS header keys; default ingest list when omitted.
    debounce_ms : int
        Debounce interval for filesystem events.
    subpaths : list of str
        Optional relative subfolders to watch; empty watches the full beamtime tree.
    memory_mb : int
        Best-effort RSS hint for logging; optional rlimit applied on Linux after import.
    cpu : int or None
        Linux: ``sched_setaffinity`` to this CPU index when set.
    workers : int
        Ingest worker thread count passed to each triggered ingest (typically ``1``).
    log_file : pathlib.Path, optional
        Log path; default ``<daemons>/<key>.log``.

    Returns
    -------
    int
        Worker process PID.
    """
    key = daemon_key(beamtime)
    ddir = _daemons_dir()
    log_path = log_file if log_file is not None else ddir / f"{key}.log"
    spec = {
        "beamtime": str(beamtime.resolve()),
        "header_keys": header_keys,
        "debounce_ms": debounce_ms,
        "subpaths": subpaths,
        "memory_mb": memory_mb,
        "cpu": cpu,
        "workers": workers,
        "log_file": str(log_path),
        "key": key,
    }
    env = os.environ.copy()
    env["PYREF_CLI_WATCH_SPEC"] = json.dumps(spec)
    proc = subprocess.Popen(
        [sys.executable, "-m", "pyref.cli.daemon"],
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    pid_path = ddir / f"{key}.pid"
    pid_path.write_text(str(proc.pid), encoding="utf-8")
    rec = DaemonRecord(
        pid=proc.pid,
        beamtime=str(beamtime.resolve()),
        subpaths=list(subpaths),
        started_at=datetime.now(UTC).isoformat(),
        memory_mb=memory_mb,
        cpu=cpu,
        log_file=str(log_path.resolve()),
    )
    _update_index_record(key, rec)
    return proc.pid


def run_watch_daemon_from_env() -> None:
    """
    Worker entry: read ``PYREF_CLI_WATCH_SPEC`` and block in ``py_run_catalog_watcher_blocking``.

    This function is only intended for the subprocess spawned by
    :func:`start_watch_daemon_subprocess`.
    """
    raw = os.environ.get("PYREF_CLI_WATCH_SPEC")
    if not raw:
        sys.stderr.write("error: PYREF_CLI_WATCH_SPEC missing\n")
        raise SystemExit(2)
    spec = json.loads(raw)
    beamtime = Path(spec["beamtime"])
    header_keys = spec.get("header_keys")
    debounce_ms = int(spec["debounce_ms"])
    subpaths = list(spec.get("subpaths") or [])
    memory_mb = int(spec["memory_mb"])
    cpu = spec.get("cpu")
    workers = int(spec.get("workers", 1))
    log_file = Path(spec["log_file"])
    key = spec["key"]

    log_file.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    fh = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,
        backupCount=2,
        encoding="utf-8",
    )
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root.addHandler(fh)

    try:
        import resource

        if sys.platform.startswith("linux"):
            lim = memory_mb * 1024 * 1024
            try:
                resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
            except OSError as exc:
                logging.warning("rlimit AS not applied: %s", exc)
            if cpu is not None:
                try:
                    os.sched_setaffinity(0, {cpu})
                except OSError as exc:
                    logging.warning("sched_setaffinity not applied: %s", exc)
    except Exception as exc:
        logging.warning("resource tuning skipped: %s", exc)

    from pyref.pyref import CatalogWatcherCancel, py_run_catalog_watcher_blocking

    cancel = CatalogWatcherCancel()

    def handle_sig(_sig: int, _frame: Any) -> None:
        cancel.cancel()

    signal.signal(signal.SIGTERM, handle_sig)
    signal.signal(signal.SIGINT, handle_sig)

    os.environ["PYREF_INGEST_WORKER_THREADS"] = str(max(1, workers))

    try:
        py_run_catalog_watcher_blocking(
            str(beamtime),
            list(header_keys) if header_keys else [],
            debounce_ms,
            subpaths,
            cancel,
        )
    except OSError as exc:
        logging.exception("watcher failed: %s", exc)
        raise SystemExit(2) from exc
    finally:
        pid_path = _daemons_dir() / f"{key}.pid"
        pid_path.unlink(missing_ok=True)
        _update_index_record(key, None)


def stop_daemon(beamtime: Path, wait_s: float = 10.0) -> bool:
    """
    Send ``SIGTERM`` to the watcher PID recorded for ``beamtime``, then ``SIGKILL`` if needed.

    Returns
    -------
    bool
        ``True`` if the process was signaled and exited.
    """
    key = daemon_key(beamtime)
    pid_path = _daemons_dir() / f"{key}.pid"
    if not pid_path.is_file():
        return False
    pid = int(pid_path.read_text(encoding="utf-8").strip())
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pid_path.unlink(missing_ok=True)
        _update_index_record(key, None)
        return True
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pid_path.unlink(missing_ok=True)
            _update_index_record(key, None)
            return True
        time.sleep(0.2)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    pid_path.unlink(missing_ok=True)
    _update_index_record(key, None)
    return True


def tail_log(log_file: Path, n: int, follow: bool) -> None:
    """
    Print the last ``n`` lines of ``log_file``; optionally wait for new lines (simple poll).
    """
    if not log_file.is_file():
        sys.stderr.write(f"error: log file not found: {log_file}\n")
        raise SystemExit(3)
    data = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in data[-n:]:
        sys.stdout.write(line + "\n")
    if not follow:
        return
    pos = log_file.stat().st_size
    try:
        while True:
            time.sleep(0.5)
            with log_file.open(encoding="utf-8", errors="replace") as f:
                f.seek(pos)
                chunk = f.read()
                if chunk:
                    sys.stdout.write(chunk)
                    pos = log_file.stat().st_size
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    run_watch_daemon_from_env()

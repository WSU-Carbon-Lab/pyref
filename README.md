Now archived, see [pxr-reduce](https://github.com/ALS-RSOXS/pxr-reduce) and [refloxide](https://github.com/ALS-RSOXS/refloxide) maintained by the ALS beam-line scientists. 

## Table of Contents

This repository contains the Python package, Rust acceleration layer, and tooling for PRSoXR beamtime ingestion and reduction.

- [`python/pyref`](python/pyref) - User-facing Python API for loading, reduction, and analysis.
- [`src`](src) - Rust backend for high-throughput IO and performance-critical paths.
- [`tests`](tests) - Python test suite and integration checks.
- [`AGENTS.md`](AGENTS.md) - Full project architecture, conventions, and contributor guide.
- [Documentation Site](https://wsu-carbon-lab.github.io/pyref/) - Hosted docs and API references.

[Report an Issue](https://github.com/WSU-Carbon-Lab/pyref/issues/new)

## Quickstart

Install:

```bash
pip install pyref
```

Python `>=3.12` is required.

Load a beamtime and inspect reduced data:

```python
from pathlib import Path
from pyref.loader import PrsoxrLoader

loader = PrsoxrLoader(Path("/path/to/beamtime"))

refl = loader.refl
meta = loader.meta

loader.plot_data()
```

Optional interactive tools:

```python
loader.mask_image()
loader.check_spot()
```

## Development

All development commands use `uv` for Python and `cargo` for Rust:

```bash
git clone https://github.com/WSU-Carbon-Lab/pyref.git
cd pyref
uv sync
uv run pytest
```

Build the experiment browser TUI:

```bash
cargo browser
```

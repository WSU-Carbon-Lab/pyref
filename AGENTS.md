# PyRef Architecture and Data Flow

## Build and test (uv, maturin, cargo only)

- **Install and run tests**: `uv sync` then `uv run pytest tests/test_rust_fits_io.py` (or `uv run pytest` for full suite). `uv sync` builds the Rust extension via the maturin build backend and installs the project.
- **Build wheel only**: `uv run --group dev maturin build`. Output: `target/wheels/pyref-*.whl`.
- **Rust**: Use `cargo build` only for checking compilation of non-cdylib targets (e.g. bins). The Python extension is built by maturin so that linker flags for the extension are correct. Do not rely on `cargo test` for the main crate; it links the cdylib into the test binary and fails with unresolved Python symbols. Rust unit tests live in `src/` (e.g. `src/fits/header.rs`); integration tests in `tests/integration_test.rs` are `#[ignore]` (require Python runtime; validate via pytest instead).
- **TUI binary**: The lib is built with default feature `extension-module` (pyo3/pyo3-polars). Building the standalone TUI must not link Python. Use: `cargo run --bin pyref-tui --no-default-features --features tui`. The `--no-default-features` disables `extension-module`, so the lib is built without pyo3 and the binary links successfully. Running the TUI requires a real TTY (interactive terminal); in a headless or IDE run context you may see "Device not configured".

## Overview

PyRef is a library for reducing 2D X-ray reflectivity detector images into 1D reflectivity signals. The library handles experimental data collected in "stitches" - separate measurement chunks where beamline configuration parameters (higher-order suppressor, exit slits, exposure times) are adjusted to capture reflectivity across multiple orders of magnitude.

## Core Components

### 1. Data Loading (`src/loader.rs`, `python/pyref/io/readers.py`)

The Rust backend (`src/loader.rs`) handles parallel reading of FITS files:
- Reads FITS file headers and image data
- Processes images to extract beam spot locations
- Calculates simple reflectivity using ROI (Region of Interest) analysis
- Combines multiple FITS files into a single Polars DataFrame
- Adds calculated columns (Q-vector from energy and theta)

Key functions:
- `read_experiment()`: Reads all FITS files in a directory
- `read_experiment_pattern()`: Reads FITS files matching a pattern
- `read_multiple_fits()`: Reads specific FITS files
- `combine_dataframes_with_alignment()`: Merges DataFrames with schema alignment

#### FITS DataFrame Accessor (`df.fits`)

For metadata DataFrames from `scan_experiment().collect()`, use the `fits` accessor to load images. Collect your LazyFrame before using: `df = lf.filter(...).collect()` then `df.fits.img[0]`.

- `df.fits.img[i]` / `df.fits.img[slice]`: Raw detector image(s); slice returns iterator
- `df.fits.corrected(idx, bg_rows=10, bg_cols=10)`: Background-corrected image(s)
- `df.fits.filtered(idx, sigma, bg_rows=10, bg_cols=10)`: Background-corrected + gaussian blurred (Rust pipeline)
- `df.fits.custom(idx, callable, **kwargs)`: Apply custom Python callable to image(s)

Background correction uses edge-based subtraction: per-row (left/right) and per-column (top/bottom) with configurable `bg_rows`, `bg_cols`.

### 2. Image Processing (`python/pyref/image.py`, `src/io.rs`)

Image processing reduces 2D detector images to reflectivity values:

**Rust implementation** (`src/io.rs`):
- `subtract_background()`: Row-by-row background subtraction
- `simple_reflectivity()`: Calculates beam signal vs background using ROI
- `process_image()`: Main image processing pipeline

**Python implementation** (`python/pyref/image.py`):
- `reduce_data()`: Full image reduction pipeline (dezinger, filtering, masking)
- `locate_beam()`: Locates beam spot in processed image
- `reduction()`: Calculates reflectivity from masked image and beam spot

Uncertainty in reflectivity originates from:
- Counting statistics in beam signal
- Background subtraction uncertainty
- Exposure time and beam current normalization

### 3. Masking (`python/pyref/masking.py`)

- `InteractiveImageMasker`: Allows interactive rectangular masking of images
- `ImageSeries.mask()`: Automatic mask generation based on CDF of mean image
- Mask defines which pixels contribute to reflectivity calculation

### 4. Main Loader (`python/pyref/loader.py`)

`PrsoxrLoader` orchestrates the data reduction workflow:

- Loads experiment data via `read_experiment()`
- Processes images using the mask
- Creates reflectivity DataFrame with columns: `file_name`, `Q`, `r`, `dr`
- Groups data by `file_name` (each file_name represents a stitch)
- Calculates uncertainty: `dr = sqrt(r)` (Poisson counting statistics)

Key properties:
- `refl`: DataFrame containing reflectivity data
- `meta`: Full metadata DataFrame with images
- `mask`: Image mask for beam isolation

### 5. Uncertainty Propagation (`src/lib.rs`, `python/pyref/utils/__init__.py`)

Uncertainty propagation implemented as Polars plugins:

**Rust functions** (`src/lib.rs`):
- `err_prop_mult()`: Error propagation for multiplication
  - Formula: `σ(xy) = |xy| * sqrt((σx/x)² + (σy/y)²)`
- `err_prop_div()`: Error propagation for division
  - Formula: `σ(x/y) = |x/y| * sqrt((σx/x)² + (σy/y)²)`
- `weighted_mean()`: Weighted average using inverse variance weights
- `weighted_std()`: Weighted standard deviation

**Python interface** (`python/pyref/utils/__init__.py`):
- Exposes Rust functions as Polars expressions
- Used throughout the data processing pipeline

### 6. Catalog (beamtime index)

One SQLite database per beamtime directory (`.pyref_catalog.db`) caches FITS metadata and supports overrides so scans avoid repeated header I/O.

**Source resolution in `scan_experiment(source)`** (Python `python/pyref/io/readers.py`):
- If `source` is a directory and `path / ".pyref_catalog.db"` exists, the scan is served from the catalog (Rust `scan_from_catalog`); no FITS I/O.
- If `source` is a file path to `.pyref_catalog.db`, the scan is from that catalog.
- Otherwise, discovery and header reads use the existing directory/list path (resolve_fits_paths + batched `read_multiple_fits_headers_only`). The LazyFrame schema is the same in both cases so `df.fits` and the loader work unchanged.

**Rust** (`src/catalog/`): Schema (files + overrides), discovery (walkdir, skippable stems), ingest (batch header read, Q/Lambda, upsert, prune), `scan_from_catalog`, `get_overrides`, `set_override`, `list_beamtime_entries`, `query_files`. The TUI (Rust binary) uses these when a beamtime has a catalog; no Python in the TUI process.

**Programmatic APIs** (Python `pyref.io`):
- `ingest_beamtime(beamtime_path, header_items=None, incremental=True)` -> Path to DB
- `get_overrides(catalog_path, path=None)` -> DataFrame
- `set_override(catalog_path, path, sample_name=..., tag=..., notes=...)`
- `query_catalog(catalog_path, sample_name=..., tag=..., experiment_numbers=..., energy_min=..., energy_max=...)` -> DataFrame

Catalog is built by default (feature `catalog`); the TUI feature includes catalog so the binary can read/write the same DB.

**Catalog watch**: When the TUI has a beamtime selected and `.pyref_catalog.db` exists, a catalog watcher runs in the background (Rust `run_catalog_watcher` in `src/catalog/watch.rs`). It uses the `notify-debouncer-mini` crate to watch the beamtime directory for FITS create/modify events, debounces them (about 1.5 s), and runs incremental ingest so new or changed files are added to the catalog without a full rescan. The watcher is stopped when the TUI exits (or when the user navigates away from that beamtime). The TUI feature enables the `watch` feature; the Python wheel does not depend on the watcher.

### 7. Data Stitching (Conceptual)

While explicit stitching code is not fully implemented, the infrastructure supports it:

**Current capabilities:**
- Data grouped by `file_name` (each stitch is a separate file_name)
- Uncertainty propagation functions ready for scale factor calculations
- Weighted statistics for combining overlapping points
- `OverlapError` exception defined for overlap validation

**Stitching workflow (as designed):**
1. Identify overlapping theta/Q values between consecutive stitches
2. Calculate multiplicative scale factor from overlapping region
3. Apply scale factor to subsequent stitch
4. Propagate uncertainty through scaling operation
5. Combine overlapping points using weighted mean
6. Continue recursively for all stitches

**Key considerations:**
- Scale factors determined from overlapping measurements
- Uncertainty propagation through multiplication (scale factor application)
- Weighted combination preserves statistical information
- Must handle cases where overlap is insufficient

### 8. Reflectivity Fitting (`python/pyref/fitting/`)

The fitting module provides:
- `XrayReflectDataset`: Dataset class for reflectivity data with polarization handling
- `ReflectModel`: Model for fitting reflectivity curves
- `Structure`: Layer structure definitions
- Uniaxial and birefringent reflectivity calculations

## Data Flow

```
FITS Files
    ↓
[Rust IO Layer] - Parallel reading, image processing, initial reflectivity
    ↓
(optional) [Catalog] - ingest_beamtime writes .pyref_catalog.db per beamtime; scan_from_catalog reads it
    ↓
scan_experiment(source) - directory/list -> FITS I/O; directory with .pyref_catalog.db or path to .db -> catalog only
    ↓
Polars DataFrame (meta) - All images, metadata, calculated Q (same schema either path)
    ↓
[Python Loader] - Masking, grouping by file_name
    ↓
Reflectivity DataFrame (refl) - Q, r, dr grouped by stitch
    ↓
[Stitching] - Scale factor calculation, uncertainty propagation (conceptual)
    ↓
Combined Reflectivity Dataset - Complete R vs Q curve
    ↓
[Fitting Module] - Model fitting with refnx
```

## Key Design Decisions

1. **Rust backend**: Critical path operations (FITS reading, image processing) implemented in Rust for performance
2. **Polars DataFrame**: Efficient columnar operations, lazy evaluation, built-in parallelization
3. **Uncertainty tracking**: Uncertainty propagated at every step, not added post-hoc
4. **Grouped processing**: Data naturally organized by file_name (stitch) for easy stitching
5. **Plugin architecture**: Uncertainty functions as Polars plugins for seamless integration

## Usage Patterns

### Basic Workflow
1. Initialize loader: `loader = PrsoxrLoader(directory)`
2. Optionally mask images: `loader.mask_image()`
3. Access reflectivity: `refl_df = loader.refl`
4. Process by stitch: `loader.refl.group_by("file_name")`

### Uncertainty Handling
- Use `pyref.utils.err_prop_mult()` and `err_prop_div()` for calculations
- Use `weighted_mean()` and `weighted_std()` for combining overlapping points
- Uncertainty propagates automatically through Polars expressions

### Catalog usage
- Ingest once (or incrementally): `pyref.io.ingest_beamtime(beamtime_path)`; then `scan_experiment(beamtime_path)` uses the DB.
- Overrides: `set_override(catalog_path, path, sample_name=..., tag=..., notes=...)`; resolved values (e.g. for scan) use COALESCE(override, file).
- Query with filters: `query_catalog(catalog_path, sample_name=..., tag=..., experiment_numbers=...)`.

### Stitching (Future Implementation)
- Identify overlapping Q ranges between consecutive file_name groups
- Calculate scale factors from weighted mean of overlapping points
- Apply scale factors with proper uncertainty propagation
- Combine using weighted statistics

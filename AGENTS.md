# PyRef Architecture and Data Flow

## Build and test (uv, maturin, cargo only)

- **Install and run tests**: `uv sync` then `uv run pytest tests/test_rust_fits_io.py` (or `uv run pytest` for full suite). `uv sync` builds the Rust extension via the maturin build backend and installs the project.
- **Build wheel only**: `uv run --group dev maturin build`. Output: `target/wheels/pyref-*.whl`.
- **Rust**: Use `cargo build` only for checking compilation of non-cdylib targets (e.g. bins). The Python extension is built by maturin so that linker flags for the extension are correct. Do not rely on `cargo test` for the main crate; it links the cdylib into the test binary and fails with unresolved Python symbols. Rust unit tests live in `src/` (e.g. `src/fits/header.rs`); integration tests in `tests/integration_test.rs` are `#[ignore]` (require Python runtime; validate via pytest instead).
- **TUI binary**: The lib is built with default feature `extension-module` (pyo3/pyo3-polars). Building the standalone TUI must not link Python. Use: `cargo browser` (alias) or `cargo run --bin pyref-tui --no-default-features --features tui`. The `--no-default-features` disables `extension-module`, so the lib is built without pyo3 and the binary links successfully. Running the TUI requires a real TTY (interactive terminal); in a headless or IDE run context you may see "Device not configured".

## Overview

PyRef is a library for reducing 2D X-ray reflectivity detector images into 1D reflectivity signals. The library handles experimental data collected in "stitches" - separate measurement chunks where beamline configuration parameters (higher-order suppressor, exit slits, exposure times) are adjusted to capture reflectivity across multiple orders of magnitude.

## Terminology (glossary)

See also `CONTRIBUTE.md` for the full glossary.

- **ingest**: Populate the catalog from FITS (discover, read headers, upsert into SQLite). Rust `ingest_beamtime`; Python `pyref.io.ingest_beamtime`.
- **discover**: Find FITS paths under a directory; no header read. Rust `discover_fits_paths`.
- **scan (IO)**: Build a LazyFrame of metadata. `scan_experiment(source)` returns a LazyFrame from catalog or from FITS; "scan from catalog" = read from SQLite (`scan_from_catalog`). Distinct from a CCD Scan (measurement run).
- **scan (experiment)**: A single measurement run with a scan number (e.g. CCD Scan 88169, Scan ID); one reflectivity profile. Not the same as the `scan_experiment()` IO function.
- **read**: FITS file I/O (headers and/or images). Rust `read_fits_headers_only`, `read_multiple_fits_headers_only`, `read_experiment_headers_only`.
- **experiment**: Beamtime directory or a logical group; "experiment number" in headers refers to one scan (run), not the whole beamtime.

## Core Components

### 1. Data Loading (`src/loader.rs`, `python/pyref/io/readers.py`)

The Rust backend (`src/loader.rs`) handles parallel reading of FITS files:
- Reads FITS file headers and image data
- Processes images to extract beam spot locations
- Calculates simple reflectivity using ROI (Region of Interest) analysis
- Combines multiple FITS files into a single Polars DataFrame
- Adds calculated columns (Q-vector from energy and theta)

Key functions:
- `read_fits(source, options)`: High-level eager read. Resolves `source` (file, paths, dir, catalog) with `ResolvePreference`; returns one DataFrame from catalog or from disk.
- `scan_fits(source, options)`: High-level lazy scan. Returns a LazyFrame from catalog (fast) or from disk; use when you want to filter/select before collect.
- `read_fits_metadata_batch(paths, options)`: Canonical batch read (headers + optional calculated domains). Used by ingest and by `read_fits` when source is disk.
- `read_experiment_headers_only()`, `read_multiple_fits_headers_only()`: Lower-level header-only reads.

**Source and options (Polars-style API):**
- `FitsSource`: enum `File(PathBuf)`, `Paths(Vec<PathBuf>)`, `Dir(PathBuf)`, `Catalog(PathBuf)`. Impl `From<PathBuf>`, `From<Vec<PathBuf>>`, `From<&Path>`.
- `ResolvePreference`: `PreferCatalog`, `PreferDisk`, `FromCatalog`, `FromDisk` when both catalog and disk could satisfy the source.
- `ReadFitsOptions` / `ScanFitsOptions`: `header_items`, `header_only`, `add_calculated_domains`, `schema`, `batch_size`, `resolve_preference`, and (for catalog) `catalog_filter`.
- `FitsMetadataSchema`: canonical column names and optional Polars `Schema` for one FITS row; used by `scan_from_catalog` and batch read output.

**Catalog hook:** When source is `Dir(path)` or `Catalog(path)` and `.pyref_catalog.db` exists, `read_fits`/`scan_fits` use it when preference is `FromCatalog` or `PreferCatalog`, otherwise discover and read from disk. All library code must pass `cargo clippy` with no unwrap/expect in non-test code; every public function and module has docstrings.

#### FITS DataFrame Accessor (`df.fits`)

For metadata DataFrames from `scan_experiment().collect()`, use the `fits` accessor to load images. Collect your LazyFrame before using: `df = lf.filter(...).collect()` then `df.fits.img[0]`.

- `df.fits.img[i]` / `df.fits.img[slice]`: Raw detector image(s); slice returns iterator
- `df.fits.corrected(idx, bg_rows=10, bg_cols=10)`: Background-corrected image(s)
- `df.fits.filtered(idx, sigma, bg_rows=10, bg_cols=10)`: Background-corrected + gaussian blurred (Rust pipeline)
- `df.fits.custom(idx, callable, **kwargs)`: Apply custom Python callable to image(s)

Background correction uses edge-based subtraction: per-row (left/right) and per-column (top/bottom) with configurable `bg_rows`, `bg_cols`.

### 2. Image Processing (`python/pyref/image.py`, `src/io.rs`)

Image processing reduces 2D detector images to reflectivity values:

**Rust implementation** (`src/io/mod.rs`):
- Raw preview pipeline: trim edges (remove detector nonlinearity), then row-by-row background (10-pixel left/right strips, colder side per row), then cold-side dark subtraction (top vs bottom, single scalar). Used by TUI preview and `materialize_image`.
- `subtract_background()`: Legacy row-by-row background subtraction (full-size output; interior only corrected).
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

One SQLite database per beamtime directory (`.pyref_catalog.db`) caches FITS metadata and supports overrides so scans avoid repeated header I/O. A **new layout** uses a single catalog per parent directory.

**Catalog layout (legacy vs new):**
- Legacy: DB at beamtime_dir/.pyref_catalog.db; schema uses files table + overrides.
- New: DB at parent/.pyref/catalog.db for multiple beamtimes; schema uses normalized bt_* tables only (bt_beamtimes, bt_samples, bt_scans, bt_streams, bt_scan_points, etc.). Ingest writes only to bt_* for new layout (no files table).
- resolve_catalog_path(beamtime_dir) returns new path if it exists, else legacy path; new path is chosen when parent exists and is not root.
- Beamtimes list: launcher uses ~/.pyref/beamtime_index.sqlite3; list_beamtimes_from_catalog(db_path) returns beamtimes from a given catalog (bt_beamtimes) for the new layout.
- Zarr store: same location as catalog (parent/.pyref/zarr/<beamtime_key>/); one subdir per beamtime.
- FITS discovery: case-insensitive .fits extension; failed FITS reads are not silently dropped during ingest (first error returned with path).

**Source resolution in `scan_experiment(source)`** (Python `python/pyref/io/readers.py`):
- If `source` is a directory and `path / ".pyref_catalog.db"` exists, the scan is served from the catalog (Rust `scan_from_catalog`); no FITS I/O.
- If `source` is a file path to `.pyref_catalog.db`, the scan is from that catalog.
- Otherwise, discovery and header reads use the existing directory/list path (resolve_fits_paths + batched `read_multiple_fits_headers_only`). The LazyFrame schema is the same in both cases so `df.fits` and the loader work unchanged.

**Rust** (`src/catalog/`): Schema (files + overrides), discovery (walkdir, skippable stems), ingest (batch header read, Q/Lambda, upsert, prune), `scan_from_catalog`, `get_overrides`, `set_override`, `list_beamtime_entries`, `query_files`. The TUI (Rust binary) uses these when a beamtime has a catalog; no Python in the TUI process.

**Programmatic APIs** (Python `pyref.io`):
- `ingest_beamtime(beamtime_path, header_items=None, incremental=True)` -> Path to DB
- `get_overrides(catalog_path, path=None)` -> DataFrame
- `set_override(catalog_path, path, sample_name=..., tag=..., notes=...)`
- `query_catalog(catalog_path, sample_name=..., tag=..., scan_numbers=..., energy_min=..., energy_max=...)` -> DataFrame

Catalog is built by default (feature `catalog`); the TUI feature includes catalog so the binary can read/write the same DB.

**Catalog watch**: When the TUI has a beamtime selected and `.pyref_catalog.db` exists, a catalog watcher runs in the background (Rust `run_catalog_watcher` in `src/catalog/watch.rs`). It uses the `notify-debouncer-mini` crate to watch the beamtime directory for FITS create/modify events, debounces them (about 1.5 s), and runs incremental ingest so new or changed files are added to the catalog without a full rescan. The watcher runs only for the current beamtime; when the user navigates away, the watcher is stopped. When the watcher triggers an ingest, the TUI shows "Updating catalog..." in the nav line and reloads the table when ingest completes. The TUI feature enables the `watch` feature; the Python wheel does not depend on the watcher.

**Directory layout**: `$HOME/.config/pyref/` (XDG config) holds user preferences: `tui.toml` (keymap, theme, layout, `last_root`, `recent_roots`, selection export). Override with `PYREF_TUI_CONFIG`. `$HOME/.pyref/` holds data/index: `beamtime_index.sqlite3` (central index of indexed beamtimes for the launcher). Optional `pyref.toml` and subdirs `cache/`, `logs/` are reserved for future use. If `HOME` is unset, the beamtime index uses `./.pyref/` as fallback.

**TUI (pyref-tui)**: Startup: with no CLI argument, the TUI shows the launcher (list of indexed beamtimes from `~/.pyref/beamtime_index.sqlite3`, most recent first). Enter opens the selected beamtime; [o] opens the "Open directory" dialog. With a CLI argument, the TUI opens that beamtime directly. Open directory: hybrid dialog with path input (Tab autocomplete) and scrollable folder list (.. and subdirs); Enter on a valid path runs ingest, registers the beamtime in the central index, and opens that beamtime. While ingesting (from launcher or from beamtime view), the nav line shows "Ingesting N/M...". If there is no catalog in the opened directory, the TUI shows an empty state and the message to run "Ingest directory" (key [i]) or `pyref.io.ingest_beamtime(path)` from Python. The beamtime browse panel lists **reflectivity profiles**: one row per experiment (scan) for a given sample, tag, and polarization. Two scan modes are supported: **fixed-energy (theta scan)** where sample theta varies and energy is fixed, and **fixed-angle (energy scan)** where energy varies and theta is fixed or limited to a few angles. The table shows Emin (eV), Emax (eV), Type (theta-scan / E-scan), theta min/max, frame count, and duration. Scan type is inferred from the data using energy/theta range tolerances, distinct-value counts, and an Izero heuristic (many points at theta near zero indicate a theta scan). Each row is a single reflectivity profile; Enter expands it to show the underlying FITS files in a table (Scan, Frame, pol, E (eV), sample theta). When expanded, j/k scroll the file list; Enter collapses. Rename/Retag write catalog overrides via `set_override` and the table reloads. On exit, the TUI saves config to `PYREF_TUI_CONFIG` or `~/.config/pyref/tui.toml`, including `last_root`, `selected_samples`, `selected_tags`, and `selected_scan_numbers` (when on beamtime view). Scripts can read that config and use `scan_experiment(last_root).filter(...)` with the same sample/tag/scan filters to match the TUI view.

**Parallel ingest (feature `parallel_ingest`, enabled by default)**: Enables `crossbeam-channel` and a pipelined ingest path. The TUI/browser is built with `--features tui`, which includes `parallel_ingest`, so `ingest_beamtime` uses the pipelined path and partitioned discovery when applicable. Discovery can use partitioned parallel walks over top-level subdirs (`discover_fits_paths_parallel`) when multiple subdirs exist; results are merged and sorted by path. A single thread holds the SQLite connection; a reader thread sends bounded batches (sync channel capacity 2) so FITS reads overlap with batched commits. For the new catalog layout only (`is_new_catalog_layout`), the hot path can use row structs (`BtIngestRow` from `build_bt_ingest_row` / `read_multiple_fits_headers_only_rows`) and `upsert_bt_batch_rows` instead of building Polars DataFrames per batch. The DataFrame path (`read_fits_metadata_batch`, `upsert_bt_batch`) remains for Python and legacy layout. Build without Python: `cargo check --no-default-features --features catalog,parallel_ingest`.

**Deferred work (calculated columns, beam spot, zarr)**: Catalog ingest stores header fields and source path/mtime in `bt_scan_points`. Calculated columns (e.g. Q, Lambda) can be deferred to export (Parquet/HDF5) rather than computed during ingest. Beam spot columns in `bt_scan_points` stay NULL until a processing stage updates them. Zarr materialization (`materialize_beamtime`) is an explicit stage that re-reads FITS from paths recorded in the catalog; a single combined pass (one read for catalog row plus zarr chunk) is optional and not the default, so back-to-back full-tree reread for catalog then zarr implies two NAS reads unless data is copied local first.

**NAS usage**: Prefer one open/read per file for catalog metadata ingest. Parallel discovery reduces wall time on wide trees; batch transactions on the single writer reduce fsync churn. For repeated materialize runs, copy beamtime tree to local storage then run zarr materialize to avoid hammering NAS.

**FITS Preview (three-panel, marquee)**: When a FITS file is selected in the TUI, a preview window shows one figure with three panels (non-macOS: egui; macOS: single composite PNG). Left panel: Raw image (trimmed, then row-by-row background with 10-pixel strips, then cold-side dark subtraction). Middle panel: Gaussian-filtered image. Right panel: 4-sigma crop around the fitted beam center with 1/2/3/4-sigma Gaussian fit contours overlaid. Sigma for the filter is derived from the 2D Gaussian fit (average of row/col sigma, clamped to 0.5--20.0); if the fit fails, a default sigma (2.0) is used and the right panel shows a center crop. On non-macOS, the user can optionally draw a marquee (drag a rectangle) on the left or middle panel; on mouse release the ROI is stored for that path and the fit is re-run inside the ROI, updating all three panels and contours. ROI is per-path and in-memory for the session.

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

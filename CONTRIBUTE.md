## Glossary

Terms used for FITS I/O, catalog, and experiment data.

- **ingest**  
  Populate the catalog (SQLite) from FITS files: discover paths, read headers, add derived columns (e.g. Q, Lambda), then upsert and prune. One catalog per beamtime directory. Rust: `ingest_beamtime`; Python: `pyref.io.ingest_beamtime`.

- **discover**  
  Find FITS file paths under a directory (e.g. recursive walk, filter by extension and skippable stems). No header or image read. Rust: `discover_fits_paths`.

- **scan (IO)**  
  Build a LazyFrame of experiment metadata. The operation `scan_experiment(source)` either reads from the catalog (if `.pyref_catalog.db` exists) or discovers and reads FITS headers; in both cases it returns a LazyFrame with the same schema. "Scan from catalog" means reading that LazyFrame from SQLite (Rust: `scan_from_catalog`). No FITS disk read when the catalog is used.

- **scan (experiment)**  
  A single measurement run: one CCD Scan with a scan number (e.g. "CCD Scan 88169", Scan ID in headers). One such run is one reflectivity profile (fixed energy or fixed angle). This is distinct from the IO operation `scan_experiment()`.

- **read**  
  Actual FITS file I/O: read headers and/or image data from disk. Used for single files, a list of paths, or a directory. Rust: `read_fits_headers_only`, `read_multiple_fits_headers_only`, `read_experiment_headers_only`.

- **experiment**  
  A beamtime directory (root containing FITS and optionally `.pyref_catalog.db`), or a logical group of runs. "Experiment number" in metadata (e.g. Scan ID) identifies a single scan (measurement run), not the whole beamtime.

## Rust

- Library code must not use `unwrap()` or `expect()`; use `Result` and `?`. Run `cargo clippy --lib` and `cargo fmt`.
- Public functions, modules, and files should have docstrings (e.g. `///` and `//!`).

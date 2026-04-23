# pyref Contributor Quickstart Guide

`pyref` is a Python library for reducing and analyzing polarized resonant soft X-ray reflectivity (PRSoXR) data collected at ALS Beamline 11.0.1.2, Lawrence Berkeley National Laboratory. The library couples a Python interface with a Rust backend via PyO3 bindings to achieve the parallel throughput required for large beamtime datasets. It is organized into three primary components: `IO`, which handles raw data ingestion and cataloging; `Reduction`, which reduces 2D detector images into 1D reflectivity profiles; and `Fitting`, which fits reduced profiles against optical models. Each component is designed to be independently extensible.

## Technical Jargon

- **Beamtime**: An experimental allocation at the ALS during which a user group collects data across multiple samples and scans, typically spanning one to several days.
- **Sample**: A physical thin-film specimen characterized by a preparation recipe and a set of deposition conditions. Samples are grouped into series, where each member shares a common recipe and is distinguished by one or more user-assigned tags in the filename.
- **Scan**: A sequential series of frames collected continuously by the instrument under a common set of experimental parameters. Scans are nominally associated with a single sample and a single experiment type.
- **Profile**: A reduced 1D reflectivity curve extracted from a scan, expressed as intensity vs. Q, intensity vs. 2theta, or intensity vs. energy. A single scan may contain multiple profiles collected at different fixed energies or fixed angles.
- **Frame / Image**: A single detector acquisition within a scan, stored as a 2D CCD image accompanied by a FITS header containing motor positions, analog input (AI) values, timestamps, and other metadata.
- **Motor**: A physical positioning device controlling sample theta, CCD theta, beamline energy, steering mirrors, slits, or other instrument components.
- **AI (Analog Input)**: A scalar value recorded by one of the instrument's analog input channels. Relevant AI channels include beam current (ring current), upstream gold mesh absorption current (Ai 3 Izero), photodiode signal, CCD temperature, and TEY signals.
- **Beamspot**: The location of the specularly reflected beam on the 2D detector image, typically a compact Gaussian intensity distribution.
- **ROI (Region of Interest)**: A small rectangular subregion of the detector image, typically 10x10 pixels, centered on the beamspot and used to integrate the reflected beam intensity.
- **Direct Beam**: A series of frames collected with the sample at theta = 0 degrees, where the incident beam hits the detector directly. Used to establish the incident beam intensity I0 as a function of energy or exposure time.
- **I0**: The incident beam intensity used to normalize a reflectivity profile. For fixed-energy scans, I0 is extracted from direct beam frames at the start of the scan. For fixed-angle scans, I0 may be sourced from a separate dedicated scan.
- **I0 Point**: A frame collected as part of the direct beam measurement sequence. I0 points are identified by Sample Theta = 0 and are used both to normalize the reflectivity and to characterize the counting statistics of the incident beam.
- **Stitch Point**: A frame collected at the moment the independent variable reverses direction, signaling the start of a new measurement stitch. Stitch points are typically repeated several times to allow the motors to settle, and they establish the baseline statistics for the new stitch region.
- **Overlap Point**: A frame whose independent variable value falls within the range already covered by the preceding stitch. Overlap points are used to compute the multiplicative scaling factor that aligns the new stitch to the previous one.

## Fundamental Concepts

### IO Module

The IO module is responsible for ingesting raw FITS files, cataloging their contents into a structured SQLite database, and exposing the resulting data through a lazy interface that returns pandas or polars DataFrames on demand. By default, the module expects the FITS file conventions and directory structures produced by ALS Beamline 11.0.1.2. New beamline formats can be added by extending the IO layer.

All performance-critical IO operations are implemented in Rust and exposed to Python via PyO3 bindings. Rust is responsible for parallel FITS file reading, header card extraction, image data loading, filename parsing, directory traversal, Diesel-managed SQLite catalog construction, and zarr archive management. Python is responsible for the user-facing query interface, DataFrame construction from catalog results, and any logic that does not require parallel throughput. This boundary is a design constraint, not a guideline: do not implement parallelism in Python and do not implement user-facing query logic in Rust.

The catalog database is managed by [Diesel](https://diesel.rs) with the SQLite backend. The schema is defined in `src/schema.rs` and all database interactions must go through Diesel's type-checked query builder. Raw SQL is prohibited except inside Diesel migration files. SQLite foreign key enforcement is off by default; the Rust connection initializer must execute `PRAGMA foreign_keys = ON` on every new connection before any other statement.

The IO module is accessed via `pyref.io` and the cataloging subsystem via `pyref.io.catalog`.

### Reduction Module

The Reduction module converts per-frame 2D detector images into normalized, stitched 1D reflectivity profiles. Reduction proceeds in three sequential stages.

The first stage localizes the beamspot in each 2D detector image, integrates the ROI intensity, and subtracts the background estimated from a dark region of the detector. The second stage normalizes the extracted beam intensity against I0, beam current, exposure time, and the upstream gold mesh absorption current (Ai 3 Izero). The third stage identifies the scan domain (fixed energy or fixed angle), classifies each frame as an I0 point, stitch point, or overlap point, computes per-stitch scaling corrections from the weighted mean of overlap regions, and assembles the stitched profile.

The Reduction module is accessed via `pyref.reduction`. The primary user-facing class is `PrsoxrLoader`.

### Fitting Module

The Fitting module fits reduced reflectivity profiles against optical layer-stack models. The primary backend is `refnx`, which implements the 4x4 transfer matrix method for anisotropic and resonant systems. The module is responsible for model definition, parameter specification, constraint enforcement, objective function construction, fitting algorithm selection, and output formatting. The fitting module is accessed via `pyref.fitting`.

## Reduction Subtleties

### Uncertainty Quantification

Each frame is a photon-counting measurement. The raw intensity in each detector pixel follows a Poisson distribution to first approximation, giving a per-pixel standard deviation equal to the square root of the pixel count. Two classes of non-Poissonian noise are also present and must be accounted for.

Systematic noise originates from the detector itself (readout noise, dark current, stray light) and is characterized using a dark region of the detector image that is far from the beamspot. The mean and variance of this dark region are used to estimate the per-pixel systematic noise floor, which is subtracted from the ROI intensity and propagated into the final uncertainty.

Random non-Poissonian noise is characterized by the Fano factor, defined as the ratio of the observed variance in I0 measurements to the hypothesized Poissonian variance at the same intensity level. The Fano factor is computed as a function of incident energy from the ensemble of I0 frames within a scan, and is applied as a multiplicative scale on the Poissonian uncertainty for all frames at that energy. Agents implementing or modifying the uncertainty pipeline must propagate both the dark-region contribution and the Fano-scaled Poissonian contribution in quadrature at every reduction step. Silent variance truncation or implicit dtype coercion that reduces numerical precision is a correctness bug.

### Beamspot Localization

Beamspot localization is applied to each frame independently. The algorithm proceeds in the following fixed order and agents must not reorder or skip steps without explicit justification.

First, camera edge artifacts are removed by zeroing or masking a fixed border of pixels around the image perimeter. Second, a row-by-row background subtraction is applied: for each row, the median of a set of dark columns (columns known to be outside the beamspot region) is subtracted from all pixels in that row. Third, a column-by-column background subtraction is applied analogously. Fourth, a Gaussian filter is applied to suppress residual high-frequency noise. Fifth, a 2D peak fitting routine locates the beamspot centroid, integrated intensity, and fit standard deviation. The background intensity and its uncertainty are extracted from a designated dark region of the post-subtraction image.

Failed detections occur when the peak fitter cannot identify a credible Gaussian peak above the noise floor. A detection is considered failed when the fitted peak amplitude is less than a configurable multiple of the dark region standard deviation, or when the fitted centroid falls outside the detector boundary. Failed detections must be flagged in the BeamFinding Table and must not silently propagate NaN or zero values into the Reflectivity Table. Beamspot drift across a scan is expected and is not itself a failure condition; drift is characterized by fitting a linear model to the centroid coordinates as a function of Q or theta. Frames where the centroid deviates from the linear trend by more than a configurable threshold are flagged separately.

### Scan Type and Domain Identification

The scan domain is determined by inspecting the motor trajectory across all frames in a scan. The classification procedure is as follows.

A scan is classified as a fixed-energy reflectivity scan when a leading block of frames has Sample Theta = 0 and a constant beamline energy (these are the I0 frames), followed by frames in which Sample Theta and CCD Theta increase monotonically (subject to stitch reversals). A scan is classified as a fixed-angle reflectivity scan when either a leading block of frames has Sample Theta = 0 and a varying beamline energy (I0 frames collected as a function of energy), or the scan contains no I0 block at all and beamline energy varies monotonically throughout. A multi-profile scan is not a distinct scan type but is a repetition of the above patterns within a single experimental scan: the instrument completes one full fixed-energy or fixed-angle sweep, then changes the fixed parameter (energy or angle) and repeats the sweep. Multi-profile scans are decomposed into their constituent profiles during reduction, each profile being treated as an independent fixed-energy or fixed-angle scan.

Once the domain is identified, stitch points are located by finding frames where the independent variable decreases relative to the preceding frame. Overlap points are the initial frames of a new stitch whose independent variable values fall within the range already covered by the preceding stitch. The scaling correction for each stitch is the weighted mean of the reflectivity values at the overlap points, where the weights are the inverse squared uncertainties of those frames.

## Filename Parsing

The cataloging system parses FITS filenames to extract the sample name, zero or more tags, the scan number, and the frame number. The parsing contract is as follows and must be implemented exactly as specified.

The frame number is always the five-digit zero-padded integer to the right of the last hyphen in the filename stem (before the `.fits` extension). The scan number is always the five-digit zero-padded integer immediately to the left of that hyphen. The remainder of the stem to the left of the scan number is the concatenation of the sample name and any tags, optionally separated by underscores or hyphens. Because no separator is guaranteed between the sample name and the scan number, the scan number anchor is the five digits immediately left of the hyphen; the parser must split there first before attempting to tokenize the sample name and tags. The following filename patterns are all valid and must be handled without special-casing individual formats.

```
<sample_name>_<tag1>_<tag2>_<scan_number>-<frame_number>.fits
<sample_name>-<tag1>-<tag2>-<scan_number>-<frame_number>.fits
<sample_name>_<scan_number>-<frame_number>.fits
<sample_name><scan_number>-<frame_number>.fits
<sample_name><tag1><tag2><scan_number>-<frame_number>.fits
<sample_name>_<tag1>-<tag2>_<tag3>_<scan_number>-<frame_number>.fits
```

The number of tags is unbounded. Tags may contain alphanumeric characters and hyphens. Parsing failures must be logged and the offending file flagged in the File Table rather than silently skipped or allowed to panic.

## Directory Layout Traversal

Two directory layouts are supported. The cataloging system must detect which layout is present by inspection and handle both without user configuration.

The first layout places each scan in its own instrument subdirectory within a date-grouped scan directory. Detection criterion: the beamtime root contains one or more date directories, each of which contains one or more scan directories (named with a scan number prefix), each of which contains an instrument subdirectory named either `CCD` or `Axis Photonique`. FITS files live inside the instrument subdirectory.

```
<beamtime_root>/
    <date_dir>/
        CCD Scan <scan_number>/
            CCD/
                <sample>_<tags>_<scan>-<frame>.fits
            <sample>_<tags>_<scan>-AI.txt
        CCD Scan <scan_number>/
            Axis Photonique/
                <sample>_<tags>_<scan>-<frame>.fits
            <sample>_<tags>_<scan>-<frame>_AI.txt
```

The second layout places all FITS files in a single flat instrument directory directly under the beamtime root. Detection criterion: the beamtime root contains a directory named `CCD` or `Axis Photonique` that holds FITS files from multiple scan numbers.

```
<beamtime_root>/
    CCD/
        <sample1>_<tags>_<scan1>-<frame>.fits
        <sample2>_<tags>_<scan2>-<frame>.fits
    <sample1>_<tags>_<scan1>-AI.txt
    <sample2>_<tags>_<scan2>-AI.txt
```

AI text files are supplementary and are not required for cataloging. If present, they should be associated with their scan by matching the scan number extracted from the filename. If neither layout is detected, the cataloging system must emit a structured error identifying the unrecognized layout rather than silently producing an empty catalog.

## I/O Operations and Cataloging

Connecting individual frames back to their originating sample, scan, and beamtime requires meticulous bookkeeping that is impractical to maintain manually across a full beamtime. `pyref` provides an automated cataloging system that ingests a beamtime directory and populates a Diesel-managed SQLite database encoding this hierarchy. The database is the structural backbone for all downstream reduction and fitting workflows. Users interact with it primarily through the lazy DataFrame interface exposed by `pyref.io`, which allows them to filter by sample name, tag, energy, or angle and receive a polars or pandas DataFrame containing the relevant frame metadata and image retrieval handles.

### Catalog and Cache Storage

#### Default: catalog and zarr cache under `~/.config/pyref` (single tree)

By default, `pyref` maintains a single persistent catalog that accumulates every beamtime the user has ever ingested. The **catalog** and **local zarr cache** share the same config root (not macOS “Application Support” unless you override with `PYREF_CATALOG_DB` / `PYREF_CACHE_ROOT`):


| Scope                  | Default path                                                                                                                                                                                                                                                                                                                                                                    |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `catalog.db`           | On macOS, always `~/.config/pyref/catalog.db`. On Linux and Windows, `$XDG_CONFIG_HOME/pyref/catalog.db` when `XDG_CONFIG_HOME` is set; otherwise `~/.config/pyref/catalog.db` (on Windows, `~` is the user profile, e.g. `C:\Users\<user>\.config\pyref\catalog.db`).                                                                                                          |
| Zarr (`beamtime.zarr`) | `<same_dir_as_catalog_parent>/cache/<beamtime_hash>/beamtime.zarr`. Example on macOS: `~/.config/pyref/cache/<beamtime_hash>/beamtime.zarr`. `<beamtime_hash>` is a stable SHA-256 digest of the beamtime root path recorded at ingestion time. The zarr tree is local-only; NAS-backed FITS are used for ingestion and re-ingestion, not for routine image reads after ingest. |


macOS ignores `XDG_CONFIG_HOME` for this default tree so a common misconfiguration (`XDG_CONFIG_HOME=$HOME/Library/Application Support`) cannot relocate pyref into Application Support. Use `PYREF_HOME` (tests) or `PYREF_CATALOG_DB` / `PYREF_CACHE_ROOT` when you need a non-default location.

When `PYREF_HOME` is set (common in tests), both tooling expectations may still point at that directory for the catalog file (`<PYREF_HOME>/catalog.db`) as implemented in the Rust path resolver; production use relies on the defaults above unless overridden.

Optional environment overrides: `PYREF_CATALOG_DB` (absolute path to `catalog.db`) and `PYREF_CACHE_ROOT` (parent of `<beamtime_hash>/beamtime.zarr` directories). Parallel FITS reads during ingest honor `PYREF_INGEST_WORKER_THREADS` or `PYREF_INGEST_RESOURCE_FRACTION` when explicit kwargs or TUI config fields are unset.

Ingestion is a **single pipeline**: catalog metadata (Diesel/SQLite) and zarr array writes happen in one pass. There is no separate user-visible “metadata only” phase followed by a later image materialization step.

The raw FITS files on the NAS are only required during initial ingestion and re-ingestion. After ingestion, reduction and browsing workflows operate from the local catalog and zarr store. If the NAS is unavailable, previously ingested beamtimes remain fully accessible from local storage.

#### Path aliasing for NAS-sourced data

Because NAS mount points differ across machines (e.g., `/Volumes/beamdata` on macOS vs. `/mnt/beamdata` on Linux), paths stored in `beamtimes.path` and `files.path` are recorded as logical URIs of the form `nas://<label>/<relative_path>` rather than absolute filesystem paths. The `label` component is a short user-assigned name for the NAS volume (e.g., `als-data`). Physical mount point resolution is handled by the `path_aliases` table in the catalog, which stores `(label, physical_path)` pairs for the current machine.

On a new machine, the user registers the mount point once:

```
pyref config set-mount als-data /mnt/beamdata
```

This inserts or updates the row in `path_aliases`. All path resolution in the Rust IO layer goes through this table before any filesystem operation. If a label has no registered physical path, the IO layer must return a structured `UnresolvedAlias` error that names the missing label explicitly, rather than a generic file-not-found error. The path aliasing layer is transparent to Diesel queries; aliases are resolved by the IO layer before constructing filesystem paths, never inside SQL.

The zarr cache path stored in `beamtimes` is an absolute local path and is never aliased, since it lives on the local machine by definition. Ingestion records both the NAS logical URI (for FITS provenance) and the resolved local zarr path (for image retrieval) in `beamtimes` as separate columns.

#### Alternative: shared catalog on a network drive or mounted filesystem

For groups where multiple users share a beamtime dataset and want a common catalog, `pyref` supports an explicit catalog path override. The user specifies a path to a directory that is visible to all machines, typically a network share or a FUSE-mounted filesystem:

```
pyref config set-catalog /mnt/shared/pyref/catalog.db
```

When using a shared catalog, the following constraints apply and must be enforced by the IO layer.

SQLite over NFS is unsafe for concurrent writes due to unreliable advisory file locking. If the configured catalog path resolves to a network filesystem (detected by comparing the device ID of the catalog file against known local device IDs, or by an explicit `--network` flag acknowledged by the user at config time), the IO layer must open the connection in WAL mode with a generous busy timeout and must warn the user that concurrent write access from multiple machines is not supported. Concurrent read access is safe in WAL mode. The recommended usage pattern for shared catalogs is that one designated machine performs all ingestion (writes) and all other machines open the catalog read-only.

The zarr cache may also be redirected to a shared location using a separate config key:

```
pyref config set-cache /mnt/shared/pyref/zarr
```

A zarr archive on a fast local network share (e.g., 10GbE NFS or SMB) is acceptable for read-heavy workloads such as browsing and fitting. It is not acceptable as the primary location for zarr writes during ingestion, which should always target local storage first and be copied to the shared location afterwards if shared access is desired.

#### `path_aliases` table

This table lives in the catalog database and is machine-local in semantics, even when the catalog is on a shared drive. It stores one row per registered NAS label for the current machine. The Rust IO layer reads this table on startup and caches the mappings in memory for the duration of the process. Agents must never read `path_aliases` directly from Python; path resolution is an IO-layer concern exposed through the `pyref.io` interface.


| Column          | Type      | Description                                                           |
| --------------- | --------- | --------------------------------------------------------------------- |
| `id`            | `Integer` | Primary key.                                                          |
| `label`         | `Text`    | Short user-assigned NAS label (e.g., `als-data`). Unique per catalog. |
| `physical_path` | `Text`    | Absolute filesystem path to the mount point on this machine.          |
| `registered_at` | `Text`    | ISO 8601 timestamp of last registration.                              |


### Cataloging System

The cataloging system ingests a beamtime directory and populates a Diesel-managed SQLite database that serves as the structural backbone for all downstream reduction and fitting workflows. Its primary purpose is to resolve individual FITS frames back to their originating sample, scan, and beamtime, and to expose this hierarchy as a queryable, lazily accessible interface. The schema is defined in `src/schema.rs` and must be kept in sync with the Diesel migration files under `migrations/`. The most common access pattern is retrieving all frames associated with a given sample and tag combination, grouped by energy or angle, for assembly into a stitched reflectivity profile.

The `profiles` table is the primary user-facing entry point. Users browse profiles, not scans. Scans are administrative provenance; profiles are the scientific deliverable.

#### Header Card Tiering

The 115 FITS primary HDU cards per frame are split into two tiers at ingestion time. Eleven cards that directly drive scan classification, beamspot localization, normalization, and profile identity are promoted to first-class typed columns on the `frames` table: `sample_x`, `sample_y`, `sample_z`, `sample_theta`, `ccd_theta`, `beamline_energy`, `epu_polarization`, `exposure`, `ring_current`, `ai3_izero`, and `beam_current`. All remaining cards are stored in the `frame_header_values` EAV table, keyed through the `header_cards` registry. The `header_cards` table is populated automatically on first ingestion from whatever cards are present in the FITS files; subsequent beamtimes with new or renamed channels append rows to this table without requiring a schema migration. If a card that was previously treated as non-critical needs to be queried as a first-class column, the correct remedy is a Diesel migration that adds the column to `frames` and backfills it from `frame_header_values`, not a workaround join.

#### `beamtimes`

Root of the catalog hierarchy. Stores two path columns: `nas_uri`, which is the logical `nas://<label>/<relative_path>` URI pointing to the original FITS data on the NAS, and `zarr_path`, which is the absolute local filesystem path to the beamtime's zarr archive at `<data_dir>/pyref/.cache/<beamtime_hash>/beamtime.zarr` (same `<data_dir>` convention as the default catalog path). The `nas_uri` is used for provenance and during ingestion and re-ingestion; all post-ingestion image retrieval goes through `zarr_path`. The date parsed from the beamtime directory name is also stored here. All other tables carry a foreign key to this table.

#### `samples`

One row per unique sample name within a beamtime. Stores the sample name and the median `sample_x`, `sample_y`, and `sample_z` stage positions computed across all frames attributed to that name. Stage positions are nominally fixed per sample; frames that deviate beyond a configurable tolerance are flagged `mislabeled_sample` in `frames.quality_flag` rather than creating a second sample row.

#### `tags`

One row per unique tag slug parsed from FITS filenames. Tags carry no intrinsic meaning to the catalog. The many-to-many relationship between tags and files is resolved through the `file_tags` junction table.

#### `file_tags`

Junction table linking `files` to `tags`. Allows many tags to map to many files without duplication.

#### `files`

One row per FITS file ingested. Stores the absolute path, bare filename, scan number, frame number, sample ID, and beamtime ID. This is the canonical provenance reference for raw file locations and the join target for tag resolution. Image data is not retrieved through this table; it is accessed via the zarr keys on `frames`.

#### `scans`

One row per scan. Stores the scan number, scan type (`fixed_energy` or `fixed_angle`), start and end timestamps, sample ID, and beamtime ID. Scan type is determined during ingestion from the motor trajectory analysis and recorded here as a first-class attribute so downstream reduction does not recompute it.

#### `header_cards`

Registry of FITS header card names discovered during ingestion. One row per unique card name. Each row stores the raw card name as it appears in the FITS header, a human-readable display name, and a category label (`motor`, `ai`, `camera`, or `metadata`). This table is the lookup key for the `frame_header_values` EAV table. Agents must not hard-code card name strings outside of this table and the first-class column definitions on `frames`.

#### `frames`

One row per frame per scan. Stores the eleven first-class header card values as typed `Double` columns, plus the zarr retrieval keys (`zarr_group_key` and `zarr_frame_index`) needed to fetch the detector image. The monolithic zarr archive is located at `beamtimes.zarr_path` for the parent beamtime. Within the archive, scan number is the group key, frame number is the dataset index within that group, and each group stores two datasets per frame named `raw` and `processed`. The `raw` dataset is the image as extracted from the FITS file; the `processed` dataset is the image after edge artifact removal, row-wise and column-wise background subtraction, and Gaussian filtering. Both share the same frame index. Each row also carries FKs to `files` and `scans`, providing the full provenance chain from pixel to beamtime.

#### `frame_header_values`

EAV store for all FITS header cards not promoted to first-class columns on `frames`. All values are stored as `Double`. The card name is resolved through `header_cards`. This table is append-only after initial ingestion; values are never updated in place.

#### `profiles`

The primary user-facing table. One row per reduced reflectivity profile, where a profile is a single continuous 1D curve assembled from one or more stitches. Multi-profile scans produce multiple rows sharing the same `scan_id`, distinguished by `profile_index`. Each row stores the profile type (`fixed_energy` or `fixed_angle`), the value of the fixed parameter (energy in eV for fixed-angle profiles, theta in degrees for fixed-energy profiles), `epu_polarization`, and the median stage position (`sample_x`, `sample_y`, `sample_z`) over all member frames. The stage position columns here are denormalized from `frames` for query convenience; the authoritative per-frame positions remain in `frames`.

#### `profile_frames`

Junction table mapping profiles to their constituent frames, with a `frame_role` column classifying each frame's function in the reduction pipeline. Valid roles are `i0`, `stitch`, `overlap`, and `reflectivity`. I0 frames appear here multiple times when they serve as the normalization reference for more than one profile in a multi-profile scan; this is by design and is the correct resolution of the shared-I0 problem. Agents must not attempt to enforce uniqueness on `frame_id` in this table.

#### `beam_finding`

Per-frame output of the beamspot localization pipeline. Stores the preprocessing parameters applied (edge removal flag, dark column and row counts for background subtraction, Gaussian kernel sigma), the peak fitting result (centroid row and column, ROI intensity, fit standard deviation), the dark region statistics (mean and standard deviation), and a `detection_flag` with values `ok`, `beam_detection_failed`, or `beam_drift_anomaly`. Each row carries a FK to `frames`. Frames with `detection_flag = beam_detection_failed` must not appear in `reflectivity`.

#### `stitch_corrections`

Per-stitch correction factors computed during the normalization and stitching pipeline. One row per stitch segment within a profile. Stores the `fano_factor` (always non-null; 1.0 when no Fano correction was applied), the `overlap_scale_factor` (null for the first stitch, which has no preceding stitch), the `i0_normalization_value`, and `i0_source_scan_id` (null when I0 comes from frames within the current profile; set to the external scan's ID for fixed-angle profiles that borrow I0 from a separate scan). Each row carries a FK to `profiles`.

#### `reflectivity`

Frame-level reduced reflectivity data after normalization and stitching. One row per reduced frame. Stores Q (inverse angstroms), theta (degrees), energy (eV), normalized intensity, propagated one-sigma uncertainty, and `frame_type` (`i0`, `stitch`, `overlap`, or `reflectivity`). Carries FKs to `profiles`, `frames`, and `beam_finding`. Frames flagged `beam_detection_failed` must not appear here. Stitched profile assembly and parquet export are performed by the packaging utility in `pyref.reduction`, which joins this table against `stitch_corrections` filtered by `profile_id`.

## Data Quality Flagging

Several conditions arising during cataloging and reduction require flagging rather than silent failure or hard error. The following flags are first-class catalog attributes stored in typed text columns, not log messages, and must be queryable from the DataFrame interface. Flag values use lowercase snake_case to match the string literals defined in `schema.rs`.

Frames attributed to a sample name whose inferred stage position deviates from the representative position for that sample by more than a configurable threshold are flagged `mislabeled_sample` in `frames.quality_flag`. The threshold is configurable per beamtime but should default to a value that tolerates sub-millimeter drift while rejecting stage position changes consistent with a deliberate sample move.

Frames for which the beamspot localization algorithm fails to identify a credible peak are flagged `beam_detection_failed` in `beam_finding.detection_flag`. These frames must not appear in `reflectivity` and must not be silently zeroed or filled.

Frames whose fitted beamspot centroid deviates from the linear trend model across the scan by more than a configurable multiple of the fit residual standard deviation are flagged `beam_drift_anomaly` in `beam_finding.detection_flag`. These frames may still appear in `reflectivity` but are surfaced to the user for manual inspection before the packaging utility includes them in a parquet export.

Files that fail the filename parser are flagged `parse_failure` in `files.parse_flag`. Files whose directory layout does not match either supported pattern are flagged at the beamtime level via a structured error returned to the caller rather than a catalog row.

## Profile Packaging Utility

The packaging utility assembles a stitched reflectivity profile from the `reflectivity` and `stitch_corrections` tables and exports it as a flat parquet file. It is a standalone tool in `pyref.reduction` that operates on catalog query results and is not part of the catalog itself. The primary input is a `profile_id`; the utility joins `reflectivity` against `stitch_corrections` filtered by that ID to assemble the full stitched curve. The output parquet schema is fixed and contains one row per reduced frame with the following columns at minimum: `q` (inverse angstroms), `theta` (degrees), `energy` (eV), `intensity` (normalized, dimensionless), `uncertainty` (one-sigma), `frame_type`, `scan_number`, `sample_name`, and `overlap_scale_factor`. The utility must reject frames with `beam_finding.detection_flag = beam_detection_failed` and must warn the user before including frames with `beam_drift_anomaly`. Agents must not add or remove columns from the output schema without updating this specification.

## Fitting Module Architecture

The Fitting module wraps `refnx` to provide a domain-specific interface for PRSoXR model construction and optimization. The 4x4 transfer matrix method implemented in `refnx` handles anisotropic dielectric tensors and resonant scattering contrast, which are the physically relevant cases for this beamline. Agents working in `pyref.fitting` are expected to understand the transfer matrix formalism and the relationship between the model layer stack (thickness, roughness, optical constants) and the computed reflectivity curve.

The module boundary is as follows. Model construction (layer stack definition, optical constant assignment, parameter bounds, and inter-parameter constraints) lives in `pyref.fitting`. Numerical optimization and MCMC sampling are delegated entirely to `refnx` and must not be reimplemented. Objective function construction follows `refnx` conventions. Fitting output is returned as a structured result object containing the optimized parameters, their uncertainties, the fitted reflectivity curve, and the residuals, and must be serializable to parquet or HDF5 for archival.

## Numerical Precision and Scientific Correctness

This library processes experimental data where uncertainty quantification is not decorative but determines the validity of downstream physical conclusions. The following constraints are non-negotiable.

Uncertainty propagation must be exact to first order at every reduction step. Any operation that discards, truncates, or implicitly zeros an uncertainty is a correctness bug. Weighted means must use inverse-variance weights throughout; unweighted means are not acceptable substitutes in any reduction context.

Silent dtype coercion is prohibited. Operations that would coerce float64 to float32, or integer counts to floating point without explicit intent, must be made explicit or prevented. The catalog and parquet outputs must preserve float64 precision for all physical quantities.

Frame provenance must be preserved end-to-end. Every row in `reflectivity` must be traceable back to the originating FITS file and zarr frame index through the FK chain `reflectivity -> frames -> files`. Any reduction step that aggregates without recording the constituent frame IDs breaks this chain and is unacceptable.

Fano factor computation and application must be documented in `stitch_corrections` for every scan. A scan processed without a Fano correction must record `fano_factor = 1.0` rather than leaving the field null.



# General

## General Structure

This codebase is maintained by contributors with physics PhDs and extensive backgrounds in scientific and engineering software, including numerical computing, data analysis, instrumentation, simulation, and research-grade reproducibility. Maintainers are highly mathematically literate, comfortable with linear algebra and statistics, and expect rigorous numerics with explicit type handling—silent coercion and imprecise computations are not acceptable.

## Operating principles

- Prefer the smallest coherent change set that satisfies the stated specification. Avoid drive-by refactors, unrelated formatting sweeps, and scope expansion.
- Treat the repository's existing patterns as the default contract. Match naming, module boundaries, error-handling style, and test layout unless the user explicitly requests a migration.
- Default to production-grade output: complete, runnable, and reviewable. Do not ship placeholder text such as ellipses, "the rest of the implementation here", "TODO: implement", or "fill in later" inside code or patches unless the user explicitly authorizes a stub.
- Remain non-lazy: if a command fails, diagnose, adjust, and retry with a different approach when reasonable. Do not stop after the first error without analysis.
- Do not use emoji in code, comments, documentation strings, commit messages, or user-facing text unless the user explicitly requests emoji.

## Communication and documentation outside code

- Avoid standalone documentation files or long narrative write-ups unless the user asks for them or the repository already uses them for the same purpose.
- Prefer editing the code and tests that enforce behavior over adding parallel prose that can drift out of date.
- When the user asks for explanation, keep it precise and tied to the change set.

## Public API documentation (language-agnostic)

- Every **public** function, method, or type exported from a library module carries documentation appropriate to the language (for example Python docstrings, Rust `///` on public items, TSDoc/JSDoc on exported symbols).
- Documentation states the **surface**: name, purpose, parameters, return value, and thrown or returned error shapes when that is part of the contract.
- Prefer **prescriptive** voice that states what the symbol **does** and **means** for callers. Prefer "Maximum `foo` grouped by `bar` using a stable sort on `bar`." over "Returns the max of foo by bar." or "Computes the maximum `foo` grouped by `bar` using a stable sort on `bar`." or "Returns the maximum `foo` grouped by `bar` using a stable sort on `bar`."
- For each parameter: name, type as used in the project, allowed ranges or invariants when non-obvious, and interaction with other parameters.
- For results: type, semantics, units when relevant, ordering guarantees, and stability promises when they matter for science or reproducibility.
- Describe **what** the function does at the abstraction level of the API, **how** only when algorithmic choices affect correctness, performance contracts, or numerical stability, and **why** that approach is chosen when trade-offs are non-obvious (for example streaming vs materializing, online vs batch statistics).
- **Internal** helpers omit the long-form contract unless complexity warrants a short note. **Private** helpers keep documentation minimal (a phrase or single sentence at most).

## Module and package documentation

- Each library module (or the closest equivalent in the language's module system) includes a short module-level description of responsibility: what problems it solves, what it explicitly does not handle, and which invariants callers should respect.
- Module docs are prescriptive about intent and boundaries so new contributors and agents do not duplicate concerns across modules.

## Tooling, skills, and continued learning

- Use project rules, agent skills, and MCP documentation tools when they apply to the task. Prefer authoritative library and framework documentation over memory when behavior, defaults, or breaking changes matter.
- When touching unfamiliar APIs, verify signatures, deprecations, and error modes against current docs or source in the dependency before guessing.
- Prefer file-scoped or package-scoped commands when the repository documents them (typecheck, lint, format, test on a single path) to shorten feedback loops.
- State permission-sensitive actions clearly (dependency installs, destructive commands, credential access) and follow the user's safety expectations for the workspace.

## Task shape (goal, context, constraints, completion)

- Restate the goal in terms of observable outcomes: behavior, tests, or interfaces that change.
- Ground work in the relevant files, modules, and existing tests named by the user or discovered through search.
- Honor explicit constraints (performance, numerics, compatibility, style) before proposing alternatives.
- Stop when the completion criteria are met: tests pass where applicable, edge cases called out by the user are handled, and no placeholder implementation remains.

## Quality bar for agent output

- Do not substitute templates, pseudocode, or abbreviated implementations when the user asked for working code.
- If scope is too large for one pass, propose a staged plan and complete the first stage fully rather than leaving partial files full of omissions.

# Python

## Python

The following applies to **Python** work in this repository: scientific and general-purpose code, with emphasis on clear structure, reproducible tooling, and documentation that matches how the team uses Cursor (skills, subagents, and editor rules).

### Conventions

- Follow **PEP 8** surface style; treat **Ruff** configuration in `pyproject.toml` as the enforced interpretation of those conventions.
- Prefer **Python 3.12+** unless the repo pins an older interpreter.
- Prefer **readability** over micro-optimizations; vectorize numerics when a library primitive exists instead of tight Python loops over large data.
- **NumPy-style docstrings** on **public** APIs (parameters, returns, and examples where they clarify behavior). Keep implementation bodies clear **without** long narrative **inline comments**—use names, small functions, and docstrings instead.
- **Tables and time series**: explicit **index/column** semantics when using **pandas**; **lazy** queries when standardizing on **Polars** for heavy pipelines.
- **Lab / instruments**: separate **resource lifecycle** (open, configure, close) from **command strings** and parsing (e.g. **pyvisa** patterns).

### Tooling

Use **[uv](https://docs.astral.sh/uv/latest/)** for environments, runs, and dependency changes. Pair it with the **[Astral](https://astral.sh/)** stack as configured in this project.

- **Dependencies**: add, upgrade, and remove packages with `**uv add`**, `**uv add … --upgrade**`, `**uv remove**`—do **not** hand-edit version pins in `pyproject.toml`.
- **Environment**: `**uv sync`** after cloning or when the lockfile changes; `**uv run …**` to execute Python, tools, and tests.
- **Dev tools**: keep `**ruff`** and `**ty**` in the development (or project) dependency group; run `**ruff check**` (and project formatting if applicable) plus `**ty check**` on changed code. `**uvx**` remains an option for one-off tool runs.
- **pytest**: install via `**uv add --dev pytest`** (or the project's dev group); run with `**uv run pytest**`.

If a `**uv**` subcommand differs by version, use `**uv --help**` or the [uv docs](https://docs.astral.sh/uv/latest/).

### Testing

- Prefer **fast, deterministic** unit tests; isolate I/O and timing-sensitive checks when the team uses markers or separate jobs.
- **Regression tests** for fixed bugs; for numerics, assert **shapes**, **dtypes**, and stability expectations when science or reproducibility requires it.

### Cursor: skills

Load these **skills** by **name** when the task matches (each skill's own `SKILL.md` and references hold the full detail). Installed skills usually live under `.cursor/skills/` (or your editor's equivalent).


| Skill                     | Use it for                                                                                                                                                                                        |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **general-python**        | Hub: **uv** / **ruff** / **ty** workflow, builtins and collections, functions and classes, **dataclasses**, typing boundaries, **pytest**, scientific defaults, and pointers to the other skills. |
| **numpy-scientific**      | **NumPy**: dtypes, views vs copies, broadcasting, ufuncs and reductions, **linalg** / **einsum**, `**Generator`**, I/O, interop with tables and plotting.                                         |
| **dataframes**            | **pandas** and **Polars**: when to use which, indexing, joins, lazy execution, I/O, nulls, Arrow interop.                                                                                         |
| **numpy-docstrings**      | **Numpydoc**-style docstrings: section order, semantics (what belongs in docstrings vs types vs tests), anti-patterns, **Parameters** / **Returns** / **Examples** / classes / modules.           |
| **matplotlib-scientific** | Publication-style **Matplotlib**: OO API, axes and legends, layout, export, journal widths, optional **SciencePlots**.                                                                            |
| **lab-instrumentation**   | **PyVISA** / VISA sessions, **sockets** vs VISA, **hardware abstraction**, **input validation** before I/O, **testing** without hardware, **PDF** extraction for datasheets and manuals.          |


### Cursor: subagents

Delegate by **subagent name** when a focused pass is better than inline editing. Subagents usually live under `.cursor/agents/` (or your editor's equivalent).


| Subagent            | Use it for                                                                                                                        |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| **python-reviewer** | Reviewing changes: **uv** hygiene, typing, numerics footguns, tests, docstring quality.                                           |
| **python-types**    | Deep **typing** for **ty**: annotations, PEP 695-style generics, exhaustive `**match`**, fixing checker output.                   |
| **python-refactor** | **Structure**: unclear multi-value returns, composition vs inheritance, oversized functions or classes, deterministic boundaries. |


### Cursor: rules

- A **Python** Cursor **rule** applies to Python sources (typically `**/*.py` when the rule is configured for those globs). It restates **interpreter preference**, **uv** usage, **ruff** / **ty** expectations, numerics and docstring defaults, and points to **general-python**, domain skills such as **lab-instrumentation** when editing drivers or lab I/O, and the subagents above.
- **Rule text is authoritative for "always on" editor hints**; **skills** carry the long-form patterns and examples. When the two differ on a detail, follow **this spec** and `**pyproject.toml`**, then the **rule**, then skill nuance.

### External references

- [numpydoc format](https://numpydoc.readthedocs.io/en/latest/format.html)
- [uv](https://docs.astral.sh/uv/latest/), [Ruff](https://docs.astral.sh/ruff/), [ty](https://docs.astral.sh/ty/)

# Rust

This workspace uses the **dotagent Rust** stack.

- Prefer `cargo` as the interface for builds, tests, and dependency changes; add crates with `cargo add`.
- Match edition and MSRV documented in the repository; do not bump them casually.
- Prefer `clippy` clean builds when the project already enforces clippy in CI.
- Error handling should be structured (`Result`) at boundaries; avoid `unwrap` in library code unless explicitly documented.

# Python - Jupyter

This workspace extends Python with **Jupyter notebook** expectations.

### General Guidelines

This project will make strong use of jupyter notebooks to solve a number of problems, primarily with a scientific focus, but may also have a general purpose focus. Notebooks should be written and well documented. But keep in mind that a notebook allows for a lot of flexibility, and as such, the code should be written in a way that is easy to understand and maintain.

!!!!Make sure that you use your new jupyter tools instead of generating as a json!!!!

 As a general rule, we will use notebooks for one of the following purposes:

#### Lightweight Exploration Notebooks

Here we will have a few cells that will build the idea or concept, test it with some data, and then present the results in a clean and easy way. Use matplotlib for static plotting, or hvplot/altair/plotly for interactive plotting.

It is important to note that these notebooks are not really designed to be robust, and as such we should not focus too hard on making them so. Write the minimal code to get the job done, and then move on to the next notebook.

#### Prototyping Notebooks

This is where we will prototype a robust coding solution to a problem. We will use this to test and validate each chunk of code in the complex workflow. As such it is important that we use many small cells to test and validate each chunk of code. Eventually, we will want to move this code into a final script/library.

It is important to note that these notebooks, are not really designed to showcase the code, but rather to test and validate each chunk of code in the complex workflow. As such, we should not focus too hard on making them look nice, but rather ensure that we atomize the code into small cells that can be tested and validated individually. Testing and validation might be done using simple displays, or plotting, or other cases. But assertstetements are not necessary, and should be avoided if possible in favor of displaying the results to the user.

#### Demonstration Notebooks

These notebooks are designed to showcase a workflow of production ready code. Ideally, after a library is complete and ready to use, the user will be able to import the library and use the code treating the notebook as a production environment. They will have a minimal ammount of cells, mixing in documentation and examples of how to use the library.

It is important to note that these notebooks are designed to be robust. These should mix in a healthy ammount of markdown documentation and explaination. but not be too heavy handed. Keep in mind that the goal of these notebooks is that a user can copy them into their own notebooks and know how to use the library.

### Use of Cells

- Use markdown cells sparingly to explain the code, or why it works the way it does. But avoid long narrative markdown cells that are overly robust.
- Use code cells to write the code. Ensure that each cell is small and atomic. Each cell should have one responsibility and deterministically produce a result. If you define a function, call it in the same cell. Avoid using global variables, or mutable state if possible.
- Class definitions should idealy be avoided in notebooks, unless we are prototyping a library. If we need a class, it should be defined in a separate cell and have a minimal footprint.
- Variables should be defined in the cells that use them, and should usually be displayed to the user. Avoid using global variables, or mutable state if possible.
- When prototping a library, use the `%autoreload 2` magic to ensure that the code is reloaded when it is changed.

# Python - PyO3

This workspace extends Python with **PyO3 / Maturin** integration expectations.

### General Guidelines

We are using the [pyo3](https://pyo3.rs/) library to create a python extension for our project. The ONLY reason we are doing this is to speed up the execution of critical code. As such, we should only use pyo3 for code that is critical to the performance of the project, and should avoid using it for code that is not performance critical. Generally, iO operations are easier to implement in python, while multi-threaded operations are faster in rust.

The python GIL is the global interpreter lock, limiting the ability for python to be truly multi-threaded. However, rust executions can bypass the GIL leading to a significant speedup in performance. As such, we should not multi-thread in python, but rather pass tasks into rust for parallel processing, and then return the results back to python.

### Tooling

`uv` is still king within the project, and should be used for all python needs. However, we will use the `maturin` tool as the build system for the project. See the [maturin documentation](https://www.maturin.rs/) for more information on how to use it. Ensure that `maturin` is installed in the dev group. See the [uv documentation](https://docs.astral.sh/uv/concepts/projects/init/#projects-with-extension-modules) for more information on how to use maturin as a tool. In general, we prefer a structure native to maturin following the following format:

```
.
├── pyproject.toml
├── Cargo.toml
├── python/
│   ├── <module_name>/
│   │   ├── __init__.py
│   │   └── ...
├── src/
│   ├── lib.rs
│   └── bindings.rs
```

Ensure that the `python/` directory is a valid python package, and that the `src/` directory is a valid rust crate.

# Rust - TUI

This workspace extends Rust with **TUI** application expectations.

- Prefer established TUI crates already present in the dependency graph; do not introduce a parallel framework without a migration plan.
- Treat terminal sizing, resize bursts, and partial redraws as normal inputs; avoid assuming a static viewport.
- Separate rendering from application state: model updates should not directly depend on widget internals.
- Keyboard and mouse interactions should degrade gracefully when a capability is unavailable.

# Rust - PyO3

This workspace extends Rust with **PyO3 / Maturin** extension expectations.

- Design the Rust crate as an extension first: minimal Python surface, maximal safety around lifetimes and exceptions.
- Use Maturin's layout and metadata conventions; keep `pyproject.toml` and `Cargo.toml` agreeing on names and versions.
- Document unsafe blocks with the project's standard (even if you avoid new unsafe code).
- Prefer thin Python modules that re-export a small Rust API rather than exposing many low-level Rust objects.



## Learned User Preferences

- Prefer **Rich** over **tqdm** for Python-facing ingest progress (nested or multi-segment bars, transient display when finished).
- When the user supplies an authoritative **staged file list** for a commit, stage or commit **only** those paths and do not broaden `git add` unless they explicitly change the instruction.
- For pull requests, default to the **parent integration branch** named in the thread rather than opening against `main` when the user specifies a non-main target.
- For notebook-first catalog workflows, keep setup and queries in **Jupyter** with minimal required steps outside the notebook when that is the stated goal.
- Keep **macOS Finder artifacts** such as `****/.DS_Store`** out of git via `.gitignore` rather than tracking or committing them.
- For multi-task implementation plans, default to **subagent-driven-development** with a per-task two-stage review (spec compliance first, then code quality) instead of executing everything inline.
- Benchmark and profile ingest against a **local or synthetic replica** first; only run profilers against the live NAS beamtime once the local case is well-characterized (live NAS runs can exhaust resources and crash the editor).

## Learned Workspace Facts

- **Implementation plans** should always include two closing steps: (1) **greenfield cleanup** — remove unused code and tests that are no longer needed but were tied to the change; (2) **quality gates** — intentional deprecations where relevant, typing and lint fixes, and verification that the code **builds with zero errors and zero warnings** (Rust + Python per repo tooling). Apply these at the end of every substantive plan, not only as optional polish.
- Python ingest progress integrates `**beamtime_ingest_layout`** (total FITS count and per-scan file counts) with `**ingest_beamtime(..., progress_callback=...)**` emitting event dicts of kind `layout`, `phase`, `file_complete`, or `catalog_row`; pair this with Rich or tqdm-style handlers.
- Keep `**.cursor/hooks/state/**` out of git: add it to `**.gitignore**` so hook state and the continual-learning index stay local.
- Ingestion and zarr writes from **network-mounted beamtime roots** can be far slower than from a **local replica**; validate progress UX against a local tree when iterating.
- **Rust + PyO3:** Use a default feature set (for example `**bindings`**) that links `**libpython**` for `**cargo test**`; Maturin wheel builds use a separate `**extension-module**` feature that enables `**pyo3/extension-module**`. Putting `**extension-module**` in the default test feature set can produce undefined Python symbols (for example `_Py_DecRef`, `Py_IsInitialized`) on Linux CI linkers.
- `**read_beamtime(..., ingest=True)**` runs ingest against the **default global catalog path** from the Rust layer; an explicit `**catalog_path`** mainly selects which database is **read** for the returned view. Beamtime lookup keys must match absolute URI form (for example `file:///Volumes/...`), and offline lookup must avoid strict canonicalization so unmounted NAS paths can still match indexed beamtimes.
- **Ruff** may exclude `**python/pyref/beamline`**, `**notebooks**`, and `**tests**` per `pyproject.toml`; treat those paths as out of scope for Ruff unless configuration changes.
- Ingest phases are modeled by the Rust `**IngestPhase` enum** (not string labels); the catalog phase **coalesces short scans into a single SQLite transaction** (small-scan batching), which is a deliberate design choice.
- CI-safe ingest benchmarking uses the synthetic harness: the Rust helper at `**tests/common/mod.rs`** (consumed by `tests/synthetic_harness.rs` and `tests/ingest_streaming.rs`) plus `**uv run pyref bench synthetic**`; shared progress/table helpers live in `**python/pyref/ingest_profile.py**` and are also used by `**uv run pyref bench profile --beamtime <path>**` for real beamtimes.
- Rust integration tests for ingest require `**cargo test --features catalog,parallel_ingest**`; tests that mutate env vars (`**PYREF_CATALOG_DB**`, `**PYREF_CACHE_ROOT**`) must serialize with a `Mutex` guard (pattern in `src/io/raw_pixels.rs` tests) and must point those vars at isolated tempdirs so they never write to the default catalog.
- `**tests/fixtures/minimal.fits**` is the canonical 2x2 BITPIX=16 FITS reference; new synthetic fixtures must match its header/block layout (2880-byte header, BZERO=32768 for unsigned-as-signed-i16, stems of the form `<sample>-<scan>-<frame>.fits`).
- **Typer CLI** (`pyref` entry point): implementation under `**python/pyref/cli/`** with groups `**nas**` (single registered NAS root in `**config.toml**` next to the data dir), `**beamtime**` (list/describe coverage using `**py_beamtime_ingest_layout**` + `**py_catalog_file_count**`), `**catalog**` (`path`, `ingest` with `**--max-scans**` / `**--scans**`), `**watch**` (subprocess daemon via `**PYREF_CLI_WATCH_SPEC**`, worker module `**python -m pyref.cli.daemon**`, PID/logs under `**<pyref_data_dir>/daemons/**`), and `**bench**` (`synthetic`, `profile` for ingest timing). Legacy `**pyref-ingest**` delegates to `**pyref catalog ingest**`.
- Beamtime **Zarr** raw images are **per-scan** 3D `uint16` stacks at `/images/scans/<scan>/raw` under the beamtime archive (default cache root `<pyref_config_dir>/cache/<sha256>/beamtime.zarr`; shuffle + Zstd; see `catalog::zarr_write`, schema, `**migrate-zarr-3d`**); standalone Rust migration or tooling that initializes **Polars** through **PyO3** may print `**failed to get allocator capsule`** on stderr even when the run succeeds.
- **Rust bindings for CLI:** `**py_pyref_data_dir`**, `**py_catalog_file_count**`, optional ingest subset via `**IngestSelection**` (`max_scans`, `scan_numbers`), and `**CatalogWatcherCancel**` + `**py_run_catalog_watcher_blocking**` when the `**watch**` feature is enabled (included in default features for extension builds).
- For the catalog-first PRSoXR prototyping effort, start at `**docs/prototyping/README.md**` and treat `**docs/prototyping/comprehensive_implementation_plan.md**` as the execution source of truth; keep `**docs/prototyping/api-design/api_flow_design.md**` and `**docs/prototyping/workflows/cell_*.md**` aligned with it before implementation changes.


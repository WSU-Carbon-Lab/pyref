# Cell 01: Catalog Connection and Beamtime Selection

## Goal

Start from the catalog, list available beamtimes, inspect scan/sample views, apply metadata corrections, sync updates, and fetch scoped frame rows.

## Why This Cell Exists

This cell establishes the authoritative working context for all downstream processing. It must resolve one beamtime key to one catalog-backed dataset and expose fast review/edit tables before any heavy frame/image operations.

## Canonical Usage

```python
import pyref as pr

cat = pr.catalog()
cat.beamtimes()

bt = cat.beamtime("2026Feb")
scan_table = bt.scans_view()
sample_table = bt.samples_view()
frames = bt.frames(sample_name="sample_a", scan_numbers=[1201, 1202], engine="polars")
```

## Implementation Context

### Catalog entities this cell depends on

- `beamtimes`: key, aliases, canonical source identity, zarr location.
- `scans`: scan-level identity, scan classification, summary ranges.
- `samples`: beamtime-local sample names and representative stage metadata.
- `files` and `tags`/`file_tags`: naming/tag provenance and user corrections.
- `frames`: row-level index used by all following cells.

### Required API responsibilities

- `CatalogSession.beamtimes()` returns a compact index table for quick inspection.
- `BeamtimeHandle.scans_view()` returns an editable scan summary view.
- `BeamtimeHandle.samples_view()` returns an editable sample-name verification view.
- `BeamtimeHandle.frames(...)` returns filtered frame rows with normalized headers and image locator fields.
- `engine` selection returns either polars or pandas without changing semantic column content.

### DataFrame contract at end of this cell

`frames` must include, at minimum:

- beamtime/scan/frame provenance keys
- normalized header columns used by later cells (`sam_theta`, `beamline_energy`, `exposure`, `beam_current`, `ai3_izero`, `ccd_theta`)
- image locator fields needed by the image accessor

## Required Behaviors

- no FITS paths required
- beamtime key resolution with explicit ambiguity errors
- dedicated editable scans view
- dedicated samples verification view
- backend can be changed before loading:

```python
cat.use_backend("shared-lab-catalog")
```

## Metadata Correction in This Step

Corrections should happen before downstream processing:

```python
bt.rename_scan_sample(1201, "sample_a")
bt.retag_scan(1201, "annealed")
bt.set_scan_profile_type(1201, "fixed_energy")
bt.commit_metadata_updates("fix mislabeled scans")
bt.reingest_metadata(sync=True)
```

Bulk correction pattern:

```python
bt.apply_scan_edits(scan_table)
bt.apply_sample_edits(sample_table)
bt.commit_metadata_updates()
bt.reingest_metadata(sync=True)
```

## Write-Back and Sync Semantics

- scan/sample edits are staged in correction tables, not destructive rewrites of raw provenance.
- `commit_metadata_updates()` persists a transactionally consistent correction set.
- `reingest_metadata(sync=True)` refreshes derived scan/profile surfaces so downstream calls read corrected values by default.
- all corrections remain traceable to underlying file/scan provenance for audit and rollback.

## Notebook Execution Contract

- this cell should rerun quickly because it reads scan/sample summary tables before frame-heavy paths.
- rerunning this cell after watcher updates should surface new scans while preserving existing corrections.

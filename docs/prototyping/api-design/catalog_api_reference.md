# Catalog API Reference

## Scope

Defines user-facing catalog APIs for selecting beamtimes, correcting metadata, and querying scan/profile/frame tables without direct FITS path usage.

## Session and Backend APIs

- `pr.catalog(backend: str | None = None) -> CatalogSession`
- `CatalogSession.register_backend(name: str, path: str, make_default: bool = False)`
- `CatalogSession.backends() -> DataFrame`
- `CatalogSession.use_backend(name: str) -> CatalogSession`

Behavior:

- one default backend is active at a time
- backend registration is persisted in local config
- switching backend does not mutate catalog contents

## Beamtime Discovery and Resolution

- `CatalogSession.beamtimes() -> DataFrame`
- `CatalogSession.beamtime(key: str) -> BeamtimeHandle`
- `CatalogSession.resolve_beamtime(key: str) -> dict`

Resolution order:

1. exact key match
2. alias match
3. deterministic partial key match

Ambiguous match behavior:

- raise `AmbiguousBeamtimeKeyError`
- include candidate keys and short metadata summary

## Beamtime Query APIs

- `BeamtimeHandle.scans(...) -> DataFrame`
- `BeamtimeHandle.scans_view(...) -> DataFrame`
- `BeamtimeHandle.profiles(...) -> DataFrame`
- `BeamtimeHandle.frames(...) -> DataFrame`
- `BeamtimeHandle.frames_lazy(...) -> LazyFrame`
- `BeamtimeHandle.samples_view(...) -> DataFrame`
- `BeamtimeHandle.summary() -> DataFrame`

Selector fields:

- `sample_name`
- `tag`
- `scan_numbers`
- `profile_type`
- `energy_min`, `energy_max`
- `theta_min`, `theta_max`

## Metadata Correction APIs

- `BeamtimeHandle.rename_scan_sample(scan_number: int, sample_name: str)`
- `BeamtimeHandle.retag_scan(scan_number: int, tag: str | None)`
- `BeamtimeHandle.set_scan_profile_type(scan_number: int, profile_type: str)`
- `BeamtimeHandle.apply_scan_edits(scans_df: DataFrame)`
- `BeamtimeHandle.apply_sample_edits(samples_df: DataFrame)`
- `BeamtimeHandle.commit_metadata_updates(message: str | None = None)`
- `BeamtimeHandle.reingest_metadata(sync: bool = True)`
- `BeamtimeHandle.audit_overrides() -> DataFrame`

Persistence contract:

- overrides are stored as correction rows in catalog
- raw parsed labels remain queryable for provenance
- corrected labels are default on future loads

## Inspection Views Contract

### `scans_view`

Purpose:

- fast per-scan summary view for review and editing

Required columns:

- `scan_number`
- `sample_name`
- `tag`
- `profile_type`
- `n_frames`
- `energy_min`
- `energy_max`
- `theta_min`
- `theta_max`
- `last_modified`

### `samples_view`

Purpose:

- verify sample naming consistency across scans

Required columns:

- `sample_name`
- `n_scans`
- `n_frames`
- `scan_numbers`
- `tags`

Behavior:

- edits from either view can be applied in bulk
- `commit_metadata_updates` persists changes
- `reingest_metadata(sync=True)` updates derived catalog surfaces for future load calls

## Recommended Minimal Notebook Pattern

```python
import pyref as pr

cat = pr.catalog()
bt = cat.beamtime("2026Feb")
scans = bt.scans_view()
samples = bt.samples_view()
bt.rename_scan_sample(1201, "sample_a")
bt.retag_scan(1201, "annealed")
bt.set_scan_profile_type(1201, "fixed_energy")
bt.commit_metadata_updates("manual correction")
bt.reingest_metadata(sync=True)
frames = bt.frames(scan_numbers=[1201])
```

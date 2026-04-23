# Cell 05: Persist Mask and Beamspot Overrides

## Goal

Persist user-selected mask and corrected beamspot points so future load calls replay the same setup.

## Why This Cell Exists

Interactive corrections have no scientific value unless they are reproducible. This cell converts UI edits into catalog state that can be reapplied by future sessions, machines, and processing runs.

## Canonical Usage

```python
beam.refl.save_mask(mask_name="scan_default")
beam.refl.save_beamspot_overrides()
```

## Implementation Context

### Persistence granularity

- mask state must be storable at scan and/or profile scope.
- beamspot overrides are per-frame records keyed by frame ID.
- both mask and beamspot records carry run metadata (author, time, model context).

## Mask Storage Contract

- store `mask_mode` (`include_region` or `exclude_region`)
- store good-region slice indices
- scope by beamtime, scan, and optionally profile
- version masks to avoid destructive edits
- retain interactive tool state needed to reopen and continue editing in notebooks

### Recommended mask payload fields

- `mask_mode`
- `good_region_rows` and `good_region_cols` slice bounds
- optional serialized polygon/selection payload for richer UI replay
- version and parent-version identifiers

## Beamspot Override Contract

- store per-frame row/col override values
- preserve original auto-detected values for provenance
- track author and timestamp metadata

## Catalog touchpoints

- writes to mask persistence table (or equivalent run-state table).
- writes override-linked records for `beam_finding`.
- keeps immutable original detection values and mutable override layer separate.

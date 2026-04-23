# Cell 03: Frame Classification, Stitch Summary, and i0 Linking

## Goal

Classify points (`i0`, `stitch`, `overlap`, `reflectivity`), summarize stitch structure, and explicitly control external i0 linking.

## Why This Cell Exists

All downstream uncertainty and stitching calculations depend on correct role assignment. This cell is the semantic gate that converts profile membership into operational frame roles and i0 relationships.

## Canonical Usage

```python
valid_i0 = bt.scans().refl.valid_i0_scans()

classified = profiles.refl.classify_points(
    i0_rules={"sam_theta_zero_tol": 0.005},
    merge_scans="auto",      # or merge_scans=[1180, 1181]
    i0_scan_numbers=[1180],  # optional explicit i0 source
)
```

## Implementation Context

### Role classification model

Each frame in each profile is assigned one operational role:

- `i0`: direct-beam normalization frame
- `stitch`: start of a new stitch segment (independent variable reversal boundary)
- `overlap`: frame inside overlap region between adjacent stitches
- `reflectivity`: non-overlap measurement frame

Classification rules are domain-aware:

- fixed-energy profiles detect i0 by `sam_theta ~ 0`.
- fixed-angle profiles can consume external i0 from linked scan sets.
- stitch detection uses independent-variable reversal and stabilization windows.

### External i0 and merge semantics

- `valid_i0_scans()` reports candidate scans that can serve as i0 sources.
- `merge_scans="auto"` enables deterministic compatibility matching across scans.
- `merge_scans=[...]` and `i0_scan_numbers=[...]` provide explicit operator control.
- ambiguous legacy flag `allow_external_i0` is not used.

## Stitch Summary Header Selection

Use auto header family selection:

- sample-related headers
- ccd-related headers
- jj slit related headers

```python
classified.refl.stitch_summary(
    header_selector="auto_sample_ccd_jj",
    header_config="header_summary_rules.json",
)
```

### Dynamic stitch summary behavior

`stitch_summary` auto-selects comparison headers by family:

- sample-family headers
- ccd/camera-family headers
- jj slit-family headers

It excludes nuisance fields (time/file counters) and emits:

- stitch count
- overlap count per stitch boundary
- per-boundary header deltas
- short narrative summary of what changed between stitch segments

## External i0 Policy

- deprecated: ambiguous `allow_external_i0`
- preferred:
  - `merge_scans="auto"` for automatic compatible merge
  - `merge_scans=[...]` for explicit merge set
  - `i0_scan_numbers=[...]` for explicit i0 source scans

## Catalog touchpoints

- reads from `profiles`, `profile_frames`, `frames`, `scans`.
- writes role labels to `profile_frames.frame_role`.
- writes i0 linkage metadata to `stitch_corrections` or a linked run-metadata surface.

# Cell 02: Profile Split and Header-Driven Structure Inspection

## Goal

Split frame rows into profile groups and inspect scan structure using normalized headers.

## Why This Cell Exists

Scan-level metadata is too coarse for reflectivity reduction. This cell decomposes selected frames into profile units that represent physically coherent sweeps and creates first-pass diagnostic plots on normalized headers.

## Canonical Usage

```python
profiles = frames.refl.split_profiles(domain_hint="auto")
profiles.refl.plot_headers(
    x="sam_theta",
    y=["beamline_energy", "ai3_izero", "beam_current", "exposure"],
    by="profile_id",
)
```

## Implementation Context

### Input assumptions from Cell 01

- frame rows are already scoped to a beamtime and optional scan/sample filters
- header columns are normalized to snake_case (`sam_*`, `beamline_energy`, `ccd_theta`, `exposure`, `ai3_izero`)
- scan-level mislabels were corrected or explicitly accepted

### Required split behavior

- profile decomposition respects fixed-energy vs fixed-angle semantics.
- each output row retains frame provenance, while adding:
  - `profile_id`
  - `profile_type`
  - `domain_name`
  - `domain_value`
- profile boundaries are deterministic for identical inputs.

### Catalog touchpoints

- reads from `frames` and `scans`.
- writes or updates `profiles` and profile membership mapping to `profile_frames` when persistence is enabled.

## Header Contract

- header columns must follow normalized snake_case names
- `sample` fields use `sam_*` abbreviation

Examples:

- `sam_theta`
- `sam_x`
- `beamline_energy`

## Expected Output

`profiles` includes:

- `profile_id`
- `profile_type`
- `domain_value`
- frame provenance keys

## Diagnostic intent

- `plot_headers` is not cosmetic; it is a structural sanity check for:
  - domain continuity
  - fixed-parameter stability
  - obvious scan segmentation mistakes
- this step should make profile partition errors visible before expensive image-level processing.

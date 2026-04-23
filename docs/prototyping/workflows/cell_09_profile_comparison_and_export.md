# Cell 09: Profile Comparison and Export

## Goal

Compare profiles with an explicit display modality and export processed outputs.

## Why This Cell Exists

This is the analysis and delivery step: inspect final curves across profiles/datasets and publish portable outputs for fitting, reporting, and downstream pipelines.

## Canonical Usage

```python
stitched.refl.compare_profiles(
    mode="waterfall",      # or "subplots"
    waterfall_scale=1.7,
)

stitched.refl.export_parquet(out_dir="processed")
```

## Implementation Context

### Comparison semantics

- comparison operates on final stitched outputs, not pre-stitch diagnostics
- mode is explicit to avoid ambiguous plotting behavior:
  - `waterfall` for dense multi-profile overlays with multiplicative offsets
  - `subplots` for independent per-profile visual inspection

## Comparison Mode Contract

- `mode` is required
- `waterfall`:
  - multiplicative offset between profiles
  - controlled by `waterfall_scale`
- `subplots`:
  - one panel per profile
  - optionally accepts provided axes set

## Export Contract

- write flat parquet outputs for processed profile data
- include provenance columns required for traceability

### Required export fields

- independent variable (`q` or chosen domain representation)
- `intensity`
- `uncertainty`
- `profile_id`
- `scan_number`
- `beamtime_key`
- role/quality flags needed for downstream filtering

### Packaging behavior

- export should support both per-profile files and combined table output
- schema must remain stable across runs to support automated consumers

## Catalog touchpoints

- reads from final stitched `reflectivity` and `stitch_corrections`.
- optional export manifest entries may be stored in processing-run metadata.

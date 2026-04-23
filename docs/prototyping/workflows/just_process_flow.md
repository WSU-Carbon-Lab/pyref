# Just Process Flow

## Goal

Provide a minimal, reliable entry point for users who want complete processing without manual tuning.

## Why This Flow Exists

This flow mirrors the practical behavior of legacy loader workflows while still running through the same catalog-first and zarr-backed pipeline as the DIY path.

## Canonical Usage

```python
import pyref as pr

result = pr.process(
    beamtime="2026Feb",
    sample_name="sample_a",
    scan_numbers=[1201, 1202],
    engine="polars",
)
```

## Implementation Context

### Required internal composition

`pr.process(...)` should compose the same core stages as Cells 1-9:

1. beamtime resolution and frame/profile loading
2. profile split and role classification
3. mask-aware beamspot detection
4. reduction with uncertainty propagation
5. QC filtering and stitch computation
6. final profile packaging and optional export handles

## Required Behavior

- starts from catalog beamtime key
- applies trusted defaults for:
  - classification
  - mask and beamspot detection
  - reduction and uncertainty
  - stitching
- persists run parameters and outcomes in catalog

### Defaulting rules

- defaults should be deterministic and documented.
- each default maps directly to one DIY parameter so users can escalate without semantic drift.
- every run writes a reproducibility record (parameters, versions, data scope).

## Escalation Path

`result` should expose intermediate DataFrames so users can switch into DIY mode without rerunning from raw ingest.

Recommended result surface:

- `result.frames`
- `result.profiles`
- `result.classified`
- `result.beam`
- `result.reduced`
- `result.stitched`

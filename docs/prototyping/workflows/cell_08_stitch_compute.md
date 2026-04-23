# Cell 08: Stitch Computation

## Goal

Compute final stitched reflectivity using explicit channel normalization semantics and propagated uncertainty.

## Why This Cell Exists

This step converts role-labeled reduced frames into a single calibrated profile by applying normalization and overlap-derived scale factors with full uncertainty handling.

## Canonical Usage

```python
stitched = reduced.refl.compute_stitch(
    normalize_chans=["beam_current", "exposure", "i0"],
    overlap_estimator="weighted_mean",
    propagate_errors=True,
)
```

## Implementation Context

### Normalization sequence

1. normalize by selected instrument channels in `normalize_chans` (for example beam current and exposure)
2. normalize first stitch block by mean i0 reference
3. compute overlap scale factor for each subsequent stitch boundary
4. propagate scaling and uncertainty through entire stitched chain

## API Notes

- required argument name is `normalize_chans`
- overlap scaling computed per stitch transition
- uncertainties propagated through all scaling operations

### Overlap estimator requirements

- default estimator is weighted mean on overlap ratios
- weights derive from inverse variance
- estimator diagnostics should report fit quality and effective overlap count

## Persistence

- write stitch factors and diagnostics to catalog
- preserve per-frame role and quality flags in output tables

## Catalog touchpoints

- writes per-stitch records to `stitch_corrections`.
- updates stitched frame outputs in `reflectivity` with final normalized intensity and uncertainty.

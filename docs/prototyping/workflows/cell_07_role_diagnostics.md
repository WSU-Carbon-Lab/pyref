# Cell 07: Role-Specific Diagnostics

## Goal

Provide diagnostics focused on specific frame roles with selectable layout mode.

## Why This Cell Exists

Before stitching, role-specific failure modes must be surfaced and optionally excluded. This cell isolates diagnostics by physics role so users can make defensible bad-point decisions.

## Canonical Usage

```python
reduced.refl.plot_role_diagnostics(
    roles=["i0", "stitch", "overlap"],
    layout="subplots",        # or "contiguous"
    split_overlap_regions=True,
)
```

## Implementation Context

### Role segmentation inputs

- uses `frame_role` labels from classification step
- uses reduced intensities and uncertainty components from reduction step
- uses quality metadata from beamspot/mask steps

## Role Modes

- `i0`: direct beam consistency checks
- `stitch`: initial stitch-point checks
- `overlap`: overlap alignment checks

## Layout Modes

- `contiguous`: single axes with contiguous concatenation
- `subplots`: separate axes for i0, each stitch start, and overlap regions

## QC Targets

- saturation and near-saturation points
- beam-loss candidates
- overlap inconsistencies used for exclusion decisions

### Additional checks expected

- i0 constancy across direct-beam blocks
- stitch-start stabilization checks
- overlap-region agreement metrics before scale-factor fitting

### Output requirements

- role-filtered diagnostic tables for programmatic filtering
- visualization mode chosen by `layout`
- optional write-back of bad-point flags with reason codes

## Catalog touchpoints

- reads `profile_frames.frame_role`, `reflectivity`, and quality flags.
- writes or updates bad-point quality flags used by stitch computation.

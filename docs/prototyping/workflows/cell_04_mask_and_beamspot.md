# Cell 04: Integrated Mask and Beamspot Detection

## Goal

Run mask-aware beamspot detection with polished interactive notebook tooling, regression diagnostics, and remediation.

## Why This Cell Exists

Beamspot localization is the key geometric primitive for intensity integration. This step validates beam position models, surfaces outliers, and gives the user interactive correction tools before reduction.

## Canonical Usage

```python
beam = classified.refl.detect_beamspots(
    mask_mode="include_region",   # or "exclude_region"
    regression="linear",          # "fixed", "poly-3", "poly-{n}"
    confidence_band=0.95,
    interactive=True,
)

beam.refl.plot_beamspot_diagnostics(
    y=["beam_row", "beam_col"],
    with_confidence=True,
)
```

## Implementation Context

### Processing sequence

- materialize per-frame images from catalog-linked zarr storage through the DataFrame image accessor.
- apply configured mask policy (`include_region` or `exclude_region`).
- run preprocessing (edge handling, background subtraction, gaussian smoothing).
- fit beamspot and beam shape parameters (center row/col, width/height, fit quality).
- fit row and col trajectories versus domain variable with selected regression mode.

### Regression/diagnostic outputs

- fitted row/col trajectory
- confidence band envelopes
- residuals and outlier flags
- model mode metadata (`linear`, `fixed`, `poly-n`)

## Regression Modes

- `linear`: row and col each fit with linear model
- `fixed`: single constant row/col target
- `poly-3`: cubic model
- `poly-{n}`: arbitrary polynomial degree

## Interactive Remediation

- show model-predicted spot
- user can refine with marquee or crosshair
- user can select bad spots and apply remediation actions
- interactions are expected to be ergonomic and production-ready for notebook use
- plot widgets should support frame stepping, zoom, and clear state feedback

### Interactive API requirements

- marquee and crosshair correction modes
- overlay of suggested/model-predicted beam location
- one-click mark-bad plus reason tagging
- ability to reopen a frame and edit prior correction without losing audit trail

## Persistence

- corrected points persist as overrides in catalog
- model diagnostics and confidence metadata are stored for audit

## Catalog touchpoints

- reads frame/image locators from `frames` and zarr paths via `beamtimes`.
- writes detection results to `beam_finding`.
- writes remediation overrides and quality flags linked to frame IDs.

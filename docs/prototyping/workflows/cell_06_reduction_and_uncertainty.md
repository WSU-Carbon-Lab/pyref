# Cell 06: Reduction and Uncertainty Contributions

## Goal

Reduce masked images into intensities and explicitly visualize contributions from each uncertainty source.

## Why This Cell Exists

This is the quantitative core of the pipeline: convert 2D detector data into reflectivity intensities with traceable uncertainty propagation.

## Canonical Usage

```python
reduced = beam.refl.reduce_intensity(
    roi_strategy="i0_gaussian_3sigma",
    uncertainty={"shot_noise": True, "roi_loss": True, "fano_factor": True},
)

reduced.refl.plot_uncertainty_contributions(x="domain_value")
```

## Implementation Context

### Reduction model

Per frame:

- integrate signal in ROI centered at localized beamspot
- integrate dark/background region
- compute net intensity as `signal - dark`

ROI policy:

- derive expected ROI geometry from i0 beam-shape statistics
- use gaussian-fit widths/heights and a `3 sigma` rule by default

### Uncertainty model

Supported contributors:

- shot/counting noise
- ROI-loss term from finite ROI truncation relative to beam model
- Fano factor scaling inferred from i0 point ensembles
- optional background-variance terms

Propagation requirements:

- contributors are propagated in quadrature where appropriate
- per-source components remain queryable, not only combined uncertainty

## Uncertainty Sources

- `shot_noise`
- `roi_loss`
- `fano_factor`
- optional dark/background terms if enabled in pipeline

## Plot Contract

- stacked or grouped contribution view
- per-source absolute and fractional contribution
- consistent domain axis with main reduced intensity output

## Catalog touchpoints

- reads beamspot and mask state from prior cells.
- writes reduced frame-level outputs to `reflectivity` (or staged run table).
- writes uncertainty-component metadata to run/pipeline state for later diagnostics.

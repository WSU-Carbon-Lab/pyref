# Layout: subplots, GridSpec, mosaic, separate figures

## Decision guide

| Situation | Prefer |
|-----------|--------|
| Regular grid of equal panels, shared labels | `plt.subplots(nrows, ncols, sharex=..., sharey=..., layout="constrained")` |
| Variable row/column sizes, spans | `matplotlib.gridspec.GridSpec` or `fig.subplot_mosaic` with width ratios |
| Semantic names (`"A"`, `"B"`) instead of integer positions | `fig.subplot_mosaic` |
| One-off complex arrangement | `GridSpec` with `subplot_spec` spans |
| Same plot reused in slide vs paper with different sizes | **Separate** `fig.savefig` calls or separate scripts; avoid one giant figure that is always resized in a GUI |
| Final page composition with captions from a manuscript | Export **individual** high-quality panels (PDF/SVG) and place in LaTeX/Quarto; optional single composite for preview only |

## `plt.subplots`

- Use for the common case: `fig, axs = plt.subplots(2, 2, layout="constrained")`.
- `sharex` / `sharey` reduce duplicate ticks; call `ax.label_outer()` or hide inner tick labels for clarity.
- `axs` may be 1D or 2D array; flatten with `axs.flat` for iteration.

## `subplot_mosaic`

- Readable layout strings and named axes:

```python
fig, axd = plt.subplot_mosaic(
    "AB\nCC",
    layout="constrained",
)
axd["A"].plot(...)
```

- Good when panel sizes are uneven or you refer to panels by role.

## `GridSpec`

- Use when you need **explicit height/width ratios** and **row/column spans** that are awkward in `subplots`.
- Pair with `fig.add_subplot(gs[i, j])` or `subgridspec` for nested regions.

## Layout engines

- Prefer **`layout="constrained"`** (or `fig.set_layout_engine("constrained")`) for label and colorbar spacing over manual `subplots_adjust` when possible. See [Constrained layout guide](https://matplotlib.org/stable/users/explain/axes/constrainedlayout_guide.html).
- **`tight_layout`** remains useful for quick notebooks; constrained layout is usually better for publication figures with long labels.

## Assemble elsewhere

- **Assemble in the document** when the journal controls gutters, multi-panel captions, or when vector panels must align with equations.
- **Assemble in matplotlib** when you need a single raster for Twitter/slides or a strict pixel budget.

## Multi-panel data stories

- **Small multiples** (one highlighted series per panel, others grey) help many categories; see [Plotting Zoo](https://www.practicaldatascience.org/notebooks/class_5/week_1/1.5.1_plotting_zoo.html).

## Anti-pattern

- Using **`plt.plot` after `subplots`** without `plt.sca(ax)` routes draws to the **last-created** axes and stacks traces incorrectly. Always **`ax.plot`** ([implicit vs explicit](https://www.practicaldatascience.org/notebooks/class_5/week_1/1.4.5_explicit_vs_implicit_syntax.html)).

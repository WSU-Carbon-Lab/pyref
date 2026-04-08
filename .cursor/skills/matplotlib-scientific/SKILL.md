---
author: dotagents
name: matplotlib-scientific
description: Build publication-quality Matplotlib figures for scientific Python. Use when plotting experiment or analysis results, multi-panel figures, journal-sized layouts, tick formatting, legends (fancy/plain, placement, compact handles), point annotations, inset zooms, twin axes, colored spines/ticks, colorblind-safe palettes, PNG/PDF export, or SciencePlots styling. Triggers on matplotlib, pyplot, subplots, savefig, legend, annotate, inset, twinx, scientific figures, panel labels.
---

# Matplotlib for scientific figures

## Quick start

1. Use the **object-oriented API**: create `fig, ax = plt.subplots()` (or `subplot_mosaic` / `GridSpec`), then call methods on `ax`. Avoid `plt.plot` when multiple axes exist; implicit pyplot targets the wrong axis by default. See [Practical Data Science: explicit vs implicit syntax](https://www.practicaldatascience.org/notebooks/class_5/week_1/1.4.5_explicit_vs_implicit_syntax.html) and [basic plotting](https://www.practicaldatascience.org/notebooks/class_5/week_1/1.2.1_basic_plotting_with_matplotlib.html).
2. **Name every quantitative axis** with units in the label (e.g. `Time (s)`, `Voltage (V)`). Match tick style to audience: disable offset/scientific clutter when plain decimals read better; use scientific or engineering notation when magnitudes span orders. Details: [reference-axes-ticks.md](references/reference-axes-ticks.md).
3. **Save deliberately**: high DPI raster (often 300–600) for PNG; `bbox_inches="tight"` when labels are clipped; **`transparent=True`** for PNG overlays and slides. Prefer vector PDF/SVG for final print when the journal allows. See [reference-export-naming.md](references/reference-export-naming.md).
4. **Panel letters** `(a)`, `(b)` for multi-panel figures: consistent placement, bold, same size as body or slightly larger. See [Matplotlib: labelling subplots](https://matplotlib.org/stable/gallery/text_labels_and_annotations/label_subplots.html) and [reference-axes-ticks.md](references/reference-axes-ticks.md).
5. **Figure width**: size figures to **single-column** or **full-page / double-column** width per target venue; do not default to square notebook sizes for papers. See [reference-journal-layout.md](references/reference-journal-layout.md) and [Simplified Science: figure design rules](https://www.simplifiedsciencepublishing.com/resources/how-to-make-good-figures-for-scientific-papers).
6. **Composable code**: write helpers that **accept `Axes` or `Figure` and return** the same object after mutation. See [reference-api-patterns.md](references/reference-api-patterns.md).
7. **Polish**: match legend frame style to venue (plain for print, slightly rounded for slides); place legends where they do not obscure data; annotate only points the text discusses; use insets and twin axes only when they sharpen the claim. See [reference-legend-style.md](references/reference-legend-style.md), [reference-annotations-insets-twins.md](references/reference-annotations-insets-twins.md), [reference-axes-appearance.md](references/reference-axes-appearance.md).

## Stack synergy

| Resource | Role |
|----------|------|
| **general-python** | uv, **ty**, export scripts, **python-reviewer** |
| **numpy-scientific** | Array inputs to plots |
| **dataframes** | `df.plot(ax=ax)` and Polars-to-pandas at plot boundaries |
| **numpy-docstrings** | Docstrings for plotting helpers and figure builders |

## Reference index (load the section you need)

| Topic | File |
|--------|------|
| Subplots vs `GridSpec` vs `subplot_mosaic`; assembling panels | [reference-layout.md](references/reference-layout.md) |
| Axis labels, units, ticks, scientific notation, panel labels | [reference-axes-ticks.md](references/reference-axes-ticks.md) |
| Spine/tick/label color, twin-axis styling | [reference-axes-appearance.md](references/reference-axes-appearance.md) |
| Legends: fancy frame, title, placement, compact handles | [reference-legend-style.md](references/reference-legend-style.md) |
| Point annotations, inset zooms, twin / secondary axes | [reference-annotations-insets-twins.md](references/reference-annotations-insets-twins.md) |
| Color maps, cycles, colorblind safety, data types | [reference-color.md](references/reference-color.md) |
| `savefig`, transparency, DPI, auto-increment filenames | [reference-export-naming.md](references/reference-export-naming.md) |
| Journal widths, composition, storytelling | [reference-journal-layout.md](references/reference-journal-layout.md) |
| SciencePlots / style sheets; `pandas` `.plot(ax=ax)` | [reference-styles-pandas.md](references/reference-styles-pandas.md) |
| Functions that take/return `Axes` / `Figure` | [reference-api-patterns.md](references/reference-api-patterns.md) |
| Rasterized PDF, `align_labels`, CI, memory | [reference-advanced.md](references/reference-advanced.md) |

## Best-practice topics (checklist)

- **Chart choice**: match geometry to the claim (trend vs part-to-whole vs distribution). Same data, many encodings: [Plotting Zoo](https://www.practicaldatascience.org/notebooks/class_5/week_1/1.5.1_plotting_zoo.html).
- **Anatomy**: figure, axes, marks, axis labels, ticks and tick labels, limits, grid, legend, title, spines; compact `ax.set(...)`. See [A figure in 10 pieces](https://www.practicaldatascience.org/notebooks/class_5/week_1/1.2.2_ten_figure_pieces.html).
- **Layering**: set `zorder` and distinct `color` when mixing `plot` / `scatter` / `bar` on one axes ([basic plotting](https://www.practicaldatascience.org/notebooks/class_5/week_1/1.2.1_basic_plotting_with_matplotlib.html)).
- **Legends**: plain frame and no shadow for most papers; `bbox_to_anchor` outside or `ncol` below when the data region is crowded ([reference-legend-style.md](references/reference-legend-style.md)).
- **Annotations**: few, uniform `annotate` styling; insets for zoom only when the story needs local detail ([reference-annotations-insets-twins.md](references/reference-annotations-insets-twins.md)).
- **Axes color**: tie spine/tick/label color to a twin axis accent; keep grids low-contrast ([reference-axes-appearance.md](references/reference-axes-appearance.md)).
- **Ticks**: treat **locator** (where) and **formatter** (text) separately. See [Axis ticks](https://matplotlib.org/stable/users/explain/axes/axes_ticks.html) and [ScalarFormatter](https://matplotlib.org/stable/gallery/ticks/scalarformatter.html).
- **Reproducibility**: script generates the file on disk; version-control style choices; meeting previews use predictable auto-increment names ([reference-export-naming.md](references/reference-export-naming.md)).
- **Accessibility**: do not rely on color alone; use line styles / markers; check contrast; prefer perceptually uniform colormaps for continuous data ([reference-color.md](references/reference-color.md)).
- **SciencePlots** (optional): journal-oriented style sheets; often requires LaTeX. See [SciencePlots](https://github.com/garrettj403/SciencePlots) and [reference-styles-pandas.md](references/reference-styles-pandas.md).
- **Pandas quick plots**: `df.plot(..., ax=ax)` returns matplotlib artists; refine with `ax` methods. See [Plotting with Pandas](https://www.practicaldatascience.org/notebooks/class_5/week_1/1.6.1_plotting_with_pandas.html).

## Matplotlib documentation (official)

- [User guide / tutorials](https://matplotlib.org/stable/users/index.html)
- [Anatomy of a figure](https://matplotlib.org/stable/gallery/showcase/anatomy.html)
- [Arranging multiple Axes](https://matplotlib.org/stable/users/explain/axes/arranging_axes.html) (subplots, GridSpec, mosaic)
- [Constrained layout](https://matplotlib.org/stable/users/explain/axes/constrainedlayout_guide.html) and [tight layout](https://matplotlib.org/stable/users/explain/axes/tight_layout_guide.html)
- [Legend guide](https://matplotlib.org/stable/users/explain/axes/legend_guide.html), [Annotations](https://matplotlib.org/stable/tutorials/text/annotations.html)
- [Secondary axis](https://matplotlib.org/stable/gallery/subplots_axes_and_figures/secondary_axis.html), [Zoom inset](https://matplotlib.org/stable/gallery/subplots_axes_and_figures/zoom_inset_axes.html)
- [Figure.savefig](https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure.savefig) (API); workflow notes in [Saving to file (PDS)](https://www.practicaldatascience.org/notebooks/class_5/week_1/1.4.4_saving_to_file.html)

## When not to overload one figure

Split into **separate figures** when panels are reused across talks vs papers, or when layout constraints differ (poster vs manuscript). Compose in a layout script or document (LaTeX/Quarto) when the page design, not matplotlib, should own margins and alignment.

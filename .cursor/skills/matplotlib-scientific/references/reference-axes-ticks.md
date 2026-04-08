# Axes labels, units, ticks, notation, panel labels

## Axis labels and units

- Put **quantity and unit** in the label: `ax.set_xlabel("Time (ms)")`, `ax.set_ylabel("Power spectral density (V^2/Hz)")`.
- Use **consistent SI style** across a figure set; square brackets for unit-only annotations are acceptable where your field expects them (e.g. `Voltage [V]`).
- For dimensionless quantities, state it: `Strouhal number` or `Normalized frequency (f / f_0)`.

## The ten pieces (mental model)

Figure container, axes, marks (line/scatter/bar), **x/y labels**, **ticks and tick labels**, **limits**, grid, legend, title, spines. Prefer **`ax.set(...)`** for bulk updates when it improves readability. See [A figure in 10 pieces](https://www.practicaldatascience.org/notebooks/class_5/week_1/1.2.2_ten_figure_pieces.html).

## Ticks: locators and formatters

- **Positions**: `ax.set_xticks(...)`, `MultipleLocator`, `MaxNLocator`, `LogLocator`, etc.
- **Strings**: `ax.set_xticklabels(...)`, or set a `Formatter` on `ax.xaxis`.
- Default numeric formatter is **`ScalarFormatter`**. Useful controls:
  - `ax.ticklabel_format(style="plain", axis="y")` to discourage scientific notation on an axis when values are human-scaled.
  - `ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))` when all decades should use scientific form.
  - `ax.yaxis.get_major_formatter().set_useOffset(False)` to suppress the offset term when it confuses readers.
- For wide dynamic range on linear axes, consider **log scale** (`ax.set_yscale("log")`) instead of cramming exponents into tick labels.
- Deep reference: [Axis ticks](https://matplotlib.org/stable/users/explain/axes/axes_ticks.html), [ScalarFormatter demo](https://matplotlib.org/stable/gallery/ticks/scalarformatter.html).

## Panel labels (a), (b), (c)

- Convention: **bold**, upper or lower case per venue, **same font size** across panels, placed consistently (often **top-left** inside axes using axes coordinates).
- Official patterns: [Labelling subplots](https://matplotlib.org/stable/gallery/text_labels_and_annotations/label_subplots.html) (`annotate`, `text` with transforms, or title `loc="left"`).
- Minimal loop pattern:

```python
import string
for ax, tag in zip(axs.flat, string.ascii_lowercase):
    ax.text(
        0.02,
        0.98,
        f"({tag})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
        fontsize=11,
    )
```

- Adjust `(0.02, 0.98)` if constrained layout clips labels; use figure coordinates or `mpl_toolkits.axes_grid1` helpers if needed.

## Legend (overview)

- Pass **`label=`** to each plotting call, then `ax.legend(...)`.
- For crowded panels, **`bbox_to_anchor`** outside the axes preserves data ink ([ten pieces](https://www.practicaldatascience.org/notebooks/class_5/week_1/1.2.2_ten_figure_pieces.html)).
- Fancy vs plain frames, legend titles, placement above/below/outside, and compact handles: [reference-legend-style.md](reference-legend-style.md).

## Mathtext and LaTeX

- Mathtext: `ax.set_xlabel(r"$\omega$ (rad/s)")` without full LaTeX install.
- `plt.rcParams["text.usetex"] = True` requires a TeX system; **SciencePlots** often expects LaTeX. See [reference-styles-pandas.md](reference-styles-pandas.md).

## Grid and spines

- `ax.grid(True, alpha=0.3)` for read values without chartjunk.
- Hiding **top/right** spines is common for 2D scientific plots ([ten pieces](https://www.practicaldatascience.org/notebooks/class_5/week_1/1.2.2_ten_figure_pieces.html)).
- Colored spines, tick colors, and label colors (especially with **twin** axes): [reference-axes-appearance.md](reference-axes-appearance.md).

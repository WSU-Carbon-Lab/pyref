# Advanced publication topics (optional)

## Vector files with heavy scatter

- PDFs balloon when every point is a vector path. For large `scatter`, use **`rasterized=True`** on the collection or rasterize the axes in the PDF backend so data draw as image while text stays vector.

## Label alignment across panels

- After setting xlabels/ylabels on a grid, call **`fig.align_labels()`** so stacked panels line up when label lengths differ.

## Font sizes and RC

- Set **`axes.titlesize`**, **`axes.labelsize`**, **`xtick.labelsize`**, **`legend.fontsize`** once per style for consistent figures. SciencePlots and journal styles override many of these.

## Secondary axes

- `ax.secondary_xaxis` / `secondary_yaxis` for dual scales: document both units in axis labels or caption to avoid ambiguity. Patterns and twin-axis caution: [reference-annotations-insets-twins.md](reference-annotations-insets-twins.md).

## 3D and polar

- Use only when the geometry is inherently 3D or angular; avoid 3D for ranking bar charts. See Matplotlib gallery for projection setup.

## Interactive vs static

- **`plt.show()`** is for interactive sessions; batch pipelines should **`savefig`** and close figures (`plt.close(fig)`) to limit memory when generating many plots.

## Testing plots

- Smoke-test plotting code with **`matplotlib.use("Agg")`** in CI so no display server is required.

## Accessibility metadata

- For web exports, pair complex graphics with a **data table** or summary in the document; color alone must not carry exclusive meaning ([Simplified Science rules](https://www.simplifiedsciencepublishing.com/resources/how-to-make-good-figures-for-scientific-papers)).

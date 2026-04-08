# Point annotations, inset axes, twin axes

## When to annotate data points

Annotate **sparingly**:

- **Named exemplars** (a specific run, date, or condition called out in the text).
- **Threshold crossings** or **regulatory limits** (horizontal/vertical reference lines plus one short label often beat many point labels).
- **Outliers** that the narrative discusses.

Avoid labeling **every** point unless the figure is a small-N schematic; for dense scatter, prefer **interactive** exploration or a **table** supplement.

## Consistent, readable point labels

Use **`ax.annotate`** so text and arrow share one API; prefer **data coordinates** for `xy` and **offset points** for `xytext` so font size and zoom stay consistent.

```python
ax.annotate(
    "A",
    xy=(x0, y0),
    xytext=(8, 8),
    textcoords="offset points",
    ha="left",
    va="bottom",
    fontsize=9,
    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", lw=0.5),
    arrowprops=dict(arrowstyle="-", color="0.35", lw=0.6, shrinkA=0, shrinkB=2),
)
```

**Consistency rules across a figure:**

- One **`fontsize`** and **`bbox`** style for all callouts in the same figure.
- **`offset points`** in the same direction (e.g. always up-right) unless avoiding overlap.
- For overlapping labels, **nudge** `xytext` or use **`adjustText`** (third-party) in exploratory work; for publication, **manually** place or reduce the number of labels.

**Alternative without arrows:** `ax.text` in data coords when proximity is obvious.

## Reference lines vs point annotations

- Vertical/horizontal rules: `ax.axvline`, `ax.axhline` with **`label=`** and legend entry, or a short **`transform=blended`** label on the line.
- Keeps the main legend for series, not for every marker.

## Twin axes (`twinx` / `twiny`)

- **One twin** is standard when two quantities share **time or position** but differ in **unit or scale** (e.g. temperature and humidity vs depth).

```python
ax2 = ax.twinx()
ax2.plot(x, y2, color="C1")
ax2.set_ylabel("Secondary quantity (unit)", color="C1")
ax2.tick_params(axis="y", colors="C1")
```

**Multiple scales on the same side** are cognitively heavy. Prefer:

- **Normalizing** to one axis and stating the mapping in the caption, or
- **A second panel** (small multiples), or
- **`secondary_yaxis`** for an alternate **functional** scale of the same quantity (e.g. °C ↔ °F) rather than a third unrelated series.

Official pattern: [Secondary axis](https://matplotlib.org/stable/gallery/subplots_axes_and_figures/secondary_axis.html).

## Inset axes (zoom, context, detail)

Use an **inset** when the main axes shows **context** and the inset shows **a zoom** or a **secondary view** (e.g. map overview + city blow-up).

**Recommended API:** `ax.inset_axes(bounds)` where `bounds` is `[left, bottom, width, height]` in **axes coordinates** (0–1), or use `mpl_toolkits.axes_grid1.inset_locator.inset_axes` for size in **axes fraction / padding**.

```python
axins = ax.inset_axes([0.55, 0.55, 0.4, 0.4])
axins.plot(x, y)
axins.set_xlim(x0, x1)
axins.set_ylim(y0, y1)
ax.indicate_inset_zoom(axins, edgecolor="0.3")
```

- Match **line weights** and **colors** to the main panel unless the inset is a different modality.
- Add **ticks** on the inset; for tiny insets, fewer ticks are better than cluttered decimals.
- **`indicate_inset`** / **`indicate_inset_zoom`** links inset to region when helpful.

Gallery: [Inset locator demo](https://matplotlib.org/stable/gallery/axes_grid1/inset_locator_demo.html), [Zoom inset axes](https://matplotlib.org/stable/gallery/subplots_axes_and_figures/zoom_inset_axes.html).

## Z-order

- Insets and annotations usually sit **above** data: set artist **`zorder`** or add them **after** plotting series.

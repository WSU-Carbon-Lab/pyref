# Legends: frame style, titles, placement, compact handles

## Fancy frame vs plain frame

| Context | Recommendation |
|---------|------------------|
| Journal print, minimal figures | **`fancybox=False`**, thin neutral frame: `ax.legend(frameon=True, edgecolor="0.85", facecolor="white", fancybox=False)` |
| Slides, posters, light branding | **`fancybox=True`** (rounded corners) is acceptable; keep **`shadow=False`** unless depth is part of the design system |
| On busy backgrounds | **`frameon=True`** with solid **`facecolor`** (often white or figure face) so text stays legible |

`fancybox=True` uses a rounded box; `fancybox=False` is sharper and usually reads more “paper”. `shadow=True` ages poorly in print; prefer a crisp edge and padding.

## Legend title (`title=`)

Add a legend title when:

- Handles are **grouped by class** and the axis label alone is insufficient (e.g. title `"Model"`, entries `RF`, `NN`).
- The same color/marker means **different semantics** than the axis (e.g. `"Dataset"` while y-axis is a measured quantity).

Skip the title when:

- The axis label and series names are already self-explanatory.
- Space is tight; a **caption** can name the encoding instead.

```python
leg = ax.legend(title="Condition", alignment="left")
leg.get_title().set_fontsize(leg.get_texts()[0].get_fontsize())
```

Keep title **one size step** above or equal to entry text; align left for multi-line entries.

## Placement: inside vs above vs below vs outside

| Placement | When to use | Typical kwargs |
|-----------|-------------|----------------|
| **Inside** upper right / best | Few series, large margin, no occlusion | `loc="upper right"`, `bbox_to_anchor=(1, 1)` only if nudging inside |
| **Outside** right | Many series or long names; preserve data region | `bbox_to_anchor=(1.02, 1)`, `loc="upper left"` |
| **Below** the axes | Wide figures, many entries; row of handles | `bbox_to_anchor=(0.5, -0.18)`, `loc="upper center"`, **`ncol`** ≥ 2 |
| **Above** | Rare; sometimes for single-row keys under a suptitle | `bbox_to_anchor=(0.5, 1.12)`, `loc="lower center"` |

Use **`layout="constrained"`** on the figure so outside legends reserve space. After placing outside, verify **`savefig(..., bbox_inches="tight")`** does not clip the box.

## Making the legend read cleanly

- **`ncol`**: split into 2–4 columns for wide single-row legends below the plot.
- **`labelspacing`**, **`handlelength`**, **`handletextpad`**, **`borderaxespad`**: tighten uniformly; change one parameter at a time.
- **`fontsize`**: match tick labels or be one step smaller than axis labels.
- **`alignment="left"`** (Matplotlib 3.6+): left-align stacked text blocks.

## Compact handles (shorter legend rows)

When default lines are too long:

1. **Proxy artists**: build handles explicitly with short linestyle/marker only.

```python
from matplotlib.lines import Line2D

handles = [
    Line2D([0], [0], color="C0", lw=2, label="Control"),
    Line2D([0], [0], color="C1", lw=2, ls="--", label="Treated"),
]
ax.legend(handles=handles, labels=[h.get_label() for h in handles], frameon=True)
```

2. **`numpoints=1`**, **`markerscale`** for scatter-heavy legends.
3. **`handler_map`** for custom patches (e.g. thin rectangles for bars) when defaults are oversized.

## Order and merging

- **`order="sorted"`** or explicit label order for consistency across figures.
- **`labelcolor="linecolor"`** (or `"markerfacecolor"`) ties text to encoding and can replace redundant color words in labels.

## Matplotlib references

- [Legend guide](https://matplotlib.org/stable/users/explain/axes/legend_guide.html)
- [Figure legends](https://matplotlib.org/stable/gallery/text_labels_and_annotations/figlegend_demo.html)
- [Custom legend handlers](https://matplotlib.org/stable/tutorials/intermediate/legend_guide.html#implementing-a-custom-legend-handler)

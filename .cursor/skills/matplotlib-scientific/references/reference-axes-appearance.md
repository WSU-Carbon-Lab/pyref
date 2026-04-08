# Spine, tick, and label color (cohesive axes styling)

## Why color axes elements

Use **one accent color per y-axis** when that axis encodes a **single physical quantity** (especially with **`twinx`**) so readers bind ticks, label, and spine to the correct series. Keep **x-axis** styling neutral unless the bottom axis is also doubled.

## Spines

```python
accent = "C0"
ax.spines["left"].set_color(accent)
ax.spines["bottom"].set_color("0.25")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(1.0)
```

- **Muted** spine color (`0.2`–`0.4` grayscale or desaturated brand) reads more “scientific” than pure black on white.
- Match **spine color** to the **same-axis** tick and label color when emphasizing a twin.

## Ticks and tick labels

```python
ax.tick_params(axis="y", colors=accent, which="both")
ax.tick_params(axis="x", colors="0.25", which="both")
ax.yaxis.label.set_color(accent)
ax.xaxis.label.set_color("0.25")
```

- **`which="both"`** applies to major and minor ticks when minors are on.
- Minor ticks: `ax.minorticks_on()` then tune length/width via `tick_params`.

## Axis labels and titles

- **`ax.xaxis.label.set_color`** / **`ax.yaxis.label.set_color`** align label with spine accent.
- **Title** color: default body color or slightly darker; avoid competing with data saturation.

## Grid vs colored spines

- If **`ax.grid(True)`** is on, keep grid **low contrast** (`alpha=0.25–0.35`, light grey) so colored spines still read as the primary frame.

## Dark backgrounds (slides)

- Invert logic: light spines and labels on dark **`figure.facecolor`**; test **contrast** for projectors.
- Legend **`facecolor`** should match slide background or stay opaque white for readability.

## Matplotlib references

- [Spines](https://matplotlib.org/stable/api/spines_api.html)
- [tick_params](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.tick_params.html)
- [Colorblind considerations](reference-color.md) still apply: color on spines must not be the only discriminator between series.

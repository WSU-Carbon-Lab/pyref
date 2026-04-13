# Color, colormaps, and cycles for scientific data

## Match encoding to data type

| Data role | Prefer |
|-----------|--------|
| Ordered numeric (low to high) | Perceptually uniform sequential: `viridis`, `cividis`, `plasma`; avoid **jet** |
| Deviation from a reference | Diverging: `coolwarm`, `RdBu_r`; center the norm at the reference |
| Categories | Distinct hues; **limit palette** to the number of classes; add **markers or dashes** for grayscale safety |
| Uncertainty bands | Lower alpha or lighter fill; keep the central estimate high-contrast |

## Colorblind safety and grayscale

- Do not encode **only** with red vs green. Combine **hue + linestyle + marker**.
- Test a figure by **desaturating** or printing grayscale: the main comparison should survive.
- [SciencePlots](https://github.com/garrettj403/SciencePlots) includes cycles such as **bright** and **high-vis**; Paul Tol palettes are available as named styles.

## Overlays on one axes

- When mixing `plot`, `scatter`, and `bar`, set explicit **`color`** and **`zorder`** so categories do not hide each other ([basic plotting](https://www.practicaldatascience.org/notebooks/class_5/week_1/1.2.1_basic_plotting_with_matplotlib.html)).

## Storytelling vs decoration

- Per [Simplified Science figure rules](https://www.simplifiedsciencepublishing.com/resources/how-to-make-good-figures-for-scientific-papers), use **one or two accent colors** for the main claim and mute context series (grey).

## Norms for images and fields

- Use `TwoSlopeNorm` or `SymLogNorm` when data spans zero with outliers; document the norm in the caption.
- For maps and fields, state the **colorbar** label with units (`imshow(...); fig.colorbar(im, ax=ax, label="K")`).

## Defaults

- Set **`axes.prop_cycle`** via a style sheet rather than hard-coding colors in every script when the project has a house style.

## Coordinated axis color

- When a series and its **y-axis** share an accent (e.g. after `twinx`), align **spine, ticks, and axis label** to that color without breaking grayscale legibility. See [reference-axes-appearance.md](reference-axes-appearance.md).

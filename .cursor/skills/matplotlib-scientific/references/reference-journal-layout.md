# Journal widths, composition, and scientific storytelling

## Width targets (verify against current author guide)

Journals differ; **always read the target journal’s figure guidelines**. Order-of-magnitude widths from public summaries:

| Venue (examples) | Single column (order of magnitude) | Double / full width |
|-------------------|-------------------------------------|----------------------|
| Nature (typical) | ~89 mm | ~183 mm |
| Science (typical) | ~55 mm | ~230 mm |

Convert to inches for `figsize`: **inches = mm / 25.4**. Example: 89 mm ~ 3.5 in wide; height from aspect ratio and content.

```python
mm = 89
w_in = mm / 25.4
aspect = 0.75
fig, ax = plt.subplots(figsize=(w_in, w_in * aspect), layout="constrained")
```

## Four design rules (summary)

From [How to make good figures for scientific papers](https://www.simplifiedsciencepublishing.com/resources/how-to-make-good-figures-for-scientific-papers):

1. **Purpose**: choose the graphic form from the claim (compare, change over time, relationship, process).
2. **Composition**: left-to-right or top-to-bottom flow; remove chartjunk; emphasize the main series.
3. **Color**: few accents; colorblind-safe; grayscale should still read.
4. **Refine**: iterate; check “does the graphic alone convey the point?” before polishing text.

## Choosing the chart type

- Same table, many views: lines for **trends**, stacked bars or stackplots for **composition over time**, grouped bars for **within-year comparison**, pies only when **parts of one whole** and few slices. See [Plotting Zoo](https://www.practicaldatascience.org/notebooks/class_5/week_1/1.5.1_plotting_zoo.html).

## Tables as figures

- A **formatted table** is a valid figure for lookup-heavy results; pandas `DataFrame` display or manual annotation in the manuscript are both acceptable ([Plotting Zoo](https://www.practicaldatascience.org/notebooks/class_5/week_1/1.5.1_plotting_zoo.html)).

## Supplementary material

- Move **sensitivity analyses and extra scans** to supplement; keep main figures focused ([Iceberg-style thinking](https://www.simplifiedsciencepublishing.com/resources/how-to-make-good-figures-for-scientific-papers)).

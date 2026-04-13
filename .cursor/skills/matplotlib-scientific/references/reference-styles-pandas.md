# Style sheets, SciencePlots, pandas plotting

## Object-oriented + pandas

- Pattern from [Plotting with Pandas](https://www.practicaldatascience.org/notebooks/class_5/week_1/1.6.1_plotting_with_pandas.html):

```python
fig, ax = plt.subplots(layout="constrained")
df.plot(kind="bar", ax=ax, stacked=True, edgecolor="white")
ax.set_ylabel("Tons")
```

- `kind` options include `bar`, `barh`, `hist`, `box`, `kde`, `density`, `area`, `scatter`, `hexbin`, `pie`. The return value is a matplotlib object you can tweak further.

## Global styles

- Built-in: `plt.style.use("seaborn-v0_8-whitegrid")` or `ggplot`, `fivethirtyeight`, etc. ([pandas plotting example](https://www.practicaldatascience.org/notebooks/class_5/week_1/1.6.1_plotting_with_pandas.html)).
- Prefer **`plt.style.context(...)`** in libraries so you do not mutate global RC for importers.

## SciencePlots

- Project: [garrettj403/SciencePlots](https://github.com/garrettj403/SciencePlots).
- Install: `uv add SciencePlots` (or `pip install SciencePlots`). **LaTeX** is required for the default `"science"` style; the README documents `no-latex` variants and CJK font add-ons.
- Usage (v2+): **`import scienceplots`** before activating styles:

```python
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science", "ieee"])
```

- Combine styles: later entries override earlier (e.g. `ieee` column width over `science`).
- Temporary: `with plt.style.context(["science", "notebook"]): ...`

## Resolution in notebooks

- For crisp inline display, Practical Data Science suggests `matplotlib_inline.backend_inline.set_matplotlib_formats("retina")` instead of magic-only config when exporting notebooks to other runners ([basic plotting](https://www.practicaldatascience.org/notebooks/class_5/week_1/1.2.1_basic_plotting_with_matplotlib.html)).

## When not to use heavy styles

- Exploratory notebooks: keep default or light grid for speed.
- Submission: match **journal** requirements; SciencePlots is a shortcut, not a substitute for the author checklist.

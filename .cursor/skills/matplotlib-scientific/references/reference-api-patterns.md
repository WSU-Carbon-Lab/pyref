# API patterns: functions that accept and return `Axes` / `Figure`

## Principle

Plotting helpers should be **pure in intent**: given the same data and axes, they draw the same artists. Side effects stay on the **`Axes`** you pass in. Return the **`Axes`** (or **`Figure`**) so callers can chain or save.

## Recommended signatures

```python
from matplotlib.axes import Axes
from matplotlib.figure import Figure

def plot_waveform(ax: Axes, t, y, *, label: str | None = None) -> Axes:
    ax.plot(t, y, label=label)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (V)")
    return ax

def finalize_figure(fig: Figure) -> Figure:
    fig.align_labels()
    return fig
```

## Optional axes

- If `ax is None`, create a figure and axes; otherwise draw on the provided axes:

```python
def plot_spectrum(data, *, ax: Axes | None = None) -> tuple[Figure, Axes]:
    if ax is None:
        fig, ax = plt.subplots(layout="constrained")
    else:
        fig = ax.figure
    ax.plot(data.f, data.psd)
    ax.set_xlabel("Frequency (Hz)")
    return fig, ax
```

## Typing

- Annotate with **`Axes`** and **`Figure`** from `matplotlib.axes` / `matplotlib.figure` for static checking (project may use `ty` / pyright).

## Composition

- **Higher-level** functions call lower-level ones, always threading `ax`:

```python
def plot_panel(ax: Axes, dataset) -> Axes:
    plot_waveform(ax, dataset.t, dataset.y, label="signal")
    ax.legend()
    return ax
```

## Anti-patterns

- Relying on **`plt.gca()`** inside library code.
- Mixing **`plt.plot`** and passed **`ax`** in the same module without `plt.sca`.
- Creating a **new figure** inside a helper without returning it, leaving callers unable to save.

## Layout-aware helpers

- After adding colorbars or twin axes, call **`fig.align_labels()`** or rely on **`layout="constrained"`** so panel labels stay aligned across a row.

# Saving figures: DPI, transparency, formats, auto-increment names

## `fig.savefig` essentials

- Always save the **`Figure`** you built: `fig.savefig(path, ...)`, not `plt.savefig` in library code (acceptable in quick scripts if there is a single current figure).
- **Raster**: PNG (and TIFF if required). **Vector**: PDF or SVG for line art and typography at journal resolution.
- **DPI**: 300–600 is a common print range; match journal guidance. See [Saving to file](https://www.practicaldatascience.org/notebooks/class_5/week_1/1.4.4_saving_to_file.html).
- **Clipping**: if labels are cut off, use `bbox_inches="tight"` (optionally with `pad_inches`).
- **Transparent PNG**: `transparent=True` for slides and compositing on non-white backgrounds.

Example:

```python
fig.savefig(
    "outputs/fig01.png",
    dpi=400,
    bbox_inches="tight",
    pad_inches=0.02,
    transparent=True,
)
```

## Meeting-friendly auto-increment

Use a small helper so repeated script runs produce `plot_001.png`, `plot_002.png`, without overwriting. Persist the counter in a dotfile or JSON next to outputs.

```python
from pathlib import Path
import json

def next_plot_path(dir_path: Path, stem: str = "plot", suffix: str = ".png") -> Path:
    """Return the next sequential path ``{stem}_{n:03d}{suffix}`` under ``dir_path``."""
    dir_path.mkdir(parents=True, exist_ok=True)
    state = dir_path / ".plot_counter.json"
    n = 0
    if state.is_file():
        n = int(json.loads(state.read_text()).get("n", 0))
    n += 1
    state.write_text(json.dumps({"n": n}))
    return dir_path / f"{stem}_{n:03d}{suffix}"

path = next_plot_path(Path("outputs/meetings"))
fig.savefig(path, dpi=400, bbox_inches="tight", transparent=True)
```

Variants: timestamp-based names for parallel runs, or `git rev-parse --short` in the stem for traceability.

## Script vs notebook

- In non-interactive scripts, call **`plt.show()`** only when debugging; for batch pipelines, rely on **`savefig`** ([basic plotting note](https://www.practicaldatascience.org/notebooks/class_5/week_1/1.2.1_basic_plotting_with_matplotlib.html)).

## Supported formats

- Discover locally: `fig.canvas.get_supported_filetypes()` ([Saving to file](https://www.practicaldatascience.org/notebooks/class_5/week_1/1.4.4_saving_to_file.html)).

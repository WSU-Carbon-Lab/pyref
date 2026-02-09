[![DOI](https://zenodo.org/badge/659964712.svg)](https://doi.org/10.5281/zenodo.14758701)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/WSU-Carbon-Lab/pyref)
<h1 align="center">
    <img src="https://github.com/WSU-Carbon-Lab/pyref/assets/73567020/f4883d3b-829e-48da-9a66-df50ecf357e5" alt="Pyref logo">
    <br>
</h1>

# PyRef: X-ray Reflectivity Data Reduction

PyRef is a Python library for reducing 2D X-ray reflectivity detector images into 1D reflectivity signals. It is designed for processing experimental data from polarized resonant soft X-ray reflectivity (PRSoXR) beamlines, specifically beamline 11.0.1.2 at the Advanced Light Source (ALS).

## Features

- **Efficient Data Loading**: Rust-accelerated parallel reading of FITS files
- **Image Processing**: Automatic beam spot detection and background subtraction
- **Uncertainty Propagation**: Built-in error propagation through all calculations
- **Interactive Masking**: Tools for defining regions of interest in detector images
- **Data Organization**: Automatic grouping of data by experimental "stitches"
- **Reflectivity Fitting**: Integration with refnx for model fitting

## Installation

```bash
pip install pyref
```

For development installations, see the [development setup](#development) section.

## Quick Start

### Loading an Experiment

```python
from pathlib import Path
from pyref.loader import PrsoxrLoader

# Initialize loader with experiment directory
loader = PrsoxrLoader(Path("/path/to/experiment/data"))

# Access reflectivity data
refl_df = loader.refl
print(refl_df)
```

### Interactive Image Masking

```python
# Open interactive masking tool
loader.mask_image()

# The mask is automatically applied to all subsequent processing
# Press 'm' to save and close the mask editor
```

### Accessing Data

```python
# Reflectivity DataFrame
refl = loader.refl  # Columns: file_name, Q, r, dr

# Full metadata DataFrame
meta = loader.meta  # All images, headers, calculated columns

# Plot reflectivity data
plot = loader.plot_data()
```

### Working with Stitches

Experimental data is organized into "stitches" - separate measurement chunks where beamline parameters are adjusted to capture reflectivity across multiple orders of magnitude.

```python
# Group by stitch (file_name)
for file_name, stitch_data in loader.refl.group_by("file_name"):
    print(f"Stitch: {file_name}")
    print(f"Q range: {stitch_data['Q'].min()} to {stitch_data['Q'].max()}")
    print(f"Number of points: {len(stitch_data)}")
```

### Uncertainty Propagation

PyRef automatically propagates uncertainty through all calculations. The reflectivity DataFrame includes `dr` (uncertainty in reflectivity) calculated from counting statistics.

```python
import polars as pl
from pyref.utils import err_prop_mult, err_prop_div, weighted_mean

# Example: Propagate uncertainty through multiplication
# If you need to apply a scale factor with uncertainty
result = df.with_columns([
    err_prop_mult(
        pl.col("r"), pl.col("dr"),
        pl.lit(scale_factor), pl.lit(scale_uncertainty)
    ).alias("scaled_r_err")
])

# Weighted mean for combining overlapping points
combined = df.group_by("Q").agg([
    weighted_mean(pl.col("r"), pl.col("dr")).alias("r_mean"),
    weighted_std(pl.col("dr")).alias("r_err")
])
```

## How It Works

### Data Reduction Pipeline

1. **FITS File Reading**: Parallel reading of FITS files with Rust backend
   - Extracts header metadata (energy, theta, exposure time, etc.)
   - Loads raw CCD images
   - Calculates Q-vector from energy and theta

2. **Image Processing**: 2D to 1D reduction
   - Background subtraction (row-by-row)
   - Beam spot location (automatic or edge-detection fallback)
   - ROI (Region of Interest) analysis
   - Signal extraction: beam counts - scaled background

3. **Reflectivity Calculation**:
   - Normalization by exposure time and beam current
   - Uncertainty from Poisson counting statistics: `dr = sqrt(r)`
   - Data grouped by file_name (stitch)

4. **Data Organization**:
   - Each experimental "stitch" is identified by `file_name`
   - Stitches contain overlapping theta/Q values for scaling
   - Metadata and reflectivity data stored in separate DataFrames

### Stitching Multiple Datasets

To combine multiple reflectivity scans collected in separate stitches:

1. **Identify Overlaps**: Find overlapping Q ranges between consecutive stitches
2. **Calculate Scale Factors**: Determine multiplicative scale from overlapping region
3. **Propagate Uncertainty**: Use error propagation functions for scale factor application
4. **Combine Points**: Use weighted statistics to merge overlapping measurements

Example workflow:
```python
# This is a conceptual example - full stitching implementation may vary
stitches = loader.refl.group_by("file_name")

stitched_data = []
cumulative_scale = 1.0
cumulative_scale_err = 0.0

for i, (file_name, stitch) in enumerate(stitches):
    if i == 0:
        # First stitch is reference
        stitched_data.append(stitch)
        continue

    # Find overlap with previous stitch
    prev_stitch = stitched_data[-1]
    overlap = find_overlapping_q(prev_stitch, stitch)

    if len(overlap) > 0:
        # Calculate scale factor from overlap
        scale, scale_err = calculate_scale_factor(overlap)

        # Apply scale with uncertainty propagation
        scaled_stitch = stitch.with_columns([
            (pl.col("r") * scale).alias("r"),
            err_prop_mult(
                pl.col("r"), pl.col("dr"),
                pl.lit(scale), pl.lit(scale_err)
            ).alias("dr")
        ])

        # Combine overlapping points using weighted mean
        combined = combine_overlapping_points(prev_stitch, scaled_stitch)
        stitched_data[-1] = combined
    else:
        stitched_data.append(stitch)
```

### Uncertainty Propagation

PyRef implements standard error propagation formulas:

**Multiplication**: `σ(xy) = |xy| * sqrt((σx/x)² + (σy/y)²)`

**Division**: `σ(x/y) = |x/y| * sqrt((σx/x)² + (σy/y)²)`

**Weighted Mean**: Uses inverse-variance weighting: `w = 1/σ²`

These are implemented as Polars expressions and can be used in any DataFrame operation.

## Advanced Usage

### Custom Header Keys

```python
from pyref.types import HeaderValue

loader = PrsoxrLoader(
    directory,
    extra_keys=[
        HeaderValue.HIGHER_ORDER_SUPPRESSOR,
        HeaderValue.EXIT_SLIT_SIZE,
        # Add any custom header keys
    ]
)
```

### Direct FITS Reading

```python
from pyref.io import read_experiment, read_fits

# Read entire experiment directory
df = read_experiment("/path/to/data", headers=["Beamline Energy", "EXPOSURE"])

# Read specific FITS file
df = read_fits("/path/to/file.fits", headers=["Beamline Energy"])
```

### Exporting Data

```python
# Save as CSV files (one per stitch)
loader.write_csv()

# Save as Parquet (more efficient)
loader.write_parquet()
```

### Interactive Spot Checking

```python
# Visualize individual frames with beam spot and properties
loader.check_spot()
```

## API Reference

### PrsoxrLoader

Main class for loading and processing reflectivity data.

**Parameters**:
- `directory` (Path): Path to directory containing FITS files
- `extra_keys` (list[HeaderValue] | None): Additional header keys to extract

**Attributes**:
- `refl` (pl.DataFrame): Reflectivity DataFrame with Q, r, dr
- `meta` (pl.DataFrame): Full metadata DataFrame with all images
- `mask` (np.ndarray): Image mask for beam isolation
- `energy` (np.ndarray): Unique energy values in experiment
- `polarization` (Literal["s", "p"]): Polarization state
- `name` (str | np.ndarray): Sample name(s)
- `files` (np.ndarray): List of file names (stitches)

**Methods**:
- `mask_image()`: Open interactive mask editor
- `check_spot()`: Interactive frame viewer
- `plot_data()`: Plot reflectivity data
- `write_csv()`: Export to CSV files
- `write_parquet()`: Export to Parquet files

### Uncertainty Functions

- `err_prop_mult(lhs, lhs_err, rhs, rhs_err)`: Error propagation for multiplication
- `err_prop_div(lhs, lhs_err, rhs, rhs_err)`: Error propagation for division
- `weighted_mean(values, weights)`: Weighted mean using inverse-variance weights
- `weighted_std(weights)`: Weighted standard deviation

## Development

### Building from Source

```bash
# Clone repository
git clone https://github.com/WSU-Carbon-Lab/pyref.git
cd pyref

# Install in development mode
pip install -e .

# Or use uv
uv pip install -e .
```

### Experiment browser TUI

A terminal UI for browsing experiment datasets (sample/tag/experiment selectors and reflectivity profile table). Build and run with:

```bash
cargo run --bin pyref-tui --features tui
```

Config is stored at `$PYREF_TUI_CONFIG` if set, otherwise `$HOME/.config/pyref/tui.toml`. Use `q` to quit, Tab to move focus, and j/k or arrows to move within lists and the table.

### Project Structure

```
pyref/
├── python/pyref/          # Python package
│   ├── loader.py          # Main PrsoxrLoader class
│   ├── image.py           # Image processing functions
│   ├── masking.py         # Interactive masking tools
│   ├── io/                # I/O functions
│   ├── fitting/           # Reflectivity fitting
│   └── utils/             # Utility functions
├── src/                   # Rust backend
│   ├── loader.rs          # FITS file reading
│   ├── io.rs              # Image processing
│   └── lib.rs             # Polars plugins
└── tests/                 # Test suite
```

## Contributing

Contributions are welcome! Please see the contributing guidelines for details.

## Citation

If you use PyRef in your research, please cite:

```
[Add citation information here]
```

## License

[Add license information here]

## Acknowledgments

Developed for processing data from beamline 11.0.1.2 at the Advanced Light Source, Lawrence Berkeley National Laboratory.

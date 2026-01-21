# PyRef Architecture and Data Flow

## Overview

PyRef is a library for reducing 2D X-ray reflectivity detector images into 1D reflectivity signals. The library handles experimental data collected in "stitches" - separate measurement chunks where beamline configuration parameters (higher-order suppressor, exit slits, exposure times) are adjusted to capture reflectivity across multiple orders of magnitude.

## Core Components

### 1. Data Loading (`src/loader.rs`, `python/pyref/io/readers.py`)

The Rust backend (`src/loader.rs`) handles parallel reading of FITS files:
- Reads FITS file headers and image data
- Processes images to extract beam spot locations
- Calculates simple reflectivity using ROI (Region of Interest) analysis
- Combines multiple FITS files into a single Polars DataFrame
- Adds calculated columns (Q-vector from energy and theta)

Key functions:
- `read_experiment()`: Reads all FITS files in a directory
- `read_experiment_pattern()`: Reads FITS files matching a pattern
- `read_multiple_fits()`: Reads specific FITS files
- `combine_dataframes_with_alignment()`: Merges DataFrames with schema alignment

### 2. Image Processing (`python/pyref/image.py`, `src/io.rs`)

Image processing reduces 2D detector images to reflectivity values:

**Rust implementation** (`src/io.rs`):
- `subtract_background()`: Row-by-row background subtraction
- `simple_reflectivity()`: Calculates beam signal vs background using ROI
- `process_image()`: Main image processing pipeline

**Python implementation** (`python/pyref/image.py`):
- `reduce_data()`: Full image reduction pipeline (dezinger, filtering, masking)
- `locate_beam()`: Locates beam spot in processed image
- `reduction()`: Calculates reflectivity from masked image and beam spot

Uncertainty in reflectivity originates from:
- Counting statistics in beam signal
- Background subtraction uncertainty
- Exposure time and beam current normalization

### 3. Masking (`python/pyref/masking.py`)

- `InteractiveImageMasker`: Allows interactive rectangular masking of images
- `ImageSeries.mask()`: Automatic mask generation based on CDF of mean image
- Mask defines which pixels contribute to reflectivity calculation

### 4. Main Loader (`python/pyref/loader.py`)

`PrsoxrLoader` orchestrates the data reduction workflow:

- Loads experiment data via `read_experiment()`
- Processes images using the mask
- Creates reflectivity DataFrame with columns: `file_name`, `Q`, `r`, `dr`
- Groups data by `file_name` (each file_name represents a stitch)
- Calculates uncertainty: `dr = sqrt(r)` (Poisson counting statistics)

Key properties:
- `refl`: DataFrame containing reflectivity data
- `meta`: Full metadata DataFrame with images
- `mask`: Image mask for beam isolation

### 5. Uncertainty Propagation (`src/lib.rs`, `python/pyref/utils/__init__.py`)

Uncertainty propagation implemented as Polars plugins:

**Rust functions** (`src/lib.rs`):
- `err_prop_mult()`: Error propagation for multiplication
  - Formula: `σ(xy) = |xy| * sqrt((σx/x)² + (σy/y)²)`
- `err_prop_div()`: Error propagation for division
  - Formula: `σ(x/y) = |x/y| * sqrt((σx/x)² + (σy/y)²)`
- `weighted_mean()`: Weighted average using inverse variance weights
- `weighted_std()`: Weighted standard deviation

**Python interface** (`python/pyref/utils/__init__.py`):
- Exposes Rust functions as Polars expressions
- Used throughout the data processing pipeline

### 6. Data Stitching (Conceptual)

While explicit stitching code is not fully implemented, the infrastructure supports it:

**Current capabilities:**
- Data grouped by `file_name` (each stitch is a separate file_name)
- Uncertainty propagation functions ready for scale factor calculations
- Weighted statistics for combining overlapping points
- `OverlapError` exception defined for overlap validation

**Stitching workflow (as designed):**
1. Identify overlapping theta/Q values between consecutive stitches
2. Calculate multiplicative scale factor from overlapping region
3. Apply scale factor to subsequent stitch
4. Propagate uncertainty through scaling operation
5. Combine overlapping points using weighted mean
6. Continue recursively for all stitches

**Key considerations:**
- Scale factors determined from overlapping measurements
- Uncertainty propagation through multiplication (scale factor application)
- Weighted combination preserves statistical information
- Must handle cases where overlap is insufficient

### 7. Reflectivity Fitting (`python/pyref/fitting/`)

The fitting module provides:
- `XrayReflectDataset`: Dataset class for reflectivity data with polarization handling
- `ReflectModel`: Model for fitting reflectivity curves
- `Structure`: Layer structure definitions
- Uniaxial and birefringent reflectivity calculations

## Data Flow

```
FITS Files
    ↓
[Rust IO Layer] - Parallel reading, image processing, initial reflectivity
    ↓
Polars DataFrame (meta) - All images, metadata, calculated Q
    ↓
[Python Loader] - Masking, grouping by file_name
    ↓
Reflectivity DataFrame (refl) - Q, r, dr grouped by stitch
    ↓
[Stitching] - Scale factor calculation, uncertainty propagation (conceptual)
    ↓
Combined Reflectivity Dataset - Complete R vs Q curve
    ↓
[Fitting Module] - Model fitting with refnx
```

## Key Design Decisions

1. **Rust backend**: Critical path operations (FITS reading, image processing) implemented in Rust for performance
2. **Polars DataFrame**: Efficient columnar operations, lazy evaluation, built-in parallelization
3. **Uncertainty tracking**: Uncertainty propagated at every step, not added post-hoc
4. **Grouped processing**: Data naturally organized by file_name (stitch) for easy stitching
5. **Plugin architecture**: Uncertainty functions as Polars plugins for seamless integration

## Usage Patterns

### Basic Workflow
1. Initialize loader: `loader = PrsoxrLoader(directory)`
2. Optionally mask images: `loader.mask_image()`
3. Access reflectivity: `refl_df = loader.refl`
4. Process by stitch: `loader.refl.group_by("file_name")`

### Uncertainty Handling
- Use `pyref.utils.err_prop_mult()` and `err_prop_div()` for calculations
- Use `weighted_mean()` and `weighted_std()` for combining overlapping points
- Uncertainty propagates automatically through Polars expressions

### Stitching (Future Implementation)
- Identify overlapping Q ranges between consecutive file_name groups
- Calculate scale factors from weighted mean of overlapping points
- Apply scale factors with proper uncertainty propagation
- Combine using weighted statistics

# Ufuncs, reductions, `where`, NaNs

## Ufuncs

- **Unary / binary** elementwise: `np.sin`, `np.exp`, `np.maximum`, `np.logical_and`, etc.
- **`out=`** avoids allocations when you reuse buffers (advanced; ensure **no overlap** unless ufunc supports it).
- **Casting rules** follow ufunc **signature**; mixed dtypes **promote** per NumPy rules—verify for **int** vs **float** mixes.

Official: [Universal functions](https://numpy.org/doc/stable/reference/ufuncs.html).

## Reductions

- **`axis=None`**: whole array; **`axis=tuple`**: reduce multiple axes at once.
- **`keepdims=True`**: preserves reduced axes as length 1 for **broadcasting** with the original shape.
- **`ddof`** for variance/std (Bessel correction).

## NaNs

- **`np.nanmean`**, **`np.nansum`**, **`np.nanstd`**, etc. **ignore** NaNs; plain **`mean`** propagates NaN.
- Sorting: **`np.nanargmin`** / **`nanargmax`** or **`np.ma`** when masks are first-class.

## `np.where`

- **`np.where(cond, x, y)`** selects elementwise; **`np.where(cond)`** returns **indices** (tuple of 1D arrays).
- For **assignment**, **`np.where`** on the RHS is not the same as masked write; use **`np.putmask`** or boolean slice assignment when appropriate.

## Sorting and searching

- **`np.sort`**, **`argsort`**, **`partition`**, **`searchsorted`** for monotonic bins (histograms, digitize).

## Reproducibility note

- **`sum`** on **float** may reorder for SIMD; bitwise-identical sums across machines are **not** guaranteed unless you control order (e.g. **`math.fsum`** on Python iterables, or **`np.add.reduce`** with fixed association in specialized cases). For science reporting, document **stability** expectations; **`python-reviewer`** flags when this matters.

## `einsum`-light

- Many **axis reductions** are clearer as **`einsum`**; see [reference-linalg-einsum.md](reference-linalg-einsum.md).

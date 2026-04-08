# Interop, pandas, and performance

## pandas and Polars

- **pandas**: **`Series.values` / `to_numpy()`**, **`DataFrame.to_numpy()`**—watch **copy** vs **view** and **dtype=object** columns.
- **Polars**: convert via **`to_numpy()`** on expressions/Series; prefer staying in Polars for query-heavy pipelines per the Python spec.
- Establish a **single owner** of labels (index/columns) at module boundaries; do not duplicate metadata in raw ndarrays without documentation.
- Table-level patterns: **dataframes** skill.

## SciPy, C, CUDA

- **SciPy** routines accept **`array_like`**; pass **`np.asarray`** with **`dtype`** when you need a contiguous buffer.
- **ctypes** / **cffi**: **`array.ctypes`**, **`__array_interface__`**; enforce **contiguity** (`C` or `F`) before passing pointers.

## Vectorization discipline

- Prefer **ufuncs** and **broadcasting** over Python **`for`** loops over rows when **N** is large.
- When loops remain, **Numba**, **Cython**, or moving hot paths to **Rust/PyO3** are separate project decisions—do not introduce heavy deps without **`uv add`** and team agreement.

## Memory

- **`dtype` downcast** when safe; **`in-place`** ufuncs only when readability and aliasing are controlled.
- **`del`** large temporaries in tight notebooks is cosmetic; **streaming** or **chunking** fixes real pressure.

## Threads and BLAS

- Underlying **BLAS/OpenMP** may use threads; nested **joblib** / process pools can **oversubscribe** CPU. Set **`OMP_NUM_THREADS`** / vendor vars in HPC scripts when jobs are parallel at a higher level.

## Testing

- **Parametrize** shapes and dtypes; assert **`out.shape`**, **`dtype`**, and known **analytic** cases for numerics (**`python-reviewer`**).

## Plotting

- Pass **`np.asarray`** to **matplotlib** when consuming duck arrays; use **`matplotlib-scientific`** skill for figure polish.

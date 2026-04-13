---
author: dotagents
name: numpy-scientific
description: NumPy for scientific Python: dtypes and casting, creation, reshaping, broadcasting, indexing vs views, ufuncs and reductions, linear algebra and einsum, random Generator, I/O, structured arrays, performance and pandas boundaries. Use when writing or reviewing ndarray code. Triggers on numpy, ndarray, broadcasting, dtype, ufunc, einsum, vectorization.
---

# NumPy (scientific computing)

## Quick start

1. **Shape and dtype first**: decide **`shape`**, **`dtype`**, and memory **`order`** (`C` vs `F`) before filling arrays; avoid silent **`astype`** widening. See [reference-arrays-dtypes.md](references/reference-arrays-dtypes.md).
2. **Vectorize**: prefer **ufuncs** and **whole-array** expressions over Python loops on large data; push loops to compiled layers when needed. See [reference-ufuncs-reductions.md](references/reference-ufuncs-reductions.md) and [reference-interop-performance.md](references/reference-interop-performance.md).
3. **Know view vs copy**: basic slicing is usually a **view**; fancy indexing is a **copy**; **`reshape`** may view or copy. Mutations alias bugs are common. See [reference-indexing-views.md](references/reference-indexing-views.md).
4. **Broadcasting**: align trailing dimensions; use **`None` / `newaxis`** and **`np.broadcast_to`** deliberately. See [reference-broadcasting-shape.md](references/reference-broadcasting-shape.md).
5. **Reductions**: always pass **`axis`** and use **`keepdims=True`** when broadcasting the result back; prefer **`np.nanmean`** etc. when NaNs appear. Mind **float summation order** for reproducibility. See [reference-ufuncs-reductions.md](references/reference-ufuncs-reductions.md).
6. **Random**: use **`np.random.Generator`** (`PCG64`), not legacy global `RandomState`, for reproducible science. See [reference-random-io-structured.md](references/reference-random-io-structured.md).

## Stack synergy

| Resource | Role |
|----------|------|
| **general-python** | uv, ruff, ty, vectorization policy, **python-reviewer** numerics footguns |
| **dataframes** | Table handoff at boundaries |
| **numpy-docstrings** | Public API docstrings for array-heavy modules |
| **matplotlib-scientific** | Plotting array results |
| **lab-instrumentation** | Binary waveform I/O, instrument-side buffers |
| **python-reviewer** | dtype/shape/reproducibility in review |

## Reference index

| Topic | File |
|--------|------|
| Creation, `dtype`, casting, `order`, strides | [reference-arrays-dtypes.md](references/reference-arrays-dtypes.md) |
| Slicing, fancy index, views, copies, `reshape` | [reference-indexing-views.md](references/reference-indexing-views.md) |
| Broadcasting, stacking, meshgrid / ogrid | [reference-broadcasting-shape.md](references/reference-broadcasting-shape.md) |
| Ufuncs, reductions, `axis`, `where`, NaNs | [reference-ufuncs-reductions.md](references/reference-ufuncs-reductions.md) |
| `linalg`, `@`, `einsum`, `tensordot` | [reference-linalg-einsum.md](references/reference-linalg-einsum.md) |
| `Generator`, `.npy` / `.npz`, structured dtypes | [reference-random-io-structured.md](references/reference-random-io-structured.md) |
| Pandas/Polars handoff, memory, threads | [reference-interop-performance.md](references/reference-interop-performance.md) |

## Official documentation

- [NumPy user guide](https://numpy.org/doc/stable/user/index.html)
- [Absolute basics](https://numpy.org/doc/stable/user/absolute_beginners.html)
- [Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)
- [Indexing](https://numpy.org/doc/stable/user/basics.indexing.html)
- [Copies and views](https://numpy.org/doc/stable/user/basics.copies.html)
- [NumPy reference](https://numpy.org/doc/stable/reference/index.html)

## Dependency

Add or upgrade with **`uv add numpy`** per project policy; do not hand-edit pins in `pyproject.toml`.

---
author: dotagents
name: dataframes
description: Pandas and Polars for tabular data: when to use which, indexing and dtypes, joins and reshaping, lazy Polars, I/O and Parquet, nulls, Arrow interop, and performance. Use for DataFrame/Series/LazyFrame code. Triggers on pandas, polars, DataFrame, LazyFrame, parquet, groupby, merge, join.
---

# Dataframes (pandas + Polars)

## Quick start

1. **Pick one engine per pipeline**: **pandas** when **index semantics**, **mixed columns**, and **ecosystem** (statsmodels, sklearn I/O) matter; **Polars** when **large scans**, **lazy plans**, and **expression** clarity win. See [reference-when-which.md](references/reference-when-which.md).
2. **pandas**: be explicit with **`loc` / `iloc`**, **`assign`**, and **dtypes**; understand **views vs copies** and **Copy-on-Write** (2.x). See [reference-pandas-core.md](references/reference-pandas-core.md).
3. **Polars**: default to **`scan_*` + `lazy()`** for big data; build with **`select` / `with_columns` / `filter`** and **`pl.col`**; **`collect`** at the boundary. See [reference-polars.md](references/reference-polars.md).
4. **Joins and groups**: validate **row counts** and **duplicates**; name **suffixes**; prefer **Polars** for parallel group-by on large tables when the project already uses it. See [reference-pandas-group-join.md](references/reference-pandas-group-join.md) and [reference-polars.md](references/reference-polars.md).
5. **I/O**: **Parquet** (often **PyArrow**) for analytics interchange; **`uv add pyarrow`** when the stack needs it. See [reference-io-dtypes-nulls.md](references/reference-io-dtypes-nulls.md).
6. **Handoff**: convert at **module boundaries** with **`from_pandas`** / **`to_pandas`** or **Arrow**; avoid ping-pong in inner loops. See [reference-interop-performance.md](references/reference-interop-performance.md).

## Stack synergy

| Resource | Role |
|----------|------|
| **Python spec** | Explicit pandas index/columns; lazy Polars for heavy queries |
| **numpy-scientific** | `to_numpy()`, dtypes, contiguous buffers |
| **matplotlib-scientific** | `df.plot(ax=ax)` and Polars `.to_pandas()` for plotting when needed |
| **numpy-docstrings** | Public API docstrings for DataFrame-returning functions |
| **general-python** | uv, ty, **python-reviewer** |

## Reference index

| Topic | File |
|--------|------|
| pandas vs Polars, boundaries | [reference-when-which.md](references/reference-when-which.md) |
| Index, `loc`/`iloc`, dtypes, COW, assignment | [reference-pandas-core.md](references/reference-pandas-core.md) |
| GroupBy, merge/join, concat, pivot | [reference-pandas-group-join.md](references/reference-pandas-group-join.md) |
| LazyFrame, expressions, group, join | [reference-polars.md](references/reference-polars.md) |
| CSV/Parquet, nulls, dtypes, Arrow | [reference-io-dtypes-nulls.md](references/reference-io-dtypes-nulls.md) |
| Conversion, streaming, typing | [reference-interop-performance.md](references/reference-interop-performance.md) |

## Official documentation

- [pandas user guide](https://pandas.pydata.org/docs/user_guide/index.html)
- [Polars user guide](https://docs.pola.rs/user-guide/)
- [Polars API](https://docs.pola.rs/api/python/stable/reference/index.html)

## Dependencies

Use **`uv add pandas`**, **`uv add polars`**, **`uv add pyarrow`** as needed; do not hand-edit version pins in `pyproject.toml`.

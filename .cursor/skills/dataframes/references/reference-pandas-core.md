# pandas: index, selection, dtypes, mutation

Official: [Indexing and selecting data](https://pandas.pydata.org/docs/user_guide/indexing.html), [Copy-on-Write](https://pandas.pydata.org/docs/user_guide/copy_on_write.html).

## Index and columns

- **`Index`** carries **labels** and optional **`name`**; **`MultiIndex`** for hierarchical keys.
- **`reset_index` / `set_index`**: make columns into index or the reverse when modeling needs change.

## `loc`, `iloc`, `[]`

- **`loc`**: label-based (slice end **inclusive** on labels).
- **`iloc`**: integer position (Python slice end **exclusive**).
- **`df[col]`** for a single column **Series**; **`df[[col]]`** for one-column **DataFrame**.
- Avoid **chained indexing** (`df[a][b] = x`); use **`loc`** or **`assign`**.

## Copy-on-Write (pandas 2+)

- With **CoW** enabled (recommended in modern defaults), many chained reads are safer; still treat **writes** as requiring explicit **`loc`** assignment.
- **`copy()`** when you must guarantee an independent buffer before mutation.

## dtypes

- **Nullable**: **`Int64`**, **`string`**, **`boolean`** (capitalized extension dtypes) vs **`object`** for messy text.
- **`astype`** can widen or lose precision; prefer **`pd.to_numeric(..., errors="coerce")`** for ingestion.
- **Categorical** for low-cardinality strings saves memory and speeds **groupby**.

## Assignment

- **`df.loc[row_indexer, col] = values`**; align with **index** on the RHS when assigning Series.
- **`pd.concat`** along axis for new rows/columns instead of repeated **`append`** in hot paths.

## Series

- **Alignment** on index in **arithmetic** with other Series/DataFrame—often a feature, sometimes a bug; **`reset_index`** or **`.values`** when you intend **positional** math.

## Performance hints

- **Vectorized** ops on columns; **`apply`** row-wise is slow at scale—reshape or use **Polars/NumPy**.

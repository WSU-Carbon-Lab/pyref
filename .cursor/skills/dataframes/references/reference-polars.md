# Polars: LazyFrame, expressions, group, join

Official: [User guide](https://docs.pola.rs/user-guide/), [Expressions](https://docs.pola.rs/user-guide/expressions/).

## Eager vs lazy

- **Eager** `DataFrame`: small/medium data, interactive work.
- **Lazy** `LazyFrame`: **`pl.scan_parquet`**, **`pl.scan_csv`**, **`df.lazy()`**; chain transforms, then **`collect()`** (or **`collect(streaming=True)`** when enabled for your version).

## Expressions

- **`pl.col("a")`**, **`pl.col("^pat.*$")`**, **`pl.all()`**.
- **`.alias`**, **`.cast`**, **`.fill_null`**, **`.replace`**, string and datetime **`.str.*` / `.dt.*`** namespaces.
- **Conditional**: **`pl.when(cond).then(x).otherwise(y)`**.
- **Horizontal**: **`pl.sum_horizontal`**, **`pl.concat_list`** for list columns.

## select / with_columns / filter

- **`select`** chooses/reorders; **`with_columns`** adds or replaces by name; **`filter`** is row predicate.
- **`with_row_index`** when you need a stable row id column.

## Joins

- **`join`**, **`join_asof`** for temporal alignment; check **how** (`inner`, `left`, `semi`, `anti`).
- **Suffix** for name clashes; **validate** row counts like in SQL (assert in code).

## Group by

- **`group_by("k").agg(pl.col("x").mean())`**; multiple metrics in one **`agg`**.
- **Dynamic groups** for time windows: **`group_by_dynamic`** when applicable.

## Sort and distinct

- **`sort`**, **`unique`**, **`n_unique`**; **`is_duplicated`** for QA.

## Strings and categoricals

- **String** dtype; **Categorical** for low-cardinality columns to save memory in **lazy** plans.

## Errors

- Polars tends to **fail fast** on schema issues; fix **dtypes** at **scan** or first **`with_columns`**.

## pandas interop

- **`df.to_pandas()`**, **`pl.from_pandas(df)`**; see [reference-interop-performance.md](reference-interop-performance.md).

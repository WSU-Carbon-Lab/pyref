# When to use pandas vs Polars

Aligned with the project **Python** spec: explicit **index/column** semantics in pandas, **lazy** Polars for heavy queries when the project standardizes on it.

## Prefer **pandas**

- **Row labels** are part of the model (time series index alignment, `reindex`, `asfreq`).
- **Wide** mix of dtypes in one table with frequent **per-column** Python logic.
- **Downstream** expects pandas (`sklearn` with minimal friction, statsmodels, many tutorials).
- **Incremental** cell updates and **in-place**-style workflows are entrenched (still prefer explicit assignment).

## Prefer **Polars**

- **Large** files: **`scan_parquet` / `scan_csv`** and **lazy** plans push work to the engine.
- **Complex column expressions** without index alignment surprises: **`pl.col`** pipelines read linearly.
- **Parallel** group-by and joins on **big** tables (hardware-dependent; profile).
- **Stricter** null model (`null`) and **dtype** consistency across the query.

## One boundary per layer

- Do not **alternate** libraries inside a tight inner loop; **convert once** at imports/exports of a stage (ETL end, modeling start, plot start).
- Document the **owner** of the index: pandas keeps it on the frame; Polars is mostly **column-first** (row index is positional unless you add a column).

## Time series

- **pandas**: **`DatetimeIndex`**, **`resample`**, **`rolling`**, **`tz`** handling are mature.
- **Polars**: temporal **expressions** and **`group_by_dynamic`** fit large **event** tables; verify **timezone** rules for your version.

## Testing

- **Assert** shapes and **key uniqueness** after joins in both libraries.
- **`python-reviewer`** for merge footguns and silent **`NaN`** propagation.

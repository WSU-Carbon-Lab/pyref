# I/O, dtypes, and nulls (both libraries)

## Reading and writing

### pandas

- **`read_csv`**, **`read_parquet`**, **`read_feather`**, **`read_sql`**; **`to_parquet`**, etc.
- **`dtype`**, **`parse_dates`**, **`usecols`** reduce memory on ingest.
- Engine **`pyarrow`** or **`fastparquet`** for Parquet—project should pick one consistently.

### Polars

- **`read_csv` / `read_parquet`** (eager) vs **`scan_csv` / `scan_parquet`** (lazy).
- **`infer_schema_length`**, **`try_parse_dates`**, **`columns`** to trim work.

## Parquet and Arrow

- **Parquet** preserves **schema** and compresses well for analytics pipelines.
- **`uv add pyarrow`** when either stack reads/writes Parquet heavily or shares **Arrow** buffers.

## Nulls and NA

- **pandas**: **`NaN`** for floats; **`pd.NA`** for nullable dtypes; **`NaT`** for datetimes—**boolean reductions** can be tri-state.
- **Polars**: **`null`** unified; predicates use **`is_null()`**, **`is_not_null()`**, **`fill_null`**, **`drop_nulls`**.
- **Joins**: null keys usually **do not match**; document behavior for outer joins.

## Strings

- pandas **`string`** dtype vs **`object`**; Polars **`String`** (Utf8 in older docs)—check installed Polars version in migration notes.

## Time zones

- Store **UTC** internally where possible; localize/convert explicitly for display and reporting.

## Excel and odd formats

- Prefer **CSV/Parquet** in pipelines; **Excel** last-mile only; watch **sheet** and **type** inference.

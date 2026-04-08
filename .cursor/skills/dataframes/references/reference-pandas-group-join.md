# pandas: groupby, merge, reshape

Official: [Group by](https://pandas.pydata.org/docs/user_guide/groupby.html), [Merge, join, concatenate](https://pandas.pydata.org/docs/user_guide/merging.html).

## GroupBy

- **`groupby(keys, as_index=...)`** then **`.agg`**, **`.transform`**, **`.filter`**.
- **`as_index=False`** keeps grouping columns as columns.
- **Named aggregation**: **`.agg(mean_x=("x", "mean"))`** for clear output names.
- **Watch**: **NA** in keys drops groups unless **`dropna=False`** (version-dependent defaults—check docs).

## Merge and join

- **`merge`**: SQL-style on **columns**; **`how`** (`inner`, `left`, `right`, `outer`); **`on`**, **`left_on`/`right_on`**; **`suffixes`** for overlapping names.
- **`join`**: index-based; easy to misuse if indexes are not unique—validate **`validate`** (`"one_to_one"`, etc.) when available.
- **Row explosion** from duplicate keys: assert **uniqueness** or **deduplicate** before merge when unintended.

## Concat

- **`pd.concat`** along **`axis=0`** (stack) or **`axis=1`** (side by side); **`keys`** for MultiIndex source labels.

## Reshape

- **`pivot`**, **`pivot_table`**, **`melt`**, **`stack`/`unstack`** for wide ↔ long.
- **`crosstab`** for counts with optional normalization.

## Sort and rank

- **`sort_values`**, **`sort_index`**; **`rank`** with **`method`** for ties.

## Validation

- **`merge`** + **`indicator=True`** to audit **left_only / right_only / both**.
- **`assert_frame_equal`** in tests with **`check_dtype`** toggled as needed.

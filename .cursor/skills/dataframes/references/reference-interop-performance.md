# Interop, NumPy, plotting, performance

## pandas ↔ Polars

- **`pl.from_pandas(df)`** / **`df.to_pandas()`**; cost is **non-trivial** on huge frames—do **once** per stage.
- **Arrow**: **`df.to_arrow()`** / **`pl.DataFrame(arrow_table)`** when zero-copy paths exist (versions and dtypes must align).

## NumPy

- **pandas**: **`Series.to_numpy()`**, **`DataFrame.to_numpy()`**; **`copy`** parameter matters; **nullable** dtypes may yield **object** or **masked** outputs—check.
- **Polars**: **`to_numpy()`** on Series; DataFrame to NumPy often goes through **column stack** or **pandas**—prefer **expressions** inside Polars.

## Matplotlib

- **pandas**: **`df.plot(ax=ax, kind=...)`** per **`matplotlib-scientific`** skill.
- **Polars**: **`to_pandas()`** for quick plots or export **CSV** to plotting tools; native plotting ecosystem is thinner.

## Memory and chunks

- **pandas**: **`read_csv(chunksize=...)`** for bounded memory; **Polars lazy** + **sink_parquet** for out-of-core style workflows (see current Polars docs for **sink** APIs).

## Typing

- **pandas** stubs: **`pd.DataFrame`**, **`Series`**; **Polars** ships types for **`DataFrame`**, **`LazyFrame`**, **`Expr`**—run **`ty check`** on public APIs that return frames.

## Parallelism

- Polars uses **Rayon**-style parallelism internally; avoid **nested** parallel Python over Polars in the same process without care.
- pandas **releases GIL** in some ops but not all; **vectorized** column ops beat Python loops.

## Reproducibility

- **Sort** before comparisons; **seed** any sampling; document **Polars** and **pandas** **versions** in science appendices when results must replay exactly.

# Scientific and lab defaults

From the project **Python** spec (general guidelines and style).

## Numerics

- Prefer **vectorized** NumPy / SciPy (or array libraries the project uses) over Python loops on **large** arrays.
- Be explicit about **shapes**, **dtypes**, and **missing data**; watch silent **dtype widening** and **unstable reduction order** when it affects science or reproducibility.
- **`python-reviewer`** is the right agent for numerics footguns in review.
- Detailed NumPy patterns: **numpy-scientific** skill.

## Tables and time series

- **pandas**: explicit **index/column** semantics; know **views vs copies** for chained assignment.
- **polars**: prefer **lazy** frames when the codebase standardizes on Polars for heavy queries.

Pick one table stack per module or project layer; do not mix idioms in one pipeline without a boundary.

- Combined guidance: **dataframes** skill.

## Instruments (PyVISA, sockets, lab I/O)

- Separate **resource open/close and configuration** from **command strings** and parsing.
- Use **context managers** or explicit lifecycle so handles are not leaked across tests.
- Full patterns: **lab-instrumentation** skill (PyVISA sessions, sockets vs VISA, HALs, validation, PDF manuals, offline tests).

## Plotting

- Use the **`matplotlib-scientific`** skill for figure quality; keep analysis modules free of ad hoc `plt` state when a library API is expected.

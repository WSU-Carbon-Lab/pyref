# Testing and documentation

From the project **Python** spec and **Python Cursor rule**.

## pytest

- Install in **dev** group: **`uv add pytest --dev`** (or project group).
- Run: **`uv run pytest`**; narrow with path or `-k` as needed.
- Prefer **fast, deterministic** unit tests; isolate **I/O** and **flaky timing** in markers or separate jobs when the team agrees.

## Docstrings

- **NumPy style** on **public** APIs: Parameters, Returns, Raises, Examples when they clarify non-obvious contracts.
- Keep **implementations** readable **without** step-by-step narrative comments (per Python spec / rule).
- Section-by-section guide: **numpy-docstrings** skill.

## What to test

- **Pure logic**: parametrized cases and edge values.
- **Bug fixes**: regression test that fails before the fix.
- **Numerics** (scientific code): dtype/shape expectations and stable reductions when the paper or spec demands it (**python-reviewer**).

## Imports and package layout

- Tests should import the **installed package** (`src` layout) so CI matches user installs.

## Review

- Before merge, use **`python-reviewer`** for uv + typing + tests + numerics in one pass.

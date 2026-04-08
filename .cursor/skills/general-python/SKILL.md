---
author: dotagents
name: general-python
description: Python stack hub: uv/ruff/ty workflows, builtins and collections, functions and classes, dataclasses, typing boundaries, pytest and NumPy-style docs. Use for any Python task in repos that follow this stack; pairs with python-reviewer, python-types, and python-refactor. Related skills: numpy-scientific, dataframes, numpy-docstrings, matplotlib-scientific, lab-instrumentation. Triggers on Python, uv, ruff, ty, pytest, dataclass, typing.
---

# General Python

This skill is the **hub** for conventions that come from the project **Python** section (often merged into **AGENTS.md**), the **Python Cursor rule** on `**/*.py`, and the **python-reviewer** / **python-types** / **python-refactor** subagents. Load topic files below instead of drifting from project defaults.

## Synergy map

| Source | Role |
|--------|------|
| **Python spec** (AGENTS.md block) | Canonical tooling (uv, ruff, ty), style, pytest, numerics, tables, instruments |
| **Python rule** | Globs `**/*.py`: 3.12+, uv, NumPy-style public docstrings, pyvisa split |
| **python-reviewer** | Post-change review: uv hygiene, typing, numerics footguns, tests |
| **python-types** | Deep typing for Astral **ty**, PEP 695, exhaustive `match` |
| **python-refactor** | Structure: tuples vs dataclasses, composition vs inheritance, size |
| **matplotlib-scientific** | Figures only; keep plotting out of this hub |
| **numpy-scientific** | `ndarray` design: dtypes, views, broadcasting, ufuncs, `linalg`, I/O, random |
| **dataframes** | pandas + Polars: tables, lazy frames, joins, I/O, interop |
| **numpy-docstrings** | numpydoc sections for public API docstrings |
| **lab-instrumentation** | PyVISA, sockets, HALs, validation, PDF datasheets, instrument tests |

## Quick decisions

1. **Dependencies**: `uv add` / `uv remove` / `uv sync` / `uv run`. Never hand-edit version pins in `pyproject.toml`. See [reference-tooling.md](references/reference-tooling.md).
2. **Quality gate**: `ruff check` (and format per project config), then **`ty check`** on touched code. See [reference-tooling.md](references/reference-tooling.md).
3. **Data**: prefer the right **builtin or collections ABC** before a custom class; use **dataclasses** for labeled bundles of data. See [reference-data-structures.md](references/reference-data-structures.md) and [reference-classes-dataclasses.md](references/reference-classes-dataclasses.md).
4. **APIs**: small **pure** core, **I/O at edges**, explicit types on public boundaries. See [reference-functions.md](references/reference-functions.md) and [reference-types-and-agents.md](references/reference-types-and-agents.md).
5. **Tests and docs**: `uv run pytest`; **NumPy-style docstrings** on public APIs. See [reference-testing-docs.md](references/reference-testing-docs.md).

## Reference index

| Topic | File |
|--------|------|
| uv, ruff, ty, layout, CI-style verification | [reference-tooling.md](references/reference-tooling.md) |
| list, dict, tuple, set, comprehensions, `collections` | [reference-data-structures.md](references/reference-data-structures.md) |
| Functions: purity, parameters, errors, resources | [reference-functions.md](references/reference-functions.md) |
| Classes, dataclasses, slots, immutability | [reference-classes-dataclasses.md](references/reference-classes-dataclasses.md) |
| Types, ty, when to delegate to **python-types** | [reference-types-and-agents.md](references/reference-types-and-agents.md) |
| pytest, docstrings, readability | [reference-testing-docs.md](references/reference-testing-docs.md) |
| Vectorized numerics, tables, instruments, plotting boundary | [reference-scientific-numerics.md](references/reference-scientific-numerics.md) |

## External anchors

- [uv](https://docs.astral.sh/uv/latest/), [Ruff](https://docs.astral.sh/ruff/), [ty](https://docs.astral.sh/ty/) (Astral)
- [PEP 8](https://peps.python.org/pep-0008/) style surface; project **ruff** rules are authoritative
- [Python data model](https://docs.python.org/3/reference/datamodel.html) for dunder semantics

## Modern practice snapshot

Reproducible projects combine **locked deps**, **fast lint/format**, and **static typing** on API surfaces; **src layout** avoids import ambiguity; automate checks in CI or pre-commit. See [reference-tooling.md](references/reference-tooling.md).

# Tooling: uv, Ruff, ty, project shape

Aligned with the project **Python** spec (often in **AGENTS.md**) and the **Python Cursor rule**.

## uv

- **Add / remove**: `uv add <pkg>`, `uv add --dev <pkg>`, `uv add --group <name> <pkg>`, `uv remove <pkg>`, `uv add <pkg> --upgrade`.
- **Environment**: `uv sync` after lock or clone.
- **Run**: `uv run python …`, `uv run pytest`, `uv run ruff check .`, `uv run ty check`.
- **Do not** edit dependency version pins in `pyproject.toml` by hand; use **`uv add`** so the lockfile stays truthful.
- Docs: [uv](https://docs.astral.sh/uv/latest/).

## Ruff

- **Lint**: `ruff check` (or `uv run ruff check`). Fix what the project enables (E/F/I/B/UP/… per `pyproject.toml`).
- **Format**: `ruff format` when the repo uses Ruff as formatter.
- Treat **project config as law** over generic PEP 8 advice.
- Docs: [Ruff](https://docs.astral.sh/ruff/).

## ty

- **Typecheck**: `ty check` (or `uv run ty check`) on the project or paths you changed.
- Ensure **ty** and **ruff** stay in the **dev** dependency group per spec.
- Docs: [ty](https://docs.astral.sh/ty/).

## Typical verification sequence

1. `ruff check` (and format if applicable)
2. `ty check`
3. `uv run pytest`

Order may vary; fix **ruff** before **ty** when both report the same line.

## Project layout (greenfield or refactors)

- Prefer **`src/<package>/`** layout so tests and tooling import the package the same way users do (avoids accidental imports from repo root). Pair with **`pyproject.toml`** (PEP 621).
- Keep **lockfile** (`uv.lock`) committed when the team relies on reproducible installs.

## Automation

- **pre-commit** or CI running ruff + ty + pytest catches drift early (common 2024–2025 practice; align with team policy).

## Delegation

- Broad **review** after substantive edits: **`python-reviewer`** agent.
- Heavy **annotation** passes: **`python-types`** agent.

# Functions: design and boundaries

Aligned with **python-refactor** (functional core, imperative shell) and the project **Python** spec (clear control flow, minimal narrative comments).

## Size and shape

- One **obvious responsibility** per function; long procedures become **named steps** or small private helpers.
- Prefer **pure** functions in the core (deterministic from inputs; no hidden globals). Put **I/O**, **time**, **randomness**, and **global config** at the edges.

## Signatures

- **Explicit parameters** over catch-all `*args` unless wrapping or forwarding.
- **`**kwargs`**: prefer **`Unpack[TypedDict]`** (PEP 692) for structured options when using modern typing; otherwise document keys strictly.
- **Returns**: prefer **single typed return**; multiple values as **named tuple**, **dataclass**, or **TypedDict** (see **python-refactor** agent for tuple smell).

## Errors

- Raise **specific exceptions**; avoid bare `except:`.
- Let exceptions carry **actionable messages**; use exception chaining (`raise … from e`) when re-raising.
- For expected failure modes, document **what callers should catch**.

## Resources

- **`with`** / context managers for files, locks, devices; **pyvisa** resource separation per the Python spec and rule.

## Idioms

- **`pathlib.Path`** for filesystem paths when the codebase already does; stay consistent within a module.
- **`if __name__ == "__main__":`** for CLI entrypoints; prefer **`uv run`** for execution in projects.

## Types at the boundary

- Annotate **public** functions completely; internals can lean on inference where **ty** still passes. Details: [reference-types-and-agents.md](reference-types-and-agents.md).

## Delegation

- **Structural** critique (too large, wrong abstraction): **`python-refactor`** agent.

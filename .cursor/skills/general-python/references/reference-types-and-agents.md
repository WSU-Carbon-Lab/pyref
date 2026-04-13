# Types and Astral ty

This file is the **bridge** between day-to-day edits and the **`python-types`** agent. Do not duplicate the full typing playbook here.

## Project rule

- **`ty check`** is the type gate; configuration lives in **`pyproject.toml`**.
- Prefer fixing types over **`# type: ignore`**; if unavoidable, **one line**, narrow code, and state **why**.

## Boundaries

- **Public** functions and methods: annotate **parameters and return**.
- **Locals**: rely on inference when obvious (`x = []` may need `list[T]` or construction that fixes `T`).
- Prefer **`collections.abc`** (`Sequence`, `Mapping`, `Iterable`) for inputs you only read.

## Python 3.12+ (when helpful)

- **PEP 695** `type` aliases and `def f[T](...)`.
- **PEP 692** `Unpack[TypedDict]` for kwargs.
- **PEP 698** `@override` on overrides.
- **PEP 688** `Buffer` over deprecated `ByteString` patterns.

Reference: [What is new in Python 3.12](https://docs.python.org/3/whatsnew/3.12.html).

## Exhaustive discrimination

- Use **`match` / `case`** so **ty** can verify coverage; **`assert_never`** for impossible tails after exhaustive matches. See **`python-types`** and [match exhaustiveness](https://dogweather.dev/2022/10/03/i-discovered-that-python-now-can-do-true-match-exhaustiveness-checking/).

## When to invoke **python-types**

- Large refactors to protocols/generics, fixing many **ty** errors, or aligning modules with **PEP 695** and **`match`** exhaustiveness.

## Ruff and types

- **Ruff** may flag typing imports (`TCH`, `UP`); keep **runtime** vs **TYPE_CHECKING** imports consistent with project rules.

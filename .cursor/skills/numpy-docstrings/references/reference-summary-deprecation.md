# Short summary, extended summary, deprecation

Source: [numpydoc — Short summary](https://numpydoc.readthedocs.io/en/latest/format.html#short-summary), [Extended summary](https://numpydoc.readthedocs.io/en/latest/format.html#extended-summary), [Deprecation](https://numpydoc.readthedocs.io/en/latest/format.html#deprecation-warning).

## Short summary

- **One line** (no blank line before extended text if you continue in the same opening paragraph, or blank line then extended summary—follow project style; numpydoc allows extended summary after a blank line).
- Do **not** use **variable names** from the signature in the short summary.
- Prefer avoiding repeating the **function name** as the opening phrase when it reads redundant with `help()`; state **what** it does in plain language.

Good pattern: `"""Compute the discrete Fourier transform along the specified axis.`

## C signature line (rare)

- For objects **without** introspectable signatures, put the signature as the **first** line inside the docstring, then a blank line, then the short summary (see numpydoc examples).

## Deprecation

- Use the Sphinx **`.. deprecated:: version`** **directive**, not a hyphen underlined section, when an API is deprecated.
- Include: **version** deprecated, **removal** timeline if known, **reason** when useful, and **replacement** API.

```text
.. deprecated:: 1.6.0
    `old_func` is replaced by `new_func` because ...
```

## Extended summary

- A few sentences **after** the short summary (separated by a blank line when it is its own block).
- Use for **what** the object does at a higher level; **not** for implementation trivia or long theory—those belong in **Notes** or **References**.
- You **may** mention parameters or the function name here, but **parameter definitions** still belong only under **Parameters**.

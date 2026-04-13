# Raises, Warns, Warnings

Source: [Raises](https://numpydoc.readthedocs.io/en/latest/format.html#raises), [Warns](https://numpydoc.readthedocs.io/en/latest/format.html#warns), [Warnings](https://numpydoc.readthedocs.io/en/latest/format.html#warnings).

## Raises

- List **exception types** and **when** they are raised.
- Use **judiciously**: obvious `ValueError` on bad input may be omitted if universal; **non-obvious** or **high-probability** errors deserve a line.

```text
Raises
------
LinAlgError
    If the matrix is singular.
```

## Warns

- Same shape as **Raises** but for **`Warning`** subclasses or user-visible warnings issued under stated conditions.

## Warnings (free-text cautions)

- **User-facing caveats** not tied to a single exception (numerical instability, thread safety, heuristic behavior).
- Free-form reST paragraph(s) under the **Warnings** heading—not the same as Python’s `warnings.warn`.

## Relation to typing

- **`ty` / PEP 484** do not replace **Raises**; document **semantic** error contracts for readers and Sphinx.

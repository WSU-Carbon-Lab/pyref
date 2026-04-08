# Examples

Source: [numpydoc — Examples](https://numpydoc.readthedocs.io/en/latest/format.html#examples).

## Role

- **Illustrate usage**; they are **not** the primary test suite—keep **`tests/`** as the authority for CI.
- Still **strongly encouraged** for user-facing APIs.

## Doctest format

- Use **`>>>`** prompts and expected output as in the Python REPL.
- **Separate** multiple examples with **blank lines**.
- Put **blank lines above and below** comment lines that explain an example.

## Continuations

- Continuation lines after the first `>>>` start with **`...`**.

## Random or platform-dependent output

- Mark with **`#random`** (or project convention) so doctest runners can skip strict comparison.

## Imports in examples

- NumPy docs assume **`import numpy as np`** is pre-run for numpy examples; still be **explicit** for anything else (`matplotlib.pyplot`, local modules).
- **Explicitly import** the function under documentation if it aids copy-paste.

## Empty lines in output

- Empty output lines in doctests do not need special markup per numpydoc.

## Matplotlib

- If **`matplotlib`** is imported in the example, Sphinx may use matplotlib’s **plot** directive when configured; otherwise `.. plot::` can be used in `.rst` sources—not always inside docstrings.

## Running doctests

- Projects may run **`pytest --doctest-modules`** or library-specific test hooks; align with the project **Python** spec (**pytest**).

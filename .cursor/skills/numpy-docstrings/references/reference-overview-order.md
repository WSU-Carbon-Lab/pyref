# Overview: format, order, and markup

Source: [numpydoc format](https://numpydoc.readthedocs.io/en/latest/format.html).

## Docstring wrapper

- Use **triple double quotes** `"""` for module, class, function, and method docstrings.
- First line after the opening `"""` is often the **short summary** (or signature line for C extensions without introspection).

## Line length

- Keep lines to about **75 characters** so docstrings read in plain terminals.

## Section headings

- Each section title is a **line of text** followed by a line of **hyphens** the **same length** as the title:

```text
Parameters
----------
```

- **Order** for functions (omit unused sections): Short summary → deprecation (directive) → Extended summary → Parameters → Returns → Yields → Receives → Other Parameters → Raises → Warns → Warnings → See Also → Notes → References → Examples.

## reStructuredText subset

- Docstrings use **reST** for Sphinx; prefer a **small** subset: inline `` `param` ``, `` :math:`x` ``, `.. deprecated::`, `.. math::`, `.. note::` sparingly.
- **Human readability** beats contorting text for HTML output ([numpydoc principle](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard)).

## Import names in narrative docs

- NumPy docs convention: `import numpy as np`, `import matplotlib as mpl`, `import matplotlib.pyplot as plt`; **do not** abbreviate `scipy` as something nonstandard in prose.

## What not to duplicate

- **Signature** is usually shown by `help()`; the short summary should **not** repeat the function name as a title unless documenting a C API without a visible signature.

## `array_like`

- Use **`array_like`** when an argument accepts **ndarrays** and values **coercible** to arrays (scalars, nested sequences).

## reST `.. note::` and `.. warning::`

- Use **sparingly** in sections; they render poorly in plain terminals ([numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html#other-points-to-keep-in-mind)).

## Monospace in prose

- **Parameter names**: single backticks (numpydoc convention).
- **Other code snippets** in running text: often **double backticks** in reST; follow project Sphinx/numpydoc version guidance for link vs monospace.

## Hyperlinks in docstrings

- Some sections parse **non-standard** reST; fragile `.. target` lines can confuse numpydoc—prefer **inline** links where Sphinx warns ([numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html#other-points-to-keep-in-mind)).

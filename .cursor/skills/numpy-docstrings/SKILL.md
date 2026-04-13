---
author: dotagents
name: numpy-docstrings
description: Write and review NumPy-style (numpydoc) docstrings for Python public APIs. Covers section order, semantics (what belongs in docstrings vs types vs tests), anti-patterns (comment soup, stub docs, wrong style), Parameters, Returns, Yields, Raises, See Also, Notes, References, Examples, and class/module docs. Use when authoring or auditing docstrings. Triggers on docstring, numpydoc, Parameters, Returns, Examples, Sphinx, TODO, comment.
---

# NumPy-style docstrings (numpydoc)

## Quick start

1. **Public APIs only** per project **Python spec** and **Python rule**: document **modules, classes, functions, and methods** users import; private helpers stay minimal unless behavior is non-obvious.
2. **Follow section order** from [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html#sections); each heading is **underlined with hyphens** the same length as the title.
3. **Line length**: aim for **~75 characters** in docstring prose for terminal readability.
4. **Markup**: subset of **reST**; parameter names in **single backticks** in running text.
5. **Load the chunk** you are editing from the table below—each reference file targets one part of the standard.
6. **Semantics**: choose **docstring vs annotation vs test vs prose** deliberately; avoid **comment narration** and **stub** docstrings—see [reference-semantics-and-anti-patterns.md](references/reference-semantics-and-anti-patterns.md).

## Stack synergy

| Resource | Role |
|----------|------|
| **general-python** | pytest, “NumPy style on public APIs” |
| **python-reviewer** | Checks docstring quality alongside typing and tests |
| **python-types** | Types in signatures complement docstring types |

## Section reference index

| Docstring chunk | File |
|-----------------|------|
| Global rules, section order, line length, reST | [reference-overview-order.md](references/reference-overview-order.md) |
| Short summary, extended summary, deprecation | [reference-summary-deprecation.md](references/reference-summary-deprecation.md) |
| Parameters, Other Parameters, `*args` / `**kwargs` | [reference-parameters.md](references/reference-parameters.md) |
| Returns, Yields, Receives | [reference-returns-yields-receives.md](references/reference-returns-yields-receives.md) |
| Raises, Warns, Warnings | [reference-raises-warns-warnings.md](references/reference-raises-warns-warnings.md) |
| See Also, Notes, References | [reference-see-also-notes-references.md](references/reference-see-also-notes-references.md) |
| Examples (doctest style) | [reference-examples.md](references/reference-examples.md) |
| Classes, Attributes, Methods, modules | [reference-classes-modules.md](references/reference-classes-modules.md) |
| Why each mechanism; bad comments/docstrings; fixes | [reference-semantics-and-anti-patterns.md](references/reference-semantics-and-anti-patterns.md) |

## Canonical specification

- [numpydoc style guide](https://numpydoc.readthedocs.io/en/latest/format.html)
- [Example docstring](https://numpydoc.readthedocs.io/en/latest/example.html)
- [PEP 257](https://peps.python.org/pep-0257/) (baseline docstring conventions)

## Optional tooling

- **`uv add numpydoc`** when Sphinx builds use the numpydoc extension; do not hand-edit pins in `pyproject.toml`.

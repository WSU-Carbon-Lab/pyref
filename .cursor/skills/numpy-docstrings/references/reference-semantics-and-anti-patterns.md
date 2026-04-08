# Semantics: which documentation mechanism, and anti-patterns

This file explains **where** information should live (docstring section vs type hint vs test vs narrative) and shows **bad habits** next to **spec-aligned** fixes. It aligns with the project **Python** spec and **Python rule**: NumPy-style docstrings on **public** APIs, **obvious control flow** without step-by-step **inline comments** in implementations.

## Choosing a mechanism

| What the reader needs | Prefer |
|------------------------|--------|
| **Callable contract** (what goes in, what comes out, what can fail) | **Parameters**, **Returns** / **Yields**, **Raises** in a numpydoc docstring |
| **Meaning that types cannot express** (units, ranges, physical interpretation, invariants) | Same sections’ prose, or **Notes** if it spans several parameters |
| **Pure type information** already clear from annotations | **`ty`-checked** signatures; **omit** redundant docstring types unless you add **semantics** |
| **How to call it** (minimal copy-paste) | **Examples** (short, doctest-style when appropriate) |
| **Where to go next** (alternatives, partners) | **See Also** |
| **Theory, algorithm, equations** | **Notes**; stable citations in **References** |
| **User-facing caution** (not one specific exception) | **Warnings** section |
| **Deprecation and replacement** | **`.. deprecated::`** directive ([summary/deprecation](reference-summary-deprecation.md)) |
| **Whole module map** (big packages) | **Module** docstring + **routine listings** ([classes/modules](reference-classes-modules.md)) |
| **Behavior guaranteed by tests** | **`tests/`** + pytest; docstring **Examples** illustrate, they do not replace CI |
| **Private helper** with a non-obvious invariant | One-line docstring **or** a **name** that carries the invariant; not a public numpydoc essay |

**Principle:** put **contract and usage** where **`help(obj)`** and Sphinx readers look first; keep **implementation noise** out of public docstrings and out of **comment trails** per project Python style.

---

## Anti-pattern: comments as a substitute for names and structure

**Bad:** narrating the code line-by-line or stating the obvious.

```python
def process(a):
    # loop over each row
    for i in range(a.shape[0]):
        # get the value
        v = a[i, 0]
        # add one
        a[i, 0] = v + 1
    return a
```

**Better:** **vectorize** when the stack allows; if a loop remains, **name** intent and move **contract** to the docstring, not inline chatter.

```python
import numpy as np

def increment_first_column(a: np.ndarray) -> np.ndarray:
    """Increment every value in the first column by one.

    Parameters
    ----------
    a : numpy.ndarray, shape (M, N)
        Input array; not modified in place.

    Returns
    -------
    numpy.ndarray
        A copy of ``a`` with column 0 incremented by one.
    """
    out = np.array(a, copy=True)
    out[:, 0] = out[:, 0] + 1
    return out
```

---

## Anti-pattern: docstring repeats the signature with no added meaning

**Bad:**

```python
def mean(a, axis=None):
    """mean(a, axis=None)

    Takes a and axis and returns the mean.
    """
```

**Better:** short summary **without** mirroring parameter names as the only content; put types and defaults in **Parameters** / **Returns**.

```python
import numpy as np

def mean(a: np.ndarray, axis: int | None = None) -> np.floating | np.ndarray:
    """Arithmetic mean along the given axis.

    Parameters
    ----------
    a : array_like
        Input data.
    axis : int, optional
        Axis along which to compute the mean. Default is None (flatten).

    Returns
    -------
    numpy.floating or ndarray
        Mean of ``a`` over the given axis.
    """
```

---

## Anti-pattern: parameter descriptions only in the extended summary

**Bad:** long paragraph that lists args but no **Parameters** section (breaks numpydoc parsing and `help()` structure).

**Better:** keep extended summary for **intent**; every public argument gets a **Parameters** line (or **Other Parameters** if rarely used). See [reference-parameters.md](reference-parameters.md).

---

## Anti-pattern: empty or stub sections

**Bad:**

```text
Returns
-------

Raises
------
```

**Better:** omit entire sections that add nothing; numpydoc expects **content** under each heading you include.

---

## Anti-pattern: “TODO” and placeholders in shipped APIs

**Bad:** `"""TODO: document."""` on a public function.

**Better:** ship with at least a **short summary** and **Parameters**/**Returns** for anything exported in `__all__` or documented as stable; track internal debt in issues, not user-facing docstrings.

---

## Anti-pattern: author, license, or change log in the module docstring

**Bad:** module docstring that is only copyright or “written by Alice 2021.”

**Better:** numpydoc reserves the module docstring for **summary**, optional listings, **See Also**, **Examples**; put license in **`LICENSE`** and attribution in **VCS** or **`pyproject.toml`** ([classes/modules](reference-classes-modules.md)).

---

## Anti-pattern: duplicating type hints with zero extra semantics

**Bad:**

```python
def f(x: int, y: float) -> str:
    """Do something.

    Parameters
    ----------
    x : int
        An integer.
    y : float
        A float.

    Returns
    -------
    str
        A string.
    """
```

**Better:** keep **types** in annotations; use docstring lines for **meaning** (role, units, constraints, edge cases).

```python
def f(x: int, y: float) -> str:
    """Format a channel index and gain for display.

    Parameters
    ----------
    x : int
        Zero-based acquisition channel index.
    y : float
        Linear gain in decibels applied before quantization.

    Returns
    -------
    str
        Human-readable label, e.g. ``"ch3 (+6.0 dB)"``.
    """
```

---

## Anti-pattern: wrong style for the project

**Bad:** Google-style `Args:` / `Returns:` blocks in a repo that standardizes on **NumPy** / numpydoc and Sphinx.

**Better:** match **this stack**: hyphen underlines and section names from [numpydoc format](https://numpydoc.readthedocs.io/en/latest/format.html#sections).

---

## Anti-pattern: Examples that are really integration tests

**Bad:** dozens of lines of setup, mocks, and assertions inside **Examples** that belong in **`tests/`**.

**Better:** **Examples** show **minimal** usage; heavy behavior is **pytest** with stable fixtures ([reference-examples.md](reference-examples.md)).

---

## Anti-pattern: documenting `self` on methods

**Bad:** listing `self` under **Parameters** on instance methods.

**Better:** omit **`self`** / **`cls`** per numpydoc ([reference-classes-modules.md](reference-classes-modules.md)).

---

## Anti-pattern: critical exceptions only in comments

**Bad:**

```python
def invert(m):
    # raises LinAlgError if singular
    return np.linalg.inv(m)
```

**Better:** **Raises** in the docstring so `help()` and HTML docs show the contract.

```text
Raises
------
numpy.linalg.LinAlgError
    If ``m`` is singular.
```

---

## Quick checklist

- Public API: **short summary** plus the **sections** that carry real information—**no** filler headings.
- Implementation: **readable structure** and **names**, not comment narration (**Python** rule).
- Types: **annotations + ty** for static truth; docstring for **semantics** types miss.
- Red flags: **TODO** docstrings, **license-only** module docs, **Args/Returns** markdown in a numpydoc project, **comment soup** instead of refactors.

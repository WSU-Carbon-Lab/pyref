# Classes, Attributes, Methods, modules

Source: [Documenting classes](https://numpydoc.readthedocs.io/en/latest/format.html#documenting-classes), [Documenting modules](https://numpydoc.readthedocs.io/en/latest/format.html#documenting-modules) (if present in full page—module section exists in numpydoc).

## Class docstring

- Use the **same sections** as functions **except** **Returns** (unless a special method behaves like a function with a documented return—rare in class docstring body).
- **`__init__` parameters** belong in the class docstring **Parameters** section (constructor arguments).
- Optional separate **`__init__` docstring** for extra initialization detail if the project allows duplication control.

## Attributes

- Section **below Parameters** listing **non-method** attributes:

```text
Attributes
----------
x : float
    The current x coordinate.
```

- Properties with their own docstrings may be listed **by name only** in **Attributes** when numpydoc defers to the property.

## Methods

- Each **public** method is documented like a **function**.
- **Do not** list **`self`** (or **`cls`**) in **Parameters**.
- If a method mirrors a **standalone function**, put the **detailed** narrative on the **function** docstring; the method may carry a **short summary** plus **See Also** pointing to the function ([numpydoc — method docstrings](https://numpydoc.readthedocs.io/en/latest/format.html#method-docstrings)).

## Methods section (on the class)

- Optional **Methods** block on the **class** docstring when only a **few** methods matter (e.g. large subclasses); list signatures in one line each with a short description—**not** for private methods.

## `property`

- Document **getter behavior**, **raises**, and **side effects** if any; type in **Returns** or in the one-line summary as appropriate.

## Module docstring

- At least a **summary line**; optional sections in **function order** where appropriate: extended summary, **routine listings** for large modules, **See Also**, **Notes**, **References**, **Examples** ([documenting modules](https://numpydoc.readthedocs.io/en/latest/format.html#documenting-modules)).
- **License and author** metadata belong outside the docstring (comments or project files), not in the module docstring per numpydoc.

## Constants

- Use summary, optional extended summary, **See Also**, **References**, **Examples** as needed ([documenting constants](https://numpydoc.readthedocs.io/en/latest/format.html#documenting-constants)); some immutable constants cannot carry `__doc__` in the REPL.

## Private objects

- Leading underscore names: minimal or no numpydoc sections unless they are part of a **subclassing contract**.

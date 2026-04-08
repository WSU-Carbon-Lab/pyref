# See Also, Notes, References

Source: [See Also](https://numpydoc.readthedocs.io/en/latest/format.html#see-also), [Notes](https://numpydoc.readthedocs.io/en/latest/format.html#notes), [References](https://numpydoc.readthedocs.io/en/latest/format.html#references).

## See Also

- Point to **related** APIs the reader might miss; avoid listing the whole module.
- Form: `name : Short description.` or **`name` only** if the name is self-explanatory.
- Same submodule: **unqualified** name; other submodules: **`submod.func`**; other packages: **`package.module.func`**.
- Long descriptions: put **name :** on first line, description **indented** four spaces on the next.

## Notes

- **Algorithms**, **complexity**, **numerical details**, **background theory** that would clutter **Extended summary**.
- **Equations**: `.. math::` block or inline `` :math:`\alpha` ``; keep LaTeX **readable**—equations are hard in plain text.
- **Images**: `.. image:: path` only if the docstring still makes sense **without** the image (numpydoc guidance).
- Variable emphasis in math: `\mathtt{var}` when needed for typographic distinction.

## References

- **Numbered** citations supporting **Notes**; use `.. [1] Author, "Title", ...` and cite with `[1]_` in Notes.
- Prefer **stable** sources; **avoid** fragile URLs as the only citation.
- **Caveat** ([numpydoc #130](https://github.com/numpy/numpydoc/issues/130)): citation markers like `[1]` inside **tables** in docstrings can break numpydoc processing—avoid that combination or restructure.

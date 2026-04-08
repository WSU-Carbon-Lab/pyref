# Parameters and Other Parameters

Source: [numpydoc — Parameters](https://numpydoc.readthedocs.io/en/latest/format.html#parameters), [Other Parameters](https://numpydoc.readthedocs.io/en/latest/format.html#other-parameters).

## Heading

```text
Parameters
----------
```

## Each parameter

- Form: **`name : type`** then newline, then **indented** description (4 spaces).
- numpydoc: the colon may be **omitted if there is no type**; otherwise use **`name : type`** (space before the colon after the name, space after the colon before the type).
- Type may be **omitted**: then `name` alone with description on the next line.
- Refer to a parameter in prose with **single backticks**: `` `x` ``.

## Types (be specific)

Examples from numpydoc: `str`, `bool`, `array_like`, `int or tuple of int`, `list of str`, `dtype`, `callable`, union with **or**.

## Optional and defaults

- Keyword-only optionals: **`optional`** in the type line: `x : int, optional`
- Or give **default** in type: `copy : bool, default True` (also `default=True`, `default: True`—pick one style per project).

## Fixed set of values

- `order : {'C', 'F', 'A'}` with default listed first in braces when applicable.

## Combining parameters

- Same type and meaning: `x1, x2 : array_like` then one description referencing `` `x1` `` and `` `x2` ``.

## `*args` and `**kwargs`

- Keep the **stars** in the name; **do not** give a type on the `*args` / `**kwargs` line in numpydoc style.
- Describe how forwarded kwargs map to underlying functions if relevant.

## Other Parameters

- Use **only** when **many** keyword parameters would clutter **Parameters**; move **rare** or **advanced** kwargs here.
- Same formatting rules as **Parameters**.

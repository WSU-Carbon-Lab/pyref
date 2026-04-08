# Returns, Yields, Receives

Source: [Returns](https://numpydoc.readthedocs.io/en/latest/format.html#returns), [Yields](https://numpydoc.readthedocs.io/en/latest/format.html#yields), [Receives](https://numpydoc.readthedocs.io/en/latest/format.html#receives).

## Returns

- For each return value: **type is required**; **name is optional**.
- Anonymous single return:

```text
Returns
-------
int
    Description of return value.
```

- Named returns mirror **Parameters**:

```text
Returns
-------
err_code : int
    Non-zero indicates error.
err_msg : str or None
    Message or None on success.
```

## Yields

- For **generators** only; same structure as **Returns** (type required, name optional).
- numpydoc **0.6+** supports **Yields**.

## Receives

- Documents what a generator receives via **`.send()`**; format like **Parameters**.
- If **Receives** appears, **Yields** must also appear.

## Multiple return paths

- Document **all** stable contracts; if behavior is `None` or sentinel, state **when** each occurs in the description lines.

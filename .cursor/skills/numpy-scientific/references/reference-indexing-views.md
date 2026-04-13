# Indexing, views, and copies

Official guides: [Indexing](https://numpy.org/doc/stable/user/basics.indexing.html), [Copies and views](https://numpy.org/doc/stable/user/basics.copies.html).

## Basic indexing (views)

- **Slices** `start:stop:step`, **ellipsis** `...`, and **integer** indexing that **reduces** dimensionality yield **views** when the memory layout allows.
- **Mutating** a slice can change the **base** array—this is the main footgun.

## Advanced indexing (copies)

- **Integer arrays**, **boolean masks** (unless `numpy.ma`), and combinations that cannot be expressed as a strided window generally produce a **copy**.
- After `b = a[mask]`, changing **`b`** does not change **`a`**.

## `reshape`, `ravel`, `transpose`, `swapaxes`

- **`reshape`**: returns **view** when strides allow, else **copy**; use **`arr.reshape(-1)`** for flattening with explicit intent.
- **`ravel`**: often a view; **`flatten`** always a copy.
- **`transpose` / `.T`**: usually a view.

## `copy` and `base`

- **`arr.copy()`** for an independent buffer.
- **`arr.base`**: if not `None`, data is owned elsewhere; trace mutations through **views**.

## Boolean indexing vs `numpy.ma`

- Plain **boolean** indexing on `ndarray` **copies** selected elements.
- **Masked arrays** (`numpy.ma`) carry a mask alongside data; use when NaNs are insufficient or you need mask algebra.

## Assignment

- **`a[i:j] = x`** writes through to base; shapes must broadcast.
- **Chained** fancy indexing assignment has subtleties; prefer **single** assignment with **`np.put`** or explicit loop only when necessary.

## Reading suggestions

- When unsure, **`np.shares_memory(a, b)`** answers whether two arrays might alias.

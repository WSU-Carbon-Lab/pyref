# Broadcasting and array shape

Official: [Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html).

## Rule

From trailing dimensions forward, pairs are compatible if **equal**, or **one is 1**, or **one is missing** (implicit 1). The result shape is the **maximum** of each aligned size.

## New axes

- **`arr[:, np.newaxis]`** or **`arr[:, None]`** inserts a length-1 axis for alignment.
- **`np.expand_dims`** for clarity in library code.

## Explicit broadcast

- **`np.broadcast_to`** materializes a **read-only** strided view when possible; good for debugging shapes.
- **`np.broadcast_arrays`** aligns several inputs to a common shape.

## Stacking and splitting

- **`np.stack`** (new axis), **`np.concatenate`** (existing axis), **`np.vstack` / `np.hstack` / `np.dstack`** (convenience; know what they do to 1D).
- **`np.split`**, **`array_split`** for uneven chunks.

## Grids

- **`np.meshgrid`** (`indexing="xy"` vs `"ij"`): know which axis varies **fast**; mistakes swap rows/columns in images.
- **`np.ogrid`** / **`mgrid`**: open grids that **broadcast** without huge temporaries.

## Tile and repeat

- **`np.tile`** repeats blocks; **`np.repeat`** repeats elements; different memory patterns.

## Gotchas

- **Scalar + array** broadcasts naturally; **list of arrays** does not become one ndarray without **`stack`**/**`array`**.
- **In-place** `+=` with broadcast RHS can create **temporary** behavior surprises; verify with small examples when optimizing.

# Random, file I/O, structured dtypes

## Random (`numpy.random`)

- Use **`np.random.default_rng(seed)`** → **`Generator`** with **`PCG64`** (or other bit generators per docs).
- Avoid legacy **`np.random.seed`** / **`RandomState`** in new library code; they complicate **parallel** and **library composition**.

Official: [Random sampling](https://numpy.org/doc/stable/reference/random/index.html).

## Common patterns

```python
rng = np.random.default_rng(42)
rng.normal(size=(1000, 2))
rng.integers(low=0, high=10, size=5, endpoint=False)
rng.permutation(x)
```

- **`SeedSequence`** for **spawned** streams in parallel jobs.

## Saving and loading

- **`.npy`**: single array, preserves shape and dtype.
- **`.npz`**: archive of **named** arrays (`np.savez`, **`np.savez_compressed`**).
- **`np.load`**: `allow_pickle=False` default in modern NumPy—**do not** unpickle untrusted files.

Official: [NumPy binary formats](https://numpy.org/doc/stable/reference/routines.io.html).

## Text and CSV

- **`np.loadtxt`**, **`genfromtxt`** for simple cases; **pandas** / **polars** often scale better for messy tables—hand off at a clear **module or pipeline** boundary.

## Structured and record arrays

- **`dtype=[('x', 'f8'), ('id', 'i4')]`** for **columnar** heterogeneous data in one block.
- **Alignment** and **offsets** matter for **C interop**; use **`np.dtype`** descriptors carefully.
- For analytics-heavy workflows, consider **DataFrame** after one **`from_records`** step.

## Memory maps

- **`np.memmap`** for arrays larger than RAM; understand **flush** and **write** semantics.

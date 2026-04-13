# Linear algebra and `einsum`

## Matrix multiply

- **`@`** operator and **`np.matmul`**: **batched** last two dimensions for ndim > 2; not the same as **`np.dot`** for stacks (prefer **`@`** for clarity).
- **Elementwise** multiply: **`*`** or **`np.multiply`**.

Official: [numpy.linalg](https://numpy.org/doc/stable/reference/routines.linalg.html).

## `linalg`

- **`np.linalg.solve`**: linear systems **`Ax = b`** (prefer over **`inv(A) @ b`** numerically).
- **`lstsq`**, **`svd`**, **`eig`**, **`eigh`** (Hermitian), **`cholesky`**, **`norm`**, **`det`**, **`matrix_rank`**.
- **`cond`** and **`rcond`** for ill-conditioned systems.

## `einsum`

- **Einstein summation**: specify **subscripts** or **`->` output**; **sum** repeated indices.
- **`optimize=True`** fuses paths for large contractions (default in recent NumPy for many cases).
- Use for **tensor contractions**, **batched** matmuls, and **axis reductions** that are hard to read as nested **`sum`**.

Official: [numpy.einsum](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html).

## `tensordot`, `kron`, `outer`

- **`np.tensordot`**: contract chosen axes between two tensors.
- **`np.kron`**: Kronecker product; can explode **size** quickly.
- **`np.outer`**: rank-1 outer product.

## dtypes

- **`linalg`** generally expects **floating** or **complex floating**; integer arrays may be promoted or rejected—cast explicitly.

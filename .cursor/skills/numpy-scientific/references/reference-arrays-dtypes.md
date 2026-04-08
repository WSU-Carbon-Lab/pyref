# Array creation, dtypes, casting

## ndarray mental model

- **`shape`**: tuple of lengths per axis; **`ndim`**; **`size`** total elements.
- **`dtype`**: element type (size and interpretation); **`itemsize`** bytes per element.
- **`order`**: **`C`** row-major (default) vs **`F`** column-major; matters for **reshape**, **flat** iteration, and **FFI** with other libraries.

Official: [The N-dimensional array](https://numpy.org/doc/stable/reference/arrays.ndarray.html).

## Constructors

| Need | Typical API |
|------|-------------|
| Zeros / ones | `np.zeros`, `np.ones`, `dtype=...` |
| Uninitialized (then fill) | `np.empty` (fastest when you overwrite every element) |
| Constant fill | `np.full` |
| Sequences | `np.asarray` (no copy if already ndarray and compatible), `np.array` (new array) |
| Ranges | `np.arange` (half-open, prefer **integer** steps; watch **float** accumulation), `np.linspace`, `np.logspace`, `np.geomspace` |
| Identity / diag | `np.eye`, `np.diag` |

## Dtype choices

- **Integers**: `np.int32`, `np.int64`, `np.uint8` for images, etc.; match **file format** and **downstream** APIs.
- **Floats**: `float64` default; `float32` for memory/bandwidth when precision allows; **`float16`** mainly for storage or ML hooks, not general numerics.
- **Complex**: `complex128` default pair with `float64` real/imag.
- **Booleans**: `bool_`; avoid Python `bool` in dtype lists for homogeneous arrays.

## Casting and `astype`

- **`astype`** can **round** or **truncate**; floating to integer is not the same as `round()` policy-wise—verify for your domain.
- **`astype(copy=False)`** may still copy if layout or dtype forces it.
- Prefer **`np.can_cast`** and explicit **`dtype=`** on creation when ingesting untyped buffers.

## Strings and objects

- **Fixed-width strings**: `dtype="U10"`, `"S10"`; variable-length text often belongs in **pandas** or Python lists, not `object` arrays, unless you accept **`object`** dtype costs.

## Alignment and record dtypes

- **Structured / record** dtypes: field offsets and alignment; see [reference-random-io-structured.md](reference-random-io-structured.md).

## Verification

- **`array.shape`**, **`array.dtype`**, **`array.flags`** (`writeable`, `owndata`) when debugging aliases.

# Builtins and collections

## Choosing a structure

| Need | Prefer |
|------|--------|
| Ordered sequence, homogeneous | **`list`**; fixed small arity: **`tuple`** |
| Key-value lookup, unique keys | **`dict`** (3.7+ insertion order is part of language spec) |
| Membership testing, uniqueness | **`set`** / **`frozenset`** when immutable |
| Immutable record of fields | **`tuple`**, **`NamedTuple`**, or **`@dataclass(frozen=True)`** (see [reference-classes-dataclasses.md](reference-classes-dataclasses.md)) |
| FIFO / LRU-ish queue | **`collections.deque`** |
| Counting | **`collections.Counter`** |
| Grouping by key | **`collections.defaultdict(list)`** (or similar) |
| Read-only mapping view | **`Mapping`** from **`collections.abc`** in type hints |

## Comprehensions and literals

- Prefer **comprehensions** or **generator expressions** over `map`/`filter` when readability wins.
- Avoid **mutable default arguments**; use `None` and assign inside the function or use **`dataclasses.field(default_factory=…)`**.

## Immutability and copying

- **`tuple`**, **`frozenset`**, **frozen dataclass**: safe as dict keys and for defensive sharing.
- **Shallow vs deep copy**: default assignment and `list(old)` share nested mutables; use **`copy.deepcopy`** only when the data model requires it.

## Sorting and keys

- **`sorted(iterable, key=…)`** and **`list.sort(key=…)`** for stable, explicit ordering.
- **`bisect`** on sorted sequences for insertion and search (stdlib).

## Typing shapes

- **`list[int]`**, **`dict[str, float]`**, **`set[str]`** in annotations; read-only parameters as **`Sequence[T]`**, **`Mapping[str, T]`** from **`collections.abc`** when you do not need mutation.

## When not to use a dict

- Fixed, named fields with validation: consider **dataclass** or **Pydantic** (if project already depends on it). Do not add Pydantic via hand-edited pins; use **`uv add`**.

## Scientific stacks

- **NumPy arrays** for numeric tensors; **pandas** / **polars** for labeled tables per the Python spec; do not emulate DataFrames with nested dicts unless prototyping.
- Tabular workflows: **dataframes** skill.

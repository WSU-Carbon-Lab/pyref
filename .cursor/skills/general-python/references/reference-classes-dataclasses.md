# Classes and dataclasses

## When to use a class

- **Stateful service** with invariants (device client, session, parser with buffer).
- **True is-a** hierarchy with shared behavior; otherwise prefer **composition** and **`Protocol`** (see **`python-refactor`**).

## Dataclasses (`dataclasses`)

- Use for **labeled data bundles** with minimal behavior: configs, results, DTOs between layers.
- **`frozen=True`**: immutable value objects; safer hashing and sharing.
- **`slots=True`** (3.10+): lower memory and faster attribute access when you do not need a `__dict__`.
- **`order=True`**: only when ordering is meaningful and documented.
- **`field(default_factory=…)`** for mutable defaults (lists, dicts).
- Replace opaque **`tuple[int, int, int, int]`** returns with a **frozen dataclass** or **`NamedTuple`** when names carry meaning.

## Regular classes

- **`@staticmethod`**: rare; often a module function is clearer.
- **`@classmethod`**: alternate constructors (`from_path`, `from_config`).
- **`__init__`**: minimal; heavy work in dedicated methods or factories so tests stay small.

## Special methods

- Implement **`__repr__`** for debuggability; **`__str__`** when user-facing text differs.
- **`__eq__`**: dataclass generates it; hand-roll only with clear semantics.

## Dataclass vs TypedDict vs NamedTuple

- **`TypedDict`**: JSON-like dict shapes, especially for **`**kwargs`** typing.
- **`NamedTuple`**: immutable, tuple-unpacking ergonomics, light memory.
- **`dataclass`**: mutable or frozen records with optional methods.

## Typing and tooling

- After modeling data, run **`ty check`**; align with **`python-types`** for generics and protocols.

## Delegation

- **Inheritance vs composition**, god classes: **`python-refactor`** agent.

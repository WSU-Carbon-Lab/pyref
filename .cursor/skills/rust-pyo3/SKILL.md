---
author: dotagents
name: rust-pyo3
description: Applies dotagent conventions for Rust PyO3 and Maturin extension crates on top of the base Rust stack.
---

When this skill applies:

- Keep the Python surface small and stable; align naming between `pyproject.toml`, `Cargo.toml`, and the published module.
- Use Maturin as the build entry point unless the repository already standardizes on a different documented flow.
- Be explicit about GIL, `Send`, and exception mapping across the FFI boundary.
- Prefer thin Python modules that re-export a narrow Rust API over exposing many low-level handles.

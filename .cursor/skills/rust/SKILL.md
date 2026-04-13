---
author: dotagents
name: rust
description: Applies the dotagent Rust stack defaults when editing Rust crates and workspaces.
---

When this skill applies:

- Use cargo for builds, tests, and dependency changes; add crates with `cargo add` when introducing new dependencies.
- Respect the repository edition, MSRV, and any clippy or rustfmt configuration already checked in.
- Prefer explicit `Result` handling at API boundaries; keep `unwrap` out of library code unless the project documents an exception.
- Align module layout and visibility with existing crates in the workspace before introducing new patterns.

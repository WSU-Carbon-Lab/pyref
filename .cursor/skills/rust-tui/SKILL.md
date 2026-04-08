---
author: dotagents
name: rust-tui
description: Applies dotagent conventions for Rust terminal user interfaces on top of the base Rust stack.
---

When this skill applies:

- Follow the TUI stack already present in the repository; avoid introducing a second framework without a migration plan.
- Separate rendering from state: keep model updates independent of widget internals where the codebase already does.
- Treat resize, focus changes, and partial redraws as normal; avoid assuming a fixed terminal size.
- Prefer keyboard-first flows; document mouse or alternative input when the project exposes them.

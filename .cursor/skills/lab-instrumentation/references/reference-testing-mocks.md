# Testing instrument code without hardware

## Layers to test separately

- **Parsing**: response strings to **`float`**, **unit stripping**, **status bit** interpretation—**pure functions**, **table-driven** pytest.
- **Validation**: range checks and **enum** mapping—no sockets.
- **Transport**: thin adapter that **`write`/`read`**; swap for **fake** in tests.

## Fakes and protocols

- Define a **`Protocol`** with **`query`**, **`write`**, **`read_raw`** matching your **session** surface; implement **`FakeInstrument`** with **canned** responses and **counters** to assert **call order**.
- For **stateful** gear, keep **minimal** state in the fake (**output on**, **last voltage**) so tests encode **realistic** sequences.

## PyVISA simulation

- **`pyvisa-sim`** (when compatible with your stack) can serve **YAML-defined** instruments for **integration-style** tests without benches.
- Vendor **simulators** and **socket echo** servers are acceptable **fixtures** if CI can start them **deterministically**.

## Record and replay

- Capture **transcripts** (command, response, timing hints) from a **golden** session; replay in tests to detect **protocol drift**. **Sanitize** secrets and **unique** serial numbers if committing files.

## Markers and CI

- Mark **slow** or **hardware-required** tests (`@pytest.mark.hardware`); default **`pytest`** in CI runs **offline** only. Document **`pytest -m hardware`** for the lab machine.

## Socket tests

- Use **pytest fixtures** with **ephemeral ports** (`bind(("", 0))`) and **short-lived** server threads or **`asyncio`** test servers; always **join** threads to avoid **flaky** teardown.

Fix typo: **`pytest`**** fixtures** -> **pytest fixtures**
</think>


<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>
StrReplace
